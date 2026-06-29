"""Base relative permeability tables for multi-phase flow simulations."""

import threading
import typing

import attrs
import numba
import numpy as np
from scipy.interpolate import PchipInterpolator

from bores.constants import c
from bores.errors import ValidationError
from bores.precision import get_floating_point_info
from bores.serialization import make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.tables.rock_fluid.relperm.mixing_rules import (
    MixingRule,
    deserialize_mixing_rule,
    get_mixing_rule,
    get_mixing_rule_partial_derivatives,
    serialize_mixing_rule,
)
from bores.tables.rock_fluid.utils import build_pchip_interpolant
from bores.typing import (
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
    Spacing,
)
from bores.utils import array as bores_array
from bores.utils import is_scalar_like

__all__ = [
    "RelPermEndpoints",
    "ThreePhaseRelPermTable",
    "TwoPhaseRelPermTable",
    "get_relperm_table",
    "list_relperm_tables",
    "relperm_table",
]


def _show_invalid_saturation(
    val: NumberOrArray[NDimension], *, max_display: int = 20
) -> str:
    if is_scalar_like(val) and (val < 0 or val > 1):
        return str(val)

    invalid = val[(val < 0) | (val > 1)]  # type: ignore
    length = len(invalid)
    if length == 0:
        return "OK"
    if length == 1:
        return str(invalid[0])
    return (
        f"{invalid}"
        if length <= max_display
        else f"{invalid[:max_display]} ... (count={length})"
    )


#: Sentinel type for the `minimum_*_relperm` attributes on relperm tables and
#: analytical models. `"auto"` derives the min_value from the active dtype;
#: `None` disables the min/floor value entirely; a `Number` sets an explicit min_value.
MinimumRelPerm = typing.Union[typing.Literal["auto"], None, Number]


def _resolve_min_relperm(min_value: MinimumRelPerm) -> typing.Optional[Number]:
    """
    Resolve a `MinimumRelPerm` sentinel to a concrete Number or `None`.

    `"auto"` - `max(4 * machine_epsilon, 1e-8)`, consistent with the
    approach used by CMG IMEX/GEM for minimum mobility min_values: tight enough
    that it does not distort the physics, but sufficient to prevent
    near-singular transmissibility rows in the Jacobian.

    `None` - `None` (no min_value; kr can reach zero exactly).

    `Number` - that value, used as-is after a sign / sanity check.

    :param min_value: Sentinel value from a relperm table or analytical model.
    :return: Resolved Number min_value or `None`.
    """
    if min_value is None:
        return None

    if min_value == "auto":
        info = get_floating_point_info()
        return max(4.0 * info.eps, 1e-8)

    if not isinstance(min_value, (int, Number, np.floating, np.integer)):
        raise ValidationError(
            f"`min_*_relperm` must be 'auto', None, or a Number. Got {min_value!r}."
        )

    if min_value < 0.0:
        raise ValidationError(
            f"`min_*_relperm` min_value must be non-negative. Got {min_value}."
        )
    return min_value


@numba.njit(cache=True, inline="always")
def _clamp_relperm(
    kr: NumberOrArray[NDimension],
    min_value: typing.Optional[Number],
) -> NumberOrArray[NDimension]:
    """
    Clamp `kr` to `[min_value, ∞)` in-place-compatible fashion.

    Smoothly transitions from the raw `kr` values to the min_value value, avoiding
    non-differentiability and ensuring well-behaved derivatives for solvers.

    When `min_value` is `None` the array is returned unchanged.

    :param kr: Relative permeability value(s).
    :param min_value: Resolved min_value/minimum value or `None`.
    :return: Floored relative permeability value(s).
    """
    if min_value is None:
        return kr
    return np.sqrt(kr**2 + min_value**2).astype(kr.dtype)  # type: ignore[return-value]


@numba.njit(cache=True, inline="always")
def _clamp_relperm_derivative(
    dkr: NumberOrArray[NDimension],
    kr_raw: NumberOrArray[NDimension],
    min_value: typing.Optional[Number],
) -> NumberOrArray[NDimension]:
    """
    Smoothly clamp the derivative of `kr` to zero in the min_value region.
    When `kr` is above the min_value, the derivative is unchanged. As `kr` approaches
    and falls below the min_value, the derivative transitions smoothly to zero, ensuring
    that the Jacobian remains well-conditioned and solvers remain stable.

    When `min_value` is `None` the derivative is returned unchanged.

    The derivative of `max(kr, min_value)` w.r.t. any variable is:

    - `dkr / d(var)`  when `kr > min_value`  (active region)
    - `kr / np.sqrt(kr**2 + min_value**2)`  when `kr <= min_value` (min_value / flat region)

    :param dkr: Derivative of the raw (pre-min_value) kr w.r.t. some variable.
    :param kr_raw: Raw (pre-min_value) kr value(s), used to identify min_value cells.
    :param min_value: Resolved min_value/minimum value or `None`.
    :return: Derivative with min_value cells zeroed out.
    """
    if min_value is None:
        return dkr
    return np.where(  # type: ignore[return-value]
        kr_raw > min_value, dkr, kr_raw / np.sqrt(kr_raw**2 + min_value**2)
    ).astype(dkr.dtype)


@attrs.frozen
class RelPermEndpoints:
    """
    Relative permeability endpoint values for all three phases.

    All values are dimensionless and in [0, 1] for normalized tables.
    """

    oil: Number
    """
    Oil relative permeability at connate water saturation with no gas (krocw).
    Reference value for three-phase mixing rule normalization.
    """

    water: Number
    """
    Water relative permeability endpoint (krw at residual oil saturation, Sor).
    Used as the injector mobility min_value reference for water injectors.
    """

    gas: Number
    """
    Gas relative permeability endpoint (krg at connate water + residual oil).
    Used as the injector mobility min_value reference for gas injectors.
    """


class RelativePermeabilityTable(StoreSerializable):
    """
    Protocol for a relative permeability model/table that
    computes relative permeabilities based on fluid saturations.
    """

    __abstract_serializable__ = True

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        raise NotImplementedError

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        raise NotImplementedError

    def get_oil_relperm_endpoint(self) -> Number:
        """Oil relative permeability at connate water saturation (krocw). Defaults to 1.0."""
        return 1.0

    def get_water_relperm_endpoint(self) -> Number:
        """
        Water relative permeability endpoint (krw at residual oil saturation).

        Returns the maximum krw value from the two-phase oil-water table.
        Defaults to 1.0 for unit-normalized tables and analytical models.
        """
        return 1.0

    def get_gas_relperm_endpoint(self) -> Number:
        """
        Gas relative permeability endpoint (krg at connate water + residual oil).

        Returns the maximum krg value from the two-phase gas-oil table.
        Defaults to 1.0 for unit-normalized tables and analytical models.
        """
        return 1.0

    def get_relperm_endpoints(self) -> RelPermEndpoints:
        """
        Compute all three phase relative permeability endpoints in a single call.

        Returns a `RelPermEndpoints` container holding kro, krw, and krg endpoint
        values. For unit-normalized tables all three are 1.0. For unnormalized
        tabular data the values are read from the stored kr arrays.

        :return: `RelPermEndpoints` with oil, water, and gas endpoint values.
        """
        return RelPermEndpoints(
            oil=self.get_oil_relperm_endpoint(),
            water=self.get_water_relperm_endpoint(),
            gas=self.get_gas_relperm_endpoint(),
        )

    def get_relative_permeabilities(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param kwargs: Other key word arguments for computing the relative peremabilities.
        :return: `RelativePermeabilities` dictionary with keys "water", "oil", "gas".
        """
        raise NotImplementedError

    def get_relative_permeability_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute relative permeability derivatives for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param kwargs: Other key word arguments for computing the derivatives.
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        raise NotImplementedError

    def __call__(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param kwargs: Other key word arguments for computing the relative peremabilities.
        :return: `RelativePermeabilities` dictionary.
        """
        return self.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute relative permeability derivatives for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param kwargs: Other key word arguments for computing the derivatives.
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        return self.get_relative_permeability_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )


_RELPERM_TABLES: typing.Dict[str, typing.Type[RelativePermeabilityTable]] = {}
"""Registry of relative permeability table types."""

_relperm_tables_lock = threading.Lock()
relperm_table = make_serializable_type_registrar(
    base_cls=RelativePermeabilityTable,
    registry=_RELPERM_TABLES,
    key_attr="__type__",
    lock=_relperm_tables_lock,
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)


def list_relperm_tables() -> typing.List[str]:
    """
    List all registered relative permeability table types.

    :return: List of registered relative permeability table type names.
    """
    with _relperm_tables_lock:
        return list(_RELPERM_TABLES.keys())


def get_relperm_table(name: str) -> typing.Type[RelativePermeabilityTable]:
    """
    Get a registered relative permeability table type by name.

    :param name: Registered name of the relative permeability table type.
    :return: Relative permeability table class.
    :raises ValidationError: If the relative permeability table type is not registered.
    """
    with _relperm_tables_lock:
        if name not in _RELPERM_TABLES:
            raise ValidationError(
                f"Relative permeability table type '{name}' is not registered. "
                f"Use `@relperm_table` to register it. "
                f"Available types: {list(_RELPERM_TABLES.keys())}"
            )
        return _RELPERM_TABLES[name]


@relperm_table
@attrs.frozen
class TwoPhaseRelPermTable(
    RelativePermeabilityTable,
    load_exclude={
        "_wetting_interp",
        "_wetting_d_interp",
        "_non_wetting_interp",
        "_non_wetting_d_interp",
    },
    dump_exclude={
        "_wetting_interp",
        "_wetting_d_interp",
        "_non_wetting_interp",
        "_non_wetting_d_interp",
    },
):
    """
    Two-phase relative permeability lookup table backed by a PCHIP interpolant.

    Interpolates relative permeabilities for two fluid phases based on a
    reference saturation value. The reference saturation can be either the
    wetting or non-wetting phase saturation, depending on how the table was
    constructed (e.g. from lab data indexed by Sg vs So).

    Supports both scalar and array inputs up to 3D.

    Examples:

    - Oil-Water system (water-wet): reference is Sw (wetting phase),
      `reference_phase="wetting"`
    - Gas-Oil system indexed by So: reference is So (wetting phase),
      `reference_phase="wetting"`
    - Gas-Oil system indexed by Sg: reference is Sg (non-wetting phase),
      `reference_phase="non_wetting"`

    **Grid scaling** (`number_of_base_points` / `number_of_endpoint_extra_points`):

    When `number_of_base_points > 0` and the raw knot count is smaller than
    `number_of_base_points`, the table expands the knot grid to `number_of_base_points`
    base points (plus `number_of_endpoint_extra_points` extra knots in each boundary
    decade) before fitting the PCHIP interpolant. This gives smoother curves
    and better derivative accuracy near residual saturations without requiring
    the caller to pre-supply a dense grid.  Pass `number_of_base_points=0` to
    disable scaling entirely and use the raw knots directly.

    **Minimum relperm min_values** (`min_wetting_relperm` / `min_non_wetting_relperm`):

    When non-`None`, a min_value is applied to the interpolated kr value and
    the derivative is zeroed out wherever the raw (pre-min_value) kr is at or
    below the min_value. This gives a derivative that is exactly consistent with
    the min_value kr, preventing Jacobian-residual mismatches that cause MBE.

    `"auto"` derives the min_value from the active floating-point dtype:
    `max(4 * machine_epsilon, 1e-8)`. `None` disables the min_value (kr can
    reach zero exactly).
    """

    wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The wetting fluid phase, e.g. WATER (oil-water) or OIL (gas-oil)."""

    non_wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The non-wetting fluid phase, e.g. OIL (oil-water) or GAS (gas-oil)."""

    reference_saturation: NumberArray[OneDimension] = attrs.field(converter=bores_array)
    """
    Saturation values used as the x-axis for interpolation, monotonically
    increasing. May represent either the wetting or non-wetting phase
    saturation depending on `reference_phase`.
    """

    wetting_phase_relative_permeability: NumberArray[OneDimension] = attrs.field(
        converter=bores_array
    )
    """Relative permeability values for the wetting phase at each reference saturation."""

    non_wetting_phase_relative_permeability: NumberArray[OneDimension] = attrs.field(
        converter=bores_array
    )
    """Relative permeability values for the non-wetting phase at each reference saturation."""

    reference_phase: typing.Literal["wetting", "non_wetting"] = attrs.field(
        default="wetting"
    )
    """
    Which phase the `reference_saturation` axis represents.
 
    - `"wetting"` - `reference_saturation` holds wetting phase saturation
      values.  krw increases and krnw decreases as `reference_saturation`
      increases.
    - `"non_wetting"` - `reference_saturation` holds non-wetting phase
      saturation values.  krnw increases and krw decreases as
      `reference_saturation` increases.
 
    This does not change the interpolation mechanics. It only records which
    physical saturation the caller must supply when querying the table, so
    that `ThreePhaseRelPermTable` (and any other consumer) can dispatch the
    correct saturation grid without hard-coding assumptions.
    """

    min_wetting_relperm: MinimumRelPerm = None
    """
    Minimum min_value for the wetting-phase relative permeability.
 
    `"auto"` - `max(4 * machine_epsilon, 1e-8)` (dtype-aware).
    `None` - no min_value; kr can reach zero exactly.
    `Number` - explicit user-supplied min_value value.
 
    The min_value is applied to the interpolated kr value, and the derivative is
    zeroed out in the min_value region so that the Jacobian is consistent with
    the kr value (no MBE from mismatched kr/derivative pairs).
    """

    min_non_wetting_relperm: MinimumRelPerm = None
    """
    Minimum min_value for the non-wetting-phase relative permeability.
 
    Same semantics as `min_wetting_relperm`.
    """

    number_of_base_points: int = attrs.field(default=200)
    """
    Target number of base knot points used when expanding the raw saturation
    grid before fitting the PCHIP interpolant.
 
    When the number of raw knots is already >= `number_of_base_points`, no
    expansion is performed and the raw knots are used as-is.
    Pass `0` to disable grid scaling entirely.
    """

    number_of_endpoint_extra_points: int = attrs.field(default=20)
    """
    Number of extra knots injected into the first and last 10 % of the
    saturation range during grid expansion (see `number_of_base_points`).
 
    These additional knots improve derivative accuracy near residual
    saturations where kr curves vary most rapidly.
    Pass `0` to disable endpoint enrichment.
    """

    spacing: Spacing = attrs.field(default="cosine")
    """
    Grid spacing mode used when building the expanded knot grid.
 
    Typical values are `"cosine"` (default, clusters points near endpoints) and
    `"linspace"` (uniform).
    """

    _wetting_interp: PchipInterpolator = attrs.field(init=False, repr=False)
    _wetting_d_interp: PchipInterpolator = attrs.field(init=False, repr=False)
    _non_wetting_interp: PchipInterpolator = attrs.field(init=False, repr=False)
    _non_wetting_d_interp: PchipInterpolator = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.reference_phase not in ("wetting", "non_wetting"):
            raise ValidationError(
                f"`reference_phase` must be 'wetting' or 'non_wetting', "
                f"got {self.reference_phase!r}"
            )
        if len(self.reference_saturation) != len(
            self.wetting_phase_relative_permeability
        ):
            raise ValidationError(
                f"`reference_saturation` and wetting phase kr arrays must have same "
                f"length. Got {len(self.reference_saturation)} vs "
                f"{len(self.wetting_phase_relative_permeability)}"
            )
        if len(self.reference_saturation) != len(
            self.non_wetting_phase_relative_permeability
        ):
            raise ValidationError(
                f"`reference_saturation` and non-wetting phase kr arrays must have "
                f"same length. Got {len(self.reference_saturation)} vs "
                f"{len(self.non_wetting_phase_relative_permeability)}"
            )
        if len(self.reference_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation")
        if not np.all(np.diff(self.reference_saturation) >= 0):
            raise ValidationError(
                "`reference_saturation` must be monotonically increasing"
            )

        # Validate min_value sentinels eagerly so errors surface at construction time
        _resolve_min_relperm(self.min_wetting_relperm)
        _resolve_min_relperm(self.min_non_wetting_relperm)

        # Build interpolants
        wetting_pchip, wetting_dpchip = build_pchip_interpolant(
            reference_saturation=self.reference_saturation,
            values=self.wetting_phase_relative_permeability,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
        )
        non_wetting_pchip, non_wetting_dpchip = build_pchip_interpolant(
            reference_saturation=self.reference_saturation,
            values=self.non_wetting_phase_relative_permeability,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
        )
        object.__setattr__(self, "_wetting_interp", wetting_pchip)
        object.__setattr__(self, "_wetting_d_interp", wetting_dpchip)
        object.__setattr__(self, "_non_wetting_interp", non_wetting_pchip)
        object.__setattr__(self, "_non_wetting_d_interp", non_wetting_dpchip)

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return typing.cast(FluidPhase, self.wetting_phase)

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return typing.cast(FluidPhase, self.wetting_phase)

    def _resolve_reference(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        """
        Return whichever saturation array corresponds to the reference axis.

        :param wetting_saturation: Current wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Current non-wetting phase saturation (scalar or array).
        :return: The saturation to use as the interpolation x-value.
        """
        if self.reference_phase == "non_wetting":
            return non_wetting_saturation
        return wetting_saturation

    def _query_pchip(
        self,
        interpolant: PchipInterpolator,
        reference: NumberOrArray[NDimension],
        extrapolate_left: Number,
        extrapolate_right: Number,
    ) -> NumberOrArray[NDimension]:
        """
        Evaluate a pre-built PCHIP interpolant at `reference`, applying
        constant extrapolation at the boundaries.

        :param interpolant: Pre-built `PchipInterpolator` instance.
        :param reference: Query saturation value(s) - scalar or array.
        :param extrapolate_left: Constant returned for values below the knot range.
        :param extrapolate_right: Constant returned for values above the knot range.
        :return: Interpolated value(s) with the same shape as `reference`.
        """
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(np.asarray(reference, dtype=np.float64))
        x_min = interpolant.x[0]
        x_max = interpolant.x[-1]

        result = interpolant(np.clip(sat, x_min, x_max))
        result = np.where(sat < x_min, extrapolate_left, result)
        result = np.where(sat > x_max, extrapolate_right, result)

        if is_scalar:
            return result.item()
        return result.reshape(sat.shape)

    def _query_dpchip(
        self,
        d_interpolant: PchipInterpolator,
        reference: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        """
        Evaluate a pre-built PCHIP derivative interpolant at `reference`,
        returning zero outside the knot range (constant extrapolation = zero slope).

        :param d_interpolant: Pre-built derivative `PchipInterpolator`.
        :param reference: Query saturation value(s) - scalar or array.
        :return: Derivative value(s) with the same shape as `reference`.
        """
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(np.asarray(reference, dtype=np.float64))
        x_min = d_interpolant.x[0]
        x_max = d_interpolant.x[-1]

        result = d_interpolant(np.clip(sat, x_min, x_max))
        result = np.where((sat < x_min) | (sat > x_max), 0.0, result)

        if is_scalar:
            return result.item()
        return result.reshape(sat.shape)

    def get_wetting_phase_relative_permeability(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
    ) -> NumberOrArray[NDimension]:
        """
        Get wetting phase relative permeability.

        When `reference_phase="wetting"`, only `wetting_saturation` is
        needed. When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        The `min_wetting_relperm` min_value (if set) is applied to the result.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Relative permeability value(s).
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        kr = self._query_pchip(
            self._wetting_interp,
            ref,
            extrapolate_left=self.wetting_phase_relative_permeability[0],
            extrapolate_right=self.wetting_phase_relative_permeability[-1],
        )
        min_value = _resolve_min_relperm(self.min_wetting_relperm)
        return _clamp_relperm(kr, min_value)

    def get_non_wetting_phase_relative_permeability(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
    ) -> NumberOrArray[NDimension]:
        """
        Get non-wetting phase relative permeability.

        When `reference_phase="wetting"`, only `wetting_saturation` is
        needed. When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        The `min_non_wetting_relperm` min_value (if set) is applied to the result.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Relative permeability value(s).
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        kr = self._query_pchip(
            self._non_wetting_interp,
            ref,
            extrapolate_left=self.non_wetting_phase_relative_permeability[0],
            extrapolate_right=self.non_wetting_phase_relative_permeability[-1],
        )
        min_value = _resolve_min_relperm(self.min_non_wetting_relperm)
        return _clamp_relperm(kr, min_value)

    def get_two_phase_relative_permeabilities(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
    ) -> typing.Tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
        """
        Get both wetting and non-wetting phase relative permeabilities.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Tuple of `(wetting_kr, non_wetting_kr)`.
        """
        kr_wetting = self.get_wetting_phase_relative_permeability(
            wetting_saturation, non_wetting_saturation
        )
        kr_non_wetting = self.get_non_wetting_phase_relative_permeability(
            wetting_saturation, non_wetting_saturation
        )
        return kr_wetting, kr_non_wetting

    def get_wetting_phase_relative_permeability_derivative(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
    ) -> NumberOrArray[NDimension]:
        """
        Derivative of the wetting-phase relative permeability with respect to
        the reference saturation axis of this table, evaluated from the
        analytical PCHIP derivative.

        The derivative is zero outside the tabulated saturation range
        (constant extrapolation = zero slope).

        Where the `min_wetting_relperm` min_value is active (raw kr ≤ min_value),
        the derivative is zeroed out to be consistent with the min_value
        (constant) kr value, preventing Jacobian-residual mismatches.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Derivative value(s) with the same shape as the input.
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        dkr = self._query_dpchip(self._wetting_d_interp, ref)
        min_value = _resolve_min_relperm(self.min_wetting_relperm)
        if min_value is None:
            return dkr

        # Need raw kr to decide where min_value is active
        kr_raw = self._query_pchip(
            self._wetting_interp,
            ref,
            extrapolate_left=self.wetting_phase_relative_permeability[0],
            extrapolate_right=self.wetting_phase_relative_permeability[-1],
        )
        return _clamp_relperm_derivative(dkr, kr_raw, min_value)

    def get_non_wetting_phase_relative_permeability_derivative(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
    ) -> NumberOrArray[NDimension]:
        """
        Derivative of the non-wetting-phase relative permeability with respect
        to the reference saturation axis of this table, evaluated from the
        analytical PCHIP derivative.

        The derivative is zero outside the tabulated saturation range
        (constant extrapolation = zero slope).

        Where the `min_non_wetting_relperm` min_value is active (raw kr ≤
        min_value), the derivative is zeroed out to be consistent with the min_value
        (constant) kr value, preventing Jacobian-residual mismatches.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Derivative value(s) with the same shape as the input.
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        dkr = self._query_dpchip(self._non_wetting_d_interp, ref)
        min_value = _resolve_min_relperm(self.min_non_wetting_relperm)
        if min_value is None:
            return dkr

        kr_raw = self._query_pchip(
            self._non_wetting_interp,
            ref,
            extrapolate_left=self.non_wetting_phase_relative_permeability[0],
            extrapolate_right=self.non_wetting_phase_relative_permeability[-1],
        )
        return _clamp_relperm_derivative(dkr, kr_raw, min_value)

    def get_oil_relperm_endpoint(self) -> Number:
        if self.non_wetting_phase == FluidPhase.OIL:
            return np.max(self.non_wetting_phase_relative_permeability)
        elif self.wetting_phase == FluidPhase.OIL:
            return np.max(self.wetting_phase_relative_permeability)
        return 1.0

    def get_water_relperm_endpoint(self) -> Number:
        phases = {self.wetting_phase, self.non_wetting_phase}
        if FluidPhase.WATER not in phases:
            return 1.0  # gas-oil table, water not represented
        if self.wetting_phase == FluidPhase.WATER:
            return np.max(self.wetting_phase_relative_permeability)
        return np.max(self.non_wetting_phase_relative_permeability)

    def get_gas_relperm_endpoint(self) -> Number:
        phases = {self.wetting_phase, self.non_wetting_phase}
        if FluidPhase.GAS not in phases:
            return 1.0  # oil-water table, gas not represented
        if self.wetting_phase == FluidPhase.GAS:
            return np.max(self.wetting_phase_relative_permeability)
        return np.max(self.non_wetting_phase_relative_permeability)

    def get_relative_permeabilities(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for all three phases from a two-phase table.

        The absent phase always returns zero. The reference saturation dispatched
        to the underlying PCHIP interpolant is chosen by inspecting which phases
        this table covers and what `reference_phase` is declared:

        - **Oil-water table** (phases are OIL and WATER): krg = 0. Sw or So is
        forwarded to the interpolant according to `reference_phase`.
        - **Gas-oil table** (phases are GAS and OIL): krw = 0. So or Sg is
        forwarded according to `reference_phase`.

        Minimum relperm min_values declared on the table are applied automatically
        inside the underlying query methods and propagate transparently.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :return: `RelativePermeabilities` dict with keys `"water"`, `"oil"`, `"gas"`.
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)

        phases = {self.wetting_phase, self.non_wetting_phase}

        if phases == {FluidPhase.OIL, FluidPhase.WATER}:
            if self.wetting_phase == FluidPhase.WATER:
                krw = self.get_wetting_phase_relative_permeability(
                    sw, non_wetting_saturation=so
                )
                kro = self.get_non_wetting_phase_relative_permeability(
                    sw, non_wetting_saturation=so
                )
            else:
                kro = self.get_wetting_phase_relative_permeability(
                    so, non_wetting_saturation=sw
                )
                krw = self.get_non_wetting_phase_relative_permeability(
                    so, non_wetting_saturation=sw
                )
            if is_scalar:
                return RelativePermeabilities(
                    water=krw.item(),  # type: ignore
                    oil=kro.item(),  # type: ignore
                    gas=0.0,
                )
            return RelativePermeabilities(water=krw, oil=kro, gas=zeros)  # type: ignore[typeddict-item]

        if phases == {FluidPhase.OIL, FluidPhase.GAS}:
            if self.wetting_phase == FluidPhase.OIL:
                kro = self.get_wetting_phase_relative_permeability(
                    so, non_wetting_saturation=sg
                )
                krg = self.get_non_wetting_phase_relative_permeability(
                    so, non_wetting_saturation=sg
                )
            else:
                krg = self.get_wetting_phase_relative_permeability(
                    sg, non_wetting_saturation=so
                )
                kro = self.get_non_wetting_phase_relative_permeability(
                    sg, non_wetting_saturation=so
                )
            if is_scalar:
                return RelativePermeabilities(
                    water=0.0,
                    oil=kro.item(),  # type: ignore
                    gas=krg.item(),  # type: ignore
                )
            return RelativePermeabilities(water=zeros, oil=kro, gas=krg)  # type: ignore[typeddict-item]

        raise ValidationError(
            f"Cannot dispatch three-phase saturations to a two-phase table with phases "
            f"{self.wetting_phase!r} / {self.non_wetting_phase!r}. "
            f"Expected OIL+WATER or OIL+GAS."
        )

    def get_relative_permeability_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute the nine partial derivatives of three-phase relative permeabilities
        for a two-phase table.

        The absent phase contributes zero to every derivative. The two active
        phases contribute only along their natural saturation axis; all cross
        derivatives (e.g. dkrw/dSg for an oil-water table) are zero.

        Where a minimum relperm min_value is active on this table (raw kr ≤ min_value),
        the corresponding derivative is zeroed out by the underlying query method,
        keeping the Jacobian consistent with the min_value kr value.

        Returned layout (same as `ThreePhaseRelPermTable`):

            dKrw_dSw, dKro_dSw, dKrg_dSw,
            dKrw_dSo, dKro_dSo, dKrg_dSo,
            dKrw_dSg, dKro_dSg, dKrg_dSg

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )

        phases = {self.wetting_phase, self.non_wetting_phase}

        if phases == {FluidPhase.OIL, FluidPhase.WATER}:
            # krg = 0 everywhere; all krg derivatives are zero
            if self.wetting_phase == FluidPhase.WATER:
                # reference: wetting=Sw or non_wetting=So depending on reference_phase
                d_krw = self.get_wetting_phase_relative_permeability_derivative(
                    sw, non_wetting_saturation=so
                )
                d_kro = self.get_non_wetting_phase_relative_permeability_derivative(
                    sw, non_wetting_saturation=so
                )
                # Both curves live on the same reference axis
                if self.reference_phase == "wetting":
                    # reference is Sw: d/dSw is non-zero, d/dSo and d/dSg are zero
                    d_krw_d_sw, d_krw_d_so, d_krw_d_sg = d_krw, zeros, zeros
                    d_kro_d_sw, d_kro_d_so, d_kro_d_sg = d_kro, zeros, zeros
                else:
                    # reference is So: d/dSo is non-zero
                    d_krw_d_sw, d_krw_d_so, d_krw_d_sg = zeros, d_krw, zeros
                    d_kro_d_sw, d_kro_d_so, d_kro_d_sg = zeros, d_kro, zeros
            else:
                # wetting phase is OIL
                d_kro = self.get_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sw
                )
                d_krw = self.get_non_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sw
                )
                if self.reference_phase == "wetting":
                    d_kro_d_sw, d_kro_d_so, d_kro_d_sg = zeros, d_kro, zeros
                    d_krw_d_sw, d_krw_d_so, d_krw_d_sg = zeros, d_krw, zeros
                else:
                    d_kro_d_sw, d_kro_d_so, d_kro_d_sg = d_kro, zeros, zeros
                    d_krw_d_sw, d_krw_d_so, d_krw_d_sg = d_krw, zeros, zeros

            # Repack into the 9-tuple order expected by RelativePermeabilityDerivatives
            results = (
                d_krw_d_sw,
                d_kro_d_sw,
                zeros,
                d_krw_d_so,
                d_kro_d_so,
                zeros,
                d_krw_d_sg,
                d_kro_d_sg,
                zeros,
            )
            if is_scalar:
                results = tuple(r.item() for r in results)  # type: ignore
            return RelativePermeabilityDerivatives(
                dKrw_dSw=results[0],
                dKro_dSw=results[1],
                dKrg_dSw=results[2],
                dKrw_dSo=results[3],
                dKro_dSo=results[4],
                dKrg_dSo=results[5],
                dKrw_dSg=results[6],
                dKro_dSg=results[7],
                dKrg_dSg=results[8],
            )

        if phases == {FluidPhase.OIL, FluidPhase.GAS}:
            # krw = 0 everywhere; all krw derivatives are zero
            if self.wetting_phase == FluidPhase.OIL:
                d_kro = self.get_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sg
                )
                d_krg = self.get_non_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sg
                )
                if self.reference_phase == "wetting":
                    # reference is So
                    d_kro_d_sw, d_kro_d_so, d_kro_d_sg = zeros, d_kro, zeros
                    d_krg_d_sw, d_krg_d_so, d_krg_d_sg = zeros, d_krg, zeros
                else:
                    # reference is Sg
                    d_kro_d_sw, d_kro_d_so, d_kro_d_sg = zeros, zeros, d_kro
                    d_krg_d_sw, d_krg_d_so, d_krg_d_sg = zeros, zeros, d_krg
            else:
                # wetting phase is GAS
                d_krg = self.get_wetting_phase_relative_permeability_derivative(
                    sg, non_wetting_saturation=so
                )
                d_kro = self.get_non_wetting_phase_relative_permeability_derivative(
                    sg, non_wetting_saturation=so
                )
                if self.reference_phase == "wetting":
                    # reference is Sg
                    d_krg_d_sw, d_krg_d_so, d_krg_d_sg = zeros, zeros, d_krg
                    d_kro_d_sw, d_kro_d_so, d_kro_d_sg = zeros, zeros, d_kro
                else:
                    # reference is So
                    d_krg_d_sw, d_krg_d_so, d_krg_d_sg = zeros, d_krg, zeros
                    d_kro_d_sw, d_kro_d_so, d_kro_d_sg = zeros, d_kro, zeros

            results = (
                zeros,
                d_kro_d_sw,
                d_krg_d_sw,
                zeros,
                d_kro_d_so,
                d_krg_d_so,
                zeros,
                d_kro_d_sg,
                d_krg_d_sg,
            )
            if is_scalar:
                results = tuple(r.item() for r in results)  # type: ignore
            return RelativePermeabilityDerivatives(
                dKrw_dSw=results[0],
                dKro_dSw=results[1],
                dKrg_dSw=results[2],
                dKrw_dSo=results[3],
                dKro_dSo=results[4],
                dKrg_dSo=results[5],
                dKrw_dSg=results[6],
                dKro_dSg=results[7],
                dKrg_dSg=results[8],
            )

        raise ValidationError(
            f"Cannot dispatch three-phase derivatives to a two-phase table with phases "
            f"{self.wetting_phase!r} / {self.non_wetting_phase!r}. "
            f"Expected OIL+WATER or OIL+GAS."
        )


@relperm_table
@attrs.frozen
class ThreePhaseRelPermTable(
    RelativePermeabilityTable,
    serializers={"mixing_rule": serialize_mixing_rule},
    deserializers={"mixing_rule": deserialize_mixing_rule},
    load_exclude={"supports_vector"},
    dump_exclude={"supports_vector"},
):
    """
    Three-phase relative permeability lookup table, with mixing rules.

    Interpolates relative permeabilities for water, oil, and gas based on
    saturation values. Uses two `TwoPhaseRelPermTable` instances (oil-water
    and gas-oil) and a mixing rule for oil in the three-phase system.

    Each two-phase table declares its own `reference_phase` ("wetting" or
    "non_wetting"), so the correct saturation is dispatched automatically -
    no assumptions are hard-coded about whether a table is indexed by So or Sg.

    Minimum relperm min_values on the two-phase tables propagate automatically
    into this three-phase table since the two-phase table methods apply the
    min_value before returning.  For the mixing-rule chain rule the min_value
    derivative (zeroed in the flat region) is also used consistently, so the
    assembled Jacobian never disagrees with the kr values.

    Supported mixing rules: `max_rule`, `stone_I_rule`, `stone_II_rule`, etc.
    """

    __type__ = "three_phase_relperm_table"

    oil_water_table: TwoPhaseRelPermTable
    """Relative permeability table for the oil-water system."""

    gas_oil_table: TwoPhaseRelPermTable
    """Relative permeability table for the gas-oil system."""

    mixing_rule: typing.Optional[typing.Union[MixingRule, str]] = None
    """
    Mixing rule function or name to compute oil relative permeability in the
    three-phase system. Signature:

        rule(kro_w, kro_g, krw, krg, kr_max,
             water_saturation, oil_saturation, gas_saturation) -> kro

    If None, falls back to min(kro_w, kro_g).
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        if {
            self.oil_water_table.wetting_phase,
            self.oil_water_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_table` must be between water and oil phases."
            )

        if {
            self.gas_oil_table.wetting_phase,
            self.gas_oil_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError("`gas_oil_table` must be between oil and gas phases.")

        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return self.oil_water_table.wetting_phase  # type:ignore[return-value]  # ty:ignore[invalid-return-type]

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return self.gas_oil_table.wetting_phase  # type:ignore[return-value]    # ty:ignore[invalid-return-type]

    def get_oil_relperm_endpoint(self) -> Number:
        return self.oil_water_table.get_oil_relperm_endpoint()

    def get_water_relperm_endpoint(self) -> Number:
        return self.oil_water_table.get_water_relperm_endpoint()

    def get_gas_relperm_endpoint(self) -> Number:
        return self.gas_oil_table.get_gas_relperm_endpoint()

    def get_relative_permeabilities(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        Each two-phase table is queried using its declared `reference_phase`:
        - `reference_phase="wetting"` - the wetting phase saturation is passed
        - `reference_phase="non_wetting"` - the non-wetting phase saturation is passed

        Minimum relperm min_values declared on the two-phase tables are applied
        automatically inside the table query methods and propagate into the
        mixing-rule inputs transparently.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :return: `RelativePermeabilities` dict with keys "water", "oil", "gas".
        """
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )

        if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
            raise ValidationError(
                f"Saturations must be between 0 and 1. Sw: {_show_invalid_saturation(sw)}, So: {_show_invalid_saturation(so)}, Sg: {_show_invalid_saturation(sg)}"
            )

        # Normalize if saturations do not sum to 1
        total_saturation = sw + so + sg
        needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
        if np.any(needs_norm):
            sw = np.where(needs_norm, sw / total_saturation, sw)
            so = np.where(needs_norm, so / total_saturation, so)
            sg = np.where(needs_norm, sg / total_saturation, sg)

        # For Oil-water table
        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            krw = self.oil_water_table.get_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
            kro_w = self.oil_water_table.get_non_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
        else:
            # Oil is wetting phase
            kro_w = self.oil_water_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )
            krw = self.oil_water_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )

        # For Gas-oil table
        if self.gas_oil_table.wetting_phase == FluidPhase.OIL:
            kro_g = self.gas_oil_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
            krg = self.gas_oil_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
        else:
            # Gas is wetting phase
            krg = self.gas_oil_table.get_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )
            kro_g = self.gas_oil_table.get_non_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )

        kr_max = self.get_oil_relperm_endpoint()

        # Three-phase oil mixing rule
        if self.mixing_rule is not None:
            kro = self.mixing_rule(  # type: ignore
                kro_w=kro_w,
                kro_g=kro_g,
                krw=krw,
                krg=krg,
                kr_max=kr_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )
        else:
            kro = np.minimum(kro_w, kro_g)

        if is_scalar:
            return RelativePermeabilities(
                water=krw.item(),  # type: ignore
                oil=kro.item(),  # type: ignore
                gas=krg.item(),  # type: ignore
            )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def get_relative_permeability_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute all nine partial derivatives of the three-phase relative
        permeabilities with respect to water saturation, oil saturation, and
        gas saturation.

        Returns a dictionary containing:
        `
        (dkrw/dSw, dkrw/dSo, dkrw/dSg,
        dkro/dSw, dkro/dSo, dkro/dSg,
        dkrg/dSw, dkrg/dSo, dkrg/dSg)
        ``

        Water and gas relative permeability derivatives are computed analytically
        from the slopes of the underlying two-phase lookup tables.  Where a
        minimum relperm min_value is active on a two-phase table, the derivative
        returned by that table is already zeroed out in the min_value region,
        so the assembled nine-derivative Jacobian is automatically consistent
        with the min_value kr values.

        Oil relative permeability derivatives use the full extended chain rule
        through the three-phase mixing rule:
        ``
        dkro/dSalpha = (d_kro/d_kro_w) * (d_kro_w/d_Salpha)
                     + (d_kro/d_kro_g) * (d_kro_g/d_Salpha)
                     + (d_kro/d_krw)   * (d_krw/d_Salpha)
                     + (d_kro/d_krg)   * (d_krg/d_Salpha)
                     + (d_kro/d_Salpha)_explicit_in_mixing_rule
        ``

        The `d_kro/d_krw` and `d_kro/d_krg` terms are non-zero only for
        rules that use the actual two-phase water/gas kr values (e.g. the full
        Stone II rule); they are zero for all other built-in rules.

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        so = np.atleast_1d(np.asarray(oil_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)

        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        # Oil-water table derivatives: krw and kro_w
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            if oil_water_table.reference_phase == "wetting":
                d_krw_d_sw = (
                    oil_water_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=sw,
                        non_wetting_saturation=so,
                    )
                )
                d_krw_d_so = zeros.copy()
                d_krw_d_sg = zeros.copy()

                d_kro_w_d_sw = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=sw,
                    non_wetting_saturation=so,
                )
                d_kro_w_d_so = zeros.copy()
                d_kro_w_d_sg = zeros.copy()
            else:
                d_krw_d_sw = zeros.copy()
                d_krw_d_so = (
                    oil_water_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=sw,
                        non_wetting_saturation=so,
                    )
                )
                d_krw_d_sg = zeros.copy()

                d_kro_w_d_sw = zeros.copy()
                d_kro_w_d_so = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=sw,
                    non_wetting_saturation=so,
                )
                d_kro_w_d_sg = zeros.copy()
        else:
            # Oil is wetting phase (oil-wet system)
            if oil_water_table.reference_phase == "wetting":
                d_krw_d_sw = zeros.copy()
                d_krw_d_so = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=so,
                    non_wetting_saturation=sw,
                )
                d_krw_d_sg = zeros.copy()

                d_kro_w_d_sw = zeros.copy()
                d_kro_w_d_so = (
                    oil_water_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=so,
                        non_wetting_saturation=sw,
                    )
                )
                d_kro_w_d_sg = zeros.copy()
            else:
                d_krw_d_sw = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=so,
                    non_wetting_saturation=sw,
                )
                d_krw_d_so = zeros.copy()
                d_krw_d_sg = zeros.copy()

                d_kro_w_d_sw = (
                    oil_water_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=so,
                        non_wetting_saturation=sw,
                    )
                )
                d_kro_w_d_so = zeros.copy()
                d_kro_w_d_sg = zeros.copy()

        # Gas-oil table derivatives: krg and kro_g
        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            if gas_oil_table.reference_phase == "wetting":
                d_krg_d_sw = zeros.copy()
                d_krg_d_so = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=so,
                    non_wetting_saturation=sg,
                )
                d_krg_d_sg = zeros.copy()

                d_kro_g_d_sw = zeros.copy()
                d_kro_g_d_so = (
                    gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=so,
                        non_wetting_saturation=sg,
                    )
                )
                d_kro_g_d_sg = zeros.copy()
            else:
                d_krg_d_sw = zeros.copy()
                d_krg_d_so = zeros.copy()
                d_krg_d_sg = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=so,
                    non_wetting_saturation=sg,
                )

                d_kro_g_d_sw = zeros.copy()
                d_kro_g_d_so = zeros.copy()
                d_kro_g_d_sg = (
                    gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=so,
                        non_wetting_saturation=sg,
                    )
                )
        else:
            # Gas is wetting phase (uncommon)
            if gas_oil_table.reference_phase == "wetting":
                d_krg_d_sw = zeros.copy()
                d_krg_d_so = zeros.copy()
                d_krg_d_sg = (
                    gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=sg,
                        non_wetting_saturation=so,
                    )
                )

                d_kro_g_d_sw = zeros.copy()
                d_kro_g_d_so = zeros.copy()
                d_kro_g_d_sg = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=sg,
                    non_wetting_saturation=so,
                )
            else:
                d_krg_d_sw = zeros.copy()
                d_krg_d_so = (
                    gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=sg,
                        non_wetting_saturation=so,
                    )
                )
                d_krg_d_sg = zeros.copy()

                d_kro_g_d_sw = zeros.copy()
                d_kro_g_d_so = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=sg,
                    non_wetting_saturation=so,
                )
                d_kro_g_d_sg = zeros.copy()

        # Forward two-phase oil/water/gas kr values for mixing rule call
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            kro_w = oil_water_table.get_non_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
            krw = oil_water_table.get_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
        else:
            kro_w = oil_water_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )
            krw = oil_water_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )

        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            kro_g = gas_oil_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
            krg = gas_oil_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
        else:
            kro_g = gas_oil_table.get_non_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )
            krg = gas_oil_table.get_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )

        kr_max = self.get_oil_relperm_endpoint()

        # Three-phase oil mixing rule derivatives
        mixing_rule = typing.cast(typing.Optional[MixingRule], self.mixing_rule)
        if mixing_rule is None:
            # Fallback: kro = min(kro_w, kro_g)
            kro_w_arr = np.asarray(kro_w, dtype=np.float64)
            kro_g_arr = np.asarray(kro_g, dtype=np.float64)
            d_kro_d_kro_w = np.where(kro_w_arr <= kro_g_arr, 1.0, 0.0)
            d_kro_d_kro_g = np.where(kro_g_arr <= kro_w_arr, 1.0, 0.0)
            d_kro_d_krw = zeros.copy()
            d_kro_d_krg = zeros.copy()
            d_kro_d_sw_explicit = zeros.copy()
            d_kro_d_so_explicit = zeros.copy()
            d_kro_d_sg_explicit = zeros.copy()
        else:
            derivatives = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,  # type: ignore[arg-type]
                kro_w=kro_w,
                kro_g=kro_g,
                krw=krw,
                krg=krg,
                kr_max=kr_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            d_kro_d_kro_w = derivatives["d_kro_d_kro_w"]
            d_kro_d_kro_g = derivatives["d_kro_d_kro_g"]
            d_kro_d_krw = derivatives["d_kro_d_krw"]
            d_kro_d_krg = derivatives["d_kro_d_krg"]
            d_kro_d_sw_explicit = derivatives["d_kro_d_sw_explicit"]
            d_kro_d_so_explicit = derivatives["d_kro_d_so_explicit"]
            d_kro_d_sg_explicit = derivatives["d_kro_d_sg_explicit"]

        # Extended chain rule:
        # dkro/dSalpha = (d_kro/d_kro_w) * (d_kro_w/d_Salpha)
        #              + (d_kro/d_kro_g) * (d_kro_g/d_Salpha)
        #              + (d_kro/d_krw)   * (d_krw/d_Salpha)
        #              + (d_kro/d_krg)   * (d_krg/d_Salpha)
        #              + (d_kro/d_Salpha)_explicit
        d_kro_d_sw = (
            d_kro_d_kro_w * d_kro_w_d_sw
            + d_kro_d_kro_g * d_kro_g_d_sw
            + d_kro_d_krw * d_krw_d_sw
            + d_kro_d_krg * d_krg_d_sw
            + d_kro_d_sw_explicit
        )
        d_kro_d_so = (
            d_kro_d_kro_w * d_kro_w_d_so
            + d_kro_d_kro_g * d_kro_g_d_so
            + d_kro_d_krw * d_krw_d_so
            + d_kro_d_krg * d_krg_d_so
            + d_kro_d_so_explicit
        )
        d_kro_d_sg = (
            d_kro_d_kro_w * d_kro_w_d_sg
            + d_kro_d_kro_g * d_kro_g_d_sg
            + d_kro_d_krw * d_krw_d_sg
            + d_kro_d_krg * d_krg_d_sg
            + d_kro_d_sg_explicit
        )

        results = (
            d_krw_d_sw,
            d_kro_d_sw,
            d_krg_d_sw,
            d_krw_d_so,
            d_kro_d_so,
            d_krg_d_so,
            d_krw_d_sg,
            d_kro_d_sg,
            d_krg_d_sg,
        )
        if is_scalar:
            results = tuple(r.item() for r in results)  # type: ignore
            return RelativePermeabilityDerivatives(
                dKrw_dSw=results[0],
                dKro_dSw=results[1],
                dKrg_dSw=results[2],
                dKrw_dSo=results[3],
                dKro_dSo=results[4],
                dKrg_dSo=results[5],
                dKrw_dSg=results[6],
                dKro_dSg=results[7],
                dKrg_dSg=results[8],
            )

        return RelativePermeabilityDerivatives(
            dKrw_dSw=d_krw_d_sw,
            dKro_dSw=d_kro_d_sw,
            dKrg_dSw=d_krg_d_sw,
            dKrw_dSo=d_krw_d_so,
            dKro_dSo=d_kro_d_so,
            dKrg_dSo=d_krg_d_so,
            dKrw_dSg=d_krw_d_sg,
            dKro_dSg=d_kro_d_sg,
            dKrg_dSg=d_krg_d_sg,
        )
