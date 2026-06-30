"""Base capillary pressure tables for multi-phase flow simulations."""

import threading
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from bores.errors import ValidationError
from bores.serialization import Serializable, make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.tables.rock_fluid.utils import build_pchip_interpolant
from bores.typing import (
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FluidPhase,
    NDimension,
    NumberArray,
    NumberOrArray,
    OneDimension,
    Spacing,
)
from bores.utils import array as bores_array

__all__ = [
    "ThreePhaseCapillaryPressureTable",
    "TwoPhaseCapillaryPressureTable",
    "capillary_pressure_table",
]


class CapillaryPressureTable(StoreSerializable):
    """
    Protocol for a capillary pressure model that computes
    capillary pressures based on fluid saturations.
    """

    __abstract_serializable__ = True

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return FluidPhase.WATER

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return FluidPhase.OIL

    def get_capillary_pressures(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        raise NotImplementedError

    def get_capillary_pressure_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute capillary pressure derivatives for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        raise NotImplementedError

    def __call__(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Computes capillary pressures based on fluid saturations.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        return self.get_capillary_pressures(
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
    ) -> CapillaryPressureDerivatives:
        """
        Compute capillary pressure derivatives for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        return self.get_capillary_pressure_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )


_CAPILLARY_PRESSURE_TABLES: typing.Dict[str, typing.Type[CapillaryPressureTable]] = {}
"""Registry for capillary pressure table types."""
_capillary_pressure_table_lock = threading.Lock()
capillary_pressure_table = make_serializable_type_registrar(
    base_cls=CapillaryPressureTable,
    registry=_CAPILLARY_PRESSURE_TABLES,
    key_attr="__type__",
    lock=_capillary_pressure_table_lock,
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)


def list_capillary_pressure_tables() -> typing.List[str]:
    """
    List all registered capillary pressure table types.

    :return: List of capillary pressure table type names.
    """
    with _capillary_pressure_table_lock:
        return list(_CAPILLARY_PRESSURE_TABLES.keys())


def get_capillary_pressure_table(name: str) -> typing.Type[CapillaryPressureTable]:
    """
    Get a registered capillary pressure table type by name.

    :param name: Name of the capillary pressure table type.
    :return: Capillary pressure table class.
    :raises KeyError: If the type name is not registered.
    """
    with _capillary_pressure_table_lock:
        if name not in _CAPILLARY_PRESSURE_TABLES:
            raise ValidationError(
                f"Capillary pressure table type '{name}' is not registered. "
                f"Use `@capillary_pressure_table` to register it. "
                f"Available types: {list(_CAPILLARY_PRESSURE_TABLES.keys())}"
            )
        return _CAPILLARY_PRESSURE_TABLES[name]


@attrs.frozen
class TwoPhaseCapillaryPressureTable(
    Serializable,
    load_exclude={"_interp", "_d_interp"},
    dump_exclude={"_interp", "_d_interp"},
):
    """
    Two-phase capillary pressure lookup table backed by a PCHIP interpolant.

    Interpolates capillary pressure for two fluid phases based on a
    **reference saturation** value. The reference saturation can be either
    the wetting or non-wetting phase saturation, depending on how the table
    was constructed, e.g. a gas-oil table may be indexed by oil saturation
    (wetting) or by gas saturation (non-wetting).

    Supports both scalar and array inputs.

    **Grid scaling** (`number_of_base_points` / `number_of_endpoint_extra_points`):

    Identical semantics to `TwoPhaseRelPermTable`. The default
    `number_of_endpoint_extra_points=30` (vs 20 for relperm) reflects that Pc curves are
    typically unbounded near residual saturation, making endpoint fidelity
    especially important for implicit convergence. Pass `number_of_base_points=0`
    to disable scaling.
    """

    wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The wetting fluid phase, e.g. WATER (oil-water system) or OIL (gas-oil system)."""

    non_wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The non-wetting fluid phase, e.g. OIL (oil-water system) or GAS (gas-oil system)."""

    reference_saturation: NumberArray[OneDimension] = attrs.field(converter=bores_array)
    """
    Saturation values used as the x-axis for interpolation, monotonically
    increasing. May represent either the wetting or non-wetting phase
    saturation depending on `reference_phase`.
    """

    capillary_pressure: NumberArray[OneDimension] = attrs.field(converter=bores_array)
    """
    Capillary pressure values `Pc = P_non_wetting - P_wetting` corresponding
    to each `reference_saturation` point.
    """

    reference_phase: typing.Literal["wetting", "non_wetting"] = attrs.field(
        default="wetting"
    )
    """
    Which phase the `reference_saturation` axis represents.
 
    - `"wetting"` - the x-axis holds wetting-phase saturation values.
      This is the standard convention for oil-water tables (Sw axis) and for
      gas-oil tables indexed by So.
    - `"non_wetting"` - the x-axis holds non-wetting-phase saturation
      values.  Use this for gas-oil tables indexed by Sg.
 
    This attribute does not change the interpolation mechanics. It only
    records which physical saturation must be supplied by the caller so that
    `ThreePhaseCapillaryPressureTable` (and any other consumer) can dispatch
    the correct saturation without hard-coding assumptions.
    """

    number_of_base_points: int = attrs.field(default=200)
    """
    Target number of base knot points used when expanding the raw saturation
    grid before fitting the PCHIP interpolant.
 
    Pass `0` to disable grid scaling and use the raw knots directly.
    """

    number_of_endpoint_extra_points: int = attrs.field(default=30)
    """
    Number of extra knots injected into the first and last 10 % of the
    saturation range during grid expansion (see `number_of_base_points`).
 
    The higher default of 30 (vs 20 for relperm) reflects that Pc curves vary
    most steeply near residual saturations. Pass `0` to disable.
    """

    spacing: Spacing = attrs.field(default="cosine")
    """Grid spacing mode used when building the expanded knot grid."""

    _interp: PchipInterpolator = attrs.field(init=False, repr=False)
    _d_interp: PchipInterpolator = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.reference_phase not in ("wetting", "non_wetting"):
            raise ValidationError(
                f"`reference_phase` must be 'wetting' or 'non_wetting', "
                f"got {self.reference_phase!r}"
            )
        if len(self.reference_saturation) != len(self.capillary_pressure):
            raise ValidationError(
                f"`reference_saturation` and `capillary_pressure` arrays must have "
                f"the same length.  Got {len(self.reference_saturation)} vs "
                f"{len(self.capillary_pressure)}"
            )
        if len(self.reference_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation.")
        if not np.all(np.diff(self.reference_saturation) >= 0):
            raise ValidationError(
                "`reference_saturation` must be monotonically increasing."
            )

        # Build interpolant
        pchip, dpchip = build_pchip_interpolant(
            reference_saturation=self.reference_saturation,
            values=self.capillary_pressure,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
        )
        object.__setattr__(self, "_interp", pchip)
        object.__setattr__(self, "_d_interp", dpchip)

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

        :param wetting_saturation: Current wetting-phase saturation.
        :param non_wetting_saturation: Current non-wetting-phase saturation.
        :return: The saturation to use as the interpolation x-value.
        """
        if self.reference_phase == "non_wetting":
            return non_wetting_saturation
        return wetting_saturation

    def _query_interp(
        self,
        reference: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        """
        Evaluate the capillary pressure PCHIP interpolant at `reference`,
        applying constant extrapolation at the boundaries.

        :param reference: Query saturation value(s) - scalar or array.
        :return: Capillary pressure value(s).
        """
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(reference)
        x_min = self._interp.x[0]
        x_max = self._interp.x[-1]

        result = self._interp(np.clip(sat, x_min, x_max))
        result = np.where(sat < x_min, self.capillary_pressure[0], result)
        result = np.where(sat > x_max, self.capillary_pressure[-1], result)

        if is_scalar:
            return result.item()
        return typing.cast(
            NumberOrArray[NDimension], result.reshape(sat.shape, copy=False)
        )

    def _query_d_interp(
        self, reference: NumberOrArray[NDimension]
    ) -> NumberOrArray[NDimension]:
        """
        Evaluate the analytical PCHIP derivative at `reference`, returning
        zero outside the knot range.

        :param reference: Query saturation value(s) - scalar or array.
        :return: Derivative value(s).
        """
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(reference)
        x_min = self._d_interp.x[0]
        x_max = self._d_interp.x[-1]

        result = self._d_interp(np.clip(sat, x_min, x_max))
        result = np.where((sat < x_min) | (sat > x_max), 0.0, result)

        if is_scalar:
            return result.item()
        return typing.cast(
            NumberOrArray[NDimension], result.reshape(sat.shape, copy=False)
        )

    def get_capillary_pressure(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
    ) -> NumberOrArray[NDimension]:
        """
        Get capillary pressure at the given saturation(s).

        When `reference_phase="wetting"`, only `wetting_saturation` is
        needed. When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Capillary pressure value(s) matching the input shape.
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        return self._query_interp(ref)

    def get_capillary_pressure_derivative(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
    ) -> NumberOrArray[NDimension]:
        """
        Derivative of capillary pressure with respect to the reference
        saturation axis of this table: `dPc / d(reference_saturation)`.

        Evaluated from the analytical PCHIP derivative. Zero outside the
        tabulated range (constant extrapolation = zero slope).

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
        return self._query_d_interp(ref)

    def __call__(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        **kwargs: typing.Any,
    ) -> NumberOrArray[NDimension]:
        """
        Get capillary pressure at the given saturation(s).

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Capillary pressure value(s).
        """
        return self.get_capillary_pressure(wetting_saturation, non_wetting_saturation)


@capillary_pressure_table
@attrs.frozen
class ThreePhaseCapillaryPressureTable(
    CapillaryPressureTable,
    load_exclude={"supports_vector"},
    dump_exclude={"supports_vector"},
):
    """
    Three-phase capillary pressure lookup table.

    Uses two two-phase tables (oil-water and gas-oil) to compute capillary pressures
    in a three-phase system (water, oil, gas).

    Pcow = Po - Pw (oil-water capillary pressure)
    Pcgo = Pg - Po (gas-oil capillary pressure)
    """

    __type__ = "three_phase_capillary_pressure_table"

    oil_water_table: TwoPhaseCapillaryPressureTable
    """
    Capillary pressure table for oil-water system (wetting phase = water or oil).

    A table of Pcow against wetting phase saturation (water saturation if water is wetting phase,
    oil saturation if oil is wetting phase).
    """

    gas_oil_table: TwoPhaseCapillaryPressureTable
    """
    Capillary pressure table for gas-oil system (wetting phase = oil).

    A table of Pcgo against oil saturation.
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        """Validate that the tables are set up correctly for three-phase flow."""
        if {
            self.oil_water_table.wetting_phase,
            self.oil_water_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_table` must be between water and oil phases."
            )
        if {self.gas_oil_table.wetting_phase, self.gas_oil_table.non_wetting_phase} != {
            FluidPhase.OIL,
            FluidPhase.GAS,
        }:
            raise ValidationError("`gas_oil_table` must be between oil and gas phases.")

        if self.oil_water_table.wetting_phase == self.gas_oil_table.non_wetting_phase:
            raise ValidationError(
                "Wetting phase of `oil_water_table` cannot be the same as non-wetting phase of `gas_oil_table`."
            )

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return typing.cast(FluidPhase, self.oil_water_table.wetting_phase)

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return typing.cast(FluidPhase, self.gas_oil_table.wetting_phase)

    def get_capillary_pressures(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for the three-phase system.

        Each sub-table is queried using its declared `reference_phase`:

        - `reference_phase="wetting"` - the wetting-phase saturation is passed.
        - `reference_phase="non_wetting"` - the non-wetting-phase saturation is
        passed.

        For the oil-water table the wetting phase is either WATER or OIL.
        For the gas-oil table the wetting phase is always OIL, but the table may
        be indexed by So (`reference_phase="wetting"`) or by Sg
        (`reference_phase="non_wetting"`).

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :return: `CapillaryPressures` dictionary.
        """
        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        # Oil-water capillary pressure. Dispatch to the wetting-phase saturation
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            pcow = oil_water_table.get_capillary_pressure(
                wetting_saturation=water_saturation,
                non_wetting_saturation=oil_saturation,
            )
        else:
            # Oil is the wetting phase
            pcow = oil_water_table.get_capillary_pressure(
                wetting_saturation=oil_saturation,
                non_wetting_saturation=water_saturation,
            )

        # Gas-oil capillary pressure. Dispatch to the correct saturation
        # depending on the table's reference_phase axis.
        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            pcgo = gas_oil_table.get_capillary_pressure(
                wetting_saturation=oil_saturation,
                non_wetting_saturation=gas_saturation,
            )
        else:
            # Gas is the wetting phase (uncommon but supported)
            pcgo = gas_oil_table.get_capillary_pressure(
                wetting_saturation=gas_saturation,
                non_wetting_saturation=oil_saturation,
            )

        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def get_capillary_pressure_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the oil-water and gas-oil capillary
        pressures with respect to saturation.

        Returns a `CapillaryPressureDerivatives` dictionary with four entries:

        - `dPcow_dSw`: non-zero when the oil-water table's reference axis is
        water saturation (`wetting_phase=WATER, reference_phase="wetting"`).
        - `dPcow_dSo`: non-zero when the oil-water table's reference axis is oil
        saturation (`wetting_phase=OIL, reference_phase="wetting"`).
        - `dPcgo_dSo`: non-zero when the gas-oil table's reference axis is oil
        saturation (`reference_phase="wetting"`, the wetting phase being OIL).
        - `dPcgo_dSg`: non-zero when the gas-oil table's reference axis is gas
        saturation (`reference_phase="non_wetting"`, or gas is the wetting
        phase with `reference_phase="wetting"`).

        At most one of `dPcow_dSw` / `dPcow_dSo` is non-zero, and at most one
        of `dPcgo_dSo` / `dPcgo_dSg` is non-zero, for a given table
        configuration.  All derivatives are exact piecewise-linear slopes from the
        underlying two-phase tables.

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array).
        :param gas_saturation: Gas saturation (scalar or array).
        :return: `CapillaryPressureDerivatives` dictionary.
        """
        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        zero = 0.0 if is_scalar else np.zeros_like(water_saturation)

        # Oil-water derivatives
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            if oil_water_table.reference_phase == "wetting":
                # Table indexed by Sw (wetting phase) -> derivative is dPcow/dSw
                d_pcow_d_sw = oil_water_table.get_capillary_pressure_derivative(
                    wetting_saturation=water_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                )
                d_pcow_d_so = zero
            else:
                # reference_phase="non_wetting" and wetting_phase=WATER means
                # table is indexed by So (non-wetting phase) -> derivative is dPcow/dSo
                d_pcow_d_sw = zero
                d_pcow_d_so = oil_water_table.get_capillary_pressure_derivative(
                    wetting_saturation=water_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                )
        else:
            # Oil is the wetting phase.  The reference_phase attribute then
            # controls whether the table is indexed by So ("wetting") or Sw
            # ("non_wetting").  Either way the derivative is with respect to
            # whichever saturation is the reference axis.
            if oil_water_table.reference_phase == "wetting":
                # Table indexed by So -> derivative is dPcow/dSo
                d_pcow_d_sw = zero
                d_pcow_d_so = oil_water_table.get_capillary_pressure_derivative(
                    wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=water_saturation,  # type: ignore[arg-type]
                )
            else:
                # reference_phase="non_wetting" and wetting_phase=OIL means the
                # table is indexed by water saturation (the non-wetting phase here
                # is water) -> derivative is dPcow/dSw
                d_pcow_d_sw = oil_water_table.get_capillary_pressure_derivative(
                    wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=water_saturation,  # type: ignore[arg-type]
                )
                d_pcow_d_so = zero

        # Gas-oil derivatives
        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            if gas_oil_table.reference_phase == "wetting":
                # Table indexed by So -> derivative is dPcgo/dSo
                d_pcgo_d_so = gas_oil_table.get_capillary_pressure_derivative(
                    wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=gas_saturation,  # type: ignore[arg-type]
                )
                d_pcgo_d_sg = zero
            else:
                # reference_phase="non_wetting" -> table indexed by Sg
                d_pcgo_d_so = zero
                d_pcgo_d_sg = gas_oil_table.get_capillary_pressure_derivative(
                    wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=gas_saturation,  # type: ignore[arg-type]
                )
        else:
            # Gas is the wetting phase (uncommon). reference_phase="wetting"
            # means indexed by Sg, "non_wetting" means indexed by So.
            if gas_oil_table.reference_phase == "wetting":
                d_pcgo_d_so = zero
                d_pcgo_d_sg = gas_oil_table.get_capillary_pressure_derivative(
                    wetting_saturation=gas_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                )
            else:
                d_pcgo_d_so = gas_oil_table.get_capillary_pressure_derivative(
                    wetting_saturation=gas_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                )
                d_pcgo_d_sg = zero

        return CapillaryPressureDerivatives(
            dPcow_dSw=d_pcow_d_sw,
            dPcow_dSo=d_pcow_d_so,
            dPcgo_dSo=d_pcgo_d_so,
            dPcgo_dSg=d_pcgo_d_sg,
        )
