import logging
import typing
import warnings
from collections.abc import Mapping
from os import PathLike

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import (  # type: ignore[import-untyped]
    PchipInterpolator,
    RectBivariateSpline,
    RegularGridInterpolator,
    interp1d,
)
from typing_extensions import Self

from bores.constants import UnitConversionTable, c
from bores.correlations.arrays import (
    compute_gas_density,
    compute_live_oil_density,
    compute_oil_api_gravity,
)
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.model.properties import PVT
from bores.precision import get_dtype
from bores.stores import StoreSerializable
from bores.tables.pvt.data import PVTData, PVTDataSet
from bores.typing import (
    Boolean,
    BooleanArray,
    FluidPhase,
    InterpolationMethod,
    NDimension,
    Number,
    NumberArray,
    OneDimension,
    TableQuery,
    TableResult,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PVTTable",
    "PVTTables",
    "DEFAULT_GAS_CLAMPS",
    "DEFAULT_OIL_CLAMPS",
    "DEFAULT_WATER_CLAMPS",
]

INTERPOLATION_DEGREES: typing.Dict[str, int] = {"linear": 1, "cubic": 3}


########################
# Default clamp ranges #
########################

DEFAULT_OIL_CLAMPS: typing.Dict[str, typing.Tuple[float, float]] = {
    "viscosity": (1e-6, 1e4),
    "density": (1.0, 80.0),
    "formation_volume_factor": (0.5, 5.0),
    "compressibility": (0.0, 1e-1),
    "solution_gor": (0.0, 5000.0),
    "specific_gravity": (0.5, 1.2),
    "molecular_weight": (10.0, 600.0),
}

DEFAULT_GAS_CLAMPS: typing.Dict[str, typing.Tuple[float, float]] = {
    "viscosity": (1e-6, 1e2),
    "density": (0.001, 50.0),
    "formation_volume_factor": (1e-6, 100.0),
    "compressibility": (0.0, 1e-1),
    "compressibility_factor": (0.1, 3.0),
    "solubility_in_water": (0.0, 500.0),
    "specific_gravity": (0.5, 2.0),
    "molecular_weight": (2.0, 100.0),
    "vaporized_oil_ratio": (0.0, 1000.0),
}

DEFAULT_WATER_CLAMPS: typing.Dict[str, typing.Tuple[float, float]] = {
    "viscosity": (1e-6, 10.0),
    "density": (30.0, 80.0),
    "formation_volume_factor": (0.9, 2.0),
    "compressibility": (0.0, 1e-2),
    "specific_gravity": (0.9, 1.3),
    "molecular_weight": (18.0, 20.0),
}

PHASE_DEFAULT_CLAMPS: typing.Dict[
    FluidPhase, typing.Dict[str, typing.Tuple[float, float]]
] = {
    FluidPhase.OIL: DEFAULT_OIL_CLAMPS,
    FluidPhase.GAS: DEFAULT_GAS_CLAMPS,
    FluidPhase.WATER: DEFAULT_WATER_CLAMPS,
}


def _build_pchip_2d_interpolator(
    pressures: NumberArray[NDimension],
    temperatures: NumberArray[NDimension],
    table: NumberArray[NDimension],
    dtype: npt.DTypeLike,
) -> typing.Callable[
    [NumberArray[NDimension], NumberArray[NDimension]], NumberArray[OneDimension]
]:
    """
    Build a two-stage PCHIP 2-D interpolator for a property table.

    Fits independent PCHIP interpolants along the pressure axis at each
    temperature knot, then re-interpolates along temperature for each query
    point. Preserves monotonicity along each axis independently.

    The returned callable has the same interface as
    `RectBivariateSpline.ev`: `(pressure_points, temperature_points) -> values`.

    :param pressures: 1-D array of pressure knots (psi), strictly increasing.
    :param temperatures: 1-D array of temperature knots (°F), strictly increasing.
    :param table: 2-D array of shape `(n_p, n_t)` containing property values.
    :returns: Callable with signature `(p, t) -> values`.
    """
    _p_interps: typing.List[PchipInterpolator] = [
        PchipInterpolator(pressures, table[:, j], extrapolate=True)
        for j in range(len(temperatures))
    ]

    def _ev(
        p: NumberArray[NDimension], t: NumberArray[NDimension]
    ) -> NumberArray[OneDimension]:
        p = p.astype(dtype, copy=False).ravel()  # type: ignore
        t = t.astype(dtype, copy=False).ravel()  # type: ignore
        n = len(p)

        p_interp_values = np.empty((len(temperatures), n), dtype=dtype)
        for j, interp in enumerate(_p_interps):
            p_interp_values[j] = interp(p)

        result = np.empty(n, dtype=dtype)
        for i in range(n):
            result[i] = PchipInterpolator(
                temperatures, p_interp_values[:, i], extrapolate=True
            )(t[i])
        return result

    return _ev


def _build_pchip_2d_derivative_interpolator(
    pressures: NumberArray[NDimension],
    temperatures: NumberArray[NDimension],
    table: NumberArray[NDimension],
    dtype: npt.DTypeLike,
) -> typing.Callable[
    [NumberArray[NDimension], NumberArray[NDimension]], NumberArray[OneDimension]
]:
    """
    Build a two-stage PCHIP interpolator for `∂table/∂P`.

    For each temperature column, builds a PCHIP and takes its analytical
    first derivative (via `PchipInterpolator.derivative`), then
    re-interpolates along temperature exactly as the primal interpolator does.

    :param pressures: 1-D array of pressure knots, strictly increasing.
    :param temperatures: 1-D array of temperature knots, strictly increasing.
    :param table: 2-D array of shape `(n_p, n_t)`.
    :returns: Callable `(p, t) -> ∂table/∂P`.
    """
    _dp_interps: typing.List[PchipInterpolator] = [
        PchipInterpolator(pressures, table[:, j], extrapolate=True).derivative(1)
        for j in range(len(temperatures))
    ]

    def _ev(
        p: NumberArray[NDimension], t: NumberArray[NDimension]
    ) -> NumberArray[OneDimension]:
        p = p.astype(dtype, copy=False).ravel()  # type: ignore
        t = t.astype(dtype, copy=False).ravel()  # type: ignore
        n = len(p)

        vals = np.empty((len(temperatures), n), dtype=dtype)
        for j, d_interp in enumerate(_dp_interps):
            vals[j] = d_interp(p)

        result = np.empty(n, dtype=dtype)
        for i in range(n):
            result[i] = PchipInterpolator(temperatures, vals[:, i], extrapolate=True)(
                t[i]
            )
        return result

    return _ev


class PVTTable(StoreSerializable):
    """
    Phase-aware PVT property lookup with pre-built interpolators.

    Wraps a `PVTData` instance and builds scipy / PCHIP interpolators for
    every table column, including derived tables (density, compressibility) if
    they were not supplied in the data.

    **Construction-time derivation**

    When `PVT` reference densities are supplied and the data
    does not already contain a `density_table` or `compressibility_table`,
    this constructor builds those tables now from the primary columns
    (Bo / Rs / Bg / Rv / Bw) so that all subsequent simulation lookups are a
    single interpolator call.

    **Derivative methods**

    Every primary property `prop(p, t)` has a matching `prop_dp(p, t)`
    that returns `∂prop/∂P` using the analytical derivative of the same PCHIP
    or the partial-derivative of the `RectBivariateSpline`. No finite
    differences are used.

    **Interpolation modes**

    - `"linear"`: `RectBivariateSpline` (kx=ky=1) for 2-D; `interp1d` for 1-D.
    - `"cubic"`: two-stage PCHIP-along-axes for 2-D; monotone cubic for 1-D.

    The cubic path preserves monotonicity independently along each axis,
    avoiding oscillation on noisy PVT data.

    **Water phase**

    When only one salinity value is present the 3-D tables are pre-sliced to
    2-D at construction time and the fast 2-D interpolators are used.

    **Extrapolation**

    Outside table bounds the interpolators extrapolate monotonically. Set
    `warn_on_extrapolation=True` to log warnings.
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        data: typing.Union[PVTData, PathLike[str], str],
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        clamps: typing.Union[
            typing.Literal[False],
            typing.Optional[typing.Mapping[str, typing.Tuple[float, float]]],
        ] = None,
        pvt: typing.Optional[PVT] = None,
        dtype: typing.Optional[npt.DTypeLike] = None,
    ) -> None:
        """
        Build a `PVTTable` from raw tabulated data.

        :param data: Phase-tagged `PVTData` or path to a serialised file.
        :param interpolation_method: `"linear"` (default) or `"cubic"`.
        :param validate: Run physical-consistency checks on `data`.
        :param warn_on_extrapolation: Log a warning when queries fall outside
            the table pressure / temperature / salinity bounds.
        :param clamps: Override per-property clamp ranges. Merged on top of
            phase-appropriate defaults. Pass `False` to disable clamping.
        :param pvt: Optional `PVT` carrying stock-tank
            reference densities (`standard_oil_density`, `standard_gas_density`,
            `standard_water_density`). When provided, missing `density_table` and
            `compressibility_table` entries are built automatically.
        """
        if interpolation_method not in INTERPOLATION_DEGREES:
            raise ValidationError(
                f"Invalid interpolation_method {interpolation_method!r}. "
                f"Must be one of: {list(INTERPOLATION_DEGREES.keys())}"
            )
        if not isinstance(data, PVTData):
            pvt_data = PVTData.from_file(data)  # type: ignore[assignment]
            if pvt_data is None:
                raise ValidationError(f"Invalid PVT data file {data!s}")
            data = pvt_data

        self._phase = data.phase
        self.interpolation_method: InterpolationMethod = interpolation_method
        self.validate = validate
        self.warn_on_extrapolation = warn_on_extrapolation

        if clamps is False:
            self.clamps: typing.Dict[str, typing.Tuple[float, float]] = {}
        else:
            self.clamps = {
                **PHASE_DEFAULT_CLAMPS[data.phase],  # type: ignore
                **(clamps or {}),
            }

        if interpolation_method != "linear":
            if len(data.pressures) < 4:
                raise ValidationError(
                    f"At least 4 pressure points required for {interpolation_method!r} "
                    f"interpolation, got {len(data.pressures)}."
                )
            if len(data.temperatures) < 4:
                raise ValidationError(
                    f"At least 4 temperature points required for "
                    f"{interpolation_method!r} interpolation, "
                    f"got {len(data.temperatures)}."
                )

        self._validate_data(data)
        if validate:
            self._check_physical_consistency(data)

        self._extrapolation_bounds: typing.Dict[str, typing.Tuple[float, float]] = {
            "pressure": (data.pressures[0], data.pressures[-1]),
            "temperature": (data.temperatures[0], data.temperatures[-1]),
        }
        if data.salinities is not None:
            self._extrapolation_bounds["salinity"] = (
                data.salinities[0],
                data.salinities[-1],
            )

        self._bubble_point_ndim: typing.Optional[int] = None
        if data.phase == FluidPhase.OIL and data.bubble_point_pressures is not None:
            self._bubble_point_ndim = data.bubble_point_pressures.ndim

        self.default_salinity: typing.Optional[float] = (
            data.salinities[0] if data.salinities is not None else None
        )
        self._water_constant_salinity: bool = (
            data.phase == FluidPhase.WATER
            and data.salinities is not None
            and len(data.salinities) == 1
        )

        if dtype is None:
            if data.dtype is not None:
                self.dtype = data.dtype
            else:
                self.dtype = get_dtype()
                data.ensure_dtype(self.dtype, force=True)
        else:
            self.dtype = np.dtype(dtype)
            data.ensure_dtype(self.dtype, force=False)

        # Potentially augment data with derived tables before building interpolators
        data = self._build_derived_tables(data, pvt)

        self._data = data
        self._interpolatants: typing.Dict[str, typing.Any] = {}
        self._derivative_interpolatants: typing.Dict[str, typing.Any] = {}
        self._build_interpolants(data)

        logger.debug(
            "PVTTable init: phase=%s, n_p=%d, n_t=%d, method=%r, "
            "n_interp=%d, n_deriv=%d",
            data.phase.value.upper(),  # type: ignore
            len(data.pressures),
            len(data.temperatures),
            interpolation_method,
            len(self._interpolatants),
            len(self._derivative_interpolatants),
        )

    def _build_derived_tables(
        self, data: PVTData, pvt: typing.Optional[PVT]
    ) -> PVTData:
        """
        Build `density_table` and `compressibility_table` if absent.

        Uses the standard reservoir-engineering formulas so that at
        simulation time every property is a single interpolator call.

        :param data: Source `PVTData`.
        :param pvt: Reference densities carrier.
        :returns: `PVTData` with derived tables filled in (may be the same
            object if nothing was missing).
        """
        phase = typing.cast(FluidPhase, data.phase)
        pressures = data.pressures
        temperatures = data.temperatures
        n_p = len(pressures)
        n_t = len(temperatures)
        pressure_table, temperature_table = np.meshgrid(
            pressures, temperatures, indexing="ij"
        )
        dtype = self.dtype

        updates: typing.Dict[str, typing.Any] = {}

        # Oil Phase
        if phase == FluidPhase.OIL:
            oil_fvf_table = data.formation_volume_factor_table
            solution_gor_table = data.solution_gor_table

            # Density: ρo = (ρo,SC + Rs·ρg,SC) / Bo
            if (
                data.density_table is None
                and oil_fvf_table is not None
                and solution_gor_table is not None
            ):
                if pvt is not None:
                    standard_oil_density = pvt.standard_gas_density
                    standard_gas_density = pvt.standard_gas_density
                    if (
                        standard_oil_density is not None
                        and standard_gas_density is not None
                    ):
                        density_table = (
                            standard_oil_density
                            + solution_gor_table * standard_gas_density
                        ) / oil_fvf_table
                        updates["density_table"] = density_table.astype(
                            dtype, copy=False
                        )
                else:
                    # Fall back to correlation-based density
                    specific_gravity_table = (
                        data.specific_gravity_table
                        if data.specific_gravity_table is not None
                        else np.full((n_p, n_t), 0.85, dtype=dtype)
                    )
                    oil_api_table = compute_oil_api_gravity(specific_gravity_table)
                    gas_gravity_table = np.full((n_p, n_t), 0.65, dtype=dtype)
                    updates["density_table"] = compute_live_oil_density(
                        api_gravity=oil_api_table,
                        gas_gravity=gas_gravity_table,
                        gas_to_oil_ratio=solution_gor_table,
                        formation_volume_factor=oil_fvf_table,  # type: ignore
                    )

            # Compressibility: co = -(1/Bo)·(∂Bo/∂P)
            if data.compressibility_table is None and oil_fvf_table is not None:
                # Build dBo/dP via PCHIP derivative at each temperature column
                dbo_dp = np.empty((n_p, n_t), dtype=dtype)
                for j in range(n_t):
                    d = PchipInterpolator(pressures, oil_fvf_table[:, j]).derivative(1)
                    dbo_dp[:, j] = d(pressures)

                oil_compressibility_table = -(1.0 / oil_fvf_table) * dbo_dp
                np.clip(
                    oil_compressibility_table, 0.0, 1e-1, out=oil_compressibility_table
                )
                updates["compressibility_table"] = oil_compressibility_table.astype(
                    dtype, copy=False
                )

            # Specific gravity
            if data.specific_gravity_table is None and data.density_table is not None:
                specific_gravity_table = (
                    data.density_table / c.STANDARD_WATER_DENSITY_IMPERIAL
                )
                updates["specific_gravity_table"] = specific_gravity_table.astype(
                    dtype, copy=False
                )

        # Gas Phase
        elif phase == FluidPhase.GAS:
            gas_fvf_table = data.formation_volume_factor_table
            vaporized_oil_ratio_table = data.vaporized_oil_ratio_table
            compressibility_factor_table = data.compressibility_factor_table

            # Density: ρg = (ρg,SC + Rv·ρo,SC) / Bg  [wet] or ρg,SC / Bg [dry]
            if data.density_table is None and gas_fvf_table is not None:
                if pvt is not None:
                    standard_oil_density = pvt.standard_gas_density
                    standard_gas_density = pvt.standard_gas_density
                    if standard_gas_density is not None:
                        if (
                            vaporized_oil_ratio_table is not None
                            and standard_oil_density is not None
                        ):
                            density = (
                                standard_gas_density
                                + vaporized_oil_ratio_table * standard_oil_density
                            ) / gas_fvf_table
                        else:
                            density = standard_gas_density / gas_fvf_table
                        updates["density_table"] = density.astype(dtype, copy=False)
                else:
                    # Correlation path - requires z-factor
                    if compressibility_factor_table is not None:
                        specific_gravity_table = (
                            data.specific_gravity_table
                            if data.specific_gravity_table is not None
                            else np.full((n_p, n_t), 0.65, dtype=dtype)
                        )
                        updates["density_table"] = compute_gas_density(
                            pressure=pressure_table,
                            temperature=temperature_table,
                            gas_gravity=specific_gravity_table,
                            gas_compressibility_factor=compressibility_factor_table,
                        )

            # Compressibility: cg = 1/P - (1/z)·(∂z/∂P)
            if data.compressibility_table is None:
                if compressibility_factor_table is not None:
                    dz_dp = np.empty((n_p, n_t), dtype=dtype)
                    for j in range(n_t):
                        d = PchipInterpolator(
                            pressures, compressibility_factor_table[:, j]
                        ).derivative(1)
                        dz_dp[:, j] = d(pressures)

                    gas_compressibility_table = (
                        1.0 / pressure_table
                        - (1.0 / compressibility_factor_table) * dz_dp
                    )
                    np.clip(
                        gas_compressibility_table,
                        0.0,
                        1e-1,
                        out=gas_compressibility_table,
                    )
                    updates["compressibility_table"] = gas_compressibility_table.astype(
                        dtype, copy=False
                    )

                elif gas_fvf_table is not None:
                    # Fallback: cg ≈ -(1/Bg)·(∂Bg/∂P)
                    dbg_dp = np.empty((n_p, n_t), dtype=dtype)
                    for j in range(n_t):
                        d = PchipInterpolator(
                            pressures, gas_fvf_table[:, j]
                        ).derivative(1)
                        dbg_dp[:, j] = d(pressures)

                    gas_compressibility_table = -(1.0 / gas_fvf_table) * dbg_dp
                    np.clip(
                        gas_compressibility_table,
                        0.0,
                        1e-1,
                        out=gas_compressibility_table,
                    )
                    updates["compressibility_table"] = gas_compressibility_table.astype(
                        dtype, copy=False
                    )

        # Water Phase
        elif phase == FluidPhase.WATER:
            gas_free_water_fvf_table = data.gas_free_water_fvf_table

            # Density: ρw = ρw,SC / Bw  (simplified; gas-free)
            if data.density_table is None and gas_free_water_fvf_table is not None:
                if pvt is not None:
                    standard_water_density = pvt.standard_water_density
                    if standard_water_density is not None:
                        salinities = data.salinities
                        if salinities is not None:
                            n_s = len(salinities)
                            density_3d_table = np.empty((n_p, n_t, n_s), dtype=dtype)
                            for s_idx in range(n_s):
                                density_3d_table[:, :, s_idx] = (
                                    standard_water_density / gas_free_water_fvf_table
                                )
                            updates["density_table"] = density_3d_table

            # Compressibility: cw = -(1/Bw_gf)·(∂Bw_gf/∂P)  [undersaturated]
            if (
                data.compressibility_table is None
                and gas_free_water_fvf_table is not None
            ):
                dbw_dp = np.empty((n_p, n_t), dtype=dtype)
                for j in range(n_t):
                    d = PchipInterpolator(
                        pressures, gas_free_water_fvf_table[:, j]
                    ).derivative(1)
                    dbw_dp[:, j] = d(pressures)

                water_compressibility_2d_table = (
                    -(1.0 / gas_free_water_fvf_table) * dbw_dp
                )
                np.clip(
                    water_compressibility_2d_table,
                    0.0,
                    1e-2,
                    out=water_compressibility_2d_table,
                )
                if data.salinities is not None:
                    n_s = len(data.salinities)
                    water_compressibility_3d_table = np.broadcast_to(
                        water_compressibility_2d_table[:, :, np.newaxis],
                        (n_p, n_t, n_s),
                    ).copy()
                    updates["compressibility_table"] = (
                        water_compressibility_3d_table.astype(dtype, copy=False)
                    )

        if not updates:
            return data
        return attrs.evolve(data, **updates)

    def _validate_data(self, data: PVTData) -> None:
        """Validate grid monotonicity, shapes, and phase-specific field rules."""
        pressures = data.pressures
        temperatures = data.temperatures
        salinities = data.salinities
        n_p = len(pressures)
        n_t = len(temperatures)
        n_s = len(salinities) if salinities is not None else None

        if pressures.ndim != 1:
            raise ValidationError("`pressures` must be 1-dimensional.")
        if temperatures.ndim != 1:
            raise ValidationError("`temperatures` must be 1-dimensional.")
        if not np.all(np.diff(pressures) > 0):
            raise ValidationError(
                "`pressures` must be strictly monotonically increasing."
            )
        if not np.all(np.diff(temperatures) > 0):
            raise ValidationError(
                "`temperatures` must be strictly monotonically increasing."
            )
        if salinities is not None:
            if salinities.ndim != 1:
                raise ValidationError("`salinities` must be 1-dimensional.")
            if not np.all(np.diff(salinities) > 0):
                raise ValidationError(
                    "`salinities` must be strictly monotonically increasing."
                )

        bubble_point_arr = data.bubble_point_pressures
        if bubble_point_arr is not None:
            if bubble_point_arr.ndim == 1:
                if len(bubble_point_arr) != n_t:
                    raise ValidationError(
                        f"`bubble_point_pressures` 1-D length {len(bubble_point_arr)} must "
                        f"match n_temperatures={n_t}."
                    )

            elif bubble_point_arr.ndim == 2:
                rs_axis = data.solution_gas_to_oil_ratios
                if rs_axis is None:
                    raise ValidationError(
                        "2-D `bubble_point_pressures` requires `solution_gas_to_oil_ratios`."
                    )
                if not np.all(np.diff(rs_axis) > 0):
                    raise ValidationError(
                        "`solution_gas_to_oil_ratios` must be strictly monotonically increasing."
                    )
                if bubble_point_arr.shape != (len(rs_axis), n_t):
                    raise ValidationError(
                        f"`bubble_point_pressures` shape {bubble_point_arr.shape} must be "
                        f"({len(rs_axis)}, {n_t})."
                    )
            else:
                raise ValidationError("`bubble_point_pressures` must be 1-D or 2-D.")

        _2d_fields = [
            "viscosity_table",
            "density_table",
            "formation_volume_factor_table",
            "compressibility_table",
            "specific_gravity_table",
            "molecular_weight_table",
            "solution_gor_table",
            "compressibility_factor_table",
            "gas_free_water_fvf_table",
            "vaporized_oil_ratio_table",
        ]
        for field_name in _2d_fields:
            arr = getattr(data, field_name, None)
            if arr is not None and arr.ndim == 2 and arr.shape != (n_p, n_t):
                raise ValidationError(
                    f"`{field_name}` shape {arr.shape} must be "
                    f"(n_pressures={n_p}, n_temperatures={n_t})."
                )

        _3d_field_names = [
            "solubility_in_water_table",
            "bubble_point_pressure_table",
        ]
        phase = typing.cast(FluidPhase, data.phase)
        shared_3d = (
            [
                "viscosity_table",
                "density_table",
                "formation_volume_factor_table",
                "compressibility_table",
                "specific_gravity_table",
                "molecular_weight_table",
            ]
            if phase == FluidPhase.WATER
            else []
        )
        for field_name in _3d_field_names + shared_3d:
            arr = getattr(data, field_name, None)
            if arr is None:
                continue
            if arr.ndim == 3:
                if n_s is None:
                    raise ValidationError(
                        f"`{field_name}` is 3-D but `salinities` was not provided."
                    )
                if arr.shape != (n_p, n_t, n_s):
                    raise ValidationError(
                        f"`{field_name}` shape {arr.shape} must be "
                        f"(n_p={n_p}, n_t={n_t}, n_s={n_s})."
                    )

        if data.solubility_in_water_table is not None and salinities is None:
            raise ValidationError(
                "`solubility_in_water_table` is 3-D and requires `salinities`."
            )

    def _check_physical_consistency(self, data: PVTData) -> None:
        """Phase-aware physical sanity checks on table values."""
        phase = typing.cast(FluidPhase, data.phase)
        if data.viscosity_table is not None and np.any(data.viscosity_table <= 0):
            raise ValidationError(
                f"{phase.value.upper()} viscosity must be positive everywhere."
            )
        if data.density_table is not None and np.any(data.density_table <= 0):
            raise ValidationError(
                f"{phase.value.upper()} density must be positive everywhere."
            )
        if data.formation_volume_factor_table is not None and np.any(
            data.formation_volume_factor_table <= 0
        ):
            raise ValidationError(
                f"{phase.value.upper()} FVF must be positive everywhere."
            )
        if (
            phase == FluidPhase.GAS
            and data.compressibility_factor_table is not None
            and np.any(data.compressibility_factor_table <= 0)
        ):
            raise ValidationError("Gas z-factor must be positive everywhere.")

        if (
            phase == FluidPhase.GAS
            and data.density_table is not None
            and np.any(data.density_table >= 50.0)
        ):
            warnings.warn(
                "Gas density table contains values >= 50 lbm/ft³ - unusually max_value. "
                "Verify units (expected lbm/ft³).",
                UserWarning,
                stacklevel=3,
            )

    def _build_interpolants(self, data: PVTData) -> None:
        """Build scipy / PCHIP interpolators and their pressure-derivatives."""
        phase = typing.cast(FluidPhase, data.phase)
        pressures = data.pressures
        temperatures = data.temperatures
        salinities = data.salinities
        k = INTERPOLATION_DEGREES[self.interpolation_method]
        use_pchip = self.interpolation_method == "cubic"

        def _register_2d(name: str, table: typing.Optional[npt.NDArray]) -> None:
            """Register a 2-D property interpolator and its ∂/∂P derivative."""
            if table is None:
                return

            if use_pchip:
                self._interpolatants[name] = _build_pchip_2d_interpolator(
                    pressures, temperatures, table, dtype=self.dtype
                )
                self._derivative_interpolatants[name] = (
                    _build_pchip_2d_derivative_interpolator(
                        pressures, temperatures, table, dtype=self.dtype
                    )
                )
            else:
                spline = RectBivariateSpline(
                    x=pressures, y=temperatures, z=table, kx=k, ky=k
                )
                self._interpolatants[name] = spline
                self._derivative_interpolatants[name] = spline.partial_derivative(1, 0)

        def _register_3d(name: str, table: typing.Optional[npt.NDArray]) -> None:
            """
            Register a 3-D property interpolator.

            For single-salinity tables the 3-D array is pre-sliced to 2-D and
            a fast 2-D interpolator is registered instead.
            """
            if table is None or salinities is None:
                return

            if self._water_constant_salinity:
                _register_2d(name, table[:, :, 0])
            else:
                self._interpolatants[name] = RegularGridInterpolator(
                    points=(pressures, temperatures, salinities),
                    values=table,
                    method=self.interpolation_method,
                    bounds_error=False,
                    fill_value=None,
                )

        # Shared properties
        if phase == FluidPhase.WATER:
            _register_3d("viscosity", data.viscosity_table)
            _register_3d("density", data.density_table)
            _register_3d("formation_volume_factor", data.formation_volume_factor_table)
            _register_3d("compressibility", data.compressibility_table)
            _register_3d("specific_gravity", data.specific_gravity_table)
            _register_3d("molecular_weight", data.molecular_weight_table)
        else:
            _register_2d("viscosity", data.viscosity_table)
            _register_2d("density", data.density_table)
            _register_2d("formation_volume_factor", data.formation_volume_factor_table)
            _register_2d("compressibility", data.compressibility_table)
            _register_2d("specific_gravity", data.specific_gravity_table)
            _register_2d("molecular_weight", data.molecular_weight_table)

        # Phase-specific
        if phase == FluidPhase.OIL:
            _register_2d("solution_gor", data.solution_gor_table)
            _register_2d("vaporized_oil_ratio", None)  # not applicable for oil

            bubble_point_arr = data.bubble_point_pressures
            if bubble_point_arr is not None:
                if bubble_point_arr.ndim == 1:
                    self._interpolatants["bubble_point_pressure"] = interp1d(
                        x=temperatures,
                        y=bubble_point_arr,
                        kind=self.interpolation_method,
                        bounds_error=False,
                        fill_value="extrapolate",  # type: ignore[arg-type]
                    )
                else:
                    rs_axis = data.solution_gas_to_oil_ratios
                    assert rs_axis is not None
                    if use_pchip:
                        self._interpolatants["bubble_point_pressure"] = (
                            _build_pchip_2d_interpolator(
                                rs_axis,
                                temperatures,
                                bubble_point_arr,  # type: ignore
                                dtype=self.dtype,
                            )
                        )
                        self._derivative_interpolatants["bubble_point_pressure_drs"] = (
                            _build_pchip_2d_derivative_interpolator(
                                rs_axis,
                                temperatures,
                                bubble_point_arr,  # type: ignore
                                dtype=self.dtype,
                            )
                        )
                    else:
                        spline = RectBivariateSpline(
                            x=rs_axis, y=temperatures, z=bubble_point_arr, kx=k, ky=k
                        )
                        self._interpolatants["bubble_point_pressure"] = spline
                        self._derivative_interpolatants["bubble_point_pressure_drs"] = (
                            spline.partial_derivative(1, 0)
                        )

        if phase == FluidPhase.GAS:
            _register_2d("compressibility_factor", data.compressibility_factor_table)
            _register_2d("vaporized_oil_ratio", data.vaporized_oil_ratio_table)
            _register_3d("solubility_in_water", data.solubility_in_water_table)

            dp = data.dew_point_pressures
            if dp is not None:
                self._interpolatants["dew_point_pressure"] = interp1d(
                    x=temperatures,
                    y=dp,
                    kind=self.interpolation_method,
                    bounds_error=False,
                    fill_value="extrapolate",  # type: ignore[arg-type]
                )

        if phase == FluidPhase.WATER:
            _register_3d("bubble_point_pressure", data.bubble_point_pressure_table)
            _register_2d("gas_free_fvf", data.gas_free_water_fvf_table)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "data": self._data.dump(recurse),
            "interpolation_method": self.interpolation_method,
            "validate": self.validate,
            "warn_on_extrapolation": self.warn_on_extrapolation,
            "clamps": {
                k: list(v)
                for k, v in self.clamps.items()
                if v != PHASE_DEFAULT_CLAMPS.get(self.phase, {}).get(k)
            },
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> "PVTTable":
        pvt_data = PVTData.load(data["data"])
        clamps = data.get("clamps", None)
        if isinstance(clamps, dict):
            clamps = {k: tuple(v) for k, v in clamps.items()}
        return cls(
            data=pvt_data,
            interpolation_method=data.get("interpolation_method", "linear"),
            validate=data.get("validate", True),
            warn_on_extrapolation=data.get("warn_on_extrapolation", False),
            clamps=clamps,
        )

    @property
    def phase(self) -> FluidPhase:
        """Fluid phase this table describes."""
        return typing.cast(FluidPhase, self._phase)

    @property
    def unit_system(self) -> UnitSystem:
        """Unit system of the underlying table data."""
        return typing.cast(UnitSystem, self._data.unit_system)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `PVTTable` with all dimensional quantities rescaled to *target*.

        Dimensionless properties (specific gravity, compressibility factor,
        vaporized oil ratio, solution GOR ratio) and multiplier-type quantities
        are copied unchanged. Pressure axes, densities, FVFs, and viscosities
        are rescaled using `get_conversion_factors`.

        :param target: Target `UnitSystem`.
        :returns `PVTTable`: New `PVTTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        return self.__class__(
            data=self._data.convert(target, table=table),
            interpolation_method=self.interpolation_method,
            validate=False,  # already validated at construction
            warn_on_extrapolation=self.warn_on_extrapolation,
            clamps=False,  # caller's clamps are in old units; let them re-specify
            dtype=self.dtype,
        )

    def exists(self, name: str) -> bool:
        """Return `True` if an interpolator for *name* was built."""
        return name in self._interpolatants

    def _warn_extrapolation(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> None:
        if not self.warn_on_extrapolation:
            return

        pressure_arr = np.atleast_1d(pressure)
        temperature_arr = np.atleast_1d(temperature)
        min_pressure, max_pressure = self._extrapolation_bounds["pressure"]
        min_temperature, max_temperature = self._extrapolation_bounds["temperature"]
        if np.any(pressure_arr < min_pressure) or np.any(pressure_arr > max_pressure):
            logger.warning(
                "Pressure extrapolation: queried P ∈ [%.1f, %.1f] psi, "
                "table range [%.1f, %.1f] psi",
                float(pressure_arr.min()),
                float(pressure_arr.max()),
                min_pressure,
                max_pressure,
            )

        if np.any(temperature_arr < min_temperature) or np.any(
            temperature_arr > max_temperature
        ):
            logger.warning(
                "Temperature extrapolation: queried T ∈ [%.1f, %.1f] °F, "
                "table range [%.1f, %.1f] °F",
                float(temperature_arr.min()),
                float(temperature_arr.max()),
                min_temperature,
                max_temperature,
            )

        if salinity is not None and "salinity" in self._extrapolation_bounds:
            salinity_arr = np.atleast_1d(salinity)
            min_salinity, max_salinity = self._extrapolation_bounds["salinity"]
            if np.any(salinity_arr < min_salinity) or np.any(
                salinity_arr > max_salinity
            ):
                logger.warning(
                    "Salinity extrapolation: queried S ∈ [%.0f, %.0f] ppm, "
                    "table range [%.0f, %.0f] ppm",
                    float(salinity_arr.min()),
                    float(salinity_arr.max()),
                    min_salinity,
                    max_salinity,
                )

    def _pt_query(
        self,
        name: str,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        *,
        derivative: bool = False,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Evaluate a 2-D interpolator at `(pressure, temperature)` pairs.

        :param name: Key in `_interpolatants` / `_derivative_interpolatants`.
        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param derivative: If `True`, return `∂/∂P` instead of the value.
        :returns: Interpolated value (or derivative), or `None` when the
            table was not built.
        """
        store = self._derivative_interpolatants if derivative else self._interpolatants
        interp = store.get(name)
        if interp is None:
            return None
        if not derivative:
            self._warn_extrapolation(pressure, temperature)

        is_scalar = isinstance(pressure, (int, float, np.floating)) and isinstance(
            temperature, (int, float, np.floating)
        )
        dtype = self.dtype
        if is_scalar:
            raw = (
                interp.ev(pressure, temperature)
                if hasattr(interp, "ev")
                else interp(
                    np.atleast_1d(pressure), np.atleast_1d(temperature)
                ).ravel()[0]
            )
            result = dtype.type(raw)  # type: ignore[attr-defined]
        else:
            pressure_arr = np.atleast_1d(pressure)
            temperature_arr = np.atleast_1d(temperature)
            if pressure_arr.shape != temperature_arr.shape:
                if pressure_arr.size == 1:
                    pressure_arr = np.full_like(
                        temperature_arr, pressure_arr[0], dtype=dtype
                    )
                elif temperature_arr.size == 1:
                    temperature_arr = np.full_like(
                        pressure_arr, temperature_arr[0], dtype=dtype
                    )
                else:
                    raise ValidationError(
                        f"Incompatible shapes: pressure {pressure_arr.shape}, temperature {temperature_arr.shape}"
                    )
            result = (
                interp.ev(pressure_arr, temperature_arr)
                if hasattr(interp, "ev")
                else interp(pressure_arr, temperature_arr)
            )

        if not derivative:
            clamps = self.clamps
            if name in clamps:
                min_value, max_value = clamps[name]
                if isinstance(result, np.ndarray):
                    np.clip(result, min_value, max_value, out=result, dtype=dtype)
                else:
                    result = np.clip(result, min_value, max_value, dtype=dtype)  # type: ignore[assignment]

        if (is_array := isinstance(result, np.ndarray)) and result.ndim == 0:
            return typing.cast(Number, dtype.type(result))  # type: ignore[attr-defined]
        elif is_array and result.size == 1:
            return typing.cast(Number, dtype.type(result.item()))  # type: ignore[attr-defined]
        return typing.cast(NumberArray[NDimension], result)

    def _pts_query(
        self,
        name: str,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension],
        *,
        derivative: bool = False,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Evaluate a 3-D interpolator at `(pressure, temperature, salinity)` points.

        When `_water_constant_salinity` is `True` the registered
        interpolator is actually 2-D (the salinity slice was taken at
        construction) and `_pt_query` is called directly.

        :param name: Interpolator key.
        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).
        :param derivative: Return `∂/∂P` when `True`.
        :returns: Interpolated value or `None`.
        """
        if self._water_constant_salinity:
            return self._pt_query(name, pressure, temperature, derivative=derivative)

        interp = self._interpolatants.get(name)
        if interp is None:
            return None
        self._warn_extrapolation(pressure, temperature, salinity)

        dtype = self.dtype
        pressure_arr = np.atleast_1d(pressure)
        temperature_arr = np.atleast_1d(temperature)
        salinity_arr = np.atleast_1d(salinity)
        pressure_arr, temperature_arr, salinity_arr = np.broadcast_arrays(
            pressure_arr, temperature_arr, salinity_arr
        )
        points = np.column_stack([
            pressure_arr.ravel(),
            temperature_arr.ravel(),
            salinity_arr.ravel(),
        ])
        result = interp(points).reshape(pressure_arr.shape).astype(dtype, copy=False)

        clamps = self.clamps
        if not derivative and name in clamps:
            min_value, max_value = clamps[name]
            np.clip(result, min_value, max_value, out=result, dtype=dtype)

        if result.ndim == 0:
            return typing.cast(Number, dtype.type(result))  # type: ignore[attr-defined]
        if result.size == 1:
            return typing.cast(Number, dtype.type(result.item()))  # type: ignore[attr-defined]
        return typing.cast(NumberArray[NDimension], result)

    def _resolve_salinity(
        self, salinity: typing.Optional[TableQuery[NDimension]]
    ) -> TableQuery[NDimension]:
        if salinity is not None:
            return salinity
        if self.default_salinity is not None:
            return self.default_salinity
        raise ValidationError(
            "Salinity is required for water-phase PVT lookup but none was provided "
            "and no default salinity is set."
        )

    def formation_volume_factor(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
        solution_gor: typing.Optional[TableQuery[NDimension]] = None,
        bubble_point_pressure: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get formation volume factor `B` (bbl/STB for oil/water, ft³/scf for gas).

        **Oil phase** - saturated / undersaturated switching:

        - Saturated (P ≤ bubble_point_arr): interpolated directly from the Bo table.
        - Undersaturated (P > bubble_point_arr): applies the McCain exponential correction
          `Bo = Bob · exp(-co · (P - bubble_point_arr))`, where `co` comes from the
          compressibility table. Falls back to `Bob` if unavailable.

        **Gas and water** - direct table interpolation; no saturation switching.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :param solution_gor: Solution GOR (scf/STB). Oil, 2-D bubble_point_arr table only.
        :param bubble_point_pressure: Pre-computed bubble_point_arr (psi). Oil only; skips
            internal bubble_point_arr lookup when supplied.
        :returns: FVF or `None` if the table is not present.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "formation_volume_factor",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        if self._phase == FluidPhase.GAS:
            return self._pt_query("formation_volume_factor", pressure, temperature)

        if "formation_volume_factor" not in self._interpolatants:
            return None

        bubble_point_arr = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(
                temperature=temperature, solution_gor=solution_gor
            )
        )
        if bubble_point_arr is None:
            return self._pt_query("formation_volume_factor", pressure, temperature)

        dtype = self.dtype
        pressure_arr = np.atleast_1d(pressure).astype(dtype, copy=False)
        temperature_arr = np.atleast_1d(temperature).astype(dtype, copy=False)
        bubble_point_arr = np.atleast_1d(bubble_point_arr).astype(dtype, copy=False)
        pressure_arr, temperature_arr, bubble_point_arr = np.broadcast_arrays(
            pressure_arr, temperature_arr, bubble_point_arr
        )

        result = np.zeros_like(pressure_arr, dtype=dtype)
        saturated = pressure_arr <= bubble_point_arr
        unsaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self._pt_query(  # type: ignore[index]
                "formation_volume_factor",
                pressure_arr[saturated],
                temperature_arr[saturated],
            )
        if np.any(unsaturated):
            fvf_at_bubble_point = self._pt_query(
                "formation_volume_factor",
                bubble_point_arr[unsaturated],
                temperature_arr[unsaturated],
            )
            if "compressibility" in self._interpolatants:
                compressibility_at_bubble_point = self._pt_query(
                    "compressibility",
                    bubble_point_arr[unsaturated],
                    temperature_arr[unsaturated],
                )
                compressibility_at_pressure = self._pt_query(
                    "compressibility",
                    pressure_arr[unsaturated],
                    temperature_arr[unsaturated],
                )
                avg_compressibility_arr = 0.5 * (
                    np.asarray(compressibility_at_bubble_point)
                    + np.asarray(compressibility_at_pressure)
                )
                result[unsaturated] = np.asarray(fvf_at_bubble_point) * np.exp(
                    -avg_compressibility_arr
                    * (pressure_arr[unsaturated] - bubble_point_arr[unsaturated])
                )
            else:
                result[unsaturated] = np.asarray(fvf_at_bubble_point)

        if "formation_volume_factor" in self.clamps:
            min_value, max_value = self.clamps["formation_volume_factor"]
            np.clip(result, min_value, max_value, out=result, dtype=dtype)

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result) if result.size == 1 else result,  # type: ignore[attr-defined]
        )

    def formation_volume_factor_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂B/∂P` (psi⁻¹ · bbl/STB or psi⁻¹ · ft³/scf).

        Direct derivative of the FVF interpolator - no finite differences.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :returns: `∂B/∂P` or `None` if table is absent.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "formation_volume_factor",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
                derivative=True,
            )
        return self._pt_query(
            "formation_volume_factor", pressure, temperature, derivative=True
        )

    def viscosity(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
        solution_gor: typing.Optional[TableQuery[NDimension]] = None,
        bubble_point_pressure: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get fluid viscosity `μ` (cP).

        **Oil phase** - saturated / undersaturated switching using the
        Beggs-Robinson undersaturated correction above bubble point.

        **Water / gas** - direct table interpolation.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).  Water phase only.
        :param solution_gor: Solution GOR (scf/STB).  Oil, 2-D bubble_point_arr table only.
        :param bubble_point_pressure: Pre-computed bubble_point_arr (psi).  Oil only.
        :returns: Viscosity in cP, or `None` if table is absent.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "viscosity",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        if self._phase == FluidPhase.GAS:
            return self._pt_query("viscosity", pressure, temperature)

        if "viscosity" not in self._interpolatants:
            return None

        bubble_point_arr = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(
                temperature=temperature, solution_gor=solution_gor
            )
        )
        if bubble_point_arr is None:
            return self._pt_query("viscosity", pressure, temperature)

        dtype = self.dtype
        pressure_arr = np.atleast_1d(pressure).astype(dtype, copy=False)
        temperature_arr = np.atleast_1d(temperature).astype(dtype, copy=False)
        bubble_point_arr = np.atleast_1d(bubble_point_arr).astype(dtype, copy=False)
        pressure_arr, temperature_arr, bubble_point_arr = np.broadcast_arrays(
            pressure_arr, temperature_arr, bubble_point_arr
        )

        result = np.zeros_like(pressure_arr, dtype=dtype)
        saturated = pressure_arr <= bubble_point_arr
        unsaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self._pt_query(
                "viscosity", pressure_arr[saturated], temperature_arr[saturated]
            )  # type: ignore[index]

        if np.any(unsaturated):
            mu_ob = np.asarray(
                self._pt_query(
                    "viscosity",
                    bubble_point_arr[unsaturated],
                    temperature_arr[unsaturated],
                )
            )
            p_under = pressure_arr[unsaturated]
            pb_under = bubble_point_arr[unsaturated]
            X = 2.6 * (p_under**1.187) * np.exp(-11.513 - 8.98e-5 * p_under)
            X = np.clip(X, 0.0, 5.0, dtype=dtype)
            ratio = np.clip(p_under / pb_under, 1.0, None, dtype=dtype)
            result[unsaturated] = np.clip(
                mu_ob * (ratio**X), mu_ob, mu_ob * 100.0, dtype=dtype
            )

        if "viscosity" in self.clamps:
            min_value, max_value = self.clamps["viscosity"]
            np.clip(result, min_value, max_value, out=result, dtype=dtype)

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result) if result.size == 1 else result,  # type: ignore[attr-defined]
        )

    def viscosity_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂μ/∂P` (cP/psi).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).  Water phase only.
        :returns: `∂μ/∂P` or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "viscosity",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
                derivative=True,
            )
        return self._pt_query("viscosity", pressure, temperature, derivative=True)

    def density(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get fluid density `ρ` (lbm/ft³).

        Returned from the pre-built density table which was constructed using
        the standard formula at `PVTTable` initialisation time:

        - Oil:   `ρo = (ρo,SC + Rs·ρg,SC) / Bo`
        - Gas:   `ρg = (ρg,SC + Rv·ρo,SC) / Bg`  [wet gas]
                 `ρg = ρg,SC / Bg`                 [dry gas]
        - Water: `ρw = ρw,SC / Bw`

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).  Water phase only.
        :returns: Density in lbm/ft³, or `None` if table not present.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "density",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        return self._pt_query("density", pressure, temperature)

    def density_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂ρ/∂P` (lbm/ft³/psi).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).  Water phase only.
        :returns: `∂ρ/∂P` or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "density",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
                derivative=True,
            )
        return self._pt_query("density", pressure, temperature, derivative=True)

    def compressibility(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get fluid compressibility `c` (psi⁻¹).

        Returned from the pre-built compressibility table constructed at
        `PVTTable` initialisation time using:

        - Oil / water: `c = -(1/B) · (∂B/∂P)`
        - Gas:         `cg = 1/P - (1/z) · (∂z/∂P)`

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).  Water phase only.
        :returns: Compressibility in psi⁻¹, or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "compressibility",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        return self._pt_query("compressibility", pressure, temperature)

    def compressibility_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂c/∂P` (psi⁻²).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).  Water phase only.
        :returns: `∂c/∂P` or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "compressibility",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
                derivative=True,
            )
        return self._pt_query("compressibility", pressure, temperature, derivative=True)

    def specific_gravity(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get fluid specific gravity `γ` (dimensionless).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).  Water phase only.
        :returns: Specific gravity or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "specific_gravity",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        return self._pt_query("specific_gravity", pressure, temperature)

    def molecular_weight(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get fluid molecular weight `M` (lbm/lb-mol).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).  Water phase only.
        :returns: Molecular weight or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_query(
                "molecular_weight",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        return self._pt_query("molecular_weight", pressure, temperature)

    def bubble_point_pressure(
        self,
        temperature: TableQuery[NDimension],
        solution_gor: typing.Optional[TableQuery[NDimension]] = None,
        pressure: typing.Optional[TableQuery[NDimension]] = None,
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get bubble-point pressure `bubble_point_arr` (psi).

        **Oil phase:**

        - 1-D table bubble_point_arr(T): pass `temperature` only.
        - 2-D table bubble_point_arr(Rs, T): both `temperature` and `solution_gor` required.

        **Water phase:** Pbw(P, T, S) - `pressure` required; uses
        `default_salinity` when `salinity` is `None`.

        **Gas phase:** returns `None`.

        :param temperature: Temperature (°F).
        :param solution_gor: Solution GOR (scf/STB). Required for 2-D bubble_point_arr table.
        :param pressure: Pressure (psi). Required for water bubble_point_arr table.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :returns: Bubble-point pressure in psi, or `None`.
        """
        if self._phase == FluidPhase.GAS:
            return None

        if self._phase == FluidPhase.WATER:
            if pressure is None:
                raise ValidationError(
                    "Water bubble-point pressure requires the `pressure` argument."
                )
            return self._pts_query(
                "bubble_point_pressure",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )

        interp = self._interpolatants.get("bubble_point_pressure")
        if interp is None:
            return None

        if self._bubble_point_ndim == 1:
            result = interp(temperature)
            return float(result) if np.isscalar(temperature) else result

        if solution_gor is None:
            raise ValidationError(
                "2-D bubble-point table requires the `solution_gor` argument."
            )

        dtype = self.dtype
        solution_gor_arr = np.atleast_1d(solution_gor).astype(dtype, copy=False)
        temperature_arr = np.atleast_1d(temperature).astype(dtype, copy=False)
        if solution_gor_arr.shape != temperature_arr.shape:
            if solution_gor_arr.size == 1:
                solution_gor_arr = np.full_like(temperature_arr, solution_gor_arr[0])
            elif temperature_arr.size == 1:
                temperature_arr = np.full_like(solution_gor_arr, temperature_arr[0])
            else:
                raise ValidationError(
                    "`solution_gor` and `temperature` must have compatible shapes."
                )
        result = (
            interp.ev(solution_gor_arr, temperature_arr)
            if hasattr(interp, "ev")
            else interp(solution_gor_arr, temperature_arr)
        )
        return typing.cast(
            TableResult[NDimension],
            dtype.type(result) if result.size == 1 else result,  # type: ignore[attr-defined]
        )

    def bubble_point_pressure_drs(
        self,
        solution_gor: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂Pb/∂Rs` (psi · STB/scf).

        Only available when the bubble-point table is 2-D (bubble_point_arr(Rs, T)).
        Returns `None` for 1-D bubble_point_arr(T) tables or gas / water phases.

        :param solution_gor: Solution GOR (scf/STB).
        :param temperature: Temperature (°F).
        :returns: `∂Pb/∂Rs` or `None`.
        """
        if self._phase != FluidPhase.OIL or self._bubble_point_ndim != 2:
            return None

        interp = self._derivative_interpolatants.get("bubble_point_pressure_drs")
        if interp is None:
            return None

        dtype = self.dtype
        solution_gor_arr = np.atleast_1d(solution_gor).astype(dtype, copy=False)
        temperature_arr = np.atleast_1d(temperature).astype(dtype, copy=False)
        result = (
            interp.ev(solution_gor_arr, temperature_arr)
            if hasattr(interp, "ev")
            else interp(solution_gor_arr, temperature_arr)
        )
        return typing.cast(
            TableResult[NDimension],
            dtype.type(result) if result.size == 1 else result,  # type: ignore[attr-defined]
        )

    def solution_gas_to_oil_ratio(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        solution_gor: typing.Optional[TableQuery[NDimension]] = None,
        bubble_point_pressure: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get solution gas-to-oil ratio `Rs` (scf/STB).

        **Oil phase only.**  Rs is frozen at Rsb above bubble point.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param solution_gor: Solution GOR (scf/STB).  Required for 2-D bubble_point_arr table.
        :param bubble_point_pressure: Pre-computed bubble_point_arr (psi).
        :returns: Rs in scf/STB, or `None`.
        """
        if self._phase != FluidPhase.OIL:
            return None
        if "solution_gor" not in self._interpolatants:
            return None

        dtype = self.dtype
        bubble_point_arr = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(
                temperature=temperature, solution_gor=solution_gor
            )
        )
        pressure_arr = np.atleast_1d(pressure).astype(dtype, copy=False)
        temperature_arr = np.atleast_1d(temperature).astype(dtype, copy=False)

        if bubble_point_arr is None:
            return self._pt_query("solution_gor", pressure_arr, temperature_arr)

        bubble_point_arr = np.atleast_1d(bubble_point_arr).astype(dtype, copy=False)
        pressure_arr, temperature_arr, bubble_point_arr = np.broadcast_arrays(
            pressure_arr, temperature_arr, bubble_point_arr
        )

        result = np.zeros_like(pressure_arr)
        saturated = pressure_arr <= bubble_point_arr
        unsaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self._pt_query(
                "solution_gor", pressure_arr[saturated], temperature_arr[saturated]
            )  # type: ignore[index]
        if np.any(unsaturated):
            # Rs is constant at Rsb above bubble point
            result[unsaturated] = self._pt_query(  # type: ignore[index]
                "solution_gor",
                bubble_point_arr[unsaturated],
                temperature_arr[unsaturated],
            )

        if "solution_gor" in self.clamps:
            min_value, max_value = self.clamps["solution_gor"]
            np.clip(result, min_value, max_value, out=result, dtype=dtype)

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result) if result.size == 1 else result,  # type: ignore[attr-defined]
        )

    def solution_gas_to_oil_ratio_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂Rs/∂P` (scf/STB/psi). Oil phase only.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :returns: `∂Rs/∂P` or `None`.
        """
        if self._phase != FluidPhase.OIL:
            return None
        return self._pt_query("solution_gor", pressure, temperature, derivative=True)

    def is_saturated(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        solution_gor: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[typing.Union[Boolean, BooleanArray[NDimension]]]:
        """
        Determine whether conditions are saturated (P ≤ bubble_point_arr). Oil phase only.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param solution_gor: Solution GOR (scf/STB). Required for 2-D bubble_point_arr table.
        :returns: Boolean mask (True = saturated), or `None` for gas.
        """
        if self._phase == FluidPhase.GAS:
            return None

        bubble_point_arr = self.bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
        )
        if bubble_point_arr is None:
            return None
        
        pressure_arr = np.atleast_1d(pressure)
        bubble_point_arr = np.atleast_1d(bubble_point_arr)
        result = pressure_arr <= bubble_point_arr
        return typing.cast(
            typing.Union[Boolean, BooleanArray[NDimension]],
            bool(result) if result.size == 1 else result,
        )

    def compressibility_factor(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get gas z-factor `z` (dimensionless). Gas phase only.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :returns: z-factor or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self._pt_query("compressibility_factor", pressure, temperature)

    def compressibility_factor_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂z/∂P` (psi⁻¹).  Gas phase only.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :returns: `∂z/∂P` or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self._pt_query(
            "compressibility_factor", pressure, temperature, derivative=True
        )

    def vaporized_oil_ratio(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        dew_point_pressure: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get vaporised oil ratio `Rv` (STB/Mscf).  Gas / condensate phase only.

        Rv is capped at Rv_sat (the value at dew-point pressure) above the dew
        point, analogous to Rs being capped at Rsb above bubble point for oil.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param dew_point_pressure: Pre-computed dew-point pressure (psi).  When
            provided, skips the internal dew-point lookup.
        :returns: Rv in STB/Mscf, or `None` if table is absent.
        """
        if self._phase != FluidPhase.GAS:
            return None
        if "vaporized_oil_ratio" not in self._interpolatants:
            return None

        dew_point_pressure = (
            dew_point_pressure
            if dew_point_pressure is not None
            else self.dew_point_pressure(temperature=temperature)
        )

        dtype = self.dtype
        pressure_arr = np.atleast_1d(pressure).astype(dtype, copy=False)
        temperature_arr = np.atleast_1d(temperature).astype(dtype, copy=False)

        if dew_point_pressure is None:
            return self._pt_query("vaporized_oil_ratio", pressure_arr, temperature_arr)

        pdew_arr = np.atleast_1d(dew_point_pressure).astype(dtype, copy=False)
        pressure_arr, temperature_arr, pdew_arr = np.broadcast_arrays(
            pressure_arr, temperature_arr, pdew_arr
        )

        result = np.zeros_like(pressure_arr, dtype=dtype)
        above = pressure_arr >= pdew_arr  # above dew point: Rv = Rv_sat (frozen)
        below = ~above

        if np.any(below):
            result[below] = self._pt_query(  # type: ignore[index]
                "vaporized_oil_ratio", pressure_arr[below], temperature_arr[below]
            )
        if np.any(above):
            result[above] = self._pt_query(  # type: ignore[index]
                "vaporized_oil_ratio", pdew_arr[above], temperature_arr[above]
            )

        if "vaporized_oil_ratio" in self.clamps:
            min_value, max_value = self.clamps["vaporized_oil_ratio"]
            np.clip(result, min_value, max_value, out=result, dtype=dtype)

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result) if result.size == 1 else result,  # type: ignore[attr-defined]
        )

    def vaporized_oil_ratio_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂Rv/∂P` (STB/Mscf/psi).  Gas / condensate phase only.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :returns: `∂Rv/∂P` or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self._pt_query(
            "vaporized_oil_ratio", pressure, temperature, derivative=True
        )

    def dew_point_pressure(
        self,
        temperature: TableQuery[NDimension],
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get gas dew-point pressure `Pdew(T)` (psi).  Gas phase only.

        :param temperature: Temperature (°F).
        :returns: Dew-point pressure in psi, or `None` if table is absent.
        """
        if self._phase != FluidPhase.GAS:
            return None

        interp = self._interpolatants.get("dew_point_pressure")
        if interp is None:
            return None

        dtype = self.dtype
        result = interp(temperature)
        return dtype.type(result) if np.isscalar(temperature) else result  # type: ignore[attr-defined]

    def solubility_in_water(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Get gas solubility in water `Rsw` (scf/STB).  Gas phase only.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl); uses default if `None`.
        :returns: Rsw in scf/STB, or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self._pts_query(
            "solubility_in_water",
            pressure,
            temperature,
            self._resolve_salinity(salinity),
        )

    def solubility_in_water_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: typing.Optional[TableQuery[NDimension]] = None,
    ) -> typing.Optional[TableResult[NDimension]]:
        """
        Return `∂Rsw/∂P` (scf/STB/psi). Gas phase only.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl).
        :returns: `∂Rsw/∂P` or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self._pts_query(
            "solubility_in_water",
            pressure,
            temperature,
            self._resolve_salinity(salinity),
            derivative=True,
        )


@typing.final
@attrs.frozen
class PVTTables(StoreSerializable):
    """
    Bundle of phase-specific `PVTTable` instances for a single PVT region.

    Access phase tables via attributes:

    ```python
    pvt_tables.oil.viscosity(p, t)
    pvt_tables.gas.formation_volume_factor(p, t)
    pvt_tables.water.density(p, t, salinity=35_000)
    ```

    Build from a `PVTDataSet` or from individual serialised files:

    ```python
    tables = PVTTables.from_dataset(dataset, interpolation_method="cubic")
    tables = PVTTables.from_files(oil="oil.h5", gas="gas.h5")
    tables = PVTTables.from_deck_file(data_file, temperature=200.0)
    ```
    """

    oil: typing.Optional[PVTTable] = None
    """PVT table for the oil phase."""

    gas: typing.Optional[PVTTable] = None
    """PVT table for the gas phase."""

    water: typing.Optional[PVTTable] = None
    """PVT table for the water phase."""

    @classmethod
    def from_dataset(
        cls,
        dataset: PVTDataSet,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        clamps: typing.Union[
            typing.Literal[False],
            typing.Mapping[FluidPhase, typing.Mapping[str, typing.Tuple[float, float]]],
            None,
        ] = None,
        pvt: typing.Optional[PVT] = None,
        dtype: typing.Optional[npt.DTypeLike] = None,
    ) -> Self:
        """
        Build a `PVTTables` bundle from a `PVTDataSet`.

        :param dataset: Source `PVTDataSet`.
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param validate: Run physical-consistency checks on each phase.
        :param warn_on_extrapolation: Log warnings when queries exceed table bounds.
        :param clamps: Per-phase property clamp overrides. Set to `False` to
            disable all clamping.
        :param pvt: Reference densities for derived table construction
            (`standard_oil_density`, `standard_gas_density`, `standard_water_density`).
        :returns: `PVTTables` ready for simulation.
        """
        base_kwargs: typing.Dict[str, typing.Any] = dict(
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            pvt=pvt,
            dtype=dtype,
        )

        def _clamp_for(phase: FluidPhase) -> typing.Any:
            if isinstance(clamps, Mapping):
                return clamps.get(phase, None)
            return clamps  # False or None applies to all

        oil_table = (
            PVTTable(dataset.oil, clamps=_clamp_for(FluidPhase.OIL), **base_kwargs)
            if dataset.oil is not None
            else None
        )
        gas_table = (
            PVTTable(dataset.gas, clamps=_clamp_for(FluidPhase.GAS), **base_kwargs)
            if dataset.gas is not None
            else None
        )
        water_table = (
            PVTTable(
                dataset.water,
                clamps=_clamp_for(FluidPhase.WATER),
                **base_kwargs,
            )
            if dataset.water is not None
            else None
        )
        return cls(oil=oil_table, gas=gas_table, water=water_table)

    @classmethod
    def from_files(
        cls,
        oil: typing.Optional[typing.Union[PathLike[str], str]] = None,
        gas: typing.Optional[typing.Union[PathLike[str], str]] = None,
        water: typing.Optional[typing.Union[PathLike[str], str]] = None,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        clamps: typing.Union[
            typing.Literal[False],
            typing.Mapping[FluidPhase, typing.Mapping[str, typing.Tuple[float, float]]],
            None,
        ] = None,
        pvt: typing.Optional[PVT] = None,
        dtype: typing.Optional[npt.DTypeLike] = None,
        **load_kwargs: typing.Any,
    ) -> Self:
        """
        Build a `PVTTables` bundle directly from per-phase data files.

        :param oil: Path to serialised oil `PVTData` file.
        :param gas: Path to serialised gas `PVTData` file.
        :param water: Path to serialised water `PVTData` file.
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param validate: Run physical-consistency checks.
        :param warn_on_extrapolation: Log warnings on extrapolation.
        :param clamps: Per-phase clamp overrides.
        :param pvt: Reference densities for derived table construction.
        :returns: `PVTTables` ready for simulation.
        """
        dataset = PVTDataSet.from_files(
            oil=oil,
            gas=gas,
            water=water,
            dtype=dtype,
            **load_kwargs,
        )
        return cls.from_dataset(
            dataset,
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            clamps=clamps,
            pvt=pvt,
            dtype=dtype,
        )

    @classmethod
    def from_deck_file(
        cls,
        deck_file: DeckFile,
        temperature: float,
        pvtnum: int = 1,
        *,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        pvt: typing.Optional[PVT] = None,
        dtype: typing.Optional[npt.DTypeLike] = None,
    ) -> Self:
        """
        Build `PVTTables` for a single PVT region from a parsed `DeckFile`.

        Convenience wrapper around `PVTRegions.from_deck_file` for the
        common single-region case.

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param temperature: Reservoir temperature (°F) - used as the single
            temperature value for deck-loaded isothermal tables.
        :param pvtnum: 1-based PVT region index to extract (default 1).
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param validate: Run physical-consistency checks.
        :param warn_on_extrapolation: Log warnings on extrapolation.
        :param pvt: Reference densities for derived table derivation.
        :returns: `PVTTables` for the specified region.
        """
        from bores.tables.pvt.regions import PVTRegions

        regions = PVTRegions.from_deck_file(
            deck_file=deck_file,
            temperature=temperature,
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            pvt=pvt,
            dtype=dtype,
        )
        return typing.cast(Self, regions.for_region(pvtnum))

    @property
    def dataset(self) -> PVTDataSet:
        """
        Return the underlying raw data as a `PVTDataSet`.

        Useful when you want to persist the raw tabulated data independently
        of interpolator settings.

        :returns: `PVTDataSet` with each non-`None` phase's raw data.
        """
        return PVTDataSet(
            oil=self.oil._data if self.oil is not None else None,
            gas=self.gas._data if self.gas is not None else None,
            water=self.water._data if self.water is not None else None,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `PVTTables` with all phase tables converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PVTTables` in *target* units.
        """
        return self.__class__(
            oil=self.oil.convert(target, table=table) if self.oil is not None else None,
            gas=self.gas.convert(target, table=table) if self.gas is not None else None,
            water=self.water.convert(target, table=table)
            if self.water is not None
            else None,
        )
