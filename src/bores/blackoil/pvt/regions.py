import logging
import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import (  # type: ignore[import-untyped]
    PchipInterpolator,
    interp1d,
)
from typing_extensions import Self

from bores.blackoil.pvt.data import PVTData, PVTDataSet
from bores.blackoil.pvt.static import StaticPVT
from bores.blackoil.pvt.tables import PVTTables
from bores.constants import UnitConversionTable
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.reservoir.temperature import (
    Temperature,
    TemperatureGradient,
    TemperatureSpec,
    TemperatureTable,
)
from bores.serialization.base import Serializable
from bores.serialization.stores import StoreSerializable
from bores.typing import (
    FloatArray,
    FluidPhase,
    InterpolationMethod,
    Number,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = ["PVTRegions", "load_pvt_regions"]


@attrs.frozen(slots=True)
class PVTRegion(Serializable):
    """Collection of PVT tables and static properties for a single Eclipse PVT region."""

    static: StaticPVT
    """Static PVT properties for this region (e.g. DENSITY, VISCOSITY)."""

    tables: PVTTables
    """Dynamic PVT tables for this region (e.g. PVTO, PVTG, PVTW)."""

    unit_system: UnitSystem
    """Unit system of the tables and static properties."""

    def __attrs_post_init__(self) -> None:
        if self.static.unit_system != self.unit_system:
            raise ValidationError(
                f"Static PVT unit system {self.static.unit_system} does not match "
                f"region unit system {self.unit_system}."
            )
        if self.tables.unit_system != self.unit_system:
            raise ValidationError(
                f"Tables unit system {self.tables.unit_system} does not match "
                f"region unit system {self.unit_system}."
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `PVTRegion` with all region tables converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PVTRegion` in *target* units.
        """
        return attrs.evolve(
            self,
            static=self.static.convert(target, table=table),
            tables=self.tables.convert(target, table=table),
            unit_system=self.unit_system,
        )


class PVTRegions(StoreSerializable):
    """
    Multi-region PVT tables keyed by 1-based `PVTNUM` region index.

    Eclipse supports multiple PVT regions via the `PVTNUM` keyword. Each
    cell is assigned a region index and its PVT properties are evaluated from
    the corresponding `PVTTables` instance.

    Use `for_region(pvtnum)` to retrieve the tables and static properties for a given region, and
    `from_deck_file` to construct from a deck.

    Example:

    ```python
    pvt_regions = PVTRegions.from_deck_file(deck_file, temperature=200.0)
    region = pvt_regions.for_region(pvtnum_array[cell_idx])
    bo = region.tables.oil.formation_volume_factor(p, t)
    ```
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        regions: typing.Dict[int, PVTRegion],
    ) -> None:
        """
        Build a `PVTRegions` from a pre-built regions dict.

        :param regions: Mapping from 1-based PVTNUM index to `PVTRegion`.
        :raises ValidationError: If *regions* is empty.
        """
        if not regions:
            raise ValidationError("`regions` must contain at least one entry.")

        # Assert all regions have the same unit system
        unit_systems = {region.unit_system for region in regions.values()}
        if len(unit_systems) > 1:
            raise ValidationError(
                "All PVT regions must share the same unit system. "
                f"Found: {sorted(unit_system.value for unit_system in unit_systems)}."
            )
        self._regions = regions

    def for_region(self, pvtnum: int) -> PVTRegion:
        """
        Return the `PVTRegion` for a given 1-based region index.

        :param pvtnum: 1-based PVT region index.
        :returns: `PVTRegion` for that region.
        :raises KeyError: If the region index does not exist.
        """
        regions = self._regions.get(pvtnum)
        if regions is None:
            available = sorted(self._regions.keys())
            raise KeyError(
                f"PVT region {pvtnum} not found. Available regions: {available}."
            )
        return regions

    @property
    def unit_system(self) -> UnitSystem:
        """Unit system of all region tables and static properties."""
        assert self._regions
        return next(iter(self._regions.values())).unit_system

    @property
    def n_regions(self) -> int:
        """Number of PVT regions."""
        return len(self._regions)

    @classmethod
    def single_region(cls, region: PVTRegion) -> Self:
        """
        Wrap a single `PVTRegion` as region 1.

        Convenience factory for the common single-region case.

        :param region: `PVTRegion` instance.
        :returns: `PVTRegions` with one entry at key 1.
        """
        return cls(regions={1: region})

    @classmethod
    def from_deck_file(
        cls,
        deck_file: DeckFile,
        temperature: typing.Union[Temperature, Number],
        *,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build all PVT regions' tables from a parsed `DeckFile`.

        Detects which Eclipse PVT keywords are present (`PVTO` > `PVDO` > `PVCO` for oil;
        `PVTG` > `PVDG` for gas; `PVTW` for water) and builds one `PVTRegion` per `PVTNUM` region.

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param temperature: Reservoir temperature (°F) used for all regions,
            or a reservoir regional `Temperature` instance.
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param validate: Run physical-consistency checks.
        :param warn_on_extrapolation: Log warnings on extrapolation.
        :returns: `PVTRegions` keyed by 1-based PVTNUM index.
        """
        regions = load_pvt_regions(
            deck_file=deck_file,
            temperature=temperature
            if isinstance(temperature, Temperature)
            else Temperature(temperature, unit_system=UnitSystem.FIELD),
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            dtype=dtype,
        )
        return cls(regions=regions)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `PVTRegions` with all region tables converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PVTRegions` in *target* units.
        """
        return self.__class__(
            regions={
                pvtnum: region.convert(target, table=table)
                for pvtnum, region in self._regions.items()
            }
        )

    def __getitem__(self, key: int) -> PVTRegion:
        return self.for_region(key)

    def __iter__(self) -> typing.Iterator[int]:
        return iter(self._regions)

    def __len__(self) -> int:
        return len(self._regions)

    def __contains__(self, key: object) -> bool:
        return key in self._regions

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "regions": {
                str(pvtnum): region.dump(recurse)
                for pvtnum, region in self._regions.items()
            }
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        regions = {
            int(pvtnum): PVTRegion.load(region_data)
            for pvtnum, region_data in data["regions"].items()
        }
        return cls(regions=regions)


def _min_temperature_points(interpolation_method: InterpolationMethod) -> int:
    """Minimum temperature-axis length required by `PVTTable` for a given method."""
    return 4 if interpolation_method == "cubic" else 2


def _ensure_strictly_increasing(
    values: npt.NDArray, min_points: int, dtype: npt.DTypeLike
) -> npt.NDArray:
    """
    Deduplicate, sort, and pad *values* so the result is strictly increasing
    and has at least *min_points* entries (both required by `PVTTable`).

    Padding interpolates additional points between the existing extremes
    rather than fabricating unrelated ones, so the physical range of the
    axis is preserved.
    """
    values = np.unique(values.astype(dtype, copy=False))
    if len(values) == 1:
        # Degenerate (isothermal) case - bracket it with a tiny symmetric span.
        min_value, max_value = values[0] - 1.0, values[0] + 1.0
        values = np.linspace(min_value, max_value, max(min_points, 2), dtype=dtype)
    elif len(values) < min_points:
        # Preserve the real knots exactly; fill in between them to satisfy
        # the interpolator's minimum-point requirement.
        filler = np.linspace(values[0], values[-1], min_points, dtype=dtype)
        values = np.unique(np.concatenate([values, filler])).astype(dtype, copy=False)
    return values


def _generate_temperature_axis(
    temperature: TemperatureSpec,
    dtype: npt.DTypeLike,
    *,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: typing.Optional[typing.Tuple[Number, Number]] = None,
    n_points: typing.Optional[int] = None,
    max_points: int = 25,
) -> npt.NDArray:
    """
    Build a temperature axis appropriate to *temperature*'s spec type.

    Eclipse PVT tables are 2-D `(n_p, n_t)`. `PVTTable` requires the `n_t`
    axis to be strictly increasing, with at least 2 points for `"linear"`
    interpolation or 4 for `"cubic"`. This builds the smallest axis that
    faithfully represents each spec type while satisfying that constraint:

    - `Number` (scalar / isothermal): a tight symmetric bracket around the
      value. All property columns are broadcast identically across it
      (see `_broadcast_to_2d`), so extra points cost nothing and change
      nothing physically - they only exist to satisfy `PVTTable`.
    - `TemperatureGradient`: samples `gradient.at_depth(...)` across
      *depth_range* (the actual depth extent of the region's cells, when
      known) so the axis truly brackets the temperatures the region will
      see. Without a known depth extent, falls back to a minimal bracket
      around the reference temperature and warns, since the gradient's
      true range can't be determined.
    - `TemperatureTable`: uses the table's own (sorted, unique) temperature
      knots directly. This is the most faithful axis possible - it
      reproduces the table's actual breakpoints with zero extra
      interpolation error, rather than resampling onto an arbitrary grid.
      Very dense tables are downsampled to *max_points* (endpoints
      preserved) to keep the resulting PVT table size bounded.

    :param temperature: Reservoir temperature spec for this region.
    :param dtype: Output dtype.
    :param interpolation_method: `"linear"` or `"cubic"` - determines the
        minimum number of axis points required downstream.
    :param depth_range: Optional `(min_depth, max_depth)` of the cells this
        region covers, in the same length units as `temperature`. Only
        used for `TemperatureGradient`; ignored otherwise.
    :param n_points: Override the number of points to sample (gradient) or
        the target density (ignored for scalar/table, which use their own
        natural sizing). Defaults to a method-appropriate value.
    :param max_points: Cap on axis length for `TemperatureTable` downsampling.
    :returns: 1-D strictly increasing array, dtype *dtype*.
    """
    dtype = np.dtype(dtype)
    min_points = _min_temperature_points(interpolation_method)

    if isinstance(temperature, TemperatureGradient):
        count = n_points or max(min_points, 8)
        if depth_range is not None:
            min_value, max_value = sorted((depth_range[0], depth_range[1]))
            if max_value <= min_value:
                max_value = min_value + 1.0
            depths = np.linspace(min_value, max_value, count, dtype=dtype)
        else:
            warnings.warn(
                "TemperatureGradient without a `depth_range`: falling back to a "
                "minimal bracket around the reference temperature. Pass the "
                "region's actual cell-depth extent for a physically accurate "
                "temperature axis.",
                UserWarning,
                stacklevel=3,
            )
            ref = temperature.reference_depth
            depths = np.linspace(ref - 1.0, ref + 1.0, max(min_points, 2), dtype=dtype)
        temperatures = temperature.at_depth(depths).astype(dtype, copy=False)  # type: ignore[arg-type]
        return _ensure_strictly_increasing(temperatures, min_points, dtype)

    if isinstance(temperature, TemperatureTable):
        knots = np.unique(temperature.temperatures.astype(dtype, copy=False))
        if len(knots) > max_points:
            # Downsample onto an evenly spaced grid spanning the same range;
            # endpoints (and thus the full physical range) are preserved.
            knots = np.linspace(knots[0], knots[-1], max_points, dtype=dtype)
        return _ensure_strictly_increasing(knots, min_points, dtype)

    # Scalar Number - unchanged behavior, generalized to respect min_points.
    count = n_points or min_points
    return np.linspace(temperature - 1.0, temperature + 1.0, max(count, 2), dtype=dtype)


def _broadcast_to_2d(values_1d: npt.NDArray, n_t: int = 2) -> npt.NDArray:
    """
    Broadcast a 1-D array of shape `(n_p,)` to 2-D shape `(n_p, n_t)`.

    Used to satisfy the `(n_p, n_t)` shape requirement of `PVTTable`
    when deck data contains only a pressure axis (isothermal tables).

    :param values_1d: 1-D array of shape `(n_p,)`.
    :param n_t: Number of temperature knots (usually 2 for degenerate axis).
    :returns: 2-D array of shape `(n_p, n_t)`.
    """
    return np.tile(values_1d[:, np.newaxis], (1, n_t)).astype(
        values_1d.dtype, copy=False
    )


def _build_oil_data_from_pvto(
    pvto_records: typing.List[typing.Dict[str, typing.Any]],
    density_record: typing.Optional[typing.Dict[str, Number]],
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: typing.Optional[typing.Tuple[Number, Number]] = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build oil `PVTData` from a parsed `PVTO` record set.

    `PVTO` format: Rs is the outer key; each Rs group contains rows of
    `(pressure, bo, viscosity)` ordered by ascending pressure. The first
    row in each Rs group is the saturated row at bubble-point pressure;
    subsequent rows at higher pressure are the undersaturated branch.

    The builder:

    1. Extracts the saturated branch `(Pb(Rs), Bo_sat(Rs), μo_sat(Rs))`.
    2. Builds a regular pressure grid from the union of all bubble-point
       pressures extended to cover the maximum undersaturated pressure seen.
    3. At each pressure, interpolates Bo and μo from the appropriate Rs
       group's undersaturated branch (or uses the saturated value when
       P ≤ Pb).
    4. Derives the density table using `ρo = (ρo,SC + Rs·ρg,SC) / Bo`
       when reference densities are available.
    5. Derives compressibility from `co = -(1/Bo)·(∂Bo/∂P)`.

    :param pvto_records: List of row dicts from the parsed `PVTO` keyword.
        Each dict has keys `"rs"`, `"pressure"`, `"bo"`, `"viscosity"`.
    :param density_record: `DENSITY` record dict with `"oil"` and `"gas"` keys
        (lbm/ft³ at standard conditions).
    :param temperature: Reservoir temperature.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the oil phase.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    temperatures = _generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)

    # Group records by Rs value
    solution_gor_to_rows: typing.Dict[float, typing.List[typing.Dict]] = {}
    for row in pvto_records:
        solution_gor = row["rs"]
        solution_gor_to_rows.setdefault(solution_gor, []).append(row)

    if len(solution_gor_to_rows) < 2:
        raise ValidationError(
            f"PVTO table requires at least 2 Rs values; got {len(solution_gor_to_rows)}."
        )

    solution_gor_values = np.array(sorted(solution_gor_to_rows.keys()), dtype=dtype)
    n_rs = len(solution_gor_values)

    # Saturated branch: lowest-pressure row in each Rs group = bubble point
    bubble_point_pressure_values = np.empty(n_rs, dtype=dtype)
    saturated_oil_fvf = np.empty(n_rs, dtype=dtype)
    saturated_oil_viscosity = np.empty(n_rs, dtype=dtype)

    for i, solution_gor in enumerate(solution_gor_values):
        rows = sorted(solution_gor_to_rows[solution_gor], key=lambda r: r["pressure"])
        saturated_row = rows[0]
        bubble_point_pressure_values[i] = saturated_row["pressure"]
        saturated_oil_fvf[i] = saturated_row["bo"]
        saturated_oil_viscosity[i] = saturated_row["viscosity"]

    # Pressure grid: bubble-point pressures + extension to max undersaturated pressure.
    # We do not merge all undersaturated rows into one flat grid - that would mix
    # physically distinct branches. Instead we extend monotonically from the
    # highest Pb to cover the undersaturated range.
    max_pressure = max(
        float(row["pressure"]) for rows in solution_gor_to_rows.values() for row in rows
    )
    extension = np.linspace(
        float(bubble_point_pressure_values[-1]),
        max_pressure,
        max(10, n_rs),
        dtype=dtype,
    )
    pressures = np.unique(
        np.concatenate([bubble_point_pressure_values, extension])
    ).astype(dtype)
    n_p = len(pressures)

    # Per-Rs interpolators for the full (saturated + undersaturated) branch
    oil_fvf_interps: typing.List[interp1d] = []
    oil_viscosity_interps: typing.List[interp1d] = []

    for solution_gor in solution_gor_values:
        rows = sorted(solution_gor_to_rows[solution_gor], key=lambda r: r["pressure"])
        pressure_arr = np.array([r["pressure"] for r in rows], dtype=dtype)
        oil_fvf_arr = np.array([r["bo"] for r in rows], dtype=dtype)
        oil_viscosity_arr = np.array([r["viscosity"] for r in rows], dtype=dtype)
        oil_fvf_interps.append(
            interp1d(
                pressure_arr,
                oil_fvf_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(oil_fvf_arr[0], oil_fvf_arr[-1]),
            )
        )
        oil_viscosity_interps.append(
            interp1d(
                pressure_arr,
                oil_viscosity_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(oil_viscosity_arr[0], oil_viscosity_arr[-1]),
            )
        )

    # Rs(P) on the saturated branch via inverse of Pb(Rs).
    # Pb must be monotonically increasing with Rs for a well-formed PVTO table.
    # If not strictly monotone, we clip safely.
    if not np.all(np.diff(bubble_point_pressure_values) > 0):
        warnings.warn(
            "PVTO bubble-point pressures are not strictly increasing with Rs. "
            "Rs(P) inversion may be inaccurate for some pressure values.",
            UserWarning,
            stacklevel=4,
        )

    solution_gor_of_pressure = interp1d(
        bubble_point_pressure_values,
        solution_gor_values,
        kind="linear",
        bounds_error=False,
        fill_value=(solution_gor_values[0], solution_gor_values[-1]),
    )

    oil_fvf_2d = np.empty((n_p, n_t), dtype=dtype)
    oil_viscosity_2d = np.empty((n_p, n_t), dtype=dtype)
    solution_gor_2d = np.empty((n_p, n_t), dtype=dtype)

    for i, pressure in enumerate(pressures):
        # Determine which Rs group governs at this pressure:
        # find the Rs group whose Pb is closest to (and ≤) this pressure.
        rs_idx = (
            int(np.searchsorted(bubble_point_pressure_values, pressure, side="right"))
            - 1
        )
        rs_idx = int(np.clip(rs_idx, 0, n_rs - 1))

        # Rs at this pressure on the saturated envelope
        rs_at_pressure = solution_gor_of_pressure(pressure)
        solution_gor_2d[i, :] = rs_at_pressure

        oil_fvf = oil_fvf_interps[rs_idx](pressure)
        oil_viscosity = oil_viscosity_interps[rs_idx](pressure)

        oil_fvf_2d[i, :] = oil_fvf
        oil_viscosity_2d[i, :] = oil_viscosity

    # Bubble-point table: Pb(Rs, T) - 2-D, one Pb per Rs per temperature
    bubble_point_pressure_2d = np.tile(
        bubble_point_pressure_values[:, np.newaxis], (1, n_t)
    ).astype(dtype, copy=False)

    # Resolve reference densities: pvt takes precedence over DENSITY record
    stock_tank_oil_density: typing.Optional[Number] = None
    stock_tank_gas_density: typing.Optional[Number] = None
    if density_record is not None:
        stock_tank_oil_density = density_record.get("oil")
        stock_tank_gas_density = density_record.get("gas")

    # Density: ρo = (ρo,SC + Rs·ρg,SC) / Bo
    oil_density_2d: typing.Optional[npt.NDArray] = None
    if stock_tank_oil_density is not None and stock_tank_gas_density is not None:
        oil_density_2d = (
            (stock_tank_oil_density + solution_gor_2d * stock_tank_gas_density)
            / oil_fvf_2d
        ).astype(dtype, copy=False)

    # Compressibility: co = -(1/Bo)·(∂Bo/∂P) via PCHIP derivative
    oil_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbo_dp = PchipInterpolator(pressures, oil_fvf_2d[:, j]).derivative(1)(pressures)
        oil_compressibility = -(1.0 / oil_fvf_2d[:, j]) * dbo_dp
        # Compressibility must be non-negative; clamp to physical range
        oil_compressibility_2d[:, j] = np.clip(
            oil_compressibility, 0.0, 1e-1, dtype=dtype
        )

    return PVTData(
        phase=FluidPhase.OIL,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        bubble_point_pressures=typing.cast(
            FloatArray[TwoDimensions], bubble_point_pressure_2d
        ),
        solution_gas_to_oil_ratios=typing.cast(
            FloatArray[OneDimension], solution_gor_values
        ),
        formation_volume_factor_table=typing.cast(
            FloatArray[TwoDimensions], oil_fvf_2d
        ),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], oil_viscosity_2d),
        solution_gor_table=typing.cast(FloatArray[TwoDimensions], solution_gor_2d),
        density_table=typing.cast(FloatArray[TwoDimensions], oil_density_2d)
        if oil_density_2d is not None
        else None,
        compressibility_table=typing.cast(
            FloatArray[TwoDimensions], oil_compressibility_2d
        ),
        dtype=dtype,
        unit_system=unit_system,
    )


def _build_oil_data_from_pvdo(
    pvdo_records: typing.List[typing.Dict[str, typing.Any]],
    density_record: typing.Optional[typing.Dict[str, Number]],
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: typing.Optional[typing.Tuple[Number, Number]] = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build dead-oil `PVTData` from a parsed `PVDO` record set.

    `PVDO` format: single table of `(pressure, bo, viscosity)` rows for
    dead oil (Rs = 0 everywhere). No bubble-point switching is required
    since dead oil has no dissolved gas.

    :param pvdo_records: List of row dicts with keys `"pressure"`, `"bo"`,
        `"viscosity"`.
    :param density_record: `DENSITY` record; `"oil"` key used for ρo,SC.
    :param temperature: Reservoir temperature.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the oil phase.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    temperatures = _generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)

    rows = sorted(pvdo_records, key=lambda r: r["pressure"])
    if len(rows) < 2:
        raise ValidationError(f"PVDO table requires at least 2 rows; got {len(rows)}.")

    pressures = np.array([r["pressure"] for r in rows], dtype=dtype)
    oil_fvf_1d = np.array([r["bo"] for r in rows], dtype=dtype)
    oil_viscosity_1d = np.array([r["viscosity"] for r in rows], dtype=dtype)
    n_p = len(pressures)

    if not np.all(np.diff(pressures) > 0):
        raise ValidationError("PVDO pressures must be strictly increasing.")
    if np.any(oil_fvf_1d <= 0):
        raise ValidationError("PVDO Bo values must be positive.")
    if np.any(oil_viscosity_1d <= 0):
        raise ValidationError("PVDO viscosity values must be positive.")

    oil_fvf_2d = _broadcast_to_2d(oil_fvf_1d, n_t)
    oil_viscosity_2d = _broadcast_to_2d(oil_viscosity_1d, n_t)
    # Dead oil: Rs = 0 everywhere
    solution_gor_2d = np.zeros((n_p, n_t), dtype=dtype)

    stock_tank_oil_density: typing.Optional[Number] = None
    if density_record is not None:
        stock_tank_oil_density = density_record.get("oil")

    # Density: ρo = ρo,SC / Bo (dead oil - Rs = 0)
    oil_density_2d: typing.Optional[npt.NDArray] = None
    if stock_tank_oil_density is not None:
        oil_density_2d = (stock_tank_oil_density / oil_fvf_2d).astype(dtype, copy=False)

    # Compressibility: co = -(1/Bo)·(∂Bo/∂P)
    oil_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbo_dp = PchipInterpolator(pressures, oil_fvf_2d[:, j]).derivative(1)(pressures)
        oil_compressibility_2d[:, j] = np.clip(
            -(1.0 / oil_fvf_2d[:, j]) * dbo_dp, 0.0, 1e-1, dtype=dtype
        )

    return PVTData(
        phase=FluidPhase.OIL,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        formation_volume_factor_table=typing.cast(
            FloatArray[TwoDimensions], oil_fvf_2d
        ),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], oil_viscosity_2d),
        solution_gor_table=typing.cast(FloatArray[TwoDimensions], solution_gor_2d),
        density_table=typing.cast(FloatArray[TwoDimensions], oil_density_2d)
        if oil_density_2d is not None
        else None,
        compressibility_table=typing.cast(
            FloatArray[TwoDimensions], oil_compressibility_2d
        ),
        dtype=dtype,
        unit_system=unit_system,
    )


def _build_gas_data_from_pvdg(
    pvdg_records: typing.List[typing.Dict[str, typing.Any]],
    density_record: typing.Optional[typing.Dict[str, Number]],
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: typing.Optional[typing.Tuple[Number, Number]] = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build dry-gas `PVTData` from a parsed `PVDG` record set.

    `PVDG` format: single table of `(pressure, bg, viscosity)` rows.
    Eclipse stores Bg in rb/Mscf; this builder converts to ft³/SCF:
    `Bg_ft3_scf = Bg_rb_Mscf x 5.615 / 1000`.

    :param pvdg_records: List of row dicts with keys `"pressure"`, `"bg"`,
        `"viscosity"`.
    :param density_record: `DENSITY` record; `"gas"` key used for ρg,SC.
    :param temperature: Reservoir temperature.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the gas phase.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    temperatures = _generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)

    rows = sorted(pvdg_records, key=lambda r: r["pressure"])
    if len(rows) < 2:
        raise ValidationError(f"PVDG table requires at least 2 rows; got {len(rows)}.")

    pressures = np.array([r["pressure"] for r in rows], dtype=dtype)
    # Convert rb/Mscf → ft³/SCF
    gas_fvf_1d = np.array([r["bg"] * 5.615 / 1000.0 for r in rows], dtype=dtype)
    gas_viscosity_1d = np.array([r["viscosity"] for r in rows], dtype=dtype)

    if not np.all(np.diff(pressures) > 0):
        raise ValidationError("PVDG pressures must be strictly increasing.")
    if np.any(gas_fvf_1d <= 0):
        raise ValidationError("PVDG Bg values must be positive.")
    if np.any(gas_viscosity_1d <= 0):
        raise ValidationError("PVDG viscosity values must be positive.")

    n_p = len(pressures)
    gas_fvf_2d = _broadcast_to_2d(gas_fvf_1d, n_t)
    gas_viscosity_2d = _broadcast_to_2d(gas_viscosity_1d, n_t)

    stock_tank_gas_density: typing.Optional[Number] = None
    if density_record is not None:
        stock_tank_gas_density = density_record.get("gas")

    # Density: ρg = ρg,SC / Bg
    gas_density_2d: typing.Optional[npt.NDArray] = None
    if stock_tank_gas_density is not None:
        gas_density_2d = (stock_tank_gas_density / gas_fvf_2d).astype(dtype, copy=False)

    # Compressibility: cg ≈ -(1/Bg)·(∂Bg/∂P)
    gas_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbg_dp = PchipInterpolator(pressures, gas_fvf_2d[:, j]).derivative(1)(pressures)
        gas_compressibility_2d[:, j] = np.clip(
            -(1.0 / gas_fvf_2d[:, j]) * dbg_dp, 0.0, 1e-1, dtype=dtype
        )

    return PVTData(
        phase=FluidPhase.GAS,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        formation_volume_factor_table=typing.cast(
            FloatArray[TwoDimensions], gas_fvf_2d
        ),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], gas_viscosity_2d),
        density_table=typing.cast(FloatArray[TwoDimensions], gas_density_2d)
        if gas_density_2d is not None
        else None,
        compressibility_table=typing.cast(
            FloatArray[TwoDimensions], gas_compressibility_2d
        ),
        dtype=dtype,
        unit_system=unit_system,
    )


def _build_gas_data_from_pvtg(
    pvtg_records: typing.List[typing.Dict[str, typing.Any]],
    density_record: typing.Optional[typing.Dict[str, Number]],
    unit_system: UnitSystem,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build wet-gas `PVTData` from a parsed `PVTG` record set.

    `PVTG` format: pressure is the outer key; each pressure group contains
    rows of `(rv, bg, viscosity)` ordered by ascending Rv. The first row
    in each group is the dry-gas value (Rv = 0).

    The temperature axis of the returned `PVTData` carries Rv values rather
    than temperatures, matching the wet-gas table convention used by `PVTTable`.
    All Rv values from all pressure groups are unioned to form a common Rv grid;
    missing values at a given pressure are linearly interpolated from the
    group's own rows.

    :param pvtg_records: List of row dicts with keys `"pressure"`, `"rv"`,
        `"bg"`, `"viscosity"`.
    :param density_record: `DENSITY` record; `"gas"` and `"oil"` keys used.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the gas phase with Rv as the second (temperature) axis.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())

    pressure_to_rows: typing.Dict[float, typing.List[typing.Dict]] = {}
    for row in pvtg_records:
        pressure_to_rows.setdefault(float(row["pressure"]), []).append(row)

    if len(pressure_to_rows) < 2:
        raise ValidationError(
            f"PVTG table requires at least 2 pressure values; got {len(pressure_to_rows)}."
        )

    pressure_values = np.array(sorted(pressure_to_rows.keys()), dtype=dtype)
    n_p = len(pressure_values)

    if not np.all(np.diff(pressure_values) > 0):
        raise ValidationError("PVTG pressures must be strictly increasing.")

    # Union of all Rv values across all pressure groups → common Rv grid
    all_rv = sorted({
        float(row["rv"]) for rows in pressure_to_rows.values() for row in rows
    })
    if len(all_rv) < 1:
        raise ValidationError("PVTG table contains no Rv values.")

    rv_values = np.array(all_rv, dtype=dtype)
    n_rv = len(rv_values)

    gas_fvf_2d = np.empty((n_p, n_rv), dtype=dtype)
    gas_viscosity_2d = np.empty((n_p, n_rv), dtype=dtype)

    for i, pressure in enumerate(pressure_values):
        rows = sorted(pressure_to_rows[pressure], key=lambda r: r["rv"])
        rv_arr = np.array([r["rv"] for r in rows], dtype=dtype)
        # Convert rb/Mscf → ft³/SCF
        gas_fvf_arr = np.array([r["bg"] * 5.615 / 1000.0 for r in rows], dtype=dtype)
        gas_viscosity_arr = np.array([r["viscosity"] for r in rows], dtype=dtype)

        if np.any(gas_fvf_arr <= 0):
            raise ValidationError(
                f"PVTG Bg values must be positive at pressure {pressure} psi."
            )
        if np.any(gas_viscosity_arr <= 0):
            raise ValidationError(
                f"PVTG viscosity values must be positive at pressure {pressure} psi."
            )

        # If this pressure group has only one Rv point, broadcast it
        if len(rv_arr) == 1:
            gas_fvf_2d[i, :] = gas_fvf_arr[0]
            gas_viscosity_2d[i, :] = gas_viscosity_arr[0]
        else:
            gas_fvf_2d[i, :] = interp1d(
                rv_arr,
                gas_fvf_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(gas_fvf_arr[0], gas_fvf_arr[-1]),
            )(rv_values)
            gas_viscosity_2d[i, :] = interp1d(
                rv_arr,
                gas_viscosity_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(gas_viscosity_arr[0], gas_viscosity_arr[-1]),
            )(rv_values)

    # Resolve reference densities
    stock_tank_gas_density: typing.Optional[Number] = None
    stock_tank_oil_density: typing.Optional[Number] = None

    if density_record is not None:
        stock_tank_gas_density = density_record.get("gas")
        stock_tank_oil_density = density_record.get("oil")

    # Density: ρg = (ρg,SC + Rv·ρo,SC) / Bg  [wet gas]
    #          ρg = ρg,SC / Bg                  [dry gas, Rv = 0 column]
    gas_density_2d: typing.Optional[npt.NDArray] = None
    if stock_tank_gas_density is not None:
        rv_grid = np.tile(rv_values[np.newaxis, :], (n_p, 1))
        if stock_tank_oil_density is not None:
            gas_density_2d = (
                (stock_tank_gas_density + rv_grid * stock_tank_oil_density) / gas_fvf_2d
            ).astype(dtype, copy=False)
        else:
            gas_density_2d = (stock_tank_gas_density / gas_fvf_2d).astype(
                dtype, copy=False
            )

    # Compressibility: cg ≈ -(1/Bg)·(∂Bg/∂P) along each Rv column
    gas_compressibility_2d = np.empty((n_p, n_rv), dtype=dtype)
    for j in range(n_rv):
        dbg_dp = PchipInterpolator(pressure_values, gas_fvf_2d[:, j]).derivative(1)(
            pressure_values
        )
        gas_compressibility_2d[:, j] = np.clip(
            -(1.0 / gas_fvf_2d[:, j]) * dbg_dp, 0.0, 1e-1, dtype=dtype
        )

    # Rv table: shape (n_p, n_rv) - same Rv values at every pressure
    vaporized_oil_ratio_table = np.tile(rv_values[np.newaxis, :], (n_p, 1)).astype(
        dtype, copy=False
    )
    return PVTData(
        phase=FluidPhase.GAS,
        pressures=typing.cast(FloatArray[OneDimension], pressure_values),
        # Rv axis stored here for wet-gas table - PVTTable is aware of this convention
        temperatures=typing.cast(FloatArray[OneDimension], rv_values),
        formation_volume_factor_table=typing.cast(
            FloatArray[TwoDimensions], gas_fvf_2d
        ),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], gas_viscosity_2d),
        vaporized_oil_ratio_table=typing.cast(
            FloatArray[TwoDimensions], vaporized_oil_ratio_table
        ),
        density_table=typing.cast(FloatArray[TwoDimensions], gas_density_2d)
        if gas_density_2d is not None
        else None,
        compressibility_table=typing.cast(
            FloatArray[TwoDimensions], gas_compressibility_2d
        ),
        dtype=dtype,
        unit_system=unit_system,
    )


def _build_water_data_from_pvtw(
    pvtw_record: typing.Dict[str, Number],
    density_record: typing.Optional[typing.Dict[str, Number]],
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    salinity: Number = 0.0,
    n_pressure_points: int = 50,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: typing.Optional[typing.Tuple[Number, Number]] = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build water `PVTData` from a `PVTW` analytical record.

    `PVTW` provides four scalars per region - reference pressure, Bw, cw,
    μw, and optionally cv (viscosibility). Bw(P) and μw(P) are evaluated
    analytically on a pressure grid and stored as tables so all subsequent
    lookups are interpolator calls.

    The exponential models used are:

    - `Bw(P) = Bw_ref · exp(-cw · (P - P_ref))`
    - `μw(P) = μw_ref · exp(-cv · (P - P_ref))`

    The pressure grid spans `[max(14.696, P_ref/10), P_ref x 10]` so that
    the reference pressure always sits comfortably within the table bounds.

    :param pvtw_record: Dict with keys `"p_ref"`, `"bw"`, `"cw"`,
        `"viscosity"`, and optionally `"cv"` (default 0).
    :param density_record: `DENSITY` record; `"water"` key used for ρw,SC.
    :param temperature: Reservoir temperature.
    :param salinity: Water salinity (ppm NaCl).
    :param n_pressure_points: Points in the synthetic pressure grid.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the water phase.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    temperatures = _generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)
    salinities = np.array([salinity], dtype=dtype)

    reference_pressure = float(pvtw_record["p_ref"])
    reference_water_fvf = float(pvtw_record["bw"])
    water_compressibility = float(pvtw_record["cw"])
    reference_water_viscosity = float(pvtw_record["viscosity"])
    water_viscosibility = float(pvtw_record.get("cv", 0.0))

    if reference_water_fvf <= 0:
        raise ValidationError("PVTW Bw must be positive.")
    if water_compressibility < 0:
        raise ValidationError("PVTW cw (compressibility) must be non-negative.")
    if reference_water_viscosity <= 0:
        raise ValidationError("PVTW viscosity must be positive.")

    min_pressure = max(0.0, reference_pressure / 10.0)
    max_pressure = reference_pressure * 10.0
    pressures = np.linspace(min_pressure, max_pressure, n_pressure_points, dtype=dtype)
    n_p = len(pressures)

    delta_p = pressures - reference_pressure
    # True Exponential approach (May give negatives by more accurate)
    # water_fvf_1d = (
    #     reference_water_fvf * np.exp(-water_compressibility * delta_p)
    # ).astype(dtype, copy=False)

    # Taylor's exponential approximation (more stable). Used by Eclipse.
    x = water_compressibility * delta_p
    water_fvf_1d = (reference_water_fvf / (1.0 + x + 0.5 * x * x)).astype(
        dtype, copy=False
    )

    # True Exponential approach (May give negatives by more accurate)
    # water_viscosity_1d = (
    #     reference_water_viscosity * np.exp(-water_viscosibility * delta_p)
    # ).astype(dtype, copy=False)

    # Taylor's exponential approximation (more stable). Used by Eclipse.
    y = water_viscosibility * delta_p
    water_viscosity_1d = (reference_water_viscosity / (1.0 + y + 0.5 * y * y)).astype(
        dtype, copy=False
    )

    water_fvf_2d = _broadcast_to_2d(water_fvf_1d, n_t)
    water_viscosity_2d = _broadcast_to_2d(water_viscosity_1d, n_t)
    # Expand to 3-D: (n_p, n_t, n_s=1)
    water_fvf_3d = water_fvf_2d[:, :, np.newaxis].astype(dtype, copy=False)
    water_viscosity_3d = water_viscosity_2d[:, :, np.newaxis].astype(dtype, copy=False)

    stock_tank_water_density: typing.Optional[Number] = None
    if density_record is not None:
        stock_tank_water_density = density_record.get("water")

    # Density: ρw = ρw,SC / Bw
    water_density_3d: typing.Optional[npt.NDArray] = None
    if stock_tank_water_density is not None:
        water_density_3d = (stock_tank_water_density / water_fvf_3d).astype(
            dtype, copy=False
        )

    # Compressibility: cw is constant for this model - store as a uniform table
    # so the lookup API is consistent with oil and gas phases
    water_compressibility_3d = np.full(
        (n_p, n_t, 1), water_compressibility, dtype=dtype
    )

    return PVTData(
        phase=FluidPhase.WATER,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        salinities=typing.cast(FloatArray[OneDimension], salinities),
        formation_volume_factor_table=typing.cast(
            FloatArray[ThreeDimensions], water_fvf_3d
        ),
        viscosity_table=typing.cast(FloatArray[ThreeDimensions], water_viscosity_3d),
        density_table=typing.cast(FloatArray[ThreeDimensions], water_density_3d)
        if water_density_3d is not None
        else None,
        compressibility_table=typing.cast(
            FloatArray[ThreeDimensions], water_compressibility_3d
        ),
        gas_free_water_fvf_table=typing.cast(FloatArray[TwoDimensions], water_fvf_2d),
        dtype=dtype,
        unit_system=unit_system,
    )


def load_pvt_regions(
    deck_file: DeckFile,
    temperature: Temperature,
    *,
    interpolation_method: InterpolationMethod = "linear",
    validate: bool = True,
    warn_on_extrapolation: bool = False,
    dtype: npt.DTypeLike = None,
) -> typing.Dict[int, PVTRegion]:
    """
    Build a `PVTRegions` object from a parsed `DeckFile`.

    Detects which Eclipse PVT keywords are present and builds one
    `PVTRegion` per `PVTNUM` region:

    - Oil: `PVTO` (live oil, preferred) -> `PVDO` (dead oil) -> `PVCO`
      (analytical; treated as dead oil with constant Bo).
    - Gas: `PVTG` (wet gas, preferred) -> `PVDG` (dry gas).
    - Water: `PVTW` (always analytical; converted to a table internally).

    `DENSITY` records supply the stock-tank reference densities used to
    derive density tables. When `pvt` is also provided it
    takes precedence.

    :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
    :param temperature: `Temperature` instance.
    :param interpolation_method: `"linear"` (default) or `"cubic"`.
    :param validate: Run physical-consistency checks.
    :param warn_on_extrapolation: Log warnings when queries exceed table bounds.
    :returns: Mapping of 1-based PVTNUM region index to corresponding `PVTRegion`.
    :raises ValidationError: If no recognisable PVT keyword is found.
    """
    # Retrieve deck records (each is a list-of-lists: outer = regions)
    pvto_all: typing.Optional[typing.List] = deck_file.get("PVTO")
    pvdo_all: typing.Optional[typing.List] = deck_file.get("PVDO")
    pvco_all: typing.Optional[typing.List] = deck_file.get("PVCO")
    pvtg_all: typing.Optional[typing.List] = deck_file.get("PVTG")
    pvdg_all: typing.Optional[typing.List] = deck_file.get("PVDG")
    pvtw_all: typing.Optional[typing.List] = deck_file.get("PVTW")
    density_all: typing.Optional[typing.List] = deck_file.get("DENSITY")

    if pvto_all is None and pvdo_all is None and pvco_all is None:
        raise ValidationError(
            "No oil PVT keyword found in DeckFile. Expected one of: `PVTO`, `PVDO`, `PVCO`."
        )

    # Number of regions is the maximum length across all keyword lists
    n_regions = max(
        len(x)
        for x in [pvto_all, pvdo_all, pvco_all, pvtg_all, pvdg_all, pvtw_all]
        if x is not None
    )
    unit_system = deck_file.unit_system
    table_kwargs: typing.Dict[str, typing.Any] = dict(
        interpolation_method=interpolation_method,
        validate=validate,
        warn_on_extrapolation=warn_on_extrapolation,
        dtype=dtype,
    )
    regions: typing.Dict[int, PVTRegion] = {}
    if unit_system != temperature.unit_system:
        temperature = temperature.convert(unit_system)

    for region_idx in range(n_regions):
        pvtnum = region_idx + 1  # 1-based

        # Density record for this region
        density_record: typing.Optional[typing.Dict[str, Number]] = None
        if density_all is not None and region_idx < len(density_all):
            # Each DENSITY region entry is a list containing one row dict
            region_rows = density_all[region_idx]
            if region_rows:
                density_record = region_rows[0]

        # Oil Phase
        oil_data: typing.Optional[PVTData] = None
        if pvto_all is not None and region_idx < len(pvto_all):
            oil_data = _build_oil_data_from_pvto(
                pvto_records=pvto_all[region_idx],
                density_record=density_record,
                temperature=temperature.for_region(pvtnum),
                unit_system=unit_system,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )
        elif pvdo_all is not None and region_idx < len(pvdo_all):
            oil_data = _build_oil_data_from_pvdo(
                pvdo_records=pvdo_all[region_idx],
                density_record=density_record,
                temperature=temperature.for_region(pvtnum),
                unit_system=unit_system,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )
        elif pvco_all is not None and region_idx < len(pvco_all):
            # PVCO: single-record analytical model - treat as a two-point PVDO
            pvco_record = pvco_all[region_idx]
            if pvco_record:
                record = pvco_record[0]
                reference_pressure = float(record["p_ref"])
                reference_oil_fvf = float(record["bo"])
                oil_compressibility = float(record["co"])
                reference_viscosity = float(record["viscosity"])
                # Build a small synthetic pressure grid around the reference
                min_pressure = max(0.0, reference_pressure / 5.0)
                max_pressure = reference_pressure * 5.0
                pvco_pressures = np.linspace(min_pressure, max_pressure, 40)
                delta_p = pvco_pressures - reference_pressure
                oil_fvf = reference_oil_fvf * np.exp(-oil_compressibility * delta_p)
                oil_viscosity = np.full_like(pvco_pressures, reference_viscosity)
                synthetic_rows = [
                    {"pressure": float(p), "bo": float(b), "viscosity": float(v)}
                    for p, b, v in zip(pvco_pressures, oil_fvf, oil_viscosity)
                ]
                oil_data = _build_oil_data_from_pvdo(
                    pvdo_records=synthetic_rows,
                    density_record=density_record,
                    temperature=temperature.for_region(pvtnum),
                    unit_system=unit_system,
                    interpolation_method=interpolation_method,
                    dtype=dtype,
                )

        # Gas Phase
        gas_data: typing.Optional[PVTData] = None
        if pvtg_all is not None and region_idx < len(pvtg_all):
            gas_data = _build_gas_data_from_pvtg(
                pvtg_records=pvtg_all[region_idx],
                density_record=density_record,
                unit_system=unit_system,
                dtype=dtype,
            )
        elif pvdg_all is not None and region_idx < len(pvdg_all):
            gas_data = _build_gas_data_from_pvdg(
                pvdg_records=pvdg_all[region_idx],
                density_record=density_record,
                temperature=temperature.for_region(pvtnum),
                unit_system=unit_system,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )

        # Water
        water_data: typing.Optional[PVTData] = None
        salinity = 0.0  # Salinity is not stored in the PVTW record; default to 0 ppm
        if pvtw_all is not None and region_idx < len(pvtw_all):
            pvtw_rows = pvtw_all[region_idx]
            if pvtw_rows:
                water_data = _build_water_data_from_pvtw(
                    pvtw_record=pvtw_rows[0],
                    density_record=density_record,
                    temperature=temperature.for_region(pvtnum),
                    unit_system=unit_system,
                    salinity=salinity,
                    interpolation_method=interpolation_method,
                    dtype=dtype,
                )

        # `StaticPVT` for this region
        # Resolve stock-tank densities from DENSITY record
        stock_tank_oil_density = (
            density_record["oil"] if density_record is not None else None
        )
        stock_tank_water_density = (
            density_record["water"] if density_record is not None else None
        )
        stock_tank_gas_density = (
            density_record["gas"] if density_record is not None else None
        )

        # PVTW scalars for this region
        water_reference_pressure: typing.Optional[Number] = None
        water_reference_fvf: typing.Optional[Number] = None
        water_reference_viscosity: typing.Optional[Number] = None
        water_reference_compressibility: typing.Optional[Number] = None
        water_viscosibility: typing.Optional[Number] = None
        if pvtw_all is not None and region_idx < len(pvtw_all):
            pvtw_rows = pvtw_all[region_idx]
            if pvtw_rows:
                pvtw_record = pvtw_rows[0]
                water_reference_pressure = pvtw_record["p_ref"]
                water_reference_fvf = pvtw_record["bw"]
                water_reference_compressibility = pvtw_record["cw"]
                water_reference_viscosity = pvtw_record["viscosity"]
                water_viscosibility = pvtw_record.get("cv", 0.0)

        static = StaticPVT(
            stock_tank_oil_density=stock_tank_oil_density,
            water_reference_pressure=water_reference_pressure,
            water_reference_fvf=water_reference_fvf,
            water_reference_viscosity=water_reference_viscosity,
            water_reference_compressibility=water_reference_compressibility,
            stock_tank_water_density=stock_tank_water_density,
            stock_tank_gas_density=stock_tank_gas_density,
            water_viscosibility=water_viscosibility,
            water_salinity=salinity,
            unit_system=unit_system,
        )

        # Assemble `PVTRegion`
        dataset = PVTDataSet(oil=oil_data, gas=gas_data, water=water_data)
        tables = PVTTables.from_dataset(dataset, **table_kwargs)
        regions[pvtnum] = PVTRegion(
            static=static, tables=tables, unit_system=unit_system
        )

        logger.debug(
            "Built PVT tables and properties for region %d: oil=%s, gas=%s, water=%s, salinity=%.0f ppm",
            pvtnum,
            "PVTO"
            if pvto_all and region_idx < len(pvto_all)
            else "PVDO"
            if pvdo_all and region_idx < len(pvdo_all)
            else "PVCO"
            if pvco_all and region_idx < len(pvco_all)
            else "none",
            "PVTG"
            if pvtg_all and region_idx < len(pvtg_all)
            else "PVDG"
            if pvdg_all and region_idx < len(pvdg_all)
            else "none",
            "PVTW" if pvtw_all and region_idx < len(pvtw_all) else "none",
            salinity,
        )
    return regions
