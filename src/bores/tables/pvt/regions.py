import logging
import typing
import warnings

import numpy as np
import numpy.typing as npt
from scipy.interpolate import (  # type: ignore[import-untyped]
    PchipInterpolator,
    interp1d,
)
from typing_extensions import Self

from bores.constants import UnitConversionTable, c
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.model.properties import PVT
from bores.precision import get_dtype
from bores.stores import StoreSerializable
from bores.tables.pvt.base import InterpolationMethod, PVTTable, PVTTables
from bores.tables.pvt.data import PVTData, PVTDataSet
from bores.typing import (
    FloatArray,
    FluidPhase,
    Number,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = ["PVTRegions", "load_pvt_regions"]


class PVTRegions(StoreSerializable):
    """
    Multi-region PVT tables keyed by 1-based `PVTNUM` region index.

    Eclipse supports multiple PVT regions via the `PVTNUM` keyword. Each
    cell is assigned a region index and its PVT properties are evaluated from
    the corresponding `PVTTables` instance.

    Use `for_region(pvtnum)` to retrieve the tables for a given region, and
    `load_pvt_regions` to construct from a deck.

    Example:

    ```python
    regions = PVTRegions.from_deck_file(deck_file, temperature=200.0)
    cell_tables = regions.for_region(pvtnum_array[cell_idx])
    bo = cell_tables.oil.formation_volume_factor(p, t)
    ```
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        regions: typing.Dict[int, PVTTables],
    ) -> None:
        """
        Build a `PVTRegions` from a pre-built regions dict.

        :param regions: Mapping from 1-based PVTNUM index to `PVTTables`.
        :raises ValidationError: If *regions* is empty.
        """
        if not regions:
            raise ValidationError("`regions` must contain at least one entry.")
        self.regions = regions

    def for_region(self, pvtnum: int) -> PVTTables:
        """
        Return the `PVTTables` for a given 1-based region index.

        :param pvtnum: 1-based PVT region index.
        :returns: `PVTTables` for that region.
        :raises KeyError: If the region index does not exist.
        """
        tables = self.regions.get(pvtnum)
        if tables is None:
            available = sorted(self.regions.keys())
            raise KeyError(
                f"PVT region {pvtnum} not found.  Available regions: {available}."
            )
        return tables

    @property
    def n_regions(self) -> int:
        """Number of PVT regions."""
        return len(self.regions)

    @classmethod
    def single_region(cls, tables: PVTTables) -> Self:
        """
        Wrap a single `PVTTables` as region 1.

        Convenience factory for the common single-region case.

        :param tables: `PVTTables` instance.
        :returns: `PVTRegions` with one entry at key 1.
        """
        return cls(regions={1: tables})

    @classmethod
    def from_deck_file(
        cls,
        deck_file: DeckFile,
        temperature: float,
        *,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        pvt: typing.Optional[PVT] = None,
        dtype: typing.Optional[npt.DTypeLike] = None,
    ) -> Self:
        """
        Build all PVT regions from a parsed `DeckFile`.

        Detects which Eclipse PVT keywords are present (`PVTO` > `PVDO` > `PVCO` for oil;
        `PVTG` > `PVDG` for gas; `PVTW` for water) and builds one `PVTTables` per `PVTNUM` region.

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param temperature: Reservoir temperature (°F).
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param validate: Run physical-consistency checks.
        :param warn_on_extrapolation: Log warnings on extrapolation.
        :param pvt: Reference densities for derived table derivation.
        :returns: `PVTRegions` keyed by 1-based PVTNUM index.
        """
        regions = load_pvt_regions(
            deck_file=deck_file,
            temperature=temperature,
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            pvt=pvt,
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
                pvtnum: tables.convert(target, table=table)
                for pvtnum, tables in self.regions.items()
            }
        )

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "regions": {
                str(pvtnum): tables.dump(recurse)
                for pvtnum, tables in self.regions.items()
            }
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        regions = {
            int(pvtnum): PVTTables.load(table_data)
            for pvtnum, table_data in data["regions"].items()
        }
        return cls(regions=regions)


def _degenerate_temperature_axis(
    temperature: Number, dtype: npt.DTypeLike
) -> npt.NDArray:
    """
    Build a minimal two-point temperature axis bracketing *temperature*.

    Eclipse PVT tables are isothermal, i.e there is only one reservoir
    temperature. The existing `PVTTable` infrastructure requires a
    2-D `(n_p, n_t)` table shape with n_t ≥ 1. To satisfy this while
    keeping interpolation valid we create a degenerate two-point T axis
    `[T-1, T+1]` and broadcast values across it; the resulting 2-D table
    evaluates exactly to the 1-D values at T.

    :param temperature: Reservoir temperature in °F.
    :returns: 1-D array `[T-1, T+1]`.
    """
    return np.array([temperature - 1.0, temperature + 1.0], dtype=dtype)


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
    temperature: Number,
    pvt: typing.Optional[PVT],
    unit_system: UnitSystem,
    dtype: typing.Optional[npt.DTypeLike] = None,
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
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional; provides reference densities when available.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the oil phase.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    temperatures = _degenerate_temperature_axis(temperature, dtype=dtype)
    n_t = len(temperatures)

    # Group records by Rs value
    solution_gor_to_rows: typing.Dict[float, typing.List[typing.Dict]] = {}
    for row in pvto_records:
        solution_gor_val = float(row["rs"])
        solution_gor_to_rows.setdefault(solution_gor_val, []).append(row)

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

    for i, solution_gor_val in enumerate(solution_gor_values):
        rows = sorted(
            solution_gor_to_rows[solution_gor_val], key=lambda r: r["pressure"]
        )
        sat_row = rows[0]
        bubble_point_pressure_values[i] = float(sat_row["pressure"])
        saturated_oil_fvf[i] = float(sat_row["bo"])
        saturated_oil_viscosity[i] = float(sat_row["viscosity"])

    # Pressure grid: bubble-point pressures + extension to max undersaturated pressure.
    # We do NOT merge all undersaturated rows into one flat grid - that would mix
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

    for solution_gor_val in solution_gor_values:
        rows = sorted(
            solution_gor_to_rows[solution_gor_val], key=lambda r: r["pressure"]
        )
        pressure_arr = np.array([r["pressure"] for r in rows], dtype=dtype)
        oil_fvf_arr = np.array([r["bo"] for r in rows], dtype=dtype)
        oil_viscosity_arr = np.array([r["viscosity"] for r in rows], dtype=dtype)
        oil_fvf_interps.append(
            interp1d(
                pressure_arr,
                oil_fvf_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(float(oil_fvf_arr[0]), float(oil_fvf_arr[-1])),
            )
        )
        oil_viscosity_interps.append(
            interp1d(
                pressure_arr,
                oil_viscosity_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(float(oil_viscosity_arr[0]), float(oil_viscosity_arr[-1])),
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
        fill_value=(float(solution_gor_values[0]), float(solution_gor_values[-1])),
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
        rs_at_pressure = float(solution_gor_of_pressure(pressure))
        solution_gor_2d[i, :] = rs_at_pressure

        oil_fvf_val = float(oil_fvf_interps[rs_idx](pressure))
        oil_viscosity_val = float(oil_viscosity_interps[rs_idx](pressure))

        oil_fvf_2d[i, :] = oil_fvf_val
        oil_viscosity_2d[i, :] = oil_viscosity_val

    # Bubble-point table: Pb(Rs, T) - 2-D, one Pb per Rs per temperature
    bubble_point_pressure_2d = np.tile(
        bubble_point_pressure_values[:, np.newaxis], (1, n_t)
    ).astype(dtype)

    # Resolve reference densities: pvt takes precedence over DENSITY record
    standard_oil_density: typing.Optional[Number] = None
    standard_gas_density: typing.Optional[Number] = None

    if pvt is not None:
        standard_oil_density = pvt.standard_gas_density
        standard_gas_density = pvt.standard_gas_density
    if standard_oil_density is None and density_record is not None:
        standard_oil_density = density_record.get("oil")
    if standard_gas_density is None and density_record is not None:
        standard_gas_density = density_record.get("gas")

    # Density: ρo = (ρo,SC + Rs·ρg,SC) / Bo
    oil_density_2d: typing.Optional[npt.NDArray] = None
    if standard_oil_density is not None and standard_gas_density is not None:
        oil_density_2d = (
            (standard_oil_density + solution_gor_2d * standard_gas_density) / oil_fvf_2d
        ).astype(dtype)

    # Compressibility: co = -(1/Bo)·(∂Bo/∂P) via PCHIP derivative
    oil_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbo_dp = PchipInterpolator(pressures, oil_fvf_2d[:, j]).derivative(1)(pressures)
        oil_compressibility = -(1.0 / oil_fvf_2d[:, j]) * dbo_dp
        # Compressibility must be non-negative; clamp to physical range
        oil_compressibility_2d[:, j] = np.clip(oil_compressibility, 0.0, 1e-1)

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
    temperature: Number,
    pvt: typing.Optional[PVT],
    unit_system: UnitSystem,
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> PVTData:
    """
    Build dead-oil `PVTData` from a parsed `PVDO` record set.

    `PVDO` format: single table of `(pressure, bo, viscosity)` rows for
    dead oil (Rs = 0 everywhere). No bubble-point switching is required
    since dead oil has no dissolved gas.

    :param pvdo_records: List of row dicts with keys `"pressure"`, `"bo"`,
        `"viscosity"`.
    :param density_record: `DENSITY` record; `"oil"` key used for ρo,SC.
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional reference densities.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the oil phase.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    temperatures = _degenerate_temperature_axis(temperature, dtype=dtype)
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

    standard_oil_density: typing.Optional[Number] = None
    if pvt is not None:
        standard_oil_density = pvt.standard_oil_density
    if standard_oil_density is None and density_record is not None:
        standard_oil_density = density_record.get("oil")

    # Density: ρo = ρo,SC / Bo (dead oil - Rs = 0)
    oil_density_2d: typing.Optional[npt.NDArray] = None
    if standard_oil_density is not None:
        oil_density_2d = (standard_oil_density / oil_fvf_2d).astype(dtype)

    # Compressibility: co = -(1/Bo)·(∂Bo/∂P)
    oil_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbo_dp = PchipInterpolator(pressures, oil_fvf_2d[:, j]).derivative(1)(pressures)
        oil_compressibility_2d[:, j] = np.clip(
            -(1.0 / oil_fvf_2d[:, j]) * dbo_dp, 0.0, 1e-1
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
    temperature: Number,
    pvt: typing.Optional[PVT],
    unit_system: UnitSystem,
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> PVTData:
    """
    Build dry-gas `PVTData` from a parsed `PVDG` record set.

    `PVDG` format: single table of `(pressure, bg, viscosity)` rows.
    Eclipse stores Bg in rb/Mscf; this builder converts to ft³/scf:
    `Bg_ft3_scf = Bg_rb_Mscf x 5.615 / 1000`.

    :param pvdg_records: List of row dicts with keys `"pressure"`, `"bg"`,
        `"viscosity"`.
    :param density_record: `DENSITY` record; `"gas"` key used for ρg,SC.
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional reference densities.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the gas phase.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    temperatures = _degenerate_temperature_axis(temperature, dtype=dtype)
    n_t = len(temperatures)

    rows = sorted(pvdg_records, key=lambda r: r["pressure"])
    if len(rows) < 2:
        raise ValidationError(f"PVDG table requires at least 2 rows; got {len(rows)}.")

    pressures = np.array([r["pressure"] for r in rows], dtype=dtype)
    # Convert rb/Mscf → ft³/scf
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

    standard_gas_density: typing.Optional[Number] = None
    if pvt is not None:
        standard_gas_density = pvt.standard_gas_density
    if standard_gas_density is None and density_record is not None:
        standard_gas_density = density_record.get("gas")

    # Density: ρg = ρg,SC / Bg
    gas_density_2d: typing.Optional[npt.NDArray] = None
    if standard_gas_density is not None:
        gas_density_2d = (standard_gas_density / gas_fvf_2d).astype(dtype)

    # Compressibility: cg ≈ -(1/Bg)·(∂Bg/∂P)
    gas_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbg_dp = PchipInterpolator(pressures, gas_fvf_2d[:, j]).derivative(1)(pressures)
        gas_compressibility_2d[:, j] = np.clip(
            -(1.0 / gas_fvf_2d[:, j]) * dbg_dp, 0.0, 1e-1
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
    temperature: Number,
    pvt: typing.Optional[PVT],
    unit_system: UnitSystem,
    dtype: typing.Optional[npt.DTypeLike] = None,
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
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional reference densities.
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
    all_rv = sorted(
        {float(row["rv"]) for rows in pressure_to_rows.values() for row in rows}
    )
    if len(all_rv) < 1:
        raise ValidationError("PVTG table contains no Rv values.")

    rv_values = np.array(all_rv, dtype=dtype)
    n_rv = len(rv_values)

    gas_fvf_2d = np.empty((n_p, n_rv), dtype=dtype)
    gas_viscosity_2d = np.empty((n_p, n_rv), dtype=dtype)

    for i, pressure in enumerate(pressure_values):
        rows = sorted(pressure_to_rows[pressure], key=lambda r: r["rv"])
        rv_arr = np.array([r["rv"] for r in rows], dtype=dtype)
        # Convert rb/Mscf → ft³/scf
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
            gas_fvf_2d[i, :] = float(gas_fvf_arr[0])
            gas_viscosity_2d[i, :] = float(gas_viscosity_arr[0])
        else:
            gas_fvf_2d[i, :] = interp1d(
                rv_arr,
                gas_fvf_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(float(gas_fvf_arr[0]), float(gas_fvf_arr[-1])),
            )(rv_values)
            gas_viscosity_2d[i, :] = interp1d(
                rv_arr,
                gas_viscosity_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(float(gas_viscosity_arr[0]), float(gas_viscosity_arr[-1])),
            )(rv_values)

    # Resolve reference densities
    standard_gas_density: typing.Optional[Number] = None
    standard_oil_density: typing.Optional[Number] = None

    if pvt is not None:
        standard_oil_density = pvt.standard_gas_density
        standard_gas_density = pvt.standard_gas_density
    if standard_gas_density is None and density_record is not None:
        standard_gas_density = density_record.get("gas")
    if standard_oil_density is None and density_record is not None:
        standard_oil_density = density_record.get("oil")

    # Density: ρg = (ρg,SC + Rv·ρo,SC) / Bg  [wet gas]
    #          ρg = ρg,SC / Bg                  [dry gas, Rv = 0 column]
    gas_density_2d: typing.Optional[npt.NDArray] = None
    if standard_gas_density is not None:
        rv_grid = np.tile(rv_values[np.newaxis, :], (n_p, 1))
        if standard_oil_density is not None:
            gas_density_2d = (
                (standard_gas_density + rv_grid * standard_oil_density) / gas_fvf_2d
            ).astype(dtype)
        else:
            gas_density_2d = (standard_gas_density / gas_fvf_2d).astype(dtype)

    # Compressibility: cg ≈ -(1/Bg)·(∂Bg/∂P) along each Rv column
    gas_compressibility_2d = np.empty((n_p, n_rv), dtype=dtype)
    for j in range(n_rv):
        dbg_dp = PchipInterpolator(pressure_values, gas_fvf_2d[:, j]).derivative(1)(
            pressure_values
        )
        gas_compressibility_2d[:, j] = np.clip(
            -(1.0 / gas_fvf_2d[:, j]) * dbg_dp, 0.0, 1e-1
        )

    # Rv table: shape (n_p, n_rv) - same Rv values at every pressure
    vaporized_oil_ratio_table = np.tile(rv_values[np.newaxis, :], (n_p, 1)).astype(
        dtype
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
    temperature: Number,
    pvt: typing.Optional[PVT],
    unit_system: UnitSystem,
    salinity: Number = 0.0,
    n_pressure_points: int = 50,
    dtype: typing.Optional[npt.DTypeLike] = None,
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
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional reference densities.
    :param salinity: Water salinity (ppm NaCl).
    :param n_pressure_points: Points in the synthetic pressure grid.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the water phase.
    """
    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    temperatures = _degenerate_temperature_axis(temperature, dtype=dtype)
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

    min_pressure = max(14.696, reference_pressure / 10.0)
    max_pressure = reference_pressure * 10.0
    pressures = np.linspace(min_pressure, max_pressure, n_pressure_points, dtype=dtype)
    n_p = len(pressures)

    delta_p = pressures - reference_pressure
    water_fvf_1d = (
        reference_water_fvf * np.exp(-water_compressibility * delta_p)
    ).astype(dtype)
    water_viscosity_1d = (
        reference_water_viscosity * np.exp(-water_viscosibility * delta_p)
    ).astype(dtype)

    water_fvf_2d = _broadcast_to_2d(water_fvf_1d, n_t)
    water_viscosity_2d = _broadcast_to_2d(water_viscosity_1d, n_t)
    # Expand to 3-D: (n_p, n_t, n_s=1)
    water_fvf_3d = water_fvf_2d[:, :, np.newaxis].astype(dtype)
    water_viscosity_3d = water_viscosity_2d[:, :, np.newaxis].astype(dtype)

    standard_water_density: typing.Optional[Number] = None
    if pvt is not None:
        standard_water_density = pvt.standard_water_density
    if standard_water_density is None and density_record is not None:
        standard_water_density = density_record.get("water")

    # Density: ρw = ρw,SC / Bw
    water_density_3d: typing.Optional[npt.NDArray] = None
    if standard_water_density is not None:
        water_density_3d = (standard_water_density / water_fvf_3d).astype(dtype)

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
    temperature: Number,
    *,
    interpolation_method: InterpolationMethod = "linear",
    validate: bool = True,
    warn_on_extrapolation: bool = False,
    pvt: typing.Optional[PVT] = None,
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> typing.Dict[int, PVTTables]:
    """
    Build a `PVTRegions` object from a parsed `DeckFile`.

    Detects which Eclipse PVT keywords are present and builds one
    `PVTTables` per `PVTNUM` region:

    - Oil: `PVTO` (live oil, preferred) -> `PVDO` (dead oil) -> `PVCO`
      (analytical; treated as dead oil with constant Bo).
    - Gas: `PVTG` (wet gas, preferred) -> `PVDG` (dry gas).
    - Water: `PVTW` (always analytical; converted to a table internally).

    `DENSITY` records supply the stock-tank reference densities used to
    derive density tables. When `pvt` is also provided it
    takes precedence.

    :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
    :param temperature: Reservoir temperature (°F).
    :param interpolation_method: `"linear"` (default) or `"cubic"`.
    :param validate: Run physical-consistency checks.
    :param warn_on_extrapolation: Log warnings when queries exceed table bounds.
    :param pvt: Optional `PVT` carrying reference
        densities (`oil_density_sc`, `gas_density_sc`, `water_density_sc`).
        When provided, `DENSITY` keyword values are not required.
    :returns: `PVTRegions` keyed by 1-based PVTNUM region index.
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
        pvt=pvt,
        dtype=dtype,
    )
    regions: typing.Dict[int, PVTTables] = {}

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
                temperature=temperature,
                pvt=pvt,
                unit_system=unit_system,
                dtype=dtype,
            )
        elif pvdo_all is not None and region_idx < len(pvdo_all):
            oil_data = _build_oil_data_from_pvdo(
                pvdo_records=pvdo_all[region_idx],
                density_record=density_record,
                temperature=temperature,
                pvt=pvt,
                unit_system=unit_system,
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
                min_pressure = max(14.696, reference_pressure / 5.0)
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
                    temperature=temperature,
                    pvt=pvt,
                    unit_system=unit_system,
                    dtype=dtype,
                )

        # Gas Phase
        gas_data: typing.Optional[PVTData] = None
        if pvtg_all is not None and region_idx < len(pvtg_all):
            gas_data = _build_gas_data_from_pvtg(
                pvtg_records=pvtg_all[region_idx],
                density_record=density_record,
                temperature=temperature,
                pvt=pvt,
                unit_system=unit_system,
                dtype=dtype,
            )
        elif pvdg_all is not None and region_idx < len(pvdg_all):
            gas_data = _build_gas_data_from_pvdg(
                pvdg_records=pvdg_all[region_idx],
                density_record=density_record,
                temperature=temperature,
                pvt=pvt,
                unit_system=unit_system,
                dtype=dtype,
            )

        # Water
        water_data: typing.Optional[PVTData] = None
        if pvtw_all is not None and region_idx < len(pvtw_all):
            pvtw_rows = pvtw_all[region_idx]
            if pvtw_rows:
                salinity = pvt.water_salinity if pvt is not None else 0.0
                water_data = _build_water_data_from_pvtw(
                    pvtw_record=pvtw_rows[0],
                    density_record=density_record,
                    temperature=temperature,
                    pvt=pvt,
                    unit_system=unit_system,
                    salinity=salinity,
                    dtype=dtype,
                )

        # Assemble `PVTTables` for this region
        dataset = PVTDataSet(oil=oil_data, gas=gas_data, water=water_data)
        tables = PVTTables.from_dataset(dataset, **table_kwargs)
        regions[pvtnum] = tables

        logger.debug(
            "Built PVT tables for region %d: oil=%s, gas=%s, water=%s",
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
        )
    return regions
