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
    interp1d,
)
from typing_extensions import Self

from bores.constants import c
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
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
)

logger = logging.getLogger(__name__)

__all__ = ["PVTRegions", "load_pvt_regions"]


@typing.final
@attrs.frozen
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

    regions: typing.Dict[int, PVTTables]
    """Mapping from 1-based PVTNUM index to `PVTTables`."""

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


def _degenerate_temperature_axis(
    temperature: float, dtype: npt.DTypeLike
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
    density_record: typing.Optional[typing.Dict[str, float]],
    temperature: float,
    pvt: typing.Optional[PVT],
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> PVTData:
    """
    Build oil `PVTData` from a parsed `PVTO` record set.

    `PVTO` format: Rs is the outer key; each Rs group contains rows of
    `(pressure, bo, viscosity)` ordered by ascending pressure. The first
    row in each Rs group is at bubble-point pressure (saturated); subsequent
    rows at higher pressure represent the undersaturated branch.

    The builder:

    1. Extracts the saturated branch `(Pb(Rs), Bo_sat(Rs), μo_sat(Rs))`.
    2. Builds a 2-D table on a regular `(P, Rs)` grid by interpolating
       across Rs groups using the saturated Bo and μo at each pressure.
    3. Constructs the bubble-point array `Pb(Rs)` for the
       `bubble_point_pressures` field.
    4. Derives the density table using
       `ρo = (ρo,SC + Rs·ρg,SC) / Bo` if reference densities are available.
    5. Derives the compressibility table from `∂Bo/∂P`.

    :param pvto_records: List of row dicts from the parsed `PVTO` keyword.
        Each dict has keys `"rs"`, `"pressure"`, `"bo"`, `"viscosity"`.
    :param density_record: `DENSITY` record dict with `"oil"` and `"gas"` keys
        (lbm/ft³ at standard conditions).
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional; provides reference densities when available.
    :returns: `PVTData` for the oil phase.
    """
    dtype = dtype if dtype is not None else get_dtype()
    temperatures = _degenerate_temperature_axis(temperature, dtype=dtype)
    n_t = len(temperatures)

    # Group records by Rs value
    solution_gor_to_rows: typing.Dict[float, typing.List[typing.Dict]] = {}
    for row in pvto_records:
        solution_gor = float(row["rs"])
        solution_gor_to_rows.setdefault(solution_gor, []).append(row)

    solution_gor_values = np.array(sorted(solution_gor_to_rows.keys()), dtype=dtype)
    n_rs = len(solution_gor_values)

    if n_rs < 2:
        raise ValidationError(f"PVTO table requires at least 2 Rs values; got {n_rs}.")

    # Saturated branch: first (lowest P) row in each Rs group
    bubble_point_pressure_values = np.empty(n_rs, dtype=dtype)
    saturated_oil_fvf = np.empty(n_rs, dtype=dtype)
    saturated_oil_viscosity = np.empty(n_rs, dtype=dtype)

    for i, solution_gor in enumerate(solution_gor_values):
        rows = sorted(solution_gor_to_rows[solution_gor], key=lambda r: r["pressure"])
        saturated_row = rows[0]
        bubble_point_pressure_values[i] = float(saturated_row["pressure"])
        saturated_oil_fvf[i] = float(saturated_row["bo"])
        saturated_oil_viscosity[i] = float(saturated_row["viscosity"])

    # Build a regular pressure grid spanning the full range seen in the table
    all_pressures = sorted(
        {
            float(row["pressure"])
            for rows in solution_gor_to_rows.values()
            for row in rows
        }
    )
    pressures = np.array(all_pressures, dtype=dtype)
    n_p = len(pressures)

    # For each (P, Rs) point: interpolate along the undersaturated branch of
    # the closest Rs group, or use the saturated value when P ≤ Pb(Rs).
    oil_fvf_2d = np.empty((n_p, n_t), dtype=dtype)
    oil_viscosity_2d = np.empty((n_p, n_t), dtype=dtype)
    solution_gor_2d = np.empty((n_p, n_t), dtype=dtype)  # Rs(P) saturated branch

    # Build per-Rs interpolators for the undersaturated branch
    oil_fvf_interps: typing.List[typing.Optional[typing.Callable]] = []
    oil_viscosity_interps: typing.List[typing.Optional[typing.Callable]] = []
    for solution_gor in solution_gor_values:
        rows = sorted(solution_gor_to_rows[solution_gor], key=lambda r: r["pressure"])
        pressure_arr = np.array([r["pressure"] for r in rows], dtype=dtype)
        oil_fvf_arr = np.array([r["bo"] for r in rows], dtype=dtype)
        oil_viscosity_arr = np.array([r["viscosity"] for r in rows], dtype=dtype)
        if len(pressure_arr) >= 2:
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
        else:
            oil_fvf_interps.append(None)
            oil_viscosity_interps.append(None)

    # Rs(P) on the saturated branch: interpolate Pb^{-1}(P) -> Rs
    # (valid for P ≤ max(Pb); Rs is constant at Rsb when P > Pb)
    solution_gor_of_pressure_interp = interp1d(
        bubble_point_pressure_values,
        solution_gor_values,
        kind="linear",
        bounds_error=False,
        fill_value=(solution_gor_values[0], solution_gor_values[-1]),
    )

    for i, pressure in enumerate(pressures):
        solution_gor_at_pressure = float(solution_gor_of_pressure_interp(pressure))
        solution_gor_2d[i, :] = solution_gor_at_pressure

        # Find the Rs group at or just above this pressure on the sat branch
        # to get the saturated Bo / μo at this pressure
        solution_gor_idx = np.searchsorted(bubble_point_pressure_values, pressure)
        solution_gor_idx = int(np.clip(solution_gor_idx, 0, n_rs - 1))

        oil_fvf_interp = oil_fvf_interps[solution_gor_idx]
        oil_viscosity_interp = oil_viscosity_interps[solution_gor_idx]

        oil_fvf = (
            float(oil_fvf_interp(pressure))
            if oil_fvf_interp is not None
            else float(saturated_oil_fvf[solution_gor_idx])
        )
        oil_viscosity = (
            float(oil_viscosity_interp(pressure))
            if oil_viscosity_interp is not None
            else float(saturated_oil_viscosity[solution_gor_idx])
        )

        oil_fvf_2d[i, :] = oil_fvf
        oil_viscosity_2d[i, :] = oil_viscosity

    # For a 2-D Pb(Rs, T) we broadcast
    bubble_point_pressure_2d = np.tile(
        bubble_point_pressure_values[:, np.newaxis], (1, n_t)
    ).astype(dtype)

    # Density table
    oil_density_2d: typing.Optional[npt.NDArray] = None
    standard_oil_density: typing.Optional[float] = None
    standard_gas_density: typing.Optional[float] = None

    if pvt is not None:
        standard_oil_density = getattr(pvt, "oil_density_sc", None)
        standard_gas_density = getattr(pvt, "gas_density_sc", None)
    if standard_oil_density is None and density_record is not None:
        standard_oil_density = density_record.get("oil")
        standard_gas_density = density_record.get("gas")

    if standard_oil_density is not None and standard_gas_density is not None:
        oil_density_2d = (
            standard_oil_density + solution_gor_2d * standard_gas_density
        ) / oil_fvf_2d
        oil_density_2d = oil_density_2d.astype(dtype)

    # Compressibility: co = -(1/Bo)·(∂Bo/∂P) per temperature column
    oil_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        d = PchipInterpolator(pressures, oil_fvf_2d[:, j]).derivative(1)
        dbo_dp = d(pressures)
        oil_compressibility = -(1.0 / oil_fvf_2d[:, j]) * dbo_dp
        oil_compressibility_2d[:, j] = np.clip(oil_compressibility, 0.0, 1e-1)

    return PVTData(
        phase=FluidPhase.OIL,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        bubble_point_pressures=typing.cast(
            FloatArray[OneDimension], bubble_point_pressure_2d
        ),
        solution_gas_to_oil_ratios=typing.cast(
            FloatArray[OneDimension], solution_gor_values
        ),
        formation_volume_factor_table=oil_fvf_2d,
        viscosity_table=oil_viscosity_2d,
        solution_gor_table=solution_gor_2d,
        density_table=oil_density_2d,
        compressibility_table=oil_compressibility_2d,
        dtype=dtype,
    )


def _build_oil_data_from_pvdo(
    pvdo_records: typing.List[typing.Dict[str, typing.Any]],
    density_record: typing.Optional[typing.Dict[str, float]],
    temperature: float,
    pvt: typing.Optional[PVT],
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> PVTData:
    """
    Build dead-oil `PVTData` from a parsed `PVDO` record set.

    `PVDO` format: single table of `(pressure, bo, viscosity)` rows for
    dead oil (no dissolved gas, Rs = 0 everywhere).

    :param pvdo_records: List of row dicts with keys `"pressure"`, `"bo"`,
        `"viscosity"`.
    :param density_record: `DENSITY` record; only `"oil"` key is used.
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional reference densities.
    :returns: `PVTData` for the oil phase.
    """
    dtype = dtype if dtype is not None else get_dtype()
    temperatures = _degenerate_temperature_axis(temperature, dtype=dtype)
    n_t = len(temperatures)

    rows = sorted(pvdo_records, key=lambda r: r["pressure"])
    pressures = np.array([r["pressure"] for r in rows], dtype=dtype)
    oil_fvf_1d = np.array([r["bo"] for r in rows], dtype=dtype)
    oil_viscosity_1d = np.array([r["viscosity"] for r in rows], dtype=dtype)
    n_p = len(pressures)

    if n_p < 2:
        raise ValidationError(f"PVDO table requires at least 2 rows; got {n_p}.")

    oil_fvf_2d = _broadcast_to_2d(oil_fvf_1d, n_t)
    oil_viscosity_2d = _broadcast_to_2d(oil_viscosity_1d, n_t)
    solution_gor_2d = np.zeros((n_p, n_t), dtype=dtype)  # dead oil: Rs = 0

    # Density: ρo = ρo,SC / Bo  (Rs = 0 for dead oil)
    oil_density_2d: typing.Optional[npt.NDArray] = None
    standard_oil_density: typing.Optional[float] = None

    if pvt is not None:
        standard_oil_density = getattr(pvt, "oil_density_sc", None)
    if standard_oil_density is None and density_record is not None:
        standard_oil_density = density_record.get("oil")

    if standard_oil_density is not None:
        oil_density_2d = (standard_oil_density / oil_fvf_2d).astype(dtype)

    # Compressibility
    oil_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        d = PchipInterpolator(pressures, oil_fvf_2d[:, j]).derivative(1)
        oil_compressibility = -(1.0 / oil_fvf_2d[:, j]) * d(pressures)
        oil_compressibility_2d[:, j] = np.clip(oil_compressibility, 0.0, 1e-1)

    return PVTData(
        phase=FluidPhase.OIL,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        formation_volume_factor_table=typing.cast(
            FloatArray[TwoDimensions], oil_fvf_2d
        ),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], oil_viscosity_2d),
        solution_gor_table=solution_gor_2d,
        density_table=oil_density_2d,
        compressibility_table=oil_compressibility_2d,
        dtype=dtype,
    )


def _build_gas_data_from_pvdg(
    pvdg_records: typing.List[typing.Dict[str, typing.Any]],
    density_record: typing.Optional[typing.Dict[str, float]],
    temperature: float,
    pvt: typing.Optional[PVT],
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> PVTData:
    """
    Build dry-gas `PVTData` from a parsed `PVDG` record set.

    `PVDG` format: single table of `(pressure, bg, viscosity)` rows.
    `Bg` is in rb/Mscf in Eclipse; stored as ft³/scf here (`Bg_ft3_scf =
    Bg_rb_Mscf x 5.615 / 1000`).

    :param pvdg_records: List of row dicts with keys `"pressure"`, `"bg"`,
        `"viscosity"`.
    :param density_record: `DENSITY` record; `"gas"` key used for ρg,SC.
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional reference densities.
    :returns: `PVTData` for the gas phase.
    """
    dtype = dtype if dtype is not None else get_dtype()
    temperatures = _degenerate_temperature_axis(temperature, dtype=dtype)
    n_t = len(temperatures)

    rows = sorted(pvdg_records, key=lambda r: r["pressure"])
    pressures = np.array([r["pressure"] for r in rows], dtype=dtype)
    # Eclipse Bg in rb/Mscf -> ft³/scf: multiply by 5.615 / 1000
    gas_fvf_1d = np.array([r["bg"] * 5.615 / 1000.0 for r in rows], dtype=dtype)
    oil_viscosity_1d = np.array([r["viscosity"] for r in rows], dtype=dtype)
    n_p = len(pressures)

    if n_p < 2:
        raise ValidationError(f"PVDG table requires at least 2 rows; got {n_p}.")

    gas_fvf_2d = _broadcast_to_2d(gas_fvf_1d, n_t)
    gas_viscosity_2d = _broadcast_to_2d(oil_viscosity_1d, n_t)

    # Density: ρg = ρg,SC / Bg
    gas_density_2d: typing.Optional[npt.NDArray] = None
    standard_gas_density: typing.Optional[float] = None

    if pvt is not None:
        standard_gas_density = getattr(pvt, "gas_density_sc", None)
    if standard_gas_density is None and density_record is not None:
        standard_gas_density = density_record.get("gas")

    if standard_gas_density is not None:
        gas_density_2d = (standard_gas_density / gas_fvf_2d).astype(dtype)

    # Compressibility: cg ≈ -(1/Bg)·(∂Bg/∂P)
    gas_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        d = PchipInterpolator(pressures, gas_fvf_2d[:, j]).derivative(1)
        gas_sompressibility = -(1.0 / gas_fvf_2d[:, j]) * d(pressures)
        gas_compressibility_2d[:, j] = np.clip(gas_sompressibility, 0.0, 1e-1)

    return PVTData(
        phase=FluidPhase.GAS,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        formation_volume_factor_table=typing.cast(
            FloatArray[TwoDimensions], gas_fvf_2d
        ),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], gas_viscosity_2d),
        density_table=gas_density_2d,
        compressibility_table=gas_compressibility_2d,
        dtype=dtype,
    )


def _build_gas_data_from_pvtg(
    pvtg_records: typing.List[typing.Dict[str, typing.Any]],
    density_record: typing.Optional[typing.Dict[str, float]],
    temperature: float,
    pvt: typing.Optional[PVT],
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> PVTData:
    """
    Build wet-gas `PVTData` from a parsed `PVTG` record set.

    `PVTG` format: pressure is the outer key; each pressure group contains
    rows of `(rv, bg, viscosity)` ordered by ascending Rv. The first row
    in each pressure group is the dry-gas value (Rv = 0).

    The builder constructs a 2-D table on a regular `(P, Rv)` grid where
    the temperature axis is replaced by Rv, matching the convention for wet-gas
    tables (`PVTTable` is flag-aware).

    :param pvtg_records: List of row dicts with keys `"pressure"`, `"rv"`,
        `"bg"`, `"viscosity"`.
    :param density_record: `DENSITY` record; `"gas"` and `"oil"` keys used.
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional reference densities.
    :returns: `PVTData` for the gas phase with Rv as the second axis.
    """
    dtype = dtype if dtype is not None else get_dtype()

    # Group by pressure
    pressure_to_rows: typing.Dict[float, typing.List[typing.Dict]] = {}
    for row in pvtg_records:
        pressure = float(row["pressure"])
        pressure_to_rows.setdefault(pressure, []).append(row)

    pressure_values = np.array(sorted(pressure_to_rows.keys()), dtype=dtype)
    n_p = len(pressure_values)

    if n_p < 2:
        raise ValidationError(
            f"PVTG table requires at least 2 pressure values; got {n_p}."
        )

    # Collect all Rv values across all pressure groups
    all_vaporized_oil_ratio = sorted(
        {float(row["rv"]) for rows in pressure_to_rows.values() for row in rows}
    )
    vaporized_oil_ratio_values = np.array(all_vaporized_oil_ratio, dtype=dtype)
    n_rv = len(vaporized_oil_ratio_values)

    # Build 2-D (P, Rv) tables
    gas_fvf_2d = np.empty((n_p, n_rv), dtype=dtype)
    gas_viscosity_2d = np.empty((n_p, n_rv), dtype=dtype)

    for i, pressure in enumerate(pressure_values):
        rows = sorted(pressure_to_rows[pressure], key=lambda r: r["rv"])
        vaporized_oil_ratio_arr = np.array([r["rv"] for r in rows], dtype=dtype)
        gas_fvf_arr = np.array([r["bg"] * 5.615 / 1000.0 for r in rows], dtype=dtype)
        gas_viscosity_arr = np.array([r["viscosity"] for r in rows], dtype=dtype)

        gas_fvf_interp = interp1d(
            vaporized_oil_ratio_arr,
            gas_fvf_arr,
            kind="linear",
            bounds_error=False,
            fill_value=(gas_fvf_arr[0], gas_fvf_arr[-1]),
        )
        gas_viscosity_interp = interp1d(
            vaporized_oil_ratio_arr,
            gas_viscosity_arr,
            kind="linear",
            bounds_error=False,
            fill_value=(gas_viscosity_arr[0], gas_viscosity_arr[-1]),
        )
        gas_fvf_2d[i, :] = gas_fvf_interp(vaporized_oil_ratio_values)
        gas_viscosity_2d[i, :] = gas_viscosity_interp(vaporized_oil_ratio_values)

    # Density: ρg = (ρg,SC + Rv·ρo,SC) / Bg
    gas_density_2d: typing.Optional[npt.NDArray] = None
    standard_gas_density: typing.Optional[float] = None
    standard_oil_density: typing.Optional[float] = None

    if pvt is not None:
        standard_gas_density = getattr(pvt, "gas_density_sc", None)
        standard_oil_density = getattr(pvt, "oil_density_sc", None)
    if standard_gas_density is None and density_record is not None:
        standard_gas_density = density_record.get("gas")
        standard_oil_density = density_record.get("oil")

    if standard_gas_density is not None and standard_oil_density is not None:
        vaporized_oil_ratio_grid = np.tile(
            vaporized_oil_ratio_values[np.newaxis, :], (n_p, 1)
        )
        gas_density_2d = (
            (standard_gas_density + vaporized_oil_ratio_grid * standard_oil_density)
            / gas_fvf_2d
        ).astype(dtype)
    elif standard_gas_density is not None:
        gas_density_2d = (standard_gas_density / gas_fvf_2d).astype(dtype)

    # Compressibility: cg ≈ -(1/Bg)·(∂Bg/∂P) along dry-gas column
    gas_compressibility_2d = np.empty((n_p, n_rv), dtype=dtype)
    for j in range(n_rv):
        d_bg = PchipInterpolator(pressure_values, gas_fvf_2d[:, j]).derivative(1)
        gas_compressibility = -(1.0 / gas_fvf_2d[:, j]) * d_bg(pressure_values)
        gas_compressibility_2d[:, j] = np.clip(gas_compressibility, 0.0, 1e-1)

    return PVTData(
        phase=FluidPhase.GAS,
        pressures=typing.cast(FloatArray[OneDimension], pressure_values),
        # Note: Rv axis stored here for wet-gas table
        temperatures=typing.cast(FloatArray[OneDimension], vaporized_oil_ratio_values),
        formation_volume_factor_table=gas_fvf_2d,
        viscosity_table=gas_viscosity_2d,
        vaporized_oil_ratio_table=typing.cast(
            FloatArray[TwoDimensions],
            np.tile(vaporized_oil_ratio_values[np.newaxis, :], (n_p, 1)).astype(dtype),
        ),
        density_table=gas_density_2d,
        compressibility_table=gas_compressibility_2d,
        dtype=dtype,
    )


def _build_water_data_from_pvtw(
    pvtw_record: typing.Dict[str, float],
    density_record: typing.Optional[typing.Dict[str, float]],
    temperature: float,
    pvt: typing.Optional[PVT],
    salinity: float = 0.0,
    n_pressure_points: int = 50,
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> PVTData:
    """
    Build water `PVTData` from a `PVTW` analytical record.

    `PVTW` is not a table keyword - it provides four scalars:

    - `p_ref` - reference pressure (psi)
    - `bw`    - Bw at `p_ref` (bbl/STB)
    - `cw`    - water compressibility (psi⁻¹)
    - `viscosity` - μw at `p_ref` (cP)
    - `cv`    - water viscosibility (psi⁻¹, often 0)

    From these, Bw(P) and μw(P) are evaluated analytically on a pressure grid
    spanning `[p_ref / 10, p_ref x 10]` and stored as 2-D tables so that
    all subsequent lookups are interpolator calls (no per-step analytics).

    :param pvtw_record: Dict with keys `"p_ref"`, `"bw"`, `"cw"`,
        `"viscosity"`, and optionally `"cv"`.
    :param density_record: `DENSITY` record; `"water"` key used for ρw,SC.
    :param temperature: Reservoir temperature (°F).
    :param pvt: Optional reference densities.
    :param salinity: Water salinity (ppm NaCl); used for `default_salinity` only.
    :param n_pressure_points: Number of pressure points in the synthetic table.
    :returns: `PVTData` for the water phase.
    """
    dtype = dtype if dtype is not None else get_dtype()
    temperatures = _degenerate_temperature_axis(temperature, dtype=dtype)
    n_t = len(temperatures)
    salinities = np.array([salinity], dtype=dtype)

    reference_pressure = float(pvtw_record["p_ref"])
    reference_water_fvf = float(pvtw_record["bw"])
    water_compressibility = float(pvtw_record["cw"])
    reference_water_viscosity = float(pvtw_record["viscosity"])
    water_viscosibility = float(pvtw_record.get("cv", 0.0))

    # Build a pressure grid spanning a physically reasonable range
    min_pressure = max(14.696, reference_pressure / 10.0)
    max_pressure = reference_pressure * 10.0
    pressures = np.linspace(min_pressure, max_pressure, n_pressure_points, dtype=dtype)
    n_p = len(pressures)

    # Analytical Bw(P): exponential model
    # Bw(P) = Bw_ref * exp(-cw * (P - P_ref))
    delta_p = pressures - reference_pressure
    water_fvf_1d = reference_water_fvf * np.exp(-water_compressibility * delta_p)

    # Analytical μw(P): viscosibility model
    # μw(P) = μw_ref * exp(-cv * (P - P_ref))
    water_viscosity_1d = reference_water_viscosity * np.exp(
        -water_viscosibility * delta_p
    )

    # Broadcast to (n_p, n_t, n_s) - single salinity
    water_fvf_2d = _broadcast_to_2d(water_fvf_1d, n_t)
    water_viscosity_2d = _broadcast_to_2d(water_viscosity_1d, n_t)
    water_fvf_3d = water_fvf_2d[:, :, np.newaxis]  # (n_p, n_t, 1)
    water_viscosity_3d = water_viscosity_2d[:, :, np.newaxis]

    # Density: ρw = ρw,SC / Bw
    water_density_3d: typing.Optional[npt.NDArray] = None
    standard_water_density: typing.Optional[float] = None

    if pvt is not None:
        standard_water_density = getattr(pvt, "water_density_sc", None)
    if standard_water_density is None and density_record is not None:
        standard_water_density = density_record.get("water")

    if standard_water_density is not None:
        water_density_3d = (standard_water_density / water_fvf_3d).astype(dtype)

    # Compressibility: cw(P) = -(1/Bw)·(dBw/dP) = cw (constant for this model)
    # But we store as a table to remain consistent with the lookup API
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
        density_table=water_density_3d,
        compressibility_table=water_compressibility_3d,
        gas_free_water_fvf_table=typing.cast(FloatArray[TwoDimensions], water_fvf_2d),
        dtype=dtype,
    )


def load_pvt_regions(
    deck_file: DeckFile,
    temperature: float,
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
            "No oil PVT keyword found in DeckFile.  Expected one of: PVTO, PVDO, PVCO."
        )

    # Number of regions is the maximum length across all keyword lists
    n_regions = max(
        len(x)
        for x in [pvto_all, pvdo_all, pvco_all, pvtg_all, pvdg_all, pvtw_all]
        if x is not None
    )

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
        density_record: typing.Optional[typing.Dict[str, float]] = None
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
                dtype=dtype,
            )
        elif pvdo_all is not None and region_idx < len(pvdo_all):
            oil_data = _build_oil_data_from_pvdo(
                pvdo_records=pvdo_all[region_idx],
                density_record=density_record,
                temperature=temperature,
                pvt=pvt,
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
                dtype=dtype,
            )
        elif pvdg_all is not None and region_idx < len(pvdg_all):
            gas_data = _build_gas_data_from_pvdg(
                pvdg_records=pvdg_all[region_idx],
                density_record=density_record,
                temperature=temperature,
                pvt=pvt,
                dtype=dtype,
            )

        # Water
        water_data: typing.Optional[PVTData] = None
        if pvtw_all is not None and region_idx < len(pvtw_all):
            pvtw_rows = pvtw_all[region_idx]
            if pvtw_rows:
                salinity = (
                    getattr(pvt, "water_salinity", 0.0) if pvt is not None else 0.0
                )
                water_data = _build_water_data_from_pvtw(
                    pvtw_record=pvtw_rows[0],
                    density_record=density_record,
                    temperature=temperature,
                    pvt=pvt,
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
