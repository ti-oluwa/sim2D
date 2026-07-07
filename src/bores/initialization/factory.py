"""
State initialization from EQUIL, explicit deck arrays, or user-supplied
arrays.

Owns every initialization algorithm - turning static reservoir description
(`Reservoir` + `BlackOilFluid`) plus initial-condition data
(`EquilibriumRegions`, explicit deck keywords, or arrays passed directly)
into a `State`. `State` itself knows nothing about `EQUIL`, decks, or
initialization algorithms (see `bores.reservoir.state`); this module is the
only place that logic lives.

**Precedence** (per field, matching Eclipse - see `initialize_state`):

    Restart > explicit deck/keyword array > EQUIL > error

Restart-file initialization (Step 11 of the architecture plan) is not yet
implemented - a `deck_file` with a `RESTART` keyword raises
`NotImplementedError` rather than silently doing the wrong thing.

**Equilibration model (v1 - center-point, sharp-contact)**

`initialize_equilibrium_state` computes a self-consistent pressure and
saturation profile for each `EQLNUM` region by:

1. Integrating the oil-referenced hydrostatic pressure `Po(z)` outward from
   the `EQUIL` datum via fixed-step RK4, using phase density evaluated at
   local `(P, T)` from the region's `PVTTable`s. In the gas cap and
   aquifer, `Pg`/`Pw` are integrated using gas/water density from the
   contact outward and re-referenced to `Po` using `pcog_goc`/`pcow_woc`
   as **constant** offsets - i.e. capillary pressure is *not* modelled as a
   function of saturation here, only as the fixed value supplied at the
   contact.
2. Assigning saturations via a **sharp** contact: `Sw = 1` below the WOC,
   `Sw = Swc` (connate) in the oil zone, `Sg = 1 - Swc` above the GOC.
   There is no smooth capillary transition zone in this version.
3. Assuming saturated oil/gas throughout each zone (`Rs = Rs_sat(P)`,
   `Rv = 0`) unless the region supplies an `RSVD`/`RVVD`
   `bores.reservoir.equilibrium.DepthTable`, in which case Rs/Rv (and, for
   Rs, the resulting bubble-point pressure) come from that table instead,
   allowing undersaturated oil columns.

**Known limitations - read before relying on this for anything but a
first cut:**

- No capillary-pressure-vs-saturation transition zone (see above). Accurate
  near contacts only if the true transition zone is thin relative to cell
  size, or capillary pressure is genuinely negligible in the field.
- `accuracy_flag` dispatch: only center-point (`accuracy_flag == 0`) is
  implemented. Horizontal-subdivision (`> 0`) and tilted-cell (`< 0`)
  variants raise `NotImplementedError` - they are the natural place to add
  the capillary transition zone, since sub-dividing each cell is what makes
  resolving it worthwhile.
- Wet-gas / gas-condensate `EQUIL` is not supported: when a region's gas
  table was built from `PVTG`, the underlying `PVTTable`'s second axis is
  Rv rather than temperature (see `bores.blackoil.pvt.regions`), and this
  module does not yet handle that convention - raises `NotImplementedError`.
  Dry-gas (`PVDG`) is fully supported.
- `datum_depth` must lie within the oil zone (between `goc_depth` and
  `woc_depth`, treating an absent contact as unbounded). Datum points in
  the gas cap or aquifer are not yet supported.
- `_HYDROSTATIC_GRADIENT_FACTOR` below is a set of standard physical
  constants (density x standard gravity, expressed per `UnitSystem`), not
  derived from `bores.constants`. Worth double-checking against that
  module's conventions, particularly for LAB units.
- `Temperature.from_deck` is called with just `deck_file`;
  verify this matches its actual signature - it was not directly
  re-confirmed while writing this module.
- An `EQLNUM` region spanning multiple `PVTNUM` regions is not supported
  (raises `ValidationError`) - each equilibration region must map to
  exactly one PVT region.
"""

import typing

import numpy as np
import numpy.typing as npt

from bores.blackoil.pvt.regions import PVTRegions
from bores.blackoil.pvt.tables import PVTTable
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.initialization.equilibrium import (
    DepthTable,
    EquilibriumRegion,
    EquilibriumRegions,
)
from bores.precision import get_dtype
from bores.reservoir.model import Reservoir
from bores.reservoir.state import Hysteresis, State
from bores.reservoir.temperature import (
    Temperature,
    TemperatureGradient,
    TemperatureTable,
)
from bores.typing import CellArray, IntCellArray, Number, UnitSystem
from bores.utils import get_hydrostatic_gradient_factor

__all__ = ["initialize_state", "initialize_equilibrium_state", "EquilibriumArrays"]


class EquilibriumArrays(typing.NamedTuple):
    """Per-cell arrays produced by an equilibration algorithm for one region."""

    pressure: CellArray
    water_saturation: CellArray
    gas_saturation: CellArray
    solution_gor: CellArray
    oil_bubble_point_pressure: CellArray
    vaporized_oil_ratio: CellArray
    gas_dew_point_pressure: CellArray


def _rk4_march(
    density_fn: typing.Callable[[Number, Number], Number],
    z0: Number,
    p0: Number,
    z_end: Number,
    gradient_factor: Number,
    step: Number = 1.0,
    dtype: npt.DTypeLike = None,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Integrate `dP/dz = gradient_factor * density_fn(P, z)` from `z0` to
    `z_end` via fixed-step RK4.

    :param density_fn: `(pressure, depth) -> density` at that point.
    :param z0: Starting depth.
    :param p0: Pressure at `z0`.
    :param z_end: Ending depth; may be less than `z0` (marches upward).
    :param gradient_factor: Unit-system hydrostatic-gradient constant.
    :param step: Target step size; the interval is divided into a whole
        number of steps no larger than this.
    :returns: `(depth, pressures)`, both ascending in depth and spanning
        `[min(z0, z_end), max(z0, z_end)]`.
    """
    if np.isclose(z0, z_end):
        return np.array([z0], dtype=dtype), np.array([p0], dtype=dtype)

    direction = 1.0 if z_end > z0 else -1.0
    n_steps = max(1, int(np.ceil(abs(z_end - z0) / step)))
    h = direction * abs(z_end - z0) / n_steps

    depth = np.empty(n_steps + 1, dtype=dtype)
    pressures = np.empty(n_steps + 1, dtype=dtype)
    depth[0], pressures[0] = z0, p0
    z, p = z0, p0
    for i in range(n_steps):
        k1 = gradient_factor * density_fn(p, z)
        k2 = gradient_factor * density_fn(p + 0.5 * h * k1, z + 0.5 * h)
        k3 = gradient_factor * density_fn(p + 0.5 * h * k2, z + 0.5 * h)
        k4 = gradient_factor * density_fn(p + h * k3, z + h)
        p = p + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z = z + h
        depth[i + 1], pressures[i + 1] = z, p

    if direction < 0:
        depth = depth[::-1]
        pressures = pressures[::-1]
    return depth, pressures


def _initialize_center_point_equilibrium(
    *,
    region: EquilibriumRegion,
    depth: CellArray,
    connate_water_saturation: CellArray,
    oil_table: PVTTable,
    gas_table: typing.Optional[PVTTable],
    water_table: typing.Optional[PVTTable],
    rsvd_table: typing.Optional[DepthTable],
    rvvd_table: typing.Optional[DepthTable],
    temperature: CellArray,
    unit_system: UnitSystem,
    depth_step: Number = 1.0,
    dtype: npt.DTypeLike = None,
) -> EquilibriumArrays:
    """
    Center-point EQUIL initialization: evaluate pressure and saturation once
    at each cell's centroid depth (`region.accuracy_flag == 0`).

    See the module docstring for the sharp-contact / no-transition-zone
    caveat and other v1 limitations.

    :param region: `EquilibriumRegion` for this region.
    :param depth: Shape `(n_cells,)` centroid depth of cells in this region.
    :param connate_water_saturation: Shape `(n_cells,)` Swc for cells in this
        region (from `Rock`).
    :param oil_table: Region's oil `PVTTable`.
    :param gas_table: Region's gas `PVTTable`, or `None`.
    :param water_table: Region's water `PVTTable`, or `None`.
    :param rsvd_table: `DepthTable` for `region.rsvd_table`, or `None`.
    :param rvvd_table: `DepthTable` for `region.rvvd_table`, or `None`.
    :param temperature: Shape `(n_cells,)` temperature at each cell's depth.
    :param unit_system: Unit system of all inputs.
    :param depth_step: RK4 step size for the pressure integration grid.
    :returns: `EquilibriumArrays` for this region's cells, same order as
        `depth`.
    :raises ValidationError: If `datum_depth` is outside the oil zone, or a
        contact is present without its corresponding PVT table.
    :raises NotImplementedError: If the gas table is wet-gas (`PVTG`-based).
    """
    is_wet_gas = gas_table is not None and gas_table.exists("vaporized_oil_ratio")
    if is_wet_gas and rvvd_table is None:
        raise NotImplementedError(
            "Wet-gas / gas-condensate EQUIL initialization without an RVVD table "
            "is not yet supported (no way to determine Rv(depth) to query the "
            "gas PVT table's Rv-indexed axis). Supply an RVVD table for this "
            "region, use a PVDG (dry-gas) table, or supply pressure/saturations "
            "explicitly for this region."
        )

    gradient_factor = get_hydrostatic_gradient_factor(unit_system)
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()

    oil_zone_low = (
        region.goc_depth if region.has_goc else min(depth.min(), region.datum_depth)
    )
    oil_zone_high = (
        region.woc_depth if region.has_woc else max(depth.max(), region.datum_depth)
    )
    if not (oil_zone_low <= region.datum_depth <= oil_zone_high):
        raise ValidationError(
            f"EQUIL datum_depth ({region.datum_depth}) must lie within the oil "
            f"zone [{oil_zone_low}, {oil_zone_high}] for center-point "
            "initialization; datum points in the gas cap or aquifer are not "
            "yet supported."
        )

    # Representative temperature at an arbitrary integration depth, via
    # linear interpolation against this region's cell depth/temperatures.
    order = np.argsort(depth)
    sorted_depths = depth[order]
    sorted_temperature = temperature[order]

    def temperature_at(z: Number) -> Number:
        return np.interp(z, sorted_depths, sorted_temperature)

    def oil_density(p: Number, z: Number) -> Number:
        return oil_table.density(p, temperature_at(z)).astype(dtype, copy=False)  # type: ignore[return-value]

    def gas_density(p: Number, z: Number) -> Number:
        assert gas_table is not None
        second_axis = rvvd_table.at_depth(z) if is_wet_gas else temperature_at(z)  # type: ignore[union-attr]
        return gas_table.density(p, second_axis).astype(dtype, copy=False)  # type: ignore[return-value]

    def water_density(p: Number, z: Number) -> Number:
        assert water_table is not None
        return water_table.density(p, temperature_at(z)).astype(dtype, copy=False)  # type: ignore[return-value]

    z_min = min(depth.min(), region.datum_depth, oil_zone_low)
    z_max = max(depth.max(), region.datum_depth, oil_zone_high)

    # Oil zone: integrate from datum outward to the zone edges.
    oil_depths_up, oil_pressures_up = _rk4_march(
        oil_density,
        region.datum_depth,
        region.datum_pressure,
        oil_zone_low,
        gradient_factor,
        step=depth_step,
        dtype=dtype,
    )
    oil_depths_down, oil_pressures_down = _rk4_march(
        oil_density,
        region.datum_depth,
        region.datum_pressure,
        oil_zone_high,
        gradient_factor,
        step=depth_step,
        dtype=dtype,
    )
    all_depths = [oil_depths_up[:-1], oil_depths_down]
    all_pressures = [oil_pressures_up[:-1], oil_pressures_down]

    # Gas cap: integrate Pg from GOC upward, re-reference to Po via pcog_goc.
    if region.has_goc and z_min < region.goc_depth:
        if gas_table is None:
            raise ValidationError(
                "Region has a gas-oil contact but no gas PVT table was supplied."
            )
        pg_at_goc = (
            np.interp(region.goc_depth, oil_depths_up, oil_pressures_up)
            + region.pcog_goc
        )
        gas_depths, gas_pressures = _rk4_march(
            gas_density,
            region.goc_depth,
            pg_at_goc,
            z_min,
            gradient_factor,
            step=depth_step,
            dtype=dtype,
        )
        all_depths.append(gas_depths[:-1])
        all_pressures.append(gas_pressures[:-1] - region.pcog_goc)

    # Aquifer: integrate Pw from WOC downward, re-reference to Po via pcow_woc.
    if region.has_woc and z_max > region.woc_depth:
        if water_table is None:
            raise ValidationError(
                "Region has a water-oil contact but no water PVT table was supplied."
            )
        pw_at_woc = (
            np.interp(region.woc_depth, oil_depths_down, oil_pressures_down)
            - region.pcow_woc
        )
        water_depths, water_pressures = _rk4_march(
            water_density,
            region.woc_depth,
            pw_at_woc,
            z_max,
            gradient_factor,
            step=depth_step,
            dtype=dtype,
        )
        all_depths.append(water_depths[1:])
        all_pressures.append(water_pressures[1:] + region.pcow_woc)

    grid_depths = np.concatenate(all_depths, dtype=dtype)
    grid_pressures = np.concatenate(all_pressures, dtype=dtype)
    grid_order = np.argsort(grid_depths)
    grid_depths = grid_depths[grid_order]
    grid_pressures = grid_pressures[grid_order]

    pressure = np.interp(depth, grid_depths, grid_pressures).astype(dtype, copy=False)

    # Saturations: sharp contact (see module docstring)
    n = len(depth)
    water_saturation = connate_water_saturation.astype(dtype, copy=False).copy()
    gas_saturation = np.zeros(n, dtype=dtype)
    if region.has_goc:
        in_gas_cap = depth < region.goc_depth
        gas_saturation[in_gas_cap] = 1.0 - connate_water_saturation[in_gas_cap]
        water_saturation[in_gas_cap] = connate_water_saturation[in_gas_cap]
    if region.has_woc:
        in_aquifer = depth > region.woc_depth
        water_saturation[in_aquifer] = 1.0
        gas_saturation[in_aquifer] = 0.0

    has_oil = (1.0 - water_saturation - gas_saturation) > 0.0

    # Solution GOR / bubble-point pressure
    solution_gor = np.zeros(n, dtype=dtype)
    oil_bubble_point_pressure = np.zeros(n, dtype=dtype)
    if np.any(has_oil):
        oil_pressure = pressure[has_oil]
        oil_temperature = temperature[has_oil]
        if rsvd_table is not None:
            gor = rsvd_table.at_depth(depth[has_oil]).astype(  # type: ignore[arg-type]
                dtype, copy=False
            )
            bubble_point_pressure = oil_table.bubble_point_pressure(
                temperature=oil_temperature, solution_gor=gor
            ).astype(dtype, copy=False)  # type: ignore[union-attr]
        else:
            # No RSVD: assume saturated oil (Pbub = P) everywhere in the oil
            # zone. Forcing `bubble_point_pressure=oil_pressure` makes the saturated
            # branch always trigger, reading Rs straight off the table at P.
            gor = oil_table.solution_gas_to_oil_ratio(
                pressure=oil_pressure,
                temperature=oil_temperature,
                bubble_point_pressure=oil_pressure,
            ).astype(dtype, copy=False)  # type: ignore[union-attr]
            bubble_point_pressure = oil_pressure.copy()

        solution_gor[has_oil] = gor
        oil_bubble_point_pressure[has_oil] = bubble_point_pressure

    # Vaporized oil ratio / dew-point pressure (dry-gas: Rv = 0 unless RVVD)
    vaporized_oil_ratio = np.zeros(n, dtype=dtype)
    gas_dew_point_pressure = np.zeros(n, dtype=dtype)
    has_free_gas = gas_saturation > 0.0
    if rvvd_table is not None and np.any(has_free_gas):
        vaporized_oil_ratio[has_free_gas] = rvvd_table.at_depth(
            depth[has_free_gas]  # type: ignore[arg-type]
        ).astype(dtype, copy=False)  # type: ignore[union-attr]

        if gas_table is not None:
            dew_point_arg = (
                rvvd_table.at_depth(depth[has_free_gas])  # type: ignore[arg-type]
                if is_wet_gas
                else temperature[has_free_gas]
            )
            dew_point = gas_table.dew_point_pressure(temperature=dew_point_arg)
            if dew_point is not None:
                gas_dew_point_pressure[has_free_gas] = dew_point.astype(  # type: ignore[union-attr]
                    dtype, copy=False
                )

    return EquilibriumArrays(
        pressure=typing.cast(CellArray, pressure),
        water_saturation=typing.cast(CellArray, water_saturation),
        gas_saturation=typing.cast(CellArray, gas_saturation),
        solution_gor=typing.cast(CellArray, solution_gor),
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure),
        vaporized_oil_ratio=typing.cast(CellArray, vaporized_oil_ratio),
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure),
    )


def _initialize_horizontal_subdivision_equilibrium(
    **kwargs: typing.Any,
) -> EquilibriumArrays:
    """
    Vertical-equilibrium averaging via `N` horizontal sub-divisions per cell
    (`region.accuracy_flag > 0`).

    Not yet implemented. This is the natural place to add a capillary-
    pressure-vs-saturation transition zone (currently a sharp contact
    everywhere - see the module docstring), since sub-dividing each cell is
    what makes resolving a smooth transition worthwhile.
    """
    raise NotImplementedError(
        "Horizontal-subdivision EQUIL initialization (accuracy_flag > 0) is "
        "not yet implemented. Use accuracy_flag = 0 (center-point) for now."
    )


def _initialize_tilted_subdivision_equilibrium(
    **kwargs: typing.Any,
) -> EquilibriumArrays:
    """
    Tilted-cell variant of horizontal subdivision, additionally accounting
    for cell dip (`region.accuracy_flag < 0`). Not yet implemented.
    """
    raise NotImplementedError(
        "Tilted-cell EQUIL initialization (accuracy_flag < 0) is not yet "
        "implemented. Use accuracy_flag = 0 (center-point) for now."
    )


def initialize_equilibrium_state(
    reservoir: Reservoir,
    pvt: PVTRegions,
    equilibrium: EquilibriumRegions,
    temperature: CellArray,
    *,
    depth_step: Number = 1.0,
    dtype: npt.DTypeLike = None,
) -> EquilibriumArrays:
    """
    Compute EQUIL-derived pressure/saturation/Rs/Rv arrays for every cell.

    Dispatches each `EQLNUM` region to the algorithm selected by that
    region's `accuracy_flag` and assembles the full-grid arrays.

    :param reservoir: Reservoir geometry, rock properties, and region
        assignments (`EQLNUM` via `reservoir.regions.equilibrium_regions`,
        `PVTNUM` via `reservoir.regions.pvt_regions`; both default to
        region 1 everywhere if absent).
    :param black_oil_model: PVT region tables.
    :param equilibrium: Parsed `EQUIL` (+ `RSVD`/`RVVD`) data.
    :param temperature: Shape `(n_cells,)` per-cell temperature.
    :param depth_step: RK4 step size for the pressure integration grid.
    :returns: Full-grid `EquilibriumArrays`.
    :raises ValidationError: If a cell's `EQLNUM` has no matching
        `EquilibriumRegion`, an `EQLNUM` region spans more than one `PVTNUM`
        region, or a required PVT table is missing.
    """
    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    equilibrium_regions: IntCellArray = (
        reservoir.regions.equilibrium_regions
        if reservoir.regions is not None
        and reservoir.regions.equilibrium_regions is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    pvt_regions: IntCellArray = (
        reservoir.regions.pvt_regions
        if reservoir.regions is not None and reservoir.regions.pvt_regions is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    depth = reservoir.depth
    connate_water_saturation = reservoir.rock.connate_water_saturation

    pressure = np.zeros(n_cells, dtype=dtype)
    water_saturation = np.zeros(n_cells, dtype=dtype)
    gas_saturation = np.zeros(n_cells, dtype=dtype)
    solution_gor = np.zeros(n_cells, dtype=dtype)
    oil_bubble_point_pressure = np.zeros(n_cells, dtype=dtype)
    vaporized_oil_ratio = np.zeros(n_cells, dtype=dtype)
    gas_dew_point_pressure = np.zeros(n_cells, dtype=dtype)

    for eqlnum in np.unique(equilibrium_regions):
        mask = equilibrium_regions == eqlnum
        equilibrium_region = equilibrium.region(int(eqlnum))

        region_pvtnum = np.unique(pvt_regions[mask])
        if len(region_pvtnum) != 1:
            raise ValidationError(
                f"EQLNUM {eqlnum} cells span multiple PVTNUM regions "
                f"({sorted(region_pvtnum.tolist())}); each equilibration "
                "region must map to exactly one PVT region for "
                "`initialize_equilibrium_state`."
            )

        pvt_region = pvt.region(int(region_pvtnum[0]))
        rsvd_table = (
            equilibrium.rsvd_tables.get(equilibrium_region.rsvd_table)
            if equilibrium_region.uses_rsvd and equilibrium.rsvd_tables
            else None
        )
        rvvd_table = (
            equilibrium.rvvd_tables.get(equilibrium_region.rvvd_table)
            if equilibrium_region.uses_rvvd and equilibrium.rvvd_tables
            else None
        )

        if equilibrium_region.accuracy_flag == 0:
            algorithm = _initialize_center_point_equilibrium
        elif equilibrium_region.accuracy_flag > 0:
            algorithm = _initialize_horizontal_subdivision_equilibrium
        else:
            algorithm = _initialize_tilted_subdivision_equilibrium

        region_arrays = algorithm(
            region=equilibrium_region,
            depth=depth[mask],  # type: ignore[arg-type]
            connate_water_saturation=connate_water_saturation[mask],  # type: ignore[arg-type]
            oil_table=pvt_region.tables.oil,  # type: ignore[arg-type]
            gas_table=pvt_region.tables.gas,
            water_table=pvt_region.tables.water,
            rsvd_table=rsvd_table,
            rvvd_table=rvvd_table,
            temperature=temperature[mask],  # type: ignore[arg-type]
            unit_system=reservoir.unit_system,
            depth_step=depth_step,
            dtype=dtype,
        )

        pressure[mask] = region_arrays.pressure
        water_saturation[mask] = region_arrays.water_saturation
        gas_saturation[mask] = region_arrays.gas_saturation
        solution_gor[mask] = region_arrays.solution_gor
        oil_bubble_point_pressure[mask] = region_arrays.oil_bubble_point_pressure
        vaporized_oil_ratio[mask] = region_arrays.vaporized_oil_ratio
        gas_dew_point_pressure[mask] = region_arrays.gas_dew_point_pressure

    return EquilibriumArrays(
        pressure=typing.cast(CellArray, pressure),
        water_saturation=typing.cast(CellArray, water_saturation),
        gas_saturation=typing.cast(CellArray, gas_saturation),
        solution_gor=typing.cast(CellArray, solution_gor),
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure),
        vaporized_oil_ratio=typing.cast(CellArray, vaporized_oil_ratio),
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure),
    )


def _temperature_array_from_regions(
    temperature: Temperature,
    pvt_regions: IntCellArray,
    depth: CellArray,
    dtype: npt.DTypeLike = None,
) -> CellArray:
    """
    Evaluate a `Temperature` at every cell's depth, grouped by
    `EQLNUM` to avoid redundant `region` lookups.

    Duck-types each region's spec: a bare number is broadcast, anything
    with an `at_depth` method (e.g. a depth-dependent table) is evaluated
    per cell.

    :param temperature: Source `Temperature`.
    :param pvt_regions: Shape `(n_cells,)` PVT region per cell.
    :param depth: Shape `(n_cells,)` cell centroid depth.
    :returns: Shape `(n_cells,)` temperature per cell.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    result = np.empty(len(pvt_regions), dtype=dtype)
    for pvtnum in np.unique(pvt_regions):
        mask = pvt_regions == pvtnum
        spec = temperature.region(int(pvtnum))
        if isinstance(spec, (TemperatureGradient, TemperatureTable)):
            result[mask] = spec.at_depth(depth[mask]).astype(dtype, copy=False)  # type: ignore[union-attr]
        else:
            result[mask] = float(spec)  # type: ignore[arg-type]
    return typing.cast(CellArray, result)


def _resolve_temperature(
    reservoir: Reservoir,
    deck_file: typing.Optional[DeckFile],
    temperature: typing.Optional[typing.Union[Temperature, Number]],
    dtype: npt.DTypeLike = None,
) -> CellArray:
    """
    Resolve per-cell temperature: explicit `temperature` kwarg > deck
    `RTEMP`/`TEMPVD` > error. (Restart temperature is not yet supported.)

    :raises ValidationError: If no temperature source is available.
    """
    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()

    source: typing.Optional[Temperature] = None
    if isinstance(temperature, Temperature):
        source = temperature
    elif temperature is not None:
        return typing.cast(CellArray, np.full(n_cells, temperature, dtype=dtype))
    elif deck_file is not None and (
        deck_file.has("RTEMP") or deck_file.has("TEMPVD") or deck_file.has("RTEMPVD")
    ):
        source = Temperature.from_deck(deck_file, dtype=dtype)
    else:
        raise ValidationError(
            "No temperature source available. Pass `temperature=` explicitly "
            "(a constant, or a `Temperature`), or supply a `deck_file` "
            "with `RTEMP`/`TEMPVD`."
        )

    pvt_regions: IntCellArray = (
        reservoir.regions.pvt_regions
        if reservoir.regions is not None and reservoir.regions.pvt_regions is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    return _temperature_array_from_regions(
        source, pvt_regions=pvt_regions, depth=reservoir.depth, dtype=dtype
    )


def initialize_state(
    reservoir: Reservoir,
    pvt: PVTRegions,
    *,
    deck_file: typing.Optional[DeckFile] = None,
    equilibrium: typing.Optional[EquilibriumRegions] = None,
    temperature: typing.Optional[typing.Union[Temperature, Number]] = None,
    pressure: typing.Optional[CellArray] = None,
    water_saturation: typing.Optional[CellArray] = None,
    gas_saturation: typing.Optional[CellArray] = None,
    solution_gor: typing.Optional[CellArray] = None,
    vaporized_oil_ratio: typing.Optional[CellArray] = None,
    depth_step: Number = 1.0,
    with_hysteresis: bool = False,
    dtype: npt.DTypeLike = None,
) -> State:
    """
    Build a complete `State` from EQUIL and/or explicit arrays.

    Precedence for each of `pressure`, `water_saturation`, `gas_saturation`,
    `solution_gor`, `vaporized_oil_ratio` independently:

    `Restart (not yet implemented) > explicit array/keyword > EQUIL > error`

    "Explicit array/keyword" means either the corresponding kwarg here, or
    (if the kwarg is `None`) the matching `PRESSURE`/`SWAT`/`SGAS`/`RS`/`RV`
    keyword in `deck_file`. Any field left uncovered by both is filled in
    from equilibration - which therefore only runs at all if at least one
    field needs it.

    `oil_saturation` is always derived as `1 - water_saturation -
    gas_saturation` per Step 7, never taken from an explicit array.

    :param reservoir: Reservoir geometry, rock, and region assignments.
    :param black_oil_model: PVT region tables. Must share
        `reservoir.unit_system`.
    :param deck_file: Optional parsed `DeckFile`, used as a fallback source
        for `EQUIL`/`RSVD`/`RVVD`, explicit `PRESSURE`/`SWAT`/`SGAS`/`RS`/`RV`
        arrays, and `RTEMP`/`TEMPVD` temperature - for any of these not
        supplied directly via the kwargs below.
    :param equilibrium: Parsed `EquilibriumRegions`; takes
        precedence over `deck_file`'s `EQUIL` keyword if both are given.
    :param temperature: Constant or `Temperature`; takes precedence
        over `deck_file`'s `RTEMP`/`TEMPVD`.
    :param pressure: Explicit per-cell pressure; overrides EQUIL for these
        cells only (other fields for the same cells may still come from EQUIL).
    :param water_saturation: Explicit per-cell Sw; overrides EQUIL.
    :param gas_saturation: Explicit per-cell Sg; overrides EQUIL.
    :param solution_gor: Explicit per-cell Rs; overrides EQUIL.
    :param vaporized_oil_ratio: Explicit per-cell Rv; overrides EQUIL.
    :param depth_step: RK4 step size for EQUIL pressure integration.
    :param with_hysteresis: If `True`, attach an initial `Hysteresis`
        derived from the resulting saturations.
    :returns: Fully populated `State`.
    :raises NotImplementedError: If `deck_file` has a `RESTART` keyword
        (Step 11 - not yet supported).
    :raises ValidationError: If some field is covered by neither an
        explicit array/keyword nor `EquilibriumRegions`/`deck_file`, if
        `reservoir` and `black_oil_model` unit systems disagree, or
        if saturations are physically inconsistent.
    """
    if deck_file is not None and deck_file.has("RESTART"):
        raise NotImplementedError(
            "Restart-file state initialization is not yet implemented. "
            "Supply `equilibrium` and/or explicit arrays instead."
        )

    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    unit_system = reservoir.unit_system
    if pvt.unit_system != unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({unit_system.value!r}) does not "
            "match `pvt.unit_system` "
            f"({pvt.unit_system.value!r})."
        )

    temperature_arr = _resolve_temperature(
        reservoir, deck_file, temperature, dtype=dtype
    )

    explicit: typing.Dict[str, typing.Optional[CellArray]] = {
        "pressure": pressure,
        "water_saturation": water_saturation,
        "gas_saturation": gas_saturation,
        "solution_gor": solution_gor,
        "vaporized_oil_ratio": vaporized_oil_ratio,
    }
    deck_keyword_by_field = {
        "pressure": "PRESSURE",
        "water_saturation": "SWAT",
        "gas_saturation": "SGAS",
        "solution_gor": "RS",
        "vaporized_oil_ratio": "RV",
    }
    if deck_file is not None:
        for field, keyword in deck_keyword_by_field.items():
            if explicit[field] is None:
                deck_array = deck_file.get(keyword)
                if deck_array is not None:
                    explicit[field] = typing.cast(CellArray, deck_array)

    equilibrium_arrays: typing.Optional[EquilibriumArrays] = None
    if any(v is None for v in explicit.values()):
        if equilibrium is None:
            if deck_file is not None and deck_file.get("EQUIL"):
                equilibrium = EquilibriumRegions.from_deck(deck_file)
            else:
                missing = [f for f, v in explicit.items() if v is None]
                raise ValidationError(
                    f"No explicit array/keyword and no EQUIL data available "
                    f"for: {missing}. Supply `equilibrium`, a "
                    "`deck_file` with an `EQUIL` keyword, or explicit arrays "
                    "for these fields."
                )
        equilibrium_arrays = initialize_equilibrium_state(
            reservoir,
            pvt,
            equilibrium,
            temperature_arr,
            depth_step=depth_step,
            dtype=dtype,
        )

    def _resolve(
        field: str, equilibrium_value: typing.Optional[CellArray]
    ) -> CellArray:
        value = explicit[field]
        if value is not None:
            arr = np.asarray(value, dtype=dtype)
            if arr.shape != (n_cells,):
                raise ValidationError(
                    f"Explicit `{field}` has shape {arr.shape}; expected ({n_cells},)."
                )
            return typing.cast(CellArray, arr)
        assert equilibrium_value is not None  # guaranteed by the branch above
        return equilibrium_value

    pressure_arr = _resolve(
        "pressure",
        equilibrium_arrays.pressure if equilibrium_arrays is not None else None,
    )
    water_saturation_arr = _resolve(
        "water_saturation",
        equilibrium_arrays.water_saturation if equilibrium_arrays is not None else None,
    )
    gas_saturation_arr = _resolve(
        "gas_saturation",
        equilibrium_arrays.gas_saturation if equilibrium_arrays is not None else None,
    )
    solution_gor_arr = _resolve(
        "solution_gor",
        equilibrium_arrays.solution_gor if equilibrium_arrays is not None else None,
    )
    vaporized_oil_ratio_arr = _resolve(
        "vaporized_oil_ratio",
        equilibrium_arrays.vaporized_oil_ratio
        if equilibrium_arrays is not None
        else None,
    )
    oil_bubble_point_pressure_arr = (
        equilibrium_arrays.oil_bubble_point_pressure
        if equilibrium_arrays is not None
        else pressure_arr.copy()
    )
    gas_dew_point_pressure_arr = (
        equilibrium_arrays.gas_dew_point_pressure
        if equilibrium_arrays is not None
        else np.zeros(n_cells, dtype=dtype)
    )

    saturation_sum_excess = (water_saturation_arr + gas_saturation_arr) - 1.0
    if np.any(saturation_sum_excess > 1e-6):
        raise ValidationError(
            "water_saturation + gas_saturation exceeds 1.0 in at least one "
            f"cell (max excess {float(saturation_sum_excess.max()):.3e})."
        )
    oil_saturation_arr = np.clip(
        1.0 - water_saturation_arr - gas_saturation_arr, 0.0, 1.0, dtype=dtype
    )

    pore_volumes = reservoir.pore_volumes
    pvt_regions: IntCellArray = (
        reservoir.regions.pvt_regions
        if reservoir.regions is not None and reservoir.regions.pvt_regions is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )

    oil_mass = np.zeros(n_cells, dtype=dtype)
    water_mass = np.zeros(n_cells, dtype=dtype)
    free_gas_mass = np.zeros(n_cells, dtype=dtype)
    dissolved_gas_mass_in_oil = np.zeros(n_cells, dtype=dtype)
    # Zero for standard black-oil (see State docstring); left unset here
    # rather than wired up, since Rv is not condensate-aware in v1 (see
    # module docstring wet-gas caveat).
    vaporized_oil_mass_in_gas = np.zeros(n_cells, dtype=dtype)

    for pvtnum in np.unique(pvt_regions):
        mask = pvt_regions == pvtnum
        pvt_region = pvt.region(int(pvtnum))
        static = pvt_region.static
        if static.stock_tank_oil_density is None:
            raise ValidationError(
                f"PVTNUM {pvtnum}: `stock_tank_oil_density` (DENSITY "
                "keyword) is required to assemble State masses."
            )

        rho_o_sc = static.stock_tank_oil_density
        rho_g_sc = static.stock_tank_gas_density
        rho_w_sc = static.stock_tank_water_density

        p = pressure_arr[mask]
        t = temperature_arr[mask]
        so = oil_saturation_arr[mask]
        sw = water_saturation_arr[mask]
        sg = gas_saturation_arr[mask]
        rs = solution_gor_arr[mask]
        pv = pore_volumes[mask]

        bo = pvt_region.tables.oil.formation_volume_factor(  # type: ignore[union-attr]
            pressure=p, temperature=t, solution_gor=rs
        )
        if bo is None:
            raise ValidationError(
                f"PVTNUM {pvtnum}: oil formation volume factor table is unavailable."
            )
        oil_mass[mask] = so * pv / bo * rho_o_sc

        if np.any(sg > 0.0):
            if pvt_region.tables.gas is None:
                raise ValidationError(
                    f"PVTNUM {pvtnum}: free gas saturation is present "
                    "but no gas PVT table is available."
                )

            bg = pvt_region.tables.gas.formation_volume_factor(
                pressure=p, temperature=t
            )
            if bg is None or rho_g_sc is None:
                raise ValidationError(
                    f"PVTNUM {pvtnum}: gas FVF table or "
                    "`stock_tank_gas_density` is unavailable but Sg > 0."
                )
            free_gas_mass[mask] = sg * pv / bg * rho_g_sc

        if np.any(sw > 0.0):
            if pvt_region.tables.water is None:
                raise ValidationError(
                    f"PVTNUM {pvtnum}: water saturation is present "
                    "but no water PVT table is available."
                )

            bw = pvt_region.tables.water.formation_volume_factor(
                pressure=p, temperature=t
            )
            if bw is None or rho_w_sc is None:
                raise ValidationError(
                    f"PVTNUM {pvtnum}: water FVF table or "
                    "`stock_tank_water_density` is unavailable."
                )
            water_mass[mask] = sw * pv / bw * rho_w_sc

        if rho_g_sc is not None:
            dissolved_gas_mass_in_oil[mask] = (
                rs * oil_mass[mask] * (rho_g_sc / rho_o_sc)
            )

    zeros = np.zeros(n_cells, dtype=dtype)
    hysteresis = (
        Hysteresis.from_initial_saturation(water_saturation_arr, gas_saturation_arr)
        if with_hysteresis
        else None
    )
    return State(
        pressure=pressure_arr,
        temperature=temperature_arr,
        oil_saturation=oil_saturation_arr,  # type: ignore[arg-type]
        water_saturation=water_saturation_arr,
        gas_saturation=gas_saturation_arr,
        solution_gor=solution_gor_arr,
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure_arr),
        vaporized_oil_ratio=vaporized_oil_ratio_arr,
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure_arr),
        gas_solubility_in_water=zeros,
        water_bubble_point_pressure=zeros,
        oil_mass=typing.cast(CellArray, oil_mass),
        water_mass=typing.cast(CellArray, water_mass),
        free_gas_mass=typing.cast(CellArray, free_gas_mass),
        dissolved_gas_mass_in_oil=typing.cast(CellArray, dissolved_gas_mass_in_oil),
        dissolved_gas_mass_in_water=zeros,
        vaporized_oil_mass_in_gas=typing.cast(CellArray, vaporized_oil_mass_in_gas),
        hysteresis=hysteresis,
        unit_system=unit_system,
    )
