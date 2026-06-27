import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores import RelativeMobilityGrids
from bores.config import Config
from bores.constants import c
from bores.datastructures import (
    BottomHolePressure,
    BottomHolePressures,
    FormationVolumeFactors,
    Rates,
    SparseTensor,
)
from bores.grids.base import RelPermGrids
from bores.model import FluidProperties
from bores.rock_fluid.relperm import RelPermEndpoints
from bores.solvers.base import _warn_injection_rate, _warn_production_rate, to_1D_index
from bores.typing import (
    FluidPhase,
    NDimension,
    NDimensionalGrid,
    OneDimension,
    ThreeDimensions,
)
from bores.wells.base import Wells
from bores.wells.controls import ProducerRateControl
from bores.wells.core import get_pseudo_pressure_table
from bores.wells.indices import WellsIndices


@attrs.frozen
class WellRates(typing.Generic[NDimension]):
    """
    Container for all well rate quantities computed in a single explicit pass
    over all injection and production perforations.
    """

    net_water_well_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net volumetric water rate per cell (ft³/day)."""

    net_oil_well_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net volumetric oil rate per cell (ft³/day)."""

    net_gas_well_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net volumetric gas rate per cell (ft³/day)."""

    net_water_well_mass_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net water mass rate per cell (lbm/day)."""

    net_oil_well_mass_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net oil mass rate per cell (lbm/day)."""

    net_gas_well_mass_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net gas mass rate per cell (lbm/day)."""

    injection_rates: Rates[float, ThreeDimensions]
    """Per-phase volumetric injection rates (ft³/day)."""

    production_rates: Rates[float, ThreeDimensions]
    """Per-phase volumetric production rates (ft³/day)."""

    injection_mass_rates: Rates[float, ThreeDimensions]
    """Per-phase injection mass rates (lbm/day)."""

    production_mass_rates: Rates[float, ThreeDimensions]
    """Per-phase production mass rates (lbm/day)."""

    injection_fvfs: FormationVolumeFactors[float, ThreeDimensions]
    """Formation volume factors at injection perforation conditions."""

    production_fvfs: FormationVolumeFactors[float, ThreeDimensions]
    """Formation volume factors at production perforation conditions."""

    injection_bhps: BottomHolePressures[float, ThreeDimensions]
    """Effective bottom hole pressures for injection perforations (psi)."""

    production_bhps: BottomHolePressures[float, ThreeDimensions]
    """Effective bottom hole pressures for production perforations (psi)."""

    diagonal_contributions: NDimensionalGrid[OneDimension]
    """Wells' diagonal contributions to the pressure Jacobian"""

    rhs_contributions: NDimensionalGrid[OneDimension]
    """Wells' right-hand side contributions to the pressure Jacobian"""


def compute_well_rates(
    fluid_properties: FluidProperties[ThreeDimensions],
    relperm_grids: RelPermGrids[ThreeDimensions],
    relperm_endpoints: RelPermEndpoints,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    time: float,
    config: Config,
    wells_indices: WellsIndices,
    dtype: npt.DTypeLike,
) -> WellRates[ThreeDimensions]:
    """
    Compute explicit well rates for all injection and production perforations
    and return all scalar grids and per-phase tensors packed into a `WellRates` container.

    :param fluid_properties: Fluid properties container holding PVT state at
        the current pressure level.
    :param water_relative_mobility_grid: Water relative mobility grid (1/cP).
    :param oil_relative_mobility_grid: Oil relative mobility grid (1/cP).
    :param gas_relative_mobility_grid: Gas relative mobility grid (1/cP).
    :param wells: Wells container with injection and production well definitions.
    :param time: Total simulation time elapsed, this time step inclusive (s).
    :param config: Simulation configuration.
    :param wells_indices: Cache of well indices and allocation fractions.
    :param dtype: NumPy dtype for the returned grid arrays.
    :return: `WellRates` container holding the total volumetric well-rate grid,
        the six per-phase volumetric and mass rate grids, and the per-phase
        rate, mass rate, fvf, and bhp tensors. All grids use ft³/day for
        volumetric rates and lbm/day for mass rates, with positive values
        denoting injection and negative values denoting production.
    """
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET
    # Because of the radial flow formulation used in well calculations we multiply by 2π.
    # This would give around 1.127e-3 * 2π = 7.08e-3 approximately
    md_per_cp_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY * 2 * np.pi
    )
    pressure_grid = fluid_properties.pressure_grid
    grid_shape = pressure_grid.shape
    cell_count_x, cell_count_y, cell_count_z = grid_shape
    cell_count = cell_count_x * cell_count_y * cell_count_z
    water_relative_mobility_grid = relative_mobility_grids.water_relative_mobility
    oil_relative_mobility_grid = relative_mobility_grids.oil_relative_mobility
    gas_relative_mobility_grid = relative_mobility_grids.gas_relative_mobility

    net_water_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_oil_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_gas_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_water_well_mass_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_oil_well_mass_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_gas_well_mass_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    rhs_contributions = np.zeros(cell_count, dtype=dtype)
    diagonal_contributions = np.zeros(cell_count, dtype=dtype)

    injection_rates = Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )
    production_rates = Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )
    injection_mass_rates = Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )
    production_mass_rates = Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )
    injection_fvfs = FormationVolumeFactors(
        oil=SparseTensor(grid_shape, dtype=float, default=np.nan),
        water=SparseTensor(grid_shape, dtype=float, default=np.nan),
        gas=SparseTensor(grid_shape, dtype=float, default=np.nan),
    )
    production_fvfs = FormationVolumeFactors(
        oil=SparseTensor(grid_shape, dtype=float, default=np.nan),
        water=SparseTensor(grid_shape, dtype=float, default=np.nan),
        gas=SparseTensor(grid_shape, dtype=float, default=np.nan),
    )
    injection_bhps = BottomHolePressures(
        oil=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        water=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        gas=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
    )
    production_bhps = BottomHolePressures(
        oil=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        water=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        gas=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
    )

    temperature_grid = fluid_properties.temperature_grid
    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid
    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    water_viscosity_grid = fluid_properties.water_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid
    oil_formation_volume_factor_grid = fluid_properties.oil_formation_volume_factor_grid
    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid
    )
    water_bubble_point_pressure_grid = fluid_properties.water_bubble_point_pressure_grid
    gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid
    solution_gas_to_oil_ratio_grid = fluid_properties.solution_gas_to_oil_ratio_grid

    # Injection wells
    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_fluid = well.injected_fluid
        injected_phase = injected_fluid.phase
        is_gas = injected_phase == FluidPhase.GAS
        use_pseudo_pressure = is_gas and config.use_pseudo_pressure
        indices = wells_indices.injection[well.name]

        for perforation_index in indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            allocation_fraction = indices.get_allocation_fraction(perforation_index)
            cell_pressure = typing.cast(float, pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])

            phase_fvf = typing.cast(
                float,
                injected_fluid.get_formation_volume_factor(
                    pressure=cell_pressure, temperature=cell_temperature
                ),
            )
            phase_viscosity = typing.cast(
                float,
                injected_fluid.get_viscosity(
                    pressure=cell_pressure, temperature=cell_temperature
                ),
            )

            if is_gas:
                # Only needed for oil and water
                phase_compressibility = None
                phase_mobility = typing.cast(
                    float,
                    injected_fluid.get_mobility(
                        relative_permeability=relperm_grids.krg[i, j, k],
                        endpoint_relative_permeability=relperm_endpoints.gas,
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        viscosity=phase_viscosity,
                        eta=1e-3,
                    ),
                )
            else:
                phase_compressibility = typing.cast(
                    float,
                    injected_fluid.get_compressibility(
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        bubble_point_pressure=water_bubble_point_pressure_grid[i, j, k],
                        gas_formation_volume_factor=gas_formation_volume_factor_grid[
                            i, j, k
                        ],
                        gas_solubility_in_water=gas_solubility_in_water_grid[i, j, k],
                    ),
                )
                phase_mobility = typing.cast(
                    float,
                    injected_fluid.get_mobility(
                        relative_permeability=relperm_grids.krw[i, j, k],
                        endpoint_relative_permeability=relperm_endpoints.water,
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        viscosity=phase_viscosity,
                        eta=1e-3,
                    ),
                )

            control = well.get_control(
                pressure=cell_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_viscosity=phase_viscosity,
                phase_mobility=phase_mobility,
                fluid=injected_fluid,
                fluid_compressibility=phase_compressibility,
                use_pseudo_pressure=use_pseudo_pressure,
                phase_fvf=phase_fvf,
                allocation_fraction=allocation_fraction,
                pvt_tables=None,
            )
            flow_rate = control.rate
            effective_bhp = control.bhp

            if flow_rate < 0.0 and config.warn_well_anomalies:
                _warn_injection_rate(
                    injection_rate=flow_rate,
                    well_name=well.name,
                    time=time,
                    cell=(i, j, k),
                    rate_unit="ft³/day" if is_gas else "bbls/day",
                )

            # When bhp returned is same as cell pressure, there no drawdown
            # so no flow. Hence, bhp should be unset, and no rhs or diagonal addition should be made
            can_flow = effective_bhp != cell_pressure
            cell_idx = to_1D_index(
                i=i,
                j=j,
                k=k,
                cell_count_x=cell_count_x,
                cell_count_y=cell_count_y,
                cell_count_z=cell_count_z,
            )
            if not is_gas:
                flow_rate *= bbl_to_ft3

            phase_density = typing.cast(
                float,
                injected_fluid.get_density(
                    pressure=cell_pressure, temperature=cell_temperature
                ),
            )
            if can_flow:
                # Compute mass-based mobility for injected phase
                mass_mobility = phase_density * phase_mobility
                if not use_pseudo_pressure:
                    mass_transmissibility = (
                        well_index * mass_mobility * md_per_cp_to_ft2_per_psi_per_day
                    )
                    diagonal_contributions[cell_idx] += mass_transmissibility
                    rhs_contributions[cell_idx] += mass_transmissibility * effective_bhp
                else:
                    # Pseudo pressure linearization
                    mass_transmissibility = (
                        well_index
                        * mass_mobility
                        * phase_viscosity
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    _, pseudo_pressure_table = get_pseudo_pressure_table(
                        fluid=injected_fluid,
                        temperature=cell_temperature,
                        use_pseudo_pressure=True,
                        pvt_tables=None,
                    )
                    assert pseudo_pressure_table is not None
                    dm_dp = pseudo_pressure_table.gradient(cell_pressure)
                    diagonal_contributions[cell_idx] += mass_transmissibility * dm_dp
                    rhs_contributions[cell_idx] += (
                        mass_transmissibility * dm_dp * effective_bhp
                    )

            if is_gas:
                net_gas_well_rate_grid[i, j, k] += flow_rate
                net_gas_well_mass_rate_grid[i, j, k] += flow_rate * phase_density
                injection_rates.gas[i, j, k] = flow_rate
                injection_mass_rates.gas[i, j, k] = flow_rate * phase_density
                injection_fvfs.gas[i, j, k] = phase_fvf
                if can_flow:
                    injection_bhps.gas[i, j, k] = effective_bhp
            else:
                net_water_well_rate_grid[i, j, k] += flow_rate
                net_water_well_mass_rate_grid[i, j, k] += flow_rate * phase_density
                injection_rates.water[i, j, k] += flow_rate
                injection_mass_rates.water[i, j, k] += flow_rate * phase_density
                injection_fvfs.water[i, j, k] = phase_fvf
                if can_flow:
                    injection_bhps.water[i, j, k] = effective_bhp

    # Production wells
    for well in wells.production_wells:
        if not well.is_open:
            continue

        is_producer_control = isinstance(well.control, ProducerRateControl)
        indices = wells_indices.production[well.name]

        for perforation_index in indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            allocation_fraction = indices.get_allocation_fraction(perforation_index)
            cell_pressure = typing.cast(float, pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])

            shared_bhp: typing.Optional[float] = None
            if is_producer_control:
                # Compute shared BHP once, before the phase loop
                context = well.control.build_context(  # type: ignore
                    produced_fluids=well.produced_fluids,
                    oil_mobility=oil_relative_mobility_grid[i, j, k],
                    water_mobility=water_relative_mobility_grid[i, j, k],
                    gas_mobility=gas_relative_mobility_grid[i, j, k],
                    oil_fvf=oil_formation_volume_factor_grid[i, j, k],
                    water_fvf=water_formation_volume_factor_grid[i, j, k],
                    gas_fvf=gas_formation_volume_factor_grid[i, j, k],
                    oil_compressibility=oil_compressibility_grid[i, j, k],
                    water_compressibility=water_compressibility_grid[i, j, k],
                    gas_compressibility=gas_compressibility_grid[i, j, k],
                    oil_viscosity=oil_viscosity_grid[i, j, k],
                    water_viscosity=water_viscosity_grid[i, j, k],
                    gas_viscosity=gas_viscosity_grid[i, j, k],
                )
                if context:
                    shared_bhp = well.control.compute_bhp(  # type: ignore
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        well_index=well_index,
                        allocation_fraction=allocation_fraction,
                        **context,
                    )

            # Pre-compute alpha factors for dissolved gas contributions
            alpha_solution_gor = (
                typing.cast(float, solution_gas_to_oil_ratio_grid[i, j, k])
                * typing.cast(float, gas_formation_volume_factor_grid[i, j, k])
                / (
                    bbl_to_ft3
                    * typing.cast(float, oil_formation_volume_factor_grid[i, j, k])
                )
            )
            alpha_gas_solubility_in_water = (
                typing.cast(float, gas_solubility_in_water_grid[i, j, k])
                * typing.cast(float, gas_formation_volume_factor_grid[i, j, k])
                / (
                    bbl_to_ft3
                    * typing.cast(float, water_formation_volume_factor_grid[i, j, k])
                )
            )
            water_flow_rate = 0.0
            oil_flow_rate = 0.0
            gas_flow_rate = 0.0
            water_mass_flow_rate = 0.0
            oil_mass_flow_rate = 0.0
            gas_mass_flow_rate = 0.0
            water_fvf = np.nan
            oil_fvf = np.nan
            gas_fvf = np.nan
            water_effective_bhp = np.nan
            oil_effective_bhp = np.nan
            gas_effective_bhp = np.nan

            for produced_fluid in well.produced_fluids:
                produced_phase = produced_fluid.phase
                is_gas = produced_phase == FluidPhase.GAS
                is_water = produced_phase == FluidPhase.WATER
                use_pseudo_pressure = is_gas and config.use_pseudo_pressure

                if is_gas:
                    phase_mobility = typing.cast(
                        float, gas_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, gas_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, gas_formation_volume_factor_grid[i, j, k]
                    )
                    phase_viscosity = typing.cast(float, gas_viscosity_grid[i, j, k])
                elif is_water:
                    phase_mobility = typing.cast(
                        float, water_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, water_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, water_formation_volume_factor_grid[i, j, k]
                    )
                    phase_viscosity = typing.cast(float, water_viscosity_grid[i, j, k])
                else:
                    phase_mobility = typing.cast(
                        float, oil_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, oil_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, oil_formation_volume_factor_grid[i, j, k]
                    )
                    phase_viscosity = typing.cast(float, oil_viscosity_grid[i, j, k])

                if is_producer_control and shared_bhp is not None:
                    # BHP already computed above. Each phase uses its own mobility
                    # but all phases share the same BHP.
                    flow_rate = well.control.get_flow_rate(
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        well_index=well_index,
                        fluid=produced_fluid,
                        phase_fvf=phase_fvf,
                        shared_bhp=shared_bhp,
                        phase_viscosity=phase_viscosity,
                        phase_mobility=phase_mobility,
                        allocation_fraction=allocation_fraction,
                        is_active=well.is_open,
                        use_pseudo_pressure=use_pseudo_pressure,
                        fluid_compressibility=phase_compressibility,
                        pvt_tables=config.pvt_tables,
                    )
                    effective_bhp = shared_bhp
                else:
                    control = well.get_control(
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        well_index=well_index,
                        phase_viscosity=phase_viscosity,
                        phase_mobility=phase_mobility,
                        fluid=produced_fluid,
                        fluid_compressibility=phase_compressibility,
                        use_pseudo_pressure=use_pseudo_pressure,
                        phase_fvf=phase_fvf,
                        allocation_fraction=allocation_fraction,
                        pvt_tables=config.pvt_tables,
                    )
                    flow_rate = control.rate
                    effective_bhp = control.bhp

                if flow_rate > 0.0 and config.warn_well_anomalies:
                    _warn_production_rate(
                        production_rate=flow_rate,
                        well_name=well.name,
                        time=time,
                        cell=(i, j, k),
                        rate_unit="ft³/day" if is_gas else "bbls/day",
                    )

                can_flow = effective_bhp != cell_pressure
                cell_idx = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                if not is_gas:
                    flow_rate *= bbl_to_ft3

                if can_flow:
                    # Compute mass-based mobility for this phase including dissolved gas
                    if is_gas:
                        phase_density = typing.cast(float, gas_density_grid[i, j, k])
                        phase_mass_mobility = phase_density * phase_mobility
                    elif is_water:
                        phase_density = typing.cast(float, water_density_grid[i, j, k])
                        phase_mass_mobility = (
                            phase_density
                            + gas_density_grid[i, j, k] * alpha_gas_solubility_in_water
                        ) * phase_mobility
                    else:  # oil
                        phase_density = typing.cast(float, oil_density_grid[i, j, k])
                        phase_mass_mobility = (
                            phase_density
                            + gas_density_grid[i, j, k] * alpha_solution_gor
                        ) * phase_mobility

                    phase_mass_transmissibility = (
                        well_index
                        * phase_mass_mobility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not use_pseudo_pressure:
                        diagonal_contributions[cell_idx] += phase_mass_transmissibility
                        rhs_contributions[cell_idx] += (
                            phase_mass_transmissibility * effective_bhp
                        )
                    else:
                        mass_pseudo_transmissibility = (
                            phase_mass_transmissibility * phase_viscosity
                        )
                        _, pseudo_pressure_table = get_pseudo_pressure_table(
                            fluid=produced_fluid,
                            temperature=cell_temperature,
                            use_pseudo_pressure=True,
                            pvt_tables=config.pvt_tables,
                        )
                        assert pseudo_pressure_table is not None
                        dm_dp = pseudo_pressure_table.gradient(cell_pressure)
                        diagonal_contributions[cell_idx] += (
                            mass_pseudo_transmissibility * dm_dp
                        )
                        rhs_contributions[cell_idx] += (
                            mass_pseudo_transmissibility * dm_dp * effective_bhp
                        )

                if is_gas:
                    phase_density = typing.cast(float, gas_density_grid[i, j, k])
                    gas_flow_rate += flow_rate
                    gas_mass_flow_rate += flow_rate * phase_density
                    gas_fvf = phase_fvf
                    if can_flow:
                        gas_effective_bhp = effective_bhp
                    net_gas_well_rate_grid[i, j, k] += flow_rate
                    net_gas_well_mass_rate_grid[i, j, k] += flow_rate * phase_density
                elif is_water:
                    phase_density = typing.cast(float, water_density_grid[i, j, k])
                    water_flow_rate += flow_rate
                    water_mass_flow_rate += flow_rate * phase_density
                    water_fvf = phase_fvf
                    if can_flow:
                        water_effective_bhp = effective_bhp
                    net_water_well_rate_grid[i, j, k] += flow_rate
                    net_water_well_mass_rate_grid[i, j, k] += flow_rate * phase_density
                else:
                    phase_density = typing.cast(float, oil_density_grid[i, j, k])
                    oil_flow_rate += flow_rate
                    oil_mass_flow_rate += flow_rate * phase_density
                    oil_fvf = phase_fvf
                    if can_flow:
                        oil_effective_bhp = effective_bhp
                    net_oil_well_rate_grid[i, j, k] += flow_rate
                    net_oil_well_mass_rate_grid[i, j, k] += flow_rate * phase_density

            production_rates[i, j, k] = (water_flow_rate, oil_flow_rate, gas_flow_rate)
            production_mass_rates[i, j, k] = (
                water_mass_flow_rate,
                oil_mass_flow_rate,
                gas_mass_flow_rate,
            )
            production_fvfs[i, j, k] = (water_fvf, oil_fvf, gas_fvf)
            production_bhps[i, j, k] = (
                water_effective_bhp,
                oil_effective_bhp,
                gas_effective_bhp,
            )

    return WellRates(
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        net_water_well_mass_rate_grid=net_water_well_mass_rate_grid,
        net_oil_well_mass_rate_grid=net_oil_well_mass_rate_grid,
        net_gas_well_mass_rate_grid=net_gas_well_mass_rate_grid,
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_mass_rates=injection_mass_rates,
        production_mass_rates=production_mass_rates,
        injection_fvfs=injection_fvfs,
        production_fvfs=production_fvfs,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        rhs_contributions=rhs_contributions,
        diagonal_contributions=diagonal_contributions,
    )
