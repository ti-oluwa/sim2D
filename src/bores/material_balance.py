import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.constants import c
from bores.datastructures import Rates
from bores.models import FluidProperties, RockProperties
from bores.types import NDimensionalGrid, ThreeDimensions

__all__ = ["MaterialBalanceErrors", "compute_material_balance_errors"]


@attrs.frozen(slots=True)
class MaterialBalanceErrors:
    """
    Material balance errors computed at the end of a time step.

    All quantities are in pounds-mass (lb). Relative errors are dimensionless
    fractions; multiply by 100 for percent.

    The gas phase accounts for free gas plus solution gas dissolved in oil
    (via Rs, SCF/STB) and in water (via Rsw, SCF/STB), all converted to a
    mass basis using the grid-cell gas density (lb/ft³).
    """

    absolute_oil_mbe: float
    """
    Absolute oil material balance error (lb).

    Defined as (mass_oil_new - mass_oil_old) - net_oil_mass_inflow.
    Positive means more oil mass accumulated than the net inflow accounts for;
    negative means a deficit.
    """

    absolute_water_mbe: float
    """
    Absolute water material balance error (lb).

    Analogous to `absolute_oil_mbe` for the water phase.
    """

    absolute_gas_mbe: float
    """
    Absolute gas material balance error (lb).

    Covers free gas plus dissolved gas in both oil and water phases.
    """

    total_absolute_mbe: float
    """
    Sum of absolute oil, water, and gas material balance errors (lb).
    """

    relative_oil_mbe: float
    """
    Oil MBE normalised by the previous-step oil mass in place (dimensionless fraction).

    `relative_oil_mbe = absolute_oil_mbe / max(|previous_oil_mass|, 1)`.
    Multiply by 100 for percent.
    """

    relative_water_mbe: float
    """
    Water MBE normalised by the previous-step water mass in place (dimensionless fraction).
    """

    relative_gas_mbe: float
    """
    Gas MBE normalised by the previous-step total gas mass in place (dimensionless fraction).
    """

    total_relative_mbe: float
    """
    Total MBE normalised by the sum of previous-step phase mass references (dimensionless fraction).
    """

    @classmethod
    def null(cls) -> Self:
        return cls(
            absolute_oil_mbe=0,
            absolute_gas_mbe=0,
            absolute_water_mbe=0,
            total_absolute_mbe=0,
            relative_oil_mbe=0,
            relative_gas_mbe=0,
            relative_water_mbe=0,
            total_relative_mbe=0,
        )


def compute_material_balance_errors(
    current_fluid_properties: FluidProperties[ThreeDimensions],
    previous_fluid_properties: FluidProperties[ThreeDimensions],
    rock: RockProperties[ThreeDimensions],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    time_step_size: float,
    injection_mass_rates: typing.Optional[Rates[float, ThreeDimensions]] = None,
    production_mass_rates: typing.Optional[Rates[float, ThreeDimensions]] = None,
) -> MaterialBalanceErrors:
    """
    Compute per-phase material balance errors for a timestep using the mass formulation.

    The MBE for each phase is:

        MBE = (mass_new - mass_old) - net_mass_inflow * dt

    :param current_fluid_properties: Fluid properties at end of time step.
    :param previous_fluid_properties: Fluid properties at start of time step.
    :param rock: Rock properties (porosity, NTG).
    :param thickness_grid: Cell thickness grid (ft).
    :param cell_dimension: (dx, dy) in ft.
    :param injection_mass_rates: Injection rate sparse tensors (lbm/day, reservoir condition).
    :param production_mass_rates: Production rate sparse tensors (lbm/day, reservoir condition).
    :param time_step_size: Time step size in seconds.
    :return: `MaterialBalanceErrors` instance with mass-based errors in lbm.
    """
    cell_size_x, cell_size_y = cell_dimension
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET

    # Cell pore volume in ft³
    pore_volume_grid = (
        rock.porosity_grid
        * rock.net_to_gross_grid
        * thickness_grid
        * cell_size_x
        * cell_size_y
    )

    # Oil mass = oil_density * So * phi * V
    current_oil_mass = float(
        np.sum(
            pore_volume_grid
            * current_fluid_properties.oil_effective_density_grid
            * current_fluid_properties.oil_saturation_grid
        )
    )
    previous_oil_mass = float(
        np.sum(
            pore_volume_grid
            * previous_fluid_properties.oil_effective_density_grid
            * previous_fluid_properties.oil_saturation_grid
        )
    )
    oil_mass_change = current_oil_mass - previous_oil_mass

    # Net mass oil inflow: injection - production (lbm)
    # Well rates are volumetric (ft³/day, res. condition); multiply by current
    # oil density to get lb/day. Use current density as the best estimate of the
    # density at which fluid entered/left the reservoir this step.
    net_oil_mass_inflow = 0.0
    if injection_mass_rates is not None:
        for key in injection_mass_rates.oil:
            net_oil_mass_inflow += float(injection_mass_rates.oil[key])

    if production_mass_rates is not None:
        for key in production_mass_rates.oil:
            net_oil_mass_inflow += float(production_mass_rates.oil[key])

    net_oil_mass_inflow *= time_step_in_days  # lb

    absolute_oil_mbe = oil_mass_change - net_oil_mass_inflow
    reference_oil = max(abs(previous_oil_mass), 1.0)
    relative_oil_mbe = absolute_oil_mbe / reference_oil

    # Water mass = water_density * Sw * phi * V
    current_water_mass = float(
        np.sum(
            pore_volume_grid
            * current_fluid_properties.water_density_grid
            * current_fluid_properties.water_saturation_grid
        )
    )
    previous_water_mass = float(
        np.sum(
            pore_volume_grid
            * previous_fluid_properties.water_density_grid
            * previous_fluid_properties.water_saturation_grid
        )
    )
    water_mass_change = current_water_mass - previous_water_mass

    net_water_mass_inflow = 0.0
    if injection_mass_rates is not None:
        for key in injection_mass_rates.water:
            net_water_mass_inflow += float(injection_mass_rates.water[key])

    if production_mass_rates is not None:
        for key in production_mass_rates.water:
            net_water_mass_inflow += float(production_mass_rates.water[key])

    net_water_mass_inflow *= time_step_in_days  # lbm

    absolute_water_mbe = water_mass_change - net_water_mass_inflow
    reference_water = max(abs(previous_water_mass), 1.0)
    relative_water_mbe = absolute_water_mbe / reference_water

    # GAS (total: free + dissolved in oil + dissolved in water)
    # M_g = phi*V * (gas_density*Sg + oil_density*alpha_Rs*So + water_density*alpha_Rsw*Sw)
    # alpha_Rs  = Rs * Bg / Bo   [dimensionless, reservoir vol. of dissolved gas per res. vol. oil]
    # alpha_Rsw = Rsw * Bg / Bw   [same for water]
    def _safe_divide(a: np.ndarray, b: np.ndarray, floor: float = 1e-30) -> np.ndarray:
        return a / np.maximum(b, floor)

    # Current conditions
    current_gas_density = current_fluid_properties.gas_density_grid
    current_oil_density = current_fluid_properties.oil_effective_density_grid
    current_water_density = current_fluid_properties.water_density_grid
    current_gas_saturation = current_fluid_properties.gas_saturation_grid
    current_oil_saturation = current_fluid_properties.oil_saturation_grid
    current_water_saturation = current_fluid_properties.water_saturation_grid
    current_solution_gor = current_fluid_properties.solution_gas_to_oil_ratio_grid
    current_gas_solubility_in_water = (
        current_fluid_properties.gas_solubility_in_water_grid
    )
    # Bg in bbl/SCF; Bo and Bw in bbl/STB - ratio Bg/Bo and Bg/Bw is dimensionless
    current_gas_fvf = current_fluid_properties.gas_formation_volume_factor_grid
    current_oil_fvf = current_fluid_properties.oil_formation_volume_factor_grid
    current_water_fvf = current_fluid_properties.water_formation_volume_factor_grid
    current_alpha_solution_gor = current_solution_gor * _safe_divide(
        current_gas_fvf, current_oil_fvf * bbl_to_ft3
    )
    current_alpha_gas_solubility_in_water = (
        current_gas_solubility_in_water
        * _safe_divide(current_gas_fvf, current_water_fvf * bbl_to_ft3)
    )

    current_gas_mass = float(
        np.sum(
            pore_volume_grid
            * (
                current_gas_density * current_gas_saturation
                + current_oil_density
                * current_alpha_solution_gor
                * current_oil_saturation
                + current_water_density
                * current_alpha_gas_solubility_in_water
                * current_water_saturation
            )
        )
    )

    # Previous conditions
    previous_gas_density = previous_fluid_properties.gas_density_grid
    previous_oil_density = previous_fluid_properties.oil_effective_density_grid
    previous_water_density = previous_fluid_properties.water_density_grid
    previous_gas_saturation = previous_fluid_properties.gas_saturation_grid
    previous_oil_saturation = previous_fluid_properties.oil_saturation_grid
    previous_water_saturation = previous_fluid_properties.water_saturation_grid
    previous_solution_gor = previous_fluid_properties.solution_gas_to_oil_ratio_grid
    previous_gas_solubility_in_water = (
        previous_fluid_properties.gas_solubility_in_water_grid
    )
    previous_gas_fvf = previous_fluid_properties.gas_formation_volume_factor_grid
    previous_oil_fvf = previous_fluid_properties.oil_formation_volume_factor_grid
    previous_water_fvf = previous_fluid_properties.water_formation_volume_factor_grid
    previous_alpha_solution_gor = previous_solution_gor * _safe_divide(
        previous_gas_fvf, previous_oil_fvf * bbl_to_ft3
    )
    previous_alpha_gas_solubility_in_water = (
        previous_gas_solubility_in_water
        * _safe_divide(previous_gas_fvf, previous_water_fvf * bbl_to_ft3)
    )

    previous_gas_mass = float(
        np.sum(
            pore_volume_grid
            * (
                previous_gas_density * previous_gas_saturation
                + previous_oil_density
                * previous_alpha_solution_gor
                * previous_oil_saturation
                + previous_water_density
                * previous_alpha_gas_solubility_in_water
                * previous_water_saturation
            )
        )
    )
    gas_mass_change = current_gas_mass - previous_gas_mass

    # Net mass gas inflow: includes free gas from the gas well rate stream, plus
    # solution gas carried by produced/injected oil and water.
    net_gas_mass_inflow = 0.0

    if injection_mass_rates is not None:
        for key in injection_mass_rates.gas:
            net_gas_mass_inflow += float(injection_mass_rates.gas[key])

        # Solution gas carried in oil stream
        for key in injection_mass_rates.oil:
            i, j, k = key
            alpha_solution_gor = float(current_alpha_solution_gor[i, j, k])
            net_gas_mass_inflow += (
                float(injection_mass_rates.oil[key]) * alpha_solution_gor
            )

        # Solution gas carried in water stream
        for key in injection_mass_rates.water:
            i, j, k = key
            alpha_gas_solubility_in_water = float(
                current_alpha_gas_solubility_in_water[i, j, k]
            )
            net_gas_mass_inflow += (
                float(injection_mass_rates.water[key]) * alpha_gas_solubility_in_water
            )

    if production_mass_rates is not None:
        for key in production_mass_rates.gas:
            net_gas_mass_inflow += float(production_mass_rates.gas[key])

        for key in production_mass_rates.oil:
            i, j, k = key
            alpha_solution_gor = float(current_alpha_solution_gor[i, j, k])
            net_gas_mass_inflow += (
                float(production_mass_rates.oil[key]) * alpha_solution_gor
            )

        for key in production_mass_rates.water:
            i, j, k = key
            alpha_gas_solubility_in_water = float(
                current_alpha_gas_solubility_in_water[i, j, k]
            )
            net_gas_mass_inflow += (
                float(production_mass_rates.water[key]) * alpha_gas_solubility_in_water
            )

    net_gas_mass_inflow *= time_step_in_days  # lbm

    absolute_gas_mbe = gas_mass_change - net_gas_mass_inflow
    reference_gas = max(abs(previous_gas_mass), 1.0)
    relative_gas_mbe = absolute_gas_mbe / reference_gas

    total_absolute_mbe = absolute_oil_mbe + absolute_water_mbe + absolute_gas_mbe
    total_reference = max(
        abs(previous_oil_mass) + abs(previous_water_mass) + abs(previous_gas_mass),
        1.0,
    )
    total_relative_mbe = total_absolute_mbe / total_reference
    return MaterialBalanceErrors(
        absolute_oil_mbe=absolute_oil_mbe,
        absolute_water_mbe=absolute_water_mbe,
        absolute_gas_mbe=absolute_gas_mbe,
        total_absolute_mbe=total_absolute_mbe,
        relative_oil_mbe=relative_oil_mbe,
        relative_water_mbe=relative_water_mbe,
        relative_gas_mbe=relative_gas_mbe,
        total_relative_mbe=total_relative_mbe,
    )
