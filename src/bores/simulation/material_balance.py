import attrs
import numpy as np
from typing_extensions import Self

from bores.models import FluidProperties
from bores.typing import ThreeDimensions

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
) -> MaterialBalanceErrors:
    """
    Compute per-phase material balance errors for a timestep using the mass formulation.

    The MBE for each phase is:

        Absolute MBE = (mass_new - mass_old)

        Relative MBE = (mass_new - mass_old) / mass_old

    :param current_fluid_properties: Fluid properties at end of time step.
    :param previous_fluid_properties: Fluid properties at start of time step.
    :return: `MaterialBalanceErrors` instance with mass-based errors in lbm.
    """
    # Oil mass
    current_oil_mass = float(np.sum(current_fluid_properties.oil_mass_grid))
    previous_oil_mass = float(np.sum(previous_fluid_properties.oil_mass_grid))
    oil_mass_change = current_oil_mass - previous_oil_mass

    absolute_oil_mbe = oil_mass_change
    reference_oil = max(abs(previous_oil_mass), 1.0)
    relative_oil_mbe = absolute_oil_mbe / reference_oil

    # Water mass
    current_water_mass = float(np.sum(current_fluid_properties.water_mass_grid))
    previous_water_mass = float(np.sum(previous_fluid_properties.water_mass_grid))
    water_mass_change = current_water_mass - previous_water_mass

    absolute_water_mbe = water_mass_change
    reference_water = max(abs(previous_water_mass), 1.0)
    relative_water_mbe = absolute_water_mbe / reference_water

    # Gas mass
    current_gas_mass = float(np.sum(current_fluid_properties.total_gas_mass_grid))
    previous_gas_mass = float(np.sum(previous_fluid_properties.total_gas_mass_grid))
    gas_mass_change = current_gas_mass - previous_gas_mass

    absolute_gas_mbe = gas_mass_change
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
