import logging
import typing

from bores.constants import c
from bores.correlations.core import (
    PropsSI,
    clip_pressure,
    clip_temperature,
    fahrenheit_to_kelvin,
)
from bores.errors import ValidationError
from bores.typing import Number

logger = logging.getLogger(__name__)

__all__ = [
    "compute_fluid_density",
    "compute_fluid_viscosity",
    "compute_fluid_compressibility_factor",
    "compute_fluid_compressibility",
    "compute_total_fluid_compressibility",
    "compute_hydrocarbon_in_place",
]


def compute_fluid_density(pressure: Number, temperature: Number, fluid: str) -> Number:
    """
    Compute fluid density from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Density in lbm/ft³
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    density = PropsSI(
        "D",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )
    return density * c.KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET


def compute_fluid_viscosity(
    pressure: Number, temperature: Number, fluid: str
) -> Number:
    """
    Compute fluid dynamic viscosity from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Viscosity in centipoise (cP)
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    viscosity = PropsSI(
        "V",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )
    return viscosity * c.PASCAL_SECONDS_TO_CENTIPOISE


def compute_fluid_compressibility_factor(
    pressure: Number, temperature: Number, fluid: str
) -> Number:
    """
    Compute fluid compressibility factor Z from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Methane")
    :return: Compressibility factor Z (dimensionless)
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    return PropsSI(
        "Z",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )


def compute_fluid_compressibility(
    pressure: Number, temperature: Number, fluid: str
) -> Number:
    """
    Computes the isothermal compressibility of a fluid at a given pressure and temperature.

    Compressibility is defined as:

        C_f = -(1/ρ) * (dρ/dP) at constant temperature

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name supported by CoolProp (e.g., 'n-Octane')
    :return: Compressibility in psi⁻¹
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    return (
        PropsSI(
            "ISOTHERMAL_COMPRESSIBILITY",
            "P",
            clip_pressure(pressure_in_pascals, fluid),
            "T",
            clip_temperature(temperature_in_kelvin, fluid),
            fluid,
        )
        / c.PASCAL_TO_PSI
    )


def compute_total_fluid_compressibility(
    water_saturation: Number,
    oil_saturation: Number,
    water_compressibility: Number,
    oil_compressibility: Number,
    gas_saturation: typing.Optional[Number] = None,
    gas_compressibility: typing.Optional[Number] = None,
) -> Number:
    """
    Calculates the total fluid compressibility as a saturation-weighted average of
    individual phase compressibilities.

    :param water_saturation: Water saturation (fraction).
    :param oil_saturation: Oil saturation (fraction).
    :param water_compressibility: Compressibility of the water phase (psi⁻¹).
    :param oil_compressibility: Compressibility of the oil phase (psi⁻¹).
    :param gas_saturation: Optional gas saturation (fraction) for three-phase systems.
    :param gas_compressibility: Optional gas compressibility (psi⁻¹) for three-phase systems.
    :return: Total fluid compressibility (psi⁻¹).
    """
    total_fluid_compressibility = (water_saturation * water_compressibility) + (
        oil_saturation * oil_compressibility
    )
    if gas_saturation is not None and gas_compressibility is not None:
        total_fluid_compressibility += gas_saturation * gas_compressibility

    return total_fluid_compressibility


def compute_hydrocarbon_in_place(
    area: Number,
    thickness: Number,
    porosity: Number,
    phase_saturation: Number,
    formation_volume_factor: Number,
    net_to_gross_ratio: Number = 1.0,
    hydrocarbon_type: typing.Literal["oil", "gas", "water"] = "oil",
    acre_ft_to_bbl: Number = 7758.0,
    acre_ft_to_ft3: Number = 43560.0,
) -> Number:
    """
    Computes the (free) hydrocarbon (or free water) in place (HCIP or FWIP) in stock tank barrels (STB) or standard cubic feet (SCF)
    using the volumetric method.

    The formula for oil in place (OIP) is:
        OIP = 7758 * A * h * φ * S_o * N/G / B_o

    The formula for gas in place (GIP) is:
        GIP = 43560 * A * h * φ * S_g * N/G / B_g

    S_o = 1 - S_w - S_g (oil saturation)
    S_g = 1 - S_w - S_o (gas saturation)

    where:
    - OIP is the oil in place in stock tank barrels (STB).
    - GIP is the free gas in place in standard cubic feet (SCF).
    - A is the area in acres.
    - h is the thickness in feet.
    - φ is the porosity (fraction).
    - S_o is the oil saturation (fraction).
    - B_o is the formation volume factor for oil (RB/STB).
    - S_g is the gas saturation (fraction).
    - B_g is the formation volume factor for gas (RB/SCF).
    - N/G is the net-to-gross ratio (fraction).
    - 7758 is the conversion factor from acre-feet to stock tank barrels.
    - 43560 is the conversion factor from acre-feet to cubic feet.

    Note: This calculates **free** phase volumes:
    - Free oil (excludes dissolved gas)
    - Free gas (excludes solution gas in oil)
    - Free water

    Total gas = Free gas + (Oil volume x Rs)
    where Rs is the solution gas-oil ratio.

    :param area: Area in acres.
    :param thickness: Thickness in feet.
    :param porosity: Porosity as a fraction (e.g., 0.2 for 20%).
    :param phase_saturation: Phase saturation as a fraction (e.g., 0.8 for 80%).
    :param formation_volume_factor: Formation volume factor (RB/STB or RB/SCF).
    :param hydrocarbon_type: Type of hydrocarbon ("oil" or "gas").
    :return: Free hydrocarbon/water in place (OIP/WIP in STB, and GIP in SCF).
    """
    if hydrocarbon_type not in {"oil", "gas", "water"}:
        raise ValidationError(
            "Hydrocarbon type must be either 'oil', 'gas', or 'water'."
        )
    if area <= 0 or thickness <= 0:
        raise ValidationError("Area and thickness must be positive values.")
    if porosity < 0 or porosity > 1:
        raise ValidationError("Porosity must be a fraction between 0 and 1.")
    if phase_saturation < 0 or phase_saturation > 1:
        raise ValidationError("Phase saturation must be a fraction between 0 and 1.")
    if formation_volume_factor <= 0:
        raise ValidationError("Formation volume factor must be a positive value.")

    if hydrocarbon_type in {"oil", "water"}:
        # Oil in Place (OIP) calculation (May include dissolved gas in undersaturated reservoirs)
        oip = (
            acre_ft_to_bbl
            * area
            * thickness
            * porosity
            * phase_saturation
            * net_to_gross_ratio
            / formation_volume_factor
        )
        return oip

    # Free Gas in Place (GIP) calculation
    free_gip = (
        acre_ft_to_ft3
        * area
        * thickness
        * porosity
        * phase_saturation
        * net_to_gross_ratio
        / formation_volume_factor
    )
    return free_gip
