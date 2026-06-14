"""Utils for building N-Dimensional cartesian grids for reservoir PVT properties."""

import typing

import numba  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import DTypeLike

from bores.correlations.arrays import (
    compute_fluid_compressibility,
    compute_fluid_density,
    compute_fluid_viscosity,
    compute_gas_compressibility,
    compute_gas_compressibility_factor,
    compute_gas_density,
    compute_gas_formation_volume_factor,
    compute_gas_free_water_formation_volume_factor,
    compute_gas_gravity_from_density,
    compute_gas_molecular_weight,
    compute_gas_solubility_in_water,
    compute_gas_to_oil_ratio,
    compute_gas_viscosity,
    compute_live_oil_density,
    compute_miscibility_transition_factor,
    compute_oil_api_gravity,
    compute_oil_bubble_point_pressure,
    compute_oil_compressibility,
    compute_oil_formation_volume_factor,
    compute_oil_specific_gravity,
    compute_oil_viscosity,
    compute_todd_longstaff_effective_density,
    compute_todd_longstaff_effective_viscosity,
    compute_total_fluid_compressibility,
    compute_water_bubble_point_pressure,
    compute_water_compressibility,
    compute_water_density,
    compute_water_formation_volume_factor,
    compute_water_viscosity,
    estimate_solution_gor,
)
from bores.typing import NDimension, NDimensionalGrid, NumberOrArray

__all__ = [
    "build_estimated_solution_gas_to_oil_ratio_grid",
    "build_fluid_compressibility_grid",
    "build_fluid_density_grid",
    "build_fluid_viscosity_grid",
    "build_gas_compressibility_factor_grid",
    "build_gas_gravity_from_density_grid",
    "build_oil_api_gravity_grid",
    "build_oil_formation_volume_factor_grid",
    "build_oil_specific_gravity_grid",
    "build_total_fluid_compressibility_grid",
]


def build_fluid_viscosity_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    fluid: typing.Union[str, str],
) -> NDimensionalGrid[NDimension]:
    """
    Builds a N-Dimensional grid of fluid viscosities.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute viscosity (e.g., "CO2", "water") supported by `CoolProp`.
    :return: N-Dimensional array of fluid viscosities (cP) corresponding to each grid cell.
    """
    result = compute_fluid_viscosity(
        pressure=pressure_grid, temperature=temperature_grid, fluid=fluid
    )
    return result  # type: ignore[return-value]


def build_fluid_compressibility_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    fluid: str,
) -> NDimensionalGrid[NDimension]:
    """
    Builds a N-Dimensional grid of fluid compressibilities.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute compressibility (e.g., "CO2", "water") supported by `CoolProp`.
    :return: N-Dimensional array of fluid compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    result = compute_fluid_compressibility(
        pressure=pressure_grid, temperature=temperature_grid, fluid=fluid
    )
    return result  # type: ignore[return-value]


def build_fluid_density_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    fluid: str,
    dtype: DTypeLike = np.float64,
) -> NDimensionalGrid[NDimension]:
    """
    Builds a N-Dimensional grid of fluid densities.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute density (e.g., "CO2", "water") supported by `CoolProp`.
    :return: N-Dimensional array of fluid densities (lbm/ft³) corresponding to each grid cell.
    """
    result = compute_fluid_density(
        pressure=pressure_grid, temperature=temperature_grid, fluid=fluid
    )
    return result  # type: ignore[return-value]


def build_gas_gravity_from_density_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    density_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    omputes a N-Dimensional grid of gas gravity based on pressure, temperature, and gas den.

    The gas gravity is computed using the pressure and temperature conditions.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param density_grid: N-Dimensional array of gas density values (lbm/ft³) corresponding to each grid cell.
    :return: N-Dimensional array of gas gravity values (dimensionless) corresponding to each grid cell.
    """
    result = compute_gas_gravity_from_density(
        pressure=pressure_grid,
        temperature=temperature_grid,
        density=density_grid,
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_total_fluid_compressibility_grid(
    oil_saturation_grid: NDimensionalGrid[NDimension],
    oil_compressibility_grid: NDimensionalGrid[NDimension],
    water_saturation_grid: NDimensionalGrid[NDimension],
    water_compressibility_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional array of total fluid compressibilities.

    The total fluid compressibility is defined as:
        C_f_total = (S_w * C_w) + (S_o * C_o) + (S_g * C_g)

    where:
        - C_f_total is the total fluid compressibility (psi⁻¹)
        - S_w is the saturation of water (fraction)
        - C_w is the compressibility of water (psi⁻¹)
        - S_o is the saturation of oil (fraction)
        - C_o is the compressibility of oil (psi⁻¹)
        - S_g is the saturation of the gas (fraction)
        - C_g is the compressibility of the gas (psi⁻¹)

    :param water_saturation_grid: N-Dimensional array of water saturation values (fraction)
    :param oil_saturation_grid: N-Dimensional array of oil saturation values (fraction)
    :param water_compressibility_grid: N-Dimensional array of water compressibility values (psi⁻¹)
    :param oil_compressibility_grid: N-Dimensional array of oil compressibility values (psi⁻¹)
    :param gas_saturation_grid: Optional N-Dimensional array of gas saturation values (fraction)
    :param gas_compressibility_grid: Optional N-Dimensional array of gas compressibility values (psi⁻¹)

    :return: N-Dimensional array of total fluid compressibility values (psi⁻¹) corresponding to each grid cell
    """
    result = compute_total_fluid_compressibility(
        water_saturation=water_saturation_grid,
        oil_saturation=oil_saturation_grid,
        water_compressibility=water_compressibility_grid,
        oil_compressibility=oil_compressibility_grid,
        gas_saturation=gas_saturation_grid,
        gas_compressibility=gas_compressibility_grid,
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_oil_api_gravity_grid(
    oil_specific_gravity_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil API gravities based on oil density.

    The API gravity is computed using the formula:
    API Gravity = (141.5 / Specific Gravity) - 131.5

    :param oil_specific_gravity_grid: N-Dimensional array of oil specific gravity values (dimensionless).
    :return: N-Dimensional array of oil API gravity values (dimensionless) corresponding to each grid cell.
    """
    result = compute_oil_api_gravity(oil_specific_gravity=oil_specific_gravity_grid)
    return result  # type: ignore[return-value]


def build_oil_specific_gravity_grid(
    oil_density_grid: NDimensionalGrid[NDimension],
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    oil_compressibility_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil specific gravities.

    The specific gravity is computed as the ratio of the oil density to the standard density of water (1000 kg/m³).

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param oil_density_grid: N-Dimensional array of oil density values (lbm/ft³).
    :param oil_compressibility_grid: N-Dimensional array of oil compressibility values (psi⁻¹).
    :return: N-Dimensional array of oil specific gravity (dimensionless) corresponding to each grid cell.
    """
    result = compute_oil_specific_gravity(
        oil_density=oil_density_grid,
        pressure=pressure_grid,
        temperature=temperature_grid,
        oil_compressibility=oil_compressibility_grid,
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_gas_compressibility_factor_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas compressibility factors.

    The compressibility factor is computed using the gas gravity and the pressure and temperature conditions.

    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param h2s_mole_fraction: Mole fraction of H₂S in the gas mixture (default: 0.0).
    :param co2_mole_fraction: Mole fraction of CO₂ in the gas mixture (default: 0.0).
    :param n2_mole_fraction: Mole fraction of N₂ in the gas mixture (default: 0.0).
    :return: N-Dimensional array of gas compressibility factors (dimensionless) corresponding to each grid cell.
    """
    result = compute_gas_compressibility_factor(
        pressure=pressure_grid,
        temperature=temperature_grid,
        gas_gravity=gas_gravity_grid,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
        method="dak",
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_oil_formation_volume_factor_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    oil_specific_gravity_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    solution_gas_to_oil_ratio_grid: NDimensionalGrid[NDimension],
    oil_compressibility_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil formation volume factors.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of oil bubble point pressures (psi) corresponding to each grid cell.
    :param oil_specific_gravity_grid: N-Dimensional array of oil specific gravity values (dimensionless),
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param solution_gas_to_oil_ratio_grid: N-Dimensional array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param oil_compressibility_grid: N-Dimensional array of oil compressibility values (psi⁻¹) representing the compressibility of oil.
    :return: N-Dimensional array of oil formation volume factors (bbl/STB) corresponding to each grid cell.
    """
    result = compute_oil_formation_volume_factor(
        pressure=pressure_grid,
        temperature=temperature_grid,
        bubble_point_pressure=bubble_point_pressure_grid,
        oil_specific_gravity=oil_specific_gravity_grid,
        gas_gravity=gas_gravity_grid,
        gas_to_oil_ratio=solution_gas_to_oil_ratio_grid,
        oil_compressibility=oil_compressibility_grid,
    )
    return result  # type: ignore[return-value]


def build_water_formation_volume_factor_grid(
    water_density_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water formation volume factors.

    :param water_density_grid: N-Dimensional array of water density values (lbm/ft³) representing the density of water at reservoir conditions.
    :param salinity_grid: N-Dimensional array of water salinity values (ppm of NaCl) representing the salinity of water at reservoir conditions.
    :return: N-Dimensional array of water formation volume factors (bbl/STB) corresponding to each grid cell.
    """
    result = compute_water_formation_volume_factor(
        water_density=water_density_grid,
        salinity=salinity_grid,
    )
    return result  # type: ignore[return-value]


def build_gas_formation_volume_factor_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_compressibility_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas formation volume factors.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_compressibility_factor_grid: N-Dimensional array of gas compressibility factor values (dimensionless) representing the compressibility of gas.
    :return: N-Dimensional array of gas formation volume factors (ft³/SCF) corresponding to each grid cell.
    """
    result = compute_gas_formation_volume_factor(
        pressure=pressure_grid,
        temperature=temperature_grid,
        gas_compressibility_factor=gas_compressibility_factor_grid,
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_solution_gas_to_oil_ratio_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    gor_at_bubble_point_pressure_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of solution gas-to-oil ratios.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.

    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of bubble point pressures (psi) corresponding to each grid cell.
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param oil_api_gravity_grid: N-Dimensional array of oil API gravity values (dimensionless) representing the density of oil relative to water.
    :param gor_at_bubble_point_pressure_grid: Optional N-Dimensional array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: N-Dimensional array of solution gas-to-oil ratios (SCF/STB) corresponding to each grid cell.
    """
    result = compute_gas_to_oil_ratio(
        pressure=pressure_grid,
        temperature=temperature_grid,
        bubble_point_pressure=bubble_point_pressure_grid,
        gas_gravity=gas_gravity_grid,
        oil_api_gravity=oil_api_gravity_grid,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure_grid,
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_estimated_solution_gas_to_oil_ratio_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    maximum_iterations: int = 20,
    tolerance: float = 1e-4,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of solution gas-to-oil ratios using iterative solver.

    This function solves for Rs(P, T) at each grid point by iterating between:
    - Rs estimate from Standing correlation
    - Bubble point pressure Pb from Vazquez-Beggs correlation

    This is useful when bubble point pressures are composition-dependent (2D Pb table)
    and Rs cannot be computed directly without knowing Pb, which itself depends on Rs.

    The algorithm handles both:
    - Saturated oil (P <= Pb): Rs varies with pressure
    - Undersaturated oil (P > Pb): Rs = Rsb (constant at bubble point value)

    :param pressure_grid: N-Dimensional array of pressure values (psi)
    :param temperature_grid: N-Dimensional array of temperature values (°F)
    :param oil_api_gravity_grid: N-Dimensional array of oil API gravity values (°API)
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless)
    :param maximum_iterations: Maximum iterations for convergence at each point (default: 20)
    :param tolerance: Relative tolerance for convergence (default: 1e-4)
    :return: N-Dimensional array of solution gas-to-oil ratios (SCF/STB)

    Example:
    ```python
    import numpy as np

    pressures = np.linspace(500, 5000, 50)
    temperatures = np.linspace(100, 250, 30)
    p_grid, t_grid = np.meshgrid(pressures, temperatures, indexing='ij')
    api_grid = np.full_like(p_grid, 35.0)  # 35° API oil
    gg_grid = np.full_like(p_grid, 0.65)   # Gas gravity
    rs_grid = build_estimated_solution_gas_to_oil_ratio_grid(
        pressure_grid=p_grid,
        temperature_grid=t_grid,
        oil_api_gravity_grid=api_grid,
        gas_gravity_grid=gg_grid,
    )
    ```
    """
    result = estimate_solution_gor(
        pressure=pressure_grid,
        temperature=temperature_grid,
        oil_api_gravity=oil_api_gravity_grid,
        gas_gravity=gas_gravity_grid,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    return result  # type: ignore[return-value]


def build_oil_bubble_point_pressure_grid(
    gas_gravity_grid: NDimensionalGrid[NDimension],
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    solution_gas_to_oil_ratio_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil bubble point pressures.

    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param oil_api_gravity_grid: N-Dimensional array of API gravity values.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gor_at_bubble_point_pressure_grid: N-Dimensional array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: N-Dimensional array of oil bubble point pressures (psi) corresponding to each grid cell.
    """
    result = compute_oil_bubble_point_pressure(
        gas_gravity=gas_gravity_grid,
        oil_api_gravity=oil_api_gravity_grid,
        temperature=temperature_grid,
        gas_to_oil_ratio=solution_gas_to_oil_ratio_grid,
    )
    return result  # type: ignore[return-value]


def build_gas_solubility_in_water_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
    gas: str = "methane",
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas solubilities.

    The solubility is computed using the Henry's law constant for the specific gas in water.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param salinity_grid: N-Dimensional array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas: Type of gas dissolved in water (default: "methane"). Can be CO₂, N₂, etc.
    :return: N-Dimensional array of gas solubility in water (SCF/STB) corresponding to each grid cell.
    """
    result = compute_gas_solubility_in_water(
        pressure=pressure_grid,
        temperature=temperature_grid,
        salinity=salinity_grid,
        gas=gas,
    )
    return result  # type: ignore[return-value]


def build_water_bubble_point_pressure_grid(
    temperature_grid: NDimensionalGrid[NDimension],
    gas_solubility_in_water_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
    gas: str = "methane",
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water bubble point pressures.

    The bubble point pressure is computed using the gas solubility in water and the salinity of the water.

    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_solubility_in_water_grid: N-Dimensional array of gas solubility in water (SCF/STB) at bubble point pressure corresponding to each grid cell.
    :param salinity_grid: N-Dimensional array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas: Type of gas dissolved in water (default: "methane"). Can be CO₂, N₂, etc.
    :return: N-Dimensional array of water bubble point pressures (psi) corresponding to each grid cell.
    """
    result = compute_water_bubble_point_pressure(
        temperature=temperature_grid,
        gas_solubility_in_water=gas_solubility_in_water_grid,
        salinity=salinity_grid,
        gas=gas,
    )
    return result  # type: ignore[return-value]


def build_oil_viscosity_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    oil_specific_gravity_grid: NDimensionalGrid[NDimension],
    solution_gas_to_oil_ratio_grid: NDimensionalGrid[NDimension],
    gor_at_bubble_point_pressure_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil viscosities.

    The viscosity is computed using the oil specific gravity, gas gravity, and the gas-to-oil ratio.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of bubble point pressures (psi) corresponding to each grid cell.
    :param oil_specific_gravity_grid: N-Dimensional array of oil specific gravity values (dimensionless).
    :param solution_gas_to_oil_ratio_grid: N-Dimensional array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param gor_at_bubble_point_pressure_grid: Optional N-Dimensional array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: N-Dimensional array of oil viscosities (cP) corresponding to each grid cell.
    """
    result = compute_oil_viscosity(
        pressure=pressure_grid,
        temperature=temperature_grid,
        bubble_point_pressure=bubble_point_pressure_grid,
        oil_specific_gravity=oil_specific_gravity_grid,
        gas_to_oil_ratio=solution_gas_to_oil_ratio_grid,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure_grid,
    )
    return result  # type: ignore[return-value]


def build_gas_molecular_weight_grid(
    gas_gravity_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas molecular weights.

    The molecular weight is computed using the formula:
    Molecular Weight = Gas Gravity * 28.9644 g/mol (molecular weight of air)

    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :return: N-Dimensional array of gas molecular weights (g/mol) corresponding to each grid cell.
    """
    result = compute_gas_molecular_weight(gas_gravity=gas_gravity_grid)
    return result  # type: ignore[return-value]


def build_gas_viscosity_grid(
    temperature_grid: NDimensionalGrid[NDimension],
    gas_density_grid: NDimensionalGrid[NDimension],
    gas_molecular_weight_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas viscosities.


    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_density_grid: N-Dimensional array of gas density values (lbm/ft³) representing the density of gas in each grid cell.
    :param gas_molecular_weight_grid: N-Dimensional array of gas molecular weight values (g/mol) representing the molecular weight of gas in each grid cell.
    :return: N-Dimensional array of gas viscosities (cP) corresponding to each grid cell.
    """
    result = compute_gas_viscosity(
        temperature=temperature_grid,
        gas_density=gas_density_grid,
        gas_molecular_weight=gas_molecular_weight_grid,
    )
    return result  # type: ignore[return-value]


def build_water_viscosity_grid(
    temperature_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
    pressure_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water/brine viscosities.

    The viscosity is computed using the pressure, temperature, and salinity conditions.

    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param salinity_grid: N-Dimensional array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :return: N-Dimensional array of water/brine viscosities (cP) corresponding to each grid cell.
    """
    result = compute_water_viscosity(
        temperature=temperature_grid,
        salinity=salinity_grid,
        pressure=pressure_grid,
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_oil_compressibility_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    gor_at_bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    gas_formation_volume_factor_grid: NDimensionalGrid[NDimension],
    oil_formation_volume_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil compressibilities.

    The compressibility is computed using the oil API gravity, gas gravity, and the gas-to-oil ratio.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of bubble point pressures (psi) corresponding to each grid cell.
    :param oil_api_gravity_grid: N-Dimensional array of oil API gravity values (dimensionless).
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param solution_gas_to_oil_ratio_grid: N-Dimensional array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :return: N-Dimensional array of oil compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    result = compute_oil_compressibility(
        pressure=pressure_grid,
        temperature=temperature_grid,
        bubble_point_pressure=bubble_point_pressure_grid,
        oil_api_gravity=oil_api_gravity_grid,
        gas_gravity=gas_gravity_grid,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure_grid,
        gas_formation_volume_factor=gas_formation_volume_factor_grid,
        oil_formation_volume_factor=oil_formation_volume_factor_grid,
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_gas_compressibility_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    gas_compressibility_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    h2s_mole_fraction: NumberOrArray = 0.0,
    co2_mole_fraction: NumberOrArray = 0.0,
    n2_mole_fraction: NumberOrArray = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas compressibilities.

    The compressibility is computed using the gas gravity and the pressure and temperature conditions.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_compressibility_factor_grid: Optional N-Dimensional array of precalculated gas compressibility factors (dimensionless).
        If provided, it will be used directly; otherwise, it will be computed.
    :param h2s_mole_fraction: Mole fraction of H₂S in the gas mixture (default: 0.0).
    :param co2_mole_fraction: Mole fraction of CO₂ in the gas mixture (default: 0.0).
    :param n2_mole_fraction: Mole fraction of N₂ in the gas mixture (default: 0.0).
    :return: N-Dimensional array of gas compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    result = compute_gas_compressibility(
        pressure=pressure_grid,
        temperature=temperature_grid,
        gas_gravity=gas_gravity_grid,
        gas_compressibility_factor=gas_compressibility_factor_grid,
        h2s_mole_fraction=h2s_mole_fraction,  # type: ignore[arg-type]
        co2_mole_fraction=co2_mole_fraction,  # type: ignore[arg-type]
        n2_mole_fraction=n2_mole_fraction,  # type: ignore[arg-type]
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_gas_free_water_formation_volume_factor_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas free water formation volume factors.

    The gas free water formation volume factor is computed using the pressure and temperature conditions.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :return: N-Dimensional array of gas free water formation volume factors (bbl/STB) corresponding to each grid cell.
    """
    result = compute_gas_free_water_formation_volume_factor(
        pressure=pressure_grid,
        temperature=temperature_grid,
    )
    return result  # type: ignore[return-value]


def build_water_compressibility_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    gas_formation_volume_factor_grid: NDimensionalGrid[NDimension],
    gas_solubility_in_water_grid: NDimensionalGrid[NDimension],
    gas_free_water_formation_volume_factor_grid: NDimensionalGrid[NDimension],
    salinity: NumberOrArray = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water compressibilities.

    The compressibility is computed using the gas formation volume factor, gas solubility in water, and gas free water formation volume factor.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of water bubble point pressures (psi) corresponding to each grid cell.
    :param gas_formation_volume_factor_grid: N-Dimensional array of gas formation volume factors (ft³/SCF) corresponding to each grid cell.
    :param gas_solubility_in_water_grid: N-Dimensional array of gas solubility in water (SCF/STB) corresponding to each grid cell.
    :param gas_free_water_formation_volume_factor_grid: N-Dimensional array of gas free water formation volume factors (bbl/STB).
    :return: N-Dimensional array of water compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    result = compute_water_compressibility(
        pressure=pressure_grid,
        temperature=temperature_grid,
        bubble_point_pressure=bubble_point_pressure_grid,
        gas_formation_volume_factor=gas_formation_volume_factor_grid,
        gas_solubility_in_water=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor=gas_free_water_formation_volume_factor_grid,
        salinity=salinity,
    )
    return result  # type: ignore[return-value]


def build_live_oil_density_grid(
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    solution_gas_to_oil_ratio_grid: NDimensionalGrid[NDimension],
    formation_volume_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of live oil densities.

    The density is computed using the oil API gravity, gas gravity, and the gas-to-oil ratio.

    :param oil_api_gravity_grid: N-Dimensional array of oil API gravity values (dimensionless).
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param solution_gas_to_oil_ratio_grid: N-Dimensional array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param formation_volume_factor_grid: N-Dimensional array of formation volume factors (bbl/STB).
    :return: N-Dimensional array of oil densities (lbm/ft³) corresponding to each grid cell.
    """
    result = compute_live_oil_density(
        api_gravity=oil_api_gravity_grid,
        gas_gravity=gas_gravity_grid,
        gas_to_oil_ratio=solution_gas_to_oil_ratio_grid,
        formation_volume_factor=formation_volume_factor_grid,
    )
    return result  # type: ignore[return-value]


def build_gas_density_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    gas_compressibility_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas densities.

    The density is computed using the gas gravity and the pressure and temperature conditions.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_compressibility_factor_grid: N-Dimensional array of gas compressibility factor values (dimensionless)
        representing the compressibility of gas.
    :return: N-Dimensional array of gas densities (lbm/ft³) corresponding to each grid cell.
    """
    result = compute_gas_density(
        pressure=pressure_grid,
        temperature=temperature_grid,
        gas_gravity=gas_gravity_grid,
        gas_compressibility_factor=gas_compressibility_factor_grid,
    )
    return result  # type: ignore[return-value]


def build_water_density_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
    gas_solubility_in_water_grid: NDimensionalGrid[NDimension],
    gas_free_water_formation_volume_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water/brine densities.

    The density is computed using the gas gravity, salinity, and gas solubility in water.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param salinity_grid: N-Dimensional array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas_solubility_in_water_grid: N-Dimensional array of gas solubility in water (SCF/STB).
    :param gas_free_water_formation_volume_factor_grid: N-Dimensional array of gas free water formation volume factors (bbl/STB).
    :return: N-Dimensional array of water/brine densities (lbm/ft³) corresponding to each grid cell.
    """
    result = compute_water_density(
        pressure=pressure_grid,
        temperature=temperature_grid,
        gas_gravity=gas_gravity_grid,
        salinity=salinity_grid,
        gas_solubility_in_water=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor=gas_free_water_formation_volume_factor_grid,
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_oil_effective_viscosity_grid(
    oil_viscosity_grid: NDimensionalGrid[NDimension],
    solvent_viscosity_grid: NDimensionalGrid[NDimension],
    solvent_concentration_grid: NDimensionalGrid[NDimension],
    base_omega: float,
    pressure_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    minimum_miscibility_pressure: typing.Optional[float] = None,
    transition_width: float = 500.0,
) -> NDimensionalGrid[NDimension]:
    """
    Build effective oil-solvent mixture viscosity grid using Todd-Longstaff model.

    Computes spatially-varying effective viscosity accounting for:
    - Local oil and solvent viscosities
    - Local solvent concentration
    - Pressure-dependent miscibility (if pressure grid provided)

    :param oil_viscosity_grid: Pure oil viscosity grid (cP)
    :param solvent_viscosity_grid: Pure solvent viscosity grid (cP)
    :param solvent_concentration_grid: Solvent concentration grid (fraction 0-1)
    :param base_omega: Base Todd-Longstaff mixing parameter (0-1).
        Typical: 0.67 for CO2-oil. This is the maximum omega achieved when fully miscible.
    :param pressure_grid: Optional pressure grid (psi) for pressure-dependent miscibility.
        If provided, omega varies spatially based on local pressure vs MMP.
    :param minimum_miscibility_pressure: Minimum miscibility pressure (MMP, psi).
        Required if pressure_grid is provided.
    :param transition_width: Pressure width of miscibility transition zone (psi), default 500.
        Only used if pressure_grid is provided.
    :return: Effective mixture viscosity grid (cP)

    Example:

    ```python
    # Simple case: constant omega everywhere (no pressure effects)
    visc_grid = build_oil_effective_viscosity_grid(
        oil_viscosity_grid=oil_visc,
        solvent_viscosity_grid=solvent_visc,
        solvent_concentration_grid=conc,
        base_omega=0.67
    )

    # Advanced: pressure-dependent miscibility
    visc_grid = build_oil_effective_viscosity_grid(
        oil_viscosity_grid=oil_visc,
        solvent_viscosity_grid=solvent_visc,
        solvent_concentration_grid=conc,
        base_omega=0.67,
        pressure_grid=pressure,
        minimum_miscibility_pressure=2000.0,
        transition_width=500.0
    )
    ```
    """
    input_type = oil_viscosity_grid.dtype
    # Compute pressure-dependent omega if pressure grid provided
    if pressure_grid is not None and minimum_miscibility_pressure is not None:
        # Compute transition factor grid
        transition_factor = compute_miscibility_transition_factor(
            pressure=pressure_grid,
            minimum_miscibility_pressure=minimum_miscibility_pressure,
            transition_width=transition_width,
        )
        # Scale by base omega to get effective omega grid
        omega_grid = (base_omega * transition_factor).astype(input_type)
    else:
        # Use constant omega everywhere
        omega_grid = np.full_like(oil_viscosity_grid, base_omega)

    # Compute effective viscosity using Todd-Longstaff model
    result = compute_todd_longstaff_effective_viscosity(
        oil_viscosity=oil_viscosity_grid,
        solvent_viscosity=solvent_viscosity_grid,
        solvent_concentration=solvent_concentration_grid,
        omega=omega_grid,  # type: ignore[arg-type]
    )
    return result  # type: ignore[return-value]


@numba.njit(cache=True)
def build_oil_effective_density_grid(
    oil_density_grid: NDimensionalGrid[NDimension],
    solvent_density_grid: NDimensionalGrid[NDimension],
    oil_viscosity_grid: NDimensionalGrid[NDimension],
    solvent_viscosity_grid: NDimensionalGrid[NDimension],
    solvent_concentration_grid: NDimensionalGrid[NDimension],
    base_omega: float,
    pressure_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    minimum_miscibility_pressure: typing.Optional[float] = None,
    transition_width: float = 500.0,
) -> NDimensionalGrid[NDimension]:
    """
    Build effective oil-solvent mixture density grid using Todd-Longstaff model.

    Computes spatially-varying effective density accounting for:
    - Local oil and solvent densities
    - Local oil and solvent viscosities (needed for flow fractions)
    - Local solvent concentration
    - Pressure-dependent miscibility (if pressure grid provided)

    :param oil_density_grid: Pure oil density grid (lb/ft³ or kg/m³)
    :param solvent_density_grid: Pure solvent density grid (lb/ft³ or kg/m³)
    :param oil_viscosity_grid: Pure oil viscosity grid (cP)
        Required to compute flow fractions for segregated density
    :param solvent_viscosity_grid: Pure solvent viscosity grid (cP)
        Required to compute flow fractions for segregated density
    :param solvent_concentration_grid: Solvent concentration grid (fraction 0-1)
    :param base_omega: Base Todd-Longstaff mixing parameter for density (0-1).
        Typical: 0.67 for CO2-oil. Maximum omega when fully miscible.
    :param pressure_grid: Optional pressure grid (psi) for pressure-dependent miscibility.
        If provided, omega varies spatially based on local pressure vs MMP.
    :param minimum_miscibility_pressure: Minimum miscibility pressure (MMP, psi).
        Required if pressure_grid is provided.
    :param transition_width: Pressure width of miscibility transition zone (psi), default 500.
        Only used if pressure_grid is provided.
    :return: Effective mixture density grid (same units as input density grids)

    Example:
    ```python
    # Simple case: no pressure effects
    density_grid = build_oil_effective_density_grid(
        oil_density_grid=oil_dens,
        solvent_density_grid=solvent_dens,
        oil_viscosity_grid=oil_visc,
        solvent_viscosity_grid=solvent_visc,
        solvent_concentration_grid=conc,
        base_omega=0.67
    )

    # With pressure-dependent miscibility
    density_grid = build_oil_effective_density_grid(
        oil_density_grid=oil_dens,
        solvent_density_grid=solvent_dens,
        oil_viscosity_grid=oil_visc,
        solvent_viscosity_grid=solvent_visc,
        solvent_concentration_grid=conc,
        base_omega=0.67,
        pressure_grid=pressure,
        minimum_miscibility_pressure=2000.0
    )
    ```
    """
    input_type = oil_density_grid.dtype
    # Compute pressure-dependent omega if pressure grid provided
    if pressure_grid is not None and minimum_miscibility_pressure is not None:
        # Compute transition factor grid
        transition_factor = compute_miscibility_transition_factor(
            pressure=pressure_grid,
            minimum_miscibility_pressure=minimum_miscibility_pressure,
            transition_width=transition_width,
        )
        # Scale by base omega to get effective omega grid
        omega_grid = (base_omega * transition_factor).astype(input_type)
    else:
        # Use constant omega everywhere
        omega_grid = np.full_like(oil_density_grid, base_omega)

    # Compute effective density using Todd-Longstaff model
    result = compute_todd_longstaff_effective_density(
        oil_density=oil_density_grid,
        solvent_density=solvent_density_grid,
        oil_viscosity=oil_viscosity_grid,
        solvent_viscosity=solvent_viscosity_grid,
        solvent_concentration=solvent_concentration_grid,
        omega=omega_grid,  # type: ignore[arg-type]
    )
    return result  # type: ignore[return-value]
