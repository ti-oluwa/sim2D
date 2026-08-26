import logging

import numba  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.correlations.arrays.core import compute_fluid_density
from bores.correlations.core import fahrenheit_to_kelvin, fahrenheit_to_rankine
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.typing import (
    NDimension,
    NumberArray,
    NumberOrArray,
    ZFactorMethod,
)
from bores.utils import clip, max_, min_

logger = logging.getLogger(__name__)

__all__ = [
    "compute_gas_compressibility",
    "compute_gas_compressibility_factor",
    "compute_gas_compressibility_factor_dranchuk_abou_kassem",
    "compute_gas_compressibility_factor_hall_yarborough",
    "compute_gas_compressibility_factor_papay",
    "compute_gas_density",
    "compute_gas_formation_volume_factor",
    "compute_gas_gravity",
    "compute_gas_gravity_from_density",
    "compute_gas_molecular_weight",
    "compute_gas_pseudocritical_properties",
    "compute_gas_viscosity",
]


def compute_gas_gravity(gas: str, dtype: npt.DTypeLike = None) -> float:
    """
    Computes the specific gravity of a gas at a given pressure and temperature.

    Gas gravity is defined as the ratio of the density of the gas to the density of air at standard conditions.

    :param gas: gas name supported by CoolProp (e.g., 'Methane')
    :return: Gas gravity (dimensionless)
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    gas_density_at_stp = compute_fluid_density(
        c.STANDARD_PRESSURE_IMPERIAL, c.STANDARD_TEMPERATURE_IMPERIAL, fluid=gas
    )
    air_density_at_stp = compute_fluid_density(
        c.STANDARD_PRESSURE_IMPERIAL, c.STANDARD_TEMPERATURE_IMPERIAL, fluid="Air"
    )
    return np.divide(gas_density_at_stp, air_density_at_stp, dtype=dtype)  # type: ignore[return-value]


def compute_gas_gravity_from_density(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    density: NumberArray[NDimension],
) -> NumberArray[NDimension]:
    """
    Computes the gas gravity from density.

    Gas gravity for this case, is derived as the ratio of the gas density to the
    air density at the same temperature and pressure.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param density: Density of the gas in lbm/ft³
    :return: Gas gravity (dimensionless)
    """
    dtype = pressure.dtype
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_in_pascals = np.multiply(pressure, c.PSI_TO_PASCAL, dtype=dtype)
    air_density = compute_fluid_density(
        pressure=pressure_in_pascals,
        temperature=temperature_in_kelvin,  # type: ignore
        fluid="Air",
    )
    return np.divide(  # type: ignore[return-value]
        density,
        np.multiply(
            air_density,
            c.KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET,
            dtype=dtype,
        ),
        dtype=dtype,
    )


def compute_gas_formation_volume_factor(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    gas_compressibility_factor: NumberArray[NDimension],
) -> NumberArray[NDimension]:
    """
    Computes the gas formation volume factor (B_g) in ft³/SCF, using the real gas law.

    Formula (real gas EOS):

        B_g = (Z * T * P_std) / (P * T_std)

    Where:
        - B_g: Gas formation volume factor (ft³/SCF)
        - Z: Gas compressibility factor (dimensionless)
        - T: Reservoir temperature (°F)
        - P: Reservoir pressure (psi)
        - P_std: Standard pressure = 14.696 psi
        - T_std: Standard temperature = 60°F

    Assumes ideal gas law corrected with Z-factor (real gas behavior).

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param gas_compressibility_factor: Z-factor (dimensionless)
    :return: Gas formation volume factor (ft³/SCF)
    """
    if min_(pressure) <= 0 or min_(temperature) <= 0:
        raise ValidationError("Pressure and temperature must be positive.")
    if min_(gas_compressibility_factor) <= 0:
        raise ValidationError("Z-factor must be positive.")

    dtype = pressure.dtype
    temperature_in_rankine = fahrenheit_to_rankine(temperature)
    return (
        gas_compressibility_factor
        * temperature_in_rankine
        * c.STANDARD_PRESSURE_IMPERIAL
        / (pressure * c.STANDARD_TEMPERATURE_RANKINE)
    ).astype(dtype)


@numba.njit(cache=True)
def compute_gas_compressibility_factor_papay(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    h2s_mole_fraction: NumberOrArray[NDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[NDimension] = 0.0,
) -> NumberArray[NDimension]:
    """
    Computes gas compressibility factor using Papay's correlation,
    with corrections for sour gases using the Wichert-Aziz method.

    Papay's correlation is a widely used empirical relationship to estimate the
    compressibility factor of natural gas based on its pseudo-reduced properties.

    The equation is:

        Z = 1 - ((3.52 * P_r * exp(-0.869 * T_r)) / T_r) + ((0.274 * P_r**2)/T_r**2)

    where:
    - Z is the compressibility factor (dimensionless)
    - P_r is the pseudo-reduced pressure (dimensionless)
    - T_r is the pseudo-reduced temperature (dimensionless)
    - P_r = P / P_pc
    - T_r = T / T_pc
    - P_pc is the pseudo-critical pressure (psi)
    - T_pc is the pseudo-critical temperature (°F)
    - P is the pressure (psi)
    - T is the temperature (°F)
    - P_pc and T_pc are calculated based on the gas specific gravity (gas_gravity).

    Valid Range:
        - Pseudo-reduced pressure (Pr): 0.2 < Pr < 15
        - Pseudo-reduced temperature (Tr): 1.05 < Tr < 3.0
        - Gas gravity: 0.55 < γg < 1.0
        - H₂S + CO₂ < 40 mol%
        - H₂S alone < 25 mol%

    :param gas_gravity: Gas specific gravity (dimensionless)
    :param pressure: Pressure array (psi)
    :param temperature: Temperature array (°F)
    :param h2s_mole_fraction: Mole fraction of H2S in the gas (dimensionless, default is 0.0)
    :param co2_mole_fraction: Mole fraction of CO2 in the gas (dimensionless, default is 0.0)
    :param n2_mole_fraction: Mole fraction of N2 in the gas (dimensionless, default is 0.0)
    :return: Compressibility factor Z (dimensionless)
    """
    if min_(pressure) <= 0 or min_(temperature) <= 0 or min_(gas_gravity) <= 0:
        raise ValidationError("Pressure, temperature, and gas specific gravity must be positive.")

    pseudo_critical_pressure, pseudo_critical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )

    pseudo_reduced_pressure = pressure / pseudo_critical_pressure
    pseudo_reduced_temperature = fahrenheit_to_rankine(temperature) / pseudo_critical_temperature
    # Papay's correlation for gas compressibility factor
    compressibility_factor = (
        1
        - (
            (3.52 * pseudo_reduced_pressure * np.exp(-0.869 * pseudo_reduced_temperature))
            / pseudo_reduced_temperature
        )
        + ((0.274 * pseudo_reduced_pressure**2) / pseudo_reduced_temperature**2)
    )
    dtype = pressure.dtype
    # Ensure Z is not negative or too low
    return np.maximum(0.1, compressibility_factor).astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_gas_compressibility_factor_hall_yarborough(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    h2s_mole_fraction: NumberOrArray[NDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    maximum_iterations: int = 50,
    tolerance: float = 1e-10,
) -> NumberArray[NDimension]:
    """
    Computes gas compressibility factor using Hall-Yarborough (1973) implicit correlation.

    This vectorized implementation solves for reduced density (y) using Newton-Raphson
    at each grid point:
        f(y) = -A * Pr + (y + y² + y³ - y⁴) / (1 - y)³ - B * y² + C * y^D = 0

    where:
        A = 0.06125 * Pr * t * exp(-1.2 * (1 - t)²)
        B = t * (14.76 - 9.76 * t + 4.58 * t²)
        C = t * (90.7 - 242.2 * t + 42.4 * t²)
        D = 2.18 + 2.82 * t
        t = 1 / Tr (reciprocal reduced temperature)

    Then: Z = A * Pr / y

    Valid Range:
        - Pr: 0.2 < Pr < 30 (wider than Papay)
        - Tr: 1.0 < Tr < 3.0
        - Most accurate for Pr > 1.0

    :param pressure: Pressure array (psi)
    :param temperature: Temperature array (°F)
    :param gas_gravity: Gas specific gravity array (dimensionless)
    :param h2s_mole_fraction: H₂S mole fraction (0.0 to 1.0)
    :param co2_mole_fraction: CO₂ mole fraction (0.0 to 1.0)
    :param n2_mole_fraction: N₂ mole fraction (0.0 to 1.0)
    :param maximum_iterations: Maximum Newton-Raphson iterations
    :param tolerance: Convergence tolerance
    :return: Compressibility factor Z array (dimensionless)

    References:
        Hall, K.R. and Yarborough, L. (1973). "A New Equation of ReservoirState for Z-factor Calculations."
        Oil & Gas Journal, June 18, 1973, pp. 82-92.
    """
    if min_(pressure) <= 0 or min_(temperature) <= 0 or min_(gas_gravity) <= 0:
        raise ValidationError("Pressure, temperature, and gas gravity must be positive.")

    # Get pseudocritical properties with Wichert-Aziz correction
    pseudo_critical_pressure, pseudo_critical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )

    Pr = pressure / pseudo_critical_pressure
    Tr = fahrenheit_to_rankine(temperature) / pseudo_critical_temperature

    # For very low pressure, use ideal gas approximation
    Z = np.where(Pr < 0.01, 1.0, 0.0)  # Will be overwritten where Pr >= 0.01

    # Reciprocal reduced temperature
    t = 1.0 / Tr

    # Coefficients
    A = 0.06125 * Pr * t * np.exp(-1.2 * (1.0 - t) ** 2)
    B = t * (14.76 - 9.76 * t + 4.58 * t**2)
    C = t * (90.7 - 242.2 * t + 42.4 * t**2)
    D = 2.18 + 2.82 * t

    # Initial guess for reduced density (y) - broadcast to shape
    y = clip(0.27 * Pr / Tr, 0.01, 0.9)

    # Create mask for points that need iteration
    active_mask = Pr >= 0.01

    # Newton-Raphson iteration (vectorized)
    for _ in range(maximum_iterations):
        y_old = y.copy()

        # Function f(y) and its derivative f'(y)
        y2 = y * y
        y3 = y2 * y
        y4 = y3 * y

        one_minus_y = 1.0 - y
        one_minus_y_cubed = one_minus_y**3

        # f(y) = -A*Pr + (y + y² + y³ - y⁴)/(1-y)³ - B*y² + C*y^D
        numerator = y + y2 + y3 - y4
        f = -A * Pr + numerator / one_minus_y_cubed - B * y2 + C * (y**D)

        # f'(y) = d/dy[(y + y² + y³ - y⁴)/(1-y)³] - 2*B*y + C*D*y^(D-1)
        d_numerator = 1.0 + 2.0 * y + 3.0 * y2 - 4.0 * y3
        d_denominator = 3.0 * one_minus_y**2

        df = (
            (d_numerator * one_minus_y_cubed + numerator * d_denominator)
            / (one_minus_y_cubed * one_minus_y_cubed)
            - 2.0 * B * y
            + C * D * (y ** (D - 1.0))
        )

        # Newton-Raphson update (avoid division by zero)
        df_safe = np.where(np.abs(df) < 1e-15, 1e-15, df)
        y_new = y_old - f / df_safe

        # Clamp y to physical range [0, 1) and only update active points
        y = np.where(active_mask, clip(y_new, 0.0, 0.99), y)

        # Check convergence
        converged = np.abs(y - y_old) < tolerance
        if np.all(converged | ~active_mask):
            break

    # Compute Z-factor
    y_safe = np.where(np.abs(y) < 1e-15, 1e-15, y)
    Z = np.where(active_mask, A * Pr / y_safe, 1.0)

    # Clamp to physical range
    dtype = pressure.dtype
    return clip(Z, 0.2, 3.0).astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_gas_compressibility_factor_dranchuk_abou_kassem(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    h2s_mole_fraction: NumberOrArray[NDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    maximum_iterations: int = 50,
    tolerance: float = 1e-10,
) -> NumberArray[NDimension]:
    """
    Computes gas compressibility factor using Dranchuk-Abou-Kassem (DAK, 1975) correlation.

    This vectorized implementation uses an 11-parameter fit to Standing-Katz Z-factor
    chart data, solved iteratively at each grid point:

        Z = 1 + (A₁ + A₂/Tr + A₃/Tr³ + A₄/Tr⁴ + A₅/Tr⁵)*ρr
            + (A₆ + A₇/Tr + A₈/Tr²)*ρr²
            - A₉*(A₇/Tr + A₈/Tr²)*ρr⁵
            + A₁₀*(1 + A₁₁*ρr²)*(ρr²/Tr³)*exp(-A₁₁*ρr²)

    where:
        ρr = 0.27 * Pr / (Z * Tr)  (reduced density)

    Coefficients (from Dranchuk & Abou-Kassem, 1975):
        A₁ = 0.3265, A₂ = -1.0700, A₃ = -0.5339, A₄ = 0.01569, A₅ = -0.05165
        A₆ = 0.5475, A₇ = -0.7361, A₈ = 0.1844, A₉ = 0.1056, A₁₀ = 0.6134, A₁₁ = 0.7210

    Valid Range:
        - Pr: 0.2 < Pr < 30 (widest range)
        - Tr: 1.0 < Tr < 3.0
        - Highly accurate across entire range

    :param pressure: Pressure array (psi)
    :param temperature: Temperature array (°F)
    :param gas_gravity: Gas specific gravity array (dimensionless)
    :param h2s_mole_fraction: H₂S mole fraction (0.0 to 1.0)
    :param co2_mole_fraction: CO₂ mole fraction (0.0 to 1.0)
    :param n2_mole_fraction: N₂ mole fraction (0.0 to 1.0)
    :param maximum_iterations: Maximum iterations for density convergence
    :param tolerance: Convergence tolerance
    :return: Compressibility factor Z array (dimensionless)

    References:
        Dranchuk, P.M. and Abou-Kassem, J.H. (1975). "Calculation of Z Factors for
        Natural Gases Using Equations of ReservoirState." Journal of Canadian Petroleum Technology,
        July-September 1975, pp. 34-36.
    """
    if min_(pressure) <= 0 or min_(temperature) <= 0 or min_(gas_gravity) <= 0:
        raise ValidationError(
            f"Pressure, temperature, and gas gravity must be positive.\n"
            f"Pressure={min_(pressure)}psi, Temperature={min_(temperature)}°F, Gas gravity={min_(gas_gravity)}"
        )

    # Get pseudocritical properties
    pseudo_critical_pressure, pseudo_critical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )

    Pr = pressure / pseudo_critical_pressure
    Tr = fahrenheit_to_rankine(temperature) / pseudo_critical_temperature

    # For very low pressure, use ideal gas
    Z = np.where(Pr < 0.01, 1.0, 1.0)  # Initial guess

    # DAK coefficients
    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.7210

    # Create mask for points that need iteration
    active_mask = Pr >= 0.01

    # Iterative solution for Z (vectorized)
    for _ in range(maximum_iterations):
        Z_old = Z.copy()

        # Reduced density: ρr = 0.27 * Pr / (Z * Tr)
        Z_safe = np.where(np.abs(Z) < 1e-15, 1e-15, Z)
        rho_r = 0.27 * Pr / (Z_safe * Tr)
        rho_r2 = rho_r * rho_r
        rho_r5 = rho_r2 * rho_r2 * rho_r

        # Reciprocal reduced temperature terms
        Tr_inv = 1.0 / Tr
        Tr_inv2 = Tr_inv * Tr_inv
        Tr_inv3 = Tr_inv2 * Tr_inv
        Tr_inv4 = Tr_inv3 * Tr_inv
        Tr_inv5 = Tr_inv4 * Tr_inv

        # Compute Z from DAK equation
        term1 = (A1 + A2 * Tr_inv + A3 * Tr_inv3 + A4 * Tr_inv4 + A5 * Tr_inv5) * rho_r
        term2 = (A6 + A7 * Tr_inv + A8 * Tr_inv2) * rho_r2
        term3 = -A9 * (A7 * Tr_inv + A8 * Tr_inv2) * rho_r5

        # Exponential term (clamp to avoid overflow)
        exp_arg = clip(-A11 * rho_r2, -700, 700)
        exp_term = np.exp(exp_arg)
        term4 = A10 * (1.0 + A11 * rho_r2) * (rho_r2 * Tr_inv3) * exp_term

        Z_new = 1.0 + term1 + term2 + term3 + term4

        # Only update active points
        Z = np.where(active_mask, Z_new, Z)

        # Check convergence
        converged = np.abs(Z - Z_old) < tolerance
        if np.all(converged | ~active_mask):
            break

    # Return with same dtype as input pressure
    dtype = pressure.dtype
    return clip(Z, 0.2, 3.0).astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_gas_compressibility_factor(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    h2s_mole_fraction: NumberOrArray[NDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    method: ZFactorMethod = "dak",
) -> NumberArray[NDimension]:
    """
    Computes (natural) gas compressibility factor.

    Method Selection Strategy:
    1. **High Pressure (Pr > 15)**: Use DAK (most accurate for Pr up to 30)
    2. **Medium Pressure (1 < Pr ≤ 15)**: Use Hall-Yarborough (best balance)
    3. **Low Pressure (Pr ≤ 1)**: Use Papay (fast, accurate for low Pr)
    4. **Fallback**: If any method produces invalid results (Z < 0.2 or Z > 3.0),
        try alternative methods

    However, DAK is usually more than enough for black-oil simulations.

    Available Methods:
    - "papay": Papay's correlation (fastest, valid Pr: 0.2-15)
    - "hall-yarborough": Hall-Yarborough (accurate, valid Pr: 0.2-30)
    - "dak": Dranchuk-Abou-Kassem (most accurate, valid Pr: 0.2-30)

    :param pressure: Pressure array (psi)
    :param temperature: Temperature array (°F)
    :param gas_gravity: Gas specific gravity array (dimensionless, air=1.0)
    :param h2s_mole_fraction: H₂S mole fraction (0.0 to 1.0)
    :param co2_mole_fraction: CO₂ mole fraction (0.0 to 1.0)
    :param n2_mole_fraction: N₂ mole fraction (0.0 to 1.0)
    :param method: Correlation to use ("papay", "hall-yarborough", "dak"). Defaults to "dak".
    :return: Compressibility factor Z array (dimensionless)

    Example:
    ```python
    Z = compute_gas_compressibility_factor(P, T, G, method="dak")
    ```

    References:
    - Papay, J. (1985). "A Termelestechnologiai Parametereinek Valtozasa..."
    - Hall, K.R. and Yarborough, L. (1973). "A New Equation of ReservoirState..."
    - Dranchuk, P.M. and Abou-Kassem, J.H. (1975). "Calculation of Z Factors..."
    """
    if method == "papay":
        return compute_gas_compressibility_factor_papay(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    elif method == "hall-yarborough":
        return compute_gas_compressibility_factor_hall_yarborough(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    return compute_gas_compressibility_factor_dranchuk_abou_kassem(
        pressure=pressure,
        temperature=temperature,
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )


def compute_gas_molecular_weight(
    gas_gravity: NumberArray[NDimension],
) -> NumberArray[NDimension]:
    """
    Computes the apparent molecular weight of a gas (g/mol) from its specific gravity relative to air.

    Formula:
        MW_gas = γ_gas * MW_air

    where:
    - MW_gas is the molecular weight of the gas (g/mol)
    - γ_gas is the gas specific gravity (dimensionless, air = 1.0)
    - MW_air = 28.96 (g/mol) is the molecular weight of air

    :param gas_gravity: Specific gravity of the gas relative to air (dimensionless)
    :return: Molecular weight of the gas in grams per mole (g/mol)
    """
    if min_(gas_gravity) <= 0:
        raise ValidationError("Gas specific gravity must be greater than zero.")

    dtype = gas_gravity.dtype
    return np.multiply(gas_gravity, c.MOLECULAR_WEIGHT_AIR, dtype=dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_gas_pseudocritical_properties(
    gas_gravity: NumberArray[NDimension],
    h2s_mole_fraction: NumberOrArray[NDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[NDimension] = 0.0,
) -> tuple[NumberArray[NDimension], NumberArray[NDimension]]:
    """
    Computes pseudocritical pressure and temperature of natural gas in psi and °F.

    The pseudocritical properties are estimated using Sutton's correlation, which is
    widely used for sweet natural gas. For sour gases, the Wichert-Aziz correction
    adjusts the values based on H₂S, CO₂, and N₂ content.

    This is used as input to pseudo-reduced property models and EOS calculations.

    Sutton's correlation (for sweet gas):
        P_pc = 756.8 - 131.0 * γ_g - 3.6 * γ_g²     [psia]
        T_pc = 169.2 + 349.5 * γ_g - 74.0 * γ_g²    [°R]

    Wichert-Aziz Correction:
        ε = 120[(X_H2S + X_CO2)^0.9 - (X_H2S + X_CO2)^1.6] + 15[√X_H2S - X_H2S⁴]

        Then:
        T_pc' = T_pc - ε
        P_pc' = P_pc * T_pc' / (T_pc + X_H2S(1 - X_H2S) * ε)

    Only valid for:
        - Total acid gas fraction (H₂S + CO₂) ≤ 40%
        - H₂S mole fraction ≤ 25%

    :param gas_gravity: Gas specific gravity (dimensionless, air = 1.0).
    :param h2s_mole_fraction: Mole fraction of H₂S (dimensionless).
    :param co2_mole_fraction: Mole fraction of CO₂ (dimensionless).
    :param n2_mole_fraction: Mole fraction of N₂ (dimensionless).
    :return: Tuple (P_pc in psi, T_pc in °F)
    """
    if min_(gas_gravity) <= 0:
        raise ValidationError("Gas specific gravity must be greater than zero.")

    total_acid_gas_fraction = h2s_mole_fraction + co2_mole_fraction  # type: ignore
    if max_(total_acid_gas_fraction) > 0.40:
        raise ValidationError(
            f"Total acid gas fraction ({max_(total_acid_gas_fraction)}) exceeds 40% limit "
            "for Wichert-Aziz correction."
        )
    if max_(h2s_mole_fraction) > 0.25:
        raise ValidationError(
            f"H₂S mole fraction ({max_(h2s_mole_fraction)}) exceeds 25% limit "
            "for Wichert-Aziz correction."
        )

    # Sutton's pseudocritical properties (psia and Rankine)
    pseudocritical_pressure = 756.8 - 131.0 * gas_gravity - 3.6 * gas_gravity**2
    pseudocritical_temperature_rankine = 169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity**2

    if max_(total_acid_gas_fraction) > 0.001:
        A = h2s_mole_fraction + co2_mole_fraction
        B = h2s_mole_fraction
        epsilon = 120.0 * (A**0.9 - A**1.6) + 15.0 * (B**0.5 - B**4)

        pseudocritical_temperature_rankine -= epsilon
        pseudocritical_pressure = (
            pseudocritical_pressure
            * pseudocritical_temperature_rankine
            / (
                pseudocritical_temperature_rankine
                + h2s_mole_fraction * (1 - h2s_mole_fraction) * epsilon  # type: ignore
            )
        )

    dtype = gas_gravity.dtype
    return pseudocritical_pressure.astype(dtype), pseudocritical_temperature_rankine.astype(dtype)  # type: ignore[return-value]


def compute_gas_density(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    gas_compressibility_factor: NumberOrArray[NDimension],
) -> NumberArray[NDimension]:
    """
    Calculates the gas density (lbm/ft³) using the real gas equation of state.

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :param gas_gravity: Gas specific gravity (dimensionless, air=1.0).
    :param gas_compressibility_factor: Gas compressibility factor (dimensionless).
    :return: Gas density in lbm/ft³.
    """
    dtype = pressure.dtype
    temperature_in_rankine = temperature + dtype.type(459.67)
    # lbm/lbmol is the same numerical value as g/mol
    gas_molecular_weight_lbm_per_lbmole = compute_gas_molecular_weight(gas_gravity)
    # Density in lbm/ft3
    gas_density = (pressure * gas_molecular_weight_lbm_per_lbmole) / (
        np.multiply(gas_compressibility_factor, c.IDEAL_GAS_CONSTANT_IMPERIAL, dtype=dtype)
        * temperature_in_rankine
    )
    return gas_density  # type: ignore[return-value]


def compute_gas_viscosity(
    temperature: NumberArray[NDimension],
    gas_density: NumberArray[NDimension],
    gas_molecular_weight: NumberOrArray[NDimension],
) -> NumberArray[NDimension]:
    """
    Calculates the gas viscosity (cP) using the Lee-Gonzalez-Eakin (LGE) correlation.

    This correlation estimates gas viscosity from gas density and temperature using:

        μ_g = (k * 1e-4) * exp(x * ρ_g^y)

    where:
        - μ_g is the gas viscosity in centipoise (cP)
        - k is a temperature-dependent constant
        - x and y are empirical coefficients based on gas molecular weight and temperature
        - k = [(9.4 + 0.02 * M_g) * T^1.5] / [209 + 19 * M_g + T]
        - x = 3.5 + 986 / T + 0.01 * M_g
        - y = 2.4 - 0.2 * x
        - M_g is the gas molecular weight (lbm/lbmol)
        - ρ_g is gas density in g/cm³
        - T is temperature in Rankine

    This correlation is valid for a wide range of temperatures and pressures,
    typically from 100 °F to 400 °F and pressures up to 10,000 psi.

    :param temperature: Temperature (°F)
    :param gas_density: Gas density (lbm/ft³)
    :param gas_molecular_weight: Gas molecular weight (g/mol)
    :return: Gas viscosity in (cP)
    """
    dtype = temperature.dtype
    temperature_in_rankine = temperature + dtype.type(459.67)
    # g/mol is numerically equal to lb/lbmol
    gas_molecular_weight_lbm_per_lbmole = gas_molecular_weight
    density_in_grams_per_cm3 = gas_density * c.POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER

    k = (
        (9.4 + (0.02 * gas_molecular_weight_lbm_per_lbmole))
        * (temperature_in_rankine**1.5)
        / (209 + (19 * gas_molecular_weight_lbm_per_lbmole) + temperature_in_rankine)  # type: ignore
    )

    x = 3.5 + (986 / temperature_in_rankine) + (0.01 * gas_molecular_weight_lbm_per_lbmole)
    y = 2.4 - (0.2 * x)

    exponent = x * (density_in_grams_per_cm3**y)
    exponent = np.minimum(np.maximum(exponent, -700), 700, out=exponent)  # cap to prevent overflow

    gas_viscosity = (k * 1e-4) * np.exp(exponent)
    return np.maximum(0.0, gas_viscosity).astype(temperature.dtype)


@numba.njit(cache=True)
def compute_gas_compressibility(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    gas_compressibility_factor: NumberOrArray | None = None,
    h2s_mole_fraction: NumberOrArray[NDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[NDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[NDimension] = 0.0,
) -> NumberArray[NDimension]:
    """
    Calculates isothermal gas compressibility (C_g) in psi^-1 using a
    numerically consistent derivative of the Z-factor.

    :param pressure: Reservoir pressure (psi).
    :param temperature: Reservoir temperature (°F).
    :param gas_gravity: Specific gravity of the gas (air=1).
    :param gas_compressibility_factor: Optional pre-computed Z-factor (dimensionless).
        If provided, it will be used directly instead of (re)calculating it.
    :param h2s_mole_fraction: H2S mole fraction (0 to 1).
    :param co2_mole_fraction: CO2 mole fraction (0 to 1).
    :param n2_mole_fraction: N2 mole fraction (0 to 1).
    :return: Gas compressibility in psi⁻¹.
    """
    if min_(pressure) <= 0 or min_(temperature) <= 0 or min_(gas_gravity) <= 0:
        raise ValidationError("Pressure, temperature, and gas specific gravity must be positive.")

    if gas_compressibility_factor is not None:
        Z = gas_compressibility_factor
    else:
        Z = compute_gas_compressibility_factor(
            gas_gravity=gas_gravity,
            pressure=pressure,
            temperature=temperature,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )

    # Numerical derivative dZ/dP (central difference)
    dP = 1e-4 * pressure
    dP = np.where(dP == 0.0, 1e-6, dP)

    P_plus = pressure + dP
    P_minus = pressure - dP

    # Handle near-zero pressure safely
    use_forward = P_minus <= 0.0

    # Compute Z at perturbed pressures
    Z_plus = compute_gas_compressibility_factor(
        gas_gravity=gas_gravity,
        pressure=P_plus,
        temperature=temperature,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )

    Z_minus = compute_gas_compressibility_factor(
        gas_gravity=gas_gravity,
        pressure=P_minus,
        temperature=temperature,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )

    # Central difference
    dZ_dP_central = (Z_plus - Z_minus) / (2.0 * dP)
    # Forward difference where needed
    dZ_dP_forward = (Z_plus - Z) / dP

    # Select scheme
    dZ_dP = np.where(use_forward, dZ_dP_forward, dZ_dP_central)
    gas_compressibility = (1.0 / pressure) - (1.0 / Z) * dZ_dP
    dtype = pressure.dtype
    return np.maximum(0.0, gas_compressibility).astype(dtype)  # type: ignore[return-value]
