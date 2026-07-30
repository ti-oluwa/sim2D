# Array PVT Correlations

## Overview

The `bores.correlations.arrays` module contains vectorized versions of every scalar PVT correlation from `bores.correlations.scalars`. Where the scalar module operates on single float values, the array module operates on NumPy arrays of any shape and dimension. These are the functions the simulator calls internally to compute fluid properties across the entire reservoir grid in a single pass, and you can use them for post-processing, custom property computations, or building property grids from scratch.

Every function in this module accepts `NDimensionalGrid` arrays (typed NumPy arrays parameterized by dimension) and returns arrays of the same shape. The functions use the same correlations, formulas, and valid ranges as their scalar counterparts. Many are compiled with Numba's `@numba.njit` decorator for performance-critical inner loops, while CoolProp-based functions iterate over array elements internally.

All units are the same as the scalar module: pressure in psi, temperature in degrees Fahrenheit, density in lbm/ft³, viscosity in cP, and formation volume factors in bbl/STB or ft³/SCF.

```python
from bores.correlations.arrays import (
    compute_oil_formation_volume_factor,
    compute_gas_compressibility_factor,
    compute_oil_viscosity,
)
```

!!! info "Relationship to Scalar Correlations"

    Every public function in `bores.correlations.arrays` has a corresponding scalar version in `bores.correlations.scalars` with the same name and the same parameters. The only difference is that array functions accept and return NumPy arrays instead of floats. See the [Scalar Correlations](correlations-scalar.md) page for detailed descriptions of each correlation's formula, valid ranges, and physical interpretation.

---

## How Array Functions Differ from Scalar

The key differences between the array and scalar modules are:

**Input types.** Scalar functions accept `float` parameters. Array functions accept `NDimensionalGrid[NDimension]` parameters, which are NumPy arrays of shape `(nx,)`, `(nx, ny)`, or `(nx, ny, nz)` depending on the simulation dimensionality.

**Broadcasting.** Array functions follow NumPy broadcasting rules. You can pass a scalar value for a parameter that is uniform across the grid (like gas gravity) and it will be broadcast against the grid-shaped parameters (like pressure). Some parameters explicitly accept `NumberOrArray` types for this purpose.

**Validation.** Scalar functions raise `ValidationError` immediately when a single invalid value is detected. Array functions use `min_()` and `max_()` helpers to check the entire grid at once, raising an error if any element violates the constraint.

**Precision.** Array functions respect the global precision setting (`use_32bit_precision()` / `use_64bit_precision()`). Results are cast to the same dtype as the input arrays using the `get_dtype()` helper.

**Numba compilation.** Many array functions are decorated with `@numba.njit(cache=True)` for JIT compilation. The first call triggers compilation (typically 1-5 seconds), and subsequent calls use the cached machine code. CoolProp-based functions cannot be Numba-compiled and iterate over array elements with a Python loop instead.

---

## Quick Example

```python
import numpy as np
from bores.correlations.arrays import (
    compute_oil_formation_volume_factor,
    compute_gas_compressibility_factor,
    compute_oil_viscosity,
)

# Grid-shaped pressure and temperature arrays
grid_shape = (10, 10, 3)
pressure = np.full(grid_shape, 3000.0, dtype=np.float32)
temperature = np.full(grid_shape, 200.0, dtype=np.float32)
bubble_point = np.full(grid_shape, 2500.0, dtype=np.float32)
oil_sg = np.full(grid_shape, 0.85, dtype=np.float32)
gas_gravity = np.full(grid_shape, 0.70, dtype=np.float32)
gor = np.full(grid_shape, 500.0, dtype=np.float32)
oil_comp = np.full(grid_shape, 1e-5, dtype=np.float32)

# Compute Bo for every cell at once
Bo_grid = compute_oil_formation_volume_factor(
    pressure=pressure,
    temperature=temperature,
    bubble_point_pressure=bubble_point,
    oil_specific_gravity=oil_sg,
    gas_gravity=gas_gravity,
    gas_to_oil_ratio=gor,
    oil_compressibility=oil_comp,
)
print(f"Bo range: {Bo_grid.min():.4f} to {Bo_grid.max():.4f} bbl/STB")
```

---

## Function Reference

The table below lists every public function in the array module. Each function has the same name, the same parameters, and uses the same correlation as its scalar counterpart. Click through to the [Scalar Correlations](correlations-scalar.md) page for detailed descriptions of formulas, valid ranges, and physical interpretation.

### Generic Fluid Properties (CoolProp)

| Function | Returns | Unit |
| --- | --- | --- |
| `compute_fluid_density(pressure, temperature, fluid)` | Fluid density grid | lbm/ft³ |
| `compute_fluid_viscosity(pressure, temperature, fluid)` | Fluid viscosity grid | cP |
| `compute_fluid_compressibility_factor(pressure, temperature, fluid)` | Z-factor grid | dimensionless |
| `compute_fluid_compressibility(pressure, temperature, fluid)` | Compressibility grid | psi-1 |

These functions iterate over each element of the input arrays and call CoolProp for each cell individually. They are slower than the correlation-based functions below but work for any fluid supported by CoolProp (CO2, N2, Methane, n-Octane, etc.).

### Gas Gravity

| Function | Returns | Unit |
| --- | --- | --- |
| `compute_gas_gravity(gas)` | Gas specific gravity | dimensionless |
| `compute_gas_gravity_from_density(pressure, temperature, density)` | Gas gravity from measured density | dimensionless |

!!! note

    `compute_gas_gravity(gas)` takes a single string and returns a scalar float, identical to the scalar version. It does not operate on arrays.

### Oil Properties

| Function | Returns | Unit |
| --- | --- | --- |
| `compute_oil_specific_gravity(oil_density, pressure, temperature, oil_compressibility)` | Oil specific gravity grid | dimensionless |
| `compute_oil_api_gravity(oil_specific_gravity)` | API gravity grid | degrees |
| `compute_oil_formation_volume_factor_standing(temperature, oil_specific_gravity, gas_gravity, gas_to_oil_ratio)` | Bo grid (Standing) | bbl/STB |
| `compute_oil_formation_volume_factor_vazquez_and_beggs(temperature, oil_specific_gravity, gas_gravity, gas_to_oil_ratio)` | Bo grid (Vazquez-Beggs) | bbl/STB |
| `correct_oil_fvf_for_pressure(saturated_oil_fvf, oil_compressibility, bubble_point_pressure, current_pressure)` | Corrected Bo grid | bbl/STB |
| `compute_oil_formation_volume_factor(pressure, temperature, bubble_point_pressure, oil_specific_gravity, gas_gravity, gas_to_oil_ratio, oil_compressibility)` | Bo grid (unified) | bbl/STB |
| `compute_dead_oil_viscosity_modified_beggs(temperature, oil_specific_gravity)` | Dead oil viscosity grid | cP |
| `compute_oil_viscosity(pressure, temperature, bubble_point_pressure, oil_specific_gravity, gas_to_oil_ratio, gor_at_bubble_point_pressure)` | Oil viscosity grid | cP |
| `compute_oil_compressibility(pressure, temperature, bubble_point_pressure, oil_api_gravity, gas_gravity, gor_at_bubble_point_pressure, ...)` | Oil compressibility grid | psi-1 |
| `compute_live_oil_density(api_gravity, gas_gravity, gas_to_oil_ratio, formation_volume_factor)` | Live oil density grid | lbm/ft³ |

### Gas Properties

| Function | Returns | Unit |
| --- | --- | --- |
| `compute_gas_molecular_weight(gas_gravity)` | Gas molecular weight grid | g/mol |
| `compute_gas_pseudocritical_properties(gas_gravity, h2s_mole_fraction, co2_mole_fraction, n2_mole_fraction)` | (Ppc, Tpc) tuple of grids | psi, R |
| `compute_gas_formation_volume_factor(pressure, temperature, gas_compressibility_factor)` | Bg grid | ft³/SCF |
| `compute_gas_compressibility_factor_papay(pressure, temperature, gas_gravity, ...)` | Z-factor grid (Papay) | dimensionless |
| `compute_gas_compressibility_factor_hall_yarborough(pressure, temperature, gas_gravity, ...)` | Z-factor grid (Hall-Yarborough) | dimensionless |
| `compute_gas_compressibility_factor_dranchuk_abou_kassem(pressure, temperature, gas_gravity, ...)` | Z-factor grid (DAK) | dimensionless |
| `compute_gas_compressibility_factor(pressure, temperature, gas_gravity, ..., method)` | Z-factor grid (unified) | dimensionless |
| `compute_gas_density(pressure, temperature, gas_gravity, gas_compressibility_factor)` | Gas density grid | lbm/ft³ |
| `compute_gas_viscosity(temperature, gas_density, gas_molecular_weight)` | Gas viscosity grid (LGE) | cP |
| `compute_gas_compressibility(pressure, temperature, gas_gravity, ...)` | Gas compressibility grid | psi-1 |

### Water Properties

| Function | Returns | Unit |
| --- | --- | --- |
| `compute_water_formation_volume_factor(water_density, salinity)` | Bw grid | bbl/STB |
| `compute_water_formation_volume_factor_mccain(pressure, temperature, salinity, gas_solubility)` | Bw grid (McCain) | bbl/STB |
| `compute_water_viscosity(temperature, salinity, pressure)` | Water viscosity grid | cP |
| `compute_water_density(pressure, temperature, gas_gravity, salinity, gas_solubility_in_water, gas_free_water_formation_volume_factor)` | Live water density grid | lbm/ft³ |
| `compute_water_density_mccain(pressure, temperature, salinity)` | Water density grid (McCain) | lbm/ft³ |
| `compute_water_density_batzle(pressure, temperature, salinity)` | Water density grid (Batzle-Wang) | lbm/ft³ |
| `compute_water_compressibility(pressure, temperature, bubble_point_pressure, gas_formation_volume_factor, gas_solubility_in_water, gas_free_water_formation_volume_factor, salinity)` | Water compressibility grid | psi-1 |
| `compute_gas_free_water_formation_volume_factor(pressure, temperature)` | Gas-free Bw grid | bbl/STB |

### Bubble Point and GOR

| Function | Returns | Unit |
| --- | --- | --- |
| `compute_oil_bubble_point_pressure(gas_gravity, oil_api_gravity, temperature, gas_to_oil_ratio)` | Bubble point grid | psi |
| `compute_water_bubble_point_pressure(temperature, gas_solubility_in_water, salinity, gas)` | Water bubble point grid | psi |
| `compute_gas_to_oil_ratio(pressure, temperature, bubble_point_pressure, gas_gravity, oil_api_gravity, ...)` | GOR grid (Vazquez-Beggs) | SCF/STB |
| `compute_gas_to_oil_ratio_standing(pressure, oil_api_gravity, gas_gravity)` | GOR grid (Standing) | SCF/STB |
| `estimate_solution_gor(pressure, temperature, oil_api_gravity, gas_gravity)` | Iterative GOR estimate grid | SCF/STB |
| `estimate_bubble_point_pressure_standing(oil_api_gravity, gas_gravity, observed_gas_to_oil_ratio)` | Estimated Pb grid | psi |
| `compute_gas_solubility_in_water(pressure, temperature, salinity, gas)` | Gas solubility in water grid | SCF/STB |

### Multi-Phase and Volumetrics

| Function | Returns | Unit |
| --- | --- | --- |
| `compute_total_fluid_compressibility(water_saturation, oil_saturation, water_compressibility, oil_compressibility, gas_saturation, gas_compressibility)` | Total compressibility grid | psi-1 |
| `compute_hydrocarbon_in_place(area, thickness, porosity, phase_saturation, formation_volume_factor, net_to_gross_ratio, hydrocarbon_type)` | HCIP grid | STB or SCF |
| `convert_surface_rate_to_reservoir(surface_rate, formation_volume_factor)` | Reservoir rate grid | bbl/day |
| `convert_reservoir_rate_to_surface(reservoir_rate, formation_volume_factor)` | Surface rate grid | STB/day |

### Miscible Flooding (Todd-Longstaff)

| Function | Returns | Unit |
| --- | --- | --- |
| `compute_miscibility_transition_factor(pressure, minimum_miscibility_pressure, transition_width)` | Transition factor grid | dimensionless |
| `compute_effective_todd_longstaff_omega(pressure, base_omega, minimum_miscibility_pressure, transition_width)` | Effective omega grid | dimensionless |
| `compute_todd_longstaff_effective_viscosity(oil_viscosity, solvent_viscosity, solvent_concentration, omega)` | Effective viscosity grid | cP |
| `compute_todd_longstaff_effective_density(oil_density, solvent_density, oil_viscosity, solvent_viscosity, solvent_concentration, omega)` | Effective density grid | lbm/ft³ |

---

## Performance Notes

The array functions are designed for grid-level computation and are significantly faster than calling the scalar versions in a Python loop over cells. For a 100x100x20 grid (200,000 cells):

- **Numba-compiled functions** (most correlations): Run at near-C speed after the first compilation. Typical computation times are 1-10 ms per grid evaluation.
- **CoolProp-based functions** (`compute_fluid_density`, `compute_fluid_viscosity`, etc.): Iterate over elements in Python calling CoolProp for each cell. These are 10-100x slower than correlation-based functions but provide equation-of-state accuracy for non-standard fluids.

If you need CoolProp-level accuracy for standard reservoir fluids, consider computing properties once and storing them in a PVT table using `bores.PVTTables`, then using table lookup during simulation instead of repeated CoolProp calls.
