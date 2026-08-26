# API Reference

## Overview

This section documents the programmatic interfaces of the BORES framework. The reference is organized into three areas: scalar PVT correlations (single-value functions for computing fluid properties), array PVT correlations (vectorized equivalents that operate on numpy grids), and the full public API surface exported from the `bores` package.

The correlations modules are the computational foundation of the simulator. Every time BORES needs to compute an oil formation volume factor, gas compressibility, water viscosity, or any other pressure-volume-temperature property, it calls one of these functions. You can also call them directly in your own scripts for standalone PVT calculations, unit conversions, or validation against laboratory data.

---

## Module Organization

| Module | Description |
| --- | --- |
| `bores.correlations.scalars` | Scalar PVT correlation functions (single float inputs and outputs) |
| `bores.correlations.arrays` | Array PVT correlation functions (numpy array inputs and outputs) |
| `bores` | Top-level package re-exporting all public classes, functions, and constants |

The scalar correlations in `bores.correlations.scalars` are also accessible through `bores.correlations` directly (via a wildcard re-export). The array correlations must be imported from `bores.correlations.arrays` explicitly.

---

## How to Use This Reference

**Scalar correlations** are for point calculations: computing a single property value from a single set of conditions. Use them for quick checks, unit conversions, or building custom PVT tables.

```python
from bores.correlations.scalars import compute_oil_formation_volume_factor_standing

Bo = compute_oil_formation_volume_factor_standing(
    temperature=200.0,  # degrees F
    oil_specific_gravity=0.85,
    gas_gravity=0.7,
    gas_to_oil_ratio=500.0,  # SCF/STB
)
print(f"Bo = {Bo:.4f} bbl/STB")
```

**Array correlations** are for grid-level calculations: computing property grids from pressure, temperature, and composition grids. The simulator uses these internally, but you can also use them for post-processing or custom property computations.

```python
from bores.correlations.arrays import compute_oil_formation_volume_factor

Bo_grid = compute_oil_formation_volume_factor(
    pressure=pressure_grid,
    temperature=temperature_grid,
    bubble_point_pressure=Pb_grid,
    oil_specific_gravity=0.85,
    gas_gravity=0.7,
    gas_to_oil_ratio=Rs_grid,
    oil_compressibility=co_grid,
)
```

**Full API** lists every class, function, and constant exported from the `bores` package. Use it as a lookup when you know the name of something but need to confirm the import path or check the signature.

---

## Pages

- [Scalar Correlations](correlations-scalar.md) - Single-value PVT functions for point calculations
- [Array Correlations](correlations-array.md) - Vectorized PVT functions for grid-level calculations
- [Full API](full-api.md) - Complete listing of the public API surface
