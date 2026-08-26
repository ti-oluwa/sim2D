# Faults and Fractures

## Overview

Faults and fractures are geological discontinuities that profoundly affect fluid flow in reservoirs. A fault is a surface along which rock layers have been displaced, often creating a barrier (or sometimes a conduit) to cross-fault flow. A fracture is a crack or opening in the rock that creates an enhanced permeability pathway. Both are modeled in BORES through transmissibility multipliers, following the same approach used by commercial simulators.

The transmissibility multiplier approach works by scaling the inter-block flow transmissibility across fault or fracture planes. A multiplier of 0.0001 means the fault reduces cross-fault flow to 0.01% of what the matrix permeability alone would allow (a nearly sealing fault). A multiplier of 10.0 means the fracture enhances flow to 10 times the matrix value. A multiplier of 1.0 has no effect (no fault or fracture).

This approach has several advantages. It does not require changes to the grid geometry, so wells, boundary conditions, and property distributions remain unaffected. The fault masks are computed once during model setup and apply minimal overhead at runtime. Multiple faults can be combined through vectorized operations, and the Numba JIT-compiled mask generation functions ensure fast setup even for large grids.

---

## Factory Functions

BORES provides four factory functions for creating common fault and fracture configurations. These are the recommended entry points, as they handle geometry validation and mask generation automatically.

### Vertical Sealing Faults

The most common fault type is a vertical plane that acts as a barrier to cross-fault flow:

```python
from bores.fractures import vertical_sealing_fault, apply_fracture

# Vertical fault at x=25, spanning the entire grid
fault = vertical_sealing_fault(
    fault_id="main_fault",
    orientation="x",
    index=25,
    permeability_multiplier=1e-4,  # 99.99% sealing
)

model = apply_fracture(model=reservoir_model, fracture=fault)
```

The `orientation` parameter specifies which axis the fault plane is perpendicular to. An `"x"`-oriented fault creates a plane in the y-z direction at the specified x-index, blocking flow in the x-direction across that plane.

You can limit the fault extent using range parameters:

```python
# Fault in upper layers only
shallow_fault = vertical_sealing_fault(
    fault_id="shallow_fault",
    orientation="y",
    index=40,
    z_range=(0, 15),  # Only in upper 16 layers
    permeability_multiplier=5e-5,
)

# Fault with limited lateral extent
partial_fault = vertical_sealing_fault(
    fault_id="fault_f3",
    orientation="x",
    index=50,
    permeability_multiplier=0.01,
    y_range=(0, 60),  # Limited lateral extent
    z_range=(10, 35),  # Offsetting layers 10-35
)
```

### Inclined Faults

Non-vertical faults are modeled using a slope parameter that tilts the fault plane:

```python
from bores.fractures import inclined_sealing_fault

# Fault dipping at 60 degrees (slope = tan(60) = 1.73)
dipping_fault = inclined_sealing_fault(
    fault_id="dipping_fault",
    orientation="y",  # Strikes in y-direction
    index=30,  # Intersects at y=30
    slope=1.73,  # dz/dx (eastward dip)
    intercept=5.0,  # Fault at z=5 when x=0
    permeability_multiplier=1e-4,
)
```

The slope defines the fault plane equation. For y-oriented faults, the equation is $z = \text{intercept} + \text{slope} \times x$. The fault mask marks grid cells that intersect this plane.

### Damage Zone Faults

Major faults often have a zone of damaged rock with altered permeability and porosity:

```python
from bores.fractures import damage_zone_fault

fault_with_damage = damage_zone_fault(
    fault_id="thrust_fault",
    orientation="x",
    cell_range=(48, 52),  # 5 cells wide
    permeability_multiplier=1e-5,  # Very low cross-fault flow
    zone_permeability=10.0,  # Damaged rock: 10 mD (vs 100 mD matrix)
    zone_porosity=0.12,  # Reduced porosity (vs 0.20 matrix)
    z_range=(15, 45),
)
```

The `cell_range` parameter specifies the width of the damage zone in grid cells. The `zone_permeability` and `zone_porosity` values are applied to all cells within the zone, overriding the original matrix properties.

### Conductive Fracture Networks

Natural or hydraulic fractures that enhance permeability are modeled with `conductive_fracture_network`:

```python
from bores.fractures import conductive_fracture_network

fracture_swarm = conductive_fracture_network(
    fracture_id="fracture_corridor",
    orientation="y",
    cell_range=(15, 18),  # 3-cell-wide corridor
    fracture_permeability=5000.0,  # High perm: 5 Darcy
    fracture_porosity=0.01,  # Low storage (fractures)
    permeability_multiplier=10.0,  # 10x enhanced cross-corridor flow
    z_range=(20, 40),
)
```

Unlike sealing faults where the multiplier reduces flow, conductive fractures use multipliers greater than 1.0 to enhance cross-fracture flow. The `fracture_permeability` and `fracture_porosity` are applied to cells within the fracture corridor.

---

## Applying Faults to a Model

### Single Fault

```python
from bores.fractures import apply_fracture

model = apply_fracture(model=reservoir_model, fracture=fault)
```

### Multiple Faults

For compartmentalized reservoirs with multiple faults, use `apply_fractures` to apply them all at once:

```python
from bores.fractures import apply_fractures, vertical_sealing_fault

faults = [
    vertical_sealing_fault("fault_1", "x", 25, permeability_multiplier=1e-4),
    vertical_sealing_fault("fault_2", "y", 35, permeability_multiplier=5e-4),
    vertical_sealing_fault("fault_3", "x", 60, permeability_multiplier=2e-4),
]

model = apply_fractures(reservoir_model, *faults)
```

`apply_fractures` handles the efficient combination of multiple fault masks using vectorized operations.

---

## Horizontal Barriers

You can model horizontal barriers (shale layers, tight streaks, cemented zones) using the `"z"` orientation:

```python
shale_barrier = vertical_sealing_fault(
    fault_id="shale_layer",
    orientation="z",  # Horizontal plane
    index=12,  # At layer 12
    permeability_multiplier=1e-6,
    x_range=(0, 80),
    y_range=(0, 60),
)
```

This creates a horizontal barrier that reduces vertical flow across layer 12 to 0.0001% of the matrix transmissibility. This is useful for modeling interbedded shales, tight carbonate stringers, or any low-permeability layer that compartmentalizes the reservoir vertically.

---

## Fracture Geometry

The `FractureGeometry` class defines the spatial extent and orientation of a fault or fracture. The factory functions create this automatically, but you can also build it directly:

```python
from bores.fractures import FractureGeometry

# Vertical fault at x=25, limited extent
geom = FractureGeometry(
    orientation="x",
    x_range=(25, 25),  # Single cell plane
    y_range=(10, 40),  # Lateral extent
    z_range=(0, 20),  # Vertical extent
    slope=0.0,  # Vertical (no dip)
    intercept=0.0,
)
```

The range parameters use inclusive grid indices. Setting a range to `None` means the fault extends across the full grid dimension in that direction.

---

## Validation

Before applying faults, you can validate their configuration against the grid shape:

```python
from bores.fractures import validate_fracture

errors = validate_fracture(fracture=fault, grid_shape=(100, 80, 50))
if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

Validation checks that all range indices are within the grid bounds and that the geometry is internally consistent (minimum index is less than or equal to maximum index, orientation has the required primary range).

---

## Choosing Permeability Multipliers

| Fault Type | Multiplier Range | Description |
|---|---|---|
| Completely sealing | 1e-6 to 1e-5 | No cross-fault flow |
| Highly sealing | 1e-4 to 1e-3 | Very little cross-fault flow |
| Partially sealing | 0.01 to 0.1 | Some cross-fault flow |
| Leaky fault | 0.1 to 0.5 | Significant cross-fault flow |
| No barrier | 1.0 | No effect on flow |
| Conductive fracture | 1.0 to 100.0 | Enhanced flow |

In practice, fault transmissibility multipliers are calibrated through history matching. Start with estimates from geological interpretation (seismically mapped faults are often more sealing than sub-seismic fractures) and refine based on well test interference data, tracer studies, or production history.

!!! tip "Damage Zone Properties"

    For damage zone faults, typical property alterations are:

    - Permeability reduction: 10x to 100x relative to matrix
    - Porosity reduction: 20% to 40% of matrix values
    - Zone width: 1% to 10% of fault displacement
