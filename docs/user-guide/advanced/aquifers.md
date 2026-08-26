# Aquifer Models

## Overview

Aquifers are large bodies of water-saturated rock connected to the reservoir that provide pressure support through water influx. When reservoir pressure drops due to production, the pressure differential drives water from the aquifer into the reservoir, partially replacing the produced fluid volume and slowing pressure decline. The strength and timing of this pressure support depend on the aquifer's size, permeability, compressibility, and geometry.

BORES provides the Carter-Tracy aquifer model, which is the industry standard for modeling finite aquifer behavior in black-oil simulators. The Carter-Tracy model captures the transient nature of aquifer response: early in the aquifer's life, the pressure signal has not reached the outer boundary, so the aquifer behaves as if it were infinite. Later, the outer boundary begins to affect the response, and the influx rate declines as the aquifer's finite energy is depleted.

The Carter-Tracy model sits between two simpler alternatives. A constant pressure boundary (Dirichlet condition) models an infinite aquifer that provides unlimited pressure support, which is overly optimistic for most real aquifers. A no-flow boundary models the complete absence of aquifer support, which is overly conservative when aquifer communication exists. The Carter-Tracy model provides a physically realistic middle ground.

---

## Carter-Tracy Model

The Carter-Tracy model computes water influx using the Van Everdingen-Hurst dimensionless water influx solution combined with a superposition (convolution) integral over the pressure history:

$$Q(t) = B \cdot \sum_{i} \Delta P(t_i) \cdot W_D'(t_D - t_{Di})$$

where:

- $B$ is the aquifer constant, computed from physical properties
- $\Delta P(t_i)$ is the pressure change at the reservoir-aquifer boundary at time $t_i$
- $W_D'(t_D)$ is the dimensionless water influx derivative
- $t_D$ is the dimensionless time

The aquifer constant $B$ captures the aquifer's total influx capacity:

$$B = \frac{1.119 \cdot \phi \cdot c_t \cdot (r_e^2 - r_w^2) \cdot h \cdot \theta}{360 \cdot \mu_w}$$

where $\phi$ is porosity, $c_t$ is total compressibility, $r_e$ and $r_w$ are the outer and inner radii, $h$ is thickness, $\theta$ is the angle of aquifer contact, and $\mu_w$ is water viscosity.

---

## Physical Properties Mode (Recommended)

The recommended way to configure a Carter-Tracy aquifer is by specifying the physical properties of the aquifer rock and fluid:

```python
from bores.boundary_conditions import (
    CarterTracyAquifer,
    GridBoundaryCondition,
    BoundaryConditions,
)

# Edge water drive with known aquifer properties
edge_aquifer = CarterTracyAquifer(
    aquifer_permeability=500.0,  # mD
    aquifer_porosity=0.25,  # fraction
    aquifer_compressibility=3e-6,  # psi-1 (rock + water)
    water_viscosity=0.5,  # cP
    inner_radius=1000.0,  # ft (reservoir-aquifer contact)
    outer_radius=10000.0,  # ft (aquifer outer boundary)
    aquifer_thickness=50.0,  # ft
    initial_pressure=2500.0,  # psi
    angle=180.0,  # degrees (half-circle, edge drive)
)
```

In this mode, BORES computes the aquifer constant $B$ and the dimensionless time $t_D$ from first principles. This gives you physically meaningful parameters that you can vary independently for sensitivity analysis.

### Parameter Descriptions

| Parameter | Unit | Description |
|---|---|---|
| `aquifer_permeability` | mD | Aquifer rock permeability |
| `aquifer_porosity` | fraction | Aquifer rock porosity |
| `aquifer_compressibility` | psi-1 | Total compressibility (rock + water) |
| `water_viscosity` | cP | Aquifer water viscosity |
| `inner_radius` | ft | Radius at reservoir-aquifer contact |
| `outer_radius` | ft | Outer radius of aquifer extent |
| `aquifer_thickness` | ft | Net aquifer thickness |
| `initial_pressure` | psi | Initial equilibrium pressure |
| `angle` | degrees | Angular extent of aquifer contact |

### Aquifer Geometry

The `angle` parameter controls how much of the reservoir perimeter is in contact with the aquifer:

- **360 degrees**: Full radial aquifer (reservoir surrounded by aquifer)
- **180 degrees**: Half-circle (edge water drive from one side)
- **90 degrees**: Quarter-circle (corner aquifer)

The `inner_radius` should match the effective radius of the reservoir at the aquifer contact. For a rectangular reservoir, this is approximately $\sqrt{A/\pi}$ where $A$ is the reservoir area.

### Dimensionless Radius Ratio

The ratio $r_D = r_e / r_w$ (outer radius / inner radius) controls how quickly the aquifer's finite boundaries affect the response:

- **Small $r_D$ (2 to 5)**: Small aquifer, boundary effects appear early, influx declines quickly
- **Large $r_D$ (10 to 50)**: Large aquifer, behaves as infinite for a long time before declining
- **Very large $r_D$ (> 100)**: Effectively infinite aquifer within the simulation timeframe

---

## Calibrated Constant Mode

When the physical properties of the aquifer are uncertain (common in practice), you can specify a pre-computed aquifer constant from history matching:

```python
calibrated_aquifer = CarterTracyAquifer(
    aquifer_constant=5000.0,  # Pre-computed B value
    dimensionless_radius_ratio=10.0,  # r_e / r_w
    initial_pressure=2500.0,
)
```

In this mode, the aquifer constant $B$ is used directly without computing it from physical properties. This is useful when you have calibrated $B$ from a history match but do not know the individual aquifer properties that produce that value.

---

## Applying the Aquifer

The aquifer is applied as a boundary condition on one or more grid faces. Since `BoundaryConditions` maps property names to `GridBoundaryCondition` objects, you assign the aquifer to the pressure boundary:

```python
from bores.boundary_conditions import BoundaryConditions, GridBoundaryCondition

# Edge aquifer on the left face
boundary_conditions = BoundaryConditions(
    conditions={
        "pressure": GridBoundaryCondition(
            left=edge_aquifer,
        ),
    },
)

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    boundary_conditions=boundary_conditions,
)
```

### Bottom Water Drive

For a bottom water drive (aquifer below the reservoir), apply the aquifer to the bottom face:

```python
bottom_aquifer = CarterTracyAquifer(
    aquifer_permeability=800.0,
    aquifer_porosity=0.28,
    aquifer_compressibility=4e-6,
    water_viscosity=0.4,
    inner_radius=2000.0,
    outer_radius=15000.0,
    aquifer_thickness=100.0,
    initial_pressure=2800.0,
    angle=360.0,  # Full contact from below
)

boundary_conditions = BoundaryConditions(
    conditions={
        "pressure": GridBoundaryCondition(
            bottom=bottom_aquifer,
        ),
    },
)
```

---

## Aquifer Behavior Over Time

The Carter-Tracy aquifer response has three characteristic phases:

1. **Early time (infinite-acting)**: The pressure disturbance has not reached the outer boundary. All aquifer sizes produce the same influx rate at early time. The dimensionless influx function $W_D'$ follows the infinite-acting solution.

2. **Transition**: The pressure disturbance reaches the outer boundary. The influx rate begins to deviate from the infinite-acting solution. Smaller aquifers (smaller $r_D$) enter this phase sooner.

3. **Late time (boundary-dominated)**: The aquifer's finite energy is being depleted. The influx rate declines, and the aquifer pressure converges toward the reservoir pressure. Eventually, the aquifer can no longer provide meaningful pressure support.

The transition time scales approximately as $t_D \propto r_D^2$. A ten-fold increase in the radius ratio delays boundary effects by approximately 100 times.

---

## Comparison of Aquifer Models

| Model | Complexity | Pressure Support | Best For |
|---|---|---|---|
| No-flow boundary | None | None | Sealed reservoirs, initial screening |
| Constant pressure | Low | Infinite | Strong aquifers, short simulations |
| Carter-Tracy | Moderate | Finite, declining | Production forecasting, history matching |

!!! tip "When to Use Carter-Tracy"

    Use Carter-Tracy when:

    - You need realistic pressure decline forecasts
    - The aquifer has a known or estimated finite extent
    - You are history matching production data and need pressure support calibration
    - The simulation runs long enough for aquifer depletion to matter

    Use constant pressure when:

    - The aquifer is very large relative to the reservoir
    - The simulation period is short relative to the aquifer response time
    - You want a simple upper bound on aquifer support
