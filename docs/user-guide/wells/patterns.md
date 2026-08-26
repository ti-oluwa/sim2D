# Well Patterns

## Overview

Well placement is one of the most important decisions in field development planning. The arrangement of production and injection wells (the "well pattern") controls sweep efficiency, injection support, and ultimately recovery factor. BORES does not have a dedicated pattern generation function, but the `duplicate()` method and the factory functions make it straightforward to build common patterns programmatically.

This page demonstrates how to construct industry-standard well patterns and provides guidance on choosing the right pattern for different reservoir geometries and recovery mechanisms.

---

## Five-Spot Pattern

The five-spot is the most common waterflood pattern. It places four producers at the corners of a square with one injector at the center. In a repeated five-spot, each injector is shared by four producers, giving a 1:1 injection-to-production well ratio.

```python
import bores

grid_shape = (30, 30, 5)

# Define common produced fluids
produced_fluids = [
    bores.ProducedFluid(
        name="Oil", phase=bores.FluidPhase.OIL, specific_gravity=0.85, molecular_weight=200.0
    ),
    bores.ProducedFluid(
        name="Water", phase=bores.FluidPhase.WATER, specific_gravity=1.0, molecular_weight=18.015
    ),
    bores.ProducedFluid(
        name="Gas", phase=bores.FluidPhase.GAS, specific_gravity=0.65, molecular_weight=16.04
    ),
]

# Production control
prod_control = bores.ProducerRateControl(
    primary_phase=bores.FluidPhase.OIL,
    primary_control=bores.AdaptiveRateControl(
        target_rate=-300.0,
        target_phase="oil",
        bhp_limit=1000.0,
    ),
    secondary_clamp=bores.ProductionClamp(),
)

# Four corner producers
corners = [(5, 5), (5, 25), (25, 5), (25, 25)]
producers = []
for i, (x, y) in enumerate(corners):
    producers.append(
        bores.production_well(
            well_name=f"PROD-{i + 1}",
            perforating_intervals=[((x, y, 0), (x, y, 4))],
            radius=0.25,
            produced_fluids=produced_fluids,
            control=prod_control,
        )
    )

# Center injector
injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((15, 15, 0), (15, 15, 4))],
    radius=0.25,
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.0,
        molecular_weight=18.015,
    ),
    control=bores.RateControl(
        target_rate=1200.0,
        bhp_limit=5000.0,
    ),
)

wells = bores.wells_(producers=producers, injectors=[injector])
```

The five-spot pattern provides uniform sweep in homogeneous, isotropic reservoirs. In anisotropic reservoirs (where permeability differs between x and y directions), the pattern may need to be elongated in the low-permeability direction to maintain symmetric sweep.

---

## Line Drive Pattern

A line drive places producers and injectors in alternating parallel rows. This pattern is simpler to operate than a five-spot and works well for elongated reservoirs or when directional permeability favors flow in one direction.

```python
import bores

grid_shape = (40, 20, 5)

# Row of producers at x=30
producers = []
for j in range(2, 18, 4):  # y = 2, 6, 10, 14
    producers.append(
        bores.production_well(
            well_name=f"PROD-{len(producers) + 1}",
            perforating_intervals=[((30, j, 0), (30, j, 4))],
            radius=0.25,
            produced_fluids=produced_fluids,  # From earlier
            control=prod_control,
        )
    )

# Row of injectors at x=10
injectors = []
for j in range(2, 18, 4):
    injectors.append(
        bores.injection_well(
            well_name=f"INJ-{len(injectors) + 1}",
            perforating_intervals=[((10, j, 0), (10, j, 4))],
            radius=0.25,
            injected_fluid=bores.InjectedFluid(
                name="Water",
                phase=bores.FluidPhase.WATER,
                specific_gravity=1.0,
                molecular_weight=18.015,
            ),
            control=bores.RateControl(target_rate=600.0, bhp_limit=5000.0),
        )
    )

wells = bores.wells_(producers=producers, injectors=injectors)
```

---

## Peripheral Flood

A peripheral flood places injectors around the edges of the reservoir and producers in the interior. This pattern is common for maintaining pressure support from the periphery, mimicking natural aquifer drive.

```python
import bores

grid_shape = (30, 30, 5)

# Edge injectors
edge_positions = [
    (0, 15),
    (15, 0),
    (29, 15),
    (15, 29),  # Mid-edge
    (0, 0),
    (0, 29),
    (29, 0),
    (29, 29),  # Corners
]
injectors = []
for i, (x, y) in enumerate(edge_positions):
    injectors.append(
        bores.injection_well(
            well_name=f"INJ-{i + 1}",
            perforating_intervals=[((x, y, 0), (x, y, 4))],
            radius=0.25,
            injected_fluid=bores.InjectedFluid(
                name="Water",
                phase=bores.FluidPhase.WATER,
                specific_gravity=1.0,
                molecular_weight=18.015,
            ),
            control=bores.RateControl(target_rate=400.0, bhp_limit=5000.0),
        )
    )

# Interior producers
interior_positions = [(10, 10), (10, 20), (20, 10), (20, 20), (15, 15)]
producers = []
for i, (x, y) in enumerate(interior_positions):
    producers.append(
        bores.production_well(
            well_name=f"PROD-{i + 1}",
            perforating_intervals=[((x, y, 0), (x, y, 4))],
            radius=0.25,
            produced_fluids=produced_fluids,
            control=prod_control,
        )
    )

wells = bores.wells_(producers=producers, injectors=injectors)
```

---

## Using duplicate() for Patterns

The `duplicate()` method makes it easy to create well arrays from a template:

```python
import bores

# Define a template producer
template = bores.production_well(
    well_name="template",
    perforating_intervals=[((0, 0, 0), (0, 0, 4))],
    radius=0.25,
    produced_fluids=produced_fluids,
    control=prod_control,
    skin_factor=2.0,
)

# Create a grid of producers
producers = []
for ix in range(5, 26, 10):
    for iy in range(5, 26, 10):
        producers.append(
            template.duplicate(
                name=f"PROD-{ix}-{iy}",
                perforating_intervals=[((ix, iy, 0), (ix, iy, 4))],
            )
        )
```

All duplicated wells share the same control, skin factor, radius, and fluid properties as the template. Only the name and location differ.

---

## Choosing a Pattern

| Pattern | Best For | Typical Recovery (waterflood) |
| --- | --- | --- |
| Five-spot | Homogeneous, isotropic reservoirs | 30-45% |
| Line drive | Anisotropic reservoirs, elongated structures | 25-40% |
| Peripheral flood | Edge-water drive, maintaining pressure | 30-50% |
| Single well (depletion) | Primary recovery, testing | 10-25% |

The pattern choice depends on the reservoir geometry, permeability anisotropy, fluid properties (viscosity ratio), and economic constraints (number of wells). For initial studies, the five-spot is a robust default. For history matching or field-specific work, use the actual well locations from the field data.

!!! tip "Visualize Well Locations"

    After building your wells, visualize their locations on the grid using `bores.plotly3d.DataVisualizer` to verify they are in the correct positions. Well placement errors (off-by-one in grid indices, wells outside the grid) are among the most common setup mistakes and are instantly visible in a 3D plot.
