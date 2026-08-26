---
title: Well Basics - BORES User Guide
description: Learn how to define production and injection wells in BORES. Covers well types, perforating intervals, wellbore radius, skin factors, and control strategies for reservoir simulation.
tags:
  - guide
  - intermediate
  - reservoir-simulation
  - wells
keywords: production well, injection well, well control, wellbore radius, skin factor, perforating intervals, BORES wells, rate control, BHP control
author: ti-oluwa
---

# Well Basics

## Overview

Wells are the primary interface between the reservoir and the surface. They are the points where fluids enter or leave the simulation domain, and they drive the pressure and saturation changes that form the core of any reservoir simulation study. In BORES, wells are defined by their physical properties (location, radius, skin factor), the fluids they handle, and the control strategy that governs how they operate.

BORES provides two well types: `ProductionWell` and `InjectionWell`. Production wells remove fluids from the reservoir (oil, water, and gas flow from the formation into the wellbore). Injection wells add fluids to the reservoir (water or gas is pumped from the surface into the formation). The direction of flow is determined by the pressure difference between the wellbore and the surrounding reservoir rock: fluid flows from high pressure to low pressure.

Rather than constructing well objects directly, BORES provides factory functions (`bores.production_well()` and `bores.injection_well()`) that handle validation and default parameter computation. These factories wrap the `ProductionWell` and `InjectionWell` classes and ensure that all required fields are properly initialized.

---

## Production Wells

A production well removes fluids from the reservoir. You define it by specifying a name, the grid cells it perforates, the wellbore radius, the produced fluids, and the control strategy.

```python
import bores

producer = bores.production_well(
    well_name="PROD-1",
    perforating_intervals=[((10, 10, 0), (10, 10, 4))],
    radius=0.25,  # ft
    produced_fluids=[
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.85,
            molecular_weight=200.0,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.02,
            molecular_weight=18.015,
        ),
        bores.ProducedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04,
        ),
    ],
    control=bores.ProducerRateControl(
        primary_phase=bores.FluidPhase.OIL,
        primary_control=bores.AdaptiveRateControl(
            target_rate=-500.0,
            target_phase="oil",
            bhp_limit=1000.0,
        ),
        secondary_clamp=bores.ProductionClamp(),
    ),
    skin_factor=2.0,
)
```

The `produced_fluids` parameter takes a list of `ProducedFluid` objects, one for each phase that can flow into the wellbore. You must provide the fluid name, phase, specific gravity (relative to water for liquids, relative to air for gas), and molecular weight. The specific gravity and molecular weight are used by PVT correlations to compute formation volume factors, viscosities, and densities during the simulation.

The `control` parameter specifies how the well operates. See [Well Controls](controls.md) for details on the available control strategies.

### Produced Fluids

A `ProducedFluid` describes the properties of a fluid being produced by a well. It inherits from `WellFluid` and requires four parameters:

| Parameter | Units | Description |
| --- | --- | --- |
| `name` | string | Human-readable name (e.g., "Oil", "Water", "Gas") |
| `phase` | `FluidPhase` | Phase designation: `OIL`, `WATER`, or `GAS` |
| `specific_gravity` | dimensionless | Gravity relative to reference (water for liquids, air for gas) |
| `molecular_weight` | g/mol | Molecular weight of the fluid |

In most simulations, you define all three produced fluids (oil, water, gas) for each production well, even if you expect only oil to flow initially. As the simulation progresses, water and gas may break through, and the well needs to know the properties of all phases that could be produced.

---

## Injection Wells

An injection well adds fluids to the reservoir. You define it similarly to a production well, but instead of produced fluids, you specify a single `InjectedFluid` that describes the fluid being injected.

```python
import bores

injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((0, 0, 0), (0, 0, 4))],
    radius=0.25,  # ft
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.0,
        molecular_weight=18.015,
    ),
    control=bores.RateControl(
        target_rate=800.0,  # Positive = injection
        bhp_limit=5000.0,  # Max injection pressure
    ),
)
```

Notice the sign convention: positive rates mean injection (fluid entering the reservoir), and negative rates mean production (fluid leaving the reservoir). This convention is consistent throughout the BORES library.

### Injected Fluids

An `InjectedFluid` describes the properties of the fluid being injected. It extends `WellFluid` with additional parameters for miscible flooding and property overrides:

| Parameter | Default | Description |
| --- | --- | --- |
| `name` | required | Human-readable name (e.g., "Water", "CO2") |
| `phase` | required | Must be `WATER` or `GAS` (oil injection is not supported) |
| `specific_gravity` | required | Gravity relative to reference |
| `molecular_weight` | required | Molecular weight (g/mol) |
| `salinity` | `None` | Water salinity in ppm NaCl (for water injection) |
| `is_miscible` | `False` | Whether this fluid is miscible with oil |
| `todd_longstaff_omega` | `0.67` | Todd-Longstaff mixing parameter (0 to 1) |
| `minimum_miscibility_pressure` | `None` | MMP in psi (required if miscible) |
| `density` | `None` | Override density in lbm/ft³ (bypasses correlations) |
| `viscosity` | `None` | Override viscosity in cP (bypasses correlations) |

See [Well Fluids](fluids.md) for detailed guidance on configuring injected fluids, especially for CO2 and miscible gas injection.

---

## Perforating Intervals

Perforating intervals define which grid cells the well penetrates. Each interval is a tuple of two coordinates: the start and end of the perforation in grid index space (i, j, k). A well can have multiple intervals, and each interval can span multiple cells.

```python
# Single interval: well penetrates cells (10, 10, 0) through (10, 10, 4)
# This is a vertical well completing through all 5 layers at column (10, 10)
perforating_intervals = [((10, 10, 0), (10, 10, 4))]

# Multiple intervals: well is perforated in two separate zones
perforating_intervals = [
    ((10, 10, 0), (10, 10, 1)),  # Upper zone: layers 0-1
    ((10, 10, 3), (10, 10, 4)),  # Lower zone: layers 3-4
]
```

The coordinates use zero-based indexing, matching the NumPy array convention. The start coordinate must be within the grid, and the end coordinate must also be within the grid. BORES validates this when you build the simulation configuration.

For a vertical well in a grid with shape `(nx, ny, nz)`, the x and y indices select the column position and the z range selects which layers are perforated. For horizontal wells, the x or y range spans multiple cells while z remains constant.

### Well Orientation

BORES automatically detects the well orientation from the perforating intervals by finding the dominant axis of the perforation trajectory. A well that spans more cells in the z-direction is classified as vertical (Z-oriented), one that spans more in x is X-oriented, and so on. The orientation affects how the Peaceman well index is calculated.

You can also set the orientation explicitly:

```python
producer = bores.production_well(
    well_name="HORIZ-1",
    perforating_intervals=[((5, 10, 3), (15, 10, 3))],
    radius=0.25,
    orientation=bores.Orientation.X,  # Horizontal well in x-direction
    produced_fluids=[...],
    control=control,
)
```

---

## Well Index and the Peaceman Equation

The well index $WI$ (also called the productivity index or connection transmissibility factor) controls how much fluid flows between the wellbore and the grid cell. It is computed using the Peaceman equation, which accounts for the wellbore radius, the effective drainage radius, the cell permeability, and the skin factor.

$$WI = \frac{k_{eff} \cdot h}{\ln(r_e / r_w) + s}$$

where $k_{eff}$ is the effective permeability in the plane perpendicular to the well, $h$ is the cell thickness in the well direction, $r_e$ is the effective drainage radius (computed from the cell dimensions and permeability anisotropy using the Peaceman formula), $r_w$ is the wellbore radius, and $s$ is the skin factor.

BORES computes the well index automatically from the grid properties and well parameters. You do not need to calculate it manually. The well index determines how the flow rate relates to the pressure drawdown:

$$q = WI \cdot \frac{k_{r\alpha}}{\mu_\alpha B_\alpha} \cdot (P_{res} - P_{wf})$$

where $P_{res}$ is the reservoir pressure, $P_{wf}$ is the flowing bottom-hole pressure, and the middle term is the phase mobility divided by the formation volume factor.

---

## Skin Factor

The skin factor $s$ is a dimensionless number that accounts for additional pressure drop (or reduction) near the wellbore that is not captured by the grid-scale permeability. Positive skin indicates formation damage (reduced near-wellbore permeability due to drilling mud invasion, completion damage, or scale buildup), which reduces productivity. Negative skin indicates stimulation (fracturing or acidizing that improves near-wellbore flow), which increases productivity.

| Skin Factor | Interpretation |
| --- | --- |
| $s < 0$ | Stimulated well (fractured or acidized) |
| $s = 0$ | Undamaged well (ideal completion) |
| $s = 1$ to $5$ | Moderate damage |
| $s > 5$ | Severe damage |

```python
# Fractured well (stimulated)
fractured = bores.production_well(
    well_name="FRAC-1",
    perforating_intervals=[((10, 10, 0), (10, 10, 4))],
    radius=0.25,
    skin_factor=-2.0,  # Stimulated
    produced_fluids=[...],
    control=control,
)

# Damaged well
damaged = bores.production_well(
    well_name="DAMAGED-1",
    perforating_intervals=[((10, 10, 0), (10, 10, 4))],
    radius=0.25,
    skin_factor=5.0,  # Damaged
    produced_fluids=[...],
    control=control,
)
```

In history matching, the skin factor is often one of the first parameters adjusted because it directly affects well productivity without changing the regional rock properties. If a well produces less than expected from the grid permeability alone, a positive skin factor can account for the discrepancy.

---

## Combining Wells

After creating individual wells, combine them into a `Wells` container using `bores.wells_()`:

```python
import bores

wells = bores.wells_(
    producers=[producer],
    injectors=[injector],
)
```

The `Wells` object is then passed to the `Config` for simulation:

```python
config = bores.Config(
    timer=timer,
    wells=wells,
    rock_fluid_tables=rock_fluid_tables,
    scheme="impes",
)
```

You can include any number of producers and injectors. Wells are identified by their `name` attribute, which should be unique within each category (producers and injectors).

---

## Shutting In and Opening Wells

During a simulation, you can shut in or open wells programmatically. A shut-in well has zero flow rate and no pressure interaction with the reservoir.

```python
# Shut in the well
producer.shut_in()
assert producer.is_shut_in  # True
assert not producer.is_open  # True

# Re-open the well
producer.open()
assert producer.is_open  # True
```

For time-dependent well operations (shutting in at a specific time, changing controls mid-simulation), use [Well Schedules](schedules.md).

---

## Duplicating Wells

You can create copies of a well with modified properties using the `duplicate()` method. This is useful for creating well patterns where multiple wells share the same configuration but differ in location or name.

```python
# Create a second producer at a different location
prod_2 = producer.duplicate(
    name="PROD-2",
    perforating_intervals=[((15, 15, 0), (15, 15, 4))],
)
```

The `duplicate()` method uses `attrs.evolve()` internally, so it creates a shallow copy with the specified fields overridden. Any fields not listed in the keyword arguments retain their values from the original well.

!!! tip "Well Naming Convention"

    Use descriptive, unique names for your wells. Common conventions include "PROD-1", "INJ-1" for basic studies, or field-standard naming like "A-01", "B-02" for field models. The well name is used in logging, schedules, and post-processing, so clear naming saves debugging time.
