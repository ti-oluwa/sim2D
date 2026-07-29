---
title: Your First Reservoir Simulation - BORES Tutorial
description: Step-by-step tutorial for building a complete 3D depletion simulation with BORES. Learn grid construction, well placement, PVT properties, and result visualization using Python.
tags:
  - tutorial
  - beginner
  - reservoir-simulation
  - python
keywords: reservoir simulation tutorial, primary depletion, BORES first simulation, black-oil modeling, Python petroleum engineering, production well, solution gas drive
author: ti-oluwa
---

# Your First Simulation

Build a complete depletion simulation from scratch and learn every step of the BORES workflow.

---

## Overview

In this tutorial, you will build a small 3D reservoir model with a single production well, run a primary depletion simulation, and visualize the results. Unlike the [Quickstart](../getting-started/quickstart.md), which focused on getting a simulation running as fast as possible, this tutorial explains the *why* behind every parameter choice. By the end, you will understand how pressure, saturation, and production evolve during depletion drive and be ready to tackle more complex scenarios.

Primary depletion is the simplest recovery mechanism: you produce oil from the reservoir without injecting any fluid to maintain pressure. As oil is withdrawn, reservoir pressure declines, dissolved gas comes out of solution (if pressure drops below the bubble point), and eventually the production rate falls because there is not enough pressure to push fluid to the wellbore. Understanding this baseline behavior is essential before studying enhanced recovery methods like waterflooding or gas injection.

We will work with a 10x10x3 homogeneous reservoir - small enough to run in seconds but large enough to observe realistic pressure and saturation trends across the grid.

---

## Physical Setup

Our model represents a small sandstone reservoir with the following characteristics:

- **Grid**: 10 cells in X, 10 cells in Y, 3 layers in Z (300 total cells)
- **Cell size**: 100 ft x 100 ft horizontally, 20 ft thick per layer
- **Porosity**: 20% (uniform)
- **Permeability**: 100 mD isotropic
- **Initial pressure**: 3,000 psi (above bubble point)
- **Bubble point**: 2,500 psi
- **Temperature**: 180 F
- **Oil viscosity**: 1.5 cP (light to medium oil)
- **Initial saturations**: 75% oil, 25% connate water, 0% gas

The single production well sits at grid location (5, 5) - roughly the center of the reservoir - and is perforated across all three layers. It operates under adaptive BHP-rate control, targeting 200 STB/day of oil production with a minimum BHP of 500 psi.

!!! info "Why Start Above Bubble Point?"

    Starting with the reservoir pressure above the bubble point means we begin in the undersaturated regime. All gas is dissolved in the oil phase. As we produce and pressure drops, we will eventually cross the bubble point, free gas will appear, and the system transitions to two-phase (oil + gas) flow. This transition is one of the most important phenomena in reservoir engineering, and observing it in simulation helps build intuition for real-world behavior.

---

## Step 1 - Import and Set Precision

```python
import bores
import logging
import numpy as np

# Set log level
logging.basicConfig(level=logging.INFO)

# Use 32-bit precision (default, faster computation)
bores.use_32bit_precision()
```

BORES defaults to 32-bit floating point for performance. On modern hardware, 32-bit SIMD operations are roughly twice as fast as 64-bit. For this tutorial, 32-bit precision is more than adequate. If you were comparing results against analytical solutions or running very long simulations where small errors accumulate, you would switch to `bores.use_64bit_precision()`.

---

## Step 2 - Define Grid and Properties

```python
# Grid dimensions
grid_shape = (10, 10, 3)
cell_dimension = (100.0, 100.0)  # (dx, dy) in feet

# Build uniform property grids
thickness = bores.build_uniform_grid(grid_shape, value=20.0)        # ft per layer
pressure = bores.build_uniform_grid(grid_shape, value=3000.0)       # psi
porosity = bores.build_uniform_grid(grid_shape, value=0.20)         # fraction
temperature = bores.build_uniform_grid(grid_shape, value=180.0)     # deg F
oil_viscosity = bores.build_uniform_grid(grid_shape, value=1.5)     # cP
bubble_point = bores.build_uniform_grid(grid_shape, value=2500.0)   # psi
oil_sg = bores.build_uniform_grid(grid_shape, value=0.85)           # ~35 deg API

# Residual and irreducible saturations
sorw = bores.build_uniform_grid(grid_shape, value=0.20)  # Residual oil (waterflood)
sorg = bores.build_uniform_grid(grid_shape, value=0.15)  # Residual oil (gas flood)
sgr  = bores.build_uniform_grid(grid_shape, value=0.05)  # Residual gas
Swir = bores.build_uniform_grid(grid_shape, value=0.20)  # Irreducible water
swc  = bores.build_uniform_grid(grid_shape, value=0.20)  # Connate water

# Build initial saturations from fluid contacts
# Depth grid: top of reservoir at 5000 ft, 20 ft per layer
depth = bores.depth_grid(thickness_grid=thickness, datum=5000.0)

Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4900.0,       # GOC above reservoir (no initial gas cap)
    oil_water_contact=5055.0,     # OWC near reservoir bottom
    connate_water_saturation_grid=swc,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    residual_gas_saturation_grid=sgr,
    porosity_grid=porosity,
)
```

Each call to `bores.build_uniform_grid()` creates a NumPy array of the specified shape filled with a single value. In a real study, you would load heterogeneous data from geological models, well logs, or geostatistical realizations. For this tutorial, uniform properties let us focus on understanding the simulation workflow without the complexity of spatial heterogeneity.

The `bores.build_saturation_grids()` function computes physically consistent initial saturations from fluid contact depths. You provide the gas-oil contact (GOC) and oil-water contact (OWC) depths, along with residual saturation grids, and BORES assigns each cell to the correct fluid zone. Cells above the GOC get gas and connate water, cells between GOC and OWC get oil and connate water, and cells below the OWC get water. This ensures that $S_o + S_w + S_g = 1.0$ everywhere, which is fundamental to the black-oil formulation. Here we place the GOC above the reservoir (no gas cap) and the OWC near the bottom, giving mostly oil with connate water.

The oil specific gravity of 0.85 corresponds to roughly 35 degrees API, which is a light to medium crude oil. This value drives the PVT correlations that BORES uses to estimate properties like solution gas-oil ratio, formation volume factor, and oil density.

---

## Step 3 - Set Up Permeability

```python
# Isotropic permeability: 100 mD in all directions
perm_grid = bores.build_uniform_grid(grid_shape, value=100.0)
permeability = bores.RockPermeability(x=perm_grid, y=perm_grid, z=perm_grid)
```

Permeability controls how easily fluid flows through the rock. We use `bores.RockPermeability` to define permeability in each direction. For isotropic rock (same permeability everywhere), you can pass the same grid for x, y, and z. If you only provide the x-direction grid, BORES assumes isotropy and copies it to y and z automatically.

A value of 100 mD is typical of a good quality sandstone reservoir. In the [next tutorial](02-building-models.md), you will learn how to create anisotropic permeability where vertical permeability differs from horizontal, which is very common in real reservoirs due to layered deposition.

!!! tip "Shortcut for Isotropic Permeability"

    If your permeability is isotropic, you can pass just the x-direction grid and BORES will use it for all directions:

    ```python
    permeability = bores.RockPermeability(x=perm_grid)
    # y and z are automatically set equal to x
    ```

---

## Step 4 - Build the Reservoir Model

```python
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness,
    pressure_grid=pressure,
    rock_compressibility=3e-6,            # psi⁻¹
    absolute_permeability=permeability,
    porosity_grid=porosity,
    temperature_grid=temperature,
    water_saturation_grid=Sw,
    gas_saturation_grid=Sg,
    oil_saturation_grid=So,
    oil_viscosity_grid=oil_viscosity,
    oil_specific_gravity_grid=oil_sg,
    oil_bubble_point_pressure_grid=bubble_point,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    residual_gas_saturation_grid=sgr,
    irreducible_water_saturation_grid=Swir,
    connate_water_saturation_grid=swc,
    datum_depth=5000,
)
```

The `bores.reservoir_model()` factory does a lot of work behind the scenes. It validates all your inputs, then computes every fluid property you did not provide explicitly: gas viscosity, gas density, gas compressibility factor, gas formation volume factor, water density, water compressibility, oil formation volume factor, oil compressibility, solution gas-oil ratio, and many others. All of these are estimated from pressure, temperature, and fluid gravity using industry-standard PVT correlations (Standing, Vasquez-Beggs, Lee-Gonzalez, and others).

The rock compressibility of $3 \times 10^{-6}$ psi$^{-1}$ is typical for consolidated sandstone. This parameter controls how much the pore volume changes with pressure. While small compared to fluid compressibilities, rock compressibility contributes to the total system compressibility that governs pressure diffusion.

The resulting `model` object is immutable. You cannot modify its properties after construction, which ensures your initial conditions remain untouched throughout the simulation.

!!! warning "Oil Specific Gravity is Required"

    You must provide `oil_specific_gravity_grid` (or use PVT tables) so that BORES can compute API gravity and all downstream oil PVT properties. Without it, the factory will raise a `ValidationError`.

---

## Step 5 - Define the Production Well

```python
producer = bores.production_well(
    well_name="PROD-1",
    perforating_intervals=[((5, 5, 0), (5, 5, 2))],
    radius=0.25,  # ft
    control=bores.ProducerRateControl(
        primary_phase=bores.FluidPhase.OIL,
        primary_control=bores.AdaptiveRateControl(
            target_rate=-200.0,    # produce 200 STB/day of oil
            target_phase="oil",
            bhp_limit=500.0,       # minimum BHP constraint
            clamp=bores.ProductionClamp(),
        ),
        secondary_clamp=bores.ProductionClamp(),
    ),
    produced_fluids=[
        bores.ProducedFluid(
            name="Oil", phase=bores.FluidPhase.OIL,
            specific_gravity=0.85, molecular_weight=200.0,
        ),
        bores.ProducedFluid(
            name="Water", phase=bores.FluidPhase.WATER,
            specific_gravity=1.0, molecular_weight=18.015,
        ),
        bores.ProducedFluid(
            name="Gas", phase=bores.FluidPhase.GAS,
            specific_gravity=0.65, molecular_weight=16.04,
        ),
    ],
)

wells = bores.wells_(producers=[producer])
```

The well is placed at grid location (5, 5) and perforated from layer 0 to layer 2, connecting it to all three reservoir layers. The perforating interval is specified as a pair of (x, y, z) coordinates: the start and end of the perforation.

We use `ProducerRateControl` because it is the standard approach in reservoir simulation for production wells. You fix the oil rate (the primary phase), and the simulator computes the BHP required to deliver that rate. Water and gas then flow at whatever their natural Darcy rates are at the resulting BHP. This is far more realistic than applying the same rate to all phases.

Inside the primary control, `AdaptiveRateControl` handles the oil rate target of -200 STB/day (negative = production). When reservoir pressure is high, the well achieves this rate. As pressure declines, eventually the drawdown needed to maintain -200 STB/day would push BHP below 500 psi. At that point, the control automatically switches to constant-BHP mode and lets the oil rate decline naturally. The `ProductionClamp` on the secondary phases prevents backflow of water or gas into the reservoir.

The `produced_fluids` list tells the simulator which phases can flow into this well. Even though we start with no free gas, we include a gas `ProducedFluid` because gas will appear once pressure drops below the bubble point.

The `wells_()` factory groups wells into a `Wells` container. Since this is a depletion study with no injection, we only have producers.

---

## Step 6 - Configure Rock-Fluid Properties

```python
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyRelPermModel(
        irreducible_water_saturation=0.25,
        residual_oil_saturation_water=0.30,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.05,
        water_exponent=2.0,
        oil_exponent=2.0,
        gas_exponent=2.0,
    ),
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(
        irreducible_water_saturation=0.25,
        residual_oil_saturation_water=0.30,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.05,
    ),
)
```

The `BrooksCoreyRelPermModel` defines how relative permeability varies with saturation using the Corey power-law model. The exponents control curve shape: an exponent of 2.0 produces moderately curved functions that are a reasonable starting point for sandstone. Higher exponents (3-4) would make the curves steeper, meaning phases need higher saturation before they can flow significantly.

The `BrooksCoreyCapillaryPressureModel` uses default parameters suitable for a first approximation. Capillary pressure represents the pressure difference between phases at the pore scale due to surface tension. For this tutorial, the defaults are adequate. In the [Building Reservoir Models](02-building-models.md) tutorial, you will see how to customize these parameters.

Both models are bundled into a `RockFluidTables` object, which the simulator queries at every time step.

---

## Step 7 - Set Up the Timer and Configuration

```python
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=1),
        maximum_step_size=bores.Time(days=15),
        minimum_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(days=730),   # 2 years
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)
```

The `bores.Time()` helper converts human-readable durations to seconds, which is the internal time unit used by BORES. You can combine units freely: `bores.Time(years=1, days=30)` gives you one year plus thirty days.

The `Timer` manages adaptive time stepping. It starts with a 1-day step, can grow to 15 days when conditions are stable, and can shrink to 1 hour if the solver needs smaller steps for convergence. The timer automatically adjusts based on CFL conditions, pressure changes, and saturation changes. For a depletion study, pressure changes are usually gradual, so the timer will ramp up to larger steps fairly quickly.

We simulate for 730 days (2 years) to observe the full depletion cycle: initial high-rate production, pressure decline through the bubble point, gas liberation, and eventual rate decline as pressure support diminishes.

The `scheme="impes"` setting selects the IMPES (Implicit Pressure, Explicit Saturation) evolution scheme. This is the standard choice for most black-oil simulations, offering good stability from implicit pressure solving with the efficiency of explicit saturation transport.

---

## Step 8 - Run the Simulation

```python
# Run and collect all states
states = list(bores.run(model, config))

# Print summary
final = states[-1]
print(f"Completed {final.step} steps in {final.time_in_days:.1f} days")
print(f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi")
print(f"Final avg oil saturation: {final.model.fluid_properties.oil_saturation_grid.mean():.4f}")
```

The `bores.run()` function returns a Python generator. Each iteration advances the simulation by one time step and yields a `ModelState` snapshot. Wrapping it with `list()` collects all snapshots into memory. For this small model, that is perfectly fine. For larger models, you would use `StateStream` to persist states to disk as they are generated.

Each `ModelState` contains the complete reservoir state at that time step, including the updated model, well states, injection and production rate grids, relative permeabilities, and capillary pressures.

!!! note "First-Run Compilation"

    The first time you run a BORES simulation, Numba compiles several internal functions to machine code. This one-time compilation typically takes 10-30 seconds. Subsequent runs reuse the cached compiled code and start immediately.

---

## Step 9 - Visualize Results

### Pressure Decline

```python
import numpy as np

# Extract time series data
time_days = np.array([s.time_in_days for s in states])
avg_pressure = np.array([
    s.model.fluid_properties.pressure_grid.mean() for s in states
])

# Plot pressure vs time
pressure_series = np.column_stack([time_days, avg_pressure])
fig = bores.make_series_plot(
    data=pressure_series,
    title="Average Reservoir Pressure During Depletion",
    x_label="Time (days)",
    y_label="Pressure (psi)",
)
fig.show()
```

You should see pressure declining from 3,000 psi toward lower values over the 2-year period. The rate of decline depends on the production rate relative to the total compressible volume of the reservoir. When pressure crosses the bubble point (2,500 psi), the decline may slow because gas liberation provides additional drive energy.

### Saturation Evolution

```python
avg_So = np.array([
    s.model.fluid_properties.oil_saturation_grid.mean() for s in states
])
avg_Sg = np.array([
    s.model.fluid_properties.gas_saturation_grid.mean() for s in states
])

fig = bores.make_series_plot(
    data={
        "Oil Saturation": np.column_stack([time_days, avg_So]),
        "Gas Saturation": np.column_stack([time_days, avg_Sg]),
    },
    title="Average Saturations During Depletion",
    x_label="Time (days)",
    y_label="Saturation (fraction)",
)
fig.show()
```

As pressure drops below the bubble point, gas comes out of solution and gas saturation increases. Oil saturation decreases both because oil is being produced and because the oil formation volume factor changes (oil shrinks as gas leaves it). This is the classic solution gas drive mechanism.

### 3D Pressure Distribution

```python
viz = bores.plotly3d.DataVisualizer()
fig = viz.make_plot(
    source=states[-1],
    property="pressure",
    plot_type="volume",
    title="Final Pressure Distribution",
)
fig.show()
```

The 3D visualization shows the pressure distribution at the final time step. You should see lower pressure near the production well and higher pressure at the reservoir boundaries, reflecting the pressure drawdown cone around the wellbore.

---

## Key Takeaways

1. **Depletion drive** relies solely on the natural energy stored in compressed rock and fluids. Without pressure maintenance (injection), production rates eventually decline.

2. **The bubble point** is a critical threshold. Above it, the system behaves as single-phase (undersaturated) oil. Below it, gas liberates from solution, creating a two-phase system with very different flow characteristics.

3. **`ProducerRateControl`** is the standard way to control production wells. You fix the oil (or gas) rate and let the other phases flow at whatever BHP results from the primary phase. `AdaptiveRateControl` within it handles the automatic switch from rate to BHP mode as reservoir pressure declines.

4. **The `reservoir_model()` factory** computes most PVT properties automatically from correlations. You only need to provide the properties you know; the factory estimates the rest.

5. **BORES uses field units**: psi for pressure, ft for length, mD for permeability, cP for viscosity, STB/day for oil rates, and degrees Fahrenheit for temperature.

---

## Next Steps

In the [next tutorial](02-building-models.md), you will learn how to build more realistic reservoir models with heterogeneous properties, anisotropic permeability, and structural dip. These features are essential for capturing the geological complexity that drives real reservoir behavior.
