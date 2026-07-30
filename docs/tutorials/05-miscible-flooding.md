# Miscible Gas Flooding

Model miscible displacement using the Todd-Longstaff method, configure CO2 injection with custom fluid properties, and analyze solvent mixing behavior.

---

## Overview

In the [previous tutorial](04-gas-injection.md), you saw that immiscible gas injection suffers from gravity override and poor sweep efficiency. Miscible gas injection overcomes many of these limitations by creating conditions where the injected gas dissolves into and mixes with the reservoir oil at the molecular level. When miscibility is achieved, the interfacial tension between gas and oil vanishes, residual oil trapping is eliminated, and the displacement efficiency approaches 100% in the swept zone.

This tutorial introduces the Todd-Longstaff miscible displacement model, which is the industry-standard approach for simulating miscible flooding in black-oil simulators. You will configure CO2 injection with custom density and viscosity properties, run a miscible flood, visualize solvent concentration evolution, and study the sensitivity of the mixing parameter (omega).

Miscible flooding is one of the most effective enhanced oil recovery (EOR) techniques, particularly for CO2 injection in light to medium oil reservoirs. Understanding how to set up and interpret miscible simulations is essential for evaluating EOR potential.

---

## The Todd-Longstaff Model

The Todd-Longstaff model is a practical engineering approach to miscible displacement that avoids the computational cost of full compositional simulation. It works by modifying the effective viscosity and density of the oil phase based on how much solvent has mixed into it. The model uses a single mixing parameter, $\omega$ (omega), that controls the degree of mixing between the injected solvent and the reservoir oil.

The effective viscosity is computed as:

$$\mu_{\text{eff}} = \mu_{\text{mix}}^{\omega} \cdot \mu_{\text{seg}}^{1-\omega}$$

where $\mu_{\text{mix}}$ is the fully-mixed viscosity (computed from oil and solvent viscosities), $\mu_{\text{seg}}$ is the segregated (unmixed) viscosity, and $\omega$ ranges from 0 to 1. When $\omega = 1$, the fluids are perfectly mixed within each grid cell. When $\omega = 0$, the fluids are completely segregated and behave as two distinct phases despite being in the same grid cell.

The density follows the same form:

$$\rho_{\text{eff}} = \rho_{\text{mix}}^{\omega} \cdot \rho_{\text{seg}}^{1-\omega}$$

In practice, $\omega$ typically ranges from 0.5 to 0.8 for most reservoir applications. A value of 2/3 (0.667) is the most common default. The choice of $\omega$ depends on grid resolution: coarse grids need lower $\omega$ values to account for unresolved sub-grid mixing, while fine grids can use higher values because more of the physical mixing is resolved by the grid itself.

The model also accounts for pressure-dependent miscibility using the minimum miscibility pressure (MMP). Below the MMP, the displacement is immiscible. Above the MMP, full miscibility is achieved. BORES uses a smooth hyperbolic tangent transition between these regimes, controlled by the `miscibility_transition_width` parameter.

---

## Physical Setup

We use the same 15x15x3 grid for consistency, but now inject CO2 with miscible properties:

- **Grid**: 15x15x3 (675 cells)
- **Initial pressure**: 3,000 psi (above MMP)
- **CO2 injector**: Corner (0, 0)
- **MMP**: 1,800 psi
- **Todd-Longstaff omega**: 0.67
- **CO2 density**: 35.0 lbm/ft³ (at reservoir conditions)
- **CO2 viscosity**: 0.05 cP (at reservoir conditions)

!!! info "Minimum Miscibility Pressure (MMP)"

    The MMP is the lowest pressure at which the injected gas achieves miscibility with the reservoir oil. Below this pressure, a gas-oil interface exists and trapping occurs. Above it, the fluids become fully miscible and can mix in all proportions. The MMP depends on oil composition, gas composition, and temperature. For CO2 injection:

    - **Light oils (35-45 API)**: MMP = 1,200 - 2,500 psi
    - **Medium oils (25-35 API)**: MMP = 2,000 - 3,500 psi
    - **Heavy oils (< 25 API)**: MMP > 3,500 psi (often impractical)

---

## Step 1 - Build the Reservoir Model

```python
import bores
import numpy as np

bores.use_32bit_precision()

grid_shape = (15, 15, 3)
cell_dimension = (80.0, 80.0)

thickness = bores.build_uniform_grid(grid_shape, value=20.0)
pressure = bores.build_uniform_grid(grid_shape, value=3000.0)
porosity = bores.build_uniform_grid(grid_shape, value=0.20)
temperature = bores.build_uniform_grid(grid_shape, value=180.0)
oil_viscosity = bores.build_uniform_grid(grid_shape, value=2.0)
bubble_point = bores.build_uniform_grid(grid_shape, value=2500.0)
oil_sg = bores.build_uniform_grid(grid_shape, value=0.87)

sorw = bores.build_uniform_grid(grid_shape, value=0.22)
sorg = bores.build_uniform_grid(grid_shape, value=0.15)
sgr  = bores.build_uniform_grid(grid_shape, value=0.05)
Swir = bores.build_uniform_grid(grid_shape, value=0.22)
swc  = bores.build_uniform_grid(grid_shape, value=0.22)

# Build initial saturations from fluid contacts
depth = bores.build_depth_grid(thickness, datum=5000.0)

Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4900.0,       # GOC above reservoir (no initial gas cap)
    oil_water_contact=5100.0,     # OWC near reservoir bottom
    connate_water_saturation_grid=swc,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    residual_gas_saturation_grid=sgr,
    porosity_grid=porosity,
)

perm_grid = bores.build_uniform_grid(grid_shape, value=150.0)
permeability = bores.Permeability(x=perm_grid)

model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness,
    pressure_grid=pressure,
    rock_compressibility=3e-6,
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
    reservoir_gas="CO2",
    datum_depth=5000,
)
```

The only difference in the model construction is `reservoir_gas="CO2"`, which tells the PVT correlation engine that the gas phase properties should be computed for CO2 rather than methane. This affects gas viscosity, density, compressibility factor, and other derived properties.

---

## Step 2 - Configure the CO2 Injection Well

```python
co2_injector = bores.injection_well(
    well_name="CO2-INJ-1",
    perforating_intervals=[((0, 0, 0), (0, 0, 2))],
    radius=0.25,
    control=bores.RateControl(
        target_rate=5_000_000.0, 
        bhp_limit=5000,
    ),
    injected_fluid=bores.InjectedFluid(
        name="CO2",
        phase=bores.FluidPhase.GAS,
        specific_gravity=1.52,
        molecular_weight=44.01,
        is_miscible=True,
        minimum_miscibility_pressure=1800.0,
        todd_longstaff_omega=0.67,
        density=35.0,       # lbm/ft³ at reservoir conditions
        viscosity=0.05,     # cP at reservoir conditions
    ),
)
```

This is the most important part of the tutorial. The `InjectedFluid` has several new parameters compared to the immiscible gas injection:

**`is_miscible=True`** activates the Todd-Longstaff miscible displacement model. Without this flag, the gas and oil phases remain separate regardless of pressure.

**`minimum_miscibility_pressure=1800.0`** sets the MMP in psi. Since our reservoir pressure is 3,000 psi, which is well above the MMP, the displacement will be fully miscible from the start. If reservoir pressure drops below the MMP during the simulation, the model automatically transitions to immiscible behavior.

**`todd_longstaff_omega=0.67`** is the mixing parameter. The value of 2/3 is the most commonly used default in the industry. We will explore the sensitivity to this parameter later in the tutorial.

**`density=35.0` and `viscosity=0.05`** override the correlation-based property calculations. This is important for CO2 because standard gas correlations (designed for hydrocarbon gases) significantly underestimate CO2 density and may give inaccurate viscosity values. At reservoir conditions (3,000 psi, 180 F), supercritical CO2 has a density around 35 lbm/ft³ and a viscosity around 0.05 cP. These values should ideally come from laboratory measurements or an equation of state model.

!!! danger "CO2 Properties Require Overrides"

    Standard gas PVT correlations in BORES (and most black-oil simulators) are calibrated for hydrocarbon gases and can give errors of 25% or more for CO2 density. Always provide explicit `density` and `viscosity` values for CO2 injection based on lab data, NIST tables, or an equation of state calculation. The overrides are backward compatible - if you omit them, BORES falls back to correlations.

---

## Step 3 - Define the Producer and Configure

```python
producer = bores.production_well(
    well_name="PROD-1",
    perforating_intervals=[((14, 14, 0), (14, 14, 2))],
    radius=0.25,
    control=bores.ProducerRateControl(
        primary_phase=bores.FluidPhase.OIL,
        primary_control=bores.AdaptiveRateControl(
            target_rate=-400.0,
            target_phase="oil",
            bhp_limit=800.0,
        ),
        secondary_clamp=bores.ProductionClamp(),
    ),
    produced_fluids=[
        bores.ProducedFluid(
            name="Oil", phase=bores.FluidPhase.OIL,
            specific_gravity=0.87, molecular_weight=200.0,
        ),
        bores.ProducedFluid(
            name="Water", phase=bores.FluidPhase.WATER,
            specific_gravity=1.0, molecular_weight=18.015,
        ),
        bores.ProducedFluid(
            name="CO2", phase=bores.FluidPhase.GAS,
            specific_gravity=1.52, molecular_weight=44.01,
        ),
    ],
)

wells = bores.wells_(injectors=[co2_injector], producers=[producer])

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyRelPermModel(
        irreducible_water_saturation=0.25,
        residual_oil_saturation_water=0.30,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.05,
        water_exponent=2.5,
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

config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=0.5),
        maximum_step_size=bores.Time(days=5),
        minimum_step_size=bores.Time(hours=0.5),
        simulation_time=bores.Time(days=1095),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
    miscibility_model="todd_longstaff",
)
```

The critical configuration change is `miscibility_model="todd_longstaff"` in the `Config`. This tells the simulator to use the Todd-Longstaff miscible model during saturation updates. Without this setting, the simulator would treat all gas injection as immiscible regardless of the `InjectedFluid` settings.

The timer settings are the same as the immiscible gas injection tutorial because miscible fronts can be equally sharp.

---

## Step 4 - Run the Simulation

```python
states = list(bores.run(model, config))
final = states[-1]
print(f"Completed {final.step} steps in {final.time_in_days:.1f} days")
print(f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi")
print(f"Final avg oil saturation: {final.model.fluid_properties.oil_saturation_grid.mean():.4f}")
```

---

## Step 5 - Visualize Solvent Concentration

One of the unique outputs of a miscible simulation is the solvent concentration field. This tracks how much CO2 has mixed into the oil phase in each cell, ranging from 0 (pure oil) to 1 (pure solvent).

```python
viz = bores.plotly3d.DataVisualizer()

# Visualize solvent concentration at midpoint and end
for state in [states[len(states) // 2], states[-1]]:
    fig = viz.make_plot(
        source=state,
        property="solvent-concentration",
        plot_type="volume",
        title=f"Solvent Concentration at Day {state.time_in_days:.0f}",
    )
    fig.show()
```

The solvent concentration visualization shows the mixing zone between the pure CO2 bank and the undisplaced oil. In a miscible flood, this transition zone is smoother than the sharp front you see in immiscible displacement. The concentration field reveals how effectively the CO2 is contacting and mixing with the oil.

### Concentration Profile Over Time

```python
time_days = np.array([s.time_in_days for s in states])
avg_concentration = np.array([
    s.model.fluid_properties.solvent_concentration_grid.mean() for s in states
])

fig = bores.make_series_plot(
    data=np.column_stack([time_days, avg_concentration]),
    title="Average Solvent Concentration Over Time",
    x_label="Time (days)",
    y_label="Solvent Concentration (fraction)",
)
fig.show()
```

The average solvent concentration increases over time as more CO2 enters the reservoir and mixes with the oil. The rate of increase depends on the injection rate relative to the pore volume and the mixing efficiency (controlled by omega).

---

## Step 6 - Compare Miscible vs Immiscible Recovery

```python
avg_So = np.array([
    s.model.fluid_properties.oil_saturation_grid.mean() for s in states
])

initial_So = 0.75
recovery_factor = (initial_So - avg_So) / initial_So

fig = bores.make_series_plot(
    data=np.column_stack([time_days, recovery_factor]),
    title="Oil Recovery Factor (Miscible CO2 Flood)",
    x_label="Time (days)",
    y_label="Recovery Factor (fraction of OOIP)",
)
fig.show()
```

Compare this recovery curve with the immiscible gas flood from [Tutorial 4](04-gas-injection.md). The miscible flood should show significantly higher ultimate recovery because:

1. **Zero residual oil in swept zones** - Miscibility eliminates capillary trapping, so oil saturation can drop to zero in cells fully contacted by CO2 (instead of being limited to sorg = 0.15 in the immiscible case).

2. **Better sweep efficiency** - The mixing of CO2 with oil reduces the viscosity contrast, improving the mobility ratio and reducing viscous fingering.

3. **Density modification** - As CO2 dissolves into oil, it changes the oil density, which can reduce gravity segregation effects.

---

## Step 7 - Omega Sensitivity Analysis

The mixing parameter $\omega$ has a significant effect on miscible flood performance. Here is how you would set up a sensitivity study by running multiple simulations with different omega values:

```python
# This is a conceptual example showing how to run a sensitivity study
omega_values = [0.33, 0.50, 0.67, 0.80, 1.00]
results = {}

for omega in omega_values:
    # Create a new injector with different omega
    inj = bores.injection_well(
        well_name="CO2-INJ-1",
        perforating_intervals=[((0, 0, 0), (0, 0, 2))],
        radius=0.25,
        control=bores.RateControl(target_rate=500.0),
        injected_fluid=bores.InjectedFluid(
            name="CO2",
            phase=bores.FluidPhase.GAS,
            specific_gravity=1.52,
            molecular_weight=44.01,
            is_miscible=True,
            minimum_miscibility_pressure=1800.0,
            todd_longstaff_omega=omega,
            density=35.0,
            viscosity=0.05,
        ),
    )

    # Update wells and config
    wells = bores.wells_(injectors=[inj], producers=[producer])
    config = config.new(wells=wells)

    # Run simulation
    states = list(bores.run(model, config))

    # Store recovery factor
    time_d = np.array([s.time_in_days for s in states])
    avg_so = np.array([
        s.model.fluid_properties.oil_saturation_grid.mean() for s in states
    ])
    rf = (initial_So - avg_so) / initial_So
    results[f"omega = {omega:.2f}"] = np.column_stack([time_d, rf])

# Plot all recovery curves together
fig = bores.make_series_plot(
    data=results,
    title="Recovery Factor Sensitivity to Todd-Longstaff Omega",
    x_label="Time (days)",
    y_label="Recovery Factor (fraction of OOIP)",
)
fig.show()
```

You should observe the following trends:

- **$\omega$ = 0.33** (low mixing): Behaves closer to immiscible displacement. Poor mixing, lower recovery.
- **$\omega$ = 0.67** (default): Moderate mixing. Represents typical field-scale behavior.
- **$\omega$ = 1.00** (perfect mixing): Each grid cell is perfectly mixed. Highest recovery but may be optimistic for coarse grids.

!!! tip "Choosing Omega"

    The appropriate $\omega$ value depends on your grid resolution:

    - **Coarse grids (> 100 ft cells)**: Use $\omega$ = 0.5 - 0.67 to compensate for unresolved mixing
    - **Medium grids (30 - 100 ft cells)**: Use $\omega$ = 0.67 - 0.8
    - **Fine grids (< 30 ft cells)**: Use $\omega$ = 0.8 - 1.0 since the grid resolves more mixing
    - **When in doubt**: Start with $\omega$ = 2/3 (0.667)

---

## When to Use Miscible Flooding

Miscible gas flooding is most appropriate when:

**Reservoir pressure exceeds the MMP.** If the reservoir pressure is below the MMP, you cannot achieve miscibility without first re-pressurizing the reservoir (typically through waterflooding). This is why miscible floods are often implemented as tertiary recovery after a waterflood.

**The oil is light to medium gravity.** Heavier oils have higher MMPs, making miscibility harder to achieve. For oils below 25 API, the required MMP may exceed the fracture pressure of the formation.

**CO2 or enriched gas is available.** CO2 is the most common miscible injectant because it achieves miscibility at lower pressures than lean natural gas. Enriched gas (natural gas with added intermediate hydrocarbons) is an alternative when CO2 is not available.

**The reservoir has reasonable continuity.** Miscible floods are most effective in continuous, well-connected reservoirs where the solvent can contact a large fraction of the oil volume. Highly fractured or compartmentalized reservoirs may not benefit from miscible injection.

---

## Key Takeaways

1. **The Todd-Longstaff model** simulates miscible displacement by modifying effective oil viscosity and density based on solvent concentration and the mixing parameter $\omega$.

2. **`InjectedFluid` with `is_miscible=True`** enables miscible behavior. You must also provide `minimum_miscibility_pressure` and `todd_longstaff_omega`.

3. **CO2 properties require explicit overrides** (`density` and `viscosity` parameters) because standard gas correlations are inaccurate for CO2 at reservoir conditions.

4. **`miscibility_model="todd_longstaff"`** must be set in the `Config` to activate miscible simulation during the solver loop.

5. **The mixing parameter $\omega$** controls the degree of sub-grid mixing. Use 0.67 as a default and adjust based on grid resolution and calibration data.

6. **Miscible flooding eliminates residual oil trapping**, achieving near-100% displacement efficiency in swept zones, which can dramatically increase recovery compared to immiscible injection.

---

## What You Have Learned

Across these five tutorials, you have progressed from a simple depletion study to an advanced miscible flooding simulation:

1. **[Your First Simulation](01-first-simulation.md)** - The complete BORES workflow, primary depletion drive
2. **[Building Reservoir Models](02-building-models.md)** - Heterogeneity, anisotropy, structural dip
3. **[Waterflood Simulation](03-waterflood.md)** - Secondary recovery with water injection
4. **[Gas Injection](04-gas-injection.md)** - Immiscible gas displacement, gravity override
5. **Miscible Gas Flooding** (this tutorial) - Todd-Longstaff miscible displacement with CO2

You now have the foundational skills to set up, run, and analyze a wide range of reservoir simulation studies with BORES. For deeper exploration of specific topics, consult the [User Guide](../user-guide/index.md) for detailed coverage of wells, boundary conditions, PVT models, and solver configuration, or the [API Reference](../api-reference/index.md) for complete documentation of every class and function.
