# Gas Injection

Simulate immiscible gas injection, observe gravity override effects, and compare gas flood recovery against waterflooding.

---

## Overview

Gas injection is an alternative to waterflooding for reservoirs where water is scarce, where the oil is too viscous for efficient water displacement, or where associated gas needs to be reinjected. In this tutorial, you will set up an immiscible gas injection simulation, observe the characteristic gravity override phenomenon, and compare the results against the waterflood from the [previous tutorial](03-waterflood.md).

Immiscible gas injection means the injected gas does not mix with the oil at the molecular level. Instead, the gas forms a separate phase that displaces oil by pushing it ahead of the gas front. The key challenge is that gas is much lighter than oil and much less viscous, creating two problems: gravity override (gas rises to the top of the reservoir and bypasses oil below) and viscous fingering (the highly mobile gas front is unstable and fingers through the oil).

Understanding these challenges is essential before moving to the [next tutorial](05-miscible-flooding.md) on miscible flooding, where the gas actually mixes with oil to eliminate interfacial tension and achieve much higher displacement efficiency.

We use the same 15x15x3 grid from the waterflood tutorial so you can make a direct comparison between the two recovery methods.

---

## Physical Setup

The reservoir is identical to the waterflood tutorial, but the injection fluid changes from water to gas:

- **Grid**: 15x15x3 (675 cells)
- **Cell size**: 80 ft x 80 ft, 20 ft thick per layer
- **Permeability**: 150 mD isotropic
- **Initial pressure**: 3,000 psi
- **Oil viscosity**: 2.0 cP
- **Gas injector**: Corner (0, 0), 2,000 MSCF/day
- **Oil producer**: Corner (14, 14), adaptive BHP control

!!! warning "Gas Mobility"

    Gas viscosity at reservoir conditions is typically 0.01-0.03 cP, compared to 0.5-1.0 cP for water and 1-5 cP for oil. This extreme mobility difference (gas is 50-200 times more mobile than oil) means gas injection has a very unfavorable mobility ratio. Without gravity trapping, structural dip benefits, or miscibility, gas floods tend to have lower sweep efficiency than waterfloods.

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
sgr = bores.build_uniform_grid(grid_shape, value=0.05)
Swir = bores.build_uniform_grid(grid_shape, value=0.22)
swc = bores.build_uniform_grid(grid_shape, value=0.22)

# Build initial saturations from fluid contacts
depth = bores.build_depth_grid(thickness, datum=5000.0)

Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4900.0,  # GOC above reservoir (no initial gas cap)
    oil_water_contact=5100.0,  # OWC near reservoir bottom
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
    datum_depth=5000,
)
```

The model is intentionally identical to the waterflood tutorial. Using the same reservoir lets you make a clean comparison between the two injection strategies with no confounding geological differences.

---

## Step 2 - Define the Gas Injection Well

```python
gas_injector = bores.injection_well(
    well_name="GAS-INJ-1",
    perforating_intervals=[((0, 0, 0), (0, 0, 2))],
    radius=0.25,
    control=bores.RateControl(
        target_rate=5_000_000.0,
        bhp_limit=5000,
    ),
    injected_fluid=bores.InjectedFluid(
        name="Methane",
        phase=bores.FluidPhase.GAS,
        specific_gravity=0.65,
        molecular_weight=16.04,
    ),
)
```

The key difference from the waterflood is the `InjectedFluid` configuration. Here we inject gas (`phase=FluidPhase.GAS`) with a specific gravity of 0.65, which is typical for methane-rich natural gas. The molecular weight of 16.04 g/mol corresponds to pure methane.

The `target_rate=5_000_000.0` is in surface volume units (SCF/day for gas at standard conditions). BORES uses the gas formation volume factor to convert between surface and reservoir volumes internally.

Notice that we do not set `is_miscible=True`. By default, `InjectedFluid` creates an immiscible gas - the gas and oil remain as separate phases with a clear interface between them. Miscible injection, where the gas dissolves into and mixes with the oil, is covered in the [next tutorial](05-miscible-flooding.md).

!!! info "Gas Specific Gravity"

    The specific gravity of a gas is relative to air (air = 1.0). Common injection gases:

    - **Methane (CH4)**: 0.554
    - **Natural gas**: 0.6 - 0.8
    - **CO2**: 1.52
    - **Nitrogen (N2)**: 0.967

    Lighter gases (lower specific gravity) have stronger gravity override tendencies. CO2 is denser than methane and therefore shows less gravity segregation.

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
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.87,
            molecular_weight=200.0,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.0,
            molecular_weight=18.015,
        ),
        bores.ProducedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04,
        ),
    ],
)

wells = bores.wells_(injectors=[gas_injector], producers=[producer])

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
        simulation_time=bores.Time(days=1095),  # 3 years
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)
```

There are two important differences in the timer configuration compared to the waterflood. The initial step size is smaller (0.5 days vs 1 day) and the maximum step size is reduced (5 days vs 10 days). This is because gas injection creates sharper saturation fronts and higher flow velocities that require smaller time steps for numerical stability.

Gas is much more compressible and mobile than water, so the pressure and saturation equations change more rapidly near the gas front. The adaptive timer will manage this automatically, but starting with smaller steps reduces the number of rejected steps during the early, most dynamic phase of the flood.

!!! tip "Time Step Management for Gas Injection"

    If you encounter convergence issues during gas injection, try:

    1. Reducing `initial_step_size` to `Time(hours=6)` or smaller
    2. Reducing `maximum_step_size` to `Time(days=3)`
    3. Increasing the gas Corey exponent (e.g., 3.0) to smooth the gas relative permeability curve
    4. These adjustments help the solver handle the high-mobility gas front

---

## Step 4 - Run the Simulation

```python
states = list(bores.run(model, config))
final = states[-1]
print(f"Completed {final.step} steps in {final.time_in_days:.1f} days")
print(f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi")
print(f"Final avg gas saturation: {final.model.fluid_properties.gas_saturation_grid.mean():.4f}")
```

---

## Step 5 - Observe Gravity Override

Gravity override is the dominant feature of immiscible gas injection. Because gas is lighter than oil, it tends to rise to the top of the reservoir and flow along the upper layers, bypassing oil in the lower layers.

```python
# Gas saturation in top layer vs bottom layer over time
time_days = np.array([s.time_in_days for s in states])
avg_Sg_top = np.array([
    s.model.fluid_properties.gas_saturation_grid[:, :, 0].mean() for s in states
])
avg_Sg_bottom = np.array([
    s.model.fluid_properties.gas_saturation_grid[:, :, 2].mean() for s in states
])

fig = bores.make_series_plot(
    data={
        "Gas Saturation (Top Layer)": np.column_stack([time_days, avg_Sg_top]),
        "Gas Saturation (Bottom Layer)": np.column_stack([time_days, avg_Sg_bottom]),
    },
    title="Gas Saturation by Layer - Gravity Override",
    x_label="Time (days)",
    y_label="Gas Saturation (fraction)",
)
fig.show()
```

You should see gas saturation increasing much faster in the top layer than in the bottom layer. This is gravity override in action. The gas preferentially flows through the upper part of the reservoir, leaving significant unswept oil in the lower layers.

### Visualize the Gas Front in 3D

```python
viz = bores.plotly3d.DataVisualizer()

fig = viz.make_plot(
    source=states[len(states) // 2],
    property="gas-saturation",
    plot_type="volume",
    title=f"Gas Saturation at Day {states[len(states) // 2].time_in_days:.0f}",
)
fig.show()
```

The 3D visualization makes the gravity override visually obvious. You should see high gas saturation concentrated in the top layer, with much less gas penetration into the lower layers. This non-uniform sweep pattern is the primary limitation of immiscible gas injection in reservoirs without significant structural dip.

---

## Step 6 - Compare Gas Flood vs Waterflood Recovery

```python
avg_So = np.array([s.model.fluid_properties.oil_saturation_grid.mean() for s in states])

initial_So = 0.75
Sor_gas = 0.15  # Residual oil to gas
recovery_factor = (initial_So - avg_So) / (initial_So - Sor_gas)

fig = bores.make_series_plot(
    data=np.column_stack([time_days, recovery_factor]),
    title="Oil Recovery Factor (Gas Injection)",
    x_label="Time (days)",
    y_label="Recovery Factor (fraction)",
)
fig.show()
```

Compare this recovery curve with the waterflood result from the [previous tutorial](03-waterflood.md). In most cases, the gas flood will show:

1. **Earlier breakthrough** because gas moves much faster through the reservoir
2. **Lower ultimate recovery** because gravity override leaves unswept oil in lower layers
3. **Different curve shape** with a steeper initial rise but earlier flattening

The residual oil saturation to gas (sorg = 0.15) is actually lower than to water (sorw = 0.22), meaning gas can displace oil more efficiently at the microscopic (pore) level. But the macroscopic sweep efficiency is worse due to gravity override and viscous fingering, so the overall recovery is often lower.

---

## Step 7 - Pressure Response

```python
avg_pressure = np.array([s.model.fluid_properties.pressure_grid.mean() for s in states])

fig = bores.make_series_plot(
    data=np.column_stack([time_days, avg_pressure]),
    title="Average Reservoir Pressure (Gas Injection)",
    x_label="Time (days)",
    y_label="Pressure (psi)",
)
fig.show()
```

Gas injection provides some pressure maintenance, but the response differs from waterflooding. Gas is highly compressible, so a large volume of gas at surface conditions translates to a relatively small volume at reservoir conditions. The pressure support from gas injection is typically less effective than from water injection at the same surface injection rate.

---

## Discussion

The gas injection results highlight the fundamental trade-off of gas flooding. At the pore scale, gas is an excellent displacement agent - it can reduce oil saturation below what water achieves (sorg < sorw). But at the reservoir scale, the extreme mobility contrast and density difference between gas and oil lead to poor sweep efficiency through gravity override and viscous fingering.

This is why gas injection is often most effective in reservoirs with significant structural dip (where gravity helps push gas updip through the oil column), in thin reservoirs (where there is less room for gravity segregation), or when the gas is injected at conditions that achieve miscibility with the oil (eliminating the interfacial tension that causes trapping).

In real field operations, operators often use WAG (Water Alternating Gas) injection to combine the microscopic displacement efficiency of gas with the better sweep efficiency of water. The water slugs control the gas mobility and reduce gravity override, while the gas slugs provide superior pore-level displacement.

---

## Key Takeaways

1. **Gas injection** uses `InjectedFluid` with `phase=FluidPhase.GAS` and appropriate gas properties (specific gravity, molecular weight).

2. **Gravity override** is the dominant challenge: gas rises to the top of the reservoir, bypassing oil in lower layers and reducing sweep efficiency.

3. **Smaller time steps** are needed for gas injection (compared to waterflooding) because gas fronts are sharper and gas is more compressible.

4. **Pore-level displacement** by gas is efficient (low sorg), but **macroscopic sweep** is poor due to unfavorable mobility ratio and density contrast.

5. **Comparison with waterflooding** shows that immiscible gas injection typically achieves earlier breakthrough and lower ultimate recovery in flat, thick reservoirs.

---

## Next Steps

In the [next tutorial](05-miscible-flooding.md), you will learn how to make gas injection dramatically more effective by achieving miscibility between the injected gas and the reservoir oil. The Todd-Longstaff model in BORES enables you to simulate miscible displacement where the gas-oil interfacial tension vanishes, eliminating residual oil trapping in the swept zone.
