# Waterflood Simulation

Set up a complete waterflood with injection and production wells, track water breakthrough, and analyze oil recovery.

---

## Overview

Waterflooding is the most widely used secondary recovery technique in the oil industry. By injecting water into the reservoir, you maintain pressure (preventing the decline you saw in the [depletion tutorial](01-first-simulation.md)) and physically displace oil toward the production wells. The key engineering questions are: when does water break through at the producer? How much oil can you recover? How does the water cut evolve over time?

In this tutorial, you will set up a waterflood simulation with one injection well and one production well in opposite corners of the grid (a "corner-to-corner" pattern). You will track water breakthrough, water cut, oil recovery factor, and pressure maintenance. At the end, you will compare the waterflood results against the depletion case from the first tutorial to quantify the benefit of water injection.

Waterflooding works because water is cheap, readily available (especially offshore), and displaces oil reasonably efficiently in many reservoir types. The injected water pushes a saturation front through the reservoir. Ahead of the front, oil saturation is high and water saturation is at connate levels. Behind the front, oil saturation has been reduced to the residual value and water saturation is high. The sharpness and stability of this front depend on the mobility ratio between water and oil.

---

## Physical Setup

We use a 15x15x3 reservoir with a water injector in corner (0, 0) and an oil producer in the opposite corner (14, 14). Both wells are perforated across all three layers.

- **Grid**: 15x15x3 (675 cells)
- **Cell size**: 80 ft x 80 ft, 20 ft thick per layer
- **Porosity**: 20%
- **Permeability**: 150 mD isotropic
- **Initial pressure**: 3,000 psi
- **Oil viscosity**: 2.0 cP
- **Injection rate**: 400 STB/day water
- **Production rate**: 400 STB/day (adaptive BHP control)

!!! info "Mobility Ratio"

    The mobility ratio $M = \lambda_w / \lambda_o$ controls sweep efficiency. Here, water viscosity (~0.5 cP at reservoir conditions) is much lower than oil viscosity (2.0 cP), giving an unfavorable mobility ratio ($M > 1$). This means the water front will be somewhat unstable, with water tending to finger through the oil rather than pushing it as a clean piston. Higher oil viscosity or lower water mobility would improve this ratio.

---

## Step 1 - Build the Reservoir Model

```python
import bores
import numpy as np

bores.use_32bit_precision()

grid_shape = (15, 15, 3)
cell_dimension = (80.0, 80.0)

# Property grids
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

The setup is similar to the first tutorial with a few differences. The oil viscosity is slightly higher (2.0 cP) to make the displacement dynamics more interesting. The residual oil saturation during waterflood (sorw = 0.22) determines the maximum oil recovery: in the swept zone, oil saturation will drop to 22%, meaning up to 78% of the pore volume in the swept region is eventually occupied by water.

We use `build_saturation_grids()` to compute physically consistent initial saturations from the fluid contact depths. The GOC is placed above the reservoir (no gas cap) and the OWC near the bottom, giving us an oil-filled reservoir with connate water. This approach ensures $S_o + S_w + S_g = 1.0$ in every cell and uses the correct residual saturations for each zone.

Note that we pass only the x-direction permeability to `Permeability`. BORES automatically copies it to y and z, giving us isotropic permeability with less code.

---

## Step 2 - Define the Injection Well

```python
injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((0, 0, 0), (0, 0, 2))],
    radius=0.25,
    control=bores.RateControl(target_rate=400.0),
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.0,
        molecular_weight=18.015,
    ),
)
```

The injection well is placed at grid corner (0, 0) and perforated from layer 0 to layer 2. The `RateControl` with `target_rate=400.0` (positive = injection) injects water at a constant 400 STB/day. We do not set a BHP limit on the injector in this example, though in practice you would want to limit injection pressure to avoid fracturing the formation.

The `InjectedFluid` specifies that we are injecting water with standard properties. BORES uses the specific gravity and molecular weight to compute water density and viscosity at reservoir conditions using its internal correlations.

!!! tip "Matching Injection and Production Rates"

    In a voidage-replacement waterflood, you typically match the volumetric injection rate to the volumetric production rate. This maintains reservoir pressure at approximately the initial level. If injection exceeds production, pressure rises. If production exceeds injection, pressure declines despite the water support.

---

## Step 3 - Define the Production Well

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
    ],
)
```

The producer sits in the opposite corner at (14, 14). We use `ProducerRateControl` with oil as the primary phase and a target rate of -400 STB/day. The simulator computes the BHP needed to deliver that oil rate, and water flows naturally at whatever rate corresponds to that BHP. The `bhp_limit=800.0` prevents the BHP from dropping below 800 psi. The `ProductionClamp` prevents any accidental backflow of water into the reservoir.

Notice that the `produced_fluids` list includes both oil and water. Before water breakthrough, the producer will mainly produce oil. After breakthrough, an increasing fraction of the produced fluid will be water. Because we use `ProducerRateControl`, the oil rate stays at the target while the water rate grows with increasing water cut.

---

## Step 4 - Group Wells and Configure

```python
wells = bores.wells_(injectors=[injector], producers=[producer])

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
        initial_step_size=bores.Time(days=1),
        maximum_step_size=bores.Time(days=10),
        minimum_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(days=1095),  # 3 years
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)
```

We set the water Corey exponent to 2.5 (slightly higher than oil's 2.0), which makes the water relative permeability curve steeper. This is physically reasonable because water in a water-wet rock (`BrooksCoreyRelPermModel` assumes wettability as water wet by default) needs to occupy more pore space before it can flow efficiently.

The simulation runs for 3 years (1,095 days), which is enough time to observe water breakthrough, the transition to high water cut, and the plateau in recovery factor.

---

## Step 5 - Run the Simulation

```python
states = list(bores.run(model, config))
final = states[-1]
print(f"Completed {final.step} steps in {final.time_in_days:.1f} days")
```

---

## Step 6 - Analyze Water Breakthrough

Water breakthrough is the moment when injected water first arrives at the production well. Before breakthrough, the producer makes essentially dry oil. After breakthrough, water cut increases rapidly.

```python
time_days = np.array([s.time_in_days for s in states])

# Water saturation at the producer location (14, 14, center layer)
Sw_at_producer = np.array([
    s.model.fluid_properties.water_saturation_grid[14, 14, 1] for s in states
])

# Detect breakthrough: first time Sw exceeds initial connate value significantly
breakthrough_mask = Sw_at_producer > 0.30
if np.any(breakthrough_mask):
    bt_index = np.argmax(breakthrough_mask)
    print(f"Water breakthrough at approximately {time_days[bt_index]:.0f} days")
else:
    print("Water has not broken through during the simulation period")
```

The timing of breakthrough depends on the distance between injector and producer, the injection rate, the pore volume between them, and the mobility ratio. In an ideal piston-like displacement, breakthrough would occur when one pore volume of water has been injected. In reality, the unfavorable mobility ratio causes early breakthrough because water fingers through the oil.

---

## Step 7 - Plot Water Cut and Recovery

```python
# Average saturations over time
avg_So = np.array([s.model.fluid_properties.oil_saturation_grid.mean() for s in states])
avg_Sw = np.array([s.model.fluid_properties.water_saturation_grid.mean() for s in states])

# Water cut: fraction of total production that is water
# Approximate using average saturation change
initial_So = 0.75
oil_recovered_fraction = (initial_So - avg_So) / (initial_So - 0.22)  # Normalize by movable oil

fig = bores.make_series_plot(
    data={
        "Oil Saturation": np.column_stack([time_days, avg_So]),
        "Water Saturation": np.column_stack([time_days, avg_Sw]),
    },
    title="Average Saturations During Waterflood",
    x_label="Time (days)",
    y_label="Saturation (fraction)",
)
fig.show()
```

You should see water saturation gradually increasing from the initial connate value (0.25) as the flood front progresses through the reservoir. Oil saturation decreases correspondingly. After breakthrough, the rate of change accelerates because water is now taking a shortcut through the swept zone to the producer.

### Recovery Factor

```python
fig = bores.make_series_plot(
    data=np.column_stack([time_days, oil_recovered_fraction]),
    title="Oil Recovery Factor (Waterflood)",
    x_label="Time (days)",
    y_label="Recovery Factor (fraction)",
)
fig.show()
```

The recovery factor curve is the primary metric for evaluating a waterflood. You should see rapid initial recovery as the flood front sweeps through the reservoir, a change in slope around breakthrough time (when water starts being produced instead of displacing oil), and a gradual approach toward the ultimate recovery limit.

---

## Step 8 - Pressure Maintenance

```python
avg_pressure = np.array([s.model.fluid_properties.pressure_grid.mean() for s in states])

fig = bores.make_series_plot(
    data=np.column_stack([time_days, avg_pressure]),
    title="Average Reservoir Pressure (Waterflood)",
    x_label="Time (days)",
    y_label="Pressure (psi)",
)
fig.show()
```

Unlike the depletion case from Tutorial 1, where pressure declined continuously, the waterflood should maintain pressure close to the initial value (3,000 psi) because the injected water replaces the produced fluid volume. You may see slight pressure fluctuations as the flood front moves through the reservoir and the simulator adjusts well rates.

!!! note "Pressure vs Depletion"

    Compare this pressure plot with the depletion result from [Your First Simulation](01-first-simulation.md). The difference is dramatic: waterflooding maintains pressure near the initial value, while depletion allows continuous decline. This pressure support is the primary benefit of waterflooding, enabling higher production rates for longer periods.

---

## Step 9 - Visualize the Water Front

```python
viz = bores.plotly3d.DataVisualizer()

# Visualize water saturation at a few time steps
for state in [states[len(states) // 4], states[len(states) // 2], states[-1]]:
    fig = viz.make_plot(
        source=state,
        property="water-saturation",
        plot_type="volume",
        title=f"Water Saturation at Day {state.time_in_days:.0f}",
    )
    fig.show()
```

These 3D visualizations show the water saturation front advancing from the injector corner (0, 0) toward the producer corner (14, 14). You should observe the front spreading diagonally across the reservoir. The front may not be perfectly sharp due to numerical dispersion and the unfavorable mobility ratio.

---

## Discussion

The waterflood results illustrate several important principles of secondary recovery. First, water injection dramatically improves recovery compared to primary depletion. The depletion case from Tutorial 1 recovered oil only through pressure decline and solution gas drive, while the waterflood physically displaces oil by pushing it toward the producer.

Second, the timing and sharpness of water breakthrough depend on the mobility ratio. With our oil viscosity of 2.0 cP and water viscosity around 0.5 cP, the mobility ratio is approximately 4:1 (unfavorable). This means water moves about 4 times faster than oil at the same saturation, leading to early breakthrough and a gradual increase in water cut rather than a sharp transition.

Third, pressure maintenance is a major advantage of waterflooding. By keeping reservoir pressure above the bubble point, we avoid the complications of free gas (gas coning, reduced oil mobility) that plague depletion operations. This is why waterflooding is typically initiated early in field life, before significant pressure decline.

---

## Key Takeaways

1. **Water injection** maintains reservoir pressure and displaces oil, yielding significantly higher recovery than primary depletion.

2. **`InjectedFluid` with `phase=FluidPhase.WATER`** configures a water injection well. The positive `target_rate` follows BORES's sign convention (positive = injection).

3. **Water breakthrough** occurs when injected water first reaches the producer. After breakthrough, water cut increases and oil production rate declines.

4. **Mobility ratio** ($M = \lambda_w / \lambda_o$) controls sweep efficiency. Higher oil viscosity or lower Corey exponents for water improve the displacement.

5. **3D visualization** of the saturation front helps you understand how the flood progresses through the reservoir and identify sweep inefficiencies.

---

## Next Steps

In the [next tutorial](04-gas-injection.md), you will replace the water injector with a gas injector and observe how gas injection differs from waterflooding: gravity override, higher gas mobility, and different displacement characteristics.
