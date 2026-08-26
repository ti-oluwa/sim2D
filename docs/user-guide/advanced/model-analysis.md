# Model Analysis

## Overview

After running a simulation, you need to evaluate how the reservoir performed: how much oil was recovered, which drive mechanisms dominated, how efficiently the waterflood swept the reservoir, whether wells are producing optimally, and what future production might look like. The `ModelAnalyst` class provides all of these analyses in a single, coherent API.

The analyst accepts a collection of `ModelState` objects and indexes them by step number. From there, it can compute original oil and gas in place (STOIIP, STGIIP), cumulative production and injection volumes, instantaneous flow rates, material balance drive indices, sweep efficiency metrics, well productivity, voidage replacement ratios, decline curves, production forecasts, and estimated ultimate recovery. All computations use field units (STB for oil, SCF for gas, psi for pressure) and convert from internal simulator units automatically.

Results from the analyst are returned as frozen attrs classes. Each result type is a self-documenting data container with named fields and units in the docstrings. This means you can inspect the result object to see exactly what each number represents, and you can pass result objects to downstream code without worrying about mutation.

The analyst caches computed results internally. Repeated calls with the same parameters return cached values without recomputation. This makes it efficient to call multiple methods that depend on the same underlying production totals.

---

## Creating an Analyst

The `ModelAnalyst` class accepts any iterable of `ModelState` objects. You can pass a list collected from `bores.run()`, a replay from a `StateStream`, or states loaded from a store:

```python
from bores.analyses import ModelAnalyst

# From a list of states
states = list(bores.run(model, config))
analyst = ModelAnalyst(states)

# From a stream replay
analyst = ModelAnalyst(stream.replay())

# From a store directly
from bores.states import ModelState

with store(mode="r") as s:
    analyst = ModelAnalyst(s.load(ModelState))
```

The analyst stores all states internally in a dictionary keyed by step number. This means it needs to load all states into memory. For very large simulations, consider replaying only a subset of states (using `replay(steps=...)` or `replay(indices=...)`) to reduce memory.

### EOR and Continuation Scenarios

When the earliest available state is not step 0, or when you want to override the initial volumes (for example, in an EOR simulation that starts from a depleted state), you can provide pre-calculated initial volumes:

```python
analyst = ModelAnalyst(
    states,
    stoiip=5_000_000.0,  # STB
    stgiip=2_000_000_000.0,  # SCF
    stwiip=10_000_000.0,  # STB
)
```

If you do not provide these values, the analyst computes them from the earliest available state using the hydrocarbon-in-place calculation with formation volume factors and pore volume.

### Navigation Properties

The analyst exposes several properties for navigating the state collection:

```python
analyst.min_step  # Earliest step number
analyst.max_step  # Latest step number
analyst.available_steps  # Sorted list of all step numbers

# Retrieve a specific state
state = analyst.get_state(50)  # Returns None if step 50 is not available
```

---

## Initial Volumes

The analyst computes stock tank oil initially in place (STOIIP) and stock tank gas initially in place (STGIIP) from the earliest available state. These values serve as the denominators for recovery factor calculations.

```python
# Full property names
print(f"STOIIP: {analyst.stock_tank_oil_initially_in_place:,.0f} STB")
print(f"STGIIP: {analyst.stock_tank_gas_initially_in_place:,.0f} SCF")
print(f"STWIIP: {analyst.stock_tank_water_initially_in_place:,.0f} STB")

# Short aliases
print(f"STOIIP: {analyst.stoiip:,.0f} STB")
print(f"STGIIP: {analyst.stgiip:,.0f} SCF")
```

The `stoiip` and `stgiip` aliases are convenient shorthand for the full property names. Both return the same value.

---

## Fluids in Place

These methods compute the volume of each phase remaining in the reservoir at a given time step. They account for formation volume factors to convert from reservoir conditions to stock tank conditions.

```python
# Oil, gas, and water in place at the final step
oil_remaining = analyst.oil_in_place(step=-1)  # STB
gas_remaining = analyst.gas_in_place(step=-1)  # SCF (free + solution gas)
water_in_place = analyst.water_in_place(step=-1)  # STB

# Free gas only (excludes solution gas dissolved in oil)
free_gas_remaining = analyst.free_gas_in_place(step=-1)  # SCF

# At a specific step
oil_at_50 = analyst.oil_in_place(step=50)
```

All four methods accept a `step` parameter that defaults to `-1` (the last available step). Negative indices work like Python lists.

The distinction between `gas_in_place()` and `free_gas_in_place()` matters for reservoirs with significant solution gas. `gas_in_place()` returns the total gas volume including gas dissolved in oil (computed from the solution GOR and oil volume), while `free_gas_in_place()` counts only the free gas phase occupying pore space. In an undersaturated reservoir with no gas cap, `free_gas_in_place()` returns zero even though `gas_in_place()` may be large because of dissolved gas.

### In-Place History

To track how fluids in place change over time, use the history generators:

```python
# Oil in place at every 10th step
for step, oil_ip in analyst.oil_in_place_history(from_step=0, to_step=-1, interval=10):
    print(f"Step {step}: {oil_ip:,.0f} STB remaining")

# Gas in place history
for step, gas_ip in analyst.gas_in_place_history(interval=5):
    print(f"Step {step}: {gas_ip:,.0f} SCF remaining")

# Water in place history
for step, water_ip in analyst.water_in_place_history():
    print(f"Step {step}: {water_ip:,.0f} STB")
```

Each history method accepts `from_step`, `to_step`, and `interval` parameters and returns a generator of `(step, value)` tuples.

---

## Cumulative Production and Injection

The analyst can compute cumulative production and injection between any two time steps. These methods sum the per-step volumes across the specified range, converting from reservoir cubic feet per day to stock tank barrels (for oil and water) or standard cubic feet (for gas) using the formation volume factor at each step.

### Production

```python
# Total oil produced over the entire simulation
total_oil = analyst.oil_produced(from_step=0, to_step=-1)  # STB

# Oil produced in a specific interval
oil_50_to_100 = analyst.oil_produced(from_step=50, to_step=100)

# Gas and water
gas_produced = analyst.free_gas_produced(from_step=0, to_step=-1)  # SCF
water_produced = analyst.water_produced(from_step=0, to_step=-1)  # STB
```

### Injection

```python
# Total injection volumes
oil_injected = analyst.oil_injected(from_step=0, to_step=-1)  # STB
water_injected = analyst.water_injected(from_step=0, to_step=-1)  # STB
gas_injected = analyst.gas_injected(from_step=0, to_step=-1)  # SCF
```

### Cell Filtering

All production and injection methods accept a `cells` parameter that filters the calculation to specific cells, wells, or regions. The filter applies to the grid cells where production or injection occurs.

```python
# Production from a specific well (by name)
oil_from_well = analyst.oil_produced(0, -1, cells="PROD-1")

# Production from a single cell
oil_from_cell = analyst.oil_produced(0, -1, cells=(5, 5, 0))

# Production from multiple cells
oil_from_cells = analyst.oil_produced(0, -1, cells=[(5, 5, 0), (6, 5, 0), (7, 5, 0)])

# Production from a region (using slices)
oil_from_region = analyst.oil_produced(0, -1, cells=(slice(0, 10), slice(0, 10), slice(None)))
```

The `cells` parameter accepts:

| Type | Example | Description |
| --- | --- | --- |
| `None` | `cells=None` | Entire reservoir (default) |
| `str` | `cells="PROD-1"` | Well name |
| `Well/Wells` | `cells=production_well` | `Well` or `Wells` object |
| `Sequence[Well]` | `cells=[production_well, injection_well]` | Sequence of `Well` objects |
| `tuple(int,int,int)` | `cells=(5,5,0)` | Single cell index |
| `list[tuple]` | `cells=[(5,5,0),(6,5,0)]` | Multiple cell indices |
| `tuple(slice,...)` | `cells=(slice(0,10),...)` | Grid region |

### Cumulative Properties

For quick access to full-simulation cumulative values, the analyst provides read-only properties:

```python
print(f"Cumulative oil: {analyst.cumulative_oil_produced:,.0f} STB")
print(f"Cumulative gas: {analyst.cumulative_free_gas_produced:,.0f} SCF")
print(f"Cumulative water: {analyst.cumulative_water_produced:,.0f} STB")

# Short aliases using standard petroleum notation
print(f"Np: {analyst.No:,.0f} STB")  # Same as cumulative_oil_produced
print(f"Gp: {analyst.Ng:,.0f} SCF")  # Same as cumulative_gas_produced
print(f"Wp: {analyst.Nw:,.0f} STB")  # Same as cumulative_water_produced
```

---

## Recovery Factors

Recovery factors express cumulative production as a fraction of the original volume in place. The analyst provides several recovery factor properties:

```python
# Oil recovery factor (cumulative oil / STOIIP)
print(f"Oil RF: {analyst.oil_recovery_factor:.2%}")

# Gas recovery factor (free gas + solution gas produced / STGIIP)
print(f"Gas RF: {analyst.gas_recovery_factor:.2%}")
```

The `gas_recovery_factor` is a comprehensive metric that accounts for both free gas produced from the gas phase and solution gas liberated from produced oil. The denominator is the total initial gas in place (free gas cap plus gas dissolved in oil at initial conditions). This gives a true measure of total gas resource recovery.

### Recovery Factor History

To track recovery factors over time:

```python
for step, rf in analyst.oil_recovery_factor_history(interval=10):
    print(f"Step {step}: Oil RF = {rf:.2%}")

for step, rf in analyst.gas_recovery_factor_history(interval=10):
    print(f"Step {step}: Gas RF = {rf:.2%}")
```

---

## Production and Injection Histories

The history methods return generators of `(step, value)` tuples, which are efficient for building time series plots or feeding into numpy arrays. Each method supports `from_step`, `to_step`, `interval`, `cumulative`, and `cells` parameters.

```python
# Per-step oil production rate (STB at each step)
for step, rate in analyst.oil_production_history(interval=5):
    print(f"Step {step}: {rate:.0f} STB")

# Cumulative oil production
for step, cum in analyst.oil_production_history(cumulative=True):
    print(f"Step {step}: {cum:,.0f} STB cumulative")

# Gas production from a specific well
for step, rate in analyst.gas_production_history(cells="PROD-1"):
    print(f"Step {step}: {rate:.0f} SCF")

# Water production
for step, rate in analyst.water_production_history(interval=10):
    print(f"Step {step}: {rate:.0f} STB")
```

Injection histories follow the same pattern:

```python
for step, rate in analyst.oil_injection_history(interval=5):
    print(f"Step {step}: {rate:.0f} STB injected")

for step, rate in analyst.gas_injection_history(cumulative=True):
    print(f"Step {step}: {rate:,.0f} SCF cumulative")

for step, rate in analyst.water_injection_history():
    print(f"Step {step}: {rate:.0f} STB injected")
```

### History Method Parameters

All history methods share these parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `from_step` | 0 | Starting step index (inclusive) |
| `to_step` | -1 | Ending step index (inclusive, -1 for last) |
| `interval` | 1 | Step sampling interval |
| `cumulative` | `False` | If `True`, return running cumulative totals |
| `cells` | `None` | Cell filter (well name, cell tuple, or region) |

---

## Reservoir Volumetrics Analysis

The `reservoir_volumetrics_analysis()` method computes a comprehensive volumetric snapshot at a single time step. It returns a `ReservoirVolumetrics` object containing oil, gas, and water in place, pore volume, and hydrocarbon pore volume.

```python
vol = analyst.reservoir_volumetrics_analysis(step=-1)
print(f"Oil in place: {vol.oil_in_place:,.0f} STB")
print(f"Gas in place: {vol.gas_in_place:,.0f} SCF")
print(f"Water in place: {vol.water_in_place:,.0f} STB")
print(f"Pore volume: {vol.pore_volume:,.0f} ft³")
print(f"HCPV: {vol.hydrocarbon_pore_volume:,.0f} ft³")
```

The `ReservoirVolumetrics` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `oil_in_place` | STB | Total oil in place |
| `gas_in_place` | SCF | Total gas in place |
| `water_in_place` | STB | Total water in place |
| `pore_volume` | ft³ | Total pore volume |
| `hydrocarbon_pore_volume` | ft³ | Hydrocarbon-bearing pore volume |

### Volumetrics History

```python
for step, vol in analyst.reservoir_volumetrics_history(interval=20):
    print(f"Step {step}: OIP={vol.oil_in_place:,.0f}, PV={vol.pore_volume:,.0f}")
```

---

## Cumulative Production Analysis

The `cumulative_production_analysis()` method provides a summary of cumulative production along with recovery factors in a single result object:

```python
cum = analyst.cumulative_production_analysis(step=-1)
print(f"Cumulative oil: {cum.cumulative_oil:,.0f} STB")
print(f"Cumulative gas: {cum.cumulative_free_gas:,.0f} SCF")
print(f"Cumulative water: {cum.cumulative_water:,.0f} STB")
print(f"Oil RF: {cum.oil_recovery_factor:.2%}")
print(f"Gas RF: {cum.gas_recovery_factor:.2%}")
```

The `CumulativeProduction` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `cumulative_oil` | STB | Cumulative oil produced |
| `cumulative_free_gas` | SCF | Cumulative free gas produced |
| `cumulative_water` | STB | Cumulative water produced |
| `oil_recovery_factor` | fraction | Oil recovery as fraction of STOIIP |
| `gas_recovery_factor` | fraction | Gas recovery as fraction of STGIIP |

### Cumulative Production History

```python
for step, cum in analyst.cumulative_production_history(interval=10):
    print(f"Step {step}: Oil RF = {cum.oil_recovery_factor:.2%}")
```

---

## Instantaneous Rates

The `instantaneous_production_rates()` and `instantaneous_injection_rates()` methods compute snapshot rates at a single time step. Unlike the cumulative methods (which sum over a range), these methods report what is happening right now at the specified step. Rates are converted to surface conditions using formation volume factors.

```python
rates = analyst.instantaneous_production_rates(step=-1)
print(f"Oil rate: {rates.oil_rate:,.0f} STB/day")
print(f"Gas rate: {rates.gas_rate:,.0f} SCF/day")
print(f"Water rate: {rates.water_rate:,.0f} STB/day")
print(f"Total liquid: {rates.total_liquid_rate:,.0f} STB/day")
print(f"Water cut: {rates.water_cut:.2%}")
print(f"GOR: {rates.gas_oil_ratio:,.0f} SCF/STB")
```

Both methods accept an optional `cells` parameter for filtering by well or region:

```python
# Rates for a specific well
well_rates = analyst.instantaneous_production_rates(step=-1, cells="PROD-1")

# Injection rates
inj_rates = analyst.instantaneous_injection_rates(step=-1, cells="INJ-1")
```

The `InstantaneousRates` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `oil_rate` | STB/day | Oil production/injection rate |
| `gas_rate` | SCF/day | Total gas rate (free gas + solution gas from oil) |
| `water_rate` | STB/day | Water production/injection rate |
| `total_liquid_rate` | STB/day | Oil + water rate |
| `gas_oil_ratio` | SCF/STB | Produced GOR (free gas + solution gas) / oil |
| `water_cut` | fraction | Water cut (0 to 1) |
| `free_gas_rate` | SCF/day | Free gas phase rate only |
| `solution_gas_rate` | SCF/day | Solution gas (dissolved in produced oil, released at surface) |

### Instantaneous Rates History

```python
for step, rates in analyst.instantaneous_rates_history(interval=5):
    print(f"Step {step}: Oil={rates.oil_rate:.0f}, WC={rates.water_cut:.2%}")
```

---

## Material Balance Analysis

The `material_balance_analysis()` method identifies and quantifies the drive mechanisms in your reservoir using the generalized material balance equation. The drive indices sum to 1.0 and indicate the relative contribution of each mechanism to production.

```python
mbal = analyst.material_balance_analysis(step=-1)
print(f"Reservoir pressure: {mbal.pressure:.0f} psi")
print(f"Oil expansion: {mbal.oil_expansion_factor:.4f}")
print(f"Solution gas drive: {mbal.solution_gas_drive_index:.2%}")
print(f"Gas cap drive: {mbal.gas_cap_drive_index:.2%}")
print(f"Water drive: {mbal.water_drive_index:.2%}")
print(f"Compaction drive: {mbal.compaction_drive_index:.2%}")
print(f"Aquifer influx: {mbal.aquifer_influx:,.0f} STB")
```

The short alias `analyst.mbal(step=-1)` is equivalent to `analyst.material_balance_analysis(step=-1)`.

The `MaterialBalanceAnalysis` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `pressure` | psia | Average reservoir pressure |
| `oil_expansion_factor` | dimensionless | Oil expansion relative to initial conditions |
| `solution_gas_drive_index` | fraction | Fraction of production from solution gas expansion |
| `gas_cap_drive_index` | fraction | Fraction from gas cap expansion |
| `water_drive_index` | fraction | Fraction from water influx |
| `compaction_drive_index` | fraction | Fraction from pore compaction and fluid expansion |
| `aquifer_influx` | STB | Estimated cumulative aquifer water influx |

### Material Balance History

```python
for step, mbal in analyst.material_balance_history(interval=20):
    print(f"Step {step}: P={mbal.pressure:.0f}, SGD={mbal.solution_gas_drive_index:.2%}")
```

---

## Material Balance Error

The `material_balance_error()` method quantifies how well the simulator conserves mass over a simulation interval. It computes two complementary error metrics: per-phase material balance errors (oil, water, gas) that test whether the saturation solver conserved pore-volume occupancy for each phase, and a total Havlena-Odeh material balance error that tests whether expansion energy accounts for underground withdrawal.

The per-phase errors are computed as the difference between the change in pore volume occupied by a phase and the net flux of that phase, divided by the initial pore volume. A value of zero means the simulator perfectly conserved that phase. The total error uses the standard Havlena-Odeh formulation where expansion terms (solution gas drive, gas cap drive, water drive, and compaction drive) should account for all underground withdrawal. Together, the two checks diagnose different problems: a large phase error points to a numerical conservation bug in the simulator, while a large total error suggests a PVT or STOIIP mismatch.

Each result includes a `quality` rating based on standard industry thresholds. An absolute MBE below 0.1% is rated "excellent" and indicates reliable results. Between 0.1% and 1% is "acceptable" but worth monitoring for drift. Between 1% and 5% is "marginal", suggesting you should refine the grid or reduce the timestep. Above 5% is "unacceptable", and you should investigate before using the results.

```python
mbe = analyst.material_balance_error(from_step=0, to_step=-1)

print(f"Quality: {mbe.quality}")
print(f"Total MBE: {mbe.total_mbe:.4%}")
print(f"Oil MBE: {mbe.oil_mbe:.4%}")
print(f"Water MBE: {mbe.water_mbe:.4%}")
print(f"Gas MBE: {mbe.gas_mbe:.4%}")

# Drive mechanism breakdown
print(f"Solution gas drive: {mbe.solution_gas_drive_index:.2%}")
print(f"Gas cap drive: {mbe.gas_cap_drive_index:.2%}")
print(f"Water drive: {mbe.water_drive_index:.2%}")
print(f"Compaction drive: {mbe.compaction_drive_index:.2%}")
```

The short alias `analyst.mbe(...)` is equivalent to `analyst.material_balance_error(...)`.

The `MaterialBalanceError` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `total_mbe` | dimensionless | Havlena-Odeh total material balance error |
| `oil_mbe` | dimensionless | Oil phase conservation error |
| `water_mbe` | dimensionless | Water phase conservation error |
| `gas_mbe` | dimensionless | Gas phase conservation error |
| `solution_gas_drive` | bbl | Solution gas expansion energy |
| `gas_cap_drive` | bbl | Gas cap expansion energy |
| `water_drive` | bbl | Aquifer influx energy |
| `compaction_drive` | bbl | Rock and connate water compression energy |
| `solution_gas_drive_index` | fraction | Fraction from solution gas expansion |
| `gas_cap_drive_index` | fraction | Fraction from gas cap expansion |
| `water_drive_index` | fraction | Fraction from water influx |
| `compaction_drive_index` | fraction | Fraction from compaction |
| `underground_withdrawal` | bbl | Total underground withdrawal (F) |
| `quality` | string | `"excellent"`, `"acceptable"`, `"marginal"`, or `"unacceptable"` |
| `from_step` | int | Start of the analysis interval |
| `to_step` | int | End of the analysis interval |

### Material Balance Error History

You can track how the MBE evolves over time to detect drift. The `material_balance_error_history()` method computes the cumulative MBE from `from_step` to each successive timestep, so you can see whether the error grows, stays constant, or oscillates.

```python
for step, mbe in analyst.material_balance_error_history(from_step=0, interval=20):
    print(f"Step {step}: Total MBE = {mbe.total_mbe:.4%} ({mbe.quality})")
```

The alias `analyst.mbe_history(...)` is also available.

---

## Sweep Efficiency Analysis

The `sweep_efficiency_analysis()` method evaluates how effectively the displacing phase has contacted and displaced oil in the reservoir. It decomposes recovery into the product of volumetric sweep efficiency (what fraction of the original oil was reached) and displacement efficiency (how much oil was removed from the contacted zones).

```python
sweep = analyst.sweep_efficiency_analysis(
    step=-1,
    displacing_phase="water",
    delta_water_saturation_threshold=0.02,
)
print(f"Volumetric sweep: {sweep.volumetric_sweep_efficiency:.2%}")
print(f"Displacement efficiency: {sweep.displacement_efficiency:.2%}")
print(f"Recovery efficiency: {sweep.recovery_efficiency:.2%}")
print(f"Areal sweep: {sweep.areal_sweep_efficiency:.2%}")
print(f"Vertical sweep: {sweep.vertical_sweep_efficiency:.2%}")
print(f"Contacted oil: {sweep.contacted_oil:,.0f} STB")
print(f"Uncontacted oil: {sweep.uncontacted_oil:,.0f} STB")
```

The method determines which cells have been "contacted" by comparing the current displacing phase saturation to the initial saturation. A cell is contacted if the saturation change exceeds the threshold. For gas injection or miscible flooding, set `displacing_phase="gas"`. In miscible floods, cells where the solvent concentration exceeds `solvent_concentration_threshold` are also counted as contacted.

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `step` | -1 | Time step to analyze |
| `displacing_phase` | `"water"` | Phase doing the displacing: `"water"`, `"gas"`, or `"oil"` |
| `delta_water_saturation_threshold` | 0.02 | Minimum water saturation increase to declare contact |
| `delta_gas_saturation_threshold` | 0.01 | Minimum gas saturation increase to declare contact |
| `solvent_concentration_threshold` | 0.01 | Minimum solvent concentration for miscible contact |

### Result Fields

The `SweepEfficiencyAnalysis` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `volumetric_sweep_efficiency` | fraction | Fraction of initial oil contacted |
| `displacement_efficiency` | fraction | Oil removal efficiency in contacted zones |
| `recovery_efficiency` | fraction | Product of volumetric and displacement efficiency |
| `contacted_oil` | STB | Initial oil in contacted zones |
| `uncontacted_oil` | STB | Initial oil in uncontacted zones |
| `areal_sweep_efficiency` | fraction | Fraction of planform area contacted |
| `vertical_sweep_efficiency` | fraction | Saturation-weighted vertical contact fraction |

### Sweep Efficiency History

```python
for step, sweep in analyst.sweep_efficiency_history(
    displacing_phase="water",
    interval=20,
):
    print(f"Step {step}: Sweep={sweep.volumetric_sweep_efficiency:.2%}")
```

---

## Injection Front Analysis

The `injection_front_analysis()` method tracks the spatial position and character of an injection fluid front as it moves through the reservoir. While `sweep_efficiency_analysis()` reports aggregate efficiency scalars, this method returns the full saturation-delta grid and a front centroid so you can track plume migration step by step.

The front is defined as every cell whose displacing-phase saturation has risen by at least the `threshold` above its value in the earliest available state. This is consistent with the contact detection used in sweep efficiency analysis. The method returns a boolean mask of contacted cells, the saturation change per cell, and a saturation-delta-weighted centroid that gives a physically representative centre of the plume.

This method is particularly useful for monitoring CO2 plume migration in storage studies, tracking waterflood front advancement, or validating that injected fluid is reaching the intended parts of the reservoir. By comparing front centroids at successive timesteps, you can estimate front velocity and direction.

```python
front = analyst.injection_front_analysis(step=-1, phase="gas", threshold=0.02)

print(f"Contacted cells: {front.front_cell_count}")
print(f"Pore volume contacted: {front.front_volume_fraction:.2%}")
print(f"Average front saturation: {front.average_front_saturation:.3f}")
print(f"Maximum front saturation: {front.max_front_saturation:.3f}")
print(f"Front centroid (i,j,k): {front.front_centroid}")

# Access the saturation change grid for visualization
delta_grid = front.saturation_delta_grid  # 3D numpy array
contact_mask = front.front_cells  # 3D boolean array
```

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `step` | -1 | Time step to analyze |
| `phase` | `"water"` | Displacing phase to track: `"water"` or `"gas"` |
| `threshold` | 0.02 | Minimum saturation increase to declare a cell as contacted |

### Result Fields

The `InjectionFrontAnalysis` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `phase` | string | Displacing phase tracked |
| `front_cells` | ndarray (bool) | Boolean mask of contacted cells |
| `front_cell_count` | int | Number of contacted cells |
| `front_volume_fraction` | fraction | Pore-volume-weighted fraction of reservoir contacted |
| `average_front_saturation` | fraction | Mean displacing-phase saturation in contacted cells |
| `max_front_saturation` | fraction | Maximum displacing-phase saturation in contacted cells |
| `saturation_delta_grid` | ndarray (float) | Per-cell saturation change from initial conditions |
| `front_centroid` | tuple(float,float,float) | Saturation-delta-weighted centre of the plume in (i,j,k) |

### Injection Front History

```python
for step, front in analyst.injection_front_history(phase="water", threshold=0.02, interval=20):
    print(
        f"Step {step}: {front.front_cell_count} cells contacted, "
        f"centroid at {front.front_centroid}"
    )
```

---

## Well Productivity Analysis

The `productivity_analysis()` method evaluates well performance using actual flow rates and reservoir properties at the perforation intervals. It does not require bottom-hole pressure data. Instead, it computes metrics from production rates, formation permeability, relative mobility, and the skin factor assigned to each well.

```python
prod = analyst.productivity_analysis(step=-1, phase="oil", cells="PROD-1")
print(f"Flow rate: {prod.total_flow_rate:,.0f} STB/day")
print(f"Avg pressure: {prod.average_reservoir_pressure:.0f} psi")
print(f"Skin factor: {prod.skin_factor:.2f}")
print(f"Flow efficiency: {prod.flow_efficiency:.2%}")
print(f"Well index: {prod.well_index:.4f} rb/day/psi")
print(f"Avg mobility: {prod.average_mobility:.4f} cP⁻¹")
```

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `step` | -1 | Time step to analyze |
| `phase` | `"oil"` | Phase to analyze: `"oil"`, `"gas"`, or `"water"` |
| `cells` | `None` | Filter: well name, cell tuple, or region |

### Result Fields

The `ProductivityAnalysis` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `total_flow_rate` | STB/day or SCF/day | Total flow rate across all matched cells |
| `average_reservoir_pressure` | psia | Average pressure at perforation intervals |
| `skin_factor` | dimensionless | Average skin factor across active wells |
| `flow_efficiency` | fraction | Flow efficiency accounting for skin |
| `well_index` | rb/day/psi | Average geometric well index |
| `average_mobility` | cP⁻¹ | Average phase mobility at perforations |

### Productivity History

```python
for step, prod in analyst.productivity_history(phase="oil", cells="PROD-1", interval=10):
    print(f"Step {step}: Rate={prod.total_flow_rate:.0f}, FE={prod.flow_efficiency:.2%}")
```

---

## IPR Method Recommendation

The `recommend_ipr_method()` method analyzes the current reservoir state and suggests the most appropriate inflow performance relationship (IPR) correlation for your wells. It examines average oil and gas saturations and compares reservoir pressure to the bubble point to determine which IPR model best fits the current flow regime.

The method returns one of four IPR methods. It selects `"fetkovich"` for gas-dominated reservoirs (average gas saturation above 0.6), `"linear"` for undersaturated oil reservoirs where pressure remains above the bubble point, `"jones"` for complex multi-phase systems with significant oil and gas saturations, and `"vogel"` as the default for solution gas drive reservoirs with two-phase oil and gas flow.

This is useful when you need to construct an IPR curve for well design or artificial lift optimization and want a data-driven recommendation rather than guessing which correlation to apply.

```python
ipr_method = analyst.recommend_ipr_method(step=-1)
print(f"Recommended IPR method: {ipr_method}")
# Output: "vogel", "linear", "fetkovich", or "jones"
```

| Return Value | Best For |
| --- | --- |
| `"vogel"` | Two-phase oil/gas systems (solution gas drive) |
| `"linear"` | Undersaturated oil (single-phase, above bubble point) |
| `"fetkovich"` | Gas wells and gas-dominated reservoirs |
| `"jones"` | Complex multi-phase systems with significant oil and gas |

---

## Voidage Replacement Ratio

The `voidage_replacement_ratio()` method computes the ratio of injected reservoir volumes to produced reservoir volumes. This is a key metric for pressure maintenance programs. The VRR accounts for formation volume factors at current reservoir conditions to convert between stock tank and reservoir volumes.

$$VRR = \frac{W_i \cdot B_{wi} + G_{gi} \cdot B_{gi}}{N_p \cdot B_o + W_p \cdot B_w + (GOR - R_s) \cdot N_p \cdot B_g}$$

```python
vrr = analyst.voidage_replacement_ratio(step=-1)
print(f"VRR: {vrr:.3f}")
```

Interpretation:

- **VRR > 1.0**: Injection exceeds production, pressure is increasing
- **VRR = 1.0**: Balanced reservoir, pressure is maintained
- **VRR < 1.0**: Production exceeds injection, pressure is declining

The method accepts an optional `cells` parameter for computing VRR for a specific well pattern or region.

Short aliases: `analyst.vrr(step=-1)` and `analyst.VRR(step=-1)`.

### VRR History

```python
for step, vrr_val in analyst.voidage_replacement_ratio_history(interval=10):
    print(f"Step {step}: VRR = {vrr_val:.3f}")
```

The alias `analyst.vrr_history(...)` is also available.

---

## Mobility Ratio

The `mobility_ratio()` method calculates the mobility ratio between the displacing and displaced phases. Mobility is defined as relative permeability divided by viscosity ($\lambda = k_r / \mu$), and the mobility ratio is the ratio of the displacing phase mobility to the displaced phase mobility.

$$M = \frac{\lambda_{displacing}}{\lambda_{displaced}} = \frac{k_{r,displacing} / \mu_{displacing}}{k_{r,displaced} / \mu_{displaced}}$$

```python
M = analyst.mobility_ratio(
    displaced_phase="oil",
    displacing_phase="water",
    step=-1,
)
print(f"Mobility ratio: {M:.3f}")
```

A mobility ratio less than 1.0 indicates a stable displacement (the displacing fluid moves slower than the displaced fluid). A mobility ratio greater than 1.0 indicates an unstable displacement where the displacing fluid tends to finger through the displaced fluid.

| Parameter | Default | Description |
| --- | --- | --- |
| `displaced_phase` | `"oil"` | Phase being displaced: `"oil"` or `"water"` |
| `displacing_phase` | `"water"` | Phase doing the displacing: `"oil"`, `"water"`, or `"gas"` |
| `step` | -1 | Time step to analyze |

The short alias `analyst.mr(...)` is also available.

### Mobility Ratio History

```python
for step, M in analyst.mobility_ratio_history(
    displaced_phase="oil",
    displacing_phase="water",
    interval=10,
):
    print(f"Step {step}: M = {M:.3f}")
```

The alias `analyst.mr_history(...)` is also available.

---

## Decline Curve Analysis

Decline curve analysis (DCA) fits production data to standard decline models to characterize production trends and forecast future behavior. The analyst supports three decline types: exponential, harmonic, and hyperbolic.

### Fitting Decline Curves

The `decline_curve_analysis()` method fits a single decline model to the production history:

```python
result = analyst.decline_curve_analysis(
    from_step=50,  # Skip early transient
    to_step=-1,
    phase="oil",
    decline_type="exponential",
)
print(f"Initial rate: {result.initial_rate:,.0f} STB/day")
print(f"Decline rate: {result.decline_rate_per_timestep:.6f}")
print(f"b-factor: {result.b_factor:.3f}")
print(f"R-squared: {result.r_squared:.4f}")
```

The `DeclineCurveResult` result class contains:

| Field | Unit | Description |
| --- | --- | --- |
| `decline_type` | string | `"exponential"`, `"hyperbolic"`, or `"harmonic"` |
| `initial_rate` | STB/day or SCF/day | Fitted initial production rate |
| `decline_rate_per_timestep` | fraction/step | Decline rate per simulation time step |
| `b_factor` | dimensionless | Hyperbolic exponent (0 for exponential, 1 for harmonic) |
| `r_squared` | dimensionless | Goodness of fit (0 to 1) |
| `phase` | string | Phase analyzed |
| `error` | string or None | Error message if fitting failed |
| `steps` | list[int] | Time steps used in the fit |
| `actual_rates` | list[float] | Historical rates |
| `predicted_rates` | list[float] | Fitted rates |

### Recommending the Best Model

The `recommend_decline_model()` method fits all three decline types and recommends the best one based on statistical fit quality and physical reasonableness:

```python
best_model, all_results = analyst.recommend_decline_model(
    from_step=50,
    to_step=-1,
    phase="oil",
    max_decline_per_year=2.0,
)
print(f"Recommended: {best_model}")

# Access individual results
for name, result in all_results.items():
    print(f"  {name}: R2={result.r_squared:.4f}, qi={result.initial_rate:.0f}")
```

The method returns a tuple of (best model name, dictionary of all results). The selection criteria include R-squared, physical reasonableness of the b-factor, and whether the decline rate exceeds `max_decline_per_year`.

The short alias `analyst.dca(...)` is equivalent to `analyst.decline_curve_analysis(...)`.

### Forecasting Production

The `forecast_production()` method extrapolates future rates from a fitted decline curve:

```python
result = analyst.decline_curve_analysis(
    from_step=50, to_step=-1, phase="oil", decline_type="exponential"
)

# Forecast 200 steps into the future
forecast = analyst.forecast_production(
    decline_result=result,
    steps=200,
    economic_limit=10.0,  # Stop when rate drops below 10 STB/day
)

for step, rate in forecast:
    print(f"Step {step}: {rate:.0f} STB/day (forecast)")
```

The forecast uses the fitted decline equations:

- **Exponential**: $q(t) = q_i \cdot e^{-D_i \cdot t}$
- **Harmonic**: $q(t) = q_i / (1 + D_i \cdot t)$
- **Hyperbolic**: $q(t) = q_i / (1 + b \cdot D_i \cdot t)^{1/b}$

The `economic_limit` parameter stops the forecast when the predicted rate drops below the specified value.

### Estimated Ultimate Recovery

The `estimate_economic_ultimate_recovery()` method calculates the total cumulative production expected over the economic life of the well or reservoir:

```python
eur = analyst.estimate_economic_ultimate_recovery(
    decline_result=result,
    forecast_steps=500,
    economic_limit=5.0,  # STB/day
)
print(f"EUR: {eur:,.0f} STB")
```

EUR is computed using analytical integration of the decline curve equations rather than numerical summation, giving exact results regardless of time step size.

---

## Method Aliases Quick Reference

Many analyst methods have shorter aliases for convenience. The table below lists every alias and its full method name. You can use either form interchangeably.

| Alias | Full Method Name | Returns |
|---|---|---|
| `stoiip` | `stock_tank_oil_initially_in_place` | STOIIP in STB (property) |
| `stgiip` | `stock_tank_gas_initially_in_place` | STGIIP in SCF (property) |
| `No` | `cumulative_oil_produced` | Cumulative oil in STB (property) |
| `Ng` | `cumulative_gas_produced` | Cumulative gas in SCF (property) |
| `Nw` | `cumulative_water_produced` | Cumulative water in STB (property) |
| `mbal(...)` | `material_balance(...)` | Material balance analysis result |
| `mbe(...)` | `material_balance_error(...)` | Material balance error result |
| `vrr(...)` / `VRR(...)` | `voidage_replacement_ratio(...)` | VRR as float |
| `dca(...)` | `decline_curve_analysis(...)` | Decline curve result |
| `mr(...)` | `mobility_ratio(...)` | Mobility ratio result |
| `mbal_history(...)` | `material_balance_history(...)` | Generator of (step, result) |
| `mbe_history(...)` | `material_balance_error_history(...)` | Generator of (step, result) |
| `vrr_history(...)` | `voidage_replacement_ratio_history(...)` | Generator of (step, VRR) |
| `mr_history(...)` | `mobility_ratio_history(...)` | Generator of (step, result) |

```python
# These are equivalent:
analyst.material_balance(step=-1)
analyst.mbal(step=-1)

# So are these:
analyst.voidage_replacement_ratio(step=-1)
analyst.vrr(step=-1)
analyst.VRR(step=-1)
```

---

## Complete Workflow

A typical post-simulation analysis workflow combines several of these methods:

```python
import bores
from bores.analyses import ModelAnalyst
from bores.states import ModelState
from bores.stores import ZarrStore
from bores.streams import StateStream

# Run and save
store = ZarrStore("simulation.zarr")
with StateStream(
    states=bores.run(model, config),
    store=store,
    background_io=True,
) as stream:
    stream.consume()

# Create analyst from saved states
with store(mode="r") as s:
    analyst = ModelAnalyst(s.load(ModelState))

# Summary
print(f"STOIIP: {analyst.stoiip:,.0f} STB")
print(f"Recovery factor: {analyst.oil_recovery_factor:.2%}")

# Material balance
mbal = analyst.mbal(step=-1)
print(f"Primary drive: solution gas = {mbal.solution_gas_drive_index:.1%}")

# Sweep efficiency
sweep = analyst.sweep_efficiency_analysis(step=-1)
print(f"Volumetric sweep: {sweep.volumetric_sweep_efficiency:.1%}")

# Well productivity
prod = analyst.productivity_analysis(step=-1, cells="PROD-1")
print(f"Well rate: {prod.total_flow_rate:.0f} STB/day")

# VRR (if injection is active)
print(f"VRR: {analyst.vrr(step=-1):.3f}")

# Decline curve and forecast
best, results = analyst.recommend_decline_model(from_step=50, phase="oil")
forecast = analyst.forecast_production(results[best], steps=200, economic_limit=10.0)
eur = analyst.estimate_economic_ultimate_recovery(results[best], forecast_steps=500)
print(f"Best decline model: {best}, EUR: {eur:,.0f} STB")
```
