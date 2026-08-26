# Running and Monitoring Simulations

After building your reservoir model and assembling your configuration, you are ready to run a simulation. BORES provides two closely related entry points for this: `bores.run()` executes the simulation and yields results, while `bores.monitor()` wraps that same execution with a live progress display and diagnostic statistics. Both functions produce the same `ModelState` output at the same cadence - the choice between them is purely about how much visibility you want during the run.

This separation is intentional. In batch or headless environments (HPC clusters, automated workflows, overnight runs), you want `bores.run()` with no terminal output overhead. During development and debugging, you want `bores.monitor()` with its live dashboard showing solver convergence, per-well rates, and pressure evolution. You can switch between them by changing a single function name - everything else stays the same.

Both functions are generator-based. They yield `ModelState` objects rather than returning a list of results. This means memory usage is bounded even for long simulations: BORES yields one state, you process it, and then the next step begins. If you want to retain history, you collect states into a list yourself. If you are running a 10,000-step simulation and only care about the final result, you iterate and discard intermediate states.

---

## Quick Start

The minimum viable simulation run looks like this:

```python
import bores

model = bores.BlackOil.from_file("reservoir.h5")
config = bores.Config.from_file("simulation.yaml")

for state in bores.run(model, config):
    print(
        f"Step {state.step}: time = {state.time:.1f} s, "
        f"avg pressure = {state.average_pressure:.1f} psi"
    )
```

For interactive work where you want a live display in your terminal:

```python
import bores

model = bores.BlackOil.from_file("reservoir.h5")
config = bores.Config.from_file("simulation.yaml")

for state in bores.monitor(model, config):
    # Your per-step analysis here
    pass
```

Both loops behave identically in terms of what `state` contains and when it is yielded. The only difference is the terminal output.

---

## `bores.run()`: Core Simulation Execution

### Function Signature

```python
def run(
    input: Union[BlackOil[ThreeDimensions], Run],
    config: Optional[Config] = None,
    *,
    on_step_rejected: Optional[StepCallback] = None,
    on_step_accepted: Optional[StepCallback] = None,
) -> Generator[ModelState[ThreeDimensions], None, None]:
```

### Parameters

**`input`** accepts either:

- A `BlackOil[ThreeDimensions]` - the static model you built. When you pass a `BlackOil`, you must also supply `config`.
- A `Run` - a pre-packaged specification that bundles model and config together. See the [Run Specification](#run-specification) section below.

**`config`** is the `Config` object containing your simulation parameters, scheme choice, wells, PVT tables, and output frequency.

- Required when `input` is a `BlackOil`.
- Optional when `input` is a `Run` - the `Run`'s own config is used by default, but you can override it by passing a new `config` here.

**`on_step_rejected`** is an optional callback invoked whenever a proposed time step fails due to convergence problems or saturation change violations. BORES will automatically retry the step with a smaller time step size, but this callback gives you visibility into rejection events. It receives three arguments: the `StepResult` (containing residuals and failure messages), the rejected step size in seconds, and the total elapsed simulation time in seconds.

```python
def handle_rejection(result, step_size, elapsed_time):
    print(f"Step rejected at t={elapsed_time:.1f}s, step size {step_size:.3f}s: {result.message}")


for state in bores.run(model, config, on_step_rejected=handle_rejection):
    pass
```

**`on_step_accepted`** is the mirror callback, invoked after each successfully completed step. It receives the same three arguments. Useful for collecting lightweight per-step metrics without storing full `ModelState` objects.

```python
newton_counts = []


def on_accepted(result, step_size, elapsed_time):
    newton_counts.append(result.timer_kwargs.get("newton_iterations", -1))


for state in bores.run(model, config, on_step_accepted=on_accepted):
    pass
```

### What `ModelState` Contains

Each yielded `ModelState` is a complete snapshot of the simulation at that point in time. The key fields you will use most often are:

- **`state.step`** - The accepted step index (1-based).
- **`state.time`** - Total elapsed simulation time in seconds.
- **`state.step_size`** - The time step size used for this step in seconds.
- **`state.average_pressure`**, **`state.min_pressure`**, **`state.max_pressure`** - Grid-level pressure statistics in psi.
- **`state.average_water_saturation`** - Mean water saturation across all cells.
- **`state.model.fluid_properties`** - Full grid arrays for pressure, water saturation, oil saturation, and gas saturation.
- **`state.model.rock_properties`** - Porosity, absolute permeability, and residual saturations.
- **`state.wells`** - Well configuration with current schedule applied.
- **`state.production_rates`** and **`state.injection_rates`** - `SparseTensor` objects holding surface-condition rates by well location (STB/day for oil and water, SCF/day for gas).
- **`state.production_bhps`** and **`state.injection_bhps`** - Bottom-hole pressures for each well in psi.
- **`state.timer_state`** - Serializable timer state for checkpointing and resumption.

See the [ModelState Documentation](../advanced/states-streams.md) for complete field access patterns.

### Output Frequency

By default, `bores.run()` yields a `ModelState` after every accepted step (`output_frequency=1`). For long simulations, this produces a lot of data. You can reduce output by setting `output_frequency` in your `Config`:

```python
# Yield every 100 accepted steps - good for 10,000+ step simulations
config = bores.Config(
    ...,
    output_frequency=100,
)
```

BORES always yields the initial state (step 0) and the final state, regardless of `output_frequency`, so you never miss the start or end of the simulation.

### Step Rejection and Retry

When a time step fails - because pressures became unphysical, saturation changes exceeded their limits, or Newton iteration did not converge - BORES does not abort. Instead, it:

1. Calls `on_step_rejected` if you provided one.
2. Reduces the step size adaptively (the reduction strategy depends on how badly the step failed).
3. Retries the same time interval with the smaller step.

This continues until the step succeeds, or until the step size cannot be reduced any further (at which point `SimulationError` is raised). The adaptive step control is transparent to you as the caller - from the outside, you simply see fewer states per unit time during difficult periods.

### Example: Collecting Production History

```python
import bores

model = bores.BlackOil.from_file("model.h5")
config = bores.Config.from_file("config.yaml")

production_history = {}

for state in bores.run(model, config):
    for well_name, well in state.wells.items():
        if well_name not in production_history:
            production_history[well_name] = {"time": [], "oil": [], "water": [], "gas": []}

        loc = well.location
        production_history[well_name]["time"].append(state.time)
        production_history[well_name]["oil"].append(state.production_rates.oil.get(loc, 0.0))
        production_history[well_name]["water"].append(state.production_rates.water.get(loc, 0.0))
        production_history[well_name]["gas"].append(state.production_rates.gas.get(loc, 0.0))

for well_name, history in production_history.items():
    final_oil = history["oil"][-1] if history["oil"] else 0.0
    print(f"{well_name}: {final_oil:.1f} STB/day at end")
```

---

## Run Specification

The `Run` class bundles a reservoir model and configuration into a single named object. This is useful for organizing multiple scenarios, serializing run definitions to disk, or packaging simulation inputs for distribution.

### Creating a Run

```python
from bores import BlackOil, Config, Run

model = BlackOil.from_file("path/to/3d_model.h5")
config = Config.from_file("path/to/simulation_config.yaml")

run_spec = Run(
    model=model,
    config=config,
    name="Primary Depletion - Base Case",
    description="30-year primary depletion from 3,000 psi with no injection support",
    tags=("baseline", "primary-depletion"),
)
```

### Executing a Run

The `Run` class is callable and iterable - you can execute it in several equivalent ways:

```python
# Direct iteration
for state in run_spec:
    process(state)

# Calling it as a function
for state in run_spec():
    process(state)

# Passing to bores.run() - equivalent, and allows config override
for state in bores.run(run_spec):
    process(state)

# Override the config for a sensitivity case
sensitivity_config = bores.Config(scheme="full-sequential-implicit", ...)
for state in bores.run(run_spec, config=sensitivity_config):
    process(state)
```

### Loading from Files

```python
from bores import Run

run_spec = Run.from_files(
    model_path="path/to/model.h5",
    config_path="path/to/config.yaml",
    pvt_tables_path="path/to/pvt_tables.h5",  # Optional
)

for state in run_spec:
    process(state)
```

The `Run.from_files()` method handles loading the PVT tables and attaching them to the config automatically, which simplifies the setup code when all your simulation inputs are stored as files.

---

## `bores.monitor()`: Live Monitoring and Diagnostics

`bores.monitor()` wraps the simulation execution with a live terminal display and a `RunStats` accumulator. It yields the exact same `ModelState` objects as `bores.run()`, so you can switch to monitoring without changing any of your downstream processing code.

### Function Signature

```python
def monitor(
    input: Union[BlackOil, Run, Iterable[ModelState]],
    config: Optional[Config] = None,
    *,
    monitor: Optional[MonitorConfig] = None,
    on_step_rejected: Optional[StepCallback] = None,
    on_step_accepted: Optional[StepCallback] = None,
    return_stats: bool = False,
) -> Generator[ModelState | Tuple[ModelState, RunStats], None, None]:
```

The `input` parameter accepts one additional type compared to `bores.run()`: an `Iterable[ModelState]`. This allows you to run the monitor over a stream of pre-loaded states from a saved run, getting diagnostics without re-running the simulation. See the [Post-Processing Example](#example-post-processing-saved-runs) below.

### Enabling Diagnostic Statistics

Set `return_stats=True` to receive a `(ModelState, RunStats)` tuple at each yield. The `RunStats` object accumulates in-place throughout the loop and remains valid after the loop completes:

```python
import bores

model = bores.BlackOil.from_file("reservoir.h5")
config = bores.Config.from_file("config.yaml")

for state, stats in bores.monitor(model, config, return_stats=True):
    # stats is updated in-place at every step
    pass

# Inspect after the loop
print(f"Accepted steps:      {stats.accepted_steps}")
print(f"Rejected steps:      {stats.rejected_steps}")
print(f"Total wall time:     {stats.total_wall_time:.2f} s")
print(f"Avg step time:       {stats.average_step_wall_ms:.2f} ms")
print(f"p95 step time:       {stats.get_percentile_wall_time_ms(95):.2f} ms")
print(f"Avg Newton iters:    {stats.average_newton_iterations:.2f}")

# Print the full summary table
print(stats.summary_table())
```

### `MonitorConfig`: Controlling the Display

`MonitorConfig` controls which display backends are active and how they behave:

```python
from bores import MonitorConfig

monitor_cfg = MonitorConfig(
    use_rich=True,  # Live Rich panel (default: True)
    use_tqdm=True,  # tqdm progress bar (default: False)
    refresh_interval=1,  # Update display every N accepted steps
    extended_every=10,  # Show p95 wall time and avg Newton every N steps
    show_wells=True,  # Include per-well rates table in display
    color_theme="dark",  # "dark" or "light"
)

for state in bores.monitor(model, config, monitor=monitor_cfg):
    pass
```

**`use_rich`** enables the live Rich panel. This is a compact two-column display that shows reservoir physics on the left (pressure statistics, saturations) and solver diagnostics on the right (step size, wall time, Newton iterations, CFL number, rejected/accepted step counts). The panel updates in-place and is preserved in your terminal scroll-back history when the run ends.

**`use_tqdm`** adds a standard tqdm progress bar below the Rich panel. The bar tracks simulation-time progress from 0 to 100% and shows per-step timing in the postfix. This is useful when you want a clean progress summary in logs or CI output.

**`extended_every`** controls how often the Rich panel includes extended performance diagnostics (p95 step wall time and average Newton iterations). Setting it to 0 disables extended stats entirely, which gives a slightly cleaner display during fast-running simulations.

**`color_theme`** applies to the Rich panel. The `"dark"` theme uses charcoal background with amber accents. The `"light"` theme uses off-white with navy accents. Choose based on your terminal background.

### `RunStats`: Diagnostic Accumulator

`RunStats` is the statistics object returned alongside states when `return_stats=True`. It accumulates data after every accepted output step and provides both live-readable properties and a post-run summary.

Key properties and methods:

- **`stats.accepted_steps`** - Number of steps that succeeded.
- **`stats.rejected_steps`** - Number of step rejections (reduces when step size is reduced and retried).
- **`stats.total_wall_time`** - Cumulative wall clock time in seconds.
- **`stats.average_step_wall_ms`** - Mean time per accepted step in milliseconds.
- **`stats.get_percentile_wall_time_ms(95)`** - 95th percentile step wall time - useful for identifying slow outlier steps.
- **`stats.average_newton_iterations`** - Mean Newton iterations per step (only counts steps that used Newton iteration).
- **`stats.steps`** - A list of `StepDiagnostics` objects, one per output step, containing the full scalar snapshot for each step.
- **`stats.summary_table()`** - Returns a formatted Rich `Table` with the complete run summary.
- **`stats.summary()`** - Returns a plain-text summary string suitable for logging.

The `stats` object is the same instance throughout the loop. If you want to inspect intermediate statistics (say, after the first 1000 steps), you can read from it inside the loop without any special handling.

### Example: Full Monitoring Workflow

```python
import bores

model = bores.BlackOil.from_file("reservoir.h5")
config = bores.Config.from_file("config.yaml")

monitor_cfg = bores.MonitorConfig(
    use_rich=True,
    use_tqdm=True,
    show_wells=True,
    refresh_interval=1,
)

pressure_history = []
hysteresis_state = []

for state, stats in bores.monitor(
    model,
    config,
    monitor=monitor_cfg,
    return_stats=True,
):
    pressure_history.append(state.average_pressure)
    hysteresis_state.append(state.average_water_saturation)

# Post-simulation analysis
print("\n" + "=" * 60)
print(stats.summary())

final = pressure_history[-1]
print(f"Final average pressure: {final:.2f} psi")
print(f"Rejection rate: {stats.rejected_steps}/{stats.accepted_steps + stats.rejected_steps}")
```

### Example: Post-Processing Saved Runs

`bores.monitor()` also accepts an iterable of `ModelState` objects, which lets you run the diagnostic display over previously saved simulation states without re-running the simulation:

```python
import bores

# Load states from disk - your loading function here
states = load_states_from_disk("simulation_states.h5")

# Collect diagnostics without re-running
for state, stats in bores.monitor(
    states,
    monitor=bores.MonitorConfig(use_rich=True),
    return_stats=True,
):
    pass

print(stats.summary_table())
```

---

## Best Practices

### Output Frequency vs. Memory Usage

Yielding every step generates large amounts of state data. For long simulations, increase `output_frequency` to match your actual analysis needs:

```python
# For a 10,000-step simulation where you only need daily snapshots,
# set output_frequency to match your timestep-to-time ratio
config = bores.Config(
    ...,
    output_frequency=100,
)
```

Alternatively, if you only need specific aggregates and not full grid snapshots, use the `on_step_accepted` callback to collect them without storing `ModelState` objects at all.

### Use Callbacks for Lightweight Metrics

If your analysis requires per-step data (not just output-frequency data), callbacks are more efficient than storing states:

```python
metrics = {"rejections": 0, "max_newton": 0, "step_times": []}


def on_rejected(result, step_size, elapsed):
    metrics["rejections"] += 1


def on_accepted(result, step_size, elapsed):
    ni = result.timer_kwargs.get("newton_iterations", 0)
    metrics["max_newton"] = max(metrics["max_newton"], ni)


for state in bores.run(model, config, on_step_rejected=on_rejected, on_step_accepted=on_accepted):
    pass
```

### Accessing Well Rates via SparseTensor

Production and injection rates are stored as `SparseTensor` objects. A `SparseTensor` is a dictionary-like container keyed by cell index tuples. Access it like this:

```python
import numpy as np

for state in bores.run(model, config):
    # Compute total production rates across all wells (sum all surface rates)
    total_oil_production_ft3_day = state.production_rates.oil.sum()
    total_water_production_ft3_day = state.production_rates.water.sum()

    # Convert to surface conditions using formation volume factors
    # Production rates are negative by convention, so take absolute value
    total_oil_production_stb = (
        abs(total_oil_production_ft3_day)
        * 5.615
        / state.production_formation_volume_factors.oil.mean()
    )

    # Compute statistics on production across all active cells
    oil_prod_grid = state.production_rates.oil.array()
    producing_cells = np.abs(oil_prod_grid[oil_prod_grid != 0])
    if producing_cells.size > 0:
        mean_rate_ft3_day = producing_cells.mean()
        max_rate_ft3_day = producing_cells.max()

    print(
        f"Step {state.step} ({state.time_in_days:.1f} days): "
        f"Oil production = {total_oil_production_stb:.1f} STB/day, "
        f"Water production = {abs(total_water_production_ft3_day):.1f} ft³/day"
    )
```

### Resuming from a Checkpoint

BORES supports resuming a simulation from a previously yielded state. The `timer_state` field on each `ModelState` contains a serializable snapshot of the timer that you can use to create a new timer starting from that point:

```python
import bores

# First run - stop after reaching a target time
states = []
for state in bores.run(model, config):
    states.append(state)
    if state.time > 1e6:
        break

# Resume from the last yielded state
last_state = states[-1]
new_model = last_state.model
new_timer = bores.Timer.from_state(last_state.timer_state)
resumed_config = config.new(timer=new_timer)

for state in bores.run(new_model, resumed_config):
    process(state)
```

---

## Troubleshooting

### Simulation Runs Slowly

High step rejection rates are the most common cause of slow simulations. Check with a callback:

```python
rejection_count = 0


def track(result, step_size, elapsed):
    global rejection_count
    rejection_count += 1


for state in bores.run(model, config, on_step_rejected=track):
    pass

print(f"Total rejections: {rejection_count}")
```

If the rejection count is high relative to accepted steps, consider:

- Reducing `cfl_safety_margin` if the scheme is too aggressive with step sizes.
- Switching to a more stable scheme (Sequential Implicit or Full Sequential Implicit) for stiff problems with large mobility contrasts.
- Adding local grid refinement near wells if rejections are triggered by near-well pressure spikes.

### Out of Memory During Long Simulations

Increase `output_frequency` to reduce the number of yielded states. For the most memory-efficient runs, process states in streaming fashion and do not store them:

```python
config = bores.Config(..., output_frequency=1000)

# Stream without storing
for state in bores.run(model, config):
    write_to_disk(state)  # Write and discard
```

### Convergence Failures

Use `RunStats` to diagnose:

```python
for state, stats in bores.monitor(model, config, return_stats=True):
    pass

print(f"Avg Newton iterations: {stats.average_newton_iterations:.2f}")
print(f"p95 step time:         {stats.get_percentile_wall_time_ms(95):.1f} ms")
```

High average Newton iterations (above 8-10 per step) indicate that pressure-saturation coupling is tight and the solver is struggling. In this case, try switching from `sequential-implicit` to `full-sequential-implicit`, which adds outer iteration to enforce coupling consistency and typically converges in fewer total Newton iterations even though each step does more work.
