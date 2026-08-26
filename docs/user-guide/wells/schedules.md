# Well Schedules

## Overview

Real reservoir operations change over time. Wells are drilled and brought on-stream at different dates, production rates are adjusted based on facility constraints, injectors are converted from producers after waterflood initiation, and wells are shut in for workovers or when they reach economic limits. BORES models these time-dependent changes through the well scheduling system.

The scheduling system is built on two primitives: **predicates** (conditions that determine when an event fires) and **actions** (modifications that are applied to the well when the event fires). These are combined into `WellEvent` objects, collected into `WellSchedule` containers, and then organized by well name in `WellSchedules`. The simulator checks the schedule at every timestep and applies any events whose predicates evaluate to true.

This event-driven approach is more flexible than a simple time-step table because predicates can depend on simulation state (pressure, saturation, production rates), not just time. This lets you model conditional operations like "shut in the producer when water cut exceeds 95%" or "increase injection rate when reservoir pressure drops below 2500 psi."

---

## `WellEvent`: Predicate + Action

A `WellEvent` combines a predicate (when to trigger) with an action (what to do). At each timestep, the simulator calls the predicate with the well and the current model state. If the predicate returns `True`, the action is executed.

```python
import bores

event = bores.WellEvent(
    predicate=bores.time_predicate(time_step=50),
    action=bores.update_well(
        control=bores.ProducerRateControl(
            primary_phase=bores.FluidPhase.OIL,
            primary_control=bores.AdaptiveRateControl(
                target_rate=-200.0,
                target_phase="oil",
                bhp_limit=800.0,
            ),
            secondary_clamp=bores.ProductionClamp(),
        ),
    ),
)
```

This event triggers at timestep 50 and reduces the oil production target from whatever it was before to 200 STB/day.

---

## Built-in Predicates

### `time_predicate`

The most common predicate triggers at a specific timestep or simulation time:

```python
import bores

# Trigger at timestep 100
pred = bores.time_predicate(time_step=100)

# Trigger at simulation time 365.0 days
pred = bores.time_predicate(time=bores.Time(days=365.0))
```

You can specify either `time_step` (integer, zero-based timestep counter) or `time` (float, simulation time in seconds). If the simulation time exceeds the specified value during a timestep, the predicate fires.

### Custom Predicates

You can write any function that accepts a well and a model state and returns a boolean. Register it with the `@event_predicate` decorator so it can be serialized:

```python
import bores


@bores.event_predicate
def high_water_cut(well, state):
    """Trigger when water cut exceeds 90%."""
    water_rate = abs(state.well_results.get(well.name, {}).get("water_rate", 0.0))
    oil_rate = abs(state.well_results.get(well.name, {}).get("oil_rate", 1e-10))
    water_cut = water_rate / (water_rate + oil_rate)
    return water_cut > 0.90


@bores.event_predicate
def low_pressure(well, state):
    """Trigger when average pressure drops below 2000 psi."""
    avg_pressure = state.pressure_grid.mean()
    return avg_pressure < 2000.0
```

Registered predicates are serializable, meaning they can be saved and loaded as part of a `Config`. This is important for reproducibility.

---

## Built-in Actions

### `update_well`

The `update_well` function creates an action that modifies well properties. You can change the control, skin factor, active status, or fluid properties:

```python
import bores

# Change control strategy
change_rate = bores.update_well(
    control=bores.ProducerRateControl(
        primary_phase=bores.FluidPhase.OIL,
        primary_control=bores.AdaptiveRateControl(
            target_rate=-100.0,
            target_phase="oil",
            bhp_limit=500.0,
        ),
        secondary_clamp=bores.ProductionClamp(),
    ),
)

# Shut in the well
shut_in_well = bores.update_well(is_active=False)

# Re-open the well
open_well = bores.update_well(is_active=True)

# Change skin factor (after workover)
workover_done = bores.update_well(skin_factor=-1.0)

# Change injected fluid
switch_fluid = bores.update_well(
    injected_fluid=bores.InjectedFluid(
        name="CO2",
        phase=bores.FluidPhase.GAS,
        specific_gravity=1.52,
        molecular_weight=44.01,
        density=35.0,
        viscosity=0.05,
    ),
)
```

You can pass any combination of parameters. Only the specified parameters are updated; all others remain unchanged.

### Custom Actions

Like predicates, you can write custom action functions and register them:

```python
import bores


@bores.event_action
def reduce_rate_by_half(well, state):
    """Reduce the well's target rate by 50%."""
    current_control = well.control
    if hasattr(current_control, "primary_control"):
        current_rate = current_control.primary_control.target_rate
        new_control = bores.ProducerRateControl(
            primary_phase=current_control.primary_phase,
            primary_control=bores.AdaptiveRateControl(
                target_rate=current_rate * 0.5,
                target_phase=current_control.primary_control.target_phase,
                bhp_limit=current_control.primary_control.bhp_limit,
            ),
            secondary_clamp=current_control.secondary_clamp,
        )
        well.control = new_control
```

---

## Building Schedules

### Single Well Schedule

A `WellSchedule` collects events for a single well. Events are identified by string IDs:

```python
import bores

schedule = bores.WellSchedule()

# Add events using `add(id, event)`
schedule.add(
    "rate_reduction",
    bores.WellEvent(
        predicate=bores.time_predicate(time_step=50),
        action=bores.update_well(
            control=bores.ProducerRateControl(
                primary_phase=bores.FluidPhase.OIL,
                primary_control=bores.AdaptiveRateControl(
                    target_rate=-200.0,
                    target_phase="oil",
                    bhp_limit=800.0,
                ),
                secondary_clamp=bores.ProductionClamp(),
            ),
        ),
    ),
)

# Or use like a dictionary
schedule["shut_in"] = bores.WellEvent(
    predicate=bores.time_predicate(time=730.0),  # After 2 years
    action=bores.update_well(is_active=False),
)
```

### Multi-Well Schedules

A `WellSchedules` object organizes schedules for all wells by name:

```python
import bores

schedules = bores.WellSchedules()

# Add schedule for a production well
prod_schedule = bores.WellSchedule()
prod_schedule.add(
    "reduce_rate",
    bores.WellEvent(
        predicate=bores.time_predicate(time_step=100),
        action=bores.update_well(
            control=bores.ProducerRateControl(
                primary_phase=bores.FluidPhase.OIL,
                primary_control=bores.AdaptiveRateControl(
                    target_rate=-200.0,
                    target_phase="oil",
                    bhp_limit=800.0,
                ),
                secondary_clamp=bores.ProductionClamp(),
            ),
        ),
    ),
)
schedules["PROD-1"] = prod_schedule

# Add schedule for an injection well
inj_schedule = bores.WellSchedule()
inj_schedule.add(
    "increase_rate",
    bores.WellEvent(
        predicate=bores.time_predicate(time_step=100),
        action=bores.update_well(
            control=bores.RateControl(
                target_rate=1200.0,
                bhp_limit=5000.0,
            ),
        ),
    ),
)
schedules.add("INJ-1", inj_schedule)  # Same as doing `schedules["INJ-1"] = inj_schedule`
```

Pass the schedules to the `Config`:

```python
config = bores.Config(
    timer=timer,
    wells=wells,
    well_schedules=schedules,
    rock_fluid_tables=rock_fluid_tables,
    scheme="impes",
)
```

---

## Composing Predicates

The `EventPredicate` class supports logical composition using Python operators:

```python
import bores

# Create individual predicates
after_year_one = bores.EventPredicate.from_func(bores.time_predicate(time=365.0))

# Combine with AND
combined = bores.EventPredicate.all_of(
    bores.time_predicate(time=365.0),
    low_pressure,  # Custom predicate from earlier
)

# Combine with OR
either = bores.EventPredicate.any_of(
    high_water_cut,
    bores.time_predicate(time=1825.0),  # 5 years
)
```

Composed predicates are fully serializable.

---

## Composing Actions

Similarly, you can chain multiple actions together:

```python
import bores

# Execute both actions when the event triggers
combined_action = bores.EventAction.sequence(
    bores.update_well(skin_factor=-2.0),  # Workover
    bores.update_well(
        control=bores.ProducerRateControl(
            primary_phase=bores.FluidPhase.OIL,
            primary_control=bores.AdaptiveRateControl(
                target_rate=-800.0,
                target_phase="oil",
                bhp_limit=1200.0,
            ),
            secondary_clamp=bores.ProductionClamp(),
        ),
    ),
)
```

---

## Removing Events

You can remove events from a schedule by their ID:

```python
schedule.remove("rate_reduction")
```

This is useful for dynamically modifying schedules between simulation restarts.

!!! tip "Event Naming Convention"

    Use descriptive event IDs that indicate what happens and when. Examples: "reduce_rate_year_2", "shut_in_high_wc", "convert_to_injector_phase_2". Good naming makes it easy to find and modify events later.
