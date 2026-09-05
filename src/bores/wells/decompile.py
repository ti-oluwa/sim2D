"""Turns a resolved compiled well structure back into rich, readable objects."""

import math

from bores.types import FluidPhase, UnitSystem
from bores.wells.base import Wells
from bores.wells.compile import (
    UNSET_INT,
    CompiledWellControls,
    CompiledWellSystem,
    EconomicQuantityTag,
    FluidPhaseTag,
    InjectorControlModeTag,
    LimitKind,
    ProducerControlModeTag,
    RateQuantityTag,
    WellKind,
    WorkoverActionTag,
)
from bores.wells.controls import (
    BHPLimit,
    EconomicLimit,
    EconomicQuantity,
    InjectorControl,
    InjectorControlMode,
    Limit,
    ProducerControl,
    ProducerControlMode,
    RateLimit,
    RateQuantity,
    THPLimit,
    WellControl,
    WorkoverAction,
)
from bores.wells.resolution.compile import CompiledWellResolution
from bores.wells.states import PerforationState, PhaseValues, WellsStates, WellState

__all__ = ["build_wells_states"]

PRODUCER_MODE_FROM_TAG = {
    ProducerControlModeTag.OIL_RATE: ProducerControlMode.OIL_RATE,
    ProducerControlModeTag.WATER_RATE: ProducerControlMode.WATER_RATE,
    ProducerControlModeTag.GAS_RATE: ProducerControlMode.GAS_RATE,
    ProducerControlModeTag.LIQUID_RATE: ProducerControlMode.LIQUID_RATE,
    ProducerControlModeTag.RESERVOIR_VOLUME_RATE: ProducerControlMode.RESERVOIR_VOLUME_RATE,
    ProducerControlModeTag.BHP: ProducerControlMode.BHP,
    ProducerControlModeTag.THP: ProducerControlMode.THP,
    ProducerControlModeTag.GROUP: ProducerControlMode.GROUP,
}
INJECTOR_MODE_FROM_TAG = {
    InjectorControlModeTag.RATE: InjectorControlMode.RATE,
    InjectorControlModeTag.RESERVOIR_VOLUME_RATE: InjectorControlMode.RESERVOIR_VOLUME_RATE,
    InjectorControlModeTag.BHP: InjectorControlMode.BHP,
    InjectorControlModeTag.THP: InjectorControlMode.THP,
    InjectorControlModeTag.GROUP: InjectorControlMode.GROUP,
}
RATE_QUANTITY_FROM_TAG = {
    RateQuantityTag.OIL: RateQuantity.OIL,
    RateQuantityTag.WATER: RateQuantity.WATER,
    RateQuantityTag.GAS: RateQuantity.GAS,
    RateQuantityTag.LIQUID: RateQuantity.LIQUID,
    RateQuantityTag.RESERVOIR: RateQuantity.RESERVOIR,
}
ECONOMIC_QUANTITY_FROM_TAG = {
    EconomicQuantityTag.WATER_CUT: EconomicQuantity.WATER_CUT,
    EconomicQuantityTag.GOR: EconomicQuantity.GOR,
    EconomicQuantityTag.WATER_GAS_RATIO: EconomicQuantity.WATER_GAS_RATIO,
    EconomicQuantityTag.OIL_RATE: EconomicQuantity.OIL_RATE,
    EconomicQuantityTag.GAS_RATE: EconomicQuantity.GAS_RATE,
}
WORKOVER_ACTION_FROM_TAG = {
    WorkoverActionTag.WELL: WorkoverAction.WELL,
    WorkoverActionTag.PLUG: WorkoverAction.PLUG,
    WorkoverActionTag.CON: WorkoverAction.CON,
    WorkoverActionTag.PLUS_CON: WorkoverAction.PLUS_CON,
}
FLUID_PHASE_FROM_TAG = {
    FluidPhaseTag.OIL: FluidPhase.OIL,
    FluidPhaseTag.WATER: FluidPhase.WATER,
    FluidPhaseTag.GAS: FluidPhase.GAS,
}


def none_if_nan(value: float) -> float | None:
    """
    Returns `value`, or `None` if it's `NaN`

    :param value: A possibly-`NaN` float from a compiled array.
    :returns: `value`, or `None` if it's `NaN`.
    """
    return None if math.isnan(value) else value


def decompile_limit(controls: CompiledWellControls, row: int, unit_system: UnitSystem) -> Limit:
    """
    Rebuilds one rich `Limit` from a single row of `CompiledLimits`.

    :param controls: This well's `CompiledWellControls`, for `.limits`.
    :param row: The row to rebuild.
    :param unit_system: Unit system to tag the rebuilt limit with.
    :returns: The rebuilt `BHPLimit`, `THPLimit`, `RateLimit`, or `EconomicLimit`.
    :raises ValueError: If the row's `LimitKind` isn't recognized.
    """
    limits = controls.limits
    kind = limits.kinds[row]
    min_value = none_if_nan(limits.min_values[row])
    max_value = none_if_nan(limits.max_values[row])

    if kind == LimitKind.BHP:
        return BHPLimit(min_value=min_value, max_value=max_value, unit_system=unit_system)
    if kind == LimitKind.THP:
        return THPLimit(min_value=min_value, max_value=max_value, unit_system=unit_system)
    if kind == LimitKind.RATE:
        assert max_value is not None, "`LimitKind.RATE` `max_value` row cannot be None"
        return RateLimit(
            quantity=RATE_QUANTITY_FROM_TAG[limits.quantities[row]],
            max_value=max_value,
            unit_system=unit_system,
        )
    if kind == LimitKind.ECONOMIC:
        return EconomicLimit(
            quantity=ECONOMIC_QUANTITY_FROM_TAG[limits.quantities[row]],
            min_value=min_value,
            max_value=max_value,
            workover_action=WORKOVER_ACTION_FROM_TAG[limits.workover_actions[row]],
            end_run=bool(limits.end_run_flags[row]),
            unit_system=unit_system,
        )
    raise ValueError(f"Unknown `LimitKind`: {kind!r}.")


def decompile_control(
    controls: CompiledWellControls,
    well_row: int,
    all_limits: tuple[Limit, ...],
    unit_system: UnitSystem,
) -> WellControl:
    """
    Rebuilds one well's rich `ProducerControl`/`InjectorControl` from its
    row in `CompiledWellControls`.

    Read from the compiled row rather than looked up from the well's
    original rich control, since the compiled row is the one that could
    have moved since compile time (e.g, from a group allocation share, a `WELTARG`
    event), and it should be what this reflects.

    :param controls: `CompiledWellControls` for the whole system.
    :param well_row: This well's row.
    :param all_limits: This well's already-rebuilt limits, in `CompiledLimits` row order.
    :param unit_system: Unit system to tag the rebuilt control with.
    :returns: The rebuilt `ProducerControl` or `InjectorControl`.
    """
    target_rate = none_if_nan(controls.target_rates[well_row])
    target_bhp = none_if_nan(controls.target_bhps[well_row])
    target_thp = none_if_nan(controls.target_thps[well_row])
    efficiency_factor = controls.efficiency_factors[well_row]
    guide_rate = none_if_nan(controls.guide_rates[well_row])

    if controls.well_kinds[well_row] == WellKind.INJECTOR:
        return InjectorControl(
            injected_phase=FLUID_PHASE_FROM_TAG[controls.injected_phases[well_row]],
            mode=INJECTOR_MODE_FROM_TAG[controls.control_modes[well_row]],
            target_rate=target_rate,
            target_bhp=target_bhp,
            target_thp=target_thp,
            limits=all_limits,
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    return ProducerControl(
        mode=PRODUCER_MODE_FROM_TAG[controls.control_modes[well_row]],
        target_rate=target_rate,
        target_bhp=target_bhp,
        target_thp=target_thp,
        limits=all_limits,
        efficiency_factor=efficiency_factor,
        guide_rate=guide_rate,
        unit_system=unit_system,
    )


def build_wells_states(
    wells: Wells,
    compiled_system: CompiledWellSystem,
    resolution: CompiledWellResolution,
) -> WellsStates:
    """
    Builds `WellsStates` from a resolved `CompiledWellResolution`.

    Only covers wells actually resolved this pass. A well whose
    `WellStatus` is still `PENDING` (its BHP is left `NaN` by
    `resolve_control`) is skipped rather than reported with meaningless
    values.

    Each connection's own phase rates aren't tracked past a resolve pass
    and so are not set on the returned `PerforationState`s (left `NaN`,
    the same way any other not-computed value is elsewhere in this
    package). Only that connection's own flowing pressure is real. The
    well's own total rates, reservoir- and surface-condition alike, are real.

    :param wells: The original rich `Wells` this system was compiled
        from. This supplies each `PerforationState.perforation`, which the
        compiled layer doesn't retain a reference to.
    :param compiled_system: The system `resolution` was resolved against.
    :param resolution: A `CompiledWellResolution` from a completed resolve pass.
    :returns: One `WellState` per resolved well, keyed by well name.
    """
    controls = compiled_system.controls
    perforations = compiled_system.perforations
    unit_system = compiled_system.unit_system

    states: dict[str, WellState] = {}
    for well_row, well_name in enumerate(compiled_system.names):
        bhp = resolution.bhps[well_row]
        if math.isnan(bhp):
            continue  # not resolved this pass (PENDING, or UNSET control)

        perforation_start = perforations.well_offsets[well_row]
        perforation_end = perforations.well_offsets[well_row + 1]
        rich_perforations = wells[well_name].perforations

        perforation_states = []
        for row in range(perforation_start, perforation_end):
            pressure = resolution.connection_pressures[row]
            if math.isnan(pressure):
                continue  # this connection wasn't active this pass (shut or pending)

            local_index = row - perforation_start
            perforation_states.append(
                PerforationState(
                    perforation=rich_perforations[local_index],
                    cell_index=int(perforations.cell_indices[row]),
                    flowing_pressure=pressure,
                    phase_rates=PhaseValues(
                        oil=resolution.connection_oil_rates[row],
                        water=resolution.connection_water_rates[row],
                        gas=resolution.connection_gas_rates[row],
                    ),
                    unit_system=unit_system,
                )
            )

        limits_start = controls.limits.well_offsets[well_row]
        limits_end = controls.limits.well_offsets[well_row + 1]
        well_limits = tuple(
            decompile_limit(controls, row, unit_system) for row in range(limits_start, limits_end)
        )
        active_limit_row = resolution.active_limit_rows[well_row]
        active_limit = (
            None
            if active_limit_row == UNSET_INT
            else decompile_limit(controls, active_limit_row, unit_system)
        )

        states[well_name] = WellState(
            well_name=well_name,
            is_open=not bool(resolution.economic_shutins[well_row]),
            active_control=decompile_control(controls, well_row, well_limits, unit_system),
            bhp=bhp,
            perforation_states=tuple(perforation_states),
            phase_rates=PhaseValues(
                oil=resolution.oil_rates[well_row],
                water=resolution.water_rates[well_row],
                gas=resolution.gas_rates[well_row],
            ),
            surface_phase_rates=PhaseValues(
                oil=resolution.surface_oil_rates[well_row],
                water=resolution.surface_water_rates[well_row],
                gas=resolution.surface_gas_rates[well_row],
            ),
            active_limit=active_limit,
            thp=none_if_nan(resolution.thps[well_row]),
            unit_system=unit_system,
        )

    return WellsStates(states=states, unit_system=unit_system)
