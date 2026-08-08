"""
Per-mode numerical solvers for well control resolution.

Two coupled numerical problems, both against a single scalar (`reference_pressure`, i.e. BHP):

1. **Fixed-point**: at a *fixed* BHP, perforation flowing pressures and
   IPR-derived phase rates depend on each other (`wellbore` needs
   rates to compute mixture density/friction; the IPR needs flowing
   pressures to compute rates), `_iterate_perforation_pressures_and_rates`
   reconciles the two.

2. **Bisection**: for a rate-mode or THP-limit target, the BHP itself is
   the unknown. Hence `_bisect_bhp` wraps (1) in a bisection search, since both
   total rate and THP are monotonic in BHP (opposite directions for
   producer vs. injector rate; same direction for THP - see `_bisect_bhp`).
"""

import typing

from bores.errors import ValidationError
from bores.typing import FluidPhase, Number, NumberArray
from bores.wells.base import Well
from bores.wells.controls import (
    InjectorControl,
    InjectorControlMode,
    ProducerControl,
    ProducerControlMode,
    RateQuantity,
)
from bores.wells.hydraulics.base import SurfaceFluidProperties, Wellbore
from bores.wells.indices.perforations import PerforationIndex
from bores.wells.resolution.base import ControlResolution, ControlResolverSpec
from bores.wells.states import ConnectionSample

__all__ = [
    "ALL_PHASES",
    "PRODUCER_RATE_MODE_PHASES",
    "RATE_QUANTITY_PHASES",
    "get_default_pressure_bracket",
    "solve_producer_rate_mode",
    "solve_producer_bhp_mode",
    "solve_injector_rate_mode",
    "solve_injector_bhp_mode",
]

PRODUCER_RATE_MODE_PHASES: typing.Dict[
    ProducerControlMode, typing.Tuple[FluidPhase, ...]
] = {
    ProducerControlMode.ORAT: (FluidPhase.OIL,),
    ProducerControlMode.WRAT: (FluidPhase.WATER,),
    ProducerControlMode.GRAT: (FluidPhase.GAS,),
    ProducerControlMode.LRAT: (FluidPhase.OIL, FluidPhase.WATER),
    ProducerControlMode.RESV: (FluidPhase.OIL, FluidPhase.WATER, FluidPhase.GAS),
}

RATE_QUANTITY_PHASES: typing.Dict[RateQuantity, typing.Tuple[FluidPhase, ...]] = {
    RateQuantity.OIL: (FluidPhase.OIL,),
    RateQuantity.WATER: (FluidPhase.WATER,),
    RateQuantity.GAS: (FluidPhase.GAS,),
    RateQuantity.LIQUID: (FluidPhase.OIL, FluidPhase.WATER),
    RateQuantity.RESERVOIR: (FluidPhase.OIL, FluidPhase.WATER, FluidPhase.GAS),
}

ALL_PHASES: typing.Tuple[FluidPhase, ...] = (
    FluidPhase.OIL,
    FluidPhase.WATER,
    FluidPhase.GAS,
)


def get_default_pressure_bracket(
    connection_samples: typing.Sequence[ConnectionSample],
    *,
    is_injector: bool,
    resolver_spec: ControlResolverSpec,
) -> typing.Tuple[Number, Number]:
    """
    Default `(min_pressure, max_pressure)` BHP bisection bracket.

    Producer: `[resolver_spec.producer_bhp_floor, max(cell pressures)]` -
    a producer's BHP can't usefully exceed the highest connected-cell
    pressure (no drawdown left to give a rate above that).

    Injector: `[min(cell pressures), resolver_spec.injector_bhp_bracket_multiplier
    * max(cell pressures)]` - the multiplier gives room above reservoir
    pressure for the injector to actually inject; widen
    `resolver_spec.injector_bhp_bracket_multiplier` if a target rate isn't
    reachable within it.

    :param connection_samples: Reservoir samples for every connection.
    :param is_injector: Selects which bracket shape to use.
    :param resolver_spec: Supplies `producer_bhp_floor`/
        `injector_bhp_bracket_multiplier`.
    :returns: `(min_pressure, max_pressure)`.
    """
    cell_pressures = [sample.pressure for sample in connection_samples]
    if is_injector:
        return min(cell_pressures), resolver_spec.injector_bhp_bracket_multiplier * max(
            cell_pressures
        )
    return resolver_spec.producer_bhp_floor, max(cell_pressures)


def _compute_perforation_drawdown(
    cell_pressure: Number, flowing_pressure: Number, *, is_injector: bool
) -> Number:
    """Linear IPR drawdown, clipped to `>= 0` - no crossflow between
    connections; a connection with reversed drawdown contributes zero
    rather than a negative rate."""
    if is_injector:
        return max(flowing_pressure - cell_pressure, 0.0)
    return max(cell_pressure - flowing_pressure, 0.0)


def iterate_perforation_pressures_and_rates(
    *,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore: Wellbore,
    reference_pressure: Number,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    resolver_spec: ControlResolverSpec,
) -> typing.Tuple[
    NumberArray, typing.Dict[FluidPhase, Number], typing.Dict[FluidPhase, Number]
]:
    """
    Fixed-point iterate connection flowing pressures against IPR-derived
    rates at a fixed `reference_pressure` (BHP), until the well-total
    relevant-phase rate stabilizes.

    Starts from static (zero-rate) connection pressures, derives IPR rates
    from those (`well_index * mobility * drawdown` per connection), feeds
    the resulting `phase_rates` back through
    `wellbore.perforation_pressures` for updated flowing pressures,
    and repeats until the relative change in total rate is within
    `resolver_spec.rate_convergence_tolerance`, or
    `resolver_spec.max_fixed_point_iterations` is reached (best-effort).

    Every connection's IPR contribution is also divided by that
    connection's own formation volume factor and accumulated separately,
    since the two totals are needed for different purposes: `phase_rates`
    (reservoir-condition) is what `wellbore` and a RESERVOIR-condition
    target need, `surface_condition_phase_rates` is what a surface-condition
    target (ORAT/WRAT/GRAT/LRAT) needs to be compared against.

    :param well: Static well data.
    :param perforation_indices: Connections, `well_index.perforations` order.
    :param connection_samples: Reservoir samples, same order as `perforation_indices`.
    :param wellbore: Hydraulics strategy for this well.
    :param reference_pressure: BHP held fixed while iterating.
    :param relevant_phases: Phases to accumulate rates for.
    :param is_injector: Selects the drawdown sign convention.
    :param resolver_spec: Supplies the iteration cap and convergence tolerance.
    :returns: `(connection_pressures, phase_rates, surface_condition_phase_rates)`.
    """
    phase_rates: typing.Dict[FluidPhase, Number] = {
        phase: 0.0 for phase in relevant_phases
    }
    connection_pressures = wellbore.compute_perforation_pressures(
        well=well,
        reference_pressure=reference_pressure,
        phase_rates=phase_rates,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        is_injector=is_injector,
    )
    total_rate = 0.0
    surface_condition_phase_rates: typing.Dict[FluidPhase, Number] = dict(phase_rates)

    for _ in range(resolver_spec.max_fixed_point_iterations):
        phase_rates = {phase: 0.0 for phase in relevant_phases}
        surface_condition_phase_rates = {phase: 0.0 for phase in relevant_phases}
        for pidx, sample, flowing_pressure in zip(
            perforation_indices, connection_samples, connection_pressures
        ):
            assert pidx.well_index is not None
            drawdown = _compute_perforation_drawdown(
                cell_pressure=sample.pressure,
                flowing_pressure=flowing_pressure,
                is_injector=is_injector,
            )
            for phase in relevant_phases:
                mobility = sample.phase_mobilities.get(phase, 0.0)
                reservoir_condition_contribution = pidx.well_index * mobility * drawdown
                phase_rates[phase] += reservoir_condition_contribution
                formation_volume_factor = sample.phase_formation_volume_factors.get(
                    phase, 1.0
                )
                surface_condition_phase_rates[phase] += (
                    reservoir_condition_contribution / formation_volume_factor
                )

        new_total_rate = sum(phase_rates.values())
        connection_pressures = wellbore.compute_perforation_pressures(
            well=well,
            reference_pressure=reference_pressure,
            phase_rates=phase_rates,
            perforation_indices=perforation_indices,
            connection_samples=connection_samples,
            is_injector=is_injector,
        )
        if abs(
            new_total_rate - total_rate
        ) <= resolver_spec.rate_convergence_tolerance * max(abs(new_total_rate), 1.0):
            total_rate = new_total_rate
            break
        total_rate = new_total_rate

    return connection_pressures, phase_rates, surface_condition_phase_rates


def compute_full_phase_rates_at(
    *,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore: Wellbore,
    reference_pressure: Number,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    resolver_spec: ControlResolverSpec,
) -> typing.Dict[FluidPhase, Number]:
    """
    `iterate_perforation_pressures_and_rates` at a fixed BHP, reporting
    `relevant_phases` only and `0.0` for every other `FluidPhase`.

    For a producer, `relevant_phases` is every phase (`ALL_PHASES`) - a
    producer can legitimately flow oil, water, and gas simultaneously. For
    an injector, `relevant_phases` must be `(control.injected_phase,)` only -
    an injector displaces reservoir fluid rather than having an
    IPR-driven "oil rate" of its own; computing one via the same
    mobility-weighted formula used for producers would be physically
    wrong, not just cosmetically incomplete.

    :param well: Static well data.
    :param perforation_indices: Connections, `well_index.perforations` order.
    :param connection_samples: Reservoir samples, same order as `perforation_indices`.
    :param wellbore: Hydraulics strategy for this well.
    :param reference_pressure: BHP to report rates at.
    :param relevant_phases: Phases to compute a real rate for.
    :param is_injector: Selects the drawdown sign convention.
    :param resolver_spec: Forwarded to `iterate_perforation_pressures_and_rates`.
    :returns: Reservoir-condition phase rates, one entry per `FluidPhase`.
    """
    _, computed_rates, _ = iterate_perforation_pressures_and_rates(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore=wellbore,
        reference_pressure=reference_pressure,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        resolver_spec=resolver_spec,
    )
    return {phase: computed_rates.get(phase, 0.0) for phase in ALL_PHASES}


def bisect_bhp(
    *,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore: Wellbore,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    target: Number,
    min_pressure: Number,
    max_pressure: Number,
    resolver_spec: ControlResolverSpec,
    metric: typing.Literal["rate", "thp"] = "rate",
    target_rate_condition: typing.Literal["surface", "reservoir"] = "surface",
    surface_fluid_properties: typing.Optional[SurfaceFluidProperties] = None,
) -> typing.Tuple[Number, NumberArray, typing.Dict[FluidPhase, Number]]:
    """
    Bisect `reference_pressure` in `[min_pressure, max_pressure]` until
    the resulting metric (`relevant_phases` total rate, or THP) matches
    `target`.

    Both metrics are monotonic in BHP, in a direction that depends on the
    metric and (for rate) the well type:

    - `metric="rate"`: lower BHP -> higher producer rate; higher BHP ->
      higher injector rate.
    - `metric="thp"`: higher BHP -> higher THP, for both well types - a
      producer's rate (and so friction magnitude) falls as BHP rises, and
      an injector's rate rises as BHP rises, but `Wellbore`
      implementations apply friction opposing flow direction rather than
      by geometric position (`is_injector`), so THP moves with BHP in the
      same direction either way.

    `min_pressure`/`max_pressure` must bracket the target - callers pick
    these via `get_default_pressure_bracket` (or their own). Best-effort (no
    exception) if `target` isn't achievable within the bracket - returns
    the closest bound reached after `resolver_spec.max_bisection_iterations`.

    :param well: Static well data.
    :param perforation_indices: Connections, `well_index.perforations` order.
    :param connection_samples: Reservoir samples, same order as `perforation_indices`.
    :param wellbore: Hydraulics strategy for this well.
    :param relevant_phases: Phases the `"rate"` metric sums over.
    :param is_injector: Selects the drawdown sign convention and the
        direction `"rate"` moves in as BHP changes.
    :param target: Value the chosen metric is bisected toward.
    :param min_pressure: Lower bracket bound.
    :param max_pressure: Upper bracket bound.
    :param resolver_spec: Supplies the iteration cap and convergence tolerance.
    :param metric: `"rate"` (default) matches `relevant_phases`' total
        rate against `target`; `"thp"` matches `wellbore.tubing_head_pressure`
        against `target` instead (requires `surface_fluid_properties`).
    :param target_rate_condition: `"surface"` (default) bisects against
        the surface-condition total (ORAT/WRAT/GRAT/LRAT-style targets,
        and most `RateLimit` quantities); `"reservoir"` bisects against
        the raw reservoir-condition total instead (RESV-mode targets,
        `RateLimit(quantity=RateQuantity.RESERVOIR)`). Ignored when
        `metric="thp"`.
    :param surface_fluid_properties: Required if `metric="thp"`.
    :returns: `(bhp, connection_pressures, phase_rates)`, `phase_rates`
        restricted to `relevant_phases` and always reservoir-condition,
        regardless of `target_rate_condition`.
    :raises ValidationError: If `metric="thp"` and `surface_fluid_properties`
        is `None`.
    """
    if metric == "thp" and surface_fluid_properties is None:
        raise ValidationError(
            "bisect_bhp(..., metric='thp') requires `surface_fluid_properties`."
        )

    low, high = min_pressure, max_pressure
    average_pressure = 0.5 * (low + high)
    pressures = None
    phase_rates: typing.Dict[FluidPhase, Number] = {}

    for _ in range(resolver_spec.max_bisection_iterations):
        average_pressure = 0.5 * (low + high)
        pressures, phase_rates, surface_condition_phase_rates = (
            iterate_perforation_pressures_and_rates(
                well=well,
                perforation_indices=perforation_indices,
                connection_samples=connection_samples,
                wellbore=wellbore,
                reference_pressure=average_pressure,
                relevant_phases=relevant_phases,
                is_injector=is_injector,
                resolver_spec=resolver_spec,
            )
        )

        if metric == "rate":
            rates_for_target = (
                phase_rates
                if target_rate_condition == "reservoir"
                else surface_condition_phase_rates
            )
            value = sum(rates_for_target.values())
            increasing_with_bhp = is_injector
        else:
            assert surface_fluid_properties is not None
            value = wellbore.compute_tubing_head_pressure(
                well=well,
                reference_pressure=average_pressure,
                phase_rates=phase_rates,
                surface_fluid_properties=surface_fluid_properties,
                is_injector=is_injector,
            )
            increasing_with_bhp = True

        if abs(value - target) <= resolver_spec.rate_convergence_tolerance * max(
            abs(target), 1.0
        ):
            break
        if (value < target) == increasing_with_bhp:
            low = average_pressure
        else:
            high = average_pressure

    assert pressures is not None
    return average_pressure, pressures, phase_rates


def solve_producer_rate_mode(
    control: ProducerControl,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    wellbore: Wellbore,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> ControlResolution:
    """
    Resolve a producer under `mode in {ORAT, WRAT, GRAT, LRAT, RESV}`.

    Nominal resolution only. Does not check `control.limits`; the caller
    should apply limits afterward via `wells.resolution.limits.apply_limits`.

    :param control: The producer's nominal control for this timestep.
    :param well: Static well data.
    :param perforation_indices: Connections, `well_index.perforations` order.
    :param wellbore: Hydraulics strategy for this well.
    :param connection_samples: Reservoir samples, same order as `perforation_indices`.
    :param resolver_spec: Solver tunables.
    :returns: Nominal `ControlResolution`.
    """
    relevant_phases = PRODUCER_RATE_MODE_PHASES[control.mode]
    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples, is_injector=False, resolver_spec=resolver_spec
    )
    assert control.target_rate is not None
    bhp, _, _ = bisect_bhp(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore=wellbore,
        relevant_phases=relevant_phases,
        is_injector=False,
        target=control.target_rate,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        resolver_spec=resolver_spec,
        target_rate_condition="reservoir"
        if control.mode is ProducerControlMode.RESV
        else "surface",
    )
    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore=wellbore,
        reference_pressure=bhp,
        relevant_phases=ALL_PHASES,
        is_injector=False,
        resolver_spec=resolver_spec,
    )
    return ControlResolution(bhp=bhp, phase_rates=phase_rates, active_limit=None)


def solve_producer_bhp_mode(
    target_bhp: Number,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    wellbore: Wellbore,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> ControlResolution:
    """
    Resolve a producer held at a fixed BHP; rates are the output.

    Nominal resolution only - see `solve_producer_rate_mode`.
    """
    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore=wellbore,
        reference_pressure=target_bhp,
        relevant_phases=ALL_PHASES,
        is_injector=False,
        resolver_spec=resolver_spec,
    )
    return ControlResolution(bhp=target_bhp, phase_rates=phase_rates, active_limit=None)


def solve_injector_rate_mode(
    control: InjectorControl,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    wellbore: Wellbore,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> ControlResolution:
    """
    Injector analogue of `solve_producer_rate_mode` -
    `control.injected_phase` selects the single phase being allocated.

    Nominal resolution only - see `solve_producer_rate_mode`.

    :param control: The injector's nominal control for this timestep.
    :param well: Static well data.
    :param perforation_indices: Connections, `well_index.perforations` order.
    :param wellbore: Hydraulics strategy for this well.
    :param connection_samples: Reservoir samples, same order as `perforation_indices`.
    :param resolver_spec: Solver tunables.
    :returns: Nominal `ControlResolution`.
    """
    relevant_phases = (control.injected_phase,)
    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples=connection_samples,
        is_injector=True,
        resolver_spec=resolver_spec,
    )
    assert control.target_rate is not None
    bhp, _, _ = bisect_bhp(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore=wellbore,
        relevant_phases=relevant_phases,
        is_injector=True,
        target=control.target_rate,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        resolver_spec=resolver_spec,
        target_rate_condition="reservoir"
        if control.mode is InjectorControlMode.RESV
        else "surface",
    )
    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore=wellbore,
        reference_pressure=bhp,
        relevant_phases=(control.injected_phase,),
        is_injector=True,
        resolver_spec=resolver_spec,
    )
    return ControlResolution(bhp=bhp, phase_rates=phase_rates, active_limit=None)


def solve_injector_bhp_mode(
    target_bhp: Number,
    injected_phase: FluidPhase,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    wellbore: Wellbore,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> ControlResolution:
    """
    Injector analogue of `solve_producer_bhp_mode`.

    Nominal resolution only - see `solve_producer_rate_mode`.
    """
    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore=wellbore,
        reference_pressure=target_bhp,
        relevant_phases=(injected_phase,),
        is_injector=True,
        resolver_spec=resolver_spec,
    )
    return ControlResolution(bhp=target_bhp, phase_rates=phase_rates, active_limit=None)
