"""
Per-mode numerical solvers for well control resolution.

Two coupled numerical problems, both against a single scalar
(`reference_pressure`, i.e. BHP):

1. **Fixed-point**: at a *fixed* BHP, perforation flowing pressures and
   IPR-derived phase rates depend on each other (`wellbore_model` needs
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
    ProducerControl,
    ProducerControlMode,
    RateQuantity,
)
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellboreModel
from bores.wells.perforations import PerforationIndex
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
    # Every rate this engine computes (`well_index * mobility * drawdown`,
    # Darcy's law) is already reservoir-condition. RESV's "sum of
    # reservoir-condition rates across all three phases" isn't an
    # approximation of a voidage rate, it is the voidage rate, exactly, by
    # construction. No FVF conversion needed at this layer for RESV.
    #
    # TODO: ORAT/WRAT/GRAT/LRAT are a separate question: those targets are
    # conventionally *surface*-condition (deck WCONPROD items), and
    # nothing in this module converts between the two. Either
    # `ProducerControl.target_rate` is expected to already arrive here
    # pre-converted to reservoir condition by whatever builds the control, or
    # there's a live surface/reservoir unit mismatch for any well with FVF
    # meaningfully different from 1. Not resolved in this module - flagged
    # here rather than silently assumed consistent.
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
    Default `(pressure_low, pressure_high)` BHP bisection bracket.

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
    :returns: `(pressure_low, pressure_high)`.
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
    wellbore_model: WellboreModel,
    reference_pressure: Number,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    resolver_spec: ControlResolverSpec,
) -> typing.Tuple[NumberArray, typing.Dict[FluidPhase, Number]]:
    """
    Fixed-point iterate connection flowing pressures against IPR-derived
    rates at a fixed `reference_pressure` (BHP), until the well-total
    relevant-phase rate stabilizes.

    Starts from static (zero-rate) connection pressures, derives IPR rates
    from those (`well_index * mobility * drawdown` per connection), feeds
    the resulting `phase_rates` back through
    `wellbore_model.perforation_pressures` for updated flowing pressures,
    and repeats until the relative change in total rate is within
    `resolver_spec.rate_convergence_tolerance`, or
    `resolver_spec.max_fixed_point_iterations` is reached (best-effort).

    :returns: `(connection_pressures, phase_rates)`.
    """
    phase_rates: typing.Dict[FluidPhase, Number] = {
        phase: 0.0 for phase in relevant_phases
    }
    connection_pressures = wellbore_model.perforation_pressures(
        well,
        reference_pressure,
        phase_rates,
        perforation_indices,
        connection_samples,
        is_injector,
    )
    total_rate = 0.0

    for _ in range(resolver_spec.max_fixed_point_iterations):
        phase_rates = {phase: 0.0 for phase in relevant_phases}
        for pidx, sample, p_wf in zip(
            perforation_indices, connection_samples, connection_pressures
        ):
            assert pidx.well_index is not None
            drawdown = _compute_perforation_drawdown(
                sample.pressure, p_wf, is_injector=is_injector
            )
            for phase in relevant_phases:
                mobility = sample.phase_mobilities.get(phase, 0.0)
                phase_rates[phase] += pidx.well_index * mobility * drawdown

        new_total_rate = sum(phase_rates.values())
        connection_pressures = wellbore_model.perforation_pressures(
            well,
            reference_pressure,
            phase_rates,
            perforation_indices,
            connection_samples,
            is_injector,
        )
        if abs(
            new_total_rate - total_rate
        ) <= resolver_spec.rate_convergence_tolerance * max(abs(new_total_rate), 1.0):
            total_rate = new_total_rate
            break
        total_rate = new_total_rate

    return connection_pressures, phase_rates


def compute_full_phase_rates_at(
    *,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore_model: WellboreModel,
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
    an injector, `relevant_phases` must be `(spec.injected_phase,)` only -
    an injector displaces reservoir fluid rather than having an
    IPR-driven "oil rate" of its own; computing one via the same
    mobility-weighted formula used for producers would be physically
    wrong, not just cosmetically incomplete.
    """
    _, computed_rates = iterate_perforation_pressures_and_rates(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
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
    wellbore_model: WellboreModel,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    target: Number,
    pressure_low: Number,
    pressure_high: Number,
    resolver_spec: ControlResolverSpec,
    metric: typing.Literal["rate", "thp"] = "rate",
    surface_fluid_properties: typing.Optional[SurfaceFluidProperties] = None,
) -> typing.Tuple[Number, NumberArray, typing.Dict[FluidPhase, Number]]:
    """
    Bisect `reference_pressure` in `[pressure_low, pressure_high]` until
    the resulting metric (`relevant_phases` total rate, or THP) matches
    `target`.

    Both metrics are monotonic in BHP, in a direction that depends on the
    metric and (for rate) the well type:

    - `metric="rate"`: lower BHP -> higher producer rate; higher BHP ->
      higher injector rate.
    - `metric="thp"`: higher BHP -> higher THP, for both well types - a
      producer's rate (and so friction magnitude) falls as BHP rises, and
      an injector's rate rises as BHP rises, but `WellboreModel`
      implementations apply friction opposing flow direction rather than
      by geometric position (`is_injector`), so THP moves with BHP in the
      same direction either way.

    `pressure_low`/`pressure_high` must bracket the target - callers pick
    these via `get_default_pressure_bracket` (or their own). Best-effort (no
    exception) if `target` isn't achievable within the bracket - returns
    the closest bound reached after `resolver_spec.max_bisection_iterations`.

    :param metric: `"rate"` (default) matches `relevant_phases`' total
        rate against `target`; `"thp"` matches `wellbore_model.tubing_head_pressure`
        against `target` instead (requires `surface_fluid_properties`).
    :returns: `(bhp, connection_pressures, phase_rates)`, `phase_rates`
        restricted to `relevant_phases`.
    :raises ValidationError: If `metric="thp"` and `surface_fluid_properties`
        is `None`.
    """
    if metric == "thp" and surface_fluid_properties is None:
        raise ValidationError(
            "bisect_bhp(..., metric='thp') requires `surface_fluid_properties`."
        )

    low, high = pressure_low, pressure_high
    mid = 0.5 * (low + high)
    pressures = None
    phase_rates: typing.Dict[FluidPhase, Number] = {}

    for _ in range(resolver_spec.max_bisection_iterations):
        mid = 0.5 * (low + high)
        pressures, phase_rates = iterate_perforation_pressures_and_rates(
            well=well,
            perforation_indices=perforation_indices,
            connection_samples=connection_samples,
            wellbore_model=wellbore_model,
            reference_pressure=mid,
            relevant_phases=relevant_phases,
            is_injector=is_injector,
            resolver_spec=resolver_spec,
        )

        if metric == "rate":
            value = sum(phase_rates.values())
            increasing_with_bhp = is_injector
        else:
            assert surface_fluid_properties is not None
            value = wellbore_model.tubing_head_pressure(
                well, mid, phase_rates, surface_fluid_properties, is_injector
            )
            increasing_with_bhp = True

        if abs(value - target) <= resolver_spec.rate_convergence_tolerance * max(
            abs(target), 1.0
        ):
            break
        if (value < target) == increasing_with_bhp:
            low = mid
        else:
            high = mid

    assert pressures is not None
    return mid, pressures, phase_rates


def solve_producer_rate_mode(
    control: ProducerControl,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> ControlResolution:
    """
    Resolve a producer under `mode in {ORAT, WRAT, GRAT, LRAT, RESV}`.

    Nominal resolution only - does not check `control.limits`; the caller
    (`wells.control.engine.resolve_control`) applies those afterward via
    `wells.control.limits.apply_limits`.
    """
    relevant_phases = PRODUCER_RATE_MODE_PHASES[control.mode]
    pressure_low, pressure_high = get_default_pressure_bracket(
        connection_samples, is_injector=False, resolver_spec=resolver_spec
    )
    assert control.target_rate is not None
    bhp, _, _ = bisect_bhp(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        relevant_phases=relevant_phases,
        is_injector=False,
        target=control.target_rate,
        pressure_low=pressure_low,
        pressure_high=pressure_high,
        resolver_spec=resolver_spec,
    )
    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
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
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> ControlResolution:
    """Resolve a producer held at a fixed BHP; rates are the output.
    Nominal resolution only - see `solve_producer_rate_mode`."""
    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
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
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> ControlResolution:
    """Injector analogue of `solve_producer_rate_mode` -
    `control.injected_phase` selects the single phase being allocated.
    Nominal resolution only - see `solve_producer_rate_mode`."""
    relevant_phases = (control.injected_phase,)
    pressure_low, pressure_high = get_default_pressure_bracket(
        connection_samples, is_injector=True, resolver_spec=resolver_spec
    )
    assert control.target_rate is not None
    bhp, _, _ = bisect_bhp(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        relevant_phases=relevant_phases,
        is_injector=True,
        target=control.target_rate,
        pressure_low=pressure_low,
        pressure_high=pressure_high,
        resolver_spec=resolver_spec,
    )
    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
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
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    resolver_spec: ControlResolverSpec,
) -> ControlResolution:
    """Injector analogue of `solve_producer_bhp_mode`. Nominal resolution
    only - see `solve_producer_rate_mode`."""
    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        reference_pressure=target_bhp,
        relevant_phases=(injected_phase,),
        is_injector=True,
        resolver_spec=resolver_spec,
    )
    return ControlResolution(bhp=target_bhp, phase_rates=phase_rates, active_limit=None)
