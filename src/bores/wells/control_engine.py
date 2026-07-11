"""Well control resolution: target -> per-perforation result.

Depends on `wells.data`, `wells.controls`, `wells.state`,
`wells.hydraulics.WellboreModel`, `wells.index.WellIndex`. Implements D8.

**v1 granularity limitation, not addressed in `02_BEHAVIOR_LAYER.md` and
found only by tracing the two modules against each other:**
`WellboreModel.perforation_pressures` operates per `Perforation` (one
flowing pressure per entry in `well.open_perforations`), while
`WellIndex`/`PerforationIndex` (`wells.index`) operate per resolved cell -
a single `Perforation` spanning multiple cells produces multiple
`PerforationIndex` entries. These coincide for the common case (one cell
per perforation - also the SPE1 validation target) but not in general.
`_single_cell_perforation_indices` enforces the 1:1 case and raises
`NotImplementedError` rather than silently averaging or picking one cell
when they don't coincide.

**Numerical solve:** `02_BEHAVIOR_LAYER.md` describes the BHP/rate
reconciliation only qualitatively ("iterate... until converged"). Implemented
here as a bounded fixed-point iteration (flowing pressures <-> IPR rates)
nested inside a bisection search on BHP for rate-mode targets. Iteration
counts/tolerances are named module constants, not buried literals, so
they're easy to retune.
"""

import typing

import attrs

from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import FluidPhase, Number, NumberArray
from bores.wells.controls import (
    BHPLimit,
    ControlSpec,
    InjectorControlMode,
    InjectorControlSpec,
    Limit,
    ProducerControlMode,
    ProducerControlSpec,
    RateLimit,
    RateQuantity,
)
from bores.wells.data import Perforation, WellSpec
from bores.wells.hydraulics import WellboreModel
from bores.wells.index import WellIndex
from bores.wells.location import PerforationIndex
from bores.wells.states import (
    ConnectionSample,
    PerforationState,
    WellState,
    as_phase_mapping,
)

__all__ = ["ControlResolution", "resolve_control", "build_well_state"]

_MAX_FIXED_POINT_ITERATIONS = 20
_RATE_CONVERGENCE_TOLERANCE = 1.0e-6  # relative
_MAX_BISECTION_ITERATIONS = 40
_MIN_BHP = 1.0e-3  # near-vacuum floor for producer BHP search

_PRODUCER_RATE_MODE_PHASES: typing.Dict[
    ProducerControlMode, typing.Tuple[FluidPhase, ...]
] = {
    ProducerControlMode.ORAT: (FluidPhase.OIL,),
    ProducerControlMode.WRAT: (FluidPhase.WATER,),
    ProducerControlMode.GRAT: (FluidPhase.GAS,),
    ProducerControlMode.LRAT: (FluidPhase.OIL, FluidPhase.WATER),
    # RESV: no FVF/voidage conversion available at this layer (D7), so v1
    # approximates "reservoir volume rate" as the sum of reservoir-condition
    # mobility-driven rates across all three phases. Flagged, not silently
    # treated as physically exact.
    ProducerControlMode.RESV: (FluidPhase.OIL, FluidPhase.WATER, FluidPhase.GAS),
}

_RATE_QUANTITY_PHASES: typing.Dict[RateQuantity, typing.Tuple[FluidPhase, ...]] = {
    RateQuantity.OIL: (FluidPhase.OIL,),
    RateQuantity.WATER: (FluidPhase.WATER,),
    RateQuantity.GAS: (FluidPhase.GAS,),
    RateQuantity.LIQUID: (FluidPhase.OIL, FluidPhase.WATER),
    RateQuantity.RESERVOIR: (FluidPhase.OIL, FluidPhase.WATER, FluidPhase.GAS),
}

_ALL_PHASES: typing.Tuple[FluidPhase, ...] = (
    FluidPhase.OIL,
    FluidPhase.WATER,
    FluidPhase.GAS,
)


@attrs.frozen(kw_only=True, slots=True)
class ControlResolution(Serializable):
    """Output of `resolve_control` - everything needed to build a
    `WellState` for this timestep, one level below the final `WellState`
    (doesn't yet include per-perforation allocation, see `build_well_state`).
    """

    bhp: Number
    phase_rates: typing.Mapping[FluidPhase, Number] = attrs.field(
        converter=as_phase_mapping
    )
    active_limit: typing.Optional[Limit] = None
    """Which `Limit` from the spec's `limits` tuple is currently binding,
    `None` if the primary target is achieved without hitting any limit."""


def _single_cell_perforation_indices(
    well: WellSpec, well_index: WellIndex
) -> typing.Tuple[PerforationIndex, ...]:
    """Match each `well.open_perforations` entry to exactly one
    `PerforationIndex` in `well_index`. See module docstring.

    :raises NotImplementedError: If any open perforation resolved to more
        or less than one cell.
    """
    by_perforation: typing.Dict[Perforation, typing.List[PerforationIndex]] = {}
    for pidx in well_index.perforations:
        by_perforation.setdefault(pidx.perforation, []).append(pidx)

    ordered: typing.List[PerforationIndex] = []
    for perforation in well.open_perforations:
        matches = by_perforation.get(perforation, [])
        if len(matches) != 1:
            raise NotImplementedError(
                f"Perforation [{perforation.top_depth}, {perforation.bottom_depth}] "
                f"on well {well.name!r} resolved to {len(matches)} cells; "
                "control_engine.py v1 only supports exactly one cell per "
                "perforation (see module docstring)."
            )
        ordered.append(matches[0])
    return tuple(ordered)


def _perforation_drawdown(
    cell_pressure: Number, flowing_pressure: Number, *, is_injector: bool
) -> Number:
    """Linear IPR drawdown, clipped to `>= 0` (no crossflow in v1)."""
    if is_injector:
        return max(flowing_pressure - cell_pressure, 0.0)
    return max(cell_pressure - flowing_pressure, 0.0)


def _iterate_perforation_pressures_and_rates(
    *,
    well: WellSpec,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore_model: WellboreModel,
    reference_pressure: Number,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
) -> typing.Tuple[NumberArray, typing.Dict[FluidPhase, Number]]:
    """
    Fixed-point iterate perforation flowing pressures against IPR-derived
    rates at a fixed `reference_pressure` (BHP), until the well-total
    relevant-phase rate stabilizes.

    Starts from static (zero-rate) perforation pressures, derives IPR rates
    from those (`well_index * mobility * drawdown` per open perforation),
    feeds the resulting `phase_rates` back through
    `wellbore_model.perforation_pressures` for updated flowing pressures,
    and repeats.

    :returns: `(perforation_pressures, phase_rates)` at convergence (or
        after `_MAX_FIXED_POINT_ITERATIONS`, best-effort).
    """
    phase_rates: typing.Dict[FluidPhase, Number] = {
        phase: 0.0 for phase in relevant_phases
    }
    perforation_pressures = wellbore_model.perforation_pressures(
        well, reference_pressure, phase_rates, connection_samples
    )
    total_rate = 0.0

    for _ in range(_MAX_FIXED_POINT_ITERATIONS):
        phase_rates = {phase: 0.0 for phase in relevant_phases}
        for pidx, sample, p_wf in zip(
            perforation_indices, connection_samples, perforation_pressures
        ):
            drawdown = _perforation_drawdown(
                sample.pressure, p_wf, is_injector=is_injector
            )
            for phase in relevant_phases:
                mobility = sample.phase_mobilities.get(phase, 0.0)
                phase_rates[phase] += pidx.well_index * mobility * drawdown

        new_total_rate = sum(phase_rates.values())
        perforation_pressures = wellbore_model.perforation_pressures(
            well, reference_pressure, phase_rates, connection_samples
        )
        if abs(new_total_rate - total_rate) <= _RATE_CONVERGENCE_TOLERANCE * max(
            abs(new_total_rate), 1.0
        ):
            total_rate = new_total_rate
            break
        total_rate = new_total_rate

    return perforation_pressures, phase_rates


def _solve_bhp_for_target_rate(
    *,
    well: WellSpec,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore_model: WellboreModel,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    target_rate: Number,
    pressure_low: Number,
    pressure_high: Number,
) -> typing.Tuple[Number, NumberArray, typing.Dict[FluidPhase, Number]]:
    """
    Bisect `reference_pressure` in `[pressure_low, pressure_high]` until
    the resulting `relevant_phases` total rate matches `target_rate`.

    Monotonic by construction: lower BHP -> higher producer rate; higher
    BHP -> higher injector rate. `pressure_low`/`pressure_high` must
    bracket the target (callers pick these from the connection samples'
    cell pressures - see `_solve_producer_rate_mode`/`_solve_injector_rate_mode`).
    Best-effort (no exception) if `target_rate` isn't achievable within
    the bracket - returns the closest bound reached after
    `_MAX_BISECTION_ITERATIONS`.

    :returns: `(bhp, perforation_pressures, phase_rates)` restricted to
        `relevant_phases`.
    """
    low, high = pressure_low, pressure_high
    mid = 0.5 * (low + high)
    pressures = connection_samples  # placeholder for type checkers; overwritten below
    phase_rates: typing.Dict[FluidPhase, Number] = {}

    for _ in range(_MAX_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        pressures, phase_rates = _iterate_perforation_pressures_and_rates(
            well=well,
            perforation_indices=perforation_indices,
            connection_samples=connection_samples,
            wellbore_model=wellbore_model,
            reference_pressure=mid,
            relevant_phases=relevant_phases,
            is_injector=is_injector,
        )
        rate = sum(phase_rates.values())
        if abs(rate - target_rate) <= _RATE_CONVERGENCE_TOLERANCE * max(
            abs(target_rate), 1.0
        ):
            break
        higher_bhp_gives_higher_rate = is_injector
        if (rate < target_rate) == higher_bhp_gives_higher_rate:
            low = mid
        else:
            high = mid

    return mid, pressures, phase_rates


def _full_phase_rates_at(
    *,
    well: WellSpec,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore_model: WellboreModel,
    reference_pressure: Number,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
) -> typing.Dict[FluidPhase, Number]:
    """`_iterate_perforation_pressures_and_rates` at a fixed BHP, reporting
    `relevant_phases` only and `0.0` for every other `FluidPhase`.

    For a producer, `relevant_phases` is every phase (`_ALL_PHASES`) - a
    producer can legitimately flow oil, water, and gas simultaneously. For
    an injector, `relevant_phases` must be `(spec.injected_phase,)` only -
    an injector displaces reservoir fluid, it doesn't have an IPR-driven
    "oil rate" of its own; computing one via the same mobility-weighted
    formula used for producers is physically wrong, not just cosmetically
    incomplete. Caught by testing an injector end-to-end, not visible from
    the formula alone.
    """
    _, computed_rates = _iterate_perforation_pressures_and_rates(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        reference_pressure=reference_pressure,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
    )
    return {phase: computed_rates.get(phase, 0.0) for phase in _ALL_PHASES}


def _apply_limits(
    *,
    spec: ControlSpec,
    well: WellSpec,
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore_model: WellboreModel,
    perforation_indices: typing.Sequence[PerforationIndex],
    full_relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    resolution: ControlResolution,
    pressure_low: Number,
    pressure_high: Number,
) -> ControlResolution:
    """
    Check `spec.limits` in order against `resolution`; if any is violated,
    re-resolve at exactly that limit's boundary and return, with
    `active_limit` set. Stops at the first violated limit. Returns
    `resolution` unchanged if none are violated.

    :param full_relevant_phases: `_ALL_PHASES` for a producer;
        `(spec.injected_phase,)` for an injector - see `_full_phase_rates_at`.
    """
    for limit in spec.limits:
        if isinstance(limit, BHPLimit):
            floor_violated = (
                not is_injector
                and limit.min_value is not None
                and resolution.bhp < limit.min_value
            )
            ceiling_violated = (
                is_injector
                and limit.max_value is not None
                and resolution.bhp > limit.max_value
            )
            if not (floor_violated or ceiling_violated):
                continue
            bound_bhp = limit.min_value if floor_violated else limit.max_value
            phase_rates = _full_phase_rates_at(
                well=well,
                perforation_indices=perforation_indices,
                connection_samples=connection_samples,
                wellbore_model=wellbore_model,
                reference_pressure=bound_bhp,
                relevant_phases=full_relevant_phases,
                is_injector=is_injector,
            )
            return ControlResolution(
                bhp=bound_bhp, phase_rates=phase_rates, active_limit=limit
            )

        if isinstance(limit, RateLimit):
            quantity_phases = _RATE_QUANTITY_PHASES[limit.quantity]
            current = sum(
                resolution.phase_rates.get(phase, 0.0) for phase in quantity_phases
            )
            if current <= limit.max_value:
                continue
            bound_bhp, _, _ = _solve_bhp_for_target_rate(
                well=well,
                perforation_indices=perforation_indices,
                connection_samples=connection_samples,
                wellbore_model=wellbore_model,
                relevant_phases=quantity_phases,
                is_injector=is_injector,
                target_rate=limit.max_value,
                pressure_low=pressure_low,
                pressure_high=pressure_high,
            )
            phase_rates = _full_phase_rates_at(
                well=well,
                perforation_indices=perforation_indices,
                connection_samples=connection_samples,
                wellbore_model=wellbore_model,
                reference_pressure=bound_bhp,
                relevant_phases=full_relevant_phases,
                is_injector=is_injector,
            )
            return ControlResolution(
                bhp=bound_bhp, phase_rates=phase_rates, active_limit=limit
            )

    return resolution


def _solve_producer_rate_mode(
    spec: ProducerControlSpec,
    well: WellSpec,
    well_index: WellIndex,
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
) -> ControlResolution:
    """Resolve a producer under `mode in {ORAT, WRAT, GRAT, LRAT, RESV}`."""
    perforation_indices = _single_cell_perforation_indices(well, well_index)
    relevant_phases = _PRODUCER_RATE_MODE_PHASES[spec.mode]
    pressure_low = _MIN_BHP
    pressure_high = max(sample.pressure for sample in connection_samples)

    bhp, _, _ = _solve_bhp_for_target_rate(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        relevant_phases=relevant_phases,
        is_injector=False,
        target_rate=spec.target_rate,
        pressure_low=pressure_low,
        pressure_high=pressure_high,
    )
    phase_rates = _full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        reference_pressure=bhp,
        relevant_phases=_ALL_PHASES,
        is_injector=False,
    )
    resolution = ControlResolution(bhp=bhp, phase_rates=phase_rates, active_limit=None)
    return _apply_limits(
        spec=spec,
        well=well,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        perforation_indices=perforation_indices,
        full_relevant_phases=_ALL_PHASES,
        is_injector=False,
        resolution=resolution,
        pressure_low=pressure_low,
        pressure_high=pressure_high,
    )


def _solve_producer_bhp_mode(
    target_bhp: Number,
    well: WellSpec,
    well_index: WellIndex,
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
) -> ControlResolution:
    """Resolve a producer held at a fixed BHP; rates are the output."""
    perforation_indices = _single_cell_perforation_indices(well, well_index)
    phase_rates = _full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        reference_pressure=target_bhp,
        relevant_phases=_ALL_PHASES,
        is_injector=False,
    )
    return ControlResolution(bhp=target_bhp, phase_rates=phase_rates, active_limit=None)


def _solve_injector_rate_mode(
    spec: InjectorControlSpec,
    well: WellSpec,
    well_index: WellIndex,
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
) -> ControlResolution:
    """Injector analogue of `_solve_producer_rate_mode` - `spec.injected_phase`
    selects the single phase being allocated."""
    perforation_indices = _single_cell_perforation_indices(well, well_index)
    relevant_phases = (spec.injected_phase,)
    pressure_low = min(sample.pressure for sample in connection_samples)
    pressure_high = 10.0 * max(sample.pressure for sample in connection_samples)

    bhp, _, _ = _solve_bhp_for_target_rate(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        relevant_phases=relevant_phases,
        is_injector=True,
        target_rate=spec.target_rate,
        pressure_low=pressure_low,
        pressure_high=pressure_high,
    )
    phase_rates = _full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        reference_pressure=bhp,
        relevant_phases=(spec.injected_phase,),
        is_injector=True,
    )
    resolution = ControlResolution(bhp=bhp, phase_rates=phase_rates, active_limit=None)
    return _apply_limits(
        spec=spec,
        well=well,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        perforation_indices=perforation_indices,
        full_relevant_phases=(spec.injected_phase,),
        is_injector=True,
        resolution=resolution,
        pressure_low=pressure_low,
        pressure_high=pressure_high,
    )


def _solve_injector_bhp_mode(
    target_bhp: Number,
    injected_phase: FluidPhase,
    well: WellSpec,
    well_index: WellIndex,
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
) -> ControlResolution:
    """Injector analogue of `_solve_producer_bhp_mode`."""
    perforation_indices = _single_cell_perforation_indices(well, well_index)
    phase_rates = _full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        reference_pressure=target_bhp,
        relevant_phases=(injected_phase,),
        is_injector=True,
    )
    return ControlResolution(bhp=target_bhp, phase_rates=phase_rates, active_limit=None)


def resolve_control(
    spec: ControlSpec,
    well: WellSpec,
    well_index: WellIndex,
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
) -> ControlResolution:
    """
    Single entry point the simulation loop calls once per well per
    timestep. Dispatches to a mode-specific solver based on `type(spec)`
    and `spec.mode`.

    :param spec: The well's nominal control (from `WellSchedule`/`WellState`)
        - not necessarily what ends up governing if a limit fires.
    :param well: Static well data.
    :param well_index: Resolved connection factors.
    :param wellbore_model: Hydraulics strategy for this well.
    :param connection_samples: Reservoir samples for every open perforation,
        `well.open_perforations` order.
    :returns: `ControlResolution`.
    :raises NotImplementedError: If `spec.mode` is `THP` (VFP-based THP
        resolution is out of scope), or if a perforation resolved to more
        than one cell (see module docstring).
    :raises ValidationError: If `len(connection_samples) != len(well.open_perforations)`,
        or `spec` is neither `ProducerControlSpec` nor `InjectorControlSpec`.
    """
    if len(connection_samples) != len(well.open_perforations):
        raise ValidationError(
            f"len(connection_samples)={len(connection_samples)} != "
            f"len(well.open_perforations)={len(well.open_perforations)} for "
            f"well {well.name!r}."
        )

    if isinstance(spec, ProducerControlSpec):
        if spec.mode is ProducerControlMode.THP:
            raise NotImplementedError(
                "ProducerControlMode.THP resolution requires VFP tables; "
                "out of scope for v1."
            )
        if spec.mode is ProducerControlMode.BHP:
            return _solve_producer_bhp_mode(
                spec.target_bhp, well, well_index, wellbore_model, connection_samples
            )
        return _solve_producer_rate_mode(
            spec, well, well_index, wellbore_model, connection_samples
        )

    if isinstance(spec, InjectorControlSpec):
        if spec.mode is InjectorControlMode.THP:
            raise NotImplementedError(
                "InjectorControlMode.THP resolution requires VFP tables; "
                "out of scope for v1."
            )
        if spec.mode is InjectorControlMode.BHP:
            return _solve_injector_bhp_mode(
                spec.target_bhp,
                spec.injected_phase,
                well,
                well_index,
                wellbore_model,
                connection_samples,
            )
        return _solve_injector_rate_mode(
            spec, well, well_index, wellbore_model, connection_samples
        )

    raise ValidationError(f"Unknown ControlSpec type: {type(spec)!r}.")


def build_well_state(
    well: WellSpec,
    resolution: ControlResolution,
    well_index: WellIndex,
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    *,
    active_control: ControlSpec,
) -> WellState:
    """
    Assemble the final `WellState` for this timestep from a
    `ControlResolution` - turns the well-total numbers in `resolution` into
    the per-perforation `PerforationState` tuple `WellState` requires.

    :param well: Static well data.
    :param resolution: Output of `resolve_control`.
    :param well_index: Resolved connection factors (allocation weights).
    :param wellbore_model: Used to get each perforation's flowing pressure
        at `resolution.bhp`.
    :param connection_samples: Reservoir samples, `well.open_perforations` order.
    :param active_control: The spec actually used to produce `resolution`
        (may differ from the well's nominal spec if a limit was hit - the
        caller is responsible for passing the *effective* one here).
    :returns: Complete `WellState`, `is_open=True`.
    """
    is_injector = isinstance(active_control, InjectorControlSpec)
    perforation_indices = _single_cell_perforation_indices(well, well_index)
    perforation_pressures = wellbore_model.perforation_pressures(
        well, resolution.bhp, resolution.phase_rates, connection_samples
    )

    perforation_states: typing.List[PerforationState] = []
    for perforation, pidx, p_wf in zip(
        well.open_perforations, perforation_indices, perforation_pressures
    ):
        allocation_fraction = well_index.get_allocation_fraction(pidx)
        perforation_phase_rates = {
            phase: rate * allocation_fraction
            for phase, rate in resolution.phase_rates.items()
        }
        perforation_states.append(
            PerforationState(
                perforation=perforation,
                cell_index=pidx.cell_index,
                flowing_pressure=float(p_wf),
                phase_rates=perforation_phase_rates,
            )
        )

    return WellState(
        well_name=well.name,
        is_open=True,
        active_control=active_control,
        bhp=resolution.bhp,
        perforation_states=tuple(perforation_states),
        phase_rates=resolution.phase_rates,
        active_limit=resolution.active_limit,
    )
