"""
Secondary-limit enforcement for a resolved well control.

`apply_limits` is called once, by `wells.control.engine.resolve_control`,
after a mode solver in `wells.control.solvers` has already produced a
*nominal* `ControlResolution` (the well operating exactly at its primary
target, ignoring `control.limits` entirely).

This module's job is simple as is as follows:

Given a nominal resolution, decide whether any `Limit` in `control.limits`
is violated, and if so, which one actually governs.

**"Most restrictive wins", not "first in list wins":** every `Limit` type
here (`BHPLimit`, `RateLimit`, `THPLimit`) ultimately reduces to "the BHP
that exactly satisfies this limit". A scalar on the same axis every mode
solver already searches. Because total rate and THP are both monotonic in
BHP (see `wells.control.solvers.bisect_bhp`), the *single* most extreme
required BHP among every violated limit automatically satisfies every
other violated limit too.

A producer's BHP floor set by picking the
highest required BHP is, by definition, at least as restrictive as every
lower BHP a less-binding limit would have asked for (the injector case is
the mirror image: lowest required BHP wins). No outer iteration is needed;
evaluating each limit once and taking the extreme is exact, not a heuristic.
"""

import typing

from bores.errors import ValidationError
from bores.typing import FluidPhase, Number
from bores.wells.base import Well
from bores.wells.controls import BHPLimit, Limit, RateLimit, THPLimit, WellControl
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellboreModel
from bores.wells.perforations import PerforationIndex
from bores.wells.resolution.base import ControlResolution, ControlResolverSpec
from bores.wells.resolution.solvers import (
    RATE_QUANTITY_PHASES,
    bisect_bhp,
    compute_full_phase_rates_at,
)
from bores.wells.states import ConnectionSample

__all__ = ["apply_limits"]


def _bhp_bound(
    limit: BHPLimit, resolution: ControlResolution, *, is_injector: bool
) -> typing.Optional[Number]:
    """Bounding BHP if `limit` is violated by `resolution.bhp`, else `None`."""
    if is_injector:
        if limit.max_value is not None and resolution.bhp > limit.max_value:
            return limit.max_value
        return None
    if limit.min_value is not None and resolution.bhp < limit.min_value:
        return limit.min_value
    return None


def _rate_bound(
    limit: RateLimit,
    resolution: ControlResolution,
    *,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore_model: WellboreModel,
    is_injector: bool,
    min_pressure: Number,
    max_pressure: Number,
    resolver_spec: ControlResolverSpec,
) -> typing.Optional[Number]:
    """
    Bounding BHP if `limit` is violated by `resolution.phase_rates`, else `None`.

    Found by bisecting BHP against `limit.max_value` as a rate target.
    """
    quantity_phases = RATE_QUANTITY_PHASES[limit.quantity]
    current = sum(resolution.phase_rates.get(phase, 0.0) for phase in quantity_phases)
    if current <= limit.max_value:
        return None

    bound_bhp, _, _ = bisect_bhp(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        relevant_phases=quantity_phases,
        is_injector=is_injector,
        target=limit.max_value,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        resolver_spec=resolver_spec,
        metric="rate",
    )
    return bound_bhp


def _thp_bound(
    limit: THPLimit,
    resolution: ControlResolution,
    *,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore_model: WellboreModel,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    min_pressure: Number,
    max_pressure: Number,
    resolver_spec: ControlResolverSpec,
    surface_fluid_properties: SurfaceFluidProperties,
) -> typing.Optional[Number]:
    """
    Bounding BHP if `limit` is violated by `resolution`'s implied THP, else `None`.

    Checked in the forward direction only (candidate BHP ->
    `wellbore_model.tubing_head_pressure`), bisected against
    `limit.min_value`/`max_value` as a target.
    """
    current_thp = wellbore_model.tubing_head_pressure(
        well,
        resolution.bhp,
        resolution.phase_rates,
        surface_fluid_properties,
        is_injector,
    )
    if is_injector:
        violated = limit.max_value is not None and current_thp > limit.max_value
        target = limit.max_value
    else:
        violated = limit.min_value is not None and current_thp < limit.min_value
        target = limit.min_value
    if not violated:
        return None

    assert target is not None
    bound_bhp, _, _ = bisect_bhp(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        target=target,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        resolver_spec=resolver_spec,
        metric="thp",
        surface_fluid_properties=surface_fluid_properties,
    )
    return bound_bhp


def apply_limits(
    *,
    control: WellControl,
    well: Well,
    perforation_indices: typing.Sequence[PerforationIndex],
    connection_samples: typing.Sequence[ConnectionSample],
    wellbore_model: WellboreModel,
    relevant_phases: typing.Sequence[FluidPhase],
    is_injector: bool,
    resolution: ControlResolution,
    min_pressure: Number,
    max_pressure: Number,
    resolver_spec: ControlResolverSpec,
    surface_fluid_properties: typing.Optional[SurfaceFluidProperties] = None,
) -> ControlResolution:
    """
    Evaluate every `Limit` in `control.limits` against `resolution`. If one or
    more are violated, re-resolve at whichever implies the most
    restrictive BHP and return that as the governing resolution with
    `active_limit` set to whichever `Limit` produced it (ties broken by
    `control.limits` order).

    Returns `resolution` unchanged if none are violated.

    Called for every control mode (rate and BHP alike) as a well held at a
    fixed BHP can still violate a `RateLimit`/`THPLimit` configured
    alongside that BHP target, so this isn't skipped for BHP-mode resolutions.

    :param relevant_phases: `wells.control.solvers.ALL_PHASES` for a
        producer; `(control.injected_phase,)` for an injector.
    :param surface_fluid_properties: Required if `control.limits` contains a
        `THPLimit`; unused otherwise.
    :raises ValidationError: If `control.limits` contains a `THPLimit` but
        `surface_fluid_properties` wasn't supplied.
    """
    candidates: typing.List[typing.Tuple[Number, Limit]] = []

    for limit in control.limits:
        if isinstance(limit, BHPLimit):
            bound = _bhp_bound(limit, resolution, is_injector=is_injector)
        elif isinstance(limit, RateLimit):
            bound = _rate_bound(
                limit,
                resolution,
                well=well,
                perforation_indices=perforation_indices,
                connection_samples=connection_samples,
                wellbore_model=wellbore_model,
                is_injector=is_injector,
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                resolver_spec=resolver_spec,
            )
        elif isinstance(limit, THPLimit):
            if surface_fluid_properties is None:
                raise ValidationError(
                    f"Well {well.name!r}'s control spec has a THPLimit but no "
                    "`surface_fluid_properties` was supplied to `apply_limits`/"
                    "`resolve_control`. THP limits can't be evaluated without it."
                )

            bound = _thp_bound(
                limit,
                resolution,
                well=well,
                perforation_indices=perforation_indices,
                connection_samples=connection_samples,
                wellbore_model=wellbore_model,
                relevant_phases=relevant_phases,
                is_injector=is_injector,
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                resolver_spec=resolver_spec,
                surface_fluid_properties=surface_fluid_properties,
            )
        else:
            raise ValidationError(f"Unknown Limit type: {type(limit)!r}.")

        if bound is not None:
            candidates.append((bound, limit))

    if not candidates:
        return resolution

    # Most restrictive: highest bounding BHP for a producer (lowest
    # resulting rate), lowest bounding BHP for an injector (lowest
    # resulting injection rate).
    governing_bhp, governing_limit = (
        min(candidates, key=lambda pair: pair[0])
        if is_injector
        else max(candidates, key=lambda pair: pair[0])
    )

    phase_rates = compute_full_phase_rates_at(
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        reference_pressure=governing_bhp,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        resolver_spec=resolver_spec,
    )
    return ControlResolution(
        bhp=governing_bhp, phase_rates=phase_rates, active_limit=governing_limit
    )
