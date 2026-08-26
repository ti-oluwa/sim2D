"""Top-level well-control resolution: one well's row, one call."""

import typing

from bores.errors import ValidationError
from bores.typing import Integer, NumberArray, OneDimension
from bores.wells.compile import (
    CompiledWellSystem,
    InjectorControlModeTag,
    ProducerControlModeTag,
    WellKind,
)
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellBoreModel
from bores.wells.resolution.compiled import (
    CompiledControlResolverSpec,
    CompiledWellResolution,
    build_perforation_workspace,
)
from bores.wells.resolution.limits import apply_limits
from bores.wells.resolution.solvers import (
    ALL_PHASES,
    bisect_bhp,
    compute_phase_rates,
    compute_tubing_head_pressure,
    get_default_pressure_bracket,
    phase_mask,
    solve_injector_bhp_mode,
    solve_injector_rate_mode,
    solve_producer_bhp_mode,
    solve_producer_rate_mode,
)
from bores.wells.states import ConnectionSample

__all__ = ["resolve_control"]


def resolve_control(
    *,
    compiled_system: CompiledWellSystem,
    well_row: Integer,
    wellbore: WellBoreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    resolution: CompiledWellResolution,
    resolver_spec: CompiledControlResolverSpec,
    surface_fluid_properties: SurfaceFluidProperties | None = None,
) -> None:
    """
    Resolves one well's control and writes the result into `resolution`'s
    row at `well_row`.

    A `PENDING` well (`compiled_system.schedule_statuses[well_row] == 0`)
    or an `UNSET`-mode well is left untouched (still `NaN`/`UNSET_INT`/`0`
    from `compile_well_resolution`) as there's nothing to resolve.

    :param compiled_system: The compiled well system.
    :param well_row: Which well, by row index into `compiled_system`.
    :param wellbore: Hydraulics correlation for this well.
    :param connection_samples: Reservoir samples for this well's active,
        open connections, in the same order `compiled_system.perforations`'
        rows for this well appear (after filtering to
        `completion_statuses == 1` and `schedule_statuses == 1`).
    :param resolution: The system-wide `CompiledWellResolution` to update
        in place - never reallocated by this function.
    :param resolver_spec: Solver tunables.
    :param surface_fluid_properties: Required to compute THP, or to
        resolve a `THP`-mode control, or to check a `THPLimit`. Omit if
        none of those apply to this well.
    :raises ValidationError: If `control_modes[well_row]` isn't a
        recognized tag for this well's `well_kinds[well_row]`.
    """
    if compiled_system.schedule_statuses[well_row] == 0:
        return

    controls = compiled_system.controls
    control_mode = controls.control_modes[well_row]
    well_kind = controls.well_kinds[well_row]
    is_injector = well_kind == WellKind.INJECTOR
    reference_depth = compiled_system.reference_depths[well_row]

    perforations = compiled_system.perforations
    perf_start, perf_end = (
        perforations.well_offsets[well_row],
        perforations.well_offsets[well_row + 1],
    )
    active_open = [
        i
        for i in range(perf_start, perf_end)
        if perforations.completion_statuses[i] == 1 and perforations.schedule_statuses[i] == 1
    ]
    if not active_open or len(active_open) != len(connection_samples):
        raise ValidationError(
            f"`resolve_control`: `well_row` {well_row} has {len(active_open)} "
            f"active/open connections but was given {len(connection_samples)} "
            "connection_samples - these must match 1:1."
        )

    workspace = build_perforation_workspace(
        well_indices=typing.cast(
            NumberArray[OneDimension], perforations.well_indices[active_open]
        ),
        representative_depths=typing.cast(
            NumberArray[OneDimension], perforations.representative_depths[active_open]
        ),
        inclinations_from_vertical=typing.cast(
            NumberArray[OneDimension],
            perforations.inclinations_from_vertical[active_open],
        ),
        connection_samples=connection_samples,
    )

    if is_injector:
        if control_mode == InjectorControlModeTag.UNSET:
            return
        injected_phase = controls.injected_phases[well_row]
        relevant_phases = phase_mask(injected_phase)
        if control_mode in (
            InjectorControlModeTag.RATE,
            InjectorControlModeTag.RESV,
        ):
            bhp, phase_rates = solve_injector_rate_mode(
                control_mode=control_mode,
                target_rate=controls.target_rates[well_row],
                injected_phase=injected_phase,
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                resolver_spec=resolver_spec,
            )
        elif control_mode == InjectorControlModeTag.BHP:
            bhp, phase_rates = solve_injector_bhp_mode(
                target_bhp=controls.target_bhps[well_row],
                injected_phase=injected_phase,
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                resolver_spec=resolver_spec,
            )
        elif control_mode == InjectorControlModeTag.THP:
            if surface_fluid_properties is None:
                raise ValidationError("A THP-mode injector requires `surface_fluid_properties`.")
            min_pressure, max_pressure = get_default_pressure_bracket(
                connection_samples, is_injector=True, resolver_spec=resolver_spec
            )
            bhp, _, _ = bisect_bhp(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                relevant_phases=relevant_phases,
                is_injector=True,
                target=controls.target_thps[well_row],
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                resolver_spec=resolver_spec,
                metric="thp",
                surface_fluid_properties=surface_fluid_properties,
            )
            phase_rates = compute_phase_rates(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                reference_pressure=bhp,
                relevant_phases=relevant_phases,
                is_injector=True,
                resolver_spec=resolver_spec,
            )
        elif control_mode == InjectorControlModeTag.GRUP:
            raise ValidationError(
                f"Well row {well_row} is under GRUP control - resolve group "
                "allocation (wells.resolution.allocation) into a concrete "
                "rate/BHP target before calling resolve_control."
            )
        else:
            raise ValidationError(f"Unknown InjectorControlModeTag: {control_mode!r}.")
    else:
        if control_mode == ProducerControlModeTag.UNSET:
            return

        relevant_phases = ALL_PHASES
        if control_mode in (
            ProducerControlModeTag.ORAT,
            ProducerControlModeTag.WRAT,
            ProducerControlModeTag.GRAT,
            ProducerControlModeTag.LRAT,
            ProducerControlModeTag.RESV,
        ):
            bhp, phase_rates = solve_producer_rate_mode(
                control_mode=control_mode,
                target_rate=controls.target_rates[well_row],
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                resolver_spec=resolver_spec,
            )
        elif control_mode == ProducerControlModeTag.BHP:
            bhp, phase_rates = solve_producer_bhp_mode(
                target_bhp=controls.target_bhps[well_row],
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                resolver_spec=resolver_spec,
            )
        elif control_mode == ProducerControlModeTag.THP:
            if surface_fluid_properties is None:
                raise ValidationError("A THP-mode producer requires `surface_fluid_properties`.")
            min_pressure, max_pressure = get_default_pressure_bracket(
                connection_samples, is_injector=False, resolver_spec=resolver_spec
            )
            bhp, _, _ = bisect_bhp(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                relevant_phases=relevant_phases,
                is_injector=False,
                target=controls.target_thps[well_row],
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                resolver_spec=resolver_spec,
                metric="thp",
                surface_fluid_properties=surface_fluid_properties,
            )
            phase_rates = compute_phase_rates(
                wellbore=wellbore,
                reference_depth=reference_depth,
                workspace=workspace,
                connection_samples=connection_samples,
                reference_pressure=bhp,
                relevant_phases=relevant_phases,
                is_injector=False,
                resolver_spec=resolver_spec,
            )
        elif control_mode == ProducerControlModeTag.GRUP:
            raise ValidationError(
                f"Well row {well_row} is under GRUP control - resolve group "
                "allocation (wells.resolution.allocation) into a concrete "
                "rate/BHP target before calling resolve_control."
            )
        else:
            raise ValidationError(f"Unknown ProducerControlModeTag: {control_mode!r}.")

    limits = controls.limits
    limits_start, limits_end = (
        limits.well_offsets[well_row],
        limits.well_offsets[well_row + 1],
    )
    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples, is_injector=is_injector, resolver_spec=resolver_spec
    )
    bhp, phase_rates, active_limit_row, economic_shutin = apply_limits(
        limits=limits,
        limits_start=limits_start,
        limits_end=limits_end,
        wellbore=wellbore,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        bhp=bhp,
        phase_rates=phase_rates,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        resolver_spec=resolver_spec,
        surface_fluid_properties=surface_fluid_properties,
    )

    resolution.bhps[well_row] = bhp
    resolution.oil_rates[well_row] = phase_rates.oil
    resolution.water_rates[well_row] = phase_rates.water
    resolution.gas_rates[well_row] = phase_rates.gas
    resolution.active_limit_rows[well_row] = active_limit_row
    resolution.economic_shutins[well_row] = 1 if economic_shutin else 0

    if surface_fluid_properties is not None:
        resolution.thps[well_row] = compute_tubing_head_pressure(
            wellbore=wellbore,
            reference_depth=reference_depth,
            reference_pressure=bhp,
            phase_rates=phase_rates,
            surface_fluid_properties=surface_fluid_properties,
            is_injector=is_injector,
        )
    return None
