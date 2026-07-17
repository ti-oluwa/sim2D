"""
Well control resolution: target -> per-connection result.

Depends on `wells.data`, `wells.controls`, `wells.state`, `wells.location`,
`wells.hydraulics.WellboreModel`, `wells.index.WellIndex`,
`wells.control.solvers`, `wells.control.limits`.

`resolve_control` operates at connection (`PerforationIndex`) granularity,
via `well_index.perforations` directly.

These two coincide when every open perforation resolves to exactly one
cell; they don't in general (a perforation spanning multiple cells
produces multiple `PerforationIndex` entries in `well_index.perforations`
but only one entry in `well.open_perforations`). `connection_samples`
passed to `resolve_control`/`build_well_state` must be in
`well_index.perforations` order. One `ConnectionSample` per connected
cell, not one per `Perforation` as both functions validate this length and
raise rather than silently truncating/zipping the shorter sequence.
"""

import typing

import attrs

from bores.errors import ValidationError
from bores.wells.base import Well
from bores.wells.controls import (
    InjectorControl,
    InjectorControlMode,
    ProducerControl,
    ProducerControlMode,
    WellControl,
)
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellboreModel
from bores.wells.indices import WellIndex
from bores.wells.resolution.base import ControlResolution, ControlResolverSpec
from bores.wells.resolution.limits import apply_limits
from bores.wells.resolution.solvers import (
    ALL_PHASES,
    get_default_pressure_bracket,
    solve_injector_bhp_mode,
    solve_injector_rate_mode,
    solve_producer_bhp_mode,
    solve_producer_rate_mode,
)
from bores.wells.states import ConnectionSample, PerforationState, WellState

__all__ = ["resolve_control", "build_well_state"]

DEFAULT_RESOLVER_SPEC = ControlResolverSpec()


def resolve_control(
    control: WellControl,
    well: Well,
    well_index: WellIndex,
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    *,
    resolver_spec: ControlResolverSpec = DEFAULT_RESOLVER_SPEC,
    surface_fluid_properties: typing.Optional[SurfaceFluidProperties] = None,
) -> ControlResolution:
    """
    Single entry point the simulation loop calls once per well per
    timestep.

    Dispatches to a mode-specific solver
    in (`wells.control.solvers`) based on `type(control)` and `control.mode`, then
    applies `control.limits` (`wells.control.limits.apply_limits`).

    :param control: The well's nominal control (from `WellSchedule`/`WellState`).
        May not be necessarily what ends up governing if a limit fires.
    :param well: Static well data.
    :param well_index: Resolved connection factors (`wells.index`).
    :param wellbore_model: Hydraulics strategy for this well.
    :param connection_samples: Reservoir samples, one per connection,
        `well_index.perforations` order (see module docstring - **not**
        `well.open_perforations` order).
    :param resolver_spec: Solver tunables; `DEFAULT_RESOLVER_SPEC`
        (all-default) if not supplied.
    :param surface_fluid_properties: Required only if `control.limits`
        contains a `THPLimit`.
    :returns: `ControlResolution`.
    :raises NotImplementedError: If `control.mode` is `THP` - VFP-based THP
        *control* resolution is out of scope. A `THPLimit` alongside a
        non-THP mode is fully supported (see `wells.control.limits`) and
        does not hit this.
    :raises ValidationError: If `len(connection_samples) != len(well_index.perforations)`,
        `control` is neither `ProducerControl` nor `InjectorControl`,
        or `control.limits` contains a `THPLimit` with no
        `surface_fluid_properties` supplied.
    """
    perforation_indices = well_index.perforations
    if len(connection_samples) != len(perforation_indices):
        raise ValidationError(
            f"len(connection_samples)={len(connection_samples)} != "
            f"len(well_index.perforations)={len(perforation_indices)} for "
            f"well {well.name!r}. connection_samples must be supplied one "
            "per connection (well_index.perforations order), not one per "
            "well.open_perforations - see module docstring."
        )

    if isinstance(control, ProducerControl):
        if control.mode is ProducerControlMode.THP:
            raise NotImplementedError(
                "ProducerControlMode.THP resolution requires VFP tables; "
                "out of scope. (A THPLimit alongside a non-THP mode is "
                "supported - this error is about the *control mode*.)"
            )
        if control.mode is ProducerControlMode.BHP:
            assert control.target_bhp is not None
            nominal = solve_producer_bhp_mode(
                control.target_bhp,
                well,
                perforation_indices,
                wellbore_model,
                connection_samples,
                resolver_spec,
            )
        else:
            nominal = solve_producer_rate_mode(
                control,
                well,
                perforation_indices,
                wellbore_model,
                connection_samples,
                resolver_spec,
            )
        relevant_phases: typing.Sequence = ALL_PHASES
        is_injector = False

    elif isinstance(control, InjectorControl):
        if control.mode is InjectorControlMode.THP:
            raise NotImplementedError(
                "InjectorControlMode.THP resolution requires VFP tables; "
                "out of scope. (A THPLimit alongside a non-THP mode is "
                "supported - this error is about the *control mode*.)"
            )
        if control.mode is InjectorControlMode.BHP:
            assert control.target_bhp is not None
            nominal = solve_injector_bhp_mode(
                control.target_bhp,
                control.injected_phase,
                well,
                perforation_indices,
                wellbore_model,
                connection_samples,
                resolver_spec,
            )
        else:
            nominal = solve_injector_rate_mode(
                control,
                well,
                perforation_indices,
                wellbore_model,
                connection_samples,
                resolver_spec,
            )
        relevant_phases = (control.injected_phase,)
        is_injector = True

    else:
        raise ValidationError(f"Unknown WellControl type: {type(control)!r}.")

    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples,
        is_injector=is_injector,
        resolver_spec=resolver_spec,
    )
    resolution = apply_limits(
        control=control,
        well=well,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        wellbore_model=wellbore_model,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        resolution=nominal,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        resolver_spec=resolver_spec,
        surface_fluid_properties=surface_fluid_properties,
    )
    if surface_fluid_properties is not None and not resolution.economic_shutin:
        thp = wellbore_model.tubing_head_pressure(
            well,
            reference_pressure=resolution.bhp,
            phase_rates=resolution.phase_rates,
            surface_fluid_properties=surface_fluid_properties,
            is_injector=is_injector,
        )
        resolution = attrs.evolve(resolution, thp=thp)
    return resolution


def build_well_state(
    well: Well,
    resolution: ControlResolution,
    well_index: WellIndex,
    wellbore_model: WellboreModel,
    connection_samples: typing.Sequence[ConnectionSample],
    *,
    active_control: WellControl,
    surface_fluid_properties: typing.Optional[SurfaceFluidProperties] = None,
) -> WellState:
    """
    Assembles the final `WellState` for this timestep from a `ControlResolution`.

    Turns the well-total numbers in `resolution` into
    the per-connection `PerforationState` tuple `WellState` requires. A
    `Perforation` spanning multiple cells produces multiple
    `PerforationState` entries (i.e same `.perforation`, different `.cell_index`).

    :param well: Static well data.
    :param resolution: Output of `resolve_control`.
    :param well_index: Resolved connection factors (allocation weights).
    :param wellbore_model: Used to get each connection's flowing pressure
        at `resolution.bhp`.
    :param connection_samples: Reservoir samples, `well_index.perforations`
        order, same contract as `resolve_control`.
    :param active_control: The control actually used to produce `resolution`.
        As this may differ from the well's nominal control if a limit was hit.
        The caller is responsible for passing the *effective* one here).
    :returns: Complete `WellState`, `is_open=True`.
    :raises ValidationError: If `len(connection_samples) != len(well_index.perforations)`.
    """
    if well.unit_system != well_index.unit_system:
        raise ValidationError(
            f"Well `unit_system` ({well.unit_system.value}) != WellIndex "
            f"`unit_system` ({well_index.unit_system.value})."
        )

    perforation_indices = well_index.perforations
    if len(connection_samples) != len(perforation_indices):
        raise ValidationError(
            f"len(connection_samples)={len(connection_samples)} != "
            f"len(well_index.perforations)={len(perforation_indices)} for "
            f"well {well.name!r}."
        )

    connection_pressures = wellbore_model.perforation_pressures(
        well,
        resolution.bhp,
        phase_rates=resolution.phase_rates,
        perforation_indices=perforation_indices,
        connection_samples=connection_samples,
        is_injector=isinstance(active_control, InjectorControl),
    )

    perforation_states: typing.List[PerforationState] = []
    for pidx, p_wf in zip(perforation_indices, connection_pressures):
        allocation_fraction = well_index.get_allocation_fraction(pidx)
        connection_phase_rates = {
            phase: rate * allocation_fraction
            for phase, rate in resolution.phase_rates.items()
        }
        perforation_states.append(
            PerforationState(
                perforation=pidx.perforation,
                cell_index=pidx.cell_index,
                flowing_pressure=float(p_wf),
                phase_rates=connection_phase_rates,
                unit_system=well.unit_system,
            )
        )
    return WellState(
        well_name=well.name,
        is_open=True,
        active_control=active_control,
        bhp=resolution.bhp,
        perforation_states=(
            () if resolution.economic_shutin else tuple(perforation_states)
        ),
        phase_rates=resolution.phase_rates,
        active_limit=resolution.active_limit,
        thp=resolution.thp,
        unit_system=well.unit_system,
    )
