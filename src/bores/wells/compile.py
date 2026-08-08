"""
Compiled (Structure-of-Arrays) well data for the reservoir solver hot path.

`compile_well_system` flattens a rich `Wells`/`WellControls` pair into a
`CompiledWellSystem`: fixed-shape, numba-`njit`-friendly arrays instead of
`attrs` object graphs, following the same conventions established for
`simulation.blackoil.caches` (`NamedTuple`-based, "create once or reuse
`out`, always return", fixed sentinel/tag columns replacing
`isinstance`/registry polymorphism, CSR for variable-length per-entity
data).

This module does not change well/control *semantics* - every dispatch it
performs (BHP/THP-bound-vs-limit selection, economic-shutin handling,
producer-vs-injector control-mode interpretation) mirrors
`wells.resolution.limits`/`wells.controls` exactly, just re-expressed as
int tag columns a jitted kernel can branch on instead of `isinstance`.
Compiling is a one-time (or per-schedule-event) cost; the arrays it
produces are what the actual hot loop (`wells.resolution.solvers`,
restructured in a later step) consumes.

**Load-once roster**: `compile_well_system` compiles *every* perforation
on *every* well passed to it, regardless of `CompletionStatus`/
`WellStatus` - a `CompletionStatus.SHUT` or `WellStatus.PENDING` row is
still present in the arrays, just tagged accordingly, so that a later
`WELOPEN`/schedule-clock event can flip its status in place without
re-resolving connection geometry (the expensive, non-jittable step -
spatial search, ray-vs-face-plane intersection). `resolve_perforations_indices`/
`resolve_md_perforations_indices` only resolve `well.open_perforations`, so
getting geometry for a currently-`SHUT` perforation means resolving a
"shadow" well with every perforation temporarily forced `CompletionStatus.OPEN`
for that call only, then reading each connection's real status back off
the original (unmodified) perforation object. `PerforationIndex.perforation`
is a direct reference to the exact object passed in
(`resolve_perforations_indices` never copies it - confirmed by reading and
empirically exercising that code), so the shadow perforations are matched
back to their originals by identity (`id(...)`), not position - one
`Perforation` can resolve to several `PerforationIndex` entries (a
multi-cell completion), so a positional zip would misalign.
"""

import enum
import typing

import attrs
import numpy as np

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.typing import (
    FloatArray,
    FluidPhase,
    IntArray,
    Number,
    NumberArray,
    OneDimension,
    Orientation,
    UnitSystem,
)
from bores.wells.base import (
    AnyPerforation,
    CompletionStatus,
    MDPerforation,
    Perforation,
    Well,
    Wells,
    WellStatus,
    WellType,
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
    WellControls,
)
from bores.wells.indices.perforations import PerforationIndex
from bores.wells.indices.wells import build_wells_indices

__all__ = [
    "WellKind",
    "LimitKind",
    "CompiledPerforations",
    "CompiledLimits",
    "CompiledWellControls",
    "CompiledWellSystem",
    "compile_well_system",
]


UNSET_INT = -1
"""Sentinel for an int tag column entry that doesn't apply to that row's `kind`."""


class WellKind(enum.IntEnum):
    """Tag column value for `CompiledWellSystem.well_kind`/`CompiledWellControls.well_kind`."""

    PRODUCER = 0
    INJECTOR = 1


class LimitKind(enum.IntEnum):
    """Tag column value for `CompiledLimits.kind`."""

    BHP = 0
    RATE = 1
    THP = 2
    ECONOMIC = 3


class ProducerControlModeTag(enum.IntEnum):
    """
    Int mirror of `ProducerControlMode`, for `CompiledWellControls.control_mode`
    on a `WellKind.PRODUCER` row.
    """

    ORAT = 0
    WRAT = 1
    GRAT = 2
    LRAT = 3
    RESV = 4
    BHP = 5
    THP = 6
    GRUP = 7
    UNSET = 8


class InjectorControlModeTag(enum.IntEnum):
    """
    Int mirror of `InjectorControlMode`, for `CompiledWellControls.control_mode`
    on a `WellKind.INJECTOR` row.

    Deliberately a *different* tag space from `ProducerControlModeTag` (both
    start at 0) - a kernel must branch on `well_kind` before interpreting
    `control_mode`, exactly as the rich model branches on
    `isinstance(control, ProducerControl)` vs `InjectorControl` before
    reading `control.mode`. Collapsing the two into one shared tag space
    would let a kernel silently misinterpret e.g. `INJECTOR` tag `0`
    (`RATE`) as `PRODUCER` tag `0` (`ORAT`) if the `well_kind` branch were
    ever skipped by mistake; keeping them distinct-but-overlapping-valued
    makes that class of bug loud (wrong physics) rather than silent.
    """

    RATE = 0
    RESV = 1
    BHP = 2
    THP = 3
    GRUP = 4
    UNSET = 5


class RateQuantityTag(enum.IntEnum):
    """Int mirror of `RateQuantity`, for `CompiledLimits.quantity_tag` on a `RATE` row."""

    OIL = 0
    WATER = 1
    GAS = 2
    LIQUID = 3
    RESERVOIR = 4


class EconomicQuantityTag(enum.IntEnum):
    """Int mirror of `EconomicQuantity`, for `CompiledLimits.quantity_tag` on an `ECONOMIC` row."""

    WATER_CUT = 0
    GOR = 1
    WATER_GAS_RATIO = 2


class FluidPhaseTag(enum.IntEnum):
    """Int mirror of `FluidPhase`, for `CompiledWellControls.injected_phase`."""

    OIL = 0
    WATER = 1
    GAS = 2


PRODUCER_MODE_TAG = {
    ProducerControlMode.ORAT: ProducerControlModeTag.ORAT,
    ProducerControlMode.WRAT: ProducerControlModeTag.WRAT,
    ProducerControlMode.GRAT: ProducerControlModeTag.GRAT,
    ProducerControlMode.LRAT: ProducerControlModeTag.LRAT,
    ProducerControlMode.RESV: ProducerControlModeTag.RESV,
    ProducerControlMode.BHP: ProducerControlModeTag.BHP,
    ProducerControlMode.THP: ProducerControlModeTag.THP,
    ProducerControlMode.GRUP: ProducerControlModeTag.GRUP,
    ProducerControlMode.UNSET: ProducerControlModeTag.UNSET,
}
INJECTOR_MODE_TAG = {
    InjectorControlMode.RATE: InjectorControlModeTag.RATE,
    InjectorControlMode.RESV: InjectorControlModeTag.RESV,
    InjectorControlMode.BHP: InjectorControlModeTag.BHP,
    InjectorControlMode.THP: InjectorControlModeTag.THP,
    InjectorControlMode.GRUP: InjectorControlModeTag.GRUP,
    InjectorControlMode.UNSET: InjectorControlModeTag.UNSET,
}
RATE_QUANTITY_TAG = {
    RateQuantity.OIL: RateQuantityTag.OIL,
    RateQuantity.WATER: RateQuantityTag.WATER,
    RateQuantity.GAS: RateQuantityTag.GAS,
    RateQuantity.LIQUID: RateQuantityTag.LIQUID,
    RateQuantity.RESERVOIR: RateQuantityTag.RESERVOIR,
}
ECONOMIC_QUANTITY_TAG = {
    EconomicQuantity.WATER_CUT: EconomicQuantityTag.WATER_CUT,
    EconomicQuantity.GOR: EconomicQuantityTag.GOR,
    EconomicQuantity.WATER_GAS_RATIO: EconomicQuantityTag.WATER_GAS_RATIO,
}
FLUID_PHASE_TAG = {
    FluidPhase.OIL: FluidPhaseTag.OIL,
    FluidPhase.WATER: FluidPhaseTag.WATER,
    FluidPhase.GAS: FluidPhaseTag.GAS,
}


class CompiledPerforations(typing.NamedTuple):
    """
    Flat, CSR-indexed-by-well perforation/connection data.

    Row order is established once by `compile_well_system` at load-once
    time from every well's full (open + shut) perforation list, in
    well-then-perforation-then-cell order; never reordered afterward -
    schedule-driven status changes mutate the parallel tag arrays in
    place instead. CSR convention matches `grids.base`'s
    `cell_face_offsets`/`cell_face_indices`: well `w`'s rows are
    `well_offsets[w]:well_offsets[w + 1]`.
    """

    well_offsets: IntArray
    """Shape `(n_wells + 1,)`."""

    cell_index: IntArray
    """Shape `(n_rows,)`. Grid cell each connection resolves to."""

    well_index: FloatArray
    """
    Shape `(n_rows,)`. Precomputed connection factor (`WellIndex`
    equivalent). Purely geometric/static-rock-property derived (Peaceman
    or equivalent-radius, per `wells.indices`) - computed once here, not
    re-derived per timestep, since neither well geometry nor absolute
    permeability changes during a run.
    """

    wellbore_radius: FloatArray
    """Shape `(n_rows,)`."""

    skin: FloatArray
    """Shape `(n_rows,)`."""

    partial_penetration_fraction: FloatArray
    """Shape `(n_rows,)`."""

    representative_depth: FloatArray
    """Shape `(n_rows,)`. For the hydrostatic term in wellbore pressure-drop."""

    inclination_from_vertical: FloatArray
    """Shape `(n_rows,)`, radians. For `wells.hydraulics`."""

    completion_status: IntArray
    """
    Shape `(n_rows,)`. `1` = `CompletionStatus.OPEN`, `0` = `SHUT`. Mutable
    in place by a `WELOPEN`/`WELTARG`-driven event on a single connection -
    flipping this never requires re-resolving connection geometry.
    """

    schedule_status: IntArray
    """
    Shape `(n_rows,)`. `1` = `WellStatus.ACTIVE`, `0` = `PENDING`. See
    `bores.wells.base.WellStatus`. A row with `schedule_status == 0` is
    skipped entirely by every hot-loop kernel, same as a row with
    `completion_status == 0`, but the two are independent: a `PENDING`
    connection can carry either `completion_status`, and once flipped
    `ACTIVE` its `completion_status` still governs.
    """

    saturation_region: IntArray
    """Shape `(n_rows,)`. Per-connection SATNUM override; `-1` means "use
    the cell's own region"."""


class CompiledLimits(typing.NamedTuple):
    """
    Flat, CSR-indexed-by-well secondary-limit data, aligned 1:1 with
    `CompiledWellControls`' `n_wells` rows (`well_offsets` has
    `n_wells + 1` entries regardless of whether every well has limits). A
    well with no limits has an empty row range
    (`well_offsets[w] == well_offsets[w + 1]`).
    """

    well_offsets: IntArray
    """Shape `(n_wells + 1,)`."""

    kind: IntArray
    """Shape `(n_rows,)`. `LimitKind` value."""

    quantity_tag: IntArray
    """
    Shape `(n_rows,)`. `RateQuantityTag` value when `kind == LimitKind.RATE`,
    `EconomicQuantityTag` value when `kind == LimitKind.ECONOMIC`,
    `UNSET_INT` for `BHP`/`THP` (no quantity axis).
    """

    min_value: FloatArray
    """Shape `(n_rows,)`. `NaN` where not applicable - `RateLimit`/
    `EconomicLimit` never set this; a `BHPLimit`/`THPLimit` with only a
    ceiling set leaves it `NaN` too."""

    max_value: FloatArray
    """Shape `(n_rows,)`. `NaN` where not applicable - a `BHPLimit`/
    `THPLimit` with only a floor set leaves it `NaN`."""


class CompiledWellControls(typing.NamedTuple):
    """
    Flat, one-row-per-well control-target data, positionally aligned with
    `CompiledWellSystem`'s `n_wells` rows.
    """

    well_kind: IntArray
    """Shape `(n_wells,)`. `WellKind` value."""

    control_mode: IntArray
    """
    Shape `(n_wells,)`. `ProducerControlModeTag` value on a `PRODUCER` row,
    `InjectorControlModeTag` value on an `INJECTOR` row - a kernel must
    branch on `well_kind` before interpreting this column (see
    `InjectorControlModeTag`'s docstring). `UNSET` (in whichever tag space
    applies) for a well with no control record yet - the common case for a
    `WellStatus.PENDING` well under the load-once roster.
    """

    injected_phase: IntArray
    """Shape `(n_wells,)`. `FluidPhaseTag` value on an `INJECTOR` row with
    a control set; `UNSET_INT` for a `PRODUCER` row or an uncontrolled well."""

    target_rate: FloatArray
    """Shape `(n_wells,)`. `NaN` where unset. The single active-mode-selected
    rate target (`ProducerControl`/`InjectorControl` keep separate
    `ORAT`/`WRAT`/`GRAT`/`LRAT`/`RESV` fields in the rich model so
    switching `mode` doesn't require re-supplying targets - the compiled
    row only needs the one `control_mode` currently selects, so
    `compile_well_system` resolves that selection once here rather than
    carrying all five)."""

    target_bhp: FloatArray
    """Shape `(n_wells,)`. `NaN` where unset."""

    target_thp: FloatArray
    """Shape `(n_wells,)`. `NaN` where unset."""

    efficiency_factor: FloatArray
    """Shape `(n_wells,)`. `1.0` for an uncontrolled well (matches
    `WellControl`'s own default)."""

    guide_rate: FloatArray
    """Shape `(n_wells,)`. `NaN` where unset (equal-weight group-allocation
    fallback, matching the rich model)."""

    limits: CompiledLimits
    """CSR limits table; `limits.well_offsets` has `n_wells + 1` entries,
    1:1 aligned with these rows."""


class CompiledWellSystem(typing.NamedTuple):
    """
    Top-level compiled bundle for the wells hot path.

    Owns its own name<->index map (`names`), mirroring how `Grid` owns its
    own connectivity info rather than a parent object owning it on
    `Grid`'s behalf. `names` and `unit_system` are metadata for the
    Python-side caller only - a numba kernel should take `.perforations`/
    `.controls` (or their individual arrays) directly, never this whole
    `NamedTuple`; a `Tuple[str, ...]` and a plain `Enum` field are not
    numba-safe the way the all-numeric-array `CompiledPerforations`/
    `CompiledWellControls` sub-structures are.
    """

    names: typing.Tuple[str, ...]
    """Shape `(n_wells,)` (as a plain tuple). Well name per row index -
    the only place a well's name lives in the compiled form; every other
    structure here is purely positional."""

    well_kind: IntArray
    """
    Shape `(n_wells,)`. Duplicated from `controls.well_kind` for kernels
    that only need `perforations`/well-identity and shouldn't have to
    thread `controls` through as well - kept in lockstep with
    `controls.well_kind` by construction (`compile_well_system` writes
    both from the same source), never independently mutated afterward.
    """

    schedule_status: IntArray
    """
    Shape `(n_wells,)`. Well-level `WellStatus` value (`1` = `ACTIVE`, `0`
    = `PENDING`). Independent of `perforations.schedule_status`: a
    `PENDING` well's perforations are also all `PENDING` at compile time,
    but a well flipping `ACTIVE` and one of its individually-`PENDING`
    perforations flipping `ACTIVE` later (a `COMPDAT` reissue adding a
    completion) are two separate events.
    """

    reference_depth: FloatArray
    """Shape `(n_wells,)`."""

    perforations: CompiledPerforations
    controls: CompiledWellControls
    unit_system: UnitSystem
    """Python-side only; never read inside a jitted kernel."""


def get_well_status_tag(status: WellStatus) -> int:
    return 1 if status is WellStatus.ACTIVE else 0


def get_completion_status_tag(status: CompletionStatus) -> int:
    return 1 if status is CompletionStatus.OPEN else 0


def _resolve_all_perforation_geometry(
    grid: Grid,
    well: Well,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    **resolve_kwargs: typing.Any,
) -> typing.Tuple[
    typing.Tuple[PerforationIndex, ...], typing.Dict[int, AnyPerforation]
]:
    """
    Resolve connection geometry (+ well index) for *every* perforation on
    `well`, open or shut, via the "shadow well" trick described in this
    module's docstring.

    :param grid: Grid to resolve against.
    :param well: The well to resolve. Not modified.
    :param permeabilities: Forwarded to `build_wells_indices`.
    :param resolve_kwargs: Forwarded to `build_wells_indices`
        (`regime_constant`, `net_to_gross`, or perforation-resolution
        kwargs it passes through).
    :returns: `(perforation_indices, original_by_id)` - every resolved
        `PerforationIndex` for this well (using the shadow, forced-`OPEN`
        perforation objects), and a mapping from `id(shadow_perforation)`
        to the corresponding *original* `Perforation`/`MDPerforation`, for
        reading back the real `status`/`schedule_status`/
        `saturation_region` that resolution itself doesn't need.
    :raises ValidationError: Propagated from `build_wells_indices` for a
        dangling completion (a perforation whose interval overlaps no
        cell at all - a real deck/geometry error, not something forcing
        `status=OPEN` can paper over).
    """
    shadow_perforations = tuple(
        attrs.evolve(perforation, status=CompletionStatus.OPEN)
        for perforation in well.perforations
    )
    original_by_id: typing.Dict[int, AnyPerforation] = {
        id(shadow): original
        for shadow, original in zip(shadow_perforations, well.perforations)
    }
    shadow_well = attrs.evolve(well, perforations=shadow_perforations)
    shadow_wells = Wells(wells={well.name: shadow_well})

    result = build_wells_indices(grid, shadow_wells, permeabilities, **resolve_kwargs)
    well_index = result[well.name]
    return well_index.perforations, original_by_id


def compile_perforations(
    names: typing.Sequence[str],
    wells: Wells,
    grid: Grid,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    **resolve_kwargs: typing.Any,
) -> CompiledPerforations:
    """
    Build `CompiledPerforations` for every well in `names`, in order.

    :param names: Well names, in the row order `CompiledWellSystem` will use.
    :param wells: Source `Wells`.
    :param grid: Forwarded to `_resolve_all_perforation_geometry`.
    :param permeabilities: Forwarded to `_resolve_all_perforation_geometry`.
    :param resolve_kwargs: Forwarded to `_resolve_all_perforation_geometry`.
    :returns: `CompiledPerforations` with one row per (well, perforation,
        cell) triple, `well_offsets`-CSR-indexed in `names` order.
    """
    well_offsets = [0]
    cell_index: typing.List[int] = []
    well_index: typing.List[Number] = []
    wellbore_radius: typing.List[Number] = []
    skin: typing.List[Number] = []
    partial_penetration_fraction: typing.List[Number] = []
    representative_depth: typing.List[Number] = []
    inclination_from_vertical: typing.List[Number] = []
    completion_status: typing.List[int] = []
    schedule_status: typing.List[int] = []
    saturation_region: typing.List[int] = []

    for name in names:
        well = wells[name]
        perforation_indices, original_by_id = _resolve_all_perforation_geometry(
            grid, well, permeabilities, **resolve_kwargs
        )
        for connection in perforation_indices:
            original = original_by_id[id(connection.perforation)]
            cell_index.append(connection.cell_index)
            well_index.append(
                connection.well_index if connection.well_index is not None else np.nan
            )
            wellbore_radius.append(original.wellbore_radius)
            skin.append(original.skin)
            partial_penetration_fraction.append(connection.partial_penetration_fraction)
            representative_depth.append(connection.representative_depth)
            inclination_from_vertical.append(connection.inclination_from_vertical)
            completion_status.append(get_completion_status_tag(original.status))
            schedule_status.append(get_well_status_tag(original.schedule_status))
            saturation_region.append(
                original.saturation_region
                if original.saturation_region is not None
                else UNSET_INT
            )
        well_offsets.append(len(cell_index))

    return CompiledPerforations(
        well_offsets=np.asarray(well_offsets, dtype=np.int64),
        cell_index=np.asarray(cell_index, dtype=np.int64),
        well_index=np.asarray(well_index, dtype=np.float64),
        wellbore_radius=np.asarray(wellbore_radius, dtype=np.float64),
        skin=np.asarray(skin, dtype=np.float64),
        partial_penetration_fraction=np.asarray(
            partial_penetration_fraction, dtype=np.float64
        ),
        representative_depth=np.asarray(representative_depth, dtype=np.float64),
        inclination_from_vertical=np.asarray(
            inclination_from_vertical, dtype=np.float64
        ),
        completion_status=np.asarray(completion_status, dtype=np.int64),
        schedule_status=np.asarray(schedule_status, dtype=np.int64),
        saturation_region=np.asarray(saturation_region, dtype=np.int64),
    )


def compile_limits(
    limits: typing.Sequence[Limit],
) -> typing.Tuple[
    typing.List[int], typing.List[int], typing.List[Number], typing.List[Number]
]:
    """
    Flatten one well's `control.limits` tuple into parallel
    `kind`/`quantity_tag`/`min_value`/`max_value` lists.

    :param limits: `WellControl.limits` for one well.
    :returns: `(kind, quantity_tag, min_value, max_value)`, all the same
        length as `limits`.
    :raises ValidationError: If `limits` contains a `Limit` subtype this
        module doesn't know about - mirrors `apply_limits`'s own
        `else: raise ValidationError(...)` for an unknown `Limit` type.
    """
    kind: typing.List[int] = []
    quantity_tag: typing.List[int] = []
    min_value: typing.List[Number] = []
    max_value: typing.List[Number] = []

    for limit in limits:
        if isinstance(limit, BHPLimit):
            kind.append(LimitKind.BHP)
            quantity_tag.append(UNSET_INT)
            min_value.append(limit.min_value if limit.min_value is not None else np.nan)
            max_value.append(limit.max_value if limit.max_value is not None else np.nan)
        elif isinstance(limit, THPLimit):
            kind.append(LimitKind.THP)
            quantity_tag.append(UNSET_INT)
            min_value.append(limit.min_value if limit.min_value is not None else np.nan)
            max_value.append(limit.max_value if limit.max_value is not None else np.nan)
        elif isinstance(limit, RateLimit):
            kind.append(LimitKind.RATE)
            quantity_tag.append(RATE_QUANTITY_TAG[limit.quantity])
            min_value.append(np.nan)
            max_value.append(limit.max_value)
        elif isinstance(limit, EconomicLimit):
            kind.append(LimitKind.ECONOMIC)
            quantity_tag.append(ECONOMIC_QUANTITY_TAG[limit.quantity])
            min_value.append(np.nan)
            max_value.append(limit.max_value)
        else:
            raise ValidationError(f"Unknown Limit type: {type(limit)!r}.")

    return kind, quantity_tag, min_value, max_value


def compile_well_controls(
    names: typing.Sequence[str], controls: WellControls, wells: Wells
) -> CompiledWellControls:
    """
    Build `CompiledWellControls` for every well in `names`, in order.

    A well with no entry in `controls` (a `WellStatus.PENDING` well under
    the load-once roster, most commonly) gets an `UNSET`-mode row with
    every target `NaN`, `efficiency_factor=1.0` (the rich-model default),
    and no limits - not an error, since an uncontrolled well is a normal,
    expected roster state, not a data problem.

    :param names: Well names, in row order.
    :param controls: Source `WellControls`.
    :param wells: Source `Wells` - only `well.well_type` is read, to set
        `well_kind` even for a well with no control record yet.
    :returns: `CompiledWellControls`.
    """
    well_kind: typing.List[int] = []
    control_mode: typing.List[int] = []
    injected_phase: typing.List[int] = []
    target_rate: typing.List[Number] = []
    target_bhp: typing.List[Number] = []
    target_thp: typing.List[Number] = []
    efficiency_factor: typing.List[Number] = []
    guide_rate: typing.List[Number] = []

    limits_well_offsets = [0]
    limits_kind: typing.List[int] = []
    limits_quantity_tag: typing.List[int] = []
    limits_min_value: typing.List[Number] = []
    limits_max_value: typing.List[Number] = []

    for name in names:
        well = wells[name]
        control = controls.get(name)
        kind = (
            WellKind.INJECTOR
            if well.well_type is WellType.INJECTOR
            else WellKind.PRODUCER
        )
        well_kind.append(kind)

        if control is None:
            control_mode.append(
                ProducerControlModeTag.UNSET
                if kind is WellKind.PRODUCER
                else InjectorControlModeTag.UNSET
            )
            injected_phase.append(UNSET_INT)
            target_rate.append(np.nan)
            target_bhp.append(np.nan)
            target_thp.append(np.nan)
            efficiency_factor.append(1.0)
            guide_rate.append(np.nan)
            limits_well_offsets.append(len(limits_kind))
            continue

        efficiency_factor.append(control.efficiency_factor)
        guide_rate.append(
            control.guide_rate if control.guide_rate is not None else np.nan
        )
        target_bhp.append(
            control.target_bhp if control.target_bhp is not None else np.nan
        )
        target_thp.append(
            control.target_thp if control.target_thp is not None else np.nan
        )
        target_rate.append(
            control.target_rate if control.target_rate is not None else np.nan
        )

        if isinstance(control, InjectorControl):
            control_mode.append(INJECTOR_MODE_TAG[control.mode])
            injected_phase.append(FLUID_PHASE_TAG[control.injected_phase])
        elif isinstance(control, ProducerControl):
            control_mode.append(PRODUCER_MODE_TAG[control.mode])
            injected_phase.append(UNSET_INT)
        else:
            raise ValidationError(f"Unknown WellControl type: {type(control)!r}.")

        kind_list, quantity_list, min_list, max_list = compile_limits(control.limits)
        limits_kind.extend(kind_list)
        limits_quantity_tag.extend(quantity_list)
        limits_min_value.extend(min_list)
        limits_max_value.extend(max_list)
        limits_well_offsets.append(len(limits_kind))

    limits = CompiledLimits(
        well_offsets=np.asarray(limits_well_offsets, dtype=np.int64),
        kind=np.asarray(limits_kind, dtype=np.int64),
        quantity_tag=np.asarray(limits_quantity_tag, dtype=np.int64),
        min_value=np.asarray(limits_min_value, dtype=np.float64),
        max_value=np.asarray(limits_max_value, dtype=np.float64),
    )
    return CompiledWellControls(
        well_kind=np.asarray(well_kind, dtype=np.int64),
        control_mode=np.asarray(control_mode, dtype=np.int64),
        injected_phase=np.asarray(injected_phase, dtype=np.int64),
        target_rate=np.asarray(target_rate, dtype=np.float64),
        target_bhp=np.asarray(target_bhp, dtype=np.float64),
        target_thp=np.asarray(target_thp, dtype=np.float64),
        efficiency_factor=np.asarray(efficiency_factor, dtype=np.float64),
        guide_rate=np.asarray(guide_rate, dtype=np.float64),
        limits=limits,
    )


def compile_well_system(
    wells: Wells,
    controls: WellControls,
    grid: Grid,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    **resolve_kwargs: typing.Any,
) -> CompiledWellSystem:
    """
    Compile a rich `Wells`/`WellControls` pair into a `CompiledWellSystem`.

    Every well `wells` contains is included, regardless of
    `Well.schedule_status` - this is the "load full roster once" step
    (plan §0); a `WellStatus.PENDING` well is compiled the same as an
    `ACTIVE` one, just tagged `PENDING`, so a later schedule event can
    flip it to `ACTIVE` in place without recompiling.

    :param wells: Every well that will ever exist across the run (the
        full schedule roster, not just currently-active wells).
    :param controls: Current controls. A well absent from `controls`
        compiles to an `UNSET`-mode row (see `compile_well_controls`).
    :param grid: Grid to resolve perforation geometry against.
    :param permeabilities: Per-axis permeability arrays, forwarded to
        `wells.indices.build_wells_indices`.
    :param resolve_kwargs: Forwarded to `build_wells_indices` (e.g.
        `regime_constant`, `net_to_gross`, `horizontal_tolerance`, `method`).
    :returns: `CompiledWellSystem`, well rows in `wells.names` order.
    :raises ValidationError: If `grid.unit_system != wells.unit_system`
        (propagated from `build_wells_indices`), or `wells`/`controls`
        don't share `wells.unit_system` (propagated from
        `compile_well_controls`'s callers being expected to have already
        converted both to one system, same precondition
        `build_wells_indices` already has).
    """
    if controls.unit_system != wells.unit_system:
        raise ValidationError(
            f"`controls` unit_system ({controls.unit_system.value}) != "
            f"`wells` unit_system ({wells.unit_system.value}). Convert both "
            "to the same unit system before compiling."
        )

    names = wells.names
    perforations = compile_perforations(
        names, wells, grid, permeabilities, **resolve_kwargs
    )
    compiled_controls = compile_well_controls(names, controls, wells)

    well_kind = compiled_controls.well_kind
    schedule_status = np.asarray(
        [get_well_status_tag(wells[name].schedule_status) for name in names],
        dtype=np.int64,
    )
    reference_depth = np.asarray(
        [wells[name].reference_depth for name in names], dtype=np.float64
    )
    return CompiledWellSystem(
        names=tuple(names),
        well_kind=well_kind,
        schedule_status=schedule_status,
        reference_depth=reference_depth,
        perforations=perforations,
        controls=compiled_controls,
        unit_system=wells.unit_system,
    )
