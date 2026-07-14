"""Build Well/WellControl/Group/GroupControl objects: from parsed deck
records, or manually with sane defaults for direct API use.

COMPDAT specifies completions by (I, J, K1, K2) grid cell range, not by
depth - Perforation is depth-based, so well_from_records derives a depth
range from the grid's cell bounds at that (I, J, K1:K2) range. This
re-enters resolve_perforations_indices' geometric search rather than using
the COMPDAT-named cell(s) directly, which is redundant and can pick a
different cell in a distorted grid. A direct cell-index Perforation path
that skips geometric resolution entirely isn't built yet - flagged here,
not worked around.
"""

import typing

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.typing import FluidPhase, Orientation, UnitSystem
from bores.wells.base import CompletionStatus, Perforation, Well, Wells, WellType
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
    WellControl,
    WellControls,
)
from bores.wells.groups import (
    Group,
    GroupControl,
    GroupControls,
    GroupInjectorControlMode,
    GroupProducerControlMode,
    Groups,
)

__all__ = [
    "well_from_records",
    "wells_from_records",
    "producer_control_from_record",
    "injector_control_from_record",
    "controls_from_records",
    "groups_from_records",
    "group_control_from_record",
    "group_controls_from_records",
    "make_producer",
    "make_injector",
]

_DIRECTION_MAP = {"X": Orientation.X, "Y": Orientation.Y, "Z": Orientation.Z}


def _cell_index(grid: Grid, i: int, j: int, k: int) -> int:
    """
    Zero-based flat cell index for one-based Eclipse (I, J, K).

    :param grid: Grid to index into.
    :param i: One-based I index.
    :param j: One-based J index.
    :param k: One-based K index.
    :returns: Zero-based flat cell index (Eclipse i-fastest order).
    """
    return (i - 1) + (j - 1) * grid.nx + (k - 1) * grid.nx * grid.ny


def well_from_records(
    grid: Grid,
    welspecs_record: typing.Mapping[str, typing.Any],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    well_type: WellType = WellType.PRODUCER,
    wellbore_radius: float = 0.25,
    tubing_inner_diameter: typing.Optional[float] = None,
    tubing_roughness: typing.Optional[float] = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
) -> Well:
    """
    Build one `Well` from its `WELSPECS` record and its `COMPDAT` records.

    :param grid: Grid the well's completions are resolved against, for
        converting each COMPDAT (I, J, K1, K2) to a depth range.
    :param welspecs_record: One parsed `WELSPECS` record.
    :param compdat_records: All parsed `COMPDAT` records for this well
        (already filtered to this well's name by the caller).
    :param well_type: `Well.well_type` - not derivable from WELSPECS/COMPDAT
        alone; the caller determines this from which of WCONPROD/WCONINJE
        the well appears in.
    :param wellbore_radius: Default `Well.wellbore_radius` if `COMPDAT`
        item 7 is absent on every record.
    :param tubing_inner_diameter: `Well.tubing_inner_diameter` - `COMPDAT`
        has no tubing-diameter item, so this must be supplied separately.
    :param tubing_roughness: `Well.tubing_roughness`.
    :param unit_system: `Well.unit_system`.
    :returns: Constructed `Well`.
    :raises ValidationError: If `compdat_records` is empty.
    """
    if not compdat_records:
        raise ValidationError(
            f"No COMPDAT records for well {welspecs_record['well']!r}."
        )

    perforations = []
    for record in compdat_records:
        i, j = record["i"], record["j"]
        k1, k2 = record["k1"], record["k2"]
        top_cell = _cell_index(grid, i, j, k1)
        bottom_cell = _cell_index(grid, i, j, k2)
        top_depth = grid.cell_min_xyz[top_cell, 2]
        bottom_depth = grid.cell_max_xyz[bottom_cell, 2]

        direction = record.get("direction")
        perforations.append(
            Perforation(
                top_depth=top_depth,
                bottom_depth=bottom_depth,
                skin=record.get("skin", 0.0) or 0.0,
                wellbore_radius=record.get("wellbore_radius"),
                status=(
                    CompletionStatus.OPEN
                    if record.get("status", "OPEN") == "OPEN"
                    else CompletionStatus.SHUT
                ),
                connection_factor_override=record.get("connection_factor"),
                direction=_DIRECTION_MAP.get(direction) if direction else None,
            )
        )

    deepest_bottom_depth = max(p.bottom_depth for p in perforations)
    return Well(
        name=welspecs_record["well"],
        well_type=well_type,
        surface_location=(welspecs_record["i"], welspecs_record["j"]),
        reference_depth=welspecs_record.get("ref_depth") or deepest_bottom_depth,
        perforations=tuple(perforations),
        preferred_phase=(
            FluidPhase(welspecs_record["phase"].lower())
            if welspecs_record.get("phase")
            else None
        ),
        group=welspecs_record.get("group"),
        wellbore_radius=wellbore_radius,
        tubing_inner_diameter=tubing_inner_diameter,
        tubing_roughness=tubing_roughness,
        unit_system=unit_system,
    )


def wells_from_records(
    grid: Grid,
    welspecs_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    injector_names: typing.Container[str] = (),
    **well_kwargs: typing.Any,
) -> Wells:
    """
    Build a `Wells` container from every `WELSPECS`/`COMPDAT` record in a
    deck.

    :param grid: Forwarded to `well_from_records`.
    :param welspecs_records: All parsed `WELSPECS` records in the deck.
    :param compdat_records: All parsed `COMPDAT` records in the deck.
    :param injector_names: Well names that appear in `WCONINJE` - every
        other well is built as `WellType.PRODUCER`. WELSPECS/COMPDAT alone
        don't say which a well is.
    :param well_kwargs: Forwarded to every `well_from_records` call
        (`wellbore_radius`, `tubing_inner_diameter`, etc. - deck has no
        per-well tubing-diameter keyword, so this is deck-wide).
    :returns: `Wells` keyed by well name.
    """
    compdat_by_well: typing.Dict[str, typing.List[typing.Mapping[str, typing.Any]]] = {}
    for record in compdat_records:
        compdat_by_well.setdefault(record["well"], []).append(record)

    wells = {
        record["well"]: well_from_records(
            grid,
            record,
            compdat_by_well.get(record["well"], []),
            well_type=(
                WellType.INJECTOR
                if record["well"] in injector_names
                else WellType.PRODUCER
            ),
            **well_kwargs,
        )
        for record in welspecs_records
    }
    return Wells(wells=wells)


def producer_control_from_record(
    record: typing.Mapping[str, typing.Any],
) -> ProducerControl:
    """
    Build a `ProducerControl` from one `WCONPROD` record.

    :param record: One parsed `WCONPROD` record.
    :returns: Constructed `ProducerControl`.
    """
    mode = ProducerControlMode(record["cmode"])
    return ProducerControl(
        mode=mode,
        target_rate={
            ProducerControlMode.ORAT: record.get("orat"),
            ProducerControlMode.WRAT: record.get("wrat"),
            ProducerControlMode.GRAT: record.get("grat"),
            ProducerControlMode.LRAT: record.get("lrat"),
            ProducerControlMode.RESV: record.get("resv"),
        }.get(mode),
        target_bhp=record.get("bhp"),
        target_thp=record.get("thp"),
    )


def injector_control_from_record(
    record: typing.Mapping[str, typing.Any],
) -> InjectorControl:
    """
    Build an `InjectorControl` from one `WCONINJE` record.

    :param record: One parsed `WCONINJE` record.
    :returns: Constructed `InjectorControl`.
    """
    return InjectorControl(
        injected_phase=FluidPhase(record["phase"].lower()),
        mode=InjectorControlMode(record["cmode"]),
        target_rate=record.get("rate"),
        target_bhp=record.get("bhp"),
        target_thp=record.get("thp"),
    )


def controls_from_records(
    wconprod_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    wconinje_records: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> WellControls:
    """
    Build a `WellControls` from every `WCONPROD`/`WCONINJE` record in a
    deck. Later records for the same well name overwrite earlier ones
    (matches Eclipse's own reissue semantics).

    :param wconprod_records: All parsed `WCONPROD` records, file order.
    :param wconinje_records: All parsed `WCONINJE` records, file order.
    :returns: `WellControls` keyed by well name.
    """
    controls: typing.Dict[str, WellControl] = {}
    for record in wconprod_records:
        controls[record["well"]] = producer_control_from_record(record)
    for record in wconinje_records:
        controls[record["well"]] = injector_control_from_record(record)
    return WellControls(controls=controls)


def groups_from_records(
    gruptree_records: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> Groups:
    """
    Build a `Groups` hierarchy from `GRUPTREE` records.

    :param gruptree_records: All parsed `GRUPTREE` records
        (`child`/`parent` fields).
    :returns: `Groups`.
    """
    groups = {
        record["child"]: Group(name=record["child"], parent=record["parent"])
        for record in gruptree_records
    }
    return Groups(groups=groups)


def group_control_from_record(
    record: typing.Mapping[str, typing.Any], *, is_injection: bool
) -> GroupControl:
    """
    Build a `GroupControl` from one `GCONPROD`/`GCONINJE` record.

    :param record: One parsed `GCONPROD` or `GCONINJE` record.
    :param is_injection: `True` for a `GCONINJE` record, `False` for `GCONPROD`.
    :returns: Constructed `GroupControl`.
    """
    if is_injection:
        return GroupControl(
            mode=GroupInjectorControlMode(record["cmode"]),
            target_rate=record.get("rate"),
            injected_phase=(
                FluidPhase(record["phase"].lower()) if record.get("phase") else None
            ),
        )
    return GroupControl(
        mode=GroupProducerControlMode(record["cmode"]),
        target_rate={
            "ORAT": record.get("orat"),
            "WRAT": record.get("wrat"),
            "GRAT": record.get("grat"),
            "LRAT": record.get("lrat"),
            "RESV": record.get("resv"),
        }.get(record["cmode"]),
    )


def group_controls_from_records(
    gconprod_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    gconinje_records: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> GroupControls:
    """
    Build a `GroupControls` from every `GCONPROD`/`GCONINJE` record.

    :param gconprod_records: All parsed `GCONPROD` records, file order.
    :param gconinje_records: All parsed `GCONINJE` records, file order.
    :returns: `GroupControls` keyed by group name.
    """
    controls: typing.Dict[str, GroupControl] = {}
    for record in gconprod_records:
        controls[record["group"]] = group_control_from_record(
            record, is_injection=False
        )
    for record in gconinje_records:
        controls[record["group"]] = group_control_from_record(record, is_injection=True)
    return GroupControls(controls=controls)


def make_producer(
    name: str,
    *,
    surface_location: typing.Tuple[float, float],
    perforation_depths: typing.Tuple[float, float],
    reference_depth: typing.Optional[float] = None,
    target_rate: typing.Optional[float] = None,
    rate_quantity: RateQuantity = RateQuantity.OIL,
    target_bhp: typing.Optional[float] = None,
    min_bhp: typing.Optional[float] = None,
    max_rate: typing.Optional[float] = None,
    wellbore_radius: float = 0.25,
    tubing_inner_diameter: typing.Optional[float] = None,
    skin: float = 0.0,
    group: typing.Optional[str] = None,
) -> typing.Tuple[Well, ProducerControl]:
    """
    Construct a producer `Well` + `ProducerControl` pair manually, with
    sane defaults, for direct (non-deck) API use.

    :param name: Well name.
    :param surface_location: `(x, y)` wellhead location.
    :param perforation_depths: `(top_depth, bottom_depth)` for one single
        perforation. Use `Well(...)` directly for a multi-perforation well.
    :param reference_depth: BHP datum. Defaults to `perforation_depths[1]`.
    :param target_rate: Rate target, used if `target_bhp` is `None`.
    :param rate_quantity: Which rate `target_rate` is - selects the
        resulting `ProducerControlMode`.
    :param target_bhp: BHP target. Takes priority over `target_rate` if
        both are given.
    :param min_bhp: If given, adds a `BHPLimit(min_value=min_bhp)`.
    :param max_rate: If given, adds a `RateLimit` on `rate_quantity` capped
        at `max_rate`.
    :param wellbore_radius: `Well.wellbore_radius`.
    :param tubing_inner_diameter: `Well.tubing_inner_diameter`.
    :param skin: Skin factor on the single perforation.
    :param group: `Well.group`.
    :returns: `(Well, ProducerControl)`.
    :raises ValidationError: If neither `target_rate` nor `target_bhp` is given.
    """
    if target_rate is None and target_bhp is None:
        raise ValidationError("Supply either `target_rate` or `target_bhp`.")

    top_depth, bottom_depth = perforation_depths
    well = Well(
        name=name,
        well_type=WellType.PRODUCER,
        surface_location=surface_location,
        reference_depth=reference_depth
        if reference_depth is not None
        else bottom_depth,
        perforations=(
            Perforation(
                top_depth=top_depth,
                bottom_depth=bottom_depth,
                skin=skin,
            ),
        ),
        wellbore_radius=wellbore_radius,
        tubing_inner_diameter=tubing_inner_diameter,
        group=group,
    )

    limits: typing.List[Limit] = []
    if min_bhp is not None:
        limits.append(BHPLimit(min_value=min_bhp))
    if max_rate is not None:
        limits.append(RateLimit(quantity=rate_quantity, max_value=max_rate))

    if target_bhp is not None:
        control = ProducerControl(
            mode=ProducerControlMode.BHP, target_bhp=target_bhp, limits=tuple(limits)
        )
    else:
        mode_by_quantity = {
            RateQuantity.OIL: ProducerControlMode.ORAT,
            RateQuantity.WATER: ProducerControlMode.WRAT,
            RateQuantity.GAS: ProducerControlMode.GRAT,
            RateQuantity.LIQUID: ProducerControlMode.LRAT,
            RateQuantity.RESERVOIR: ProducerControlMode.RESV,
        }
        control = ProducerControl(
            mode=mode_by_quantity[rate_quantity],
            target_rate=target_rate,
            limits=tuple(limits),
        )

    return well, control


def make_injector(
    name: str,
    *,
    injected_phase: FluidPhase,
    surface_location: typing.Tuple[float, float],
    perforation_depths: typing.Tuple[float, float],
    reference_depth: typing.Optional[float] = None,
    target_rate: typing.Optional[float] = None,
    target_bhp: typing.Optional[float] = None,
    max_bhp: typing.Optional[float] = None,
    wellbore_radius: float = 0.25,
    tubing_inner_diameter: typing.Optional[float] = None,
    skin: float = 0.0,
    group: typing.Optional[str] = None,
) -> typing.Tuple[Well, InjectorControl]:
    """
    Construct an injector `Well` + `InjectorControl` pair manually, with
    sane defaults, for direct (non-deck) API use.

    :param name: Well name.
    :param injected_phase: Phase being injected.
    :param surface_location: `(x, y)` wellhead location.
    :param perforation_depths: `(top_depth, bottom_depth)` for one single
        perforation. Use `Well(...)` directly for a multi-perforation well.
    :param reference_depth: BHP datum. Defaults to `perforation_depths[1]`.
    :param target_rate: Rate target, used if `target_bhp` is `None`.
    :param target_bhp: BHP target. Takes priority over `target_rate` if
        both are given.
    :param max_bhp: If given, adds a `BHPLimit(max_value=max_bhp)`.
    :param wellbore_radius: `Well.wellbore_radius`.
    :param tubing_inner_diameter: `Well.tubing_inner_diameter`.
    :param skin: Skin factor on the single perforation.
    :param group: `Well.group`.
    :returns: `(Well, InjectorControl)`.
    :raises ValidationError: If neither `target_rate` nor `target_bhp` is given.
    """
    if target_rate is None and target_bhp is None:
        raise ValidationError("Supply either `target_rate` or `target_bhp`.")

    top_depth, bottom_depth = perforation_depths
    well = Well(
        name=name,
        well_type=WellType.INJECTOR,
        surface_location=surface_location,
        reference_depth=reference_depth
        if reference_depth is not None
        else bottom_depth,
        perforations=(
            Perforation(top_depth=top_depth, bottom_depth=bottom_depth, skin=skin),
        ),
        wellbore_radius=wellbore_radius,
        tubing_inner_diameter=tubing_inner_diameter,
        group=group,
    )

    limits: typing.List[Limit] = []
    if max_bhp is not None:
        from bores.wells.controls import BHPLimit

        limits.append(BHPLimit(max_value=max_bhp))

    if target_bhp is not None:
        control = InjectorControl(
            injected_phase=injected_phase,
            mode=InjectorControlMode.BHP,
            target_bhp=target_bhp,
            limits=tuple(limits),
        )
    else:
        control = InjectorControl(
            injected_phase=injected_phase,
            mode=InjectorControlMode.RATE,
            target_rate=target_rate,
            limits=tuple(limits),
        )

    return well, control
