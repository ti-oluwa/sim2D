"""
Utilities for building well model definition objects from parsed Eclipse deck records.
"""

import typing

import attrs

from bores.constants import c
from bores.deck.core import DeckParseError
from bores.deck.file import DeckFile
from bores.errors import NotSupportedError, ValidationError
from bores.grids.base import Grid
from bores.typing import FluidPhase, Orientation, UnitSystem
from bores.wells.base import CompletionStatus, Perforation, Well, Wells, WellStatus, WellType
from bores.wells.controls import (
    BHPLimit,
    EconomicLimit,
    EconomicQuantity,
    InjectorControl,
    InjectorControlMode,
    Limit,
    ProducerControl,
    ProducerControlMode,
    WellControl,
    WellControls,
)
from bores.wells.groups import (
    GroupControl,
    GroupControls,
    GroupInjectorControlMode,
    GroupProducerControlMode,
    WellGroup,
    WellGroups,
)

DIRECTION_MAP = {"X": Orientation.X, "Y": Orientation.Y, "Z": Orientation.Z}


def load_well_from_records(
    grid: Grid,
    welspecs_record: typing.Mapping[str, typing.Any],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    well_type: WellType = WellType.PRODUCER,
    current_time: float = 0.0,
) -> Well:
    """
    Builds one well from its `WELSPECS` record and its `COMPDAT` records.

    Every `COMPDAT` record for the well is included, no matter when in the
    schedule it takes effect. A completion whose scheduled time has not
    been reached yet is included but marked as not yet active, so a
    workover completion added later in the schedule is already part of the
    well from the start and just needs switching on when its time comes,
    rather than being added on the fly. The well itself is marked the same
    way, based on when its `WELSPECS` record takes effect.

    :param grid: Grid the well's completions are resolved against.
    :param welspecs_record: The well's `WELSPECS` record.
    :param compdat_records: Every `COMPDAT` record for this well, from any
        point in the schedule.
    :param unit_system: The deck's unit system.
    :param well_type: Whether this well is a producer or an injector. Not
        derivable from `WELSPECS`/`COMPDAT` alone.
    :param current_time: The point on the schedule clock this well is
        being built for, in the deck's time unit. Anything scheduled at or
        before this time is marked active; anything later is marked
        pending. Defaults to zero, the start of the run.
    :returns: The constructed well.
    :raises ValidationError: If there are no `COMPDAT` records for this well.
    """
    if not compdat_records:
        raise ValidationError(f"No `COMPDAT` records for well {welspecs_record['well']!r}.")

    perforations = []
    dims = grid.dimensions
    if dims is None:
        raise ValidationError(
            "Cannot ascertain grid dimensions. Ensure that the provided `Grid` has `dimensions`."
        )

    if welspecs_record["inflow_eq"] != "STD":
        raise NotSupportedError("Only the standard inflow equation (STD) is currently supported.")

    whole_well_multiplier: float | None = None
    multiplier_by_ijk: dict[tuple[int, int, int, int], float] = {}
    if wpimult_records:
        for record in wpimult_records:
            i, j = record.get("i", 0), record.get("j", 0)
            k1, k2 = record.get("k1", 0), record.get("k2", 0)
            if i == 0 and j == 0 and k1 == 0 and k2 == 0:
                # WPIMULT's own schema defaults I/J/K1/K2 to 0, meaning
                # "every connection on this well" - that key would never
                # match any COMPDAT's real (nonzero) indices below, so a
                # whole-well multiplier (the common case) needs its own
                # fallback slot rather than living in multiplier_by_ijk.
                # Later WPIMULT reissues overwrite earlier ones, matching
                # this file's other reissue semantics (WCONPROD/WCONINJE).
                whole_well_multiplier = record["multiplier"]
            else:
                multiplier_by_ijk[i, j, k1, k2] = record["multiplier"]

    for record in compdat_records:
        # Minus 1, to move from 1-based to 0-based indexing used internally
        i, j = record["i"], record["j"]
        k1, k2 = record["k1"], record["k2"]
        top_cell = dims.flat_index(i - 1, j - 1, k1 - 1)
        bottom_cell = dims.flat_index(i - 1, j - 1, k2 - 1)
        top_depth = grid.cell_min_xyz[top_cell, 2]
        bottom_depth = grid.cell_max_xyz[bottom_cell, 2]
        multiplier_key = (i, j, k1, k2)
        direction = record.get("direction")
        saturation_region = record.get("sat_table") or None  # 0 should map to None too
        radius = (record.get("diameter") or 0) * 0.5
        skin = record.get("skin", 0.0) or 0.0
        status = (
            CompletionStatus.OPEN
            if record.get("status", "OPEN") == "OPEN"
            else CompletionStatus.SHUT
        )
        schedule_status = (
            WellStatus.ACTIVE
            if record.get("schedule_time", 0.0) <= current_time
            else WellStatus.PENDING
        )
        perforations.append(
            Perforation(
                top_depth=top_depth,
                bottom_depth=bottom_depth,
                skin=skin,
                wellbore_radius=radius,
                status=status,
                saturation_region=saturation_region,
                connection_factor_override=record.get("connection_factor"),
                connection_factor_multiplier=multiplier_by_ijk.get(
                    multiplier_key, whole_well_multiplier
                ),
                direction=DIRECTION_MAP.get(direction) if direction else None,
                schedule_status=schedule_status,
            )
        )

    reference_depth = welspecs_record.get("ref_depth")
    deepest_bottom_depth = max(perforation.bottom_depth for perforation in perforations)
    surface_location = grid.get_cell_center_at(
        welspecs_record["i"] - 1,
        welspecs_record["j"] - 1,
        0,  # At surface
    )[:2]
    pvt_region = welspecs_record.get("pvt_table") or None  # 0 should map to None too
    preferred_phase = (
        FluidPhase(welspecs_record["phase"].lower()) if welspecs_record.get("phase") else None
    )
    well_schedule_status = (
        WellStatus.ACTIVE
        if welspecs_record.get("schedule_time", 0.0) <= current_time
        else WellStatus.PENDING
    )
    return Well(
        name=welspecs_record["well"],
        well_type=well_type,
        surface_location=surface_location,
        reference_depth=reference_depth if reference_depth is not None else deepest_bottom_depth,
        perforations=tuple(perforations),
        preferred_phase=preferred_phase,
        group=welspecs_record.get("group"),
        pvt_region=pvt_region,
        unit_system=unit_system,
        schedule_status=well_schedule_status,
    )


def load_wells_from_records(
    grid: Grid,
    welspecs_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    injector_names: typing.Container[str] = (),
    current_time: float = 0.0,
) -> Wells:
    """
    Builds a full well roster from every `WELSPECS` and `COMPDAT` record in a
    deck, no matter where in the schedule each one occurs.

    A well introduced later in the schedule, or a completion added to an
    existing well later (a workover, for example), is still built now and
    included in the roster, just marked as not yet active. This means the
    whole run's wells and completions only need building once. As the
    schedule reaches each one's time, it can be switched on in place
    instead of being added partway through the run.

    :param grid: Grid the wells' completions are resolved against.
    :param welspecs_records: Every `WELSPECS` record in the deck.
    :param compdat_records: Every `COMPDAT` record in the deck.
    :param unit_system: The deck's unit system.
    :param injector_names: Names of wells that appear in `WCONINJE`. Every
        other well is built as a producer, since `WELSPECS` and `COMPDAT`
        alone don't say which a well is.
    :param current_time: The point on the schedule clock the roster is
        being built for, in the deck's time unit. Anything scheduled at or
        before this time is marked active; anything later is marked
        pending. Defaults to zero, the start of the run.
    :returns: Wells keyed by name, covering the well's whole life in the schedule.
    """
    compdat_by_well: dict[str, list[typing.Mapping[str, typing.Any]]] = {}
    wpimult_by_well: dict[str, list[typing.Mapping[str, typing.Any]]] = {}
    for record in compdat_records:
        compdat_by_well.setdefault(record["well"], []).append(record)

    if wpimult_records:
        for record in wpimult_records:
            wpimult_by_well.setdefault(record["well"], []).append(record)

    wells = {
        record["well"]: load_well_from_records(
            grid,
            welspecs_record=record,
            compdat_records=compdat_by_well.get(record["well"], []),
            wpimult_records=wpimult_by_well.get(record["well"]),
            well_type=(
                WellType.INJECTOR if record["well"] in injector_names else WellType.PRODUCER
            ),
            unit_system=unit_system,
            current_time=current_time,
        )
        for record in welspecs_records
    }
    return Wells(wells=wells)


def from_deck_gas_rate(value: float | None, unit_system: UnitSystem) -> float | None:
    """
    :param value: Raw gas rate value as written in the deck, or None.
    :param unit_system: The deck's unit system.
    :returns: value unchanged, except multiplied by 1000 when
        `unit_system` is FIELD. Eclipse's FIELD convention reports gas
        rates in Mscf/day; this codebase's internal FIELD convention is
        raw scf/day, dimensionally consistent with oil/water in stb/day.
        Not applied for any other `unit_system`.
    """
    if value is None:
        return None
    return value * c.MSCF_TO_SCF if unit_system is UnitSystem.FIELD else value


def select_current_records(
    records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    key: str,
    current_time: float,
) -> dict[str, typing.Mapping[str, typing.Any]]:
    """
    Picks, for each well or group name, whichever one record from a
    keyword is actually in effect at a given point in the schedule.

    A record scheduled for later than the given time is not counted yet.
    Among the records that have already taken effect for a name, the one
    scheduled most recently wins, since it is the latest change.

    :param records: Every record for one keyword, from any point in the schedule.
    :param key: Which field on a record holds the well or group name.
    :param current_time: The point on the schedule clock to resolve records for.
    :returns: One record per name, the one currently in effect.
    """
    current: dict[str, typing.Mapping[str, typing.Any]] = {}
    current_times: dict[str, float] = {}
    for record in records:
        schedule_time = record.get("schedule_time", 0.0)
        if schedule_time > current_time:
            continue
        name = record[key]
        if name not in current or schedule_time >= current_times[name]:
            current[name] = record
            current_times[name] = schedule_time
    return current


def load_producer_control_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> ProducerControl:
    """
    Build a `ProducerControl` from one `WCONPROD` record.

    :param record: One parsed `WCONPROD` record.
    :param unit_system: The deck's unit system.
    :returns: Constructed `ProducerControl`. If item bhp is present and mode
        isn't BHP, adds an implicit `BHPLimit(min_value=bhp)`.
    """
    mode = ProducerControlMode(record["control_mode"])
    limits: list[Limit] = []
    bhp = record.get("bhp")
    if bhp is not None and mode is not ProducerControlMode.BHP:
        limits.append(BHPLimit(min_value=bhp, unit_system=unit_system))

    return ProducerControl(
        mode=mode,
        target_rate={
            ProducerControlMode.ORAT: record.get("orat"),
            ProducerControlMode.WRAT: record.get("wrat"),
            ProducerControlMode.GRAT: from_deck_gas_rate(record.get("grat"), unit_system),
            ProducerControlMode.LRAT: record.get("lrat"),
            ProducerControlMode.RESV: record.get("resv"),
        }.get(mode),
        target_bhp=bhp,
        target_thp=record.get("thp"),
        limits=tuple(limits),
        unit_system=unit_system,
    )


def load_injector_control_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> InjectorControl:
    """
    Build an `InjectorControl` from one `WCONINJE` record.

    :param record: One parsed `WCONINJE` record.
    :param unit_system: The deck's unit system.
    :returns: Constructed `InjectorControl`. If item bhp is present and mode
        isn't BHP, adds an implicit `BHPLimit(max_value=bhp)`.
    """
    mode = InjectorControlMode(record["control_mode"])
    phase = FluidPhase(record["injector_type"].lower())
    limits: list[Limit] = []
    bhp = record.get("bhp")
    if bhp is not None and mode is not InjectorControlMode.BHP:
        limits.append(BHPLimit(max_value=bhp, unit_system=unit_system))

    rate = record.get("rate")
    if phase is FluidPhase.GAS:
        rate = from_deck_gas_rate(rate, unit_system)

    return InjectorControl(
        injected_phase=phase,
        mode=mode,
        target_rate=rate,
        target_bhp=bhp,
        target_thp=record.get("thp"),
        limits=tuple(limits),
        unit_system=unit_system,
    )


ECONOMIC_QUANTITY_FIELDS = {
    EconomicQuantity.WATER_CUT: "max_wcut",
    EconomicQuantity.GOR: "max_gor",
    EconomicQuantity.WATER_GAS_RATIO: "max_wgr",
}


def load_economic_limits_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> tuple[EconomicLimit, ...]:
    """
    Load `EconomicLimit`s from `WECON` records.

    :param record: One parsed `WECON` record.
    :param unit_system: The deck's unit system.
    :returns: One `EconomicLimit` per non-`None` ratio item present on the
        record (water cut, GOR, water-gas ratio). Min-rate items
        (min_oil_rate/min_gas_rate) aren't covered by `EconomicLimit`'s
        current shape - not converted here, flagged rather than dropped
        silently.
    """
    limits = []
    for quantity, field_name in ECONOMIC_QUANTITY_FIELDS.items():
        value = record.get(field_name)
        if value is None:
            continue

        # GOR (scf/stb) and water-gas ratio (stb/scf) both carry a gas
        # term. The Mscf/scf deck convention applies to that term the
        # same way it does to a standalone gas rate.
        if quantity == EconomicQuantity.GOR:
            value = from_deck_gas_rate(value, unit_system)
        elif quantity is EconomicQuantity.WATER_GAS_RATIO and unit_system is UnitSystem.FIELD:
            value /= c.MSCF_TO_SCF
        limits.append(
            EconomicLimit(quantity=quantity, max_value=value, unit_system=unit_system)  # type: ignore
        )
    return tuple(limits)


def apply_economic_limits(
    controls: WellControls,
    wecon_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
    current_time: float = 0.0,
) -> None:
    """
    Adds each well's economic limits onto its existing control in
    `controls`, in place, using whichever `WECON` record is in effect for
    that well at a given point in the schedule.

    If a well has more than one `WECON` record over the schedule, only the
    most recent one that has already taken effect is used. A reissue
    replaces that well's earlier economic limits rather than adding to
    them, matching how a well's other controls are reissued.

    :param controls: Well controls to update.
    :param wecon_records: Every `WECON` record in the deck.
    :param unit_system: The deck's unit system.
    :param current_time: The point on the schedule clock to resolve limits
        for, in the deck's time unit. Defaults to zero, the start of the run.
    :raises KeyError: If a record's well has no control set in `controls` yet.
    """
    current_records = select_current_records(wecon_records, key="well", current_time=current_time)
    for well_name, record in current_records.items():
        current_control = controls[well_name]
        new_limits = load_economic_limits_from_record(record, unit_system=unit_system)
        if not new_limits:
            continue
        new_quantities = {limit.quantity for limit in new_limits}
        kept_limits = tuple(
            limit
            for limit in current_control.limits
            if not (isinstance(limit, EconomicLimit) and limit.quantity in new_quantities)
        )
        controls.set(well_name, attrs.evolve(current_control, limits=kept_limits + new_limits))


def load_controls_from_records(
    wconprod_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    wconinje_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
    current_time: float = 0.0,
) -> WellControls:
    """
    Builds well controls from every `WCONPROD` and `WCONINJE` record in a
    deck, resolved to whichever control is actually in effect for each
    well at a given point in the schedule.

    A well with more than one control record over the course of the
    schedule is changing control mode partway through the run, or in some
    cases converting between producer and injector. Only the most recent
    record that has already taken effect is used, comparing `WCONPROD` and
    `WCONINJE` records for the same well against each other by their actual
    time in the schedule rather than assuming one keyword always wins.

    :param wconprod_records: Every `WCONPROD` record in the deck.
    :param wconinje_records: Every `WCONINJE` record in the deck.
    :param unit_system: The deck's unit system.
    :param current_time: The point on the schedule clock to resolve
        controls for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :returns: Well controls keyed by well name, one per well that has a
        control in effect by this time.
    """
    candidates: list[tuple[typing.Mapping[str, typing.Any], bool]] = [
        (record, False) for record in wconprod_records
    ] + [(record, True) for record in wconinje_records]

    current: dict[str, tuple[typing.Mapping[str, typing.Any], bool]] = {}
    current_times: dict[str, float] = {}
    for record, is_injection in candidates:
        schedule_time = record.get("schedule_time", 0.0)
        if schedule_time > current_time:
            continue
        well_name = record["well"]
        if well_name not in current or schedule_time >= current_times[well_name]:
            current[well_name] = (record, is_injection)
            current_times[well_name] = schedule_time

    controls: dict[str, WellControl] = {}
    for well_name, (record, is_injection) in current.items():
        controls[well_name] = (
            load_injector_control_from_record(record, unit_system=unit_system)
            if is_injection
            else load_producer_control_from_record(record, unit_system=unit_system)
        )
    return WellControls(controls=controls)


def load_groups_from_records(
    gruptree_records: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> WellGroups:
    """
    Build a `WellGroups` hierarchy from `GRUPTREE` records.

    :param gruptree_records: All parsed `GRUPTREE` records (`child`/`parent` fields).
    :returns: `WellGroups`.
    """
    groups = {
        record["child"]: WellGroup(name=record["child"], parent=record["parent"])
        for record in gruptree_records
    }
    return WellGroups(groups=groups)


def load_group_control_from_record(
    record: typing.Mapping[str, typing.Any],
    *,
    is_injection: bool,
    unit_system: UnitSystem,
) -> GroupControl:
    """
    Build a `GroupControl` from one `GCONPROD`/`GCONINJE` record.

    :param record: One parsed `GCONPROD` or `GCONINJE` record.
    :param is_injection: `True` for a `GCONINJE` record, `False` for `GCONPROD`.
    :param unit_system: The deck's unit system.
    :returns: Constructed `GroupControl`.
    """
    if is_injection:
        return GroupControl(
            mode=GroupInjectorControlMode(record["control_mode"]),
            target_rate=record.get("rate"),
            injected_phase=(
                FluidPhase(record["injector_type"].lower())
                if record.get("injector_type")
                else None
            ),
            unit_system=unit_system,
        )
    return GroupControl(
        mode=GroupProducerControlMode(record["control_mode"]),
        target_rate={
            "ORAT": record.get("orat"),
            "WRAT": record.get("wrat"),
            "GRAT": record.get("grat"),
            "LRAT": record.get("lrat"),
            "RESV": record.get("resv"),
        }.get(record["control_mode"]),
        unit_system=unit_system,
    )


def load_group_controls_from_records(
    gconprod_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    gconinje_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
    current_time: float = 0.0,
) -> GroupControls:
    """
    Builds group controls from every `GCONPROD` and `GCONINJE` record in a
    deck, resolved to whichever control is actually in effect for each
    group at a given point in the schedule.

    A group with more than one control record over the course of the
    schedule is changing control mode partway through the run. Only the
    most recent one that has already taken effect is used, comparing
    `GCONPROD` and `GCONINJE` records for the same group against each other by
    their actual time in the schedule.

    :param gconprod_records: Every `GCONPROD` record in the deck.
    :param gconinje_records: Every `GCONINJE` record in the deck.
    :param unit_system: The deck's unit system.
    :param current_time: The point on the schedule clock to resolve
        controls for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :returns: Group controls keyed by group name, one per group that has a
        control in effect by this time.
    """
    candidates: list[tuple[typing.Mapping[str, typing.Any], bool]] = [
        (record, False) for record in gconprod_records
    ] + [(record, True) for record in gconinje_records]

    current: dict[str, tuple[typing.Mapping[str, typing.Any], bool]] = {}
    current_times: dict[str, float] = {}
    for record, is_injection in candidates:
        schedule_time = record.get("schedule_time", 0.0)
        if schedule_time > current_time:
            continue
        group_name = record["group"]
        if group_name not in current or schedule_time >= current_times[group_name]:
            current[group_name] = (record, is_injection)
            current_times[group_name] = schedule_time

    controls: dict[str, GroupControl] = {}
    for group_name, (record, is_injection) in current.items():
        controls[group_name] = load_group_control_from_record(
            record, is_injection=is_injection, unit_system=unit_system
        )
    return GroupControls(controls=controls)


def apply_guide_rates(
    controls: WellControls,
    wgrupcon_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    current_time: float = 0.0,
) -> None:
    """
    Sets each well's guide rate on its existing control in `controls`, in
    place, using whichever `WGRUPCON` record is in effect for that well at a
    given point in the schedule.

    :param controls: Well controls to update.
    :param wgrupcon_records: Every `WGRUPCON` record in the deck.
    :param current_time: The point on the schedule clock to resolve guide
        rates for, in the deck's time unit. Defaults to zero, the start of the run.
    :raises KeyError: If a record's well has no control set in `controls` yet.
    """
    current_records = select_current_records(
        wgrupcon_records, key="well", current_time=current_time
    )
    for well_name, record in current_records.items():
        guide_rate = record.get("guide_rate")
        if guide_rate is None:
            continue
        current_control = controls[well_name]
        controls.set(well_name, attrs.evolve(current_control, guide_rate=guide_rate))


def load_wells_from_deck(deck_file: DeckFile, grid: Grid, current_time: float = 0.0) -> Wells:
    """
    Builds the full well roster from a parsed deck, covering every well
    and completion the deck ever defines across the whole schedule.

    :param deck_file: Parsed deck containing `WELSPECS`, `COMPDAT`, and `WCONINJE`.
    :param grid: Grid built from the same deck, used to resolve completion depths.
    :param current_time: The point on the schedule clock the roster is
        being built for, in the deck's time unit. Anything scheduled at or
        before this time is marked active; anything later is marked
        pending. Defaults to zero, the start of the run.
    :returns: Wells for every well the deck defines, at any point in the schedule.
    :raises ValidationError: If the deck has no grid dimensions.
    """
    if deck_file.dimensions is None:
        raise ValidationError(
            "Deck has no `SPECGRID`/`DIMENS`; `COMPDAT` (I, J, K) can't be resolved."
        )

    welspecs = deck_file.get("WELSPECS") or []
    compdat = deck_file.get("COMPDAT") or []
    wconinje = deck_file.get("WCONINJE") or []
    wpimult = deck_file.get("WPIMULT")
    injector_names = {record["well"] for record in wconinje}
    return load_wells_from_records(
        grid=grid,
        welspecs_records=welspecs,
        compdat_records=compdat,
        wpimult_records=wpimult,
        unit_system=deck_file.unit_system,
        injector_names=injector_names,
        current_time=current_time,
    )


def load_well_controls_from_deck(deck_file: DeckFile, current_time: float = 0.0) -> WellControls:
    """
    Builds well controls from a parsed deck, resolved to whatever is
    actually in effect for each well at a given point in the schedule.

    :param deck_file: Parsed deck containing `WCONPROD`, `WCONINJE`, `WECON`, and `WGRUPCON`.
    :param current_time: The point on the schedule clock to resolve
        controls for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :returns: Well controls for every well that has a control in effect by this time.
    """
    controls = load_controls_from_records(
        wconprod_records=deck_file.get("WCONPROD") or [],
        wconinje_records=deck_file.get("WCONINJE") or [],
        unit_system=deck_file.unit_system,
        current_time=current_time,
    )
    wecon = deck_file.get("WECON") or []
    if wecon:
        apply_economic_limits(
            controls=controls,
            wecon_records=wecon,
            unit_system=deck_file.unit_system,
            current_time=current_time,
        )

    wgrupcon = deck_file.get("WGRUPCON") or []
    if wgrupcon:
        apply_guide_rates(controls=controls, wgrupcon_records=wgrupcon, current_time=current_time)
    return controls


def load_groups_from_deck(deck_file: DeckFile) -> WellGroups:
    """
    Builds the group hierarchy from a parsed deck.

    :param deck_file: Parsed deck.
    :returns: Well groups from `GRUPTREE`.
    :raises DeckParseError: If the deck has no `GRUPTREE`.
    """
    gruptree = deck_file.get("GRUPTREE")
    if not gruptree:
        raise DeckParseError("Cannot load well groups from deck. `GRUPTREE` is missing.")
    return load_groups_from_records(gruptree)


def load_group_controls_from_deck(deck_file: DeckFile, current_time: float = 0.0) -> GroupControls:
    """
    Builds group controls from a parsed deck, resolved to whatever is
    actually in effect for each group at a given point in the schedule.

    :param deck_file: Parsed deck.
    :param current_time: The point on the schedule clock to resolve
        controls for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :returns: Group controls for every group that has a control in effect by this time.
    :raises DeckParseError: If the deck has neither `GCONPROD` nor `GCONINJE`.
    """
    gconprod = deck_file.get("GCONPROD") or []
    gconinje = deck_file.get("GCONINJE") or []
    if not gconprod and not gconinje:
        raise DeckParseError(
            "Cannot load well group controls from deck. `GCONPROD` and `GCONINJE` are both missing. "
            "At least one should be present."
        )
    return load_group_controls_from_records(
        gconprod_records=gconprod,
        gconinje_records=gconinje,
        unit_system=deck_file.unit_system,
        current_time=current_time,
    )
