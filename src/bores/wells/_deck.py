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
) -> Well:
    """
    Build one `Well` from its `WELSPECS` record and its `COMPDAT` records.

    :param grid: Grid the well's completions are resolved against, for
        converting each `COMPDAT` (I, J, K1, K2) to a depth range.
    :param welspecs_record: One parsed `WELSPECS` record.
    :param compdat_records: All parsed `COMPDAT` records for this well
        (already filtered to this well's name by the caller).
    :param unit_system: The deck's unit system.
    :param well_type: `Well.well_type`. Not derivable from WELSPECS/COMPDAT
        alone; the caller determines this from which of WCONPROD/WCONINJE
        the well appears in.
    :param unit_system: `Well.unit_system`.
    :returns: Constructed `Well`.
    :raises ValidationError: If `compdat_records` is empty.
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
                multiplier_by_ijk[(i, j, k1, k2)] = record["multiplier"]

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
    )


def load_wells_from_records(
    grid: Grid,
    welspecs_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    injector_names: typing.Container[str] = (),
) -> Wells:
    """
    Build a `Wells` container from every `WELSPECS`/`COMPDAT` record in a
    deck.

    :param grid: Forwarded to `load_well_from_records`.
    :param welspecs_records: All parsed `WELSPECS` records in the deck.
    :param compdat_records: All parsed `COMPDAT` records in the deck.
    :param unit_system: The deck's unit system.
    :param injector_names: Well names that appear in `WCONINJE`. Every
        other well is built as `WellType.PRODUCER`. WELSPECS/COMPDAT alone
        don't say which a well is.
    :param well_kwargs: Forwarded to every `load_well_from_records` call
        (`wellbore_radius`, `tubing_inner_diameter`, etc. Deck has no
        per-well tubing-diameter keyword, so this is deck-wide).
    :returns: `Wells` keyed by well name.
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
        )
        for record in welspecs_records
    }
    return Wells(wells=wells)


def from_deck_gas_rate(value: float | None, unit_system: UnitSystem) -> float | None:
    """
    :param value: Raw gas rate value as written in the deck, or None.
    :param unit_system: The deck's unit system.
    :returns: value unchanged, except multiplied by 1000 when
        `unit_system` is FIELD - Eclipse's FIELD convention reports gas
        rates in Mscf/day; this codebase's internal FIELD convention is
        raw scf/day, dimensionally consistent with oil/water in stb/day.
        Not applied for any other `unit_system`.
    """
    if value is None:
        return None
    return value * c.MSCF_TO_SCF if unit_system is UnitSystem.FIELD else value


def load_producer_control_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> ProducerControl:
    """
    Build a `ProducerControl` from one `WCONPROD` record.

    :param record: One parsed WCONPROD record.
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
    EconomicQuantity.WATER_CUT: "max_water_cut",
    EconomicQuantity.GOR: "max_gor",
    EconomicQuantity.WATER_GAS_RATIO: "max_water_gas_ratio",
}


def load_economic_limits_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> tuple[EconomicLimit, ...]:
    """
    Load `EconomicLimit`s from `WECON` records.

    :param record: One parsed WECON record.
    :param unit_system: The deck's unit system.
    :returns: One `EconomicLimit` per non-None ratio item present on the
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
) -> None:
    """
    Add each WECON record's `EconomicLimit`s onto the matching well's
    existing control in `controls`, in place.

    :param controls: `WellControls` to update.
    :param wecon_records: All parsed WECON records.
    :param unit_system: The deck's unit system.
    :raises KeyError: If a record's well has no control set in `controls` yet.
    """
    for record in wecon_records:
        well_name = record["well"]
        current = controls[well_name]
        new_limits = load_economic_limits_from_record(record, unit_system=unit_system)
        if new_limits:
            controls.set(well_name, attrs.evolve(current, limits=current.limits + new_limits))


def load_controls_from_records(
    wconprod_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    wconinje_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
) -> WellControls:
    """
    Build a `WellControls` object from every `WCONPROD`/`WCONINJE` record in a
    deck. Later records for the same well name overwrite earlier ones
    (matches Eclipse's own reissue semantics).

    :param wconprod_records: All parsed `WCONPROD` records, file order.
    :param wconinje_records: All parsed `WCONINJE` records, file order.
    :param unit_system: The deck's unit system.
    :returns: `WellControls` keyed by well name.
    """
    controls: dict[str, WellControl] = {}
    for record in wconprod_records:
        controls[record["well"]] = load_producer_control_from_record(
            record, unit_system=unit_system
        )
    for record in wconinje_records:
        controls[record["well"]] = load_injector_control_from_record(
            record, unit_system=unit_system
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
) -> GroupControls:
    """
    Build a `GroupControls` object from every `GCONPROD`/`GCONINJE` record.

    :param gconprod_records: All parsed `GCONPROD` records, file order.
    :param gconinje_records: All parsed `GCONINJE` records, file order.
    :param unit_system: The deck's unit system.
    :returns: `GroupControls` keyed by group name.
    """
    controls: dict[str, GroupControl] = {}
    for record in gconprod_records:
        controls[record["group"]] = load_group_control_from_record(
            record, is_injection=False, unit_system=unit_system
        )
    for record in gconinje_records:
        controls[record["group"]] = load_group_control_from_record(
            record, is_injection=True, unit_system=unit_system
        )
    return GroupControls(controls=controls)


def apply_guide_rates(
    controls: WellControls,
    wgrupcon_records: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> None:
    """
    :param controls: `WellControls` to update in place.
    :param wgrupcon_records: All parsed `WGRUPCON` records.
    :raises KeyError: If a record's well has no control set in controls yet.
    """
    for record in wgrupcon_records:
        guide_rate = record.get("guide_rate")
        if guide_rate is None:
            continue
        well_name = record["well"]
        current = controls[well_name]
        controls.set(well_name, attrs.evolve(current, guide_rate=guide_rate))


def load_wells_from_deck(deck_file: DeckFile, grid: Grid) -> Wells:
    """
    :param deck_file: Parsed deck containing `WELSPECS`/`COMPDAT`/`WCONINJE`.
    :param grid: Grid built from the same deck, for completion depth lookups.
    :param well_kwargs: Forwarded to `load_wells_from_records` (wellbore_radius, etc.).
    :returns: Wells for every `WELSPECS`/`COMPDAT` well in the deck.
    :raises ValidationError: If the deck has no SPECGRID/DIMENS.
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
    )


def load_well_controls_from_deck(deck_file: DeckFile) -> WellControls:
    """
    :param deck_file: Parsed deck containing `WCONPROD`/`WCONINJE`/`WECON`/`WGRUPCON`.
    :returns: `WellControls` for every well with a `WCONPROD` or `WCONINJE`
        record, with WECON limits merged in where present.
    """
    controls = load_controls_from_records(
        wconprod_records=deck_file.get("WCONPROD") or [],
        wconinje_records=deck_file.get("WCONINJE") or [],
        unit_system=deck_file.unit_system,
    )
    wecon = deck_file.get("WECON") or []
    if wecon:
        apply_economic_limits(
            controls=controls,
            wecon_records=wecon,
            unit_system=deck_file.unit_system,
        )

    wgrupcon = deck_file.get("WGRUPCON") or []
    if wgrupcon:
        apply_guide_rates(controls=controls, wgrupcon_records=wgrupcon)
    return controls


def load_groups_from_deck(deck_file: DeckFile) -> WellGroups:
    """
    :param deck_file: Parsed deck.
    :returns: WellGroups from GRUPTREE, or None if the deck has no GRUPTREE.
    """
    gruptree = deck_file.get("GRUPTREE")
    if not gruptree:
        raise DeckParseError("Cannot load well groups from deck. `GRUPTREE` is missing.")
    return load_groups_from_records(gruptree)


def load_group_controls_from_deck(deck_file: DeckFile) -> GroupControls:
    """
    :param deck_file: Parsed deck.
    :returns: `GroupControls` from `GCONPROD`/`GCONINJE`, or None if the deck
        has neither.
    """
    gconprod = deck_file.get("GCONPROD") or []
    gconinje = deck_file.get("GCONINJE") or []
    if not gconprod and not gconinje:
        raise DeckParseError(
            "Cannot load well group controls from deck. `GCONPROD` and `GCONINJE` are both missing. "
            "Atleast one should be present."
        )
    return load_group_controls_from_records(
        gconprod_records=gconprod,
        gconinje_records=gconinje,
        unit_system=deck_file.unit_system,
    )
