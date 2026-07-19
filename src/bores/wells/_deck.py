"""
Utilities for building well model definition objects from parsed Eclipse deck records.
"""

import typing

import attrs

from bores.constants import c
from bores.datastructures import GridDimensions
from bores.deck.core import DeckParseError
from bores.deck.file import DeckFile
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


def _cell_index(dims: GridDimensions, i: int, j: int, k: int) -> int:
    """
    Zero-based flat cell index for one-based Eclipse (I, J, K).

    :param dims: Resolved grid dimensions (deck_file.dimensions).
    :param i: One-based I index.
    :param j: One-based J index.
    :param k: One-based K index.
    :returns: Zero-based flat cell index, Eclipse i-fastest order - matches
        how `ArrayKeyword` (PORO/PERMX/etc.) and `Grid` both index cells.
    """
    return (i - 1) + (j - 1) * dims.nx + (k - 1) * dims.nx * dims.ny


def make_well_from_records(
    grid: Grid,
    welspecs_record: typing.Mapping[str, typing.Any],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Optional[
        typing.Sequence[typing.Mapping[str, typing.Any]]
    ] = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    well_type: WellType = WellType.PRODUCER,
    wellbore_radius: float = 0.25,
    tubing_inner_diameter: typing.Optional[float] = None,
    tubing_roughness: typing.Optional[float] = None,
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
    :param wellbore_radius: Default `Well.wellbore_radius` if `COMPDAT`
        item 7 is absent on every record.
    :param tubing_inner_diameter: `Well.tubing_inner_diameter`. `COMPDAT`
        has no tubing-diameter item, so this must be supplied separately.
    :param tubing_roughness: `Well.tubing_roughness`.
    :param unit_system: `Well.unit_system`.
    :returns: Constructed `Well`.
    :raises ValidationError: If `compdat_records` is empty.
    """
    if not compdat_records:
        raise ValidationError(
            f"No `COMPDAT` records for well {welspecs_record['well']!r}."
        )

    perforations = []
    dims = grid.dimensions
    if dims is None:
        raise ValidationError(
            "Cannot ascertain grid dimensions. Ensure that the provided `Grid` has `dimensions`."
        )

    if wpimult_records:
        multiplier_by_ijk = {
            (record["i"], record["j"], record["k1"], record["k2"]): record["multiplier"]
            for record in wpimult_records
        }
    else:
        multiplier_by_ijk = {}

    for record in compdat_records:
        i, j = record["i"], record["j"]
        k1, k2 = record["k1"], record["k2"]
        top_cell = _cell_index(dims, i, j, k1)
        bottom_cell = _cell_index(dims, i, j, k2)
        top_depth = grid.cell_min_xyz[top_cell, 2]
        bottom_depth = grid.cell_max_xyz[bottom_cell, 2]
        multiplier_key = (record["i"], record["j"], record["k1"], record["k2"])
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
                connection_factor_multiplier=multiplier_by_ijk.get(multiplier_key),
                direction=DIRECTION_MAP.get(direction) if direction else None,
            )
        )

    deepest_bottom_depth = max(perforation.bottom_depth for perforation in perforations)
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


def make_wells_from_records(
    grid: Grid,
    welspecs_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Optional[
        typing.Sequence[typing.Mapping[str, typing.Any]]
    ] = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    injector_names: typing.Container[str] = (),
    **well_kwargs: typing.Any,
) -> Wells:
    """
    Build a `Wells` container from every `WELSPECS`/`COMPDAT` record in a
    deck.

    :param grid: Forwarded to `make_well_from_records`.
    :param welspecs_records: All parsed `WELSPECS` records in the deck.
    :param compdat_records: All parsed `COMPDAT` records in the deck.
    :param unit_system: The deck's unit system.
    :param injector_names: Well names that appear in `WCONINJE`. Every
        other well is built as `WellType.PRODUCER`. WELSPECS/COMPDAT alone
        don't say which a well is.
    :param well_kwargs: Forwarded to every `make_well_from_records` call
        (`wellbore_radius`, `tubing_inner_diameter`, etc. Deck has no
        per-well tubing-diameter keyword, so this is deck-wide).
    :returns: `Wells` keyed by well name.
    """
    compdat_by_well: typing.Dict[str, typing.List[typing.Mapping[str, typing.Any]]] = {}
    wpimult_by_well: typing.Dict[str, typing.List[typing.Mapping[str, typing.Any]]] = {}
    for record in compdat_records:
        compdat_by_well.setdefault(record["well"], []).append(record)

    if wpimult_records:
        for record in wpimult_records:
            wpimult_by_well.setdefault(record["well"], []).append(record)

    wells = {
        record["well"]: make_well_from_records(
            grid,
            record,
            compdat_by_well.get(record["well"], []),
            wpimult_records=wpimult_by_well.get(record["well"]),
            well_type=(
                WellType.INJECTOR
                if record["well"] in injector_names
                else WellType.PRODUCER
            ),
            unit_system=unit_system,
            **well_kwargs,
        )
        for record in welspecs_records
    }
    return Wells(wells=wells)


def _deck_gas_rate(
    value: typing.Optional[float], unit_system: UnitSystem
) -> typing.Optional[float]:
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


def make_producer_control_from_record(
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
    limits: typing.List[Limit] = []
    bhp = record.get("bhp")
    if bhp is not None and mode is not ProducerControlMode.BHP:
        limits.append(BHPLimit(min_value=bhp, unit_system=unit_system))

    return ProducerControl(
        mode=mode,
        target_rate={
            ProducerControlMode.ORAT: record.get("orat"),
            ProducerControlMode.WRAT: record.get("wrat"),
            ProducerControlMode.GRAT: _deck_gas_rate(record.get("grat"), unit_system),
            ProducerControlMode.LRAT: record.get("lrat"),
            ProducerControlMode.RESV: record.get("resv"),
        }.get(mode),
        target_bhp=bhp,
        target_thp=record.get("thp"),
        limits=tuple(limits),
        unit_system=unit_system,
    )


def make_injector_control_from_record(
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
    limits: typing.List[Limit] = []
    bhp = record.get("bhp")
    if bhp is not None and mode is not InjectorControlMode.BHP:
        limits.append(BHPLimit(max_value=bhp, unit_system=unit_system))

    rate = record.get("rate")
    if phase is FluidPhase.GAS:
        rate = _deck_gas_rate(rate, unit_system)

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


def make_economic_limits_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> typing.Tuple[EconomicLimit, ...]:
    """
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
            value = _deck_gas_rate(value, unit_system)
        elif (
            quantity is EconomicQuantity.WATER_GAS_RATIO
            and unit_system is UnitSystem.FIELD
        ):
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
    :raises KeyError: If a record's well has no control set in `controls`
        yet - call this after make_controls_from_records, not before.
    """
    for record in wecon_records:
        well_name = record["well"]
        current = controls[well_name]
        new_limits = make_economic_limits_from_record(record, unit_system=unit_system)
        if new_limits:
            controls.set(
                well_name, attrs.evolve(current, limits=current.limits + new_limits)
            )


def make_controls_from_records(
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
    controls: typing.Dict[str, WellControl] = {}
    for record in wconprod_records:
        controls[record["well"]] = make_producer_control_from_record(
            record, unit_system=unit_system
        )
    for record in wconinje_records:
        controls[record["well"]] = make_injector_control_from_record(
            record, unit_system=unit_system
        )
    return WellControls(controls=controls)


def make_groups_from_records(
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


def make_group_control_from_record(
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


def make_group_controls_from_records(
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
    controls: typing.Dict[str, GroupControl] = {}
    for record in gconprod_records:
        controls[record["group"]] = make_group_control_from_record(
            record, is_injection=False, unit_system=unit_system
        )
    for record in gconinje_records:
        controls[record["group"]] = make_group_control_from_record(
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


def load_wells_from_deck(
    deck_file: DeckFile, grid: Grid, **well_kwargs: typing.Any
) -> Wells:
    """
    :param deck_file: Parsed deck containing `WELSPECS`/`COMPDAT`/`WCONINJE`.
    :param grid: Grid built from the same deck (via Grid.from_deck), for
        completion depth lookups.
    :param well_kwargs: Forwarded to `make_wells_from_records` (wellbore_radius, etc.).
    :returns: Wells for every `WELSPECS`/`COMPDAT` well in the deck.
    :raises ValidationError: If the deck has no SPECGRID/DIMENS.
    """
    if deck_file.dimensions is None:
        raise ValidationError(
            "Deck has no SPECGRID/DIMENS; COMPDAT (I, J, K) can't be resolved."
        )

    welspecs = deck_file.get("WELSPECS") or []
    compdat = deck_file.get("COMPDAT") or []
    wconinje = deck_file.get("WCONINJE") or []
    wpimult = deck_file.get("WPIMULT")
    injector_names = {record["well"] for record in wconinje}
    return make_wells_from_records(
        grid=grid,
        welspecs_records=welspecs,
        compdat_records=compdat,
        wpimult_records=wpimult,
        unit_system=deck_file.unit_system,
        injector_names=injector_names,
        **well_kwargs,
    )


def load_well_controls_from_deck(deck_file: DeckFile) -> WellControls:
    """
    :param deck_file: Parsed deck containing `WCONPROD`/`WCONINJE`/`WECON`/`WGRUPCON`.
    :returns: `WellControls` for every well with a `WCONPROD` or `WCONINJE`
        record, with WECON limits merged in where present.
    """
    controls = make_controls_from_records(
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
        apply_guide_rates(controls, wgrupcon)
    return controls


def load_groups_from_deck(deck_file: DeckFile) -> WellGroups:
    """
    :param deck_file: Parsed deck.
    :returns: WellGroups from GRUPTREE, or None if the deck has no GRUPTREE.
    """
    gruptree = deck_file.get("GRUPTREE")
    if not gruptree:
        raise DeckParseError(
            "Cannot load well groups from deck. `GRUPTREE` is missing."
        )
    return make_groups_from_records(gruptree)


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
    return make_group_controls_from_records(
        gconprod_records=gconprod,
        gconinje_records=gconinje,
        unit_system=deck_file.unit_system,
    )
