"""Convenience constructors for building wells directly, without a deck."""

import typing

from bores.errors import ValidationError
from bores.typing import FluidPhase, Number, Orientation, UnitSystem
from bores.wells.base import Perforation, Well, Wells, WellStatus, WellType
from bores.wells.controls import (
    BHPLimit,
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
    GroupControl,
    GroupInjectorControlMode,
    GroupProducerControlMode,
)

__all__ = [
    "make_producer",
    "make_injector",
    "make_group_control",
    "make_wells",
    "make_well_controls",
]


QUANTITY_PRODUCER_CONTROL_MODES = {
    RateQuantity.OIL: ProducerControlMode.ORAT,
    RateQuantity.WATER: ProducerControlMode.WRAT,
    RateQuantity.GAS: ProducerControlMode.GRAT,
    RateQuantity.LIQUID: ProducerControlMode.LRAT,
    RateQuantity.RESERVOIR: ProducerControlMode.RESV,
}

PHASE_RATE_QUANTITY = {
    FluidPhase.OIL: RateQuantity.OIL,
    FluidPhase.WATER: RateQuantity.WATER,
    FluidPhase.GAS: RateQuantity.GAS,
}


def make_producer(
    name: str,
    *,
    surface_location: typing.Tuple[Number, Number],
    perforation_depths: typing.Tuple[Number, Number],
    reference_depth: typing.Optional[Number] = None,
    target_rate: typing.Optional[Number] = None,
    rate_quantity: RateQuantity = RateQuantity.OIL,
    target_bhp: typing.Optional[Number] = None,
    target_thp: typing.Optional[Number] = None,
    min_bhp: typing.Optional[Number] = None,
    max_rate: typing.Optional[Number] = None,
    limits: typing.Optional[typing.Sequence[Limit]] = None,
    efficiency_factor: Number = 1.0,
    guide_rate: typing.Optional[Number] = None,
    wellbore_radius: Number = 0.25,
    skin: Number = 0.0,
    saturation_region: typing.Optional[int] = None,
    connection_factor_override: typing.Optional[Number] = None,
    connection_factor_multiplier: typing.Optional[Number] = None,
    direction: typing.Optional[Orientation] = None,
    preferred_phase: typing.Optional[FluidPhase] = None,
    pvt_region: typing.Optional[int] = None,
    group: typing.Optional[str] = None,
    schedule_status: WellStatus = WellStatus.ACTIVE,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> typing.Tuple[Well, ProducerControl]:
    """
    Builds a producer `Well` and its `ProducerControl`, for direct
    (non-deck) API use.

    Control mode is chosen by priority: `target_bhp`, then `target_thp`,
    then `target_rate` at `rate_quantity`.

    :param name: Well name.
    :param surface_location: `(x, y)` wellhead location.
    :param perforation_depths: `(top_depth, bottom_depth)` for one single
        perforation. Use `Well(...)` directly for a multi-perforation well.
    :param reference_depth: BHP/THP reporting datum. `perforation_depths[1]` if not given.
    :param target_rate: Rate target.
    :param rate_quantity: Which rate `target_rate` is.
    :param target_bhp: BHP target.
    :param target_thp: THP target.
    :param min_bhp: If given, adds a `BHPLimit(min_value=min_bhp)`.
    :param max_rate: If given, adds a `RateLimit` on `rate_quantity` capped at `max_rate`.
    :param limits: Additional limits, appended after any built from
        `min_bhp`/`max_rate`.
    :param efficiency_factor: `ProducerControl.efficiency_factor`.
    :param guide_rate: `ProducerControl.guide_rate`.
    :param wellbore_radius: `Perforation.wellbore_radius`.
    :param skin: `Perforation.skin`.
    :param saturation_region: `Perforation.saturation_region`.
    :param connection_factor_override: `Perforation.connection_factor_override`.
    :param connection_factor_multiplier: `Perforation.connection_factor_multiplier`.
    :param direction: `Perforation.direction`.
    :param preferred_phase: `Well.preferred_phase`.
    :param pvt_region: `Well.pvt_region`.
    :param group: `Well.group`.
    :param schedule_status: `Well.schedule_status`.
    :param unit_system: `Well.unit_system` and `ProducerControl.unit_system`.
    :param metadata: `Well.metadata`.
    :returns: `(Well, ProducerControl)`.
    :raises ValidationError: If none of `target_rate`, `target_bhp`, or `target_thp` is given.
    """
    if target_rate is None and target_bhp is None and target_thp is None:
        raise ValidationError(
            "Supply one of `target_rate`, `target_bhp`, or `target_thp`."
        )

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
                wellbore_radius=wellbore_radius,
                skin=skin,
                saturation_region=saturation_region,
                connection_factor_override=connection_factor_override,
                connection_factor_multiplier=connection_factor_multiplier,
                direction=direction,
                schedule_status=schedule_status,
            ),
        ),
        preferred_phase=preferred_phase,
        pvt_region=pvt_region,
        group=group,
        schedule_status=schedule_status,
        unit_system=unit_system,
        metadata=metadata,
    )

    resolved_limits: typing.List[Limit] = []
    if min_bhp is not None:
        resolved_limits.append(BHPLimit(min_value=min_bhp))
    if max_rate is not None:
        resolved_limits.append(RateLimit(quantity=rate_quantity, max_value=max_rate))
    if limits:
        resolved_limits.extend(limits)

    if target_bhp is not None:
        control = ProducerControl(
            mode=ProducerControlMode.BHP,
            target_bhp=target_bhp,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    elif target_thp is not None:
        control = ProducerControl(
            mode=ProducerControlMode.THP,
            target_thp=target_thp,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    else:
        control = ProducerControl(
            mode=QUANTITY_PRODUCER_CONTROL_MODES[rate_quantity],
            target_rate=target_rate,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )

    return well, control


def make_injector(
    name: str,
    *,
    injected_phase: FluidPhase,
    surface_location: typing.Tuple[Number, Number],
    perforation_depths: typing.Tuple[Number, Number],
    reference_depth: typing.Optional[Number] = None,
    target_rate: typing.Optional[Number] = None,
    target_bhp: typing.Optional[Number] = None,
    target_thp: typing.Optional[Number] = None,
    max_bhp: typing.Optional[Number] = None,
    max_rate: typing.Optional[Number] = None,
    limits: typing.Optional[typing.Sequence[Limit]] = None,
    efficiency_factor: Number = 1.0,
    guide_rate: typing.Optional[Number] = None,
    wellbore_radius: Number = 0.25,
    skin: Number = 0.0,
    saturation_region: typing.Optional[int] = None,
    connection_factor_override: typing.Optional[Number] = None,
    connection_factor_multiplier: typing.Optional[Number] = None,
    direction: typing.Optional[Orientation] = None,
    pvt_region: typing.Optional[int] = None,
    group: typing.Optional[str] = None,
    schedule_status: WellStatus = WellStatus.ACTIVE,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> typing.Tuple[Well, InjectorControl]:
    """
    Builds an injector `Well` and its `InjectorControl`, for direct
    (non-deck) API use.

    Control mode is chosen by priority: `target_bhp`, then `target_thp`,
    then `target_rate`.

    :param name: Well name.
    :param injected_phase: Phase being injected.
    :param surface_location: `(x, y)` wellhead location.
    :param perforation_depths: `(top_depth, bottom_depth)` for one single
        perforation. Use `Well(...)` directly for a multi-perforation well.
    :param reference_depth: BHP/THP reporting datum. `perforation_depths[1]` if not given.
    :param target_rate: Rate target.
    :param target_bhp: BHP target.
    :param target_thp: THP target.
    :param max_bhp: If given, adds a `BHPLimit(max_value=max_bhp)`.
    :param max_rate: If given, adds a `RateLimit` on `injected_phase` capped at `max_rate`.
    :param limits: Additional limits, appended after any built from
        `max_bhp`/`max_rate`.
    :param efficiency_factor: `InjectorControl.efficiency_factor`.
    :param guide_rate: `InjectorControl.guide_rate`.
    :param wellbore_radius: `Perforation.wellbore_radius`.
    :param skin: `Perforation.skin`.
    :param saturation_region: `Perforation.saturation_region`.
    :param connection_factor_override: `Perforation.connection_factor_override`.
    :param connection_factor_multiplier: `Perforation.connection_factor_multiplier`.
    :param direction: `Perforation.direction`.
    :param pvt_region: `Well.pvt_region`.
    :param group: `Well.group`.
    :param schedule_status: `Well.schedule_status`.
    :param unit_system: `Well.unit_system` and `InjectorControl.unit_system`.
    :param metadata: `Well.metadata`.
    :returns: `(Well, InjectorControl)`.
    :raises ValidationError: If none of `target_rate`, `target_bhp`, or `target_thp` is given.
    """
    if target_rate is None and target_bhp is None and target_thp is None:
        raise ValidationError(
            "Supply one of `target_rate`, `target_bhp`, or `target_thp`."
        )

    top_depth, bottom_depth = perforation_depths
    well = Well(
        name=name,
        well_type=WellType.INJECTOR,
        surface_location=surface_location,
        reference_depth=reference_depth
        if reference_depth is not None
        else bottom_depth,
        perforations=(
            Perforation(
                top_depth=top_depth,
                bottom_depth=bottom_depth,
                wellbore_radius=wellbore_radius,
                skin=skin,
                saturation_region=saturation_region,
                connection_factor_override=connection_factor_override,
                connection_factor_multiplier=connection_factor_multiplier,
                direction=direction,
                schedule_status=schedule_status,
            ),
        ),
        preferred_phase=injected_phase,
        pvt_region=pvt_region,
        group=group,
        schedule_status=schedule_status,
        unit_system=unit_system,
        metadata=metadata,
    )

    resolved_limits: typing.List[Limit] = []
    if max_bhp is not None:
        resolved_limits.append(BHPLimit(max_value=max_bhp))
    if max_rate is not None:
        resolved_limits.append(
            RateLimit(quantity=PHASE_RATE_QUANTITY[injected_phase], max_value=max_rate)
        )
    if limits:
        resolved_limits.extend(limits)

    if target_bhp is not None:
        control = InjectorControl(
            injected_phase=injected_phase,
            mode=InjectorControlMode.BHP,
            target_bhp=target_bhp,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    elif target_thp is not None:
        control = InjectorControl(
            injected_phase=injected_phase,
            mode=InjectorControlMode.THP,
            target_thp=target_thp,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    else:
        control = InjectorControl(
            injected_phase=injected_phase,
            mode=InjectorControlMode.RATE,
            target_rate=target_rate,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )

    return well, control


QUANTITY_GROUP_CONTROL_MODES = {
    RateQuantity.OIL: GroupProducerControlMode.ORAT,
    RateQuantity.WATER: GroupProducerControlMode.WRAT,
    RateQuantity.GAS: GroupProducerControlMode.GRAT,
    RateQuantity.LIQUID: GroupProducerControlMode.LRAT,
    RateQuantity.RESERVOIR: GroupProducerControlMode.RESV,
}


def make_group_control(
    *,
    target_rate: Number,
    quantity: RateQuantity = RateQuantity.OIL,
    injected_phase: typing.Optional[FluidPhase] = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
) -> GroupControl:
    """
    Builds a `GroupControl`.

    :param target_rate: Group rate target.
    :param quantity: Which rate `target_rate` is. Ignored if `injected_phase` is given.
    :param injected_phase: If given, builds an injection `GroupControl`
        (`mode=GroupInjectorControlMode.RATE`) instead of a production one.
    :param unit_system: `GroupControl.unit_system`.
    :returns: Constructed `GroupControl`.
    """
    if injected_phase is not None:
        return GroupControl(
            mode=GroupInjectorControlMode.RATE,
            target_rate=target_rate,
            injected_phase=injected_phase,
            unit_system=unit_system,
        )
    return GroupControl(
        mode=QUANTITY_GROUP_CONTROL_MODES[quantity],
        target_rate=target_rate,
        unit_system=unit_system,
    )


def make_wells(*wells: Well) -> Wells:
    """
    Builds a `Wells` from a set of `Well` objects.

    :param wells: `Well` objects to include, each already carrying its own name.
    :returns: `Wells`, keyed by each well's own `.name`.
    """
    return Wells(wells={well.name: well for well in wells})


def make_well_controls(**controls: WellControl) -> WellControls:
    """
    Builds a `WellControls` from a set of named controls.

    :param controls: `well_name=control` keyword pairs.
    :returns: `WellControls`, keyed by the given names.
    """
    return WellControls(controls=dict(controls))
