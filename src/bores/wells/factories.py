import typing

from bores.errors import ValidationError
from bores.typing import FluidPhase
from bores.wells.base import Perforation, Well, Wells, WellType
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
    GroupControls,
    GroupInjectorControlMode,
    GroupProducerControlMode,
    WellGroups,
)
from bores.wells.hydraulics.base import WellboreModel
from bores.wells.model import WellModel

__all__ = ["make_producer", "make_injector"]


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


QUANTITY_CONTROL_MODES = {
    RateQuantity.OIL: GroupProducerControlMode.ORAT,
    RateQuantity.WATER: GroupProducerControlMode.WRAT,
    RateQuantity.GAS: GroupProducerControlMode.GRAT,
    RateQuantity.LIQUID: GroupProducerControlMode.LRAT,
    RateQuantity.RESERVOIR: GroupProducerControlMode.RESV,
}


def make_group_control(
    *,
    target_rate: float,
    quantity: RateQuantity = RateQuantity.OIL,
    injected_phase: typing.Optional[FluidPhase] = None,
) -> GroupControl:
    """
    :param target_rate: WellGroup rate target.
    :param quantity: Which rate quantity target_rate is - selects the
        resulting GroupProducerControlMode. Ignored if injected_phase is given.
    :param injected_phase: If given, builds an injection GroupControl
        (mode=GroupInjectorControlMode.RATE) instead of a production one.
    :returns: Constructed GroupControl.
    """
    if injected_phase is not None:
        return GroupControl(
            mode=GroupInjectorControlMode.RATE,
            target_rate=target_rate,
            injected_phase=injected_phase,
        )
    return GroupControl(mode=QUANTITY_CONTROL_MODES[quantity], target_rate=target_rate)


def make_wells(*wells: Well) -> Wells:
    """
    :param wells: Well objects to include, each already carrying its own name.
    :returns: Wells keyed by each well's own .name.
    """
    return Wells(wells={well.name: well for well in wells})


def make_well_controls(**controls: WellControl) -> WellControls:
    """
    :param controls: well_name=control keyword pairs.
    :returns: WellControls keyed by the given names.
    """
    return WellControls(controls=dict(controls))
