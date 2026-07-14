"""
Group target allocation. Logic for spliting a `GroupControl` target rate across
member wells by guide rate, producing concrete per-well WellControl
objects for wells whose mode is GRUP.

First-pass proportional allocation only: does not iterate to redistribute
a shortfall when an allocated well can't actually reach its share (e.g.
hits its own BHP limit). That redistribution loop is not implemented.
"""

import typing

import attrs

from bores.errors import ValidationError
from bores.typing import Number
from bores.wells.controls import (
    InjectorControl,
    InjectorControlMode,
    ProducerControl,
    ProducerControlMode,
    WellControl,
)
from bores.wells.groups import GroupInjectorControlMode, GroupProducerControlMode
from bores.wells.model import WellsModel

__all__ = ["allocate_group_targets"]

PRODUCER_MODE_MAP: typing.Dict[GroupProducerControlMode, ProducerControlMode] = {
    GroupProducerControlMode.ORAT: ProducerControlMode.ORAT,
    GroupProducerControlMode.WRAT: ProducerControlMode.WRAT,
    GroupProducerControlMode.GRAT: ProducerControlMode.GRAT,
    GroupProducerControlMode.LRAT: ProducerControlMode.LRAT,
    GroupProducerControlMode.RESV: ProducerControlMode.RESV,
}
INJECTOR_MODE_MAP: typing.Dict[GroupInjectorControlMode, InjectorControlMode] = {
    GroupInjectorControlMode.RATE: InjectorControlMode.RATE,
    GroupInjectorControlMode.RESV: InjectorControlMode.RESV,
}


def allocate_group_targets(
    group_name: str, wells_model: WellsModel
) -> typing.Dict[str, WellControl]:
    """
    Allocate `group_name`'s current `GroupControl` target across its
    member wells whose control mode is `GRUP`, by guide rate.

    Writes the resulting per-well `WellControl` (mode switched to a
    concrete rate mode, `target_rate` set to the well's share) into
    `wells_model.controls` via `WellControls.set`.

    :param group_name: Group to allocate.
    :param wells_model: Supplies `wells_model.groups`, `wells_model.group_controls`,
        `wells_model.controls`, and `wells_model.wells_in_group`.
    :returns: Mapping from well name to its updated `WellControl`, for the
        wells actually allocated (empty if none are eligible).
    :raises ValidationError: If `wells_model.group_controls` is `None`, or
        `group_name` has no `GroupControl` set, or its `mode` has no
        allocatable target (`FLD`/`NONE`/`VREP`/`REIN`).
    """
    if wells_model.group_controls is None:
        raise ValidationError("`wells_model.group_controls` is not set.")

    group_control = wells_model.group_controls.get(group_name)
    if group_control is None:
        raise ValidationError(f"No `GroupControl` set for group {group_name!r}.")

    is_injection = isinstance(group_control.mode, GroupInjectorControlMode)
    if is_injection:
        target_mode = INJECTOR_MODE_MAP.get(group_control.mode)  # type: ignore[arg-type]
    else:
        target_mode = PRODUCER_MODE_MAP.get(group_control.mode)  # type: ignore[arg-type]

    if target_mode is None:
        raise ValidationError(
            f"Group {group_name!r}'s control mode {group_control.mode} has no "
            "directly allocatable rate target."
        )
    if group_control.target_rate is None:
        raise ValidationError(f"Group {group_name!r}'s control has no target_rate.")

    member_names = wells_model.wells_in_group(group_name)
    eligible: typing.List[str] = []
    for name in member_names:
        control = wells_model.controls.get(name)
        if control is None:
            continue
        if is_injection:
            if not isinstance(control, InjectorControl):
                continue
            if control.mode is not InjectorControlMode.GRUP:
                continue
        else:
            if not isinstance(control, ProducerControl):
                continue
            if control.mode is not ProducerControlMode.GRUP:
                continue
        eligible.append(name)

    if not eligible:
        return {}

    weights: typing.Dict[str, Number] = {}
    for name in eligible:
        guide_rate = wells_model.controls[name].guide_rate
        if guide_rate is not None:
            weights[name] = guide_rate
        else:
            weights[name] = 1.0
    total_weight = sum(weights.values())

    updated: typing.Dict[str, WellControl] = {}
    for name in eligible:
        share = group_control.target_rate * (weights[name] / total_weight)
        current = wells_model.controls[name]
        new_control = attrs.evolve(current, mode=target_mode, target_rate=share)
        wells_model.controls.set(name, new_control)
        updated[name] = new_control

    return updated
