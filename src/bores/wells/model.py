"""The wells sub-system bundle."""

import typing

import attrs

from bores.serde.base import Serializable
from bores.wells.base import Wells
from bores.wells.controls import WellControls
from bores.wells.groups import GroupControls, Groups
from bores.wells.hydraulics.base import WellboreModel
from bores.wells.resolution.base import ControlResolverSpec

__all__ = ["WellsModel"]


@attrs.frozen(kw_only=True, slots=True)
class WellsModel(Serializable):
    wells: Wells
    controls: WellControls
    default_wellbore_model: WellboreModel
    wellbore_model_overrides: typing.Mapping[str, WellboreModel] = {}
    """
    Per-well hydraulics override (e.g. a gas well on Beggs-Brill while
    every oil well uses the mechanistic no-slip model). Falls back to
    `default_wellbore_model` for any well not listed here.
    """
    groups: typing.Optional[Groups] = None
    group_controls: typing.Optional[GroupControls] = None
    resolver_spec: ControlResolverSpec = attrs.field(factory=ControlResolverSpec)

    def get_wellbore_model(self, well_name: str) -> WellboreModel:
        """
        The `WellboreModel` to use for `well_name`.

        Uses `wellbore_model_overrides[well_name]` if present, else
        `default_wellbore_model`.
        """
        return self.wellbore_model_overrides.get(well_name, self.default_wellbore_model)

    def wells_in_group(self, group_name: str) -> typing.Tuple[str, ...]:
        """
        Every well name whose `Well.group` is `group_name` or any group
        under it (recursively, via `self.groups`).

        :raises ValidationError: If `self.groups` is `None`.
        """
        if self.groups is None:
            from bores.errors import ValidationError

            raise ValidationError(
                "wells_in_group requires `groups` to be set on this WellsModel."
            )
        member_group_names = {group_name, *self.groups.descendants(group_name)}
        return tuple(
            name for name in self.wells if self.wells[name].group in member_group_names
        )
