"""The wells sub-system bundle."""

import typing

import attrs
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.base import Serializable
from bores.typing import UnitConversionTable, UnitSystem
from bores.wells.base import Wells
from bores.wells.controls import WellControls
from bores.wells.groups import GroupControls, WellGroups
from bores.wells.hydraulics.base import WellboreModel
from bores.wells.resolution.base import ControlResolverSpec

__all__ = ["WellModel"]


@attrs.frozen(kw_only=True, slots=True)
class WellModel(Serializable):
    wells: Wells
    controls: WellControls
    wellbore_model: WellboreModel
    wellbore_model_overrides: typing.Mapping[str, WellboreModel] = attrs.field(
        factory=dict
    )
    """
    Per-well hydraulics override (e.g. a gas well on Beggs-Brill while
    every oil well uses the mechanistic no-slip model). Falls back to
    `wellbore_model` for any well not listed here.
    """
    groups: typing.Optional[WellGroups] = None
    group_controls: typing.Optional[GroupControls] = None
    resolver_spec: ControlResolverSpec = attrs.field(factory=ControlResolverSpec)

    def __attrs_post_init__(self) -> None:
        if self.controls.unit_system != self.wells.unit_system:
            raise ValidationError(
                f"`controls.unit_system` ({self.controls.unit_system.value}) != "
                f"`wells.unit_system` ({self.wells.unit_system.value})."
            )
        if (
            self.group_controls is not None
            and self.group_controls.unit_system != self.wells.unit_system
        ):
            raise ValidationError(
                f"`group_controls.unit_system` ({self.group_controls.unit_system.value}) "
                f"!= `wells.unit_system` ({self.wells.unit_system.value})."
            )

    @property
    def unit_system(self) -> UnitSystem:
        """Unit system shared by wells/controls/group_controls."""
        return self.wells.unit_system

    def get_wellbore_model(self, well_name: str) -> WellboreModel:
        """
        The `WellboreModel` to use for `well_name`.

        Uses `wellbore_model_overrides[well_name]` if present, else `wellbore_model`.
        """
        return self.wellbore_model_overrides.get(well_name, self.wellbore_model)

    def wells_in_group(self, group_name: str) -> typing.Tuple[str, ...]:
        """
        Returns every well name whose `group == group_name` or any group
        under it (recursively, via `self.groups`).

        :raises ValidationError: If `self.groups` is `None`.
        """
        if self.groups is None:
            from bores.errors import ValidationError

            raise ValidationError(
                "wells_in_group requires `groups` to be set on this WellModel."
            )
        member_group_names = {group_name, *self.groups.descendants(group_name)}
        return tuple(
            name for name in self.wells if self.wells[name].group in member_group_names
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        grid: Grid,
        wellbore_model: WellboreModel,
        **well_kwargs: typing.Any,
    ) -> Self:
        """
        Load a `WellModel` from a parsed `DeckFile`.

        :param deck_file: Parsed deck.
        :param grid: `Grid` built from the same deck.
        :param wellbore_model: `WellModel.wellbore_model`.
        :param well_kwargs: Forwarded to `Wells.from_deck`.
        :returns: `WellModel` built from every well/control/group keyword
            present in the deck. groups/group_controls are None if the
            deck has no GRUPTREE/GCONPROD/GCONINJE.
        """
        groups = WellGroups.from_deck(deck_file) if deck_file.has("GRUPTREE") else None
        group_controls = (
            GroupControls.from_deck(deck_file)
            if (deck_file.has("GCONPROD") or deck_file.has("GCONINJE"))
            else None
        )
        return cls(
            wells=Wells.from_deck(deck_file, grid=grid, **well_kwargs),
            controls=WellControls.from_deck(deck_file),
            wellbore_model=wellbore_model,
            groups=groups,
            group_controls=group_controls,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Returns a  new `WellModel` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New WellsModel with wells/controls/group_controls
            converted to target. groups (pure hierarchy, no dimensioned
            data) and the wellbore models are unchanged.
        """
        if target == self.unit_system:
            return self
        return attrs.evolve(
            self,
            wells=self.wells.convert(target, table=table),
            controls=self.controls.convert(target, table=table),
            group_controls=(
                self.group_controls.convert(target, table=table)
                if self.group_controls is not None
                else None
            ),
        )
