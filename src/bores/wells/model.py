"""The wells sub-system bundle."""

import typing

import attrs
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.stores.base import StoreSerializable
from bores.typing import UnitConversionTable, UnitSystem
from bores.wells.base import Wells
from bores.wells.controls import WellControls
from bores.wells.groups import GroupControls, WellGroups
from bores.wells.hydraulics.base import Wellbore
from bores.wells.resolution.base import ControlResolverSpec

__all__ = ["WellSystem"]


@attrs.frozen(kw_only=True, slots=True)
class WellSystem(StoreSerializable):
    """The wells sub-system"""

    wells: Wells
    well_controls: WellControls
    default_wellbore: Wellbore
    wellbore_overrides: typing.Mapping[str, Wellbore] = attrs.field(factory=dict)
    """
    Per-well hydraulics override (e.g. a gas well on Beggs-Brill while
    every oil well uses the mechanistic no-slip model). Falls back to
    `default_wellbore` for any well not listed here.
    """
    groups: typing.Optional[WellGroups] = None
    group_controls: typing.Optional[GroupControls] = None
    resolver_spec: ControlResolverSpec = attrs.field(factory=ControlResolverSpec)

    def __attrs_post_init__(self) -> None:
        if self.well_controls.unit_system != self.wells.unit_system:
            raise ValidationError(
                f"`well_controls.unit_system` ({self.well_controls.unit_system.value}) != "
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
        """Unit system shared by `wells`/`well_controls`/`group_controls`."""
        return self.wells.unit_system

    def get_wellbore_model(self, well_name: str) -> Wellbore:
        """
        The `Wellbore` to use for `well_name`.

        Uses `wellbore_overrides[well_name]` if present, else `default_wellbore`.
        """
        return self.wellbore_overrides.get(well_name, self.default_wellbore)

    def get_wells_in_group(self, group_name: str) -> typing.Tuple[str, ...]:
        """
        Returns every well name whose `group == group_name` or any group
        under it (recursively, via `self.groups`).

        :raises ValidationError: If `self.groups` is `None`.
        """
        if self.groups is None:
            from bores.errors import ValidationError

            raise ValidationError(
                "`get_wells_in_group` requires `groups` to be set on this WellSystem."
            )
        member_group_names = {group_name, *self.groups.descendants(group_name)}
        return tuple(
            name for name in self.wells if self.wells[name].group in member_group_names
        )

    @classmethod
    def from_deck(
        cls, deck_file: DeckFile, *, grid: Grid, default_wellbore: Wellbore
    ) -> Self:
        """
        Load a `WellSystem` from a parsed `DeckFile`.

        :param deck_file: Parsed deck.
        :param grid: `Grid` built from the same deck.
        :param default_wellbore: `Wellbore` that describes the wells' hydraulics.
            Defaults to `MechanisticWellbore` if not provided.
        :param well_kwargs: Forwarded to `Wells.from_deck`.
        :returns: `WellSystem` built from every well/control/group keyword
            present in the deck. groups/group_controls are None if the
            deck has no `GRUPTREE`/`GCONPROD`/`GCONINJE`.
        """
        groups = WellGroups.from_deck(deck_file) if deck_file.has("GRUPTREE") else None
        group_controls = (
            GroupControls.from_deck(deck_file)
            if (deck_file.has("GCONPROD") or deck_file.has("GCONINJE"))
            else None
        )
        return cls(
            wells=Wells.from_deck(deck_file, grid=grid),
            well_controls=WellControls.from_deck(deck_file),
            default_wellbore=default_wellbore,
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
        Returns a  new `WellSystem` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `WellsModel` with wells/well_controls/group_controls
            converted to target. groups (pure hierarchy, no dimensioned
            data) and the wellbore models are unchanged.
        """
        if target == self.unit_system:
            return self
        return attrs.evolve(
            self,
            wells=self.wells.convert(target, table=table),
            well_controls=self.well_controls.convert(target, table=table),
            default_wellbore=self.default_wellbore.convert(target, table=table),
            wellbore_overrides={
                well_name: wellbore.convert(target, table=table)
                for well_name, wellbore in self.wellbore_overrides.items()
            },
            group_controls=(
                self.group_controls.convert(target, table=table)
                if self.group_controls is not None
                else None
            ),
        )
