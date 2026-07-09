"""Black-oil fluid physics model."""

import typing

from typing_extensions import Self

from bores.blackoil.pvt.regions import PVTRegions
from bores.blackoil.rock_fluid.regions import RockFluidRegions
from bores.constants import UnitConversionTable, build_unit_conversion_table
from bores.errors import ValidationError
from bores.serde.stores import StoreSerializable
from bores.typing import UnitSystem

__all__ = ["BlackOilFluid"]


class BlackOilFluid(
    StoreSerializable,
    fields={
        "pvt": PVTRegions,
        "rock_fluid": RockFluidRegions,
        "unit_system": UnitSystem,
    },
):
    """
    Black-oil fluid physics model.

    Holds the two multi-region table objects that define the
    fluid physics for a black-oil simulation:

    - `pvt` - PVT tables and static fluid properties, one
      `PVTRegion` per Eclipse `PVTNUM` region.
    - `rock_fluid` - relative permeability and capillary pressure
      tables, one per Eclipse `SATNUM` region.

    All region tables are normalised to `unit_system` at construction.
    Use `convert(target)` to produce a fully rescaled copy.
    """

    def __init__(
        self,
        pvt: PVTRegions,
        rock_fluid: RockFluidRegions,
        unit_system: typing.Optional[UnitSystem] = None,
    ) -> None:
        """
        :param pvt: PVT region tables keyed by 1-based `PVTNUM` index.
        :param rock_fluid: Relperm and capillary pressure tables keyed
            by 1-based `SATNUM` index.
        :param unit_system: Target unit system for all region tables. When
            `None`, all supplied tables must share the same unit system -
            if they do not, a `ValidationError` is raised. When provided,
            each table group is converted to `unit_system` as needed.
        :raises ValidationError: If `unit_system` is `None` and the supplied
            tables do not all share the same unit system.
        """
        unit_conversion_table = build_unit_conversion_table()

        # Resolve unit systems of each supplied group
        pvt_unit_system = pvt.unit_system
        rock_fluid_unit_system = rock_fluid.unit_system

        if unit_system is None:
            # All present groups must agree
            systems = {pvt_unit_system, rock_fluid_unit_system}
            if len(systems) > 1:
                raise ValidationError(
                    "All region tables must share the same unit system when "
                    "`unit_system` is not explicitly provided. "
                    f"Found: {sorted(s.value for s in systems)}. "
                    "Pass `unit_system` explicitly to convert all tables to a "
                    "common system."
                )
            unit_system = pvt_unit_system

        # Convert each group to the target unit system
        if pvt_unit_system != unit_system:
            pvt = pvt.convert(unit_system, table=unit_conversion_table)
        if rock_fluid_unit_system != unit_system:
            rock_fluid = rock_fluid.convert(unit_system, table=unit_conversion_table)

        self.pvt = pvt
        """PVT region tables - one `PVTRegion` per `PVTNUM` region."""

        self.rock_fluid = rock_fluid
        """Relperm and capillary pressure tables - one per `SATNUM` region."""

        self.unit_system = unit_system
        """Unit system in which all region tables are expressed."""

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `BlackOilFluid` with all region tables rescaled to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `BlackOilFluid` in *target* units.
        """
        if target == self.unit_system:
            return self

        return self.__class__(
            pvt=self.pvt.convert(target, table=table),
            rock_fluid=self.rock_fluid.convert(target, table=table),
            unit_system=target,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_pvt_regions={self.pvt.n_regions}, "
            f"n_rock_fluid_regions={self.rock_fluid.n_regions}, "
            f"unit_system={self.unit_system.value!r}"  # type: ignore[union-attr]
            f")"
        )
