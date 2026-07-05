"""Black-oil fluid physics model."""

import typing

from typing_extensions import Self

from bores.blackoil.compressibility import RockCompressibilityRegions
from bores.blackoil.pvt.regions import PVTRegions
from bores.blackoil.rock_fluid.regions import RockFluidRegions
from bores.constants import UnitConversionTable, build_unit_conversion_table
from bores.errors import ValidationError
from bores.stores import StoreSerializable
from bores.typing import UnitSystem

__all__ = ["BlackOilModel"]


class BlackOilModel(
    StoreSerializable,
    fields={
        "pvt_regions": PVTRegions,
        "rock_fluid_regions": RockFluidRegions,
        "compressibility_regions": typing.Optional[RockCompressibilityRegions],
        "unit_system": UnitSystem,
    },
):
    """
    Black-oil fluid physics model.

    Holds the three multi-region table objects that define the fluid and
    rock physics for a black-oil simulation:

    - `pvt_regions` - PVT tables and static fluid properties, one
      `PVTRegion` per Eclipse `PVTNUM` region.
    - `rock_fluid_regions` - relative permeability and capillary pressure
      tables, one per Eclipse `SATNUM` region.
    - `compressibility_regions` - rock compressibility tables, one per
      Eclipse `ROCKNUM` region. Optional; `None` when rock compressibility
      is not modelled (cr = 0 everywhere).

    Grid geometry, rock properties, dynamic state, and region metadata
    live on `bores.reservoir.model.ReservoirModel`. This class holds only
    the fluid physics lookup tables.

    All region tables are normalised to `unit_system` at construction.
    Use `convert(target)` to produce a fully rescaled copy.
    """

    def __init__(
        self,
        pvt_regions: PVTRegions,
        rock_fluid_regions: RockFluidRegions,
        compressibility_regions: typing.Optional[RockCompressibilityRegions] = None,
        unit_system: typing.Optional[UnitSystem] = None,
    ) -> None:
        """
        :param pvt_regions: PVT region tables keyed by 1-based `PVTNUM` index.
        :param rock_fluid_regions: Relperm and capillary pressure tables keyed
            by 1-based `SATNUM` index.
        :param compressibility_regions: Rock compressibility tables keyed by
            1-based `ROCKNUM` index. `None` when rock compressibility is not
            modelled.
        :param unit_system: Target unit system for all region tables. When
            `None`, all supplied tables must share the same unit system -
            if they do not, a `ValidationError` is raised. When provided,
            each table group is converted to `unit_system` as needed.
        :raises ValidationError: If `unit_system` is `None` and the supplied
            tables do not all share the same unit system.
        """
        unit_conversion_table = build_unit_conversion_table()

        # Resolve unit systems of each supplied group
        pvt_unit_system = pvt_regions.unit_system
        rock_fluid_unit_system = rock_fluid_regions.unit_system
        compressibility_unit_system = (
            compressibility_regions.unit_system
            if compressibility_regions is not None
            else None
        )

        if unit_system is None:
            # All present groups must agree
            systems = {pvt_unit_system, rock_fluid_unit_system}
            if compressibility_unit_system is not None:
                systems.add(compressibility_unit_system)
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
            pvt_regions = pvt_regions.convert(unit_system, table=unit_conversion_table)
        if rock_fluid_unit_system != unit_system:
            rock_fluid_regions = rock_fluid_regions.convert(
                unit_system, table=unit_conversion_table
            )
        if (
            compressibility_regions is not None
            and compressibility_unit_system != unit_system
        ):
            compressibility_regions = compressibility_regions.convert(
                unit_system, table=unit_conversion_table
            )

        self.pvt_regions = pvt_regions
        """PVT region tables - one `PVTRegion` per `PVTNUM` region."""

        self.rock_fluid_regions = rock_fluid_regions
        """Relperm and capillary pressure tables - one per `SATNUM` region."""

        self.compressibility_regions = compressibility_regions
        """
        Rock compressibility tables - one per `ROCKNUM` region.
        `None` when rock compressibility is not modelled.
        """

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
        Return a new `BlackOilModel` with all region tables rescaled to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `BlackOilModel` in *target* units.
        """
        if target == self.unit_system:
            return self

        return self.__class__(
            pvt_regions=self.pvt_regions.convert(target, table=table),
            rock_fluid_regions=self.rock_fluid_regions.convert(target, table=table),
            compressibility_regions=(
                self.compressibility_regions.convert(target, table=table)
                if self.compressibility_regions is not None
                else None
            ),
            unit_system=target,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_pvt_regions={self.pvt_regions.n_regions}, "
            f"n_rock_fluid_regions={self.rock_fluid_regions.n_regions}, "
            f"n_compressibility_regions="
            f"{self.compressibility_regions.n_regions if self.compressibility_regions is not None else 0}, "
            f"unit_system={self.unit_system.value!r}"  # type: ignore[union-attr]
            f")"
        )
