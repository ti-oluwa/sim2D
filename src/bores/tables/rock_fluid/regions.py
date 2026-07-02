import logging
import typing

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import UnitConversionTable
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.stores import StoreSerializable
from bores.tables.rock_fluid.base import RockFluidTables
from bores.tables.rock_fluid.capillary_pressure.base import (
    ThreePhaseCapillaryPressureTable,
)
from bores.tables.rock_fluid.relperm.base import MinimumRelPerm, ThreePhaseRelPermTable
from bores.tables.rock_fluid.relperm.mixing_rules import MixingRule
from bores.typing import Spacing, UnitSystem

logger = logging.getLogger(__name__)

__all__ = ["RockFluidRegions"]


class RockFluidRegions(StoreSerializable):
    """
    Multi-region rock-fluid tables keyed by 1-based `SATNUM` region index.

    Eclipse supports multiple saturation function regions via the `SATNUM`
    keyword - each cell is assigned a region index and its relative
    permeability and capillary pressure are evaluated from the corresponding
    `RockFluidTables` instance.

    Use `for_region(satnum)` to retrieve the tables for a given region, and
    `from_deck_file` to construct from a deck.
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        regions: typing.Dict[int, RockFluidTables],
    ) -> None:
        """
        Build a `RockFluidRegions` from a pre-built regions dict.

        :param regions: Mapping from 1-based SATNUM index to `RockFluidTables`.
        :raises ValidationError: If *regions* is empty.
        """
        if not regions:
            raise ValidationError("`regions` must contain at least one entry.")
        self.regions = regions

    @property
    def n_regions(self) -> int:
        """Number of saturation function regions."""
        return len(self.regions)

    def for_region(self, satnum: int) -> RockFluidTables:
        """
        Return the `RockFluidTables` for a given 1-based region index.

        :param satnum: 1-based SATNUM region index.
        :returns: `RockFluidTables` for that region.
        :raises KeyError: If the region index does not exist.
        """
        tables = self.regions.get(satnum)
        if tables is None:
            available = sorted(self.regions.keys())
            raise KeyError(
                f"Saturation function region {satnum} not found. "
                f"Available regions: {available}."
            )
        return tables

    @classmethod
    def single_region(cls, tables: RockFluidTables) -> Self:
        """
        Wrap a single `RockFluidTables` as region 1.

        Convenience factory for the common single-region case.

        :param tables: `RockFluidTables` instance.
        :returns: `RockFluidRegions` with one entry at key 1.
        """
        return cls(regions={1: tables})

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `RockFluidRegions` with capillary pressure in every
        region rescaled to *target*.

        Relative permeability is dimensionless and is unaffected.

        :param target: Target `UnitSystem`.
        :returns: New `RockFluidRegions` in *target* units.
        """
        return self.__class__(
            regions={
                satnum: tables.convert(target, table=table)
                for satnum, tables in self.regions.items()
            }
        )

    @classmethod
    def from_deck_file(
        cls,
        deck_file: DeckFile,
        *,
        mixing_rule: typing.Optional[typing.Union[MixingRule, str]] = None,
        keyword_family: typing.Literal["first", "second", "auto"] = "auto",
        number_of_base_points: int = 200,
        number_of_endpoint_extra_points: int = 20,
        capillary_number_of_endpoint_extra_points: int = 30,
        spacing: Spacing = "cosine",
        min_wetting_relperm: MinimumRelPerm = None,
        min_non_wetting_relperm: MinimumRelPerm = None,
        include_capillary_pressure: bool = True,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build all saturation function regions from a parsed `DeckFile`.

        Detects which Eclipse saturation-function keyword family is present
        (`SWOF`/`SGOF` or `SWFN`/`SGFN` + `SOF2`/`SOF3`) and builds one
        `RockFluidTables` per `SATNUM` region, each combining a
        `ThreePhaseRelPermTable` and (when available) a
        `ThreePhaseCapillaryPressureTable`.

        The number of regions is inferred from the longest region list across
        the relevant keywords (`SWOF`/`SWFN` for oil-water, `SGOF`/`SGFN` for
        gas-oil).

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param mixing_rule: Three-phase oil mixing rule (name or callable),
            applied to every region's relative permeability table.
        :param keyword_family: `"first"`, `"second"`, or `"auto"` (default).
        :param number_of_base_points: Passed to relperm PCHIP grid scaling.
        :param number_of_endpoint_extra_points: Passed to relperm PCHIP endpoint
            enrichment.
        :param capillary_number_of_endpoint_extra_points: Passed to capillary
            pressure PCHIP endpoint enrichment (defaults higher, matching
            `TwoPhaseCapillaryPressureTable`'s own default).
        :param spacing: Grid spacing mode for PCHIP scaling.
        :param min_wetting_relperm: Min-value sentinel for wetting phase kr in
            every region's relperm sub-tables.
        :param min_non_wetting_relperm: Min-value sentinel for non-wetting phase
            kr in every region's relperm sub-tables.
        :param include_capillary_pressure: When `True` (default), also build
            the capillary pressure table for each region. When `False`, skip
            capillary pressure entirely (useful for runs that ignore Pc).
        :param dtype: Array dtype shared by every region's tables.
        :returns: `RockFluidRegions` keyed by 1-based SATNUM index.
        :raises ValidationError: If no recognised saturation-function keywords
            are found.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()

        def _region_count(keyword: str) -> int:
            all_regions = deck_file.get(keyword)
            return len(all_regions) if all_regions is not None else 0

        n_regions = max(
            _region_count("SWOF"),
            _region_count("SGOF"),
            _region_count("SWFN"),
            _region_count("SGFN"),
        )
        if n_regions == 0:
            raise ValidationError(
                "No recognised saturation-function keywords found in the DeckFile. "
                "Expected one of: SWOF, SGOF (first family) or SWFN, SGFN "
                "(second family)."
            )

        regions: typing.Dict[int, RockFluidTables] = {}
        for region_index in range(n_regions):
            satnum = region_index + 1  # 1-based

            relative_permeability = ThreePhaseRelPermTable.from_deck_file(
                deck_file=deck_file,
                region_index=region_index,
                mixing_rule=mixing_rule,
                keyword_family=keyword_family,
                number_of_base_points=number_of_base_points,
                number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                spacing=spacing,
                min_wetting_relperm=min_wetting_relperm,
                min_non_wetting_relperm=min_non_wetting_relperm,
                dtype=dtype,
            )

            capillary_pressure: typing.Optional[ThreePhaseCapillaryPressureTable] = None
            if include_capillary_pressure:
                capillary_pressure = ThreePhaseCapillaryPressureTable.from_deck_file(
                    deck_file=deck_file,
                    region_index=region_index,
                    keyword_family=keyword_family,
                    number_of_base_points=number_of_base_points,
                    number_of_endpoint_extra_points=(
                        capillary_number_of_endpoint_extra_points
                    ),
                    spacing=spacing,
                    dtype=dtype,
                )

            regions[satnum] = RockFluidTables(
                relative_permeability=relative_permeability,
                capillary_pressure=capillary_pressure,
            )
            logger.debug(
                "Built rock-fluid tables for SATNUM region %d (capillary_pressure=%s)",
                satnum,
                include_capillary_pressure,
            )
        return cls(regions=regions)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "regions": {
                str(satnum): tables.dump(recurse)
                for satnum, tables in self.regions.items()
            }
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        regions = {
            int(satnum): RockFluidTables.load(table_data)
            for satnum, table_data in data["regions"].items()
        }
        return cls(regions=regions)
