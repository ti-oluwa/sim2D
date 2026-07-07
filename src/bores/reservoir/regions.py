import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.serialization.stores import StoreSerializable
from bores.typing import IntCellArray

__all__ = ["Regions"]


def _load_region_array(
    deck_file: DeckFile,
    keyword: str,
    n_cells: int,
    default_region: typing.Optional[int] = None,
) -> typing.Optional[IntCellArray]:
    arr = deck_file.get(keyword)
    if arr is None:
        if default_region is not None:
            return np.full(n_cells, default_region, dtype=np.int32)
        return None

    arr = arr.astype(np.int32, copy=False)
    if arr.size != n_cells:
        raise ValidationError(f"{keyword} has {arr.size} values; expected {n_cells}.")
    return typing.cast(IntCellArray, arr)


@attrs.frozen(slots=True)
class Regions(StoreSerializable):
    """
    Reservoir per-cell region assignments metadata.

    Populated from the REGIONS section of an Eclipse deck, or supplied
    directly by the user. All region arrays are 1-based integer indices
    selecting which PVT, saturation-function, equilibration, or rock
    compaction table applies to each cell.

    All fields are optional, but an important behaviour to note is that when absent,
    region 1 is assumed for every cell in the `get_*_region` methods.
    """

    pvt_regions: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - PVT region number per cell (1-based).
    Selects which PVTTables entry from PVTRegions applies.
    Read from PVTNUM. Default: 1 everywhere.
    """

    saturation_regions: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - saturation function region number (1-based).
    Selects SWOF/SGOF/SWFN/SGFN table. Read from SATNUM.
    """

    imbibition_regions: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - imbibition saturation function region number (1-based).
    Used for hysteresis scanning curves. Read from IMBNUM.
    """

    equilibrium_regions: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - equilibration region number (1-based).
    Selects which EQUIL record governs initialisation. Read from EQLNUM.
    """

    rock_regions: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - rock compaction region number (1-based).
    Selects ROCK/ROCKTAB table. Read from ROCKNUM.
    """

    fluid_in_place_regions: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - fluid-in-place reporting region (1-based).
    Controls which cells contribute to ROIP/RGIP/RWIP output groups.
    Read from FIPNUM.
    """

    @classmethod
    def from_deck(
        cls, deck_file: DeckFile, *, n_cells: int, use_default: bool = False
    ) -> Self:
        """
        Build `Regions` from a parsed DeckFile.

        Missing keywords default to None (region 1 is assumed by callers).

        :param deck_file: Parsed DeckFile.
        :param n_cells: Number of active cells, for validation.
        :param use_default: If True, missing keywords are filled with region 1.
            Else, they are left as None. Default: False.
        :returns: `Regions` object loaded from ECLIPSE deck.
        """
        default_region = 1 if use_default else None
        return cls(
            pvt_regions=_load_region_array(
                deck_file, "PVTNUM", n_cells=n_cells, default_region=default_region
            ),
            saturation_regions=_load_region_array(
                deck_file, "SATNUM", n_cells=n_cells, default_region=default_region
            ),
            imbibition_regions=_load_region_array(
                deck_file, "IMBNUM", n_cells=n_cells, default_region=default_region
            ),
            equilibrium_regions=_load_region_array(
                deck_file, "EQLNUM", n_cells=n_cells, default_region=default_region
            ),
            rock_regions=_load_region_array(
                deck_file, "ROCKNUM", n_cells=n_cells, default_region=default_region
            ),
            fluid_in_place_regions=_load_region_array(
                deck_file, "FIPNUM", n_cells=n_cells, default_region=default_region
            ),
        )

    def get_pvt_region(self, cell_idx: int) -> int:
        """Return the PVT region for a cell, defaulting to 1."""
        if self.pvt_regions is None:
            return 1
        return int(self.pvt_regions[cell_idx])

    def get_saturation_region(self, cell_idx: int) -> int:
        """Return the saturation function region for a cell, defaulting to 1."""
        if self.saturation_regions is None:
            return 1
        return int(self.saturation_regions[cell_idx])

    def get_imbibition_region(self, cell_idx: int) -> int:
        """Return the imbibition region for a cell, defaulting to 1."""
        if self.imbibition_regions is None:
            return 1
        return int(self.imbibition_regions[cell_idx])

    def get_equilibration_region(self, cell_idx: int) -> int:
        """Return the equilibration region for a cell, defaulting to 1."""
        if self.equilibrium_regions is None:
            return 1
        return int(self.equilibrium_regions[cell_idx])

    def get_fluid_in_place_region(self, cell_idx: int) -> int:
        """Return the fluid-in-place region for a cell, defaulting to 1."""
        if self.fluid_in_place_regions is None:
            return 1
        return int(self.fluid_in_place_regions[cell_idx])

    def get_rock_region(self, cell_idx: int) -> int:
        """Return the rock function region for a cell, defaulting to 1."""
        if self.rock_regions is None:
            return 1
        return int(self.rock_regions[cell_idx])
