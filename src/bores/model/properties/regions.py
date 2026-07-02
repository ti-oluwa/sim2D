import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.stores import StoreSerializable
from bores.typing import IntCellArray

__all__ = ["Regions"]


def _load_region(
    deck_file: DeckFile, keyword: str, n_cells: int
) -> typing.Optional[IntCellArray]:
    arr = deck_file.get(keyword)
    if arr is None:
        return None

    arr = arr.astype(np.int32, copy=False)
    if arr.size != n_cells:
        raise ValidationError(f"{keyword} has {arr.size} values; expected {n_cells}.")
    return typing.cast(IntCellArray, arr)


@attrs.frozen(slots=True)
class Regions(StoreSerializable):
    """
    Per-cell region assignments metadata.

    Populated from the REGIONS section of an Eclipse deck, or supplied
    directly by the user. All region arrays are 1-based integer indices
    selecting which PVT, saturation-function, equilibration, or rock
    compaction table applies to each cell.

    All fields are optional - when absent, region 1 is assumed for every
    cell (Eclipse default behaviour).
    """

    pvt_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - PVT region index per cell (1-based).
    Selects which PVTTables entry from PVTRegions applies.
    Read from PVTNUM. Default: 1 everywhere.
    """

    saturation_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - saturation function region index (1-based).
    Selects SWOF/SGOF/SWFN/SGFN table. Read from SATNUM.
    """

    imbibition_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - imbibition saturation function region index (1-based).
    Used for hysteresis scanning curves. Read from IMBNUM.
    """

    equilibration_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - equilibration region index (1-based).
    Selects which EQUIL record governs initialisation. Read from EQLNUM.
    """

    rock_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - rock compaction region index (1-based).
    Selects ROCK/ROCKTAB table. Read from ROCKNUM.
    """

    fluid_in_place_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - fluid-in-place reporting region (1-based).
    Controls which cells contribute to ROIP/RGIP/RWIP output groups.
    Read from FIPNUM.
    """

    @classmethod
    def from_deck_file(cls, deck_file: DeckFile, n_cells: int) -> Self:
        """
        Build `Regions` from a parsed DeckFile.

        Missing keywords default to None (region 1 is assumed by callers).

        :param deck_file: Parsed DeckFile.
        :param n_cells: Number of active cells, for validation.
        :returns: `Regions` object loaded from ECLIPSE deck.
        """
        return cls(
            pvt_region=_load_region(deck_file, "PVTNUM", n_cells),
            saturation_region=_load_region(deck_file, "SATNUM", n_cells),
            imbibition_region=_load_region(deck_file, "IMBNUM", n_cells),
            equilibration_region=_load_region(deck_file, "EQLNUM", n_cells),
            rock_region=_load_region(deck_file, "ROCKNUM", n_cells),
            fluid_in_place_region=_load_region(deck_file, "FIPNUM", n_cells),
        )

    def get_pvt_region(self, cell_index: int) -> int:
        """Return the PVT region for a cell, defaulting to 1."""
        if self.pvt_region is None:
            return 1
        return int(self.pvt_region[cell_index])

    def get_saturation_region(self, cell_index: int) -> int:
        """Return the saturation function region for a cell, defaulting to 1."""
        if self.saturation_region is None:
            return 1
        return int(self.saturation_region[cell_index])

    def get_imbibition_region(self, cell_index: int) -> int:
        """Return the imbibition region for a cell, defaulting to 1."""
        if self.imbibition_region is None:
            return 1
        return int(self.imbibition_region[cell_index])

    def get_equilibration_region(self, cell_index: int) -> int:
        """Return the equilibration region for a cell, defaulting to 1."""
        if self.equilibration_region is None:
            return 1
        return int(self.equilibration_region[cell_index])

    def get_fluid_in_place_region(self, cell_index: int) -> int:
        """Return the fluid-in-place region for a cell, defaulting to 1."""
        if self.fluid_in_place_region is None:
            return 1
        return int(self.fluid_in_place_region[cell_index])

    def get_rock_region(self, cell_index: int) -> int:
        """Return the rock function region for a cell, defaulting to 1."""
        if self.rock_region is None:
            return 1
        return int(self.rock_region[cell_index])
