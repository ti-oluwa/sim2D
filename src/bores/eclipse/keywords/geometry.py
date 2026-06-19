"""
Geometry and coordinate-system keyword implementations.

All keywords in this module produce pure Python values (dicts, NumPy arrays)
from their `Record` bodies.
"""

import typing

import numpy as np

from bores.eclipse.core import Deck, DeckParseError, GridDimensions, tokenise
from bores.eclipse.keywords.base import Field, Keyword, RecordKeyword
from bores.typing import FloatArray, ThreeDimensions

__all__ = [
    "SpecGrid",
    "Dimens",
    "Coord",
    "ZCorn",
    "MapAxes",
    "GridUnit",
    "MapUnits",
    "Pinch",
]


# Dimension keywords


class SpecGrid(RecordKeyword):
    """`SPECGRID NX NY NZ [NUMRES [GFLAG]] /` - structured grid dimensions."""

    def __init__(self) -> None:
        super().__init__(
            "SPECGRID",
            fields=[
                Field("nx", int),
                Field("ny", int),
                Field("nz", int),
                Field("numres", int, required=False, default=1),
                Field("gflag", str, required=False, default="F"),
            ],
        )


class Dimens(RecordKeyword):
    """`DIMENS NX NY NZ /` - structured grid dimensions (non-corner-point)."""

    def __init__(self) -> None:
        super().__init__(
            "DIMENS",
            fields=[
                Field("nx", int),
                Field("ny", int),
                Field("nz", int),
            ],
        )


# Corner-point arrays (own Keyword subclasses because their shapes depend
# on grid dimensions in a non-trivial way)


class Coord(Keyword[FloatArray[ThreeDimensions]]):
    """
    `COORD` - corner-point pillar array.

    :returns: Shape `(ny+1, nx+1, 6)` float64 array in C order (i.e.
        j-index outermost, pillar-coordinate innermost).  Each pillar stores
        `[x_top, y_top, z_top, x_bot, y_bot, z_bot]`.
    """

    def __init__(self) -> None:
        super().__init__("COORD")

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[FloatArray[ThreeDimensions]]:
        if dims is None:
            raise DeckParseError("COORD requires grid dimensions (SPECGRID/DIMENS).")

        record = deck.first_record_for(self.name)
        if record is None:
            return None

        tokens = tokenise(record.body)
        expected = (dims.nx + 1) * (dims.ny + 1) * 6
        if len(tokens) != expected:
            raise DeckParseError(
                f"COORD expected {expected} values for a {dims.nx}x{dims.ny} "
                f"grid; got {len(tokens)}."
            )

        # Eclipse stores pillars in Fortran order (i fastest, then j).
        # reshape(nx+1, ny+1, 6, order='F') → shape (nx+1, ny+1, 6), i-fastest.
        # transpose(1, 0, 2) → (ny+1, nx+1, 6) C-order.
        return (
            np.array(tokens, dtype=np.float64)
            .reshape(dims.nx + 1, dims.ny + 1, 6, order="F")
            .transpose(1, 0, 2)
        )


class ZCorn(Keyword[FloatArray[ThreeDimensions]]):
    """
    `ZCORN` - corner-point depth array.

    :returns: Shape `(nz*2, ny*2, nx*2)` float64 array in C order
        (k-index outermost, i-index innermost within each layer pair).
    """

    def __init__(self) -> None:
        super().__init__("ZCORN")

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[FloatArray[ThreeDimensions]]:
        if dims is None:
            raise DeckParseError("ZCORN requires grid dimensions (SPECGRID/DIMENS).")

        record = deck.first_record_for(self.name)
        if record is None:
            return None

        tokens = tokenise(record.body)
        expected = dims.nx * dims.ny * dims.nz * 8
        if len(tokens) != expected:
            raise DeckParseError(
                f"ZCORN expected {expected} values for a "
                f"{dims.nx}x{dims.ny}x{dims.nz} grid; got {len(tokens)}."
            )
        # Eclipse stores ZCORN in Fortran order: (nx*2) fastest, then ny*2, nz*2.
        # reshape(nx*2, ny*2, nz*2, order='F') → i-fastest in memory.
        # transpose(2, 1, 0) → (nz*2, ny*2, nx*2) C-order.
        return (
            np.array(tokens, dtype=np.float64)
            .reshape(dims.nx * 2, dims.ny * 2, dims.nz * 2, order="F")
            .transpose(2, 1, 0)
        )


# Coordinate-system and unit keywords


class MapAxes(RecordKeyword):
    """
    `MAPAXES X_YAXIS Y_YAXIS X_ORIGIN Y_ORIGIN X_XAXIS Y_XAXIS /`
    - map coordinate system axes.

    Eclipse stores six floats in the order:
    `(Y-axis point X, Y-axis point Y, origin X, origin Y, X-axis point X, X-axis point Y)`.

    The parsed dict uses those field names verbatim so callers can decide
    how to assemble their own `MapAxes` object.
    """

    def __init__(self) -> None:
        super().__init__(
            "MAPAXES",
            fields=[
                Field("y_axis_x", float),
                Field("y_axis_y", float),
                Field("origin_x", float),
                Field("origin_y", float),
                Field("x_axis_x", float),
                Field("x_axis_y", float),
            ],
        )


class GridUnit(RecordKeyword):
    """`GRIDUNIT 'UNIT' ['TYPE'] /` - geometry length unit declaration."""

    def __init__(self) -> None:
        super().__init__(
            "GRIDUNIT",
            fields=[
                Field("unit", str),
                Field("type", str, required=False, default=""),
            ],
        )


class MapUnits(RecordKeyword):
    """`MAPUNITS 'UNIT' /` - map coordinate unit declaration."""

    def __init__(self) -> None:
        super().__init__(
            "MAPUNITS",
            fields=[Field("unit", str)],
        )


class Pinch(RecordKeyword):
    """`PINCH thickness [gap_mode [pinchout_option [multz_option [comp_option]]]] /`
    - pinchout handling parameters."""

    def __init__(self) -> None:
        super().__init__(
            "PINCH",
            fields=[
                Field("thickness", float, required=False, default=1e-6),
                Field("gap_mode", str, required=False, default="GAP"),
                Field("pinchout_option", str, required=False, default="TOPBOT"),
                Field("multz_option", str, required=False, default="TOP"),
                Field("comp_option", str, required=False, default="COMPZ"),
            ],
        )
