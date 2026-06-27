import typing

import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.typing import FloatArray, IntArray, OneDimension


def resolve_temperature(
    deck_file: DeckFile,
    grid: Grid,
    default: typing.Optional[float] = None,
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> FloatArray[OneDimension]:
    """
    Resolve the initial per-cell reservoir temperature field from the deck.

    :param deck_file: Parsed DataFile.
    :param grid: Grid providing cell centroid depths.
    :param default: Fallback temperature (°F) when nothing is in deck. 
        Will default to `c.STANDARD_TEMPERATURE_IMPERIAL` if not provided.
    :param dtype: NumPy floating dtype for output array. Defaults to `bores.get_dtype()`.
    :returns: (n_cells,) float64 temperature array.
    """
    rtemp = deck_file.get("RTEMP")
    if rtemp is not None and len(rtemp) != 1:
        raise ValidationError("`RTEMP` must contain exactly one table.")

    n_cells = grid.n_cells
    dtype = dtype if dtype is not None else get_dtype()
    if rtemp:
        temperature = float(rtemp[0][0]["temperature"])
        return np.full(n_cells, temperature, dtype=dtype)

    return np.full(
        n_cells,
        default if default is not None else c.STANDARD_TEMPERATURE_IMPERIAL,
        dtype=dtype,
    )
