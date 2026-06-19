import numpy as np

from bores.deck.keywords.base import GridArrayKeyword

__all__ = ["SATNUM", "PVTNUM", "EQLNUM", "ROCKNUM", "IMBNUM"]


SATNUM = GridArrayKeyword("SATNUM", dtype=np.int32, default_value=1)
"""`SATNUM` - saturation function region number (1-based int)."""

PVTNUM = GridArrayKeyword("PVTNUM", dtype=np.int32, default_value=1)
"""`PVTNUM` - PVT region number (1-based int)."""

EQLNUM = GridArrayKeyword("EQLNUM", dtype=np.int32, default_value=1)
"""`EQLNUM` - equilibration region number (1-based int)."""

FIPNUM = GridArrayKeyword("FIPNUM", dtype=np.int32, default_value=1)
"""`FIPNUM` - fluid-in-place region number (1-based int)."""

ROCKNUM = GridArrayKeyword("ROCKNUM", dtype=np.int32, default_value=1)
"""
`ROCKNUM` - rock-compaction region number (1-based int).
 
Selects which `ROCKTAB` (or single `ROCK` compressibility record) table
applies to each cell, when more than one rock region is defined. Default
is `1` (the first/only rock region) when absent.
"""

IMBNUM = GridArrayKeyword("IMBNUM", dtype=np.int32, default_value=1)
"""
`IMBNUM` - imbibition saturation-function region number (1-based int).
 
Selects the imbibition (re-saturation) relative-permeability/capillary-
pressure table for hysteresis modelling, paired with the drainage table
selected by `SATNUM`. Default is `1` when absent.
"""
