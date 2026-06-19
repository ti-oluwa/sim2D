import numpy as np

from bores.eclipse.keywords.base import GridArrayKeyword

__all__ = ["SATNUM", "PVTNUM", "EQLNUM"]


SATNUM = GridArrayKeyword("SATNUM", dtype=np.int32, default_value=1)
"""`SATNUM` - saturation function region number (1-based int)."""

PVTNUM = GridArrayKeyword("PVTNUM", dtype=np.int32, default_value=1)
"""`PVTNUM` - PVT region number (1-based int)."""

EQLNUM = GridArrayKeyword("EQLNUM", dtype=np.int32, default_value=1)
"""`EQLNUM` - equilibration region number (1-based int)."""

FIPNUM = GridArrayKeyword("FIPNUM", dtype=np.int32, default_value=1)
"""`FIPNUM` - fluid-in-place region number (1-based int)."""
