import numpy as np

from bores.eclipse.keywords.base import GridArrayKeyword

__all__ = ["SWAT", "SOIL", "SGAS", "SW", "SO", "SG", "PRESSURE", "RS", "RV"]

SWAT = SW = GridArrayKeyword("SWAT", dtype=np.float64, default_value=0.0)
"""`SWAT` / `SW` - initial water saturation `[0, 1]`."""

SOIL = SO = GridArrayKeyword("SOIL", dtype=np.float64, default_value=0.0)
"""`SOIL` / `SO` - initial oil saturation `[0, 1]`."""

SGAS = SG = GridArrayKeyword("SGAS", dtype=np.float64, default_value=0.0)
"""`SGAS` / `SG` - initial gas saturation `[0, 1]`."""

PRESSURE = GridArrayKeyword("PRESSURE", dtype=np.float64, default_value=0.0)
"""`PRESSURE` - initial reservoir pressure (psi in FIELD, barsa in METRIC)."""

RS = GridArrayKeyword("RS", dtype=np.float64, default_value=0.0)
"""`RS` - initial solution gas-oil ratio (scf/stb in FIELD)."""

RV = GridArrayKeyword("RV", dtype=np.float64, default_value=0.0)
"""`RV` - initial vaporised oil-gas ratio (stb/scf in FIELD)."""
