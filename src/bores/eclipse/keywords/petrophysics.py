"""
Petrophysical and grid-property keyword implementations.

Every keyword in this module is a `bores.eclipse.keywords.array.GridArrayKeyword`
subclass (or a direct instance of one). They differ only in name, dtype, and
`default_value`; all per-cell array mechanics (`N*value` repeat expansion,
`BOX`/operator timeline, `COPY`, caching) are inherited for free.
"""

import numpy as np

from bores.eclipse.keywords.base import GridArrayKeyword

__all__ = [
    # Geometry
    "Tops",
    "Dx",
    "Dy",
    "Dz",
    "ActNum",
    # Multipliers
    "MultX",
    "MultY",
    "MultZ",
    "MultXMinus",
    "MultYMinus",
    "MultZMinus",
    # Petrophysics
    "Poro",
    "PermX",
    "PermY",
    "PermZ",
    "PorV",
    "Ntg",
    # Fluid state
    "Sw",
    "So",
    "Sg",
    "Pressure",
    "Rs",
    "Rv",
    # Region
    "SatNum",
    "PVTNum",
    "EQLNum",
    "FIPNum",
]


# Geometry arrays


class Tops(GridArrayKeyword):
    """
    `TOPS` - depth to the top face of each cell in the first layer
    (one value per column `nx * ny`, or all `nx * ny * nz` cells for
    non-uniform decks).

    In practice Eclipse accepts either `nx * ny` values (applying the same
    top to every layer) or `nx * ny * nz` values.  The array length
    reported by :meth:`.GridArrayKeyword.parse` matches `n_cells`;
    callers should slice `[:nx*ny]` to get the top-layer tops.
    """

    def __init__(self) -> None:
        super().__init__("TOPS", dtype=np.float64, default_value=0.0)


class Dx(GridArrayKeyword):
    """`DX` - cell size in the x direction (one value per cell)."""

    def __init__(self) -> None:
        super().__init__("DX", dtype=np.float64, default_value=0.0)


class Dy(GridArrayKeyword):
    """`DY` - cell size in the y direction (one value per cell)."""

    def __init__(self) -> None:
        super().__init__("DY", dtype=np.float64, default_value=0.0)


class Dz(GridArrayKeyword):
    """`DZ` - cell size in the z direction (one value per cell)."""

    def __init__(self) -> None:
        super().__init__("DZ", dtype=np.float64, default_value=0.0)


class ActNum(GridArrayKeyword):
    """
    `ACTNUM` - active-cell mask.

    Values are `1` (active) or `0` (inactive).  Stored as int32.
    Default is `1` (all cells active) when the keyword is absent.

    Note:
        A missing `ACTNUM` keyword means all cells are active in Eclipse,
        so :meth:`.GridArrayKeyword.parse` returns `None` (keyword
        absent) rather than an all-ones array.  Callers should treat
        `None` as "all active".
    """

    def __init__(self) -> None:
        super().__init__("ACTNUM", dtype=np.int32, default_value=1)


# Transmissibility multipliers


class MultX(GridArrayKeyword):
    """
    `MULTX` - transmissibility multiplier for the positive-x face
    of each cell.
    """

    def __init__(self) -> None:
        super().__init__("MULTX", is_multiplier=True)


class MultY(GridArrayKeyword):
    """
    `MULTY` - transmissibility multiplier for the positive-y face
    of each cell.
    """

    def __init__(self) -> None:
        super().__init__("MULTY", is_multiplier=True)


class MultZ(GridArrayKeyword):
    """
    `MULTZ` - transmissibility multiplier for the positive-z face
    of each cell.
    """

    def __init__(self) -> None:
        super().__init__("MULTZ", is_multiplier=True)


class MultXMinus(GridArrayKeyword):
    """
    `MULTX-` - transmissibility multiplier for the negative-x face
    of each cell.
    """

    def __init__(self) -> None:
        super().__init__("MULTX-", is_multiplier=True)


class MultYMinus(GridArrayKeyword):
    """
    `MULTY-` - transmissibility multiplier for the negative-y face
    of each cell.
    """

    def __init__(self) -> None:
        super().__init__("MULTY-", is_multiplier=True)


class MultZMinus(GridArrayKeyword):
    """
    `MULTZ-` - transmissibility multiplier for the negative-z face
    of each cell.
    """

    def __init__(self) -> None:
        super().__init__("MULTZ-", is_multiplier=True)


# Petrophysical properties


class Poro(GridArrayKeyword):
    """
    `PORO` - porosity fraction `[0, 1]`.

    A missing `PORO` keyword returns `None`; the simulator should treat
    that as zero porosity (dead rock).
    """

    def __init__(self) -> None:
        super().__init__("PORO", dtype=np.float64, default_value=0.0)


class PermX(GridArrayKeyword):
    """`PERMX` - permeability in the x direction (mD)."""

    def __init__(self) -> None:
        super().__init__("PERMX", dtype=np.float64, default_value=0.0)


class PermY(GridArrayKeyword):
    """`PERMY` - permeability in the y direction (mD)."""

    def __init__(self) -> None:
        super().__init__("PERMY", dtype=np.float64, default_value=0.0)


class PermZ(GridArrayKeyword):
    """`PERMZ` - permeability in the z direction (mD)."""

    def __init__(self) -> None:
        super().__init__("PERMZ", dtype=np.float64, default_value=0.0)


class PorV(GridArrayKeyword):
    """
    `PORV` - pore volume per cell (bbl in FIELD, m³ in METRIC).

    When present, the simulator should use this directly rather than
    computing pore volume from geometry and porosity.
    """

    def __init__(self) -> None:
        super().__init__("PORV", dtype=np.float64, default_value=0.0)


class Ntg(GridArrayKeyword):
    """
    `NTG` - net-to-gross ratio `[0, 1]`.

    Effective pore volume: `PORV = NTG * PORO * bulk_volume`.
    Default is `1.0` (100 % net) when absent.
    """

    def __init__(self) -> None:
        super().__init__("NTG", dtype=np.float64, default_value=1.0)


# Fluid state arrays (initial conditions)


class Sw(GridArrayKeyword):
    """`SWAT` / `SW` - initial water saturation `[0, 1]`."""

    def __init__(self) -> None:
        super().__init__("SWAT", dtype=np.float64, default_value=0.0)


class So(GridArrayKeyword):
    """`SOIL` / `SO` - initial oil saturation `[0, 1]`."""

    def __init__(self) -> None:
        super().__init__("SOIL", dtype=np.float64, default_value=0.0)


class Sg(GridArrayKeyword):
    """`SGAS` / `SG` - initial gas saturation `[0, 1]`."""

    def __init__(self) -> None:
        super().__init__("SGAS", dtype=np.float64, default_value=0.0)


class Pressure(GridArrayKeyword):
    """
    `PRESSURE` - initial reservoir pressure (psi in FIELD, bara in METRIC).
    """

    def __init__(self) -> None:
        super().__init__("PRESSURE", dtype=np.float64, default_value=0.0)


class Rs(GridArrayKeyword):
    """`RS` - initial solution gas-oil ratio (scf/stb in FIELD)."""

    def __init__(self) -> None:
        super().__init__("RS", dtype=np.float64, default_value=0.0)


class Rv(GridArrayKeyword):
    """`RV` - initial vaporised oil-gas ratio (stb/scf in FIELD)."""

    def __init__(self) -> None:
        super().__init__("RV", dtype=np.float64, default_value=0.0)


# Region arrays (integer)


class SatNum(GridArrayKeyword):
    """`SATNUM` - saturation function region number (1-based int)."""

    def __init__(self) -> None:
        super().__init__("SATNUM", dtype=np.int32, default_value=1)


class PVTNum(GridArrayKeyword):
    """`PVTNUM` - PVT region number (1-based int)."""

    def __init__(self) -> None:
        super().__init__("PVTNUM", dtype=np.int32, default_value=1)


class EQLNum(GridArrayKeyword):
    """`EQLNUM` - equilibration region number (1-based int)."""

    def __init__(self) -> None:
        super().__init__("EQLNUM", dtype=np.int32, default_value=1)


class FIPNum(GridArrayKeyword):
    """`FIPNUM` - fluid-in-place region number (1-based int)."""

    def __init__(self) -> None:
        super().__init__("FIPNUM", dtype=np.int32, default_value=1)
