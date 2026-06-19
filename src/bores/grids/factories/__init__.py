"""
Grid factory classes for constructing `bores.grids.base.Grid` objects
from various source representations.

**Face winding convention**:

All factories produce faces whose vertices are ordered **counter-clockwise when
viewed from the owner cell** (``face_cell_indices[:, 0]``).  Under this
convention, the Newell normal produced by `bores.grid.grid._compute_face_geometry`
points **from owner toward neighbour** (i.e. outward for the owner cell).
Boundary faces carry ``neighbour_index == -1``.

**Coordinate system**:

The z-axis is positive **downward** (reservoir depth convention), matching
`bores.grids.base.Grid`.
"""

from .base import *  # noqa.
from .cartesian import *  # noqa
from .corner_point import *  # noqa
from .polyhedral import *  # noqa
from .voronoi import *  # noqa
