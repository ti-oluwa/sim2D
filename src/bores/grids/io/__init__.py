"""
Grid import / export functions. Each sub-module handles one file family.

IO code only converts between external representations and the
`bores.grids.base.Grid` data model. 

GRDECL / Eclipse text:

    from bores.grids.io.grdecl import load_grdecl, dump_grdecl

Eclipse binary / unified restart (.EGRID, .GRID):

    from bores.grids.io.eclipse import load_eclipse_grid

Gmsh (.msh):

    from bores.grids.io.gmsh import load_msh

VTK / VTU / meshio-readable formats:

    from bores.grids.io.vtk import load_vtk, load_vtu, dump_vtk, dump_vtu, load_mesh

PyVista conversion:

    from bores.grids.io.pyvista import to_pyvista
"""

from .gmsh import *  # noqa
from .grdecl import *  # noqa
from .meshio import *  # noqa
