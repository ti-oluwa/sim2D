import pyvista as pv

from bores.grids.io.grdecl import load_grdecl
from bores.grids.utils import as_pyvista_grid

grid = load_grdecl("data/Norne.GRDECL", encoding="utf-8")
print(f"cells   : {grid.n_cells}")
print(f"faces   : {grid.n_faces}")
print(f"volumes : {grid.cell_volumes}")
print(f"bbox    : {grid.bounding_box}")
pv_grid = as_pyvista_grid(grid)
pl = pv.Plotter()
pl.add_mesh(pv_grid, scalars="cell_depth", show_edges=True)
pl.set_scale(zscale=2, xscale=2, yscale=2)  # type:ignore
pl.show()
