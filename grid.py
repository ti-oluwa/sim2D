import pyvista as pv

from bores.grids.io.grdecl import load_grdecl
from bores.grids.utils import as_pyvista_grid

grid = load_grdecl("data/dome.grdecl", encoding="utf-8")
print(f"cells   : {grid.n_cells}")  # 3*3*2 - 1 = 17
print(f"faces   : {grid.n_faces}")
print(f"volumes : {grid.cell_volumes}")
print(f"bbox    : {grid.bounding_box}")
pv_grid = as_pyvista_grid(grid)
pl = pv.Plotter()
pl.add_mesh(pv_grid, show_edges=True)
# pl.set_scale(zscale=12)  # type:ignore
pl.show()
