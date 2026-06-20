import numpy as np

from bores.grids.io.grdecl import load_grdecl
from bores.grids.utils import as_pyvista_grid

grid = load_grdecl("Norne.GRDECL", encoding="utf-8")
print(f"cells   : {grid.n_cells}")  # 3*3*2 - 1 = 17
print(f"faces   : {grid.n_faces}")
print(f"volumes : {grid.cell_volumes}")
print(f"bbox    : {grid.bounding_box}")
pv_grid = as_pyvista_grid(grid)
pv_grid.plot(scalars="cell_volume", show_edges=True)
