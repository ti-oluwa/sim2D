import numpy as np

from bores.grids.factories.corner_point import make_corner_point_grid
from bores.grids.factories.voronoi import make_voronoi_grid
from bores.grids.utils import to_pyvista

# 3x3x2 grid (NX=3, NY=3, NZ=2)
NX, NY, NZ = 3, 3, 2

# Cell sizes
dx = 100.0  # metres per cell in x
dy = 100.0
dz = 10.0  # layer thickness

# ------------------------------------------------------------------
# COORD — shape (NY+1, NX+1, 6)
# Each pillar: [x_top, y_top, z_top, x_bot, y_bot, z_bot]
# Pillars are vertical here (x,y constant from top to bot)
# ------------------------------------------------------------------
coord = np.zeros((NY + 1, NX + 1, 6), dtype=np.float64)
for j in range(NY + 1):
    for i in range(NX + 1):
        x = i * dx
        y = j * dy
        coord[j, i] = [x, y, 0.0, x, y, 1000.0]  # vertical pillar

# ------------------------------------------------------------------
# ZCORN — shape (NZ*2, NY*2, NX*2)
# For each cell (i,j,k): zcorn[2k:2k+2, 2j:2j+2, 2i:2i+2]
# gives the 8 corner depths (top face then bottom face).
# ------------------------------------------------------------------
z_top = 2000.0  # depth of grid top (positive downward)
zcorn = np.zeros((NZ * 2, NY * 2, NX * 2), dtype=np.float64)
dip = 15.0
for k in range(NZ):
    for j in range(NY):
        for i in range(NX):
            offset = i * dip  # cells dip eastward
            zcorn[2 * k, 2 * j, 2 * i] = z_top + k * dz + offset
            zcorn[2 * k, 2 * j, 2 * i + 1] = z_top + k * dz + (i + 1) * dip
            zcorn[2 * k, 2 * j + 1, 2 * i] = z_top + k * dz + offset
            zcorn[2 * k, 2 * j + 1, 2 * i + 1] = z_top + k * dz + (i + 1) * dip
            zcorn[2 * k + 1, 2 * j, 2 * i] = z_top + (k + 1) * dz + offset
            zcorn[2 * k + 1, 2 * j, 2 * i + 1] = z_top + (k + 1) * dz + (i + 1) * dip
            zcorn[2 * k + 1, 2 * j + 1, 2 * i] = z_top + (k + 1) * dz + offset
            zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1] = (
                z_top + (k + 1) * dz + (i + 1) * dip
            )

# ------------------------------------------------------------------
# ACTNUM — shape (NZ, NY, NX) — optional, 1=active 0=inactive
# Knock out one cell to show it works
# ------------------------------------------------------------------
actnum = np.ones((NZ, NY, NX), dtype=np.int32)
actnum[0, 1, 1] = 0  # deactivate cell (i=1, j=1, k=0)

grid = make_corner_point_grid(coord=coord, zcorn=zcorn, actnum=actnum)
print(f"cells   : {grid.n_cells}")  # 3*3*2 - 1 = 17
print(f"faces   : {grid.n_faces}")
print(f"volumes : {grid.cell_volumes}")
print(f"bbox    : {grid.bounding_box}")
pv_grid = to_pyvista(grid)
pv_grid.plot(scalars="cell_volume", show_edges=True)
