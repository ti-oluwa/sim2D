import pyvista as pv

from bores.blackoil import BlackOilFluid, PVTRegions, RockFluidRegions
from bores.deck import DeckFile
from bores.grids import Grid
from bores.grids.utils import as_pyvista_grid
from bores.reservoir import Regions, Reservoir, Rock, Temperature

df = DeckFile("data/SPE1CASE1.DATA", encoding="utf-8")

# The reservoir
grid = Grid.from_deck(df)
regions = Regions.from_deck(df, n_cells=grid.n_cells)
rock = Rock.from_deck(df, grid=grid, rock_regions=regions.rock_regions)
reservoir = Reservoir(grid=grid, rock=rock, regions=regions)

# The fluid
temperature = Temperature(200)
pvt = PVTRegions.from_deck(df, temperature=temperature)
rock_fluid = RockFluidRegions.from_deck(df)
black_oil = BlackOilFluid(pvt=pvt, rock_fluid=rock_fluid)

print(pvt.region(1).tables.gas.viscosity_dp(3200, 200))

# # Plot the grid
# print(f"cells   : {grid.n_cells}")
# print(f"faces   : {grid.n_faces}")
# print(f"volumes : {grid.cell_volumes}")
# print(f"bbox    : {grid.bounding_box}")

# pv_grid = as_pyvista_grid(grid)
# pl = pv.Plotter()
# pl.add_mesh(pv_grid, scalars="cell_volume", show_edges=True)
# pl.set_scale(zscale=15, xscale=2, yscale=2)  # type:ignore
# pl.show()
