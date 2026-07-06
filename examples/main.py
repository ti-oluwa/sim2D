import pyvista as pv

from bores.blackoil import (
    BlackOilModel,
    PVTRegions,
    RockCompressibilityRegions,
    RockFluidRegions,
)
from bores.deck import DeckFile
from bores.grids import Grid
from bores.grids.utils import as_pyvista_grid
from bores.reservoir import Regions, ReservoirModel, Rock, Temperature

df = DeckFile("data/SPE1CASE1.DATA", encoding="utf-8")

# The Black Oil model
temperature = Temperature(200)
pvt_regions = PVTRegions.from_deck_file(df, temperature=temperature)
rock_fluid_regions = RockFluidRegions.from_deck_file(df)
compressibility_regions = RockCompressibilityRegions.from_deck_file(df)
black_oil_model = BlackOilModel(
    pvt_regions=pvt_regions,
    rock_fluid_regions=rock_fluid_regions,
    compressibility_regions=compressibility_regions,
)

# The Reservoir model
grid = Grid.from_deck_file(df)
regions = Regions.from_deck_file(df, n_cells=grid.n_cells)
# rock = Rock.from_deck_file(df, grid=grid, rock_region=regions.rock_region)
# reservoir = ReservoirModel(grid=grid, rock=rock, regions=regions)


# Plot the grid
print(f"cells   : {grid.n_cells}")
print(f"faces   : {grid.n_faces}")
print(f"volumes : {grid.cell_volumes}")
print(f"bbox    : {grid.bounding_box}")

pv_grid = as_pyvista_grid(grid)
pl = pv.Plotter()
pl.add_mesh(pv_grid, scalars="cell_volume", show_edges=True)
pl.set_scale(zscale=5, xscale=2, yscale=2)  # type:ignore
pl.show()
