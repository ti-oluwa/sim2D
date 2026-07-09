import pyvista as pv

from bores.blackoil import BlackOilFluid, PVTRegions, RockFluidRegions
from bores.deck import DeckFile
from bores.grids import Grid
from bores.grids.utils import as_pyvista_grid
from bores.initialization import EquilibriumRegions, initialize_reservoir_state
from bores.reservoir import Regions, Reservoir, Rock, Temperature
from bores.typing import UnitSystem

df = DeckFile("data/SPE1CASE1.DATA", encoding="utf-8")
print(df.keywords)

# The reservoir
grid = Grid.from_deck(df)
regions = Regions.from_deck(df, n_cells=grid.n_cells, use_default=True)
rock = Rock.from_deck(df, grid=grid, rock_regions=regions.rock_regions)
reservoir = Reservoir(grid=grid, rock=rock, regions=regions)

# The fluid
temperature = Temperature(200)
pvt = PVTRegions.from_deck(df, temperature=temperature)
rock_fluid = RockFluidRegions.from_deck(df, mixing_rule="eclipse_rule")
black_oil = BlackOilFluid(pvt=pvt, rock_fluid=rock_fluid)
table = pvt.region(1).tables.gas
assert table is not None, "`table` should not be None"
print(table.density([4700, 200, 3456, 10000, 4000], 200))

# The initial state
equilibrium = EquilibriumRegions.from_deck(df)
initial_state = initialize_reservoir_state(
    reservoir=reservoir,
    pvt=pvt,
    equilibrium=equilibrium,
    rock_fluid=rock_fluid,
    temperature=temperature,
)
print(initial_state.oil_mass.sum())

# # Plot the grid
# print(f"cells   : {grid.n_cells}")
# print(f"faces   : {grid.n_faces}")
# print(f"bbox    : {grid.bounding_box}")

# pv_grid = as_pyvista_grid(grid, cell_data={"pressure": initial_state.pressure})
# pl = pv.Plotter()
# pl.add_mesh(pv_grid, scalars="pressure", show_edges=True)
# pl.set_scale(zscale=15, xscale=2, yscale=2)  # type:ignore
# pl.show()
