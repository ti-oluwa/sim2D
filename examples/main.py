import pyvista as pv

from bores.blackoil.fluid import BlackOilFluid
from bores.blackoil.model import BlackOilModel
from bores.blackoil.pvt import PVTRegions
from bores.blackoil.saturation_functions import SaturationFunctionRegions
from bores.deck import DeckFile
from bores.grids import Grid
from bores.grids.utils import as_pyvista_grid
from bores.initialization import initialize_reservoir_state
from bores.reservoir import Regions, ReservoirModel, Temperature
from bores.reservoir.rock import Rock
from bores.reservoir.state import EquilibriumRegions
from bores.wells import MechanisticWellboreModel, WellModel

df = DeckFile("data/SPE1CASE1.DATA", encoding="utf-8")

# The reservoir
grid = Grid.from_deck(df)
regions = Regions.from_deck(df, n_cells=grid.n_cells, use_default=True)
satfunc = SaturationFunctionRegions.from_deck(df, mixing_rule="eclipse_rule")
rock = Rock.from_deck(
    df,
    grid=grid,
    rock_regions=regions.rock_regions,
    satfunc=satfunc,
    saturation_regions=regions.saturation_regions,
)
reservoir = ReservoirModel(grid=grid, rock=rock, regions=regions)

# The fluid
temperature = Temperature(200)
pvt = PVTRegions.from_deck(df, temperature=temperature)
blackoil = BlackOilFluid(pvt=pvt, satfunc=satfunc)
table = pvt.region(1).tables.gas
assert table is not None, "`table` should not be None"
print(table.viscosity([4700, 200, 3456, 10000, 4000], 400))

# The initial state
equilibrium = EquilibriumRegions.from_deck(df)
initial_state = initialize_reservoir_state(
    reservoir=reservoir,
    pvt=pvt,
    deck_file=df,
    equilibrium=equilibrium,
    satfunc=satfunc,
    temperature=temperature,
)

# Load Wells
wells = WellModel.from_deck(
    df,
    grid=grid,
    wellbore_model=MechanisticWellboreModel(),
)
d = wells.dump()
w = WellModel.load(d)
print(w)

# Construct the final model
model = BlackOilModel(reservoir=reservoir, fluid=blackoil, wells=wells)

# Plot the grid
print(f"cells   : {grid.n_cells}")
print(f"faces   : {grid.n_faces}")
print(f"bbox    : {grid.bounding_box}")

pv_grid = as_pyvista_grid(grid, cell_data={"pressure": initial_state.pressure})
pl = pv.Plotter()
pl.add_mesh(pv_grid, scalars="pressure", show_edges=True)
pl.set_scale(zscale=15, xscale=2, yscale=2)  # type:ignore
pl.show()


# TODO: Support region in well keywords
