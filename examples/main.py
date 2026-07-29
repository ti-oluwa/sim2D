import pyvista as pv
import rich

from bores.blackoil.fluid import BlackOil
from bores.blackoil.model import BlackOilModel
from bores.blackoil.pvt import PVTRegions
from bores.blackoil.satfunc import SatFuncRegions
from bores.deck import DeckFile
from bores.grids import Grid
from bores.grids.utils import as_pyvista_grid
from bores.initialization import initialize_reservoir_state
from bores.reservoir import Reservoir, ReservoirRegions, Temperature
from bores.reservoir.rock import Rock
from bores.reservoir.state import EquilibriumRegions
from bores.typing import UnitSystem
from bores.wells import WellSystem
from bores.wells.hydraulics.mechanistic import MechanisticWellbore

df = DeckFile("data/SPE1CASE1.DATA", encoding="utf-8")

# Load reservoir model
grid = Grid.from_deck(df)
regions = ReservoirRegions.from_deck(df, n_cells=grid.n_cells, use_default=True)
satfunc = SatFuncRegions.from_deck(df, mixing_rule="eclipse_rule")
rock = Rock.from_deck(
    df,
    grid=grid,
    rock_region=regions.rock_region,
    satfunc=satfunc,
    saturation_region=regions.saturation_region,
)
reservoir = Reservoir(grid=grid, rock=rock, regions=regions)

# Define fluid
temperature = Temperature(200)
pvt = PVTRegions.from_deck(df, temperature=temperature)
blackoil = BlackOil(pvt=pvt, satfunc=satfunc)
table = pvt.region(1).tables.gas
assert table is not None, "`table` should not be None"
print(table.viscosity([4700, 200, 3456, 10000, 4000], 400))

# Load initial state
equilibrium = EquilibriumRegions.from_deck(df)
initial_state = initialize_reservoir_state(
    reservoir=reservoir,
    pvt=pvt,
    deck_file=df,
    equilibrium=equilibrium,
    satfunc=satfunc,
    temperature=temperature,
)

# Load wells
wells = WellSystem.from_deck(
    df,
    grid=grid,
    default_wellbore=MechanisticWellbore(tubing_inner_diameter=0.5),
)
# rich.print(wells.dump())

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
