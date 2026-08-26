# Grid Systems

## Why Divide the Reservoir into Cells?

A real petroleum reservoir is a continuous body of porous rock extending over thousands of acres and hundreds of feet of thickness. The fluid properties, rock properties, and flow conditions vary continuously across this volume. But continuous systems cannot be solved analytically for realistic geometries - you need to discretize them. This means dividing the reservoir into a finite number of small blocks (cells), assigning properties to each block, and solving the governing equations on this discrete representation.

Each cell in the grid acts as a tiny control volume. Within each cell, properties like porosity, permeability, pressure, and saturation are assumed to be uniform - they are represented by a single value at the cell center. Flow between adjacent cells is computed from the pressure difference between their centers, using Darcy's law and the properties at the shared interface. The finer you make the grid (more cells, smaller cells), the better you approximate the continuous reality - but the more equations you have to solve and the longer the simulation takes.

BORES uses a structured Cartesian grid, which is the simplest and most common grid type in reservoir simulation. "Structured" means cells are arranged in a regular pattern indexed by integers $(i, j, k)$. "Cartesian" means the cells are rectangular blocks aligned with the coordinate axes. This makes the data structures simple (3D NumPy arrays), the neighbor relationships trivial (cell $(i,j,k)$ neighbors $(i \pm 1, j, k)$, $(i, j \pm 1, k)$, $(i, j, k \pm 1)$), and the finite difference stencils straightforward to implement.

The trade-off is geometric flexibility. Structured Cartesian grids cannot conform to curved reservoir boundaries, faults at arbitrary angles, or complex geological features without staircase approximations. For most practical simulations - especially educational, research, and screening studies - this limitation is acceptable. When you need complex geometry, you can use BORES in combination with external grid generation tools, or rely on the fracture and fault features that modify transmissibility at cell interfaces.

## Cartesian Grid Convention

BORES uses a specific coordinate convention that you should internalize before building models, because it affects how you specify well locations, interpret results, and visualize output.

The grid is indexed by three integers: $i$ (x-direction, columns), $j$ (y-direction, rows), and $k$ (z-direction, layers/depth). These correspond to the first, second, and third axes of a 3D NumPy array, respectively.

```
Grid Coordinate System:

         j (y) increases
            ↑
            |
            |  k (z) increases downward (into the page/screen)
            |  /
            | /
            |/________-> i (x) increases
          (0,0,0)
```

Several important conventions to note:

- **Cell $(0, 0, 0)$ is the top-left-shallow corner** - the cell at the minimum x, minimum y, and shallowest depth.
- **The z-axis points downward.** $k = 0$ is the shallowest (topmost) layer, and $k$ increases with depth. This is the standard depth-positive convention used throughout the petroleum industry.
- **Indices are zero-based**, following Python and NumPy conventions. The first cell is $(0, 0, 0)$, not $(1, 1, 1)$.

!!! note "Depth-Positive Convention"
    The z-axis in BORES increases downward, which means that a depth grid has larger values for deeper cells. This matches the petroleum engineering convention where depth is measured from the surface downward. If you are coming from a physics or mathematics background where z typically points upward, keep this in mind when interpreting elevation and depth grids.

When you access a property grid like `pressure_grid[i, j, k]`, you are reading the pressure at cell $(i, j, k)$. When you specify a well location as `(5, 3, 0)`, you are placing it in the cell at $i=5$, $j=3$, $k=0$ (the shallowest layer).

## Grid Shape

The shape of a BORES grid is always expressed as a tuple of three integers: `(nx, ny, nz)`, where `nx` is the number of cells in the x-direction, `ny` in the y-direction, and `nz` in the z-direction (depth). Even for lower-dimensional models, you still use a 3D tuple - you simply set the unused dimensions to 1.

Here are common configurations:

| Model Type | Grid Shape | Total Cells | Description |
|---|---|---|---|
| 1D (horizontal) | `(100, 1, 1)` | 100 | One row of 100 cells along x |
| 1D (vertical) | `(1, 1, 50)` | 50 | One column of 50 cells along z |
| 2D (areal) | `(50, 50, 1)` | 2,500 | A single-layer 50x50 areal model |
| 2D (cross-section) | `(100, 1, 20)` | 2,000 | A vertical cross-section along x |
| 3D (typical) | `(30, 30, 10)` | 9,000 | Standard 3D model with 10 layers |
| 3D (fine) | `(100, 100, 50)` | 500,000 | Fine-grid 3D model |

The cell dimensions (the physical size of each cell) are specified separately via the `cell_dimension` parameter, which is a tuple of `(dx, dy)` in feet. Cell thickness in the z-direction is specified per-cell using the `thickness_grid`, allowing layers of varying thickness.

!!! warning "Grid Size and Performance"
    Simulation time scales roughly with the number of cells. A 100x100x50 grid (500,000 cells) will take significantly longer than a 20x20x5 grid (2,000 cells). When developing and testing your model, start with a coarse grid to verify that the setup is correct, then refine once you are confident in the configuration. You can use `bores.coarsen_grid()` to downsample an existing fine grid for testing.

## Cell Properties

In a finite difference simulator like BORES, all intensive properties are stored at cell centers. Each cell carries its own values for:

- **Pressure** (psi) - the oil-phase pressure at the cell center
- **Saturations** - oil, water, and gas saturations (fractions that sum to 1.0)
- **Porosity** (fraction) - the fraction of the cell volume that is pore space
- **Permeability** (mD) - how easily fluid flows through the rock, which can be different in x, y, and z directions (anisotropic)
- **Temperature** (degrees F) - the reservoir temperature at the cell
- **Fluid properties** - viscosity, density, compressibility, formation volume factor, etc.

All of these are stored as 3D NumPy arrays with shape `(nx, ny, nz)`. For example, if your grid shape is `(30, 30, 10)`, then `pressure_grid` is a NumPy array of shape `(30, 30, 10)` containing 9,000 pressure values.

**Transmissibility** - the quantity that controls flow between two adjacent cells - is a face property, not a cell-center property. BORES computes transmissibility at each cell interface from the permeabilities and geometries of the two cells sharing that face, using a harmonic mean to properly handle permeability contrasts. This calculation happens internally during the simulation and is not something you typically need to set up manually.

!!! info "Harmonic Mean for Transmissibility"
    When computing flow between two cells with different permeabilities, BORES uses the harmonic mean rather than the arithmetic mean. The harmonic mean ensures that a low-permeability cell acts as a flow barrier, which is physically correct. If cell A has permeability 1000 mD and cell B has 1 mD, the effective interface permeability is about 2 mD (dominated by the low-perm cell), not 500.5 mD (the arithmetic average).

## Grid Construction in BORES

BORES provides several utility functions for building property grids. These are designed to be composed together - you build each property grid independently and pass them all to `bores.reservoir_model()`.

### Uniform Grids

The simplest grid type fills every cell with the same value. This is useful for homogeneous models and for initial conditions.

```python
import bores

grid_shape = (30, 30, 10)

# Uniform pressure of 3000 psi everywhere
pressure = bores.build_uniform_grid(grid_shape, value=3000.0)

# Uniform porosity of 20%
porosity = bores.build_uniform_grid(grid_shape, value=0.20)

# Uniform thickness of 15 ft per layer
thickness = bores.build_uniform_grid(grid_shape, value=15.0)
```

The `bores.build_uniform_grid()` function returns a NumPy array of the specified shape, filled with the given value and cast to the active precision (32-bit by default).

### Layered Grids

Real reservoirs often have properties that vary by layer - each geological formation has its own porosity, permeability, and thickness. `bores.build_layered_grid()` lets you specify a value for each layer along any axis.

```python
import bores

grid_shape = (30, 30, 5)

# Porosity varies by depth layer (z-direction):
# Layer 0 (shallowest): 0.22, Layer 1: 0.18, ..., Layer 4 (deepest): 0.12
porosity = bores.build_layered_grid(
    grid_shape=grid_shape,
    layer_values=[0.22, 0.18, 0.15, 0.14, 0.12],
    orientation=bores.Orientation.Z,
)

# Permeability varies in the x-direction (e.g., facies change):
# 30 values, one for each column along x
import numpy as np

perm_values = np.linspace(200.0, 50.0, 30)  # Decreasing from left to right
perm_x = bores.build_layered_grid(
    grid_shape=grid_shape,
    layer_values=perm_values,
    orientation=bores.Orientation.X,
)
```

The `orientation` parameter accepts `bores.Orientation.X`, `bores.Orientation.Y`, or `bores.Orientation.Z`, and the number of layer values must match the number of cells in that direction.

### Depth and Elevation Grids

BORES can compute depth or elevation grids from a thickness grid. These are needed for gravity calculations (hydrostatic pressure gradients) and structural dip.

```python
import bores

grid_shape = (30, 30, 10)
thickness = bores.build_uniform_grid(grid_shape, value=15.0)

# Depth grid: measured from top downward (depth-positive)
# Cell centers are placed based on cumulative thickness
depth = bores.build_depth_grid(thickness)
# depth[0, 0, 0] = 7.5 (center of first 15-ft layer)
# depth[0, 0, 1] = 22.5 (center of second layer)
# depth[0, 0, 9] = 142.5 (center of tenth layer)

# Elevation grid: measured from bottom upward
elevation = bores.build_elevation_grid(thickness)
```

The depth grid is used internally by the simulator to compute gravity terms in the pressure equation. The `build_depth_grid()` function places cell centers at the midpoint of each layer's cumulative thickness.

### Structural Dip

Real reservoirs are rarely perfectly flat. The `bores.apply_structural_dip()` function tilts an elevation or depth grid by a specified angle and azimuth, simulating a dipping reservoir structure.

```python
import bores

grid_shape = (30, 30, 10)
cell_dimension = (100.0, 100.0)
thickness = bores.build_uniform_grid(grid_shape, value=15.0)
depth = bores.build_depth_grid(thickness)

# Apply a 5-degree dip toward the east (azimuth = 90 degrees)
dipped_depth = bores.apply_structural_dip(
    elevation_grid=depth,
    cell_dimension=cell_dimension,
    elevation_direction="downward",
    dip_angle=5.0,
    dip_azimuth=90.0,
)
```

The dip azimuth follows the compass convention: 0 degrees is North, 90 degrees is East, 180 degrees is South, and 270 degrees is West. The surface tilts downward in the azimuth direction, meaning cells in that direction become deeper.

!!! tip "Start Without Dip"
    When building a new model, set `dip_angle=0.0` initially and add structural dip only after you have verified that the flat model works correctly. Gravity effects from dip can complicate debugging, especially for new users. You can also disable dip effects during simulation by setting `disable_structural_dip=True` in the `Config`.

### Grid Coarsening

When you have a fine-grid model and want a coarser version for faster testing, `bores.coarsen_grid()` aggregates blocks of cells:

```python
import bores
import numpy as np

# Start with a fine grid
fine_porosity = np.random.uniform(0.15, 0.25, size=(100, 100, 50))

# Coarsen by a factor of 2 in all directions: (100,100,50) -> (50,50,25)
coarse_porosity = bores.coarsen_grid(
    fine_porosity,
    batch_size=(2, 2, 2),
    method="mean",  # Use arithmetic average for porosity
)

# Coarsen permeability using direction-appropriate averaging
fine_kx = np.random.uniform(10.0, 500.0, size=(100, 100, 50))
fine_ky = np.random.uniform(10.0, 500.0, size=(100, 100, 50))
fine_kz = np.random.uniform(1.0, 50.0, size=(100, 100, 50))

coarse_kx, coarse_ky, coarse_kz = bores.coarsen_permeability_grids(
    fine_kx, fine_ky, fine_kz, batch_size=(2, 2, 2)
)
```

The `method` parameter on `coarsen_grid` supports `"mean"`, `"sum"`, `"max"`, `"min"`, and `"harmonic"`. Choose the aggregation method that is physically appropriate for each property: arithmetic mean for porosity, sum for pore volume-weighted quantities, and `coarsen_permeability_grids` for permeability (which applies harmonic averaging in the flow direction and arithmetic averaging perpendicular to it).

## Grid Visualization

BORES includes visualization tools for inspecting your grid. The `bores.visualization.plotly3d` module provides browser-based 3D volume rendering, and the optional `bores.visualization.pyvista3d` module provides GPU-accelerated cell-block rendering with interactive slice planes. Both let you rotate, zoom, and slice through your model to verify that properties are assigned correctly before running a simulation.

```python
import bores
from bores.visualization import plotly3d

# Assuming you have built a model
# Visualize the initial pressure distribution
viz = plotly3d.DataVisualizer()
plot = viz.make_plot(
    data=model.fluid_properties.pressure_grid,
    title="Initial Pressure (psi)",
    plot_type="volume",
)
plot.show()
```

For 2D maps of a single layer, use `bores.visualization.plotly2d`, and for 1D property profiles along a line, use `bores.visualization.plotly1d`. These tools are particularly useful during model construction to catch setup errors - a misplaced well, an inverted permeability layer, or a saturation that does not sum to one will often be immediately obvious in a visualization.

!!! example "Verifying Your Grid"
    Before running any simulation, visualize at least your pressure, porosity, and permeability grids. Common mistakes include:

    - Accidentally transposing grid dimensions (e.g., assigning a `(ny, nx, nz)` array to a `(nx, ny, nz)` model)
    - Setting thickness to zero in some cells (creates inactive cells with undefined behavior)
    - Placing wells outside the grid bounds (BORES would normally warn on this)

    Five minutes of visual inspection can save hours of debugging cryptic simulation failures.
