# Grids

## Overview

In BORES, the reservoir is divided into a structured Cartesian grid of rectangular cells. Each cell stores properties like porosity, permeability, pressure, and saturation at its center. Fluxes between cells are computed at the shared faces between adjacent cells. The grid is the foundation of every simulation - before you can run anything, you need to build property grids that describe your reservoir.

BORES provides several utility functions for constructing grids. You can create uniform grids where every cell has the same value, layered grids where properties vary by geological layer, depth grids that track subsurface elevation, and saturation grids initialized from fluid contact depths. All grid functions return standard NumPy arrays, so you can also construct grids manually using any NumPy operation.

The grid shape is always a tuple `(nx, ny, nz)` representing the number of cells in the x, y, and z directions. The z-axis points downward (depth increases with k-index), k=0 is the shallowest layer, and cell (0, 0, 0) is the top-left corner when viewed from above.

---

## Uniform Grids

The simplest grid type fills every cell with the same value. Use `bores.build_uniform_grid()` for constant-property reservoirs or as a starting point that you modify later.

```python
import bores

grid_shape = (20, 20, 5)

# Every cell gets the same value
porosity = bores.build_uniform_grid(grid_shape, value=0.22)
pressure = bores.build_uniform_grid(grid_shape, value=3500.0)  # psi
thickness = bores.build_uniform_grid(grid_shape, value=15.0)   # ft per layer
```

The function returns a NumPy array of the specified shape, filled with the given value, using the currently active precision (float32 by default). You can also use the shorter alias `bores.uniform_grid()`.

---

## Layered Grids

Real reservoirs have properties that vary from layer to layer due to geological deposition. Use `bores.build_layered_grid()` to assign different values to each layer along a chosen axis.

```python
import bores

grid_shape = (20, 20, 5)

# Porosity decreasing with depth (5 layers along z)
porosity = bores.build_layered_grid(
    grid_shape,
    layer_values=[0.25, 0.22, 0.20, 0.18, 0.15],
    orientation="z",
)

# Permeability varying along x-axis (20 layers along x)
perm_x = bores.build_layered_grid(
    grid_shape,
    layer_values=[100 + i * 5 for i in range(20)],
    orientation="x",
)
```

The `orientation` parameter selects the layering axis: `"z"` for vertical layering (most common), `"x"` for east-west variation, or `"y"` for north-south variation. The number of values in `layer_values` must match the number of cells along the chosen axis. You can also use the alias `bores.layered_grid()`.

!!! tip "Combining Uniform and Layered"

    You can build a base grid with `build_uniform_grid()` and then overwrite specific regions using NumPy slicing:

    ```python
    perm = bores.build_uniform_grid(grid_shape, value=100.0)
    perm[:, :, 3] = 10.0   # Low-permeability barrier in layer 3
    perm[:, :, 4] = 5.0    # Even tighter at the bottom
    ```

---

## Depth Grids

A depth grid computes the true vertical depth of each cell center based on the thickness grid. BORES measures depth downward from the top of the reservoir.

```python
import bores

grid_shape = (20, 20, 5)
thickness = bores.build_uniform_grid(grid_shape, value=15.0)

# Depth from top of reservoir (cell-center depths)
depth = bores.build_depth_grid(thickness)

# For the top layer: depth = 15/2 = 7.5 ft
# For layer 2: depth = 7.5 + 15/2 + 15/2 = 22.5 ft
# And so on...
```

If you need absolute depths (referenced to a datum like sea level), add the top depth:

```python
# Use a top depth datum of 5000ft
depth = bores.build_depth_grid(thickness, datum=5000)

# Or;
top_depth = 5000.0  # ft subsea
absolute_depth = depth + top_depth
```

For elevation measured upward from the base, use `bores.build_elevation_grid()` instead. It also accepts an optional `datum` parameter:

```python
elevation = bores.build_elevation_grid(thickness, datum=5000.0)
```

!!! note "Depth Convention"

    BORES uses depth-positive convention: the z-axis points downward, k=0 is the shallowest layer, and depth increases with k-index. This matches the standard petroleum engineering convention.

---

## Structural Dip

Real reservoirs are rarely flat. Use `bores.apply_structural_dip()` to tilt an elevation or depth grid, simulating structural dip caused by tectonic forces.

```python
import bores
import numpy as np

grid_shape = (20, 20, 5)
thickness = bores.build_uniform_grid(grid_shape, value=15.0)
elevation = bores.build_elevation_grid(thickness)

# Apply a 5-degree dip toward the east (azimuth = 90 degrees)
dipped = bores.apply_structural_dip(
    elevation_grid=elevation,
    dip_angle=5.0,           # degrees from horizontal
    dip_azimuth=90.0,        # direction of dip (degrees from north)
    cell_dimensions=(100.0, 100.0),  # (dx, dy) in feet
)
```

The dip angle is measured from horizontal (0 = flat, 90 = vertical). The azimuth specifies the compass direction that the formation dips toward: 0 = north, 90 = east, 180 = south, 270 = west.

Structural dip affects gravity-driven flow. In a dipping reservoir, gas migrates updip while water moves downdip. This is particularly important for gas injection simulations where gravity override interacts with the structural geometry.

!!! warning "Dip and Grid Resolution"

    Large dip angles with coarse grids can cause cells to have unrealistic geometry. Keep dip angles moderate (under 15-20 degrees) for standard Cartesian grids, or use finer grid resolution in the dip direction.

---

## Saturation Grids from Fluid Contacts

The `bores.build_saturation_grids()` function computes physically consistent initial saturations from gas-oil contact (GOC) and oil-water contact (OWC) depths. This is the recommended way to initialize saturations because it ensures the constraint $S_o + S_w + S_g = 1.0$ is satisfied everywhere and uses the correct residual saturations in each zone.

```python
import bores

grid_shape = (20, 20, 5)
thickness = bores.build_uniform_grid(grid_shape, value=15.0)
depth = bores.build_depth_grid(thickness, datum=5000.0) # Absolute depth from datum

# Residual saturation grids
Swc  = bores.build_uniform_grid(grid_shape, value=0.25)
Sorw = bores.build_uniform_grid(grid_shape, value=0.30)
Sorg = bores.build_uniform_grid(grid_shape, value=0.15)
Sgr  = bores.build_uniform_grid(grid_shape, value=0.05)
porosity = bores.build_uniform_grid(grid_shape, value=0.22)

# Sharp contacts (no transition zones)
Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4950.0,       # GOC depth (ft)
    oil_water_contact=5060.0,     # OWC depth (ft)
    connate_water_saturation_grid=Swc,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    porosity_grid=porosity,
)
```

The function divides the reservoir into three zones based on depth:

- **Gas cap** (above GOC): $S_g = 1 - S_{or,g} - S_{wc}$, $S_o = S_{or,g}$, $S_w = S_{wc}$
- **Oil zone** (between GOC and OWC): $S_o = 1 - S_{wc} - S_{gr}$, $S_w = S_{wc}$, $S_g = S_{gr}$
- **Water zone** (below OWC): $S_w = 1 - S_{or,w}$, $S_o = S_{or,w}$, $S_g = 0$

Notice that the function uses different residual oil saturations depending on the displacing fluid: $S_{or,g}$ in the gas cap and $S_{or,w}$ in the water zone.

### Transition Zones

For more realistic initialization, enable smooth saturation transitions at the contacts:

```python
Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4950.0,
    oil_water_contact=5060.0,
    connate_water_saturation_grid=Swc,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    porosity_grid=porosity,
    use_transition_zones=True,
    gas_oil_transition_thickness=10.0,   # ft
    oil_water_transition_thickness=15.0, # ft
    transition_curvature_exponent=2.0,   # Power-law shape
)
```

The `transition_curvature_exponent` controls the shape of the saturation profile in the transition zone. Values less than 1 give abrupt transitions, 1 gives linear interpolation, and values greater than 1 give smoother S-shaped curves that better approximate capillary pressure effects.

!!! info "When to Use Transition Zones"

    Transition zones are most important when your grid is fine enough to resolve the capillary fringe (cells smaller than the transition thickness). For coarse field-scale grids, sharp contacts are usually sufficient and avoid potential numerical artifacts from partially saturated cells.

---

## Seeding Phase Saturation

When an injection well targets a phase (gas or water) that has zero saturation at initial conditions, the injector cannot flow because its productivity index is proportional to phase mobility. With zero saturation, relative permeability is zero, PI is zero, and the injector contributes nothing regardless of BHP. This deadlock persists indefinitely.

`bores.seed_phase_saturation()` breaks this by setting a small nonzero saturation at the injector perforations, just above the relative permeability threshold. Oil saturation is reduced by the same amount to preserve $S_o + S_w + S_g = 1$.

```python
import bores

# After building initial saturation grids
Sw, So, Sg = bores.build_saturation_grids(...)

# Seed gas saturation at gas injector cells
Sw, So, Sg = bores.seed_phase_saturation(
    oil_saturation_grid=So,
    cells=[(9, 9, 0), (9, 9, 1), (9, 9, 2)],  # Injector perforations
    phase="gas",
    gas_saturation_grid=Sg,
    water_saturation_grid=Sw,
    relperm_table=relperm_model,  # Auto-detects minimum kr > 0
)
```

You can either provide a `relperm_table` for automatic seed detection (recommended), or specify an explicit `seed_saturation` value:

```python
# Explicit seed value
Sw, So, Sg = bores.seed_phase_saturation(
    oil_saturation_grid=So,
    cells=injector_cells,
    phase="water",
    water_saturation_grid=Sw,
    seed_saturation=0.05,  # 5% water saturation at injector cells
)
```

Use `inplace=True` to modify grids in place instead of creating copies.

---

## Grid Shape Conventions

BORES supports 1D, 2D, and 3D grids:

| Dimensionality | Grid Shape | Example Use Case |
|---|---|---|
| 1D | `(100, 1, 1)` | Buckley-Leverett displacement, core floods |
| 2D areal | `(50, 50, 1)` | Pattern studies, sweep efficiency |
| 2D cross-section | `(100, 1, 10)` | Gravity effects, layered reservoirs |
| 3D | `(30, 30, 10)` | Full-field simulation |

All property grids must have the same shape as the model's `grid_shape`. BORES validates this when you call `bores.reservoir_model()`.

---

## Visualizing Grids

You can visualize any property grid in 3D before running a simulation using the `bores.plotly3d.DataVisualizer` class. This is one of the most valuable debugging tools available to you, because it lets you catch setup errors before spending time on a simulation that will fail or produce nonsensical results.

The `make_plot()` method accepts either a `BlackOilModel` (showing named properties like "porosity" or "pressure"), a `ModelState` (showing simulation results), or a raw NumPy array. When you pass a raw array, the visualizer renders it as a 3D volume with the grid geometry you provide. When you pass a model or state, you select the property by name.

```python
import bores

# Build a model first (see Rock Properties and Fluid Properties for full example)
model = bores.reservoir_model(...)

# Visualize a named property from the model
viz = bores.plotly3d.DataVisualizer()
fig = viz.make_plot(
    source=model,
    property="porosity",
    plot_type="volume",
    title="Porosity Distribution",
    opacity=0.7,
)
fig.show()
# Output: [PLACEHOLDER: Insert porosity_3d_volume.png]
```

You can also visualize a raw grid array before you even build the model. This is useful for checking your porosity, permeability, or saturation grids during setup:

```python
import bores

grid_shape = (20, 20, 5)
thickness = bores.build_uniform_grid(grid_shape, value=15.0)
porosity = bores.build_layered_grid(grid_shape, [0.25, 0.22, 0.18, 0.20, 0.15], "z")

viz = bores.plotly3d.DataVisualizer()
fig = viz.make_plot(
    source=porosity,
    plot_type="volume",
    title="Porosity Grid Check",
)
fig.show()
# Output: [PLACEHOLDER: Insert porosity_grid_check.png]
```

The 3D visualizer supports several plot types including `"volume"` (default, shows all cells as colored blocks), `"isosurface"` (shows surfaces of constant property value), and `"slice"` (shows a cross-section through the grid). For a complete guide to the visualization system, see the [Visualization](../visualization/index.md) section.

!!! danger "Always Visualize Before Running"

    Before running any simulation, visualize at least your pressure, porosity, and permeability grids. Common mistakes include transposing dimensions, setting zero thickness in some cells, or placing wells outside the grid bounds. A few minutes of visual inspection can save hours of debugging cryptic simulation failures.
