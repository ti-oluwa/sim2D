# 2D Spatial Maps

## Overview

The 2D visualization module creates spatial maps of reservoir properties on two-dimensional grids. Heatmaps, contour maps, scatter plots, cross-section line plots, and 3D surface plots all take 2D numpy arrays and render them as interactive Plotly figures. These are the standard tools for examining property distributions across a reservoir slice: pressure maps, saturation fronts, permeability fields, and fluid contact movements.

The module lives in `bores.visualization.plotly2d` and centers on the `DataVisualizer` class. Unlike the 1D module (which works with time series), the 2D module works with grid data where each element represents a cell in the simulation grid. Every plot method returns a Plotly `Figure` object that you can display interactively with `.show()`, save as HTML with `.write_html()`, or export as a static image with `.write_image()`.

All 2D visualizations support property metadata through the `PropertyMeta` class. Metadata controls the colorbar label, display name in hover text, unit string, log-scale transformation, and default color scheme. You can pass metadata explicitly or let the visualizer create default metadata automatically. When you use the `PropertyRegistry` (covered in the [overview](index.md)), the metadata is pre-configured for common reservoir properties like pressure, saturation, permeability, and porosity.

The 2D visualizer expects raw 2D numpy arrays, not `ModelState` objects. You extract the property grid you want from your simulation state and pass it to the visualizer. This separation gives you full control over data preparation, including slicing 3D grids to get 2D cross-sections, applying unit conversions, or computing derived quantities before visualization.

---

## Data Input

The 2D visualizer works with 2D numpy arrays. Each array represents a spatial grid where rows correspond to the y-axis and columns correspond to the x-axis. The array shape `(ny, nx)` follows numpy's row-major convention: `data[i, j]` refers to row `i` (y-direction) and column `j` (x-direction).

```python
import numpy as np

# Extract a 2D slice from a 3D simulation state
# For a 3D grid with shape (nz, ny, nx), take a horizontal slice at layer k=2
pressure_slice = states[-1].model.fluid_properties.pressure_grid[:, :, 2]

# Or extract the full 2D grid from a 2D simulation
pressure_grid = states[-1].model.fluid_properties.pressure_grid
```

If you do not provide coordinate arrays, the visualizer uses integer indices (0, 1, 2, ...) for both axes. To map grid cells to physical coordinates, pass `x_coords` and `y_coords` arrays:

```python
# Physical coordinates in feet
dx = 100.0  # cell width
dy = 100.0  # cell height
nx, ny = 20, 15
x_coords = np.arange(nx) * dx + dx / 2  # cell centers
y_coords = np.arange(ny) * dy + dy / 2
```

---

## Property Metadata

The `PropertyMeta` class controls how a property is displayed. It sets the plot title, unit string, whether to apply a log-scale transformation, and the default color scheme. You can create metadata manually or retrieve it from the `PropertyRegistry`.

```python
from bores.visualization.base import PropertyMeta, ColorScheme

# Manual metadata
pressure_meta = PropertyMeta(
    name="pressure_grid",
    display_name="Pressure",
    unit="psi",
    log_scale=False,
    color_scheme=ColorScheme.VIRIDIS,
)

# For permeability (log-scale is often appropriate)
perm_meta = PropertyMeta(
    name="permeability_x",
    display_name="Permeability (x)",
    unit="mD",
    log_scale=True,
    color_scheme=ColorScheme.PLASMA,
)
```

When `log_scale` is `True`, the visualizer applies a base-10 logarithm to the data before mapping it to colors. Hover text still shows the original (un-logged) values. This is useful for permeability, which can span several orders of magnitude across a reservoir.

If you do not pass metadata, the visualizer creates a default `PropertyMeta` with `display_name="Data"`, no unit, no log scale, and viridis color scheme. For quick exploration this is fine, but for publication-quality figures you should always provide metadata with meaningful labels and units.

---

## Creating Plots

### DataVisualizer

The `DataVisualizer` class is the main entry point for 2D plotting. Create one with optional configuration:

```python
from bores.visualization.plotly2d import DataVisualizer, PlotConfig

# Default configuration
viz = DataVisualizer()

# Custom configuration
viz = DataVisualizer(
    config=PlotConfig(
        width=1000,
        height=800,
        color_scheme="plasma",
        contour_levels=25,
        show_colorbar=True,
        font_size=14,
    )
)
```

### `make_plot`

The `make_plot()` method creates a single 2D plot:

```python
fig = viz.make_plot(
    data=pressure_grid,
    plot_type="heatmap",
    metadata=pressure_meta,
    x_coords=x_coords,
    y_coords=y_coords,
    x_label="X (ft)",
    y_label="Y (ft)",
    title="Reservoir Pressure at t = 365 days",
)
fig.show()
```

The `plot_type` parameter accepts either a `PlotType` enum value or a string. Available types are `"heatmap"`, `"contour"`, `"contour_filled"`, `"scatter"`, `"line"`, and `"surface"`.

The method also accepts optional `width` and `height` parameters that override the config values for this specific plot. You can pass an existing `figure` to add the plot as an additional trace to an existing figure.

### `make_plots` (Subplots)

The `make_plots()` method creates a grid of subplots from multiple datasets:

```python
oil_sat = states[-1].model.fluid_properties.oil_saturation_grid
water_sat = states[-1].model.fluid_properties.water_saturation_grid
pressure = states[-1].model.fluid_properties.pressure_grid

fig = viz.make_plots(
    data_list=[pressure, oil_sat, water_sat],
    plot_types="heatmap",
    rows=1,
    cols=3,
    subplot_titles=["Pressure (psi)", "Oil Saturation", "Water Saturation"],
    shared_xaxes=True,
    shared_yaxes=True,
    x_coords=x_coords,
    y_coords=y_coords,
    x_label="X (ft)",
    y_label="Y (ft)",
)
fig.show()
```

You can mix plot types across subplots by passing a list of types:

```python
fig = viz.make_plots(
    data_list=[pressure, pressure],
    plot_types=["heatmap", "contour"],
    rows=1,
    cols=2,
    subplot_titles=["Pressure Heatmap", "Pressure Contours"],
)
```

You can control spacing between subplots with `vertical_spacing` and `horizontal_spacing` (both as fractions from 0.0 to 1.0). The `metadata_list` parameter accepts a sequence of `PropertyMeta` objects, one per subplot.

---

## Plot Types

### Heatmaps

Heatmaps are the default and most common 2D plot type. They display a color-coded grid where each cell is colored according to its value. The color mapping is continuous, with a colorbar showing the value-to-color correspondence.

```python
from bores.visualization.plotly2d import DataVisualizer
from bores.visualization.base import PropertyMeta, ColorScheme

viz = DataVisualizer()

pressure_meta = PropertyMeta(
    name="pressure_grid",
    display_name="Pressure",
    unit="psi",
    log_scale=False,
    color_scheme=ColorScheme.VIRIDIS,
)

fig = viz.make_plot(
    data=pressure_grid,
    plot_type="heatmap",
    metadata=pressure_meta,
    x_label="X (ft)",
    y_label="Y (ft)",
)
fig.show()
```

Heatmaps support optional `cmin` and `cmax` keyword arguments that fix the color range. This is useful when comparing multiple snapshots of the same property across different time steps, where you want a consistent color scale:

```python
# Fix color range for consistent comparison across timesteps
fig = viz.make_plot(
    data=pressure_grid,
    plot_type="heatmap",
    metadata=pressure_meta,
    cmin=1500.0,
    cmax=3000.0,
)
```

Hover text shows the x-coordinate, y-coordinate, property name, value, and unit for each cell.

### Contour Maps

Contour maps draw isolines connecting points of equal value. They are the standard way to visualize pressure fields, fluid contacts, and saturation fronts in reservoir engineering. The `"contour"` type draws line contours, while `"contour_filled"` fills the regions between contour lines with color.

```python
# Line contours
fig = viz.make_plot(
    data=pressure_grid,
    plot_type="contour",
    metadata=pressure_meta,
    x_label="X (ft)",
    y_label="Y (ft)",
    contour_levels=15,
)

# Filled contours
fig = viz.make_plot(
    data=pressure_grid,
    plot_type="contour_filled",
    metadata=pressure_meta,
    x_label="X (ft)",
    y_label="Y (ft)",
    contour_levels=20,
)
```

The `contour_levels` keyword argument controls how many contour lines to draw. It defaults to 20 if not specified (or to the value set in `PlotConfig`). Contour lines are automatically labeled with their values. You can also pass `cmin` and `cmax` to control the range of contour levels.

Filled contours are particularly useful for saturation maps where you want to see the flood front as a continuous color field rather than discrete isolines. Line contours work better for pressure maps where you want to see the pressure gradient direction (perpendicular to the isolines).

### Scatter Plots

Scatter plots display individual grid cells as markers, colored by their value. They are useful for sparse data or for highlighting cells that meet a threshold condition. The scatter renderer filters out cells below a threshold value, so only cells of interest appear on the plot.

```python
# Show only cells where oil saturation > 0.3
fig = viz.make_plot(
    data=oil_saturation_grid,
    plot_type="scatter",
    metadata=oil_sat_meta,
    x_label="X (ft)",
    y_label="Y (ft)",
    threshold=0.3,
    marker_size=10,
)
```

The `threshold` parameter (default 0.0) sets the minimum value for a cell to be included. The `marker_size` parameter controls the size of the scatter markers. Markers are colored according to the value, using the same color scale as heatmaps.

Scatter plots are useful for visualizing well locations overlaid on a property map, or for showing the spatial distribution of high-permeability channels in a heterogeneous reservoir.

### Cross-Section Line Plots

Line plots extract 1D cross-sections from a 2D grid and display them as line charts. This is useful for examining how a property varies along a specific row or column of the grid. You can plot horizontal cross-sections (constant y, varying x), vertical cross-sections (constant x, varying y), or both.

```python
# Horizontal cross-section at row index 5
fig = viz.make_plot(
    data=pressure_grid,
    plot_type="line",
    metadata=pressure_meta,
    x_coords=x_coords,
    y_coords=y_coords,
    x_label="Distance (ft)",
    y_label="Pressure (psi)",
    line_mode="horizontal",
    line_indices=[5],
)

# Both horizontal and vertical cross-sections
fig = viz.make_plot(
    data=pressure_grid,
    plot_type="line",
    metadata=pressure_meta,
    line_mode="both",
    line_indices=[5, 10],  # row 5 for horizontal, column 10 for vertical
)
```

The `line_mode` parameter accepts `"horizontal"`, `"vertical"`, or `"both"`. The `line_indices` parameter specifies which row/column indices to extract. When `line_mode` is `"both"`, the first index is used for the horizontal cross-section and the second for the vertical. If `line_indices` is not provided, the visualizer defaults to the middle row and column.

Cross-section line plots are valuable for comparing analytical solutions against simulation results along a 1D profile, or for examining pressure drawdown from a well.

### 3D Surface Plots

Surface plots render a 2D grid as a 3D surface where the z-height represents the property value. This gives an intuitive sense of spatial gradients and peaks that can be harder to see in flat 2D maps.

```python
fig = viz.make_plot(
    data=pressure_grid,
    plot_type="surface",
    metadata=pressure_meta,
    x_coords=x_coords,
    y_coords=y_coords,
    x_label="X (ft)",
    y_label="Y (ft)",
)
fig.show()
```

The surface is colored according to the z-values using the same color scheme as heatmaps. You can pass `cmin` and `cmax` to fix the color range. The resulting figure is fully interactive: you can rotate, zoom, and pan the 3D view.

Surface plots work best for smooth, continuous properties like pressure. For discontinuous properties like saturation near a flood front, heatmaps or filled contours are usually more informative.

---

## Configuration Reference

The `PlotConfig` class controls all visual aspects of 2D plots:

| Parameter | Default | Description |
| --- | --- | --- |
| `width` | 800 | Figure width in pixels |
| `height` | 600 | Figure height in pixels |
| `show_colorbar` | `True` | Whether to display color scale bar |
| `title` | `None` | Default title for plots |
| `color_scheme` | `"viridis"` | Default colorscale name |
| `opacity` | 0.8 | Default opacity for plot elements |
| `show_grid` | `True` | Whether to show grid lines |
| `grid_color` | `"lightgray"` | Color of grid lines |
| `axis_line_color` | `"black"` | Color of axis lines |
| `axis_line_width` | 1.0 | Width of axis lines |
| `font_family` | `"Arial, sans-serif"` | Font family |
| `font_size` | 12 | Default font size |
| `title_font_size` | 16 | Title font size |
| `axis_title_font_size` | 14 | Axis title font size |
| `colorbar_thickness` | 20 | Colorbar thickness in pixels |
| `colorbar_len` | 0.8 | Colorbar length as fraction of plot height |
| `contour_line_width` | 1.5 | Default contour line width |
| `contour_levels` | 20 | Default number of contour levels |
| `scatter_marker_size` | 6 | Default scatter marker size |
| `line_width` | 2.0 | Default line width |
| `margin_left` | 80 | Left margin in pixels |
| `margin_right` | 80 | Right margin in pixels |
| `margin_top` | 80 | Top margin in pixels |
| `margin_bottom` | 80 | Bottom margin in pixels |
| `plot_bgcolor` | `"#f8f9fa"` | Background color of plot area |
| `paper_bgcolor` | `"#ffffff"` | Background color of entire figure |

---

## Common Workflows

### Comparing Timesteps

To compare a property across multiple timesteps, create a subplot grid with consistent color ranges:

```python
import numpy as np
from bores.visualization.plotly2d import DataVisualizer
from bores.visualization.base import PropertyMeta, ColorScheme

viz = DataVisualizer()
states = list(bores.run(model, config))

pressure_meta = PropertyMeta(
    name="pressure_grid",
    display_name="Pressure",
    unit="psi",
    log_scale=False,
    color_scheme=ColorScheme.VIRIDIS,
)

# Select snapshots at day 0, 365, and 730
snapshots = [states[0], states[len(states) // 2], states[-1]]
pressure_grids = [s.model.fluid_properties.pressure_grid for s in snapshots]
titles = [f"Day {s.time_in_days:.0f}" for s in snapshots]

fig = viz.make_plots(
    data_list=pressure_grids,
    plot_types="heatmap",
    metadata_list=[pressure_meta] * 3,
    rows=1,
    cols=3,
    subplot_titles=titles,
    shared_yaxes=True,
)
fig.show()
```

### Waterflood Front Tracking

Visualize the water saturation front advancing through the reservoir:

```python
water_sat_meta = PropertyMeta(
    name="water_saturation_grid",
    display_name="Water Saturation",
    unit="fraction",
    log_scale=False,
    color_scheme=ColorScheme.VIRIDIS,
)

# Final water saturation as filled contour
final_water_sat = states[-1].model.fluid_properties.water_saturation_grid

fig = viz.make_plot(
    data=final_water_sat,
    plot_type="contour_filled",
    metadata=water_sat_meta,
    x_label="X (ft)",
    y_label="Y (ft)",
    title="Water Saturation Front",
    contour_levels=15,
    cmin=0.0,
    cmax=1.0,
)
fig.show()
```

### Permeability Field (Log Scale)

Display a heterogeneous permeability field on a logarithmic color scale:

```python
perm_meta = PropertyMeta(
    name="permeability_x",
    display_name="Permeability",
    unit="mD",
    log_scale=True,
    color_scheme=ColorScheme.PLASMA,
)

perm_grid = states[0].model.rock_properties.absolute_permeability.x[0, :, :]  # top layer

fig = viz.make_plot(
    data=perm_grid,
    plot_type="heatmap",
    metadata=perm_meta,
    x_label="X (ft)",
    y_label="Y (ft)",
    title="Permeability Distribution (Top Layer)",
)
fig.show()
```

### Pressure Surface with Cross-Section

Combine a 3D surface view with a cross-section line plot:

```python
viz = DataVisualizer(config=PlotConfig(width=1200, height=500))

# Surface plot
fig_surface = viz.make_plot(
    data=pressure_grid,
    plot_type="surface",
    metadata=pressure_meta,
    x_coords=x_coords,
    y_coords=y_coords,
    x_label="X (ft)",
    y_label="Y (ft)",
    title="Pressure Surface",
)
fig_surface.show()

# Cross-section at the middle row
fig_line = viz.make_plot(
    data=pressure_grid,
    plot_type="line",
    metadata=pressure_meta,
    x_coords=x_coords,
    y_coords=y_coords,
    x_label="Distance (ft)",
    line_mode="horizontal",
    line_indices=[pressure_grid.shape[0] // 2],
    title="Pressure Cross-Section (Middle Row)",
)
fig_line.show()
```
