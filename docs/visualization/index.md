# Visualization

BORES includes a built-in visualization system for creating interactive plots of simulation results. The system covers three levels of visualization: 1D time series plots for production histories and performance metrics, 2D maps for spatial property distributions across reservoir layers, and 3D renderings for full-field property visualization with well overlays.

The default 3D module uses Plotly and renders in the browser. For GPU-accelerated rendering, interactive slice planes, and true voxel cell-block displays, BORES also provides an optional PyVista-based 3D module that renders in a native desktop window. Both 3D modules share the same API design and accept the same data sources.

All visualization is handled through `DataVisualizer` classes, one for each module. Each visualizer uses a registry of renderers that map plot types to rendering strategies. You pass in your data (typically from a `ModelState`) and get back a Plotly `Figure` or PyVista `Plotter` object that you can display interactively, save as an image, or embed in a report.

The visualization modules are designed to work directly with BORES data structures. The 3D visualizers accept `ModelState` or `BlackOil` objects and can extract any registered property by name. The 1D and 2D visualizers work with numpy arrays, which you extract from states yourself. This gives you full control over data selection and transformation while keeping the plotting API clean.

## Visualization Modules

| Module | Import | Use Case |
| --- | --- | --- |
| 1D Series | `bores.visualization.plotly1d` | Production rates, pressure decline, recovery factors |
| 2D Maps | `bores.visualization.plotly2d` | Layer slices, areal distributions, cross-sections |
| 3D Volumes (Plotly) | `bores.visualization.plotly3d` | Browser-based 3D: volumes, isosurfaces, well locations |
| 3D Volumes (PyVista) | `bores.visualization.pyvista3d` | GPU-accelerated 3D: cell blocks, slice planes, batch export |
| Base | `bores.visualization.base` | Color schemes, property registry, plot merging |

!!! info "PyVista is Optional"

    The PyVista module requires an additional dependency. Install it with `pip install "bores-framework[pyvista]"`. If PyVista is not installed, all other visualization modules continue to work normally.

## Quick Start

```python
import bores
from bores.visualization import plotly1d, plotly2d, plotly3d

# Run a simulation
states = list(bores.run(model, config))

# 1D: Plot pressure decline over time
import numpy as np

time_pressure = np.column_stack([
    [s.time_in_days for s in states],
    [s.model.fluid_properties.pressure_grid.mean() for s in states],
])
viz1d = plotly1d.DataVisualizer()
fig = viz1d.make_plot(time_pressure, x_label="Time (days)", y_label="Pressure (psi)")
fig.show()

# 2D: Heatmap of oil saturation in a layer
state = states[-1]
So_layer = state.model.fluid_properties.oil_saturation_grid[:, :, 0]
viz2d = plotly2d.DataVisualizer()
fig = viz2d.make_plot(So_layer, plot_type="heatmap")
fig.show()

# 3D (Plotly): Volume rendering of pressure in the browser
viz3d = plotly3d.DataVisualizer()
fig = viz3d.make_plot(state, "pressure")
fig.show()

# 3D (PyVista): Cell block rendering in a native window
from bores.visualization import pyvista3d

viz_pv = pyvista3d.DataVisualizer()
plotter = viz_pv.make_plot(state, "pressure", plot_type="cell_blocks")
plotter.show()
```

## Color Schemes

BORES provides colorblind-friendly color schemes through the `ColorScheme` enum. The default schemes are chosen to be perceptually uniform and accessible:

| Scheme | Type | Best For |
| --- | --- | --- |
| `VIRIDIS` | Sequential | Pressure, permeability (recommended default) |
| `CIVIDIS` | Sequential | Oil saturation, porosity (colorblind-optimized) |
| `PLASMA` | Sequential | Density, concentration |
| `INFERNO` | Sequential | Temperature, viscosity |
| `MAGMA` | Sequential | Gas saturation |
| `RdBu` | Diverging | Pressure changes, saturation differences |
| `RdYlBu` | Diverging | Multi-value diverging data |

Each registered property in the `PropertyRegistry` has a default color scheme assigned, so you do not need to specify one manually unless you want to override the default.

## Property Registry

The 3D visualizers use a `PropertyRegistry` that maps property names to metadata including display names, units, color schemes, and scaling options. When you call `viz3d.make_plot(state, "pressure")`, the registry looks up the property metadata to determine how to label, scale, and color the visualization.

You can access the registry directly:

```python
from bores.visualization.base import property_registry

# List all registered properties
for name in property_registry:
    meta = property_registry[name]
    print(f"{name}: {meta.display_name} ({meta.unit})")

# Check if a property exists
if "pressure" in property_registry:
    meta = property_registry["pressure"]
    print(f"Color scheme: {meta.color_scheme}")
```

Properties with `log_scale=True` (like viscosity and compressibility) are automatically log-transformed for visualization while showing the original physical values in hover text and colorbar labels.

## Merging Plots

The `merge_plots()` function combines multiple Plotly figures into a single subplot grid. This is useful for creating multi-panel comparison views:

```python
from bores.visualization.base import merge_plots

fig_pressure = viz3d.make_plot(state, "pressure")
fig_oil_sat = viz3d.make_plot(state, "oil_saturation")
fig_water_sat = viz3d.make_plot(state, "water_saturation")

combined = merge_plots(
    fig_pressure,
    fig_oil_sat,
    fig_water_sat,
    cols=3,
    subplot_titles=["Pressure", "Oil Saturation", "Water Saturation"],
    title="Reservoir State at Final Time Step",
)
combined.show()
```

## Exporting Images

Configure Plotly image export settings globally with `image_config()`:

```python
from bores.visualization.base import image_config

# Set default export format and resolution
image_config(fmt="png", scale=3)

# Save a Plotly figure
fig.write_image("pressure_map.png")
```

For PyVista plots, use the plotter's screenshot method:

```python
plotter = viz_pv.make_plot(state, "pressure")
plotter.screenshot("pressure_map.png")
```

## Animation Export

Both 3D modules support animation export. The Plotly module exports to HTML with interactive play/pause controls. The PyVista module exports to GIF, MP4, or WebP frame sequences:

```python
# Plotly: Interactive HTML animation
from bores.visualization import plotly3d

viz3d = plotly3d.DataVisualizer()
fig = viz3d.animate(states, "oil_saturation", save="animation.html")

# PyVista: GIF animation
from bores.visualization import pyvista3d

viz_pv = pyvista3d.DataVisualizer()
viz_pv.animate(states, "oil_saturation", save="animation.gif")

# PyVista: MP4 video
viz_pv.animate(states, "oil_saturation", save="animation.mp4")
```

## Environment Configuration

Performance limits and styling defaults can be customized through environment variables. Set these before importing BORES:

```bash
export BORES_MAX_VOLUME_CELLS=2000000      # Max cells for 3D volume rendering
export BORES_COLORBAR_THICKNESS=20         # Colorbar thickness in pixels
export BORES_RECOMMENDED_VOLUME_CELLS=512000  # Target cells after auto-coarsening
```

To see the current configuration:

```python
from bores.visualization.config import config_summary

print(config_summary())
```
