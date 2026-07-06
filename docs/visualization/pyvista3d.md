# PyVista 3D Rendering

## Overview

The PyVista 3D visualization module provides GPU-accelerated volume rendering, interactive slice planes, and true voxel cell-block displays using the VTK rendering engine. While the Plotly-based 3D module (covered in [3D Volume Rendering](3d-rendering.md)) produces interactive browser-based figures, the PyVista module renders in a native desktop window with hardware-accelerated graphics. This makes it better suited for large grids, detailed cell-level inspection, and workflows that need interactive manipulation tools like draggable slice planes, box cropping, and real-time threshold adjustment.

PyVista is an optional dependency. Install it with:

```bash
pip install "bores-framework[pyvista]"
```

If PyVista is not installed, the `bores.visualization.pyvista3d` module is silently unavailable and all other visualization modules continue to work normally.

The module mirrors the Plotly 3D API closely. Both modules provide a `DataVisualizer` class with `make_plot()` and `animate()` methods that accept the same `source`, `property`, `plot_type`, and slicing parameters. The main differences are that the PyVista module returns `pv.Plotter` objects instead of Plotly `Figure` objects, supports an additional `CELL_BLOCKS` plot type, and provides interactive widgets for adjusting rendering parameters without rerunning your code.

Choose PyVista when you need to inspect individual cells, use interactive slice planes, render grids above 100,000 cells smoothly, or export high-resolution screenshots from a native window. Choose Plotly when you need browser-based interactivity, HTML export for sharing, or Jupyter notebook embedding without a display server.

---

## Data Sources

The PyVista 3D visualizer accepts the same three input types as the Plotly 3D module:

**`ModelState` (recommended for simulation results):**

```python
from bores.visualization.pyvista3d import DataVisualizer

viz = DataVisualizer()

states = list(bores.run(model, config))
plotter = viz.make_plot(states[-1], property="pressure")
plotter.show()
```

When you pass a `ModelState`, the visualizer uses the `PropertyRegistry` to look up metadata for the named property. It also extracts the cell dimensions and depth grid from the model to compute physical coordinates. The `property` parameter is a registry key such as `"pressure"`, `"oil_saturation"`, `"water_saturation"`, `"gas_saturation"`, `"permeability"`, or any other registered property.

**`BlackOilFluid` (for initial conditions):**

```python
plotter = viz.make_plot(model, property="permeability")
plotter.show()
```

This works the same as `ModelState` but uses the model directly. Useful for inspecting the static reservoir description before running a simulation.

**Raw 3D numpy array (for custom data):**

```python
import numpy as np

custom_data = np.random.rand(20, 15, 5)
plotter = viz.make_plot(custom_data)
plotter.show()
```

When you pass a raw array, the visualizer creates generic metadata unless you provide a `property` name that matches a registry entry. Physical coordinates are not available for raw arrays, so axis labels show cell indices instead of distances in feet.

---

## Creating Plots

### `DataVisualizer`

The `DataVisualizer` class is the main entry point for PyVista 3D rendering. Create one with optional configuration:

```python
from bores.visualization.pyvista3d import DataVisualizer, PlotConfig, PlotType
from bores.visualization.base import ColorScheme

# Default configuration
viz = DataVisualizer()

# Custom configuration
viz = DataVisualizer(config=PlotConfig(
    width=1400,
    height=1000,
    plot_type=PlotType.CELL_BLOCKS,
    color_scheme=ColorScheme.PLASMA,
    opacity=0.7,
    show_colorbar=True,
    show_axes=True,
    show_cell_outlines=True,
    enable_interactive=True,
))
```

A global instance `viz` is also available for quick use:

```python
from bores.visualization.pyvista3d import viz

plotter = viz.make_plot(states[-1], "pressure")
plotter.show()
```

### `make_plot`

The `make_plot()` method creates a single 3D visualization and returns a `pv.Plotter`:

```python
plotter = viz.make_plot(
    source=states[-1],
    property="pressure",
    plot_type="cell_blocks",
    title="Reservoir Pressure at Day 365",
    width=1200,
    height=900,
)
plotter.show()
```

The `plot_type` parameter accepts either a `PlotType` enum value or a string. Available types are `"volume"`, `"isosurface"`, `"scatter_3d"`, and `"cell_blocks"`. Cell blocks is the default and is unique to the PyVista module.

The `source` parameter accepts a `ModelState`, `BlackOilFluid`, or raw 3D numpy array. When using a `ModelState` or `BlackOilFluid`, the `property` parameter is required and must match a key in the `PropertyRegistry`. When using a raw array, `property` is optional.

Because `make_plot` returns a `pv.Plotter`, you interact with it differently than a Plotly figure:

```python
plotter = viz.make_plot(states[-1], "oil_saturation")

# Display interactively
plotter.show()

# Save a screenshot
plotter.screenshot("oil_saturation.png")

# Save as vector graphic
plotter.save_graphic("oil_saturation.svg")
```

### `animate`

The `animate()` method creates a sequence of frames showing a property changing over time:

```python
states = list(bores.run(model, config))

plotters = viz.animate(
    sequence=states,
    property="oil_saturation",
    plot_type="cell_blocks",
    frame_duration=200,
    step_size=5,
    title="Oil Saturation Evolution",
    save="saturation_animation.gif",
)
```

The `sequence` parameter accepts a list of `ModelState` objects, `BlackOilFluid` objects, or raw 3D arrays. The `frame_duration` parameter sets how many milliseconds each frame is displayed. The `step_size` parameter lets you skip frames for performance (1 means every frame, 5 means every fifth frame).

The `save` parameter accepts a file path string or an exporter object. The format is inferred from the file extension:

```python
# GIF animation
viz.animate(states, "pressure", save="pressure.gif")

# MP4 video
viz.animate(states, "pressure", save="pressure.mp4")

# WebP animation
viz.animate(states, "pressure", save="pressure.webp")
```

You can also pass an exporter object directly for more control:

```python
from bores.visualization.utils import GifExporter, Mp4Exporter

# GIF with infinite loop
viz.animate(states, "pressure", save=GifExporter("output.gif", loop=0))

# MP4 with custom quality
viz.animate(states, "pressure", save=Mp4Exporter("output.mp4", codec="libx264", quality=8))
```

---

## Plot Types

### Cell Blocks (Default)

Cell block rendering displays each reservoir cell as a solid 3D voxel, creating a faithful representation of the grid geometry. This is the default plot type and the most useful for cell-level inspection. Each cell is colored according to its property value and rendered as a separate block, making individual cells visually distinct when cell outlines are enabled.

```python
plotter = viz.make_plot(
    states[-1],
    property="permeability",
    plot_type="cell_blocks",
)
plotter.show()
```

Cell blocks are the recommended plot type for most reservoir visualization tasks. They give an accurate picture of the grid structure, especially for models with variable cell sizes or non-uniform layering. Unlike volume rendering (which interpolates between cells), cell blocks show the actual discrete values in each cell.

You can control subsampling for large grids and threshold filtering to hide low-value cells:

```python
plotter = viz.make_plot(
    states[-1],
    "oil_saturation",
    plot_type="cell_blocks",
    subsampling_factor=2,       # Sample every 2nd cell per axis
    threshold_percentile=10.0,  # Hide cells below 10th percentile
)
```

### Volume Rendering

Volume rendering displays a continuous 3D scalar field using GPU-accelerated ray casting. Each cell is colored and its opacity is modulated by the data value, allowing you to see through low-value regions to the high-value interior structure.

```python
plotter = viz.make_plot(
    states[-1],
    property="pressure",
    plot_type="volume",
)
plotter.show()
```

The volume renderer automatically coarsens grids that exceed the configured cell limit (`BORES_MAX_VOLUME_CELLS_3D`) to maintain interactive frame rates. You can control the rendering backend with the `volume_mapper` keyword:

```python
plotter = viz.make_plot(
    states[-1],
    "pressure",
    plot_type="volume",
    volume_mapper="smart",  # "smart" (auto), "gpu" (force GPU), "fixed_point" (CPU)
    shade=True,             # Enable surface shading
)
```

Volume rendering works well for smooth, continuous properties like pressure and temperature. For properties with sharp boundaries (like saturation fronts), cell blocks or isosurfaces are usually more informative.

### Isosurface

Isosurface plots extract 3D surfaces at specific value thresholds within the data using VTK Marching Cubes. Each surface connects all points where the property equals a specific value.

```python
plotter = viz.make_plot(
    states[-1],
    property="oil_saturation",
    plot_type="isosurface",
    surface_count=5,
)
plotter.show()
```

Isosurfaces are particularly useful for visualizing flood fronts (where water saturation crosses a threshold), gas-oil contacts, and pressure isobars. You can control the isosurface range:

```python
plotter = viz.make_plot(
    states[-1],
    "water_saturation",
    plot_type="isosurface",
    isomin=0.3,
    isomax=0.9,
    surface_count=4,
)
```

### 3D Scatter

Scatter plots display cells above a threshold as individual points in 3D space. Each point is positioned at the cell center and colored according to the property value. This is the lightest-weight plot type, making it a good choice for quick exploration of very large grids.

```python
plotter = viz.make_plot(
    states[-1],
    property="gas_saturation",
    plot_type="scatter_3d",
    threshold=0.05,     # Only show cells with Sg > 0.05
    sample_rate=0.5,    # Render 50% of qualifying cells
    point_size=5.0,
)
plotter.show()
```

---

## Interactive Features

When `enable_interactive` is `True` in the `PlotConfig` (the default), the PyVista viewer adds interactive widgets and keyboard shortcuts that let you manipulate the visualization in real time without rerunning your code. These tools are what distinguish PyVista rendering from Plotly: you can slice, crop, adjust thresholds, and change viewing angles interactively in the native window.

### Sliders

A panel of sliders appears on the left side of the rendering window:

| Slider | Range | Effect |
| --- | --- | --- |
| Opacity | 0.0 to 1.0 | Controls overall opacity of the rendered data |
| Threshold | Data range | Hides cells below the threshold value (grid meshes only) |
| Z-scale | 0.1 to 20.0 | Vertical exaggeration factor for thin reservoirs |
| C-min | Data range | Lower bound of the colormap (grid meshes only) |
| C-max | Data range | Upper bound of the colormap (grid meshes only) |

The threshold and colormap sliders only appear for grid-based plot types (cell blocks, volume). The Z-scale slider is useful for reservoirs that are much wider than they are thick. Setting it to 5 or 10 makes vertical structure easier to see.

### Keyboard Shortcuts

Press `h` at any time to display a help overlay listing all available shortcuts. The shortcuts work in the interactive rendering window:

| Key | Action |
| --- | --- |
| `0` | Reset camera to default position |
| `s` | Save screenshot to file |
| `a` | Toggle axes visibility |
| `g` | Toggle grid and cell edge visibility |
| `k` | Toggle colorbar visibility |
| `1` | Add or remove X-axis slice plane |
| `2` | Add or remove Y-axis slice plane |
| `3` | Add or remove Z-axis slice plane |
| `b` | Toggle box-crop widget |
| `v` | Cycle through view presets (isometric, top, front, right) |
| `h` | Show or hide help overlay |

### Slice Planes

The slice plane shortcuts (`1`, `2`, `3`) add draggable cutting planes to the scene. Each plane has an orange handle that you can grab and drag to move the slice position interactively. This is one of the most powerful features for inspecting internal reservoir structure: you can cut through the model at any position and see the property values on the exposed face.

Slice planes are additive. Press `1` once to add an X-slice, press `2` to add a Y-slice. Press the same key again to remove that slice. You can have all three slice planes active simultaneously.

### Box Cropping

Press `b` to activate a 3D bounding box widget. The box has draggable faces that let you crop the model to any rectangular subvolume. This is useful for isolating a region of interest (for example, the near-well area) without modifying your data or rerunning the visualization.

---

## Data Slicing

For large 3D grids, you can slice the data programmatically before rendering. The `make_plot()` method supports slicing along any combination of the x, y, and z axes, identical to the Plotly 3D module:

```python
# Single layer at z-index 2
plotter = viz.make_plot(states[-1], "pressure", z_slice=2)

# Range of cells in x-direction
plotter = viz.make_plot(states[-1], "pressure", x_slice=(10, 20))

# Corner section
plotter = viz.make_plot(
    states[-1],
    "oil_saturation",
    x_slice=(0, 25),
    y_slice=(0, 25),
    z_slice=(0, 10),
)

# Every 2nd cell in x using a slice object
plotter = viz.make_plot(states[-1], "pressure", x_slice=slice(0, 50, 2))
```

Programmatic slicing is useful when you know the region of interest in advance. For exploratory work, the interactive slice planes (press `1`, `2`, `3`) are more convenient because you can move them in real time.

---

## Well Visualization

When working with `ModelState` data, you can overlay well trajectories on the 3D plot. Wells are rendered as colored tubes with surface markers and well name labels:

```python
plotter = viz.make_plot(
    states[-1],
    "pressure",
    show_wells=True,
)
plotter.show()
```

The well visualization uses color coding to distinguish well types:

- Injection wells: red (default `#ff4444`)
- Production wells: green (default `#44dd44`)
- Shut-in wells: gray (default `#888888`)

Each well is rendered with three components: a casing segment from the surface to the first perforation (dotted gray), colored perforation intervals, and a directional surface marker (arrow pointing down for injectors, up for producers).

You can customize the well appearance through keyword arguments:

```python
plotter = viz.make_plot(
    states[-1],
    "oil_saturation",
    show_wells=True,
    injection_color="#ff6b6b",
    production_color="#51cf66",
    shut_in_color="#aaaaaa",
    wellbore_width=11.0,
    show_surface_marker=True,
    show_well_labels=True,
)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `show_wellbore` | `True` | Show wellbore trajectory as colored tube |
| `show_surface_marker` | `True` | Show directional arrow at surface location |
| `show_well_labels` | `True` | Show well name labels |
| `injection_color` | `"#ff4444"` | Color for injection wells |
| `production_color` | `"#44dd44"` | Color for production wells |
| `shut_in_color` | `"#888888"` | Color for shut-in wells |
| `wellbore_width` | 11.0 | Width of wellbore line in pixels |
| `surface_marker_size` | 2.4 | Size scaling factor for surface markers |

Well visualization only works when `source` is a `ModelState` with active wells. When you pass a raw array or a `BlackOilFluid`, the `show_wells` parameter is ignored.

---

## Configuration Reference

The `PlotConfig` class for PyVista 3D plots extends the base configuration with rendering-engine-specific options:

### General Settings

| Parameter | Default | Description |
| --- | --- | --- |
| `width` | 1200 | Window width in pixels |
| `height` | 960 | Window height in pixels |
| `plot_type` | `CELL_BLOCKS` | Default plot type |
| `color_scheme` | `VIRIDIS` | Default color scheme |
| `opacity` | 0.85 | Default opacity (0.0 to 1.0) |
| `show_colorbar` | `True` | Display color scale bar |
| `show_axes` | `True` | Display 3D axis labels and grid |
| `title` | `""` | Default plot title |
| `show_labels` | `True` | Global toggle for labels |
| `aspect_mode` | `None` | Aspect mode: `"cube"`, `"data"`, or `"auto"` |

### Cell Display

| Parameter | Default | Description |
| --- | --- | --- |
| `show_cell_outlines` | `True` | Show wireframe edges around cells |
| `cell_outline_color` | `"#404040"` | Color for cell outline wireframes |
| `cell_outline_width` | 1.0 | Width of cell outline lines |
| `show_edges` | `True` | Show edges on rendered meshes |
| `n_colors` | 256 | Number of discrete colors in the colormap |

### Rendering

| Parameter | Default | Description |
| --- | --- | --- |
| `background_color` | `"white"` | Background color of the rendering window |
| `off_screen` | `False` | Render without displaying a window (for batch export) |
| `smooth_shading` | `True` | Enable smooth surface shading |
| `notebook` | `False` | Use trame backend for Jupyter notebooks |
| `enable_picking` | `True` | Enable cell picking (click to inspect values) |

### Interactivity

| Parameter | Default | Description |
| --- | --- | --- |
| `enable_interactive` | `True` | Enable sliders, keyboard shortcuts, and widgets |
| `use_opacity_scaling` | `False` | Data-driven opacity modulation |
| `scalar_bar_args` | `None` | Custom arguments for the colorbar (pyvista scalar bar API) |

### Off-Screen Rendering

For batch processing or CI pipelines where no display is available, set `off_screen=True`:

```python
viz = DataVisualizer(config=PlotConfig(
    off_screen=True,
    width=1920,
    height=1080,
))

plotter = viz.make_plot(states[-1], "pressure")
plotter.screenshot("pressure_highres.png")
```

### Jupyter Notebooks

To use PyVista inside Jupyter notebooks, set `notebook=True`. This uses the trame backend to render the 3D scene inline:

```python
viz = DataVisualizer(config=PlotConfig(notebook=True))
plotter = viz.make_plot(states[-1], "pressure")
plotter.show()
```

---

## Comparison with Plotly 3D

Both 3D modules share the same API design but target different use cases:

| Feature | Plotly (`plotly3d`) | PyVista (`pyvista3d`) |
| --- | --- | --- |
| Rendering engine | WebGL (browser) | VTK (native window) |
| Cell blocks plot type | No | Yes |
| Interactive slice planes | No | Yes (press `1`/`2`/`3`) |
| Box cropping | No | Yes (press `b`) |
| Real-time threshold slider | No | Yes |
| GPU volume rendering | No | Yes |
| Jupyter inline | Yes (native) | Yes (via trame, set `notebook=True`) |
| HTML export | Yes (`.write_html()`) | No |
| Animation controls | Play/pause slider in browser | Frame export to GIF/MP4/WebP |
| Large grid performance | Limited (~500K cells) | Good (~2M+ cells with GPU) |
| `show_perforations` wells kwarg | Yes | No |

In general, use Plotly for sharing and embedding, and PyVista for detailed inspection and large models.

---

## Common Workflows

### Cell-Level Inspection

Examine individual cells with cell outlines enabled:

```python
from bores.visualization.pyvista3d import DataVisualizer, PlotConfig

viz = DataVisualizer(config=PlotConfig(
    show_cell_outlines=True,
    cell_outline_color="#404040",
    cell_outline_width=1.0,
    enable_interactive=True,
))

plotter = viz.make_plot(states[-1], "permeability", plot_type="cell_blocks")
plotter.show()
# Use interactive slice planes (press 1/2/3) to cut through the model
# Drag the orange handles to move the slices
```

### Cross-Section with Slice Planes

Use programmatic slicing for a fixed cross-section, or interactive slicing for exploration:

```python
# Fixed cross-section at the center of the model
ny = states[-1].model.grid_shape[1]
plotter = viz.make_plot(
    states[-1],
    "pressure",
    y_slice=ny // 2,
    title="Vertical Cross-Section (Center)",
)
plotter.show()
```

### Waterflood Front with Threshold Filtering

Track the waterflood front by hiding cells where water saturation has not changed:

```python
plotter = viz.make_plot(
    states[-1],
    "water_saturation",
    plot_type="cell_blocks",
    threshold_percentile=20.0,  # Hide cells in the bottom 20%
)
plotter.show()
# Then use the Threshold slider to adjust interactively
```

### High-Resolution Batch Export

Generate publication-quality images without a display:

```python
viz = DataVisualizer(config=PlotConfig(
    off_screen=True,
    width=3840,
    height=2160,
    show_cell_outlines=True,
))

for prop in ["pressure", "oil_saturation", "water_saturation"]:
    plotter = viz.make_plot(states[-1], prop)
    plotter.screenshot(f"{prop}_final.png")
```

### Animated Saturation Front to GIF

Export an animation of the waterflood advancing through the reservoir:

```python
states = list(bores.run(model, config))

viz = DataVisualizer(config=PlotConfig(
    off_screen=True,
    show_cell_outlines=False,
))

viz.animate(
    sequence=states,
    property="water_saturation",
    plot_type="cell_blocks",
    step_size=10,
    save="waterflood.gif",
)
```

### Wells with Property Overlay

Visualize well placement in the context of the reservoir property distribution:

```python
viz = DataVisualizer(config=PlotConfig(opacity=0.5))

plotter = viz.make_plot(
    states[-1],
    "oil_saturation",
    show_wells=True,
    injection_color="#4488ff",
    production_color="#ff8844",
    wellbore_width=11.0,
    show_well_labels=True,
    title="Oil Saturation with Well Layout",
)
plotter.show()
```

---

## Performance Considerations

PyVista renders through VTK, which uses hardware-accelerated graphics. This gives it better performance than browser-based Plotly for large grids, but there are still practical limits.

For cell block rendering, grids up to about 500,000 cells render smoothly on modern hardware. Above 1 million cells, use the `subsampling_factor` parameter or programmatic slicing to reduce the rendered cell count. Volume rendering automatically coarsens grids that exceed `BORES_MAX_VOLUME_CELLS_3D` (default 1 million).

For batch rendering (screenshots, animations), set `off_screen=True` to avoid creating a display window. This is required on headless servers and CI environments. The rendering quality is identical to the interactive window.

The vertical exaggeration slider (Z-scale) is particularly useful for reservoir models that are much wider than they are thick. A typical reservoir might be 5,000 ft wide but only 100 ft thick, making it appear as a thin sheet at 1:1 scale. Setting Z-scale to 10 or 20 makes the vertical structure visible without distorting the horizontal layout.
