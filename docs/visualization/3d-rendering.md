# 3D Volume Rendering

## Overview

The 3D visualization module renders reservoir simulation data as interactive three-dimensional volumes, isosurfaces, scatter plots, and cell block displays. These visualizations let you inspect the full spatial structure of a reservoir: pressure gradients, saturation fronts, permeability heterogeneity, and well placement in all three dimensions simultaneously.

The module lives in `bores.visualization.plotly3d` and centers on the `DataVisualizer` class. Unlike the 2D module (which takes raw numpy arrays), the 3D module can work directly with `ModelState` and `BlackOil` objects. When you pass a model state and a property name, the visualizer extracts the 3D grid data, looks up the property metadata from the `PropertyRegistry`, maps physical coordinates from the depth grid, and renders the result with appropriate labels and color scales. You can also pass raw 3D numpy arrays for custom data.

Every plot method returns a Plotly `Figure` object with full 3D interactivity. You can rotate, zoom, pan, and hover for cell-level data values. The figures support well overlays, data slicing, custom labels, and animated sequences showing property evolution over time.

3D rendering is computationally more demanding than 2D maps. BORES includes configurable cell count limits (set through environment variables) to prevent browser crashes on large grids. For very large models, you can use the slicing feature to render a subvolume, or reduce resolution with grid coarsening before visualization.

!!! tip "Cell Block Rendering and Interactive Slice Planes"

    If you need true voxel cell-block displays, interactive slice planes, or GPU-accelerated volume rendering, see the [PyVista 3D Rendering](pyvista3d.md) module. The PyVista module provides a `CELL_BLOCKS` plot type and interactive widgets that are not available in the browser-based Plotly module.

---

## Data Sources

The 3D visualizer accepts three types of input:

**`ModelState` (recommended for simulation results):**

```python
from bores.visualization.plotly3d import DataVisualizer

viz = DataVisualizer()

# Plot pressure from a simulation state
states = list(bores.run(model, config))
fig = viz.make_plot(states[-1], property="pressure")
fig.show()
```

When you pass a `ModelState`, the visualizer uses the `PropertyRegistry` to look up metadata for the named property. It also extracts the cell dimensions and depth grid from the model to compute physical coordinates for hover text and labels. The `property` parameter is a registry key such as `"pressure"`, `"oil_saturation"`, `"water_saturation"`, `"gas_saturation"`, `"permeability"`, or any other registered property.

**`BlackOil` (for initial conditions):**

```python
# Plot initial permeability distribution
fig = viz.make_plot(model, property="permeability")
fig.show()
```

This works the same as `ModelState` but uses the model directly. Useful for inspecting the static reservoir description before running a simulation.

**Raw 3D numpy array (for custom data):**

```python
import numpy as np

custom_data = np.random.rand(20, 15, 5)
fig = viz.make_plot(custom_data)
fig.show()

# With a registered property name for metadata
fig = viz.make_plot(custom_data, property="pressure")
fig.show()
```

When you pass a raw array, the visualizer creates generic metadata unless you provide a `property` name that matches a registry entry. Physical coordinates are not available for raw arrays, so hover text shows cell indices instead of distances in feet.

---

## Creating Plots

### `DataVisualizer`

The `DataVisualizer` class is the main entry point for 3D rendering. Create one with optional configuration and property registry:

```python
from bores.visualization.plotly3d import DataVisualizer, PlotConfig, PlotType
from bores.visualization.base import ColorScheme

# Default configuration
viz = DataVisualizer()

# Custom configuration
viz = DataVisualizer(
    config=PlotConfig(
        width=1400,
        height=1000,
        plot_type=PlotType.VOLUME,
        color_scheme=ColorScheme.PLASMA,
        opacity=0.7,
        show_colorbar=True,
        show_axes=True,
    )
)
```

The `PlotConfig` for 3D plots has additional parameters not found in the 2D config, including camera position, lighting, opacity scaling, cell outline styling, and aspect mode. These are covered in the [Configuration Reference](#configuration-reference) section below.

### `make_plot`

The `make_plot()` method creates a single 3D visualization:

```python
fig = viz.make_plot(
    source=states[-1],
    property="pressure",
    plot_type="volume",
    title="Reservoir Pressure at Day 365",
    width=1200,
    height=900,
)
fig.show()
```

The `plot_type` parameter accepts either a `PlotType` enum value or a string. Available types are `"volume"`, `"isosurface"`, and `"scatter_3d"`.

The `source` parameter accepts a `ModelState`, `BlackOil`, or raw 3D numpy array. When using a `ModelState` or `BlackOil`, the `property` parameter is required and must match a key in the `PropertyRegistry`. When using a raw array, `property` is optional.

### `animate`

The `animate()` method creates an animated sequence showing a property changing over time:

```python
states = list(bores.run(model, config))

fig = viz.animate(
    sequence=states,
    property="oil_saturation",
    plot_type="volume",
    frame_duration=200,
    step_size=5,
    title="Oil Saturation Evolution",
)
fig.show()
```

The `sequence` parameter accepts a list of `ModelState` objects, `BlackOil` objects, or raw 3D arrays. The `frame_duration` parameter sets how many milliseconds each frame is displayed. The `step_size` parameter lets you skip frames for performance (1 means every frame, 5 means every fifth frame).

The animation automatically computes consistent color ranges (`cmin` and `cmax`) across all frames so the color mapping stays constant. The resulting figure includes play/pause controls and a time slider.

---

## Plot Types

### Volume Rendering

Volume rendering displays a continuous 3D scalar field where each cell is colored according to its value and the opacity varies to reveal internal structure. This is the default plot type and the most versatile for examining reservoir properties.

```python
fig = viz.make_plot(
    states[-1],
    property="pressure",
    plot_type="volume",
)
fig.show()
```

Volume rendering works well for smooth, continuous properties like pressure and temperature. For properties with sharp boundaries (like saturation fronts), isosurface or cell block plots may be more informative.

You can control opacity scaling through the `PlotConfig` to emphasize high or low values:

```python
viz = DataVisualizer(
    config=PlotConfig(
        use_opacity_scaling=True,
        opacity_scale_values=[
            [0.0, 0.1],  # Low values are nearly transparent
            [0.5, 0.5],  # Mid values are semi-transparent
            [1.0, 1.0],  # High values are fully opaque
        ],
    )
)
```

### Isosurface

Isosurface plots draw 3D surfaces at specific value thresholds within the data. They are the 3D equivalent of contour lines: each surface connects all points where the property equals a specific value.

```python
fig = viz.make_plot(
    states[-1],
    property="oil_saturation",
    plot_type="isosurface",
)
fig.show()
```

Isosurfaces are particularly useful for visualizing flood fronts (where water saturation crosses a threshold), gas-oil contacts, and pressure isobars. They give a clear picture of the 3D geometry of these interfaces.

### 3D Scatter

Scatter plots display individual cells as markers in 3D space. Each marker is positioned at the cell center and colored according to the property value. This is useful for sparse data or for highlighting cells that meet specific criteria.

```python
fig = viz.make_plot(
    states[-1],
    property="water_saturation",
    plot_type="scatter_3d",
)
fig.show()
```

Scatter plots are lighter weight than volume rendering, making them a good choice for quick exploration of large grids where full volume rendering would be slow.

!!! tip "Cell Block Rendering"

    For true voxel cell-block displays with wireframe outlines, use the [PyVista 3D module](pyvista3d.md). The `CELL_BLOCKS` plot type renders each reservoir cell as a solid 3D box with optional outlines, giving an accurate representation of the grid geometry. This plot type is only available in the PyVista module because it requires VTK rendering.

---

## Data Slicing

For large 3D grids, you often want to examine a subvolume rather than the entire reservoir. The `make_plot()` method supports slicing along any combination of the x, y, and z axes. Slicing reduces the data volume while preserving the 3D structure of the result.

Each slice parameter accepts an integer (single index), a tuple (range), a Python `slice` object, or `None` (full dimension):

```python
# Single layer at z-index 2 (maintains 3D structure as a thin slab)
fig = viz.make_plot(states[-1], "pressure", z_slice=2)

# Range of cells in x-direction
fig = viz.make_plot(states[-1], "pressure", x_slice=(10, 20))

# Corner section
fig = viz.make_plot(
    states[-1],
    "oil_saturation",
    x_slice=(0, 25),
    y_slice=(0, 25),
    z_slice=(0, 10),
)

# Every 2nd cell in x using a slice object
fig = viz.make_plot(states[-1], "pressure", x_slice=slice(0, 50, 2))
```

When you slice the data, the plot title is automatically updated to show which portion of the grid is displayed (for example, "X[10:20], Z[0:5]"). The depth grid is sliced to match, so physical coordinates remain correct in hover text.

Slicing is essential for inspecting cross-sections through a 3D model. For example, slicing at a single z-index gives you a plan view of one layer, while slicing at a single x-index gives you a vertical cross-section.

---

## Well Visualization

When working with `ModelState` data, you can overlay well trajectories on the 3D plot. Wells are rendered as colored tubes with optional surface markers and perforation highlights:

```python
fig = viz.make_plot(
    states[-1],
    "pressure",
    show_wells=True,
)
fig.show()
```

The well visualization uses color coding to distinguish well types:

- Injection wells: red (default `#ff4444`)
- Production wells: green (default `#44dd44`)
- Shut-in wells: gray (default `#888888`)

You can customize the well appearance through keyword arguments:

```python
fig = viz.make_plot(
    states[-1],
    "oil_saturation",
    show_wells=True,
    injection_color="#ff6b6b",
    production_color="#51cf66",
    shut_in_color="#aaaaaa",
    wellbore_width=8.0,
    show_surface_marker=True,
    show_perforations=True,
    surface_marker_size=2.0,
)
```

The `WellKwargs` TypedDict defines all available well visualization options:

| Parameter | Default | Description |
| --- | --- | --- |
| `show_wellbore` | `True` | Show wellbore trajectory as colored tube |
| `show_surface_marker` | `True` | Show arrow at surface location |
| `show_perforations` | `False` | Highlight perforated intervals |
| `injection_color` | `"#ff4444"` | Color for injection wells |
| `production_color` | `"#44dd44"` | Color for production wells |
| `shut_in_color` | `"#888888"` | Color for shut-in wells |
| `wellbore_width` | 15.0 | Width of wellbore line in pixels |
| `surface_marker_size` | 2.0 | Size scaling factor for surface markers |

Well visualization only works when `source` is a `ModelState` with active wells. When you pass a raw array or a `BlackOil`, the `show_wells` parameter is ignored.

---

## Labels

Labels are text annotations placed at specific 3D coordinates on the plot. They can display cell values, physical coordinates, property names, and custom text. Labels are useful for annotating specific cells of interest, well locations, or reference points.

### Creating Labels

A `Label` is positioned using a `LabelCoordinate` (grid indices) and displays text from a customizable template:

```python
from bores.visualization.plotly3d import Label, Labels, LabelCoordinate

# Create a label at cell (5, 10, 2)
label = Label(
    position=LabelCoordinate(x=5, y=10, z=2),
    text_template="Pressure: {formatted_value} {unit}",
    font_size=12,
    font_color="#333333",
    background_color="rgba(240, 240, 240, 0.9)",
)
```

The text template supports these format variables:

| Variable | Description |
| --- | --- |
| `{x_index}`, `{y_index}`, `{z_index}` | Cell grid indices |
| `{x_physical}`, `{y_physical}`, `{z_physical}` | Physical coordinates in feet |
| `{value}` | Raw data value at the cell |
| `{formatted_value}` | Formatted value (handles log scale) |
| `{property_name}` | Property display name from metadata |
| `{unit}` | Property unit from metadata |
| `{name}` | Label name (if assigned) |

### Label Collections

The `Labels` class manages collections of labels and provides convenience methods for batch creation:

```python
labels = Labels()

# Add individual labels
labels.add(
    Label(
        position=LabelCoordinate(x=5, y=10, z=0),
        text_template="Injector: {formatted_value} {unit}",
        name="INJ-1",
    )
)

# Add labels at regular grid intervals
labels.add_grid_labels(
    data_shape=(20, 15, 5),
    spacing=(10, 10, 5),
    template="({x_index}, {y_index}, {z_index})",
)

# Add labels at grid corners
labels.add_boundary_labels(
    data_shape=(20, 15, 5),
    template="Corner ({x_index}, {y_index}, {z_index})",
)

# Add labels at well positions
labels.add_well_labels(
    well_positions=[(5, 10, 0), (15, 10, 0)],
    well_names=["INJ-1", "PROD-1"],
)

# Pass to make_plot
fig = viz.make_plot(states[-1], "pressure", labels=labels)
```

Labels are rendered as Plotly 3D annotations with arrows pointing to their position. When cell dimensions and depth grids are available (from `ModelState` source), label positions are converted to physical coordinates automatically.

You can toggle all labels on or off through the `PlotConfig.show_labels` flag.

---

## Configuration Reference

The `PlotConfig` class for 3D plots provides extensive control over rendering:

| Parameter | Default | Description |
| --- | --- | --- |
| `width` | 1200 | Figure width in pixels |
| `height` | 960 | Figure height in pixels |
| `plot_type` | `VOLUME` | Default plot type |
| `color_scheme` | `VIRIDIS` | Default color scheme |
| `opacity` | 0.85 | Default opacity (0.0 to 1.0) |
| `show_colorbar` | `True` | Display color scale bar |
| `show_axes` | `True` | Display 3D axis labels and grid |
| `title` | `""` | Default plot title |
| `use_opacity_scaling` | `False` | Data-driven opacity scaling |
| `opacity_scale_values` | `[[0,0.8],[0.5,0.9],[1,1.0]]` | Opacity scale mapping |
| `aspect_mode` | `None` | Aspect mode: `"cube"`, `"data"`, or `"auto"` |
| `paper_bgcolor` | `"#ffffff"` | Background color of figure |
| `scene_bgcolor` | `"#f8f9fa"` | Background color of 3D scene |
| `show_labels` | `True` | Global toggle for labels |

### Camera Position

The camera position controls the initial viewing angle. It is specified as a `CameraPosition` TypedDict with three components:

```python
from bores.visualization.plotly3d import PlotConfig, CameraPosition

config = PlotConfig(
    camera_position=CameraPosition(
        eye={"x": 2.2, "y": 2.2, "z": 1.8},  # Camera location
        center={"x": 0.0, "y": 0.0, "z": 0.0},  # Look-at point
        up={"x": 0.0, "y": 0.0, "z": 1.0},  # Up direction
    ),
)
```

The `eye` vector sets where the camera is positioned in 3D space. Larger values move the camera further away. The `center` vector sets the point the camera looks at. The `up` vector defines which direction is "up" in the view.

### Lighting

Lighting controls how surfaces are shaded in the 3D scene. The `Lighting` TypedDict provides physically-based parameters:

```python
from bores.visualization.plotly3d import PlotConfig, Lighting

config = PlotConfig(
    lighting=Lighting(
        ambient=0.5,  # Background illumination
        diffuse=0.8,  # Surface scattering
        specular=0.2,  # Shiny highlights
        roughness=0.5,  # Surface roughness
        fresnel=0.2,  # Edge reflections
    ),
)
```

Higher `ambient` values make the scene brighter overall. Higher `specular` values create shinier surfaces. Higher `roughness` values make surfaces appear more matte. The defaults provide good results for most reservoir visualizations.

---

## Common Workflows

### Pressure and Saturation Dashboard

Create multiple 3D views of different properties:

```python
from bores.visualization.plotly3d import DataVisualizer, PlotConfig
from bores.visualization.base import merge_plots

viz = DataVisualizer()
states = list(bores.run(model, config))
final_state = states[-1]

fig_pressure = viz.make_plot(final_state, "pressure", title="Pressure (psi)")
fig_oil_sat = viz.make_plot(final_state, "oil_saturation", title="Oil Saturation")
fig_water_sat = viz.make_plot(final_state, "water_saturation", title="Water Saturation")

# Display each individually
fig_pressure.show()
fig_oil_sat.show()
fig_water_sat.show()
```

### Cross-Section Inspection

Use slicing to examine vertical and horizontal cross-sections:

```python
viz = DataVisualizer()

# Plan view (single layer)
fig_plan = viz.make_plot(
    states[-1],
    "pressure",
    z_slice=0,
    title="Top Layer Pressure",
)

# Vertical cross-section (single row in y)
fig_xsec = viz.make_plot(
    states[-1],
    "pressure",
    y_slice=states[-1].model.grid_shape[1] // 2,
    title="Vertical Cross-Section (Center)",
)

fig_plan.show()
fig_xsec.show()
```

### Animated Saturation Front

Track the waterflood front advancing through the reservoir:

```python
viz = DataVisualizer(
    config=PlotConfig(
        use_opacity_scaling=True,
        opacity_scale_values=[
            [0.0, 0.0],  # Dry cells are transparent
            [0.3, 0.3],  # Partially swept cells semi-transparent
            [0.8, 0.8],  # Mostly swept cells visible
            [1.0, 1.0],  # Fully swept cells opaque
        ],
    )
)

states = list(bores.run(model, config))

fig = viz.animate(
    sequence=states,
    property="water_saturation",
    plot_type="volume",
    frame_duration=300,
    step_size=10,
    title="Waterflood Front Progression",
)
fig.show()
```

### Wells with Property Overlay

Visualize well placement in the context of the reservoir property distribution:

```python
viz = DataVisualizer(
    config=PlotConfig(
        opacity=0.5,  # Make reservoir semi-transparent to see wells
    )
)

fig = viz.make_plot(
    states[-1],
    "oil_saturation",
    show_wells=True,
    injection_color="#4488ff",
    production_color="#ff8844",
    wellbore_width=10.0,
    show_perforations=True,
    title="Oil Saturation with Well Layout",
)
fig.show()
```

---

## Performance Considerations

3D rendering can be resource-intensive for large grids. BORES includes built-in safeguards through environment variable configuration:

- `BORES_MAX_VOLUME_CELLS_3D`: Maximum total cells allowed for volume rendering (prevents browser crashes)
- `BORES_RECOMMENDED_VOLUME_CELLS_3D`: Recommended cell count for smooth interactivity

When the grid exceeds these limits, the visualizer logs a warning. To handle large grids, you have several options:

1. **Slice the data** to render only the region of interest
2. **Use scatter plots** instead of volume rendering (lighter weight)
3. **Increase step_size** for animations (render fewer frames)
4. **Coarsen the grid** before visualization using `bores.coarsen_grid()`
5. **Use the PyVista module** for grids above 500,000 cells (GPU-accelerated rendering handles large grids better than WebGL)

For publication-quality static images, you can export at high resolution using `.write_image()` without performance concerns, since the rendering is done once rather than interactively.
