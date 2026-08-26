# 1D Time Series Plots

## Overview

The 1D visualization module creates line plots, bar charts, scatter plots, and tornado diagrams from series data. These are the standard tools for analyzing production histories, pressure decline curves, recovery factors, and sensitivity results. The module lives in `bores.visualization.plotly1d` and centers on the `DataVisualizer` class.

Every plot method returns a Plotly `Figure` object. You can display it interactively with `.show()`, save it as HTML with `.write_html()`, or export it as a static image with `.write_image()`. The figures are fully interactive: you can zoom, pan, hover for data values, and toggle individual series in the legend.

The 1D visualizer does not take `ModelState` objects directly. Instead, you extract the data you want from your states as numpy arrays and pass those arrays to the visualizer. This gives you full control over what to plot and how to prepare the data (averaging, unit conversion, filtering by well or region) before visualization.

---

## Data Formats

The visualizer accepts three data formats through the `SeriesData` type:

**Single series as a (n, 2) array:**

```python
import numpy as np

# Each row is (x, y)
time_pressure = np.column_stack([
    [s.time_in_days for s in states],
    [s.model.fluid_properties.pressure_grid.mean() for s in states],
])
```

**Multiple series as a list of arrays:**

```python
# Each array is (n, 2)
oil_series = np.column_stack([time_days, avg_oil_sat])
water_series = np.column_stack([time_days, avg_water_sat])
gas_series = np.column_stack([time_days, avg_gas_sat])

data = [oil_series, water_series, gas_series]
```

**Named series as a dictionary:**

```python
data = {
    "Oil Saturation": oil_series,
    "Water Saturation": water_series,
    "Gas Saturation": gas_series,
}
```

When you pass a dictionary, the keys become the legend labels. When you pass a list, names default to "Series 1", "Series 2", etc., unless you provide `series_names` in the render call.

---

## Creating Plots

### `DataVisualizer`

The `DataVisualizer` class is the main entry point for 1D plotting. Create one with optional configuration:

```python
from bores.visualization.plotly1d import DataVisualizer, PlotConfig

# Default configuration
viz = DataVisualizer()

# Custom configuration
viz = DataVisualizer(
    config=PlotConfig(
        width=1000,
        height=500,
        line_width=3.0,
        font_size=14,
        show_legend=True,
        legend_position="top",
    )
)
```

### `make_plot`

The `make_plot()` method creates a single plot:

```python
fig = viz.make_plot(
    data,
    plot_type="line",
    x_label="Time (days)",
    y_label="Average Pressure (psi)",
    title="Reservoir Pressure Decline",
)
fig.show()
```

The `plot_type` parameter accepts either a `PlotType` enum value or a string. Available types are `"line"`, `"bar"`, `"scatter"`, and `"tornado"`.

### `make_plots` (Subplots)

The `make_plots()` method creates a grid of subplots from multiple datasets:

```python
fig = viz.make_plots(
    data_list=[time_pressure, time_oil_sat, time_water_sat],
    plot_types="line",
    rows=3,
    cols=1,
    subplot_titles=["Pressure", "Oil Saturation", "Water Saturation"],
    shared_xaxes=True,
)
fig.show()
```

You can mix plot types across subplots by passing a list of types:

```python
fig = viz.make_plots(
    data_list=[production_data, sensitivity_data],
    plot_types=["line", "tornado"],
    rows=1,
    cols=2,
)
```

### `make_series_plot` (Convenience Function)

For quick one-off line plots, the module provides a standalone function:

```python
from bores.visualization.plotly1d import make_series_plot

fig = make_series_plot(
    data,
    title="Oil Production Rate",
    x_label="Time (days)",
    y_label="Rate (STB/day)",
)
```

This creates a `LineRenderer` with default configuration and renders the data. It is equivalent to creating a `DataVisualizer` and calling `make_plot()` with `plot_type="line"`.

---

## Plot Types

### Line Plots

Line plots are the default and most common type. They connect data points with continuous lines, making them ideal for time series data.

```python
import numpy as np

time_days = np.array([s.time_in_days for s in states])
avg_pressure = np.array([s.model.fluid_properties.pressure_grid.mean() for s in states])

data = {
    "Average Pressure": np.column_stack([time_days, avg_pressure]),
}
fig = viz.make_plot(data, plot_type="line", x_label="Time (days)", y_label="Pressure (psi)")
```

You can plot multiple series on the same axes by including them all in the data dictionary or list.

### Bar Charts

Bar charts display discrete or categorical data. They are useful for comparing values across categories, such as production by well or recovery factor by scenario.

```python
# Recovery factors by scenario
scenarios = np.array([1, 2, 3, 4])
recovery = np.array([0.25, 0.32, 0.41, 0.38])
data = np.column_stack([scenarios, recovery])

fig = viz.make_plot(
    data,
    plot_type="bar",
    x_label="Scenario",
    y_label="Recovery Factor",
    series_names=["Recovery Factor"],
)
```

### Scatter Plots

Scatter plots show individual data points without connecting lines. They support optional trendline fitting. Use them for cross-plots like permeability vs. porosity or rate vs. pressure drawdown.

```python
fig = viz.make_plot(
    data,
    plot_type="scatter",
    x_label="Porosity (fraction)",
    y_label="Permeability (mD)",
)
```

### Tornado Plots

Tornado plots display sensitivity analysis results as horizontal bars, showing the impact of parameter variations on a target metric. Each parameter has two bars: one for the low case and one for the high case, centered on the base case value.

```python
# Each row: [parameter_index, low_delta, high_delta]
sensitivity_data = np.array([
    [1, -50, 80],  # Permeability: base - 50, base + 80
    [2, -30, 25],  # Porosity
    [3, -20, 15],  # Water saturation
    [4, -10, 12],  # Oil viscosity
])
fig = viz.make_plot(
    sensitivity_data,
    plot_type="tornado",
    series_names=["Permeability", "Porosity", "Sw", "Viscosity"],
)
```

---

## Configuration Reference

The `PlotConfig` class controls all visual aspects of 1D plots:

| Parameter | Default | Description |
| --- | --- | --- |
| `width` | 800 | Figure width in pixels |
| `height` | 600 | Figure height in pixels |
| `title` | `None` | Default title for plots |
| `show_legend` | `True` | Whether to display legend |
| `legend_position` | `"right"` | Legend position: top, bottom, left, right |
| `line_width` | 2.0 | Default line width |
| `marker_size` | 8 | Default marker size |
| `opacity` | 0.8 | Default opacity for plot elements |
| `font_family` | `"Arial, sans-serif"` | Font family |
| `font_size` | 12 | Default font size |
| `title_font_size` | 16 | Title font size |
| `show_grid` | `True` | Whether to show grid lines |
| `show_hover` | `True` | Whether to show hover information |
| `hover_mode` | `"x unified"` | Hover interaction mode |
| `background_color` | `"white"` | Figure background color |
| `plot_background_color` | `"white"` | Plot area background color |

---

## Common Workflows

### Production Rate Over Time

```python
import numpy as np
from bores.visualization.plotly1d import DataVisualizer

viz = DataVisualizer()
states = list(bores.run(model, config))

time_days = np.array([s.time_in_days for s in states])
oil_rate = np.array([s.production.oil.sum() for s in states])
water_rate = np.array([s.production.water.sum() for s in states])

data = {
    "Oil Rate": np.column_stack([time_days, oil_rate]),
    "Water Rate": np.column_stack([time_days, water_rate]),
}
fig = viz.make_plot(data, x_label="Time (days)", y_label="Rate (ft³/day)")
fig.show()
```

### Pressure and Saturation Dashboard

```python
avg_pressure = np.array([s.model.fluid_properties.pressure_grid.mean() for s in states])
avg_So = np.array([s.model.fluid_properties.oil_saturation_grid.mean() for s in states])
avg_Sw = np.array([s.model.fluid_properties.water_saturation_grid.mean() for s in states])

fig = viz.make_plots(
    data_list=[
        np.column_stack([time_days, avg_pressure]),
        {
            "Oil": np.column_stack([time_days, avg_So]),
            "Water": np.column_stack([time_days, avg_Sw]),
        },
    ],
    plot_types="line",
    rows=2,
    cols=1,
    subplot_titles=["Average Pressure (psi)", "Average Saturations"],
    shared_xaxes=True,
)
fig.show()
```
