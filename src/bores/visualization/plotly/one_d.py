"""
Plotly-based 1D Visualization Suite for 1 Dimensional Series and Time-Series Data.
"""

import collections.abc
import logging
import typing
import weakref
from abc import ABC, abstractmethod
from enum import Enum

import attrs
import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from bores.errors import ValidationError
from bores.typing import TwoDimensionalGrid

logger = logging.getLogger(__name__)


_active_figures: list[weakref.ref] = []


def _track_figure(figure: go.Figure) -> None:
    """Track active figures for memory cleanup."""
    _active_figures.append(weakref.ref(figure, _untrack_figure))


def _untrack_figure(ref: typing.Any) -> None:
    """Remove dead weakref from registry."""
    try:
        _active_figures.remove(ref)
    except ValueError:
        pass


def cleanup_figures() -> None:
    """
    Release memory from all tracked figures.

    Clears figure data to free memory. Useful after creating large
    time-series animations or multiple figures. Call when figures are no longer needed.

    Example:
        ```python
        from bores.visualization.plotly1d import cleanup_figures

        # Create many figures
        for i in range(100):
            plotter.make_plot(...)

        # Free memory when done
        cleanup_figures()
        ```
    """
    for ref in _active_figures[:]:
        fig = ref()
        if fig is not None:
            try:
                fig.data = []
                del fig
            except (AttributeError, RuntimeError) as exc:
                logger.debug(f"Failed to cleanup figure: {exc}")
    _active_figures.clear()


SeriesData = typing.Union[
    TwoDimensionalGrid,  # Single series as (n, 2) array with (x, y) pairs
    typing.Sequence[TwoDimensionalGrid],  # Multiple series
    typing.Mapping[str, TwoDimensionalGrid],  # Named series
]


@attrs.frozen
class PlotConfig:
    """Configuration for 1D plots."""

    # Dimensions
    width: int = 800
    """Figure width in pixels"""

    height: int = 600
    """Figure height in pixels"""

    # Display options
    title: str | None = None
    """Optional title for plots"""

    show_legend: bool = True
    """Whether to display legend"""

    legend_position: typing.Literal["top", "bottom", "left", "right"] = "right"
    """Position of the legend"""

    # Styling
    color_palette: typing.Sequence[str] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    """Default color palette for series"""

    opacity: float = 0.8
    """Default opacity for plot elements"""

    # Grid and axes
    show_grid: bool = True
    """Whether to show grid lines"""

    grid_color: str = "lightgray"
    """Color of grid lines"""

    xaxis_grid_color: str | None = None
    """Color of x-axis grid lines (if None, uses grid_color)"""

    yaxis_grid_color: str | None = None
    """Color of y-axis grid lines (if None, uses grid_color)"""

    axis_line_color: str = "black"
    """Color of axis lines"""

    axis_line_width: float = 1.0
    """Width of axis lines"""

    # Text and labels
    font_family: str = "Arial, sans-serif"
    """Font family for text elements"""

    font_size: int = 12
    """Default font size"""

    title_font_size: int = 16
    """Font size for titles"""

    axis_title_font_size: int = 14
    """Font size for axis titles"""

    # Plot-specific defaults
    line_width: float = 2.0
    """Default width for line plots"""

    marker_size: int = 8
    """Default size for markers"""

    bar_width: float | None = None
    """Width for bars (None = auto)"""

    # Layout margins
    margin_left: int = 80
    """Left margin in pixels"""

    margin_right: int = 80
    """Right margin in pixels"""

    margin_top: int = 80
    """Top margin in pixels"""

    margin_bottom: int = 80
    """Bottom margin in pixels"""

    # Background colors
    background_color: str = "white"
    """Background color of entire figure"""

    plot_background_color: str = "white"
    """Background color of plot area"""

    # Interactive features
    show_hover: bool = True
    """Whether to show hover information"""

    hover_mode: typing.Literal["x", "y", "closest", "x unified", "y unified"] = "x unified"
    """Hover mode for interactive plots"""


class PlotType(str, Enum):
    """Enumeration of available 1D plot types."""

    LINE = "line"
    """Line plots for time series and continuous data."""
    BAR = "bar"
    """Bar charts for categorical or discrete data."""
    TORNADO = "tornado"
    """Tornado plots for sensitivity analysis."""
    SCATTER = "scatter"
    """Scatter plots with optional trendlines."""


class BaseRenderer(ABC):
    """
    Abstract base class for 1D renderers.

    All 1D renderers must implement the render() method to create
    plotly figures from series data.
    """

    def __init__(self, config: PlotConfig | None = None):
        """
        Initialize the renderer with optional configuration.

        :param config: Configuration object for the renderer
        """
        self.config = config or PlotConfig()

    @abstractmethod
    def render(
        self,
        figure: go.Figure,
        data: SeriesData,
        x_label: str = "X",
        y_label: str = "Y",
        series_names: typing.Sequence[str] | None = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render the data as a plotly figure.

        :param data: Series data to render
        :param x_label: Label for x-axis
        :param y_label: Label for y-axis
        :param series_names: Optional names for each series
        :param kwargs: Additional rendering options
        :return: Plotly Figure object
        """
        pass

    def normalize_data(self, data: SeriesData) -> tuple[list[TwoDimensionalGrid], list[str]]:
        """
        Normalize input data into a standard format.

        :param data: Input data in various formats
        :return: Tuple of (list of arrays with shape (n, 2), list of names)
        """
        series_list: list[TwoDimensionalGrid] = []
        names_list: list[str] = []

        if isinstance(data, collections.abc.Mapping):
            # Mapping input: {name: data_array}
            for name, series_data in data.items():
                self.validate_series(series_data=series_data, name=name)  # type: ignore
                series_list.append(series_data)  # type: ignore
                names_list.append(name)  # type: ignore
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, np.ndarray):
            # Sequence input: [data_array1, data_array2, ...]
            for i, series_data in enumerate(data):
                self.validate_series(series_data=series_data, name=f"Series {i}")
                series_list.append(series_data)
                names_list.append(f"Series {i + 1}")
        else:
            # Single array input
            self.validate_series(series_data=data, name="Series")
            series_list.append(typing.cast(TwoDimensionalGrid, data))
            names_list.append("Series")

        return series_list, names_list

    def validate_series(self, series_data: typing.Any, name: str) -> None:
        """
        Validate that series data has correct shape.

        :param series_data: Data array to validate
        :param name: Name of the series for error messages
        """
        if not hasattr(series_data, "ndim"):
            raise ValidationError(f"Series '{name}' must be a numpy array")
        if series_data.ndim != 2:
            raise ValidationError(f"Series '{name}' requires 2D data array")
        if series_data.shape[1] != 2:
            raise ValidationError(f"Series '{name}' requires shape (n, 2) for (x, y) pairs")

    def normalize_param(
        self,
        param: typing.Any,
        default_value: typing.Any,
        num_series: int,
        param_name: str,
    ) -> list[typing.Any]:
        """
        Normalize a parameter to a list matching the number of series.

        :param param: Parameter value (scalar, list, or None)
        :param default_value: Default value if param is None
        :param num_series: Number of series
        :param param_name: Name of parameter for error messages
        :return: List of values, one per series
        """
        if param is None:
            return [default_value] * num_series
        elif isinstance(param, (str, int, float, bool)):
            return [param] * num_series
        elif isinstance(param, (list, tuple)):
            if len(param) != num_series:
                raise ValidationError(
                    f"Length of {param_name} ({len(param)}) must match "
                    f"number of series ({num_series})"
                )
            return list(param)
        raise ValidationError(f"Invalid type for {param_name}")

    def get_color(self, index: int, custom_color: str | None = None) -> str:
        """
        Get color for a series, either custom or from palette.

        :param index: Index of the series
        :param custom_color: Custom color if provided
        :return: Color string
        """
        if custom_color is not None:
            return custom_color
        return self.config.color_palette[index % len(self.config.color_palette)]

    def update_layout(
        self,
        figure: go.Figure,
        title: str | None,
        x_label: str,
        y_label: str,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        log_x: bool = False,
        log_y: bool = False,
        num_series: int = 1,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """
        Apply common layout settings to a figure.

        :param fig: Figure to modify
        :param title: Plot title
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param x_range: X-axis range
        :param y_range: Y-axis range
        :param log_x: Use log scale for x-axis
        :param log_y: Use log scale for y-axis
        :param num_series: Number of series (for legend display)
        :param width: Figure width in pixels (overrides config)
        :param height: Figure height in pixels (overrides config)
        """
        title_text = title or self.config.title

        # Configure legend
        legend_config = dict(
            orientation="v" if self.config.legend_position in ["left", "right"] else "h",
            x=1.02
            if self.config.legend_position == "right"
            else (0 if self.config.legend_position == "left" else 0.5),
            y=0.5
            if self.config.legend_position in ["left", "right"]
            else (1.02 if self.config.legend_position == "top" else -0.1),
            xanchor="left"
            if self.config.legend_position == "right"
            else ("right" if self.config.legend_position == "left" else "center"),
            yanchor="middle"
            if self.config.legend_position in ["left", "right"]
            else ("bottom" if self.config.legend_position == "top" else "top"),
        )

        figure.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor="center",
                font=dict(size=self.config.title_font_size),
            )
            if title_text
            else None,
            xaxis_title=dict(text=x_label, font=dict(size=self.config.axis_title_font_size)),
            yaxis_title=dict(text=y_label, font=dict(size=self.config.axis_title_font_size)),
            width=width if width is not None else self.config.width,
            height=height if height is not None else self.config.height,
            showlegend=self.config.show_legend and num_series > 1,
            legend=legend_config if self.config.show_legend and num_series > 1 else None,
            plot_bgcolor=self.config.plot_background_color,
            paper_bgcolor=self.config.background_color,
            font=dict(family=self.config.font_family, size=self.config.font_size),
            xaxis=dict(
                showgrid=self.config.show_grid,
                gridwidth=1,
                gridcolor=self.config.xaxis_grid_color or self.config.grid_color,
                range=x_range,
                type="log" if log_x else "linear",
                linecolor=self.config.axis_line_color,
                linewidth=self.config.axis_line_width,
            ),
            yaxis=dict(
                showgrid=self.config.show_grid,
                gridwidth=1,
                gridcolor=self.config.yaxis_grid_color or self.config.grid_color,
                range=y_range,
                type="log" if log_y else "linear",
                linecolor=self.config.axis_line_color,
                linewidth=self.config.axis_line_width,
            ),
            hovermode=self.config.hover_mode if self.config.show_hover else False,
            margin=dict(
                l=self.config.margin_left,
                r=self.config.margin_right,
                t=self.config.margin_top,
                b=self.config.margin_bottom,
            ),
        )

    def help(self) -> str:
        """
        Return a help string describing the renderer and its usage.

        :return: Help string
        """
        return f"""
{self.__class__.__name__} renderer

{self.render.__doc__ or ""}
        """


class LineRenderer(BaseRenderer):
    """
    Renderer for line plots (time series, trends).

    Supports multiple series with customizable colors, markers, line styles,
    fills, and interactive features.
    """

    def render(
        self,
        figure: go.Figure,
        data: SeriesData,
        x_label: str = "X",
        y_label: str = "Y",
        series_names: typing.Sequence[str] | None = None,
        line_colors: str | typing.Sequence[str] | None = None,
        line_widths: float | typing.Sequence[float] | None = None,
        line_styles: str | typing.Sequence[str] | None = None,
        show_markers: bool = True,
        marker_sizes: int | typing.Sequence[int] | None = None,
        marker_symbols: str | typing.Sequence[str] | None = None,
        marker_colors: str | typing.Sequence[str] | None = None,
        mode: typing.Literal["lines", "markers", "lines+markers"] = "lines+markers",
        fill_area: bool | typing.Sequence[bool] | None = None,
        fill_colors: str | typing.Sequence[str] | None = None,
        fill_opacity: float = 0.3,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        log_x: bool = False,
        log_y: bool = False,
        hover_template: str | None = None,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render line plot for time-series or trend data.

        :param data: Series data in various formats (see SeriesData type)
        :param x_label: Label for x-axis
        :param y_label: Label for y-axis
        :param series_names: Names for each series (for legend)
        :param line_colors: Color(s) for lines
        :param line_widths: Width(s) for lines
        :param line_styles: Style(s) for lines ('solid', 'dash', 'dot', 'dashdot')
        :param show_markers: Whether to show markers on data points
        :param marker_sizes: Size(s) for markers
        :param marker_symbols: Symbol(s) for markers
        :param marker_colors: Color(s) for markers
        :param mode: Plot mode ('lines', 'markers', or 'lines+markers')
        :param fill_area: Whether to fill area under curves
        :param fill_colors: Colors for filled areas
        :param fill_opacity: Opacity for filled areas
        :param x_range: X-axis range as (min, max)
        :param y_range: Y-axis range as (min, max)
        :param log_x: Use logarithmic x-axis
        :param log_y: Use logarithmic y-axis
        :param hover_template: Custom hover template
        :param title: Plot title
        :param width: Figure width in pixels (overrides config)
        :param height: Figure height in pixels (overrides config)
        :param kwargs: Additional parameters passed to go.Scatter
        :return: Plotly Figure

        Example
        ```python
        import numpy as np

        from bores.visualization.plotly1d import LineRenderer

        # Sample data
        data = {
            "Series A": np.array([[1, 10], [2, 15], [3, 13]]),
            "Series B": np.array([[1, 12], [2, 9], [3, 14]]),
        }
        renderer = LineRenderer()
        fig = renderer.render(data)
        fig.show()
        ```
        """
        # Normalize data
        series_list, names_list = self.normalize_data(data)
        num_series = len(series_list)

        # Override with user-provided names
        if series_names is not None:
            if len(series_names) != num_series:
                raise ValidationError(
                    f"Number of series names ({len(series_names)}) must match "
                    f"number of series ({num_series})"
                )
            names_list = list(series_names)

        # Normalize all parameters
        colors = self.normalize_param(
            param=line_colors,
            default_value=None,
            num_series=num_series,
            param_name="line_colors",
        )
        widths = self.normalize_param(
            param=line_widths,
            default_value=self.config.line_width,
            num_series=num_series,
            param_name="line_widths",
        )
        styles = self.normalize_param(
            param=line_styles,
            default_value="solid",
            num_series=num_series,
            param_name="line_styles",
        )
        m_sizes = self.normalize_param(
            param=marker_sizes,
            default_value=self.config.marker_size,
            num_series=num_series,
            param_name="marker_sizes",
        )
        m_symbols = self.normalize_param(
            param=marker_symbols,
            default_value="circle",
            num_series=num_series,
            param_name="marker_symbols",
        )
        m_colors = self.normalize_param(
            param=marker_colors,
            default_value=None,
            num_series=num_series,
            param_name="marker_colors",
        )
        fills = self.normalize_param(
            param=fill_area,
            default_value=False,
            num_series=num_series,
            param_name="fill_area",
        )
        f_colors = self.normalize_param(
            param=fill_colors,
            default_value=None,
            num_series=num_series,
            param_name="fill_colors",
        )

        # Add each series to the figure
        for i, (series_data, name) in enumerate(zip(series_list, names_list, strict=False)):
            x_values = series_data[:, 0]
            y_values = series_data[:, 1]

            # Determine colors
            line_color = self.get_color(index=i, custom_color=colors[i])
            marker_color = m_colors[i] if m_colors[i] is not None else line_color
            fill_color = f_colors[i] if f_colors[i] is not None else line_color

            # Configure line style
            line_dash_map = {
                "solid": "solid",
                "dash": "dash",
                "dot": "dot",
                "dashdot": "dashdot",
            }
            line_dash = line_dash_map.get(str(styles[i]), "solid")

            # Configure fill
            fill_mode = None
            fillcolor_rgba = None
            if fills[i]:
                fill_mode = "tonexty" if i > 0 else "tozeroy"
                # Convert hex color to rgba
                if fill_color.startswith("#") and len(fill_color) == 7:
                    r = int(fill_color[1:3], 16)
                    g = int(fill_color[3:5], 16)
                    b = int(fill_color[5:7], 16)
                    fillcolor_rgba = f"rgba({r}, {g}, {b}, {fill_opacity})"

            # Create hover template
            if hover_template is None:
                custom_hover = (
                    f"<b>{name}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
                )
            else:
                custom_hover = hover_template

            # Add trace
            trace_kwargs = dict(
                x=x_values,
                y=y_values,
                mode=mode,
                name=name,
                line=dict(color=line_color, width=widths[i], dash=line_dash),
                hovertemplate=custom_hover if self.config.show_hover else None,
                hoverinfo="skip" if not self.config.show_hover else None,
            )

            if show_markers or "markers" in mode:
                trace_kwargs["marker"] = dict(
                    color=marker_color,
                    size=m_sizes[i],
                    symbol=m_symbols[i],
                    line=dict(width=1, color="white"),
                )

            if fill_mode:
                trace_kwargs["fill"] = fill_mode
                if fillcolor_rgba:
                    trace_kwargs["fillcolor"] = fillcolor_rgba

            trace_kwargs.update(kwargs)
            figure.add_trace(go.Scatter(**trace_kwargs))

        # Apply layout
        self.update_layout(
            figure=figure,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_range=x_range,
            y_range=y_range,
            log_x=log_x,
            log_y=log_y,
            num_series=num_series,
            width=width,
            height=height,
        )
        return figure


class BarRenderer(BaseRenderer):
    """
    Renderer for vertical bar charts.

    Supports single, grouped, and stacked bar configurations.
    """

    def render(
        self,
        figure: go.Figure,
        data: SeriesData,
        x_label: str = "Category",
        y_label: str = "Value",
        series_names: typing.Sequence[str] | None = None,
        bar_colors: str | typing.Sequence[str] | None = None,
        bar_mode: typing.Literal["group", "stack", "overlay", "relative"] = "group",
        bar_width: float | None = None,
        show_values: bool = False,
        value_format: str = ".4f",
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        log_y: bool = False,
        hover_template: str | None = None,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        categories: typing.Sequence[str] | None = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render vertical bar chart.

        :param data: Series data (x, y) pairs for each bar series
        :param x_label: Label for x-axis (categories)
        :param y_label: Label for y-axis (values)
        :param series_names: Names for each series
        :param bar_colors: Color(s) for bars
        :param bar_mode: How to display multiple series ('group', 'stack', 'overlay', 'relative')
        :param bar_width: Width of bars (None = auto)
        :param show_values: Whether to show values on bars
        :param value_format: Format string for values on bars
        :param x_range: X-axis range
        :param y_range: Y-axis range
        :param log_y: Use logarithmic y-axis
        :param hover_template: Custom hover template
        :param title: Plot title
        :param width: Figure width in pixels (overrides config)
        :param height: Figure height in pixels (overrides config)
        :param categories: Custom labels for x-axis tick marks (e.g., ["Jan", "Feb", "Mar"])
        :param kwargs: Additional parameters passed to go.Bar
        :return: Plotly `go.Figure`

        Example
        ```python
        import numpy as np

        from bores.visualization.plotly1d import BarRenderer

        # Sample data with numeric x-values
        data = {
            "Series A": np.array([[1, 10], [2, 15], [3, 13]]),
            "Series B": np.array([[1, 12], [2, 9], [3, 14]]),
        }

        # Render with custom category labels
        renderer = BarRenderer()
        fig = renderer.render(
            data,
            categories=["Q1", "Q2", "Q3"],  # Custom x-axis labels
            x_label="Quarter",
            y_label="Revenue ($M)",
        )
        fig.show()
        ```
        """
        # Normalize data
        series_list, names_list = self.normalize_data(data)
        num_series = len(series_list)

        # Override with user-provided names
        if series_names is not None:
            if len(series_names) != num_series:
                raise ValidationError("Number of series names must match number of series")
            names_list = list(series_names)

        # Normalize colors
        colors = self.normalize_param(
            param=bar_colors,
            default_value=None,
            num_series=num_series,
            param_name="bar_colors",
        )

        # Add each series as a bar trace
        for i, (series_data, name) in enumerate(zip(series_list, names_list, strict=False)):
            x_values = series_data[:, 0]
            y_values = series_data[:, 1]

            bar_color = self.get_color(index=i, custom_color=colors[i])

            # Create hover template
            if hover_template is None:
                custom_hover = (
                    f"<b>{name}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
                )
            else:
                custom_hover = hover_template

            # Text on bars if requested
            text_values = None
            if show_values:
                text_values = [f"{val:{value_format}}" for val in y_values]

            trace_kwargs = dict(
                x=x_values,
                y=y_values,
                name=name,
                marker_color=bar_color,
                text=text_values,
                textposition="outside" if show_values else None,
                hovertemplate=custom_hover if self.config.show_hover else None,
                hoverinfo="skip" if not self.config.show_hover else None,
                width=bar_width or self.config.bar_width,
                opacity=self.config.opacity,
            )
            trace_kwargs.update(kwargs)
            figure.add_trace(go.Bar(**trace_kwargs))

        # Set bar mode
        figure.update_layout(barmode=bar_mode)

        # Apply custom x-axis category labels if provided
        if categories is not None:
            # Get unique x-values from first series to determine positions
            first_series = series_list[0]
            x_positions = first_series[:, 0]
            if len(categories) != len(x_positions):
                raise ValidationError(
                    f"Number of categories ({len(categories)}) must match number of x-positions ({len(x_positions)})"
                )
            # Update x-axis with custom tick labels
            figure.update_xaxes(
                tickmode="array",
                tickvals=x_positions,
                ticktext=categories,
            )

        # Apply layout
        self.update_layout(
            figure=figure,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_range=x_range,
            y_range=y_range,
            log_x=False,  # log_x not typically used for categorical
            log_y=log_y,
            num_series=num_series,
            width=width,
            height=height,
        )
        return figure


class TornadoRenderer(BaseRenderer):
    """
    Renderer for tornado plots (sensitivity analysis).

    Tornado plots display horizontal bars showing the impact of different
    variables on a base case, with bars extending left (negative impact)
    and right (positive impact) from a center baseline.
    """

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid | typing.Mapping[str, tuple[float, float, float]] | SeriesData,
        x_label: str = "Impact",
        y_label: str = "Variable",
        series_names: typing.Sequence[str] | None = None,
        positive_color: str = "#2ca02c",
        negative_color: str = "#d62728",
        base_value: float | None = None,
        show_values: bool = True,
        value_format: str = ".2f",
        sort_by_impact: bool = True,
        hover_template: str | None = None,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render tornado plot for sensitivity analysis.

        :param data: Either an (n, 3) array [low, base, high] or dict {var_name: (low, base, high)}
        :param x_label: Label for x-axis (impact values)
        :param y_label: Label for y-axis (variable names)
        :param series_names: Names for variables (if data is array)
        :param positive_color: Color for positive impacts
        :param negative_color: Color for negative impacts
        :param base_value: Base case value (if None, computed from data)
        :param show_values: Whether to show values on bars
        :param value_format: Format string for values
        :param sort_by_impact: Sort variables by total impact magnitude
        :param hover_template: Custom hover template
        :param title: Plot title
        :param width: Figure width in pixels (overrides config)
        :param height: Figure height in pixels (overrides config)
        :param kwargs: Additional parameters
        :return: Plotly Figure

        Example
        ```python
        import numpy as np

        from bores.visualization.plotly1d import TornadoRenderer

        # Sample data
        data = {
            "Variable A": (8, 10, 12),
            "Variable B": (5, 10, 15),
            "Variable C": (9, 10, 11),
        }
        # Or as array:
        # data = np.array([[8, 10, 12], [5, 10, 15], [9, 10, 11]])
        renderer = TornadoRenderer()
        fig = renderer.render(data, title="Tornado Plot Example", series_names=["A", "B", "C"])
        ```
        """
        # Parse data
        if isinstance(data, collections.abc.Mapping):
            var_names = list(data.keys())
            values_array = np.array([data[name] for name in var_names])  # type: ignore
        else:
            if not isinstance(data, np.ndarray):
                raise ValidationError("Data must be numpy array or mapping")
            if data.ndim != 2:
                raise ValidationError("Data array must have shape (n, 3) for [low, base, high]")
            values_array = data
            if series_names is not None:
                if len(series_names) != len(values_array):
                    raise ValidationError("Number of series names must match data rows")
                var_names = list(series_names)
            else:
                var_names = [f"Variable {i + 1}" for i in range(len(values_array))]

        # Extract low, base, high values
        low_values = values_array[:, 0]  # type: ignore
        base_values = values_array[:, 1]  # type: ignore
        high_values = values_array[:, 2]  # type: ignore

        # Compute base value if not provided
        if base_value is None:
            base_value = float(np.mean(base_values))

        # Compute impacts from base
        negative_impacts = low_values - base_value
        positive_impacts = high_values - base_value

        # Sort by total impact if requested
        if sort_by_impact:
            total_impacts = np.abs(negative_impacts) + np.abs(positive_impacts)
            sorted_indices = np.argsort(total_impacts)
            var_names = [var_names[i] for i in sorted_indices]
            negative_impacts = negative_impacts[sorted_indices]
            positive_impacts = positive_impacts[sorted_indices]

        # Add negative impact bars (left side)
        figure.add_trace(
            go.Bar(
                y=var_names,
                x=negative_impacts,
                name="Decrease",
                orientation="h",
                marker_color=negative_color,
                text=[f"{val:{value_format}}" for val in negative_impacts]
                if show_values
                else None,
                textposition="inside",
                hovertemplate=f"<b>%{{y}}</b><br>Low Impact: %{{x:{value_format}}}<extra></extra>"
                if self.config.show_hover
                else None,
            )
        )

        # Add positive impact bars (right side)
        figure.add_trace(
            go.Bar(
                y=var_names,
                x=positive_impacts,
                name="Increase",
                orientation="h",
                marker_color=positive_color,
                text=[f"{val:{value_format}}" for val in positive_impacts]
                if show_values
                else None,
                textposition="inside",
                hovertemplate=f"<b>%{{y}}</b><br>High Impact: %{{x:{value_format}}}<extra></extra>"
                if self.config.show_hover
                else None,
            )
        )

        # Configure layout for tornado plot
        figure.update_layout(
            barmode="overlay",
            title=dict(  # noqa
                text=title or self.config.title or "Tornado Plot - Sensitivity Analysis",
                x=0.5,
                xanchor="center",
                font=dict(size=self.config.title_font_size),  # noqa
            ),
            xaxis_title=dict(  # noqa
                text=x_label,
                font=dict(size=self.config.axis_title_font_size),  # noqa
            ),
            yaxis_title=dict(  # noqa
                text=y_label,
                font=dict(size=self.config.axis_title_font_size),  # noqa
            ),
            width=width if width is not None else self.config.width,
            height=height if height is not None else self.config.height,
            showlegend=self.config.show_legend,
            plot_bgcolor=self.config.plot_background_color,
            paper_bgcolor=self.config.background_color,
            font=dict(family=self.config.font_family, size=self.config.font_size),  # noqa
            xaxis=dict(  # noqa
                showgrid=self.config.show_grid,
                gridcolor=self.config.xaxis_grid_color or self.config.grid_color,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="black",
            ),
            yaxis=dict(  # noqa
                showgrid=False,
                autorange="reversed",  # Top-to-bottom ordering
            ),
            hovermode="y" if self.config.show_hover else False,
            margin=dict(  # noqa
                l=self.config.margin_left + 50,  # Extra space for variable names
                r=self.config.margin_right,
                t=self.config.margin_top,
                b=self.config.margin_bottom,
            ),
        )

        # Add vertical line at base value
        figure.add_vline(
            x=0,
            line_width=2,
            line_color="black",
            annotation_text=f"Base: {base_value:{value_format}}",
            annotation_position="top",
        )
        return figure


class ScatterRenderer(BaseRenderer):
    """
    Renderer for 1D scatter plots with optional trendlines.
    """

    def render(
        self,
        figure: go.Figure,
        data: SeriesData,
        x_label: str = "X",
        y_label: str = "Y",
        series_names: typing.Sequence[str] | None = None,
        marker_colors: str | typing.Sequence[str] | None = None,
        marker_sizes: int | typing.Sequence[int] | None = None,
        marker_symbols: str | typing.Sequence[str] | None = None,
        show_trendline: bool = False,
        trendline_type: typing.Literal["linear", "polynomial", "exponential"] = "linear",
        polynomial_order: int = 2,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        log_x: bool = False,
        log_y: bool = False,
        hover_template: str | None = None,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render scatter plot with optional trendlines.

        :param data: Series data (x, y) pairs
        :param x_label: Label for x-axis
        :param y_label: Label for y-axis
        :param series_names: Names for each series
        :param marker_colors: Color(s) for markers
        :param marker_sizes: Size(s) for markers
        :param marker_symbols: Symbol(s) for markers
        :param show_trendline: Whether to show trendline
        :param trendline_type: Type of trendline ('linear', 'polynomial', 'exponential')
        :param polynomial_order: Order for polynomial trendline
        :param x_range: X-axis range
        :param y_range: Y-axis range
        :param log_x: Use logarithmic x-axis
        :param log_y: Use logarithmic y-axis
        :param hover_template: Custom hover template
        :param title: Plot title
        :param width: Figure width in pixels (overrides config)
        :param height: Figure height in pixels (overrides config)
        :param kwargs: Additional parameters passed to go.Scatter
        :return: Plotly `go.Figure`

        Example:
        ```python
        import numpy as np

        from bores.visualization.plotly1d import ScatterRenderer

        # Sample data
        data = {
            "Series A": np.array([[1, 10], [2, 15], [3, 13]]),
            "Series B": np.array([[1, 12], [2, 9], [3, 14]]),
        }
        renderer = ScatterRenderer()
        fig = renderer.render(data, show_trendline=True, trendline_type="linear")
        fig.show()
        ```
        """
        # Normalize data
        series_list, names_list = self.normalize_data(data)
        num_series = len(series_list)

        # Override with user-provided names
        if series_names is not None:
            if len(series_names) != num_series:
                raise ValidationError("Number of series names must match number of series")
            names_list = list(series_names)

        # Normalize parameters
        colors = self.normalize_param(
            param=marker_colors,
            default_value=None,
            num_series=num_series,
            param_name="marker_colors",
        )
        sizes = self.normalize_param(
            param=marker_sizes,
            default_value=self.config.marker_size,
            num_series=num_series,
            param_name="marker_sizes",
        )
        symbols = self.normalize_param(
            param=marker_symbols,
            default_value="circle",
            num_series=num_series,
            param_name="marker_symbols",
        )

        # Add each series as scatter trace
        for i, (series_data, name) in enumerate(zip(series_list, names_list, strict=False)):
            x_values = series_data[:, 0]
            y_values = series_data[:, 1]

            marker_color = self.get_color(index=i, custom_color=colors[i])

            # Create hover template
            if hover_template is None:
                custom_hover = (
                    f"<b>{name}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
                )
            else:
                custom_hover = hover_template

            # Add scatter trace
            trace_kwargs = dict(  # noqa
                x=x_values,
                y=y_values,
                mode="markers",
                name=name,
                marker=dict(color=marker_color, size=sizes[i], symbol=symbols[i]),  # noqa
                hovertemplate=custom_hover if self.config.show_hover else None,
                hoverinfo="skip" if not self.config.show_hover else None,
            )
            trace_kwargs.update(kwargs)
            figure.add_trace(go.Scatter(**trace_kwargs))

            # Add trendline if requested
            if show_trendline:
                self._add_trendline(
                    figure=figure,
                    x_values=x_values,
                    y_values=y_values,
                    trendline_type=trendline_type,
                    polynomial_order=polynomial_order,
                    color=marker_color,
                    name=name,
                )

        self.update_layout(
            figure=figure,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_range=x_range,
            y_range=y_range,
            log_x=log_x,
            log_y=log_y,
            num_series=num_series,
            width=width,
            height=height,
        )
        return figure

    def _add_trendline(
        self,
        figure: go.Figure,
        x_values: np.ndarray,
        y_values: np.ndarray,
        trendline_type: typing.Literal["linear", "polynomial", "exponential"],
        polynomial_order: int,
        color: str,
        name: str,
    ) -> None:
        """
        Add trendline to the figure.

        :param fig: Figure to add trendline to
        :param x_values: X coordinates (1D array)
        :param y_values: Y coordinates (1D array)
        :param trendline_type: Type of trendline
        :param polynomial_order: Order for polynomial fit
        :param color: Color for trendline
        :param name: Name of the series
        """
        if trendline_type == "linear":
            coeffs = np.polyfit(x_values, y_values, 1)
            x_trend = np.linspace(x_values.min(), x_values.max(), 100)
            y_trend = np.polyval(coeffs, x_trend)
            equation = f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}"
        elif trendline_type == "polynomial":
            coeffs = np.polyfit(x_values, y_values, polynomial_order)
            x_trend = np.linspace(x_values.min(), x_values.max(), 100)
            y_trend = np.polyval(coeffs, x_trend)
            equation = f"Polynomial (order {polynomial_order})"
        elif trendline_type == "exponential":
            # Fit exponential: y = a * exp(b * x)
            coeffs = np.polyfit(x_values, np.log(y_values + 1e-10), 1)
            x_trend = np.linspace(x_values.min(), x_values.max(), 100)
            y_trend = np.exp(np.polyval(coeffs, x_trend))
            equation = f"y = {np.exp(coeffs[1]):.3f} * exp({coeffs[0]:.3f}x)"
        else:
            return

        figure.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name=f"{name} - Trendline",
                line=dict(color=color, dash="dash", width=2),
                hovertemplate=f"<b>Trendline</b><br>{equation}<extra></extra>",
            )
        )


class Plotter:
    """
    Plots 1D series data.

    Supports multiple plot types via registered renderers.
    """

    def __init__(self, config: PlotConfig | None = None):
        """
        Initialize visualizer with configuration.

        :param config: Configuration for all renderers
        """
        self._config = config or PlotConfig()
        self._renderers: dict[PlotType, BaseRenderer] = {
            PlotType.LINE: LineRenderer(self._config),
            PlotType.BAR: BarRenderer(self._config),
            PlotType.TORNADO: TornadoRenderer(self._config),
            PlotType.SCATTER: ScatterRenderer(self._config),
        }

    @property
    def config(self) -> PlotConfig:
        """Get the current plot configuration."""
        return self._config

    def register_renderer(self, plot_type: PlotType, renderer_instance: BaseRenderer) -> None:
        """
        Register a renderer for a plot type.

        :param plot_type: PlotType enum value
        :param renderer_instance: Renderer instance implementing BaseRenderer
        """
        if not isinstance(plot_type, PlotType):
            raise ValidationError(f"Invalid plot type: {plot_type}")
        self._renderers[plot_type] = renderer_instance

    def get_renderer(self, plot_type: PlotType) -> BaseRenderer:
        """
        Get the renderer for a specific plot type.

        :param plot_type: The type of plot to get
        :return: The renderer instance for the specified plot type
        """
        if not isinstance(plot_type, PlotType):
            raise ValidationError(f"Invalid plot type: {plot_type}")
        renderer = self._renderers.get(plot_type, None)
        if renderer is None:
            raise ValidationError(f"No renderer registered for plot type: {plot_type}")
        return renderer

    def make_plot(
        self,
        data: SeriesData,
        plot_type: PlotType | str = PlotType.LINE,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create a 1D plot of the provided data.

        :param data: Series data to visualize
        :param plot_type: Type of plot (PlotType enum or string)
        :param kwargs: Additional parameters for the renderer
        :return: Plotly Figure
        """
        # Convert string plot_type to PlotType enum if needed
        if isinstance(plot_type, str):
            try:
                plot_type = PlotType(plot_type)
            except ValueError:
                raise ValidationError(
                    f"Invalid plot_type string: '{plot_type}'. "
                    f"Valid options are: {', '.join(pt.value for pt in PlotType)}"
                )

        renderer = self.get_renderer(plot_type)
        figure = go.Figure()
        return renderer.render(figure, data, **kwargs)

    def make_plots(
        self,
        data_list: typing.Sequence[SeriesData],
        plot_types: PlotType | str | typing.Sequence[PlotType | str] = PlotType.LINE,
        rows: int = 1,
        cols: int = 1,
        subplot_titles: typing.Sequence[str] | None = None,
        shared_xaxes: bool = True,
        shared_yaxes: bool = True,
        vertical_spacing: float | None = None,
        horizontal_spacing: float | None = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create subplots with multiple 1D visualizations.

        :param data_list: Sequence of series data to visualize
        :param plot_types: Plot type(s) to use (single type applied to all, or list matching data_list)
        :param rows: Number of subplot rows
        :param cols: Number of subplot columns
        :param subplot_titles: Optional titles for each subplot
        :param shared_xaxes: Whether to share x-axes across subplots
        :param shared_yaxes: Whether to share y-axes across subplots
        :param vertical_spacing: Vertical spacing between subplots (0.0 to 1.0)
        :param horizontal_spacing: Horizontal spacing between subplots (0.0 to 1.0)
        :param kwargs: Additional parameters passed to renderers
        :return: Plotly figure with subplots
        """
        if len(data_list) > rows * cols:
            raise ValidationError(
                f"Too many plots ({len(data_list)}) for {rows}x{cols} subplot grid"
            )

        if isinstance(plot_types, str):
            try:
                plot_types = PlotType(plot_types)
            except ValueError:
                raise ValidationError(
                    f"Invalid plot_type string: '{plot_types}'. "
                    f"Valid options are: {', '.join(pt.value for pt in PlotType)}"
                )

        # Handle plot types. Expand to list if single value
        if isinstance(plot_types, PlotType):
            plot_types_list = [plot_types] * len(data_list)
        elif isinstance(plot_types, collections.abc.Sequence):
            plot_types_list = [PlotType(pt) if isinstance(pt, str) else pt for pt in plot_types]
        else:
            plot_types_list = [plot_types] * len(data_list)

        if len(plot_types_list) != len(data_list):
            raise ValidationError(
                f"Number of plot types ({len(plot_types_list)}) must match "
                f"number of data arrays ({len(data_list)})"
            )

        # Create subplot figure
        subplot_kwargs: dict[str, typing.Any] = {
            "rows": rows,
            "cols": cols,
            "subplot_titles": subplot_titles,
            "shared_xaxes": shared_xaxes,
            "shared_yaxes": shared_yaxes,
        }
        if vertical_spacing is not None:
            subplot_kwargs["vertical_spacing"] = vertical_spacing
        if horizontal_spacing is not None:
            subplot_kwargs["horizontal_spacing"] = horizontal_spacing

        figure = make_subplots(**subplot_kwargs)

        # Add each plot to its subplot
        for idx, (data, plot_type) in enumerate(zip(data_list, plot_types_list, strict=False)):
            row = (idx // cols) + 1
            col = (idx % cols) + 1

            # Create temporary figure for this subplot
            temp_figure = go.Figure()
            # Render on temporary figure
            renderer = self.get_renderer(plot_type)
            temp_figure = renderer.render(temp_figure, data, **kwargs)

            # Add traces to main figure at the correct subplot position
            for trace in temp_figure.data:
                figure.add_trace(trace, row=row, col=col)

        # Update layout
        figure.update_layout(
            width=self.config.width,
            height=self.config.height * rows,  # Scale height by number of rows
            showlegend=self.config.show_legend,
        )
        _track_figure(figure)
        return figure

    def help(self, plot_type: PlotType | None = None) -> str:
        """
        Print help information about available plot types and their parameters.

        :param plot_type: Specific plot type to get help for (or None for all)
        :return: The help string

        Example:
        ```python
        from bores.visualization.plotly1d import viz, PlotType

        # Get help for all plot types
        print(viz.help())"""
        if plot_type is not None:
            renderer = self.get_renderer(plot_type)
            return renderer.help()

        help_strings = []
        for pt, renderer in self._renderers.items():
            help_strings.append(f"=== {pt.value} Plot ===\n{renderer.help()}\n")
        return "\n".join(help_strings)


plotter = Plotter()
"""Global Plotter instance for 1D plots."""


def make_series_plot(
    data: SeriesData,
    title: str = "Series Plot",
    x_label: str = "X",
    y_label: str = "Y",
    **kwargs: typing.Any,
) -> go.Figure:
    """
    Create a series plot (line plot). Convenience wrapper for `LineRenderer`.

    :param data: Series data to plot
    :param title: Plot title
    :param x_label: X-axis label
    :param y_label: Y-axis label
    :param kwargs: Additional parameters passed to `LineRenderer`
    :return: Plotly Figure
    """
    config = PlotConfig()
    renderer = LineRenderer(config)
    return renderer.render(data, x_label=x_label, y_label=y_label, title=title, **kwargs)
