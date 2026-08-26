"""
Configuration for the visualization module.

Loads visualization parameters from environment variables with sensible defaults.
Override these by setting environment variables before importing BORES.
"""

import os

__all__ = [
    "COLORBAR_BOUNDARY_PADDING",
    "COLORBAR_HEIGHT_FACTOR",
    "COLORBAR_MIN_LENGTH",
    "COLORBAR_VERTICAL_PADDING",
    "COLORBAR_XPAD",
    "DEFAULT_COLORBAR_MIN_THICKNESS",
    "DEFAULT_COLORBAR_THICKNESS",
    "DEFAULT_MARKER_LINE_COLOR",
    "DEFAULT_MARKER_LINE_WIDTH",
    "MAX_ISOSURFACE_CELLS_3D",
    "MAX_VOLUME_CELLS_3D",
    "RECOMMENDED_VOLUME_CELLS_3D",
    "config_summary",
]


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable with fallback to default."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable with fallback to default."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


MAX_VOLUME_CELLS_3D = _get_int_env("BORES_MAX_VOLUME_CELLS", 1_000_000)
"""Maximum cells for volume rendering before auto-coarsening (default: 1M)"""

RECOMMENDED_VOLUME_CELLS_3D = _get_int_env("BORES_RECOMMENDED_VOLUME_CELLS", 512_000)
"""Target cells after auto-coarsening (default: 512K for 80x80x80)"""

MAX_ISOSURFACE_CELLS_3D = _get_int_env("BORES_MAX_ISOSURFACE_CELLS", 2_000_000)
"""Maximum cells for isosurface rendering (default: 2M)"""


# Colorbar and Plot Styling Defaults
DEFAULT_COLORBAR_THICKNESS = _get_int_env("BORES_COLORBAR_THICKNESS", 15)
"""Default colorbar thickness in pixels"""

DEFAULT_COLORBAR_MIN_THICKNESS = _get_int_env("BORES_COLORBAR_MIN_THICKNESS", 12)
"""Minimum colorbar thickness when scaling down"""

COLORBAR_VERTICAL_PADDING = _get_float_env("BORES_COLORBAR_VERTICAL_PADDING", 0.08)
"""Vertical padding between colorbars in subplots"""

COLORBAR_HEIGHT_FACTOR = _get_float_env("BORES_COLORBAR_HEIGHT_FACTOR", 0.85)
"""Height factor for colorbars relative to subplot height"""

COLORBAR_MIN_LENGTH = _get_float_env("BORES_COLORBAR_MIN_LENGTH", 0.25)
"""Minimum colorbar length as fraction of plot"""

COLORBAR_BOUNDARY_PADDING = _get_float_env("BORES_COLORBAR_BOUNDARY_PADDING", 0.01)
"""Padding from subplot boundary for colorbar positioning"""

COLORBAR_XPAD = _get_int_env("BORES_COLORBAR_XPAD", 10)
"""Horizontal padding for colorbar in pixels"""


# Marker and Line Styling Defaults
DEFAULT_MARKER_LINE_WIDTH = _get_int_env("BORES_MARKER_LINE_WIDTH", 1)
"""Default width for marker outlines"""

DEFAULT_MARKER_LINE_COLOR = os.environ.get("BORES_MARKER_LINE_COLOR", "white")
"""Default color for marker outlines"""


def config_summary() -> str:
    """
    Get a summary of current visualization configuration.

    Returns a formatted string showing all configuration values and their sources
    (environment variable or default).

    Example:
    ```python
    from bores.visualization.config import config_summary

    print(config_summary())
    ```
    """
    lines = [
        "BORES Visualization Configuration",
        "=" * 50,
        "",
        "3D Performance Limits:",
        f"  MAX_VOLUME_CELLS_3D:        {MAX_VOLUME_CELLS_3D:,}",
        f"  RECOMMENDED_VOLUME_CELLS_3D: {RECOMMENDED_VOLUME_CELLS_3D:,}",
        f"  MAX_ISOSURFACE_CELLS_3D:    {MAX_ISOSURFACE_CELLS_3D:,}",
        "",
        "Colorbar Styling:",
        f"  DEFAULT_COLORBAR_THICKNESS:  {DEFAULT_COLORBAR_THICKNESS}px",
        f"  COLORBAR_VERTICAL_PADDING:   {COLORBAR_VERTICAL_PADDING}",
        f"  COLORBAR_HEIGHT_FACTOR:      {COLORBAR_HEIGHT_FACTOR}",
        "",
        "To override, set environment variables before importing BORES:",
        "  export BORES_MAX_VOLUME_CELLS=2000000",
        "  export BORES_COLORBAR_THICKNESS=20",
    ]
    return "\n".join(lines)
