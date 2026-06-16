"""Gridding utilities"""

import typing

import numba
import numpy as np

from bores.grids.base import Grid
from bores.typing import UnitSystem

__all__ = ["to_pyvista"]


@numba.njit(parallel=True, cache=True)
def _build_pyvista_arrays(
    cell_min_xyz: np.ndarray,
    cell_max_xyz: np.ndarray,
    all_points: np.ndarray,
    flat_cells: np.ndarray,
    verts_per_cell: int,
):
    n_cells = cell_min_xyz.shape[0]
    for cell_idx in numba.prange(n_cells):  # type: ignore
        low_x = cell_min_xyz[cell_idx, 0]
        low_y = cell_min_xyz[cell_idx, 1]
        low_z = cell_min_xyz[cell_idx, 2]

        high_x = cell_max_xyz[cell_idx, 0]
        high_y = cell_max_xyz[cell_idx, 1]
        high_z = cell_max_xyz[cell_idx, 2]

        base = cell_idx * verts_per_cell

        all_points[base + 0, 0] = low_x
        all_points[base + 0, 1] = low_y
        all_points[base + 0, 2] = low_z

        all_points[base + 1, 0] = high_x
        all_points[base + 1, 1] = low_y
        all_points[base + 1, 2] = low_z

        all_points[base + 2, 0] = high_x
        all_points[base + 2, 1] = high_y
        all_points[base + 2, 2] = low_z

        all_points[base + 3, 0] = low_x
        all_points[base + 3, 1] = high_y
        all_points[base + 3, 2] = low_z

        all_points[base + 4, 0] = low_x
        all_points[base + 4, 1] = low_y
        all_points[base + 4, 2] = high_z

        all_points[base + 5, 0] = high_x
        all_points[base + 5, 1] = low_y
        all_points[base + 5, 2] = high_z

        all_points[base + 6, 0] = high_x
        all_points[base + 6, 1] = high_y
        all_points[base + 6, 2] = high_z

        all_points[base + 7, 0] = low_x
        all_points[base + 7, 1] = high_y
        all_points[base + 7, 2] = high_z

        flat_offset = cell_idx * 9
        flat_cells[flat_offset] = 8
        for i in range(8):
            flat_cells[flat_offset + 1 + i] = base + i


def to_pyvista(
    grid: Grid,
    *,
    cell_data: typing.Optional[typing.Dict[str, np.ndarray]] = None,
) -> typing.Any:
    """
    Convert a `bores.grids.base.Grid` to a `pyvista.UnstructuredGrid`.

    Each grid cell is represented by a single VTK hexahedron whose 8
    vertices are the corners of the cell's axis-aligned bounding box
    (`cell_min_xyz` / `cell_max_xyz`). This is exact for axis-aligned
    Cartesian grids and a bounding-box approximation for Voronoi / polyhedral
    cells.

    :param grid: Source `bores.grids.base.Grid`.
    :param cell_data: Optional mapping of field name to a shape `(n_cells,)`
        or `(n_cells, k)` NumPy array.  Each entry is attached as a
        PyVista cell-data array so it can be visualised with
        `pv_grid.plot(scalars="pressure")`.
    :returns: A `pyvista.UnstructuredGrid` instance ready for rendering or further processing.
    :raises UnsupportedGridFormatError: If `pyvista` is not installed.
    :raises ValueError: If a `cell_data` array has a length inconsistent
        with `grid.n_cells`.

    **Example**:

    ```python
    from bores.grids.utils import to_pyvista

    pv_grid = to_pyvista(grid, cell_data={"pressure": pressure})
    pv_grid.plot(scalars="pressure", show_edges=True)
    ```
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]

    except ImportError as exc:
        raise ImportError(
            "The 'pyvista' library is required for PyVista conversion. "
            "Install it with: pip install pyvista"
        ) from exc

    n_cells = grid.n_cells
    verts_per_cell = 8  # VTK_HEXAHEDRON = 12

    # Build one hex per grid cell from bounding-box corners
    # PyVista flat cell array layout: [n_pts, p0, p1, ..., p7,  n_pts, ...]
    all_points = np.empty((n_cells * verts_per_cell, 3), dtype=np.float64)
    flat_cells = np.empty(n_cells * (verts_per_cell + 1), dtype=np.int64)

    _build_pyvista_arrays(
        cell_min_xyz=grid.cell_min_xyz,
        cell_max_xyz=grid.cell_max_xyz,
        all_points=all_points,
        flat_cells=flat_cells,
        verts_per_cell=verts_per_cell,
    )
    cell_types = np.full(n_cells, 12, dtype=np.uint8)  # VTK_HEXAHEDRON = 12
    pv_grid = pv.UnstructuredGrid(flat_cells, cell_types, all_points)

    # Attach built-in geometric arrays
    pv_grid.cell_data["cell_volume"] = grid.cell_volumes
    pv_grid.cell_data["cell_depth"] = grid.cell_center_depths
    pv_grid.cell_data["cell_thickness"] = grid.cell_thickness

    # Attach caller-supplied arrays
    if cell_data:
        for field_name, field_array in cell_data.items():
            arr = np.asarray(field_array)
            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"cell_data[{field_name!r}] has {arr.shape[0]} entries "
                    f"but grid has {n_cells} cells."
                )
            pv_grid.cell_data[field_name] = arr
    return pv_grid


def convert(grid: Grid, *, to: UnitSystem) -> Grid:
    if grid.unit_system == to:
        return grid
    return grid
