"""Gridding utilities"""

import typing

import numpy as np

from bores.errors import UnsupportedGridFormatError
from bores.grids.base import Grid

__all__ = ["to_pyvista"]


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
    import pyvista as pv
    from bores.grids.io.pyvista import to_pyvista

    pv_grid = to_pyvista(grid, cell_data={"pressure": pressure})
    pv_grid.plot(scalars="pressure", show_edges=True)
    ```
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]

    except ImportError as exc:
        raise UnsupportedGridFormatError(
            "The 'pyvista' library is required for PyVista conversion. "
            "Install it with: pip install pyvista"
        ) from exc

    n_cells = grid.n_cells
    verts_per_cell = 8  # VTK_HEXAHEDRON = 12

    # Build one hex per grid cell from bounding-box corners
    # PyVista flat cell array layout: [n_pts, p0, p1, ..., p7,  n_pts, ...]
    all_points = np.empty((n_cells * verts_per_cell, 3), dtype=np.float64)
    flat_cells = np.empty(n_cells * (verts_per_cell + 1), dtype=np.int64)
    cell_types = np.full(n_cells, 12, dtype=np.uint8)  # VTK_HEXAHEDRON = 12

    for cell_idx in range(n_cells):
        lo = grid.cell_min_xyz[cell_idx]
        hi = grid.cell_max_xyz[cell_idx]
        base = cell_idx * verts_per_cell

        # VTK hex ordering: bottom face CCW (viewed from below), then top face CCW
        all_points[base + 0] = (lo[0], lo[1], lo[2])
        all_points[base + 1] = (hi[0], lo[1], lo[2])
        all_points[base + 2] = (hi[0], hi[1], lo[2])
        all_points[base + 3] = (lo[0], hi[1], lo[2])
        all_points[base + 4] = (lo[0], lo[1], hi[2])
        all_points[base + 5] = (hi[0], lo[1], hi[2])
        all_points[base + 6] = (hi[0], hi[1], hi[2])
        all_points[base + 7] = (lo[0], hi[1], hi[2])

        flat_offset = cell_idx * (verts_per_cell + 1)
        flat_cells[flat_offset] = verts_per_cell
        flat_cells[flat_offset + 1 : flat_offset + 1 + verts_per_cell] = np.arange(
            base, base + verts_per_cell, dtype=np.int64
        )

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
