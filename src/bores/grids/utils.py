"""Gridding utilities"""

import typing

import numba
import numpy as np

from bores.grids.base import Grid
from bores.typing import UnitSystem

__all__ = ["as_pyvista_grid", "convert"]


@numba.njit(parallel=True, cache=True)
def _build_pyvista_arrays(
    cell_min_xyz: np.ndarray,
    cell_max_xyz: np.ndarray,
    all_points: np.ndarray,
    flat_cells: np.ndarray,
    n_vertices_per_cell: int,
):
    n_cells = cell_min_xyz.shape[0]
    for cell_idx in numba.prange(n_cells):  # type: ignore
        low_x = cell_min_xyz[cell_idx, 0]
        low_y = cell_min_xyz[cell_idx, 1]
        low_z = cell_min_xyz[cell_idx, 2]

        high_x = cell_max_xyz[cell_idx, 0]
        high_y = cell_max_xyz[cell_idx, 1]
        high_z = cell_max_xyz[cell_idx, 2]

        base = cell_idx * n_vertices_per_cell

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


def as_pyvista_grid(
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

    ``python
    from bores.grids.utils import as_pyvista_grid

    pv_grid = as_pyvista_grid(grid, cell_data={"pressure": pressure})
    pv_grid.plot(scalars="pressure", show_edges=True)
    ``
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]

    except ImportError as exc:
        raise ImportError(
            "The 'pyvista' library is required for PyVista conversion. "
            "Install it with: pip install pyvista"
        ) from exc

    n_cells = grid.n_cells
    n_vertices_per_cell = 8  # VTK_HEXAHEDRON = 12

    # Build one hex per grid cell from bounding-box corners
    # PyVista flat cell array layout: [n_pts, p0, p1, ..., p7,  n_pts, ...]
    all_points = np.empty((n_cells * n_vertices_per_cell, 3), dtype=np.float64)
    flat_cells = np.empty(n_cells * (n_vertices_per_cell + 1), dtype=np.int64)

    _build_pyvista_arrays(
        cell_min_xyz=grid.cell_min_xyz,
        cell_max_xyz=grid.cell_max_xyz,
        all_points=all_points,
        flat_cells=flat_cells,
        n_vertices_per_cell=n_vertices_per_cell,
    )

    # Apply MapAxes rotation when available
    meta = getattr(grid, "metadata", {}) or {}
    map_axes = meta.get("map_axes", None)
    if map_axes is not None:
        map_axes = map_axes.convert(grid.unit_system)
        # rotation_matrix is (2,2): maps local XY -> map XY
        # all_points[:, :2] has shape (N, 2); rotate in-place
        xy_local = all_points[:, :2] - map_axes.origin  # translate to map origin
        xy_map = xy_local @ map_axes.rotation_matrix.T  # (N,2) @ (2,2) = (N,2)
        all_points[:, :2] = xy_map + map_axes.origin

    cell_types = np.full(n_cells, 12, dtype=np.uint8)  # VTK_HEXAHEDRON = 12
    pv_grid = pv.UnstructuredGrid(flat_cells, cell_types, all_points)

    # Attach built-in geometric arrays
    pv_grid.cell_data["cell_volume"] = grid.cell_volumes
    pv_grid.cell_data["cell_depth"] = grid.cell_center_depths
    pv_grid.cell_data["cell_thickness"] = grid.cell_thickness

    # Attach caller-supplied arrays
    if cell_data:
        for name, array in cell_data.items():
            arr = np.asarray(array)
            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"cell_data[{name!r}] has {arr.shape[0]} entries "
                    f"but grid has {n_cells} cells."
                )
            pv_grid.cell_data[name] = arr
    return pv_grid


_METRES_PER_LENGTH_UNIT: typing.Dict[UnitSystem, float] = {
    UnitSystem.FIELD: 0.3048,  # 1 ft  = 0.3048 m
    UnitSystem.METRIC: 1.0,  # 1 m   = 1 m
    UnitSystem.LAB: 0.01,  # 1 cm  = 0.01 m
    UnitSystem.SI: 1.0,  # 1 m   = 1 m  (SI length unit is metre)
}


def _get_length_conversion_factor(
    from_system: UnitSystem, to_system: UnitSystem
) -> float:
    """
    Return the multiplicative factor to convert a length value from one unit
    system to another.

    :param from_system: Source unit system.
    :param to_system: Target unit system.
    :returns: Conversion factor `f` such that `value_to = value_from * f`.
    """
    return _METRES_PER_LENGTH_UNIT[from_system] / _METRES_PER_LENGTH_UNIT[to_system]


def convert(grid: Grid, *, to: UnitSystem) -> Grid:
    """
    Return a new `bores.grids.base.Grid` with all coordinates
    expressed in the target unit system.

    The `Grid` stores raw numbers and carries a declared
    `bores.typing.UnitSystem` tag. This function rescales
    `vertex_coordinates` by the appropriate length factor and constructs
    a new `Grid` with `unit_system=to`. All derived geometry (face
    areas, cell volumes, centroids, bounding boxes …) is recomputed
    automatically after `Grid` initialization.

    If `grid.unit_system == to` the original grid object is returned
    unchanged (no copy, no allocation).

    **Supported conversions** (any combination of FIELD ↔ METRIC ↔ LAB ↔ SI):

    ```md
    =========  =======  =========
    From       To       Length
    =========  =======  =========
    FIELD      METRIC   0.3048
    FIELD      LAB      30.48
    FIELD      SI       0.3048
    METRIC     FIELD    3.28084
    METRIC     LAB      100.0
    METRIC     SI       1.0
    LAB        METRIC   0.01
    LAB        FIELD    0.032808
    SI         METRIC   1.0
    =========  =======  =========
    ```

    :param grid: Source grid. Must have a valid `unit_system` tag.
    :param to: Target `bores.typing.UnitSystem`.
    :returns: A new `Grid` with rescaled coordinates and `unit_system=to`,
        or the original `grid` if already in the target system.
    :raises ValueError: If `grid.unit_system` is not a recognised `UnitSystem` member.

    Example:

    ```python
    from bores.grids.factories.cartesian import make_cartesian_grid
    from bores.grids.utils import convert
    from bores.typing import UnitSystem

    # Build a grid in field units (feet)
    grid_ft = make_cartesian_grid(
        nx=10, ny=10, nz=5,
        dx=328.084, dy=328.084, dz=16.4042,   # ≈ 100 m cells
        unit_system=UnitSystem.FIELD,
    )

    # Convert to metric (metres)
    grid_m = convert(grid_ft, to=UnitSystem.METRIC)
    assert grid_m.unit_system == UnitSystem.METRIC
    # cell volume should now be ≈ 100 * 100 * 5 = 50,000 m³
    ```
    """
    if grid.unit_system == to:
        return grid

    factor = _get_length_conversion_factor(grid.unit_system, to)
    # Rescale vertex coordinates only.
    # All other geometry is derived and will be recomputed in __attrs_post_init__.
    new_vertex_coordinates = grid.vertex_coordinates * factor
    return Grid(
        vertex_coordinates=new_vertex_coordinates,
        face_vertex_indices=grid.face_vertex_indices,
        face_vertex_offsets=grid.face_vertex_offsets,
        face_cell_indices=grid.face_cell_indices,
        unit_system=to,
        index_dtype=grid.index_dtype,
        floating_dtype=grid.floating_dtype,
        metadata=grid.metadata,
        cell_statuses=grid.cell_statuses,
        connection_types=grid.connection_types,
        face_statuses=grid.face_statuses,
        nnc_cell_pairs=grid.nnc_cell_pairs,
        nnc_transmissibilities=grid.nnc_transmissibilities,
        fault_face_indices=grid.fault_face_indices,
        fault_transmissibility_multipliers=grid.fault_transmissibility_multipliers,
        positive_x_transmissibility_multipliers=grid.positive_x_transmissibility_multipliers,
        negative_x_transmissibility_multipliers=grid.negative_x_transmissibility_multipliers,
        positive_y_transmissibility_multipliers=grid.positive_y_transmissibility_multipliers,
        negative_y_transmissibility_multipliers=grid.negative_y_transmissibility_multipliers,
        positive_z_transmissibility_multipliers=grid.positive_z_transmissibility_multipliers,
        negative_z_transmissibility_multipliers=grid.negative_z_transmissibility_multipliers,
    )
