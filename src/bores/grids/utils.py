"""Gridding utilities"""

import typing

import numba
import numpy as np

from bores.grids.base import Grid
from bores.typing import UnitSystem

__all__ = ["as_pyvista_grid", "convert"]


@numba.njit(parallel=True, cache=True)
def _count_cell_entries(
    cell_face_offsets: np.ndarray,
    cell_face_indices: np.ndarray,
    face_vertex_offsets: np.ndarray,
) -> np.ndarray:
    """
    Count the flat-buffer entries each cell contributes to the VTK polyhedron
    face stream (type 42).

    VTK polyhedron per-cell layout:

        [total_count, n_faces, nv_f0, v0, v1, …, nv_f1, v0, …]

    So the entry count for a cell with *F* faces of vertex counts
    *V0, V1, …, V_{F-1}* is:

        1  (total_count)
      + 1  (n_faces)
      + F  (one nv_fi per face)
      + ΣVi  (all vertex indices)

    Cells with 0 faces (fully suppressed pinchouts) get count 0 and are
    excluded from the output buffer.

    :param cell_face_offsets: CSR offsets array, length `n_cells + 1`.
    :param cell_face_indices: CSR face-index data array.
    :param face_vertex_offsets: CSR offsets into vertex indices, length
        `n_faces + 1`.
    :returns: Shape `(n_cells,)` int64 entry-count array.
    """
    n_cells = cell_face_offsets.shape[0] - 1
    counts = np.zeros(n_cells, dtype=np.int64)

    for cell_idx in numba.prange(n_cells):  # type: ignore
        face_start = cell_face_offsets[cell_idx]
        face_end = cell_face_offsets[cell_idx + 1]
        n_faces = face_end - face_start
        if n_faces == 0:
            continue
        n_verts_total = np.int64(0)
        for face_idx_local in range(face_start, face_end):
            face_idx = cell_face_indices[face_idx_local]
            n_verts_total += (
                face_vertex_offsets[face_idx + 1] - face_vertex_offsets[face_idx]
            )
        # 1 (total_count) + 1 (n_faces) + n_faces (per-face counts) + n_verts_total
        counts[cell_idx] = np.int64(2) + np.int64(n_faces) + n_verts_total

    return counts


@numba.njit(parallel=True, cache=True)
def _fill_cell_entries(
    cell_face_offsets: np.ndarray,
    cell_face_indices: np.ndarray,
    face_vertex_offsets: np.ndarray,
    face_vertex_indices: np.ndarray,
    cell_starts: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Fill the pre-allocated VTK polyhedron face-stream buffer in parallel.

    Each cell owns the slice `out[cell_starts[cell_idx] : cell_starts[cell_idx] + count[cell_idx]]`
    and writes to it independently.

    Per-cell output layout:

        out[cell_starts[cell_idx]]     = total_count   (entries that follow)
        out[cell_starts[cell_idx] + 1] = n_faces
        then for each face face_idx:
            out[position]   = n_verts_fi
            out[position+1 … position+n_verts_fi] = global vertex indices

    :param cell_face_offsets: CSR offsets, length `n_cells + 1`.
    :param cell_face_indices: CSR face-index data.
    :param face_vertex_offsets: CSR offsets into vertex data, length `n_faces + 1`.
    :param face_vertex_indices: Flat vertex index data.
    :param cell_starts: Start position in `out` for each cell, derived
        from the exclusive prefix sum of `_count_cell_entries`.
        Cells with count 0 are skipped (their start value is arbitrary).
    :param out: Pre-allocated int64 output buffer of length
        `sum(_count_cell_entries(…))`.
    """
    n_cells = cell_face_offsets.shape[0] - 1

    for cell_idx in numba.prange(n_cells):  # type: ignore
        face_start = cell_face_offsets[cell_idx]
        face_end = cell_face_offsets[cell_idx + 1]
        n_faces = face_end - face_start
        if n_faces == 0:
            continue

        position = cell_starts[cell_idx]
        total_count_pos = position  # filled last
        position += 1
        out[position] = n_faces
        position += 1

        for face_idx_local in range(face_start, face_end):
            face_idx = cell_face_indices[face_idx_local]
            vertex_start = face_vertex_offsets[face_idx]
            vertex_end = face_vertex_offsets[face_idx + 1]
            n_verts = vertex_end - vertex_start
            out[position] = n_verts
            position += 1
            for vi in range(vertex_start, vertex_end):
                out[position] = face_vertex_indices[vi]
                position += 1

        # total_count = everything written after the leading integer
        out[total_count_pos] = position - total_count_pos - np.int64(1)


def as_pyvista_grid(
    grid: Grid,
    *,
    cell_data: typing.Optional[typing.Dict[str, np.ndarray]] = None,
) -> typing.Any:
    """
    Convert a `bores.grids.base.Grid` to a `pyvista.UnstructuredGrid`.

    Each grid cell is represented as a **VTK polyhedron (type 42)** built
    from the cell's actual face and vertex geometry. This is geometrically
    exact for all grid types; corner-point, Voronoi, Cartesian, and general
    polyhedral.

    Pinched-out cells (cells whose faces were all suppressed during
    corner-point construction) have no VTK representation and are silently
    omitted from the output mesh. Their data is also omitted from any
    attached cell-data arrays so that array lengths always match the number
    of rendered cells. The cells remain present in the source `Grid`
    object so that physics code is unaffected.


    **`MapAxes`**:

    When `grid.metadata["map_axes"]` is present the grid's vertex
    coordinates are rotated into map space before being passed to PyVista.
    The rotation is applied to the shared `vertex_coordinates` array (a
    copy), so the source `Grid` is not mutated.

    :param grid: Source `bores.grids.base.Grid`.
    :param cell_data: Optional mapping of scalar field name to a shape
        `(n_cells,)` NumPy array. Each entry is attached as a PyVista
        cell-data array and can be visualised with
        `pv_grid.plot(scalars="pressure")`. Arrays must have length
        `grid.n_cells`; they are automatically filtered to the valid
        (non-pinched) cells before attachment.
    :returns: A `pyvista.UnstructuredGrid` ready for rendering or
        further PyVista processing.
    :raises ImportError: If `pyvista` is not installed.
    :raises ValueError: If a `cell_data` array has length != `grid.n_cells`.

    Example:
    ```python
    from bores.grids.utils import as_pyvista_grid
    import pyvista as pv

    pv_grid = as_pyvista_grid(grid, cell_data={"pressure": pressure})

    pl = pv.Plotter()
    pl.add_mesh(pv_grid, scalars="pressure", show_edges=True)
    pl.set_scale(zscale=-1)   # flip z: depth increases downward
    pl.show()
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

    # Vertex coordinates (copy so we can rotate in-place)
    all_points = grid.vertex_coordinates.copy()

    # Apply MapAxes rotation when present.
    # Rotates only XY; Z (depth) is unchanged.
    meta = getattr(grid, "metadata", {}) or {}
    map_axes = meta.get("map_axes", None)
    if map_axes is not None:
        map_axes = map_axes.convert(grid.unit_system)
        rotation_matrix = map_axes.rotation_matrix
        if np.all(np.isfinite(rotation_matrix)):
            all_points[:, :2] = all_points[:, :2] @ rotation_matrix.T

    # Build VTK polyhedron face stream
    # Step 1: count entries per cell (parallel)
    counts = _count_cell_entries(
        grid.cell_face_offsets.astype(np.int64),
        grid.cell_face_indices.astype(np.int64),
        grid.face_vertex_offsets.astype(np.int64),
    )

    # valid_cell_mask: cells that will appear in the PyVista mesh
    valid_cell_mask = counts > 0

    # Step 2: exclusive prefix sum -> start positions for each cell
    cell_starts = np.zeros(n_cells, dtype=np.int64)
    cell_starts[valid_cell_mask] = np.concatenate(
        [[0], np.cumsum(counts[valid_cell_mask])[:-1]]
    )
    total_entries = int(counts.sum())

    # Step 3: fill buffer (parallel, no locks)
    flat_cells = np.empty(total_entries, dtype=np.int64)
    _fill_cell_entries(
        grid.cell_face_offsets.astype(np.int64),
        grid.cell_face_indices.astype(np.int64),
        grid.face_vertex_offsets.astype(np.int64),
        grid.face_vertex_indices.astype(np.int64),
        cell_starts,
        flat_cells,
    )

    # Assemble PyVista `UnstructuredGrid`
    n_valid = int(valid_cell_mask.sum())
    cell_types = np.full(n_valid, 42, dtype=np.uint8)  # VTK_POLYHEDRON = 42
    pv_grid = pv.UnstructuredGrid(flat_cells, cell_types, all_points)

    # Attach built-in geometric arrays (filtered to valid cells)
    assert grid.cell_volumes is not None
    pv_grid.cell_data["cell_volume"] = grid.cell_volumes[valid_cell_mask]
    pv_grid.cell_data["cell_depth"] = grid.cell_center_depths[valid_cell_mask]
    pv_grid.cell_data["cell_thickness"] = grid.cell_thickness[valid_cell_mask]

    # Attach caller-supplied arrays
    if cell_data:
        for name, array in cell_data.items():
            arr = np.asarray(array)
            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"cell_data[{name!r}] has {arr.shape[0]} entries "
                    f"but grid has {n_cells} cells."
                )
            pv_grid.cell_data[name] = arr[valid_cell_mask]

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

    ``md
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
    ``

    :param grid: Source grid. Must have a valid `unit_system` tag.
    :param to: Target `bores.typing.UnitSystem`.
    :returns: A new `Grid` with rescaled coordinates and `unit_system=to`,
        or the original `grid` if already in the target system.
    :raises ValueError: If `grid.unit_system` is not a recognised `UnitSystem` member.

    Example:

    ``python
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
    ``
    """
    if grid.unit_system == to:
        return grid

    factor = _get_length_conversion_factor(grid.unit_system, to)
    # Rescale vertex coordinates only.
    # All other geometry is derived and will be recomputed on Grid initialization.
    vertex_coordinates = grid.vertex_coordinates * factor
    cell_volumes = (
        grid.cell_volumes * (factor**3) if grid.cell_volumes is not None else None
    )
    cell_centroids = (
        grid.cell_centroids * factor if grid.cell_centroids is not None else None
    )
    return Grid(
        vertex_coordinates=vertex_coordinates,
        face_vertex_indices=grid.face_vertex_indices,
        face_vertex_offsets=grid.face_vertex_offsets,
        face_cell_indices=grid.face_cell_indices,
        cell_volumes=cell_volumes,
        cell_centroids=cell_centroids,
        unit_system=to,
        index_dtype=grid.index_dtype,
        floating_dtype=grid.floating_dtype,
        metadata=grid.metadata,
        cell_statuses=grid.cell_statuses,
        connection_types=grid.connection_types,
        face_statuses=grid.face_statuses,
        nnc_cell_indices=grid.nnc_cell_indices,
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
