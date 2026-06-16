import typing

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from bores.errors import InvalidFaceConnectivityError, InvalidGridError, ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import (
    ELEMENT_FACE_TABLES,
    CanonicalFaceKey,
    _FaceRecord,
    assemble_grid,
)
from bores.typing import (
    FloatArray,
    IntArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["make_corner_point_grid"]

VertexCoordinates: TypeAlias = FloatArray[TwoDimensions]
"""Shape `(n_points, 3)` — 3-D (x, y, z) vertex coordinates."""

FaceVertexList: TypeAlias = typing.List[int]
"""Ordered list of vertex indices for a single face (CCW from owner)."""


CoordArray: TypeAlias = FloatArray[ThreeDimensions]
"""Corner-point COORD array, shape `(NY+1, NX+1, 6)`."""

ZcornArray: TypeAlias = FloatArray[ThreeDimensions]
"""Corner-point ZCORN array, shape `(NZ*2, NY*2, NX*2)`."""

ActnumArray: TypeAlias = IntArray[ThreeDimensions]
"""Corner-point ACTNUM array, shape `(NZ, NY, NX)`; 1 = active, 0 = inactive."""


def make_corner_point_grid(
    *,
    coord: CoordArray,
    zcorn: ZcornArray,
    actnum: typing.Optional[ActnumArray] = None,
    vertex_tolerance: float = 1e-8,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Factory for ECLIPSE-style corner-point (pillar) grids.
    Builds a corner-point grid from COORD, ZCORN, and optional ACTNUM.

    Corner-point grids define cell geometry via pillar lines (COORD) and
    corner depths (ZCORN). Each cell is bounded by 4 pillars and has 8
    corner points obtained by intersecting depth planes with the pillars.

    This is the standard representation for GRDECL / ECLIPSE / ResInsight files.

    Example usage:

    ```python
    grid = make_corner_point_grid(
        coord=coord_array,   # shape (NY+1, NX+1, 6)
        zcorn=zcorn_array,   # shape (NZ*2, NY*2, NX*2)
        actnum=actnum_array, # shape (NZ, NY, NX) (optional)
    )
    ```

    :param coord: Shape `(NY+1, NX+1, 6)` pillar array. Each entry
        contains `[x_top, y_top, z_top, x_bot, y_bot, z_bot]` defining
        the top and bottom anchor points of a pillar.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array. For each
        cell `(i, j, k)` the 8 corner z-values are at indices
        `[2k:2k+2, 2j:2j+2, 2i:2i+2]` in `[top/bot, left/right, near/far]` order.
    :param actnum: Shape `(NZ, NY, NX)` integer mask (1=active, 0=inactive).
        If `None`, all cells are treated as active.
    :param metadata: Optional metadata dictionary.
    :returns: A fully initialised `bores.grids.base.Grid` containing
        only active cells.
    :raises ValidationError: If COORD or ZCORN shapes are inconsistent.
    """
    coord_arr = np.asarray(coord, dtype=np.float64)
    zcorn_arr = np.asarray(zcorn, dtype=np.float64)
    if coord_arr.ndim != 3 or coord_arr.shape[2] != 6:
        raise ValidationError(
            f"coord must have shape (NY+1, NX+1, 6); got {coord_arr.shape!r}."
        )
    if zcorn_arr.ndim != 3:
        raise ValidationError(f"zcorn must be a 3-D array; got ndim={zcorn_arr.ndim}.")

    ny_plus1, nx_plus1 = coord_arr.shape[:2]
    nx = nx_plus1 - 1
    ny = ny_plus1 - 1
    nz = zcorn_arr.shape[0] // 2

    if zcorn_arr.shape != (nz * 2, ny * 2, nx * 2):
        raise ValidationError(
            f"zcorn shape {zcorn_arr.shape!r} is inconsistent with "
            f"coord-derived grid dimensions ({nx} x {ny} x {nz})."
        )

    if actnum is None:
        actnum_arr = np.ones((nz, ny, nx), dtype=np.int32)
    else:
        actnum_arr = np.asarray(actnum, dtype=np.int32)
        if actnum_arr.shape != (nz, ny, nx):
            raise ValidationError(
                f"actnum shape {actnum_arr.shape!r} does not match "
                f"grid dimensions ({nx} x {ny} x {nz})."
            )

    vertex_coordinates, face_vertex_indices, face_vertex_offsets, face_cell_indices = (
        _compute_corner_point_geometry(
            coord=coord_arr,
            zcorn=zcorn_arr,
            actnum=actnum_arr,
            vertex_tolerance=vertex_tolerance,
        )
    )
    return assemble_grid(
        vertex_coordinates,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
    )


@numba.njit(cache=True)
def _interpolate_pillar_point(
    pillar_top: FloatArray[OneDimension],
    pillar_bottom: FloatArray[OneDimension],
    z: float,
) -> FloatArray[OneDimension]:
    """
    Interpolate an (x, y) position along a pillar at depth `z`.

    The pillar is a straight line from `pillar_top` to `pillar_bot`.
    The interpolation parameter is linear in z.

    :param pillar_top: Shape `(3,)` array `[x, y, z]` of pillar top.
    :param pillar_bot: Shape `(3,)` array `[x, y, z]` of pillar bottom.
    :param z: Depth at which to evaluate the pillar position.
    :returns: Shape `(3,)` array `[x, y, z]` on the pillar at depth `z`.
    """
    xyz = np.empty(3, dtype=np.float64)
    dz = pillar_bottom[2] - pillar_top[2]
    if abs(dz) < 1e-14:
        xyz[0] = pillar_top[0]
        xyz[1] = pillar_top[1]
        xyz[2] = z
        return xyz

    t = (z - pillar_top[2]) / dz
    xyz[0] = pillar_top[0] + t * (pillar_bottom[0] - pillar_top[0])
    xyz[1] = pillar_top[1] + t * (pillar_bottom[1] - pillar_top[1])
    xyz[2] = z
    return xyz


@numba.njit(parallel=True, cache=True)
def _compute_active_cell_corner_coordinates(
    active_cells: np.ndarray, coord: CoordArray, zcorn: ZcornArray
) -> FloatArray[ThreeDimensions]:
    """
    Compute all active-cell corner coordinates.

    :return: Array of shape (n_active_cells, 8, 3)
    """
    n_active_cells = active_cells.shape[0]
    corner_coordinates = np.empty(
        (n_active_cells, 8, 3),
        dtype=np.float64,
    )
    pillar_order = np.array(
        [0, 1, 2, 3, 0, 1, 2, 3],
        dtype=np.int64,
    )

    for cell_index in numba.prange(n_active_cells):  # type: ignore
        k = active_cells[cell_index, 0]
        j = active_cells[cell_index, 1]
        i = active_cells[cell_index, 2]

        pillar_tops = np.empty((4, 3), dtype=np.float64)
        pillar_bottoms = np.empty((4, 3), dtype=np.float64)

        pillar_tops[0] = coord[j, i, :3]
        pillar_bottoms[0] = coord[j, i, 3:]

        pillar_tops[1] = coord[j, i + 1, :3]
        pillar_bottoms[1] = coord[j, i + 1, 3:]

        pillar_tops[2] = coord[j + 1, i, :3]
        pillar_bottoms[2] = coord[j + 1, i, 3:]

        pillar_tops[3] = coord[j + 1, i + 1, :3]
        pillar_bottoms[3] = coord[j + 1, i + 1, 3:]

        z_values = np.empty(8, dtype=np.float64)
        z_values[0] = zcorn[2 * k, 2 * j, 2 * i]
        z_values[1] = zcorn[2 * k, 2 * j, 2 * i + 1]
        z_values[2] = zcorn[2 * k, 2 * j + 1, 2 * i]
        z_values[3] = zcorn[2 * k, 2 * j + 1, 2 * i + 1]

        z_values[4] = zcorn[2 * k + 1, 2 * j, 2 * i]
        z_values[5] = zcorn[2 * k + 1, 2 * j, 2 * i + 1]
        z_values[6] = zcorn[2 * k + 1, 2 * j + 1, 2 * i]
        z_values[7] = zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1]

        for corner_index in range(8):
            pillar_index = pillar_order[corner_index]
            xyz = _interpolate_pillar_point(
                pillar_top=pillar_tops[pillar_index],
                pillar_bottom=pillar_bottoms[pillar_index],
                z=z_values[corner_index],
            )
            corner_coordinates[cell_index, corner_index, 0] = xyz[0]
            corner_coordinates[cell_index, corner_index, 1] = xyz[1]
            corner_coordinates[cell_index, corner_index, 2] = xyz[2]

    return corner_coordinates


def _compute_corner_point_geometry(
    coord: CoordArray,
    zcorn: ZcornArray,
    actnum: ActnumArray,
    vertex_tolerance: float = 1e-8,
) -> typing.Tuple[
    VertexCoordinates,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """
    Compute 3-D corner coordinates for every active cell.

    Each active cell `(i, j, k)` (0-indexed) has 8 corners obtained by
    interpolating its 4 pillars at the 8 ZCORN depth values. The pillars
    for cell `(i, j, k)` are at pillar indices
    `(j, i)`, `(j, i+1)`, `(j+1, i)`, `(j+1, i+1)` in the COORD
    array. Corner ZCORN indices follow the convention
    `zcorn[2k+top, 2j+near, 2i+left]` where top=0/1, near=0/1, left=0/1.

    :param coord: Shape `(NY+1, NX+1, 6)` pillar array.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array.
    :param actnum: Shape `(NZ, NY, NX)` active cell mask.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :returns: Tuple `(vertex_coordinates, per_cell_face_vertex_lists)`.
    """
    active_cells = np.argwhere(actnum > 0)
    if active_cells.size == 0:
        raise InvalidGridError(
            "No active cells found in the corner-point grid (ACTNUM is all zeros)."
        )

    corner_coordinates = _compute_active_cell_corner_coordinates(
        active_cells=active_cells,
        coord=coord,
        zcorn=zcorn,
    )
    # Deduplicate vertices
    flat_corner_coordinates = corner_coordinates.reshape(-1, 3)
    quantized_coordinates = np.round(flat_corner_coordinates / vertex_tolerance).astype(
        np.int64
    )
    (
        _,
        unique_vertex_indices,
        inverse_indices,
    ) = np.unique(
        quantized_coordinates,
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    vertex_coordinate_array = flat_corner_coordinates[unique_vertex_indices]
    corner_global_indices = inverse_indices.reshape(len(active_cells), 8)

    # Build cell face lists
    vtk_to_corner = np.array(
        [0, 1, 3, 2, 4, 5, 7, 6],
        dtype=np.int64,
    )
    hexahedron_face_table = ELEMENT_FACE_TABLES["hexahedron"]

    face_registry: typing.Dict[CanonicalFaceKey, _FaceRecord] = {}
    for cell_index in range(len(active_cells)):
        vtk_vertices = [
            int(
                corner_global_indices[
                    cell_index,
                    vtk_to_corner[local_corner],
                ]
            )
            for local_corner in range(8)
        ]
        for face_local in hexahedron_face_table:
            face_vertices = [vtk_vertices[v] for v in face_local]
            canonical_key = tuple(sorted(face_vertices))
            if canonical_key not in face_registry:
                face_registry[canonical_key] = _FaceRecord(
                    owner_cell_index=cell_index,
                    vertex_indices=face_vertices,
                )
            else:
                record = face_registry[canonical_key]
                if record.neighbour_cell_index != -1:
                    raise InvalidFaceConnectivityError(
                        f"Face {canonical_key} shared by more than two cells."
                    )
                record.neighbour_cell_index = cell_index

    flat_vertex_indices: list[int] = []
    face_vertex_offsets: list[int] = [0]
    face_cell_pairs: list[tuple[int, int]] = []
    for record in face_registry.values():
        flat_vertex_indices.extend(record.vertex_indices)
        face_vertex_offsets.append(len(flat_vertex_indices))
        face_cell_pairs.append((record.owner_cell_index, record.neighbour_cell_index))

    return (
        vertex_coordinate_array,
        np.asarray(flat_vertex_indices, dtype=np.int32),
        np.asarray(face_vertex_offsets, dtype=np.int32),
        np.asarray(face_cell_pairs, dtype=np.int32),
    )
