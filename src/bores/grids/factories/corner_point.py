import typing
import warnings

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from bores.errors import InvalidGridError, ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import (
    FaceKey,
    VertexCoordinates,
    _FaceRecord,
    assemble_grid,
)
from bores.typing import (
    FloatArray,
    IntArray,
    OneDimension,
    ThreeDimensions,
    UnitSystem,
)

__all__ = ["make_corner_point_grid"]


CoordArray: TypeAlias = FloatArray[ThreeDimensions]
"""Corner-point COORD array, shape `(NY+1, NX+1, 6)`."""

ZcornArray: TypeAlias = FloatArray[ThreeDimensions]
"""Corner-point ZCORN array, shape `(NZ*2, NY*2, NX*2)`."""

ActnumArray: TypeAlias = IntArray[ThreeDimensions]
"""Corner-point ACTNUM array, shape `(NZ, NY, NX)`; 1 = active, 0 = inactive."""


# Hexahedron face table correct for z-positive-downward (reservoir depth convention).
#
# VTK vertex layout after vtk_to_corner remapping:
#   vtk[0] = (x0, y0, z_top)   vtk[1] = (x1, y0, z_top)
#   vtk[2] = (x1, y1, z_top)   vtk[3] = (x0, y1, z_top)
#   vtk[4] = (x0, y0, z_bottom)   vtk[5] = (x1, y0, z_bottom)
#   vtk[6] = (x1, y1, z_bottom)   vtk[7] = (x0, y1, z_bottom)
#
# Face winding is CCW from outside (outward-normal convention).
# All normals verified by explicit cross-product check.
_HEXAHEDRON_FACES_ZDOWN: typing.List[typing.List[int]] = [
    [
        0,
        1,
        2,
        3,
    ],  # top    face (z_top, shallower) - outward normal = -z (toward surface)
    [
        4,
        7,
        6,
        5,
    ],  # bottom face (z_bottom, deeper)    - outward normal = +z (toward depth)
    [0, 1, 5, 4],  # -y face (near, y=y0)           - outward normal = -y
    [3, 7, 6, 2],  # +y face (far,  y=y1)           - outward normal = +y
    [0, 4, 7, 3],  # -x face (left, x=x0)           - outward normal = -x
    [1, 2, 6, 5],  # +x face (right,x=x1)           - outward normal = +x
]


def make_corner_point_grid(
    *,
    coord: CoordArray,
    zcorn: ZcornArray,
    actnum: typing.Optional[ActnumArray] = None,
    vertex_tolerance: float = 1e-8,
    pinch_tolerance: typing.Optional[float] = None,
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

    :param coord: Shape `(NY+1, NX+1, 6)` pillar array. Each entry
        contains `[x_top, y_top, z_top, x_bot, y_bot, z_bottom]` defining
        the top and bottom anchor points of a pillar.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array. For each
        cell `(i, j, k)` the 8 corner z-values are at indices
        `[2k:2k+2, 2j:2j+2, 2i:2i+2]` in `[top/bot, near/far, left/right]` order.
    :param actnum: Shape `(NZ, NY, NX)` integer mask (1=active, 0=inactive).
        If `None`, all cells are treated as active.
    :param vertex_tolerance: Two corner points closer than this distance
        (in grid units) are merged into a single vertex. Default 1e-8.
    :param pinch_tolerance: Maximum cell thickness (z_bottom - z_top averaged
        over all 4 pillars) below which a cell is considered pinched out and
        its top/bottom faces are suppressed, allowing transmissibility across
        the pinch. If `None`, pinch detection falls back to the value stored
        in ``metadata["pinch"]`` (set by the GRDECL reader), or defaults to
        0.0 (no pinch tolerance - only fully-collapsed cells are skipped).
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

    # Resolve pinch tolerance: explicit arg > metadata > default 0.0
    if pinch_tolerance is None:
        pinch_tolerance = float((metadata or {}).get("pinch", 0.0))

    vertex_coordinates, face_vertex_indices, face_vertex_offsets, face_cell_indices = (
        _compute_corner_point_geometry(
            coord=coord_arr,
            zcorn=zcorn_arr,
            actnum=actnum_arr,
            vertex_tolerance=vertex_tolerance,
            pinch_tolerance=pinch_tolerance,
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
    Interpolate an (x, y, z) position along a pillar at depth `z`.

    The pillar is a straight line from `pillar_top` to `pillar_bottom`.
    The interpolation parameter is linear in z.

    :param pillar_top: Shape `(3,)` array `[x, y, z]` of pillar top.
    :param pillar_bottom: Shape `(3,)` array `[x, y, z]` of pillar bottom.
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

    Corner layout in the returned array (index 0..7):
        0 -> pillar (j,   i  ), z = zcorn[2k,   2j,   2i  ]   ← top-near-left
        1 -> pillar (j,   i+1), z = zcorn[2k,   2j,   2i+1]   ← top-near-right
        2 -> pillar (j+1, i  ), z = zcorn[2k,   2j+1, 2i  ]   ← top-far-left
        3 -> pillar (j+1, i+1), z = zcorn[2k,   2j+1, 2i+1]   ← top-far-right
        4 -> pillar (j,   i  ), z = zcorn[2k+1, 2j,   2i  ]   ← bot-near-left
        5 -> pillar (j,   i+1), z = zcorn[2k+1, 2j,   2i+1]   ← bot-near-right
        6 -> pillar (j+1, i  ), z = zcorn[2k+1, 2j+1, 2i  ]   ← bot-far-left
        7 -> pillar (j+1, i+1), z = zcorn[2k+1, 2j+1, 2i+1]   ← bot-far-right

    :param active_cells: Shape `(n_active_cells, 3)` array of (k, j, i) indices.
    :param coord: Shape `(NY+1, NX+1, 6)` pillar array.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array.
    :returns: Shape `(n_active_cells, 8, 3)` corner coordinate array.
    """
    n_active_cells = active_cells.shape[0]
    corner_coordinates = np.empty((n_active_cells, 8, 3), dtype=np.float64)

    # Pillar index for each of the 8 corners (same 4 pillars, top then bottom)
    pillar_order = [0, 1, 2, 3, 0, 1, 2, 3]

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


def _is_cell_pinched(
    vtk_vertices: typing.List[int],
    vertex_coordinates: VertexCoordinates,
    pinch_tolerance: float,
) -> bool:
    """
    Return True if the cell's average thickness is at or below `pinch_tolerance`.

    Thickness is the mean of the 4 pillar z-differences (z_bottom - z_top).
    A cell whose top and bottom corners have already been merged to the same
    vertex indices (fully collapsed) always returns True regardless of tolerance.

    :param vtk_vertices: 8 global vertex indices in VTK hex order (top 0-3, bot 4-7).
    :param vertex_coordinates: Shape `(n_verts, 3)` world coordinate array.
    :param pinch_tolerance: Thickness threshold in grid length units.
    :returns: True if the cell should be treated as pinched out.
    """
    # Check if is fully collapsed (all top == all bottom after deduplication)
    top_set = set(vtk_vertices[:4])
    bottom_set = set(vtk_vertices[4:])
    if top_set == bottom_set:
        return True

    if pinch_tolerance <= 0.0:
        return False

    # Average thickness across the 4 pillar corners
    total_dz = 0.0
    for k in range(4):
        z_top = vertex_coordinates[vtk_vertices[k], 2]
        z_bottom = vertex_coordinates[vtk_vertices[k + 4], 2]
        total_dz += abs(z_bottom - z_top)
    avg_dz = total_dz / 4.0
    return avg_dz <= pinch_tolerance


def _compute_corner_point_geometry(
    coord: CoordArray,
    zcorn: ZcornArray,
    actnum: ActnumArray,
    vertex_tolerance: float = 1e-8,
    pinch_tolerance: float = 0.0,
) -> typing.Tuple[
    VertexCoordinates,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """
    Compute 3-D corner coordinates for every active cell and build face arrays.

    Handles pinchouts in two ways:

    1. **Fully collapsed cells** (top and bottom corners map to identical vertices
       after deduplication): all faces are skipped entirely. Neighbouring active
       cells on either side are connected via the shared lateral face that
       survived deduplication.

    2. **Partially collapsed cells** (average thickness ≤ `pinch_tolerance`):
       the top and bottom faces of the pinched cell are suppressed so that the
       cells above and below it can share those faces directly (non-neighbour
       connection across the pinch).

    In both cases, if a face ends up claimed by more than two cells (the
    classic symptom of an Eclipse pinchout), the third claim is silently
    dropped - the first two claimants (owner and neighbour) already represent
    the correct connection.

    :param coord: Shape `(NY+1, NX+1, 6)` pillar array.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array.
    :param actnum: Shape `(NZ, NY, NX)` active cell mask.
    :param vertex_tolerance: Vertex merge distance.
    :param pinch_tolerance: Average thickness threshold for pinch detection.
    :returns: Tuple `(vertex_coordinates, face_vertex_indices,
        face_vertex_offsets, face_cell_indices)`.
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

    # Vertex deduplication
    flat_corner_coordinates = corner_coordinates.reshape(-1, 3)
    quantized_coordinates = np.round(flat_corner_coordinates / vertex_tolerance).astype(
        np.int64
    )
    _, unique_vertex_indices, inverse_indices = np.unique(
        quantized_coordinates,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    vertex_coordinates = flat_corner_coordinates[unique_vertex_indices]
    n_active_cells = len(active_cells)
    corner_global_indices = inverse_indices.reshape(n_active_cells, 8)

    # VTK vertex reordering
    # Map from VTK hex local index -> our corner array index.
    # Our corners:  0=(x0,y0,zt) 1=(x1,y0,zt) 2=(x0,y1,zt) 3=(x1,y1,zt)
    #               4=(x0,y0,zb) 5=(x1,y0,zb) 6=(x0,y1,zb) 7=(x1,y1,zb)
    # VTK bottom:   v0=(x0,y0)   v1=(x1,y0)   v2=(x1,y1)   v3=(x0,y1)
    # VTK top:      v4=(x0,y0)   v5=(x1,y0)   v6=(x1,y1)   v7=(x0,y1)
    #
    # So VTK v0=corner0, v1=corner1, v2=corner3, v3=corner2 (note swap 2↔3)
    #        v4=corner4, v5=corner5, v6=corner7, v7=corner6 (note swap 6↔7)
    vtk_to_corner = [0, 1, 3, 2, 4, 5, 7, 6]

    # Face registry
    face_registry: typing.Dict[FaceKey, _FaceRecord] = {}
    n_pinched = 0
    n_degenerate_faces = 0
    n_extra_claims = 0

    # Indices of faces that are top/bottom for pinched cells.
    # These face keys should NOT be registered as boundary faces; instead they
    # allow a through-connection from the cell above to the cell below.
    # We achieve this by simply not adding them to the registry when the cell
    # is pinched - the cells on either side will register the same key normally.
    #
    # Face indices 0 and 1 in _HEXAHEDRON_FACES_ZDOWN are top and bottom.
    TOP_FACE_LOCAL = 0
    BOTTOM_FACE_LOCAL = 1

    for cell_index in range(n_active_cells):
        vtk_vertices = [
            int(corner_global_indices[cell_index, vtk_to_corner[v]]) for v in range(8)
        ]

        pinched = _is_cell_pinched(vtk_vertices, vertex_coordinates, pinch_tolerance)
        if pinched:
            n_pinched += 1

        for face_local_idx, face_local in enumerate(_HEXAHEDRON_FACES_ZDOWN):
            face_vertex_indices = [vtk_vertices[v] for v in face_local]

            # Skip degenerate faces (duplicate vertex indices -> zero area)
            if len(set(face_vertex_indices)) < len(face_vertex_indices):
                n_degenerate_faces += 1
                continue

            # For pinched cells, suppress top and bottom faces so that the
            # cells above and below can connect directly through this layer.
            if pinched and face_local_idx in (TOP_FACE_LOCAL, BOTTOM_FACE_LOCAL):
                continue

            key: FaceKey = tuple(sorted(face_vertex_indices))
            if key not in face_registry:
                face_registry[key] = _FaceRecord(
                    owner_cell_index=cell_index,
                    face_vertex_indices=face_vertex_indices,
                )
            elif face_registry[key].neighbour_cell_index == -1:
                face_registry[key].neighbour_cell_index = cell_index
            else:
                # Third claim: pinchout NNC - drop silently.
                # The owner/neighbour pair already models the correct connection.
                n_extra_claims += 1

    if n_pinched > 0:
        warnings.warn(
            f"{n_pinched} pinched-out cell(s) detected "
            f"(pinch_tolerance={pinch_tolerance:.3g}). "
            f"Their top/bottom faces have been suppressed to enable "
            f"transmissibility across the pinch.",
            stacklevel=4,
        )

    # Flatten to CSR arrays
    flat_face_vertex_indices: typing.List[int] = []
    face_vertex_offsets: typing.List[int] = [0]
    face_cell_pairs: typing.List[typing.Tuple[int, int]] = []

    for record in face_registry.values():
        flat_face_vertex_indices.extend(record.face_vertex_indices)
        face_vertex_offsets.append(len(flat_face_vertex_indices))
        face_cell_pairs.append((record.owner_cell_index, record.neighbour_cell_index))

    return (
        vertex_coordinates,
        np.asarray(flat_face_vertex_indices, dtype=np.int32),
        np.asarray(face_vertex_offsets, dtype=np.int32),
        np.asarray(face_cell_pairs, dtype=np.int32),
    )
