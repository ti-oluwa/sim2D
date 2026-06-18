"""
Corner-point (pillar) grid factory.

Builds a `bores.grids.base.Grid` from ECLIPSE-style COORD / ZCORN / ACTNUM
arrays as produced by `bores.grids.io.grdecl`.

**Coordinate convention**: z-axis positive downward (reservoir depth).

**Face winding**: vertices are wound counter-clockwise when viewed from the
**owner** cell (`face_cell_indices[:, 0]`), so the Newell normal points
from owner toward neighbour (outward for the owner).

**Pinchout handling**:

A cell is considered pinched out when its average pillar-to-pillar thickness
is at or below `pinch_tolerance`.  For such cells:

- Their top and bottom faces are suppressed so that the cells immediately
  above and below can share those face keys directly — forming a
  non-neighbour connection (NNC) across the collapsed layer.
- The surviving lateral face that bridges the pinch is tagged
  `ConnectionType.PINCHOUT`.
- A third claimant on any face key (the classic symptom of a pinchout in
  Eclipse's representation) is recorded as an explicit NNC pair rather than
  silently discarded.

**Fault handling**:

Named faults passed in as `fault_records` (a sequence of
`FaultRecord` objects parsed from the GRDECL `FAULTS` keyword) are
resolved to face indices after the unstructured grid is built.  Each
`FaultRecord` specifies an IJK range and a face direction.  The factory
maps the structured cell pairs implied by that range to unstructured face
indices via a reverse cell-index lookup table.  Resolved fault face arrays
and MULTFLT multipliers are stored directly on the returned `Grid`.
"""

import typing
import warnings
from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt

from bores.errors import GridExportError, InvalidGridError, ValidationError
from bores.grids.base import (
    CellStatus,
    ConnectionType,
    FaceStatus,
    Grid,
)
from bores.grids.factories.base import (
    FaceKey,
    VertexCoordinates,
    _FaceRecord,
)
from bores.typing import (
    FloatArray,
    IntArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["make_corner_point_grid", "FaultRecord"]


CoordArray: typing.TypeAlias = FloatArray[ThreeDimensions]
"""Corner-point COORD array, shape `(NY+1, NX+1, 6)`."""

ZcornArray: typing.TypeAlias = FloatArray[ThreeDimensions]
"""Corner-point ZCORN array, shape `(NZ*2, NY*2, NX*2)`."""

ActnumArray: typing.TypeAlias = IntArray[ThreeDimensions]
"""Corner-point ACTNUM array, shape `(NZ, NY, NX)`; 1 = active, 0 = inactive."""


########################################################################
# Hexahedron face table for z-positive-downward reservoir convention.
#
# VTK vertex layout after vtk_to_corner remapping:
#   vtk[0] = (x0, y0, z_top)    vtk[1] = (x1, y0, z_top)
#   vtk[2] = (x1, y1, z_top)    vtk[3] = (x0, y1, z_top)
#   vtk[4] = (x0, y0, z_bottom) vtk[5] = (x1, y0, z_bottom)
#   vtk[6] = (x1, y1, z_bottom) vtk[7] = (x0, y1, z_bottom)
#
# Winding is CCW from outside (outward-normal convention).
########################################################################
_HEXAHEDRON_FACES_ZDOWN: typing.List[typing.List[int]] = [
    [0, 1, 2, 3],  # top    (z_top,    shallower) — outward normal = -z
    [4, 7, 6, 5],  # bottom (z_bottom, deeper)    — outward normal = +z
    [0, 1, 5, 4],  # -y face (near, y=y0)         — outward normal = -y
    [3, 7, 6, 2],  # +y face (far,  y=y1)         — outward normal = +y
    [0, 4, 7, 3],  # -x face (left, x=x0)         — outward normal = -x
    [1, 2, 6, 5],  # +x face (right,x=x1)         — outward normal = +x
]

# Local face indices for the top and bottom faces in _HEXAHEDRON_FACES_ZDOWN.
_TOP_FACE_LOCAL: int = 0
_BOTTOM_FACE_LOCAL: int = 1

# Local face indices for lateral faces and their Eclipse face-direction labels.
# Used for fault face resolution.
# Face direction -> local face index in _HEXAHEDRON_FACES_ZDOWN.
_FACE_DIR_TO_LOCAL: typing.Dict[str, int] = {
    "I": 5,  # +x face between cell(i,j,k) and cell(i+1,j,k)
    "I-": 4,  # -x face (same physical face, opposite direction)
    "J": 3,  # +y face between cell(i,j,k) and cell(i,j+1,k)
    "J-": 2,  # -y face
    "K": 1,  # +z (bottom) face between cell(i,j,k) and cell(i,j,k+1)
    "K-": 0,  # -z (top) face
}


@dataclass(frozen=True)
class FaultRecord:
    """
    One record from the GRDECL `FAULTS` keyword.

    Each record declares a named fault plane over a rectangular IJK range
    and a face direction.

    :param name: Fault name (case-sensitive as written in the GRDECL file).
    :param i1: 1-based start index in the I (x) direction (inclusive).
    :param i2: 1-based end index in the I direction (inclusive).
    :param j1: 1-based start index in the J (y) direction (inclusive).
    :param j2: 1-based end index in the J direction (inclusive).
    :param k1: 1-based start index in the K (z) direction (inclusive).
    :param k2: 1-based end index in the K direction (inclusive).
    :param face_dir: Eclipse face direction string, one of
        `'I'`, `'I-'`, `'J'`, `'J-'`, `'K'`, `'K-'`.
    """

    name: str
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    face_dir: str


def make_corner_point_grid(
    *,
    coord: CoordArray,
    zcorn: ZcornArray,
    actnum: typing.Optional[ActnumArray] = None,
    vertex_tolerance: float = 1e-8,
    pinch_tolerance: typing.Optional[float] = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    # NNC data from the GRDECL NNC keyword
    nnc_cell_pairs: typing.Optional[IntArray[TwoDimensions]] = None,
    nnc_transmissibilities: typing.Optional[FloatArray[OneDimension]] = None,
    # Fault data from the GRDECL FAULTS / MULTFLT keywords
    fault_records: typing.Optional[typing.Sequence[FaultRecord]] = None,
    fault_transmissibility_multipliers: typing.Optional[
        typing.Mapping[str, float]
    ] = None,
) -> Grid:
    """
    Build a corner-point (pillar) grid from ECLIPSE-style COORD / ZCORN /
    ACTNUM arrays.

    Corner-point grids define cell geometry via pillar lines (COORD) and
    corner depths (ZCORN). Each cell is bounded by 4 pillars and has 8
    corner points obtained by intersecting depth planes with the pillars.
    This is the standard representation for GRDECL / ECLIPSE / ResInsight
    files.

    :param coord: Shape `(NY+1, NX+1, 6)` pillar array. Each entry
        contains `[x_top, y_top, z_top, x_bot, y_bot, z_bottom]` defining
        the top and bottom anchor points of a pillar.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array. For each
        cell `(i, j, k)` the 8 corner z-values are at indices
        `[2k:2k+2, 2j:2j+2, 2i:2i+2]` in `[top/bot, near/far,
        left/right]` order.
    :param actnum: Shape `(NZ, NY, NX)` integer mask (1 = active,
        0 = inactive). If `None`, all cells are treated as active.
    :param vertex_tolerance: Two corner points closer than this distance
        (in grid units) are merged into a single vertex.  Default 1e-8.
    :param pinch_tolerance: Maximum average cell thickness below which a
        cell is treated as pinched out and its top / bottom faces are
        suppressed to allow transmissibility across the pinch.  If `None`,
        the value stored in `metadata["pinch"]` is used, or 0.0 (no
        pinch detection) if that is also absent.
    :param unit_system: Declared unit system for all coordinate arrays.
    :param metadata: Optional free-form metadata dictionary attached to the
        returned `Grid`.
    :param nnc_cell_pairs: Shape `(n_nnc, 2)` int32 array of 0-based
        cell index pairs for explicit NNCs from the GRDECL `NNC` keyword.
        These are merged with any pinchout-detected NNCs from geometry.
    :param nnc_transmissibilities: Shape `(n_nnc,)` float64 array of
        transmissibilities corresponding to `nnc_cell_pairs`.  Must have
        the same length as `nnc_cell_pairs` if provided.
    :param fault_records: Sequence of `FaultRecord` objects parsed from
        the GRDECL `FAULTS` keyword. The factory resolves each record to
        a set of unstructured face indices and tags them
        `ConnectionType.FAULT`.
    :param fault_transmissibility_multipliers: Mapping from fault name to
        multiplier value (from GRDECL `MULTFLT`). Stored verbatim on the
        returned `Grid`.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises ValidationError: If COORD or ZCORN shapes are inconsistent, or
        if NNC transmissibility length mismatches NNC pair count.
    :raises InvalidGridError: If no active cells are found (ACTNUM all zeros).
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

    # Resolve pinch tolerance: explicit arg > metadata > 0.0 (no pinching)
    if pinch_tolerance is None:
        pinch_tolerance = float((metadata or {}).get("pinch", 0.0))

    # Validate NNC transmissibility length consistency up front
    if nnc_cell_pairs is not None and nnc_transmissibilities is not None:
        if len(nnc_cell_pairs) != len(nnc_transmissibilities):
            raise ValidationError(
                f"nnc_cell_pairs has {len(nnc_cell_pairs)} rows but "
                f"nnc_transmissibilities has {len(nnc_transmissibilities)} entries; "
                "they must have the same length."
            )

    # Build geometry and face arrays
    (
        vertex_coordinates,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
        connection_types_arr,
        geom_nnc_pairs,
        active_cells,  # shape (n_active, 3) — (k, j, i) for fault resolution
    ) = _compute_corner_point_geometry(
        coord=coord_arr,
        zcorn=zcorn_arr,
        actnum=actnum_arr,
        vertex_tolerance=vertex_tolerance,
        pinch_tolerance=pinch_tolerance,
    )

    n_active_cells = len(active_cells)

    # cell_statuses: all stored cells are active
    cell_statuses_arr = np.full(n_active_cells, int(CellStatus.ACTIVE), dtype=np.int8)

    # Merge geometry-detected NNCs with caller-supplied NNCs
    pair_parts = [p for p in (geom_nnc_pairs, nnc_cell_pairs) if p is not None]
    if pair_parts:
        merged_nnc_pairs = (
            np.vstack(pair_parts).astype(np.int32)
            if len(pair_parts) > 1
            else pair_parts[0]
        )
        # Transmissibilities: geometry pinchouts don't carry T values;
        # only the caller-supplied NNCs may have them.
        # Align: geometric NNCs get NaN (unknown T), caller NNCs get their T.
        if nnc_transmissibilities is not None:
            n_geom = len(geom_nnc_pairs) if geom_nnc_pairs is not None else 0
            geom_trans = np.full(n_geom, np.nan, dtype=np.float64)
            merged_nnc_transmissibilities = np.concatenate(
                [geom_trans, np.asarray(nnc_transmissibilities, dtype=np.float64)]
            )
        else:
            merged_nnc_transmissibilities = None
    else:
        merged_nnc_pairs = None
        merged_nnc_transmissibilities = None

    # Fault face resolution
    resolved_fault_face_indices: typing.Optional[
        typing.Dict[str, npt.NDArray[np.int32]]
    ] = None

    if fault_records:
        resolved_fault_face_indices = _resolve_fault_face_indices(
            fault_records=fault_records,
            active_cells=active_cells,
            face_cell_indices=face_cell_indices,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        # Tag fault faces in connection_types_arr
        for face_idx_arr in resolved_fault_face_indices.values():
            connection_types_arr[face_idx_arr] = int(ConnectionType.FAULT)

    return Grid(
        vertex_coordinates=np.asarray(vertex_coordinates, dtype=np.float64),
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
        cell_statuses=cell_statuses_arr,
        connection_types=connection_types_arr,
        nnc_cell_pairs=merged_nnc_pairs,
        nnc_transmissibilities=merged_nnc_transmissibilities,
        fault_face_indices=resolved_fault_face_indices,
        fault_transmissibility_multipliers=(
            dict(fault_transmissibility_multipliers)
            if fault_transmissibility_multipliers is not None
            else None
        ),
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
    Interpolation is linear in z.

    :param pillar_top: Shape `(3,)` — `[x, y, z]` of pillar top.
    :param pillar_bottom: Shape `(3,)` — `[x, y, z]` of pillar bottom.
    :param z: Target depth.
    :returns: Shape `(3,)` — `[x, y, z]` on the pillar at depth `z`.
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
    active_cells: np.ndarray,
    coord: CoordArray,
    zcorn: ZcornArray,
) -> FloatArray[ThreeDimensions]:
    """
    Compute all active-cell corner coordinates.

    Corner layout in the returned array (index 0..7):

    ```md
    ===  =================  =========================
    Idx  Pillar             ZCORN index
    ===  =================  =========================
    0    (j,   i  )         zcorn[2k,   2j,   2i  ]
    1    (j,   i+1)         zcorn[2k,   2j,   2i+1]
    2    (j+1, i  )         zcorn[2k,   2j+1, 2i  ]
    3    (j+1, i+1)         zcorn[2k,   2j+1, 2i+1]
    4    (j,   i  )         zcorn[2k+1, 2j,   2i  ]
    5    (j,   i+1)         zcorn[2k+1, 2j,   2i+1]
    6    (j+1, i  )         zcorn[2k+1, 2j+1, 2i  ]
    7    (j+1, i+1)         zcorn[2k+1, 2j+1, 2i+1]
    ===  =================  =========================
    ```

    :param active_cells: Shape `(n_active, 3)` — `(k, j, i)` indices.
    :param coord: Shape `(NY+1, NX+1, 6)` pillar array.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array.
    :returns: Shape `(n_active, 8, 3)` corner coordinate array.
    """
    n_active = active_cells.shape[0]
    corners = np.empty((n_active, 8, 3), dtype=np.float64)
    pillar_order = [0, 1, 2, 3, 0, 1, 2, 3]

    for cell_idx in numba.prange(n_active):  # type: ignore
        k = active_cells[cell_idx, 0]
        j = active_cells[cell_idx, 1]
        i = active_cells[cell_idx, 2]

        pt = np.empty((4, 3), dtype=np.float64)
        pb = np.empty((4, 3), dtype=np.float64)
        pt[0] = coord[j, i, :3]
        pb[0] = coord[j, i, 3:]
        pt[1] = coord[j, i + 1, :3]
        pb[1] = coord[j, i + 1, 3:]
        pt[2] = coord[j + 1, i, :3]
        pb[2] = coord[j + 1, i, 3:]
        pt[3] = coord[j + 1, i + 1, :3]
        pb[3] = coord[j + 1, i + 1, 3:]

        zv = np.empty(8, dtype=np.float64)
        zv[0] = zcorn[2 * k, 2 * j, 2 * i]
        zv[1] = zcorn[2 * k, 2 * j, 2 * i + 1]
        zv[2] = zcorn[2 * k, 2 * j + 1, 2 * i]
        zv[3] = zcorn[2 * k, 2 * j + 1, 2 * i + 1]
        zv[4] = zcorn[2 * k + 1, 2 * j, 2 * i]
        zv[5] = zcorn[2 * k + 1, 2 * j, 2 * i + 1]
        zv[6] = zcorn[2 * k + 1, 2 * j + 1, 2 * i]
        zv[7] = zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1]

        for c in range(8):
            p = pillar_order[c]
            xyz = _interpolate_pillar_point(pt[p], pb[p], zv[c])
            corners[cell_idx, c, 0] = xyz[0]
            corners[cell_idx, c, 1] = xyz[1]
            corners[cell_idx, c, 2] = xyz[2]

    return corners


def _is_cell_pinched(
    vtk_vertices: typing.List[int],
    vertex_coordinates: VertexCoordinates,
    pinch_tolerance: float,
) -> bool:
    """
    Return `True` if the cell's average thickness is at or below
    `pinch_tolerance`.

    A cell whose top and bottom corners already share the same vertex
    indices (fully collapsed after deduplication) always returns `True`
    regardless of tolerance.

    :param vtk_vertices: 8 global vertex indices in VTK hex order
        (top 0-3, bottom 4-7).
    :param vertex_coordinates: Shape `(n_verts, 3)` world coordinate
        array.
    :param pinch_tolerance: Thickness threshold in grid length units.
    :returns: `True` if the cell should be treated as pinched out.
    """
    top_set = set(vtk_vertices[:4])
    bottom_set = set(vtk_vertices[4:])
    if top_set == bottom_set:
        return True
    if pinch_tolerance <= 0.0:
        return False
    total_dz = 0.0
    for k in range(4):
        z_top = vertex_coordinates[vtk_vertices[k], 2]
        z_bottom = vertex_coordinates[vtk_vertices[k + 4], 2]
        total_dz += abs(z_bottom - z_top)
    return (total_dz / 4.0) <= pinch_tolerance


def _compute_corner_point_geometry(
    coord: CoordArray,
    zcorn: ZcornArray,
    actnum: ActnumArray,
    vertex_tolerance: float = 1e-8,
    pinch_tolerance: float = 0.0,
) -> typing.Tuple[
    VertexCoordinates,  # vertex_coordinates
    npt.NDArray[np.int32],  # face_vertex_indices  (flat CSR)
    npt.NDArray[np.int32],  # face_vertex_offsets  (CSR offsets)
    npt.NDArray[np.int32],  # face_cell_indices    (n_faces, 2)
    npt.NDArray[np.int8],  # connection_types     (n_faces,)
    typing.Optional[npt.NDArray[np.int32]],  # nnc_cell_pairs (n_nnc, 2)
    npt.NDArray[np.int32],  # active_cells  (n_active, 3) k,j,i
]:
    """
    Compute 3-D corner coordinates and build face arrays for a corner-point grid.

    Handles pinchouts in two ways:

    1. **Fully collapsed cells** (top and bottom corners map to identical
       vertices after deduplication): all faces are skipped entirely.
       Neighbouring active cells on either side are connected via the shared
       lateral face that survived deduplication.

    2. **Partially collapsed cells** (average thickness ≤ `pinch_tolerance`):
       top and bottom faces are suppressed. The first two cells that claim
       the surviving lateral face become owner and neighbour; any third
       claimant is recorded as an explicit NNC pair rather than discarded.

    :param coord: Shape `(NY+1, NX+1, 6)` pillar array.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array.
    :param actnum: Shape `(NZ, NY, NX)` active cell mask.
    :param vertex_tolerance: Vertex merge distance.
    :param pinch_tolerance: Average thickness threshold for pinch detection.
    :returns: 7-tuple — see return type annotation.
    :raises InvalidGridError: If no active cells are found.
    """
    active_cells = np.argwhere(actnum > 0).astype(np.int32)
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
    flat_corners = corner_coordinates.reshape(-1, 3)
    quantized = np.round(flat_corners / vertex_tolerance).astype(np.int64)
    _, unique_indices, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )
    vertex_coordinates = flat_corners[unique_indices]
    n_active = len(active_cells)
    corner_global = inverse.reshape(n_active, 8)

    # VTK vertex reordering
    # Our corners:  0=(x0,y0,zt) 1=(x1,y0,zt) 2=(x0,y1,zt) 3=(x1,y1,zt)
    #               4=(x0,y0,zb) 5=(x1,y0,zb) 6=(x0,y1,zb) 7=(x1,y1,zb)
    # VTK hex:      v0=c0  v1=c1  v2=c3  v3=c2  (note swap 2↔3)
    #               v4=c4  v5=c5  v6=c7  v7=c6  (note swap 6↔7)
    vtk_to_corner = [0, 1, 3, 2, 4, 5, 7, 6]

    # Face registry
    face_registry: typing.Dict[FaceKey, _FaceRecord] = {}
    nnc_pairs: typing.List[typing.Tuple[int, int]] = []
    # Track which face keys are pinchout-bridging NNCs
    nnc_face_keys: typing.Set[FaceKey] = set()

    n_pinched = 0
    n_degenerate = 0

    for cell_idx in range(n_active):
        vtk_verts = [int(corner_global[cell_idx, vtk_to_corner[v]]) for v in range(8)]
        pinched = _is_cell_pinched(vtk_verts, vertex_coordinates, pinch_tolerance)
        if pinched:
            n_pinched += 1

        for local_idx, local_face in enumerate(_HEXAHEDRON_FACES_ZDOWN):
            fvi = [vtk_verts[v] for v in local_face]

            # Skip degenerate faces (repeated vertices → zero area)
            if len(set(fvi)) < len(fvi):
                n_degenerate += 1
                continue

            # Suppress top / bottom faces on pinched cells so that the
            # cells above and below can claim those face keys directly.
            if pinched and local_idx in (_TOP_FACE_LOCAL, _BOTTOM_FACE_LOCAL):
                continue

            key: FaceKey = tuple(sorted(fvi))
            if key not in face_registry:
                face_registry[key] = _FaceRecord(
                    owner_cell_index=cell_idx,
                    face_vertex_indices=fvi,
                )
            elif face_registry[key].neighbour_cell_index == -1:
                face_registry[key].neighbour_cell_index = cell_idx
            else:
                # Third claimant: pinchout NNC.
                # The owner/neighbour pair already bridges the standard
                # connection; this extra pair is an explicit NNC.
                existing = face_registry[key]
                nnc_pairs.append((existing.owner_cell_index, cell_idx))
                nnc_face_keys.add(key)

    if n_pinched > 0:
        warnings.warn(
            f"{n_pinched} pinched-out cell(s) detected "
            f"(pinch_tolerance={pinch_tolerance:.3g}). "
            f"Their top/bottom faces have been suppressed to enable "
            f"transmissibility across the pinch.  "
            f"{len(nnc_pairs)} explicit NNC pair(s) were recorded.",
            stacklevel=4,
        )

    # Flatten face registry to CSR arrays + build connection_types
    flat_fvi: typing.List[int] = []
    face_vertex_offsets: typing.List[int] = [0]
    face_cell_pairs: typing.List[typing.Tuple[int, int]] = []
    connection_type_list: typing.List[int] = []

    for key, record in face_registry.items():
        flat_fvi.extend(record.face_vertex_indices)
        face_vertex_offsets.append(len(flat_fvi))
        face_cell_pairs.append((record.owner_cell_index, record.neighbour_cell_index))
        if key in nnc_face_keys:
            connection_type_list.append(int(ConnectionType.PINCHOUT))
        elif record.neighbour_cell_index < 0:
            connection_type_list.append(int(ConnectionType.BOUNDARY))
        else:
            connection_type_list.append(int(ConnectionType.INTERIOR))

    nnc_array: typing.Optional[npt.NDArray[np.int32]] = (
        np.asarray(nnc_pairs, dtype=np.int32).reshape(-1, 2) if nnc_pairs else None
    )
    return (
        vertex_coordinates,
        np.asarray(flat_fvi, dtype=np.int32),
        np.asarray(face_vertex_offsets, dtype=np.int32),
        np.asarray(face_cell_pairs, dtype=np.int32),
        np.asarray(connection_type_list, dtype=np.int8),
        nnc_array,
        active_cells,
    )


def _resolve_fault_face_indices(
    fault_records: typing.Sequence[FaultRecord],
    active_cells: npt.NDArray[np.int32],
    face_cell_indices: npt.NDArray[np.int32],
    nx: int,
    ny: int,
    nz: int,
) -> typing.Dict[str, npt.NDArray[np.int32]]:
    """
    Resolve `FaultRecord` IJK ranges to unstructured face index arrays.

    For each fault record, iterates over the declared IJK range and
    face direction to identify the structured cell pair
    `(cell_a, cell_b)` on either side of the fault plane.  Those pairs
    are mapped to face indices using a cell-pair → face-index lookup
    table built from `face_cell_indices`.

    Faces that cannot be resolved (e.g. one or both cells are inactive,
    or the cell pair shares no face in the unstructured grid) are silently
    skipped with a warning.

    :param fault_records: Sequence of `FaultRecord` objects.
    :param active_cells: Shape `(n_active, 3)` — `(k, j, i)` for each
        active cell, indexed 0 .. n_active-1.
    :param face_cell_indices: Shape `(n_faces, 2)` — owner/neighbour
        pairs in the unstructured grid.
    :param nx: Structured grid dimension in x.
    :param ny: Structured grid dimension in y.
    :param nz: Structured grid dimension in z.
    :returns: Dict mapping fault name → 1-D int32 array of face indices.
    """
    # Build: structured (k, j, i) -> unstructured cell index
    kji_to_cell: typing.Dict[typing.Tuple[int, int, int], int] = {}
    for cell_idx, (k, j, i) in enumerate(active_cells):
        kji_to_cell[(int(k), int(j), int(i))] = cell_idx

    # Build: frozenset{owner, neighbour} -> face index
    # Use frozenset so lookup is order-independent.
    cell_pair_to_face: typing.Dict[typing.FrozenSet[int], int] = {}
    for face_idx, (owner, neighbour) in enumerate(face_cell_indices):
        if owner >= 0 and neighbour >= 0:
            cell_pair_to_face[frozenset((int(owner), int(neighbour)))] = face_idx

    result: typing.Dict[str, typing.List[int]] = {}

    for rec in fault_records:
        face_dir = rec.face_dir.upper()
        if face_dir not in _FACE_DIR_TO_LOCAL:
            warnings.warn(
                f"Fault {rec.name!r}: unrecognised face direction {rec.face_dir!r}. "
                f"Valid directions: {sorted(_FACE_DIR_TO_LOCAL)}. Skipping.",
                stacklevel=4,
            )
            continue

        # Convert FAULTS direction to the offset of the neighbour cell
        # relative to the owner cell.
        # I / I-  → neighbour is at i±1, same j, k
        # J / J-  → neighbour is at j±1, same i, k
        # K / K-  → neighbour is at k±1, same i, j
        if face_dir in ("I", "I-"):
            di, dj, dk = 1, 0, 0
        elif face_dir in ("J", "J-"):
            di, dj, dk = 0, 1, 0
        else:  # K, K-
            di, dj, dk = 0, 0, 1

        face_indices: typing.List[int] = []
        n_missed = 0

        # IJK ranges are 1-based and inclusive in Eclipse convention
        for k in range(rec.k1 - 1, rec.k2):
            for j in range(rec.j1 - 1, rec.j2):
                for i in range(rec.i1 - 1, rec.i2):
                    # Cell A (the cell at (k, j, i))
                    cell_a = kji_to_cell.get((k, j, i))
                    # Cell B (the neighbour across the fault plane)
                    nb_k, nb_j, nb_i = k + dk, j + dj, i + di
                    cell_b = kji_to_cell.get((nb_k, nb_j, nb_i))

                    if cell_a is None or cell_b is None:
                        n_missed += 1
                        continue

                    face_idx = cell_pair_to_face.get(frozenset((cell_a, cell_b)))
                    if face_idx is None:
                        n_missed += 1
                        continue

                    face_indices.append(face_idx)

        if n_missed > 0:
            warnings.warn(
                f"Fault {rec.name!r}: {n_missed} cell pair(s) in the IJK range "
                f"I=[{rec.i1},{rec.i2}] J=[{rec.j1},{rec.j2}] K=[{rec.k1},{rec.k2}] "
                f"could not be resolved to face indices (inactive cells or no shared "
                f"face). These pairs are skipped.",
                stacklevel=4,
            )

        if face_indices:
            existing = result.get(rec.name)
            if existing is not None:
                existing.extend(face_indices)
            else:
                result[rec.name] = face_indices
        elif n_missed == 0:
            warnings.warn(
                f"Fault {rec.name!r} produced no resolvable face indices.",
                stacklevel=4,
            )

    # Deduplicate and convert to numpy arrays
    return {
        name: np.unique(np.asarray(idxs, dtype=np.int32))
        for name, idxs in result.items()
    }


@numba.njit(cache=True)
def _accumulate_pillars(
    cell_min_xyz: FloatArray[TwoDimensions],
    cell_max_xyz: FloatArray[TwoDimensions],
    nx: int,
    ny: int,
    nz: int,
    pillar_x: FloatArray[TwoDimensions],
    pillar_y: FloatArray[TwoDimensions],
    pillar_z_top: FloatArray[TwoDimensions],
    pillar_z_bottom: FloatArray[TwoDimensions],
    pillar_count: IntArray[TwoDimensions],
) -> None:
    """
    Accumulate per-pillar XY positions and Z extents from cell bounding boxes.

    Cell ordering: `cell_idx = i + j*nx + k*nx*ny`.
    Pillar `(pj, pi)` collects contributions from up to 4 cells per layer.

    :param cell_min_xyz: Shape `(n_cells, 3)` bounding-box minima.
    :param cell_max_xyz: Shape `(n_cells, 3)` bounding-box maxima.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param pillar_x: Accumulator for pillar X (zeroed on entry).
    :param pillar_y: Accumulator for pillar Y (zeroed on entry).
    :param pillar_z_top: Accumulator for minimum pillar Z (`+inf` on entry).
    :param pillar_z_bottom: Accumulator for maximum pillar Z (`-inf` on entry).
    :param pillar_count: Contribution counter per pillar (zeroed on entry).
    """
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell_idx = i + j * nx + k * nx * ny
                lx = cell_min_xyz[cell_idx, 0]
                ly = cell_min_xyz[cell_idx, 1]
                lz = cell_min_xyz[cell_idx, 2]
                hx = cell_max_xyz[cell_idx, 0]
                hy = cell_max_xyz[cell_idx, 1]
                hz = cell_max_xyz[cell_idx, 2]

                for corner in range(4):
                    if corner == 0:
                        pj, pi, px, py = j, i, lx, ly
                    elif corner == 1:
                        pj, pi, px, py = j, i + 1, hx, ly
                    elif corner == 2:
                        pj, pi, px, py = j + 1, i, lx, hy
                    else:
                        pj, pi, px, py = j + 1, i + 1, hx, hy

                    pillar_x[pj, pi] += px
                    pillar_y[pj, pi] += py
                    if lz < pillar_z_top[pj, pi]:
                        pillar_z_top[pj, pi] = lz
                    if hz > pillar_z_bottom[pj, pi]:
                        pillar_z_bottom[pj, pi] = hz
                    pillar_count[pj, pi] += 1


@numba.njit(parallel=True, cache=True)
def _fill_zcorn(
    cell_min_xyz: FloatArray[TwoDimensions],
    cell_max_xyz: FloatArray[TwoDimensions],
    nx: int,
    ny: int,
    nz: int,
    zcorn: ZcornArray,
) -> None:
    """
    Fill ZCORN array from per-cell Z bounding-box extents.

    Parallel over `k` (layer index); each layer writes to a disjoint
    `[2k:2k+2, :, :]` slice of `zcorn` so there are no data races.

    ZCORN indexing (Eclipse convention)::

        zcorn[2k,   2j,   2i  ] ... [2k,   2j+1, 2i+1] = top    Z of (i,j,k)
        zcorn[2k+1, 2j,   2i  ] ... [2k+1, 2j+1, 2i+1] = bottom Z of (i,j,k)

    :param cell_min_xyz: Shape `(n_cells, 3)` bounding-box minima.
    :param cell_max_xyz: Shape `(n_cells, 3)` bounding-box maxima.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param zcorn: Output array, pre-allocated as `(nz*2, ny*2, nx*2)`.
    """
    for k in numba.prange(nz):  # type: ignore
        for j in range(ny):
            for i in range(nx):
                cell_idx = i + j * nx + k * nx * ny
                z_top = cell_min_xyz[cell_idx, 2]
                z_bottom = cell_max_xyz[cell_idx, 2]
                zcorn[2 * k, 2 * j, 2 * i] = z_top
                zcorn[2 * k, 2 * j, 2 * i + 1] = z_top
                zcorn[2 * k, 2 * j + 1, 2 * i] = z_top
                zcorn[2 * k, 2 * j + 1, 2 * i + 1] = z_top
                zcorn[2 * k + 1, 2 * j, 2 * i] = z_bottom
                zcorn[2 * k + 1, 2 * j, 2 * i + 1] = z_bottom
                zcorn[2 * k + 1, 2 * j + 1, 2 * i] = z_bottom
                zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1] = z_bottom


def rederive_corner_point_arrays(
    grid: Grid,
) -> typing.Tuple[CoordArray, ZcornArray, int, int, int]:
    """
    Reconstruct approximate COORD and ZCORN arrays from a `Grid` whose
    geometry originates from a corner-point source.

    The reconstruction uses each cell's axis-aligned bounding box.  Pillars
    are assumed to be straight and vertical, so the result is lossy for
    grids with lateral pillar displacement (faults, dipping layers).  A
    warning is always emitted.

    Layout assumption: cells are stored in k-major, j-middle, i-minor order
    (Eclipse Fortran order), i.e. `cell_index = i + j*nx + k*nx*ny`.

    :param grid: A `Grid` whose `metadata["source_format"]` is `"grdecl_corner_point"`.
    :returns: Tuple `(coord_arr, zcorn_arr, nx, ny, nz)`.
    :raises GridExportError: If the cell count cannot be factored into a
        valid `nx x ny x nz` product.
    """
    n_cells = grid.n_cells
    meta = getattr(grid, "metadata", {}) or {}
    nx = meta.get("nx")
    ny = meta.get("ny")
    nz = meta.get("nz")

    if nx is None or ny is None or nz is None:
        found = False
        for nz_try in range(1, n_cells + 1):
            if n_cells % nz_try != 0:
                continue
            nxy = n_cells // nz_try
            for nx_try in range(1, int(nxy**0.5) + 1):
                if nxy % nx_try == 0:
                    nx, ny, nz = nx_try, nxy // nx_try, nz_try
                    found = True
            if found:
                break
        if not found or (nx * ny * nz) != n_cells:  # type: ignore
            raise GridExportError(
                f"Cannot determine (nx, ny, nz) factorisation for "
                f"n_cells={n_cells}.  Store 'nx', 'ny', 'nz' in "
                "grid.metadata to enable GRDECL export."
            )

    warnings.warn(
        "Exporting a corner-point Grid to GRDECL without stored COORD/ZCORN "
        "arrays. Pillars are reconstructed as straight vertical lines from "
        "cell bounding boxes. This is lossy for grids with lateral pillar "
        "displacement (faults, dipping layers).",
        stacklevel=3,
    )
    assert nx is not None and ny is not None and nz is not None

    pillar_x = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    pillar_y = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    pillar_z_top = np.full((ny + 1, nx + 1), np.inf, dtype=np.float64)
    pillar_z_bot = np.full((ny + 1, nx + 1), -np.inf, dtype=np.float64)
    pillar_count = np.zeros((ny + 1, nx + 1), dtype=np.int32)

    _accumulate_pillars(
        grid.cell_min_xyz,
        grid.cell_max_xyz,
        nx,
        ny,
        nz,
        pillar_x,
        pillar_y,
        pillar_z_top,
        pillar_z_bot,
        pillar_count,
    )
    nonzero = pillar_count > 0
    pillar_x[nonzero] /= pillar_count[nonzero]
    pillar_y[nonzero] /= pillar_count[nonzero]

    coord = np.empty((ny + 1, nx + 1, 6), dtype=np.float64)
    coord[:, :, 0] = pillar_x
    coord[:, :, 1] = pillar_y
    coord[:, :, 2] = pillar_z_top
    coord[:, :, 3] = pillar_x
    coord[:, :, 4] = pillar_y
    coord[:, :, 5] = pillar_z_bot

    zcorn = np.empty((nz * 2, ny * 2, nx * 2), dtype=np.float64)
    _fill_zcorn(grid.cell_min_xyz, grid.cell_max_xyz, nx, ny, nz, zcorn)

    return coord, zcorn, nx, ny, nz
