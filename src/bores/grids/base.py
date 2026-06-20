"""Face-based unstructured polyhedral grid implementation for reservoir simulation."""

import enum
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from bores.errors import (
    CellNotFoundError,
    InvalidFaceAreaError,
    InvalidFaceConnectivityError,
    InvalidNormalVectorError,
    InvalidPointArrayError,
    InvalidVolumeError,
    ValidationError,
)
from bores.typing import (
    FloatArray,
    IntArray,
    NumberOrArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["Grid", "ConnectionType", "FaceStatus", "CellStatus"]


class ConnectionType(enum.IntEnum):
    """
    Topological classification of a connection between two grid cells.

    Each face in the grid represents a connection.  The type governs how
    transmissibility is computed and how boundary conditions are applied.

    `INTERIOR`
        A standard shared face between two active cells.

    `BOUNDARY`
        A face on the outer hull of the domain; one side is the exterior
        (`neighbour_cell_index == -1`).

    `FAULT`
        An interior face that crosses a fault plane.  Transmissibility may
        be modified by a fault multiplier (see `Grid.fault_transmissibility_multipliers`).

    `NON_NEIGHBOR_CONNECTION`
        A connection between two cells that do not share a face in the
        geometric sense - typically across a pinched-out layer or an
        explicit NNC declared in the input file.  These connections are
        stored both as faces (when a surviving lateral face bridges the
        pinch) and as explicit NNC pairs in `Grid.nnc_cell_pairs`.

    `PINCHOUT`
        A face whose existence is due to a pinched-out layer being collapsed.
        Functionally equivalent to `NON_NEIGHBOR_CONNECTION` but marks
        specifically the surviving lateral face that now connects cells on
        either side of the collapsed layer.
    """

    INTERIOR = 0
    BOUNDARY = 1
    FAULT = 2
    PINCHOUT = 3
    NON_NEIGHBOR_CONNECTION = 4


class FaceStatus(enum.IntEnum):
    """
    Activation status of a grid face.

    `ACTIVE`
        The face transmits flow normally.

    `INACTIVE`
        The face is closed (e.g. a closed fault or a zero-multiplier face);
        transmissibility is treated as zero by the flow solver.
    """

    ACTIVE = 1
    INACTIVE = 0


class CellStatus(enum.IntEnum):
    """
    Activation status of a grid cell.

    `ACTIVE`
        The cell participates in flow simulation.

    `INACTIVE`
        The cell is excluded from flow simulation (e.g. ACTNUM == 0).
        Inactive cells are not stored in the `Grid` (they are filtered
        out during construction), so in practice `cell_statuses` for a
        fully constructed `Grid` will always be `ACTIVE` for every
        stored cell.  The field exists so that partial-activity sub-grids
        can be represented without re-indexing.
    """

    ACTIVE = 1
    INACTIVE = 0


# Absolute tolerance used in geometry validation.
_GEOMETRY_TOLERANCE: float = 1e-14


@numba.njit(parallel=True, cache=True)
def _compute_face_geometry(
    face_vertex_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    vertex_coordinates: FloatArray[TwoDimensions],
) -> typing.Tuple[
    FloatArray[TwoDimensions],
    FloatArray[OneDimension],
    FloatArray[TwoDimensions],
]:
    """
    Compute face centroids, areas, and unit outward normals via Newell's method.

    Uses Newell's method [Sutherland et al. 1974] which is robust for planar
    polygons with an arbitrary number of vertices and does not require a
    pre-computed face centroid.  The normal magnitude equals twice the face
    area, so `area = ||n|| / 2`.

    :param face_vertex_indices: Flat CSR data array of vertex indices (all faces).
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :returns: Tuple `(face_centroids, face_areas, face_unit_normals)` of
        shapes `(n_faces, 3)`, `(n_faces,)`, `(n_faces, 3)` respectively.
    """
    n_faces = face_vertex_offsets.shape[0] - 1
    face_centroids = np.zeros((n_faces, 3), dtype=np.float64)
    face_unit_normals = np.zeros((n_faces, 3), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)

    for face_idx in numba.prange(n_faces):  # type: ignore
        start = face_vertex_offsets[face_idx]
        end = face_vertex_offsets[face_idx + 1]
        n_verts = end - start

        cx = 0.0
        cy = 0.0
        cz = 0.0
        for local_idx in range(n_verts):
            vert_idx = face_vertex_indices[start + local_idx]
            cx += vertex_coordinates[vert_idx, 0]
            cy += vertex_coordinates[vert_idx, 1]
            cz += vertex_coordinates[vert_idx, 2]
        cx /= n_verts
        cy /= n_verts
        cz /= n_verts
        face_centroids[face_idx, 0] = cx
        face_centroids[face_idx, 1] = cy
        face_centroids[face_idx, 2] = cz

        nx = 0.0
        ny = 0.0
        nz = 0.0
        for local_idx in range(n_verts):
            a_idx = face_vertex_indices[start + local_idx]
            b_idx = face_vertex_indices[start + (local_idx + 1) % n_verts]

            ax = vertex_coordinates[a_idx, 0]
            ay = vertex_coordinates[a_idx, 1]
            az = vertex_coordinates[a_idx, 2]
            bx = vertex_coordinates[b_idx, 0]
            by = vertex_coordinates[b_idx, 1]
            bz = vertex_coordinates[b_idx, 2]

            nx += (ay - by) * (az + bz)
            ny += (az - bz) * (ax + bx)
            nz += (ax - bx) * (ay + by)

        normal_magnitude = np.sqrt(nx * nx + ny * ny + nz * nz)
        if normal_magnitude > 0.0:
            face_unit_normals[face_idx, 0] = nx / normal_magnitude
            face_unit_normals[face_idx, 1] = ny / normal_magnitude
            face_unit_normals[face_idx, 2] = nz / normal_magnitude
            face_areas[face_idx] = normal_magnitude * 0.5

    return face_centroids, face_areas, face_unit_normals


@numba.njit(cache=True)
def _compute_cell_volumes_and_centroids(
    face_cell_indices: IntArray[TwoDimensions],
    face_vertex_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    vertex_coordinates: FloatArray[TwoDimensions],
    n_cells: int,
) -> typing.Tuple[FloatArray[OneDimension], FloatArray[TwoDimensions]]:
    """
    Compute cell volumes and centroids via the divergence theorem.

    Decomposes every face into a fan of triangles anchored at the first face
    vertex, then accumulates signed tetrahedral contributions for each
    owner / neighbour cell.

    **Sign convention**: face vertices are wound counter-clockwise from the
    **owner** (c1) side, so the Newell normal points from c1 toward c2.
    The signed tet volume is positive for the owner and negative for the
    neighbour; both therefore accumulate positive contributions.

    :param face_cell_indices: Shape `(n_faces, 2)` - `(owner, neighbour)`
        per face.
    :param face_vertex_indices: Flat CSR vertex index data array.
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :param n_cells: Total number of cells in the grid.
    :returns: Tuple `(cell_volumes, cell_centroids)` of shapes
        `(n_cells,)` and `(n_cells, 3)`.
    """
    n_faces = face_cell_indices.shape[0]
    cell_volumes = np.zeros(n_cells, dtype=np.float64)
    centroid_accumulators = np.zeros((n_cells, 3), dtype=np.float64)

    for face_idx in range(n_faces):
        owner_cell = face_cell_indices[face_idx, 0]
        neighbour_cell = face_cell_indices[face_idx, 1]

        start = face_vertex_offsets[face_idx]
        end = face_vertex_offsets[face_idx + 1]

        apex = vertex_coordinates[face_vertex_indices[start]]

        for iteration in range(2):
            if iteration == 0:
                cell_idx = owner_cell
                sign = 1.0
            else:
                cell_idx = neighbour_cell
                sign = -1.0

            if cell_idx < 0:
                continue

            for fan_idx in range(start + 1, end - 1):
                v1 = vertex_coordinates[face_vertex_indices[fan_idx]]
                v2 = vertex_coordinates[face_vertex_indices[fan_idx + 1]]

                ax = apex[0]
                ay = apex[1]
                az = apex[2]
                bx = v1[0]
                by = v1[1]
                bz = v1[2]
                cx = v2[0]
                cy = v2[1]
                cz = v2[2]

                signed_tetrahedron_volume = (
                    ax * (by * cz - bz * cy)
                    + ay * (bz * cx - bx * cz)
                    + az * (bx * cy - by * cx)
                ) / 6.0
                cell_volumes[cell_idx] += sign * signed_tetrahedron_volume

                x_barycentre = (ax + bx + cx) / 4.0
                y_barycentre = (ay + by + cy) / 4.0
                z_barycentre = (az + bz + cz) / 4.0

                weighted_volume = sign * signed_tetrahedron_volume
                centroid_accumulators[cell_idx, 0] += weighted_volume * x_barycentre
                centroid_accumulators[cell_idx, 1] += weighted_volume * y_barycentre
                centroid_accumulators[cell_idx, 2] += weighted_volume * z_barycentre

    cell_centroids = np.zeros((n_cells, 3), dtype=np.float64)
    for cell_idx in range(n_cells):
        cell_volume = cell_volumes[cell_idx]
        if abs(cell_volume) > 0.0:
            cell_centroids[cell_idx, 0] = (
                centroid_accumulators[cell_idx, 0] / cell_volume
            )
            cell_centroids[cell_idx, 1] = (
                centroid_accumulators[cell_idx, 1] / cell_volume
            )
            cell_centroids[cell_idx, 2] = (
                centroid_accumulators[cell_idx, 2] / cell_volume
            )

    return cell_volumes, cell_centroids


@numba.njit(cache=True)
def _compute_cell_bounding_boxes(
    face_cell_indices: IntArray[TwoDimensions],
    face_vertex_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    vertex_coordinates: FloatArray[TwoDimensions],
    n_cells: int,
    dtype: npt.DTypeLike = np.float64,
) -> typing.Tuple[FloatArray[TwoDimensions], FloatArray[TwoDimensions]]:
    """
    Compute per-cell axis-aligned bounding boxes.

    :param face_cell_indices: Shape `(n_faces, 2)` owner/neighbour array.
    :param face_vertex_indices: Flat CSR vertex index data array.
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :param n_cells: Total number of cells.
    :param dtype: Floating dtype for the output arrays.
    :returns: Tuple `(cell_min_xyz, cell_max_xyz)` each of shape `(n_cells, 3)`.
    """
    cell_min = np.full((n_cells, 3), np.inf, dtype=dtype)
    cell_max = np.full((n_cells, 3), -np.inf, dtype=dtype)
    n_faces = face_cell_indices.shape[0]

    for face_idx in range(n_faces):
        owner = face_cell_indices[face_idx, 0]
        neighbour = face_cell_indices[face_idx, 1]

        start = face_vertex_offsets[face_idx]
        end = face_vertex_offsets[face_idx + 1]

        for i in range(start, end):
            vid = face_vertex_indices[i]

            vx = vertex_coordinates[vid, 0]
            vy = vertex_coordinates[vid, 1]
            vz = vertex_coordinates[vid, 2]

            if owner >= 0:
                if vx < cell_min[owner, 0]:
                    cell_min[owner, 0] = vx
                if vy < cell_min[owner, 1]:
                    cell_min[owner, 1] = vy
                if vz < cell_min[owner, 2]:
                    cell_min[owner, 2] = vz
                if vx > cell_max[owner, 0]:
                    cell_max[owner, 0] = vx
                if vy > cell_max[owner, 1]:
                    cell_max[owner, 1] = vy
                if vz > cell_max[owner, 2]:
                    cell_max[owner, 2] = vz

            if neighbour >= 0:
                if vx < cell_min[neighbour, 0]:
                    cell_min[neighbour, 0] = vx
                if vy < cell_min[neighbour, 1]:
                    cell_min[neighbour, 1] = vy
                if vz < cell_min[neighbour, 2]:
                    cell_min[neighbour, 2] = vz
                if vx > cell_max[neighbour, 0]:
                    cell_max[neighbour, 0] = vx
                if vy > cell_max[neighbour, 1]:
                    cell_max[neighbour, 1] = vy
                if vz > cell_max[neighbour, 2]:
                    cell_max[neighbour, 2] = vz

    return cell_min, cell_max


@attrs.define(frozen=True, slots=True, kw_only=True)
class Grid:
    """
    Immutable face-based unstructured polyhedral grid.

    All topology and geometry is computed once during construction and stored
    as read-only NumPy arrays. The grid is fully unstructured: cells may be
    arbitrary convex polyhedra; faces may be arbitrary planar polygons.

    **Construction**:

    Supply the three mandatory arrays and any optional metadata; all derived
    arrays (connectivity, geometry, bounding boxes) are computed automatically.

    **Coordinate convention**:

    The z-axis is positive **downward** (reservoir depth convention).

    **Non-neighbour connections (NNCs)**:

    Connections that cannot be expressed as a shared face - typically across
    pinched-out layers or declared via the GRDECL `NNC` keyword - are
    stored explicitly in `nnc_cell_pairs` and `nnc_transmissibilities`.
    The surviving lateral face that bridges a pinch is additionally present
    in the face arrays and tagged `ConnectionType.PINCHOUT`.

    **Fault transmissibility**:

    Named faults parsed from `FAULTS` / `MULTFLT` keywords are stored in
    `fault_face_indices` and `fault_transmissibility_multipliers`.  These
    are purely advisory; the flow solver must apply them when assembling the
    transmissibility matrix.

    **Raises**:

    `InvalidPointArrayError`
        If `vertex_coordinates` is not a 2-D `(n_vertices, 3)` array.
    `InvalidFaceConnectivityError`
        If `face_cell_indices` is not a 2-D `(n_faces, 2)` array, or if
        `face_vertex_offsets` does not start at 0 or is inconsistent with
        `face_vertex_indices`.
    `InvalidVolumeError`
        If any cell ends up with a non-positive volume after construction.
    """

    # Mandatory primary inputs

    vertex_coordinates: FloatArray[TwoDimensions]
    """
    Shape `(n_vertices, 3)` - world (x, y, z) coordinates of every vertex.

    The z-axis is positive downward (reservoir depth convention).
    """

    face_vertex_indices: IntArray[OneDimension]
    """
    Flat CSR data array: concatenated vertex index lists for all faces.

    Face *f* uses
    `face_vertex_indices[face_vertex_offsets[f]:face_vertex_offsets[f+1]]`.
    Vertices are wound counter-clockwise when viewed from the **owner** cell.
    """

    face_vertex_offsets: IntArray[OneDimension]
    """
    CSR offset array of length `n_faces + 1`.

    `face_vertex_offsets[0]` must be 0; `face_vertex_offsets[-1]` must
    equal `len(face_vertex_indices)`.
    """

    face_cell_indices: IntArray[TwoDimensions]
    """
    Shape `(n_faces, 2)` - `(owner_cell_index, neighbour_cell_index)`
    per face.

    Boundary faces have `neighbour_cell_index == -1`.  Interior faces have
    both indices >= 0.  The owner cell is the one from whose perspective the
    face vertices are wound counter-clockwise.
    """

    # Optional dtype / unit hints

    index_dtype: np.dtype = attrs.field(default=np.dtype(np.int32))
    """NumPy integer dtype used for all connectivity index arrays."""

    floating_dtype: np.dtype = attrs.field(default=np.dtype(np.float64))
    """NumPy floating-point dtype used for all coordinate and geometry arrays."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """
    Declared unit system for all coordinate and geometry arrays.

    Set by the factory or IO reader that built this grid.  Downstream
    physics code reads this field to perform any necessary unit conversions.
    """

    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = attrs.field(
        default=None
    )
    """Optional free-form metadata mapping (e.g. units, CRS, source filename)."""

    # Optional caller-supplied status / classification arrays

    cell_statuses: typing.Optional[IntArray[OneDimension]] = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell `CellStatus` flags.

    When `None` on construction, auto-populated to all
    `CellStatus.ACTIVE`.  Factories that honour ACTNUM should populate
    this explicitly; note that inactive cells are typically excluded from the
    grid entirely (they are not stored as cells), so this array will almost
    always be all-active for a fully constructed grid.
    """

    connection_types: typing.Optional[IntArray[OneDimension]] = attrs.field(
        default=None
    )
    """
    Shape `(n_faces,)` - per-face `ConnectionType` classification.

    When `None` on construction, auto-populated from topology:
    `BOUNDARY` for faces with `neighbour == -1`, `INTERIOR` for all
    others.  Factories that know about faults or pinchouts should pass an
    explicit array.
    """

    face_statuses: typing.Optional[IntArray[OneDimension]] = attrs.field(default=None)
    """
    Shape `(n_faces,)` - per-face `FaceStatus` flags.

    When `None` on construction, auto-populated to all
    `FaceStatus.ACTIVE`.  Set faces to `FaceStatus.INACTIVE` to model
    closed faults or zero-transmissibility connections.
    """

    # NNC (non-neighbour connections)

    nnc_cell_pairs: typing.Optional[IntArray[TwoDimensions]] = attrs.field(default=None)
    """
    Shape `(n_nnc, 2)` - non-neighbour connection (NNC) cell index pairs.

    Each row `(cell_a, cell_b)` identifies two cells that are connected
    across a pinchout, fault, or other non-standard topology that has no
    corresponding shared face. Both indices are >= 0.

    Sources:

    - Pinchout geometry: the corner-point factory emits these when a pinched
      layer creates a third claimant for a face key.
    - GRDECL `NNC` keyword: explicit NNCs declared in the input file.

    `None` when no NNCs are present.
    """

    nnc_transmissibilities: typing.Optional[FloatArray[OneDimension]] = attrs.field(
        default=None
    )
    """
    Shape `(n_nnc,)` - transmissibility for each NNC pair in
    `nnc_cell_pairs`.

    Units follow the grid's declared `unit_system` (e.g. md·ft/cP in
    FIELD, md·m/cP in METRIC). `None` when transmissibilities were not
    supplied - physics code must compute them from geometry in that case.
    """

    # Fault classification

    fault_face_indices: typing.Optional[typing.Mapping[str, IntArray[OneDimension]]] = (
        attrs.field(default=None)
    )
    """
    Mapping from fault name to a 1-D array of face indices belonging to that
    fault.

    Populated from the GRDECL `FAULTS` keyword when present.  `None`
    when no fault definitions were parsed.  The face indices reference the
    same ordering as `face_cell_indices`.
    """

    fault_transmissibility_multipliers: typing.Optional[typing.Mapping[str, float]] = (
        attrs.field(default=None)
    )
    """
    Mapping from fault name to its transmissibility multiplier.

    Populated from the GRDECL `MULTFLT` keyword.  A multiplier of 0.0
    closes the fault entirely; 1.0 leaves it fully open; values in between
    reduce flow proportionally.  `None` when no `MULTFLT` data was
    parsed.

    To obtain the effective transmissibility for a named fault face::

        T_effective = T_geometric * fault_transmissibility_multipliers[fault_name]
    """

    # Directional transmissibility multipliers (MULTX / MULTY / MULTZ and their - variants)

    positive_x_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell transmissibility multiplier applied to
    the **positive-x** face of each cell (GRDECL ``MULTX`` keyword).

    The effective inter-cell transmissibility between cell *i* and its
    +x neighbour *i+1* is scaled by
    ``positive_x_transmissibility_multipliers[i] * negative_x_transmissibility_multipliers[i+1]``.
    `None` when not supplied (equivalent to all-ones).
    """

    negative_x_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell transmissibility multiplier applied to
    the **negative-x** face of each cell (GRDECL ``MULTX-`` keyword).

    Combined with `positive_x_transmissibility_multipliers` on the adjacent cell to give the
    face multiplier in the x-direction.  `None` when not supplied.
    """

    positive_y_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell transmissibility multiplier applied to
    the **positive-y** face (GRDECL ``MULTY``).  `None` when not supplied.
    """

    negative_y_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell transmissibility multiplier applied to
    the **negative-y** face (GRDECL ``MULTY-``).  `None` when not supplied.
    """

    positive_z_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell transmissibility multiplier applied to
    the **positive-z** face (GRDECL ``MULTZ``).  `None` when not supplied.
    """

    negative_z_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell transmissibility multiplier applied to
    the **negative-z** face (GRDECL ``MULTZ-``).  `None` when not supplied.
    """

    # Derived topology

    cell_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """
    Flat CSR data array: concatenated face index lists for all cells.

    Cell *c* uses
    `cell_face_indices[cell_face_offsets[c]:cell_face_offsets[c+1]]`.
    """

    cell_face_offsets: IntArray[OneDimension] = attrs.field(init=False)
    """CSR offset array of length `n_cells + 1` for the cell-to-face map."""

    cell_neighbor_indices: IntArray[OneDimension] = attrs.field(init=False)
    """
    Flat CSR data array: concatenated neighbour cell index lists for all
    cells.

    Cell *c* uses
    `cell_neighbor_indices[cell_neighbor_offsets[c]:cell_neighbor_offsets[c+1]]`.
    Only cells sharing an **interior** face (`ConnectionType.INTERIOR`,
    `FAULT`, or `PINCHOUT`) are listed; boundary faces do not contribute.
    """

    cell_neighbor_offsets: IntArray[OneDimension] = attrs.field(init=False)
    """
    CSR offset array of length `n_cells + 1` for the cell-to-neighbour map.
    """

    boundary_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Indices of all boundary faces (`neighbour_cell_index == -1`)."""

    interior_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Indices of all interior faces (both owner and neighbour cells >= 0)."""

    # Derived geometry (computed in __attrs_post_init__)

    face_centroids: FloatArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_faces, 3)` - (x, y, z) centroid of each face polygon."""

    face_areas: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_faces,)` - geometric area of each face in grid units²."""

    face_unit_normals: FloatArray[TwoDimensions] = attrs.field(init=False)
    """
    Shape `(n_faces, 3)` - unit outward normal from the owner cell for
    each face.
    """

    cell_centroids: FloatArray[TwoDimensions] = attrs.field(init=False)
    """
    Shape `(n_cells, 3)` - volume-weighted (x, y, z) centroid of each
    cell.
    """

    cell_volumes: FloatArray[OneDimension] = attrs.field(init=False)
    """
    Shape `(n_cells,)` - bulk geometric volume of each cell in grid
    units³.
    """

    cell_min_xyz: FloatArray[TwoDimensions] = attrs.field(init=False)
    """
    Shape `(n_cells, 3)` - axis-aligned bounding box minimum corner per
    cell.
    """

    cell_max_xyz: FloatArray[TwoDimensions] = attrs.field(init=False)
    """
    Shape `(n_cells, 3)` - axis-aligned bounding box maximum corner per
    cell.
    """

    bounding_box: tuple[float, float, float, float, float, float] = attrs.field(
        init=False
    )
    """
    Axis-aligned bounding box of the entire grid.

    `(x_min, x_max, y_min, y_max, z_min, z_max)` in grid coordinates.
    """

    cell_length_x: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - bounding-box extent in the x direction."""

    cell_length_y: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - bounding-box extent in the y direction."""

    cell_length_z: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - bounding-box extent in the z direction (thickness)."""

    cell_thickness: FloatArray[OneDimension] = attrs.field(init=False)
    """
    Shape `(n_cells,)` - vertical thickness of each cell (alias for
    `cell_length_z`).
    """

    cell_center_depths: FloatArray[OneDimension] = attrs.field(init=False)
    """
    Shape `(n_cells,)` - depth of each cell centroid (positive downward =
    centroid z).
    """

    cell_center_elevations: FloatArray[OneDimension] = attrs.field(init=False)
    """
    Shape `(n_cells,)` - elevation of each cell centroid (positive upward =
    -depth).
    """

    _spatial_index: typing.Optional[cKDTree] = attrs.field(init=False, default=None)
    """
    KD-tree built on cell centroids for fast spatial lookup.  Internal use
    only.
    """

    # Construction

    def __attrs_post_init__(self) -> None:
        """
        Validate inputs and compute all derived topology and geometry.
        Steps performed in order:

        1. Validate primary input arrays.
        2. Classify faces into boundary / interior subsets.
        3. Auto-populate status and connection-type arrays when not supplied.
        4. Build cell-to-face CSR connectivity.
        5. Build cell-to-neighbour CSR connectivity.
        6. Compute face geometry (centroids, areas, normals).
        7. Compute cell geometry (volumes, centroids).
        8. Compute per-cell bounding boxes and the global bounding box.
        9. Derive scalar cell dimensions.
        10. Build the KD-tree spatial index.
        """
        self._validate_inputs()
        self._classify_faces()
        self._populate_default_statuses()
        self._build_cell_face_connectivity()
        self._build_cell_neighbor_connectivity()
        self._compute_face_geometry()
        self._compute_cell_geometry()
        self._compute_bounding_boxes()
        self._compute_derived_dimensions()
        self._build_spatial_index()

    def _validate_inputs(self) -> None:
        """
        Validate primary input arrays for shape and internal consistency.

        :raises InvalidPointArrayError: If `vertex_coordinates` is not
            `(N, 3)`.
        :raises InvalidFaceConnectivityError: If face connectivity arrays are
            malformed.
        """
        if self.vertex_coordinates.ndim != 2 or self.vertex_coordinates.shape[1] != 3:
            raise InvalidPointArrayError(
                f"`vertex_coordinates` must be shape (n_vertices, 3); "
                f"got {self.vertex_coordinates.shape!r}."
            )
        if self.face_cell_indices.ndim != 2 or self.face_cell_indices.shape[1] != 2:
            raise InvalidFaceConnectivityError(
                f"`face_cell_indices` must be shape (n_faces, 2); "
                f"got {self.face_cell_indices.shape!r}."
            )
        if self.face_vertex_offsets.ndim != 1 or self.face_vertex_offsets[0] != 0:
            raise InvalidFaceConnectivityError(
                "`face_vertex_offsets` must be a 1-D array starting at 0."
            )
        expected_n_faces = self.face_cell_indices.shape[0]
        if self.face_vertex_offsets.shape[0] != expected_n_faces + 1:
            raise InvalidFaceConnectivityError(
                f"`face_vertex_offsets` length must be n_faces + 1 = "
                f"{expected_n_faces + 1}; got {self.face_vertex_offsets.shape[0]}."
            )
        if int(self.face_vertex_offsets[-1]) != len(self.face_vertex_indices):
            raise InvalidFaceConnectivityError(
                f"face_vertex_offsets[-1] = {self.face_vertex_offsets[-1]} does not "
                f"match len(face_vertex_indices) = {len(self.face_vertex_indices)}."
            )
        max_valid_vertex = self.vertex_coordinates.shape[0] - 1
        if self.face_vertex_indices.size > 0:
            if int(self.face_vertex_indices.max()) > max_valid_vertex:
                raise InvalidFaceConnectivityError(
                    f"`face_vertex_indices` contains vertex index "
                    f"{int(self.face_vertex_indices.max())} which exceeds "
                    f"the maximum valid index {max_valid_vertex}."
                )
        min_cell_index = int(self.face_cell_indices.min())
        if min_cell_index < -1:
            raise InvalidFaceConnectivityError(
                f"`face_cell_indices` contains negative cell index {min_cell_index}; "
                "only -1 is allowed (boundary sentinel)."
            )

    def _classify_faces(self) -> None:
        """
        Partition face indices into boundary and interior subsets.

        A boundary face has `neighbour_cell == -1`; an interior face has
        both owner and neighbour cells >= 0.

        Results are stored in `boundary_face_indices` and
        `interior_face_indices`.
        """
        owner_cells = self.face_cell_indices[:, 0]
        neighbour_cells = self.face_cell_indices[:, 1]

        boundary_mask = (owner_cells < 0) | (neighbour_cells < 0)
        interior_mask = (owner_cells >= 0) & (neighbour_cells >= 0)

        object.__setattr__(
            self,
            "boundary_face_indices",
            np.where(boundary_mask)[0].astype(self.index_dtype),
        )
        object.__setattr__(
            self,
            "interior_face_indices",
            np.where(interior_mask)[0].astype(self.index_dtype),
        )

    def _populate_default_statuses(self) -> None:
        """
        Auto-populate `connection_types`, `cell_statuses`, and
        `face_statuses` when the caller did not supply explicit arrays.

        `connection_types`
            `BOUNDARY` for faces with `neighbour == -1`, `INTERIOR`
            for all others. Factories that know about faults, pinchouts, or
            NNCs should pass an overriding array.

        `cell_statuses`
            All `CellStatus.ACTIVE` by default.

        `face_statuses`
            All `FaceStatus.ACTIVE` by default.
        """
        n_faces = self.face_cell_indices.shape[0]
        n_cells = int(self.face_cell_indices.max()) + 1

        if self.connection_types is None:
            ct = np.full(n_faces, ConnectionType.INTERIOR, dtype=np.int8)
            ct[self.boundary_face_indices] = int(ConnectionType.BOUNDARY)
            object.__setattr__(self, "connection_types", ct)

        if self.cell_statuses is None:
            cs = np.full(n_cells, CellStatus.ACTIVE, dtype=np.int8)
            object.__setattr__(self, "cell_statuses", cs)

        if self.face_statuses is None:
            fs = np.full(n_faces, FaceStatus.ACTIVE, dtype=np.int8)
            object.__setattr__(self, "face_statuses", fs)

    def _build_cell_face_connectivity(self) -> None:
        """
        Build CSR cell-to-face adjacency lists from `face_cell_indices`.

        Each cell accumulates the indices of every face that touches it (as
        either owner or neighbour).  Results are stored in `cell_face_indices`
        and `cell_face_offsets`.
        """
        n_cells = int(self.face_cell_indices.max()) + 1

        cell_face_lists: list[list[int]] = [[] for _ in range(n_cells)]
        for face_idx, (owner, neighbour) in enumerate(self.face_cell_indices):
            if owner >= 0:
                cell_face_lists[owner].append(face_idx)
            if neighbour >= 0:
                cell_face_lists[neighbour].append(face_idx)

        flat_face_indices: list[int] = []
        csr_offsets: list[int] = [0]
        for faces in cell_face_lists:
            flat_face_indices.extend(faces)
            csr_offsets.append(len(flat_face_indices))

        object.__setattr__(
            self,
            "cell_face_indices",
            np.asarray(flat_face_indices, dtype=self.index_dtype),
        )
        object.__setattr__(
            self,
            "cell_face_offsets",
            np.asarray(csr_offsets, dtype=self.index_dtype),
        )

    def _build_cell_neighbor_connectivity(self) -> None:
        """
        Build CSR cell-to-neighbour adjacency lists from `face_cell_indices`.

        Two cells are neighbours if they share an interior face (i.e. both
        owner and neighbour indices >= 0).  Boundary faces do not contribute
        neighbours.  Results are stored in `cell_neighbor_indices` and
        `cell_neighbor_offsets`.
        """
        n_cells = int(self.face_cell_indices.max()) + 1

        neighbor_sets: list[set[int]] = [set() for _ in range(n_cells)]
        for owner, neighbour in self.face_cell_indices:
            if owner >= 0 and neighbour >= 0:
                neighbor_sets[owner].add(neighbour)
                neighbor_sets[neighbour].add(owner)

        flat_neighbor_indices: list[int] = []
        csr_offsets: list[int] = [0]
        for neighbors in neighbor_sets:
            flat_neighbor_indices.extend(sorted(neighbors))
            csr_offsets.append(len(flat_neighbor_indices))

        object.__setattr__(
            self,
            "cell_neighbor_indices",
            np.asarray(flat_neighbor_indices, dtype=self.index_dtype),
        )
        object.__setattr__(
            self,
            "cell_neighbor_offsets",
            np.asarray(csr_offsets, dtype=self.index_dtype),
        )

    def _compute_face_geometry(self) -> None:
        """Compute face centroids, areas, and unit outward normals."""
        face_centroids, face_areas, face_unit_normals = _compute_face_geometry(
            face_vertex_indices=self.face_vertex_indices,
            face_vertex_offsets=self.face_vertex_offsets,
            vertex_coordinates=self.vertex_coordinates,
        )
        object.__setattr__(self, "face_centroids", face_centroids)
        object.__setattr__(self, "face_areas", face_areas)
        object.__setattr__(self, "face_unit_normals", face_unit_normals)

    def _compute_cell_geometry(self) -> None:
        """
        Compute cell volumes and volume-weighted centroids.

        :raises InvalidVolumeError: If any cell has a non-positive volume.
        """
        n_cells = int(self.face_cell_indices.max()) + 1
        cell_volumes, cell_centroids = _compute_cell_volumes_and_centroids(
            face_cell_indices=self.face_cell_indices,
            face_vertex_indices=self.face_vertex_indices,
            face_vertex_offsets=self.face_vertex_offsets,
            vertex_coordinates=self.vertex_coordinates,
            n_cells=n_cells,
        )
        invalid_volume_mask = cell_volumes <= 0.0
        if invalid_volume_mask.any():
            bad_cells = np.where(invalid_volume_mask)[0].tolist()
            raise InvalidVolumeError(
                f"Cells {bad_cells[:10]}{'...' if len(bad_cells) > 10 else ''} "
                f"have non-positive computed volumes.  Check face winding order "
                f"(vertices must be CCW from the owner-cell side)."
            )
        object.__setattr__(self, "cell_volumes", cell_volumes)
        object.__setattr__(self, "cell_centroids", cell_centroids)

    def _compute_bounding_boxes(self) -> None:
        """
        Compute per-cell axis-aligned bounding boxes from face vertex
        coordinates.
        """
        n_cells = int(self.face_cell_indices.max()) + 1
        cell_min, cell_max = _compute_cell_bounding_boxes(
            face_cell_indices=self.face_cell_indices,
            face_vertex_indices=self.face_vertex_indices,
            face_vertex_offsets=self.face_vertex_offsets,
            vertex_coordinates=self.vertex_coordinates,
            n_cells=n_cells,
            dtype=self.floating_dtype,
        )
        bounding_box = (
            float(cell_min[:, 0].min()),
            float(cell_max[:, 0].max()),
            float(cell_min[:, 1].min()),
            float(cell_max[:, 1].max()),
            float(cell_min[:, 2].min()),
            float(cell_max[:, 2].max()),
        )
        object.__setattr__(self, "cell_min_xyz", cell_min)
        object.__setattr__(self, "cell_max_xyz", cell_max)
        object.__setattr__(self, "bounding_box", bounding_box)

    def _compute_derived_dimensions(self) -> None:
        """
        Derive per-cell scalar dimensions from axis-aligned bounding boxes.

        Computes `cell_length_x/y/z`, `cell_thickness`,
        `cell_center_depths`, and `cell_center_elevations`.

        Depth is positive downward (z-axis convention); elevation is the
        negation of depth.
        """
        delta = self.cell_max_xyz - self.cell_min_xyz
        object.__setattr__(self, "cell_length_x", delta[:, 0])
        object.__setattr__(self, "cell_length_y", delta[:, 1])
        object.__setattr__(self, "cell_length_z", delta[:, 2])
        object.__setattr__(self, "cell_thickness", delta[:, 2])

        depths = self.cell_centroids[:, 2].copy()
        object.__setattr__(self, "cell_center_depths", depths)
        object.__setattr__(self, "cell_center_elevations", -depths)

    def _build_spatial_index(self) -> None:
        """Construct a KD-tree on cell centroids for fast nearest-cell queries."""
        object.__setattr__(self, "_spatial_index", cKDTree(self.cell_centroids))

    @property
    def n_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.cell_centroids.shape[0]

    @property
    def n_faces(self) -> int:
        """Total number of faces (boundary + interior) in the grid."""
        return self.face_cell_indices.shape[0]

    @property
    def n_vertices(self) -> int:
        """Total number of vertex points in the grid."""
        return self.vertex_coordinates.shape[0]

    @property
    def n_boundary_faces(self) -> int:
        """Number of boundary faces (faces on the outer hull of the domain)."""
        return len(self.boundary_face_indices)

    @property
    def n_interior_faces(self) -> int:
        """Number of interior faces (faces shared between two cells)."""
        return len(self.interior_face_indices)

    @property
    def n_nnc(self) -> int:
        """
        Number of non-neighbour connections (NNCs).

        Returns 0 when `nnc_cell_pairs` is `None`.
        """
        if self.nnc_cell_pairs is None:
            return 0
        return self.nnc_cell_pairs.shape[0]

    @property
    def n_faults(self) -> int:
        """
        Number of named faults in the grid.

        Returns 0 when `fault_face_indices` is `None`.
        """
        if self.fault_face_indices is None:
            return 0
        return len(self.fault_face_indices)

    @property
    def has_transmissibility_multipliers(self) -> bool:
        """
        Return `True` if any directional transmissibility multiplier array
        is present on this grid.

        Checks all six MULT arrays (MULTX, MULTX-, MULTY, MULTY-, MULTZ,
        MULTZ-). A return value of `False` means all inter-cell
        transmissibilities can be computed from geometry alone.
        """
        return any(
            arr is not None
            for arr in (
                self.positive_x_transmissibility_multipliers,
                self.negative_x_transmissibility_multipliers,
                self.positive_y_transmissibility_multipliers,
                self.negative_y_transmissibility_multipliers,
                self.positive_z_transmissibility_multipliers,
                self.negative_z_transmissibility_multipliers,
            )
        )

    def get_cell_face_indices(self, cell_index: int) -> IntArray[OneDimension]:
        """
        Return the indices of all faces belonging to a given cell.

        :param cell_index: Zero-based cell index.
        :returns: 1-D array of face indices for the requested cell.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )
        start = self.cell_face_offsets[cell_index]
        end = self.cell_face_offsets[cell_index + 1]
        return self.cell_face_indices[start:end]

    def get_cell_neighbor_indices(self, cell_index: int) -> IntArray[OneDimension]:
        """
        Return the indices of all cells neighbouring a given cell.

        Only cells sharing an interior face are considered neighbours.

        :param cell_index: Zero-based cell index.
        :returns: 1-D array of neighbouring cell indices.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )
        start = self.cell_neighbor_offsets[cell_index]
        end = self.cell_neighbor_offsets[cell_index + 1]
        return self.cell_neighbor_indices[start:end]

    def get_face_vertex_coordinates(self, face_index: int) -> FloatArray[TwoDimensions]:
        """
        Return the (x, y, z) coordinates of all vertices of a given face.

        :param face_index: Zero-based face index.
        :returns: Shape `(n_verts_for_face, 3)` coordinate array.
        :raises IndexError: If `face_index` is out of range.
        """
        start = int(self.face_vertex_offsets[face_index])
        end = int(self.face_vertex_offsets[face_index + 1])
        return self.vertex_coordinates[self.face_vertex_indices[start:end]]

    def get_face_normal_for_cell(
        self, face_index: int, cell_index: int
    ) -> FloatArray[OneDimension]:
        """
        Return the outward unit normal of a face relative to a specific cell.

        The stored normal points outward from the owner cell.  For the
        neighbour cell the normal is reversed.

        :param face_index: Zero-based face index.
        :param cell_index: Zero-based cell index (must be owner or neighbour
            of the face).
        :returns: Shape `(3,)` unit normal vector pointing outward from
            `cell_index`.
        :raises ValidationError: If `cell_index` is not connected to
            `face_index`.
        """
        owner = int(self.face_cell_indices[face_index, 0])
        neighbour = int(self.face_cell_indices[face_index, 1])

        if cell_index == owner:
            return self.face_unit_normals[face_index]
        elif cell_index == neighbour:
            return -self.face_unit_normals[face_index]
        raise ValidationError(
            f"Cell {cell_index} is not connected to face {face_index} "
            f"(owner={owner}, neighbour={neighbour})."
        )

    def get_boundary_cell_indices(self) -> IntArray[OneDimension]:
        """
        Return the indices of all cells that touch at least one boundary face.

        :returns: 1-D sorted array of boundary-adjacent cell indices.
        """
        boundary_owners = self.face_cell_indices[self.boundary_face_indices, 0]
        boundary_neighbours = self.face_cell_indices[self.boundary_face_indices, 1]

        all_boundary_cells = np.concatenate(
            [
                boundary_owners[boundary_owners >= 0],
                boundary_neighbours[boundary_neighbours >= 0],
            ]
        )
        return np.unique(all_boundary_cells).astype(self.index_dtype)

    def get_interior_cell_indices(self) -> IntArray[OneDimension]:
        """
        Return the indices of all cells that have no boundary faces.

        :returns: 1-D sorted array of fully interior cell indices.
        """
        boundary_cells = self.get_boundary_cell_indices()
        all_cells = np.arange(self.n_cells, dtype=self.index_dtype)
        return np.setdiff1d(all_cells, boundary_cells)

    def is_boundary_cell(self, cell_index: int) -> bool:
        """
        Return whether a given cell is adjacent to at least one boundary face.

        :param cell_index: Zero-based cell index.
        :returns: `True` if the cell has at least one boundary face.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )
        face_indices = self.get_cell_face_indices(cell_index)
        for face_idx in face_indices:
            owner = int(self.face_cell_indices[face_idx, 0])
            neighbour = int(self.face_cell_indices[face_idx, 1])
            if owner < 0 or neighbour < 0:
                return True
        return False

    def get_fault_face_indices(self, fault_name: str) -> IntArray[OneDimension]:
        """
        Return the face indices for a named fault.

        :param fault_name: Name of the fault as declared in the GRDECL
            `FAULTS` keyword.
        :returns: 1-D array of face indices belonging to that fault.
        :raises KeyError: If `fault_name` is not found in
            `fault_face_indices`.
        :raises ValidationError: If no fault data is available on this grid.
        """
        if self.fault_face_indices is None:
            raise ValidationError(
                "No fault data available on this grid (fault_face_indices is None)."
            )
        if fault_name not in self.fault_face_indices:
            available = sorted(self.fault_face_indices.keys())
            raise KeyError(
                f"Fault {fault_name!r} not found.  Available faults: {available}."
            )
        return self.fault_face_indices[fault_name]

    def get_fault_transmissibility_multiplier(self, fault_name: str) -> float:
        """
        Return the transmissibility multiplier for a named fault.

        :param fault_name: Name of the fault as declared in `MULTFLT`.
        :returns: Multiplier value (1.0 = fully open, 0.0 = fully closed).
        :raises KeyError: If `fault_name` is not found.
        :raises ValidationError: If no multiplier data is available.
        """
        if self.fault_transmissibility_multipliers is None:
            raise ValidationError(
                "No fault transmissibility multipliers available on this grid."
            )
        if fault_name not in self.fault_transmissibility_multipliers:
            available = sorted(self.fault_transmissibility_multipliers.keys())
            raise KeyError(
                f"Fault {fault_name!r} not found in MULTFLT data.  "
                f"Available: {available}."
            )
        return self.fault_transmissibility_multipliers[fault_name]

    def find_nearest_cell(self, x: float, y: float, z: float) -> int:
        """
        Find the cell whose centroid is nearest to the given (x, y, z) point.

        Uses a pre-built KD-tree for O(log n) lookup.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (depth, positive downward).
        :returns: Zero-based index of the nearest cell.
        """
        _, cell_index = self._spatial_index.query([x, y, z])  # type: ignore
        return int(cell_index)

    def find_cells_in_radius(
        self, x: float, y: float, z: float, radius: float
    ) -> IntArray[OneDimension]:
        """
        Return all cell indices whose centroids fall within `radius` of a
        point.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (depth, positive downward).
        :param radius: Search radius in grid length units.
        :returns: 1-D array of matching cell indices (unsorted).
        """
        raw_indices = self._spatial_index.query_ball_point(  # type: ignore
            [x, y, z], r=radius
        )
        return np.asarray(raw_indices, dtype=self.index_dtype)

    def compute_pore_volume(
        self,
        porosity: NumberOrArray[OneDimension],
        net_to_gross: NumberOrArray[OneDimension],
    ) -> FloatArray[OneDimension]:
        """
        Compute the pore volume for each cell given a porosity field.

        :param porosity: Scalar or shape `(n_cells,)` array of porosity
            values (dimensionless, in `[0, 1]`).
        :returns: Pore volumes in the same units³ as `cell_volumes`,
            broadcast against `porosity`.
        """
        return porosity * net_to_gross * self.cell_volumes

    def validate_geometry(self) -> None:
        """
        Validate that all computed geometry values are physically reasonable.

        Checks that all cell volumes are strictly positive, all face areas are
        non-negative, and all face unit normals have unit magnitude.

        :raises InvalidVolumeError: If any cell volume is <= 0.
        :raises InvalidFaceAreaError: If any face area is negative.
        :raises InvalidNormalVectorError: If any face normal deviates from
            unit length by more than a loose tolerance.
        """
        if (self.cell_volumes <= 0.0).any():
            bad = np.where(self.cell_volumes <= 0.0)[0]
            raise InvalidVolumeError(
                f"{len(bad)} cell(s) have non-positive volume: {bad[:5].tolist()}..."
            )
        if (self.face_areas < 0.0).any():
            bad = np.where(self.face_areas < 0.0)[0]
            raise InvalidFaceAreaError(
                f"{len(bad)} face(s) have negative area: {bad[:5].tolist()}..."
            )
        normal_magnitudes = np.linalg.norm(self.face_unit_normals, axis=1)
        active_mask = self.face_areas > _GEOMETRY_TOLERANCE
        if active_mask.any():
            deviation = np.abs(normal_magnitudes[active_mask] - 1.0)
            if (deviation > 1e-10).any():
                raise InvalidNormalVectorError(
                    "One or more face unit normals do not have unit magnitude "
                    f"(max deviation = {deviation.max():.3e})."
                )
