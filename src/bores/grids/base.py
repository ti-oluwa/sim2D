"""Face-based unstructured polyhedral grid for reservoir simulation."""

import enum
import typing

import attrs
import numba
import numpy as np
from scipy.spatial import cKDTree
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import (
    CellNotFoundError,
    InvalidFaceAreaError,
    InvalidFaceConnectivityError,
    InvalidNormalVectorError,
    InvalidPointArrayError,
    InvalidVolumeError,
    ValidationError,
)
from bores.serialization.base import Serializable
from bores.typing import (
    IntArray,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["Grid", "ConnectionType", "CellStatus"]


class ConnectionType(enum.IntEnum):
    """
    Classification of flow connections between reservoir cells.

    Face-based connections originate from the grid topology and correspond
    to geometric faces in the mesh. NNC-based connections represent
    additional cell-to-cell flow paths that do not correspond to a shared
    geometric face.

    Connection types determine how transmissibilities and transmissibility
    multipliers are applied during simulation.

    **Face connections**:

    `INTERIOR_FACE`
        Standard internal face shared by two active cells.

    `BOUNDARY_FACE`
        Boundary face connecting an active cell to the exterior domain.
        The neighbour cell index is `-1`.

    `INTERIOR_FAULT_FACE`
        Internal face shared by two active cells and belonging to a named
        fault. Directional transmissibility multipliers and fault
        transmissibility multipliers may both apply.

    `BOUNDARY_FAULT_FACE`
        Boundary face belonging to a named fault.


    **NNC connections**:

    `PINCHOUT_NNC`
        Non-neighbour connection generated from pinchout processing where
        flow occurs between cells that do not share a geometric face.

    `USER_NNC`
        Explicit user-defined non-neighbour connection, typically created
        from an Eclipse `NNC` keyword.

    `FAULT_NNC`
        Non-neighbour connection generated from fault juxtaposition where
        communicating cells do not share a geometric face.
    """

    INTERIOR_FACE = 0
    """Standard interior face between two active cells."""

    BOUNDARY_FACE = 1
    """Boundary face between an active cell and the exterior domain."""

    INTERIOR_FAULT_FACE = 2
    """
    Interior face belonging to a fault.

    Both directional transmissibility multipliers and fault
    transmissibility multipliers may apply.
    """

    BOUNDARY_FAULT_FACE = 3
    """
    Boundary face belonging to a fault.
    """

    PINCHOUT_NNC = 4
    """
    Non-neighbour connection generated from pinchout processing.
    """

    USER_NNC = 5
    """
    Explicit user-defined non-neighbour connection.
    """

    FAULT_NNC = 6
    """
    Non-neighbour connection generated from fault juxtaposition.
    """


class CellStatus(enum.IntEnum):
    """
    Activation status of a grid cell.

    `ACTIVE`
        The cell participates in flow simulation.

    `INACTIVE`
        The cell is excluded from flow simulation (e.g. ACTNUM == 0).
    """

    ACTIVE = 1
    INACTIVE = 0


_GEOMETRY_TOLERANCE: float = 1e-14


@numba.njit(parallel=True, cache=True)
def _compute_face_geometry(
    face_vertex_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
) -> typing.Tuple[
    NumberArray[TwoDimensions],
    NumberArray[OneDimension],
    NumberArray[TwoDimensions],
]:
    """
    Compute face centroids, areas, and unit outward normals via Newell's method.

    :param face_vertex_indices: Flat CSR data array of vertex indices.
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :returns: Tuple `(face_centroids, face_areas, face_unit_normals)`.
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
    vertex_coordinates: NumberArray[TwoDimensions],
    n_cells: int,
) -> typing.Tuple[NumberArray[OneDimension], NumberArray[TwoDimensions]]:
    """
    Compute cell volumes and centroids via the divergence theorem.

    :param face_cell_indices: Shape `(n_faces, 2)`.
    :param face_vertex_indices: Flat CSR vertex index data array.
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)`.
    :param n_cells: Total number of cells.
    :returns: Tuple `(cell_volumes, cell_centroids)`.
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

                x_bar = (ax + bx + cx) / 4.0
                y_bar = (ay + by + cy) / 4.0
                z_bar = (az + bz + cz) / 4.0

                w = sign * signed_tetrahedron_volume
                centroid_accumulators[cell_idx, 0] += w * x_bar
                centroid_accumulators[cell_idx, 1] += w * y_bar
                centroid_accumulators[cell_idx, 2] += w * z_bar

    cell_centroids = np.zeros((n_cells, 3), dtype=np.float64)
    for cell_idx in range(n_cells):
        vol = cell_volumes[cell_idx]
        if abs(vol) > 0.0:
            cell_centroids[cell_idx, 0] = centroid_accumulators[cell_idx, 0] / vol
            cell_centroids[cell_idx, 1] = centroid_accumulators[cell_idx, 1] / vol
            cell_centroids[cell_idx, 2] = centroid_accumulators[cell_idx, 2] / vol

    return cell_volumes, cell_centroids


@numba.njit(cache=True)
def _compute_cell_bounding_boxes(
    face_cell_indices: IntArray[TwoDimensions],
    face_vertex_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
    n_cells: int,
) -> typing.Tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]]:
    """
    Compute per-cell axis-aligned bounding boxes.

    :param face_cell_indices: Shape `(n_faces, 2)`.
    :param face_vertex_indices: Flat CSR vertex index data array.
    :param face_vertex_offsets: CSR offset array.
    :param vertex_coordinates: Shape `(n_vertices, 3)`.
    :param n_cells: Total number of cells.
    :returns: Tuple `(cell_min_xyz, cell_max_xyz)` each of shape `(n_cells, 3)`.
    """
    cell_min = np.full((n_cells, 3), np.inf, dtype=np.float64)
    cell_max = np.full((n_cells, 3), -np.inf, dtype=np.float64)
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


@typing.final
@attrs.mutable(frozen=True, slots=True, kw_only=True)
class Grid(
    Serializable,
    fields={
        "vertex_coordinates": NumberArray[TwoDimensions],
        "face_vertex_indices": IntArray[OneDimension],
        "face_vertex_offsets": IntArray[OneDimension],
        "face_cell_indices": IntArray[TwoDimensions],
        "unit_system": UnitSystem,
        "metadata": typing.Optional[typing.Mapping[str, typing.Any]],
        "cell_volumes": typing.Optional[NumberArray[OneDimension]],
        "cell_centroids": typing.Optional[NumberArray[TwoDimensions]],
        "nnc_cell_indices": typing.Optional[IntArray[TwoDimensions]],
        "nnc_transmissibilities": typing.Optional[NumberArray[OneDimension]],
        "nnc_connection_types": typing.Optional[IntArray[OneDimension]],
        "nnc_fault_indices": typing.Optional[
            typing.Mapping[str, IntArray[OneDimension]]
        ],
        "fault_face_indices": typing.Optional[
            typing.Mapping[str, IntArray[OneDimension]]
        ],
        "fault_transmissibility_multipliers": typing.Optional[
            typing.Mapping[str, Number]
        ],
        "positive_x_transmissibility_multipliers": typing.Optional[
            NumberArray[OneDimension]
        ],
        "negative_x_transmissibility_multipliers": typing.Optional[
            NumberArray[OneDimension]
        ],
        "positive_y_transmissibility_multipliers": typing.Optional[
            NumberArray[OneDimension]
        ],
        "negative_y_transmissibility_multipliers": typing.Optional[
            NumberArray[OneDimension]
        ],
        "positive_z_transmissibility_multipliers": typing.Optional[
            NumberArray[OneDimension]
        ],
        "negative_z_transmissibility_multipliers": typing.Optional[
            NumberArray[OneDimension]
        ],
    },
):
    """
    Immutable face-based unstructured polyhedral grid.

    All topology and geometry is computed once during construction and stored as
    read-only NumPy arrays. All index arrays use int32 and all floating-point
    arrays use float64.

    **Connection model**

    Two layers of connection data are maintained:

    `face_connection_types` (shape `(n_faces,)`)
        Per-face type: `BOUNDARY_FACE`, `INTERIOR_FACE`, `INTERIOR_FAULT_FACE`,
        `BOUNDARY_FAULT_FACE`. This covers every geometric face in the grid.

    `nnc_connection_types` (shape `(n_nnc,)`)
        Per-NNC type: `USER_NNC`, `FAULT_NNC`, or `PINCHOUT_NNC`.
        These are connections with no corresponding shared geometric face.

    **Raises**:

    `InvalidPointArrayError`
        If `vertex_coordinates` is not a 2-D `(n_vertices, 3)` array.
    `InvalidFaceConnectivityError`
        If face connectivity arrays are malformed.
    `InvalidVolumeError`
        If any cell has a non-positive volume after construction.
    """

    vertex_coordinates: NumberArray[TwoDimensions]
    """
    Shape `(n_vertices, 3)` - world (x, y, z) coordinates.
    z-axis is positive downward (reservoir depth convention).
    """

    face_vertex_indices: IntArray[OneDimension]
    """
    Flat CSR data array: concatenated vertex index lists for all faces.
    Face *f* uses
    `face_vertex_indices[face_vertex_offsets[f]:face_vertex_offsets[f+1]]`.
    """

    face_vertex_offsets: IntArray[OneDimension]
    """CSR offset array of length `n_faces + 1`."""

    face_cell_indices: IntArray[TwoDimensions]
    """
    Shape `(n_faces, 2)` - `(owner_cell_index, neighbour_cell_index)`.
    Boundary faces have `neighbour_cell_index == -1`.
    """

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """Declared unit system for all coordinate and geometry arrays."""

    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = attrs.field(
        default=None
    )
    """Optional free-form metadata mapping."""

    cell_statuses: typing.Optional[IntArray[OneDimension]] = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell `CellStatus` flags.
    Auto-populated to all `CellStatus.ACTIVE` when `None`.
    """

    face_connection_types: typing.Optional[IntArray[OneDimension]] = attrs.field(
        default=None
    )
    """
    Shape `(n_faces,)` - per-face `ConnectionType`.
    Auto-populated from topology (`BOUNDARY` / `INTERIOR`) when `None`.
    Factories that know about faults or pinchouts supply an explicit array.
    """

    cell_volumes: typing.Optional[NumberArray[OneDimension]] = attrs.field(default=None)
    """
    Shape `(n_cells,)` pre-computed cell volumes from the factory.
    When provided, the divergence-theorem computation is skipped.
    """

    cell_centroids: typing.Optional[NumberArray[TwoDimensions]] = attrs.field(
        default=None
    )
    """
    Shape `(n_cells, 3)` pre-computed cell centroids from the factory.
    Must be provided together with `cell_volumes`.
    """

    nnc_cell_indices: typing.Optional[IntArray[TwoDimensions]] = attrs.field(
        default=None
    )
    """
    Shape `(n_nnc, 2)` - non-neighbour connection cell index pairs.
    `None` when no NNCs are present.
    """

    nnc_connection_types: typing.Optional[IntArray[OneDimension]] = attrs.field(
        default=None
    )
    """
    Shape `(n_nnc,)` - `ConnectionType` for each NNC entry.
    Auto-populated to all `NNC` when `nnc_cell_indices`
    is provided but this is `None`.
    """

    nnc_transmissibilities: typing.Optional[NumberArray[OneDimension]] = attrs.field(
        default=None
    )
    """
    Shape `(n_nnc,)` - transmissibility for each NNC pair.
    `None` when not supplied; NaN entries indicate geometrically-detected
    connections whose T must be computed.
    """

    nnc_fault_indices: typing.Optional[typing.Mapping[str, IntArray[OneDimension]]] = (
        attrs.field(default=None)
    )
    """
    Mapping from fault name to 1-D array of NNC indices (positions into
    `nnc_cell_indices`) belonging to that fault.

    Mirrors `fault_face_indices` but for NNC-type connections. Populated by
    factories when fault cell pairs have no shared geometric face.
    `None` when no fault NNCs are present.
    """

    fault_face_indices: typing.Optional[typing.Mapping[str, IntArray[OneDimension]]] = (
        attrs.field(default=None)
    )
    """
    Mapping from fault name to 1-D array of face indices belonging to that fault.
    Populated from the GRDECL `FAULTS` keyword. `None` when absent.
    """

    fault_transmissibility_multipliers: typing.Optional[typing.Mapping[str, Number]] = (
        attrs.field(default=None)
    )
    """
    Mapping from fault name to its transmissibility multiplier (from `MULTFLT`).
    `None` when absent.
    """

    positive_x_transmissibility_multipliers: typing.Optional[
        NumberArray[OneDimension]
    ] = attrs.field(default=None)
    """Shape `(n_cells,)` MULTX multipliers. `None` when not supplied."""

    negative_x_transmissibility_multipliers: typing.Optional[
        NumberArray[OneDimension]
    ] = attrs.field(default=None)
    """Shape `(n_cells,)` MULTX- multipliers. `None` when not supplied."""

    positive_y_transmissibility_multipliers: typing.Optional[
        NumberArray[OneDimension]
    ] = attrs.field(default=None)
    """Shape `(n_cells,)` MULTY multipliers. `None` when not supplied."""

    negative_y_transmissibility_multipliers: typing.Optional[
        NumberArray[OneDimension]
    ] = attrs.field(default=None)
    """Shape `(n_cells,)` MULTY- multipliers. `None` when not supplied."""

    positive_z_transmissibility_multipliers: typing.Optional[
        NumberArray[OneDimension]
    ] = attrs.field(default=None)
    """Shape `(n_cells,)` MULTZ multipliers. `None` when not supplied."""

    negative_z_transmissibility_multipliers: typing.Optional[
        NumberArray[OneDimension]
    ] = attrs.field(default=None)
    """Shape `(n_cells,)` MULTZ- multipliers. `None` when not supplied."""

    cell_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Flat CSR data array: face indices per cell."""

    cell_face_offsets: IntArray[OneDimension] = attrs.field(init=False)
    """CSR offset array of length `n_cells + 1` for the cell-to-face map."""

    cell_neighbor_indices: IntArray[OneDimension] = attrs.field(init=False)
    """
    Flat CSR data array: neighbour cell indices per cell (interior faces only).
    """

    cell_neighbor_offsets: IntArray[OneDimension] = attrs.field(init=False)
    """CSR offset array of length `n_cells + 1` for the cell-to-neighbour map."""

    boundary_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Indices of all boundary faces."""

    interior_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Indices of all interior faces."""

    face_centroids: NumberArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_faces, 3)` - centroid of each face polygon."""

    face_areas: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_faces,)` - geometric area of each face."""

    face_unit_normals: NumberArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_faces, 3)` - unit outward normal from the owner cell."""

    cell_min_xyz: NumberArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_cells, 3)` - AABB minimum corner per cell."""

    cell_max_xyz: NumberArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_cells, 3)` - AABB maximum corner per cell."""

    bounding_box: tuple[float, float, float, float, float, float] = attrs.field(
        init=False
    )
    """Global AABB: `(x_min, x_max, y_min, y_max, z_min, z_max)`."""

    cell_length_x: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - AABB extent in x."""

    cell_length_y: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - AABB extent in y."""

    cell_length_z: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - AABB extent in z (thickness)."""

    cell_thickness: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - vertical thickness (alias for `cell_length_z`)."""

    cell_center_depths: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - depth of cell centroid (positive downward)."""

    cell_center_elevations: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - elevation of cell centroid (negation of depth)."""

    _spatial_index: typing.Optional[cKDTree] = attrs.field(init=False, default=None)
    """KD-tree on cell centroids for fast nearest-cell queries."""

    def __attrs_post_init__(self) -> None:
        self._validate_inputs()
        self._classify_faces()
        self._populate_defaults()
        self._build_cell_face_connectivity()
        self._build_cell_neighbor_connectivity()
        self._compute_face_geometry()
        self._compute_cell_geometry()
        self._compute_bounding_boxes()
        self._compute_derived_dimensions()
        self._build_spatial_index()

    def _validate_inputs(self) -> None:
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
                    f"`face_vertex_indices` contains index "
                    f"{int(self.face_vertex_indices.max())} which exceeds "
                    f"max valid index {max_valid_vertex}."
                )
        min_cell_index = int(self.face_cell_indices.min())
        if min_cell_index < -1:
            raise InvalidFaceConnectivityError(
                f"`face_cell_indices` contains negative cell index {min_cell_index}; "
                "only -1 is allowed (boundary sentinel)."
            )
        if self.nnc_cell_indices is not None:
            if self.nnc_connection_types is not None and len(
                self.nnc_connection_types
            ) != len(self.nnc_cell_indices):
                raise InvalidFaceConnectivityError(
                    f"`nnc_connection_types` length {len(self.nnc_connection_types)} does not match "
                    f"`nnc_cell_indices` length {len(self.nnc_cell_indices)}."
                )
            if self.nnc_transmissibilities is not None and len(
                self.nnc_transmissibilities
            ) != len(self.nnc_cell_indices):
                raise InvalidFaceConnectivityError(
                    f"`nnc_transmissibilities` length {len(self.nnc_transmissibilities)} "
                    f"does not match `nnc_cell_indices` length {len(self.nnc_cell_indices)}."
                )
            if self.nnc_cell_indices is not None and self.nnc_fault_indices is not None:
                n_nnc = len(self.nnc_cell_indices)
                for fault_name, nnc_indices in self.nnc_fault_indices.items():
                    if nnc_indices.max() >= n_nnc:
                        raise InvalidFaceConnectivityError(
                            f"`nnc_fault_indices[{fault_name!r}]` contains NNC index "
                            f"{int(nnc_indices.max())} which exceeds max valid index "
                            f"{n_nnc - 1}."
                        )

    def _classify_faces(self) -> None:
        owner_cells = self.face_cell_indices[:, 0]
        neighbour_cells = self.face_cell_indices[:, 1]
        boundary_mask = (owner_cells < 0) | (neighbour_cells < 0)
        interior_mask = ~boundary_mask
        object.__setattr__(
            self,
            "boundary_face_indices",
            np.where(boundary_mask)[0].astype(np.int32),
        )
        object.__setattr__(
            self,
            "interior_face_indices",
            np.where(interior_mask)[0].astype(np.int32),
        )

    def _populate_defaults(self) -> None:
        n_faces = self.face_cell_indices.shape[0]
        n_cells = int(self.face_cell_indices.max()) + 1

        if self.face_connection_types is None:
            face_connection_types = np.full(
                n_faces, ConnectionType.INTERIOR_FACE, dtype=np.int8
            )
            face_connection_types[self.boundary_face_indices] = int(
                ConnectionType.BOUNDARY_FACE
            )

            if self.fault_face_indices is not None:
                for face_indices in self.fault_face_indices.values():
                    boundary_fault_mask = (
                        self.face_cell_indices[face_indices, 0] < 0
                    ) | (self.face_cell_indices[face_indices, 1] < 0)
                    boundary_fault_faces = face_indices[boundary_fault_mask]
                    interior_fault_faces = face_indices[~boundary_fault_mask]

                    face_connection_types[interior_fault_faces] = int(
                        ConnectionType.INTERIOR_FAULT_FACE
                    )
                    face_connection_types[boundary_fault_faces] = int(
                        ConnectionType.BOUNDARY_FAULT_FACE
                    )
            object.__setattr__(self, "face_connection_types", face_connection_types)

        if self.cell_statuses is None:
            cell_statuses = np.full(n_cells, CellStatus.ACTIVE, dtype=np.int8)
            object.__setattr__(self, "cell_statuses", cell_statuses)

        if self.nnc_cell_indices is not None and self.nnc_connection_types is None:
            nnc_connection_types = np.full(
                len(self.nnc_cell_indices), int(ConnectionType.USER_NNC), dtype=np.int8
            )
            object.__setattr__(self, "nnc_connection_types", nnc_connection_types)

    def _build_cell_face_connectivity(self) -> None:
        n_cells = int(self.face_cell_indices.max()) + 1
        cell_face_lists: typing.List[typing.List[int]] = [[] for _ in range(n_cells)]
        for face_idx, (owner, neighbour) in enumerate(self.face_cell_indices):
            if owner >= 0:
                cell_face_lists[owner].append(face_idx)
            if neighbour >= 0:
                cell_face_lists[neighbour].append(face_idx)

        flat: typing.List[int] = []
        offsets: typing.List[int] = [0]
        for faces in cell_face_lists:
            flat.extend(faces)
            offsets.append(len(flat))
        object.__setattr__(self, "cell_face_indices", np.asarray(flat, dtype=np.int32))
        object.__setattr__(
            self, "cell_face_offsets", np.asarray(offsets, dtype=np.int32)
        )

    def _build_cell_neighbor_connectivity(self) -> None:
        n_cells = int(self.face_cell_indices.max()) + 1
        neighbor_sets: typing.List[typing.Set[int]] = [set() for _ in range(n_cells)]
        for owner, neighbour in self.face_cell_indices:
            if owner >= 0 and neighbour >= 0:
                neighbor_sets[owner].add(neighbour)
                neighbor_sets[neighbour].add(owner)

        flat: typing.List[int] = []
        offsets: typing.List[int] = [0]
        for neighbors in neighbor_sets:
            flat.extend(sorted(neighbors))
            offsets.append(len(flat))
        object.__setattr__(
            self, "cell_neighbor_indices", np.asarray(flat, dtype=np.int32)
        )
        object.__setattr__(
            self, "cell_neighbor_offsets", np.asarray(offsets, dtype=np.int32)
        )

    def _compute_face_geometry(self) -> None:
        face_centroids, face_areas, face_unit_normals = _compute_face_geometry(
            face_vertex_indices=self.face_vertex_indices,
            face_vertex_offsets=self.face_vertex_offsets,
            vertex_coordinates=self.vertex_coordinates,
        )
        object.__setattr__(self, "face_centroids", face_centroids)
        object.__setattr__(self, "face_areas", face_areas)
        object.__setattr__(self, "face_unit_normals", face_unit_normals)

    def _compute_cell_geometry(self) -> None:
        if self.cell_volumes is not None and self.cell_centroids is not None:
            return

        n_cells = int(self.face_cell_indices.max()) + 1
        cell_volumes, cell_centroids = _compute_cell_volumes_and_centroids(
            face_cell_indices=self.face_cell_indices,
            face_vertex_indices=self.face_vertex_indices,
            face_vertex_offsets=self.face_vertex_offsets,
            vertex_coordinates=self.vertex_coordinates,
            n_cells=n_cells,
        )
        invalid_mask = cell_volumes <= 0.0
        if invalid_mask.any():
            bad = np.where(invalid_mask)[0].tolist()
            raise InvalidVolumeError(
                f"Cells {bad[:20]}{'...' if len(bad) > 20 else ''} "
                f"have non-positive volumes. Check face winding order."
            )
        object.__setattr__(self, "cell_volumes", cell_volumes)
        object.__setattr__(self, "cell_centroids", cell_centroids)

    def _compute_bounding_boxes(self) -> None:
        n_cells = int(self.face_cell_indices.max()) + 1
        cell_min, cell_max = _compute_cell_bounding_boxes(
            face_cell_indices=self.face_cell_indices,
            face_vertex_indices=self.face_vertex_indices,
            face_vertex_offsets=self.face_vertex_offsets,
            vertex_coordinates=self.vertex_coordinates,
            n_cells=n_cells,
        )
        if self.cell_centroids is not None:
            no_face_mask = ~np.isfinite(cell_min).all(axis=1)
            if no_face_mask.any():
                cell_min[no_face_mask] = self.cell_centroids[no_face_mask]
                cell_max[no_face_mask] = self.cell_centroids[no_face_mask]

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
        delta = self.cell_max_xyz - self.cell_min_xyz
        object.__setattr__(self, "cell_length_x", delta[:, 0])
        object.__setattr__(self, "cell_length_y", delta[:, 1])
        object.__setattr__(self, "cell_length_z", delta[:, 2])
        object.__setattr__(self, "cell_thickness", delta[:, 2])

        assert self.cell_centroids is not None
        depths = self.cell_centroids[:, 2].copy()
        object.__setattr__(self, "cell_center_depths", depths)
        object.__setattr__(self, "cell_center_elevations", -depths)

    def _build_spatial_index(self) -> None:
        assert self.cell_centroids is not None
        object.__setattr__(self, "_spatial_index", cKDTree(self.cell_centroids))

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        assert self.cell_centroids is not None
        return self.cell_centroids.shape[0]

    @property
    def n_faces(self) -> int:
        """Total number of faces (boundary + interior)."""
        return self.face_cell_indices.shape[0]

    @property
    def n_vertices(self) -> int:
        """Total number of vertex points."""
        return self.vertex_coordinates.shape[0]

    @property
    def n_boundary_faces(self) -> int:
        """Number of boundary faces."""
        return len(self.boundary_face_indices)

    @property
    def n_interior_faces(self) -> int:
        """Number of interior faces."""
        return len(self.interior_face_indices)

    @property
    def n_nnc(self) -> int:
        """Number of non-neighbour connections. 0 when `nnc_cell_indices` is `None`."""
        if self.nnc_cell_indices is None:
            return 0
        return self.nnc_cell_indices.shape[0]

    @property
    def n_connections(self) -> int:
        """Total connections: `n_faces + n_nnc`."""
        return self.n_faces + self.n_nnc

    @property
    def n_faults(self) -> int:
        """Number of named faults (face-based and/or NNC-only). 0 when no fault data."""
        names: set[str] = set()
        if self.fault_face_indices is not None:
            names.update(self.fault_face_indices.keys())
        if self.nnc_fault_indices is not None:
            names.update(self.nnc_fault_indices.keys())
        return len(names)

    @property
    def has_transmissibility_multipliers(self) -> bool:
        """`True` if any directional `MULT*` array is present."""
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

    def is_cell_active(self, cell_index: int) -> bool:
        """
        Return whether a given cell is active.

        :param cell_index: Zero-based cell index.
        :returns: `True` if the cell has `CellStatus.ACTIVE`.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )

        assert self.cell_statuses is not None
        return bool(self.cell_statuses[cell_index])

    def get_nnc_type(self, nnc_index: int) -> ConnectionType:
        """
        Return the `ConnectionType` for a given NNC.

        :param nnc_index: Zero-based NNC index.
        :returns: `ConnectionType` enum value.
        :raises IndexError: If `nnc_index` is out of range.
        """
        assert self.nnc_connection_types is not None
        if nnc_index < 0 or nnc_index >= self.n_connections:
            raise IndexError(
                f"NNC index {nnc_index} is out of range [0, {self.n_nnc - 1}]."
            )
        return ConnectionType(int(self.nnc_connection_types[nnc_index]))

    def get_face_type(self, face_index: int) -> ConnectionType:
        """
        Return the `ConnectionType` for a given face.

        :param face_index: Zero-based face index.
        :returns: `ConnectionType` enum value.
        :raises IndexError: If `face_index` is out of range.
        """
        assert self.face_connection_types is not None
        if face_index < 0 or face_index >= self.n_faces:
            raise IndexError(
                f"Face index {face_index} is out of range [0, {self.n_faces - 1}]."
            )
        return ConnectionType(int(self.face_connection_types[face_index]))

    def get_cell_face_indices(self, cell_index: int) -> IntArray[OneDimension]:
        """
        Return the indices of all faces belonging to a given cell.

        :param cell_index: Zero-based cell index.
        :returns: 1-D array of face indices.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )

        start = self.cell_face_offsets[cell_index]
        end = self.cell_face_offsets[cell_index + 1]
        return typing.cast(IntArray[OneDimension], self.cell_face_indices[start:end])

    def get_cell_neighbor_indices(self, cell_index: int) -> IntArray[OneDimension]:
        """
        Return the indices of all face-adjacent neighbours of a given cell.

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
        return typing.cast(
            IntArray[OneDimension], self.cell_neighbor_indices[start:end]
        )

    def get_face_vertex_coordinates(
        self, face_index: int
    ) -> NumberArray[TwoDimensions]:
        """
        Return the vertex coordinates of a given face.

        :param face_index: Zero-based face index.
        :returns: Shape `(n_verts_for_face, 3)` coordinate array.
        """
        start = int(self.face_vertex_offsets[face_index])
        end = int(self.face_vertex_offsets[face_index + 1])
        return typing.cast(
            NumberArray[TwoDimensions],
            self.vertex_coordinates[self.face_vertex_indices[start:end]],
        )

    def get_face_cell_indices(self, face_index: int) -> IntArray[OneDimension]:
        """
        Return the indices of all cells that share a given face.

        :param face_index: Zero-based face index.
        :returns: 1-D array of cell indices (usually of length=2).
        :raises IndexError: If `face_index` is out of range.
        """
        if face_index < 0 or face_index >= self.n_faces:
            raise IndexError(
                f"Face index {face_index} is out of range [0, {self.n_faces - 1}]."
            )
        return typing.cast(IntArray[OneDimension], self.face_cell_indices[face_index])

    def get_face_normal_for_cell(
        self, face_index: int, cell_index: int
    ) -> NumberArray[OneDimension]:
        """
        Return the outward unit normal of a face relative to a specific cell.

        :param face_index: Zero-based face index.
        :param cell_index: Must be owner or neighbour of `face_index`.
        :returns: Shape `(3,)` unit normal pointing outward from `cell_index`.
        :raises ValidationError: If `cell_index` is not connected to `face_index`.
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
        """Return sorted indices of all cells that touch at least one boundary face."""
        owners = self.face_cell_indices[self.boundary_face_indices, 0]
        neighbours = self.face_cell_indices[self.boundary_face_indices, 1]
        all_boundary = np.concatenate([
            owners[owners >= 0],
            neighbours[neighbours >= 0],
        ])
        return typing.cast(
            IntArray[OneDimension], np.unique(all_boundary).astype(np.int32)
        )

    def get_interior_cell_indices(self) -> IntArray[OneDimension]:
        """Return sorted indices of all cells that have no boundary faces."""
        boundary_cells = self.get_boundary_cell_indices()
        all_cells = np.arange(self.n_cells, dtype=np.int32)
        return typing.cast(
            IntArray[OneDimension], np.setdiff1d(all_cells, boundary_cells)
        )

    def is_boundary_cell(self, cell_index: int) -> bool:
        """
        Return whether a given cell is adjacent to at least one boundary face.

        :param cell_index: Zero-based cell index.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )

        face_indices = self.get_cell_face_indices(cell_index)
        for face_idx in face_indices:
            if (
                int(self.face_cell_indices[face_idx, 0]) < 0
                or int(self.face_cell_indices[face_idx, 1]) < 0
            ):
                return True
        return False

    def is_boundary_face(self, face_index: int) -> bool:
        """
        Return whether a given face is adjacent to at least one boundary cell.

        :param face_index: Zero-based face index.
        :raises IndexError: If `face_index` is out of range.
        """
        face_type = self.get_face_type(face_index)
        return face_type in (
            ConnectionType.BOUNDARY_FACE,
            ConnectionType.BOUNDARY_FAULT_FACE,
        )

    def is_fault_face(self, face_index: int) -> bool:
        """
        Return whether a given face belongs to fault.

        :param face_index: Zero-based face index.
        :raises IndexError: If `face_index` is out of range.
        """
        face_type = self.get_face_type(face_index)
        return face_type in (
            ConnectionType.INTERIOR_FAULT_FACE,
            ConnectionType.BOUNDARY_FAULT_FACE,
        )

    def get_fault_face_indices(self, fault_name: str) -> IntArray[OneDimension]:
        """
        Return face indices for a named fault.

        :param fault_name: Fault name as declared in `FAULTS`.
        :raises KeyError: If `fault_name` is not found.
        :raises ValidationError: If no fault data is available.
        """
        if self.fault_face_indices is None:
            raise ValidationError(
                "No fault data available on this grid (fault_face_indices is None)."
            )

        if fault_name not in self.fault_face_indices:
            available = sorted(self.fault_face_indices.keys())
            raise KeyError(
                f"Fault {fault_name!r} not found. Available faults: {available}."
            )
        return self.fault_face_indices[fault_name]

    def get_fault_transmissibility_multiplier(self, fault_name: str) -> Number:
        """
        Return the transmissibility multiplier for a named fault.

        :param fault_name: Fault name as declared in `MULTFLT`.
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
                f"Fault {fault_name!r} not found in MULTFLT data. "
                f"Available: {available}."
            )
        return self.fault_transmissibility_multipliers[fault_name]

    def find_nearest_cell(self, x: Number, y: Number, z: Number) -> int:
        """
        Find the cell whose centroid is nearest to `(x, y, z)`.

        O(log n) via the pre-built KD-tree.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (positive downward).
        :returns: Zero-based index of the nearest cell.
        """
        _, cell_index = self._spatial_index.query([x, y, z])  # type: ignore
        return int(cell_index)

    def find_cells_in_radius(
        self, x: Number, y: Number, z: Number, radius: Number
    ) -> IntArray[OneDimension]:
        """
        Return all cell indices whose centroids fall within `radius` of a point.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (positive downward).
        :param radius: Search radius in grid length units.
        :returns: 1-D array of matching cell indices.
        """
        raw = self._spatial_index.query_ball_point([x, y, z], r=radius)  # type: ignore
        return typing.cast(IntArray[OneDimension], np.asarray(raw, dtype=np.int32))

    def compute_pore_volume(
        self,
        porosity: NumberOrArray[OneDimension],
        net_to_gross: NumberOrArray[OneDimension],
    ) -> NumberArray[OneDimension]:
        """
        Compute the pore volume for each cell.

        :param porosity: Scalar or shape `(n_cells,)` porosity values in `[0, 1]`.
        :param net_to_gross: Scalar or shape `(n_cells,)` NTG values.
        :returns: Pore volumes in the same units³ as `cell_volumes`.
        """
        assert self.cell_volumes is not None
        return typing.cast(
            NumberArray[OneDimension], porosity * net_to_gross * self.cell_volumes
        )

    def validate_geometry(self) -> None:
        """
        Validate that all computed geometry values are physically reasonable.

        :raises InvalidVolumeError: If any cell volume is <= 0.
        :raises InvalidFaceAreaError: If any face area is negative.
        :raises InvalidNormalVectorError: If any face normal deviates from unit length.
        """
        assert self.cell_volumes is not None
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

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        unit_system: typing.Optional[UnitSystem] = None,
        metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    ) -> Self:
        """
        Build a `Grid` from a parsed `DeckFile`.

        Detects the grid format (corner-point via `COORD`/`ZCORN` or Cartesian
        via `DX`/`DY`/`DZ`) and delegates to the appropriate factory.

        Corner-point grids are built via `make_corner_point_grid`; Cartesian
        grids via `make_cartesian_grid`. Both factories consume the GRDECL
        keywords already parsed in the deck.

        Delegates directly to `load_grdecl`, which handles both corner-point
        and Cartesian grids, fault processing, NNC resolution, and unit
        system detection from the deck.

        :param deck_file: Parsed `DeckFile` containing GRID-section keywords.
        :param unit_system: If provided, convert the grid to this unit system
            after loading. When `None`, the unit system declared in the deck
            is used as-is.
        :param metadata: Optional extra key/value pairs merged into
            `Grid.metadata`.
        :returns: `Grid` for the deck.
        :raises ValidationError: If no recognisable grid keywords are found.
        """
        from bores.grids.io.grdecl import load_grdecl

        return typing.cast(
            Self,
            load_grdecl(
                source=deck_file,
                unit_system=unit_system,
                metadata=metadata,
            ),
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `Grid` with all coordinates expressed in the target unit system.

        If `grid.unit_system == to` the original grid object is returned
        unchanged (no copy, no allocation).

        :param grid: Source grid. Must have a valid `unit_system` tag.
        :param target: Target `bores.typing.UnitSystem`.
        :returns: A new `Grid` with rescaled coordinates,
            or the original `grid` if already in the target system.

        Example:

        ```python
        from bores.grids.factories.cartesian import make_cartesian_grid
        from bores.typing import UnitSystem

        # Build a grid in field units (feet)
        grid_ft = make_cartesian_grid(
            nx=10, ny=10, nz=5,
            dx=328.084, dy=328.084, dz=16.4042,   # ≈ 100 m cells
            unit_system=UnitSystem.FIELD,
        )

        # Convert to metric (metres)
        grid_m = grid_ft.convert(UnitSystem.METRIC)
        assert grid_m.unit_system == UnitSystem.METRIC
        # cell volume should now be ≈ 100 * 100 * 5 = 50,000 m³
        ```
        """
        if self.unit_system == target:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        length_factor = factors["length"]
        volume_factor = length_factor**3
        # Rescale vertex coordinates only.
        # All other geometry is derived and will be recomputed on Grid initialization.
        vertex_coordinates = self.vertex_coordinates * length_factor
        cell_volumes = (
            self.cell_volumes * volume_factor if self.cell_volumes is not None else None
        )
        cell_centroids = (
            self.cell_centroids * length_factor
            if self.cell_centroids is not None
            else None
        )
        return attrs.evolve(
            self,
            vertex_coordinates=vertex_coordinates,
            cell_volumes=cell_volumes,
            cell_centroids=cell_centroids,
            unit_system=target,
        )

    def __repr__(self) -> str:
        bb = self.bounding_box
        fault_info = (
            f", n_faults={self.n_faults}"
            if (self.fault_face_indices or self.nnc_fault_indices)
            else ""
        )
        nnc_info = f", n_nnc={self.n_nnc}" if self.n_nnc > 0 else ""
        return (
            f"{self.__class__.__name__}("
            f"n_cells={self.n_cells}, "
            f"n_faces={self.n_faces}, "
            f"n_interior={self.n_interior_faces}, "
            f"n_boundary={self.n_boundary_faces}"
            f"{nnc_info}"
            f"{fault_info}, "
            f"unit_system={self.unit_system.value!r}, "
            f"bbox=({bb[0]:.2f}..{bb[1]:.2f}, {bb[2]:.2f}..{bb[3]:.2f}, {bb[4]:.2f}..{bb[5]:.2f})"
            f")"
        )
