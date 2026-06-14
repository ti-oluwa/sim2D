"""
Face-based unstructured polyhedral grid implementation for reservoir simulation.

**Uses**:

- Face-centric finite-volume topology stored in CSR (compressed-sparse-row) format.
- Fully unstructured polyhedral cells (triangles, quads, arbitrary polygons).
- Numba-accelerated geometry kernels (Newell normals, divergence-theorem volumes,
  simultaneous centroid accumulation).
"""

import enum
import typing

import attrs
import numba
import numpy as np
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
)

__all__ = ["Grid"]


class FaceType(enum.IntEnum):
    """Topological classification of a grid face."""

    INTERIOR = 0
    BOUNDARY = 1
    FAULT = 2
    PINCHOUT = 3
    NON_NEIGHBOR_CONNECTION = 4


class FaceStatus(enum.IntEnum):
    """Activation status of a grid face (e.g. closed faults)."""

    ACTIVE = 1
    INACTIVE = 2


class CellStatus(enum.IntEnum):
    """Activation status of a grid cell (e.g. pinched-out cells)."""

    ACTIVE = 1
    INACTIVE = 2


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
    pre-computed face centroid. The normal magnitude equals twice the face area,
    so `area = ||n|| / 2`.

    :param face_vertex_indices: Flat CSR data array of vertex indices (all faces).
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :returns: Tuple `(face_centroids, face_areas, face_unit_normals)` each of
        shape `(n_faces, 3)`, `(n_faces,)`, `(n_faces, 3)` respectively.
    """
    n_faces = face_vertex_offsets.shape[0] - 1
    face_centroids = np.zeros((n_faces, 3), dtype=np.float64)
    face_unit_normals = np.zeros((n_faces, 3), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)

    for face_idx in numba.prange(n_faces):  # type: ignore[attr-defined]
        start = face_vertex_offsets[face_idx]
        end = face_vertex_offsets[face_idx + 1]
        n_verts = end - start

        # Centroid: simple vertex average (exact for planar convex polygons)
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

        # Newell's method for outward normal and area
        # n_x += (y_i - y_{i+1}) * (z_i + z_{i+1})
        # n_y += (z_i - z_{i+1}) * (x_i + x_{i+1})
        # n_z += (x_i - x_{i+1}) * (y_i + y_{i+1})
        # ||n|| = 2 * face_area for any planar polygon.
        nx = 0.0
        ny = 0.0
        nz = 0.0
        for local_idx in range(n_verts):
            idx_a = face_vertex_indices[start + local_idx]
            idx_b = face_vertex_indices[start + (local_idx + 1) % n_verts]

            ax = vertex_coordinates[idx_a, 0]
            ay = vertex_coordinates[idx_a, 1]
            az = vertex_coordinates[idx_a, 2]
            bx = vertex_coordinates[idx_b, 0]
            by = vertex_coordinates[idx_b, 1]
            bz = vertex_coordinates[idx_b, 2]

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
    owner/neighbour cell.

    **Sign convention**:

    Face vertices are wound counter-clockwise from the **owner** (c1) side, so
    the Newell normal points from c1 toward c2 (outward for c1). The signed tet
    volume is positive when computed relative to the c1 centroid, and negative
    when computed relative to the c2 centroid. Therefore:

    - `cell_volumes[owner] += signed_tet_vol`
    - `cell_volumes[neighbour] -= signed_tet_vol`

    Both accumulate positive contributions. The centroid is computed
    simultaneously as the volume-weighted average of tet barycentres.

    :param face_cell_indices: Shape `(n_faces, 2)` - `(owner, neighbour)` per face.
    :param face_vertex_indices: Flat CSR vertex index data array.
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :param n_cells: Total number of cells in the grid.
    :returns: Tuple `(cell_volumes, cell_centroids)` of shapes `(n_cells,)`
              and `(n_cells, 3)`.
    """
    n_faces = face_cell_indices.shape[0]

    cell_volumes = np.zeros(n_cells, dtype=np.float64)
    # Accumulator for volume-weighted centroid sum: centroid = accum / volume
    centroid_accumulators = np.zeros((n_cells, 3), dtype=np.float64)

    for face_idx in range(n_faces):
        owner_cell = face_cell_indices[face_idx, 0]
        neighbour_cell = face_cell_indices[face_idx, 1]

        start = face_vertex_offsets[face_idx]
        end = face_vertex_offsets[face_idx + 1]

        # Fan triangulation anchored at the first face vertex
        apex = vertex_coordinates[face_vertex_indices[start]]

        # Process owner (sign=+1) then neighbour (sign=-1)
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

                # Signed tet volume: (1/6) * apex . (v1 x v2) using the
                # scalar triple product expanded about the origin (all verts
                # already in world coords - ref is subtracted implicitly via
                # the standard divergence-theorem formula):
                #   vol = (apex . ((v1 - apex) x (v2 - apex))) / 6
                # Equivalently (Kahan's form for numerical stability):
                ax = apex[0]
                ay = apex[1]
                az = apex[2]
                bx = v1[0]
                by = v1[1]
                bz = v1[2]
                cx = v2[0]
                cy = v2[1]
                cz = v2[2]

                signed_tet_vol = (
                    ax * (by * cz - bz * cy)
                    + ay * (bz * cx - bx * cz)
                    + az * (bx * cy - by * cx)
                ) / 6.0

                cell_volumes[cell_idx] += sign * signed_tet_vol

                # Tet barycentre (origin + 3 face verts) / 4, weighted by tet vol
                bary_x = (ax + bx + cx) / 4.0
                bary_y = (ay + by + cy) / 4.0
                bary_z = (az + bz + cz) / 4.0

                weighted_vol = sign * signed_tet_vol
                centroid_accumulators[cell_idx, 0] += weighted_vol * bary_x
                centroid_accumulators[cell_idx, 1] += weighted_vol * bary_y
                centroid_accumulators[cell_idx, 2] += weighted_vol * bary_z

    # Finalise centroids: divide accumulated weighted sums by total volume
    cell_centroids = np.zeros((n_cells, 3), dtype=np.float64)
    for cell_idx in range(n_cells):
        vol = cell_volumes[cell_idx]
        if abs(vol) > 0.0:
            cell_centroids[cell_idx, 0] = centroid_accumulators[cell_idx, 0] / vol
            cell_centroids[cell_idx, 1] = centroid_accumulators[cell_idx, 1] / vol
            cell_centroids[cell_idx, 2] = centroid_accumulators[cell_idx, 2] / vol

    return cell_volumes, cell_centroids


@attrs.define(frozen=True, slots=True, kw_only=True)
class Grid:
    """
    Immutable face-based unstructured polyhedral grid.

    All topology and geometry is computed once during construction and stored as
    read-only NumPy arrays.  The grid is fully unstructured: cells may be
    arbitrary convex polyhedra; faces may be arbitrary planar polygons.

    **Construction**:

    Supply the three mandatory arrays and any optional metadata; all derived
    arrays (connectivity, geometry, bounding boxes) are computed automatically.

    **Raises**:

    `InvalidPointArrayError`:
        If `vertex_coordinates` is not a 2-D `(n_vertices, 3)` array.
    `InvalidFaceConnectivityError`:
        If `face_cell_indices` is not a 2-D `(n_faces, 2)` array, or if
        `face_vertex_offsets` does not start at 0 or is inconsistent with
        `face_vertex_indices`.
    `InvalidVolumeError`:
        If any cell ends up with a non-positive volume after construction.
    """

    vertex_coordinates: FloatArray[TwoDimensions]
    """
    Shape `(n_vertices, 3)` - world (x, y, z) coordinates of every vertex.

    The z-axis is positive downward (reservoir depth convention).
    """

    face_vertex_indices: IntArray[OneDimension]
    """
    Flat CSR data array: concatenated vertex index lists for all faces.

    Face *f* uses `face_vertex_indices[face_vertex_offsets[f]:face_vertex_offsets[f+1]]`.
    Vertices are wound counter-clockwise when viewed from the **owner** cell side.
    """

    face_vertex_offsets: IntArray[OneDimension]
    """CSR offset array of length `n_faces + 1`.

    `face_vertex_offsets[0]` must be 0; `face_vertex_offsets[-1]` must equal
    `len(face_vertex_indices)`.
    """

    face_cell_indices: IntArray[TwoDimensions]
    """
    Shape `(n_faces, 2)` - `(owner_cell_index, neighbour_cell_index)` per face.

    Boundary faces have `neighbour_cell_index == -1`. Interior faces have
    both indices `>= 0`. The owner cell is the one from whose perspective the
    face vertices are wound counter-clockwise.
    """

    index_dtype: np.dtype = attrs.field(default=np.dtype(np.int32))
    """NumPy integer dtype used for all connectivity index arrays."""

    floating_dtype: np.dtype = attrs.field(default=np.dtype(np.float64))
    """NumPy floating-point dtype used for all coordinate and geometry arrays."""

    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = attrs.field(
        default=None
    )
    """Optional free-form metadata mapping (e.g. units, CRS, source filename)."""

    cell_statuses: typing.Optional[IntArray[OneDimension]] = attrs.field(default=None)
    """Shape `(n_cells,)` - per-cell `CellStatus` flags (optional)."""

    face_types: typing.Optional[IntArray[OneDimension]] = attrs.field(default=None)
    """Shape `(n_faces,)` - per-face `FaceType` classification (optional)."""

    face_statuses: typing.Optional[IntArray[OneDimension]] = attrs.field(default=None)
    """Shape `(n_faces,)` - per-face `FaceStatus` flags (optional)."""

    # Derived topology
    cell_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Flat CSR data array: concatenated face index lists for all cells.

    Cell *c* uses `cell_face_indices[cell_face_offsets[c]:cell_face_offsets[c+1]]`.
    """

    cell_face_offsets: IntArray[OneDimension] = attrs.field(init=False)
    """CSR offset array of length `n_cells + 1` for the cell to face map."""

    cell_neighbor_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Flat CSR data array: concatenated neighbour cell index lists for all cells.

    Cell *c* uses `cell_neighbor_indices[cell_neighbor_offsets[c]:cell_neighbor_offsets[c+1]]`.
    """

    cell_neighbor_offsets: IntArray[OneDimension] = attrs.field(init=False)
    """CSR offset array of length `n_cells + 1` for the cell to neighbour map."""

    boundary_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Indices of all boundary faces (faces with `neighbour_cell == -1`)."""

    interior_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Indices of all interior faces (faces with both owner and neighbour cells)."""

    # Derived geometry
    face_centroids: FloatArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_faces, 3)` - (x, y, z) centroid of each face polygon."""

    face_areas: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_faces,)` - geometric area of each face in grid units²."""

    face_unit_normals: FloatArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_faces, 3)` - unit outward normal from the owner cell for each face."""

    cell_centroids: FloatArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_cells, 3)` - volume-weighted (x, y, z) centroid of each cell."""

    cell_volumes: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - bulk geometric volume of each cell in grid units³."""

    cell_min_xyz: FloatArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_cells, 3)` - axis-aligned bounding box minimum corner per cell."""

    cell_max_xyz: FloatArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_cells, 3)` - axis-aligned bounding box maximum corner per cell."""

    cell_length_x: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - bounding-box extent in the x direction per cell."""

    cell_length_y: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - bounding-box extent in the y direction per cell."""

    cell_length_z: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - bounding-box extent in the z direction per cell (thickness)."""

    cell_thickness: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - vertical thickness of each cell (alias for `cell_length_z`)."""

    cell_center_depths: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - depth of each cell centroid (positive downward = centroid z)."""

    cell_center_elevations: FloatArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - elevation of each cell centroid (positive upward = −depth)."""

    _spatial_index: typing.Optional[cKDTree] = attrs.field(init=False, default=None)
    """KD-tree built on cell centroids for fast spatial lookup. Internal use only."""

    def __attrs_post_init__(self) -> None:
        """Validate inputs and compute all derived topology and geometry.

        Called automatically by `attrs` after `__init__`.  Raises grid-specific
        errors on any validation failure.
        """
        self._validate_inputs()
        self._classify_faces()
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

        :raises `InvalidPointArrayError`: If `vertex_coordinates` is not `(N, 3)`.
        :raises `InvalidFaceConnectivityError`: If face connectivity arrays are malformed.
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
                f"`face_vertex_offsets` length must be n_faces + 1 = {expected_n_faces + 1}; "
                f"got {self.face_vertex_offsets.shape[0]}."
            )
        if int(self.face_vertex_offsets[-1]) != len(self.face_vertex_indices):
            raise InvalidFaceConnectivityError(
                f"face_vertex_offsets[-1] = {self.face_vertex_offsets[-1]} does not match "
                f"len(face_vertex_indices) = {len(self.face_vertex_indices)}."
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

        A boundary face has `neighbour_cell == -1`; an interior face has both
        owner and neighbour cells `>= 0`.
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

    def _build_cell_face_connectivity(self) -> None:
        """
        Build CSR cell to face adjacency lists from `face_cell_indices`.

        Each cell accumulates the indices of every face that touches it (as
        either owner or neighbour).  The result is stored in
        `cell_face_indices` and `cell_face_offsets`.
        """
        n_cells = int(self.face_cell_indices.max()) + 1

        # Build per-cell face lists
        cell_face_lists: list[list[int]] = [[] for _ in range(n_cells)]
        for face_idx, (owner, neighbour) in enumerate(self.face_cell_indices):
            if owner >= 0:
                cell_face_lists[owner].append(face_idx)
            if neighbour >= 0:
                cell_face_lists[neighbour].append(face_idx)

        # Flatten to CSR
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
        Build CSR cell to neighbour adjacency lists from face_cell_indices.

        Two cells are neighbours if they share an interior face.  Boundary faces
        do not contribute neighbours.  The result is stored in
        `cell_neighbor_indices` and `cell_neighbor_offsets`.
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
            flat_neighbor_indices.extend(sorted(neighbors))  # sorted for determinism
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
        """Compute face centroids, areas, and normals."""
        face_centroids, face_areas, face_unit_normals = _compute_face_geometry(
            self.face_vertex_indices,
            self.face_vertex_offsets,
            self.vertex_coordinates,
        )
        object.__setattr__(self, "face_centroids", face_centroids)
        object.__setattr__(self, "face_areas", face_areas)
        object.__setattr__(self, "face_unit_normals", face_unit_normals)

    def _compute_cell_geometry(self) -> None:
        """
        Compute cell volumes and volume-weighted centroids.

        :raises `InvalidVolumeError`: If any cell has a non-positive volume.
        """
        n_cells = int(self.face_cell_indices.max()) + 1
        cell_volumes, cell_centroids = _compute_cell_volumes_and_centroids(
            self.face_cell_indices,
            self.face_vertex_indices,
            self.face_vertex_offsets,
            self.vertex_coordinates,
            n_cells,
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
        Compute per-cell axis-aligned bounding boxes from face vertex coordinates.

        Each cell's bounding box is derived from the min/max of all vertices
        belonging to its faces.
        """
        n_cells = len(self.cell_centroids)

        cell_min = np.full((n_cells, 3), np.inf, dtype=self.floating_dtype)
        cell_max = np.full((n_cells, 3), -np.inf, dtype=self.floating_dtype)

        for face_idx, (owner, neighbour) in enumerate(self.face_cell_indices):
            start = self.face_vertex_offsets[face_idx]
            end = self.face_vertex_offsets[face_idx + 1]
            face_vertex_coords = self.vertex_coordinates[
                self.face_vertex_indices[start:end]
            ]
            face_min = face_vertex_coords.min(axis=0)
            face_max = face_vertex_coords.max(axis=0)

            for cell_idx in (owner, neighbour):
                if cell_idx >= 0:
                    np.minimum(cell_min[cell_idx], face_min, out=cell_min[cell_idx])
                    np.maximum(cell_max[cell_idx], face_max, out=cell_max[cell_idx])

        object.__setattr__(self, "cell_min_xyz", cell_min)
        object.__setattr__(self, "cell_max_xyz", cell_max)

    def _compute_derived_dimensions(self) -> None:
        """
        Derive per-cell scalar dimensions from the axis-aligned bounding boxes.

        Computes `cell_length_x/y/z`, `cell_thickness`, `cell_center_depths`,
        and `cell_center_elevations`.

        Note: depth is positive downward (z-axis convention); elevation is the
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
        """
        Total number of cells in the grid.

        :returns: Integer count of grid cells.
        """
        return self.cell_centroids.shape[0]

    @property
    def n_faces(self) -> int:
        """
        Total number of faces (boundary + interior) in the grid.

        :returns: Integer count of grid faces.
        """
        return self.face_cell_indices.shape[0]

    @property
    def n_vertices(self) -> int:
        """
        Total number of vertex points in the grid.

        :returns: Integer count of grid vertices.
        """
        return self.vertex_coordinates.shape[0]

    @property
    def n_boundary_faces(self) -> int:
        """
        Number of boundary faces (faces on the outer hull of the domain).

        :returns: Integer count of boundary faces.
        """
        return len(self.boundary_face_indices)

    @property
    def n_interior_faces(self) -> int:
        """
        Number of interior faces (faces shared between two cells).

        :returns: Integer count of interior faces.
        """
        return len(self.interior_face_indices)

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

        The stored normal points outward from the owner cell. For the neighbour
        cell the normal is reversed.

        :param face_index: Zero-based face index.
        :param cell_index: Zero-based cell index (must be owner or neighbour of the face).
        :returns: Shape `(3,)` unit normal vector pointing outward from `cell_index`.
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
        """
        Return the indices of all cells that touch at least one boundary face.

        :returns: 1-D sorted array of boundary-adjacent cell indices.
        """
        boundary_owners = self.face_cell_indices[self.boundary_face_indices, 0]
        boundary_neighbours = self.face_cell_indices[self.boundary_face_indices, 1]

        # Only one of owner/neighbour is a real cell on a boundary face
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

    def find_nearest_cell(self, x: float, y: float, z: float) -> int:
        """
        Find the cell whose centroid is nearest to the given (x, y, z) point.

        Uses a pre-built KD-tree for O(log n) lookup.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (depth, positive downward).
        :returns: Zero-based index of the nearest cell.
        """
        _, cell_index = self._spatial_index.query([x, y, z])  # type: ignore[union-attr]
        return int(cell_index)

    def find_cells_in_radius(
        self, x: float, y: float, z: float, radius: float
    ) -> IntArray[OneDimension]:
        """
        Return all cell indices whose centroids fall within `radius` of a point.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (depth, positive downward).
        :param radius: Search radius in grid length units.
        :returns: 1-D array of matching cell indices (unsorted).
        """
        raw_indices = self._spatial_index.query_ball_point([x, y, z], r=radius)  # type: ignore[union-attr]
        return np.asarray(raw_indices, dtype=self.index_dtype)

    def compute_pore_volume(
        self, porosity: NumberOrArray[OneDimension]
    ) -> FloatArray[OneDimension]:
        """
        Compute the pore volume for each cell given a porosity field.

        :param porosity: Scalar or shape `(n_cells,)` array of porosity values
            (dimensionless, in `[0, 1]`).
        :returns: Pore volumes in the same units³ as `cell_volumes`, broadcast
            against `porosity`.
        """
        return porosity * self.cell_volumes

    def validate_geometry(self) -> None:
        """
        Validate that all computed geometry values are physically reasonable.

        Checks that all cell volumes are strictly positive, all face areas are
        non-negative, and all face unit normals have unit magnitude.

        :raises InvalidVolumeError: If any cell volume is `<= 0`.
        :raises InvalidFaceAreaError: If any face area is negative.
        :raises InvalidNormalVectorError: If any face normal deviates from unit length
            by more than a loose tolerance.
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
        # Faces with zero area legitimately have zero-magnitude normals - allow those.
        active_mask = self.face_areas > _GEOMETRY_TOLERANCE
        if active_mask.any():
            deviation = np.abs(normal_magnitudes[active_mask] - 1.0)
            if (deviation > 1e-10).any():
                raise InvalidNormalVectorError(
                    "One or more face unit normals do not have unit magnitude "
                    f"(max deviation = {deviation.max():.3e})."
                )
