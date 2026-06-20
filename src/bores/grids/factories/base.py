import typing

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from bores.errors import InvalidFaceConnectivityError
from bores.grids.base import Grid
from bores.typing import FloatArray, TwoDimensions, UnitSystem

VertexCoordinates: TypeAlias = FloatArray[TwoDimensions]
"""Shape `(n_points, 3)` - 3-D (x, y, z) vertex coordinates."""

FaceVertexIndices: TypeAlias = typing.List[int]
"""Ordered list of vertex indices for a single face (CCW from owner)."""

FaceKey: TypeAlias = typing.Tuple[int, ...]
"""Sorted tuple of vertex indices used as a face deduplication key."""

ElementFaces: TypeAlias = typing.List[FaceVertexIndices]
"""
Per-element-type local face definitions; each entry is a list of local
vertex indices wound CCW from outside (outward normal).
"""


# Element face tables (VTK convention)

#: Outward-pointing face definitions for standard element types.
#: Each value is a list of faces; each face is a list of *local* vertex
#: indices wound counter-clockwise when viewed from outside the cell.
ELEMENT_FACES: typing.Dict[str, ElementFaces] = {
    # #####################################################################
    # Tetrahedron (4 vertices: v0 v1 v2 = base CCW from below, v3 = apex)
    # #####################################################################
    "tetra": [
        [0, 2, 1],  # base - outward normal points downward (away from apex)
        [0, 1, 3],  # front face opposite v2
        [1, 2, 3],  # right face opposite v0
        [0, 3, 2],  # left face opposite v1
    ],
    # #####################################################################
    # Hexahedron / brick (8 vertices, VTK ordering)
    # Bottom face: v0 v1 v2 v3 (z = z_bot); top face: v4 v5 v6 v7 (z = z_top)
    # v0=(x0,y0,z_bot) v1=(x1,y0,z_bot) v2=(x1,y1,z_bot) v3=(x0,y1,z_bot)
    # v4=(x0,y0,z_top) v5=(x1,y0,z_top) v6=(x1,y1,z_top) v7=(x0,y1,z_top)
    # #####################################################################
    "hexahedron": [
        [0, 3, 2, 1],  # bottom  - outward = −z
        [4, 5, 6, 7],  # top     - outward = +z
        [0, 1, 5, 4],  # front   - outward = −y
        [2, 3, 7, 6],  # back    - outward = +y
        [0, 4, 7, 3],  # left    - outward = −x
        [1, 2, 6, 5],  # right   - outward = +x
    ],
    # #####################################################################
    # Wedge / triangular prism (6 vertices)
    # Bottom triangle: v0 v1 v2 (CCW from below); top: v3 v4 v5
    # #####################################################################
    "wedge": [
        [0, 2, 1],  # bottom triangle - outward = −z
        [3, 4, 5],  # top triangle    - outward = +z
        [0, 1, 4, 3],  # lateral face 01 - outward away from opposite edge
        [1, 2, 5, 4],  # lateral face 12
        [0, 3, 5, 2],  # lateral face 20
    ],
    # #####################################################################
    # Pyramid (5 vertices: v0 v1 v2 v3 = base CCW from below, v4 = apex)
    # #####################################################################
    "pyramid": [
        [0, 3, 2, 1],  # base   - outward = −z
        [0, 1, 4],  # front  face
        [1, 2, 4],  # right  face
        [2, 3, 4],  # back   face
        [0, 4, 3],  # left   face
    ],
}

#: Mapping from VTK integer cell-type codes to element type names.
VTK_CELL_TYPE_NAMES: typing.Dict[int, str] = {
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}


class _FaceRecord:
    """Mutable record accumulating owner/neighbour information for one face."""

    __slots__ = ("owner_cell_index", "neighbour_cell_index", "face_vertex_indices")

    def __init__(
        self, owner_cell_index: int, face_vertex_indices: FaceVertexIndices
    ) -> None:
        """
        Initialise with only an owner; neighbour is set later if face is interior.

        :param owner_cell_index: The cell that first registered this face.
        :param face_vertex_indices: Face vertex indices winding from the owner's perspective.
        """
        self.owner_cell_index: int = owner_cell_index
        self.neighbour_cell_index: int = -1
        self.face_vertex_indices: FaceVertexIndices = face_vertex_indices


def build_csr_face_arrays(
    vertex_coordinates: VertexCoordinates,
    per_cell_face_vertex_lists: typing.List[typing.List[FaceVertexIndices]],
) -> typing.Tuple[
    FloatArray,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """
    Deduplicate faces across cells and build CSR face arrays.

    Each cell provides a list of faces; each face is a list of vertex
    indices wound CCW from **outside** that cell (outward-normal convention).
    Interior faces (shared by exactly two cells) appear once in each cell's
    list; the first cell to register a canonical key becomes the owner.

    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :param per_cell_face_vertex_lists: Outer list indexed by cell index;
        each element is a list of face vertex lists for that cell.
    :returns: Tuple `(vertex_coordinates, face_vertex_indices,
        face_vertex_offsets, face_cell_indices)` ready to pass to
        `bores.grids.base.Grid`.
    :raises InvalidFaceConnectivityError: If any face is shared by more
        than two cells.
    """
    face_registry: typing.Dict[FaceKey, _FaceRecord] = {}
    for cell_index, cell_faces in enumerate(per_cell_face_vertex_lists):
        for face_vertex_indices in cell_faces:
            key: FaceKey = tuple(sorted(face_vertex_indices))
            if key not in face_registry:
                # First time seen: this cell becomes the owner.
                # Store the winding as-is (CCW from owner's outside = outward from owner).
                face_registry[key] = _FaceRecord(
                    owner_cell_index=cell_index,
                    face_vertex_indices=face_vertex_indices,
                )
            else:
                record = face_registry[key]
                if record.neighbour_cell_index != -1:
                    raise InvalidFaceConnectivityError(
                        f"Face with canonical key {key} is shared by "
                        f"more than two cells (already has owner "
                        f"{record.owner_cell_index} and neighbour "
                        f"{record.neighbour_cell_index}; tried to add cell "
                        f"{cell_index})."
                    )
                record.neighbour_cell_index = cell_index

    # Flatten registry to CSR arrays
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


def assemble_grid(
    vertex_coordinates: VertexCoordinates,
    face_vertex_indices: npt.NDArray[np.int32],
    face_vertex_offsets: npt.NDArray[np.int32],
    face_cell_indices: npt.NDArray[np.int32],
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Construct a `bores.grids.base.Grid` from pre-built CSR arrays.

    :param vertex_coordinates: Shape `(n_vertices, 3)` float64 array.
    :param face_vertex_indices: Flat CSR vertex indices data.
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param face_cell_indices: Shape `(n_faces, 2)` owner/neighbour array.
    :param metadata: Optional free-form metadata dictionary.
    :returns: A fully initialised `bores.grids.base.Grid`.
    """
    return Grid(
        vertex_coordinates=np.asarray(vertex_coordinates, dtype=np.float64),
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
    )


class FaultRecord(typing.NamedTuple):
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
    :param face_direction: Eclipse face direction string, one of
        `'I'`, `'I-'`, `'J'`, `'J-'`, `'K'`, `'K-'`.
    """

    name: str
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    face_direction: str
