import typing

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from bores.errors import InvalidFaceConnectivityError
from bores.grids.base import Grid
from bores.typing import FloatArray

VertexCoordinates3D: TypeAlias = FloatArray
"""Shape `(n_points, 3)` - 3-D (x, y, z) vertex coordinates."""

FaceVertexList: TypeAlias = typing.List[int]
"""Ordered list of vertex indices for a single face (CCW from owner)."""

CanonicalFaceKey: TypeAlias = typing.Tuple[int, ...]
"""Sorted tuple of vertex indices used as a face deduplication key."""

CellFaceTable: TypeAlias = typing.List[typing.List[int]]
"""Per-element-type local face definitions; each entry is a list of local
vertex indices wound CCW from outside (outward normal)."""


# Element face tables

#: Outward-pointing face definitions for standard element types.
#: Each value is a list of faces; each face is a list of *local* vertex
#: indices wound counter-clockwise when viewed from outside the cell.
ELEMENT_FACE_TABLES: typing.Dict[str, CellFaceTable] = {
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
    """
    Mutable record accumulating owner/neighbour information for one face.

    :param owner_cell_index: Index of the first cell that claimed this face.
    :param vertex_indices: Vertex index list wound CCW from the owner side.
    """

    __slots__ = ("owner_cell_index", "neighbour_cell_index", "vertex_indices")

    def __init__(self, owner_cell_index: int, vertex_indices: FaceVertexList) -> None:
        """Initialise with only an owner; neighbour is set later if face is interior.

        :param owner_cell_index: The cell that first registered this face.
        :param vertex_indices: Vertex winding from the owner's perspective.
        """
        self.owner_cell_index: int = owner_cell_index
        self.neighbour_cell_index: int = -1
        self.vertex_indices: FaceVertexList = vertex_indices


def build_csr_face_arrays(
    vertex_coordinates: VertexCoordinates3D,
    per_cell_face_vertex_lists: typing.List[typing.List[FaceVertexList]],
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
    face_registry: typing.Dict[CanonicalFaceKey, _FaceRecord] = {}

    for cell_index, cell_faces in enumerate(per_cell_face_vertex_lists):
        for face_vertex_indices in cell_faces:
            canonical_key: CanonicalFaceKey = tuple(sorted(face_vertex_indices))

            if canonical_key not in face_registry:
                # First time seen: this cell becomes the owner.
                # Store the winding as-is (CCW from owner's outside = outward from owner).
                face_registry[canonical_key] = _FaceRecord(
                    owner_cell_index=cell_index,
                    vertex_indices=list(face_vertex_indices),
                )
            else:
                record = face_registry[canonical_key]
                if record.neighbour_cell_index != -1:
                    raise InvalidFaceConnectivityError(
                        f"Face with canonical key {canonical_key} is shared by "
                        f"more than two cells (already has owner "
                        f"{record.owner_cell_index} and neighbour "
                        f"{record.neighbour_cell_index}; tried to add cell "
                        f"{cell_index})."
                    )
                record.neighbour_cell_index = cell_index

    # Flatten registry to CSR arrays
    flat_vertex_indices: typing.List[int] = []
    face_vertex_offsets_list: typing.List[int] = [0]
    face_cell_pairs: typing.List[typing.Tuple[int, int]] = []

    for record in face_registry.values():
        flat_vertex_indices.extend(record.vertex_indices)
        face_vertex_offsets_list.append(len(flat_vertex_indices))
        face_cell_pairs.append((record.owner_cell_index, record.neighbour_cell_index))

    face_vertex_indices_arr = np.asarray(flat_vertex_indices, dtype=np.int32)
    face_vertex_offsets_arr = np.asarray(face_vertex_offsets_list, dtype=np.int32)
    face_cell_indices_arr = np.asarray(face_cell_pairs, dtype=np.int32)

    return (
        vertex_coordinates,
        face_vertex_indices_arr,
        face_vertex_offsets_arr,
        face_cell_indices_arr,
    )


def assemble_grid(
    vertex_coordinates: VertexCoordinates3D,
    face_vertex_indices: npt.NDArray[np.int32],
    face_vertex_offsets: npt.NDArray[np.int32],
    face_cell_indices: npt.NDArray[np.int32],
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Construct a `bores.grids.base.Grid` from pre-built CSR arrays.

    :param vertex_coordinates: Shape `(n_vertices, 3)` float64 array.
    :param face_vertex_indices: Flat CSR vertex index data.
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
        metadata=metadata,
    )


@typing.runtime_checkable
class GridFactory(typing.Protocol):
    """
    Protocol implemented by all `Grid` factory callables.

    Any callable that accepts arbitrary arguments and returns a `Grid`
    instance satisfies this protocol.  Use it as a type hint wherever
    interchangeable grid builders are expected.
    """

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> Grid:
        """
        Construct and return a `Grid` instance.

        :param args: Positional arguments forwarded to the factory.
        :param kwargs: Keyword arguments forwarded to the factory.
        :returns: A fully initialised `Grid` object.
        """
        ...
