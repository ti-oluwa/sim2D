import typing

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from bores.errors import InvalidPointArrayError, ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import (
    ELEMENT_FACE_TABLES,
    assemble_grid,
    build_csr_face_arrays,
)
from bores.typing import FloatArray, TwoDimensions, UnitSystem

__all__ = ["make_tetrahedral_grid"]

VertexCoordinates: TypeAlias = FloatArray[TwoDimensions]
"""Shape `(n_points, 3)` — 3-D (x, y, z) vertex coordinates."""

FaceVertexList: TypeAlias = typing.List[int]
"""Ordered list of vertex indices for a single face (CCW from owner)."""


def make_tetrahedral_grid(
    *,
    vertex_coordinates: VertexCoordinates,
    element_vertex_indices: npt.ArrayLike,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[dict] = None,
) -> Grid:
    """
    Factory for unstructured tetrahedral meshes.

    Accepts a point array and an element connectivity array, extracts all
    unique triangular faces, determines owner/neighbour relationships, and
    assembles a `bores.grids.base.Grid`.

    Typical input sources include `meshio`, `gmsh`, or `pyvista`
    tetrahedral mesh objects.

    Example usage:

    ```python
    grid = make_tetrahedral_grid(
        vertex_coordinates=points,      # (N, 3) float array
        element_vertex_indices=cells,   # (M, 4) int array
    )
    ```

    Builds a tetrahedral mesh grid.

    :param vertex_coordinates: Shape `(n_vertices, 3)` float64 point array.
    :param element_vertex_indices: Shape `(n_elements, 4)` int array; each
        row lists the 4 vertex indices of one tetrahedron. Vertices should
        be ordered so that the tet has positive orientation (det > 0), i.e.
        v0 v1 v2 form a base with counter-clockwise winding when viewed from
        outside, and v3 is the apex above that base.
    :param metadata: Optional metadata dictionary.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises InvalidPointArrayError: If `vertex_coordinates` is not `(N, 3)`.
    :raises InvalidFaceConnectivityError: If elements reference out-of-range
        vertex indices.
    """
    points = np.asarray(vertex_coordinates, dtype=np.float64)
    elements = np.asarray(element_vertex_indices, dtype=np.int32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise InvalidPointArrayError(
            f"vertex_coordinates must be shape (n_vertices, 3); got {points.shape!r}."
        )
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValidationError(
            f"element_vertex_indices must be shape (n_elements, 4); got {elements.shape!r}."
        )

    per_cell_face_vertex_lists = _extract_tetrahedron_faces(elements)
    _, face_vertex_indices, face_vertex_offsets, face_cell_indices = (
        build_csr_face_arrays(points, per_cell_face_vertex_lists)
    )
    return assemble_grid(
        points,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
    )


def _extract_tetrahedron_faces(
    elements: npt.NDArray[np.int32],
) -> typing.List[typing.List[FaceVertexList]]:
    """
    Extract outward-facing triangular faces for every tetrahedron.

    Uses the face table for `"tetra"` from `ELEMENT_FACE_TABLES`.
    Each face's local vertex indices are mapped to global vertex indices
    using the element connectivity.

    :param elements: Shape `(n_elements, 4)` element connectivity array.
    :returns: Per-cell face vertex lists; outer list is indexed by cell
        (element) index, inner list contains 4 triangular faces each given
        as a list of 3 global vertex indices.
    """
    tetrahedron_face_table = ELEMENT_FACE_TABLES["tetra"]  # 4 faces × 3 local verts
    per_cell_face_vertex_lists: typing.List[typing.List[FaceVertexList]] = []

    for global_vert_indices in elements:
        cell_faces: typing.List[FaceVertexList] = [
            [int(global_vert_indices[local_v]) for local_v in face_local]
            for face_local in tetrahedron_face_table
        ]
        per_cell_face_vertex_lists.append(cell_faces)

    return per_cell_face_vertex_lists
