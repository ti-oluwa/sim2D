import typing

import numpy as np
from typing_extensions import TypeAlias

from bores.errors import InvalidPointArrayError, ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import (
    ELEMENT_FACE_TABLES,
    VTK_CELL_TYPE_NAMES,
    assemble_grid,
    build_csr_face_arrays,
)
from bores.typing import FloatArray, TwoDimensions

__all__ = ["make_polyhedral_grid"]

VertexCoordinate: TypeAlias = FloatArray[TwoDimensions]
"""Shape `(n_points, 3)` — 3-D (x, y, z) vertex coordinates."""

FaceVertexList: TypeAlias = typing.List[int]
"""Ordered list of vertex indices for a single face (CCW from owner)."""

CellFaceTable: TypeAlias = typing.List[typing.List[int]]
"""Per-element-type local face definitions; each entry is a list of local
vertex indices wound CCW from outside (outward normal)."""


def make_polyhedral_grid(
    *,
    vertex_coordinates: VertexCoordinate,
    cell_blocks: typing.Sequence[typing.Dict[str, typing.Any]],
    custom_cell_faces: typing.Optional[typing.Dict[str, CellFaceTable]] = None,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Factory for general mixed-element polyhedral meshes.

    This is the **most general** factory and is used internally by the IO
    layer to convert VTK, meshio, and other mesh formats into a
    `bores.grids.base.Grid`.  It supports any combination of
    tetrahedra, hexahedra, wedges, pyramids, and custom polyhedral cells.

    For custom cell types (not in `ELEMENT_FACE_TABLES`), callers must
    supply the face definitions explicitly via `custom_cell_faces`.

    Example usage:

    ```python

    # From meshio-style cell blocks:
    grid = make_polyhedral_grid(
        vertex_coordinates=mesh.points,
        cell_blocks=[
            {"cell_type": "hexahedron", "connectivity": hex_cells},
            {"cell_type": "tetra", "connectivity": tet_cells},
        ],
    )

    # From VTK cell arrays:
    grid = make_polyhedral_grid(
        vertex_coordinates=points,
        cell_blocks=[
            {"vtk_type": 12, "connectivity": vtk_hex_cells},
            {"vtk_type": 10, "connectivity": vtk_tet_cells},
        ],
    )
    ```

    Builds a grid from a sequence of mixed-element cell blocks.

    :param vertex_coordinates: Shape `(n_vertices, 3)` float64 point array.
    :param cell_blocks: List of dictionaries, one per element block.
        Each dict must contain a `"connectivity"` key whose value is an
        array-like of shape `(n_cells_in_block, n_verts_per_cell)`, plus one of:

        - `"cell_type"`: string name matching a key in
            `ELEMENT_FACE_TABLES` or `custom_cell_faces` (e.g.
            `"hexahedron"`, `"tetra"`).
        - `"vtk_type"`: integer VTK cell type code (e.g. `12` for hex).

    :param custom_cell_faces: Optional mapping from cell-type name to face
        table, extending or overriding `ELEMENT_FACE_TABLES`.  
        Use this for non-standard polyhedral element types.
    :param metadata: Optional metadata dictionary.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises ValidationError: If a cell block contains an unrecognised element type.
    :raises InvalidPointArrayError: If `vertex_coordinates` is not `(N, 3)`.
    """
    pts = np.asarray(vertex_coordinates, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise InvalidPointArrayError(
            f"vertex_coordinates must be shape (n_vertices, 3); got {pts.shape!r}."
        )

    combined_face_table = dict(ELEMENT_FACE_TABLES)
    if custom_cell_faces:
        combined_face_table.update(custom_cell_faces)

    all_per_cell_faces: typing.List[typing.List[FaceVertexList]] = []

    for block_index, block in enumerate(cell_blocks):
        cell_type_name = _resolve_cell_type_name(
            block, combined_face_table, block_index
        )
        face_table = combined_face_table[cell_type_name]
        connectivity = np.asarray(block["connectivity"], dtype=np.int32)

        if connectivity.ndim != 2:
            raise ValidationError(
                f"Block {block_index}: connectivity must be 2-D; "
                f"got ndim={connectivity.ndim}."
            )

        expected_n_verts = max(max(face) for face in face_table) + 1
        if connectivity.shape[1] < expected_n_verts:
            raise ValidationError(
                f"Block {block_index} (type '{cell_type_name}'): "
                f"connectivity has {connectivity.shape[1]} vertices per cell but "
                f"face table requires at least {expected_n_verts}."
            )

        for global_vert_indices in connectivity:
            cell_faces: typing.List[FaceVertexList] = [
                [int(global_vert_indices[local_v]) for local_v in face_local]
                for face_local in face_table
            ]
            all_per_cell_faces.append(cell_faces)

    if not all_per_cell_faces:
        raise ValidationError("No cells found across all provided cell blocks.")

    _, face_vertex_indices, face_vertex_offsets, face_cell_indices = (
        build_csr_face_arrays(pts, all_per_cell_faces)
    )
    return assemble_grid(
        pts,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
        metadata=metadata,
    )


def _resolve_cell_type_name(
    block: typing.Dict[str, typing.Any],
    combined_face_table: typing.Dict[str, CellFaceTable],
    block_index: int,
) -> str:
    """
    Resolve a cell block's element type to a string name.

    Accepts either `"cell_type"` (string) or `"vtk_type"` (integer)
    keys in the block dictionary.

    :param block: Cell block dictionary from the caller.
    :param combined_face_table: Merged face table (built-in + custom).
    :param block_index: Index of this block (for error messages).
    :returns: Resolved element type name string.
    :raises ValidationError: If neither key is present or the type is
        unrecognised.
    """
    if "cell_type" in block:
        name = block["cell_type"]
        if name not in combined_face_table:
            raise ValidationError(
                f"Block {block_index}: unrecognised cell_type '{name}'. "
                f"Known types: {sorted(combined_face_table.keys())}."
            )
        return name
    elif "vtk_type" in block:
        vtk_code = int(block["vtk_type"])
        if vtk_code not in VTK_CELL_TYPE_NAMES:
            raise ValidationError(
                f"Block {block_index}: unrecognised vtk_type {vtk_code}. "
                f"Supported codes: {sorted(VTK_CELL_TYPE_NAMES.keys())}."
            )
        name = VTK_CELL_TYPE_NAMES[vtk_code]
        if name not in combined_face_table:
            raise ValidationError(
                f"Block {block_index}: VTK type {vtk_code} maps to '{name}' "
                f"but no face table is defined for it."
            )
        return name

    raise ValidationError(
        f"Block {block_index} must contain either 'cell_type' (str) "
        f"or 'vtk_type' (int). Got keys: {list(block.keys())}."
    )
