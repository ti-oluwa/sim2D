import typing

import numpy as np
from typing_extensions import TypeAlias

from bores.errors import InvalidGridError, ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import (
    ELEMENT_FACE_TABLES,
    assemble_grid,
    build_csr_face_arrays,
)
from bores.typing import (
    FloatArray,
    IntArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
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
        actnum=actnum_array, # shape (NZ, NY, NX)  — optional
    )
    ```

    :param coord: Shape `(NY+1, NX+1, 6)` pillar array. Each entry
        contains `[x_top, y_top, z_top, x_bot, y_bot, z_bot]` defining
        the top and bottom anchor points of a pillar.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array. For each
        cell `(i, j, k)` the 8 corner z-values are at indices
        `[2k:2k+2, 2j:2j+2, 2i:2i+2]` in `[top/bot, left/right, near/far]`
        order.
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

    vertex_coordinates, per_cell_face_vertex_lists = _compute_corner_point_geometry(
        coord_arr, zcorn_arr, actnum_arr, nx, ny, nz
    )
    _, face_vertex_indices, face_vertex_offsets, face_cell_indices = (
        build_csr_face_arrays(vertex_coordinates, per_cell_face_vertex_lists)
    )
    return assemble_grid(
        vertex_coordinates,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
        metadata=metadata,
    )


def _interpolate_pillar_point(
    pillar_top: FloatArray[OneDimension],
    pillar_bot: FloatArray[OneDimension],
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
    dz_pillar = pillar_bot[2] - pillar_top[2]
    if abs(dz_pillar) < 1e-14:
        # Vertical or degenerate pillar: return top xy at requested z
        return np.array([pillar_top[0], pillar_top[1], z])
    t = (z - pillar_top[2]) / dz_pillar
    x = pillar_top[0] + t * (pillar_bot[0] - pillar_top[0])
    y = pillar_top[1] + t * (pillar_bot[1] - pillar_top[1])
    return np.array([x, y, z])


def _compute_corner_point_geometry(
    coord: FloatArray[ThreeDimensions],
    zcorn: FloatArray[ThreeDimensions],
    actnum: IntArray[ThreeDimensions],
    nx: int,
    ny: int,
    nz: int,
) -> typing.Tuple[VertexCoordinates, typing.List[typing.List[FaceVertexList]]]:
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
    all_vertices: typing.List[FloatArray[OneDimension]] = []
    vertex_dedup: typing.Dict[typing.Tuple[float, float, float], int] = {}
    per_cell_face_vertex_lists: typing.List[typing.List[FaceVertexList]] = []
    # Map from (i,j,k) cell to its sequential active-cell index
    active_cell_global_index: typing.Dict[typing.Tuple[int, int, int], int] = {}

    def _add_vertex(xyz: FloatArray[OneDimension]) -> int:
        """
        Register or retrieve a vertex by rounded coordinates.

        :param xyz: Shape `(3,)` coordinate array.
        :returns: Global vertex index.
        """
        key = (
            round(float(xyz[0]), 8),
            round(float(xyz[1]), 8),
            round(float(xyz[2]), 8),
        )
        if key not in vertex_dedup:
            vertex_dedup[key] = len(all_vertices)
            all_vertices.append(xyz.copy())
        return vertex_dedup[key]

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if actnum[k, j, i] == 0:
                    continue

                cell_global_idx = len(active_cell_global_index)
                active_cell_global_index[(i, j, k)] = cell_global_idx

                # Retrieve the 4 pillars bounding cell (i,j,k)
                # Pillar ordering: (j,i), (j,i+1), (j+1,i), (j+1,i+1)
                pillars = [
                    (coord[j, i, :3], coord[j, i, 3:]),  # NW
                    (coord[j, i + 1, :3], coord[j, i + 1, 3:]),  # NE
                    (coord[j + 1, i, :3], coord[j + 1, i, 3:]),  # SW
                    (coord[j + 1, i + 1, :3], coord[j + 1, i + 1, 3:]),  # SE
                ]

                # ZCORN indices for this cell:
                # top layer k → zcorn[2k, :, :]; bottom → zcorn[2k+1, :, :]
                # Within the cell layer: near row j → zcorn[:, 2j, :]; far → zcorn[:, 2j+1, :]
                # Left col i → zcorn[:, :, 2i]; right → zcorn[:, :, 2i+1]
                z_corners = np.array(
                    [
                        # Top face (k layer top): [NW-top, NE-top, SW-top, SE-top]
                        zcorn[2 * k, 2 * j, 2 * i],  # corner 0: NW top
                        zcorn[2 * k, 2 * j, 2 * i + 1],  # corner 1: NE top
                        zcorn[2 * k, 2 * j + 1, 2 * i],  # corner 2: SW top
                        zcorn[2 * k, 2 * j + 1, 2 * i + 1],  # corner 3: SE top
                        # Bottom face (k layer bottom):
                        zcorn[2 * k + 1, 2 * j, 2 * i],  # corner 4: NW bot
                        zcorn[2 * k + 1, 2 * j, 2 * i + 1],  # corner 5: NE bot
                        zcorn[2 * k + 1, 2 * j + 1, 2 * i],  # corner 6: SW bot
                        zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1],  # corner 7: SE bot
                    ]
                )

                # Pillar order matching corner order:
                # corners [0,1,2,3,4,5,6,7] → pillars [NW,NE,SW,SE,NW,NE,SW,SE]
                pillar_order = [0, 1, 2, 3, 0, 1, 2, 3]
                corner_xyz = [
                    _interpolate_pillar_point(
                        pillars[pillar_order[c]][0],
                        pillars[pillar_order[c]][1],
                        float(z_corners[c]),
                    )
                    for c in range(8)
                ]

                # Register global vertex indices for this cell's 8 corners
                corner_global_indices = [_add_vertex(xyz) for xyz in corner_xyz]

                # Build cell faces (hex element: corners 0-3=top, 4-7=bottom)
                # Remap from local corner layout to VTK hex convention:
                # VTK hex: v0=(NW-top), v1=(NE-top), v2=(SE-top), v3=(SW-top)
                #           v4=(NW-bot), v5=(NE-bot), v6=(SE-bot), v7=(SW-bot)
                # Our corner layout: 0=NW-top,1=NE-top,2=SW-top,3=SE-top
                #                    4=NW-bot,5=NE-bot,6=SW-bot,7=SE-bot
                # Map to VTK: v0=c0, v1=c1, v2=c3, v3=c2, v4=c4, v5=c5, v6=c7, v7=c6
                vtk_to_corner = [0, 1, 3, 2, 4, 5, 7, 6]
                vtk_global = [corner_global_indices[vtk_to_corner[v]] for v in range(8)]

                hex_faces: typing.List[FaceVertexList] = [
                    [vtk_global[f] for f in face_local]
                    for face_local in ELEMENT_FACE_TABLES["hexahedron"]
                ]
                per_cell_face_vertex_lists.append(hex_faces)

    if not all_vertices:
        raise InvalidGridError(
            "No active cells found in the corner-point grid (ACTNUM is all zeros)."
        )

    vertex_coordinate_array = np.array(all_vertices, dtype=np.float64)
    return vertex_coordinate_array, per_cell_face_vertex_lists
