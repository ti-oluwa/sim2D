import typing

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import assemble_grid
from bores.typing import FloatArray, NumberOrArray

__all__ = ["make_cartesian_grid"]

VertexCoordinates3D: TypeAlias = FloatArray
"""Shape `(n_points, 3)` - 3-D (x, y, z) vertex coordinates."""


def make_cartesian_grid(
    *,
    nx: typing.Optional[int] = None,
    ny: typing.Optional[int] = None,
    nz: typing.Optional[int] = None,
    dx: NumberOrArray = 1.0,
    dy: NumberOrArray = 1.0,
    dz: NumberOrArray = 1.0,
    origin: typing.Tuple[float, float, float] = (0.0, 0.0, 0.0),
    metadata: typing.Optional[dict] = None,
) -> Grid:
    """
    Factory for axis-aligned structured Cartesian hexahedral grids.

    Produces a regular Cartesian grid with uniform or variable cell spacing.
    Supports scalar (uniform) or array (variable) spacing in each direction,
    and an optional origin offset.

    Example usage:

    ``python
    grid = make_cartesian_grid(nx=10, ny=10, nz=5, dx=100.0, dy=100.0, dz=5.0)

    # Variable spacing:
    grid = make_cartesian_grid(
        dx=np.array([50., 100., 50.]),
        dy=100.0,
        dz=np.array([3., 5., 3., 5., 3.]),
        origin=(0.0, 0.0, 2000.0),
    )
    ``

    Cell counts `nx`, `ny`, `nz` may be inferred from array-valued
    spacing arguments (`len(dx)` etc.) when not explicitly provided.

    :param nx: Number of cells in the x direction.
    :param ny: Number of cells in the y direction.
    :param nz: Number of cells in the z direction (positive downward).
    :param dx: Cell width(s) in x.  Scalar = uniform; 1-D array = variable.
    :param dy: Cell width(s) in y.
    :param dz: Cell thickness(es) in z.
    :param origin: `(x0, y0, z0)` coordinate of the grid origin (top-left-front
        corner, i.e. minimum x, y, z vertex).
    :param metadata: Optional metadata dictionary attached to the grid.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises ValidationError: If cell counts cannot be determined or spacing
        values are non-positive.
    """
    dx, dy, dz = _resolve_spacing(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    n_x = len(dx)
    n_y = len(dy)
    n_z = len(dz)

    vertex_coordinates = _build_vertex_coordinates(dx=dx, dy=dy, dz=dz, origin=origin)
    face_vertex_indices, face_vertex_offsets, face_cell_indices = _build_face_arrays(
        n_x, n_y, n_z
    )
    return assemble_grid(
        vertex_coordinates,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
        metadata=metadata,
    )


def _resolve_spacing(
    nx: typing.Optional[int],
    ny: typing.Optional[int],
    nz: typing.Optional[int],
    dx: NumberOrArray,
    dy: NumberOrArray,
    dz: NumberOrArray,
) -> typing.Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Resolve scalar or array spacing arguments into 1-D float64 arrays.

    :param nx: Explicit x-cell count (or None to infer from `dx`).
    :param ny: Explicit y-cell count.
    :param nz: Explicit z-cell count.
    :param dx: Scalar or array x-spacing.
    :param dy: Scalar or array y-spacing.
    :param dz: Scalar or array z-spacing.
    :returns: Tuple of three 1-D float64 spacing arrays `(dx, dy, dz)`.
    :raises ValidationError: If counts cannot be determined or spacings ≤ 0.
    """

    def _to_array(
        value: NumberOrArray,
        count: typing.Optional[int],
        axis: str,
    ):
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
        if arr.size == 1:
            if count is None:
                raise ValidationError(
                    f"n{axis} must be provided when d{axis} is a scalar."
                )
            arr = np.full(count, arr[0])
        elif count is not None and len(arr) != count:
            raise ValidationError(
                f"Length of d{axis} array ({len(arr)}) does not match n{axis}={count}."
            )
        if np.any(arr <= 0.0):
            raise ValidationError(
                f"All d{axis} values must be strictly positive; "
                f"got min={arr.min():.6g}."
            )
        return arr

    return (
        _to_array(dx, nx, "x"),
        _to_array(dy, ny, "y"),
        _to_array(dz, nz, "z"),
    )


def _build_vertex_coordinates(
    dx: npt.NDArray[np.float64],
    dy: npt.NDArray[np.float64],
    dz: npt.NDArray[np.float64],
    origin: typing.Tuple[float, float, float],
) -> VertexCoordinates3D:
    """
    Build the `(n_vertices, 3)` vertex coordinate array via meshgrid.

    :param dx: 1-D array of x-direction cell widths.
    :param dy: 1-D array of y-direction cell widths.
    :param dz: 1-D array of z-direction cell thicknesses.
    :param origin: `(x0, y0, z0)` grid origin.
    :returns: Shape `((nx+1)*(ny+1)*(nz+1), 3)` float64 array.
    """
    x_nodes = origin[0] + np.concatenate([[0.0], np.cumsum(dx)])
    y_nodes = origin[1] + np.concatenate([[0.0], np.cumsum(dy)])
    z_nodes = origin[2] + np.concatenate([[0.0], np.cumsum(dz)])

    # meshgrid with 'ij' indexing: shape (nx+1, ny+1, nz+1)
    xx, yy, zz = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    # Reshape to (n_vertices, 3); vertex flat-index = i + j*(nx+1) + k*(nx+1)*(ny+1)
    # To match vert_id(i,j,k) = i + j*(NX+1) + k*(NX+1)*(NY+1), we need
    # the loop order k-outer, j-middle, i-inner:
    vertex_coordinates = np.column_stack(
        [
            xx.ravel(order="F"),
            yy.ravel(order="F"),
            zz.ravel(order="F"),
        ]
    )
    return vertex_coordinates


def _build_face_arrays(
    n_x: int, n_y: int, n_z: int
) -> typing.Tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """
    Build face connectivity arrays for a structured Cartesian grid.

    Uses closed-form index arithmetic (no Python loops) to build all
    X-normal, Y-normal, and Z-normal faces simultaneously.

    Face winding conventions (verified to give correct Newell normals):

    - **X-normal faces** (normal = +x):
        vertices `(i,j,k) -> (i,j+1,k) -> (i,j+1,k+1) -> (i,j,k+1)`
    - **Y-normal faces** (normal = +y):
        vertices `(i,j,k) -> (i,j,k+1) -> (i+1,j,k+1) -> (i+1,j,k)`
    - **Z-normal faces** (normal = +z):
        vertices `(i,j,k) -> (i+1,j,k) -> (i+1,j+1,k) -> (i,j+1,k)`

    Owner is always the cell at the **lower** index in the face-normal
    direction (or -1 for boundary faces).

    :param n_x: Number of cells in x.
    :param n_y: Number of cells in y.
    :param n_z: Number of cells in z.
    :returns: Tuple `(face_vertex_indices, face_vertex_offsets, face_cell_indices)`.
    """
    # Flat vertex index helper (matches _build_vertex_coordinates meshgrid order)
    # vert_id(i,j,k) = i + j*(n_x+1) + k*(n_x+1)*(n_y+1)
    stride_j = n_x + 1
    stride_k = (n_x + 1) * (n_y + 1)

    # Flat cell index helper
    # cell_id(i,j,k) = i + j*n_x + k*n_x*n_y
    cell_stride_j = n_x
    cell_stride_k = n_x * n_y

    face_vertex_indices_parts: typing.List[npt.NDArray[np.int32]] = []
    face_cell_indices_parts: typing.List[npt.NDArray[np.int32]] = []

    # X-normal faces: (n_x+1) * n_y * n_z faces
    # For plane i (i=0..n_x): owner=cell(i-1,j,k), neighbour=cell(i,j,k)
    i_planes = np.arange(n_x + 1, dtype=np.int32)  # (n_x+1,)
    j_cells = np.arange(n_y, dtype=np.int32)  # (n_y,)
    k_cells = np.arange(n_z, dtype=np.int32)  # (n_z,)

    # Build all (i, j, k) combinations with j and k as cell indices
    ii_x, jj_x, kk_x = np.meshgrid(i_planes, j_cells, k_cells, indexing="ij")
    ii_x = ii_x.ravel()
    jj_x = jj_x.ravel()
    kk_x = kk_x.ravel()

    # 4 vertices per X-face (CCW from owner = lower-i side):
    # v0=(i,j,k)  v1=(i,j+1,k)  v2=(i,j+1,k+1)  v3=(i,j,k+1)
    v0_x = ii_x * 1 + jj_x * stride_j + kk_x * stride_k
    v1_x = ii_x * 1 + (jj_x + 1) * stride_j + kk_x * stride_k
    v2_x = ii_x * 1 + (jj_x + 1) * stride_j + (kk_x + 1) * stride_k
    v3_x = ii_x * 1 + jj_x * stride_j + (kk_x + 1) * stride_k

    n_x_faces = len(ii_x)
    # Interleave: shape (n_x_faces, 4) -> flat
    x_face_verts = np.column_stack([v0_x, v1_x, v2_x, v3_x]).astype(np.int32)
    face_vertex_indices_parts.append(x_face_verts.ravel())

    owner_x = np.where(
        ii_x > 0,
        (ii_x - 1) + jj_x * cell_stride_j + kk_x * cell_stride_k,
        -1,
    ).astype(np.int32)
    neighbour_x = np.where(
        ii_x < n_x,
        ii_x + jj_x * cell_stride_j + kk_x * cell_stride_k,
        -1,
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_x, neighbour_x]).astype(np.int32)
    )

    # Y-normal faces: n_x * (n_y+1) * n_z faces
    # Winding (normal=+y): (i,j,k) -> (i,j,k+1) -> (i+1,j,k+1) -> (i+1,j,k)
    i_cells_y = np.arange(n_x, dtype=np.int32)
    j_planes = np.arange(n_y + 1, dtype=np.int32)
    k_cells_y = np.arange(n_z, dtype=np.int32)

    ii_y, jj_y, kk_y = np.meshgrid(i_cells_y, j_planes, k_cells_y, indexing="ij")
    ii_y = ii_y.ravel()
    jj_y = jj_y.ravel()
    kk_y = kk_y.ravel()

    v0_y = ii_y + jj_y * stride_j + kk_y * stride_k
    v1_y = ii_y + jj_y * stride_j + (kk_y + 1) * stride_k
    v2_y = (ii_y + 1) + jj_y * stride_j + (kk_y + 1) * stride_k
    v3_y = (ii_y + 1) + jj_y * stride_j + kk_y * stride_k

    y_face_verts = np.column_stack([v0_y, v1_y, v2_y, v3_y]).astype(np.int32)
    face_vertex_indices_parts.append(y_face_verts.ravel())

    owner_y = np.where(
        jj_y > 0,
        ii_y + (jj_y - 1) * cell_stride_j + kk_y * cell_stride_k,
        -1,
    ).astype(np.int32)
    neighbour_y = np.where(
        jj_y < n_y,
        ii_y + jj_y * cell_stride_j + kk_y * cell_stride_k,
        -1,
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_y, neighbour_y]).astype(np.int32)
    )

    # Z-normal faces: n_x * n_y * (n_z+1) faces
    # Winding (normal=+z): (i,j,k) -> (i+1,j,k) -> (i+1,j+1,k) -> (i,j+1,k)
    i_cells_z = np.arange(n_x, dtype=np.int32)
    j_cells_z = np.arange(n_y, dtype=np.int32)
    k_planes = np.arange(n_z + 1, dtype=np.int32)

    ii_z, jj_z, kk_z = np.meshgrid(i_cells_z, j_cells_z, k_planes, indexing="ij")
    ii_z = ii_z.ravel()
    jj_z = jj_z.ravel()
    kk_z = kk_z.ravel()

    v0_z = ii_z + jj_z * stride_j + kk_z * stride_k
    v1_z = (ii_z + 1) + jj_z * stride_j + kk_z * stride_k
    v2_z = (ii_z + 1) + (jj_z + 1) * stride_j + kk_z * stride_k
    v3_z = ii_z + (jj_z + 1) * stride_j + kk_z * stride_k

    z_face_verts = np.column_stack([v0_z, v1_z, v2_z, v3_z]).astype(np.int32)
    face_vertex_indices_parts.append(z_face_verts.ravel())

    owner_z = np.where(
        kk_z > 0,
        ii_z + jj_z * cell_stride_j + (kk_z - 1) * cell_stride_k,
        -1,
    ).astype(np.int32)
    neighbour_z = np.where(
        kk_z < n_z,
        ii_z + jj_z * cell_stride_j + kk_z * cell_stride_k,
        -1,
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_z, neighbour_z]).astype(np.int32)
    )

    # Concatenate all face parts
    all_face_vertices = np.concatenate(face_vertex_indices_parts)
    all_face_cell_indices = np.vstack(face_cell_indices_parts).astype(np.int32)

    n_total_faces = n_x_faces + len(ii_y) + len(ii_z)
    verts_per_face = 4
    face_vertex_offsets = np.arange(
        0, (n_total_faces + 1) * verts_per_face, verts_per_face, dtype=np.int32
    )
    return all_face_vertices, face_vertex_offsets, all_face_cell_indices
