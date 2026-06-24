import typing
import warnings

import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.grids.base import ConnectionType, Grid
from bores.grids.factories.base import FaultRecord, VertexCoordinates
from bores.typing import (
    FloatArray,
    IntArray,
    NumberOrArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["make_cartesian_grid"]


def make_cartesian_grid(
    *,
    nx: typing.Optional[int] = None,
    ny: typing.Optional[int] = None,
    nz: typing.Optional[int] = None,
    dx: NumberOrArray[OneDimension] = 1.0,
    dy: NumberOrArray[OneDimension] = 1.0,
    dz: NumberOrArray[OneDimension] = 1.0,
    origin: typing.Tuple[float, float, float] = (0.0, 0.0, 0.0),
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    fault_records: typing.Optional[typing.Sequence[FaultRecord]] = None,
    fault_transmissibility_multipliers: typing.Optional[
        typing.Mapping[str, float]
    ] = None,
    nnc_cell_indices: typing.Optional[IntArray[TwoDimensions]] = None,
    nnc_transmissibilities: typing.Optional[FloatArray[OneDimension]] = None,
    positive_x_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = None,
    negative_x_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = None,
    positive_y_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = None,
    negative_y_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = None,
    positive_z_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = None,
    negative_z_transmissibility_multipliers: typing.Optional[
        FloatArray[OneDimension]
    ] = None,
) -> Grid:
    """
    Factory for axis-aligned structured Cartesian hexahedral grids.

    Produces a regular Cartesian grid with uniform or variable cell spacing.
    Supports scalar (uniform) or array (variable) spacing in each direction,
    an optional origin offset.

    Example usage:

    ```python
    grid = make_cartesian_grid(nx=10, ny=10, nz=5, dx=100.0, dy=100.0, dz=5.0)

    # Variable spacing:
    grid = make_cartesian_grid(
        dx=np.array([50., 100., 50.]),
        dy=100.0,
        dz=np.array([3., 5., 3., 5., 3.]),
        origin=(0.0, 0.0, 2000.0),
    )
    ```

    Cell counts `nx`, `ny`, `nz` may be inferred from array-valued
    spacing arguments (`len(dx)` etc.) when not explicitly provided.

    :param nx: Number of cells in the x direction.
    :param ny: Number of cells in the y direction.
    :param nz: Number of cells in the z direction (positive downward).
    :param dx: Cell width(s) in x. Scalar = uniform; 1-D array = variable.
    :param dy: Cell width(s) in y.
    :param dz: Cell thickness(es) in z.
    :param origin: `(x0, y0, z0)` coordinate of the grid origin (minimum
        x, y, z vertex).
    :param unit_system: Declared unit system for all coordinate arrays.
    :param metadata: Optional metadata dictionary attached to the grid.
    :param fault_records: Sequence of
        `bores.grids.factories.corner_point.FaultRecord` objects
        parsed from the GRDECL `FAULTS` keyword. Each record's IJK range
        and face direction are resolved to unstructured face indices using
        closed-form Cartesian index arithmetic. Resolved faces are tagged
        :attr:`ConnectionType.FAULT`.
    :param fault_transmissibility_multipliers: Mapping `{name: multiplier}`
        from the GRDECL `MULTFLT` keyword. Stored verbatim on the
        returned `Grid`.
    :param nnc_cell_indices: Shape `(n_nnc, 2)` int32 array of 0-based
        cell index pairs for explicit NNCs from the GRDECL `NNC` keyword.
    :param nnc_transmissibilities: Shape `(n_nnc,)` float64 array of
        transmissibilities corresponding to `nnc_cell_indices`.
    :param positive_x_transmissibility_multipliers: Shape `(n_cells,)`
        per-cell MULTX values. `None` if not supplied.
    :param negative_x_transmissibility_multipliers: Shape `(n_cells,)`
        MULTX- values.
    :param positive_y_transmissibility_multipliers: Shape `(n_cells,)`
        MULTY values.
    :param negative_y_transmissibility_multipliers: Shape `(n_cells,)`
        MULTY- values.
    :param positive_z_transmissibility_multipliers: Shape `(n_cells,)`
        MULTZ values.
    :param negative_z_transmissibility_multipliers: Shape `(n_cells,)`
        MULTZ- values.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises ValidationError: If cell counts cannot be determined, spacing
        values are non-positive, or NNC array lengths are inconsistent.
    """
    dx_arr, dy_arr, dz_arr = _resolve_spacing(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    nx = len(dx_arr)
    ny = len(dy_arr)
    nz = len(dz_arr)

    if nnc_cell_indices is not None and nnc_transmissibilities is not None:
        if len(nnc_cell_indices) != len(nnc_transmissibilities):
            raise ValidationError(
                f"nnc_cell_indices has {len(nnc_cell_indices)} rows but "
                f"nnc_transmissibilities has {len(nnc_transmissibilities)} entries; "
                "they must have the same length."
            )

    vertex_coordinates = _build_vertex_coordinates(
        dx=dx_arr, dy=dy_arr, dz=dz_arr, origin=origin
    )
    face_vertex_indices, face_vertex_offsets, face_cell_indices = _build_face_arrays(
        nx, ny, nz
    )

    # Default connection_types array: BOUNDARY or INTERIOR inferred from topology.
    # We start with the default and patch in FAULT tags after resolution.
    n_faces = face_cell_indices.shape[0]
    n_x_faces = (nx + 1) * ny * nz
    n_y_faces = nx * (ny + 1) * nz
    connection_types = np.where(
        (face_cell_indices[:, 0] >= 0) & (face_cell_indices[:, 1] >= 0),
        int(ConnectionType.INTERIOR),
        int(ConnectionType.BOUNDARY),
    ).astype(np.int8)

    # Resolve FAULTS IJK records to Cartesian face indices.
    resolved_fault_face_indices: typing.Optional[
        typing.Dict[str, IntArray[OneDimension]]
    ] = None
    if fault_records:
        resolved_fault_face_indices = _resolve_fault_face_indices(
            fault_records=fault_records,
            nx=nx,
            ny=ny,
            nz=nz,
            n_x_faces=n_x_faces,
            n_y_faces=n_y_faces,
        )
        for face_indices in resolved_fault_face_indices.values():
            connection_types[face_indices] = int(ConnectionType.FAULT)

    return Grid(
        vertex_coordinates=vertex_coordinates.astype(np.float64, copy=False),
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
        connection_types=connection_types,
        nnc_cell_indices=(
            nnc_cell_indices.astype(np.int32, copy=False)
            if nnc_cell_indices is not None
            else None
        ),
        nnc_transmissibilities=(
            nnc_transmissibilities.astype(np.float64, copy=False)
            if nnc_transmissibilities is not None
            else None
        ),
        fault_face_indices=resolved_fault_face_indices,
        fault_transmissibility_multipliers=(
            dict(fault_transmissibility_multipliers)
            if fault_transmissibility_multipliers is not None
            else None
        ),
        positive_x_transmissibility_multipliers=positive_x_transmissibility_multipliers,
        negative_x_transmissibility_multipliers=negative_x_transmissibility_multipliers,
        positive_y_transmissibility_multipliers=positive_y_transmissibility_multipliers,
        negative_y_transmissibility_multipliers=negative_y_transmissibility_multipliers,
        positive_z_transmissibility_multipliers=positive_z_transmissibility_multipliers,
        negative_z_transmissibility_multipliers=negative_z_transmissibility_multipliers,
    )


_VALID_FACE_DIRS = frozenset({"X", "X-", "Y", "Y-", "Z", "Z-"})


def _resolve_fault_face_indices(
    fault_records: typing.Sequence[FaultRecord],
    nx: int,
    ny: int,
    nz: int,
    n_x_faces: int,
    n_y_faces: int,
) -> typing.Dict[str, IntArray[OneDimension]]:
    """
    Resolve `FaultRecord` IJK ranges to Cartesian face index arrays using
    closed-form index arithmetic.

    The Cartesian face array produced by `_build_face_arrays` is laid
    out in three contiguous groups:

    * **X-normal faces** (count `(nx+1)*ny*nz`):
      face at `(i_plane, j, k)` has global index
      `i_plane * ny * nz + j * nz + k`.
      The *interior* X face between `cell(i, j, k)` and `cell(i+1, j, k)`
      sits at `i_plane = i + 1`.

    * **Y-normal faces** (count `nx*(ny+1)*nz`, offset `n_x_faces`):
      face at `(i, j_plane, k)` has group-local index
      `i * (ny + 1) * nz + j_plane * nz + k`.
      Interior Y face between `cell(i, j, k)` and `cell(i, j+1, k)`
      sits at `j_plane = j + 1`.

    * **Z-normal faces** (count `nx*ny*(nz+1)`, offset `n_x_faces + n_y_faces`):
      face at `(i, j, k_plane)` has group-local index
      `i * ny * (nz + 1) + j * (nz + 1) + k_plane`.
      Interior Z face between `cell(i, j, k)` and `cell(i, j, k+1)`
      sits at `k_plane = k + 1`.

    Eclipse `X-` / `Y-` / `Z-` directions resolve to the same physical
    face as `X` / `Y` / `Z` respectively (same shared face, opposite
    normal direction).

    :param fault_records: Sequence of
        `bores.grids.factories.corner_point.FaultRecord`.
    :param nx: Grid dimension in x.
    :param ny: Grid dimension in y.
    :param nz: Grid dimension in z.
    :param n_x_faces: Total number of X-normal faces `(nx+1)*ny*nz`.
    :param n_y_faces: Total number of Y-normal faces `nx*(ny+1)*nz`.
    :returns: `{fault_name: int32 face index array}` mapping.
    """
    result: typing.Dict[str, typing.List[int]] = {}

    for record in fault_records:
        face_dir = record.face_direction.upper()
        if face_dir not in _VALID_FACE_DIRS:
            warnings.warn(
                f"Fault {record.name!r}: unrecognised face direction "
                f"{record.face_direction!r}. "
                f"Valid directions: {sorted(_VALID_FACE_DIRS)}. Skipping.",
                stacklevel=4,
            )
            continue

        face_indices: typing.List[int] = []
        n_missed = 0

        # IJK ranges are 1-based inclusive in Eclipse convention.
        for k in range(record.k1 - 1, record.k2):
            for j in range(record.j1 - 1, record.j2):
                for i in range(record.i1 - 1, record.i2):
                    face_idx: typing.Optional[int] = None

                    if face_dir in ("X", "X-"):
                        # Face between cell(i, j, k) and cell(i+1, j, k).
                        # Plane index = i + 1; must be in [1, nx-1] for interior.
                        i_plane = i + 1
                        if 0 <= i < nx - 1 and 0 <= j < ny and 0 <= k < nz:
                            face_idx = i_plane * ny * nz + j * nz + k
                        else:
                            n_missed += 1
                            continue

                    elif face_dir in ("Y", "Y-"):
                        # Face between cell(i, j, k) and cell(i, j+1, k).
                        j_plane = j + 1
                        if 0 <= i < nx and 0 <= j < ny - 1 and 0 <= k < nz:
                            face_idx = n_x_faces + i * (ny + 1) * nz + j_plane * nz + k
                        else:
                            n_missed += 1
                            continue

                    else:  # Z, Z-
                        # Face between cell(i, j, k) and cell(i, j, k+1).
                        k_plane = k + 1
                        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz - 1:
                            face_idx = (
                                n_x_faces
                                + n_y_faces
                                + i * ny * (nz + 1)
                                + j * (nz + 1)
                                + k_plane
                            )
                        else:
                            n_missed += 1
                            continue

                    if face_idx is not None:
                        face_indices.append(face_idx)

        if n_missed > 0:
            warnings.warn(
                f"Fault {record.name!r}: {n_missed} cell pair(s) in the IJK range "
                f"I=[{record.i1},{record.i2}] J=[{record.j1},{record.j2}] "
                f"K=[{record.k1},{record.k2}] "
                f"could not be resolved (boundary or out-of-range). Skipped.",
                stacklevel=4,
            )

        if face_indices:
            existing = result.get(record.name)
            if existing is not None:
                existing.extend(face_indices)
            else:
                result[record.name] = face_indices
        elif n_missed == 0:
            warnings.warn(
                f"Fault {record.name!r} produced no resolvable face indices.",
                stacklevel=4,
            )

    return {
        name: np.unique(np.asarray(idxs, dtype=np.int32))
        for name, idxs in result.items()
    }


def _resolve_spacing(
    nx: typing.Optional[int],
    ny: typing.Optional[int],
    nz: typing.Optional[int],
    dx: NumberOrArray[OneDimension],
    dy: NumberOrArray[OneDimension],
    dz: NumberOrArray[OneDimension],
) -> typing.Tuple[
    FloatArray[OneDimension],
    FloatArray[OneDimension],
    FloatArray[OneDimension],
]:
    """
    Resolve scalar or array spacing arguments into 1-D float64 arrays.

    :param nx: Explicit x-cell count (or `None` to infer from `dx`).
    :param ny: Explicit y-cell count.
    :param nz: Explicit z-cell count.
    :param dx: Scalar or array x-spacing.
    :param dy: Scalar or array y-spacing.
    :param dz: Scalar or array z-spacing.
    :returns: Tuple of three 1-D float64 spacing arrays `(dx, dy, dz)`.
    :raises ValidationError: If counts cannot be determined or spacings ≤ 0.
    """

    def _to_array(
        value: NumberOrArray[OneDimension],
        count: typing.Optional[int],
        axis: str,
    ) -> FloatArray[OneDimension]:
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
    dx: FloatArray[OneDimension],
    dy: FloatArray[OneDimension],
    dz: FloatArray[OneDimension],
    origin: typing.Tuple[float, float, float],
) -> VertexCoordinates:
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
    # Ravel in Fortran order to match vert_id(i,j,k) = i + j*(nx+1) + k*(nx+1)*(ny+1)
    vertex_coordinates = np.column_stack(
        [
            xx.ravel(order="F"),
            yy.ravel(order="F"),
            zz.ravel(order="F"),
        ]
    )
    return vertex_coordinates


def _build_face_arrays(
    nx: int, ny: int, nz: int
) -> typing.Tuple[
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[TwoDimensions],
]:
    """
    Build face connectivity arrays for a structured Cartesian grid.

    Uses closed-form index arithmetic (no Python loops) to build all
    X-normal, Y-normal, and Z-normal faces simultaneously.

    Face winding conventions (verified to give correct Newell normals):

    * **X-normal faces** (normal = +x):
      vertices `(i,j,k) -> (i,j+1,k) -> (i,j+1,k+1) -> (i,j,k+1)`
    * **Y-normal faces** (normal = +y):
      vertices `(i,j,k) -> (i,j,k+1) -> (i+1,j,k+1) -> (i+1,j,k)`
    * **Z-normal faces** (normal = +z):
      vertices `(i,j,k) -> (i+1,j,k) -> (i+1,j+1,k) -> (i,j+1,k)`

    Owner is always the cell at the **lower** index in the face-normal
    direction (or -1 for boundary faces).

    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :returns: Tuple `(face_vertex_indices, face_vertex_offsets, face_cell_indices)`.
    """
    # Flat vertex index helper (matches _build_vertex_coordinates meshgrid order)
    # vert_id(i,j,k) = i + j*(nx+1) + k*(nx+1)*(ny+1)
    stride_j = nx + 1
    stride_k = (nx + 1) * (ny + 1)

    # Flat cell index helper
    # cell_id(i,j,k) = i + j*nx + k*nx*ny
    cell_stride_j = nx
    cell_stride_k = nx * ny

    face_vertex_indices_parts: typing.List[npt.NDArray[np.int32]] = []
    face_cell_indices_parts: typing.List[npt.NDArray[np.int32]] = []

    # X-normal faces: (nx+1) * ny * nz faces
    # For plane i (i=0..nx): owner=cell(i-1,j,k), neighbour=cell(i,j,k)

    i_planes = np.arange(nx + 1, dtype=np.int32)
    j_cells = np.arange(ny, dtype=np.int32)
    k_cells = np.arange(nz, dtype=np.int32)

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

    nx_faces = len(ii_x)
    x_face_verts = np.column_stack([v0_x, v1_x, v2_x, v3_x]).astype(np.int32)
    face_vertex_indices_parts.append(x_face_verts.ravel())

    owner_x = np.where(
        ii_x > 0,
        (ii_x - 1) + jj_x * cell_stride_j + kk_x * cell_stride_k,
        -1,
    ).astype(np.int32)
    neighbour_x = np.where(
        ii_x < nx,
        ii_x + jj_x * cell_stride_j + kk_x * cell_stride_k,
        -1,
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_x, neighbour_x]).astype(np.int32)
    )

    # Y-normal faces: nx * (ny+1) * nz faces
    # Winding (normal=+y): (i,j,k) -> (i,j,k+1) -> (i+1,j,k+1) -> (i+1,j,k)

    i_cells_y = np.arange(nx, dtype=np.int32)
    j_planes = np.arange(ny + 1, dtype=np.int32)
    k_cells_y = np.arange(nz, dtype=np.int32)

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
        jj_y < ny,
        ii_y + jj_y * cell_stride_j + kk_y * cell_stride_k,
        -1,
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_y, neighbour_y]).astype(np.int32)
    )

    # Z-normal faces: nx * ny * (nz+1) faces
    # Winding (normal=+z): (i,j,k) -> (i+1,j,k) -> (i+1,j+1,k) -> (i,j+1,k)

    i_cells_z = np.arange(nx, dtype=np.int32)
    j_cells_z = np.arange(ny, dtype=np.int32)
    k_planes = np.arange(nz + 1, dtype=np.int32)

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
        kk_z < nz,
        ii_z + jj_z * cell_stride_j + kk_z * cell_stride_k,
        -1,
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_z, neighbour_z]).astype(np.int32)
    )

    # Concatenate all face parts
    all_face_vertices = np.concatenate(face_vertex_indices_parts)
    all_face_cell_indices = np.vstack(face_cell_indices_parts).astype(np.int32)

    n_total_faces = nx_faces + len(ii_y) + len(ii_z)
    verts_per_face = 4
    face_vertex_offsets = np.arange(
        0, (n_total_faces + 1) * verts_per_face, verts_per_face, dtype=np.int32
    )
    return all_face_vertices, face_vertex_offsets, all_face_cell_indices
