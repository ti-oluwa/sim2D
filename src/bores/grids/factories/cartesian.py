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

    Supports scalar (uniform) or array (variable) spacing in each direction.
    Cell counts `nx`, `ny`, `nz` may be inferred from array-valued spacing.

    Fault records whose IJK range resolves to cell pairs sharing no geometric
    face (e.g. boundary cells) are recorded as NNCs of type `FAULT` rather
    than silently dropped.

    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z (positive downward).
    :param dx: Cell width(s) in x. Scalar = uniform; 1-D array = variable.
    :param dy: Cell width(s) in y.
    :param dz: Cell thickness(es) in z.
    :param origin: `(x0, y0, z0)` coordinate of the grid origin.
    :param unit_system: Declared unit system.
    :param metadata: Optional metadata dictionary.
    :param fault_records: `FaultRecord` objects from `FAULTS` keyword.
    :param fault_transmissibility_multipliers: `{name: multiplier}` from `MULTFLT`.
    :param positive_x_transmissibility_multipliers: Per-cell MULTX.
    :param negative_x_transmissibility_multipliers: Per-cell MULTX-.
    :param positive_y_transmissibility_multipliers: Per-cell MULTY.
    :param negative_y_transmissibility_multipliers: Per-cell MULTY-.
    :param positive_z_transmissibility_multipliers: Per-cell MULTZ.
    :param negative_z_transmissibility_multipliers: Per-cell MULTZ-.
    :param nnc_cell_indices: Shape `(n_nnc, 2)` user-declared NNC pairs.
    :param nnc_transmissibilities: Shape `(n_nnc,)` user-declared NNC T.
    :returns: Fully initialised `Grid`.
    :raises ValidationError: If spacing or NNC arrays are inconsistent.
    """
    dx_arr, dy_arr, dz_arr = _resolve_spacing(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    nx = len(dx_arr)
    ny = len(dy_arr)
    nz = len(dz_arr)

    if nnc_cell_indices is not None and nnc_transmissibilities is not None:
        if len(nnc_cell_indices) != len(nnc_transmissibilities):
            raise ValidationError(
                f"nnc_cell_indices has {len(nnc_cell_indices)} rows but "
                f"nnc_transmissibilities has {len(nnc_transmissibilities)} entries."
            )

    vertex_coordinates = _build_vertex_coordinates(
        dx=dx_arr, dy=dy_arr, dz=dz_arr, origin=origin
    )
    face_vertex_indices, face_vertex_offsets, face_cell_indices = _build_face_arrays(
        nx, ny, nz
    )

    n_x_faces = (nx + 1) * ny * nz
    n_y_faces = nx * (ny + 1) * nz

    face_connection_types = np.where(
        (face_cell_indices[:, 0] >= 0) & (face_cell_indices[:, 1] >= 0),
        int(ConnectionType.INTERIOR_FACE),
        int(ConnectionType.BOUNDARY_FACE),
    ).astype(np.int8)

    fault_face_indices: typing.Optional[typing.Dict[str, IntArray[OneDimension]]] = None
    fault_nnc_pairs: typing.List[typing.Tuple[int, int, str]] = []
    if fault_records:
        fault_face_indices, fault_nnc_pairs = _resolve_fault_face_indices(
            fault_records=fault_records,
            nx=nx,
            ny=ny,
            nz=nz,
            n_x_faces=n_x_faces,
            n_y_faces=n_y_faces,
        )
        for face_indices in fault_face_indices.values():
            boundary_fault_mask = (face_cell_indices[face_indices, 0] < 0) | (
                face_cell_indices[face_indices, 1] < 0
            )
            boundary_fault_faces = face_indices[boundary_fault_mask]
            interior_fault_faces = face_indices[~boundary_fault_mask]

            face_connection_types[interior_fault_faces] = int(
                ConnectionType.INTERIOR_FAULT_FACE
            )
            face_connection_types[boundary_fault_faces] = int(
                ConnectionType.BOUNDARY_FAULT_FACE
            )

    # Merge all NNC sources: [fault NNCs] + [user NNCs]
    all_nnc_parts: typing.List[
        typing.Tuple[
            npt.NDArray[np.int32],
            npt.NDArray[np.int8],
            npt.NDArray[np.float64],
        ]
    ] = []

    fault_nnc_indices: typing.Dict[str, typing.List[int]] = {}
    if fault_nnc_pairs:
        fault_pairs = np.asarray(
            [(a, b) for a, b, _ in fault_nnc_pairs], dtype=np.int32
        ).reshape(-1, 2)
        fault_nnc_connection_types = np.full(
            len(fault_nnc_pairs), int(ConnectionType.FAULT_NNC), dtype=np.int8
        )
        fault_nnc_transmissibilities = np.full(
            len(fault_nnc_pairs), np.nan, dtype=np.float64
        )
        all_nnc_parts.append(
            (fault_pairs, fault_nnc_connection_types, fault_nnc_transmissibilities)
        )
        fault_nnc_offset = sum(len(p) for p, _, _ in all_nnc_parts[:-1])
        for local_idx, (_, _, name) in enumerate(fault_nnc_pairs):
            fault_nnc_indices.setdefault(name, []).append(fault_nnc_offset + local_idx)

    if nnc_cell_indices is not None and len(nnc_cell_indices) > 0:
        user_nnc_pairs = np.asarray(nnc_cell_indices, dtype=np.int32)
        user_nnc_connection_types = np.full(
            len(user_nnc_pairs), int(ConnectionType.USER_NNC), dtype=np.int8
        )
        user_nnc_transmissibilities = (
            np.asarray(nnc_transmissibilities, dtype=np.float64)
            if nnc_transmissibilities is not None
            else np.full(len(user_nnc_pairs), np.nan, dtype=np.float64)
        )
        all_nnc_parts.append(
            (user_nnc_pairs, user_nnc_connection_types, user_nnc_transmissibilities)
        )

    merged_nnc_pairs: typing.Optional[npt.NDArray[np.int32]] = None
    merged_nnc_connection_types: typing.Optional[npt.NDArray[np.int8]] = None
    merged_nnc_transmissibilities: typing.Optional[npt.NDArray[np.float64]] = None
    merged_nnc_fault_indices: typing.Optional[
        typing.Dict[str, IntArray[OneDimension]]
    ] = None

    if all_nnc_parts:
        merged_nnc_pairs = np.vstack([p for p, _, _ in all_nnc_parts]).astype(np.int32)
        merged_nnc_connection_types = np.concatenate(
            [t for _, t, _ in all_nnc_parts]
        ).astype(np.int8)
        merged_transmissibilities = np.concatenate([t for _, _, t in all_nnc_parts])
        merged_nnc_transmissibilities = (
            merged_transmissibilities
            if np.any(np.isfinite(merged_transmissibilities))
            else None
        )
        if fault_nnc_pairs:
            merged_nnc_fault_indices = {
                name: np.asarray(idxs, dtype=np.int32)
                for name, idxs in fault_nnc_indices.items()
            }

    return Grid(
        vertex_coordinates=vertex_coordinates.astype(np.float64, copy=False),
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
        face_connection_types=face_connection_types,
        nnc_cell_indices=merged_nnc_pairs,
        nnc_connection_types=merged_nnc_connection_types,
        nnc_transmissibilities=merged_nnc_transmissibilities,
        nnc_fault_indices=merged_nnc_fault_indices,
        fault_face_indices=fault_face_indices,
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
) -> typing.Tuple[
    typing.Dict[str, IntArray[OneDimension]],
    typing.List[typing.Tuple[int, int, str]],
]:
    """
    Resolve `FaultRecord` IJK ranges to Cartesian face index arrays.

    The Cartesian face layout (from `_build_face_arrays`):

    - X-normal faces (count `(nx+1)*ny*nz`): face at plane `i_plane`,
      cell `(j, k)` → global index `i_plane * ny * nz + j * nz + k`.
    - Y-normal faces (count `nx*(ny+1)*nz`, offset `n_x_faces`):
      `n_x_faces + i * (ny+1) * nz + j_plane * nz + k`.
    - Z-normal faces (count `nx*ny*(nz+1)`, offset `n_x_faces + n_y_faces`):
      `n_x_faces + n_y_faces + i * ny * (nz+1) + j * (nz+1) + k_plane`.

    Cell pairs that map to boundary or out-of-range faces are returned as
    fault NNC pairs instead of being silently dropped.

    :param fault_records: Sequence of `FaultRecord`.
    :param nx: Grid dimension x.
    :param ny: Grid dimension y.
    :param nz: Grid dimension z.
    :param n_x_faces: Total X-normal face count.
    :param n_y_faces: Total Y-normal face count.
    :returns: Tuple `(fault_face_dict, fault_nnc_pairs)`.
    """
    cell_stride_j = nx
    cell_stride_k = nx * ny

    result: typing.Dict[str, typing.List[int]] = {}
    fault_nnc_pairs: typing.List[typing.Tuple[int, int, str]] = []

    for record in fault_records:
        face_dir = record.face_direction.upper()
        if face_dir not in _VALID_FACE_DIRS:
            warnings.warn(
                f"Fault {record.name!r}: unrecognised face direction "
                f"{record.face_direction!r}. "
                f"Valid: {sorted(_VALID_FACE_DIRS)}. Skipping.",
                stacklevel=4,
            )
            continue

        face_indices: typing.List[int] = []

        for k in range(record.k1 - 1, record.k2):
            for j in range(record.j1 - 1, record.j2):
                for i in range(record.i1 - 1, record.i2):
                    face_idx: typing.Optional[int] = None
                    if face_dir in ("X", "X-"):
                        i_plane = i + 1
                        is_interior = 0 <= i < nx - 1 and 0 <= j < ny and 0 <= k < nz
                        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                            face_idx = i_plane * ny * nz + j * nz + k

                    elif face_dir in ("Y", "Y-"):
                        j_plane = j + 1
                        is_interior = 0 <= i < nx and 0 <= j < ny - 1 and 0 <= k < nz
                        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                            face_idx = n_x_faces + i * (ny + 1) * nz + j_plane * nz + k

                    else:  # Z, Z-
                        k_plane = k + 1
                        is_interior = 0 <= i < nx and 0 <= j < ny and 0 <= k < nz - 1
                        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                            face_idx = (
                                n_x_faces
                                + n_y_faces
                                + i * ny * (nz + 1)
                                + j * (nz + 1)
                                + k_plane
                            )

                    if face_idx is None:
                        continue

                    if not is_interior:
                        # Boundary face - no interior neighbour; record as fault derived NNC
                        # if both cells exist within the grid extents.
                        if (
                            face_dir in ("X", "X-")
                            and 0 <= i < nx
                            and 0 <= j < ny
                            and 0 <= k < nz
                        ):
                            nb_i = i + 1
                            if 0 <= nb_i < nx:
                                cell_a = i + j * cell_stride_j + k * cell_stride_k
                                cell_b = nb_i + j * cell_stride_j + k * cell_stride_k
                                fault_nnc_pairs.append((cell_a, cell_b, record.name))
                        elif (
                            face_dir in ("Y", "Y-")
                            and 0 <= i < nx
                            and 0 <= j < ny
                            and 0 <= k < nz
                        ):
                            nb_j = j + 1
                            if 0 <= nb_j < ny:
                                cell_a = i + j * cell_stride_j + k * cell_stride_k
                                cell_b = i + nb_j * cell_stride_j + k * cell_stride_k
                                fault_nnc_pairs.append((cell_a, cell_b, record.name))
                        elif (
                            face_dir in ("Z", "Z-")
                            and 0 <= i < nx
                            and 0 <= j < ny
                            and 0 <= k < nz
                        ):
                            nb_k = k + 1
                            if 0 <= nb_k < nz:
                                cell_a = i + j * cell_stride_j + k * cell_stride_k
                                cell_b = i + j * cell_stride_j + nb_k * cell_stride_k
                                fault_nnc_pairs.append((cell_a, cell_b, record.name))
                        continue

                    face_indices.append(face_idx)

        if face_indices:
            existing = result.get(record.name)
            if existing is not None:
                existing.extend(face_indices)
            else:
                result[record.name] = face_indices

    return (
        {
            name: np.unique(np.asarray(idxs, dtype=np.int32))
            for name, idxs in result.items()
        },
        fault_nnc_pairs,
    )


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
    :returns: Tuple of three 1-D float64 spacing arrays.
    :raises ValidationError: If counts cannot be determined or spacings <= 0.
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
                f"All d{axis} values must be strictly positive; got min={arr.min():.6g}."
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

    xx, yy, zz = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    return np.column_stack(
        [
            xx.ravel(order="F"),
            yy.ravel(order="F"),
            zz.ravel(order="F"),
        ]
    )


def _build_face_arrays(
    nx: int, ny: int, nz: int
) -> typing.Tuple[
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[TwoDimensions],
]:
    """
    Build face connectivity arrays for a structured Cartesian grid.

    Face winding conventions (CCW from owner = lower-index side):

    - X-normal: `(i,j,k) → (i,j+1,k) → (i,j+1,k+1) → (i,j,k+1)`
    - Y-normal: `(i,j,k) → (i,j,k+1) → (i+1,j,k+1) → (i+1,j,k)`
    - Z-normal: `(i,j,k) → (i+1,j,k) → (i+1,j+1,k) → (i,j+1,k)`

    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :returns: Tuple `(face_vertex_indices, face_vertex_offsets, face_cell_indices)`.
    """
    stride_j = nx + 1
    stride_k = (nx + 1) * (ny + 1)
    cell_stride_j = nx
    cell_stride_k = nx * ny

    face_vertex_indices_parts: typing.List[npt.NDArray[np.int32]] = []
    face_cell_indices_parts: typing.List[npt.NDArray[np.int32]] = []

    i_planes = np.arange(nx + 1, dtype=np.int32)
    j_cells = np.arange(ny, dtype=np.int32)
    k_cells = np.arange(nz, dtype=np.int32)
    ii_x, jj_x, kk_x = np.meshgrid(i_planes, j_cells, k_cells, indexing="ij")
    ii_x = ii_x.ravel()
    jj_x = jj_x.ravel()
    kk_x = kk_x.ravel()

    v0_x = ii_x + jj_x * stride_j + kk_x * stride_k
    v1_x = ii_x + (jj_x + 1) * stride_j + kk_x * stride_k
    v2_x = ii_x + (jj_x + 1) * stride_j + (kk_x + 1) * stride_k
    v3_x = ii_x + jj_x * stride_j + (kk_x + 1) * stride_k
    nx_faces = len(ii_x)
    face_vertex_indices_parts.append(
        np.column_stack([v0_x, v1_x, v2_x, v3_x]).astype(np.int32).ravel()
    )
    owner_x = np.where(
        ii_x > 0, (ii_x - 1) + jj_x * cell_stride_j + kk_x * cell_stride_k, -1
    ).astype(np.int32)
    neighbour_x = np.where(
        ii_x < nx, ii_x + jj_x * cell_stride_j + kk_x * cell_stride_k, -1
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_x, neighbour_x]).astype(np.int32)
    )

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
    face_vertex_indices_parts.append(
        np.column_stack([v0_y, v1_y, v2_y, v3_y]).astype(np.int32).ravel()
    )
    owner_y = np.where(
        jj_y > 0, ii_y + (jj_y - 1) * cell_stride_j + kk_y * cell_stride_k, -1
    ).astype(np.int32)
    neighbour_y = np.where(
        jj_y < ny, ii_y + jj_y * cell_stride_j + kk_y * cell_stride_k, -1
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_y, neighbour_y]).astype(np.int32)
    )

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
    face_vertex_indices_parts.append(
        np.column_stack([v0_z, v1_z, v2_z, v3_z]).astype(np.int32).ravel()
    )
    owner_z = np.where(
        kk_z > 0, ii_z + jj_z * cell_stride_j + (kk_z - 1) * cell_stride_k, -1
    ).astype(np.int32)
    neighbour_z = np.where(
        kk_z < nz, ii_z + jj_z * cell_stride_j + kk_z * cell_stride_k, -1
    ).astype(np.int32)
    face_cell_indices_parts.append(
        np.column_stack([owner_z, neighbour_z]).astype(np.int32)
    )

    all_face_vertices = np.concatenate(face_vertex_indices_parts)
    all_face_cell_indices = np.vstack(face_cell_indices_parts).astype(np.int32)

    n_total_faces = nx_faces + len(ii_y) + len(ii_z)
    verts_per_face = 4
    face_vertex_offsets = np.arange(
        0, (n_total_faces + 1) * verts_per_face, verts_per_face, dtype=np.int32
    )
    return all_face_vertices, face_vertex_offsets, all_face_cell_indices
