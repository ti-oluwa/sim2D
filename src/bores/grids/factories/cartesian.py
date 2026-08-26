import typing
import warnings

import numpy as np
import numpy.typing as npt

from bores.datastructures import MapAxes
from bores.errors import ValidationError
from bores.grids.base import ConnectionType, Grid
from bores.grids.factories.base import (
    VALID_FAULT_FACE_DIRECTIONS,
    FaultRecord,
    VertexCoordinates,
)
from bores.typing import (
    FloatArray,
    IntArray,
    Integer,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["make_cartesian_grid"]


def make_cartesian_grid(
    *,
    nx: Integer | None = None,
    ny: Integer | None = None,
    nz: Integer | None = None,
    dx: NumberOrArray[OneDimension] = 1.0,
    dy: NumberOrArray[OneDimension] = 1.0,
    dz: NumberOrArray[OneDimension] = 1.0,
    origin: tuple[Number, Number, Number] = (0.0, 0.0, 0.0),
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Mapping[str, typing.Any] | None = None,
    map_axes: MapAxes | None = None,
    apply_map_axes: bool = True,
    fault_records: typing.Sequence[FaultRecord] | None = None,
    fault_transmissibility_multipliers: typing.Mapping[str, Number] | None = None,
    nnc_cell_indices: IntArray[TwoDimensions] | None = None,
    nnc_transmissibilities: NumberArray[OneDimension] | None = None,
    positive_x_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_x_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    positive_y_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_y_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    positive_z_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_z_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
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
    :param origin: `(x0, y0, z0)` coordinate of the grid origin, in local
        (pre-`MAPAXES`) space.
    :param unit_system: Declared unit system.
    :param metadata: Optional metadata dictionary.
    :param map_axes: `MAPAXES` to apply to the generated vertices before
        returning the `Grid`. Falls back to `metadata['map_axes']` when
        `None`. Unlike the corner-point factory, this rotates the *entire*
        `(nx+1)*(ny+1)*(nz+1)` vertex array, not just an origin - a
        non-axis-aligned rotation means the cells are no longer axis-
        aligned boxes in map space, so there's no cheaper equivalent here.
        `Grid`'s own geometry computation (face areas, normals, volumes)
        doesn't assume axis alignment, so this is still safe; only GRDECL
        re-export (`dump_grdecl`) needs to know to undo it - see
        `bores.grids.io.grdecl._local_cartesian_cell_bounds`.
    :param apply_map_axes: When `True` (the default) and a `map_axes` is
        resolved (from this parameter or `metadata`), vertices are rotated/
        translated into map space before the `Grid` is returned. Set
        `False` to keep the grid in local space - the resolved `map_axes`
        is still stored on `grid.metadata` either way.
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
    if nnc_cell_indices is not None and nnc_transmissibilities is not None:
        if len(nnc_cell_indices) != len(nnc_transmissibilities):
            raise ValidationError(
                f"nnc_cell_indices has {len(nnc_cell_indices)} rows but "
                f"nnc_transmissibilities has {len(nnc_transmissibilities)} entries."
            )

    dx, dy, dz = _resolve_spacing(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    nx = len(dx)
    ny = len(dy)
    nz = len(dz)

    vertex_coordinates = _build_vertex_coordinates(dx=dx, dy=dy, dz=dz, origin=origin)

    resolved_map_axes = map_axes if map_axes is not None else (metadata or {}).get("map_axes")
    if resolved_map_axes is not None:
        # `MAPUNITS` (map_axes' own unit_system) can differ from GRIDUNIT
        # (this grid's unit_system) so we normalise once, upfront, so both the
        # applied transform and the stored metadata are self-consistent.
        resolved_map_axes = resolved_map_axes.convert(unit_system)

    if resolved_map_axes is not None and apply_map_axes:
        vertex_coordinates = vertex_coordinates.copy()
        vertex_coordinates[:, :2] = _map_axes_xy_forward(
            xy=vertex_coordinates[:, :2],  # type: ignore[arg-type]
            map_axes=resolved_map_axes,
        )

    if resolved_map_axes is not None:
        # Keep grid.metadata['map_axes'] consistent with whatever was
        # actually resolved above, even if an explicit `map_axes` argument
        # differed from (or `metadata` didn't yet have) one.
        metadata = {**(metadata or {}), "map_axes": resolved_map_axes}

    face_vertex_indices, face_vertex_offsets, face_cell_indices = _build_face_arrays(
        nx=nx, ny=ny, nz=nz
    )

    n_x_faces = (nx + 1) * ny * nz
    n_y_faces = nx * (ny + 1) * nz

    face_connection_types = np.where(
        (face_cell_indices[:, 0] >= 0) & (face_cell_indices[:, 1] >= 0),
        int(ConnectionType.INTERIOR_FACE),
        int(ConnectionType.BOUNDARY_FACE),
    ).astype(np.int8)

    fault_face_indices: dict[str, IntArray[OneDimension]] | None = None
    fault_nnc_pairs: list[tuple[int, int, str]] = []
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

            face_connection_types[interior_fault_faces] = int(ConnectionType.INTERIOR_FAULT_FACE)
            face_connection_types[boundary_fault_faces] = int(ConnectionType.BOUNDARY_FAULT_FACE)

    # Merge all NNC sources: [fault NNCs] + [user NNCs]
    all_nnc_parts: list[
        tuple[
            npt.NDArray[np.int32],
            npt.NDArray[np.int8],
            npt.NDArray[np.float64],
        ]
    ] = []

    fault_nnc_indices: dict[str, list[int]] = {}
    if fault_nnc_pairs:
        fault_pairs = np.asarray([(a, b) for a, b, _ in fault_nnc_pairs], dtype=np.int32).reshape(
            -1, 2
        )
        fault_nnc_connection_types = np.full(
            len(fault_nnc_pairs), int(ConnectionType.FAULT_NNC), dtype=np.int8
        )
        fault_nnc_transmissibilities = np.full(len(fault_nnc_pairs), np.nan, dtype=np.float64)
        all_nnc_parts.append(
            (
                fault_pairs,
                fault_nnc_connection_types,
                fault_nnc_transmissibilities,
            )
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
            (
                user_nnc_pairs,
                user_nnc_connection_types,
                user_nnc_transmissibilities,
            )
        )

    merged_nnc_pairs: npt.NDArray[np.int32] | None = None
    merged_nnc_connection_types: npt.NDArray[np.int8] | None = None
    merged_nnc_transmissibilities: npt.NDArray[np.float64] | None = None
    merged_nnc_fault_indices: dict[str, IntArray[OneDimension]] | None = None

    if all_nnc_parts:
        merged_nnc_pairs = np.vstack([p for p, _, _ in all_nnc_parts]).astype(np.int32)
        merged_nnc_connection_types = np.concatenate([t for _, t, _ in all_nnc_parts]).astype(
            np.int8, copy=False
        )
        merged_transmissibilities = np.concatenate([t for _, _, t in all_nnc_parts])
        merged_nnc_transmissibilities = (
            merged_transmissibilities if np.any(np.isfinite(merged_transmissibilities)) else None
        )
        if fault_nnc_pairs:
            merged_nnc_fault_indices = typing.cast(
                dict[str, IntArray[OneDimension]],
                {
                    name: np.asarray(idxs, dtype=np.int32)
                    for name, idxs in fault_nnc_indices.items()
                },
            )

    return Grid(
        vertex_coordinates=vertex_coordinates.astype(np.float64, copy=False),
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
        face_connection_types=face_connection_types,  # type: ignore[arg-type]
        nnc_cell_indices=merged_nnc_pairs,  # type: ignore[arg-type]
        nnc_connection_types=merged_nnc_connection_types,  # type: ignore[arg-type]
        nnc_transmissibilities=merged_nnc_transmissibilities,  # type: ignore[arg-type]
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


def _map_axes_xy_forward(
    xy: NumberArray[TwoDimensions], map_axes: MapAxes
) -> NumberArray[TwoDimensions]:
    """
    Map local `(x, y)` pairs into map space: `origin + rotation_matrix @ xy`.

    :param xy: Shape `(n, 2)` local-space points.
    :param map_axes: Map axes to apply.
    :returns: Shape `(n, 2)` map-space points.
    """
    return typing.cast(
        NumberArray[TwoDimensions], map_axes.origin + xy @ map_axes.rotation_matrix.T
    )


def _resolve_fault_face_indices(
    fault_records: typing.Sequence[FaultRecord],
    nx: Integer,
    ny: Integer,
    nz: Integer,
    n_x_faces: Integer,
    n_y_faces: Integer,
) -> tuple[
    dict[str, IntArray[OneDimension]],
    list[tuple[int, int, str]],
]:
    """
    Resolve `FaultRecord` IJK ranges to Cartesian face index arrays.

    The Cartesian face layout (from `_build_face_arrays`):

    - X-normal faces (count `(nx+1)*ny*nz`): face at plane `i_plane`,
      cell `(j, k)` -> global index `i_plane * ny * nz + j * nz + k`.
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

    result: dict[str, list[int]] = {}
    fault_nnc_pairs: list[tuple[int, int, str]] = []

    for record in fault_records:
        face_dir = record.face_direction.upper()
        if face_dir not in VALID_FAULT_FACE_DIRECTIONS:
            warnings.warn(
                f"Fault {record.name!r}: unrecognised face direction "
                f"{record.face_direction!r}. "
                f"Valid: {sorted(VALID_FAULT_FACE_DIRECTIONS)}. Skipping.",
                stacklevel=4,
            )
            continue

        face_indices: list[int] = []

        for k in range(record.k1 - 1, record.k2):
            for j in range(record.j1 - 1, record.j2):
                for i in range(record.i1 - 1, record.i2):
                    face_idx: int | None = None
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
                                n_x_faces + n_y_faces + i * ny * (nz + 1) + j * (nz + 1) + k_plane
                            )

                    if face_idx is None:
                        continue

                    if not is_interior:
                        # Boundary face - no interior neighbour; record as fault derived NNC
                        # if both cells exist within the grid extents.
                        if face_dir in ("X", "X-") and 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                            nb_i = i + 1
                            if 0 <= nb_i < nx:
                                cell_a = i + j * cell_stride_j + k * cell_stride_k
                                cell_b = nb_i + j * cell_stride_j + k * cell_stride_k
                                fault_nnc_pairs.append((cell_a, cell_b, record.name))
                        elif (
                            face_dir in ("Y", "Y-") and 0 <= i < nx and 0 <= j < ny and 0 <= k < nz
                        ):
                            nb_j = j + 1
                            if 0 <= nb_j < ny:
                                cell_a = i + j * cell_stride_j + k * cell_stride_k
                                cell_b = i + nb_j * cell_stride_j + k * cell_stride_k
                                fault_nnc_pairs.append((cell_a, cell_b, record.name))
                        elif (
                            face_dir in ("Z", "Z-") and 0 <= i < nx and 0 <= j < ny and 0 <= k < nz
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

    return (  # type: ignore[return-value]
        {name: np.unique(np.asarray(idxs, dtype=np.int32)) for name, idxs in result.items()},
        fault_nnc_pairs,
    )


def _resolve_spacing(
    nx: Integer | None,
    ny: Integer | None,
    nz: Integer | None,
    dx: NumberOrArray[OneDimension],
    dy: NumberOrArray[OneDimension],
    dz: NumberOrArray[OneDimension],
) -> tuple[
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
        count: Integer | None,
        axis: str,
    ) -> FloatArray[OneDimension]:
        arr = np.atleast_1d(value).astype(np.float64, copy=False).ravel()
        if arr.size == 1:
            if count is None:
                raise ValidationError(f"n{axis} must be provided when d{axis} is a scalar.")
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
    origin: tuple[Number, Number, Number],
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
    return typing.cast(
        VertexCoordinates,
        np.column_stack(
            [
                xx.ravel(order="F"),
                yy.ravel(order="F"),
                zz.ravel(order="F"),
            ]
        ),
    )


def _build_face_arrays(
    nx: Integer, ny: Integer, nz: Integer
) -> tuple[
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[TwoDimensions],
]:
    """
    Build face connectivity arrays for a structured Cartesian grid.

    Face winding conventions (CCW from owner = lower-index side):

    - X-normal: `(i,j,k) -> (i,j+1,k) -> (i,j+1,k+1) -> (i,j,k+1)`
    - Y-normal: `(i,j,k) -> (i,j,k+1) -> (i+1,j,k+1) -> (i+1,j,k)`
    - Z-normal: `(i,j,k) -> (i+1,j,k) -> (i+1,j+1,k) -> (i,j+1,k)`

    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :returns: Tuple `(face_vertex_indices, face_vertex_offsets, face_cell_indices)`.
    """
    stride_j = nx + 1
    stride_k = (nx + 1) * (ny + 1)
    cell_stride_j = nx
    cell_stride_k = nx * ny

    face_vertex_indices_parts: list[npt.NDArray[np.int32]] = []
    face_cell_indices_parts: list[npt.NDArray[np.int32]] = []

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
        np.column_stack([v0_x, v1_x, v2_x, v3_x]).astype(np.int32, copy=False).ravel()
    )
    owner_x = np.where(
        ii_x > 0, (ii_x - 1) + jj_x * cell_stride_j + kk_x * cell_stride_k, -1
    ).astype(np.int32, copy=False)
    neighbour_x = np.where(
        ii_x < nx, ii_x + jj_x * cell_stride_j + kk_x * cell_stride_k, -1
    ).astype(np.int32, copy=False)
    face_cell_indices_parts.append(
        np.column_stack([owner_x, neighbour_x]).astype(np.int32, copy=False)
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
        np.column_stack([v0_y, v1_y, v2_y, v3_y]).astype(np.int32, copy=False).ravel()
    )
    owner_y = np.where(
        jj_y > 0, ii_y + (jj_y - 1) * cell_stride_j + kk_y * cell_stride_k, -1
    ).astype(np.int32, copy=False)
    neighbour_y = np.where(
        jj_y < ny, ii_y + jj_y * cell_stride_j + kk_y * cell_stride_k, -1
    ).astype(np.int32, copy=False)
    face_cell_indices_parts.append(
        np.column_stack([owner_y, neighbour_y]).astype(np.int32, copy=False)
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
        np.column_stack([v0_z, v1_z, v2_z, v3_z]).astype(np.int32, copy=False).ravel()
    )
    owner_z = np.where(
        kk_z > 0, ii_z + jj_z * cell_stride_j + (kk_z - 1) * cell_stride_k, -1
    ).astype(np.int32, copy=False)
    neighbour_z = np.where(
        kk_z < nz, ii_z + jj_z * cell_stride_j + kk_z * cell_stride_k, -1
    ).astype(np.int32, copy=False)
    face_cell_indices_parts.append(
        np.column_stack([owner_z, neighbour_z]).astype(np.int32, copy=False)
    )

    all_face_vertices = np.concatenate(face_vertex_indices_parts)
    all_face_cell_indices = np.vstack(face_cell_indices_parts).astype(np.int32, copy=False)

    n_total_faces = nx_faces + len(ii_y) + len(ii_z)
    verts_per_face = 4
    face_vertex_offsets = np.arange(
        0, (n_total_faces + 1) * verts_per_face, verts_per_face, dtype=np.int32
    )
    return all_face_vertices, face_vertex_offsets, all_face_cell_indices  # type: ignore[return-value]
