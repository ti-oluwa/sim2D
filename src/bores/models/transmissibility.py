"""Face transmissibilities for unstructured polyhedral grids."""

import typing
import warnings

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import NamedTuple

from bores.grids.base import Grid
from bores.models.properties import RockProperties
from bores.typing import FloatArray, IntArray, OneDimension, TwoDimensions

__all__ = ["FaceTransmissibilities", "compute_face_transmissibilities"]


class FaceTransmissibilities(NamedTuple):
    """
    Precomputed face transmissibilities for a `ReservoirModel`.

    **Attributes**:

    interior:
        Shape `(n_interior_faces,)` float64 - TPFA transmissibility for
        every interior face of the grid (mD·ft in FIELD units). Indexed in
        the same order as `interior_index_map`.
    boundary:
        Shape `(n_boundary_faces,)` float64 - half-transmissibility for
        every boundary face. The flow solver multiplies these by the
        appropriate boundary-condition factor: zero for no-flow (Neumann),
        or `(P_bc - P_cell)` for constant-pressure (Dirichlet).
    nnc:
        Shape `(n_nnc,)` float64 or `None` - transmissibilities for
        non-neighbour connections, in the same order as
        `Grid.nnc_cell_indices`.  `None` when the grid has no NNCs.
    interior_index_map:
        Shape `(n_interior_faces,)` int32 - maps position *i* in
        `interior` to the global face index in `Grid.face_cell_indices`.
        Use this to scatter transmissibilities back to the full face array
        when assembling the pressure matrix.
    boundary_index_map:
        Shape `(n_boundary_faces,)` int32 - maps position *j* in
        `boundary` to the global face index in `Grid.face_cell_indices`.
    """

    interior: FloatArray[OneDimension]
    """Shape `(n_interior_faces,)` - geometric TPFA transmissibilities (mD·ft)."""

    boundary: FloatArray[OneDimension]
    """Shape `(n_boundary_faces,)` - owner half-transmissibilities for boundary faces (mD·ft)."""

    nnc: typing.Optional[FloatArray[OneDimension]]
    """Shape `(n_nnc,)` - NNC transmissibilities, or `None`."""

    interior_index_map: IntArray[OneDimension]
    """Shape `(n_interior_faces,)` - maps interior-array index -> global face index."""

    boundary_index_map: IntArray[OneDimension]
    """Shape `(n_boundary_faces,)` - maps boundary-array index -> global face index."""


def compute_face_transmissibilities(
    grid: Grid,
    rock: RockProperties,
    *,
    net_to_gross: typing.Optional[FloatArray[OneDimension]] = None,
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> FaceTransmissibilities:
    """
    Compute TPFA face transmissibilities for an unstructured polyhedral grid.

    Permeability is projected onto each face normal using the anisotropic
    Peaceman projection:

    ```text
    K_proj = |nx|·Kx + |ny|·Ky + |nz|·Kz
    ```

    where `(nx, ny, nz)` are the components of the face unit normal. This
    is the standard single-point upstream permeability projection used by
    ECLIPSE and OPM Flow.

    **Interior faces** receive the harmonic-mean transmissibility of the two
    sharing cells (owner and neighbour).

    **Boundary faces** receive the owner half-transmissibility only. The
    complementary half is supplied by the boundary condition at solve time.

    After the geometric transmissibilities are assembled, all multiplier arrays
    present on the grid are applied in order:

    1. Directional per-cell MULT arrays (MULTX, MULTX-, MULTY, MULTY-, MULTZ,
       MULTZ-) - the face multiplier is the product of the owner's outward
       multiplier and the neighbour's inward multiplier. For boundary faces
       only the owner's outward multiplier is applied (no neighbour to pair
       with).
    2. Named fault multipliers (MULTFLT) - applied to every face belonging to
       each named fault, whether interior or boundary.
    3. NNC transmissibilities are taken verbatim from `grid.nnc_transmissibilities`
       when present.

    :param grid: Fully constructed `bores.grids.base.Grid`.
    :param rock: `RockProperties` instance holding `absolute_permeability`,
        `net_to_gross`, and `rock_compressibility`.  The permeability arrays
        must be shape `(n_cells,)` matching `grid.n_cells`.
    :param net_to_gross: Optional override for the NTG array.  If `None`,
        `rock.net_to_gross` is used. Pass an all-ones array to ignore NTG.
    :param dtype: NumPy floating dtype for output arrays. Defaults to `np.float64`.
    :returns: `FaceTransmissibilities` named tuple.
    :raises ValueError: If permeability or NTG array lengths do not match `grid.n_cells`.
    """
    dtype = np.dtype(dtype) if dtype is not None else np.float64

    kx = np.asarray(rock.absolute_permeability.x, dtype=dtype)
    ky = np.asarray(rock.absolute_permeability.y, dtype=dtype)
    kz = np.asarray(rock.absolute_permeability.z, dtype=dtype)
    ntg = np.asarray(
        rock.net_to_gross if net_to_gross is None else net_to_gross,
        dtype=dtype,
    )

    n_cells = grid.n_cells
    for name, arr in (("Kx", kx), ("Ky", ky), ("Kz", kz), ("NTG", ntg)):
        if arr.shape != (n_cells,):
            raise ValueError(
                f"{name} array has shape {arr.shape}; expected ({n_cells},) "
                f"to match grid.n_cells."
            )

    # Effective permeability = anisotropic K x NTG
    effective_kx = kx * ntg
    effective_ky = ky * ntg
    effective_kz = kz * ntg

    interior_face_indices = grid.interior_face_indices
    boundary_face_indices = grid.boundary_face_indices

    # Compute harmonic-mean TPFA transmissibilities for all interior faces
    interior_transmissibilities = _compute_interior_tpfa_transmissibilities(
        interior_face_indices=interior_face_indices,
        face_cell_indices=grid.face_cell_indices,
        face_centroids=grid.face_centroids,
        face_areas=grid.face_areas,
        face_unit_normals=grid.face_unit_normals,
        cell_centroids=grid.cell_centroids,  # type: ignore[arg-type]
        effective_kx=effective_kx,
        effective_ky=effective_ky,
        effective_kz=effective_kz,
        dtype=dtype,
    )

    # Compute owner half-transmissibilities for all boundary faces
    boundary_transmissibilities = _compute_boundary_half_transmissibilities(
        boundary_face_indices=boundary_face_indices,
        face_cell_indices=grid.face_cell_indices,
        face_centroids=grid.face_centroids,
        face_areas=grid.face_areas,
        face_unit_normals=grid.face_unit_normals,
        cell_centroids=grid.cell_centroids,  # type: ignore[arg-type]
        effective_kx=effective_kx,
        effective_ky=effective_ky,
        effective_kz=effective_kz,
        dtype=dtype,
    )

    # Apply directional transmissibility multipliers (MULT*) when present
    if grid.has_transmissibility_multipliers:
        interior_transmissibilities, boundary_transmissibilities = (
            _apply_directional_multipliers(
                interior_transmissibilities=interior_transmissibilities,
                boundary_transmissibilities=boundary_transmissibilities,
                interior_face_indices=interior_face_indices,
                boundary_face_indices=boundary_face_indices,
                face_cell_indices=grid.face_cell_indices,
                face_unit_normals=grid.face_unit_normals,
                positive_x_multipliers=grid.positive_x_transmissibility_multipliers,
                negative_x_multipliers=grid.negative_x_transmissibility_multipliers,
                positive_y_multipliers=grid.positive_y_transmissibility_multipliers,
                negative_y_multipliers=grid.negative_y_transmissibility_multipliers,
                positive_z_multipliers=grid.positive_z_transmissibility_multipliers,
                negative_z_multipliers=grid.negative_z_transmissibility_multipliers,
            )
        )

    # Apply named fault multipliers (MULTFLT)
    if (
        grid.fault_face_indices is not None
        and grid.fault_transmissibility_multipliers is not None
    ):
        interior_transmissibilities, boundary_transmissibilities = (
            _apply_fault_multipliers(
                interior_transmissibilities=interior_transmissibilities,
                boundary_transmissibilities=boundary_transmissibilities,
                interior_face_indices=interior_face_indices,
                boundary_face_indices=boundary_face_indices,
                fault_face_indices=grid.fault_face_indices,
                fault_transmissibility_multipliers=grid.fault_transmissibility_multipliers,
            )
        )

    # NNC transmissibilities
    nnc_transmissibilities: typing.Optional[FloatArray[OneDimension]] = None
    if grid.nnc_transmissibilities is not None:
        # Caller-supplied (e.g. from GRDECL NNC keyword).
        nnc_transmissibilities = np.asarray(grid.nnc_transmissibilities, dtype=dtype)
        nan_mask = ~np.isfinite(nnc_transmissibilities)
        if nan_mask.any():
            warnings.warn(
                f"{nan_mask.sum()} NNC transmissibilities are NaN / Inf "
                "(recorded from pinchout geometry without explicit `T` values). "
                "These NNCs will be assigned zero transmissibility. "
                "Supply explicit transmissibilities via the GRDECL NNC keyword "
                "or a post-processing step.",
                stacklevel=2,
            )
            nnc_transmissibilities = nnc_transmissibilities.copy()
            nnc_transmissibilities[nan_mask] = 0.0
    elif grid.n_nnc > 0:
        # Grid has NNC pairs but no pre-computed T: warn, return zeros.
        warnings.warn(
            f"Grid has {grid.n_nnc} NNC pair(s) but no transmissibilities. "
            "Returning zero transmissibility for all NNCs. "
            "Provide `nnc_transmissibilities` to the grid factory to resolve this.",
            stacklevel=2,
        )
        nnc_transmissibilities = np.zeros(grid.n_nnc, dtype=dtype)

    return FaceTransmissibilities(
        interior=interior_transmissibilities.astype(dtype, copy=False),
        boundary=boundary_transmissibilities.astype(dtype, copy=False),
        nnc=nnc_transmissibilities,
        interior_index_map=interior_face_indices.astype(np.int32, copy=False),
        boundary_index_map=boundary_face_indices.astype(np.int32, copy=False),
    )


@numba.njit(parallel=True, cache=True)
def _compute_interior_tpfa_transmissibilities(
    interior_face_indices: IntArray[OneDimension],
    face_cell_indices: IntArray[TwoDimensions],
    face_centroids: FloatArray[TwoDimensions],
    face_areas: FloatArray[OneDimension],
    face_unit_normals: FloatArray[TwoDimensions],
    cell_centroids: FloatArray[TwoDimensions],
    effective_kx: FloatArray[OneDimension],
    effective_ky: FloatArray[OneDimension],
    effective_kz: FloatArray[OneDimension],
    dtype: npt.DTypeLike,
) -> FloatArray[OneDimension]:
    """
    Compute harmonic-mean TPFA transmissibilities for all interior faces.

    For each interior face the half-transmissibility of cell *c* is:

    ```text
    K_c = |nx|·Kx_c + |ny|·Ky_c + |nz|·Kz_c
    T_c = K_c · face_area / d_c
    ```

    where `d_c` is the distance from cell centroid *c* to the face centroid,
    and `(nx, ny, nz)` are the face unit-normal components.

    The full harmonic transmissibility is:

    ```text
    T = T_A · T_B / (T_A + T_B)
    ```

    :param interior_face_indices: Shape `(n_interior,)` - indices of interior
        faces into the global face arrays.
    :param face_cell_indices: Shape `(n_faces, 2)` - owner/neighbour pairs.
    :param face_centroids: Shape `(n_faces, 3)` - face centroid coordinates.
    :param face_areas: Shape `(n_faces,)` - face areas (grid units²).
    :param face_unit_normals: Shape `(n_faces, 3)` - outward unit normals
        from the owner cell.
    :param cell_centroids: Shape `(n_cells, 3)` - cell centroid coordinates.
    :param effective_kx: Shape `(n_cells,)` - effective x-permeability (Kx x NTG).
    :param effective_ky: Shape `(n_cells,)` - effective y-permeability.
    :param effective_kz: Shape `(n_cells,)` - effective z-permeability.
    :param dtype: Output dtype.
    :returns: Shape `(n_interior,)` float64 transmissibility array.
    """
    n_interior = interior_face_indices.shape[0]
    transmissibilities = np.zeros(n_interior, dtype=dtype)

    for idx in numba.prange(n_interior):  # type: ignore
        face_idx = interior_face_indices[idx]

        owner = face_cell_indices[face_idx, 0]
        neighbour = face_cell_indices[face_idx, 1]

        nx = abs(face_unit_normals[face_idx, 0])
        ny = abs(face_unit_normals[face_idx, 1])
        nz = abs(face_unit_normals[face_idx, 2])
        area = face_areas[face_idx]

        fx = face_centroids[face_idx, 0]
        fy = face_centroids[face_idx, 1]
        fz = face_centroids[face_idx, 2]

        # Half-transmissibility: owner
        dx_a = fx - cell_centroids[owner, 0]
        dy_a = fy - cell_centroids[owner, 1]
        dz_a = fz - cell_centroids[owner, 2]
        d_a = (dx_a**2 + dy_a**2 + dz_a**2) ** 0.5

        k_a = (
            nx * effective_kx[owner]
            + ny * effective_ky[owner]
            + nz * effective_kz[owner]
        )
        T_a = k_a * area / d_a if d_a > 0.0 else 0.0

        # Half-transmissibility: neighbour
        dx_b = fx - cell_centroids[neighbour, 0]
        dy_b = fy - cell_centroids[neighbour, 1]
        dz_b = fz - cell_centroids[neighbour, 2]
        d_b = (dx_b * dx_b + dy_b * dy_b + dz_b * dz_b) ** 0.5

        k_b = (
            nx * effective_kx[neighbour]
            + ny * effective_ky[neighbour]
            + nz * effective_kz[neighbour]
        )
        T_b = k_b * area / d_b if d_b > 0.0 else 0.0

        # Harmonic mean
        if T_a + T_b > 0.0:
            transmissibilities[idx] = (T_a * T_b) / (T_a + T_b)

    return transmissibilities


@numba.njit(parallel=True, cache=True)
def _compute_boundary_half_transmissibilities(
    boundary_face_indices: IntArray[OneDimension],
    face_cell_indices: IntArray[TwoDimensions],
    face_centroids: FloatArray[TwoDimensions],
    face_areas: FloatArray[OneDimension],
    face_unit_normals: FloatArray[TwoDimensions],
    cell_centroids: FloatArray[TwoDimensions],
    effective_kx: FloatArray[OneDimension],
    effective_ky: FloatArray[OneDimension],
    effective_kz: FloatArray[OneDimension],
    dtype: npt.DTypeLike,
) -> FloatArray[OneDimension]:
    """
    Compute owner half-transmissibilities for all boundary faces.

    Boundary faces have `face_cell_indices[:, 1] == -1` (no neighbour). Only
    the owner cell contributes a half-transmissibility; the complementary half
    is supplied by the boundary condition at solve time:

    - **No-flow (Neumann = 0)**: multiply by zero - equivalent to omitting
      the face from flux assembly.
    - **Constant-pressure (Dirichlet)**: assemble `T_half x (P_bc - P_cell)`
      as a source term in the pressure equation.
    - **Aquifer influx**: scale `T_half` by the aquifer productivity index.

    The half-transmissibility formula is identical to the per-cell half used
    in `_compute_interior_tpfa_transmissibilities`:

    ```text
    K_owner = |nx|·Kx_owner + |ny|·Ky_owner + |nz|·Kz_owner
    T_half   = K_owner · face_area / d_owner
    ```

    where `d_owner` is the distance from the owner centroid to the face
    centroid.

    :param boundary_face_indices: Shape `(n_boundary,)` - indices of boundary
        faces into the global face arrays.
    :param face_cell_indices: Shape `(n_faces, 2)` - owner/neighbour pairs;
        neighbour is `-1` for all boundary faces.
    :param face_centroids: Shape `(n_faces, 3)` - face centroid coordinates.
    :param face_areas: Shape `(n_faces,)` - face areas (grid units²).
    :param face_unit_normals: Shape `(n_faces, 3)` - outward unit normals
        from the owner cell.
    :param cell_centroids: Shape `(n_cells, 3)` - cell centroid coordinates.
    :param effective_kx: Shape `(n_cells,)` - effective x-permeability (Kx x NTG).
    :param effective_ky: Shape `(n_cells,)` - effective y-permeability.
    :param effective_kz: Shape `(n_cells,)` - effective z-permeability.
    :param dtype: Output dtype.
    :returns: Shape `(n_boundary,)` half-transmissibility array.
    """
    n_boundary = boundary_face_indices.shape[0]
    boundary_transmissibilities = np.zeros(n_boundary, dtype=dtype)

    for idx in numba.prange(n_boundary):  # type: ignore
        face_idx = boundary_face_indices[idx]
        owner = face_cell_indices[face_idx, 0]
        # face_cell_indices[face_idx, 1] == -1 for all boundary faces; unused.

        nx = abs(face_unit_normals[face_idx, 0])
        ny = abs(face_unit_normals[face_idx, 1])
        nz = abs(face_unit_normals[face_idx, 2])
        area = face_areas[face_idx]

        fx = face_centroids[face_idx, 0]
        fy = face_centroids[face_idx, 1]
        fz = face_centroids[face_idx, 2]

        dx = fx - cell_centroids[owner, 0]
        dy = fy - cell_centroids[owner, 1]
        dz = fz - cell_centroids[owner, 2]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5

        k = (
            nx * effective_kx[owner]
            + ny * effective_ky[owner]
            + nz * effective_kz[owner]
        )
        if d > 0.0:
            boundary_transmissibilities[idx] = k * area / d

    return boundary_transmissibilities


@numba.njit(cache=True)
def _apply_directional_multipliers(
    interior_transmissibilities: FloatArray[OneDimension],
    boundary_transmissibilities: FloatArray[OneDimension],
    interior_face_indices: IntArray[OneDimension],
    boundary_face_indices: IntArray[OneDimension],
    face_cell_indices: IntArray[TwoDimensions],
    face_unit_normals: FloatArray,
    positive_x_multipliers: typing.Optional[FloatArray[OneDimension]],
    negative_x_multipliers: typing.Optional[FloatArray[OneDimension]],
    positive_y_multipliers: typing.Optional[FloatArray[OneDimension]],
    negative_y_multipliers: typing.Optional[FloatArray[OneDimension]],
    positive_z_multipliers: typing.Optional[FloatArray[OneDimension]],
    negative_z_multipliers: typing.Optional[FloatArray[OneDimension]],
) -> typing.Tuple[FloatArray[OneDimension], FloatArray[OneDimension]]:
    """
    Scale transmissibilities by per-cell directional multiplier (`MULT*`) arrays (in-place).

    **Interior faces**

    For a face between owner cell *A* and neighbour cell *B* the multiplier is:

    ```text
    multiplier = MULT_forward(A) x MULT_backward(B)
    ```

    where *forward* is the owner's outgoing direction (e.g. positive-x if the
    face normal is predominantly in +x) and *backward* is the neighbour's
    incoming direction (negative-x for the same face).

    **Boundary faces**

    Only the owner's outgoing direction multiplier is applied - there is no
    neighbour cell to pair with:

    ```text
    multiplier = MULT_forward(owner)
    ```

    Direction is determined by the dominant component of the face unit normal
    (largest absolute component).

    :param interior_transmissibilities: Shape `(n_interior,)` - interior transmissibilities
        before applying directional multipliers.
    :param boundary_transmissibilities: Shape `(n_boundary,)` - boundary
        half-transmissibilities before applying directional multipliers.
    :param interior_face_indices: Global face indices for the interior faces.
    :param boundary_face_indices: Global face indices for the boundary faces.
    :param face_cell_indices: Shape `(n_faces, 2)` - owner/neighbour pairs.
    :param face_unit_normals: Shape `(n_faces, 3)` - face unit normals.
    :param positive_x_multipliers: Shape `(n_cells,)` MULTX or `None`.
    :param negative_x_multipliers: Shape `(n_cells,)` MULTX- or `None`.
    :param positive_y_multipliers: Shape `(n_cells,)` MULTY or `None`.
    :param negative_y_multipliers: Shape `(n_cells,)` MULTY- or `None`.
    :param positive_z_multipliers: Shape `(n_cells,)` MULTZ or `None`.
    :param negative_z_multipliers: Shape `(n_cells,)` MULTZ- or `None`.
    :returns: Tuple `(transmissibilities, boundary_transmissibilities)` with
        multipliers applied in-place.
    """
    n_interior = len(interior_face_indices)
    n_boundary = len(boundary_face_indices)

    # Interior faces - owner forward multiplier x neighbour backward multiplier
    for idx in range(n_interior):
        face_idx = int(interior_face_indices[idx])
        owner = int(face_cell_indices[face_idx, 0])
        neighbour = int(face_cell_indices[face_idx, 1])

        nx = abs(float(face_unit_normals[face_idx, 0]))
        ny = abs(float(face_unit_normals[face_idx, 1]))
        nz = abs(float(face_unit_normals[face_idx, 2]))

        multiplier = 1.0
        if nx >= ny and nx >= nz:
            if positive_x_multipliers is not None:
                multiplier *= float(positive_x_multipliers[owner])
            if negative_x_multipliers is not None:
                multiplier *= float(negative_x_multipliers[neighbour])
        elif ny >= nx and ny >= nz:
            if positive_y_multipliers is not None:
                multiplier *= float(positive_y_multipliers[owner])
            if negative_y_multipliers is not None:
                multiplier *= float(negative_y_multipliers[neighbour])
        else:
            if positive_z_multipliers is not None:
                multiplier *= float(positive_z_multipliers[owner])
            if negative_z_multipliers is not None:
                multiplier *= float(negative_z_multipliers[neighbour])

        interior_transmissibilities[idx] *= multiplier

    # Boundary faces - owner forward multiplier only (no neighbour)
    for idx in range(n_boundary):
        face_idx = int(boundary_face_indices[idx])
        owner = int(face_cell_indices[face_idx, 0])

        nx = abs(float(face_unit_normals[face_idx, 0]))
        ny = abs(float(face_unit_normals[face_idx, 1]))
        nz = abs(float(face_unit_normals[face_idx, 2]))

        multiplier = 1.0
        if nx >= ny and nx >= nz:
            if positive_x_multipliers is not None:
                multiplier *= float(positive_x_multipliers[owner])
        elif ny >= nx and ny >= nz:
            if positive_y_multipliers is not None:
                multiplier *= float(positive_y_multipliers[owner])
        else:
            if positive_z_multipliers is not None:
                multiplier *= float(positive_z_multipliers[owner])

        boundary_transmissibilities[idx] *= multiplier

    return interior_transmissibilities, boundary_transmissibilities


@numba.njit(cache=True)
def _apply_fault_multipliers(
    interior_transmissibilities: FloatArray[OneDimension],
    boundary_transmissibilities: FloatArray[OneDimension],
    interior_face_indices: IntArray[OneDimension],
    boundary_face_indices: IntArray[OneDimension],
    fault_face_indices: typing.Mapping[str, IntArray[OneDimension]],
    fault_transmissibility_multipliers: typing.Mapping[str, float],
) -> typing.Tuple[FloatArray[OneDimension], FloatArray[OneDimension]]:
    """
    Apply per-fault (`MULTFLT`) transmissibility multipliers (in-place).

    For each fault name present in *both* `fault_face_indices` and
    `fault_transmissibility_multipliers`, every face belonging to the fault
    is scaled by the corresponding multiplier - whether interior or boundary.

    Faces that appear in `fault_face_indices` but whose fault name has no
    entry in `fault_transmissibility_multipliers` are left unchanged
    (multiplier defaults to 1.0).

    :param interior_transmissibilities: Shape `(n_interior,)` - interior transmissibilities
        before applying fault multipliers.
    :param boundary_transmissibilities: Shape `(n_boundary,)` - boundary
        half-transmissibilities before applying fault multipliers.
    :param interior_face_indices: Global face indices for the interior faces.
    :param boundary_face_indices: Global face indices for the boundary faces.
    :param fault_face_indices: Mapping from fault name to global face index array.
    :param fault_transmissibility_multipliers: Mapping from fault name to scalar multiplier.
    :returns: Tuple `(transmissibilities, boundary_transmissibilities)` with
        fault multipliers applied in-place.
    """
    # Build reverse maps: global face index -> position in each local array.
    # Only interior/boundary faces can carry a finite TPFA transmissibility.
    global_to_interior: typing.Dict[int, int] = {
        int(global_idx): interior_pos
        for interior_pos, global_idx in enumerate(interior_face_indices)
    }
    global_to_boundary: typing.Dict[int, int] = {
        int(global_idx): boundary_pos
        for boundary_pos, global_idx in enumerate(boundary_face_indices)
    }

    for fault_name, face_idx_array in fault_face_indices.items():
        multiplier = fault_transmissibility_multipliers.get(fault_name, 1.0)
        if multiplier == 1.0:
            continue  # No-op, skip.

        for global_idx in face_idx_array:
            interior_pos = global_to_interior.get(int(global_idx))
            if interior_pos is not None:
                interior_transmissibilities[interior_pos] *= multiplier
                continue
            boundary_pos = global_to_boundary.get(int(global_idx))
            if boundary_pos is not None:
                boundary_transmissibilities[boundary_pos] *= multiplier

    return interior_transmissibilities, boundary_transmissibilities
