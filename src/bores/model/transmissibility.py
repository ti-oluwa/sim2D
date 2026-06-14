"""Precomputed geometric face transmissibilities for structured 3D grids."""

import typing

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import NamedTuple

from bores.correlations.core import compute_harmonic_mean
from bores.precision import get_dtype
from bores.typing import ThreeDimensionalGrid

__all__ = ["FaceTransmissibilities", "build_face_transmissibilities"]


class FaceTransmissibilities(NamedTuple):
    """
    Precomputed geometric face transmissibilities for all interior faces in x, y, z.

    Each array is shaped (nx, ny, nz). Entry [i, j, k] stores the transmissibility
    of the *forward* face:

        x: face between cell (i,j,k) and (i+1,j,k)   units: mD·ft (= mD·ft²/ft)
        y: face between cell (i,j,k) and (i,j+1,k)
        z: face between cell (i,j,k) and (i,j,k+1)

    The last slice in each direction has no forward neighbour and holds zeros.
    """

    x: ThreeDimensionalGrid
    y: ThreeDimensionalGrid
    z: ThreeDimensionalGrid


def build_face_transmissibilities(
    permeability_x: ThreeDimensionalGrid,
    permeability_y: ThreeDimensionalGrid,
    permeability_z: ThreeDimensionalGrid,
    thickness_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    dtype: typing.Optional[npt.DTypeLike] = None,
) -> FaceTransmissibilities:
    """
    Precompute geometric face transmissibilities `T_geo = k_harmonic * A / L` for
    every forward-facing interface in x, y, and z on the grid.

    Result arrays are shaped (nx + 2, ny + 2, nz + 2) where (nx + 2, ny + 2, nz + 2) = padded grid shape.
    Entry [i, j, k] is the transmissibility of the face between (i,j,k) and:

        T_x: (i+1, j,   k  )
        T_y: (i,   j+1, k  )
        T_z: (i,   j,   k+1)

    The last row/col/layer in each direction is set to zero (no forward neighbour).

    :param permeability_x: Permeability grid for flow in x-direction (mD).
    :param permeability_y: Permeability grid for flow in y-direction (mD).
    :param permeability_z: Permeability grid for flow in z-direction (mD).
    :param thickness_grid: Cell thickness grid (ft).
    :param net_to_gross_grid: Net-to-gross ratio grid (fraction).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param dtype: NumPy dtype for output arrays.
    :return: `FaceTransmissibilities` named tuple with x, y, z arrays (mD·ft).
    """
    Tx, Ty, Tz = _compute_face_transmissibilities(
        Kx=permeability_x,
        Ky=permeability_y,
        Kz=permeability_z,
        thickness_grid=thickness_grid,
        net_to_gross_grid=net_to_gross_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        dtype=dtype if dtype is not None else get_dtype(),
    )
    return FaceTransmissibilities(x=Tx, y=Ty, z=Tz)


@numba.njit(parallel=True, cache=True)
def _compute_face_transmissibilities(
    Kx: ThreeDimensionalGrid,
    Ky: ThreeDimensionalGrid,
    Kz: ThreeDimensionalGrid,
    thickness_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    nx, ny, nz = Kx.shape

    Tx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=dtype)
    Ty = np.zeros((nx + 2, ny + 2, nz + 2), dtype=dtype)
    Tz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=dtype)

    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                cell_thickness = thickness_grid[i, j, k]

                # X forward face only exists if east neighbour is real
                if i + 1 < nx:
                    kx_harmonic = compute_harmonic_mean(
                        Kx[i, j, k] * net_to_gross_grid[i, j, k],
                        Kx[i + 1, j, k] * net_to_gross_grid[i + 1, j, k],
                    )
                    east_thickness = thickness_grid[i + 1, j, k]
                    east_harmonic_thickness = compute_harmonic_mean(
                        cell_thickness, east_thickness
                    )
                    flow_area_x = cell_size_y * east_harmonic_thickness
                    Tx[i + 1, j + 1, k + 1] = kx_harmonic * flow_area_x / cell_size_x
                else:
                    # East boundary face mirrors boundary cell, since no real east neighbour
                    kx_harmonic = Kx[i, j, k] * net_to_gross_grid[i, j, k]
                    flow_area_x = cell_size_y * cell_thickness
                    Tx[i + 1, j + 1, k + 1] = (
                        2.0 * kx_harmonic * flow_area_x / cell_size_x
                    )

                # West boundary face. Only cell i=0 owns this slot
                if i == 0:
                    kx_harmonic = Kx[i, j, k] * net_to_gross_grid[i, j, k]
                    flow_area_x = cell_size_y * cell_thickness
                    Tx[0, j + 1, k + 1] = 2.0 * kx_harmonic * flow_area_x / cell_size_x

                # Y forward face only exists if south neighbour is real
                if j + 1 < ny:
                    ky_harmonic = compute_harmonic_mean(
                        Ky[i, j, k] * net_to_gross_grid[i, j, k],
                        Ky[i, j + 1, k] * net_to_gross_grid[i, j + 1, k],
                    )
                    south_thickness = thickness_grid[i, j + 1, k]
                    south_harmonic_thickness = compute_harmonic_mean(
                        cell_thickness, south_thickness
                    )
                    flow_area_y = cell_size_x * south_harmonic_thickness
                    Ty[i + 1, j + 1, k + 1] = ky_harmonic * flow_area_y / cell_size_y
                else:
                    # South boundary face mirrors boundary cell, since no real south neighbour
                    ky_harmonic = Ky[i, j, k] * net_to_gross_grid[i, j, k]
                    flow_area_y = cell_size_x * cell_thickness
                    Ty[i + 1, j + 1, k + 1] = (
                        2.0 * ky_harmonic * flow_area_y / cell_size_y
                    )

                # North boundary face. Only cell j=0 owns this slot
                if j == 0:
                    ky_harmonic = Ky[i, j, k] * net_to_gross_grid[i, j, k]
                    flow_area_y = cell_size_x * cell_thickness
                    Ty[i + 1, 0, k + 1] = 2.0 * ky_harmonic * flow_area_y / cell_size_y

                # Z forward face only exists if bottom neighbour is real
                if k + 1 < nz:
                    kz_harmonic = compute_harmonic_mean(
                        Kz[i, j, k] * net_to_gross_grid[i, j, k],
                        Kz[i, j, k + 1] * net_to_gross_grid[i, j, k + 1],
                    )
                    bottom_thickness = thickness_grid[i, j, k + 1]
                    bottom_harmonic_thickness = compute_harmonic_mean(
                        cell_thickness, bottom_thickness
                    )
                    flow_area_z = cell_size_x * cell_size_y
                    Tz[i + 1, j + 1, k + 1] = (
                        kz_harmonic * flow_area_z / bottom_harmonic_thickness
                    )
                else:
                    # Bottom boundary face mirrors boundary cell, since no real bottom neighbour
                    kz_harmonic = Kz[i, j, k] * net_to_gross_grid[i, j, k]
                    flow_area_z = cell_size_x * cell_size_y
                    Tz[i + 1, j + 1, k + 1] = (
                        2.0 * kz_harmonic * flow_area_z / cell_thickness
                    )

                # Top boundary face. Only cell k=0 owns this slot
                if k == 0:
                    kz_harmonic = Kz[i, j, k] * net_to_gross_grid[i, j, k]
                    flow_area_z = cell_size_x * cell_size_y
                    Tz[i + 1, j + 1, 0] = (
                        2.0 * kz_harmonic * flow_area_z / cell_thickness
                    )

    return Tx, Ty, Tz
