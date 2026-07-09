"""
Resolve well perforations to grid cells.

Depends on `wells.data` and `bores.grids.base.Grid`. Nothing here holds
state between calls.
"""

import math
import typing

import attrs
import numpy as np

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.base import Serializable
from bores.typing import IntArray, Number, OneDimension
from bores.wells.data import Perforation, WellSpec

__all__ = ["PerforationIndex", "resolve_perforations"]


@attrs.frozen(kw_only=True, slots=True)
class PerforationIndex(Serializable):
    """One resolved (perforation, cell) pair."""

    perforation: Perforation
    cell_index: int
    partial_penetration_fraction: Number
    well_index: typing.Optional[Number] = None
    """`None` here - populated by `wells.index.build_wells_indices`."""


def _default_horizontal_tolerance(grid: Grid, x: Number, y: Number) -> Number:
    """
    Derive a default horizontal search radius for `resolve_perforations`.

    Evaluated locally at the well's own position (half the diagonal of the
    nearest cell), since cell size can vary across an unstructured grid.

    :param grid: Grid to sample local cell size from.
    :param x: Well surface x-coordinate.
    :param y: Well surface y-coordinate.
    :returns: Radius in the grid's length units.
    """
    z = grid.bounding_box[4]
    nearest = grid.find_nearest_cell(x, y, z)
    dx = grid.cell_length_x[nearest]
    dy = grid.cell_length_y[nearest]
    return 0.5 * math.sqrt(dx**2 + dy**2)


def _vertical_column_candidates(
    grid: Grid, x: Number, y: Number, tolerance: Number
) -> IntArray[OneDimension]:
    """
    Return every cell index whose centroid falls within `tolerance` of
    `(x, y)` in the horizontal plane, across all depths.

    `grid.find_cells_in_radius` is a 3-D sphere query, not a vertical
    column query, so this queries once with a radius large enough to span
    the full depth range from the grid's mid-depth, then filters down to
    the true horizontal distance using `grid.cell_centroids[:, :2]`.

    :param grid: Grid to query.
    :param x: Well surface x-coordinate.
    :param y: Well surface y-coordinate.
    :param tolerance: Horizontal radius (grid length units).
    :returns: Sorted 1-D int array of candidate cell indices, any depth, no
        duplicates, not yet filtered by perforation depth.
    """
    z_min, z_max = grid.bounding_box[4], grid.bounding_box[5]
    mid_z = 0.5 * (z_min + z_max)
    half_depth_span = 0.5 * (z_max - z_min)
    query_radius = math.sqrt(tolerance**2 + half_depth_span**2)

    raw_candidates = grid.find_cells_in_radius(x, y, mid_z, query_radius)
    if raw_candidates.size == 0:
        return raw_candidates
    
    assert grid.cell_centroids is not None
    centroids_xy = grid.cell_centroids[raw_candidates, :2]
    horizontal_distances = np.sqrt(
        (centroids_xy[:, 0] - x) ** 2 + (centroids_xy[:, 1] - y) ** 2
    )
    filtered = raw_candidates[horizontal_distances <= tolerance]
    return typing.cast(IntArray[OneDimension], np.sort(filtered))


def _overlap_length_aabb(
    grid: Grid, cell_index: int, top_depth: Number, bottom_depth: Number
) -> Number:
    """
    v1 (AABB, approximate). Overlap length between `[top_depth, bottom_depth]`
    and cell `cell_index`'s z-extent.

    Returns 0.0 (no overlap) rather than raising when the interval and the
    cell's AABB don't intersect - the caller filters zero-overlap results
    out.

    :param grid: Grid providing `cell_min_xyz`/`cell_max_xyz`.
    :param cell_index: Candidate cell to test.
    :param top_depth: Perforation top depth.
    :param bottom_depth: Perforation bottom depth.
    :returns: Overlap length (>= 0), same length units as `grid`.
    """
    cell_min_z = grid.cell_min_xyz[cell_index, 2]
    cell_max_z = grid.cell_max_xyz[cell_index, 2]
    overlap = min(bottom_depth, cell_max_z) - max(top_depth, cell_min_z)
    return max(overlap, 0.0)


def _overlap_length_exact(
    grid: Grid, cell_index: int, top_depth: Number, bottom_depth: Number
) -> Number:
    """
    v2 (exact, point-in-polyhedron) - not implemented in Phase 1.

    Where v1 brackets against the cell's bounding box, this will sample the
    perforation's depth interval against the cell's actual vertical extent
    at `(x, y)` by intersecting a vertical line with the cell's bounding
    faces, rather than its AABB.

    :param grid: Grid providing face/vertex geometry.
    :param cell_index: Candidate cell to test.
    :param top_depth: Perforation top depth.
    :param bottom_depth: Perforation bottom depth.
    :returns: Exact overlap length (>= 0), same length units as `grid`.
    """
    raise NotImplementedError(
        "method='exact' is not implemented yet; use method='aabb'."
    )


def resolve_perforations(
    grid: Grid,
    well: WellSpec,
    *,
    horizontal_tolerance: typing.Optional[Number] = None,
    method: typing.Literal["aabb", "exact"] = "aabb",
) -> typing.Tuple[PerforationIndex, ...]:
    """
    Resolve every open `Perforation` on `well` to the `Grid` cell(s) it
    intersects.

    For each open perforation, finds every cell whose vertical extent
    overlaps `[top_depth, bottom_depth]` at `well.surface_location`, and
    computes a partial-penetration fraction per cell from the overlap
    length. A point perforation (`top_depth == bottom_depth`) resolves to
    exactly one cell with `partial_penetration_fraction = 1.0`.

    :param grid: Grid to resolve against.
    :param well: `WellSpec` whose perforations are resolved. Not modified.
    :param horizontal_tolerance: Search radius around `surface_location`
        used to gather vertical-column candidates, in `well.unit_system`
        length units. `None` derives a default from the grid's local cell
        size at `surface_location`.
    :param method: `"aabb"` (v1, default) or `"exact"` (v2, not yet
        implemented).
    :returns: Tuple of `PerforationIndex`, one per (perforation, cell) pair
        that overlaps.
    :raises ValidationError: If a perforation's depth interval overlaps no
        cell at all (dangling completion).
    """
    x, y = well.surface_location
    tolerance = (
        horizontal_tolerance
        if horizontal_tolerance is not None
        else _default_horizontal_tolerance(grid, x, y)
    )
    candidates = _vertical_column_candidates(grid, x, y, tolerance)
    overlap_fn = _overlap_length_aabb if method == "aabb" else _overlap_length_exact

    results: typing.List[PerforationIndex] = []
    for perforation in well.open_perforations:
        matches: typing.List[PerforationIndex] = []

        if perforation.is_point_perforation:
            depth = perforation.top_depth
            for cell_index in candidates:
                cell_index = int(cell_index)
                if (
                    grid.cell_min_xyz[cell_index, 2]
                    <= depth
                    <= grid.cell_max_xyz[cell_index, 2]
                ):
                    matches.append(
                        PerforationIndex(
                            perforation=perforation,
                            cell_index=cell_index,
                            partial_penetration_fraction=1.0,
                        )
                    )
                    break
        else:
            for cell_index in candidates:
                cell_index = int(cell_index)
                overlap = overlap_fn(
                    grid, cell_index, perforation.top_depth, perforation.bottom_depth
                )
                if overlap <= 0.0:
                    continue
                matches.append(
                    PerforationIndex(
                        perforation=perforation,
                        cell_index=cell_index,
                        partial_penetration_fraction=overlap / perforation.length,
                    )
                )

        if not matches:
            raise ValidationError(
                f"Perforation [{perforation.top_depth}, {perforation.bottom_depth}] "
                f"on well {well.name!r} does not overlap any grid cell near "
                f"surface_location={well.surface_location} (dangling completion)."
            )
        results.extend(matches)

    return tuple(results)
