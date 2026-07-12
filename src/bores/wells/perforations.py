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
from bores.typing import (
    Boolean,
    IntArray,
    Number,
    NumberArray,
    OneDimension,
    TwoDimensions,
)
from bores.wells.base import Perforation, Well

__all__ = ["PerforationIndex", "resolve_perforations_indices"]


@attrs.frozen(kw_only=True, slots=True)
class PerforationIndex(Serializable):
    """One resolved (perforation, cell) pair - a connection."""

    perforation: Perforation
    cell_index: int
    partial_penetration_fraction: Number
    representative_depth: Number
    """Midpoint depth of the overlap between `perforation`'s interval and
    this cell's vertical extent."""
    well_index: typing.Optional[Number] = None
    """Connection factor for this connection. `None` until computed."""


def _default_horizontal_tolerance(grid: Grid, x: Number, y: Number) -> Number:
    """
    Derive a default horizontal search radius for `resolve_perforations_indices`.

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


def _local_vertical_extent_aabb(
    grid: Grid, cell_index: int, x: Number, y: Number
) -> typing.Optional[typing.Tuple[Number, Number]]:
    """
    Vertical extent of cell `cell_index`'s axis-aligned bounding box.

    Every point within the cell's horizontal footprint sees the same
    `[cell_min_z, cell_max_z]` window under this method - `x`/`y` are
    accepted for signature parity with `_local_vertical_extent_exact` and
    otherwise unused.

    :param grid: Grid providing `cell_min_xyz`/`cell_max_xyz`.
    :param cell_index: Cell to measure.
    :param x: Unused.
    :param y: Unused.
    :returns: `(top_depth, bottom_depth)`, always.
    """
    return (
        grid.cell_min_xyz[cell_index, 2],
        grid.cell_max_xyz[cell_index, 2],
    )


def _point_on_segment(
    px: Number,
    py: Number,
    x1: Number,
    y1: Number,
    x2: Number,
    y2: Number,
    tolerance: Number,
) -> Boolean:
    """True if `(px, py)` lies on segment `(x1, y1)-(x2, y2)` within `tolerance`."""
    segment_length = math.hypot(x2 - x1, y2 - y1)
    if segment_length < tolerance:
        return math.hypot(px - x1, py - y1) < tolerance
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) / segment_length > tolerance:
        return False
    dot = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
    return -tolerance <= dot <= (segment_length**2 + tolerance)


def _point_in_polygon_2d(
    x: Number,
    y: Number,
    polygon_xy: NumberArray[TwoDimensions],
    tolerance: Number = 1e-9,
) -> bool:
    """
    Even-odd ray-casting point-in-polygon test, with a boundary tolerance
    so a point falling exactly on a shared cell face is counted as inside
    rather than dropped by floating-point roundoff.

    :param x: Query x-coordinate.
    :param y: Query y-coordinate.
    :param polygon_xy: Shape `(n, 2)` polygon vertices, in order.
    :param tolerance: Distance tolerance for the boundary case.
    :returns: `True` if `(x, y)` is inside or on the boundary of the polygon.
    """
    n = polygon_xy.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon_xy[i, 0]), float(polygon_xy[i, 1])
        xj, yj = float(polygon_xy[j, 0]), float(polygon_xy[j, 1])
        if _point_on_segment(x, y, xi, yi, xj, yj, tolerance):
            return True
        if (yi > y) != (yj > y):
            x_intersect = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _local_vertical_extent_exact(
    grid: Grid, cell_index: int, x: Number, y: Number
) -> typing.Optional[typing.Tuple[Number, Number]]:
    """
    Exact vertical extent of cell `cell_index` at horizontal position
    `(x, y)`.

    A vertical line through `(x, y)` pierces a convex cell's boundary at
    exactly two faces. For each face of the cell, intersects the line
    with that face's plane (via the face centroid and outward normal),
    then keeps the intersection only if `(x, y)` falls inside that face's
    actual polygon (projected to the horizontal plane) rather than just
    its infinite plane. Faces whose outward normal is close to horizontal
    are skipped outright as a vertical line doesn't pierce a vertical
    (side) face at a well-defined single point, and those faces bound the
    cell's sides, not its top or bottom.

    :param grid: Grid providing face topology and geometry.
    :param cell_index: Cell to measure.
    :param x: Horizontal x-coordinate of the vertical line.
    :param y: Horizontal y-coordinate of the vertical line.
    :returns: `(top_depth, bottom_depth)`, or `None` if `(x, y)` does not actually
        fall within this cell's horizontal footprint.
    """
    crossings: typing.List[float] = []
    for face_idx in grid.get_cell_face_indices(cell_index):
        face_idx = int(face_idx)
        normal = grid.get_face_normal_for_cell(face_idx, cell_index)
        normal_z = normal[2]
        if abs(normal_z) < 1e-9:
            continue

        centroid = grid.face_centroids[face_idx]
        z_intersect = (
            centroid[2]
            - (normal[0] * (x - centroid[0]) + normal[1] * (y - centroid[1])) / normal_z
        )

        face_vertex_xy = grid.get_face_vertex_coordinates(face_idx)[:, :2]
        if _point_in_polygon_2d(x, y, polygon_xy=face_vertex_xy):  # type: ignore[arg-type]
            crossings.append(z_intersect)

    if not crossings:
        return None
    return min(crossings), max(crossings)


def resolve_perforations_indices(
    grid: Grid,
    well: Well,
    *,
    horizontal_tolerance: typing.Optional[Number] = None,
    method: typing.Literal["aabb", "exact"] = "aabb",
) -> typing.Tuple[PerforationIndex, ...]:
    """
    Resolve every open `Perforation` on `well` to the `Grid` cell(s) it
    intersects.

    For each open perforation, finds every cell whose vertical extent
    overlaps `[top_depth, bottom_depth]` at `well.surface_location`, and
    computes a partial-penetration fraction and a representative depth per
    cell from the overlap window. A point perforation
    (`top_depth == bottom_depth`) resolves to exactly one cell with
    `partial_penetration_fraction = 1.0`.

    :param grid: Grid to resolve against.
    :param well: `Well` whose perforations are resolved. Not modified.
    :param horizontal_tolerance: Search radius around `surface_location`
        used to gather vertical-column candidates, in `well.unit_system`
        length units. `None` derives a default from the grid's local cell
        size at `surface_location`.
    :param method: `"aabb"` (default) or `"exact"`.
    :returns: Tuple of `PerforationIndex`, one per (perforation, cell) pair
        that overlaps - a single `Perforation` spanning multiple layers
        yields multiple `PerforationIndex` entries.
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
    extent_func = (
        _local_vertical_extent_aabb
        if method == "aabb"
        else _local_vertical_extent_exact
    )

    results: typing.List[PerforationIndex] = []
    for perforation in well.open_perforations:
        matches: typing.List[PerforationIndex] = []

        if perforation.is_point_perforation:
            depth = perforation.top_depth
            for cell_index in candidates:
                cell_index = int(cell_index)
                extent = extent_func(grid, cell_index, x, y)
                if extent is None:
                    continue

                top_depth, bottom_depth = extent
                if top_depth <= depth <= bottom_depth:
                    matches.append(
                        PerforationIndex(
                            perforation=perforation,
                            cell_index=cell_index,
                            partial_penetration_fraction=1.0,
                            representative_depth=depth,
                        )
                    )
                    break
        else:
            for cell_index in candidates:
                cell_index = int(cell_index)
                extent = extent_func(grid, cell_index, x, y)
                if extent is None:
                    continue

                top_depth, bottom_depth = extent
                overlap_top = max(perforation.top_depth, top_depth)
                overlap_bottom = min(perforation.bottom_depth, bottom_depth)
                length = max(overlap_bottom - overlap_top, 0.0)
                if length <= 0.0:
                    continue
                matches.append(
                    PerforationIndex(
                        perforation=perforation,
                        cell_index=cell_index,
                        partial_penetration_fraction=length / perforation.length,
                        representative_depth=0.5 * (overlap_top + overlap_bottom),
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
