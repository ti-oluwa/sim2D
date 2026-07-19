"""Resolve well perforations to grid cells."""

import math
import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.base import Serializable
from bores.typing import (
    Boolean,
    IntArray,
    Number,
    NumberArray,
    OneDimension,
    Orientation,
    TwoDimensions,
    UnitConversionTable,
    UnitSystem,
)
from bores.wells.base import MDPerforation, Perforation, Well

__all__ = [
    "PerforationIndex",
    "resolve_perforation_orientation",
    "resolve_perforations_indices",
    "resolve_md_perforations_indices",
]


@attrs.frozen(kw_only=True, slots=True)
class PerforationIndex(Serializable):
    """One resolved (perforation, cell) pair - a connection."""

    perforation: typing.Union[Perforation, MDPerforation]
    cell_index: int
    partial_penetration_fraction: Number
    representative_depth: Number
    """Midpoint true vertical depth of the overlap between `perforation`'s
    interval and this cell."""
    inclination_from_vertical: Number
    """Angle in radians between this connection's local wellbore direction
    and vertical - `0` for a straight-down connection, `pi/2` for a
    horizontal one. `wells.hydraulics` reads this directly; nothing in
    that package derives an inclination itself."""
    well_index: typing.Optional[Number] = None
    """Connection factor for this connection. `None` until computed."""
    unit_system: UnitSystem = UnitSystem.FIELD

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Returns a  new `PerforationIndex` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New PerforationIndex with representative_depth and
            well_index converted to target.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        well_index_factor = factors["permeability"] * factors["length"]
        return attrs.evolve(
            self,
            representative_depth=self.representative_depth * factors["length"],
            well_index=self.well_index * well_index_factor
            if self.well_index is not None
            else None,
            unit_system=target,
        )


def resolve_perforation_orientation(perforation: Perforation) -> Orientation:
    """
    Resolve which axis a `Perforation`'s completion runs along.

    Resolution order: `perforation.direction` if set (and not
    `Orientation.UNSET`); otherwise `Orientation.Z`.

    :param perforation: The perforation being resolved.
    :returns: `Orientation.X`, `Orientation.Y`, or `Orientation.Z`.
    """
    if (
        perforation.direction is not None
        and perforation.direction is not Orientation.UNSET
    ):
        return perforation.direction
    return Orientation.Z


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


def _get_vertical_column_candidates(
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


def _compute_local_vertical_extent_aabb(
    grid: Grid, cell_index: int, x: Number, y: Number
) -> typing.Optional[typing.Tuple[Number, Number]]:
    """
    Get vertical extent of cell `cell_index`'s axis-aligned bounding box.

    Every point within the cell's horizontal footprint sees the same
    `[cell_min_z, cell_max_z]` window under this method - `x`/`y` are
    accepted for signature parity with `_compute_local_vertical_extent_exact` and
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


def _point_in_polygon_3d(
    point: typing.Tuple[Number, Number, Number],
    polygon_xyz: NumberArray[TwoDimensions],
    normal: NumberArray[OneDimension],
    tolerance: Number = 1e-9,
) -> bool:
    """
    Point-in-polygon test for a planar polygon in 3-D, given a point
    already known to lie on (or very near) the polygon's plane.

    Projects both the polygon and the point onto whichever pair of axes
    `normal` is *least* aligned with (drops the axis `normal` is most
    aligned with), then runs `_point_in_polygon_2d` on that projection.

    Dropping a fixed axis (as the purely-vertical-line special case could
    get away with, since every face it tested was near-horizontal) would
    collapse a steeply-tilted face's projection to near-zero area here,
    since a general 3-D segment can exit through a face of any
    orientation.

    :param point: 3-D point on the polygon's plane.
    :param polygon_xyz: Shape `(n, 3)` polygon vertices, in order.
    :param normal: The polygon's (unit) normal.
    :param tolerance: Forwarded to `_point_in_polygon_2d`.
    :returns: `True` if `point` is inside or on the boundary of the polygon.
    """
    abs_normal = (abs(float(normal[0])), abs(float(normal[1])), abs(float(normal[2])))
    dominant_axis = abs_normal.index(max(abs_normal))
    u_axis, v_axis = (axis for axis in (0, 1, 2) if axis != dominant_axis)
    polygon_uv = polygon_xyz[:, [u_axis, v_axis]]
    return _point_in_polygon_2d(
        point[u_axis],
        point[v_axis],
        polygon_xy=polygon_uv,  # type: ignore[arg-type]
        tolerance=tolerance,
    )


def _compute_local_vertical_extent_exact(
    grid: Grid, cell_index: int, x: Number, y: Number
) -> typing.Optional[typing.Tuple[Number, Number]]:
    """
    Get exact vertical extent of cell `cell_index` at horizontal position
    `(x, y)`.

    A vertical line through `(x, y)` pierces a convex cell's boundary at
    exactly two faces. For each face of the cell, intersects the line
    with that face's plane (via the face centroid and outward normal),
    then keeps the intersection only if `(x, y)` falls inside that face's
    actual polygon (projected to the horizontal plane) rather than just
    its infinite plane.

    Faces whose outward normal is close to horizontal
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
    intersects. Only valid for a `well` with no `trajectory` - see
    `resolve_md_perforations_indices` for one with a trajectory.

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
    :raises ValidationError: If `well.trajectory` is set (use
        `resolve_md_perforations_indices` instead), or a perforation's
        depth interval overlaps no cell at all (dangling completion).
    """
    if well.trajectory is not None:
        raise ValidationError(
            f"Well {well.name!r} has a trajectory; use "
            "resolve_md_perforations_indices instead."
        )

    if grid.unit_system != well.unit_system:
        raise ValidationError(
            f"Grid `unit_system` ({grid.unit_system.value}) != well "
            f"{well.name!r}'s `unit_system` ({well.unit_system.value})."
        )

    x, y = well.surface_location
    tolerance = (
        horizontal_tolerance
        if horizontal_tolerance is not None
        else _default_horizontal_tolerance(grid, x, y)
    )
    candidates = _get_vertical_column_candidates(grid, x, y, tolerance)
    extent_func = (
        _compute_local_vertical_extent_aabb
        if method == "aabb"
        else _compute_local_vertical_extent_exact
    )

    results: typing.List[PerforationIndex] = []
    for perforation in well.open_perforations:
        assert isinstance(perforation, Perforation)
        inclination = (
            0.0
            if resolve_perforation_orientation(perforation) is Orientation.Z
            else math.pi / 2.0
        )
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
                            inclination_from_vertical=inclination,
                            unit_system=well.unit_system,
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
                        inclination_from_vertical=inclination,
                        unit_system=well.unit_system,
                    )
                )

        if not matches:
            raise ValidationError(
                f"`Perforation` [{perforation.top_depth}, {perforation.bottom_depth}] "
                f"on well {well.name!r} does not overlap any grid cell near "
                f"surface_location={well.surface_location} (dangling completion)."
            )
        results.extend(matches)

    return tuple(results)


def _cell_contains_point(
    grid: Grid,
    cell_index: int,
    point: typing.Tuple[Number, Number, Number],
    tolerance: Number = 1e-6,
) -> bool:
    """
    `True` if `point` lies inside (or on the boundary of, within
    `tolerance`) cell `cell_index`'s convex hull - every face's outward
    normal points away from `point`.

    :param grid: Grid providing face geometry.
    :param cell_index: Cell to test.
    :param point: Point to test.
    :param tolerance: Signed-distance tolerance for the boundary case.
    :returns: `True` if inside.
    """
    for face_idx in grid.get_cell_face_indices(cell_index):
        face_idx = int(face_idx)
        normal = grid.get_face_normal_for_cell(face_idx, cell_index)
        centroid = grid.face_centroids[face_idx]
        signed_distance = (
            normal[0] * (point[0] - centroid[0])
            + normal[1] * (point[1] - centroid[1])
            + normal[2] * (point[2] - centroid[2])
        )
        if signed_distance > tolerance:
            return False
    return True


def _locate_cell(
    grid: Grid, point: typing.Tuple[Number, Number, Number], search_radius: Number
) -> typing.Optional[int]:
    """
    Find the grid cell containing `point`.

    Starts from the nearest-centroid guess (cheap, right most of the
    time); if that guess isn't actually the containing cell (common near a
    distorted or unstructured cell boundary), falls back to an
    expanding-radius search over every cell whose centroid is nearby,
    testing each with `_cell_contains_point`.

    :param grid: Grid to search.
    :param point: Point to locate.
    :param search_radius: Initial fallback search radius; doubles each
        retry, up to 6 attempts.
    :returns: Containing cell index, or `None` if not found within the
        search budget (point outside the grid, or resolution too coarse).
    """
    nearest = int(grid.find_nearest_cell(*point))
    if _cell_contains_point(grid, nearest, point):
        return nearest

    radius = search_radius
    for _ in range(6):
        candidates = grid.find_cells_in_radius(point[0], point[1], point[2], radius)
        for candidate in candidates:
            candidate = int(candidate)
            if _cell_contains_point(grid, candidate, point):
                return candidate
        radius *= 2.0
    return None


def _get_segment_face_intersection(
    start: typing.Tuple[Number, Number, Number],
    direction: typing.Tuple[Number, Number, Number],
    grid: Grid,
    face_index: int,
    cell_index: int,
) -> typing.Optional[Number]:
    """
    Get a parameter `t` (`point = start + t * direction`) at which the ray from
    `start` along `direction` crosses `face_index`'s plane *and* actually
    lands within that face's polygon - `None` if the ray is (near-)
    parallel to the plane, or the crossing point falls outside the
    polygon.

    :param start: Ray origin.
    :param direction: Ray direction (any length - `t` is in units of
        `direction`'s own length, i.e. `t=1` reaches `start + direction`).
    :param grid: Grid providing face geometry.
    :param face_index: Face to test.
    :param cell_index: Cell `face_index` belongs to (for its outward normal).
    :returns: `t`, or `None`.
    """
    normal = grid.get_face_normal_for_cell(face_index, cell_index)
    denominator = (
        normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2]
    )
    if abs(denominator) < 1e-12:
        return None

    centroid = grid.face_centroids[face_index]
    numerator = (
        normal[0] * (centroid[0] - start[0])
        + normal[1] * (centroid[1] - start[1])
        + normal[2] * (centroid[2] - start[2])
    )
    t = numerator / denominator

    point = (
        start[0] + t * direction[0],
        start[1] + t * direction[1],
        start[2] + t * direction[2],
    )
    face_vertices = grid.get_face_vertex_coordinates(face_index)
    if not _point_in_polygon_3d(point, face_vertices, normal):
        return None
    return t


def _walk_segment_through_grid(
    grid: Grid,
    start: typing.Tuple[Number, Number, Number],
    end: typing.Tuple[Number, Number, Number],
    start_md: Number,
    end_md: Number,
    search_radius: Number,
) -> typing.List[typing.Tuple[int, Number, Number]]:
    """
    Walk the straight 3-D segment from `start` to `end` through `grid`,
    cell by cell.

    Locates the cell containing `start`, then repeatedly finds the nearest
    face (in walk order) the segment exits the current cell through and
    crosses to that face's neighbor (`face_cell_indices`), accumulating
    `(cell_index, entry_md, exit_md)` for each cell entered, until the
    segment reaches `end` or exits the grid through a boundary face (no
    neighbor - the trajectory leaves the active domain; the walk simply
    stops there rather than guessing where it re-enters).

    :param grid: Grid to walk.
    :param start: Segment start point.
    :param end: Segment end point.
    :param start_md: Measured depth at `start`.
    :param end_md: Measured depth at `end`.
    :param search_radius: Forwarded to `_locate_cell` for the initial cell.
    :returns: Ordered list of `(cell_index, entry_md, exit_md)`. Empty if
        `start` isn't inside any cell.
    """
    direction = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    segment_length = math.sqrt(sum(component**2 for component in direction))
    if segment_length == 0.0:
        return []

    current_cell = _locate_cell(grid, start, search_radius)
    if current_cell is None:
        return []

    results: typing.List[typing.Tuple[int, Number, Number]] = []
    current_t = 0.0
    md_span = end_md - start_md
    max_visits = 4 * grid.n_cells + 64

    for _ in range(max_visits):
        best_t: typing.Optional[Number] = None
        best_face: typing.Optional[int] = None
        for face_idx in grid.get_cell_face_indices(current_cell):
            face_idx = int(face_idx)
            t = _get_segment_face_intersection(
                start, direction, grid, face_idx, current_cell
            )
            if t is None or t <= current_t + 1e-9 or t > 1.0 + 1e-9:
                continue
            if best_t is None or t < best_t:
                best_t, best_face = t, face_idx

        if best_t is None or best_face is None:
            # Segment ends inside current_cell without crossing another face.
            results.append(
                (
                    current_cell,
                    start_md + current_t * md_span,
                    start_md + 1.0 * md_span,
                )
            )
            break

        exit_t = min(best_t, 1.0)
        results.append(
            (
                current_cell,
                start_md + current_t * md_span,
                start_md + exit_t * md_span,
            )
        )
        if best_t >= 1.0 - 1e-9:
            break

        owner, neighbour = (
            grid.face_cell_indices[best_face, 0],
            grid.face_cell_indices[best_face, 1],
        )
        next_cell = neighbour if owner == current_cell else owner
        if next_cell < 0:
            break  # boundary face - segment leaves the active grid domain

        current_cell = next_cell
        current_t = best_t

    return results


def resolve_md_perforations_indices(
    grid: Grid,
    well: Well,
    *,
    search_radius: typing.Optional[Number] = None,
) -> typing.Tuple[PerforationIndex, ...]:
    """
    Resolve every open `MDPerforation` on `well` to the `Grid` cell(s) its
    measured-depth interval passes through. Only valid for a `well` with a
    `trajectory` set - see `resolve_perforations_indices` for one without.

    For each open perforation, gets the trajectory polyline vertices
    covering `[top_md, bottom_md]` (`WellTrajectory.stations_between`),
    then walks each leg of that polyline through `grid`
    (`_walk_segment_through_grid`) to find every cell it passes through,
    the measured-depth sub-range within each, and (from the leg's
    direction) the local inclination from vertical.

    :param grid: Grid to resolve against.
    :param well: `Well` whose perforations are resolved. Not modified.
    :param search_radius: Forwarded to `_locate_cell` for each leg's
        starting-cell search. `None` derives a default from the grid's
        local cell size at `well.surface_location`.
    :returns: Tuple of `PerforationIndex`, one per (perforation, cell) the
        trajectory passes through in that perforation's MD range, in
        along-hole order. A perforation crossing several cells yields
        several entries.
    :raises ValidationError: If `well.trajectory` is `None` (use
        `resolve_perforations_indices` instead), or a perforation's MD
        range doesn't intersect any grid cell at all (trajectory passes
        entirely outside the active grid over that range).
    """
    if well.trajectory is None:
        raise ValidationError(
            f"Well {well.name!r} has no trajectory; use "
            "resolve_perforations_indices instead."
        )

    if grid.unit_system != well.unit_system:
        raise ValidationError(
            f"Grid `unit_system` ({grid.unit_system.value}) != well "
            f"{well.name!r}'s `unit_system` ({well.unit_system.value})."
        )

    trajectory = well.trajectory
    x, y = well.surface_location
    radius = (
        search_radius
        if search_radius is not None
        else _default_horizontal_tolerance(grid, x, y)
    )

    results: typing.List[PerforationIndex] = []
    for perforation in well.open_perforations:
        assert isinstance(perforation, MDPerforation)
        top_md = perforation.top_md
        bottom_md = max(perforation.bottom_md, perforation.top_md + 1e-9)
        # A point perforation (top_md == bottom_md) has no direction to
        # walk - nudge to an infinitesimal interval so it still resolves
        # to exactly one cell via the same walk machinery, rather than a
        # separate point-lookup code path.

        stations = trajectory.stations_between(top_md, bottom_md)
        matches: typing.List[PerforationIndex] = []

        for previous, current in zip(stations, stations[1:]):
            start = (previous.x, previous.y, previous.z)
            end = (current.x, current.y, current.z)
            leg_dx, leg_dy, leg_dz = (
                end[0] - start[0],
                end[1] - start[1],
                end[2] - start[2],
            )
            leg_length = math.sqrt(leg_dx**2 + leg_dy**2 + leg_dz**2)
            if leg_length == 0.0:
                continue
            inclination = math.acos(max(-1.0, min(1.0, leg_dz / leg_length)))

            crossings = _walk_segment_through_grid(
                grid,
                start,
                end,
                previous.measured_depth,
                current.measured_depth,
                radius,
            )
            for cell_index, entry_md, exit_md in crossings:
                sub_length = exit_md - entry_md
                if sub_length <= 0.0:
                    continue
                matches.append(
                    PerforationIndex(
                        perforation=perforation,
                        cell_index=cell_index,
                        partial_penetration_fraction=(
                            1.0
                            if perforation.is_point_perforation
                            else sub_length
                            / (perforation.bottom_md - perforation.top_md)
                        ),
                        representative_depth=trajectory.position_at(
                            0.5 * (entry_md + exit_md)
                        )[2],
                        inclination_from_vertical=inclination,
                        unit_system=well.unit_system,
                    )
                )

        if not matches:
            raise ValidationError(
                f"`MDPerforation` [{perforation.top_md}, {perforation.bottom_md}] on "
                f"well {well.name!r} does not intersect any grid cell along its "
                "trajectory (dangling completion, or the trajectory passes "
                "outside the active grid over this range)."
            )
        results.extend(matches)

    return tuple(results)
