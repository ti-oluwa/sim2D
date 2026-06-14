import typing
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial import Voronoi
from typing_extensions import TypeAlias

from bores.errors import (
    InvalidFaceConnectivityError,
    InvalidGridError,
    InvalidPointArrayError,
    ValidationError,
)
from bores.grid.grid import Grid
from bores.grids.factories.base import assemble_grid, build_csr_face_arrays

VertexCoordinates2D: TypeAlias = npt.NDArray[np.floating]
"""Shape `(n_points, 2)` — 2-D (x, y) seed or vertex coordinates."""

VertexCoordinates3D: TypeAlias = npt.NDArray[np.floating]
"""Shape `(n_points, 3)` — 3-D (x, y, z) vertex coordinates."""

CellConnectivity: TypeAlias = List[List[int]]
"""Ragged list of vertex-index lists; one inner list per cell."""

FaceVertexList: TypeAlias = List[int]
"""Ordered list of vertex indices for a single face (CCW from owner)."""

CanonicalFaceKey: TypeAlias = Tuple[int, ...]
"""Sorted tuple of vertex indices used as a face deduplication key."""

CellFaceTable: TypeAlias = List[List[int]]
"""Per-element-type local face definitions; each entry is a list of local
vertex indices wound CCW from outside (outward normal)."""

CoordArray: TypeAlias = npt.NDArray[np.floating]
"""Corner-point COORD array, shape `(NY+1, NX+1, 6)`."""

ZcornArray: TypeAlias = npt.NDArray[np.floating]
"""Corner-point ZCORN array, shape `(NZ*2, NY*2, NX*2)`."""

ActnumArray: TypeAlias = npt.NDArray[np.integer]
"""Corner-point ACTNUM array, shape `(NZ, NY, NX)`; 1 = active, 0 = inactive."""

# Element face tables

#: Outward-pointing face definitions for standard element types.
#: Each value is a list of faces; each face is a list of *local* vertex
#: indices wound counter-clockwise when viewed from outside the cell.
ELEMENT_FACE_TABLES: Dict[str, CellFaceTable] = {
    # ------------------------------------------------------------------
    # Tetrahedron (4 vertices: v0 v1 v2 = base CCW from below, v3 = apex)
    # ------------------------------------------------------------------
    "tetra": [
        [0, 2, 1],  # base — outward normal points downward (away from apex)
        [0, 1, 3],  # front face opposite v2
        [1, 2, 3],  # right face opposite v0
        [0, 3, 2],  # left face opposite v1
    ],
    # ------------------------------------------------------------------
    # Hexahedron / brick (8 vertices, VTK ordering)
    # Bottom face: v0 v1 v2 v3 (z = z_bot); top face: v4 v5 v6 v7 (z = z_top)
    # v0=(x0,y0,z_bot) v1=(x1,y0,z_bot) v2=(x1,y1,z_bot) v3=(x0,y1,z_bot)
    # v4=(x0,y0,z_top) v5=(x1,y0,z_top) v6=(x1,y1,z_top) v7=(x0,y1,z_top)
    # ------------------------------------------------------------------
    "hexahedron": [
        [0, 3, 2, 1],  # bottom  — outward = −z
        [4, 5, 6, 7],  # top     — outward = +z
        [0, 1, 5, 4],  # front   — outward = −y
        [2, 3, 7, 6],  # back    — outward = +y
        [0, 4, 7, 3],  # left    — outward = −x
        [1, 2, 6, 5],  # right   — outward = +x
    ],
    # ------------------------------------------------------------------
    # Wedge / triangular prism (6 vertices)
    # Bottom triangle: v0 v1 v2 (CCW from below); top: v3 v4 v5
    # ------------------------------------------------------------------
    "wedge": [
        [0, 2, 1],  # bottom triangle — outward = −z
        [3, 4, 5],  # top triangle    — outward = +z
        [0, 1, 4, 3],  # lateral face 01 — outward away from opposite edge
        [1, 2, 5, 4],  # lateral face 12
        [0, 3, 5, 2],  # lateral face 20
    ],
    # ------------------------------------------------------------------
    # Pyramid (5 vertices: v0 v1 v2 v3 = base CCW from below, v4 = apex)
    # ------------------------------------------------------------------
    "pyramid": [
        [0, 3, 2, 1],  # base   — outward = −z
        [0, 1, 4],  # front  face
        [1, 2, 4],  # right  face
        [2, 3, 4],  # back   face
        [0, 4, 3],  # left   face
    ],
}

#: Mapping from VTK integer cell-type codes to element type names.
VTK_CELL_TYPE_NAMES: Dict[int, str] = {
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}


def make_2d_voronoi_grid(
    *,
    seed_xy_coordinates: VertexCoordinates2D,
    z_top: float = 0.0,
    dz: float = 1.0,
    bounding_box: typing.Optional[Tuple[float, float, float, float]] = None,
    metadata: typing.Optional[dict] = None,
) -> Grid:
    """
    Factory for 2-D Voronoi (PEBI) grids extruded uniformly in depth.

    Computes a 2-D Voronoi tessellation from seed points, clips infinite
    ridges and boundary cells to a bounding box, then extrudes each
    polygonal column into a prism of constant thickness `dz`.

    Clipping uses `shapely` when available for robustness; falls back to
    a simple bounding-box intersection otherwise.

    Example usage:

    ```python
    import numpy as np

    factory = VoronoiGridFactory()
    seeds = np.random.rand(50, 2) * 1000.0
    grid = factory(
        seed_xy_coordinates=seeds,
        z_top=2000.0,
        dz=10.0,
        bounding_box=(0.0, 1000.0, 0.0, 1000.0),
    )
    ```

    Builds a 2-D Voronoi grid extruded uniformly in depth.

    :param seed_xy_coordinates: Shape `(n_seeds, 2)` array of (x, y)
        generator / seed point coordinates.
    :param z_top: Depth of the top face of the extruded grid (positive
        downward). The bottom face is at `z_top + dz`.
    :param dz: Uniform layer thickness in the z direction.
    :param bounding_box: `(x_min, x_max, y_min, y_max)` clipping
        rectangle.  If `None`, inferred as the seed point bounding box
        padded by 5 % in each direction.
    :param metadata: Optional metadata dictionary.
    :returns: A fully initialised :class:`~bores.grids.base.Grid`.
    :raises ValidationError: If fewer than 4 seed points are provided,
        `dz` is non-positive, or seed points are collinear.
    """
    seeds = np.asarray(seed_xy_coordinates, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValidationError(
            f"seed_xy_coordinates must be shape (n_seeds, 2); got {seeds.shape!r}."
        )
    if seeds.shape[0] < 4:
        raise ValidationError(
            "At least 4 seed points are required for a valid Voronoi tessellation."
        )
    if dz <= 0.0:
        raise ValidationError(f"dz must be strictly positive; got {dz!r}.")

    bbox = _resolve_bounding_box(seeds, bounding_box)
    clipped_regions, clipped_ridges = _clip_voronoi(seeds, bbox)
    (
        vertex_coordinates,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
    ) = _build_voronoi_grid_arrays(
        seeds=seeds,
        clipped_regions=clipped_regions,
        clipped_ridges=clipped_ridges,
        z_top=z_top,
        dz=dz,
    )
    return assemble_grid(
        vertex_coordinates,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
        metadata=metadata,
    )


def _resolve_bounding_box(
    seeds: npt.NDArray[np.float64],
    bounding_box: typing.Optional[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """
    Compute a bounding box, padding the seed extents by 5 % if not supplied.

    :param seeds: Shape `(n_seeds, 2)` seed coordinate array.
    :param bounding_box: Caller-supplied `(x_min, x_max, y_min, y_max)`, or None.
    :returns: `(x_min, x_max, y_min, y_max)` float tuple.
    """
    if bounding_box is not None:
        x_min, x_max, y_min, y_max = bounding_box
    else:
        pad_x = (seeds[:, 0].max() - seeds[:, 0].min()) * 0.05 or 1.0
        pad_y = (seeds[:, 1].max() - seeds[:, 1].min()) * 0.05 or 1.0
        x_min = seeds[:, 0].min() - pad_x
        x_max = seeds[:, 0].max() + pad_x
        y_min = seeds[:, 1].min() - pad_y
        y_max = seeds[:, 1].max() + pad_y
    return float(x_min), float(x_max), float(y_min), float(y_max)


def _clip_voronoi(
    seeds: npt.NDArray[np.float64],
    bounding_box: Tuple[float, float, float, float],
) -> Tuple[
    List[Optional[npt.NDArray[np.float64]]],
    List[Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int, int]]],
]:
    """
    Compute and clip the 2-D Voronoi tessellation.

    Clips both infinite and finite regions to `bounding_box`.  Uses
    `shapely` when available for a robust polygon intersection; falls back
    to a simple convex-hull intersection approach otherwise.

    :param seeds: Shape `(n_seeds, 2)` seed array.
    :param bounding_box: `(x_min, x_max, y_min, y_max)`.
    :returns: Tuple `(clipped_regions, clipped_ridges)` where:

        - `clipped_regions[i]` is either `None` (cell clipped out) or a
            float64 array of shape `(n_verts_i, 2)` — the ordered polygon
            vertices for seed `i`'s Voronoi cell.
        - `clipped_ridges[i]` is either `None` or a tuple
            `(point_a_xy, point_b_xy, seed_idx_left, seed_idx_right)`.
    """
    x_min, x_max, y_min, y_max = bounding_box
    vor = Voronoi(seeds)

    # Extend infinite ridges to a large rectangle just outside the bbox
    far = max(x_max - x_min, y_max - y_min) * 10.0
    center = seeds.mean(axis=0)

    # Build a map from seed-index to list of (other_seed, ridge_vertices)
    # to reconstruct full polygons including clipped infinite ridges.
    # We use shapely when available for the actual polygon clipping.
    try:
        from shapely.geometry import Polygon  # noqa: PLC0415

        use_shapely = True
    except ImportError:
        warnings.warn(
            "shapely is not installed. Voronoi clipping will fall back to "
            "a less robust implementation. Install shapely for best results.",
            ImportWarning,
            stacklevel=4,
        )
        use_shapely = False

    clipped_regions: List[Optional[npt.NDArray[np.float64]]] = []
    bbox_polygon = None
    if use_shapely:
        from shapely.geometry import box as shapely_box  # noqa: PLC0415

        bbox_polygon = shapely_box(x_min, y_min, x_max, y_max)

    for seed_idx in range(len(seeds)):
        region_idx = vor.point_region[seed_idx]
        region_vert_indices = vor.regions[region_idx]

        if len(region_vert_indices) == 0:
            clipped_regions.append(None)
            continue

        if -1 not in region_vert_indices:
            # Finite region — all vertices are real
            polygon_xy = vor.vertices[region_vert_indices]
        else:
            # Infinite region — reconstruct boundary using ridge information
            polygon_xy = _reconstruct_infinite_region(
                seed_idx=seed_idx,
                vor=vor,
                center=center,
                far=far,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )

        if polygon_xy is None or len(polygon_xy) < 3:
            clipped_regions.append(None)
            continue

        # Clip to bounding box
        if use_shapely:
            from shapely.geometry import Polygon as ShapelyPolygon  # noqa: PLC0415

            cell_poly = ShapelyPolygon(polygon_xy)
            clipped = cell_poly.intersection(bbox_polygon)
            if clipped.is_empty or clipped.area < 1e-14:
                clipped_regions.append(None)
            else:
                coords = np.array(clipped.exterior.coords[:-1])  # drop closing dup
                clipped_regions.append(coords)
        else:
            # Simple bbox clip: just clip each vertex to inside bbox
            inside = (
                (polygon_xy[:, 0] >= x_min)
                & (polygon_xy[:, 0] <= x_max)
                & (polygon_xy[:, 1] >= y_min)
                & (polygon_xy[:, 1] <= y_max)
            )
            if not inside.any():
                clipped_regions.append(None)
            else:
                clipped_regions.append(polygon_xy[inside])

    # Build ridge list: one entry per ridge (pair of seeds sharing an edge)
    clipped_ridges = []
    for ridge_idx, (seed_pair, ridge_vert_indices) in enumerate(
        zip(vor.ridge_points, vor.ridge_vertices)
    ):
        seed_left, seed_right = int(seed_pair[0]), int(seed_pair[1])
        if -1 in ridge_vert_indices:
            # Infinite ridge — extend to bbox boundary
            finite_vert_idx = (
                ridge_vert_indices[0]
                if ridge_vert_indices[1] == -1
                else ridge_vert_indices[1]
            )
            tangent = seeds[seed_right] - seeds[seed_left]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else 1
            pt_finite = vor.vertices[finite_vert_idx]
            pt_infinite = pt_finite + normal * far
            # Clip the infinite endpoint to bbox
            pt_clipped = _clip_point_to_bbox(
                pt_finite, pt_infinite, x_min, x_max, y_min, y_max
            )
            if pt_clipped is None:
                clipped_ridges.append(None)
                continue
            ridge_pts = (pt_finite, pt_clipped)
        else:
            a = vor.vertices[ridge_vert_indices[0]]
            b = vor.vertices[ridge_vert_indices[1]]
            ridge_pts = (a, b)

        clipped_ridges.append((ridge_pts[0], ridge_pts[1], seed_left, seed_right))

    return clipped_regions, clipped_ridges


def _reconstruct_infinite_region(
    seed_idx: int,
    vor,
    center: npt.NDArray[np.float64],
    far: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Optional[npt.NDArray[np.float64]]:
    """Reconstruct the polygon for a Voronoi region containing infinite vertices.

    Replaces each `-1` vertex index in the region with an extended point
    outside the bounding box so that shapely can clip it properly.

    :param seed_idx: Index of the seed whose region is being reconstructed.
    :param vor: `scipy.spatial.Voronoi` result object.
    :param center: Centroid of all seed points (used for normal direction).
    :param far: Distance to extend infinite ridges (should be >> bbox diagonal).
    :param x_min: Bounding box x minimum.
    :param x_max: Bounding box x maximum.
    :param y_min: Bounding box y minimum.
    :param y_max: Bounding box y maximum.
    :returns: Shape `(n_verts, 2)` polygon vertex array, or `None` if
        reconstruction fails.
    """
    region_idx = vor.point_region[seed_idx]
    region_vert_indices = vor.regions[region_idx]
    vertices = list(vor.vertices)

    result_polygon: List[npt.NDArray[np.float64]] = []
    for vert_idx in region_vert_indices:
        if vert_idx >= 0:
            result_polygon.append(vor.vertices[vert_idx])
        else:
            # Find the ridge that has this -1 vertex and belongs to seed_idx
            for ridge_vert_ids, ridge_seed_ids in zip(
                vor.ridge_vertices, vor.ridge_points
            ):
                if -1 in ridge_vert_ids and seed_idx in ridge_seed_ids:
                    other_seed = int(
                        ridge_seed_ids[0]
                        if ridge_seed_ids[1] == seed_idx
                        else ridge_seed_ids[1]
                    )
                    finite_v_idx = (
                        ridge_vert_ids[0]
                        if ridge_vert_ids[1] == -1
                        else ridge_vert_ids[1]
                    )
                    tangent = vor.points[other_seed] - vor.points[seed_idx]
                    normal = np.array([-tangent[1], tangent[0]])
                    normal_magnitude = np.linalg.norm(normal)
                    if normal_magnitude > 0:
                        normal /= normal_magnitude
                    midpoint = (vor.points[seed_idx] + vor.points[other_seed]) / 2.0
                    if np.dot(normal, midpoint - center) < 0:
                        normal = -normal
                    extended = vor.vertices[finite_v_idx] + normal * far
                    result_polygon.append(extended)
                    break

    return np.array(result_polygon) if len(result_polygon) >= 3 else None


def _clip_point_to_bbox(
    pt_inside: npt.NDArray[np.float64],
    pt_outside: npt.NDArray[np.float64],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Optional[npt.NDArray[np.float64]]:
    """Find the intersection of the segment `pt_inside→pt_outside` with the bbox boundary.

    :param pt_inside: A point known to be inside (or on the boundary of) the bbox.
    :param pt_outside: A point extended far outside the bbox.
    :param x_min: Bounding box x minimum.
    :param x_max: Bounding box x maximum.
    :param y_min: Bounding box y minimum.
    :param y_max: Bounding box y maximum.
    :returns: The intersection point as a 2-D float64 array, or `None`
        if no intersection is found.
    """
    d = pt_outside - pt_inside
    t_min, t_max = 0.0, 1.0
    for dim, (lo, hi) in enumerate([(x_min, x_max), (y_min, y_max)]):
        if abs(d[dim]) < 1e-14:
            if pt_inside[dim] < lo or pt_inside[dim] > hi:
                return None
        else:
            t_lo = (lo - pt_inside[dim]) / d[dim]
            t_hi = (hi - pt_inside[dim]) / d[dim]
            t_entry = min(t_lo, t_hi)
            t_exit = max(t_lo, t_hi)
            t_min = max(t_min, t_entry)
            t_max = min(t_max, t_exit)
    if t_min > t_max:
        return None
    return pt_inside + t_max * d


def _build_voronoi_grid_arrays(
    seeds: npt.NDArray[np.float64],
    clipped_regions: List[Optional[npt.NDArray[np.float64]]],
    clipped_ridges: List,
    z_top: float,
    dz: float,
) -> Tuple[
    VertexCoordinates3D,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Assemble 3-D grid arrays from clipped 2-D Voronoi regions.

    Each 2-D polygonal Voronoi cell is extruded to a prism with top at
    `z_top` and bottom at `z_top + dz`.  For each seed (cell):

    - Top polygon face (normal = −z, outward from cell looking upward).
    - Bottom polygon face (normal = +z, outward downward).
    - One rectangular lateral face per polygon edge.

    :param seeds: Shape `(n_seeds, 2)` seed array.
    :param clipped_regions: Per-seed polygon vertex arrays (or None for
        clipped-out cells).
    :param clipped_ridges: Ridge information from :meth:`_clip_voronoi`.
    :param z_top: Depth of the top face.
    :param dz: Layer thickness.
    :returns: Tuple of CSR arrays ready for
        :meth:`BaseGridFactory._assemble_grid`.
    """
    z_bottom = z_top + dz

    # Collect only valid (non-None) cells and build a seed→cell mapping
    cell_seed_indices: List[int] = []
    cell_polygons: List[npt.NDArray[np.float64]] = []
    seed_to_cell: Dict[int, int] = {}
    for seed_idx, polygon in enumerate(clipped_regions):
        if polygon is not None and len(polygon) >= 3:
            cell_idx = len(cell_seed_indices)
            seed_to_cell[seed_idx] = cell_idx
            cell_seed_indices.append(seed_idx)
            cell_polygons.append(polygon)

    if len(cell_polygons) == 0:
        raise InvalidGridError(
            "No valid Voronoi cells remain after clipping to the bounding box."
        )

    # Build unique 3-D vertex list: for each polygon vertex, create bottom and top
    # We register all polygon vertices globally and deduplicate by proximity.
    all_3d_vertices: List[npt.NDArray[np.float64]] = []
    vertex_dedup: Dict[Tuple[float, float, float], int] = {}

    def _get_or_add_vertex(x: float, y: float, z: float) -> int:
        """Return index of an existing nearby vertex or register a new one.

        :param x: X-coordinate.
        :param y: Y-coordinate.
        :param z: Z-coordinate.
        :returns: Global vertex index.
        """
        key = (round(x, 10), round(y, 10), round(z, 10))
        if key not in vertex_dedup:
            vertex_dedup[key] = len(all_3d_vertices)
            all_3d_vertices.append(np.array([x, y, z]))
        return vertex_dedup[key]

    # Per-cell face vertex lists (CCW from outside that cell = outward normal)
    per_cell_face_vertex_lists: List[List[FaceVertexList]] = [
        [] for _ in range(len(cell_polygons))
    ]

    for cell_idx, polygon_xy in enumerate(cell_polygons):
        n_polygon_verts = len(polygon_xy)
        # Register all top and bottom vertices for this polygon
        top_vertex_indices = [
            _get_or_add_vertex(float(polygon_xy[k, 0]), float(polygon_xy[k, 1]), z_top)
            for k in range(n_polygon_verts)
        ]
        bottom_vertex_indices = [
            _get_or_add_vertex(
                float(polygon_xy[k, 0]), float(polygon_xy[k, 1]), z_bottom
            )
            for k in range(n_polygon_verts)
        ]

        # Top face: outward normal = −z (upward), CCW from above.
        # Looking from above (−z direction): the polygon vertices should be CW
        # so that CCW from outside (= from above = −z) gives normal −z.
        # Original polygon_xy is in some order; reverse it for outward−z top face.
        top_face = list(reversed(top_vertex_indices))
        per_cell_face_vertex_lists[cell_idx].append(top_face)

        # Bottom face: outward normal = +z (downward), CCW from below.
        bottom_face = list(bottom_vertex_indices)  # original CCW order gives +z normal
        per_cell_face_vertex_lists[cell_idx].append(bottom_face)

        # Lateral faces: one per polygon edge
        for edge_idx in range(n_polygon_verts):
            next_idx = (edge_idx + 1) % n_polygon_verts
            # Edge from polygon_xy[edge_idx] to polygon_xy[next_idx]
            a_top = top_vertex_indices[edge_idx]
            a_bot = bottom_vertex_indices[edge_idx]
            b_top = top_vertex_indices[next_idx]
            b_bot = bottom_vertex_indices[next_idx]

            # 2-D edge vector a→b
            a_xy = polygon_xy[edge_idx]
            b_xy = polygon_xy[next_idx]
            # Seed (cell centroid) for orientation test
            seed_xy = seeds[cell_seed_indices[cell_idx]]

            # signed_area > 0 means seed is to the LEFT of a→b
            signed_area = 0.5 * (
                (b_xy[0] - a_xy[0]) * (seed_xy[1] - a_xy[1])
                - (b_xy[1] - a_xy[1]) * (seed_xy[0] - a_xy[0])
            )
            # When seed is LEFT of a→b: outward from seed is to the RIGHT of a→b.
            # Face winding CCW from outside (outward from seed):
            # [a_bot, a_top, b_top, b_bot] gives outward normal
            # when seed is to the RIGHT (i.e. signed_area < 0).
            # When signed_area > 0 (seed left of a→b): use reversed winding.
            if signed_area > 0:
                lateral_face = [a_bot, b_bot, b_top, a_top]
            else:
                lateral_face = [a_bot, a_top, b_top, b_bot]

            per_cell_face_vertex_lists[cell_idx].append(lateral_face)

    vertex_coordinate_array = np.array(all_3d_vertices, dtype=np.float64)

    (
        vertex_coordinates_out,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
    ) = build_csr_face_arrays(
        vertex_coordinate_array, per_cell_face_vertex_lists
    )

    return (
        vertex_coordinates_out,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
    )
