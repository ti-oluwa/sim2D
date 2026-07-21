"""
Convenience factories for building `BoundaryRegion` / `BoundaryConditions`
objects from a `Grid`, without hand-deriving boundary face positions.

Building a `BoundaryRegion` by hand requires: pulling `Grid.boundary_face_indices`,
figuring out which of those global face indices actually sit on the flank of
the domain you care about, then converting the survivors back into
*positions* within `boundary_face_indices` - which is the indexing
`BoundaryRegion.face_positions` actually expects, not global face indices.

None of that is exposed as a one-call helper elsewhere in the grid or
reservoir modules, and it's easy to get subtly wrong by hand:

- Confusing global face indices with positions into `boundary_face_indices`.
- Trusting `Grid.face_unit_normals`'s stored sign without checking it's
  actually outward-pointing for the face in question.
- Picking up boundary faces that happen to point in roughly the right
  direction (e.g. on a rough or faulted flank) without actually sitting at
  the domain's edge.

This module classifies every boundary face geometrically - by its
(re-derived, defensively outward-checked) normal direction *and* its
proximity to the grid's bounding-box extremity on that axis - into one of
six `Side`s, then hands back ready-to-use `BoundaryRegion`/`BoundaryConditions`
objects.

Works for both structured (Cartesian/corner-point) and genuinely
unstructured grids, since it never relies on `Grid.dimensions` /
(i, j, k) indexing - only on face geometry, which every `Grid` has.

Note on `MAPAXES`: a deck's `MAPAXES` rotation is only ever applied when
converting a `Grid` for PyVista rendering (`grids.utils.as_pyvista_grid`) -
`Grid`'s own `face_centroids`/`face_unit_normals`/`bounding_box` are always
in the raw, unrotated GRDECL frame, same as everywhere else internal to
BORES. So classification here always agrees with "west"/"east"/etc.
everywhere else in the codebase, regardless of whether the source deck has
a `MAPAXES` card.
"""

import enum
import logging
import typing

import numpy as np

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.reservoir.boundary.base import BoundaryCondition
from bores.reservoir.boundary.conditions import BoundaryConditions, BoundaryRegion
from bores.typing import IntCellArray, UnitSystem

logger = logging.getLogger(__name__)

__all__ = [
    "Side",
    "classify_boundary_faces",
    "make_boundary_region",
    "make_axis_aligned_boundary_conditions",
]


class Side(enum.Enum):
    """
    One of the six axis-aligned flanks of a grid's bounding box.

    `WEST`/`EAST` - the X axis (min / max).
    `SOUTH`/`NORTH` - the Y axis (min / max).
    `TOP`/`BOTTOM` - the Z axis. Depth increases downward throughout BORES
    (positive-down convention), so `TOP` is the min-Z (shallowest) flank and
    `BOTTOM` is the max-Z (deepest) flank.
    """

    WEST = "west"
    EAST = "east"
    SOUTH = "south"
    NORTH = "north"
    TOP = "top"
    BOTTOM = "bottom"


_ALIASES: typing.Dict[str, Side] = {
    "left": Side.WEST,
    "right": Side.EAST,
    "front": Side.SOUTH,
    "back": Side.NORTH,
    "up": Side.TOP,
    "down": Side.BOTTOM,
}

# axis index -> (negative-normal side, positive-normal side)
_AXIS_SIDES: typing.Dict[int, typing.Tuple[Side, Side]] = {
    0: (Side.WEST, Side.EAST),
    1: (Side.SOUTH, Side.NORTH),
    2: (Side.TOP, Side.BOTTOM),
}


def _resolve_side(side: typing.Union[Side, str]) -> Side:
    """
    Resolve a `Side`, its `.value` string, or a common alias
    ('left'/'right'/'front'/'back'/'up'/'down') to a `Side` member.

    :param side: `Side` member, value string, or alias.
    :returns: Resolved `Side`.
    :raises ValidationError: If `side` doesn't resolve to a known `Side`.
    """
    if isinstance(side, Side):
        return side
    key = side.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    try:
        return Side(key)
    except ValueError as exc:
        valid = sorted({member.value for member in Side} | set(_ALIASES))
        raise ValidationError(
            f"Unknown boundary side {side!r}. Valid values: {valid}."
        ) from exc


def classify_boundary_faces(
    grid: Grid, *, tolerance: float = 1e-3
) -> typing.Dict[Side, IntCellArray]:
    """
    Classify every boundary face of `grid` into one of the six `Side`s.

    For each boundary face:

    1. Its outward normal is re-derived defensively as
       `sign(dot(face_unit_normals[face], face_centroid - owner_cell_centroid))`
       rather than trusting `Grid.face_unit_normals`'s stored sign directly -
       cheap, and removes any dependency on trusting that convention held
       for every face.
    2. The axis whose (re-oriented) normal component has the largest
       magnitude decides X/Y/Z; the sign of that component decides which of
       the two `Side`s on that axis.
    3. The face's centroid must also fall within `tolerance` (a fraction of
       that axis's extent) of the grid's bounding-box extremity on that
       axis. This excludes faces that happen to point roughly outward along
       an axis (e.g. on a rough or faulted flank, or a stair-stepped
       corner-point edge) without actually sitting at the domain's true
       edge on that axis.

    A face that doesn't clear the extremity check on its dominant axis is
    left unassigned. `classify_boundary_faces` is a best-effort geometric
    classification for the common case of a roughly box-shaped domain, not
    a guarantee that every boundary face gets a side. Call
    `Grid.n_boundary_faces` and compare against the total length of the
    returned arrays if you need to know how many faces were left out.

    :param grid: Grid to classify. Must have resolved face geometry
        (`face_unit_normals`, `face_centroids`, `bounding_box`) - true for
        any `Grid` built through a grid factory.
    :param tolerance: Fraction of the axis extent a face's centroid may sit
        away from the bounding-box extremity and still count. `0.0` requires
        exact alignment (only safe for a perfectly orthogonal, unrotated
        Cartesian/corner-point grid); the default `1e-3` tolerates ordinary
        floating-point and minor geometric irregularity.
    :returns: Mapping from every `Side` to the *positions* (0-based, into
        `Grid.boundary_face_indices`, not global face indices) of the
        faces assigned to that side, as an `int32` array. A `Side` with no
        matching faces maps to an empty array rather than being omitted.
    :raises ValidationError: If `tolerance` is negative, or the grid's face
        geometry hasn't been resolved.
    """
    if tolerance < 0.0:
        raise ValidationError(f"`tolerance` must be >= 0.0; got {tolerance!r}.")

    if grid.face_unit_normals is None or grid.face_centroids is None:
        raise ValidationError(
            "`grid` has no `face_unit_normals`/`face_centroids`. Grid "
            "geometry must be resolved (e.g. by building it through a grid "
            "factory) before classifying boundary faces."
        )
    if grid.cell_centroids is None:
        raise ValidationError(
            "`grid` has no `cell_centroids`; cannot re-derive outward face orientation."
        )

    boundary = grid.boundary_face_indices
    empty = {side: np.empty(0, dtype=np.int32) for side in Side}
    if len(boundary) == 0:
        return empty  # type: ignore[return-value]

    owners = grid.face_cell_indices[boundary, 0]
    face_centroids = grid.face_centroids[boundary]
    cell_centroids = grid.cell_centroids[owners]

    # Re-derive outward orientation defensively rather than trusting
    # face_unit_normals' stored sign for every face.
    raw_normals = grid.face_unit_normals[boundary]
    outward_ref = face_centroids - cell_centroids
    flip = np.sign(np.einsum("ij,ij->i", raw_normals, outward_ref))
    flip[flip == 0.0] = 1.0
    normals = raw_normals * flip[:, None]

    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounding_box
    extents = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64)
    mins = np.array([xmin, ymin, zmin], dtype=np.float64)
    maxs = np.array([xmax, ymax, zmax], dtype=np.float64)

    dominant_axis = np.argmax(np.abs(normals), axis=1)

    result: typing.Dict[Side, typing.List[int]] = {side: [] for side in Side}
    for position in range(len(boundary)):
        axis = int(dominant_axis[position])
        extent = extents[axis] or 1.0
        coord = face_centroids[position, axis]
        near_min = abs(coord - mins[axis]) <= tolerance * extent
        near_max = abs(coord - maxs[axis]) <= tolerance * extent
        negative_side, positive_side = _AXIS_SIDES[axis]

        if normals[position, axis] < 0.0 and near_min:
            result[negative_side].append(position)
        elif normals[position, axis] > 0.0 and near_max:
            result[positive_side].append(position)
        # else: face's dominant-axis normal doesn't clear the extremity
        # check - left unassigned (e.g. a face on a rough/faulted flank).

    classified = {
        side: np.asarray(positions, dtype=np.int32)
        for side, positions in result.items()
    }

    n_assigned = sum(len(positions) for positions in classified.values())
    if n_assigned < len(boundary):
        logger.warning(
            "%d of %d boundary faces could not be assigned to a side "
            "(tolerance=%r). Increase `tolerance`, or handle those faces "
            "with an explicit `BoundaryRegion` built from your own face "
            "positions.",
            len(boundary) - n_assigned,
            len(boundary),
            tolerance,
        )
    return classified  # type: ignore[return-value]


def make_boundary_region(
    name: str,
    grid: Grid,
    side: typing.Union[Side, str],
    condition: BoundaryCondition,
    *,
    tolerance: float = 1e-3,
    classified: typing.Optional[typing.Mapping[Side, IntCellArray]] = None,
) -> BoundaryRegion:
    """
    Build a `BoundaryRegion` covering one axis-aligned flank of `grid`.

    ```python
    aquifer_region = make_boundary_region(
        name="south_aquifer", grid=grid, side="south",
        condition=CarterTracyAquifer(initial_pressure=4500.0, aquifer_constant=1.2e6),
    )
    ```

    :param name: `BoundaryRegion.name`.
    :param grid: Grid to classify. Ignored if `classified` is supplied.
    :param side: Which flank - a `Side` member, its `.value` string
        ('west'/'east'/'south'/'north'/'top'/'bottom'), or a common alias
        ('left'/'right'/'front'/'back'/'up'/'down').
    :param condition: `BoundaryCondition` applied at every face on that side.
    :param tolerance: Forwarded to `classify_boundary_faces` when
        `classified` isn't supplied.
    :param classified: Pre-computed `classify_boundary_faces(grid)` result.
        Pass this when building several regions on the same grid to avoid
        re-classifying every boundary face once per region.
    :returns: `BoundaryRegion` covering every face on that side. May have
        zero faces if the grid has no boundary faces on that flank (e.g. a
        `top`/`bottom` request on a grid with only a single active layer at
        that depth already covered elsewhere) - construction still succeeds;
        an empty region simply contributes nothing when evaluated.
    :raises ValidationError: If `side` doesn't resolve to a known `Side`.
    """
    resolved_side = _resolve_side(side)
    faces = (
        classified
        if classified is not None
        else classify_boundary_faces(grid, tolerance=tolerance)
    )
    return BoundaryRegion(
        name=name,
        face_positions=faces[resolved_side],
        condition=condition,
    )


def make_axis_aligned_boundary_conditions(
    grid: Grid,
    sides: typing.Mapping[typing.Union[Side, str], BoundaryCondition],
    *,
    default: typing.Optional[BoundaryCondition] = None,
    tolerance: float = 1e-3,
    unit_system: typing.Optional[UnitSystem] = None,
) -> BoundaryConditions:
    """
    Build a complete `BoundaryConditions` from a `{side: condition}` mapping
    in one call.

    ```python
    boundary_conditions = make_axis_aligned_boundary_conditions(
        grid,
        sides={
            "south": CarterTracyAquifer(initial_pressure=4500.0, aquifer_constant=1.2e6),
            "north": ConstantPressureBoundary(pressure=4200.0),
        },
        default=ConstantFluxBoundary(flux=0.0),  # explicit no-flow elsewhere
    )
    ```

    :param grid: Grid to classify.
    :param sides: Maps each flank you want to assign to its
        `BoundaryCondition`. Keys accept a `Side`, its `.value` string, or a
        common alias, same as `make_boundary_region`.
    :param default: `BoundaryCondition` applied to every `Side` not present
        in `sides`. When `None` (the default), sides not present in `sides`
        get no explicit region at all - they fall through to
        `BoundaryConditions`' own implicit zero-flux default rather than an
        explicit `ConstantFluxBoundary(flux=0.0)` region. The two are
        physically equivalent; passing an explicit `default` is mainly
        useful when you want every side to show up by name in
        `BoundaryConditions.regions` (e.g. for logging/introspection).
    :param tolerance: Forwarded to `classify_boundary_faces`.
    :param unit_system: Forwarded to `BoundaryConditions`. Every region's
        condition is converted to this unit system if it isn't already in
        it; when `None`, every condition passed in `sides`/`default` must
        already share the same unit system.
    :returns: `BoundaryConditions` with one `BoundaryRegion` per side that
        has a condition (from `sides` or `default`) and at least one
        classified face. Sides with zero classified faces are skipped with
        a warning rather than added as an empty region.
    :raises ValidationError: If any key in `sides` doesn't resolve to a
        known `Side`, or the regions end up with mismatched unit systems
        and `unit_system` is `None`.
    """
    classified = classify_boundary_faces(grid, tolerance=tolerance)
    resolved_sides = {_resolve_side(key): condition for key, condition in sides.items()}

    regions: typing.List[BoundaryRegion] = []
    for side in Side:
        condition = resolved_sides.get(side, default)
        if condition is None:
            continue

        positions = classified[side]
        if len(positions) == 0:
            logger.warning(
                "No boundary faces classified for side %r on this grid; "
                "skipping region for it.",
                side.value,
            )
            continue

        regions.append(
            BoundaryRegion(
                name=side.value,
                face_positions=positions,
                condition=condition,
            )
        )
    return BoundaryConditions(regions=regions, unit_system=unit_system)
