"""Well index (connection factor) computation.

Depends on `wells.data`, `wells.location` (`PerforationIndex` without
`well_index`), and `bores.grids.base.Grid`.

`compute_well_index`, `compute_3D_effective_drainage_radius`, and
`compute_2D_effective_drainage_radius` are ported verbatim from the current
`wells/core.py` - same formulas, same numba bodies, unchanged. Note this is
a 3-step pipeline in the real implementation (effective permeability ->
effective drainage radius -> well index formula), not the single flattened
`compute_well_index(permeability_i, permeability_j, ...)` signature
sketched in `02_BEHAVIOR_LAYER.md` - that sketch doesn't match what's
actually in `core.py`, so the real functions are ported here instead of
the fictional one.
"""

import typing

import attrs
import numba
import numpy as np

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.base import Serializable
from bores.typing import Number, NumberArray, OneDimension, Orientation
from bores.wells.collection import Wells
from bores.wells.data import Perforation
from bores.wells.location import PerforationIndex, resolve_perforations

__all__ = [
    "WellIndex",
    "resolve_well_index_direction",
    "compute_well_index",
    "compute_3D_effective_drainage_radius",
    "compute_2D_effective_drainage_radius",
    "compute_effective_permeability_for_well",
    "compute_equivalent_radius_well_index",
    "build_wells_indices",
]


@attrs.frozen(kw_only=True, slots=True)
class WellIndex(Serializable):
    """Computed well indices for all perforations of a single well."""

    well_name: str
    perforations: typing.Tuple[PerforationIndex, ...] = attrs.field(converter=tuple)
    total_well_index: Number

    def get_allocation_fraction(self, perforation: PerforationIndex) -> Number:
        """Fraction of `total_well_index` contributed by one `perforation`."""
        if self.total_well_index <= 0:
            return 1.0
        if perforation.well_index is None:
            raise ValidationError()
        return perforation.well_index / self.total_well_index

    def __iter__(self) -> typing.Iterator[PerforationIndex]:
        return iter(self.perforations)


def resolve_well_index_direction(
    perforation: Perforation, grid: Grid, cell_index: int
) -> Orientation:
    """
    Resolve which axis a Peaceman-style well index should treat as the
    "along-wellbore" direction for `perforation` in `cell_index`.

    Resolution order: `perforation.direction` if set (and not `UNSET`);
    otherwise `Orientation.Z` (vertical is the overwhelmingly common case).

    :param perforation: The perforation being resolved.
    :param grid: Grid the cell belongs to (unused in this v1 body - kept as
        a parameter so a future geometry-based strategy can be added
        without a signature change).
    :param cell_index: Cell the perforation resolved into.
    :returns: `Orientation.X`, `Orientation.Y`, or `Orientation.Z`.
    """
    if (
        perforation.direction is not None
        and perforation.direction is not Orientation.UNSET
    ):
        return perforation.direction
    return Orientation.Z


def _is_locally_cartesian(
    grid: Grid, cell_index: int, tolerance: Number = 0.05
) -> bool:
    """
    Decide whether `cell_index` is "Cartesian-like enough" for Peaceman's
    formula to be valid, vs. requiring the equivalent-radius fallback.

    Compares the cell's true volume against the volume of its own AABB; if
    the ratio is within `tolerance` of 1.0, the cell is box-like enough for
    Peaceman.

    :param grid: Grid providing `cell_volumes`, `cell_length_x/y/z`.
    :param cell_index: Cell to classify.
    :param tolerance: Allowed fractional deviation from a perfect box.
    :returns: `True` if Peaceman should be used, `False` for the fallback.
    """
    aabb_volume = (
        grid.cell_length_x[cell_index]
        * grid.cell_length_y[cell_index]
        * grid.cell_length_z[cell_index]
    )
    if aabb_volume <= 0.0:
        return False

    assert grid.cell_volumes is not None
    ratio = grid.cell_volumes[cell_index] / aabb_volume
    return abs(ratio - 1.0) <= tolerance


@numba.njit(cache=True)
def compute_well_index(
    permeability: Number,
    interval_thickness: Number,
    wellbore_radius: Number,
    effective_drainage_radius: Number,
    skin_factor: Number = 0.0,
    net_to_gross: Number = 1.0,
    regime_constant: Number = -3 / 4,
) -> Number:
    """
    Compute the well index for a given well using the Peaceman equation.

    W = (k * h * N/G) / (ln(re/rw) + C + s)

    :param permeability: Effective permeability at the cell (mD).
    :param interval_thickness: Completion length within the cell (ft).
    :param wellbore_radius: Radius of the wellbore (ft).
    :param effective_drainage_radius: Effective drainage radius (ft).
    :param skin_factor: Skin factor for the well (dimensionless).
    :param net_to_gross: Net-to-gross ratio of the reservoir interval.
    :param regime_constant: 0 for steady, -3/4 for pseudo steady, 1/2 for transient.
    :return: The well index (mD*ft).
    """
    return (permeability * interval_thickness * net_to_gross) / (
        np.log(effective_drainage_radius / wellbore_radius)
        + regime_constant
        + skin_factor
    )


@numba.njit(cache=True)
def compute_3D_effective_drainage_radius(
    interval_thickness: typing.Tuple[Number, Number, Number],
    permeability: typing.Tuple[Number, Number, Number],
    well_orientation: Orientation,
) -> Number:
    """
    Compute the effective drainage radius for a well in a 3D reservoir
    model using Peaceman's (1983) anisotropic effective drainage radius
    formula.

    Ported unchanged from `wells/core.py::compute_3D_effective_drainage_radius`.

    :param interval_thickness: Tuple of cell dimensions (dx, dy, dz) in ft.
    :param permeability: Tuple of permeabilities (kx, ky, kz) in mD.
    :param well_orientation: Wellbore axis.
    :return: Effective drainage (Peaceman) radius in ft, or 0.0 if either
        perpendicular permeability is zero.
    """
    dx, dy, dz = interval_thickness
    kx, ky, kz = permeability

    if well_orientation == Orientation.X:
        if ky <= 0.0 or kz <= 0.0:
            return 0.0
        r1, r2 = ky / kz, kz / ky
        numerator = np.sqrt(r1) * dy**2 + np.sqrt(r2) * dz**2
        denominator = r1**0.25 + r2**0.25
    elif well_orientation == Orientation.Y:
        if kx <= 0.0 or kz <= 0.0:
            return 0.0
        r1, r2 = kx / kz, kz / kx
        numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dz**2
        denominator = r1**0.25 + r2**0.25
    elif well_orientation == Orientation.Z:
        if kx <= 0.0 or ky <= 0.0:
            return 0.0
        r1, r2 = ky / kx, kx / ky
        numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dy**2
        denominator = r1**0.25 + r2**0.25
    else:
        raise ValidationError("Invalid well orientation")

    return 0.28 * np.sqrt(numerator / denominator)


@numba.njit(cache=True)
def compute_2D_effective_drainage_radius(
    interval_thickness: typing.Tuple[Number, Number],
    permeability: typing.Tuple[Number, Number],
) -> Number:
    """
    Compute the effective drainage radius for a well in a 2D reservoir
    model using Peaceman's (1983) anisotropic formula.

    Ported unchanged from `wells/core.py::compute_2D_effective_drainage_radius`.

    :param interval_thickness: Tuple of cell dimensions (dx, dy) in ft.
    :param permeability: Tuple of permeabilities (kx, ky) in mD.
    :return: Effective drainage (Peaceman) radius in ft, or 0.0 if either
        permeability is zero or negative.
    """
    kx, ky = permeability
    if kx <= 0.0 or ky <= 0.0:
        return 0.0

    dx, dy = interval_thickness[0], interval_thickness[1]
    r1, r2 = ky / kx, kx / ky
    numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dy**2
    denominator = r1**0.25 + r2**0.25
    return 0.28 * np.sqrt(numerator / denominator)


@numba.njit(cache=True, inline="always")
def _geometric_mean(values: typing.Sequence[Number]) -> Number:
    """Ported unchanged from `wells/core.py::_geometric_mean`."""
    prod = 1.0
    n = 0
    for v in values:
        prod *= max(v, 0.0)
        n += 1
    if n == 0:
        raise ValidationError("No permeability values provided")
    return prod ** (1.0 / n)


@numba.njit(cache=True)
def compute_effective_permeability_for_well(
    permeability: typing.Tuple[Number, Number, Number], orientation: Orientation
) -> Number:
    """
    Compute `k_eff` for Peaceman WI using geometric mean of the two
    permeabilities perpendicular to the well axis.

    :param permeability: `(kx, ky, kz)`.
    :param orientation: `Orientation.X`/`Y`/`Z`.
    :return: Effective (geometric-mean) permeability.
    """
    if len(permeability) != 3:
        return _geometric_mean(permeability)

    kx, ky, kz = permeability
    if orientation == Orientation.Z:
        return np.sqrt(max(kx, 0.0) * max(ky, 0.0))
    elif orientation == Orientation.X:
        return np.sqrt(max(ky, 0.0) * max(kz, 0.0))
    elif orientation == Orientation.Y:
        return np.sqrt(max(kx, 0.0) * max(kz, 0.0))
    return _geometric_mean((kx, ky, kz))


@numba.njit(cache=True)
def compute_equivalent_radius_well_index(
    permeability: Number,
    cell_volume: Number,
    completion_length: Number,
    wellbore_radius: Number,
    skin: Number,
) -> Number:
    """
    Well index for a cell where `_is_locally_cartesian` returned `False`.

    Isotropic equivalent-radius formulation: `r_0 = 0.14 * sqrt(cell_volume
    / completion_length)`, the volume-per-unit-length analogue of
    Peaceman's `r_0 = 0.14 * sqrt(dx**2 + dy**2)`, valid without assuming a
    clean local (dx, dy) pair.

    :param permeability: Isotropic (or geometric-mean) permeability (mD).
    :param cell_volume: `grid.cell_volumes[cell_index]`.
    :param completion_length: Perforation length within this cell (ft).
    :param wellbore_radius: Perforation or WellSpec wellbore radius (ft).
    :param skin: Perforation skin factor.
    :returns: Well index (mD*ft).
    """
    r0 = 0.14 * np.sqrt(cell_volume / completion_length)
    return (
        2.0
        * np.pi
        * permeability
        * completion_length
        / (np.log(r0 / wellbore_radius) + skin)
    )


def _resolve_connection_factor(
    perforation: Perforation,
    grid: Grid,
    cell_index: int,
    partial_penetration_fraction: Number,
    wellbore_radius: Number,
    permeabilities: typing.Mapping[Orientation, Number],
) -> Number:
    """
    Single entry point `build_wells_indices` calls per (perforation, cell)
    pair.

    Order: `perforation.connection_factor_override` if set > Peaceman
    (`compute_well_index`) if `_is_locally_cartesian` > equivalent-radius
    fallback (`compute_equivalent_radius_well_index`) otherwise.

    Deviates from `02_BEHAVIOR_LAYER.md`'s literal signature by adding
    `partial_penetration_fraction` and `wellbore_radius` as explicit
    parameters - required to compute `completion_length` and resolve the
    `Perforation.wellbore_radius` (may be `None`, falls back to
    `WellSpec.wellbore_radius`) fallback, neither of which `perforation`
    alone can supply.

    :param perforation: Source perforation (for skin, override).
    :param grid: Grid providing geometry.
    :param cell_index: Resolved cell.
    :param partial_penetration_fraction: From the matching `PerforationIndex`.
    :param wellbore_radius: Already-resolved radius (perforation override or
        well default).
    :param permeabilities: Per-axis permeability at `cell_index`, keyed by
        `Orientation.X`/`Y`/`Z`.
    :returns: Connection factor / well index for this (perforation, cell) pair.
    """
    if perforation.connection_factor_override is not None:
        return perforation.connection_factor_override

    direction = resolve_well_index_direction(perforation, grid, cell_index)
    kx = permeabilities[Orientation.X]
    ky = permeabilities[Orientation.Y]
    kz = permeabilities[Orientation.Z]

    cell_length_by_axis = {
        Orientation.X: grid.cell_length_x[cell_index],
        Orientation.Y: grid.cell_length_y[cell_index],
        Orientation.Z: grid.cell_length_z[cell_index],
    }
    completion_length = partial_penetration_fraction * cell_length_by_axis[direction]
    effective_permeability = compute_effective_permeability_for_well(
        permeability=(kx, ky, kz), orientation=direction
    )

    if _is_locally_cartesian(grid, cell_index):
        effective_drainage_radius = compute_3D_effective_drainage_radius(
            interval_thickness=(
                grid.cell_length_x[cell_index],
                grid.cell_length_y[cell_index],
                grid.cell_length_z[cell_index],
            ),
            permeability=(kx, ky, kz),
            well_orientation=direction,
        )
        return compute_well_index(
            permeability=effective_permeability,
            interval_thickness=completion_length,
            wellbore_radius=wellbore_radius,
            effective_drainage_radius=effective_drainage_radius,
            skin_factor=perforation.skin,
            net_to_gross=1.0,
            regime_constant=-3 / 4,
        )

    assert grid.cell_volumes is not None
    return compute_equivalent_radius_well_index(
        permeability=effective_permeability,
        cell_volume=grid.cell_volumes[cell_index],
        completion_length=completion_length,
        wellbore_radius=wellbore_radius,
        skin=perforation.skin,
    )


def build_wells_indices(
    grid: Grid,
    wells: Wells,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    *,
    resolve_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> typing.Dict[str, WellIndex]:
    """
    Resolve every well in `wells` against `grid` and compute connection
    factors for every open perforation.

    :param grid: Grid to resolve against.
    :param wells: `Wells` container (all wells to resolve).
    :param permeabilities: Per-axis permeability arrays, shape `(n_cells,)`
        each, keyed by `Orientation.X`/`Y`/`Z`.
    :param resolve_kwargs: Passed through to `resolve_perforations`.
    :returns: Mapping from well name to `WellIndex`.
    :raises ValidationError: Propagated from `resolve_perforations` for any
        well with a dangling completion.
    """
    resolve_kwargs = resolve_kwargs or {}
    result: typing.Dict[str, WellIndex] = {}

    for name in wells:
        spec = wells[name]
        raw_indices = resolve_perforations(grid, spec, **resolve_kwargs)

        resolved: typing.List[PerforationIndex] = []
        total_well_index = 0.0
        for pidx in raw_indices:
            wellbore_radius = (
                pidx.perforation.wellbore_radius
                if pidx.perforation.wellbore_radius is not None
                else spec.wellbore_radius
            )
            per_cell_permeabilities = {
                axis: array[pidx.cell_index] for axis, array in permeabilities.items()
            }
            well_index = _resolve_connection_factor(
                perforation=pidx.perforation,
                grid=grid,
                cell_index=pidx.cell_index,
                partial_penetration_fraction=pidx.partial_penetration_fraction,
                wellbore_radius=wellbore_radius,
                permeabilities=per_cell_permeabilities,
            )
            resolved.append(attrs.evolve(pidx, well_index=well_index))
            total_well_index += well_index

        result[spec.name] = WellIndex(
            well_name=spec.name,
            perforations=tuple(resolved),
            total_well_index=total_well_index,
        )
    return result
