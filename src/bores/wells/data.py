"""Static well identity and completion data."""

import enum
import typing

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import (
    FluidPhase,
    Number,
    Orientation,
    UnitConversionTable,
    UnitSystem,
)

__all__ = ["WellType", "CompletionStatus", "Perforation", "WellSpec"]


class WellType(enum.Enum):
    """Producer/injector identity."""

    PRODUCER = "producer"
    INJECTOR = "injector"

    def __str__(self) -> str:
        return self.value


class CompletionStatus(enum.Enum):
    """
    Static intent for a perforation (deck `COMPDAT` item 6 `OPEN`/`SHUT`).

    This is **not** the same as a well being shut in by a control action at
    runtime, that's `WellState.is_open`. `CompletionStatus.SHUT`
    means "this completion was never meant to flow" (e.g. a deck author
    disabling one layer of a multi-layer completion); `WellState.is_open =
    False` means "the whole well is currently shut for operational reasons".

    A perforation with `CompletionStatus.SHUT` is excluded from
    `resolve_perforations()` output entirely; one with `OPEN` is still
    subject to the well-level open/shut flag in `WellState` at simulation
    time.
    """

    OPEN = "open"
    SHUT = "shut"

    def __str__(self) -> str:
        return self.value


@attrs.frozen(kw_only=True, slots=True)
class Perforation(Serializable):
    """A single completion interval, defined purely by depth.

    Two `Perforation` instances with identical fields are interchangeable;
    nothing about a `Perforation` depends on which well it belongs to
    (`WellSpec` is responsible for `wellbore_radius` fallback, not this
    class).
    """

    top_depth: Number
    """
    Positive-down depth, same convention as `Grid.vertex_coordinates`
    z-axis and `Grid.cell_center_depths`.
    """

    bottom_depth: Number
    """`== top_depth` for a point perforation. Validated `>= top_depth`."""

    skin: Number = 0.0
    """
    Dimensionless skin factor. Deck `COMPDAT` has no direct skin item in
    the base keyword, hence this is carried for use by `WPIMULT`/manual skin
    workflows; harmless default.
    """

    wellbore_radius: typing.Optional[Number] = None
    """
    Overrides `WellSpec.wellbore_radius` for this completion only. `None`
    inherits from `WellSpec`.
    """

    status: CompletionStatus = CompletionStatus.OPEN
    """See `CompletionStatus`."""

    connection_factor_override: typing.Optional[Number] = None
    """
    Deck `COMPDAT` item 8 (`CF`). When present, `wells.index` uses this
    directly instead of computing a Peaceman/equivalent-radius well index.
    """

    direction: typing.Optional[Orientation] = None
    """
    `bores.typing.Orientation` (`X`/`Y`/`Z`/`UNSET`). `None` means
    `wells.index` resolves a direction per `D4`'s stated rule; this class
    does not silently assume `Z`.
    """

    partial_penetration_fraction: typing.Optional[Number] = None
    """
    **Not set by the user.** Populated by
    `wells.location.resolve_perforations` (overlap-length / cell-thickness
    ratio). `None` on a freshly constructed `Perforation` is the correct/
    expected state - the field exists on this class rather than a wrapper
    struct so `PerforationIndex` (`wells.index`) can carry it through
    without a second lookup. Validated: if set, must be in `(0, 1]`.
    """

    def __attrs_post_init__(self) -> None:
        if self.bottom_depth < self.top_depth:
            raise ValidationError(
                f"`bottom_depth` ({self.bottom_depth}) must be >= "
                f"`top_depth` ({self.top_depth})."
            )
        if self.wellbore_radius is not None and self.wellbore_radius <= 0:
            raise ValidationError("`wellbore_radius` must be positive.")
        if self.connection_factor_override is not None and (
            self.connection_factor_override <= 0
        ):
            raise ValidationError("`connection_factor_override` must be positive.")
        if self.partial_penetration_fraction is not None and not (
            0 < self.partial_penetration_fraction <= 1
        ):
            raise ValidationError(
                "`partial_penetration_fraction` must be in (0, 1]; got "
                f"{self.partial_penetration_fraction}."
            )

    @property
    def is_point_perforation(self) -> bool:
        """`True` if `top_depth == bottom_depth` (no completion length)."""
        return self.top_depth == self.bottom_depth

    @property
    def length(self) -> Number:
        """`bottom_depth - top_depth`. Zero for a point perforation."""
        return self.bottom_depth - self.top_depth


@attrs.frozen(kw_only=True, slots=True)
class WellSpec(Serializable):
    """
    Static well identity and configuration.

    No status field (`D2`), no methods beyond validation, trivial
    properties, and unit conversion. A `WellSpec` is built once - from a
    deck via `wells.factories`, by hand via `wells.factories`' manual sugar
    constructors, or directly - and never mutated; operational changes are
    expressed as new `WellState` snapshots or `WellSchedule` events, not
    edits to this object.
    """

    name: str
    """Unique identifier, deck `WELSPECS` item 1."""

    well_type: WellType
    """See `D1`."""

    surface_location: typing.Tuple[Number, Number] = attrs.field(converter=tuple)  # type: ignore
    """`(x, y)` in `Grid` coordinate units"""

    reference_depth: Number
    """BHP/THP reporting datum, deck `WELSPECS` item 5."""

    perforations: typing.Tuple[Perforation, ...] = attrs.field(converter=tuple)
    """Non-empty, validated"""

    preferred_phase: typing.Optional[FluidPhase] = None
    """
    Deck `WELSPECS` item 6. `None` allowed for manual construction where
    it's not yet decided.
    """

    group: typing.Optional[str] = None
    """
    Deck `WELSPECS` item 2. `None` = ungrouped (fine - group control is out
    of scope for this rewrite).
    """

    wellbore_radius: Number = 0.25
    """
    Default matches SPE1's `COMPDAT` example (0.5 ft diameter => 0.25 ft
    radius); overridable per `Perforation`.
    """

    tubing_inner_diameter: typing.Optional[Number] = None
    """
    Required by `wells.hydraulics` if flowing pressure-drop is ever
    requested for this well; `None` is valid for a well that will only ever
    be queried for static/no-flow hydrostatic profiles.
    """

    tubing_roughness: typing.Optional[Number] = None
    """
    Absolute roughness, same length unit as `unit_system`. `None` means
    `wells.hydraulics` uses a smooth-pipe assumption (stated explicitly
    there, not silent).
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Matches the `Grid`/`PVTRegion` convention of carrying `unit_system` on
    every physically-dimensioned data object.
    """

    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None
    """Free-form, mirrors `Grid.metadata`."""

    def __attrs_post_init__(self) -> None:
        if not self.name:
            raise ValidationError("`name` must be a non-empty string.")
        if not self.perforations:
            raise ValidationError(
                f"WellSpec {self.name!r} must have at least one perforation."
            )
        if self.wellbore_radius <= 0:
            raise ValidationError("`wellbore_radius` must be positive.")
        if self.tubing_inner_diameter is not None and self.tubing_inner_diameter <= 0:
            raise ValidationError("`tubing_inner_diameter` must be positive.")
        # Perforations carry no `unit_system` of their own (by design, D3) -
        # they are always interpreted in this WellSpec's `unit_system`, so
        # there is nothing left to cross-validate here. Checked, not
        # overlooked.

    @property
    def n_perforations(self) -> int:
        """Total perforation count, including any with `CompletionStatus.SHUT`."""
        return len(self.perforations)

    @property
    def open_perforations(self) -> typing.Tuple[Perforation, ...]:
        """Perforations with `CompletionStatus.OPEN` only."""
        return tuple(
            perforation
            for perforation in self.perforations
            if perforation.status is CompletionStatus.OPEN
        )

    @property
    def min_perforation_depth(self) -> Number:
        """Shallowest `top_depth` across all perforations (open or shut)."""
        return min(perforation.top_depth for perforation in self.perforations)

    @property
    def max_perforation_depth(self) -> Number:
        """Deepest `bottom_depth` across all perforations (open or shut)."""
        return max(perforation.bottom_depth for perforation in self.perforations)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `WellSpec` with all dimensioned fields converted to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional explicit `UnitConversionTable` override,
            same as every other `.convert()` in this codebase.
        :returns: New `WellSpec` in `target` units, or `self` if already there.
        """
        if self.unit_system == target:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        length_factor = factors["length"]

        converted_perforations = tuple(
            attrs.evolve(
                perforation,
                top_depth=perforation.top_depth * length_factor,
                bottom_depth=perforation.bottom_depth * length_factor,
                wellbore_radius=(
                    perforation.wellbore_radius * length_factor
                    if perforation.wellbore_radius is not None
                    else None
                ),
            )
            for perforation in self.perforations
        )
        return attrs.evolve(
            self,
            surface_location=(
                self.surface_location[0] * length_factor,
                self.surface_location[1] * length_factor,
            ),
            reference_depth=self.reference_depth * length_factor,
            perforations=converted_perforations,
            wellbore_radius=self.wellbore_radius * length_factor,
            tubing_inner_diameter=(
                self.tubing_inner_diameter * length_factor
                if self.tubing_inner_diameter is not None
                else None
            ),
            tubing_roughness=(
                self.tubing_roughness * length_factor
                if self.tubing_roughness is not None
                else None
            ),
            unit_system=target,
        )
