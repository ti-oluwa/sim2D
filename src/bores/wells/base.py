"""Static well identity and completion data."""

import enum
import typing

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.serde.stores import StoreSerializable
from bores.typing import (
    FluidPhase,
    Number,
    Orientation,
    UnitConversionTable,
    UnitSystem,
)
from bores.wells.base import Well, WellType

__all__ = ["WellType", "CompletionStatus", "Perforation", "Well", "Wells"]


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
    `resolve_perforations_indices()` output entirely; one with `OPEN` is still
    subject to the well-level open/shut flag in `WellState` at simulation
    time.
    """

    OPEN = "open"
    SHUT = "shut"

    def __str__(self) -> str:
        return self.value


@attrs.frozen(kw_only=True, slots=True)
class Perforation(Serializable):
    """
    A single completion interval, defined purely by depth.

    Two `Perforation` instances with identical fields are interchangeable;
    nothing about a `Perforation` depends on which well it belongs to
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
    Overrides `Well.wellbore_radius` for this completion only. `None`
    inherits from `Well`.
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
    `wells.index` resolves a direction.
    """

    partial_penetration_fraction: typing.Optional[Number] = None
    """
    **Not set by the user.** Populated by
    `wells.location.resolve_perforations_indices` (overlap-length / cell-thickness
    ratio). `None` on a freshly constructed `Perforation` is the correct/
    expected state. Validated: if set, must be in `(0, 1]`.
    """
    # The field exists on this class rather than a wrapper
    # struct so `PerforationIndex` (`wells.index`) can carry it through
    # without a second lookup

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
class Well(Serializable):
    """Static well identity and configuration."""

    name: str
    """Unique identifier, deck `WELSPECS` item 1."""

    well_type: WellType

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

    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None
    """Free-form, mirrors `Grid.metadata`."""

    def __attrs_post_init__(self) -> None:
        if not self.name:
            raise ValidationError("`name` must be a non-empty string.")
        if not self.perforations:
            raise ValidationError(
                f"Well {self.name!r} must have at least one perforation."
            )
        if self.wellbore_radius <= 0:
            raise ValidationError("`wellbore_radius` must be positive.")
        if self.tubing_inner_diameter is not None and self.tubing_inner_diameter <= 0:
            raise ValidationError("`tubing_inner_diameter` must be positive.")
        # Perforations carry no `unit_system` of their own (by design, D3) -
        # they are always interpreted in this Well's `unit_system`, so
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
        Return a new `Well` with all dimensioned fields converted to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional explicit `UnitConversionTable` override,
            same as every other `.convert()` in this codebase.
        :returns: New `Well` in `target` units, or `self` if already there.
        """
        if self.unit_system == target:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        length_factor = factors["length"]

        perforations = tuple(
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
            perforations=perforations,
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


class Wells(StoreSerializable):
    """Name-keyed container of `Well`. Lookup and iteration only."""

    __abstract_serializable__ = True

    def __init__(self, wells: typing.Dict[str, Well]) -> None:
        """
        :param wells: Mapping from well name to `Well`.
        :raises ValidationError: If `wells` is empty, or if any key doesn't
            match its value's `Well.name`.
        """
        if not wells:
            raise ValidationError("`wells` must contain at least one entry.")

        mismatched = {key: well.name for key, well in wells.items() if key != well.name}
        if mismatched:
            raise ValidationError(
                f"`wells` dict keys must match `Well.name`; mismatches "
                f"(key -> well.name): {mismatched}."
            )
        self._wells = wells

    def well(self, name: str) -> Well:
        """
        :param name: `Well` name.
        :returns: `Well` for that well.
        :raises KeyError: If no well with that name exists.
        """
        well = self._wells.get(name)
        if well is None:
            raise KeyError(f"No well named {name!r}. Available: {sorted(self._wells)}.")
        return well

    @property
    def names(self) -> typing.Tuple[str, ...]:
        """All well names, insertion order."""
        return tuple(self._wells.keys())

    @property
    def producers(self) -> typing.Tuple[Well, ...]:
        """All wells with `well_type is WellType.PRODUCER`."""
        return tuple(
            well for well in self._wells.values() if well.well_type is WellType.PRODUCER
        )

    @property
    def injectors(self) -> typing.Tuple[Well, ...]:
        """All wells with `well_type is WellType.INJECTOR`."""
        return tuple(
            well for well in self._wells.values() if well.well_type is WellType.INJECTOR
        )

    def __getitem__(self, name: str) -> Well:
        return self.well(name)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._wells)

    def __len__(self) -> int:
        return len(self._wells)

    def __contains__(self, name: object) -> bool:
        return name in self._wells

    def __dump__(self) -> typing.Dict[str, typing.Any]:
        return {"wells": {name: well.dump() for name, well in self._wells.items()}}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> "Wells":
        wells = {
            name: Well.load(spec_data) for name, spec_data in data["wells"].items()
        }
        return cls(wells=wells)
