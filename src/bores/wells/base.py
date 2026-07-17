"""Static well identity and completion data."""

import enum
import typing

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.base import Serializable
from bores.serde.stores import StoreSerializable
from bores.typing import (
    FluidPhase,
    Number,
    Orientation,
    UnitConversionTable,
    UnitSystem,
)
from bores.wells.trajectory import WellTrajectory

__all__ = [
    "WellType",
    "CompletionStatus",
    "Perforation",
    "MDPerforation",
    "Well",
    "Wells",
]


class WellType(enum.Enum):
    """Producer/injector identity."""

    PRODUCER = "producer"
    INJECTOR = "injector"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


class CompletionStatus(enum.Enum):
    """
    Static intent for a perforation (deck `COMPDAT` item 6 `OPEN`/`SHUT`).

    This is not the same as a well being shut in by a control action at
    runtime, that's `WellState.is_open`. `CompletionStatus.SHUT`
    means **"this completion was never meant to flow"** (e.g. a deck author
    disabling one layer of a multi-layer completion); `WellState.is_open =
    False` means **"the whole well is currently shut for operational reasons"**.

    A perforation with `CompletionStatus.SHUT` is excluded from
    the perforation indices resolution output entirely; one with `OPEN` is still
    subject to the well-level open/shut flag in `WellState` at simulation time.
    """

    OPEN = "open"
    SHUT = "shut"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


@attrs.frozen(kw_only=True, slots=True)
class Perforation(Serializable):
    """
    A single completion interval on a vertical well, defined by true
    vertical depth.

    Two `Perforation` instances with identical fields are interchangeable;
    nothing about a `Perforation` depends on which well it belongs to. Only
    valid on a `Well` with no `trajectory`. 
    
    See `MDPerforation` for a well with one.
    """

    top_depth: Number
    """
    Positive-down depth, same convention as `Grid.vertex_coordinates`
    z-axis and `Grid.cell_center_depths`.
    """

    bottom_depth: Number
    """Equals `top_depth` for a point perforation. Validated `>= top_depth`."""

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
    Deck `COMPDAT` item 8 (`CF`). When present, wells indices computation uses this
    directly instead of computing a Peaceman/equivalent-radius well index.
    """

    connection_factor_multiplier: typing.Optional[Number] = None
    """
    Deck `WPIMULT`. Scales the computed well index rather than replacing
    it. Applied after `connection_factor_override`, if that's also set,
    though the two would not normally both be present on one perforation.
    """

    direction: typing.Optional[Orientation] = None
    """
    `bores.typing.Orientation` (`X`/`Y`/`Z`/`UNSET`). `None` means
    `wells.indices` resolves a direction.
    """

    partial_penetration_fraction: typing.Optional[Number] = None
    """
    **Not set by the user.** 

    Populated during perforation indices computation (overlap-length /
    cell-thickness ratio). `None` on a freshly constructed `Perforation` is
    the correct/expected state. Validated: if set, must be in `(0, 1]`.
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
class MDPerforation(Serializable):
    """
    A single completion interval on a well with a `WellTrajectory`, defined
    by measured depth rather than true vertical depth.

    True vertical depth is not invertible along a horizontal or S-shaped
    section, multiple measured depths can share the same TVD, so
    measured depth is the only interval representation that identifies a
    unique location on an arbitrary path.

    Carries no `direction` field, unlike `Perforation`, orientation
    comes from the trajectory's local tangent at this interval, not a discrete axis 
    choice, and wells indices computations never runs Peaceman's formula against an
    `MDPerforation` connection for the same reason. As Peaceman's formula
    assumes a wellbore aligned with a principal permeability axis, which an
    arbitrary trajectory azimuth generally isn't.

    Therefore, connections at an `MDPerforation` always resolve through the isotropic
    equivalent-radius well index instead.
    """

    top_md: Number
    """Measured depth. Must fall within the owning `Well`'s trajectory range."""

    bottom_md: Number
    """Equals `top_md` for a point perforation. Validated `>= top_md`."""

    skin: Number = 0.0
    """Dimensionless skin factor."""

    wellbore_radius: typing.Optional[Number] = None
    """
    Overrides `Well.wellbore_radius` for this completion only. `None` inherits from `Well`."""

    status: CompletionStatus = CompletionStatus.OPEN
    """See `CompletionStatus`."""

    connection_factor_override: typing.Optional[Number] = None
    """
    When present, `wells.indices` uses this directly instead of
    computing an equivalent-radius well index.
    """

    connection_factor_multiplier: typing.Optional[Number] = None
    """
    Deck `WPIMULT`. Scales the computed well index rather than replacing
    it. Applied after `connection_factor_override`, if that's also set,
    though the two would not normally both be present on one perforation.
    """

    partial_penetration_fraction: typing.Optional[Number] = None
    """
    **Not set by the user.** Populated by `wells.perforations.resolve_perforations_indices`. 
    `None` on a freshly constructed `MDPerforation` is the correct/expected state. 

    Validated: if set, must be in `(0, 1]`.
    """

    def __attrs_post_init__(self) -> None:
        if self.bottom_md < self.top_md:
            raise ValidationError(
                f"`bottom_md` ({self.bottom_md}) must be >= `top_md` ({self.top_md})."
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
        """`True` if `top_md == bottom_md` (no completion length)."""
        return self.top_md == self.bottom_md

    @property
    def length(self) -> Number:
        """
        `bottom_md - top_md`.
        
        Measured-depth length, not a true vertical depth length; 
        along a horizontal section these differ substantially.
        """
        return self.bottom_md - self.top_md


AnyPerforation = typing.Union[Perforation, MDPerforation]


@attrs.frozen(kw_only=True, slots=True)
class Well(Serializable):
    """Static well identity and configuration."""

    name: str
    """Unique identifier, deck `WELSPECS` item 1."""

    well_type: WellType

    surface_location: typing.Tuple[Number, Number] = attrs.field(converter=tuple)  # type: ignore
    """
    `(x, y)` in `Grid` coordinate units - the wellhead location,
    regardless of whether the well is vertical or has a `trajectory`.
    """

    reference_depth: Number
    """BHP/THP reporting datum, deck `WELSPECS` item 5."""

    perforations: typing.Tuple[AnyPerforation, ...] = attrs.field(converter=tuple)
    """
    `Perforation` (TVD) if `trajectory` is `None`;
    `MDPerforation` (measured depth) if it's set. Do not mix.

    Must not be empty. 
    """

    trajectory: typing.Optional[WellTrajectory] = None
    """
    Deviation survey. 

    **`None`** (default): a vertical well at `surface_location`, and 
    `perforations` must be `Perforation`. 
    
    **Set**: a deviated/horizontal well, and `perforations` must be `MDPerforation`.
    """

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
    radius); overridable per perforation.
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
    `wells.hydraulics` uses a smooth-pipe assumption.
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

        if self.trajectory is not None:
            if not all(isinstance(p, MDPerforation) for p in self.perforations):
                raise ValidationError(
                    f"Well {self.name!r} has a `trajectory`; every entry in "
                    "`perforations` must be an `MDPerforation`, not `Perforation`."
                )
            for perforation in self.perforations:
                assert isinstance(perforation, MDPerforation)
                if not (
                    self.trajectory.top_measured_depth
                    <= perforation.top_md
                    <= perforation.bottom_md
                    <= self.trajectory.bottom_measured_depth
                ):
                    raise ValidationError(
                        f"Perforation measured-depth range "
                        f"[{perforation.top_md}, {perforation.bottom_md}] falls "
                        f"outside well {self.name!r}'s trajectory range "
                        f"[{self.trajectory.top_measured_depth}, "
                        f"{self.trajectory.bottom_measured_depth}]."
                    )
        else:
            if not all(isinstance(p, Perforation) for p in self.perforations):
                raise ValidationError(
                    f"Well {self.name!r} has no `trajectory`; every entry in "
                    "`perforations` must be a `Perforation`, not `MDPerforation`. "
                    "Set `trajectory` to use `MDPerforation`."
                )

    @property
    def n_perforations(self) -> int:
        """Total perforation count, including any with `CompletionStatus.SHUT`."""
        return len(self.perforations)

    @property
    def open_perforations(self) -> typing.Tuple[AnyPerforation, ...]:
        """Perforations with `CompletionStatus.OPEN` only."""
        return tuple(
            perforation
            for perforation in self.perforations
            if perforation.status is CompletionStatus.OPEN
        )

    @property
    def min_perforation_depth(self) -> Number:
        """Shallowest true vertical depth across all perforations (open or shut)."""
        if self.trajectory is not None:
            return min(
                self.trajectory.position_at(perforation.top_md)[2]  # type: ignore[union-attr]
                for perforation in self.perforations
            )
        return min(perforation.top_depth for perforation in self.perforations)  # type: ignore[union-attr]

    @property
    def max_perforation_depth(self) -> Number:
        """Deepest true vertical depth across all perforations (open or shut)."""
        if self.trajectory is not None:
            return max(
                self.trajectory.position_at(perforation.bottom_md)[2]  # type: ignore[union-attr]
                for perforation in self.perforations
            )
        return max(perforation.bottom_depth for perforation in self.perforations)  # type: ignore[union-attr]

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

        trajectory: typing.Optional[WellTrajectory]
        perforations: typing.Tuple[AnyPerforation, ...]

        if self.trajectory is not None:
            trajectory = WellTrajectory(
                stations=tuple(
                    attrs.evolve(
                        station,
                        x=station.x * length_factor,
                        y=station.y * length_factor,
                        z=station.z * length_factor,
                        measured_depth=station.measured_depth * length_factor,
                    )
                    for station in self.trajectory.stations
                )
            )
            perforations = tuple(
                attrs.evolve(
                    perforation,
                    top_md=perforation.top_md * length_factor,  # type: ignore[union-attr]
                    bottom_md=perforation.bottom_md * length_factor,  # type: ignore[union-attr]
                    wellbore_radius=(
                        perforation.wellbore_radius * length_factor
                        if perforation.wellbore_radius is not None
                        else None
                    ),
                )
                for perforation in self.perforations
            )
        else:
            trajectory = None
            perforations = tuple(
                attrs.evolve(
                    perforation,
                    top_depth=perforation.top_depth * length_factor,  # type: ignore[union-attr]
                    bottom_depth=perforation.bottom_depth * length_factor,  # type: ignore[union-attr]
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
            trajectory=trajectory,
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
    """Name-keyed container of `Well` objects"""

    __abstract_serializable__ = True
    __slots__ = ("_wells", "unit_system")

    def __init__(
        self,
        wells: typing.Dict[str, Well],
        unit_system: typing.Optional[UnitSystem] = None,
    ) -> None:
        """
        :param wells: Mapping from well name to Well.
        :param unit_system: Target unit system for every well. None
            requires all wells to already share the same unit system.
        :raises ValidationError: If wells is empty, any key doesn't match
            its value's Well.name, or (unit_system is None) the wells
            don't all share one unit system.
        """
        if not wells:
            raise ValidationError("`wells` must contain at least one entry.")

        mismatched = {key: well.name for key, well in wells.items() if key != well.name}
        if mismatched:
            raise ValidationError(
                f"`wells` dict keys must match Well.name; mismatches "
                f"(key -> well.name): {mismatched}."
            )

        if unit_system is None:
            systems = {well.unit_system for well in wells.values()}
            if len(systems) > 1:
                raise ValidationError(
                    "All wells must share the same unit system when "
                    "`unit_system` is not explicitly provided. Found: "
                    f"{sorted(s.value for s in systems)}."
                )
            unit_system = systems.pop()
        else:
            wells = {
                name: well
                if well.unit_system == unit_system
                else well.convert(unit_system)
                for name, well in wells.items()
            }

        self._wells = wells
        self.unit_system = unit_system

    def well(self, name: str) -> Well:
        """
        Retrieve a registered `Well` by name.

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
        """All well names, in insertion order."""
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

    @classmethod
    def from_deck(
        cls, deck_file: DeckFile, *, grid: Grid, **well_kwargs: typing.Any
    ) -> Self:
        """
        Load the `Wells` object from a parsed `DeckFile`.

        :param deck_file: Parsed deck containing WELSPECS/COMPDAT/WCONINJE.
        :param grid: Grid built from the same deck.
        :param well_kwargs: Forwarded to `wells_from_records` and passed
            to the loaded `Well` instance.
        :returns: `Wells` for every well in the deck.
        """
        from bores.wells._deck import load_wells_from_deck

        return typing.cast(Self, load_wells_from_deck(deck_file, grid, **well_kwargs))

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

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Returns a new `Wells` object in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New Wells with every well converted to target.
        """
        if target == self.unit_system:
            return self
        return self.__class__(
            wells={
                name: well.convert(target, table=table)
                for name, well in self._wells.items()
            },
            unit_system=target,
        )
