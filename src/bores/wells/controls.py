"""Well control targets and secondary limits."""

import enum
import threading
import typing

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.serde.base import (
    Serializable,
)
from bores.serde.registry import (
    make_serializable_type_registrar,
)
from bores.serde.stores.base import StoreSerializable
from bores.typing import FluidPhase, Number, UnitConversionTable, UnitSystem

__all__ = [
    "RateQuantity",
    "ProducerControlMode",
    "InjectorControlMode",
    "Limit",
    "RateLimit",
    "BHPLimit",
    "THPLimit",
    "WellControl",
    "ProducerControl",
    "InjectorControl",
    "WellControls",
    "EconomicQuantity",
    "EconomicLimit",
]


class RateQuantity(enum.Enum):
    """What a `RateLimit`, or a rate-mode control target, is measured in."""

    OIL = "oil"
    WATER = "water"
    GAS = "gas"
    LIQUID = "liquid"
    RESERVOIR = "reservoir"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


class EconomicQuantity(enum.Enum):
    WATER_CUT = "water_cut"
    GOR = "gor"
    WATER_GAS_RATIO = "wgr"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


class ProducerControlMode(enum.Enum):
    """
    Deck `WCONPROD` item 2. Which target field on `ProducerControl`
    is the active one for this well
    """

    ORAT = "orat"
    WRAT = "wrat"
    GRAT = "grat"
    LRAT = "lrat"
    RESV = "resv"
    BHP = "bhp"
    THP = "thp"
    GRUP = "grup"
    UNSET = "none"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


class InjectorControlMode(enum.Enum):
    """
    Deck `WCONINJE` item 3 - which target field on `InjectorControl`
    is the active one for this well.
    """

    RATE = "rate"
    RESV = "resv"
    BHP = "bhp"
    THP = "thp"
    GRUP = "grup"
    UNSET = "none"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


class Limit(Serializable):
    """Abstract base for well-control secondary limits."""

    __abstract_serializable__ = True

    unit_system: UnitSystem

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        raise NotImplementedError


_LIMIT_TYPES: dict[str, type[Limit]] = {}
_limit_type = make_serializable_type_registrar(
    base_cls=Limit,
    registry=_LIMIT_TYPES,
    lock=threading.Lock(),
    key_attr="__type__",
)


@_limit_type
@attrs.frozen(kw_only=True, slots=True)
class RateLimit(Limit):
    """
    A secondary rate constraint (e.g. deck `WECON`/economic limits, or a
    manually-imposed cap orthogonal to the well's primary control mode).

    `quantity` is independent of the well's active `ProducerControlMode`/
    `InjectorControlMode`. A well on `BHP` control can still carry an
    `ORAT` `RateLimit` that forces a switch to rate control if exceeded.
    """

    __type__ = "rate"

    quantity: RateQuantity
    max_value: Number
    unit_system: UnitSystem = UnitSystem.FIELD

    def __attrs_post_init__(self) -> None:
        if self.max_value <= 0:
            raise ValidationError(f"`max_value` must be positive; got {self.max_value}.")

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert the limit to the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `RateLimit` with `max_value` converted to target.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        factor = (
            factors["gas_surface_rate"]
            if self.quantity is RateQuantity.GAS
            else factors["reservoir_rate"]
            if self.quantity is RateQuantity.RESERVOIR
            else factors["liquid_surface_rate"]
        )
        return attrs.evolve(self, max_value=self.max_value * factor, unit_system=target)


@_limit_type
@attrs.frozen(kw_only=True, slots=True)
class BHPLimit(Limit):
    """
    A bottom-hole pressure floor/ceiling.

    At least one of `min_value`/`max_value` must be set. For a producer
    this is typically `min_value` only (don't draw the well below this
    BHP); for an injector, typically `max_value` only (don't exceed
    fracture pressure). Both may be set to bracket a range.
    """

    __type__ = "bhp"

    min_value: Number | None = None
    max_value: Number | None = None
    unit_system: UnitSystem = UnitSystem.FIELD

    def __attrs_post_init__(self) -> None:
        if self.min_value is None and self.max_value is None:
            raise ValidationError(
                f"{self.__class__.__name__} must set at least one of `min_value`/`max_value`."
            )
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValidationError(
                f"`min_value` ({self.min_value}) must be <= `max_value` ({self.max_value})."
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert the limit to the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `BHPLimit` with `min_value`/`max_value` converted to target.
        """
        if target == self.unit_system:
            return self
        factor = get_conversion_factors(self.unit_system, target, table=table)["pressure"]
        return attrs.evolve(
            self,
            min_value=self.min_value * factor if self.min_value is not None else None,
            max_value=self.max_value * factor if self.max_value is not None else None,
            unit_system=target,
        )


@_limit_type
@attrs.frozen(kw_only=True, slots=True)
class THPLimit(Limit):
    """
    A tubing-head (surface) pressure floor/ceiling.

    Kept a distinct class rather than a generic `PressureLimit` so
    `control_engine.py` can pattern-match on type (`isinstance(limit,
    BHPLimit)` vs `isinstance(limit, THPLimit)`) without an extra
    discriminant field on a shared class - mirrors how `RateLimit`/
    `BHPLimit` are already distinguished by type, not a tag. Same
    validation shape as `BHPLimit`; kept separate rather than sharing a
    base because a BHP limit and a THP limit are never interchangeable at
    the call site and accidentally passing one where the other is expected
    should be a type error, not a silent bug.
    """

    __type__ = "thp"

    min_value: Number | None = None
    max_value: Number | None = None
    unit_system: UnitSystem = UnitSystem.FIELD

    def __attrs_post_init__(self) -> None:
        if self.min_value is None and self.max_value is None:
            raise ValidationError("THPLimit must set at least one of `min_value`/`max_value`.")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValidationError(
                f"`min_value` ({self.min_value}) must be <= `max_value` ({self.max_value})."
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert the limit to the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `THPLimit` with min_value/max_value converted to target.
        """
        if target == self.unit_system:
            return self
        factor = get_conversion_factors(self.unit_system, target, table=table)["pressure"]
        return attrs.evolve(
            self,
            min_value=self.min_value * factor if self.min_value is not None else None,
            max_value=self.max_value * factor if self.max_value is not None else None,
            unit_system=target,
        )


@_limit_type
@attrs.frozen(kw_only=True, slots=True)
class EconomicLimit(Limit):
    """
    A fractional-flow economic limit (deck `WECON`).

    Shuts the well in when the given ratio of produced phases exceeds `max_value`.
    """

    __type__ = "economic"

    quantity: EconomicQuantity
    max_value: Number
    unit_system: UnitSystem = UnitSystem.FIELD

    def __attrs_post_init__(self) -> None:
        if self.max_value <= 0:
            raise ValidationError(f"`max_value` must be positive; got {self.max_value}.")

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert the limit to the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New EconomicLimit with max_value converted to target
            (unchanged for WATER_CUT, which is dimensionless).
        """
        if target == self.unit_system or self.quantity is EconomicQuantity.WATER_CUT:
            return attrs.evolve(self, unit_system=target)

        factors = get_conversion_factors(self.unit_system, target, table=table)
        factor = (
            factors["gas_oil_ratio"]
            if self.quantity is EconomicQuantity.GOR
            else factors["oil_gas_ratio"]  # `WATER_GAS_RATIO`: water is dimensionally
        )  # identical to oil (liquid_surface_volume)
        return attrs.evolve(self, max_value=self.max_value * factor, unit_system=target)


class WellControl(Serializable):
    """Abstract base for producer/injector control targets."""

    __abstract_serializable__ = True

    limits: tuple[Limit, ...]
    efficiency_factor: Number = 1.0
    guide_rate: Number | None
    """
    Weight used by group-target allocation (deck WGRUPCON item 3).
    `None` falls back to equal-weight allocation among eligible wells.

    See `wells.resolution.allocation`.
    """
    unit_system: UnitSystem

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        raise NotImplementedError


_CONTROL_TYPES: dict[str, type[WellControl]] = {}
control_type = make_serializable_type_registrar(
    base_cls=WellControl,
    registry=_CONTROL_TYPES,
    lock=threading.Lock(),
    key_attr="__type__",
)


PRODUCER_RATE_MODES = (
    ProducerControlMode.ORAT,
    ProducerControlMode.WRAT,
    ProducerControlMode.GRAT,
    ProducerControlMode.LRAT,
    ProducerControlMode.RESV,
)
INJECTOR_RATE_MODES = (InjectorControlMode.RATE, InjectorControlMode.RESV)


@control_type
@attrs.frozen(kw_only=True, slots=True)
class ProducerControl(WellControl):
    """
    A producer's control target plus its secondary limits.

    Exactly one `target_*` field is "active" determined selected by `mode`.
    Although all three may be populated at once (e.g. an `ORAT` target alongside a
    `BHP` floor as a `BHPLimit`, an `ORAT` *cap* is a `RateLimit`.

    `target_bhp` here is the value used only when `mode is ProducerControlMode.BHP`).
    Deck `WCONPROD` naturally supplies several target items at once regardless of
    which one item 2 selects, so all three are kept rather than only the active one, so
    switching `mode` at runtime (e.g. rate-to-BHP on limit violation) does
    not require re-supplying the other targets.
    """

    __type__ = "producer"

    mode: ProducerControlMode
    target_rate: Number | None = None
    target_bhp: Number | None = None
    target_thp: Number | None = None
    limits: tuple[Limit, ...] = attrs.field(factory=tuple, converter=tuple)
    efficiency_factor: Number = 1.0
    guide_rate: Number | None = None
    """
    Weight used by group-target allocation (deck `WGRUPCON` item 3).
    `None` falls back to equal-weight allocation among eligible wells.
    """
    unit_system: UnitSystem = UnitSystem.FIELD

    def __attrs_post_init__(self) -> None:
        if self.mode in PRODUCER_RATE_MODES and self.target_rate is None:
            raise ValidationError(f"`target_rate` is required when `mode` is {self.mode}.")
        if self.mode is ProducerControlMode.BHP and self.target_bhp is None:
            raise ValidationError(
                "`target_bhp` is required when `mode` is `ProducerControlMode.BHP`."
            )
        if self.mode is ProducerControlMode.THP and self.target_thp is None:
            raise ValidationError(
                "`target_thp` is required when `mode` is `ProducerControlMode.THP`."
            )
        if not (0 < self.efficiency_factor <= 1):
            raise ValidationError(
                f"`efficiency_factor` must be in (0, 1]; got {self.efficiency_factor}."
            )

        mismatched = [limit for limit in self.limits if limit.unit_system != self.unit_system]
        if mismatched:
            raise ValidationError(
                f"All `limits` must share this control's `unit_system` "
                f"({self.unit_system.value}); found {mismatched[0].unit_system.value}."
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert the control to the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `ProducerControl` with every rate/pressure field and
            every entry in limits converted to target.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        rate_modes = {
            ProducerControlMode.ORAT: "liquid_surface_rate",
            ProducerControlMode.WRAT: "liquid_surface_rate",
            ProducerControlMode.GRAT: "gas_surface_rate",
            ProducerControlMode.LRAT: "liquid_surface_rate",
            ProducerControlMode.RESV: "reservoir_rate",
        }
        rate_factor = factors[rate_modes[self.mode]] if self.mode in rate_modes else None
        return attrs.evolve(
            self,
            target_rate=self.target_rate * rate_factor
            if rate_factor and self.target_rate is not None
            else self.target_rate,
            target_bhp=self.target_bhp * factors["pressure"]
            if self.target_bhp is not None
            else None,
            target_thp=self.target_thp * factors["pressure"]
            if self.target_thp is not None
            else None,
            limits=tuple(limit.convert(target, table=table) for limit in self.limits),
            unit_system=target,
        )


@control_type
@attrs.frozen(kw_only=True, slots=True)
class InjectorControl(WellControl):
    """
    An injector's control target plus its secondary limits.

    Same "all targets may be populated, `mode` selects the active one"
    shape as `ProducerControl`, plus `injected_phase` since an
    injector's fluid identity is a control decision (deck `WCONINJE` item
    2), not a static well property.
    """

    __type__ = "injector"

    injected_phase: FluidPhase
    mode: InjectorControlMode
    target_rate: Number | None = None
    target_bhp: Number | None = None
    target_thp: Number | None = None
    limits: tuple[Limit, ...] = attrs.field(factory=tuple, converter=tuple)
    efficiency_factor: Number = 1.0
    guide_rate: Number | None = None
    """
    Weight used by group-target allocation (deck WGRUPCON item 3).
    `None` falls back to equal-weight allocation among eligible wells.

    See `wells.resolution.allocation`.
    """
    unit_system: UnitSystem = UnitSystem.FIELD

    def __attrs_post_init__(self) -> None:
        if self.mode in INJECTOR_RATE_MODES and self.target_rate is None:
            raise ValidationError(f"`target_rate` is required when `mode` is {self.mode}.")
        if self.mode is InjectorControlMode.BHP and self.target_bhp is None:
            raise ValidationError(
                "`target_bhp` is required when `mode` is InjectorControlMode.BHP."
            )
        if self.mode is InjectorControlMode.THP and self.target_thp is None:
            raise ValidationError(
                "`target_thp` is required when `mode` is InjectorControlMode.THP."
            )
        if not (0 < self.efficiency_factor <= 1):
            raise ValidationError(
                f"`efficiency_factor` must be in (0, 1]; got {self.efficiency_factor}."
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert the control to the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `InjectorControl` with every rate/pressure field and
            every entry in limits converted to target.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        rate_factor = (
            factors["gas_surface_rate"]
            if self.injected_phase is FluidPhase.GAS
            else factors["liquid_surface_rate"]
        )
        # Note: `InjectorControlMode.RESV` also exists and
        # we should use factors["reservoir_rate"] instead
        if self.mode is InjectorControlMode.RESV:
            rate_factor = factors["reservoir_rate"]

        return attrs.evolve(
            self,
            target_rate=self.target_rate * rate_factor if self.target_rate is not None else None,
            target_bhp=self.target_bhp * factors["pressure"]
            if self.target_bhp is not None
            else None,
            target_thp=self.target_thp * factors["pressure"]
            if self.target_thp is not None
            else None,
            limits=tuple(limit.convert(target, table=table) for limit in self.limits),
            unit_system=target,
        )


class WellControls(
    StoreSerializable,
    fields={
        "controls": typing.Mapping[str, WellControl],
        "unit_system": typing.Optional[UnitSystem],
    },
):
    """Name-keyed, mutable mapping from well name to its current `WellControl`."""

    __slots__ = ("controls", "unit_system")

    def __init__(
        self,
        controls: typing.Mapping[str, WellControl],
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        :param controls: Mapping from well name to WellControl.
        :param unit_system: Target unit system for every control. None
            requires all controls to already share the same unit system.
        :raises ValidationError: If unit_system is None and the controls
            don't all share one unit system.
        """
        if unit_system is None:
            systems = {control.unit_system for control in controls.values()}
            if len(systems) > 1:
                raise ValidationError(
                    "All controls must share the same unit system when "
                    "`unit_system` is not explicitly provided. Found: "
                    f"{sorted(s.value for s in systems)}."
                )
            unit_system = systems.pop() if systems else UnitSystem.FIELD
        else:
            controls = {
                name: control
                if control.unit_system == unit_system
                else control.convert(unit_system)
                for name, control in controls.items()
            }

        self.controls = dict(controls)
        self.unit_system = unit_system

    def get(self, name: str) -> WellControl | None:
        """Current control for `name`, or `None` if unset."""
        return self.controls.get(name)

    def set(self, name: str, control: WellControl) -> None:
        """
        Replace the current control for `name` wholesale (say for a
        `WCONPROD`/`WCONINJE` reissue, a limit-triggered mode switch, or an
        initial assignment).
        """
        self.controls[name] = control

    def update(self, name: str, **fields: typing.Any) -> None:
        """
        Modify one or more fields of `name`'s current control without
        replacing it wholesale.

        The `WellControl` analogue of deck `WELTARG`/`WELCNTL` (single-target edits),
        once `factories.py` parses those.

        :raises KeyError    groups: typing.Optional[WellGroups] = None
        : If `name` has no current control set.
        """
        current = self.controls.get(name)
        if current is None:
            raise KeyError(f"No control set for well {name!r}.")
        self.controls[name] = attrs.evolve(current, **fields)

    @classmethod
    def from_deck(cls, deck_file: DeckFile) -> Self:
        """
        Load a `WellControls` object froma parsed `DeckFile`.

        :param deck_file: Parsed deck containing WCONPROD/WCONINJE/WECON.
        :returns: WellControls for every well with a control record.
        """
        from bores.wells._deck import load_well_controls_from_deck

        return typing.cast(Self, load_well_controls_from_deck(deck_file))

    def __getitem__(self, name: str) -> WellControl:
        control = self.get(name)
        if control is None:
            raise KeyError(f"No control set for well {name!r}.")
        return control

    def __setitem__(self, name: str, control: WellControl) -> None:
        self.set(name, control)

    def __delitem__(self, name: str) -> None:
        if name not in self.controls:
            raise KeyError(f"No control set for well {name!r}.")
        del self.controls[name]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.controls)

    def __len__(self) -> int:
        return len(self.controls)

    def __contains__(self, name: object) -> bool:
        return name in self.controls

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert every control in this `WellControls` to the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `WellControls` with every control converted to target.
        """
        if target == self.unit_system:
            return self
        return self.__class__(
            controls={
                name: control.convert(target, table=table)
                for name, control in self.controls.items()
            },
            unit_system=target,
        )
