"""Well control targets and secondary limits."""

import enum
import threading
import typing

import attrs
from typing_extensions import Self

from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.serde.registry import make_serializable_type_registrar
from bores.serde.stores.base import StoreSerializable
from bores.typing import FluidPhase, Number

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

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


class Limit(Serializable):
    """
    Abstract base for well-control secondary limits.

    Carries no fields of its own - see the module docstring for why this
    is a real base class (registered against a type registry) rather than
    the `typing.Union[RateLimit, BHPLimit, THPLimit]` alias the original
    sketch used.
    """

    __abstract_serializable__ = True


_LIMIT_TYPES: typing.Dict[str, typing.Type[Limit]] = {}
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
    `ORAT` `RateLimit` that forces a switch to rate control if exceeded;
    resolving that switch is `control_engine.py`'s job, not this class's.
    """

    __type__ = "rate"

    quantity: RateQuantity
    max_value: Number

    def __attrs_post_init__(self) -> None:
        if self.max_value <= 0:
            raise ValidationError(
                f"`max_value` must be positive; got {self.max_value}."
            )


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

    min_value: typing.Optional[Number] = None
    max_value: typing.Optional[Number] = None

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
                f"`min_value` ({self.min_value}) must be <= `max_value` "
                f"({self.max_value})."
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

    min_value: typing.Optional[Number] = None
    max_value: typing.Optional[Number] = None

    def __attrs_post_init__(self) -> None:
        if self.min_value is None and self.max_value is None:
            raise ValidationError(
                "THPLimit must set at least one of `min_value`/`max_value`."
            )
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValidationError(
                f"`min_value` ({self.min_value}) must be <= `max_value` "
                f"({self.max_value})."
            )


@_limit_type
@attrs.frozen(kw_only=True, slots=True)
class EconomicLimit(Limit):
    """
    A fractional-flow economic limit (deck WECON).

    Shuts the well in when the given ratio of produced phases exceeds `max_value`.
    """

    __type__ = "economic"

    quantity: EconomicQuantity
    max_value: Number

    def __attrs_post_init__(self) -> None:
        if self.max_value <= 0:
            raise ValidationError(
                f"`max_value` must be positive; got {self.max_value}."
            )


class WellControl(Serializable):
    """
    Abstract base for producer/injector control targets.

    Carries no fields of its own - see the module docstring for why this
    is a real base class rather than the `typing.Union[ProducerControl,
    InjectorControl]` alias the original sketch used.
    """

    __abstract_serializable__ = True

    limits: typing.Tuple[Limit, ...] = attrs.field(factory=tuple, converter=tuple)
    efficiency_factor: Number = 1.0
    guide_rate: typing.Optional[Number] = None
    """
    Weight used by group-target allocation (deck WGRUPCON item 3).
    `None` falls back to equal-weight allocation among eligible wells.

    See `wells.resolution.allocation`.
    """


_CONTROL_TYPES: typing.Dict[str, typing.Type[WellControl]] = {}
_control_type = make_serializable_type_registrar(
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


@_control_type
@attrs.frozen(kw_only=True, slots=True)
class ProducerControl(WellControl):
    """
    A producer's control target plus its secondary limits.

    Exactly one `target_*` field is "active" determined selected by `mode`.
    Although all three may be populated at once (e.g. an `ORAT` target alongside a
    `BHP` floor as a `BHPLimit`, an `ORAT` *cap* is a `RateLimit`.

    `target_bhp` here is the value used only
    when `mode is ProducerControlMode.BHP`). Deck `WCONPROD` naturally
    supplies several target items at once regardless of which one item 2
    selects, so all three are kept rather than only the active one, so
    switching `mode` at runtime (e.g. rate-to-BHP on limit violation) does
    not require re-supplying the other targets.
    """

    __type__ = "producer"

    mode: ProducerControlMode
    target_rate: typing.Optional[Number] = None
    target_bhp: typing.Optional[Number] = None
    target_thp: typing.Optional[Number] = None
    limits: typing.Tuple[Limit, ...] = attrs.field(factory=tuple, converter=tuple)
    efficiency_factor: Number = 1.0
    guide_rate: typing.Optional[Number] = None
    """
    Weight used by group-target allocation (deck WGRUPCON item 3).
    `None` falls back to equal-weight allocation among eligible wells.

    See `wells.resolution.allocation`.
    """

    def __attrs_post_init__(self) -> None:
        if self.mode in PRODUCER_RATE_MODES and self.target_rate is None:
            raise ValidationError(
                f"`target_rate` is required when `mode` is {self.mode}."
            )
        if self.mode is ProducerControlMode.BHP and self.target_bhp is None:
            raise ValidationError(
                "`target_bhp` is required when `mode` is ProducerControlMode.BHP."
            )
        if self.mode is ProducerControlMode.THP and self.target_thp is None:
            raise ValidationError(
                "`target_thp` is required when `mode` is ProducerControlMode.THP."
            )
        if not (0 < self.efficiency_factor <= 1):
            raise ValidationError(
                f"`efficiency_factor` must be in (0, 1]; got {self.efficiency_factor}."
            )


@_control_type
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
    target_rate: typing.Optional[Number] = None
    target_bhp: typing.Optional[Number] = None
    target_thp: typing.Optional[Number] = None
    limits: typing.Tuple[Limit, ...] = attrs.field(factory=tuple, converter=tuple)
    efficiency_factor: Number = 1.0
    guide_rate: typing.Optional[Number] = None
    """
    Weight used by group-target allocation (deck WGRUPCON item 3).
    `None` falls back to equal-weight allocation among eligible wells.

    See `wells.resolution.allocation`.
    """

    def __attrs_post_init__(self) -> None:
        if self.mode in INJECTOR_RATE_MODES and self.target_rate is None:
            raise ValidationError(
                f"`target_rate` is required when `mode` is {self.mode}."
            )
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


class WellControls(StoreSerializable):
    """Name-keyed, mutable mapping from well name to its current `WellControl`."""

    __abstract_serializable__ = True

    def __init__(self, controls: typing.Dict[str, WellControl]) -> None:
        self._controls = dict(controls)

    def get(self, name: str) -> typing.Optional[WellControl]:
        """Current control for `name`, or `None` if unset."""
        return self._controls.get(name)

    def set(self, name: str, control: WellControl) -> None:
        """
        Replace the current control for `name` wholesale (say fo a
        WCONPROD/WCONINJE reissue, a limit-triggered mode switch, or an
        initial assignment).
        """
        self._controls[name] = control

    def update(self, name: str, **fields: typing.Any) -> None:
        """
                Modify one or more fields of `name`'s current control without
                replacing it wholesale.

                The `WellControl` analogue of deck `WELTARG`/`WELCNTL` (single-target edits),
                once `factories.py` parses those.

                :raises KeyError    groups: typing.Optional[Groups] = None
        : If `name` has no current control set.
        """
        current = self._controls.get(name)
        if current is None:
            raise KeyError(f"No control set for well {name!r}.")
        self._controls[name] = attrs.evolve(current, **fields)

    def __getitem__(self, name: str) -> WellControl:
        control = self.get(name)
        if control is None:
            raise KeyError(f"No control set for well {name!r}.")
        return control

    def __setitem__(self, name: str, control: WellControl) -> None:
        self.set(name, control)

    def __delitem__(self, name: str) -> None:
        if name not in self._controls:
            raise KeyError(f"No control set for well {name!r}.")
        del self._controls[name]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._controls)

    def __len__(self) -> int:
        return len(self._controls)

    def __contains__(self, name: object) -> bool:
        return name in self._controls

    def __dump__(self) -> typing.Dict[str, typing.Any]:
        return {
            "controls": {
                name: control.dump() for name, control in self._controls.items()
            }
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        controls = {
            name: WellControl.load(control_data)
            for name, control_data in data["controls"].items()
        }
        return cls(controls=controls)
