import typing

import attrs

from bores.constants import c
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import FluidPhase, Number
from bores.wells.controls import Limit

__all__ = ["ControlResolution", "ControlResolverSpec"]


@attrs.frozen(kw_only=True, slots=True)
class ControlResolution(Serializable):
    """Result of resolving one well's control for one timestep."""

    bhp: Number

    phase_rates: typing.Mapping[FluidPhase, Number]

    active_limit: typing.Optional[Limit] = None
    """
    Which `Limit` from the spec's `limits` tuple is currently binding,
    `None` if the primary target is achieved without hitting any limit.
    """

    economic_shutin: bool = False
    """
    True if an `EconomicLimit` was violated. `phase_rates` are zeroed in
    that case. The caller (simulation loop) decides whether/how to act on
    this (e.g. set WellState.is_open=False for the next timestep).
    """

    thp: typing.Optional[Number] = None


@attrs.frozen(kw_only=True, slots=True)
class ControlResolverSpec(Serializable):
    """
    Numerical tuning for well-control resolution.

    Solver tunables for well control resolution.
    """

    max_fixed_point_iterations: typing.Optional[int] = None
    rate_convergence_tolerance: typing.Optional[Number] = None
    max_bisection_iterations: typing.Optional[int] = None
    producer_bhp_floor: typing.Optional[Number] = None
    injector_bhp_bracket_multiplier: typing.Optional[Number] = None

    def __attrs_post_init__(self) -> None:
        if self.max_fixed_point_iterations is None:
            object.__setattr__(
                self,
                "max_fixed_point_iterations",
                c.CONTROL_MAX_FIXED_POINT_ITERATIONS,
            )
        elif self.max_fixed_point_iterations < 1:
            raise ValidationError(
                "`max_fixed_point_iterations` must be >= 1; got "
                f"{self.max_fixed_point_iterations}."
            )

        if self.rate_convergence_tolerance is None:
            object.__setattr__(
                self,
                "rate_convergence_tolerance",
                c.CONTROL_RATE_CONVERGENCE_TOLERANCE,
            )
        elif self.rate_convergence_tolerance <= 0:
            raise ValidationError(
                "`rate_convergence_tolerance` must be positive; got "
                f"{self.rate_convergence_tolerance}."
            )

        if self.max_bisection_iterations is None:
            object.__setattr__(
                self,
                "max_bisection_iterations",
                c.CONTROL_MAX_BISECTION_ITERATIONS,
            )
        elif self.max_bisection_iterations < 1:
            raise ValidationError(
                "`max_bisection_iterations` must be >= 1; got "
                f"{self.max_bisection_iterations}."
            )

        if self.producer_bhp_floor is None:
            object.__setattr__(
                self,
                "producer_bhp_floor",
                c.MINIMUM_VALID_PRESSURE,
            )
        elif self.producer_bhp_floor <= 0:
            raise ValidationError(
                f"`producer_bhp_floor` must be positive; got {self.producer_bhp_floor}."
            )

        if self.injector_bhp_bracket_multiplier is None:
            object.__setattr__(
                self,
                "injector_bhp_bracket_multiplier",
                c.CONTROL_INJECTOR_BHP_BRACKET_MULTIPLIER,
            )
        elif self.injector_bhp_bracket_multiplier <= 1.0:
            raise ValidationError(
                "`injector_bhp_bracket_multiplier` must be > 1.0; got "
                f"{self.injector_bhp_bracket_multiplier}."
            )
