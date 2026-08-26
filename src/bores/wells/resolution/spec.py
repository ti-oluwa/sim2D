
import attrs

from bores.constants import c
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import Integer, Number

__all__ = ["ControlResolverSpec"]


@attrs.frozen(kw_only=True, slots=True)
class ControlResolverSpec(Serializable):
    """
    Numerical tuning for well-control resolution.

    Solver tunables for well control resolution.
    """

    max_fixed_point_iterations: Integer | None = None
    rate_convergence_tolerance: Number | None = None
    max_bisection_iterations: Integer | None = None
    producer_bhp_floor: Number | None = None
    injector_bhp_bracket_multiplier: Number | None = None

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
                f"`max_bisection_iterations` must be >= 1; got {self.max_bisection_iterations}."
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
