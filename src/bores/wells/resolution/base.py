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

    Solver tunables for `resolve_control`/`build_well_state`.

    Construct once and pass explicitly to override any default; every
    function in `wells.control` that needs one of these falls back to
    `ControlResolverSpec()` (all-default) if the caller doesn't supply
    one.
    """

    _max_fixed_point_iterations: typing.Optional[int] = attrs.field(
        default=None, alias="max_fixed_point_iterations"
    )
    _rate_convergence_tolerance: typing.Optional[Number] = attrs.field(
        default=None, alias="rate_convergence_tolerance"
    )
    _max_bisection_iterations: typing.Optional[int] = attrs.field(
        default=None, alias="max_bisection_iterations"
    )
    _producer_bhp_floor: typing.Optional[Number] = attrs.field(
        default=None, alias="producer_bhp_floor"
    )
    _injector_bhp_bracket_multiplier: typing.Optional[Number] = attrs.field(
        default=None, alias="injector_bhp_bracket_multiplier"
    )

    def __attrs_post_init__(self) -> None:
        if (
            self._max_fixed_point_iterations is not None
            and self._max_fixed_point_iterations < 1
        ):
            raise ValidationError(
                "`max_fixed_point_iterations` must be >= 1; got "
                f"{self._max_fixed_point_iterations}."
            )
        if (
            self._max_bisection_iterations is not None
            and self._max_bisection_iterations < 1
        ):
            raise ValidationError(
                f"`max_bisection_iterations` must be >= 1; got "
                f"{self._max_bisection_iterations}."
            )
        if (
            self._rate_convergence_tolerance is not None
            and self._rate_convergence_tolerance <= 0
        ):
            raise ValidationError(
                "`rate_convergence_tolerance` must be positive; got "
                f"{self._rate_convergence_tolerance}."
            )
        if self._producer_bhp_floor is not None and self._producer_bhp_floor <= 0:
            raise ValidationError(
                f"`producer_bhp_floor` must be positive; got {self._producer_bhp_floor}."
            )
        if (
            self._injector_bhp_bracket_multiplier is not None
            and self._injector_bhp_bracket_multiplier <= 1.0
        ):
            raise ValidationError(
                "`injector_bhp_bracket_multiplier` must be > 1.0; got "
                f"{self._injector_bhp_bracket_multiplier}."
            )

    @property
    def max_fixed_point_iterations(self) -> int:
        """Bound on the perforation-pressure <-> IPR-rate fixed-point loop
        that reconciles flowing pressures against rates at a fixed BHP."""
        if self._max_fixed_point_iterations is not None:
            return self._max_fixed_point_iterations

        return c.CONTROL_MAX_FIXED_POINT_ITERATIONS

    @property
    def rate_convergence_tolerance(self) -> Number:
        """Relative convergence tolerance on total phase rate, shared by
        the fixed-point loop and the BHP bisection search."""
        if self._rate_convergence_tolerance is not None:
            return self._rate_convergence_tolerance

        return c.CONTROL_RATE_CONVERGENCE_TOLERANCE

    @property
    def max_bisection_iterations(self) -> int:
        """Bound on the BHP bisection search for a rate-mode or THP-limit
        target. Best-effort, not exact: the search returns its closest
        bound reached after this many iterations rather than raising if
        the target isn't achievable within the bracket."""
        if self._max_bisection_iterations is not None:
            return self._max_bisection_iterations

        return c.CONTROL_MAX_BISECTION_ITERATIONS

    @property
    def producer_bhp_floor(self) -> Number:
        """Lower bracket bound for a producer's BHP bisection search."""
        if self._producer_bhp_floor is not None:
            return self._producer_bhp_floor

        return c.MINIMUM_VALID_PRESSURE

    @property
    def injector_bhp_bracket_multiplier(self) -> Number:
        """Upper bracket bound for an injector's BHP bisection search,
        expressed as a multiple of the highest connected-cell pressure."""
        if self._injector_bhp_bracket_multiplier is not None:
            return self._injector_bhp_bracket_multiplier

        return c.CONTROL_INJECTOR_BHP_BRACKET_MULTIPLIER
