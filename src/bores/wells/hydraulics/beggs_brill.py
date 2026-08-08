"""
Beggs & Brill (1973) two-phase wellbore hydraulics.

Closed-form flow-pattern-dependent liquid holdup and friction correction.

Reference: Beggs, H.D. and Brill, J.P., "A Study of Two-Phase Flow in
Inclined Pipes", Journal of Petroleum Technology, May 1973.
"""

import math
import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.constants import c, get_conversion_factors
from bores.errors import ValidationError
from bores.typing import (
    FluidPhase,
    Number,
    NumberArray,
    OneDimension,
    UnitConversionTable,
    UnitSystem,
)
from bores.wells.base import Well
from bores.wells.hydraulics.base import (
    SurfaceFluidProperties,
    Wellbore,
    compute_friction_factor,
    compute_mixture_velocity,
    compute_static_hydrostatic_drop,
    static_mixture_density,
    wellbore_type,
)
from bores.wells.indices.perforations import PerforationIndex
from bores.wells.states import ConnectionSample

__all__ = ["FlowPattern", "compute_beggs_brill_holdup", "BeggsBrillWellbore"]


class FlowPattern(typing.NamedTuple):
    """Result of the Beggs-Brill horizontal flow-pattern classification."""

    name: typing.Literal["segregated", "transition", "intermittent", "distributed"]
    no_slip_holdup: Number
    froude_number: Number


def _get_flow_pattern_boundaries(
    no_slip_holdup: Number,
) -> typing.Tuple[Number, Number, Number, Number]:
    """
    `(L1, L2, L3, L4)` Froude-number flow-pattern boundaries at this
    `no_slip_holdup` (`lambda_l`).
    """
    l1 = 316.0 * no_slip_holdup**0.302
    l2 = 0.0009252 * no_slip_holdup**-2.4684
    l3 = 0.10 * no_slip_holdup**-1.4516
    l4 = 0.5 * no_slip_holdup**-6.738
    return l1, l2, l3, l4


def _classify_flow_pattern(
    no_slip_holdup: Number, froude_number: Number
) -> FlowPattern:
    """
    Classify horizontal flow pattern from `no_slip_holdup` (`lambda_l`)
    and `froude_number` against the Beggs-Brill boundaries.

    :param no_slip_holdup: `v_sl / (v_sl + v_sg)`.
    :param froude_number: `v_m**2 / (g * d)`.
    :returns: `FlowPattern` with `name` one of `"segregated"`,
        `"transition"`, `"intermittent"`, `"distributed"`.
    """
    l1, l2, l3, l4 = _get_flow_pattern_boundaries(no_slip_holdup)
    if (no_slip_holdup < 0.01 and froude_number < l1) or (
        no_slip_holdup >= 0.01 and froude_number < l2
    ):
        pattern_name = "segregated"
    elif no_slip_holdup >= 0.01 and l2 <= froude_number <= l3:
        pattern_name = "transition"
    elif (0.01 <= no_slip_holdup < 0.4 and l3 < froude_number <= l1) or (
        no_slip_holdup >= 0.4 and l3 < froude_number <= l4
    ):
        pattern_name = "intermittent"
    else:
        pattern_name = "distributed"

    return FlowPattern(
        name=pattern_name,
        no_slip_holdup=no_slip_holdup,
        froude_number=froude_number,
    )


HORIZONTAL_HOLDUP_COEFFICIENTS: typing.Dict[
    str, typing.Tuple[Number, Number, Number]
] = {
    # (a, b, c) in H_L(0) = a * lambda_l**b / Fr**c
    "segregated": (0.98, 0.4846, 0.0868),
    "intermittent": (0.845, 0.5351, 0.0173),
    "distributed": (1.065, 0.5824, 0.0609),
}

_InclinationCoefficients = typing.Tuple[Number, Number, Number, Number]

INCLINATION_COEFFICIENTS_UPHILL: typing.Dict[
    str, typing.Optional[_InclinationCoefficients]
] = {
    # (d, e, f, g) in C = (1 - lambda_l) * ln(d * lambda_l**e * Nlv**f * Fr**g)
    "segregated": (0.011, -3.768, 3.539, -1.614),
    "intermittent": (2.96, 0.305, -0.4473, 0.0978),
    "distributed": None,  # no correction: C = 0, psi = 1
}
INCLINATION_COEFFICIENTS_DOWNHILL: _InclinationCoefficients = (
    4.70,
    -0.3692,
    0.1244,
    -0.5056,
)


def _compute_horizontal_holdup(pattern: FlowPattern) -> Number:
    """
    `H_L(0)`, the flow-pattern-dependent horizontal liquid holdup, clipped
    to `>= no_slip_holdup` (holdup can't physically fall below the
    no-slip/input liquid fraction).

    Interpolates between the segregated and intermittent results for
    `pattern.name == "transition"`, per Beggs & Brill (1973).
    """
    if pattern.name == "transition":
        _, l2, l3, _ = _get_flow_pattern_boundaries(pattern.no_slip_holdup)
        a_segregated, b_segregated, c_segregated = HORIZONTAL_HOLDUP_COEFFICIENTS[
            "segregated"
        ]
        a_intermittent, b_intermittent, c_intermittent = HORIZONTAL_HOLDUP_COEFFICIENTS[
            "intermittent"
        ]
        hl_segregated = (
            a_segregated
            * pattern.no_slip_holdup**b_segregated
            / pattern.froude_number**c_segregated
        )
        hl_intermittent = (
            a_intermittent
            * pattern.no_slip_holdup**b_intermittent
            / pattern.froude_number**c_intermittent
        )
        interpolation_weight = (l3 - pattern.froude_number) / (l3 - l2)
        holdup = (
            interpolation_weight * hl_segregated
            + (1.0 - interpolation_weight) * hl_intermittent
        )
    else:
        a, b, cc = HORIZONTAL_HOLDUP_COEFFICIENTS[pattern.name]
        holdup = a * pattern.no_slip_holdup**b / pattern.froude_number**cc

    return max(holdup, pattern.no_slip_holdup)


def compute_beggs_brill_holdup(
    *,
    superficial_liquid_velocity: Number,
    superficial_gas_velocity: Number,
    liquid_density: Number,
    liquid_surface_tension: Number,
    tubing_inner_diameter: Number,
    inclination_from_vertical: Number,
    is_injector: bool,
    gravitational_acceleration: Number,
) -> Number:
    """
    In-situ liquid holdup `H_L(theta)` by the Beggs & Brill (1973)
    correlation.

    Determines the horizontal flow pattern (segregated/transition/
    intermittent/distributed) from the no-slip holdup and Froude number,
    reads the flow-pattern-dependent horizontal holdup `H_L(0)`, then
    applies the inclination correction factor `psi` using the uphill
    coefficient set when flow moves from lower to higher elevation
    (a producer) and the downhill set when it moves from higher to lower
    (an injector), regardless of the segment's own geometric orientation.

    :param superficial_liquid_velocity: `v_sl = q_l / A`.
    :param superficial_gas_velocity: `v_sg = q_g / A`.
    :param liquid_density: Liquid-phase density.
    :param liquid_surface_tension: Gas-liquid surface tension.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param inclination_from_vertical: `0` = vertical, `pi/2` = horizontal.
    :param is_injector: Selects the uphill/downhill inclination
        coefficient set - see above.
    :param gravitational_acceleration: For the Froude number.
    :returns: `H_L(theta)`, clipped to `[no_slip_holdup, 1.0]`.
    :raises ValidationError: If both superficial velocities are zero.
    """
    mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
    if mixture_velocity <= 0.0:
        raise ValidationError(
            "`compute_beggs_brill_holdup` requires a positive mixture velocity; "
            "use `compute_static_hydrostatic_drop` for the no-flow case instead."
        )

    no_slip_holdup = superficial_liquid_velocity / mixture_velocity
    froude_number = mixture_velocity**2 / (
        gravitational_acceleration * tubing_inner_diameter
    )
    liquid_velocity_number = (
        1.938
        * superficial_liquid_velocity
        * (liquid_density / liquid_surface_tension) ** 0.25
    )

    pattern = _classify_flow_pattern(no_slip_holdup, froude_number)
    horizontal_holdup = _compute_horizontal_holdup(pattern)

    theta_from_horizontal = (math.pi / 2.0) - inclination_from_vertical

    if is_injector:
        d_coef, e_coef, f_coef, g_coef = INCLINATION_COEFFICIENTS_DOWNHILL
        correction_argument = (
            d_coef
            * no_slip_holdup**e_coef
            * liquid_velocity_number**f_coef
            * froude_number**g_coef
        )
        correction_coefficient = max(
            0.0, (1.0 - no_slip_holdup) * math.log(correction_argument)
        )
    else:
        uphill_coefficients = INCLINATION_COEFFICIENTS_UPHILL[pattern.name]
        if uphill_coefficients is None:  # distributed: no correction
            correction_coefficient = 0.0
        else:
            d_coef, e_coef, f_coef, g_coef = uphill_coefficients
            correction_argument = (
                d_coef
                * no_slip_holdup**e_coef
                * liquid_velocity_number**f_coef
                * froude_number**g_coef
            )
            correction_coefficient = max(
                0.0, (1.0 - no_slip_holdup) * math.log(correction_argument)
            )

    psi = 1.0 + correction_coefficient * (
        math.sin(1.8 * theta_from_horizontal)
        - (1.0 / 3.0) * math.sin(1.8 * theta_from_horizontal) ** 3
    )
    in_situ_holdup = horizontal_holdup * psi
    return min(max(in_situ_holdup, no_slip_holdup), 1.0)


def _compute_two_phase_friction_factor(
    *,
    no_slip_holdup: Number,
    in_situ_holdup: Number,
    no_slip_reynolds_number: Number,
    relative_roughness: Number,
    friction_method: typing.Literal["simplified", "colebrook"],
    laminar_reynolds_limit: typing.Optional[Number],
    turbulent_reynolds_limit: typing.Optional[Number],
    friction_max_iterations: typing.Optional[int],
    friction_tolerance: typing.Optional[Number],
) -> Number:
    """
    Two-phase Darcy friction factor: the no-slip friction factor corrected
    by the Beggs-Brill ratio `f_tp / f_ns = exp(S)`.

    :param no_slip_holdup: `lambda_l`.
    :param in_situ_holdup: `H_L(theta)` from `compute_beggs_brill_holdup`.
    :param no_slip_reynolds_number: `rho_ns * v_m * d / mu_ns`.
    :param relative_roughness: `tubing_roughness / tubing_inner_diameter`.
    :param friction_method: Forwarded to `compute_friction_factor` for `f_ns`.
    :param laminar_reynolds_limit: Forwarded to `compute_friction_factor`.
    :param turbulent_reynolds_limit: Forwarded to `compute_friction_factor`.
    :param friction_max_iterations: Forwarded to `compute_friction_factor`.
    :param friction_tolerance: Forwarded to `compute_friction_factor`.
    :returns: Two-phase Darcy friction factor.
    """
    no_slip_friction_factor = compute_friction_factor(
        reynolds_number=no_slip_reynolds_number,
        relative_roughness=relative_roughness,
        method=friction_method,
        laminar_reynolds_limit=laminar_reynolds_limit,
        turbulent_reynolds_limit=turbulent_reynolds_limit,
        max_iterations=friction_max_iterations,
        tolerance=friction_tolerance,
    )

    y = no_slip_holdup / in_situ_holdup**2
    if 1.0 < y < 1.2:
        s = math.log(2.2 * y - 1.2)
    else:
        ln_y = math.log(y)
        s = ln_y / (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4)
    return no_slip_friction_factor * math.exp(s)


@wellbore_type
@attrs.frozen(kw_only=True, slots=True)
class BeggsBrillWellbore(Wellbore):
    """
    Beggs & Brill (1973) `Wellbore`.

    Implements flow-pattern-dependent liquid holdup (in-situ mixture
    density differs from the no-slip/rate-weighted density
    `MechanisticWellbore` uses) and a corrected two-phase friction
    factor.

    One segment per (reference_depth -> connection) pair, same as
    `MechanisticWellbore`. Inclination comes from
    `PerforationIndex.inclination_from_vertical` and is populated by
    `wells.perforations` from `Perforation.direction` (vertical/horizontal
    only) for a well with no trajectory, or from the trajectory's local
    tangent for one with a trajectory.
    """

    __type__ = "beggs_brill"

    tubing_inner_diameter: Number

    tubing_roughness: typing.Optional[Number] = None
    """
    Absolute roughness, same length unit as `unit_system`. `None` means
    use a smooth-pipe assumption.
    """
    friction_method: typing.Literal["simplified", "colebrook"] = "simplified"

    unit_system: UnitSystem = UnitSystem.FIELD

    gravitational_acceleration: typing.Optional[Number] = None
    """
    Resolved on post initialization if left `None`. 
    
    Once resolved, this field always holds a real number and not `None` after construction.
    """

    laminar_reynolds_limit: typing.Optional[Number] = None
    """
    `c.WELLBORE_LAMINAR_REYNOLDS_LIMIT` if `None` - dimensionless
    (a Reynolds number), so no unit conversion applies regardless of
    `unit_system`.
    """

    turbulent_reynolds_limit: typing.Optional[Number] = None
    """
    `c.WELLBORE_TURBULENT_REYNOLDS_LIMIT` if `None`. Dimensionless,
    same as `laminar_reynolds_limit`.
    """

    friction_max_iterations: typing.Optional[int] = None
    """`c.COLEBROOK_MAX_ITERATIONS` if `None`. An iteration count, not
    unit-system-dependent."""

    friction_tolerance: typing.Optional[Number] = None
    """`c.COLEBROOK_TOLERANCE` if `None`. A dimensionless convergence
    tolerance, not unit-system-dependent."""

    def __attrs_post_init__(self) -> None:
        if self.tubing_inner_diameter <= 0:
            raise ValidationError("`tubing_inner_diameter` must be positive.")

        if self.gravitational_acceleration is None:
            field_gravitational_acceleration = (
                c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
            )
            if self.unit_system is UnitSystem.FIELD:
                resolved_gravitational_acceleration = field_gravitational_acceleration
            else:
                length_factor = get_conversion_factors(
                    UnitSystem.FIELD, self.unit_system
                )["length"]
                resolved_gravitational_acceleration = (
                    field_gravitational_acceleration * length_factor
                )
            object.__setattr__(
                self,
                "gravitational_acceleration",
                resolved_gravitational_acceleration,
            )

        if self.laminar_reynolds_limit is None:
            object.__setattr__(
                self, "laminar_reynolds_limit", c.WELLBORE_LAMINAR_REYNOLDS_LIMIT
            )
        if self.turbulent_reynolds_limit is None:
            object.__setattr__(
                self, "turbulent_reynolds_limit", c.WELLBORE_TURBULENT_REYNOLDS_LIMIT
            )
        if self.friction_max_iterations is None:
            object.__setattr__(
                self, "friction_max_iterations", c.COLEBROOK_MAX_ITERATIONS
            )
        if self.friction_tolerance is None:
            object.__setattr__(self, "friction_tolerance", c.COLEBROOK_TOLERANCE)

    def _get_segment_drop(
        self,
        *,
        length: Number,
        inclination_from_vertical: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        phase_densities: typing.Mapping[FluidPhase, Number],
        phase_viscosities: typing.Mapping[FluidPhase, Number],
        gas_liquid_surface_tension: Number,
        tubing_inner_diameter: Number,
        tubing_roughness: typing.Optional[Number],
        is_injector: bool,
    ) -> typing.Tuple[Number, Number]:
        """
        Return `(hydrostatic, friction)` for one segment, using Beggs-Brill
        in-situ holdup for the hydrostatic term and two-phase-corrected
        friction for the friction term.

        Liquid/gas split: `FluidPhase.GAS` is the gas phase; every other
        phase present (oil, water) is pooled into "liquid" - Beggs-Brill
        is a two-phase (liquid/gas) correlation and doesn't distinguish
        oil from water within the liquid phase.
        """
        assert self.gravitational_acceleration is not None
        cross_sectional_area = math.pi * (tubing_inner_diameter / 2.0) ** 2

        gas_rate = phase_rates.get(FluidPhase.GAS, 0.0)
        liquid_rate = sum(
            rate for phase, rate in phase_rates.items() if phase != FluidPhase.GAS
        )
        superficial_gas_velocity = gas_rate / cross_sectional_area
        superficial_liquid_velocity = liquid_rate / cross_sectional_area

        liquid_phases = [phase for phase in phase_rates if phase != FluidPhase.GAS]
        if liquid_rate > 0.0:
            liquid_density = (
                sum(
                    phase_rates[phase] * phase_densities[phase]
                    for phase in liquid_phases
                    if phase_rates[phase] != 0.0
                )
                / liquid_rate
            )
            liquid_viscosity = (
                sum(
                    phase_rates[phase] * phase_viscosities[phase]
                    for phase in liquid_phases
                    if phase_rates[phase] != 0.0
                )
                / liquid_rate
            )
        else:
            liquid_density = phase_densities.get(FluidPhase.OIL, 0.0)
            liquid_viscosity = phase_viscosities.get(FluidPhase.OIL, 0.0)

        gas_density = phase_densities.get(FluidPhase.GAS, 0.0)
        gas_viscosity = phase_viscosities.get(FluidPhase.GAS, 0.0)

        in_situ_holdup = compute_beggs_brill_holdup(
            superficial_liquid_velocity=superficial_liquid_velocity,
            superficial_gas_velocity=superficial_gas_velocity,
            liquid_density=liquid_density,
            liquid_surface_tension=gas_liquid_surface_tension,
            tubing_inner_diameter=tubing_inner_diameter,
            inclination_from_vertical=inclination_from_vertical,
            is_injector=is_injector,
            gravitational_acceleration=self.gravitational_acceleration,
        )
        in_situ_density = (liquid_density * in_situ_holdup) + gas_density * (
            1.0 - in_situ_holdup
        )
        hydrostatic_drop = in_situ_density * self.gravitational_acceleration * length

        mixture_velocity = superficial_liquid_velocity + superficial_gas_velocity
        no_slip_holdup = superficial_liquid_velocity / mixture_velocity
        no_slip_density = (liquid_density * no_slip_holdup) + gas_density * (
            1.0 - no_slip_holdup
        )
        no_slip_viscosity = (liquid_viscosity * no_slip_holdup) + gas_viscosity * (
            1.0 - no_slip_holdup
        )
        relative_roughness = (
            tubing_roughness / tubing_inner_diameter
            if tubing_roughness is not None
            else 0.0
        )
        no_slip_reynolds_number = (
            no_slip_density
            * mixture_velocity
            * tubing_inner_diameter
            / no_slip_viscosity
        )
        friction_factor = _compute_two_phase_friction_factor(
            no_slip_holdup=no_slip_holdup,
            in_situ_holdup=in_situ_holdup,
            no_slip_reynolds_number=no_slip_reynolds_number,
            relative_roughness=relative_roughness,
            friction_method=self.friction_method,
            laminar_reynolds_limit=self.laminar_reynolds_limit,
            turbulent_reynolds_limit=self.turbulent_reynolds_limit,
            friction_max_iterations=self.friction_max_iterations,
            friction_tolerance=self.friction_tolerance,
        )
        friction_drop = (
            friction_factor
            * (length / tubing_inner_diameter)
            * (no_slip_density * mixture_velocity**2 / 2.0)
        )
        return hydrostatic_drop, friction_drop

    def compute_perforation_pressures(
        self,
        well: Well,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        perforation_indices: typing.Sequence[PerforationIndex],
        connection_samples: typing.Sequence[ConnectionSample],
        is_injector: bool,
    ) -> NumberArray[OneDimension]:
        if len(connection_samples) != len(perforation_indices):
            raise ValidationError(
                f"len(connection_samples)={len(connection_samples)} != "
                f"len(perforation_indices)={len(perforation_indices)} for "
                f"well {well.name!r}."
            )
        assert self.gravitational_acceleration is not None

        total_rate = sum(phase_rates.values())
        pressures = np.empty(len(perforation_indices), dtype=np.float64)
        friction_sign = -1.0 if is_injector else 1.0

        for i, (pidx, sample) in enumerate(
            zip(perforation_indices, connection_samples)
        ):
            dz = pidx.representative_depth - well.reference_depth
            geometric_sign = 1.0 if dz >= 0 else -1.0

            if total_rate == 0.0:
                drop = compute_static_hydrostatic_drop(
                    mixture_density=static_mixture_density(sample),
                    length=abs(dz),
                    gravitational_acceleration=self.gravitational_acceleration,
                    unit_system=self.unit_system,
                )
                pressures[i] = reference_pressure + geometric_sign * drop.total
                continue

            hydrostatic_drop, friction_drop = self._get_segment_drop(
                length=abs(dz),
                inclination_from_vertical=pidx.inclination_from_vertical,
                phase_rates=phase_rates,
                phase_densities=sample.phase_densities,
                phase_viscosities=sample.phase_viscosities,
                gas_liquid_surface_tension=sample.gas_liquid_surface_tension,
                tubing_inner_diameter=self.tubing_inner_diameter,
                tubing_roughness=self.tubing_roughness,
                is_injector=is_injector,
            )
            pressures[i] = (
                reference_pressure
                + (geometric_sign * hydrostatic_drop)
                + (friction_sign * friction_drop)
            )
        return pressures

    def compute_tubing_head_pressure(
        self,
        well: Well,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        surface_fluid_properties: SurfaceFluidProperties,
        is_injector: bool,
    ) -> Number:
        """
        Uses a real Beggs-Brill surface-condition holdup and two-phase
        friction for this segment when `surface_fluid_properties` carries
        `phase_densities`/`phase_viscosities`/`gas_liquid_surface_tension`;
        falls back to a no-slip mixture otherwise. The surface segment
        (reference_depth -> surface) is always treated as vertical either
        way - tubing above the last connection isn't part of any
        trajectory this package resolves.
        """
        assert self.gravitational_acceleration is not None
        dz = 0.0 - well.reference_depth
        total_rate = sum(phase_rates.values())
        friction_sign = -1.0 if is_injector else 1.0
        mixture_density = surface_fluid_properties.get_mixture_density(phase_rates)

        if total_rate == 0.0:
            drop = compute_static_hydrostatic_drop(
                mixture_density=mixture_density,
                length=abs(dz),
                gravitational_acceleration=self.gravitational_acceleration,
                unit_system=self.unit_system,
            )
            return reference_pressure - drop.total

        has_full_surface_properties = (
            surface_fluid_properties.phase_densities is not None
            and surface_fluid_properties.phase_viscosities is not None
            and surface_fluid_properties.gas_liquid_surface_tension is not None
        )
        tubing_inner_diameter = self.tubing_inner_diameter
        if has_full_surface_properties:
            assert surface_fluid_properties.phase_densities is not None
            assert surface_fluid_properties.phase_viscosities is not None
            assert surface_fluid_properties.gas_liquid_surface_tension is not None
            hydrostatic_drop, friction_drop = self._get_segment_drop(
                length=abs(dz),
                inclination_from_vertical=0.0,
                phase_rates=phase_rates,
                phase_densities=surface_fluid_properties.phase_densities,
                phase_viscosities=surface_fluid_properties.phase_viscosities,
                gas_liquid_surface_tension=surface_fluid_properties.gas_liquid_surface_tension,
                tubing_inner_diameter=tubing_inner_diameter,
                tubing_roughness=self.tubing_roughness,
                is_injector=is_injector,
            )
            return (
                reference_pressure - hydrostatic_drop - (friction_sign * friction_drop)
            )

        mixture_viscosity = surface_fluid_properties.get_mixture_viscosity(phase_rates)
        velocity = compute_mixture_velocity(phase_rates, tubing_inner_diameter)
        relative_roughness = (
            self.tubing_roughness / tubing_inner_diameter
            if self.tubing_roughness is not None
            else 0.0
        )
        reynolds_number = (
            mixture_density * velocity * tubing_inner_diameter / mixture_viscosity
        )
        friction_factor = compute_friction_factor(
            reynolds_number=reynolds_number,
            relative_roughness=relative_roughness,
            method=self.friction_method,
            laminar_reynolds_limit=self.laminar_reynolds_limit,
            turbulent_reynolds_limit=self.turbulent_reynolds_limit,
            max_iterations=self.friction_max_iterations,
            tolerance=self.friction_tolerance,
        )
        hydrostatic_drop = mixture_density * self.gravitational_acceleration * abs(dz)
        friction_drop = (
            friction_factor
            * (abs(dz) / tubing_inner_diameter)
            * (mixture_density * velocity**2 / 2.0)
        )
        return reference_pressure - hydrostatic_drop - (friction_sign * friction_drop)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Returns a new `BeggsBrillWellbore` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New model with `gravitational_acceleration` converted to
            `target`. `laminar_reynolds_limit`/`turbulent_reynolds_limit`/
            `friction_max_iterations`/`friction_tolerance` are dimensionless
            and unchanged.
        """
        if target == self.unit_system:
            return self

        assert self.gravitational_acceleration is not None
        length_factor = get_conversion_factors(self.unit_system, target, table=table)[
            "length"
        ]
        return attrs.evolve(
            self,
            tubing_inner_diameter=self.tubing_inner_diameter * length_factor,
            tubing_roughness=self.tubing_roughness * length_factor
            if self.tubing_roughness is not None
            else None,
            gravitational_acceleration=self.gravitational_acceleration * length_factor,
            unit_system=target,
        )
