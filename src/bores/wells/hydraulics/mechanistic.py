"""No-slip homogeneous mixture wellbore hydraulics."""

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
    compute_mixture_density,
    compute_mixture_velocity,
    compute_mixture_viscosity,
    compute_segment_pressure_drop,
    compute_static_hydrostatic_drop,
    static_mixture_density,
    wellbore_type,
)
from bores.wells.perforations import PerforationIndex
from bores.wells.states import ConnectionSample

__all__ = ["MechanisticWellbore"]


@wellbore_type
@attrs.frozen(kw_only=True, slots=True)
class MechanisticWellbore(Wellbore):
    """
    No-slip mixture `Wellbore`.

    One segment per (reference_depth -> connection) pair,
    no intermediate discretization.
    """

    __type__ = "mechanistic"

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
    (a Reynolds number).
    """

    turbulent_reynolds_limit: typing.Optional[Number] = None
    """
    `c.WELLBORE_TURBULENT_REYNOLDS_LIMIT` if `None`. Dimensionless,
    same as `laminar_reynolds_limit`.
    """

    friction_max_iterations: typing.Optional[int] = None
    """
    `c.COLEBROOK_MAX_ITERATIONS` if `None`. An iteration count, not
    unit-system-dependent.
    """

    friction_tolerance: typing.Optional[Number] = None
    """
    `c.COLEBROOK_TOLERANCE` if `None`. A dimensionless convergence
    tolerance, not unit-system-dependent.
    """

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
        pressures = np.empty(len(perforation_indices), dtype=float)
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

            mixture_density = compute_mixture_density(
                phase_rates, sample.phase_densities
            )
            mixture_viscosity = compute_mixture_viscosity(
                phase_rates, sample.phase_viscosities
            )
            velocity = compute_mixture_velocity(phase_rates, self.tubing_inner_diameter)
            drop = compute_segment_pressure_drop(
                length=abs(dz),
                inclination_from_vertical=pidx.inclination_from_vertical,
                tubing_inner_diameter=self.tubing_inner_diameter,
                tubing_roughness=self.tubing_roughness,
                mixture_density=mixture_density,
                mixture_viscosity=mixture_viscosity,
                mixture_velocity_in=velocity,
                mixture_velocity_out=velocity,
                gravitational_acceleration=self.gravitational_acceleration,
                unit_system=self.unit_system,
                friction_method=self.friction_method,
                laminar_reynolds_limit=self.laminar_reynolds_limit,
                turbulent_reynolds_limit=self.turbulent_reynolds_limit,
                friction_max_iterations=self.friction_max_iterations,
                friction_tolerance=self.friction_tolerance,
            )
            pressures[i] = (
                reference_pressure
                + geometric_sign * (drop.hydrostatic + drop.acceleration)
                + friction_sign * drop.friction
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

        mixture_viscosity = surface_fluid_properties.get_mixture_viscosity(phase_rates)
        velocity = compute_mixture_velocity(phase_rates, self.tubing_inner_diameter)
        drop = compute_segment_pressure_drop(
            length=abs(dz),
            inclination_from_vertical=0.0,  # Surface tubing is always vertical
            tubing_inner_diameter=self.tubing_inner_diameter,
            tubing_roughness=self.tubing_roughness,
            mixture_density=mixture_density,
            mixture_viscosity=mixture_viscosity,
            mixture_velocity_in=velocity,
            mixture_velocity_out=velocity,
            gravitational_acceleration=self.gravitational_acceleration,
            unit_system=self.unit_system,
            friction_method=self.friction_method,
            laminar_reynolds_limit=self.laminar_reynolds_limit,
            turbulent_reynolds_limit=self.turbulent_reynolds_limit,
            friction_max_iterations=self.friction_max_iterations,
            friction_tolerance=self.friction_tolerance,
        )
        return (
            reference_pressure
            - (drop.hydrostatic + drop.acceleration)
            - friction_sign * drop.friction
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Returns a new `MechanisticWellbore` in the *target* unit system.

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
