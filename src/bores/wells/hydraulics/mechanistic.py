"""No-slip homogeneous mixture wellbore hydraulics."""

import typing

import attrs
import numpy as np

from bores.constants import c
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import FluidPhase, Number, NumberArray, OneDimension
from bores.wells.base import Well
from bores.wells.hydraulics.base import (
    SurfaceFluidProperties,
    compute_mixture_density,
    compute_mixture_velocity,
    compute_mixture_viscosity,
    compute_segment_pressure_drop,
    compute_static_hydrostatic_drop,
    static_mixture_density,
)
from bores.wells.perforations import PerforationIndex
from bores.wells.states import ConnectionSample

__all__ = ["MechanisticWellboreModel"]


@attrs.frozen(kw_only=True, slots=True)
class MechanisticWellboreModel(Serializable):
    """
    No-slip mixture `WellboreModel`.

    One segment per (reference_depth -> connection) pair, 
    no intermediate discretization. 
    """

    friction_method: typing.Literal["simplified", "colebrook"] = "simplified"
    
    gravitational_acceleration: typing.Optional[Number] = None
    """
    `c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE` if `None` -
    a `METRIC`/`SI` well needs the matching
    `c.ACCELERATION_DUE_TO_GRAVITY_*` value supplied explicitly, never
    silently assumed for a non-FIELD well.
    """

    laminar_reynolds_limit: typing.Optional[Number] = None
    """`c.WELLBORE_LAMINAR_REYNOLDS_LIMIT` if `None`."""

    turbulent_reynolds_limit: typing.Optional[Number] = None
    """`c.WELLBORE_TURBULENT_REYNOLDS_LIMIT` if `None`."""

    friction_max_iterations: typing.Optional[int] = None
    """`c.COLEBROOK_MAX_ITERATIONS` if `None`."""

    friction_tolerance: typing.Optional[Number] = None
    """`c.COLEBROOK_TOLERANCE` if `None`."""

    def _get_gravitational_acceleration(self) -> Number:
        if self.gravitational_acceleration is not None:
            return self.gravitational_acceleration
        return c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE

    def perforation_pressures(
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
        assert well.tubing_inner_diameter is not None
        gravitational_acceleration = self._get_gravitational_acceleration()
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
                    gravitational_acceleration=gravitational_acceleration,
                )
                pressures[i] = reference_pressure + geometric_sign * drop.total
                continue

            mixture_density = compute_mixture_density(
                phase_rates, sample.phase_densities
            )
            mixture_viscosity = compute_mixture_viscosity(
                phase_rates, sample.phase_viscosities
            )
            velocity = compute_mixture_velocity(phase_rates, well.tubing_inner_diameter)
            drop = compute_segment_pressure_drop(
                length=abs(dz),
                inclination_from_vertical=pidx.inclination_from_vertical,
                tubing_inner_diameter=well.tubing_inner_diameter,
                tubing_roughness=well.tubing_roughness,
                mixture_density=mixture_density,
                mixture_viscosity=mixture_viscosity,
                mixture_velocity_in=velocity,
                mixture_velocity_out=velocity,
                gravitational_acceleration=gravitational_acceleration,
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

    def tubing_head_pressure(
        self,
        well: Well,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        surface_fluid_properties: SurfaceFluidProperties,
        is_injector: bool,
    ) -> Number:
        assert well.tubing_inner_diameter is not None
        gravitational_acceleration = self._get_gravitational_acceleration()
        dz = 0.0 - well.reference_depth
        total_rate = sum(phase_rates.values())
        friction_sign = -1.0 if is_injector else 1.0

        if total_rate == 0.0:
            drop = compute_static_hydrostatic_drop(
                mixture_density=surface_fluid_properties.density,
                length=abs(dz),
                gravitational_acceleration=gravitational_acceleration,
            )
            return reference_pressure - drop.total

        velocity = compute_mixture_velocity(phase_rates, well.tubing_inner_diameter)
        drop = compute_segment_pressure_drop(
            length=abs(dz),
            inclination_from_vertical=0.0,  # surface tubing: always vertical
            tubing_inner_diameter=well.tubing_inner_diameter,
            tubing_roughness=well.tubing_roughness,
            mixture_density=surface_fluid_properties.density,
            mixture_viscosity=surface_fluid_properties.viscosity,
            mixture_velocity_in=velocity,
            mixture_velocity_out=velocity,
            gravitational_acceleration=gravitational_acceleration,
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
