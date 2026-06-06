"""Static data models and schemas for an N-dimensional reservoir model."""

import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.errors import ValidationError
from bores.grids.base import (
    apply_structural_dip,
    build_depth_grid,
    build_elevation_grid,
)
from bores.precision import get_dtype
from bores.stores import StoreSerializable
from bores.transmissibility import FaceTransmissibilities, build_face_transmissibilities
from bores.types import NDimension, NDimensionalGrid

__all__ = [
    "FluidProperties",
    "ReservoirModel",
    "RockPermeability",
    "RockProperties",
    "HysteresisState",
]


@typing.final
@attrs.frozen(slots=True)
class FluidProperties(StoreSerializable, typing.Generic[NDimension]):
    """
    Fluid properties of a reservoir model.

    Some of these properties are liable to change over time due to flow.

    Varying properties include:

    - Pressure
    - Temperature
    - Oil saturation
    - Oil viscosity
    - Oil compressibility
    - Oil density
    - Water saturation
    - Water viscosity
    - Water compressibility
    - Water density
    - Gas saturation
    - Gas viscosity
    - Gas compressibility
    - Gas density
    - Gas-to-oil ratio
    - Oil formation volume factor
    - Gas formation volume factor
    - Water formation volume factor
    - Oil bubble point pressure
    - Water bubble point pressure
    - Gas solubility in water
    - Solvent mass concentration in oil phase
    - Effective oil-solvent mixture viscosity
    - Effective oil-solvent mixture density

    Constant properties include:
    - Oil specific gravity
    - Oil API gravity
    - Water salinity
    - Gas gravity
    - Gas molecular weight

    These properties are typically constant for a given fluid type, e.g., light oil, seawater, methane.
    """

    pressure_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the pressure distribution in the reservoir (psi)."""
    temperature_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the temperature distribution across the reservoir (°F)."""
    oil_bubble_point_pressure_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the bubble point pressure distribution in the reservoir (psi)."""
    oil_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Oil) saturation distribution in the reservoir (fraction)."""
    oil_viscosity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Oil) viscosity distribution in the reservoir in (cP)."""
    oil_compressibility_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil compressibility distribution (on bulk volume basis) in the reservoir (psi⁻¹)."""
    oil_specific_gravity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil specific gravity distribution in the reservoir (dimensionless). Should be constant for a given oil, e.g., 0.85 for light oil)."""
    oil_api_gravity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil API gravity distribution in the reservoir (°API). should be constant for a given oil, e.g., 35°API for light oil)."""
    oil_density_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil density distribution (usually live-oil) in the reservoir (lbm/ft³)."""
    water_bubble_point_pressure_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the bubble point pressure distribution for water in the reservoir (psi)."""
    water_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Water) saturation distribution in the reservoir (fraction)."""
    water_viscosity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Water) viscosity distribution in the reservoir in (cP)."""
    water_compressibility_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water compressibility distribution (on bulk volume basis) in the reservoir (psi⁻¹)."""
    water_density_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water density distribution in the reservoir (lbm/ft³)."""
    gas_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Gas) saturation distribution in the reservoir (fraction)."""
    gas_viscosity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Gas) viscosity distribution in the reservoir in (cP)."""
    gas_compressibility_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas compressibility distribution (on bulk volume basis) in the reservoir (psi⁻¹)."""
    gas_compressibility_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas compressibility factor distribution in the reservoir (dimensionless)."""
    gas_gravity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas gravity distribution in the reservoir (dimensionless). Should be constant for a given gas, e.g., Methane = 0.556)."""
    gas_molecular_weight_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas molecular weight distribution in the reservoir (g/mol). Should be constant for a given gas, e.g., Methane = 16.04 g/mol)."""
    gas_density_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas density distribution in the reservoir (lbm/ft³)."""
    solution_gas_to_oil_ratio_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the solution gas-to-oil ratio distribution at standard conditions (SCF/STB)."""
    gas_solubility_in_water_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas solubility in water distribution at standard conditions (SCF/STB)."""
    oil_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil formation volume factor distribution (bbl/STB)."""
    gas_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas formation volume factor distribution (ft³/SCF)."""
    water_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water formation volume factor distribution (bbl/STB)."""
    water_salinity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water salinity distribution (ppm NaCl). Should be constant for a given water type, e.g., seawater = 35,000 ppm NaCl)."""
    solvent_concentration_grid: NDimensionalGrid[NDimension]
    """
    Solvent volume concentration in oil phase (0=pure oil, 1=pure solvent)

    This given by solvent volume in oil phase / (solvent volume in oil phase + oil volume in oil phase) (ft³ of solvent per ft³ of oil-solvent mixture)
    """
    oil_effective_viscosity_grid: NDimensionalGrid[NDimension]
    """
    Effective oil-solvent mixture viscosity using miscible model (e.g Todd Longstaff) (cP).

    This will be same as `oil_viscosity_grid` for immiscible flow.
    """
    oil_effective_density_grid: NDimensionalGrid[NDimension]
    """
    Effective oil-solvent mixture density using miscible model (lbm/ft³).

    This will be same as `oil_density_grid` for immiscible flow.
    """
    oil_mass_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil mass distribution in the reservoir (lbm)."""
    water_mass_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water mass distribution in the reservoir (lbm)."""
    free_gas_mass_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the free gas mass distribution in the reservoir (lbm)."""
    dissolved_gas_mass_in_oil_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas mass distribution in the oil phase (lbm)."""
    dissolved_gas_mass_in_water_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas mass distribution in the water phase (lbm)."""
    reservoir_gas: str = "methane"
    """Name of the reservoir gas (e.g., Methane, Ethane, CO2, N2). Can also be the name of the gas injected into the reservoir."""

    @property
    def total_gas_mass_grid(self) -> NDimensionalGrid[NDimension]:
        """N-dimensional numpy array representing the total gas mass distribution in the reservoir (lbm)."""
        return typing.cast(
            NDimensionalGrid[NDimension],
            self.free_gas_mass_grid
            + self.dissolved_gas_mass_in_oil_grid
            + self.dissolved_gas_mass_in_water_grid,
        )


@typing.final
@attrs.frozen(slots=True)
class RockPermeability(StoreSerializable, typing.Generic[NDimension]):
    """
    Rock permeability in the reservoir, in milliDarcy (mD).

    Permeability can be anisotropic, meaning it can vary in different directions (x, y, z).
    If only the x-direction permeability is provided, it is assumed that the y and z directions have the same permeability (isotropic).
    """

    x: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the permeability distribution in the x-direction (mD)."""
    y: NDimensionalGrid[NDimension] = attrs.field(
        factory=lambda: np.empty((0, 0), dtype=get_dtype())
    )  # type: ignore[assignment]
    """N-dimensional numpy array representing the permeability distribution in the y-direction (mD)."""
    z: NDimensionalGrid[NDimension] = attrs.field(
        factory=lambda: np.empty((0, 0), dtype=get_dtype())
    )  # type: ignore[assignment]
    """N-dimensional numpy array representing the permeability distribution in the z-direction (mD)."""
    mean: NDimensionalGrid[NDimension] = attrs.field(
        factory=lambda: np.empty((0, 0), dtype=get_dtype())
    )  # type: ignore[assignment]
    """N-dimensional numpy array representing the mean (geometric by default) of permeability distribution (mD)."""

    def __attrs_post_init__(self) -> None:
        if self.y.size == 0:
            object.__setattr__(self, "y", self.x)
        if self.z.size == 0:
            object.__setattr__(self, "z", self.x)

        if self.mean.size == 0:
            isotropic = self.y.size == 0 and self.z.size == 0
            if isotropic:
                object.__setattr__(self, "mean", self.x)
            else:  # anisotropic
                mean = (self.x * self.y * self.z) ** 1 / 3
                object.__setattr__(self, "mean", mean)


@typing.final
@attrs.frozen(slots=True)
class RockProperties(StoreSerializable, typing.Generic[NDimension]):
    """
    Rock properties of a reservoir model.

    These properties remain constant over time.
    """

    compressibility: float
    """Reservoir rock compressibility in (psi⁻¹)"""
    absolute_permeability: RockPermeability[NDimension]
    """Rock permeability in the reservoir, in milliDarcy (mD)."""
    net_to_gross_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the net-to-gross ratio distribution across the reservoir rock (fraction)."""
    porosity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the porosity distribution across the reservoir rock (fraction)."""
    connate_water_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the connate water saturation distribution (fraction)."""
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the irreducible water saturation distribution (fraction). This assumes imbibition process."""
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the residual oil saturation distribution during water flooding (fraction). This assumes imbibition process."""
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the residual oil saturation distribution during gas flooding (fraction). This assumes imbibition process."""
    residual_gas_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the residual gas saturation distribution (fraction). This assumes imbibition process."""


@typing.final
@attrs.frozen(slots=True)
class HysteresisState(StoreSerializable, typing.Generic[NDimension]):
    """
    Tracks hysteresis state for drainage-imbibition effects in multi-phase flow.

    This class maintains the information required for Killough scanning curve hysteresis modeling:
    - Historical maximum saturations (water and gas) to determine current displacement regime
    - Imbibition flags indicating whether each fluid is currently displacing or being displaced
    - Reversal saturation points for accurate scanning curve behavior

    The data is used to compute effective residual saturations that vary based on whether the system
    is in drainage or imbibition, which is critical for accurate relative permeability and capillary pressure calculations.
    """

    max_water_saturation_grid: NDimensionalGrid[NDimension]
    """Maximum water saturation reached (historical)"""
    max_gas_saturation_grid: NDimensionalGrid[NDimension]
    """Maximum gas saturation reached (historical)"""

    # Flags to track current displacement regime
    water_imbibition_flag_grid: np.ndarray[NDimension, np.dtype[np.bool]]
    """Flag grid indicating if the current water displacement is imbibition (True) or drainage (False)"""
    gas_imbibition_flag_grid: np.ndarray[NDimension, np.dtype[np.bool]]
    """Flag grid indicating if the current gas displacement is imbibition (True) or drainage (False)"""

    # Grids to track reversal points for Killough scanning curves
    water_reversal_saturation_grid: NDimensionalGrid[NDimension]
    """Water saturation at the last reversal point (drainage to imbibition or vice versa) for Killough scanning curves"""
    gas_reversal_saturation_grid: NDimensionalGrid[NDimension]
    """Gas saturation at the last reversal point (drainage to imbibition or vice versa) for Killough scanning curves"""

    @classmethod
    def from_initial_saturations(
        cls,
        water_saturation_grid: NDimensionalGrid[NDimension],
        gas_saturation_grid: NDimensionalGrid[NDimension],
    ) -> Self:
        """
        Create a `HysteresisState` instance from initial water and gas saturation grids.

        :param water_saturation_grid: N-dimensional numpy array representing the initial water saturation distribution in the reservoir (fraction).
        :param gas_saturation_grid: N-dimensional numpy array representing the initial gas saturation distribution in the reservoir (fraction).
        :return: `HysteresisState` instance initialized with the provided saturation grids.
        """
        water_imbibition_flag_grid = np.zeros_like(water_saturation_grid, dtype=bool)
        gas_imbibition_flag_grid = np.zeros_like(gas_saturation_grid, dtype=bool)
        return cls(
            max_water_saturation_grid=water_saturation_grid,
            max_gas_saturation_grid=gas_saturation_grid,
            water_imbibition_flag_grid=water_imbibition_flag_grid,  # type: ignore[arg-type]
            gas_imbibition_flag_grid=gas_imbibition_flag_grid,  # type: ignore[arg-type]
            water_reversal_saturation_grid=water_saturation_grid.copy(),
            gas_reversal_saturation_grid=gas_saturation_grid.copy(),
        )


class ReservoirModel(
    StoreSerializable,
    typing.Generic[NDimension],
    fields={
        "grid_shape": tuple,
        "cell_dimension": tuple,
        "thickness_grid": np.ndarray,
        "fluid_properties": FluidProperties,
        "rock_properties": RockProperties,
        "hysteresis_state": HysteresisState,
        "face_transmissibilities": typing.Optional[FaceTransmissibilities],
        "dip_angle": float,
        "dip_azimuth": float,
        "datum_depth": typing.Optional[float],
    },
):
    """Models a reservoir in N-dimensional space for simulation."""

    def __init__(
        self,
        grid_shape: NDimension,
        cell_dimension: typing.Tuple[float, float],
        thickness_grid: NDimensionalGrid[NDimension],
        fluid_properties: FluidProperties[NDimension],
        rock_properties: RockProperties[NDimension],
        hysteresis_state: HysteresisState[NDimension],
        face_transmissibilities: typing.Optional[FaceTransmissibilities] = None,
        dip_angle: float = 0.0,
        dip_azimuth: float = 0.0,
        datum_depth: typing.Optional[float] = 0.0,
        pore_volume_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    ) -> None:
        """
        Initialize the reservoir model.

        :param grid_shape: Shape of the reservoir grid (num_cells_x, num_cells_y, num_cells_z)
        :param cell_dimension: Size of each cell in the grid (cell_size_x, cell_size_y) in ft
        :param thickness_grid: N-dimensional numpy array representing the thickness of each cell in the reservoir (ft)
        :param fluid_properties: Fluid properties for fluid properties
        :param rock_fluid_properties: Rock-fluid properties
        :param hysteresis_state: `HysteresisState` instance tracking historical saturation extrema and displacement regimes for hysteresis effects
        :param face_transmissibilities: Optional precomputed face transmissibilities. If not provided,
            they will be computed from the permeability and grid properties when accessed.
        :param dip_angle: Dip angle of the reservoir in degrees (0 = horizontal, 90 = vertical)
        :param dip_azimuth: Dip azimuth of the reservoir in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        :param datum_depth: Reference depth for reservoir model. Basically the reservoir top depth (below sea level)
        """
        if not (0.0 <= dip_angle <= 90.0):
            raise ValidationError(
                f"`dip_angle` must be between 0.0 and 90.0, got {dip_angle}"
            )
        if not (0.0 <= dip_azimuth < 360.0):
            raise ValidationError(
                f"`dip_azimuth` must be between 0.0 and 360.0, got {dip_azimuth}"
            )

        if datum_depth is not None and datum_depth < 0:
            raise ValidationError("`datum_depth` cannot be assigned a negative value")

        self.grid_shape = grid_shape
        self.cell_dimension = cell_dimension
        self.thickness_grid = thickness_grid
        self.fluid_properties = fluid_properties
        self.rock_properties = rock_properties
        self.hysteresis_state = hysteresis_state
        self.face_transmissibilities = face_transmissibilities
        self.dip_angle = dip_angle
        self.dip_azimuth = dip_azimuth
        self.datum_depth = datum_depth
        if pore_volume_grid is not None:
            self.pore_volume_grid = pore_volume_grid
        else:
            self.pore_volume_grid = (
                self.rock_properties.porosity_grid
                * self.rock_properties.net_to_gross_grid
                * self.thickness_grid
                * self.cell_dimension[0]
                * self.cell_dimension[1]
            )

    @property
    def dimensions(self) -> int:
        """Return the number of dimensions of the reservoir model."""
        return len(self.grid_shape)

    @property
    def volume(self) -> float:
        """Return the total volume of the reservoir model."""
        return (
            np.prod(self.grid_shape)
            * np.prod(self.cell_dimension)
            * self.thickness_grid.sum()
        )

    def evolve(self, **kwargs: typing.Any) -> Self:
        """
        Create a new `ReservoirModel` instance with updated attributes.

        :param kwargs: Attributes to update in the new reservoir model.
        :return: New `ReservoirModel` instance with updated attributes.
        """
        attrs = {
            "grid_shape": self.grid_shape,
            "cell_dimension": self.cell_dimension,
            "thickness_grid": self.thickness_grid,
            "fluid_properties": self.fluid_properties,
            "rock_properties": self.rock_properties,
            "hysteresis_state": self.hysteresis_state,
            "face_transmissibilities": self.face_transmissibilities,
            "dip_angle": self.dip_angle,
            "dip_azimuth": self.dip_azimuth,
        }
        return type(self)(**{**attrs, **kwargs})  # type: ignore[arg-type]

    def build_elevation_grid(
        self, apply_dip: bool = False
    ) -> NDimensionalGrid[NDimension]:
        """
        Build an elevation grid of the reservoir cells.

        The elevation grid is generated based on the thickness of each cell, starting from the base elevation (0 ft).

        :param apply_dip: If True, applies the reservoir dip angle and direction to create
            a tilted elevation grid. If False, generates a flat (horizontal) elevation grid.
        :return: N-dimensional numpy array representing the elevation of each cell in the reservoir (ft).

        Example:
        ```python
        # Flat reservoir
        elevation = model.build_elevation_grid(apply_dip=False)

        # Dipping reservoir (5° toward North)
        model = ReservoirModel(dip_angle=5.0, dip_direction="N", ...)
        elevation = model.build_elevation_grid(apply_dip=True)
        ```
        """
        if self.datum_depth is None:
            datum_elevation = 0.0
        else:
            # `datum_depth` is the depth of the top of the reservoir (positive)
            # We need the elevation of the bottom of the reservoir (negative)
            # Bottom elevation = -(top_depth + total_thickness)
            total_thickness = np.max(np.sum(self.thickness_grid, axis=2))
            datum_elevation = -(self.datum_depth + total_thickness)

        base_elevation_grid = build_elevation_grid(
            self.thickness_grid, datum=datum_elevation
        )
        # If no dip is requested or dip angle is zero, return flat grid
        if not apply_dip or self.dip_angle == 0.0:
            return base_elevation_grid

        return apply_structural_dip(
            elevation_grid=base_elevation_grid,
            cell_dimension=self.cell_dimension,
            elevation_direction="upward",
            dip_angle=self.dip_angle,
            dip_azimuth=self.dip_azimuth,
        )

    def build_depth_grid(self, apply_dip: bool = False) -> NDimensionalGrid[NDimension]:
        """
        Build a depth grid of the reservoir cells.

        The depth grid is generated based on the thickness of each cell, starting from the surface (0 ft).

        :param apply_dip: If True, applies the reservoir dip angle and direction to create
            a tilted depth grid. If False, generates a flat (horizontal) depth grid.
        :return: N-dimensional numpy array representing the depth of each cell in the reservoir (ft).

        Example:

        ```python
        # Flat reservoir
        depth = model.build_depth_grid(apply_dip=False)

        # Dipping reservoir (5° toward North)
        model = ReservoirModel(dip_angle=5.0, ...)
        depth = model.build_depth_grid(apply_dip=True)
        ```
        """
        base_depth_grid = build_depth_grid(
            self.thickness_grid, datum=self.datum_depth or 0.0
        )
        # If no dip is requested or dip angle is zero, return flat grid
        if not apply_dip or self.dip_angle == 0.0:
            return base_depth_grid

        return apply_structural_dip(
            elevation_grid=base_depth_grid,
            cell_dimension=self.cell_dimension,
            elevation_direction="downward",
            dip_angle=self.dip_angle,
            dip_azimuth=self.dip_azimuth,
        )

    def build_face_transmissibilities(
        self, dtype: typing.Optional[npt.DTypeLike] = None
    ) -> FaceTransmissibilities:
        """
        Retrieve or build face transmissibilities for the reservoir model.

        If face transmissibilities were provided during initialization, they are returned.
        Otherwise, they are computed from the rock properties and grid parameters,
        cached in the model, and then returned.

        :return: `FaceTransmissibilities` object containing x, y, z face transmissibilities.
        """
        if self.face_transmissibilities is not None:
            return self.face_transmissibilities

        assert len(self.grid_shape) == 3
        face_transmissibilities = build_face_transmissibilities(
            permeability_x=self.rock_properties.absolute_permeability.x,  # type: ignore[arg-type]
            permeability_y=self.rock_properties.absolute_permeability.y,  # type: ignore[arg-type]
            permeability_z=self.rock_properties.absolute_permeability.z,  # type: ignore[arg-type]
            thickness_grid=self.thickness_grid,  # type: ignore[arg-type]
            net_to_gross_grid=self.rock_properties.net_to_gross_grid,  # type: ignore[arg-type]
            cell_size_x=self.cell_dimension[0],
            cell_size_y=self.cell_dimension[1],
            dtype=dtype,
        )
        # Set the computed face transmissibilities on the model
        self.face_transmissibilities = face_transmissibilities
        return self.face_transmissibilities
