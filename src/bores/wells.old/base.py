"""Well implementations and base classes."""

import functools
import itertools
import logging
import threading
import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.blackoil.tables.pvt import PVTTables
from bores.errors import ValidationError
from bores.serde.registry import (
    make_registry_deserializer,
    make_registry_serializer,
    make_serializable_type_registrar,
    register_type_deserializer,
    register_type_serializer,
)
from bores.serde.stores import StoreSerializable
from bores.typing import (
    Coordinates,
    FluidPhase,
    Orientation,
    ThreeDimensions,
    TwoDimensions,
)
from bores.wells.controls import ControlInfo, WellControl
from bores.wells.core import (
    InjectedFluid,
    ProducedFluid,
    WellFluidT,
    compute_2D_effective_drainage_radius,
    compute_3D_effective_drainage_radius,
    compute_effective_permeability_for_well,
    compute_well_index,
)

logger = logging.getLogger(__name__)

__all__ = ["InjectionWell", "ProductionWell", "Well", "Wells", "well_type"]


@attrs.mutable(hash=True)
class Well(StoreSerializable, typing.Generic[Coordinates, WellFluidT]):
    """Models a well in the reservoir model."""

    name: str
    """Name of the well."""
    perforating_intervals: typing.Sequence[tuple[Coordinates, Coordinates]]
    """Perforating intervals of the well. Each interval is a tuple of (start_location, end_location)."""
    radius: float
    """Radius of the wellbore (ft)."""
    control: WellControl[WellFluidT]
    """Control strategy for the well (e.g., rate control, pressure control)."""
    skin_factor: float = 0.0
    """Skin factor for the well, affecting flow performance."""
    orientation: Orientation = attrs.field(default=Orientation.UNSET, converter=Orientation)
    """Orientation of the well, indicating its dominant direction in the reservoir grid."""
    is_active: bool = True
    """Indicates whether the well is active or not. Set to False if the well is shut in or inactive."""

    def __attrs_post_init__(self) -> None:
        """Ensure the well has a valid orientation."""
        if self.orientation == Orientation.UNSET:
            self.orientation = self.get_orientation()

    @property
    def is_shut_in(self) -> bool:
        """Check if the well is shut in."""
        return not self.is_active

    @property
    def is_open(self) -> bool:
        """Check if the well is open."""
        return self.is_active

    def get_orientation(self) -> Orientation:
        """
        Determine the dominant orientation of the well by computing the total
        length along each axis across all perforating intervals.

        For wells with multiple intervals (including multilateral or complex wells),
        this finds the axis with the greatest cumulative length and returns that
        as the dominant orientation.

        :returns: The dominant orientation of the well (X, Y, or Z)
        """
        if not self.perforating_intervals:
            return Orientation.Z  # Default to Z if no intervals

        # Sum up absolute direction components across all intervals
        total_length = np.array([0.0, 0.0, 0.0])

        for start, end in self.perforating_intervals:
            # Convert to numpy arrays and pad to 3D if needed
            start_3d = np.array(start + (0,) * (3 - len(start)))
            end_3d = np.array(end + (0,) * (3 - len(end)))

            # Accumulate absolute length along each axis
            direction = np.abs(end_3d - start_3d)
            total_length += direction

        # Find axis with maximum cumulative length
        if np.all(total_length == 0):
            return Orientation.Z  # Default if all intervals are zero-length

        axis = np.argmax(total_length)
        return Orientation(("x", "y", "z")[axis])

    def check_location(self, grid_shape: tuple[int, ...]) -> None:
        """
        Check if the well's perforating intervals are within the grid dimensions.

        :param grid_shape: The shape of the reservoir grid (nx, ny, nz).
        :raises ValidationError: If any of the well's perforating intervals are out of bounds.
        """
        for interval_idx, (start, end) in enumerate(self.perforating_intervals):
            if not all(0 <= coord < dim for coord, dim in zip(start, grid_shape, strict=False)):
                raise ValidationError(
                    f"Start location {start} for interval {interval_idx} of well {self.name!r} is out of bounds."
                )
            if not all(0 <= coord < dim for coord, dim in zip(end, grid_shape, strict=False)):
                raise ValidationError(
                    f"End location {end} for interval {interval_idx} of well {self.name!r} is out of bounds."
                )

    def get_effective_drainage_radius(
        self,
        interval_thickness: tuple[float, ...],
        permeability: tuple[float, ...],
    ) -> float:
        """
        Compute the effective drainage radius for the well based on its orientation,
        with corrections for wells at no-flow boundaries.

        :param interval_thickness: A tuple representing the thickness of the reservoir interval in each direction (ft).
        :param permeability: A tuple representing the permeability of the reservoir rock in each direction (mD).
        :param well_location: Grid indices (i, j, k) of the well perforation. Required for boundary correction.
        :param grid_shape: Grid dimensions (nx, ny, nz). Required for boundary correction.
        :return: The effective drainage radius in the direction of the well (ft).
        """
        dimensions = len(interval_thickness)
        if dimensions < 2 or dimensions > 3:
            raise ValidationError("2D/3D locations are required")

        if dimensions == 2:
            if len(permeability) != 2:
                raise ValidationError("Permeability must be a 2D tuple for 2D locations")
            interval_thickness = typing.cast(TwoDimensions, interval_thickness)
            permeability = typing.cast(TwoDimensions, permeability)
            return compute_2D_effective_drainage_radius(
                interval_thickness=interval_thickness,
                permeability=permeability,
            )

        if len(permeability) != 3:
            raise ValidationError("Permeability must be a 3D tuple for 3D locations")

        interval_thickness = typing.cast(ThreeDimensions, interval_thickness)
        permeability = typing.cast(ThreeDimensions, permeability)
        return compute_3D_effective_drainage_radius(
            interval_thickness=interval_thickness,
            permeability=permeability,
            well_orientation=self.orientation,
        )

    def get_well_index(
        self,
        interval_thickness: tuple[float, ...],
        permeability: tuple[float, ...],
        skin_factor: float | None = None,
        net_to_gross: float = 1.0,
        regime_constant: float = -3 / 4,
    ) -> float:
        """
        Compute the well index for the well using the Peaceman equation.

        :param interval_thickness: A tuple representing the thickness of the reservoir interval in each direction (ft).
        :param permeability: A tuple representing the permeability of the reservoir rock in each direction (mD).
        :param skin_factor: Well skin factor (dimensionless). If None, uses self.skin_factor.
        :param net_to_gross: Net-to-gross ratio of the reservoir interval (default is 1.0).
            Reduces the effective thickness of the reservoir contributing to flow into the well.
        :param well_location: Grid indices (i, j, k) of the well. Required for boundary correction.
        :param grid_shape: Grid dimensions (nx, ny, nz). Required for boundary correction.
        :param regime_constant: The flow regime constant. 0 for steady, -3/4 for pseudo steady, 1/2 for transient regime
        :return: The well index (md*ft).
        """
        dimensions = len(interval_thickness)
        if dimensions < 2 or dimensions > 3:
            raise ValidationError("2D/3D locations are required")

        orientation = self.orientation
        effective_drainage_radius = self.get_effective_drainage_radius(
            interval_thickness=interval_thickness, permeability=permeability
        )
        skin_factor = skin_factor if skin_factor is not None else self.skin_factor
        radius = self.radius
        effective_permeability = compute_effective_permeability_for_well(
            permeability=permeability, orientation=orientation
        )

        if orientation == Orientation.X:
            directional_thickness = interval_thickness[0]
        elif orientation == Orientation.Y:
            directional_thickness = interval_thickness[1]
        elif dimensions == 3 and orientation == Orientation.Z:
            directional_thickness = interval_thickness[2]
        else:  # dimensions == 2 and orientation == Orientation.Z:
            raise ValidationError("Z-oriented wells are not supported in 2D models")

        return compute_well_index(
            permeability=effective_permeability,
            interval_thickness=directional_thickness,
            wellbore_radius=radius,
            effective_drainage_radius=effective_drainage_radius,
            skin_factor=skin_factor,
            net_to_gross=net_to_gross,
            regime_constant=regime_constant,
        )

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidT,
        phase_fvf: float,
        phase_mobility: float,
        phase_viscosity: float | None = None,
        allocation_fraction: float = 1.0,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: float | None = None,
        pvt_tables: PVTTables | None = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute the flow rate for the well using the configured control strategy.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_viscosity: The viscosity of the fluid phase being produced or injected.
        :param phase_mobility: The relative mobility of the fluid phase being produced or injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected.
        :param phase_fvf: Formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of target rate to allocate to this cell (for multi-cell wells).
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :param kwargs: Additional control-specific context (e.g., primary phase properties for ProducerRateControl).
        :return: The flow rate in (bbl/day or ft³/day).
        """
        return self.control.get_flow_rate(
            pressure=pressure,
            temperature=temperature,
            phase_viscosity=phase_viscosity,
            phase_mobility=phase_mobility,
            well_index=well_index,
            fluid=fluid,
            phase_fvf=phase_fvf,
            allocation_fraction=allocation_fraction,
            is_active=self.is_open,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            pvt_tables=pvt_tables,
            **kwargs,
        )

    def get_bottom_hole_pressure(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidT,
        phase_fvf: float,
        phase_mobility: float,
        phase_viscosity: float | None = None,
        allocation_fraction: float = 1.0,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: float | None = None,
        pvt_tables: PVTTables | None = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute the bottom-hole pressure for the well using the configured control strategy.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_viscosity: The viscosity of the fluid phase being produced or injected.
        :param phase_mobility: The relative mobility of the fluid phase being produced or injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected.
        :param phase_fvf: Formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of target rate to allocate to this cell (for multi-cell wells).
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :param kwargs: Additional control-specific context (e.g., primary phase properties for ProducerRateControl).
        :return: The bottom-hole pressure (psi).
        """
        return self.control.get_bottom_hole_pressure(
            pressure=pressure,
            temperature=temperature,
            phase_viscosity=phase_viscosity,
            phase_mobility=phase_mobility,
            well_index=well_index,
            fluid=fluid,
            phase_fvf=phase_fvf,
            allocation_fraction=allocation_fraction,
            is_active=self.is_open,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            pvt_tables=pvt_tables,
            **kwargs,
        )

    def get_control(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidT,
        phase_fvf: float,
        phase_mobility: float,
        phase_viscosity: float | None = None,
        allocation_fraction: float = 1.0,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: float | None = None,
        pvt_tables: PVTTables | None = None,
        **kwargs: typing.Any,
    ) -> ControlInfo:
        """
        Compute both the flow rate and effective bottom-hole pressure for the well in a single pass,
        using the configured control strategy.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_viscosity: The viscosity of the fluid phase being produced or injected.
        :param phase_mobility: The relative mobility of the fluid phase being produced or injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected.
        :param phase_fvf: Formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of target rate to allocate to this cell (for multi-cell wells).
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :param kwargs: Additional control-specific context (e.g., primary phase properties for ProducerRateControl).
        :return: `ControlInfo` containing the flow rate (bbl/day or
            ft³/day) and effective BHP (psi).
        """
        return self.control.get_control(
            pressure=pressure,
            temperature=temperature,
            phase_viscosity=phase_viscosity,
            phase_mobility=phase_mobility,
            well_index=well_index,
            fluid=fluid,
            phase_fvf=phase_fvf,
            allocation_fraction=allocation_fraction,
            is_active=self.is_open,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            pvt_tables=pvt_tables,
            **kwargs,
        )

    def shut_in(self) -> None:
        """Shut in the well."""
        self.is_active = False

    def open(self) -> None:
        """Open the well."""
        self.is_active = True

    def duplicate(self: Self, *, name: str | None = None, **kwargs) -> Self:
        """
        Create a duplicate of the well with an optional new name.

        :param name: The name for the duplicated well. If None, uses the original well's name.
        :kwargs: Additional properties to override in the duplicated well.
        :return: A new instance of the well with the same properties.
        """
        return attrs.evolve(self, name=name or self.name, **kwargs)


WellT = typing.TypeVar("WellT", bound=Well)


_WELL_TYPES = {}
"""Registry for supported well types."""
well_type = make_serializable_type_registrar(
    base_cls=Well,
    registry=_WELL_TYPES,
    lock=threading.Lock(),
    key_attr="__type__",
    override=False,
    # Do not register serializers/deserializers for the base `Well` class yet
    auto_register_serializer=False,
    auto_register_deserializer=False,
)
"""Decorator to register a new well type."""

# Build and register serializers/deserializers for `Well` base class
serialize_well = make_registry_serializer(
    base_cls=Well,
    registry=_WELL_TYPES,
    key_attr="__type__",
)
register_type_serializer(
    typ=Well,
    serializer=serialize_well,
)
deserialize_well = make_registry_deserializer(
    base_cls=Well,
    registry=_WELL_TYPES,
)
register_type_deserializer(
    typ=Well,
    deserializer=deserialize_well,
)


@typing.final
@well_type
@attrs.mutable(hash=True)
class InjectionWell(Well[Coordinates, InjectedFluid]):
    """
    Models an injection well in the reservoir model.

    This well injects fluids into the reservoir.
    """

    __type__ = "injection_well"

    injected_fluid: InjectedFluid | None = None
    """Properties of the fluid being injected into the well."""

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: InjectedFluid,
        phase_fvf: float,
        phase_mobility: float,
        phase_viscosity: float | None = None,
        allocation_fraction: float = 1.0,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: float | None = None,
        pvt_tables: PVTTables | None = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute the flow rate for the injection well using the configured control strategy.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_viscosity: The viscosity of the fluid phase being injected.
        :param phase_mobility: The relative mobility of the fluid phase being injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being injected into the well. If None, uses the well's injected_fluid property.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹). For slightly compressible fluids, this can be used to adjust the flow rate calculation.
        :param phase_fvf: The formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: The flow rate (bbl/day or ft³/day)
        """
        return super().get_flow_rate(
            pressure=pressure,
            temperature=temperature,
            phase_viscosity=phase_viscosity,
            phase_mobility=phase_mobility,
            well_index=well_index,
            fluid=fluid,
            phase_fvf=phase_fvf,
            allocation_fraction=allocation_fraction,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            pvt_tables=pvt_tables,
            **kwargs,
        )


@typing.final
@well_type
@attrs.mutable(hash=True)
class ProductionWell(Well[Coordinates, ProducedFluid]):
    """
    Models a production well in the reservoir model.

    This well produces fluids from the reservoir.
    """

    __type__ = "production_well"

    produced_fluids: typing.Sequence[ProducedFluid] = attrs.field(factory=list)
    """List of fluids produced by the well. This can include multiple phases (e.g., oil, gas, water)."""

    def __attrs_post_init__(self) -> None:
        # Ensure the only unique phases are stored in the `produced_fluids` list
        unique_phases = []
        for fluid in self.produced_fluids:
            if fluid.phase not in unique_phases:
                unique_phases.append(fluid.phase)
            else:
                raise ValidationError(
                    "Duplicate fluid phase found in `produced_fluids`. Each phase should only be listed once."
                )
        super().__attrs_post_init__()

    @functools.cached_property
    def produced_phases(self) -> list[FluidPhase]:
        """List of unique fluid phases produced by the well (e.g., ['oil', 'gas', 'water'])."""
        return [fluid.phase for fluid in self.produced_fluids]  # type: ignore[return-value]


InjectionWellT = typing.TypeVar("InjectionWellT", bound=InjectionWell)
ProductionWellT = typing.TypeVar("ProductionWellT", bound=ProductionWell)


def _expand_interval(
    interval: tuple[Coordinates, Coordinates], orientation: Orientation
) -> list[Coordinates]:
    """Expand a well perforating interval into a list of grid locations."""
    start, end = interval
    dimensions = len(start)
    if dimensions < 2:
        raise ValidationError("2D/3D locations are required")

    # Normalize start and end to ensure ranges are valid regardless of order
    start = tuple(min(s, e) for s, e in zip(start, end, strict=False))
    end = tuple(max(s, e) for s, e in zip(start, end, strict=False))

    if dimensions == 2:
        start = start + (0,)
        end = end + (0,)
        dimensions = 3  # Pad to 3D for uniform logic

    # Create iterator for the correct orientation
    if orientation == Orientation.X:
        locations = list(
            itertools.product(
                range(start[0], end[0] + 1),
                [start[1]],
                [start[2]],
            )
        )
    elif orientation == Orientation.Y:
        locations = list(
            itertools.product(
                [start[0]],
                range(start[1], end[1] + 1),
                [start[2]],
            )
        )
    elif orientation == Orientation.Z:
        locations = list(
            itertools.product(
                [start[0]],
                [start[1]],
                range(start[2], end[2] + 1),
            )
        )
    else:
        raise ValidationError(f"Invalid well orientation {orientation!r}")

    return typing.cast(list[Coordinates], locations)


def _expand_intervals(
    intervals: typing.Sequence[tuple[Coordinates, Coordinates]],
    orientation: Orientation,
) -> list[Coordinates]:
    """Expand multiple well perforating intervals into a list of grid locations."""
    locations = []
    for interval in intervals:
        locations.extend(_expand_interval(interval=interval, orientation=orientation))
    return locations


def _prepare_wells_map(
    wells: typing.Sequence[WellT],
) -> dict[tuple[int, ...], WellT]:
    """Prepare the wells map for quick access."""
    wells_map = {
        loc: well
        for well in wells
        for loc in _expand_intervals(
            intervals=well.perforating_intervals,
            orientation=well.orientation,
        )
    }
    return wells_map


@attrs.frozen
class _WellsProxy(typing.Generic[Coordinates, WellT]):
    """A proxy class for quick access to wells by their location."""

    wells: typing.Sequence[WellT]
    """A map of well perforating intervals to the well objects."""

    wells_map: dict[Coordinates, WellT] = attrs.field(init=False)
    """A map to store wells by their location for quick access."""
    allow_interval_overlap: bool = True
    """
    Whether to allow overlapping perforating intervals between wells.

    You can disable this if you are certain there are no overlapping wells or
    you want to allow overlapping wells (e.g in multi-layered reservoirs or multi-lateral wells).
    """

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "wells_map", _prepare_wells_map(self.wells))
        if self.allow_interval_overlap:
            return
        # Check for overlapping wells
        expected_location_count = sum(
            len(_expand_intervals(well.perforating_intervals, well.orientation))
            for well in self.wells
        )
        actual_location_count = len(self.wells_map)
        if expected_location_count != actual_location_count:
            raise ValidationError(
                f"Overlapping wells found at some locations. Expected {expected_location_count} unique locations, but got {actual_location_count}."
            )

    def __getitem__(self, location: Coordinates) -> WellT | None:
        """Get a well by its location."""
        return self.wells_map.get(location, None)

    def __setitem__(self, location: Coordinates, well: WellT) -> None:
        """Set a well at a specific location."""
        self.wells_map[location] = well

    def __iter__(self) -> typing.Iterator[tuple[Coordinates, WellT]]:
        """Iterate over well locations and their corresponding well objects."""
        return iter(self.wells_map.items())


# Serialize/deserialize list of wells as dictionaries of well name to well object
def _serialize_wells(
    wells: typing.Sequence[WellT], recurse: bool = True
) -> dict[str, dict[str, typing.Any]]:
    """Serialize a list of wells to a dictionary."""
    return {well.name: serialize_well(well, recurse) for well in wells}


def _deserialize_wells(
    data: dict[str, dict[str, typing.Any]],
) -> list[Well]:
    """Deserialize a dictionary of wells to a list."""
    return [deserialize_well(item) for item in data.values()]


_wells_serializers = {
    "injection_wells": _serialize_wells,
    "production_wells": _serialize_wells,
}
_wells_deserializers = {
    "injection_wells": _deserialize_wells,
    "production_wells": _deserialize_wells,
}


@typing.final
@attrs.frozen
class Wells(
    StoreSerializable,
    typing.Generic[Coordinates],
    fields={
        "injection_wells": typing.Sequence[InjectionWell],
        "production_wells": typing.Sequence[ProductionWell],
    },
    serializers=_wells_serializers,
    deserializers=_wells_deserializers,
):
    """
    Models a collection of injection and production wells in the reservoir model.

    This includes both production and injection wells.
    """

    injection_wells: typing.Sequence[InjectionWell[Coordinates]] = attrs.field(factory=list)
    """List of injection wells in the reservoir."""
    production_wells: typing.Sequence[ProductionWell[Coordinates]] = attrs.field(factory=list)
    """List of production wells in the reservoir."""
    injectors: _WellsProxy[Coordinates, InjectionWell[Coordinates]] = attrs.field(init=False)
    """
    Proxy for injection wells.

    This allows quick access to injection wells by their location.
    """
    producers: _WellsProxy[Coordinates, ProductionWell[Coordinates]] = attrs.field(init=False)
    """
    Proxy for production wells.

    This allows quick access to production wells by their location.
    """
    allow_interval_overlap: bool = True
    """
    Whether to allow overlapping perforating intervals between injection wells and/or production wells.

    You can disable this if you are certain there are no overlapping wells or
    you want to allow overlapping wells (e.g in multi-layered reservoirs or multi-lateral wells).
    """

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self,
            "injectors",
            _WellsProxy(
                wells=self.injection_wells,
                allow_interval_overlap=self.allow_interval_overlap,
            ),
        )
        object.__setattr__(
            self,
            "producers",
            _WellsProxy(
                wells=self.production_wells,
                allow_interval_overlap=self.allow_interval_overlap,
            ),
        )

        if self.allow_interval_overlap:
            # Check for overlapping wells. Injection and production wells should not overlap.
            overlapping_locations = set(self.injectors.wells_map).intersection(
                self.producers.wells_map
            )
            if overlapping_locations:
                raise ValidationError(
                    f"Overlapping wells found at locations: {overlapping_locations}"
                )

    def get_by_location(
        self, location: Coordinates
    ) -> tuple[
        InjectionWell[Coordinates] | None,
        ProductionWell[Coordinates] | None,
    ]:
        """
        Get wells by their grid coordinates.

        :param location: The (i, j) coordinates of the well in the reservoir grid.
        :return: Well or None: The well at the specified location, or None if not found.
        """
        return self.injectors[location], self.producers[location]

    def get_by_name(
        self, name: str
    ) -> tuple[
        InjectionWell[Coordinates] | None,
        ProductionWell[Coordinates] | None,
    ]:
        """
        Get wells by their name.

        :param name: The name of the well.
        :return: A tuple of (injection_well, production_well) or (None, None) if not found.
        """
        injection_well = next((well for well in self.injection_wells if well.name == name), None)
        production_well = next((well for well in self.production_wells if well.name == name), None)
        return injection_well, production_well

    def __getitem__(
        self, key: Coordinates | str, /
    ) -> tuple[
        InjectionWell[Coordinates] | None,
        ProductionWell[Coordinates] | None,
    ]:
        """
        Get a well by its grid coordinates.

        :param key: The (i, j, k) coordinates of the well in the reservoir grid or the name of the well.
        :return: Well or None: The well at the specified location, or None if not found.
        """
        if isinstance(key, str):
            return self.get_by_name(key)
        return self.get_by_location(key)

    def __iter__(
        self,
    ) -> typing.Iterator[InjectionWell[Coordinates] | ProductionWell[Coordinates]]:
        """Iterate over all wells in the collection (both injection and production wells)."""
        return iter(itertools.chain(self.injection_wells, self.production_wells))

    @property
    def locations(
        self,
    ) -> tuple[list[Coordinates], list[Coordinates]]:
        """
        Get the starting locations of all wells in the reservoir.

        :return: A tuple of (injection_well_locations, production_well_locations).
        This returns a tuple containing two lists:
            - A list of locations for injection wells (starting location of first interval).
            - A list of locations for production wells (starting location of first interval).
        """
        injection_well_heads = [
            well.perforating_intervals[0][0]
            for well in self.injection_wells
            if well.perforating_intervals
        ]
        production_well_heads = [
            well.perforating_intervals[0][0]
            for well in self.production_wells
            if well.perforating_intervals
        ]
        return injection_well_heads, production_well_heads

    @property
    def names(self) -> tuple[list[str], list[str]]:
        """
        Get all well names in the reservoir.

        :return: A tuple of (injection_well_names, production_well_names).
        This returns a tuple containing two lists:
            - A list of names for injection wells.
            - A list of names for production wells.
        """
        return (
            [well.name for well in self.injection_wells],
            [well.name for well in self.production_wells],
        )

    def check_location(self, grid_shape: tuple[int, ...]) -> None:
        """
        Check if all wells' perforating intervals are within the grid dimensions.

        :param grid_shape: The shape of the reservoir grid (nx, ny, nz).
        :raises ValidationError: If any well's perforating interval is out of bounds.
        """
        for well in itertools.chain(self.injection_wells, self.production_wells):
            well.check_location(grid_shape)

    def exists(self) -> bool:
        """
        Check if there are any wells in the reservoir model.

        :return: True if there are injection or production wells, False otherwise.
        """
        return bool(self.injection_wells or self.production_wells)

    def any_open(self) -> bool:
        return any(
            well.is_open for well in itertools.chain(self.injection_wells, self.production_wells)
        )
