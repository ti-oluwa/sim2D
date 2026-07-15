"""Per-timestep dynamic well state."""

import typing

import attrs

from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import FluidPhase, Number
from bores.wells.base import AnyPerforation
from bores.wells.controls import Limit, WellControl

__all__ = ["ConnectionSample", "PerforationState", "WellState"]


@attrs.frozen(kw_only=True, slots=True)
class ConnectionSample(Serializable):
    """Reservoir conditions at one connected cell"""

    cell_index: int
    pressure: Number
    temperature: Number
    phase_saturations: typing.Mapping[FluidPhase, Number]
    phase_mobilities: typing.Mapping[FluidPhase, Number]
    phase_densities: typing.Mapping[FluidPhase, Number]
    phase_viscosities: typing.Mapping[FluidPhase, Number]
    gas_liquid_surface_tension: Number

    def __attrs_post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValidationError(f"`cell_index` must be >= 0; got {self.cell_index}.")


@attrs.frozen(kw_only=True, slots=True)
class PerforationState(Serializable):
    """Per-perforation dynamic snapshot for one timestep."""

    perforation: AnyPerforation
    cell_index: int
    flowing_pressure: Number
    phase_rates: typing.Mapping[FluidPhase, Number]

    def __attrs_post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValidationError(f"`cell_index` must be >= 0; got {self.cell_index}.")


@attrs.frozen(kw_only=True, slots=True)
class WellState(Serializable):
    """Per-well dynamic snapshot for one timestep."""

    well_name: str
    is_open: bool
    active_control: WellControl
    bhp: Number
    perforation_states: typing.Tuple[PerforationState, ...]
    phase_rates: typing.Mapping[FluidPhase, Number]
    active_limit: typing.Optional[Limit] = None
    thp: typing.Optional[Number] = None

    def __attrs_post_init__(self) -> None:
        if not self.well_name:
            raise ValidationError("`well_name` must be a non-empty string.")
        if not self.is_open and self.perforation_states:
            raise ValidationError(
                f"WellState for {self.well_name!r} is shut (`is_open=False`) "
                "but has non-empty `perforation_states`; a shut well must "
                "have an empty tuple."
            )

    @property
    def total_liquid_rate(self) -> Number:
        """`phase_rates[OIL] + phase_rates[WATER]`, 0 for missing phases."""
        return self.phase_rates.get(FluidPhase.OIL, 0.0) + self.phase_rates.get(
            FluidPhase.WATER, 0.0
        )

    def perforation_state_at(self, cell_index: int) -> PerforationState:
        """
        Look up the `PerforationState` connected to a given cell.

        :param cell_index: Zero-based `Grid` cell index.
        :returns: Matching `PerforationState`.
        :raises KeyError: If no open perforation is connected to `cell_index`.
        """
        for state in self.perforation_states:
            if state.cell_index == cell_index:
                return state
        raise KeyError(
            f"No open perforation connected to cell_index={cell_index} for "
            f"well {self.well_name!r}."
        )


# TODO: Add `WellStates` class
