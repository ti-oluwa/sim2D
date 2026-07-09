"""Per-timestep dynamic well state.

Holds no `ReservoirState` reference and no `WellSpec` reference - only a
plain `well_name` string key - so a `WellState` stays cheap to keep around
across a whole simulation timeline. `ConnectionSample` is the only thing
this module needs the reservoir side to produce.
"""

import typing

import attrs

from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import FluidPhase, Number
from bores.wells.controls import ControlSpec, Limit
from bores.wells.data import Perforation

__all__ = ["ConnectionSample", "PerforationState", "WellState"]


def _as_phase_mapping(
    mapping: typing.Mapping[typing.Any, Number],
) -> typing.Dict[FluidPhase, Number]:
    """
    Normalizes a phase-keyed mapping to `FluidPhase` keys.

    `Serializable`'s generic deserializer serializes `Mapping` keys through
    their declared type but doesn't convert them back on load, so a
    dump()/load() round trip otherwise leaves these keys as plain strings.
    Accepts either a `FluidPhase` or its string value as input.
    """
    return {
        (key if isinstance(key, FluidPhase) else FluidPhase(key)): value
        for key, value in mapping.items()
    }


@attrs.frozen(kw_only=True, slots=True)
class ConnectionSample(Serializable):
    """Reservoir conditions at one connected cell, for use by
    `wells.index`, `wells.hydraulics`, and `wells.control_engine`.
    """

    cell_index: int
    pressure: Number
    temperature: Number
    phase_saturations: typing.Mapping[FluidPhase, Number] = attrs.field(
        converter=_as_phase_mapping
    )
    phase_mobilities: typing.Mapping[FluidPhase, Number] = attrs.field(
        converter=_as_phase_mapping
    )

    def __attrs_post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValidationError(f"`cell_index` must be >= 0; got {self.cell_index}.")


@attrs.frozen(kw_only=True, slots=True)
class PerforationState(Serializable):
    """Per-perforation dynamic snapshot for one timestep."""

    perforation: Perforation
    cell_index: int
    flowing_pressure: Number
    phase_rates: typing.Mapping[FluidPhase, Number] = attrs.field(
        converter=_as_phase_mapping
    )

    def __attrs_post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValidationError(f"`cell_index` must be >= 0; got {self.cell_index}.")


@attrs.frozen(kw_only=True, slots=True)
class WellState(Serializable):
    """Per-well dynamic snapshot for one timestep."""

    well_name: str
    is_open: bool
    active_control: ControlSpec
    bhp: Number
    perforation_states: typing.Tuple[PerforationState, ...] = attrs.field(
        converter=tuple
    )
    phase_rates: typing.Mapping[FluidPhase, Number] = attrs.field(
        converter=_as_phase_mapping
    )
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
