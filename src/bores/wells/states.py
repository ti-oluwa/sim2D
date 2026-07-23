"""Per-timestep dynamic well state."""

import typing

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.serde.stores import StoreSerializable
from bores.typing import FluidPhase, Number, UnitConversionTable, UnitSystem
from bores.wells.base import AnyPerforation
from bores.wells.controls import Limit, WellControl

__all__ = ["ConnectionSample", "PerforationState", "WellState", "WellsStates"]


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
    phase_formation_volume_factors: typing.Mapping[FluidPhase, Number] = attrs.field(
        factory=dict
    )
    """
    Reservoir volume per surface volume, per phase, at this
    connection's current pressure. A phase missing from this mapping is
    treated as having a formation volume factor of 1 - correct for a
    RESERVOIR-condition target (no conversion needed there) but wrong for
    a surface-condition target (ORAT/WRAT/GRAT/LRAT) if the phase's real
    formation volume factor differs from 1; not silently assumed correct.
    """

    def __attrs_post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValidationError(f"`cell_index` must be >= 0; got {self.cell_index}.")


@attrs.frozen(kw_only=True, slots=True)
class PerforationState(StoreSerializable):
    """Per-perforation dynamic snapshot for one timestep."""

    perforation: AnyPerforation
    cell_index: int
    flowing_pressure: Number
    phase_rates: typing.Mapping[FluidPhase, Number]
    unit_system: UnitSystem = UnitSystem.FIELD

    def __attrs_post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValidationError(f"`cell_index` must be >= 0; got {self.cell_index}.")

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `PerforationState` with `flowing_pressure` and every
            entry in `phase_rates` converted to target.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        reservoir_rate_factor = factors["reservoir_rate"]
        return attrs.evolve(
            self,
            flowing_pressure=self.flowing_pressure * factors["pressure"],
            phase_rates={
                phase: rate * reservoir_rate_factor
                for phase, rate in self.phase_rates.items()
            },
            unit_system=target,
        )


@attrs.frozen(kw_only=True, slots=True)
class WellState(StoreSerializable):
    """Per-well dynamic snapshot for one timestep."""

    well_name: str
    is_open: bool
    active_control: WellControl
    bhp: Number
    perforation_states: typing.Tuple[PerforationState, ...]
    phase_rates: typing.Mapping[FluidPhase, Number]
    active_limit: typing.Optional[Limit] = None
    thp: typing.Optional[Number] = None
    unit_system: UnitSystem = UnitSystem.FIELD

    def __attrs_post_init__(self) -> None:
        if not self.well_name:
            raise ValidationError("`well_name` must be a non-empty string.")

        if not self.is_open and self.perforation_states:
            raise ValidationError(
                f"WellState for {self.well_name!r} is shut (`is_open=False`) "
                "but has non-empty `perforation_states`; a shut well must "
                "have an empty tuple."
            )

        if self.active_control.unit_system != self.unit_system:
            raise ValidationError(
                f"`active_control.unit_system` ({self.active_control.unit_system.value}) "
                f"!= this WellState's unit_system ({self.unit_system.value})."
            )
        if (
            self.active_limit is not None
            and self.active_limit.unit_system != self.unit_system
        ):
            raise ValidationError(
                f"`active_limit.unit_system` ({self.active_limit.unit_system.value}) "
                f"!= this WellState's unit_system ({self.unit_system.value})."
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

        :param cell_index: 0-based `Grid` cell index.
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

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Returns a  new `WellState` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `WellState` with bhp, thp, phase_rates,
            `perforation_states`, `active_control`, and `active_limit` (if set)
            all converted to target.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        reservoir_rate_factor = factors["reservoir_rate"]
        return attrs.evolve(
            self,
            bhp=self.bhp * factors["pressure"],
            thp=self.thp * factors["pressure"] if self.thp is not None else None,
            phase_rates={
                phase: rate * reservoir_rate_factor
                for phase, rate in self.phase_rates.items()
            },
            perforation_states=tuple(
                ps.convert(target, table=table) for ps in self.perforation_states
            ),
            active_control=self.active_control.convert(target, table=table),
            active_limit=self.active_limit.convert(target, table=table)
            if self.active_limit is not None
            else None,
            unit_system=target,
        )


class WellsStates(
    StoreSerializable,
    fields={
        "states": typing.Mapping[str, WellState],
        "unit_system": typing.Optional[UnitSystem],
    },
):
    """Name-keyed collection of `WellState`, one per well, for a single timestep."""

    __slots__ = ("states", "unit_system")

    def __init__(
        self,
        states: typing.Mapping[str, WellState],
        unit_system: typing.Optional[UnitSystem] = None,
    ) -> None:
        """
        Initialize a `WellsStates` instance.

        :param states: Mapping from well name to `WellState`.
        :param unit_system: Target unit system for every state. None
            requires all states to already share the same unit system.
        :raises ValidationError: If any key doesn't match its value's
            `WellState.well_name`, or (unit_system is None) the states
            don't all share one unit system.
        """
        mismatched = {
            key: state.well_name
            for key, state in states.items()
            if key != state.well_name
        }
        if mismatched:
            raise ValidationError(
                f"`states` dict keys must match `WellState.well_name`; "
                f"mismatches (key -> well_name): {mismatched}."
            )

        if unit_system is None:
            systems = {state.unit_system for state in states.values()}
            if len(systems) > 1:
                raise ValidationError(
                    "All states must share the same unit system when "
                    "`unit_system` is not explicitly provided. Found: "
                    f"{sorted(s.value for s in systems)}."
                )
            unit_system = systems.pop() if systems else UnitSystem.FIELD
        else:
            states = {
                name: state
                if state.unit_system == unit_system
                else state.convert(unit_system)
                for name, state in states.items()
            }

        self.states = dict(states)
        self.unit_system = unit_system

    def get(self, well_name: str) -> typing.Optional[WellState]:
        """
        Get the `WellState` for a given well.

        :param well_name: Well to look up.
        :returns: WellState, or None if unset.
        """
        return self.states.get(well_name)

    def set(self, state: WellState) -> None:
        """
        Set the `WellState` for a given well.

        :param state: WellState to store, keyed by `state.well_name`.
        """
        self.states[state.well_name] = state

    @property
    def open_wells(self) -> typing.Tuple[WellState, ...]:
        """Returns Every WellState with is_open=True."""
        return tuple(state for state in self.states.values() if state.is_open)

    @property
    def shut_wells(self) -> typing.Tuple[WellState, ...]:
        """Returns every `WellState` with is_open=False."""
        return tuple(state for state in self.states.values() if not state.is_open)

    def __getitem__(self, well_name: str) -> WellState:
        """
        Get the `WellState` for a given well.

        :raises KeyError: If no state is set for `well_name`.
        """
        state = self.get(well_name)
        if state is None:
            raise KeyError(f"No state set for well {well_name!r}.")
        return state

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.states)

    def __len__(self) -> int:
        return len(self.states)

    def __contains__(self, well_name: object) -> bool:
        return well_name in self.states

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Returns a  new `WellsStates` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `WellsStates` with every state converted to target.
        """
        if target == self.unit_system:
            return self
        return self.__class__(
            states={
                name: state.convert(target, table=table)
                for name, state in self.states.items()
            },
            unit_system=target,
        )
