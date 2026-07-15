"""Dynamic simulation state for one point in time: reservoir state plus optional well states."""

import typing

import attrs

from bores.reservoir.state.base import ReservoirState
from bores.serde.base import Serializable
from bores.typing import Number
from bores.wells.states import WellStates

__all__ = ["BlackOilModelState"]


@attrs.frozen(kw_only=True, slots=True)
class BlackOilModelState(Serializable):
    """Reservoir state plus well states at one simulation time."""

    reservoir: ReservoirState
    wells: typing.Optional[WellStates] = None
    time: Number = 0.0
    """Simulation time this state corresponds to (days)."""
