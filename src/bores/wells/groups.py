"""Well-group hierarchy and group-level control targets."""

import enum
import typing

import attrs
from typing_extensions import Self

from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.serde.stores import StoreSerializable
from bores.typing import FluidPhase, Number

__all__ = [
    "Group",
    "Groups",
    "GroupProducerControlMode",
    "GroupInjectorControlMode",
    "GroupControl",
    "GroupControls",
    "FIELD_GROUP",
]

FIELD_GROUP = "FIELD"
"""The implicit root group name every top-level group's `parent` resolves to."""


@attrs.frozen(kw_only=True, slots=True)
class Group(Serializable):
    """One node in the group hierarchy - deck `GRUPTREE` child/parent link."""

    name: str
    parent: typing.Optional[str] = None
    """
    `None` only for the implicit root (`name == "FIELD"`); every other
    group must have a parent (possibly `"FIELD"` itself).
    """

    def __attrs_post_init__(self) -> None:
        if not self.name:
            raise ValidationError("`name` must be a non-empty string.")
        if self.parent is None and self.name != FIELD_GROUP:
            raise ValidationError(
                f"Group {self.name!r} must have a `parent` (only the "
                f"{FIELD_GROUP!r} root may omit one)."
            )


class Groups(StoreSerializable):
    """Name-keyed group hierarchy. Lookup and traversal only."""

    __abstract_serializable__ = True

    def __init__(self, groups: typing.Dict[str, Group]) -> None:
        groups = dict(groups)
        groups.setdefault(FIELD_GROUP, Group(name=FIELD_GROUP, parent=None))
        for name, group in groups.items():
            if group.parent is not None and group.parent not in groups:
                raise ValidationError(
                    f"Group {name!r}'s parent {group.parent!r} is not itself "
                    "a declared group."
                )
        self._groups = groups

    def group(self, name: str) -> Group:
        """
        Returns group with the given name.

        :param name: The group name
        :raises KeyError: If no group with that name exists.
        """
        group = self._groups.get(name)
        if group is None:
            raise KeyError(f"No group named {name!r}.")
        return group

    def children(self, name: str) -> typing.Tuple[str, ...]:
        """Direct child group names of `name` only (not grandchildren)."""
        if name not in self._groups:
            raise KeyError(f"No group named {name!r}.")
        return tuple(
            name for name, group in self._groups.items() if group.parent == name
        )

    def descendants(self, name: str) -> typing.Tuple[str, ...]:
        """Every group under `name`, at any depth (not including `name` itself)."""
        result: typing.List[str] = []
        frontier = list(self.children(name))
        while frontier:
            current = frontier.pop()
            result.append(current)
            frontier.extend(self.children(current))
        return tuple(result)

    def __getitem__(self, name: str) -> Group:
        return self.group(name)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __contains__(self, name: object) -> bool:
        return name in self._groups

    def __dump__(self) -> typing.Dict[str, typing.Any]:
        return {"groups": {name: group.dump() for name, group in self._groups.items()}}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        groups = {name: Group.load(gd) for name, gd in data["groups"].items()}
        return cls(groups=groups)


class GroupProducerControlMode(enum.Enum):
    """Deck `GCONPROD` item 2 - which target field is active for a production group."""

    ORAT = "orat"
    WRAT = "wrat"
    GRAT = "grat"
    LRAT = "lrat"
    RESV = "resv"
    FLD = "fld"
    """Controlled by its parent group's allocation instead of its own target."""
    NONE = "none"

    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


class GroupInjectorControlMode(enum.Enum):
    """Deck `GCONINJE` item 3 - which target field is active for an injection group."""

    RATE = "rate"
    RESV = "resv"
    VREP = "vrep"
    REIN = "rein"
    FLD = "fld"

    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(str(value).lower())


@attrs.frozen(kw_only=True, slots=True)
class GroupControl(Serializable):
    """A group's production or injection target.

    Same "all targets may be populated, `mode` selects the active one"
    shape as `ProducerControl`/`InjectorControl` - deck `GCONPROD`/
    `GCONINJE` supplies several target items regardless of which one is
    active.
    """

    mode: typing.Union[GroupProducerControlMode, GroupInjectorControlMode]
    target_rate: typing.Optional[Number] = None
    injected_phase: typing.Optional[FluidPhase] = None
    """Set only for an injection group (`mode` is a `GroupInjectorControlMode`)."""


class GroupControls(StoreSerializable):
    """
    Name-keyed, mutable mapping from group name to its current `GroupControl`.
    """

    __abstract_serializable__ = True

    def __init__(self, controls: typing.Dict[str, GroupControl]) -> None:
        self._controls = dict(controls)

    def get(self, name: str) -> typing.Optional[GroupControl]:
        return self._controls.get(name)

    def set(self, name: str, control: GroupControl) -> None:
        self._controls[name] = control

    def __getitem__(self, name: str) -> GroupControl:
        control = self.get(name)
        if control is None:
            raise KeyError(f"No control set for group {name!r}.")
        return control

    def __setitem__(self, name: str, control: GroupControl) -> None:
        self.set(name, control)

    def __delitem__(self, name: str) -> None:
        if name not in self._controls:
            raise KeyError(f"No control set for group {name!r}.")
        del self._controls[name]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._controls)

    def __len__(self) -> int:
        return len(self._controls)

    def __contains__(self, name: object) -> bool:
        return name in self._controls

    def __dump__(self) -> typing.Dict[str, typing.Any]:
        return {
            "controls": {
                name: control.dump() for name, control in self._controls.items()
            }
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        controls = {
            name: GroupControl.load(control_data)
            for name, control_data in data["controls"].items()
        }
        return cls(controls=controls)
