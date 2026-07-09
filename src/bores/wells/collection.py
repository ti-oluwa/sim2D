"""Name-keyed container of well specs."""

import typing

from bores.errors import ValidationError
from bores.serde.stores import StoreSerializable
from bores.wells.data import WellSpec, WellType

__all__ = ["Wells"]


class Wells(StoreSerializable):
    """Name-keyed container of `WellSpec`. Lookup and iteration only."""

    __abstract_serializable__ = True

    def __init__(self, wells: typing.Dict[str, WellSpec]) -> None:
        """
        :param wells: Mapping from well name to `WellSpec`.
        :raises ValidationError: If `wells` is empty, or if any key doesn't
            match its value's `WellSpec.name`.
        """
        if not wells:
            raise ValidationError("`wells` must contain at least one entry.")
        mismatched = {key: spec.name for key, spec in wells.items() if key != spec.name}
        if mismatched:
            raise ValidationError(
                f"`wells` dict keys must match `WellSpec.name`; mismatches "
                f"(key -> spec.name): {mismatched}."
            )
        self._wells = wells

    def well(self, name: str) -> WellSpec:
        """
        :param name: Well name.
        :returns: `WellSpec` for that well.
        :raises KeyError: If no well with that name exists.
        """
        spec = self._wells.get(name)
        if spec is None:
            raise KeyError(f"No well named {name!r}. Available: {sorted(self._wells)}.")
        return spec

    @property
    def names(self) -> typing.Tuple[str, ...]:
        """All well names, insertion order."""
        return tuple(self._wells.keys())

    @property
    def producers(self) -> typing.Tuple[WellSpec, ...]:
        """All wells with `well_type is WellType.PRODUCER`."""
        return tuple(
            spec for spec in self._wells.values() if spec.well_type is WellType.PRODUCER
        )

    @property
    def injectors(self) -> typing.Tuple[WellSpec, ...]:
        """All wells with `well_type is WellType.INJECTOR`."""
        return tuple(
            spec for spec in self._wells.values() if spec.well_type is WellType.INJECTOR
        )

    def __getitem__(self, name: str) -> WellSpec:
        return self.well(name)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._wells)

    def __len__(self) -> int:
        return len(self._wells)

    def __contains__(self, name: object) -> bool:
        return name in self._wells

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "wells": {name: spec.dump(recurse) for name, spec in self._wells.items()}
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> "Wells":
        wells = {
            name: WellSpec.load(spec_data) for name, spec_data in data["wells"].items()
        }
        return cls(wells=wells)
