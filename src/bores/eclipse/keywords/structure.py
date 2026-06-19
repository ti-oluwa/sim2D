"""Grid structural keyword implementations: faults and non-neighbour connections."""

import typing

from bores.eclipse.core import Deck, DeckParseError, GridDimensions
from bores.eclipse.keywords.base import Field, RepeatedRecordKeyword

__all__ = ["Faults", "MultFLT", "NNC"]


class Faults(RepeatedRecordKeyword):
    """
    `FAULTS 'NAME' I1 I2 J1 J2 K1 K2 FACE / ... /` — named fault planes.

    Multiple `FAULTS` keyword blocks in the same deck are concatenated in
    file order.  Indices `I1`/`I2`/`J1`/`J2`/`K1`/`K2` are
    *1-based* Eclipse IJK indices, passed through unchanged to the caller.

    :raises DeckParseError: If `FACE` is not one of `I`, `I-`, `J`,
        `J-`, `K`, `K-`.
    """

    _VALID_FACES: typing.FrozenSet[str] = frozenset({"I", "I-", "J", "J-", "K", "K-"})

    def __init__(self) -> None:
        super().__init__(
            "FAULTS",
            fields=[
                Field("name", str),
                Field("i1", int),
                Field("i2", int),
                Field("j1", int),
                Field("j2", int),
                Field("k1", int),
                Field("k2", int),
                Field("face", str),
            ],
        )

    def _parse_tokens(
        self, tokens: typing.Sequence[str]
    ) -> typing.Dict[str, typing.Any]:
        result = super()._parse_tokens(tokens)
        face = str(result["face"]).upper()
        if face not in self._VALID_FACES:
            raise DeckParseError(
                f"FAULTS record for {result.get('name')!r}: unrecognised "
                f"face direction {face!r}.  "
                f"Valid values: {sorted(self._VALID_FACES)}."
            )
        result["face"] = face
        return result


class MultFLT(RepeatedRecordKeyword):
    """
    `MULTFLT 'NAME' MULTIPLIER / ... /`
    — per-fault transmissibility multiplier.

    Eclipse semantics: when a fault name appears in multiple records
    across one or more `MULTFLT` blocks, the *last* value wins.
    :meth:`parse` enforces this automatically.
    """

    def __init__(self) -> None:
        super().__init__(
            "MULTFLT",
            fields=[
                Field("name", str),
                Field("multiplier", float),
            ],
        )

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[typing.List[typing.Dict[str, typing.Any]]]:
        records = super().parse(deck, dims)
        if records is None:
            return None
        
        # Last value for each fault name wins.
        by_name: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        for rec in records:
            by_name[rec["name"]] = rec
        return list(by_name.values())


class NNC(RepeatedRecordKeyword):
    """
    `NNC I1 J1 K1 I2 J2 K2 T / ... /`
    — explicit non-neighbour connections.

    Indices `I1/J1/K1` and `I2/J2/K2` are *1-based* Eclipse IJK
    structured cell indices.  `T` is the transmissibility value in the
    grid's declared unit system.

    Multiple `NNC` keyword blocks in the same deck are concatenated.
    """

    def __init__(self) -> None:
        super().__init__(
            "NNC",
            fields=[
                Field("i1", int),
                Field("j1", int),
                Field("k1", int),
                Field("i2", int),
                Field("j2", int),
                Field("k2", int),
                Field("transmissibility", float),
            ],
        )
