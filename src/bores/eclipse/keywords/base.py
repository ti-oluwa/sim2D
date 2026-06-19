"""
Base implementation for Eclipse keyword parsers, and concrete implementations
for various keyword patterns.
"""

import abc
import typing
import warnings

import numpy as np
import numpy.typing as npt

from bores.eclipse.core import Deck, DeckParseError, GridDimensions, tokenise
from bores.eclipse.operators import (
    BoxOperation,
    apply_box_operation,
    resolve_box_operations,
)
from bores.typing import FloatArray, OneDimension

__all__ = [
    "Keyword",
    "Field",
    "RecordKeyword",
    "RepeatedRecordKeyword",
    "GridArrayKeyword"
]

T = typing.TypeVar("T")


class Field(typing.NamedTuple):
    """
    One positional field in a `RecordKeyword` or
    `RepeatedRecordKeyword` layout.

    :param name: Field name (used as key in the returned `dict`).
    :param type: Callable that converts the raw token string to the field
        value (e.g. `int`, `float`, `str`).
    :param required: Whether this field must be present.  Trailing optional
        fields fall back to `default` when the record has fewer tokens.
    :param default: Default value when the field is absent and
        `required=False`.
    """

    name: str
    type: typing.Callable[[str], typing.Any]
    required: bool = True
    default: typing.Any = None


class Keyword(typing.Generic[T], abc.ABC):
    """
    Base class for a single supported Eclipse keyword.

    A `Keyword` is pure parsing / shape logic: given the
    `bores.eclipse.core.Record`(s) that belong to it (already located by
    `bores.eclipse.core.Deck`) plus the deck's resolved grid dimensions,
    produce the keyword's Python value.

    Each concrete subclass only has to declare *shape*: field names/types
    for record keywords, expected array length for grid-array keywords, or
    nothing extra for nullary keywords. All cross-cutting deck mechanics
    (comments, includes, repeats, operators, boxing, caching) live in
    `bores.eclipse.datafile.DataFile` and are shared by every
    keyword for free.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name.upper()

    @abc.abstractmethod
    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[T]:
        """
        Parse this keyword's value out of `deck`.

        :param deck: The scanned deck to read records from.
        :param dims: Resolved grid dimensions, or `None` if
            `SPECGRID` / `DIMENS` has not been parsed yet (some
            keywords, e.g. `MAPAXES`, do not need it; array keywords
            will raise if it is required and missing).
        :returns: The parsed value, or `None` if the keyword is absent
            from the deck.
        """
        ...

    def has(self, deck: Deck) -> bool:
        """Return whether this keyword occurs anywhere in `deck`."""
        return deck.has(self.name)


def _parse_tokens_into_dict(
    keyword_name: str,
    fields: typing.Sequence[Field],
    tokens: typing.Sequence[str],
) -> typing.Dict[str, typing.Any]:
    """
    Convert a flat token sequence to a `{field_name: value}` dict
    according to `fields`.

    :param keyword_name: Keyword name, used only in error messages.
    :param fields: Field descriptors in positional order.
    :param tokens: Already-expanded token list for this record.
    :returns: Parsed field dict.
    :raises DeckParseError: If a required field is missing or has an
        invalid value.
    """
    result: typing.Dict[str, typing.Any] = {}
    for idx, field in enumerate(fields):
        if idx < len(tokens):
            raw = tokens[idx]
            try:
                result[field.name] = field.type(raw)
            except ValueError as exc:
                if field.required:
                    raise DeckParseError(
                        f"{keyword_name}: field {field.name!r} got invalid "
                        f"value {raw!r}: {exc}"
                    ) from exc
                result[field.name] = field.default
        elif field.required:
            raise DeckParseError(
                f"{keyword_name}: missing required field {field.name!r} "
                f"(got {len(tokens)} token(s), expected at least {idx + 1})."
            )
        else:
            result[field.name] = field.default
    return result


class RecordKeyword(Keyword[typing.Dict[str, typing.Any]]):
    """
    A keyword holding exactly one fixed-layout record of mixed-type fields
    (`SPECGRID`, `MAPAXES`, `GRIDUNIT`, `PINCH`).

    Subclasses declare `fields: Sequence[Field]` describing each
    positional token; :meth:`parse` returns a `dict` keyed by field name.
    Trailing optional fields (`required=False`) fall back to their
    `default` when the deck record has fewer tokens than declared fields.
    """

    __slots__ = ("fields",)

    def __init__(self, name: str, fields: typing.Sequence[Field]) -> None:
        super().__init__(name)
        self.fields = list(fields)

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[typing.Dict[str, typing.Any]]:
        record = deck.first_record_for(self.name)
        if record is None:
            return None
        tokens = tokenise(record.body)
        return self._parse_tokens(tokens)

    def _parse_tokens(
        self, tokens: typing.Sequence[str]
    ) -> typing.Dict[str, typing.Any]:
        return _parse_tokens_into_dict(self.name, self.fields, tokens)


class RepeatedRecordKeyword(Keyword[typing.List[typing.Dict[str, typing.Any]]]):
    """
    A keyword whose body holds zero or more individually `/`-terminated
    records sharing one field layout (`FAULTS`, `MULTFLT`, `NNC`).

    Multiple occurrences of the keyword in the deck (e.g. several
    `FAULTS` blocks) are concatenated in file order.

    Note:
        The body is split on `"/"` before tokenising individual lines.
        Quoted field values that contain a literal `"/"` character are
        therefore not supported (they are vanishingly rare in Eclipse
        practice).
    """

    __slots__ = ("fields",)

    def __init__(self, name: str, fields: typing.Sequence[Field]) -> None:
        super().__init__(name)
        self.fields = list(fields)

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[typing.List[typing.Dict[str, typing.Any]]]:
        records = deck.records_for(self.name)
        if not records:
            return None

        results: typing.List[typing.Dict[str, typing.Any]] = []
        for record in records:
            for line in record.body.split("/"):
                tokens = tokenise(line)
                if not tokens:
                    continue
                results.append(self._parse_tokens(tokens))
        return results or None

    def _parse_tokens(
        self, tokens: typing.Sequence[str]
    ) -> typing.Dict[str, typing.Any]:
        return _parse_tokens_into_dict(self.name, self.fields, tokens)


class GridArrayKeyword(Keyword[FloatArray[OneDimension]]):
    """
    A flat per-cell array keyword of length `nx * ny * nz`
    (`PORO`, `PERMX`, `ACTNUM`, `MULTX`, `TOPS`, …).

    Handles `N*value` repeat expansion and is automatically a valid
    target for `BOX` / `EQUALS` / `ADD` / `MULTIPLY` / `COPY` /
    `MAXVALUE` / `MINVALUE` operator records anywhere in the deck.
    Operator resolution is shared across all instances via
    `bores.eclipse.datafile.DataFile`; no individual keyword
    subclass needs to implement it.

    If the keyword itself never appears as a standalone data block but *is*
    the target of one or more operator records (a common Eclipse idiom for
    keywords like `MULTX` that are only ever set via `BOX`/`EQUALS`),
    the array still starts from `default_value` and operators are applied
    on top.
    """

    __slots__ = ("dtype", "default_value", "is_multiplier")

    def __init__(
        self,
        name: str,
        *,
        dtype: npt.DTypeLike = np.float64,
        default_value: float = 0.0,
        is_multiplier: bool = False,
    ) -> None:
        """
        :param name: Keyword name (e.g. `"PORO"`).
        :param dtype: Output array dtype. The internal accumulation is
            always `float64`; the result is cast to `dtype` at the end.
        :param default_value: Value cells start at before any explicit data
            block or operator is applied.
        :param is_multiplier: Convenience flag; when `True`, overrides
            `default_value` to `1.0` (the multiplicative identity)
            regardless of what was passed for `default_value`.  Use for
            `MULTX`, `MULTY`, `MULTZ`, and their `-` variants.
        """
        super().__init__(name)
        self.dtype = np.dtype(dtype)
        self.default_value = 1.0 if is_multiplier else default_value
        self.is_multiplier = is_multiplier

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[FloatArray[OneDimension]]:
        """
        Parse and return the resolved per-cell array.

        :param deck: The scanned deck.
        :param dims: Resolved grid dimensions (required).
        :returns: Shape `(n_cells,)` array in Eclipse flat order
            (i-fastest), or `None` if the keyword is absent from the
            deck and no operator targets it.
        :raises DeckParseError: If `dims` is `None` or the keyword
            body is malformed.
        """
        if dims is None:
            raise DeckParseError(
                f"Cannot parse grid array keyword {self.name!r} without "
                "resolved grid dimensions (SPECGRID/DIMENS not found)."
            )
        return self._resolve(deck, dims, stop_before_order=None)

    def _resolve(
        self,
        deck: Deck,
        dims: GridDimensions,
        *,
        stop_before_order: typing.Optional[typing.Tuple[int, int]],
    ) -> typing.Optional[FloatArray[OneDimension]]:
        """
        Replay every event affecting this keyword's array — its own
        explicit data block(s) *and* every operator record targeting it —
        strictly in the order they occur in the deck.

        This is what makes e.g. `MULTX` set via `EQUALS` and later
        overwritten by an explicit `MULTX ... /` block (or vice versa)
        behave like Eclipse: whichever event is last in the file wins for
        the cells it touches.

        :param deck: Scanned deck.
        :param dims: Resolved grid dimensions.
        :param stop_before_order: If given, only replay events with
            `order < stop_before_order`. Used by `COPY` to resolve a
            source keyword's value *as of* the point in the file where the
            `COPY` record itself occurs.
        :returns: The resolved array, or `None` if this keyword has no
            data block and no operator ever targets it.
        """
        events = self._timeline(deck, dims)
        if stop_before_order is not None:
            events = [e for e in events if e[0] < stop_before_order]
        if not events:
            return None

        array = np.full(dims.n_cells, self.default_value, dtype=np.float64)

        def _resolve_source_at(
            keyword_name: str, as_of_order: typing.Tuple[int, int]
        ) -> typing.Optional[FloatArray[OneDimension]]:
            # Guess the default_value for arbitrary source keywords:
            # multiplier-style arrays default to 1.0, property arrays to 0.0.
            # We cannot know for certain without a registered Keyword instance,
            # but this heuristic is correct for all standard Eclipse arrays.
            default = 1.0 if keyword_name.startswith("MULT") else 0.0
            probe = GridArrayKeyword(keyword_name, default_value=default)
            return probe._resolve(deck, dims, stop_before_order=as_of_order)

        for order, kind, payload in events:
            if kind == "assign":
                tokens: typing.List[str] = payload  # type: ignore[assignment]
                if len(tokens) == 0:
                    # Edge case: bare keyword with empty body (e.g. "3*" -> ["","",""])
                    # after float conversion would fail; treat as no-op.
                    warnings.warn(
                        f"{self.name}: assign event has zero tokens; skipping.",
                        stacklevel=4,
                    )
                    continue
                if len(tokens) == 1:
                    try:
                        array[:] = float(tokens[0])
                    except ValueError as exc:
                        raise DeckParseError(
                            f"{self.name}: expected a numeric scalar, got "
                            f"{tokens[0]!r}: {exc}"
                        ) from exc
                elif len(tokens) == dims.n_cells:
                    try:
                        array[:] = np.array(tokens, dtype=np.float64)
                    except ValueError as exc:
                        raise DeckParseError(
                            f"{self.name} contains a non-numeric value: {exc}"
                        ) from exc
                else:
                    raise DeckParseError(
                        f"{self.name} expected 1 or {dims.n_cells} value(s); "
                        f"got {len(tokens)}."
                    )
            else:  # kind == "operate"
                operation: BoxOperation = payload  # type: ignore[assignment]
                apply_box_operation(
                    array,
                    operation,
                    dims,
                    resolve_source=lambda name, _order=order: _resolve_source_at(
                        name, _order
                    ),
                )

        return array.astype(self.dtype, copy=False)  # type: ignore[return-value]

    def _timeline(
        self, deck: Deck, dims: GridDimensions
    ) -> typing.List[typing.Tuple[typing.Tuple[int, int], str, typing.Any]]:
        """
        Build the ordered `(order, kind, payload)` events affecting this
        keyword: `("assign", tokens)` for each of its own explicit
        data-block records, interleaved with `("operate", BoxOperation)`
        for every operator record that targets it.

        Both event kinds share one `order` namespace — a
        `(record_start, line_index)` tuple.

        :returns: Events sorted ascending (earliest in file first).
        """
        events: typing.List[typing.Tuple[typing.Tuple[int, int], str, typing.Any]] = []

        for record in deck.records_for(self.name):
            raw_tokens = tokenise(record.body)
            # Filter out tokens that are empty strings (produced by bare "N*"
            # repeat syntax). A record that expands to only empty strings is
            # dropped with a warning rather than crashing float() later.
            if any(t == "" for t in raw_tokens):
                clean_tokens = [t for t in raw_tokens if t != ""]
                warnings.warn(
                    f"{self.name}: record body contains bare 'N*' repeat(s) "
                    f"(empty-value tokens); {len(raw_tokens) - len(clean_tokens)} "
                    "token(s) dropped.",
                    stacklevel=6,
                )
                raw_tokens = clean_tokens
            events.append(((record.start, 0), "assign", raw_tokens))

        for op in resolve_box_operations(deck, dims):
            if op.target == self.name:
                events.append((op.order, "operate", op))

        events.sort(key=lambda e: e[0])
        return events
