"""
Base implementation for Eclipse keyword parsers, and concrete implementations
for various common keyword patterns.
"""

import abc
import datetime
import typing
import warnings
from collections.abc import Collection

import attrs
import numpy as np
import numpy.typing as npt

from bores.deck.core import Deck, DeckParseError, GridDimensions, tokenise
from bores.deck.operators import Operation, apply_operation, resolve_operations
from bores.typing import FloatArray, Number, OneDimension

__all__ = [
    "Keyword",
    "Field",
    "RecordKeyword",
    "RepeatedRecordKeyword",
    "GridArrayKeyword",
    "FlagKeyword",
    "DateKeyword",
    "DatesKeyword",
    "PVTTableKeyword",
]

T = typing.TypeVar("T")


@attrs.define(slots=True, frozen=True)
class Field(typing.Generic[T]):
    """
    One positional field in a `RecordKeyword` or
    `RepeatedRecordKeyword` layout.

    :param name: Field name (used as key in the returned `dict`).
    :param type: Callable that converts the raw token string to the field
        value (e.g. `int`, `float`, `str`, or a function/validator).
    :param required: Whether this field must be present.  Trailing optional
        fields fall back to `default` when the record has fewer tokens.
    :param default: Default value when the field is absent and
        `required=False`.
    """

    name: str
    type: typing.Callable[[str], T]
    required: bool = True
    default: typing.Optional[T] = None
    options: typing.Optional[Collection[T]] = None

    def __attrs_post_init__(self) -> None:
        if not self.name:
            raise ValueError("Field name required. `name` can not be empty.")

        options = frozenset(self.options) if self.options else None
        if options and self.default is not None and self.default not in options:
            raise ValueError(
                f"Invalid default value for {self.name!r} - {self.default!r}. "
                f"Default should be one of {', '.join([str(option) for option in options])}"
            )
        object.__setattr__(self, "options", options)

    def parse(self, raw: str, keyword_name: str) -> typing.Optional[T]:
        default = False
        try:
            value = self.type(raw)
        except ValueError as exc:
            if self.required:
                raise DeckParseError(
                    f"{keyword_name} record: {self.name!r} got invalid value {raw!r}: {exc}"
                ) from exc
            value = self.default
            default = True

        if self.options and not default and value not in self.options:
            raise DeckParseError(
                f"{keyword_name} record: {self.name!r} got unrecognised value {raw!r}. "
                f"Value should be one of {', '.join([str(option) for option in self.options])}"
            )
        return value


class Keyword(typing.Generic[T], abc.ABC):
    """
    Base class for a single supported Eclipse keyword.

    A `Keyword` is pure parsing / shape logic: given the
    `bores.deck.core.Record`(s) that belong to it (already located by
    `bores.deck.core.Deck`) plus the deck's resolved grid dimensions,
    produce the keyword's Python value.

    Each concrete subclass only has to declare *shape*: field names/types
    for record keywords, expected array length for grid-array keywords, or
    nothing extra for nullary keywords. All cross-cutting deck mechanics
    (comments, includes, repeats, operators, boxing, caching) live in
    `bores.deck.file.DeckFile` and are shared by every
    keyword for free.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name.upper()

    @abc.abstractmethod
    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
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

    def __hash__(self) -> int:
        return hash(self.name)


def _parse_tokens(
    keyword_name: str,
    fields: typing.Sequence[Field[T]],
    tokens: typing.Sequence[str],
) -> typing.Dict[str, typing.Optional[T]]:
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
    result: typing.Dict[str, typing.Optional[T]] = {}
    for idx, field in enumerate(fields):
        if idx < len(tokens):
            raw = tokens[idx]
            result[field.name] = field.parse(raw, keyword_name)
        elif field.required:
            raise DeckParseError(
                f"{keyword_name} record: missing required field {field.name!r} "
                f"(got {len(tokens)} token(s), expected at least {idx + 1})."
            )
        else:
            result[field.name] = field.default
    return result


class RecordKeyword(Keyword[typing.Dict[str, typing.Optional[T]]]):
    """
    A keyword holding exactly one fixed-layout record of mixed-type fields
    (`SPECGRID`, `MAPAXES`, `GRIDUNIT`, `PINCH`).

    Subclasses declare `fields: Sequence[Field]` describing each
    positional token; `parse` returns a `dict` keyed by field name.
    Trailing optional fields (`required=False`) fall back to their
    `default` when the deck record has fewer tokens than declared fields.
    """

    __slots__ = ("fields",)

    def __init__(self, name: str, fields: typing.Sequence[Field[T]]) -> None:
        super().__init__(name)
        self.fields = list(fields)

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> typing.Optional[typing.Dict[str, typing.Optional[T]]]:
        record = deck.first_record_for(self.name)
        if record is None:
            return None
        tokens = tokenise(record.body)
        return self._parse_tokens(tokens)

    def _parse_tokens(
        self, tokens: typing.Sequence[str]
    ) -> typing.Dict[str, typing.Optional[T]]:
        return _parse_tokens(self.name, self.fields, tokens)


class RepeatedRecordKeyword(Keyword[typing.List[typing.Dict[str, typing.Optional[T]]]]):
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

    def __init__(self, name: str, fields: typing.Sequence[Field[T]]) -> None:
        super().__init__(name)
        self.fields = list(fields)

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> typing.Optional[typing.List[typing.Dict[str, typing.Optional[T]]]]:
        records = deck.records_for(self.name)
        if not records:
            return None

        results: typing.List[typing.Dict[str, typing.Optional[T]]] = []
        for record in records:
            for line in record.body.split("/"):
                tokens = tokenise(line)
                if not tokens:
                    continue
                results.append(self._parse_tokens(tokens))
        return results or None

    def _parse_tokens(
        self, tokens: typing.Sequence[str]
    ) -> typing.Dict[str, typing.Optional[T]]:
        return _parse_tokens(self.name, self.fields, tokens)


class GridArrayKeyword(Keyword[FloatArray[OneDimension]]):
    """
    A flat per-cell array keyword of length `nx * ny * nz`
    (`PORO`, `PERMX`, `ACTNUM`, `MULTX`, `TOPS`, …).

    Handles `N*value` repeat expansion and is automatically a valid
    target for `BOX` / `EQUALS` / `ADD` / `MULTIPLY` / `COPY` /
    `MAXVALUE` / `MINVALUE` operator records anywhere in the deck.
    Operator resolution is shared across all instances via
    `bores.deck.file.DeckFile`; no individual keyword
    subclass needs to implement it.

    If the keyword itself never appears as a standalone data block but *is*
    the target of one or more operator records (a common Eclipse idiom for
    keywords like `MULTX` that are only ever set via `BOX`/`EQUALS`),
    the array still starts from `default_value` and operators are applied
    on top.
    """

    __slots__ = ("dtype", "default_value", "is_multiplier", "column_shape")

    def __init__(
        self,
        name: str,
        *,
        dtype: npt.DTypeLike = np.float64,
        default_value: float = 0.0,
        is_multiplier: bool = False,
        column_shape: typing.Optional[typing.Tuple[str, ...]] = None,
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
        :param column_shape: Optional tuple of dimension-name strings declaring
            a valid "short form" token count for this keyword.  Each string
            must be one of `"nx"`, `"ny"`, `"nz"`.  When the token count
            equals the product of the named dimensions, the parsed values are
            broadcast (tiled) along the remaining axes to fill `n_cells`.

            Example - `TOPS` accepts either `nx*ny` column tops or a full
            `nx*ny*nz` per-cell array:

            ```python
            TOPS = GridArrayKeyword("TOPS", column_shape=("nx", "ny"))
            ```

            Supported broadcast patterns (axes not listed in `column_shape`
            are tiled):

            ```md
            ======================  ========================  =====================
            `column_shape`          Short-form token count    Broadcast direction
            ======================  ========================  =====================
            `("nx", "ny")`          `nx * ny`                 repeated `nz` times
            `("nx", "nz")`          `nx * nz`                 repeated `ny` times
            `("ny", "nz")`          `ny * nz`                 repeated `nx` times
            `("nx",)`               `nx`                      repeated `ny*nz` times
            `("ny",)`               `ny`                      repeated `nx*nz` times
            `("nz",)`               `nz`                      repeated `nx*ny` times
            ======================  ========================  =====================
            ```

        """
        super().__init__(name)
        self.dtype = np.dtype(dtype)
        self.default_value = 1.0 if is_multiplier else default_value
        self.is_multiplier = is_multiplier

        if column_shape and len(column_shape) > 3:
            raise ValueError(f"Invalid size for `column_shape`: {column_shape!r}")
        self.column_shape = (
            [col.lower() for col in column_shape] if column_shape else None
        )

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
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
        return self._resolve(
            deck,
            dims,
            operations=operations,
            stop_before_order=None,
        )

    def _resolve(
        self,
        deck: Deck,
        dims: GridDimensions,
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
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
        events = self._timeline(deck, dims, operations=operations)
        if stop_before_order is not None:
            events = [e for e in events if e[0] < stop_before_order]
        if not events:
            return None

        array = np.full(dims.n_cells, self.default_value, dtype=np.float64)

        def _resolve_source_at(
            keyword_name: str,
            as_of_order: typing.Tuple[int, int],
            *,
            operations: typing.Optional[typing.List[Operation]] = None,
        ) -> typing.Optional[FloatArray[OneDimension]]:
            # Guess the default_value for arbitrary source keywords:
            # multiplier-style arrays default to 1.0, property arrays to 0.0.
            # We cannot know for certain without a registered Keyword instance,
            # but this heuristic is correct for all standard Eclipse arrays.
            default = 1.0 if keyword_name.startswith("MULT") else 0.0
            probe = GridArrayKeyword(keyword_name, default_value=default)
            return probe._resolve(
                deck, dims, operations=operations, stop_before_order=as_of_order
            )

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
                    # Try `column_shape` broadcast if declared
                    broadcast_ok = False
                    if self.column_shape is not None:
                        short_count = self._short_form_count(dims)
                        if len(tokens) == short_count:
                            try:
                                array[:] = self._broadcast_short_form(
                                    np.array(tokens, dtype=np.float64), dims
                                )
                                broadcast_ok = True
                            except ValueError as exc:
                                raise DeckParseError(
                                    f"{self.name} contains a non-numeric value: {exc}"
                                ) from exc
                    if not broadcast_ok:
                        expected_desc = f"1 or {dims.n_cells}"
                        if self.column_shape is not None:
                            short_count = self._short_form_count(dims)
                            expected_desc += f" or {short_count} (short form: {' * '.join(self.column_shape)})"
                        raise DeckParseError(
                            f"{self.name} expected {expected_desc} value(s); "
                            f"got {len(tokens)}."
                        )

            else:  # kind == "operate"
                operation: Operation = payload  # type: ignore[assignment]
                apply_operation(
                    array,
                    operation,
                    dims,
                    resolve_source=lambda name, _order=order: _resolve_source_at(
                        name, _order, operations=operations
                    ),
                )

        return array.astype(self.dtype, copy=False)  # type: ignore[return-value]

    def _short_form_count(self, dims: GridDimensions) -> int:
        """
        Return the number of tokens accepted in the short (column) form.

        :param dims: Resolved grid dimensions.
        :returns: Product of the dimensions named in `self.column_shape`.
        """
        assert self.column_shape is not None
        dim_map = {"nx": dims.nx, "ny": dims.ny, "nz": dims.nz}
        result = 1
        for axis in self.column_shape:
            result *= dim_map[axis]
        return result

    def _broadcast_short_form(
        self,
        short_array: npt.NDArray[np.float64],
        dims: GridDimensions,
    ) -> npt.NDArray[np.float64]:
        """
        Broadcast a short-form array to full `n_cells` length.

        The short array covers the axes in `self.column_shape` in Eclipse
        flat order (i fastest). The missing axes are tiled so that the
        result is in standard Eclipse flat order `i + j*nx + k*nx*ny`.

        :param short_array: 1-D float64 array of length `_short_form_count(dims)`.
        :param dims: Resolved grid dimensions.
        :returns: 1-D float64 array of length `n_cells`.
        """
        assert self.column_shape is not None
        dim_map = {"nx": dims.nx, "ny": dims.ny, "nz": dims.nz}

        # Reshape short array into its natural (i,j,k) sub-space
        shape_present = tuple(dim_map[ax] for ax in self.column_shape)
        arr = short_array.reshape(shape_present, order="F")  # i fastest

        # Expand missing axes by inserting new dimensions and tiling
        # We build the full (nx, ny, nz) array then ravel in Fortran order
        full = np.empty((dims.nx, dims.ny, dims.nz), dtype=np.float64)

        # Map axis names to indices in the full (nx,ny,nz) array
        axis_indices = {"nx": 0, "ny": 1, "nz": 2}

        # Reshape arr to align with full array dimensions
        # Insert size-1 dims for missing axes, then broadcast
        target_shape = [1, 1, 1]
        for local_i, ax in enumerate(self.column_shape):
            target_shape[axis_indices[ax]] = dim_map[ax]
        arr_expanded = arr.reshape(target_shape)
        full[:] = np.broadcast_to(arr_expanded, (dims.nx, dims.ny, dims.nz))
        return full.ravel(order="F")  # Eclipse flat order: i fastest

    def _timeline(
        self,
        deck: Deck,
        dims: GridDimensions,
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> typing.List[typing.Tuple[typing.Tuple[int, int], str, typing.Any]]:
        """
        Build the ordered `(order, kind, payload)` events affecting this
        keyword: `("assign", tokens)` for each of its own explicit
        data-block records, interleaved with `("operate", Operation)`
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

        if operations is None:
            operations = resolve_operations(deck, dims)

        for operation in operations:
            if operation.target == self.name:
                events.append((operation.order, "operate", operation))

        events.sort(key=lambda e: e[0])
        return events


_MONTH_MAP: typing.Dict[str, int] = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _parse_date(tokens: typing.Sequence[str], keyword_name: str) -> datetime.date:
    """
    Parse a three-token Eclipse date `[day, month, year]` into a `datetime.date`.

    Eclipse month tokens may be bare (`JAN`) or quoted (`'JAN'`);
    quoting is already stripped by `bores.deck.core.tokenise`.

    :param tokens: At least three string tokens: day, month abbreviation, year.
    :param keyword_name: Keyword name for error messages.
    :returns: Parsed date.
    :raises DeckParseError: If the tokens cannot be parsed as a valid date.
    """
    if len(tokens) < 3:
        raise DeckParseError(
            f"{keyword_name}: expected 3 tokens for a date (DAY MON YEAR); "
            f"got {len(tokens)}: {list(tokens)!r}."
        )
    try:
        day = int(tokens[0])
    except ValueError as exc:
        raise DeckParseError(
            f"{keyword_name}: day token {tokens[0]!r} is not an integer."
        ) from exc

    month_str = tokens[1].upper()
    month = _MONTH_MAP.get(month_str)
    if month is None:
        raise DeckParseError(
            f"{keyword_name}: unrecognised month abbreviation {tokens[1]!r}. "
            f"Expected one of {sorted(_MONTH_MAP)}."
        )

    try:
        year = int(tokens[2])
    except ValueError as exc:
        raise DeckParseError(
            f"{keyword_name}: year token {tokens[2]!r} is not an integer."
        ) from exc

    try:
        return datetime.date(year, month, day)
    except ValueError as exc:
        raise DeckParseError(
            f"{keyword_name}: invalid date {day}/{month}/{year}: {exc}"
        ) from exc


class FlagKeyword(Keyword[bool]):
    """
    A bare presence-only switch keyword that carries no data section.

    Examples: `OIL`, `GAS`, `WATER`, `DISGAS`, `VAPOIL`,
    `FIELD`, `METRIC`, `LAB`, `NOSIM`, `UNIFIN`, `UNIFOUT`.

    `parse` returns `True` when the keyword appears anywhere in
    the deck, `False` otherwise.  It never returns `None` so callers
    can always do a truthiness test.
    """

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> bool:
        """
        :returns: `True` if the keyword is present in the deck,
            `False` if absent.
        """
        return deck.has(self.name)


class DateKeyword(Keyword[datetime.date]):
    """
    A keyword whose single record is an Eclipse-format date
    `D MON YYYY /`.

    Used for `START`.

    `parse` returns `datetime.date`, or `None` when the
    keyword is absent.
    """

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> typing.Optional[datetime.date]:
        record = deck.first_record_for(self.name)
        if record is None:
            return None
        tokens = tokenise(record.body)
        return _parse_date(tokens, self.name)


class DatesKeyword(Keyword[typing.List[datetime.date]]):
    """
    The `DATES` keyword: a sequence of one or more `/`-terminated
    date entries inside one `DATES … /` block, each of the form
    `D MON YYYY /`.

    Multiple `DATES` blocks in the same deck are concatenated in file
    order and returned as a single flat list.

    `parse` returns a list o `datetime.date` objects, or
    `None` when the keyword is absent.

    Example deck fragment:

        DATES
         1 JAN 2020 /
         1 FEB 2020 /
        /
    """

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> typing.Optional[typing.List[datetime.date]]:
        records = deck.records_for(self.name)
        if not records:
            return None

        dates: typing.List[datetime.date] = []
        for record in records:
            for segment in record.body.split("/"):
                tokens = tokenise(segment)
                if not tokens:
                    continue
                dates.append(_parse_date(tokens, self.name))

        return dates or None


PVTRow = typing.Dict[str, T]
"""One row of a PVT/saturation table: a `{column_name: value}` dict."""

PVTTable = typing.List[PVTRow[T]]
"""One saturation/PVT table: a list of row dicts in ascending primary-key order."""


class PVTTableKeyword(Keyword[typing.List[PVTTable[Number]]]):
    """
    A keyword whose body contains one or more tabulated data blocks,
    each terminated by `/`. Multiple keyword occurrences (e.g. one per
    PVT region) are collected in order.

    This covers two sub-patterns:

    **Simple (immiscible) tables** — every `/`-terminated segment is one
    complete row; all rows in a block share the same column layout.
    Examples: `SWOF`, `SGOF`, `PVDG`, `PVDO`, `SWFN`, `SGFN`,
    `SOF2`, `SOF3`, `ROCK`, `DENSITY`, `PVTW`.

    **Miscible (saturated/under-saturated) tables** — a single primary-key
    value on its own `/`-separated segment introduces a new "inner table",
    and subsequent segments are rows belonging to that inner table until the
    next primary-key segment or a double `//` terminates the table.
    Examples: `PVTO`, `PVTG`.

    :param name: Keyword name.
    :param columns: Column descriptors for each field in a data row.
    :param primary_key: If not `None`, this column name is the
        "outer" key used in miscible (bracketed) PVT tables such as
        `PVTO` / `PVTG`. When `None`, the table is flat (immiscible
        / simple-tabular).
    :param table_terminator: The string that separates tables within one
        keyword block (default `"/"` — i.e. every block is one table).
        Pass `"//"` for the double-slash convention used by some keywords.

    The parsed value is `List[List[Dict]]` — a list of tables (one per
    keyword occurrence / PVT region), each table being a list of row dicts.
    For miscible tables the `primary_key` field is duplicated into every
    inner-table row for convenience.
    """

    __slots__ = ("columns", "primary_key")

    def __init__(
        self,
        name: str,
        columns: typing.Sequence[Field[Number]],
        *,
        primary_key: typing.Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.columns: typing.List[Field] = list(columns)
        self.primary_key = primary_key

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> typing.Optional[typing.List[PVTTable[Number]]]:
        records = deck.records_for(self.name)
        if not records:
            return None

        all_tables: typing.List[PVTTable] = []
        for record in records:
            if self.primary_key is not None:
                tables = self._parse_miscible(record.body)
            else:
                tables = [self._parse_flat(record.body)]
            all_tables.extend(tables)

        return all_tables or None

    def _row_from_tokens(self, tokens: typing.Sequence[str]) -> PVTRow[Number]:
        """
        Convert a token list to a row dict using `self.columns`.

        Trailing optional columns that are absent from `tokens` fall back
        to their declared `default` values.  The Eclipse `1*` default
        designator is treated as absent (falls back to `default`).
        """
        row: PVTRow = {}
        for idx, col in enumerate(self.columns):
            if idx < len(tokens):
                raw = tokens[idx]
                if raw == "1*":
                    # Eclipse default designator — use the column default.
                    if col.required:
                        raise DeckParseError(
                            f"{self.name}: required column {col.name!r} "
                            "has a default designator ('1*') but no default value."
                        )
                    row[col.name] = col.default
                else:
                    try:
                        row[col.name] = col.type(raw)
                    except ValueError as exc:
                        if col.required:
                            raise DeckParseError(
                                f"{self.name}: column {col.name!r} got invalid "
                                f"value {raw!r}: {exc}"
                            ) from exc
                        row[col.name] = col.default
            elif col.required:
                raise DeckParseError(
                    f"{self.name}: missing required column {col.name!r} "
                    f"(got {len(tokens)} token(s), need at least {idx + 1})."
                )
            else:
                row[col.name] = col.default
        return row

    def _parse_flat(self, body: str) -> PVTTable[Number]:
        """
        Parse a flat (immiscible/simple) table body: every `/`-delimited
        segment is one row.
        """
        table: PVTTable = []
        for segment in body.split("/"):
            tokens = tokenise(segment)
            if not tokens:
                continue
            table.append(self._row_from_tokens(tokens))
        return table

    def _parse_miscible(self, body: str) -> typing.List[PVTTable[Number]]:
        """
        Parse a miscible (PVTO/PVTG-style) keyword body.

        The convention is:
        - A segment with exactly one token that is a number acts as a
          primary-key introducer (e.g. Rs value for PVTO).
        - Subsequent segments (until the next single-token segment or an
          empty segment that follows a full table) are inner rows.

        Each primary-key value starts a new inner table; one keyword
        occurrence can contain multiple inner tables separated by blank `/`
        lines. The function returns a **list of tables** (one per
        outer-key group, concatenated across all `//`-separated blocks).
        """
        tables: typing.List[PVTTable] = []
        current_table: PVTTable = []
        current_pk_value: typing.Optional[float] = None
        pk_col_name = self.primary_key
        assert pk_col_name is not None  # guarded by caller

        segments = body.split("/")
        i = 0
        while i < len(segments):
            seg = segments[i]
            tokens = tokenise(seg)
            i += 1

            if not tokens:
                # An empty segment signals the end of the current inner
                # table group. If we have accumulated rows, save the table.
                if current_table:
                    tables.append(current_table)
                    current_table = []
                    current_pk_value = None
                continue

            if len(tokens) == 1:
                # Single numeric token -> primary-key introducer.
                try:
                    current_pk_value = float(tokens[0])
                except ValueError:
                    # Shouldn't happen in well-formed decks, but be lenient.
                    pass
                continue

            # Multiple tokens -> a data row. Prepend the primary-key column.
            row = self._row_from_tokens(tokens)
            if current_pk_value is not None:
                row[pk_col_name] = current_pk_value
            current_table.append(row)

        # Flush any remaining rows.
        if current_table:
            tables.append(current_table)

        return tables
