"""
Generic Eclipse / GRDECL deck parsing engine.

This module provides a small, composable framework for reading Eclipse-style
keyword decks (GRDECL include files, full `.DATA` decks, etc.) without each
keyword having to re-implement comment stripping, tokenisation, `N*value`
repeat expansion, or `BOX` / `EQUALS` / `ADD` / `MULTIPLY` / `COPY` operator
resolution.

**Pipeline**:

1. `_strip_comments` removes `--` line comments.
2. `_resolve_includes` recursively inlines `INCLUDE 'path' /` directives.
3. `Deck.scan` tokenises the resulting text into an ordered list of
   `Record` objects — one per keyword occurrence, in file order. This is a
   single linear pass; no per-keyword regex search over the whole file.
4. `DataFile` owns a `Deck` plus a registry of `Keyword` instances. Calling
   `data_file.get("NAME")` finds that keyword's record(s) in the deck,
   resolves any `BOX` / operator records that target it (in the order they
   appear, using the box scope active at that point in the file), and
   returns the fully-parsed, fully-operated-on value.  Results are cached.

**Keyword taxonomy**:

`GridArrayKeyword`
    A flat per-cell array of length `nx*ny*nz` (`PORO`, `PERMX`, `ACTNUM`,
    `MULTX`, ...). Supports `N*value` repeat syntax and is a valid target
    for `BOX` / `EQUALS` / `ADD` / `MULTIPLY` / `COPY` operators.

`RecordKeyword`
    A single fixed-arity record of mixed field types (`SPECGRID`,
    `MAPAXES`, `GRIDUNIT`, `PINCH`).

`RepeatedRecordKeyword`
    Zero or more `/`-terminated records sharing one field layout, optionally
    led by a quoted name (`FAULTS`, `MULTFLT`, `NNC`).

Each concrete keyword subclasses one of the above and only has to declare
*shape*: field names/types, expected array length, or nothing at all beyond
the base behaviour.  All cross-cutting deck mechanics (comments, includes,
repeats, operators, boxing, caching) live in `Deck` / `DataFile` and are
shared by every keyword for free.

**Adding a new keyword**:

```python
class Poro(GridArrayKeyword):
    def __init__(self) -> None:
        super().__init__("PORO", dtype=np.float64)

DataFile.DEFAULT_KEYWORDS.append(Poro())
```

That's it — `BOX`/`EQUALS`/`MULTIPLY`/etc. targeting `PORO` and `N*value`
repeat expansion are handled automatically because `GridArrayKeyword`
implements them once.
"""

import abc
import re
import typing
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "GridDimensions",
    "Record",
    "Deck",
    "Keyword",
    "GridArrayKeyword",
    "RecordKeyword",
    "RepeatedRecordKeyword",
    "DataFile",
    "DeckParseError",
]

T = typing.TypeVar("T")

_TextOrPath = typing.Union[str, bytes, Path]


class DeckParseError(ValueError):
    """Raised when an Eclipse deck or one of its keyword records is malformed."""


# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------


class GridDimensions(typing.NamedTuple):
    """
    Structured grid extent, as declared by `SPECGRID` (or `DIMENS`).

    Needed by every per-cell array keyword (for expected length / reshape)
    and by `BOX` / operator resolution (for IJK -> flat index mapping).
    """

    nx: int
    ny: int
    nz: int

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    def flat_index(self, i: int, j: int, k: int) -> int:
        """
        Convert 0-based `(i, j, k)` to a flat index in Eclipse's natural
        ordering (`i` fastest, `k` slowest): `index = i + j*nx + k*nx*ny`.
        """
        return i + j * self.nx + k * self.nx * self.ny


_COMMENT_RE = re.compile(r"--[^\n]*")
_INCLUDE_RE = re.compile(
    r"\bINCLUDE\b\s*['\"]([^'\"]+)['\"]\s*/", re.IGNORECASE | re.DOTALL
)


def _strip_comments(text: str) -> str:
    """Remove `--` line comments from Eclipse text."""
    return _COMMENT_RE.sub("", text)


def _resolve_includes(text: str, source_dir: typing.Optional[Path]) -> str:
    """
    Recursively inline `INCLUDE 'path' /` directives.

    :param text: Eclipse text that may contain `INCLUDE` directives.
    :param source_dir: Directory the current text was read from, used to
        resolve relative include paths. `None` disables resolution (raw
        text input with no filesystem anchor); directives are dropped with
        a warning in that case.
    :returns: Text with every `INCLUDE` block replaced by the referenced
        file's (recursively resolved) contents.
    :raises DeckParseError: If an included file cannot be found or read.
    """

    def _replace(match: "re.Match[str]") -> str:
        relative_path = match.group(1).strip()
        if source_dir is None:
            warnings.warn(
                f"INCLUDE directive for {relative_path!r} encountered in raw-text "
                "input with no source directory; ignoring. Load from a file path "
                "to enable INCLUDE resolution.",
                stacklevel=6,
            )
            return ""
        include_path = source_dir / relative_path
        if not include_path.is_file():
            raise DeckParseError(
                f"INCLUDE references {include_path!r}, which does not exist."
            )
        try:
            included_text = include_path.read_text(encoding="ascii", errors="replace")
        except OSError as exc:
            raise DeckParseError(
                f"Cannot read INCLUDE file {include_path!r}: {exc}"
            ) from exc
        return _resolve_includes(included_text, include_path.parent)

    return _INCLUDE_RE.sub(_replace, text)


def _resolve_source(source: _TextOrPath, *, encoding: str) -> str:
    """
    Coerce `source` (path, raw text, or bytes) to a single fully
    `INCLUDE`-resolved text blob.
    """
    source_dir: typing.Optional[Path] = None

    if isinstance(source, bytes):
        text = source.decode(encoding)
    elif isinstance(source, Path):
        source_dir = source.parent
        try:
            text = source.read_text(encoding=encoding)
        except OSError as exc:
            raise DeckParseError(f"Cannot read deck file {source!r}: {exc}") from exc
    else:
        candidate = Path(source)
        if candidate.is_file():
            source_dir = candidate.parent
            try:
                text = candidate.read_text(encoding=encoding)
            except OSError as exc:
                raise DeckParseError(
                    f"Cannot read deck file {source!r}: {exc}"
                ) from exc
        else:
            text = source

    return _resolve_includes(text, source_dir)


_REPEAT_RE = re.compile(r"^(\d+)\*(.*)$")
_QUOTED_RE = re.compile(r"""(['"])((?:(?!\1).)*)\1""")


def tokenise(text: str) -> typing.List[str]:
    """
    Split text into whitespace-separated tokens, expanding `N*value` repeat
    syntax and preserving quoted strings (with embedded whitespace) as
    single tokens.

    Examples:

        "100*0"        -> ["0"] * 100
        "3*1.5"        -> ["1.5", "1.5", "1.5"]
        "'MY FAULT' 1" -> ["MY FAULT", "1"]

    :param text: Comment-stripped Eclipse text.
    :returns: Flat list of expanded string tokens (quotes stripped).
    """
    # Pull out quoted substrings first so repeat-expansion / whitespace
    # splitting never sees their interior.
    placeholders: typing.List[str] = []

    def _stash(match: "re.Match[str]") -> str:
        placeholders.append(match.group(2))
        return f"\x00{len(placeholders) - 1}\x00"

    stashed_text = _QUOTED_RE.sub(_stash, text)

    tokens: typing.List[str] = []
    for raw_tok in stashed_text.split():
        m = _REPEAT_RE.match(raw_tok)
        if m:
            count, value = int(m.group(1)), m.group(2)
            tokens.extend([value] * count)
        else:
            tokens.append(raw_tok)

    def _unstash(tok: str) -> str:
        if tok.startswith("\x00") and tok.endswith("\x00"):
            try:
                return placeholders[int(tok[1:-1])]
            except (ValueError, IndexError):
                return tok
        return tok

    return [_unstash(tok) for tok in tokens]


class Record(typing.NamedTuple):
    """One keyword occurrence in a scanned deck, in file order."""

    keyword: str
    """Upper-cased keyword name as it appeared in the deck."""

    body: str
    """Raw text between the keyword and its terminating `/` (untokenised),
    or empty for a bare/nullary keyword that takes no data section."""

    start: int
    """Character offset of the keyword name in the source text."""

    end: int
    """Character offset just past the terminating `/` (or just past the
    keyword name itself, for a bare/nullary keyword)."""


# A keyword name: a bare uppercase token, 1-8 characters (Eclipse's own
# limit), optionally ending in '-' (MULTX-, MULTY-, MULTZ-), standing alone
# on its own line (only whitespace or end-of-line after it on that line).
# This is what distinguishes a keyword header from an ordinary data token,
# since Eclipse keywords always start a fresh line.
_KEYWORD_LINE_RE = re.compile(
    r"^[ \t]*(?P<keyword>[A-Z][A-Z0-9_]{0,7}-?)[ \t]*\r?$",
    re.MULTILINE,
)


class Deck:
    """
    A single linear scan of comment-stripped, include-resolved Eclipse text
    into an ordered sequence of `Record` objects.

    Scanning is performed once in `__init__`; every `DataFile.get()` call
    afterwards re-uses the same `records` list, so the cost of finding
    keyword and operator occurrences is paid exactly once per deck
    regardless of how many keywords are subsequently requested.

    **Scanning strategy**: Eclipse keywords always start at the beginning
    of a line, alone (nothing else on that line). A keyword's data section
    is everything from there up to and including its terminating `/` -
    *unless* another keyword line is encountered first, in which case the
    first keyword is a bare/nullary keyword (e.g. `ENDBOX`, `ECHO`,
    `RUNSPEC`) with an empty body, and the second keyword line starts a new
    record from that point. This mirrors Eclipse's own grammar without
    needing a hardcoded list of which keywords are nullary.
    """

    __slots__ = ("text", "records")

    def __init__(self, text: str) -> None:
        """
        :param text: Fully comment-stripped, `INCLUDE`-resolved Eclipse text.
        """
        self.text = text
        self.records: typing.List[Record] = self._scan(text)

    @staticmethod
    def _scan(text: str) -> typing.List[Record]:
        """
        Tokenise `text` into an ordered list of `Record` objects.

        :param text: Clean Eclipse text.
        :returns: Records in the order they appear in the file.
        """
        keyword_lines = list(_KEYWORD_LINE_RE.finditer(text))
        records: typing.List[Record] = []

        for idx, m in enumerate(keyword_lines):
            keyword = m.group("keyword").upper()
            body_start = m.end()
            # The next keyword line (if any) bounds how far this keyword's
            # body may extend, even if no '/' is found before it.
            next_keyword_start = (
                keyword_lines[idx + 1].start()
                if idx + 1 < len(keyword_lines)
                else len(text)
            )
            window = text[body_start:next_keyword_start]
            slash_pos = window.find("/")

            if slash_pos == -1:
                # No terminator before the next keyword line: this keyword
                # is bare/nullary (e.g. ENDBOX, ECHO, RUNSPEC, FIELD).
                records.append(
                    Record(keyword=keyword, body="", start=m.start(), end=body_start)
                )
            else:
                body = window[:slash_pos]
                end = body_start + slash_pos + 1
                records.append(
                    Record(keyword=keyword, body=body, start=m.start(), end=end)
                )

        return records

    def records_for(self, keyword: str) -> typing.List[Record]:
        """Return every record for `keyword`, in file order."""
        upper = keyword.upper()
        return [r for r in self.records if r.keyword == upper]

    def first_record_for(self, keyword: str) -> typing.Optional[Record]:
        """Return the first record for `keyword`, or `None` if absent."""
        upper = keyword.upper()
        for r in self.records:
            if r.keyword == upper:
                return r
        return None

    def has(self, keyword: str) -> bool:
        """Return whether `keyword` occurs anywhere in the deck."""
        upper = keyword.upper()
        return any(r.keyword == upper for r in self.records)


_BOX_KEYWORD = "BOX"
_ENDBOX_KEYWORD = "ENDBOX"
_OPERATOR_KEYWORDS: typing.FrozenSet[str] = frozenset(
    {"EQUALS", "ADD", "MULTIPLY", "COPY", "MAXVALUE", "MINVALUE"}
)
#: Keywords that scope or drive operators rather than holding their own
#: per-cell data; never matched as a plain `GridArrayKeyword` value.
OPERATOR_CONTROL_KEYWORDS: typing.FrozenSet[str] = (
    frozenset({_BOX_KEYWORD, _ENDBOX_KEYWORD}) | _OPERATOR_KEYWORDS
)


class _BoxOperation(typing.NamedTuple):
    """One resolved `EQUALS` / `ADD` / `MULTIPLY` / `COPY` / `MAXVALUE` / `MINVALUE` instruction."""

    op: str
    target: str
    value: typing.Optional[float]
    source: typing.Optional[str]
    box: typing.Tuple[int, int, int, int, int, int]
    order: typing.Tuple[int, int]
    """`(record_start, line_index)` - `record_start` is the character
    offset of the operator keyword's own `Record` (the same offset space
    used by `GridArrayKeyword._timeline` for its `"assign"` events, so the
    two event kinds sort correctly against each other), and `line_index`
    breaks ties between multiple operation lines within the same operator
    block (e.g. several `EQUALS` lines in one block, applied top to
    bottom)."""


def _default_box(dims: GridDimensions) -> typing.Tuple[int, int, int, int, int, int]:
    return (0, dims.nx - 1, 0, dims.ny - 1, 0, dims.nz - 1)


def _resolve_box_operations(
    deck: Deck, dims: GridDimensions
) -> typing.List[_BoxOperation]:
    """
    Walk every `BOX` / `ENDBOX` / `EQUALS` / `ADD` / `MULTIPLY` / `COPY` /
    `MAXVALUE` / `MINVALUE` record in deck order and resolve each operator
    record to a concrete `_BoxOperation` (target keyword, scalar value or
    source keyword, and the IJK box active at that point in the file).

    `BOX` updates the active box for all subsequent operator records until
    `ENDBOX` resets it to the full grid extent. The active box is *deck-order*
    state, independent of which keyword each operator record targets.

    :param deck: Pre-scanned `Deck`.
    :param dims: Grid extent, needed to clamp/resolve box bounds and the
        default (whole-grid) box.
    :returns: Resolved operations in file order (by `order`).
    """
    default_box = _default_box(dims)
    current_box = default_box
    operations: typing.List[_BoxOperation] = []

    for record in deck.records:
        if record.keyword == _BOX_KEYWORD:
            tokens = tokenise(record.body)
            if len(tokens) < 6:
                warnings.warn(
                    f"BOX record has {len(tokens)} token(s); expected 6 "
                    "(I1 I2 J1 J2 K1 K2). Ignoring.",
                    stacklevel=4,
                )
                continue
            try:
                i1, i2, j1, j2, k1, k2 = (int(t) - 1 for t in tokens[:6])
            except ValueError:
                warnings.warn(
                    f"BOX record has non-integer tokens {tokens[:6]!r}; ignoring.",
                    stacklevel=4,
                )
                continue
            i1, i2 = sorted(
                (max(0, min(i1, dims.nx - 1)), max(0, min(i2, dims.nx - 1)))
            )
            j1, j2 = sorted(
                (max(0, min(j1, dims.ny - 1)), max(0, min(j2, dims.ny - 1)))
            )
            k1, k2 = sorted(
                (max(0, min(k1, dims.nz - 1)), max(0, min(k2, dims.nz - 1)))
            )
            current_box = (i1, i2, j1, j2, k1, k2)

        elif record.keyword == _ENDBOX_KEYWORD:
            current_box = default_box

        elif record.keyword in _OPERATOR_KEYWORDS:
            for line_index, op in enumerate(
                _parse_operator_records(record.body, record.keyword, current_box)
            ):
                operations.append(op._replace(order=(record.start, line_index)))

    return operations


def _parse_operator_records(
    body: str, op: str, box: typing.Tuple[int, int, int, int, int, int]
) -> typing.Iterator[_BoxOperation]:
    """
    Parse the (possibly multi-record) body of a single `EQUALS` / `ADD` /
    `MULTIPLY` / `COPY` / `MAXVALUE` / `MINVALUE` block.

    Each line is one record: `TARGET value [box-override]` for `EQUALS`
    `/` `ADD` `/` `MULTIPLY` `/` `MAXVALUE` `/` `MINVALUE`, or
    `SOURCE TARGET [box-override]` for `COPY`. An optional trailing 6
    integers on the same record override the ambient `BOX` for that record
    only (standard Eclipse shorthand).

    :param body: Raw record body (between the operator keyword and its
        terminating `/`), which may itself contain embedded `/` characters
        separating individual records.
    :param op: Operator name (`"EQUALS"`, `"ADD"`, `"MULTIPLY"`, `"COPY"`,
        `"MAXVALUE"`, `"MINVALUE"`).
    :param box: Ambient box active at this point in the deck.
    :yields: One `_BoxOperation` per record line (with placeholder `order`).
    """
    for line in body.split("/"):
        tokens = tokenise(line)
        if not tokens:
            continue

        record_box = box
        if op == "COPY":
            if len(tokens) < 2:
                warnings.warn(
                    f"COPY record {tokens!r} needs a source and target keyword; skipping.",
                    stacklevel=6,
                )
                continue
            source, target = tokens[0].upper(), tokens[1].upper()
            value = None
            extra = tokens[2:]
        else:
            if len(tokens) < 2:
                warnings.warn(
                    f"{op} record {tokens!r} needs a target keyword and value; skipping.",
                    stacklevel=6,
                )
                continue
            target = tokens[0].upper()
            source = None
            try:
                value = float(tokens[1])
            except ValueError:
                warnings.warn(
                    f"{op} record for {target!r} has non-numeric value "
                    f"{tokens[1]!r}; skipping.",
                    stacklevel=6,
                )
                continue
            extra = tokens[2:]

        if len(extra) >= 6:
            try:
                i1, i2, j1, j2, k1, k2 = (int(t) - 1 for t in extra[:6])
                record_box = (i1, i2, j1, j2, k1, k2)
            except ValueError:
                pass  # fall back to ambient box

        yield _BoxOperation(
            op=op,
            target=target,
            value=value,
            source=source,
            box=record_box,
            order=0,
        )


def _apply_box_operation(
    array: npt.NDArray[np.float64],
    operation: _BoxOperation,
    dims: GridDimensions,
    resolve_target: typing.Callable[[str], typing.Optional[npt.NDArray[np.float64]]],
) -> None:
    """
    Apply one resolved `_BoxOperation` to `array` in place, over its box.

    :param array: Flat `(n_cells,)` array for the operation's target keyword
        (already validated to belong to that keyword).
    :param operation: The resolved operation to apply.
    :param dims: Grid extent (for IJK -> flat index mapping).
    :param resolve_target: Callback returning the current array for a given
        keyword name, used by `COPY` to fetch the source array. May return
        `None` if the source keyword has no data yet (treated as all-ones
        for multiplier-style arrays, all-zeros otherwise — callers decide).
    """
    i1, i2, j1, j2, k1, k2 = operation.box
    indices = np.array(
        [
            dims.flat_index(i, j, k)
            for k in range(k1, k2 + 1)
            for j in range(j1, j2 + 1)
            for i in range(i1, i2 + 1)
        ],
        dtype=np.intp,
    )

    if operation.op == "EQUALS":
        array[indices] = operation.value
    elif operation.op == "ADD":
        array[indices] += operation.value
    elif operation.op == "MULTIPLY":
        array[indices] *= operation.value
    elif operation.op == "MAXVALUE":
        array[indices] = np.maximum(array[indices], operation.value)
    elif operation.op == "MINVALUE":
        array[indices] = np.minimum(array[indices], operation.value)
    elif operation.op == "COPY":
        assert operation.source is not None
        source_array = resolve_target(operation.source)
        if source_array is None:
            warnings.warn(
                f"COPY source {operation.source!r} has no data; skipping "
                f"COPY into {operation.target!r}.",
                stacklevel=2,
            )
            return
        array[indices] = source_array[indices]


class Keyword(typing.Generic[T], abc.ABC):
    """
    Base class for a single supported Eclipse keyword.

    A `Keyword` is pure parsing/shape logic: given the `Record`(s) that
    belong to it (already located by `Deck`) plus the deck's resolved grid
    dimensions, produce the keyword's Python value. `Keyword` subclasses
    never search text themselves and never need to know about `BOX` /
    operator records targeting *other* keywords.
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
        :param dims: Resolved grid dimensions, or `None` if `SPECGRID` /
            `DIMENS` has not been parsed (some keywords, e.g. `MAPAXES`,
            don't need it; array keywords will raise if it's required and
            missing).
        :returns: The parsed value, or `None` if the keyword is absent
            from the deck.
        """
        ...

    def is_present(self, deck: Deck) -> bool:
        """Return whether this keyword occurs anywhere in `deck`."""
        return deck.has(self.name)


class GridArrayKeyword(Keyword[npt.NDArray[np.float64]]):
    """
    A flat per-cell array keyword of length `nx * ny * nz` (`PORO`,
    `PERMX`, `ACTNUM`, `MULTX`, `TOPS`, ...).

    Handles `N*value` repeat expansion and is automatically a valid target
    for `BOX` / `EQUALS` / `ADD` / `MULTIPLY` / `COPY` / `MAXVALUE` /
    `MINVALUE` operator records anywhere in the deck — operator resolution
    is shared across all `GridArrayKeyword` instances via `DataFile`, so no
    individual keyword subclass needs to implement it.

    If the keyword itself never appears as a standalone data block but
    *is* the target of one or more `EQUALS`/operator records (a common
    Eclipse idiom for keywords like `MULTX` that are only ever set via
    `BOX`/`EQUALS`), the array still starts from `default_value` and the
    operators are applied on top of it.
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
        :param dtype: Output array dtype.
        :param default_value: Value cells start at before any explicit data
            block or operator is applied. Multiplier-style keywords
            (`MULTX`, ...) should use `default_value=1.0` via
            `is_multiplier=True`.
        :param is_multiplier: Convenience flag; when `True`, overrides
            `default_value` to `1.0` (the multiplicative identity) unless
            `default_value` was explicitly passed as something else.
        """
        super().__init__(name)
        self.dtype = np.dtype(dtype)
        self.default_value = 1.0 if is_multiplier else default_value
        self.is_multiplier = is_multiplier

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[npt.NDArray[np.float64]]:
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
    ) -> typing.Optional[npt.NDArray[np.float64]]:
        """
        Replay every event affecting this keyword's array - its own
        explicit data block(s) *and* every `BOX` / operator record
        targeting it - strictly in the order they occur in the deck.

        This is what makes e.g. `MULTX` set via `EQUALS` and later
        overwritten by an explicit `MULTX ... /` block (or vice versa)
        behave like Eclipse: whichever event is last in the file wins for
        the cells it touches, rather than "data block always wins" or
        "operators always win".

        :param deck: Scanned deck.
        :param dims: Resolved grid dimensions.
        :param stop_before_order: If given, only replay events with
            `order < stop_before_order`. Used by `COPY` to resolve a
            source keyword's value *as of* the point in the file where the
            `COPY` record itself occurs, rather than its eventual final
            value.
        :returns: The resolved array, or `None` if this keyword has no
            data block and no operator ever targets it.
        """
        events = self._timeline(deck, dims)
        if stop_before_order is not None:
            events = [e for e in events if e[0] < stop_before_order]
        if not events:
            return None

        array = np.full(dims.n_cells, self.default_value, dtype=np.float64)

        def _resolve_source(
            keyword_name: str, as_of_order: typing.Tuple[int, int]
        ) -> typing.Optional[npt.NDArray[np.float64]]:
            default = 1.0 if keyword_name.startswith("MULT") else 0.0
            probe = GridArrayKeyword(keyword_name, default_value=default)
            return probe._resolve(deck, dims, stop_before_order=as_of_order)

        for order, kind, payload in events:
            if kind == "assign":
                tokens = typing.cast(typing.List[str], payload)
                if len(tokens) == 1:
                    array[:] = float(tokens[0])
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
                operation = typing.cast(_BoxOperation, payload)
                _apply_box_operation(
                    array,
                    operation,
                    dims,
                    resolve_target=lambda name, _order=order: _resolve_source(
                        name, _order
                    ),
                )

        return array.astype(self.dtype, copy=False)

    def _timeline(
        self, deck: Deck, dims: GridDimensions
    ) -> typing.List[typing.Tuple[typing.Tuple[int, int], str, typing.Any]]:
        """
        Build the ordered `(order, kind, payload)` events affecting this
        keyword: `("assign", tokens)` for each of its own explicit
        data-block records, interleaved with `("operate", _BoxOperation)`
        for every operator record that targets it.

        Both event kinds share one `order` namespace: a `(record_start,
        line_index)` tuple, where `record_start` is the character offset
        of the owning `Record` in the source text (so a data-block assign
        and an operator record sort correctly against each other by
        position in the file) and `line_index` is `0` for data-block
        assigns or the within-block line index for operator records (so
        several `EQUALS` lines inside one block apply top-to-bottom).

        :returns: Events sorted ascending (earliest in file first).
        """
        events: typing.List[typing.Tuple[typing.Tuple[int, int], str, typing.Any]] = []
        for record in deck.records_for(self.name):
            events.append(((record.start, 0), "assign", tokenise(record.body)))
        for op in _resolve_box_operations(deck, dims):
            if op.target == self.name:
                events.append((op.order, "operate", op))
        events.sort(key=lambda e: e[0])
        return events


class Field(typing.NamedTuple):
    """One positional field in a `RecordKeyword` or `RepeatedRecordKeyword` layout."""

    name: str
    type: typing.Callable[[str], typing.Any]
    required: bool = True
    default: typing.Any = None


class RecordKeyword(Keyword[typing.Dict[str, typing.Any]]):
    """
    A keyword holding exactly one fixed-layout record of mixed-type fields
    (`SPECGRID`, `MAPAXES`, `GRIDUNIT`, `PINCH`).

    Subclasses declare `fields: Sequence[Field]` describing each positional
    token; `parse` returns a `dict` keyed by field name. Trailing optional
    fields (`required=False`) fall back to their `default` when the deck
    record has fewer tokens than declared fields.
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
        result: typing.Dict[str, typing.Any] = {}
        for idx, field in enumerate(self.fields):
            if idx < len(tokens):
                raw = tokens[idx]
                try:
                    result[field.name] = field.type(raw)
                except ValueError as exc:
                    if field.required:
                        raise DeckParseError(
                            f"{self.name}: field {field.name!r} got invalid "
                            f"value {raw!r}: {exc}"
                        ) from exc
                    result[field.name] = field.default
            elif field.required:
                raise DeckParseError(
                    f"{self.name}: missing required field {field.name!r} "
                    f"(got {len(tokens)} token(s), expected at least {idx + 1})."
                )
            else:
                result[field.name] = field.default
        return result


class RepeatedRecordKeyword(Keyword[typing.List[typing.Dict[str, typing.Any]]]):
    """
    A keyword whose body holds zero or more individually `/`-terminated
    records sharing one field layout (`FAULTS`, `MULTFLT`, `NNC`).

    Multiple occurrences of the keyword in the deck (e.g. several `FAULTS`
    blocks) are concatenated, in file order.
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
        return results

    def _parse_tokens(
        self, tokens: typing.Sequence[str]
    ) -> typing.Dict[str, typing.Any]:
        result: typing.Dict[str, typing.Any] = {}
        for idx, field in enumerate(self.fields):
            if idx < len(tokens):
                raw = tokens[idx]
                try:
                    result[field.name] = field.type(raw)
                except ValueError as exc:
                    if field.required:
                        raise DeckParseError(
                            f"{self.name}: field {field.name!r} got invalid "
                            f"value {raw!r}: {exc}"
                        ) from exc
                    result[field.name] = field.default
            elif field.required:
                raise DeckParseError(
                    f"{self.name}: record {tokens!r} is missing required "
                    f"field {field.name!r}."
                )
            else:
                result[field.name] = field.default
        return result


class Specgrid(RecordKeyword):
    """`SPECGRID NX NY NZ NUMRES GFLAG /` — structured grid dimensions."""

    def __init__(self) -> None:
        super().__init__(
            "SPECGRID",
            fields=[
                Field("nx", int),
                Field("ny", int),
                Field("nz", int),
                Field("numres", int, required=False, default=1),
                Field("gflag", str, required=False, default="F"),
            ],
        )


class Dimens(RecordKeyword):
    """`DIMENS NX NY NZ /` — structured grid dimensions (non-corner-point decks)."""

    def __init__(self) -> None:
        super().__init__(
            "DIMENS", fields=[Field("nx", int), Field("ny", int), Field("nz", int)]
        )


class Mapaxes(RecordKeyword):
    """`MAPAXES X0 Y0 X1 Y1 X2 Y2 /` — map coordinate system axes."""

    def __init__(self) -> None:
        super().__init__(
            "MAPAXES",
            fields=[
                Field("y_axis_x", float),
                Field("y_axis_y", float),
                Field("origin_x", float),
                Field("origin_y", float),
                Field("x_axis_x", float),
                Field("x_axis_y", float),
            ],
        )


class Gridunit(RecordKeyword):
    """`GRIDUNIT 'UNIT' 'TYPE' /` — geometry length unit declaration."""

    def __init__(self) -> None:
        super().__init__(
            "GRIDUNIT",
            fields=[
                Field("unit", str),
                Field("type", str, required=False, default=""),
            ],
        )


class Mapunits(RecordKeyword):
    """`MAPUNITS 'UNIT' /` — map coordinate unit declaration."""

    def __init__(self) -> None:
        super().__init__("MAPUNITS", fields=[Field("unit", str)])


class Pinch(RecordKeyword):
    """`PINCH thickness ... /` — pinchout thickness tolerance."""

    def __init__(self) -> None:
        super().__init__(
            "PINCH",
            fields=[Field("thickness", float, required=False, default=1e-6)],
        )


class Coord(Keyword[npt.NDArray[np.float64]]):
    """`COORD` — corner-point pillar array, shape `((nx+1)*(ny+1), 6)`."""

    def __init__(self) -> None:
        super().__init__("COORD")

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[npt.NDArray[np.float64]]:
        if dims is None:
            raise DeckParseError("COORD requires grid dimensions (SPECGRID).")
        record = deck.first_record_for(self.name)
        if record is None:
            return None
        tokens = tokenise(record.body)
        expected = (dims.nx + 1) * (dims.ny + 1) * 6
        if len(tokens) != expected:
            raise DeckParseError(
                f"COORD expected {expected} values for a {dims.nx}x{dims.ny} "
                f"grid; got {len(tokens)}."
            )
        return (
            np.array(tokens, dtype=np.float64)
            .reshape(dims.nx + 1, dims.ny + 1, 6, order="F")
            .transpose(1, 0, 2)  # -> (ny+1, nx+1, 6) C-order
        )


class Zcorn(Keyword[npt.NDArray[np.float64]]):
    """`ZCORN` — corner-point depth array, shape `(nz*2, ny*2, nx*2)`."""

    def __init__(self) -> None:
        super().__init__("ZCORN")

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[npt.NDArray[np.float64]]:
        if dims is None:
            raise DeckParseError("ZCORN requires grid dimensions (SPECGRID).")
        record = deck.first_record_for(self.name)
        if record is None:
            return None
        tokens = tokenise(record.body)
        expected = dims.nx * dims.ny * dims.nz * 8
        if len(tokens) != expected:
            raise DeckParseError(
                f"ZCORN expected {expected} values for a "
                f"{dims.nx}x{dims.ny}x{dims.nz} grid; got {len(tokens)}."
            )
        return (
            np.array(tokens, dtype=np.float64)
            .reshape(dims.nx * 2, dims.ny * 2, dims.nz * 2, order="F")
            .transpose(2, 1, 0)  # -> (nz*2, ny*2, nx*2) C-order
        )


class Faults(RepeatedRecordKeyword):
    """`FAULTS 'NAME' I1 I2 J1 J2 K1 K2 FACE / ... /` — named fault planes."""

    _VALID_FACES = frozenset({"I", "I-", "J", "J-", "K", "K-"})

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
                f"face direction {face!r}. Valid values: {sorted(self._VALID_FACES)}."
            )
        result["face"] = face
        return result


class Multflt(RepeatedRecordKeyword):
    """`MULTFLT 'NAME' MULTIPLIER / ... /` — per-fault transmissibility multiplier."""

    def __init__(self) -> None:
        super().__init__(
            "MULTFLT",
            fields=[Field("name", str), Field("multiplier", float)],
        )

    def parse(
        self, deck: Deck, dims: typing.Optional[GridDimensions]
    ) -> typing.Optional[typing.List[typing.Dict[str, typing.Any]]]:
        records = super().parse(deck, dims)
        if records is None:
            return None
        # Eclipse semantics: last value for a given fault name wins.
        by_name: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        for rec in records:
            by_name[rec["name"]] = rec
        return list(by_name.values())


class Nnc(RepeatedRecordKeyword):
    """`NNC I1 J1 K1 I2 J2 K2 T / ... /` — explicit non-neighbour connections."""

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


#: Keywords resolved before anything else, since other keywords' shapes
#: depend on them.
_DIMENSION_KEYWORDS: typing.Tuple[str, ...] = ("SPECGRID", "DIMENS")


class DataFile:
    """
    A parsed Eclipse deck: comment-stripped, `INCLUDE`-resolved text plus a
    registry of `Keyword` objects, exposed through `get(name)`.

    Grid dimensions (`SPECGRID` or `DIMENS`) are resolved once, eagerly, at
    construction time, because every `GridArrayKeyword` and the corner-point
    keywords (`COORD`, `ZCORN`) need them to know expected array lengths and
    to resolve `BOX` operator scopes.

    Usage::

        data_file = DataFile(text, keywords=[Poro(), PermX(), Faults(), ...])
        poro = data_file.get("PORO")
        faults = data_file.get("FAULTS")
    """

    DEFAULT_KEYWORDS: typing.ClassVar[typing.List[Keyword[typing.Any]]] = [
        Specgrid(),
        Dimens(),
        Mapaxes(),
        Gridunit(),
        Mapunits(),
        Pinch(),
        Coord(),
        Zcorn(),
        Faults(),
        Multflt(),
        Nnc(),
    ]

    __slots__ = ("_deck", "_keywords", "_cache", "dimensions")

    def __init__(
        self,
        source: _TextOrPath,
        *,
        keywords: typing.Optional[typing.Sequence[Keyword[typing.Any]]] = None,
        encoding: str = "ascii",
    ) -> None:
        """
        :param source: A file path, raw deck text, or raw deck bytes.
        :param keywords: Extra `Keyword` instances to register in addition
            to `DEFAULT_KEYWORDS`. Keywords with the same `name` as a
            default override it.
        :param encoding: Text encoding used when reading from a path or
            decoding `bytes`.
        """
        text = _resolve_source(source, encoding=encoding)
        clean_text = _strip_comments(text)
        self._deck = Deck(clean_text)

        self._keywords: typing.Dict[str, Keyword[typing.Any]] = {
            kw.name: kw for kw in self.DEFAULT_KEYWORDS
        }
        if keywords:
            self._keywords.update({kw.name: kw for kw in keywords})

        self._cache: typing.Dict[str, typing.Any] = {}
        self.dimensions = self._resolve_dimensions()

    def _resolve_dimensions(self) -> typing.Optional[GridDimensions]:
        """
        Resolve grid extent from `SPECGRID` (preferred) or `DIMENS`.

        :returns: A `GridDimensions`, or `None` if neither keyword is
            present (decks consisting only of, e.g., PVT/relperm tables).
        """
        for name in _DIMENSION_KEYWORDS:
            record = self._deck.first_record_for(name)
            if record is None:
                continue

            keyword = self._keywords.get(name)
            if keyword is None:
                continue

            parsed = keyword.parse(self._deck, None)
            if parsed is None:
                continue
            return GridDimensions(
                nx=int(parsed["nx"]), ny=int(parsed["ny"]), nz=int(parsed["nz"])
            )
        return None

    @property
    def deck(self) -> Deck:
        """The underlying scanned `Deck`."""
        return self._deck

    def add_keywords(self, *keywords: Keyword[typing.Any]) -> None:
        """
        Register additional keywords (or override existing ones by name).

        Clears any cached value for a keyword name that is being replaced.
        """
        for keyword in keywords:
            self._keywords[keyword.name] = keyword
            self._cache.pop(keyword.name, None)

    def is_present(self, name: str) -> bool:
        """Return whether `name` occurs anywhere in the deck."""
        return self._deck.has(name.upper())

    def get(
        self, name: str, /, *, use_cache: bool = True
    ) -> typing.Optional[typing.Any]:
        """
        Parse and return the value of keyword `name`.

        :param name: Keyword name, case-insensitive.
        :param use_cache: Whether to read/write the per-`DataFile` value
            cache. Disable for keywords whose value should never be reused
            (rarely needed; mainly for testing).
        :returns: The keyword's parsed value, or `None` if it is absent
            from the deck or not registered.
        """
        key = name.upper()
        if key not in self._keywords:
            return None

        if use_cache and key in self._cache:
            return self._cache[key]

        keyword = self._keywords[key]
        value = keyword.parse(self._deck, self.dimensions)
        if use_cache:
            self._cache[key] = value
        return value

    def get_many(
        self, *names: str, use_cache: bool = True
    ) -> typing.Dict[str, typing.Optional[typing.Any]]:
        """Convenience batch form of `get`, returned as `{name: value}`."""
        return {name: self.get(name, use_cache=use_cache) for name in names}
