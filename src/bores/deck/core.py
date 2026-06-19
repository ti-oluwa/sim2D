"""Eclipse / GRDECL deck core: scanning, tokenisation, and grid dimensions."""

import re
import typing
import warnings
from pathlib import Path

__all__ = [
    "GridDimensions",
    "Record",
    "Deck",
    "DeckParseError",
    "tokenise",
]

_TextOrPath = typing.Union[str, bytes, Path]


class DeckParseError(ValueError):
    """Raised when an Eclipse deck or one of its keyword records is malformed."""


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
        """Total number of cells `nx * ny * nz`."""
        return self.nx * self.ny * self.nz

    def flat_index(self, i: int, j: int, k: int) -> int:
        """
        Convert 0-based `(i, j, k)` to a flat index in Eclipse's natural
        ordering (`i` fastest, `k` slowest):
        `index = i + j*nx + k*nx*ny`.

        :param i: 0-based x index.
        :param j: 0-based y index.
        :param k: 0-based z index.
        :returns: Flat cell index.
        """
        return i + j * self.nx + k * self.nx * self.ny

    def __iter__(self) -> typing.Iterator[int]:
        yield self.nx
        yield self.ny
        yield self.nz


_COMMENT_RE = re.compile(r"--[^\n]*")
_INCLUDE_RE = re.compile(
    r"\bINCLUDE\b\s*['\"]([^'\"]+)['\"]\s*/", re.IGNORECASE | re.DOTALL
)


def strip_comments(text: str) -> str:
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
                "input with no source directory; ignoring.  Load from a file path "
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


def resolve_source(source: _TextOrPath, *, encoding: str) -> str:
    """
    Coerce `source` (path, raw text, or bytes) to a single fully
    `INCLUDE`-resolved, comment-stripped text blob.

    :param source: Path, raw deck text string, or raw bytes.
    :param encoding: Character encoding for file/bytes input.
    :returns: Clean text ready for `Deck` scanning.
    :raises DeckParseError: If a file cannot be read.
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


# Matches "N*value" repeat syntax. Value must be non-empty; a bare "N*" is
# treated as N repetitions of the empty string and rejected by callers that
# call float() on the result. We keep the permissive match here and let
# callers produce a meaningful error message.
_REPEAT_RE = re.compile(r"^(\d+)\*(.*)$")
_QUOTED_RE = re.compile(r"""(['"])((?:(?!\1).)*)\1""")


def tokenise(text: str) -> typing.List[str]:
    """
    Split text into whitespace-separated tokens, expanding `N*value`
    repeat syntax and preserving quoted strings (with embedded whitespace)
    as single tokens.

    Examples:

        "100*0"        -> ["0"] * 100
        "3*1.5"        -> ["1.5", "1.5", "1.5"]
        "'MY FAULT' 1" -> ["MY FAULT", "1"]

    Note:
        A bare `"N*"` (empty value) expands to `N` empty strings.
        Callers that convert tokens to `float` will receive a
        `ValueError`; they should surface this as a
        `DeckParseError` with context about which keyword failed.

    :param text: Comment-stripped Eclipse text.
    :returns: Flat list of expanded string tokens (quotes stripped).
    """
    # Stash quoted substrings so that repeat-expansion and whitespace
    # splitting never see their interior.
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
    """
    Raw text between the keyword and its terminating `/` (untokenised),
    or empty string for a bare/nullary keyword that takes no data section.
    """

    start: int
    """Character offset of the keyword name in the source text."""

    end: int
    """
    Character offset just past the terminating `/` (or just past the
    keyword name itself for a bare/nullary keyword).
    """


# A keyword name is a bare uppercase token, 1-8 characters (Eclipse's own
# limit), optionally ending in `-` (`MULTX-`, `MULTY-`, `MULTZ-`),
# standing alone on its own line (only whitespace or end-of-line after it).
# The underscore in `[A-Z0-9_]` is intentional because some simulator extensions
# use keywords like `COORD_V`.
_KEYWORD_LINE_RE = re.compile(
    r"^[ \t]*(?P<keyword>[A-Z][A-Z0-9_]{0,7}-?)[ \t]*\r?$",
    re.MULTILINE,
)


class Deck:
    """
    A single linear scan of comment-stripped, include-resolved Eclipse text
    into an ordered sequence of `Record` objects.

    Scanning is performed once on initiaization; every subsequent lookup
    re-uses the same `records` list, so the cost of finding keyword and
    operator occurrences is paid exactly once per deck regardless of how
    many keywords are subsequently requested.

    **Scanning strategy**: Eclipse keywords always start at the beginning
    of a line, alone (nothing else on that line). A keyword's data section
    is everything from there up to and including its terminating `/` -
    *unless* another keyword line is encountered first, in which case the
    first keyword is treated as bare/nullary (e.g. `ENDBOX`, `ECHO`,
    `RUNSPEC`) with an empty body, and the second keyword line starts a
    new record. This mirrors Eclipse's own grammar without needing a
    hardcoded list of nullary keywords.

    Note:
        If the **last** keyword in a deck has data but no terminating
        `/`, that data is silently dropped (the keyword becomes nullary).
        Well-formed Eclipse decks always terminate data blocks with `/`;
        this edge case only arises with truncated or hand-written files.
    """

    __slots__ = ("text", "records")

    def __init__(self, text: str) -> None:
        """
        :param text: Fully comment-stripped, `INCLUDE`-resolved Eclipse
            text.
        """
        self.text = text
        self.records: typing.List[Record] = self._scan(text)

    @staticmethod
    def _scan(text: str) -> typing.List[Record]:
        """
        Tokenise `text` into an ordered list of `Record` objects.

        :param text: Clean Eclipse text (no comments, no INCLUDE
            directives).
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
                # No terminator before the next keyword line (or end of
                # file): treat as bare/nullary keyword.
                records.append(
                    Record(
                        keyword=keyword,
                        body="",
                        start=m.start(),
                        end=body_start,
                    )
                )
            else:
                body = window[:slash_pos]
                end = body_start + slash_pos + 1
                records.append(
                    Record(
                        keyword=keyword,
                        body=body,
                        start=m.start(),
                        end=end,
                    )
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
