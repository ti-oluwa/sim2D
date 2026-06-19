"""
GRDECL text-format reader and writer.

GRDECL is the ASCII keyword-based format used by Eclipse, ResInsight, and
most other reservoir simulators to describe corner-point pillar grids.

**Supported keywords (read)**:

`SPECGRID`, `COORD`, `ZCORN`, `ACTNUM`, `GRIDTYPE`,
`TOPS`, `DX`, `DY`, `DZ`,
`MAPAXES`, `GRIDUNIT`, `MAPUNIT`,
`PINCH`, `PINCHOUT`,
`NNC`,
`FAULTS`, `MULTFLT`.

**Supported keywords (write)**:

`SPECGRID`, `COORD`, `ZCORN`, `ACTNUM` (corner-point grids);
`SPECGRID`, `TOPS`, `DX`, `DY`, `DZ`, `ACTNUM` (Cartesian grids).

**NNC parsing**:

The `NNC` keyword block is expected in the form:

    NNC
      I1 J1 K1  I2 J2 K2  T
      ...
    /

where `(I, J, K)` are 1-based structured cell indices and `T` is the
transmissibility in the grid's unit system.  Each group of 7 tokens forms
one record.

**Fault parsing**:

`FAULTS` records have the form:

    FAULTS
      'NAME'  I1 I2  J1 J2  K1 K2  FACE /
      ...
    /

where `FACE` is one of `I`, `I-`, `J`, `J-`, `K`, `K-`.

`MULTFLT` records have the form:

    MULTFLT
      'NAME'  MULTIPLIER /
      ...
    /
"""

import re
import typing
import warnings
from pathlib import Path

import attrs
import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.errors import GridExportError, GridImportError
from bores.grids.base import Grid
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import (
    ActnumArray,
    CoordArray,
    FaultRecord,
    ZcornArray,
    make_corner_point_grid,
    rederive_corner_point_arrays,
)
from bores.grids.utils import _get_length_conversion_factor, convert
from bores.typing import (
    FloatArray,
    IntArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["load_grdecl", "dump_grdecl"]


_PathOrStr = typing.Union[str, Path]
_TextOrPath = typing.Union[str, bytes, Path]


@attrs.define(frozen=True, slots=True)
class MapAxes:
    """
    Map coordinate system axes parsed from the GRDECL `MAPAXES` keyword.

    Stores the map origin and two unit-vector axis points so that grid
    coordinates can be rotated into the map CRS.

    :param origin: Shape `(2,)` - `(X, Y)` map coordinate origin.
    :param map_x_axis_point: Shape `(2,)` - point on the map X-axis.
    :param map_y_axis_point: Shape `(2,)` - point on the map Y-axis.
    :param unit_system: Unit system in which the axis coordinates are
        expressed.
    """

    origin: FloatArray[OneDimension]
    map_x_axis_point: FloatArray[OneDimension]
    map_y_axis_point: FloatArray[OneDimension]
    unit_system: UnitSystem = UnitSystem.FIELD
    rotation_matrix: FloatArray[ThreeDimensions] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "rotation_matrix", self._compute_rotation_matrix())

    def _compute_rotation_matrix(self) -> FloatArray[TwoDimensions]:
        origin = self.origin
        x_dir = self.map_x_axis_point - origin
        x_dir /= np.linalg.norm(x_dir)
        y_dir = self.map_y_axis_point - origin
        y_dir /= np.linalg.norm(y_dir)
        return np.array(
            [[x_dir[0], y_dir[0]], [x_dir[1], y_dir[1]]],
            dtype=np.float64,
        )

    def convert(self, to: UnitSystem) -> Self:
        """
        Return a new `MapAxes` with all coordinates expressed in `to`.

        :param to: Target `UnitSystem`.
        :returns: New `MapAxes` in the target unit system, or `self` if
            already in the target system.
        """
        if self.unit_system == to:
            return self
        factor = _get_length_conversion_factor(self.unit_system, to)
        return self.__class__(
            origin=self.origin * factor,
            map_x_axis_point=self.map_x_axis_point * factor,
            map_y_axis_point=self.map_y_axis_point * factor,
            unit_system=to,
        )


@typing.overload
def load_grdecl(
    source: Path,
    *,
    encoding: str = ...,
    unit_system: typing.Optional[UnitSystem] = ...,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = ...,
) -> Grid: ...


@typing.overload
def load_grdecl(
    source: str,
    *,
    encoding: str = ...,
    unit_system: typing.Optional[UnitSystem] = ...,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = ...,
) -> Grid: ...


@typing.overload
def load_grdecl(
    source: bytes,
    *,
    encoding: str = ...,
    unit_system: typing.Optional[UnitSystem] = ...,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = ...,
) -> Grid: ...


def load_grdecl(
    source: _TextOrPath,
    *,
    encoding: str = "ascii",
    unit_system: typing.Optional[UnitSystem] = None,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Load a GRDECL corner-point grid from a file path, raw string, or bytes.

    Automatically detects whether `source` is a filesystem path or raw
    GRDECL text / bytes. Recursively resolves `INCLUDE` directives when
    loading from a file path.

    Parsed keyword coverage:

    - Grid dimensions: `SPECGRID`
    - Geometry:        `COORD`, `ZCORN`, `TOPS`, `DX`, `DY`, `DZ`
    - Activity:        `ACTNUM`
    - Units:           `GRIDUNIT`, `FIELD`, `METRIC`, `LAB`, `SI`
    - Map CRS:         `MAPAXES`, `MAPUNITS`
    - Pinchouts:       `PINCH`, `PINCHOUT`
    - Connections:     `NNC` (with transmissibilities)
    - Faults:          `FAULTS`, `MULTFLT`

    :param source: One of:

        - `pathlib.Path` - path to a `.grdecl` file.
        - `str` - filesystem path *or* raw GRDECL text content.
        - `bytes` - raw GRDECL content encoded as `encoding`.

    :param encoding: Text encoding used when decoding `bytes` / file
        input (default `"ascii"`).
    :param unit_system: If provided, the returned `Grid` is converted
        to this unit system after loading.
    :param metadata: Optional extra key/value pairs merged into the
        `Grid.metadata` dict.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords (`SPECGRID`, `COORD`
        or `TOPS` + `DX`/`DY`/`DZ`, `ZCORN`) are missing or
        malformed.
    """
    text = _resolve_source(source, encoding=encoding)
    grid = _parse_grdecl(text, metadata=metadata)
    return convert(grid, to=unit_system) if unit_system is not None else grid


@typing.overload
def dump_grdecl(
    grid: Grid,
    destination: Path,
    *,
    actnum: typing.Optional[ActnumArray] = ...,
    encoding: str = ...,
) -> None: ...


@typing.overload
def dump_grdecl(
    grid: Grid,
    destination: None = None,
    *,
    actnum: typing.Optional[ActnumArray] = ...,
    encoding: str = ...,
) -> str: ...


@typing.overload
def dump_grdecl(
    grid: Grid,
    destination: _PathOrStr,
    *,
    actnum: typing.Optional[ActnumArray] = ...,
    encoding: str = ...,
) -> None: ...


def dump_grdecl(
    grid: Grid,
    destination: typing.Union[_PathOrStr, None] = None,
    *,
    actnum: typing.Optional[ActnumArray] = None,
    encoding: str = "ascii",
) -> typing.Optional[str]:
    """
    Serialise a `bores.grids.base.Grid` to GRDECL text format.

    .. note:
        GRDECL is a corner-point format.  Grids from other factories
        (Voronoi, tetrahedral, …) are approximated by writing their cell
        bounding-box vertices as degenerate hexahedral cells; round-trip
        fidelity is limited to cell-level geometry.

    :param grid: The grid to serialise.
    :param destination: One of:

        - `pathlib.Path` or `str` path - write to file, return `None`.
        - `None` - return the GRDECL text as a `str`.

    :param actnum: Optional shape `(n_cells,)` integer array of
        active-cell flags (1 = active, 0 = inactive).  If `None`, all
        cells are written as active.
    :param encoding: Encoding used when writing to a file (default
        `"ascii"`).
    :returns: GRDECL text as a `str` when `destination` is `None`;
        `None` otherwise.
    :raises GridExportError: If the grid cannot be serialised.
    """
    try:
        text = _build_grdecl_text(grid, actnum=actnum)
    except Exception as exc:
        raise GridExportError(f"Failed to serialise grid to GRDECL: {exc}") from exc

    if destination is None:
        return text

    path = Path(destination)
    path.write_text(text, encoding=encoding)
    return None


def _resolve_source(source: _TextOrPath, *, encoding: str) -> str:
    """
    Coerce `source` to a plain text string and inline any `INCLUDE`
    directives.

    :param source: Path, raw string, or bytes.
    :param encoding: Byte-decoding encoding.
    :returns: Raw GRDECL text as a `str`.
    :raises GridImportError: If a filesystem path cannot be read.
    """
    source_dir: typing.Optional[Path] = None

    if isinstance(source, bytes):
        text = source.decode(encoding)

    elif isinstance(source, Path):
        source_dir = source.parent
        try:
            text = source.read_text(encoding=encoding)
        except OSError as exc:
            raise GridImportError(f"Cannot read GRDECL file {source!r}: {exc}") from exc

    else:
        candidate = Path(source)
        source_dir = candidate.parent
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding=encoding)
            except OSError as exc:
                raise GridImportError(
                    f"Cannot read GRDECL file {source!r}: {exc}"
                ) from exc
        else:
            text = source

    return _resolve_includes(text, source_dir=source_dir)


def _resolve_includes(text: str, source_dir: typing.Optional[Path] = None) -> str:
    """
    Recursively inline `INCLUDE` directives in GRDECL text.

    Eclipse `INCLUDE` syntax:

        INCLUDE
            'relative/path/to/file.grdecl' /

    Paths are resolved relative to `source_dir`.  When `source_dir` is
    `None` (raw text input), `INCLUDE` directives are dropped with a
    warning because there is no filesystem anchor.

    :param text: GRDECL text that may contain `INCLUDE` directives.
    :param source_dir: Directory of the file being parsed, or `None`.
    :returns: Text with all `INCLUDE` blocks replaced by file contents.
    :raises GridImportError: If an included file cannot be read.
    """
    include_pattern = re.compile(
        r"\bINCLUDE\b\s*['\"]([^'\"]+)['\"]\s*/",
        re.IGNORECASE | re.DOTALL,
    )

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        relative_path = match.group(1).strip()
        if source_dir is None:
            warnings.warn(
                f"GRDECL INCLUDE directive for {relative_path!r} encountered in "
                "raw-text input (no source directory available).  The directive "
                "will be ignored.  Load from a file path to enable INCLUDE "
                "resolution.",
                stacklevel=6,
            )
            return ""
        include_path = source_dir / relative_path
        if not include_path.is_file():
            raise GridImportError(
                f"GRDECL INCLUDE references {include_path!r} which does not exist."
            )
        try:
            included_text = include_path.read_text(encoding="ascii", errors="replace")
        except OSError as exc:
            raise GridImportError(
                f"Cannot read GRDECL INCLUDE file {include_path!r}: {exc}"
            ) from exc
        return _resolve_includes(included_text, include_path.parent)

    return include_pattern.sub(_replace, text)


def _strip_comments(text: str) -> str:
    """
    Remove `--` line comments from GRDECL text.

    :param text: Raw GRDECL text.
    :returns: Text with all `--` comments blanked out.
    """
    return re.sub(r"--[^\n]*", "", text)


def _tokenise(text: str) -> typing.List[str]:
    """
    Split comment-stripped GRDECL text into whitespace-separated tokens,
    expanding `N*value` repeat syntax in-place.

    Examples:

        "100*0"  -> ["0", "0", ..., "0"]  (100 times)
        "3*1.5"  -> ["1.5", "1.5", "1.5"]

    :param text: Comment-stripped GRDECL text.
    :returns: Flat list of expanded string tokens.
    """
    raw_tokens = text.split()
    expanded: typing.List[str] = []
    repeat_re = re.compile(r"^(\d+)\*(.+)$")
    for tok in raw_tokens:
        m = repeat_re.match(tok)
        if m:
            expanded.extend([m.group(2)] * int(m.group(1)))
        else:
            expanded.append(tok)
    return expanded


def _extract_keyword_block(text: str, keyword: str) -> typing.Optional[str]:
    """
    Extract the data block following a GRDECL keyword up to its `/`
    terminator.

    Matches the keyword as a whole word (`\\b` boundaries), then captures
    everything up to (but not including) the next `/` that is either
    preceded by optional whitespace on a line of its own, or is
    whitespace-surrounded inline.

    :param text: Full comment-stripped GRDECL text.
    :param keyword: Keyword name (e.g. `"COORD"`).
    :returns: The raw data string between the keyword and its `/`
        terminator, or `None` if the keyword is absent.
    """
    pattern = re.compile(
        r"\b" + re.escape(keyword) + r"\b\s*(.*?)(?:\n\s*/\s*|\s*/\s*)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _extract_all_keyword_blocks(text: str, keyword: str) -> typing.List[str]:
    """
    Extract **all** data blocks for a keyword that may appear multiple times
    (e.g. `FAULTS`, `MULTFLT`).

    :param text: Full comment-stripped GRDECL text.
    :param keyword: Keyword name.
    :returns: List of block strings, one per occurrence (may be empty).
    """
    pattern = re.compile(
        r"\b" + re.escape(keyword) + r"\b\s*(.*?)(?:\n\s*/\s*|\s*/\s*)",
        re.DOTALL | re.IGNORECASE,
    )
    return [m.group(1).strip() for m in pattern.finditer(text)]


_UNITS_MAP: typing.Dict[str, UnitSystem] = {
    "METRES": UnitSystem.METRIC,
    "METER": UnitSystem.METRIC,
    "M": UnitSystem.METRIC,
    "FEET": UnitSystem.FIELD,
    "FT": UnitSystem.FIELD,
    "CM": UnitSystem.LAB,
    "CENTIMETRES": UnitSystem.LAB,
    "CENTIMETERS": UnitSystem.LAB,
}

_UNIT_KEYWORDS: typing.Dict[str, UnitSystem] = {
    "FIELD": UnitSystem.FIELD,
    "METRIC": UnitSystem.METRIC,
    "LAB": UnitSystem.LAB,
    "SI": UnitSystem.SI,
}

_OPERATOR_KEYWORDS: typing.FrozenSet[str] = frozenset(
    {
        "MULTX",
        "MULTY",
        "MULTZ",
        "MULTX-",
        "MULTY-",
        "MULTZ-",
        "ACTNUM",
    }
)


@numba.njit(cache=True, inline="always")
def _get_flat_index(i0: int, j0: int, k0: int, nx: int, ny: int) -> int:
    """
    Convert 0-based (i, j, k) to flat C-order index for a (nz, ny, nx) array.

    Flat index = k*ny*nx + j*nx + i  (k outermost, i innermost).

    :param i0: 0-based x index.
    :param j0: 0-based y index.
    :param k0: 0-based z index.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :returns: Flat cell index.
    """
    return k0 * ny * nx + j0 * nx + i0


@numba.njit(cache=True)
def _apply_box_operation(
    array: npt.NDArray,
    box: typing.Tuple[int, int, int, int, int, int],
    op: str,
    value: float,
    src_array: typing.Optional[npt.NDArray],
    nx: int,
    ny: int,
) -> None:
    """
    Apply an Eclipse operator keyword to `array` within a BOX sub-region.

    Modifies `array` **in-place**.

    :param array: Flat `(n_cells,)` array in k,j,i C-order (k outermost).
    :param box: `(i1, i2, j1, j2, k1, k2)` 0-based inclusive box corners.
    :param op: One of `"EQUALS"`, `"ADD"`, `"MULTIPLY"`, `"COPY"`.
    :param value: Scalar operand (ignored for `"COPY"`).
    :param src_array: Source array for `"COPY"`; `None` otherwise.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    """
    i1, i2, j1, j2, k1, k2 = box
    for k in range(k1, k2 + 1):
        for j in range(j1, j2 + 1):
            for i in range(i1, i2 + 1):
                idx = _get_flat_index(i, j, k, nx, ny)
                if op == "EQUALS":
                    array[idx] = value
                elif op == "ADD":
                    array[idx] += value
                elif op == "MULTIPLY":
                    array[idx] *= value
                elif op == "COPY" and src_array is not None:
                    array[idx] = src_array[idx]


def _parse_mult_arrays(
    clean: str,
    nx: int,
    ny: int,
    nz: int,
) -> typing.Dict[str, npt.NDArray[np.float64]]:
    """
    Parse all directional transmissibility multiplier arrays from
    comment-stripped GRDECL text.

    Handles two modes:

    1. **Direct keyword blocks** - e.g. `MULTX` followed by per-cell values
       and a `/` terminator. Values may use the `N*v` repeat syntax.

    2. **Operator blocks** - `BOX` / `ENDBOX` / `EQUALS` / `ADD` /
       `MULTIPLY` / `COPY` sequences that modify MULT arrays (and
       `ACTNUM`) over a restricted IJK sub-region. The current box
       accumulates across multiple operator records until `ENDBOX` (or a
       new `BOX`) resets it.

    Arrays that are not present in the text are returned with all-ones (the
    neutral multiplier). An array is included in the return dict only when
    it has been explicitly modified so that callers can detect `None` vs
    all-ones.

    :param clean: Comment-stripped GRDECL text (no `--` comments).
    :param nx: Number of cells in x (from `SPECGRID`).
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :returns: Dict mapping keyword name (e.g. `"MULTX"`, `"MULTX-"`) to a
        flat `(nz*ny*nx,)` float64 array in k-major C-order. Only keywords
        that were actually set are present.
    """
    n_cells = nx * ny * nz
    default_box = (0, nx - 1, 0, ny - 1, 0, nz - 1)

    # Lazily-created arrays (only allocate when a keyword is first touched)
    arrays: typing.Dict[str, npt.NDArray[np.float64]] = {}

    def _get_or_create(name: str) -> npt.NDArray[np.float64]:
        if name not in arrays:
            arrays[name] = np.ones(n_cells, dtype=np.float64)
        return arrays[name]

    # Pass 1: parse standalone keyword data blocks (MULTX ... /)
    for kw in _OPERATOR_KEYWORDS:
        if kw == "ACTNUM":
            continue  # ACTNUM handled separately in the main parser

        block = _extract_keyword_block(clean, kw)
        if block is None:
            continue

        tokens = _tokenise(block)
        if not tokens:
            continue

        if len(tokens) == 1:
            arr = _get_or_create(kw)
            arr[:] = float(tokens[0])
        elif len(tokens) == n_cells:
            arr = _get_or_create(kw)
            arr[:] = np.array(tokens, dtype=np.float64)
        else:
            warnings.warn(
                f"Keyword {kw!r} has {len(tokens)} values but grid has {n_cells} cells. Skipping.",
                stacklevel=5,
            )

    # Pass 2: scan for operator blocks (BOX / ENDBOX / EQUALS / ADD / COPY /
    # MULTIPLY) and apply them.
    #
    # Strategy: tokenise the entire file once more as a flat token stream,
    # then drive a small state machine that tracks the current box and applies
    # operations when operator keywords are encountered.
    #
    # We process tokens sequentially. When we see a recognised operator
    # keyword we consume the following tokens up to and including '/'.

    # Build a flat token list that preserves keyword boundaries
    # We need to be careful: the repeat N*v expansion must happen, and
    # we need to detect keyword names at the start of "records".
    # Use a simple two-pass: first strip pure data blocks already consumed
    # in pass 1 so we don't re-process them.  Instead of stripping we just
    # use a dedicated regex scan for the operator keywords.

    _OP_KEYWORDS_RE = re.compile(
        r"\b(BOX|ENDBOX|EQUALS|ADD|MULTIPLY|COPY)\b",
        re.IGNORECASE,
    )

    # Collect all operator-keyword positions in order
    op_matches = list(_OP_KEYWORDS_RE.finditer(clean))
    if not op_matches:
        return arrays

    current_box = default_box

    i_match = 0
    while i_match < len(op_matches):
        m = op_matches[i_match]
        kw_upper = m.group(1).upper()
        # Advance past the keyword and collect tokens up to '/'
        text_after = clean[m.end() :]
        # Find the next '/' - that terminates the record
        slash_pos = text_after.find("/")
        if slash_pos == -1:
            # No terminator - skip this record
            i_match += 1
            continue

        record_text = text_after[:slash_pos]
        raw_tokens = record_text.split()
        # Expand N*v repeat syntax
        tokens: typing.List[str] = []
        repeat_re = re.compile(r"^(\d+)\*(.+)$")
        for tok in raw_tokens:
            mm = repeat_re.match(tok)
            if mm:
                tokens.extend([mm.group(2)] * int(mm.group(1)))
            else:
                tokens.append(tok)

        if kw_upper == "BOX":
            # BOX i1 i2 j1 j2 k1 k2
            if len(tokens) >= 6:
                try:
                    bi1 = int(tokens[0]) - 1
                    bi2 = int(tokens[1]) - 1
                    bj1 = int(tokens[2]) - 1
                    bj2 = int(tokens[3]) - 1
                    bk1 = int(tokens[4]) - 1
                    bk2 = int(tokens[5]) - 1
                    # Clamp to grid bounds
                    bi1 = max(0, min(bi1, nx - 1))
                    bi2 = max(0, min(bi2, nx - 1))
                    bj1 = max(0, min(bj1, ny - 1))
                    bj2 = max(0, min(bj2, ny - 1))
                    bk1 = max(0, min(bk1, nz - 1))
                    bk2 = max(0, min(bk2, nz - 1))
                    current_box = (bi1, bi2, bj1, bj2, bk1, bk2)
                except (ValueError, IndexError):
                    warnings.warn(
                        f"BOX record has invalid tokens {tokens!r}; ignoring.",
                        stacklevel=5,
                    )
            else:
                warnings.warn(
                    f"BOX record has only {len(tokens)} tokens; expected 6. Ignoring.",
                    stacklevel=5,
                )

        elif kw_upper == "ENDBOX":
            current_box = default_box

        elif kw_upper in ("EQUALS", "ADD", "MULTIPLY"):
            # EQUALS/ADD/MULTIPLY target_keyword  value /
            # Multiple records may appear in the same block (each with its own
            # slash, but we've already split on the first slash in this pass).
            # Each record is: keyword value (the slash terminates the record).
            if len(tokens) >= 2:
                target_kw = tokens[0].upper()
                if target_kw in _OPERATOR_KEYWORDS and target_kw != "ACTNUM":
                    try:
                        val = float(tokens[1])
                        arr = _get_or_create(target_kw)
                        _apply_box_operation(
                            arr,
                            current_box,
                            kw_upper,
                            val,
                            None,
                            nx,
                            ny,
                        )
                    except (ValueError, IndexError):
                        warnings.warn(
                            f"{kw_upper} record for {target_kw!r} has invalid value "
                            f"{tokens[1:]!r}; skipping.",
                            stacklevel=5,
                        )

        elif kw_upper == "COPY":
            # COPY src_keyword dst_keyword /
            if len(tokens) >= 2:
                src_kw = tokens[0].upper()
                dst_kw = tokens[1].upper()
                if (
                    dst_kw in _OPERATOR_KEYWORDS
                    and dst_kw != "ACTNUM"
                    and src_kw in _OPERATOR_KEYWORDS
                    and src_kw != "ACTNUM"
                ):
                    src_arr = arrays.get(src_kw)
                    if src_arr is None:
                        src_arr = np.ones(n_cells, dtype=np.float64)
                    dst_arr = _get_or_create(dst_kw)
                    _apply_box_operation(
                        dst_arr,
                        current_box,
                        "COPY",
                        0.0,
                        src_arr,
                        nx,
                        ny,
                    )

        i_match += 1

    return arrays


def _parse_gridunit(clean: str) -> typing.Optional[UnitSystem]:
    """
    Parse the `GRIDUNIT` keyword to determine the grid geometry unit system.

    `GRIDUNIT` format:

        GRIDUNIT
          'METRES  ' '        ' /

    The first quoted token is the unit string.

    :param clean: Comment-stripped GRDECL text.
    :returns: `UnitSystem` or `None` if `GRIDUNIT` is absent.
    """
    block = _extract_keyword_block(clean, "GRIDUNIT")
    if block is None:
        return None
    quoted = re.findall(r"['\"]([^'\"]*)['\"]", block)
    if not quoted:
        return None
    return _UNITS_MAP.get(quoted[0].strip().upper())


def _parse_mapunits(clean: str) -> typing.Optional[UnitSystem]:
    """
    Parse the `MAPUNITS` keyword to determine the map coordinate unit system.

    :param clean: Comment-stripped GRDECL text.
    :returns: `UnitSystem` or `None` if `MAPUNITS` is absent.
    """
    block = _extract_keyword_block(clean, "MAPUNITS")
    if block is None:
        return None
    quoted = re.findall(r"['\"]([^'\"]*)['\"]", block)
    if not quoted:
        return None
    return _UNITS_MAP.get(quoted[0].strip().upper())


def _detect_unit_system(clean: str) -> UnitSystem:
    """
    Determine the grid geometry unit system from comment-stripped GRDECL text.

    Resolution order (highest to lowest priority):

    1. `GRIDUNIT` keyword - explicit geometry unit declaration.
    2. Bare section keywords `FIELD`, `METRIC`, `LAB`, `SI`.
    3. Default: `FIELD` (Eclipse default when no unit keyword is present).

    :param clean: Comment-stripped GRDECL text.
    :returns: The declared `UnitSystem`.
    """
    gridunit = _parse_gridunit(clean)
    if gridunit is not None:
        return gridunit
    for keyword, us in _UNIT_KEYWORDS.items():
        if re.search(r"(?<!\w)" + keyword + r"(?!\w)", clean, re.IGNORECASE):
            return us
    return UnitSystem.FIELD


def _parse_mapaxes(clean: str) -> typing.Optional[MapAxes]:
    """
    Parse the optional `MAPAXES` keyword.

    Eclipse `MAPAXES` format (6 values):

        MAPAXES
          X_ORIGIN  Y_ORIGIN  X_UNITX  Y_UNITX  X_UNITY  Y_UNITY /

    :param clean: Comment-stripped GRDECL text.
    :returns: A `MapAxes` instance, or `None` if absent or unparseable.
    """
    block = _extract_keyword_block(clean, "MAPAXES")
    if block is None:
        return None
    tokens = block.split()
    if len(tokens) < 6:
        warnings.warn(
            f"MAPAXES block has {len(tokens)} values; expected 6. Ignoring.",
            stacklevel=4,
        )
        return None
    try:
        vals = [float(t) for t in tokens[:6]]
    except ValueError as exc:
        warnings.warn(
            f"MAPAXES values are not valid floats ({exc}). Ignoring.",
            stacklevel=4,
        )
        return None
    map_unit = _parse_mapunits(clean) or UnitSystem.FIELD
    return MapAxes(
        origin=np.array([vals[0], vals[1]], dtype=np.float64),
        map_x_axis_point=np.array([vals[2], vals[3]], dtype=np.float64),
        map_y_axis_point=np.array([vals[4], vals[5]], dtype=np.float64),
        unit_system=map_unit,
    )


def _parse_pinch(clean: str) -> typing.Optional[float]:
    """
    Parse `PINCH` or `PINCHOUT` from comment-stripped GRDECL text.

    Returns the pinch thickness tolerance in grid length units.
    `PINCH` takes precedence over `PINCHOUT` when both are present.

    `PINCH` format:

        PINCH
          0.01 /

    `PINCHOUT` format (uses the Eclipse default tolerance 1e-6):

        PINCHOUT
        /

    :param clean: Comment-stripped GRDECL text.
    :returns: Pinch thickness tolerance, or `None` if neither keyword
        exists.
    :raises GridImportError: If `PINCH` contains an invalid value.
    """
    pinch_block = _extract_keyword_block(clean, "PINCH")
    if pinch_block is not None:
        tokens = pinch_block.split()
        if not tokens:
            warnings.warn(
                "PINCH keyword found but no thickness tolerance was specified. "
                "Using Eclipse default of 1e-6.",
                stacklevel=4,
            )
            return 1e-6
        try:
            return float(tokens[0])
        except ValueError as exc:
            raise GridImportError(
                f"Invalid PINCH thickness tolerance: {tokens[0]!r}."
            ) from exc

    if _extract_keyword_block(clean, "PINCHOUT") is not None:
        return 1e-6
    return None


def _parse_nnc(
    clean: str, nx: int, ny: int, nz: int
) -> typing.Tuple[
    typing.Optional[IntArray[TwoDimensions]],
    typing.Optional[FloatArray[OneDimension]],
]:
    """
    Parse the `NNC` keyword from comment-stripped GRDECL text.

    Each record in the `NNC` block consists of 7 whitespace-separated
    tokens:

        I1 J1 K1  I2 J2 K2  T

    where `(I, J, K)` are 1-based structured cell indices and `T` is the
    transmissibility value in the grid's unit system.

    All records are collected; incomplete trailing groups are ignored with a
    warning.

    :param clean: Comment-stripped GRDECL text.
    :param nx: Number of cells in x (from `SPECGRID`).
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :returns: 2-tuple of:

        - Shape `(n_nnc, 2)` int32 array of 0-based cell index pairs,
          or `None` if the keyword is absent.
        - Shape `(n_nnc,)` float64 transmissibility array,
          or `None` if the keyword is absent.

    :raises GridImportError: If a record contains non-numeric values or
        out-of-bounds cell indices.
    """
    block = _extract_keyword_block(clean, "NNC")
    if block is None:
        return None, None

    tokens = block.split()
    if not tokens:
        return None, None

    n_complete = len(tokens) // 7
    remainder = len(tokens) % 7
    if remainder != 0:
        warnings.warn(
            f"NNC block has {len(tokens)} tokens, which is not a multiple of 7.  "
            f"The last {remainder} token(s) will be ignored.",
            stacklevel=4,
        )

    pairs: typing.List[typing.Tuple[int, int]] = []
    transmissibilities: typing.List[float] = []

    for rec in range(n_complete):
        base = rec * 7
        try:
            i1, j1, k1 = int(tokens[base]), int(tokens[base + 1]), int(tokens[base + 2])
            i2, j2, k2 = (
                int(tokens[base + 3]),
                int(tokens[base + 4]),
                int(tokens[base + 5]),
            )
            transmissibility = float(tokens[base + 6])
        except ValueError as exc:
            raise GridImportError(
                f"NNC record {rec}: cannot parse value: {exc}"
            ) from exc

        for label, i, j, k in [("first", i1, j1, k1), ("second", i2, j2, k2)]:
            if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz):
                raise GridImportError(
                    f"NNC record {rec} {label} cell ({i},{j},{k}) is outside "
                    f"grid bounds ({nx}, {ny}, {nz})."
                )

        # Eclipse ordering: I fastest -> flat index = (i-1) + (j-1)*nx + (k-1)*nx*ny
        c1 = (i1 - 1) + (j1 - 1) * nx + (k1 - 1) * nx * ny
        c2 = (i2 - 1) + (j2 - 1) * nx + (k2 - 1) * nx * ny
        pairs.append((c1, c2))
        transmissibilities.append(transmissibility)

    if not pairs:
        return None, None

    return (
        np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        np.asarray(transmissibilities, dtype=np.float64),
    )


def _parse_faults(clean: str) -> typing.List[FaultRecord]:
    """
    Parse all `FAULTS` keyword blocks from comment-stripped GRDECL text.

    `FAULTS` format (one record per line, block terminated by `/`):

        FAULTS
          'FAULT_NAME'  I1 I2  J1 J2  K1 K2  FACE /
          ...
        /

    Multiple `FAULTS` blocks in the same file are concatenated.  Each
    record is terminated by its own `/` within the block.

    :param clean: Comment-stripped GRDECL text.
    :returns: List of `FaultRecord` objects in file order.
    :raises GridImportError: If a record contains non-integer indices or an
        unrecognised face direction.
    """
    valid_dirs = {"I", "I-", "J", "J-", "K", "K-"}
    records: typing.List[FaultRecord] = []

    for block in _extract_all_keyword_blocks(clean, "FAULTS"):
        if not block.strip():
            continue
        # Each fault record may be terminated by its own /; strip those
        # per-record slashes to get a flat token stream.
        block_clean = re.sub(r"/", " ", block)

        # Each record: NAME I1 I2 J1 J2 K1 K2 FACE  (8 tokens, name is quoted)
        # After stripping quotes, we process groups of 8.
        # But names with spaces inside quotes need special handling.
        # Extract quoted names then the remaining integer/dir tokens.
        # Strategy: use regex to find each record as a quoted name followed
        # by 7 whitespace-separated tokens.
        record_pattern = re.compile(
            r"['\"]([^'\"]+)['\"]\s+"  # 'NAME'
            r"(\d+)\s+(\d+)\s+"  # I1 I2
            r"(\d+)\s+(\d+)\s+"  # J1 J2
            r"(\d+)\s+(\d+)\s+"  # K1 K2
            r"(I-?|J-?|K-?)",  # FACE
            re.IGNORECASE,
        )
        for m in record_pattern.finditer(block_clean):
            name = m.group(1).strip()
            i1, i2 = int(m.group(2)), int(m.group(3))
            j1, j2 = int(m.group(4)), int(m.group(5))
            k1, k2 = int(m.group(6)), int(m.group(7))
            face_direction = m.group(8).upper()

            if face_direction not in valid_dirs:
                raise GridImportError(
                    f"FAULTS record for {name!r}: unrecognised face direction "
                    f"{face_direction!r}.  Valid values: {sorted(valid_dirs)}."
                )
            records.append(
                FaultRecord(
                    name=name,
                    i1=i1,
                    i2=i2,
                    j1=j1,
                    j2=j2,
                    k1=k1,
                    k2=k2,
                    face_direction=face_direction,
                )
            )

    return records


def _parse_multflt(clean: str) -> typing.Dict[str, float]:
    """
    Parse all `MULTFLT` keyword blocks from comment-stripped GRDECL text.

    `MULTFLT` format (one record per line, block terminated by `/`):

        MULTFLT
          'FAULT_NAME'  MULTIPLIER /
          ...
        /

    When the same fault name appears in multiple records, the last value wins
    (consistent with Eclipse behaviour).

    :param clean: Comment-stripped GRDECL text.
    :returns: Dict mapping fault name -> transmissibility multiplier.
    :raises GridImportError: If a multiplier value cannot be parsed as float.
    """
    result: typing.Dict[str, float] = {}

    for block in _extract_all_keyword_blocks(clean, "MULTFLT"):
        if not block.strip():
            continue
        block_clean = re.sub(r"/", " ", block)
        record_pattern = re.compile(
            r"['\"]([^'\"]+)['\"]\s+([\d.eE+\-]+)",
            re.IGNORECASE,
        )
        for m in record_pattern.finditer(block_clean):
            name = m.group(1).strip()
            try:
                multiplier = float(m.group(2))
            except ValueError as exc:
                raise GridImportError(
                    f"MULTFLT record for {name!r}: cannot parse multiplier "
                    f"{m.group(2)!r}: {exc}"
                ) from exc
            result[name] = multiplier

    return result


def _parse_grdecl_cartesian(
    clean: str,
    nx: int,
    ny: int,
    nz: int,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    mult_arrays: typing.Optional[typing.Dict[str, npt.NDArray[np.float64]]] = None,
) -> Grid:
    """
    Build a Cartesian `Grid` from `TOPS` / `DX` / `DY` / `DZ`
    keywords.

    `TOPS` gives the depth of the top face of every cell in the top layer
    (column-major, x-fastest).  `DX` / `DY` / `DZ` give uniform or
    per-cell spacings in each direction.

    For variable-spacing files, this function extracts the first row/column
    slice for each axis.  This is exact when spacing is uniform within each
    axis slice (the common case) but approximate otherwise.

    :param clean: Comment-stripped GRDECL text.
    :param nx: Number of cells in x (from `SPECGRID`).
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param unit_system: Already-detected unit system.
    :param metadata: Optional metadata dict; augmented in-place.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or have wrong
        sizes.
    """

    def _read_array(keyword: str, expected: int) -> npt.NDArray[np.float64]:
        block = _extract_keyword_block(clean, keyword)
        if block is None:
            raise GridImportError(
                f"Cartesian GRDECL grid is missing required keyword {keyword!r}."
            )
        tokens = _tokenise(block)
        if len(tokens) == 1:
            return np.full(expected, float(tokens[0]), dtype=np.float64)
        if len(tokens) != expected:
            raise GridImportError(
                f"{keyword} expected {expected} values; got {len(tokens)}."
            )
        return np.array(tokens, dtype=np.float64)

    def _read_vector(
        keyword: str, expected: int
    ) -> typing.Optional[npt.NDArray[np.float64]]:
        """Read a DXV/DYV/DZV vector keyword (length = expected)."""
        block = _extract_keyword_block(clean, keyword)
        if block is None:
            return None
        tokens = _tokenise(block)
        if len(tokens) != expected:
            warnings.warn(
                f"{keyword} expected {expected} values; got {len(tokens)}. Ignoring.",
                stacklevel=4,
            )
            return None
        return np.array(tokens, dtype=np.float64)

    n_cells = nx * ny * nz
    n_columns = nx * ny

    tops_flat = _read_array("TOPS", n_columns)
    z_top = float(tops_flat.min())
    if tops_flat.max() - tops_flat.min() > 1.0:
        warnings.warn(
            "GRDECL TOPS values vary by more than 1 unit; the Cartesian factory "
            "uses a flat top surface at the minimum TOPS value.  Geometry may be "
            "approximate for dipping grids.",
            stacklevel=4,
        )

    # Spacing: prefer DXV/DYV/DZV (vector form) over per-cell DX/DY/DZ
    dx_1d = _read_vector("DXV", nx)
    dy_1d = _read_vector("DYV", ny)
    dz_1d = _read_vector("DZV", nz)

    if dx_1d is None:
        dx_arr = _read_array("DX", n_cells)
        dx_1d = dx_arr.reshape(nz, ny, nx, order="F")[0, 0, :]

    if dy_1d is None:
        dy_arr = _read_array("DY", n_cells)
        dy_1d = dy_arr.reshape(nz, ny, nx, order="F")[0, :, 0]

    if dz_1d is None:
        dz_arr = _read_array("DZ", n_cells)
        dz_1d = dz_arr.reshape(nz, ny, nx, order="F")[:, 0, 0]

    actnum_block = _extract_keyword_block(clean, "ACTNUM")
    meta: typing.Dict[str, typing.Any] = dict(metadata or {})
    meta["source_format"] = "grdecl_cartesian"

    if actnum_block is not None:
        actnum_tokens = _tokenise(actnum_block)
        if len(actnum_tokens) == n_cells:
            meta["actnum"] = (
                np.array(actnum_tokens, dtype=np.int32)
                .reshape(nx, ny, nz, order="F")
                .transpose(2, 1, 0)
            )

    return make_cartesian_grid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx_1d,
        dy=dy_1d,
        dz=dz_1d,
        origin=(0.0, 0.0, z_top),
        unit_system=unit_system,
        metadata=meta,
    )


def _parse_grdecl_corner_point(
    clean: str,
    nx: int,
    ny: int,
    nz: int,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    nnc_cell_pairs: typing.Optional[IntArray[TwoDimensions]] = None,
    nnc_transmissibilities: typing.Optional[FloatArray[OneDimension]] = None,
    fault_records: typing.Optional[typing.List[FaultRecord]] = None,
    fault_transmissibility_multipliers: typing.Optional[typing.Dict[str, float]] = None,
    mult_arrays: typing.Optional[typing.Dict[str, npt.NDArray[np.float64]]] = None,
) -> Grid:
    """
    Build a corner-point `Grid` from `COORD` / `ZCORN` / `ACTNUM`
    keywords.

    :param clean: Comment-stripped GRDECL text.
    :param nx: Number of cells in x (from `SPECGRID`).
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param unit_system: Already-detected unit system.
    :param metadata: Optional metadata dict; augmented in-place.
    :param nnc_cell_pairs: Parsed NNC cell index pairs from the `NNC`
        keyword (0-based, shape `(n_nnc, 2)`).
    :param nnc_transmissibilities: Transmissibility values for each NNC pair.
    :param fault_records: Parsed `FaultRecord` list from `FAULTS`.
    :param fault_transmissibility_multipliers: Mapping from fault name to
        multiplier from `MULTFLT`.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or have wrong
        sizes.
    """
    # COORD
    coord_block = _extract_keyword_block(clean, "COORD")
    if coord_block is None:
        raise GridImportError("GRDECL file is missing the required COORD keyword.")

    coord_tokens = _tokenise(coord_block)
    expected_coord = (nx + 1) * (ny + 1) * 6
    if len(coord_tokens) != expected_coord:
        raise GridImportError(
            f"COORD expected {expected_coord} values for {nx}x{ny} grid; "
            f"got {len(coord_tokens)}."
        )
    # Flat order: (x_pillar, y_pillar, 6) with x fastest -> reshape then transpose
    coord_arr = (
        np.array(coord_tokens, dtype=np.float64)
        .reshape(nx + 1, ny + 1, 6, order="F")
        .transpose(1, 0, 2)  # -> (ny+1, nx+1, 6) C-order
    )

    # ZCORN
    zcorn_block = _extract_keyword_block(clean, "ZCORN")
    if zcorn_block is None:
        raise GridImportError("GRDECL file is missing the required ZCORN keyword.")

    zcorn_tokens = _tokenise(zcorn_block)
    expected_zcorn = nx * ny * nz * 8
    if len(zcorn_tokens) != expected_zcorn:
        raise GridImportError(
            f"ZCORN expected {expected_zcorn} values for {nx}x{ny}x{nz} grid; "
            f"got {len(zcorn_tokens)}."
        )
    zcorn_arr = (
        np.array(zcorn_tokens, dtype=np.float64)
        .reshape(nx * 2, ny * 2, nz * 2, order="F")
        .transpose(2, 1, 0)  # -> (nz*2, ny*2, nx*2) C-order
    )

    # ACTNUM (optional)
    actnum_block = _extract_keyword_block(clean, "ACTNUM")
    if actnum_block is not None:
        actnum_tokens = _tokenise(actnum_block)
        expected_actnum = nx * ny * nz
        if len(actnum_tokens) != expected_actnum:
            raise GridImportError(
                f"ACTNUM expected {expected_actnum} values; got {len(actnum_tokens)}."
            )
        actnum_arr = (
            np.array(actnum_tokens, dtype=np.int32)
            .reshape(nx, ny, nz, order="F")
            .transpose(2, 1, 0)  # -> (nz, ny, nx) C-order
        )
    else:
        actnum_arr = None

    meta: typing.Dict[str, typing.Any] = dict(metadata or {})
    meta["source_format"] = "grdecl_corner_point"
    meta["actnum"] = actnum_arr

    # Extract directional multiplier arrays if present
    _ma = mult_arrays or {}

    def _to_active(key: str) -> typing.Optional[npt.NDArray[np.float64]]:
        """
        Re-index a full-grid (nz*ny*nx) flat array to active-cell order.

        Corner-point grids may have inactive cells filtered out; the
        multiplier arrays from the GRDECL file are in full structured order
        (k*ny*nx + j*nx + i). We defer the active-cell subsetting to
        `make_corner_point_grid` — since the factory returns an unstructured
        Grid we cannot easily do the remapping here. Instead we pass the
        full flat array and let the factory store it as-is. The flow solver
        is responsible for indexing into the structured array using the
        active-cell (k,j,i) tuples stored in `Grid.metadata`.

        For now: pass through unchanged.
        """
        return _ma.get(key)

    return make_corner_point_grid(
        coord=coord_arr,
        zcorn=zcorn_arr,
        actnum=actnum_arr,
        unit_system=unit_system,
        metadata=meta,
        nnc_cell_pairs=nnc_cell_pairs,
        nnc_transmissibilities=nnc_transmissibilities,
        fault_records=fault_records or [],
        fault_transmissibility_multipliers=fault_transmissibility_multipliers,
        positive_x_transmissibility_multipliers=_to_active("MULTX"),
        negative_x_transmissibility_multipliers=_to_active("MULTX-"),
        positive_y_transmissibility_multipliers=_to_active("MULTY"),
        negative_y_transmissibility_multipliers=_to_active("MULTY-"),
        positive_z_transmissibility_multipliers=_to_active("MULTZ"),
        negative_z_transmissibility_multipliers=_to_active("MULTZ-"),
    )


def _parse_grdecl(
    text: str,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Parse a complete GRDECL text blob into a `bores.grids.base.Grid`.

    Dispatches to either `_parse_grdecl_cartesian` (when the file has
    `TOPS` / `DX` / `DY` / `DZ` but no `COORD`) or
    `_parse_grdecl_corner_point` (when `COORD` / `ZCORN` are present).

    :param text: Raw GRDECL text (may contain `--` comments).
    :param metadata: Optional extra metadata.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or arrays have
        inconsistent shapes.
    """
    clean = _strip_comments(text)

    # SPECGRID (mandatory)
    specgrid_block = _extract_keyword_block(clean, "SPECGRID")
    if specgrid_block is None:
        raise GridImportError("GRDECL file is missing the required SPECGRID keyword.")
    specgrid_tokens = specgrid_block.split()
    if len(specgrid_tokens) < 3:
        raise GridImportError(
            f"SPECGRID requires at least 3 values (NX NY NZ); "
            f"got {len(specgrid_tokens)}."
        )
    try:
        nx = int(specgrid_tokens[0])
        ny = int(specgrid_tokens[1])
        nz = int(specgrid_tokens[2])
    except ValueError as exc:
        raise GridImportError(f"SPECGRID values are not valid integers: {exc}") from exc

    has_coord = _extract_keyword_block(clean, "COORD") is not None
    has_tops = _extract_keyword_block(clean, "TOPS") is not None

    # Optional keywords
    map_axes = _parse_mapaxes(clean)
    pinch = _parse_pinch(clean)
    unit_system = _detect_unit_system(clean)

    meta: typing.Dict[str, typing.Any] = dict(metadata or {})
    meta["map_axes"] = map_axes
    meta["pinch"] = pinch
    meta["nx"] = nx
    meta["ny"] = ny
    meta["nz"] = nz

    # NNC (corner-point only)
    nnc_pairs: typing.Optional[IntArray[TwoDimensions]] = None
    nnc_transmissibilities: typing.Optional[FloatArray[OneDimension]] = None
    fault_records: typing.List[FaultRecord] = []
    multflt: typing.Dict[str, float] = {}

    if has_coord:
        nnc_pairs, nnc_transmissibilities = _parse_nnc(clean, nx=nx, ny=ny, nz=nz)
        fault_records = _parse_faults(clean)
        multflt = _parse_multflt(clean)

    mult_arrays = _parse_mult_arrays(clean, nx=nx, ny=ny, nz=nz)

    try:
        if not has_coord and has_tops:
            return _parse_grdecl_cartesian(
                clean=clean,
                nx=nx,
                ny=ny,
                nz=nz,
                unit_system=unit_system,
                metadata=meta,
                mult_arrays=mult_arrays,
            )
        return _parse_grdecl_corner_point(
            clean=clean,
            nx=nx,
            ny=ny,
            nz=nz,
            unit_system=unit_system,
            metadata=meta,
            nnc_cell_pairs=nnc_pairs,
            nnc_transmissibilities=nnc_transmissibilities,
            fault_records=fault_records,
            fault_transmissibility_multipliers=multflt or None,
            mult_arrays=mult_arrays,
        )
    except GridImportError:
        raise
    except Exception as exc:
        raise GridImportError(
            f"Failed to construct grid from parsed GRDECL arrays: {exc}"
        ) from exc


_GRDECL_SOURCES = {"grdecl_corner_point", "grdecl_cartesian"}


def _build_grdecl_text(
    grid: Grid,
    *,
    actnum: typing.Optional[ActnumArray] = None,
) -> str:
    """
    Render a `bores.grids.base.Grid` as GRDECL text.

    Only grids that originated from `load_grdecl` are supported.  Grids
    from other factories (Voronoi, tetrahedral, …) cannot be reliably
    expressed in GRDECL format and will raise `GridExportError`.

    Corner-point grids are written as
    `SPECGRID / COORD / ZCORN / ACTNUM`.  Because `COORD` and `ZCORN`
    are not cached in metadata, they are re-derived from the Grid's
    bounding-box geometry via `rederive_corner_point_arrays` (lossy for
    deformed pillar grids - a warning is emitted).

    Cartesian grids are written as
    `SPECGRID / TOPS / DX / DY / DZ` - an exact lossless round-trip.

    :param grid: The grid to serialise.
    :param actnum: Optional active-cell flag array.
    :returns: GRDECL text as a string.
    :raises GridExportError: If the grid's source format is not a GRDECL
        variant or dimensions cannot be determined.
    """
    meta: typing.Mapping[str, typing.Any] = getattr(grid, "metadata", {}) or {}
    source_format: str = meta.get("source_format", "")
    if source_format not in _GRDECL_SOURCES:
        raise GridExportError(
            f"Cannot export a Grid with source_format={source_format!r} to GRDECL. "
            "Only grids originally loaded by load_grdecl() support GRDECL export.  "
            f"Supported source formats: {sorted(_GRDECL_SOURCES)}."
        )
    if source_format == "grdecl_cartesian":
        return _build_grdecl_cartesian_text(grid, actnum=actnum)
    return _build_grdecl_corner_point_text(grid, actnum=actnum)


def _build_grdecl_cartesian_text(
    grid: Grid,
    *,
    actnum: typing.Optional[ActnumArray] = None,
) -> str:
    """
    Render a Cartesian-source Grid as `SPECGRID / TOPS / DX / DY / DZ`.

    :param grid: Grid whose `metadata["source_format"] == "grdecl_cartesian"`.
    :param actnum: Optional active-cell flag array.
    :returns: GRDECL text.
    :raises GridExportError: If grid dimensions cannot be recovered.
    """
    meta: typing.Mapping[str, typing.Any] = getattr(grid, "metadata", {}) or {}
    nx = meta.get("nx")
    ny = meta.get("ny")
    nz = meta.get("nz")

    if nx is None or ny is None or nz is None:
        nx, ny, nz = 1, 1, grid.n_cells

    n_cells = nx * ny * nz
    if n_cells != grid.n_cells:
        raise GridExportError(
            f"Stored dimensions ({nx}x{ny}x{nz}={n_cells}) do not match "
            f"grid.n_cells={grid.n_cells}."
        )

    lines: typing.List[str] = []
    lines.append("SPECGRID")
    lines.append(f"  {nx}  {ny}  {nz}  1  F /")
    lines.append("")

    _emit_gridunit(lines, grid.unit_system)

    map_axes: typing.Optional[MapAxes] = meta.get("map_axes")
    if map_axes is not None:
        _emit_mapaxes(lines, map_axes)

    # TOPS
    top_layer_indices = [i + j * nx for j in range(ny) for i in range(nx)]
    tops_vals = grid.cell_min_xyz[top_layer_indices, 2]
    lines.append("TOPS")
    for i in range(0, len(tops_vals), 6):
        chunk = tops_vals[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")
    lines.append("")

    # DX / DY / DZ
    for axis_name, column in [("DX", 0), ("DY", 1), ("DZ", 2)]:
        extents = grid.cell_max_xyz[:, column] - grid.cell_min_xyz[:, column]
        extents_3d = extents.reshape(nz, ny, nx)
        flat = extents_3d.transpose(2, 1, 0).ravel(order="F")
        lines.append(axis_name)
        for i in range(0, len(flat), 6):
            chunk = flat[i : i + 6]
            lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
        lines.append("/")
        lines.append("")

    if actnum is not None:
        _emit_actnum(lines, actnum, grid.n_cells, nx, ny, nz)

    # Emit MULT arrays if present
    _mult_keyword_map = [
        ("MULTX", grid.positive_x_transmissibility_multipliers),
        ("MULTX-", grid.negative_x_transmissibility_multipliers),
        ("MULTY", grid.positive_y_transmissibility_multipliers),
        ("MULTY-", grid.negative_y_transmissibility_multipliers),
        ("MULTZ", grid.positive_z_transmissibility_multipliers),
        ("MULTZ-", grid.negative_z_transmissibility_multipliers),
    ]
    for kw, arr in _mult_keyword_map:
        if arr is not None:
            _emit_mult_array(lines, kw, arr, nx, ny, nz)

    return "\n".join(lines)


def _build_grdecl_corner_point_text(
    grid: Grid,
    *,
    actnum: typing.Optional[ActnumArray] = None,
) -> str:
    """
    Render a corner-point-source Grid as
    `SPECGRID / COORD / ZCORN / ACTNUM`.

    `COORD` and `ZCORN` are re-derived from the Grid's bounding-box
    geometry via `rederive_corner_point_arrays`.

    :param grid: Grid whose
        `metadata["source_format"] == "grdecl_corner_point"`.
    :param actnum: Optional active-cell flag array.
    :returns: GRDECL text.
    :raises GridExportError: If grid dimensions cannot be determined.
    """
    coord, zcorn, nx, ny, nz = rederive_corner_point_arrays(grid)
    meta: typing.Mapping[str, typing.Any] = getattr(grid, "metadata", {}) or {}

    lines: typing.List[str] = []
    lines.append("SPECGRID")
    lines.append(f"  {nx}  {ny}  {nz}  1  F /")
    lines.append("")

    _emit_gridunit(lines, grid.unit_system)

    map_axes: typing.Optional[MapAxes] = meta.get("map_axes")
    if map_axes is not None:
        _emit_mapaxes(lines, map_axes)

    # COORD - Eclipse Fortran order: x (pillar i) fastest
    lines.append("COORD")
    flat_coord = coord.transpose(1, 0, 2).reshape(-1, 6)
    for row in flat_coord:
        x1, y1, z1, x2, y2, z2 = row
        lines.append(f"  {x1:.6f}  {y1:.6f}  {z1:.6f}  {x2:.6f}  {y2:.6f}  {z2:.6f}")
    lines.append("/")
    lines.append("")

    # ZCORN - Eclipse Fortran order: x fastest, z slowest
    lines.append("ZCORN")
    flat_zcorn = zcorn.transpose(2, 1, 0).ravel(order="F")
    for i in range(0, len(flat_zcorn), 6):
        chunk = flat_zcorn[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")

    # ACTNUM
    effective_actnum = actnum if actnum is not None else meta.get("actnum")
    if effective_actnum is not None:
        _emit_actnum(lines, effective_actnum, grid.n_cells, nx, ny, nz)

    # Emit MULT arrays if present
    _mult_keyword_map = [
        ("MULTX", grid.positive_x_transmissibility_multipliers),
        ("MULTX-", grid.negative_x_transmissibility_multipliers),
        ("MULTY", grid.positive_y_transmissibility_multipliers),
        ("MULTY-", grid.negative_y_transmissibility_multipliers),
        ("MULTZ", grid.positive_z_transmissibility_multipliers),
        ("MULTZ-", grid.negative_z_transmissibility_multipliers),
    ]
    for kw, arr in _mult_keyword_map:
        if arr is not None:
            _emit_mult_array(lines, kw, arr, nx, ny, nz)

    return "\n".join(lines)


def _emit_actnum(
    lines: typing.List[str],
    actnum: npt.NDArray,
    n_cells: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Append an `ACTNUM` block in Eclipse Fortran order to `lines`.

    :param lines: Accumulator list of GRDECL text lines (mutated in-place).
    :param actnum: Shape `(n_cells,)` integer array in k,j,i (C) order.
    :param n_cells: Expected total cell count.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :raises GridExportError: If `actnum` length mismatches `n_cells`.
    """
    actnum_arr = np.asarray(actnum, dtype=np.int32)
    if len(actnum_arr) != n_cells:
        raise GridExportError(
            f"actnum length {len(actnum_arr)} does not match n_cells {n_cells}."
        )
    lines.append("")
    lines.append("ACTNUM")
    flat = actnum_arr.reshape(nz, ny, nx).transpose(2, 1, 0).ravel(order="F")
    for i in range(0, len(flat), 20):
        chunk = flat[i : i + 20]
        lines.append("  " + "  ".join(str(int(v)) for v in chunk))
    lines.append("/")


def _emit_mapaxes(lines: typing.List[str], map_axes: MapAxes) -> None:
    """
    Append a `MAPAXES` block to `lines` in-place.

    Eclipse `MAPAXES` format:

        MAPAXES
          X_ORIGIN  Y_ORIGIN  X_UNITX  Y_UNITX  X_UNITY  Y_UNITY /

    :param lines: Accumulator list (mutated in-place).
    :param map_axes: Source `MapAxes` instance.
    """
    o = map_axes.origin
    mx = map_axes.map_x_axis_point
    my = map_axes.map_y_axis_point
    lines.append("MAPAXES")
    lines.append(
        f"  {o[0]:.6f}  {o[1]:.6f}"
        f"  {mx[0]:.6f}  {mx[1]:.6f}"
        f"  {my[0]:.6f}  {my[1]:.6f}  /"
    )
    lines.append("")


def _emit_gridunit(lines: typing.List[str], unit_system: UnitSystem) -> None:
    """
    Append a ``GRIDUNIT`` keyword block to `lines`.

    :param lines: Accumulator list (mutated in-place).
    :param unit_system: The unit system to declare.
    """
    _US_TO_GRIDUNIT: typing.Dict[UnitSystem, str] = {
        UnitSystem.FIELD: "FEET",
        UnitSystem.METRIC: "METRES",
        UnitSystem.LAB: "CM",
        UnitSystem.SI: "METRES",
    }
    unit_str = _US_TO_GRIDUNIT.get(unit_system, "FEET")
    lines.append("GRIDUNIT")
    lines.append(f"  '{unit_str}  ' '        ' /")
    lines.append("")


def _emit_mult_array(
    lines: typing.List[str],
    keyword: str,
    arr: npt.NDArray[np.float64],
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Append a MULTX / MULTY / MULTZ (or ``-`` variant) keyword block in
    Eclipse Fortran order (x-fastest, z-slowest).

    :param lines: Accumulator list (mutated in-place).
    :param keyword: Eclipse keyword name (e.g. ``"MULTX"``).
    :param arr: Flat ``(nz*ny*nx,)`` array in k,j,i C-order.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    """
    lines.append("")
    lines.append(keyword)
    # Reshape to (nz, ny, nx) then transpose to Fortran order (nx, ny, nz)
    flat_fortran = arr.reshape(nz, ny, nx).transpose(2, 1, 0).ravel(order="F")
    for i in range(0, len(flat_fortran), 6):
        chunk = flat_fortran[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")
