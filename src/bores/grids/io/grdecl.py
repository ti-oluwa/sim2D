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
    clean: str,
    nx: int,
    ny: int,
    nz: int,
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
            t_val = float(tokens[base + 6])
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
        transmissibilities.append(t_val)

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
            face_dir = m.group(8).upper()

            if face_dir not in valid_dirs:
                raise GridImportError(
                    f"FAULTS record for {name!r}: unrecognised face direction "
                    f"{face_dir!r}.  Valid values: {sorted(valid_dirs)}."
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
                    face_dir=face_dir,
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

    n_cells = nx * ny * nz
    n_col = nx * ny

    tops_flat = _read_array("TOPS", n_col)
    z_top = float(tops_flat.min())
    if tops_flat.max() - tops_flat.min() > 1.0:
        warnings.warn(
            "GRDECL TOPS values vary by more than 1 unit; the Cartesian factory "
            "uses a flat top surface at the minimum TOPS value.  Geometry may be "
            "approximate for dipping grids.",
            stacklevel=4,
        )

    dx_arr = _read_array("DX", n_cells)
    dy_arr = _read_array("DY", n_cells)
    dz_arr = _read_array("DZ", n_cells)

    dx_1d = dx_arr.reshape(nz, ny, nx, order="F")[0, 0, :]
    dy_1d = dy_arr.reshape(nz, ny, nx, order="F")[0, :, 0]
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

    try:
        if not has_coord and has_tops:
            return _parse_grdecl_cartesian(
                clean=clean,
                nx=nx,
                ny=ny,
                nz=nz,
                unit_system=unit_system,
                metadata=meta,
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
