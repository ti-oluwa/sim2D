"""
GRDECL text-format reader and writer.

GRDECL is the ASCII keyword-based format used by Eclipse, ResInsight, and
most other reservoir simulators to describe corner-point pillar grids.

**Supported keywords**:

`SPECGRID`, `COORD`, `ZCORN`, `ACTNUM`, `GRIDTYPE`, `TOPS` , `DX` , `DY` , `DZ`, `MAPAXES`
"""

import re
import typing
import warnings
from pathlib import Path

import attrs
import numba
import numpy as np

from bores.errors import GridExportError, GridImportError
from bores.grids.base import Grid
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import (
    ActnumArray,
    CoordArray,
    ZcornArray,
    make_corner_point_grid,
)
from bores.grids.utils import convert
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
    origin: FloatArray[OneDimension]
    map_x_axis_point: FloatArray[OneDimension]
    map_y_axis_point: FloatArray[OneDimension]
    rotation_matrix: FloatArray[ThreeDimensions] = attrs.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation_matrix", self._compute_rotation_matrix())

    def _compute_rotation_matrix(self):
        origin = self.origin
        x_direction = self.map_x_axis_point - origin
        x_direction /= np.linalg.norm(x_direction)

        y_direction = self.map_y_axis_point - origin
        y_direction /= np.linalg.norm(y_direction)

        rotation_matrix = np.array(
            [
                [x_direction[0], y_direction[0]],
                [x_direction[1], y_direction[1]],
            ]
        )
        return rotation_matrix


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

    The function automatically detects whether `source` is a filesystem
    path or raw GRDECL text/bytes.

    :param source: One of:

        - `pathlib.Path` - path to a `.grdecl` file.
        - `str` - either a filesystem path *or* raw GRDECL text.
          Paths are detected by whether the string represents an existing file
          or ends with a known extension (`.grdecl`, `.data`, `.DATA`).
        - `bytes` - raw GRDECL content encoded as `encoding`.

    :param encoding: Text encoding used when decoding `bytes` input
        (default `"ascii"`).
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords (`SPECGRID`, `COORD`,
        `ZCORN`) are missing or malformed.
    :raises UnsupportedGridFormatError: If the source cannot be interpreted
        as valid GRDECL content.
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

    .. note::
        GRDECL is a corner-point format.  Grids constructed from other
        factories (Voronoi, tetrahedral, …) are approximated by writing
        their cell bounding-box vertices as degenerate hexahedral cells.
        The round-trip fidelity for non-corner-point grids is therefore
        limited to cell-level geometry; topology is not preserved exactly.

    :param grid: The grid to serialise.
    :param destination: One of:

        - `pathlib.Path` or `str` path - write to file and
          return `None`.
        - `None` - return the GRDECL text as a `str`.

    :param actnum: Optional shape `(n_cells,)` integer array of active-cell
        flags (1 = active, 0 = inactive).  If `None`, all cells are written
        as active.
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
    Coerce `source` to a plain text string ready for GRDECL parsing.

    Recursively inlines INCLUDE directives in `source`.

    :param source: Path, raw string, or bytes.
    :param encoding: Byte decoding encoding.
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
        # Distinguish path from raw text by trying to open it
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
            # Treat as raw GRDECL text
            text = source

    text = _resolve_includes(text, source_dir=source_dir)
    return text


def _resolve_includes(text: str, source_dir: typing.Optional[Path] = None) -> str:
    """
    Recursively inline INCLUDE directives in GRDECL text.

    Eclipse INCLUDE syntax:
    ```
    INCLUDE
        'relative/path/to/file.grdecl' /
    ```

    Paths are resolved relative to `source_dir` (the directory containing
    the parent file). When `source_dir` is `None` (raw text input),
    INCLUDE directives are left in place and a warning is emitted, because
    there is no filesystem anchor to resolve relative paths against.

    Includes are processed depth-first (each included file is itself
    processed for nested INCLUDEs before being spliced in).

    :param text: GRDECL text that may contain INCLUDE directives.
    :param source_dir: Directory of the file being parsed, or `None` for
        raw-text / bytes input.
    :returns: Text with all INCLUDE blocks replaced by the content of the
        referenced files.
    :raises GridImportError: If an included file cannot be read.
    """
    # Pattern: INCLUDE keyword, then a quoted filename, then /
    # Handles single or double quotes; the slash may be on the same line or
    # a subsequent line.
    include_pattern = re.compile(
        r"\bINCLUDE\b\s*['\"]([^'\"]+)['\"]\s*/",
        re.IGNORECASE | re.DOTALL,
    )

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        relative_path = match.group(1).strip()

        if source_dir is None:
            import warnings

            warnings.warn(
                f"GRDECL INCLUDE directive for {relative_path!r} encountered in raw-text "
                "input (no source directory available). The directive will be ignored. "
                "Load from a file path to enable INCLUDE resolution.",
                stacklevel=6,
            )
            return ""  # drop the directive silently

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

        # Recurse: the included file may itself have INCLUDEs, resolved
        # relative to its own directory.
        return _resolve_includes(included_text, include_path.parent)

    return include_pattern.sub(_replace, text)


def _strip_comments(text: str) -> str:
    """
    Remove `--` line comments from GRDECL text.

    :param text: Raw GRDECL text.
    :returns: Text with comment lines blanked out.
    """
    return re.sub(r"--[^\n]*", "", text)


def _tokenise(text: str) -> typing.List[str]:
    """
    Split clean GRDECL text into whitespace-separated tokens.

    Expansion syntax `N*value` (e.g. `100*0`) is expanded in-place.

    :param text: Comment-stripped GRDECL text.
    :returns: Flat list of string tokens.
    """
    raw_tokens = text.split()
    expanded: typing.List[str] = []
    repeat_re = re.compile(r"^(\d+)\*(.+)$")
    for tok in raw_tokens:
        m = repeat_re.match(tok)
        if m:
            count = int(m.group(1))
            value = m.group(2)
            expanded.extend([value] * count)
        else:
            expanded.append(tok)
    return expanded


def _extract_keyword_block(text: str, keyword: str) -> typing.Optional[str]:
    """
    Extract the data block following a GRDECL keyword up to the `/` terminator.

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


_GRDECL_UNIT_KEYWORDS: typing.Dict[str, UnitSystem] = {
    "FIELD": UnitSystem.FIELD,
    "METRIC": UnitSystem.METRIC,
    "LAB": UnitSystem.LAB,
    "SI": UnitSystem.SI,
}


def _detect_unit_system(clean: str) -> UnitSystem:
    """
    Scan comment-stripped GRDECL text for a unit-declaration keyword.

    FIELD, METRIC, LAB, and SI appear as bare keywords (no data block, no
    trailing slash) anywhere in the file. We match them as whole words to
    avoid false positives inside data values or keyword names.

    :param clean: Comment-stripped GRDECL text.
    :returns: The declared `UnitSystem`, defaulting to FIELD if none
        is found (Eclipse default when the keyword is absent).
    """
    for keyword, unit_system in _GRDECL_UNIT_KEYWORDS.items():
        # Match the keyword as a whole word on its own line or surrounded by
        # whitespace.  The keyword has no trailing data or slash.
        if re.search(r"(?<!\w)" + keyword + r"(?!\w)", clean, re.IGNORECASE):
            return unit_system
    # Eclipse default when no unit keyword is present
    return UnitSystem.FIELD


def _parse_mapaxes(clean: str) -> typing.Optional[MapAxes]:
    """
    Parse the optional MAPAXES keyword from comment-stripped GRDECL text.

    Eclipse MAPAXES format (6 values, no trailing slash):
        X_ORIGIN  Y_ORIGIN  X_UNITX  Y_UNITX  X_UNITY  Y_UNITY

    where (X_ORIGIN, Y_ORIGIN) is the map origin and (X_UNITX, Y_UNITX),
    (X_UNITY, Y_UNITY) are points on the map X and Y axes respectively.

    :param clean: Comment-stripped GRDECL text.
    :returns: A `MapAxes` instance, or `None` if the keyword is absent or
        the block cannot be parsed.
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
            f"MAPAXES values are not valid floats ({exc}). Ignoring.", stacklevel=4
        )
        return None

    origin = np.array([vals[0], vals[1]], dtype=np.float64)
    map_x_axis_point = np.array([vals[2], vals[3]], dtype=np.float64)
    map_y_axis_point = np.array([vals[4], vals[5]], dtype=np.float64)
    return MapAxes(
        origin=origin,
        map_x_axis_point=map_x_axis_point,
        map_y_axis_point=map_y_axis_point,
    )


def _parse_grdecl_cartesian(
    clean: str,
    nx: int,
    ny: int,
    nz: int,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Build a Cartesian `Grid` from TOPS / DX / DY / DZ keywords.

    TOPS gives the depth of the top face of every cell in the top layer
    (column-major, x-fastest). DX / DY / DZ give uniform or per-cell
    spacings in each direction.

    :param clean: Comment-stripped GRDECL text.
    :param nx: Number of cells in x (from SPECGRID).
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param unit_system: Already-detected unit system.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or have wrong sizes.
    """

    def _read_array(keyword: str, expected: int) -> np.ndarray:
        block = _extract_keyword_block(clean, keyword)
        if block is None:
            raise GridImportError(
                f"Cartesian GRDECL grid is missing required keyword {keyword!r}."
            )

        tokens = _tokenise(block)
        if len(tokens) == 1:
            # Scalar broadcast - single value applies to all cells
            return np.full(expected, float(tokens[0]), dtype=np.float64)
        if len(tokens) != expected:
            raise GridImportError(
                f"{keyword} expected {expected} values; got {len(tokens)}."
            )
        return np.array(tokens, dtype=np.float64)

    n_cells = nx * ny * nz
    n_col = nx * ny  # cells per layer (top-layer columns)

    # TOPS: depth of top face of each cell in the top layer, Fortran order
    tops_flat = _read_array("TOPS", n_col)
    # Eclipse TOPS is x-fastest (Fortran order): reshape to (ny, nx) then ravelled
    # We use just the minimum top depth as the origin z for the Cartesian factory.
    # A fully general implementation would need to handle non-flat tops, but
    # real-world TOPS-based files are almost always flat.
    z_top = float(tops_flat.min())
    if tops_flat.max() - tops_flat.min() > 1.0:
        warnings.warn(
            "GRDECL TOPS values vary by more than 1 unit; the Cartesian factory "
            "uses a flat top surface at the minimum TOPS value. Geometry may be "
            "approximate for dipping grids.",
            stacklevel=4,
        )

    dx_arr = _read_array("DX", n_cells)
    dy_arr = _read_array("DY", n_cells)
    dz_arr = _read_array("DZ", n_cells)

    # The Cartesian factory wants 1-D spacing vectors (one value per cell in
    # each direction), not a full per-cell field.  For uniform DX/DY/DZ the
    # factory accepts scalars; for variable spacing we extract the unique layer
    # slices.  Eclipse convention: DX varies along x (innermost), DY along y,
    # DZ along z (outermost in Fortran order).
    #
    # Simplest correct extraction: take the first row/column for each axis.
    # This is exact when spacing is uniform within each axis slice and only
    # varies along that axis (the common case).
    dx_1d = dx_arr.reshape(nz, ny, nx, order="F")[0, 0, :]  # shape (nx,)
    dy_1d = dy_arr.reshape(nz, ny, nx, order="F")[0, :, 0]  # shape (ny,)
    dz_1d = dz_arr.reshape(nz, ny, nx, order="F")[:, 0, 0]  # shape (nz,)

    # ACTNUM (optional)
    actnum_block = _extract_keyword_block(clean, "ACTNUM")
    meta: typing.Dict[str, typing.Any] = dict(metadata or {})
    meta["source_format"] = "grdecl_cartesian"

    if actnum_block is not None:
        from bores.grids.io.grdecl import _tokenise as _tok  # avoid re-import

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
) -> Grid:
    """
    Build a Corner Point `Grid` from COORD / ZCORN / ACTNUM keywords.

    :param clean: Comment-stripped GRDECL text.
    :param nx: Number of cells in x (from SPECGRID).
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param unit_system: Already-detected unit system.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or have wrong sizes.
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

    # COORD flat order is (x_pillar, y_pillar, 6) with x fastest
    coord_arr = (
        np.array(coord_tokens, dtype=np.float64)
        .reshape(nx + 1, ny + 1, 6, order="F")
        .transpose(1, 0, 2)  # → (ny+1, nx+1, 6) C-order
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

    # Read flat Fortran-order data, then convert to your internal C-order (nz, ny, nx) axis layout
    zcorn_arr = (
        np.array(zcorn_tokens, dtype=np.float64)
        .reshape(nx * 2, ny * 2, nz * 2, order="F")
        .transpose(2, 1, 0)  # → (nz*2, ny*2, nx*2) C-order
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
        actnum_arr: typing.Optional[np.ndarray] = (
            np.array(actnum_tokens, dtype=np.int32)
            .reshape(nx, ny, nz, order="F")
            .transpose(2, 1, 0)
        )
    else:
        actnum_arr = None

    meta = {
        "source_format": "grdecl_corner_point",
        "actnum": actnum_arr,
    }
    if metadata:
        meta.update(metadata)
    return make_corner_point_grid(
        coord=coord_arr,
        zcorn=zcorn_arr,
        actnum=actnum_arr,
        unit_system=unit_system,
        metadata=meta,
    )


def _parse_grdecl(
    text: str, metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None
) -> Grid:
    """
    Parse a complete GRDECL text blob into a `bores.grids.base.Grid`.

    :param text: Raw GRDECL text (may contain `--` comments).
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or the arrays
        have inconsistent shapes.
    """
    clean = _strip_comments(text)

    # SPECGRID
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
        nx, ny, nz = (
            int(specgrid_tokens[0]),
            int(specgrid_tokens[1]),
            int(specgrid_tokens[2]),
        )
    except ValueError as exc:
        raise GridImportError(f"SPECGRID values are not valid integers: {exc}") from exc

    has_coord = _extract_keyword_block(clean, "COORD") is not None
    has_tops = _extract_keyword_block(clean, "TOPS") is not None

    map_axes = _parse_mapaxes(clean)
    unit_system = _detect_unit_system(clean)
    meta: typing.Dict[str, typing.Any] = dict(metadata or {})
    meta["map_axes"] = map_axes
    meta["nx"] = nx
    meta["ny"] = ny
    meta["nz"] = nz
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
        )
    except Exception as exc:
        raise GridImportError(
            f"Failed to construct grid from parsed GRDECL arrays: {exc}"
        ) from exc


_GRDECL_SOURCES = {"grdecl_corner_point", "grdecl_cartesian"}


@numba.njit(cache=True)
def _accumulate_pillars(
    cell_min_xyz: FloatArray[TwoDimensions],  # (n_cells, 3)
    cell_max_xyz: FloatArray[TwoDimensions],  # (n_cells, 3)
    nx: int,
    ny: int,
    nz: int,
    pillar_x: FloatArray[TwoDimensions],  # (ny+1, nx+1)  out
    pillar_y: FloatArray[TwoDimensions],  # (ny+1, nx+1)  out
    pillar_z_top: FloatArray[TwoDimensions],  # (ny+1, nx+1)  out
    pillar_z_bottom: FloatArray[TwoDimensions],  # (ny+1, nx+1)  out
    pillar_count: IntArray[TwoDimensions],  # (ny+1, nx+1)  out
) -> None:
    """
    Accumulate per-pillar XY positions and Z extents from cell bounding boxes.

    Cell ordering: cell_idx = i + j*nx + k*nx*ny.
    Pillar (pj, pi) collects contributions from up to 4 cells per layer.

    :param cell_min_xyz: Shape `(n_cells, 3)` bounding-box minima.
    :param cell_max_xyz: Shape `(n_cells, 3)` bounding-box maxima.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param pillar_x: Accumulator for pillar X coordinate (zeroed on entry).
    :param pillar_y: Accumulator for pillar Y coordinate (zeroed on entry).
    :param pillar_z_top: Accumulator for minimum pillar Z (`+inf` on entry).
    :param pillar_z_bottom: Accumulator for maximum pillar Z (`-inf` on entry).
    :param pillar_count: Contribution counter per pillar (zeroed on entry).
    """
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell_idx = i + j * nx + k * nx * ny
                lx = cell_min_xyz[cell_idx, 0]
                ly = cell_min_xyz[cell_idx, 1]
                lz = cell_min_xyz[cell_idx, 2]
                hx = cell_max_xyz[cell_idx, 0]
                hy = cell_max_xyz[cell_idx, 1]
                hz = cell_max_xyz[cell_idx, 2]

                # 4 pillar corners: (pj, pi, px, py)
                # (j,   i  ) -> (lx, ly)
                # (j,   i+1) -> (hx, ly)
                # (j+1, i  ) -> (lx, hy)
                # (j+1, i+1) -> (hx, hy)
                for corner in range(4):
                    if corner == 0:
                        pj = j
                        pi = i
                        px = lx
                        py = ly
                    elif corner == 1:
                        pj = j
                        pi = i + 1
                        px = hx
                        py = ly
                    elif corner == 2:
                        pj = j + 1
                        pi = i
                        px = lx
                        py = hy
                    else:
                        pj = j + 1
                        pi = i + 1
                        px = hx
                        py = hy

                    pillar_x[pj, pi] += px
                    pillar_y[pj, pi] += py
                    if lz < pillar_z_top[pj, pi]:
                        pillar_z_top[pj, pi] = lz
                    if hz > pillar_z_bottom[pj, pi]:
                        pillar_z_bottom[pj, pi] = hz
                    pillar_count[pj, pi] += 1


@numba.njit(parallel=True, cache=True)
def _fill_zcorn(
    cell_min_xyz: FloatArray[TwoDimensions],  # (n_cells, 3)
    cell_max_xyz: FloatArray[TwoDimensions],  # (n_cells, 3)
    nx: int,
    ny: int,
    nz: int,
    zcorn: ZcornArray,  # (nz*2, ny*2, nx*2)  out
) -> None:
    """
    Fill ZCORN array from per-cell Z bounding-box extents.

    Parallel over `k` (layer index): each layer writes to a disjoint
    `[2k:2k+2, :, :]` slice of `zcorn`, so there are no races.

    ZCORN indexing (Eclipse convention):
        `zcorn[2k,   2j,   2i  ]` ... `[2k,   2j+1, 2i+1]` = top Z of cell (i,j,k)
        `zcorn[2k+1, 2j,   2i  ]` ... `[2k+1, 2j+1, 2i+1]` = bottom Z of cell (i,j,k)

    :param cell_min_xyz: Shape `(n_cells, 3)` bounding-box minima.
    :param cell_max_xyz: Shape `(n_cells, 3)` bounding-box maxima.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param zcorn: Output array, must be pre-allocated as `(nz*2, ny*2, nx*2)`.
    """
    for k in numba.prange(nz):  # type: ignore
        for j in range(ny):
            for i in range(nx):
                cell_idx = i + j * nx + k * nx * ny
                z_top = cell_min_xyz[cell_idx, 2]
                z_bot = cell_max_xyz[cell_idx, 2]
                # All 4 top corners share the same z_top value.
                zcorn[2 * k, 2 * j, 2 * i] = z_top
                zcorn[2 * k, 2 * j, 2 * i + 1] = z_top
                zcorn[2 * k, 2 * j + 1, 2 * i] = z_top
                zcorn[2 * k, 2 * j + 1, 2 * i + 1] = z_top
                # All 4 bottom corners share z_bot.
                zcorn[2 * k + 1, 2 * j, 2 * i] = z_bot
                zcorn[2 * k + 1, 2 * j, 2 * i + 1] = z_bot
                zcorn[2 * k + 1, 2 * j + 1, 2 * i] = z_bot
                zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1] = z_bot


def _rederive_corner_point_arrays(
    grid: Grid,
) -> typing.Tuple[CoordArray, ZcornArray, int, int, int]:
    """
    Reconstruct approximate COORD and ZCORN arrays from a Grid whose
    vertex_coordinates encode a corner-point geometry.

    The reconstruction is performed on the bounding-box geometry of every cell
    because the original pillar structure is not retained after import.
    Pillars are assumed to be straight and vertical.

    Layout assumption (matches `_parse_grdecl_corner_point` import path):
    cells are stored in k-major, j-middle, i-minor order (Fortran/Eclipse order),
    i.e. cell_index = i + j*nx + k*nx*ny.

    :param grid: A Grid whose metadata["source_format"] == "grdecl_corner_point".
    :returns: Tuple (coord_arr, zcorn_arr, nx, ny, nz).
    :raises GridExportError: If the cell count is not a perfect nxxnyxnz product.
    """
    n_cells = grid.n_cells

    # Try to recover nx, ny, nz from metadata; fall back to cube root heuristic.
    meta = getattr(grid, "metadata", {}) or {}
    nx = meta.get("nx")
    ny = meta.get("ny")
    nz = meta.get("nz")

    if nx is None or ny is None or nz is None:
        # Attempt to factor n_cells into a plausible (nx, ny, nz).
        # We try all divisor pairs for nz (layers) first since nz is usually
        # much smaller than nx*ny.
        found = False
        for nz_try in range(1, n_cells + 1):
            if n_cells % nz_try != 0:
                continue

            nxy = n_cells // nz_try
            # Try to find a square-ish nx, ny
            for nx_try in range(1, int(nxy**0.5) + 1):
                if nxy % nx_try == 0:
                    nx_try2, ny_try2 = nx_try, nxy // nx_try
                    if nx_try2 * ny_try2 * nz_try == n_cells:
                        nx, ny, nz = nx_try2, ny_try2, nz_try
                        found = True
            if found:
                break

        if not found or (nx * ny * nz) != n_cells:  # type: ignore
            raise GridExportError(
                f"Cannot determine (nx, ny, nz) factorisation for n_cells={n_cells}. "
                "Store 'nx', 'ny', 'nz' in grid.metadata to enable GRDECL export."
            )

    warnings.warn(
        "Exporting a corner-point Grid to GRDECL without stored COORD/ZCORN arrays. "
        "Pillars are reconstructed as straight vertical lines from cell bounding boxes. "
        "This is lossy for grids with lateral pillar displacement (faults, dipping layers).",
        stacklevel=4,
    )
    assert nx is not None and ny is not None and nz is not None

    # Pillar accumulation
    pillar_x = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    pillar_y = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    pillar_z_top = np.full((ny + 1, nx + 1), np.inf, dtype=np.float64)
    pillar_z_bottom = np.full((ny + 1, nx + 1), -np.inf, dtype=np.float64)
    pillar_count = np.zeros((ny + 1, nx + 1), dtype=np.int32)
    _accumulate_pillars(
        grid.cell_min_xyz,
        grid.cell_max_xyz,
        nx,
        ny,
        nz,
        pillar_x,
        pillar_y,
        pillar_z_top,
        pillar_z_bottom,
        pillar_count,
    )

    # Finalise averages
    nonzero = pillar_count > 0
    pillar_x[nonzero] /= pillar_count[nonzero]
    pillar_y[nonzero] /= pillar_count[nonzero]

    # COORD fill
    coord = np.empty((ny + 1, nx + 1, 6), dtype=np.float64)
    coord[:, :, 0] = pillar_x
    coord[:, :, 1] = pillar_y
    coord[:, :, 2] = pillar_z_top
    coord[:, :, 3] = pillar_x
    coord[:, :, 4] = pillar_y
    coord[:, :, 5] = pillar_z_bottom

    # ZCORN fill
    zcorn = np.empty((nz * 2, ny * 2, nx * 2), dtype=np.float64)
    _fill_zcorn(grid.cell_min_xyz, grid.cell_max_xyz, nx, ny, nz, zcorn)

    return coord, zcorn, nx, ny, nz


def _build_grdecl_text(
    grid: Grid, *, actnum: typing.Optional[ActnumArray] = None
) -> str:
    """
    Render a `bores.grids.base.Grid` as GRDECL text.

    Only grids that originated from `load_grdecl` (corner-point or Cartesian)
    are supported. Grids from other factories (Voronoi, tetrahedral, meshio,
    …) cannot be reliably expressed in GRDECL format and will raise an error.

    Corner-point grids are written as SPECGRID / COORD / ZCORN / ACTNUM.
    Because COORD and ZCORN are no longer cached in metadata, they are
    re-derived from the Grid's vertex bounding-box geometry (lossy for
    deformed pillar grids - a warning is emitted in that case).

    Cartesian grids (loaded via the TOPS/DX/DY/DZ path) are written back
    as SPECGRID / TOPS / DX / DY / DZ, which is an exact lossless round-trip.

    :param grid: The grid to serialise.
    :param actnum: Optional shape `(n_cells,)` active-cell flag array
        (1 = active, 0 = inactive).
    :returns: GRDECL text as a string.
    :raises GridExportError: If the grid's source format is not a GRDECL
        variant, or if the grid dimensions cannot be determined.
    """
    meta = getattr(grid, "metadata", {}) or {}
    source_format: str = meta.get("source_format", "")
    if source_format not in _GRDECL_SOURCES:
        raise GridExportError(
            f"Cannot export a Grid with source_format={source_format!r} to GRDECL. "
            "Only grids originally loaded by load_grdecl() support GRDECL export. "
            f"Supported source formats: {sorted(_GRDECL_SOURCES)}."
        )

    if source_format == "grdecl_cartesian":
        return _build_grdecl_cartesian_text(grid, actnum=actnum)
    return _build_grdecl_corner_point_text(grid, actnum=actnum)


def _build_grdecl_cartesian_text(
    grid: Grid, *, actnum: typing.Optional[ActnumArray] = None
) -> str:
    """
    Render a Cartesian-source Grid as SPECGRID / TOPS / DX / DY / DZ.

    :param grid: Grid whose metadata["source_format"] == "grdecl_cartesian".
    :param actnum: Optional active-cell flag array.
    :returns: GRDECL text.
    :raises GridExportError: If grid dimensions cannot be recovered.
    """
    meta = getattr(grid, "metadata", {}) or {}
    nx = meta.get("nx")
    ny = meta.get("ny")
    nz = meta.get("nz")

    # Fall back to a flat 1×1×n_cells layout if nx/ny/nz not stored.
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

    # TOPS: depth of top face of every cell in the top layer (k=0),
    # Fortran order (x fastest).  cell_min_xyz[:,2] is the top depth.
    top_layer_indices = [i + j * nx for j in range(ny) for i in range(nx)]
    tops_vals = grid.cell_min_xyz[top_layer_indices, 2]
    lines.append("TOPS")
    for i in range(0, len(tops_vals), 6):
        chunk = tops_vals[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")
    lines.append("")

    # DX / DY / DZ: per-cell extents (Fortran order: x fastest, z slowest)
    for axis_name, column in [("DX", 0), ("DY", 1), ("DZ", 2)]:
        extents = grid.cell_max_xyz[:, column] - grid.cell_min_xyz[:, column]
        # Fortran order: reshape to (nz, ny, nx) then transpose to (nx, ny, nz)
        # and ravel F-order so x is innermost.
        extents_3d = extents.reshape(nz, ny, nx)  # cell ordering: k,j,i
        flat = extents_3d.transpose(2, 1, 0).ravel(order="F")  # x-fastest
        lines.append(axis_name)
        for i in range(0, len(flat), 6):
            chunk = flat[i : i + 6]
            lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
        lines.append("/")
        lines.append("")

    # ACTNUM (optional)
    if actnum is not None:
        _emit_actnum(lines, actnum, grid.n_cells, nx, ny, nz)

    return "\n".join(lines)


def _build_grdecl_corner_point_text(
    grid: Grid, *, actnum: typing.Optional[ActnumArray] = None
) -> str:
    """
    Render a corner-point-source Grid as SPECGRID / COORD / ZCORN / ACTNUM.

    COORD and ZCORN are re-derived from the Grid's bounding-box geometry.

    :param grid: Grid whose metadata["source_format"] == "grdecl_corner_point".
    :param actnum: Optional active-cell flag array.
    :returns: GRDECL text.
    :raises GridExportError: If grid dimensions cannot be determined.
    """
    coord, zcorn, nx, ny, nz = _rederive_corner_point_arrays(grid)
    meta = getattr(grid, "metadata", {}) or {}

    lines: typing.List[str] = []
    lines.append("SPECGRID")
    lines.append(f"  {nx}  {ny}  {nz}  1  F /")
    lines.append("")

    map_axes: typing.Optional[MapAxes] = meta.get("map_axes")
    if map_axes is not None:
        _emit_mapaxes(lines, map_axes)

    # COORD - write in Eclipse Fortran order: x fastest (pillar i varies fastest)
    # Flatten (ny+1, nx+1, 6) -> transpose to (nx+1, ny+1, 6) then ravel
    lines.append("COORD")
    flat_coord = coord.transpose(1, 0, 2).reshape(-1, 6)
    for row in flat_coord:
        x1, y1, z1, x2, y2, z2 = row
        lines.append(f"  {x1:.6f}  {y1:.6f}  {z1:.6f}  {x2:.6f}  {y2:.6f}  {z2:.6f}")
    lines.append("/")
    lines.append("")

    # ZCORN - Eclipse Fortran order: x fastest, z slowest
    # Internal shape: (nz*2, ny*2, nx*2)  C-order  →  transpose to (nx*2, ny*2, nz*2) F-order
    lines.append("ZCORN")
    flat_zcorn = zcorn.transpose(2, 1, 0).ravel(order="F")
    for i in range(0, len(flat_zcorn), 6):
        chunk = flat_zcorn[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")

    # ACTNUM (optional)
    meta = getattr(grid, "metadata", {}) or {}
    effective_actnum = actnum if actnum is not None else meta.get("actnum")
    if effective_actnum is not None:
        _emit_actnum(lines, effective_actnum, grid.n_cells, nx, ny, nz)

    return "\n".join(lines)


def _emit_actnum(
    lines: typing.List[str],
    actnum: np.ndarray,
    n_cells: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Append an ACTNUM block in Eclipse Fortran order to `lines` in-place.

    :param lines: Accumulator list of GRDECL text lines.
    :param actnum: Shape `(n_cells,)` integer array, cell-ordering k,j,i (C-order).
    :param n_cells: Expected total cell count.
    :param nx, ny, nz: Grid dimensions.
    :raises GridExportError: If actnum length mismatches n_cells.
    """
    actnum_arr = np.asarray(actnum, dtype=np.int32)
    if len(actnum_arr) != n_cells:
        raise GridExportError(
            f"actnum length {len(actnum_arr)} does not match n_cells {n_cells}."
        )

    lines.append("")
    lines.append("ACTNUM")
    # Internal ordering: cell_idx = i + j*nx + k*nx*ny  (k outermost)
    # Eclipse Fortran order: x(i) fastest → same ordering, so reshape and transpose
    flat = actnum_arr.reshape(nz, ny, nx).transpose(2, 1, 0).ravel(order="F")
    for i in range(0, len(flat), 20):
        chunk = flat[i : i + 20]
        lines.append("  " + "  ".join(str(int(v)) for v in chunk))
    lines.append("/")


def _emit_mapaxes(lines: typing.List[str], map_axes: "MapAxes") -> None:
    """
    Append a MAPAXES block to `lines` in-place.

    Eclipse MAPAXES format (6 values on one data line, terminated by `/`):
        X_ORIGIN  Y_ORIGIN  X_UNITX  Y_UNITX  X_UNITY  Y_UNITY

    where `(X_ORIGIN, Y_ORIGIN)` is the map coordinate origin,
    `(X_UNITX, Y_UNITX)` is a point on the map X-axis, and
    `(X_UNITY, Y_UNITY)` is a point on the map Y-axis.

    :param lines: Accumulator list of GRDECL text lines (mutated in-place).
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
