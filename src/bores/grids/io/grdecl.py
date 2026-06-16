"""
GRDECL text-format reader and writer.

GRDECL is the ASCII keyword-based format used by Eclipse, ResInsight, and
most other reservoir simulators to describe corner-point pillar grids.

**Supported keywords**:

`SPECGRID`, `COORD`, `ZCORN`, `ACTNUM`, `GRIDTYPE`, `TOPS` , `DX` , `DY` , `DZ`

**References**:

Schlumberger Eclipse Reference Manual - Grid section.
"""

import re
import typing
import warnings
from pathlib import Path

import attrs
import numpy as np

from bores.errors import GridExportError, GridImportError
from bores.grids.base import Grid
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import make_corner_point_grid
from bores.grids.utils import convert
from bores.typing import FloatArray, IntArray, OneDimension, ThreeDimensions, UnitSystem

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
    actnum: typing.Optional[IntArray[OneDimension]] = ...,
    encoding: str = ...,
) -> None: ...


@typing.overload
def dump_grdecl(
    grid: Grid,
    destination: None = None,
    *,
    actnum: typing.Optional[IntArray[OneDimension]] = ...,
    encoding: str = ...,
) -> str: ...


@typing.overload
def dump_grdecl(
    grid: Grid,
    destination: _PathOrStr,
    *,
    actnum: typing.Optional[IntArray[OneDimension]] = ...,
    encoding: str = ...,
) -> None: ...


def dump_grdecl(
    grid: Grid,
    destination: typing.Union[_PathOrStr, None] = None,
    *,
    actnum: typing.Optional[IntArray[OneDimension]] = None,
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
    unit_system = _detect_unit_system(clean)

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
    try:
        if not has_coord and has_tops:
            return _parse_grdecl_cartesian(
                clean=clean,
                nx=nx,
                ny=ny,
                nz=nz,
                unit_system=unit_system,
                metadata=metadata,
            )
        return _parse_grdecl_corner_point(
            clean=clean,
            nx=nx,
            ny=ny,
            nz=nz,
            unit_system=unit_system,
            metadata=metadata,
        )
    except Exception as exc:
        raise GridImportError(
            f"Failed to construct grid from parsed GRDECL arrays: {exc}"
        ) from exc


def _build_grdecl_text(
    grid: Grid,
    *,
    actnum: typing.Optional[IntArray[OneDimension]] = None,
) -> str:
    """
    Render a `bores.grids.base.Grid` as GRDECL text.

    If the grid originates from a GRDECL file and contains stored
    metadata (`coord`, `zcorn`, `actnum`), the function performs a
    lossless reconstruction.

    Otherwise, it falls back to an approximate representation using
    cell bounding boxes, producing a degenerate but valid GRDECL grid.

    :param grid: Source grid.
    :param actnum: Optional override active-cell array.
    :returns: GRDECL text as a string.
    :raises GridExportError: If grid dimensions are inconsistent.
    """
    lines: typing.List[str] = []
    meta = getattr(grid, "metadata", {}) or {}
    coord = meta.get("coord", None)
    zcorn = meta.get("zcorn", None)
    stored_actnum = meta.get("actnum", None)

    use_native = coord is not None and zcorn is not None
    if use_native:
        coord_arr = np.asarray(coord, dtype=np.float64)
        zcorn_arr = np.asarray(zcorn, dtype=np.float64)
        if stored_actnum is not None and actnum is None:
            actnum_arr = np.asarray(stored_actnum, dtype=np.int32)
        elif actnum is not None:
            actnum_arr = np.asarray(actnum, dtype=np.int32)
        else:
            actnum_arr = None

        ny_plus1, nx_plus1 = coord_arr.shape[:2]
        nx = nx_plus1 - 1
        ny = ny_plus1 - 1
        nz = zcorn_arr.shape[0] // 2

        lines.append("SPECGRID")
        lines.append(f"  {nx}  {ny}  {nz}  1  F /")
        lines.append("")

        lines.append("COORD")
        flat_coord = coord_arr.transpose(1, 0, 2).reshape(-1, 6)
        for i in range(0, len(flat_coord)):
            x1, y1, z1, x2, y2, z2 = flat_coord[i]
            lines.append(
                f"  {x1:.6f}  {y1:.6f}  {z1:.6f}  {x2:.6f}  {y2:.6f}  {z2:.6f}"
            )
        lines.append("/")
        lines.append("")

        lines.append("ZCORN")
        flat_zcorn = zcorn_arr.transpose(2, 1, 0).reshape(-1)
        for i in range(0, len(flat_zcorn), 6):
            chunk = flat_zcorn[i : i + 6]
            lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
        lines.append("/")

        if actnum_arr is not None:
            nz, ny, nx = actnum_arr.shape
            lines.append("")
            lines.append("ACTNUM")
            flat_actnum = actnum_arr.transpose(2, 1, 0).reshape(-1)
            for i in range(0, len(flat_actnum), 20):
                chunk = flat_actnum[i : i + 20]
                lines.append("  " + "  ".join(str(int(v)) for v in chunk))
            lines.append("/")

        return "\n".join(lines)

    n_cells = grid.n_cells
    x_min, x_max, y_min, y_max, z_min, z_max = grid.bounding_box

    lines.append("SPECGRID")
    lines.append(f"  1  1  {n_cells}  1  F /")
    lines.append("")

    lines.append("COORD")
    for px in (x_min, x_max):
        for py in (y_min, y_max):
            lines.append(
                f"  {px:.6f}  {py:.6f}  {z_min:.6f}  {px:.6f}  {py:.6f}  {z_max:.6f}"
            )
    lines.append("/")
    lines.append("")

    lines.append("ZCORN")
    zcorn_values = []
    for c in range(n_cells):
        zt = float(grid.cell_min_xyz[c, 2])
        zb = float(grid.cell_max_xyz[c, 2])
        zcorn_values.extend([zt, zt, zt, zt, zb, zb, zb, zb])

    for i in range(0, len(zcorn_values), 6):
        chunk = zcorn_values[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")

    if actnum is not None:
        if len(actnum) != n_cells:
            raise GridExportError(
                f"actnum length {len(actnum)} does not match n_cells {n_cells}."
            )
        lines.append("")
        lines.append("ACTNUM")
        for i in range(0, n_cells, 20):
            chunk = actnum[i : i + 20]
            lines.append("  " + "  ".join(str(int(v)) for v in chunk))
        lines.append("/")

    return "\n".join(lines)
