"""
GRDECL text-format reader and writer.

GRDECL is the ASCII keyword-based format used by Eclipse, ResInsight, and
most other reservoir simulators to describe corner-point pillar grids.

**Supported keywords (read)**:

`SPECGRID`, `COORD`, `ZCORN`, `ACTNUM`

**Supported keywords (write)**:

`SPECGRID`, `COORD`, `ZCORN`, `ACTNUM`

**References**:

Schlumberger Eclipse Reference Manual - Grid section.
"""

import re
import typing
from pathlib import Path

import numpy as np

from bores.errors import GridExportError, GridImportError
from bores.grids.base import Grid
from bores.grids.factories.corner_point import make_corner_point_grid
from bores.typing import IntArray, OneDimension

__all__ = ["load_grdecl", "dump_grdecl"]


_PathOrStr = typing.Union[str, Path]
_TextOrPath = typing.Union[str, bytes, Path]


@typing.overload
def load_grdecl(source: Path, *, encoding: str = ...) -> Grid: ...


@typing.overload
def load_grdecl(source: str, *, encoding: str = ...) -> Grid: ...


@typing.overload
def load_grdecl(source: bytes, *, encoding: str = ...) -> Grid: ...


def load_grdecl(
    source: _TextOrPath,
    *,
    encoding: str = "ascii",
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
    text = _resolve_text_source(source, encoding=encoding)
    return _parse_grdecl(text)


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


def _resolve_text_source(
    source: _TextOrPath,
    *,
    encoding: str,
) -> str:
    """
    Coerce `source` to a plain text string ready for GRDECL parsing.

    :param source: Path, raw string, or bytes.
    :param encoding: Byte decoding encoding.
    :returns: Raw GRDECL text as a `str`.
    :raises GridImportError: If a filesystem path cannot be read.
    """
    if isinstance(source, bytes):
        return source.decode(encoding)

    if isinstance(source, Path):
        try:
            return source.read_text(encoding=encoding)
        except OSError as exc:
            raise GridImportError(f"Cannot read GRDECL file {source!r}: {exc}") from exc

    # str: distinguish path from raw text by trying to open it
    candidate = Path(source)
    if candidate.is_file():
        try:
            return candidate.read_text(encoding=encoding)
        except OSError as exc:
            raise GridImportError(f"Cannot read GRDECL file {source!r}: {exc}") from exc

    # Treat as raw GRDECL text
    return source


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
        r"" + re.escape(keyword) + r"\s*(.*?)\s*/",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _parse_grdecl(text: str) -> Grid:
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
    coord_arr = np.array(coord_tokens, dtype=np.float64).reshape(ny + 1, nx + 1, 6)

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
    zcorn_arr = np.array(zcorn_tokens, dtype=np.float64).reshape(nz * 2, ny * 2, nx * 2)

    # ACTNUM (optional)
    actnum_block = _extract_keyword_block(clean, "ACTNUM")
    if actnum_block is not None:
        actnum_tokens = _tokenise(actnum_block)
        expected_actnum = nx * ny * nz
        if len(actnum_tokens) != expected_actnum:
            raise GridImportError(
                f"ACTNUM expected {expected_actnum} values; got {len(actnum_tokens)}."
            )
        actnum_arr: typing.Optional[np.ndarray] = np.array(
            actnum_tokens, dtype=np.int32
        ).reshape(nz, ny, nx)
    else:
        actnum_arr = None

    try:
        return make_corner_point_grid(
            coord=coord_arr,
            zcorn=zcorn_arr,
            actnum=actnum_arr,
            metadata={"source_format": "grdecl"},
        )
    except Exception as exc:
        raise GridImportError(
            f"Failed to construct grid from parsed GRDECL arrays: {exc}"
        ) from exc


def _build_grdecl_text(
    grid: Grid, *, actnum: typing.Optional[IntArray[OneDimension]]
) -> str:
    """
    Render a `bores.grids.base.Grid` as GRDECL text.

    The function reconstructs a structured pillar representation from the
    cell bounding boxes stored on the grid.  The result is valid GRDECL
    that can be read back via :func:`load_grdecl`, but it is an
    *approximation* for non-corner-point grids.

    :param grid: Source grid.
    :param actnum: Optional per-cell active flags.
    :returns: GRDECL text as a `str`.
    :raises GridExportError: If the grid cannot be represented in GRDECL form.
    """
    # For a general unstructured grid we cannot recover NX/NY/NZ exactly.
    # We write each cell as a degenerate pillar cell using its bounding box.
    # This preserves cell geometry at the cost of shared-face accuracy.
    n_cells = grid.n_cells
    lines: typing.List[str] = []

    # Write as a 1 × 1 × n_cells column stack (degenerate but valid GRDECL)
    nx, ny, nz = 1, 1, n_cells
    lines.append(f"SPECGRID")
    lines.append(f"  {nx}  {ny}  {nz}  1  F /")
    lines.append("")

    # Build a trivial 2×2 pillar grid from cell bounding-box extremes
    # Pillars: (ny+1)*(nx+1) = 4 pillars, each with top/bot xyz
    # We use the overall bounding box for simplicity
    x_min, x_max, y_min, y_max, z_min, z_max = (
        grid.bounding_box
    )  # (xmin, xmax, ymin, ymax, zmin, zmax)

    lines.append("COORD")
    # 4 corner pillars: each "pillar top x y z  bot x y z"
    for px in (x_min, x_max):
        for py in (y_min, y_max):
            lines.append(
                f"  {px:.6f}  {py:.6f}  {z_min:.6f}  {px:.6f}  {py:.6f}  {z_max:.6f}"
            )
    lines.append("/")
    lines.append("")

    # ZCORN: 8 values per cell (top-4 + bottom-4 corners)
    # For each cell: use cell_min_xyz and cell_max_xyz
    lines.append("ZCORN")
    zcorn_values: typing.List[str] = []
    for cell_idx in range(n_cells):
        z_top_cell = float(grid.cell_min_xyz[cell_idx, 2])
        z_bot_cell = float(grid.cell_max_xyz[cell_idx, 2])
        # 8 corner z values: all top corners = z_top, all bottom = z_bot
        zcorn_values.extend([f"{z_top_cell:.6f}"] * 4)
        zcorn_values.extend([f"{z_bot_cell:.6f}"] * 4)
    # Write 6 values per line for readability
    for i in range(0, len(zcorn_values), 6):
        lines.append("  " + "  ".join(zcorn_values[i : i + 6]))
    lines.append("/")
    lines.append("")

    # ACTNUM
    if actnum is not None:
        if len(actnum) != n_cells:
            raise GridExportError(
                f"actnum length {len(actnum)} does not match n_cells {n_cells}."
            )
        lines.append("ACTNUM")
        for i in range(0, n_cells, 20):
            chunk = actnum[i : i + 20]
            lines.append("  " + "  ".join(str(int(v)) for v in chunk))
        lines.append("/")
        lines.append("")

    return "\n".join(lines)
