"""
Eclipse binary grid reader (.EGRID / .GRID).

Reads Schlumberger / SLB Eclipse binary grid files, which use a simple
block-structured binary format: each record is wrapped by two Fortran-style
4-byte length markers (big-endian on modern files, little-endian legacy).

**Supported file types**:

- `*.EGRID`  - Extended grid (Eclipse 2005+, preferred)
- `*.GRID`   - Legacy binary grid

The reader extracts `COORD`, `ZCORN`, and `ACTNUM` from the binary
stream and delegates construction to
`bores.grids.factories.corner_point.make_corner_point_grid`.

**References**:

Schlumberger Eclipse Technical Description - Grid file format.
"""

import struct
import typing
from pathlib import Path

import numpy as np

from bores.errors import GridImportError, UnsupportedGridFormatError
from bores.grids.base import Grid
from bores.grids.factories.corner_point import make_corner_point_grid
from bores.typing import IntArray, ThreeDimensions

__all__ = ["load_eclipse_grid"]


_BytesOrPath = typing.Union[bytes, Path, str]

# Known Eclipse binary keyword dtypes
_ECLIPSE_DTYPE_MAP: typing.Dict[bytes, np.dtype] = {
    b"REAL    ": np.dtype(">f4"),  # 4-byte big-endian float
    b"INTE    ": np.dtype(">i4"),  # 4-byte big-endian int
    b"DOUB    ": np.dtype(">f8"),  # 8-byte big-endian double
    b"LOGI    ": np.dtype(">i4"),  # logical (stored as int)
    b"CHAR    ": np.dtype("S8"),  # 8-byte character strings
    b"MESS    ": np.dtype("S8"),  # message (no data follows)
}

_INT_STRUCT_BE = struct.Struct(">i")
_INT_STRUCT_LE = struct.Struct("<i")


@typing.overload
def load_eclipse_grid(source: Path) -> Grid: ...


@typing.overload
def load_eclipse_grid(source: str) -> Grid: ...


@typing.overload
def load_eclipse_grid(source: bytes) -> Grid: ...


def load_eclipse_grid(source: _BytesOrPath) -> Grid:
    """
    Load an Eclipse binary grid (`.EGRID` / `.GRID`) from a path or bytes.

    :param source: One of:

        - `pathlib.Path` - path to an `.EGRID` or `.GRID` file.
        - `str` - filesystem path.
        - `bytes` - raw binary content of the Eclipse grid file.

    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If the binary stream is malformed, or required
        keywords (`COORD`, `ZCORN`) cannot be found.
    :raises UnsupportedGridFormatError: If the file does not appear to be a
        recognised Eclipse binary grid.
    """
    raw = _resolve_source(source)
    return _parse_binary(raw)


def _resolve_source(source: _BytesOrPath) -> bytes:
    """
    Coerce `source` to raw bytes ready for binary parsing.

    :param source: Path (Path/str) or raw bytes.
    :returns: Binary content as `bytes`.
    :raises GridImportError: If a path cannot be read.
    """
    if isinstance(source, bytes):
        return source
    path = Path(source)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise GridImportError(f"Cannot read Eclipse grid file {path!r}: {exc}") from exc


def _iter_records(raw: bytes) -> typing.Iterator[typing.Tuple[str, np.ndarray]]:
    """
    Iterate over keyword records in an Eclipse Fortran unformatted binary stream.

    Each record has the layout:

        [4-byte int: header_len] [16-byte header: name(8) + type(8)] [4-byte int: header_len]
        [4-byte int: data_len]   [data_len bytes of data]            [4-byte int: data_len]

    Endianness (big-endian EGRID or little-endian legacy GRID) is detected
    automatically from the first length marker, which must equal 16 (the fixed
    header record size of 8 keyword bytes + 8 type bytes).

    :param raw: Complete file bytes.
    :yields: `(keyword_name, data_array)` tuples, with numeric arrays
        normalised to native byte order. `MESS` keywords and unrecognised
        type codes are silently skipped.
    :raises UnsupportedGridFormatError: If the file is too short or the first
        length marker does not equal 16 in either byte order.
    """
    total = len(raw)
    if total < 4:
        raise UnsupportedGridFormatError(
            "Eclipse binary file is too short to contain any records."
        )

    # Detect endianness once from the first Fortran record length marker.
    # The header record is always exactly 16 bytes (name=8, type=8), so
    # whichever interpretation gives 16 is the correct byte order.
    first_len_be = _INT_STRUCT_BE.unpack_from(raw, 0)[0]
    first_len_le = _INT_STRUCT_LE.unpack_from(raw, 0)[0]
    if first_len_be == 16:
        endian = ">"
        int_struct = _INT_STRUCT_BE
    elif first_len_le == 16:
        endian = "<"
        int_struct = _INT_STRUCT_LE
    else:
        raise UnsupportedGridFormatError(
            f"Unrecognised Eclipse binary record header: first 4 bytes = {raw[:4]!r}."
        )

    offset = 0
    while offset < total:
        # Header record:  [4] [name:8  type:8] [4]
        if offset + 4 > total:
            break

        hdr_len = int_struct.unpack_from(raw, offset)[0]
        offset += 4
        if hdr_len != 16 or offset + hdr_len + 4 > total:
            break

        keyword_bytes = raw[offset : offset + 8].rstrip()
        type_bytes = raw[offset + 8 : offset + 16]
        offset += hdr_len
        offset += 4  # trailing Fortran length marker
        keyword = keyword_bytes.decode("ascii", errors="replace").strip()

        # Look up the NumPy dtype for this Eclipse type code.
        # MESS has no data record following it, so skip immediately.
        dtype = _ECLIPSE_DTYPE_MAP.get(type_bytes)
        if dtype is None or keyword == "MESS":
            continue

        # Data record:  [4] [data_len bytes] [4]
        if offset + 4 > total:
            break

        data_len = int_struct.unpack_from(raw, offset)[0]
        offset += 4
        if offset + data_len + 4 > total:
            break

        data_bytes = raw[offset : offset + data_len]
        offset += data_len
        offset += 4  # trailing Fortran length marker

        # Interpret bytes and normalise to native endian in one pass.
        arr = np.frombuffer(data_bytes, dtype=dtype).copy()
        if endian != "=" and dtype.byteorder not in ("=", "|"):
            arr = arr.byteswap().newbyteorder("=")

        yield keyword, arr


def _parse_binary(raw: bytes) -> Grid:
    """
    Parse a complete Eclipse binary blob and construct a `bores.grids.base.Grid`.

    :param raw: Complete `.EGRID` / `.GRID` file bytes.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are absent or dimensions
        are inconsistent.
    """
    keywords: typing.Dict[str, np.ndarray] = {}
    for keyword, arr in _iter_records(raw):
        keywords[keyword] = arr

    # GRIDHEAD / DIMENS
    nx = ny = nz = None
    if "GRIDHEAD" in keywords:
        # EGRID: GRIDHEAD[1..3] = NX NY NZ (0-indexed: [0] is grid type)
        gh = keywords["GRIDHEAD"].astype(int)
        if len(gh) >= 4:
            nx, ny, nz = int(gh[1]), int(gh[2]), int(gh[3])
    if nx is None and "DIMENS" in keywords:
        d = keywords["DIMENS"].astype(int)
        if len(d) >= 3:
            nx, ny, nz = int(d[0]), int(d[1]), int(d[2])
    if nx is None:
        raise GridImportError(
            "Eclipse binary grid is missing GRIDHEAD or DIMENS keyword; "
            "cannot determine grid dimensions."
        )

    assert nx is not None
    assert ny is not None
    assert nz is not None
    # COORD
    if "COORD" not in keywords:
        raise GridImportError("Eclipse binary grid is missing required COORD keyword.")

    coord_flat = keywords["COORD"].astype(np.float64, order="F")
    expected_coord = (nx + 1) * (ny + 1) * 6
    if len(coord_flat) != expected_coord:
        raise GridImportError(
            f"COORD expected {expected_coord} values; got {len(coord_flat)}."
        )
    coord_arr = coord_flat.reshape(
        nx + 1,
        ny + 1,
        6,
        order="F",
    ).transpose(1, 0, 2)  # → (ny+1, nx+1, 6) C-order

    # ZCORN
    if "ZCORN" not in keywords:
        raise GridImportError("Eclipse binary grid is missing required ZCORN keyword.")

    zcorn_flat = keywords["ZCORN"].astype(np.float64, order="F")
    expected_zcorn = nx * ny * nz * 8
    if len(zcorn_flat) != expected_zcorn:
        raise GridImportError(
            f"ZCORN expected {expected_zcorn} values; got {len(zcorn_flat)}."
        )
    zcorn_arr = zcorn_flat.reshape(
        nx * 2,
        ny * 2,
        nz * 2,
        order="F",
    ).transpose(2, 1, 0)

    # ACTNUM (optional)
    actnum_arr: typing.Optional[IntArray[ThreeDimensions]] = None
    if "ACTNUM" in keywords:
        actnum_flat = keywords["ACTNUM"].astype(np.int32, order="F")
        expected_actnum = nx * ny * nz
        if len(actnum_flat) != expected_actnum:
            raise GridImportError(
                f"ACTNUM expected {expected_actnum} values; got {len(actnum_flat)}."
            )
        actnum_arr = actnum_flat.reshape(nx, ny, nz, order="F").transpose(2, 1, 0)

    try:
        return make_corner_point_grid(
            coord=coord_arr,
            zcorn=zcorn_arr,
            actnum=actnum_arr,
            metadata={
                "source_format": "eclipse_binary",
                "zcorn": zcorn_arr,
                "coord": coord_arr,
                "actnum": actnum_arr,
            },
        )
    except Exception as exc:
        raise GridImportError(
            f"Failed to construct grid from Eclipse binary arrays: {exc}"
        ) from exc
