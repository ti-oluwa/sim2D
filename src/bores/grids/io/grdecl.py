"""
GRDECL text-format reader and writer.

GRDECL is the ASCII keyword-based format used by Eclipse, ResInsight, and
most other reservoir simulators to describe corner-point pillar grids.

This module delegates all keyword parsing to
`bores.deck.DataFile`; it owns only the grid-assembly logic that
turns parsed keyword values into a `bores.grids.base.Grid`.

**Supported keywords (read)**:

`SPECGRID`, `COORD`, `ZCORN`, `ACTNUM`, `GRIDUNIT`,
`TOPS`, `DX`, `DY`, `DZ`, `MAPAXES`, `MAPUNITS`,
`PINCH`, `PINCHOUT`, `NNC`, `FAULTS`, `MULTFLT`,
`MULTX`, `MULTY`, `MULTZ`, `MULTX-`, `MULTY-`, `MULTZ-`.

**Supported keywords (write)**:

`SPECGRID`, `COORD`, `ZCORN`, `ACTNUM` (corner-point grids);
`SPECGRID`, `TOPS`, `DX`, `DY`, `DZ`, `ACTNUM` (Cartesian grids).
"""

import re
import typing
import warnings
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.deck.core import DeckParseError
from bores.deck.datafile import DataFile
from bores.errors import GridExportError, GridImportError
from bores.grids.base import Grid
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import (
    ActNumArray,
    FaultRecord,
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

_BARE_UNIT_KEYWORDS: typing.Dict[str, UnitSystem] = {
    "FIELD": UnitSystem.FIELD,
    "METRIC": UnitSystem.METRIC,
    "LAB": UnitSystem.LAB,
    "SI": UnitSystem.SI,
}


def _detect_unit_system(data_file: DataFile) -> UnitSystem:
    """
    Determine the grid geometry unit system from a parsed `DataFile`.

    Resolution order (highest to lowest priority):

    1. `GRIDUNIT` keyword - explicit geometry unit declaration.
    2. Bare section keywords `FIELD`, `METRIC`, `LAB`, `SI`
       (detected in the raw deck text).
    3. Default: `FIELD` (Eclipse default when no unit keyword is present).

    :param data_file: Already-constructed `DataFile`.
    :returns: The declared `bores.typing.UnitSystem`.
    """
    gridunit = data_file.get("GRIDUNIT")
    if gridunit is not None:
        unit_str = str(gridunit.get("unit", "")).strip().upper()
        us = _UNITS_MAP.get(unit_str)
        if us is not None:
            return us

    deck_text = data_file.deck.text
    for keyword, us in _BARE_UNIT_KEYWORDS.items():
        if re.search(r"(?<!\w)" + keyword + r"(?!\w)", deck_text, re.IGNORECASE):
            return us

    return UnitSystem.FIELD


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
        x_dir = x_dir / np.linalg.norm(x_dir)
        y_dir = self.map_y_axis_point - origin
        y_dir = y_dir / np.linalg.norm(y_dir)
        return np.array(
            [[x_dir[0], y_dir[0]], [x_dir[1], y_dir[1]]],
            dtype=np.float64,
        )

    def convert(self, to: UnitSystem) -> Self:
        """
        Return a new `MapAxes` with all coordinates expressed in `to`.

        :param to: Target `bores.typing.UnitSystem`.
        :returns: New `MapAxes` in the target unit system, or `self`
            if already in the target system.
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


def _build_map_axes(data_file: DataFile) -> typing.Optional[MapAxes]:
    """
    Construct a `MapAxes` from parsed `MAPAXES` / `MAPUNITS`
    keyword dicts.

    Eclipse `MAPAXES` field order:
    `(Y-axis X, Y-axis Y, origin X, origin Y, X-axis X, X-axis Y)`.

    :param data_file: Parsed deck.
    :returns: `MapAxes` or `None` if `MAPAXES` is absent.
    """
    mapaxes = data_file.get("MAPAXES")
    if mapaxes is None:
        return None

    mapunits = data_file.get("MAPUNITS")
    map_unit_str = str(mapunits.get("unit", "")).strip().upper() if mapunits else ""
    map_unit = _UNITS_MAP.get(map_unit_str, UnitSystem.FIELD)

    return MapAxes(
        # Eclipse ordering: Y-axis point first, then origin, then X-axis point
        origin=np.array([mapaxes["origin_x"], mapaxes["origin_y"]], dtype=np.float64),
        map_x_axis_point=np.array(
            [mapaxes["x_axis_x"], mapaxes["x_axis_y"]], dtype=np.float64
        ),
        map_y_axis_point=np.array(
            [mapaxes["y_axis_x"], mapaxes["y_axis_y"]], dtype=np.float64
        ),
        unit_system=map_unit,
    )


def _build_nnc_arrays(
    data_file: DataFile,
    nx: int,
    ny: int,
    nz: int,
) -> typing.Tuple[
    typing.Optional[IntArray[TwoDimensions]],
    typing.Optional[FloatArray[OneDimension]],
]:
    """
    Convert parsed `NNC` keyword records to flat cell-index arrays.

    :param data_file: Parsed deck (`NNC` keyword already registered).
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :returns: `(pairs, transmissibilities)` - shape `(n_nnc, 2)` int32
        and shape `(n_nnc,)` float64 arrays, or `(None, None)` if the
        keyword is absent.
    :raises GridImportError: If a record has out-of-bounds cell indices.
    """
    nnc_records = data_file.get("NNC")
    if not nnc_records:
        return None, None

    pairs: typing.List[typing.Tuple[int, int]] = []
    transmissibilities: typing.List[float] = []

    for idx, rec in enumerate(nnc_records):
        i1, j1, k1 = rec["i1"], rec["j1"], rec["k1"]
        i2, j2, k2 = rec["i2"], rec["j2"], rec["k2"]
        t = rec["transmissibility"]

        for label, i, j, k in [("first", i1, j1, k1), ("second", i2, j2, k2)]:
            if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz):
                raise GridImportError(
                    f"NNC record {idx} {label} cell ({i},{j},{k}) is outside "
                    f"grid bounds ({nx}x{ny}x{nz})."
                )

        # Eclipse flat index: i fastest (i-1) + (j-1)*nx + (k-1)*nx*ny
        c1 = (i1 - 1) + (j1 - 1) * nx + (k1 - 1) * nx * ny
        c2 = (i2 - 1) + (j2 - 1) * nx + (k2 - 1) * nx * ny
        pairs.append((c1, c2))
        transmissibilities.append(t)

    if not pairs:
        return None, None

    return (
        np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        np.asarray(transmissibilities, dtype=np.float64),
    )


def _build_fault_records(
    data_file: DataFile,
) -> typing.List[FaultRecord]:
    """
    Convert parsed `FAULTS` keyword records to `FaultRecord` objects.

    :param data_file: Parsed deck.
    :returns: List of `FaultRecord` in file order.
    """
    faults = data_file.get("FAULTS")
    if not faults:
        return []

    return [
        FaultRecord(
            name=rec["name"],
            i1=rec["i1"],
            i2=rec["i2"],
            j1=rec["j1"],
            j2=rec["j2"],
            k1=rec["k1"],
            k2=rec["k2"],
            face_direction=rec["face"],
        )
        for rec in faults
    ]


def _build_multflt(data_file: DataFile) -> typing.Optional[typing.Dict[str, float]]:
    """
    Convert parsed `MULTFLT` records to a `{name: multiplier}` dict.

    :param data_file: Parsed deck.
    :returns: Dict or `None` if `MULTFLT` is absent.
    """
    multflt_records = data_file.get("MULTFLT")
    if not multflt_records:
        return None
    return {rec["name"]: rec["multiplier"] for rec in multflt_records}


def load_grdecl(
    source: _TextOrPath,
    *,
    encoding: str = "ascii",
    unit_system: typing.Optional[UnitSystem] = None,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Load a GRDECL corner-point (or Cartesian) grid from a file path, raw
    string, or bytes.

    Automatically detects whether `source` is a filesystem path or raw
    GRDECL text / bytes.  Recursively resolves `INCLUDE` directives when
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
    - Multipliers:     `MULTX`, `MULTY`, `MULTZ` and `-` variants

    :param source: One of:

        - `pathlib.Path` - path to a `.grdecl` file.
        - `str` - filesystem path *or* raw GRDECL text content.
        - `bytes` - raw GRDECL content encoded as `encoding`.

    :param encoding: Text encoding used when decoding `bytes` / file
        input (default `"ascii"`).
    :param unit_system: If provided, the returned `bores.grids.base.Grid`
        is converted to this unit system after loading.
    :param metadata: Optional extra key/value pairs merged into the
        `Grid.metadata` dict.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or malformed.
    """
    try:
        data_file = DataFile(source, encoding=encoding)
    except DeckParseError as exc:
        raise GridImportError(f"Failed to parse GRDECL deck: {exc}") from exc

    grid = _assemble_grid(data_file, metadata=metadata)
    return convert(grid, to=unit_system) if unit_system is not None else grid


def dump_grdecl(
    grid: Grid,
    destination: typing.Union[_PathOrStr, None] = None,
    *,
    actnum: typing.Optional[ActNumArray] = None,
    encoding: str = "ascii",
) -> typing.Optional[str]:
    """
    Serialise a `bores.grids.base.Grid` to GRDECL text format.

    Note:
        GRDECL is a corner-point format. Only grids that originated from
        `load_grdecl` are supported for export.

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


def _assemble_grid(
    data_file: DataFile,
    metadata: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> Grid:
    """
    Assemble a `bores.grids.base.Grid` from a parsed
    `bores.deck.DataFile`.

    Dispatches to `_assemble_corner_point` or
    `_assemble_cartesian` based on which geometry keywords are
    present.

    :param data_file: Parsed deck with grid dimensions resolved.
    :param metadata: Optional extra metadata for the returned grid.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or the deck
        has no grid-dimension keyword.
    """
    if data_file.dimensions is None:
        raise GridImportError("GRDECL file is missing the required SPECGRID keyword.")
    
    dims = data_file.dimensions
    nx, ny, nz = dims.nx, dims.ny, dims.nz

    unit_system = _detect_unit_system(data_file)
    map_axes = _build_map_axes(data_file)

    pinch_rec = data_file.get("PINCH")
    pinch = pinch_rec["thickness"] if pinch_rec is not None else None
    # Also honour bare PINCHOUT keyword (no data).
    if pinch is None and data_file.has("PINCHOUT"):
        pinch = 1e-6

    meta: typing.Dict[str, typing.Any] = dict(metadata or {})
    meta.update(
        map_axes=map_axes,
        pinch=pinch,
        nx=nx,
        ny=ny,
        nz=nz,
    )

    has_coord = data_file.has("COORD")
    has_tops = data_file.has("TOPS")

    try:
        if has_coord:
            return _assemble_corner_point(data_file, nx, ny, nz, unit_system, meta)
        if has_tops:
            return _assemble_cartesian(data_file, nx, ny, nz, unit_system, meta)
    except GridImportError:
        raise
    except Exception as exc:
        raise GridImportError(
            f"Failed to construct grid from parsed GRDECL arrays: {exc}"
        ) from exc

    raise GridImportError(
        "GRDECL file contains neither COORD (corner-point) nor TOPS "
        "(Cartesian) geometry keywords."
    )


def _assemble_corner_point(
    data_file: DataFile,
    nx: int,
    ny: int,
    nz: int,
    unit_system: UnitSystem,
    meta: typing.Dict[str, typing.Any],
) -> Grid:
    """
    Build a corner-point `bores.grids.base.Grid` from a parsed deck.

    :param data_file: Parsed deck.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :param unit_system: Detected unit system.
    :param meta: Metadata dict (augmented in-place and passed to factory).
    :returns: Fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If `COORD` or `ZCORN` are missing or
        malformed.
    """
    coord = data_file.get("COORD")
    if coord is None:
        raise GridImportError("GRDECL file is missing the required COORD keyword.")

    zcorn = data_file.get("ZCORN")
    if zcorn is None:
        raise GridImportError("GRDECL file is missing the required ZCORN keyword.")

    # ACTNUM: DataFile returns flat (n_cells,) in Eclipse order (i-fastest).
    # `make_corner_point_grid` expects shape (nz, ny, nx) C-order.
    actnum_flat = data_file.get("ACTNUM")
    if actnum_flat is not None:
        actnum = actnum_flat.astype(np.int32, copy=False).reshape(nz, ny, nx)
    else:
        actnum = None

    meta["source_format"] = "grdecl_corner_point"
    meta["actnum"] = actnum

    nnc_pairs, nnc_transmissibilities = _build_nnc_arrays(data_file, nx, ny, nz)
    fault_records = _build_fault_records(data_file)
    multflt = _build_multflt(data_file)

    return make_corner_point_grid(
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        unit_system=unit_system,
        metadata=meta,
        nnc_cell_pairs=nnc_pairs,
        nnc_transmissibilities=nnc_transmissibilities,
        fault_records=fault_records,
        fault_transmissibility_multipliers=multflt,
        positive_x_transmissibility_multipliers=data_file.get("MULTX"),
        negative_x_transmissibility_multipliers=data_file.get("MULTX-"),
        positive_y_transmissibility_multipliers=data_file.get("MULTY"),
        negative_y_transmissibility_multipliers=data_file.get("MULTY-"),
        positive_z_transmissibility_multipliers=data_file.get("MULTZ"),
        negative_z_transmissibility_multipliers=data_file.get("MULTZ-"),
    )


def _assemble_cartesian(
    data_file: DataFile,
    nx: int,
    ny: int,
    nz: int,
    unit_system: UnitSystem,
    meta: typing.Dict[str, typing.Any],
) -> Grid:
    """
    Build a Cartesian `bores.grids.base.Grid` from `TOPS` /
    `DX` / `DY` / `DZ` keywords.

    Spacing: prefers `DXV` / `DYV` / `DZV` vector forms if present
    (not in the default keyword registry - add them if needed); otherwise
    reads per-cell `DX` / `DY` / `DZ` and extracts a 1-D slice.

    :param data_file: Parsed deck.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :param unit_system: Detected unit system.
    :param meta: Metadata dict (augmented in-place and passed to factory).
    :returns: Fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or malformed.
    """

    def _require(name: str) -> npt.NDArray[np.float64]:
        arr = data_file.get(name)
        if arr is None:
            raise GridImportError(
                f"Cartesian GRDECL grid is missing required keyword {name!r}."
            )
        return arr

    tops_flat = _require("TOPS")
    # TOPS may have nx*ny or nx*ny*nz values.  Use the first nx*ny as the
    # top-layer depths; warn if they vary significantly.
    n_columns = nx * ny
    tops_col = tops_flat[:n_columns]
    z_top = float(tops_col.min())
    if tops_col.max() - tops_col.min() > 1.0:
        warnings.warn(
            "GRDECL TOPS values vary by more than 1 unit; the Cartesian factory "
            "uses a flat top surface at the minimum TOPS value.  Geometry may be "
            "approximate for dipping grids.",
            stacklevel=6,
        )

    dx_flat = _require("DX")
    dy_flat = _require("DY")
    dz_flat = _require("DZ")

    # Extract representative 1-D spacing vectors.
    # Eclipse stores per-cell arrays in Fortran order (i fastest).
    # Reshape to (nz, ny, nx) C-order and pick first row/column/layer.
    dx_1d = dx_flat.reshape(nz, ny, nx)[0, 0, :]  # varies in x
    dy_1d = dy_flat.reshape(nz, ny, nx)[0, :, 0]  # varies in y
    dz_1d = dz_flat.reshape(nz, ny, nx)[:, 0, 0]  # varies in z

    actnum_flat = data_file.get("ACTNUM")
    meta["source_format"] = "grdecl_cartesian"
    if actnum_flat is not None:
        meta["actnum"] = actnum_flat.astype(np.int32).reshape(nz, ny, nx)

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


_GRDECL_SOURCES: typing.FrozenSet[str] = frozenset(
    {"grdecl_corner_point", "grdecl_cartesian"}
)


def _build_grdecl_text(
    grid: Grid,
    *,
    actnum: typing.Optional[ActNumArray] = None,
) -> str:
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
    actnum: typing.Optional[ActNumArray] = None,
) -> str:
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
    for axis_name, col_idx in [("DX", 0), ("DY", 1), ("DZ", 2)]:
        extents = grid.cell_max_xyz[:, col_idx] - grid.cell_min_xyz[:, col_idx]
        # Reshape to (nz, ny, nx) then transpose to Fortran order (nx, ny, nz)
        flat = extents.reshape(nz, ny, nx).transpose(2, 1, 0).ravel(order="F")
        lines.append(axis_name)
        for i in range(0, len(flat), 6):
            chunk = flat[i : i + 6]
            lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
        lines.append("/")
        lines.append("")

    if actnum is not None:
        _emit_actnum(lines, actnum, grid.n_cells, nx, ny, nz)

    _emit_mult_arrays(lines, grid, nx, ny, nz)
    return "\n".join(lines)


def _build_grdecl_corner_point_text(
    grid: Grid,
    *,
    actnum: typing.Optional[ActNumArray] = None,
) -> str:
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

    # COORD - Eclipse Fortran order: i (pillar) fastest
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

    effective_actnum = actnum if actnum is not None else meta.get("actnum")
    if effective_actnum is not None:
        _emit_actnum(lines, effective_actnum, grid.n_cells, nx, ny, nz)

    _emit_mult_arrays(lines, grid, nx, ny, nz)
    return "\n".join(lines)


def _emit_actnum(
    lines: typing.List[str],
    actnum: ActNumArray,
    n_cells: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Append an `ACTNUM` block in Eclipse Fortran order to `lines`."""
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
    """Append a `MAPAXES` block to `lines`."""
    # Eclipse order: Y-axis point, origin, X-axis point
    my = map_axes.map_y_axis_point
    o = map_axes.origin
    mx = map_axes.map_x_axis_point
    lines.append("MAPAXES")
    lines.append(
        f"  {my[0]:.6f}  {my[1]:.6f}"
        f"  {o[0]:.6f}  {o[1]:.6f}"
        f"  {mx[0]:.6f}  {mx[1]:.6f}  /"
    )
    lines.append("")


def _emit_gridunit(lines: typing.List[str], unit_system: UnitSystem) -> None:
    """Append a `GRIDUNIT` block to `lines`."""
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
    arr: FloatArray[OneDimension],
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Append a `MULT*` array block in Eclipse Fortran order."""
    lines.append("")
    lines.append(keyword)
    flat = arr.reshape(nz, ny, nx).transpose(2, 1, 0).ravel(order="F")
    for i in range(0, len(flat), 6):
        chunk = flat[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")


def _emit_mult_arrays(
    lines: typing.List[str],
    grid: Grid,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Emit all non-`None` MULT* arrays from `grid`."""
    pairs = [
        ("MULTX", grid.positive_x_transmissibility_multipliers),
        ("MULTX-", grid.negative_x_transmissibility_multipliers),
        ("MULTY", grid.positive_y_transmissibility_multipliers),
        ("MULTY-", grid.negative_y_transmissibility_multipliers),
        ("MULTZ", grid.positive_z_transmissibility_multipliers),
        ("MULTZ-", grid.negative_z_transmissibility_multipliers),
    ]
    for kw, arr in pairs:
        if arr is not None:
            _emit_mult_array(lines, kw, arr, nx, ny, nz)
