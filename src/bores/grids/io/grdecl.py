"""
GRDECL text-format reader and writer.

GRDECL is the ASCII keyword-based format used by Eclipse, ResInsight, and
most other reservoir simulators to describe corner-point pillar grids.

**Supported keywords (read)**:

`SPECGRID`, `DIMENS`, `COORD`, `ZCORN`, `ACTNUM`, `GRIDUNIT`,
`TOPS`, `DX`, `DY`, `DZ`, `DXV`, `DYV`, `DZV`,
`MAPAXES`, `MAPUNITS`, `MAPUNIT`,
`PINCH`, `PINCHOUT`, `NNC`, `FAULTS`, `MULTFLT`,
`MULTX`, `MULTY`, `MULTZ`, `MULTX-`, `MULTY-`, `MULTZ-`.

**Supported keywords (write)**:

`SPECGRID`, `GRIDUNIT`, `MAPAXES`, `MAPUNITS`,
`COORD`, `ZCORN`, `ACTNUM` (corner-point grids);
`SPECGRID`, `GRIDUNIT`, `MAPAXES`, `MAPUNITS`,
`TOPS`, `DXV`, `DYV`, `DZV`, `ACTNUM` (Cartesian grids).

Both grid types also emit (when present on the grid):
`MULTX`, `MULTY`, `MULTZ`, `MULTX-`, `MULTY-`, `MULTZ-`,
`FAULTS`, `MULTFLT`, `NNC`, `PINCH`.
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

_US_TO_GRIDUNIT: typing.Dict[UnitSystem, str] = {
    UnitSystem.FIELD: "FEET",
    UnitSystem.METRIC: "METRES",
    UnitSystem.LAB: "CM",
    UnitSystem.SI: "METRES",
}

_US_TO_MAPUNITS: typing.Dict[UnitSystem, str] = {
    UnitSystem.FIELD: "FEET",
    UnitSystem.METRIC: "METRES",
    UnitSystem.LAB: "CM",
    UnitSystem.SI: "METRES",
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
        unit_system = _UNITS_MAP.get(unit_str)
        if unit_system is not None:
            return unit_system

    deck_text = data_file.deck.text
    for keyword, unit_system in _BARE_UNIT_KEYWORDS.items():
        if re.search(r"(?<!\w)" + keyword + r"(?!\w)", deck_text, re.IGNORECASE):
            return unit_system

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
        x_vec = self.map_x_axis_point - origin
        y_vec = self.map_y_axis_point - origin
        x_norm = np.linalg.norm(x_vec)
        y_norm = np.linalg.norm(y_vec)
        if x_norm < 1e-14 or y_norm < 1e-14:
            warnings.warn(
                "MAPAXES has a degenerate (zero-length) axis vector. "
                "The map coordinate rotation will be skipped.",
                stacklevel=3,
            )
            return np.eye(2, dtype=np.float64)
        x_dir = x_vec / x_norm
        y_dir = y_vec / y_norm
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
    Construct a `MapAxes` from parsed `MAPAXES` / `MAPUNITS` keyword dicts.

    Eclipse `MAPAXES` field order:
    `(Y-axis X, Y-axis Y, origin X, origin Y, X-axis X, X-axis Y)`.

    :param data_file: Parsed deck.
    :returns: `MapAxes` or `None` if `MAPAXES` is absent.
    """
    mapaxes = data_file.get("MAPAXES")
    if mapaxes is None:
        return None

    # Honour both MAPUNITS and MAPUNIT (the latter is an Eclipse alias).
    mapunits = data_file.get("MAPUNITS") or data_file.get("MAPUNIT")
    map_unit_str = str(mapunits.get("unit", "")).strip().upper() if mapunits else ""
    map_unit = _UNITS_MAP.get(map_unit_str, UnitSystem.FIELD)

    return MapAxes(
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

        # Eclipse flat index: i fastest → (i-1) + (j-1)*nx + (k-1)*nx*ny
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


def _build_fault_records(data_file: DataFile) -> typing.List[FaultRecord]:
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


def _build_multflt(
    data_file: DataFile,
) -> typing.Optional[typing.Dict[str, float]]:
    """
    Convert parsed `MULTFLT` records to a `{name: multiplier}` dict.

    :param data_file: Parsed deck.
    :returns: Dict or `None` if `MULTFLT` is absent.
    """
    multflt_records = data_file.get("MULTFLT")
    if not multflt_records:
        return None
    return {rec["name"]: rec["multiplier"] for rec in multflt_records}


def _resolve_vector_spacing(
    data_file: DataFile,
    vector_key: str,
    per_cell_key: str,
    count: int,
    axis: str,
    nz: int,
    ny: int,
    nx: int,
) -> typing.Optional[npt.NDArray[np.float64]]:
    """
    Try to obtain a 1-D spacing vector for one axis.

    Resolution order:

    1. `DXV` / `DYV` / `DZV` - single-valued vectors of length
       `nx`, `ny`, `nz` respectively, already 1-D in the deck.
    2. `DX` / `DY` / `DZ` - per-cell arrays of length
       `nx*ny*nz`.  Reshaped to `(nz, ny, nx)` and the first
       row / column / layer is extracted.
    3. Returns `None` if neither keyword is present.

    :param data_file: Parsed deck.
    :param vector_key: Vector keyword name (`"DXV"`, `"DYV"`, `"DZV"`).
    :param per_cell_key: Per-cell keyword name (`"DX"`, `"DY"`, `"DZ"`).
    :param count: Expected length of the vector (`nx`, `ny`, or `nz`).
    :param axis: Axis label for error messages (`"x"`, `"y"`, `"z"`).
    :param nz: Grid depth count.
    :param ny: Grid lateral count (y).
    :param nx: Grid lateral count (x).
    :returns: 1-D float64 spacing array or `None`.
    :raises GridImportError: If the vector keyword has the wrong length.
    """
    # Prefer the vector form (DXV / DYV / DZV).
    vec = data_file.get(vector_key)
    if vec is not None:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        if len(arr) != count:
            raise GridImportError(
                f"{vector_key} has {len(arr)} values but expected {count} "
                f"(grid has n{axis}={count})."
            )
        return arr

    # Fall back to the per-cell array.
    per_cell = data_file.get(per_cell_key)
    if per_cell is None:
        return None

    flat = np.asarray(per_cell, dtype=np.float64).ravel()
    reshaped = flat.reshape(nz, ny, nx)
    if axis == "x":
        return reshaped[0, 0, :]  # varies in x; take first y-row, first z-layer
    if axis == "y":
        return reshaped[0, :, 0]  # varies in y
    return reshaped[:, 0, 0]  # varies in z


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
    GRDECL text / bytes. Recursively resolves `INCLUDE` directives when
    loading from a file path.

    Parsed keyword coverage:

    * Grid dimensions:   `SPECGRID`, `DIMENS`
    * Geometry:          `COORD`, `ZCORN`, `TOPS`, `DX`, `DY`,
                         `DZ`, `DXV`, `DYV`, `DZV`
    * Activity:          `ACTNUM`
    * Units:             `GRIDUNIT`, `FIELD`, `METRIC`, `LAB`, `SI`
    * Map CRS:           `MAPAXES`, `MAPUNITS`, `MAPUNIT`
    * Pinchouts:         `PINCH`, `PINCHOUT`
    * Connections:       `NNC` (with transmissibilities)
    * Faults:            `FAULTS`, `MULTFLT`
    * Multipliers:       `MULTX`, `MULTY`, `MULTZ` and `-` variants

    :param source: One of:

        * `pathlib.Path` - path to a `.grdecl` file.
        * `str` - filesystem path *or* raw GRDECL text content.
        * `bytes` - raw GRDECL content encoded as `encoding`.

    :param encoding: Text encoding used when decoding `bytes` / file
        input (default `"ascii"`).
    :param unit_system: If provided, the returned
        `bores.grids.base.Grid` is converted to this unit system
        after loading.
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

    Emits all keywords that can be reconstructed from the grid object.
    The full set that may appear in the output is:

    `SPECGRID`, `GRIDUNIT`, `MAPAXES`, `MAPUNITS`,
    `COORD` / `ZCORN` (corner-point) or `TOPS` / `DXV` / `DYV` /
    `DZV` (Cartesian), `ACTNUM`, `MULTX`, `MULTY`, `MULTZ`,
    `MULTX-`, `MULTY-`, `MULTZ-`, `FAULTS`, `MULTFLT`,
    `NNC`, `PINCH`.

    :param grid: The grid to serialise.
    :param destination: One of:

        * `pathlib.Path` or `str` path - write to file, return `None`.
        * `None` - return the GRDECL text as a `str`.

    :param actnum: Optional shape `(n_cells,)` integer array of
        active-cell flags (1 = active, 0 = inactive).  If `None`, the
        value stored in `grid.metadata["actnum"]` is used; if that is
        also absent, all cells are written as active.
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
    Assemble a `bores.grids.base.Grid` from a parsed `bores.deck.DataFile`.

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
    has_tops = data_file.has("TOPS") or data_file.has("DX") or data_file.has("DXV")

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
        "GRDECL file contains neither COORD (corner-point) nor TOPS / DX / DXV "
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
    `DX` / `DY` / `DZ` (per-cell) or `DXV` / `DYV` / `DZV`
    (vector) keywords.

    Vector forms (`DXV` / `DYV` / `DZV`) are preferred when present
    and specify spacing directly as 1-D arrays of length `nx`, `ny`,
    and `nz`.  Per-cell forms are reshaped and sliced to 1-D.

    Also reads `FAULTS`, `MULTFLT`, `NNC`, and all six `MULT*`
    arrays, passing them on to :func:`make_cartesian_grid`.

    :param data_file: Parsed deck.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :param unit_system: Detected unit system.
    :param meta: Metadata dict passed to the factory.
    :returns: Fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or malformed.
    """
    # Z origin from TOPS
    tops_flat = data_file.get("TOPS")
    if tops_flat is not None:
        tops_flat = np.asarray(tops_flat, dtype=np.float64)
        n_columns = nx * ny
        tops_col = tops_flat[:n_columns]
        z_top = float(tops_col.min())
        if tops_col.max() - tops_col.min() > 1.0:
            warnings.warn(
                "GRDECL TOPS values vary by more than 1 unit; the Cartesian factory "
                "uses a flat top surface at the minimum TOPS value. Geometry may be "
                "approximate for dipping grids.",
                stacklevel=6,
            )
    else:
        z_top = 0.0

    # Spacing vectors
    dx_1d = _resolve_vector_spacing(data_file, "DXV", "DX", nx, "x", nz, ny, nx)
    dy_1d = _resolve_vector_spacing(data_file, "DYV", "DY", ny, "y", nz, ny, nx)
    dz_1d = _resolve_vector_spacing(data_file, "DZV", "DZ", nz, "z", nz, ny, nx)

    if dx_1d is None:
        raise GridImportError(
            "Cartesian GRDECL grid is missing required spacing keyword DX or DXV."
        )
    if dy_1d is None:
        raise GridImportError(
            "Cartesian GRDECL grid is missing required spacing keyword DY or DYV."
        )
    if dz_1d is None:
        raise GridImportError(
            "Cartesian GRDECL grid is missing required spacing keyword DZ or DZV."
        )

    actnum_flat = data_file.get("ACTNUM")
    meta["source_format"] = "grdecl_cartesian"
    if actnum_flat is not None:
        meta["actnum"] = actnum_flat.astype(np.int32).reshape(nz, ny, nx)

    nnc_pairs, nnc_transmissibilities = _build_nnc_arrays(data_file, nx, ny, nz)
    fault_records = _build_fault_records(data_file)
    multflt = _build_multflt(data_file)

    # Store pinch in metadata so dump_grdecl can re-emit it.
    if meta.get("pinch") is not None:
        meta["pinch"] = meta["pinch"]

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
        fault_records=fault_records,
        fault_transmissibility_multipliers=multflt,
        nnc_cell_pairs=nnc_pairs,
        nnc_transmissibilities=nnc_transmissibilities,
        positive_x_transmissibility_multipliers=data_file.get("MULTX"),
        negative_x_transmissibility_multipliers=data_file.get("MULTX-"),
        positive_y_transmissibility_multipliers=data_file.get("MULTY"),
        negative_y_transmissibility_multipliers=data_file.get("MULTY-"),
        positive_z_transmissibility_multipliers=data_file.get("MULTZ"),
        negative_z_transmissibility_multipliers=data_file.get("MULTZ-"),
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


def _emit_specgrid(lines: typing.List[str], nx: int, ny: int, nz: int) -> None:
    """Append a `SPECGRID` block."""
    lines.append("SPECGRID")
    lines.append(f"  {nx}  {ny}  {nz}  1  F /")
    lines.append("")


def _emit_gridunit(lines: typing.List[str], unit_system: UnitSystem) -> None:
    """Append a `GRIDUNIT` block."""
    unit_str = _US_TO_GRIDUNIT.get(unit_system, "FEET")
    lines.append("GRIDUNIT")
    lines.append(f"  '{unit_str}  ' '        ' /")
    lines.append("")


def _emit_mapaxes(lines: typing.List[str], map_axes: MapAxes) -> None:
    """Append `MAPUNITS` and `MAPAXES` blocks."""
    # Eclipse order for MAPAXES: Y-axis point, origin, X-axis point
    my = map_axes.map_y_axis_point
    o = map_axes.origin
    mx = map_axes.map_x_axis_point

    unit_str = _US_TO_MAPUNITS.get(map_axes.unit_system, "FEET")
    lines.append("MAPUNITS")
    lines.append(f"  '{unit_str}' /")
    lines.append("")

    lines.append("MAPAXES")
    lines.append(
        f"  {my[0]:.6f}  {my[1]:.6f}"
        f"  {o[0]:.6f}  {o[1]:.6f}"
        f"  {mx[0]:.6f}  {mx[1]:.6f}  /"
    )
    lines.append("")


def _emit_actnum(
    lines: typing.List[str],
    actnum: ActNumArray,
    n_cells: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Append an `ACTNUM` block in Eclipse Fortran order (i fastest)."""
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


def _emit_mult_array(
    lines: typing.List[str],
    keyword: str,
    arr: FloatArray[OneDimension],
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Append a `MULT*` array block in Eclipse Fortran order (i fastest)."""
    lines.append("")
    lines.append(keyword)
    flat = (
        np.asarray(arr, dtype=np.float64)
        .reshape(nz, ny, nx)
        .transpose(2, 1, 0)
        .ravel(order="F")
    )
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


def _emit_faults(
    lines: typing.List[str],
    grid: Grid,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Emit a `FAULTS` block by converting the unstructured face indices stored
    in `grid.fault_face_indices` back to structured IJK records.

    Each face in a Cartesian or corner-point grid has exactly two adjacent
    cells.  The cell flat indices are converted to 1-based IJK coordinates
    using the cell ordering `cell_idx = i + j*nx + k*nx*ny`.  The face
    direction is inferred from which coordinate differs between the two cells.

    Faces that cannot be cleanly resolved (e.g. boundary faces with
    `neighbour == -1`) are skipped with a warning.

    :param lines: Output text lines list (mutated in-place).
    :param grid: Source grid with `fault_face_indices` populated.
    :param nx: Grid dimension in x.
    :param ny: Grid dimension in y.
    :param nz: Grid dimension in z.
    """
    if not grid.fault_face_indices:
        return

    def _flat_to_ijk(flat: int) -> typing.Tuple[int, int, int]:
        """Convert 0-based flat cell index to 1-based (i, j, k)."""
        i = flat % nx
        j = (flat // nx) % ny
        k = flat // (nx * ny)
        return i + 1, j + 1, k + 1

    lines.append("")
    lines.append("FAULTS")

    for fault_name, face_indices in sorted(grid.fault_face_indices.items()):
        for face_idx in face_indices:
            owner = int(grid.face_cell_indices[face_idx, 0])
            neighbour = int(grid.face_cell_indices[face_idx, 1])
            if owner < 0 or neighbour < 0:
                continue  # boundary face - skip

            oi, oj, ok = _flat_to_ijk(owner)
            ni, nj, nk = _flat_to_ijk(neighbour)

            # Determine face direction from which IJK coordinate differs.
            if ni != oi:
                face_dir = "I" if ni > oi else "I-"
                # Normalise: always record from the lower-index cell.
                if ni < oi:
                    oi, oj, ok = ni, nj, nk
            elif nj != oj:
                face_dir = "J" if nj > oj else "J-"
                if nj < oj:
                    oi, oj, ok = ni, nj, nk
            elif nk != ok:
                face_dir = "K" if nk > ok else "K-"
                if nk < ok:
                    oi, oj, ok = ni, nj, nk
            else:
                warnings.warn(
                    f"Fault {fault_name!r}: face {face_idx} connects cells with "
                    f"identical IJK ({oi},{oj},{ok}) - cannot determine direction. "
                    f"Skipping.",
                    stacklevel=4,
                )
                continue

            lines.append(
                f"  '{fault_name}'  {oi}  {oi}  {oj}  {oj}  {ok}  {ok}  '{face_dir}'  /"
            )

    lines.append("/")
    lines.append("")


def _emit_multflt(
    lines: typing.List[str],
    grid: Grid,
) -> None:
    """Append a `MULTFLT` block from `grid.fault_transmissibility_multipliers`."""
    if not grid.fault_transmissibility_multipliers:
        return
    lines.append("")
    lines.append("MULTFLT")
    for name, mult in sorted(grid.fault_transmissibility_multipliers.items()):
        lines.append(f"  '{name}'  {mult:.6f}  /")
    lines.append("/")
    lines.append("")


def _emit_nnc(
    lines: typing.List[str],
    grid: Grid,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Emit a `NNC` block from `grid.nnc_cell_pairs` /
    `grid.nnc_transmissibilities`.

    Flat cell indices are converted to 1-based `(I, J, K)` using the
    Eclipse ordering `cell_idx = (i-1) + (j-1)*nx + (k-1)*nx*ny`.
    NNCs without a stored transmissibility (stored as `NaN`) are emitted
    with a placeholder value of `0.0` and a comment.

    :param lines: Output text lines list.
    :param grid: Source grid.
    :param nx: Grid dimension in x.
    :param ny: Grid dimension in y.
    :param nz: Grid dimension in z.
    """
    if grid.nnc_cell_pairs is None or len(grid.nnc_cell_pairs) == 0:
        return

    def _flat_to_ijk(flat: int) -> typing.Tuple[int, int, int]:
        i = flat % nx + 1
        j = (flat // nx) % ny + 1
        k = flat // (nx * ny) + 1
        return i, j, k

    has_t = grid.nnc_transmissibilities is not None and len(
        grid.nnc_transmissibilities
    ) == len(grid.nnc_cell_pairs)

    lines.append("")
    lines.append("NNC")
    for idx, (c1, c2) in enumerate(grid.nnc_cell_pairs):
        i1, j1, k1 = _flat_to_ijk(int(c1))
        i2, j2, k2 = _flat_to_ijk(int(c2))
        if has_t:
            t = float(grid.nnc_transmissibilities[idx])  # type: ignore
            t_str = f"{t:.6e}" if not np.isnan(t) else "0.0 -- T unknown"
        else:
            t_str = "0.0 -- T unknown"
        lines.append(f"  {i1}  {j1}  {k1}  {i2}  {j2}  {k2}  {t_str}  /")
    lines.append("/")
    lines.append("")


def _emit_pinch(
    lines: typing.List[str],
    meta: typing.Mapping[str, typing.Any],
) -> None:
    """
    Emit a `PINCH` block when the pinch tolerance is stored in metadata.

    :param lines: Output text lines list.
    :param meta: Grid metadata mapping; reads `meta["pinch"]`.
    """
    pinch = meta.get("pinch")
    if pinch is None:
        return
    lines.append("")
    lines.append("PINCH")
    lines.append(f"  {float(pinch):.6g}  /")
    lines.append("")


def _build_grdecl_cartesian_text(
    grid: Grid,
    *,
    actnum: typing.Optional[ActNumArray] = None,
) -> str:
    """
    Build a GRDECL text representation for a Cartesian grid.

    Emits (in order): `SPECGRID`, `GRIDUNIT`, `MAPAXES` (if present),
    `TOPS`, `DXV`, `DYV`, `DZV`, `ACTNUM` (if present),
    `MULTX` / `MULTY` / `MULTZ` / `-` variants (if present),
    `FAULTS` (if present), `MULTFLT` (if present),
    `NNC` (if present), `PINCH` (if present).

    :param grid: Cartesian grid with `source_format = "grdecl_cartesian"`
        in metadata.
    :param actnum: Optional active-cell mask; falls back to
        `grid.metadata["actnum"]`.
    :returns: GRDECL text string.
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

    _emit_specgrid(lines, nx, ny, nz)
    _emit_gridunit(lines, grid.unit_system)

    map_axes: typing.Optional[MapAxes] = meta.get("map_axes")
    if map_axes is not None:
        _emit_mapaxes(lines, map_axes)

    # TOPS - use the top-layer cell minimum z values.
    # Order: i fastest (Eclipse Fortran / C with transposed axes).
    top_layer_indices = [i + j * nx for j in range(ny) for i in range(nx)]
    tops_vals = grid.cell_min_xyz[top_layer_indices, 2]
    lines.append("TOPS")
    for i in range(0, len(tops_vals), 6):
        chunk = tops_vals[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")
    lines.append("")

    # DXV / DYV / DZV - one representative value per cell count.
    # We extract from cell bounding boxes: first j=0, k=0 row for DXV; etc.
    dx_vals = grid.cell_max_xyz[:nx, 0] - grid.cell_min_xyz[:nx, 0]
    lines.append("DXV")
    lines.append("  " + "  ".join(f"{v:.6f}" for v in dx_vals))
    lines.append("/")
    lines.append("")

    # DYV: cells at i=0, k=0 → indices 0, nx, 2*nx, …
    dy_indices = np.arange(ny) * nx
    dy_vals = grid.cell_max_xyz[dy_indices, 1] - grid.cell_min_xyz[dy_indices, 1]
    lines.append("DYV")
    lines.append("  " + "  ".join(f"{v:.6f}" for v in dy_vals))
    lines.append("/")
    lines.append("")

    # DZV: cells at i=0, j=0, k=0..nz-1 → indices 0, nx*ny, 2*nx*ny, …
    dz_indices = np.arange(nz) * nx * ny
    dz_vals = grid.cell_max_xyz[dz_indices, 2] - grid.cell_min_xyz[dz_indices, 2]
    lines.append("DZV")
    lines.append("  " + "  ".join(f"{v:.6f}" for v in dz_vals))
    lines.append("/")
    lines.append("")

    effective_actnum = actnum if actnum is not None else meta.get("actnum")
    if effective_actnum is not None:
        _emit_actnum(lines, effective_actnum, grid.n_cells, nx, ny, nz)

    _emit_mult_arrays(lines, grid, nx, ny, nz)
    _emit_faults(lines, grid, nx, ny, nz)
    _emit_multflt(lines, grid)
    _emit_nnc(lines, grid, nx, ny, nz)
    _emit_pinch(lines, meta)

    return "\n".join(lines)


def _build_grdecl_corner_point_text(
    grid: Grid,
    *,
    actnum: typing.Optional[ActNumArray] = None,
) -> str:
    """
    Build a GRDECL text representation for a corner-point grid.

    Emits (in order): `SPECGRID`, `GRIDUNIT`, `MAPAXES` (if present),
    `COORD`, `ZCORN`, `ACTNUM` (if present),
    `MULTX` / `MULTY` / `MULTZ` / `-` variants (if present),
    `FAULTS` (if present), `MULTFLT` (if present),
    `NNC` (if present), `PINCH` (if present).

    :param grid: Corner-point grid with `source_format = "grdecl_corner_point"`
        in metadata.
    :param actnum: Optional active-cell mask.
    :returns: GRDECL text string.
    """
    coord, zcorn, nx, ny, nz = rederive_corner_point_arrays(grid)
    meta: typing.Mapping[str, typing.Any] = getattr(grid, "metadata", {}) or {}

    lines: typing.List[str] = []

    _emit_specgrid(lines, nx, ny, nz)
    _emit_gridunit(lines, grid.unit_system)

    map_axes: typing.Optional[MapAxes] = meta.get("map_axes")
    if map_axes is not None:
        _emit_mapaxes(lines, map_axes)

    # COORD (ny + 1, nx + 1, 6)
    # Already in Eclipse COORD order, so we flatten and write as is
    lines.append("COORD")
    flat_coord = coord.reshape(-1, 6)
    for row in flat_coord:
        x1, y1, z1, x2, y2, z2 = row
        lines.append(f"  {x1:.6f}  {y1:.6f}  {z1:.6f}  {x2:.6f}  {y2:.6f}  {z2:.6f}")
    lines.append("/")
    lines.append("")

    # ZCORN (nz*2, ny*2, nx*2)
    # We flatten the 3D array; x vary fastest, and z slowest just as Eclipse writes it
    lines.append("ZCORN")
    flat_zcorn = zcorn.ravel()
    for i in range(0, len(flat_zcorn), 6):
        chunk = flat_zcorn[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")
    lines.append("")

    effective_actnum = actnum if actnum is not None else meta.get("actnum")
    if effective_actnum is not None:
        _emit_actnum(lines, effective_actnum, grid.n_cells, nx, ny, nz)

    _emit_mult_arrays(lines, grid, nx, ny, nz)
    _emit_faults(lines, grid, nx, ny, nz)
    _emit_multflt(lines, grid)
    _emit_nnc(lines, grid, nx, ny, nz)
    _emit_pinch(lines, meta)

    return "\n".join(lines)
