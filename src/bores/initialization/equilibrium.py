"""Reservoir equilibrium (`EQUIL`) region data."""

import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serialization.stores import StoreSerializable
from bores.typing import (
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    UnitSystem,
)

__all__ = [
    "DepthTable",
    "EquilibriumRegion",
    "EquilibriumRegions",
    "load_equilibrium_infos",
]


@attrs.frozen(slots=True)
class DepthTable(StoreSerializable):
    """
    A generic depth-indexed lookup table for a single equilibration region,
    corresponding to one table from `RSVD` (solution GOR vs. depth) or
    `RVVD` (vaporised oil-gas ratio vs. depth).

    Deliberately generic (`values`, not `rs`/`rv`) rather than two near-
    identical `RsvdTable` / `RvvdTable` classes - RSVD and RVVD are the same
    shape (a monotone depth axis + one dependent column) and only differ in
    which physical quantity they carry, which the caller already knows from
    context (`EquilibriumRegions.rsvd_tables` vs. `.rvvd_tables`).

    Values are linearly interpolated at a query depth; values outside the
    table range are clamped to the endpoint (no extrapolation), matching
    Eclipse behaviour and `bores.reservoir.temperature.TemperatureTable`.
    """

    depths: NumberArray[OneDimension]
    """1-D strictly increasing depth array in `unit_system` length units."""

    values: NumberArray[OneDimension]
    """
    1-D dependent-variable array (Rs in SCF/STB or Rv in STB/Mscf,
    depending on which keyword this table was built from). Must have the
    same length as `depths`.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `depths`. `values` unit conversion is not handled
    here - GOR-type quantities are converted by the caller (same factor
    used for `State.solution_gor` / `vaporized_oil_ratio`)."""

    def __attrs_post_init__(self) -> None:
        if self.depths.ndim != 1:
            raise ValidationError("`depths` must be a 1-D array.")
        if self.values.ndim != 1:
            raise ValidationError("`values` must be a 1-D array.")
        if len(self.depths) != len(self.values):
            raise ValidationError(
                f"`depths` length {len(self.depths)} must match "
                f"`values` length {len(self.values)}."
            )
        if len(self.depths) < 2:
            raise ValidationError(
                "`DepthTable` requires at least 2 (depth, value) pairs."
            )
        if not np.all(np.diff(self.depths) > 0):
            raise ValidationError("`depths` must be strictly increasing.")

    def at_depth(
        self, depth: NumberOrArray[OneDimension]
    ) -> NumberOrArray[OneDimension]:
        """
        Return the interpolated value at the given depth or array of depths.

        :param depth: Scalar or shape `(n_cells,)` depth array in
            `unit_system` length units.
        :returns: Interpolated value(s), clamped to the table endpoints
            outside its range.
        """
        is_scalar = np.isscalar(depth)
        depth_arr = np.atleast_1d(depth)
        result = np.interp(depth_arr, self.depths, self.values)
        return (
            typing.cast(Number, result[0])
            if is_scalar
            else typing.cast(NumberArray[OneDimension], result)
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `DepthTable` with `depths` rescaled to *target*.
        `values` are left unchanged - see the `unit_system` docstring.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `DepthTable` in *target* units.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            depths=typing.cast(
                NumberArray[OneDimension], self.depths * factors["length"]
            ),
            values=self.values,
            unit_system=target,
        )


def _load_depth_tables(
    deck_file: DeckFile,
    keyword: str,
    value_column: str,
    dtype: typing.Any = None,
) -> typing.Optional[typing.Dict[int, DepthTable]]:
    """
    Parse an `RSVD`/`RVVD`-shaped keyword into `{table_number: DepthTable}`.

    Shared helper for both `RSVD` and `RVVD`, which have identical shape
    and differ only in which column holds the dependent variable.

    :param deck_file: Parsed `DeckFile`.
    :param keyword: `"RSVD"` or `"RVVD"`.
    :param value_column: `"rs"` or `"rv"` - the column name to extract.
    :returns: `{1-based table number: DepthTable}`, or `None` if the
        keyword is absent from the deck.
    :raises ValidationError: If the keyword is present but a table has
        non-increasing depths.
    """
    all_tables: typing.Optional[typing.List] = deck_file.get(keyword)
    if not all_tables:
        return None

    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    unit_system = deck_file.unit_system
    tables: typing.Dict[int, DepthTable] = {}
    for table_idx, rows in enumerate(all_tables):
        if not rows:
            continue
        table_number = table_idx + 1  # 1-based
        depths = np.array([row["depth"] for row in rows], dtype=dtype)
        values = np.array([row[value_column] for row in rows], dtype=dtype)
        try:
            tables[table_number] = DepthTable(
                depths=typing.cast(NumberArray[OneDimension], depths),
                values=typing.cast(NumberArray[OneDimension], values),
                unit_system=unit_system,
            )
        except ValidationError as exc:
            raise ValidationError(f"{keyword} table {table_number}: {exc}") from exc

    return tables or None


@attrs.frozen(slots=True)
class EquilibriumRegion(StoreSerializable):
    """
    Gravity/capillary equilibration data for a single `EQLNUM` region -
    one record of the Eclipse `EQUIL` keyword.

    Stores one `EQUIL` record, validate it, support
    unit conversion and (de)serialization, and support loading from an
    Eclipse deck.

    **Convention**: `woc_depth` / `goc_depth` of `0.0` (the Eclipse default
    when the corresponding `EQUIL` item is defaulted via `1*`) is
    treated as "no such contact in this region" rather than a literal
    zero-depth contact - use `has_woc` / `has_goc` rather than comparing
    the raw depth to zero directly.
    """

    datum_depth: Number
    """Depth of the datum point at which `datum_pressure` applies."""

    datum_pressure: Number
    """Reservoir pressure at `datum_depth`."""

    woc_depth: Number = 0.0
    """
    Water-oil contact depth. `0.0` means no WOC in this region (e.g. an
    all-oil column with no underlying aquifer, or fully water-saturated).
    """

    pcow_woc: Number = 0.0
    """Oil-water capillary pressure at the WOC (usually `0`)."""

    goc_depth: Number = 0.0
    """
    Gas-oil contact depth. `0.0` means no GOC in this region (no gas
    cap - oil is at or below its bubble point everywhere, or absent).
    """

    pcog_goc: Number = 0.0
    """Gas-oil capillary pressure at the GOC (usually `0`)."""

    rsvd_table: int = 0
    """
    `RSVD`-table number for depth-dependent solution GOR below the
    bubble point. `0` = use the PVT table's saturated (bubble-point) Rs
    instead, i.e. assume saturated oil throughout the region.
    """

    rvvd_table: int = 0
    """
    `RVVD`-table number for depth-dependent vaporised-oil ratio above
    the dew point. `0` = use the PVT table's saturated (dew-point) Rv
    instead, i.e. assume saturated gas throughout the region.
    """

    accuracy_flag: int = 0
    """
    `EQUIL` item 9 - initialization accuracy/option switch.

    - `0`  - center-point: evaluate pressure/saturation once at each
      cell's centroid depth (fast, standard black-oil default).
    - `>0` - `N` sub-divisions per cell for horizontal-subdivision
      (vertical-equilibrium averaging) initialization.
    - `<0` - `|N|` sub-divisions per cell using the tilted-cell variant,
      which additionally accounts for cell dip / non-horizontal faces.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for all dimensional fields."""

    def __attrs_post_init__(self) -> None:
        if self.datum_depth < 0:
            raise ValidationError(
                f"`datum_depth` must be non-negative; got {self.datum_depth}."
            )
        if self.woc_depth < 0 or self.goc_depth < 0:
            raise ValidationError(
                "`woc_depth` / `goc_depth` must be non-negative; got "
                f"woc_depth={self.woc_depth}, goc_depth={self.goc_depth}."
            )
        if self.has_woc and self.has_goc and self.goc_depth > self.woc_depth:
            raise ValidationError(
                f"`goc_depth` ({self.goc_depth}) must be shallower than or "
                f"equal to `woc_depth` ({self.woc_depth}) when both contacts "
                "are present."
            )
        if self.rsvd_table < 0 or self.rvvd_table < 0:
            raise ValidationError(
                "`rsvd_table` / `rvvd_table` must be non-negative table "
                f"numbers; got rsvd_table={self.rsvd_table}, "
                f"rvvd_table={self.rvvd_table}."
            )

    @property
    def has_woc(self) -> bool:
        """Whether this region has an explicit water-oil contact."""
        return bool(self.woc_depth > 0.0)

    @property
    def has_goc(self) -> bool:
        """Whether this region has an explicit gas-oil contact (gas cap)."""
        return bool(self.goc_depth > 0.0)

    @property
    def uses_rsvd(self) -> bool:
        """Whether Rs below the bubble point should come from an `RSVD` table."""
        return self.rsvd_table > 0

    @property
    def uses_rvvd(self) -> bool:
        """Whether Rv above the dew point should come from an `RVVD` table."""
        return self.rvvd_table > 0

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `EquilibriumRegion` with all dimensional fields rescaled
        to *target*.

        Depths use the length factor; pressures (`datum_pressure`,
        `pcow_woc`, `pcog_goc`) use the pressure factor. Table numbers and
        `accuracy_flag` are dimensionless and copied unchanged.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the
            default.
        :returns: New `EquilibriumRegion` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        length_factor = factors["length"]
        pressure_factor = factors["pressure"]
        return self.__class__(
            datum_depth=self.datum_depth * length_factor,
            datum_pressure=self.datum_pressure * pressure_factor,
            woc_depth=self.woc_depth * length_factor,
            pcow_woc=self.pcow_woc * pressure_factor,
            goc_depth=self.goc_depth * length_factor,
            pcog_goc=self.pcog_goc * pressure_factor,
            rsvd_table=self.rsvd_table,
            rvvd_table=self.rvvd_table,
            accuracy_flag=self.accuracy_flag,
            unit_system=target,
        )

    @classmethod
    def from_deck(cls, deck_file: DeckFile, eqlnum: int = 1) -> Self:
        """
        Build a single `EquilibriumRegion` from a parsed `DeckFile`.

        Reads only the requested `EQLNUM` record - does not construct
        `EquilibriumRegion` objects for any other region, unlike routing
        through `load_equilibrium_infos`.

        :param deck_file: Parsed `DeckFile` containing a `SOLUTION`-section
            `EQUIL` keyword.
        :param eqlnum: 1-based equilibration region index to load.
        :returns: `EquilibriumRegion` for that region.
        :raises ValidationError: If `EQUIL` is absent, or `eqlnum` is out
            of range.
        """
        records = deck_file.get("EQUIL")
        if not records:
            raise ValidationError(
                "No EQUIL keyword found in the DeckFile. Supply equilibration "
                "data explicitly via `EquilibriumRegion(...)`, or add an "
                "`EQUIL` block to the deck."
            )
        if not (1 <= eqlnum <= len(records)):
            raise ValidationError(
                f"EQLNUM {eqlnum} not found in EQUIL. "
                f"Available regions: 1..{len(records)}."
            )
        return typing.cast(
            Self,
            _build_equilibrium_region_from_record(
                records[eqlnum - 1], eqlnum, deck_file.unit_system
            ),
        )


class EquilibriumRegions(StoreSerializable):
    """
    Container mapping 1-based `EQLNUM` region index to `EquilibriumRegion`.

    Responsibilities: map EQLNUM -> `EquilibriumRegion`, provide lookup and
    iteration, support (de)serialization and Eclipse loading. Use
    `region(eqlnum)` to retrieve a region's data, and `from_deck`
    to construct from a deck.

    Example:

    ```python
    equilibrium = EquilibriumRegions.from_deck(deck_file)
    region = equilibrium.region(eqlnum_array[cell_idx])
    ```
    """

    __slots__ = ("_regions", "rsvd_tables", "rvvd_tables", "unit_system")

    def __init__(
        self,
        regions: typing.Dict[int, EquilibriumRegion],
        *,
        rsvd_tables: typing.Optional[typing.Dict[int, DepthTable]] = None,
        rvvd_tables: typing.Optional[typing.Dict[int, DepthTable]] = None,
        unit_system: UnitSystem = UnitSystem.FIELD,
    ) -> None:
        """
        Create a new `EquilibriumRegions` container.

        :param regions: Mapping from 1-based EQLNUM index to that region's
            `EquilibriumRegion`.
        :param rsvd_tables: `{rsvd_table number: DepthTable}`, keyed by
            `EquilibriumRegion.rsvd_table` (not by EQLNUM - multiple regions may
            share one `RSVD` table number). `None` if no region uses `RSVD`.
        :param rvvd_tables: Same as `rsvd_tables` but for `RVVD`, keyed by
            `rvvd_table`.
        :param unit_system: Unit system shared by all regions and depth tables.
        """
        if not regions:
            raise ValidationError("`regions` must contain at least one entry.")

        for region in regions.values():
            if region.uses_rsvd and (
                rsvd_tables is None or region.rsvd_table not in rsvd_tables
            ):
                raise ValidationError(
                    f"EquilibriumRegion references rsvd_table={region.rsvd_table} "
                    "but no matching table was supplied in `rsvd_tables`."
                )
            if region.uses_rvvd and (
                rvvd_tables is None or region.rvvd_table not in rvvd_tables
            ):
                raise ValidationError(
                    f"EquilibriumRegion references rvvd_table={region.rvvd_table} "
                    "but no matching table was supplied in `rvvd_tables`."
                )

        mismatched = {
            num: region.unit_system
            for num, region in regions.items()
            if region.unit_system != unit_system
        }
        if mismatched:
            raise ValidationError(
                f"All `EquilibriumRegion` entries must share `{self.__class__.__name__}`."
                f"unit_system` ({unit_system.value!r}); mismatches "
                f"(eqlnum -> unit_system): "
                f"{ {k: v.value for k, v in mismatched.items()} }."
            )

        self._regions = regions
        self.rsvd_tables = rsvd_tables
        self.rvvd_tables = rvvd_tables
        self.unit_system = unit_system

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize EquilibriumRegions to a dictionary."""
        return {
            "regions": {
                str(num): region.dump(recurse) for num, region in self._regions.items()
            },
            "rsvd_tables": {
                str(num): table.dump(recurse)
                for num, table in (self.rsvd_tables or {}).items()
            }
            if self.rsvd_tables
            else None,
            "rvvd_tables": {
                str(num): table.dump(recurse)
                for num, table in (self.rvvd_tables or {}).items()
            }
            if self.rvvd_tables
            else None,
            "unit_system": self.unit_system.value,
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Deserialize EquilibriumRegions from a dictionary."""
        unit_system = UnitSystem(data["unit_system"])

        regions = {
            int(num): EquilibriumRegion.load(region_data)
            for num, region_data in data["regions"].items()
        }

        rsvd_tables = None
        if data.get("rsvd_tables"):
            rsvd_tables = {
                int(num): DepthTable.load(table_data)
                for num, table_data in data["rsvd_tables"].items()
            }

        rvvd_tables = None
        if data.get("rvvd_tables"):
            rvvd_tables = {
                int(num): DepthTable.load(table_data)
                for num, table_data in data["rvvd_tables"].items()
            }
        return cls(
            regions=regions,
            rsvd_tables=rsvd_tables,
            rvvd_tables=rvvd_tables,
            unit_system=unit_system,
        )

    @property
    def n_regions(self) -> int:
        """Number of equilibration regions."""
        return len(self._regions)

    def region(self, eqlnum: int) -> EquilibriumRegion:
        """
        Return the `EquilibriumRegion` for a given 1-based EQLNUM index.

        :param eqlnum: 1-based equilibration region index.
        :returns: `EquilibriumRegion` for that region.
        :raises KeyError: If the region index does not exist.
        """
        region = self._regions.get(eqlnum)
        if region is None:
            raise KeyError(
                f"EQLNUM {eqlnum} not found. Available regions: {sorted(self._regions)}."
            )
        return region

    def __iter__(self) -> typing.Iterator[int]:
        return iter(self._regions)

    def __len__(self) -> int:
        return len(self._regions)

    def __contains__(self, key: object) -> bool:
        return key in self._regions

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `EquilibriumRegions` with every region converted to
        *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `EquilibriumRegions` in *target* units.
        """
        if target == self.unit_system:
            return self
        return self.__class__(
            regions={
                num: region.convert(target, table=table)
                for num, region in self._regions.items()
            },
            rsvd_tables={
                num: depth_table.convert(target, table=table)
                for num, depth_table in self.rsvd_tables.items()
            }
            if self.rsvd_tables
            else None,
            rvvd_tables={
                num: depth_table.convert(target, table=table)
                for num, depth_table in self.rvvd_tables.items()
            }
            if self.rvvd_tables
            else None,
            unit_system=target,
        )

    @classmethod
    def from_deck(cls, deck_file: DeckFile) -> Self:
        """
        Build `EquilibriumRegions` from a parsed `DeckFile`.

        Delegates to `load_equilibrium_infos` for the `EQUIL` records so
        that parsing logic lives in exactly one place, shared with
        `EquilibriumRegion.from_deck`. Also loads `RSVD`/`RVVD` tables
        (see `bores.deck.keywords.solution`) if present in the deck.

        :param deck_file: Parsed `DeckFile` containing a `SOLUTION`-section
            `EQUIL` keyword.
        :returns: `EquilibriumRegions` keyed by 1-based EQLNUM index.
        :raises ValidationError: If `EQUIL` is absent from the deck, or an
            `EquilibriumRegion` references an `RSVD`/`RVVD` table number that
            isn't present in the deck.
        """
        mapping = load_equilibrium_infos(deck_file)
        rsvd_tables = _load_depth_tables(deck_file, "RSVD", "rs")
        rvvd_tables = _load_depth_tables(deck_file, "RVVD", "rv")
        return cls(
            regions=mapping,
            rsvd_tables=rsvd_tables,
            rvvd_tables=rvvd_tables,
            unit_system=deck_file.unit_system,
        )


def _build_equilibrium_region_from_record(
    record: typing.Mapping[str, typing.Any],
    eqlnum: int,
    unit_system: UnitSystem,
) -> EquilibriumRegion:
    """
    Build one `EquilibriumRegion` from a single parsed `EQUIL` record dict.

    The single field-mapping implementation shared by
    `EquilibriumRegion.from_deck` and `load_equilibrium_infos`, so the
    two call sites can never drift out of sync on how raw record fields
    map onto `EquilibriumRegion` fields.

    :param record: One row of `deck_file.get("EQUIL")`.
    :param eqlnum: 1-based EQLNUM index this record belongs to (used only
        for error messages).
    :param unit_system: Unit system of the source `DeckFile`.
    :returns: `EquilibriumRegion` for this record.
    :raises ValidationError: If the record fails validation.
    """
    try:
        return EquilibriumRegion(
            datum_depth=record["datum_depth"],
            datum_pressure=record["datum_pressure"],
            woc_depth=record["woc_depth"] or 0.0,
            pcow_woc=record["pcow_woc"] or 0.0,
            goc_depth=record["goc_depth"] or 0.0,
            pcog_goc=record["pcog_goc"] or 0.0,
            rsvd_table=record["rsvd_table"] or 0,
            rvvd_table=record["rvvd_table"] or 0,
            accuracy_flag=record["accuracy_flag"] or 0,
            unit_system=unit_system,
        )
    except (ValidationError, TypeError, KeyError) as exc:
        raise ValidationError(f"EQUIL record {eqlnum}: {exc}") from exc


def load_equilibrium_infos(deck_file: DeckFile) -> typing.Dict[int, EquilibriumRegion]:
    """
    Parse every `EQUIL` record in *deck_file* into `EquilibriumRegion`
    objects, keyed by 1-based EQLNUM index.

    Unlike `EquilibriumRegion.from_deck`, this genuinely needs every
    region, so building all of them here is not wasteful - it's the
    single bulk-parsing implementation used by `EquilibriumRegions.from_deck`.

    :param deck_file: Parsed `DeckFile`.
    :returns: `{eqlnum: EquilibriumRegion}` mapping, one entry per `EQUIL`
        record in file order.
    :raises ValidationError: If `EQUIL` is absent from the deck, or a
        record fails validation.
    """
    records = deck_file.get("EQUIL")
    if not records:
        raise ValidationError(
            "No EQUIL keyword found in the DeckFile. Supply equilibration "
            "data explicitly via `EquilibriumRegion(...)` / "
            "`EquilibriumRegions(regions={...})`, or add an `EQUIL` block "
            "to the deck."
        )

    unit_system = deck_file.unit_system
    return {
        idx + 1: _build_equilibrium_region_from_record(record, idx + 1, unit_system)
        for idx, record in enumerate(records)
    }
