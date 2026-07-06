"""
Equilibrium (`EQUIL`) region data.

Holds the raw gravity/capillary equilibration inputs parsed from the
Eclipse `SOLUTION` section - one `EquilibriumInfo` per `EQLNUM` region.
No pressures, saturations, or Rs/Rv are computed here; that is the job
of `bores.reservoir.initialization.initialize_equilibrium_state`, which
consumes an `EquilibriumRegions` instance alongside a `Grid`, `Rock`,
and `BlackOilModel`.
"""

import typing

import attrs
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.stores import StoreSerializable
from bores.typing import Number, UnitSystem

__all__ = ["EquilibriumInfo", "EquilibriumRegions", "load_equilibrium_infos"]


@attrs.frozen(slots=True)
class EquilibriumInfo(StoreSerializable):
    """
    Gravity/capillary equilibration data for a single `EQLNUM` region -
    one record of the Eclipse `EQUIL` keyword.

    Responsibilities: store one `EQUIL` record, validate it, support
    unit conversion and (de)serialization, and support loading from an
    Eclipse deck. It does **not** compute pressures, saturations, Rs, or
    Rv - see `bores.reservoir.initialization` for that.

    Convention: `woc_depth` / `goc_depth` of `0.0` (the Eclipse default
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

    Dispatched in `bores.reservoir.initialization` by
    `_initialize_center_point_equilibrium`,
    `_initialize_horizontal_subdivision_equilibrium`, and
    `_initialize_tilted_subdivision_equilibrium` respectively.
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
        Return a new `EquilibriumInfo` with all dimensional fields rescaled
        to *target*.

        Depths use the length factor; pressures (`datum_pressure`,
        `pcow_woc`, `pcog_goc`) use the pressure factor. Table numbers and
        `accuracy_flag` are dimensionless and copied unchanged.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the
            default.
        :returns: New `EquilibriumInfo` in *target* units.
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
    def from_deck_file(cls, deck_file: DeckFile, eqlnum: int = 1) -> Self:
        """
        Build a single `EquilibriumInfo` from a parsed `DeckFile`.

        Locates the requested `EQLNUM` record and constructs an object;
        performs no state initialization.

        :param deck_file: Parsed `DeckFile` containing a `SOLUTION`-section
            `EQUIL` keyword.
        :param eqlnum: 1-based equilibration region index to load.
        :returns: `EquilibriumInfo` for that region.
        :raises ValidationError: If `EQUIL` is absent, or `eqlnum` is out
            of range.
        """
        infos = load_equilibrium_infos(deck_file)
        info = infos.get(eqlnum)
        if info is None:
            raise ValidationError(
                f"EQLNUM {eqlnum} not found in EQUIL. "
                f"Available regions: {sorted(infos)}."
            )
        return info


@attrs.frozen(slots=True)
class EquilibriumRegions(StoreSerializable):
    """
    Container mapping 1-based `EQLNUM` region index to `EquilibriumInfo`.

    Responsibilities: map EQLNUM -> `EquilibriumInfo`, provide lookup and
    iteration, support (de)serialization and Eclipse loading. Use
    `for_region(eqlnum)` to retrieve a region's data, and `from_deck_file`
    to construct from a deck.

    Example:

    ```python
    equilibrium_regions = EquilibriumRegions.from_deck_file(deck_file)
    info = equilibrium_regions.for_region(eqlnum_array[cell_idx])
    ```
    """

    regions: typing.Dict[int, EquilibriumInfo]
    """Mapping from 1-based EQLNUM index to that region's `EquilibriumInfo`."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system shared by all regions."""

    def __attrs_post_init__(self) -> None:
        if not self.regions:
            raise ValidationError("`regions` must contain at least one entry.")

        mismatched = {
            num: info.unit_system
            for num, info in self.regions.items()
            if info.unit_system != self.unit_system
        }
        if mismatched:
            raise ValidationError(
                "All `EquilibriumInfo` entries must share `EquilibriumRegions."
                f"unit_system` ({self.unit_system.value!r}); mismatches "
                f"(eqlnum -> unit_system): "
                f"{ {k: v.value for k, v in mismatched.items()} }."
            )

    @property
    def n_regions(self) -> int:
        """Number of equilibration regions."""
        return len(self.regions)

    def for_region(self, eqlnum: int) -> EquilibriumInfo:
        """
        Return the `EquilibriumInfo` for a given 1-based EQLNUM index.

        :param eqlnum: 1-based equilibration region index.
        :returns: `EquilibriumInfo` for that region.
        :raises KeyError: If the region index does not exist.
        """
        info = self.regions.get(eqlnum)
        if info is None:
            raise KeyError(
                f"EQLNUM {eqlnum} not found. Available regions: {sorted(self.regions)}."
            )
        return info

    def __iter__(self) -> typing.Iterator[int]:
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    def __contains__(self, key: object) -> bool:
        return key in self.regions

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
                num: info.convert(target, table=table)
                for num, info in self.regions.items()
            },
            unit_system=target,
        )

    @classmethod
    def from_deck_file(cls, deck_file: DeckFile) -> Self:
        """
        Build `EquilibriumRegions` from a parsed `DeckFile`.

        Delegates to `load_equilibrium_infos` so the parsing logic lives in
        exactly one place, shared with `EquilibriumInfo.from_deck_file`.

        :param deck_file: Parsed `DeckFile` containing a `SOLUTION`-section
            `EQUIL` keyword.
        :returns: `EquilibriumRegions` keyed by 1-based EQLNUM index.
        :raises ValidationError: If `EQUIL` is absent from the deck.
        """
        mapping = load_equilibrium_infos(deck_file)
        return cls(regions=mapping, unit_system=deck_file.unit_system)


def load_equilibrium_infos(deck_file: DeckFile) -> typing.Dict[int, EquilibriumInfo]:
    """
    Parse every `EQUIL` record in *deck_file* into `EquilibriumInfo`
    objects, keyed by 1-based EQLNUM index.

    The single parsing implementation shared by both
    `EquilibriumInfo.from_deck_file` and `EquilibriumRegions.from_deck_file` -
    avoids duplicated parsing logic between the two call sites.

    :param deck_file: Parsed `DeckFile`.
    :returns: `{eqlnum: EquilibriumInfo}` mapping, one entry per `EQUIL`
        record in file order.
    :raises ValidationError: If `EQUIL` is absent from the deck, or a
        record fails validation.
    """
    records = deck_file.get("EQUIL")
    if not records:
        raise ValidationError(
            "No EQUIL keyword found in the DeckFile. Supply equilibration "
            "data explicitly via `EquilibriumInfo(...)` / "
            "`EquilibriumRegions(regions={...})`, or add an `EQUIL` block "
            "to the deck."
        )

    unit_system = deck_file.unit_system
    infos: typing.Dict[int, EquilibriumInfo] = {}
    for region_idx, record in enumerate(records):
        eqlnum = region_idx + 1  # 1-based
        try:
            infos[eqlnum] = EquilibriumInfo(
                datum_depth=float(record["datum_depth"]),
                datum_pressure=float(record["datum_pressure"]),
                woc_depth=float(record["woc_depth"] or 0.0),
                pcow_woc=float(record["pcow_woc"] or 0.0),
                goc_depth=float(record["goc_depth"] or 0.0),
                pcog_goc=float(record["pcog_goc"] or 0.0),
                rsvd_table=int(record["rsvd_table"] or 0),
                rvvd_table=int(record["rvvd_table"] or 0),
                accuracy_flag=int(record["accuracy_flag"] or 0),
                unit_system=unit_system,
            )
        except (ValidationError, TypeError, KeyError) as exc:
            raise ValidationError(f"EQUIL record {eqlnum}: {exc}") from exc

    return infos
