import typing

import attrs
import numpy as np
import numpy.typing as npt
from bores.reservoir.blackoil.regions import _load_region
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.stores import StoreSerializable
from bores.typing import (
    CellArray,
    IntCellArray,
    InterpolationMethod,
    Number,
    UnitSystem,
)
from bores.utils import scale, scale_and_offset

__all__ = ["RockPermeability", "Rock", "RockCompressibility", "TemperatureRegions"]


def _load_cell_array(
    deck_file: DeckFile,
    keyword: str,
    n_cells: int,
    dtype: npt.DTypeLike = None,
) -> typing.Optional[CellArray]:
    arr = deck_file.get(keyword)
    if arr is None:
        return None

    arr = arr.astype(dtype, copy=False)
    if arr.size != n_cells:
        raise ValidationError(f"{keyword} has {arr.size} values; expected {n_cells}.")
    return typing.cast(CellArray, arr)


@attrs.frozen(slots=True)
class RockPermeability(StoreSerializable):
    """
    Absolute permeability tensor stored as three orthogonal components.

    If only `x` is supplied, `y` and `z` default to `x` (isotropic
    assumption). The geometric-mean `mean` is computed automatically when
    not provided.

    Units should follow the parent `Rock.unit_system`.
    """

    x: CellArray
    """
    Shape (n_cells,) - permeability in the x-direction.

    Units: mD (FIELD / METRIC / LAB) or m² (SI).
    Must be strictly positive for every active cell.
    """

    y: CellArray = attrs.field(factory=lambda: np.empty(0, dtype=get_dtype()))
    """
    Shape (n_cells,) - permeability in the y-direction.

    Defaults to `x` (isotropic y) when not supplied.
    Units: same as `x`.
    """

    z: CellArray = attrs.field(factory=lambda: np.empty(0, dtype=get_dtype()))
    """
    Shape (n_cells,) - permeability in the z-direction.

    Defaults to `x` (isotropic z) when not supplied.
    Units: same as `x`.
    """

    mean: CellArray = attrs.field(factory=lambda: np.empty(0, dtype=get_dtype()))
    """
    Shape (n_cells,) - geometric-mean permeability (Kx·Ky·Kz)^(1/3).

    Computed automatically from `x`, `y`, `z` when not supplied.
    Units: same as `x`.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all quantities on this object are expressed.
    """

    def __attrs_post_init__(self) -> None:
        if self.y.size == 0:
            object.__setattr__(self, "y", self.x)
        if self.z.size == 0:
            object.__setattr__(self, "z", self.x)
        if self.mean.size == 0:
            if np.array_equal(self.x, self.y) and np.array_equal(self.x, self.z):
                object.__setattr__(self, "mean", self.x)
            else:
                object.__setattr__(
                    self, "mean", (self.x * self.y * self.z) ** (1.0 / 3.0)
                )

    def scale(self, factor: Number) -> Self:
        """Return a new instance with all components multiplied by *factor*."""
        if factor == 1.0:
            return self
        return self.__class__(
            x=scale(self.x, factor),
            y=scale(self.y, factor),
            z=scale(self.z, factor),
            mean=scale(self.mean, factor),
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
        factor: typing.Optional[Number] = None,
    ) -> Self:
        """
        Return a new `RockPermeability` with all quantities rescaled
        to *target*.

        Conversion factors are sourced from `get_conversion_factors`.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `RockPermeability` in *target* units.
        """
        if target == self.unit_system:
            return self

        if factor is None:
            factors = get_conversion_factors(self.unit_system, target, table=table)
            factor = factors["permeability"]
        if factor == 1.0:
            return self

        return self.__class__(
            x=scale(self.x, factor),
            y=scale(self.y, factor),
            z=scale(self.z, factor),
            mean=scale(self.mean, factor),
        )

    @classmethod
    def from_deck_file(
        cls,
        deck_file: DeckFile,
        *,
        n_cells: int,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        unit_system = deck_file.unit_system
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()

        permx = _load_cell_array(deck_file, "PERMX", n_cells, dtype=dtype)
        if permx is None:
            raise ValidationError("`PERMX` is required but not found in the DeckFile.")

        permy = _load_cell_array(deck_file, "PERMY", n_cells, dtype=dtype)
        # We do not copy here to save memory, since permeability is a static prop through out any simulation
        # Except the user then tries to modify it. Then its up to them as its already indicated that the class'
        # attributes are immutable and should be treated as such.
        if permy is None:
            permy = permx
        permz = _load_cell_array(deck_file, "PERMZ", n_cells, dtype=dtype)
        if permz is None:
            permz = permx
        return cls(x=permx, y=permy, z=permz, unit_system=unit_system)


@attrs.frozen(slots=True)
class RockCompressibility(StoreSerializable):
    reference_pressure: CellArray
    """
    Shape (n_cells,) - reference pressure at which each cell's pore volume equals the
    geometrically calculated value.

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    """

    compressibility: CellArray
    """
    Shape (n_cells,) - formation compressibility.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Used in the pore-volume accumulation term: dPV/dP = PV · cr.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all quantities on this object are expressed.
    """

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `RockCompressibility` with all quantities rescaled
        to *target*.

        Conversion factors are sourced from `get_conversion_factors`.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `RockCompressibility` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            reference_pressure=scale(self.reference_pressure, factors["pressure"]),
            compressibility=scale(self.compressibility, factors["compressibility"]),
            unit_system=target,
        )


@attrs.frozen(slots=True)
class TemperatureRegions:
    """
    Reservoir temperature specification.

    Represents either a single uniform reservoir temperature (`default`) or
    a mapping of 1-based PVT region indices to temperature values (`regions`).

    - `default`: Single temperature applied when no per-region mapping exists.
    - `regions`: Mapping from 1-based region index to temperature value. A
        special key `-1` is used as the default region value when present.
    - `unit_system`: Unit system used for stored temperature values.

    The companion method `as_cell_array` broadcasts region temperatures to a
    per-cell array using a provided region index array (e.g. `pvt_region`).
    Use `convert(target)` to produce a copy in a different `UnitSystem`.
    """

    default: typing.Optional[Number] = None
    """Single temperature applied when no per-region mapping exists."""
    regions: typing.Optional[typing.Dict[int, Number]] = None
    """
    Mapping from 1-based region index to temperature value. 
    A special key `-1` is used as the default region value when present.
    """
    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system used for stored temperature values."""

    def __attrs_post_init__(self) -> None:
        if self.default is None and self.regions is None:
            raise ValidationError("Either `default` or `regions` must be provided.")

        if not self.regions and not self.default:
            raise ValidationError(
                "`regions` cannot be empty when `default` is not provided."
            )

        regions = self.regions or {}
        regions[-1] = (  # -1 is the default region
            self.default
            if self.default is not None
            else np.mean(list(regions.values()))  # type: ignore
        )
        object.__setattr__(self, "regions", regions)

    def for_region(self, pvtnum: int) -> Number:
        assert self.regions
        if pvtnum in self.regions:
            return self.regions[pvtnum]
        return self.regions[-1]

    def as_cell_array(
        self, pvt_region: IntCellArray, dtype: npt.DTypeLike = None
    ) -> CellArray:
        """
        Broadcast per-region temperatures to a per-cell array.

        :param pvt_region: Shape `(n_cells,)` int array of 1-based region
            indices. Usually `Regions.pvt_region`. For each cell the temperature
            is `regions[pvt_region[i]]` when present, otherwise the default region
            value `regions[-1]` is used (or `self.default` when `regions` is absent).
        :param dtype: Optional numpy dtype for the returned array. When
            omitted `get_dtype()` is used.
        :returns: `CellArray` of shape `(n_cells,)` with temperature values.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        n_cells = pvt_region.size

        # Resolve per-region mapping and default
        assert self.regions
        regions = self.regions
        default = regions[-1]
        out = np.full(n_cells, default, dtype=dtype)
        # If there are explicit region values, broadcast them
        # Assign each region value to cells belonging to that region
        for region_idx, temperature in regions.items():
            if region_idx == -1:
                continue
            mask = pvt_region == region_idx
            if np.any(mask):
                out[mask] = temperature
        return typing.cast(CellArray, out)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """Return a copy with temperatures converted to *target* units."""
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        factor = factors["temperature"]
        offset = factors["temperature_offset"]

        new_default = (
            None
            if self.default is None
            else scale_and_offset(self.default, factor=factor, offset=offset)
        )
        new_regions: typing.Optional[typing.Dict[int, Number]] = None
        if self.regions is not None:
            new_regions = {
                k: scale_and_offset(v, factor=factor, offset=offset)
                for k, v in self.regions.items()
            }

        return self.__class__(
            default=new_default, regions=new_regions, unit_system=target
        )


@attrs.frozen(slots=True)
class Rock(StoreSerializable):
    """
    Static petrophysical properties of the reservoir rock.

    These arrays are constant between simulation time steps and are populated
    from GRDECL keywords such as `PORO`, `PERMX/Y/Z`, `NTG`, `SWCON`,
    `SWCRIT`, and `TEMPVD`.

    `temperature` lives here because it is static in standard black-oil
    (isothermal) simulations. For thermal extensions it becomes spatially
    varying but still does not change between Newton iterations; it is not
    part of the primary variable set that the solver updates.

    All saturation arrays are dimensionless fractions in [0, 1].
    Use `convert(target)` to rescale dimensional quantities to another
    unit system.
    """

    porosity: CellArray
    """
    Shape (n_cells,) - pore volume fraction (dimensionless, [0, 1]).

    Used to compute pore volume: PV = φ x NTG x V_cell.
    """

    absolute_permeability: RockPermeability
    """
    Absolute permeability tensor.

    Units: mD (FIELD / METRIC / LAB), m² (SI).
    """

    net_to_gross: CellArray
    """
    Shape (n_cells,) - net-to-gross ratio (dimensionless, [0, 1]).

    Fraction of the gross cell volume that is net reservoir rock.
    Applied as a multiplier in pore-volume and transmissibility calculations.
    """

    compressibility: RockCompressibility
    """
    Formation compressibility tensor.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    """

    temperature: CellArray
    """
    Shape (n_cells,) - reservoir temperature, one value per cell.

    Units: °F (FIELD), °C (METRIC / LAB), K (SI).

    For standard isothermal black-oil the array is uniform (a single
    reservoir temperature broadcast across all cells) or a per-region temperature distribution.

    Note:
        This quantity belongs on `Rock` rather than
        `State` because it is *not* a primary unknown as the solver
         does not update it during Newton iterations.
    """

    connate_water_saturation: CellArray
    """
    Shape (n_cells,) - connate (initial irreducible) water saturation
    (fraction).

    Lower bound on water saturation; set from geological initial conditions.
    """

    irreducible_water_saturation: CellArray
    """
    Shape (n_cells,) - irreducible water saturation during imbibition
    (fraction).

    Equal to or greater than `connate_water_saturation`.
    """

    residual_oil_saturation_water_flood: CellArray
    """
    Shape (n_cells,) - residual oil saturation at end of water flooding
    (Sor,w - fraction).

    Oil is immobile below this saturation during water-flood imbibition.
    """

    residual_oil_saturation_gas_flood: CellArray
    """
    Shape (n_cells,) - residual oil saturation at end of gas flooding
    (Sor,g - fraction).

    Oil is immobile below this saturation during gas injection.
    """

    residual_gas_saturation: CellArray
    """
    Shape (n_cells,) - residual gas saturation during imbibition (fraction).

    Gas is immobile below this saturation when water or liquid displaces gas.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities on this object are expressed.

    Dimensionless arrays (porosity, NTG, saturations) are unaffected by
    unit conversion.
    """

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `Rock` with all dimensional quantities rescaled
        to *target*.

        Dimensionless arrays (porosity, NTG, saturations) are copied unchanged.
        Conversion factors are sourced from `get_conversion_factors`.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `Rock` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            porosity=self.porosity,
            absolute_permeability=self.absolute_permeability.scale(
                factors["permeability"]
            ),
            net_to_gross=self.net_to_gross,
            compressibility=self.compressibility.convert(target, table=table),
            temperature=scale_and_offset(
                self.temperature,
                factor=factors["temperature"],
                offset=factors["temperature_offset"],
            ),
            connate_water_saturation=self.connate_water_saturation,
            irreducible_water_saturation=self.irreducible_water_saturation,
            residual_oil_saturation_water_flood=self.residual_oil_saturation_water_flood,
            residual_oil_saturation_gas_flood=self.residual_oil_saturation_gas_flood,
            residual_gas_saturation=self.residual_gas_saturation,
            unit_system=target,
        )

    @classmethod
    def from_deck_file(
        cls,
        deck_file: DeckFile,
        *,
        grid: Grid,
        pressure: CellArray,
        temperature: typing.Union[
            CellArray, Number
        ],  # Must tally with whatever is used build `PVTTables` and vice versa.
        rock_region: typing.Optional[IntCellArray] = None,
        interpolation_method: InterpolationMethod = "linear",
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `Rock` from a parsed `DeckFile` and an already-loaded `Grid`.

        Reads `PORO`, `PERMX/Y/Z`, `NTG`, `SWCON`, `SWCRIT`, `SOWCR`,
        `SOGCR`, `SGCR`, `ROCK`/`ROCKTAB`, and `TEMPVD`/`RTEMP`.

        :param deck_file: Parsed `DeckFile` containing PROPS/GRID keywords.
        :param grid: Already-loaded `Grid` (provides `n_cells` and cell
            centroid depths for temperature interpolation).
        :returns: `Rock` in the deck's unit system.
        :raises ValidationError: If required keywords (`PORO`, `PERMX`) are
            missing.
        """
        from bores.blackoil.compressibility import RockCompressibilityRegions

        unit_system = deck_file.unit_system
        n_cells = grid.n_cells
        dtype = get_dtype()

        def _required(keyword: str) -> CellArray:
            data = _load_cell_array(deck_file, keyword, n_cells, dtype=dtype)
            if data is None:
                raise ValidationError(
                    f"`{keyword}` is required but not found in the DeckFile."
                )
            return data

        def _optional(keyword: str, default: float) -> CellArray:
            data = _load_cell_array(deck_file, keyword, n_cells, dtype=dtype)
            if data is None:
                return np.full(n_cells, default, dtype=dtype)
            return data

        permeability = RockPermeability.from_deck_file(
            deck_file, n_cells=n_cells, dtype=dtype
        )
        compressibility_regions = RockCompressibilityRegions.from_deck_file(
            deck_file, interpolation_method=interpolation_method, dtype=dtype
        )
        if rock_region is None:
            rock_region = _load_region(deck_file, "ROCKNUM", n_cells)

        compressibility = compressibility_regions.to_rock_compressibility(
            pressure=pressure,
            rock_region=rock_region,
            unit_system=unit_system,
        )
        if np.isscalar(temperature):
            temperature = typing.cast(
                CellArray, np.full(n_cells, temperature, dtype=dtype)
            )
        return cls(
            porosity=_required("PORO"),
            absolute_permeability=permeability,
            net_to_gross=_optional("NTG", 1.0),
            compressibility=compressibility,
            temperature=typing.cast(CellArray, temperature),
            connate_water_saturation=_optional("SWCON", 0.0),
            irreducible_water_saturation=_optional("SWCRIT", 0.0),
            residual_oil_saturation_water_flood=_optional("SOWCR", 0.0),
            residual_oil_saturation_gas_flood=_optional("SOGCR", 0.0),
            residual_gas_saturation=_optional("SGCR", 0.0),
            unit_system=unit_system,
        )
