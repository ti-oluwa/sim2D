import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.reservoir.compressibility import (
    RockCompressibility,
    RockCompressibilityRegions,
)
from bores.reservoir.regions import _load_region_array
from bores.serde.stores import StoreSerializable
from bores.typing import (
    CellArray,
    IntCellArray,
    InterpolationMethod,
    Number,
    UnitSystem,
)
from bores.utils import scale

__all__ = ["RockPermeability", "Rock"]


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

    y: CellArray = attrs.field(factory=lambda: np.empty(0))
    """
    Shape (n_cells,) - permeability in the y-direction.

    Defaults to `x` (isotropic y) when not supplied.
    Units: same as `x`.
    """

    z: CellArray = attrs.field(factory=lambda: np.empty(0))
    """
    Shape (n_cells,) - permeability in the z-direction.

    Defaults to `x` (isotropic z) when not supplied.
    Units: same as `x`.
    """

    mean: CellArray = attrs.field(factory=lambda: np.empty(0))
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
    def from_deck(
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
class Rock(StoreSerializable):
    """
    Static petrophysical properties of the reservoir rock.

    These arrays are constant between simulation time steps and are populated
    from deck keywords such as `PORO`, `PERMX/Y/Z`, `NTG`, `SWCON`,
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

    compressibility_regions: typing.Optional[RockCompressibilityRegions] = None
    """
    Formation compressibility tables - one per `ROCKNUM` region in `unit_system`.

    Use `get_compressibility(...)` to get a `RockCompressibility` for given pressures
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities on this object are expressed.

    Dimensionless arrays (porosity, NTG, saturations) are unaffected by
    unit conversion.
    """

    def __attrs_post_init__(self) -> None:
        if (
            self.compressibility_regions is not None
            and self.compressibility_regions.unit_system != self.unit_system
        ):
            raise ValidationError(
                "`compressibility_regions.unit_system does` not match `unit_system`: "
                f"{self.compressibility_regions.unit_system} != {self.unit_system}"
            )

    def get_compressibility(
        self,
        *,
        pressure: CellArray,
        rock_regions: typing.Optional[IntCellArray] = None,
        unit_system: typing.Optional[UnitSystem] = None,
        dtype: npt.DTypeLike = None,
    ) -> RockCompressibility:
        """
        Return formation compressibility tensor.

        Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
        """
        target_unit_system = (
            unit_system if unit_system is not None else self.unit_system
        )
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        if self.compressibility_regions is None:
            return RockCompressibility(
                reference_pressure=pressure,
                compressibility=typing.cast(
                    CellArray, np.zeros_like(pressure, dtype=dtype)
                ),
                unit_system=target_unit_system,
            )

        compressibility = self.compressibility_regions.to_rock_compressibility(
            pressure=pressure,
            rock_regions=rock_regions,
            unit_system=target_unit_system,
            dtype=dtype,
        )
        return compressibility

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
            compressibility_regions=self.compressibility_regions.convert(
                target, table=table
            )
            if self.compressibility_regions is not None
            else None,
            connate_water_saturation=self.connate_water_saturation,
            irreducible_water_saturation=self.irreducible_water_saturation,
            residual_oil_saturation_water_flood=self.residual_oil_saturation_water_flood,
            residual_oil_saturation_gas_flood=self.residual_oil_saturation_gas_flood,
            residual_gas_saturation=self.residual_gas_saturation,
            unit_system=target,
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        grid: Grid,
        rock_regions: typing.Optional[IntCellArray] = None,
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
        unit_system = deck_file.unit_system
        n_cells = grid.n_cells
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()

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

        permeability = RockPermeability.from_deck(
            deck_file, n_cells=n_cells, dtype=dtype
        )
        compressibility_regions: typing.Optional[RockCompressibilityRegions] = None
        if deck_file.has("ROCK") or deck_file.has("ROCKTAB"):
            compressibility_regions = RockCompressibilityRegions.from_deck(
                deck_file, interpolation_method=interpolation_method, dtype=dtype
            )

        if rock_regions is None:
            rock_regions = _load_region_array(deck_file, "ROCKNUM", n_cells)

        return cls(
            porosity=_required("PORO"),
            absolute_permeability=permeability,
            net_to_gross=_optional("NTG", 1.0),
            compressibility_regions=compressibility_regions,
            connate_water_saturation=_optional("SWCON", 0.0),
            irreducible_water_saturation=_optional("SWCRIT", 0.0),
            residual_oil_saturation_water_flood=_optional("SOWCR", 0.0),
            residual_oil_saturation_gas_flood=_optional("SOGCR", 0.0),
            residual_gas_saturation=_optional("SGCR", 0.0),
            unit_system=unit_system,
        )
