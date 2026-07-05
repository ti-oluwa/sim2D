import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.constants import UnitConversionTable, c, get_conversion_factors
from bores.correlations import scalars
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.stores import StoreSerializable
from bores.typing import CellArray, Number, UnitSystem
from bores.utils import scale

__all__ = ["StaticPVT"]


@attrs.frozen(slots=True)
class StaticPVT(StoreSerializable):
    """
    Static/immutable fluid characterization. Pressure- and temperature-dependent
    properties are evaluated from the PVT tables or correlations during the simulation
    and may be cached separately for performance.

    Use `convert(target)` to rescale dimensional quantities to another unit system.
    """

    # Oil
    stock_tank_oil_density: typing.Optional[Number] = None
    """
    Stock-tank oil density at standard conditions.

    Units: lbm/ft³ (FIELD), kg/m³ (METRIC), g/cm³ (LAB).
    Read from the DENSITY keyword (column 1).
    Used in: ρo,res = (stock_tank_oil_density + Rs · stock_tank_gas_density) / Bo
    """

    # Water
    water_reference_pressure: typing.Optional[Number] = None
    """
    Reference pressure at which `water_reference_fvf` and
    `water_reference_compressibility` are defined.

    This is the `PVTW` reference pressure (item 1).

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    """

    water_reference_fvf: typing.Optional[Number] = None
    """
    Water formation volume factor at `water_reference_pressure` (Bw_ref).

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Approximately 1.00-1.08 depending on salinity and temperature.
    """

    water_reference_viscosity: typing.Optional[Number] = None
    """
    Water viscosity at reference conditions (μw_ref).

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Approximately 0.3-1.0 cP at reservoir temperature.
    """

    water_reference_compressibility: typing.Optional[Number] = None
    """
    Water compressibility at `water_reference_pressure` (cw_ref).

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Typically 3-5 x 10⁻⁶ psi⁻¹.
    """

    stock_tank_water_density: typing.Optional[Number] = None
    """
    Stock-tank water density at standard conditions.

    Units: same as `stock_tank_oil_density`.
    Read from the DENSITY keyword (column 2).
    Used in: ρw,res = stock_tank_water_density / Bw
    """

    # Gas
    stock_tank_gas_density: typing.Optional[Number] = None
    """
    Stock-tank gas density at standard conditions.

    Units: same as `stock_tank_oil_density`.
    Read from the DENSITY keyword (column 3).
    Used in: ρg,res = (stock_tank_gas_density + Rv · stock_tank_oil_density) / Bg  [wet gas]
            ρg,res = stock_tank_gas_density / Bg                           [dry gas]
    """

    water_viscosibility: typing.Optional[Number] = None
    """
    Water viscosibility - rate of change of water viscosity with pressure
    (d ln μw / dP), item 5 of the `PVTW` record.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Zero for incompressible-viscosity water (the common default).
    """

    water_salinity: typing.Optional[Number] = None
    """
    Formation water salinity (ppm NaCl).

    Assumed spatially and temporally constant. Used in brine density and
    viscosity correlations (e.g. Batzle-Wang). Typical seawater: 35 000 ppm.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities are expressed.

    Dimensionless fields (specific gravity, API, gas gravity, molecular
    weight, miscibility_model, reservoir_gas) are unaffected by unit
    conversion.
    """

    @property
    def oil_specific_gravity(self) -> typing.Optional[Number]:
        """
        Oil specific gravity relative to fresh water at standard conditions (dimensionless).
        """
        if self.stock_tank_oil_density is None:
            return None

        # Derive oil specific gravity from stock-tank oil density relative to
        # fresh water at standard conditions in unit_system.
        if self.unit_system == UnitSystem.FIELD:
            reference_water_density = c.STANDARD_WATER_DENSITY_IMPERIAL
        elif self.unit_system == UnitSystem.METRIC:
            reference_water_density = c.STANDARD_WATER_DENSITY_METRIC
        elif self.unit_system == UnitSystem.SI:
            reference_water_density = c.STANDARD_WATER_DENSITY_SI
        else:  # UnitSystem.LAB
            reference_water_density = c.STANDARD_WATER_DENSITY_LAB
        return self.stock_tank_oil_density / reference_water_density

    @property
    def gas_gravity(self) -> typing.Optional[Number]:
        """
        Gas specific gravity relative to air (dimensionless).
        """
        if self.stock_tank_gas_density is None:
            return None
        # Derive gas gravity from stock-tank gas density relative to air.
        if self.unit_system == UnitSystem.FIELD:
            reference_air_density = c.STANDARD_AIR_DENSITY_IMPERIAL
        elif self.unit_system == UnitSystem.METRIC:
            reference_air_density = c.STANDARD_AIR_DENSITY_METRIC
        elif self.unit_system == UnitSystem.SI:
            reference_air_density = c.STANDARD_AIR_DENSITY_SI
        else:  # UnitSystem.LAB
            reference_air_density = c.STANDARD_AIR_DENSITY_LAB
        return self.stock_tank_gas_density / reference_air_density

    @property
    def oil_api_gravity(self) -> typing.Optional[Number]:
        """
        Oil API gravity (°API), computed as 141.5 / SG - 131.5.

        Provided for convenience; redundant with `oil_specific_gravity`.
        """
        if self.oil_specific_gravity is None:
            return None
        return scalars.compute_oil_api_gravity(self.oil_specific_gravity)

    @property
    def gas_molecular_weight(self) -> typing.Optional[Number]:
        """
        Gas molecular weight (g/mol) computed from the gas gravity.
        """
        if self.gas_gravity is None:
            return None
        return scalars.compute_gas_molecular_weight(self.gas_gravity)

    @classmethod
    def from_deck_file(
        cls,
        deck_file: DeckFile,
        *,
        pvtnum: int = 0,
        salinity: Number = 0.0,
    ) -> Self:
        """
        Load static PVT properties from a parsed `DeckFile` for a given region.

        Extracts DENSITY and PVTW records for the specified PVT region and
        constructs a `StaticPVT` instance with the stock-tank densities and
        water reference properties.

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param pvtnum: 1-based PVT region index (matches Eclipse PVTNUM).
        :param salinity: Water salinity in ppm NaCl (default 0).
        :returns: New `StaticPVT` instance populated from deck data.
        :raises ValidationError: If required PVTW record is missing for the region.
        """
        # Convert 1-based pvtnum to 0-based index
        region_idx = max(pvtnum - 1, 0)

        # Extract DENSITY record for this region
        density_record: typing.Optional[typing.Dict[str, Number]] = None
        density_all = deck_file.get("DENSITY")
        if density_all is not None and region_idx < len(density_all):
            region_density_rows = density_all[region_idx]
            if region_density_rows:
                density_record = region_density_rows[0]

        # Extract PVTW record for this region
        pvtw_record: typing.Optional[typing.Dict[str, Number]] = None
        pvtw_all = deck_file.get("PVTW")
        if pvtw_all is not None and region_idx < len(pvtw_all):
            region_pvtw_rows = pvtw_all[region_idx]
            if region_pvtw_rows:
                pvtw_record = region_pvtw_rows[0]

        if pvtw_record is None:
            raise ValidationError(
                f"PVTW record not found for region {pvtnum}. "
                "PVTW is required to specify water reference properties."
            )

        # Extract stock-tank densities from DENSITY record
        stock_tank_oil_density: typing.Optional[Number] = None
        stock_tank_water_density: typing.Optional[Number] = None
        stock_tank_gas_density: typing.Optional[Number] = None
        if density_record is not None:
            stock_tank_oil_density = density_record.get("oil")
            stock_tank_water_density = density_record.get("water")
            stock_tank_gas_density = density_record.get("gas")

        # Extract water reference properties from PVTW record
        water_reference_pressure = pvtw_record.get("p_ref")
        water_reference_fvf = pvtw_record.get("bw")
        water_reference_viscosity = pvtw_record.get("viscosity")
        water_reference_compressibility = pvtw_record.get("cw")
        water_viscosibility = pvtw_record.get("cv", 0.0)
        return cls(
            stock_tank_oil_density=stock_tank_oil_density,
            water_reference_pressure=water_reference_pressure,
            water_reference_fvf=water_reference_fvf,
            water_reference_viscosity=water_reference_viscosity,
            water_reference_compressibility=water_reference_compressibility,
            stock_tank_water_density=stock_tank_water_density,
            stock_tank_gas_density=stock_tank_gas_density,
            water_viscosibility=water_viscosibility,
            water_salinity=salinity,
            unit_system=deck_file.unit_system,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `PVT` with dimensional quantities rescaled to
        *target*.

        Dimensionless fields are copied unchanged.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `PVT` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            water_salinity=self.water_salinity,
            water_reference_pressure=scale(
                self.water_reference_pressure, factors["pressure"]
            ),
            water_reference_fvf=scale(self.water_reference_fvf, factors["liquid_fvf"]),
            water_reference_viscosity=scale(
                self.water_reference_viscosity, factors["viscosity"]
            ),
            water_reference_compressibility=scale(
                self.water_reference_compressibility, factors["compressibility"]
            ),
            water_viscosibility=scale(
                self.water_viscosibility, factors["compressibility"]
            ),
            stock_tank_oil_density=scale(
                self.stock_tank_oil_density, factors["density"]
            ),
            stock_tank_water_density=scale(
                self.stock_tank_water_density, factors["density"]
            ),
            stock_tank_gas_density=scale(
                self.stock_tank_gas_density, factors["density"]
            ),
            unit_system=target,
        )


@attrs.frozen(slots=True)
class PVTCache:
    """
    Transient per-cell PVT quantities derived from pressure, temperature, and the PVT tables.

    Every field in this class is a function of the current `State`
    pressure (and optionally Rs, Rv, Rsw) evaluated against the parsed PVT
    tables. They are recomputed at the start of each Newton iteration and
    discarded at the end of each time step.

    All arrays are shape `(n_cells,)` and indexed in the same order as
    `Grid.cell_centroids`.  Units follow `State.unit_system`.
    """

    oil_fvf: CellArray
    """
    Oil formation volume factor at reservoir pressure (Bo).

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Interpolated from the Bo-pressure PVT table (`PVTO` saturated branch
    or undersaturated correction above bubble point).
    """

    water_fvf: CellArray
    """
    Water formation volume factor at reservoir pressure (Bw).

    Units: same as `oil_fvf`.
    Evaluated from the `PVTW` table using the exponential FVF model:
    Bw = Bw_ref x exp(-cw x (P - P_ref)).
    """

    gas_fvf: CellArray
    """
    Gas formation volume factor at reservoir pressure (Bg).

    Units: ft³/SCF (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Derived from the real-gas law: Bg = (z · T / P) x (P_STC / T_STC).
    """

    oil_viscosity: CellArray
    """
    Live-oil viscosity at reservoir conditions (μo).

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Interpolated from `PVTO` using the Beggs-Robinson or tabular method.
    """

    water_viscosity: CellArray
    """
    Water viscosity at reservoir conditions (μw).

    Units: same as `oil_viscosity`.
    Evaluated from the `PVTW` viscosibility model:
    μw = μw_ref x exp(-cvw x (P - P_ref)).
    """

    gas_viscosity: CellArray
    """
    Free-gas viscosity at reservoir conditions (μg).

    Units: same as `oil_viscosity`. Typically 0.01-0.05 cP.
    Evaluated via the Lee-Kesler / Carr-Kobayashi-Burrows correlations or
    interpolated from `PVDG`.
    """

    oil_density: CellArray
    """
    Live-oil density at reservoir conditions (ρo).

    Units: lbm/ft³ (FIELD), kg/m³ (METRIC / SI), g/cm³ (LAB).
    Computed from stock-tank density and FVF:
    ρo_res = (ρo_STC + Rs · ρg_STC) / Bo.
    """

    water_density: CellArray
    """
    Water density at reservoir conditions (ρw).

    Units: same as `oil_density`.
    Computed as: ρw_res = ρw_STC / Bw.
    """

    gas_density: CellArray
    """
    Free-gas density at reservoir conditions (ρg).

    Units: same as `oil_density`.
    Computed as: ρg_res = ρg_STC / Bg (for dry gas);
    includes Rv correction for wet-gas / condensate:
    ρg_res = (ρg_STC + Rv · ρo_STC) / Bg.
    """

    oil_compressibility: CellArray
    """
    Oil compressibility at current pressure (co).

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Pressure-dependent; used in the undersaturated-oil accumulation term.
    In the saturated region derived analytically from the Bo-Rs relationship:
    co = (1/Bo) x dBo/dP - (1/Bo) x Rs x dRs/dP x Bg_sat.
    """

    water_compressibility: CellArray
    """
    Water compressibility at current pressure (cw).

    Units: same as `oil_compressibility`.
    Approximately 3-5 x 10⁻⁶ psi⁻¹ at reservoir conditions.
    Sourced from item 3 of `PVTW` (constant with pressure in the standard
    Eclipse black-oil water model).
    """

    gas_compressibility: CellArray
    """
    Gas compressibility at current pressure (cg).

    Units: same as `oil_compressibility`.
    For real gas: cg = 1/P - (1/z) x (dz/dP).
    """

    gas_compressibility_factor: CellArray
    """
    Real-gas z-factor (dimensionless).

    Interpolated from the z-pressure table or computed via a correlation
    (e.g. Pitzer-Curl, Hall-Yarborough, Dranchuk-Abou-Kassem).
    Used in gas FVF: Bg = z · T / P x P_STC / T_STC, and in gas
    compressibility: cg = 1/P - (1/z) x (dz/dP).
    """

    oil_effective_viscosity: CellArray = attrs.field(
        factory=lambda: np.zeros(0, dtype=get_dtype())
    )
    """
    Effective oil-solvent mixture viscosity (μo_eff).

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Computed via the Todd-Longstaff mixing rule when
    `State.solvent_concentration` is non-zero; equals
    `oil_viscosity` for immiscible flow.
    Empty array for standard black-oil runs.
    """

    oil_effective_density: CellArray = attrs.field(
        factory=lambda: np.zeros(0, dtype=get_dtype())
    )
    """
    Effective oil-solvent mixture density (ρo_eff).

    Units: same as `oil_density`.
    Empty array for standard black-oil runs.
    """
