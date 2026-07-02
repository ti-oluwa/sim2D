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
from bores.stores import StoreSerializable
from bores.typing import (
    BooleanCellArray,
    CellArray,
    IntCellArray,
    MiscibilityModel,
    Number,
    UnitSystem,
)
from bores.utils import scale, scale_and_offset

__all__ = ["PVT", "PVTCache"]


@attrs.frozen(slots=True)
class PVT(StoreSerializable):
    """
    Static PVT reference characterisation of the reservoir fluids.

    Stores the per-fluid scalars read once from the PVT deck that do not
    change between time steps. Quantities that *do* vary with pressure -
    FVFs, viscosities, densities, Rs, Rv, Rsw, bubble-point and dew-point
    pressures, z-factor, and compressibilities, live in `PVTCache` and
    are recomputed from the PVT tables each Newton iteration.

    This split reflects the physical distinction between what fluids *are*
    (static characterisation, stored here) and what the current reservoir
    *condition* looks like (transient evaluation, stored in `PVTCache`).

    Use `convert(target)` to rescale dimensional quantities to another
    unit system.
    """

    reference_temperature: Number

    # Oil

    oil_specific_gravity: Number
    """
    Oil specific gravity relative to fresh water at 60 °F (dimensionless).

    Constant for a given crude; typically 0.75-0.95.  Used to derive the
    stock-tank oil density: ρ_o,STC = oil_specific_gravity x ρ_water_STC.
    """

    oil_api_gravity: Number
    """
    Oil API gravity (°API), computed as 141.5 / SG - 131.5.

    Provided for convenience; redundant with `oil_specific_gravity`.
    """

    oil_reference_fvf: Number
    """
    Oil formation volume factor at bubble-point (reference) pressure (Bo_ref).

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Used to initialise `PVTCache.oil_fvf` before the first Newton iteration.
    """

    oil_reference_viscosity: Number
    """
    Dead-oil viscosity at standard conditions.

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Serves as the correlation reference value; live-oil viscosity at
    reservoir conditions is evaluated in `PVTCache`.
    """

    oil_reference_compressibility: Number
    """
    Oil compressibility at bubble-point (reference) pressure.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Used for the undersaturated-oil compressibility term above bubble point.
    """

    standard_oil_density: Number
    """
    Stock-tank oil density at standard conditions.

    Units: lbm/ft³ (FIELD), kg/m³ (METRIC), g/cm³ (LAB).
    Read from the DENSITY keyword (column 1).
    Used in: ρo,res = (standard_oil_density + Rs · standard_gas_density) / Bo
    """

    # Water

    water_salinity: Number
    """
    Formation water salinity (ppm NaCl).

    Assumed spatially and temporally constant. Used in brine density and
    viscosity correlations (e.g. Batzle-Wang). Typical seawater: 35 000 ppm.
    """

    water_reference_pressure: Number
    """
    Reference pressure at which `water_reference_fvf` and
    `water_reference_compressibility` are defined.

    This is the `PVTW` reference pressure (item 1).

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    """

    water_reference_fvf: Number
    """
    Water formation volume factor at `water_reference_pressure` (Bw_ref).

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Approximately 1.00-1.08 depending on salinity and temperature.
    """

    water_reference_viscosity: Number
    """
    Water viscosity at reference conditions (μw_ref).

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Approximately 0.3-1.0 cP at reservoir temperature.
    """

    water_reference_compressibility: Number
    """
    Water compressibility at `water_reference_pressure` (cw_ref).

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Typically 3-5 x 10⁻⁶ psi⁻¹.
    """

    standard_water_density: Number
    """
    Stock-tank water density at standard conditions.

    Units: same as standard_oil_density.
    Read from the DENSITY keyword (column 2).
    Used in: ρw,res = standard_water_density / Bw
    """

    standard_gas_density: Number
    """
    Stock-tank gas density at standard conditions.

    Units: same as standard_oil_density.
    Read from the DENSITY keyword (column 3).
    Used in: ρg,res = (standard_gas_density + Rv · standard_oil_density) / Bg  [wet gas]
            ρg,res = standard_gas_density / Bg                           [dry gas]
    """

    water_viscosibility: Number = 0.0
    """
    Water viscosibility - rate of change of water viscosity with pressure
    (d ln μw / dP), item 5 of the `PVTW` record.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Zero for incompressible-viscosity water (the common default).
    """

    # Gas

    reservoir_gas: str = "methane"
    """
    Name or identifier of the reservoir (or injected) gas
    (e.g. `"methane"`, `"CO2"`).

    Used to select z-factor correlations and for documentation.
    """

    gas_gravity: Number = 0.6
    """
    Gas specific gravity relative to air (dimensionless).

    0.556 for pure methane; up to ~0.9 for rich condensate gas. Input to
    pseudo-critical property correlations (Sutton, Pitzer) for z-factor and
    viscosity.
    """

    gas_molecular_weight: Number = 16.04
    """
    Gas molecular weight (g/mol).

    Methane: 16.04 g/mol; CO₂: 44.01 g/mol.
    Should satisfy MW ≈ 28.97 x gas_gravity.
    """

    gas_reference_viscosity: Number = 0.012
    """
    Gas viscosity at reference (standard) conditions.

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Typically 0.01-0.03 cP at reservoir temperature.
    """

    # Miscible / solvent (EOR)

    miscibility_model: MiscibilityModel = "immiscible"
    """
    Miscibility model identifier.

    `"immiscible"` - standard black-oil.
    `"todd-longstaff"` - first-contact miscible EOR.

    Controls how solvent concentration in the oil phase is handled and which
    mixing rules are applied for effective viscosity and density.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities are expressed.

    Dimensionless fields (specific gravity, API, gas gravity, molecular
    weight, miscibility_model, reservoir_gas) are unaffected by unit
    conversion.
    """

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

        Dimensionless fields (gravities, API, molecular weight,
        `miscibility_model`, `reservoir_gas`) are copied unchanged.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `PVT` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            reference_temperature=scale_and_offset(
                self.reference_temperature,
                factor=factors["temperature"],
                offset=factors["temperature_offset"],
            ),
            oil_specific_gravity=self.oil_specific_gravity,
            oil_api_gravity=self.oil_api_gravity,
            oil_reference_fvf=scale(self.oil_reference_fvf, factors["liquid_fvf"]),
            oil_reference_viscosity=scale(
                self.oil_reference_viscosity, factors["viscosity"]
            ),
            oil_reference_compressibility=scale(
                self.oil_reference_compressibility, factors["compressibility"]
            ),
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
            standard_oil_density=scale(self.standard_oil_density, factors["density"]),
            standard_water_density=scale(
                self.standard_water_density, factors["density"]
            ),
            standard_gas_density=scale(self.standard_gas_density, factors["density"]),
            reservoir_gas=self.reservoir_gas,
            gas_gravity=self.gas_gravity,
            gas_molecular_weight=self.gas_molecular_weight,
            gas_reference_viscosity=scale(
                self.gas_reference_viscosity, factors["viscosity"]
            ),
            miscibility_model=self.miscibility_model,
            unit_system=target,
        )


@attrs.frozen(slots=True)
class PVTCache:
    """
    Transient per-cell PVT quantities derived from pressure and the PVT tables.

    Every field in this class is a function of the current `State`
    pressure (and optionally Rs, Rv, Rsw) evaluated against the parsed PVT
    tables.  They are recomputed at the start of each Newton iteration and
    discarded at the end of each time step.

    **This class is deliberately not** `Serializable` **and not**
    `StoreSerializable`. It must never be checkpointed, persisted, or
    passed to `evolve_state`. If you need to write PVT quantities to an
    output file for post-processing, compute them on demand from the stored
    `State` and the PVT tables at output time - exactly as Eclipse
    does when it writes `DENO`, `DENG` etc. via `RPTRST`.

    Use `bores.model.pvt.compute_pvt_cache` to construct an instance.

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

    Units: ft³/scf (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
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
