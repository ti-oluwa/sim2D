"""Per-cell and simulation property definitions for a black-oil reservoir model."""

import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypedDict

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
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
from spe1 import reference_pressure

__all__ = [
    "PVT",
    "Hysteresis",
    "PVTCache",
    "RockPermeability",
    "Rock",
    "State",
    "Meta",
]


def _scale(arr: CellArray, factor: Number) -> CellArray:
    """Return `arr * factor` as the same dtype; identity when factor == 1.0."""
    if factor == 1.0:
        return arr
    return typing.cast(CellArray, (arr * factor).astype(arr.dtype))


def _scale_non_empty(arr: CellArray, factor: Number) -> CellArray:
    """Scale only if the optional EOR array is non-empty."""
    return _scale(arr, factor) if arr.size > 0 else arr


def _scale_and_offset(arr: CellArray, scale: Number, offset: Number) -> CellArray:
    """Return `arr * scale + offset` as the same dtype; identity when trivial."""
    if scale == 1.0 and offset == 0.0:
        return arr
    return typing.cast(CellArray, ((arr * scale) + offset).astype(arr.dtype))


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

    def scale(self, factor: float) -> Self:
        """Return a new instance with all components multiplied by *factor*."""
        if factor == 1.0:
            return self
        return self.__class__(
            x=_scale(self.x, factor),
            y=_scale(self.y, factor),
            z=_scale(self.z, factor),
            mean=_scale(self.mean, factor),
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
            x=_scale(self.x, factor),
            y=_scale(self.y, factor),
            z=_scale(self.z, factor),
            mean=_scale(self.mean, factor),
        )


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
            reference_pressure=typing.cast(
                CellArray, self.reference_pressure * factors["pressure"]
            ),
            compressibility=typing.cast(
                CellArray, self.compressibility * factors["compressibility"]
            ),
            unit_system=target,
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
    reservoir temperature broadcast across all cells).  For thermal
    extensions the values vary spatially and are interpolated from the
    `TEMPVD` keyword (temperature vs depth) at each cell centroid.

    This quantity belongs on `Rock` rather than
    `State` because it is *not* a primary unknown - the solver
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
            temperature=_scale_and_offset(
                self.temperature,
                factors["temperature_scale"],
                factors["temperature_offset"],
            ),
            connate_water_saturation=self.connate_water_saturation,
            irreducible_water_saturation=self.irreducible_water_saturation,
            residual_oil_saturation_water_flood=self.residual_oil_saturation_water_flood,
            residual_oil_saturation_gas_flood=self.residual_oil_saturation_gas_flood,
            residual_gas_saturation=self.residual_gas_saturation,
            unit_system=target,
        )


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

    reference_temperature: float

    # Oil

    oil_specific_gravity: float
    """
    Oil specific gravity relative to fresh water at 60 °F (dimensionless).

    Constant for a given crude; typically 0.75-0.95.  Used to derive the
    stock-tank oil density: ρ_o,STC = oil_specific_gravity x ρ_water_STC.
    """

    oil_api_gravity: float
    """
    Oil API gravity (°API), computed as 141.5 / SG - 131.5.

    Provided for convenience; redundant with `oil_specific_gravity`.
    """

    oil_reference_fvf: float
    """
    Oil formation volume factor at bubble-point (reference) pressure (Bo_ref).

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Used to initialise `PVTCache.oil_fvf` before the first Newton iteration.
    """

    oil_reference_viscosity: float
    """
    Dead-oil viscosity at standard conditions.

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Serves as the correlation reference value; live-oil viscosity at
    reservoir conditions is evaluated in `PVTCache`.
    """

    oil_reference_compressibility: float
    """
    Oil compressibility at bubble-point (reference) pressure.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Used for the undersaturated-oil compressibility term above bubble point.
    """

    standard_oil_density: float
    """
    Stock-tank oil density at standard conditions.

    Units: lbm/ft³ (FIELD), kg/m³ (METRIC), g/cm³ (LAB).
    Read from the DENSITY keyword (column 1).
    Used in: ρo,res = (standard_oil_density + Rs · standard_gas_density) / Bo
    """

    # Water

    water_salinity: float
    """
    Formation water salinity (ppm NaCl).

    Assumed spatially and temporally constant. Used in brine density and
    viscosity correlations (e.g. Batzle-Wang). Typical seawater: 35 000 ppm.
    """

    water_reference_pressure: float
    """
    Reference pressure at which `water_reference_fvf` and
    `water_reference_compressibility` are defined.

    This is the `PVTW` reference pressure (item 1).

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    """

    water_reference_fvf: float
    """
    Water formation volume factor at `water_reference_pressure` (Bw_ref).

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Approximately 1.00-1.08 depending on salinity and temperature.
    """

    water_reference_viscosity: float
    """
    Water viscosity at reference conditions (μw_ref).

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Approximately 0.3-1.0 cP at reservoir temperature.
    """

    water_reference_compressibility: float
    """
    Water compressibility at `water_reference_pressure` (cw_ref).

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Typically 3-5 x 10⁻⁶ psi⁻¹.
    """

    standard_water_density: float
    """
    Stock-tank water density at standard conditions.

    Units: same as standard_oil_density.
    Read from the DENSITY keyword (column 2).
    Used in: ρw,res = standard_water_density / Bw
    """

    standard_gas_density: float
    """
    Stock-tank gas density at standard conditions.

    Units: same as standard_oil_density.
    Read from the DENSITY keyword (column 3).
    Used in: ρg,res = (standard_gas_density + Rv · standard_oil_density) / Bg  [wet gas]
            ρg,res = standard_gas_density / Bg                           [dry gas]
    """

    water_viscosibility: float = 0.0
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

    gas_gravity: float = 0.6
    """
    Gas specific gravity relative to air (dimensionless).

    0.556 for pure methane; up to ~0.9 for rich condensate gas. Input to
    pseudo-critical property correlations (Sutton, Pitzer) for z-factor and
    viscosity.
    """

    gas_molecular_weight: float = 16.04
    """
    Gas molecular weight (g/mol).

    Methane: 16.04 g/mol; CO₂: 44.01 g/mol.
    Should satisfy MW ≈ 28.97 x gas_gravity.
    """

    gas_reference_viscosity: float = 0.012
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
            reference_temperature=(
                self.reference_temperature * factors["temperature_scale"]
            )
            + factors["temperature_offset"],
            oil_specific_gravity=self.oil_specific_gravity,
            oil_api_gravity=self.oil_api_gravity,
            oil_reference_fvf=self.oil_reference_fvf * factors["liquid_fvf"],
            oil_reference_viscosity=(
                self.oil_reference_viscosity * factors["viscosity"]
            ),
            oil_reference_compressibility=(
                self.oil_reference_compressibility * factors["compressibility"]
            ),
            water_salinity=self.water_salinity,
            water_reference_pressure=(
                self.water_reference_pressure * factors["pressure"]
            ),
            water_reference_fvf=self.water_reference_fvf * factors["liquid_fvf"],
            water_reference_viscosity=(
                self.water_reference_viscosity * factors["viscosity"]
            ),
            water_reference_compressibility=(
                self.water_reference_compressibility * factors["compressibility"]
            ),
            water_viscosibility=self.water_viscosibility * factors["compressibility"],
            standard_oil_density=self.standard_oil_density * factors["density"],
            standard_water_density=self.standard_water_density * factors["density"],
            standard_gas_density=self.standard_gas_density * factors["density"],
            reservoir_gas=self.reservoir_gas,
            gas_gravity=self.gas_gravity,
            gas_molecular_weight=self.gas_molecular_weight,
            gas_reference_viscosity=(
                self.gas_reference_viscosity * factors["viscosity"]
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


@attrs.frozen(slots=True)
class Hysteresis(StoreSerializable):
    """
    Drainage / imbibition hysteresis tracking for Killough scanning curves.

    Maintains historical saturation extrema and displacement-regime flags
    required to compute effective residual saturations on the scanning curves.
    These are consumed inside relative-permeability and capillary-pressure
    evaluation routines; the flow solver does not interpret them directly.

    All arrays are dimensionless (saturations, flags) and therefore require no unit conversion.
    """

    max_water_saturation: CellArray
    """
    Shape (n_cells,) - historical maximum water saturation reached in each
    cell (fraction).

    Initialised to the initial water saturation. Updated whenever the
    current water saturation exceeds the stored maximum. Determines the
    imbibition end-point on the scanning curve when drainage reverses.
    """

    max_gas_saturation: CellArray
    """
    Shape (n_cells,) - historical maximum gas saturation reached in each
    cell (fraction).

    Analogous to `max_water_saturation` for the gas phase.
    """

    water_imbibition_flag: BooleanCellArray
    """
    Shape (n_cells,) - `True` if the current water-phase displacement is
    imbibition (water saturation increasing toward `max_water_saturation`).

    `False` indicates drainage (water saturation decreasing).
    """

    gas_imbibition_flag: BooleanCellArray
    """
    Shape (n_cells,) - `True` if the current gas-phase displacement is
    imbibition (gas saturation decreasing - water or liquid displacing gas).
    """

    water_reversal_saturation: CellArray
    """
    Shape (n_cells,) - water saturation at the most recent
    drainage-to-imbibition (or reverse) reversal point (fraction).

    Starting saturation of the Killough scanning curve when the displacement
    regime changes.
    """

    gas_reversal_saturation: CellArray
    """
    Shape (n_cells,) - gas saturation at the most recent reversal point
    (fraction).

    Analogous to `water_reversal_saturation` for the gas phase.
    """

    @classmethod
    def from_initial_saturations(
        cls,
        water_saturation: npt.ArrayLike,
        gas_saturation: npt.ArrayLike,
    ) -> Self:
        """
        Construct a `Hysteresis` from initial saturation arrays.

        Sets maximum saturations to the initial values, marks all cells as
        drainage (not yet reversing), and places reversal points at the
        initial saturation values.

        :param water_saturation: Array-like (n_cells,) - initial water
            saturation per cell (fraction).
        :param gas_saturation: Array-like (n_cells,) - initial gas saturation
            per cell (fraction).
        :returns: Initialised `Hysteresis`.
        """
        sw = np.asarray(water_saturation, dtype=get_dtype())
        sg = np.asarray(gas_saturation, dtype=get_dtype())
        return cls(
            max_water_saturation=typing.cast(CellArray, sw.copy()),
            max_gas_saturation=typing.cast(CellArray, sg.copy()),
            water_imbibition_flag=typing.cast(
                BooleanCellArray, np.zeros(sw.shape, dtype=np.bool_)
            ),
            gas_imbibition_flag=typing.cast(
                BooleanCellArray, np.zeros(sg.shape, dtype=np.bool_)
            ),
            water_reversal_saturation=typing.cast(CellArray, sw.copy()),
            gas_reversal_saturation=typing.cast(CellArray, sg.copy()),
        )

    def evolve(self, **kwargs: typing.Any) -> Self:
        """
        Return a new `Hysteresis` with selected fields replaced.

        :param kwargs: Field names and their replacement values.
        :returns: New immutable `Hysteresis`.
        """
        return attrs.evolve(self, **kwargs)


@attrs.frozen(slots=True)
class State(StoreSerializable):
    """
    Dynamic per-cell simulation state, updated at every time step.

    This is the canonical state that the solver integrates forward in time.
    It holds exactly the quantities that cannot be recomputed from static
    data alone:

    **Primary unknowns** (solved implicitly):

    - `pressure` - oil-phase reference pressure.
    - `oil_saturation`, `water_saturation`, `gas_saturation` - phase
      saturations (must sum to 1 in every cell).
    - `solution_gor` (Rs) - gas dissolved in oil per unit stock-tank oil
      volume.  *Primary variable in saturated cells* (Sg > 0); capped at
      bubble-point Rs in undersaturated cells.
    - `oil_bubble_point_pressure` - *primary variable in undersaturated oil
      cells* (Sg = 0, Po < Pbub is not possible, so here Pbub tracks the
      evolving bubble-point as the reservoir depletes below saturation
      pressure).
    - `vaporized_oil_ratio` (Rv) - stock-tank oil vaporized in gas per unit
      standard gas volume. Primary variable for volatile-oil / gas-condensate
      models when So = 0.
    - `water_bubble_point_pressure` - bubble-point pressure of the water
      phase with respect to dissolved gas (Rsw).  Relevant for CO₂ or sour-gas
      reservoirs where gas solubility in water is non-negligible.
    - `gas_dew_point_pressure` - dew-point pressure of the gas phase; primary
      variable in condensate models when So = 0.
    - `gas_solubility_in_water` (Rsw) - gas dissolved in water per unit
      stock-tank water volume.

    **Conserved component masses** (updated explicitly):

    - `oil_mass`, `water_mass`, `free_gas_mass`
    - `dissolved_gas_mass_in_oil`, `dissolved_gas_mass_in_water`
    - `vaporized_oil_mass_in_gas`

    **What is NOT here** (see `PVTCache`):

    FVFs, viscosities, densities, z-factor, and compressibilities are all
    derived from `pressure`, `solution_gor`, `vaporized_oil_ratio`,
    and the PVT tables.  They are evaluated transiently each Newton iteration
    and never checkpointed.  This mirrors how Eclipse manages its restart
    files: only primary variables are written to `.UNRST`; everything else
    is recomputed on load.

    **Temperature** lives on `Rock.temperature` because it is a
    static field (not a solver unknown) in standard isothermal black-oil.

    Use `convert(target)` to rescale to another unit system, or
    `evolve(**kwargs)` to produce a new state with selected fields replaced.
    """

    pressure: CellArray
    """
    Shape (n_cells,) - oil-phase (reference) pressure.

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    Primary implicit unknown in the pressure equation.
    Phase pressures for water and gas are recovered via capillary pressure:
    Pw = Po - Pcow,  Pg = Po + Pcgo.
    """

    oil_saturation: CellArray
    """
    Shape (n_cells,) - oil-phase saturation (fraction, [0, 1]).

    Derived as So = 1 - Sw - Sg after updating Sw and Sg.
    Must satisfy So + Sw + Sg = 1 in every cell.
    """

    water_saturation: CellArray
    """
    Shape (n_cells,) - water-phase saturation (fraction, [0, 1]).

    Updated explicitly in the IMPES saturation step.
    """

    gas_saturation: CellArray
    """
    Shape (n_cells,) - free gas-phase saturation (fraction, [0, 1]).

    Includes only free (non-dissolved, non-vaporized) gas.
    Updated explicitly in the IMPES saturation step.
    """

    solution_gor: CellArray
    """
    Shape (n_cells,) - solution gas-to-oil ratio (Rs).

    Units: scf/STB (FIELD), sm³/sm³ (METRIC / SI), scc/scc (LAB).

    Gas dissolved in oil per unit stock-tank oil volume at current pressure.

    *Primary variable in saturated cells* (Sg > 0): updated by the solver.
    *In undersaturated cells* (Sg = 0): fixed at the bubble-point value
    corresponding to `oil_bubble_point_pressure`; not independently solved.
    """

    oil_bubble_point_pressure: CellArray
    """
    Shape (n_cells,) - bubble-point pressure of the oil phase (Pbub).

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).

    *In saturated cells* (Sg > 0): computed from Rs via the PVT table - not
    a stored primary variable; could be derived from `solution_gor`.

    *In undersaturated cells* (Sg = 0): this IS a primary variable, tracking
    the evolving bubble-point as the reservoir depletes while remaining single-
    phase.  Must be stored and checkpointed in this regime.

    The two-regime handling is the same saturated/undersaturated switching
    used by Eclipse E100 (PBPD switching logic).
    """

    vaporized_oil_ratio: CellArray
    """
    Shape (n_cells,) - vaporized oil ratio (Rv).

    Units: STB/Mscf (FIELD), sm³/sm³ (METRIC / SI), scc/scc (LAB).

    Stock-tank oil vaporized in gas per unit standard gas volume.

    Non-zero only for volatile-oil and gas-condensate reservoirs.
    Set to all-zeros for standard dry-gas black-oil.
    """

    gas_dew_point_pressure: CellArray
    """
    Shape (n_cells,) - dew-point pressure of the gas phase (Pdew).

    Units: same as `oil_bubble_point_pressure`.

    Analogous to `oil_bubble_point_pressure` but for the gas phase in
    volatile-oil / gas-condensate models:

    *In two-phase cells* (So > 0): derived from Rv via the PVT table.
    *In single-phase gas cells* (So = 0): primary variable, tracking the
    evolving dew-point as the condensate reservoir depletes.

    Set to all-zeros for standard black-oil.
    """

    gas_solubility_in_water: CellArray
    """
    Shape (n_cells,) - gas solubility in water (Rsw).

    Units: same as `solution_gor`.

    Non-negligible for CO₂ or sour-gas injection scenarios; zero for
    standard black-oil with no dissolved gas in the water phase.
    """

    water_bubble_point_pressure: CellArray
    """
    Shape (n_cells,) - bubble-point pressure of the water phase with respect
    to dissolved gas (Pbub,w).

    Units: same as `oil_bubble_point_pressure`.

    Tracks the pressure at which dissolved gas (Rsw) begins to exsolve from
    water.  Relevant for CO₂ or sour-gas reservoirs.  In the undersaturated
    water regime (no free gas exsolving from water), this is a primary
    variable analogous to `oil_bubble_point_pressure`.

    Set to all-zeros for standard black-oil with Rsw = 0.
    """

    oil_mass: CellArray
    """
    Shape (n_cells,) - oil-component mass in each cell.

    Units: lbm (FIELD), kg (METRIC / SI), g (LAB).

    Conserved quantity in the oil-component material balance.
    Tracks only the liquid oil phase; vaporized oil in gas is accumulated
    separately in `vaporized_oil_mass_in_gas`.
    """

    water_mass: CellArray
    """
    Shape (n_cells,) - water-component mass in each cell.

    Units: same as `oil_mass`.
    """

    free_gas_mass: CellArray
    """
    Shape (n_cells,) - free gas-component mass in each cell.

    Units: same as `oil_mass`.

    The total gas material balance is:
    free_gas_mass + dissolved_gas_mass_in_oil + dissolved_gas_mass_in_water.
    """

    dissolved_gas_mass_in_oil: CellArray
    """
    Shape (n_cells,) - gas dissolved in the oil phase.

    Units: same as `oil_mass`.

    Equal to Rs x oil_mass x (ρg_STC / ρo_STC) at standard conditions.
    Tracked separately so the gas material balance can be assembled without
    recomputing Rs at every cell.
    """

    dissolved_gas_mass_in_water: CellArray
    """
    Shape (n_cells,) - gas dissolved in the water phase.

    Units: same as `oil_mass`.

    Non-negligible for CO₂ injection or sour-gas reservoirs (Rsw > 0).
    Zero for standard black-oil.
    """

    vaporized_oil_mass_in_gas: CellArray
    """
    Shape (n_cells,) - oil (condensate) vaporized in the gas phase.

    Units: same as `oil_mass`.

    Equal to Rv x free_gas_mass x (ρo_STC / ρg_STC) at standard conditions.
    Part of the *oil-component* material balance, not the gas balance.
    Zero for standard dry-gas black-oil; required for volatile-oil and
    gas-condensate simulations.
    """

    solvent_concentration: CellArray = attrs.field(
        factory=lambda: np.zeros(0, dtype=get_dtype())
    )
    """
    Shape (n_cells,) - solvent volume fraction in the oil-phase mixture
    (dimensionless, [0, 1]).

    0 = pure oil; 1 = pure solvent.  Populated only for Todd-Longstaff or
    similar EOR miscibility models.  Defaults to an empty array for standard
    black-oil (zero memory cost).
    """

    hysteresis: typing.Optional[Hysteresis] = None
    """
    Optional `HysteresisState` for Killough scanning curves.
    `None` (default) for simulations without hysteresis.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities are expressed.

    Use `convert(target)` to produce a rescaled copy.
    """

    @property
    def total_gas_mass(self) -> CellArray:
        """
        Shape (n_cells,) - total gas-component mass per cell.

        m_g,total = free_gas_mass + dissolved_gas_mass_in_oil
                    + dissolved_gas_mass_in_water

        This is the conserved quantity in the gas-component material balance.
        Note that `vaporized_oil_mass_in_gas` belongs to the *oil* component
        balance, not the gas balance.
        """
        return typing.cast(
            CellArray,
            self.free_gas_mass
            + self.dissolved_gas_mass_in_oil
            + self.dissolved_gas_mass_in_water,
        )

    @property
    def total_oil_mass(self) -> CellArray:
        """
        Shape (n_cells,) - total oil-component mass per cell.

        m_o,total = oil_mass + vaporized_oil_mass_in_gas

        This is the conserved quantity in the oil-component material balance
        for volatile-oil / gas-condensate models.
        """
        return typing.cast(CellArray, self.oil_mass + self.vaporized_oil_mass_in_gas)

    def evolve(self, **kwargs: typing.Any) -> Self:
        """
        Return a new `State` with selected fields replaced.

        All fields not present in *kwargs* are carried forward unchanged.
        Preferred solver pattern:

        ```python
        new_state = state.evolve(
            pressure=new_p,
            oil_saturation=new_so,
            water_saturation=new_sw,
            gas_saturation=new_sg,
            oil_mass=new_mo,
            free_gas_mass=new_mg,
        )
        ```

        :param kwargs: Field names and their replacement values.
        :returns: New immutable `State`.
        :raises TypeError: If an unknown field name is passed.
        """
        return attrs.evolve(self, **kwargs)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `State` with all dimensional quantities rescaled
        to *target*.

        Dimensionless fields (saturations, `solvent_concentration`) are
        copied unchanged.  Rs, Rv, and Rsw are scaled by the GOR factor.
        Pressures use the pressure factor.  Masses use the combined density x
        length³ factor.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `State` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        mass_factor: float = factors["density"] * (factors["length"] ** 3)
        p_factor: float = factors["pressure"]
        gor_factor: float = factors["gor"]

        return self.__class__(
            pressure=_scale(self.pressure, p_factor),
            oil_saturation=self.oil_saturation,
            water_saturation=self.water_saturation,
            gas_saturation=self.gas_saturation,
            solution_gor=_scale(self.solution_gor, gor_factor),
            oil_bubble_point_pressure=_scale(self.oil_bubble_point_pressure, p_factor),
            vaporized_oil_ratio=_scale(self.vaporized_oil_ratio, gor_factor),
            gas_dew_point_pressure=_scale(self.gas_dew_point_pressure, p_factor),
            gas_solubility_in_water=_scale(self.gas_solubility_in_water, gor_factor),
            water_bubble_point_pressure=_scale(
                self.water_bubble_point_pressure, p_factor
            ),
            oil_mass=_scale(self.oil_mass, mass_factor),
            water_mass=_scale(self.water_mass, mass_factor),
            free_gas_mass=_scale(self.free_gas_mass, mass_factor),
            dissolved_gas_mass_in_oil=_scale(
                self.dissolved_gas_mass_in_oil, mass_factor
            ),
            dissolved_gas_mass_in_water=_scale(
                self.dissolved_gas_mass_in_water, mass_factor
            ),
            vaporized_oil_mass_in_gas=_scale(
                self.vaporized_oil_mass_in_gas, mass_factor
            ),
            solvent_concentration=self.solvent_concentration,
            hysteresis=self.hysteresis,
            unit_system=target,
        )


@attrs.frozen(slots=True)
class Meta(StoreSerializable):
    """
    Per-cell region assignments and simulation metadata.

    Populated from the REGIONS section of an Eclipse deck, or supplied
    directly by the user. All region arrays are 1-based integer indices
    selecting which PVT, saturation-function, equilibration, or rock
    compaction table applies to each cell.

    All fields are optional - when absent, region 1 is assumed for every
    cell (Eclipse default behaviour).
    """

    pvt_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - PVT region index per cell (1-based).
    Selects which PVTTables entry from PVTRegions applies.
    Read from PVTNUM. Default: 1 everywhere.
    """

    saturation_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - saturation function region index (1-based).
    Selects SWOF/SGOF/SWFN/SGFN table. Read from SATNUM.
    """

    imbibition_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - imbibition saturation function region index (1-based).
    Used for hysteresis scanning curves. Read from IMBNUM.
    """

    equilibration_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - equilibration region index (1-based).
    Selects which EQUIL record governs initialisation. Read from EQLNUM.
    """

    rock_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - rock compaction region index (1-based).
    Selects ROCK/ROCKTAB table. Read from ROCKNUM.
    """

    fluid_in_place_region: typing.Optional[IntCellArray] = None
    """
    Shape (n_cells,) - fluid-in-place reporting region (1-based).
    Controls which cells contribute to ROIP/RGIP/RWIP output groups.
    Read from FIPNUM.
    """

    @classmethod
    def from_deck_file(cls, data_file: DeckFile, n_cells: int) -> Self:
        """
        Build Meta from a parsed DeckFile.

        Missing keywords default to None (region 1 is assumed by callers).

        :param data_file: Parsed DeckFile.
        :param n_cells: Number of active cells, for validation.
        :returns: Meta.
        """

        def _load(keyword: str) -> typing.Optional[IntCellArray]:
            arr = data_file.get(keyword)
            if arr is None:
                return None

            arr = np.asarray(arr, dtype=np.int32)
            if arr.size != n_cells:
                raise ValidationError(
                    f"{keyword} has {arr.size} values; expected {n_cells}."
                )
            return typing.cast(IntCellArray, arr)

        return cls(
            pvt_region=_load("PVTNUM"),
            saturation_region=_load("SATNUM"),
            imbibition_region=_load("IMBNUM"),
            equilibration_region=_load("EQLNUM"),
            rock_region=_load("ROCKNUM"),
            fluid_in_place_region=_load("FIPNUM"),
        )

    def get_pvt_region(self, cell_index: int) -> int:
        """Return the PVT region for a cell, defaulting to 1."""
        if self.pvt_region is None:
            return 1
        return int(self.pvt_region[cell_index])

    def get_saturation_region(self, cell_index: int) -> int:
        """Return the saturation function region for a cell, defaulting to 1."""
        if self.saturation_region is None:
            return 1
        return int(self.saturation_region[cell_index])

    def get_imbibition_region(self, cell_index: int) -> int:
        """Return the imbibition region for a cell, defaulting to 1."""
        if self.imbibition_region is None:
            return 1
        return int(self.imbibition_region[cell_index])

    def get_equilibration_region(self, cell_index: int) -> int:
        """Return the equilibration region for a cell, defaulting to 1."""
        if self.equilibration_region is None:
            return 1
        return int(self.equilibration_region[cell_index])

    def get_fluid_in_place_region(self, cell_index: int) -> int:
        """Return the fluid-in-place region for a cell, defaulting to 1."""
        if self.fluid_in_place_region is None:
            return 1
        return int(self.fluid_in_place_region[cell_index])

    def get_rock_region(self, cell_index: int) -> int:
        """Return the rock function region for a cell, defaulting to 1."""
        if self.rock_region is None:
            return 1
        return int(self.rock_region[cell_index])
