""" "Per-cell" properties definitions for a black-oil reservoir model."""

import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypedDict

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.precision import get_dtype
from bores.stores import StoreSerializable
from bores.typing import BooleanCellArray, CellArray, UnitSystem

__all__ = [
    "FluidProperties",
    "HysteresisState",
    "RockPermeability",
    "RockProperties",
    "ReservoirState",
]


def _scale(arr: CellArray, factor: float) -> CellArray:
    """Return `arr * factor` as the same dtype; identity when factor == 1.0."""
    if factor == 1.0:
        return arr
    return (arr * factor).astype(arr.dtype)


def _scale_non_empty(arr: CellArray, fac: float) -> CellArray:
    """Scale only if the optional EOR array is non-empty."""
    return _scale(arr, fac) if arr.size > 0 else arr


def _scale_and_offset(arr: CellArray, scale: float, offset: float) -> CellArray:
    """Return `arr * scale + offset` as the same dtype; identity when trivial."""
    if scale == 1.0 and offset == 0.0:
        return arr
    return ((arr * scale) + offset).astype(arr.dtype)


@attrs.frozen(slots=True)
class RockPermeability(StoreSerializable):
    """
    Absolute permeability tensor stored as three orthogonal components.

    If only `x` is supplied, the y and z components default to `x`
    (isotropic assumption).  The geometric-mean `mean` is computed
    automatically when not provided.

    Units follow the parent `RockProperties.unit_system`.
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

    def __attrs_post_init__(self) -> None:
        if self.y.size == 0:
            object.__setattr__(self, "y", self.x)
        if self.z.size == 0:
            object.__setattr__(self, "z", self.x)
        if self.mean.size == 0:
            if np.array_equal(self.x, self.y) and np.array_equal(self.x, self.z):
                object.__setattr__(self, "mean", self.x)
            else:
                geom = (self.x * self.y * self.z) ** (1.0 / 3.0)
                object.__setattr__(self, "mean", geom)

    def scale(self, factor: float) -> Self:
        """Return a new instance with all components multiplied by factor."""
        if factor == 1.0:
            return self
        return self.__class__(
            x=_scale(self.x, factor),
            y=_scale(self.y, factor),
            z=_scale(self.z, factor),
            mean=_scale(self.mean, factor),
        )


@attrs.frozen(slots=True)
class RockProperties(StoreSerializable):
    """
    Static petrophysical properties of the reservoir rock.

    These arrays are constant between simulation time steps.  They are
    populated from GRDECL keywords: PORO, PERMX/Y/Z, NTG,
    SWCON, SWCRIT, etc.

    All saturation arrays are dimensionless fractions in [0, 1].
    Use convert(target) to rescale dimensional quantities to another
    unit system.
    """

    porosity: CellArray
    """
    Shape (n_cells,) - pore volume fraction (dimensionless, [0, 1]).

    Used to compute pore volume: PV = φ x NTG x Vcell.
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

    rock_compressibility: CellArray
    """
    Shape (n_cells,) - formation compressibility.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Used in the pore-volume accumulation term: dPV/dP = PV · cr.
    """

    connate_water_saturation: CellArray
    """
    Shape (n_cells,) - connate (initial irreducible) water saturation
    (fraction).

    Lower bound on water saturation; set at geological initial conditions.
    """

    irreducible_water_saturation: CellArray
    """
    Shape (n_cells,) - irreducible water saturation during imbibition
    (fraction).

    Equal to or greater than connate_water_saturation.
    """

    residual_oil_saturation_water_flood: CellArray
    """
    Shape (n_cells,) - residual oil saturation at end of water flooding
    (Sor,w, fraction).

    Oil is immobile below this saturation during water-flood imbibition.
    """

    residual_oil_saturation_gas_flood: CellArray
    """
    Shape (n_cells,) - residual oil saturation at end of gas flooding
    (Sor,g, fraction).

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

    Dimensionless arrays (porosities, saturations, NTG) are unaffected by
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
        Return a new `RockProperties` with all dimensional quantities rescaled
        to target.

        Dimensionless arrays (porosity, NTG, saturations) are copied unchanged.
        Conversion factors are sourced from `get_conversion_factors`.

        :param target: Desired `UnitSystem`.
        :returns: New `RockProperties` in target units.
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
            rock_compressibility=self.rock_compressibility * factors["compressibility"],
            connate_water_saturation=self.connate_water_saturation,
            irreducible_water_saturation=self.irreducible_water_saturation,
            residual_oil_saturation_water_flood=self.residual_oil_saturation_water_flood,
            residual_oil_saturation_gas_flood=self.residual_oil_saturation_gas_flood,
            residual_gas_saturation=self.residual_gas_saturation,
            unit_system=target,
        )


@attrs.frozen(slots=True)
class FluidProperties(StoreSerializable):
    """
    Static PVT reference characterisation of the reservoir fluids.

    Stores the per-fluid scalars that are read once from the PVT deck and do
    not change between time steps.  Quantities that *do* vary with pressure
    (FVFs, viscosities, densities, solution GOR, Rv, bubble-point,
    dew-point, compressibilities) live in `ReservoirState`.

    The split reflects the physical distinction between what fluids *are*
    (static characterisation, stored here) and what the current reservoir
    *condition* looks like (dynamic state, stored in `ReservoirState`).

    Use convert(target) to rescale dimensional quantities to another
    unit system.
    """

    # Oil

    oil_specific_gravity: float
    """
    Oil specific gravity relative to fresh water at 60 °F (dimensionless).

    Constant for a given crude; typically 0.75-0.95.  Used to derive
    stock-tank oil density: ρ_o,STC = oil_specific_gravity x ρ_water_STC.
    """

    oil_api_gravity: float
    """
    Oil API gravity (°API), computed as 141.5 / SG - 131.5.

    Provided for convenience; redundant with oil_specific_gravity.
    """

    oil_reference_fvf: float
    """
    Oil formation volume factor at bubble-point (reference) pressure.

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Used to initialise ReservoirState.oil_fvf.
    """

    oil_reference_viscosity: float
    """
    Dead-oil viscosity at standard conditions.

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Serves as the correlation reference; live-oil viscosity at reservoir
    conditions is stored in `ReservoirState`.
    """

    oil_reference_compressibility: float
    """
    Oil compressibility at bubble-point (reference) pressure.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Used for the undersaturated-oil compressibility term above bubble point.
    """

    # Water

    water_salinity: float
    """
    Formation water salinity (ppm NaCl).

    Assumed spatially and temporally constant.  Used in brine density and
    viscosity correlations (e.g. Batzle-Wang).  Typical seawater: 35 000 ppm.
    """

    water_reference_fvf: float
    """
    Water formation volume factor at reference pressure.

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Approximately 1.00-1.08 depending on salinity and temperature.
    """

    water_reference_viscosity: float
    """
    Water viscosity at reference conditions.

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Approximately 0.3-1.0 cP at reservoir temperature.
    """

    water_reference_compressibility: float
    """
    Water compressibility at reference pressure.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Typically 3-5 x 10⁻⁶ psi⁻¹.
    """

    # Gas

    reservoir_gas: str
    """
    Name or identifier of the reservoir (or injected) gas
    (e.g. "methane", "CO2").

    Used to select z-factor correlations and for documentation.
    """

    gas_gravity: float
    """
    Gas specific gravity relative to air (dimensionless).

    0.556 for pure methane; up to ~0.9 for rich condensate gas.  Input to
    pseudo-critical property correlations (Sutton, Pitzer) for z-factor and
    viscosity.
    """

    gas_molecular_weight: float
    """
    Gas molecular weight (g/mol).

    Methane: 16.04 g/mol; CO₂: 44.01 g/mol.  Should satisfy
    MW ≈ 28.97 x gas_gravity.
    """

    gas_reference_viscosity: float
    """
    Gas viscosity at reference conditions.

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Typically 0.01-0.03 cP at reservoir temperature.
    """

    # Miscible / solvent

    miscibility_model: str = "immiscible"
    """
    Miscibility model identifier.

    "immiscible" for standard black-oil; "todd-longstaff" for first-
    contact miscible EOR.  Controls how solvent concentration in the oil
    phase is handled.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities are expressed.

    Dimensionless fields (specific gravity, API, gas gravity, molecular
    weight, miscibility model, reservoir_gas name) are unaffected by unit
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
        Return a new `FluidProperties` with dimensional quantities rescaled to
        target.

        Dimensionless fields (gravities, API, molecular weight,
        miscibility_model, reservoir_gas) are copied unchanged.

        :param target: Desired `UnitSystem`.
        :returns: New `FluidProperties` in target units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            oil_specific_gravity=self.oil_specific_gravity,
            oil_api_gravity=self.oil_api_gravity,
            oil_reference_fvf=self.oil_reference_fvf * factors["liquid_fvf"],
            oil_reference_viscosity=self.oil_reference_viscosity * factors["viscosity"],
            oil_reference_compressibility=(
                self.oil_reference_compressibility * factors["compressibility"]
            ),
            water_salinity=self.water_salinity,  # ppm is dimensionless
            water_reference_fvf=self.water_reference_fvf * factors["liquid_fvf"],
            water_reference_viscosity=self.water_reference_viscosity
            * factors["viscosity"],
            water_reference_compressibility=(
                self.water_reference_compressibility * factors["compressibility"]
            ),
            reservoir_gas=self.reservoir_gas,
            gas_gravity=self.gas_gravity,
            gas_molecular_weight=self.gas_molecular_weight,
            gas_reference_viscosity=self.gas_reference_viscosity * factors["viscosity"],
            miscibility_model=self.miscibility_model,
            unit_system=target,
        )


@attrs.frozen(slots=True)
class ReservoirState(StoreSerializable):
    """
    Dynamic per-cell simulation state, updated at every time step.

    This is the only property group that the solver mutates.  It holds:

    - Thermodynamic state (pressure, temperature).
    - Phase saturations (oil, water, free gas).
    - Component masses - the IMPES conserved quantities:

      - oil_mass: oil component (liquid phase).
      - water_mass: water component.
      - free_gas_mass: gas component in the free-gas phase.
      - dissolved_gas_mass_in_oil: gas dissolved in oil (Rs contribution).
      - dissolved_gas_mass_in_water: gas dissolved in water (Rsw).
      - vaporized_oil_mass_in_gas: oil vaporized in gas (Rv contribution).

    - Pressure-dependent PVT quantities updated each Newton iteration:
      FVFs, viscosities, densities, Rs, Rv, Rsw, bubble-point pressure,
      dew-point pressure, z-factor, and per-phase compressibilities.

    All shape-(n_cells,) arrays are indexed in the same order as
    Grid.cell_centroids.

    Use convert(target) to rescale to another unit system, or
    evolve(**kwargs) to produce a new state with selected fields replaced.
    """

    # Thermodynamic state

    pressure: CellArray
    """
    Shape (n_cells,) - oil-phase (reference) pressure.

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    Primary implicit unknown in the IMPES pressure equation.
    Phase pressures for water and gas are recovered via capillary pressure:
    Pw = Po - Pcow, Pg = Po + Pcgo.
    """

    temperature: CellArray
    """
    Shape (n_cells,) - reservoir temperature.

    Units: °F (FIELD), °C (METRIC / LAB), K (SI).
    Constant for isothermal simulation; spatially varying for thermal models.
    """

    # Saturations

    oil_saturation: CellArray
    """
    Shape (n_cells,) - oil-phase saturation (fraction, [0, 1]).

    Updated explicitly in IMPES.
    Must satisfy So + Sw + Sg = 1 in every cell.
    """

    water_saturation: CellArray
    """Shape (n_cells,) - water-phase saturation (fraction, [0, 1])."""

    gas_saturation: CellArray
    """
    Shape (n_cells,) - free gas-phase saturation (fraction, [0, 1]).

    Includes only free (non-dissolved, non-vaporized) gas.
    """

    # Component masses

    oil_mass: CellArray
    """
    Shape (n_cells,) - oil-component mass in each cell.

    Units: lbm (FIELD), kg (METRIC / SI), g (LAB).
    Conserved quantity in the oil component material balance.
    Includes only the liquid oil; vaporized oil in gas is tracked separately
    in vaporized_oil_mass_in_gas.
    """

    water_mass: CellArray
    """
    Shape (n_cells,) - water-component mass in each cell.

    Units: same as oil_mass.
    """

    free_gas_mass: CellArray
    """
    Shape (n_cells,) - free gas-component mass in each cell.

    Units: same as oil_mass.
    The total gas material balance is
    free_gas_mass + dissolved_gas_mass_in_oil + dissolved_gas_mass_in_water.
    """

    dissolved_gas_mass_in_oil: CellArray
    """
    Shape (n_cells,) - gas dissolved in the oil phase (lbm / kg / g).

    Equal to Rs x oil_mass x (ρ_g,STC / ρ_o,STC) at standard conditions.
    Tracked separately so the gas material balance can be assembled without
    recomputing Rs at every cell.
    """

    dissolved_gas_mass_in_water: CellArray
    """
    Shape (n_cells,) - gas dissolved in the water phase (lbm / kg / g).

    Non-negligible for CO₂ injection or sour-gas reservoirs (Rsw > 0).
    """

    vaporized_oil_mass_in_gas: CellArray
    """
    Shape (n_cells,) - oil (condensate) vaporized in the gas phase
    (lbm / kg / g).

    Equal to Rv x free_gas_mass x (ρ_o,STC / ρ_g,STC) at standard
    conditions.  Part of the *oil* component material balance, not the gas
    balance.  Zero for standard dry-gas black-oil models; required for
    volatile-oil and gas-condensate simulations.
    """

    # Pressure-dependent PVT (updated each Newton iteration)

    oil_fvf: CellArray
    """
    Shape (n_cells,) - oil formation volume factor at reservoir pressure.

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Interpolated from the Bo-pressure PVT table.
    """

    water_fvf: CellArray
    """
    Shape (n_cells,) - water formation volume factor at reservoir pressure.

    Units: same as oil_fvf.  Typically close to 1.0.
    """

    gas_fvf: CellArray
    """
    Shape (n_cells,) - gas formation volume factor at reservoir pressure.

    Units: ft³/scf (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Derived from the real-gas law: Bg = (z·T/P) x (P_STC/T_STC).
    """

    oil_viscosity: CellArray
    """
    Shape (n_cells,) - live-oil viscosity at reservoir conditions.

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    """

    water_viscosity: CellArray
    """
    Shape (n_cells,) - water viscosity at reservoir conditions.

    Units: same as oil_viscosity.
    """

    gas_viscosity: CellArray
    """
    Shape (n_cells,) - free-gas viscosity at reservoir conditions.

    Units: same as oil_viscosity.  Typically 0.01-0.05 cP.
    """

    oil_density: CellArray
    """
    Shape (n_cells,) - live-oil density at reservoir conditions.

    Units: lbm/ft³ (FIELD), kg/m³ (METRIC / SI), g/cm³ (LAB).
    Computed from stock-tank density and FVF:
    ρo_res = (ρo_STC + Rs·ρg_STC) / Bo.
    """

    water_density: CellArray
    """
    Shape (n_cells,) - water density at reservoir conditions.

    Units: same as oil_density.
    """

    gas_density: CellArray
    """
    Shape (n_cells,) - free-gas density at reservoir conditions.

    Units: same as oil_density.
    """

    solution_gor: CellArray
    """
    Shape (n_cells,) - solution gas-to-oil ratio (Rs).

    Units: scf/STB (FIELD), sm³/sm³ (METRIC / SI), scc/scc (LAB).
    Gas dissolved in oil per unit stock-tank oil volume at current pressure.
    Capped at the bubble-point value when the cell is above bubble point.
    """

    vaporized_oil_ratio: CellArray
    """
    Shape (n_cells,) - vaporized oil ratio (Rv).

    Units: STB/Mscf (FIELD), sm³/sm³ (METRIC / SI), scc/scc (LAB).
    Stock-tank oil vaporized in gas per unit standard gas volume.
    Non-zero only for volatile-oil and gas-condensate reservoirs; set to
    all-zeros for standard black-oil.
    """

    gas_solubility_in_water: CellArray
    """
    Shape (n_cells,) - gas solubility in water (Rsw).

    Units: same as solution_gor.
    Non-negligible for CO₂ or sour-gas injection scenarios.
    """

    oil_bubble_point_pressure: CellArray
    """
    Shape (n_cells,) - bubble-point pressure of the oil phase.

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    Used to select between saturated and undersaturated oil PVT branches.
    """

    gas_dew_point_pressure: CellArray
    """
    Shape (n_cells,) - dew-point pressure of the gas phase.

    Units: same as oil_bubble_point_pressure.
    Used in volatile-oil / gas-condensate models to select the gas PVT
    branch.  Set to all-zeros for standard black-oil.
    """

    gas_compressibility_factor: CellArray
    """
    Shape (n_cells,) - real-gas z-factor (dimensionless).

    Interpolated from the z-pressure table or computed via a correlation
    (e.g. Pitzer-Curl, Hall-Yarborough).
    Used in gas FVF: Bg = z·T/P x P_STC/T_STC.
    """

    oil_compressibility: CellArray
    """
    Shape (n_cells,) - oil compressibility at current pressure.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Pressure-dependent; used in the undersaturated-oil accumulation term.
    """

    water_compressibility: CellArray
    """
    Shape (n_cells,) - water compressibility at current pressure.

    Units: same as oil_compressibility.
    Approximately 3-5 x 10⁻⁶ psi⁻¹ at reservoir conditions.
    """

    gas_compressibility: CellArray
    """
    Shape (n_cells,) - gas compressibility at current pressure.

    Units: same as oil_compressibility.
    For real gas: cg = 1/P - (1/z)(dz/dP).
    """

    # Miscible / solvent (EOR)

    solvent_concentration: CellArray = attrs.field(
        factory=lambda: np.zeros(0, dtype=get_dtype())
    )
    """
    Shape (n_cells,) - solvent volume fraction in the oil-phase mixture
    (dimensionless, [0, 1]).

    0 = pure oil; 1 = pure solvent.  Populated only for Todd-Longstaff
    or similar EOR miscibility models.  Defaults to empty (i.e. no solvent)
    for standard black-oil.
    """

    oil_effective_viscosity: CellArray = attrs.field(
        factory=lambda: np.zeros(0, dtype=get_dtype())
    )
    """
    Shape (n_cells,) - effective oil-solvent mixture viscosity.

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Computed via the Todd-Longstaff mixing rule when solvent_concentration
    is non-zero; equals oil_viscosity for immiscible flow.
    Empty for standard black-oil runs.
    """

    oil_effective_density: CellArray = attrs.field(
        factory=lambda: np.zeros(0, dtype=get_dtype())
    )
    """
    Shape (n_cells,) - effective oil-solvent mixture density.

    Units: same as oil_density.  Empty for standard black-oil runs.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities are expressed.

    Use convert(target) to produce a rescaled copy.
    """

    @property
    def total_gas_mass(self) -> CellArray:
        """
        Shape (n_cells,) - total gas-component mass per cell.

        m_g = free_gas_mass + dissolved_gas_mass_in_oil
              + dissolved_gas_mass_in_water

        This is the conserved quantity in the gas component material balance.
        Note: vaporized_oil_mass_in_gas is part of the *oil* component
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

        m_o = oil_mass + vaporized_oil_mass_in_gas

        The conserved quantity in the oil component material balance for
        volatile-oil / gas-condensate models.
        """
        return typing.cast(CellArray, self.oil_mass + self.vaporized_oil_mass_in_gas)

    def evolve(self, **kwargs: typing.Any) -> Self:
        """
        Return a new `ReservoirState` with selected fields replaced.

        All fields not present in kwargs are carried forward from self
        unchanged. Preferred solver pattern:

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

        :param kwargs: Field names and replacement values.
        :returns: New immutable `ReservoirState`.
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
        Return a new `ReservoirState` with all dimensional quantities rescaled
        to target.

        Dimensionless fields (saturations, z-factor, solvent concentration)
        are copied unchanged.

        Mass unit: derived as density_factor x length_factor³, e.g.
        lbm -> kg uses 16.0185 x 0.3048³ ≈ 0.4536.

        Temperature uses the affine map from `get_conversion_factors`:
        T_to = T_from * scale + offset.

        :param target: Desired `UnitSystem`.
        :returns: New `ReservoirState` in target units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        mass_factor = factors["density"] * (factors["length"] ** 3)
        return self.__class__(
            pressure=_scale(self.pressure, factors["pressure"]),
            temperature=_scale_and_offset(
                self.temperature,
                factors["temperature_scale"],
                factors["temperature_offset"],
            ),
            oil_saturation=self.oil_saturation,
            water_saturation=self.water_saturation,
            gas_saturation=self.gas_saturation,
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
            oil_fvf=_scale(self.oil_fvf, factors["liquid_fvf"]),
            water_fvf=_scale(self.water_fvf, factors["liquid_fvf"]),
            gas_fvf=_scale(self.gas_fvf, factors["gaseous_fvf"]),
            oil_viscosity=_scale(self.oil_viscosity, factors["viscosity"]),
            water_viscosity=_scale(self.water_viscosity, factors["viscosity"]),
            gas_viscosity=_scale(self.gas_viscosity, factors["viscosity"]),
            oil_density=_scale(self.oil_density, factors["density"]),
            water_density=_scale(self.water_density, factors["density"]),
            gas_density=_scale(self.gas_density, factors["density"]),
            solution_gor=_scale(self.solution_gor, factors["gor"]),
            vaporized_oil_ratio=_scale(self.vaporized_oil_ratio, factors["gor"]),
            gas_solubility_in_water=_scale(
                self.gas_solubility_in_water, factors["gor"]
            ),
            oil_bubble_point_pressure=_scale(
                self.oil_bubble_point_pressure, factors["pressure"]
            ),
            gas_dew_point_pressure=_scale(
                self.gas_dew_point_pressure, factors["pressure"]
            ),
            gas_compressibility_factor=self.gas_compressibility_factor,
            oil_compressibility=_scale(
                self.oil_compressibility, factors["compressibility"]
            ),
            water_compressibility=_scale(
                self.water_compressibility, factors["compressibility"]
            ),
            gas_compressibility=_scale(
                self.gas_compressibility, factors["compressibility"]
            ),
            solvent_concentration=self.solvent_concentration,
            oil_effective_viscosity=_scale_non_empty(
                self.oil_effective_viscosity, factors["viscosity"]
            ),
            oil_effective_density=_scale_non_empty(
                self.oil_effective_density, factors["density"]
            ),
            unit_system=target,
        )


@attrs.frozen(slots=True)
class HysteresisState(StoreSerializable):
    """
    Drainage / imbibition hysteresis tracking for Killough scanning curves.

    Maintains historical saturation extrema and displacement-regime flags
    required to compute effective residual saturations on the scanning curves.
    These are consumed inside relative permeability and capillary pressure
    evaluation routines - the flow solver does not interpret them directly.

    Kept separate from `ReservoirState` so that simulations without hysteresis
    pay zero memory cost.  All arrays are dimensionless (saturations, flags)
    and therefore require no unit conversion.
    """

    max_water_saturation: CellArray
    """
    Shape (n_cells,) - historical maximum water saturation reached in
    each cell (fraction).

    Initialised to the initial water saturation.  Updated whenever the
    current water saturation exceeds the stored maximum.  Determines the
    imbibition end-point on the scanning curve when drainage reverses.
    """

    max_gas_saturation: CellArray
    """
    Shape (n_cells,) - historical maximum gas saturation reached in
    each cell (fraction).

    Analogous to max_water_saturation for the gas phase.
    """

    water_imbibition_flag: BooleanCellArray
    """
    Shape (n_cells,) - True if the current water-phase displacement is
    imbibition (water saturation increasing toward max_water_saturation).

    False indicates drainage (water saturation decreasing).
    """

    gas_imbibition_flag: BooleanCellArray
    """
    Shape (n_cells,) - True if the current gas-phase displacement is
    imbibition (gas saturation decreasing, i.e. water or liquid displacing gas).
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

    Analogous to water_reversal_saturation for the gas phase.
    """

    @classmethod
    def from_initial_saturations(
        cls, water_saturation: npt.ArrayLike, gas_saturation: npt.ArrayLike
    ) -> Self:
        """
        Construct a `HysteresisState` from initial saturation arrays.

        Sets maximum saturations to the initial values, marks all cells as
        drainage (not yet reversing), and places reversal points at the
        initial saturation values.

        :param water_saturation: Array-like (n_cells,) - initial water
            saturation per cell (fraction).
        :param gas_saturation: Array-like (n_cells,) - initial gas
            saturation per cell (fraction).
        :returns: Initialised `HysteresisState`.
        """
        sw = np.asarray(water_saturation, dtype=get_dtype())
        sg = np.asarray(gas_saturation, dtype=get_dtype())
        return cls(
            max_water_saturation=sw.copy(),
            max_gas_saturation=sg.copy(),
            water_imbibition_flag=np.zeros(sw.shape, dtype=np.bool_),  # type: ignore
            gas_imbibition_flag=np.zeros(sg.shape, dtype=np.bool_),  # type: ignore
            water_reversal_saturation=sw.copy(),
            gas_reversal_saturation=sg.copy(),
        )

    def evolve(self, **kwargs: typing.Any) -> Self:
        """
        Return a new `HysteresisState` with selected fields replaced.

        :param kwargs: Field names and replacement values.
        :returns: New immutable `HysteresisState`.
        """
        return attrs.evolve(self, **kwargs)
