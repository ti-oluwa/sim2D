import logging
import typing
import warnings
from os import PathLike

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.stores import StoreSerializable
from bores.typing import (
    FluidPhase,
    NumberArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)
from bores.utils import scale

logger = logging.getLogger(__name__)

__all__ = ["PVTData", "PVTDataSet"]


@attrs.frozen(slots=True)
class PVTData(StoreSerializable):
    """
    Raw PVT table data for a single fluid phase.

    Phase-tagged container for tabulated fluid properties. All table arrays
    are optional - only `phase`, `pressures`, and `temperatures` are
    required. `PVTTable` validates which fields are meaningful for the
    phase at initialisation time.

    **Array shapes**

    - Oil / Gas: 2-D arrays with shape `(n_pressures, n_temperatures)`.
    - Water: 3-D arrays with shape `(n_pressures, n_temperatures, n_salinities)`.
    - For wet-gas (PVTG) tables: `(n_pressures, n_rv)` where the Rv axis
      replaces the temperature axis (`temperatures` carries the Rv values in
      that case - see `PVTTable.__init__` notes).

    **Primary (interpolated) properties**

    These are the only quantities read directly from Eclipse deck keywords:

    - Oil: `formation_volume_factor_table` (Bo), `viscosity_table` (μo),
      `solution_gor_table` (Rs).
    - Gas: `formation_volume_factor_table` (Bg), `viscosity_table` (μg),
      `compressibility_factor_table` (z), `vaporized_oil_ratio_table` (Rv,
      wet-gas / condensate only).
    - Water: scalars on `PVT` - no table needed.

    **Derived (pre-built) properties**

    These are built from primary quantities plus stock-tank reference densities
    at `PVTTable` construction time. They are stored here as optional arrays
    so they can be serialised and reloaded without rebuilding:

    - `density_table` - ρ(P, T) in lbm/ft³.
    - `compressibility_table` - c(P, T) in psi⁻¹.

    Oil-specific fields: `bubble_point_pressures`, `solution_gas_to_oil_ratios`,
    `solution_gor_table`.

    Gas-specific fields: `compressibility_factor_table`,
    `solubility_in_water_table`, `vaporized_oil_ratio_table`,
    `dew_point_pressures`.

    Water-specific fields: `salinities`, `bubble_point_pressure_table`,
    `gas_free_water_fvf_table`.
    """

    phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """Fluid phase this data describes."""

    # Coordinate grids
    pressures: NumberArray[OneDimension]
    """1-D array of pressures (psi), strictly increasing."""

    temperatures: NumberArray[OneDimension]
    """
    1-D array of temperatures (°F), strictly increasing.

    For wet-gas (PVTG) tables this axis carries Rv values instead of
    temperatures; the `pvtg` flag on `PVTTable` signals this.
    """

    # Water-only coordinate
    salinities: typing.Optional[NumberArray[OneDimension]] = None
    """1-D array of salinities (ppm NaCl), strictly increasing. Water phase only."""

    # Oil-only coordinates / meta
    bubble_point_pressures: typing.Optional[
        typing.Union[NumberArray[OneDimension], NumberArray[TwoDimensions]]
    ] = None
    """
    Bubble-point pressures (psi).  Oil phase only.

    - 1-D shape `(n_t,)`      -> Pb(T).
    - 2-D shape `(n_rs, n_t)` -> Pb(Rs, T); requires `solution_gas_to_oil_ratios`.
    """

    solution_gas_to_oil_ratios: typing.Optional[NumberArray[OneDimension]] = None
    """
    1-D array of Rs values (scf/STB) for the first axis of a 2-D
    `bubble_point_pressures` table. Required when `bubble_point_pressures`
    is 2-D. Oil phase only.
    """

    # Gas-only: dew point and Rv
    dew_point_pressures: typing.Optional[NumberArray[OneDimension]] = None
    """
    Dew-point pressures Pdew(T) (psi). Gas / condensate phase only.
    Shape `(n_t,)`.
    """

    vaporized_oil_ratio_table: typing.Optional[NumberArray[TwoDimensions]] = None
    """
    Vaporised oil ratio Rv(P, T) in STB/Mscf. Gas / condensate phase only.
    Shape `(n_p, n_t)`. Rv is capped at Rv_sat above dew point (analogous
    to Rs being capped at Rsb above bubble point for oil).
    """

    # Shared primary tables (2-D for oil/gas; 3-D for water)
    viscosity_table: typing.Optional[
        typing.Union[NumberArray[TwoDimensions], NumberArray[ThreeDimensions]]
    ] = None
    """Viscosity μ(P, T) in cP. 2-D for oil/gas, 3-D for water."""

    formation_volume_factor_table: typing.Optional[
        typing.Union[NumberArray[TwoDimensions], NumberArray[ThreeDimensions]]
    ] = None
    """
    Formation volume factor B(P, T).

    Units: bbl/STB (oil / water), ft³/scf (gas). 2-D for oil/gas, 3-D for water.
    """

    # Shared derived tables (optional; built at PVTTable construction when absent)
    density_table: typing.Optional[
        typing.Union[NumberArray[TwoDimensions], NumberArray[ThreeDimensions]]
    ] = None
    """
    Density ρ(P, T) in lbm/ft³.  2-D for oil/gas, 3-D for water.

    Derived from FVF and stock-tank reference densities:

    - Oil:  ρo = (ρo,SC + Rs·ρg,SC) / Bo
    - Gas:  ρg = (ρg,SC + Rv·ρo,SC) / Bg   [wet gas]
            ρg = ρg,SC / Bg                  [dry gas]
    - Water: ρw = ρw,SC / Bw

    Set automatically by `PVTTable` if absent and reference densities are
    provided.
    """

    compressibility_table: typing.Optional[
        typing.Union[NumberArray[TwoDimensions], NumberArray[ThreeDimensions]]
    ] = None
    """
    Compressibility c(P, T) in psi⁻¹.  2-D for oil/gas, 3-D for water.

    Derived from the pressure-derivative of FVF:

    - Oil / water: c = -(1/B) · (∂B/∂P)
    - Gas:         cg = 1/P - (1/z) · (∂z/∂P)

    Set automatically by `PVTTable` if absent and the FVF table is present.
    """

    # Oil-only primary
    solution_gor_table: typing.Optional[NumberArray[TwoDimensions]] = None
    """Solution GOR Rs(P, T) in scf/STB. Oil phase only."""

    # Gas-only primary
    compressibility_factor_table: typing.Optional[NumberArray[TwoDimensions]] = None
    """Z-factor z(P, T), dimensionless. Gas phase only."""

    solubility_in_water_table: typing.Optional[NumberArray[ThreeDimensions]] = None
    """
    Gas solubility in water Rsw(P, T, S) in scf/STB.
    Gas phase only. 3-D shape `(n_p, n_t, n_s)`; requires `salinities`.
    """

    # Water-only primary
    bubble_point_pressure_table: typing.Optional[NumberArray[ThreeDimensions]] = None
    """Water bubble-point pressure Pbw(P, T, S) in psi. Water phase only."""

    gas_free_water_fvf_table: typing.Optional[NumberArray[TwoDimensions]] = None
    """
    Gas-free water FVF Bw_gf(P, T) in bbl/STB. Water phase only.

    Used internally to compute `density_table` and `compressibility_table`
    for the water phase; not exposed as a direct query method on `PVTTable`.
    """
    dtype: npt.DTypeLike = None

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """
    Unit system in which all dimensional quantities in this data are expressed.

    Pressure axes and density values in FIELD are in psi and lbm/ft³;
    in METRIC they are bar and kg/m³; in LAB they are atm and g/cm³.
    """

    def __attrs_post_init__(self) -> None:
        self._warn_phase_mismatches()
        self.ensure_dtype(self.dtype, force=True)

    def _warn_phase_mismatches(self) -> None:
        phase = typing.cast(FluidPhase, self.phase)
        if phase == FluidPhase.GAS and self.solution_gor_table is not None:
            warnings.warn(
                f"{type(self).__name__}: `solution_gor_table` is oil-only and will "
                "be ignored for GAS phase.",
                UserWarning,
                stacklevel=3,
            )
        if phase == FluidPhase.OIL and self.compressibility_factor_table is not None:
            warnings.warn(
                f"{type(self).__name__}: `compressibility_factor_table` is gas-only "
                "and will be ignored for OIL phase.",
                UserWarning,
                stacklevel=3,
            )
        if phase == FluidPhase.WATER and self.bubble_point_pressures is not None:
            warnings.warn(
                f"{type(self).__name__}: `bubble_point_pressures` is oil-only. For "
                "water bubble point use `bubble_point_pressure_table` (3-D).",
                UserWarning,
                stacklevel=3,
            )
        if (
            self.bubble_point_pressures is not None
            and isinstance(self.bubble_point_pressures, np.ndarray)
            and self.bubble_point_pressures.ndim == 2
            and self.solution_gas_to_oil_ratios is None
        ):
            raise ValidationError(
                f"{type(self).__name__}: 2-D `bubble_point_pressures` requires "
                "`solution_gas_to_oil_ratios` to be provided."
            )

    def ensure_dtype(self, dtype: npt.DTypeLike = None, force: bool = True) -> None:
        if not force and self.dtype is not None and self.dtype == np.dtype(dtype):
            return

        dtype = np.dtype(dtype if dtype is not None else get_dtype())
        for field in attrs.fields(type(self)):
            value = getattr(self, field.name)
            if (
                value is not None
                and isinstance(value, np.ndarray)
                and value.dtype != dtype
            ):
                object.__setattr__(self, field.name, value.astype(dtype, copy=False))

        if self.dtype != dtype:
            object.__setattr__(self, "dtype", dtype if dtype is not None else None)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `PVTData` with all dimensional quantities rescaled to *target*.

        Dimensionless properties (specific gravity, compressibility factor,
        vaporized oil ratio, solution GOR ratio) and multiplier-type quantities
        are copied unchanged. Pressure axes, densities, FVFs, and viscosities
        are rescaled using appropriate factors.

        :param target: Target `UnitSystem`.
        :returns `PVTData`: New `PVTData` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        density_factor = factors["density"]
        viscosity_factor = factors["viscosity"]
        liquid_fvf_factor = factors["liquid_fvf"]
        gas_fvf_factor = factors["gas_fvf"]
        fvf_factor = (
            gas_fvf_factor if self.phase == FluidPhase.GAS else liquid_fvf_factor
        )
        # Compressibility is 1/pressure
        compressibility_factor = 1.0 / pressure_factor
        return attrs.evolve(
            self,
            pressures=scale(self.pressures, pressure_factor),
            bubble_point_pressures=scale(self.bubble_point_pressures, pressure_factor),
            dew_point_pressures=scale(self.dew_point_pressures, pressure_factor),
            formation_volume_factor_table=scale(
                self.formation_volume_factor_table, fvf_factor
            ),
            viscosity_table=scale(self.viscosity_table, viscosity_factor),
            density_table=scale(self.density_table, density_factor),
            compressibility_table=scale(
                self.compressibility_table, compressibility_factor
            ),
            gas_free_water_fvf_table=scale(
                self.gas_free_water_fvf_table, liquid_fvf_factor
            ),
            # Others are dimensionless, so no scaling.
            unit_system=target,
        )


@attrs.frozen(slots=True)
class PVTDataSet(StoreSerializable):
    """
    Serialisable bundle of raw `PVTData` for all three fluid phases.

    Stores the raw tabulated data for oil, gas, and water independently of
    any interpolation settings. Use it to persist PVT data and rebuild
    `PVTTables` later with different interpolation options.

    Typical workflow:

    ```python
    # Build and persist raw data
    dataset = PVTDataSet(oil=oil_data, gas=gas_data, water=water_data)
    dataset.save("run/pvt_data.h5")

    # Reload and build tables with a specific interpolation config
    dataset = PVTDataSet.load("run/pvt_data.h5")
    tables = PVTTables.from_dataset(dataset, interpolation_method="cubic")
    ```
    """

    oil: typing.Optional[PVTData] = None
    """Raw PVT data for the oil phase."""

    gas: typing.Optional[PVTData] = None
    """Raw PVT data for the gas phase."""

    water: typing.Optional[PVTData] = None
    """Raw PVT data for the water phase."""

    @classmethod
    def from_files(
        cls,
        oil: typing.Optional[typing.Union[PathLike[str], str]] = None,
        gas: typing.Optional[typing.Union[PathLike[str], str]] = None,
        water: typing.Optional[typing.Union[PathLike[str], str]] = None,
        dtype: npt.DTypeLike = None,
        **load_kwargs: typing.Any,
    ) -> Self:
        """
        Load a `PVTDataSet` from individual per-phase files.

        :param oil: Path to a serialised oil `PVTData` file.
        :param gas: Path to a serialised gas `PVTData` file.
        :param water: Path to a serialised water `PVTData` file.
        :returns: `PVTDataSet` with the requested phases populated.
        """
        return cls(
            oil=PVTData.from_file(oil, dtype=dtype, **load_kwargs)
            if oil is not None
            else None,
            gas=PVTData.from_file(gas, dtype=dtype, **load_kwargs)
            if gas is not None
            else None,
            water=PVTData.from_file(water, dtype=dtype, **load_kwargs)
            if water is not None
            else None,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `PVTDataset` with all pvt data converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PVTDataset` in *target* units.
        """
        return self.__class__(
            oil=self.oil.convert(target, table=table) if self.oil is not None else None,
            gas=self.gas.convert(target, table=table) if self.gas is not None else None,
            water=self.water.convert(target, table=table)
            if self.water is not None
            else None,
        )
