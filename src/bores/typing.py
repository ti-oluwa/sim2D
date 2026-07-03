import enum
import typing

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]
from typing_extensions import Self, TypeAlias, TypedDict

__all__ = [
    "ArrayLike",
    "CapillaryPressures",
    "Coordinates",
    "EvolutionScheme",
    "FluidPhase",
    "Interpolator",
    "MiscibilityModel",
    "MixingRuleFunc",
    "NDimension",
    "OneDimension",
    "OneDimensionalGrid",
    "Orientation",
    "Preconditioner",
    "RelativePermeabilities",
    "Solver",
    "SolverFunc",
    "ThreeDimensionalGrid",
    "ThreeDimensions",
    "TwoDimensionalGrid",
    "TwoDimensions",
    "WellFluidType",
    "Wettability",
    "Wettability",
    "UnitSystem",
]

T = typing.TypeVar("T")
Tco = typing.TypeVar("Tco", covariant=True)
S = typing.TypeVar("S")


class UnitSystem(enum.Enum):
    """
    Declared unit system.

    Systems follow Eclipse/OPM conventions:

    ```
    =========  =======  ======  =======  =========  =======  ========  ========
    System     Length   Area    Volume   Perm.      Press.   Temp.     Rates (/day)
    =========  =======  ======  =======  =========  =======  ========  ========
    FIELD      ft       ft²     ft³      mD         psi      °F        STB/SCF
    METRIC     m        m²      m³       mD         bar      °C        Sm³
    LAB        cm       cm²     cm³      mD         atm      °C        cm³ or cc
    SI         m        m²      m³       m²         Pa       K         m³
    =========  =======  ======  =======  =========  =======  ========  ========
    ```
    """

    FIELD = "field"
    METRIC = "metric"
    LAB = "lab"
    SI = "si"


NDimension = typing.TypeVar("NDimension", bound=typing.Tuple[int, ...])
Coordinates = typing.TypeVar("Coordinates", bound=typing.Tuple[int, ...])

ThreeDimensions: TypeAlias = typing.Tuple[int, int, int]
"""3D indices"""
TwoDimensions: TypeAlias = typing.Tuple[int, int]
"""2D indices"""
OneDimension: TypeAlias = typing.Tuple[int]
"""1D index"""

Number: TypeAlias = typing.Union[int, float, np.floating, np.integer]
Boolean: TypeAlias = typing.Union[bool, np.bool_]
FloatArray: TypeAlias = np.ndarray[NDimension, np.dtype[np.floating]]
IntArray: TypeAlias = np.ndarray[NDimension, np.dtype[np.integer]]
NumberArray: TypeAlias = typing.Union[FloatArray[NDimension], IntArray[NDimension]]
BooleanArray: TypeAlias = np.ndarray[NDimension, np.dtype[np.bool_]]
NumberOrArray: TypeAlias = typing.Union[
    Number, FloatArray[NDimension], IntArray[NDimension]
]


CellArray: typing.TypeAlias = NumberArray[OneDimension]
"""Shape (n_cells,) per-cell integer or floating array."""

IntCellArray: typing.TypeAlias = IntArray[OneDimension]
"""Shape (n_cells,) per-cell integer array."""

BooleanCellArray: typing.TypeAlias = BooleanArray[OneDimension]
"""Shape (n_cells,) per-cell boolean array."""


NDimensionalGrid: TypeAlias = np.ndarray[NDimension, np.dtype[np.floating]]
ThreeDimensionalGrid = NDimensionalGrid[ThreeDimensions]
"""3D grid type for simulation data, represented as a 3D NumPy array of floats"""
TwoDimensionalGrid = NDimensionalGrid[TwoDimensions]
"""2D grid type for simulation data, represented as a 2D NumPy array of floats"""
OneDimensionalGrid = NDimensionalGrid[OneDimension]
"""1D grid type for simulation data, represented as a 1D NumPy array of floats"""


TableQuery = typing.Union[NumberOrArray[NDimension], npt.ArrayLike]
TableResult = NumberOrArray[NDimension]
InterpolationMethod = typing.Literal["linear", "cubic"]


class Orientation(enum.Enum):
    """
    Enum representing directional orientation in a 2D/3D simulation.
    """

    X = "x"
    Y = "y"
    Z = "z"
    UNSET = "unset"

    def __str__(self) -> str:
        return self.value


class FluidPhase(enum.Enum):
    """Enum representing the phase of the fluid in the reservoir."""

    WATER = "water"
    GAS = "gas"
    OIL = "oil"

    def __str__(self) -> str:
        return self.value


WellFluidType = typing.Literal["water", "oil", "gas"]
"""Types of fluids that can be injected in the simulation"""

EvolutionScheme = typing.Literal[
    "impes",
    "sequential-implicit",
    "full-sequential-implicit",
    "si",
    "full-si",
]
"""
Discretization methods for numerical simulations

- `"impes"`: Implicit pressure, Explicit saturation
- `"sequential-implicit"` or `"si"`: Both pressure and saturation are treated (sequentially) implicitly
- `"full-sequential-implicit"` or `"full-si"`: Both pressure and saturation are treated (sequentially) implicitly
"""

MiscibilityModel = typing.Literal["immiscible", "todd-longstaff"]
"""Miscibility models for fluid interactions in the simulation"""

Spacing = typing.Literal["cosine", "linspace"]


class ArrayLike(typing.Protocol[Tco]):
    """
    Protocol for an array-like object that supports
    basic operations like length, indexing, iteration, and containment checks.
    """

    def __len__(self) -> int:
        """Returns the length of the array-like object."""
        ...

    def __getitem__(self, index: int, /) -> Tco:
        """Returns the item at the specified index."""
        ...

    def __iter__(self) -> typing.Iterator[Tco]:
        """Returns an iterator over the items in the array-like object."""
        ...

    def __contains__(self, obj: typing.Any, /) -> bool:
        """Checks if the object is in the array-like object."""
        ...


Interpolator = typing.Callable[[float], float]


PreconditionerStr = typing.Union[
    typing.Literal["cpr", "ilu", "amg", "block_jacobi", "polynomial", "diagonal"], str
]
PreconditionerFactory = typing.Callable[
    [typing.Union[csr_array, csr_matrix]], LinearOperator
]
Preconditioner = typing.Union[LinearOperator, PreconditionerStr, PreconditionerFactory]

SolverStr = typing.Union[
    typing.Literal[
        "gmres",
        "lgmres",
        "bicgstab",
        "tfqmr",
        "cg",
        "cgs",
        "minres",
        "bicg",
        "qmr",
        "gcrotmk",
        "direct",
    ],
    str,
]


class SolverFunc(typing.Protocol):
    """
    Protocol for a solver function compatible with SciPy's linear solvers.
    """

    def __call__(
        self,
        A: typing.Any,
        b: typing.Any,
        x0: typing.Optional[typing.Any],
        *,
        rtol: float,
        atol: float,
        maxiter: typing.Optional[int],
        M: typing.Optional[typing.Any],
        callback: typing.Optional[typing.Callable[[npt.NDArray], None]],
    ) -> npt.NDArray: ...


Solver = typing.Union[SolverFunc, SolverStr]


class MixingRuleFunc(typing.Protocol):
    """
    Protocol for a mixing rule function that combines two-phase oil relative
    permeabilities into a three-phase estimate.

    Parameters that a simple rules does not need can simply be ignored in the implementation.
    """

    def __call__(
        self,
        *,
        kro_w: NumberOrArray[NDimension],
        kro_g: NumberOrArray[NDimension],
        krw: NumberOrArray[NDimension],
        krg: NumberOrArray[NDimension],
        kr_max: NumberOrArray[NDimension],
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        """
        Compute the three-phase oil relative permeability.

        :param kro_w: Two-phase oil relative permeability from the oil-water
            table at the current water saturation.
        :param kro_g: Two-phase oil relative permeability from the gas-oil
            table at the current gas (or oil) saturation.
        :param krw: Two-phase water relative permeability from the oil-water
            table at the current water saturation.
        :param krg: Two-phase gas relative permeability from the gas-oil
            table at the current gas saturation.
        :param kr_max: Usually oil relative permeability at connate water
            saturation (kro at Sw=Swc, Sg=0).
        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: Three-phase oil relative permeability.
        """
        ...


class MixingRulePartialDerivatives(TypedDict):
    """
    The seven partial derivatives of a three-phase oil relative permeability
    mixing rule with respect to each of its seven saturation-dependent
    arguments.

    The mixing rule signature is:

        kro = rule(kro_w, kro_g, krw, krg, kr_max,
                   water_saturation, oil_saturation, gas_saturation)

    The derivatives here are taken with respect to the *inputs that vary
    with saturation* in a way that is explicit in the mixing rule itself.
    The chain-rule terms through `kro_w`, `kro_g`, `krw`, and `krg`
    back to physical saturations are assembled at the call site in the table
    derivative methods.

    Fields:

    d_kro_d_kro_w :
        ∂kro / ∂kro_w  - sensitivity to the oil-water two-phase oil kr.
    d_kro_d_kro_g :
        ∂kro / ∂kro_g  - sensitivity to the gas-oil two-phase oil kr.
    d_kro_d_krw :
        ∂kro / ∂krw  - sensitivity to the two-phase water kr.
        Zero for rules that do not use `krw` directly (most simple rules).
    d_kro_d_krg :
        ∂kro / ∂krg  - sensitivity to the two-phase gas kr.
        Zero for rules that do not use `krg` directly (most simple rules).
    d_kro_d_sw_explicit :
        ∂kro / ∂Sw  through the explicit water-saturation argument of the
        mixing rule (e.g. saturation weighting in `eclipse_rule`).
        Zero for rules that do not depend directly on saturation.
    d_kro_d_so_explicit :
        ∂kro / ∂So  through the explicit oil-saturation argument.
    d_kro_d_sg_explicit :
        ∂kro / ∂Sg  through the explicit gas-saturation argument.
    """

    d_kro_d_kro_w: NumberOrArray
    d_kro_d_kro_g: NumberOrArray
    d_kro_d_krw: NumberOrArray
    d_kro_d_krg: NumberOrArray
    d_kro_d_sw_explicit: NumberOrArray
    d_kro_d_so_explicit: NumberOrArray
    d_kro_d_sg_explicit: NumberOrArray


class MixingRuleDFunc(typing.Protocol):
    """
    Protocol for a mixing rule partial derivatives function.
    """

    def __call__(
        self,
        *,
        kro_w: NumberOrArray[NDimension],
        kro_g: NumberOrArray[NDimension],
        krw: NumberOrArray[NDimension],
        krg: NumberOrArray[NDimension],
        kr_max: NumberOrArray[NDimension],
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
    ) -> typing.Union[
        MixingRulePartialDerivatives,
        typing.Tuple[
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
        ],
    ]:
        """
        Return the seven partial derivatives of the mixing rule.

        The tuple order (when not returning a `MixingRulePartialDerivatives` dict) is:

        `(d_kro_d_kro_w, d_kro_d_kro_g, d_kro_d_krw, d_kro_d_krg,
           d_kro_d_sw_explicit, d_kro_d_so_explicit, d_kro_d_sg_explicit)`

        :param kro_w: Two-phase oil kr from oil-water table.
        :param kro_g: Two-phase oil kr from gas-oil table.
        :param krw: Two-phase water kr from oil-water table.
        :param krg: Two-phase gas kr from gas-oil table.
        :param kr_max: Oil kr at connate water (normalization reference).
        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: The seven partial derivatives, either as a
            `MixingRulePartialDerivatives` dict or a 7-tuple.
        """
        ...


class RelativePermeabilities(TypedDict):
    """Dictionary holding relative permeabilities for different phases."""

    water: NumberOrArray
    oil: NumberOrArray
    gas: NumberOrArray


class RelativePermeabilityDerivatives(TypedDict):
    """Dictionary holding relative permeabilities derivatives."""

    # w.r.t water
    dKrw_dSw: NumberOrArray
    dKro_dSw: NumberOrArray
    dKrg_dSw: NumberOrArray
    # w.r.t oil
    dKrw_dSo: NumberOrArray
    dKro_dSo: NumberOrArray
    dKrg_dSo: NumberOrArray
    # w.r.t gas
    dKrw_dSg: NumberOrArray
    dKro_dSg: NumberOrArray
    dKrg_dSg: NumberOrArray


class CapillaryPressures(TypedDict):
    """Dictionary containing capillary pressures for different phase pairs."""

    oil_water: NumberOrArray  # Pcow = Po - Pw
    gas_oil: NumberOrArray  # Pcgo = Pg - Po


class CapillaryPressureDerivatives(TypedDict):
    """Dictionary containing capillary pressure derivatives for different phase pairs."""

    dPcow_dSw: NumberOrArray
    dPcow_dSo: NumberOrArray
    dPcgo_dSg: NumberOrArray
    dPcgo_dSo: NumberOrArray


class Wettability(enum.Enum):
    """Enum representing the wettability type of the reservoir rock."""

    WATER_WET = "water_wet"
    OIL_WET = "oil_wet"
    MIXED_WET = "mixed_wet"

    def __str__(self) -> str:
        return self.value


Kcon = typing.TypeVar("Kcon", contravariant=True)
Vcon = typing.TypeVar("Vcon", contravariant=True)


ZFactorMethod = typing.Literal["papay", "hall-yarborough", "dak"]


class UnitConversionFactors(TypedDict):
    pressure: Number
    """Pressure conversion factor."""

    length: Number
    """Length conversion factor."""

    density: Number
    """Density conversion factor."""

    viscosity: Number
    """Dynamic viscosity conversion factor."""

    permeability: Number
    """Permeability conversion factor."""

    compressibility: Number
    """Compressibility conversion factor."""

    liquid_fvf: Number
    """Liquid formation volume factor conversion factor."""

    gas_fvf: Number
    """Gas formation volume factor conversion factor."""

    gor: Number
    """Gas-oil ratio conversion factor."""

    temperature: Number
    """Multiplicative temperature conversion factor."""

    temperature_offset: Number
    """Additive temperature conversion offset."""

    volumetric_rate: Number
    """
    Volumetric flow rate conversion factor (volume/time).

    FIELD: ft³/day, METRIC: m³/day, LAB: cm³/hour, SI: m³/s.
    Derived as `volume_factor / time_factor` where time is always in days
    for FIELD/METRIC, hours for LAB, and seconds for SI.
    """

    mass: Number
    """
    Mass conversion factor.

    FIELD: lbm, METRIC: kg, LAB: g, SI: kg.
    Derived as `density_factor * length_factor³`.
    """


UnitConversionTable = typing.Dict[
    typing.Tuple[UnitSystem, UnitSystem], UnitConversionFactors
]
"""Mapping of unit system pairs `(from, target)` to unit conversion factors"""


class SupportsUnitSystem(typing.Protocol):
    """Protocol defining `UnitSystem` support."""

    unit_system: UnitSystem

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self: ...
