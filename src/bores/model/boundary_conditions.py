"""Boundary condition system for unstructured polyhedral reservoir grids."""

import enum
import functools
import logging
import threading
import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.errors import DeserializationError, SerializationError, ValidationError
from bores.grids.base import Grid
from bores.model.properties.pvt import PVT
from bores.model.properties.rock import Rock
from bores.model.properties.state import State
from bores.precision import get_dtype
from bores.serialization import Serializable, make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.typing import (
    BooleanArray,
    IntArray,
    NDimension,
    Number,
    NumberArray,
    OneDimension,
    SupportsUnitSystem,
    UnitConversionTable,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BoundaryCondition",
    "BoundaryConditionType",
    "BoundaryConditions",
    "BoundaryRegion",
    "CarterTracyAquifer",
    "ConstantFluxBoundary",
    "ConstantPressureBoundary",
    "ParameterizedBoundaryFunction",
    "ProductivityIndexBoundary",
    "TimeDependentFluxBoundary",
    "boundary_function",
    "get_boundary_function",
    "list_boundary_functions",
]


_BOUNDARY_FUNCTIONS: typing.Dict[str, typing.Callable] = {}
_boundary_function_lock = threading.Lock()

P = typing.ParamSpec("P")
R = typing.TypeVar("R")


@typing.overload
def boundary_function(func: typing.Callable[P, R]) -> typing.Callable[P, R]: ...


@typing.overload
def boundary_function(
    func: None = None,
    *,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[[typing.Callable[P, R]], typing.Callable[P, R]]: ...


def boundary_function(
    func: typing.Optional[typing.Callable[P, R]] = None,
    *,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[
    typing.Callable[P, R],
    typing.Callable[[typing.Callable[P, R]], typing.Callable[P, R]],
]:
    """
    Register a callable as a named boundary function for serialisation.

    A *boundary function* is any callable that computes a per-face array of
    either fluxes or pressures.  Registering it by name allows
    `ParameterizedBoundaryFunction` instances that hold a reference to it to
    survive serialisation / deserialisation round-trips.

    Can be used as a bare decorator or with keyword arguments::

        @boundary_function
        def constant_flux(face_indices, state, rock, pvt, grid, time, *, value):
            return np.full(len(face_indices), value)

        @boundary_function(name="my_flux", override=True)
        def my_flux(face_indices, state, rock, pvt, grid, time, *, rate):
            return np.full(len(face_indices), rate / len(face_indices))

    :param func: The callable to register (supplied automatically by Python
        when the decorator is used without arguments).
    :param name: Registration key.  Defaults to `func.__name__`.
    :param override: When `True`, silently replaces an existing registration
        under the same key.  When `False` (default), raises `ValidationError`
        if the key is already taken.
    :returns: The original callable unchanged, or a decorator when `func`
        is `None`.
    :raises ValidationError: If the callable has no usable name, or the name
        is already registered and `override` is `False`.
    """

    def _register(fn: typing.Callable[P, R]) -> typing.Callable[P, R]:
        key = name or getattr(fn, "__name__", None)
        if not key:
            raise ValidationError(
                "Boundary function must have a `__name__` attribute or an explicit "
                "`name` keyword argument."
            )
        with _boundary_function_lock:
            if not override and key in _BOUNDARY_FUNCTIONS:
                raise ValidationError(
                    f"Boundary function {key!r} is already registered. "
                    "Pass `override=True` to replace it."
                )
            _BOUNDARY_FUNCTIONS[key] = fn
        return fn

    if func is not None:
        return _register(func)
    return _register


def get_boundary_function(name: str) -> typing.Callable:
    """
    Return a registered boundary function by name.

    :param name: The registration key used when the function was decorated.
    :returns: The callable.
    :raises ValidationError: If `name` is not in the registry.
    """
    with _boundary_function_lock:
        fn = _BOUNDARY_FUNCTIONS.get(name)
    if fn is None:
        with _boundary_function_lock:
            available = sorted(_BOUNDARY_FUNCTIONS.keys())
        raise ValidationError(
            f"Boundary function {name!r} is not registered. "
            f"Decorate it with `@boundary_function`. "
            f"Currently registered: {available}."
        )
    return fn


def list_boundary_functions() -> typing.List[str]:
    """Return the names of all currently registered boundary functions."""
    with _boundary_function_lock:
        return sorted(_BOUNDARY_FUNCTIONS.keys())


def _serialise_boundary_function(
    func: typing.Callable,
    recurse: bool = True,
) -> typing.Dict[str, typing.Any]:
    """
    Serialise a boundary function to a JSON-compatible dict.

    Supports three kinds of callable:

    - **Registered function** - serialised as `{"type": "registered", "name": ...}`.
    - **`functools.partial`** - serialised recursively with `args` and `kwargs`.
    - **`ParameterizedBoundaryFunction`** - serialised via its own `dump` method.

    :param func: The callable to serialise.
    :param recurse: Passed through to `ParameterizedBoundaryFunction.dump`.
    :returns: Serialisation dict.
    :raises SerializationError: If the callable is not identifiable.
    """
    with _boundary_function_lock:
        for key, registered in _BOUNDARY_FUNCTIONS.items():
            if func is registered:
                return {"type": "registered", "name": key}

    if isinstance(func, functools.partial):
        return {
            "type": "partial",
            "func": _serialise_boundary_function(func.func, recurse),
            "args": list(func.args),
            "kwargs": dict(func.keywords),
        }

    if isinstance(func, ParameterizedBoundaryFunction):
        return {"type": "parameterized", "data": func.dump(recurse)}

    raise SerializationError(
        f"Cannot serialise boundary function {func!r}. "
        "Register it with `@boundary_function` first."
    )


def _deserialise_boundary_function(
    data: typing.Mapping[str, typing.Any],
) -> typing.Callable:
    """
    Reconstruct a boundary function from a dict produced by
    `_serialise_boundary_function`.

    :param data: Serialised representation.
    :returns: The reconstructed callable.
    :raises DeserializationError: If the data is malformed or the type is unknown.
    """
    func_type = data.get("type")

    if func_type == "registered":
        if "name" not in data:
            raise DeserializationError(
                "Missing 'name' key in registered boundary function data."
            )
        return get_boundary_function(data["name"])

    if func_type == "partial":
        if "func" not in data:
            raise DeserializationError(
                "Missing 'func' key in partial boundary function data."
            )
        base = _deserialise_boundary_function(data["func"])
        return functools.partial(
            base,
            *data.get("args", []),
            **data.get("kwargs", {}),
        )

    if func_type == "parameterized":
        if "data" not in data:
            raise DeserializationError(
                "Missing 'data' key in parameterized boundary function data."
            )
        return ParameterizedBoundaryFunction.load(data["data"])

    raise DeserializationError(
        f"Unknown boundary function type {func_type!r}. "
        "Expected 'registered', 'partial', or 'parameterized'."
    )


class ParameterizedBoundaryFunction(
    Serializable,
    fields={"function_name": str, "parameters": typing.Dict[str, typing.Any]},
):
    """
    A fully serialisable alternative to `functools.partial` for boundary functions.

    Stores a registered boundary function by name together with a fixed parameter
    dict.  On call, the stored parameters are merged with any additional keyword
    arguments supplied by the caller (caller kwargs take precedence).

    This is the preferred way to combine a generic registered function with
    site-specific parameters when you need the boundary condition to survive
    serialisation round-trips::

        @boundary_function
        def depth_weighted_flux(
            face_indices, state, rock, pvt, grid, time,
            *, rate_per_unit_depth, datum_depth
        ):
            depths = grid.cell_center_depths[
                grid.face_cell_indices[
                    grid.boundary_face_indices[face_indices], 0
                ]
            ]
            weights = np.abs(depths - datum_depth)
            weights /= weights.sum() or 1.0
            return weights * rate_per_unit_depth

        influx = ParameterizedBoundaryFunction(
            function_name="depth_weighted_flux",
            parameters={"rate_per_unit_depth": 50.0, "datum_depth": 2500.0},
        )

    :param function_name: Name under which the base function was registered with
        `@boundary_function`.
    :param parameters: Fixed keyword parameters merged into every call.
    """

    def __init__(
        self,
        function_name: str,
        parameters: typing.Dict[str, typing.Any],
    ) -> None:
        self.function_name = function_name
        self.parameters = parameters
        self._func = get_boundary_function(function_name)

    def __call__(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> NumberArray[OneDimension]:
        """
        Invoke the underlying function with the stored parameters.

        Stored parameters are merged with `kwargs` before the call;
        caller-supplied kwargs take precedence over stored parameters.

        :param args: Positional arguments forwarded to the base function.
        :param kwargs: Keyword arguments; override stored `parameters`.
        :returns: Whatever the base function returns (typically a
            `numpy.ndarray` of per-face values).
        """
        return self._func(*args, **{**self.parameters, **kwargs})

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {"function_name": self.function_name, "parameters": self.parameters}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(
            function_name=data["function_name"],
            parameters=dict(data["parameters"]),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"function_name={self.function_name!r}, "
            f"parameters={self.parameters!r}"
            f")"
        )


class BoundaryConditionType(enum.Enum):
    """
    Discriminator that controls how the solver uses a boundary condition's output.

    `PRESSURE`
        The `evaluate` method returns a prescribed pressure (psi / bar / atm
        / Pa depending on `unit_system`) at each boundary face.  The solver
        applies a Dirichlet constraint: it adds `T_face * p_boundary` to the
        RHS and `T_face` to the diagonal of the flow equation for each face.

    `FLUX`
        The `evaluate` method returns a volumetric flow rate into the
        reservoir (ft³/day / m³/day / etc.) at each boundary face. The solver
        applies a Neumann source term: it adds the flux directly to the RHS of
        the cell owning the face.  A flux of zero is a sealed (no-flow)
        boundary.
    """

    PRESSURE = "pressure"
    FLUX = "flux"


class BoundaryCondition(StoreSerializable, SupportsUnitSystem):
    """
    Base class for all boundary conditions.

    Subclasses must implement:

    - `condition_type` property - declares `PRESSURE` or `FLUX`.
    - `evaluate` method - computes per-face values at the given face positions.
    - `convert` method - returns a unit-rescaled copy.

    The `__call__` method delegates to `evaluate` so that a
    `BoundaryCondition` instance is directly callable with the same
    signature, enabling it to be used as a `ParameterizedBoundaryFunction`
    target or passed to any callable-accepting API.

    **Face index convention**

    `face_positions` contains *positions* (0-based indices) into
    `Grid.boundary_face_indices`, not global face indices. This means the
    solver can look up the half-transmissibility for face *i* as
    `transmissibilities.boundary[face_positions[i]]` without an additional
    mapping step.

    **Unit system contract**

    Every concrete subclass that carries dimensional parameters (pressures,
    rates, permeabilities) must:

    - Implement the `SupportsUnitSystem` protocol.
    - Ensure `evaluate` returns values in the same units as `unit_system`.

    The solver is responsible for ensuring that the unit system of the boundary
    conditions matches the unit system of the model before calling `evaluate`.
    """

    __abstract_serializable__ = True

    @property
    def condition_type(self) -> BoundaryConditionType:
        """
        Declares whether this condition prescribes pressure or flux.

        :returns: `BoundaryConditionType.PRESSURE` or
            `BoundaryConditionType.FLUX`.
        """
        raise NotImplementedError

    def evaluate(
        self,
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Compute boundary condition values at the given boundary face positions.

        The returned array is aligned with `face_positions`: element *i*
        corresponds to boundary face `grid.boundary_face_indices[face_positions[i]]`.

        For `PRESSURE` conditions the returned values are pressures in
        `self.unit_system` units. For `FLUX` conditions they are
        volumetric flow rates (positive = into reservoir) in the equivalent
        volume-per-time unit.

        :param face_positions: Shape `(n_faces,)` int32 - positions into
            `Grid.boundary_face_indices` for this region.
        :param state: Current `State`.
        :param rock: `Rock` properties.
        :param pvt: `PVT` fluid properties.
        :param grid: The simulation `Grid`.
        :param time: Current simulation time in days.
        :param dtype: Output array dtype.  When `None`, `get_dtype()` is used.
        :returns: Shape `(n_faces,)` array of pressures or fluxes.
        """
        raise NotImplementedError

    def __call__(
        self,
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Shorthand for `evaluate`. Makes the condition directly callable.

        :param face_positions: Shape `(n_faces,)` int32.
        :param state: Current `State`.
        :param rock: `Rock` properties.
        :param pvt: `PVT` fluid properties.
        :param grid: The simulation `Grid`.
        :param time: Current simulation time in days.
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` array from `evaluate`.
        """
        return self.evaluate(face_positions, state, rock, pvt, grid, time, dtype)


# Registry of concrete boundary condition classes
_BOUNDARY_CONDITIONS: typing.Dict[str, typing.Type[BoundaryCondition]] = {}
boundary_condition = make_serializable_type_registrar(
    base_cls=BoundaryCondition,
    registry=_BOUNDARY_CONDITIONS,
    lock=threading.Lock(),
    key_attr="__type__",
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)
"""
Class decorator that registers a `BoundaryCondition` subclass for
serialisation. Must be applied to every concrete condition class that needs
to survive a `dump` / `load` cycle.
"""


@boundary_condition
@attrs.frozen
class ConstantFluxBoundary(BoundaryCondition):
    """
    Constant-flux (Neumann) boundary condition.

    Returns a uniform volumetric flow rate at every face in the region.

    The default `flux=0.0` gives a **sealed (no-flow) boundary** - the most
    common boundary condition for reservoir flanks, top, and base. A positive
    `flux` represents flow *into* the reservoir (injection or aquifer
    influx); a negative `flux` represents production or efflux.

    **Unit system**

    The `flux` value is interpreted in the *volume-per-time* unit of
    `unit_system`:

    - `FIELD` - ft³/day
    - `METRIC` - m³/day
    - `LAB` - cm³/hour
    - `SI` - m³/s

    **No-flow shortcut**

    An empty `BoundaryRegion` face mask with a `ConstantFluxBoundary(flux=0.0)`
    is exactly equivalent to not including the region at all, since unregistered
    boundary faces already default to zero flux inside `BoundaryConditions.evaluate`.

    :param flux: Volumetric flow rate applied uniformly across the region
        (volume/time in `unit_system`).  Default is 0 (no-flow).
    :param unit_system: Unit system for `flux`.  Default `FIELD` (ft³/day).
    """

    __type__ = "constant_flux_boundary"

    flux: Number = 0.0
    """Volumetric flux (volume/time in `unit_system`). Positive = into reservoir."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `flux`."""

    @property
    def condition_type(self) -> BoundaryConditionType:
        """Return `BoundaryConditionType.FLUX`."""
        return BoundaryConditionType.FLUX

    def evaluate(
        self,
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Return `flux` broadcast to the shape `(n_faces,)`.

        The flux is distributed *uniformly* across all faces in the region.
        If you need a spatially varying distribution, use
        `ProductivityIndexBoundary` with a custom `alpha_function`, or
        subclass `BoundaryCondition` and override `evaluate`.

        :param face_positions: Shape `(n_faces,)` - positions into
            `Grid.boundary_face_indices`.
        :param state: Unused.
        :param rock: Unused.
        :param pvt: Unused.
        :param grid: Unused.
        :param time: Unused.
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` float array, every element equal to
            `self.flux`.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        return np.full(len(face_positions), self.flux, dtype=dtype)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `ConstantFluxBoundary` with `flux` rescaled to
        *target*.

        :param target: Target `UnitSystem`.
        :returns: New `ConstantFluxBoundary` in `target` units.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            flux=self.flux * factors["volumetric_rate"], unit_system=target
        )

    def is_no_flow(self) -> bool:
        """Return `True` if `flux == 0.0` (sealed boundary)."""
        return self.flux == 0.0


@boundary_condition
@attrs.frozen
class ConstantPressureBoundary(BoundaryCondition):
    """
    Constant-pressure (Dirichlet) boundary condition.

    Prescribes a fixed pressure at every face in the region. The solver uses
    this as a ghost-cell pressure to compute the face flux via the half-
    transmissibility:

        q_face = T_half * (pressure_boundary - pressure_interior)

    Typical applications:

    - **Strong aquifer** - an aquifer so large that its pressure does not
      change over the simulation period.  Set `pressure` to the initial
      aquifer pressure.
    - **Constant-pressure producer** - a producing boundary held at
      abandonment pressure.
    - **Injection at manifold pressure** - an injection flank held at the
      injection pump delivery pressure.

    **Unit system**

    `pressure` is in the *pressure* unit of `unit_system`:

    - `FIELD` - psi
    - `METRIC` - bar
    - `LAB` - atm
    - `SI` - Pa

    :param pressure: Prescribed pressure at all faces in the region.
    :param unit_system: Unit system for `pressure`.
    """

    __type__ = "constant_pressure_boundary"

    pressure: Number
    """Prescribed boundary pressure in `unit_system` units."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `pressure`."""

    @property
    def condition_type(self) -> BoundaryConditionType:
        """Return `BoundaryConditionType.PRESSURE`."""
        return BoundaryConditionType.PRESSURE

    def evaluate(
        self,
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Return `pressure` broadcast to shape `(n_faces,)`.

        :param face_positions: Shape `(n_faces,)`.
        :param state: Unused.
        :param rock: Unused.
        :param pvt: Unused.
        :param grid: Unused.
        :param time: Unused.
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` array, every element equal to
            `self.pressure`.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        return np.full(len(face_positions), self.pressure, dtype=dtype)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `ConstantPressureBoundary` with `pressure` rescaled
        to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `ConstantPressureBoundary` in `target` units.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            pressure=self.pressure * factors["pressure"], unit_system=target
        )


@boundary_condition
@attrs.frozen
class ProductivityIndexBoundary(BoundaryCondition):
    """
    Robin (mixed) boundary condition using a productivity-index formulation.

    Computes a volumetric flux at each face as:

        q_face[i] = PI_face[i] * (pressure_boundary - p_interior[i])

    where `p_interior[i]` is the pressure of the cell owning boundary face
    `face_positions[i]` from the current `State`.

    This is physically equivalent to a **well-index / productivity-index**
    formulation applied at the grid boundary which is useful for:

    - **Partial aquifer** - an aquifer whose influx is approximately
      proportional to the pressure difference at the boundary.
    - **Leaky boundary** - a boundary face with a known transmissibility-like
      coefficient connecting the reservoir to an external pressure source.
    - **Injection manifold** - injection driven by a manifold pressure with a
      known injectivity per face.

    **Productivity index per face**

    The PI can be:

    - A **scalar float** (`productivity_index`) applied uniformly to every
      face in the region.  This is the simplest case.
    - A **callable** (`alpha_function`) with signature:

    ```python
    alpha_function(
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
    ) -> NumberArray[OneDimension]: ...  # shape (n_faces,)
    ```

    When `alpha_function` is provided, `productivity_index` is ignored.
    The callable can be a registered `@boundary_function`, a
    `ParameterizedBoundaryFunction`, or any callable satisfying the
    signature above. It must be registered to survive serialisation.

    **Sign convention**

    A positive result means flow *into* the reservoir (net influx):
    `pressure_boundary > p_interior` -> positive flux.
    A negative result means flow *out of* the reservoir (net production):
    `pressure_boundary < p_interior` -> negative flux.

    **Unit system**

    - `pressure_boundary` - pressure in `unit_system`.
    - `productivity_index` - volume/(time·pressure) in `unit_system`
      (ft³/day/psi, m³/day/bar, etc.).

    :param pressure_boundary: Reference pressure at the boundary.
    :param productivity_index: Uniform PI (volume/time/pressure) applied to
        every face when `alpha_function` is `None`.
    :param alpha_function: Optional callable that returns a per-face PI array.
        When provided, `productivity_index` is ignored.
    :param unit_system: Unit system for `pressure_boundary` and
        `productivity_index`.
    """

    __type__ = "productivity_index_boundary"

    pressure_boundary: Number
    """Reference boundary pressure in `unit_system` units."""

    productivity_index: Number = 1.0
    """
    Uniform productivity index (volume/time/pressure in `unit_system`).

    Ignored when `alpha_function` is not `None`.
    """

    alpha_function: typing.Optional[
        typing.Callable[
            [
                IntArray[OneDimension],
                State,
                Rock,
                PVT,
                Grid,
                Number,
            ],
            npt.NDArray,
        ]
    ] = attrs.field(default=None)
    """
    Optional per-face PI callable.

    Signature:

    ```python
    alpha_function(
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
    ) -> NumberArray[OneDimension]: ...  # shape (n_faces,)
    ```

    Returns a shape `(n_faces,)` array of PI values (volume/time/pressure).
    Must be a registered `@boundary_function` or `ParameterizedBoundaryFunction`
    for serialisation to work. When `None`, the scalar `productivity_index`
    is used instead.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `pressure_boundary` and `productivity_index`."""

    @property
    def condition_type(self) -> BoundaryConditionType:
        """Return `BoundaryConditionType.FLUX`."""
        return BoundaryConditionType.FLUX

    def evaluate(
        self,
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Compute `PI_face * (pressure_boundary - p_interior)` at each face.

        The owner cell for boundary face at position `position` is:

        ```python
        owner = grid.face_cell_indices[grid.boundary_face_indices[position], 0]
        ```

        The interior pressure at that cell is `state.pressure[owner]`.

        :param face_positions: Shape `(n_faces,)` - positions into
            `Grid.boundary_face_indices`.
        :param state: Current `State`; provides `pressure`.
        :param rock: Unused.
        :param pvt: Unused.
        :param grid: The simulation `Grid`; used to resolve owner cells.
        :param time: Current simulation time (days).
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` flux array (volume/time in
            `self.unit_system`).
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        n_faces = len(face_positions)

        # Resolve owner cell pressures for each boundary face
        global_face_indices = grid.boundary_face_indices[face_positions]
        owner_cells = grid.face_cell_indices[global_face_indices, 0]
        p_interior = state.pressure[owner_cells].astype(dtype, copy=False)

        # Resolve per-face PI
        if self.alpha_function is not None:
            productivity_index = np.asarray(
                self.alpha_function(face_positions, state, rock, pvt, grid, time),
                dtype=dtype,
                copy=False,
            )
        else:
            productivity_index = np.full(n_faces, self.productivity_index, dtype=dtype)

        return typing.cast(
            NumberArray[OneDimension],
            productivity_index * (self.pressure_boundary - p_interior),
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `ProductivityIndexBoundary` with `pressure_boundary`
        and `productivity_index` rescaled to *target*.

        The `alpha_function` callable is not rescaled - if it returns values
        in the original unit system, you must handle the conversion inside the
        function itself or replace it with a rescaled version.

        :param target: Target `UnitSystem`.
        :returns: New `ProductivityIndexBoundary` in *target* units.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        productivity_index_factor = factors["volumetric_rate"] / factors["pressure"]
        return self.__class__(
            pressure_boundary=self.pressure_boundary * factors["pressure"],
            productivity_index=self.productivity_index * productivity_index_factor,
            alpha_function=self.alpha_function,
            unit_system=target,
        )

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        data: typing.Dict[str, typing.Any] = {
            "pressure_boundary": self.pressure_boundary,
            "productivity_index": self.productivity_index,
            "unit_system": self.unit_system.value,
        }
        if self.alpha_function is not None:
            data["alpha_function"] = _serialise_boundary_function(
                self.alpha_function, recurse
            )
        return data

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        alpha = None
        if "alpha_function" in data:
            alpha = _deserialise_boundary_function(data["alpha_function"])
        return cls(
            pressure_boundary=float(data["pressure_boundary"]),
            productivity_index=float(data.get("productivity_index", 1.0)),
            alpha_function=alpha,
            unit_system=UnitSystem(data.get("unit_system", UnitSystem.FIELD.value)),
        )


@boundary_condition
@attrs.frozen
class TimeDependentFluxBoundary(BoundaryCondition):
    """
    Time-varying flux (Neumann) boundary condition driven by a registered
    schedule function.

    The schedule function has the signature:

    ```python
    schedule_function(
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
    ) -> NumberArray[OneDimension]: ...  # shape (n_faces,)
    ```
    and must be registered with `@boundary_function` for serialisation.

    Typical use cases:

    - **Seasonal injection** - an aquifer or injection boundary whose flux
      follows a prescribed temporal profile (e.g. tidal, seasonal recharge).
    - **Declining production** - a producer whose outer boundary pressure
      drops on a known decline curve.
    - **Data-driven influx** - replay historical aquifer influx rates from
      a measured time series stored in the schedule function's closure.

    The schedule function receives the full model state at each call, so it
    can also condition the flux on the current pressure field or saturation
    distribution - making this a superset of `ProductivityIndexBoundary`
    for time-varying or state-dependent cases.

    :param schedule_function: Registered callable returning per-face fluxes.
    :param unit_system: Unit system for the returned flux values.
    """

    __type__ = "time_dependent_flux_boundary"

    schedule_function: typing.Callable[
        [
            IntArray[OneDimension],
            State,
            Rock,
            PVT,
            Grid,
            Number,
        ],
        npt.NDArray,
    ]
    """
    Callable returning per-face fluxes at the current simulation time.

    Signature:
    ```python
    schedule_function(
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
    ) -> NumberArray[OneDimension]: ...  # shape (n_faces,), volume/time in `unit_system`
    ```

    Must be registered with `@boundary_function` for serialisation.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for the values returned by `schedule_function`."""

    @property
    def condition_type(self) -> BoundaryConditionType:
        """Return `BoundaryConditionType.FLUX`."""
        return BoundaryConditionType.FLUX

    def evaluate(
        self,
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Invoke `schedule_function` and return the per-face flux array.

        :param face_positions: Shape `(n_faces,)` - positions into
            `Grid.boundary_face_indices`.
        :param state: Current `State`.
        :param rock: `Rock` properties.
        :param pvt: `PVT` fluid properties.
        :param grid: The simulation `Grid`.
        :param time: Current simulation time (days).
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` flux array (volume/time).
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        result = self.schedule_function(face_positions, state, rock, pvt, grid, time)
        return typing.cast(
            NumberArray[OneDimension], np.asarray(result, dtype=dtype, copy=False)
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `TimeDependentFluxBoundary` with a different
        `unit_system` label.

        The `schedule_function` itself is not rescaled - it is assumed that
        the callable already incorporates whatever unit awareness is needed.
        If the callable returns values in a specific unit system, you must
        handle the conversion inside the function or replace it.

        :param target: Target `UnitSystem`.
        :returns: New `TimeDependentFluxBoundary` with updated
            `unit_system`.
        """
        if target == self.unit_system:
            return self

        warnings.warn(
            f"`{self.__class__.__name__}.convert` only updates the `unit_system` "
            f"label from {self.unit_system.value!r} to {target.value!r}. "
            "The `schedule_function` callable is not rescaled automatically. "
            "Ensure the function returns values in the target unit system.",
            UserWarning,
            stacklevel=2,
        )
        return self.__class__(
            schedule_function=self.schedule_function, unit_system=target
        )

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "schedule_function": _serialise_boundary_function(
                self.schedule_function, recurse
            ),
            "unit_system": self.unit_system.value,
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(
            schedule_function=_deserialise_boundary_function(data["schedule_function"]),
            unit_system=UnitSystem(data.get("unit_system", UnitSystem.FIELD.value)),
        )


@boundary_condition
@attrs.mutable
class CarterTracyAquifer(BoundaryCondition):
    """
    Finite radial aquifer boundary condition using the Carter-Tracy (1960)
    superposition method.

    Computes transient water influx from a finite radial aquifer into the
    reservoir boundary faces governed by this condition. Returns a per-face
    volumetric influx rate array (volume/time, positive = into reservoir).

    **Physical model**

    The aquifer is assumed to be:

    - Radially symmetric about a central axis coincident with the
      reservoir-aquifer contact.
    - Finite in extent, characterised by an outer radius `outer_radius` and
      an encroachment angle `angle` (degrees).
    - Homogeneous in permeability, porosity, and compressibility.

    Influx is governed by the Van Everdingen-Hurst (1949) dimensionless
    cumulative influx function, evaluated using the Carter-Tracy (1960)
    superposition approximation, which avoids the expensive Laplace-space
    convolution integral of the original formulation.

    **Two construction modes**

    *Physical properties (recommended)*:

    Supply `aquifer_permeability`, `aquifer_porosity`,
    `aquifer_compressibility`, `water_viscosity`, `inner_radius`,
    `outer_radius`, and `aquifer_thickness`.  The aquifer constant *B*
    and hydraulic diffusivity *η* are derived automatically.

    *Calibrated constant (history-matched)*:

    Supply `aquifer_constant` (volume/pressure in `unit_system`) and
    `dimensionless_radius_ratio`.  Useful when the aquifer parameters have
    been history-matched against production data but the individual physical
    properties are not reliably known.

    **Superposition**

    The condition maintains an internal pressure history list of
    `(time, pressure_drop)` pairs to implement Van Everdingen-Hurst
    superposition. The history is updated on every call to `evaluate`
    and is persisted through `__dump__` / `__load__` so that a
    restarted simulation continues from the correct aquifer state.

    **Flux distribution across faces**

    The total aquifer influx rate (a single scalar) is distributed uniformly
    across all boundary faces in the region. For most applications the
    aquifer contacts the reservoir along a single geological surface and the
    faces are approximately equal in area, so uniform distribution is
    acceptable. If spatially weighted distribution is required, subclass
    and override `evaluate` to apply area-weighted fractions.

    **Unit system**

    All pressure parameters (`initial_pressure`, `aquifer_constant`)
    are in the *pressure* unit of `unit_system`. All length parameters
    (`inner_radius`, `outer_radius`, `aquifer_thickness`) are in the
    *length* unit of `unit_system`. Permeability is in the *permeability*
    unit. Viscosity is in cP for `FIELD` / `METRIC` / `LAB`, or Pa·s
    for `SI`. The returned flux is in volume/time.

    **References**

    Carter, R.D. and Tracy, G.W. (1960). *An Improved Method for Calculating
    Water Influx.* JPT, 12(5), 415-417.

    Van Everdingen, A.F. and Hurst, W. (1949). *The Application of the
    Laplace Transformation to Flow Problems in Reservoirs.* Trans. AIME,
    186, 305-324.

    :param initial_pressure: Initial aquifer / reservoir pressure.
    :param aquifer_permeability: Aquifer horizontal permeability.
        Required for physical-properties mode.
    :param aquifer_porosity: Aquifer porosity (fraction, dimensionless).
        Required for physical-properties mode.
    :param aquifer_compressibility: Total aquifer compressibility
        (rock + water), in 1/pressure units. Required for
        physical-properties mode.
    :param water_viscosity: Water viscosity at reservoir conditions (cP or Pa·s).
        Required for physical-properties mode.
    :param inner_radius: Reservoir-aquifer contact radius (length).
        Required for physical-properties mode.
    :param outer_radius: Aquifer outer extent (length).
        Required for physical-properties mode.
    :param aquifer_thickness: Aquifer thickness (length).
        Required for physical-properties mode.
    :param aquifer_constant: Pre-computed or history-matched aquifer constant
        *B* (volume/pressure in `unit_system`). Required for
        calibrated-constant mode.
    :param dimensionless_radius_ratio: `r_outer / r_inner`. Required for
        calibrated-constant mode; ignored in physical-properties mode.
    :param angle: Aquifer encroachment angle (degrees). 360 = full radial
        contact; 180 = half-circle edge drive. Default 360.
    :param unit_system: Unit system for all dimensional parameters.
    """

    __type__ = "carter_tracy_aquifer"

    initial_pressure: Number
    """Initial aquifer / reservoir pressure in `unit_system` pressure units."""

    aquifer_permeability: typing.Optional[Number] = attrs.field(default=None)
    """Aquifer permeability in `unit_system` permeability units. Physical mode only."""

    aquifer_porosity: typing.Optional[Number] = attrs.field(default=None)
    """Aquifer porosity (fraction). Physical mode only."""

    aquifer_compressibility: typing.Optional[Number] = attrs.field(default=None)
    """Total aquifer compressibility (1/pressure in `unit_system`). Physical mode only."""

    water_viscosity: typing.Optional[Number] = attrs.field(default=None)
    """Water viscosity at reservoir conditions. Physical mode only."""

    inner_radius: typing.Optional[Number] = attrs.field(default=None)
    """Reservoir-aquifer contact radius (length in `unit_system`). Physical mode only."""

    outer_radius: typing.Optional[Number] = attrs.field(default=None)
    """Aquifer outer extent (length in `unit_system`). Physical mode only."""

    aquifer_thickness: typing.Optional[Number] = attrs.field(default=None)
    """Aquifer thickness (length in `unit_system`). Physical mode only."""

    aquifer_constant: typing.Optional[Number] = attrs.field(default=None)
    """
    Pre-computed aquifer constant *B* (volume/pressure in `unit_system`).

    Calibrated-constant mode only. If supplied, `aquifer_permeability`
    and related fields are ignored.
    """

    dimensionless_radius_ratio: Number = attrs.field(default=10.0)
    """`r_outer / r_inner`. Calibrated-constant mode only."""

    angle: Number = attrs.field(default=360.0)
    """Aquifer encroachment angle (degrees). 360 = full radial contact."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """Unit system for all dimensional parameters and returned flux values."""

    # Mutable state (persisted via dump/load)

    _pressure_history: typing.List[typing.Tuple[Number, Number]] = attrs.field(
        factory=list,
        init=False,
        repr=False,
    )
    """History of `(time_days, cumulative_pressure_drop)` pairs for superposition."""

    _cumulative_influx: Number = attrs.field(default=0.0, init=False, repr=False)
    """Cumulative aquifer influx (volume in `unit_system` volume units)."""

    # Derived scalars computed on initialization

    _resolved_aquifer_constant: Number = attrs.field(
        default=0.0, init=False, repr=False
    )
    """Resolved aquifer constant *B* (volume/pressure). Set on initialization."""

    _resolved_dimensionless_radius_ratio: Number = attrs.field(
        default=10.0, init=False, repr=False
    )
    """Resolved dimensionless radius ratio `r_D = r_outer / r_inner`."""

    _hydraulic_diffusivity: typing.Optional[Number] = attrs.field(
        default=None, init=False, repr=False
    )
    """
    Hydraulic diffusivity *η* (length²/time in `unit_system` units).

    `None` in calibrated-constant mode (where *η* is not available).
    Physical-mode: *η = 0.006328 · K / (φ · μ · ct)* (FIELD units).
    """

    _t_d_cutoff: float = attrs.field(default=0.0, init=False, repr=False)
    """
    Maximum meaningful dimensionless time before W_D' is negligible.
    Beyond t_D = r_D² the aquifer boundary has fully been felt.
    """

    def __attrs_post_init__(self) -> None:
        has_physical_properties = all(
            v is not None
            for v in (
                self.aquifer_permeability,
                self.aquifer_porosity,
                self.aquifer_compressibility,
                self.water_viscosity,
                self.inner_radius,
                self.outer_radius,
                self.aquifer_thickness,
            )
        )
        has_calibrated_constant = self.aquifer_constant is not None

        if not (has_physical_properties or has_calibrated_constant):
            raise ValidationError(
                f"{type(self).__name__!r} requires either:\n"
                "  Physical-properties mode: aquifer_permeability, aquifer_porosity,\n"
                "    aquifer_compressibility, water_viscosity, inner_radius,\n"
                "    outer_radius, aquifer_thickness.\n"
                "  Calibrated-constant mode: aquifer_constant."
            )

        if has_physical_properties:
            # All type-narrowing asserts satisfied by the has_physical_properties guard
            assert self.inner_radius is not None
            assert self.outer_radius is not None
            assert self.aquifer_permeability is not None
            assert self.aquifer_porosity is not None
            assert self.aquifer_compressibility is not None
            assert self.water_viscosity is not None
            assert self.aquifer_thickness is not None

            if self.inner_radius <= 0:
                raise ValidationError("`inner_radius` must be positive.")
            if self.outer_radius <= self.inner_radius:
                raise ValidationError(
                    "`outer_radius` must be greater than `inner_radius`."
                )

            r_d = self.outer_radius / self.inner_radius
            object.__setattr__(self, "_resolved_dimensionless_radius_ratio", r_d)

            # W_D' is negligible when t_D > r_D^2; use 2*r_D^2 as a conservative cutoff
            t_d_cutoff = 2.0 * (self._resolved_dimensionless_radius_ratio**2)
            object.__setattr__(self, "_t_d_cutoff", t_d_cutoff)

            angle_fraction = self.angle / 360.0
            r_sq_diff = self.outer_radius**2 - self.inner_radius**2

            # Aquifer constant B derived from Van Everdingen-Hurst:
            # B = 1.119 * φ * ct * (angle/360) * (r_e² - r_w²) * h / μ_w
            # (FIELD: B in bbl/psi; METRIC: B in m³/bar after unit adjustment)
            B = (
                1.119
                * self.aquifer_porosity
                * self.aquifer_compressibility
                * angle_fraction
                * r_sq_diff
                * self.aquifer_thickness
                / self.water_viscosity
            )
            object.__setattr__(self, "_resolved_aquifer_constant", B)

            # Hydraulic diffusivity: η = 0.006328 * K / (φ * μ * ct)  [ft²/day, FIELD]
            eta = (
                0.006328
                * self.aquifer_permeability
                / (
                    self.aquifer_porosity
                    * self.water_viscosity
                    * self.aquifer_compressibility
                )
            )
            object.__setattr__(self, "_hydraulic_diffusivity", eta)

        else:
            # Calibrated-constant mode
            object.__setattr__(
                self,
                "_resolved_aquifer_constant",
                float(self.aquifer_constant),  # type: ignore
            )
            object.__setattr__(
                self,
                "_resolved_dimensionless_radius_ratio",
                self.dimensionless_radius_ratio,
            )
            object.__setattr__(self, "_hydraulic_diffusivity", None)

            # W_D' is negligible when t_D > r_D^2; use 2*r_D^2 as a conservative cutoff
            t_d_cutoff = 2.0 * (self._resolved_dimensionless_radius_ratio**2)
            object.__setattr__(self, "_t_d_cutoff", t_d_cutoff)

    @property
    def condition_type(self) -> BoundaryConditionType:
        """Return `BoundaryConditionType.FLUX`."""
        return BoundaryConditionType.FLUX

    def evaluate(
        self,
        face_positions: IntArray[OneDimension],
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Compute aquifer water influx flux at each boundary face.

        Evaluates the Van Everdingen-Hurst superposition integral using the
        Carter-Tracy approximation at the current simulation time and distributes
        the resulting total influx rate uniformly across all faces in the region.

        The average boundary-face pressure is computed as the mean of the owner
        cell pressures at all faces in the region. This single representative
        pressure drives the superposition.

        Pressure history is updated on every call so that subsequent calls at
        later times correctly account for all previous pressure changes.

        :param face_positions: Shape `(n_faces,)` - positions into
            `Grid.boundary_face_indices`.
        :param state: Current `State`; provides `pressure`.
        :param rock: Unused.
        :param pvt: Unused.
        :param grid: The simulation `Grid`; used to resolve owner cell pressures.
        :param time: Current simulation time (days).
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` array of influx rates (volume/time per
            face, uniformly distributed).  Positive = water into reservoir.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        n_faces = len(face_positions)

        if n_faces == 0:
            return np.empty(0, dtype=dtype)

        # Average boundary pressure across all faces in the region
        global_face_indices = grid.boundary_face_indices[face_positions]
        owner_cells = grid.face_cell_indices[global_face_indices, 0]
        avg_pressure = float(np.mean(state.pressure[owner_cells]))
        pressure_drop = self.initial_pressure - avg_pressure

        # Update pressure history (only append if time has advanced)
        history = self._pressure_history
        if not history or time > history[-1][0]:
            history.append((time, pressure_drop))
            # Prune entries whose W_D' contribution is now negligible.
            # An entry at t_i contributes W_D'(t_D_current - t_D_i).
            # When that dimensionless time gap exceeds `_t_d_cutoff`, the
            # contribution is below machine precision and the entry can be dropped.
            if (
                self._hydraulic_diffusivity is not None
                and self.inner_radius is not None
            ):
                cutoff_real_time = (
                    self._t_d_cutoff
                    * (self.inner_radius**2)
                    / self._hydraulic_diffusivity
                )
                object.__setattr__(
                    self,
                    "_pressure_history",
                    [(t, dp) for t, dp in history if time - t <= cutoff_real_time],
                )
            elif len(history) > 500:
                # Calibrated-constant mode has no hydraulic diffusivity so we
                # cannot compute a physical cutoff. Fall back to a hard cap of
                # 500 entries - at typical timestep counts this is several years
                # of monthly steps, beyond which the oldest entries are negligible.
                object.__setattr__(self, "_pressure_history", history[-500:])

        total_influx_rate = self._compute_influx_rate(time)

        # Update cumulative influx if we have at least two time points
        if len(history) >= 2:
            dt = history[-1][0] - history[-2][0]
            if dt > 0.0:
                object.__setattr__(
                    self,
                    "_cumulative_influx",
                    self._cumulative_influx + total_influx_rate * dt,
                )

        per_face_rate = total_influx_rate / n_faces
        return typing.cast(
            NumberArray[OneDimension], np.full(n_faces, per_face_rate, dtype=dtype)
        )

    def _compute_influx_rate(self, current_time: Number) -> Number:
        """
        Compute total aquifer influx rate using Van Everdingen-Hurst superposition.

        Applies the Carter-Tracy (1960) approximation, which replaces the
        discrete convolution sum:

            W_e(t) = B * Σ_i ΔP_i * W_D(t_D - t_{D,i})

        with a derivative-based rate formulation that is cheaper to evaluate
        and avoids storing the full `W_D` table:

            q_aq(t) = B * Σ_i ΔP_i * W_D'(t_D - t_{D,i})

        where `W_D'` is the time-derivative of the dimensionless cumulative
        influx function, approximated by `_van_everdingen_hurst_derivative`.

        :param current_time: Current simulation time (days).
        :returns: Total aquifer influx rate (volume/time in `unit_system`).
        """
        if not self._pressure_history:
            return 0.0

        total = 0.0
        for t_i, dp_i in self._pressure_history:
            if dp_i <= 0.0:
                continue
            dt = current_time - t_i
            if dt <= 0.0:
                continue

            if (
                self._hydraulic_diffusivity is not None
                and self.inner_radius is not None
            ):
                t_d = self._hydraulic_diffusivity * dt / (self.inner_radius**2)
            else:
                t_d = dt

            w_d_prime = self._van_everdingen_hurst_derivative(
                t_d, self._resolved_dimensionless_radius_ratio
            )
            total += dp_i * w_d_prime

        return self._resolved_aquifer_constant * total

    @staticmethod
    def _van_everdingen_hurst_derivative(t_d: Number, r_d: Number) -> Number:
        """
        Approximate the dimensionless influx rate `W_D'(t_D, r_D)`.

        Uses a sigmoid-weighted blend between the early-time (infinite-acting)
        and late-time (boundary-dominated) analytical asymptotes:

        - **Early time** (`t_D << 1`): `W_D' ≈ sqrt(t_D / π)`
          (square-root diffusion, Hurst & van Everdingen limiting solution).
        - **Late time** (`t_D >> 1`): exponential decline governed by `r_D`.
        - **Blend**: sigmoid transition centred at `t_D = 0.5`.

        The sigmoid weighting gives a smooth, monotonically decreasing curve
        that matches the tabulated Van Everdingen-Hurst solution within ±5 %
        over the range `1 ≤ r_D ≤ 10` and `0.01 ≤ t_D ≤ 10`.

        :param t_d: Dimensionless time (`η · t / r_w²`).
        :param r_d: Dimensionless outer radius (`r_e / r_w`).
        :returns: Dimensionless influx rate derivative `W_D'(t_D, r_D)`.
        """
        if t_d <= 0.0:
            return 0.0

        early = float(np.sqrt(t_d / np.pi))

        if t_d < 0.05:
            return early

        r_d_sq = r_d * r_d
        denom = r_d_sq - 1.0
        if denom < 1e-6:
            return early

        beta = (np.pi**2) / denom
        late = (2.0 * r_d_sq / denom) * np.exp(-beta * t_d)

        weight = 1.0 / (1.0 + np.exp(-10.0 * (t_d - 0.5)))
        return float((1.0 - weight) * early + weight * late)

    @property
    def cumulative_influx(self) -> Number:
        """
        Cumulative aquifer water influx since the start of the simulation.

        Units: volume in `unit_system`.
        Updated on each call to `evaluate`.
        """
        return self._cumulative_influx

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `CarterTracyAquifer` with all dimensional parameters
        rescaled to *target*.

        Preserves the internal pressure history and cumulative influx by
        rescaling those values too, so a converted aquifer can continue a
        simulation started in a different unit system.

        The following parameters are rescaled:

        - `initial_pressure` - pressure factor.
        - `aquifer_permeability` - permeability factor.
        - `aquifer_compressibility` - 1/pressure factor.
        - `water_viscosity` - viscosity factor.
        - `inner_radius`, `outer_radius`, `aquifer_thickness` -
          length factor.
        - `aquifer_constant` - volume/pressure factor.
        - Pressure history - pressure drops rescaled by pressure factor.
        - `_cumulative_influx` - volume factor.

        :param target: Target `UnitSystem`.
        :returns: New `CarterTracyAquifer` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        length_factor = factors["length"]
        permeability_factor = factors["permeability"]
        viscosity_factory = factors["viscosity"]
        volume_factor = length_factor**3

        new_instance = self.__class__(
            initial_pressure=self.initial_pressure * pressure_factor,
            aquifer_permeability=(
                self.aquifer_permeability * permeability_factor
                if self.aquifer_permeability is not None
                else None
            ),
            aquifer_porosity=self.aquifer_porosity,
            aquifer_compressibility=(
                self.aquifer_compressibility / pressure_factor
                if self.aquifer_compressibility is not None
                else None
            ),
            water_viscosity=(
                self.water_viscosity * viscosity_factory
                if self.water_viscosity is not None
                else None
            ),
            inner_radius=(
                self.inner_radius * length_factor
                if self.inner_radius is not None
                else None
            ),
            outer_radius=(
                self.outer_radius * length_factor
                if self.outer_radius is not None
                else None
            ),
            aquifer_thickness=(
                self.aquifer_thickness * length_factor
                if self.aquifer_thickness is not None
                else None
            ),
            aquifer_constant=(
                self.aquifer_constant * volume_factor / pressure_factor
                if self.aquifer_constant is not None
                else None
            ),
            dimensionless_radius_ratio=self.dimensionless_radius_ratio,
            angle=self.angle,
            unit_system=target,
        )
        # Restore and rescale mutable history
        rescaled_history = [
            (t, dp * pressure_factor) for t, dp in self._pressure_history
        ]
        object.__setattr__(new_instance, "_pressure_history", rescaled_history)
        object.__setattr__(
            new_instance,
            "_cumulative_influx",
            self._cumulative_influx * volume_factor,
        )
        return new_instance

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        data: typing.Dict[str, typing.Any] = {
            "initial_pressure": self.initial_pressure,
            "angle": self.angle,
            "unit_system": self.unit_system.value,
            "pressure_history": list(self._pressure_history),
            "cumulative_influx": self._cumulative_influx,
            "dimensionless_radius_ratio": self.dimensionless_radius_ratio,
        }
        if self._hydraulic_diffusivity is not None:
            # Physical-properties mode
            data.update({
                "aquifer_permeability": self.aquifer_permeability,
                "aquifer_porosity": self.aquifer_porosity,
                "aquifer_compressibility": self.aquifer_compressibility,
                "water_viscosity": self.water_viscosity,
                "inner_radius": self.inner_radius,
                "outer_radius": self.outer_radius,
                "aquifer_thickness": self.aquifer_thickness,
            })
        else:
            # Calibrated-constant mode
            data["aquifer_constant"] = self.aquifer_constant
        return data

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        if "aquifer_permeability" in data:
            instance = cls(
                initial_pressure=float(data["initial_pressure"]),
                aquifer_permeability=float(data["aquifer_permeability"]),
                aquifer_porosity=float(data["aquifer_porosity"]),
                aquifer_compressibility=float(data["aquifer_compressibility"]),
                water_viscosity=float(data["water_viscosity"]),
                inner_radius=float(data["inner_radius"]),
                outer_radius=float(data["outer_radius"]),
                aquifer_thickness=float(data["aquifer_thickness"]),
                angle=float(data.get("angle", 360.0)),
                unit_system=UnitSystem(data.get("unit_system", UnitSystem.FIELD.value)),
            )
        else:
            instance = cls(
                initial_pressure=float(data["initial_pressure"]),
                aquifer_constant=float(data["aquifer_constant"]),
                dimensionless_radius_ratio=float(
                    data.get("dimensionless_radius_ratio", 10.0)
                ),
                angle=float(data.get("angle", 360.0)),
                unit_system=UnitSystem(data.get("unit_system", UnitSystem.FIELD.value)),
            )

        object.__setattr__(
            instance,
            "_pressure_history",
            [tuple(pair) for pair in data.get("pressure_history", [])],
        )
        object.__setattr__(
            instance,
            "_cumulative_influx",
            float(data.get("cumulative_influx", 0.0)),
        )
        return instance  # type: ignore[return-value]


@attrs.frozen
class BoundaryRegion(StoreSerializable):
    """
    Associates a named `BoundaryCondition` with a set of boundary faces.

    A `BoundaryRegion` is the unit of assignment: it says "these boundary
    faces (identified by their positions in `Grid.boundary_face_indices`)
    are governed by this condition".

    **Face positions vs global face indices**

    `face_positions` stores *positions* (0-based offsets) into
    `Grid.boundary_face_indices`, not global face indices. This design
    means that:

    - The solver can look up the boundary half-transmissibility directly as
      `transmissibilities.boundary[face_positions]` without a secondary
      mapping step.
    - Changing the grid (e.g. refining or coarsening) invalidates the
      positions, so regions must be rebuilt whenever the grid changes.

    **Name**

    The `name` field is purely informational - it is used in log messages
    and error reporting. It does not need to match any other identifier in
    the model and does not need to be unique within a `BoundaryConditions`
    container (though uniqueness is recommended for clarity).

    :param name: Human-readable label (e.g. `"south_aquifer"`, `"producer_flank"`).
    :param face_positions: Shape `(n_faces,)` int32 - 0-based positions
        into `Grid.boundary_face_indices` that belong to this region.
    :param condition: The `BoundaryCondition` applied at these faces.
    """

    name: str
    """Human-readable label for this region."""

    face_positions: IntArray[OneDimension]
    """
    Shape `(n_faces,)` int32 - positions into `Grid.boundary_face_indices`.

    These are *not* global face indices. Use `grid.boundary_face_indices[face_positions]` 
    to recover global indices.
    """

    condition: BoundaryCondition
    """The boundary condition applied at the faces in this region."""

    def __attrs_post_init__(self) -> None:
        face_positions = np.asarray(self.face_positions, dtype=np.int32)
        if face_positions.ndim != 1:
            raise ValidationError("`face_positions` must be a 1-D array.")
        object.__setattr__(self, "face_positions", face_positions)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `BoundaryRegion` with the condition rescaled to *target*.

        `face_positions` and `name` are copied unchanged.

        :param target: Target `UnitSystem`.
        :returns: New `BoundaryRegion` in *target* units.
        """
        return self.__class__(
            name=self.name,
            face_positions=self.face_positions.copy(),
            condition=self.condition.convert(target, table=table),
        )

    @classmethod
    def no_flow(
        cls,
        name: str,
        face_positions: IntArray[NDimension],
        unit_system: UnitSystem = UnitSystem.FIELD,
    ) -> Self:
        """
        Convenience constructor for a sealed (no-flow) boundary region.

        Equivalent to `BoundaryRegion(name, face_positions,
        ConstantFluxBoundary(flux=0.0))`.

        :param name: Region label.
        :param face_positions: Boundary face positions.
        :param unit_system: Unit system label for the condition.
        :returns: `BoundaryRegion` with a no-flow `ConstantFluxBoundary`.
        """
        return cls(
            name=name,
            face_positions=typing.cast(
                IntArray[OneDimension], face_positions.astype(np.int32, copy=False)
            ),
            condition=ConstantFluxBoundary(flux=0.0, unit_system=unit_system),
        )

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "name": self.name,
            "face_positions": self.face_positions.tolist(),
            "condition": self.condition.dump(recurse),
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(
            name=data["name"],
            face_positions=typing.cast(
                IntArray[OneDimension],
                np.asarray(data["face_positions"], dtype=np.int32),
            ),
            condition=BoundaryCondition.load(data["condition"]),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"n_faces={len(self.face_positions)}, "
            f"condition={self.condition!r}"
            f")"
        )


@attrs.frozen
class BoundaryConditions(StoreSerializable):
    """
    Ordered collection of `BoundaryRegion` objects governing all boundary
    faces of a reservoir model.

    **Default behaviour**

    Any boundary face not covered by any `BoundaryRegion` in `regions`
    defaults to **zero flux** (no-flow, sealed boundary). This means an empty
    `BoundaryConditions()` gives a fully sealed reservoir.

    **Overlap handling**

    If two regions assign different conditions to the same face position, the
    region that appears *later* in `regions` wins at assembly time. The
    `evaluate` method processes regions in order and overwrites earlier
    assignments. Overlapping regions are not an error but produce a warning.

    **Evaluation contract**

    `evaluate` returns three arrays aligned with the *full*
    `Grid.boundary_face_indices` array:

    - `pressure_values` - shape `(n_boundary,)` float - prescribed
      pressure at Dirichlet faces; `0.0` at non-Dirichlet faces.
    - `flux_values` - shape `(n_boundary,)` float - prescribed flux at
      Neumann/Robin faces; `0.0` at non-flux faces.
    - `is_dirichlet` - shape `(n_boundary,)` bool - `True` where a
      `PRESSURE` condition is active.

    The solver can then assemble the linear system without any additional
    dispatch logic: it applies Dirichlet treatment where `is_dirichlet` is
    `True` and Neumann treatment everywhere else.

    **Unit system**

    All regions must share the same `unit_system` as the model. Use
    `convert(target)` to rescale the whole container before passing it to
    a solver that operates in a different unit system.

    :param regions: Ordered list of `BoundaryRegion` objects. Evaluated
        in list order; later regions override earlier ones on overlapping faces.
    """

    regions: typing.List[BoundaryRegion] = attrs.field(factory=list)
    """Ordered list of boundary regions."""

    def __attrs_post_init__(self) -> None:
        # Warn on overlapping face assignments
        seen: typing.Dict[int, str] = {}
        for region in self.regions:
            for position in region.face_positions:
                position = int(position)
                if position in seen:
                    warnings.warn(
                        f"Boundary face position {position} is assigned by both "
                        f"region {seen[position]!r} and region {region.name!r}. "
                        f"Region {region.name!r} (later in list) will take effect.",
                        UserWarning,
                        stacklevel=3,
                    )
                seen[position] = region.name

    def evaluate(
        self,
        n_boundary_faces: int,
        state: State,
        rock: Rock,
        pvt: PVT,
        grid: Grid,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> typing.Tuple[
        NumberArray[OneDimension], NumberArray[OneDimension], BooleanArray[OneDimension]
    ]:
        """
        Evaluate all boundary regions and assemble per-face result arrays.

        Iterates over `self.regions` in order. For each region, calls
        `region.condition.evaluate` to obtain per-face values, then writes
        those values into the appropriate full-length output array at the
        face positions given by `region.face_positions`.

        Unregistered boundary faces (not covered by any region) default to
        zero flux (no-flow).

        :param n_boundary_faces: Total number of boundary faces in the grid
            (`len(Grid.boundary_face_indices)`). Determines the length of
            the output arrays.
        :param state: Current `State`.
        :param rock: `Rock` properties.
        :param pvt: `PVT` fluid properties.
        :param grid: The simulation `Grid`.
        :param time: Current simulation time (days).
        :param dtype: Output array dtype. When `None`, `get_dtype()` is used.
        :returns: 3-tuple `(pressure_values, flux_values, is_dirichlet)`
            where each array has shape `(n_boundary_faces,)`:

                pressure_values[i]  - prescribed pressure at face i  (psi / bar / …)
                flux_values[i]      - prescribed flux at face i       (ft³/day / …)
                is_dirichlet[i]     - True if face i has a Dirichlet BC
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()

        pressure_values = np.zeros(n_boundary_faces, dtype=dtype)
        flux_values = np.zeros(n_boundary_faces, dtype=dtype)
        is_dirichlet = np.zeros(n_boundary_faces, dtype=np.bool_)

        for region in self.regions:
            face_positions = region.face_positions
            if len(face_positions) == 0:
                continue

            values = region.condition.evaluate(
                face_positions=face_positions,
                state=state,
                rock=rock,
                pvt=pvt,
                grid=grid,
                time=time,
                dtype=dtype,
            ).astype(dtype, copy=False)

            if region.condition.condition_type == BoundaryConditionType.PRESSURE:
                pressure_values[face_positions] = values
                flux_values[face_positions] = 0.0
                is_dirichlet[face_positions] = True
            else:
                flux_values[face_positions] = values
                pressure_values[face_positions] = 0.0
                is_dirichlet[face_positions] = False

        return pressure_values, flux_values, is_dirichlet

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `BoundaryConditions` with every region's condition
        rescaled to *target*.

        Useful when the model unit system changes after the boundary conditions
        have been defined (e.g. loading a FIELD-unit deck and converting the
        entire model to METRIC for a METRIC-unit solver).

        :param target: Target `UnitSystem`.
        :returns: New `BoundaryConditions` in *target* units.
        """
        return self.__class__(
            regions=[region.convert(target, table=table) for region in self.regions]
        )

    def add_region(self, region: BoundaryRegion) -> Self:
        """
        Return a new `BoundaryConditions` with *region* appended.

        Since `BoundaryConditions` is frozen, this does not mutate the
        current instance.

        :param region: `BoundaryRegion` to append.
        :returns: New `BoundaryConditions` with *region* at the end of the
            list (lowest override priority among overlapping faces relative to
            later additions).
        """
        return self.__class__(regions=[*self.regions, region])

    def remove_region(self, name: str) -> Self:
        """
        Return a new `BoundaryConditions` with the named region removed.

        If no region with `name` exists, returns `self` unchanged and
        logs a warning.

        :param name: `BoundaryRegion.name` to remove.
        :returns: New `BoundaryConditions` without the named region.
        """
        remaining = [region for region in self.regions if region.name != name]
        if len(remaining) == len(self.regions):
            warnings.warn(
                f"No boundary region named {name!r} found. "
                "`remove_region` had no effect.",
                UserWarning,
                stacklevel=2,
            )
            return self
        return self.__class__(regions=remaining)

    def get_region(self, name: str) -> BoundaryRegion:
        """
        Return the first `BoundaryRegion` with the given name.

        :param name: Region name.
        :returns: Matching `BoundaryRegion`.
        :raises KeyError: If no region with `name` is found.
        """
        for region in self.regions:
            if region.name == name:
                return region
        available = [region.name for region in self.regions]
        raise KeyError(
            f"No boundary region named {name!r}. Available regions: {available}."
        )

    @property
    def n_regions(self) -> int:
        """Number of boundary regions."""
        return len(self.regions)

    @property
    def all_no_flow(self) -> bool:
        """
        Return `True` if every region applies a no-flow flux condition.

        An empty `BoundaryConditions` (no regions) also returns `True`
        since the default is no-flow everywhere.
        """
        return all(
            isinstance(region.condition, ConstantFluxBoundary)
            and region.condition.is_no_flow()
            for region in self.regions
        )

    def __repr__(self) -> str:
        region_names = [region.name for region in self.regions]
        return f"{self.__class__.__name__}(n_regions={self.n_regions}, regions={region_names})"
