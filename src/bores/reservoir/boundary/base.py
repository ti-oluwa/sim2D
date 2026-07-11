import enum
import functools
import logging
import threading
import typing

import numpy.typing as npt
from typing_extensions import Self

from bores.constants import c, get_conversion_factors
from bores.errors import DeserializationError, SerializationError, ValidationError
from bores.precision import get_dtype
from bores.reservoir.model import Reservoir
from bores.reservoir.state import ReservoirState
from bores.serde.base import Serializable
from bores.serde.registry import make_serializable_type_registrar
from bores.serde.stores import StoreSerializable
from bores.typing import (
    BooleanArray,
    IntArray,
    NDimension,
    Number,
    NumberArray,
    OneDimension,
    UnitConversionTable,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BoundaryCondition",
    "BoundaryConditionType",
    "ParameterizedBoundaryFunction",
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
    either fluxes or pressures. Registering it by name allows
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
    :param name: Registration key. Defaults to `func.__name__`.
    :param override: When `True`, silently replaces an existing registration
        under the same key. When `False` (default), raises `ValidationError`
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


def _serialise_boundary_func(func: typing.Callable) -> typing.Dict[str, typing.Any]:
    """
    Serialise a boundary function to a JSON-compatible dict.

    Supports three kinds of callable:

    - **Registered function** - serialised as `{"type": "registered", "name": ...}`.
    - **`functools.partial`** - serialised recursively with `args` and `kwargs`.
    - **`ParameterizedBoundaryFunction`** - serialised via its own `dump` method.

    :param func: The callable to serialise.
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
            "func": _serialise_boundary_func(func.func),
            "args": list(func.args),
            "kwargs": dict(func.keywords),
        }

    if isinstance(func, ParameterizedBoundaryFunction):
        return {"type": "parameterized", "data": func.dump()}

    raise SerializationError(
        f"Cannot serialise boundary function {func!r}. "
        "Register it with `@boundary_function` first."
    )


def _deserialise_boundary_func(
    data: typing.Mapping[str, typing.Any],
) -> typing.Callable:
    """
    Reconstruct a boundary function from a dict produced by
    `_serialise_boundary_func`.

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
        base = _deserialise_boundary_func(data["func"])
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
    dict. On call, the stored parameters are merged with any additional keyword
    arguments supplied by the caller (caller kwargs take precedence).

    This is the preferred way to combine a generic registered function with
    site-specific parameters when you need the boundary condition to survive
    serialisation round-trips:

    ```python
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
    ```

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

    def __dump__(self) -> typing.Dict[str, typing.Any]:
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
        / Pa depending on `unit_system`) at each boundary face. The solver
        applies a Dirichlet constraint: it adds `T_face * p_boundary` to the
        RHS and `T_face` to the diagonal of the flow equation for each face.

    `FLUX`
        The `evaluate` method returns a volumetric flow rate into the
        reservoir (`ft³/day` / `m³/day` / etc.) at each boundary face. The solver
        applies a Neumann source term: it adds the flux directly to the RHS of
        the cell owning the face. A flux of zero is a sealed (no-flow) boundary.
    """

    PRESSURE = "pressure"
    FLUX = "flux"


class BoundaryCondition(StoreSerializable):
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

    unit_system: UnitSystem

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
        state: ReservoirState,
        reservoir: Reservoir,
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
        :param state: Current `ReservoirState`.
        :param time: Current simulation time in days.
        :param dtype: Output array dtype. When `None`, `get_dtype()` is used.
        :returns: Shape `(n_faces,)` array of pressures or fluxes.
        """
        raise NotImplementedError

    def __call__(
        self,
        face_positions: IntArray[OneDimension],
        state: ReservoirState,
        reservoir: Reservoir,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Shorthand for `evaluate`. Makes the condition directly callable.

        :param face_positions: Shape `(n_faces,)` int32.
        :param state: Current `ReservoirState`.
        :param time: Current simulation time in days.
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` array from `evaluate`.
        """
        return self.evaluate(face_positions, state, reservoir, time, dtype)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        raise NotImplementedError


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
