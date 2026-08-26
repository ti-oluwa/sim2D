import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.precision import get_dtype
from bores.reservoir.boundary.base import (
    BoundaryCondition,
    BoundaryConditionType,
    _deserialise_boundary_func,
    _serialise_boundary_func,
    boundary_condition,
)
from bores.reservoir.model import Reservoir
from bores.reservoir.state import ReservoirState
from bores.typing import (
    IntArray,
    Number,
    NumberArray,
    OneDimension,
    UnitConversionTable,
    UnitSystem,
)

__all__ = [
    "ConstantFluxBoundary",
    "ConstantPressureBoundary",
    "ProductivityIndexBoundary",
    "TimeDependentFluxBoundary",
]


@boundary_condition
@attrs.frozen(slots=True)
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
        (volume/time in `unit_system`). Default is 0 (no-flow).
    :param unit_system: Unit system for `flux`. Default `FIELD` (ft³/day).
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
        state: ReservoirState,
        reservoir: Reservoir,
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
        table: UnitConversionTable | None = None,
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
        return self.__class__(flux=self.flux * factors["reservoir_rate"], unit_system=target)

    def is_no_flow(self) -> bool:
        """Return `True` if `flux == 0` (sealed boundary)."""
        return self.flux == 0


@boundary_condition
@attrs.frozen(slots=True)
class ConstantPressureBoundary(BoundaryCondition):
    """
    Constant-pressure (Dirichlet) boundary condition.

    Prescribes a fixed pressure at every face in the region. The solver uses
    this as a ghost-cell pressure to compute the face flux via the half-
    transmissibility:

        q_face = T_half * (pressure_boundary - pressure_interior)

    Typical applications:

    - **Strong aquifer** - an aquifer so large that its pressure does not
      change over the simulation period. Set `pressure` to the initial
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
        state: ReservoirState,
        reservoir: Reservoir,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Return `pressure` broadcast to shape `(n_faces,)`.

        :param face_positions: Shape `(n_faces,)`.
        :param state: Unused.
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
        table: UnitConversionTable | None = None,
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
        return self.__class__(pressure=self.pressure * factors["pressure"], unit_system=target)


@boundary_condition
@attrs.frozen(slots=True)
class ProductivityIndexBoundary(BoundaryCondition):
    """
    Robin (mixed) boundary condition using a productivity-index formulation.

    Computes a volumetric flux at each face as:

        q_face[i] = PI_face[i] * (pressure_boundary - p_interior[i])

    where `p_interior[i]` is the pressure of the cell owning boundary face
    `face_positions[i]` from the current `ReservoirState`.

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
      face in the region. This is the simplest case.
    - A **callable** (`alpha_function`) with signature:

    ```python
    alpha_function(
        face_positions: IntArray[OneDimension],
        state: ReservoirState,
        reservoir: Reservoir,
        time: Number,
    ) -> NumberArray[OneDimension]: ... # shape (n_faces,)
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

    alpha_function: (
        typing.Callable[[IntArray[OneDimension], ReservoirState, Reservoir, Number], npt.NDArray]
        | None
    ) = attrs.field(default=None)
    """
    Optional per-face PI callable.

    Signature:

    ```python
    alpha_function(
        face_positions: IntArray[OneDimension],
        state: ReservoirState,
        reservoir: Reservoir,
        time: Number,
    ) -> NumberArray[OneDimension]: ... # shape (n_faces,)
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
        state: ReservoirState,
        reservoir: Reservoir,
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
        :param state: Current `ReservoirState`; provides `pressure`.
        :param time: Current simulation time (days).
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` flux array (volume/time in
            `self.unit_system`).
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        n_faces = len(face_positions)

        # Resolve owner cell pressures for each boundary face
        grid = reservoir.grid
        global_face_indices = grid.boundary_face_indices[face_positions]
        owner_cells = grid.face_cell_indices[global_face_indices, 0]
        p_interior = state.pressure[owner_cells].astype(dtype, copy=False)

        # Resolve per-face PI
        if self.alpha_function is not None:
            productivity_index = np.asarray(
                self.alpha_function(face_positions, state, reservoir, time),
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
        table: UnitConversionTable | None = None,
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
        productivity_index_factor = factors["reservoir_rate"] / factors["pressure"]
        return self.__class__(
            pressure_boundary=self.pressure_boundary * factors["pressure"],
            productivity_index=self.productivity_index * productivity_index_factor,
            alpha_function=self.alpha_function,
            unit_system=target,
        )

    def __dump__(self) -> dict[str, typing.Any]:
        data: dict[str, typing.Any] = {
            "pressure_boundary": self.pressure_boundary,
            "productivity_index": self.productivity_index,
            "unit_system": self.unit_system.value,
        }
        if self.alpha_function is not None:
            data["alpha_function"] = _serialise_boundary_func(self.alpha_function)
        return data

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        alpha = None
        if "alpha_function" in data:
            alpha = _deserialise_boundary_func(data["alpha_function"])
        return cls(
            pressure_boundary=float(data["pressure_boundary"]),
            productivity_index=float(data.get("productivity_index", 1.0)),
            alpha_function=alpha,
            unit_system=UnitSystem(data.get("unit_system", UnitSystem.FIELD.value)),
        )


@boundary_condition
@attrs.frozen(slots=True)
class TimeDependentFluxBoundary(BoundaryCondition):
    """
    Time-varying flux (Neumann) boundary condition driven by a registered
    schedule function.

    The schedule function has the signature:

    ```python
    schedule_function(
        face_positions: IntArray[OneDimension],
        state: ReservoirState,
        reservoir: Reservoir,
        time: Number,
    ) -> NumberArray[OneDimension]: ... # shape (n_faces,)
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
            ReservoirState,
            Reservoir,
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
        state: ReservoirState,
        reservoir: Reservoir,
        time: Number,
    ) -> NumberArray[OneDimension]: ... # shape (n_faces,), volume/time in `unit_system`
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
        state: ReservoirState,
        reservoir: Reservoir,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[OneDimension]:
        """
        Invoke `schedule_function` and return the per-face flux array.

        :param face_positions: Shape `(n_faces,)` - positions into
            `Grid.boundary_face_indices`.
        :param state: Current `ReservoirState`.
        :param time: Current simulation time (days).
        :param dtype: Output dtype.
        :returns: Shape `(n_faces,)` flux array (volume/time).
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        result = self.schedule_function(face_positions, state, reservoir, time)
        return typing.cast(NumberArray[OneDimension], np.asarray(result, dtype=dtype, copy=False))

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
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
        return self.__class__(schedule_function=self.schedule_function, unit_system=target)

    def __dump__(self) -> dict[str, typing.Any]:
        return {
            "schedule_function": _serialise_boundary_func(self.schedule_function),
            "unit_system": self.unit_system.value,
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(
            schedule_function=_deserialise_boundary_func(data["schedule_function"]),
            unit_system=UnitSystem(data.get("unit_system", UnitSystem.FIELD.value)),
        )
