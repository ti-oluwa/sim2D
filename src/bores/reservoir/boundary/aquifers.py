import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import c, get_conversion_factors
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.reservoir.boundary.base import (
    BoundaryCondition,
    BoundaryConditionType,
    boundary_condition,
)
from bores.reservoir.model import ReservoirModel
from bores.reservoir.state.base import ReservoirState
from bores.typing import (
    IntArray,
    NDimension,
    Number,
    NumberArray,
    UnitConversionTable,
    UnitSystem,
)

__all__ = ["CarterTracyAquifer"]


@boundary_condition
@attrs.define
class CarterTracyAquifer(BoundaryCondition):
    """
    Transient aquifer water influx boundary condition using the Carter-Tracy
    (1960) recursive approximation to the Van Everdingen-Hurst solution.

    Computes cumulative and incremental water influx from a finite radial
    aquifer using the Carter-Tracy recurrence, which avoids the superposition
    convolution of the original Van Everdingen-Hurst method while preserving
    its physical basis. The recurrence operates in O(1) memory and O(1) CPU
    per time step regardless of simulation length.

    **Carter-Tracy recurrence** (Carter & Tracy, 1960, Eq. 3):

        (We)_n = (We)_{n-1}
                 + [(tD)_n - (tD)_{n-1}]
                   * [aquifer_constant*ΔP_n - (We)_{n-1} * pD'_n]
                   / [pD_n - (tD)_{n-1} * pD'_n]

    where:

    - `(We)_n` - cumulative influx at time step n (reservoir volume).
    - `aquifer_constant` - aquifer constant (reservoir volume / pressure).
    - `ΔP_n` - current cumulative pressure drop (initial - current pressure).
    - `tD_n` - current dimensionless time.
    - `pD_n` - dimensionless pressure at `tD_n`.
    - `pD'_n` - derivative of dimensionless pressure at `tD_n`.

    The influx rate returned by `evaluate` is the incremental influx divided
    by the elapsed time step:

        q_n = [(We)_n - (We)_{n-1}] / Δt

    **Dimensionless time** (FIELD units, Carter & Tracy 1960, Eq. 1):

        tD = 6.328e-3 * k * t / (φ * μ_w * ct * r_w²)

    where `r_w` is the inner (reservoir-aquifer contact) radius in ft and
    `t` is in days.

    **Aquifer constant aquifer_constant** (FIELD units, Carter & Tracy 1960, Eq. 2):

        aquifer_constant = 1.119 * φ * ct * (r_e² - r_w²) * h * f

    where `f = θ/360` is the encroachment angle fraction and `r_e` is the
    outer aquifer radius in ft.

    **pD and pD' approximations**:

    For `tD ≤ 100`, the Edwardson et al. (1962) polynomial:

        pD = (370.529*√tD + 137.582*tD + 5.69549*tD^1.5)
             / (328.834 + 265.488*√tD + 45.2157*tD + tD^1.5)

        pD' = E / F
        E = 716.441 + 46.7984*√tD + 270.038*tD + 71.0098*tD^1.5
        F = 1296.86*√tD + 1204.73*tD + 618.618*tD^1.5 + 538.072*tD² + 142.41*tD^2.5

    For `tD > 100`, the logarithmic approximation:

        pD  = 0.5 * (ln(tD) + 0.80907)
        pD' = 1.0 / (2 * tD)

    **Two construction modes**:

    *Physical-properties mode* - supply `aquifer_permeability`,
    `aquifer_porosity`, `aquifer_compressibility`, `water_viscosity`,
    `inner_radius`, `outer_radius`, `aquifer_thickness`. aquifer_constant and η are
    derived automatically in FIELD units then stored in `unit_system` units.

    *Calibrated-constant mode* - supply `aquifer_constant` and
    `dimensionless_radius_ratio`. Useful when parameters are history-matched
    rather than measured directly.

    **Unit system**:

    All user-supplied dimensional inputs must be in `unit_system`. Internally,
    the FIELD-unit constants (1.119, 6.328e-3) are applied after converting
    inputs to FIELD; aquifer_constant and η are then converted back to `unit_system` for
    storage. The rate returned by `evaluate` is in reservoir volume / time
    in `unit_system`.

    **References**:

    - Carter, R.D. & Tracy, G.W. (1960). *An Improved Method for Calculating
      Water Influx.* Trans. AIME, 219, 415-417.
    - Edwardson, M.J. et al. (1962). *Calculation of Formation Temperature
      Disturbances Caused by Mud Circulation.* JPT, 14(4), 416-426.
      (source of the pD polynomial approximations)
    - Ahmed, T. (2010). *ReservoirModel Engineering Handbook*, 4th ed.
      Gulf Professional Publishing. (Carter-Tracy chapter.)
    """

    __type__ = "carter_tracy_aquifer"

    initial_pressure: Number
    """Initial aquifer / reservoir pressure in `unit_system` pressure units."""

    aquifer_permeability: typing.Optional[Number] = attrs.field(default=None)
    """Aquifer permeability. Physical mode only."""

    aquifer_porosity: typing.Optional[Number] = attrs.field(default=None)
    """Aquifer porosity (fraction). Physical mode only."""

    aquifer_compressibility: typing.Optional[Number] = attrs.field(default=None)
    """Total aquifer compressibility. Physical mode only."""

    water_viscosity: typing.Optional[Number] = attrs.field(default=None)
    """Water viscosity at reservoir conditions. Physical mode only."""

    inner_radius: typing.Optional[Number] = attrs.field(default=None)
    """ReservoirModel-aquifer contact radius. Physical mode only."""

    outer_radius: typing.Optional[Number] = attrs.field(default=None)
    """Outer aquifer extent. Physical mode only."""

    aquifer_thickness: typing.Optional[Number] = attrs.field(default=None)
    """Aquifer thickness. Physical mode only."""

    aquifer_constant: typing.Optional[Number] = attrs.field(default=None)
    """
    Pre-computed or history-matched aquifer constant aquifer_constant
    (reservoir volume / pressure in `unit_system`).
    Calibrated-constant mode only.
    """

    dimensionless_radius_ratio: Number = attrs.field(default=10.0)
    """`r_e / r_w`. Calibrated-constant mode only."""

    angle: Number = attrs.field(default=360.0)
    """Aquifer encroachment angle in degrees."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """Unit system for all dimensional parameters and returned flux values."""

    # Derived scalars

    _resolved_aquifer_constant: Number = attrs.field(
        default=0.0, init=False, repr=False
    )
    """
    Resolved aquifer constant aquifer_constant in [reservoir volume / pressure]
    in `unit_system` units. Set on initialization.
    """

    _resolved_dimensionless_radius_ratio: Number = attrs.field(
        default=10.0, init=False, repr=False
    )
    """
    Resolved `r_e / r_w`. Set on initialization.
    """

    _hydraulic_diffusivity: typing.Optional[Number] = attrs.field(
        default=None, init=False, repr=False
    )
    """
    Hydraulic diffusivity η in [length² / time] in `unit_system` units.
    Used to compute dimensionless time: `tD = η * t / r_w²`.
    `None` in calibrated-constant mode.
    Set on initialization.
    """

    # Recursive state

    _previous_time: Number = attrs.field(default=0.0, init=False, repr=False)
    """
    Simulation time at the previous `evaluate` call, in `unit_system`
    time units (days for FIELD/METRIC, hours for LAB, seconds for SI).
    """

    _previous_pressure: Number = attrs.field(default=0.0, init=False, repr=False)
    """
    Average boundary pressure at the previous `evaluate` call, in
    `unit_system` pressure units. Initialised to `initial_pressure`.
    """

    _previous_dimensionless_time: Number = attrs.field(
        default=0.0, init=False, repr=False
    )
    """
    Dimensionless time `tD` at the previous `evaluate` call.
    Initialised to 0.0.
    """

    _previous_cumulative_influx: Number = attrs.field(
        default=0.0, init=False, repr=False
    )
    """
    Cumulative aquifer influx `We` at the previous `evaluate` call,
    in `unit_system` reservoir volume units. Initialised to 0.0.
    """

    def __attrs_post_init__(self) -> None:
        has_physical = all(
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
        has_calibrated = self.aquifer_constant is not None

        if not (has_physical or has_calibrated):
            raise ValidationError(
                f"{type(self).__name__!r} requires either:\n"
                "  Physical-properties mode: aquifer_permeability, aquifer_porosity,\n"
                "    aquifer_compressibility, water_viscosity, inner_radius,\n"
                "    outer_radius, aquifer_thickness.\n"
                "  Calibrated-constant mode: aquifer_constant."
            )

        if has_physical:
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

            # Convert inputs to FIELD for physics constants
            # 1.119 and 6.328e-3 are FIELD-unit constants from Carter & Tracy
            # (1960). All inputs are rescaled to FIELD, physics applied, then
            # results converted back to unit_system for storage.
            if self.unit_system != UnitSystem.FIELD:
                to_field = get_conversion_factors(self.unit_system, UnitSystem.FIELD)
                r_w_ft = self.inner_radius * to_field["length"]
                r_e_ft = self.outer_radius * to_field["length"]
                height_ft = self.aquifer_thickness * to_field["length"]
                compressibility_psi = (
                    self.aquifer_compressibility * to_field["compressibility"]
                )
                permeability_md = self.aquifer_permeability * to_field["permeability"]
                viscosity_cp = self.water_viscosity * to_field["viscosity"]
                from_field = get_conversion_factors(UnitSystem.FIELD, self.unit_system)
            else:
                r_w_ft = self.inner_radius
                r_e_ft = self.outer_radius
                height_ft = self.aquifer_thickness
                compressibility_psi = self.aquifer_compressibility
                permeability_md = self.aquifer_permeability
                viscosity_cp = self.water_viscosity
                from_field = None

            r_d = r_e_ft / r_w_ft
            object.__setattr__(self, "_resolved_dimensionless_radius_ratio", r_d)

            angle_fraction = self.angle / 360.0

            # aquifer_constant = 1.119 * φ * ct * (r_e² - r_w²) * h * f
            # Units: bbl/psi  (FIELD)
            # Note: the formula uses r_e² - r_w²; for large aquifers r_e² >> r_w²
            # so the approximation aquifer_constant ≈ 1.119*φ*ct*r_e²*h*f is sometimes used, but
            # we use the exact form here.
            aquifer_constant_bbl_per_psi = (
                1.119
                * self.aquifer_porosity
                * compressibility_psi
                * (r_e_ft**2 - r_w_ft**2)
                * height_ft
                * angle_fraction
            )

            # Convert aquifer_constant from bbl/psi (FIELD) to [reservoir volume / pressure]
            # in unit_system.
            # 1 reservoir bbl = BARRELS_TO_CUBIC_FEET ft³.
            # The `volume` factor in unit conversion table is ft³ -> user volume.
            # So: aquifer_constant = B_bbl_psi * (bbl->ft³) * (ft³->user_vol) / (psi->user_pres)
            bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET  # 5.614583
            aquifer_constant_ft3_per_psi = aquifer_constant_bbl_per_psi * bbl_to_ft3

            if from_field is not None:
                aquifer_constant = (
                    aquifer_constant_ft3_per_psi
                    * from_field["volume"]
                    / from_field["pressure"]
                )
            else:
                aquifer_constant = aquifer_constant_ft3_per_psi

            object.__setattr__(self, "_resolved_aquifer_constant", aquifer_constant)

            # η = 6.328e-3 * k / (φ * μ * ct)   [ft²/day, FIELD]
            # Carter & Tracy (1960) Eq. 1 gives:
            #   tD = 6.328e-3 * k * t / (φ * μ * ct * r_w²)
            #      = (η / r_w²) * t
            # so η = 6.328e-3 * k / (φ * μ * ct)
            hydraulic_diffusivity_ft2_per_day = (
                6.328e-3
                * permeability_md
                / (self.aquifer_porosity * viscosity_cp * compressibility_psi)
            )

            # Convert η from ft²/day to [length² / time] in unit_system.
            if from_field is not None:
                hydraulic_diffusivity = (
                    hydraulic_diffusivity_ft2_per_day
                    * (from_field["length"] ** 2)
                    / from_field["time"]
                )
            else:
                hydraulic_diffusivity = hydraulic_diffusivity_ft2_per_day

            object.__setattr__(self, "_hydraulic_diffusivity", hydraulic_diffusivity)

        else:
            # Calibrated-constant mode: aquifer_constant supplied directly in unit_system
            object.__setattr__(
                self, "_resolved_aquifer_constant", self.aquifer_constant
            )
            object.__setattr__(
                self,
                "_resolved_dimensionless_radius_ratio",
                self.dimensionless_radius_ratio,
            )
            object.__setattr__(self, "_hydraulic_diffusivity", None)

        # Initialise recursive state
        object.__setattr__(self, "_previous_time", 0.0)
        object.__setattr__(self, "_previous_pressure", float(self.initial_pressure))
        object.__setattr__(self, "_previous_dimensionless_time", 0.0)
        object.__setattr__(self, "_previous_cumulative_influx", 0.0)

    @property
    def condition_type(self) -> BoundaryConditionType:
        """Return ``BoundaryConditionType.FLUX``."""
        return BoundaryConditionType.FLUX

    @staticmethod
    def _dimensionless_pressure(t_d: Number) -> Number:
        """
        Dimensionless pressure `pD(tD)` for an infinite-acting radial aquifer.

        Uses the Edwardson et al. (1962) polynomial for `tD ≤ 100` and the
        logarithmic approximation for `tD > 100`.

        Edwardson polynomial (Equation 4 of Carter-Tracy seminar document,
        attributed to Edwardson et al., 1962):

            pD = (370.529*√tD + 137.582*tD + 5.69549*tD^1.5)
                 / (328.834 + 265.488*√tD + 45.2157*tD + tD^1.5)

        Logarithmic approximation (valid for tD > 100):

            pD = 0.5 * (ln(tD) + 0.80907)

        Both approximations are from the Carter-Tracy (1960) original paper
        and its referenced polynomial fits.

        :param t_d: Dimensionless time (dimensionless).
        :returns: Dimensionless pressure (dimensionless).
        """
        if t_d <= 0.0:
            return 0.0
        if t_d > 100.0:
            return 0.5 * ((np.log(t_d)) + 0.80907)

        sqrt_td = np.sqrt(t_d)
        td_15 = t_d**1.5
        numerator = 370.529 * sqrt_td + 137.582 * t_d + 5.69549 * td_15
        denominator = 328.834 + 265.488 * sqrt_td + 45.2157 * t_d + td_15
        return numerator / denominator

    @staticmethod
    def _dimensionless_pressure_derivative(t_d: Number) -> Number:
        """
        Derivative of dimensionless pressure `pD'(tD)` for an infinite-acting
        radial aquifer.

        Uses the Edwardson et al. (1962) polynomial ratio for `tD ≤ 100`
        and the analytical derivative of the logarithmic approximation for
        `tD > 100`.

        Polynomial (Equations 5, E, F of Carter-Tracy seminar document):

            pD' = E / F
            E = 716.441 + 46.7984*√tD + 270.038*tD + 71.0098*tD^1.5
            F = 1296.86*√tD + 1204.73*tD + 618.618*tD^1.5 + 538.072*tD² + 142.41*tD^2.5

        Logarithmic derivative (tD > 100):

            pD' = 1 / (2 * tD)

        :param t_d: Dimensionless time (dimensionless).
        :returns: Dimensionless pressure derivative (dimensionless).
        """
        if t_d <= 0.0:
            return 0.0
        if t_d > 100.0:
            return 1.0 / (2.0 * t_d)

        sqrt_td = np.sqrt(t_d)
        td_15 = t_d**1.5
        td_2 = t_d**2.0
        td_25 = t_d**2.5
        E = 716.441 + 46.7984 * sqrt_td + 270.038 * t_d + 71.0098 * td_15
        F = (
            1296.86 * sqrt_td
            + 1204.73 * t_d
            + 618.618 * td_15
            + 538.072 * td_2
            + 142.41 * td_25
        )
        if F == 0.0:
            return 0.0
        return E / F

    def _compute_cumulative_influx(
        self, current_t_d: Number, current_delta_p: Number
    ) -> Number:
        """
        Apply the Carter-Tracy (1960) recurrence to compute cumulative influx
        at the current dimensionless time.

        Implements exactly Equation 3 of Carter & Tracy (1960):

            (We)_n = (We)_{n-1}
                     + [(tD)_n - (tD)_{n-1}]
                       * [aquifer_constant * ΔP_n - (We)_{n-1} * pD'_n]
                       / [pD_n - (tD)_{n-1} * pD'_n]

        :param current_t_d: Current dimensionless time `(tD)_n`.
        :param current_delta_p: Current cumulative pressure drop
            `P_initial - P_current` in `unit_system` pressure units.
        :returns: Current cumulative influx `(We)_n` in `unit_system`
            reservoir volume units.
        """
        previous_we = self._previous_cumulative_influx
        previous_t_d = self._previous_dimensionless_time
        aquifer_constant = self._resolved_aquifer_constant

        current_p_d = self._dimensionless_pressure(current_t_d)
        current_p_d_prime = self._dimensionless_pressure_derivative(current_t_d)

        delta_t_d = current_t_d - previous_t_d
        if delta_t_d <= 0.0:
            return previous_we

        denominator = current_p_d - previous_t_d * current_p_d_prime
        if abs(denominator) < 1e-30:
            # Degenerate: pD ≈ tD_{n-1} * pD'; return previous value unchanged
            return previous_we

        numerator_bracket = (
            aquifer_constant * current_delta_p - previous_we * current_p_d_prime
        )
        return previous_we + delta_t_d * (numerator_bracket / denominator)

    def evaluate(
        self,
        face_positions: IntArray[NDimension],
        state: ReservoirState,
        reservoir: ReservoirModel,
        time: Number,
        dtype: npt.DTypeLike = None,
    ) -> NumberArray[NDimension]:
        """
        Compute aquifer water influx rate at each boundary face using the
        Carter-Tracy recursive approximation.

        **Algorithm** (per Carter & Tracy, 1960):

        1. Compute average boundary pressure from owner cell pressures.
        2. Compute cumulative pressure drop: `ΔP_n = P_initial - P_n`.
        3. Compute current dimensionless time `tD_n`.
        4. Apply Carter-Tracy recurrence to obtain cumulative influx `We_n`.
        5. Compute incremental influx rate:
           `q = (We_n - We_{n-1}) / Δt`.
        6. Update recursive state.
        7. Return `q / n_faces` uniformly distributed across boundary faces.

        When `Δt ≤ 0` (first call or repeated call at same time), returns
        zero to avoid division by zero.

        :param face_positions: Shape `(n_faces,)` int32 - positions into
            `Grid.boundary_face_indices`.
        :param state: Current `ReservoirState`; provides `pressure`.
        :param rock: Unused.
        :param pvt: Unused.
        :param grid: The simulation `Grid`; resolves owner cell pressures.
        :param time: Current simulation time in `unit_system` time units
            (days for FIELD/METRIC, hours for LAB, seconds for SI).
        :param dtype: Output array dtype; defaults to `get_dtype()`.
        :returns: Shape `(n_faces,)` array of influx rates in
            [reservoir volume / time / face] in `unit_system`.
            Positive = water into reservoir.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        n_faces = len(face_positions)

        if n_faces == 0:
            return typing.cast(NumberArray[NDimension], np.empty(0, dtype=dtype))

        # Average boundary pressure from owner cell pressures
        grid = reservoir.grid
        global_face_indices = grid.boundary_face_indices[face_positions]
        owner_cells = grid.face_cell_indices[global_face_indices, 0]
        average_pressure = np.mean(state.pressure[owner_cells])

        # Cumulative pressure drop ΔP_n = P_initial - P_current
        current_delta_p = self.initial_pressure - average_pressure
        # Dimensionless time tD_n
        dt = time - self._previous_time

        if self._hydraulic_diffusivity is not None and self.inner_radius is not None:
            # tD = η * t / r_w²
            current_t_d = self._hydraulic_diffusivity * time / (self.inner_radius**2)
        else:
            # Calibrated-constant mode: tD is not computable without η.
            # Use elapsed real time as a proxy dimensionless time.
            current_t_d = time

        # Carter-Tracy recurrence: cumulative influx We_n
        we_n = self._compute_cumulative_influx(current_t_d, current_delta_p)

        # Incremental rate q = ΔWe / Δt
        if dt > 0.0:
            rate = (we_n - self._previous_cumulative_influx) / dt
        else:
            rate = 0.0

        # Update recursive state
        object.__setattr__(self, "_previous_time", time)
        object.__setattr__(self, "_previous_pressure", average_pressure)
        object.__setattr__(self, "_previous_dimensionless_time", current_t_d)
        object.__setattr__(self, "_previous_cumulative_influx", we_n)

        per_face_rate = rate / n_faces
        return typing.cast(
            NumberArray[NDimension], np.full(n_faces, per_face_rate, dtype=dtype)
        )

    @property
    def cumulative_influx(self) -> Number:
        """
        Cumulative aquifer water influx since simulation start.

        Units: reservoir volume in `unit_system` (ft³ in FIELD, m³ in
        METRIC/SI, cm³ in LAB).

        Returns the value from the previous `evaluate` call - i.e. the
        cumulative influx up to and including the last completed time step.
        Zero before the first call.
        """
        return self._previous_cumulative_influx

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

        **What is rescaled**:

        - `initial_pressure` - pressure factor.
        - `aquifer_permeability` - permeability factor.
        - `aquifer_compressibility` - compressibility (1/pressure) factor.
        - `water_viscosity` - viscosity factor.
        - `inner_radius`, `outer_radius`, `aquifer_thickness` - length factor.
        - `aquifer_constant` - reservoir volume / pressure factor.
        - `_previous_pressure` - pressure factor.
        - `_previous_cumulative_influx` - reservoir volume factor.

        **What is not rescaled**:

        - `_previous_time` and `_previous_dimensionless_time` - time and
          dimensionless time remain unchanged (dimensionless time is always
          dimensionless; real time must stay in the original unit to maintain
          continuity of the time axis in a restarted simulation; the caller
          is responsible for converting the simulation clock if needed).
        - `angle`, `dimensionless_radius_ratio`, `aquifer_porosity` - dimensionless.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `CarterTracyAquifer` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        length_factor = factors["length"]
        permeability_factor = factors["permeability"]
        viscosity_factor = factors["viscosity"]
        volume_factor = factors["volume"]
        compressibility_factor = factors["compressibility"]

        new_instance = self.__class__(
            initial_pressure=self.initial_pressure * pressure_factor,
            aquifer_permeability=(
                self.aquifer_permeability * permeability_factor
                if self.aquifer_permeability is not None
                else None
            ),
            aquifer_porosity=self.aquifer_porosity,
            aquifer_compressibility=(
                self.aquifer_compressibility * compressibility_factor
                if self.aquifer_compressibility is not None
                else None
            ),
            water_viscosity=(
                self.water_viscosity * viscosity_factor
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

        # Rescale recursive state. Pressure and cumulative volume only
        object.__setattr__(
            new_instance,
            "_previous_pressure",
            self._previous_pressure * pressure_factor,
        )
        object.__setattr__(
            new_instance,
            "_previous_cumulative_influx",
            self._previous_cumulative_influx * volume_factor,
        )
        # Carry over time and dimensionless time unchanged
        object.__setattr__(new_instance, "_previous_time", self._previous_time)
        object.__setattr__(
            new_instance,
            "_previous_dimensionless_time",
            self._previous_dimensionless_time,
        )
        return new_instance

    def __dump__(self) -> typing.Dict[str, typing.Any]:
        data: typing.Dict[str, typing.Any] = {
            "initial_pressure": self.initial_pressure,
            "angle": self.angle,
            "unit_system": self.unit_system.value,
            "previous_time": self._previous_time,
            "previous_pressure": self._previous_pressure,
            "previous_dimensionless_time": self._previous_dimensionless_time,
            "previous_cumulative_influx": self._previous_cumulative_influx,
            "dimensionless_radius_ratio": self.dimensionless_radius_ratio,
        }
        if self._hydraulic_diffusivity is not None:
            data.update(
                {
                    "aquifer_permeability": self.aquifer_permeability,
                    "aquifer_porosity": self.aquifer_porosity,
                    "aquifer_compressibility": self.aquifer_compressibility,
                    "water_viscosity": self.water_viscosity,
                    "inner_radius": self.inner_radius,
                    "outer_radius": self.outer_radius,
                    "aquifer_thickness": self.aquifer_thickness,
                }
            )
        else:
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
            instance, "_previous_time", float(data.get("previous_time", 0.0))
        )
        object.__setattr__(
            instance,
            "_previous_pressure",
            float(data.get("previous_pressure", instance.initial_pressure)),
        )
        object.__setattr__(
            instance,
            "_previous_dimensionless_time",
            float(data.get("previous_dimensionless_time", 0.0)),
        )
        object.__setattr__(
            instance,
            "_previous_cumulative_influx",
            float(data.get("previous_cumulative_influx", 0.0)),
        )
        return instance  # type: ignore[return-value]
