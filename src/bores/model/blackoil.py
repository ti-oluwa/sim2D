"""
Reservoir model: Top-level assembly of grid, rock, pvt, metadata and dynamic state.

`BlackOilModel` is the single entity passed (implicitly or explicitly) between the deck reader, the
initialisation routines, and the flow solver.
"""

import typing

import attrs
from typing_extensions import Self

from bores.constants import (
    UnitConversionTable,
    build_unit_conversion_table,
    get_conversion_factors,
)
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.model.properties import PVT, Hysteresis, Regions, Rock, State
from bores.model.transmissibility import (
    ConnectionTransmissibilities,
    compute_connection_transmissibilities,
    get_face_transmissibility_map,
)
from bores.serialization import Serializable
from bores.typing import CellArray, Number, UnitSystem

__all__ = ["BlackOilModel"]


def _validate_state(state: State, n_cells: int) -> None:
    """
    Verify that all mandatory per-cell arrays in `state` have length
    `n_cells`.

    Optional EOR arrays (`solvent_concentration` etc.) are validated only
    when non-empty.

    :param state: `State` to validate.
    :param n_cells: Expected array length.
    :raises ValidationError: On any mismatch.
    """
    mandatory = {
        "state.pressure": state.pressure,
        "state.oil_saturation": state.oil_saturation,
        "state.water_saturation": state.water_saturation,
        "state.gas_saturation": state.gas_saturation,
        "state.oil_mass": state.oil_mass,
        "state.water_mass": state.water_mass,
        "state.free_gas_mass": state.free_gas_mass,
        "state.dissolved_gas_mass_in_oil": state.dissolved_gas_mass_in_oil,
        "state.dissolved_gas_mass_in_water": state.dissolved_gas_mass_in_water,
        "state.vaporized_oil_mass_in_gas": state.vaporized_oil_mass_in_gas,
        "state.solution_gor": state.solution_gor,
        "state.vaporized_oil_ratio": state.vaporized_oil_ratio,
        "state.gas_solubility_in_water": state.gas_solubility_in_water,
        "state.oil_bubble_point_pressure": state.oil_bubble_point_pressure,
        "state.gas_dew_point_pressure": state.gas_dew_point_pressure,
        "state.water_bubble_point_pressure": state.water_bubble_point_pressure,
        "state.gas_solubility_in_water": state.gas_solubility_in_water,
    }
    for name, arr in mandatory.items():
        if arr.shape != (n_cells,):
            raise ValidationError(
                f"`{name}` has shape {arr.shape}; expected ({n_cells},)."
            )

    for name, arr in (("state.solvent_concentration", state.solvent_concentration),):
        if arr.size > 0 and arr.shape != (n_cells,):
            raise ValidationError(
                f"`{name}` has shape {arr.shape}; expected ({n_cells},) or empty."
            )

    if state.hysteresis is not None:
        _validate_hysteresis(state.hysteresis, n_cells)


def _validate_hysteresis(hysteresis: Hysteresis, n_cells: int) -> None:
    """
    Verify that all per-cell arrays in `hysteresis` have length `n_cells`.

    :param hysteresis: `Hysteresis` to validate.
    :param n_cells: Expected array length.
    :raises ValidationError: On any mismatch.
    """
    checks = {
        "hysteresis.max_water_saturation": hysteresis.max_water_saturation,
        "hysteresis.max_gas_saturation": hysteresis.max_gas_saturation,
        "hysteresis.water_imbibition_flag": hysteresis.water_imbibition_flag,
        "hysteresis.gas_imbibition_flag": hysteresis.gas_imbibition_flag,
        "hysteresis.water_reversal_saturation": hysteresis.water_reversal_saturation,
        "hysteresis.gas_reversal_saturation": hysteresis.gas_reversal_saturation,
    }
    for name, arr in checks.items():
        if arr.shape != (n_cells,):
            raise ValidationError(
                f"`{name}` has shape {arr.shape}; expected ({n_cells},)."
            )


def _validate_rock(rock: Rock, n_cells: int) -> None:
    """
    Verify that all per-cell arrays in `rock` have length `n_cells`.

    :param rock: `Rock` to validate.
    :param n_cells: Expected array length.
    :raises ValidationError: On any mismatch.
    """
    checks = {
        "rock.porosity": rock.porosity,
        "rock.temperature": rock.temperature,
        "rock.absolute_permeability.x": rock.absolute_permeability.x,
        "rock.absolute_permeability.y": rock.absolute_permeability.y,
        "rock.absolute_permeability.z": rock.absolute_permeability.z,
        "rock.net_to_gross": rock.net_to_gross,
        "rock.connate_water_saturation": rock.connate_water_saturation,
        "rock.irreducible_water_saturation": rock.irreducible_water_saturation,
        "rock.residual_oil_saturation_water_flood": (
            rock.residual_oil_saturation_water_flood
        ),
        "rock.residual_oil_saturation_gas_flood": (
            rock.residual_oil_saturation_gas_flood
        ),
        "rock.residual_gas_saturation": rock.residual_gas_saturation,
    }
    for name, arr in checks.items():
        if arr.shape != (n_cells,):
            raise ValidationError(
                f"`{name}` has shape {arr.shape}; expected ({n_cells},)."
            )


class BlackOilModel(
    Serializable,
    fields={
        "grid": Grid,
        "rock": Rock,
        "pvt": PVT,
        "state": State,
        "regions": typing.Optional[Regions],
        "datum_depth": typing.Optional[Number],
        "unit_system": typing.Optional[UnitSystem],
    },
):
    """
    Reservoir model for black-oil simulation.

    Binds a polyhedral `Grid` to per-cell rock, pvt, metadata, and dynamic state
    arrays. On construction all property groups are normalised to the
    declared `unit_system` (defaults to the grid's own unit system).
    """

    def __init__(
        self,
        grid: Grid,
        rock: Rock,
        pvt: PVT,
        state: State,
        regions: typing.Optional[Regions] = None,
        datum_depth: typing.Optional[Number] = None,
        unit_system: typing.Optional[UnitSystem] = None,
    ) -> None:
        """
        Initialize the reservoir model.

        :param grid: Fully constructed `bores.grids.base.Grid`.
        :param rock: Static petrophysical properties. Array lengths must equal `grid.n_cells`.
        :param pvt: Static PVT characterisation of the reservoir fluids.
        :param state: Initial (or current) dynamic simulation state. Array lengths must
            equal `grid.n_cells`. Should never be modified during simulation. 
            If modification is necessary, create a copy.
        :param datum_depth: Reference depth (positive downward, grid length units) of the datum
            plane used for pressure initialisation by the equilibration routine.
            `None` means no explicit datum is declared.
        :param unit_system: Target unit system for all property groups. When `None`, defaults
            to `grid.unit_system`. If the grid's declared unit system does not
            match `unit_system` a `ValueError` is raised (grid coordinates are
            not converted here).

        :raises ValueError: If `unit_system` does not match `grid.unit_system`.
        :raises ValidationError: If any array length in `rock`, `state`, or `hysteresis` does not
            match `grid.n_cells`, or if `datum_depth` is negative.
        """
        target_unit_system = (
            unit_system if unit_system is not None else grid.unit_system
        )
        if target_unit_system != grid.unit_system:
            raise ValueError(
                f"unit_system={target_unit_system.value!r} does not match "
                f"grid.unit_system={grid.unit_system.value!r}.  "
                "Grid coordinates are not re-projected here; use a grid factory "
                "that produces a grid in the desired unit system, or omit "
                "unit_system to accept the grid's native units."
            )

        if datum_depth is not None and datum_depth < 0.0:
            raise ValidationError(
                f"`datum_depth` must be non-negative (positive downward); "
                f"got {datum_depth}."
            )

        unit_conversion_table = build_unit_conversion_table()
        # Normalise property groups to the target unit system.
        rock = rock.convert(target_unit_system, table=unit_conversion_table)
        pvt = pvt.convert(target_unit_system, table=unit_conversion_table)
        state = state.convert(target_unit_system, table=unit_conversion_table)

        n_cells = grid.n_cells
        _validate_rock(rock, n_cells)
        _validate_state(state, n_cells)

        self.grid = grid
        """The unstructured polyhedral grid with all geometry and topology."""

        self.rock = rock
        """Static petrophysical properties in `unit_system`."""

        self.pvt = pvt
        """Static PVT characterisation in `unit_system`."""

        self.state = state
        """Dynamic per-cell simulation state in `unit_system`."""

        self.regions = regions
        """Per-cell region assignments metadata."""

        self.datum_depth = datum_depth
        """
        Reference depth (grid length units, positive downward) for pressure
        equilibration. `None` when no explicit datum is declared.
        """

        self.unit_system = target_unit_system
        """
        Unit system in which all property groups are expressed.

        Matches `grid.unit_system` by construction.
        """
        self._transmissibilities: typing.Optional[ConnectionTransmissibilities] = None
        self._face_transmissibility_map: typing.Optional[typing.Dict[int, Number]] = (
            None
        )

    @property
    def n_cells(self) -> int:
        """Total number of active cells in the grid."""
        return self.grid.n_cells

    @property
    def n_faces(self) -> int:
        """Total number of faces (boundary + interior) in the grid."""
        return self.grid.n_faces

    @property
    def n_nnc(self) -> int:
        """Total number of non-neighbour connections in the grid."""
        return self.grid.n_nnc

    @property
    def n_interior_faces(self) -> int:
        """Number of interior faces (shared between two active cells)."""
        return self.grid.n_interior_faces

    @property
    def n_boundary_faces(self) -> int:
        """Number of boundary faces (one side is the exterior domain)."""
        return self.grid.n_boundary_faces

    @property
    def depth(self) -> CellArray:
        """
        Shape `(n_cells,)` — depth of each cell centroid (positive downward).

        Alias for `grid.cell_center_depths`. Units follow `unit_system`.
        """
        return self.grid.cell_center_depths  # type: ignore[return-value]

    @property
    def elevation(self) -> CellArray:
        """
        Shape `(n_cells,)` — elevation of each cell centroid (positive upward).

        Alias for `grid.cell_center_elevations`. Units follow `unit_system`.
        """
        return self.grid.cell_center_elevations  # type: ignore[return-value]

    @property
    def pore_volumes(self) -> CellArray:
        """
        Shape `(n_cells,)` — bulk pore volume of each active cell.

        Computed as `φ x NTG x Vcell`.

        Units: ft³ (FIELD), m³ (METRIC / SI), cm³ (LAB).
        """
        assert self.grid.cell_volumes is not None, (
            "`grid.cell_volumes` is None; the grid was constructed without "
            "pre-computed volumes and the divergence-theorem computation failed."
        )
        return (  # type: ignore[return-value]
            self.rock.porosity * self.rock.net_to_gross * self.grid.cell_volumes
        )

    @property
    def hydrocarbon_pore_volumes(self) -> CellArray:
        """
        Shape `(n_cells,)` — hydrocarbon pore volume of each active cell.

        Computed as `pore_volumes x (1 - Swc)` where Swc is the connate
        water saturation from `rock`.

        Units: same as `pore_volumes`.
        """
        return (  # type: ignore[return-value]
            self.pore_volumes * (1.0 - self.rock.connate_water_saturation)
        )

    @property
    def transmissibilities(self) -> ConnectionTransmissibilities:
        """
        TPFA connection transmissibilities, computed on first access and
        then cached for subsequent access.

        Returns a `ConnectionTransmissibilities` named tuple with:

        - `interior`      — shape `(n_interior,)` harmonic-mean T.
        - `boundary`      — shape `(n_boundary,)` owner half-T.
        - `nnc`           — shape `(n_nnc,)` or `None`.
        - `interior_map`  — global face index for each interior entry.
        - `boundary_map`  — global face index for each boundary entry.

        To force recomputation (e.g. after updating `rock`), call
        `invalidate_transmissibilities()` first.
        """
        if self._transmissibilities is None:
            self._transmissibilities = compute_connection_transmissibilities(
                grid=self.grid, rock=self.rock
            )
        return self._transmissibilities

    @property
    def face_transmissibility_map(self) -> typing.Dict[int, Number]:
        if self._face_transmissibility_map is None:
            self._face_transmissibility_map = get_face_transmissibility_map(
                self.grid, self.transmissibilities
            )
        return self._face_transmissibility_map

    def invalidate_transmissibilities(self) -> None:
        """
        Clear the cached transmissibilities.

        Call this whenever `rock.absolute_permeability` or `rock.net_to_gross`
        is updated (e.g. history-matching) so the next access to
        `transmissibilities` triggers a fresh computation.
        """
        self._transmissibilities = None
        self._face_transmissibility_map = None

    def evolve_state(self, new_state: State) -> Self:
        """
        Return a new `ReservoirModel` with the dynamic state replaced.

        The grid, rock, pvt, datum depth, and unit system are
        carried forward unchanged.  The transmissibility cache is **preserved**
        (it depends only on grid geometry and rock, which have not changed).

        :param new_state: Updated `State` from the solver.
        :returns: New `ReservoirModel`.
        :raises ValidationError: If `new_state` array lengths do not match
            `grid.n_cells`.
        """
        _validate_state(new_state, self.n_cells)
        new_model = self.__class__(
            grid=self.grid,
            rock=self.rock,
            pvt=self.pvt,
            regions=self.regions,
            state=new_state,
            datum_depth=self.datum_depth,
            unit_system=self.unit_system,
        )
        new_model._transmissibilities = self._transmissibilities
        return new_model

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: typing.Optional[UnitConversionTable] = None,
    ) -> Self:
        """
        Return a new `ReservoirModel` with all property groups rescaled to `target`.

        The grid's coordinate arrays are **not** re-projected (Eclipse
        convention); only the property scalars and per-cell arrays that carry
        dimensional units are converted.  The returned model's `grid`
        carries an updated `unit_system` declaration but identical vertex
        coordinates.

        For a true coordinate reprojection (e.g. ft → m for all vertex
        positions), use a grid factory or IO utility that rebuilds the grid.

        :param target: Desired `UnitSystem`.
        :returns: New `ReservoirModel` in `target` units.
        """
        if target == self.unit_system:
            return self

        # Rebuild grid with updated `unit_system` declaration only
        new_grid = attrs.evolve(self.grid, unit_system=target)
        table = table or build_unit_conversion_table()
        factors = get_conversion_factors(self.unit_system, target, table=table)
        new_model = self.__class__(
            grid=new_grid,
            rock=self.rock.convert(target, table=table),
            pvt=self.pvt.convert(target, table=table),
            state=self.state.convert(target, table=table),
            regions=self.regions,
            datum_depth=(
                self.datum_depth * factors["length"]
                if self.datum_depth is not None
                else None
            ),
            unit_system=target,
        )
        # Transmissibility cache is invalidated automatically since rock was
        # converted and the new model starts with a clean cache.
        return new_model

    def get_transmissibility_for_face(self, face_index: int) -> Number:
        """
        Return the transmissibility (or half-transmissibility for boundary) of a single face.

        :param face_index: Index into `grid.face_cell_indices`.
        :returns: Transmissibility in grid units (mD·ft in FIELD etc.).
        :raises KeyError: If `face_index` is not a valid face.
        """
        t = self.face_transmissibility_map.get(face_index)
        if t is None:
            raise KeyError(
                f"Face {face_index} not found in interior or boundary faces of this grid."
            )
        return t

    def summary(self) -> typing.Dict[str, typing.Any]:
        """
        Return a lightweight diagnostic summary dictionary.

        Includes cell counts, pore-volume totals, pressure statistics, and
        saturation range checks. Useful for logging and quick sanity checks.

        :returns: Dictionary of scalar statistics.
        """
        pv = self.pore_volumes
        p = self.state.pressure
        sw = self.state.water_saturation
        so = self.state.oil_saturation
        sg = self.state.gas_saturation
        return {
            "n_cells": self.n_cells,
            "n_faces": self.n_faces,
            "n_interior_faces": self.n_interior_faces,
            "n_boundary_faces": self.n_boundary_faces,
            "n_nnc": self.grid.n_nnc,
            "unit_system": self.unit_system.value,
            "total_pore_volume": float(pv.sum()),
            "min_pressure": float(p.min()),
            "max_pressure": float(p.max()),
            "mean_pressure": float(p.mean()),
            "min_water_saturation": float(sw.min()),
            "max_water_saturation": float(sw.max()),
            "min_oil_saturation": float(so.min()),
            "max_oil_saturation": float(so.max()),
            "min_gas_saturation": float(sg.min()),
            "max_gas_saturation": float(sg.max()),
            "saturation_balance_max_error": float(abs(so + sw + sg - 1.0).max()),
            "has_transmissibility_multipliers": (
                self.grid.has_transmissibility_multipliers
            ),
            "datum_depth": self.datum_depth,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_cells={self.n_cells}, "
            f"n_faces={self.n_faces}, "
            f"n_interior={self.n_interior_faces}, "
            f"n_boundary={self.n_boundary_faces}, "
            f"n_nnc={self.n_nnc}, "
            f"unit_system={self.unit_system.value!r}, "
            f"has_hysteresis={self.state.hysteresis is not None}"
            f")"
        )
