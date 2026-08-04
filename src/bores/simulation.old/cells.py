import typing
from collections.abc import Mapping

import attrs
import numpy as np
from typing_extensions import Self

from bores.errors import ValidationError
from bores.wells.base import Well, Wells, _expand_intervals

__all__ = ["CellFilter", "CellLocation", "Cells"]


def _make_hashable(obj: typing.Any) -> typing.Any:
    """
    Convert potentially unhashable objects to hashable equivalents.

    This is used for making cell filters hashable for caching.
    Handles lists, numpy arrays, Well/Wells objects, and other sequences by converting to tuples.

    :param obj: Object to make hashable
    :return: Hashable version of the object
    """
    if obj is None:
        return None

    # Already hashable types
    if isinstance(obj, (int, float, str, bool, type(None), slice)):
        return obj

    if isinstance(obj, Well):
        return (
            "__well__",
            obj.name,
            _make_hashable(obj.perforating_intervals),
            str(obj.orientation),
        )

    if isinstance(obj, Wells):
        inj_names = tuple(sorted(w.name for w in obj.injection_wells))
        prod_names = tuple(sorted(w.name for w in obj.production_wells))
        return ("__wells__", inj_names, prod_names)

    if isinstance(obj, np.ndarray):
        return tuple(_make_hashable(item) for item in obj.tolist())

    if isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)

    if isinstance(obj, set):
        return frozenset(_make_hashable(item) for item in obj)

    # Handle tuples recursively (in case they contain unhashable items)
    if isinstance(obj, tuple):
        return tuple(_make_hashable(item) for item in obj)

    if isinstance(obj, Mapping):
        return tuple(
            sorted((_make_hashable(k), _make_hashable(v)) for k, v in obj.items())
        )

    # If we can't make it hashable, convert to string representation
    try:
        hash(obj)
    except TypeError:
        return str(obj)
    return obj


def _get_cell_mask(
    cells: typing.Any,  # CellFilter type
    grid_shape: typing.Tuple[int, ...],
    wells: typing.Optional[typing.Any] = None,  # Wells type
) -> typing.Optional[np.ndarray]:
    """
    Create a boolean mask for filtering cells based on the cells parameter.

    :param cells: Cell filter specification (None, well name, Well object, Wells object,
                  sequence of Well objects, cell, tuple of cells, or slice tuple)
    :param grid_shape: Shape of the reservoir grid
    :param wells: Wells object (required if cells is a well name)
    :return: Boolean mask array or None if no filtering
    """
    if cells is None:
        return None

    # Initialize mask to False
    mask = np.zeros(grid_shape, dtype=bool)

    # Handle Wells object - get all cells from all wells
    if isinstance(cells, Wells):
        # This is a Wells object
        for well in cells.injection_wells:
            cell_locations = _expand_intervals(
                intervals=well.perforating_intervals,
                orientation=well.orientation,
            )
            for loc in cell_locations:
                if len(loc) == 2:
                    mask[loc[0], loc[1]] = True
                else:
                    mask[loc[0], loc[1], loc[2]] = True

        for well in cells.production_wells:
            cell_locations = _expand_intervals(
                intervals=well.perforating_intervals,
                orientation=well.orientation,
            )
            for loc in cell_locations:
                if len(loc) == 2:
                    mask[loc[0], loc[1]] = True
                else:
                    mask[loc[0], loc[1], loc[2]] = True
        return mask

    # Handle single Well object
    if isinstance(cells, Well):
        # This is a Well object
        cell_locations = _expand_intervals(
            intervals=cells.perforating_intervals,
            orientation=cells.orientation,
        )
        for loc in cell_locations:
            if len(loc) == 2:
                mask[loc[0], loc[1]] = True
            else:
                mask[loc[0], loc[1], loc[2]] = True
        return mask

    # Handle sequence of Well objects
    if isinstance(cells, (tuple, list)) and len(cells) > 0:
        first_item = cells[0]
        # Check if it's a sequence of Well objects
        if isinstance(first_item, Well):
            for well in cells:
                cell_locations = _expand_intervals(
                    intervals=well.perforating_intervals,
                    orientation=well.orientation,
                )
                for loc in cell_locations:
                    if len(loc) == 2:
                        mask[loc[0], loc[1]] = True
                    else:
                        mask[loc[0], loc[1], loc[2]] = True
            return mask

    if isinstance(cells, str):
        # Filter by well name
        if wells is None:
            raise ValidationError("Wells object required when filtering by well name")

        # Find the well
        well = None
        for injection_well in wells.injection_wells:
            if injection_well.name == cells:
                well = injection_well
                break

        if well is None:
            for production_well in wells.production_wells:
                if production_well.name == cells:
                    well = production_well
                    break

        if well is None:
            raise ValidationError(f"Well '{cells}' not found")

        # Get all perforated cells for this well
        cell_locations = _expand_intervals(
            intervals=well.perforating_intervals,
            orientation=well.orientation,
        )
        for loc in cell_locations:
            if len(loc) == 2:
                mask[loc[0], loc[1]] = True
            else:
                mask[loc[0], loc[1], loc[2]] = True

    elif isinstance(cells, (tuple, list)) and len(cells) == 3:
        # Check if it's a slice tuple or a single cell
        if all(isinstance(c, slice) for c in cells):
            # Slice region
            mask[cells[0], cells[1], cells[2]] = True
        elif all(isinstance(c, int) for c in cells):
            # Single cell
            if len(grid_shape) == 2:
                mask[cells[0], cells[1]] = True
            else:
                mask[cells[0], cells[1], cells[2]] = True
        else:
            raise ValidationError("Tuple must be either all slices or all integers")

    elif isinstance(cells, (tuple, list)):
        # Tuple/list of cells (multiple cell locations)
        for cell in cells:
            if len(grid_shape) == 2:
                mask[cell[0], cell[1]] = True
            else:
                mask[cell[0], cell[1], cell[2]] = True

    else:
        raise ValidationError(
            f"Invalid cells parameter type: {type(cells)}. "
            "Expected None, str (well name), `Well` object, `Wells` object, sequence of `Well` objects, "
            "tuple (cell/slices/multiple cells)."
        )
    return mask


CellLocation = typing.Tuple[int, int, int]
"""A single cell location as (i, j, k) coordinates."""

CellFilter = typing.Union[
    None,  # No filter, entire reservoir
    str,  # Well name
    typing.Any,  # Well/Wells object (has name, perforating_intervals, orientation)
    typing.Sequence[typing.Any],  # Sequence of Well objects
    CellLocation,  # Single cell (i, j, k)
    typing.Union[
        typing.Tuple[CellLocation, ...], typing.List[CellLocation]
    ],  # Tuple/list of cells
    typing.Tuple[slice, slice, slice],  # Slice region
]
"""
Filter specification for selecting specific cells, wells, or regions.

Can be:
- None: No filter, entire reservoir
- str: Well name (requires Wells object for lookup)
- Well: Single Well object (uses perforating_intervals)
- Wells: Wells object (uses all injection and production wells)
- Sequence[Well]: List or tuple of Well objects
- CellLocation: Single cell (i, j, k)
- Tuple/List of CellLocations: Multiple specific cells
- Tuple[slice, slice, slice]: Slice region
"""


@attrs.frozen(slots=True)
class Cells:
    """
    Container for cell filter specifications.

    This class wraps `CellFilter` types and provides:
    - Robust hashing that handles unhashable types (lists, numpy arrays, Well/Wells objects, etc.)
    - Lazy evaluation of cell masks
    - Type safety and validation

    Examples:
    ```python
    from bores.wells import Well, Wells, InjectionWell, ProductionWell

    # No filter (entire reservoir)
    cells = Cells(None)

    # Filter by well name
    cells = Cells("PROD-1")

    # Filter by Well object
    well = InjectionWell(name="INJ-1", perforating_intervals=[((0, 0, 0), (0, 0, 10))], ...)
    cells = Cells(well)

    # Filter by Wells object (all wells)
    wells = Wells(injection_wells=[inj1, inj2], production_wells=[prod1, prod2])
    cells = Cells(wells)

    # Filter by sequence of Well objects
    cells = Cells([well1, well2, well3])
    cells = Cells((well1, well2))  # tuple also works

    # Single cell
    cells = Cells((10, 10, 5))

    # Multiple cells - tuple (preferred)
    cells = Cells(((10, 10, 5), (11, 10, 5), (12, 10, 5)))

    # Multiple cells - list (automatically converted to hashable form)
    cells = Cells([(10, 10, 5), (11, 10, 5)])

    # Region slice
    cells = Cells((slice(0, 10), slice(0, 10), slice(0, 5)))
    ```
    """

    filter: CellFilter = attrs.field()
    _hash: typing.Any = attrs.field(init=False, repr=False, hash=True, eq=True)

    def __attrs_post_init__(self) -> None:
        """Convert filter to hashable form after initialization."""
        object.__setattr__(self, "_hash", hash(_make_hashable(self.filter)))

    def __hash__(self) -> int:
        """
        Compute hash using the hashable version of the filter.

        This ensures that even if users pass lists or other unhashable types,
        the Cells object can still be hashed for use with functools.cache.
        """
        return self._hash

    @classmethod
    def from_filter(cls, cells: typing.Optional[CellFilter] = None) -> Self:
        """
        Create a Cells instance from a CellFilter specification.

        :param cells: Cell filter specification (None, well name, cell(s), or slice)
        :return: Cells instance
        """
        if isinstance(cells, cls):
            return cells
        return cls(filter=cells)

    def get_mask(
        self,
        grid_shape: typing.Tuple[int, ...],
        wells: typing.Optional[typing.Any] = None,
    ) -> typing.Optional[np.ndarray]:
        """
        Get the boolean mask for this cell filter.

        This method is called by the analysis methods to create the actual mask.
        It's separated from __init__ to allow lazy evaluation.

        :param grid_shape: Shape of the reservoir grid
        :param wells: Wells object (required if filter is a well name)
        :return: Boolean mask array or None if no filtering
        """
        return _get_cell_mask(self.filter, grid_shape, wells)

    @property
    def is_none(self) -> bool:
        """Check if this represents no filtering (entire reservoir)."""
        return self.filter is None

    @property
    def is_well_name(self) -> bool:
        """Check if this represents a well name filter."""
        return isinstance(self.filter, str)

    @property
    def is_single_cell(self) -> bool:
        """Check if this represents a single cell filter."""
        return (
            isinstance(self.filter, tuple)
            and len(self.filter) == 3
            and all(isinstance(x, int) for x in self.filter)
        )

    @property
    def is_multiple_cells(self) -> bool:
        """Check if this represents multiple cells filter."""
        return (
            isinstance(self.filter, (tuple, list))
            and len(self.filter) > 0
            and isinstance(self.filter[0], (tuple, list))
        )

    @property
    def is_slice(self) -> bool:
        """Check if this represents a slice region filter."""
        return (
            isinstance(self.filter, tuple)
            and len(self.filter) == 3
            and all(isinstance(x, slice) for x in self.filter)
        )

    def __repr__(self) -> str:
        """String representation."""
        cls_name = self.__class__.__name__
        if self.is_none:
            return f"{cls_name}(entire reservoir)"
        elif self.is_well_name:
            return f"{cls_name}(well='{self.filter}')"
        elif self.is_single_cell:
            return f"{cls_name}(cell={self.filter})"
        elif self.is_multiple_cells:
            assert isinstance(self.filter, (tuple, list))  # Type narrowing
            return f"{cls_name}({len(self.filter)} cells)"
        elif self.is_slice:
            return f"{cls_name}(slice region)"
        return f"{cls_name}({self.filter})"
