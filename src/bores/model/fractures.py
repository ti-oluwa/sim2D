"""
Fracture and Fault Modeling for 3D Reservoir Simulation

API for modeling fractures, faults, and their effects on fluid flow in 3D reservoir simulations.
Faults can act as barriers, conduits, or both, depending on their properties and the reservoir conditions.
"""

import logging
import typing

import attrs
import numba  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.errors import ValidationError
from bores.models import ReservoirModel
from bores.serialization import Serializable
from bores.stores import StoreSerializable
from bores.transmissibility import FaceTransmissibilities
from bores.typing import ThreeDimensions

__all__ = [
    "Fracture",
    "FractureGeometry",
    "apply_fracture",
    "apply_fractures",
    "conductive_fracture_network",
    "damage_zone_fault",
    "inclined_sealing_fault",
    "validate_fracture",
    "vertical_sealing_fault",
]

logger = logging.getLogger(__name__)


@numba.njit(cache=True, parallel=True)
def _mask_orientation_x(
    mask: npt.NDArray,
    cell_min: int,
    cell_max: int,
    slope: float,
    intercept: float,
    z_min: int,
    z_max: int,
    y_min: int,
    y_max: int,
) -> None:
    """
    Build fracture mask for x-oriented fractures (cell-space, shape nx x ny x nz).

    For a planar fault (slope=0) every cell in `x_range x y_range x z_range`
    is marked True. For an inclined fault the z-index of the fracture plane at
    each y-position is computed as `z = int(intercept + slope * j)` and only
    that cell is marked.

    :param mask: Output boolean mask array, shape (nx, ny, nz).
    :param cell_min: Minimum x-index of fracture plane (inclusive).
    :param cell_max: Maximum x-index of fracture plane (inclusive).
    :param slope: Slope for inclined fractures (dz/dy). Zero for planar.
    :param intercept: Z-intercept of fracture plane equation.
    :param z_min: Minimum z-index extent (inclusive).
    :param z_max: Maximum z-index extent (inclusive).
    :param y_min: Minimum y-index lateral extent (inclusive).
    :param y_max: Maximum y-index lateral extent (inclusive).
    """
    nx, _, _ = mask.shape
    for i in numba.prange(cell_min, cell_max + 1):  # type: ignore
        if 0 <= i < nx:
            if abs(slope) < 1e-6:
                for j in range(y_min, y_max + 1):
                    for k in range(z_min, z_max + 1):
                        mask[i, j, k] = True
            else:
                for j in range(y_min, y_max + 1):
                    z_fracture = int(intercept + slope * j)
                    if z_min <= z_fracture <= z_max:
                        mask[i, j, z_fracture] = True


@numba.njit(cache=True, parallel=True)
def _mask_orientation_y(
    mask: npt.NDArray,
    cell_min: int,
    cell_max: int,
    slope: float,
    intercept: float,
    z_min: int,
    z_max: int,
    x_min: int,
    x_max: int,
) -> None:
    """
    Build fracture mask for y-oriented fractures (cell-space, shape nx x ny x nz).

    :param mask: Output boolean mask array, shape (nx, ny, nz).
    :param cell_min: Minimum y-index of fracture plane (inclusive).
    :param cell_max: Maximum y-index of fracture plane (inclusive).
    :param slope: Slope for inclined fractures (dz/dx). Zero for planar.
    :param intercept: Z-intercept of fracture plane equation.
    :param z_min: Minimum z-index extent (inclusive).
    :param z_max: Maximum z-index extent (inclusive).
    :param x_min: Minimum x-index lateral extent (inclusive).
    :param x_max: Maximum x-index lateral extent (inclusive).
    """
    _, ny, _ = mask.shape
    for j in numba.prange(cell_min, cell_max + 1):  # type: ignore
        if 0 <= j < ny:
            if abs(slope) < 1e-6:
                for i in range(x_min, x_max + 1):
                    for k in range(z_min, z_max + 1):
                        mask[i, j, k] = True
            else:
                for i in range(x_min, x_max + 1):
                    z_fracture = int(intercept + slope * i)
                    if z_min <= z_fracture <= z_max:
                        mask[i, j, z_fracture] = True


@numba.njit(cache=True, parallel=True)
def _mask_orientation_z(
    mask: npt.NDArray,
    cell_min: int,
    cell_max: int,
    slope: float,
    intercept: float,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> None:
    """
    Build fracture mask for z-oriented fractures (cell-space, shape nx x ny x nz).

    :param mask: Output boolean mask array, shape (nx, ny, nz).
    :param cell_min: Minimum z-index of fracture plane (inclusive).
    :param cell_max: Maximum z-index of fracture plane (inclusive).
    :param slope: Slope for inclined fractures (dy/dx). Zero for planar.
    :param intercept: Y-intercept of fracture plane equation.
    :param x_min: Minimum x-index lateral extent (inclusive).
    :param x_max: Maximum x-index lateral extent (inclusive).
    :param y_min: Minimum y-index lateral extent (inclusive).
    :param y_max: Maximum y-index lateral extent (inclusive).
    """
    _, _, nz = mask.shape
    for k in numba.prange(cell_min, cell_max + 1):  # type: ignore
        if 0 <= k < nz:
            if abs(slope) < 1e-6:
                for i in range(x_min, x_max + 1):
                    for j in range(y_min, y_max + 1):
                        mask[i, j, k] = True
            else:
                for i in range(x_min, x_max + 1):
                    y_fracture = int(intercept + slope * i)
                    if y_min <= y_fracture <= y_max:
                        mask[i, y_fracture, k] = True


# `FaceTransmissibilities` uses a padded layout where Tx[i+1, j+1, k+1] is the
# forward-x face of cell (i, j, k). Entry [i+1, j+1, k+1] therefore
# represents the face *between* cells (i, j, k) and (i+1, j, k).
#
# For a fault whose cells are marked in the cell-space mask, we scale the
# forward face on the *low* side of the fault plane:
#
#   x-fault at cell i  -> Tx[i+1, j+1, k+1]  (face leaving cell i eastward)
#   y-fault at cell j  -> Ty[i+1, j+1, k+1]  (face leaving cell j southward)
#   z-fault at cell k  -> Tz[i+1, j+1, k+1]  (face leaving cell k downward)
#
# For damage zones (where `cell_range` spans multiple cells) every forward face of
# every marked cell is scaled, which correctly restricts flow into and out of
# the damage zone.
#
# Additionally, for x/y oriented faults the vertical (z-direction) faces are
# also scaled to capture across-layer leakage and is physically justified for
# fault zones that also impede vertical cross-fault flow.
#
# Damage zone boundary face handling:
# For a single-cell fault at x=i there is exactly one face to scale:
# Tx[i+1, j+1, k+1], which is the shared face between cell i and i+1.
# Scaling it once correctly restricts flow in both directions.
#
# For a multi-cell damage zone spanning x=i_min..i_max the forward-face
# loop covers Tx[i_min+1, ..] through Tx[i_max+1, ..], i.e, all internal faces
# and the exit face. But it misses Tx[i_min, ..], the face entering the zone
# from cell i_min-1. We therefore also scale the *entry* face of each masked
# cell whose low-side neighbour is outside the mask.


@numba.njit(cache=True, parallel=True)
def _scale_transmissibility_x_faces(
    tx: npt.NDArray, mask: npt.NDArray, scale: float
) -> None:
    """
    Scale x-direction face transmissibilities for cells in the fracture mask.

    For each cell (i, j, k) where `mask[i, j, k]` is True:

    - The forward face `Tx[i+1, j+1, k+1]` (between cell i and i+1) is always scaled.
    - The entry face `Tx[i, j+1, k+1]` (between cell i-1 and i) is scaled
      only when cell i-1 is **not** in the mask, i.e. it is the western
      boundary of the damage zone. This avoids double-scaling internal faces
      while correctly sealing the zone boundary for multi-cell damage zones.

    :param tx: Padded x-face transmissibility array, shape (nx+2, ny+2, nz+2).
    :param mask: Cell-space boolean mask, shape (nx, ny, nz).
    :param scale: Multiplier to apply (< 1 seals, > 1 enhances).
    """
    nx, ny, nz = mask.shape
    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    # Forward (exit) face: between (i,j,k) and (i+1,j,k)
                    tx[i + 1, j + 1, k + 1] *= scale
                    # Entry face: between (i-1,j,k) and (i,j,k) - only when
                    # the western neighbour is outside the mask (zone boundary)
                    if i == 0 or not mask[i - 1, j, k]:
                        tx[i, j + 1, k + 1] *= scale


@numba.njit(cache=True, parallel=True)
def _scale_transmissibility_y_faces(
    ty: npt.NDArray, mask: npt.NDArray, scale: float
) -> None:
    """
    Scale y-direction face transmissibilities for cells in the fracture mask.

    For each cell (i, j, k) where `mask[i, j, k]` is True:

    - The forward face `Ty[i+1, j+1, k+1]` (between cell j and j+1) is always scaled.
    - The entry face `Ty[i+1, j, k+1]` (between cell j-1 and j) is scaled
      only when cell j-1 is **not** in the mask (northern zone boundary).

    :param ty: Padded y-face transmissibility array, shape (nx+2, ny+2, nz+2).
    :param mask: Cell-space boolean mask, shape (nx, ny, nz).
    :param scale: Multiplier to apply (< 1 seals, > 1 enhances).
    """
    nx, ny, nz = mask.shape
    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    # Forward (exit) face: between (i,j,k) and (i,j+1,k)
                    ty[i + 1, j + 1, k + 1] *= scale
                    # Entry face: between (i,j-1,k) and (i,j,k) - only at
                    # the northern boundary of the damage zone
                    if j == 0 or not mask[i, j - 1, k]:
                        ty[i + 1, j, k + 1] *= scale


@numba.njit(cache=True, parallel=True)
def _scale_transmissibility_z_faces(
    tz: npt.NDArray, mask: npt.NDArray, scale: float
) -> None:
    """
    Scale z-direction face transmissibilities for cells in the fracture mask.

    For each cell (i, j, k) where `mask[i, j, k]` is True:

    - The forward face `Tz[i+1, j+1, k+1]` (between cell k and k+1) is always scaled.
    - The entry face `Tz[i+1, j+1, k]` (between cell k-1 and k) is scaled
      only when cell k-1 is **not** in the mask (top boundary of the zone).

    :param tz: Padded z-face transmissibility array, shape (nx+2, ny+2, nz+2).
    :param mask: Cell-space boolean mask, shape (nx, ny, nz).
    :param scale: Multiplier to apply (< 1 seals, > 1 enhances).
    """
    nx, ny, nz = mask.shape
    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    # Forward (exit) face: between (i,j,k) and (i,j,k+1)
                    tz[i + 1, j + 1, k + 1] *= scale
                    # Entry face: between (i,j,k-1) and (i,j,k) - only at
                    # the top boundary of the damage zone
                    if k == 0 or not mask[i, j, k - 1]:
                        tz[i + 1, j + 1, k] *= scale


@attrs.frozen
class FractureGeometry(Serializable):
    """
    Fracture geometry specification.

    Fracture Orientation:
    - `"x"`: Fracture plane perpendicular to x-axis (strikes in y-direction)
    - `"y"`: Fracture plane perpendicular to y-axis (strikes in x-direction)
    - `"z"`: Fracture plane perpendicular to z-axis (horizontal fracture/bedding slip)

    Coordinate Specification:
    The fracture is defined by specifying ranges in each dimension:

    For x-oriented fracture:
        - `x_range`: Location of fracture plane (can be single cell or damage zone)
        - `y_range`: Optional lateral extent (None = full extent)
        - `z_range`: Optional vertical extent (None = full extent)

    For y-oriented fracture:
        - `y_range`: Location of fracture plane (can be single cell or damage zone)
        - `x_range`: Optional lateral extent (None = full extent)
        - `z_range`: Optional vertical extent (None = full extent)

    For z-oriented fracture (horizontal):
        - `z_range`: Location of fracture plane (can be single layer or zone)
        - `x_range`: Optional lateral extent (None = full extent)
        - `y_range`: Optional lateral extent (None = full extent)

    Examples:

    1. Simple vertical fault at x=25:

    ```python
    FractureGeometry(orientation="x", x_range=(25, 25))
    ```

    2. Wide damage zone from x=20 to x=30, only in upper reservoir:

    ```python
    FractureGeometry(orientation="x", x_range=(20, 30), z_range=(0, 15))
    ```

    3. Horizontal bedding plane slip at z=10:

    ```python
    FractureGeometry(orientation="z", z_range=(10, 10))
    ```

    4. Y-oriented fault with limited lateral extent:

    ```python
    FractureGeometry(
        orientation="y",
        y_range=(15, 15),
        x_range=(10, 40),
        z_range=(5, 25)
    )
    ```

    """

    orientation: typing.Literal["x", "y", "z"]
    """
    Fracture plane orientation:
    - `"x"`: Perpendicular to x-axis (fracture plane in y-z plane)
    - `"y"`: Perpendicular to y-axis (fracture plane in x-z plane)
    - `"z"`: Perpendicular to z-axis (horizontal fracture plane in x-y plane)
    """

    x_range: typing.Optional[typing.Tuple[int, int]] = None
    """
    Range of x-indices affected by fracture (inclusive).

    - For x-oriented fractures: Location of fracture plane
    - For y/z-oriented fractures: Optional lateral extent
    - None means full x-extent of grid
    """

    y_range: typing.Optional[typing.Tuple[int, int]] = None
    """
    Range of y-indices affected by fracture (inclusive).

    - For y-oriented fractures: Location of fracture plane
    - For x/z-oriented fractures: Optional lateral extent
    - None means full y-extent of grid
    """

    z_range: typing.Optional[typing.Tuple[int, int]] = None
    """
    Range of z-indices affected by fracture (inclusive).

    - For z-oriented fractures: Location of fracture plane
    - For x/y-oriented fractures: Optional vertical extent
    - None means full z-extent of grid
    """

    slope: float = 0.0
    """
    Slope of inclined fracture plane (for non-vertical/non-horizontal fractures).

    For x-oriented fractures: z = intercept + slope * y
    For y-oriented fractures: z = intercept + slope * x
    For z-oriented fractures: y = intercept + slope * x

    A slope of 0.0 creates a planar fracture (vertical for x/y, horizontal for z).
    """

    intercept: float = 0.0
    """
    Intercept of the fracture plane equation.
    Interpretation depends on orientation and slope.
    """

    def __attrs_post_init__(self) -> None:
        """Validate geometry configuration."""
        if self.orientation == "x" and self.x_range is None:
            raise ValidationError(
                "For x-oriented fractures, `x_range` must be specified"
            )
        if self.orientation == "y" and self.y_range is None:
            raise ValidationError(
                "For y-oriented fractures, `y_range` must be specified"
            )
        if self.orientation == "z" and self.z_range is None:
            raise ValidationError(
                "For z-oriented fractures, `z_range` must be specified"
            )

        for range_name, range_val in [
            ("x_range", self.x_range),
            ("y_range", self.y_range),
            ("z_range", self.z_range),
        ]:
            if range_val is not None:
                if range_val[0] > range_val[1]:
                    raise ValidationError(
                        f"{range_name} min ({range_val[0]}) > max ({range_val[1]})"
                    )
                if range_val[0] < 0:
                    raise ValidationError(
                        f"{range_name} min ({range_val[0]}) must be >= 0"
                    )

    def get_fracture_plane_range(self) -> typing.Tuple[int, int]:
        """Get the primary fracture plane range based on orientation."""
        if self.orientation == "x":
            assert self.x_range is not None
            return self.x_range
        elif self.orientation == "y":
            assert self.y_range is not None
            return self.y_range
        assert self.z_range is not None
        return self.z_range


@attrs.frozen
class Fracture(StoreSerializable):
    """
    Configuration for applying fractures to reservoir models.

    Defines the geometric and hydraulic properties of a fracture, including
    its orientation, transmissibility/permeability properties, and modeling
    approach.

    **Transmissibility vs Permeability:**

    Use `transmissibility_multiplier` (preferred for faults) to directly
    scale the pre-computed `FaceTransmissibilities` on the forward faces
    that cross the fault plane without touching the underlying rock permeability
    arrays. This is the industry-standard approach for sealing/partially-sealing
    faults and conductive fracture corridors.

    Use `permeability` (preferred for damage zones and fracture swarms) to
    set absolute permeability values inside the fracture zone cells. This
    modifies `RockPermeability` so that any subsequent call to
    `build_face_transmissibilities()` honours the altered rock fabric. This
    is appropriate when you also want to set `porosity` or when the fracture
    zone has a distinctly different rock type.

    Both can be combined. Set `permeability` to define the zone fabric and
    `transmissibility_multiplier` to additionally restrict cross-fault flow
    at the fault plane itself.

    Usage Examples:

    ```python

    # Example 1: Sealing fault using transmissibility multiplier
    geometry = FractureGeometry(orientation="x", x_range=(25, 25), z_range=(0, 15))
    fracture = Fracture(
        id="fault_1",
        geometry=geometry,
        transmissibility_multiplier=1e-4
    )

    # Example 2: Damage zone with absolute permeability values
    damage_geometry = FractureGeometry(orientation="y", y_range=(10, 15))
    damage_fracture = Fracture(
        id="damage_zone",
        geometry=damage_geometry,
        permeability=0.01,
        porosity=0.05
    )

    # Example 3: Conductive fracture: both rock fabric and enhanced flow
    frac_geometry = FractureGeometry(orientation="x", x_range=(30, 32), z_range=(5, 20))
    conductive = Fracture(
        id="nat_frac",
        geometry=frac_geometry,
        permeability=5000.0,
        transmissibility_multiplier=10.0,
        conductive=True
    )
    ```
    """

    id: str
    """Unique identifier for the fracture."""

    geometry: FractureGeometry
    """
    Fracture geometry specification.

    Defines the spatial location, orientation, and extent of the fracture
    using a consistent coordinate system.
    """

    transmissibility_multiplier: typing.Optional[float] = None
    """
    Multiplier applied directly to `FaceTransmissibilities` on the forward
    faces that cross the fault plane.

    - Values < 1.0 create sealing barriers (typical: 1e-3 to 1e-6).
    - Values > 1.0 create conductive zones (enhanced fractures).
    - Values == 1.0 have no effect (no-op).
    - Must be > `c.MINIMUM_TRANSMISSIBILITY_FACTOR` for numerical stability;
      values below this are clamped automatically with a warning.
    - `None` means no transmissibility scaling is applied.

    Unlike `permeability`, this operates on the pre-computed
    `FaceTransmissibilities` stored on the model and does **not** modify
    `RockPermeability`. Use this for faults where you want to control
    cross-fault flow precisely without altering the rock fabric.

    Mutually exclusive with `permeability` as the sole property: you *can*
    combine them, but `transmissibility_multiplier` and `permeability`
    serve distinct purposes and both may be set simultaneously.
    """

    permeability: typing.Optional[float] = None
    """
    Absolute permeability value assigned to fracture zone cells (mD).

    When set, all three permeability components (x, y, z) of the
    `RockPermeability` are overwritten to this value for every cell
    marked by the fracture mask. This is the appropriate choice for
    damage zones and conductive fracture networks where the fracture zone
    has a distinct rock fabric.

    `None` means rock permeability is not modified.
    """

    porosity: typing.Optional[float] = None
    """
    Porosity value assigned to fracture zone cells (fraction).
    `None` means porosity is not modified.
    """

    conductive: bool = False
    """
    If True, the fracture acts as a high-permeability conduit.

    Triggers an additional validation check to ensure
    `transmissibility_multiplier` >= 1.0 when both are provided.
    """

    mask: typing.Optional[np.ndarray] = None
    """
    Optional pre-computed 3D boolean mask defining fracture geometry in
    cell-space (shape `(nx, ny, nz)`).

    When provided, overrides the geometric fracture plane calculation from
    `FractureGeometry`. Must match the reservoir model grid dimensions.
    """

    def __attrs_post_init__(self) -> None:
        """Validate fracture configuration parameters."""
        if self.transmissibility_multiplier is not None:
            if self.transmissibility_multiplier < c.MINIMUM_TRANSMISSIBILITY_FACTOR:
                object.__setattr__(
                    self,
                    "transmissibility_multiplier",
                    c.MINIMUM_TRANSMISSIBILITY_FACTOR,
                )
                logger.warning(
                    f"Fracture {self.id!r}: `transmissibility_multiplier` clamped to "
                    f"{c.MINIMUM_TRANSMISSIBILITY_FACTOR}"
                )

            if self.conductive and self.transmissibility_multiplier < 1.0:
                raise ValidationError(
                    f"Fracture {self.id!r}: conductive fractures must have "
                    f"`transmissibility_multiplier` >= 1.0"
                )

        if self.permeability is not None and self.permeability < 0:
            raise ValidationError(
                f"Fracture {self.id!r}: `permeability` must be non-negative"
            )

        if self.porosity is not None and not (0 <= self.porosity <= 1):
            raise ValidationError(
                f"Fracture {self.id!r}: `porosity` must be between 0 and 1"
            )


def apply_fracture(
    model: ReservoirModel[ThreeDimensions], fracture: Fracture
) -> ReservoirModel[ThreeDimensions]:
    """
    Apply a single fracture to a reservoir model.

    The function modifies, in order:

    1. **Rock permeability and porosity** (if `fracture.permeability` or
       `fracture.porosity` are set): modifies `RockProperties` arrays
       in-place so that a subsequent `build_face_transmissibilities()` call
       will reflect the altered rock fabric.

    2. **Face transmissibilities** (if `fracture.transmissibility_multiplier`
       is set): scales the forward faces that cross the fault plane directly
       in the padded `FaceTransmissibilities` arrays. If face
       transmissibilities have not yet been built on the model they are
       computed first so the multiplier can be applied.

    The input model is not replaced (all mutations are in-place on numpy
    arrays); the same model object is returned for chaining convenience.

    :param model: Input reservoir model to modify.
    :param fracture: Fracture configuration defining geometry and properties.
    :return: The reservoir model with the fracture applied (same object).
    :raises ValidationError: If the fracture configuration is invalid for the
        given grid or the model is not 3-D.
    """
    logger.debug(f"Applying fracture '{fracture.id}' to reservoir model")

    errors = validate_fracture(fracture=fracture, grid_shape=model.grid_shape)
    if errors:
        msg = "\n".join(f" - {e}" for e in errors)
        raise ValidationError(
            f"Fracture {fracture.id!r} configuration is invalid:\n{msg}"
        )

    if len(model.grid_shape) != 3:
        raise ValidationError("Fracture application requires 3D reservoir models")

    grid_shape = model.grid_shape

    # Build or validate the cell-space mask
    if fracture.mask is not None:
        if fracture.mask.shape != grid_shape:
            raise ValidationError(
                f"Fracture {fracture.id!r}: mask shape {fracture.mask.shape} "
                f"!= grid shape {grid_shape}"
            )
        fracture_mask = fracture.mask.copy()
    else:
        fracture_mask = make_fracture_mask(
            grid_shape=grid_shape,  # type: ignore[arg-type]
            fracture=fracture,
        )

    # Do rock fabric modifications (permeability / porosity) first
    if fracture.permeability is not None or fracture.porosity is not None:
        _apply_fracture_zone_properties(
            model=model,
            fracture_mask=fracture_mask,
            fracture=fracture,
        )
        # Invalidate any cached face transmissibilities so the new rock
        # fabric is picked up on the next build call.
        if model.face_transmissibilities is not None:
            logger.debug(
                f"Fracture '{fracture.id}': invalidating cached face transmissibilities "
                "after permeability/porosity modification."
            )
            model.face_transmissibilities = None  # type: ignore[attr-defined]

    # Then transmissibility scaling next
    if fracture.transmissibility_multiplier is not None:
        # Ensure face transmissibilities exist. Build them if necessary.
        face_transmissibilities = model.build_face_transmissibilities()
        _scale_face_transmissibilities(
            face_transmissibilities=face_transmissibilities,
            fracture_mask=fracture_mask,
            fracture=fracture,
        )

    logger.debug(f"Successfully applied fracture '{fracture.id}'")
    return model


def apply_fractures(
    model: ReservoirModel[ThreeDimensions], *fractures: Fracture
) -> ReservoirModel[ThreeDimensions]:
    """
    Apply multiple fractures to a reservoir model sequentially.

    Fractures are applied in the order provided. Each `apply_fracture` call
    may invalidate the cached `FaceTransmissibilities` if permeability or
    porosity is modified; transmissibility multipliers are then applied to
    the rebuilt arrays.

    :param model: Input reservoir model.
    :param fractures: Sequence of fracture configurations.
    :return: The reservoir model with all fractures applied (same object).
    """
    logger.debug(f"Applying {len(fractures)} fractures to reservoir model")

    for fracture in fractures:
        apply_fracture(model=model, fracture=fracture)

    logger.debug(f"Successfully applied all {len(fractures)} fractures")
    return model


def make_fracture_mask(
    grid_shape: typing.Tuple[int, int, int], fracture: Fracture
) -> np.ndarray:
    """
    Generate a 3D boolean mask in cell-space `(nx, ny, nz)` defining the
    fracture geometry.

    For inclined fractures the mask follows the equation:

    ```
    z = intercept + slope * coord
    ```

    where *coord* is the y-position for x-oriented faults or the x-position
    for y-oriented faults.

    :param grid_shape: Shape of the reservoir grid `(nx, ny, nz)`.
    :param fracture: Fracture configuration.
    :return: 3D boolean array of shape `(nx, ny, nz)` marking fracture cells.
    """
    nx, ny, nz = grid_shape
    mask = np.zeros((nx, ny, nz), dtype=np.bool_)
    geometry = fracture.geometry

    cell_min, cell_max = geometry.get_fracture_plane_range()

    # Determine vertical (z) extent
    if geometry.z_range is not None:
        z_min = max(0, min(geometry.z_range[0], nz - 1))
        z_max = max(0, min(geometry.z_range[1], nz - 1))
    else:
        z_min, z_max = 0, nz - 1

    if geometry.orientation == "x":
        y_min, y_max = (
            (
                max(0, min(geometry.y_range[0], ny - 1)),
                max(0, min(geometry.y_range[1], ny - 1)),
            )
            if geometry.y_range is not None
            else (0, ny - 1)
        )
        _mask_orientation_x(
            mask=mask,
            cell_min=cell_min,
            cell_max=cell_max,
            slope=float(geometry.slope),
            intercept=float(geometry.intercept),
            z_min=z_min,
            z_max=z_max,
            y_min=y_min,
            y_max=y_max,
        )

    elif geometry.orientation == "y":
        x_min, x_max = (
            (
                max(0, min(geometry.x_range[0], nx - 1)),
                max(0, min(geometry.x_range[1], nx - 1)),
            )
            if geometry.x_range is not None
            else (0, nx - 1)
        )
        _mask_orientation_y(
            mask=mask,
            cell_min=cell_min,
            cell_max=cell_max,
            slope=float(geometry.slope),
            intercept=float(geometry.intercept),
            z_min=z_min,
            z_max=z_max,
            x_min=x_min,
            x_max=x_max,
        )

    elif geometry.orientation == "z":
        x_min, x_max = (
            (
                max(0, min(geometry.x_range[0], nx - 1)),
                max(0, min(geometry.x_range[1], nx - 1)),
            )
            if geometry.x_range is not None
            else (0, nx - 1)
        )
        y_min, y_max = (
            (
                max(0, min(geometry.y_range[0], ny - 1)),
                max(0, min(geometry.y_range[1], ny - 1)),
            )
            if geometry.y_range is not None
            else (0, ny - 1)
        )
        _mask_orientation_z(
            mask=mask,
            cell_min=cell_min,
            cell_max=cell_max,
            slope=float(geometry.slope),
            intercept=float(geometry.intercept),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    return mask


def _scale_face_transmissibilities(
    face_transmissibilities: FaceTransmissibilities,
    fracture_mask: npt.NDArray,
    fracture: Fracture,
) -> None:
    """
    Scale face transmissibilities across fault boundaries.

    The scaling targets the forward faces of each cell marked in
    `fracture_mask` that are oriented along the fault's principal direction.
    For x/y oriented faults the z-direction faces of the marked cells are
    *also* scaled to capture cross-layer leakage across the fault zone.

    The padded `FaceTransmissibilities` arrays are modified **in-place**.

    :param face_transmissibilities: Pre-computed padded face transmissibilities
        (shape `(nx+2, ny+2, nz+2)` each).
    :param fracture_mask: Cell-space boolean mask, shape `(nx, ny, nz)`.
    :param fracture: Fracture configuration carrying the multiplier and
        geometry orientation.
    """
    if fracture.transmissibility_multiplier is None:
        return

    scale = float(fracture.transmissibility_multiplier)
    orientation = fracture.geometry.orientation

    logger.debug(
        f"Scaling transmissibilities for fracture '{fracture.id}' "
        f"(orientation={orientation!r}, scale={scale:.3e})"
    )

    if orientation == "x":
        _scale_transmissibility_x_faces(
            tx=face_transmissibilities.x,
            mask=fracture_mask,
            scale=scale,
        )
        # Also restrict z-faces to capture cross-layer leakage across the fault
        _scale_transmissibility_z_faces(
            tz=face_transmissibilities.z,
            mask=fracture_mask,
            scale=scale,
        )

    elif orientation == "y":
        _scale_transmissibility_y_faces(
            ty=face_transmissibilities.y,
            mask=fracture_mask,
            scale=scale,
        )
        # Also restrict z-faces for the same reason
        _scale_transmissibility_z_faces(
            tz=face_transmissibilities.z,
            mask=fracture_mask,
            scale=scale,
        )

    elif orientation == "z":
        _scale_transmissibility_z_faces(
            tz=face_transmissibilities.z,
            mask=fracture_mask,
            scale=scale,
        )


def _apply_fracture_zone_properties(
    model: ReservoirModel[ThreeDimensions],
    fracture_mask: npt.NDArray,
    fracture: Fracture,
) -> None:
    """
    Apply fracture zone rock properties (absolute permeability, porosity) to
    all cells within the fracture mask.

    Modifies `RockProperties` arrays **in-place**.

    :param model: Reservoir model whose rock properties are to be modified.
    :param fracture_mask: Cell-space boolean mask, shape `(nx, ny, nz)`.
    :param fracture: Fracture configuration carrying `permeability` and
        `porosity` values.
    """
    logger.debug(f"Applying fracture zone properties for fracture '{fracture.id}'")

    if fracture.permeability is not None:
        perm = float(fracture.permeability)
        model.rock_properties.absolute_permeability.x[fracture_mask] = perm
        model.rock_properties.absolute_permeability.y[fracture_mask] = perm
        model.rock_properties.absolute_permeability.z[fracture_mask] = perm

        # Rebuild the mean permeability for the affected cells. Because all
        # three components are set to the same value the geometric mean equals
        # that value: we just overwrite directly for efficiency.
        model.rock_properties.absolute_permeability.mean[fracture_mask] = perm

    if fracture.porosity is not None:
        model.rock_properties.porosity_grid[fracture_mask] = float(fracture.porosity)


def vertical_sealing_fault(
    fault_id: str,
    orientation: typing.Literal["x", "y", "z"],
    index: int,
    transmissibility_multiplier: float = 1e-4,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
) -> Fracture:
    """
    Create a sealing barrier at a grid index with optional extent control.

    Creates a planar barrier that reduces fluid flow by scaling the
    `FaceTransmissibilities` on the forward faces that cross the fault
    plane. Despite the name `"vertical_sealing_fault"` this function
    supports all orientations including horizontal barriers (z-orientation).

    :param fault_id: Unique identifier for the fault.
    :param orientation: Fault orientation (`"x"`, `"y"` for vertical,
        `"z"` for horizontal).
    :param index: Grid index where the fault/barrier is located.
    :param transmissibility_multiplier: Flow reduction factor applied to
        `FaceTransmissibilities` (default: 1e-4 = 99.99% sealing).
    :param x_range: Optional lateral extent in x.
    :param y_range: Optional lateral extent in y.
    :param z_range: Optional vertical extent (for x/y orientation) or layer
        index (for z orientation).
    :return: Configured `Fracture` object.

    Examples:

    ```python

    # Simple vertical fault through entire grid
    vertical_sealing_fault(fault_id="f1", orientation="x", index=25)

    # Shallow fault only in top 10 layers
    vertical_sealing_fault(
        fault_id="f1",
        orientation="x",
        index=25,
        z_range=(0, 10)
    )

    # Horizontal sealing layer (e.g., shale barrier)
    vertical_sealing_fault(
        fault_id="shale_barrier",
        orientation="z",
        index=15,
        x_range=(0, 50),
        y_range=(0, 50)
    )
    ```
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=(index, index),
            y_range=y_range,
            z_range=z_range,
        )
    elif orientation == "y":
        geometry = FractureGeometry(
            orientation="y",
            y_range=(index, index),
            x_range=x_range,
            z_range=z_range,
        )
    else:
        geometry = FractureGeometry(
            orientation="z",
            z_range=(index, index),
            x_range=x_range,
            y_range=y_range,
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        transmissibility_multiplier=transmissibility_multiplier,
    )


def inclined_sealing_fault(
    fault_id: str,
    orientation: typing.Literal["x", "y"],
    index: int,
    slope: float,
    intercept: float,
    transmissibility_multiplier: float = 1e-4,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
) -> Fracture:
    """
    Create an inclined (non-vertical) sealing fault.

    The fault plane follows:

    ```
    z = intercept + slope * coordinate
    ```

    where *coordinate* is the x-position for y-oriented faults or the
    y-position for x-oriented faults.

    :param fault_id: Unique identifier for the fault.
    :param orientation: Fault orientation (`"x"` or `"y"`).
    :param index: Grid index where the fault intersects.
    :param slope: Slope of the fault plane (dz/dx or dz/dy).
    :param intercept: Z-intercept of the fault plane.
    :param transmissibility_multiplier: Flow reduction factor applied to
        `FaceTransmissibilities` (default: 1e-4).
    :param x_range: Optional lateral extent in x (for y-oriented faults).
    :param y_range: Optional lateral extent in y (for x-oriented faults).
    :param z_range: Optional vertical extent to clip the inclined plane.
    :return: Configured `Fracture` object.
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=(index, index),
            y_range=y_range,
            z_range=z_range,
            slope=slope,
            intercept=intercept,
        )
    else:
        geometry = FractureGeometry(
            orientation="y",
            y_range=(index, index),
            x_range=x_range,
            z_range=z_range,
            slope=slope,
            intercept=intercept,
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        transmissibility_multiplier=transmissibility_multiplier,
    )


def damage_zone_fault(
    fault_id: str,
    orientation: typing.Literal["x", "y", "z"],
    cell_range: typing.Tuple[int, int],
    transmissibility_multiplier: float = 1e-3,
    zone_permeability: typing.Optional[float] = None,
    zone_porosity: typing.Optional[float] = None,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
) -> Fracture:
    """
    Create a fault/barrier with a damage zone (multiple cells wide).

    Fault damage zones are regions of reduced permeability and altered rock
    properties surrounding the main fault plane. This is more realistic than
    single-cell faults for large displacement faults.

    The function sets **both** the absolute zone permeability (via
    `RockPermeability`) and the transmissibility multiplier (via
    `FaceTransmissibilities`) so that:

    - The rock fabric inside the damage zone is correctly altered for PVT and
      mobility calculations.
    - The cross-zone flow is additionally restricted at each internal face.

    :param fault_id: Unique identifier for the fault.
    :param orientation: Fault orientation (`"x"`, `"y"` for vertical,
        `"z"` for horizontal).
    :param cell_range: Range of cells defining the damage zone (inclusive).
    :param transmissibility_multiplier: Flow reduction across the zone faces
        (default: 1e-3).
    :param zone_permeability: Permeability within damage zone (mD).  If
        `None` only the transmissibility multiplier is applied.
    :param zone_porosity: Porosity within damage zone (fraction).  If
        `None` porosity is not modified.
    :param x_range: Optional lateral extent in x.
    :param y_range: Optional lateral extent in y.
    :param z_range: Optional vertical extent (for x/y) or limits for z-orientation.
    :return: Configured `Fracture` object.

    Examples:

    ```python
    # Wide vertical damage zone from x=20 to x=30
    damage_zone_fault(fault_id="f1", orientation="x", cell_range=(20, 30))

    # Damage zone only in middle layers
    damage_zone_fault(
        fault_id="f1",
        orientation="x",
        cell_range=(20, 25),
        z_range=(10, 20)
    )

    # Horizontal low-permeability layer (shale, z=10 to z=12)
    damage_zone_fault(
        fault_id="shale",
        orientation="z",
        cell_range=(10, 12),
        zone_permeability=0.01
    )
    ```
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x", x_range=cell_range, y_range=y_range, z_range=z_range
        )
    elif orientation == "y":
        geometry = FractureGeometry(
            orientation="y", y_range=cell_range, x_range=x_range, z_range=z_range
        )
    else:
        geometry = FractureGeometry(
            orientation="z", z_range=cell_range, x_range=x_range, y_range=y_range
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        transmissibility_multiplier=transmissibility_multiplier,
        permeability=zone_permeability,
        porosity=zone_porosity,
    )


def conductive_fracture_network(
    fracture_id: str,
    orientation: typing.Literal["x", "y", "z"],
    cell_range: typing.Tuple[int, int],
    fracture_permeability: float,
    fracture_porosity: float = 0.01,
    transmissibility_multiplier: float = 10.0,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
) -> Fracture:
    """
    Create a highly conductive fracture network (opposite of sealing fault).

    Used for modeling natural fracture systems, hydraulically-fractured zones,
    highly permeable conduits, or high-permeability horizontal layers.

    The function **always** sets both `permeability` (to define the high-perm
    rock fabric) and `transmissibility_multiplier` (to enhance cross-fracture
    flow at the cell faces). This combined approach is more physically accurate
    than using either property alone for conductive features.

    :param fracture_id: Unique identifier for the fracture.
    :param orientation: Fracture orientation (`"x"`, `"y"` for vertical,
        `"z"` for horizontal).
    :param cell_range: Range of cells defining the fracture network.
    :param fracture_permeability: High permeability within fracture (mD).
    :param fracture_porosity: Fracture porosity (typically low, default: 0.01).
    :param transmissibility_multiplier: Flow enhancement factor applied to
        `FaceTransmissibilities` (default: 10.0).
    :param x_range: Optional lateral extent in x.
    :param y_range: Optional lateral extent in y.
    :param z_range: Optional vertical extent or limits for z-orientation.
    :return: Configured `Fracture` object.

    Examples:

    ```python

    # Conductive vertical fracture in pay zone only
    conductive_fracture_network(
        fracture_id="frac1",
        orientation="x",
        cell_range=(25, 27),
        fracture_permeability=1000,
        z_range=(10, 20)
    )

    # Horizontal high-permeability layer (karst zone)
    conductive_fracture_network(
        fracture_id="karst",
        orientation="z",
        cell_range=(15, 17),
        fracture_permeability=5000
    )
    ```
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=cell_range,
            y_range=y_range,
            z_range=z_range,
        )
    elif orientation == "y":
        geometry = FractureGeometry(
            orientation="y",
            y_range=cell_range,
            x_range=x_range,
            z_range=z_range,
        )
    else:
        geometry = FractureGeometry(
            orientation="z",
            z_range=cell_range,
            x_range=x_range,
            y_range=y_range,
        )

    return Fracture(
        id=fracture_id,
        geometry=geometry,
        permeability=fracture_permeability,
        porosity=fracture_porosity,
        transmissibility_multiplier=transmissibility_multiplier,
        conductive=True,
    )


def validate_fracture(
    fracture: Fracture, grid_shape: typing.Tuple[int, ...]
) -> typing.List[str]:
    """
    Validate fracture configuration against grid dimensions.

    :param fracture: Fracture configuration to validate.
    :param grid_shape: Shape of the reservoir grid `(nx, ny, nz)`.
    :return: List of validation error messages (empty list if valid).
    """
    errors: typing.List[str] = []

    if len(grid_shape) != 3:
        errors.append("Grid must be 3D")
        return errors

    nx, ny, nz = grid_shape
    geometry = fracture.geometry

    def _check_range(
        name: str,
        range_: typing.Optional[typing.Tuple[int, int]],
        bound: int,
    ) -> None:
        if range_ is None:
            return
        lo, hi = range_
        if lo > hi:
            errors.append(f"Fracture {fracture.id!r}: {name} min > max ({lo} > {hi})")
        if not (0 <= lo < bound and 0 <= hi < bound):
            errors.append(
                f"Fracture {fracture.id!r}: {name} ({lo}, {hi}) out of bounds "
                f"[0, {bound - 1}]"
            )

    if geometry.orientation == "x":
        _check_range("x_range", geometry.x_range, nx)
    elif geometry.orientation == "y":
        _check_range("y_range", geometry.y_range, ny)
    else:
        _check_range("z_range", geometry.z_range, nz)

    # Secondary extents
    if geometry.orientation in ("x", "y"):
        _check_range("z_range", geometry.z_range, nz)
    if geometry.orientation == "y":
        _check_range("x_range", geometry.x_range, nx)
    if geometry.orientation == "x":
        _check_range("y_range", geometry.y_range, ny)
    if geometry.orientation == "z":
        _check_range("x_range", geometry.x_range, nx)
        _check_range("y_range", geometry.y_range, ny)

    # Intercept must be a valid z-index for inclined faults
    if geometry.intercept != 0.0 and not (0 <= geometry.intercept < nz):
        errors.append(
            f"Fracture {fracture.id!r}: intercept {geometry.intercept} "
            f"out of z-range [0, {nz - 1}]"
        )

    # Custom mask shape
    if fracture.mask is not None and fracture.mask.shape != grid_shape:
        errors.append(
            f"Fracture {fracture.id!r}: mask shape {fracture.mask.shape} "
            f"!= grid shape {grid_shape}"
        )
    return errors
