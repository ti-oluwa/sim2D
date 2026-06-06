import typing
from dataclasses import dataclass
from enum import Enum

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from bores.visualization.config import (
    COLORBAR_BOUNDARY_PADDING as _COLORBAR_BOUNDARY_PADDING,
)
from bores.visualization.config import (
    COLORBAR_HEIGHT_FACTOR as _COLORBAR_HEIGHT_FACTOR,
)
from bores.visualization.config import (
    COLORBAR_MIN_LENGTH as _COLORBAR_MIN_LENGTH,
)
from bores.visualization.config import (
    COLORBAR_VERTICAL_PADDING as _COLORBAR_VERTICAL_PADDING,
)
from bores.visualization.config import (
    COLORBAR_XPAD as _COLORBAR_XPAD,
)
from bores.visualization.config import (
    DEFAULT_COLORBAR_MIN_THICKNESS as _COLORBAR_MIN_THICKNESS,
)
from bores.visualization.config import (
    DEFAULT_COLORBAR_THICKNESS as _COLORBAR_BASE_THICKNESS,
)

__all__ = [
    "ColorScheme",
    "PropertyMeta",
    "PropertyRegistry",
    "image_config",
    "merge_plots",
    "property_registry",
]


class ColorScheme(str, Enum):
    """
    Color schemes for reservoir visualization.

    **Colorblind-Friendly Schemes (Recommended):**

    These color schemes are designed to be accessible to users with color vision deficiencies
    and are recommended for scientific visualization:

    - `VIRIDIS`: Perceptually uniform, colorblind-friendly (blue -> green -> yellow)
    - `CIVIDIS`: Optimized for colorblind accessibility, similar to viridis
    - `PLASMA`: Colorblind-friendly (purple -> pink -> yellow)
    - `INFERNO`: Colorblind-friendly (black -> purple -> yellow -> white)
    - `MAGMA`: Colorblind-friendly (black -> purple -> pink -> yellow)

    **Diverging Schemes:**

    Use these for data that diverges around a central value (e.g., pressure changes):

    - `RdBu`: Red-Blue diverging (warning: not ideal for red-green colorblindness)
    - `RdYlBu`: Red-Yellow-Blue diverging
    - `SPECTRAL`: Spectral diverging (warning: poor for colorblindness)
    - `BALANCE`: Balanced diverging, better accessibility

    **Other Schemes:**

    - `TURBO`: High-contrast rainbow (warning: not colorblind-friendly)
    - `EARTH`: Earth tones for geological visualization

    For maximum accessibility, prefer `VIRIDIS`, `CIVIDIS`, `PLASMA`, `INFERNO`, or `MAGMA`.
    """

    VIRIDIS = "viridis"
    """Perceptually uniform, colorblind-friendly (recommended)"""

    PLASMA = "plasma"
    """Colorblind-friendly sequential (recommended)"""

    INFERNO = "inferno"
    """Colorblind-friendly sequential (recommended)"""

    MAGMA = "magma"
    """Colorblind-friendly sequential (recommended)"""

    CIVIDIS = "cividis"
    """Optimized for colorblindness (recommended)"""

    TURBO = "turbo"
    """High-contrast rainbow (not colorblind-friendly)"""

    RdYlBu = "rdylbu"
    """Red-Yellow-Blue diverging"""

    RdBu = "rdbu"
    """Red-Blue diverging (caution: red-green colorblindness)"""

    SPECTRAL = "spectral"
    """Spectral diverging (not colorblind-friendly)"""

    BALANCE = "balance"
    """Balanced diverging with better accessibility"""

    EARTH = "earth"
    """Earth tones for geological visualization"""

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PropertyMeta:
    """Metadata for model properties used in property registry."""

    name: str
    """Internal grid property name in the ModelState object (e.g., 'pressure_grid', 'oil_viscosity_grid').
    This should match the actual attribute name in the simulation data structure."""

    display_name: str
    """Human-readable name shown in plot titles, legends, and hover text (e.g., 'Pressure', 'Oil Viscosity').
    This is what users see in the visualization interface."""

    unit: str
    """Physical unit of measurement displayed with values (e.g., 'psi', 'cP', '°F', 'fraction').
    Used in hover text, colorbar labels, and axis titles for proper scientific notation."""

    color_scheme: ColorScheme
    """Default color scheme for visualizing this property. Different properties use different color schemes
    optimized for their typical value ranges and physical meaning (e.g., pressure uses 'viridis',
    temperature uses 'inferno' for heat-like appearance)."""

    aliases: typing.Optional[typing.Sequence[str]] = None
    """Alternative names that can be used to refer to this property (e.g., 'pressure' for 'oil_pressure')."""

    log_scale: bool = False
    """Whether to apply logarithmic scaling (base 10) to the data before visualization.
    
    Use `log_scale=True` for properties that vary over many orders of magnitude, such as:
    - Viscosity (0.1 to 10,000+ cP)
    - Compressibility (1e-6 to 1e-3 psi⁻¹)
    - Permeability (0.001 to 10,000+ mD)
    
    When True:
    - Data values are transformed using log₁₀(value) for plotting
    - Color mapping and isosurfaces use log-transformed values
    - Hover text and colorbar show original physical values (not log values)
    - Zero/negative values are handled by replacing with small positive values
    
    Example: 0.5 cP viscosity becomes log₁₀(0.5) = -0.301 for plotting,
    but hover text still shows "0.5 cP" to the user."""

    min_val: typing.Optional[float] = None
    """Minimum value for data clipping and normalization. If specified along with max_val,
    all data values will be clipped to this range before visualization.
    
    Useful for:
    - Saturations (min_val=0, max_val=1) to ensure physically meaningful bounds
    - Removing outliers that might distort color scaling
    - Focusing visualization on a specific value range of interest
    
    Set to None for no minimum clipping."""

    max_val: typing.Optional[float] = None
    """Maximum value for data clipping and normalization. If specified along with min_val,
    all data values will be clipped to this range before visualization.
    
    Useful for:
    - Saturations (min_val=0, max_val=1) to ensure physically meaningful bounds
    - Removing outliers that might distort color scaling
    - Focusing visualization on a specific value range of interest
    
    Set to None for no maximum clipping."""


class PropertyRegistry:
    """Registry of all reservoir model and model state properties available for visualization."""

    _defaults = dict(  # noqa
        # Pressure and Temperature
        oil_pressure=PropertyMeta(
            name="model.fluid_properties.pressure_grid",
            display_name="Oil Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["pressure"],
        ),
        temperature=PropertyMeta(
            name="model.fluid_properties.temperature_grid",
            display_name="Temperature",
            unit="°F",
            color_scheme=ColorScheme.INFERNO,
        ),
        # Oil Properties
        oil_saturation=PropertyMeta(
            name="model.fluid_properties.oil_saturation_grid",
            display_name="Oil Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            min_val=0,
            max_val=1,
            aliases=["oil_sat"],
        ),
        oil_viscosity=PropertyMeta(
            name="model.fluid_properties.oil_viscosity_grid",
            display_name="Oil Viscosity",
            unit="cP",
            color_scheme=ColorScheme.INFERNO,
            log_scale=True,
        ),
        oil_effective_viscosity=PropertyMeta(
            name="model.fluid_properties.oil_effective_viscosity_grid",
            display_name="Oil Effective Viscosity [Oil + Solvent (if any)]",
            unit="cP",
            color_scheme=ColorScheme.INFERNO,
            log_scale=True,
        ),
        solvent_concentration=PropertyMeta(
            name="model.fluid_properties.solvent_concentration_grid",
            display_name="Solvent Concentration in Oil",
            unit="fraction",
            color_scheme=ColorScheme.PLASMA,
            aliases=["solvent_conc"],
        ),
        oil_density=PropertyMeta(
            name="model.fluid_properties.oil_density_grid",
            display_name="Oil Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.PLASMA,
        ),
        oil_effective_density=PropertyMeta(
            name="model.fluid_properties.oil_effective_density_grid",
            display_name="Oil Effective Density [Oil + Solvent (if any)]",
            unit="lbm/ft³",
            color_scheme=ColorScheme.PLASMA,
        ),
        oil_compressibility=PropertyMeta(
            name="model.fluid_properties.oil_compressibility_grid",
            display_name="Oil Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.TURBO,
            log_scale=True,
        ),
        oil_formation_volume_factor=PropertyMeta(
            name="model.fluid_properties.oil_formation_volume_factor_grid",
            display_name="Oil FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["oil_fvf"],
        ),
        oil_bubble_point_pressure=PropertyMeta(
            name="model.fluid_properties.oil_bubble_point_pressure_grid",
            display_name="Oil Bubble Point Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["oil_bpp"],
        ),
        # Water Properties
        water_saturation=PropertyMeta(
            name="model.fluid_properties.water_saturation_grid",
            display_name="Water Saturation",
            unit="fraction",
            color_scheme=ColorScheme.RdBu,
            min_val=0,
            max_val=1,
            aliases=["water_sat"],
        ),
        water_viscosity=PropertyMeta(
            name="model.fluid_properties.water_viscosity_grid",
            display_name="Water Viscosity",
            unit="cP",
            color_scheme=ColorScheme.CIVIDIS,
            log_scale=True,
        ),
        water_density=PropertyMeta(
            name="model.fluid_properties.water_density_grid",
            display_name="Water Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.EARTH,
        ),
        water_compressibility=PropertyMeta(
            name="model.fluid_properties.water_compressibility_grid",
            display_name="Water Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.VIRIDIS,
            log_scale=True,
        ),
        water_formation_volume_factor=PropertyMeta(
            name="model.fluid_properties.water_formation_volume_factor_grid",
            display_name="Water FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.PLASMA,
            aliases=["water_fvf"],
        ),
        water_bubble_point_pressure=PropertyMeta(
            name="model.fluid_properties.water_bubble_point_pressure_grid",
            display_name="Water Bubble Point Pressure",
            unit="psi",
            color_scheme=ColorScheme.RdBu,
            aliases=["water_bpp"],
        ),
        water_salinity=PropertyMeta(
            name="model.fluid_properties.water_salinity_grid",
            display_name="Water Salinity",
            unit="ppm NaCl",
            color_scheme=ColorScheme.CIVIDIS,
            aliases=["salinity"],
        ),
        gas_saturation=PropertyMeta(
            name="model.fluid_properties.gas_saturation_grid",
            display_name="Gas Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            min_val=0,
            max_val=1,
            aliases=["gas_sat"],
        ),
        gas_viscosity=PropertyMeta(
            name="model.fluid_properties.gas_viscosity_grid",
            display_name="Gas Viscosity",
            unit="cP",
            color_scheme=ColorScheme.TURBO,
            log_scale=True,
        ),
        gas_density=PropertyMeta(
            name="model.fluid_properties.gas_density_grid",
            display_name="Gas Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.RdYlBu,
        ),
        gas_compressibility=PropertyMeta(
            name="model.fluid_properties.gas_compressibility_grid",
            display_name="Gas Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.SPECTRAL,
            log_scale=True,
        ),
        gas_compressibility_factor=PropertyMeta(
            name="model.fluid_properties.gas_compressibility_factor_grid",
            display_name="Gas Compressibility Factor (Z)",
            unit="dimensionless",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["gas_z_factor", "z_factor", "z"],
        ),
        gas_formation_volume_factor=PropertyMeta(
            name="model.fluid_properties.gas_formation_volume_factor_grid",
            display_name="Gas FVF",
            unit="ft³/SCF",
            color_scheme=ColorScheme.RdBu,
            aliases=["gas_fvf"],
        ),
        gas_to_oil_ratio=PropertyMeta(
            name="model.fluid_properties.solution_gas_to_oil_ratio_grid",
            display_name="Gas-Oil Ratio",
            unit="SCF/STB",
            color_scheme=ColorScheme.TURBO,
            aliases=[
                "gor",
                "gas_oil_ratio",
                "solution_gas_oil_ratio",
                "solution_gor",
                "solution_gas_to_oil_ratio",
            ],
        ),
        gas_gravity=PropertyMeta(
            name="model.fluid_properties.gas_gravity_grid",
            display_name="Gas Gravity",
            unit="dimensionless",
            color_scheme=ColorScheme.EARTH,
            aliases=["gas_specific_gravity", "gas_sg"],
        ),
        gas_molecular_weight=PropertyMeta(
            name="model.fluid_properties.gas_molecular_weight_grid",
            display_name="Gas Molecular Weight",
            unit="g/mol",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["gas_mw"],
        ),
        # API Gravity
        oil_api_gravity=PropertyMeta(
            name="model.fluid_properties.oil_api_gravity_grid",
            display_name="Oil API Gravity",
            unit="°API",
            color_scheme=ColorScheme.PLASMA,
            aliases=["api_gravity", "api"],
        ),
        oil_specific_gravity=PropertyMeta(
            name="model.fluid_properties.oil_specific_gravity_grid",
            display_name="Oil Specific Gravity",
            unit="dimensionless",
            color_scheme=ColorScheme.INFERNO,
            aliases=["oil_sg", "oil_gravity"],
        ),
        thickness=PropertyMeta(
            name="model.thickness_grid",
            display_name="Cell Thickness",
            unit="ft",
            color_scheme=ColorScheme.CIVIDIS,
        ),
        permeability_x=PropertyMeta(
            name="model.rock_properties.absolute_permeability.x",
            display_name="Permeability X",
            unit="mD",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["kx", "perm_x"],
        ),
        permeability_y=PropertyMeta(
            name="model.rock_properties.absolute_permeability.y",
            display_name="Permeability Y",
            unit="mD",
            color_scheme=ColorScheme.TURBO,
            aliases=["ky", "perm_y"],
        ),
        permeability_z=PropertyMeta(
            name="model.rock_properties.absolute_permeability.z",
            display_name="Permeability Z",
            unit="mD",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["kz", "perm_z"],
        ),
        porosity=PropertyMeta(
            name="model.rock_properties.porosity_grid",
            display_name="Porosity",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
        ),
        net_to_gross_ratio=PropertyMeta(
            name="model.rock_properties.net_to_gross_grid",
            display_name="Net to Gross Ratio",
            unit="fraction",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["ngr", "ntg", "net_to_gross", "ntg_ratio"],
        ),
        irreducible_water_saturation=PropertyMeta(
            name="model.rock_properties.irreducible_water_saturation_grid",
            display_name="Irreducible Water Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            aliases=["swc", "irreducible_water_sat"],
        ),
        residual_oil_saturation_water=PropertyMeta(
            name="model.rock_properties.residual_oil_saturation_water_grid",
            display_name="Residual Oil Saturation (Water Flooded)",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            aliases=["sorw", "sor", "residual_oil_sat", "residual_oil_saturation"],
        ),
        residual_oil_saturation_gas=PropertyMeta(
            name="model.rock_properties.residual_oil_saturation_gas_grid",
            display_name="Residual Oil Saturation (Gas Flooded)",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            aliases=["sorg"],
        ),
        residual_gas_saturation=PropertyMeta(
            name="model.rock_properties.residual_gas_saturation_grid",
            display_name="Residual Gas Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            aliases=["sgr", "residual_gas_sat"],
        ),
        maximum_water_saturation=PropertyMeta(
            name="model.hysteresis_state.max_water_saturation_grid",
            display_name="Maximum Water Saturation",
            unit="fraction",
            color_scheme=ColorScheme.RdBu,
            aliases=["max_water_sat", "max_sw", "max_water_saturation"],
        ),
        maximum_gas_saturation=PropertyMeta(
            name="model.hysteresis_state.max_gas_saturation_grid",
            display_name="Maximum Gas Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            aliases=["max_gas_sat", "max_sg", "max_gas_saturation"],
        ),
        water_imbibition_map=PropertyMeta(
            name="model.hysteresis_state.water_imbibition_flag_grid",
            display_name="Water Imbibition Map",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            aliases=["water_imbibition"],
        ),
        gas_imbibition_map=PropertyMeta(
            name="model.hysteresis_state.gas_imbibition_flag_grid",
            display_name="Gas Imbibition Map",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            aliases=["gas_imbibition"],
        ),
        oil_relative_permeability=PropertyMeta(
            name="relative_permeabilities.kro",
            display_name="Oil Relative Permeability",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            aliases=["kro", "oil_rel_perm"],
        ),
        water_relative_permeability=PropertyMeta(
            name="relative_permeabilities.krw",
            display_name="Water Relative Permeability",
            unit="fraction",
            color_scheme=ColorScheme.RdBu,
            aliases=["krw", "water_rel_perm"],
        ),
        gas_relative_permeability=PropertyMeta(
            name="relative_permeabilities.krg",
            display_name="Gas Relative Permeability",
            unit="fraction",
            color_scheme=ColorScheme.TURBO,
            aliases=["krg", "gas_rel_perm"],
        ),
        oil_relative_mobility=PropertyMeta(
            name="relative_mobilities.oil_relative_mobility",
            display_name="Oil Relative Mobility",
            unit="cP⁻¹",
            color_scheme=ColorScheme.MAGMA,
            aliases=["moil", "oil_rel_mobility"],
        ),
        water_relative_mobility=PropertyMeta(
            name="relative_mobilities.water_relative_mobility",
            display_name="Water Relative Mobility",
            unit="cP⁻¹",
            color_scheme=ColorScheme.RdBu,
            aliases=["mwater", "water_rel_mobility"],
        ),
        gas_relative_mobility=PropertyMeta(
            name="relative_mobilities.gas_relative_mobility",
            display_name="Gas Relative Mobility",
            unit="cP⁻¹",
            color_scheme=ColorScheme.TURBO,
            aliases=["mgas", "gas_rel_mobility"],
        ),
        oil_water_capillary_pressure=PropertyMeta(
            name="capillary_pressures.oil_water_capillary_pressure",
            display_name="Oil-Water Capillary Pressure",
            unit="psi",
            color_scheme=ColorScheme.TURBO,
            aliases=["ow_cap", "oil_water_capillary", "pcow"],
        ),
        gas_oil_capillary_pressure=PropertyMeta(
            name="capillary_pressures.gas_oil_capillary_pressure",
            display_name="Gas-Oil Capillary Pressure",
            unit="psi",
            color_scheme=ColorScheme.TURBO,
            aliases=["go_cap", "gas_oil_capillary", "pcgo"],
        ),
        oil_mass=PropertyMeta(
            name="model.fluid_properties.oil_mass_grid",
            display_name="Oil Phase Mass",
            unit="lbm",
            color_scheme=ColorScheme.MAGMA,
            aliases=["oil_phase_mass"],
        ),
        water_mass=PropertyMeta(
            name="model.fluid_properties.water_mass_grid",
            display_name="Water Phase Mass",
            unit="lbm",
            color_scheme=ColorScheme.RdBu,
            aliases=["water_phase_mass"],
        ),
        total_gas_mass=PropertyMeta(
            name="model.fluid_properties.total_gas_mass_grid",
            display_name="Total Gas Mass",
            unit="lbm",
            color_scheme=ColorScheme.TURBO,
            aliases=["gas_phase_mass"],
        ),
        free_gas_mass=PropertyMeta(
            name="model.fluid_properties.free_gas_mass_grid",
            display_name="Free Gas Component Mass",
            unit="lbm",
            color_scheme=ColorScheme.TURBO,
            aliases=["free_gas_component_mass"],
        ),
        dissolved_gas_mass_in_oil=PropertyMeta(
            name="model.fluid_properties.dissolved_gas_mass_in_oil_grid",
            display_name="Dissolved Gas Component Mass in Oil",
            unit="lbm",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["dissolved_gas_component_mass"],
        ),
        dissolved_gas_mass_in_water=PropertyMeta(
            name="model.fluid_properties.dissolved_gas_mass_in_water_grid",
            display_name="Dissolved Gas Component Mass in Water",
            unit="lbm",
            color_scheme=ColorScheme.CIVIDIS,
            aliases=["dissolved_gas_component_mass_in_water"],
        ),
    )

    def __init__(self) -> None:
        """
        Initialize the registry.
        """
        defaults = type(self)._defaults
        self._count = len(defaults)
        self._properties = _explode_properties(defaults)

    @property
    def properties(self) -> typing.List[str]:
        """
        A list of all available property names.

        :return: List of property names that can be used for visualization
        """
        return list(self._properties.keys())

    @property
    def count(self) -> int:
        return self._count

    @staticmethod
    def _clean_name(name: str) -> str:
        """Clean and standardize property name for lookup."""
        return name.strip().replace("-", "_").replace(" ", "_").lower()

    def get(self, name: str) -> PropertyMeta:
        """
        Get metadata for a specific property.

        :param property: Name of the property to get metadata for
        :return: `PropertyMeta` object containing display information
        :raises ValueError: If property is not found in the registry
        """
        name = self._clean_name(name)
        if name not in self._properties:
            raise ValueError(
                f"Unknown property: {name}. Available: {', '.join(self.properties)}"
            )
        return self._properties[name]

    def __getitem__(self, name: str, /) -> PropertyMeta:
        return self.get(name)

    def __setitem__(self, name: str, value: PropertyMeta, /) -> None:
        if not isinstance(value, PropertyMeta):
            raise TypeError("Value must be a `PropertyMeta` instance")

        name = self._clean_name(name)
        self._properties[name] = value
        self._count += 1

    def __contains__(self, name: str, /) -> bool:
        """Check if a property exists in the registry."""
        if self._count == 0:
            return False

        name = self._clean_name(name)
        return name in self._properties

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over the property names."""
        return iter(self._properties.keys())


def _explode_properties(
    properties: typing.Dict[str, PropertyMeta],
) -> typing.Dict[str, PropertyMeta]:
    """Return a new dict with all aliases expanded to point to the same metadata."""
    expanded = {}
    for prop_name, metadata in properties.items():
        expanded[prop_name] = metadata
        if metadata.aliases:
            for alias in metadata.aliases:
                alias_clean = PropertyRegistry._clean_name(alias)
                if alias_clean in properties or alias_clean in expanded:
                    raise ValueError(
                        f"Alias '{alias}' for property '{prop_name}' conflicts with existing property or alias."
                    )
                expanded[alias_clean] = metadata
    return expanded


property_registry = PropertyRegistry()


def _copy_scene_properties(
    source_fig: go.Figure,
    target_fig: go.Figure,
    scene_key: str,
) -> None:
    """
    Copy all 3D scene properties from source figure to target figure.

    This function performs a comprehensive copy of all scene-related properties including
    camera settings, axis configurations, annotations, backgrounds, and interaction modes.

    :param source_fig: Source Plotly Figure to copy scene properties from
    :param target_fig: Target Plotly Figure to copy scene properties to
    :param scene_key: Scene key in the target layout (e.g., 'scene', 'scene2', 'scene3')
    """
    if not hasattr(source_fig, "layout") or not hasattr(source_fig.layout, "scene"):
        return

    source_scene = source_fig.layout.scene  # type: ignore[attr-defined]
    if not source_scene:
        return

    target_scene = target_fig.layout[scene_key]

    # Camera settings (viewpoint, projection)
    if hasattr(source_scene, "camera") and source_scene.camera:
        target_scene.camera = source_scene.camera

    # Axis properties (labels, ranges, gridlines, etc.)
    if hasattr(source_scene, "xaxis") and source_scene.xaxis:
        target_scene.xaxis = source_scene.xaxis

    if hasattr(source_scene, "yaxis") and source_scene.yaxis:
        target_scene.yaxis = source_scene.yaxis

    if hasattr(source_scene, "zaxis") and source_scene.zaxis:
        target_scene.zaxis = source_scene.zaxis

    # Aspect ratio and scaling
    if hasattr(source_scene, "aspectmode") and source_scene.aspectmode:
        target_scene.aspectmode = source_scene.aspectmode

    if hasattr(source_scene, "aspectratio") and source_scene.aspectratio:
        target_scene.aspectratio = source_scene.aspectratio

    # Background and grid settings
    if hasattr(source_scene, "bgcolor") and source_scene.bgcolor:
        target_scene.bgcolor = source_scene.bgcolor

    # Interaction modes
    if hasattr(source_scene, "dragmode") and source_scene.dragmode:
        target_scene.dragmode = source_scene.dragmode

    if hasattr(source_scene, "hovermode") and source_scene.hovermode:
        target_scene.hovermode = source_scene.hovermode

    # Domain (position in the layout)
    if hasattr(source_scene, "domain") and source_scene.domain:
        # Note: Domain is usually managed by make_subplots, so we may skip this
        # to avoid conflicts, but include it for completeness
        pass

    # Annotations specific to this scene
    if hasattr(source_scene, "annotations") and source_scene.annotations:
        target_scene.annotations = source_scene.annotations

    # Uirevision (for preserving user interactions)
    if hasattr(source_scene, "uirevision") and source_scene.uirevision:
        target_scene.uirevision = source_scene.uirevision


def _copy_2d_axis_properties(
    source_axis: typing.Any,
    target_axis: typing.Any,
) -> None:
    """
    Copy all 2D axis properties from source axis to target axis.

    This function performs a comprehensive copy of all axis-related properties including
    titles, ranges, colors, gridlines, tick settings, and fonts.

    :param source_axis: Source axis object to copy properties from
    :param target_axis: Target axis object to copy properties to
    """
    # Title
    if hasattr(source_axis, "title") and source_axis.title:
        target_axis.title = source_axis.title

    # Axis type (linear, log, date, category)
    if hasattr(source_axis, "type") and source_axis.type:
        target_axis.type = source_axis.type

    # Range
    if hasattr(source_axis, "range") and source_axis.range:
        target_axis.range = source_axis.range

    # Autorange
    if hasattr(source_axis, "autorange"):
        target_axis.autorange = source_axis.autorange

    # Fixed range (disable zoom/pan)
    if hasattr(source_axis, "fixedrange"):
        target_axis.fixedrange = source_axis.fixedrange

    # Colors
    if hasattr(source_axis, "color") and source_axis.color:
        target_axis.color = source_axis.color

    if hasattr(source_axis, "gridcolor") and source_axis.gridcolor:
        target_axis.gridcolor = source_axis.gridcolor

    if hasattr(source_axis, "zerolinecolor") and source_axis.zerolinecolor:
        target_axis.zerolinecolor = source_axis.zerolinecolor

    if hasattr(source_axis, "linecolor") and source_axis.linecolor:
        target_axis.linecolor = source_axis.linecolor

    # Line widths
    if hasattr(source_axis, "gridwidth") and source_axis.gridwidth:
        target_axis.gridwidth = source_axis.gridwidth

    if hasattr(source_axis, "zerolinewidth") and source_axis.zerolinewidth:
        target_axis.zerolinewidth = source_axis.zerolinewidth

    if hasattr(source_axis, "linewidth") and source_axis.linewidth:
        target_axis.linewidth = source_axis.linewidth

    # Show/hide settings
    if hasattr(source_axis, "showgrid"):
        target_axis.showgrid = source_axis.showgrid

    if hasattr(source_axis, "zeroline"):
        target_axis.zeroline = source_axis.zeroline

    if hasattr(source_axis, "showline"):
        target_axis.showline = source_axis.showline

    if hasattr(source_axis, "showticklabels"):
        target_axis.showticklabels = source_axis.showticklabels

    # Tick settings
    if hasattr(source_axis, "ticks") and source_axis.ticks:
        target_axis.ticks = source_axis.ticks

    if hasattr(source_axis, "tickmode") and source_axis.tickmode:
        target_axis.tickmode = source_axis.tickmode

    if hasattr(source_axis, "tick0"):
        target_axis.tick0 = source_axis.tick0

    if hasattr(source_axis, "dtick"):
        target_axis.dtick = source_axis.dtick

    if hasattr(source_axis, "tickvals"):
        target_axis.tickvals = source_axis.tickvals

    if hasattr(source_axis, "ticktext"):
        target_axis.ticktext = source_axis.ticktext

    if hasattr(source_axis, "tickangle"):
        target_axis.tickangle = source_axis.tickangle

    if hasattr(source_axis, "tickcolor") and source_axis.tickcolor:
        target_axis.tickcolor = source_axis.tickcolor

    if hasattr(source_axis, "ticklen"):
        target_axis.ticklen = source_axis.ticklen

    if hasattr(source_axis, "tickwidth"):
        target_axis.tickwidth = source_axis.tickwidth

    if hasattr(source_axis, "tickformat") and source_axis.tickformat:
        target_axis.tickformat = source_axis.tickformat

    if hasattr(source_axis, "tickprefix") and source_axis.tickprefix:
        target_axis.tickprefix = source_axis.tickprefix

    if hasattr(source_axis, "ticksuffix") and source_axis.ticksuffix:
        target_axis.ticksuffix = source_axis.ticksuffix

    # Font settings
    if hasattr(source_axis, "tickfont") and source_axis.tickfont:
        target_axis.tickfont = source_axis.tickfont

    # Spikes (hover lines)
    if hasattr(source_axis, "showspikes"):
        target_axis.showspikes = source_axis.showspikes

    if hasattr(source_axis, "spikecolor") and source_axis.spikecolor:
        target_axis.spikecolor = source_axis.spikecolor

    if hasattr(source_axis, "spikethickness"):
        target_axis.spikethickness = source_axis.spikethickness

    if hasattr(source_axis, "spikedash") and source_axis.spikedash:
        target_axis.spikedash = source_axis.spikedash

    if hasattr(source_axis, "spikemode") and source_axis.spikemode:
        target_axis.spikemode = source_axis.spikemode

    # Side (placement of axis)
    if hasattr(source_axis, "side") and source_axis.side:
        target_axis.side = source_axis.side

    # Overlaying
    if hasattr(source_axis, "overlaying") and source_axis.overlaying:
        target_axis.overlaying = source_axis.overlaying

    # Mirror
    if hasattr(source_axis, "mirror"):
        target_axis.mirror = source_axis.mirror

    # Anchor
    if hasattr(source_axis, "anchor") and source_axis.anchor:
        target_axis.anchor = source_axis.anchor


def _calculate_colorbar_position(
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> typing.Dict[str, typing.Any]:
    """
    Calculate smart colorbar positioning for subplot grids.

    This helper function computes the optimal position and size for colorbars
    in multi-subplot layouts to prevent overlap and maintain visual clarity.

    :param row: Current subplot row (1-indexed)
    :param col: Current subplot column (1-indexed)
    :param rows: Total number of rows in the grid
    :param cols: Total number of columns in the grid
    :return: Dictionary with colorbar position and size parameters
    """
    # Calculate horizontal position based on column
    col_end = col / cols
    colorbar_x_position = col_end - _COLORBAR_BOUNDARY_PADDING

    # Calculate vertical position based on row (inverted because plotly counts from bottom)
    row_fraction = (rows - row + 1) / rows
    subplot_height = 1.0 / rows
    colorbar_y_position = row_fraction - subplot_height / 2

    # Calculate colorbar length with vertical padding for gaps
    colorbar_length = max(
        _COLORBAR_MIN_LENGTH,
        (subplot_height * _COLORBAR_HEIGHT_FACTOR) - (2 * _COLORBAR_VERTICAL_PADDING),
    )

    # Scale thickness based on number of columns to prevent overlap
    colorbar_thickness = min(
        _COLORBAR_MIN_THICKNESS, int(_COLORBAR_BASE_THICKNESS / cols)
    )

    return {
        "x": colorbar_x_position,
        "y": colorbar_y_position,
        "len": colorbar_length,
        "thickness": colorbar_thickness,
        "xanchor": "left",
        "yanchor": "middle",
        "xpad": _COLORBAR_XPAD,
    }


def merge_plots(
    *figures: go.Figure,
    rows: typing.Optional[int] = None,
    cols: typing.Optional[int] = None,
    subplot_titles: typing.Optional[typing.Sequence[str]] = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    vertical_spacing: typing.Optional[float] = None,
    horizontal_spacing: typing.Optional[float] = None,
    height: typing.Optional[int] = None,
    width: typing.Optional[int] = None,
    title: typing.Optional[str] = None,
    show_legend: bool = True,
    hovermode: typing.Optional[str] = None,
) -> go.Figure:
    """
    Display multiple Plotly figures in a single unified plot with subplots.

    This function combines multiple independent Plotly figures into a single figure
    with subplots arranged in a grid layout. It intelligently handles figure layout,
    traces, annotations, and preserves original styling while creating a cohesive
    multi-panel visualization.

    **Layout Strategy:**
    - If `rows` and `cols` are not specified, automatically calculates optimal grid dimensions
    - Arranges figures left-to-right, top-to-bottom in the subplot grid
    - Preserves individual figure properties (colorbars, annotations, axes labels)
    - Adjusts spacing based on number of subplots for optimal viewing

    **Use Cases:**
    - Compare multiple reservoir properties side-by-side
    - Create multi-panel diagnostic plots
    - Combine pressure, saturation, and production rate visualizations
    - Build comprehensive reservoir analysis dashboards

    :param figures: Variable number of Plotly Figure objects to display together.
        Each figure is treated as an independent subplot. Empty figures are skipped.
    :param rows: Number of rows in the subplot grid. If None, automatically calculated
        based on number of figures and cols parameter.
    :param cols: Number of columns in the subplot grid. If None, automatically calculated
        to create a roughly square grid layout.
    :param subplot_titles: Sequence of titles for each subplot. Length must match number
        of figures. If None, uses original figure titles if available.
    :param shared_xaxes: If True, all subplots share the same x-axis range. Useful for
        comparing data over the same spatial or temporal domain (default: False).
    :param shared_yaxes: If True, all subplots share the same y-axis range. Useful for
        comparing magnitudes across different properties (default: False).
    :param vertical_spacing: Vertical spacing between subplot rows as a fraction of
        plot height (0 to 1). If None, automatically calculated based on grid size.
        Smaller values create tighter layouts.
    :param horizontal_spacing: Horizontal spacing between subplot columns as a fraction
        of plot width (0 to 1). If None, automatically calculated based on grid size.
        Smaller values create tighter layouts.
    :param height: Total height of the combined figure in pixels. If None, scales
        automatically based on number of rows (default: 400 * rows).
    :param width: Total width of the combined figure in pixels. If None, scales
        automatically based on number of columns (default: 600 * cols).
    :param title: Main title for the entire combined figure. Displayed at the top
        of the visualization. If None, no main title is shown.
    :param show_legend: If True, displays legends for all traces. If False, hides
        all legends to reduce clutter (default: True).
    :param hovermode: Hover interaction mode for the plot. Options include:
        - "x unified": All traces show hover info together when hovering (2D only)
        - "x": Hover shows closest point on x-axis per trace
        - "y": Hover shows closest point on y-axis per trace
        - "closest": Hover shows single closest point across all traces
        - False/None: No hover interaction
        Default is None, which preserves original figure hover modes.
    :return: Combined Plotly Figure object with all input figures as subplots.
        Can be displayed with .show() or saved with .write_html() / .write_image().
    :raises ValueError: If subplot_titles length doesn't match number of figures,
        or if rows * cols is insufficient for the number of figures.
    :raises TypeError: If any input is not a valid Plotly Figure object.

    Example Usage:

    ```python
    # Basic usage with automatic layout:
    fig1 = create_pressure_plot(...)
    fig2 = create_saturation_plot(...)
    combined = merge_plots(fig1, fig2)
    combined.show()

    # Custom grid with titles:
    merge_plots(
        pressure_fig,
        oil_sat_fig,
        water_sat_fig,
        gas_sat_fig,
        rows=2,
        cols=2,
        subplot_titles=["Pressure", "Oil Sat", "Water Sat", "Gas Sat"],
        title="Reservoir Properties at Time Step 100",
        height=800,
        width=1200,
    )

    # Shared axes for comparison:
    merge_plots(
        timestep_0_fig,
        timestep_50_fig,
        timestep_100_fig,
        cols=3,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=["Initial", "Mid", "Final"],
    )
    ```
    """
    if not figures:
        raise ValueError("At least one figure must be provided")

    # Filter out None values and validate figure types
    valid_figures = []
    for idx, fig in enumerate(figures):
        if fig is None:
            continue
        if not isinstance(fig, go.Figure):
            raise TypeError(
                f"Argument at position {idx} is not a Plotly Figure object. "
                f"Got type: {type(fig).__name__}"
            )
        # Skip empty figures (no traces)
        if len(fig.data) > 0:  # type: ignore
            valid_figures.append(fig)

    if not valid_figures:
        raise ValueError("No valid figures with data to display")

    num_figures = len(valid_figures)

    # Calculate optimal grid dimensions if not specified
    if rows is None and cols is None:
        # Create roughly square grid, slightly favoring more columns
        cols = int(np.ceil(np.sqrt(num_figures)))
        rows = int(np.ceil(num_figures / cols))
    elif rows is None:
        if cols is None or cols <= 0:
            raise ValueError("cols must be a positive integer")
        rows = int(np.ceil(num_figures / cols))
    elif cols is None:
        if rows <= 0:
            raise ValueError("rows must be a positive integer")
        cols = int(np.ceil(num_figures / rows))

    cols = typing.cast(int, cols)
    # Validate grid can accommodate all figures
    if rows * cols < num_figures:
        raise ValueError(
            f"Grid size ({rows} rows x {cols} cols = {rows * cols} subplots) "
            f"is insufficient for {num_figures} figures. "
            f"Need at least {num_figures} subplot positions."
        )

    # Validate `subplot_titles` if provided
    if subplot_titles is not None:
        if len(subplot_titles) != num_figures:
            raise ValueError(
                f"Length of subplot_titles ({len(subplot_titles)}) must match "
                f"number of valid figures ({num_figures})"
            )
    else:
        # Try to extract titles from original figures
        subplot_titles = []
        for fig in valid_figures:
            fig_title = ""
            if hasattr(fig, "layout") and hasattr(fig.layout, "title"):
                if isinstance(fig.layout.title, dict):
                    fig_title = fig.layout.title.get("text", "")
                elif hasattr(fig.layout.title, "text"):
                    fig_title = fig.layout.title.text or ""
                else:
                    fig_title = str(fig.layout.title) if fig.layout.title else ""
            subplot_titles.append(fig_title)

    # Calculate automatic spacing if not specified
    if vertical_spacing is None:
        # More spacing for fewer rows, less for many rows
        vertical_spacing = max(0.05, min(0.15, 0.3 / rows))

    if horizontal_spacing is None:
        # More spacing for fewer columns, less for many columns
        horizontal_spacing = max(0.05, min(0.15, 0.3 / cols))

    # Calculate automatic dimensions if not specified
    if height is None:
        height = int(400 * rows)

    if width is None:
        width = int(600 * cols)

    # Determine subplot types and check if any figures have 3D traces
    specs = [[{"type": "xy"} for _ in range(cols)] for _ in range(rows)]
    for idx, fig in enumerate(valid_figures):
        row = idx // cols
        col = idx % cols

        # Check if figure contains 3D traces (volume, isosurface, scatter3d, surface, mesh3d, cone)
        has_3d = any(
            hasattr(trace, "type")
            and str(trace.type).lower()
            in (
                "volume",
                "isosurface",
                "scatter3d",
                "surface",
                "mesh3d",
                "cone",
                "streamtube",
            )
            for trace in fig.data
        )
        if has_3d:
            specs[row][col] = {"type": "scene"}

    # Create subplot figure
    try:
        combined_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(subplot_titles),
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            specs=specs,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to create subplot layout: {exc}. "
            f"Check that your grid dimensions ({rows}x{cols}) and spacing parameters are valid."
        ) from exc

    # Track subplot-specific background colors and templates
    subplot_backgrounds: typing.Dict[
        typing.Tuple[int, int], typing.Dict[str, typing.Any]
    ] = {}
    subplot_fonts: typing.Dict[
        typing.Tuple[int, int], typing.Dict[str, typing.Any]
    ] = {}

    # Add traces from each figure to the combined figure
    for idx, fig in enumerate(valid_figures):
        row = idx // cols + 1
        col = idx % cols + 1

        # Add all traces from the figure
        for trace in fig.data:
            # Create a copy to avoid modifying original
            trace_copy = trace.__class__(trace)

            # Preserve legend group if it exists
            if not show_legend:
                trace_copy.showlegend = False
            elif hasattr(trace, "legendgroup") and trace.legendgroup:
                # Append subplot index to legend group to avoid conflicts
                trace_copy.legendgroup = f"{trace.legendgroup}_subplot{idx}"

            combined_fig.add_trace(trace_copy, row=row, col=col)

        # Copy axis properties from original figure
        if hasattr(fig, "layout"):
            # Check if this subplot is a 3D scene
            is_scene = specs[row - 1][col - 1].get("type") == "scene"

            if is_scene:
                # Copy all the 3D scene properties comprehensively
                scene_key = (
                    "scene"
                    if row == 1 and col == 1
                    else f"scene{(row - 1) * cols + col}"
                )
                _copy_scene_properties(fig, combined_fig, scene_key)
            else:
                # Copy 2D x-axis and y-axis properties
                if hasattr(fig.layout, "xaxis") and fig.layout.xaxis:
                    xaxis_key = (
                        "xaxis"
                        if row == 1 and col == 1
                        else f"xaxis{(row - 1) * cols + col}"
                    )
                    _copy_2d_axis_properties(
                        fig.layout.xaxis, combined_fig.layout[xaxis_key]
                    )

                if hasattr(fig.layout, "yaxis") and fig.layout.yaxis:
                    yaxis_key = (
                        "yaxis"
                        if row == 1 and col == 1
                        else f"yaxis{(row - 1) * cols + col}"
                    )
                    _copy_2d_axis_properties(
                        fig.layout.yaxis, combined_fig.layout[yaxis_key]
                    )

                # Store background colors and font settings for this subplot
                if hasattr(fig.layout, "plot_bgcolor") and fig.layout.plot_bgcolor:
                    subplot_backgrounds[(row, col)] = {
                        "plot_bgcolor": fig.layout.plot_bgcolor,
                    }
                if hasattr(fig.layout, "paper_bgcolor") and fig.layout.paper_bgcolor:
                    if (row, col) not in subplot_backgrounds:
                        subplot_backgrounds[(row, col)] = {}
                    subplot_backgrounds[(row, col)]["paper_bgcolor"] = (
                        fig.layout.paper_bgcolor
                    )

                # Store font settings for this subplot
                if hasattr(fig.layout, "font") and fig.layout.font:
                    subplot_fonts[(row, col)] = fig.layout.font.to_plotly_json()

            # Copy colorbar settings for heatmaps/contours with smart positioning
            for trace_idx, trace in enumerate(fig.data):
                if hasattr(trace, "colorbar") and trace.colorbar:
                    # Find corresponding trace in combined figure
                    combined_trace_idx = (
                        sum(len(valid_figures[i].data) for i in range(idx)) + trace_idx
                    )
                    if combined_trace_idx < len(combined_fig.data):  # type: ignore
                        # Create a copy of colorbar to avoid modifying original
                        colorbar_dict = (
                            dict(trace.colorbar)
                            if isinstance(trace.colorbar, dict)
                            else {}
                        )

                        # Calculate smart colorbar positioning for this subplot
                        colorbar_position = _calculate_colorbar_position(
                            row=row, col=col, rows=rows, cols=cols
                        )

                        # Set colorbar position and size to fit within subplot bounds with gaps
                        colorbar_dict.update(colorbar_position)
                        combined_fig.data[combined_trace_idx].colorbar = colorbar_dict

    # Apply subplot-specific background colors using shapes
    # Plotly's `plot_bgcolor` is global, so we use shapes to create per-subplot backgrounds
    shapes = []
    for (row, col), bg_settings in subplot_backgrounds.items():
        if "plot_bgcolor" in bg_settings:
            # Get the axis references for this subplot
            xaxis_key = "x" if row == 1 and col == 1 else f"x{(row - 1) * cols + col}"
            yaxis_key = "y" if row == 1 and col == 1 else f"y{(row - 1) * cols + col}"

            # Add a rectangle shape covering the subplot area using axis domain coordinates
            shapes.append(
                {
                    "type": "rect",
                    "xref": f"{xaxis_key} domain",
                    "yref": f"{yaxis_key} domain",
                    "x0": 0,
                    "y0": 0,
                    "x1": 1,
                    "y1": 1,
                    "fillcolor": bg_settings["plot_bgcolor"],
                    "layer": "below",
                    "line": {"width": 0},
                }
            )

    # Apply font settings per subplot
    for (row, col), font_settings in subplot_fonts.items():
        combined_fig.update_xaxes(row=row, col=col, tickfont=font_settings)
        combined_fig.update_yaxes(
            row=row, col=col, tickfont=font_settings
        )  # Apply global layout settings
    combined_fig.update_layout(
        height=height,
        width=width,
        showlegend=show_legend,
    )

    # Add background color shapes if any were created
    if shapes:
        combined_fig.update_layout(shapes=shapes)

    # Set hovermode if specified (for 2D plots with unified hover behavior)
    if hovermode is not None:
        combined_fig.update_layout(hovermode=hovermode)

    # Add main title if provided
    if title:
        combined_fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
    return combined_fig


def image_config(
    fmt: typing.Literal["png", "jpeg", "webp", "svg", "pdf"] = "png",
    scale: int = 2,
    width: typing.Optional[int] = None,
    height: typing.Optional[int] = None,
) -> None:
    """
    Configure image export settings for a Plotly figure.

    :param fig: Plotly Figure object to configure
    :param file_path: Path to the output image file
    :param fmt: Image format to save as (e.g., 'png', 'jpeg', 'webp', 'svg', 'pdf')
        Default is 'png'.
    :param scale: Scale factor for the image resolution. Default is 2 (double size).
    :param width: Optional width of the output image in pixels. If None, uses figure width.
    :param height: Optional height of the output image in pixels. If None, uses figure height.
    """
    pio.defaults.default_format = fmt
    pio.defaults.default_scale = scale
    if width is not None:
        pio.defaults.default_width = width
    if height is not None:
        pio.defaults.default_height = height
