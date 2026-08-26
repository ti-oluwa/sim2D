"""Tests for visualization base module (property registry, colorbars, etc.)."""

import pytest

from bores.visualization.base import (
    ColorScheme,
    PropertyMeta,
    PropertyRegistry,
    property_registry,
)


class TestColorScheme:
    """Tests for ColorScheme enum."""

    def test_color_scheme_values(self):
        """Test that color schemes have correct values."""
        assert ColorScheme.VIRIDIS.value == "viridis"
        assert ColorScheme.PLASMA.value == "plasma"
        assert ColorScheme.INFERNO.value == "inferno"

    def test_color_scheme_str(self):
        """Test string conversion of color schemes."""
        assert str(ColorScheme.VIRIDIS) == "viridis"


class TestPropertyMeta:
    """Tests for PropertyMeta dataclass."""

    def test_create_property_meta(self):
        """Test creating PropertyMeta instance."""
        meta = PropertyMeta(
            name="pressure_grid",
            display_name="Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
        )
        assert meta.name == "pressure_grid"
        assert meta.display_name == "Pressure"
        assert meta.unit == "psi"
        assert not meta.log_scale

    def test_property_meta_with_log_scale(self):
        """Test PropertyMeta with log scale."""
        meta = PropertyMeta(
            name="viscosity",
            display_name="Viscosity",
            unit="cP",
            color_scheme=ColorScheme.INFERNO,
            log_scale=True,
        )
        assert meta.log_scale

    def test_property_meta_with_min_max(self):
        """Test PropertyMeta with min/max clipping."""
        meta = PropertyMeta(
            name="saturation",
            display_name="Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            min_val=0.0,
            max_val=1.0,
        )
        assert meta.min_val == 0
        assert meta.max_val == 1.0

    def test_property_meta_with_aliases(self):
        """Test PropertyMeta with aliases."""
        meta = PropertyMeta(
            name="pressure",
            display_name="Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["p", "pres"],
        )
        assert meta.aliases is not None
        assert "p" in meta.aliases
        assert "pres" in meta.aliases


class TestPropertyRegistry:
    """Tests for PropertyRegistry class."""

    def test_registry_has_properties(self):
        """Test that registry contains expected properties."""
        assert "oil_pressure" in property_registry
        assert "water_saturation" in property_registry
        assert "gas_saturation" in property_registry

    def test_registry_get_property(self):
        """Test getting property from registry."""
        meta = property_registry.get("oil_pressure")
        assert meta.display_name == "Oil Pressure"
        assert meta.unit == "psi"

    def test_registry_get_with_alias(self):
        """Test getting property using alias."""
        meta = property_registry.get("pressure")
        assert meta.display_name == "Oil Pressure"

    def test_registry_get_unknown_property_raises(self):
        """Test that getting unknown property raises ValueError."""
        with pytest.raises(ValueError, match="Unknown property"):
            property_registry.get("nonexistent_property")

    def test_registry_contains(self):
        """Test __contains__ method."""
        assert "oil_pressure" in property_registry
        assert "pressure" in property_registry  # alias
        assert "nonexistent" not in property_registry

    def test_registry_iteration(self):
        """Test iterating over registry."""
        properties = list(property_registry)
        assert "oil_pressure" in properties
        assert "water_saturation" in properties

    def test_registry_properties_list(self):
        """Test getting list of all properties."""
        props = property_registry.properties
        assert isinstance(props, list)
        assert len(props) > 0

    def test_registry_count(self):
        """Test registry count."""
        assert property_registry.count > 0

    def test_registry_clean_name(self):
        """Test name cleaning."""
        clean = PropertyRegistry._clean_name("Oil-Pressure ")
        assert clean == "oil_pressure"

    def test_registry_setitem(self):
        """Test adding new property to registry."""
        registry = PropertyRegistry()
        new_meta = PropertyMeta(
            name="custom",
            display_name="Custom Property",
            unit="units",
            color_scheme=ColorScheme.VIRIDIS,
        )
        registry["custom_property"] = new_meta
        assert "custom_property" in registry
        retrieved = registry.get("custom_property")
        assert retrieved.display_name == "Custom Property"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
