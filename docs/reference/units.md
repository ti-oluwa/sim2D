# Units

## Overview

BORES uses oil-field units throughout the entire framework. This is the most common unit system in petroleum engineering in North America, and it is the system used by the empirical correlations that BORES implements (Standing, Vazquez-Beggs, Hall-Yarborough, etc.). Using a consistent unit system eliminates the need for unit conversions inside the simulator and reduces the risk of conversion errors.

If you are accustomed to SI units, you will need to convert your input data to oil-field units before passing it to BORES. The conversion tables below cover the most common quantities. There is no built-in unit conversion system in BORES, so conversions must be done externally (using NumPy, pint, or manual multiplication).

---

## Standard Unit System

### Pressure

| Quantity | Unit | Symbol |
| --- | --- | --- |
| Reservoir pressure | pounds per square inch | psi |
| Bottom-hole pressure | pounds per square inch | psi |
| Capillary pressure | pounds per square inch | psi |
| Standard pressure | 14.696 psi | psi |

### Temperature

| Quantity | Unit | Symbol |
| --- | --- | --- |
| Reservoir temperature | degrees Fahrenheit | F |
| Standard temperature | 60.0 F | F |

!!! note "Temperature in Correlations"

    Some internal calculations convert to Rankine ($R = F + 459.67$) for absolute temperature ratios. You always provide temperatures in Fahrenheit; the conversion happens internally.

### Length

| Quantity | Unit | Symbol |
| --- | --- | --- |
| Grid cell dimensions | feet | ft |
| Reservoir depth | feet | ft |
| Wellbore radius | feet | ft |
| Formation thickness | feet | ft |

### Area and Volume

| Quantity | Unit | Symbol |
| --- | --- | --- |
| Cross-sectional area | square feet | ft² |
| Pore volume | reservoir barrels | bbl |
| Grid cell volume | cubic feet | ft³ |

### Flow Rates

| Quantity | Unit | Symbol |
| --- | --- | --- |
| Oil rate (reservoir) | barrels per day | bbl/day |
| Oil rate (surface) | stock-tank barrels per day | STB/day |
| Gas rate (reservoir) | cubic feet per day | ft³/day |
| Gas rate (surface) | standard cubic feet per day | SCF/day |
| Water rate | barrels per day | bbl/day |

### Fluid Properties

| Quantity | Unit | Symbol |
| --- | --- | --- |
| Oil formation volume factor | barrels per stock-tank barrel | bbl/STB |
| Gas formation volume factor | cubic feet per standard cubic foot | ft³/SCF |
| Water formation volume factor | barrels per stock-tank barrel | bbl/STB |
| Oil viscosity | centipoise | cP |
| Gas viscosity | centipoise | cP |
| Water viscosity | centipoise | cP |
| Oil density | pounds-mass per cubic foot | lbm/ft³ |
| Gas density | pounds-mass per cubic foot | lbm/ft³ |
| Water density | pounds-mass per cubic foot | lbm/ft³ |
| Oil compressibility | inverse psi | psi-1 |
| Gas compressibility | inverse psi | psi-1 |
| Water compressibility | inverse psi | psi-1 |
| Oil specific gravity | dimensionless (relative to water) | - |
| Gas specific gravity | dimensionless (relative to air) | - |
| API gravity | degrees API | API |
| Gas-oil ratio | standard cubic feet per stock-tank barrel | SCF/STB |
| Gas solubility in water | standard cubic feet per stock-tank barrel | SCF/STB |
| Salinity | parts per million | ppm |

### Rock Properties

| Quantity | Unit | Symbol |
| --- | --- | --- |
| Permeability | millidarcies | mD |
| Porosity | dimensionless (fraction) | - |
| Rock compressibility | inverse psi | psi-1 |
| Relative permeability | dimensionless (fraction) | - |

### Time

| Quantity | Unit | Symbol |
| --- | --- | --- |
| Internal simulation time | seconds | s |
| Timer parameters | seconds | s |
| Display convention | days, years | - |

!!! info "Time Units"

    BORES stores all time values internally in seconds. The `Time()` helper function converts from human-readable units (days, months, years) to seconds:

    ```python
    import bores

    one_day = bores.Time(days=1)  # 86400.0 seconds
    one_year = bores.Time(years=1)  # 31557600.0 seconds
    combined = bores.Time(days=30, hours=6)  # 2613600.0 seconds
    ```

### Dimensionless Quantities

| Quantity | Range |
| --- | --- |
| Porosity | 0 to 1 (typically 0.05 to 0.35) |
| Saturation (oil, water, gas) | 0 to 1 |
| Relative permeability | 0 to 1 |
| Z-factor (gas compressibility factor) | 0 to ~2 (typically 0.3 to 1.0) |
| CFL number | 0 to ~1 (must be < 1 for stability) |
| Skin factor | typically -5 to +20 |
| Todd-Longstaff omega | 0 (fully segregated) to 1 (fully mixed) |

---

## Common Conversions from SI

If your data is in SI units, use these conversion factors:

| Quantity | SI Unit | Oil-Field Unit | Multiply SI by |
| --- | --- | --- | --- |
| Pressure | Pa (Pascal) | psi | 1.450377e-4 |
| Pressure | MPa | psi | 145.0377 |
| Pressure | bar | psi | 14.50377 |
| Pressure | atm | psi | 14.696 |
| Temperature | Celsius | Fahrenheit | $F = C \times 9/5 + 32$ |
| Temperature | Kelvin | Fahrenheit | $F = (K - 273.15) \times 9/5 + 32$ |
| Length | meters | feet | 3.28084 |
| Length | centimeters | feet | 0.0328084 |
| Permeability | m² | mD | 1.01325e+15 |
| Permeability | Darcy | mD | 1000 |
| Viscosity | Pa.s | cP | 1000 |
| Viscosity | mPa.s | cP | 1.0 |
| Density | kg/m³ | lbm/ft³ | 0.062428 |
| Volume | m³ | bbl | 6.28981 |
| Volume | liters | bbl | 0.00628981 |
| Flow rate | m³/s | bbl/day | 543439.65 |
| Flow rate | m³/day | bbl/day | 6.28981 |
| Compressibility | Pa⁻¹ | psi-1 | 6894.757 |

### Example Conversion

```python
import numpy as np

# Convert SI pressure data to oil-field units
pressure_mpa = np.array([20.0, 25.0, 30.0])  # MPa
pressure_psi = pressure_mpa * 145.0377  # psi

# Convert SI permeability to oil-field units
perm_m2 = 1e-13  # m^2
perm_md = perm_m2 * 1.01325e15  # mD (= 101.325 mD)

# Convert Celsius to Fahrenheit
temp_c = 93.3  # Celsius
temp_f = temp_c * 9.0 / 5.0 + 32.0  # Fahrenheit (= 200 F)
```

---

## Standard Conditions

BORES uses the following standard (surface) conditions, consistent with petroleum industry conventions:

| Quantity | Value |
| --- | --- |
| Standard pressure | 14.696 psi (1 atm) |
| Standard temperature | 60.0 F (15.56 C) |

These are the conditions at which surface volumes (STB, SCF) are defined. Formation volume factors convert between reservoir conditions and these standard conditions.

You can access these values programmatically through the constants system:

```python
import bores

print(bores.c.STANDARD_PRESSURE_SI)  # 14.696 psi
print(bores.c.STANDARD_TEMPERATURE_SI)  # 60.0 F
```

---

## Sign Conventions

BORES uses the following sign conventions throughout the library:

| Quantity | Positive Means | Negative Means |
| --- | --- | --- |
| Flow rate at wells | Injection into reservoir | Production from reservoir |
| Flux at boundaries | Inflow to domain | Outflow from domain |
| Depth | Increasing downward | - |
| Elevation | Increasing upward | - |
| Skin factor | Formation damage | Stimulation |
