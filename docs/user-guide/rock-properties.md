# Rock Properties

## Overview

Rock properties define the static framework of your reservoir model. They control how much fluid the rock can store (porosity), how easily fluid flows through it (permeability), and how the pore volume changes with pressure (compressibility). In BORES, you specify these as NumPy arrays with one value per grid cell, then pass them to `bores.reservoir_model()`.

Understanding rock properties is essential because they directly control sweep efficiency, pressure communication, and ultimately recovery. A high-permeability streak can channel injected water past oil-bearing zones. A low-porosity layer can act as a barrier to vertical flow. These effects cannot be captured without realistic rock property distributions.

---

## Porosity

Porosity ($\phi$) is the fraction of rock volume that is pore space, available to store fluids. Values range from 0 (solid rock) to roughly 0.35 for unconsolidated sands.

| Rock Type | Typical Porosity |
|---|---|
| Tight sandstone | 0.05 - 0.10 |
| Consolidated sandstone | 0.15 - 0.25 |
| Unconsolidated sand | 0.25 - 0.35 |
| Carbonate (vuggy) | 0.10 - 0.30 |
| Shale (barrier) | 0.01 - 0.05 |

```python
import bores

grid_shape = (20, 20, 5)

# Uniform porosity
porosity = bores.build_uniform_grid(grid_shape, value=0.22)

# Layered porosity (decreasing with depth)
porosity = bores.build_layered_grid(
    grid_shape,
    layer_values=[0.25, 0.22, 0.20, 0.18, 0.15],
    orientation="z",
)
```

Cells with zero or NaN porosity are treated as inactive by the simulator. You can use this to model impermeable boundaries or irregular reservoir shapes within the rectangular grid.

---

## Permeability

Absolute permeability ($k$) measures how easily a single-phase fluid flows through the rock under a pressure gradient. BORES uses millidarcies (mD) as the unit. Permeability can vary by direction (anisotropy), which is extremely common in real reservoirs because of layered deposition.

### Isotropic Permeability

If permeability is the same in all directions, you can pass a single grid:

```python
import bores

grid_shape = (20, 20, 5)
perm_grid = bores.build_uniform_grid(grid_shape, value=100.0)  # 100 mD

# Option 1: Just pass x, BORES copies to y and z
permeability = bores.Permeability(x=perm_grid)

# Option 2: Explicitly set all directions
permeability = bores.Permeability(x=perm_grid, y=perm_grid, z=perm_grid)
```

### Anisotropic Permeability

In most sedimentary rocks, vertical permeability ($k_z$) is lower than horizontal permeability ($k_x$, $k_y$) due to compaction and bedding planes. A typical $k_v/k_h$ ratio is 0.1 to 0.3.

```python
import bores

grid_shape = (20, 20, 5)

kx = bores.build_uniform_grid(grid_shape, value=200.0)   # 200 mD horizontal
ky = bores.build_uniform_grid(grid_shape, value=200.0)   # Same in y
kz = bores.build_uniform_grid(grid_shape, value=20.0)    # 20 mD vertical (kv/kh = 0.1)

permeability = bores.Permeability(x=kx, y=ky, z=kz)
```

Anisotropy has a major impact on vertical sweep efficiency. Low vertical permeability limits gravity override in gas injection and reduces water coning near production wells.

### Heterogeneous Permeability

For realistic models, permeability varies spatially. You can build layered distributions or use random perturbations:

```python
import bores
import numpy as np

grid_shape = (20, 20, 5)

# Layer-by-layer permeability
kx = bores.build_layered_grid(
    grid_shape,
    layer_values=[300.0, 150.0, 50.0, 200.0, 100.0],
    orientation="z",
)

# Add log-normal heterogeneity within each layer
rng = np.random.default_rng(seed=42)
noise = np.exp(rng.normal(0, 0.3, grid_shape))  # Log-normal multiplier
kx = kx * noise

# Apply kv/kh ratio
kz = kx * 0.1

permeability = bores.Permeability(x=kx, y=kx, z=kz)
```

!!! info "Permeability Range"

    Realistic permeability values span many orders of magnitude: from 0.01 mD (tight rock) to 10,000 mD (highly permeable sand). High contrasts (ratios exceeding 1000:1) can cause solver convergence difficulties. Consider using 64-bit precision and stronger preconditioners (ILU or CPR) for such cases.

---

## Rock Compressibility

Rock compressibility ($c_r$) describes how much the pore volume changes with pressure. It is specified in psi$^{-1}$ and is typically much smaller than fluid compressibility.

| Rock Type | Typical $c_r$ (psi$^{-1}$) |
|---|---|
| Consolidated sandstone | $3 \times 10^{-6}$ to $5 \times 10^{-6}$ |
| Unconsolidated sand | $10 \times 10^{-6}$ to $20 \times 10^{-6}$ |
| Limestone | $2 \times 10^{-6}$ to $4 \times 10^{-6}$ |

You pass rock compressibility as a scalar to `bores.reservoir_model()`:

```python
model = bores.reservoir_model(
    # ... other parameters ...
    rock_compressibility=3e-6,  # psi⁻¹
)
```

Although small, rock compressibility contributes to total system compressibility and affects pressure diffusion. It becomes more important in tight formations where fluid compressibility is also low, making rock compressibility a larger fraction of the total.

---

## Complete Example

Here is a full example building a heterogeneous reservoir model with layered porosity, anisotropic permeability, and rock compressibility:

```python
import bores
import numpy as np

bores.use_32bit_precision()

grid_shape = (20, 20, 5)
cell_dimensions = (200.0, 200.0)  # ft

# Layered porosity
porosity = bores.build_layered_grid(
    grid_shape,
    layer_values=[0.25, 0.22, 0.18, 0.20, 0.15],
    orientation="z",
)

# Heterogeneous permeability with kv/kh = 0.1
rng = np.random.default_rng(seed=42)
base_perm = bores.build_layered_grid(
    grid_shape,
    layer_values=[300.0, 200.0, 50.0, 250.0, 100.0],
    orientation="z",
)
noise = np.exp(rng.normal(0, 0.25, grid_shape)).astype(np.float32)
kx = base_perm * noise
permeability = bores.Permeability(x=kx, y=kx, z=kx * 0.1)

# Other properties
thickness = bores.build_uniform_grid(grid_shape, value=15.0)
pressure = bores.build_uniform_grid(grid_shape, value=3500.0)
temperature = bores.build_uniform_grid(grid_shape, value=200.0)
oil_sg = bores.build_uniform_grid(grid_shape, value=0.85)
oil_visc = bores.build_uniform_grid(grid_shape, value=1.5)
bubble_pt = bores.build_uniform_grid(grid_shape, value=2800.0)

# Saturations from fluid contacts
depth = bores.build_depth_grid(thickness) + 5000.0
swc  = bores.build_uniform_grid(grid_shape, value=0.25)
sorw = bores.build_uniform_grid(grid_shape, value=0.25)
sorg = bores.build_uniform_grid(grid_shape, value=0.15)
sgr  = bores.build_uniform_grid(grid_shape, value=0.05)

Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4950.0,
    oil_water_contact=5060.0,
    connate_water_saturation_grid=swc,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    residual_gas_saturation_grid=sgr,
    porosity_grid=porosity,
)

# Build the model
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimensions,
    thickness_grid=thickness,
    pressure_grid=pressure,
    rock_compressibility=3e-6,
    absolute_permeability=permeability,
    porosity_grid=porosity,
    temperature_grid=temperature,
    water_saturation_grid=Sw,
    gas_saturation_grid=Sg,
    oil_saturation_grid=So,
    oil_viscosity_grid=oil_visc,
    oil_specific_gravity_grid=oil_sg,
    oil_bubble_point_pressure_grid=bubble_pt,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    residual_gas_saturation_grid=sgr,
    irreducible_water_saturation_grid=swc,
    connate_water_saturation_grid=swc,
)
```

!!! tip "Start Simple, Add Complexity"

    Begin with uniform properties to verify your simulation runs correctly, then add heterogeneity layer by layer. This makes it much easier to isolate the effect of each property on the simulation results.
