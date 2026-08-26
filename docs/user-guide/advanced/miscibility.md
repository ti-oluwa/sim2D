# Miscible Flooding

## Overview

Miscible flooding is an enhanced oil recovery (EOR) technique where the injected gas dissolves into and mixes with the reservoir oil at the molecular level, eliminating the interfacial tension between the gas and oil phases. When interfacial tension vanishes, residual oil trapping ceases, and the displacement efficiency approaches 100% in the swept zone. This makes miscible flooding dramatically more effective than immiscible gas injection, where gas bypasses trapped oil due to capillary forces.

Miscibility occurs when the reservoir pressure exceeds the minimum miscibility pressure (MMP), a property that depends on the compositions of both the injection gas and the reservoir oil. Below the MMP, the gas and oil remain as distinct phases with a clear interface. Above the MMP, they mix freely. In real reservoirs, there is a transition zone near the MMP where the fluids are partially miscible.

BORES implements miscible flooding through the Todd-Longstaff mixing model, which is the industry standard for black-oil simulators. The Todd-Longstaff model introduces a mixing parameter $\omega$ (omega) that controls the degree of mixing between the solvent (injected gas) and the oil. When $\omega = 0$, the phases are completely segregated (immiscible). When $\omega = 1$, they are fully mixed. Intermediate values represent partial mixing.

The model computes effective viscosities and densities for the mixed fluid using the mixing parameter:

$$\mu_{\text{eff}} = \mu_{\text{mix}}^{\omega} \cdot \mu_{\text{seg}}^{1-\omega}$$

$$\rho_{\text{eff}} = \rho_{\text{mix}}^{\omega} \cdot \rho_{\text{seg}}^{1-\omega}$$

where $\mu_{\text{mix}}$ and $\rho_{\text{mix}}$ are the fully mixed properties and $\mu_{\text{seg}}$ and $\rho_{\text{seg}}$ are the segregated (immiscible) properties.

---

## Enabling Miscible Flooding

To enable miscible flooding, you need two things: an `InjectedFluid` configured for miscibility and a `Config` that uses the Todd-Longstaff miscibility model.

### Configuring the Injected Fluid

```python
import bores

co2_fluid = bores.InjectedFluid(
    name="CO2",
    phase=bores.FluidPhase.GAS,
    specific_gravity=1.52,
    molecular_weight=44.01,
    is_miscible=True,
    minimum_miscibility_pressure=1200.0,  # psi
)
```

The key parameters are:

- `is_miscible=True`: Enables miscibility calculations for this fluid
- `minimum_miscibility_pressure`: The pressure above which the gas achieves first-contact miscibility with the oil. Below this pressure, miscibility effects are reduced through a smooth pressure-dependent transition.

### Configuring the Simulation

```python
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(hours=6),
        maximum_step_size=bores.Time(days=3),
        minimum_step_size=bores.Time(minutes=30),
        simulation_time=bores.Time(years=3),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    miscibility_model="todd_longstaff",
    freeze_saturation_pressure=False,
)
```

Setting `miscibility_model="todd_longstaff"` activates the Todd-Longstaff mixing calculations. Setting `freeze_saturation_pressure=False` allows the bubble point pressure to update as the solvent dissolves into the oil, which is important for capturing the compositional effects of miscible flooding.

---

## The Todd-Longstaff Model

The Todd-Longstaff model tracks the solvent concentration $C$ in each grid cell as a volume fraction (0 to 1). The concentration evolves through advection: solvent is transported with the oil-phase velocity, upwinding the concentration at cell faces based on the direction of oil flow.

### Mixing Parameter ($\omega$)

The mixing parameter $\omega$ is set globally through the `InjectedFluid` configuration. The default value (when not explicitly set) depends on the pressure relative to the MMP:

- **Above MMP**: $\omega$ approaches 1.0 (fully miscible)
- **Below MMP**: $\omega$ approaches 0.0 (immiscible)
- **Near MMP**: Smooth transition using a hyperbolic tangent function

This pressure-dependent miscibility means the simulation automatically handles situations where reservoir pressure drops below the MMP in some regions (for example, near a low-pressure producer), reducing miscibility in those regions while maintaining full miscibility where pressure is sufficient.

### Effective Viscosity

The effective oil-phase viscosity in each cell depends on the local solvent concentration and the mixing parameter:

$$\mu_{\text{mix}} = \mu_o^{1-C} \cdot \mu_s^{C}$$

$$\mu_{\text{eff}} = \mu_{\text{mix}}^{\omega} \cdot \mu_o^{1-\omega}$$

where $\mu_o$ is the oil viscosity, $\mu_s$ is the solvent viscosity, and $C$ is the solvent concentration. When $\omega = 1$ and $C = 1$ (fully mixed, pure solvent), the effective viscosity equals the solvent viscosity. When $\omega = 0$ (completely segregated), the effective viscosity equals the oil viscosity regardless of concentration.

### Effective Density

The density mixing follows the same pattern:

$$\rho_{\text{mix}} = \rho_o^{1-C} \cdot \rho_s^{C}$$

$$\rho_{\text{eff}} = \rho_{\text{mix}}^{\omega} \cdot \rho_o^{1-\omega}$$

---

## Custom Fluid Properties

For gases like CO2 that deviate significantly from standard correlations, you can provide override values for density and viscosity:

```python
co2_fluid = bores.InjectedFluid(
    name="CO2",
    phase=bores.FluidPhase.GAS,
    specific_gravity=1.52,
    molecular_weight=44.01,
    is_miscible=True,
    minimum_miscibility_pressure=1200.0,
    density=35.0,  # lbm/ft³ from lab data or EOS
    viscosity=0.05,  # cP from lab data or EOS
)
```

CO2 density at reservoir conditions is typically around 35 lbm/ft³, while the standard gas correlations predict only 3 to 7 lbm/ft³. Similarly, CO2 viscosity is around 0.05 cP versus the correlation prediction of 0.01 to 0.02 cP. Using the override values improves accuracy by approximately 25%.

---

## Solvent Concentration Transport

The solvent concentration is transported using pure advection (no diffusion). At each time step, the concentration in each cell is updated based on the oil-phase velocity at cell faces, using upwinding to ensure numerical stability:

1. The oil velocity at each cell face determines the flow direction
2. The upstream cell's concentration is used for the solvent flux
3. The concentration is updated based on the net flux into each cell
4. A dissolution limit prevents concentration from exceeding 1.0 in low-saturation cells

The dissolution limit is physically important. In cells with very low oil saturation, a small volume of injected solvent could push the concentration above 1.0 (unphysical). BORES computes the maximum dissolvable volume before each update and caps the concentration accordingly.

---

## Time Step Considerations

Miscible flooding requires smaller time steps than immiscible injection for two reasons. First, the concentration transport equation is explicit and subject to CFL constraints. Second, the viscosity and density changes near the miscible front are large and rapid, requiring small steps to accurately capture the front shape.

Recommended timer settings for miscible flooding:

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=6),
    maximum_step_size=bores.Time(days=3),
    minimum_step_size=bores.Time(minutes=30),
    simulation_time=bores.Time(years=3),
)
```

If you encounter convergence issues, try reducing `maximum_step_size` to 1 to 2 days or tightening the gas saturation change limit:

```python
config = bores.Config(
    ...,
    maximum_gas_saturation_change=0.5,
    maximum_pressure_change=75.0,
)
```

---

## Comparison: Immiscible vs. Miscible Gas Injection

| Aspect | Immiscible | Miscible |
|---|---|---|
| Gas-oil interfacial tension | Non-zero | Zero (above MMP) |
| Residual oil in swept zone | sorg (typically 15-25%) | Approaches 0% |
| Microscopic displacement efficiency | Moderate | Near 100% |
| Gravity override | Severe | Still present |
| Viscosity reduction | None | Significant |
| Required pressure | Any | Above MMP |
| Config setting | `miscibility_model="immiscible"` | `miscibility_model="todd_longstaff"` |

!!! info "Sweep Efficiency vs. Displacement Efficiency"

    Miscibility eliminates residual oil in the swept zone (high displacement efficiency), but it does not solve the gravity override problem (sweep efficiency). Gas is still lighter and more mobile than oil, so it preferentially flows through the upper part of the reservoir. The total recovery is the product of sweep efficiency and displacement efficiency. Miscible flooding with poor sweep can still underperform a well-designed waterflood.
