# Glossary

## Petroleum Engineering Terms

**API gravity**
: A measure of how heavy or light a petroleum liquid is compared to water. Defined as $\text{API} = \frac{141.5}{\gamma_o} - 131.5$ where $\gamma_o$ is the oil specific gravity. Higher API values indicate lighter oils. Light crude is typically above 31 API, medium crude is 22 to 31, and heavy crude is below 22.

**Aquifer**
: A body of water-bearing rock connected to the reservoir that provides pressure support as fluids are produced. Aquifers can be modeled analytically (using Carter-Tracy or Van Everdingen-Hurst models) or numerically (by extending the simulation grid into the water zone).

**Aquifer influx**
: The volume of water flowing from an aquifer into the reservoir over time in response to pressure decline. Measured in reservoir barrels (bbl).

**Black-oil model**
: A simplified thermodynamic model that describes reservoir fluids as three phases (oil, water, gas) with pressure-dependent properties. Gas can dissolve in oil (solution gas) but oil and water are immiscible with each other. This is the most widely used model for conventional reservoir simulation.

**Bottom-hole pressure (BHP)**
: The pressure measured at the bottom of a well at the depth of the producing formation. BHP is the pressure used in flow rate calculations and well control specifications. Measured in psi.

**Bottom water drive**
: A recovery mechanism where an aquifer located below the oil zone pushes water upward into the oil column as pressure declines, displacing oil toward the wells.

**Bubble point pressure**
: The pressure at which the first gas bubble forms in an oil that is initially at a higher pressure (undersaturated). Below the bubble point, gas comes out of solution and forms a free gas phase. Also called saturation pressure.

**Capillary pressure**
: The pressure difference across the interface between two immiscible fluids in a porous medium, caused by surface tension and wettability. Capillary pressure depends on the saturations of the phases and controls the distribution of fluids in the transition zones between fluid contacts.

**Carter-Tracy model**
: An analytical aquifer model that computes water influx using a time-stepping approximation to the Van Everdingen-Hurst solution. It avoids the full convolution integral by using the influx rate at each timestep rather than the cumulative influx function. In BORES, the Carter-Tracy model supports both physical aquifer properties (permeability, porosity, compressibility) and a pre-computed aquifer constant.

**Compressibility**
: A measure of how much a fluid or rock changes volume in response to a change in pressure. Defined as $c = -\frac{1}{V}\frac{\partial V}{\partial P}$. Measured in psi$^{-1}$.

**Connate water**
: The water present in the pore spaces of a reservoir rock at the time of hydrocarbon accumulation. Connate water is held in place by capillary forces and is typically immobile during production. Also called irreducible water saturation ($S_{wc}$).

**Critical gas saturation**
: The minimum gas saturation at which gas becomes mobile in a porous medium. Below this saturation, gas is trapped in isolated bubbles and cannot flow. Typically 2% to 5% of pore volume.

**Decline curve analysis (DCA)**
: A technique for fitting production rate data to mathematical decline models (exponential, hyperbolic, harmonic) to characterize production trends and forecast future behavior. Used to estimate remaining reserves and economic ultimate recovery.

**Dip (structural)**
: The angle of inclination of a rock layer relative to the horizontal. Structural dip affects gravity-driven flow and the position of fluid contacts. Measured in degrees.

**Displacement efficiency**
: The fraction of oil that has been removed from the pore space in the zone contacted by the displacing fluid. High displacement efficiency means the displacing fluid effectively mobilizes oil in the cells it reaches. Distinct from volumetric sweep efficiency, which measures how much of the reservoir has been contacted.

**Drainage**
: A flow process in which the non-wetting phase displaces the wetting phase. In a water-wet reservoir, gas displacing oil or oil displacing water is drainage. Drainage relative permeability curves differ from imbibition curves due to hysteresis effects.

**Drive index**
: A dimensionless number (0 to 1) indicating the fractional contribution of a particular energy source to total production. The four main drive indices (solution gas drive, gas cap drive, water drive, and compaction drive) sum to 1.0 and are computed from the material balance equation.

**Edge water drive**
: A recovery mechanism where an aquifer located at the lateral edges of the reservoir supplies water that displaces oil toward the producing wells.

**Effective permeability**
: The permeability of a rock to a specific fluid phase when multiple phases are present. It is the product of absolute permeability and relative permeability: $k_{\text{eff}} = k \cdot k_r$.

**Enhanced oil recovery (EOR)**
: Recovery methods applied after primary depletion and secondary recovery (waterflooding) to extract additional oil. EOR techniques include miscible gas injection (CO2, nitrogen), chemical flooding (polymer, surfactant), and thermal methods (steam injection). EOR typically targets the residual oil that is immobile under conventional flooding.

**Estimated ultimate recovery (EUR)**
: The total cumulative production expected over the economic life of a well or reservoir. Computed by integrating a fitted decline curve from current production to an economic limit rate. Measured in STB for oil or SCF for gas.

**Formation volume factor (FVF)**
: The ratio of the volume of a fluid at reservoir conditions (pressure, temperature) to its volume at standard (surface) conditions. Oil FVF ($B_o$) is measured in bbl/STB. Gas FVF ($B_g$) is measured in ft³/SCF. Water FVF ($B_w$) is measured in bbl/STB.

**Gas cap**
: A zone of free gas overlying the oil zone in a reservoir. Gas caps form when the reservoir pressure at the top of the structure is below the bubble point, allowing gas to accumulate above the gas-oil contact. Gas cap expansion is a primary drive mechanism during depletion.

**Gas-oil contact (GOC)**
: The depth at which the gas cap meets the oil zone. Above the GOC, the pore space is primarily filled with gas. Below it, the pore space is primarily filled with oil (and connate water).

**Gas-oil ratio (GOR)**
: The volume of gas produced per unit volume of oil produced, both measured at standard conditions. Measured in SCF/STB. Solution GOR ($R_s$) refers specifically to the gas dissolved in oil at reservoir conditions.

**Hydrocarbon pore volume (HCPV)**
: The total pore volume occupied by hydrocarbons (oil and free gas), excluding the volume occupied by connate water. Computed as $\text{HCPV} = V_p \cdot (1 - S_w)$ where $V_p$ is pore volume and $S_w$ is water saturation. Measured in reservoir barrels or cubic feet.

**Imbibition**
: A flow process in which the wetting phase displaces the non-wetting phase. In a water-wet reservoir, water displacing oil is imbibition. Waterflooding is an imbibition process. Imbibition relative permeability curves differ from drainage curves because of capillary trapping and contact angle hysteresis.

**Immiscible displacement**
: A displacement process in which the injected fluid and the displaced fluid remain as separate phases (they do not mix at the molecular level). Waterflooding is the most common example of immiscible displacement.

**IMPES**
: Implicit Pressure, Explicit Saturation. A numerical solution method that solves the pressure equation implicitly (unconditionally stable) and updates saturations explicitly (subject to CFL stability condition). This is the most common method for black-oil simulation because it balances stability and computational cost.

**Inflow performance relationship (IPR)**
: A curve or equation that relates the flow rate of a well to the bottom-hole flowing pressure for a given reservoir pressure. IPR models include Vogel (solution gas drive), Fetkovich (gas wells), linear (single-phase undersaturated oil), and Jones (multi-phase with non-Darcy effects). Used for well design and artificial lift sizing.

**Irreducible water saturation**
: See Connate water.

**Material balance equation**
: A volumetric accounting equation that relates cumulative production, injection, and fluid expansion to the original fluids in place. The general form states that underground withdrawal equals the sum of oil expansion, gas cap expansion, water influx, and pore compaction. Used to estimate original oil in place and identify drive mechanisms.

**Material balance error (MBE)**
: A measure of how well the simulator conserves mass over a time interval. Computed as the difference between the change in fluids in place and the net production/injection, normalized by the initial volume. An MBE below 0.1% is considered excellent; above 5% is unacceptable.

**Minimum miscibility pressure (MMP)**
: The lowest pressure at which a particular injection gas achieves miscible displacement with a particular oil. Below the MMP, the displacement is immiscible. Above the MMP, it is miscible. Measured in psi.

**Miscible displacement**
: A displacement process in which the injected fluid and the displaced fluid mix completely at the molecular level, eliminating the interface between them. Miscible displacement eliminates capillary trapping and can achieve very high recovery factors. Requires the injected fluid to be miscible with the oil, which depends on pressure, temperature, and composition.

**Mobility**
: The ratio of a phase's effective permeability to its viscosity: $\lambda = k_r / \mu$. Mobility determines how easily a phase flows through the porous medium. The mobility ratio between displacing and displaced fluids controls sweep efficiency.

**Mobility ratio**
: The ratio of the displacing phase mobility to the displaced phase mobility: $M = \lambda_{\text{displacing}} / \lambda_{\text{displaced}}$. A mobility ratio less than 1.0 indicates a stable, piston-like displacement. A mobility ratio greater than 1.0 indicates an unstable displacement prone to viscous fingering, which reduces sweep efficiency.

**Oil-water contact (OWC)**
: The depth at which the oil zone meets the water zone. Below the OWC, the pore space is primarily filled with water.

**Permeability (absolute)**
: A measure of a rock's ability to transmit fluid, independent of the fluid properties. Measured in millidarcies (mD). Higher permeability means fluid flows more easily through the rock.

**Pore volume**
: The total volume of void space in the reservoir rock that can contain fluids. Computed as the product of bulk volume and porosity. Measured in reservoir barrels or cubic feet.

**Porosity**
: The fraction of the total rock volume that is pore space (voids). Effective porosity includes only the interconnected pores that can transmit fluid. Dimensionless, typically expressed as a decimal (0.20 = 20%).

**Productivity index (PI)**
: The ratio of flow rate to pressure drawdown for a well: $J = q / (P_r - P_{wf})$. Measured in STB/day/psi for oil or SCF/day/psi for gas. PI depends on permeability, fluid properties, completion geometry, and skin factor.

**Pseudo-pressure**
: A transformed pressure variable used for gas flow that accounts for the strong pressure dependence of gas properties (viscosity and compressibility). Defined as $m(P) = 2 \int \frac{P}{\mu Z} dP$. Using pseudo-pressure linearizes the gas diffusivity equation, improving solver convergence and accuracy.

**PVT (Pressure-Volume-Temperature)**
: The study of how reservoir fluid properties (density, viscosity, compressibility, formation volume factor, solution GOR) change with pressure and temperature. PVT data can come from laboratory measurements, empirical correlations, or equation-of-state calculations. In BORES, PVT properties are computed from correlations or looked up from pre-built PVT tables.

**Recovery factor**
: The fraction of the original fluids in place that has been produced. Oil recovery factor is cumulative oil produced divided by STOIIP. Expressed as a fraction (0 to 1) or percentage. Typical primary recovery factors range from 5% to 30% depending on drive mechanism.

**Relative permeability**
: The ratio of the effective permeability of a phase at a given saturation to the absolute permeability of the rock. Dimensionless, ranges from 0 to 1. Relative permeability curves describe how easily each phase flows as a function of saturation.

**Residual oil saturation**
: The fraction of oil that remains trapped in the pore spaces after displacement by water ($S_{orw}$) or gas ($S_{org}$). This oil is immobile and cannot be recovered by the displacing fluid. Typically 15% to 35% of pore volume for water flooding.

**Skin factor**
: A dimensionless number that accounts for the additional pressure drop (positive skin) or reduced pressure drop (negative skin) near the wellbore due to formation damage, perforation, or stimulation. Positive skin indicates damage; negative skin indicates stimulation (e.g., hydraulic fracture).

**Solution gas**
: Natural gas that is dissolved in oil at reservoir pressure and temperature. As pressure drops below the bubble point, solution gas comes out of the oil and forms a free gas phase. The amount of gas dissolved is described by the gas-oil ratio ($R_s$).

**Solution gas drive**
: A primary recovery mechanism in which the expansion of gas coming out of solution in the oil provides the energy to drive oil toward the wells. This mechanism becomes active when reservoir pressure drops below the bubble point. It is the dominant drive in many reservoirs without strong aquifer support or gas cap.

**Specific gravity**
: The ratio of a substance's density to the density of a reference substance. For oil, the reference is water at standard conditions ($\gamma_o = \rho_o / \rho_w$). For gas, the reference is air ($\gamma_g = M_g / M_{\text{air}}$).

**STGIIP (Stock Tank Gas Initially In Place)**
: The total volume of gas in the reservoir at initial conditions, converted to standard (surface) conditions. Includes both free gas in the gas cap and gas dissolved in oil. Measured in SCF. Used as the denominator for gas recovery factor calculations. Also abbreviated as GIIP.

**STOIIP (Stock Tank Oil Initially In Place)**
: The total volume of oil in the reservoir at initial conditions, converted to standard (surface) conditions using the initial formation volume factor. Measured in STB. Used as the denominator for oil recovery factor calculations. Also abbreviated as OOIP (Original Oil In Place).

**Sweep efficiency**
: The fraction of the reservoir that has been contacted by the displacing fluid. Volumetric sweep efficiency is the product of areal sweep (planform area contacted) and vertical sweep (vertical interval contacted). Overall recovery efficiency is the product of sweep efficiency and displacement efficiency.

**Todd-Longstaff model**
: A mixing model for miscible displacement that uses a single parameter ($\omega$, the mixing parameter) to interpolate between fully segregated flow (each fluid occupies its own streamlines) and fully mixed flow (perfect molecular mixing). At $\omega = 0$, the phases are fully segregated. At $\omega = 1$, they are fully mixed. The model modifies effective viscosity and density using the mixing rule: $\mu_{\text{eff}} = \mu_{\text{mix}}^\omega \cdot \mu_{\text{seg}}^{1-\omega}$.

**Transmissibility**
: A quantity that describes the flow capacity between two adjacent grid cells. It incorporates the permeability, cross-sectional area, and distance between cell centers: $T = \frac{k \cdot A}{\Delta x}$. Transmissibility modification is how faults and fractures affect flow.

**Undersaturated oil**
: Oil at a pressure above its bubble point, meaning all the gas that can dissolve in the oil at that temperature is dissolved. No free gas phase exists. As pressure drops below the bubble point, the oil becomes saturated and free gas appears.

**Voidage replacement ratio (VRR)**
: The ratio of total injected fluid volumes to total produced fluid volumes, both measured at reservoir conditions using formation volume factors. A VRR of 1.0 means injection exactly replaces production and reservoir pressure is maintained. VRR above 1.0 means pressure is increasing; below 1.0 means pressure is declining.

**Water cut**
: The fraction of the total produced liquid that is water, measured at surface conditions. Water cut increases during a waterflood as the injected water breaks through to the producing wells. Expressed as a fraction (0 to 1) or percentage.

**Water drive**
: A natural recovery mechanism in which pressure support comes from water encroachment (from an aquifer or injected water), displacing oil toward the producing wells.

**Well index**
: A geometric factor that relates the flow rate between a well and its grid cell to the pressure difference between the cell and the wellbore. It depends on cell dimensions, permeability, wellbore radius, and skin factor. Computed using the Peaceman formula for vertical wells. Measured in rb/day/psi.

**Wettability**
: The preference of the rock surface for one fluid over another. In a water-wet rock, water preferentially contacts the rock surface and occupies the smaller pores. In an oil-wet rock, oil preferentially contacts the surface. Wettability strongly affects relative permeability and capillary pressure curves.

---

## Relative Permeability and Capillary Pressure Models

**Brooks-Corey model**
: A relative permeability model that uses a power-law relationship with a pore size distribution index ($\lambda$) to compute water, oil, and gas relative permeabilities from normalized saturations. Widely used for its physical basis in relating pore geometry to flow behavior. Related to the Corey model but with an explicit connection to capillary pressure data.

**Corey model**
: A simple power-law relative permeability model: $k_r = k_{r,\max} \cdot S_n^{n}$ where $S_n$ is the normalized saturation and $n$ is the Corey exponent. Separate exponents for each phase control the curvature of the relative permeability curves. Exponents typically range from 1 (linear) to 6 (strongly non-linear).

**LET model**
: A three-parameter relative permeability correlation (L, E, T) that provides more flexibility than Corey curves for matching laboratory data. The L parameter controls the low-saturation shape, E controls the intermediate region, and T controls the high-saturation shape. Useful when Corey curves cannot match measured data across the full saturation range.

**Leverett J-function**
: A dimensionless capillary pressure function that normalizes laboratory capillary pressure data by porosity and permeability: $J(S_w) = P_c \sqrt{k / \phi} / (\sigma \cos\theta)$. Allows capillary pressure curves measured on core samples to be scaled to different rock types with different porosity and permeability values.

**Stone II model**
: A method for estimating three-phase oil relative permeability from two sets of two-phase data (oil-water and oil-gas). The model assumes that the presence of a third phase reduces the oil relative permeability below what either two-phase curve would predict alone. The formula combines the two-phase curves as $k_{ro} = k_{ro,w} + k_{ro,g} - 1$ in the normalized form.

**Van Genuchten model**
: A capillary pressure model commonly used in soil science and hydrology that relates capillary pressure to water saturation using shape parameters ($\alpha$, $n$, $m$). Produces smooth S-shaped curves and is mathematically well-behaved (no singularities at the endpoints).

---

## PVT Correlations

**Dranchuk-Abou-Kassem correlation**
: A correlation for computing the gas compressibility factor (Z-factor) as a function of pseudo-reduced pressure and temperature. Based on the Benedict-Webb-Rubin equation of state fitted to experimental data. Used in BORES for gas PVT calculations when no laboratory data is available.

**Standing correlation**
: A set of empirical correlations by M.B. Standing for estimating oil bubble point pressure, solution GOR, and oil formation volume factor from API gravity, gas specific gravity, and temperature. One of the most widely used PVT correlation sets for black-oil simulation.

**Vasquez-Beggs correlation**
: Empirical correlations for oil formation volume factor, solution GOR, and oil viscosity as functions of pressure, temperature, API gravity, and gas specific gravity. Provides separate coefficients for light oils (API > 30) and heavy oils (API < 30).

---

## Numerical and Simulation Terms

**AMG (Algebraic Multigrid)**
: A preconditioner that builds a hierarchy of progressively coarser representations of the linear system to accelerate convergence. AMG automatically constructs the coarsening from the matrix structure (no geometric grid information needed). Effective for elliptic pressure equations but expensive to set up. In BORES, AMG setup cost can be reduced by using preconditioner caching.

**BiCGSTAB**
: Bi-Conjugate Gradient Stabilized. A Krylov iterative solver for non-symmetric linear systems. It requires two matrix-vector products per iteration and converges smoothly without the erratic residual behavior of plain BiCG. The default solver in BORES for both pressure and saturation equations.

**Boundary condition**
: A mathematical specification of what happens at the edges of the simulation domain. Common types include no-flow (Neumann with zero flux), constant pressure (Dirichlet), and specified flux (Neumann with non-zero flux). Boundary conditions determine whether fluid can enter or leave the domain at its edges.

**CFL condition**
: The Courant-Friedrichs-Lewy condition, a stability requirement for explicit numerical schemes. The CFL number must be less than 1 for stability: $\text{CFL} = v \cdot \Delta t / \Delta x < 1$. This limits the maximum timestep for a given grid resolution and flow velocity.

**Convergence tolerance**
: The threshold that determines when an iterative solver has found a solution that is "good enough." Tighter tolerances produce more accurate solutions but require more iterations. In BORES, pressure tolerance defaults to $10^{-6}$ and saturation tolerance to $10^{-4}$.

**CPR (Constrained Pressure Residual)**
: A two-stage preconditioner designed for coupled multiphase flow. The first stage extracts and solves the pressure subsystem (using AMG or ILU), then the second stage applies ILU to the full system. CPR is the most effective preconditioner for multiphase reservoir simulation because it targets the elliptic pressure component that dominates solver difficulty.

**Deserialization**
: The process of reconstructing an object from its stored representation (dictionary, JSON, HDF5, etc.). The reverse of serialization. In BORES, deserialization uses the `load()` function or `from_dict()` method.

**Dirichlet boundary condition**
: A boundary condition that specifies a fixed value (e.g., constant pressure) at a boundary face. In reservoir simulation, a constant pressure boundary represents an infinite aquifer or a maintained pressure source.

**Explicit scheme**
: A numerical method that computes the solution at the next timestep directly from the current timestep values, without solving a system of equations. Fast per step but subject to CFL stability limits.

**GMRES**
: Generalized Minimal Residual. A Krylov iterative solver for non-symmetric linear systems that minimizes the residual norm over the Krylov subspace at each iteration. GMRES is mathematically optimal but requires storing all previous search directions, so memory usage grows with iteration count. Restarted GMRES (GMRES(m)) limits memory by restarting after $m$ iterations. LGMRES is a variant that augments the Krylov subspace with approximations from previous restarts to improve convergence.

**Grid cell**
: The fundamental spatial unit of the simulation domain. Each cell has a position, size, and a set of property values (pressure, saturations, porosity, permeability). Cells exchange fluid with their neighbors according to Darcy's law.

**ILU (Incomplete LU)**
: A preconditioner formed by computing an approximate LU factorization of the coefficient matrix, where fill-in (new non-zeros created during factorization) is discarded or limited. ILU is the most commonly used general-purpose preconditioner because it offers a good balance between setup cost and effectiveness. In BORES, ILU is the default preconditioner for both pressure and saturation solvers.

**Implicit scheme**
: A numerical method that computes the solution at the next timestep by solving a system of equations that couples the new and current timestep values. Unconditionally stable (no CFL limit) but more expensive per step and requires iterative solvers.

**Krylov method**
: A class of iterative methods for solving large sparse linear systems. Named after Alexei Krylov. Examples include Conjugate Gradient (CG), BiCGSTAB, and GMRES. These methods build a solution in a subspace that grows with each iteration.

**Neumann boundary condition**
: A boundary condition that specifies the flux (flow rate per unit area) at a boundary face. A zero-flux Neumann condition represents a sealed boundary (no flow). A non-zero flux specifies injection or production at the boundary.

**Preconditioner**
: A transformation applied to a linear system to make it easier for an iterative solver to converge. A good preconditioner approximates the inverse of the coefficient matrix. Common preconditioners include ILU (Incomplete LU), AMG (Algebraic Multigrid), and Jacobi (diagonal).

**Preconditioner caching**
: The practice of reusing a previously built preconditioner for several consecutive timesteps instead of rebuilding it every step. Effective because the coefficient matrix structure stays constant and values change slowly between steps. In BORES, the `CachedPreconditionerFactory` wraps any preconditioner and rebuilds it only when the matrix changes significantly or a specified number of steps have elapsed.

**Registration**
: In BORES, the process of adding a custom object (solver, preconditioner, well control, boundary function) to the framework's internal registry so it can be referenced by name and serialized/deserialized correctly. Done using decorator functions like `@solver_func`, `@preconditioner_factory`, `@well_control`.

**Renderer**
: In the visualization system, a class responsible for producing a specific type of plot (line, heatmap, volume, etc.). Renderers are used by the `DataVisualizer` class to generate Plotly figures.

**Robin boundary condition**
: A boundary condition that specifies a linear combination of the value and its flux at a boundary face: $\alpha u + \beta \frac{\partial u}{\partial n} = g$. In reservoir simulation, BHP-controlled wells use Robin-type conditions where the well rate depends on the difference between cell pressure and bottom-hole pressure through the productivity index.

**Serialization**
: The process of converting an object into a storable representation (dictionary, JSON, HDF5, etc.) that can be saved to disk and later reconstructed. In BORES, serialization uses the `dump()` function or `to_dict()` method.

**Timestep**
: The discrete time interval over which the simulator advances the solution from one state to the next. Smaller timesteps are more accurate but more expensive. Adaptive timestep control adjusts the interval automatically based on solution behavior.

**Transmissibility**
: See the petroleum term above. In numerical context, transmissibility is precomputed between each pair of adjacent cells and stored in the coefficient matrix. Modifying transmissibility (via faults or fractures) changes the flow connectivity of the grid.

---

## BORES Framework Terms

**Config**
: The immutable configuration object (`bores.Config`) that holds all simulation parameters: timer, solver settings, convergence tolerances, well definitions, boundary conditions, rock-fluid tables, and physical controls. Because `Config` is frozen, modifications require creating a new instance using `copy()` or `update()`.

**DataStore**
: The abstract base class for all storage backends in BORES. Concrete implementations include `ZarrStore`, `HDF5Store`, `JSONStore`, and `YAMLStore`. Stores support `dump()`, `load()`, `append()`, and `entries()` operations and can be used as context managers for efficient persistent-handle access.

**ModelAnalyst**
: The post-simulation analysis class (`bores.analyses.ModelAnalyst`) that accepts a collection of `ModelState` objects and provides methods for computing recovery factors, drive indices, sweep efficiency, decline curves, material balance, and other reservoir engineering metrics.

**ModelState**
: A snapshot of the entire simulation state at a single timestep, containing the reservoir model (with current pressures, saturations, and fluid properties), production/injection volumes, and timing information. `ModelState` objects are yielded by the simulation generator and can be persisted to a `DataStore`.

**PVT tables**
: Pre-computed lookup tables (`bores.PVTTables`) that map pressure to fluid properties (formation volume factor, viscosity, density, compressibility, solution GOR). Using PVT tables instead of correlations replaces floating-point arithmetic with interpolation, which is typically 2 to 5x faster for large grids.

**BlackOilModel**
: The central data object (`bores.BlackOilModel`) that contains the grid geometry, rock properties (porosity, permeability, compressibility), fluid properties (pressure, saturations, PVT data), and initial conditions. The model is immutable; each timestep produces a new model with updated state.

**Rock-fluid tables**
: Tabular data (`bores.RockFluidTables`) defining relative permeability and capillary pressure as functions of saturation for each rock type in the model. These tables control how easily each phase flows and how saturations distribute within the reservoir.

**Run**
: A serializable bundle (`bores.Run`) that pairs a `BlackOilModel` with a `Config` and optional metadata (name, description, tags). A `Run` is callable and iterable: iterating over it executes the simulation and yields `ModelState` objects. Saving a `Run` to a file captures the complete simulation definition for later reproduction.

**StateStream**
: A memory-efficient wrapper (`bores.StateStream`) around the simulation generator that persists states to a `DataStore` as they are produced. The stream acts as both a context manager and an iterator, keeping peak memory proportional to the batch size rather than the total number of states. Supports background I/O, checkpointing, filtering, and replay.

**Task pool**
: A `ThreadPoolExecutor` passed to `Config` via the `task_pool` parameter that enables concurrent matrix assembly during simulation. The pressure solver submits up to 3 independent assembly stages and the saturation solver submits up to 2, so a pool with 3 workers covers both without waste. Created using the `bores.new_task_pool()` context manager.
