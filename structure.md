# My target structure

```
Reservoir
├── Grid
├── Rock
├── Faults
├── Regions
└── Geometry

BlackOil
├── PVT
└── Saturation

Simulation
├── reservoir
├── fluid
├── state
├── wells
├── schedule
├── boundary_conditions
└── summary
```

SolverWorkspace -for reducing allocations for every iteration

    residual
    delta
    accumulation
    flux
    mobility
    transmissibility
    jacobian_values
    rowptr
    colind

Finite volume and TPFA or MPFA for solver
