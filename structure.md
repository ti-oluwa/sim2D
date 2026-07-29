# My target structure

```
Reservoir
├── Grid
├── Rock
├── Faults
├── ReservoirRegions
└── Geometry

BlackOil
├── PVT
└── SaturationFunction

Simulation
├── model
├── schedule
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

Finite volume and TPFA

So my naming convention would be:

Physical/property fields: singular (porosity, pressure, temperature)
Region-number fields: singular (pvt_region, rock_region, equilibrium_region)
Collections of tables: plural (PVTTables, SatFuncTables, RockCompressibilityRegions)
