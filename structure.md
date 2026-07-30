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
└── SatFunc

Simulation[ModelT, StateT]
├── ModelT
├── Spec
├── Timer
├── Schedule[ModelT, StateT]
└── Summary[ModelT, StateT]
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
Collections of tables: plural (PVTTables, SatFuncTables, RockCompressibilityTables)
