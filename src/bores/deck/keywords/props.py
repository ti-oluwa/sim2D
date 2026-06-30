"""
PROPS section keyword implementations.

The PROPS section contains fluid and rock property tables used by the
simulator: PVT data (formation volume factors, viscosities, compressibilities)
and relative permeability / capillary pressure curves.

**PVT keywords** (return `List[List[Dict]]` - one inner list per PVT region):

- `DENSITY`  - stock-tank densities (oil, water, gas) [one record per region]
- `PVTW`     - water PVT table (ref pressure, Bw, Cw, viscosity, Cv)
- `PVDO`     - dead-oil PVT (pressure, Bo, viscosity) tabulated
- `PVTO`     - live-oil PVT (Rs-bracketed) tabulated
- `PVCO`     - compressible-oil PVT shorthand
- `PVDG`     - dry-gas PVT (pressure, Bg, viscosity) tabulated
- `PVTG`     - wet-gas PVT (Rv-bracketed) tabulated
- `ROCK`     - rock compressibility (ref pressure, compressibility)
- `ROCKTAB`  - tabulated pore-volume vs. pressure compaction table

**Relative permeability / capillary pressure keywords**:

- `SWOF`  - Sw-indexed: Krw, Krow, Pcow
- `SGOF`  - Sg-indexed: Krg, Krog, Pcog
- `SWFN`  - Sw-indexed: Krw, Pcow (two-function family)
- `SGFN`  - Sg-indexed: Krg, Pcog (two-function family)
- `SOF2`  - So-indexed: Kro (two-phase oil)
- `SOF3`  - So-indexed: Krow, Krog (three-phase oil)
"""

import numpy as np

from bores.deck.keywords.base import Field, PVTTableKeyword

__all__ = [
    "DENSITY",
    "PVTW",
    "PVDO",
    "PVTO",
    "PVCO",
    "PVDG",
    "PVTG",
    "ROCK",
    "ROCKTAB",
    "SWOF",
    "SGOF",
    "SWFN",
    "SGFN",
    "SOF2",
    "SOF3",
]

DENSITY = PVTTableKeyword(
    "DENSITY",
    columns=[
        Field("oil", np.float64),
        Field("water", np.float64),
        Field("gas", np.float64),
    ],
)
"""
`DENSITY  OIL_DENSITY  WATER_DENSITY  GAS_DENSITY /`
- stock-tank fluid densities at standard conditions.

One record per PVT region. Units depend on the deck's declared unit
system:

- FIELD: lb/ft³
- METRIC: kg/m³
- LAB: g/cc

Columns:

- `oil`   - oil density at standard conditions.
- `water` - water density at standard conditions.
- `gas`   - gas density at standard conditions.

`parse` returns `List[List[Dict]]` where each inner list
contains a single-row table (one density record per PVT region).
"""

PVTW = PVTTableKeyword(
    "PVTW",
    columns=[
        Field("p_ref", np.float64),
        Field("bw", np.float64),
        Field("cw", np.float64),
        Field("viscosity", np.float64),
        Field("cv", np.float64, required=False, default=0.0),
    ],
)
"""
`PVTW  P_REF  BW  CW  MU_W  CV /` - water PVT table (one record per PVT region).

Columns:

- `p_ref`        - reference pressure (psi / bar).
- `bw`           - water formation volume factor at `p_ref` (rb/stb / rm³/sm³).
- `cw`           - water compressibility (1/psi / 1/bar).
- `viscosity`    - water viscosity at `p_ref` (cP).
- `cv`           - water viscosibility (1/psi / 1/bar), optional (default 0).
"""


PVDO = PVTTableKeyword(
    "PVDO",
    columns=[
        Field("pressure", np.float64),
        Field("bo", np.float64),
        Field("viscosity", np.float64),
    ],
)
"""
`PVDO` - dead-oil (no dissolved gas) PVT table.

Each table (one per PVT region) is a sequence of rows terminated
by `/`, with columns:

- `pressure`  - oil pressure (psi / bar).
- `bo`        - oil formation volume factor (rb/stb / rm³/sm³).
- `viscosity` - oil viscosity (cP).

Rows must be in ascending pressure order.
"""

PVTO = PVTTableKeyword(
    "PVTO",
    columns=[
        Field("pressure", np.float64),
        Field("bo", np.float64),
        Field("viscosity", np.float64),
    ],
    primary_key="rs",
)
"""
`PVTO` - live-oil (dissolved-gas, Rs-bracketed) PVT table.

The Eclipse format uses a primary Rs value on a line by itself,
followed by rows of `(pressure, Bo, viscosity)` at that Rs.
Each Rs group is terminated by `/`; the table block ends with `/`.

Columns in each data row (the `rs` field is injected from the
bracketing primary-key line):

- `rs`        - solution gas-oil ratio (scf/stb / sm³/sm³).
- `pressure`  - oil pressure (psi / bar).
- `bo`        - oil FVF (rb/stb / rm³/sm³).
- `viscosity` - oil viscosity (cP).
"""


PVCO = PVTTableKeyword(
    "PVCO",
    columns=[
        Field("p_ref", np.float64),
        Field("bo", np.float64),
        Field("co", np.float64),
        Field("viscosity", np.float64),
        Field("cv", np.float64, required=False, default=0.0),
    ],
)
"""
`PVCO` - compressible-oil PVT shorthand table (single-row format).

Alternative to `PVTO` for oil above bubble point. One record per
PVT region.

Columns:

- `p_ref`       - reference (bubble-point) pressure.
- `bo`          - FVF at `p_ref`.
- `co`          - oil compressibility (1/psi / 1/bar).
- `viscosity`   - oil viscosity at `p_ref`.
- `cv`          - viscosibility (optional, default 0).
"""


PVDG = PVTTableKeyword(
    "PVDG",
    columns=[
        Field("pressure", np.float64),
        Field("bg", np.float64),
        Field("viscosity", np.float64),
    ],
)
"""
`PVDG` - dry-gas (no vaporised oil) PVT table.

Each table (one per PVT region) contains rows:

- `pressure`  - gas pressure (psi / bar).
- `bg`        - gas FVF (rb/Mscf / rm³/sm³).
- `viscosity` - gas viscosity (cP).

Rows must be in ascending pressure order.
"""


PVTG = PVTTableKeyword(
    "PVTG",
    columns=[
        Field("rv", np.float64),
        Field("bg", np.float64),
        Field("viscosity", np.float64),
    ],
    primary_key="pressure",
)
"""
`PVTG` - wet-gas (vaporised-oil, Rv-bracketed) PVT table.

Mirrors the structure of `PVTO` but with pressure as the outer
(primary) key and Rv as the inner column.

Columns in each data row (the `pressure` field is injected from the
primary-key line):

- `pressure`  - gas pressure (psi / bar).
- `rv`        - vaporised oil-gas ratio (stb/Mscf / sm³/sm³).
- `bg`        - gas FVF (rb/Mscf / rm³/sm³).
- `viscosity` - gas viscosity (cP).
"""


ROCK = PVTTableKeyword(
    "ROCK",
    columns=[Field("p_ref", np.float64), Field("cr", np.float64)],
)
"""
`ROCK  P_REF  CR /`
- rock compressibility data (one record per rock region).

Columns:

- `p_ref` - reference pressure at which the pore volume equals the
    geometrically calculated value (psi / bar).
- `cr` - rock compressibility (1/psi / 1/bar).
"""

ROCKTAB = PVTTableKeyword(
    "ROCKTAB",
    columns=[
        Field("pressure", np.float64),
        Field("pv_mult", np.float64),
        Field("trans_mult", np.float64, required=False, default=1.0),
    ],
)
"""
`ROCKTAB` - tabulated pore-volume multiplier vs. pressure.

Used when rock compaction cannot be described by a single
compressibility value. Each table (one per rock region) contains rows:

- `pressure`  - pressure (psi / bar).
- `pv_mult`   - pore-volume multiplier at that pressure (dimensionless).
- `trans_mult` - transmissibility multiplier (optional, default 1.0).
"""

SWOF = PVTTableKeyword(
    "SWOF",
    columns=[
        Field("sw", np.float64),
        Field("krw", np.float64),
        Field("krow", np.float64),
        Field("pcow", np.float64),
    ],
)
"""
`SWOF` - water saturation function table (first relperm family).

Each table (one per saturation region) contains rows:

- `sw`   - water saturation [-].
- `krw`  - water relative permeability [-].
- `krow` - oil relative permeability in presence of water [-].
- `pcow` - oil-water capillary pressure Pcow = Po - Pw (psi / bar).

Rows must be in ascending `sw` order.
"""

SGOF = PVTTableKeyword(
    "SGOF",
    columns=[
        Field("sg", np.float64),
        Field("krg", np.float64),
        Field("krog", np.float64),
        Field("pcog", np.float64),
    ],
)
"""
`SGOF` - gas saturation function table (first relperm family).

Each table (one per saturation region) contains rows:

- `sg`   - gas saturation [-].
- `krg`  - gas relative permeability [-].
- `krog` - oil relative permeability in presence of gas [-].
- `pcog` - gas-oil capillary pressure Pcog = Pg - Po (psi / bar).

Rows must be in ascending `sg` order.
"""

SWFN = PVTTableKeyword(
    "SWFN",
    columns=[
        Field("sw", np.float64),
        Field("krw", np.float64),
        Field("pcow", np.float64),
    ],
)
"""
`SWFN` - water saturation function table (second relperm family).

Each table contains rows:

- `sw`   - water saturation [-].
- `krw`  - water relative permeability [-].
- `pcow` - oil-water capillary pressure (psi / bar).

Used alongside `SOF2` or `SOF3` in the second saturation-function
family (as opposed to the `SWOF`/`SGOF` first family).
"""

SGFN = PVTTableKeyword(
    "SGFN",
    columns=[
        Field("sg", np.float64),
        Field("krg", np.float64),
        Field("pcog", np.float64),
    ],
)
"""
`SGFN` - gas saturation function table (second relperm family).

Each table contains rows:

- `sg`   - gas saturation [-].
- `krg`  - gas relative permeability [-].
- `pcog` - gas-oil capillary pressure (psi / bar).
"""


SOF2 = PVTTableKeyword(
    "SOF2",
    columns=[
        Field("so", np.float64),
        Field("kro", np.float64),
    ],
)
"""
`SOF2` - oil relative permeability vs. oil saturation (two-phase).

Each table contains rows:

- `so`  - oil saturation [-].
- `kro` - oil relative permeability [-].

Used in two-phase (oil-water or oil-gas) runs with the second
saturation-function family.
"""

SOF3 = PVTTableKeyword(
    "SOF3",
    columns=[
        Field("so", np.float64),
        Field("krow", np.float64),
        Field("krog", np.float64),
    ],
)
"""
`SOF3` - oil relative permeability vs. oil saturation (three-phase).

Each table contains rows:

- `so`   - oil saturation [-].
- `krow` - oil relative permeability (water-oil system) [-].
- `krog` - oil relative permeability (gas-oil system) [-].

Used in three-phase runs with the second saturation-function family
alongside `SWFN` and `SGFN`.
"""
