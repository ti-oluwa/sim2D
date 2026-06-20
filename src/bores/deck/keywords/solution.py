"""
SOLUTION section keyword implementations.

The SOLUTION section specifies initial reservoir conditions: either as
explicit per-cell arrays (`PRESSURE`, `SWAT`, `SGAS`, `RS`,
`RV`) or via gravity/capillary equilibration (`EQUIL`), or by
restarting from a previously saved state (`RESTART`).

**Explicit initialisation keywords** (return shape `(n_cells,)`
float64 arrays):

- `PRESSURE` - initial reservoir pressure (psi in FIELD, bar in
  METRIC).
- `SWAT` / `SW` - initial water saturation.
- `SOIL` / `SO` - initial oil saturation.
- `SGAS` / `SG` - initial gas saturation.
- `RS` - initial solution gas-oil ratio.
- `RV` - initial vaporised oil-gas ratio.

**Equilibration keyword**:

- `EQUIL` - one record per equilibration region specifying datum
  depth/pressure, contact depths, and capillary pressure at contacts.

**Restart keyword**:

- `RESTART` - resumes from a previously written restart file instead
  of explicit initial conditions.
"""

import typing

import numpy as np

from bores.deck.core import Deck, DeckParseError, GridDimensions, tokenise
from bores.deck.keywords.base import (
    Field,
    GridArrayKeyword,
    Keyword,
    RepeatedRecordKeyword,
)
from bores.deck.operators import Operation

__all__ = [
    "SWAT",
    "SOIL",
    "SGAS",
    "SW",
    "SO",
    "SG",
    "PRESSURE",
    "RS",
    "RV",
    "EQUIL",
    "RESTART",
]

SWAT = SW = GridArrayKeyword("SWAT", dtype=np.float64, default_value=0.0)
"""`SWAT` / `SW` - initial water saturation `[0, 1]`."""

SOIL = SO = GridArrayKeyword("SOIL", dtype=np.float64, default_value=0.0)
"""`SOIL` / `SO` - initial oil saturation `[0, 1]`."""

SGAS = SG = GridArrayKeyword("SGAS", dtype=np.float64, default_value=0.0)
"""`SGAS` / `SG` - initial gas saturation `[0, 1]`."""

PRESSURE = GridArrayKeyword("PRESSURE", dtype=np.float64, default_value=0.0)
"""`PRESSURE` - initial reservoir pressure (psi in FIELD, barsa in METRIC)."""

RS = GridArrayKeyword("RS", dtype=np.float64, default_value=0.0)
"""`RS` - initial solution gas-oil ratio (scf/stb in FIELD)."""

RV = GridArrayKeyword("RV", dtype=np.float64, default_value=0.0)
"""`RV` - initial vaporised oil-gas ratio (stb/scf in FIELD)."""


EQUIL = RepeatedRecordKeyword(
    "EQUIL",
    fields=[
        Field("datum_depth", float),
        Field("datum_pressure", float),
        Field("woc_depth", float, required=False, default=0.0),
        Field("pcow_woc", float, required=False, default=0.0),
        Field("goc_depth", float, required=False, default=0.0),
        Field("pcog_goc", float, required=False, default=0.0),
        Field("rsvd_table", int, required=False, default=0),
        Field("rvvd_table", int, required=False, default=0),
        Field("accuracy_flag", int, required=False, default=0),
    ],
)
"""
`EQUIL  DATUM_DEPTH  DATUM_PRESSURE  WOC  PCOW_WOC  GOC  PCOG_GOC
RSVD_TABLE  RVVD_TABLE  ACCURACY_FLAG /` (one record per equilibration
region) - gravity/capillary equilibration data, used to derive initial
pressure and saturation distributions instead of supplying explicit
`PRESSURE` / `SWAT` / `SGAS` arrays.
 
One record per `EQLNUM` region; multiple records (one per region) are
separated by `/` within a single `EQUIL` block, and multiple `EQUIL`
blocks in the same deck are concatenated in file order.
 
Fields:
 
- `datum_depth`    - depth of the datum point at which `datum_pressure`
  applies.
- `datum_pressure` - reservoir pressure at `datum_depth`.
- `woc_depth`      - water-oil contact depth.
- `pcow_woc`       - oil-water capillary pressure at the WOC (usually
  `0`).
- `goc_depth`      - gas-oil contact depth.
- `pcog_goc`       - gas-oil capillary pressure at the GOC (usually
  `0`).
- `rsvd_table`      - `RSVD`-table number for depth-dependent solution
  GOR below the bubble point (`0` = use the `PVTO` bubble-point Rs
  instead, i.e. assume saturated oil).
- `rvvd_table`      - `RVVD`-table number for depth-dependent
  vaporised-oil ratio above the dew point (`0` = use the `PVTG`
  dew-point Rv instead, i.e. assume saturated gas).
- `accuracy_flag`   - initialization accuracy/option switch (e.g.
  selecting compositional-vs-black-oil equilibration nuances); `0` is
  the simulator default behaviour.
 
Note:
    The exact semantics of `accuracy_flag` (item 9) vary slightly
    between Eclipse 100 and Eclipse 300 / compositional runs; verify
    against your simulator's manual if this run uses anything beyond
    plain black-oil equilibration.
"""


class RestartKeyword(Keyword[typing.Dict[str, typing.Any]]):
    """
    The `RESTART` keyword: resume a run from a previously written
    restart file instead of the explicit `PRESSURE` / `SWAT` / `SGAS` /
    `EQUIL` initial-conditions keywords.

    `parse` returns `{"root_name": str, "report_step": int}`, or
    `None` if absent.

    Example deck fragment:

        RESTART
         CASE1 100 /
    """

    def __init__(self) -> None:
        super().__init__("RESTART")

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> typing.Optional[typing.Dict[str, typing.Any]]:
        record = deck.first_record_for(self.name)
        if record is None:
            return None

        tokens = tokenise(record.body)
        if len(tokens) < 2:
            raise DeckParseError(
                f"RESTART: expected 2 tokens (ROOT_NAME REPORT_STEP); "
                f"got {len(tokens)}: {list(tokens)!r}."
            )
        try:
            report_step = int(tokens[1])
        except ValueError as exc:
            raise DeckParseError(
                f"RESTART: report step token {tokens[1]!r} is not an integer."
            ) from exc

        return {"root_name": tokens[0], "report_step": report_step}


RESTART = RestartKeyword()
"""
`RESTART  ROOT_NAME  REPORT_STEP /` - resume from a saved restart file.
 
Fields (returned as a dict rather than via `Field`, since the second
token's "is this required" rule depends only on simple positional
arity, not Eclipse's usual optional/`1*` trailing-default convention):
 
- `root_name`   - root filename (without extension) of the restart
  case to resume from.
- `report_step` - report step number within that case to resume from.
 
`parse` returns `{"root_name": str, "report_step": int}`, or `None`
if absent.
"""
