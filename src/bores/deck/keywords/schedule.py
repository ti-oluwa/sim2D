"""
SCHEDULE section keyword implementations.

The SCHEDULE section is the last section of an Eclipse black-oil deck. It
advances simulated time (`DATES` / `TSTEP`) and defines/controls wells and
groups as the run proceeds. Unlike the GRID/PROPS sections, most SCHEDULE
keywords are *time-ordered events* rather than static data: the same well
can be opened, closed, and re-targeted by repeated keyword occurrences at
different points in the deck, and callers are expected to apply them in
file order alongside the `DATES`/`TSTEP` timeline.

**Time advancement**:

- `DATES` - see `bores.deck.keywords.base.DatesKeyword`, registered
  here as part of the SCHEDULE keyword set.
- `TSTEP` - see `bores.deck.keywords.base.TStepKeyword`, likewise.

**Well/group definition**:

- `WELSPECS` - declare a well (location, group, preferred phase).
- `COMPDAT`  - declare/modify well connections (completions).
- `GRUPTREE` - declare the group hierarchy.

**Well/group control**:

- `WCONPROD` - producer rate/pressure targets.
- `WCONINJE` - injector rate/pressure targets.
- `WELOPEN`  - open/shut/stop a well or specific connections.
- `WELTARG`  - modify a single control target on an existing well.
- `WPIMULT`  - well productivity-index multiplier.
- `GCONPROD` - group production targets/guide rates.
- `GCONINJE` - group injection targets.

**Economic limits / well testing**:

- `WECON` - well economic limits (auto-shut-in thresholds).
- `WTEST` - automatic well re-opening / testing schedule.
"""
import typing

from bores.deck.core import Deck, DeckParseError, GridDimensions, tokenise
from bores.deck.keywords.base import DatesKeyword, Field, Keyword, RepeatedRecordKeyword
from bores.deck.operators import Operation

__all__ = [
    "DATES",
    "TSTEP",
    "WELSPECS",
    "COMPDAT",
    "WCONPROD",
    "WCONINJE",
    "WELOPEN",
    "WELTARG",
    "WPIMULT",
    "GRUPTREE",
    "GCONPROD",
    "GCONINJE",
    "WECON",
    "WTEST",
]


DATES = DatesKeyword("DATES")
"""
`DATES  D MON YYYY / ... /` - advance simulated time to one or more
explicit calendar dates.

See `bores.deck.keywords.base.DatesKeyword`. `parse` returns a
`List[datetime.date]`, or `None` if absent.
"""


class TStepKeyword(Keyword[typing.List[float]]):
    """
    The `TSTEP` keyword: a flat list of time-step sizes terminated by
    `/`.

    `N*value` repeat syntax is already expanded by
    `bores.deck.core.tokenise`, so `30*30` correctly yields
    thirty entries of `30.0`.

    Multiple `TSTEP` blocks in the same deck are concatenated in file
    order, consistent with Eclipse semantics.

    `parse` returns a `List[float]`, or `None` when the keyword
    is absent.

    Example deck fragment:

        TSTEP
         30 30 30 90 /
    """

    def __init__(self) -> None:
        super().__init__("TSTEP")

    def parse(
        self,
        deck: Deck,
        dims: typing.Optional[GridDimensions],
        *,
        operations: typing.Optional[typing.List[Operation]] = None,
    ) -> typing.Optional[typing.List[float]]:
        records = deck.records_for(self.name)
        if not records:
            return None

        steps: typing.List[float] = []
        for record in records:
            tokens = tokenise(record.body)
            for token in tokens:
                try:
                    steps.append(float(token))
                except ValueError as exc:
                    raise DeckParseError(
                        f"TSTEP: non-numeric time-step value {token!r}: {exc}"
                    ) from exc

        return steps or None


TSTEP = TStepKeyword()
"""
`TSTEP  dt1 dt2 ... /` - advance simulated time by one or more explicit
step sizes (in the deck's declared time unit, days for FIELD/METRIC).
"""

WELSPECS = RepeatedRecordKeyword(
    "WELSPECS",
    fields=[
        Field("well", str),
        Field("group", str),
        Field("i", int),
        Field("j", int),
        Field("ref_depth", float, required=False, default=None),
        Field("phase", str, required=False, default="OIL"),
        Field("drainage_radius", float, required=False, default=0.0),
        Field("inflow_eq", str, required=False, default="STD"),
        Field("auto_shut", str, required=False, default="SHUT"),
        Field("crossflow", str, required=False, default="YES"),
        Field("pvt_table", int, required=False, default=0),
        Field("density_calc", str, required=False, default="SEG"),
    ],
)
"""
`WELSPECS 'WELL' 'GROUP' I J [ref_depth] [phase] ... / ... /`
- declare a well's name, parent group, surface location, and defaults.

Multiple `WELSPECS` blocks in the same deck are concatenated in file
order (a deck typically only declares each well once, but Eclipse permits
later blocks to add more wells as drilling progresses).

Fields:

- `well`            - well name.
- `group`           - parent group name.
- `i` / `j`         - 1-based structured grid location of the wellhead.
- `ref_depth`       - BHP reference depth; defaults to the first
  completion's mid-perforation depth when absent (`None` here).
- `phase`           - preferred phase (`OIL`, `WATER`, `GAS`).
- `drainage_radius` - well drainage radius for PI calculations.
- `inflow_eq`       - inflow equation type (`STD` or `NO`).
- `auto_shut`       - automatic shut-in behaviour (`SHUT` or `STOP`).
- `crossflow`       - whether crossflow between completions is allowed.
- `pvt_table`       - PVT region override (`0` = use cell's `PVTNUM`).
- `density_calc`    - wellbore density calculation method (`SEG` or
  `AVG`).
"""

COMPDAT = RepeatedRecordKeyword(
    "COMPDAT",
    fields=[
        Field("well", str),
        Field("i", int),
        Field("j", int),
        Field("k1", int),
        Field("k2", int),
        Field("status", str, required=False, default="OPEN"),
        Field("sat_table", int, required=False, default=0),
        Field("connection_factor", float, required=False, default=None),
        Field("diameter", float, required=False, default=0.0),
        Field("kh", float, required=False, default=None),
        Field("skin", float, required=False, default=0.0),
        Field("d_factor", float, required=False, default=0.0),
        Field("direction", str, required=False, default="Z"),
        Field("perm_thickness_mult", float, required=False, default=1.0),
    ],
)
"""
`COMPDAT 'WELL' I J K1 K2 [status] [sat_table] [conn_factor] ... / ... /`
- declare or modify well connections (completions).

14 standard items; the last several are very commonly defaulted (`1*`).
Multiple `COMPDAT` blocks are concatenated in file order, since the same
well/connection can be re-completed or re-parametrised later in the
schedule.

Fields:

- `well`                - well name.
- `i` / `j`             - 1-based structured grid column of the connection.
- `k1` / `k2`           - 1-based top/bottom layer of the connection
  interval (a single-layer connection has `k1 == k2`).
- `status`              - `OPEN` or `SHUT`.
- `sat_table`           - saturation table number override (`0` = use
  the cell's `SATNUM`).
- `connection_factor`   - explicit transmissibility/connection factor;
  `None` means Eclipse computes it from geometry and `PERMX`/`PERMY`.
- `diameter`            - wellbore diameter at this connection.
- `kh`                  - explicit permeability-thickness product;
  `None` means Eclipse computes it from the grid.
- `skin`                - mechanical skin factor.
- `d_factor`            - non-Darcy (rate-dependent) skin factor.
- `direction`           - completion direction (`X`, `Y`, or `Z`).
- `perm_thickness_mult` - additional permeability-thickness multiplier.
"""

WCONPROD = RepeatedRecordKeyword(
    "WCONPROD",
    fields=[
        Field("well", str),
        Field("status", str, required=False, default="OPEN"),
        Field("control_mode", str, required=False, default=None),
        Field("orat", float, required=False, default=0.0),
        Field("wrat", float, required=False, default=0.0),
        Field("grat", float, required=False, default=0.0),
        Field("lrat", float, required=False, default=0.0),
        Field("resv", float, required=False, default=0.0),
        Field("bhp", float, required=False, default=None),
        Field("thp", float, required=False, default=None),
        Field("vfp_table", int, required=False, default=0),
    ],
)
"""
`WCONPROD 'WELL' [status] [control_mode] [orat] ... / ... /`
- producer rate/pressure targets and the active control mode.

Fields:

- `well`         - well name.
- `status`       - `OPEN`, `SHUT`, `STOP`, or `AUTO`.
- `control_mode` - the constraint Eclipse actively controls the well
  by (e.g. `ORAT`, `WRAT`, `GRAT`, `LRAT`, `RESV`, `BHP`, `THP`,
  `GRUP`); `None` is only valid if the well is shut/stopped.
- `orat` / `wrat` / `grat` / `lrat` - oil/water/gas/liquid rate
  upper-limit targets.
- `resv`         - reservoir-volume rate upper-limit target.
- `bhp` / `thp`  - bottom-hole / tubing-head pressure limits;
  `None` means no limit.
- `vfp_table`    - VFP (vertical flow performance) table number for
  THP-to-BHP conversion (`0` = none assigned).
"""

WCONINJE = RepeatedRecordKeyword(
    "WCONINJE",
    fields=[
        Field("well", str),
        Field("injector_type", str),
        Field("status", str, required=False, default="OPEN"),
        Field("control_mode", str, required=False, default=None),
        Field("rate", float, required=False, default=0.0),
        Field("resv", float, required=False, default=0.0),
        Field("bhp", float, required=False, default=None),
        Field("thp", float, required=False, default=None),
        Field("vfp_table", int, required=False, default=0),
    ],
)
"""
`WCONINJE 'WELL' TYPE [status] [control_mode] [rate] ... / ... /`
- injector rate/pressure targets, active control mode, and injected
fluid type.

Fields:

- `well`          - well name.
- `injector_type` - injected fluid: `WATER`, `GAS`, or `OIL`.
- `status`        - `OPEN`, `SHUT`, `STOP`, or `AUTO`.
- `control_mode`  - the constraint Eclipse actively controls the well
  by (`RATE`, `RESV`, `BHP`, `THP`, or `GRUP`).
- `rate`          - surface injection rate upper-limit target.
- `resv`          - reservoir-volume injection rate upper-limit target.
- `bhp` / `thp`   - bottom-hole / tubing-head pressure limits;
  `None` means no limit.
- `vfp_table`     - VFP table number for THP-to-BHP conversion
  (`0` = none assigned).
"""


class WelOpenKeyword(RepeatedRecordKeyword):
    """
    `WELOPEN 'WELL' STATUS [I J K1 K2] ... / ... /`
    - open, shut, or stop a well, or one of its connections.

    When `i`/`j`/`k1`/`k2` are all absent (or `0`), the action applies to
    the whole well; when given, it applies only to the connection(s) at
    that location (or layer range `k1`-`k2` at column `(i, j)`).
    """

    _VALID_STATUSES: typing.FrozenSet[str] = frozenset({"OPEN", "SHUT", "STOP", "AUTO"})

    def __init__(self) -> None:
        super().__init__(
            "WELOPEN",
            fields=[
                Field("well", str),
                Field("status", str),
                Field("i", int, required=False, default=0),
                Field("j", int, required=False, default=0),
                Field("k1", int, required=False, default=0),
                Field("k2", int, required=False, default=0),
            ],
        )

    def _parse_tokens(
        self, tokens: typing.Sequence[str]
    ) -> typing.Dict[str, typing.Any]:
        result = super()._parse_tokens(tokens)
        status = str(result["status"]).upper()
        if status not in self._VALID_STATUSES:
            raise DeckParseError(
                f"WELOPEN record for {result.get('well')!r}: unrecognised "
                f"status {status!r}. Valid values: "
                f"{sorted(self._VALID_STATUSES)}."
            )
        result["status"] = status
        return result


WELOPEN = WelOpenKeyword()
"""
`WELOPEN 'WELL' STATUS [I J K1 K2] ... / ... /`
- open, shut, or stop a well, or one of its connections.

Fields:

- `well`   - well name.
- `status` - `OPEN`, `SHUT`, `STOP`, or `AUTO`.
- `i` / `j` / `k1` / `k2` - optional connection location/layer range to
  restrict the action to a single connection (or range of layers) rather
  than the whole well; all default to `0`, meaning "whole well".
"""

WELTARG = RepeatedRecordKeyword(
    "WELTARG",
    fields=[
        Field("well", str),
        Field("control_mode", str),
        Field("value", float),
    ],
)
"""
`WELTARG 'WELL' CONTROL_MODE VALUE / ... /`
- modify a single existing control target on a well without re-stating
its full `WCONPROD` / `WCONINJE` record.

Fields:

- `well`         - well name.
- `control_mode` - target being modified (e.g. `ORAT`, `BHP`, `RESV`).
- `value`        - new value for that target.
"""

WPIMULT = RepeatedRecordKeyword(
    "WPIMULT",
    fields=[
        Field("well", str),
        Field("multiplier", float),
        Field("i", int, required=False, default=0),
        Field("j", int, required=False, default=0),
        Field("k", int, required=False, default=0),
    ],
)
"""
`WPIMULT 'WELL' MULTIPLIER [I J K] / ... /`
- multiply a well's (or a single connection's) productivity index.

Fields:

- `well`       - well name.
- `multiplier` - PI multiplier applied on top of the existing value.
- `i` / `j` / `k` - optional connection location to restrict the
  multiplier to a single connection; all default to `0`, meaning
  "every connection on this well".
"""

GRUPTREE = RepeatedRecordKeyword(
    "GRUPTREE",
    fields=[
        Field("child", str),
        Field("parent", str),
    ],
)
"""
`GRUPTREE 'CHILD' 'PARENT' / ... /`
- declare one parent/child link in the well-group hierarchy.

Each record adds one group (or well group membership) under a parent
group; the implicit root group is always named `'FIELD'`. Multiple
`GRUPTREE` blocks are concatenated in file order.

Fields:

- `child`  - group (or sub-tree) name being attached.
- `parent` - parent group name (`'FIELD'` for top-level groups).
"""

GCONPROD = RepeatedRecordKeyword(
    "GCONPROD",
    fields=[
        Field("group", str),
        Field("control_mode", str),
        Field("orat", float, required=False, default=0.0),
        Field("wrat", float, required=False, default=0.0),
        Field("grat", float, required=False, default=0.0),
        Field("lrat", float, required=False, default=0.0),
        Field("exceed_action", str, required=False, default="NONE"),
    ],
)
"""
`GCONPROD 'GROUP' CONTROL_MODE [orat] [wrat] [grat] [lrat] [exceed_action] / ... /`
- group-level production targets / guide-rate control.

Fields:

- `group`         - group name.
- `control_mode`  - constraint the group is controlled by (`ORAT`,
  `WRAT`, `GRAT`, `LRAT`, `RESV`, `FLD`, or `NONE`).
- `orat` / `wrat` / `grat` / `lrat` - oil/water/gas/liquid rate
  upper-limit targets for the group.
- `exceed_action` - action when an individual well in the group would
  exceed its share of the group target (`NONE`, `RATE`, `CON`, ...).
"""

GCONINJE = RepeatedRecordKeyword(
    "GCONINJE",
    fields=[
        Field("group", str),
        Field("injector_type", str),
        Field("control_mode", str),
        Field("rate", float, required=False, default=0.0),
        Field("resv", float, required=False, default=0.0),
    ],
)
"""
`GCONINJE 'GROUP' TYPE CONTROL_MODE [rate] [resv] / ... /`
- group-level injection targets.

Fields:

- `group`         - group name.
- `injector_type` - injected fluid: `WATER`, `GAS`, or `OIL`.
- `control_mode`  - constraint the group is controlled by (`RATE`,
  `RESV`, `VREP`, `REIN`, or `FLD`).
- `rate`          - surface injection rate upper-limit target for the
  group.
- `resv`          - reservoir-volume injection rate upper-limit target
  for the group.
"""

WECON = RepeatedRecordKeyword(
    "WECON",
    fields=[
        Field("well", str),
        Field("min_orat", float, required=False, default=0.0),
        Field("max_wcut", float, required=False, default=None),
        Field("max_gor", float, required=False, default=None),
        Field("max_wgr", float, required=False, default=None),
        Field("workover_action", str, required=False, default="WELL"),
        Field("end_run_flag", str, required=False, default="NO"),
    ],
)
"""
`WECON 'WELL' [min_orat] [max_wcut] [max_gor] [max_wgr] [workover_action] [end_run_flag] / ... /`
- economic limits that automatically shut in or work over a well.

Fields:

- `well`            - well name.
- `min_orat`        - minimum economic oil production rate; the well is
  shut in (per `workover_action`) once production falls below this.
- `max_wcut`        - maximum water cut before workover/shut-in;
  `None` means no limit.
- `max_gor`         - maximum gas-oil ratio before workover/shut-in;
  `None` means no limit.
- `max_wgr`         - maximum water-gas ratio before workover/shut-in;
  `None` means no limit.
- `workover_action` - action taken when a limit is breached (`NONE`,
  `CON`, `+CON`, `WELL`, `PLUG`).
- `end_run_flag`    - whether breaching this limit should end the
  simulation run (`YES` or `NO`).
"""

WTEST = RepeatedRecordKeyword(
    "WTEST",
    fields=[
        Field("well", str),
        Field("interval", float),
        Field("reason", str, required=False, default="PEW"),
    ],
)
"""
`WTEST 'WELL' INTERVAL [reason] / ... /`
- schedule automatic periodic re-opening ("testing") of a well that was
shut in for an economic or operational reason.

Fields:

- `well`     - well name.
- `interval` - time between re-open attempts, in the deck's declared
  time unit.
- `reason`   - which shut-in reason(s) this test schedule applies to,
  as a string of one-letter codes: `P` (economic), `E` (group control
  efficiency), `W` (workover/economic limit, `WECON`); default `"PEW"`
  applies to all three.
"""
