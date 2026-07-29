"""Phase mobility cache for black-oil simulation"""

import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.precision import get_dtype
from bores.simulation.blackoil.caches.pvt import PVTCache
from bores.simulation.blackoil.caches.satfunc import SatFuncCache
from bores.typing import CellArray

__all__ = ["MobilityCache", "compute_mobility_cache"]


class MobilityCache(typing.NamedTuple):
    """
    Per-cell phase mobilities, fractional flows, and their pressure/saturation
    derivatives. Every field is shape `(n_cells,)`.

    `total_mobility` and its derivatives are the sum of the three phase
    mobilities/derivatives.

    Fractional flows (`water_fractional_flow` etc.) are values only, no
    derivatives - whether a solver formulation needs
    `d(fractional_flow)/d(...)` at all depends on whether it ever works in
    terms of fractional flow rather than phase mobility x phase potential
    directly, which isn't settled yet. Add them here if/when it turns out
    they're needed rather than building unused surface now.

    Where `total_mobility[i] == 0.0` (all three phases immobile, e.g. a cell
    pinned at residual saturations for every phase), fractional flows are
    `0.0`, not `NaN`: unlike `PVTCache`/`SatFuncCache`'s
    NaN-for-absent-data convention, a zero-total-mobility cell is a normal,
    physically meaningful state, not a modelling gap as that means "no phase is flowing"
    and is not a failure.
    """

    water_mobility: CellArray
    oil_mobility: CellArray
    gas_mobility: CellArray
    total_mobility: CellArray

    water_fractional_flow: CellArray
    oil_fractional_flow: CellArray
    gas_fractional_flow: CellArray

    dʎw_dP: CellArray
    dʎo_dP: CellArray
    dʎg_dP: CellArray
    dʎT_dP: CellArray

    # d(phase mobility)/d(water saturation)
    dʎw_dsw: CellArray
    dʎo_dsw: CellArray
    dʎg_dsw: CellArray

    # d(phase mobility)/d(oil saturation)
    dʎw_dso: CellArray
    dʎo_dso: CellArray
    dʎg_dso: CellArray

    # d(phase mobility)/d(gas saturation)
    dʎw_dsg: CellArray
    dʎo_dsg: CellArray
    dʎg_dsg: CellArray

    dʎT_dsw: CellArray
    dʎT_dso: CellArray
    dʎT_dsg: CellArray


CACHE_FIELDS: typing.Tuple[str, ...] = MobilityCache._fields


def make_new_cache(n_cells: int, dtype: npt.DTypeLike) -> MobilityCache:
    return MobilityCache(
        **{name: np.zeros(n_cells, dtype=dtype) for name in CACHE_FIELDS}
    )


@numba.njit(cache=True, parallel=True)
def _update_mobility_cache(
    pvt_cache: PVTCache, satfunc_cache: SatFuncCache, out: MobilityCache
) -> None:
    """
    Fill `out` in place from `pvt_cache`/`satfunc_cache`. Pure arithmetic,
    one cell at a time, no allocation - every `out` array is written to by
    index, nothing is created here. Not part of the public API - called only
    by `compute_mobility_cache`, which guarantees `out` is already correctly
    sized.

    Every array used in the loop below is bound to a local variable *before*
    `numba.prange` starts, never accessed via `pvt_cache.<field>`/`out.<field>`
    attribute lookup inside the loop body itself. This isn't just style: with
    `parallel=True`, resolving a `NamedTuple` field *inside* the `prange`
    loop silently produced wrong results when checked against the
    non-parallel version (numba's auto-parallelisation array-analysis pass
    doesn't see through the attribute access correctly) - hoisting every
    field to a local array reference first, so the loop body only ever
    indexes plain local arrays, is the fix, verified the same way.
    """
    water_viscosity = pvt_cache.water_viscosity
    oil_viscosity = pvt_cache.oil_viscosity
    gas_viscosity = pvt_cache.gas_viscosity
    dμw_dp = pvt_cache.dμw_dp
    dμo_dp = pvt_cache.dμo_dp
    dμg_dp = pvt_cache.dμg_dp

    water_relative_permeability = satfunc_cache.water_relative_permeability
    oil_relative_permeability = satfunc_cache.oil_relative_permeability
    gas_relative_permeability = satfunc_cache.gas_relative_permeability
    dkrw_dsw = satfunc_cache.dkrw_dsw
    dkro_dsw = satfunc_cache.dkro_dsw
    dkrg_dsw = satfunc_cache.dkrg_dsw
    dkrw_dso = satfunc_cache.dkrw_dso
    dkro_dso = satfunc_cache.dkro_dso
    dkrg_dso = satfunc_cache.dkrg_dso
    dkrw_dsg = satfunc_cache.dkrw_dsg
    dkro_dsg = satfunc_cache.dkro_dsg
    dkrg_dsg = satfunc_cache.dkrg_dsg

    out_water_mobility = out.water_mobility
    out_oil_mobility = out.oil_mobility
    out_gas_mobility = out.gas_mobility
    out_total_mobility = out.total_mobility
    out_water_fractional_flow = out.water_fractional_flow
    out_oil_fractional_flow = out.oil_fractional_flow
    out_gas_fractional_flow = out.gas_fractional_flow
    out_dʎw_dP = out.dʎw_dP
    out_dʎo_dP = out.dʎo_dP
    out_dʎg_dP = out.dʎg_dP
    out_dʎT_dP = out.dʎT_dP
    out_dʎw_dsw = out.dʎw_dsw
    out_dʎo_dsw = out.dʎo_dsw
    out_dʎg_dsw = out.dʎg_dsw
    out_dʎT_dsw = out.dʎT_dsw
    out_dʎw_dso = out.dʎw_dso
    out_dʎo_dso = out.dʎo_dso
    out_dʎg_dso = out.dʎg_dso
    out_dʎT_dso = out.dʎT_dso
    out_dʎw_dsg = out.dʎw_dsg
    out_dʎo_dsg = out.dʎo_dsg
    out_dʎg_dsg = out.dʎg_dsg
    out_dʎT_dsg = out.dʎT_dsg

    n_cells = water_viscosity.shape[0]
    for i in numba.prange(n_cells):
        mu_w = water_viscosity[i]
        mu_o = oil_viscosity[i]
        mu_g = gas_viscosity[i]
        water_relative_permeability = water_relative_permeability[i]
        oil_relative_permeability = oil_relative_permeability[i]
        gas_relative_permeability = gas_relative_permeability[i]

        # Values: ʎphase = kr_phase / mu_phase
        ʎw = water_relative_permeability / mu_w
        ʎo = oil_relative_permeability / mu_o
        ʎg = gas_relative_permeability / mu_g
        ʎt = ʎw + ʎo + ʎg

        out_water_mobility[i] = ʎw
        out_oil_mobility[i] = ʎo
        out_gas_mobility[i] = ʎg
        out_total_mobility[i] = ʎt

        if ʎt != 0.0:
            out_water_fractional_flow[i] = ʎw / ʎt
            out_oil_fractional_flow[i] = ʎo / ʎt
            out_gas_fractional_flow[i] = ʎg / ʎt
        else:
            out_water_fractional_flow[i] = 0.0
            out_oil_fractional_flow[i] = 0.0
            out_gas_fractional_flow[i] = 0.0

        # Pressure derivatives: kr doesn't depend on P, so this is a plain
        # quotient rule with kr held constant:
        # d(ʎphase)/dP = -kr_phase * d(mu_phase)/dP / mu_phase**2
        dʎw_dp = -water_relative_permeability * dμw_dp[i] / mu_w**2
        dʎo_dp = -oil_relative_permeability * dμo_dp[i] / mu_o**2
        dʎg_dp = -gas_relative_permeability * dμg_dp[i] / mu_g**2
        out_dʎw_dP[i] = dʎw_dp
        out_dʎo_dP[i] = dʎo_dp
        out_dʎg_dP[i] = dʎg_dp
        out_dʎT_dP[i] = dʎw_dp + dʎo_dp + dʎg_dp

        # Saturation derivatives: mu doesn't depend on saturation, so this is
        # a plain chain rule: d(ʎphase)/dS_j = d(kr_phase)/dS_j / mu_phase
        dʎw_dsw = dkrw_dsw[i] / mu_w
        dʎo_dsw = dkro_dsw[i] / mu_o
        dʎg_dsw = dkrg_dsw[i] / mu_g
        out_dʎw_dsw[i] = dʎw_dsw
        out_dʎo_dsw[i] = dʎo_dsw
        out_dʎg_dsw[i] = dʎg_dsw
        out_dʎT_dsw[i] = dʎw_dsw + dʎo_dsw + dʎg_dsw

        dʎw_dso = dkrw_dso[i] / mu_w
        dʎo_dso = dkro_dso[i] / mu_o
        dʎg_dso = dkrg_dso[i] / mu_g
        out_dʎw_dso[i] = dʎw_dso
        out_dʎo_dso[i] = dʎo_dso
        out_dʎg_dso[i] = dʎg_dso
        out_dʎT_dso[i] = dʎw_dso + dʎo_dso + dʎg_dso

        dʎw_dsg = dkrw_dsg[i] / mu_w
        dʎo_dsg = dkro_dsg[i] / mu_o
        dʎg_dsg = dkrg_dsg[i] / mu_g
        out_dʎw_dsg[i] = dʎw_dsg
        out_dʎo_dsg[i] = dʎo_dsg
        out_dʎg_dsg[i] = dʎg_dsg
        out_dʎT_dsg[i] = dʎw_dsg + dʎo_dsg + dʎg_dsg


def compute_mobility_cache(
    pvt_cache: PVTCache,
    satfunc_cache: SatFuncCache,
    out: typing.Optional[MobilityCache] = None,
    dtype: npt.DTypeLike = None,
) -> MobilityCache:
    """
    Build (or refresh, in place) a `MobilityCache` from a `PVTCache` and a
    `SatFuncCache`.

    **Reuse**: pass a previous call's returned `MobilityCache` as `out` to
    overwrite its arrays in place; the return value is `out` itself in that
    case.

    :param pvt_cache: `PVTCache` from `caches.pvt.compute_pvt_cache`, same cell count
        and ordering as `satfunc_cache`.
    :param satfunc_cache: `SatFuncCache` from
        `caches.satfunc.compute_satfunc_cache`.
    :param out: Previous `MobilityCache` to overwrite in place, or `None` to
        allocate a new one.
    :return: The populated `MobilityCache`. `out` itself if given, otherwise
        a newly allocated one. Always returned, never `None`.
    :raises ValueError: If `pvt_cache` and `satfunc_cache` disagree on cell
        count, or `out` is given but sized for a different cell count.
    """
    n_cells = len(pvt_cache.water_viscosity)
    if len(satfunc_cache.water_relative_permeability) != n_cells:
        raise ValueError(
            f"`pvt_cache` has {n_cells} cells; `satfunc_cache` has "
            f"{len(satfunc_cache.water_relative_permeability)}."
        )

    if out is not None:
        if len(out.water_mobility) != n_cells:
            raise ValueError(
                f"`out` was allocated for {len(out.water_mobility)} cells; "
                f"`pvt_cache`/`satfunc_cache` have {n_cells}."
            )
        cache = out
    else:
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        cache = make_new_cache(n_cells, dtype=dtype)

    _update_mobility_cache(pvt_cache=pvt_cache, satfunc_cache=satfunc_cache, out=cache)
    return cache
