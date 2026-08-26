"""
Saturation-function (relative permeability / capillary pressure) property
cache for black-oil simulation.
"""

import typing

import numpy as np
import numpy.typing as npt

from bores.blackoil.satfunc.regions import SatFunc
from bores.precision import get_dtype
from bores.typing import CellArray, IntCellArray

__all__ = ["SatFuncCache", "compute_satfunc_cache"]


class SatFuncCache(typing.NamedTuple):
    """
    Per-cell relative permeabilities, capillary pressures, and their
    saturation derivatives. Every field is shape `(n_cells,)`.

    Relative permeability derivatives are the full nine-way cross set, e.g.
    `dkro_dsw` (`dkro/dsw`) is
    generally nonzero in a three-phase mixing-rule model, not just the three
    "own-saturation" diagonal terms so it's kept in full rather than reduced to
    whatever two-saturation primary-variable convention a solver ends up
    using, since that reduction is a solver-side decision this cache
    shouldn't presume.

    Capillary pressure derivatives are only the four combinations the
    underlying tables actually define (`Pcow` depends on `Sw`/`So`, `Pcgo` on
    `So`/`Sg`, not a full cross set like relative permeability).

    Capillary-pressure fields are `NaN` for any region with no capillary
    pressure table (`SatFuncTables.capillary_pressure is None`).
    """

    # Relative permeabilities
    water_relative_permeability: CellArray
    """Water relative permeability, $k_{rw}$."""
    oil_relative_permeability: CellArray
    """Oil relative permeability, $k_{ro}$."""
    gas_relative_permeability: CellArray
    """Gas relative permeability, $k_{rg}$."""

    # Relative permeability derivatives w.r.t. water saturation
    dkrw_dsw: CellArray
    """Derivative of water relative permeability w.r.t. water saturation, $dk_{rw}/dS_w$."""
    dkro_dsw: CellArray
    """Derivative of oil relative permeability w.r.t. water saturation, $dk_{ro}/dS_w$."""
    dkrg_dsw: CellArray
    """Derivative of gas relative permeability w.r.t. water saturation, $dk_{rg}/dS_w$."""

    # ... w.r.t. oil saturation
    dkrw_dso: CellArray
    """Derivative of water relative permeability w.r.t. oil saturation, $dk_{rw}/dS_o$."""
    dkro_dso: CellArray
    """Derivative of oil relative permeability w.r.t. oil saturation, $dk_{ro}/dS_o$."""
    dkrg_dso: CellArray
    """Derivative of gas relative permeability w.r.t. oil saturation, $dk_{rg}/dS_o$."""

    # ... w.r.t. gas saturation
    dkrw_dsg: CellArray
    """Derivative of water relative permeability w.r.t. gas saturation, $dk_{rw}/dS_g$."""
    dkro_dsg: CellArray
    """Derivative of oil relative permeability w.r.t. gas saturation, $dk_{ro}/dS_g$."""
    dkrg_dsg: CellArray
    """Derivative of gas relative permeability w.r.t. gas saturation, $dk_{rg}/dS_g$."""

    # Capillary pressures: Pcow = Po - Pw, Pcgo = Pg - Po
    oil_water_capillary_pressure: CellArray
    """Oil-water capillary pressure, $P_{cow} = P_o - P_w$."""
    gas_oil_capillary_pressure: CellArray
    """Gas-oil capillary pressure, $P_{cgo} = P_g - P_o$."""

    # Capillary pressure derivatives
    dpcow_dsw: CellArray
    """Derivative of oil-water capillary pressure w.r.t. water saturation, $dP_{cow}/dS_w$."""
    dpcow_dso: CellArray
    """Derivative of oil-water capillary pressure w.r.t. oil saturation, $dP_{cow}/dS_o$."""
    dpcgo_dso: CellArray
    """Derivative of gas-oil capillary pressure w.r.t. oil saturation, $dP_{cgo}/dS_o$."""
    dpcgo_dsg: CellArray
    """Derivative of gas-oil capillary pressure w.r.t. gas saturation, $dP_{cgo}/dS_g$."""

    saturation_region: IntCellArray
    """The 1-based SATNUM each cell was evaluated against."""


CACHE_NAN_FIELDS: tuple[str, ...] = tuple(
    name for name in SatFuncCache._fields if name != "saturation_region"
)


def make_new_cache(n_cells: int, dtype: npt.DTypeLike) -> SatFuncCache:
    fields = {name: np.full(n_cells, np.nan, dtype=dtype) for name in CACHE_NAN_FIELDS}
    return SatFuncCache(
        **fields,
        saturation_region=np.zeros(n_cells, dtype=np.int32),
    )


def compute_satfunc_cache(
    water_saturation: CellArray,
    oil_saturation: CellArray,
    gas_saturation: CellArray,
    saturation_region: IntCellArray,
    satfunc: SatFunc,
    irreducible_water_saturation: CellArray | None = None,
    residual_oil_saturation_water: CellArray | None = None,
    residual_oil_saturation_gas: CellArray | None = None,
    residual_gas_saturation: CellArray | None = None,
    out: SatFuncCache | None = None,
    dtype: npt.DTypeLike = None,
) -> SatFuncCache:
    """
    Build (or refresh, in place) a `SatFuncCache` from current
    cell saturations.

    Groups cells by `saturation_region` and evaluates each region's
    `SatFuncTables` once per property, vectorised over that
    region's cells.

    **Reuse**: pass a previous call's returned `SatFuncCache` as
    `out` to overwrite its arrays in place; the return value is `out` itself
    in that case. Every field is reset before being written on each call, so
    a region change between calls can't leave a stale value behind.

    :param water_saturation: Shape `(n_cells,)` current `Sw`.
    :param oil_saturation: Shape `(n_cells,)` current `So`.
    :param gas_saturation: Shape `(n_cells,)` current `Sg`.
    :param saturation_region: Shape `(n_cells,)` 1-based SATNUM per
        cell, e.g. `reservoir.regions.saturation_region` (falls back to
        all-ones there when the deck had no `SATNUM` keyword - pass that
        fallback through yourself if calling this directly without going
        through `Regions`).
    :param satfunc: `SatFunc` to
        evaluate against usually `BlackOil.satfunc`.
    :param irreducible_water_saturation: Shape `(n_cells,)` current `swc` -
        dynamic, per-cell hysteresis state from reservoir state, not
        anything this cache computes. `None` (default) omits the keyword
        entirely, so each table falls back to its own value. Forwarded
        as-is to every `evaluate`/`derivatives` call.
    :param residual_oil_saturation_water: Shape `(n_cells,)` current `sorw`.
        Same fallback behaviour as `irreducible_water_saturation`.
    :param residual_oil_saturation_gas: Shape `(n_cells,)` current `sorg`.
        Same fallback behaviour.
    :param residual_gas_saturation: Shape `(n_cells,)` current `sgr`. Same
        fallback behaviour.
    :param out: Previous `SatFuncCache` to overwrite in place, or
        `None` to allocate a new one.
    :return: The populated `SatFuncCache`. `out` itself if given,
        otherwise a newly allocated one. Always returned, never `None`.
    :raises ValueError: If `out` is given but sized for a different cell
        count than `water_saturation`.
    """
    n_cells = len(water_saturation)
    if out is not None:
        if len(out.water_relative_permeability) != n_cells:
            raise ValueError(
                f"`out` was allocated for {len(out.water_relative_permeability)} "
                f"cells; `water_saturation` has {n_cells}."
            )
        cache = out
        for name in CACHE_NAN_FIELDS:
            getattr(cache, name).fill(np.nan)
    else:
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        cache = make_new_cache(n_cells, dtype=dtype)

    cache.saturation_region[:] = saturation_region
    for region in np.unique(saturation_region):
        mask = saturation_region == region
        tables = satfunc[region]

        sw = water_saturation[mask]
        so = oil_saturation[mask]
        sg = gas_saturation[mask]

        # Dynamic per-cell saturation-state kwargs (hysteresis, if tracked).
        # Values come from the caller's reservoir state, not this cache.
        # Only included when the caller actually supplied that array, so a
        # table's own fallback applies otherwise.
        saturation_state_kwargs = {}
        if irreducible_water_saturation is not None:
            saturation_state_kwargs["irreducible_water_saturation"] = irreducible_water_saturation[
                mask
            ]
        if residual_oil_saturation_water is not None:
            saturation_state_kwargs["residual_oil_saturation_water"] = (
                residual_oil_saturation_water[mask]
            )
        if residual_oil_saturation_gas is not None:
            saturation_state_kwargs["residual_oil_saturation_gas"] = residual_oil_saturation_gas[
                mask
            ]
        if residual_gas_saturation is not None:
            saturation_state_kwargs["residual_gas_saturation"] = residual_gas_saturation[mask]

        relperm_table = tables.relative_permeability
        relperm = relperm_table.evaluate(sw, so, sg, **saturation_state_kwargs)
        cache.water_relative_permeability[mask] = relperm["water"]
        cache.oil_relative_permeability[mask] = relperm["oil"]
        cache.gas_relative_permeability[mask] = relperm["gas"]

        relperm_derivatives = relperm_table.derivatives(sw, so, sg, **saturation_state_kwargs)
        cache.dkrw_dsw[mask] = relperm_derivatives["dkrw_dsw"]
        cache.dkro_dsw[mask] = relperm_derivatives["dkro_dsw"]
        cache.dkrg_dsw[mask] = relperm_derivatives["dkrg_dsw"]
        cache.dkrw_dso[mask] = relperm_derivatives["dkrw_dso"]
        cache.dkro_dso[mask] = relperm_derivatives["dkro_dso"]
        cache.dkrg_dso[mask] = relperm_derivatives["dkrg_dso"]
        cache.dkrw_dsg[mask] = relperm_derivatives["dkrw_dsg"]
        cache.dkro_dsg[mask] = relperm_derivatives["dkro_dsg"]
        cache.dkrg_dsg[mask] = relperm_derivatives["dkrg_dsg"]

        capillary_pressure_table = tables.capillary_pressure
        if capillary_pressure_table is not None:
            capillary_pressure = capillary_pressure_table.evaluate(
                sw, so, sg, **saturation_state_kwargs
            )
            cache.oil_water_capillary_pressure[mask] = capillary_pressure["oil_water"]
            cache.gas_oil_capillary_pressure[mask] = capillary_pressure["gas_oil"]

            capillary_pressure_derivatives = capillary_pressure_table.derivatives(
                sw, so, sg, **saturation_state_kwargs
            )
            cache.dpcow_dsw[mask] = capillary_pressure_derivatives["dpcow_dsw"]
            cache.dpcow_dso[mask] = capillary_pressure_derivatives["dpcow_dso"]
            cache.dpcgo_dso[mask] = capillary_pressure_derivatives["dpcgo_dso"]
            cache.dpcgo_dsg[mask] = capillary_pressure_derivatives["dpcgo_dsg"]

    return cache
