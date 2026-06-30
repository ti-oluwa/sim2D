"""Model validation and pre-simulation checks for reservoir simulation."""

import logging
import typing
from enum import Enum

import attrs
import numpy as np

from bores.config import Config
from bores.errors import BORESError
from bores.initialization import check_zero_flow_initialization
from bores.model import FluidProperties, BlackOilModel, RockProperties
from bores.typing import ThreeDimensions

logger = logging.getLogger(__name__)

__all__ = ["ModelValidationError", "ValidationIssue", "ValidationReport", "validate"]


class ModelValidationError(BORESError):
    """Raised by `validate` when one or more fatal issues are detected."""

    def __init__(self, report: "ValidationReport") -> None:
        self.report = report
        msgs = "\n".join(str(error) for error in report.errors)
        super().__init__(
            f"Model validation failed with {len(report.errors)} error(s):\n{msgs}"
        )


class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@attrs.frozen(slots=True)
class ValidationIssue:
    """
    A single diagnostic issue produced during model validation.

    :param check: Identifier of the check that produced this issue.
    :param severity: Severity level of the issue.
    :param message: Human-readable description of the issue.
    :param detail: Optional additional context or remediation hint.
    :param corrected: True when the validator silently repaired the issue in-place.
    """

    check: str
    severity: Severity
    message: str
    detail: typing.Optional[str] = None
    corrected: bool = False

    def __str__(self) -> str:
        tag = "[CORRECTED] " if self.corrected else ""
        base = f"[{self.severity.value}] {self.check}: {tag}{self.message}"
        if self.detail:
            base += f"\n  -> {self.detail}"
        return base


@attrs.frozen
class ValidationReport:
    """
    Aggregated model validation result (returned by `validate`).

    :param issues: All issues collected across every check, in emission order.
    """

    issues: typing.List[ValidationIssue] = attrs.field(factory=list)
    """All issues collected across every check, in emission order."""

    @property
    def errors(self) -> typing.List[ValidationIssue]:
        """All issues with ERROR severity."""
        return [issue for issue in self.issues if issue.severity == Severity.ERROR]

    @property
    def warnings(self) -> typing.List[ValidationIssue]:
        """All issues with WARNING severity."""
        return [issue for issue in self.issues if issue.severity == Severity.WARNING]

    @property
    def infos(self) -> typing.List[ValidationIssue]:
        """All issues with INFO severity (includes corrections)."""
        return [issue for issue in self.issues if issue.severity == Severity.INFO]

    @property
    def passed(self) -> bool:
        """True when no ERROR-severity issues were found."""
        return len(self.errors) == 0

    def _add(
        self,
        check: str,
        severity: Severity,
        message: str,
        detail: typing.Optional[str] = None,
        corrected: bool = False,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                check=check,
                severity=severity,
                message=message,
                detail=detail,
                corrected=corrected,
            )
        )

    def info(
        self, check: str, message: str, detail: typing.Optional[str] = None
    ) -> None:
        """Record an informational issue."""
        self._add(check, Severity.INFO, message, detail)

    def warn(
        self, check: str, message: str, detail: typing.Optional[str] = None
    ) -> None:
        """Record a warning-level issue."""
        self._add(check, Severity.WARNING, message, detail)

    def error(
        self, check: str, message: str, detail: typing.Optional[str] = None
    ) -> None:
        """Record a fatal error-level issue."""
        self._add(check, Severity.ERROR, message, detail)

    def corrected(
        self, check: str, message: str, detail: typing.Optional[str] = None
    ) -> None:
        """Record a corrected issue (INFO severity with corrected=True)."""
        self._add(check, Severity.INFO, message, detail, corrected=True)

    def __str__(self) -> str:
        lines = ["=" * 70, "  MODEL VALIDATION REPORT", "=" * 70]
        if not self.issues:
            lines.append("  All checks passed - no issues found.")
        else:
            for issue in self.issues:
                lines.append(str(issue))
        lines += [
            "=" * 70,
            f"  Summary: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s), "
            f"{len(self.infos)} info/correction(s)",
            "=" * 70,
        ]
        return "\n".join(lines)

    def log(self, _logger: typing.Optional[logging.Logger] = None, /) -> None:
        """
        Emit every issue through *_logger*.

        :param _logger: Logger to use. Defaults to this module's logger.
        """
        log = _logger or logger
        for issue in self.issues:
            if issue.severity == Severity.ERROR:
                log.error(str(issue))
            elif issue.severity == Severity.WARNING:
                log.warning(str(issue))
            else:
                log.info(str(issue))


def validate(
    model: BlackOilModel[ThreeDimensions],
    config: Config,
    *,
    correct_inplace: bool = True,
    raise_on_error: bool = True,
    zero_flow_tolerance: typing.Optional[float] = None,
    emit_log: bool = True,
) -> ValidationReport:
    """
    Run a comprehensive pre-simulation validation on *model* and *config*.

    Checks performed in order include:

    1.  Saturation sum and per-phase bounds
    2.  Porosity bounds and distribution
    3.  Net-to-gross bounds
    4.  Permeability sign, zero-K cells, Kv/Kh anisotropy
    5.  Pressure vs PVT table coverage
    6.  Vertical pressure gradient plausibility
    7.  PVT table monotonicity (Bo, Rs, Bg)
    8.  Zero-flow equilibrium (contextual - not a raw pass/fail)
    9.  Well placement and perforation sanity
    10. Fluid contact ordering (GOC above OWC)
    11. Rock compressibility plausibility
    12. Bubble-point vs initial pressure
    13. Transmissibility magnitude and condition
    14. Pore-volume distribution (micro-pore and heterogeneity detection)
    15. Capillary pressure sign convention


    Validation philosophy:

    - **Correct if safe**: small, well-understood numerical artefacts (floating-point
      saturation drift, near-zero negative saturations) are silently repaired and
      reported as INFO.
    - **Warn if suspicious**: physics that may degrade accuracy but will not crash
      the simulator (extreme anisotropy ratios, pressure near PVT table limits,
      loose zero-flow tolerance).
    - **Raise if fatal**: conditions that guarantee wrong physics or a solver crash
      (saturation sums deviating beyond 1%, pressure outside PVT coverage, wells
      placed in non-existent or zero-porosity cells).

    :param model: A 3-D `BlackOilModel` instance.
    :param config: The associated `Config`.
    :param correct_inplace: When True, small correctable issues are fixed directly
        on the model arrays. When False, they are reported as warnings or errors.
    :param raise_on_error: When True, a `ModelValidationError` is raised after
        all checks complete if any fatal errors were found.
    :param zero_flow_tolerance: Override the tolerance for the zero-flow equilibrium
        check. When None the value is selected automatically from grid scale.
    :param emit_log: When True, each issue is emitted through `logging`.
    :return: Full `ValidationReport`.
    :raises ModelValidationError: If `raise_on_error=True` and any ERROR-severity
        issues were found.
    """
    report = ValidationReport()
    fluid_properties = model.fluid_properties
    rock_properties = model.rock_properties
    cell_dimension = model.cell_dimension

    _validate_saturations(
        fluid_properties,
        rock_properties,
        report,
        correct_inplace=correct_inplace,
    )
    _validate_porosity(rock_properties, report)
    _validate_net_to_gross(rock_properties, report)
    _validate_permeability(rock_properties, report)
    _validate_pressure_pvt_coverage(fluid_properties, config, report)
    _validate_pressure_gradient(model, report)
    _validate_pvt_monotonicity(config, report)
    _validate_zero_flow(
        model,
        config,
        report,
        cell_dimension=cell_dimension,
        tolerance=zero_flow_tolerance,
    )
    _validate_wells(config, model, report)
    _validate_fluid_contacts(model, report)
    _validate_rock_compressibility(rock_properties, report)
    _validate_bubble_point(fluid_properties, config, report)
    _validate_transmissibility(model, report)
    _validate_pore_volume_distribution(model, report)
    _validate_capillary_pressure_sign(config, rock_properties, report)

    if emit_log:
        report.log()

    if raise_on_error and not report.passed:
        raise ModelValidationError(report)
    return report


def _validate_saturations(
    fluid_properties: FluidProperties,
    rock_properties: RockProperties,
    report: ValidationReport,
    *,
    correct_inplace: bool,
) -> None:
    """
    Verify phase saturations are non-negative, bounded by [0, 1], and sum to unity.

    The hard threshold for auto-correction is 1 % (0.01). Beyond that the issue is
    fatal because it indicates a broken initialisation routine, not floating-point
    noise. Near-zero negative saturations (magnitude < 1e-9) are clamped to zero
    when *correct_inplace* is True. Also validates that initial Sw is at or above
    the connate water saturation from rock properties.

    :param fluid_properties: Fluid properties containing the saturation grids.
    :param rock_properties: Rock properties containing connate water saturation.
    :param report: Report to append issues to.
    :param correct_inplace: Whether to repair small artefacts directly on the arrays.
    """
    check_sum = "saturation_sum"

    water_saturation = fluid_properties.water_saturation_grid
    oil_saturation = fluid_properties.oil_saturation_grid
    gas_saturation = fluid_properties.gas_saturation_grid
    total_saturation = water_saturation + oil_saturation + gas_saturation

    n_negative_water = int(np.sum(water_saturation < -1e-9))
    n_negative_oil = int(np.sum(oil_saturation < -1e-9))
    n_negative_gas = int(np.sum(gas_saturation < -1e-9))
    n_negative_total = n_negative_water + n_negative_oil + n_negative_gas

    if n_negative_total > 0:
        if correct_inplace:
            np.maximum(water_saturation, 0.0, out=water_saturation)
            np.maximum(oil_saturation, 0.0, out=oil_saturation)
            np.maximum(gas_saturation, 0.0, out=gas_saturation)
            total_saturation = water_saturation + oil_saturation + gas_saturation
            report.corrected(
                check_sum,
                f"Clamped {n_negative_total} cell(s) with negative saturation to 0.0.",
                f"Sw<0: {n_negative_water}  So<0: {n_negative_oil}  Sg<0: {n_negative_gas}",
            )
        else:
            report.error(
                check_sum,
                f"{n_negative_total} cell(s) have negative saturation.",
                f"Sw<0: {n_negative_water}  So<0: {n_negative_oil}  Sg<0: {n_negative_gas} - "
                "all phase saturations must be >= 0.",
            )

    absolute_deviation = np.abs(total_saturation - 1.0)
    max_deviation = float(np.max(absolute_deviation))
    n_cells_off = int(np.sum(absolute_deviation > 1e-6))

    if max_deviation > 0.01:
        report.error(
            check_sum,
            f"Saturation sum deviates by up to {max_deviation:.6f} in {n_cells_off} cell(s).",
            "Sw + So + Sg must equal 1. Re-check initialisation. Deviation exceeds the 1% hard limit.",
        )
    elif max_deviation > 1e-6:
        if correct_inplace:
            residual = 1.0 - total_saturation
            np.maximum(oil_saturation + residual, 0.0, out=oil_saturation)
            report.corrected(
                check_sum,
                f"Normalised saturation sum in {n_cells_off} cell(s) (max deviation {max_deviation:.2e}).",
                "Residual absorbed into oil saturation.",
            )
        else:
            report.warn(
                check_sum,
                f"Saturation sum deviates by up to {max_deviation:.2e} in {n_cells_off} cell(s).",
                "Pass correct_inplace=True to auto-normalise.",
            )
    else:
        report.info(
            check_sum,
            "Saturation sum check passed.",
            f"Max deviation: {max_deviation:.2e}",
        )

    check_bounds = "saturation_bounds"
    n_over_water = int(np.sum(water_saturation > 1.0 + 1e-9))
    n_over_oil = int(np.sum(oil_saturation > 1.0 + 1e-9))
    n_over_gas = int(np.sum(gas_saturation > 1.0 + 1e-9))

    if n_over_water + n_over_oil + n_over_gas > 0:
        report.error(
            check_bounds,
            f"Phase saturation(s) exceed 1.0: Sw>{1}: {n_over_water}  "
            f"So>{1}: {n_over_oil}  Sg>{1}: {n_over_gas}.",
        )
    else:
        report.info(check_bounds, "All phase saturations are within [0, 1].")

    check_connate = "connate_saturation_consistency"
    connate_water_saturation = rock_properties.connate_water_saturation_grid
    n_below_connate = int(np.sum(water_saturation < connate_water_saturation - 1e-6))
    if n_below_connate > 0:
        report.warn(
            check_connate,
            f"{n_below_connate} cell(s) have initial Sw below connate water saturation (Swc).",
            "This may indicate inconsistent initialisation or saturation end-point tables.",
        )
    else:
        report.info(check_connate, "Initial Sw >= Swc everywhere.")


def _validate_porosity(
    rock_properties: RockProperties, report: ValidationReport
) -> None:
    """Validate porosity values against physical bounds and typical reservoir ranges.

    Negative values and values > 1 are fatal. Values above 0.5 are suspicious for
    conventional reservoirs. Values below 0.01 may indicate the grid is in percent
    rather than fraction.

    :param rock_properties: Rock properties containing the porosity grid.
    :param report: Report to append issues to.
    """
    check = "porosity"
    porosity_grid = rock_properties.porosity_grid

    n_negative = int(np.sum(porosity_grid < 0.0))
    n_above_one = int(np.sum(porosity_grid > 1.0))
    n_zero = int(np.sum(porosity_grid == 0.0))
    total_cells = porosity_grid.size

    if n_negative > 0:
        report.error(check, f"{n_negative} cell(s) have negative porosity.")
    if n_above_one > 0:
        report.error(
            check, f"{n_above_one} cell(s) have porosity > 1.0 (non-physical)."
        )

    if n_negative == 0 and n_above_one == 0:
        active_porosity = porosity_grid[porosity_grid > 0.0]
        if active_porosity.size > 0:
            mean_porosity = float(np.mean(active_porosity))
            p10_porosity = float(np.percentile(active_porosity, 10))
            p90_porosity = float(np.percentile(active_porosity, 90))
            max_porosity = float(np.max(active_porosity))
        else:
            mean_porosity = p10_porosity = p90_porosity = max_porosity = 0.0

        report.info(
            check,
            f"Porosity valid. Mean φ = {mean_porosity:.4f}  "
            f"P10/P90 = {p10_porosity:.4f}/{p90_porosity:.4f}  "
            f"Zero-porosity cells: {n_zero}/{total_cells}.",
        )
        if max_porosity > 0.5:
            report.warn(
                check,
                f"Maximum porosity {max_porosity:.4f} exceeds 0.5. Unusual for conventional reservoirs.",
                "Verify units and data source. Acceptable for vuggy carbonates.",
            )
        if max_porosity < 0.01:
            report.warn(
                check,
                f"Maximum porosity {max_porosity:.4f}. Check whether values are in percent rather than fraction.",
            )


def _validate_net_to_gross(
    rock_properties: RockProperties, report: ValidationReport
) -> None:
    """Check that net-to-gross values lie within [0, 1].

    Also warns when more than half the grid has NTG = 0, which is likely a
    data import error.

    :param rock_properties: Rock properties containing the net-to-gross grid.
    :param report: Report to append issues to.
    """
    check = "net_to_gross"
    net_to_gross_grid = rock_properties.net_to_gross_grid

    n_below_zero = int(np.sum(net_to_gross_grid < 0.0))
    n_above_one = int(np.sum(net_to_gross_grid > 1.0))

    if n_below_zero + n_above_one > 0:
        report.error(
            check, f"NTG out of [0, 1]: {n_below_zero} below 0, {n_above_one} above 1."
        )
        return

    ntg_mean = float(np.mean(net_to_gross_grid))
    n_zero_ntg = int(np.sum(net_to_gross_grid == 0.0))
    fraction_zero = n_zero_ntg / net_to_gross_grid.size

    report.info(
        check, f"NTG valid. Mean = {ntg_mean:.4f}  Zero-NTG cells: {n_zero_ntg}."
    )

    if fraction_zero > 0.5:
        report.warn(
            check,
            f"{fraction_zero * 100:.1f} % of cells have NTG = 0.0.",
            "More than half the grid is non-reservoir. Verify NTG import.",
        )


def _validate_permeability(
    rock_properties: RockProperties, report: ValidationReport
) -> None:
    """
    Validate absolute permeability across all three directions.

    Checks for negative values (fatal), zero horizontal permeability cells (isolated
    cells warning), Kv/Kh anisotropy outside the typical 0.001-10 range, and absolute
    magnitude outliers that suggest unit errors.

    :param rock_properties: Rock properties containing absolute permeability grids.
    :param report: Report to append issues to.
    """
    check = "permeability"
    permeability_x = rock_properties.absolute_permeability.x
    permeability_y = rock_properties.absolute_permeability.y
    permeability_z = rock_properties.absolute_permeability.z

    n_negative_total = (
        int(np.sum(permeability_x < 0.0))
        + int(np.sum(permeability_y < 0.0))
        + int(np.sum(permeability_z < 0.0))
    )
    if n_negative_total > 0:
        report.error(
            check,
            f"Negative permeability detected in {n_negative_total} component/cell pair(s).",
            "Permeability must be non-negative everywhere.",
        )
        return

    n_zero_horizontal = int(np.sum((permeability_x == 0.0) | (permeability_y == 0.0)))
    if n_zero_horizontal > 0:
        report.warn(
            check,
            f"{n_zero_horizontal} cell(s) have zero horizontal permeability. Those cells are numerically isolated.",
        )

    horizontal_permeability = 0.5 * (permeability_x + permeability_y)
    active_mask = horizontal_permeability > 0.0

    if not np.any(active_mask):
        report.warn(check, "All cells have zero horizontal permeability.")
        return

    anisotropy_ratio = (
        permeability_z[active_mask] / horizontal_permeability[active_mask]
    )
    anisotropy_max = float(np.max(anisotropy_ratio))
    anisotropy_min = float(np.min(anisotropy_ratio))
    anisotropy_median = float(np.median(anisotropy_ratio))

    if anisotropy_max > 10.0:
        report.warn(
            check,
            f"Kv/Kh up to {anisotropy_max:.2f}. Unusually high vertical permeability.",
            f"Typical range for clastics: 0.01-1.0; carbonates: up to 1.0. "
            f"Median Kv/Kh = {anisotropy_median:.3f}.",
        )
    elif anisotropy_min < 1e-4:
        report.warn(
            check,
            f"Kv/Kh as low as {anisotropy_min:.2e}. Near-sealing vertical barriers present.",
            "This can suppress gravity segregation and cause numerical difficulties with gravity terms.",
        )
    else:
        report.info(
            check,
            f"Kv/Kh range [{anisotropy_min:.3e}, {anisotropy_max:.3e}]  median = {anisotropy_median:.3f}.",
        )

    active_horizontal_permeability = horizontal_permeability[active_mask]
    horizontal_permeability_max = float(np.max(active_horizontal_permeability))
    positive_values = active_horizontal_permeability[
        active_horizontal_permeability > 0.0
    ]
    horizontal_permeability_min = (
        float(np.min(positive_values)) if positive_values.size > 0 else 0.0
    )

    if horizontal_permeability_max > 1e5:
        report.warn(
            check,
            f"Maximum horizontal permeability {horizontal_permeability_max:.2e} mD is very high (> 100 Darcy).",
            "Verify units - expected milliDarcy. Values > 100 Darcy are unusual except for fracture apertures.",
        )
    if 0.0 < horizontal_permeability_min < 1e-6:
        report.warn(
            check,
            f"Minimum non-zero horizontal permeability {horizontal_permeability_min:.2e} mD is effectively zero.",
        )


def _validate_pressure_pvt_coverage(
    fluid_properties: FluidProperties,
    config: Config,
    report: ValidationReport,
) -> None:
    """
    Verify that the initial pressure field lies within PVT table pressure bounds.

    Extrapolation outside PVT table bounds produces unphysical fluid properties and is
    treated as a fatal error. A 1% guard band triggers a warning.

    :param fluid_properties: Fluid properties containing the initial pressure grid.
    :param config: Simulation config carrying the PVT tables.
    :param report: Report to append issues to.
    """
    check = "pressure_pvt_coverage"

    if config.pvt_tables is None:
        report.info(check, "No PVT tables attached to config. Skipping coverage check.")
        return

    pressure_grid = fluid_properties.pressure_grid
    min_pressure = float(np.min(pressure_grid))
    max_pressure = float(np.max(pressure_grid))

    table_min_pressure: typing.Optional[float] = None
    table_max_pressure: typing.Optional[float] = None

    for phase_table in (
        config.pvt_tables.oil,
        config.pvt_tables.gas,
        config.pvt_tables.water,
    ):
        if phase_table is None:
            continue
        pressures = phase_table._data.pressures
        if pressures is not None and len(pressures) >= 2:
            table_min_pressure = float(np.min(pressures))
            table_max_pressure = float(np.max(pressures))
            break

    if table_min_pressure is None or table_max_pressure is None:
        report.info(check, "Cannot inspect PVT pressure axis. Skipping range check.")
        return

    guard_band = 0.01 * (table_max_pressure - table_min_pressure)

    if min_pressure < table_min_pressure - guard_band:
        report.error(
            check,
            f"Minimum grid pressure {min_pressure:.1f} psi is below PVT lower limit {table_min_pressure:.1f} psi.",
            "PVT will extrapolate - results are unphysical. Extend table or fix datum pressure.",
        )
    elif min_pressure < table_min_pressure + guard_band:
        report.warn(
            check,
            f"Minimum grid pressure {min_pressure:.1f} psi is within the 1% guard band "
            f"of the PVT lower limit {table_min_pressure:.1f} psi.",
        )

    if max_pressure > table_max_pressure + guard_band:
        report.error(
            check,
            f"Maximum grid pressure {max_pressure:.1f} psi exceeds PVT upper limit {table_max_pressure:.1f} psi.",
            "PVT will extrapolate - results are unphysical. Extend table or fix datum pressure.",
        )
    elif max_pressure > table_max_pressure - guard_band:
        report.warn(
            check,
            f"Maximum grid pressure {max_pressure:.1f} psi is within the 1% guard band "
            f"of the PVT upper limit {table_max_pressure:.1f} psi.",
        )

    if min_pressure >= table_min_pressure and max_pressure <= table_max_pressure:
        report.info(
            check,
            f"Grid pressure [{min_pressure:.1f}, {max_pressure:.1f}] psi fully within "
            f"PVT table [{table_min_pressure:.1f}, {table_max_pressure:.1f}] psi.",
        )


def _validate_pressure_gradient(model: BlackOilModel, report: ValidationReport) -> None:
    """
    Assess the vertical pressure gradient for gross physical plausibility.

    Expected gradients (psi/ft): gas 0.01-0.10, oil 0.25-0.37, water ~0.433.
    Gradients > 0.55 psi/ft suggest a datum error. Essentially flat gradients indicate
    pressure was initialised without hydrostatics. Also detects layer-to-layer pressure
    inversions that reveal depth-index convention mismatches.

    :param model: Reservoir model providing pressure and thickness grids.
    :param report: Report to append issues to.
    """
    check = "pressure_gradient"
    pressure_grid = model.fluid_properties.pressure_grid
    thickness_grid = model.thickness_grid
    nz: int = pressure_grid.shape[2]

    if nz < 2:
        report.info(check, "Single-layer model. Vertical gradient check skipped.")
        return

    top_pressure = float(np.mean(pressure_grid[:, :, 0]))
    bottom_pressure = float(np.mean(pressure_grid[:, :, nz - 1]))
    total_thickness = float(np.sum(thickness_grid[0, 0, :]))

    if total_thickness <= 0.0:
        report.warn(check, "Total reservoir thickness ≤ 0 ft. Gradient check skipped.")
        return

    vertical_gradient = abs(bottom_pressure - top_pressure) / total_thickness
    if vertical_gradient > 0.55:
        report.warn(
            check,
            f"Vertical pressure gradient {vertical_gradient:.4f} psi/ft exceeds the fresh-water gradient (0.433 psi/ft).",
            "Possible datum error or high-density fluid. Verify fluid density and reference depth.",
        )
    elif vertical_gradient < 1e-5:
        report.warn(
            check,
            f"Vertical pressure gradient {vertical_gradient:.2e} psi/ft is essentially zero.",
            "Pressure appears to have been initialised flat (no hydrostatics). "
            "This will generate large spurious initial fluxes.",
        )
    else:
        report.info(
            check,
            f"Vertical pressure gradient {vertical_gradient:.4f} psi/ft is plausible.",
        )

    layer_mean_pressures = np.mean(pressure_grid, axis=(0, 1))
    n_inversions = int(np.sum(np.diff(layer_mean_pressures) < 0))
    if n_inversions > nz // 2:
        report.warn(
            check,
            f"Pressure decreases downward in {n_inversions}/{nz - 1} layer transitions.",
            "Verify that layer indexing convention (shallow=0, deep=nz-1) matches your data.",
        )


def _validate_pvt_monotonicity(config: Config, report: ValidationReport) -> None:
    """
    Check PVT table monotonicity for Bo and Rs (increasing) and Bg (decreasing).

    Non-monotone tables produce interpolation artefacts and can cause the Newton solver
    to diverge. Violations are emitted as warnings as the table may still be physically
    correct but numerically risky.

    :param config: Simulation config carrying the PVT tables.
    :param report: Report to append issues to.
    """
    check = "pvt_monotonicity"

    if config.pvt_tables is None:
        return

    issues_found = False
    property_checks: typing.List[typing.Tuple[str, str, str, str]] = [
        ("oil", "formation_volume_factor_table", "Bo", "increasing"),
        ("oil", "solution_gor_table", "Rs", "increasing"),
        ("gas", "formation_volume_factor_table", "Bg", "decreasing"),
    ]

    for phase_name, table_attribute, label, direction in property_checks:
        phase_table = getattr(config.pvt_tables, phase_name)
        if phase_table is None:
            continue
        table_array: typing.Optional[np.ndarray] = getattr(
            phase_table._data, table_attribute, None
        )
        if table_array is None:
            continue

        differences = np.diff(np.asarray(table_array).ravel())

        if direction == "increasing":
            n_violations = int(np.sum(differences < -1e-8))
            if n_violations > 0:
                report.warn(
                    check,
                    f"PVT {label} ({phase_name}) is not strictly monotonically increasing "
                    f"({n_violations} decreasing step(s)).",
                    "Non-monotone tables cause interpolation artefacts. Smoothen or re-tabulate.",
                )
                issues_found = True
        else:
            n_violations = int(np.sum(differences > 1e-8))
            if n_violations > 0:
                report.warn(
                    check,
                    f"PVT {label} ({phase_name}) is not strictly monotonically decreasing "
                    f"({n_violations} increasing step(s)).",
                    "Non-monotone tables cause interpolation artefacts. Smoothen or re-tabulate.",
                )
                issues_found = True

    if not issues_found:
        report.info(check, "PVT Bo, Rs, Bg monotonicity checks passed.")


def _recommend_zero_flow_tolerance(cell_dimension: typing.Tuple[float, float]) -> float:
    """
    Return a grid-scale-aware zero-flow tolerance in day⁻¹.

    The hydrostatic residual flux scales approximately with cell size. For cells
    smaller than 100 ft the standard 1e-6 day⁻¹ is appropriate; for SPE1-class
    1000 ft cells we relax to ~1e-5 day⁻¹.

    :param cell_dimension: Physical cell dimensions (dx, dy) in feet.
    :param nz: Number of vertical layers (reserved for future depth-dependent scaling).
    :return: Recommended tolerance in day⁻¹.
    """
    dx, dy = cell_dimension
    scale_factor = max(dx, dy) / 100.0
    tolerance = max(1e-6 * scale_factor, 1e-6)
    return min(tolerance, 5e-5)


def _validate_zero_flow(
    model: BlackOilModel,
    config: Config,
    report: ValidationReport,
    *,
    cell_dimension: typing.Tuple[float, float],
    tolerance: typing.Optional[float],
) -> None:
    """
    Check that the initial state satisfies gravitational and capillary equilibrium.

    A reservoir at rest should have zero net inter-cell flux before wells open.
    Violations are contextualised: oil-only residuals at low excess factors are
    consistent with hydrostatic gravity discretisation artefacts and trigger a warning
    rather than an error. Large multi-phase violations indicate genuine
    pressure-saturation inconsistency and are fatal.

    :param model: Reservoir model to check.
    :param config: Simulation config.
    :param report: Report to append issues to.
    :param cell_dimension: Physical cell dimensions (dx, dy) in feet.
    :param tolerance: User-supplied tolerance override, or None for automatic selection.
    """
    check = "zero_flow_equilibrium"

    if tolerance is not None:
        tolerance_source = "user-specified"
    else:
        tolerance = _recommend_zero_flow_tolerance(cell_dimension)
        tolerance_source = "auto [grid-scale adaptive]"

    report.info(
        check, f"Zero-flow tolerance: {tolerance:.2e} day⁻¹ ({tolerance_source})."
    )

    try:
        face_transmissibilities = model.build_face_transmissibilities()
        elevation_grid = model.build_elevation_grid(
            apply_dip=not config.disable_structural_dip
        )
        zero_flow_result = check_zero_flow_initialization(
            fluid_properties=model.fluid_properties,
            rock_properties=model.rock_properties,
            face_transmissibilities=face_transmissibilities,
            elevation_grid=elevation_grid,
            config=config,
            pore_volume_grid=model.pore_volume_grid,
            relative_tolerance=tolerance,
            max_reported_violations=5,
        )
    except Exception as exc:
        report.warn(check, f"Zero-flow check could not execute: {exc}")
        return

    max_relative_flux = zero_flow_result.max_relative_flux
    if zero_flow_result.passed:
        report.info(
            check,
            f"Zero-flow equilibrium satisfied. Max relative flux = {max_relative_flux:.3e} day⁻¹  "
            f"({zero_flow_result.cells_checked} active cells checked).",
        )
        return

    violations = zero_flow_result.violations
    excess_factor = max_relative_flux / tolerance

    oil_dominated = all(
        abs(violation.net_water_flux) < 1e-2 * abs(violation.net_oil_flux or 1.0)
        and abs(violation.net_gas_flux) < 1e-2 * abs(violation.net_oil_flux or 1.0)
        for violation in violations
        if violation.net_oil_flux != 0.0
    )

    summary = (
        f"{zero_flow_result.violation_count}/{zero_flow_result.cells_checked} cells exceed "
        f"tolerance {tolerance:.2e} day⁻¹ by up to {excess_factor:.1f}x. "
        f"Max relative flux = {max_relative_flux:.3e} day⁻¹ at cell {zero_flow_result.worst_cell}."
    )

    if oil_dominated and excess_factor < 20.0:
        report.warn(
            check,
            f"Zero-flow: {summary} Violations are oil-phase only, consistent with "
            "hydrostatic gravity discretisation residual, not a physics inconsistency.",
            f"Recommended tolerance for this grid: >= {max_relative_flux * 2:.2e} day⁻¹. "
            "Pass `zero_flow_tolerance` to `validate(...)` to suppress.",
        )
    elif excess_factor < 5.0:
        report.warn(
            check,
            f"Zero-flow: {summary} Small multi-phase imbalance - likely minor "
            "pressure-density discretisation mismatch.",
        )
    else:
        report.error(
            check,
            f"Zero-flow: {summary}",
            "Likely cause: pressure and saturation/density grids are inconsistent. "
            "Re-check datum pressure, fluid contacts, or capillary pressure end-points.",
        )

    for violation in violations[:5]:
        logger.debug(
            "  `zero_flow_violation`  cell=%s  rel_flux=%.3e  "
            "qo=%.3e  qw=%.3e  qg=%.3e  (lbm/day)",
            violation.cell,
            violation.relative_flux,
            violation.net_oil_flux,
            violation.net_water_flux,
            violation.net_gas_flux,
        )


def _validate_wells(
    config: Config,
    model: BlackOilModel,
    report: ValidationReport,
) -> None:
    """
    Verify that all well perforations fall inside the grid and in cells with non-zero pore volume.

    Also flags wells whose perforations are entirely in zero-porosity cells, which is a
    guaranteed zero-rate situation that is almost always a data error.

    :param config: Simulation config carrying the wells definition.
    :param model: Reservoir model providing grid shape and porosity.
    :param report: Report to append issues to.
    """
    check = "well_placement"

    if config.wells is None or not config.wells.exists():
        report.info(check, "No wells configured. Skipping well placement check.")
        return

    nx, ny, nz = model.grid_shape
    porosity_grid = model.rock_properties.porosity_grid
    all_wells = list(config.wells.injection_wells) + list(config.wells.production_wells)

    placement_errors: typing.List[str] = []
    dead_well_names: typing.List[str] = []

    for well in all_wells:
        well_active_pore_volume = 0.0

        for interval in well.perforating_intervals:
            top_cell, bottom_cell = interval
            for cell_index in (top_cell, bottom_cell):
                ci, cj, ck = cell_index
                if not (0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz):
                    placement_errors.append(
                        f"Well '{well.name}': perforation ({ci},{cj},{ck}) is outside "
                        f"grid ({nx}x{ny}x{nz})."
                    )
                else:
                    cell_porosity = float(porosity_grid[ci, cj, ck])
                    if cell_porosity == 0.0:
                        placement_errors.append(
                            f"Well '{well.name}': perforation ({ci},{cj},{ck}) is in a zero-porosity cell."
                        )
                    else:
                        well_active_pore_volume += cell_porosity

        if well_active_pore_volume == 0.0 and not any(
            well.name in msg for msg in placement_errors
        ):
            dead_well_names.append(well.name)

    for error_message in placement_errors:
        report.error(check, error_message)

    for well_name in dead_well_names:
        report.warn(
            check,
            f"Well '{well_name}' has all perforations in non-reservoir (zero-porosity) cells.",
            "This well will produce/inject at zero rate. Verify perforation depths.",
        )

    if not placement_errors and not dead_well_names:
        report.info(
            check, f"All {len(all_wells)} well(s) have valid perforation locations."
        )


def _validate_fluid_contacts(model: BlackOilModel, report: ValidationReport) -> None:
    """
    Check macro-scale fluid contact ordering via saturation-weighted depth centroids.

    Physical expectation (depth increasing downward):
        gas centroid < oil centroid < water centroid.

    A deviation of more than 5 ft flags a potential GOC or OWC placement error.
    The check is skipped when no free gas is present (solution-gas only systems).

    :param model: Reservoir model providing saturation and depth grids.
    :param report: Report to append issues to.
    """
    check = "fluid_contacts"

    try:
        depth_grid = model.build_depth_grid(apply_dip=False)
    except Exception:
        report.info(check, "Depth grid unavailable. Contact ordering check skipped.")
        return

    gas_saturation = model.fluid_properties.gas_saturation_grid
    water_saturation = model.fluid_properties.water_saturation_grid
    oil_saturation = 1.0 - gas_saturation - water_saturation

    if float(np.max(gas_saturation)) <= 0.01:
        report.info(check, "No free gas saturation. GOC contact check skipped.")
        return

    total_gas_saturation = float(np.sum(gas_saturation))
    total_oil_saturation = float(np.sum(oil_saturation))
    total_water_saturation = float(np.sum(water_saturation))

    if total_gas_saturation < 1e-9 or total_oil_saturation < 1e-9:
        return

    gas_depth_centroid = (
        float(np.sum(gas_saturation * depth_grid)) / total_gas_saturation
    )
    oil_depth_centroid = (
        float(np.sum(oil_saturation * depth_grid)) / total_oil_saturation
    )
    water_depth_centroid = (
        float(np.sum(water_saturation * depth_grid)) / total_water_saturation
        if total_water_saturation > 1e-9
        else None
    )

    detail = (
        f"Gas centroid: {gas_depth_centroid:.1f} ft  "
        f"Oil centroid: {oil_depth_centroid:.1f} ft"
    )
    if water_depth_centroid is not None:
        detail += f"  Water centroid: {water_depth_centroid:.1f} ft"

    goc_inverted = gas_depth_centroid > oil_depth_centroid + 5.0
    owc_inverted = (
        water_depth_centroid is not None
        and oil_depth_centroid > water_depth_centroid + 5.0
    )

    if goc_inverted:
        report.warn(
            check,
            f"Gas centroid ({gas_depth_centroid:.1f} ft) is deeper than oil centroid "
            f"({oil_depth_centroid:.1f} ft). GOC may be inverted.",
            detail,
        )
    elif owc_inverted:
        report.warn(
            check,
            f"Oil centroid ({oil_depth_centroid:.1f} ft) is deeper than water centroid "
            f"({water_depth_centroid:.1f} ft). OWC may be inverted.",
            detail,
        )
    else:
        report.info(check, "Fluid contact ordering is consistent.", detail)


def _validate_rock_compressibility(
    rock_properties: RockProperties, report: ValidationReport
) -> None:
    """
    Check rock compressibility against physically reasonable ranges.

    Typical reservoir rocks: 3x10⁻⁶ to 1x10⁻⁵ psi⁻¹. Highly compactible sands can
    reach ~1x10⁻⁴ psi⁻¹. Values > 1x10⁻³ psi⁻¹ are almost certainly data errors.

    :param rock_properties: Rock properties containing the compressibility scalar.
    :param report: Report to append issues to.
    """
    check = "rock_compressibility"
    rock_compressibility = float(rock_properties.compressibility)

    if rock_compressibility < 0.0:
        report.error(
            check,
            f"Rock compressibility is negative: {rock_compressibility:.3e} psi⁻¹.",
        )
    elif rock_compressibility == 0.0:
        report.warn(
            check, "Rock compressibility is zero. Reservoir is perfectly rigid."
        )
    elif rock_compressibility > 1e-3:
        report.warn(
            check,
            f"Rock compressibility {rock_compressibility:.3e} psi⁻¹ is very high (typical: 3e-6 - 1e-5 psi⁻¹).",
            "May indicate a compaction drive model or a data entry error (check units).",
        )
    elif rock_compressibility < 1e-7:
        report.warn(
            check,
            f"Rock compressibility {rock_compressibility:.3e} psi⁻¹ is very low for a clastic reservoir.",
            "Verify source - may require a grain/bulk modulus conversion.",
        )
    else:
        report.info(
            check,
            f"Rock compressibility {rock_compressibility:.3e} psi⁻¹ is within typical range.",
        )


def _validate_bubble_point(
    fluid_properties: FluidProperties, config: Config, report: ValidationReport
) -> None:
    """
    Classify the initial reservoir state and check bubble-point PVT coverage.

    Classifies cells as saturated (P < Pb) or undersaturated (P >= Pb). Flags cells
    where Pb - P > 500 psi, which guarantees immediate gas liberation at t=0 and
    causes time-step rejection loops. Also verifies bubble-point pressures lie within
    the oil PVT table bounds.

    :param fluid_properties: Fluid properties containing pressure and bubble-point grids.
    :param config: Simulation config carrying the PVT tables.
    :param report: Report to append issues to.
    """
    check = "bubble_point_vs_pressure"
    bubble_point_pressure = fluid_properties.oil_bubble_point_pressure_grid
    pressure_grid = fluid_properties.pressure_grid

    n_saturated = int(np.sum(pressure_grid < bubble_point_pressure - 1.0))
    n_undersaturated = int(np.sum(pressure_grid >= bubble_point_pressure - 1.0))

    if n_saturated == pressure_grid.size:
        report.info(check, "All cells are SATURATED (P < Pb) at initialisation.")
    elif n_undersaturated == pressure_grid.size:
        report.info(
            check,
            f"All cells UNDERSATURATED (P >= Pb). "
            f"Mean Pb = {float(np.mean(bubble_point_pressure)):.1f} psi  "
            f"Mean P = {float(np.mean(pressure_grid)):.1f} psi.",
        )
    else:
        report.info(
            check,
            f"Mixed initial state: {n_undersaturated} undersaturated, {n_saturated} saturated cell(s).",
        )

    bubble_point_excess = bubble_point_pressure - pressure_grid
    n_large_excess = int(np.sum(bubble_point_excess > 500.0))
    if n_large_excess > 0:
        report.warn(
            check,
            f"{n_large_excess} cell(s) have Pb > P by more than 500 psi "
            f"(max excess = {float(np.max(bubble_point_excess)):.1f} psi).",
            "These cells will immediately liberate gas at t=0, which may cause time-step rejection loops.",
        )

    if config.pvt_tables is None or config.pvt_tables.oil is None:
        return

    oil_pvt_pressures = config.pvt_tables.oil._data.pressures
    if oil_pvt_pressures is None or len(oil_pvt_pressures) < 2:
        return

    table_min_pressure = float(np.min(oil_pvt_pressures))
    table_max_pressure = float(np.max(oil_pvt_pressures))
    bubble_point_min = float(np.min(bubble_point_pressure))
    bubble_point_max = float(np.max(bubble_point_pressure))

    if bubble_point_max > table_max_pressure:
        report.error(
            check,
            f"Maximum bubble-point pressure {bubble_point_max:.1f} psi exceeds "
            f"oil PVT table limit {table_max_pressure:.1f} psi.",
        )
    elif bubble_point_min < table_min_pressure:
        report.warn(
            check,
            f"Minimum bubble-point pressure {bubble_point_min:.1f} psi is below "
            f"oil PVT table lower limit {table_min_pressure:.1f} psi.",
        )
    else:
        report.info(
            check,
            f"Bubble-point pressures [{bubble_point_min:.1f}, {bubble_point_max:.1f}] psi "
            f"within PVT table [{table_min_pressure:.1f}, {table_max_pressure:.1f}] psi.",
        )


def _validate_transmissibility(model: BlackOilModel, report: ValidationReport) -> None:
    """
    Inspect face transmissibility magnitude and condition number proxy.

    Extremely small transmissibilities indicate effectively isolated cells that may cause
    singular linear systems. A `T_max/T_min` ratio above 10¹² predicts iterative solver
    difficulty before the first Newton iteration runs.

    :param model: Reservoir model providing transmissibility computation.
    :param report: Report to append issues to.
    """
    check = "transmissibility"

    try:
        face_transmissibilities = model.build_face_transmissibilities()
    except Exception as exc:
        report.warn(check, f"Transmissibility computation failed: {exc}")
        return

    nx, ny, nz = model.grid_shape

    # Each padded array is (nx+2, ny+2, nz+2).
    # Only extract the regions actually written by the transmissibility kernel:
    #   Tx: indices [0..nx,   1..ny,   1..nz  ] = west boundary + interior + east boundary
    #   Ty: indices [1..nx,   0..ny,   1..nz  ] = north boundary + interior + south boundary
    #   Tz: indices [1..nx,   1..ny,   0..nz  ] = top boundary + interior + bottom boundary
    tx = face_transmissibilities.x[: nx + 1, 1 : ny + 1, 1 : nz + 1]
    ty = face_transmissibilities.y[1 : nx + 1, : ny + 1, 1 : nz + 1]
    tz = face_transmissibilities.z[1 : nx + 1, 1 : ny + 1, : nz + 1]

    all_transmissibilities = np.concatenate([tx.ravel(), ty.ravel(), tz.ravel()])
    positive_transmissibilities = all_transmissibilities[all_transmissibilities > 0.0]

    if positive_transmissibilities.size == 0:
        report.error(
            check,
            "All face transmissibilities are zero. No inter-cell flow is possible.",
        )
        return

    transmissibility_max = float(np.max(positive_transmissibilities))
    transmissibility_min = float(np.min(positive_transmissibilities))
    transmissibility_median = float(np.median(positive_transmissibilities))
    n_near_zero = int(np.sum(all_transmissibilities < 1e-15))

    condition_proxy = (
        transmissibility_max / transmissibility_min
        if transmissibility_min > 0.0
        else float("inf")
    )

    if condition_proxy > 1e12:
        report.warn(
            check,
            f"Transmissibility range [{transmissibility_min:.2e}, {transmissibility_max:.2e}] mD·ft. "
            f"Condition proxy {condition_proxy:.1e} > 10¹².",
            "Highly ill-conditioned transmissibility field may cause iterative solver difficulties. "
            "Consider scaling or using ILU preconditioning.",
        )
    else:
        report.info(
            check,
            f"Transmissibility range [{transmissibility_min:.2e}, {transmissibility_max:.2e}] mD·ft  "
            f"median = {transmissibility_median:.2e}.",
        )

    if n_near_zero > 0:
        report.warn(
            check,
            f"{n_near_zero} face transmissibility value(s) are < 1e-15 (numerically zero).",
            "These faces contribute no flow but may appear in sparse matrix fill-in.",
        )


def _validate_pore_volume_distribution(
    model: BlackOilModel, report: ValidationReport
) -> None:
    """
    Examine the pore-volume distribution across the active grid cells.

    Extremely small pore volumes cause stiff ODEs in explicit transport. Extreme
    pore-volume heterogeneity means the CFL stability limit will be dominated by the
    smallest cells.

    :param model: Reservoir model providing geometry and rock properties.
    :param report: Report to append issues to.
    """
    check = "pore_volume_distribution"
    porosity_grid = model.rock_properties.porosity_grid
    pore_volume_grid = model.pore_volume_grid
    active_pore_volumes = pore_volume_grid[porosity_grid > 0.0]

    if active_pore_volumes.size == 0:
        report.error(check, "No active cells (all porosity = 0). Nothing to simulate.")
        return

    total_pore_volume = float(np.sum(active_pore_volumes))
    min_pore_volume = float(np.min(active_pore_volumes))
    max_pore_volume = float(np.max(active_pore_volumes))
    median_pore_volume = float(np.median(active_pore_volumes))

    report.info(
        check,
        f"Total pore volume: {total_pore_volume:.3e} ft³  min: {min_pore_volume:.3e}  "
        f"median: {median_pore_volume:.3e}  max: {max_pore_volume:.3e}.",
    )

    pore_volume_ratio = (
        max_pore_volume / min_pore_volume if min_pore_volume > 0.0 else float("inf")
    )
    if pore_volume_ratio > 1e6:
        report.warn(
            check,
            f"Pore-volume ratio max/min = {pore_volume_ratio:.1e}. Highly heterogeneous grid.",
            "CFL stability for explicit transport will be controlled by the smallest cells. "
            "Consider local grid refinement or implicit transport.",
        )

    n_micro_pore_cells = int(np.sum(active_pore_volumes < 1e-3))
    if n_micro_pore_cells > 0:
        report.warn(
            check,
            f"{n_micro_pore_cells} active cell(s) have pore volume < 1e-3 ft³ (micro-pore cells).",
            "Micro-pore cells cause stiff ODE systems in explicit transport. "
            "Consider setting their porosity to zero or using a minimum pore-volume threshold.",
        )


def _validate_capillary_pressure_sign(
    config: Config, rock_properties: RockProperties, report: ValidationReport
) -> None:
    """
    Verify the capillary pressure sign convention in the rock-fluid tables.

    Standard black-oil sign convention: Pcow = Po - Pw >= 0 and Pcgo = Pg - Po >= 0.
    A sign flip in table construction produces large spurious initial flux that
    passes the saturation sum check.

    Probes the table at a low-saturation point (Sw = 0.05, So = 0.90, Sg = 0.05)
    where capillary entry pressures should be at their largest positive values for
    a correctly oriented water-wet / gas-oil table.

    :param config: Simulation config carrying the rock-fluid tables.
    :param rock_properties: Rock properties containing residual saturation grids.
    :param report: Report to append issues to.
    """
    check = "capillary_pressure_sign"

    capillary_pressure_table = config.rock_fluid_tables.capillary_pressure_table
    if capillary_pressure_table is None:
        report.info(check, "No capillary pressure table. Skipping sign check.")
        return

    try:
        probe_water_saturation = np.float64(0.05)
        probe_oil_saturation = np.float64(0.90)
        probe_gas_saturation = np.float64(0.05)

        capillary_pressures = capillary_pressure_table(
            water_saturation=probe_water_saturation,
            oil_saturation=probe_oil_saturation,
            gas_saturation=probe_gas_saturation,
            irreducible_water_saturation=rock_properties.irreducible_water_saturation_grid.mean(),
            residual_oil_saturation_water=rock_properties.residual_oil_saturation_water_grid.mean(),
            residual_oil_saturation_gas=rock_properties.residual_oil_saturation_gas_grid.mean(),
            residual_gas_saturation=rock_properties.residual_gas_saturation_grid.mean(),
        )

        pcow = float(capillary_pressures["oil_water"])
        pcgo = float(capillary_pressures["gas_oil"])

        if pcow < -1.0:
            report.warn(
                check,
                f"Pcow = {pcow:.2f} psi at low Sw (0.05) is negative.",
                "Expected Pcow = Po - Pw >= 0 for water-wet systems. "
                "Check capillary pressure table sign convention.",
            )
        if pcgo < -1.0:
            report.warn(
                check,
                f"Pcgo = {pcgo:.2f} psi at low Sg (0.05) is negative.",
                "Expected Pcgo = Pg - Po >= 0. Check gas-oil capillary pressure sign convention.",
            )
        if pcow >= -1.0 and pcgo >= -1.0:
            report.info(
                check,
                f"Capillary pressure sign convention appears correct. "
                f"Pcow = {pcow:.2f} psi, Pcgo = {pcgo:.2f} psi at probe saturations.",
            )

    except Exception as exc:
        report.info(
            check,
            f"Capillary pressure sign check skipped. Table API probe failed: {exc}",
        )
