#!/usr/bin/env python3
"""
Forest carbon accumulation and growth-rate analysis using the Chapman-Richards model.

This script fits forest-type-specific Chapman-Richards growth curves to stand age
and biomass carbon density data, estimates uncertainty, derives the first-order
carbon accumulation rate, and exports fitted parameters, cleaned calibration data,
and publication-ready figures.

Expected input columns
----------------------
Required:
    TYPE             Forest type or ecological group used for stratified fitting.
    stand_age        Stand age in years.
    biomass_carbon   Biomass carbon density, typically Mg C ha^-1.
    biorate          Biomass carbon accumulation rate or related filtering variable.
Optional:
    ECO_ZONE         Ecozone label; renamed internally to eco_region when present.

Main outputs
------------
    fit_results.csv                         Fitted parameters and model statistics.
    corrected_quantile_data.csv             Age-bin 95th quantile points used for fitting.
    combined_fit_with_derivative.jpeg       Growth curves and first derivatives.

Author: Prepared for journal code availability / supplementary material.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import t
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration parameters controlling data cleaning, fitting, and output."""

    biorate_contamination: float = 0.10
    min_biorate: float = 0.01
    xy_contamination: float = 0.05
    isolation_estimators: int = 200
    random_state: int = 42
    age_bin_width: int = 5
    age_bin_quantile: float = 0.95
    min_samples_per_group: int = 10
    old_age_threshold: float = 100.0
    high_age_threshold: float = 150.0
    young_age_threshold: float = 20.0
    high_age_lower_percentile: float = 5.0
    young_age_upper_percentile: float = 95.0
    A_percentile_with_old_samples: float = 95.0
    A_percentile_without_old_samples: float = 90.0
    A_old_sample_ratio_threshold: float = 0.10
    bootstrap_iterations: int = 100
    cr_threshold_default: float = 0.95
    cr_threshold_dnf: float = 0.80
    dnf_min_age: float = 1.0
    dnf_growth_rate_cap: float = 2.0
    k_bounds: Tuple[float, float] = (0.001, 0.5)
    c_bounds: Tuple[float, float] = (0.5, 5.0)
    de_maxiter: int = 1000
    de_popsize: int = 15
    curve_fit_max_nfev: int = 5000
    prediction_points: int = 200


@dataclass
class GroupFitResult:
    """Container for model outputs from a single forest type."""

    forest_type: str
    A: float
    A_se: float
    k: float
    k_se: float
    c: float
    c_se: float
    threshold_age: float
    threshold_carbon: float
    threshold_growth_rate: float
    peak_age: float
    peak_growth_rate: float
    peak_growth_rate_ci: float
    n_samples: int
    r2: float
    rmse: float
    mae: float
    x_corrected: np.ndarray
    y_corrected: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    growth_rate: np.ndarray
    growth_rate_lower: np.ndarray
    growth_rate_upper: np.ndarray


def setup_logging() -> None:
    """Configure readable command-line logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def setup_plot_style() -> None:
    """Apply plotting settings used for the manuscript figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 20,
            "axes.linewidth": 0.8,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.figsize": [6.0, 4.0],
            "figure.dpi": 400,
            "axes.unicode_minus": False,
        }
    )


def chapman_richards(age: np.ndarray, A: float, k: float, c: float) -> np.ndarray:
    """Chapman-Richards carbon accumulation model."""
    age = np.asarray(age, dtype=float)
    return A * (1.0 - np.exp(-k * age)) ** c


def chapman_richards_derivative(age: np.ndarray, A: float, k: float, c: float) -> np.ndarray:
    """First derivative of the Chapman-Richards model with respect to age."""
    age = np.asarray(age, dtype=float)
    exp_term = np.exp(-k * age)
    base = np.maximum(1.0 - exp_term, 1e-12)
    return A * c * k * base ** (c - 1.0) * exp_term


def threshold_age_from_asymptote(k: float, c: float, threshold: float) -> float:
    """Return the age at which the fitted curve reaches a fraction of the asymptote A."""
    return float(-np.log(1.0 - threshold ** (1.0 / c)) / k)


def read_input_table(input_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV or Excel input table and harmonize key column names."""
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path, sheet_name=sheet_name if sheet_name else 0)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input file must be .csv, .xls, or .xlsx")

    df = df.rename(columns={"ECO_ZONE": "eco_region", "biomass_carbon": "biomass_ca"})
    required = {"TYPE", "stand_age", "biomass_ca"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    return df


def isolation_forest_mask(values: np.ndarray, contamination: float, n_estimators: int, random_state: int) -> np.ndarray:
    """Return a boolean mask of observations retained by Isolation Forest."""
    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(values)
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    return model.fit_predict(scaled_values) == 1


def filter_by_biorate(df: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Remove anomalous or very low biorate values before forest-type fitting."""
    values = df[["biorate"]].to_numpy(dtype=float)
    finite_mask = np.isfinite(values).ravel()
    mask = np.zeros(len(df), dtype=bool)
    if finite_mask.any():
        retained = isolation_forest_mask(
            values[finite_mask],
            contamination=cfg.biorate_contamination,
            n_estimators=cfg.isolation_estimators,
            random_state=cfg.random_state,
        )
        mask[np.where(finite_mask)[0]] = retained
    filtered = df.loc[mask].copy()
    filtered = filtered.loc[filtered["biorate"] >= cfg.min_biorate].copy()
    return filtered.reset_index(drop=True)


def filter_xy_outliers(age: np.ndarray, carbon: np.ndarray, cfg: AnalysisConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Remove joint age-carbon outliers using robust scaling and Isolation Forest."""
    data = np.column_stack([age, carbon])
    finite_mask = np.isfinite(data).all(axis=1)
    keep = np.zeros(len(data), dtype=bool)
    if finite_mask.any():
        retained = isolation_forest_mask(
            data[finite_mask],
            contamination=cfg.xy_contamination,
            n_estimators=cfg.isolation_estimators,
            random_state=cfg.random_state,
        )
        keep[np.where(finite_mask)[0]] = retained
    return age[keep], carbon[keep]


def apply_age_conditional_filter(
    age: np.ndarray,
    carbon: np.ndarray,
    age_threshold: float,
    percentile: float,
    comparison: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply age-specific biomass thresholds to reduce implausible tail observations."""
    if comparison == "greater":
        age_mask = age > age_threshold
        threshold = np.percentile(carbon[age_mask], percentile) if age_mask.any() else np.nan
        keep = (~age_mask) | (carbon >= threshold)
    elif comparison == "less":
        age_mask = age < age_threshold
        threshold = np.percentile(carbon[age_mask], percentile) if age_mask.any() else np.nan
        keep = (~age_mask) | (carbon <= threshold)
    else:
        raise ValueError("comparison must be 'greater' or 'less'")

    if not age_mask.any():
        return age, carbon
    return age[keep], carbon[keep]


def clean_group_data(forest_type: str, group: pd.DataFrame, cfg: AnalysisConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Clean observations for one forest type."""
    age = group["stand_age"].to_numpy(dtype=float)
    carbon = group["biomass_ca"].to_numpy(dtype=float)
    age, carbon = filter_xy_outliers(age, carbon, cfg)

    if forest_type == "DNF":
        valid = (age >= cfg.dnf_min_age) & (carbon > 0.0) & np.isfinite(age) & np.isfinite(carbon)
    else:
        valid = (age > 0.0) & (carbon > 0.0) & np.isfinite(age) & np.isfinite(carbon)
    age, carbon = age[valid], carbon[valid]

    age, carbon = apply_age_conditional_filter(
        age,
        carbon,
        age_threshold=cfg.high_age_threshold,
        percentile=cfg.high_age_lower_percentile,
        comparison="greater",
    )
    age, carbon = apply_age_conditional_filter(
        age,
        carbon,
        age_threshold=cfg.young_age_threshold,
        percentile=cfg.young_age_upper_percentile,
        comparison="less",
    )
    return age, carbon


def aggregate_to_age_quantiles(age: np.ndarray, carbon: np.ndarray, cfg: AnalysisConfig) -> pd.DataFrame:
    """Aggregate observations to age-bin upper quantiles used for curve fitting."""
    table = pd.DataFrame({"stand_age": age, "biomass_ca": carbon})
    table["age_group"] = (table["stand_age"] // cfg.age_bin_width) * cfg.age_bin_width
    return (
        table.groupby("age_group", as_index=False)["biomass_ca"]
        .quantile(cfg.age_bin_quantile)
        .rename(columns={"age_group": "stand_age", "biomass_ca": "carbon_quantile"})
    )


def estimate_asymptote_A(age: np.ndarray, carbon: np.ndarray, cfg: AnalysisConfig) -> float:
    """Estimate asymptotic biomass carbon density from high quantiles of raw observations."""
    old_ratio = float(np.mean(age > cfg.old_age_threshold))
    percentile = cfg.A_percentile_with_old_samples if old_ratio > cfg.A_old_sample_ratio_threshold else cfg.A_percentile_without_old_samples
    return float(np.percentile(carbon, percentile))


def fit_shape_parameters(x: np.ndarray, y: np.ndarray, A: float, cfg: AnalysisConfig) -> Tuple[float, float, np.ndarray]:
    """Fit k and c while holding A fixed using global and local optimization."""

    def loss(params: Iterable[float]) -> float:
        k, c = params
        pred = chapman_richards(x, A, k, c)
        if not np.all(np.isfinite(pred)):
            return np.inf
        return float(np.sum((pred - y) ** 2))

    result = differential_evolution(
        loss,
        bounds=[cfg.k_bounds, cfg.c_bounds],
        strategy="best1exp",
        maxiter=cfg.de_maxiter,
        popsize=cfg.de_popsize,
        seed=cfg.random_state,
        updating="immediate",
    )
    k_init, c_init = result.x

    popt, pcov = curve_fit(
        lambda age, k, c: chapman_richards(age, A, k, c),
        x,
        y,
        p0=[k_init, c_init],
        bounds=([cfg.k_bounds[0], cfg.c_bounds[0]], [cfg.k_bounds[1], cfg.c_bounds[1]]),
        method="trf",
        max_nfev=cfg.curve_fit_max_nfev,
    )
    return float(popt[0]), float(popt[1]), pcov


def bootstrap_A_se(age: np.ndarray, carbon: np.ndarray, cfg: AnalysisConfig) -> float:
    """Estimate the standard error of A by non-parametric bootstrap."""
    rng = np.random.default_rng(cfg.random_state)
    samples = []
    for _ in range(cfg.bootstrap_iterations):
        idx = rng.choice(len(carbon), size=len(carbon), replace=True)
        samples.append(estimate_asymptote_A(age[idx], carbon[idx], cfg))
    return float(np.std(samples, ddof=1))


def curve_confidence_interval(x: np.ndarray, A: float, k: float, c: float, pcov: np.ndarray, t_value: float) -> np.ndarray:
    """Approximate confidence interval width for fitted carbon density using the delta method."""
    exp_term = np.exp(-k * x)
    base = np.maximum(1.0 - exp_term, 1e-12)
    dk = A * c * base ** (c - 1.0) * exp_term * x
    dc = A * base**c * np.log(base)
    gradient = np.column_stack([dk, dc])
    variance = np.sum((gradient @ pcov) * gradient, axis=1)
    return t_value * np.sqrt(np.maximum(variance, 0.0))


def derivative_confidence_interval(x: np.ndarray, A: float, k: float, c: float, pcov: np.ndarray, t_value: float) -> np.ndarray:
    """Approximate confidence interval width for the first derivative using the delta method."""
    exp_term = np.exp(-k * x)
    base = np.maximum(1.0 - exp_term, 1e-12)
    derivative = chapman_richards_derivative(x, A, k, c)
    dk = derivative * (1.0 / k + (c - 1.0) * x * exp_term / base - x)
    dc = derivative * (1.0 / c + np.log(base))
    gradient = np.column_stack([dk, dc])
    variance = np.sum((gradient @ pcov) * gradient, axis=1)
    return t_value * np.sqrt(np.maximum(variance, 0.0))


def fit_forest_type(forest_type: str, group: pd.DataFrame, cfg: AnalysisConfig) -> Optional[GroupFitResult]:
    """Clean, aggregate, fit, and evaluate one forest type."""
    age, carbon = clean_group_data(forest_type, group, cfg)
    if len(age) < cfg.min_samples_per_group:
        logging.warning("Skipping %s: only %d valid samples", forest_type, len(age))
        return None

    quantile_data = aggregate_to_age_quantiles(age, carbon, cfg)
    x_corrected = quantile_data["stand_age"].to_numpy(dtype=float)
    y_corrected = quantile_data["carbon_quantile"].to_numpy(dtype=float)

    A = estimate_asymptote_A(age, carbon, cfg)
    k, c, pcov = fit_shape_parameters(x_corrected, y_corrected, A, cfg)
    k_se, c_se = np.sqrt(np.diag(pcov))
    A_se = bootstrap_A_se(age, carbon, cfg)

    dof = max(1, len(x_corrected) - 2)
    t_value = float(t.ppf(0.975, dof))
    x_fit = np.linspace(0.0, float(np.max(age)) * 1.1, cfg.prediction_points)
    y_fit = chapman_richards(x_fit, A, k, c)
    curve_ci = curve_confidence_interval(x_fit, A, k, c, pcov, t_value)

    growth_rate = chapman_richards_derivative(x_fit, A, k, c)
    derivative_ci = derivative_confidence_interval(x_fit, A, k, c, pcov, t_value)

    threshold = cfg.cr_threshold_dnf if forest_type == "DNF" else cfg.cr_threshold_default
    threshold_age = threshold_age_from_asymptote(k, c, threshold)
    threshold_carbon = float(chapman_richards(np.array([threshold_age]), A, k, c)[0])
    threshold_growth_rate = float(chapman_richards_derivative(np.array([threshold_age]), A, k, c)[0])

    peak_index = int(np.argmax(growth_rate))
    peak_age = float(x_fit[peak_index])
    peak_growth_rate = float(growth_rate[peak_index])
    peak_growth_rate_ci = float(derivative_ci[peak_index])

    if forest_type == "DNF":
        growth_rate = np.minimum(growth_rate, cfg.dnf_growth_rate_cap)
        peak_growth_rate = min(peak_growth_rate, cfg.dnf_growth_rate_cap)

    predicted = chapman_richards(x_corrected, A, k, c)
    return GroupFitResult(
        forest_type=forest_type,
        A=A,
        A_se=A_se,
        k=k,
        k_se=float(k_se),
        c=c,
        c_se=float(c_se),
        threshold_age=threshold_age,
        threshold_carbon=threshold_carbon,
        threshold_growth_rate=threshold_growth_rate,
        peak_age=peak_age,
        peak_growth_rate=peak_growth_rate,
        peak_growth_rate_ci=peak_growth_rate_ci,
        n_samples=len(age),
        r2=float(r2_score(y_corrected, predicted)),
        rmse=float(np.sqrt(mean_squared_error(y_corrected, predicted))),
        mae=float(mean_absolute_error(y_corrected, predicted)),
        x_corrected=x_corrected,
        y_corrected=y_corrected,
        x_fit=x_fit,
        y_fit=y_fit,
        y_lower=y_fit - curve_ci,
        y_upper=y_fit + curve_ci,
        growth_rate=growth_rate,
        growth_rate_lower=np.maximum(growth_rate - derivative_ci, 0.0),
        growth_rate_upper=growth_rate + derivative_ci,
    )


def result_table(results: List[GroupFitResult]) -> pd.DataFrame:
    """Convert model results to a compact summary table."""
    rows = []
    for res in results:
        rows.append(
            {
                "Forest_type": res.forest_type,
                "A": res.A,
                "A_se": res.A_se,
                "k": res.k,
                "k_se": res.k_se,
                "c": res.c,
                "c_se": res.c_se,
                "threshold_age": res.threshold_age,
                "threshold_carbon": res.threshold_carbon,
                "threshold_growth_rate": res.threshold_growth_rate,
                "peak_age": res.peak_age,
                "peak_growth_rate": res.peak_growth_rate,
                "peak_growth_rate_ci": res.peak_growth_rate_ci,
                "n_samples": res.n_samples,
                "r2": res.r2,
                "rmse": res.rmse,
                "mae": res.mae,
            }
        )
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False)


def corrected_data_table(results: List[GroupFitResult]) -> pd.DataFrame:
    """Export all age-bin quantile calibration data used in the model fitting."""
    tables = []
    for res in results:
        tables.append(
            pd.DataFrame(
                {
                    "Forest_type": res.forest_type,
                    "stand_age_bin": res.x_corrected,
                    "carbon_95th_quantile": res.y_corrected,
                }
            )
        )
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def plot_results(results: List[GroupFitResult], output_path: Path) -> None:
    """Create the combined curve-fitting and derivative figure."""
    if not results:
        return

    max_panels = min(4, len(results))
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey="row")
    axes = axes.flatten()
    subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    for idx, res in enumerate(results[:max_panels]):
        ax_curve = axes[idx]
        ax_curve.scatter(res.x_corrected, res.y_corrected, c="#FF8C00", marker="o", s=30, alpha=0.6, label="Data points")
        ax_curve.plot(
            res.x_fit,
            res.y_fit,
            "k-",
            linewidth=1.5,
            label=rf"$y={res.A:.3f}(1-e^{{-{res.k:.3f}x}})^{{{res.c:.3f}}}$",
        )
        ax_curve.fill_between(res.x_fit, res.y_lower, res.y_upper, color="gray", alpha=0.3, label="95% CI")
        ax_curve.scatter([res.threshold_age], [res.threshold_carbon], c="red", marker="^", s=100)
        ax_curve.text(0.05, 0.95, subplot_labels[idx], transform=ax_curve.transAxes, fontsize=24, va="top")
        ax_curve.text(0.95, 0.95, res.forest_type, transform=ax_curve.transAxes, fontsize=24, va="top", ha="right")
        ax_curve.set_xlabel("Forest age (years)")
        if idx % 4 == 0:
            ax_curve.set_ylabel("Carbon density (Mg C ha$^{-1}$)")
        ax_curve.grid(True, linestyle="--", alpha=0.3)
        ax_curve.legend(frameon=False, loc="upper left")
        ax_curve.text(
            0.95,
            0.05,
            rf"$R^2={res.r2:.2f}$" + f"\nRMSE={res.rmse:.2f}\nMAE={res.mae:.2f}",
            transform=ax_curve.transAxes,
            va="bottom",
            ha="right",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            fontsize=18,
        )

        ax_rate = axes[idx + 4]
        ax_rate.plot(res.x_fit, res.growth_rate, "k-", linewidth=1.5, label="Growth rate")
        ax_rate.fill_between(res.x_fit, res.growth_rate_lower, res.growth_rate_upper, color="gray", alpha=0.3, label="95% CI")
        ax_rate.scatter([res.threshold_age], [res.threshold_growth_rate], c="red", marker="^", s=100)
        ax_rate.scatter(
            res.peak_age,
            res.peak_growth_rate,
            c="skyblue",
            marker="o",
            s=100,
            label="Peak growth rate",
            facecolors="none",
            edgecolors="skyblue",
            linewidth=1.5,
        )
        ax_rate.text(0.05, 0.95, subplot_labels[idx + 4], transform=ax_rate.transAxes, fontsize=24, va="top")
        ax_rate.set_xlabel("Forest age (years)")
        if idx % 4 == 0:
            ax_rate.set_ylabel("Growth rate (Mg C ha$^{-1}$ yr$^{-1}$)")
        ax_rate.grid(True, linestyle="--", alpha=0.3)
        ax_rate.legend(frameon=False, loc="upper right")
        ax_rate.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        mask = res.x_fit >= 5.0
        if mask.any():
            upper = max(float(np.quantile(res.growth_rate[mask], 0.95) * 1.2), res.threshold_growth_rate * 1.5)
            if res.forest_type == "DNF":
                upper = min(upper, 2.0)
            ax_rate.set_ylim(0, upper)

    for idx in range(max_panels, 4):
        fig.delaxes(axes[idx])
        fig.delaxes(axes[idx + 4])

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=500)
    plt.close(fig)


def run_analysis(input_path: Path, output_dir: Path, sheet_name: Optional[str], cfg: AnalysisConfig) -> pd.DataFrame:
    """Run the complete forest carbon analysis workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_input_table(input_path, sheet_name=sheet_name)
    df = filter_by_biorate(df, cfg)

    results: List[GroupFitResult] = []
    for forest_type, group in df.groupby("TYPE"):
        logging.info("Processing forest type: %s", forest_type)
        try:
            result = fit_forest_type(str(forest_type), group, cfg)
        except RuntimeError as exc:
            logging.warning("Model fitting failed for %s: %s", forest_type, exc)
            result = None
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("No forest type had sufficient valid data for model fitting.")

    table = result_table(results)
    table.to_csv(output_dir / "fit_results.csv", index=False)
    corrected_data_table(results).to_csv(output_dir / "corrected_quantile_data.csv", index=False)
    plot_results(results, output_dir / "combined_fit_with_derivative.jpeg")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit forest-type Chapman-Richards biomass carbon curves and derivatives.")
    parser.add_argument("--input", required=True, type=Path, help="Path to input CSV or Excel file.")
    parser.add_argument("--output-dir", default=Path("forest_carbon_results"), type=Path, help="Directory for output files.")
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name. Defaults to the first sheet.")
    parser.add_argument("--bootstrap-iterations", default=100, type=int, help="Number of bootstrap samples for A standard error.")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed for reproducible filtering and optimization.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    setup_plot_style()
    args = parse_args()
    cfg = AnalysisConfig(bootstrap_iterations=args.bootstrap_iterations, random_state=args.random_state)
    table = run_analysis(args.input, args.output_dir, args.sheet_name, cfg)
    logging.info("Analysis completed. Results saved to %s", args.output_dir)
    logging.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
