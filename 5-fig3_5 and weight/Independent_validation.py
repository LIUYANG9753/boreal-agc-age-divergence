# -*- coding: utf-8 -*-
"""

from __future__ import annotations
Independent validation plotting script — FINAL version.

This version explicitly includes BOTH:
1) point-scale external validation, and
2) 1° grid-mean external validation.

It does NOT use grid-median validation.

Outputs
-------
For each validation mode:
- separate point-scale composite figure
- separate grid-mean composite figure
- one combined point + grid-mean composite figure

Metrics
-------
Only R and RMSE are calculated and shown.

Style
-----
- Times New Roman
- Clean Nature / Science / GRL-style layout
- Upper-left panel labels: 16 pt
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ============================================================
# 1. User settings
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("CRML_PROJECT_ROOT", SCRIPT_DIR))
DATA_DIR = Path(os.environ.get("CRML_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("CRML_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

INPUT_FILE = str(DATA_DIR / "inventory" / "independent_validation_811.xls")

OBSERVED_COL = "AGCD"
LAT_COL = "lat"
LON_COL = "lon"

PRODUCT_COLS = [
    "CR_ML_w010",
    "ESACCI",
    "thurner",
    "DGVM",
    "ORCHIDEE",
    "LPJGUESS",
    "ICESAT2",
]

PRODUCT_NAME_MAP = {
    "CR_ML_w010": "CR–ML",
    "ESACCI": "ESA CCI",
    "thurner": "Thurner",
    "DGVM": "DGVM ensemble",
    "ORCHIDEE": "ORCHIDEE",
    "LPJGUESS": "LPJ-GUESS",
    "ICESAT2": "ICESat-2",
}

GENERAL_INVALID_VALUES = [-99990, -9999, -999, -32768, 32767, 65535, 9999, 99999]

# Only these products treat 0 as NoData.
ZERO_AS_NODATA_PRODUCTS = {"thurner", "ORCHIDEE", "LPJGUESS"}

GRID_SIZE_DEG = 1
MIN_POINTS_PER_GRID = 3
MIN_N_FOR_FIT = 3

RUN_PRODUCT_SPECIFIC = True
RUN_COMMON_SAMPLE = True

# Draw individual product panels as well as composite figures.
WRITE_INDIVIDUAL_PRODUCT_FIGS = True

OUT_DIR = str(OUTPUT_ROOT / "independent_validation_point_and_gridmean")
os.makedirs(OUT_DIR, exist_ok=True)

# Separate 2×4 group figures
FIG_W_SEPARATE = 10.6
FIG_H_SEPARATE = 5.6

# Combined 4×4 group figure containing point and grid-mean in one figure
FIG_W_COMBINED = 10.8
FIG_H_COMBINED = 10.8

# Plot style constants
SCATTER_COLOR = "#4C78A8"
REFERENCE_LINE_COLOR = "0.55"
REGRESSION_LINE_COLOR = "black"


# ============================================================
# 2. Publication style
# ============================================================

def set_publication_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.unicode_minus": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def add_panel_label(ax, label: str) -> None:
    ax.annotate(
        f"({label})",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-35, 8),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="normal",
    )


# ============================================================
# 3. I/O and cleaning
# ============================================================

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in {".xls", ".xlsx"}:
            return pd.read_excel(path)
        if ext == ".csv":
            return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read the input file.\n"
            "Install xlrd for .xls files or openpyxl for .xlsx files.\n"
            f"File: {path}\n"
            f"Original error: {exc}"
        )

    raise ValueError(f"Unsupported file type: {ext}")


def check_required_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(map(str, df.columns))
        )


def to_numeric_and_clean(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    rows = []

    for c in numeric_cols:
        before_non_na = int(df[c].notna().sum())

        df[c] = pd.to_numeric(df[c], errors="coerce")
        after_numeric = int(df[c].notna().sum())

        general_invalid_count = int(df[c].isin(GENERAL_INVALID_VALUES).sum())
        df.loc[df[c].isin(GENERAL_INVALID_VALUES), c] = np.nan

        zero_as_nodata_count = 0
        if c in ZERO_AS_NODATA_PRODUCTS:
            zero_as_nodata_count = int((df[c] == 0).sum())
            df.loc[df[c] == 0, c] = np.nan

        arr = df[c].to_numpy(dtype=float)
        inf_mask = np.isinf(arr)
        inf_count = int(inf_mask.sum())
        if inf_count > 0:
            df.loc[inf_mask, c] = np.nan

        rows.append({
            "column": c,
            "non_na_before_numeric": before_non_na,
            "non_na_after_numeric": after_numeric,
            "general_invalid_to_nan": general_invalid_count,
            "zero_as_nodata": c in ZERO_AS_NODATA_PRODUCTS,
            "zero_as_nodata_to_nan": zero_as_nodata_count,
            "inf_to_nan": inf_count,
            "non_na_after_clean": int(df[c].notna().sum()),
        })

    return df, pd.DataFrame(rows)


def add_spatial_grid(df: pd.DataFrame, LAT_COL: str, LON_COL: str, grid_size: float) -> pd.DataFrame:
    if grid_size <= 0:
        raise ValueError("grid_size_deg must be greater than 0.")

    df = df.copy()
    df["grid_lat"] = np.floor(df[LAT_COL] / grid_size) * grid_size
    df["grid_lon"] = np.floor(df[LON_COL] / grid_size) * grid_size
    df["grid_id"] = (
        df["grid_lat"].round(8).astype(str)
        + "_"
        + df["grid_lon"].round(8).astype(str)
    )
    return df


# ============================================================
# 4. Metrics and datasets
# ============================================================

def calc_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    x = observed AGCD
    y = product AGCD
    error = y - x
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < MIN_N_FOR_FIT:
        return {
            "N": int(len(x)),
            "R": np.nan,
            "RMSE": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        r = np.nan
        slope = np.nan
        intercept = np.nan
    else:
        reg = linregress(x, y)
        r = float(reg.rvalue)
        slope = float(reg.slope)
        intercept = float(reg.intercept)

    err = y - x
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "N": int(len(x)),
        "R": r,
        "RMSE": rmse,
        "slope": slope,
        "intercept": intercept,
    }


def build_point_dataset(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """
    Point-scale external validation dataset:
    each independent validation sample is used directly.
    """
    return df[[OBSERVED_COL, y_col]].dropna().copy()


def aggregate_by_grid_mean(df_geo: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    1° grid-mean external validation dataset.
    """
    sub = df_geo[["grid_id", OBSERVED_COL, y_col]].dropna().copy()

    counts = sub.groupby("grid_id").size().rename("n_valid_xy")
    keep_ids = counts[counts >= MIN_POINTS_PER_GRID].index
    kept = sub[sub["grid_id"].isin(keep_ids)].copy()

    if kept.empty:
        summary = {
            "Y": y_col,
            "n_valid_xy_points_before_grid_filter": int(len(sub)),
            "n_grids_before_grid_filter": int(counts.size),
            "n_valid_xy_points_after_grid_filter": 0,
            "n_grids_after_grid_filter": 0,
            "min_valid_xy_per_kept_grid": np.nan,
            "median_valid_xy_per_kept_grid": np.nan,
            "max_valid_xy_per_kept_grid": np.nan,
        }
        return pd.DataFrame(columns=["grid_id", OBSERVED_COL, y_col]), summary

    g = kept.groupby("grid_id", as_index=False).agg({OBSERVED_COL: "mean", y_col: "mean"})
    kept_counts = counts.loc[keep_ids]

    summary = {
        "Y": y_col,
        "n_valid_xy_points_before_grid_filter": int(len(sub)),
        "n_grids_before_grid_filter": int(counts.size),
        "n_valid_xy_points_after_grid_filter": int(len(kept)),
        "n_grids_after_grid_filter": int(g["grid_id"].nunique()),
        "min_valid_xy_per_kept_grid": int(kept_counts.min()),
        "median_valid_xy_per_kept_grid": float(kept_counts.median()),
        "max_valid_xy_per_kept_grid": int(kept_counts.max()),
    }

    return g, summary


def get_limits_for_products(
    datasets: Dict[str, pd.DataFrame],
    pad: float = 0.05,
) -> Tuple[float, float]:
    vals = []

    for y_col, df in datasets.items():
        if df is None or df.empty:
            continue
        vals.append(df[OBSERVED_COL].to_numpy(float))
        vals.append(df[y_col].to_numpy(float))

    if not vals:
        return 0.0, 1.0

    allv = np.concatenate(vals)
    allv = allv[np.isfinite(allv)]

    if len(allv) == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(allv))
    vmax = float(np.nanmax(allv))
    span = vmax - vmin if vmax > vmin else 1.0

    return vmin - pad * span, vmax + pad * span


def append_metric_row(
    rows: List[Dict[str, object]],
    validation_mode: str,
    scale_name: str,
    y_col: str,
    metrics: Dict[str, float],
) -> None:
    rows.append({
        "validation_mode": validation_mode,
        "scale": scale_name,
        "Y": y_col,
        "grid_size_deg": GRID_SIZE_DEG if scale_name != "point" else np.nan,
        "grid_min_n": MIN_POINTS_PER_GRID if scale_name != "point" else np.nan,
        "N": metrics.get("N", np.nan),
        "R": metrics.get("R", np.nan),
        "RMSE": metrics.get("RMSE", np.nan),
    })


# ============================================================
# 5. Plotting
# ============================================================

def make_stat_text(metrics: Dict[str, float]) -> str:
    return (
        f"N = {metrics['N']}\n"
        f"R = {metrics['R']:.2f}\n"
        f"RMSE = {metrics['RMSE']:.2f}"
    )


def plot_one_validation_panel(
    ax,
    df_xy: pd.DataFrame,
    y_col: str,
    lim_min: float,
    lim_max: float,
    title: str,
) -> Dict[str, float]:
    style_axes(ax)

    ax.set_title(title, pad=3)

    if df_xy is None or df_xy.empty or len(df_xy) < MIN_N_FOR_FIT:
        ax.text(
            0.04,
            0.96,
            f"N < {MIN_N_FOR_FIT}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.82),
        )
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal", adjustable="box")
        return calc_metrics(np.array([]), np.array([]))

    x = df_xy[OBSERVED_COL].to_numpy(float)
    y = df_xy[y_col].to_numpy(float)
    metrics = calc_metrics(x, y)

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle=(0, (3, 2)),
        color=REFERENCE_LINE_COLOR,
        linewidth=0.9,
        zorder=1,
    )

    ax.scatter(
        x,
        y,
        s=10,
        facecolor=SCATTER_COLOR,
        edgecolor="none",
        alpha=0.55,
        rasterized=True,
        zorder=2,
    )

    if np.isfinite(metrics["slope"]) and np.isfinite(metrics["intercept"]):
        xx = np.linspace(lim_min, lim_max, 120)
        ax.plot(
            xx,
            metrics["slope"] * xx + metrics["intercept"],
            color=REGRESSION_LINE_COLOR,
            linewidth=1.0,
            zorder=3,
        )

    ax.text(
        0.04,
        0.96,
        make_stat_text(metrics),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.82),
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")

    return metrics


def panel_letters(n: int) -> List[str]:
    return [chr(ord("a") + i) for i in range(n)]


def plot_composite_single_scale(
    datasets: Dict[str, pd.DataFrame],
    metrics_rows: List[Dict[str, object]],
    validation_mode: str,
    scale_name: str,
    out_pdf: str,
    out_png: str,
) -> None:
    """
    A 2×4 figure for a single validation scale:
    7 product panels; the last panel is left blank.
    """
    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_SEPARATE, FIG_H_SEPARATE), dpi=300)
    axes_flat = axes.ravel()
    letters = panel_letters(8)

    lim_min, lim_max = get_limits_for_products(datasets)

    for i, y_col in enumerate(PRODUCT_COLS):
        ax = axes_flat[i]
        add_panel_label(ax, letters[i])

        title = PRODUCT_NAME_MAP.get(y_col, y_col)
        plot_one_validation_panel(
            ax=ax,
            df_xy=datasets.get(y_col),
            y_col=y_col,
            lim_min=lim_min,
            lim_max=lim_max,
            title=title,
        )

        if i % 4 == 0:
            ax.set_ylabel("Estimated AGCD (Mg C ha$^{-1}$)")
        else:
            ax.set_ylabel("")

        if i >= 4:
            ax.set_xlabel("Observed AGCD (Mg C ha$^{-1}$)")
        else:
            ax.set_xlabel("")

    axes_flat[7].axis("off")

    scale_label = "point scale" if scale_name == "point" else f"{GRID_SIZE_DEG:g}° grid mean"
    fig.suptitle(f"Independent validation", y=1.01, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def plot_composite_point_and_gridmean(
    point_datasets: Dict[str, pd.DataFrame],
    gridmean_datasets: Dict[str, pd.DataFrame],
    metrics_rows: List[Dict[str, object]],
    validation_mode: str,
    gridmean_scale_name: str,
    out_pdf: str,
    out_png: str,
) -> None:
    """
    One 4×4 combined figure:
    Row 1–2: point-scale validation
    Row 3–4: grid-mean validation
    """
    fig, axes = plt.subplots(4, 4, figsize=(FIG_W_COMBINED, FIG_H_COMBINED), dpi=300)
    axes_flat = axes.ravel()
    letters = panel_letters(16)

    point_lim = get_limits_for_products(point_datasets)
    grid_lim = get_limits_for_products(gridmean_datasets)

    # Top half: point scale
    for i, y_col in enumerate(PRODUCT_COLS):
        ax = axes_flat[i]
        add_panel_label(ax, letters[i])

        title = PRODUCT_NAME_MAP.get(y_col, y_col)
        plot_one_validation_panel(
            ax=ax,
            df_xy=point_datasets.get(y_col),
            y_col=y_col,
            lim_min=point_lim[0],
            lim_max=point_lim[1],
            title=title,
        )

        if i % 4 == 0:
            ax.set_ylabel("Estimated AGCD\n(Mg C ha$^{-1}$)")
        else:
            ax.set_ylabel("")
        if i >= 4:
            ax.set_xlabel("Observed AGCD (Mg C ha$^{-1}$)")
        else:
            ax.set_xlabel("")

    axes_flat[7].axis("off")

    # Bottom half: grid mean
    offset = 8
    for j, y_col in enumerate(PRODUCT_COLS):
        ax = axes_flat[offset + j]
        add_panel_label(ax, letters[offset + j])

        title = PRODUCT_NAME_MAP.get(y_col, y_col)
        plot_one_validation_panel(
            ax=ax,
            df_xy=gridmean_datasets.get(y_col),
            y_col=y_col,
            lim_min=grid_lim[0],
            lim_max=grid_lim[1],
            title=title,
        )

        if j % 4 == 0:
            ax.set_ylabel("Estimated AGCD\n(Mg C ha$^{-1}$)")
        else:
            ax.set_ylabel("")
        if j >= 4:
            ax.set_xlabel("Observed AGCD (Mg C ha$^{-1}$)")
        else:
            ax.set_xlabel("")

    axes_flat[15].axis("off")

    # Row labels
    fig.text(0.006, 0.755, "Point scale", rotation=90, va="center", ha="left", fontsize=11)
    fig.text(0.006, 0.265, f"{GRID_SIZE_DEG:g}° grid mean", rotation=90, va="center", ha="left", fontsize=11)

    fig.suptitle(f"Independent validation ({validation_mode})", y=1.005, fontsize=11)
    fig.tight_layout(rect=(0.018, 0.0, 1.0, 0.985))
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def plot_individual_product_figures(
    datasets: Dict[str, pd.DataFrame],
    scale_name: str,
    validation_mode: str,
    out_subdir: str,
) -> None:
    os.makedirs(out_subdir, exist_ok=True)
    lim_min, lim_max = get_limits_for_products(datasets)

    for y_col in PRODUCT_COLS:
        fig, ax = plt.subplots(figsize=(2.7, 2.7), dpi=300)
        plot_one_validation_panel(
            ax=ax,
            df_xy=datasets.get(y_col),
            y_col=y_col,
            lim_min=lim_min,
            lim_max=lim_max,
            title=PRODUCT_NAME_MAP.get(y_col, y_col),
        )
        ax.set_xlabel("Observed AGCD (Mg C ha$^{-1}$)")
        ax.set_ylabel("Estimated AGCD (Mg C ha$^{-1}$)")
        fig.tight_layout()

        out_pdf = os.path.join(out_subdir, f"{validation_mode}_{scale_name}_{y_col}.pdf")
        out_png = os.path.join(out_subdir, f"{validation_mode}_{scale_name}_{y_col}.png")
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=600)
        plt.close(fig)


# ============================================================
# 6. Validation workflow
# ============================================================

def run_validation_mode(
    df_in: pd.DataFrame,
    validation_mode: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    mode_dir = os.path.join(OUT_DIR, validation_mode)
    os.makedirs(mode_dir, exist_ok=True)

    df_geo = df_in.dropna(subset=[LAT_COL, LON_COL]).copy()
    df_geo = add_spatial_grid(df_geo, LAT_COL, LON_COL, GRID_SIZE_DEG)

    gridmean_scale = f"grid{str(GRID_SIZE_DEG).replace('.', 'p')}deg_mean"

    metrics_rows: List[Dict[str, object]] = []
    grid_summary_rows: List[Dict[str, object]] = []

    point_datasets: Dict[str, pd.DataFrame] = {}
    gridmean_datasets: Dict[str, pd.DataFrame] = {}

    for y_col in PRODUCT_COLS:
        # ----------------------------
        # POINT-SCALE EXTERNAL VALIDATION
        # ----------------------------
        point_df = build_point_dataset(df_in, y_col)
        point_datasets[y_col] = point_df

        point_metrics = calc_metrics(
            point_df[OBSERVED_COL].to_numpy(float),
            point_df[y_col].to_numpy(float),
        )
        append_metric_row(metrics_rows, validation_mode, "point", y_col, point_metrics)

        # ----------------------------
        # GRID-MEAN EXTERNAL VALIDATION
        # ----------------------------
        grid_df, grid_summary = aggregate_by_grid_mean(df_geo, y_col)
        gridmean_datasets[y_col] = grid_df

        grid_summary.update({
            "validation_mode": validation_mode,
            "scale": gridmean_scale,
            "grid_size_deg": GRID_SIZE_DEG,
            "grid_min_n": MIN_POINTS_PER_GRID,
        })
        grid_summary_rows.append(grid_summary)

        grid_metrics = calc_metrics(
            grid_df[OBSERVED_COL].to_numpy(float),
            grid_df[y_col].to_numpy(float),
        )
        append_metric_row(metrics_rows, validation_mode, gridmean_scale, y_col, grid_metrics)

    # ----------------------------
    # Output figures
    # ----------------------------
    point_pdf = os.path.join(mode_dir, f"independent_validation_POINT_{validation_mode}.pdf")
    point_png = os.path.join(mode_dir, f"independent_validation_POINT_{validation_mode}.png")
    plot_composite_single_scale(
        datasets=point_datasets,
        metrics_rows=metrics_rows,
        validation_mode=validation_mode,
        scale_name="point",
        out_pdf=point_pdf,
        out_png=point_png,
    )
    print(f"[output] Point-scale composite figure: {point_pdf}")

    grid_pdf = os.path.join(mode_dir, f"independent_validation_GRIDMEAN_{validation_mode}.pdf")
    grid_png = os.path.join(mode_dir, f"independent_validation_GRIDMEAN_{validation_mode}.png")
    plot_composite_single_scale(
        datasets=gridmean_datasets,
        metrics_rows=metrics_rows,
        validation_mode=validation_mode,
        scale_name=gridmean_scale,
        out_pdf=grid_pdf,
        out_png=grid_png,
    )
    print(f"[output] 1-degree grid-mean composite figure: {grid_pdf}")

    combined_pdf = os.path.join(mode_dir, f"independent_validation_POINT_plus_GRIDMEAN_{validation_mode}.pdf")
    combined_png = os.path.join(mode_dir, f"independent_validation_POINT_plus_GRIDMEAN_{validation_mode}.png")
    plot_composite_point_and_gridmean(
        point_datasets=point_datasets,
        gridmean_datasets=gridmean_datasets,
        metrics_rows=metrics_rows,
        validation_mode=validation_mode,
        gridmean_scale_name=gridmean_scale,
        out_pdf=combined_pdf,
        out_png=combined_png,
    )
    print(f"[output] Combined point-scale + grid-mean figure: {combined_pdf}")

    if WRITE_INDIVIDUAL_PRODUCT_FIGS:
        plot_individual_product_figures(
            datasets=point_datasets,
            scale_name="point",
            validation_mode=validation_mode,
            out_subdir=os.path.join(mode_dir, "individual_POINT"),
        )
        plot_individual_product_figures(
            datasets=gridmean_datasets,
            scale_name=gridmean_scale,
            validation_mode=validation_mode,
            out_subdir=os.path.join(mode_dir, "individual_GRIDMEAN"),
        )

    return metrics_rows, grid_summary_rows


# ============================================================
# 7. Main
# ============================================================

def main() -> None:
    set_publication_style()

    print("========== Step 1: Read data ==========")
    print(f"Input file: {INPUT_FILE}")

    df = read_table(INPUT_FILE)
    df.columns = df.columns.astype(str).str.strip()

    required_cols = [OBSERVED_COL, LAT_COL, LON_COL] + PRODUCT_COLS
    check_required_columns(df, required_cols)

    print(f"Raw record count: {len(df)}")

    print("\n========== Step 2: Numeric conversion and invalid-value handling ==========")
    df_clean, cleaning_summary = to_numeric_and_clean(df, required_cols)

    cleaned_path = os.path.join(OUT_DIR, "cleaned_point_valid_for_validation.csv")
    cleaning_summary_path = os.path.join(OUT_DIR, "cleaning_summary.csv")
    df_clean.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    cleaning_summary.to_csv(cleaning_summary_path, index=False, encoding="utf-8-sig")

    print(f"[output] Cleaned samples: {cleaned_path}")
    print(f"[output] Cleaning summary: {cleaning_summary_path}")

    all_metrics_rows: List[Dict[str, object]] = []
    all_grid_summary_rows: List[Dict[str, object]] = []

    if RUN_PRODUCT_SPECIFIC:
        print("\n========== Step 3: Product-specific validation: POINT + GRIDMEAN ==========")
        m_rows, g_rows = run_validation_mode(df_clean, "product_specific")
        all_metrics_rows.extend(m_rows)
        all_grid_summary_rows.extend(g_rows)

    if RUN_COMMON_SAMPLE:
        print("\n========== Step 4: Common-sample validation: POINT + GRIDMEAN ==========")
        common_cols = [OBSERVED_COL, LAT_COL, LON_COL] + PRODUCT_COLS
        common_df = df_clean.dropna(subset=common_cols).copy()

        common_path = os.path.join(OUT_DIR, "common_sample_all_products.csv")
        common_df.to_csv(common_path, index=False, encoding="utf-8-sig")

        print(f"Common-sample N: {len(common_df)}")
        print(f"[output] All-product common sample: {common_path}")

        if len(common_df) >= MIN_N_FOR_FIT:
            m_rows, g_rows = run_validation_mode(common_df, "common_sample")
            all_metrics_rows.extend(m_rows)
            all_grid_summary_rows.extend(g_rows)
        else:
            print(f"[warning] Common-sample N < {MIN_N_FOR_FIT}; skipping common_sample.")

    print("\n========== Step 5: Write metric tables ==========")
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_csv = os.path.join(OUT_DIR, "independent_validation_metrics_R_RMSE_POINT_and_GRIDMEAN.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    grid_summary_df = pd.DataFrame(all_grid_summary_rows)
    grid_summary_csv = os.path.join(OUT_DIR, "grid_aggregation_summary_GRIDMEAN_only.csv")
    grid_summary_df.to_csv(grid_summary_csv, index=False, encoding="utf-8-sig")

    print(f"[output] Validation metric summary: {metrics_csv}")
    print(f"[output] Grid-aggregation summary: {grid_summary_csv}")

    print("\n========== Complete ==========")
    print(f"Output directory: {OUT_DIR}")
    print("\nNotes:")
    print("1) Point-scale external-validation figures are written explicitly.")
    print("2) 1-degree grid-mean external-validation figures are also written.")
    print("3) Grid-median validation is not generated.")
    print("4) Only R and RMSE are reported.")
    print("5) Figures use Times New Roman and 16-point upper-left panel labels.")
    print("6) Each validation mode writes POINT, GRIDMEAN, and POINT_plus_GRIDMEAN composite figures.")


if __name__ == "__main__":
    main()
