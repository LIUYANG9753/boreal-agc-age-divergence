# -*- coding: utf-8 -*-
"""

from __future__ import annotations
Create Fig. 3: CR–ML low-weight sensitivity analysis based on development samples only.

Main-figure design:
- Weight selection is treated as a constrained sensitivity analysis.
- The 811 independent validation samples are not used for weight selection.
- No subjective combined score, Pareto front, or spatial-mean-AGCD increase is used
  in the main figure.

Figure structure:
(a) Endpoint AGCD distributions: AGC_ML vs CR_AGC
(b) Overall inventory error across CR_AGC weights: RMSE and |Bias|
(c) Age-stratified bias responses across CR_AGC weights: young, middle-aged, old

CR_ML_w = (1 - w) * AGC_ML + w * CR_AGC

Interpretation:
The selected low weight (w = 0.10) is supported by maintaining low overall error
while limiting age-stratified bias. The figure does not use an increase in spatial
mean AGCD as a selection criterion.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
except ImportError:
    rasterio = None
    WarpedVRT = None
    Resampling = None


# ============================================================
# 1. User settings
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("CRML_PROJECT_ROOT", SCRIPT_DIR))
DATA_DIR = Path(os.environ.get("CRML_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("CRML_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
ANALYSIS_OUTPUT_DIR = OUTPUT_ROOT / "weight_sensitivity_external_validation"

OUT_DIR = str(OUTPUT_ROOT / "fig2_weight_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

# Development-sample weight sensitivity tables. The 811 external samples are not used here.
METRICS_AGE20_CSV = str(ANALYSIS_OUTPUT_DIR / "survey_year_sensitivity_weighted_CRML_metrics_age20.csv")
METRICS_BROAD_AGE_CSV = str(ANALYSIS_OUTPUT_DIR / "survey_year_sensitivity_weighted_CRML_metrics_broad_age.csv")

# Spatial weight sensitivity table, used only for point size in panel d.
SPATIAL_WEIGHTED_CSV = str(DATA_DIR / "derived" / "spatial_stats_weighted_CRML.csv")

# Raster inputs for panel a
AGC_ML_RASTER = str(DATA_DIR / "rasters" / "AGCD_ML.tif")
CR_AGC_RASTER = str(DATA_DIR / "rasters" / "AGCD_CR.tif")

# Optional: if you already have a saved panel-a sample CSV, put its path here.
# This avoids resampling the rasters again.
PANEL_A_SAMPLE_CSV: Optional[str] = None

# Optional spatial domain mask; pixels > 0 are treated as valid.
DOMAIN_MASK_RASTER: Optional[str] = None

# Selected CR_AGC weight
SELECTED_WEIGHT = 0.10

# Low-weight interval shown as shaded region in panels b and c
LOW_WEIGHT_MAX = 0.15

# Scenario used in CSV tables
SCENARIO = "all_years"

# Raster sampling settings
MAX_RASTER_SAMPLE = 1_000_000
RANDOM_SEED = 42

# Valid AGCD range
MAX_VALID_AGCD = 500.0
MIN_VALID_FOR_AGCML_CRAGC = 0.0

# Panel a x-axis
DIST_XMAX = 160

# Figure size
FIG_WIDTH_IN = 7.4
FIG_HEIGHT_IN = 6.9


# ============================================================
# 2. Style helpers
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
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.0,
        "axes.unicode_minus": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(False)


def add_panel_label(ax, label: str) -> None:
    ax.annotate(
        f"({label})",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-40, 8),
        textcoords="offset points",
        fontsize=16,
        fontweight="normal",
        va="top",
        ha="left",
    )


def add_low_weight_band(ax) -> None:
    # Emphasize the low-weight interval without competing with data curves.
    ax.axvspan(
        0,
        LOW_WEIGHT_MAX,
        color="0.82",
        alpha=0.75,
        zorder=0,
        linewidth=0,
    )
    # Subtle boundary at the upper limit of the low-weight interval.
    ax.axvline(
        LOW_WEIGHT_MAX,
        color="0.68",
        linestyle=(0, (2, 2)),
        linewidth=0.65,
        zorder=0.5,
    )


def get_script_dir() -> str:
    try:
        return str(Path(__file__).resolve().parent)
    except NameError:
        return os.getcwd()


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    if path and os.path.exists(path):
        return path

    basename = os.path.basename(path)
    if basename == path:
        basename = re.split(r"[\\/]", str(path))[-1]

    candidates = [
        os.path.join(os.getcwd(), basename),
        os.path.join(get_script_dir(), basename),
        os.path.join(str(DATA_DIR), basename),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return path


def require_file(path: str, label: str) -> None:
    resolved = resolve_path(path)
    if not resolved or not os.path.exists(resolved):
        raise FileNotFoundError(
            f"{label} not found:\n{path}\nResolved path tried:\n{resolved}"
        )


def read_csv_resolved(path: str, label: str) -> pd.DataFrame:
    require_file(path, label)
    resolved = resolve_path(path)
    print(f"[read] {label}: {resolved}")
    return pd.read_csv(resolved)


# ============================================================
# 3. Raster sampling for panel a
# ============================================================

def clean_agcd(
    arr: np.ndarray,
    nodata: Optional[float],
    min_valid: float,
) -> Tuple[np.ndarray, np.ndarray]:

    values = arr.astype("float64", copy=True)
    valid = np.isfinite(values)

    if nodata is not None and np.isfinite(nodata):
        valid &= ~np.isclose(values, nodata)

    invalid_values = [-99990, -9999, -999, -32768, 32767, 65535, 9999, 99999]
    valid &= ~np.isin(values, invalid_values)
    valid &= values >= min_valid
    valid &= values <= MAX_VALID_AGCD

    values[~valid] = np.nan
    return values, valid


def reservoir_update_pair(
    x_res: np.ndarray,
    y_res: np.ndarray,
    n_filled: int,
    n_seen: int,
    x_new: np.ndarray,
    y_new: np.ndarray,
    max_n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int, int]:

    n_new = len(x_new)
    if n_new == 0:
        return x_res, y_res, n_filled, n_seen

    if n_filled < max_n:
        take_n = min(max_n - n_filled, n_new)
        x_res[n_filled:n_filled + take_n] = x_new[:take_n]
        y_res[n_filled:n_filled + take_n] = y_new[:take_n]
        n_filled += take_n

        x_new = x_new[take_n:]
        y_new = y_new[take_n:]
        n_seen += take_n

        n_new = len(x_new)
        if n_new == 0:
            return x_res, y_res, n_filled, n_seen

    global_index = np.arange(n_seen, n_seen + n_new, dtype="float64")
    prob = max_n / (global_index + 1.0)
    include = rng.random(n_new) < prob

    if include.any():
        slots = rng.integers(0, max_n, size=int(include.sum()))
        x_res[slots] = x_new[include]
        y_res[slots] = y_new[include]

    n_seen += n_new
    return x_res, y_res, n_filled, n_seen


def sample_agcml_cragc_from_rasters(
    agc_ml_path: str,
    cr_agc_path: str,
    mask_path: Optional[str],
    max_sample: int,
    seed: int,
) -> pd.DataFrame:

    if rasterio is None:
        raise ImportError("rasterio is required for raster sampling. Please install rasterio.")

    require_file(agc_ml_path, "AGCD_ML raster")
    require_file(cr_agc_path, "AGCD_CR raster")

    agc_ml_path = resolve_path(agc_ml_path)
    cr_agc_path = resolve_path(cr_agc_path)

    if mask_path is not None:
        require_file(mask_path, "Domain mask raster")
        mask_path = resolve_path(mask_path)

    rng = np.random.default_rng(seed)

    x_res = np.empty(max_sample, dtype="float64")
    y_res = np.empty(max_sample, dtype="float64")
    n_filled = 0
    n_seen = 0

    with rasterio.open(agc_ml_path) as ref:
        with rasterio.open(agc_ml_path) as agc_src, rasterio.open(cr_agc_path) as cr_src:
            agc_vrt = WarpedVRT(
                agc_src,
                crs=ref.crs,
                transform=ref.transform,
                width=ref.width,
                height=ref.height,
                resampling=Resampling.bilinear,
                nodata=agc_src.nodata,
            )

            cr_vrt = WarpedVRT(
                cr_src,
                crs=ref.crs,
                transform=ref.transform,
                width=ref.width,
                height=ref.height,
                resampling=Resampling.bilinear,
                nodata=cr_src.nodata,
            )

            if mask_path is not None:
                mask_src = rasterio.open(mask_path)
                mask_vrt = WarpedVRT(
                    mask_src,
                    crs=ref.crs,
                    transform=ref.transform,
                    width=ref.width,
                    height=ref.height,
                    resampling=Resampling.nearest,
                    nodata=mask_src.nodata,
                )
            else:
                mask_src = None
                mask_vrt = None

            for idx, (_, window) in enumerate(ref.block_windows(1), start=1):
                if idx % 500 == 0:
                    print(f"Sampling raster window {idx}")

                agc_raw = agc_vrt.read(1, window=window, masked=False)
                cr_raw = cr_vrt.read(1, window=window, masked=False)

                agc, valid_agc = clean_agcd(
                    agc_raw,
                    agc_vrt.nodata,
                    MIN_VALID_FOR_AGCML_CRAGC,
                )
                cr, valid_cr = clean_agcd(
                    cr_raw,
                    cr_vrt.nodata,
                    MIN_VALID_FOR_AGCML_CRAGC,
                )

                valid = valid_agc & valid_cr

                if mask_vrt is not None:
                    mask_raw = mask_vrt.read(1, window=window, masked=False).astype("float64")
                    mask_valid = np.isfinite(mask_raw)

                    if mask_vrt.nodata is not None and np.isfinite(mask_vrt.nodata):
                        mask_valid &= ~np.isclose(mask_raw, mask_vrt.nodata)

                    mask_valid &= mask_raw > 0
                    valid &= mask_valid

                if not np.any(valid):
                    continue

                x_new = agc[valid].ravel()
                y_new = cr[valid].ravel()

                x_res, y_res, n_filled, n_seen = reservoir_update_pair(
                    x_res=x_res,
                    y_res=y_res,
                    n_filled=n_filled,
                    n_seen=n_seen,
                    x_new=x_new,
                    y_new=y_new,
                    max_n=max_sample,
                    rng=rng,
                )

            agc_vrt.close()
            cr_vrt.close()

            if mask_vrt is not None:
                mask_vrt.close()
            if mask_src is not None:
                mask_src.close()

    sample = pd.DataFrame({
        "AGCD_ML": x_res[:n_filled],
        "AGCD_CR": y_res[:n_filled],
    })

    print(f"Sampled {len(sample):,} paired pixels from {n_seen:,} valid paired pixels.")
    return sample


def get_or_create_panel_a_sample() -> pd.DataFrame:
    if PANEL_A_SAMPLE_CSV is not None:
        sample_path = resolve_path(PANEL_A_SAMPLE_CSV)
        if sample_path and os.path.exists(sample_path):
            print(f"[read] Panel-a sample: {sample_path}")
            return pd.read_csv(sample_path)

    sample_path = os.path.join(OUT_DIR, "fig2a_AGCML_CRAGC_distribution_sample.csv")

    if os.path.exists(sample_path):
        print(f"[read] Existing panel-a sample: {sample_path}")
        return pd.read_csv(sample_path)

    print("Sampling AGC_ML and CR_AGC rasters for Fig. 3a.")
    sample = sample_agcml_cragc_from_rasters(
        agc_ml_path=AGC_ML_RASTER,
        cr_agc_path=CR_AGC_RASTER,
        mask_path=DOMAIN_MASK_RASTER,
        max_sample=MAX_RASTER_SAMPLE,
        seed=RANDOM_SEED,
    )

    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"[write] Panel-a sample: {sample_path}")
    return sample


# ============================================================
# 4. CSV data preparation
# ============================================================

def normalize_weight_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "CR_AGC_weight" not in df.columns:
        raise ValueError("Missing CR_AGC_weight column.")

    df["CR_AGC_weight"] = pd.to_numeric(df["CR_AGC_weight"], errors="coerce")
    df = df[pd.notna(df["CR_AGC_weight"])].copy()

    return df


def filter_weighted_crml_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "is_weighted_crml" in df.columns:
        flag = df["is_weighted_crml"].astype(str).str.lower()
        df = df[flag.isin(["true", "1", "yes"])].copy()

    return df


def get_bias_column(df: pd.DataFrame) -> str:
    candidates = [
        "Bias(product-observed)",
        "Bias(y-x)",
        "Bias",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError("No valid bias column found.")


def read_metrics_all() -> pd.DataFrame:
    df = read_csv_resolved(METRICS_AGE20_CSV, "METRICS_AGE20_CSV")

    if "survey_year_scenario" in df.columns:
        df = df[df["survey_year_scenario"].astype(str) == SCENARIO].copy()

    if "subset" in df.columns:
        df = df[df["subset"].astype(str) == "all"].copy()

    df = filter_weighted_crml_rows(df)
    df = normalize_weight_column(df)

    if "RMSE" not in df.columns:
        raise ValueError("Missing RMSE column in METRICS_AGE20_CSV.")

    if "AbsBias" not in df.columns:
        bias_col = get_bias_column(df)
        df["AbsBias"] = pd.to_numeric(df[bias_col], errors="coerce").abs()

    keep_cols = ["CR_AGC_weight", "RMSE", "AbsBias"]
    if "N" in df.columns:
        keep_cols.append("N")

    out = (
        df[keep_cols]
        .drop_duplicates(subset=["CR_AGC_weight"])
        .sort_values("CR_AGC_weight")
        .reset_index(drop=True)
    )

    return out


def read_metrics_broad_age() -> pd.DataFrame:
    df = read_csv_resolved(METRICS_BROAD_AGE_CSV, "METRICS_BROAD_AGE_CSV")

    if "survey_year_scenario" in df.columns:
        df = df[df["survey_year_scenario"].astype(str) == SCENARIO].copy()

    if "subset" in df.columns:
        df = df[df["subset"].astype(str) == "broad_age"].copy()

    df = filter_weighted_crml_rows(df)
    df = normalize_weight_column(df)

    if "age_class" not in df.columns:
        if "broad_age_class" in df.columns:
            df = df.rename(columns={"broad_age_class": "age_class"})
        else:
            raise ValueError("Missing age_class or broad_age_class column in broad-age metrics.")

    bias_col = get_bias_column(df)
    if bias_col != "Bias(product-observed)":
        df = df.rename(columns={bias_col: "Bias(product-observed)"})

    keep_ages = ["young_<50", "middle_50_150", "old_>150"]

    out = df[df["age_class"].isin(keep_ages)].copy()
    out["Bias(product-observed)"] = pd.to_numeric(out["Bias(product-observed)"], errors="coerce")

    out = out.sort_values(["age_class", "CR_AGC_weight"]).reset_index(drop=True)
    return out


def read_spatial_weighted_optional() -> pd.DataFrame:
    resolved = resolve_path(SPATIAL_WEIGHTED_CSV)

    if not resolved or not os.path.exists(resolved):
        print(
            "[warning] SPATIAL_WEIGHTED_CSV not found. "
            "Panel d point sizes will be constant."
        )
        return pd.DataFrame(columns=["CR_AGC_weight", "mean"])

    df = pd.read_csv(resolved)
    print(f"[read] SPATIAL_WEIGHTED_CSV: {resolved}")

    required = ["CR_AGC_weight", "mean"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(
            f"[warning] Spatial file missing columns {missing}. "
            "Panel d point sizes will be constant."
        )
        return pd.DataFrame(columns=["CR_AGC_weight", "mean"])

    df = df[required].copy()
    df["CR_AGC_weight"] = pd.to_numeric(df["CR_AGC_weight"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df = df.dropna(subset=["CR_AGC_weight", "mean"])

    return (
        df.drop_duplicates("CR_AGC_weight")
        .sort_values("CR_AGC_weight")
        .reset_index(drop=True)
    )


def make_tradeoff_data(
    metrics_all: pd.DataFrame,
    metrics_broad: pd.DataFrame,
    spatial_weighted: pd.DataFrame,
) -> pd.DataFrame:

    all_df = metrics_all.copy()

    broad = metrics_broad.copy()
    broad["abs_bias"] = broad["Bias(product-observed)"].abs()

    middle_old = broad[broad["age_class"].isin(["middle_50_150", "old_>150"])].copy()

    if middle_old.empty:
        raise ValueError("No middle-aged or old-forest rows found for trade-off analysis.")

    middle_old_summary = (
        middle_old
        .groupby("CR_AGC_weight", as_index=False)
        .agg(
            middle_old_abs_bias_mean=("abs_bias", "mean"),
            middle_old_abs_bias_max=("abs_bias", "max"),
        )
    )

    young_summary = (
        broad[broad["age_class"] == "young_<50"]
        [["CR_AGC_weight", "abs_bias"]]
        .rename(columns={"abs_bias": "young_abs_bias"})
    )

    middle_summary = (
        broad[broad["age_class"] == "middle_50_150"]
        [["CR_AGC_weight", "abs_bias"]]
        .rename(columns={"abs_bias": "middle_abs_bias"})
    )

    old_summary = (
        broad[broad["age_class"] == "old_>150"]
        [["CR_AGC_weight", "abs_bias"]]
        .rename(columns={"abs_bias": "old_abs_bias"})
    )

    out = all_df.merge(middle_old_summary, on="CR_AGC_weight", how="left")
    out = out.merge(young_summary, on="CR_AGC_weight", how="left")
    out = out.merge(middle_summary, on="CR_AGC_weight", how="left")
    out = out.merge(old_summary, on="CR_AGC_weight", how="left")

    if spatial_weighted is not None and not spatial_weighted.empty:
        out = out.merge(spatial_weighted, on="CR_AGC_weight", how="left")

        baseline_idx = np.argmin(np.abs(out["CR_AGC_weight"].to_numpy(float) - 0.0))
        baseline_mean = float(out.iloc[int(baseline_idx)]["mean"])

        out["spatial_mean_AGCD"] = out["mean"]
        out["spatial_increase_vs_w0"] = out["spatial_mean_AGCD"] - baseline_mean
    else:
        out["spatial_mean_AGCD"] = np.nan
        out["spatial_increase_vs_w0"] = np.nan

    out["is_selected_weight"] = np.isclose(out["CR_AGC_weight"], SELECTED_WEIGHT)

    return out.sort_values("CR_AGC_weight").reset_index(drop=True)


def scale_marker_sizes(
    values: pd.Series,
    min_size: float = 42,
    max_size: float = 135,
) -> np.ndarray:

    v = pd.to_numeric(values, errors="coerce").to_numpy(float)

    if not np.isfinite(v).any():
        return np.full(len(v), 65.0)

    finite = np.isfinite(v)
    fill_value = np.nanmedian(v[finite])
    v = np.where(finite, v, fill_value)

    v = np.clip(v, 0, None)

    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))

    if np.isclose(lo, hi):
        return np.full(len(v), 70.0)

    return min_size + (v - lo) / (hi - lo) * (max_size - min_size)


# ============================================================
# 5. Plot panels
# ============================================================
PANEL_A_XMAX = 185
def plot_panel_a(ax, sample: pd.DataFrame) -> None:
    """
    Panel a:
    Endpoint AGCD distributions.

    Improvement:
    - Mean values are integrated into the legend.
    - Separate mean text annotation is removed.
    - x-axis is slightly extended to create a clean right-side legend area.
    - Legend is placed in the low-data-density area to avoid covering histograms.
    """

    style_axes(ax)
    add_panel_label(ax, "a")

    agc_ml = sample["AGC_ML"].dropna().to_numpy(float)
    cr_agc = sample["CR_AGC"].dropna().to_numpy(float)

    bins = np.linspace(0, DIST_XMAX, 80)

    _, _, patch_agc = ax.hist(
        agc_ml,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.4,
        label=f"AGC_ML  {np.nanmean(agc_ml):.1f}",
    )

    _, _, patch_cr = ax.hist(
        cr_agc,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.4,
        label=f"CR_AGC  {np.nanmean(cr_agc):.1f}",
    )

    # Extend x-axis to create a visually empty legend area on the right.
    ax.set_xlim(0, PANEL_A_XMAX)

    ax.set_xlabel("AGCD (Mg C ha$^{-1}$)")
    ax.set_ylabel("Density")

    # Compact legend with mean values included.
    leg = ax.legend(
        handles=[patch_agc[0], patch_cr[0]],
        labels=[
            f"AGCD_ML   {np.nanmean(agc_ml):.1f}",
            f"AGCD_CR   {np.nanmean(cr_agc):.1f}",
        ],
        title="Mean AGCD",
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.6,
        borderpad=0.35,
        labelspacing=0.35,
        fontsize=8.8,
        title_fontsize=8.8,
    )

    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.88)
    leg.get_frame().set_edgecolor("none")


def plot_panel_b(ax, metrics_all: pd.DataFrame) -> None:
    style_axes(ax)
    add_panel_label(ax, "b")
    add_low_weight_band(ax)

    x = metrics_all["CR_AGC_weight"].to_numpy(float)

    ax.plot(
        x,
        metrics_all["RMSE"],
        marker="o",
        markersize=3.3,
        linewidth=1.3,
        label="RMSE",
    )

    ax.plot(
        x,
        metrics_all["AbsBias"],
        marker="^",
        markersize=3.3,
        linewidth=1.3,
        label="|Bias|",
    )

    ax.axvline(
        SELECTED_WEIGHT,
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        zorder=2,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("AGCD_CR weight")
    ax.set_ylabel("Value (Mg C ha$^{-1}$)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        SELECTED_WEIGHT,
        1.015,
        f"w={SELECTED_WEIGHT:.2f}",
        transform=ax.get_xaxis_transform(),
        fontsize=8.8,
        ha="center",
        va="bottom",
        clip_on=False,
    )

def plot_panel_c(ax, metrics_broad: pd.DataFrame) -> None:
    style_axes(ax)
    add_panel_label(ax, "c")
    add_low_weight_band(ax)

    label_map = {
        "young_<50": "Young (<50 yr)",
        "middle_50_150": "Middle-aged (50–150 yr)",
        "old_>150": "Old (≥150 yr)",
    }

    marker_map = {
        "young_<50": "o",
        "middle_50_150": "s",
        "old_>150": "^",
    }

    for age_class in ["young_<50", "middle_50_150", "old_>150"]:
        sub = metrics_broad[
            metrics_broad["age_class"] == age_class
        ].sort_values("CR_AGC_weight")

        if sub.empty:
            continue

        ax.plot(
            sub["CR_AGC_weight"],
            sub["Bias(product-observed)"],
            marker=marker_map[age_class],
            markersize=3.5,
            linewidth=1.3,
            label=label_map[age_class],
        )

    ax.axhline(0, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    ax.axvline(
        SELECTED_WEIGHT,
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        zorder=2,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_xlabel("AGCD_CR weight")
    ax.set_ylabel("Bias = product − inventory\n(Mg C ha$^{-1}$)")
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=3,
        fontsize=9.0,
        handlelength=2.0,
        columnspacing=1.3,
    )
    ax.text(
        SELECTED_WEIGHT,
        1.015,
        f"w={SELECTED_WEIGHT:.2f}",
        transform=ax.get_xaxis_transform(),
        fontsize=8.8,
        ha="center",
        va="bottom",
        clip_on=False,
    )

def plot_panel_d(ax, tradeoff_df: pd.DataFrame):
    """
    Panel d:
    Constrained RMSE–age-bias trade-off.

    No Pareto front is drawn.
    """

    style_axes(ax)
    add_panel_label(ax, "d")

    plot_df = tradeoff_df.dropna(
        subset=["RMSE", "middle_old_abs_bias_mean"]
    ).copy()

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No trade-off data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return None

    plot_df = plot_df.sort_values("CR_AGC_weight").reset_index(drop=True)

    sizes = scale_marker_sizes(plot_df["spatial_increase_vs_w0"])

    # Candidate-weight sensitivity path
    ax.plot(
        plot_df["RMSE"],
        plot_df["middle_old_abs_bias_mean"],
        linestyle="-",
        linewidth=1.15,
        color="0.68",
        alpha=0.88,
        zorder=1,
        label="Candidate-weight path",
    )

    # Candidate points
    sc = ax.scatter(
        plot_df["RMSE"],
        plot_df["middle_old_abs_bias_mean"],
        c=plot_df["CR_AGC_weight"],
        s=sizes,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.92,
        zorder=3,
    )

    # Selected weight
    selected_idx = np.argmin(
        np.abs(plot_df["CR_AGC_weight"].to_numpy(float) - SELECTED_WEIGHT)
    )
    selected = plot_df.iloc[int(selected_idx)]

    ax.scatter(
        [selected["RMSE"]],
        [selected["middle_old_abs_bias_mean"]],
        marker="*",
        s=230,
        facecolors="white",
        edgecolors="black",
        linewidths=1.35,
        zorder=8,
        label=f"Selected w={SELECTED_WEIGHT:.2f}",
    )

    ax.annotate(
        f"w={SELECTED_WEIGHT:.2f}",
        xy=(selected["RMSE"], selected["middle_old_abs_bias_mean"]),
        xytext=(7, -10),
        textcoords="offset points",
        fontsize=9.2,
        ha="left",
        va="top",
    )

    # Mark the CR_AGC-only endpoint only, to avoid clutter near w=0.
    w1_idx = np.argmin(np.abs(plot_df["CR_AGC_weight"].to_numpy(float) - 1.0))
    w1 = plot_df.iloc[int(w1_idx)]

    ax.annotate(
        "w=1",
        xy=(w1["RMSE"], w1["middle_old_abs_bias_mean"]),
        xytext=(-6, 8),
        textcoords="offset points",
        fontsize=8.8,
        ha="right",
        va="bottom",
    )

    if np.isfinite(plot_df["spatial_increase_vs_w0"]).any():
        ax.text(
            0.98,
            0.04,
            "Point size: spatial AGCD increase",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.4,
        )

    ax.set_xlabel("Overall RMSE (Mg C ha$^{-1}$)")
    ax.set_ylabel("Mean |Bias| in middle-aged\nand old forests (Mg C ha$^{-1}$)")

    ax.set_xlim(plot_df["RMSE"].min() - 0.8, plot_df["RMSE"].max() + 1.0)
    ax.set_ylim(
        plot_df["middle_old_abs_bias_mean"].min() - 0.8,
        plot_df["middle_old_abs_bias_mean"].max() + 1.0,
    )

    ax.legend(frameon=False, loc="upper left")

    return sc


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_publication_style()

    print("Preparing Fig. 3: three-panel CR–ML weight sensitivity diagnostic...")
    print("[design] Development samples only; external 811 samples are excluded from weight selection.")
    print("[design] Spatial mean AGCD increase is not used as a main-figure selection criterion.")

    sample = get_or_create_panel_a_sample()
    metrics_all = read_metrics_all()
    metrics_broad = read_metrics_broad_age()

    if metrics_all.empty:
        raise RuntimeError("No overall weight-sensitivity metrics available.")
    if metrics_broad.empty:
        raise RuntimeError("No broad-age weight-sensitivity metrics available.")

    # Export the selected-weight summary using only inventory agreement and
    # age-stratified bias. No spatial-mean AGCD increase is included.
    selected_overall = metrics_all.iloc[
        int(np.argmin(np.abs(metrics_all["CR_AGC_weight"].to_numpy(float) - SELECTED_WEIGHT)))
    ]

    selected_broad = metrics_broad[
        np.isclose(metrics_broad["CR_AGC_weight"].to_numpy(float), SELECTED_WEIGHT)
    ].copy()

    selected_summary = {
        "selected_weight": SELECTED_WEIGHT,
        "overall_RMSE": selected_overall.get("RMSE", np.nan),
        "overall_AbsBias": selected_overall.get("AbsBias", np.nan),
    }

    for age_class, prefix in [
        ("young_<50", "young"),
        ("middle_50_150", "middle"),
        ("old_>150", "old"),
    ]:
        sub = selected_broad[selected_broad["age_class"] == age_class]
        selected_summary[f"{prefix}_Bias"] = (
            float(sub.iloc[0]["Bias(product-observed)"]) if not sub.empty else np.nan
        )

    selected_summary_path = os.path.join(
        OUT_DIR,
        "fig2_selected_weight_inventory_age_bias_summary.csv",
    )
    pd.DataFrame([selected_summary]).to_csv(
        selected_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[write] Selected-weight summary: {selected_summary_path}")

    # Three-panel main figure. Panel c spans the full lower row so that the
    # three age-stage trajectories can be read without crowding.
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        dpi=300,
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.05],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_panel_a(ax_a, sample)
    plot_panel_b(ax_b, metrics_all)
    plot_panel_c(ax_c, metrics_broad)

    png_path = os.path.join(OUT_DIR, "Fig2_CRML_weight_sensitivity_3panel.png")
    pdf_path = os.path.join(OUT_DIR, "Fig2_CRML_weight_sensitivity_3panel.pdf")
    svg_path = os.path.join(OUT_DIR, "Fig2_CRML_weight_sensitivity_3panel.svg")

    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print("Done.")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"SVG: {svg_path}")

    print("\nRecommended manuscript interpretation:")
    print("- Panel a shows the two endpoint AGCD distributions.")
    print("- Panel b shows overall inventory RMSE and |Bias| across candidate CR_AGC weights.")
    print("- Panel c shows young, middle-aged and old forest bias responses across weights.")
    print("- w=0.10 is interpreted as a low-weight compromise based on inventory error and age-stratified bias.")
    print("- Spatial mean AGCD increase is not used to justify the selected weight in the main figure.")


if __name__ == "__main__":
    main()
