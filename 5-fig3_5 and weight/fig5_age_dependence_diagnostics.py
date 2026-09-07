# -*- coding: utf-8 -*-
"""
Fig. 5: Extended age-dependence diagnostics using the full forest-inventory database

Scientific scope
--------
1. This figure combines development and 811 external independent inventory records to obtain broader forest-age coverage.
2. Its primary purpose is to diagnose age-dependent residuals of remote-sensing products and DGVMs relative to forest-inventory observations.
3. CR-ML is shown only as a model-reference trajectory. Because the full inventory includes model-development samples,
   this figure must not be interpreted as an independent evaluation of CR-ML predictive performance.
4. External independent agreement of CR-ML is reported separately using the 811-sample validation figure.
5. The former "Relative difference from CR-ML" panel has been removed.

Four-panel structure
----------
(a) Mean AGCD by 20-year forest-age class
    Age trajectories for inventory, CR-ML, individual remote-sensing products, the remote-sensing ensemble, individual DGVMs, and the DGVM ensemble on a common full-inventory sample.

(b) Inventory residual bias by 20-year forest-age class
    Bias = product - inventory. Remote-sensing products and DGVMs are the primary diagnostic targets; CR-ML is a background reference.

(c) External-product diagnostic summary
    x = overall RMSE; y = residual-age slope per 20 yr.
    External remote-sensing products and DGVMs are summarized without treating CR-ML as part of a full-inventory performance ranking.

(d) Broad-age inventory bias
    Horizontal facets show bias and 1-degree spatial-block bootstrap 95% confidence intervals for individual remote-sensing products, the remote-sensing ensemble,
    individual DGVM outputs, and the DGVM ensemble across young, middle-aged, and old developmental stages.

Runtime dependencies
--------
- Place this script in the same directory as the main analysis script below, or edit MAIN_ANALYSIS_SCRIPT:
  age20_weight_sensitivity_separated_development_external811.py
- The main analysis script provides consistent inventory preprocessing, sample separation, raster extraction, unit conversion, and the CR-ML weighting formula.
- pip install numpy pandas matplotlib scipy rasterio xlrd openpyxl
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import linregress


# ============================================================
# 1. User settings
# ============================================================

# Main analysis script. Resolution checks the current working directory and this script directory.
MAIN_ANALYSIS_SCRIPT = "01_weight_sensitivity_and_external_validation.py"

# Output directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("CRML_PROJECT_ROOT", SCRIPT_DIR))
OUTPUT_ROOT = Path(os.environ.get("CRML_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
OUT_DIR = str(OUTPUT_ROOT / "fig7_age_dependence_diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

# Survey-year scenario:
# - all_years
# - survey_2000_2021
# - survey_2010_2021
SCENARIO = "all_years"
SCENARIO_RANGES = {
    "all_years": (None, None),
    "survey_2000_2021": (2000, 2021),
    "survey_2010_2021": (2010, 2021),
}

# Final CR_AGC weight.
SELECTED_CR_WEIGHT = 0.10

# Cache settings. The first run extracts all products; subsequent runs can reuse the cache.
FORCE_REBUILD_FULL_INVENTORY_CACHE = False
CACHE_DIR = os.path.join(OUT_DIR, "full_inventory_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
FULL_INVENTORY_CACHE_CSV = os.path.join(
    CACHE_DIR,
    "full_inventory_with_all_products_and_weighted_CRML.csv",
)

# Whether the main age-trajectory and bias panels use a common sample.
# True prevents differences in age/spatial sample composition from affecting curve comparisons.
USE_COMMON_SAMPLE_FOR_AGE_PANELS = True

# Whether panel c uses a common valid sample across all external products.
# True supports fair cross-product comparison. Set to False if the common sample becomes too small and report product-specific N values.
USE_COMMON_SAMPLE_FOR_EXTERNAL_SUMMARY = True

# Whether the remote-sensing ensemble requires all component products to be valid.
# True: compute the ensemble mean only when ESACCI, Thurner, and ICESat-2 are all valid.
# False: compute a mean when at least one component is valid; ensemble membership may then vary by sample.
REQUIRE_ALL_REMOTE_SENSING_PRODUCTS = True

# Minimum sample size per 20-year age class. Classes below this threshold are not plotted or connected.
MIN_N_PER_AGE_CLASS = 20

# Minimum sample size per broad age class.
MIN_N_PER_BROAD_AGE_CLASS = 20

# Spatial-block bootstrap settings. A 1-degree grid is used to approximate spatial clustering control.
BOOTSTRAP_REPS = 1000
BOOTSTRAP_GRID_DEGREES = 1.0
RANDOM_SEED = 42
CI_LEVEL = 0.95

# Panel c: CR-ML is a model reference and is compared with external remote-sensing/DGVM products within the same
# age-diagnostic sample framework using residual-age slopes. This panel is not interpreted as an independent performance ranking.
PANEL_C_PRODUCTS = [
    "CR_ML",
    "ESACCI",
    "thurner",
    "ICESAT2",
    "Remote sensing mean",
    "ORCHIDEE",
    "LPJGUESS",
    "DGVM ensemble",
]

# External products used in panel d and for external-product grouping in panel c.
EXTERNAL_PRODUCTS_FOR_SUMMARY = [
    "ESACCI",
    "thurner",
    "ICESAT2",
    "Remote sensing mean",
    "ORCHIDEE",
    "LPJGUESS",
    "DGVM ensemble",
]

# Products shown by broad age group in panel d, including individual remote-sensing products, individual DGVMs, and their ensembles.
BROAD_AGE_PRODUCTS = [
    "ESACCI",
    "thurner",
    "ICESAT2",
    "Remote sensing mean",
    "ORCHIDEE",
    "LPJGUESS",
    "DGVM ensemble",
]

# Series shown in the main age-trajectory panel.
AGE_PANEL_SERIES = [
    "Inventory",
    "CR_ML",
    "ESACCI",
    "thurner",
    "ICESAT2",
    "Remote sensing mean",
    "ORCHIDEE",
    "LPJGUESS",
    "DGVM ensemble",
]

# Residual series in panel b. AGC_ML is included to show that without the age constraint,
# the ML-only estimate also exhibits pronounced age-dependent residuals.
BIAS_PANEL_SERIES = [
    "CR_ML",
    "AGC_ML",
    "ESACCI",
    "thurner",
    "ICESAT2",
    "Remote sensing mean",
    "ORCHIDEE",
    "LPJGUESS",
    "DGVM ensemble",
]

# Figure settings
FIG_WIDTH_IN = 14 / 2.54
FIG_HEIGHT_IN = 17.0 / 2.54
SHOW_AGE_CLASS_N = False

# Panel a/b visual hierarchy:
# - Inventory / CR–ML / ensemble means are the primary trajectories.
# - AGCD_ML is additionally emphasized in panel b.
# - Individual remote-sensing and DGVM products remain visible as subdued background lines.
PRIMARY_LINEWIDTH = 1.65
PRIMARY_MARKERSIZE = 3.80
INVENTORY_LINEWIDTH = 1.75
INVENTORY_MARKERSIZE = 4.00
CRML_LINEWIDTH = 2.05
CRML_MARKERSIZE = 4.30
SECONDARY_LINEWIDTH = 0.75
SECONDARY_ALPHA = 0.72
PRIMARY_ALPHA = 1.0
PRIMARY_CI_ALPHA = 0.045
REFERENCE_CI_ALPHA = 0.060
ANNOTATION_COLOR = "0.32"  # neutral dark gray for text annotations


# ============================================================
# 2. Product names and plotting styles
# ============================================================

PRODUCT_COLORS = {
    "Inventory": "#000000",
    "CR_ML": "#C44E00",
    # Keep AGCD_ML visually distinct from ICESat-2 in panel b.
    "AGC_ML": "#2F5597",
    "Remote sensing mean": "#1B9E77",
    "DGVM ensemble": "#8E63B0",
    # Individual products use softer colors because they are background trajectories in a/b.
    "ESACCI": "#66A765",
    "thurner": "#D7AA35",
    "ICESAT2": "#72A6CF",
    "ORCHIDEE": "#B28A4A",
    "LPJGUESS": "#8A8A8A",
}

PRODUCT_LABELS = {
    "Inventory": "Inventory",
    "CR_ML": "CR–ML",
    "AGC_ML": "AGCD_ML",
    "Remote sensing mean": "Remote sensing mean",
    "DGVM ensemble": "DGVM ensemble",
    "ESACCI": "ESA CCI",
    "thurner": "Thurner",
    "ICESAT2": "ICESat-2",
    "ORCHIDEE": "ORCHIDEE",
    "LPJGUESS": "LPJ-GUESS",
}

PRODUCT_MARKERS = {
    "Inventory": "o",
    "CR_ML": "o",
    "AGC_ML": "s",
    "Remote sensing mean": "^",
    "DGVM ensemble": "D",
    "ESACCI": "o",
    "thurner": "s",
    "ICESAT2": "^",
    "ORCHIDEE": "D",
    "LPJGUESS": "P",
}

PRODUCT_LINESTYLES = {
    "Inventory": (0, (3, 2)),
    "CR_ML": "-",
    "AGC_ML": "-.",
    "ESACCI": (0, (5, 2)),
    "thurner": (0, (1, 1)),
    "ICESAT2": (0, (4, 1, 1, 1)),
    "Remote sensing mean": "--",
    "ORCHIDEE": (0, (6, 2)),
    "LPJGUESS": (0, (2, 1)),
    "DGVM ensemble": ":",
}


# ============================================================
# 3. General utilities
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
        "font.size": 9.2,
        "axes.labelsize": 9.8,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 9.0,
        "axes.unicode_minus": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8, width=0.8, pad=2.0)
    ax.xaxis.labelpad = 4.0
    ax.yaxis.labelpad = 4.0
    ax.margins(x=0.02, y=0.08)
    ax.grid(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        f"({label})",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-30, 4),
        textcoords="offset points",
        fontsize=12.8,
        va="top",
        ha="left",
    )


def script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def resolve_file(path: str) -> str:
    if path and os.path.exists(path):
        return path

    basename = re.split(r"[\\/]", str(path))[-1]
    candidates = [
        Path.cwd() / basename,
        script_dir() / basename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return path


def load_main_analysis_module():
    module_path = resolve_file(MAIN_ANALYSIS_SCRIPT)
    if not os.path.exists(module_path):
        raise FileNotFoundError(
            "Main analysis script not found. Place both scripts in the same directory or edit "
            f"MAIN_ANALYSIS_SCRIPT.\nCurrent setting: {MAIN_ANALYSIS_SCRIPT}\n"
            f"Resolved path: {module_path}"
        )

    spec = importlib.util.spec_from_file_location("crml_separated_main", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load the main analysis script: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print(f"[load] Main analysis module: {module_path}")
    return module


def display_age_label(label: str) -> str:
    text = str(label)
    return text.replace("-", "–")


def broad_age_display(label: str) -> str:
    mapping = {
        "young_<50": "Young\n(<50 yr)",
        "middle_50_150": "Middle-aged\n(50–150 yr)",
        "old_>150": "Old\n(≥150 yr)",
    }
    return mapping.get(str(label), str(label))


def add_spatial_block_id(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
    out = df.copy()
    lon = pd.to_numeric(out[lon_col], errors="coerce")
    lat = pd.to_numeric(out[lat_col], errors="coerce")

    lon_bin = np.floor((lon + 180.0) / BOOTSTRAP_GRID_DEGREES).astype("Int64")
    lat_bin = np.floor((lat + 90.0) / BOOTSTRAP_GRID_DEGREES).astype("Int64")
    out["spatial_block"] = lon_bin.astype("string") + "_" + lat_bin.astype("string")
    return out


def percentile_ci(values: np.ndarray, ci_level: float = CI_LEVEL) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    alpha = (1.0 - ci_level) / 2.0
    return (
        float(np.quantile(v, alpha)),
        float(np.quantile(v, 1.0 - alpha)),
    )


def cluster_bootstrap_stat(
    df: pd.DataFrame,
    value_col: str,
    statistic: str,
    rng: np.random.Generator,
    reps: int = BOOTSTRAP_REPS,
    age_col: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    1-degree spatial-block bootstrap.

    statistic:
      - mean
      - rmse: value_col must contain residuals
      - slope_per_20yr: value_col must contain residuals and age_col is required
    """
    required = ["spatial_block", value_col]
    if statistic == "slope_per_20yr":
        if age_col is None:
            raise ValueError("slope_per_20yr requires age_col.")
        required.append(age_col)

    sub = df.dropna(subset=required).copy()
    if sub.empty:
        return np.nan, np.nan, np.nan

    block_codes, block_names = pd.factorize(sub["spatial_block"], sort=True)
    n_blocks = len(block_names)
    if n_blocks < 2:
        return np.nan, np.nan, np.nan

    value = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(float)
    age = None
    if age_col is not None:
        age = pd.to_numeric(sub[age_col], errors="coerce").to_numpy(float)

    def weighted_stat(block_multiplicity: np.ndarray) -> float:
        weights = block_multiplicity[block_codes].astype(float)
        valid = np.isfinite(value) & (weights > 0)
        if age is not None:
            valid &= np.isfinite(age)
        if valid.sum() < 3:
            return np.nan

        y = value[valid]
        w = weights[valid]

        if statistic == "mean":
            return float(np.average(y, weights=w))
        if statistic == "rmse":
            return float(np.sqrt(np.average(y ** 2, weights=w)))
        if statistic == "slope_per_20yr":
            assert age is not None
            x = age[valid]
            if np.unique(x).size < 3:
                return np.nan
            x_bar = np.average(x, weights=w)
            y_bar = np.average(y, weights=w)
            denominator = np.sum(w * (x - x_bar) ** 2)
            if denominator <= 0:
                return np.nan
            slope = np.sum(w * (x - x_bar) * (y - y_bar)) / denominator
            return float(slope * 20.0)
        raise ValueError(f"Unknown statistic: {statistic}")

    # Point estimates are unweighted.
    ones = np.ones(n_blocks, dtype=int)
    estimate = weighted_stat(ones)

    bootstrap_values = np.full(reps, np.nan, dtype=float)
    for i in range(reps):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        multiplicity = np.bincount(sampled, minlength=n_blocks)
        bootstrap_values[i] = weighted_stat(multiplicity)

    ci_low, ci_high = percentile_ci(bootstrap_values)
    return estimate, ci_low, ci_high


# ============================================================
# 4. Full-inventory construction and product extraction
# ============================================================

def prepare_full_inventory_with_products(main_module) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct the union of development and 811-sample inventory records, then extract all products for the full sample.

    Unlike independent-validation separation, development records are not removed by coordinate here because observations from different survey years
    or repeated measurements at the same location contain useful age-diagnostic information. The workflow only prevents the same inventory record
    already present in the development file from being appended again from the 811 file.

    Returns:
      full_products: full inventory sample with all products
      audit_summary: sample-construction audit table
    """
    if os.path.exists(FULL_INVENTORY_CACHE_CSV) and not FORCE_REBUILD_FULL_INVENTORY_CACHE:
        print(f"[read cache] {FULL_INVENTORY_CACHE_CSV}")
        cached = pd.read_csv(FULL_INVENTORY_CACHE_CSV)
        audit_path = os.path.join(CACHE_DIR, "full_inventory_construction_audit.csv")
        audit = pd.read_csv(audit_path) if os.path.exists(audit_path) else pd.DataFrame()
        return cached, audit

    print("[step] Read development and 811 external inventory files.")
    development_raw = main_module.prepare_inventory_data(
        main_module.DEVELOPMENT_INVENTORY_FILE,
        sheet_name=main_module.DEVELOPMENT_INVENTORY_SHEET_NAME,
    ).copy()
    external_raw = main_module.prepare_inventory_data(
        main_module.INDEPENDENT_VALIDATION_FILE,
        sheet_name=main_module.INDEPENDENT_INVENTORY_SHEET_NAME,
    ).copy()

    # Construct record-level matching keys. The full age diagnostic removes only duplicate appends of the same inventory record,
    # and does not remove other survey years or observations at the same coordinates.
    development_raw["_record_key"] = main_module.build_record_overlap_key(development_raw)
    development_raw["_record_key_no_year"] = (
        main_module.build_record_overlap_key_without_year(development_raw)
    )
    external_raw["_record_key"] = main_module.build_record_overlap_key(external_raw)
    external_raw["_record_key_no_year"] = (
        main_module.build_record_overlap_key_without_year(external_raw)
    )

    dev_full_keys = set(development_raw["_record_key"].dropna().astype(str))
    dev_no_year_keys = set(development_raw["_record_key_no_year"].dropna().astype(str))

    external_full_match = external_raw["_record_key"].astype(str).isin(dev_full_keys)
    external_no_year_match = (
        ~external_full_match
        & external_raw["_record_key_no_year"].astype(str).isin(dev_no_year_keys)
    )
    external_already_present = external_full_match | external_no_year_match
    external_to_append = external_raw.loc[~external_already_present].copy()

    development_raw["sample_source_for_full_diagnosis"] = "development_or_full_inventory_file"
    external_to_append["sample_source_for_full_diagnosis"] = "external811_appended"

    helper_cols = ["_record_key", "_record_key_no_year"]
    full_inventory = pd.concat(
        [
            development_raw.drop(columns=helper_cols, errors="ignore"),
            external_to_append.drop(columns=helper_cols, errors="ignore"),
        ],
        ignore_index=True,
        sort=False,
    )
    full_inventory["full_inventory_row_id"] = np.arange(len(full_inventory), dtype=int)

    duplicate_cols = [
        main_module.LON_COL,
        main_module.LAT_COL,
        "Age_used",
        "AGCD_obs",
        "Survey_year",
    ]
    duplicate_mask = full_inventory.duplicated(subset=duplicate_cols, keep=False)

    audit_summary = pd.DataFrame([{
        "development_or_full_file_qc_N": int(len(development_raw)),
        "external811_qc_N": int(len(external_raw)),
        "external_records_already_present_full_key": int(external_full_match.sum()),
        "external_records_already_present_without_year": int(external_no_year_match.sum()),
        "external_records_appended": int(len(external_to_append)),
        "full_inventory_union_N": int(len(full_inventory)),
        "possible_exact_duplicate_rows_in_union": int(duplicate_mask.sum()),
        "development_file": main_module.DEVELOPMENT_INVENTORY_FILE,
        "external_file": main_module.INDEPENDENT_VALIDATION_FILE,
        "union_rule": (
            "Append only external records not already matched by coordinates, age, "
            "AGCD and survey year; do not remove coordinate-only repeated observations."
        ),
        "interpretation": (
            "Full inventory is used for extended age-dependent diagnosis; "
            "not for independent validation of CR-ML."
        ),
    }])

    audit_path = os.path.join(CACHE_DIR, "full_inventory_construction_audit.csv")
    audit_summary.to_csv(audit_path, index=False, encoding="utf-8-sig")
    external_raw.loc[external_already_present].drop(
        columns=helper_cols,
        errors="ignore",
    ).to_csv(
        os.path.join(CACHE_DIR, "external811_records_already_in_development_file.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    external_to_append.drop(columns=helper_cols, errors="ignore").to_csv(
        os.path.join(CACHE_DIR, "external811_records_appended_to_full_inventory.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if duplicate_mask.any():
        full_inventory.loc[duplicate_mask].to_csv(
            os.path.join(CACHE_DIR, "possible_exact_duplicates_in_full_inventory_union.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        warnings.warn(
            f"Detected {int(duplicate_mask.sum())} potentially exact duplicate records in the full-inventory union."
            "They are not removed automatically; inspect the audit file in the cache directory."
        )

    print("[step] Extract AGC_ML, CR_AGC, remote-sensing, and DGVM products for the full inventory sample.")
    full_products, extraction_summary, formula_summary, weighted_cols = (
        main_module._extract_and_build_weighted_products(
            inventory=full_inventory,
            raster_paths=main_module.PRODUCT_RASTER_PATHS,
            role_out_dir=CACHE_DIR,
            file_prefix="full_inventory_extended_diagnosis",
        )
    )

    selected_col = main_module.weight_to_colname(SELECTED_CR_WEIGHT)
    if selected_col not in weighted_cols or selected_col not in full_products.columns:
        raise KeyError(
            f"Selected weighted column was not generated: {selected_col}."
            f"Available weighted columns: {weighted_cols}"
        )

    full_products.to_csv(
        FULL_INVENTORY_CACHE_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    extraction_summary.to_csv(
        os.path.join(CACHE_DIR, "full_inventory_product_extraction_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    formula_summary.to_csv(
        os.path.join(CACHE_DIR, "full_inventory_weighted_formula_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[write cache] {FULL_INVENTORY_CACHE_CSV}")
    return full_products, audit_summary

def harmonize_analysis_columns(df: pd.DataFrame, main_module) -> pd.DataFrame:
    out = df.copy()

    selected_col = main_module.weight_to_colname(SELECTED_CR_WEIGHT)
    required = [
        "AGCD_obs",
        "Age_used",
        "age20_class",
        "broad_age_class",
        main_module.LON_COL,
        main_module.LAT_COL,
        "AGC_ML",
        selected_col,
        "ESACCI",
        "thurner",
        "ICESAT2",
        "DGVM",
        "ORCHIDEE",
        "LPJGUESS",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(
            "The full-inventory product table is missing required fields:\n" + "\n".join(missing)
        )

    out["Inventory"] = pd.to_numeric(out["AGCD_obs"], errors="coerce")
    out["CR_ML"] = pd.to_numeric(out[selected_col], errors="coerce")
    out["AGC_ML"] = pd.to_numeric(out["AGC_ML"], errors="coerce")
    out["DGVM ensemble"] = pd.to_numeric(out["DGVM"], errors="coerce")

    remote_cols = ["ESACCI", "thurner", "ICESAT2"]
    remote_values = out[remote_cols].apply(pd.to_numeric, errors="coerce")
    if REQUIRE_ALL_REMOTE_SENSING_PRODUCTS:
        remote_valid = remote_values.notna().all(axis=1)
        out["Remote sensing mean"] = remote_values.mean(axis=1).where(remote_valid)
        out["remote_sensing_member_count"] = remote_values.notna().sum(axis=1)
    else:
        out["Remote sensing mean"] = remote_values.mean(axis=1, skipna=True)
        out["remote_sensing_member_count"] = remote_values.notna().sum(axis=1)
        out.loc[out["remote_sensing_member_count"] == 0, "Remote sensing mean"] = np.nan

    for col in ["ESACCI", "thurner", "ICESAT2", "ORCHIDEE", "LPJGUESS"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = add_spatial_block_id(
        out,
        lon_col=main_module.LON_COL,
        lat_col=main_module.LAT_COL,
    )
    return out


def filter_scenario(df: pd.DataFrame, main_module) -> pd.DataFrame:
    if SCENARIO not in SCENARIO_RANGES:
        raise ValueError(
            f"Unknown SCENARIO={SCENARIO}. Available options: {list(SCENARIO_RANGES)}"
        )
    year_min, year_max = SCENARIO_RANGES[SCENARIO]
    out = main_module.filter_by_survey_year(df, year_min, year_max)
    print(
        f"[scenario] {SCENARIO}: year_min={year_min}, "
        f"year_max={year_max}, N={len(out)}"
    )
    return out


# ============================================================
# 5. Statistical summaries
# ============================================================

def make_age_panel_sample(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Age_used", "age20_class", "spatial_block"]
    if USE_COMMON_SAMPLE_FOR_AGE_PANELS:
        required.extend(AGE_PANEL_SERIES)
    sample = df.dropna(subset=required).copy()
    return sample


def summarize_age20_means(
    sample: pd.DataFrame,
    age_order: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for age_class in age_order:
        age_sub = sample[sample["age20_class"].astype(str) == str(age_class)].copy()
        for product in AGE_PANEL_SERIES:
            product_sub = age_sub.dropna(subset=[product]).copy()
            n = len(product_sub)
            if n < MIN_N_PER_AGE_CLASS:
                estimate = ci_low = ci_high = np.nan
            else:
                estimate, ci_low, ci_high = cluster_bootstrap_stat(
                    product_sub,
                    value_col=product,
                    statistic="mean",
                    rng=rng,
                )
            rows.append({
                "age20_class": str(age_class),
                "product": product,
                "N": int(n),
                "mean_AGCD": estimate,
                "CI_low": ci_low,
                "CI_high": ci_high,
            })
    return pd.DataFrame(rows)


def summarize_age20_bias(
    sample: pd.DataFrame,
    age_order: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for age_class in age_order:
        age_sub = sample[sample["age20_class"].astype(str) == str(age_class)].copy()
        for product in BIAS_PANEL_SERIES:
            sub = age_sub.dropna(subset=["Inventory", product]).copy()
            sub["residual"] = sub[product] - sub["Inventory"]
            n = len(sub)
            if n < MIN_N_PER_AGE_CLASS:
                estimate = ci_low = ci_high = np.nan
            else:
                estimate, ci_low, ci_high = cluster_bootstrap_stat(
                    sub,
                    value_col="residual",
                    statistic="mean",
                    rng=rng,
                )
            rows.append({
                "age20_class": str(age_class),
                "product": product,
                "N": int(n),
                "Bias(product-inventory)": estimate,
                "CI_low": ci_low,
                "CI_high": ci_high,
            })
    return pd.DataFrame(rows)


def panel_c_product_sample(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Return samples used for Panel c residual–age slopes.

    When common-sample mode is enabled, CR–ML and all external products are
    evaluated on the same inventory/age/spatial-block sample. This makes the
    slope magnitudes directly comparable within the age-diagnostic dataset,
    while CR–ML remains explicitly labelled as a model reference rather than
    an independent validation result.
    """
    base_required = ["Inventory", "Age_used", "spatial_block"]
    product_specific: Dict[str, pd.DataFrame] = {}

    if USE_COMMON_SAMPLE_FOR_EXTERNAL_SUMMARY:
        required = base_required + list(PANEL_C_PRODUCTS)
        common = df.dropna(subset=required).copy()
        for product in PANEL_C_PRODUCTS:
            product_specific[product] = common
        return common, product_specific

    common = pd.DataFrame()
    for product in PANEL_C_PRODUCTS:
        product_specific[product] = df.dropna(
            subset=base_required + [product]
        ).copy()
    return common, product_specific


def summarize_external_product_performance(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common, product_samples = panel_c_product_sample(df)
    rows: List[Dict] = []

    for product in PANEL_C_PRODUCTS:
        sub = product_samples[product].copy()
        sub["residual"] = sub[product] - sub["Inventory"]
        sub = sub.dropna(subset=["residual", "Age_used", "spatial_block"])
        n = len(sub)
        n_blocks = sub["spatial_block"].nunique()

        if n < max(MIN_N_PER_AGE_CLASS, 10) or n_blocks < 2:
            rmse = rmse_low = rmse_high = np.nan
            slope = slope_low = slope_high = np.nan
            bias = np.nan
            r = np.nan
            p = np.nan
        else:
            rmse, rmse_low, rmse_high = cluster_bootstrap_stat(
                sub,
                value_col="residual",
                statistic="rmse",
                rng=rng,
            )
            slope, slope_low, slope_high = cluster_bootstrap_stat(
                sub,
                value_col="residual",
                statistic="slope_per_20yr",
                age_col="Age_used",
                rng=rng,
            )
            bias = float(sub["residual"].mean())
            if sub["Age_used"].nunique() >= 3:
                reg = linregress(
                    pd.to_numeric(sub["Age_used"], errors="coerce"),
                    pd.to_numeric(sub["residual"], errors="coerce"),
                )
                r = float(reg.rvalue)
                p = float(reg.pvalue)
            else:
                r = p = np.nan

        rows.append({
            "product": product,
            "N": int(n),
            "N_spatial_blocks": int(n_blocks),
            "RMSE": rmse,
            "RMSE_CI_low": rmse_low,
            "RMSE_CI_high": rmse_high,
            "Bias(product-inventory)": bias,
            "residual_age_slope_per_20yr": slope,
            "slope_CI_low": slope_low,
            "slope_CI_high": slope_high,
            "linear_r_residual_vs_age": r,
            "linear_p_residual_vs_age": p,
            "sample_mode": (
                "panel_c_all_products_common_sample"
                if USE_COMMON_SAMPLE_FOR_EXTERNAL_SUMMARY
                else "panel_c_product_specific_sample"
            ),
            "panel_c_role": (
                "model_reference" if product == "CR_ML" else "external_product"
            ),
        })

    return pd.DataFrame(rows), common


def summarize_broad_age_bias(
    df: pd.DataFrame,
    broad_order: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict] = []

    # Panel d requires both ensembles to be valid by default so developmental-stage comparisons use the same sample set.
    required = [
        "Inventory",
        "Age_used",
        "broad_age_class",
        "spatial_block",
    ] + list(BROAD_AGE_PRODUCTS)
    sample = df.dropna(subset=required).copy()

    for age_class in broad_order:
        age_sub = sample[
            sample["broad_age_class"].astype(str) == str(age_class)
        ].copy()
        for product in BROAD_AGE_PRODUCTS:
            sub = age_sub.dropna(subset=[product, "Inventory"]).copy()
            sub["residual"] = sub[product] - sub["Inventory"]
            n = len(sub)
            if n < MIN_N_PER_BROAD_AGE_CLASS:
                estimate = ci_low = ci_high = np.nan
                rmse = np.nan
            else:
                estimate, ci_low, ci_high = cluster_bootstrap_stat(
                    sub,
                    value_col="residual",
                    statistic="mean",
                    rng=rng,
                )
                rmse = float(np.sqrt(np.mean(sub["residual"] ** 2)))
            rows.append({
                "broad_age_class": str(age_class),
                "product": product,
                "N": int(n),
                "Bias(product-inventory)": estimate,
                "Bias_CI_low": ci_low,
                "Bias_CI_high": ci_high,
                "RMSE": rmse,
            })
    return pd.DataFrame(rows)


# ============================================================
# 6. Plotting functions
# ============================================================

def make_product_legend_handles(products: Sequence[str]) -> List[Line2D]:
    """Create legend handles consistent with the visual hierarchy in panels a/b."""
    handles: List[Line2D] = []
    secondary_products = {
        "ESACCI", "thurner", "ICESAT2", "ORCHIDEE", "LPJGUESS",
    }

    for product in products:
        if product == "CR_ML":
            lw = CRML_LINEWIDTH
            ms = CRML_MARKERSIZE
            marker = PRODUCT_MARKERS.get(product, "o")
            alpha = 1.0
        elif product == "Inventory":
            lw = INVENTORY_LINEWIDTH
            ms = INVENTORY_MARKERSIZE
            marker = PRODUCT_MARKERS.get(product, "o")
            alpha = 1.0
        elif product in secondary_products:
            lw = SECONDARY_LINEWIDTH
            ms = 0.0
            marker = None
            # Slightly stronger in the legend than in the panels so labels remain identifiable.
            alpha = 0.72
        else:
            lw = PRIMARY_LINEWIDTH
            ms = PRIMARY_MARKERSIZE
            marker = PRODUCT_MARKERS.get(product, "o")
            alpha = 1.0

        handles.append(Line2D(
            [0], [0],
            color=PRODUCT_COLORS.get(product, "0.4"),
            marker=marker,
            linestyle=PRODUCT_LINESTYLES.get(product, "-"),
            linewidth=lw,
            markersize=ms,
            alpha=alpha,
            label=PRODUCT_LABELS.get(product, product),
        ))
    return handles


def add_shared_ab_legend(fig: plt.Figure) -> None:
    """Add one complete shared legend for panels a and b.

    The first row contains the primary diagnostic trajectories; the second row
    contains individual remote-sensing and DGVM products.
    """
    products = [
        "Inventory", "CR_ML", "AGC_ML",
        "Remote sensing mean", "DGVM ensemble",
        "ESACCI", "thurner", "ICESAT2",
        "ORCHIDEE", "LPJGUESS",
    ]
    handles = make_product_legend_handles(products)
    fig.legend(
        handles=handles,
        labels=[PRODUCT_LABELS.get(p, p) for p in products],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=5,
        frameon=False,
        fontsize=8.9,
        handlelength=2.20,
        handletextpad=0.45,
        columnspacing=0.95,
        borderaxespad=0.0,
        labelspacing=0.62,
    )


def apply_data_driven_ylim(
    ax: plt.Axes,
    summary: pd.DataFrame,
    products: Sequence[str],
    value_col: str,
    low_col: str,
    high_col: str,
    symmetric_zero: bool = False,
) -> None:
    """Tighten the y-axis range automatically using panel data to improve visual comparability."""
    tmp = summary[summary["product"].isin(list(products))].copy()
    if tmp.empty:
        return

    val = pd.to_numeric(tmp[value_col], errors="coerce")
    low = pd.to_numeric(tmp[low_col], errors="coerce")
    high = pd.to_numeric(tmp[high_col], errors="coerce")
    arr_min = np.nanmin(np.where(np.isfinite(low), low, val))
    arr_max = np.nanmax(np.where(np.isfinite(high), high, val))
    if not np.isfinite(arr_min) or not np.isfinite(arr_max):
        return

    if symmetric_zero:
        extent = max(abs(arr_min), abs(arr_max))
        pad = max(extent * 0.10, 2.0)
        ax.set_ylim(-(extent + pad), extent + pad)
    else:
        pad = max((arr_max - arr_min) * 0.10, 3.0)
        ax.set_ylim(arr_min - pad, arr_max + pad)


def plot_age_lines(
    ax: plt.Axes,
    summary: pd.DataFrame,
    value_col: str,
    ci_low_col: str,
    ci_high_col: str,
    products: Sequence[str],
    age_order: Sequence[str],
    ylabel: str,
    title: str,
    panel_label: str,
    horizontal_zero: bool = False,
    show_n: bool = False,
    grouped_legend: bool = True,
) -> None:
    """Plot panels a/b with a deliberate visual hierarchy.

    Primary series:
      - Inventory, CR–ML, Remote sensing mean, DGVM ensemble.
      - AGCD_ML is also primary in panel b because it diagnoses the effect of
        adding the age constraint to ML-only estimates.

    Secondary series:
      - Individual remote-sensing and DGVM products.
      - Drawn as thin, semi-transparent lines without markers or CI ribbons.

    This keeps the individual-product information while making the shared
    age-dependent pattern readable at the final 14-cm publication width.
    """
    style_axes(ax)
    add_panel_label(ax, panel_label)

    x = np.arange(len(age_order), dtype=float)

    primary_products = {
        "Inventory",
        "CR_ML",
        "Remote sensing mean",
        "DGVM ensemble",
    }
    if panel_label == "b":
        primary_products.add("AGC_ML")

    secondary_products = {
        "ESACCI",
        "thurner",
        "ICESAT2",
        "ORCHIDEE",
        "LPJGUESS",
    }

    # Draw the secondary products first so the main diagnostic curves remain on top.
    draw_order = (
        [p for p in products if p in secondary_products]
        + [p for p in products if p not in secondary_products]
    )

    for product in draw_order:
        tmp = (
            summary[summary["product"] == product]
            .set_index("age20_class")
            .reindex(age_order)
        )
        y = pd.to_numeric(tmp[value_col], errors="coerce").to_numpy(float)
        low = pd.to_numeric(tmp[ci_low_col], errors="coerce").to_numpy(float)
        high = pd.to_numeric(tmp[ci_high_col], errors="coerce").to_numpy(float)

        color = PRODUCT_COLORS.get(product, "0.4")
        linestyle = PRODUCT_LINESTYLES.get(product, "-")

        if product in secondary_products:
            linewidth = SECONDARY_LINEWIDTH
            markersize = 0.0
            marker = None
            alpha = SECONDARY_ALPHA
            zorder = 2
        elif product == "Inventory":
            linewidth = INVENTORY_LINEWIDTH
            markersize = INVENTORY_MARKERSIZE
            marker = PRODUCT_MARKERS.get(product, "o")
            alpha = 1.0
            zorder = 7
        elif product == "CR_ML":
            linewidth = CRML_LINEWIDTH
            markersize = CRML_MARKERSIZE
            marker = PRODUCT_MARKERS.get(product, "o")
            alpha = 1.0
            zorder = 6
        else:
            linewidth = PRIMARY_LINEWIDTH
            markersize = PRIMARY_MARKERSIZE
            marker = PRODUCT_MARKERS.get(product, "o")
            alpha = PRIMARY_ALPHA
            zorder = 5

        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=markersize,
            alpha=alpha,
            label=PRODUCT_LABELS.get(product, product),
            zorder=zorder,
        )

        # CI ribbons are retained only for the primary trajectories.
        valid_ci = np.isfinite(low) & np.isfinite(high)
        if valid_ci.any() and product not in secondary_products:
            if product in {"Inventory", "CR_ML"}:
                ci_alpha = REFERENCE_CI_ALPHA
            else:
                ci_alpha = PRIMARY_CI_ALPHA

            ax.fill_between(
                x,
                low,
                high,
                where=valid_ci,
                color=color,
                alpha=ci_alpha,
                linewidth=0,
                zorder=1,
            )

    if horizontal_zero:
        ax.axhline(
            0,
            color="0.32",
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            zorder=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_age_label(a) for a in age_order],
        rotation=40,
        ha="right",
    )
    ax.set_xlabel("Forest age class (yr)")
    ax.set_ylabel(ylabel)
    # Panel titles remain intentionally omitted for the publication layout.

    if show_n and not summary.empty:
        n_df = (
            summary[summary["product"] == products[0]]
            .set_index("age20_class")
            .reindex(age_order)
        )
        n_values = (
            pd.to_numeric(n_df["N"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        y_min, y_max = ax.get_ylim()
        y_text = y_min + 0.02 * (y_max - y_min)
        for xi, n in zip(x, n_values):
            if n > 0:
                ax.text(
                    xi,
                    y_text,
                    f"{n}",
                    fontsize=6.8,
                    ha="center",
                    va="bottom",
                    color="0.35",
                    rotation=90,
                    clip_on=False,
                )
        ax.text(
            0.01,
            0.02,
            "Numbers denote N",
            transform=ax.transAxes,
            fontsize=7.0,
            ha="left",
            va="bottom",
            color="0.35",
        )

    # Panels a and b use one shared figure-level legend added in main().


def plot_panel_c_external_summary(
    ax: plt.Axes,
    performance: pd.DataFrame,
) -> None:
    """Panel c: residual–age slope forest plot.

    CR–ML is shown at the top as a model reference. External remote-sensing
    and DGVM products are grouped below. Horizontal error bars are 95%
    1-degree spatial-block bootstrap confidence intervals.
    """
    style_axes(ax)
    # Panel-c label is placed above and slightly left of the axes to avoid
    # overlapping the top CR–ML y tick and the "Model reference" annotation.
    ax.annotate(
        "(c)",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-34, 15),
        textcoords="offset points",
        fontsize=12.8,
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.axvline(0, color="0.45", linestyle=(0, (3, 2)), linewidth=1.0, zorder=0)

    valid = performance.dropna(
        subset=["residual_age_slope_per_20yr"]
    ).copy()
    if valid.empty:
        ax.text(0.5, 0.5, "No valid residual–age slope estimates",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    # Logical order with visual gaps between model reference, remote sensing,
    # and DGVM groups.
    product_order = [
        "CR_ML",
        "ESACCI", "thurner", "ICESAT2", "Remote sensing mean",
        "ORCHIDEE", "LPJGUESS", "DGVM ensemble",
    ]
    y_positions = {
        "CR_ML": 0.0,
        "ESACCI": 2.0,
        "thurner": 3.0,
        "ICESAT2": 4.0,
        "Remote sensing mean": 5.0,
        "ORCHIDEE": 7.0,
        "LPJGUESS": 8.0,
        "DGVM ensemble": 9.0,
    }

    valid = (
        valid.set_index("product")
        .reindex([p for p in product_order if p in set(valid["product"])])
        .reset_index()
    )

    for _, row in valid.iterrows():
        product = str(row["product"])
        y = y_positions[product]
        slope = float(row["residual_age_slope_per_20yr"])
        lo = pd.to_numeric(pd.Series([row.get("slope_CI_low", np.nan)]), errors="coerce").iloc[0]
        hi = pd.to_numeric(pd.Series([row.get("slope_CI_high", np.nan)]), errors="coerce").iloc[0]
        color = PRODUCT_COLORS.get(product, "0.4")
        marker = PRODUCT_MARKERS.get(product, "o")

        xerr = None
        if np.isfinite(lo) and np.isfinite(hi):
            xerr = np.array([[max(0.0, slope - float(lo))],
                             [max(0.0, float(hi) - slope)]])

        is_reference = product == "CR_ML"
        ax.errorbar(
            slope, y,
            xerr=xerr,
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.1 if is_reference else 5.4,
            linewidth=1.2,
            ecolor=color,
            elinewidth=1.15 if is_reference else 0.95,
            capsize=3.0,
            zorder=4 if is_reference else 3,
        )

        # N is aligned in a separate right-hand rail to avoid cluttering the CI.
        n = int(row.get("N", 0))
        if n > 0:
            ax.text(
                0.985, y, f"N={n}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=7.0, color=ANNOTATION_COLOR,
                clip_on=False,
            )

    # Group separators and labels.
    ax.axhline(1.0, color="0.86", linewidth=0.8, zorder=0)
    ax.axhline(6.0, color="0.86", linewidth=0.8, zorder=0)
    ax.text(
        0.020, 0.935, "Model reference", transform=ax.transAxes,
        ha="left", va="top", fontsize=7.3, color=ANNOTATION_COLOR,
        fontweight="semibold",
    )
    ax.text(
        0.015, 0.765, "Remote sensing", transform=ax.transAxes,
        ha="left", va="top", fontsize=7.8, color=ANNOTATION_COLOR,
    )
    ax.text(
        0.015, 0.335, "DGVM", transform=ax.transAxes,
        ha="left", va="top", fontsize=7.8, color=ANNOTATION_COLOR,
    )

    yticks = [y_positions[p] for p in product_order if p in set(valid["product"])]
    ylabels = [PRODUCT_LABELS.get(p, p) for p in product_order if p in set(valid["product"])]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.3)
    # Emphasize CR–ML label without making the external products visually weak.
    for tick, product in zip(ax.get_yticklabels(), [p for p in product_order if p in set(valid["product"])]):
        if product == "CR_ML":
            tick.set_fontweight("semibold")

    ax.set_ylim(-0.9, 9.8)
    ax.invert_yaxis()

    # Data-driven x limits include zero so the direction of age dependence is explicit.
    lows = pd.to_numeric(valid["slope_CI_low"], errors="coerce")
    highs = pd.to_numeric(valid["slope_CI_high"], errors="coerce")
    centers = pd.to_numeric(valid["residual_age_slope_per_20yr"], errors="coerce")
    xmin = np.nanmin(np.r_[lows.to_numpy(float), centers.to_numpy(float), 0.0])
    xmax = np.nanmax(np.r_[highs.to_numpy(float), centers.to_numpy(float), 0.0])
    span = max(xmax - xmin, 1.0)
    ax.set_xlim(xmin - 0.10 * span, xmax + 0.18 * span)

    ax.set_xlabel(
        "Residual–age slope (Mg C ha$^{-1}$ per 20 yr)",
        fontsize=8.9,
    )
    ax.set_ylabel("")


def plot_panel_d_broad_age_facets(
    axes: Sequence[plt.Axes],
    summary: pd.DataFrame,
    broad_order: Sequence[str],
) -> None:
    """Panel d: horizontal facets for young, middle-aged, and old forests; each facet shows product bias and 95% confidence intervals."""
    product_order = list(BROAD_AGE_PRODUCTS)

    for i, (ax, age_class) in enumerate(zip(axes, broad_order)):
        style_axes(ax)
        if i == 0:
            add_panel_label(ax, "d")
        ax.axvline(0, color="0.45", linestyle=(0, (3, 2)), linewidth=0.9, zorder=0)

        tmp = (
            summary[summary["broad_age_class"] == str(age_class)]
            .set_index("product")
            .reindex(product_order)
            .reset_index()
        )

        y = np.arange(len(product_order), dtype=float)
        bias = pd.to_numeric(tmp["Bias(product-inventory)"], errors="coerce").to_numpy(float)
        low = pd.to_numeric(tmp["Bias_CI_low"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(tmp["Bias_CI_high"], errors="coerce").to_numpy(float)
        n_values = pd.to_numeric(tmp["N"], errors="coerce").fillna(0).astype(int).to_numpy()

        for yi, product, b, lo, hi, n in zip(y, product_order, bias, low, high, n_values):
            if not np.isfinite(b):
                continue
            color = PRODUCT_COLORS.get(product, "0.4")
            marker = PRODUCT_MARKERS.get(product, "o")
            xerr = None
            if np.isfinite(lo) and np.isfinite(hi):
                xerr = np.array([[b - lo], [hi - b]])
            ax.errorbar(
                b, yi,
                xerr=xerr,
                fmt=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=5.2,
                linewidth=1.0,
                ecolor=color,
                elinewidth=0.8,
                capsize=2.4,
                zorder=3,
            )
            if n > 0:
                ax.annotate(
                    f"{n}",
                    xy=(b, yi),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=7.3,
                    ha="left",
                    va="center",
                    color=ANNOTATION_COLOR,
                )

        ax.text(0.5, 1.06, broad_age_display(age_class), transform=ax.transAxes, ha="center", va="bottom", fontsize=8.8, clip_on=False)
        ax.set_xlabel("Bias (Mg C ha$^{-1}$)", fontsize=9.4)
        ax.set_ylim(-0.6, len(product_order) - 0.4)
        ax.invert_yaxis()
        # Separator between remote-sensing and DGVM groups: first four are remote-sensing products, last three are DGVMs.
        ax.axhline(3.5, color="0.82", linewidth=0.8, zorder=0)

        if i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([PRODUCT_LABELS.get(p, p) for p in product_order])
            ax.set_ylabel("Product", fontsize=9.4)
            ax.text(0.03, 0.865, "Remote sensing", transform=ax.transAxes, ha="left", va="center", fontsize=8.0, color=ANNOTATION_COLOR)
            ax.text(0.03, 0.245, "DGVM", transform=ax.transAxes, ha="left", va="center", fontsize=8.0, color=ANNOTATION_COLOR)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
            ax.set_ylabel("")

    # Use a common x-axis range.
    xmins = []
    xmaxs = []
    for age_class in broad_order:
        tmp = summary[summary["broad_age_class"] == str(age_class)]
        if tmp.empty:
            continue
        low = pd.to_numeric(tmp["Bias_CI_low"], errors="coerce")
        high = pd.to_numeric(tmp["Bias_CI_high"], errors="coerce")
        bias = pd.to_numeric(tmp["Bias(product-inventory)"], errors="coerce")
        vals_low = low.where(low.notna(), bias)
        vals_high = high.where(high.notna(), bias)
        if vals_low.notna().any() and vals_high.notna().any():
            xmins.append(vals_low.min())
            xmaxs.append(vals_high.max())
    if xmins and xmaxs:
        xmin = min(xmins)
        xmax = max(xmaxs)
        pad = max((xmax - xmin) * 0.08, 2.0)
        for ax in axes:
            ax.set_xlim(xmin - pad, xmax + pad)




# ============================================================
# 7. Main program
# ============================================================

def main() -> None:
    warnings.filterwarnings("default", category=RuntimeWarning)
    set_publication_style()
    rng = np.random.default_rng(RANDOM_SEED)

    main_module = load_main_analysis_module()
    full_products, audit = prepare_full_inventory_with_products(main_module)
    analysis = harmonize_analysis_columns(full_products, main_module)
    analysis = filter_scenario(analysis, main_module)

    if analysis.empty:
        raise RuntimeError(f"Survey-year scenario {SCENARIO} contains no samples after filtering.")

    age_order = list(main_module.AGE20_GROUP_ORDER)
    broad_order = list(main_module.BROAD_AGE_GROUP_ORDER)

    # Common sample for the main age panels.
    age_sample = make_age_panel_sample(analysis)
    if len(age_sample) < MIN_N_PER_AGE_CLASS:
        raise RuntimeError(
            f"The main age-panel common sample has only N={len(age_sample)}, which is insufficient for plotting."
            "Check product validity, units, spatial domain, or set "
            "USE_COMMON_SAMPLE_FOR_AGE_PANELS=False."
        )

    age_mean_summary = summarize_age20_means(age_sample, age_order, rng)
    age_bias_summary = summarize_age20_bias(age_sample, age_order, rng)
    external_summary, external_common = summarize_external_product_performance(analysis, rng)
    broad_bias_summary = summarize_broad_age_bias(analysis, broad_order, rng)

    # Write all plotting data and sample-audit outputs.
    analysis.to_csv(
        os.path.join(OUT_DIR, f"fig7_full_inventory_analysis_data_{SCENARIO}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    age_sample.to_csv(
        os.path.join(OUT_DIR, f"fig7_age_panel_common_sample_{SCENARIO}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    age_mean_summary.to_csv(
        os.path.join(OUT_DIR, f"fig7_panel_a_age20_mean_AGCD_{SCENARIO}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    age_bias_summary.to_csv(
        os.path.join(OUT_DIR, f"fig7_panel_b_age20_inventory_bias_{SCENARIO}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    external_summary.to_csv(
        os.path.join(OUT_DIR, f"fig7_panel_c_residual_age_slope_with_CRML_reference_{SCENARIO}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    broad_bias_summary.to_csv(
        os.path.join(OUT_DIR, f"fig7_panel_d_broad_age_bias_{SCENARIO}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    if USE_COMMON_SAMPLE_FOR_EXTERNAL_SUMMARY and not external_common.empty:
        external_common.to_csv(
            os.path.join(OUT_DIR, f"fig7_panel_c_common_sample_with_CRML_reference_{SCENARIO}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    metadata = pd.DataFrame([{
        "figure_role": "full-inventory extended age-dependent diagnosis",
        "not_independent_validation_of_CRML": True,
        "scenario": SCENARIO,
        "selected_CR_AGC_weight": SELECTED_CR_WEIGHT,
        "N_full_inventory_after_scenario_filter": int(len(analysis)),
        "N_age_panel_common_sample": int(len(age_sample)),
        "N_panel_c_common_sample": (
            int(len(external_common))
            if USE_COMMON_SAMPLE_FOR_EXTERNAL_SUMMARY
            else np.nan
        ),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_grid_degrees": BOOTSTRAP_GRID_DEGREES,
        "minimum_N_per_age20_class": MIN_N_PER_AGE_CLASS,
        "minimum_N_per_broad_age_class": MIN_N_PER_BROAD_AGE_CLASS,
        "removed_original_relative_CRML_panel": True,
    }])
    metadata.to_csv(
        os.path.join(OUT_DIR, f"fig7_analysis_metadata_{SCENARIO}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Publication layout optimized for a final width of 14 cm:
    # row 1: panels a and b side-by-side;
    # row 2: panel c spans the full figure width;
    # row 3: panel d spans the full figure width and contains three age-stage facets.
    # This preserves readable fonts at final size and avoids compressing panels c and d.
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        dpi=300,
    )
    gs = fig.add_gridspec(
        3, 2,
        left=0.09,
        right=0.985,
        bottom=0.065,
        top=0.895,
        wspace=0.36,
        hspace=0.58,
        height_ratios=[1.15, 0.82, 0.95],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel c receives the full width so the slope estimates and confidence
    # intervals can be read without crowding.
    ax_c = fig.add_subplot(gs[1, :])

    # Panel d also receives the full width; its three age-stage facets are
    # separated internally.
    gs_d = gs[2, :].subgridspec(1, 3, wspace=0.16)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1], sharey=ax_d1)
    ax_d3 = fig.add_subplot(gs_d[0, 2], sharey=ax_d1)
    ax_d_axes = [ax_d1, ax_d2, ax_d3]

    # One complete legend shared by panels a and b. This avoids clipping,
    # duplication and legend drift between nested GridSpec panels.
    add_shared_ab_legend(fig)

    plot_age_lines(
        ax=ax_a,
        summary=age_mean_summary,
        value_col="mean_AGCD",
        ci_low_col="CI_low",
        ci_high_col="CI_high",
        products=AGE_PANEL_SERIES,
        age_order=age_order,
        ylabel="Mean AGCD (Mg C ha$^{-1}$)",
        title="AGCD by forest age",
        panel_label="a",
        horizontal_zero=False,
        show_n=SHOW_AGE_CLASS_N,
        grouped_legend=False,
    )

    plot_age_lines(
        ax=ax_b,
        summary=age_bias_summary,
        value_col="Bias(product-inventory)",
        ci_low_col="CI_low",
        ci_high_col="CI_high",
        products=BIAS_PANEL_SERIES,
        age_order=age_order,
        ylabel="Bias = product − inventory\n(Mg C ha$^{-1}$)",
        title="Inventory residual bias by age",
        panel_label="b",
        horizontal_zero=True,
        show_n=False,
    )
    # Extra breathing room between panel a and the y-axis title of panel b.
    ax_b.yaxis.labelpad = 6.0

    apply_data_driven_ylim(
        ax_a,
        age_mean_summary,
        AGE_PANEL_SERIES,
        value_col="mean_AGCD",
        low_col="CI_low",
        high_col="CI_high",
        symmetric_zero=False,
    )
    apply_data_driven_ylim(
        ax_b,
        age_bias_summary,
        BIAS_PANEL_SERIES,
        value_col="Bias(product-inventory)",
        low_col="CI_low",
        high_col="CI_high",
        symmetric_zero=True,
    )

    plot_panel_c_external_summary(ax_c, external_summary)
    plot_panel_d_broad_age_facets(ax_d_axes, broad_bias_summary, broad_order)

    base_name = f"Fig7_age_diagnosis_slope_forest_CRML_reference_{SCENARIO}"
    png_path = os.path.join(OUT_DIR, base_name + ".png")
    pdf_path = os.path.join(OUT_DIR, base_name + ".pdf")
    svg_path = os.path.join(OUT_DIR, base_name + ".svg")

    fig.savefig(png_path, dpi=900)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print("\n========== Fig. 5 complete ==========")
    print(f"Full-inventory scenario N: {len(analysis)}")
    print(f"Panel a/b common-sample N: {len(age_sample)}")
    if USE_COMMON_SAMPLE_FOR_EXTERNAL_SUMMARY:
        print(f"Panel c common-sample N (including CR-ML reference): {len(external_common)}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"SVG: {svg_path}")
    print("\nInterpretation boundaries:")
    print("- This figure is an extended full-inventory age-dependence diagnostic for remote-sensing products and DGVMs.")
    print("- CR-ML is an age-constrained reference in panels a/b and a model reference for residual-age slope in panel c.")
    print("- The CR-ML slope in panel c is an age-diagnostic reference and must not be interpreted as an independent performance ranking; independent performance is reported separately using 1-degree grid validation of the 811 samples.")
    print("- Independent CR-ML performance continues to be reported using the 1-degree grid validation of the 811 samples.")
    print("- The former relative-difference-from-CR-ML panel has been removed.")


if __name__ == "__main__":
    main()
