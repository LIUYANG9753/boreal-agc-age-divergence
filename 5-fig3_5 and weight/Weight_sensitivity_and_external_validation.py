# -*- coding: utf-8 -*-
"""

from __future__ import annotations
Development-sample weight diagnostics and age-stratified validation using 811 external independent samples

This workflow strictly separates the forest-inventory data into two non-overlapping analytical roles:

1. development samples
   - Source: DEVELOPMENT_INVENTORY_FILE.
   - If this file contains any of the 811 external records, they are identified and removed using coordinates, forest age, AGCD, and survey year.
   - Used only for AGC_ML / CR_AGC / CR_ML_w weight-sensitivity diagnostics.
   - Resulting errors are development-sample diagnostics and must not be reported as independent validation.

2. external_independent_validation (811 samples)
   - Source: INDEPENDENT_VALIDATION_FILE.
   - Not used for weight sensitivity, weight selection, or the decision table.
   - Used only to compare CR-ML, AGC_ML, CR_AGC, remote-sensing products, and DGVMs after MAIN_WEIGHT has been fixed.
   - No alternative fusion weights are scanned or compared on the independent samples.
   - Age-stratified bias, RMSE, and residual-age trends are calculated from this dataset.

Fusion formula:
    CR_ML_w = (1 - w) * AGC_ML + w * CR_AGC

Important interpretation notes:
- Development and independent-validation outputs are written to separate subdirectories.
- Root-level weighted_CRML_* summary tables are derived from development samples and are used for weight-sensitivity figures.
- Root-level all_products_* summary tables are derived from the 811 independent samples and are used for final age-dependence comparisons.
- Bias = product - inventory; negative values indicate underestimation relative to inventory observations.
- linear_slope_per_20yr < 0 indicates that product residuals decrease with increasing forest age.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform as rio_transform

try:
    from pyproj import Geod
except ImportError:
    Geod = None


# ============================================================
# 1. Paths and analysis parameters
# ============================================================

# -------------------------
# 1.0 Inventory-data roles
# -------------------------
# Development inventory used only for model-development weight-sensitivity diagnostics.
# If this file contains the full 4,595-record inventory, the 811 independent records below are removed automatically.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("CRML_PROJECT_ROOT", SCRIPT_DIR))
DATA_DIR = Path(os.environ.get("CRML_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("CRML_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

DEVELOPMENT_INVENTORY_FILE = str(DATA_DIR / "inventory" / "development_inventory.xls")

# External independent-validation inventory; must not be used for weight selection.
INDEPENDENT_VALIDATION_FILE = str(DATA_DIR / "inventory" / "independent_validation_811.xls")

DEVELOPMENT_INVENTORY_SHEET_NAME = "carbon_cal"
INDEPENDENT_INVENTORY_SHEET_NAME = "carbon_cal"

# Whether records overlapping the 811 independent samples are removed automatically from the development inventory.
REMOVE_INDEPENDENT_FROM_DEVELOPMENT = True

# The primary matching key uses longitude, latitude, Age_used, AGCD_obs, and Survey_year.
# These rounding precisions are used only for record matching and do not alter analysis values.
OVERLAP_COORD_DECIMALS = 5
OVERLAP_AGE_DECIMALS = 3
OVERLAP_AGCD_DECIMALS = 3

# Whether remaining records at coordinates shared by development and independent samples are also removed.
# True enforces site/location-level independence; False removes only exact matching inventory records.
REMOVE_COORDINATE_ONLY_OVERLAPS = True

# Inventory worksheet and observed-AGCD field
INVENTORY_SHEET_NAME = "carbon_cal"  # retained for compatibility with legacy helper functions; actual reads use the two sheet settings above
USE_PRECALCULATED_CARBON_COL = True
PRECALCULATED_CARBON_COL = "AGCD"
BIOMASS_CARBON_IS_PER_HA = True
BIOMASS_CARBON_MULTIPLIER_TO_MG_C = 1.0
FALLBACK_TO_RAW_BIOMASS_IF_CARBON_MISSING = False

# Output directory
OUT_DIR = str(OUTPUT_ROOT / "weight_sensitivity_external_validation")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# 1.1 Product raster paths
# -------------------------
# AGC_ML and CR_AGC are the two endpoint products used for fusion.
# An existing final CR-ML raster can optionally be included to check consistency with the CR_ML_w product reconstructed using MAIN_WEIGHT.
PRODUCT_RASTER_PATHS = {
    # Products developed in this study
    "AGC_ML": str(DATA_DIR / "rasters" / "AGCD_ML.tif"),
    "CR_AGC": str(DATA_DIR / "rasters" / "AGCD_CR.tif"),
    # "CR_ML": str(DATA_DIR / "rasters" / "CR_ML_final_AGCD_MgCha.tif"),

    # Existing remote-sensing products and DGVM outputs
    "ESACCI": str(DATA_DIR / "rasters" / "ESACCI_2020_hansen_clip.tif"),
    "thurner": str(DATA_DIR / "rasters" / "thurner_hansen_clip.tif"),
    "DGVM": str(DATA_DIR / "rasters" / "DGVM_Ensemble_Mean_AGCD_2020_MgCha.tif"),
    "ORCHIDEE": str(DATA_DIR / "rasters" / "ORCHIDEE_Aboveground_Carbon_2020_MgCha.tif"),
    "LPJGUESS": str(DATA_DIR / "rasters" / "LPJ-GUESS_Aboveground_Carbon_2020_MgCha.tif"),
    "ICESAT2": str(DATA_DIR / "rasters" / "ICESAT2_hansen_clip.tif"),
}

# Unit conversion: raster value * multiplier -> Mg C ha-1
# Confirm these multipliers against the units of the preprocessed input rasters.
PRODUCT_MULTIPLIERS = {
    "AGC_ML": 1.0,
    "CR_AGC": 1.0,
    "ESACCI": 1.0,
    "thurner": 10.0,
    "DGVM": 1.0,
    "ORCHIDEE": 1.0,
    "LPJGUESS": 1.0,
    "ICESAT2": 0.5,
}

# Products for which zero is treated as NoData
ZERO_AS_NODATA_PRODUCTS = {"thurner", "ORCHIDEE", "LPJGUESS"}

# If near-zero values in CR_AGC or AGC_ML should not be considered valid forest AGCD,
# increase the threshold below (for example, to 1.0 or 5.0) and rerun as a sensitivity analysis.
MIN_VALID_PRODUCT_AGCD = 0.0
MAX_PRODUCT_AGCD = 500.0

RASTER_INVALID_VALUES = [-99990, -9999, -999, -32768, 32767, 65535, 9999, 99999]

# -------------------------
# 1.2 Weight settings
# -------------------------
AGC_ML_COL = "AGC_ML"
CR_AGC_COL = "CR_AGC"
CURRENT_FINAL_CRML_COL = "CR_ML"

# CR_AGC weight: w=0 is AGC_ML-only; w=1 is CR_AGC-only.
WEIGHTS = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
MAIN_WEIGHT = 0.1

# Key weights retained for common-sample comparisons across products.
# Downstream figures read these integrated outputs; therefore MAIN_WEIGHT must be included.
SELECTED_WEIGHTS_FOR_ALL_PRODUCTS = sorted(set([0.00, MAIN_WEIGHT, 0.15,0.25, 0.50, 1.00]))

# Whether to generate the main-weight CR-ML AGCD raster and per-pixel carbon-stock raster.
# AGCD units: Mg C ha-1. Carbon-stock units: Mg C per pixel.
CREATE_MAIN_WEIGHT_CRML_RASTER = True
MAIN_WEIGHT_RASTER_OUT_DIR = os.path.join(OUT_DIR, "main_weight_CRML_AGC_and_stock")

# Optional spatial-domain mask; pixels > 0 are treated as valid.
# If None, the common valid domain of AGC_ML and CR_AGC is used.
DOMAIN_MASK_RASTER = None

# Write the per-pixel carbon-stock raster. Set to False if only the total stock is needed.
WRITE_PIXEL_STOCK_RASTER = True

# NoData value used for the main-weight CR-ML output raster.
OUTPUT_NODATA_VALUE = -9999.0

# Prefix for weighted-product column names
WEIGHTED_PREFIX = "CR_ML_w"

# -------------------------
# 1.3 Inventory fields
# -------------------------
LAT_COL = "lat"
LON_COL = "lon"
BIOMASS_COL = "biomass"
BIOMASS_TYPE_COL = "biomass_type"
BIOMASS_UNIT_COL = "biomass_unit"
PLOT_AREA_COL = "plot area"
DATE_COL = "date"

SURVEY_AGE_COL = "stand age"
AGE2020_COL = "stand age2020"

# Forest-age source:
# survey_age: use stand age corresponding to the inventory observation date;
# age2020_prefer: prefer stand age2020 for sensitivity analyses aligned with 2020 rasters.
AGE_SOURCE_MODE = "survey_age"

# Survey-year sensitivity scenarios
SURVEY_YEAR_SCENARIOS = [
    {"name": "all_years", "min": None, "max": None},
    {"name": "survey_2000_2021", "min": 2000, "max": 2021},
    {"name": "survey_2010_2021", "min": 2010, "max": 2021},
]
MAX_ALLOWED_SURVEY_YEAR = 2025

# Basic quality control
MIN_VALID_AGE = 0
MAX_VALID_AGE = 500
MIN_VALID_AGCD = 0
KEEP_ZERO_AGCD = False

# Age classes: 20-year intervals, with >200 years as a separate class.
AGE_BIN_WIDTH = 20
AGE_BIN_MAX = 200

# Broad age groups
# Broad groups used in the manuscript: young <50, middle 50-150, old >150 years.
BROAD_AGE_GROUPS = [
    ("young_<50", 0, 50),
    ("middle_50_150", 50, 150),
    ("old_>150", 150, np.inf),
]

# Minimum sample sizes for metrics
MIN_N_FOR_METRICS = 3
MIN_N_FOR_CORRELATION = 10

# Whether to generate figures
MAKE_FIGURES = True

# Common-sample settings
# 1) Weight sensitivity requires valid AGC_ML, CR_AGC, and all weighted-fusion columns only, avoiding restrictions from other product coverage.
# 2) Cross-product comparisons can use a common sample across all target products.
USE_COMMON_SAMPLE_FOR_WEIGHT_SENSITIVITY = True
USE_COMMON_SAMPLE_FOR_ALL_PRODUCTS_COMPARISON = True

# Products included in the all-product common sample
ALL_PRODUCTS_COMMON_SAMPLE_PRODUCTS = [
    "CR_ML",
    "ESACCI",
    "thurner",
    "ICESAT2",
    "DGVM",
    "ORCHIDEE",
    "LPJGUESS",
]


# ============================================================
# 2. Core utilities
# ============================================================

def set_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.minorticks_on()


def read_excel_file(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read the Excel input file.\n"
            "Install openpyxl for .xlsx files or xlrd for .xls files.\n"
            f"File: {path}\n"
            f"Worksheet: {sheet_name}\n"
            f"Original error: {exc}"
        )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def check_required_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(map(str, df.columns))
        )


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_year_series(s: pd.Series) -> pd.Series:
    """
    Robustly parse survey years.
    Supports numeric years, date strings, Excel serial dates, and yyyymmdd values.
    """
    raw = s.copy()
    numeric = pd.to_numeric(raw, errors="coerce")
    year = pd.Series(np.nan, index=s.index, dtype=float)

    mask_year = numeric.between(1800, 2100)
    year.loc[mask_year] = numeric.loc[mask_year].astype(float)

    mask_excel_serial = year.isna() & numeric.between(1, 60000)
    if mask_excel_serial.any():
        dt_excel = pd.to_datetime(
            numeric.loc[mask_excel_serial],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
        year.loc[mask_excel_serial] = dt_excel.dt.year.astype(float)

    mask_yyyymmdd = year.isna() & numeric.between(18000101, 21001231)
    if mask_yyyymmdd.any():
        dt_yyyymmdd = pd.to_datetime(
            numeric.loc[mask_yyyymmdd].round().astype("Int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        year.loc[mask_yyyymmdd] = dt_yyyymmdd.dt.year.astype(float)

    dt = pd.to_datetime(raw, errors="coerce")
    mask_datetime = year.isna() & dt.notna()
    year.loc[mask_datetime] = dt.dt.year.loc[mask_datetime].astype(float)

    year.loc[~year.between(1800, 2100)] = np.nan
    return year


def safe_filename(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^0-9A-Za-z_\-]+", "_", text)
    return text.strip("_") or "scenario"


def age_group_order() -> List[str]:
    labels = [f"{i}-{i + AGE_BIN_WIDTH}" for i in range(0, AGE_BIN_MAX, AGE_BIN_WIDTH)]
    labels.append(f">{AGE_BIN_MAX}")
    return labels


AGE20_GROUP_ORDER = age_group_order()
BROAD_AGE_GROUP_ORDER = [x[0] for x in BROAD_AGE_GROUPS]


def assign_age20_group(age) -> Optional[str]:
    if pd.isna(age):
        return pd.NA

    age = float(age)
    if age < 0:
        return pd.NA

    if age > AGE_BIN_MAX:
        return f">{AGE_BIN_MAX}"

    low = int(np.floor(age / AGE_BIN_WIDTH) * AGE_BIN_WIDTH)
    low = min(low, AGE_BIN_MAX - AGE_BIN_WIDTH)
    high = low + AGE_BIN_WIDTH
    return f"{low}-{high}"


def assign_broad_age_group(age) -> Optional[str]:
    if pd.isna(age):
        return pd.NA

    age = float(age)
    if age < 0:
        return pd.NA

    for label, low, high in BROAD_AGE_GROUPS:
        if age >= low and age < high:
            return label

    return pd.NA


def filter_by_survey_year(
    df: pd.DataFrame,
    survey_year_min: Optional[int] = None,
    survey_year_max: Optional[int] = None,
) -> pd.DataFrame:
    if survey_year_min is None and survey_year_max is None:
        return df.copy()

    if "Survey_year" not in df.columns:
        warnings.warn("Survey_year is unavailable; survey-year filtering cannot be applied. Returning an empty table.")
        return df.iloc[0:0].copy()

    survey_year = pd.to_numeric(df["Survey_year"], errors="coerce")
    mask = survey_year.notna()

    if survey_year_min is not None:
        mask &= survey_year >= survey_year_min
    if survey_year_max is not None:
        mask &= survey_year <= survey_year_max

    return df[mask].copy()


def add_scenario_metadata(
    df: pd.DataFrame,
    scenario_name: str,
    survey_year_min: Optional[int],
    survey_year_max: Optional[int],
) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "survey_year_max", survey_year_max)
    out.insert(0, "survey_year_min", survey_year_min)
    out.insert(0, "survey_year_scenario", scenario_name)
    return out


def add_dataset_metadata(
    df: pd.DataFrame,
    dataset_role: str,
    source_inventory_file: str,
    is_external_independent_validation: bool,
) -> pd.DataFrame:
    """Add the dataset role to output tables to prevent confusion between development diagnostics and independent validation."""
    out = df.copy()
    out.insert(0, "is_external_independent_validation", bool(is_external_independent_validation))
    out.insert(0, "source_inventory_file", source_inventory_file)
    out.insert(0, "dataset_role", dataset_role)
    return out


# ============================================================
# 3. Inventory AGCD and forest-age processing
# ============================================================

def parse_plot_area_ha(x) -> float:
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "nac", "ni", "none", "null"}:
        return np.nan

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan

    val = float(m.group(0))
    sl = s.lower()
    if "m2" in sl or "m^2" in sl or "㎡" in sl:
        return val / 10000.0

    return val


def is_per_ha_unit(unit_text: str) -> bool:
    if unit_text is None or pd.isna(unit_text):
        return False

    u = str(unit_text).strip().lower()
    patterns = ["/ha", "ha-1", "ha^-1", "ha⁻¹", "mg c ha-1", "mg ha-1"]
    return any(p in u for p in patterns)


def is_kg_unit(unit_text: str) -> bool:
    if unit_text is None or pd.isna(unit_text):
        return False
    u = str(unit_text).strip().lower()
    return u == "kg" or u.startswith("kg ")


def is_carbon_type(biomass_type: str) -> bool:
    if biomass_type is None or pd.isna(biomass_type):
        return False
    t = str(biomass_type).strip().lower()
    return "carbon" in t or t in {"agc", "aboveground carbon", "biomass_carbon"}


def compute_agcd_from_raw_biomass(row: pd.Series) -> Tuple[float, str]:
    biomass = pd.to_numeric(row.get(BIOMASS_COL), errors="coerce")
    if pd.isna(biomass):
        return np.nan, "missing_biomass"

    btype = row.get(BIOMASS_TYPE_COL)
    unit = row.get(BIOMASS_UNIT_COL)
    area_ha = parse_plot_area_ha(row.get(PLOT_AREA_COL))

    value = float(biomass)
    note_parts = ["fallback_raw_biomass"]

    if is_kg_unit(unit):
        value = value / 1000.0
        note_parts.append("kg_to_Mg")

    if is_per_ha_unit(unit):
        density = value
        note_parts.append("per_ha")
    else:
        if pd.isna(area_ha) or area_ha <= 0:
            return np.nan, "fallback_missing_or_invalid_plot_area_for_total_mass"
        density = value / area_ha
        note_parts.append("total_mass_div_area")

    if is_carbon_type(btype):
        agcd = density
        note_parts.append("carbon")
    else:
        agcd = density * 0.5
        note_parts.append("biomass_to_carbon_0.5")

    return float(agcd), "|".join(note_parts)


def compute_inventory_agcd(row: pd.Series) -> Tuple[float, str]:
    if USE_PRECALCULATED_CARBON_COL and PRECALCULATED_CARBON_COL in row.index:
        carbon_value = pd.to_numeric(row.get(PRECALCULATED_CARBON_COL), errors="coerce")

        if pd.notna(carbon_value):
            value = float(carbon_value) * float(BIOMASS_CARBON_MULTIPLIER_TO_MG_C)

            if BIOMASS_CARBON_IS_PER_HA:
                return value, f"{PRECALCULATED_CARBON_COL}_direct|already_carbon|per_ha|no_0.5"

            area_ha = parse_plot_area_ha(row.get(PLOT_AREA_COL))
            if pd.isna(area_ha) or area_ha <= 0:
                return np.nan, f"{PRECALCULATED_CARBON_COL}_missing_or_invalid_plot_area"
            return value / area_ha, f"{PRECALCULATED_CARBON_COL}_direct|already_carbon|total_C_div_area|no_0.5"

        if not FALLBACK_TO_RAW_BIOMASS_IF_CARBON_MISSING:
            return np.nan, f"missing_{PRECALCULATED_CARBON_COL}"

    return compute_agcd_from_raw_biomass(row)


def build_age_variable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    survey_age = to_numeric_series(df[SURVEY_AGE_COL])
    age2020 = (
        to_numeric_series(df[AGE2020_COL])
        if AGE2020_COL in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    if "Survey_year" in df.columns:
        survey_year = to_numeric_series(df["Survey_year"])
    elif DATE_COL in df.columns:
        survey_year = parse_year_series(df[DATE_COL])
    else:
        survey_year = pd.Series(np.nan, index=df.index)

    if AGE_SOURCE_MODE == "survey_age":
        df["Age_used"] = survey_age
        df["Age_source"] = "stand age"

    elif AGE_SOURCE_MODE == "age2020_prefer":
        estimated_age2020 = survey_age + (2020 - survey_year)
        used = age2020.copy()
        source = pd.Series("stand age2020", index=df.index, dtype="object")

        mask_missing_age2020 = used.isna() & estimated_age2020.notna()
        used.loc[mask_missing_age2020] = estimated_age2020.loc[mask_missing_age2020]
        source.loc[mask_missing_age2020] = "stand age + (2020 - date)"

        mask_still_missing = used.isna() & survey_age.notna()
        used.loc[mask_still_missing] = survey_age.loc[mask_still_missing]
        source.loc[mask_still_missing] = "stand age fallback"

        df["Age_used"] = used
        df["Age_source"] = source

    else:
        raise ValueError("AGE_SOURCE_MODE must be 'survey_age' or 'age2020_prefer'.")

    df.loc[df["Age_used"] < MIN_VALID_AGE, "Age_used"] = np.nan
    df.loc[df["Age_used"] > MAX_VALID_AGE, "Age_used"] = np.nan

    return df


def prepare_inventory_data(
    inventory_file: str,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    effective_sheet = INVENTORY_SHEET_NAME if sheet_name is None else sheet_name
    df = read_excel_file(inventory_file, sheet_name=effective_sheet)
    df = clean_column_names(df)

    required = [LAT_COL, LON_COL, SURVEY_AGE_COL]
    if USE_PRECALCULATED_CARBON_COL:
        required.append(PRECALCULATED_CARBON_COL)
        if not BIOMASS_CARBON_IS_PER_HA:
            required.append(PLOT_AREA_COL)
        if FALLBACK_TO_RAW_BIOMASS_IF_CARBON_MISSING:
            required.extend([BIOMASS_COL, BIOMASS_TYPE_COL, BIOMASS_UNIT_COL, PLOT_AREA_COL])
    else:
        required.extend([BIOMASS_COL, BIOMASS_TYPE_COL, BIOMASS_UNIT_COL, PLOT_AREA_COL])

    required = list(dict.fromkeys(required))
    check_required_columns(df, required)

    df[LAT_COL] = to_numeric_series(df[LAT_COL])
    df[LON_COL] = to_numeric_series(df[LON_COL])

    if PRECALCULATED_CARBON_COL in df.columns:
        df[PRECALCULATED_CARBON_COL] = to_numeric_series(df[PRECALCULATED_CARBON_COL])
    if BIOMASS_COL in df.columns:
        df[BIOMASS_COL] = to_numeric_series(df[BIOMASS_COL])

    if DATE_COL in df.columns:
        df["Survey_year"] = parse_year_series(df[DATE_COL])
    else:
        df["Survey_year"] = np.nan

    if MAX_ALLOWED_SURVEY_YEAR is not None:
        survey_year_num = pd.to_numeric(df["Survey_year"], errors="coerce")
        before_n = len(df)
        df = df[survey_year_num.isna() | (survey_year_num <= MAX_ALLOWED_SURVEY_YEAR)].copy()
        removed_n = before_n - len(df)
        if removed_n > 0:
            print(f"[basic QC] Removed records with Survey_year > {MAX_ALLOWED_SURVEY_YEAR}: {removed_n} records")

    agcd_notes = df.apply(compute_inventory_agcd, axis=1)
    df["AGCD_obs"] = [x[0] for x in agcd_notes]
    df["AGCD_obs_note"] = [x[1] for x in agcd_notes]

    df = build_age_variable(df)
    df["age20_class"] = df["Age_used"].apply(assign_age20_group)
    df["broad_age_class"] = df["Age_used"].apply(assign_broad_age_group)

    mask = (
        df[LAT_COL].notna()
        & df[LON_COL].notna()
        & df["AGCD_obs"].notna()
        & df["Age_used"].notna()
        & df["age20_class"].notna()
        & df["broad_age_class"].notna()
        & (df[LAT_COL].between(-90, 90))
        & (df[LON_COL].between(-180, 180))
        & (df["AGCD_obs"] >= MIN_VALID_AGCD)
    )

    if not KEEP_ZERO_AGCD:
        mask &= df["AGCD_obs"] > 0

    df_clean = df[mask].copy()
    df_clean["age20_class"] = pd.Categorical(
        df_clean["age20_class"],
        categories=AGE20_GROUP_ORDER,
        ordered=True,
    )
    df_clean["broad_age_class"] = pd.Categorical(
        df_clean["broad_age_class"],
        categories=BROAD_AGE_GROUP_ORDER,
        ordered=True,
    )

    return df_clean


# ============================================================
# 3.1 Separation of development and 811 independent samples
# ============================================================

def _rounded_numeric_key(s: pd.Series, decimals: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round(decimals)


def build_record_overlap_key(df: pd.DataFrame) -> pd.Series:
    """
    Construct matching keys for the same inventory record.

    Uses lon, lat, Age_used, AGCD_obs, and Survey_year.
    Missing Survey_year values are represented by an NA token; an additional fallback key excludes year.
    """
    lon = _rounded_numeric_key(df[LON_COL], OVERLAP_COORD_DECIMALS)
    lat = _rounded_numeric_key(df[LAT_COL], OVERLAP_COORD_DECIMALS)
    age = _rounded_numeric_key(df["Age_used"], OVERLAP_AGE_DECIMALS)
    agcd = _rounded_numeric_key(df["AGCD_obs"], OVERLAP_AGCD_DECIMALS)
    year = pd.to_numeric(df.get("Survey_year"), errors="coerce").round(0)

    return (
        lon.astype("string").fillna("NA") + "|"
        + lat.astype("string").fillna("NA") + "|"
        + age.astype("string").fillna("NA") + "|"
        + agcd.astype("string").fillna("NA") + "|"
        + year.astype("Int64").astype("string").fillna("NA")
    )


def build_record_overlap_key_without_year(df: pd.DataFrame) -> pd.Series:
    lon = _rounded_numeric_key(df[LON_COL], OVERLAP_COORD_DECIMALS)
    lat = _rounded_numeric_key(df[LAT_COL], OVERLAP_COORD_DECIMALS)
    age = _rounded_numeric_key(df["Age_used"], OVERLAP_AGE_DECIMALS)
    agcd = _rounded_numeric_key(df["AGCD_obs"], OVERLAP_AGCD_DECIMALS)
    return (
        lon.astype("string").fillna("NA") + "|"
        + lat.astype("string").fillna("NA") + "|"
        + age.astype("string").fillna("NA") + "|"
        + agcd.astype("string").fillna("NA")
    )


def build_coordinate_key(df: pd.DataFrame) -> pd.Series:
    lon = _rounded_numeric_key(df[LON_COL], OVERLAP_COORD_DECIMALS)
    lat = _rounded_numeric_key(df[LAT_COL], OVERLAP_COORD_DECIMALS)
    return lon.astype("string").fillna("NA") + "|" + lat.astype("string").fillna("NA")


def separate_development_and_external_samples(
    development: pd.DataFrame,
    external: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Remove the 811 external independent samples from the development table and return an audit table.

    Removal order:
    1) Full record key (coordinates + age + AGCD + survey year);
    2) Year-free record key, to handle records with year missing on one side;
    3) Optionally remove remaining records at shared coordinates to enforce location-level independence.
    """
    dev = development.copy()
    ext = external.copy()

    dev["_record_key"] = build_record_overlap_key(dev)
    ext["_record_key"] = build_record_overlap_key(ext)
    dev["_record_key_no_year"] = build_record_overlap_key_without_year(dev)
    ext["_record_key_no_year"] = build_record_overlap_key_without_year(ext)
    dev["_coord_key"] = build_coordinate_key(dev)
    ext["_coord_key"] = build_coordinate_key(ext)

    ext_full_keys = set(ext["_record_key"].dropna().astype(str))
    ext_no_year_keys = set(ext["_record_key_no_year"].dropna().astype(str))
    ext_coord_keys = set(ext["_coord_key"].dropna().astype(str))

    exact_mask = dev["_record_key"].astype(str).isin(ext_full_keys)
    no_year_mask = (~exact_mask) & dev["_record_key_no_year"].astype(str).isin(ext_no_year_keys)
    coord_mask = pd.Series(False, index=dev.index)
    if REMOVE_COORDINATE_ONLY_OVERLAPS:
        coord_mask = (~exact_mask) & (~no_year_mask) & dev["_coord_key"].astype(str).isin(ext_coord_keys)

    remove_mask = exact_mask | no_year_mask | coord_mask

    audit = dev.loc[remove_mask].copy()
    audit["overlap_match_type"] = np.select(
        [exact_mask.loc[audit.index], no_year_mask.loc[audit.index], coord_mask.loc[audit.index]],
        ["full_record", "record_without_year", "coordinate_only"],
        default="unknown",
    )

    dev_clean = dev.loc[~remove_mask].copy()

    helper_cols = ["_record_key", "_record_key_no_year", "_coord_key"]
    dev_clean.drop(columns=helper_cols, inplace=True, errors="ignore")
    ext.drop(columns=helper_cols, inplace=True, errors="ignore")
    audit.drop(columns=helper_cols, inplace=True, errors="ignore")

    dev_clean["sample_role"] = "development_weight_selection"
    dev_clean["is_external_independent_validation"] = False
    ext["sample_role"] = "external_independent_validation"
    ext["is_external_independent_validation"] = True

    return dev_clean, ext, audit


# ============================================================
# 4. Raster extraction and weighted-product construction
# ============================================================

def extract_raster_values_at_points(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    raster_paths: Dict[str, str],
    multipliers: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract values from multiple rasters using longitude/latitude coordinates.
    The input dataframe longitude/latitude coordinates are assumed to be EPSG:4326.
    If a raster uses another CRS, point coordinates are transformed to the raster CRS automatically.
    """
    df_out = df.copy()

    lon = pd.to_numeric(df_out[lon_col], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(df_out[lat_col], errors="coerce").to_numpy(float)
    valid_coord = np.isfinite(lon) & np.isfinite(lat)

    extraction_rows = []

    for product_name, raster_path in raster_paths.items():
        if not os.path.exists(raster_path):
            warnings.warn(f"[skip] {product_name} raster not found: {raster_path}")
            df_out[product_name] = np.nan
            extraction_rows.append({
                "product": product_name,
                "raster_path": raster_path,
                "exists": False,
                "valid_values": 0,
                "total_points": len(df_out),
                "valid_rate": 0.0,
                "raw_min": np.nan,
                "raw_median": np.nan,
                "raw_mean": np.nan,
                "raw_max": np.nan,
                "converted_min": np.nan,
                "converted_median": np.nan,
                "converted_mean": np.nan,
                "converted_max": np.nan,
                "multiplier": multipliers.get(product_name, 1.0),
                "zero_as_nodata": product_name in ZERO_AS_NODATA_PRODUCTS,
                "zero_as_nodata_count": 0,
                "crs": None,
                "nodata": None,
            })
            continue

        print(f"[extract raster] {product_name}")
        print(f"    {raster_path}")

        values = np.full(len(df_out), np.nan, dtype=float)
        raw_values_full = np.full(len(df_out), np.nan, dtype=float)

        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError(f"{product_name} raster has no CRS: {raster_path}")

            nodata = src.nodata
            raster_crs = src.crs

            if raster_crs.to_string().upper() in {"EPSG:4326", "WGS84"}:
                xs = lon[valid_coord]
                ys = lat[valid_coord]
            else:
                xs, ys = rio_transform(
                    "EPSG:4326",
                    raster_crs,
                    lon[valid_coord].tolist(),
                    lat[valid_coord].tolist(),
                )
                xs = np.asarray(xs, dtype=float)
                ys = np.asarray(ys, dtype=float)

            coords = list(zip(xs, ys))
            sampled = []

            for val in src.sample(coords, masked=True):
                v = val[0]
                if np.ma.is_masked(v):
                    sampled.append(np.nan)
                else:
                    sampled.append(float(v))

            sampled = np.asarray(sampled, dtype=float)
            raw_sampled = sampled.copy()

            if nodata is not None and np.isfinite(nodata):
                sampled[np.isclose(sampled, nodata)] = np.nan

            for inv in RASTER_INVALID_VALUES:
                sampled[np.isclose(sampled, inv)] = np.nan

            zero_as_nodata_count = 0
            if product_name in ZERO_AS_NODATA_PRODUCTS:
                zero_mask = np.isfinite(sampled) & np.isclose(sampled, 0.0)
                zero_as_nodata_count = int(zero_mask.sum())
                sampled[zero_mask] = np.nan

            sampled[~np.isfinite(sampled)] = np.nan

            multiplier = float(multipliers.get(product_name, 1.0))
            converted = sampled * multiplier

            converted[converted < MIN_VALID_PRODUCT_AGCD] = np.nan
            converted[converted > MAX_PRODUCT_AGCD] = np.nan

            values[valid_coord] = converted
            raw_values_full[valid_coord] = raw_sampled

        df_out[product_name] = values

        valid_values = np.isfinite(values)
        valid_raw = np.isfinite(raw_values_full)

        extraction_rows.append({
            "product": product_name,
            "raster_path": raster_path,
            "exists": True,
            "valid_values": int(valid_values.sum()),
            "total_points": int(len(df_out)),
            "valid_rate": float(valid_values.mean()),
            "raw_min": float(np.nanmin(raw_values_full)) if valid_raw.any() else np.nan,
            "raw_median": float(np.nanmedian(raw_values_full)) if valid_raw.any() else np.nan,
            "raw_mean": float(np.nanmean(raw_values_full)) if valid_raw.any() else np.nan,
            "raw_max": float(np.nanmax(raw_values_full)) if valid_raw.any() else np.nan,
            "converted_min": float(np.nanmin(values)) if valid_values.any() else np.nan,
            "converted_median": float(np.nanmedian(values)) if valid_values.any() else np.nan,
            "converted_mean": float(np.nanmean(values)) if valid_values.any() else np.nan,
            "converted_max": float(np.nanmax(values)) if valid_values.any() else np.nan,
            "multiplier": multipliers.get(product_name, 1.0),
            "zero_as_nodata": product_name in ZERO_AS_NODATA_PRODUCTS,
            "zero_as_nodata_count": int(zero_as_nodata_count),
            "crs": str(raster_crs),
            "nodata": nodata,
        })

        if valid_values.any():
            print(
                f"    valid values: {int(valid_values.sum())}/{len(df_out)} "
                f"({valid_values.mean():.2%}); converted median={np.nanmedian(values):.2f}"
            )
        else:
            print(f"    valid values: 0/{len(df_out)}")

    summary = pd.DataFrame(extraction_rows)
    return df_out, summary


def weight_to_colname(w: float) -> str:
    return f"{WEIGHTED_PREFIX}{int(round(w * 100)):03d}"


def build_weighted_crml_products(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Construct CR_ML_w = (1 - w) * AGC_ML + w * CR_AGC.
    """
    df = df.copy()

    if AGC_ML_COL not in df.columns or CR_AGC_COL not in df.columns:
        raise KeyError(f"Missing {AGC_ML_COL} or {CR_AGC_COL}; unable to construct weighted-fusion products.")

    weighted_cols = []
    rows = []

    for w in WEIGHTS:
        col = weight_to_colname(w)
        weighted_cols.append(col)

        df[col] = (1.0 - w) * df[AGC_ML_COL] + w * df[CR_AGC_COL]

        rows.append({
            "weighted_product": col,
            "CR_AGC_weight": w,
            "AGC_ML_weight": 1.0 - w,
            "formula": f"{col} = {(1.0 - w):.2f} * {AGC_ML_COL} + {w:.2f} * {CR_AGC_COL}",
            "non_na": int(df[col].notna().sum()),
        })

    summary = pd.DataFrame(rows)
    return df, summary, weighted_cols


def check_current_final_vs_weighted(df: pd.DataFrame, weighted_cols: List[str]) -> pd.DataFrame:
    """
    If an existing CR_ML column is available, compare it with the CR_ML_w product reconstructed using MAIN_WEIGHT.
    """
    main_col = weight_to_colname(MAIN_WEIGHT)
    rows = []

    if CURRENT_FINAL_CRML_COL in df.columns and main_col in df.columns:
        tmp = df[[CURRENT_FINAL_CRML_COL, main_col]].dropna().copy()
        if len(tmp) > 0:
            diff = tmp[CURRENT_FINAL_CRML_COL] - tmp[main_col]
            rows.append({
                "current_final_col": CURRENT_FINAL_CRML_COL,
                "weighted_col": main_col,
                "N": int(len(tmp)),
                "mean_diff_current_minus_weighted": float(diff.mean()),
                "median_diff_current_minus_weighted": float(diff.median()),
                "max_abs_diff_current_minus_weighted": float(np.abs(diff).max()),
                "note": f"If max_abs_diff is close to 0, the existing CR_ML is effectively identical to {main_col}; otherwise, verify how the final product was generated.",
            })

    return pd.DataFrame(rows)



# ============================================================
# 4.1 Main-weight CR-ML AGCD raster and carbon-stock calculation
# ============================================================

def clean_product_array_for_raster_generation(
    arr: np.ndarray,
    nodata: Optional[float],
    min_valid: float = MIN_VALID_PRODUCT_AGCD,
    max_valid: float = MAX_PRODUCT_AGCD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean a product raster array and return values plus a validity mask."""
    values = arr.astype("float64", copy=True)
    valid = np.isfinite(values)

    if nodata is not None and np.isfinite(nodata):
        valid &= ~np.isclose(values, nodata)

    for inv in RASTER_INVALID_VALUES:
        valid &= ~np.isclose(values, inv)

    valid &= values >= min_valid
    valid &= values <= max_valid
    values[~valid] = np.nan
    return values, valid


def window_row_pixel_area_ha(transform, crs, window) -> np.ndarray:
    """
    Calculate per-pixel area (ha) for each row within a raster window.

    - For both geographic and projected CRS, use Geod to calculate ellipsoidal strip area for each raster row;
    - If pyproj is unavailable for a projected CRS, fall back to affine pixel area;
    - If pyproj is unavailable for a geographic CRS, raise an error because geographic pixel area varies with latitude.
    """
    row0 = int(window.row_off)
    col0 = int(window.col_off)
    h = int(window.height)
    w = int(window.width)

    if Geod is None:
        if crs is not None and not crs.is_geographic:
            cell_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
            return np.full(h, cell_area_m2 / 10000.0, dtype="float64")
        raise ImportError("pyproj is required to calculate pixel area for geographic rasters. Please install pyproj.")

    geod = Geod(ellps="WGS84")
    out = np.full(h, np.nan, dtype="float64")

    for i in range(h):
        r_top = row0 + i
        r_bottom = row0 + i + 1

        # The four corners of the full row strip are expressed in the raster CRS. The window left/right boundaries are used,
        # then divided by the window width to obtain the mean area per pixel for that row.
        x_left_top, y_left_top = transform * (col0, r_top)
        x_right_top, y_right_top = transform * (col0 + w, r_top)
        x_right_bottom, y_right_bottom = transform * (col0 + w, r_bottom)
        x_left_bottom, y_left_bottom = transform * (col0, r_bottom)

        xs = [x_left_top, x_right_top, x_right_bottom, x_left_bottom]
        ys = [y_left_top, y_right_top, y_right_bottom, y_left_bottom]

        if crs is not None and not crs.is_geographic:
            lon, lat = rio_transform(crs, "EPSG:4326", xs, ys)
        else:
            lon, lat = xs, ys

        area_m2, _ = geod.polygon_area_perimeter(lon, lat)
        out[i] = abs(area_m2) / max(w, 1) / 10000.0

    return out


def generate_main_weight_crml_raster_and_stock() -> pd.DataFrame:
    """
    Generate the main-weight CR-ML AGCD raster using MAIN_WEIGHT and calculate aboveground carbon stock.

    Outputs:
    - CR_ML_wXXX_AGCD.tif: main-weight AGCD raster in Mg C ha-1;
    - CR_ML_wXXX_pixel_stock_MgC.tif: per-pixel carbon stock in Mg C per pixel;
    - CR_ML_wXXX_carbon_stock_summary.csv: regional area, mean AGCD, and total carbon stock;
    - CR_ML_wXXX_formula.txt: text record of the fusion formula.
    """
    if not CREATE_MAIN_WEIGHT_CRML_RASTER:
        return pd.DataFrame()

    os.makedirs(MAIN_WEIGHT_RASTER_OUT_DIR, exist_ok=True)

    product_name = weight_to_colname(MAIN_WEIGHT)
    agc_ml_path = PRODUCT_RASTER_PATHS[AGC_ML_COL]
    cr_agc_path = PRODUCT_RASTER_PATHS[CR_AGC_COL]

    if not os.path.exists(agc_ml_path):
        raise FileNotFoundError(f"AGC_ML raster not found: {agc_ml_path}")
    if not os.path.exists(cr_agc_path):
        raise FileNotFoundError(f"CR_AGC raster not found: {cr_agc_path}")
    if DOMAIN_MASK_RASTER is not None and not os.path.exists(DOMAIN_MASK_RASTER):
        raise FileNotFoundError(f"DOMAIN_MASK_RASTER not found: {DOMAIN_MASK_RASTER}")

    out_agcd = os.path.join(MAIN_WEIGHT_RASTER_OUT_DIR, f"{product_name}_AGCD_MgCha.tif")
    out_stock = os.path.join(MAIN_WEIGHT_RASTER_OUT_DIR, f"{product_name}_pixel_stock_MgC.tif")
    out_summary = os.path.join(MAIN_WEIGHT_RASTER_OUT_DIR, f"{product_name}_carbon_stock_summary.csv")
    out_formula = os.path.join(MAIN_WEIGHT_RASTER_OUT_DIR, f"{product_name}_formula.txt")

    print("\n========== Step X: Generate main-weight CR-ML AGCD raster and carbon stock ==========")
    print(f"Main-weight product: {product_name}")
    print(f"Formula: {product_name} = {(1.0 - MAIN_WEIGHT):.2f} * {AGC_ML_COL} + {MAIN_WEIGHT:.2f} * {CR_AGC_COL}")
    print(f"AGC_ML raster: {agc_ml_path}")
    print(f"CR_AGC raster: {cr_agc_path}")
    print(f"Output AGCD raster: {out_agcd}")

    valid_pixel_count = 0
    total_area_ha = 0.0
    total_stock_MgC = 0.0
    sum_agcd_unweighted = 0.0
    min_agcd = np.inf
    max_agcd = -np.inf

    with rasterio.open(agc_ml_path) as ref, rasterio.open(cr_agc_path) as cr_src:
        profile = ref.profile.copy()
        profile.update(
            dtype="float32",
            nodata=OUTPUT_NODATA_VALUE,
            compress="lzw",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER",
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

        if DOMAIN_MASK_RASTER is not None:
            mask_src = rasterio.open(DOMAIN_MASK_RASTER)
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

        with rasterio.open(out_agcd, "w", **profile) as dst_agcd:
            if WRITE_PIXEL_STOCK_RASTER:
                stock_profile = profile.copy()
                stock_profile.update(dtype="float32", nodata=OUTPUT_NODATA_VALUE)
                dst_stock = rasterio.open(out_stock, "w", **stock_profile)
            else:
                dst_stock = None

            try:
                for idx, (_, window) in enumerate(ref.block_windows(1), start=1):
                    if idx % 500 == 0:
                        print(f"  processing window {idx}")

                    agc_raw = ref.read(1, window=window, masked=False)
                    cr_raw = cr_vrt.read(1, window=window, masked=False)

                    agc, valid_agc = clean_product_array_for_raster_generation(ref.read(1, window=window, masked=False), ref.nodata)
                    cr, valid_cr = clean_product_array_for_raster_generation(cr_raw, cr_vrt.nodata)

                    valid = valid_agc & valid_cr

                    if mask_vrt is not None:
                        mask_raw = mask_vrt.read(1, window=window, masked=False).astype("float64")
                        mask_valid = np.isfinite(mask_raw)
                        if mask_vrt.nodata is not None and np.isfinite(mask_vrt.nodata):
                            mask_valid &= ~np.isclose(mask_raw, mask_vrt.nodata)
                        mask_valid &= mask_raw > 0
                        valid &= mask_valid

                    main_agcd = (1.0 - MAIN_WEIGHT) * agc + MAIN_WEIGHT * cr
                    valid &= np.isfinite(main_agcd)
                    valid &= main_agcd >= MIN_VALID_PRODUCT_AGCD
                    valid &= main_agcd <= MAX_PRODUCT_AGCD

                    out_arr = np.full(main_agcd.shape, OUTPUT_NODATA_VALUE, dtype="float32")
                    out_arr[valid] = main_agcd[valid].astype("float32")
                    dst_agcd.write(out_arr, 1, window=window)

                    area_ha_by_row = window_row_pixel_area_ha(ref.transform, ref.crs, window)
                    area_ha = area_ha_by_row[:, None]

                    stock = np.full(main_agcd.shape, np.nan, dtype="float64")
                    stock[valid] = main_agcd[valid] * np.broadcast_to(area_ha, main_agcd.shape)[valid]

                    if dst_stock is not None:
                        stock_arr = np.full(main_agcd.shape, OUTPUT_NODATA_VALUE, dtype="float32")
                        stock_arr[valid] = stock[valid].astype("float32")
                        dst_stock.write(stock_arr, 1, window=window)

                    if np.any(valid):
                        valid_pixel_count += int(valid.sum())
                        total_area_ha += float(np.broadcast_to(area_ha, main_agcd.shape)[valid].sum())
                        total_stock_MgC += float(np.nansum(stock[valid]))
                        sum_agcd_unweighted += float(np.nansum(main_agcd[valid]))
                        min_agcd = min(min_agcd, float(np.nanmin(main_agcd[valid])))
                        max_agcd = max(max_agcd, float(np.nanmax(main_agcd[valid])))
            finally:
                if dst_stock is not None:
                    dst_stock.close()
                cr_vrt.close()
                if mask_vrt is not None:
                    mask_vrt.close()
                if mask_src is not None:
                    mask_src.close()

    mean_agcd_unweighted = sum_agcd_unweighted / valid_pixel_count if valid_pixel_count > 0 else np.nan
    mean_agcd_area_weighted = total_stock_MgC / total_area_ha if total_area_ha > 0 else np.nan

    summary = pd.DataFrame([{
        "product": product_name,
        "CR_AGC_weight": MAIN_WEIGHT,
        "AGC_ML_weight": 1.0 - MAIN_WEIGHT,
        "formula": f"{product_name} = {(1.0 - MAIN_WEIGHT):.2f} * {AGC_ML_COL} + {MAIN_WEIGHT:.2f} * {CR_AGC_COL}",
        "agcd_unit": "Mg C ha-1",
        "stock_unit": "Mg C",
        "valid_pixels": int(valid_pixel_count),
        "total_area_ha": total_area_ha,
        "total_area_km2": total_area_ha / 100.0,
        "mean_AGCD_unweighted_MgCha": mean_agcd_unweighted,
        "mean_AGCD_area_weighted_MgCha": mean_agcd_area_weighted,
        "min_AGCD_MgCha": min_agcd if np.isfinite(min_agcd) else np.nan,
        "max_AGCD_MgCha": max_agcd if np.isfinite(max_agcd) else np.nan,
        "total_carbon_MgC": total_stock_MgC,
        "total_carbon_TgC": total_stock_MgC / 1e6,
        "total_carbon_PgC": total_stock_MgC / 1e9,
        "AGCD_raster": out_agcd,
        "pixel_stock_raster": out_stock if WRITE_PIXEL_STOCK_RASTER else None,
        "domain_mask_raster": DOMAIN_MASK_RASTER,
    }])

    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    with open(out_formula, "w", encoding="utf-8") as f:
        f.write(f"Product: {product_name}\n")
        f.write(f"Formula: {product_name} = {(1.0 - MAIN_WEIGHT):.2f} * {AGC_ML_COL} + {MAIN_WEIGHT:.2f} * {CR_AGC_COL}\n")
        f.write("AGCD unit: Mg C ha-1\n")
        f.write("Pixel carbon stock: AGCD * pixel_area_ha, unit Mg C per pixel\n")
        f.write(f"AGC_ML raster: {agc_ml_path}\n")
        f.write(f"CR_AGC raster: {cr_agc_path}\n")
        f.write(f"Domain mask raster: {DOMAIN_MASK_RASTER}\n")

    print(f"[output] Main-weight CR-ML AGCD raster: {out_agcd}")
    if WRITE_PIXEL_STOCK_RASTER:
        print(f"[output] Per-pixel carbon-stock raster: {out_stock}")
    print(f"[output] Carbon-stock summary: {out_summary}")
    print(f"Total aboveground carbon stock: {total_stock_MgC / 1e9:.3f} Pg C")
    print(f"Analysis area: {total_area_ha / 100.0:.2f} km2")
    print(f"Area-weighted mean AGCD: {mean_agcd_area_weighted:.2f} Mg C ha-1")

    return summary


# ============================================================
# 5. Metric calculation
# ============================================================

def concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom == 0:
        return np.nan

    return float((2 * cov_xy) / denom)


def calc_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    x = observed AGCD
    y = product AGCD
    residual = y - x
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    e = y - x

    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    bias = float(np.mean(e))
    median_bias = float(np.median(e))
    ub_rmse = float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0)))

    denom = np.sum((x - np.mean(x)) ** 2)
    r2_1to1 = float(1 - np.sum(e ** 2) / denom) if denom > 0 else np.nan

    ccc = concordance_correlation_coefficient(x, y)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias(product-observed)": bias,
        "AbsBias": float(abs(bias)),
        "MedianBias(product-observed)": median_bias,
        "AbsMedianBias": float(abs(median_bias)),
        "ubRMSE": ub_rmse,
        "R2_1to1": r2_1to1,
        "CCC": ccc,
    }


def summarize_one_subset(
    df: pd.DataFrame,
    product: str,
    subset_name: str,
    age_group: str,
    validation_mode: str,
    comparison_group: str,
    weight_map: Dict[str, float],
) -> Dict[str, object]:
    sub = df[["AGCD_obs", product, "Age_used"]].dropna().copy()
    n = len(sub)

    row = {
        "validation_mode": validation_mode,
        "comparison_group": comparison_group,
        "product": product,
        "is_weighted_crml": product in weight_map,
        "CR_AGC_weight": weight_map.get(product, np.nan),
        "AGC_ML_weight": 1.0 - weight_map.get(product, np.nan) if product in weight_map else np.nan,
        "is_main_weight": product in weight_map and np.isclose(weight_map[product], MAIN_WEIGHT),
        "subset": subset_name,
        "age_class": age_group,
        "N": int(n),
        "age_min": float(sub["Age_used"].min()) if n else np.nan,
        "age_max": float(sub["Age_used"].max()) if n else np.nan,
        "obs_mean": float(sub["AGCD_obs"].mean()) if n else np.nan,
        "obs_median": float(sub["AGCD_obs"].median()) if n else np.nan,
        "product_mean": float(sub[product].mean()) if n else np.nan,
        "product_median": float(sub[product].median()) if n else np.nan,
        "sufficient_for_metrics": n >= MIN_N_FOR_METRICS,
        "sufficient_for_corr": n >= MIN_N_FOR_CORRELATION,
        "R": np.nan,
        "R2_corr": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "p_value": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "Bias(product-observed)": np.nan,
        "AbsBias": np.nan,
        "MedianBias(product-observed)": np.nan,
        "AbsMedianBias": np.nan,
        "ubRMSE": np.nan,
        "R2_1to1": np.nan,
        "CCC": np.nan,
    }

    if n < MIN_N_FOR_METRICS:
        return row

    x = sub["AGCD_obs"].to_numpy(float)
    y = sub[product].to_numpy(float)

    row.update(calc_metrics(x, y))

    if n >= MIN_N_FOR_CORRELATION and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        reg = linregress(x, y)
        row["R"] = float(reg.rvalue)
        row["R2_corr"] = float(reg.rvalue ** 2)
        row["slope"] = float(reg.slope)
        row["intercept"] = float(reg.intercept)
        row["p_value"] = float(reg.pvalue)

    return row


def summarize_by_age_class(
    df: pd.DataFrame,
    products: List[str],
    validation_mode: str,
    comparison_group: str,
    class_col: str,
    class_order: List[str],
    subset_name: str,
    weight_map: Dict[str, float],
) -> pd.DataFrame:
    rows = []

    for product in products:
        rows.append(
            summarize_one_subset(
                df=df,
                product=product,
                subset_name="all",
                age_group="all",
                validation_mode=validation_mode,
                comparison_group=comparison_group,
                weight_map=weight_map,
            )
        )

        for age_group in class_order:
            sub = df[df[class_col] == age_group].copy()
            rows.append(
                summarize_one_subset(
                    df=sub,
                    product=product,
                    subset_name=subset_name,
                    age_group=age_group,
                    validation_mode=validation_mode,
                    comparison_group=comparison_group,
                    weight_map=weight_map,
                )
            )

    return pd.DataFrame(rows)


def trend_test_residual_vs_age(
    df: pd.DataFrame,
    products: List[str],
    validation_mode: str,
    comparison_group: str,
    weight_map: Dict[str, float],
) -> pd.DataFrame:
    rows = []

    for product in products:
        sub = df[["AGCD_obs", product, "Age_used"]].dropna().copy()
        sub["residual"] = sub[product] - sub["AGCD_obs"]

        n = len(sub)
        row = {
            "validation_mode": validation_mode,
            "comparison_group": comparison_group,
            "product": product,
            "is_weighted_crml": product in weight_map,
            "CR_AGC_weight": weight_map.get(product, np.nan),
            "AGC_ML_weight": 1.0 - weight_map.get(product, np.nan) if product in weight_map else np.nan,
            "is_main_weight": product in weight_map and np.isclose(weight_map[product], MAIN_WEIGHT),
            "N": int(n),
            "linear_slope_per_year": np.nan,
            "linear_slope_per_20yr": np.nan,
            "abs_linear_slope_per_20yr": np.nan,
            "linear_intercept": np.nan,
            "linear_R": np.nan,
            "linear_R2": np.nan,
            "linear_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "mean_residual": float(sub["residual"].mean()) if n else np.nan,
            "median_residual": float(sub["residual"].median()) if n else np.nan,
        }

        if n >= MIN_N_FOR_CORRELATION and sub["Age_used"].std() > 0 and sub["residual"].std() > 0:
            reg = linregress(sub["Age_used"].to_numpy(float), sub["residual"].to_numpy(float))
            row["linear_slope_per_year"] = float(reg.slope)
            row["linear_slope_per_20yr"] = float(reg.slope * 20.0)
            row["abs_linear_slope_per_20yr"] = float(abs(reg.slope * 20.0))
            row["linear_intercept"] = float(reg.intercept)
            row["linear_R"] = float(reg.rvalue)
            row["linear_R2"] = float(reg.rvalue ** 2)
            row["linear_p"] = float(reg.pvalue)

            sp = spearmanr(sub["Age_used"].to_numpy(float), sub["residual"].to_numpy(float))
            row["spearman_rho"] = float(sp.correlation)
            row["spearman_p"] = float(sp.pvalue)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. Common samples, broad age groups, and weight decision table
# ============================================================

def make_common_sample(df: pd.DataFrame, products: List[str]) -> pd.DataFrame:
    missing_products = [p for p in products if p not in df.columns]
    if missing_products:
        raise KeyError(f"Missing product columns required for the common sample: {missing_products}")

    required_cols = ["AGCD_obs", "Age_used", "age20_class", "broad_age_class"] + list(products)
    return df.dropna(subset=required_cols).copy()


def add_residual_columns(df: pd.DataFrame, products: List[str]) -> pd.DataFrame:
    out = df.copy()
    for product in products:
        if product in out.columns:
            out[f"residual_{product}"] = out[product] - out["AGCD_obs"]
    return out


def export_common_sample(
    df: pd.DataFrame,
    products: List[str],
    out_path: str,
    scenario_name: str,
    survey_year_min: Optional[int],
    survey_year_max: Optional[int],
) -> None:
    out = add_residual_columns(df, products)
    out.insert(0, "survey_year_max", survey_year_max)
    out.insert(0, "survey_year_min", survey_year_min)
    out.insert(0, "survey_year_scenario", scenario_name)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")


def _minmax_scale_lower_better(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)

    mn = s.min()
    mx = s.max()
    if np.isclose(mx, mn):
        return pd.Series(0.0, index=s.index)

    return (s - mn) / (mx - mn)


def make_weight_decision_table(
    metrics_age20: pd.DataFrame,
    metrics_broad: pd.DataFrame,
    trend_df: pd.DataFrame,
    scenario_name: str,
    survey_year_min: Optional[int],
    survey_year_max: Optional[int],
) -> pd.DataFrame:
    """
    Score candidate weights using overall accuracy and age-dependent bias diagnostics.

    Lower scores are better and include:
    - all_RMSE
    - all_MAE
    - all_AbsBias
    - abs residual-age slope
    - middle_50_150 AbsBias
    - old_>150 AbsBias

    Note:
    This is not a uniquely authoritative criterion; it is a transparent diagnostic table to support weight selection.
    """
    weighted_all = metrics_age20[
        (metrics_age20["is_weighted_crml"] == True)
        & (metrics_age20["subset"] == "all")
    ].copy()

    trend_weighted = trend_df[trend_df["is_weighted_crml"] == True].copy()

    broad_weighted = metrics_broad[
        metrics_broad["is_weighted_crml"] == True
    ].copy()

    rows = []

    for _, m in weighted_all.iterrows():
        product = m["product"]
        w = m["CR_AGC_weight"]

        tr = trend_weighted[trend_weighted["product"] == product]
        tr_row = tr.iloc[0] if not tr.empty else None

        middle = broad_weighted[
            (broad_weighted["product"] == product)
            & (broad_weighted["subset"] == "broad_age")
            & (broad_weighted["age_class"] == "middle_50_150")
        ]
        old = broad_weighted[
            (broad_weighted["product"] == product)
            & (broad_weighted["subset"] == "broad_age")
            & (broad_weighted["age_class"] == "old_>150")
        ]
        young = broad_weighted[
            (broad_weighted["product"] == product)
            & (broad_weighted["subset"] == "broad_age")
            & (broad_weighted["age_class"] == "young_<50")
        ]

        mid_row = middle.iloc[0] if not middle.empty else None
        old_row = old.iloc[0] if not old.empty else None
        young_row = young.iloc[0] if not young.empty else None

        rows.append({
            "survey_year_scenario": scenario_name,
            "survey_year_min": survey_year_min,
            "survey_year_max": survey_year_max,
            "product": product,
            "CR_AGC_weight": w,
            "AGC_ML_weight": 1.0 - w,
            "is_main_weight": bool(np.isclose(w, MAIN_WEIGHT)),
            "N_all": m["N"],
            "all_R": m["R"],
            "all_CCC": m["CCC"],
            "all_RMSE": m["RMSE"],
            "all_MAE": m["MAE"],
            "all_Bias": m["Bias(product-observed)"],
            "all_AbsBias": m["AbsBias"],
            "slope_per_20yr": tr_row["linear_slope_per_20yr"] if tr_row is not None else np.nan,
            "abs_slope_per_20yr": tr_row["abs_linear_slope_per_20yr"] if tr_row is not None else np.nan,
            "slope_p": tr_row["linear_p"] if tr_row is not None else np.nan,
            "young_N": young_row["N"] if young_row is not None else np.nan,
            "young_Bias": young_row["Bias(product-observed)"] if young_row is not None else np.nan,
            "young_RMSE": young_row["RMSE"] if young_row is not None else np.nan,
            "middle_N": mid_row["N"] if mid_row is not None else np.nan,
            "middle_Bias": mid_row["Bias(product-observed)"] if mid_row is not None else np.nan,
            "middle_AbsBias": mid_row["AbsBias"] if mid_row is not None else np.nan,
            "middle_RMSE": mid_row["RMSE"] if mid_row is not None else np.nan,
            "old_N": old_row["N"] if old_row is not None else np.nan,
            "old_Bias": old_row["Bias(product-observed)"] if old_row is not None else np.nan,
            "old_AbsBias": old_row["AbsBias"] if old_row is not None else np.nan,
            "old_RMSE": old_row["RMSE"] if old_row is not None else np.nan,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    # Composite score; lower is better.
    score_terms = {
        "score_all_RMSE": "all_RMSE",
        "score_all_MAE": "all_MAE",
        "score_all_AbsBias": "all_AbsBias",
        "score_abs_slope": "abs_slope_per_20yr",
        "score_middle_AbsBias": "middle_AbsBias",
        "score_old_AbsBias": "old_AbsBias",
    }

    for score_col, metric_col in score_terms.items():
        out[score_col] = _minmax_scale_lower_better(out[metric_col])

    out["combined_score_lower_better"] = out[list(score_terms.keys())].mean(axis=1, skipna=True)
    out["rank_combined_score"] = out["combined_score_lower_better"].rank(method="min", ascending=True)

    # Best weight for each individual criterion
    lower_better_metrics = [
        "all_RMSE",
        "all_MAE",
        "all_AbsBias",
        "abs_slope_per_20yr",
        "middle_AbsBias",
        "old_AbsBias",
        "combined_score_lower_better",
    ]
    higher_better_metrics = ["all_R", "all_CCC"]

    for metric in lower_better_metrics:
        if metric in out.columns and out[metric].notna().any():
            best_w = out.loc[out[metric].idxmin(), "CR_AGC_weight"]
            out[f"best_weight_for_{metric}"] = best_w

    for metric in higher_better_metrics:
        if metric in out.columns and out[metric].notna().any():
            best_w = out.loc[out[metric].idxmax(), "CR_AGC_weight"]
            out[f"best_weight_for_{metric}"] = best_w

    return out.sort_values(["rank_combined_score", "CR_AGC_weight"]).reset_index(drop=True)


# ============================================================
# 7. Plotting
# ============================================================

def plot_weight_bias_by_age(metrics_age20: pd.DataFrame, out_dir: str, scenario_name: str) -> None:
    sub = metrics_age20[
        (metrics_age20["is_weighted_crml"] == True)
        & (metrics_age20["subset"] == "age20")
    ].copy()

    if sub.empty:
        return

    # Show only key weights to avoid overcrowding.
    weights_to_show = [0.00, 0.10,0.15, 0.20,0.25, 0.50, 1.00]
    x = np.arange(len(AGE20_GROUP_ORDER))

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=300)
    style_axes(ax)

    for w in weights_to_show:
        product = weight_to_colname(w)
        tmp = (
            sub[sub["product"] == product]
            .set_index("age_class")
            .reindex(AGE20_GROUP_ORDER)
        )

        if tmp.empty:
            continue

        y = tmp["Bias(product-observed)"].to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=1.1, markersize=3.5, label=f"w={w:.2f}")

    ax.axhline(0, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE20_GROUP_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Bias = product - inventory\n(Mg C ha$^{-1}$)")
    ax.set_xlabel("Forest age class (yr)")
    ax.set_title(f"Weighted CR-ML age-stratified bias | {scenario_name}")
    ax.legend(frameon=False, ncol=5)

    fig.savefig(os.path.join(out_dir, "weighted_CRML_bias_by_age20_key_weights.pdf"))
    fig.savefig(os.path.join(out_dir, "weighted_CRML_bias_by_age20_key_weights.png"), dpi=600)
    plt.close(fig)


def plot_weight_metric_curves(decision_df: pd.DataFrame, out_dir: str, scenario_name: str) -> None:
    if decision_df.empty:
        return

    metrics = [
        ("all_RMSE", "All RMSE", "lower"),
        ("all_MAE", "All MAE", "lower"),
        ("all_AbsBias", "All |Bias|", "lower"),
        ("abs_slope_per_20yr", "|Residual-age slope|", "lower"),
        ("middle_AbsBias", "Middle |Bias|", "lower"),
        ("old_AbsBias", "Old |Bias|", "lower"),
        ("combined_score_lower_better", "Combined score", "lower"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.8 * len(metrics), 2.7), dpi=220)

    if len(metrics) == 1:
        axes = [axes]

    df = decision_df.sort_values("CR_AGC_weight")

    for ax, (metric, title, direction) in zip(axes, metrics):
        style_axes(ax)
        ax.plot(df["CR_AGC_weight"], df[metric], marker="o", linewidth=1.2)
        ax.axvline(MAIN_WEIGHT, linestyle=(0, (3, 2)), linewidth=0.9)
        ax.set_xlabel("CR_AGC weight")
        ax.set_ylabel(metric)
        ax.set_title(title)

    fig.suptitle(f"Weight diagnostics | {scenario_name}", y=1.04, fontsize=10)
    fig.savefig(os.path.join(out_dir, "weighted_CRML_weight_metric_curves.pdf"))
    fig.savefig(os.path.join(out_dir, "weighted_CRML_weight_metric_curves.png"), dpi=600)
    plt.close(fig)


def plot_all_products_bias_by_age(metrics_age20: pd.DataFrame, out_dir: str, scenario_name: str) -> None:
    sub = metrics_age20[
        (metrics_age20["subset"] == "age20")
    ].copy()

    if sub.empty:
        return

    # Show existing products plus AGC_ML-only, equal-weight, and CR_AGC-only estimates.
    products_to_show = [
        "AGC_ML",
        weight_to_colname(MAIN_WEIGHT),
        "CR_AGC",
        "CR_ML",
        "ESACCI",
        "thurner",
        "ICESAT2",
        "DGVM",
        "ORCHIDEE",
        "LPJGUESS",
    ]
    products_to_show = [p for p in products_to_show if p in sub["product"].unique()]

    x = np.arange(len(AGE20_GROUP_ORDER))

    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=220)
    style_axes(ax)

    for product in products_to_show:
        tmp = (
            sub[sub["product"] == product]
            .set_index("age_class")
            .reindex(AGE20_GROUP_ORDER)
        )

        if tmp.empty:
            continue

        y = tmp["Bias(product-observed)"].to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=1.0, markersize=3.2, label=product)

    ax.axhline(0, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE20_GROUP_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Bias = product - inventory\n(Mg C ha$^{-1}$)")
    ax.set_xlabel("Forest age class (yr)")
    ax.set_title(f"Age-stratified bias of products | {scenario_name}")
    ax.legend(frameon=False, ncol=3)

    fig.savefig(os.path.join(out_dir, "all_products_bias_by_age20_with_weighted_CRML.pdf"))
    fig.savefig(os.path.join(out_dir, "all_products_bias_by_age20_with_weighted_CRML.png"), dpi=600)
    plt.close(fig)


# ============================================================
# 8. Scenario execution
# ============================================================

def write_inventory_summaries(df: pd.DataFrame, products: List[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(
        os.path.join(out_dir, "inventory_with_extracted_products_and_weighted_CRML.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    conversion_summary = (
        df.groupby(["AGCD_obs_note"], dropna=False)
        .size()
        .rename("N")
        .reset_index()
        .sort_values("N", ascending=False)
    )
    conversion_summary.to_csv(
        os.path.join(out_dir, "inventory_agcd_conversion_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    age_count = (
        df.groupby("age20_class", observed=False)
        .size()
        .reindex(AGE20_GROUP_ORDER)
        .rename("N")
        .reset_index()
    )
    age_count.to_csv(
        os.path.join(out_dir, "inventory_age20_counts.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    broad_count = (
        df.groupby("broad_age_class", observed=False)
        .size()
        .reindex(BROAD_AGE_GROUP_ORDER)
        .rename("N")
        .reset_index()
    )
    broad_count.to_csv(
        os.path.join(out_dir, "inventory_broad_age_counts.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if "Survey_year" in df.columns:
        survey_year_count = (
            pd.to_numeric(df["Survey_year"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .rename_axis("Survey_year")
            .reset_index(name="N")
        )
        survey_year_count.to_csv(
            os.path.join(out_dir, "inventory_survey_year_counts.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def run_one_survey_year_scenario(
    inventory_products: pd.DataFrame,
    scenario_name: str,
    survey_year_min: Optional[int],
    survey_year_max: Optional[int],
    base_out_dir: str,
    weighted_cols: List[str],
    weight_map: Dict[str, float],
    dataset_role: str,
    source_inventory_file: str,
    run_weight_sensitivity: bool,
    run_all_products_comparison: bool,
) -> Dict[str, pd.DataFrame]:
    """
    Run one survey-year scenario for a specified dataset role.

    development_weight_selection: run weight-sensitivity diagnostics only;
    external_independent_validation: run all-product independent validation only after the weight has been fixed.
    """
    scenario_dir = os.path.join(base_out_dir, safe_filename(scenario_name))
    os.makedirs(scenario_dir, exist_ok=True)

    scenario_df = filter_by_survey_year(
        inventory_products,
        survey_year_min=survey_year_min,
        survey_year_max=survey_year_max,
    )

    scenario_df.to_csv(
        os.path.join(scenario_dir, "year_filtered_inventory_with_products_before_common_sample.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 10 + f" {dataset_role} | survey-year scenario: {scenario_name} " + "=" * 10)
    print(f"Year range: {survey_year_min} - {survey_year_max}")
    print(f"Sample size after year filtering: {len(scenario_df)}")
    print(f"Output directory: {scenario_dir}")

    if len(scenario_df) == 0:
        warnings.warn(f"{dataset_role} survey-year scenario {scenario_name} is empty.")
        return {}

    is_external = dataset_role == "external_independent_validation"

    metrics_weight_age20 = pd.DataFrame()
    metrics_weight_broad = pd.DataFrame()
    trend_weight = pd.DataFrame()
    decision_weight = pd.DataFrame()
    metrics_all_age20 = pd.DataFrame()
    metrics_all_broad = pd.DataFrame()
    trend_all = pd.DataFrame()
    weight_eval_df = pd.DataFrame()
    all_eval_df = pd.DataFrame()
    weight_products: List[str] = []
    all_products: List[str] = []

    # -------------------------
    # A. Development samples: weight-sensitivity diagnostics
    # -------------------------
    if run_weight_sensitivity:
        if is_external:
            raise ValueError("External independent-validation data must not be used for weight sensitivity or weight selection.")

        weight_products = [AGC_ML_COL, CR_AGC_COL] + weighted_cols

        if USE_COMMON_SAMPLE_FOR_WEIGHT_SENSITIVITY:
            weight_eval_df = make_common_sample(scenario_df, weight_products)
        else:
            weight_eval_df = scenario_df.copy()

        weight_common_path = os.path.join(scenario_dir, "common_sample_for_weight_sensitivity.csv")
        export_common_sample(
            df=weight_eval_df,
            products=weight_products,
            out_path=weight_common_path,
            scenario_name=scenario_name,
            survey_year_min=survey_year_min,
            survey_year_max=survey_year_max,
        )
        print(f"[development weight diagnostic] common-sample N={len(weight_eval_df)}")
        print(f"[output] {weight_common_path}")

        write_inventory_summaries(
            weight_eval_df,
            weight_products,
            os.path.join(scenario_dir, "weight_sensitivity_sample_summary"),
        )

        metrics_weight_age20 = summarize_by_age_class(
            df=weight_eval_df,
            products=weight_products,
            validation_mode="development_weight_diagnostic",
            comparison_group="AGC_ML_CR_AGC_weighted_CRML",
            class_col="age20_class",
            class_order=AGE20_GROUP_ORDER,
            subset_name="age20",
            weight_map=weight_map,
        )
        metrics_weight_broad = summarize_by_age_class(
            df=weight_eval_df,
            products=weight_products,
            validation_mode="development_weight_diagnostic",
            comparison_group="AGC_ML_CR_AGC_weighted_CRML",
            class_col="broad_age_class",
            class_order=BROAD_AGE_GROUP_ORDER,
            subset_name="broad_age",
            weight_map=weight_map,
        )
        trend_weight = trend_test_residual_vs_age(
            df=weight_eval_df,
            products=weight_products,
            validation_mode="development_weight_diagnostic",
            comparison_group="AGC_ML_CR_AGC_weighted_CRML",
            weight_map=weight_map,
        )

        decision_weight = make_weight_decision_table(
            metrics_age20=metrics_weight_age20,
            metrics_broad=metrics_weight_broad,
            trend_df=trend_weight,
            scenario_name=scenario_name,
            survey_year_min=survey_year_min,
            survey_year_max=survey_year_max,
        )

        metrics_weight_age20 = add_scenario_metadata(metrics_weight_age20, scenario_name, survey_year_min, survey_year_max)
        metrics_weight_broad = add_scenario_metadata(metrics_weight_broad, scenario_name, survey_year_min, survey_year_max)
        trend_weight = add_scenario_metadata(trend_weight, scenario_name, survey_year_min, survey_year_max)

        metrics_weight_age20 = add_dataset_metadata(metrics_weight_age20, dataset_role, source_inventory_file, False)
        metrics_weight_broad = add_dataset_metadata(metrics_weight_broad, dataset_role, source_inventory_file, False)
        trend_weight = add_dataset_metadata(trend_weight, dataset_role, source_inventory_file, False)
        decision_weight = add_dataset_metadata(decision_weight, dataset_role, source_inventory_file, False)

        metrics_weight_age20.to_csv(os.path.join(scenario_dir, "weighted_CRML_metrics_age20.csv"), index=False, encoding="utf-8-sig")
        metrics_weight_broad.to_csv(os.path.join(scenario_dir, "weighted_CRML_metrics_broad_age.csv"), index=False, encoding="utf-8-sig")
        trend_weight.to_csv(os.path.join(scenario_dir, "weighted_CRML_trend_residual_vs_age.csv"), index=False, encoding="utf-8-sig")
        decision_weight.to_csv(os.path.join(scenario_dir, "weighted_CRML_decision_table.csv"), index=False, encoding="utf-8-sig")

    # -------------------------
    # B. 811 external samples: independent all-product comparison after fixing the weight
    # -------------------------
    if run_all_products_comparison:
        if not is_external:
            warnings.warn(
                "The current all-product comparison dataset is not external_independent_validation."
                "To avoid ambiguity, the main workflow does not run this branch on development samples."
            )

        # Independent validation evaluates only the prespecified MAIN_WEIGHT and does not scan alternative candidate weights,
        # preventing post-hoc weight selection after inspecting independent-validation results.
        all_products = [
            AGC_ML_COL,
            CR_AGC_COL,
            weight_to_colname(MAIN_WEIGHT),
        ]

        # Include only products with sufficient valid values in the current independent dataset,
        # so a missing raster cannot create an all-NaN column that empties the common sample.
        for p in ALL_PRODUCTS_COMMON_SAMPLE_PRODUCTS:
            if (
                p in scenario_df.columns
                and p not in all_products
                and int(scenario_df[p].notna().sum()) >= MIN_N_FOR_METRICS
            ):
                all_products.append(p)

        if USE_COMMON_SAMPLE_FOR_ALL_PRODUCTS_COMPARISON:
            all_eval_df = make_common_sample(scenario_df, all_products)
        else:
            all_eval_df = scenario_df.copy()

        all_common_path = os.path.join(scenario_dir, "common_sample_for_all_products_with_selected_weights.csv")
        export_common_sample(
            df=all_eval_df,
            products=all_products,
            out_path=all_common_path,
            scenario_name=scenario_name,
            survey_year_min=survey_year_min,
            survey_year_max=survey_year_max,
        )
        print(f"[811 external independent validation] all-product common-sample N={len(all_eval_df)}")
        print(f"[output] {all_common_path}")

        if len(all_eval_df) >= MIN_N_FOR_METRICS:
            metrics_all_age20 = summarize_by_age_class(
                df=all_eval_df,
                products=all_products,
                validation_mode="external_independent_common",
                comparison_group="all_products_plus_selected_weights",
                class_col="age20_class",
                class_order=AGE20_GROUP_ORDER,
                subset_name="age20",
                weight_map=weight_map,
            )
            metrics_all_broad = summarize_by_age_class(
                df=all_eval_df,
                products=all_products,
                validation_mode="external_independent_common",
                comparison_group="all_products_plus_selected_weights",
                class_col="broad_age_class",
                class_order=BROAD_AGE_GROUP_ORDER,
                subset_name="broad_age",
                weight_map=weight_map,
            )
            trend_all = trend_test_residual_vs_age(
                df=all_eval_df,
                products=all_products,
                validation_mode="external_independent_common",
                comparison_group="all_products_plus_selected_weights",
                weight_map=weight_map,
            )

            metrics_all_age20 = add_scenario_metadata(metrics_all_age20, scenario_name, survey_year_min, survey_year_max)
            metrics_all_broad = add_scenario_metadata(metrics_all_broad, scenario_name, survey_year_min, survey_year_max)
            trend_all = add_scenario_metadata(trend_all, scenario_name, survey_year_min, survey_year_max)

            metrics_all_age20 = add_dataset_metadata(metrics_all_age20, dataset_role, source_inventory_file, True)
            metrics_all_broad = add_dataset_metadata(metrics_all_broad, dataset_role, source_inventory_file, True)
            trend_all = add_dataset_metadata(trend_all, dataset_role, source_inventory_file, True)

            metrics_all_age20.to_csv(
                os.path.join(scenario_dir, "all_products_metrics_age20_with_selected_weights.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            metrics_all_broad.to_csv(
                os.path.join(scenario_dir, "all_products_metrics_broad_age_with_selected_weights.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            trend_all.to_csv(
                os.path.join(scenario_dir, "all_products_trend_residual_vs_age_with_selected_weights.csv"),
                index=False,
                encoding="utf-8-sig",
            )
        else:
            warnings.warn(
                f"{dataset_role} | {scenario_name} all-product common-sample N={len(all_eval_df)}, "
                f"below MIN_N_FOR_METRICS={MIN_N_FOR_METRICS}."
            )

    # -------------------------
    # C. Figures: development samples generate weight plots; independent samples generate final product-comparison plots.
    # -------------------------
    if MAKE_FIGURES:
        fig_dir = os.path.join(scenario_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        if run_weight_sensitivity and not metrics_weight_age20.empty:
            plot_weight_bias_by_age(metrics_weight_age20, fig_dir, scenario_name)
            plot_weight_metric_curves(decision_weight, fig_dir, scenario_name)

        if run_all_products_comparison and not metrics_all_age20.empty:
            plot_all_products_bias_by_age(metrics_all_age20, fig_dir, scenario_name)

    scenario_summary = pd.DataFrame([{
        "dataset_role": dataset_role,
        "source_inventory_file": source_inventory_file,
        "is_external_independent_validation": bool(is_external),
        "survey_year_scenario": scenario_name,
        "survey_year_min": survey_year_min,
        "survey_year_max": survey_year_max,
        "N_after_year_filter": int(len(scenario_df)),
        "N_weight_common_sample": int(len(weight_eval_df)) if run_weight_sensitivity else np.nan,
        "N_all_products_common_sample": int(len(all_eval_df)) if run_all_products_comparison else np.nan,
        "weight_common_products": "|".join(weight_products),
        "all_products_common_products": "|".join(all_products),
        "run_weight_sensitivity": bool(run_weight_sensitivity),
        "run_all_products_comparison": bool(run_all_products_comparison),
        "output_dir": scenario_dir,
    }])

    scenario_summary.to_csv(
        os.path.join(scenario_dir, "scenario_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "scenario_summary": scenario_summary,
        "metrics_weight_age20": metrics_weight_age20,
        "metrics_weight_broad": metrics_weight_broad,
        "trend_weight": trend_weight,
        "decision_weight": decision_weight,
        "metrics_all_age20": metrics_all_age20,
        "metrics_all_broad": metrics_all_broad,
        "trend_all": trend_all,
    }


# ============================================================
# 9. Main workflow
# ============================================================

def _print_inventory_overview(df: pd.DataFrame, label: str) -> None:
    print(f"{label} sample size after basic QC: {len(df)}")
    if len(df) == 0:
        return
    print(
        f"  AGCD_obs: mean={df['AGCD_obs'].mean():.2f}, "
        f"median={df['AGCD_obs'].median():.2f}, max={df['AGCD_obs'].max():.2f}"
    )
    print(
        f"  Age_used: min={df['Age_used'].min():.1f}, "
        f"median={df['Age_used'].median():.1f}, max={df['Age_used'].max():.1f}"
    )
    if "Survey_year" in df.columns:
        year = pd.to_numeric(df["Survey_year"], errors="coerce")
        print(f"  valid Survey_year values: {int(year.notna().sum())}/{len(df)}")
        if year.notna().any():
            print(f"  Survey_year range: {int(year.min())} - {int(year.max())}")


def _extract_and_build_weighted_products(
    inventory: pd.DataFrame,
    raster_paths: Dict[str, str],
    role_out_dir: str,
    file_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    os.makedirs(role_out_dir, exist_ok=True)

    products, extraction_summary = extract_raster_values_at_points(
        df=inventory,
        lon_col=LON_COL,
        lat_col=LAT_COL,
        raster_paths=raster_paths,
        multipliers=PRODUCT_MULTIPLIERS,
    )
    products.to_csv(
        os.path.join(role_out_dir, f"{file_prefix}_with_extracted_products.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    extraction_summary.to_csv(
        os.path.join(role_out_dir, f"{file_prefix}_product_extraction_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    products, formula_summary, weighted_cols = build_weighted_crml_products(products)
    products.to_csv(
        os.path.join(role_out_dir, f"{file_prefix}_with_extracted_products_and_weighted_CRML.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    formula_summary.to_csv(
        os.path.join(role_out_dir, "weighted_CRML_formula_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return products, extraction_summary, formula_summary, weighted_cols


def main() -> None:
    set_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    development_dir = os.path.join(OUT_DIR, "development_weight_selection")
    external_dir = os.path.join(OUT_DIR, "external_independent_validation_811")
    separation_dir = os.path.join(OUT_DIR, "sample_separation_audit")
    os.makedirs(development_dir, exist_ok=True)
    os.makedirs(external_dir, exist_ok=True)
    os.makedirs(separation_dir, exist_ok=True)

    print("========== Step 1: Read development and 811 external independent samples ==========")
    print(f"Development inventory: {DEVELOPMENT_INVENTORY_FILE}")
    print(f"External validation inventory: {INDEPENDENT_VALIDATION_FILE}")

    development_raw = prepare_inventory_data(
        DEVELOPMENT_INVENTORY_FILE,
        sheet_name=DEVELOPMENT_INVENTORY_SHEET_NAME,
    )
    external_raw = prepare_inventory_data(
        INDEPENDENT_VALIDATION_FILE,
        sheet_name=INDEPENDENT_INVENTORY_SHEET_NAME,
    )

    _print_inventory_overview(development_raw, "development file")
    _print_inventory_overview(external_raw, "811 external file")

    if REMOVE_INDEPENDENT_FROM_DEVELOPMENT:
        development, external, overlap_audit = separate_development_and_external_samples(
            development_raw,
            external_raw,
        )
    else:
        development = development_raw.copy()
        external = external_raw.copy()
        development["sample_role"] = "development_weight_selection"
        development["is_external_independent_validation"] = False
        external["sample_role"] = "external_independent_validation"
        external["is_external_independent_validation"] = True
        overlap_audit = pd.DataFrame()

    print("\n========== Step 2: Audit sample separation ==========")
    print(f"Development-file N after QC: {len(development_raw)}")
    print(f"External overlapping records removed from development file: {len(overlap_audit)}")
    print(f"Final development-sample N: {len(development)}")
    print(f"Final external independent-sample N: {len(external)}")

    if len(external) != 811:
        warnings.warn(
            f"The independent file name indicates 811 samples, but basic QC retained {len(external)} records."
            "Check exclusions due to missing coordinates, AGCD, forest age, implausible survey years, or zero AGCD."
        )

    development.to_csv(
        os.path.join(separation_dir, "development_inventory_after_excluding_external811.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    external.to_csv(
        os.path.join(separation_dir, "external_independent_inventory_after_qc.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    overlap_audit.to_csv(
        os.path.join(separation_dir, "records_removed_from_development_as_external_overlap.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Coordinate-level verification after sample separation
    dev_coord = set(build_coordinate_key(development).astype(str))
    ext_coord = set(build_coordinate_key(external).astype(str))
    remaining_coord_overlap = dev_coord.intersection(ext_coord)
    audit_summary = pd.DataFrame([{
        "development_qc_before_exclusion": len(development_raw),
        "external_qc": len(external),
        "removed_from_development": len(overlap_audit),
        "development_after_exclusion": len(development),
        "remaining_coordinate_key_overlap": len(remaining_coord_overlap),
        "remove_coordinate_only_overlaps": REMOVE_COORDINATE_ONLY_OVERLAPS,
        "development_file": DEVELOPMENT_INVENTORY_FILE,
        "external_file": INDEPENDENT_VALIDATION_FILE,
    }])
    audit_summary.to_csv(
        os.path.join(separation_dir, "sample_separation_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    if remaining_coord_overlap:
        warnings.warn(
            f"After separation, {len(remaining_coord_overlap)} coordinate keys remain shared by the development and external datasets."
            "Check coordinate precision or duplicated plots."
        )

    print("\n========== Step 3: Extract raster values and construct fixed-weight products ==========")

    # Development samples require only the two endpoint products for weight sensitivity; remote-sensing/DGVM products are not used for weight selection.
    development_raster_paths = {
        AGC_ML_COL: PRODUCT_RASTER_PATHS[AGC_ML_COL],
        CR_AGC_COL: PRODUCT_RASTER_PATHS[CR_AGC_COL],
    }
    development_products, dev_extract_summary, formula_summary, weighted_cols = _extract_and_build_weighted_products(
        inventory=development,
        raster_paths=development_raster_paths,
        role_out_dir=development_dir,
        file_prefix="development",
    )

    # Extract all target products for the 811 external samples, but do not calculate any weight-selection table.
    external_products, ext_extract_summary, _, external_weighted_cols = _extract_and_build_weighted_products(
        inventory=external,
        raster_paths=PRODUCT_RASTER_PATHS,
        role_out_dir=external_dir,
        file_prefix="external811",
    )

    if weighted_cols != external_weighted_cols:
        raise RuntimeError("Weighted columns constructed for development and independent datasets are inconsistent.")

    weight_map = {weight_to_colname(w): w for w in WEIGHTS}

    # Optional consistency check between an existing CR_ML product and the reconstructed main-weight product; run only if CR_ML is available for external samples.
    current_check = check_current_final_vs_weighted(external_products, weighted_cols)
    if not current_check.empty:
        current_check.to_csv(
            os.path.join(OUT_DIR, f"current_CR_ML_vs_recalculated_{weight_to_colname(MAIN_WEIGHT)}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # Generate the main-weight raster and carbon stock only once; this step is independent of dataset role.
    main_weight_raster_summary = generate_main_weight_crml_raster_and_stock()
    if not main_weight_raster_summary.empty:
        main_weight_raster_summary.to_csv(
            os.path.join(OUT_DIR, f"main_weight_{weight_to_colname(MAIN_WEIGHT)}_raster_stock_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print("\n========== Step 4: Run development weight sensitivity and 811-sample independent validation ==========")

    dev_outputs: Dict[str, List[pd.DataFrame]] = {
        "scenario_summary": [],
        "metrics_weight_age20": [],
        "metrics_weight_broad": [],
        "trend_weight": [],
        "decision_weight": [],
    }
    ext_outputs: Dict[str, List[pd.DataFrame]] = {
        "scenario_summary": [],
        "metrics_all_age20": [],
        "metrics_all_broad": [],
        "trend_all": [],
    }

    for scenario in SURVEY_YEAR_SCENARIOS:
        scenario_name = str(scenario["name"])
        survey_year_min = scenario.get("min")
        survey_year_max = scenario.get("max")

        dev_result = run_one_survey_year_scenario(
            inventory_products=development_products,
            scenario_name=scenario_name,
            survey_year_min=survey_year_min,
            survey_year_max=survey_year_max,
            base_out_dir=development_dir,
            weighted_cols=weighted_cols,
            weight_map=weight_map,
            dataset_role="development_weight_selection",
            source_inventory_file=DEVELOPMENT_INVENTORY_FILE,
            run_weight_sensitivity=True,
            run_all_products_comparison=False,
        )
        for key in dev_outputs:
            value = dev_result.get(key)
            if isinstance(value, pd.DataFrame) and not value.empty:
                dev_outputs[key].append(value)

        ext_result = run_one_survey_year_scenario(
            inventory_products=external_products,
            scenario_name=scenario_name,
            survey_year_min=survey_year_min,
            survey_year_max=survey_year_max,
            base_out_dir=external_dir,
            weighted_cols=weighted_cols,
            weight_map=weight_map,
            dataset_role="external_independent_validation",
            source_inventory_file=INDEPENDENT_VALIDATION_FILE,
            run_weight_sensitivity=False,
            run_all_products_comparison=True,
        )
        for key in ext_outputs:
            value = ext_result.get(key)
            if isinstance(value, pd.DataFrame) and not value.empty:
                ext_outputs[key].append(value)

    print("\n========== Step 5: Write root-level summary tables ==========")

    # Preserve legacy output filenames so downstream Fig. 2 scripts can read the same files.
    # The following weighted summary files are derived exclusively from development samples.
    dev_summary_names = {
        "metrics_weight_age20": "survey_year_sensitivity_weighted_CRML_metrics_age20.csv",
        "metrics_weight_broad": "survey_year_sensitivity_weighted_CRML_metrics_broad_age.csv",
        "trend_weight": "survey_year_sensitivity_weighted_CRML_trend_residual_vs_age.csv",
        "decision_weight": "survey_year_sensitivity_weighted_CRML_decision_table.csv",
    }
    for key, filename in dev_summary_names.items():
        if dev_outputs[key]:
            out_df = pd.concat(dev_outputs[key], ignore_index=True)
            out_df.to_csv(os.path.join(OUT_DIR, filename), index=False, encoding="utf-8-sig")
            print(f"[development summary] {os.path.join(OUT_DIR, filename)}")

    # Preserve legacy output filenames so downstream Fig. 7 scripts read the 811 independent-validation results.
    ext_summary_names = {
        "metrics_all_age20": "survey_year_sensitivity_all_products_metrics_age20_with_selected_weights.csv",
        "metrics_all_broad": "survey_year_sensitivity_all_products_metrics_broad_age_with_selected_weights.csv",
        "trend_all": "survey_year_sensitivity_all_products_trend_residual_vs_age_with_selected_weights.csv",
    }
    for key, filename in ext_summary_names.items():
        if ext_outputs[key]:
            out_df = pd.concat(ext_outputs[key], ignore_index=True)
            out_df.to_csv(os.path.join(OUT_DIR, filename), index=False, encoding="utf-8-sig")
            print(f"[811 independent-validation summary] {os.path.join(OUT_DIR, filename)}")

    scenario_frames = dev_outputs["scenario_summary"] + ext_outputs["scenario_summary"]
    if scenario_frames:
        pd.concat(scenario_frames, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "survey_year_sensitivity_scenario_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print("\n========== Complete ==========")
    print(f"Output directory: {OUT_DIR}")
    print("\nDataset-role notes:")
    print("1) survey_year_sensitivity_weighted_CRML_*: derived from development samples and used only for weight diagnostics.")
    print("2) survey_year_sensitivity_all_products_*: derived from the independent-validation inventory and used only for external independent validation.")
    print("3) external_independent_validation_811/ does not contain a weight decision table.")
    print("4) development_weight_selection/ does not contain final independent-validation results for remote-sensing products or DGVMs.")
    print("5) sample_separation_audit/ documents whether the 811 external records were completely removed from the development dataset.")
    print("\nRecommended manuscript wording:")
    print("- Weight-sensitivity figure: development-sample constrained weight-sensitivity diagnosis.")
    print("- Age-bias figure: external independent inventory validation.")
    print("- The 811 external samples were not used for ML/CR model development or CR-ML weight selection.")


if __name__ == "__main__":
    main()
