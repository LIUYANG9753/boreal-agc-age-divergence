# CR-ML Forest AGCD Analysis: Reproducible Code

## Overview

This repository contains the analysis and figure-generation scripts used for CR-ML weight-sensitivity diagnostics, age-dependent forest AGCD diagnostics, and external independent validation.

The publication version of the code preserves the analytical logic of the original scripts while improving portability and readability. All comments, docstrings, warnings, and console messages are in English. Personal absolute paths have been replaced with project-relative paths that can be configured using environment variables.

## Scientific analysis roles

The workflow deliberately separates model-development diagnostics from external independent validation.

- **Development samples** are used only for CR-ML weight-sensitivity diagnostics and weight selection.
- **811 external independent samples** are not used for weight selection. They are used only after the final CR_AGC weight has been fixed.
- The fusion equation is:

  `CR_ML_w = (1 - w) * AGC_ML + w * CR_AGC`

- The default final CR_AGC weight is `w = 0.10`.
- Bias is defined as `product - inventory`.

These distinctions should be retained when describing the analysis in the manuscript and when modifying the code.

## Repository contents

| File | Purpose |
|---|---|
| `01_weight_sensitivity_and_external_validation.py` | Main workflow. Separates development and independent samples, extracts raster products, constructs weighted CR-ML estimates, performs weight diagnostics, and calculates external-validation summaries. |
| `02_fig3_weight_sensitivity.py` | Generates the three-panel CR-ML weight-sensitivity figure from development-sample diagnostics. |
| `03_fig5_age_dependence_diagnostics.py` | Generates the extended age-dependence diagnostic figure using the full inventory database. CR-ML is treated as a model reference rather than an independent-performance ranking in the full-inventory figure. |
| `04_independent_validation.py` | Produces point-scale and 1-degree grid-mean external-validation figures and reports R and RMSE. |
| `requirements.txt` | Python package requirements. |
| `validate_repository.py` | Lightweight pre-submission checks for syntax, Chinese characters, and hard-coded Windows drive paths. |

## Recommended Python environment

Python 3.10 or newer is recommended.

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For legacy `.xls` inventory files, `xlrd` is required. For `.xlsx` files, `openpyxl` is required.

## Project directory structure

The scripts use the following portable layout by default:

```text
project_root/
├── 01_weight_sensitivity_and_external_validation.py
├── 02_fig3_weight_sensitivity.py
├── 03_fig7_age_dependence_diagnostics.py
├── 04_independent_validation.py
├── requirements.txt
├── validate_repository.py
├── data/
│   ├── inventory/
│   │   ├── development_inventory.xls
│   │   └── independent_validation_811.xls
│   ├── rasters/
│   │   ├── AGCD_ML.tif
│   │   ├── AGCD_CR.tif
│   │   ├── ESACCI_2020_hansen_clip.tif
│   │   ├── thurner_hansen_clip.tif
│   │   ├── DGVM_Ensemble_Mean_AGCD_2020_MgCha.tif
│   │   ├── ORCHIDEE_Aboveground_Carbon_2020_MgCha.tif
│   │   ├── LPJ-GUESS_Aboveground_Carbon_2020_MgCha.tif
│   │   └── ICESAT2_hansen_clip.tif
│   └── derived/
│       └── spatial_stats_weighted_CRML.csv
└── outputs/
```

Input data are not included in this code bundle. File names can be changed directly in the user-settings section of each script if the archived data use different names.

## Path configuration

Three optional environment variables can be used without editing the scripts:

- `CRML_PROJECT_ROOT`: project directory containing the scripts.
- `CRML_DATA_DIR`: directory containing `inventory/`, `rasters/`, and `derived/`.
- `CRML_OUTPUT_DIR`: directory where analysis outputs and figures are written.

If these variables are not set, each script uses its own directory as the project root, with `data/` and `outputs/` subdirectories.

Example on Windows PowerShell:

```powershell
$env:CRML_PROJECT_ROOT = "D:\CRML_publication_code"
$env:CRML_DATA_DIR = "D:\CRML_publication_data"
$env:CRML_OUTPUT_DIR = "D:\CRML_results"
```

Example on macOS/Linux:

```bash
export CRML_PROJECT_ROOT=/path/to/CRML_publication_code
export CRML_DATA_DIR=/path/to/CRML_publication_data
export CRML_OUTPUT_DIR=/path/to/CRML_results
```

## Recommended execution order

### 1. Main development and independent-validation analysis

```bash
python 01_weight_sensitivity_and_external_validation.py
```

This script performs sample-separation auditing, raster extraction, weighted-product construction, development-sample weight diagnostics, and fixed-weight external independent validation.

### 2. Figure 3: weight-sensitivity diagnostics

```bash
python 02_fig3_weight_sensitivity.py
```

This script reads the development-sample summary tables produced by step 1. It does not use the 811 external samples to select the weight.

### 3. Figure 5: age-dependence diagnostics

```bash
python 03_fig5_age_dependence_diagnostics.py
```

This script imports shared preprocessing and raster-extraction functions from `01_weight_sensitivity_and_external_validation.py` and constructs the full-inventory age-diagnostic figure.

### 4. Independent-validation plotting

```bash
python 04_independent_validation.py
```

The input table for this script must contain the observed AGCD field, latitude/longitude fields, and the product columns listed in `PRODUCT_COLS`, including the fixed-weight `CR_ML_w010` estimate.

## Reproducibility settings that should be reported

The following settings materially affect the analysis and should be documented in the manuscript, supplement, or archived repository release:

- Final CR_AGC weight (`MAIN_WEIGHT` / `SELECTED_WEIGHT` / `SELECTED_CR_WEIGHT`; default 0.10).
- Candidate weight grid used in the development-sample sensitivity analysis.
- Survey-year scenario.
- Age source and age-class definitions.
- Raster NoData rules and product-specific unit multipliers.
- Common-sample settings for cross-product comparisons.
- Minimum sample sizes for metrics and age classes.
- Spatial block-bootstrap grid size, number of replicates, confidence level, and random seed.
- 1-degree grid-mean validation minimum number of inventory points per grid cell.

## Pre-submission code check

Run:

```bash
python validate_repository.py
```

The checker verifies that all publication scripts:

1. compile successfully;
2. contain no Chinese characters; and
3. contain no hard-coded Windows drive paths such as `E:\...`.

## Notes for journal archiving

Before depositing the final repository in Zenodo, Figshare, Dryad, GitHub, or an institutional archive:

- Add the exact data-access instructions and persistent identifiers for datasets that can be shared.
- Document any restricted or third-party data that cannot be redistributed.
- Add a software license selected by the authors/institution.
- Tag the exact submitted code version (for example, `v1.0.0`).
- Record the package versions used for the final manuscript results.
- Keep generated outputs separate from source code unless the journal requires archived intermediate results.
