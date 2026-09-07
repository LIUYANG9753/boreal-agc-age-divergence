# Reproducible machine-learning training workflow for forest carbon density

## Overview

`train_forest_carbon_ml.py` is a publication-oriented refactoring of the study
script used to compare machine-learning models for forest aboveground carbon
density. The workflow retains the main analytical components of the original
code while replacing local hard-coded paths with command-line arguments and
saving all relevant settings and outputs.

## Analytical workflow

1. Read the point-level training CSV.
2. Verify the target and predictor columns.
3. Remove rows with missing target values.
4. Median-impute missing numeric predictor values.
5. Detect multivariate outliers using modified Z scores.
6. Compare Gradient Boosting, Extra Trees, Random Forest, and XGBoost.
7. Evaluate VarianceThreshold, SelectKBest, RFE, and SelectFromModel feature selection.
8. Select the model with the highest independent-test R².
9. Optionally identify the best model containing exactly 15 predictors.
10. Refit bootstrap models and estimate predictive standard deviation and 95% intervals.
11. Save fitted models, selected predictors, model metrics, SHAP importance, and diagnostic figures.

## Important terminology

`--valid-feature-fraction` corresponds to the `proportion=0.93` setting in the
original script. It controls multivariate outlier filtering: a sample is retained
when at least this fraction of numeric variables has a modified Z score below
the specified threshold. It is **not** an extrapolation proportion.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Example

```bash
python train_forest_carbon_ml.py \
  --input biomass_training_data.csv \
  --target biomass_carbon \
  --output-dir results/ml_training \
  --selected-k 15 \
  --n-bootstrap 100 \
  --z-threshold 5 \
  --valid-feature-fraction 0.93 \
  --random-seed 42
```

A custom feature list can be supplied as a JSON file:

```json
{
  "features": ["b1_clim", "b2_soil", "Age", "arid"]
}
```

Then run:

```bash
python train_forest_carbon_ml.py \
  --input biomass_training_data.csv \
  --target biomass_carbon \
  --features-file features.json
```

## Principal outputs

- `model_comparison_results.csv`
- `multivariate_outlier_report.csv`
- `best_model.pkl`
- `best_features.json`
- `predictions_with_uncertainty_best.csv`
- `bootstrap_models_best/`
- `shap_importance_best.csv`
- `shap_summary_best.png`
- `shap_summary_bar_best.png`
- corresponding outputs for the best fixed-k model
- `feature_pool.json`
- `run_configuration.json`
- `training.log`

## Reproducibility notes

The train–test split, machine-learning estimators, and bootstrap sampling are
controlled by an explicit random seed. Model performance may still vary slightly
across software versions, CPU architectures, and parallel numerical libraries.

The original workflow compares models on a single 80/20 random split. This
refactoring preserves that design. For a manuscript emphasizing spatial
generalization, spatially blocked or geographically stratified validation should
be reported separately rather than being implied by this script.

## Data availability

The code package does not include the study training observations or proprietary
input layers. Users must provide a CSV containing the target variable and all
selected predictors.
