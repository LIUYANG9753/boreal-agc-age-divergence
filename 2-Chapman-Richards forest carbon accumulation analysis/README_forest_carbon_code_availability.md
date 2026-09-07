# Code availability and workflow: forest carbon accumulation analysis

## Purpose

This repository-style code package provides a reproducible Python workflow for fitting forest-type-specific biomass carbon accumulation curves as a function of stand age. The analysis uses a Chapman-Richards growth model and its first derivative to estimate carbon accumulation trajectories and age-dependent carbon sequestration rates.

The workflow was fitted Chapman-Richards curves by forest type, filtered outliers, calculated confidence intervals, estimated the age at which the curve approaches its asymptote, and plotted both fitted carbon density curves and their first derivatives.

## Files included

- `forest_carbon_growth_analysis.py`: refactored, English-language, command-line Python script for the full analysis.
- `requirements_forest_carbon.txt`: Python package requirements.
- `README_forest_carbon_code_availability.md`: this documentation file.
- `SCI_code_availability_statement.md`: manuscript-ready code availability text.

## Input data

The script accepts `.csv`, `.xls`, or `.xlsx` files. The input table must include the following columns:

| Column | Description |
|---|---|
| `TYPE` | Forest type or ecological group used for stratified fitting. |
| `stand_age` | Stand age in years. |
| `biomass_carbon` | Biomass carbon density, typically Mg C ha^-1. |
| `biorate` | Biomass carbon accumulation rate or related quality-control variable used for filtering. |
| `ECO_ZONE` | Optional ecozone field. If present, it is renamed internally as `eco_region`. |

## Analysis logic

1. **Data import and harmonization**  
   The input file is read from a user-specified path. The column `biomass_carbon` is renamed internally to `biomass_ca`, and `ECO_ZONE` is renamed to `eco_region` if present.

2. **Initial filtering using `biorate`**  
   Robust scaling and Isolation Forest are applied to the `biorate` column. Records identified as outliers are removed, and records with `biorate < 0.01` are excluded.

3. **Forest-type stratification**  
   Data are grouped by `TYPE`, and each forest type is modelled independently.

4. **Age-carbon outlier removal**  
   Within each forest type, robust scaling and Isolation Forest are applied jointly to stand age and biomass carbon density. Non-positive or non-finite ages and carbon values are removed. For DNF, stand ages below 1 year are removed.

5. **Age-specific filtering**  
   Two additional biologically motivated filters are applied:  
   - For stands older than 150 years, very low biomass carbon values below the 5th percentile of that old-stand subset are removed.  
   - For stands younger than 20 years, very high biomass carbon values above the 95th percentile of that young-stand subset are removed.

6. **Age-bin quantile calibration**  
   Stand ages are grouped into 5-year bins. The 95th percentile of biomass carbon density is calculated within each bin and used as the calibration target for curve fitting. This upper-quantile approach is intended to approximate the potential carbon accumulation envelope while reducing the influence of local degradation, disturbance, or site limitations.

7. **Chapman-Richards model fitting**  
   The model is:

   ```text
   C(age) = A * (1 - exp(-k * age))^c
   ```

   where `C(age)` is biomass carbon density, `A` is the asymptotic carbon density, `k` is the growth-rate coefficient, and `c` controls curve shape.

   `A` is estimated from high quantiles of the cleaned raw observations. If more than 10% of observations are older than 100 years, the 95th percentile is used; otherwise, the 90th percentile is used. Parameters `k` and `c` are fitted in two steps: global differential evolution followed by local bounded nonlinear least squares.

8. **Carbon accumulation rate**  
   The first derivative of the Chapman-Richards curve is used to quantify age-dependent carbon accumulation rate:

   ```text
   dC/dage = A * c * k * (1 - exp(-k * age))^(c - 1) * exp(-k * age)
   ```

   The peak derivative is interpreted as the maximum fitted biomass carbon accumulation rate.

9. **Uncertainty estimation**  
   Standard errors for `k` and `c` are extracted from the covariance matrix returned by nonlinear least squares. The standard error of `A` is estimated using non-parametric bootstrap resampling. Approximate 95% confidence intervals for the fitted curve and first derivative are calculated by the delta method.

10. **Threshold age**  
    The threshold age is calculated as the age at which the fitted curve reaches a specified fraction of `A`. The default threshold is 95% of `A`; for DNF, the threshold is set to 80% following the original script.

11. **Model evaluation and export**  
    The script calculates R^2, RMSE, and MAE using the age-bin quantile calibration data. Outputs include fitted parameters, calibration data, and a combined figure showing fitted carbon density curves and derivatives.

## How to run

Install dependencies:

```bash
pip install -r requirements_forest_carbon.txt
```

Run the analysis:

```bash
python forest_carbon_growth_analysis.py \
  --input biomass_fi_boreal.xlsx \
  --output-dir forest_carbon_results \
  --bootstrap-iterations 100 \
  --random-state 42
```

For an Excel file with a named sheet:

```bash
python forest_carbon_growth_analysis.py \
  --input biomass_fi_boreal.xlsx \
  --sheet-name Sheet1 \
  --output-dir forest_carbon_results
```

## Outputs

| Output file | Description |
|---|---|
| `fit_results.csv` | Fitted parameters (`A`, `k`, `c`), standard errors, threshold age, peak growth rate, sample size, R^2, RMSE, and MAE. |
| `corrected_quantile_data.csv` | Forest-type-specific 5-year age-bin 95th quantile data used for model fitting. |
| `combined_fit_with_derivative.jpeg` | Combined figure showing fitted Chapman-Richards curves, 95% confidence intervals, and first derivatives. |

## Reproducibility notes

- The original script contained a local Windows data path. The refactored version uses a command-line `--input` argument.
- Random-state controls were added for Isolation Forest, differential evolution, and bootstrap sampling.
- Results may still vary slightly across Python/scientific-library versions because nonlinear optimization and floating-point routines are implementation-dependent.
- The workflow assumes that `TYPE` categories are biologically meaningful strata for separate curve fitting.
- The DNF-specific handling from the original script is retained, including removal of ages below 1 year and capping plotted growth rate at 2 Mg C ha^-1 yr^-1.

## Recommended citation in manuscript methods

> Biomass carbon accumulation was modelled separately for each forest type using a Chapman-Richards growth function. Prior to fitting, anomalous records were removed using robust scaling and Isolation Forest, followed by age-specific filters for implausible young-stand and old-stand carbon values. Five-year age-bin 95th quantiles of biomass carbon density were used as calibration targets. The asymptotic carbon density was estimated from high quantiles of cleaned observations, whereas the growth-rate and shape parameters were optimized using differential evolution followed by bounded nonlinear least squares. Age-dependent carbon accumulation rate was derived analytically as the first derivative of the fitted Chapman-Richards curve.
