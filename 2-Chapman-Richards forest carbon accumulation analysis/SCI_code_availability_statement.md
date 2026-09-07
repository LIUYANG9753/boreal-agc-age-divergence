# Code availability statement

The Python code used to fit forest-type-specific biomass carbon accumulation curves and estimate age-dependent carbon accumulation rates is provided as supplementary material. The workflow implements data cleaning, upper-quantile age-bin calibration, Chapman-Richards model fitting, uncertainty estimation, derivative-based growth-rate calculation, and figure/table export.

The analysis was conducted using Python 3 with NumPy, pandas, SciPy, scikit-learn, Matplotlib, and openpyxl. The refactored script `forest_carbon_growth_analysis.py` can be executed from the command line after installing the dependencies listed in `requirements_forest_carbon.txt`. The input dataset should contain forest type, stand age, biomass carbon density, and biorate fields. Because the empirical input dataset may be subject to data-sharing restrictions, users should replace the input path with the corresponding local dataset when reproducing the analysis.

Example command:

```bash
python forest_carbon_growth_analysis.py \
  --input biomass_fi_boreal.xlsx \
  --output-dir forest_carbon_results \
  --bootstrap-iterations 100 \
  --random-state 42
```

The script exports fitted model parameters and diagnostics to `fit_results.csv`, the 5-year age-bin 95th quantile calibration data to `corrected_quantile_data.csv`, and the combined fitted-curve/derivative figure to `combined_fit_with_derivative.jpeg`. The original local file path in the working script was replaced by command-line input arguments to improve portability and reproducibility.
