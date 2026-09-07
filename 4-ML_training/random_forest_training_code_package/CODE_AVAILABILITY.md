## Code availability

The machine-learning workflow used to train and compare forest aboveground
carbon-density models is provided with the article as supplementary code. The
workflow was implemented in Python and includes data preprocessing, modified
Z-score-based multivariate outlier screening, predictor selection, comparison of
Gradient Boosting, Extra Trees, Random Forest and XGBoost regressors, bootstrap
estimation of predictive uncertainty, and SHAP-based model interpretation.
Input paths, predictor lists, random seeds, bootstrap settings and output
locations are configurable from the command line. The archived package also
contains an environment requirements file and documentation describing the
expected input structure and generated outputs. The observational training data
and environmental raster products are distributed separately according to their
respective data-use conditions.
