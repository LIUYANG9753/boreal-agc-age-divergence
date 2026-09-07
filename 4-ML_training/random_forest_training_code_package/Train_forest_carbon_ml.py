#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate machine-learning models for forest aboveground carbon density.

This reproducible workflow:
1. Reads a tabular training dataset.
2. Imputes missing numeric values with the median.
3. Optionally removes multivariate outliers using a modified-Z-score rule.
4. Compares four regression algorithms and four feature-selection methods.
5. Selects the best overall model and the best model using exactly k features.
6. Quantifies predictive uncertainty using bootstrap refitting.
7. Produces SHAP-based feature-importance outputs for tree-based models.

The script is adapted from the original research workflow while replacing
hard-coded paths and dispersed settings with command-line arguments.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DEFAULT_FEATURES = [
        "b1_clim", "b2_clim", "b3_clim", "b4_clim",
        "b5_clim", "b6_clim", "b7_clim", "b8_clim",
        "b9_clim", "b10_clim", "b11_clim", "b12_clim", "b13_clim", "b14_clim", "b15_clim",
        "b16_clim", "b17_clim", "b18_clim", "b19_clim", "b20_clim", "b21_clim", "b22_clim",
        "b23_clim", "b2_veg", "b3_veg", "b4_veg", "b1_veg", "b2_soil", "b3_soil", "b4_soil",
        "b5_soil", "b6_soil", "b1_topo", "b2_topo", "b3_topo", "b4_topo", "b1_Human",
        "b2_Human", "NPP", "Age", "leaftype", "arid", "fire_frequency", "CSC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, compare, interpret, and bootstrap forest-carbon ML models."
    )
    parser.add_argument("--input", required=True, help="Input training CSV.")
    parser.add_argument("--target", required=True, help="Target-column name.")
    parser.add_argument(
        "--features-file",
        default=None,
        help="Optional text/JSON file containing feature names. "
             "When omitted, the study default feature pool is used.",
    )
    parser.add_argument("--output-dir", default="ml_training_results")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--selected-k", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.70)
    parser.add_argument("--z-threshold", type=float, default=5.0)
    parser.add_argument(
        "--valid-feature-fraction",
        type=float,
        default=0.93,
        help="Minimum fraction of numeric variables whose modified Z score must "
             "be below --z-threshold. This is an outlier-filter setting, not an "
             "extrapolation metric.",
    )
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="Do not remove potential multivariate outliers; only report them.",
    )
    parser.add_argument(
        "--skip-distribution-plots",
        action="store_true",
        help="Skip target and predictor histograms.",
    )
    parser.add_argument("--shap-sample-size", type=int, default=100)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("forest_carbon_ml")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_feature_names(path: Optional[str]) -> List[str]:
    if path is None:
        return DEFAULT_FEATURES.copy()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature file not found: {p}")

    if p.suffix.lower() == ".json":
        value = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("features", [])
        if not isinstance(value, list):
            raise ValueError("JSON feature file must contain a list or {'features': [...]}.")
        return [str(x) for x in value]

    return [
        line.strip() for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def modified_z_scores(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    mad = np.where(mad == 0, 1e-10, mad)
    return np.abs(0.6745 * (values - median) / mad)


def plot_histograms(
    data: pd.DataFrame,
    columns: List[str],
    output_dir: Path,
    batch_size: int = 10,
) -> None:
    for start in range(0, len(columns), batch_size):
        batch = columns[start:start + batch_size]
        fig, axes = plt.subplots(len(batch), 1, figsize=(10, max(4, 3 * len(batch))))
        axes = np.atleast_1d(axes)
        for ax, column in zip(axes, batch):
            ax.hist(data[column].dropna().to_numpy(), bins=40)
            ax.set_title(f"{column} distribution")
            ax.set_ylabel("Frequency")
        axes[-1].set_xlabel("Value")
        fig.tight_layout()
        fig.savefig(output_dir / f"predictor_histograms_{start // batch_size + 1:02d}.png", dpi=300)
        plt.close(fig)


def load_and_preprocess_data(
    input_path: str,
    feature_pool: List[str],
    target_column: str,
    output_dir: Path,
    logger: logging.Logger,
    z_threshold: float,
    valid_feature_fraction: float,
    remove_outliers: bool,
    make_plots: bool,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data = pd.read_csv(input_path, encoding="utf-8", engine="python", on_bad_lines="warn")
    logger.info("Loaded dataset with %d rows and %d columns.", *data.shape)

    required = feature_pool + [target_column]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.dropna(subset=[target_column]).copy()
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

    imputer = SimpleImputer(strategy="median")
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

    z_scores = modified_z_scores(data[numeric_columns].to_numpy(dtype=float))
    valid_fraction = (z_scores < z_threshold).sum(axis=1) / z_scores.shape[1]
    keep_mask = valid_fraction >= valid_feature_fraction

    outlier_report = pd.DataFrame({
        "row_index": data.index,
        "valid_feature_fraction": valid_fraction,
        "potential_outlier": ~keep_mask,
    })
    outlier_report.to_csv(output_dir / "multivariate_outlier_report.csv", index=False)

    if remove_outliers:
        data = data.loc[keep_mask].copy()
        logger.info(
            "Retained %d samples after modified-Z filtering "
            "(z_threshold=%.3f; valid_feature_fraction=%.3f).",
            len(data), z_threshold, valid_feature_fraction,
        )
    else:
        logger.info("Potential outliers were reported but retained.")

    if make_plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data[target_column].to_numpy(), bins=40)
        ax.set_xlabel(target_column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{target_column} distribution")
        fig.tight_layout()
        fig.savefig(output_dir / "target_distribution.png", dpi=300)
        plt.close(fig)
        plot_histograms(data, feature_pool, output_dir)

    return data[feature_pool].copy(), data[target_column].copy(), outlier_report


def build_model_pipelines(random_seed: int) -> Dict[str, Pipeline]:
    preprocessing = Pipeline([("robust_scaler", RobustScaler())])
    return {
        "GradientBoosting": Pipeline([
            ("preprocessing", clone(preprocessing)),
            ("regressor", GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=10,
                random_state=random_seed,
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("preprocessing", clone(preprocessing)),
            ("regressor", ExtraTreesRegressor(
                n_estimators=300, max_depth=10,
                random_state=random_seed, n_jobs=-1,
            )),
        ]),
        "RandomForest": Pipeline([
            ("preprocessing", clone(preprocessing)),
            ("regressor", RandomForestRegressor(
                n_estimators=300, max_depth=10,
                random_state=random_seed, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessing", clone(preprocessing)),
            ("regressor", XGBRegressor(
                n_estimators=300, learning_rate=0.10, max_depth=10,
                random_state=random_seed, n_jobs=-1,
                objective="reg:squarederror",
            )),
        ]),
    }


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    estimator,
    k: int,
) -> List[str]:
    k_eff = min(k, X_train.shape[1])

    if method == "VarianceThreshold":
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X_train)
        selected = X_train.columns[selector.get_support()].tolist()
        if len(selected) > k_eff:
            selected = selected[:k_eff]
    elif method == "SelectKBest":
        selector = SelectKBest(score_func=f_regression, k=k_eff)
        selector.fit(X_train, y_train)
        selected = X_train.columns[selector.get_support()].tolist()
    elif method == "RFE":
        selector = RFE(clone(estimator), n_features_to_select=k_eff)
        selector.fit(X_train, y_train)
        selected = X_train.columns[selector.get_support()].tolist()
    elif method == "SelectFromModel":
        selector = SelectFromModel(clone(estimator), max_features=k_eff)
        selector.fit(X_train, y_train)
        selected = X_train.columns[selector.get_support()].tolist()
        if len(selected) > k_eff:
            selected = selected[:k_eff]
    else:
        raise ValueError(f"Unknown feature-selection method: {method}")

    if not selected:
        raise RuntimeError(f"{method} selected no predictors.")
    return selected


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Pipeline],
    output_dir: Path,
    logger: logging.Logger,
    selected_k: int,
    test_size: float,
    random_seed: int,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    methods = ["VarianceThreshold", "SelectKBest", "RFE", "SelectFromModel"]
    rows = []

    for model_name, pipeline in models.items():
        for method in methods:
            estimator = pipeline.named_steps["regressor"]
            try:
                selected = select_features(
                    X_train, y_train, method, estimator, selected_k
                )
                fitted = clone(pipeline)
                fitted.fit(X_train[selected], y_train)
                pred = fitted.predict(X_test[selected])
                rows.append({
                    "model": model_name,
                    "feature_selection": method,
                    "n_features": len(selected),
                    "r2": r2_score(y_test, pred),
                    "rmse": mean_squared_error(y_test, pred) ** 0.5,
                    "selected_features": "|".join(selected),
                })
                logger.info(
                    "%s + %s: R2=%.4f, RMSE=%.4f, n_features=%d",
                    model_name, method, rows[-1]["r2"], rows[-1]["rmse"], len(selected),
                )
            except Exception as exc:
                logger.exception("%s + %s failed: %s", model_name, method, exc)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("All model-comparison runs failed.")

    results.to_csv(output_dir / "model_comparison_results.csv", index=False)

    best = results.loc[results["r2"].idxmax()]
    exact_k = results[results["n_features"] == selected_k]
    best_k = exact_k.loc[exact_k["r2"].idxmax()] if not exact_k.empty else None

    return best, best_k, X_train, X_test, y_train, y_test


def reconstruct_pipeline(
    result_row: pd.Series,
    models: Dict[str, Pipeline],
) -> Tuple[Pipeline, List[str]]:
    pipeline = clone(models[str(result_row["model"])])
    features = str(result_row["selected_features"]).split("|")
    return pipeline, features


def bootstrap_uncertainty(
    label: str,
    pipeline: Pipeline,
    features: List[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    logger: logging.Logger,
    n_bootstrap: int,
    bootstrap_fraction: float,
    random_seed: int,
    shap_sample_size: int,
) -> None:
    model_dir = output_dir / f"bootstrap_models_{label}"
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    predictions = []

    for i in range(n_bootstrap):
        sample_n = max(2, int(round(bootstrap_fraction * len(X_train))))
        idx = rng.choice(len(X_train), size=sample_n, replace=True)
        model = clone(pipeline)
        model.fit(X_train.iloc[idx][features], y_train.iloc[idx])
        predictions.append(model.predict(X_test[features]))
        joblib.dump(model, model_dir / f"bootstrap_model_{i:03d}.pkl")

    pred_array = np.asarray(predictions)
    pred_mean = pred_array.mean(axis=0)
    pred_std = pred_array.std(axis=0, ddof=1 if n_bootstrap > 1 else 0)
    pred_low = np.percentile(pred_array, 2.5, axis=0)
    pred_high = np.percentile(pred_array, 97.5, axis=0)

    out = pd.DataFrame({
        "observed": y_test.to_numpy(),
        "predicted_mean": pred_mean,
        "predicted_sd": pred_std,
        "prediction_p2.5": pred_low,
        "prediction_p97.5": pred_high,
    })
    out.to_csv(output_dir / f"predictions_with_uncertainty_{label}.csv", index=False)

    logger.info(
        "%s bootstrap ensemble: R2=%.4f, RMSE=%.4f",
        label,
        r2_score(y_test, pred_mean),
        mean_squared_error(y_test, pred_mean) ** 0.5,
    )

    final_model = clone(pipeline)
    final_model.fit(X_train[features], y_train)
    joblib.dump(final_model, output_dir / f"{label}_model.pkl")
    (output_dir / f"{label}_features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )

    regressor = final_model.named_steps["regressor"]
    preprocessor = final_model.named_steps["preprocessing"]
    sample = X_test[features].iloc[:min(shap_sample_size, len(X_test))]
    transformed = preprocessor.transform(sample)

    try:
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(transformed)
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[0]

        importance = pd.DataFrame({
            "feature": features,
            "mean_absolute_shap": np.abs(values).mean(axis=0),
        }).sort_values("mean_absolute_shap", ascending=False)
        importance.to_csv(output_dir / f"shap_importance_{label}.csv", index=False)

        shap.summary_plot(values, sample, feature_names=features, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{label}.png", dpi=300, bbox_inches="tight")
        plt.close()

        shap.summary_plot(
            values, sample, feature_names=features, plot_type="bar", show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_bar_{label}.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning("SHAP analysis for %s was skipped: %s", label, exc)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = configure_logging(output_dir)

    features = load_feature_names(args.features_file)
    (output_dir / "feature_pool.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )
    (output_dir / "run_configuration.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    X, y, _ = load_and_preprocess_data(
        input_path=args.input,
        feature_pool=features,
        target_column=args.target,
        output_dir=output_dir,
        logger=logger,
        z_threshold=args.z_threshold,
        valid_feature_fraction=args.valid_feature_fraction,
        remove_outliers=not args.keep_outliers,
        make_plots=not args.skip_distribution_plots,
    )

    models = build_model_pipelines(args.random_seed)
    best, best_k, X_train, X_test, y_train, y_test = compare_models(
        X=X,
        y=y,
        models=models,
        output_dir=output_dir,
        logger=logger,
        selected_k=args.selected_k,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    best_pipeline, best_features = reconstruct_pipeline(best, models)
    bootstrap_uncertainty(
        label="best",
        pipeline=best_pipeline,
        features=best_features,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=output_dir,
        logger=logger,
        n_bootstrap=args.n_bootstrap,
        bootstrap_fraction=args.bootstrap_fraction,
        random_seed=args.random_seed,
        shap_sample_size=args.shap_sample_size,
    )

    if best_k is not None:
        k_pipeline, k_features = reconstruct_pipeline(best_k, models)
        bootstrap_uncertainty(
            label=f"best_{args.selected_k}_features",
            pipeline=k_pipeline,
            features=k_features,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            output_dir=output_dir,
            logger=logger,
            n_bootstrap=args.n_bootstrap,
            bootstrap_fraction=args.bootstrap_fraction,
            random_seed=args.random_seed + 1,
            shap_sample_size=args.shap_sample_size,
        )
    else:
        logger.warning("No model selected exactly %d predictors.", args.selected_k)

    logger.info("Analysis completed successfully.")


if __name__ == "__main__":
    main()
