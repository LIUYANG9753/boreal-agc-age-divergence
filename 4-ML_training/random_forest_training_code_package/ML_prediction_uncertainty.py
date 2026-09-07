"""Generate full-domain ML predictions and bootstrap uncertainty rasters.

This publication-ready script applies an ensemble of bootstrap-trained machine-
learning models to spatial predictor rasters over a boreal-forest mask. It
writes four GeoTIFF products:

1. ensemble mean AGCD prediction;
2. bootstrap standard deviation as prediction uncertainty;
3. empirical 2.5th percentile of bootstrap predictions; and
4. empirical 97.5th percentile of bootstrap predictions.

The script intentionally does not calculate an extrapolation-area proportion.
Predictions are generated only where all required predictor rasters contain
valid data and the pixel falls within the supplied spatial mask.

For reproducibility, all predictor rasters must be aligned to the reference
raster (same CRS, transform, width, and height). Output rasters are written
block by block to avoid holding four full-domain arrays in memory.
"""

from __future__ import annotations

import gc
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio import features
from rasterio.io import DatasetReader
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:  # pragma: no cover - useful in interactive sessions
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = Path(os.environ.get("CRML_PROJECT_ROOT", SCRIPT_DIR)).resolve()
DATA_DIR = Path(os.environ.get("CRML_DATA_DIR", PROJECT_ROOT / "data")).resolve()
OUTPUT_ROOT = Path(
    os.environ.get("CRML_OUTPUT_DIR", PROJECT_ROOT / "outputs")
).resolve()

MODEL_DIR = Path(
    os.environ.get(
        "CRML_MODEL_DIR",
        DATA_DIR / "models" / "bootstrap_models_15_features",
    )
).resolve()
PREDICTOR_DIR = DATA_DIR / "predictors"
MASK_VECTOR = DATA_DIR / "masks" / "boreal_forest_region_fixed.shp"
OUTPUT_DIR = OUTPUT_ROOT / "full_domain_prediction"

OUTPUT_PATHS = {
    "prediction": OUTPUT_DIR / "CCI_AGC_prediction.tif",
    "uncertainty": OUTPUT_DIR / "CCI_AGC_uncertainty.tif",
    "lower_95": OUTPUT_DIR / "CCI_AGC_lower_bound.tif",
    "upper_95": OUTPUT_DIR / "CCI_AGC_upper_bound.tif",
}

# Predictor names must match the feature names used during model training.
INPUT_RASTERS = {
    "b2_soil": PREDICTOR_DIR / "resampled_soil_sand_cleaned.tif",
    "b3_soil": PREDICTOR_DIR / "resampled_soil_clay_cleaned.tif",
    "arid": PREDICTOR_DIR / "Boreal_arid_index.tif",
    "b6_soil": PREDICTOR_DIR / "resampled_soil_ocd.tif",
    "b5_clim": PREDICTOR_DIR / "CHELSA_bio5_1981-2010_V.2.1.tif",
    "b7_clim": PREDICTOR_DIR / "CHELSA_bio7_1981-2010_V.2.1.tif",
    "b8_clim": PREDICTOR_DIR / "CHELSA_bio8_1981-2010_V.2.1.tif",
    "b9_clim": PREDICTOR_DIR / "CHELSA_bio9_1981-2010_V.2.1.tif",
    "b10_clim": PREDICTOR_DIR / "CHELSA_bio10_1981-2010_V.2.1.tif",
    "b13_clim": PREDICTOR_DIR / "CHELSA_bio13_1981-2010_V.2.1.tif",
    "b15_clim": PREDICTOR_DIR / "CHELSA_bio15_1981-2010_V.2.1.tif",
    "b16_clim": PREDICTOR_DIR / "CHELSA_bio16_1981-2010_V.2.1.tif",
    "b21_clim": PREDICTOR_DIR / "CHELSA_scd_1981-2010_V.2.1_cleaned.tif",
    "b23_clim": PREDICTOR_DIR / "CHELSA_vpd_mean_1981-2010_V.2.1_cleaned.tif",
    "Age": PREDICTOR_DIR / "Mergeb12_age_output.tif",
}

DEFAULT_FEATURE_ORDER = [
    "b21_clim",
    "b2_soil",
    "Age",
    "b5_clim",
    "b15_clim",
    "b3_soil",
    "b9_clim",
    "b8_clim",
    "b13_clim",
    "b6_soil",
    "b7_clim",
    "b16_clim",
    "b23_clim",
    "b10_clim",
    "arid",
]

REFERENCE_RASTER = "Age"
N_BOOTSTRAP = 20
BLOCK_ROWS = 256
MAX_READ_WORKERS = 2
OUTPUT_NODATA = -9999.0
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5
LOW_MEMORY_WARNING_GB = 4.0

# GDAL cache size in MB. Users can override this before launching the script.
os.environ.setdefault("GDAL_CACHEMAX", "1024")


# -----------------------------------------------------------------------------
# Validation and I/O helpers
# -----------------------------------------------------------------------------


def require_file(path: Path, label: str) -> None:
    """Raise a clear error when a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_configuration(feature_order: Sequence[str]) -> None:
    """Validate model, predictor, mask, and feature configuration."""
    require_file(MASK_VECTOR, "Spatial mask")

    if REFERENCE_RASTER not in INPUT_RASTERS:
        raise KeyError(
            f"Reference raster '{REFERENCE_RASTER}' is not defined in INPUT_RASTERS."
        )

    missing_features = [name for name in feature_order if name not in INPUT_RASTERS]
    if missing_features:
        raise KeyError(
            "No raster path is configured for required model features: "
            f"{missing_features}"
        )

    for feature_name in feature_order:
        require_file(INPUT_RASTERS[feature_name], f"Predictor raster '{feature_name}'")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Bootstrap model directory not found: {MODEL_DIR}")


def load_bootstrap_models() -> tuple[list[Any], list[str]]:
    """Load bootstrap models and determine the feature order used for prediction."""
    models: list[Any] = []

    for index in range(N_BOOTSTRAP):
        model_path = MODEL_DIR / f"bootstrap_model_{index}.pkl"
        require_file(model_path, f"Bootstrap model {index}")
        model = joblib.load(model_path)
        models.append(model)
        LOGGER.info(
            "Loaded bootstrap model %d/%d: %s",
            index + 1,
            N_BOOTSTRAP,
            model_path,
        )

    if not models:
        raise RuntimeError("No bootstrap models were loaded.")

    first_features = getattr(models[0], "feature_names_in_", None)
    if first_features is None:
        feature_order = list(DEFAULT_FEATURE_ORDER)
        LOGGER.info(
            "The models do not expose feature_names_in_; using DEFAULT_FEATURE_ORDER."
        )
    else:
        feature_order = [str(name) for name in first_features]
        if feature_order != DEFAULT_FEATURE_ORDER:
            LOGGER.warning(
                "Model feature order differs from DEFAULT_FEATURE_ORDER. "
                "The model-defined order will be used."
            )
        else:
            LOGGER.info("Model feature order matches DEFAULT_FEATURE_ORDER.")

    for index, model in enumerate(models[1:], start=1):
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            continue
        model_feature_order = [str(name) for name in model_features]
        if model_feature_order != feature_order:
            raise ValueError(
                "Bootstrap models do not use a consistent feature order. "
                f"Model 0: {feature_order}; model {index}: {model_feature_order}"
            )

    validate_configuration(feature_order)
    LOGGER.info("Loaded %d bootstrap models successfully.", len(models))
    return models, feature_order


def validate_raster_alignment(
    feature_order: Sequence[str],
    reference: DatasetReader,
) -> None:
    """Require every predictor raster to match the reference grid exactly."""
    alignment_errors: list[str] = []

    for feature_name in feature_order:
        path = INPUT_RASTERS[feature_name]
        with rasterio.open(path) as src:
            problems: list[str] = []

            if src.crs != reference.crs:
                problems.append(f"CRS {src.crs!s} != {reference.crs!s}")
            if src.width != reference.width or src.height != reference.height:
                problems.append(
                    f"shape {src.width}x{src.height} != "
                    f"{reference.width}x{reference.height}"
                )
            if not src.transform.almost_equals(reference.transform):
                problems.append("affine transform differs")

            if problems:
                alignment_errors.append(
                    f"{feature_name} ({path}): " + "; ".join(problems)
                )

    if alignment_errors:
        details = "\n".join(f"- {item}" for item in alignment_errors)
        raise ValueError(
            "Predictor rasters are not aligned to the reference raster. "
            "Resample/reproject them before prediction:\n"
            f"{details}"
        )


def read_raster_block(path: Path, window: Window) -> tuple[np.ndarray, np.ndarray]:
    """Read one raster window and return values plus a finite-data validity mask."""
    with rasterio.open(path) as src:
        masked = src.read(1, window=window, masked=True)

    masked_float = masked.astype(np.float32)
    values = np.asarray(masked_float.filled(np.nan), dtype=np.float32)
    valid = ~np.ma.getmaskarray(masked)
    valid &= np.isfinite(values)
    return values, valid


def load_mask_geometries(target_crs: Any) -> list[Any]:
    """Read the spatial mask and transform its geometries to the raster CRS."""
    mask_gdf = gpd.read_file(MASK_VECTOR)
    if mask_gdf.empty:
        raise ValueError(f"Spatial mask contains no features: {MASK_VECTOR}")
    if mask_gdf.crs is None:
        raise ValueError(f"Spatial mask has no defined CRS: {MASK_VECTOR}")
    if target_crs is None:
        raise ValueError("Reference raster has no defined CRS.")

    mask_gdf = mask_gdf.to_crs(target_crs)
    geometries = [geom for geom in mask_gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        raise ValueError(f"Spatial mask contains no valid geometries: {MASK_VECTOR}")

    LOGGER.info("Loaded %d mask geometries.", len(geometries))
    return geometries


def make_spatial_mask(
    geometries: Sequence[Any],
    window: Window,
    reference_transform: Any,
) -> np.ndarray:
    """Rasterize the spatial mask for one processing window."""
    block_transform = window_transform(window, reference_transform)
    return features.geometry_mask(
        geometries=geometries,
        out_shape=(int(window.height), int(window.width)),
        transform=block_transform,
        invert=True,
        all_touched=True,
    )


# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------


def predict_bootstrap_ensemble(
    models: Sequence[Any],
    predictors: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ensemble mean, SD, and empirical 95% bootstrap bounds."""
    predictions = np.vstack([model.predict(predictors) for model in models])

    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    lower_bound = np.percentile(predictions, LOWER_PERCENTILE, axis=0)
    upper_bound = np.percentile(predictions, UPPER_PERCENTILE, axis=0)

    # AGCD and its non-negative uncertainty summaries cannot be below zero.
    mean_prediction = np.clip(mean_prediction, 0.0, None)
    uncertainty = np.clip(uncertainty, 0.0, None)
    lower_bound = np.clip(lower_bound, 0.0, None)
    upper_bound = np.clip(upper_bound, 0.0, None)

    return mean_prediction, uncertainty, lower_bound, upper_bound


def initialize_output_block(height: int, width: int) -> np.ndarray:
    """Create a float32 output block initialized to the output NoData value."""
    return np.full((height, width), OUTPUT_NODATA, dtype=np.float32)


def log_block_ranges(
    row_start: int,
    row_stop: int,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> None:
    """Report output ranges for valid pixels in the current block."""
    LOGGER.info(
        "Rows %d-%d complete | prediction %.2f-%.2f | uncertainty %.2f-%.2f | "
        "95%% lower %.2f-%.2f | 95%% upper %.2f-%.2f",
        row_start,
        row_stop,
        float(np.min(prediction)),
        float(np.max(prediction)),
        float(np.min(uncertainty)),
        float(np.max(uncertainty)),
        float(np.min(lower_bound)),
        float(np.max(lower_bound)),
        float(np.min(upper_bound)),
        float(np.max(upper_bound)),
    )


def run_full_domain_prediction() -> None:
    """Run block-wise full-domain prediction and write all output GeoTIFFs."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    LOGGER.info("Available system memory: %.2f GB", available_gb)
    if available_gb < LOW_MEMORY_WARNING_GB:
        LOGGER.warning(
            "Available memory is below %.1f GB; consider reducing BLOCK_ROWS.",
            LOW_MEMORY_WARNING_GB,
        )

    models, feature_order = load_bootstrap_models()
    reference_path = INPUT_RASTERS[REFERENCE_RASTER]
    require_file(reference_path, "Reference raster")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as reference:
        validate_raster_alignment(feature_order, reference)

        target_width = reference.width
        target_height = reference.height
        target_transform = reference.transform
        target_crs = reference.crs
        output_profile = reference.profile.copy()
        output_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=OUTPUT_NODATA,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )

        LOGGER.info(
            "Reference grid: %dx%d pixels; CRS=%s",
            target_width,
            target_height,
            target_crs,
        )

        geometries = load_mask_geometries(target_crs)

        with ExitStack() as stack:
            output_datasets = {
                key: stack.enter_context(rasterio.open(path, "w", **output_profile))
                for key, path in OUTPUT_PATHS.items()
            }

            for row_offset in range(0, target_height, BLOCK_ROWS):
                block_height = min(BLOCK_ROWS, target_height - row_offset)
                row_stop = row_offset + block_height
                window = Window(0, row_offset, target_width, block_height)

                spatial_mask = make_spatial_mask(
                    geometries,
                    window,
                    target_transform,
                )

                output_blocks = {
                    key: initialize_output_block(block_height, target_width)
                    for key in OUTPUT_PATHS
                }

                if not np.any(spatial_mask):
                    for key, dataset in output_datasets.items():
                        dataset.write(output_blocks[key], 1, window=window)
                    LOGGER.info(
                        "Rows %d-%d skipped: no pixels inside the spatial mask.",
                        row_offset,
                        row_stop,
                    )
                    continue

                with ThreadPoolExecutor(max_workers=MAX_READ_WORKERS) as executor:
                    futures = {
                        feature_name: executor.submit(
                            read_raster_block,
                            INPUT_RASTERS[feature_name],
                            window,
                        )
                        for feature_name in feature_order
                    }
                    raster_blocks = {
                        feature_name: future.result()
                        for feature_name, future in futures.items()
                    }

                valid_prediction_mask = spatial_mask.copy()
                for feature_name in feature_order:
                    _, feature_valid = raster_blocks[feature_name]
                    valid_prediction_mask &= feature_valid

                n_spatial = int(np.count_nonzero(spatial_mask))
                n_valid = int(np.count_nonzero(valid_prediction_mask))

                if n_valid == 0:
                    for key, dataset in output_datasets.items():
                        dataset.write(output_blocks[key], 1, window=window)
                    LOGGER.warning(
                        "Rows %d-%d contain %d masked pixels but no pixels with all "
                        "required predictors valid.",
                        row_offset,
                        row_stop,
                        n_spatial,
                    )
                    continue

                predictor_matrix = np.column_stack(
                    [
                        raster_blocks[feature_name][0][valid_prediction_mask]
                        for feature_name in feature_order
                    ]
                ).astype(np.float32, copy=False)
                predictor_frame = pd.DataFrame(
                    predictor_matrix,
                    columns=feature_order,
                )

                (
                    mean_prediction,
                    uncertainty,
                    lower_bound,
                    upper_bound,
                ) = predict_bootstrap_ensemble(models, predictor_frame)

                output_blocks["prediction"][valid_prediction_mask] = mean_prediction.astype(
                    np.float32,
                    copy=False,
                )
                output_blocks["uncertainty"][valid_prediction_mask] = uncertainty.astype(
                    np.float32,
                    copy=False,
                )
                output_blocks["lower_95"][valid_prediction_mask] = lower_bound.astype(
                    np.float32,
                    copy=False,
                )
                output_blocks["upper_95"][valid_prediction_mask] = upper_bound.astype(
                    np.float32,
                    copy=False,
                )

                for key, dataset in output_datasets.items():
                    dataset.write(output_blocks[key], 1, window=window)

                LOGGER.info(
                    "Rows %d-%d: predicted %d/%d pixels inside the spatial mask.",
                    row_offset,
                    row_stop,
                    n_valid,
                    n_spatial,
                )
                log_block_ranges(
                    row_offset,
                    row_stop,
                    mean_prediction,
                    uncertainty,
                    lower_bound,
                    upper_bound,
                )

                del (
                    predictor_matrix,
                    predictor_frame,
                    raster_blocks,
                    output_blocks,
                    mean_prediction,
                    uncertainty,
                    lower_bound,
                    upper_bound,
                )
                gc.collect()

    for key, path in OUTPUT_PATHS.items():
        LOGGER.info("Saved %s raster: %s", key, path)


def main() -> None:
    """Command-line entry point."""
    try:
        run_full_domain_prediction()
    except Exception:
        LOGGER.exception("Full-domain prediction failed.")
        raise


if __name__ == "__main__":
    main()
