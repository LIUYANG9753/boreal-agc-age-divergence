#!/usr/bin/env python3
"""
Map Chapman-Richards aboveground carbon lookup tables to spatial raster grids.

This script maps region-specific age-carbon lookup tables (LUTs) to spatial pixels
using an ecoregion raster and a stand-age raster. For each valid pixel, the script
identifies its ecoregion, extracts stand age, clips the age to the valid LUT range,
and assigns both the expected aboveground carbon density and the corresponding
LUT-based uncertainty.

Expected LUT columns:
    Age, AGC_LUT, AGC_LUT_std

Default ecoregion code mapping:
    2 -> BD
    3 -> NE
    4 -> ND
    5 -> Mixed

Outputs:
    1. A GeoTIFF raster of AGC_LUT values
    2. A GeoTIFF raster of AGC_LUT_std values
    3. A CSV summary table of mapped pixel counts and value ranges by ecoregion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject


DEFAULT_ECOREGION_MAP = {2: "BD", 3: "NE", 4: "ND", 5: "Mixed"}
REQUIRED_LUT_COLUMNS = {"Age", "AGC_LUT", "AGC_LUT_std"}


def parse_ecoregion_map(mapping: str | None) -> Dict[int, str]:
    """Parse an optional JSON ecoregion mapping string or file path."""
    if mapping is None:
        return DEFAULT_ECOREGION_MAP.copy()

    mapping_path = Path(mapping)
    if mapping_path.exists():
        text = mapping_path.read_text(encoding="utf-8")
    else:
        text = mapping

    parsed = json.loads(text)
    return {int(k): str(v) for k, v in parsed.items()}


def read_single_band_raster(path: Path) -> Tuple[np.ndarray, dict]:
    """Read the first band of a raster and return the array and metadata."""
    with rasterio.open(path) as src:
        array = src.read(1)
        meta = src.meta.copy()
    return array, meta


def read_age_raster_matched(age_path: Path, reference_meta: dict, align: bool) -> np.ndarray:
    """Read an age raster and optionally reproject it to the ecoregion raster grid."""
    with rasterio.open(age_path) as src:
        age = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            age = np.where(age == nodata, np.nan, age)

        same_grid = (
            src.crs == reference_meta["crs"]
            and src.transform == reference_meta["transform"]
            and src.width == reference_meta["width"]
            and src.height == reference_meta["height"]
        )

        if same_grid:
            return age

        if not align:
            raise ValueError(
                "The age raster does not match the ecoregion raster grid. "
                "Use --align-age-raster to resample the age raster to the ecoregion grid."
            )

        destination = np.full(
            (reference_meta["height"], reference_meta["width"]),
            np.nan,
            dtype="float32",
        )
        reproject(
            source=age,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=reference_meta["transform"],
            dst_crs=reference_meta["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        return destination


def load_lut(lut_path: Path, max_age: int) -> pd.DataFrame:
    """Load and validate one age-carbon lookup table."""
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    lut = pd.read_csv(lut_path)
    missing = REQUIRED_LUT_COLUMNS.difference(lut.columns)
    if missing:
        raise ValueError(f"{lut_path} is missing required columns: {sorted(missing)}")

    lut = lut.sort_values("Age").reset_index(drop=True)
    lut["Age"] = lut["Age"].astype(int)
    lut["AGC_LUT"] = lut["AGC_LUT"].astype(float)
    lut["AGC_LUT_std"] = lut["AGC_LUT_std"].astype(float)

    if lut["Age"].min() > 1:
        raise ValueError(f"{lut_path} must include age 1 or lower.")
    if lut["Age"].max() < max_age:
        raise ValueError(
            f"{lut_path} supports ages only up to {lut['Age'].max()}, "
            f"but --max-age is {max_age}."
        )

    lut = lut.set_index("Age").reindex(np.arange(1, max_age + 1))
    if lut[["AGC_LUT", "AGC_LUT_std"]].isna().any().any():
        raise ValueError(f"{lut_path} contains missing LUT values within ages 1-{max_age}.")

    return lut


def write_geotiff(path: Path, array: np.ndarray, reference_meta: dict, nodata: float) -> None:
    """Write a single-band float32 GeoTIFF using reference raster metadata."""
    output_meta = reference_meta.copy()
    output_meta.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "nodata": nodata,
            "compress": "lzw",
        }
    )
    with rasterio.open(path, "w", **output_meta) as dst:
        dst.write(array.astype("float32"), 1)


def map_lut_to_rasters(
    ecoregion_raster_path: Path,
    age_raster_path: Path,
    lut_dir: Path,
    output_agc_path: Path,
    output_uncertainty_path: Path,
    summary_path: Path,
    ecoregion_map: Dict[int, str],
    max_age: int,
    invalid_ecoregion_value: float,
    output_nodata: float,
    align_age_raster: bool,
) -> None:
    """Map region-specific AGC LUT values to raster pixels."""
    ecoregion, eco_meta = read_single_band_raster(ecoregion_raster_path)
    ecoregion = ecoregion.astype("float32")
    ecoregion = np.where(ecoregion == invalid_ecoregion_value, np.nan, ecoregion)

    age = read_age_raster_matched(age_raster_path, eco_meta, align_age_raster)

    agc_raster = np.full(ecoregion.shape, output_nodata, dtype="float32")
    uncertainty_raster = np.full(ecoregion.shape, output_nodata, dtype="float32")

    summary_records = []

    for eco_code in sorted(ecoregion_map):
        eco_name = ecoregion_map[eco_code]
        mask = ecoregion == eco_code

        if not np.any(mask):
            summary_records.append(
                {
                    "eco_code": eco_code,
                    "eco_name": eco_name,
                    "status": "not_present_in_raster",
                    "valid_pixels": 0,
                    "agc_min": np.nan,
                    "agc_max": np.nan,
                    "uncertainty_min": np.nan,
                    "uncertainty_max": np.nan,
                }
            )
            continue

        lut_path = lut_dir / f"AGC_LUT_{eco_name}.csv"
        lut = load_lut(lut_path, max_age)

        pixel_ages = age[mask]
        valid_age_mask = np.isfinite(pixel_ages)
        if not np.any(valid_age_mask):
            summary_records.append(
                {
                    "eco_code": eco_code,
                    "eco_name": eco_name,
                    "status": "no_valid_age_pixels",
                    "valid_pixels": 0,
                    "agc_min": np.nan,
                    "agc_max": np.nan,
                    "uncertainty_min": np.nan,
                    "uncertainty_max": np.nan,
                }
            )
            continue

        full_mask_indices = np.where(mask)
        valid_rows = full_mask_indices[0][valid_age_mask]
        valid_cols = full_mask_indices[1][valid_age_mask]

        clipped_ages = np.clip(pixel_ages[valid_age_mask].astype(int), 1, max_age)
        agc_values = lut.loc[clipped_ages, "AGC_LUT"].to_numpy(dtype="float32")
        uncertainty_values = lut.loc[clipped_ages, "AGC_LUT_std"].to_numpy(dtype="float32")

        agc_raster[valid_rows, valid_cols] = agc_values
        uncertainty_raster[valid_rows, valid_cols] = uncertainty_values

        summary_records.append(
            {
                "eco_code": eco_code,
                "eco_name": eco_name,
                "status": "mapped",
                "valid_pixels": int(len(agc_values)),
                "agc_min": float(np.nanmin(agc_values)),
                "agc_max": float(np.nanmax(agc_values)),
                "uncertainty_min": float(np.nanmin(uncertainty_values)),
                "uncertainty_max": float(np.nanmax(uncertainty_values)),
            }
        )

        print(
            f"{eco_name}: mapped {len(agc_values)} pixels; "
            f"AGC range = {np.nanmin(agc_values):.2f}-{np.nanmax(agc_values):.2f}; "
            f"uncertainty range = {np.nanmin(uncertainty_values):.2f}-{np.nanmax(uncertainty_values):.2f}"
        )

    output_agc_path.parent.mkdir(parents=True, exist_ok=True)
    output_uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_geotiff(output_agc_path, agc_raster, eco_meta, output_nodata)
    write_geotiff(output_uncertainty_path, uncertainty_raster, eco_meta, output_nodata)
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)

    print(f"AGC LUT raster saved to: {output_agc_path}")
    print(f"AGC uncertainty raster saved to: {output_uncertainty_path}")
    print(f"Mapping summary saved to: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    """Create command-line interface."""
    parser = argparse.ArgumentParser(
        description="Map region-specific Chapman-Richards AGC LUTs to raster pixels."
    )
    parser.add_argument("--ecoregion-raster", required=True, type=Path, help="Input ecoregion raster GeoTIFF.")
    parser.add_argument("--age-raster", required=True, type=Path, help="Input stand-age raster GeoTIFF.")
    parser.add_argument("--lut-dir", default=Path("."), type=Path, help="Directory containing AGC_LUT_<region>.csv files.")
    parser.add_argument(
        "--output-agc-raster",
        default=Path("agc_lut_raster.tif"),
        type=Path,
        help="Output GeoTIFF for mapped AGC_LUT values.",
    )
    parser.add_argument(
        "--output-uncertainty-raster",
        default=Path("agc_uncertainty_raster.tif"),
        type=Path,
        help="Output GeoTIFF for mapped AGC_LUT_std values.",
    )
    parser.add_argument(
        "--summary-csv",
        default=Path("agc_lut_raster_mapping_summary.csv"),
        type=Path,
        help="Output CSV summary of mapped pixels and value ranges.",
    )
    parser.add_argument("--max-age", default=258, type=int, help="Maximum age supported by the LUT files.")
    parser.add_argument(
        "--invalid-ecoregion-value",
        default=0,
        type=float,
        help="Ecoregion raster value to treat as invalid/background.",
    )
    parser.add_argument(
        "--output-nodata",
        default=-9999.0,
        type=float,
        help="NoData value used in output GeoTIFFs.",
    )
    parser.add_argument(
        "--ecoregion-map",
        default=None,
        help=(
            "Optional JSON string or JSON file defining the ecoregion code-to-name mapping, "
            "e.g. '{\"2\": \"BD\", \"3\": \"NE\", \"4\": \"ND\", \"5\": \"Mixed\"}'."
        ),
    )
    parser.add_argument(
        "--align-age-raster",
        action="store_true",
        help="Resample the age raster to the ecoregion raster grid if their grids differ.",
    )
    return parser


def main() -> None:
    """Run the raster mapping workflow from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args()

    ecoregion_map = parse_ecoregion_map(args.ecoregion_map)

    map_lut_to_rasters(
        ecoregion_raster_path=args.ecoregion_raster,
        age_raster_path=args.age_raster,
        lut_dir=args.lut_dir,
        output_agc_path=args.output_agc_raster,
        output_uncertainty_path=args.output_uncertainty_raster,
        summary_path=args.summary_csv,
        ecoregion_map=ecoregion_map,
        max_age=args.max_age,
        invalid_ecoregion_value=args.invalid_ecoregion_value,
        output_nodata=args.output_nodata,
        align_age_raster=args.align_age_raster,
    )


if __name__ == "__main__":
    main()
