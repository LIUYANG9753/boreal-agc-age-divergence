# Spatial Mapping of Chapman-Richards AGC Lookup Tables

## Purpose

This code maps region-specific Chapman-Richards aboveground carbon (AGC) lookup tables to spatial raster pixels. It uses an ecoregion raster to identify the forest region of each pixel and a stand-age raster to select the corresponding AGC value from regional lookup tables. The workflow also maps the associated LUT uncertainty (`AGC_LUT_std`) to a separate raster.

The script is designed as a reproducible and configurable version of the original analysis code used to generate spatial AGC and uncertainty rasters from age and ecoregion information.

## Workflow overview

1. Read the ecoregion raster and its geospatial metadata.
2. Treat the background or invalid ecoregion code, by default `0`, as invalid.
3. Read the stand-age raster.
4. Check whether the age raster and ecoregion raster share the same spatial grid.
5. For each mapped ecoregion code:
   - Load the corresponding `AGC_LUT_<region>.csv` file.
   - Extract valid age values for pixels in that ecoregion.
   - Convert ages to integer years and clip them to the supported LUT age range.
   - Assign `AGC_LUT` values to the AGC output raster.
   - Assign `AGC_LUT_std` values to the uncertainty output raster.
6. Write the mapped AGC raster, uncertainty raster, and a CSV mapping summary.

## Input files

### 1. Ecoregion raster

A single-band GeoTIFF in which pixel values represent forest ecoregion or forest-type codes. The default mapping is:

| Raster code | Region name |
|---:|---|
| 2 | BD |
| 3 | NE |
| 4 | ND |
| 5 | Mixed |

The background value is assumed to be `0` by default.

### 2. Stand-age raster

A single-band GeoTIFF containing stand age in years. The age raster should have the same extent, resolution, coordinate reference system, and transform as the ecoregion raster. If the grids do not match, the command-line option `--align-age-raster` can be used to resample the age raster to the ecoregion grid using nearest-neighbour resampling.

### 3. Regional AGC LUT files

The LUT directory must contain files named:

- `AGC_LUT_BD.csv`
- `AGC_LUT_NE.csv`
- `AGC_LUT_ND.csv`
- `AGC_LUT_Mixed.csv`

Each file must contain the following columns:

| Column | Description |
|---|---|
| `Age` | Stand age in years |
| `AGC_LUT` | Expected AGC value from the Chapman-Richards LUT |
| `AGC_LUT_std` | Uncertainty of the LUT estimate, typically from Monte Carlo simulations |

## Output files

The script writes three outputs:

1. `agc_lut_raster.tif` — spatial raster of expected AGC values.
2. `agc_uncertainty_raster.tif` — spatial raster of AGC uncertainty values.
3. `agc_lut_raster_mapping_summary.csv` — per-region summary of mapped pixels and value ranges.

All GeoTIFF outputs are written as single-band `float32` rasters using the spatial metadata of the ecoregion raster.

## Example usage

```bash
python map_agc_lut_to_raster.py \
  --ecoregion-raster data/eco_regions_CCIPFT1.tif \
  --age-raster data/Mergeb12_age_output.tif \
  --lut-dir outputs/agc_lut \
  --output-agc-raster outputs/agc_lut_raster_CCIPFT.tif \
  --output-uncertainty-raster outputs/agc_uncertainty_raster_CR.tif \
  --summary-csv outputs/agc_lut_raster_mapping_summary.csv
```

If the age raster grid differs from the ecoregion raster grid, use:

```bash
python map_agc_lut_to_raster.py \
  --ecoregion-raster data/eco_regions_CCIPFT1.tif \
  --age-raster data/Mergeb12_age_output.tif \
  --lut-dir outputs/agc_lut \
  --align-age-raster
```

## Custom ecoregion mapping

A custom code-to-name mapping can be provided as either a JSON string or a JSON file:

```bash
python map_agc_lut_to_raster.py \
  --ecoregion-raster data/ecoregions.tif \
  --age-raster data/stand_age.tif \
  --lut-dir outputs/agc_lut \
  --ecoregion-map ecoregion_mapping_template.json
```

The JSON file should have the following format:

```json
{
  "2": "BD",
  "3": "NE",
  "4": "ND",
  "5": "Mixed"
}
```

## Reproducibility notes

- The script does not perform stochastic simulations. It maps pre-generated lookup tables to raster pixels.
- The uncertainty raster represents the LUT uncertainty already stored in the input files as `AGC_LUT_std`.
- Ages outside the supported LUT range are clipped to the interval `1` to `--max-age`, with the default maximum age set to 258 years.
- Output rasters use `-9999` as NoData by default, which avoids potential interoperability problems caused by using `NaN` as GeoTIFF NoData.

## Software requirements

The required Python packages are listed in `requirements_spatial_agc_mapping.txt`:

```bash
pip install -r requirements_spatial_agc_mapping.txt
```

## Files included in this code package

- `map_agc_lut_to_raster.py` — reproducible Python script for spatial LUT mapping.
- `ecoregion_mapping_template.json` — editable template for ecoregion code mapping.
- `requirements_spatial_agc_mapping.txt` — Python dependencies.
- `README_spatial_AGC_mapping_code_availability.md` — workflow and usage documentation.
- `SCI_code_availability_statement_spatial_AGC_mapping.md` — manuscript-ready code availability statement.
