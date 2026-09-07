# Chapman-Richards AGC Lookup Table Generation

## Purpose

This code generates age-specific aboveground biomass carbon (AGC) lookup tables for multiple forest or ecological regions using the Chapman-Richards growth function. It is designed for use as supplementary code accompanying a manuscript on forest biomass carbon dynamics and carbon accumulation modelling.

The workflow propagates uncertainty in fitted Chapman-Richards model parameters through Monte Carlo simulation. For each region, the script samples the asymptotic carbon density parameter (`a`), growth-rate coefficient (`k`), and shape parameter (`c`) from normal distributions defined by their estimated values and standard errors. The resulting lookup table reports the mean simulated AGC and its standard deviation for each forest stand age.

## Model

The Chapman-Richards growth equation is:

```text
AGC(age) = a × (1 - exp(-k × age))^c
```

where:

- `AGC(age)` is aboveground biomass carbon density at a given stand age;
- `a` is the asymptotic maximum AGC;
- `k` is the growth-rate coefficient;
- `c` is the curve-shape parameter;
- `age` is forest stand age in years.

## Input parameters

The default script includes the parameter estimates and standard errors used in the manuscript analysis:

| Region | a | a_se | k | k_se | c | c_se |
|---|---:|---:|---:|---:|---:|---:|
| BD | 88.72 | 1.95 | 0.0412 | 0.0132 | 2.48 | 1.42 |
| Mixed | 103.03 | 3.68 | 0.0382 | 0.0073 | 2.55 | 0.88 |
| ND | 95.27 | 4.41 | 0.0048 | 0.0030 | 0.50 | 0.25 |
| NE | 76.24 | 1.44 | 0.0522 | 0.0191 | 3.13 | 2.24 |

Users can also supply an external parameter file using the `--params` argument. The file must be a CSV containing the following columns:

```text
region,a,a_se,k,k_se,c,c_se
```

A template is provided in `region_parameters_template.csv`.

## Output files

For each region, the script writes one CSV lookup table:

```text
AGC_LUT_<region>.csv
```

Each output table contains:

| Column | Description |
|---|---|
| `Age` | Forest stand age in years |
| `AGC_LUT` | Mean simulated AGC from Monte Carlo simulations |
| `AGC_LUT_std` | Standard deviation of simulated AGC values |

The script also writes:

```text
AGC_LUT_generation_summary.csv
```

which summarizes the age range, AGC range, uncertainty range, and output file name for each region.

## Reproducibility

The default workflow uses:

- 1,000 Monte Carlo simulations per region;
- stand ages from 1 to 258 years;
- a fixed random seed of 42.

These values can be modified through command-line arguments.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements_agc_lut.txt
```

## Example usage

Run the script using the built-in manuscript parameters:

```bash
python generate_agc_lut_chapman_richards.py --output-dir AGC_LUT_outputs
```

Run the script using an external parameter table:

```bash
python generate_agc_lut_chapman_richards.py \
  --params region_parameters_template.csv \
  --output-dir AGC_LUT_outputs \
  --min-age 1 \
  --max-age 258 \
  --n-simulations 1000 \
  --seed 42
```

## Workflow logic

1. Define or read region-specific Chapman-Richards parameter estimates and standard errors.
2. Generate an integer age vector for the selected age range.
3. For each region, repeatedly sample `a`, `k`, and `c` from normal distributions.
4. Truncate negative parameter draws to zero to avoid non-physical model parameters.
5. Calculate AGC for each age and each Monte Carlo draw.
6. Estimate the mean and standard deviation of AGC across simulations.
7. Export one lookup table per region and a summary table.

## Notes for manuscript use

The lookup tables are intended to provide age-specific AGC estimates and uncertainty for subsequent spatial modelling, forest carbon mapping, or stand-age-based carbon accounting. The standard deviation reported in the output reflects propagated uncertainty from the fitted Chapman-Richards parameters under the assumption of independent normally distributed parameter errors.
