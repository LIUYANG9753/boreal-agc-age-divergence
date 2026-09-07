#!/usr/bin/env python3
"""
Generate aboveground biomass carbon (AGC) lookup tables using Chapman-Richards curves.

This script propagates parameter uncertainty from fitted Chapman-Richards models by
Monte Carlo simulation. For each forest/eco-region, parameters a, k, and c are
sampled from normal distributions defined by their fitted estimates and standard
errors. Negative parameter draws are truncated to zero. The output is an age-wise
lookup table containing the simulated mean AGC and its standard deviation.

Model:
    AGC(age) = a * (1 - exp(-k * age)) ** c

Default parameters correspond to the values used in the accompanying manuscript
analysis. Users may provide an external parameter CSV to reproduce or update the
lookup tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChapmanRichardsParams:
    """Container for Chapman-Richards parameter estimates and standard errors."""

    a: float
    a_se: float
    k: float
    k_se: float
    c: float
    c_se: float


DEFAULT_REGION_PARAMS: Dict[str, ChapmanRichardsParams] = {
    "BD": ChapmanRichardsParams(a=88.72, a_se=1.95, k=0.0412, k_se=0.0132, c=2.48, c_se=1.42),
    "Mixed": ChapmanRichardsParams(a=103.03, a_se=3.68, k=0.0382, k_se=0.0073, c=2.55, c_se=0.88),
    "ND": ChapmanRichardsParams(a=95.27, a_se=4.41, k=0.0048, k_se=0.0030, c=0.50, c_se=0.25),
    "NE": ChapmanRichardsParams(a=76.24, a_se=1.44, k=0.0522, k_se=0.0191, c=3.13, c_se=2.24),
}


def chapman_richards(age: np.ndarray, a: float, k: float, c: float) -> np.ndarray:
    """Evaluate the Chapman-Richards growth function for a vector of stand ages."""
    base = np.maximum(1.0 - np.exp(-k * age), 0.0)
    return a * np.power(base, c)


def read_parameter_table(path: Path) -> Dict[str, ChapmanRichardsParams]:
    """
    Read Chapman-Richards parameters from a CSV file.

    Required columns are: region, a, a_se, k, k_se, c, c_se.
    """
    df = pd.read_csv(path)
    required = {"region", "a", "a_se", "k", "k_se", "c", "c_se"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Parameter file is missing required columns: {sorted(missing)}")

    params: Dict[str, ChapmanRichardsParams] = {}
    for _, row in df.iterrows():
        region = str(row["region"])
        params[region] = ChapmanRichardsParams(
            a=float(row["a"]),
            a_se=float(row["a_se"]),
            k=float(row["k"]),
            k_se=float(row["k_se"]),
            c=float(row["c"]),
            c_se=float(row["c_se"]),
        )
    return params


def simulate_agc_lookup_table(
    ages: np.ndarray,
    params: ChapmanRichardsParams,
    n_simulations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate an AGC lookup table for one region by Monte Carlo simulation.

    Parameters are independently sampled from normal distributions using fitted
    estimates as means and standard errors as standard deviations. Draws below
    zero are truncated to zero to prevent non-physical parameter values.
    """
    a_draws = np.maximum(0.0, rng.normal(params.a, params.a_se, n_simulations))
    k_draws = np.maximum(0.0, rng.normal(params.k, params.k_se, n_simulations))
    c_draws = np.maximum(0.0, rng.normal(params.c, params.c_se, n_simulations))

    simulations = np.empty((len(ages), n_simulations), dtype=float)
    for i, (a_i, k_i, c_i) in enumerate(zip(a_draws, k_draws, c_draws)):
        simulations[:, i] = chapman_richards(ages, a_i, k_i, c_i)

    return pd.DataFrame(
        {
            "Age": ages.astype(int),
            "AGC_LUT": np.nanmean(simulations, axis=1),
            "AGC_LUT_std": np.nanstd(simulations, axis=1, ddof=1),
        }
    )


def generate_lookup_tables(
    region_params: Mapping[str, ChapmanRichardsParams],
    ages: Iterable[int],
    n_simulations: int,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate and save lookup tables for all regions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    age_array = np.asarray(list(ages), dtype=float)

    summary_rows = []
    for region, params in region_params.items():
        lut = simulate_agc_lookup_table(
            ages=age_array,
            params=params,
            n_simulations=n_simulations,
            rng=rng,
        )
        output_path = output_dir / f"AGC_LUT_{region}.csv"
        lut.to_csv(output_path, index=False)

        if lut[["AGC_LUT", "AGC_LUT_std"]].isna().any().any():
            print(f"Warning: NaN values detected in simulations for {region}.")

        summary_rows.append(
            {
                "region": region,
                "n_ages": len(lut),
                "age_min": int(lut["Age"].min()),
                "age_max": int(lut["Age"].max()),
                "agc_min": float(lut["AGC_LUT"].min()),
                "agc_max": float(lut["AGC_LUT"].max()),
                "agc_std_min": float(lut["AGC_LUT_std"].min()),
                "agc_std_max": float(lut["AGC_LUT_std"].max()),
                "output_file": str(output_path.name),
            }
        )
        print(f"Generated lookup table for {region}: {output_path}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "AGC_LUT_generation_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AGC lookup tables using Chapman-Richards parameters and Monte Carlo uncertainty propagation."
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional CSV file with columns: region,a,a_se,k,k_se,c,c_se. If omitted, built-in manuscript parameters are used.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("AGC_LUT_outputs"), help="Directory for output CSV files.")
    parser.add_argument("--min-age", type=int, default=1, help="Minimum stand age included in the lookup table.")
    parser.add_argument("--max-age", type=int, default=258, help="Maximum stand age included in the lookup table.")
    parser.add_argument("--n-simulations", type=int, default=1000, help="Number of Monte Carlo simulations per region.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible Monte Carlo simulations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_age < 0 or args.max_age < args.min_age:
        raise ValueError("Age range must satisfy 0 <= min_age <= max_age.")
    if args.n_simulations < 2:
        raise ValueError("At least two simulations are required to calculate sample standard deviation.")

    params = read_parameter_table(args.params) if args.params else DEFAULT_REGION_PARAMS
    ages = range(args.min_age, args.max_age + 1)
    summary = generate_lookup_tables(
        region_params=params,
        ages=ages,
        n_simulations=args.n_simulations,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print("\nLookup table generation completed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
