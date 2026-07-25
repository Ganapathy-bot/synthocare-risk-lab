#!/usr/bin/env python3
"""
CLI entry point: generate synthetic chronic-disease cohort + reports.

Example:
  python generate_dataset.py --n-patients 3000 --seed 42

SYNTHETIC DATA ONLY — not for clinical decision-making.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_N_PATIENTS, GENERATOR_VERSION, RANDOM_SEED  # noqa: E402
from src.generate import SyntheticGenerator  # noqa: E402
from src.validate_and_report import write_all_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic chronic-disease ML dataset")
    p.add_argument("--n-patients", type=int, default=DEFAULT_N_PATIENTS, help="Number of synthetic patients")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    p.add_argument("--skip-reports", action="store_true", help="Skip validation reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("SYNTHETIC CHRONIC DISEASE DATASET GENERATOR")
    print(f"Version {GENERATOR_VERSION} | seed={args.seed} | n_patients={args.n_patients}")
    print("NOT REAL PATIENT DATA — research/education/model-development only")
    print("=" * 70)

    gen = SyntheticGenerator(n_patients=args.n_patients, seed=args.seed)
    clean, noisy = gen.run()

    print("\nClean tables:")
    for k, df in clean.items():
        print(f"  {k:22s} {len(df):8,} rows x {df.shape[1]:3d} cols")
    print("\nNoisy tables written to data/noisy/")

    if not args.skip_reports:
        print("\nRunning validation and reports on CLEAN version...")
        reports = write_all_reports(clean)
        print(f"Validation status: {reports['validation_summary'].get('overall_status')}")
        print("Reports written to reports/")

    print("\nDone. Read DATASET_CARD.md and docs/ before modeling.")


if __name__ == "__main__":
    main()
