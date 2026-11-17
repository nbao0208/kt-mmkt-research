"""
CLI Entry Point to Convert Eedi Task 3 & 4 into pyKT `kc_level` Sequence Files.

Usage:
    conda run -n kt-research-env python scripts/prepare_eedi_pykt_data.py \
        --dataset_config configs/dataset/eedi_pykt.yaml
"""

import argparse
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.eedi_pykt_preprocessor import prepare_eedi_pykt_kc_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare pyKT kc_level sequence files for Eedi NeurIPS 2020 (Task 3 & 4)."
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/dataset/eedi_pykt.yaml",
        help="Path to the Eedi pyKT dataset YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override output directory (default from dataset config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_eedi_pykt_kc_level(
        dataset_config_path=args.dataset_config,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
