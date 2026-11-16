"""
CLI Entry Point for Eedi NeurIPS 2020 (Task 3 & 4) Multimodal KT Preprocessing.

Usage:
    conda run -n kt-research-env python scripts/preprocess_eedi_task34.py \
        --config configs/dataset/eedi_task34.yaml
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eedi_preprocessor import EediTask34Preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 11-step Eedi Task 3 & 4 Multimodal Knowledge Tracing preprocessor."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/eedi_task34.yaml",
        help="Path to the Eedi dataset YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessor = EediTask34Preprocessor(config_path=args.config)
    preprocessor.run_all()


if __name__ == "__main__":
    main()
