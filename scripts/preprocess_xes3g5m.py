#!/usr/bin/env python3
"""
CLI Entry Point for XES3G5M Multimodal Preprocessing Pipeline.

Executes the complete 9-step preprocessing workflow defined in
plans/xes3g5m_mmkt_preprocessing_plan.md to prepare Tier 1 (Static Item Bank)
and Tier 2 (Dynamic Sequence Store) for MG-MMKT and ACD-MMKT.
"""

import argparse
import sys
import os
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.xes3g5m_preprocessor import XES3G5MPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess XES3G5M dataset for MG-MMKT and ACD-MMKT."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/xes3g5m.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw/XES3G5M",
        help="Directory containing raw XES3G5M data.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/XES3G5M",
        help="Directory to save preprocessed tensors.",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="hfl/chinese-roberta-wwm-ext",
        help="Pretrained text backbone model identifier.",
    )
    parser.add_argument(
        "--vision_model",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="Pretrained vision backbone model identifier.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for frozen multimodal feature extractors.",
    )
    parser.add_argument(
        "--alpha_anc",
        type=float,
        default=0.5,
        help="Hypergraph incidence attenuation for ancestor KCs.",
    )
    parser.add_argument(
        "--ctt_m",
        type=float,
        default=10.0,
        help="Empirical Bayes prior pseudo-count smoothing parameter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computing device ('mps', 'cuda', 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML configuration if present
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}...")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        raw_dir = cfg.get("raw_data_dir", args.raw_dir)
        processed_dir = cfg.get("processed_data_dir", args.processed_dir)
        text_model = cfg.get("text", {}).get("model_name", args.text_model)
        vision_model = cfg.get("vision", {}).get("model_name", args.vision_model)
        alpha_anc = cfg.get("hypergraph", {}).get("alpha_anc", args.alpha_anc)
        ctt_m = cfg.get("ctt", {}).get("prior_smoothing_m", args.ctt_m)
    else:
        raw_dir = args.raw_dir
        processed_dir = args.processed_dir
        text_model = args.text_model
        vision_model = args.vision_model
        alpha_anc = args.alpha_anc
        ctt_m = args.ctt_m

    preprocessor = XES3G5MPreprocessor(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        text_model_name=text_model,
        vision_model_name=vision_model,
        alpha_anc=alpha_anc,
        ctt_m=ctt_m,
        device=args.device,
    )

    summary = preprocessor.run_all(batch_size=args.batch_size)
    print("\nPreprocessed dataset successfully written.")
    print(f"Outputs located in: {processed_dir}")
    print(f"Summary metrics: {summary}")


if __name__ == "__main__":
    main()
