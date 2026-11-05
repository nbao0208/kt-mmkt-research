"""
One-time Auxiliary Metadata Preparation Utility for ACD-MMKTv3.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Tasks:
1. Compute geometric letterbox patch validity mask M_patch_q in {0, -1e9}^49 and binary patch_valid_mask
   for all 7,652 questions from original image dimensions on the 7x7 spatial patch grid.
2. Estimate leak-free two-tier hierarchical Empirical Bayes priors across 5 folds:
   - Skill-level smoothed failure rate bar_D_KC^(f) using m_kc = 10.0.
   - Question prior mu_q^(prior, f) in (0, 1) and fixed logit concept anchor mu_concept_q^(f) = logit(mu_q^(prior, f)).
   - Question observation count N_q^(train, f), confidence weight omega_q^(f) = N_q / (N_q + kappa) (kappa = 10.0),
     and smoothed CTT failure rate D_q^(f) using light shrinkage m_item = 1.0.
3. Synchronize pedagogical text token-to-subword alignment tables (L_q = 128, L_s = 160)
   using identical Unicode NFKC, math formula normalization ($$...$$ to $...$), [BLANK], and [IMG_k] token mapping.
4. Save everything to data/processed/XES3G5M/acd_v3_aux_metadata.pt.
"""

import os
import sys
import re
import json
import argparse
import unicodedata
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import numpy as np
import torch
from transformers import AutoTokenizer


def compute_letterbox_patch_masks(
    raw_dir: str,
    questions_raw: Dict[str, Any],
    modality_masks: np.ndarray,
    num_questions: int = 7652,
    grid_size: int = 7,
    target_canvas: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 7x7 spatial patch valid mask from raw image geometries.

    Args:
        raw_dir: Path to raw XES3G5M directory.
        questions_raw: Dictionary of raw question metadata.
        modality_masks: Binary array [7652, 3] indicating modality availability.
        num_questions: Total number of questions (7652).
        grid_size: Spatial grid dimension (7 -> 49 patches).
        target_canvas: Letterbox square canvas dimension (224 pixels).

    Returns:
        Tuple of (M_patch_q [7652, 49] float32, patch_valid_mask [7652, 49] bool).
    """
    print("Computing geometric letterbox patch masks on 7x7 grid...")
    images_dir = os.path.join(raw_dir, "metadata", "images")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(raw_dir, "images")

    disk_files = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
    disk_stem_map = {os.path.splitext(fn)[0]: fn for fn in disk_files}

    img_ref_regex = re.compile(r"((?:question|analysis)_\d+-image_\d+)(?:\.png)?")
    patch_dim = target_canvas // grid_size  # 32 pixels per patch

    patch_valid_mask = torch.ones((num_questions, grid_size * grid_size), dtype=torch.bool)
    image_sizes_cache = {}
    image_boxes = {}

    for qid in range(num_questions):
        has_stem_img = bool(modality_masks[qid, 0] > 0.5)
        if not has_stem_img:
            # Questions without stem image: all 49 patches are set to valid as safe default
            patch_valid_mask[qid] = True
            continue

        qid_str = str(qid)
        q_info = questions_raw.get(qid_str, {})
        content_matches = img_ref_regex.findall(q_info.get("content", ""))
        opt_matches = []
        raw_options = q_info.get("options", {})
        if isinstance(raw_options, dict):
            for opt_val in raw_options.values():
                opt_matches.extend(img_ref_regex.findall(str(opt_val)))

        all_stem_imgs = content_matches + opt_matches
        stem_files = [disk_stem_map[s] for s in all_stem_imgs if s in disk_stem_map]
        stem_files = sorted(list(dict.fromkeys(stem_files)))

        if not stem_files:
            patch_valid_mask[qid] = True
            continue

        q_grid_mask = np.zeros((grid_size, grid_size), dtype=bool)

        for img_fn in stem_files:
            if img_fn not in image_sizes_cache:
                img_path = os.path.join(images_dir, img_fn)
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                        image_sizes_cache[img_fn] = (w, h)
                except Exception as e:
                    print(f"Warning: could not open image {img_path}: {e}")
                    image_sizes_cache[img_fn] = (target_canvas, target_canvas)

            w_orig, h_orig = image_sizes_cache[img_fn]
            scale = target_canvas / max(w_orig, h_orig)
            w_new = max(1, int(round(w_orig * scale)))
            h_new = max(1, int(round(h_orig * scale)))

            x0 = (target_canvas - w_new) // 2
            y0 = (target_canvas - h_new) // 2
            x1 = x0 + w_new
            y1 = y0 + h_new

            if qid not in image_boxes:
                image_boxes[qid] = {
                    "orig_size": (w_orig, h_orig),
                    "letterbox_bbox_224": (x0, y0, x1, y1),
                }

            # Check overlap for each 32x32 cell
            for r in range(grid_size):
                cell_y0 = r * patch_dim
                cell_y1 = cell_y0 + patch_dim
                for c in range(grid_size):
                    cell_x0 = c * patch_dim
                    cell_x1 = cell_x0 + patch_dim

                    overlap_w = max(0, min(cell_x1, x1) - max(cell_x0, x0))
                    overlap_h = max(0, min(cell_y1, y1) - max(cell_y0, y0))

                    if overlap_w > 0 and overlap_h > 0:
                        q_grid_mask[r, c] = True

        # Safety guarantee: at least center patch must be valid
        if not q_grid_mask.any():
            q_grid_mask[grid_size // 2, grid_size // 2] = True

        patch_valid_mask[qid] = torch.from_numpy(q_grid_mask.reshape(-1))

    # M_patch_q: 0.0 for valid patches, -1e9 for padding letterbox regions
    M_patch_q = torch.zeros((num_questions, grid_size * grid_size), dtype=torch.float32)
    M_patch_q[~patch_valid_mask] = -1e9

    valid_counts = patch_valid_mask[modality_masks[:, 0] > 0.5].sum(dim=1).float()
    print(
        f"Letterbox masks generated for {num_questions} questions. "
        f"Stem image questions ({int((modality_masks[:, 0] > 0.5).sum())}): "
        f"Mean valid patches: {valid_counts.mean().item():.2f} / 49, "
        f"Min valid patches: {int(valid_counts.min().item())}, "
        f"Max valid patches: {int(valid_counts.max().item())}."
    )
    return M_patch_q, patch_valid_mask, image_boxes


def compute_per_fold_hierarchical_bayes_statistics(
    data_dir: str,
    static_bank_path: str,
    num_questions: int = 7652,
    num_kcs: int = 1175,
    kappa: float = 10.0,
    m_kc: float = 10.0,
    m_item: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute leak-free two-tier hierarchical Empirical Bayes priors and statistics across 5 folds.

    Returns:
        Tuple of (N_q_folds, D_q_folds, omega_q_folds, mu_prior_folds, mu_concept_folds).
    """
    print("Computing leak-free hierarchical Empirical Bayes statistics across 5 folds...")
    # Load question-to-KC mapping from static bank
    static_bank = torch.load(static_bank_path, map_location="cpu", mmap=True)
    leaf_kcs = static_bank["question_leaf_kcs"]  # [7652, 6]
    leaf_kc_mask = static_bank["question_kc_mask"]  # [7652, 6]

    fold_sequences = []
    for f in range(5):
        fold_file = os.path.join(data_dir, f"sequences_fold_{f}.pt")
        if not os.path.exists(fold_file):
            raise FileNotFoundError(f"Fold file not found: {fold_file}")
        fold_sequences.append(torch.load(fold_file, map_location="cpu"))

    N_q_folds = torch.zeros((5, num_questions), dtype=torch.int64)
    E_q_folds = torch.zeros((5, num_questions), dtype=torch.int64)
    D_q_folds = torch.zeros((5, num_questions), dtype=torch.float32)
    omega_q_folds = torch.zeros((5, num_questions), dtype=torch.float32)
    mu_prior_folds = torch.zeros((5, num_questions), dtype=torch.float32)
    mu_concept_folds = torch.zeros((5, num_questions), dtype=torch.float32)

    for val_fold in range(5):
        # Accumulate interactions over the 4 training folds
        N_kc = torch.zeros(num_kcs, dtype=torch.float64)
        E_kc = torch.zeros(num_kcs, dtype=torch.float64)
        total_interactions = 0
        total_errors = 0

        for train_fold in range(5):
            if train_fold == val_fold:
                continue
            data = fold_sequences[train_fold]
            q_seq = data["q_seq"]        # [B, L]
            r_seq = data["r_seq"]        # [B, L]
            seq_mask = data["seq_mask"]  # [B, L]

            valid_q = q_seq[seq_mask].long()
            valid_r = r_seq[seq_mask].long()
            n_valid = len(valid_q)
            total_interactions += n_valid

            # Item level counts
            N_q_folds[val_fold] += torch.bincount(valid_q, minlength=num_questions)
            err_mask = (valid_r == 0)
            n_err = int(err_mask.sum().item())
            total_errors += n_err
            if n_err > 0:
                E_q_folds[val_fold] += torch.bincount(valid_q[err_mask], minlength=num_questions)

            # Accumulate KC counts from question leaf KCs
            # Expand each interaction to its associated leaf KCs
            batch_leaf_kcs = leaf_kcs[valid_q]        # [n_valid, 6]
            batch_kc_mask = leaf_kc_mask[valid_q]      # [n_valid, 6]

            flat_kcs = batch_leaf_kcs[batch_kc_mask].long()
            # For errors, repeat valid_r across valid KCs
            expanded_r = valid_r.unsqueeze(-1).expand_as(batch_leaf_kcs)[batch_kc_mask].long()
            flat_err_kcs = flat_kcs[expanded_r == 0]

            N_kc += torch.bincount(flat_kcs, minlength=num_kcs).double()
            if len(flat_err_kcs) > 0:
                E_kc += torch.bincount(flat_err_kcs, minlength=num_kcs).double()

        # Global training error rate
        global_err_rate = float(total_errors) / max(1.0, float(total_interactions))

        # Smoothed KC failure rate: bar_D_KC(k)
        bar_D_kc = (E_kc + m_kc * global_err_rate) / (N_kc + m_kc)  # [num_kcs] in (0, 1)

        # Question prior mu_q^(prior, f) averaged across its leaf KCs
        for q in range(num_questions):
            q_kcs = leaf_kcs[q][leaf_kc_mask[q]]
            if len(q_kcs) > 0:
                prior_val = float(bar_D_kc[q_kcs].mean().item())
            else:
                prior_val = global_err_rate
            # Clamp strictly in (1e-4, 1 - 1e-4) to avoid infinite logit
            prior_val = max(1e-4, min(1.0 - 1e-4, prior_val))
            mu_prior_folds[val_fold, q] = prior_val
            # Logit concept difficulty: mu_concept = ln(prior / (1 - prior))
            mu_concept_folds[val_fold, q] = float(np.log(prior_val / (1.0 - prior_val)))

        # Item-level smoothed CTT failure rate D_q^(f) and confidence weight omega_q^(f)
        n_q = N_q_folds[val_fold].float()
        e_q = E_q_folds[val_fold].float()
        prior_q = mu_prior_folds[val_fold]

        # Shrinkage towards question prior mu_q^(prior) with m_item = 1.0
        d_q = (e_q + m_item * prior_q) / (n_q + m_item)
        d_q = torch.clamp(d_q, min=1e-4, max=1.0 - 1e-4)
        omega_q = n_q / (n_q + kappa)

        D_q_folds[val_fold] = d_q
        omega_q_folds[val_fold] = omega_q

        print(
            f"  Fold {val_fold}: {total_interactions} active interactions. "
            f"Global error rate: {global_err_rate:.4f}. "
            f"Mean mu_prior: {mu_prior_folds[val_fold].mean().item():.4f}, "
            f"Mean mu_concept (logit): {mu_concept_folds[val_fold].mean().item():.4f}, "
            f"Mean D_q: {d_q.mean().item():.4f}, "
            f"Mean omega_q: {omega_q.mean().item():.4f}."
        )

    return N_q_folds, D_q_folds, omega_q_folds, mu_prior_folds, mu_concept_folds


def clean_text_segment_pedagogical(text: str, img_ref_regex: re.Pattern) -> str:
    """
    Standardize text segment identical to XES3G5MPreprocessor Step 2:
    NFKC Unicode normalization, HTML stripping, [IMG_k] token mapping, $$...$$ to $...$.
    """
    if not text:
        return ""

    # Normalize Unicode to NFKC
    text = unicodedata.normalize("NFKC", text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    text = text.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")

    # Replace image filenames with standardized semantic tokens [IMG_k]
    def replace_img_token(match: re.Match) -> str:
        raw_stem = match.group(1)
        idx_match = re.search(r"image_(\d+)", raw_stem)
        k = idx_match.group(1) if idx_match else "0"
        return f" [IMG_{k}] "

    text = img_ref_regex.sub(replace_img_token, text)

    # Normalize double dollar math blocks $$...$$ to inline $...$
    text = re.sub(r"\$\$\s*(.*?)\s*\$\$", r"$\1$", text)

    # Clean multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text_and_image_metadata_v3(
    raw_metadata_path: str,
    images_dir: str,
    num_questions: int = 7652,
    max_token_len_q: int = 128,
    max_token_len_s: int = 160,
    text_model_name: str = "hfl/chinese-roberta-wwm-ext",
    image_boxes: Optional[Dict[int, Dict[str, Any]]] = None,
    static_bank: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[str]], List[List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build token lookup tables and image metadata for explainability.
    """
    print(f"Loading questions metadata from {raw_metadata_path}...")
    with open(raw_metadata_path, "r", encoding="utf-8") as f:
        questions_raw = json.load(f)

    print(f"Loading tokenizer: {text_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    img_ref_regex = re.compile(r"((?:question|analysis)_\d+-image_\d+)(?:\.png)?")
    blank_regex = re.compile(r"_{2,}|（\s*）|\(\s*\)")

    token_to_word_q = []
    token_to_word_s = []
    image_meta_q = []
    image_meta_s = []

    valid_disk_files = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
    disk_stem_map = {os.path.splitext(fn)[0]: fn for fn in valid_disk_files}

    for qid in range(num_questions):
        qid_str = str(qid)
        q_info = questions_raw.get(qid_str, {})

        raw_content = q_info.get("content", "")
        raw_analysis = q_info.get("analysis", "")
        raw_answer = q_info.get("answer", [])
        q_type = q_info.get("type", "填空")
        raw_options = q_info.get("options", {})

        # Pedagogical normalization for stem text T_q
        cleaned_content = clean_text_segment_pedagogical(raw_content, img_ref_regex)
        cleaned_content = blank_regex.sub(" [BLANK] ", cleaned_content)
        cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()

        if q_type == "单选" and isinstance(raw_options, dict) and raw_options:
            option_items = []
            for opt_key in sorted(raw_options.keys()):
                opt_val = clean_text_segment_pedagogical(str(raw_options[opt_key]), img_ref_regex)
                option_items.append(f"选项 {opt_key}: {opt_val}")
            options_str = " [SEP] " + " | ".join(option_items)
            t_q = cleaned_content + options_str
        else:
            t_q = cleaned_content

        # Pedagogical normalization for solution text T_s
        ans_str = "".join(str(a) for a in raw_answer) if isinstance(raw_answer, list) else str(raw_answer)
        cleaned_analysis = clean_text_segment_pedagogical(raw_analysis, img_ref_regex)
        t_s = f"标准答案: {ans_str} [SEP] 解析: {cleaned_analysis}"

        # Tokenize stem and solution
        encoded_q = tokenizer(
            t_q,
            max_length=max_token_len_q,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        tokens_q = tokenizer.convert_ids_to_tokens(encoded_q["input_ids"])
        token_to_word_q.append(tokens_q)

        encoded_s = tokenizer(
            t_s,
            max_length=max_token_len_s,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        tokens_s = tokenizer.convert_ids_to_tokens(encoded_s["input_ids"])
        token_to_word_s.append(tokens_s)

        # Verify mask length and element-wise alignment if static_bank provided
        if static_bank is not None and "acd_mask_txt_q" in static_bank:
            att_mask_q = np.array(encoded_q["attention_mask"], dtype=bool)
            bank_mask_q = static_bank["acd_mask_txt_q"][qid].numpy()
            assert np.array_equal(att_mask_q, bank_mask_q), f"Attention mask element-wise mismatch on qid {qid}"

        # Stem image metadata
        content_matches = img_ref_regex.findall(raw_content)
        opt_matches = []
        if isinstance(raw_options, dict):
            for opt_val in raw_options.values():
                opt_matches.extend(img_ref_regex.findall(str(opt_val)))
        all_stem_imgs = content_matches + opt_matches
        stem_files = [disk_stem_map[s] for s in all_stem_imgs if s in disk_stem_map]
        stem_files = sorted(list(dict.fromkeys(stem_files)))

        if stem_files:
            stem_img_file = stem_files[0]
            bbox_info = (image_boxes.get(qid) if image_boxes else None) or {}
            image_meta_q.append({
                "has_image": True,
                "file_name": stem_img_file,
                "rel_path": os.path.join("metadata", "images", stem_img_file),
                "all_files": stem_files,
                "orig_size": bbox_info.get("orig_size"),
                "letterbox_bbox_224": bbox_info.get("letterbox_bbox_224"),
            })
        else:
            image_meta_q.append({
                "has_image": False,
                "file_name": None,
                "rel_path": None,
                "all_files": [],
                "orig_size": None,
                "letterbox_bbox_224": None,
            })

        # Solution image metadata
        sol_matches = img_ref_regex.findall(raw_analysis)
        sol_files = [disk_stem_map[s] for s in sol_matches if s in disk_stem_map]
        sol_files = sorted(list(dict.fromkeys(sol_files)))

        if sol_files:
            sol_img_file = sol_files[0]
            image_meta_s.append({
                "has_image": True,
                "file_name": sol_img_file,
                "rel_path": os.path.join("metadata", "images", sol_img_file),
                "all_files": sol_files,
            })
        else:
            image_meta_s.append({"has_image": False, "file_name": None, "rel_path": None, "all_files": []})

    print(
        f"Token lookup tables generated for {len(token_to_word_q)} questions. "
        f"Stem images found: {sum(1 for m in image_meta_q if m['has_image'])}, "
        f"Solution images found: {sum(1 for m in image_meta_s if m['has_image'])}."
    )
    return token_to_word_q, token_to_word_s, image_meta_q, image_meta_s


def main():
    parser = argparse.ArgumentParser(description="Prepare ACD-MMKTv3 Auxiliary Metadata")
    parser.add_argument(
        "--data-dir", "--data_dir",
        dest="data_dir",
        type=str,
        default="data/processed/XES3G5M",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--raw-dir", "--raw_dir",
        dest="raw_dir",
        type=str,
        default="data/raw/XES3G5M",
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output-file", "--output_file",
        dest="output_file",
        type=str,
        default="data/processed/XES3G5M/acd_v3_aux_metadata.pt",
        help="Output path for auxiliary metadata .pt file",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=10.0,
        help="Bayesian sample size confidence parameter kappa",
    )
    parser.add_argument(
        "--m-kc", "--m_kc",
        dest="m_kc",
        type=float,
        default=10.0,
        help="Empirical Bayes KC prior smoothing constant",
    )
    parser.add_argument(
        "--m-item", "--m_item",
        dest="m_item",
        type=float,
        default=1.0,
        help="Empirical Bayes item smoothing constant",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    num_questions = 7652
    num_kcs = 1175

    questions_json_path = os.path.join(args.raw_dir, "metadata", "questions.json")
    if not os.path.exists(questions_json_path):
        questions_json_path = os.path.join(args.raw_dir, "questions.json")

    images_dir = os.path.join(args.raw_dir, "metadata", "images")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(args.raw_dir, "images")

    static_bank_path = os.path.join(args.data_dir, "static_item_bank.pt")
    static_bank_header = torch.load(static_bank_path, map_location="cpu", mmap=True)
    modality_masks = static_bank_header["modality_masks"].numpy()

    with open(questions_json_path, "r", encoding="utf-8") as f:
        questions_raw = json.load(f)

    # 1. Compute Letterbox Patch Mask M_patch_q [7652, 49]
    M_patch_q, patch_valid_mask, image_boxes = compute_letterbox_patch_masks(
        raw_dir=args.raw_dir,
        questions_raw=questions_raw,
        modality_masks=modality_masks,
        num_questions=num_questions,
    )

    # 2. Compute leak-free Hierarchical Empirical Bayes Statistics across 5 Folds
    N_q_folds, D_q_folds, omega_q_folds, mu_prior_folds, mu_concept_folds = (
        compute_per_fold_hierarchical_bayes_statistics(
            data_dir=args.data_dir,
            static_bank_path=static_bank_path,
            num_questions=num_questions,
            num_kcs=num_kcs,
            kappa=args.kappa,
            m_kc=args.m_kc,
            m_item=args.m_item,
        )
    )

    # 3. Synchronize pedagogical text tokens and image metadata
    token_to_word_q, token_to_word_s, image_meta_q, image_meta_s = (
        build_text_and_image_metadata_v3(
            raw_metadata_path=questions_json_path,
            images_dir=images_dir,
            num_questions=num_questions,
            image_boxes=image_boxes,
            static_bank=static_bank_header,
        )
    )

    # 4. Package and serialize
    aux_metadata = {
        "M_patch_q": M_patch_q,                  # [7652, 49] float32
        "patch_valid_mask": patch_valid_mask,    # [7652, 49] bool
        "N_q_folds": N_q_folds,                  # [5, 7652] int64
        "D_q_folds": D_q_folds,                  # [5, 7652] float32
        "omega_q_folds": omega_q_folds,          # [5, 7652] float32
        "mu_prior_folds": mu_prior_folds,        # [5, 7652] float32
        "mu_concept_folds": mu_concept_folds,    # [5, 7652] float32
        "token_to_word_q": token_to_word_q,
        "token_to_word_s": token_to_word_s,
        "image_meta_q": image_meta_q,
        "image_meta_s": image_meta_s,
        "kappa": args.kappa,
        "m_kc": args.m_kc,
        "m_item": args.m_item,
    }

    torch.save(aux_metadata, args.output_file)
    print(
        f"Successfully generated and saved ACD-MMKTv3 auxiliary metadata to {args.output_file} "
        f"({os.path.getsize(args.output_file) / 1e6:.2f} MB)."
    )


if __name__ == "__main__":
    main()

