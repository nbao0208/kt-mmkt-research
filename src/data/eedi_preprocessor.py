"""
End-to-End 11-Step Multimodal Preprocessing Pipeline for Eedi NeurIPS 2020 (Task 3 & 4).

Produces Two-Tier Storage Architecture artifacts under data/processed/Eedi/ compatible
without modification with ACDMMKTv3, Ablations 1-3, MGMMKT, MGMMKTv2, and pyKT:
  Tier 1: static_item_bank.pt, mg_static_item_bank.pt, acd_v3_aux_metadata.pt
  Tier 2: sequences_fold_{0..4}.pt, sequences_test_{public,private}.pt, sequences_test.pt,
          plus stratified evaluation slice masks (test_strata_masks.pt).

Strict zero-leakage guarantees:
  - Question OCR text T_q contains zero answer keys or solution steps.
  - Answer-anchored solutions T_s^q are derived strictly from public item metadata and
    placed solely inside solution channels.
  - Empirical Bayes item difficulty priors, distractor dominance/entropy, and
    dominant-misconception flags inside post_diag_vec are computed strictly within
    each fold's training split D_train^{(f)}.
"""

import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

from src.data.eedi_slide_cv import extract_no_calc_template, extract_slide_rois_and_metadata
from src.data.eedi_vlm_ocr import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_VLM_MODEL,
    normalize_math_text,
    run_step3_ocr_extraction,
    run_step4_solution_generation,
)
from src.encoders import FrozenTextEncoder, FrozenVisionEncoder, letterbox_image_and_mask


NUM_QUESTIONS = 948
MAX_SEQ_LEN = 200
TEST_STRIDE = 100
MAX_LEN_Q = 128
MAX_LEN_S = 160
NUM_PATCHES = 49
KAPPA = 10.0
M_KC = 10.0
M_ITEM = 1.0
DIRICHLET_ALPHA = 1.0
GLOBAL_ACC_PRIOR = 0.537332
LOG_MAX_LAG = math.log1p(10_000_000.0)


class EediTask34Preprocessor:
    """
    Reproducible 11-step multimodal preprocessing orchestrator for Eedi Task 3 & 4.
    """

    def __init__(self, config_path: str = "configs/dataset/eedi_task34.yaml"):
        """Load dataset configuration and prepare canonical workspace paths."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.raw_dir = Path(self.cfg["raw_data_dir"])
        self.out_dir = Path(self.cfg["data_dir"])
        self.feat_dir = Path(self.cfg["features_dir"])
        self.splits_dir = Path(self.cfg["splits_dir"])

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        self.seed = int(self.cfg.get("seed", 42))
        self.images_dir = self.raw_dir / "images"

    # ------------------------------------------------------------------
    # Step 1: Data Isolation, Integrity Audit, Hash Dedup & KC Mapping
    # ------------------------------------------------------------------
    def run_step1_audit_and_mappings(self) -> Dict[str, Any]:
        """
        Audit raw Task 3 & 4 CSVs, verify 1-1 AnswerId join and disjoint user sets,
        resolve historical CorrectAnswer updates via latest timestamp, hash-deduplicate
        948 slides into 947 unique MD5s, and construct the 85-KC canonical hierarchy.
        """
        print("[Step 1] Loading and auditing raw Eedi Task 3 & 4 CSVs...", flush=True)
        train_df = pd.read_csv(self.raw_dir / "train_data" / "train_task_3_4.csv")
        pub_df = pd.read_csv(self.raw_dir / "test_data" / "test_public_task_4_more_splits.csv")
        priv_df = pd.read_csv(self.raw_dir / "test_data" / "test_private_task_4_more_splits.csv")
        ans_meta = pd.read_csv(self.raw_dir / "metadata" / "answer_metadata_task_3_4.csv")
        q_meta = pd.read_csv(self.raw_dir / "metadata" / "question_metadata_task_3_4.csv")
        stu_meta = pd.read_csv(self.raw_dir / "metadata" / "student_metadata_task_3_4.csv")
        subj_meta = pd.read_csv(self.raw_dir / "metadata" / "subject_metadata.csv")

        # Verify 1-1 join on AnswerId across all interaction splits
        total_rows = len(train_df) + len(pub_df) + len(priv_df)
        if len(ans_meta) != total_rows:
            raise AssertionError(f"Answer metadata rows ({len(ans_meta)}) != total interactions ({total_rows})")
        all_answer_ids = set(train_df["AnswerId"]) | set(pub_df["AnswerId"]) | set(priv_df["AnswerId"])
        if len(all_answer_ids) != total_rows or set(ans_meta["AnswerId"]) != all_answer_ids:
            raise AssertionError("AnswerId 1-to-1 integrity check failed!")

        # Verify disjoint UserId partition across train, public test, and private test & 100% student_metadata coverage
        train_uids = set(train_df["UserId"].unique())
        pub_uids = set(pub_df["UserId"].unique())
        priv_uids = set(priv_df["UserId"].unique())
        if (train_uids & pub_uids) or (train_uids & priv_uids) or (pub_uids & priv_uids):
            raise AssertionError("Non-zero UserId overlap across train/public/private splits!")
        all_uids = train_uids | pub_uids | priv_uids
        stu_uids = set(stu_meta["UserId"].unique())
        if len(stu_meta) != 6148 or all_uids != stu_uids:
            raise AssertionError(
                f"Student metadata integrity check failed! rows={len(stu_meta)}, unique={len(stu_uids)}, missing={len(all_uids - stu_uids)}"
            )

        # Merge metadata & parse Unix timestamp (epoch seconds)
        ans_meta["DateAnswered_sec"] = (
            pd.to_datetime(ans_meta["DateAnswered"]).astype("int64") // 10**9
        )
        ans_cols = ["AnswerId", "DateAnswered_sec", "Confidence", "QuizId"]
        ans_indexed = ans_meta[ans_cols]

        train_merged = train_df.merge(ans_indexed, on="AnswerId", how="left", validate="one_to_one")
        pub_merged = pub_df.merge(ans_indexed, on="AnswerId", how="left", validate="one_to_one")
        priv_merged = priv_df.merge(ans_indexed, on="AnswerId", how="left", validate="one_to_one")

        # Resolve latest-timestamp official CorrectAnswer across all interactions
        combined_key_view = pd.concat(
            [
                train_merged[["QuestionId", "AnswerId", "DateAnswered_sec", "CorrectAnswer"]],
                pub_merged[["QuestionId", "AnswerId", "DateAnswered_sec", "CorrectAnswer"]],
                priv_merged[["QuestionId", "AnswerId", "DateAnswered_sec", "CorrectAnswer"]],
            ],
            ignore_index=True,
        )
        combined_key_view.sort_values(["QuestionId", "DateAnswered_sec", "AnswerId"], inplace=True)
        latest_per_q = combined_key_view.groupby("QuestionId")["CorrectAnswer"].last()
        multi_key_qids = [
            int(q) for q, grp in combined_key_view.groupby("QuestionId")["CorrectAnswer"].nunique().items()
            if grp > 1
        ]
        correct_answer_idx = torch.tensor(
            [int(latest_per_q.loc[q]) - 1 for q in range(NUM_QUESTIONS)], dtype=torch.int64
        )
        print(f"[Step 1] Verified 1-1 join ({total_rows:,} rows). Historical key revisions on Q={multi_key_qids}", flush=True)

        # Compute MD5 deduplication across 0.jpg .. 947.jpg
        md5_to_qids: Dict[str, List[int]] = defaultdict(list)
        for qid in range(NUM_QUESTIONS):
            p = self.images_dir / f"{qid}.jpg"
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            md5_to_qids[digest].append(qid)
        dup_to_rep: Dict[int, int] = {}
        unique_rep_qids: List[int] = []
        for qids in md5_to_qids.values():
            rep = min(qids)
            unique_rep_qids.append(rep)
            for q in qids:
                dup_to_rep[q] = rep
        unique_rep_qids.sort()
        dup_pairs = [qids for qids in md5_to_qids.values() if len(qids) > 1]
        print(f"[Step 1] MD5 dedup: {len(unique_rep_qids)} unique images from {NUM_QUESTIONS}; dup groups={dup_pairs}", flush=True)

        # Build canonical 85-KC curriculum mapping (L3: 0..56, L2: 57..81, L1: 82..84; root 3 excluded)
        subj_lookup = subj_meta.set_index("SubjectId").to_dict("index")
        q_meta = q_meta.sort_values("QuestionId").reset_index(drop=True)
        q_raw_leaves: Dict[int, List[int]] = {}
        active_l3: Set[int] = set()
        active_l2: Set[int] = set()
        active_l1: Set[int] = set()

        for _, row in q_meta.iterrows():
            qid = int(row["QuestionId"])
            raw_list = [int(x) for x in ast.literal_eval(str(row["SubjectId"]))]
            # Keep deepest Level-3 leaves cleanly sorted
            leaves = [s for s in raw_list if int(subj_lookup[s]["Level"]) == 3]
            if not leaves:
                leaves = raw_list
            leaves = sorted(set(leaves))
            q_raw_leaves[qid] = leaves
            for leaf in leaves:
                active_l3.add(leaf)
                p2 = subj_lookup[leaf]["ParentId"]
                if pd.notna(p2):
                    p2_id = int(p2)
                    if int(subj_lookup[p2_id]["Level"]) == 2:
                        active_l2.add(p2_id)
                        p1 = subj_lookup[p2_id]["ParentId"]
                        if pd.notna(p1):
                            p1_id = int(p1)
                            if int(subj_lookup[p1_id]["Level"]) == 1:
                                active_l1.add(p1_id)

        sorted_l3 = sorted(active_l3)
        sorted_l2 = sorted(active_l2)
        sorted_l1 = sorted(active_l1)
        if not (len(sorted_l3) == 57 and len(sorted_l2) == 25 and len(sorted_l1) == 3):
            raise AssertionError(
                f"Expected (57 L3, 25 L2, 3 L1) KCs, got ({len(sorted_l3)}, {len(sorted_l2)}, {len(sorted_l1)})"
            )

        raw_to_kc: Dict[int, int] = {}
        kc_to_raw: Dict[int, int] = {}
        for idx, raw_id in enumerate(sorted_l3 + sorted_l2 + sorted_l1):
            raw_to_kc[raw_id] = idx
            kc_to_raw[idx] = raw_id

        # Full hierarchical curriculum strings for Step 7 RoBERTa initialization & Step 4 prompting
        kc_descriptions: List[str] = []
        for idx in range(85):
            raw_id = kc_to_raw[idx]
            node = subj_lookup[raw_id]
            lvl = int(node["Level"])
            name = str(node["Name"]).strip()
            if lvl == 1:
                desc = f"Mathematics Curriculum | Discipline: {name}"
            elif lvl == 2:
                p1_name = str(subj_lookup[int(node["ParentId"])]["Name"]).strip()
                desc = f"Mathematics Curriculum | Discipline: {p1_name} | Topic: {name}"
            else:
                p2 = subj_lookup[int(node["ParentId"])]
                p2_name = str(p2["Name"]).strip()
                p1_name = str(subj_lookup[int(p2["ParentId"])]["Name"]).strip()
                desc = f"Mathematics Curriculum | Discipline: {p1_name} | Topic: {p2_name} | Skill: {name}"
            kc_descriptions.append(desc)

        question_leaf_kcs = torch.zeros((NUM_QUESTIONS, 6), dtype=torch.int64)
        question_kc_mask = torch.zeros((NUM_QUESTIONS, 6), dtype=torch.bool)
        question_all_kcs = torch.zeros((NUM_QUESTIONS, 5), dtype=torch.int64)
        question_all_kc_mask = torch.zeros((NUM_QUESTIONS, 5), dtype=torch.bool)
        kc_names_by_qid: Dict[int, str] = {}
        kc_prefix_by_qid: Dict[int, str] = {}
        leaf_to_l1: Dict[int, int] = {}

        for raw_l3 in sorted_l3:
            k_l3 = raw_to_kc[raw_l3]
            p2_raw = int(subj_lookup[raw_l3]["ParentId"])
            p1_raw = int(subj_lookup[p2_raw]["ParentId"])
            leaf_to_l1[k_l3] = raw_to_kc[p1_raw]

        for qid in range(NUM_QUESTIONS):
            leaves = q_raw_leaves[qid]
            l3_idxs = [raw_to_kc[s] for s in leaves]
            for pos, k_idx in enumerate(l3_idxs[:6]):
                question_leaf_kcs[qid, pos] = k_idx
                question_kc_mask[qid, pos] = True

            # Deduplicated ordered ancestors [L3..., L2..., L1...] capped cleanly at 5
            l2_raws = sorted({int(subj_lookup[s]["ParentId"]) for s in leaves})
            l1_raws = sorted({int(subj_lookup[p2]["ParentId"]) for p2 in l2_raws})
            l2_idxs = [raw_to_kc[x] for x in l2_raws]
            l1_idxs = [raw_to_kc[x] for x in l1_raws]
            all_k = l3_idxs + l2_idxs + l1_idxs
            for pos, k_idx in enumerate(all_k[:5]):
                question_all_kcs[qid, pos] = k_idx
                question_all_kc_mask[qid, pos] = True
            kc_names_by_qid[qid] = " ; ".join(kc_descriptions[k] for k in l3_idxs)

            l1_str = ", ".join(str(subj_lookup[x]["Name"]).strip() for x in l1_raws)
            l2_str = ", ".join(str(subj_lookup[x]["Name"]).strip() for x in l2_raws)
            l3_str = ", ".join(str(subj_lookup[x]["Name"]).strip() for x in leaves)
            kc_prefix_by_qid[qid] = f"[DOMAIN: {l1_str} | TOPIC: {l2_str} | SKILL: {l3_str}]"

        # Parse 50 Expert Pairwise Comparison Pairs from Task 3 test_data files
        expert_pairs: List[Dict[str, Any]] = []
        for split_tag, fname in [("public", "quality_response_remapped_public.csv"),
                                 ("private", "quality_response_remapped_private.csv")]:
            ep_df = pd.read_csv(self.raw_dir / "test_data" / fname)
            for row_i, row in ep_df.iterrows():
                votes = []
                vote_map = {}
                for col in ("T1_ALR", "T2_CL", "T3_GF", "T4_MQ", "T5_NS"):
                    val = row[col]
                    if pd.notna(val):
                        ival = int(val)
                        votes.append(ival)
                        vote_map[col] = ival
                    else:
                        vote_map[col] = None
                left_v = sum(1 for v in votes if v == 1)
                right_v = sum(1 for v in votes if v == 2)
                expert_pairs.append({
                    "split": split_tag,
                    "row_idx": int(row_i),
                    "left": int(row["left"]),
                    "right": int(row["right"]),
                    "votes": vote_map,
                    "left_votes": left_v,
                    "right_votes": right_v,
                    "consensus_pref": 1 if left_v >= right_v else 2,
                })

        print(f"[Step 1] Canonical 85-KC map & {len(expert_pairs)} Task-3 expert pairs constructed.", flush=True)
        return {
            "train_df": train_merged,
            "pub_df": pub_merged,
            "priv_df": priv_merged,
            "correct_answer_idx": correct_answer_idx,
            "multi_key_qids": multi_key_qids,
            "unique_rep_qids": unique_rep_qids,
            "dup_to_rep": dup_to_rep,
            "dup_pairs": dup_pairs,
            "raw_to_kc": raw_to_kc,
            "kc_to_raw": kc_to_raw,
            "kc_descriptions": kc_descriptions,
            "kc_names_by_qid": kc_names_by_qid,
            "kc_prefix_by_qid": kc_prefix_by_qid,
            "leaf_to_l1": leaf_to_l1,
            "l1_indices": [raw_to_kc[x] for x in sorted_l1],
            "l1_name_to_idx": {str(subj_lookup[x]["Name"]).strip(): raw_to_kc[x] for x in sorted_l1},
            "question_leaf_kcs": question_leaf_kcs,
            "question_kc_mask": question_kc_mask,
            "question_all_kcs": question_all_kcs,
            "question_all_kc_mask": question_all_kc_mask,
            "subj_lookup": subj_lookup,
            "q_raw_leaves": q_raw_leaves,
            "expert_pairs": expert_pairs,
        }

    # ------------------------------------------------------------------
    # Step 2: CV Slide Anatomy, Badges & Clean RoI Cropping
    # ------------------------------------------------------------------
    def _ensure_ocr_bboxes_jsonl(self, ocr_path: Path) -> Dict[int, List[Dict[str, Any]]]:
        """
        Load precomputed Apple Vision / EasyOCR bounding boxes from JSONL if all 948
        slides are cached; otherwise run cross-platform GPU/CPU OCR (EasyOCR) across
        upscaled Eedi slides and checkpoint line-by-line so Colab/Linux runs work strictly
        from raw images.
        """
        ocr_items_by_qid: Dict[int, List[Dict[str, Any]]] = {}
        if ocr_path.exists():
            with open(ocr_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        ocr_items_by_qid[int(rec["qid"])] = rec.get("items", [])

        remaining = [q for q in range(NUM_QUESTIONS) if q not in ocr_items_by_qid]
        if not remaining:
            return ocr_items_by_qid

        print(
            f"[Step 2 OCR BBoxes] Generating cross-platform OCR bboxes for {len(remaining)} slides "
            f"(cached={len(ocr_items_by_qid)})...",
            flush=True,
        )
        reader = None
        rapid_ocr = None
        try:
            import easyocr  # type: ignore

            reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore

                rapid_ocr = RapidOCR()
            except ImportError as exc:
                raise RuntimeError(
                    f"Missing OCR bbox cache at {ocr_path} and neither 'easyocr' nor "
                    "'rapidocr_onnxruntime' is installed. Install via `pip install easyocr`."
                ) from exc

        ocr_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ocr_path, "a", encoding="utf-8") as fout:
            for idx, qid in enumerate(remaining, start=1):
                im = Image.open(self.images_dir / f"{qid}.jpg").convert("RGB")
                # Upscale 320x240 to 1280x960 and 640x480 by 2x for crisp text detection
                target_w = 1280 if im.width < 640 else im.width * 2
                target_h = 960 if im.width < 640 else im.height * 2
                im_up = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
                arr_up = np.asarray(im_up)

                if reader is not None:
                    raw_dets = reader.readtext(arr_up, detail=1, paragraph=False)
                else:
                    raw_res, _ = rapid_ocr(arr_up)
                    raw_dets = raw_res or []

                items: List[Dict[str, Any]] = []
                for poly, txt, conf in raw_dets:
                    xs = [float(pt[0]) for pt in poly]
                    ys = [float(pt[1]) for pt in poly]
                    x0, x1 = max(0.0, min(xs)), min(float(target_w), max(xs))
                    y0, y1 = max(0.0, min(ys)), min(float(target_h), max(ys))
                    items.append({
                        "text": str(txt),
                        "conf": float(conf),
                        "bbox": [
                            x0 / float(target_w),
                            y0 / float(target_h),
                            max(0.0, x1 - x0) / float(target_w),
                            max(0.0, y1 - y0) / float(target_h),
                        ],
                    })

                rec = {"qid": qid, "items": items}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                ocr_items_by_qid[qid] = items
                if idx % 100 == 0 or idx == len(remaining):
                    print(f"[Step 2 OCR BBoxes] {idx}/{len(remaining)} processed.", flush=True)

        return ocr_items_by_qid

    def run_step2_cv_slide_parsing(self) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
        """
        Load or extract OCR bounding boxes (Apple Vision on macOS or EasyOCR on Colab/Linux)
        and execute Step 2 CV slide standardization, calculator detection, badge localization,
        and RoI extraction.
        """
        av_path = self.feat_dir / "apple_vision_ocr.jsonl"
        ocr_items_by_qid = self._ensure_ocr_bboxes_jsonl(av_path)

        inv_path = self.feat_dir / "layout_inventory.json"
        rois_dir = self.feat_dir / "rois"
        if inv_path.exists() and (rois_dir / "947_stem.png").exists():
            with open(inv_path, "r", encoding="utf-8") as f:
                raw_inv = json.load(f)
            if "0" in raw_inv and "stem_sat_density" in raw_inv["0"]:
                print(f"[Step 2] Loading cached CV layout inventory from {inv_path}", flush=True)
                return ocr_items_by_qid, {int(k): v for k, v in raw_inv.items()}
            print(f"[Step 2] Refreshing layout inventory with color-saturation metrics...", flush=True)

        print("[Step 2] Running CV badge detection and RoI extraction across 948 slides...", flush=True)
        ref_tpl = extract_no_calc_template(self.images_dir / "0.jpg")
        inventory: Dict[int, Dict[str, Any]] = {}
        for qid in range(NUM_QUESTIONS):
            inventory[qid] = extract_slide_rois_and_metadata(
                qid=qid,
                img_path=self.images_dir / f"{qid}.jpg",
                ocr_items=ocr_items_by_qid.get(qid, []),
                rois_out_dir=rois_dir,
                ref_calc_template=ref_tpl,
            )
        with open(inv_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in inventory.items()}, f, indent=2)
        return ocr_items_by_qid, inventory

    # ------------------------------------------------------------------
    # Step 5: Stem Visual Modality Masks & 224x224 Letterbox Patch Grid
    # ------------------------------------------------------------------
    @staticmethod
    def _compose_stem_with_visual_options(
        stem_path: str,
        opt_paths: Dict[str, str],
        gap: int = 8,
    ) -> Image.Image:
        """
        Compose clean badge-free RoI_stem above a 2x2 grid of visual option diagrams
        on a pure white RGB canvas prior to geometry-preserving 224x224 letterboxing.
        """
        stem_im = Image.open(stem_path).convert("RGB")
        opt_ims = {k: Image.open(opt_paths[k]).convert("RGB") for k in "ABCD"}
        row1_h = max(opt_ims["A"].height, opt_ims["B"].height)
        row2_h = max(opt_ims["C"].height, opt_ims["D"].height)
        col1_w = max(opt_ims["A"].width, opt_ims["C"].width)
        col2_w = max(opt_ims["B"].width, opt_ims["D"].width)
        grid_w = col1_w + gap + col2_w
        canvas_w = max(stem_im.width, grid_w)
        canvas_h = stem_im.height + gap + row1_h + gap + row2_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(stem_im, ((canvas_w - stem_im.width) // 2, 0))

        grid_x0 = (canvas_w - grid_w) // 2
        y_r1 = stem_im.height + gap
        y_r2 = y_r1 + row1_h + gap
        canvas.paste(opt_ims["A"], (grid_x0, y_r1))
        canvas.paste(opt_ims["B"], (grid_x0 + col1_w + gap, y_r1))
        canvas.paste(opt_ims["C"], (grid_x0, y_r2))
        canvas.paste(opt_ims["D"], (grid_x0 + col1_w + gap, y_r2))
        return canvas

    def run_step5_letterbox_and_masks(
        self,
        layout_inventory: Dict[int, Dict[str, Any]],
        ocr_results: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute final stem visual modality mask m_img^q, save 224x224 white-letterboxed
        RoI_stem (or composite stem + visual option diagrams), and compute 7x7 spatial
        patch validity tensors M_patch_q.
        """
        print("[Step 5] Computing modality masks and 224x224 letterboxed stem RoIs...", flush=True)
        stem_224_dir = self.feat_dir / "stem_224"
        stem_224_dir.mkdir(parents=True, exist_ok=True)

        modality_masks = torch.zeros((NUM_QUESTIONS, 3), dtype=torch.float32)
        m_patch_q = torch.zeros((NUM_QUESTIONS, NUM_PATCHES), dtype=torch.float32)
        patch_valid_mask = torch.ones((NUM_QUESTIONS, NUM_PATCHES), dtype=torch.bool)
        stem_224_images: List[Image.Image] = []
        image_meta_q: List[Dict[str, Any]] = []
        image_meta_s: List[Dict[str, Any]] = []

        for qid in range(NUM_QUESTIONS):
            has_diagram = bool(ocr_results[qid]["has_stem_diagram"])
            modality_masks[qid, 0] = 1.0 if has_diagram else 0.0
            modality_masks[qid, 1] = 1.0  # 100% text-anchored solution T_s^q
            modality_masks[qid, 2] = 0.0  # No authentic solution diagrams on Eedi

            lmeta = layout_inventory[qid]
            roi_stem_path = lmeta["roi_stem_path"]
            opts_norm = ocr_results[qid].get("options", {})
            opt_cv_count = sum(1 for v in lmeta.get("opt_cv_visual", {}).values() if v)
            has_vis_opts = bool(
                sum(1 for v in opts_norm.values() if str(v).startswith("[VISUAL_DIAGRAM_OPTION_")) >= 2
                or (ocr_results[qid].get("options_have_diagrams", False) and opt_cv_count >= 2)
            )

            if has_diagram and has_vis_opts and lmeta["layout_type"] not in ("horiz_ref", "grid_embedded"):
                vis_input: Any = self._compose_stem_with_visual_options(roi_stem_path, lmeta["roi_opt_paths"])
            else:
                vis_input = roi_stem_path

            lb_img, m_patch, p_valid, lb_bbox = letterbox_image_and_mask(
                vis_input, target_size=224, grid_size=7
            )
            lb_out = stem_224_dir / f"{qid}.png"
            lb_img.save(lb_out)
            stem_224_images.append(lb_img)

            # On text-only items (m_img^q == 0), set safe default all-valid patch mask (matching XES3G5M)
            if has_diagram:
                m_patch_q[qid] = m_patch
                patch_valid_mask[qid] = p_valid
            else:
                m_patch_q[qid] = torch.zeros(NUM_PATCHES, dtype=torch.float32)
                patch_valid_mask[qid] = torch.ones(NUM_PATCHES, dtype=torch.bool)

            image_meta_q.append({
                "qid": qid,
                "has_image": has_diagram,
                "options_have_diagrams": has_vis_opts,
                "layout_type": lmeta["layout_type"],
                "calculator_status": lmeta["calculator_status"],
                "calculator_allowed_flag": float(lmeta["calculator_allowed_flag"]),
                "stem_struct_density": float(lmeta["stem_struct_density"]),
                "stem_bbox_640x480": lmeta["stem_bbox"],
                "letterbox_bbox_224x224": list(lb_bbox),
                "roi_stem_path": roi_stem_path,
                "stem_224_path": str(lb_out),
            })
            image_meta_s.append({"qid": qid, "has_image": False, "letterbox_bbox": None})

        n_diag = int(modality_masks[:, 0].sum().item())
        print(f"[Step 5] Complete: m_img^q=1 on {n_diag}/{NUM_QUESTIONS} ({100*n_diag/NUM_QUESTIONS:.1f}%) questions.", flush=True)
        return {
            "modality_masks": modality_masks,
            "M_patch_q": m_patch_q,
            "patch_valid_mask": patch_valid_mask,
            "stem_224_images": stem_224_images,
            "image_meta_q": image_meta_q,
            "image_meta_s": image_meta_s,
        }

    # ------------------------------------------------------------------
    # Step 6 & Step 7: Frozen RoBERTa / CLIP Feature Extraction & Hypergraph
    # ------------------------------------------------------------------
    def run_step6_and_7_encoders_and_hypergraph(
        self,
        step1: Dict[str, Any],
        layout_inventory: Dict[int, Dict[str, Any]],
        ocr_results: Dict[int, Dict[str, Any]],
        sol_results: Dict[int, Dict[str, Any]],
        step5: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract frozen RoBERTa-base token/global embeddings (questions, solutions,
        options, and 85 hierarchy KC texts), extract frozen CLIP ViT-B/16 spatial
        7x7 patches + option embeddings, verify 1-to-1 token alignment, and build
        hierarchical hypergraph incidence H and normalized co-occurrence G.
        """
        print("[Step 6 & 7] Extracting frozen RoBERTa-base text & KC features...", flush=True)
        texts_q = [ocr_results[q]["text_q"] for q in range(NUM_QUESTIONS)]
        texts_s = [sol_results[q]["text_s"] for q in range(NUM_QUESTIONS)]
        opt_texts_flat: List[str] = []
        opt_is_visual_flat: List[bool] = []
        for q in range(NUM_QUESTIONS):
            opts = ocr_results[q]["options"]
            cv_vis = layout_inventory[q]["opt_cv_visual"]
            all_diag = bool(ocr_results[q]["options_have_diagrams"])
            for ltr in "ABCD":
                t = opts[ltr]
                opt_texts_flat.append(t)
                opt_is_visual_flat.append(
                    all_diag or t.startswith("[VISUAL_DIAGRAM_OPTION_") or bool(cv_vis.get(ltr, False))
                )

        txt_enc = FrozenTextEncoder(model_name=self.cfg["text"]["model_name"], local_files_only=True)
        E_txt_q, mask_txt_q, v_txt_q, tok_q = txt_enc.encode_texts(texts_q, max_length=MAX_LEN_Q, batch_size=64)
        E_txt_s, mask_txt_s, v_txt_s, tok_s = txt_enc.encode_texts(texts_s, max_length=MAX_LEN_S, batch_size=64)
        _, _, opt_txt_vecs, _ = txt_enc.encode_texts(opt_texts_flat, max_length=48, batch_size=128)
        _, _, kc_init_vecs, _ = txt_enc.encode_texts(step1["kc_descriptions"], max_length=64, batch_size=64)
        acd_kc_init_emb = kc_init_vecs.float()  # [85, 768] float32

        # Strictly verify 1-to-1 subword token string alignment against mask_txt_q/s
        for q in range(NUM_QUESTIONS):
            active_q = sum(1 for tk in tok_q[q] if tk != "<pad>")
            if len(tok_q[q]) != MAX_LEN_Q or active_q != int(mask_txt_q[q].sum().item()):
                raise AssertionError(f"1-to-1 token_to_word_q alignment mismatch at q={q}")
            active_s = sum(1 for tk in tok_s[q] if tk != "<pad>")
            if len(tok_s[q]) != MAX_LEN_S or active_s != int(mask_txt_s[q].sum().item()):
                raise AssertionError(f"1-to-1 token_to_word_s alignment mismatch at q={q}")

        txt_enc.release()

        print("[Step 6] Extracting frozen CLIP ViT-B/16 stem 7x7 patches & option crops...", flush=True)
        vis_enc = FrozenVisionEncoder(model_name=self.cfg["vision"]["model_name"], local_files_only=True)
        patch_q, _, pooled_img_q = vis_enc.encode_images(
            step5["stem_224_images"],
            patch_valid_masks=step5["patch_valid_mask"],
            batch_size=64,
        )

        # Enforce clean zero tensors on text-only stems (m_img^q == 0)
        m_img_bool = step5["modality_masks"][:, 0] > 0.5
        acd_E_img_q = torch.zeros_like(patch_q)
        mg_v_img_q = torch.zeros_like(pooled_img_q)
        acd_E_img_q[m_img_bool] = patch_q[m_img_bool]
        mg_v_img_q[m_img_bool] = pooled_img_q[m_img_bool]

        # Solution visuals strictly absent on Eedi (m_img^s == 0)
        acd_E_img_s = torch.zeros((NUM_QUESTIONS, 768), dtype=torch.float16)
        mg_v_img_s = torch.zeros((NUM_QUESTIONS, 768), dtype=torch.float16)

        # Encode all 948*4 = 3,792 letterboxed option RoIs for option_multimodal_emb
        opt_lb_imgs: List[Image.Image] = []
        opt_lb_masks: List[torch.Tensor] = []
        for q in range(NUM_QUESTIONS):
            for ltr in "ABCD":
                opath = layout_inventory[q]["roi_opt_paths"][ltr]
                lb_o, _, p_val_o, _ = letterbox_image_and_mask(opath, target_size=224, grid_size=7)
                opt_lb_imgs.append(lb_o)
                opt_lb_masks.append(p_val_o)
        opt_masks_tensor = torch.stack(opt_lb_masks, dim=0)
        _, _, opt_img_vecs = vis_enc.encode_images(opt_lb_imgs, patch_valid_masks=opt_masks_tensor, batch_size=64)
        vis_enc.release()

        # Construct option_multimodal_emb [948, 4, 768] float16
        # Pure text options retain clean RoBERTa embedding; visual options blend norm-matched CLIP + text
        opt_txt_f = opt_txt_vecs.float()
        opt_img_f = opt_img_vecs.float()
        vis_flag = torch.tensor(opt_is_visual_flat, dtype=torch.bool).unsqueeze(-1)
        txt_norm = opt_txt_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        img_norm = opt_img_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        opt_img_scaled = opt_img_f * (txt_norm / img_norm)
        fused_opt = torch.where(vis_flag, 0.5 * opt_txt_f + 0.5 * opt_img_scaled, opt_txt_f)
        option_multimodal_emb = fused_opt.view(NUM_QUESTIONS, 4, 768).to(torch.float16)

        # Step 7: Build Hierarchical Hypergraph H [85, 948] & Symmetric Normalized G [85, 85]
        alpha_l3 = float(self.cfg["hypergraph"]["alpha_l3"])
        alpha_l2 = float(self.cfg["hypergraph"]["alpha_l2"])
        alpha_l1 = float(self.cfg["hypergraph"]["alpha_l1"])
        eps = float(self.cfg["hypergraph"]["eps"])

        H = torch.zeros((85, NUM_QUESTIONS), dtype=torch.float32)
        raw_to_kc = step1["raw_to_kc"]
        subj_lookup = step1["subj_lookup"]
        for qid in range(NUM_QUESTIONS):
            for leaf_raw in step1["q_raw_leaves"][qid]:
                H[raw_to_kc[leaf_raw], qid] = alpha_l3
                p2 = int(subj_lookup[leaf_raw]["ParentId"])
                H[raw_to_kc[p2], qid] = alpha_l2
                p1 = int(subj_lookup[p2]["ParentId"])
                H[raw_to_kc[p1], qid] = alpha_l1

        dv = H.sum(dim=1)
        de = H.sum(dim=0)
        if not bool((dv > 0).all()):
            raise AssertionError("Found isolated zero-degree KC in Eedi hierarchical hypergraph!")
        dv_inv_sqrt = torch.pow(dv.clamp(min=eps), -0.5)
        de_inv = 1.0 / de.clamp(min=eps)
        H_norm = H * de_inv.unsqueeze(0)
        G = (dv_inv_sqrt.unsqueeze(1) * H_norm) @ H.t() * dv_inv_sqrt.unsqueeze(0)
        G = 0.5 * (G + G.t())  # Guarantee exact IEEE symmetry

        print("[Step 6 & 7] Encoders & Hierarchical Hypergraph (85x948) verified cleanly.", flush=True)
        return {
            "acd_E_txt_q": E_txt_q,
            "acd_mask_txt_q": mask_txt_q,
            "acd_E_txt_s": E_txt_s,
            "acd_mask_txt_s": mask_txt_s,
            "acd_E_img_q": acd_E_img_q,
            "acd_E_img_s": acd_E_img_s,
            "mg_v_txt_q": v_txt_q,
            "mg_v_txt_s": v_txt_s,
            "mg_v_img_q": mg_v_img_q,
            "mg_v_img_s": mg_v_img_s,
            "option_multimodal_emb": option_multimodal_emb,
            "acd_kc_init_emb": acd_kc_init_emb,
            "acd_hypergraph_H": H,
            "acd_hypergraph_G": G,
            "token_to_word_q": tok_q,
            "token_to_word_s": tok_s,
        }

    # ------------------------------------------------------------------
    # Step 8: 4-Level Chronological Sort & Stratified 5-Fold Student Split
    # ------------------------------------------------------------------
    def _create_stratified_5fold_split(self, train_df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Stratify all 4,918 training students into 5 balanced folds (seed 42) across
        (length_quartile, accuracy_quartile), then perform rare-question hill-climbing
        so every question (even count==4 rare items) maximizes per-fold train coverage.
        """
        split_file = self.splits_dir / "train_val_folds.json"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if sum(len(v) for v in saved.values()) == train_df["UserId"].nunique():
                print(f"[Step 8] Loaded existing 5-fold split from {split_file}", flush=True)
                return saved

        rng = random.Random(self.seed)
        user_stats = (
            train_df.groupby("UserId")
            .agg(seq_len=("QuestionId", "count"), acc=("IsCorrect", "mean"))
            .reset_index()
        )
        user_stats["len_q"] = pd.qcut(user_stats["seq_len"].rank(method="first"), 4, labels=False)
        user_stats["acc_q"] = pd.qcut(user_stats["acc"].rank(method="first"), 4, labels=False)
        user_stats["stratum"] = user_stats["len_q"].astype(str) + "_" + user_stats["acc_q"].astype(str)

        uid_to_fold: Dict[int, int] = {}
        uid_to_stratum: Dict[int, str] = {}
        folds: List[List[int]] = [[] for _ in range(5)]

        # Deal stratified shuffled blocks round-robin into folds 0..4
        stratum_groups = sorted(user_stats.groupby("stratum"), key=lambda kv: kv[0])
        offset = 0
        for _, grp in stratum_groups:
            uids = grp["UserId"].astype(int).tolist()
            rng.shuffle(uids)
            for u in uids:
                f = offset % 5
                uid_to_fold[u] = f
                uid_to_stratum[u] = str(grp["stratum"].iloc[0])
                folds[f].append(u)
                offset += 1

        # Hill-climb swaps specifically for rare questions (<= 15 total occurrences)
        user_q_sets = train_df.groupby("UserId")["QuestionId"].apply(lambda s: set(int(x) for x in s)).to_dict()
        q_counts = train_df["QuestionId"].value_counts()
        rare_qids = [int(q) for q, c in q_counts.items() if c <= 15]
        rare_q_set = set(rare_qids)

        q_fold_counts = np.zeros((NUM_QUESTIONS, 5), dtype=np.int64)
        for u, f in uid_to_fold.items():
            for q in user_q_sets[u]:
                q_fold_counts[q, f] += 1

        # Protect users touched by very rare questions from unwanted displacement
        locked_rare_users: Set[int] = set()
        for q in sorted(rare_qids, key=lambda x: int(q_counts.loc[x])):
            holders = [u for u, qs in user_q_sets.items() if q in qs]
            # Spread holders across distinct folds
            for f_idx in range(5):
                if q_fold_counts[q, f_idx] > 0:
                    continue
                # Find fold with >1 holder of q
                donor_folds = [df_i for df_i in range(5) if q_fold_counts[q, df_i] > 1]
                if not donor_folds:
                    continue
                donor_f = max(donor_folds, key=lambda x: int(q_fold_counts[q, x]))
                donor_candidates = [u for u in holders if uid_to_fold[u] == donor_f and u not in locked_rare_users]
                if not donor_candidates:
                    donor_candidates = [u for u in holders if uid_to_fold[u] == donor_f]
                if not donor_candidates:
                    continue
                u_donor = donor_candidates[0]
                # Prefer same-stratum receiver in f_idx that holds zero rare questions
                receivers = [
                    u for u in folds[f_idx]
                    if uid_to_stratum[u] == uid_to_stratum[u_donor]
                    and u not in locked_rare_users
                    and not (user_q_sets[u] & rare_q_set)
                ]
                if not receivers:
                    receivers = [
                        u for u in folds[f_idx]
                        if u not in locked_rare_users and not (user_q_sets[u] & rare_q_set)
                    ]
                if not receivers:
                    continue
                u_recv = receivers[0]
                # Swap u_donor (donor_f) <-> u_recv (f_idx)
                uid_to_fold[u_donor] = f_idx
                uid_to_fold[u_recv] = donor_f
                folds[donor_f].remove(u_donor)
                folds[donor_f].append(u_recv)
                folds[f_idx].remove(u_recv)
                folds[f_idx].append(u_donor)
                for qq in user_q_sets[u_donor]:
                    q_fold_counts[qq, donor_f] -= 1
                    q_fold_counts[qq, f_idx] += 1
                for qq in user_q_sets[u_recv]:
                    q_fold_counts[qq, f_idx] -= 1
                    q_fold_counts[qq, donor_f] += 1
            for h in holders:
                if q_fold_counts[q, uid_to_fold[h]] == 1:
                    locked_rare_users.add(h)

        split_dict = {f"fold_{f}": sorted(folds[f]) for f in range(5)}
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2)
        return split_dict

    # ------------------------------------------------------------------
    # Step 9: Leak-Free Hierarchical Empirical Bayes Priors & Distractors
    # ------------------------------------------------------------------
    def run_step9_fold_priors(
        self,
        train_df: pd.DataFrame,
        split_dict: Dict[str, List[int]],
        step1: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute strictly fold-isolated Hierarchical Empirical Bayes CTT failure priors:
          - Level 0: global error rate bar_D_global^(f)
          - Level 1: L1 discipline smoothed error rates bar_D_L1^(f)
          - Level 2: leaf KC smoothed error rates bar_D_KC^(f)(k)
          - Question skill prior mu_q^(prior, f) in (1e-4, 1 - 1e-4)
          - Fixed logit concept anchor mu_concept_q^(f) = logit(mu_q^(prior, f)) in R
          - Item confidence omega_q^(f) = N_q / (N_q + kappa)
          - Lightly shrunk CTT failure rate D_q^(f) in (1e-4, 1 - 1e-4) with m_item = 1.0
        Plus Dirichlet-smoothed distractor modes and entropy strictly on D_train^(f).
        """
        print("[Step 9] Computing strictly leak-free per-fold CTT failure priors & distractors...", flush=True)
        q_leaf_kcs = step1["question_leaf_kcs"]
        q_kc_mask = step1["question_kc_mask"]
        q_all_kcs = step1["question_all_kcs"]
        q_all_mask = step1["question_all_kc_mask"]
        leaf_to_l1 = step1["leaf_to_l1"]
        l1_indices = step1["l1_indices"]
        c_idx = step1["correct_answer_idx"]

        uid_to_fold = {u: int(k.split("_")[1]) for k, uids in split_dict.items() for u in uids}
        train_f = train_df.copy()
        train_f["fold"] = train_f["UserId"].map(uid_to_fold)

        n_q_folds = torch.zeros((5, NUM_QUESTIONS), dtype=torch.int64)
        mu_concept_folds = torch.zeros((5, NUM_QUESTIONS), dtype=torch.float32)
        mu_prior_folds = torch.zeros((5, NUM_QUESTIONS), dtype=torch.float32)
        omega_q_folds = torch.zeros((5, NUM_QUESTIONS), dtype=torch.float32)
        d_q_folds = torch.zeros((5, NUM_QUESTIONS), dtype=torch.float32)
        fold_distractor_entropy = torch.zeros((5, NUM_QUESTIONS), dtype=torch.float32)
        fold_dominant_distractor = torch.zeros((5, NUM_QUESTIONS), dtype=torch.int64)

        log3 = math.log(3.0)
        for val_f in range(5):
            df_tr = train_f[train_f["fold"] != val_f]
            err_global = 1.0 - float(df_tr["IsCorrect"].mean())

            q_arr = df_tr["QuestionId"].to_numpy(dtype=np.int64)
            r_arr = df_tr["IsCorrect"].to_numpy(dtype=np.int64)
            err_arr = 1.0 - r_arr.astype(np.float64)
            a_arr = df_tr["AnswerValue"].to_numpy(dtype=np.int64) - 1  # 0..3

            n_q = np.bincount(q_arr, minlength=NUM_QUESTIONS)
            e_q = np.bincount(q_arr, weights=err_arr, minlength=NUM_QUESTIONS)

            # Tier 1: L1 discipline failure rates smoothed toward global failure rate
            d_l1: Dict[int, float] = {}
            for l1_idx in l1_indices:
                has_l1 = ((q_all_kcs == l1_idx) & q_all_mask).any(dim=1).numpy()
                n_l1 = float(n_q[has_l1].sum())
                e_l1 = float(e_q[has_l1].sum())
                d_l1[l1_idx] = (e_l1 + M_KC * err_global) / (n_l1 + M_KC)

            # Tier 2: Leaf KC failure rates smoothed toward parent L1 discipline rate (m_kc = 10.0)
            bar_d_kc = np.zeros(57, dtype=np.float64)
            for kc in range(57):
                mask_kc = (q_leaf_kcs == kc) & q_kc_mask
                has_kc = mask_kc.any(dim=1).numpy()
                n_kc_val = float(n_q[has_kc].sum())
                e_kc_val = float(e_q[has_kc].sum())
                l1_parent_err = d_l1[leaf_to_l1[kc]]
                bar_d_kc[kc] = (e_kc_val + M_KC * l1_parent_err) / (n_kc_val + M_KC)

            # Distractor counts [948, 4] excluding official latest key c_q
            dist_counts = np.zeros((NUM_QUESTIONS, 4), dtype=np.float64)
            wrong_mask = r_arr == 0
            wq = q_arr[wrong_mask]
            wa = a_arr[wrong_mask]
            np.add.at(dist_counts, (wq, wa), 1.0)

            for q in range(NUM_QUESTIONS):
                leaves_q = q_leaf_kcs[q][q_kc_mask[q]].tolist()
                mu_p = float(np.clip(np.mean([bar_d_kc[c] for c in leaves_q]), 1e-4, 1.0 - 1e-4))
                mu_c = float(math.log(mu_p / (1.0 - mu_p)))

                nq_val = int(n_q[q])
                eq_val = float(e_q[q])
                omega = float(nq_val) / (float(nq_val) + KAPPA)
                d_val = float(np.clip((eq_val + M_ITEM * mu_p) / (float(nq_val) + M_ITEM), 1e-4, 1.0 - 1e-4))

                n_q_folds[val_f, q] = nq_val
                mu_prior_folds[val_f, q] = mu_p
                mu_concept_folds[val_f, q] = mu_c
                omega_q_folds[val_f, q] = omega
                d_q_folds[val_f, q] = d_val

                cq = int(c_idx[q].item())
                wrong_opts = [opt for opt in range(4) if opt != cq]
                cnts = np.array([dist_counts[q, opt] for opt in wrong_opts], dtype=np.float64)
                probs = (cnts + DIRICHLET_ALPHA) / (cnts.sum() + 3.0 * DIRICHLET_ALPHA)
                entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))) / log3)
                dom_local = int(np.argmax(probs))
                fold_distractor_entropy[val_f, q] = entropy
                fold_dominant_distractor[val_f, q] = wrong_opts[dom_local]

        min_nq = int(n_q_folds.min().item())
        print(
            f"[Step 9] Fold CTT & Dirichlet distractors built. min N_q={min_nq} | "
            f"mean mu_prior={mu_prior_folds.mean().item():.4f} | mean mu_concept(logit)={mu_concept_folds.mean().item():.4f}",
            flush=True,
        )
        return {
            "N_q_folds": n_q_folds,
            "mu_concept_folds": mu_concept_folds,
            "mu_prior_folds": mu_prior_folds,
            "omega_q_folds": omega_q_folds,
            "D_q_folds": d_q_folds,
            "fold_distractor_entropy": fold_distractor_entropy,
            "fold_dominant_distractor": fold_dominant_distractor,
        }

    # ------------------------------------------------------------------
    # Step 8 & Step 10: Chronological Temporal Features & Strided Windowing
    # ------------------------------------------------------------------
    def _build_split_sequences(
        self,
        df: pd.DataFrame,
        step1: Dict[str, Any],
        fold_dom_distractor: torch.Tensor,
        has_targets: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a sorted dataframe of users into strided context-preserving
        fixed-length [N_seq, 200] trajectory windows with causal temporal dynamics
        and fold-aligned [5, N_seq, 200, 8] post-response diagnostic vectors.
        """
        df_sorted = df.sort_values(["UserId", "DateAnswered_sec", "QuizId", "AnswerId"]).reset_index(drop=True)
        q_leaf_kcs = step1["question_leaf_kcs"]
        q_kc_mask = step1["question_kc_mask"]
        q_to_leaves = [
            q_leaf_kcs[q][q_kc_mask[q]].tolist() for q in range(NUM_QUESTIONS)
        ]
        dom_dist_np = fold_dom_distractor.numpy()  # [5, 948]

        target_cols = [f"IsTarget_{k}" for k in range(10)] if has_targets else []

        out_uid: List[int] = []
        out_ctx_start: List[int] = []
        out_q: List[torch.Tensor] = []
        out_r: List[torch.Tensor] = []
        out_ans: List[torch.Tensor] = []
        out_seq_mask: List[torch.Tensor] = []
        out_loss_mask: List[torch.Tensor] = []
        out_eval_mask: List[torch.Tensor] = []
        out_dt_lag: List[torch.Tensor] = []
        out_dt_kc: List[torch.Tensor] = []
        out_norm_dt_lag: List[torch.Tensor] = []
        out_norm_dt_kc: List[torch.Tensor] = []
        out_kc_first: List[torch.Tensor] = []
        out_batch: List[torch.Tensor] = []
        out_run_acc: List[torch.Tensor] = []
        out_rel_qpos: List[torch.Tensor] = []
        out_post_diag: List[torch.Tensor] = []
        out_targets: List[torch.Tensor] = []

        for uid, u_df in df_sorted.groupby("UserId", sort=True):
            T = len(u_df)
            q_arr = u_df["QuestionId"].to_numpy(dtype=np.int64, copy=True)
            r_arr = u_df["IsCorrect"].to_numpy(dtype=np.int64, copy=True)
            ans_arr = u_df["AnswerValue"].to_numpy(dtype=np.int64, copy=True) - 1  # 0..3
            ts_arr = u_df["DateAnswered_sec"].to_numpy(dtype=np.int64, copy=True)
            quiz_arr = u_df["QuizId"].to_numpy(dtype=np.int64, copy=True)
            conf_arr = u_df["Confidence"].to_numpy(dtype=np.float64, copy=True)

            dt_lag = np.zeros(T, dtype=np.float32)
            dt_kc = np.full(T, -1.0, dtype=np.float32)
            norm_dt_lag = np.zeros(T, dtype=np.float32)
            norm_dt_kc = np.zeros(T, dtype=np.float32)
            kc_first = np.ones(T, dtype=np.float32)
            batch_f = np.zeros(T, dtype=np.float32)
            run_acc = np.zeros(T, dtype=np.float32)
            rel_qpos = np.zeros(T, dtype=np.float32)

            # Count quiz lengths and per-step ordinal index within quiz
            quiz_counts = Counter(quiz_arr.tolist())
            quiz_seen: Dict[int, int] = defaultdict(int)
            last_kc_ts: Dict[int, int] = {}
            correct_sum = 0

            post_diag_user = np.zeros((5, T, 8), dtype=np.float32)

            for t in range(T):
                q_t = int(q_arr[t])
                r_t = int(r_arr[t])
                ans_t = int(ans_arr[t])
                ts_t = int(ts_arr[t])
                qz_t = int(quiz_arr[t])

                if t == 0:
                    run_acc[t] = GLOBAL_ACC_PRIOR
                else:
                    diff_s = max(0, ts_t - int(ts_arr[t - 1]))
                    val_log = math.log1p(float(diff_s))
                    dt_lag[t] = val_log
                    norm_dt_lag[t] = min(1.0, max(0.0, val_log / LOG_MAX_LAG))
                    if qz_t == int(quiz_arr[t - 1]) or diff_s == 0:
                        batch_f[t] = 1.0
                    # Causal Laplace-smoothed running accuracy up to step t - 1
                    run_acc[t] = (float(correct_sum) + 2.0 * GLOBAL_ACC_PRIOR) / (float(t) + 2.0)

                # Leaf KC recurrence tracking
                leaves = q_to_leaves[q_t]
                prev_ts_list = [last_kc_ts[c] for c in leaves if c in last_kc_ts]
                if prev_ts_list:
                    min_dt_kc = max(0, ts_t - max(prev_ts_list))
                    val_kc_log = math.log1p(float(min_dt_kc))
                    dt_kc[t] = val_kc_log
                    norm_dt_kc[t] = min(1.0, max(0.0, val_kc_log / LOG_MAX_LAG))
                    kc_first[t] = 0.0
                for c in leaves:
                    last_kc_ts[c] = ts_t

                qlen = quiz_counts[qz_t]
                qpos = quiz_seen[qz_t]
                quiz_seen[qz_t] += 1
                rel_qpos[t] = float(qpos) / float(max(1, qlen - 1)) if qlen > 1 else 0.0

                # 8-dim post-response diagnostic vector across all 5 fold views (Step 10.2):
                # [0..3]: one-hot selected option a_tau in {A,B,C,D}
                # [4]: is_dom_misconception^(f)
                # [5]: has_conf
                # [6]: conf_norm in [0, 1]
                # [7]: is_overconfident
                c_raw = conf_arr[t]
                has_c = 0.0 if np.isnan(c_raw) else 1.0
                norm_c = 0.0 if np.isnan(c_raw) else float(np.clip(float(c_raw) / 100.0, 0.0, 1.0))
                overconf_err = 1.0 if (r_t == 0 and has_c > 0.5 and norm_c >= 0.75) else 0.0

                post_diag_user[:, t, ans_t] = 1.0
                if r_t == 0:
                    dom_vec = (dom_dist_np[:, q_t] == ans_t).astype(np.float32)
                    post_diag_user[:, t, 4] = dom_vec
                post_diag_user[:, t, 5] = has_c
                post_diag_user[:, t, 6] = norm_c
                post_diag_user[:, t, 7] = overconf_err

                correct_sum += r_t

            t_mat = u_df[target_cols].to_numpy(dtype=bool) if has_targets else None

            # Emit context-preserving strided windows (L=200, S=100) so NO interactions are dropped
            windows: List[Tuple[int, int, int]] = []  # (ctx_start, eval_start_global, end_global)
            w0_end = min(MAX_SEQ_LEN, T)
            windows.append((0, 1, w0_end))
            cur_eval_start = MAX_SEQ_LEN
            while cur_eval_start < T:
                cur_end = min(cur_eval_start + TEST_STRIDE, T)
                ctx_start = max(0, cur_end - MAX_SEQ_LEN)
                windows.append((ctx_start, cur_eval_start, cur_end))
                cur_eval_start = cur_end

            for ctx_s, eval_s, g_end in windows:
                s_len = g_end - ctx_s
                q_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.int64)
                r_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.int64)
                ans_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.int64)
                sm_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
                em_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
                dtl_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
                dtk_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
                ndtl_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
                ndtk_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
                kcf_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
                bf_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
                ra_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
                rq_buf = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
                pd_buf = torch.zeros((5, MAX_SEQ_LEN, 8), dtype=torch.float32)

                q_buf[:s_len] = torch.from_numpy(q_arr[ctx_s:g_end])
                r_buf[:s_len] = torch.from_numpy(r_arr[ctx_s:g_end])
                ans_buf[:s_len] = torch.from_numpy(ans_arr[ctx_s:g_end])
                sm_buf[:s_len] = True
                loc_eval_start = eval_s - ctx_s
                if g_end > eval_s:
                    em_buf[loc_eval_start:s_len] = True
                dtl_buf[:s_len] = torch.from_numpy(dt_lag[ctx_s:g_end])
                dtk_buf[:s_len] = torch.from_numpy(dt_kc[ctx_s:g_end])
                ndtl_buf[:s_len] = torch.from_numpy(norm_dt_lag[ctx_s:g_end])
                ndtk_buf[:s_len] = torch.from_numpy(norm_dt_kc[ctx_s:g_end])
                kcf_buf[:s_len] = torch.from_numpy(kc_first[ctx_s:g_end])
                bf_buf[:s_len] = torch.from_numpy(batch_f[ctx_s:g_end])
                ra_buf[:s_len] = torch.from_numpy(run_acc[ctx_s:g_end])
                rq_buf[:s_len] = torch.from_numpy(rel_qpos[ctx_s:g_end])
                pd_buf[:, :s_len, :] = torch.from_numpy(post_diag_user[:, ctx_s:g_end, :])

                out_uid.append(int(uid))
                out_ctx_start.append(ctx_s)
                out_q.append(q_buf)
                out_r.append(r_buf)
                out_ans.append(ans_buf)
                out_seq_mask.append(sm_buf)
                out_loss_mask.append(em_buf.clone())
                out_eval_mask.append(em_buf)
                out_dt_lag.append(dtl_buf)
                out_dt_kc.append(dtk_buf)
                out_norm_dt_lag.append(ndtl_buf)
                out_norm_dt_kc.append(ndtk_buf)
                out_kc_first.append(kcf_buf)
                out_batch.append(bf_buf)
                out_run_acc.append(ra_buf)
                out_rel_qpos.append(rq_buf)
                out_post_diag.append(pd_buf)

                if has_targets and t_mat is not None:
                    tm_buf = torch.zeros((MAX_SEQ_LEN, 10), dtype=torch.bool)
                    tm_buf[:s_len] = torch.from_numpy(t_mat[ctx_s:g_end])
                    out_targets.append(tm_buf)

        # Stack post_diag_vec into [5, N_seq, 200, 8] for clean zero-leak fold indexing in EediDataset
        result: Dict[str, torch.Tensor] = {
            "uid_seq": torch.tensor(out_uid, dtype=torch.int64),
            "context_start": torch.tensor(out_ctx_start, dtype=torch.int64),
            "q_seq": torch.stack(out_q, dim=0),
            "r_seq": torch.stack(out_r, dim=0),
            "ans_val_seq": torch.stack(out_ans, dim=0),
            "seq_mask": torch.stack(out_seq_mask, dim=0),
            "loss_mask": torch.stack(out_loss_mask, dim=0),
            "eval_mask": torch.stack(out_eval_mask, dim=0),
            "delta_t_lag": torch.stack(out_dt_lag, dim=0),
            "delta_t_kc": torch.stack(out_dt_kc, dim=0),
            "norm_delta_t_lag": torch.stack(out_norm_dt_lag, dim=0),
            "norm_delta_t_kc": torch.stack(out_norm_dt_kc, dim=0),
            "kc_first_flag": torch.stack(out_kc_first, dim=0),
            "batch_flag": torch.stack(out_batch, dim=0),
            "running_acc_prev": torch.stack(out_run_acc, dim=0),
            "relative_quiz_pos": torch.stack(out_rel_qpos, dim=0),
            "post_diag_vec": torch.stack(out_post_diag, dim=1),  # [5, N_seq, 200, 8]
        }
        if has_targets:
            result["is_target_mask"] = torch.stack(out_targets, dim=0)
        return result

    def _build_question_static_ctx(
        self,
        step1: Dict[str, Any],
        layout_inventory: Dict[int, Dict[str, Any]],
        ocr_results: Dict[int, Dict[str, Any]],
        sol_results: Dict[int, Dict[str, Any]],
        step5: Dict[str, Any],
        step67: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Build 15-dimensional static question context tensor [948, 15] float32 in [0, 1]
        matching Step 10.2 & IRT item-parameter invariance:
          [0]: MCQ 4-option format flag (1.0)
          [1..3]: L1 discipline one-hot (Number, Geometry and Measure, Algebra)
          [4..6]: Slide layout archetype one-hot (horiz, grid, vert)
          [7]: Normalized leaf KC count |C_leaf| / 2.0
          [8]: Normalized full hierarchy KC count |C_all| / 5.0
          [9]: Normalized stem text token length |T_q| / L_q
          [10]: Normalized solution text token length |T_s| / L_s
          [11]: Stem diagram indicator m_img^q
          [12]: Structural ink density rho_stem (scaled & clipped to [0, 1])
          [13]: Valid 7x7 spatial patch ratio
          [14]: Calculator allowed flag
        """
        ctx = torch.zeros((NUM_QUESTIONS, 15), dtype=torch.float32)
        all_kcs = step1["question_all_kcs"]
        all_mask = step1["question_all_kc_mask"]
        leaf_mask = step1["question_kc_mask"]
        mod_masks = step5["modality_masks"]
        patch_valid = step5["patch_valid_mask"]
        mask_txt_q = step67["acd_mask_txt_q"]
        mask_txt_s = step67["acd_mask_txt_s"]
        l1_map = step1["l1_name_to_idx"]
        idx_num = l1_map.get("Number", 82)
        idx_geom = l1_map.get("Geometry and Measure", 84)
        idx_alg = l1_map.get("Algebra", 83)

        for q in range(NUM_QUESTIONS):
            lrecord = layout_inventory[q]
            layout = lrecord["layout_type"]
            active_all = set(all_kcs[q][all_mask[q]].tolist())

            ctx[q, 0] = 1.0  # 4-option MCQ format flag
            ctx[q, 1] = 1.0 if idx_num in active_all else 0.0
            ctx[q, 2] = 1.0 if idx_geom in active_all else 0.0
            ctx[q, 3] = 1.0 if idx_alg in active_all else 0.0
            ctx[q, 4] = 1.0 if layout.startswith("horiz") else 0.0
            ctx[q, 5] = 1.0 if layout.startswith("grid") else 0.0
            ctx[q, 6] = 1.0 if layout.startswith("vert") else 0.0
            ctx[q, 7] = min(1.0, float(leaf_mask[q].sum().item()) / 2.0)
            ctx[q, 8] = min(1.0, float(all_mask[q].sum().item()) / 5.0)
            ctx[q, 9] = float(mask_txt_q[q].sum().item()) / float(MAX_LEN_Q)
            ctx[q, 10] = float(mask_txt_s[q].sum().item()) / float(MAX_LEN_S)
            ctx[q, 11] = float(mod_masks[q, 0].item())
            ctx[q, 12] = min(1.0, float(lrecord["stem_struct_density"]) * 10.0)
            ctx[q, 13] = float(patch_valid[q].sum().item()) / float(NUM_PATCHES)
            ctx[q, 14] = float(lrecord["calculator_allowed_flag"])

        return ctx

    def _compute_test_strata_masks(
        self,
        seq_dict: Dict[str, torch.Tensor],
        modality_masks: torch.Tensor,
        q_train_counts: torch.Tensor,
        step1: Dict[str, Any],
        layout_inventory: Dict[int, Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Precompute comprehensive stratified boolean evaluation slice masks strictly inside eval_mask:
          - Visual vs text-only & 3 slide layout archetypes
          - Item frequency / cold-start (<=10, 11..100, >100)
          - Remedial learning (r_{t-1}=0 vs 1, post-dominant-misconception, post-overconfident-error)
          - Skill complexity (single_kc vs multi_kc) & L1 disciplines (Number, Algebra, Geometry)
          - Temporal dynamics & Task 4 target splits
        """
        em = seq_dict["eval_mask"].bool()
        seq_m = seq_dict["seq_mask"].bool()
        q_seq = seq_dict["q_seq"]
        r_seq = seq_dict["r_seq"]
        m_img = modality_masks[q_seq, 0] > 0.5
        q_cnt = q_train_counts[q_seq]
        batch_b = seq_dict["batch_flag"] > 0.5
        kc_first = seq_dict["kc_first_flag"] > 0.5

        # Question-level metadata masks indexed by q_seq
        leaf_cnt_per_q = step1["question_kc_mask"].sum(dim=1)
        single_kc_q = (leaf_cnt_per_q == 1)[q_seq]
        multi_kc_q = (leaf_cnt_per_q > 1)[q_seq]

        all_kcs = step1["question_all_kcs"]
        all_mask = step1["question_all_kc_mask"]
        l1_map = step1["l1_name_to_idx"]
        idx_num = l1_map.get("Number", 82)
        idx_alg = l1_map.get("Algebra", 83)
        idx_geom = l1_map.get("Geometry and Measure", 84)
        q_is_num = ((all_kcs == idx_num) & all_mask).any(dim=1)[q_seq]
        q_is_alg = ((all_kcs == idx_alg) & all_mask).any(dim=1)[q_seq]
        q_is_geom = ((all_kcs == idx_geom) & all_mask).any(dim=1)[q_seq]

        is_horiz_q = torch.tensor(
            [layout_inventory[q]["layout_type"].startswith("horiz") for q in range(NUM_QUESTIONS)],
            dtype=torch.bool,
        )[q_seq]
        is_grid_q = torch.tensor(
            [layout_inventory[q]["layout_type"].startswith("grid") for q in range(NUM_QUESTIONS)],
            dtype=torch.bool,
        )[q_seq]
        is_vert_q = torch.tensor(
            [layout_inventory[q]["layout_type"].startswith("vert") for q in range(NUM_QUESTIONS)],
            dtype=torch.bool,
        )[q_seq]

        # Previous-step remedial learning slices (t - 1 strictly exists for every eval_mask position)
        prev_wrong = torch.zeros_like(em)
        prev_wrong[:, 1:] = (r_seq[:, :-1] == 0) & seq_m[:, :-1]
        prev_correct = torch.zeros_like(em)
        prev_correct[:, 1:] = (r_seq[:, :-1] == 1) & seq_m[:, :-1]

        pd0 = seq_dict["post_diag_vec"][0]  # [N_seq, 200, 8]
        prev_dom_misc = torch.zeros_like(em)
        prev_dom_misc[:, 1:] = (pd0[:, :-1, 4] > 0.5) & seq_m[:, :-1]

        prev_overconf_step = torch.zeros_like(em)
        prev_overconf_step[:, 1:] = (pd0[:, :-1, 7] > 0.5) & seq_m[:, :-1]

        # Cumulative causal history check for whether user previously made an overconfident error (< t)
        overconf = (pd0[:, :, 7] > 0.5) & seq_m
        cum_overconf = torch.cumsum(overconf.to(torch.int32), dim=1)
        prev_overconf_hist = torch.zeros_like(em)
        prev_overconf_hist[:, 1:] = cum_overconf[:, :-1] > 0

        strata: Dict[str, torch.Tensor] = {
            "all_eval": em,
            "visual_diagram": em & m_img,
            "text_only": em & (~m_img),
            "layout_horizontal": em & is_horiz_q,
            "layout_grid": em & is_grid_q,
            "layout_vertical": em & is_vert_q,
            "cold_start_q": em & (q_cnt <= 10),
            "medium_freq_q": em & (q_cnt > 10) & (q_cnt <= 100),
            "frequent_q": em & (q_cnt > 100),
            "after_error": em & prev_wrong,
            "after_correct": em & prev_correct,
            "after_dom_misconception": em & prev_dom_misc,
            "after_overconfident_error": em & prev_overconf_step,
            "single_kc": em & single_kc_q,
            "multi_kc": em & multi_kc_q,
            "domain_number": em & q_is_num,
            "domain_algebra": em & q_is_alg,
            "domain_geometry": em & q_is_geom,
            "within_quiz_burst": em & batch_b,
            "cross_session": em & (~batch_b),
            "overconfident_err_history": em & prev_overconf_hist,
            "no_overconfident_err_history": em & (~prev_overconf_hist),
            "kc_first_exposure": em & kc_first,
            "repeat_kc": em & (~kc_first),
        }
        if "is_target_mask" in seq_dict:
            tm = seq_dict["is_target_mask"].bool()
            strata["task4_any_target_split"] = em & tm.any(dim=-1)
            strata["task4_split0"] = em & tm[:, :, 0]
        return strata

    # ------------------------------------------------------------------
    # Orchestrator: Execute All 11 Steps End-to-End
    # ------------------------------------------------------------------
    def run_all(self) -> Dict[str, Any]:
        """Run the complete 11-step Eedi Task 3 & 4 preprocessing pipeline."""
        step1 = self.run_step1_audit_and_mappings()
        ocr_items_by_qid, layout_inventory = self.run_step2_cv_slide_parsing()

        # RoBERTa tokenizer used solely for exact token budget verification during Steps 3 & 4
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(self.cfg["text"]["model_name"], local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(self.cfg["text"]["model_name"])
        def tok_len(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=True))

        vlm_model = self.cfg.get("ocr_vlm", {}).get("vlm_model", DEFAULT_VLM_MODEL)
        llm_model = self.cfg.get("ocr_vlm", {}).get("llm_model", DEFAULT_LLM_MODEL)
        ollama_url = self.cfg.get("ocr_vlm", {}).get("ollama_url", DEFAULT_OLLAMA_URL)

        ocr_results = run_step3_ocr_extraction(
            unique_rep_qids=step1["unique_rep_qids"],
            dup_to_rep=step1["dup_to_rep"],
            images_dir=self.images_dir,
            ocr_items_by_qid=ocr_items_by_qid,
            layout_inventory=layout_inventory,
            out_jsonl_path=self.feat_dir / "ocr_texts_q.jsonl",
            vlm_model=vlm_model,
            ollama_url=ollama_url,
            trim_tokens_fn=tok_len,
            kc_prefix_by_qid=step1["kc_prefix_by_qid"],
        )

        correct_dict = {q: int(step1["correct_answer_idx"][q].item()) for q in range(NUM_QUESTIONS)}
        sol_results = run_step4_solution_generation(
            unique_rep_qids=step1["unique_rep_qids"],
            dup_to_rep=step1["dup_to_rep"],
            images_dir=self.images_dir,
            ocr_by_qid=ocr_results,
            correct_idx_by_qid=correct_dict,
            kc_names_by_qid=step1["kc_names_by_qid"],
            out_jsonl_path=self.feat_dir / "solutions_s.jsonl",
            vlm_model=vlm_model,
            llm_model=llm_model,
            ollama_url=ollama_url,
            token_len_fn=tok_len,
        )

        step5 = self.run_step5_letterbox_and_masks(layout_inventory, ocr_results)
        step67 = self.run_step6_and_7_encoders_and_hypergraph(
            step1, layout_inventory, ocr_results, sol_results, step5
        )

        # Step 8, 9, 10: Splits, CTT/Distractors, and Temporal Sequence Construction
        split_dict = self._create_stratified_5fold_split(step1["train_df"])
        step9 = self.run_step9_fold_priors(step1["train_df"], split_dict, step1)

        # Save test public/private UID lists alongside splits
        pub_uids = sorted(int(u) for u in step1["pub_df"]["UserId"].unique())
        priv_uids = sorted(int(u) for u in step1["priv_df"]["UserId"].unique())
        with open(self.splits_dir / "test_public_uids.json", "w", encoding="utf-8") as f:
            json.dump(pub_uids, f, indent=2)
        with open(self.splits_dir / "test_private_uids.json", "w", encoding="utf-8") as f:
            json.dump(priv_uids, f, indent=2)

        question_static_ctx = self._build_question_static_ctx(
            step1, layout_inventory, ocr_results, sol_results, step5, step67
        )

        print("[Step 8 & 10] Formatting fold and test strided trajectory sequences...", flush=True)
        fold_seqs: Dict[int, Dict[str, torch.Tensor]] = {}
        for f_idx in range(5):
            uids_f = set(split_dict[f"fold_{f_idx}"])
            df_f = step1["train_df"][step1["train_df"]["UserId"].isin(uids_f)]
            fold_seqs[f_idx] = self._build_split_sequences(
                df_f, step1, step9["fold_dominant_distractor"], has_targets=False
            )

        pub_seq = self._build_split_sequences(
            step1["pub_df"], step1, step9["fold_dominant_distractor"], has_targets=True
        )
        priv_seq = self._build_split_sequences(
            step1["priv_df"], step1, step9["fold_dominant_distractor"], has_targets=True
        )
        unified_test_df = pd.concat([step1["pub_df"], step1["priv_df"]], ignore_index=True)
        unified_test_seq = self._build_split_sequences(
            unified_test_df, step1, step9["fold_dominant_distractor"], has_targets=True
        )

        # Question frequency counts across full training set for cold-start stratification
        train_q_counts_np = np.bincount(
            step1["train_df"]["QuestionId"].to_numpy(dtype=np.int64), minlength=NUM_QUESTIONS
        )
        q_train_counts = torch.from_numpy(train_q_counts_np).to(torch.int64)

        test_strata_masks = {
            "public": self._compute_test_strata_masks(
                pub_seq, step5["modality_masks"], q_train_counts, step1, layout_inventory
            ),
            "private": self._compute_test_strata_masks(
                priv_seq, step5["modality_masks"], q_train_counts, step1, layout_inventory
            ),
            "unified": self._compute_test_strata_masks(
                unified_test_seq, step5["modality_masks"], q_train_counts, step1, layout_inventory
            ),
        }

        # Step 11: Serialize Two-Tier Storage Artifacts
        print(f"[Step 11] Serializing Two-Tier artifacts under {self.out_dir}...", flush=True)
        static_item_bank: Dict[str, torch.Tensor] = {
            "modality_masks": step5["modality_masks"],
            "question_leaf_kcs": step1["question_leaf_kcs"],
            "question_kc_mask": step1["question_kc_mask"],
            "question_leaf_kc_mask": step1["question_kc_mask"],
            "question_all_kcs": step1["question_all_kcs"],
            "question_all_kc_mask": step1["question_all_kc_mask"],
            "question_static_ctx": question_static_ctx,
            "correct_answer_idx": step1["correct_answer_idx"],
            "acd_E_txt_q": step67["acd_E_txt_q"],
            "acd_mask_txt_q": step67["acd_mask_txt_q"],
            "acd_E_txt_s": step67["acd_E_txt_s"],
            "acd_mask_txt_s": step67["acd_mask_txt_s"],
            "acd_E_img_q": step67["acd_E_img_q"],
            "acd_E_img_s": step67["acd_E_img_s"],
            "acd_kc_init_emb": step67["acd_kc_init_emb"],
            "acd_hypergraph_H": step67["acd_hypergraph_H"],
            "acd_hypergraph_G": step67["acd_hypergraph_G"],
            "mg_v_txt_q": step67["mg_v_txt_q"],
            "mg_v_txt_s": step67["mg_v_txt_s"],
            "mg_v_img_q": step67["mg_v_img_q"],
            "mg_v_img_s": step67["mg_v_img_s"],
            "option_multimodal_emb": step67["option_multimodal_emb"],
            "acd_ctt_difficulty_folds": step9["D_q_folds"],
            "acd_ctt_valid_mask": step9["N_q_folds"] >= 5,
        }
        torch.save(static_item_bank, self.out_dir / "static_item_bank.pt")

        mg_static_bank = {
            "modality_masks": static_item_bank["modality_masks"],
            "question_leaf_kcs": static_item_bank["question_leaf_kcs"],
            "question_kc_mask": static_item_bank["question_kc_mask"],
            "question_static_ctx": static_item_bank["question_static_ctx"],
            "correct_answer_idx": static_item_bank["correct_answer_idx"],
            "mg_v_txt_q": static_item_bank["mg_v_txt_q"],
            "mg_v_txt_s": static_item_bank["mg_v_txt_s"],
            "mg_v_img_q": static_item_bank["mg_v_img_q"],
            "mg_v_img_s": static_item_bank["mg_v_img_s"],
            "option_multimodal_emb": static_item_bank["option_multimodal_emb"],
        }
        torch.save(mg_static_bank, self.out_dir / "mg_static_item_bank.pt")

        aux_metadata = {
            "M_patch_q": step5["M_patch_q"],
            "patch_valid_mask": step5["patch_valid_mask"],
            "N_q_folds": step9["N_q_folds"],
            "mu_prior_folds": step9["mu_prior_folds"],
            "mu_concept_folds": step9["mu_concept_folds"],
            "omega_q_folds": step9["omega_q_folds"],
            "D_q_folds": step9["D_q_folds"],
            "fold_distractor_entropy": step9["fold_distractor_entropy"],
            "fold_dominant_distractor": step9["fold_dominant_distractor"],
            "token_to_word_q": step67["token_to_word_q"],
            "token_to_word_s": step67["token_to_word_s"],
            "image_meta_q": step5["image_meta_q"],
            "image_meta_s": step5["image_meta_s"],
            "task3_expert_pairs": step1["expert_pairs"],
            "kc_id_to_raw": step1["kc_to_raw"],
            "raw_to_kc_id": step1["raw_to_kc"],
            "kc_descriptions": step1["kc_descriptions"],
            "question_train_counts": q_train_counts,
            "question_duplicate_groups": step1["dup_to_rep"],
            "kappa": KAPPA,
            "m_kc": M_KC,
            "m_item": M_ITEM,
        }
        torch.save(aux_metadata, self.out_dir / "acd_v3_aux_metadata.pt")

        for f_idx in range(5):
            torch.save(fold_seqs[f_idx], self.out_dir / f"sequences_fold_{f_idx}.pt")
        torch.save(pub_seq, self.out_dir / "sequences_test_public.pt")
        torch.save(priv_seq, self.out_dir / "sequences_test_private.pt")
        torch.save(unified_test_seq, self.out_dir / "sequences_test.pt")
        torch.save(test_strata_masks, self.out_dir / "test_strata_masks.pt")

        summary = {
            "dataset": "Eedi NeurIPS 2020 (Task 3 & 4)",
            "num_questions": NUM_QUESTIONS,
            "num_unique_images": len(step1["unique_rep_qids"]),
            "duplicate_groups": step1["dup_pairs"],
            "historical_key_revisions": step1["multi_key_qids"],
            "num_kcs_total": 85,
            "num_leaf_kcs": 57,
            "num_l2_kcs": 25,
            "num_l1_kcs": 3,
            "num_diagram_questions": int(step5["modality_masks"][:, 0].sum().item()),
            "num_text_only_questions": int((step5["modality_masks"][:, 0] < 0.5).sum().item()),
            "num_train_students": len(step1["train_df"]["UserId"].unique()),
            "num_test_public_students": len(pub_uids),
            "num_test_private_students": len(priv_uids),
            "fold_eval_counts": {f"fold_{f}": int(fold_seqs[f]["eval_mask"].sum().item()) for f in range(5)},
            "test_public_eval_steps": int(pub_seq["eval_mask"].sum().item()),
            "test_private_eval_steps": int(priv_seq["eval_mask"].sum().item()),
            "test_unified_eval_steps": int(unified_test_seq["eval_mask"].sum().item()),
            "min_fold_question_train_count": int(step9["N_q_folds"].min().item()),
        }
        with open(self.out_dir / "metadata_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[Step 11] Pipeline finished successfully! Summary:\n{json.dumps(summary, indent=2)}", flush=True)
        return summary
