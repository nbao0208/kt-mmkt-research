"""
XES3G5M Multimodal Knowledge Tracing (MMKT) Preprocessing Module.

This module implements the complete 9-step preprocessing pipeline for the XES3G5M
dataset, supporting both MG-MMKT (Modality-Gated MMKT) and ACD-MMKT (Aligned,
Cognitive-Load & Dual-Stream MMKT).

The pipeline enforces:
1. Physical and relational data integrity audit (filtering 102 orphan images and MD5 deduplication).
2. Pedagogical text normalization and strict pre/post-response partitioning.
3. Geometry-preserving image letterboxing with alpha compositing and modality masking.
4. Dual-granularity feature extraction (pooled global vectors for MG-MMKT and token/patch states for ACD-MMKT).
5. Hierarchical exercise-concept hypergraph construction (incidence matrix H and propagation matrix G).
6. Chronological sequence reconstruction, padding removal, and 5-fold anti-leakage splitting.
7. Fold-specific Classical Test Theory (CTT) difficulty estimation with Empirical Bayes shrinkage.
8. Temporal dynamics encoding (log lag time, skill spacing) and causal student context engineering.
9. Two-tier tensor serialization (Tier 1: Static Item Bank, Tier 2: Dynamic Sequence Store).
"""

import os
import re
import json
import hashlib
import unicodedata
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, CLIPImageProcessor


class XES3G5MPreprocessor:
    """
    Comprehensive preprocessor for the XES3G5M multimodal knowledge tracing dataset.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw/XES3G5M",
        processed_dir: str = "data/processed/XES3G5M",
        text_model_name: str = "hfl/chinese-roberta-wwm-ext",
        vision_model_name: str = "openai/clip-vit-base-patch16",
        max_token_len_q: int = 128,
        max_token_len_s: int = 160,
        max_seq_len: int = 200,
        alpha_anc: float = 0.5,
        ctt_m: float = 10.0,
        device: Optional[str] = None,
    ):
        """
        Initialize the XES3G5M preprocessor with path references and hyperparameters.

        Args:
            raw_dir: Directory containing raw XES3G5M data.
            processed_dir: Directory where processed artifacts will be written.
            text_model_name: Hugging Face model identifier for text feature extraction.
            vision_model_name: Hugging Face model identifier for vision patch extraction.
            max_token_len_q: Maximum token length for question stem representations.
            max_token_len_s: Maximum token length for solution text representations.
            max_seq_len: Maximum length of student interaction windows.
            alpha_anc: Hypergraph edge weight attenuation for ancestor concepts.
            ctt_m: Prior pseudo-interaction weight for Empirical Bayes CTT shrinkage.
            device: Computing device ('mps', 'cuda', 'cpu'). Auto-selected if None.
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
        self.max_token_len_q = max_token_len_q
        self.max_token_len_s = max_token_len_s
        self.max_seq_len = max_seq_len
        self.alpha_anc = alpha_anc
        self.ctt_m = ctt_m

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        os.makedirs(self.processed_dir, exist_ok=True)

        # Regex for identifying image tokens in text fields
        self.img_ref_regex = re.compile(r"((?:question|analysis)_\d+-image_\d+)(?:\.png)?")
        self.blank_regex = re.compile(r"_{2,}|（\s*）|\(\s*\)")
        self.html_tag_regex = re.compile(r"<[^>]+>")

        # Placeholder attributes populated during preprocessing
        self.questions_raw: Dict[str, Any] = {}
        self.kc_routes_map: Dict[str, str] = {}
        self.name_to_kcid: Dict[str, int] = {}
        self.orphan_images: Set[str] = set()
        self.valid_images: Set[str] = set()
        self.md5_to_filename: Dict[str, str] = {}
        self.filename_to_md5: Dict[str, str] = {}
        self.unique_md5_list: List[str] = []

        self.normalized_texts_q: Dict[int, str] = {}
        self.normalized_texts_s: Dict[int, str] = {}
        self.stem_opt_images_by_q: Dict[int, List[str]] = {}
        self.sol_images_by_q: Dict[int, List[str]] = {}
        self.modality_masks: Optional[np.ndarray] = None
        self.root_domain_to_id: Dict[str, int] = {}

    def load_metadata(self) -> None:
        """
        Load raw questions.json and kc_routes_map.json into memory.
        """
        q_path = os.path.join(self.raw_dir, "metadata", "questions.json")
        kc_path = os.path.join(self.raw_dir, "metadata", "kc_routes_map.json")

        with open(q_path, "r", encoding="utf-8") as f:
            self.questions_raw = json.load(f)

        with open(kc_path, "r", encoding="utf-8") as f:
            self.kc_routes_map = json.load(f)

        # Create bijective mapping from exact string representation to integer ID
        self.name_to_kcid = {v: int(k) for k, v in self.kc_routes_map.items()}

    # =========================================================================
    # Step 1: Data Integrity Audit & Canonical Indexing
    # =========================================================================
    def step1_audit_and_canonical_indexing(self) -> Dict[str, Any]:
        """
        Scan all questions for image references, detect orphan disk images,
        compute MD5 hashes for deduplication, and build canonical mappings.

        Returns:
            Dictionary containing audit metrics and mapping tables.
        """
        if not self.questions_raw:
            self.load_metadata()

        img_dir = os.path.join(self.raw_dir, "metadata", "images")
        disk_files = set(os.listdir(img_dir))
        disk_stems = {os.path.splitext(f)[0]: f for f in disk_files}

        referenced_stems = set()
        for qid_str, q_info in self.questions_raw.items():
            content_matches = self.img_ref_regex.findall(q_info.get("content", ""))
            opt_matches = []
            opts = q_info.get("options", {})
            if isinstance(opts, dict):
                for opt_val in opts.values():
                    opt_matches.extend(self.img_ref_regex.findall(str(opt_val)))
            sol_matches = self.img_ref_regex.findall(q_info.get("analysis", ""))

            for stem in content_matches + opt_matches + sol_matches:
                referenced_stems.add(stem)

        # Identify valid referenced disk images vs orphan images
        self.valid_images = {disk_stems[s] for s in referenced_stems if s in disk_stems}
        self.orphan_images = disk_files - self.valid_images

        # Compute MD5 hashes for all disk images and for valid images
        self.md5_to_filename = {}
        self.filename_to_md5 = {}
        all_disk_md5s = set()
        for fname in sorted(list(disk_files)):
            fpath = os.path.join(img_dir, fname)
            with open(fpath, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            all_disk_md5s.add(h)
            if fname in self.valid_images:
                self.filename_to_md5[fname] = h
                if h not in self.md5_to_filename:
                    self.md5_to_filename[h] = fname

        self.unique_md5_list = sorted(list(self.md5_to_filename.keys()))
        self.total_disk_md5_count = len(all_disk_md5s)

        audit_results = {
            "total_disk_images": len(disk_files),
            "valid_referenced_images": len(self.valid_images),
            "orphan_images_count": len(self.orphan_images),
            "total_disk_md5_clusters": self.total_disk_md5_count,
            "unique_md5_clusters": self.total_disk_md5_count,
            "valid_md5_clusters": len(self.unique_md5_list),
            "total_questions": len(self.questions_raw),
            "total_kcs": len(self.kc_routes_map),
        }
        return audit_results

    # =========================================================================
    # Step 2: Pedagogical Text Normalization & Pre/Post-Response Partitioning
    # =========================================================================
    def clean_text_segment(self, text: str) -> str:
        """
        Clean an individual text segment: normalize Unicode, strip HTML tags,
        standardize LaTeX formulas, and replace physical image filenames with tokens.
        """
        if not text:
            return ""

        # Normalize Unicode to NFKC
        text = unicodedata.normalize("NFKC", text)

        # Strip unnecessary HTML tags
        text = self.html_tag_regex.sub(" ", text)

        # Replace physical image paths with standardized semantic tokens [IMG_k]
        def replace_img_token(match: re.Match) -> str:
            raw_stem = match.group(1)
            # Extract image index k
            idx_match = re.search(r"image_(\d+)", raw_stem)
            k = idx_match.group(1) if idx_match else "0"
            return f" [IMG_{k}] "

        text = self.img_ref_regex.sub(replace_img_token, text)

        # Normalize double dollar math blocks $$...$$ to inline $...$
        text = re.sub(r"\$\$\s*(.*?)\s*\$\$", r"$\1$", text)

        # Clean multiple spaces and newlines
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def step2_pedagogical_text_normalization(self) -> Tuple[Dict[int, str], Dict[int, str]]:
        """
        Construct question stem T_q and solution analysis T_s for all 7,652 questions.

        Returns:
            Tuple of (normalized_texts_q, normalized_texts_s).
        """
        if not self.questions_raw:
            self.load_metadata()

        self.normalized_texts_q = {}
        self.normalized_texts_s = {}

        for qid_str, q_info in self.questions_raw.items():
            qid = int(qid_str)
            raw_content = q_info.get("content", "")
            raw_analysis = q_info.get("analysis", "")
            raw_answer = q_info.get("answer", [])
            q_type = q_info.get("type", "填空")
            raw_options = q_info.get("options", {})

            # Clean base content and normalize answer blanks
            cleaned_content = self.clean_text_segment(raw_content)
            cleaned_content = self.blank_regex.sub(" [BLANK] ", cleaned_content)
            cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()

            # Format options for multiple-choice questions
            if q_type == "单选" and isinstance(raw_options, dict) and raw_options:
                option_items = []
                for opt_key in sorted(raw_options.keys()):
                    opt_val = self.clean_text_segment(str(raw_options[opt_key]))
                    option_items.append(f"选项 {opt_key}: {opt_val}")
                options_str = " [SEP] " + " | ".join(option_items)
                t_q = cleaned_content + options_str
            else:
                t_q = cleaned_content

            # Format standardized solution text T_s
            ans_str = "".join(str(a) for a in raw_answer) if isinstance(raw_answer, list) else str(raw_answer)
            cleaned_analysis = self.clean_text_segment(raw_analysis)
            t_s = f"标准答案: {ans_str} [SEP] 解析: {cleaned_analysis}"

            self.normalized_texts_q[qid] = t_q
            self.normalized_texts_s[qid] = t_s

        return self.normalized_texts_q, self.normalized_texts_s

    # =========================================================================
    # Step 3: Visual Standardization & Modality Masking
    # =========================================================================
    def step3_visual_standardization_and_masks(self) -> np.ndarray:
        """
        Map valid images to stem and solution roles for each question,
        and construct static modality masks (m_img_q, m_txt_s, m_img_s).

        Returns:
            Binary array of shape [7652, 3] representing modality availability.
        """
        if not self.questions_raw:
            self.load_metadata()
        if not self.valid_images:
            self.step1_audit_and_canonical_indexing()

        img_dir = os.path.join(self.raw_dir, "metadata", "images")
        disk_stems = {os.path.splitext(f)[0]: f for f in self.valid_images}

        num_questions = 7652
        self.modality_masks = np.zeros((num_questions, 3), dtype=np.float32)
        self.stem_opt_images_by_q = {q: [] for q in range(num_questions)}
        self.sol_images_by_q = {q: [] for q in range(num_questions)}

        for qid_str, q_info in self.questions_raw.items():
            qid = int(qid_str)

            # Stem images from content and options
            content_matches = self.img_ref_regex.findall(q_info.get("content", ""))
            opt_matches = []
            opts = q_info.get("options", {})
            if isinstance(opts, dict):
                for opt_val in opts.values():
                    opt_matches.extend(self.img_ref_regex.findall(str(opt_val)))

            # Solution images from analysis
            sol_matches = self.img_ref_regex.findall(q_info.get("analysis", ""))

            stem_files = [disk_stems[s] for s in content_matches + opt_matches if s in disk_stems]
            sol_files = [disk_stems[s] for s in sol_matches if s in disk_stems]

            # Preserve unique image files per question
            self.stem_opt_images_by_q[qid] = sorted(list(dict.fromkeys(stem_files)))
            self.sol_images_by_q[qid] = sorted(list(dict.fromkeys(sol_files)))

            # Modality masks:
            # Column 0: m_img^q (question stem has image)
            # Column 1: m_txt^s (solution text exists) -> Always 1.0 on XES3G5M
            # Column 2: m_img^s (solution has image)
            m_img_q = 1.0 if len(self.stem_opt_images_by_q[qid]) > 0 else 0.0
            m_txt_s = 1.0 if len(q_info.get("analysis", "")) > 0 else 0.0
            m_img_s = 1.0 if len(self.sol_images_by_q[qid]) > 0 else 0.0

            self.modality_masks[qid] = [m_img_q, m_txt_s, m_img_s]

        return self.modality_masks

    @staticmethod
    def preprocess_image_letterbox(img_path: str, target_size: int = 224) -> Image.Image:
        """
        Alpha-composite an image onto a pure white canvas, preserve its aspect ratio
        via bicubic scaling along the longest edge, and pad remainder with white (255, 255, 255).
        """
        img = Image.open(img_path)

        # Handle transparency channels to avoid black background artifacts
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            canvas = Image.new("RGB", img.size, (255, 255, 255))
            canvas.paste(img, mask=img.split()[3])
            img = canvas
        else:
            img = img.convert("RGB")

        # Aspect-ratio preserving letterbox resize
        w, h = img.size
        scale = target_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized_img = img.resize((new_w, new_h), Image.BICUBIC)
        letterboxed = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        letterboxed.paste(resized_img, (pad_x, pad_y))

        return letterboxed

    # =========================================================================
    # Step 4: Dual-Granularity Multimodal Feature Extraction
    # =========================================================================
    def extract_text_features(
        self,
        batch_size: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract token-level hidden states (for ACD-MMKT) and pooled representations (for MG-MMKT)
        using the pretrained Chinese RoBERTa language model.
        """
        if not self.normalized_texts_q or not self.normalized_texts_s:
            self.step2_pedagogical_text_normalization()

        tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        model = AutoModel.from_pretrained(self.text_model_name).to(self.device).eval()

        num_questions = 7652
        acd_E_txt_q = torch.zeros((num_questions, self.max_token_len_q, 768), dtype=torch.float16)
        acd_mask_txt_q = torch.zeros((num_questions, self.max_token_len_q), dtype=torch.bool)
        mg_v_txt_q = torch.zeros((num_questions, 768), dtype=torch.float16)

        acd_E_txt_s = torch.zeros((num_questions, self.max_token_len_s, 768), dtype=torch.float16)
        acd_mask_txt_s = torch.zeros((num_questions, self.max_token_len_s), dtype=torch.bool)
        mg_v_txt_s = torch.zeros((num_questions, 768), dtype=torch.float16)

        # Process question stem texts T_q
        q_texts = [self.normalized_texts_q[q] for q in range(num_questions)]
        for i in range(0, num_questions, batch_size):
            batch_texts = q_texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=self.max_token_len_q,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                out = model(**encoded)
                hidden = out.last_hidden_state  # [B, L_q, 768]
                mask = encoded.attention_mask.bool()  # [B, L_q]

                # Masked mean pooling for MG-MMKT
                mask_expanded = mask.unsqueeze(-1).float()
                sum_embeddings = (hidden * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
                pooled = sum_embeddings / sum_mask

                b_size = len(batch_texts)
                acd_E_txt_q[i : i + b_size] = hidden.cpu().to(torch.float16)
                acd_mask_txt_q[i : i + b_size] = mask.cpu()
                mg_v_txt_q[i : i + b_size] = pooled.cpu().to(torch.float16)

        # Process solution texts T_s
        s_texts = [self.normalized_texts_s[q] for q in range(num_questions)]
        for i in range(0, num_questions, batch_size):
            batch_texts = s_texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=self.max_token_len_s,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                out = model(**encoded)
                hidden = out.last_hidden_state  # [B, L_s, 768]
                mask = encoded.attention_mask.bool()

                mask_expanded = mask.unsqueeze(-1).float()
                sum_embeddings = (hidden * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
                pooled = sum_embeddings / sum_mask

                b_size = len(batch_texts)
                acd_E_txt_s[i : i + b_size] = hidden.cpu().to(torch.float16)
                acd_mask_txt_s[i : i + b_size] = mask.cpu()
                mg_v_txt_s[i : i + b_size] = pooled.cpu().to(torch.float16)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "acd_E_txt_q": acd_E_txt_q,
            "acd_mask_txt_q": acd_mask_txt_q,
            "mg_v_txt_q": mg_v_txt_q,
            "acd_E_txt_s": acd_E_txt_s,
            "acd_mask_txt_s": acd_mask_txt_s,
            "mg_v_txt_s": mg_v_txt_s,
        }

    def extract_vision_features(
        self,
        batch_size: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract spatial patch features (for ACD-MMKT) and pooled representations (for MG-MMKT)
        using the pretrained Vision Transformer (CLIP ViT-B/16).
        Applies MD5 deduplication to process only unique images.
        """
        if not self.unique_md5_list:
            self.step1_audit_and_canonical_indexing()
        if self.modality_masks is None:
            self.step3_visual_standardization_and_masks()

        img_processor = CLIPImageProcessor.from_pretrained(self.vision_model_name)
        vision_model = CLIPVisionModel.from_pretrained(self.vision_model_name).to(self.device).eval()

        img_dir = os.path.join(self.raw_dir, "metadata", "images")

        # Deduplication: extract representations for unique MD5 hashes only
        md5_patch_embs: Dict[str, torch.Tensor] = {}
        md5_global_embs: Dict[str, torch.Tensor] = {}

        for i in range(0, len(self.unique_md5_list), batch_size):
            batch_md5s = self.unique_md5_list[i : i + batch_size]
            batch_imgs = []
            for h in batch_md5s:
                fname = self.md5_to_filename[h]
                fpath = os.path.join(img_dir, fname)
                processed_img = self.preprocess_image_letterbox(fpath, target_size=224)
                batch_imgs.append(processed_img)

            pixel_inputs = img_processor(images=batch_imgs, return_tensors="pt")
            pixel_values = pixel_inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                out = vision_model(pixel_values=pixel_values)
                last_hidden = out.last_hidden_state  # [B, 197, 768]
                cls_tokens = last_hidden[:, 0, :]  # [B, 768]
                patch_tokens = last_hidden[:, 1:, :]  # [B, 196, 768]

                # Reshape to 14x14 grid and perform 2x2 spatial adaptive average pooling -> 7x7 (49 patches)
                b_dim = patch_tokens.shape[0]
                p_grid = patch_tokens.view(b_dim, 14, 14, 768).permute(0, 3, 1, 2)
                p_pooled = F.adaptive_avg_pool2d(p_grid, (7, 7)).permute(0, 2, 3, 1).view(b_dim, 49, 768)

                for idx, h in enumerate(batch_md5s):
                    md5_patch_embs[h] = p_pooled[idx].cpu().to(torch.float16)
                    md5_global_embs[h] = cls_tokens[idx].cpu().to(torch.float16)

        del vision_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Map image features back to each question, averaging multiple images where applicable
        num_questions = 7652
        acd_E_img_q = torch.zeros((num_questions, 49, 768), dtype=torch.float16)
        mg_v_img_q = torch.zeros((num_questions, 768), dtype=torch.float16)

        acd_E_img_s = torch.zeros((num_questions, 49, 768), dtype=torch.float16)
        mg_v_img_s = torch.zeros((num_questions, 768), dtype=torch.float16)

        for qid in range(num_questions):
            # Question stem images
            stem_files = self.stem_opt_images_by_q[qid]
            if stem_files:
                patches_list = [md5_patch_embs[self.filename_to_md5[f]] for f in stem_files]
                globals_list = [md5_global_embs[self.filename_to_md5[f]] for f in stem_files]
                acd_E_img_q[qid] = torch.stack(patches_list, dim=0).mean(dim=0)
                mg_v_img_q[qid] = torch.stack(globals_list, dim=0).mean(dim=0)

            # Solution images
            sol_files = self.sol_images_by_q[qid]
            if sol_files:
                patches_list = [md5_patch_embs[self.filename_to_md5[f]] for f in sol_files]
                globals_list = [md5_global_embs[self.filename_to_md5[f]] for f in sol_files]
                acd_E_img_s[qid] = torch.stack(patches_list, dim=0).mean(dim=0)
                mg_v_img_s[qid] = torch.stack(globals_list, dim=0).mean(dim=0)

        return {
            "acd_E_img_q": acd_E_img_q,
            "mg_v_img_q": mg_v_img_q,
            "acd_E_img_s": acd_E_img_s,
            "mg_v_img_s": mg_v_img_s,
        }

    # =========================================================================
    # Step 5: Question-Skill Hypergraph Construction for ACD-MMKT
    # =========================================================================
    def step5_build_hypergraph(
        self,
        alpha_anc: Optional[float] = None,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """
        Construct question-skill incidence matrix H and normalized propagation matrix G
        connecting 1,175 knowledge components and 7,652 questions.
        Loads semantic node initializations E_K^(0) from cid2content_emb.json.

        Args:
            alpha_anc: Attenuation factor for ancestor KCs (default: 0.5).
            eps: Smoothing epsilon to prevent division by zero for isolated nodes.

        Returns:
            Dictionary containing H, G, and E_K^(0) tensors.
        """
        if alpha_anc is None:
            alpha_anc = self.alpha_anc
        if not self.questions_raw:
            self.load_metadata()

        num_kcs = 1175
        num_questions = 7652

        # 1. Load initial semantic embeddings for all 1,175 KCs
        cid_emb_path = os.path.join(self.raw_dir, "metadata", "embeddings", "cid2content_emb.json")
        with open(cid_emb_path, "r", encoding="utf-8") as f:
            cid_dict = json.load(f)

        acd_kc_init_emb = torch.zeros((num_kcs, 768), dtype=torch.float32)
        for k in range(num_kcs):
            acd_kc_init_emb[k] = torch.tensor(cid_dict[str(k)], dtype=torch.float32)

        # 2. Build multi-tier incidence matrix H
        H = torch.zeros((num_kcs, num_questions), dtype=torch.float32)

        for qid_str, q_info in self.questions_raw.items():
            qid = int(qid_str)
            leaf_kcs = set()
            anc_kcs = set()

            for route in q_info.get("kc_routes", []):
                parts = route.split("----")
                if not parts:
                    continue
                leaf_name = parts[-1]
                if leaf_name in self.name_to_kcid:
                    leaf_kcs.add(self.name_to_kcid[leaf_name])

                for anc_name in parts[:-1]:
                    if anc_name in self.name_to_kcid:
                        anc_kcs.add(self.name_to_kcid[anc_name])

            # Leaf KCs receive full weight 1.0
            for k in leaf_kcs:
                H[k, qid] = 1.0

            # Ancestor KCs receive attenuated weight alpha_anc
            for k in anc_kcs - leaf_kcs:
                H[k, qid] = float(alpha_anc)

        # 3. Compute normalized propagation matrix G = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        d_v = H.sum(dim=1)  # Node degrees [1175]
        d_e = H.sum(dim=0)  # Hyperedge degrees [7652]

        d_v_smooth = torch.clamp(d_v, min=eps)
        d_e_smooth = torch.clamp(d_e, min=eps)

        d_v_inv_sqrt = torch.pow(d_v_smooth, -0.5)
        d_e_inv = torch.pow(d_e_smooth, -1.0)

        # Scaled H: H_scaled = H * D_e^{-1}
        H_scaled = H * d_e_inv.unsqueeze(0)  # [1175, 7652]

        # Intermediate product: M = H * D_e^{-1} * H^T [1175, 1175]
        M = torch.matmul(H_scaled, H.t())

        # Symmetric normalization: G = D_v^{-1/2} * M * D_v^{-1/2}
        G = d_v_inv_sqrt.unsqueeze(1) * M * d_v_inv_sqrt.unsqueeze(0)

        # Numerical symmetrization
        G = 0.5 * (G + G.t())

        # Check for NaN / Inf
        if torch.isnan(G).any() or torch.isinf(G).any():
            raise ValueError("Propagation matrix G contains NaN or Inf values.")

        return {
            "acd_hypergraph_H": H,
            "acd_hypergraph_G": G,
            "acd_kc_init_emb": acd_kc_init_emb,
        }

    # =========================================================================
    # Step 6: Sequence Reconstruction & Anti-Leakage Partitioning
    # =========================================================================
    def parse_interaction_sequences(self) -> Dict[str, Any]:
        """
        Parse student interaction logs from question_level CSVs, filtering padding elements
        (selectmasks == -1) and reconstructing multi-concept questions.

        Returns:
            Dictionary containing student records grouped by train_valid folds (0..4) and test.
        """
        train_csv_path = os.path.join(self.raw_dir, "question_level", "train_valid_sequences_quelevel.csv")
        test_csv_path = os.path.join(self.raw_dir, "question_level", "test_quelevel.csv")

        # Load train_valid data
        df_train = pd.read_csv(train_csv_path)
        students_by_fold: Dict[int, List[Dict[str, Any]]] = {f: [] for f in range(5)}

        # Group rows by student to reconstruct complete chronological interaction history
        for uid, group in df_train.groupby("uid", sort=False):
            fold = int(group["fold"].iloc[0])
            u_questions: List[int] = []
            u_responses: List[int] = []
            u_timestamps: List[int] = []
            u_concepts: List[List[int]] = []

            for _, row in group.iterrows():
                q_arr = np.fromstring(str(row["questions"]), sep=",", dtype=np.int32)
                r_arr = np.fromstring(str(row["responses"]), sep=",", dtype=np.int8)
                t_arr = np.fromstring(str(row["timestamps"]), sep=",", dtype=np.int64)
                m_arr = np.fromstring(str(row["selectmasks"]), sep=",", dtype=np.int8)

                valid_mask = m_arr == 1
                q_valid = q_arr[valid_mask]
                r_valid = r_arr[valid_mask]
                t_valid = t_arr[valid_mask]

                c_items = str(row["concepts"]).split(",")
                for idx, is_valid in enumerate(valid_mask):
                    if is_valid and idx < len(c_items):
                        c_str = c_items[idx]
                        kcs = [int(x) for x in c_str.split("_") if x and x != "-1"]
                        u_concepts.append(kcs)

                u_questions.extend(q_valid.tolist())
                u_responses.extend(r_valid.tolist())
                u_timestamps.extend(t_valid.tolist())

            if u_questions:
                students_by_fold[fold].append({
                    "uid": int(uid),
                    "fold": fold,
                    "questions": np.array(u_questions, dtype=np.int32),
                    "responses": np.array(u_responses, dtype=np.int8),
                    "timestamps": np.array(u_timestamps, dtype=np.int64),
                    "concepts": u_concepts,
                })

        # Load test data
        df_test = pd.read_csv(test_csv_path)
        test_students: List[Dict[str, Any]] = []

        for _, row in df_test.iterrows():
            uid = int(row["uid"])
            q_arr = np.fromstring(str(row["questions"]), sep=",", dtype=np.int32)
            r_arr = np.fromstring(str(row["responses"]), sep=",", dtype=np.int8)
            t_arr = np.fromstring(str(row["timestamps"]), sep=",", dtype=np.int64)

            # Filter uninitialized padding (-1)
            valid_mask = q_arr != -1
            q_valid = q_arr[valid_mask]
            r_valid = r_arr[valid_mask]
            t_valid = t_arr[valid_mask]

            c_items = str(row["concepts"]).split(",")
            u_concepts: List[List[int]] = []
            for idx, is_valid in enumerate(valid_mask):
                if is_valid and idx < len(c_items):
                    c_str = c_items[idx]
                    kcs = [int(x) for x in c_str.split("_") if x and x != "-1"]
                    u_concepts.append(kcs)

            if len(q_valid) > 0:
                test_students.append({
                    "uid": uid,
                    "fold": -1,
                    "questions": q_valid,
                    "responses": r_valid,
                    "timestamps": t_valid,
                    "concepts": u_concepts,
                })

        return {
            "students_by_fold": students_by_fold,
            "test_students": test_students,
        }

    # =========================================================================
    # Step 7: Classical Test Theory (CTT) Empirical Bayes Difficulty Estimation
    # =========================================================================
    def step7_estimate_ctt_difficulty(
        self,
        students_by_fold: Dict[int, List[Dict[str, Any]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate item difficulty D_q strictly per cross-validation fold using Hierarchical
        Empirical Bayes Shrinkage on the training split D_train of that fold.

        Returns:
            Tuple of (ctt_difficulty_folds [5, 7652], ctt_valid_mask [5, 7652]).
        """
        if not self.questions_raw:
            self.load_metadata()

        num_questions = 7652
        num_folds = 5
        ctt_difficulty_folds = torch.zeros((num_folds, num_questions), dtype=torch.float32)
        ctt_valid_mask = torch.zeros((num_folds, num_questions), dtype=torch.bool)

        # Map each question to its leaf KCs
        q_to_leaf_kcs: Dict[int, List[int]] = {}
        for qid_str, q_info in self.questions_raw.items():
            qid = int(qid_str)
            leaves = set()
            for r in q_info.get("kc_routes", []):
                parts = r.split("----")
                if parts and parts[-1] in self.name_to_kcid:
                    leaves.add(self.name_to_kcid[parts[-1]])
            q_to_leaf_kcs[qid] = sorted(list(leaves))

        for target_val_fold in range(num_folds):
            # Form training split D_train by taking all folds except target_val_fold
            train_students: List[Dict[str, Any]] = []
            for f in range(num_folds):
                if f != target_val_fold:
                    train_students.extend(students_by_fold[f])

            # Accumulate interaction statistics
            total_train_interactions = 0
            total_train_incorrect = 0
            q_attempts = np.zeros(num_questions, dtype=np.int64)
            q_incorrect = np.zeros(num_questions, dtype=np.int64)
            kc_attempts: Dict[int, int] = {}
            kc_incorrect: Dict[int, int] = {}

            for student in train_students:
                q_arr = student["questions"]
                r_arr = student["responses"]
                c_list = student["concepts"]

                for q, r, kcs in zip(q_arr, r_arr, c_list):
                    total_train_interactions += 1
                    err = 1 - int(r)
                    total_train_incorrect += err

                    q_attempts[q] += 1
                    q_incorrect[q] += err

                    for k in kcs:
                        kc_attempts[k] = kc_attempts.get(k, 0) + 1
                        kc_incorrect[k] = kc_incorrect.get(k, 0) + err

            # Global training error rate
            global_err_rate = total_train_incorrect / max(1, total_train_interactions)

            # KC-specific error rates
            kc_err_rates = {
                k: kc_incorrect[k] / kc_attempts[k]
                for k in kc_attempts
                if kc_attempts[k] > 0
            }

            # Calculate Empirical Bayes smoothed difficulty D_q
            for q in range(num_questions):
                n_q = q_attempts[q]
                e_q = q_incorrect[q]

                # Prior difficulty from component concepts
                leaf_kcs = q_to_leaf_kcs.get(q, [])
                kc_priors = [kc_err_rates[k] for k in leaf_kcs if k in kc_err_rates]
                if kc_priors:
                    mu_prior = float(np.mean(kc_priors))
                else:
                    mu_prior = global_err_rate

                # Ensure prior and smoothed difficulty strictly reside in (0, 1)
                mu_prior = float(np.clip(mu_prior, 1e-4, 1.0 - 1e-4))
                d_q = (e_q + self.ctt_m * mu_prior) / (n_q + self.ctt_m)
                d_q = float(np.clip(d_q, 1e-4, 1.0 - 1e-4))

                ctt_difficulty_folds[target_val_fold, q] = d_q
                ctt_valid_mask[target_val_fold, q] = bool(n_q >= 5)

        return ctt_difficulty_folds, ctt_valid_mask

    # =========================================================================
    # Step 8: Temporal Dynamics & Causal Student Context Engineering
    # =========================================================================
    def build_static_question_context(self) -> torch.Tensor:
        """
        Build static question context vector x_static,q of shape [7652, 15] containing:
        - Question type one-hot 2D (fill-in-the-blank [1, 0], multiple-choice [0, 1]).
        - Root domain one-hot 10D.
        - Number of leaf concepts normalized by 6.0.
        - Normalized stem length |T_q| / 128.0.
        - Question stem image indicator m_img^q.
        """
        if not self.questions_raw:
            self.load_metadata()
        if not self.normalized_texts_q:
            self.step2_pedagogical_text_normalization()
        if self.modality_masks is None:
            self.step3_visual_standardization_and_masks()

        # Identify unique root domains
        root_domains = set()
        for q_info in self.questions_raw.values():
            for r in q_info.get("kc_routes", []):
                parts = r.split("----")
                if parts:
                    root_domains.add(parts[0])
        self.root_domain_to_id = {dom: idx for idx, dom in enumerate(sorted(list(root_domains)))}
        num_domains = len(self.root_domain_to_id)

        num_questions = 7652
        # Feature dimension: 2 (type) + 10 (domain) + 3 (complexity) = 15
        d_static = 2 + num_domains + 3
        static_ctx = torch.zeros((num_questions, d_static), dtype=torch.float32)

        for qid in range(num_questions):
            q_info = self.questions_raw[str(qid)]
            q_type = q_info.get("type", "填空")

            # 1. Question format one-hot (2D)
            type_vec = [1.0, 0.0] if q_type == "填空" else [0.0, 1.0]

            # 2. Root domain one-hot (10D)
            domain_vec = [0.0] * num_domains
            routes = q_info.get("kc_routes", [])
            if routes and routes[0].split("----")[0] in self.root_domain_to_id:
                dom_id = self.root_domain_to_id[routes[0].split("----")[0]]
                domain_vec[dom_id] = 1.0

            # 3. Complexity features (3D)
            num_leaves = len({r.split("----")[-1] for r in routes if r.split("----")[-1] in self.name_to_kcid})
            norm_num_leaves = min(num_leaves, 6) / 6.0
            norm_text_len = min(len(self.normalized_texts_q[qid]), 128) / 128.0
            m_img_q = float(self.modality_masks[qid, 0])

            feat = type_vec + domain_vec + [norm_num_leaves, norm_text_len, m_img_q]
            static_ctx[qid] = torch.tensor(feat, dtype=torch.float32)

        return static_ctx

    def window_student_interactions(
        self,
        students: List[Dict[str, Any]],
        global_train_acc: float = 0.7945,
        max_lag_seconds: float = 1e7,
    ) -> Dict[str, torch.Tensor]:
        """
        Segment continuous student interaction trajectories into non-overlapping windows of length 200,
        computing causal running accuracy, batch submission flags, and Ebbinghaus skill spacing.

        Returns:
            Dictionary containing sequence tensors (q_seq, r_seq, seq_mask, delta_t_lag, delta_t_kc, etc.).
        """
        all_q_seq: List[np.ndarray] = []
        all_r_seq: List[np.ndarray] = []
        all_mask: List[np.ndarray] = []
        all_delta_lag: List[np.ndarray] = []
        all_delta_kc: List[np.ndarray] = []
        all_batch_flags: List[np.ndarray] = []
        all_running_acc: List[np.ndarray] = []
        all_uids: List[int] = []

        log_max_lag = np.log1p(max_lag_seconds)

        for student in students:
            uid = student["uid"]
            q_arr = student["questions"]
            r_arr = student["responses"]
            t_arr = student["timestamps"]
            c_list = student["concepts"]
            total_steps = len(q_arr)

            if total_steps == 0:
                continue

            # Compute features over complete chronological history
            t_sec = t_arr / 1000.0  # Convert ms to seconds
            delta_t = np.zeros(total_steps, dtype=np.float32)
            delta_t[1:] = np.maximum(0.0, t_sec[1:] - t_sec[:-1])

            batch_flags = (delta_t == 0.0).astype(np.float32)
            norm_delta_lag = np.log1p(delta_t) / log_max_lag

            # Skill-specific spacing (time since last exposure to any shared KC)
            delta_kc = np.zeros(total_steps, dtype=np.float32)
            last_kc_time: Dict[int, float] = {}

            for step in range(total_steps):
                step_kcs = c_list[step]
                past_times = [last_kc_time[k] for k in step_kcs if k in last_kc_time]
                if past_times:
                    spacing = t_sec[step] - max(past_times)
                    delta_kc[step] = np.log1p(max(0.0, spacing))
                else:
                    delta_kc[step] = -1.0  # First exposure indicator

                for k in step_kcs:
                    last_kc_time[k] = t_sec[step]

            # Causal running accuracy strictly up to step tau - 1 with Laplace smoothing
            running_acc = np.zeros(total_steps, dtype=np.float32)
            cum_correct = 0
            for step in range(total_steps):
                # At step 0, running_acc is the global prior
                running_acc[step] = (cum_correct + 2.0 * global_train_acc) / (step + 2.0)
                cum_correct += int(r_arr[step])

            # Window into segments of length max_seq_len (200)
            for start_idx in range(0, total_steps, self.max_seq_len):
                end_idx = min(start_idx + self.max_seq_len, total_steps)
                chunk_len = end_idx - start_idx

                chunk_q = np.zeros(self.max_seq_len, dtype=np.int64)
                chunk_r = np.zeros(self.max_seq_len, dtype=np.int64)
                chunk_mask = np.zeros(self.max_seq_len, dtype=np.bool_)
                chunk_lag = np.zeros(self.max_seq_len, dtype=np.float32)
                chunk_kc = np.zeros(self.max_seq_len, dtype=np.float32)
                chunk_batch = np.zeros(self.max_seq_len, dtype=np.float32)
                chunk_acc = np.zeros(self.max_seq_len, dtype=np.float32)

                chunk_q[:chunk_len] = q_arr[start_idx:end_idx]
                chunk_r[:chunk_len] = r_arr[start_idx:end_idx]
                chunk_mask[:chunk_len] = True
                chunk_lag[:chunk_len] = norm_delta_lag[start_idx:end_idx]
                chunk_kc[:chunk_len] = delta_kc[start_idx:end_idx]
                chunk_batch[:chunk_len] = batch_flags[start_idx:end_idx]
                chunk_acc[:chunk_len] = running_acc[start_idx:end_idx]

                all_q_seq.append(chunk_q)
                all_r_seq.append(chunk_r)
                all_mask.append(chunk_mask)
                all_delta_lag.append(chunk_lag)
                all_delta_kc.append(chunk_kc)
                all_batch_flags.append(chunk_batch)
                all_running_acc.append(chunk_acc)
                all_uids.append(uid)

        return {
            "q_seq": torch.tensor(np.array(all_q_seq), dtype=torch.int64),
            "r_seq": torch.tensor(np.array(all_r_seq), dtype=torch.int64),
            "seq_mask": torch.tensor(np.array(all_mask), dtype=torch.bool),
            "delta_t_lag": torch.tensor(np.array(all_delta_lag), dtype=torch.float32),
            "delta_t_kc": torch.tensor(np.array(all_delta_kc), dtype=torch.float32),
            "batch_flag": torch.tensor(np.array(all_batch_flags), dtype=torch.float32),
            "running_acc_prev": torch.tensor(np.array(all_running_acc), dtype=torch.float32),
            "uid_seq": torch.tensor(np.array(all_uids), dtype=torch.int64),
        }

    # =========================================================================
    # Step 9: Two-Tier Tensor Serialization
    # =========================================================================
    def build_leaf_kc_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build static question-to-leaf-KC index tensor of shape [7652, 6]
        and its corresponding binary mask [7652, 6].
        """
        if not self.questions_raw:
            self.load_metadata()

        num_questions = 7652
        max_leaves = 6
        leaf_kc_tensor = torch.zeros((num_questions, max_leaves), dtype=torch.int64)
        leaf_kc_mask = torch.zeros((num_questions, max_leaves), dtype=torch.bool)

        for qid in range(num_questions):
            q_info = self.questions_raw[str(qid)]
            leaves = set()
            for r in q_info.get("kc_routes", []):
                parts = r.split("----")
                if parts and parts[-1] in self.name_to_kcid:
                    leaves.add(self.name_to_kcid[parts[-1]])

            sorted_leaves = sorted(list(leaves))[:max_leaves]
            for idx, k in enumerate(sorted_leaves):
                leaf_kc_tensor[qid, idx] = k
                leaf_kc_mask[qid, idx] = True

        return leaf_kc_tensor, leaf_kc_mask

    def run_all(self, batch_size: int = 64) -> Dict[str, Any]:
        """
        Execute the entire end-to-end preprocessing pipeline and save Tier 1 and Tier 2 tensors.

        Args:
            batch_size: Mini-batch size for frozen text and vision backbones.

        Returns:
            Dictionary containing audit and verification summary metrics.
        """
        print("=== Step 1: Auditing integrity, canonical indexing & MD5 dedup ===")
        audit_info = self.step1_audit_and_canonical_indexing()
        print(f"  Valid images: {audit_info['valid_referenced_images']}, Orphans excluded: {audit_info['orphan_images_count']}")
        print(f"  Unique MD5 clusters: {audit_info['unique_md5_clusters']}")

        print("=== Step 2: Pedagogical text normalization & stem/sol partitioning ===")
        self.step2_pedagogical_text_normalization()

        print("=== Step 3: Visual standardization & modality masks ===")
        mod_masks = self.step3_visual_standardization_and_masks()
        print(f"  Questions with stem image: {int(mod_masks[:, 0].sum())}")
        print(f"  Questions with solution image: {int(mod_masks[:, 2].sum())}")

        print("=== Step 4: Dual-granularity multimodal feature extraction ===")
        print("  Extracting text features via RoBERTa...")
        text_features = self.extract_text_features(batch_size=batch_size)
        print("  Extracting vision features via CLIP ViT...")
        vision_features = self.extract_vision_features(batch_size=batch_size)

        print("=== Step 5: Constructing Question-Skill Hypergraph H and G ===")
        hypergraph_features = self.step5_build_hypergraph(alpha_anc=self.alpha_anc)

        print("=== Step 6: Parsing student interaction logs ===")
        seq_data = self.parse_interaction_sequences()
        students_by_fold = seq_data["students_by_fold"]
        test_students = seq_data["test_students"]

        print("=== Step 7: Estimating CTT Difficulty D_q per fold (Empirical Bayes) ===")
        ctt_folds, ctt_masks = self.step7_estimate_ctt_difficulty(students_by_fold)

        print("=== Step 8: Engineering temporal dynamics and causal student context ===")
        static_ctx = self.build_static_question_context()
        leaf_kcs, leaf_masks = self.build_leaf_kc_tensor()

        # Window sequences per fold and for test set
        fold_sequences = {}
        for f in range(5):
            print(f"  Windowing Fold {f} ({len(students_by_fold[f])} students)...")
            fold_sequences[f] = self.window_student_interactions(students_by_fold[f])

        print(f"  Windowing Test Set ({len(test_students)} students)...")
        test_sequences = self.window_student_interactions(test_students)

        print("=== Step 9: Serializing Two-Tier Tensors ===")
        # Tier 1: Static Item Bank
        static_item_bank = {
            "modality_masks": torch.tensor(mod_masks, dtype=torch.float32),
            "question_leaf_kcs": leaf_kcs,
            "question_kc_mask": leaf_masks,
            "question_static_ctx": static_ctx,
            # MG-MMKT pooled features
            "mg_v_txt_q": text_features["mg_v_txt_q"],
            "mg_v_txt_s": text_features["mg_v_txt_s"],
            "mg_v_img_q": vision_features["mg_v_img_q"],
            "mg_v_img_s": vision_features["mg_v_img_s"],
            # ACD-MMKT fine-grained features
            "acd_E_txt_q": text_features["acd_E_txt_q"],
            "acd_mask_txt_q": text_features["acd_mask_txt_q"],
            "acd_E_txt_s": text_features["acd_E_txt_s"],
            "acd_mask_txt_s": text_features["acd_mask_txt_s"],
            "acd_E_img_q": vision_features["acd_E_img_q"],
            "acd_E_img_s": vision_features["acd_E_img_s"],
            "acd_kc_init_emb": hypergraph_features["acd_kc_init_emb"],
            "acd_hypergraph_H": hypergraph_features["acd_hypergraph_H"],
            "acd_hypergraph_G": hypergraph_features["acd_hypergraph_G"],
            "acd_ctt_difficulty_folds": ctt_folds,
            "acd_ctt_valid_mask": ctt_masks,
        }

        tier1_path = os.path.join(self.processed_dir, "static_item_bank.pt")
        print(f"  Saving Tier 1 Static Item Bank to {tier1_path}...")
        torch.save(static_item_bank, tier1_path)

        # Tier 2: Dynamic Sequence Store
        for f in range(5):
            tier2_path = os.path.join(self.processed_dir, f"sequences_fold_{f}.pt")
            print(f"  Saving Tier 2 Fold {f} sequences to {tier2_path}...")
            torch.save(fold_sequences[f], tier2_path)

        test_path = os.path.join(self.processed_dir, "sequences_test.pt")
        print(f"  Saving Tier 2 Test sequences to {test_path}...")
        torch.save(test_sequences, test_path)

        # Save metadata summary JSON
        summary_metadata = {
            "num_questions": 7652,
            "num_kcs": 1175,
            "valid_disk_images": len(self.valid_images),
            "orphan_images": len(self.orphan_images),
            "unique_md5_images": len(self.unique_md5_list),
            "m_img_q_count": int(mod_masks[:, 0].sum()),
            "m_txt_s_count": int(mod_masks[:, 1].sum()),
            "m_img_s_count": int(mod_masks[:, 2].sum()),
            "num_students_train_val": sum(len(students_by_fold[f]) for f in range(5)),
            "num_students_test": len(test_students),
            "train_val_valid_interactions": int(sum(fold_sequences[f]["seq_mask"].sum().item() for f in range(5))),
            "test_valid_interactions": int(test_sequences["seq_mask"].sum().item()),
        }

        meta_path = os.path.join(self.processed_dir, "metadata_summary.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary_metadata, f, indent=2)

        print("=== Preprocessing Pipeline Completed Successfully! ===")
        print(f"  Summary: {summary_metadata}")
        return summary_metadata
