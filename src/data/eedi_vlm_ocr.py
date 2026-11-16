"""
Offline Local VLM/LLM Question OCR Normalization (Step 3) and Answer-Anchored Solution
Recovery (Step 4) for Eedi NeurIPS 2020 (Task 3 & 4).

Uses strictly local models via Ollama (qwen2.5vl:3b for multimodal slide OCR and
visual reasoning; qwen2.5:7b for text/symbolic mathematical reasoning) fused with
macOS Apple Vision OCR bounding-box hints. All outputs are checkpointed incrementally
to JSONL files over the 947 unique MD5 slides and synced to MD5 duplicates.
"""

import base64
import io
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_VLM_MODEL = "qwen2.5vl:3b"
DEFAULT_LLM_MODEL = "qwen2.5:7b"

LEAKAGE_PATTERNS = [
    re.compile(r"\bcorrect\s+answer\s*(is|:)", re.IGNORECASE),
    re.compile(r"\bofficial\s+key\b", re.IGNORECASE),
    re.compile(r"\bsolution\s*:", re.IGNORECASE),
    re.compile(r"\u00a9\s*eedi", re.IGNORECASE),
]


def normalize_math_text(text: str) -> str:
    """
    Recover JSON-corrupted TeX control escapes (\f -> \\f, \b -> \\b, \r -> \\r, \t -> \\t),
    apply Unicode NFKC normalization, convert LaTeX display/inline delimiters to '$...$',
    standardize empty fill-in boxes as '[BLANK]', and collapse redundant whitespace.
    """
    if not text:
        return ""
    out = str(text)
    # Recover TeX commands corrupted when JSON decoders interpret single \f, \b, \r, \t as control chars
    out = out.replace("\x0c", "\\f").replace("\x08", "\\b").replace("\r", "\\r")
    out = re.sub(r"\t(?=(ext|imes|heta|an|o|au|op|riangle|herefore))", r"\\t", out)
    out = unicodedata.normalize("NFKC", out)
    # Convert \( ... \) and \[ ... \] (including slight spaces like "\ )") to compact inline $...$
    out = re.sub(r"\\\(\s*(.*?)\s*\\\s*\)", r"$\1$", out, flags=re.DOTALL)
    out = re.sub(r"\\\[\s*(.*?)\s*\\\s*\]", r"$\1$", out, flags=re.DOTALL)
    # Standardize unicode blanks / blank boxes / underscore sequences (avoid subscripts like x_1) into [BLANK]
    out = re.sub(r"[□▢◻⬜■▪]+", " [BLANK] ", out)
    out = re.sub(r"\\boxed\{\s*\}", " [BLANK] ", out)
    out = re.sub(r"(?<![A-Za-z0-9_\\])_+(?![A-Za-z0-9_{])", " [BLANK] ", out)
    # Remove any accidental Eedi slide chrome artifacts
    out = re.sub(r"(\u00a9|@)?\s*Eedi\s*20\d{2}", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def strip_option_prefix(opt_text: str, letter: str) -> str:
    """
    Strip accidental leading badge marker ('A. ', 'B) ', 'C: ') from an extracted option
    string unless the entire value is just the reference letter itself (e.g. 'A').
    """
    cleaned = normalize_math_text(opt_text)
    if cleaned.upper() == letter:
        return cleaned.upper()
    # Keep normalized [VISUAL_DIAGRAM_OPTION_X] prefixes canonical
    vis_match = re.match(r"^\[VISUAL_DIAGRAM_(?:OPTION_)?([A-D0-3])\]\s*(.*)$", cleaned, re.IGNORECASE)
    if vis_match:
        desc = vis_match.group(2).strip()
        return f"[VISUAL_DIAGRAM_OPTION_{letter}] {desc}".strip()
    cleaned = re.sub(rf"^{letter}[\.\)\:]\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def encode_clean_slide_jpeg_b64(
    img_path: Path,
    target_width: int = 504,
) -> str:
    """
    Standardize a raw Eedi slide to 640x480, strip top header and bottom footer
    chrome (y in [43, 448]), resize cleanly for Qwen2.5-VL patch alignment, and
    encode as base64 JPEG.
    """
    im = Image.open(img_path).convert("RGB")
    if im.size != (640, 480):
        im = im.resize((640, 480), Image.Resampling.LANCZOS)
    crop = im.crop((0, 43, 640, 448))
    h_new = max(28, int(round(crop.height * (target_width / 640.0))))
    crop = crop.resize((target_width, h_new), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_ollama_json(
    model: str,
    prompt: str,
    img_b64: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    num_predict: int = 380,
    timeout: int = 120,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Invoke local Ollama endpoint requesting strict JSON output with automatic retries
    and lightweight JSON recovery if trailing text or unescaped backslashes occur.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "60m",
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": 2048},
    }
    if img_b64 is not None:
        payload["images"] = [img_b64]

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw_text = body.get("response", "").strip()
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                # Extract first balanced JSON object and repair single-backslash LaTeX escapes if needed
                m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
                if m:
                    candidate = m.group(0)
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        repaired = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r"\\\\", candidate)
                        return json.loads(repaired)
                raise
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"Ollama JSON call failed after {retries} attempts for model={model}: {last_err}")


def _extract_av_content_hints(ocr_items: List[Dict[str, Any]]) -> List[str]:
    """Return Apple Vision / EasyOCR lines located cleanly inside the main content area (excluding header/footer)."""
    filtered = [
        (it["bbox"][1], it["bbox"][0], normalize_math_text(it["text"]))
        for it in ocr_items
        if 0.085 < float(it["bbox"][1]) < 0.935 and str(it.get("text", "")).strip()
    ]
    filtered.sort(key=lambda t: (round(t[0] * 25), t[1]))
    return [t[2] for t in filtered if t[2] not in ("A", "B", "C", "D")]


def _extract_region_aware_ocr_hints(
    ocr_items: List[Dict[str, Any]],
    layout_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Route OCR bounding boxes deterministically into RoI_stem and RoI_opt_{A..D}
    using calibrated 640x480 slide boxes while filtering out badge letters.
    """
    badges = layout_meta.get("badges", {})
    opt_bboxes = layout_meta.get("opt_bboxes", {})
    layout_type = layout_meta.get("layout_type", "horiz_below")

    stem_items: List[Tuple[float, float, str]] = []
    opt_items: Dict[str, List[Tuple[float, float, str]]] = {k: [] for k in "ABCD"}

    for it in ocr_items:
        bx, by, bw, bh = [float(v) for v in it.get("bbox", [0, 0, 0, 0])]
        txt = normalize_math_text(str(it.get("text", "")))
        if not txt or not (0.085 < by < 0.935):
            continue

        cx = (bx + 0.5 * bw) * 640.0
        cy = (by + 0.5 * bh) * 480.0

        # Skip standalone badge letter artifacts centered inside circular option badges
        in_badge = False
        for ltr, b in badges.items():
            if (b["x0"] - 3) <= cx <= (b["x1"] + 3) and (b["y0"] - 3) <= cy <= (b["y1"] + 3):
                if txt.upper() == ltr:
                    in_badge = True
                break
        if in_badge:
            continue

        # Assign to matching option box when not in diagram-embedded mode
        matched_opt = None
        if layout_type not in ("horiz_ref", "grid_embedded"):
            for ltr in "ABCD":
                ob = opt_bboxes.get(ltr)
                if ob and (ob[0] - 6) <= cx <= (ob[2] + 6) and (ob[1] - 6) <= cy <= (ob[3] + 6):
                    matched_opt = ltr
                    break

        if matched_opt is not None:
            opt_items[matched_opt].append((by, bx, txt))
        else:
            stem_items.append((by, bx, txt))

    stem_items.sort(key=lambda t: (round(t[0] * 25), t[1]))
    opt_hints: Dict[str, List[str]] = {}
    for ltr in "ABCD":
        opt_items[ltr].sort(key=lambda t: (round(t[0] * 25), t[1]))
        opt_hints[ltr] = [t[2] for t in opt_items[ltr]]

    return {
        "stem_hints": [t[2] for t in stem_items],
        "opt_hints": opt_hints,
    }


def _fallback_ocr_parse(
    qid: int,
    ocr_items: List[Dict[str, Any]],
    layout_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic region-aware fallback parser using spatial OCR bboxes if VLM fails."""
    routed = _extract_region_aware_ocr_hints(ocr_items, layout_meta)
    stem_hints = routed["stem_hints"]
    opt_hints = routed["opt_hints"]
    opt_cv = layout_meta.get("opt_cv_visual", {})

    stem = " ".join(stem_hints) if stem_hints else f"Eedi Question {qid}"
    opts: Dict[str, str] = {}
    for ltr in "ABCD":
        joined = " ".join(opt_hints[ltr]).strip()
        if joined:
            opts[ltr] = joined
        elif opt_cv.get(ltr, False):
            opts[ltr] = f"[VISUAL_DIAGRAM_OPTION_{ltr}] Visual option {ltr}"
        else:
            opts[ltr] = f"Option {ltr}"

    has_vis_opts = sum(1 for k in "ABCD" if opts[k].startswith("[VISUAL_DIAGRAM_OPTION_")) >= 2
    return {
        "stem_text": normalize_math_text(stem),
        "options": opts,
        "stem_has_diagram": bool(
            layout_meta.get("stem_struct_density", 0.0) >= 0.006
            or layout_meta.get("stem_sat_density", 0.0) >= 0.0008
        ),
        "options_have_diagrams": has_vis_opts,
        "stem_visual_summary": "",
        "fallback_used": True,
    }


def compute_hybrid_has_stem_diagram(
    vlm_stem_has_diagram: bool,
    vlm_opt_has_diagrams: bool,
    layout_meta: Dict[str, Any],
    opts_norm: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Combine CV RoI_stem + visual option signals with VLM semantic visual judgment to decide
    whether the question genuinely contains diagrammatic/non-textual content (m_img^q).
    """
    layout_type = layout_meta.get("layout_type", "horiz_below")
    rho_stem = float(layout_meta.get("stem_struct_density", 0.0))
    sat_density = float(layout_meta.get("stem_sat_density", 0.0))
    has_long_stroke = bool(layout_meta.get("stem_has_long_stroke", False))
    only_small_box = bool(layout_meta.get("stem_only_small_box", False))
    opt_cv_count = sum(1 for v in layout_meta.get("opt_cv_visual", {}).values() if v)
    has_vis_opt_tags = bool(
        opts_norm and sum(1 for v in opts_norm.values() if str(v).startswith("[VISUAL_DIAGRAM_OPTION_")) >= 2
    )
    has_visual_options = has_vis_opt_tags or (vlm_opt_has_diagrams and opt_cv_count >= 2)

    if layout_type in ("horiz_ref", "grid_embedded"):
        return True
    if has_visual_options:
        return True
    if rho_stem >= 0.008 and not only_small_box:
        return True
    if has_long_stroke and not only_small_box and rho_stem >= 0.0020:
        return True
    # Catch low-ink or colored dot visual glyphs inside RoI_stem (e.g. Mayan Q55 or dot grids Q77)
    if vlm_stem_has_diagram and not only_small_box and (rho_stem >= 0.0020 or sat_density >= 0.0008):
        return True
    return False


def format_question_text_T_q(
    stem_text: str,
    options: Dict[str, str],
    kc_prefix: str = "",
    trim_tokens_fn: Optional[Callable[[str], int]] = None,
    max_tokens: int = 124,
) -> str:
    """
    Format leak-free normalized question text T_q combining curriculum prefix,
    stem, and four options with explicit [SEP] delimiters matching Step 3.2.
    """
    stem_clean = normalize_math_text(stem_text)
    opts_clean = {k: strip_option_prefix(options.get(k, ""), k) for k in "ABCD"}
    for k in "ABCD":
        if not opts_clean[k]:
            opts_clean[k] = f"[VISUAL_DIAGRAM_OPTION_{k}]"
        for pat in LEAKAGE_PATTERNS:
            opts_clean[k] = pat.sub("", opts_clean[k]).strip()
    for pat in LEAKAGE_PATTERNS:
        stem_clean = pat.sub("", stem_clean).strip()

    prefix_str = f"{kc_prefix.strip()} " if kc_prefix.strip() else ""
    opts_block = (
        f"Option A: {opts_clean['A']} | Option B: {opts_clean['B']} "
        f"| Option C: {opts_clean['C']} | Option D: {opts_clean['D']}"
    )
    formatted = f"{prefix_str}Question: {stem_clean} [SEP] {opts_block}"

    if trim_tokens_fn is not None and trim_tokens_fn(formatted) > max_tokens:
        short_opts = {
            k: (re.match(r"^(\[VISUAL_DIAGRAM_OPTION_[A-D]\])", opts_clean[k]).group(1)
                if opts_clean[k].startswith("[VISUAL_DIAGRAM_OPTION_") else opts_clean[k][:40])
            for k in "ABCD"
        }
        opts_block = (
            f"Option A: {short_opts['A']} | Option B: {short_opts['B']} "
            f"| Option C: {short_opts['C']} | Option D: {short_opts['D']}"
        )
        formatted = f"{prefix_str}Question: {stem_clean} [SEP] {opts_block}"
    return normalize_math_text(formatted)


def _check_stem_word_coverage(stem_text: str, hints: List[str]) -> bool:
    """Verify VLM stem_text didn't silently drop English question words present in OCR hints."""
    if not hints:
        return True
    # Only enforce on clear multi-letter English question words in top hints (outside A..D options)
    stem_lower = stem_text.lower()
    keywords = {"what", "which", "find", "calculate", "work", "how", "complete", "true", "false", "correct", "incorrect", "value", "area", "angle", "volume", "length", "missing", "mean", "vector", "shape", "graph", "equation", "expression"}
    hint_words = set()
    for h in hints[:4]:
        for w in re.findall(r"[a-z]{3,}", h.lower()):
            if w in keywords:
                hint_words.add(w)
    if not hint_words:
        return True
    matched = sum(1 for w in hint_words if w in stem_lower)
    return matched >= max(1, len(hint_words) // 2)


def run_step3_ocr_extraction(
    unique_rep_qids: List[int],
    dup_to_rep: Dict[int, int],
    images_dir: Path,
    ocr_items_by_qid: Dict[int, List[Dict[str, Any]]],
    layout_inventory: Dict[int, Dict[str, Any]],
    out_jsonl_path: Path,
    vlm_model: str = DEFAULT_VLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    trim_tokens_fn: Optional[Callable[[str], int]] = None,
    kc_prefix_by_qid: Optional[Dict[int, str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Execute Step 3 hybrid VLM + region-aware OCR extraction across unique slides with
    incremental JSONL checkpointing, then propagate verified records to MD5 duplicates.
    """
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    cached: Dict[int, Dict[str, Any]] = {}
    if out_jsonl_path.exists():
        with open(out_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached[int(rec["qid"])] = rec

    remaining = [q for q in unique_rep_qids if q not in cached]
    if remaining:
        with open(out_jsonl_path, "a", encoding="utf-8") as fout:
            for idx, qid in enumerate(remaining, start=1):
                routed = _extract_region_aware_ocr_hints(ocr_items_by_qid.get(qid, []), layout_inventory[qid])
                hints = routed["stem_hints"]
                opt_hints = routed["opt_hints"]
                layout_meta = layout_inventory[qid]
                img_path = images_dir / f"{qid}.jpg"
                b64 = encode_clean_slide_jpeg_b64(img_path, target_width=504)
                prompt = (
                    f"Stem OCR hints: {json.dumps(hints)}.\n"
                    f"Region Option OCR hints: {json.dumps(opt_hints)}.\n"
                    "Inspect this middle-school math MCQ slide carefully and return ONLY valid JSON with exact keys:\n"
                    '{"stem_text": str, "options": {"A": str, "B": str, "C": str, "D": str}, '
                    '"stem_has_diagram": bool, "options_have_diagrams": bool, "stem_visual_summary": str}.\n'
                    "Rules:\n"
                    "1. In stem_text, copy ALL complete English question sentence(s) word-for-word from the slide and OCR hints "
                    "(never omit words like 'What does the vector ... mean?'), formatting stacked fractions, powers, roots, and column vectors "
                    "in inline LaTeX ($\\frac{a}{b}$, $\\begin{pmatrix}-5\\\\2\\end{pmatrix}$, $(-5)^2$). Represent any empty blank box or underscore gap as [BLANK]. Do NOT include answer keys.\n"
                    "2. If an option stacks multiple numbers vertically for multiple blanks (e.g. '18 / 18 / 3'), list ALL numbers comma-separated ('18, 18, 3'). If options A/B/C/D are purely visual diagrams OR letters labeling points/regions on the diagram, set value to \"[VISUAL_DIAGRAM_OPTION_A] <brief description>\" (with matching A/B/C/D).\n"
                    "3. Set stem_has_diagram = true if the stem contains ANY geometric figure, diagram, graph, chart, number line, Venn diagram, grid, table, or visual symbol (e.g., Mayan dots) needed to solve the question, and false for plain text/equations (including stacked fractions, column vectors, empty blank boxes, or equation callout boxes).\n"
                    "4. Set stem_visual_summary to \"\" if stem_has_diagram is false, else a concise factual 1-2 sentence description including any key diagram numbers, labels, axes, coordinates, or shapes."
                )
                try:
                    raw = call_ollama_json(
                        model=vlm_model,
                        prompt=prompt,
                        img_b64=b64,
                        ollama_url=ollama_url,
                        num_predict=360,
                    )
                    stem_raw = str(raw.get("stem_text", "")).strip()
                    if not _check_stem_word_coverage(stem_raw, hints):
                        raw_retry = call_ollama_json(
                            model=vlm_model,
                            prompt=prompt + " STRICT REMINDER: Keep ALL English question words around any math/vector formula!",
                            img_b64=b64,
                            ollama_url=ollama_url,
                            num_predict=360,
                        )
                        cand_stem = str(raw_retry.get("stem_text", "")).strip()
                        if _check_stem_word_coverage(cand_stem, hints):
                            raw = raw_retry
                            stem_raw = cand_stem
                    opts_raw = raw.get("options", {})
                    if not stem_raw or not isinstance(opts_raw, dict) or not all(k in opts_raw for k in "ABCD"):
                        raise ValueError("Incomplete VLM JSON structure")
                    opts_norm = {
                        k: strip_option_prefix(str(opts_raw.get(k, "")), k)
                        for k in "ABCD"
                    }
                    # Ensure diagram reference / blanked visual options never collapse to raw "BLANK"
                    for k in "ABCD":
                        if layout_meta.get("layout_type") == "horiz_ref" and (
                            opts_norm[k].upper() in (k, "BLANK", "[BLANK]", "")
                        ):
                            opts_norm[k] = f"[VISUAL_DIAGRAM_OPTION_{k}] Point or region {k} on diagram"
                        elif bool(raw.get("options_have_diagrams", False)) and (
                            not opts_norm[k].startswith("[VISUAL_DIAGRAM_OPTION_")
                            and (len(opts_norm[k]) <= 2 or opts_norm[k].upper() in ("BLANK", "[BLANK]"))
                        ):
                            opts_norm[k] = f"[VISUAL_DIAGRAM_OPTION_{k}] Visual option {k}"

                    vlm_has_diag = bool(raw.get("stem_has_diagram", False))
                    vlm_opt_diag = bool(raw.get("options_have_diagrams", False))
                    fallback_used = False
                    vis_sum = normalize_math_text(str(raw.get("stem_visual_summary", "")))
                except Exception:
                    fb = _fallback_ocr_parse(qid, ocr_items_by_qid.get(qid, []), layout_meta)
                    stem_raw = fb["stem_text"]
                    opts_norm = {k: strip_option_prefix(fb["options"][k], k) for k in "ABCD"}
                    vlm_has_diag = fb["stem_has_diagram"]
                    vlm_opt_diag = fb["options_have_diagrams"]
                    vis_sum = fb["stem_visual_summary"]
                    fallback_used = True

                final_has_diag = compute_hybrid_has_stem_diagram(vlm_has_diag, vlm_opt_diag, layout_meta, opts_norm=opts_norm)
                stem_norm = normalize_math_text(stem_raw)
                kc_prefix = kc_prefix_by_qid.get(qid, "") if kc_prefix_by_qid else ""
                text_q = format_question_text_T_q(
                    stem_norm, opts_norm, kc_prefix=kc_prefix, trim_tokens_fn=trim_tokens_fn
                )

                rec = {
                    "qid": qid,
                    "stem_text": stem_norm,
                    "options": opts_norm,
                    "text_q": text_q,
                    "vlm_stem_has_diagram": vlm_has_diag,
                    "options_have_diagrams": vlm_opt_diag,
                    "has_stem_diagram": final_has_diag,
                    "stem_visual_summary": vis_sum if final_has_diag else "",
                    "fallback_used": fallback_used,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                cached[qid] = rec
                if idx % 25 == 0 or idx == len(remaining):
                    print(f"[Step 3 OCR] {idx}/{len(remaining)} unique slides processed (total cached={len(cached)})", flush=True)

    # Re-apply deterministic diagram classification and curriculum-prefixed T_q formatting
    # across all target items, and rewrite out_jsonl_path if cached unique entries changed.
    cache_dirty = False
    for qid in unique_rep_qids:
        if qid not in cached:
            continue
        rec = cached[qid]
        opts_norm = {k: strip_option_prefix(str(rec["options"].get(k, "")), k) for k in "ABCD"}
        stem_norm = normalize_math_text(rec["stem_text"])
        final_has_diag = compute_hybrid_has_stem_diagram(
            bool(rec.get("vlm_stem_has_diagram", False)),
            bool(rec.get("options_have_diagrams", False)),
            layout_inventory[qid],
            opts_norm=opts_norm,
        )
        kc_prefix = kc_prefix_by_qid.get(qid, "") if kc_prefix_by_qid else ""
        text_q = format_question_text_T_q(
            stem_norm, opts_norm, kc_prefix=kc_prefix, trim_tokens_fn=trim_tokens_fn
        )
        vis_sum = rec.get("stem_visual_summary", "") if final_has_diag else ""
        if (
            rec.get("stem_text") != stem_norm
            or rec.get("options") != opts_norm
            or bool(rec.get("has_stem_diagram", False)) != final_has_diag
            or rec.get("text_q") != text_q
            or rec.get("stem_visual_summary", "") != vis_sum
        ):
            cache_dirty = True
            rec["stem_text"] = stem_norm
            rec["options"] = opts_norm
            rec["has_stem_diagram"] = final_has_diag
            rec["text_q"] = text_q
            rec["stem_visual_summary"] = vis_sum

    if cache_dirty:
        tmp_path = out_jsonl_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fout:
            for qid in sorted(cached.keys()):
                fout.write(json.dumps(cached[qid], ensure_ascii=False) + "\n")
        tmp_path.replace(out_jsonl_path)

    target_qids = sorted(dup_to_rep.keys()) if dup_to_rep else sorted(set(unique_rep_qids))
    full_ocr: Dict[int, Dict[str, Any]] = {}
    for qid in target_qids:
        rep = dup_to_rep.get(qid, qid)
        item = dict(cached[rep])
        item["qid"] = qid
        full_ocr[qid] = item
    return full_ocr


def _format_solution_text(
    correct_letter: str,
    correct_opt: str,
    steps: str,
    misconceptions: str,
    token_len_fn: Optional[Callable[[str], int]] = None,
    max_tokens: int = 156,
) -> str:
    """Format answer-anchored 3-part solution T_s^q bounded safely inside L_s matching Step 4.2."""
    steps_clean = normalize_math_text(steps)
    misc_clean = normalize_math_text(misconceptions)
    text_s = f"Standard Answer: ({correct_letter}) {correct_opt} [SEP] Solution Analysis: {steps_clean} | Misconception Note: {misc_clean}"
    if token_len_fn is not None and token_len_fn(text_s) > max_tokens:
        short_s = text_s
        sentences = re.split(r"(?<=[.!?])\s+", misc_clean)
        while len(sentences) > 1 and token_len_fn(short_s) > max_tokens:
            sentences.pop()
            misc_sub = " ".join(sentences)
            short_s = f"Standard Answer: ({correct_letter}) {correct_opt} [SEP] Solution Analysis: {steps_clean} | Misconception Note: {misc_sub}"
        if token_len_fn(short_s) > max_tokens:
            words = short_s.split()
            while len(words) > 20 and token_len_fn(" ".join(words)) > max_tokens:
                words.pop()
            short_s = " ".join(words)
        text_s = short_s
    return normalize_math_text(text_s)


def _verify_solution_anchor(steps: str, misconceptions: str, correct_letter: str, correct_opt: str = "") -> bool:
    """Verify the solution aligns with official key and never claims the verified answer is wrong."""
    combined = f"{steps} {misconceptions}"
    if len(combined.split()) < 12:
        return False
    # Never claim another option letter is the correct answer
    for other in "ABCD":
        if other == correct_letter:
            continue
        if re.search(rf"\boption\s+{other}\s+is\s+(the\s+)?correct\b", combined, flags=re.IGNORECASE):
            return False
        if re.search(rf"\banswer\s+is\s+(\(?{other}\)?)\b", combined, flags=re.IGNORECASE):
            return False
    # Never contradict the verified correct letter or numeric option value
    if re.search(rf"\boption\s+{correct_letter}\s+is\s+(incorrect|wrong|false|not\s+correct)\b", combined, flags=re.IGNORECASE):
        return False
    c_val = re.sub(r"^\$|\$$", "", correct_opt).strip()
    if c_val and len(c_val) <= 12 and not c_val.startswith("[VISUAL"):
        esc = re.escape(c_val)
        if re.search(rf"\b{esc}\s+is\s+(incorrect|wrong|false)\b", combined, flags=re.IGNORECASE):
            return False
    return True


def run_step4_solution_generation(
    unique_rep_qids: List[int],
    dup_to_rep: Dict[int, int],
    images_dir: Path,
    ocr_by_qid: Dict[int, Dict[str, Any]],
    correct_idx_by_qid: Dict[int, int],
    kc_names_by_qid: Dict[int, str],
    out_jsonl_path: Path,
    vlm_model: str = DEFAULT_VLM_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    token_len_fn: Optional[Callable[[str], int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Execute Step 4 Answer-Anchored Solution Generation across the 947 unique slides.
    Uses VLM (with slide image) on visual items first, then text LLM on non-visual items,
    minimizing Ollama model swap overhead while guaranteeing 100% key consistency.
    """
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    cached: Dict[int, Dict[str, Any]] = {}
    if out_jsonl_path.exists():
        with open(out_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached[int(rec["qid"])] = rec

    remaining = [q for q in unique_rep_qids if q not in cached]
    # Process visual items first with VLM (with slide image) and text-only items next with LLM
    # to eliminate repeated Ollama model swaps while maximizing multimodal diagram accuracy.
    if remaining:
        visual_qids = [
            q for q in remaining
            if ocr_by_qid[q].get("has_stem_diagram", False) or ocr_by_qid[q].get("options_have_diagrams", False)
        ]
        text_qids = [q for q in remaining if q not in set(visual_qids)]
        ordered_remaining = visual_qids + text_qids

        with open(out_jsonl_path, "a", encoding="utf-8") as fout:
            for idx, qid in enumerate(ordered_remaining, start=1):
                qrec = ocr_by_qid[qid]
                c_idx = int(correct_idx_by_qid[qid])
                c_letter = "ABCD"[c_idx]
                opts = qrec["options"]
                c_opt = opts[c_letter]
                kc_str = kc_names_by_qid.get(qid, "Mathematics")
                vis_summary = qrec.get("stem_visual_summary", "").strip()
                vis_block = f"Visual Diagram Context: {vis_summary}\n" if vis_summary else ""
                is_visual_item = bool(qrec.get("has_stem_diagram", False) or qrec.get("options_have_diagrams", False))

                prompt = (
                    f"Curriculum Skill: {kc_str}\n"
                    f"Question: {qrec['stem_text']}\n"
                    f"{vis_block}"
                    f"Options: A: {opts['A']}; B: {opts['B']}; C: {opts['C']}; D: {opts['D']}\n"
                    f"Verified Official Answer Key: {c_letter} ({c_opt}).\n"
                    "Explain clearly why this verified option is mathematically correct and what common student "
                    "misconceptions lead to the distractors. Return ONLY valid JSON with exact keys:\n"
                    '{"steps": str, "misconceptions": str}\n'
                    "Rules:\n"
                    "1. Use inline LaTeX ($...$) for formulas and end your 'steps' with the explicit conclusion matching option "
                    f"{c_letter} ({c_opt}). Assume {c_letter} ({c_opt}) IS 100% mathematically true; NEVER state that {c_letter} or {c_opt} is incorrect.\n"
                    "2. Match step granularity to actual problem complexity (concise 1-step for direct recall, explicit intermediate steps for multi-step problems; 45 to 95 words total)."
                )

                slide_b64 = encode_clean_slide_jpeg_b64(images_dir / f"{qid}.jpg") if is_visual_item else None
                model_used = vlm_model if is_visual_item else llm_model
                steps_txt = ""
                misc_txt = ""
                verified = False

                attempts_cfg = (
                    [(vlm_model, slide_b64, ""), (llm_model, None, f" STRICT REQUIREMENT: Confirm mathematically why option {c_letter} ({c_opt}) is right.")]
                    if is_visual_item
                    else [(llm_model, None, ""), (llm_model, None, f" STRICT REQUIREMENT: Confirm mathematically why option {c_letter} ({c_opt}) is right.")]
                )
                for m_name, m_img, extra_note in attempts_cfg:
                    try:
                        raw = call_ollama_json(
                            model=m_name,
                            prompt=prompt + extra_note,
                            img_b64=m_img,
                            ollama_url=ollama_url,
                            num_predict=240,
                        )
                        steps_cand = normalize_math_text(str(raw.get("steps", "")))
                        misc_cand = normalize_math_text(str(raw.get("misconceptions", "")))
                        if steps_cand and misc_cand and _verify_solution_anchor(steps_cand, misc_cand, c_letter, c_opt):
                            steps_txt, misc_txt = steps_cand, misc_cand
                            model_used = m_name
                            verified = True
                            break
                    except Exception:
                        continue

                if not verified:
                    # Deterministic verified backup anchored cleanly to KC & official key if JSON retry failed
                    steps_txt = (
                        f"Using {kc_str} on '{qrec['stem_text']}', evaluating the conditions and relationships "
                        f"identifies option {c_letter} ({c_opt}) as the valid result."
                    )
                    other_opts = ", ".join(f"{ltr} ({opts[ltr]})" for ltr in "ABCD" if ltr != c_letter)
                    misc_txt = (
                        f"Distractors {other_opts} typically arise from sign, fraction/denominator, operation-order, "
                        f"or geometric property misidentifications rather than applying {kc_str} accurately."
                    )

                text_s = _format_solution_text(
                    c_letter, c_opt, steps_txt, misc_txt, token_len_fn=token_len_fn
                )
                rec = {
                    "qid": qid,
                    "correct_idx": c_idx,
                    "correct_letter": c_letter,
                    "model_used": model_used,
                    "verified_llm": verified,
                    "steps": steps_txt,
                    "misconceptions": misc_txt,
                    "text_s": text_s,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                cached[qid] = rec
                if idx % 25 == 0 or idx == len(remaining):
                    print(f"[Step 4 Solution] {idx}/{len(remaining)} items processed (total cached={len(cached)})", flush=True)

    target_qids = sorted(dup_to_rep.keys()) if dup_to_rep else sorted(set(unique_rep_qids))
    full_sol: Dict[int, Dict[str, Any]] = {}
    for qid in target_qids:
        rep = dup_to_rep.get(qid, qid)
        item = dict(cached[rep])
        item["qid"] = qid
        full_sol[qid] = item
    return full_sol
