"""
Computer Vision Slide Anatomy Parser and RoI Extractor for Eedi NeurIPS 2020.

Implements deterministic, badge-anchored layout analysis over standardized 640x480
Eedi question slides:
1. Header/footer chrome boundary stripping (header y in [0, 43], footer y in [447, 480]).
2. Top-right calculator icon template classification (slashed no-calc vs blank/broken).
3. Color+shape connected component detection for all four circular badges A, B, C, D.
4. Layout sub-variant classification across horizontal (below-options, above-options,
   reference-only letters), vertical (right-column, left-column), and 2x2 grid
   (standard quadrants, diagram-embedded).
5. Clean badge-free Region of Interest cropping for RoI_stem and RoI_opt_{A..D},
   plus OCR-masked structural ink density and graphic primitive heuristics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


STD_WIDTH = 640
STD_HEIGHT = 480
HEADER_Y_END = 44
FOOTER_Y_START = 446

# Calibrated RGB clusters for Eedi option badges A (sky blue), B (purple/orange),
# C (mint green/pink), and D (rose pink/purple).
BADGE_COLOR_PROFILES: Dict[str, List[Tuple[int, int, int]]] = {
    "A": [(115, 196, 226), (132, 210, 254), (110, 190, 225)],
    "B": [(146, 156, 227), (140, 150, 225), (254, 176, 102), (250, 170, 95)],
    "C": [(148, 223, 130), (140, 218, 125), (254, 180, 206)],
    "D": [(229, 139, 167), (225, 135, 165), (160, 160, 254)],
}


def load_standardized_slide(img_path: Path) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Load raw Eedi slide JPEG and resize with Lanczos resampling to canonical 640x480 RGB.

    Returns:
        Tuple of (standardized 640x480 PIL RGB image, original (width, height)).
    """
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size
    if orig_size != (STD_WIDTH, STD_HEIGHT):
        img = img.resize((STD_WIDTH, STD_HEIGHT), Image.Resampling.LANCZOS)
    return img, orig_size


def extract_no_calc_template(ref_img_path: Path) -> np.ndarray:
    """
    Build background-subtracted reference patch of the top-right no-calculator icon
    from a canonical slide (e.g., 0.jpg) for lighting/shade-invariant matching.
    """
    img, _ = load_standardized_slide(ref_img_path)
    arr = np.asarray(img, dtype=np.float32)
    bg_rgb = np.median(arr[8:38, 555:595], axis=(0, 1))
    return arr[8:38, 602:634] - bg_rgb


def classify_calculator_header(
    arr: np.ndarray,
    ref_diff_template: Optional[np.ndarray] = None,
) -> Tuple[str, float]:
    """
    Classify the top-right header icon zone [8:38, 602:634] on a 640x480 slide.

    Returns:
        Tuple of (status string, calculator_allowed_flag in {0.0, 1.0}).
    """
    patch = arr[8:38, 602:634].astype(np.float32)
    std = float(patch.std())
    if std < 5.0:
        return "absent_blank", 0.0

    if ref_diff_template is not None:
        bg_rgb = np.median(arr[8:38, 555:595].astype(np.float32), axis=(0, 1))
        cur_diff = patch - bg_rgb
        mae = float(np.mean(np.abs(cur_diff - ref_diff_template)))
        if mae < 12.0:
            return "no_calculator_slashed", 0.0

    # Check if diagonal white slash stroke exists across white foreground outline
    bright = patch.mean(axis=2) > 225
    diag_pixels = sum(1 for k in range(30) if bright[min(29, k), min(31, k + 1)])
    anti_diag = sum(1 for k in range(30) if bright[min(29, k), max(0, 30 - k)])
    if max(diag_pixels, anti_diag) >= 10:
        return "no_calculator_slashed", 0.0

    return "broken_or_header_artifact", 0.0


def detect_option_badges(arr: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Detect centroids and tight bounding boxes for option badges A, B, C, D.

    Restricts candidate pixels strictly to the content vertical band y in [45, 450]
    and scores connected components by compact square-like badge geometry.
    """
    h_img, w_img, _ = arr.shape
    work = arr.astype(np.int32)
    results: Dict[str, Dict[str, Any]] = {}

    for letter, targets in BADGE_COLOR_PROFILES.items():
        mask = np.zeros((h_img, w_img), dtype=bool)
        for tr, tg, tb in targets:
            dist = (
                np.abs(work[:, :, 0] - tr)
                + np.abs(work[:, :, 1] - tg)
                + np.abs(work[:, :, 2] - tb)
            )
            mask |= dist < 65

        mask[:45, :] = False
        mask[450:, :] = False
        labeled, _ = ndimage.label(mask)
        slices = ndimage.find_objects(labeled)

        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for cid, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            y0, y1 = int(sl[0].start), int(sl[0].stop - 1)
            x0, x1 = int(sl[1].start), int(sl[1].stop - 1)
            sub = labeled[sl] == cid
            area = int(sub.sum())
            if area < 120 or area > 4000:
                continue
            bw = x1 - x0 + 1
            bh = y1 - y0 + 1
            aspect = bw / max(1, bh)
            if not (0.55 <= aspect <= 1.8 and 12 <= bw <= 85 and 12 <= bh <= 85):
                continue
            if area / float(bw * bh) < 0.40:
                continue

            score = area - 50.0 * abs(aspect - 1.0)
            if score > best_score:
                best_score = score
                ys_loc, xs_loc = np.where(sub)
                best = {
                    "cx": float(x0 + xs_loc.mean()),
                    "cy": float(y0 + ys_loc.mean()),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "area": area,
                }

        if best is None:
            raise RuntimeError(f"Failed to detect badge '{letter}' in Eedi slide.")
        results[letter] = best

    return results


def _tight_ink_bbox(
    gray: np.ndarray,
    box: Tuple[int, int, int, int],
    pad: int = 6,
    thresh: int = 235,
    min_w: int = 16,
    min_h: int = 16,
) -> Tuple[int, int, int, int]:
    """Trim uniform white borders inside a candidate crop while preserving min size."""
    x0, y0, x1, y1 = box
    x0 = max(0, x0)
    y0 = max(HEADER_Y_END, y0)
    x1 = min(STD_WIDTH, x1)
    y1 = min(FOOTER_Y_START, y1)
    if x1 <= x0 or y1 <= y0:
        return (0, HEADER_Y_END, STD_WIDTH, FOOTER_Y_START)

    sub = gray[y0:y1, x0:x1] < thresh
    ys, xs = np.where(sub)
    if len(ys) == 0:
        return (x0, y0, x1, y1)

    tx0 = max(x0, int(x0 + xs.min() - pad))
    ty0 = max(y0, int(y0 + ys.min() - pad))
    tx1 = min(x1, int(x0 + xs.max() + pad + 1))
    ty1 = min(y1, int(y0 + ys.max() + pad + 1))

    # Expand evenly to avoid degenerate ultra-thin strips
    if tx1 - tx0 < min_w:
        cx = (tx0 + tx1) // 2
        tx0 = max(x0, cx - min_w // 2)
        tx1 = min(x1, tx0 + min_w)
    if ty1 - ty0 < min_h:
        cy = (ty0 + ty1) // 2
        ty0 = max(y0, cy - min_h // 2)
        ty1 = min(y1, ty0 + min_h)
    return (tx0, ty0, tx1, ty1)


def _find_above_badge_option_band(
    ink_mask: np.ndarray,
    y_top_b: int,
    min_gap_rows: int = 10,
) -> Optional[int]:
    """
    Locate the top row separating stem content from an above-badge option row.

    First skips any narrow whitespace padding directly above the badges, traverses
    upward through the option row ink, and returns the midpoint of the clear
    horizontal whitespace band between the stem and the option shapes.
    """
    y = min(y_top_b - 2, FOOTER_Y_START - 2)
    y_min_search = HEADER_Y_END + 25
    row_counts = ink_mask[:, 15:625].sum(axis=1)

    # Skip white margin immediately above badges
    while y > y_min_search and row_counts[y] <= 2:
        y -= 1
    if y <= y_min_search + 20:
        return None

    # Walk upward through the option shapes/text block
    seen_option_ink = False
    gap_run = 0
    while y > y_min_search:
        if row_counts[y] > 2:
            seen_option_ink = True
            gap_run = 0
        elif seen_option_ink:
            gap_run += 1
            if gap_run >= min_gap_rows:
                # Found wide horizontal gap above the option block
                return y + gap_run // 2
        y -= 1
    return None


def _has_clear_vertical_gutters(
    ink_mask: np.ndarray,
    y0: int,
    y1: int,
    col_bounds: List[int],
    search_radius: int = 24,
    min_corridor_width: int = 5,
) -> bool:
    """
    Verify that a band [y0:y1] separates cleanly into 4 horizontally isolated cards/options
    with a continuous zero-ink vertical white corridor (>= min_corridor_width wide) near each inner col_bound.
    Blocks continuous lines/circles/grids (e.g. Venn diagrams Q52 or coordinate planes Q917).
    """
    if y1 <= y0:
        return False
    band = ink_mask[y0:y1, :]
    for gb in col_bounds[1:4]:
        lo = max(15, gb - search_radius)
        hi = min(STD_WIDTH - 15, gb + search_radius)
        if hi - lo < min_corridor_width:
            return False
        col_ink = band[:, lo:hi].sum(axis=0)
        # Require a continuous run of >= min_corridor_width completely empty columns
        run = 0
        found_corridor = False
        for val in col_ink:
            if int(val) == 0:
                run += 1
                if run >= min_corridor_width:
                    found_corridor = True
                    break
            else:
                run = 0
        if not found_corridor:
            return False
    return True


def _find_horizontal_split_gap(
    ink_mask: np.ndarray,
    y_search_top: int,
    y_search_bot: int,
    min_gap_rows: int = 8,
) -> Optional[int]:
    """
    Find a wide pure-white horizontal whitespace band separating stem and 2x2 grid.
    """
    y_top = max(HEADER_Y_END + 10, y_search_top)
    y_bot = min(FOOTER_Y_START - 10, y_search_bot)
    if y_bot - y_top < min_gap_rows:
        return None

    row_counts = ink_mask[y_top:y_bot, 15:625].sum(axis=1)
    clean_rows = row_counts <= 2
    run_len = 0
    best_mid = None
    for idx in range(len(clean_rows) - 1, -1, -1):
        if clean_rows[idx]:
            run_len += 1
        else:
            if run_len >= min_gap_rows:
                best_mid = y_top + idx + 1 + run_len // 2
                break
            run_len = 0
    return best_mid


def _compute_masked_stem_density(
    rgb_roi: np.ndarray,
    gray_roi: np.ndarray,
    full_crop_box: Tuple[int, int, int, int],
    badges: Dict[str, Dict[str, Any]],
    ocr_items: List[Dict[str, Any]],
    extra_mask_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[float, float, bool, bool]:
    """
    Compute OCR-masked structural ink density, color saturation density, and long-stroke flags
    strictly on RoI_stem.

    Returns:
        Tuple of:
        - rho_stem: Fraction of non-text, non-badge structural pixels in RoI_stem.
        - sat_density: Fraction of color-saturated non-text, non-badge pixels (catches colored dot grids).
        - has_long_stroke: True if any connected structural stroke spans >= 95px
          (catches thin number lines and coordinate axes).
        - only_small_box: True if remaining ink is solely tiny <= 42px blank input box(es).
    """
    rx0, ry0, rx1, ry1 = full_crop_box
    roi_h, roi_w = gray_roi.shape
    mask = gray_roi < 235

    # Color saturation mask (RGB channel spread > 40 and non-white) for colored dots/markers
    rgb_i = rgb_roi.astype(np.int32)
    sat_mask = ((rgb_i.max(axis=2) - rgb_i.min(axis=2)) > 40) & (rgb_i.min(axis=2) < 220)

    # Mask any badges intersecting RoI_stem (e.g. Venn-embedded or point-reference slides)
    for b in badges.values():
        bx0 = max(0, b["x0"] - 4 - rx0)
        by0 = max(0, b["y0"] - 4 - ry0)
        bx1 = min(roi_w, b["x1"] + 5 - rx0)
        by1 = min(roi_h, b["y1"] + 5 - ry0)
        if bx1 > bx0 and by1 > by0:
            mask[by0:by1, bx0:bx1] = False
            sat_mask[by0:by1, bx0:bx1] = False

    # Mask masked option boxes (e.g. vertical-left / vertical-right stem wipe)
    if extra_mask_boxes:
        for mx0, my0, mx1, my1 in extra_mask_boxes:
            lx0 = max(0, mx0 - rx0)
            ly0 = max(0, my0 - ry0)
            lx1 = min(roi_w, mx1 - rx0)
            ly1 = min(roi_h, my1 - ry0)
            if lx1 > lx0 and ly1 > ly0:
                mask[ly0:ly1, lx0:lx1] = False
                sat_mask[ly0:ly1, lx0:lx1] = False

    # Mask Apple Vision / EasyOCR text boxes mapped into RoI_stem coordinates
    for item in ocr_items:
        bx, by, bw, bh = item["bbox"]
        px0 = max(0, int(bx * STD_WIDTH) - 4 - rx0)
        py0 = max(0, int(by * STD_HEIGHT) - 4 - ry0)
        px1 = min(roi_w, int((bx + bw) * STD_WIDTH) + 5 - rx0)
        py1 = min(roi_h, int((by + bh) * STD_HEIGHT) + 5 - ry0)
        if px1 > px0 and py1 > py0:
            mask[py0:py1, px0:px1] = False
            sat_mask[py0:py1, px0:px1] = False

    lbl, _ = ndimage.label(mask)
    slices = ndimage.find_objects(lbl)
    kept_pixels = 0
    max_stroke_span = 0
    component_spans: List[int] = []

    for cid, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        ch = sl[0].stop - sl[0].start
        cw = sl[1].stop - sl[1].start
        span = max(ch, cw)
        if span >= 28:
            area = int((lbl[sl] == cid).sum())
            if area >= 40:
                kept_pixels += area
                component_spans.append(span)
                if span > max_stroke_span:
                    max_stroke_span = span

    # Normalize strictly against the canonical full slide content canvas (640 x 402)
    # so tight RoI_stem crops on single-line equations do not inflate box border density.
    denom = float(STD_WIDTH * (FOOTER_Y_START - HEADER_Y_END))
    rho = kept_pixels / denom
    sat_density = int(sat_mask.sum()) / denom
    has_long_stroke = max_stroke_span >= 95
    only_small_box = bool(component_spans) and max(component_spans) <= 50 and kept_pixels < 700
    return rho, sat_density, has_long_stroke, only_small_box


def _mask_boxes_white(img: Image.Image, crop_box: Tuple[int, int, int, int], mask_boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    """Fill local option/badge rectangles with pure white on a cropped RGB canvas."""
    rx0, ry0, _, _ = crop_box
    arr = np.asarray(img, dtype=np.uint8).copy()
    h, w, _ = arr.shape
    for mx0, my0, mx1, my1 in mask_boxes:
        lx0 = max(0, mx0 - rx0)
        ly0 = max(0, my0 - ry0)
        lx1 = min(w, mx1 - rx0)
        ly1 = min(h, my1 - ry0)
        if lx1 > lx0 and ly1 > ly0:
            arr[ly0:ly1, lx0:lx1] = 255
    return Image.fromarray(arr, mode="RGB")


def extract_slide_rois_and_metadata(
    qid: int,
    img_path: Path,
    ocr_items: List[Dict[str, Any]],
    rois_out_dir: Path,
    ref_calc_template: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Parse one standardized Eedi slide, classify layout & calculator flag, extract and
    save clean badge-free RoI_stem and RoI_opt_{A..D}, and compute structural visual stats.
    """
    rois_out_dir.mkdir(parents=True, exist_ok=True)
    img, orig_size = load_standardized_slide(img_path)
    arr = np.asarray(img)
    gray = np.asarray(img.convert("L"))

    calc_status, calc_allowed = classify_calculator_header(arr, ref_calc_template)
    badges = detect_option_badges(arr)

    xs = np.array([badges[k]["cx"] for k in "ABCD"])
    ys = np.array([badges[k]["cy"] for k in "ABCD"])
    y_span = float(ys.max() - ys.min())
    x_span = float(xs.max() - xs.min())

    # Global content ink mask excluding header, footer, and the 4 colored badges
    clean_ink = gray < 215
    clean_ink[:HEADER_Y_END, :] = False
    clean_ink[FOOTER_Y_START:, :] = False
    for k in "ABCD":
        b = badges[k]
        clean_ink[max(0, b["y0"] - 4):min(STD_HEIGHT, b["y1"] + 5), max(0, b["x0"] - 4):min(STD_WIDTH, b["x1"] + 5)] = False

    opt_boxes: Dict[str, Tuple[int, int, int, int]] = {}
    stem_mask_boxes: List[Tuple[int, int, int, int]] = []
    layout_type: str

    if y_span < 35.0:
        # Order badges left-to-right for crisp column gutter calculation
        ordered = sorted("ABCD", key=lambda k: badges[k]["cx"])
        cxs = [badges[k]["cx"] for k in ordered]
        col_bounds = [
            10,
            int(round(0.5 * (cxs[0] + cxs[1]))),
            int(round(0.5 * (cxs[1] + cxs[2]))),
            int(round(0.5 * (cxs[2] + cxs[3]))),
            STD_WIDTH - 10,
        ]
        y_top_b = min(badges[k]["y0"] for k in "ABCD")
        y_bot_b = max(badges[k]["y1"] for k in "ABCD")

        # Count ink columns below badges (low threshold >= 25 catches small math symbols like >, <, =)
        below_cols_active = 0
        below_total = 0
        y_bel_start = min(FOOTER_Y_START - 1, y_bot_b + 3)
        for ci in range(4):
            col_px = int(clean_ink[y_bel_start:FOOTER_Y_START, col_bounds[ci]:col_bounds[ci + 1]].sum())
            below_total += col_px
            if col_px >= 6:
                below_cols_active += 1

        # Detect Venn/embedded horizontal slides where arcs cross badge row across multiple gutters
        badge_band_ink_crossings = int(
            clean_ink[max(HEADER_Y_END, y_top_b - 6):min(FOOTER_Y_START, y_bot_b + 6),
                      max(10, col_bounds[1] - 12):min(STD_WIDTH - 10, col_bounds[3] + 12)].sum()
        )
        below_gutters_clear = _has_clear_vertical_gutters(
            clean_ink, y_bel_start, FOOTER_Y_START, col_bounds, search_radius=16, min_corridor_width=3
        )

        if below_cols_active >= 2 and below_total >= 25 and below_gutters_clear and badge_band_ink_crossings < 180:
            layout_type = "horiz_below"
            stem_box_raw = (10, HEADER_Y_END, STD_WIDTH - 10, max(HEADER_Y_END + 20, y_top_b - 4))
            for ci, letter in enumerate(ordered):
                raw_opt = (col_bounds[ci], y_bel_start, col_bounds[ci + 1], FOOTER_Y_START)
                opt_boxes[letter] = _tight_ink_bbox(gray, raw_opt, pad=6, min_w=24, min_h=24)
        else:
            # Distinguish above-badge 4-column options (Q227, Q508, Q514, Q700) vs diagram/Venn ref letters (Q17, Q52, Q917)
            gap_y = _find_above_badge_option_band(clean_ink, y_top_b=y_top_b, min_gap_rows=10)
            gutters_clear = False
            fine_ink = (gray < 232)
            fine_ink[:HEADER_Y_END, :] = False
            fine_ink[FOOTER_Y_START:, :] = False
            if gap_y is not None and y_top_b - gap_y >= 24:
                stem_above_ink = int(fine_ink[HEADER_Y_END:gap_y, 15:625].sum())
                band_cols_active = sum(
                    1 for ci in range(4)
                    if fine_ink[gap_y:y_top_b - 1, col_bounds[ci]:col_bounds[ci + 1]].sum() >= 15
                )
                if stem_above_ink >= 60 and band_cols_active >= 3 and _has_clear_vertical_gutters(
                    fine_ink, gap_y, y_top_b - 1, col_bounds, search_radius=24, min_corridor_width=6
                ):
                    gutters_clear = True

            if gutters_clear and gap_y is not None:
                layout_type = "horiz_above_options"
                stem_box_raw = (10, HEADER_Y_END, STD_WIDTH - 10, gap_y)
                # Shift column dividers slightly to exact clean white gutters inside [gap_y : y_top_b]
                adj_bounds = [col_bounds[0]]
                for gb in col_bounds[1:4]:
                    lo = max(15, gb - 20)
                    hi = min(STD_WIDTH - 15, gb + 20)
                    col_sums = clean_ink[gap_y:y_top_b - 1, lo:hi].sum(axis=0)
                    adj_bounds.append(int(lo + int(np.argmin(col_sums))))
                adj_bounds.append(col_bounds[4])
                for ci, letter in enumerate(ordered):
                    raw_opt = (adj_bounds[ci], gap_y + 2, adj_bounds[ci + 1], y_top_b - 2)
                    opt_boxes[letter] = _tight_ink_bbox(gray, raw_opt, pad=6, min_w=24, min_h=24)
            else:
                layout_type = "horiz_ref"
                # Venn diagrams (e.g. Q52) extend below badges, whereas point-ref diagrams stay above footer
                y_stem_end = FOOTER_Y_START if below_total >= 40 else max(HEADER_Y_END + 40, y_top_b - 3)
                stem_box_raw = (10, HEADER_Y_END, STD_WIDTH - 10, y_stem_end)
                if below_total >= 40:
                    for b_ref in badges.values():
                        stem_mask_boxes.append((b_ref["x0"] - 2, b_ref["y0"] - 2, b_ref["x1"] + 3, b_ref["y1"] + 3))
                for ci, letter in enumerate(ordered):
                    cx, cy = int(badges[letter]["cx"]), int(badges[letter]["cy"])
                    ox0 = max(10, cx - 55)
                    oy0 = max(HEADER_Y_END, cy - 55)
                    ox1 = min(STD_WIDTH - 10, cx + 55)
                    oy1 = min(FOOTER_Y_START, cy + 55)
                    opt_boxes[letter] = (ox0, oy0, ox1, oy1)

    elif x_span < 45.0:
        ordered = sorted("ABCD", key=lambda k: badges[k]["cy"])
        cys = [badges[k]["cy"] for k in ordered]
        row_bounds = [
            max(HEADER_Y_END, int(round(cys[0] - 0.5 * (cys[1] - cys[0])))),
            int(round(0.5 * (cys[0] + cys[1]))),
            int(round(0.5 * (cys[1] + cys[2]))),
            int(round(0.5 * (cys[2] + cys[3]))),
            min(FOOTER_Y_START, int(round(cys[3] + 0.5 * (cys[3] - cys[2])))),
        ]
        x_min_b = min(badges[k]["x0"] for k in "ABCD")
        x_max_b = max(badges[k]["x1"] for k in "ABCD")

        if xs.mean() < 200.0:
            layout_type = "vert_left"
            # Locate the FIRST wide clean whitespace gutter to the right of left-column options
            x_scan_start = x_max_b + 25
            col_counts = clean_ink[row_bounds[0]:row_bounds[-1], x_scan_start:360].sum(axis=0)
            zero_mask = col_counts <= 1
            x_opt_right = min(295, x_max_b + 140)
            run_len = 0
            for idx, is_zero in enumerate(zero_mask):
                if is_zero:
                    run_len += 1
                    if run_len >= 8:
                        x_opt_right = int(x_scan_start + idx - run_len // 2)
                        break
                else:
                    run_len = 0
            for ri, letter in enumerate(ordered):
                raw_opt = (x_max_b + 3, row_bounds[ri], x_opt_right, row_bounds[ri + 1])
                opt_boxes[letter] = _tight_ink_bbox(gray, raw_opt, pad=6, min_w=24, min_h=24)
            stem_box_raw = (10, HEADER_Y_END, STD_WIDTH - 10, FOOTER_Y_START)
            stem_mask_boxes.append((max(0, x_min_b - 6), row_bounds[0], x_opt_right, row_bounds[-1]))
        else:
            layout_type = "vert_right"
            for ri, letter in enumerate(ordered):
                raw_opt = (x_max_b + 3, row_bounds[ri], STD_WIDTH - 10, row_bounds[ri + 1])
                opt_boxes[letter] = _tight_ink_bbox(gray, raw_opt, pad=6, min_w=24, min_h=24)
            stem_box_raw = (10, HEADER_Y_END, STD_WIDTH - 10, FOOTER_Y_START)
            stem_mask_boxes.append((max(0, x_min_b - 6), row_bounds[0], STD_WIDTH - 6, row_bounds[-1]))

    else:
        # 2x2 grid (either [A,B]/[C,D] or [A,C]/[B,D])
        top_pair = sorted(sorted("ABCD", key=lambda k: badges[k]["cy"])[:2], key=lambda k: badges[k]["cx"])
        bot_pair = sorted(sorted("ABCD", key=lambda k: badges[k]["cy"])[2:], key=lambda k: badges[k]["cx"])
        y_top_row_b = min(badges[k]["y0"] for k in top_pair)
        y_bot_row_b = min(badges[k]["y0"] for k in bot_pair)
        # Split left vs right quadrants cleanly just left of the right-column badges (or cleanest whitespace gutter nearby)
        x_right_badge = min(badges[top_pair[1]]["x0"], badges[bot_pair[1]]["x0"])
        x_left_badge_end = max(badges[top_pair[0]]["x1"], badges[bot_pair[0]]["x1"])
        if x_right_badge - x_left_badge_end > 40:
            search_lo = max(x_left_badge_end + 15, x_right_badge - 40)
            search_hi = max(search_lo + 1, x_right_badge - 3)
            gutter_col = int(search_lo + np.argmin(clean_ink[y_top_row_b:FOOTER_Y_START, search_lo:search_hi].sum(axis=0)))
            x_mid = gutter_col
        else:
            x_mid = int(round(0.5 * (x_left_badge_end + x_right_badge)))
        y_mid = int(round(0.5 * (max(badges[k]["y1"] for k in top_pair) + y_bot_row_b)))

        top_stem_boundary = max(HEADER_Y_END + 10, y_top_row_b - 6)
        cross_rows = int(clean_ink[max(HEADER_Y_END, top_stem_boundary - 6):min(FOOTER_Y_START, y_top_row_b + 4), 15:625].sum())
        has_header_gap = _find_horizontal_split_gap(
            clean_ink,
            y_search_top=HEADER_Y_END + 15,
            y_search_bot=y_top_row_b + 2,
            min_gap_rows=5,
        )
        if has_header_gap is None and cross_rows > 35:
            layout_type = "grid_embedded"
            stem_box_raw = (10, HEADER_Y_END, STD_WIDTH - 10, FOOTER_Y_START)
            for letter in "ABCD":
                b = badges[letter]
                stem_mask_boxes.append((b["x0"] - 3, b["y0"] - 3, b["x1"] + 4, b["y1"] + 4))
                cx, cy = int(b["cx"]), int(b["cy"])
                opt_boxes[letter] = (max(10, cx - 70), max(HEADER_Y_END, cy - 55), min(STD_WIDTH - 10, cx + 70), min(FOOTER_Y_START, cy + 55))
        else:
            layout_type = "grid_std"
            split_top = has_header_gap if has_header_gap is not None else max(HEADER_Y_END + 20, y_top_row_b - 6)
            stem_box_raw = (10, HEADER_Y_END, STD_WIDTH - 10, split_top)

            def _quad_opt(letter: str, c_min: int, c_max: int, r0: int, r1: int) -> Tuple[int, int, int, int]:
                # Keep full quadrant around badge (which gets white-masked) so figures left/below badge never get clipped
                ox0 = min(c_min, badges[letter]["x0"] - 4)
                return _tight_ink_bbox(gray, (max(10, ox0), r0, c_max, r1), pad=6, min_w=24, min_h=24)

            opt_boxes[top_pair[0]] = _quad_opt(top_pair[0], 10, x_mid, split_top, y_mid)
            opt_boxes[top_pair[1]] = _quad_opt(top_pair[1], x_mid, STD_WIDTH - 10, split_top, y_mid)
            opt_boxes[bot_pair[0]] = _quad_opt(bot_pair[0], 10, x_mid, y_mid, FOOTER_Y_START)
            opt_boxes[bot_pair[1]] = _quad_opt(bot_pair[1], x_mid, STD_WIDTH - 10, y_mid, FOOTER_Y_START)

    # Trim outer whitespace of stem after applying option wipe masks
    stem_canvas = _mask_boxes_white(img.crop(stem_box_raw), stem_box_raw, stem_mask_boxes)
    stem_gray_masked = np.asarray(stem_canvas.convert("L"))
    non_white_stem = np.where(stem_gray_masked < 240)
    if len(non_white_stem[0]) > 0:
        sy0 = max(0, int(non_white_stem[0].min()) - 8)
        sy1 = min(stem_canvas.height, int(non_white_stem[0].max()) + 9)
        sx0 = max(0, int(non_white_stem[1].min()) - 8)
        sx1 = min(stem_canvas.width, int(non_white_stem[1].max()) + 9)
        roi_stem_img = stem_canvas.crop((sx0, sy0, sx1, sy1))
        stem_bbox_abs = (
            stem_box_raw[0] + sx0,
            stem_box_raw[1] + sy0,
            stem_box_raw[0] + sx1,
            stem_box_raw[1] + sy1,
        )
    else:
        roi_stem_img = stem_canvas
        stem_bbox_abs = stem_box_raw

    stem_path = rois_out_dir / f"{qid}_stem.png"
    roi_stem_img.save(stem_path)

    opt_paths: Dict[str, str] = {}
    opt_cv_visual: Dict[str, bool] = {}
    for letter in "ABCD":
        ob = opt_boxes[letter]
        o_crop = img.crop(ob)
        # Mask out badge rectangle if any edge intruded
        b = badges[letter]
        if layout_type in ("horiz_ref", "grid_embedded"):
            # Keep point/angle label badge visible in reference crops, or mask slight overlap on standard
            pass
        else:
            o_crop = _mask_boxes_white(o_crop, ob, [(b["x0"] - 2, b["y0"] - 2, b["x1"] + 3, b["y1"] + 3)])
        opath = rois_out_dir / f"{qid}_opt_{letter}.png"
        o_crop.save(opath)
        opt_paths[letter] = str(opath)

        # Heuristic visual option check (OCR-free ink spans / non-white area)
        og = np.asarray(o_crop.convert("L"))
        o_ink = og < 225
        opt_cv_visual[letter] = bool(og.shape[0] >= 55 and og.shape[1] >= 55 and o_ink.mean() > 0.03)

    rho_stem, sat_density, has_long_stroke, only_small_box = _compute_masked_stem_density(
        rgb_roi=np.asarray(roi_stem_img),
        gray_roi=np.asarray(roi_stem_img.convert("L")),
        full_crop_box=stem_bbox_abs,
        badges=badges,
        ocr_items=ocr_items,
        extra_mask_boxes=stem_mask_boxes,
    )

    return {
        "qid": qid,
        "orig_size": list(orig_size),
        "layout_type": layout_type,
        "calculator_status": calc_status,
        "calculator_allowed_flag": calc_allowed,
        "badges": badges,
        "stem_bbox": list(stem_bbox_abs),
        "opt_bboxes": {k: list(v) for k, v in opt_boxes.items()},
        "roi_stem_path": str(stem_path),
        "roi_opt_paths": opt_paths,
        "opt_cv_visual": opt_cv_visual,
        "stem_struct_density": round(float(rho_stem), 6),
        "stem_sat_density": round(float(sat_density), 6),
        "stem_has_long_stroke": has_long_stroke,
        "stem_only_small_box": only_small_box,
    }
