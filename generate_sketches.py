"""
generate_sketches.py
--------------------
Stage 2: HCI Sketch Grammar — clean, professional teal-green marker on white paper.
Crisp lines, external annotations with arrows. Per feedback spec.
"""

import os
import json
import cv2
import numpy as np
import glob

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
RENDERS_DIR   = os.path.join(DATASET_DIR, "renders")
CONDITION_DIR = os.path.join(DATASET_DIR, "conditioning")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")

os.makedirs(CONDITION_DIR, exist_ok=True)

# Teal-green marker (BGR)
MARKER_GREEN_BGR = (34, 139, 69)

DEFAULT_OBJECT = "Cloth"
DEFAULT_KEYWORD = "texture pattern"


def _centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))


def draw_dashed_contour(canvas, contour, color, dash_len=6, gap_len=4, thickness=2):
    """Draw contour as dashed line (Segmentation Mask style)."""
    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < 2:
        return
    i = 0
    draw = True
    while i < n:
        if draw:
            end = min(i + dash_len, n)
            for j in range(i, end - 1):
                p1 = tuple(pts[j % n].astype(int))
                p2 = tuple(pts[(j + 1) % n].astype(int))
                cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
            i = end
            draw = False
        else:
            i += gap_len
            draw = True


def _fallback_shadow_region(object_mask, h, w):
    ys, xs = np.where(object_mask > 0)
    if len(ys) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    y_med = int(np.median(ys))
    y_max = int(np.max(ys))
    y_cut = y_med + (y_max - y_med) // 2
    shadow = np.zeros((h, w), dtype=np.uint8)
    shadow[object_mask > 0] = 255
    shadow[:y_cut, :] = 0
    return shadow


def _fallback_highlight_point(object_mask, gray, h, w):
    ys, xs = np.where(object_mask > 0)
    if len(ys) == 0:
        return None
    obj_gray = gray.copy()
    obj_gray[object_mask == 0] = 0
    cy, cx = np.unravel_index(np.argmax(obj_gray), gray.shape)
    if object_mask[cy, cx] > 0:
        return (int(cx), int(cy))
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    return (int((x_min + x_max) / 2), int(y_min + (y_max - y_min) / 4))


def _fallback_texture_region(object_mask, mid_tone):
    return mid_tone if np.any(mid_tone > 0) else object_mask.copy()


def generate_hatching(shape, shadow_mask, spacing=8):
    """Clean diagonal cross-hatching for shadow area."""
    h, w = shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for i in range(-h, w + h, spacing):
        cv2.line(canvas, (max(0, i), 0), (min(w - 1, i + h), h), 255, 1, lineType=cv2.LINE_AA)
    return cv2.bitwise_and(canvas, shadow_mask)


def generate_stippling(mask, density=0.022):
    """Fine stippling for wool texture (only in texture area, not on annotations)."""
    h, w = mask.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    n = len(ys)
    if n == 0:
        return canvas
    n_dots = min(int(n * density), 4500)
    np.random.seed(43)
    idx = np.random.choice(n, size=n_dots, replace=False)
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        cv2.circle(canvas, (x, y), 1, 255, -1)
    return canvas


def generate_hci_sketch(render_path, normals_path, text_line1, text_line2):
    beauty = cv2.imread(render_path)
    if beauty is None:
        raise FileNotFoundError(render_path)

    gray = cv2.cvtColor(beauty, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if normals_path.endswith(".npy") and os.path.exists(normals_path):
        nn = np.load(normals_path)
        if nn.ndim == 3:
            ng = np.clip((np.linalg.norm(nn, axis=2) + 1) / 2 * 255, 0, 255).astype(np.uint8)
        else:
            ng = nn.astype(np.uint8)
        normal_img = cv2.cvtColor(ng, cv2.COLOR_GRAY2BGR)
    else:
        normal_img = cv2.imread(normals_path) if os.path.exists(normals_path) else None

    _, object_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # White canvas, crisp green lines
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    th = 2

    # 1. SEGMENTATION MASK — dashed outline
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_contour = None
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            draw_dashed_contour(canvas, cnt, MARKER_GREEN_BGR, dash_len=6, gap_len=4, thickness=th)
            seg_contour = cnt

    # Point on silhouette for "Segmentation Mask" arrow
    seg_pt = None
    if seg_contour is not None:
        M = cv2.moments(seg_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            seg_pt = (cx, cy)

    # 2. SHADOW — clean diagonal cross-hatching
    shadow_mask = cv2.bitwise_and(
        cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)[1],
        object_mask,
    )
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if np.sum(shadow_mask) < 100:
        shadow_mask = _fallback_shadow_region(object_mask, h, w)
    shadow_centroid = _centroid(shadow_mask)
    shadow_hatch = generate_hatching((h, w), shadow_mask, spacing=8)
    canvas[shadow_hatch > 0] = MARKER_GREEN_BGR

    # 3. HIGHLIGHT — distinct empty circle
    highlight_mask = cv2.bitwise_and(
        cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1],
        object_mask,
    )
    highlight_center = None
    for cnt in cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) > 12:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx_i, cy_i = int(cx), int(cy)
            if 0 <= cy_i < h and 0 <= cx_i < w and object_mask[cy_i, cx_i] > 0:
                r = max(6, min(int(radius * 1.05), 22))
                cv2.circle(canvas, (cx_i, cy_i), r, MARKER_GREEN_BGR, 2, lineType=cv2.LINE_AA)
                highlight_center = (cx_i, cy_i)
                break
    if highlight_center is None:
        pt = _fallback_highlight_point(object_mask, gray, h, w)
        if pt:
            cv2.circle(canvas, pt, 14, MARKER_GREEN_BGR, 2, lineType=cv2.LINE_AA)
            highlight_center = pt

    # 4. WOOL TEXTURE — fine stippling (coarse texture)
    mid_tone = cv2.bitwise_and(
        cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1],
    )
    mid_tone = cv2.bitwise_and(mid_tone, object_mask)
    texture_mask = _fallback_texture_region(object_mask, mid_tone)
    texture_centroid = _centroid(texture_mask)
    stipple = generate_stippling(texture_mask, density=0.022)
    canvas[stipple > 0] = MARKER_GREEN_BGR

    # 5. Light interior creases (crisp)
    if normal_img is not None:
        ng = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(ng, 50, 100)
        edges = cv2.bitwise_and(edges, object_mask)
        canvas[edges > 0] = MARKER_GREEN_BGR

    # 6. EXTERNAL ANNOTATIONS — labels + arrows (crisp, legible)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, lt = 0.45, 1
    margin = 18

    # Top-left
    cv2.putText(canvas, text_line1, (12, 28), font, fs, MARKER_GREEN_BGR, lt, cv2.LINE_AA)
    cv2.putText(canvas, text_line2, (12, 50), font, fs, MARKER_GREEN_BGR, lt, cv2.LINE_AA)

    # Segmentation Mask + arrow (arrow points TO the silhouette)
    if seg_pt:
        lp = (margin, h - margin - 8)
        cv2.putText(canvas, "Segmentation Mask", lp, font, 0.4, MARKER_GREEN_BGR, lt, cv2.LINE_AA)
        cv2.arrowedLine(canvas, (lp[0] + 95, lp[1]), seg_pt, MARKER_GREEN_BGR, lt, tipLength=0.2)

    # Shadow + arrow
    if shadow_centroid:
        lp = (w - margin - 55, h - margin - 8)
        cv2.putText(canvas, "Shadow", lp, font, fs, MARKER_GREEN_BGR, lt, cv2.LINE_AA)
        cv2.arrowedLine(canvas, (lp[0] - 5, lp[1]), shadow_centroid, MARKER_GREEN_BGR, lt, tipLength=0.2)

    # Highlight + arrow
    if highlight_center:
        lp = (w - margin - 65, margin + 20)
        cv2.putText(canvas, "Highlight", lp, font, fs, MARKER_GREEN_BGR, lt, cv2.LINE_AA)
        cv2.arrowedLine(canvas, (lp[0] - 5, lp[1] + 10), highlight_center, MARKER_GREEN_BGR, lt, tipLength=0.2)

    # Coarse texture + arrow
    if texture_centroid:
        lp = (margin, margin + 72)
        cv2.putText(canvas, "coarse texture", lp, font, 0.4, MARKER_GREEN_BGR, lt, cv2.LINE_AA)
        cv2.arrowedLine(canvas, (lp[0] + 75, lp[1] - 5), texture_centroid, MARKER_GREEN_BGR, lt, tipLength=0.2)

    # woben2 — bottom-right
    cv2.putText(canvas, "woben2", (w - margin - 55, h - margin - 28), font, fs, MARKER_GREEN_BGR, lt, cv2.LINE_AA)

    return canvas


# Main
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH) as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
else:
    records = []
    for p in sorted(glob.glob(os.path.join(RENDERS_DIR, "render_*.png"))):
        fn = os.path.basename(p)
        frame = fn.replace("render_", "").replace(".png", "")
        np_path = os.path.join(DATASET_DIR, f"normals/normals_{frame}.npy")
        if os.path.exists(np_path):
            records.append({"frame": frame, "file_name": f"renders/{fn}", "normals_image": f"normals/normals_{frame}.npy"})

print(f"Found {len(records)} frame(s)\n")

for meta in records:
    frame_str = meta["frame"]
    render_path = os.path.join(DATASET_DIR, meta.get("file_name", f"renders/render_{frame_str}.png"))
    normals_path = os.path.join(DATASET_DIR, meta.get("normals_image", f"normals/normals_{frame_str}.npy"))
    raw_text = meta.get("text", "")
    keyword = meta.get("keyword") or ("wool pattern" if "wool" in raw_text.lower() else "silk pattern" if "silk" in raw_text.lower() else DEFAULT_KEYWORD)
    obj_name = "Wool Scarf" if "wool" in raw_text.lower() else "Silk Scarf" if "silk" in raw_text.lower() else "Cloth"
    text_line1 = f"text: {obj_name}"
    text_line2 = f"Key word: {keyword}"
    out_path = os.path.join(CONDITION_DIR, f"conditioning_{frame_str}.png")

    if os.path.exists(out_path):
        print(f"  [{frame_str}] Skipping")
        continue
    try:
        sketch = generate_hci_sketch(render_path, normals_path, text_line1, text_line2)
        cv2.imwrite(out_path, sketch)
        print(f"  [{frame_str}] ✓ → {out_path}")
    except Exception as e:
        print(f"  [{frame_str}] [ERROR] {e}")
        import traceback
        traceback.print_exc()

print(f"\n✓ Done → {CONDITION_DIR}\n")
