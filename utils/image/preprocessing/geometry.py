# utils/image/geometry.py
"""
Geometric transforms & page shaping utilities.

- rotate_deg(image, deg)
- auto_trim_black_borders(gray)
- crop_page_for_scoring(gray)
- sanity_plus_90_if_better(gray)
- deskew_small_angle(image, max_angle=30.0) -> (image, meta)
- apply_perspective_correction(gray, debug=None)
- resize(image, factor)
"""

from __future__ import annotations

from typing import Tuple, Dict, List
import cv2
import numpy as np

from utils.image.shared_utils import _to_gray_u8

# --------- helpers ---------

# In geometry.py

def find_image_regions(gray: np.ndarray, min_area: int = 5000) -> List[np.ndarray]:
    """
    Finds large contour regions that are likely to be images or diagrams
    by checking for area, solidity, and aspect ratio.
    """
    image_regions = []

    # Invert the image so that text/diagrams are white
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    # Use a morphological close to connect nearby components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 1. Check Area
        if w * h < min_area:
            continue

        # 2. Check Solidity (ratio of contour area to bounding box area)
        # Text blocks are sparse and have low solidity. Images are high.
        solidity = cv2.contourArea(cnt) / (w * h)
        if solidity < 0.4:
            continue

        # 3. Check Aspect Ratio
        # Text blocks are usually very wide or very tall. Images are more square-like.
        aspect_ratio = w / h
        if aspect_ratio > 4 or aspect_ratio < 0.25:
            continue

        # If it passes all checks, it's likely an image/diagram
        image_regions.append((x, y, w, h))

    return image_regions

# --------- rotations / basic ops ---------

def rotate_deg(image: np.ndarray, deg: int) -> np.ndarray:
    """Fast 0/90/180/270° rotation (no arbitrary angles)."""
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def resize(image: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return image
    h, w = image.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    interp = cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


# --------- page finding / trimming ---------

def auto_trim_black_borders(gray: np.ndarray) -> np.ndarray:
    """
    Trim thick black scanner borders. Input must be gray.
    """
    g = _to_gray_u8(gray)
    H, W = g.shape[:2]
    max_frac = 0.18
    step = max(4, int(min(H, W) * 0.006))

    def black_ratio(img):
        return float((img < 60).sum()) / max(1, img.size)

    def cut_top(img):
        H, _ = img.shape[:2]
        limit = int(H * max_frac)
        y = 0
        while y + step < min(limit, H // 2):
            if black_ratio(img[y:y + step, :]) > 0.60:
                y += step
            else:
                break
        return img[y:, :]

    def cut_bottom(img):
        H, _ = img.shape[:2]
        limit = int(H * max_frac)
        y = 0
        while y + step < min(limit, H // 2):
            if black_ratio(img[H - (y + step):H - y, :]) > 0.60:
                y += step
            else:
                break
        return img[:H - y, :]

    def cut_left(img):
        _, W = img.shape[:2]
        limit = int(W * max_frac)
        x = 0
        while x + step < min(limit, W // 2):
            if black_ratio(img[:, x:x + step]) > 0.60:
                x += step
            else:
                break
        return img[:, x:]

    def cut_right(img):
        _, W = img.shape[:2]
        limit = int(W * max_frac)
        x = 0
        while x + step < min(limit, W // 2):
            if black_ratio(img[:, W - (x + step):W - x]) > 0.60:
                x += step
            else:
                break
        return img[:, :W - x]

    g = cut_top(g)
    g = cut_bottom(g)
    g = cut_left(g)
    g = cut_right(g)
    return g


def crop_page_for_scoring(gray: np.ndarray) -> np.ndarray:
    """
    Return a crop of the main page region (remove fringe/background).
    If we can't detect a reliable page, return the input gray unchanged.
    """
    g = _to_gray_u8(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ge = clahe.apply(g)
    _, thr = cv2.threshold(ge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close = cv2.morphologyEx(
        thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), 1
    )
    open_ = cv2.morphologyEx(
        close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1
    )

    cnts, _ = cv2.findContours(open_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return g

    H, W = g.shape[:2]
    page = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(page)
    if w * h < 0.5 * H * W:
        return g

    inset_x = max(2, int(0.01 * w))
    inset_y = max(2, int(0.01 * h))
    x1 = max(0, x + inset_x)
    y1 = max(0, y + inset_y)
    x2 = min(W, x + w - inset_x)
    y2 = min(H, y + h - inset_y)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return g
    return g[y1:y2, x1:x2]


# --------- deskew & sanity rotation ---------

def sanity_plus_90_if_better(gray: np.ndarray) -> np.ndarray:
    """
    Rotate +90 if baselines look more horizontal after rotation.
    """
    def dom_angle(g: np.ndarray) -> Tuple[float, int]:
        e = cv2.Canny(g, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            e, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )
        if lines is None or len(lines) == 0:
            return (0.0, 0)
        angs = []
        for x1, y1, x2, y2 in lines[:, 0]:
            a = float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
            if a > 90:
                a -= 180
            elif a < -90:
                a += 180
            angs.append(a)
        return (float(np.median(angs)), int(len(angs)))

    g = _to_gray_u8(gray)
    a1, n1 = dom_angle(g)
    alt = rotate_deg(g, 90)
    a2, n2 = dom_angle(alt)
    if (abs(a2) + 1e-3 < abs(a1)) or (abs(a2) <= abs(a1) + 1e-3 and n2 > n1):
        return alt
    return g


# In geometry.py

def deskew_small_angle(
    image: np.ndarray,
    max_angle: float = 30.0,
    threshold: int = 100,
    min_line_length: int = 100,
    max_line_gap: int = 10
) -> Tuple[np.ndarray, Dict[str, float | str]]:
    """
    Estimate a small skew via Hough lines and correct it with expanded canvas.
    Returns (deskewed_image, meta)
    meta: {"median": angle_deg, "reason": "(no_lines|near-vertical|exceeds-max|too-small|)"}
    """
    gray = _to_gray_u8(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use the new parameters in the HoughLinesP call ---
    linesP = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    angles: list[float] = []
    if linesP is not None and len(linesP) > 0:
        for x1, y1, x2, y2 in linesP[:, 0]:
            ang = float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
            if ang > 90:
                ang -= 180
            elif ang < -90:
                ang += 180
            angles.append(ang)

    if not angles:
        return image, {"median": 0.0, "reason": "(no_lines)"}

    med = float(np.median(angles))
    if abs(med) > 85:
        return image, {"median": med, "reason": "(near-vertical)"}
    if abs(med) > max_angle:
        return image, {"median": med, "reason": "(exceeds-max)"}
    if abs(med) <= 0.5:
        return image, {"median": med, "reason": "(too-small)"}

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), med, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - (w // 2)
    M[1, 2] += (new_h / 2) - (h // 2)
    out = cv2.warpAffine(
        image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return out, {"median": med, "reason": ""}


# --------- perspective correction ---------

def apply_perspective_correction(gray: np.ndarray, debug=None) -> np.ndarray:
    """
    Detect a quadrilateral page contour and warp to a rectangle.
    Input/Output are grayscale uint8.
    """
    g = _to_gray_u8(gray)
    try:
        edges = cv2.Canny(g, 60, 160)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return g
        page = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(page, True)
        approx = cv2.approxPolyDP(page, 0.02 * peri, True)
        if len(approx) != 4:
            return g
        if cv2.contourArea(approx) < 0.25 * (g.shape[0] * g.shape[1]):
            return g

        pts = approx.reshape(4, 2).astype(np.float32)
        rect = _order_points(pts)

        widthA = float(np.linalg.norm(rect[2] - rect[3]))
        widthB = float(np.linalg.norm(rect[1] - rect[0]))
        heightA = float(np.linalg.norm(rect[1] - rect[2]))
        heightB = float(np.linalg.norm(rect[0] - rect[3]))
        W = int(max(widthA, widthB))
        H = int(max(heightA, heightB))

        max_height = 2400
        if H > max_height:
            scale = max_height / H
            W = int(W * scale)
            H = int(H * scale)

        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(
            g, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        if debug and hasattr(debug, "write"):
            debug.write("perspective_edges", edges)
            debug.write("perspective_warped", warped)
        return warped
    except Exception:
        return g


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    rect[1] = pts[np.argmin(d)]  # TR
    rect[3] = pts[np.argmax(d)]  # BL
    return rect
