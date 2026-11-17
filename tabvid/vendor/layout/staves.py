import cv2
import numpy as np
from ..config import LINE_THR

def _hlines(ink, w, frac=0.25):
    klen = max(40, int(w * frac))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))


def _line_centers(mask_rows):
    ys, i, h = [], 0, len(mask_rows)
    while i < h:
        if mask_rows[i]:
            j = i
            while j + 1 < h and mask_rows[j + 1]:
                j += 1
            ys.append((i + j) / 2.0)
            i = j + 1
        else:
            i += 1
    return ys


def staff_spacing(img):
    """Pixel distance between adjacent staff lines (a zoom-independent size reference)."""
    h, w = img.shape
    prof = _hlines((img < LINE_THR).astype(np.uint8) * 255, w, frac=0.2).any(axis=1)
    ys, i = [], 0
    while i < h:
        if prof[i]:
            j = i
            while j + 1 < h and prof[j + 1]:
                j += 1
            ys.append((i + j) / 2.0)
            i = j + 1
        else:
            i += 1
    if len(ys) < 3:
        return None
    ys = np.array(ys)
    g = np.diff(ys)
    cluster = ys
    if g.max() > 3 * np.median(g):
        # notation staff on top + tab staff below: measure the TAB staff (bottom cluster)
        k = int(np.argmax(g))
        cluster = ys[k + 1:]
        if len(cluster) < 3:
            cluster = ys[:k + 1]
    g = np.diff(cluster)
    if len(g) == 0:
        return None
    m = float(g.min())
    good = [x for x in g if x < 1.5 * m]
    return float(np.median(good)) if good else None
