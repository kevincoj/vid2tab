from ..config import CFG
import cv2
import numpy as np
from ..config import LINE_THR
from .staves import _hlines, _line_centers

def crop_to_staff(img, pad=6):
    """Crop horizontally to the staff lines (plus chord names/diagrams sticking out), vertically to all ink."""
    h, w = img.shape
    ink = (img < CFG.ink_threshold).astype(np.uint8) * 255
    klen = max(40, int(w * CFG.crop_line_fraction))
    lines = cv2.morphologyEx((img < LINE_THR).astype(np.uint8) * 255, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))
    cols = np.where(lines.any(axis=0))[0]
    if len(cols) == 0:  # no staff found -> fall back to the ink bounding box
        cols = np.where((ink > 0).sum(axis=0) >= 2)[0]
        if len(cols) == 0:
            return None
        ys0, ys1 = 0, h - 1
    else:
        yy = np.where(lines.any(axis=1))[0]
        ys0, ys1 = int(yy.min()), int(yy.max())
    x0, x1 = int(cols.min()), int(cols.max())
    x0 = max(0, x0 - 2)
    x1 = min(w - 1, x1 + 2)

    # chord names / diagrams / tempo marks above or below the staff may start left of the
    # first staff line (or end right of the last one): widen the crop to keep them whole
    outside = ink > 0
    outside[max(0, ys0 - 2):ys1 + 3, :] = False
    reach = int(CFG.annotation_reach_fraction * w)
    lo = outside[:, max(0, x0 - reach):x0]
    if lo.sum() >= 10:
        x0 = max(0, x0 - reach + int(np.where(lo.any(axis=0))[0].min()) - 2)
    hi = outside[:, x1 + 1:min(w, x1 + 1 + reach)]
    if hi.sum() >= 10:
        x1 = min(w - 1, x1 + 1 + int(np.where(hi.any(axis=0))[0].max()) + 2)

    rows = np.where((ink[:, x0:x1 + 1] > 0).sum(axis=1) >= 2)[0]
    if len(rows) == 0:
        return None
    y0 = max(0, int(rows.min()) - pad)
    y1 = min(h - 1, int(rows.max()) + pad)
    return img[y0:y1 + 1, x0:x1 + 1]


def _extend_top(ink, ya, floor, x0, x1, gap_limit, max_up, sys_h=None):
    """Walk upward from the first staff keeping chord names, tempo marks and tall notes.
    Stop at a big blank gap, or at a detached BOLD text line (title / page header)."""
    lo = max(floor, ya - max_up)
    rows = np.where(ink[lo:ya, x0:x1 + 1].any(axis=1))[0] + lo
    if len(rows) == 0:
        return ya
    bands, s_, p_ = [], int(rows[0]), int(rows[0])
    for r in rows[1:]:
        if r - p_ > 2:
            bands.append((s_, p_))
            s_ = int(r)
        p_ = int(r)
    bands.append((s_, p_))
    detached = max(8, int(CFG.detached_header_gap_fraction * (sys_h or 150)))
    top = ya
    for bs, be in reversed(bands):
        gap = top - be - 1
        if gap > gap_limit:
            break
        if gap >= detached:
            sub = ink[bs:be + 1, x0:x1 + 1].astype(np.uint8)
            thick = cv2.erode(sub, np.ones((3, 3), np.uint8)).sum() / max(1, int(sub.sum()))
            if thick >= CFG.bold_header_erosion_fraction:
                break  # bold text line separated from the music: title or page header
        top = bs
    return top


def split_systems(img, first_on_sheet=False):
    """Split a whole-page screenshot (several staff systems stacked) into one image per system.

    A system = notation staff + tab staff (found via the tall bracket line at its left edge),
    or a lone tab staff when there is no bracket. Chord names / tempo marks above each system
    are kept; titles and credits are dropped.
    """
    h, w = img.shape
    ink_b = img < CFG.ink_threshold
    ink = ink_b.astype(np.uint8) * 255
    hl = _hlines((img < LINE_THR).astype(np.uint8) * 255, w)
    cols = np.where(hl.any(axis=0))[0]
    if len(cols) == 0:
        return []
    sx0 = int(cols.min())
    L = max(30, int(CFG.bracket_height_fraction * h))
    bx0 = max(0, sx0 - 14)
    band = ink[:, bx0: sx0 + 25]
    vert = cv2.morphologyEx(band, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    rows = np.where(vert.any(axis=1))[0]
    spans = []
    if len(rows):
        start = prev = int(rows[0])
        for r in rows[1:]:
            if r - prev > 8:
                spans.append((start, prev))
                start = int(r)
            prev = int(r)
        spans.append((start, prev))
    spans = [sp for sp in spans if sp[1] - sp[0] >= L]
    bracket_mode = bool(spans)
    sp_px = None
    if bracket_mode:
        bcols = np.where(vert.any(axis=0))[0]
        x0_common = max(0, bx0 + int(bcols.min()) - 3)
    else:
        # no bracket: tab-only systems -> cluster the staff lines
        cent = np.array(_line_centers(hl.any(axis=1)))
        if len(cent) < 4:
            return [img]
        g = np.diff(cent)
        sp_px = float(np.median(np.sort(g)[:max(3, len(g) // 2)]))
        clusters = np.split(cent, np.where(g > CFG.staff_cluster_gap * sp_px)[0] + 1)
        spans = [(int(c.min()), int(c.max())) for c in clusters if len(c) >= 4]
        if not spans:
            return [img]
        x0_common = None

    out = []
    prev_bot = -1
    for i, (ya, yb) in enumerate(spans):
        sys_h = yb - ya
        floor = (spans[i - 1][1] + 2) if i else 0
        hcols = np.where(hl[ya:yb + 1].any(axis=0))[0]
        x1 = min(w - 1, int(hcols.max()) + 2) if len(hcols) else w - 1
        x0 = x0_common if x0_common is not None else max(0, int(hcols.min()) - 2)
        if bracket_mode:
            gap_limit, max_up = max(8, int(CFG.system_top_gap_fraction * sys_h)), int(CFG.system_max_top_fraction * sys_h)
            bot_pad = CFG.system_bottom_fraction * sys_h
        else:
            gap_limit, max_up = max(6, int(CFG.tab_top_gap_spacing * sp_px)), int(CFG.tab_top_max_spacing * sp_px)
            bot_pad = CFG.tab_bottom_spacing * sp_px
        if i == 0:
            # above the first system there may be a title / credits / page header: walk up
            # over small text only (chord names, tempo, capo) and stop at the first title line
            top = _extend_top(ink_b, ya, floor, x0, x1, gap_limit, max_up, sys_h)
        else:
            # between two systems there is only chord names, dynamics, markers: keep it all
            top = max(prev_bot + 1, ya - max_up)
        bot = int(min(yb + bot_pad, (spans[i + 1][0] - 2) if i + 1 < len(spans) else h - 1))
        prev_bot = bot
        out.append(img[top:bot + 1, x0:x1 + 1])
    return out
