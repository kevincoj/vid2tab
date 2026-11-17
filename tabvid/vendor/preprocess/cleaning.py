from ..config import CFG
import cv2
import numpy as np
from ..config import LINE_THR
from ..layout.staves import _hlines

def trim_dark_bars(rgb, thresh=110, max_frac=0.4):
    """Remove dark video bars (e.g. YouTube header/footer) from top and bottom."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h = gray.shape[0]
    means = gray.mean(axis=1)
    top = 0
    while top < h * max_frac and means[top] < thresh:
        top += 1
    bot = h
    while bot > h * (1 - max_frac) and means[bot - 1] < thresh:
        bot -= 1
    return rgb[top:bot]


def trim_edges(rgb, max_px=10, ratio=0.93):
    """Remove thin dark card borders at the image edges."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ref = np.median(gray)
    h, w = gray.shape
    t = 0
    while t < max_px and gray[t].mean() < ratio * ref:
        t += 1
    b = 0
    while b < max_px and gray[h - 1 - b].mean() < ratio * ref:
        b += 1
    l = 0
    while l < max_px and gray[:, l].mean() < ratio * ref:
        l += 1
    r = 0
    while r < max_px and gray[:, w - 1 - r].mean() < ratio * ref:
        r += 1
    return rgb[t:h - b, l:w - r]


def _fill_runs(out, mask):
    """Replace masked pixels row by row with the lighter of the pixels just left/right of each run.
    Staff lines stay continuous, barlines don't get thicker, background stays white."""
    h, w = out.shape
    for r in range(h):
        idx = np.flatnonzero(mask[r])
        if len(idx) == 0:
            continue
        cuts = np.flatnonzero(np.diff(idx) > 1) + 1
        for run in np.split(idx, cuts):
            a, b = int(run[0]), int(run[-1])
            left = out[r, a - 1] if a > 0 else 255
            right = out[r, b + 1] if b + 1 < w else 255
            out[r, a:b + 1] = max(left, right)


def _normalize_teal_cursors(rgb, out):
    """Remove blue/teal cursor tint using visible contrast, not guessed ink.

    Isolated colored JPEG fringes are not cursors. In a persistent vertical band,
    the light/dark levels are measured in that column so glyph strokes still visible
    through the tint survive. No barline or musical shape is synthesized.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    teal = ((hsv[:,:,0] >= CFG.teal_hue_start) & (hsv[:,:,0] <= CFG.blue_hue_end) &
            (hsv[:,:,1] > CFG.teal_saturation_threshold) & (hsv[:,:,2] > CFG.teal_value_threshold))
    columns = np.flatnonzero(teal.sum(axis=0) >= CFG.cursor_column_fraction * out.shape[0])
    protected = np.zeros(out.shape[1], dtype=bool)
    if not len(columns):
        return protected
    for run in np.split(columns, np.where(np.diff(columns) > 3)[0]+1):
        left=max(0,int(run[0])-CFG.cursor_fringe_pixels)
        right=min(out.shape[1],int(run[-1])+CFG.cursor_fringe_pixels+1)
        band=out[:,left:right].astype(np.float32)
        white=np.percentile(band,CFG.cursor_white_percentile,axis=0)
        black=np.percentile(band,CFG.cursor_black_percentile,axis=0)
        contrast=white-black
        visible=contrast >= CFG.cursor_min_contrast
        normalized=np.full_like(band,255)
        normalized[:,visible]=np.clip((band[:,visible]-black[visible])*255/contrast[visible],0,255)
        # Restore paper/ink separation after removing the tint. Keep the
        # antialiased transition; do not expand masks or draw new strokes.
        normalized=np.clip((normalized-CFG.cursor_ink_point)*255 /
                           (CFG.cursor_paper_point-CFG.cursor_ink_point),0,255)
        out[:,left:right]=np.rint(normalized).astype(np.uint8)
        protected[left:right]=True
    return protected


def _handle_colors(rgb, out):
    """Remove colored UI; retain highlighted music with its original stroke edges."""
    h, w = out.shape
    cursor_columns = _normalize_teal_cursors(rgb, out)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int32)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    colored = ((sat > CFG.saturation_threshold) & (val > CFG.color_value_threshold)).astype(np.uint8)
    # A pale antialiased edge may have no near-black core within one pixel.
    # Include visible neutral-grey ink, without treating saturated UI colours
    # or the bright tinted paper as evidence of a glyph.
    source_ink = ((val < CFG.chroma_fringe_ink_value) |
                  ((val < CFG.chroma_fringe_gray_value) &
                   (sat < CFG.chroma_fringe_gray_saturation)))
    dark_neighbor = cv2.dilate(source_ink.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(colored, connectivity=8)
    remove = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, cw, chh, area = stats[i]
        if area < 3:
            continue
        comp = lab[y:y + chh, x:x + cw] == i
        hm = int(np.median(hue[y:y + chh, x:x + cw][comp]))
        long_vert = chh > CFG.long_overlay_fraction * h and cw < CFG.vertical_overlay_width * w
        long_horz = cw > CFG.long_overlay_fraction * w and chh < CFG.horizontal_overlay_height * h
        # JPEG chroma fringes can be red/yellow despite belonging to black ink.
        # Do not erase short, one/two-pixel components adjoining that ink.
        # Actual long cursors/progress bars still follow the overlay rules.
        thin_fringe = min(cw, chh) < CFG.chroma_fringe_min_extent
        adjoining_ink = dark_neighbor[y:y + chh, x:x + cw][comp].any()
        if thin_fringe and adjoining_ink and not (long_vert or long_horz):
            continue
        reddish = hm < CFG.red_hue_end or hm > CFG.red_hue_start          # red / orange annotations, progress bars
        yellowish = CFG.red_hue_end <= hm < CFG.yellow_hue_end
        if long_vert or long_horz or reddish or yellowish:
            remove[y:y + chh, x:x + cw][comp] = 1
        elif CFG.green_hue_start <= hm <= CFG.green_hue_end or CFG.blue_hue_start <= hm <= CFG.blue_hue_end:
            # The grayscale image already contains this music. Hard blackening
            # and dilation distort antialiased glyphs, especially colored TAB.
            continue
        else:                                      # teal / purple / other small bits
            remove[y:y + chh, x:x + cw][comp] = 1
    # Low-saturation teal noise outside long bands is not a removal mask.
    remove = cv2.dilate(remove, np.ones((3, 3), np.uint8))
    remove[:,cursor_columns] = 0
    _fill_runs(out, remove > 0)
    return out


def _darken_light_text(out):
    """Darken visible ink consistently, including glyphs crossing staff lines.

    Exempting a dilated line mask left pale horizontal slices in TAB digits
    and fingering. A pointwise curve changes shading without expanding strokes;
    paper and very light background pixels stay untouched.
    """
    sel = out < CFG.light_text_threshold
    f = out[sel].astype(np.float32) / 255.0
    out[sel] = np.clip(255.0 * f ** CFG.text_gamma, 0, 255).astype(np.uint8)


def _remove_broad_gray(out):
    """Whiten wide light/mid-gray regions (watermark letters, translucent highlight bands,
    player buttons). Thin staff lines and anti-aliased text are too narrow to survive the opening."""
    h, w = out.shape
    mid = ((out > CFG.broad_gray_low) & (out < CFG.broad_gray_high)).astype(np.uint8)
    k = max(7, w // CFG.broad_gray_width_divisor)
    blobs = cv2.morphologyEx(mid, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    blobs = cv2.dilate(blobs, np.ones((5, 5), np.uint8))
    out[(blobs > 0) & (out > CFG.broad_gray_low)] = 255


def _staff_end(hl, w):
    """Right end of the staff: median over the staff lines (robust to one overlay widening a few lines)."""
    ends = [int(np.flatnonzero(row)[-1]) for row in hl if row.any()]
    return int(np.median(ends)) if ends else w - 1


def _blank_box(out, hl, xe, x0, y0, x1, y1):
    """Erase a rectangular overlay, then redraw the staff lines (and the closing barline)
    that ran underneath it. hl = staff-line mask from before the erase, xe = last staff column."""
    h, w = out.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
    for r in range(y0, y1 + 1):
        left = out[r, max(0, x0 - 4):x0]
        was_line = x0 > 0 and hl[r, max(0, x0 - 6):x0].any()
        lv = int(left.min()) if left.size else 255
        out[r, x0:x1 + 1] = 255
        if was_line:
            out[r, x0:min(x1, xe) + 1] = min(lv, 200)
    if x0 <= xe <= x1 + 2:  # the right-hand barline passed behind the box
        top = out[max(0, y0 - 3), max(0, xe - 3):xe + 4]
        bot = out[min(h - 1, y1 + 3), max(0, xe - 3):xe + 4]
        if top.size and bot.size and top.min() < 150 and bot.min() < 150:
            col = max(0, xe - 3) + int(np.argmin(top))
            out[y0:y1 + 1, col] = 0


def _remove_gray_boxes(out):
    """Remove solid mid-gray rounded rectangles (player buttons, progress dots)."""
    h, w = out.shape
    hl = _hlines((out < LINE_THR).astype(np.uint8) * 255, w, frac=0.10) > 0
    xe = _staff_end(hl, w)
    m = ((out > CFG.gray_box_low) & (out < CFG.gray_box_high)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))   # thin staff lines vanish
    m = cv2.dilate(m, np.ones((13, 13), np.uint8))                        # bridge the icons inside
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, cw, chh, area = stats[i]
        if cw >= CFG.gray_box_min_width_fraction * w and chh >= CFG.gray_box_min_height and cw >= CFG.gray_box_min_aspect * chh and cw <= CFG.gray_box_max_width_fraction * w:
            boxes.append((x - 2, y - 2, x + cw + 2, y + chh + 2))
    for bx in boxes:
        _blank_box(out, hl, xe, *bx)


def _remove_overlay_boxes(out):
    """Remove video-player thumbnails / popups: thick rectangular frames, found as two long
    thick horizontal bars with the same x-range joined by vertical edges. (Works even when
    note stems or staff lines touch the frame.)"""
    h, w = out.shape
    hl = _hlines((out < LINE_THR).astype(np.uint8) * 255, w, frac=0.10) > 0
    xe = _staff_end(hl, w)
    ink = (out < CFG.ink_threshold).astype(np.uint8)
    L = max(CFG.thumbnail_min_width, int(CFG.thumbnail_width_fraction * w))
    bars = cv2.morphologyEx(ink * 255, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (L, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bars, connectivity=8)
    cands = [tuple(int(v) for v in stats[i][:4]) for i in range(1, n)
             if stats[i][3] <= 10 and stats[i][2] >= L and stats[i][2] <= CFG.thumbnail_max_width_fraction * w]
    found, used = 0, set()
    for ia, (ax, ay, aw, ah) in enumerate(cands):
        for ib, (bx, by, bw, bh) in enumerate(cands):
            if ib <= ia or ia in used or ib in used:
                continue
            gap = by - (ay + ah)
            if abs(ax - bx) <= 8 and abs(aw - bw) <= 12 and 12 <= gap <= CFG.thumbnail_max_height_fraction * h:
                y0, y1 = ay + ah, by
                left = ink[y0:y1, ax:ax + 6].any(axis=1).mean()
                right = ink[y0:y1, ax + aw - 6:ax + aw].any(axis=1).mean()
                if left > CFG.thumbnail_edge_coverage and right > CFG.thumbnail_edge_coverage:
                    _blank_box(out, hl, xe, ax - 8, ay - 8, ax + aw + 8, by + bh + 8)
                    used.update((ia, ib))
                    found += 1
    return found


def _remove_isolated_edge_rules(out):
    from ..layout.staves import _line_centers
    h,w=out.shape
    centers=np.array(_line_centers(_hlines((out<LINE_THR).astype(np.uint8)*255,w,frac=.1).any(axis=1)))
    inner=centers[(centers>CFG.card_edge_rows)&(centers<h-1-CFG.card_edge_rows)]
    gaps=np.diff(inner)
    if not len(gaps):
        return
    spacing=float(np.median(np.sort(gaps)[:max(1,len(gaps)//2)]))
    for y in list(range(min(CFG.card_edge_rows,h)))+list(range(max(0,h-CFG.card_edge_rows),h)):
        gray=(out[y]>CFG.broad_gray_low)&(out[y]<CFG.broad_gray_high)
        if gray.mean() >= CFG.card_rule_coverage and np.min(abs(inner-y)) > CFG.card_rule_staff_clearance*spacing:
            out[y,gray]=255


def clean(rgb, decolor=True):
    """Return a grayscale uint8 image with a pure-white background and black ink."""
    rgb = trim_dark_bars(rgb)
    rgb = trim_edges(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # estimate the background (gradients, tints, highlights) by "closing" away the ink
    f = max(1, w // CFG.background_sample_width)
    small = cv2.resize(gray, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ker)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    norm = np.clip(gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0, 0, 255)
    out = np.clip((norm - CFG.black_point) * 255.0 / (CFG.white_point - CFG.black_point), 0, 255).astype(np.uint8)

    if decolor:
        out = _handle_colors(rgb, out)

    # remove tiny specks that touch the image edge (rounded card corners, border remnants)
    ink = (out < CFG.ink_threshold).astype(np.uint8)
    n, labels_, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    lim = CFG.edge_speck_fraction * min(h, w)
    for i in range(1, n):
        x, y, cw, chh, area = stats[i]
        touches = x <= 1 or y <= 1 or x + cw >= w - 1 or y + chh >= h - 1
        if touches and max(cw, chh) <= lim and area < CFG.edge_speck_area:
            out[y:y + chh, x:x + cw][labels_[y:y + chh, x:x + cw] == i] = 255

    _remove_broad_gray(out)
    _remove_gray_boxes(out)
    _remove_overlay_boxes(out)
    _remove_isolated_edge_rules(out)
    _darken_light_text(out)
    return out
