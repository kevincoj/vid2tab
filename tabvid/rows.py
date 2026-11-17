"""Stage 3: crop every frame, group consecutive frames showing the same row, merge each row.

1. Crop each frame and find the TAB staff and barlines (staff.find_staff).
2. Walk frames in order; a frame joins the current row if same_row() says it shows
   the same thing as the row's last good frame. Frames with no staff (title cards, the
   outro, a full-screen shot of the player) are left out and listed in the report.
3. Drop transition frames before merging:
   - fading: the row's ink is noticeably lighter than in the row's typical frame;
   - ghosting: ink from another row shows through where the row is blank paper;
   - a one-frame "row" that is just a blend of the rows before and after it.
4. Merge the kept frames by taking the lightest value of each pixel. A moving cursor
   is dark in only a few frames, so it disappears; printed ink is dark in all of them.
   The lightest value is taken per colour channel, so a coloured cursor that crossed a
   staff line or digit leaves that channel bright (a green dash on a black line). Inside the
   staff, a tinted pixel that is dark in some channel is set to its darkest channel.
   Mask boxes from config.json (watermarks, overlays) are painted white in every frame
   before anything else; after merging they are filled with blank staff.
5. Vote whether the video scrolls (scroll.py): consecutive rows that repeat each other's
   edge at a shift. If it does, each merged row is a "view" (views/, views.json) and the
   views are stitched into strips, which become the rows stage 4 cuts (rows/, rows.json).
   A page-switching video's rows are written as they are.
"""
from dataclasses import dataclass, field
import json

import cv2
import numpy as np

from . import scroll
from .staff import (blank_staff, extension_limit, find_staff, find_staves, near_limit, notation_staff, shows_staff,
                    staff_gaps, unread_lines)
from .vendor.same_row import same_row

FADE = 20          # drop a frame whose ink is this many gray levels lighter than the row's typical frame
GHOST = .05        # ...or whose foreign ink on blank paper is over 5% of the row's ink
OVERLAY_SAMPLES = 80   # frames (with a staff) sampled to model a mask box's overlay
OVERLAY_PCT = 90       # a pixel's percentile over them: what it shows over plain paper
OVERLAY_MARK = 8       # overlay: that is this many gray levels darker than the paper...
OVERLAY_LIGHT = 4      # ...or this many lighter (a white glow)
OVERLAY_REACH = 6      # px beyond a mask box searched for the rest of the overlay
OVERLAY_NOISE = 10     # gray levels of frame noise; reversed, it grows by 1 / (1 - alpha)
RINGING = 3            # px of compression ringing around an overlay's edges
OPAQUE = .2            # an overlay pixel letting through less than this of what is under it is paper
FADE_PAPER = 40    # leave out a frame whose paper is this many gray levels darker than the video's (a fade)
BLEND_RESIDUAL = 6 # a one-frame row this close to a mix of its neighbours is a transition
SLIDE_MISSING = .05  # a one-frame row lacking under this share of a neighbouring row's ink is that page sliding
END_INK = .002       # a frame after the last staff with ink over this share of the crop (not blank)...
COVER_HIDDEN = 2.0   # spacings of staff hidden in one row but shown in the next (an overlay over the page)...
COVER_WIDTH = .6     # ...inside one band of differences at most this share of the width...
COVER_KEEP = .98     # ...(the band holds this share of the differing ink); outside it the rows match
TINT_FLOOR = 160   # a tinted pixel whose darkest channel is at least this is paper...
TINT = 28          # ...if it is clearly tinted (same colour spread same_row treats as overlay)
INK_SPREAD = 60    # colour spread of saturated ink (JPEG fringes on black ink stay under it)...
INK_LUMA = 160     # ...darker than this luma (paler tints are highlights)
NOTATION_ROWS = .6  # a video is notation + TAB if a notation staff sits over the TAB in this share of rows
STACKED_SHARE = .6  # stacked staves (two or more TAB rows per page) if this share of the frames with a staff
                    # show two or more (docs/multi-row-plan.md)
BAND_INK = 30       # splitting a page: ink is anything this many gray levels under the paper (pale numbers too)
BAND_MARGIN = 8     # px of paper kept around a band's ink
ROLL_SHARE = .5     # stacked pages "roll" (the lower row moves up, a new one comes in under it) if this share
                    # of the page changes do; then a page's top row equal to the previous page's bottom row is dropped
ROLL_BAND = 1.5     # spacings above and below a staff compared when a one-frame page is caught mid-roll


@dataclass
class View:
    index: int
    seconds: float
    rgb: np.ndarray
    staff: object            # staff.Staff or None (the lowest staff)
    dropped: str = ''        # reason, empty if kept
    staves: tuple = ()       # every TAB staff in the frame, top to bottom (staff.find_staves)

    # same_row() reads these three attributes
    @property
    def bars(self):
        return self.staff.bars

    @property
    def lines(self):
        return self.staff.lines


@dataclass
class Row:
    views: list = field(default_factory=list)

    @property
    def kept(self):
        return [v for v in self.views if not v.dropped]

    @property
    def staffed(self):
        return [v for v in self.views if v.staff is not None]


def _gray(view):
    return cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def untint(rgb):
    """Turn light tinted pixels (a yellow or pale blue cursor highlight) into plain paper.

    Saturated or dark colour (a cursor line, coloured note heads) is left alone, so same_row
    still ignores it as an overlay; only a pale wash over paper counts as paper.
    """
    light = (rgb.min(axis=2) >= TINT_FLOOR) & (np.ptp(rgb, axis=2) >= TINT)
    out = rgb.copy()
    out[light] = rgb[light].max(axis=1, keepdims=True)
    return out


def crop_masks(box, masks):
    """Mask boxes (frame coordinates) as (left, top, right, bottom) inside the crop, clipped."""
    left, top, right, bottom = box
    out = []
    for l, t, r, b in masks:
        l, t, r, b = max(l, left) - left, max(t, top) - top, min(r, right) - left, min(b, bottom) - top
        if r > l and b > t:
            out.append((l, t, r, b))
    return out


def overlay_masks(frames, box, boxes, inverted=False):
    """Model the translucent overlay (logo, watermark) inside each mask box, so it can be removed
    from every frame and the music under it recovered. Returns [(l, t, r, b, opaque, alpha,
    colour, probe, level, paper, logo)] in crop coordinates, the box grown by OVERLAY_REACH.

    The overlay stays put while the music moves under it: observed = alpha * colour + (1 -
    alpha) * under, per pixel. Two observations per pixel, over frames showing the overlay:
    - over (OVERLAY_PCT percentile): the pixel over plain paper, alpha * colour + (1 - alpha) * paper;
    - darkest: the pixel with ink under it, alpha * colour + (1 - alpha) * ink.
    So 1 - alpha = (over - darkest) / (paper - ink), and colour follows. Overlay pixels differ
    from the paper (darker by OVERLAY_MARK, or lighter by OVERLAY_LIGHT: a white glow) and
    connect to the box. A pixel no ink ever passed under reads as opaque: painting it paper loses
    nothing. Opaque pixels (letting through under OPAQUE) are painted paper.
    A frame without the overlay (V5's watermark fades in at frame 17; V4's logo is gone at the
    end) is told by `probe` pixels lighter than `level`; it is left out here and left alone.
    """
    if not boxes:
        return []
    picks = np.linspace(0, len(frames) - 1, min(OVERLAY_SAMPLES, len(frames))).round().astype(int)
    crops = []
    for i in picks:
        rgb = load_crop(frames[i], box, [], inverted)
        if find_staff(rgb) is not None:
            crops.append(rgb)
    out = []
    for l, t, r, b in boxes:
        if not crops:
            out.append((l, t, r, b, np.ones((b - t, r - l), bool)))
            continue
        height, width = crops[0].shape[:2]
        L, T = max(0, l - OVERLAY_REACH), max(0, t - OVERLAY_REACH)
        R, B = min(width, r + OVERLAY_REACH), min(height, b + OVERLAY_REACH)
        stack = np.stack([c[T:B, L:R] for c in crops]).astype(np.float32)
        paper = np.median([np.median(c.reshape(-1, 3), axis=0) for c in crops], axis=0)
        paper_gray = paper.mean()
        gray = stack.mean(axis=3)
        # Which frames show the overlay: its clearly dark pixels (median over frames) are dark.
        typical = np.median(gray, axis=0)
        probe = paper_gray - typical > 4 * OVERLAY_MARK
        level = (np.median(typical[probe]) + paper_gray) / 2 if probe.any() else 255.0
        shown = [k for k in range(len(stack)) if not probe.any() or np.median(gray[k][probe]) <= level]
        stack, gray = stack[shown], gray[shown]
        over = np.percentile(stack, OVERLAY_PCT, axis=0)
        over_gray = over.mean(axis=2)
        mark = (paper_gray - over_gray > OVERLAY_MARK) | (over_gray - paper_gray > OVERLAY_LIGHT)
        _, labels = cv2.connectedComponents(mark.astype(np.uint8), connectivity=8)
        inside = np.unique(labels[t - T:b - T, l - L:r - L])
        logo = np.isin(labels, inside[inside > 0])
        darkest = stack.min(axis=0)
        dark = np.concatenate([c[c.mean(axis=2) < paper_gray / 2] for c in crops[:5]])
        ink = np.percentile(dark, 5, axis=0) if len(dark) else np.zeros(3, np.float32)
        keep = np.clip((over_gray - darkest.mean(axis=2)) / max(paper_gray - ink.mean(), 1.0), 0, 1)
        alpha = np.where(logo, 1 - keep, 0).astype(np.float32)
        a = np.maximum(alpha, 1e-3)[..., None]
        colour = np.clip((over - (1 - a) * paper) / a, 0, 255).astype(np.float32)
        opaque = logo & (1 - alpha < OPAQUE)
        out.append((L, T, R, B, opaque, alpha, colour, probe, float(level),
                    paper.astype(np.float32), logo))
    return out


def unblend(rgb, masks):
    """Remove the mask overlays from one crop (in place): recover what is under each translucent
    pixel, paint opaque ones paper. A plain (l, t, r, b, pixels) mask is painted white."""
    for l, t, r, b, opaque, *model in masks:
        region = rgb[t:b, l:r]
        fill = 255
        if model:
            alpha, colour, probe, level, paper, logo = model
            if probe.any() and np.median(region[probe].mean(axis=1)) > level:
                continue                                        # the overlay is not in this frame
            # Nothing under the overlay is lighter than paper: capping its surroundings too removes
            # compression ringing around the logo's edges, which the lightest-pixel merge keeps.
            near = cv2.dilate(logo.astype(np.uint8), np.ones((2 * RINGING + 1,) * 2, np.uint8)) > 0
            region[near] = np.minimum(region[near], paper.round().astype(np.uint8))
            a = alpha[logo][:, None]
            keep = np.maximum(1 - a, OPAQUE)
            under = np.clip((region[logo] - a * colour[logo]) / keep, 0, paper)
            # Frame noise grows by 1 / keep when the blend is reversed: within it, it is paper.
            noise = (paper - under).mean(axis=1, keepdims=True) < OVERLAY_NOISE / keep
            region[logo] = np.where(noise, paper, under).round().astype(np.uint8)
            fill = paper.round().astype(np.uint8)
        region[opaque] = fill
    return rgb


def load_crop(path, box, masks, inverted=False):
    """The cropped frame as RGB (inverted for light-on-dark TAB), mask overlays removed, light
    tints turned to paper."""
    left, top, right, bottom = box
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)[top:bottom, left:right].copy()
    if inverted:
        rgb = 255 - rgb
    return untint(unblend(rgb, masks))


def clean_merged(rgb, staff, masks):
    """Undo what a per-channel lightest-pixel merge leaves inside the staff, and fill mask boxes.

    A coloured cursor crossing a staff line raises one channel there (a green dash on a black
    line): inside the staff band a clearly tinted pixel that is dark in some channel takes its
    darkest channel. Opaque overlay pixels (see overlay_masks), painted white in every frame,
    are filled with blank staff, so the staff lines run through them.
    """
    out = rgb.copy()
    y0, y1 = max(0, round(staff.lines[0]) - 2), round(staff.lines[-1]) + 3
    band = out[y0:y1]
    tinted = (np.ptp(band, axis=2) >= TINT) & (band.min(axis=2) < TINT_FLOOR)
    band[tinted] = band[tinted].min(axis=1, keepdims=True)
    if masks:
        column = blank_staff(out)
        for l, t, r, b, pixels, *_ in masks:
            region = out[t:b, l:r]
            region[pixels] = np.broadcast_to(column[t:b, None], region.shape)[pixels]
    return out, int(tinted.sum())


def group_rows(frames, box, fps, masks=(), inverted=False):
    """(rows, indices of frames with no staff, indices of fade frames, stacked).

    stacked: (on, frames showing two or more staves). Stacked staves (STACKED_SHARE of the frames
    with a staff show two or more TAB staves) are decided once per video; a row is then a page, and
    a frame joins it only if it also shows the same number of staves with the same barlines in each.

    A fade to or from black can leave the staff visible on darkened paper: such a frame is left
    out like a frame without a staff. It is a fade if its paper (median gray of the crop, after
    any polarity inversion) is more than FADE_PAPER levels darker than the video's paper (the
    median over frames with a staff). A row's own median cannot catch it: at the end of a video
    the fade frame is often a row by itself.
    """
    views, no_staff = [], []
    for index, path in enumerate(frames):
        rgb = load_crop(path, box, masks, inverted)
        staves = tuple(find_staves(rgb))
        view = View(index, index / fps, rgb, staves[-1] if staves else None, staves=staves)
        if view.staff is None:
            no_staff.append(index)
        else:
            views.append(view)
    papers = [float(np.median(cv2.cvtColor(v.rgb, cv2.COLOR_RGB2GRAY))) for v in views]
    level = float(np.median(papers)) if papers else 255.0
    multi = [v.index for v in views if len(v.staves) >= 2]
    stacked = bool(views) and len(multi) >= STACKED_SHARE * len(views)
    rows, fading = [], []
    for view, paper in zip(views, papers):
        if paper < level - FADE_PAPER:
            fading.append(view.index)
            continue
        if rows and same_row(rows[-1].staffed[-1], view) and (not stacked or _same_staves(rows[-1].staffed[-1], view)):
            rows[-1].views.append(view)
        else:
            rows.append(Row([view]))
    return rows, no_staff, fading, (stacked, multi)


def _same_staves(a, b):
    """Two frames of stacked staves show the same page: as many staves, each with the same barlines
    (same_row's 2 px) at the same height."""
    return len(a.staves) == len(b.staves) and all(
        len(p.bars) == len(q.bars) and all(abs(x - y) <= 2 for x, y in zip(p.bars, q.bars))
        and abs(p.lines[0] - q.lines[0]) <= 2 for p, q in zip(a.staves, b.staves))


def staff_after_end(frames, box, masks, inverted, last, paper):
    """Indices of frames after `last` (the last frame with a staff) that still show TAB: ink in the
    crop forming six evenly spaced staff lines of any length (staff.shows_staff), on the video's paper.

    K's last page (one measure, 230 px of a 1918 px crop) was not taken as a staff, so its measure
    was lost without a word. Blank frames (ink under END_INK of the crop), fades (paper more than
    FADE_PAPER darker) and outros without staff lines (a performer, a title card) stay silent.
    """
    out = []
    for index in range(last + 1, len(frames)):
        rgb = load_crop(frames[index], box, masks, inverted)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        level = float(np.median(gray))
        if level < paper - FADE_PAPER or (gray < level - 60).mean() < END_INK:
            continue
        if shows_staff(rgb):
            out.append(index)
    return out


def _ranges(indices, fps, total):
    """'#0-#16 (0.0-8.0s, start)' style spans of consecutive frame indices."""
    spans = []
    for run in np.split(np.array(indices), np.where(np.diff(indices) > 1)[0] + 1) if indices else []:
        where = 'start' if run[0] == 0 else 'end' if run[-1] == total - 1 else 'middle'
        spans.append(f'#{run[0]}-#{run[-1]} ({run[0] / fps:.1f}-{run[-1] / fps:.1f}s, {where})')
    return spans


def _drop_fading_frames(row):
    """Mark frames whose ink is fading or overlaid with another row's ink."""
    candidates = [v for v in row.views if not v.dropped]
    if len(candidates) < 3:
        return
    reference = np.median([_gray(v) for v in candidates], axis=0)
    ink, paper = reference < 110, reference > 220
    if not ink.any():
        return
    for view in candidates:
        gray = _gray(view)
        fade = float(gray[ink].mean() - reference[ink].mean())
        ghost = float(((gray < 150) & paper).sum() / ink.sum())
        if fade > FADE:
            view.dropped = f'fading (ink {fade:.0f} levels lighter)'
        elif ghost > GHOST:
            view.dropped = f'ghost ink from another row ({ghost:.0%})'


def _covered_pages(frames, box, masks, inverted, fading, rows):
    """Fade frames (left out: paper much darker than the video's) that show a staff whose barlines match
    no frame of any row (same count, each within BAR_TOL): a page shown only under a dark or coloured
    overlay. A fade in or out of a page that is also shown plainly matches its row and is not listed."""
    known = {tuple(v.bars) for row in rows for v in row.staffed}
    out = []
    for i in fading:
        staff = find_staff(load_crop(frames[i], box, masks, inverted))
        if staff is None or len(staff.bars) < 2:
            continue
        if not any(len(bars) == len(staff.bars) and all(abs(a - b) <= scroll.BAR_TOL for a, b in zip(bars, staff.bars))
                   for bars in known):
            out.append(i)
    return out


def _is_blend(view, before, after):
    """True if a single frame is a crossfade between the rows on either side of it."""
    if before is None or after is None:
        return False
    gray, a, b = _gray(view), _gray(before.kept[-1] if before.kept else before.staffed[-1]), \
        _gray(after.kept[0] if after.kept else after.staffed[0])
    best_alpha, best = None, float('inf')
    for alpha in np.linspace(0, 1, 11):
        residual = float(np.abs(gray - (alpha * a + (1 - alpha) * b)).mean())
        if residual < best:
            best_alpha, best = alpha, residual
    return 0 < best_alpha < 1 and best < BLEND_RESIDUAL


def _is_slide(view, neighbour):
    """True if a single frame shows a neighbouring row's page moved up or down: a slide transition.

    The staff sits more than MAX_DY px higher or lower. Aligned on the staff lines, every barline has a partner and the neighbour's ink, in the pixel
    rows both show, is in the frame too. (The other way round would fail on the frame's cursor,
    which a single frame cannot lose, and on whatever slid in from the far side.)
    """
    if neighbour is None or view.staff is None:
        return False
    other = neighbour.kept[-1] if neighbour.kept else neighbour.staffed[-1]
    if other.rgb.shape != view.rgb.shape or len(other.bars) != len(view.bars) or not other.bars or any(
            abs(a - b) > scroll.BAR_TOL for a, b in zip(other.bars, view.bars)):
        return False
    dy = round(view.lines[0] - other.lines[0])
    if abs(dy) <= scroll.MAX_DY:
        return False                                           # not moved: the same-view merge decides
    height = view.rgb.shape[0]
    top, bottom = max(0, -dy), min(height, height - dy)        # the neighbour's rows the frame also shows
    if top > round(other.lines[0]) or bottom <= round(other.lines[-1]):
        return False                                           # the staff itself must be in both
    ink = (_gray(other) < scroll.INK) & (np.ptp(other.rgb, axis=2) < TINT)
    for y in other.lines:
        ink[max(0, round(y) - 1):round(y) + 2] = False
    ink = ink[top:bottom]
    there = cv2.dilate((_gray(view) < scroll.INK).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    there = there[top + dy:bottom + dy]
    return bool(ink.any()) and float((ink & ~there).sum() / ink.sum()) <= SLIDE_MISSING


def _is_roll(view, before, after):
    """True if a one-frame page of stacked staves is caught mid-roll (or mid-slide): its staves are the
    staves of one neighbouring page, all moved up or down by one distance (more than MAX_DY px), each with
    the same barlines and with that staff's ink (ROLL_BAND spacings around it, in the pixel rows both
    show) in the frame too."""
    others = [r.kept[-1] if r.kept else r.staffed[-1] for r in (before,) if r] + \
        [r.kept[0] if r.kept else r.staffed[0] for r in (after,) if r]
    if not view.staves:
        return False
    there = cv2.dilate((_gray(view) < scroll.INK).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    height = view.rgb.shape[0]

    def moved_copy(staff, other, partner, dy):
        if len(partner.bars) != len(staff.bars) or not staff.bars or any(
                abs(a - b) > scroll.BAR_TOL for a, b in zip(partner.bars, staff.bars)):
            return False
        if abs(staff.lines[0] - partner.lines[0] - dy) > scroll.MAX_DY:
            return False
        band = round(ROLL_BAND * partner.spacing)
        top = max(0, round(partner.lines[0]) - band, -dy)
        bottom = min(height, round(partner.lines[-1]) + band + 1, height - dy)
        if bottom - top < 2:
            return False
        ink = (_gray(other) < scroll.INK) & (np.ptp(other.rgb, axis=2) < TINT)
        for y in partner.lines:
            ink[max(0, round(y) - 1):round(y) + 2] = False
        ink = ink[top:bottom]
        return bool(ink.any()) and float((ink & ~there[top + dy:bottom + dy]).sum() / ink.sum()) <= SLIDE_MISSING

    for other in others:
        if other.rgb.shape != view.rgb.shape:
            continue
        for first in other.staves:
            dy = round(view.staves[0].lines[0] - first.lines[0])
            if abs(dy) > scroll.MAX_DY and all(any(moved_copy(staff, other, partner, dy) for partner in other.staves)
                                               for staff in view.staves):
                return True
    return False


def coloured_ink(rgb):
    """Pixels drawn in saturated colour darker than a pale tint: a note shown red while played,
    a cursor line. Pale tints (a highlight band, a light cursor wash) have luma >= INK_LUMA."""
    luma = rgb.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    return (np.ptp(rgb, axis=2) >= INK_SPREAD) & (luma < INK_LUMA)


def merge(row):
    """Lightest pixel per channel over the kept frames, leaving out coloured ink where another
    frame shows the pixel without it.

    A note drawn red while it is played is black (or the cursor is elsewhere) in the other frames
    of its row, so it merges as black, not as the per-channel maximum (red). A moving cursor
    gives way to the frames without it, as before. A pixel coloured in every frame (a cursor that
    stays put while the music scrolls under it, coloured chord names) keeps the plain maximum:
    darkening it would turn a still cursor line into a barline.
    """
    kept = row.kept or row.staffed
    stack = np.stack([v.rgb for v in kept])
    plain = stack.max(axis=0)
    coloured = np.stack([coloured_ink(v.rgb) for v in kept])
    if not coloured.any():
        return plain
    neutral = np.where(coloured[..., None], 0, stack).max(axis=0)
    some = (~coloured).any(axis=0)
    return np.where(some[..., None], neutral, plain).astype(plain.dtype)


def _merged_row(row, masks):
    """(merged image, its staff, tinted pixels reset) for a row."""
    image = merge(row)
    staff = find_staff(image) or row.staffed[0].staff
    image, count = clean_merged(image, staff, masks)
    return image, staff, count


def _lines_shown(gray, lines):
    """Per column, how many staff lines are dark there (within 1 px of the line)."""
    return np.array([(gray[max(0, round(y) - 1):round(y) + 2] < scroll.INK).any(axis=0) for y in lines]).sum(axis=0)


def covered_copy(a_image, a_staff, b_image, b_staff):
    """True if row A is row B's page partly covered by an overlay (I: a picture of fingering hands
    over the last measure of the first page, then the same page clean).

    The rows must show the same staff (lines within 2 px). All their differences lie in one band of
    columns (COVER_KEEP of the differing ink) at most COVER_WIDTH of the width; in that band A hides
    the staff lines B shows over COVER_HIDDEN spacings or more (music never interrupts the staff; an
    opaque picture does), and outside it the ink matches like a same view (scroll.CLOSE) with every
    barline matched. Two different pages sharing their first measures differ in music, not in a
    hidden staff, so they are not taken for one.
    """
    if a_image.shape != b_image.shape or len(a_staff.lines) != len(b_staff.lines) or any(
            abs(p - q) > 2 for p, q in zip(a_staff.lines, b_staff.lines)):
        return False
    a, b = cv2.cvtColor(a_image, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b_image, cv2.COLOR_RGB2GRAY)
    a_ink, b_ink = a < scroll.INK, b < scroll.INK
    near = np.ones((3, 3), np.uint8)
    differ = (a_ink & ~cv2.dilate(b_ink.astype(np.uint8), near).astype(bool)) | \
             (b_ink & ~cv2.dilate(a_ink.astype(np.uint8), near).astype(bool))
    columns = differ.sum(axis=0)
    if not columns.sum():
        return False
    cumulative = np.cumsum(columns) / columns.sum()
    x0 = int(np.searchsorted(cumulative, (1 - COVER_KEEP) / 2))
    x1 = int(np.searchsorted(cumulative, 1 - (1 - COVER_KEEP) / 2)) + 1
    pad = round(a_staff.spacing)                         # the band's own edges (a frame) lie in the trimmed tails
    x0, x1 = max(0, x0 - pad), min(a.shape[1], x1 + pad)
    if x1 - x0 > COVER_WIDTH * a.shape[1]:
        return False
    hidden = (_lines_shown(b, b_staff.lines) >= len(b_staff.lines) - 1) & (_lines_shown(a, a_staff.lines) <= len(a_staff.lines) // 2)
    if hidden[x0:x1].sum() < COVER_HIDDEN * a_staff.spacing or hidden[:x0].any() or hidden[x1:].any():
        return False
    outside = np.ones(a.shape[1], bool)
    outside[x0:x1] = False
    ink = (a_ink | b_ink)[:, outside].sum()
    if not ink or differ[:, outside].sum() > scroll.CLOSE * ink:
        return False
    bars = lambda staff: [bar.x for bar in staff.barlines if not x0 - 2 <= bar.x <= x1 + 2]
    a_bars, b_bars = bars(a_staff), bars(b_staff)
    return len(a_bars) == len(b_bars) and all(abs(p - q) <= 2 for p, q in zip(a_bars, b_bars))


def _merge_same_views(rows, merged, masks):
    """Merge consecutive rows that show the same view (one page split by a cursor on a barline).

    same_row() vetoes only coloured overlays, so a neutral playhead blob, or a cursor that makes
    a barline fail the barline rule in a few frames, starts a new row on the same page. The
    merged rows are compared like the scroll vote's "same view" join: shift 0, every barline
    matched, ink mismatch within CLOSE and no worse than the best real shift. Two rows that
    genuinely show the same page twice in a row are indistinguishable and merge too.
    A page shown partly covered by an overlay next to its clean copy (covered_copy) is one row too:
    the clean copy's image is kept and the covered frames are dropped.
    Returns (rows, merged, ["a+b" labels of same views], ["a+b" labels of covered copies]), in
    original row numbers.
    """
    out_rows, out_merged, numbers, covers = [rows[0]], [merged[0]], [[1]], []
    for number, (row, (image, staff)) in enumerate(zip(rows[1:], merged[1:]), 2):
        last_image, last_staff = out_merged[-1]
        found, same = scroll.candidates(cv2.cvtColor(last_image, cv2.COLOR_RGB2GRAY), last_staff,
                                        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), staff)
        best = min((c.mismatch for c in found if c.confident), default=1.0)
        if same is not None and same.bars_ok and same.mismatch <= min(scroll.CLOSE, best):
            combined = Row(out_rows[-1].views + row.views)
            _drop_fading_frames(combined)
            image, staff, _ = _merged_row(combined, masks)
            out_rows[-1], out_merged[-1] = combined, (image, staff)
            numbers[-1].append(number)
        elif covered_copy(last_image, last_staff, image, staff) or covered_copy(image, staff, last_image, last_staff):
            # One page shown partly covered and then clean (or the other way): the clean copy is kept
            # as it is (a lightest-pixel merge would let a white overlay erase the music under it).
            covered = out_rows[-1] if covered_copy(last_image, last_staff, image, staff) else row
            for view in covered.views:
                view.dropped = view.dropped or 'covered copy of the next/previous page'
            combined = Row(out_rows[-1].views + row.views)
            if covered is out_rows[-1]:
                out_merged[-1] = (image, staff)
            out_rows[-1] = combined
            numbers[-1].append(number)
            covers.append('+'.join(map(str, numbers[-1])))
        else:
            out_rows.append(row)
            out_merged.append((image, staff))
            numbers.append([number])
    joined = ['+'.join(map(str, group)) for group in numbers if len(group) > 1 and '+'.join(map(str, group)) not in covers]
    return out_rows, out_merged, joined, covers


def build_rows(frames, box, fps, run_dir, report, masks=(), inverted=False):
    masks = overlay_masks(frames, box, crop_masks(box, masks), inverted)
    rows, no_staff, fading, (stacked, multi) = group_rows(frames, box, fps, masks, inverted)
    if multi:
        staffed_count = sum(len(r.staffed) for r in rows) + len(fading)
        report.stage('rows', staves_per_frame=f'2 or more in {len(multi)} of {staffed_count} frames with a staff',
                     stacked_staves='on: each page is split into one row per staff' if stacked else 'off')
        if not stacked:
            report.warn('rows', f'frames {_ranges(multi, fps, len(frames))} ({len(multi)} of {staffed_count} frames '
                        f'with a staff) show two or more TAB staves, too few for stacked mode (needs '
                        f'{STACKED_SHARE:.0%}): only the lowest staff of those frames is read; the staves above '
                        'it are kept as annotation, as printed.')
    if not rows:
        report.warn('rows', f'No TAB staff found in any of the {len(frames)} frames; check the crop. '
                    'The PDF will be empty.')
        (run_dir / 'rows.json').write_text('[]\n')
        report.stage('rows', rows=0, frames=len(frames), frames_without_staff=_ranges(no_staff, fps, len(frames)),
                     frames_fading=_ranges(fading, fps, len(frames)) or 'none')
        return []
    staffed = [v for row in rows for v in row.staffed]
    paper = float(np.median([np.median(cv2.cvtColor(v.rgb, cv2.COLOR_RGB2GRAY)) for v in staffed]))
    after_end = staff_after_end(frames, box, masks, inverted, max(v.index for v in staffed), paper)
    if after_end:
        report.warn('rows', f'frames {_ranges(after_end, fps, len(frames))} after the last measure still show staff '
                    'lines that were not read as TAB (too short, or no barlines): the sheet may be missing its ending.')
    # Mid-video only (between the first and last frames with a staff): before and after, intros, outros and
    # fades over photos are common, and the ending has its own check (staff_after_end).
    spacing = float(np.median([v.staff.spacing for row in rows for v in row.staffed]))
    shown = [v.index for row in rows for v in row.staffed]
    unread = [i for i in no_staff if min(shown) < i < max(shown)
              and unread_lines(load_crop(frames[i], box, masks, inverted), spacing)]
    if unread:
        report.warn('rows', f'frames {_ranges(unread, fps, len(frames))} ({len(unread)} frames) show long horizontal '
                    'lines but no TAB staff was found in them: music shown only there is missing from the sheet '
                    '(staff lines broken up, or the staff cut by the crop: try --recrop).')
    covered = _covered_pages(frames, box, masks, inverted, fading, rows)
    if covered:
        report.warn('rows', f'frames {_ranges(covered, fps, len(frames))} show a page under a dark or coloured '
                    'overlay (left out as a fade) whose barlines match no row: that page is missing from the '
                    'sheet (covered pages are not supported).')
    for row in rows:
        _drop_fading_frames(row)
    # A one-frame row sandwiched between two rows may just be the crossfade between them.
    # Or the page before or after it caught while it slides in or out.
    final, carried = [], []
    for i, row in enumerate(rows):
        before, after = rows[i - 1] if i else None, rows[i + 1] if i + 1 < len(rows) else None
        if len(row.views) == 1 and len(row.staffed) == 1:
            view = row.staffed[0]
            if _is_blend(view, before, after):
                view.dropped = 'crossfade between rows'
            elif _is_slide(view, before) or _is_slide(view, after):
                view.dropped = 'page sliding (same page as the row next to it, moved)'
            elif stacked and _is_roll(view, before, after):
                view.dropped = 'page rolling (rows of the pages next to it, moved)'
            if view.dropped:               # keep it in the record of a neighbouring row's frames
                if final:
                    final[-1].views.extend(row.views)
                else:
                    carried.extend(row.views)
                continue
        row.views[:0] = carried
        carried = []
        final.append(row)
    rows = final

    merged, retinted = [], 0
    for row in rows:
        image, staff, count = _merged_row(row, masks)
        retinted += count
        merged.append((image, staff))
    rows, merged, joined, covers = _merge_same_views(rows, merged, masks)
    if joined:
        report.stage('rows', same_view_rows_merged=joined)
    if covers:
        report.stage('rows', covered_copies_dropped=covers)
    voters = [(n, cv2.cvtColor(m, cv2.COLOR_RGB2GRAY), staff, len(row.kept))
              for n, (row, (m, staff)) in enumerate(zip(rows, merged), 1)]
    zone = scroll.pinned_zone([gray for _, gray, _, kept in voters if kept >= 2],
                              float(np.median([staff.spacing for _, staff in merged])),
                              np.median([staff.lines for _, staff in merged], axis=0))
    scrolls, joins, held = scroll.vote(voters, report, zone)
    report.stage('rows', pinned_header_px=zone if scrolls else f'{zone} (not used: page-switching)')

    pages, voted = None, merged
    if stacked:
        if scrolls:
            report.warn('rows', 'stacked staves in a scrolling video are not supported: laid out page by page '
                        '(each held view split into its rows), not stitched.')
            scrolls = False
        rows, merged, pages = _split_pages(rows, merged, run_dir, report)

    notations = _find_notation(merged, scrolls, report)
    if pages and any(notations):
        report.warn('rows', 'notation + TAB with stacked systems is not supported yet: the page was cut between '
                    'TAB staves, so a notation staff may sit in the wrong row.')

    name = 'view' if scrolls else 'row'
    out_dir = run_dir / ('views' if scrolls else 'rows')
    debug_dir = run_dir / 'debug' / '3-rows'
    for directory in (run_dir / 'rows', run_dir / 'views', debug_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old in [*directory.glob('row-*'), *directory.glob('view-*')]:
            old.unlink()
    for old in (run_dir / 'views.json', run_dir / 'debug' / '3-strips.jpg'):
        old.unlink(missing_ok=True)

    records, dropped = [], {}
    for number, (row, (image, staff), notation) in enumerate(zip(rows, merged, notations), 1):
        label = f'{name}-{number:02d}'
        cv2.imwrite(str(out_dir / f'{label}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        kept = row.kept
        first, last = row.staffed[0], row.staffed[-1]
        records.append({name: number, 'image': f'{out_dir.name}/{label}.png',
                        'frames': [v.index for v in row.views], 'kept_frames': [v.index for v in kept],
                        'seconds': [first.seconds, last.seconds],
                        **({'page': pages[number - 1][0], 'slot': pages[number - 1][1]} if pages else {}),
                        **_staff_record(staff),
                        **(_notation_record(notation) if notation else {})})
        if notation:
            for bar, partner in zip(records[-1]['barlines'], notation.partners):
                bar['notation'] = list(partner) if partner else None
            unpaired = [round(b['x']) for b in records[-1]['barlines'] if b['notation'] is None]
            if unpaired:
                report.warn('rows', f'{label}: TAB barlines at x={unpaired} have no barline in the notation staff '
                            'above them (a false barline, a stem or arrow?). Kept.')
        for v in row.views:
            if v.dropped and v.index not in dropped.get(v.dropped.split(' (')[0], ()):  # stacked: once per page
                dropped.setdefault(v.dropped.split(' (')[0], []).append(v.index)
        if not kept:
            report.warn('rows', f'{label}: every frame looked like a transition; merged them anyway.')
        elif len(kept) == 1 and not scrolls:  # in a scrolling video, coverage decides (below)
            report.warn('rows', f'{label}: only one usable frame ({first.seconds:.1f}s), so a cursor could not be removed.')
        if len(staff.bars) < 2 and not scrolls:
            report.warn('rows', f'{label}: fewer than two barlines found; stage 4 will treat it as one measure.')
        gaps = staff_gaps(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), staff.lines, staff.spacing)
        if gaps:
            report.warn('rows', f'{label}: the staff breaks off and starts again (x=' +
                        ', '.join(f'{a}-{b}' for a, b in gaps) + '): two systems side by side (a coda?), not '
                        'supported; laid out as one row, so measures may be joined across the gap.')
        _debug_row(debug_dir / f'{label}.jpg', label, image, staff, row)

    if any(notations):
        bars = [b for r in records for b in r['barlines']]
        report.stage('rows', notation_partners=f"{sum(b.get('notation') is not None for b in bars)} of {len(bars)} TAB "
                     'barlines have a notation barline within 1 px',
                     notation_strokes_without_tab_barline=[f"{name} {r[name]} x={x}" for r in records
                                                           for x in r.get('notation_loose_strokes', [])] or 'none')
    facts = {'rows': len(rows)} if not scrolls else {'views': len(rows)}
    report.stage('rows', **facts, frames=len(frames),
                 frames_kept=sum(len(r['kept_frames']) for r in records),
                 frames_dropped={reason: len(ix) for reason, ix in dropped.items()} or 'none',
                 measures_per_row=' '.join(str(r['measures']) for r in records))
    for reason, indices in dropped.items():
        report.stage('rows', **{f'dropped: {reason}': indices})
    report.stage('rows', staff_after_last_measure=_ranges(after_end, fps, len(frames)) or 'none',
                 frames_without_staff=_ranges(no_staff, fps, len(frames)) or 'none',
                 frames_fading=_ranges(fading, fps, len(frames)) or 'none',
                 tinted_staff_pixels_set_to_darkest_channel=retinted,
                 mask_boxes=[list(m[:4]) for m in masks] or 'none',
                 mask_overlay_pixels=[f'{int(m[10].sum()) if len(m) > 5 else int(m[4].sum())} overlay, '
                                      f'{int(m[4].sum())} opaque (painted paper) of {m[4].size}'
                                      for m in masks] or 'none')
    scroll.debug_joins(run_dir / 'debug' / '3-joins.jpg',
                       [(n, cv2.cvtColor(m, cv2.COLOR_RGB2GRAY), st, 0) for n, (m, st) in enumerate(voted, 1)], joins)
    if scrolls:
        (run_dir / 'views.json').write_text(json.dumps(records, indent=1) + '\n')
        records = _stitch(run_dir, records, merged, joins, held, report, masks, zone)
    else:
        _warn_near_limit([(f"row {r['row']}", staff) for r, (_, staff) in zip(records, merged)], report)
    (run_dir / 'rows.json').write_text(json.dumps(records, indent=1) + '\n')
    return records


def split_page(image, staves):
    """Cut a merged page of stacked staves into one band per staff, top to bottom.

    Between two staves the cut is the middle of the widest run of pixel rows with no ink in the gap
    (none: of the rows with the least ink). Ink is assigned by connected component (staff-line rows left
    out, so lines do not join everything): a component crossing a cut goes whole to the side holding
    most of it, so stems and beams stay with the staff above, chord names and numbers with the staff
    below. Returns ([(band image, y of its top in the page)], number of components that crossed a cut,
    cuts). A band holds its own ink only (the rest painted paper), trimmed to it plus BAND_MARGIN px."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    level = float(np.median(gray))
    ink = gray < level - BAND_INK
    loose = ink.copy()
    for staff in staves:
        for y in staff.lines:
            loose[max(0, round(y) - 2):round(y) + 3] = False
    cuts = []
    for above, below in zip(staves, staves[1:]):
        lo, hi = round(above.lines[-1]) + 3, round(below.lines[0]) - 2
        counts = ink[lo:hi].sum(axis=1)
        least = np.flatnonzero(counts == counts.min())
        runs = np.split(least, np.flatnonzero(np.diff(least) > 1) + 1)
        best = max(runs, key=len)
        cuts.append(lo + int(best[len(best) // 2]))
    height = gray.shape[0]
    band_of_row = np.searchsorted(np.array(cuts), np.arange(height), side='right')
    owner = np.broadcast_to(band_of_row[:, None], gray.shape).copy()
    count, labels = cv2.connectedComponents(loose.astype(np.uint8), connectivity=8)
    crossed = 0
    if count > 1:
        ys = np.nonzero(labels)[0]
        flat = labels[labels > 0]
        votes = np.zeros((count, len(staves)), np.int64)
        np.add.at(votes, (flat, band_of_row[ys]), 1)
        crossed = int(((votes > 0).sum(axis=1) > 1).sum())
        winner = votes.argmax(axis=1)
        owner[labels > 0] = winner[flat]
    paper = np.median(image[gray >= level - 10], axis=0).astype(np.uint8) if (gray >= level - 10).any() \
        else np.array([255, 255, 255], np.uint8)
    bands = []
    for k in range(len(staves)):
        mine = owner == k
        rows_inked = np.flatnonzero((mine & ink).any(axis=1))
        lines = [round(y) for y in staves[k].lines]
        top = max(0, min(int(rows_inked[0]) if len(rows_inked) else lines[0], lines[0]) - BAND_MARGIN)
        bottom = min(height, max(int(rows_inked[-1]) if len(rows_inked) else lines[-1], lines[-1]) + BAND_MARGIN + 1)
        band = np.where(mine[..., None], image, paper)[top:bottom]
        bands.append((np.ascontiguousarray(band), top))
    return bands, crossed, cuts


def _split_pages(rows, merged, run_dir, report):
    """Stacked staves: every merged page becomes one row per staff, top to bottom (split_page). All
    bands get the same height with the top staff line at the same offset (paper added), so the rows
    look like a single-staff page video's. Returns (rows, merged, [(page, slot)]), one entry per band."""
    cut = []
    for page, (row, (image, staff)) in enumerate(zip(rows, merged), 1):
        staves = find_staves(image) or list(row.staffed[0].staves) or [staff]
        bands, crossed, cuts = split_page(image, staves)
        if crossed:
            report.warn('rows', f'page {page}: {crossed} piece(s) of ink cross the cut between its staves; each was '
                        'kept whole with the staff holding most of it.')
        cut.append((row, image, staves, bands, cuts))
    above = max(round(st.lines[0]) - top for _, _, staves, bands, _ in cut for st, (_, top) in zip(staves, bands))
    below = max(band.shape[0] - (round(st.lines[0]) - top)
                for _, _, staves, bands, _ in cut for st, (band, top) in zip(staves, bands))
    out_rows, out_merged, places = [], [], []
    debug_dir = run_dir / 'debug' / '3-pages'
    debug_dir.mkdir(parents=True, exist_ok=True)
    for old in debug_dir.glob('page-*'):
        old.unlink()
    for page, (row, image, staves, bands, cuts) in enumerate(cut, 1):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        paper = np.median(image[gray >= np.median(gray) - 10], axis=0).astype(np.uint8)
        for slot, (st, (band, top)) in enumerate(zip(staves, bands)):
            offset = round(st.lines[0]) - top
            padded = np.vstack([np.full((above - offset, band.shape[1], 3), paper, np.uint8), band,
                                np.full((below - (band.shape[0] - offset), band.shape[1], 3), paper, np.uint8)])
            staff = find_staff(padded)
            if staff is None:
                report.warn('rows', f'page {page}, row {slot + 1} of {len(staves)}: no staff found in its band; skipped.')
                continue
            out_rows.append(row)
            out_merged.append((padded, staff))
            places.append((page, slot))
        shown = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for y in cuts:
            cv2.line(shown, (0, y), (shown.shape[1], y), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / f'page-{page:02d}.jpg'), shown, [cv2.IMWRITE_JPEG_QUALITY, 80])
    report.stage('rows', stacked_pages=len(cut),
                 rows_per_page=' '.join(str(sum(1 for p, _ in places if p == n)) for n in range(1, len(cut) + 1)),
                 band_height_px=above + below)
    return _drop_rolled(out_rows, out_merged, places, report)


def _same_band(a, b):
    """Two bands show the same row exactly: shift 0 with every barline matched, ink mismatch within
    scroll.CLOSE and no worse than any real shift, and the gray glyphs (measure numbers) matching too."""
    (a_image, a_staff), (b_image, b_staff) = a, b
    a_gray, b_gray = cv2.cvtColor(a_image, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b_image, cv2.COLOR_RGB2GRAY)
    if len(a_staff.bars) != len(b_staff.bars):
        return False
    found, same = scroll.candidates(a_gray, a_staff, b_gray, b_staff)
    best = min((c.mismatch for c in found if c.confident), default=1.0)
    if same is None or not same.bars_ok or same.mismatch > min(scroll.CLOSE, best):
        return False
    a_pale, b_pale = scroll._pale(a_gray, round(a_staff.lines[0])), scroll._pale(b_gray, round(b_staff.lines[0]))
    ring = np.ones((3, 3), np.uint8)
    unmatched = int((a_pale & ~cv2.dilate(b_pale.astype(np.uint8), ring).astype(bool)).sum()
                    + (b_pale & ~cv2.dilate(a_pale.astype(np.uint8), ring).astype(bool)).sum())
    # A glyph present in one band only (another number) differs; an antialiased speck does not.
    return unmatched < scroll.BLOB * a_staff.spacing ** 2 or scroll._mismatch(a_pale, b_pale) <= scroll.CLOSE


def _drop_rolled(rows, merged, places, report):
    """Rolling pages: the lower row moves up and a new row comes in under it, so a page's top row is the
    previous page's bottom row again. Decided once per video: if ROLL_SHARE of the page changes roll (the
    top row equals the previous page's bottom row exactly, _same_band), every such top row is dropped and
    listed. Otherwise nothing is dropped (the structure check still warns about repeated rows)."""
    first = {}
    for k, (page, slot) in enumerate(places):
        first.setdefault(page, k)
    pages = sorted(first)
    rolls, changes = [], 0
    for prev, page in zip(pages, pages[1:]):
        last = max(k for k, (p, _) in enumerate(places) if p == prev)
        top = first[page]
        if top == last:
            continue
        changes += 1
        if places[top][1] == 0 and _same_band(merged[last], merged[top]):
            rolls.append((top, prev, places[last][1]))
    rolling = bool(changes) and len(rolls) >= ROLL_SHARE * changes
    report.stage('rows', rolling_pages=f"{'on' if rolling else 'off'}: {len(rolls)} of {changes} page changes "
                 'show the previous bottom row again at the top')
    if not rolling or not rolls:
        return rows, merged, places
    drop = {top for top, _, _ in rolls}
    report.stage('rows', rolled_rows_dropped=[
        f'page {places[top][0]} row 1 (= page {prev} row {slot + 1}, {rows[top].staffed[0].seconds:.1f}s)'
        for top, prev, slot in rolls])
    keep = [k for k in range(len(places)) if k not in drop]
    return [rows[k] for k in keep], [merged[k] for k in keep], [places[k] for k in keep]


def _find_notation(merged, scrolls, report):
    """Per row, the notation staff printed over its TAB staff (staff.Notation), or None.

    Decided once per video: notation + TAB if found in NOTATION_ROWS of the rows. Otherwise every
    row gets None and nothing is written: a tab-only video takes exactly the old path. A row of
    a notation video without one is warned and laid out as TAB only. Scrolling videos with
    notation are not supported yet (warned; laid out as TAB with the notation as annotations).
    """
    found = [notation_staff(image, staff) for image, staff in merged]
    count = sum(f is not None for f in found)
    if not count or count < NOTATION_ROWS * len(found):
        return [None] * len(found)
    if scrolls:
        report.warn('rows', f'a notation staff sits over the TAB staff in {count} of {len(found)} views, but the '
                    'video scrolls: notation + TAB is supported for page-switching videos only; laid out as TAB '
                    'with the notation kept as it is.')
        return [None] * len(found)
    missing = [n for n, f in enumerate(found, 1) if f is None]
    if missing:
        report.warn('rows', f'rows {missing}: no notation staff found over the TAB staff (the other rows have '
                    'one); laid out as TAB only.')
    spacing = float(np.median([f.spacing for f in found if f]))
    report.stage('rows', notation=f'notation + TAB: notation staff found in {count} of {len(found)} rows',
                 notation_spacing_px=round(spacing, 2),
                 notation_to_tab_spacing=round(spacing / float(np.median([s.spacing for _, s in merged])), 2))
    return found


def _notation_record(notation):
    return {'notation_lines': [round(y, 1) for y in notation.lines], 'notation_spacing': round(notation.spacing, 2),
            'notation_loose_strokes': list(notation.loose)}


def _warn_near_limit(labelled, report):
    """One warning listing accepted barlines that run on almost as far as the barline rule allows.

    The limit is tight (strum arrows poke 5 px past the staff, real barlines up to 2), so these
    are the barlines a slightly different frame could reject; the number check shows if one was.
    """
    close = [f'{label} x={x:.0f} ({px}px)' for label, staff in labelled for x, px in near_limit(staff)]
    if close:
        limit = extension_limit(float(np.median([staff.spacing for _, staff in labelled])))
        report.warn('rows', f'{len(close)} barlines run on past the staff within 1px of the {limit:.1f}px limit '
                    f'(kept; a near miss would merge two measures): {", ".join(close[:12])}'
                    f'{" ..." if len(close) > 12 else ""}')


def _staff_record(staff):
    return {'staff_lines': [round(y, 1) for y in staff.lines], 'spacing': round(staff.spacing, 2),
            'barlines': [{'x': bar.x, 'left': bar.left, 'right': bar.right,
                          'strokes': [list(s) for s in bar.strokes]} for bar in staff.barlines],
            'measures': max(0, len(staff.bars) - 1)}


def _stitch(run_dir, views, merged, joins, held, report, masks=(), zone=0):
    """Stitch a scrolling video's views into strips: the rows stage 4 cuts."""
    by_number = {n: m for n, m in enumerate(merged, 1)}
    strips, covered = scroll.stitch(by_number, joins, held, report, masks, zone)
    frames_of = {v['view']: v['frames'] for v in views}
    records, staffs = [], []
    for k, strip in enumerate(strips, 1):
        label = f'row-{k:02d}'
        cv2.imwrite(str(run_dir / 'rows' / f'{label}.png'), cv2.cvtColor(strip.rgb, cv2.COLOR_RGB2BGR))
        staff = scroll.strip_staff(strip, by_number, find_staff)
        if staff is None:
            report.warn('rows', f'strip {k}: no staff found on the stitched strip; it is skipped.')
            continue
        staffs.append((f'strip {k}', staff))
        numbers = [n for n, _, _ in strip.parts]
        frames = sorted({f for n in numbers + [c[0] for c in strip.covered] for f in frames_of[n]})
        records.append({'row': k, 'image': f'rows/{label}.png', 'frames': frames,
                        'kept_frames': sorted({f for n in numbers for f in views[n - 1]['kept_frames']}),
                        'seconds': [views[numbers[0] - 1]['seconds'][0], views[numbers[-1] - 1]['seconds'][1]],
                        **_staff_record(staff), 'scroll': True, 'view_width': merged[0][0].shape[1],
                        'views': [{'view': n, 'x': x, 'y': y} for n, x, y in strip.parts]})
        if len(staff.bars) < 2:
            report.warn('rows', f'strip {k}: fewer than two barlines found; stage 4 will treat it as one measure.')
    _warn_near_limit(staffs, report)
    report.stage('rows', strips=len(strips),
                 strip_widths_px=[s.rgb.shape[1] for s in strips],
                 measures_per_strip=' '.join(str(r['measures']) for r in records),
                 views_per_strip=[' '.join(str(n) for n, _, _ in s.parts) for s in strips],
                 one_frame_views_dropped=[f'{n}: {why}' for n, why in covered.items()] or 'none')
    scroll.debug_strips(run_dir / 'debug' / '3-strips.jpg', strips, by_number, frames_of)
    return records


def _debug_row(path, name, merged, staff, row, thumb_width=150):
    """Merged row with staff lines (green), barlines (blue), rejected candidates (red);
    below it, every source frame, with dropped ones crossed out in red.

    Each barline's full extent (all strokes) is shaded light blue so the ink stays visible,
    with a thin tick above and below the staff at every stroke center."""
    image = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
    width = image.shape[1]
    for y in staff.lines:
        cv2.line(image, (0, round(y)), (width, round(y)), (0, 180, 0), 1)
    top, bottom, spacing = staff.lines[0], staff.lines[-1], staff.spacing
    shade = image.copy()
    for bar in staff.barlines:
        cv2.rectangle(shade, (bar.left, round(top - spacing)), (bar.right, round(bottom + spacing)), (255, 120, 0), -1)
    image = cv2.addWeighted(shade, .3, image, .7, 0)
    for bar in staff.barlines:
        for _, _, center in bar.strokes:
            x = round(center)
            cv2.line(image, (x, round(top - 2 * spacing)), (x, round(top - spacing)), (255, 80, 0), 1)
            cv2.line(image, (x, round(bottom + spacing)), (x, round(bottom + 2 * spacing)), (255, 80, 0), 1)
    for x, reason in staff.rejected:
        y = round(staff.lines[-1] + 2.5 * staff.spacing)
        cv2.line(image, (x, y - 12), (x, y + 12), (0, 0, 255), 3)
        cv2.putText(image, reason.split()[0], (x + 4, y + 8), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)

    first, last = row.staffed[0], row.staffed[-1]
    header = np.full((40, width, 3), 255, np.uint8)
    text = (f'{name} | frames #{row.views[0].index}-#{row.views[-1].index} ({first.seconds:.1f}-{last.seconds:.1f}s) | '
            f'kept {len(row.kept)}, dropped {len(row.views) - len(row.kept)} | {max(0, len(staff.bars) - 1)} measures')
    cv2.putText(header, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2, cv2.LINE_AA)

    thumbs = []
    for view in row.views:
        scale = thumb_width / view.rgb.shape[1]
        thumb = cv2.resize(cv2.cvtColor(view.rgb, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        thumb = cv2.copyMakeBorder(thumb, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        color = (0, 0, 255) if view.dropped else (0, 120, 0)
        cv2.putText(thumb, f'#{view.index} {"DROPPED" if view.dropped else "kept"}', (3, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
        if view.dropped:
            h, w = thumb.shape[:2]
            cv2.line(thumb, (0, 18), (w, h), (0, 0, 255), 2)
            cv2.line(thumb, (w, 18), (0, h), (0, 0, 255), 2)
        thumbs.append(thumb)
    per_line = max(1, width // thumbs[0].shape[1])
    blank = np.full_like(thumbs[0], 255)
    thumbs += [blank] * (-len(thumbs) % per_line)
    strip = np.vstack([np.hstack(thumbs[i:i + per_line]) for i in range(0, len(thumbs), per_line)])
    strip = cv2.copyMakeBorder(strip, 6, 0, 0, width - strip.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    reasons = sorted({v.dropped for v in row.views if v.dropped})
    notes = np.full((24 * len(reasons) + (6 if reasons else 0), width, 3), 255, np.uint8)
    for i, reason in enumerate(reasons):
        frames = ' '.join(f'#{v.index}' for v in row.views if v.dropped == reason)
        cv2.putText(notes, f'dropped {frames}: {reason}', (10, 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([header, image, strip, notes]), [cv2.IMWRITE_JPEG_QUALITY, 85])
