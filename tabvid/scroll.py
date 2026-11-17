"""Does the video scroll? For each pair of consecutive rows, look for a left shift `dx` at
which the new row's left part repeats the old row's right part.

1. Candidates: compare the two rows' ink column profiles at every shift from one staff spacing
   up to the one leaving 4 spacings of overlap, and keep the best few local minima (the last scroll
   of a piece can be tiny: A Team's moved 35 px, 2% of the width). A pinned
   header (a clef drawn over the left end of every view, see pinned_zone) is left out.
2. A candidate is confident if the overlapping ink matches pixel for pixel (1px slack) and
   every barline in the overlap has a partner within 2px. Staff-line rows are left out of
   the comparison: compression moves the gray rules across the ink threshold. An overlap
   with no barline in it needs a closer match.
3. Repeated music can match at a wrong shift too. Each join takes the confident candidate with
   the most barlines matched, then the one nearest the video's median shift (a steadily
   scrolling video moves by about the same amount each time); a join far from the median gets
   a warning.
   Before that, a candidate whose gray glyphs above the staff (printed measure numbers) fit
   clearly worse than another's is set aside: identical bars match a whole measure off.
4. A join can also be the same view again (shift 0): a held view split in two by a fade or
   by the cursor sitting on a barline. That needs a near-exact match, at least as good as the
   best shift (equal-width measures line up their barlines at shift 0 too). Those joins are
   left out of the vote; in a scrolling video they will be merged, not stitched.
5. The video scrolls if most of the remaining joins are confident. This is decided once for the whole video,
   so a page-switching video is never stitched because one page starts with the measures the
   previous page ended on (songs repeat measures on purpose).

Rows with a single kept frame are usually the blurred frame caught mid-scroll; the vote uses
the rows with two or more frames when most rows have them.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

INK = 150
SHIFT_MIN = 1           # smallest dx searched, in staff spacings (was 10% of the width: A Team's last scroll, 35 px)
MIN_OVERLAP = 4         # largest dx searched: this many spacings of overlap are left (past any pinned header)
MUSIC = .1              # an overlap matched on its barlines must also hold this many square spacings of other
                        # ink (a digit is ~.15): a page's closing barline laid over the next page's opening
                        # barline matches perfectly
PINNED = .8             # ink at the same place in this share of the views is pinned to the screen, not music
PINNED_SEARCH = 6       # a pinned header (a clef drawn over the strip) is looked for this many spacings from the left
PINNED_INK = .5         # ...and holds at least this many square spacings of ink
PINNED_PAD = 1.5        # the music is hidden this many spacings past the header's ink (A: 1.1 measured)
CANDIDATES = 5          # profile minima checked pixel by pixel
MINIMA_APART = 10       # px between two profile minima
MISMATCH = .15          # confident: under this share of the overlap's ink fails to match...
CLOSE = .05             # ...and the overlap holds a barline, or the match is closer than this
BAR_TOL = 2             # px, same tolerance same_row() uses for barlines
EDGE = 6                # px: a barline this close to the overlap's edge may be cut off
EDGE_MARGIN = .5        # spacings at each end of a shifted overlap (each view's outer edge) left out of the
                        # ink comparison: a panel can fade out there (F: the last ~25 px of every view)
MAX_DY = 3              # px of vertical drift allowed between rows
SCROLL_VOTE = .6        # share of confident joins for the video to count as scrolling
FAR = .1                # warn when a join's shift is this share of the width away from the median
PALE = 30               # gray glyphs above the staff (measure numbers): this much darker than the paper...
PALE_CORE = 100         # ...with no pixel darker than this (the same limit stage 4 uses for a dark glyph)
PALE_WORSE = .04        # a shift whose gray glyphs mismatch this much more than another's is set aside


@dataclass
class Candidate:
    dx: int
    mismatch: float
    bars: int           # barlines matched in the overlap
    bars_ok: bool
    pale: float = 0.0   # mismatch of the gray glyphs above the staff (measure numbers) in the overlap
    music: bool = True  # the overlap holds ink besides its matched barlines

    @property
    def confident(self):
        return self.music and self.bars_ok and self.mismatch <= (MISMATCH if self.bars >= 1 else CLOSE)


@dataclass
class Join:
    a: int              # row numbers
    b: int
    width: int
    candidates: list = field(default_factory=list)
    chosen: Candidate = None
    same_view: Candidate = None   # set when B shows the same view as A (shift 0)
    note: str = ''

    @property
    def confident(self):
        return self.chosen is not None


def pinned_zone(views, spacing, lines=()):
    """Width in px of a pinned header at the left of every view, or 0.

    Some players page a long strip sideways and draw a clef over its left end on every page. The
    clef hides the music under it, so in an overlap the new page shows a clef where the old one
    shows notes. Ink that sits at the same place in most views (staff-line rows aside) within
    PINNED_SEARCH spacings of the left edge is such a header; the music is hidden up to PINNED_PAD
    spacings past it. views: gray images of the held views, all the same size; lines: staff line ys.
    """
    if len(views) < 3 or len({v.shape for v in views}) != 1:
        return 0
    static = np.mean([v < INK for v in views], axis=0) >= PINNED
    static[static.mean(axis=1) > .5] = False                     # staff lines
    for y in lines:
        static[max(0, round(y) - 2):round(y) + 3] = False
    if len(lines):                                               # an opening barline is not a header
        band = static[round(lines[0]):round(lines[-1]) + 1]
        static[:, band.mean(axis=0) >= .8 * (1 - 5 * len(lines) / len(band))] = False
    search = static[:, :round(PINNED_SEARCH * spacing)]
    if search.sum() < PINNED_INK * spacing ** 2:
        return 0
    return int(np.flatnonzero(search.any(axis=0))[-1] + 1 + round(PINNED_PAD * spacing))


def _profile(ink):
    return ink.mean(axis=0)


def _barlines_agree(a_bars, b_bars, dx, overlap, zone=0):
    """(matched, ok): barlines of row A moved left by dx against row B's, inside the overlap
    (past row B's pinned header, which hides them)."""
    lo, hi = max(EDGE, zone), overlap - EDGE
    moved = [x - dx for x in a_bars if lo <= x - dx <= hi]
    inside = [x for x in b_bars if lo <= x <= hi]
    matched = sum(any(abs(x - y) <= BAR_TOL for y in inside) for x in moved)
    unmatched_b = sum(not any(abs(x - y) <= BAR_TOL for y in moved) for x in inside)
    return matched, matched == len(moved) and unmatched_b == 0


def _mismatch(a_ink, b_ink):
    ring = np.ones((3, 3), np.uint8)
    near_a = cv2.dilate(a_ink.astype(np.uint8), ring).astype(bool)
    near_b = cv2.dilate(b_ink.astype(np.uint8), ring).astype(bool)
    total = a_ink.sum() + b_ink.sum()
    return float(((a_ink & ~near_b).sum() + (b_ink & ~near_a).sum()) / total) if total else 1.0


def _pale(gray, top):
    """Gray glyphs above the staff: components darker than the paper with no dark pixel. Printed
    measure numbers are gray; identical bars a measure apart differ only in them."""
    ink = gray < float(np.median(gray)) - PALE
    ink[max(0, top - 2):] = False
    count, labels = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    darkest = np.full(count, 255, np.uint8)
    np.minimum.at(darkest, labels.ravel(), gray.ravel())
    keep = darkest >= PALE_CORE
    keep[0] = False
    return keep[labels]


def _has_music(ink, bars, spacing):
    """True if the ink holds MUSIC square spacings besides the barlines at `bars` (x in this image)."""
    rest = ink.copy()
    half = round(spacing / 2)
    for x in bars:
        # A barline just left of the overlap has x < 0: a negative slice end would blank the
        # whole overlap (E: three true joins read as "no music").
        rest[:, max(0, round(x) - half):max(0, round(x) + half + 1)] = False
    return rest.sum() >= MUSIC * spacing ** 2


def _compare(a_ink, b_ink, a_bars, b_bars, dx, a_pale=None, b_pale=None, zone=0, spacing=None):
    overlap = a_ink.shape[1] - dx
    if dx and spacing:
        # The overlap's two ends are the views' outer edges (A's right edge, B's left edge past any
        # pinned zone): leave EDGE_MARGIN spacings there out of the ink comparison.
        m = round(EDGE_MARGIN * spacing)
        a_ink, b_ink = a_ink.copy(), b_ink.copy()
        for ink, start in ((a_ink, dx + zone), (b_ink, zone)):
            ink[:, start:start + m] = False
            ink[:, start + overlap - zone - m:start + overlap - zone] = False
    bars, ok = _barlines_agree(a_bars, b_bars, dx, overlap, zone)
    music = spacing is None or dx == 0 or bars == 0 or (
        _has_music(a_ink[:, dx + zone:], [x - dx - zone for x in a_bars], spacing) and
        _has_music(b_ink[:, zone:overlap], [x - zone for x in b_bars], spacing))
    pale = 0.0
    if a_pale is not None and (a_pale[:, dx + zone:].any() or b_pale[:, zone:overlap].any()):
        pale = round(_mismatch(a_pale[:, dx + zone:], b_pale[:, zone:overlap]), 3)
    return Candidate(dx, round(_mismatch(a_ink[:, dx + zone:], b_ink[:, zone:overlap]), 3), bars, ok, pale, music)


def _numbers_agree(confident):
    """Confident candidates minus those whose gray glyphs (measure numbers) fit clearly worse than
    the best one's: with identical bars in a row, the ink matches a whole measure off, the numbers do not."""
    best = min((c.pale for c in confident), default=0.0)
    return [c for c in confident if c.pale <= best + PALE_WORSE]


def _with_strokes(staff, other):
    """The staff's barlines, plus its full-height strokes that failed the barline rule where the
    other staff has a barline (within BAR_TOL). For the same-view comparison only: ink that is
    there in some frames (a system line above an opening barline) fails a barline in one row
    and not in the next, and one page must not become two rows over it."""
    strokes = [x for x, reason in getattr(staff, 'rejected', ()) if reason != 'gap inside staff']
    return sorted([*staff.bars, *(x for x in strokes if any(abs(x - y) <= BAR_TOL for y in other.bars))])


def _first_strokes(staff):
    """Barline positions for a shifted comparison: the centre of each barline's first stroke.
    A double barline cut by the view edge shows only part of its second stroke, which moves the
    centre of the whole barline (E views 8->9: 3 px, over BAR_TOL); the first stroke is whole."""
    barlines = getattr(staff, 'barlines', None)
    return [bar.strokes[0][2] for bar in barlines] if barlines else list(staff.bars)


def candidates(a_gray, a_staff, b_gray, b_staff, zone=0):
    """(shift candidates best first, the shift-0 comparison) for row B continuing row A.
    `zone`: px at the left of row B hidden by a pinned header, left out of the comparison."""
    dy = round(b_staff.lines[0] - a_staff.lines[0])
    if abs(dy) > MAX_DY or a_gray.shape[1] != b_gray.shape[1]:
        return [], None
    height = min(a_gray.shape[0], b_gray.shape[0]) - abs(dy)
    a_ink = (a_gray < INK)[max(0, -dy):][:height]
    b_ink = (b_gray < INK)[max(0, dy):][:height]
    rules = [round(y) - max(0, -dy) for y in a_staff.lines]
    a_pale = _pale(a_gray[max(0, -dy):][:height], rules[0])
    b_pale = _pale(b_gray[max(0, dy):][:height], rules[0])
    for y in rules:
        a_ink[max(0, y - 1):y + 2] = False
        b_ink[max(0, y - 1):y + 2] = False
    width = a_ink.shape[1]
    pa, pb = _profile(a_ink), _profile(b_ink)
    last = width - zone - round(MIN_OVERLAP * a_staff.spacing)
    first = max(1, round(SHIFT_MIN * a_staff.spacing))
    shifts = np.arange(first, max(first, last) + 1)
    error = np.array([np.abs(pa[dx + zone:] - pb[zone:width - dx]).mean() for dx in shifts])
    minima = [i for i in np.argsort(error)
              if error[i] == error[max(0, i - MINIMA_APART):i + MINIMA_APART + 1].min()]
    a_bars, b_bars = _first_strokes(a_staff), _first_strokes(b_staff)
    found = [_compare(a_ink, b_ink, a_bars, b_bars, int(shifts[i]), a_pale, b_pale, zone,
                      a_staff.spacing) for i in minima[:CANDIDATES]]
    same = _compare(a_ink, b_ink, _with_strokes(a_staff, b_staff), _with_strokes(b_staff, a_staff), 0)
    return sorted(found, key=lambda c: (not c.confident, c.mismatch)), same


def _ends_strip(joins, join):
    """True if no overlapping join follows this one before the strip ends (a join with no overlap,
    or the end of the video); joins showing the same view again are passed over."""
    for later in joins[joins.index(join) + 1:]:
        if later.confident:
            return False
        if later.same_view is None:
            break
    return True


def vote(rows, report, zone=0):
    """rows: [(number, gray image, staff.Staff, kept frame count)] in order; zone: pinned_zone().

    Returns (scrolls, joins, held): held is the row numbers the joins connect.
    """
    stable = [r for r in rows if r[3] >= 2]
    used = stable if len(stable) >= max(2, len(rows) / 2) else rows
    skipped = [r[0] for r in rows if r not in used]
    joins = []
    for (na, ga, sa, _), (nb, gb, sb, _) in zip(used, used[1:]):
        found, same = candidates(ga, sa, gb, sb, zone)
        join = Join(na, nb, ga.shape[1], found)
        best = min((c.mismatch for c in found if c.confident), default=1.0)
        if same is not None and same.bars_ok and same.mismatch <= min(CLOSE, best):
            join.same_view = same
        else:
            confident = _numbers_agree([c for c in join.candidates if c.confident])
            join.chosen = confident[0] if confident else None
        joins.append(join)

    # Repeated music can match at more than one shift. Prefer the one with the most barlines
    # matched (more evidence: the last scroll of a piece is often shorter than the others, and a
    # static overlay such as a watermark adds ink mismatch at the true shift), then the one
    # nearest the median shift.
    confident = [j for j in joins if j.confident]
    counted = [j for j in joins if j.same_view is None]
    median = float(np.median([j.chosen.dx for j in confident])) if confident else None
    for join in confident:
        join.chosen = min(_numbers_agree([c for c in join.candidates if c.confident]),
                          key=lambda c: (-c.bars, abs(c.dx - median)))
        if abs(join.chosen.dx - median) > FAR * join.width and not (
                join.chosen.dx < median and _ends_strip(joins, join)):   # the last scroll is often short
            join.note = f'shift {join.chosen.dx}px is far from the median {median:.0f}px'

    scrolls = bool(counted) and len(confident) / len(counted) >= SCROLL_VOTE
    report.stage('rows', scroll_vote=f'{len(confident)} of {len(counted)} joins overlap '
                 f'({len(joins) - len(counted)} more show the same view again, not counted)',
                 scrolls='yes' if scrolls else 'no (page-switching: rows are kept separate)',
                 median_shift_px=round(median) if median is not None else 'none',
                 vote_skipped_one_frame_rows=skipped or 'none')
    if scrolls:
        report.stage('rows', joins=[_describe(j) for j in joins])
        for join in counted:
            if not join.confident:
                report.warn('rows', f'row {join.a} -> {join.b}: no confident overlap; they will be kept '
                            f'as separate strips. Check the join.')
            elif join.note:
                report.warn('rows', f'row {join.a} -> {join.b}: {join.note}; check the join.')
    elif confident:
        report.stage('rows', overlaps_ignored=[_describe(j) for j in confident])
    return scrolls, joins, [r[0] for r in used]


def _describe(join):
    if join.same_view is not None:
        c = join.same_view
        return f'{join.a}->{join.b}: same view (shift 0, {c.bars} barlines agree, {c.mismatch:.1%} ink mismatch)'
    if not join.confident:
        best = join.candidates[0] if join.candidates else None
        why = (f'best shift {best.dx}px: {best.mismatch:.0%} ink mismatch, '
               f'{best.bars} barlines matched{"" if best.bars_ok else ", others unmatched"}') if best else 'staffs not level'
        return f'{join.a}->{join.b}: none ({why})'
    c = join.chosen
    return (f'{join.a}->{join.b}: shift {c.dx}px, {c.bars} barlines agree, {c.mismatch:.1%} ink mismatch'
            + (f' ({join.note})' if join.note else ''))


def debug_joins(path, rows, joins, scale=.5):
    """For each join: row A on top, row B below placed at the chosen shift, overlap shaded."""
    images = {r[0]: r[1] for r in rows}
    panels = []
    for join in joins:
        a, b = images[join.a], images[join.b]
        dx = join.chosen.dx if join.confident else 0 if join.same_view else \
            (join.candidates[0].dx if join.candidates else 0)
        width = dx + b.shape[1]
        canvas = np.full((a.shape[0] + b.shape[0] + 30, width, 3), 255, np.uint8)
        canvas[30:30 + a.shape[0], :a.shape[1]] = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        canvas[30 + a.shape[0]:, dx:dx + b.shape[1]] = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        shade = canvas.copy()
        good = join.confident or join.same_view is not None
        cv2.rectangle(shade, (dx, 30), (a.shape[1], canvas.shape[0]), (0, 200, 0) if good else (0, 0, 255), -1)
        canvas = cv2.addWeighted(shade, .15, canvas, .85, 0)
        cv2.putText(canvas, _describe(join), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (0, 120, 0) if good else (0, 0, 200), 2, cv2.LINE_AA)
        panels.append(cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    if not panels:
        return
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 6, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=(180, 180, 180))
              for p in panels]
    cv2.imwrite(str(path), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])


# --- Stitching -------------------------------------------------------------------------------
COVERED = .05     # a one-frame view is covered if under this share of its ink is missing from the strip
CHROMA = 28       # colour spread of a cursor line or other overlay (ignored when testing coverage)
COVER_CANDIDATES = 8
BLOB = .1         # ...and no missing piece of ink is this many square spacings (a digit is ~.15)
TALL = .9         # an ink component this share of the crop height tall is a cursor line, not music
PAPER = 240       # a view whose median gray is darker than this is a fade; rescaled before testing coverage


@dataclass
class Strip:
    parts: list = field(default_factory=list)   # [(view number, x, y)] in placing order
    joins: list = field(default_factory=list)   # Join objects that placed a view in this strip
    rgb: np.ndarray = None
    lines: list = field(default_factory=list)    # staff line y positions in the strip
    at: dict = field(default_factory=dict)       # view number -> (x, y)
    covered: list = field(default_factory=list)  # [(view number, x, y, missing share)] one-frame views checked against it


def _place(strip, views, masks=(), zone=0):
    """Composite the strip's views without blending: each pixel comes from the view it is most
    interior to (the seam sits in the middle of each overlap). Mask boxes (crop coordinates) were
    painted over in every frame: there a view only counts if no other view shows that spot, so
    music scrolling under a masked logo comes from a view where it was not masked.

    Overlapping views can be registered 1 px apart (the true shift is fractional); a lightest-
    pixel blend then erased every 1 px stem in the overlap. Held views are already merged over
    their own frames (cursor removed), so blending was not needed to clean them.
    """
    ys = [y for _, _, y in strip.parts]
    xs = [x for _, x, _ in strip.parts]
    height = max(y + views[n][0].shape[0] for n, _, y in strip.parts) - min(ys)
    width = max(x + views[n][0].shape[1] for n, x, _ in strip.parts) - min(xs)
    strip.parts = [(n, x - min(xs), y - min(ys)) for n, x, y in strip.parts]
    canvas = np.zeros((height, width, 3), np.uint8)
    filled = np.zeros((height, width), bool)
    margin = np.full((height, width), -9, np.int32)
    for n, x, y in strip.parts:
        rgb = views[n][0]
        h, w = rgb.shape[:2]
        columns = np.arange(w)
        mine = np.broadcast_to(np.minimum(columns, w - 1 - columns), (h, w)).copy()
        for l, t, r, b, pixels, *_ in masks:
            mine[t:b, l:r][pixels] = -2          # below any unmasked view, above nothing (-9)
        mine[:, :zone] = -2                      # a pinned header hides the music: another view shows it
        region, best = canvas[y:y + h, x:x + w], margin[y:y + h, x:x + w]
        take = mine > best
        region[take] = rgb[take]
        best[take] = mine[take]
        filled[y:y + h, x:x + w] = True
    paper = np.median(canvas[filled], axis=0).astype(np.uint8)
    canvas[~filled] = paper
    strip.rgb = canvas


def _ink_without_rules(gray, lines, ignore=None):
    ink = gray < INK
    if ignore is not None:
        ink &= ~ignore
    for y in lines:
        ink[max(0, round(y) - 1):round(y) + 2] = False
    return ink


def _full_height(ink):
    """Ink components spanning TALL of the image height: a playhead line (no glyph is that tall)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    tall = stats[:, cv2.CC_STAT_HEIGHT] >= TALL * ink.shape[0]
    tall[0] = False
    return tall[labels]


def _coverage(view, strip, strip_lines, x_range):
    """(missing share, x, y): best place for a one-frame view inside the strip, and how much of its
    ink (minus overlay colour and staff lines) the strip lacks there."""
    rgb, staff = view
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # A fade to or from black darkens paper and ink alike: bring the paper back to paper level.
    paper = float(np.median(gray))
    gray = np.clip(gray * (PAPER / max(paper, 1)), 0, 255) if paper < PAPER else gray
    ink = _ink_without_rules(gray, staff.lines, np.ptp(rgb, axis=2) >= CHROMA)
    ink &= ~_full_height(gray < INK)
    y = round(strip_lines[0] - staff.lines[0])
    h, w = ink.shape
    if not ink.any():
        return 0.0, x_range[0], y, 0
    if y < 0 or y + h > strip.rgb.shape[0]:
        return 1.0, x_range[0], y, h * w
    strip_gray = cv2.cvtColor(strip.rgb, cv2.COLOR_RGB2GRAY)[y:y + h]
    strip_ink = _ink_without_rules(strip_gray, [l - y for l in strip_lines])
    near = cv2.dilate(strip_ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    lo, hi = max(0, x_range[0]), min(strip.rgb.shape[1] - w, x_range[1])
    if hi < lo:
        return 1.0, lo, y, h * w
    ps, pv = strip_ink.mean(axis=0), ink.mean(axis=0)
    error = np.array([np.abs(ps[x:x + w] - pv).mean() for x in range(lo, hi + 1)])
    order = [i for i in np.argsort(error) if error[i] == error[max(0, i - MINIMA_APART):i + MINIMA_APART + 1].min()]
    best = (1.0, lo, y, w * h)
    for i in order[:COVER_CANDIDATES]:
        x = lo + int(i)
        lost = ink & ~near[:, x:x + w]
        # Rejoin pieces split by a removed staff-line row, so a digit on a line counts whole.
        joined = cv2.dilate(lost.astype(np.uint8), np.ones((5, 1), np.uint8))
        _, labels = cv2.connectedComponents(joined, connectivity=8)
        sizes = np.bincount(labels[lost]) if lost.any() else np.zeros(1, int)
        biggest = int(sizes[1:].max()) if len(sizes) > 1 else 0
        best = min(best, (round(float(lost.sum() / ink.sum()), 3), x, y, biggest))
    return best


def stitch(views, joins, held, report, masks=(), zone=0):
    """Chain held views into strips; check every other view is covered by them.

    views: {number: (rgb, staff)} for every view; held: the view numbers the vote joined, in
    order; joins: the vote's joins between consecutive held views.
    Returns (strips in order, {view number: why it was dropped}).
    """
    strips = [Strip(parts=[(held[0], 0, 0)])]
    position = {held[0]: (0, 0)}
    for join in joins:
        a, b = join.a, join.b
        xa, ya = position[a]
        dy = round(views[b][1].lines[0] - views[a][1].lines[0])
        if join.same_view is not None or join.confident:
            dx = 0 if join.same_view is not None else join.chosen.dx
            position[b] = (xa + dx, ya - dy)
            strips[-1].parts.append((b, xa + dx, ya - dy))
            strips[-1].joins.append(join)
        else:
            strips.append(Strip(parts=[(b, 0, 0)]))
            position[b] = (0, 0)
    for strip in strips:
        _place(strip, views, masks, zone)
        n, x, y = strip.parts[0]
        strip.lines = [l + y for l in views[n][1].lines]
        strip.at = {n: (x, y) for n, x, y in strip.parts}

    # Every view that was not joined (mostly the frame caught mid-scroll) must be covered by
    # the strip of the held views around it; if not, it is kept as its own strip.
    dropped, extra = {}, []
    owner = {n: s for s in strips for n in s.at}
    joined = {(j.a, j.b): j for j in joins if j.same_view is not None or j.confident}
    for number in sorted(views):
        if number in owner:
            continue
        before = max((h for h in held if h < number), default=None)
        after = min((h for h in held if h > number), default=None)
        # Between two held views the strip shows continuously: the frame was caught mid-scroll
        # (or is the same view again). Only views at a strip end or across a failed join are tested.
        if (before, after) in joined:
            dropped[number] = f'between joined views {before} and {after}'
            continue
        neighbours = [h for h in (before, after) if h is not None]
        strip = owner[neighbours[0]] if neighbours else None
        width = views[number][0].shape[1]
        missing, biggest = 1.0, None
        if strip is not None:
            spots = [strip.at[h][0] for h in neighbours if owner[h] is strip]
            found = _coverage(views[number], strip, strip.lines, (min(spots) - width, max(spots) + width))
            if after is not None and owner[after] is not strip:
                other = owner[after]
                other_found = _coverage(views[number], other, other.lines,
                                        (other.at[after][0] - width, other.at[after][0] + width))
                if other_found[0] < found[0]:
                    strip, found = other, other_found
            missing, x, y, biggest = found
            strip.covered.append((number, x, y, missing))
            spacing = views[number][1].spacing
            if missing <= COVERED and biggest < BLOB * spacing ** 2:
                dropped[number] = (f'covered by strip {strips.index(strip) + 1} at x={x} '
                                   f'({missing:.1%} of its ink missing, largest piece {biggest}px)')
                continue
        extra.append((number, strip))
        why = 'no strip around it' if strip is None else \
            f'{missing:.1%} of its ink missing there, largest piece {biggest}px'
        report.warn('rows', f'view {number}: not covered by the views around it ({why}); kept as its own strip.')

    for number, _ in extra:
        lone = Strip(parts=[(number, 0, 0)])
        _place(lone, views, masks)       # on its own: its pinned header is all there is to show
        lone.lines = list(views[number][1].lines)
        lone.at = {number: (0, 0)}
        strips.append(lone)
    strips.sort(key=lambda s: s.parts[0][0])   # in time order of their first view
    return strips, dropped


def debug_strips(path, strips, views, frames_of, band=4000, scale=.4):
    """The stitched strips wrapped into bands: coloured bars for each view's span above the
    music, red lines at each join point, grey ticks where one-frame views were checked."""
    colours = [(200, 120, 0), (0, 150, 0), (160, 0, 160), (0, 140, 200)]
    out = []
    for k, strip in enumerate(strips, 1):
        title = np.full((40, round(band * scale), 3), 255, np.uint8)
        text = (f'strip {k}: {strip.rgb.shape[1]}px wide, views ' +
                ' '.join(str(n) for n, _, _ in strip.parts))
        cv2.putText(title, text, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2, cv2.LINE_AA)
        out.append(title)
        gray = cv2.cvtColor(cv2.cvtColor(strip.rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
        joins_x = sorted({x for n, x, _ in strip.parts[1:]})
        for x0 in range(0, strip.rgb.shape[1], band):
            x1 = min(x0 + band, strip.rgb.shape[1])
            lanes = np.full((3 * 26 + 8, band, 3), 255, np.uint8)
            for i, (n, x, _) in enumerate(strip.parts):
                w = views[n][0].shape[1]
                if x + w <= x0 or x >= x1:
                    continue
                lane = i % 2
                colour = colours[i % len(colours)]
                a, b = max(x, x0) - x0, min(x + w, x1) - x0
                cv2.rectangle(lanes, (a, 4 + lane * 26), (b - 1, 4 + lane * 26 + 20), colour, -1)
                frames = frames_of[n]
                label = f'view {n} #{frames[0]}-{frames[-1]}'
                cv2.putText(lanes, label, (a + 6, 4 + lane * 26 + 16), cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255, 255, 255), 2, cv2.LINE_AA)
            for n, x, _, missing in strip.covered:
                if x0 <= x < x1:
                    colour = (150, 150, 150) if missing <= COVERED else (0, 0, 255)
                    a = x - x0
                    cv2.rectangle(lanes, (a, 4 + 2 * 26), (min(band, a + views[n][0].shape[1]) - 1, 4 + 2 * 26 + 20), colour, 1)
                    cv2.putText(lanes, f'v{n} {missing:.0%}', (a + 4, 4 + 2 * 26 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                                .5, colour, 1, cv2.LINE_AA)
            body = np.full((gray.shape[0], band, 3), 255, np.uint8)
            body[:, :x1 - x0] = gray[:, x0:x1]
            for x in joins_x:
                if x0 <= x < x1:
                    cv2.line(body, (x - x0, 0), (x - x0, body.shape[0]), (0, 0, 255), 3)
                    cv2.line(lanes, (x - x0, 0), (x - x0, lanes.shape[0]), (0, 0, 255), 3)
            scale_bar = np.full((22, band, 3), 255, np.uint8)
            cv2.putText(scale_bar, f'x={x0}', (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .6, (90, 90, 90), 1, cv2.LINE_AA)
            piece = np.vstack([scale_bar, lanes, body, np.full((10, band, 3), 200, np.uint8)])
            out.append(cv2.resize(piece, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    width = max(p.shape[1] for p in out)
    out = [cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in out]
    cv2.imwrite(str(path), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 88])


def strip_staff(strip, views, find_staff):
    """Staff of a stitched strip: barlines found in view-width windows at each view's position
    (a whole strip is too long for the staff detector), mapped to strip coordinates and merged.
    A barline near a window's edge may be cut off, so each one is taken from the window where it
    sits furthest from an edge."""
    from .staff import Staff, Barline
    found, spacing = [], []
    for n, x, _ in strip.parts:
        width = views[n][0].shape[1]
        window = np.ascontiguousarray(strip.rgb[:, x:x + width])
        staff = find_staff(window)
        if staff is None:
            continue
        spacing.append(staff.spacing)
        for bar in staff.barlines:
            margin = min(bar.left, width - 1 - bar.right)
            moved = Barline(round(bar.x + x, 1), bar.left + x, bar.right + x,
                            tuple((l + x, r + x, round(c + x, 1)) for l, r, c in bar.strokes), bar.extension)
            found.append((moved, margin))
    if not spacing:
        return None
    space = float(np.median(spacing))
    merged = []
    for bar, margin in sorted(found, key=lambda f: f[0].x):
        if merged and bar.left <= merged[-1][0].right + space / 2:
            if margin > merged[-1][1]:
                merged[-1] = (bar, margin)
        else:
            merged.append((bar, margin))
    return Staff(tuple(strip.lines), round(space, 2), tuple(b for b, _ in merged), ())
