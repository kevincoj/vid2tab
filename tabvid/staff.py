"""Find the TAB staff and its real barlines in a cropped image.

A vertical stroke counts as a barline only if it is dark from the top staff line
to the bottom staff line and does not continue past either by more than
max(EXTEND_MIN px, EXTEND spacings): ink as dark as itself, unbroken from the staff edge.
That rejects note stems (which hang below the staff) and strum arrows (whose heads poke
5-9 px past it; real barlines measured 0-2 px over nine videos). Ink separated from the staff by a gap
(a chord diagram's string line, a tie arc) and gray measure numbers do not count.

A barline can have several strokes: a double barline, or the thin+thick final
barline. They are kept together as one Barline so stage 4 can keep the whole
double barline with the measure it ends.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .vendor.layout.models import Staff as SourceStaff
from .vendor.layout.source import DEFAULT, analyze_source

INK = 150            # gray below this is ink
SPAN_COVERAGE = .9   # a stroke is dark over at least 90% of top line -> bottom line
FAINT = 60           # a faint stroke (a thin barline softened by compression): its median gray is at least
                     # this much darker than the paper, yet within FAINT_INK of the ink threshold...
FAINT_INK = 25
FAINT_EVEN = 25      # ...and 90% of it is within this many gray levels of that median: one even gray from
                     # the top line to the bottom line (a dark stem broken by digits is not)
WANDER = .4          # px: a stroke whose centre, row by row, strays further than this from its median in over
WANDER_ROWS = .25    # ...a quarter of its rows is not straight: a wavy arpeggio arrow (measured: barlines of
                     # thirteen videos <= 0.09 px at that quartile, arrows 0.8 px)
WANDER_PAD = 2       # px each side of the stroke that count as the stroke (an arrow's wave is ~1.5 px)
EXTEND = .1          # ...and does not continue unbroken more than this many spacings past the staff,
EXTEND_MIN = 2       # ...or this many px if more (antialiasing at the staff edge)
LINE_REACH = 2       # px from a staff line's centre that can still be the line itself (thick lines)
NEAR_LIMIT = 1       # an accepted barline extending within this many px of the limit is reported
BREAK = 1            # rows of antialiasing gap still counted as unbroken
SAME_STROKE = 50     # outside the staff, only ink within this many gray levels of the stroke continues it
SNAP = .4            # look this many spacings left/right of a detector candidate for the stroke
GROUP_GAP = .6       # strokes closer than this many spacings form one double/final barline
ABOVE = 1.5          # barline candidates are searched from this many spacings above the top line down
MARGIN = 40          # px of paper added above and below before the staff finder runs (it needs a margin)
THICK_ROWS = 12      # with no staff found, staff lines up to this many px thick are tried...
THICK_SHARE = .25    # ...and kept if no line is thicker than this share of the line spacing
FINAL_STROKE = .22   # a barline whose last stroke is at least this many spacings wide (and 4 px) is a final
                     # barline (measured: thin strokes <= 0.14, the thick stroke of thin+thick 0.33-0.5)
PALE_INK = 225       # with no staff found, the finder's ink threshold (of 255, after its background
                     # normalisation; default 210) is raised to this: a pale line hovering at 210
SPLIT_LINE = .2      # with no staff found, a line run this share of a spacing from a fitted line is a piece of it
SHORT_COVERAGE = .05 # with no staff found, staff lines this short (share of the width) are tried, kept only
                     # if the staff ends on a final barline (K: a last page holding one 230 px measure)


@dataclass(frozen=True)
class Barline:
    x: float          # center of the whole barline (all strokes)
    left: int         # first ink column
    right: int        # last ink column
    strokes: tuple    # ((left, right, center), ...) one per stroke, left to right
    extension: int = 0  # px the strokes run on past the staff (max over strokes, above/below)
    system: bool = False  # drawn on up through a notation staff above (see _system_stroke)


@dataclass
class Staff:
    lines: tuple      # six y positions
    spacing: float
    barlines: tuple   # Barline, left to right
    rejected: tuple   # (x, reason) for candidates that failed the barline test

    @property
    def bars(self):
        """Barline centers. same_row() compares these between frames."""
        return tuple(bar.x for bar in self.barlines)


def _faint_level(band, paper):
    """Per column: the gray level under which a faint stroke's pixels count as ink, or 0 where the
    column is not a faint stroke's gray (median between INK - FAINT_INK and paper - FAINT)."""
    median = np.median(band, axis=0)
    return np.where((median >= INK - FAINT_INK) & (median <= paper - FAINT), (paper + median) / 2, 0)


def _faint_columns(band, paper):
    """Columns evenly gray from the top line to the bottom line, though lighter than INK."""
    even = (np.abs(band - np.median(band, axis=0)) <= FAINT_EVEN).mean(axis=0) >= SPAN_COVERAGE
    return even & (_faint_level(band, paper) > 0)


def _strokes_near(gray, x, lines, reach, faint=False):
    """Full-height dark column runs within `reach` of x, as (left, right, darkness-weighted center).

    With `faint`, and only if there is no such run, runs of evenly dark but lighter columns.
    """
    band = gray[max(0, round(lines[0])):round(lines[-1]) + 1].astype(np.float32)
    lo, hi = max(0, x - reach), min(gray.shape[1], x + reach + 1)
    full = [c for c in range(lo, hi) if (band[:, c] < INK).mean() >= SPAN_COVERAGE]
    if faint and not full:
        full = list(lo + np.flatnonzero(_faint_columns(band[:, lo:hi], float(np.median(gray)))))
    strokes = []
    for run in np.split(np.array(full), np.where(np.diff(full) > 1)[0] + 1) if full else []:
        weight = np.maximum(0, 255 - np.median(band[:, run], axis=0))
        center = float((run * weight).sum() / weight.sum()) if weight.sum() else float(run.mean())
        strokes.append((int(run[0]), int(run[-1]), round(center, 1)))
    return strokes


def _extension(same, left, right, start, step):
    """Rows of ink in columns [left, right] continuing unbroken from row `start` in direction `step`."""
    length, last, y = 0, start - step, start
    while 0 <= y < same.shape[0] and abs(y - last) <= BREAK + 1:
        if same[y, left:right + 1].any():
            last = y
            length = abs(y - start) + 1
        y += step
    return length


def extension_limit(spacing):
    """Px a barline may run on past the staff."""
    return max(EXTEND_MIN, EXTEND * spacing)


def _line_edge(gray, y, step):
    """The first row past the staff line at y, going in direction `step`.

    A line can be 2-3 px thick or blurred, and its estimated centre moves by a pixel between
    crops. Rows up to LINE_REACH px out that are darker, along the whole width, than halfway
    between paper and the line's centre row belong to the line, so the extension is measured
    from the line's outer edge and does not depend on that pixel.
    """
    paper = float(np.median(gray))
    halfway = (paper + float(np.median(gray[y]))) / 2
    edge = y + step
    for _ in range(LINE_REACH):
        if not 0 <= edge < gray.shape[0] or np.median(gray[edge]) >= halfway:
            break
        edge += step
    return edge


def _outside(gray, stroke, lines):
    """(px above, px below) the stroke runs on past the staff lines, unbroken and as dark as itself."""
    left, right, _ = stroke
    top, bottom = round(lines[0]), round(lines[-1])
    core = gray[max(0, top):bottom + 1, left:right + 1]
    level = min(INK, float(np.median(core)) + SAME_STROKE)
    if (core < INK).mean() < SPAN_COVERAGE:      # a faint stroke: ink as dark as itself is lighter than INK
        level = max(level, float(_faint_level(core.astype(np.float32), float(np.median(gray))).max()))
    same = gray < level
    return (_extension(same, left, right, _line_edge(gray, top, -1), -1),
            _extension(same, left, right, _line_edge(gray, bottom, 1), 1))


def _continues_outside(gray, stroke, lines, spacing):
    """Reason if the stroke runs on above or below the staff past the limit, else None."""
    above, below = _outside(gray, stroke, lines)
    limit = extension_limit(spacing)
    if above > limit:
        return 'extends above staff'
    if below > limit:
        return 'extends below staff'
    return None


def _stroke_reason(gray, stroke, lines, spacing):
    """(reason the stroke is not a barline or None, drawn through a notation staff above).

    The barline rule, except that a stroke running on above the staff is still a barline when it
    is a system barline: drawn unbroken up through a notation staff and stopping at its top line."""
    reason = _continues_outside(gray, stroke, lines, spacing)
    if reason == 'extends above staff' and _system_stroke(gray, stroke, lines, spacing):
        below = _outside(gray, stroke, lines)[1]
        return ('extends below staff' if below > extension_limit(spacing) else None), True
    return reason, False


def _staff_ends(gray, lines, x0, x1, level):
    """True if the TAB staff lines do not run on over columns x0:x1 (or those lie off the image)."""
    x0, x1 = max(0, x0), min(gray.shape[1], x1)
    if x1 - x0 < 2:
        return True
    rows = [round(y) for y in lines]
    dark = np.mean([(gray[max(0, y - 1):y + 2, x0:x1].min(axis=0) < level).mean() for y in rows])
    return dark < .5


def _five_lines(band, y0, spacing, level):
    """Centres of the rows dark over 80% of `band` (lines crossing it), if they are five evenly spaced
    lines, else None."""
    on = (band < level).mean(axis=1) >= .8
    rows = np.flatnonzero(on)
    if not len(rows):
        return None
    centres = [y0 + float(run.mean()) for run in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)]
    if len(centres) != 5:
        return None
    gaps = np.diff(centres)
    step = float(np.median(gaps))
    if not .4 * spacing <= step <= 1.5 * spacing or np.abs(gaps - step).max() > SYSTEM_EVEN * spacing + 1:
        return None
    return centres


def _system_stroke(gray, stroke, lines, spacing):
    """True if the stroke, running on above the staff, crosses exactly five evenly spaced lines
    (a notation staff: they run on at least SYSTEM_SIDE spacings on both sides of it, so not a chord
    diagram's frets) more than a spacing above the staff, and stops at the top one. A system's closing
    (or opening) barline has them on its inner side only: it counts when the TAB staff ends at it too."""
    left, right, _ = stroke
    above, _ = _outside(gray, stroke, lines)
    edge = _line_edge(gray, round(lines[0]), -1)
    stroke_top = edge - above + 1
    reach = round(SYSTEM_SIDE * spacing)
    y0, y1 = max(0, stroke_top - 3), edge - round(spacing)
    if y1 - y0 < 4:
        return False
    paper = float(np.median(gray))
    level = (paper + INK) / 2
    sides = {'left': (left - 2 - reach, left - 2), 'right': (right + 3, right + 3 + reach)}
    inside = {side: a >= 0 and b <= gray.shape[1] for side, (a, b) in sides.items()}
    centres = None
    if all(inside.values()):        # rows dark on both sides at once (ink on one side alone is no line)
        both = np.all([(gray[y0:y1, a:b] < level).mean(axis=1) >= .8 for a, b in sides.values()], axis=0)
        centres = _five_lines(np.where(both[:, None], 0, 255).astype(np.uint8), y0, spacing, level)
    # The TAB staff ends here too: no lines one to two spacings past the stroke (a final barline's
    # thick stroke follows the thin one within a spacing, with the lines running on between them).
    near, far = round(spacing), 2 * round(spacing)
    if centres is None and inside['left'] and _staff_ends(gray, lines, right + near, right + far, level):
        centres = _five_lines(gray[y0:y1, slice(*sides['left'])], y0, spacing, level)   # closing barline
    if centres is None and inside['right'] and _staff_ends(gray, lines, left - far, left - near, level):
        centres = _five_lines(gray[y0:y1, slice(*sides['right'])], y0, spacing, level)  # opening barline
    if centres is None:
        return False
    return abs(centres[0] - stroke_top) <= extension_limit(spacing) + LINE_REACH


def wander(gray, stroke, lines):
    """How far the stroke's centre strays from its median position, in px, at the WANDER_ROWS
    quantile of its rows (staff-line rows left out). A barline is straight: its centre is the same
    in every row, and a tie or digit touching it moves it in a few rows only."""
    left, right, _ = stroke
    top, bottom = round(lines[0]), round(lines[-1])
    lo, hi = max(0, left - WANDER_PAD), min(gray.shape[1], right + WANDER_PAD + 1)
    weight = np.maximum(0, float(np.median(gray)) - gray[max(0, top):bottom + 1, lo:hi].astype(np.float32))
    off_lines = np.array([all(abs(y + top - line) > LINE_REACH for line in lines) for y in range(len(weight))])
    weight = weight[off_lines & (weight.sum(axis=1) > 0)]
    if len(weight) < 4:
        return 0.0
    centre = (weight * np.arange(hi - lo)).sum(axis=1) / weight.sum(axis=1)
    return float(np.quantile(np.abs(centre - np.median(centre)), 1 - WANDER_ROWS))


def near_limit(staff):
    """Accepted barlines whose extension is within NEAR_LIMIT px of the limit: [(x, px)]."""
    limit = extension_limit(staff.spacing)
    return [(bar.x, bar.extension) for bar in staff.barlines
            if bar.extension and bar.extension >= limit - NEAR_LIMIT]


def find_barline(gray, x, lines, spacing):
    """Return (Barline, None) for a real barline near detector candidate x, else (None, reason).

    The detector may report a double barline anywhere between its strokes, so start from the
    leftmost full-height stroke in reach and collect the strokes that follow it closely.
    """
    reach = max(1, round(SNAP * spacing))
    nearby = _strokes_near(gray, x, lines, reach, faint=True)
    if not nearby:
        return None, 'gap inside staff'
    first = nearby[0]
    reason, system = _stroke_reason(gray, first, lines, spacing)
    if reason:
        return None, reason
    if wander(gray, first, lines) > WANDER:
        return None, 'wavy (not a straight stroke)'
    group = [first]
    gap = max(2, round(GROUP_GAP * spacing))
    while True:
        end = group[-1][1]
        following = [s for s in _strokes_near(gray, end + gap, lines, gap) if end < s[0] <= end + gap]
        if not following or _stroke_reason(gray, following[0], lines, spacing)[0] or \
                wander(gray, following[0], lines) > WANDER:
            break
        group.append(following[0])
    widths = [s[1] - s[0] + 1 for s in group]
    center = round(sum(s[2] * w for s, w in zip(group, widths)) / sum(widths), 1)
    extension = max(_outside(gray, s, lines)[1] if system else max(_outside(gray, s, lines)) for s in group)
    return Barline(center, group[0][0], group[-1][1], tuple(group), extension, system), None


def _systems(rgb):
    """[(barline candidates, six-line staves)] per system, with paper added above and below.

    The vendored finder needs blank margin around a staff; a tight crop (staff lines a line
    spacing or two from the edge) otherwise finds nothing. Staff y positions are returned in
    crop coordinates; barline x positions are unaffected.
    """
    out = _systems_with_margin(rgb, MARGIN)
    if not any(staves for _, staves in out):
        # The vendored finder accepts a line spacing of at most 10% of the image height, so a
        # large staff in a tight crop (V6: 40 px spacing, 318 px crop) is rejected. With half the
        # crop height of paper on each side, any staff that fits in the crop passes.
        margin = (rgb.shape[0] + 1) // 2
        if margin > MARGIN:
            out = _systems_with_margin(rgb, margin)
    if not any(staves for _, staves in out):
        # The vendored finder drops pixel-row runs over 5 px as "not a line" (H: 6-7 px lines at a
        # 40 px spacing). Only where nothing was found, thicker lines are tried, and a staff is
        # kept only if its lines stay under THICK_SHARE of its spacing (a thin rule, not a band).
        for margin in sorted({MARGIN, max(MARGIN, (rgb.shape[0] + 1) // 2)}):
            out = _systems_with_margin(rgb, margin, THICK)
            if any(staves for _, staves in out):
                break
    if not any(staves for _, staves in out):
        # A pale staff line (video 3 of blind test 5: gray 201-203 on 245 paper) sits at the finder's
        # ink threshold, and compression noise decides whether it covers enough of the width. Only
        # where nothing was found, a paler threshold is tried.
        for margin in sorted({MARGIN, max(MARGIN, (rgb.shape[0] + 1) // 2)}):
            out = _systems_with_margin(rgb, margin, PALE)
            if any(staves for _, staves in out):
                break
    if not any(staves for _, staves in out):
        # A staff line broken by a row of noise gives two line runs a pixel or two apart, which the
        # finder reads as an interleaved extra line and rejects the staff (video 5 of blind test 5:
        # 1 px of line just above a line). Only where nothing was found, such pieces are allowed.
        for margin in sorted({MARGIN, max(MARGIN, (rgb.shape[0] + 1) // 2)}):
            out = _systems_with_margin(rgb, margin, SPLIT)
            if any(staves for _, staves in out):
                break
    return out


THICK = replace(DEFAULT, max_line_rows=THICK_ROWS)
SHORT = replace(DEFAULT, horizontal_coverage=SHORT_COVERAGE)
PALE = replace(DEFAULT, ink_threshold=PALE_INK)
SPLIT = replace(DEFAULT, split_line=SPLIT_LINE)


def is_final(strokes, spacing):
    """A final barline: its last stroke is thick (thin+thick, or one thick end bar)."""
    left, right, _ = strokes[-1]
    return right - left + 1 >= max(4, FINAL_STROKE * spacing)


def _line_rows(gray, y, config):
    """Thickness in px of the staff line at row y: the run of rows around it that are ink
    (the finder's threshold) over its horizontal coverage."""
    dark = (gray < config.ink_threshold).mean(axis=1) >= config.horizontal_coverage
    y = int(round(y))
    if not 0 <= y < len(dark) or not dark[y]:
        return 0
    top, bottom = y, y
    while top > 0 and dark[top - 1]:
        top -= 1
    while bottom < len(dark) - 1 and dark[bottom + 1]:
        bottom += 1
    return bottom - top + 1


def _systems_with_margin(rgb, margin, config=DEFAULT, count=6):
    paper = np.full((margin, rgb.shape[1], 3), 255, np.uint8)
    padded = np.ascontiguousarray(np.vstack([paper, rgb, paper]))
    # the line-thickness check belongs to the retries that change what a line is (thick, pale), not to SPLIT
    gray = cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY) if config is not DEFAULT and not config.split_line else None
    out = []
    for system in analyze_source(padded, config):
        staves = [SourceStaff(tuple(y - margin for y in st.lines), st.spacing)
                  for st in system.staves if len(st.lines) == count and (
                      gray is None or max(_line_rows(gray, y, config) for y in st.lines) <= THICK_SHARE * st.spacing)]
        out.append((system.barlines, staves))
    return out


def _full_height_columns(gray, lines):
    """Our own barline candidates: centres of column runs dark over SPAN_COVERAGE of the staff.

    The vendored detector drops some real barlines: one hugged by ties and parentheses looks
    "wavy" to it, and one under a coloured highlight looks like a playhead. The barline rule
    (find_barline) still decides; this only makes sure every full-height stroke is tested.
    """
    band = gray[max(0, round(lines[0])):round(lines[-1]) + 1]
    out = []
    for columns in ((band < INK).mean(axis=0) >= SPAN_COVERAGE,
                    _faint_columns(band.astype(np.float32), float(np.median(gray)))):
        full = np.flatnonzero(columns)
        if len(full):
            out += [int(round(run.mean())) for run in np.split(full, np.flatnonzero(np.diff(full) > 1) + 1)]
    return sorted(set(out))


def _merge_candidates(vendored, own, spacing):
    """Vendored candidates plus our own ones not already within SNAP spacings of one, left to right."""
    reach = SNAP * spacing
    extra = [x for x in own if all(abs(x - v) > reach for v in vendored)]
    return sorted(list(vendored) + extra)


def find_staff(rgb):
    """The lowest six-line staff in an RGB crop, with checked barlines, or None.

    Staff lines must cover a fifth of the width. Only where none is found, a short staff (lines over
    SHORT_COVERAGE of the width) is tried, and kept only if its last barline is a final barline: a
    last page holding a measure or two (K), not a stray grid of lines.
    """
    staff = _find_staff(rgb, _systems)
    if staff is None:
        staff = _find_staff(rgb, lambda image: _systems_with_margin(image, MARGIN, SHORT))
        if staff is None or not staff.barlines or not is_final(staff.barlines[-1].strokes, staff.spacing):
            return None
    return staff


# --- Stacked staves -------------------------------------------------------------------------------
# Two or more TAB staves one above the other in a frame (GuitarZero2Hero: two rows per page). The
# lowest is find_staff's; a higher six-line fit is another TAB staff only if it looks like one.

STACK_SPACING = .15   # its spacing is within this share of the lowest staff's
STACK_COVER = .5      # its six lines are present over this share of the lowest staff's extent (a row of
                      # chord diagrams is short pieces of line)
STACK_CLEAR = 4       # spacings of clear space between its bottom line and the next staff's top line


def find_staves(rgb):
    """Every TAB staff in the crop, top to bottom ([] if none). The last one is find_staff's own; each
    higher one has its barlines found by find_staff on its band (the crop down to halfway to the
    staff below), so barline candidates never cross into another staff."""
    systems = _systems(rgb)                       # computed once: find_staff's own first step
    lowest = _find_staff(rgb, lambda image: systems if image is rgb else _systems(image)) or find_staff(rgb)
    if lowest is None:
        return []
    if sum(len(staves) for _, staves in systems) < 2:
        return [lowest]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    extent = _present(gray, lowest.lines) >= 5
    if not extent.any():
        return [lowest]
    found = [lowest]
    candidates = sorted({st.lines: st for _, staves in systems for st in staves}.values(),
                        key=lambda st: -st.lines[0])
    for st in candidates:
        below = found[0]
        if st.lines[-1] > below.lines[0] - STACK_CLEAR * below.spacing:
            continue
        if abs(st.spacing - lowest.spacing) > STACK_SPACING * lowest.spacing:
            continue
        if float((_present(gray, st.lines) >= 5)[extent].mean()) < STACK_COVER:
            continue
        cut = round((st.lines[-1] + below.lines[0]) / 2)
        upper = find_staff(rgb[:cut])
        if upper is None or abs(upper.lines[0] - st.lines[0]) > 2:
            continue
        found.insert(0, upper)
    return found


def shows_staff(rgb):
    """True if the crop shows six evenly spaced staff lines of any length over SHORT_COVERAGE of the
    width, barlines or not: TAB the staff finder did not take (too short, or no final barline)."""
    return any(staves for _, staves in _systems(rgb)) or any(
        staves for _, staves in _systems_with_margin(rgb, MARGIN, SHORT))


def _find_staff(rgb, systems):
    found = [(bars, staff) for bars, staves in systems(rgb) for staff in staves]
    if not found:
        return None
    candidates, staff = max(found, key=lambda pair: pair[1].lines[0])
    # Look for barlines in a strip starting just above the TAB staff. Otherwise a row of
    # aligned chord diagrams can pass as a second staff, and the detector then only accepts
    # vertical lines that cross both.
    y0 = max(0, round(staff.lines[0] - ABOVE * staff.spacing))
    strip = [bars for bars, staves in systems(rgb[y0:]) if staves]
    if strip:
        candidates = strip[-1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    xs = _merge_candidates([c.x for c in candidates], _full_height_columns(gray, staff.lines), staff.spacing)
    barlines, rejected = [], []
    for x in xs:
        bar, reason = find_barline(gray, x, staff.lines, staff.spacing)
        if reason:
            rejected.append((x, reason))
        elif not barlines or bar.left > barlines[-1].right + staff.spacing / 2:  # same barline seen twice
            barlines.append(bar)
    return Staff(tuple(staff.lines), staff.spacing, tuple(barlines), tuple(rejected))


UNREAD_LINES = 4    # a frame with no staff found that shows this many long horizontal lines...
UNREAD_EVEN = .1    # ...at the video's TAB line spacing (within this share of it) may hold music


def unread_lines(rgb, spacing):
    """True if a crop with no staff found still shows UNREAD_LINES or more long horizontal lines (as the
    vendored finder sees lines) one after another at the video's TAB line `spacing` (within UNREAD_EVEN of
    it): staff lines the finder could not fit, so music may be missing. Photos and logos give long lines
    too, but not evenly at the staff's spacing."""
    centres = _line_centres(rgb)
    tol = UNREAD_EVEN * spacing
    for start in centres:
        chain = [start]
        while True:
            nxt = [c for c in centres if abs(c - chain[-1] - spacing) <= tol]
            if not nxt:
                break
            chain.append(nxt[0])
        if len(chain) >= UNREAD_LINES:
            return True
    return False


def _line_centres(rgb):
    """Centres (y) of long horizontal lines, the vendored finder's way: ink after its background
    normalisation, opened with a long horizontal kernel, rows covering DEFAULT.horizontal_coverage."""
    from .vendor.layout.source import _analysis_gray
    gray = _analysis_gray(rgb)
    ink = (gray < DEFAULT.ink_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(rgb.shape[1] * DEFAULT.horizontal_fraction)), 1))
    rows = np.flatnonzero((cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel) > 0).mean(axis=1) >= DEFAULT.horizontal_coverage)
    runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if len(rows) else []
    return [float(r.mean()) for r in runs if len(r) <= DEFAULT.max_line_rows]


def staff_gaps(gray, lines, spacing, least=2):
    """Inner breaks in a staff: [(x0, x1)] runs of columns, longer than `least` spacings, where fewer than
    three of its lines are drawn, between the first and last columns showing five or more. Two systems
    printed side by side (a coda after a gap) break the staff this way."""
    present = _present(gray, lines)
    full = np.flatnonzero(present >= 5)
    if not len(full):
        return []
    broken = np.flatnonzero(present[full[0]:full[-1] + 1] < 3) + full[0]
    runs = np.split(broken, np.flatnonzero(np.diff(broken) > 1) + 1) if len(broken) else []
    # The staff must run on for `least` spacings on both sides (a dark border column at the crop's edge
    # shows "every line" for a few px: blind test 5 video 5's last page).
    return [(int(r[0]), int(r[-1]) + 1) for r in runs if len(r) > least * spacing
            and (present[:r[0]] >= 5).sum() > least * spacing and (present[r[-1] + 1:] >= 5).sum() > least * spacing]


def blank_staff(image):
    """One column of empty staff for an image: the median of every column, row by row.

    Ink is sparse, so the median of each pixel row is paper, or the staff line where one runs.
    """
    return np.median(image, axis=1).astype(image.dtype)


# --- Notation + TAB -----------------------------------------------------------------------------
# A standard (5-line) staff printed above the TAB staff, one system with it. The TAB staff stays
# the authority for rows and cuts; the notation staff is found, checked and carried along.

NOTATION_COVER = .9     # its lines run over this share of the TAB staff's extent (I, J: 1.00; a row of
                        # chord diagrams whose fret lines line up: 0.25-0.34)
PARTNER = 1             # px: a TAB barline's notation partner has its left and right edges this close
SYSTEM_SIDE = 3         # spacings: a system barline's notation lines run on at least this far each side
SYSTEM_EVEN = .15       # spacings: the five lines it crosses are evenly spaced within this


@dataclass
class Notation:
    lines: tuple        # five y positions
    spacing: float
    partners: tuple     # per TAB barline: (left, right) of its notation stroke, or None
    loose: tuple        # x of notation strokes passing the barline rule with no TAB barline (stems)


def _present(gray, lines, x0=0, x1=None):
    """Per column (x0..x1): how many of the lines are darker than halfway between the paper and the
    lines' typical darkness there."""
    rows = np.array([gray[max(0, round(y) - 1):round(y) + 2].min(axis=0) for y in lines], np.float32)
    paper = float(np.median(gray))
    line = float(np.median(rows[:, x0:x1]))
    return (rows < (paper + line) / 2).sum(axis=0)


def _five_of_six(rgb, gray, staff, x0, x1):
    """Five-line staves found as six-line fits above the TAB staff: a row of beams and noteheads one
    spacing above (or below) the notation staff can complete an even sixth "line", and the vendored
    finder prefers six (blind 4 video 3, last row). The five that run over NOTATION_COVER of the TAB
    extent are the staff if the sixth, at one end, runs over less than half of it."""
    out = []
    for _, staves in _systems_with_margin(rgb, MARGIN, count=6):
        for st in staves:
            if st.lines[-1] >= staff.lines[0] - staff.spacing:
                continue
            for five, extra in ((st.lines[1:], st.lines[0]), (st.lines[:-1], st.lines[-1])):
                cover = float((_present(gray, five, x0, x1)[x0:x1] >= 4).mean())
                alone = float((_present(gray, list(five) + [extra], x0, x1)[x0:x1] >= 6).mean())
                if cover >= NOTATION_COVER and alone < .5:
                    out.append(SourceStaff(tuple(five), float(np.median(np.diff(five)))))
    return out


def notation_staff(rgb, staff):
    """The notation staff over this TAB staff (a Notation), or None.

    The vendored finder returns 5-line staves too; one counts if it lies wholly above the TAB staff
    (more than a spacing clear of it) and at least 4 of its 5 lines run over NOTATION_COVER of the
    TAB staff's extent. A row of chord diagrams can line up into five "lines", but broken into
    diagram-wide pieces; a 5-line fit inside the TAB staff overlaps it.
    """
    from .measures import staff_extent
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x0, x1 = staff_extent(gray, staff.lines, staff.spacing)
    if x1 - x0 < 4 * staff.spacing:
        return None
    found = []
    for _, staves in _systems_with_margin(rgb, MARGIN, count=5):
        for st in staves:
            if st.lines[-1] >= staff.lines[0] - staff.spacing:
                continue
            cover = float((_present(gray, st.lines, x0, x1)[x0:x1] >= 4).mean())
            if cover >= NOTATION_COVER:
                found.append(st)
    if not found:
        found = _five_of_six(rgb, gray, staff, x0, x1)
    if not found:
        return None
    st = max(found, key=lambda s: s.lines[0])
    partners = []
    for bar in staff.barlines:
        if bar.system:                             # drawn through both staves: its own partner
            partners.append((bar.left, bar.right))
            continue
        mate, _ = find_barline(gray, round(bar.strokes[0][2]), st.lines, st.spacing)   # from its first stroke
        partners.append((mate.left, mate.right) if mate and abs(mate.left - bar.left) <= PARTNER
                        and abs(mate.right - bar.right) <= PARTNER else None)
    loose = []
    for x in _full_height_columns(gray, st.lines):
        if any(bar.left - staff.spacing / 2 <= x <= bar.right + staff.spacing / 2 for bar in staff.barlines):
            continue
        mate, _ = find_barline(gray, x, st.lines, st.spacing)
        if mate and (not loose or abs(mate.x - loose[-1]) > staff.spacing / 2):
            loose.append(round(mate.x))
    return Notation(tuple(st.lines), st.spacing, tuple(partners), tuple(loose))
