"""Stage 5: clean the measures, break them into justified lines, write the PDF.

1. Clean each measure with the original screenshot cleanup. A white border is added first
   and removed after, so clean() cannot trim the image or erase ink touching a cut edge,
   and every staff coordinate from stage 4 stays exact. Scale to the video's median staff
   spacing (one staff size everywhere). Measure images are never scaled after that.
2. Every line is justified to the same staff width T (the typical source row width, or for a
   scrolling video the width of one view). Line breaks minimise the sum over all lines (the
   last one too) of (stretch - 1)^2, stretch = T / the line's natural width; a line over
   MAX_STRETCH is avoided when possible and warned about.
3. Compose each line naturally: measures sit on a common top staff line, each staff part
   starting where the previous one ended; overhanging annotations overlap by darkest pixel.
   Measures that were not neighbours in the source row are checked for collisions outside the
   staff and shifted apart if needed (the gap bridged with their own staff lines).
4. Justify: the extra width is spread over the line's safe columns in proportion: columns
   whose only ink besides the staff lines is flat over SAFE_CLEAR spacings on both sides
   (nothing, a beam, a let-ring line, the near-horizontal top of a tie), so no digit, stem,
   barline, diagram, text, arc end, vibrato or slide mark is ever widened. Each safe column is
   repeated as needed. No run of safe columns (a gap) grows by more than GAP_GROW spacings; a
   measure with no safe column counts as one gap before its ending barline. What the capped
   gaps cannot take goes evenly, as blank staff, before every ending barline of the line (at
   the nearest clear column, so a number printed over a barline is not cut).
5. Every line image has the same left and right gutter for overhangs (capped), so all staves
   start and end at the same place; build_pdf then draws every line at the full page width
   with one scale.
"""
import json
import re

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4, letter

from .measures import CORE_PAD, FRINGE as CUT_FRINGE
from .vendor.preprocess.cleaning import clean
from .vendor.render.pdf import build_pdf

INK = 150
VISIBLE = 220           # any visibly non-white ink (clean() lightens gray numbers past INK)
BORDER = 12             # white border while cleaning (> clean()'s 10px edge trim)
COLLISION_SHIFT = 1.5   # max extra space between two measures, in staff spacings
STEP = 2                # shift step in pixels
TOUCH = 3               # pixels of contact that count as a collision
LINE_PAD = .4           # blank space kept above/below each line's ink, in spacings
MAX_STRETCH = 1.4       # warn when a line must be stretched more than this
OVER_MAX = 100          # cost multiplier for a line over MAX_STRETCH
SAFE_CLEAR = .25        # spacings a safe column keeps from any glyph (more than the gap inside a number)
LINE_ROWS = 1           # px above/below a staff line that always count as the line...
LINE_REACH = 2          # ...and up to this far, rows dark along most of the line (thick lines)
GUTTER = 2.0            # max overhang kept left of the first / right of the last staff, in spacings
TEXT_GAP = .8           # glyphs off the staff this close (share of the taller one's height) form one text group
TEXT_GLYPH = 2.0        # ...if each is at most this many spacings wide (beams, ties, rules are not glyphs)
STEM = .15              # ...and not a stroke this narrow and a spacing or more tall (a stem)
GAP_GROW = 1.0          # spacings: the most any one gap between glyphs may grow when justifying
PAD_REACH = 2.0         # spacings left of a barline searched for ink printed across it
FRAGMENT = .6           # a tie/slur fragment leaving a measure is at most this tall (spacings)
WIDTH_RANGE = (.9, 1.1)  # line width tried around the typical source width, keeping the least stretch


FRINGE_DARK = 140     # a dark tinted pixel (max channel under this)...
FRINGE_INK = 80       # ...within FRINGE_REACH px of near-black ink (max channel under this) is that ink's edge
FRINGE_REACH = 2
ERASED_INK = 128      # source pixels darker than this that clean() turned lighter than this were erased


def neutral_fringes(rgb):
    """Dark tinted pixels at the edge of black ink made neutral gray (their luma). Blind 4 video 1:
    its black digits have warm chroma fringes 3+ px wide; clean() took them for reddish UI and its
    removal mask, dilated, ate into the strokes ('5' lost its top, '0' its side). UI colours
    (cursors, highlights) are bright and keep their colour."""
    top = rgb.max(axis=2)
    ink = (top < FRINGE_INK).astype(np.uint8)
    near = cv2.dilate(ink, np.ones((2 * FRINGE_REACH + 1,) * 2, np.uint8)).astype(bool)
    fix = near & (top < FRINGE_DARK) & (top.astype(int) - rgb.min(axis=2) > 0)
    if not fix.any():
        return rgb
    out = rgb.copy()
    out[fix] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[fix][:, None]
    return out


def _clean_measure(path):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    paper = int(np.median(rgb.reshape(-1, 3), axis=0).mean())
    padded = cv2.copyMakeBorder(rgb, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=(paper,) * 3)
    out = clean(padded)
    if out.shape != padded.shape[:2]:
        # clean() trimmed it (dark paper: a fade or dark overlay). Keep the measure uncleaned
        # rather than lose every staff coordinate; the caller warns.
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), False
    neutral = neutral_fringes(padded)
    if neutral is not padded:
        # Ink clean() erased as colour, at the tinted edge of black ink: taken from a cleaning of the
        # image with those edges neutral. Nothing else changes.
        again = clean(neutral)
        if again.shape == out.shape:
            source = cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY)
            erased = (source < ERASED_INK) & (out >= ERASED_INK) & (again < out)
            out[erased] = again[erased]
    return out[BORDER:-BORDER, BORDER:-BORDER], True


def _prepare(run_dir, records):
    """Cleaned, scaled measures, the median spacing, and the measures clean() could not handle."""
    spacing = float(np.median([r['spacing'] for r in records]))
    measures, uncleaned = [], []
    for r in records:
        image, cleaned = _clean_measure(run_dir / r['image'])
        if not cleaned:
            uncleaned.append(r['measure'])
        scale = spacing / r['spacing']
        if abs(scale - 1) > .005:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        measures.append(dict(r, img=image, top=r['staff_lines'][0] * scale,
                             lines=[y * scale for y in r['staff_lines']],
                             sx0=round(r['staff_x0'] * scale), sx1=round(r['staff_x1'] * scale),
                             bar0=round(r.get('ending_x0', r['staff_x1'] - 1) * scale),
                             locks=[(round(a * scale), round(b * scale)) for a, b in r.get('spanning_blocks', [])],
                             nlines=[y * scale for y in r.get('notation_lines', [])],
                             lead=[[round(v * scale) for v in box] for box in r.get('lead_blocks', [])]))
    return measures, spacing, uncleaned


def _header(run_dir, spacing):
    """Notation mode: the header (brace, clefs, key) cut off every row in stage 4, cleaned and scaled
    like a measure, or None."""
    path = run_dir / 'header.json'
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    image, _ = _clean_measure(run_dir / info['image'])
    scale = spacing / info['spacing']
    if abs(scale - 1) > .005:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return {'img': image, 'top': info['staff_lines'][0] * scale, 'lines': [y * scale for y in info['staff_lines']],
            'nlines': [y * scale for y in info['notation_lines']], 'width': image.shape[1]}


def _lead_moved(m, spacing):
    """A measure that opened a source row, placed mid-line: blocks printed over the header (its
    number at the system start) move right, over the measure's start (its barline): centred on it,
    or as little left of that as keeps SAFE_CLEAR spacings from the measure's own ink (I m9: the
    first chord's fingering starts just after the barline). Returns the measure with its image
    cropped to its new left extent."""
    img = m['img'].copy()
    h, w = img.shape
    clear = max(2, round(SAFE_CLEAR * spacing))
    # each with its antialiased fringe (stage 4's FRINGE), or a faint ghost stays behind; a number's
    # dark digits and pale halo are separate blocks: overlapping boxes move as one
    boxes = []
    for x0, y0, x1, y1 in sorted(m['lead']):
        box = [max(0, x0 - CUT_FRINGE), max(0, y0 - CUT_FRINGE), min(w, x1 + CUT_FRINGE), min(h, y1 + CUT_FRINGE)]
        for other in boxes:
            if box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]:
                other[:] = [min(box[0], other[0]), min(box[1], other[1]), max(box[2], other[2]), max(box[3], other[3])]
                break
        else:
            boxes.append(box)
    for x0, y0, x1, y1 in boxes:
        piece = m['img'][y0:y1, x0:x1].copy()
        img[y0:y1, x0:x1] = 255
        size = x1 - x0
        centred = max(0, min(w - size, m['sx0'] - size // 2))
        ink = img[max(0, y0 - clear):y1 + clear] < VISIBLE
        nx = next((x for x in range(centred, max(-1, centred - round(1.5 * spacing)), -1)
                   if not ink[:, max(0, x - clear):x + size + clear].any()), centred)
        np.minimum(img[y0:y1, nx:nx + size], piece, out=img[y0:y1, nx:nx + size])
    inked = np.flatnonzero((img[:, :m['sx0']] < VISIBLE).any(axis=0))
    trim = int(inked[0]) if len(inked) else m['sx0']
    return dict(m, img=img[:, trim:], sx0=m['sx0'] - trim, sx1=m['sx1'] - trim, bar0=m['bar0'] - trim,
                locks=[(a - trim, b - trim) for a, b in m['locks']], lead=[])


def stretch_cost(natural, target):
    ratio = target / natural
    return (ratio - 1) ** 2 * (OVER_MAX if ratio > MAX_STRETCH else 1)


def break_lines(widths, target, with_cost=False, keep=None, lead=0):
    """Split measure widths into lines no wider than `target`, minimising the summed squared
    stretch of every line (the last one included). A single measure wider than target is a
    line of its own. keep[i]: measure i must share a line with measure i + 1 (an annotation
    spans their barline); such a group is never split, even if it is wider than target. `lead`: px
    every line starts with (the notation header), counted in each line's width."""
    n = len(widths)
    keep = keep or [False] * n
    best, cut = [0.0] + [float('inf')] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        if j < n and keep[j - 1]:
            continue                    # no line may end between measures kept together
        total = lead
        for i in range(j, 0, -1):
            total += widths[i - 1]
            if i > 1 and keep[i - 2]:
                continue                # nor start there
            if total > target and i < j and not all(keep[i - 1:j - 1]):
                break
            cost = best[i - 1] + stretch_cost(min(total, target), target)
            if cost < best[j]:
                best[j], cut[j] = cost, i - 1
    lines, j = [], n
    while j > 0:
        lines.append(list(range(cut[j], j)))
        j = cut[j]
    return (lines[::-1], best[n]) if with_cost else lines[::-1]


def choose_width(widths, typical, keep=None, lead=0):
    """The line width within WIDTH_RANGE of the typical one whose best line breaks stretch least."""
    widest = max(widths) + lead
    lo, hi = max(widest, round(typical * WIDTH_RANGE[0])), max(widest, round(typical * WIDTH_RANGE[1]))
    best = min((break_lines(widths, t, True, keep, lead)[1], abs(t - typical), t) for t in range(lo, hi + 1, 2))
    return best[2]


def _outside_core(height, lines):
    """Row mask: True outside the staff band (where overhanging annotations live)."""
    mask = np.ones(height, bool)
    mask[max(0, round(lines[0]) - CORE_PAD):round(lines[-1]) + CORE_PAD + 1] = False
    return mask


def _clip(canvas, image, x):
    """Columns of image that land inside the canvas at x: (canvas x0, x1, image x0, x1)."""
    a0, a1 = max(0, x), min(canvas.shape[1], x + image.shape[1])
    return a0, max(a0, a1), a0 - x, max(a0, a1) - x


def _paste(canvas, image, x, y):
    """Darkest-pixel paste, clipped to the canvas. Returns the number of columns with ink left out
    (an overhang wider than everything before it on the line)."""
    a0, a1, b0, b1 = _clip(canvas, image, x)
    target = canvas[y:y + image.shape[0], a0:a1]
    np.minimum(target, image[:, b0:b1], out=target)
    inked = (image < VISIBLE).any(axis=0)
    return int(inked[:b0].sum() + inked[b1:].sum())


def _collides(canvas, image, x, y, outside):
    h = image.shape[0]
    a0, a1, b0, b1 = _clip(canvas, image, x)
    region = canvas[y:y + h, a0:a1] < VISIBLE
    new = (image[:, b0:b1] < VISIBLE) & outside[:, None]
    near = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return int((new & near).sum())


def _bridge(canvas, measure, x_from, x_to, y):
    """Extend the measure's own staff lines across a gap we added (layout only, no music)."""
    img = measure['img']
    columns = img[:, measure['sx0'] + 2:measure['sx1'] - 2]
    for line in measure['lines'] + measure.get('nlines', []):
        for dy in (-1, 0, 1):
            row = round(line) + dy
            if 0 <= row < img.shape[0]:
                value = int(np.median(columns[row]))
                canvas[y + row, x_from:x_to] = np.minimum(canvas[y + row, x_from:x_to], value)


def compose_line(measures, spacing, header=None):
    """Natural layout of one line. Returns a dict: canvas (gray), staff_x0/staff_x1 (where the
    staff runs in the canvas), lines (staff line rows), spans [(x0, x1, ending barline x)] per
    measure, notes {shifted, overlaps}. With a notation `header`, the line starts with it (its
    left edge is the staff start) and its columns are locked (never widened); a measure that
    opened a source row but is not first here gets its number moved over its start."""
    measures = [_lead_moved(m, spacing) if k and m.get('lead') else m for k, m in enumerate(measures)]
    placed = measures + ([header] if header else [])
    above = max(m['top'] for m in placed)
    below = max(m['img'].shape[0] - m['top'] for m in placed)
    left_pad = 0 if header else max(0, measures[0]['sx0'])
    max_shift = round(COLLISION_SHIFT * spacing)
    width = left_pad + sum(m['sx1'] - m['sx0'] for m in measures) + max(m['img'].shape[1] for m in measures) \
        + max_shift * len(measures) + (header['width'] if header else 0)
    canvas = np.full((round(above + below) + 2, width), 255, np.uint8)
    notes = {'shifted': [], 'overlaps': [], 'clipped': []}
    spans, locks = [], []
    cursor, previous = left_pad, None
    if header:
        _paste(canvas, header['img'], 0, round(above - header['top']))
        locks.append((0, header['width']))
        cursor = header['width']
    for m in measures:
        y = round(above - m['top'])
        h, w = m['img'].shape
        outside = _outside_core(h, m['lines'])
        if m.get('nlines'):
            outside &= _outside_core(h, m['nlines'])
        neighbours = previous is not None and previous['row'] == m['row'] and previous['in_row'] + 1 == m['in_row']
        shift, contact = 0, 0
        if previous is not None and not neighbours:
            best = None
            for shift in range(0, max_shift + 1, STEP):
                contact = _collides(canvas, m['img'], cursor + shift - m['sx0'], y, outside)
                if best is None or contact < best[1]:
                    best = (shift, contact)
                if contact < TOUCH:
                    break
            shift, contact = best
            if contact >= TOUCH:
                notes['overlaps'].append((previous['measure'], m['measure'], contact))
            elif shift:
                notes['shifted'].append((previous['measure'], m['measure'], shift))
            if shift:
                _bridge(canvas, m, cursor, cursor + shift, y)
        x = cursor + shift - m['sx0']
        lost = _paste(canvas, m['img'], x, y)
        if lost:
            notes['clipped'].append((m['measure'], lost))
        spans.append((cursor + shift, x + m['sx1'], x + m['bar0']))
        locks += [(x + a, x + b) for a, b in m.get('locks', [])]
        cursor += shift + m['sx1'] - m['sx0']
        previous = m
    first = measures[0]
    lines = [round(above - first['top']) + l for l in first['lines']]
    nlines = [round(above - first['top']) + l for l in first.get('nlines', [])]
    right = max(x1 for _, x1, _ in spans)
    right = max(right, max(spans[i][0] - measures[i]['sx0'] + measures[i]['img'].shape[1]
                           for i in range(len(measures))))
    return {'canvas': canvas[:, :right], 'staff_x0': left_pad, 'staff_x1': cursor, 'lines': lines,
            'nlines': nlines, 'spans': spans, 'notes': notes, 'locks': locks}


def staff_rows(gray, lines):
    """Rows that belong to a staff line: LINE_ROWS px around each line, plus any row within
    LINE_REACH px that is visibly dark along most of the line (a thick or antialiased line)."""
    on_line = np.zeros(gray.shape[0], bool)
    for y in lines:
        on_line[max(0, round(y) - LINE_ROWS):round(y) + LINE_ROWS + 1] = True
        for row in range(max(0, round(y) - LINE_REACH), min(gray.shape[0], round(y) + LINE_REACH + 1)):
            if np.median(gray[row]) < VISIBLE:
                on_line[row] = True
    return on_line


def safe_columns(gray, lines, x0, x1, spacing):
    """Boolean per column of gray[:, x0:x1]: True where repeating the column cannot be seen as a
    stretched glyph. Off the staff lines, every column within SAFE_CLEAR spacings on both sides
    must hold the same ink rows (1 row of slack). So a column is safe when the only ink crossing
    it besides the staff lines is flat: nothing, a beam, a straight let-ring/sustain line, or the
    near-horizontal top of a tie or slur. Digits, stems, barlines, diagrams, text, the sloped ends
    of arcs, vibrato and slide marks all change from column to column and are never widened (nor
    is the gap between two digits of a number). Also returns staff_only (no ink but the lines)."""
    clear = max(2, round(SAFE_CLEAR * spacing))
    on_line = staff_rows(gray[:, x0:x1], lines)
    ink = (gray < VISIBLE)[~on_line]
    staff_only = ~ink.any(axis=0)
    loose = cv2.dilate(ink.astype(np.uint8), np.ones((3, 1), np.uint8)).astype(bool)
    width = ink.shape[1]
    flat = np.zeros(width, bool)
    flat[clear:width - clear] = True
    for d in range(1, clear + 1):
        a, b = ink[:, :width - d], ink[:, d:]
        same = ~((a & ~loose[:, d:]) | (b & ~loose[:, :width - d])).any(axis=0)   # column x vs x + d
        flat[clear:width - clear] &= same[clear:width - clear] & same[clear - d:width - clear - d]
    for a, _, b, _ in text_groups(gray[:, x0:x1], lines, spacing):
        flat[a:b] = False
    return flat[x0:x1], staff_only


def text_groups(gray, lines, spacing):
    """[(x0, y0, x1, y1)] boxes of text above the staff: glyphs (ink components at most TEXT_GLYPH
    spacings wide, not stems) whose gap is under TEXT_GAP of the taller one's height
    and whose rows overlap. A tempo mark (L: "= 80" in a font larger than the digits), a chord name
    or a technique label is then never widened inside, even where a gap between its glyphs is wider
    than SAFE_CLEAR. Only groups of two or more glyphs; a lone glyph is unsafe by itself."""
    ink = (gray < VISIBLE).astype(np.uint8)
    ink[max(0, round(lines[0]) - LINE_REACH):] = 0          # above the staff only (rhythm stems are below)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = sorted((x, y, x + w, y + h) for x, y, w, h, area in stats[1:]
                   if w <= TEXT_GLYPH * spacing and area >= 4
                   and not (w <= STEM * spacing and h >= spacing))       # a stem is not a glyph
    parent = list(range(len(boxes)))
    tallest = max((y1 - y0 for _, y0, _, y1 in boxes), default=0)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (ax0, ay0, ax1, ay1) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            reach = TEXT_GAP * max(ay1 - ay0, by1 - by0)
            if bx0 - ax1 > TEXT_GAP * tallest:
                break                                   # sorted by x: nothing further can be this close
            if bx0 - ax1 <= reach and min(ay1, by1) > max(ay0, by0):
                parent[root(j)] = root(i)
    groups = {}
    for i, box in enumerate(boxes):
        groups.setdefault(root(i), []).append(box)
    return [(min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g))
            for g in groups.values() if len(g) > 1]


def _gaps(safe_x):
    """Runs of consecutive safe columns: [(first x, length)]."""
    if not len(safe_x):
        return []
    breaks = np.flatnonzero(np.diff(safe_x) > 1) + 1
    return [(int(run[0]), len(run)) for run in np.split(safe_x, breaks)]


def _fill(sizes, extra, cap):
    """Water-fill: each gap grows in proportion to its size, none by more than `cap`. Returns the
    growth per gap; whatever is left when every gap is capped is the caller's remainder."""
    sizes = np.asarray(sizes, float)
    if not len(sizes) or extra <= 0:
        return np.zeros(len(sizes))
    if len(sizes) * cap <= extra:
        return np.full(len(sizes), float(cap))
    lo, hi = 0.0, extra / sizes.min() + cap
    for _ in range(60):
        rate = (lo + hi) / 2
        lo, hi = (rate, hi) if np.minimum(sizes * rate, cap).sum() < extra else (lo, rate)
    return np.minimum(sizes * hi, cap)


def justify(line, target, spacing):
    """Widen the line's staff part to exactly `target` px. The extra width is spread over every
    safe column of the line in proportion (each run of safe columns, a gap, grows with its length,
    each of its columns repeated), but no gap grows by more than GAP_GROW spacings, so no gap
    turns into a hole. A measure with no safe column counts as one gap of its staff width at the
    line's safe density, placed before its ending barline. What the capped gaps cannot take is
    spread evenly as blank staff before every ending barline of the line. Returns (gray image,
    staff_x0, staff_x1, report dict)."""
    gray, x0, x1 = line['canvas'], line['staff_x0'], line['staff_x1']
    extra = target - (x1 - x0)
    info = {'stretch': target / (x1 - x0), 'padded': [], 'safe_px': 0, 'barline_pad_px': 0}
    if extra <= 0:
        return gray, x0, x1, info
    staves = [line['lines']] + ([line['nlines']] if line.get('nlines') else [])
    every = sorted(y for staff in staves for y in staff)       # notation mode: both staves' lines
    safe, staff_only = safe_columns(gray, every, x0, x1, spacing)
    for a, b in line.get('locks', []):          # a block spanning a barline is never widened
        safe[max(0, a - x0):max(0, b - x0)] = False
    safe_x = np.flatnonzero(safe) + x0
    info['safe_px'] = len(safe_x)
    edge = max(2, round(SAFE_CLEAR * spacing))

    def before_barline(s0, s1, bar):
        # Just before the barline's antialiased edge; _pad_rows then moves the insertion left,
        # row by row, past anything printed across the barline.
        x = int(min(max(bar, s0), s1 - 1))
        while x - 1 > s0 and not staff_only[x - 1] and bar - x < edge:
            x -= 1
        for a, b in line.get('locks', []):      # not through a block spanning a barline
            if a < x < b:
                x = max(s0, a)
        return x

    gaps = [(x, n, 'safe') for x, n in _gaps(safe_x)]
    owners = [m for m, (s0, s1, _) in enumerate(line['spans']) if ((safe_x >= s0) & (safe_x < s1)).any()]
    with_safe = sum(line['spans'][m][1] - line['spans'][m][0] for m in owners)
    density = len(safe_x) / with_safe if with_safe else 1.0
    for m, (s0, s1, bar) in enumerate(line['spans']):
        if m not in owners:
            gaps.append((before_barline(s0, s1, bar), (s1 - s0) * density, 'pad'))
            info['padded'].append(m)
    grow = _fill([n for _, n, _ in gaps], extra, GAP_GROW * spacing)
    remainder = extra - grow.sum()
    bars = [before_barline(*span) for span in line['spans']]

    # Insertions as (x, px wanted, kind): safe columns repeat themselves, pads are blank staff.
    wanted = []
    for (x, n, kind), g in zip(gaps, grow):
        if kind == 'safe':
            wanted += [(x + k, g / n, 'safe') for k in range(int(n))]
        else:
            wanted.append((x, g, 'pad'))
    if remainder > 0:
        info['barline_pad_px'] = round(remainder)
        wanted += [(x, remainder / len(bars), 'pad') for x in bars]
    wanted.sort(key=lambda w: (w[0], w[2] == 'pad'))

    # Template for blank staff: the median of the line's staff-only columns.
    plain = np.flatnonzero(staff_only[x0:x1]) + x0
    if len(plain):
        template = np.median(gray[:, plain], axis=1).astype(np.uint8)
    else:
        template = np.full(gray.shape[0], 255, np.uint8)
        for y in every:
            template[round(y)] = 120

    # Safe columns are repeated whole; pads are inserted row by row afterwards (right to left),
    # at columns mapped through the safe insertions before them.
    pieces, start, done, acc, pads = [], 0, 0, 0.0, []
    for x, px, kind in wanted:
        acc += px
        add = round(acc) - done
        done += add
        if add <= 0:
            continue
        if kind == 'safe':
            pieces += [gray[:, start:x + 1], np.repeat(gray[:, x:x + 1], add, axis=1)]
            start = x + 1
        else:
            pads.append((x, add))
    pieces.append(gray[:, start:])
    out = np.hstack(pieces)
    if pads:
        # Map an original column c to its column in `out`: c plus the safe columns added before c.
        repeats = np.zeros(gray.shape[1], int)
        done, acc = 0, 0.0
        for x, px, kind in wanted:
            acc += px
            add = round(acc) - done
            done += add
            if add > 0 and kind == 'safe':
                repeats[x] += add
        mapped = np.concatenate([[0], np.cumsum(1 + repeats)])[:-1]   # mapped[c] = out column of original c
        spans = line['spans']
        for x, add in sorted(pads, reverse=True):
            span = next((sp for sp in spans if sp[0] <= x < sp[1]), (x, x + 1, x))
            cols, fill = _pad_rows(gray, x, span[1], every, spacing, template, staves)
            out = _insert_rows(out, mapped[cols], fill, add)
    return out, x0, x0 + out.shape[1] - gray.shape[1] + (x1 - x0), info


def _pad_rows(gray, x, end, lines, spacing, template, staves=None):
    """Where, row by row, blank staff goes in before a barline whose padding point is x (the
    measure's staff ends at `end`), and what fills it. Returns (column per row, fill per row).

    Ink is labelled near the barline with the staff lines and the barline's own stroke removed.
    - Ink crossing x (a measure number or chord name printed across the barline) moves whole
      with the barline: in its rows the insertion goes to its left edge (and so for anything
      else in those columns that reaches into the same rows).
    - A thin dark stroke inside the staff reaching the barline (a tie or slur fragment leaving the
      measure) is split at x and its column x - 1 repeated through the padding: it stays on its
      note and runs on to the barline.
    - Everything else gets blank staff at x.
    `staves` (notation mode): each staff's line list; its barline stroke is removed staff by staff,
    and a tie fragment is one inside either staff. `lines` are then all their lines.
    """
    height = gray.shape[0]
    reach = round(PAD_REACH * spacing)
    w0, w1 = max(0, x - reach), min(gray.shape[1], end + round(spacing))
    ink = gray[:, w0:w1] < VISIBLE
    ink[staff_rows(gray[:, w0:w1], lines)] = False
    bands = [(round(staff[0]), round(staff[-1])) for staff in (staves or [lines])]
    for top, bottom in bands:
        stroke = (gray[top:bottom + 1, w0:w1] < VISIBLE).mean(axis=0) >= .9
        ink[top - 1:bottom + 2, stroke] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    cx = x - w0
    cols = np.full(height, x)
    fill = template.copy()

    dark = gray[:, w0:w1] < INK

    def fragment(i):
        # A tie/slur leaving the measure: thin, dark, inside the staff (not a number or a chord
        # name above it), reaching the barline.
        top_i, tall = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
        left, right = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
        if staves and right >= cx and cx - left >= spacing and (dark & (labels == i)).any():
            # Notation: a tie leaves its note (one component with the notehead, stem and beam, or
            # alone between the staves); it is a fragment if it is thin where the padding goes.
            rows = np.flatnonzero(labels[:, cx - 1] == i)
            return len(rows) and rows[-1] - rows[0] < FRAGMENT * spacing
        return (tall <= FRAGMENT * spacing and any(top_i + tall > top and top_i <= bottom for top, bottom in bands)
                and (dark & (labels == i)).any() and right >= cx)

    crossing = [i for i in set(labels[:, cx - 1]) & set(labels[:, cx]) if i and not fragment(i)]
    # Text of two or more glyphs printed across the barline is never split by the padding (L m65: the 0
    # of "= 80" crossed it and moved, the 8 stayed). Mostly before the barline, it stays whole over its
    # measure and in its rows the padding goes in after it; mostly after it, it moves whole with the barline.
    after = []
    for gx0, gy0, gx1, gy1 in text_groups(gray[:, w0:w1], lines, spacing):
        inside = {i for i in np.unique(labels[gy0:gy1, gx0:gx1]) if i and gx0 <= stats[i, cv2.CC_STAT_LEFT]
                  and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] <= gx1}
        if not (gx0 < cx < gx1 and inside & set(crossing)):
            continue
        if gx0 + gx1 < 2 * cx and gx1 < w1 - w0:          # mostly before the barline: stays with its measure
            crossing = [i for i in crossing if i not in inside]
            after.append((gy0, gy1, gx1))
        else:                                              # mostly after it (a measure number): moves whole
            crossing = sorted(set(crossing) | inside)
    moved = np.zeros(height, bool)
    left = cx
    while crossing:
        for i in crossing:
            moved[stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]] = True
            left = min(left, stats[i, cv2.CC_STAT_LEFT])
        # Ink between the new insertion column and x, in the moved rows, that reaches outside them
        # would be torn: it moves too.
        region = np.unique(labels[moved][:, left:cx])
        crossing = [i for i in region if i and not moved[stats[i, cv2.CC_STAT_TOP]:
                                                         stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]].all()]
    cols[moved] = w0 + left
    for gy0, gy1, gx1 in after:
        rows = np.zeros(height, bool)
        rows[gy0:gy1] = True
        cols[rows & ~moved] = w0 + gx1

    for i in set(labels[:, cx - 1]) - {0}:
        top_i, tall = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
        if not fragment(i) or moved[top_i:top_i + tall].any():
            continue
        rows = np.flatnonzero(labels[:, cx - 1] == i)
        rows = np.arange(max(0, rows[0] - 1), min(height, rows[-1] + 2))
        fill[rows] = gray[rows, x - 1]
    return cols, fill


def _insert_rows(image, cols, fill, add):
    """Insert `add` columns of `fill` into each row y of image at column cols[y]."""
    height, width = image.shape
    out = np.empty((height, width + add), image.dtype)
    for y in range(height):
        c = int(cols[y])
        out[y, :c] = image[y, :c]
        out[y, c:c + add] = fill[y]
        out[y, c + add:] = image[y, c:]
    return out


NO_MEASURES = 'No measures were found in this video. See report.md for the warnings.'


def pdf_path(run_dir, title):
    """The PDF is named after the title (characters a file name cannot hold become spaces)."""
    name = ' '.join(re.sub(r'[\\/:*?"<>|\x00-\x1f]', ' ', title).split()).strip('. ')[:120]
    return run_dir / f"{name or run_dir.name}.pdf"


def heading(run_dir, title=None, artist=None):
    """(title, artist, pdf path) for a run. A title or artist given once is kept in config.json
    for re-runs; with no title the run folder's name is printed. The PDF of an earlier run under
    another name (and the old `sheet.pdf`) is removed, so the folder holds one PDF."""
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    title, artist = title or config.get('title'), artist or config.get('artist')
    pdf = pdf_path(run_dir, title or run_dir.name)
    for old in {config.get('pdf'), 'sheet.pdf'} - {None, pdf.name}:
        (run_dir / old).unlink(missing_ok=True)
    new = {**config, 'pdf': pdf.name, **({'title': title} if title else {}), **({'artist': artist} if artist else {})}
    if new != config:
        config_path.write_text(json.dumps(new, indent=1) + '\n')
    return title or run_dir.name, artist, pdf


def _empty_pdf(path, title, page_size, note=NO_MEASURES, artist=None):
    """A one-page PDF with a note instead of music (the run still produces a PDF)."""
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=page_size)
    c.setTitle(title)
    c.setFont('Helvetica-Bold', 14)
    c.drawString(72, page_size[1] - 72, title)
    c.setFont('Helvetica', 11)
    if artist:
        c.drawString(72, page_size[1] - 88, artist)
    for i, text in enumerate(note.splitlines()):
        c.drawString(72, page_size[1] - 116 - 16 * i, text[:110])
    c.save()


def build_sheet(run_dir, report, title=None, page='letter', artist=None):
    title, artist, pdf = heading(run_dir, title, artist)
    records = json.loads((run_dir / 'measures.json').read_text())
    rows = json.loads((run_dir / 'rows.json').read_text())
    if not records:
        _empty_pdf(pdf, title, A4 if page == 'a4' else letter, artist=artist)
        report.warn('sheet', 'No measures to lay out: wrote a one-page PDF with a note instead.')
        report.stage('sheet', pdf=pdf.name, title=title, artist=artist or 'none', pages=1, lines=0)
        return pdf
    measures, spacing, uncleaned = _prepare(run_dir, records)
    if uncleaned:
        report.warn('sheet', f'measures {uncleaned}: clean() would have trimmed them (dark paper: a fade '
                             'or overlay?), so they are printed uncleaned.')

    # Line width: the typical source row's staff width (keeps the video's density). A scrolling
    # video's rows are stitched strips, so there it is the width of one view.
    scrolled = [r for r in rows if r.get('scroll')]
    if scrolled:
        target = float(np.median([r['view_width'] * spacing / r['spacing'] for r in scrolled]))
    else:
        row_widths = {}
        for m in measures:
            row_widths[m['row']] = row_widths.get(m['row'], 0) + m['sx1'] - m['sx0']
        target = float(np.median(list(row_widths.values())))
    widths = [m['sx1'] - m['sx0'] for m in measures]
    header = _header(run_dir, spacing)
    lead = header['width'] if header else 0
    target += lead                     # source rows printed their header too; stage 4 cut it off
    typical = target
    keep = [bool(m.get('keep_with_next')) for m in measures]
    target = choose_width(widths, typical, keep, lead)
    lines = break_lines(widths, target, keep=keep, lead=lead)

    composed = [compose_line([measures[i] for i in indices], spacing, header) for indices in lines]
    # A collision shift can make a line wider than planned: then every line gets that width.
    target = max(target, max(c['staff_x1'] - c['staff_x0'] for c in composed))
    gutter_cap = round(GUTTER * spacing)
    left = min(gutter_cap, max(c['staff_x0'] for c in composed))
    right = min(gutter_cap, max(c['canvas'].shape[1] - c['staff_x1'] for c in composed))

    out_dir = run_dir / 'sheet'
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob('line-*.png'):
        old.unlink()
    strips, shifted, overlaps, hanging, stretches, padded, clipped, over = [], [], [], [], [], [], [], []
    barline_pad, cut_short = [], []
    for number, (indices, line) in enumerate(zip(lines, composed), 1):
        group = [measures[i] for i in indices]
        image, x0, x1, info = justify(line, target, spacing)
        # Same gutters on every line: staves start and end at the same x everywhere.
        image = cv2.copyMakeBorder(image, 0, 0, max(0, left - x0), max(0, right - (image.shape[1] - x1)),
                                   cv2.BORDER_CONSTANT, value=255)
        x0 = max(x0, left)
        cut_left, cut_right = image[:, :x0 - left], image[:, x0 + target + right:]
        if (cut_left < INK).any() or (cut_right < INK).any():
            clipped.append(number)
        image = image[:, x0 - left:x0 + target + right]
        ink_rows = np.flatnonzero((image < INK).any(axis=1))
        pad = round(LINE_PAD * spacing)
        if len(ink_rows):
            image = image[max(0, ink_rows[0] - pad):ink_rows[-1] + pad + 1]
        cv2.imwrite(str(out_dir / f'line-{number:02d}.png'), image)
        strips.append((image, spacing))
        stretches.append(info['stretch'])
        padded += [group[m]['measure'] for m in info['padded']]
        barline_pad.append(info['barline_pad_px'])
        if info['stretch'] > MAX_STRETCH:
            over.append(number)
        shifted += line['notes']['shifted']
        overlaps += line['notes']['overlaps']
        cut_short += line['notes']['clipped']
        first = group[0]
        if first['sx0'] > 0 and not header:
            hanging.append((first['measure'], first['sx0']))

    pages, _ = build_pdf(strips, str(pdf), title, A4 if page == 'a4' else letter, .4 * 72, 6, subtitle=artist)

    for a, b, contact in overlaps:
        report.warn('sheet', f'm{a} and m{b} still touch outside the staff ({contact}px) after '
                    f'{COLLISION_SHIFT} spacings of extra space; check that spot in the PDF.')
    for number in over:
        report.warn('sheet', f'line {number} is stretched {stretches[number - 1]:.2f}x (over {MAX_STRETCH}); '
                    f'no better line break was found; {barline_pad[number - 1]}px of it is blank staff '
                    'before barlines (the music has too few flat gaps).')
    if cut_short:
        report.warn('sheet', 'overhang wider than the line before it, cut off: ' +
                    ', '.join(f'm{n} ({px}px of ink columns)' for n, px in cut_short) + '; check the measure image.')
    if clipped:
        report.warn('sheet', f'lines {clipped}: an overhang wider than {GUTTER} spacings was clipped at the margin.')
    report.stage('sheet', pdf=pdf.name, title=title, artist=artist or 'none', pages=pages, lines=len(lines),
                 measures_per_line=' '.join(str(len(l)) for l in lines),
                 line_width_px=target, typical_source_width_px=round(typical), staff_spacing_px=round(spacing, 2),
                 stretch_per_line=' '.join(f'{s:.2f}' for s in stretches),
                 stretch_max=round(max(stretches), 2), stretch_limit=MAX_STRETCH,
                 padded_before_barline=[f'm{n}' for n in padded] or 'none',
                 gap_cap_px=round(GAP_GROW * spacing),
                 barline_padding_px_per_line=' '.join(str(p) for p in barline_pad),
                 gutters_px=[left, right],
                 **({'notation_header_px': header['width']} if header else {}),
                 collisions_fixed_by_spacing=[f'm{a}|m{b} +{s}px' for a, b, s in shifted] or 'none',
                 collisions_remaining=[f'm{a}|m{b}' for a, b, _ in overlaps] or 'none',
                 line_start_overhangs=[f'm{n} hangs {px}px left' for n, px in hanging] or 'none')
    _debug_lines(run_dir, lines, measures, stretches)
    return pdf


def _debug_lines(run_dir, lines, measures, stretches):
    """debug/5-lines.jpg: every justified line with its measures and stretch."""
    images = []
    for number, indices in enumerate(lines, 1):
        line = cv2.cvtColor(cv2.imread(str(run_dir / 'sheet' / f'line-{number:02d}.png'), 0), cv2.COLOR_GRAY2BGR)
        label = np.full((22, line.shape[1], 3), 255, np.uint8)
        text = (f"line {number}: stretch {stretches[number - 1]:.2f}  " +
                ' '.join(f"m{measures[i]['measure']}" for i in indices))
        cv2.putText(label, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 200), 1, cv2.LINE_AA)
        images.append(np.vstack([label, line]))
    width = max(i.shape[1] for i in images)
    images = [cv2.copyMakeBorder(i, 0, 8, 0, width - i.shape[1], cv2.BORDER_CONSTANT, value=(200, 200, 200))
              for i in images]
    stacked = np.vstack(images)
    cv2.imwrite(str(run_dir / 'debug' / '5-lines.jpg'), cv2.resize(stacked, None, fx=.5, fy=.5,
                interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
