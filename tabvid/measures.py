"""Stage 4: cut each merged row into measures at its barlines.

Inside the staff, a measure runs from just after the previous barline to the right
edge of its own ending barline, so a double or final barline (all strokes) stays with
the measure it ends. The first measure of a row also keeps the opening barline and
TAB clef.

Above and below the staff, a straight cut would slice chord diagrams, chord names and
measure numbers printed over a barline. There, nearby ink is grouped into blocks
(name + markers + grid; the digits of a number) and each block goes whole to one
measure: one straddling a barline goes to the measure starting there. A measure image can therefore overhang its staff
a little; `staff_x0`/`staff_x1` record where the staff part sits so stage 5 can
line staves up exactly. Ink that is not part of the measure is painted as paper.

A row whose first barline sits well after the start of the staff lines (no opening
barline, e.g. a page that begins with the TAB clef) gets a first measure from where the
staff lines begin to that barline. Ink outside the staff that spans most of the row (a rule
along the crop edge) is page furniture, not an annotation: it is painted as paper.

The TAB clef is kept only at the start of the piece. The clef is found as the ink that sits
at the start of the staff identically in most rows, and must be at the start of the first row
too; in every later row that has it, it is replaced by blank staff.

Measures are numbered in playback order. The printed numbers are read and checked against
this order in numbers.py.

Duplicates: page-switching videos keep every measure (repeated measures are real music), except
as written, where a page shown again with the same printed numbers is printed once (replays.py).
Scrolling videos: where a join found no overlap, a new strip that starts by repeating the old
strip's end measure for measure, printed numbers included, loses the repeats (drop_seam_repeats);
anything less is kept and warned.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .staff import blank_staff, find_staff, is_final

INK = 150
PALE = 10            # annotation blocks include ink this many gray levels darker than the paper
TRAILING = 1         # staff running on this many spacings past the piece's last barline, holding no notes, is an
                     # empty unfinished measure (E: m52 to the crop edge): its blocks (its number) are dropped
ROW_NUMBER_REACH = 4  # notation, header kept as printed: blocks above the notation staff starting within this many
                      # notation spacings of the row start (a number printed over the clef) are its row-start blocks
NARROW = .5          # warn when a measure is under half the median width of its row (not when its own printed
                     # number is read and follows its neighbour's: a false barline leaves a piece with no number)
SEAM_WIDTH = .25     # spacings: measures repeated across a strip seam have the same width within this...
SEAM_MATCH = .05     # ...and at most this share of their staff ink differs
WIDE = 1.6           # warn (possible missing barline) when over 1.6x the video's median measure width
WIDE_AROUND = 10      # a full-height stroke that failed the barline rule inside a wide measure is looked for
WIDE_ENDS = 2         # with this many spacings of staff around it, this many spacings clear of its ends
CORE_PAD = 2         # the straight-cut band is the staff (top to bottom line) plus this many pixels
NEAR = 1.5           # within this many spacings of the staff (ties, measure numbers): join glyphs sideways only
BLOCK_JOIN = .6      # ink closer than this many spacings joins one block (chord name + diagram)
TIE_ROWS = 3         # in-staff ink on both sides of a barline in this many rows = a tie/slur crossing
STRADDLE_RIGHT = .25  # a block straddling a cut goes right if at least this share of its width is past the cut
FRINGE = 2            # px around a block that belong to it (antialiased edge)
DARK_CORE = 100       # a dark glyph has a pixel darker than this (gray measure numbers do not)
DIAGRAM_H, DIAGRAM_W = 2.5, 1.5   # a chord diagram grid is at least this tall and wide (spacings)
SPAN_TOL = 3.3       # a block spans a barline if it reaches more than this many spacings past it into the
                     # other measure: beyond stage 5's line gutter (sheet.GUTTER), a line break would clip it
NUMBER_REACH = 1.5    # a pale glyph near the staff within this many spacings of a measure start is its number
OPEN_GAP = 2          # a first barline this many spacings past the staff start leaves a measure before it
SLIVER = 4            # a cut narrower than this many spacings holding no notes (only a time signature, or
                      # nothing) is not a measure: it is the start of the measure after it
FURNITURE = .8        # a block outside the staff wider than this share of the staff is page furniture
CUT_DARK = 150        # ink cut by the crop's top or bottom edge: darker than this...
CUT_INK = .1          # ...and at least this many square spacings of it (warned, kept)
EDGE_SLIVER = 2       # a pale ink run this many spacings long within FRINGE px of the crop's top or bottom
                      # edge, with paper under it, is a sliver of something beyond the crop (a panel border)
EDGE_LINE, EDGE_ROWS = .5, 8   # a border cut by the crop can be several rows thick: rows from the edge that are
                      # ink over EDGE_LINE of the width (at most EDGE_ROWS) join the outer FRINGE rows
STATIC_ROWS = .6      # ink outside the staff touching the crop edge at the same place in this share of the rows
                      # (at least 3) is furniture too: a system line, the foot of a bracket the crop cut through
LEDGER_STEPS = 6      # notation mode: ledger lines are looked for 1..6 notation spacings beyond its staff...
LEDGER_MIN, LEDGER_MAX = 1.2, 6   # ...as dark runs this many notation spacings long (a tie touching one adds to it)...
LEDGER_THIN = 4       # ...whose end columns are at most this many px of ink tall there (not a beam or notehead)
HEADER_BEFORE = 3     # notation mode: the header (brace, clefs, key) is looked for from this many TAB spacings
HEADER_SEARCH = 16    # before the staff start to this many after it...
UNDER = 1.5           # notation spacings past a glyph after the header searched for TAB ink (a note's digit)
LINE_ROWS = 1         # px each side of a staff line's centre that are the line (header halo search)
FOREIGN = .5          # ...and a dark component under this share inside the header is not the header's
HEADER_GAP = 1.0      # ...as the shared ink up to the first gap of this many notation spacings (I: 0.55 inside
                      # the header, 2.0 after it; J: 4.0 after it)
CLEF_SEARCH = 6       # the clef is looked for in this many spacings from the staff start...
CLEF_MAX = 4          # ...is at most this wide...
CLEF_GAP = .5         # ...and ends at an ink-free gap of this many spacings
CLEF_ROWS = .6        # it must be there, the same, in this share of rows (at least 3 rows)
CLEF_MATCH = .25      # a row has the clef if under this share of the clef's ink differs
CLEF_SHIFT = 3        # px of misalignment allowed between rows


def staff_extent(gray, lines, spacing):
    """(first, last + 1) column where the staff lines run (at least 5 of 6 present).

    Staff lines can be light gray, so a line counts as present where it is darker than halfway
    between the paper and the line's typical darkness.
    """
    rows = np.array([gray[max(0, round(y) - 1):round(y) + 2].min(axis=0) for y in lines], np.float32)
    paper, line = float(np.median(gray)), float(np.median(rows))
    if paper - line < 20:
        # A short staff (K's last page: one measure over 12% of the width) leaves the median at paper:
        # the line's darkness is then taken where the lines are clearly darker than paper.
        dark = rows[rows < paper - 20]
        if len(dark) < len(lines) * spacing:
            return 0, gray.shape[1]
        line = float(np.median(dark))
    on = (rows < (paper + line) / 2).sum(axis=0) >= 5
    run = max(3, round(spacing))
    full = np.flatnonzero(np.convolve(on, np.ones(run), 'valid') >= .9 * run)
    if not len(full):
        return 0, gray.shape[1]
    lines_on = np.flatnonzero(on[full[0]:full[-1] + run]) + full[0]
    return int(lines_on[0]), int(lines_on[-1]) + 1


def cut_row(row, width, start=None):
    """[(x0, x1, ending barline)] staff cuts for a row; x1 is exclusive.

    `start` is where the staff lines begin; if the first barline is well past it, the
    stretch before that barline is a measure too (a row with no opening barline). With the
    notation header cut off (`cut_start`), a key or time signature printed between it and a
    near opening barline (`# ||:`, blind 4 video 3) goes with the first measure.
    """
    bars = row['barlines']
    opening = bool(bars) and start is not None and bars[0]['left'] - start > OPEN_GAP * row['spacing']
    cuts = [(start, bars[0]['right'] + 1, bars[0])] if opening else []
    first = min(start, bars[0]['left']) if bars and 'cut_start' in row and start is not None else None
    cuts += [(a['right'] + 1 if i or opening else first if first is not None else a['left'], b['right'] + 1, b)
             for i, (a, b) in enumerate(zip(bars, bars[1:]))]
    return cuts or [(0, width, None)]


def join_slivers(cuts, gray, lines, spacing, opening=False, clef=0):
    """(cuts, joined): a narrow cut holding no notes is joined to the cut that follows it.

    A time or key change can be printed between two barlines (`| 3/4 ||:`): the stretch between
    them holds only glyphs as tall as a time signature. It belongs to the next measure. So does an
    `opening` stretch (before the row's first barline, no opening barline) holding no notes, however
    wide: a clef, key and time signature before a `||:` (blind 4 video 3). `clef`: px at the start of
    the opening stretch holding a TAB clef left as printed (notation header kept), not notes.
    """
    from .structure import holds_notes, staff_ink
    out, start, joined = [], None, 0
    for k, (x0, x1, ending) in enumerate(cuts):
        x0 = x0 if start is None else start
        start = None
        skip = clef if k == 0 and opening else 0
        if k + 1 < len(cuts) and (x1 - x0 < SLIVER * spacing or (k == 0 and opening)) and \
                not holds_notes(staff_ink(gray, lines, min(x1, x0 + skip), x1), spacing):
            start, joined = x0, joined + 1
            continue
        out.append((x0, x1, ending))
    return out, joined


def _shifted(mask, dy):
    """The mask moved down by dy rows (up if negative), False where nothing moves in."""
    out = np.zeros_like(mask)
    if dy >= 0:
        out[dy:] = mask[:mask.shape[0] - dy]
    else:
        out[:dy] = mask[-dy:]
    return out


def static_edge_ink(rows):
    """Per row, a mask of ink outside the staff that touches the crop edge at the same place
    (measured from the staff's top line: a page can sit a few px higher or lower) in most rows.
    Page-switching rows of one size, at least 3; otherwise None. The place is judged in the
    median row's window: what a row sitting higher or lower shows beyond it stays."""
    if len(rows) < 3 or len({row['gray'].shape for row in rows}) != 1 or any(row.get('scroll') for row in rows):
        return None
    tops = [round(row['staff_lines'][0]) for row in rows]
    ref = int(np.median(tops))
    static = np.mean([_shifted(row['gray'] < INK, ref - top) for row, top in zip(rows, tops)], axis=0) >= STATIC_ROWS
    span = max(round(row['staff_lines'][-1] - row['staff_lines'][0]) for row in rows)
    static[max(0, ref - CORE_PAD):ref + span + CORE_PAD + 1] = False
    count, labels = cv2.connectedComponents(static.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    mask = np.isin(labels, border[border > 0])
    if not mask.any():
        return None
    reach = max(FRINGE, round(rows[0]['spacing'] / 4))        # its antialiased edge and pale tail
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * reach + 1,) * 2, np.uint8)).astype(bool)
    mask[max(0, ref - CORE_PAD):ref + span + CORE_PAD + 1] = False
    return {row['row']: _shifted(mask, top - ref) for row, top in zip(rows, tops)}


def clear_edge_rules(row):
    """Paint out a rule along the top or bottom edge of the crop: pixel rows at the edge that are
    ink over FURNITURE of the width (a progress bar, the border of the panel), plus FRINGE rows.
    As a block it would be furniture, but a system line can tie it to the staff, and ink tied to
    the staff is cut straight into every measure. Returns the number of rows painted."""
    gray, image = row['gray'], row['image_bgr']
    inked = (gray < INK).mean(axis=1) >= FURNITURE
    paper = np.median(image[gray > 200], axis=0) if (gray > 200).any() else np.array([255, 255, 255])
    painted = 0
    for order in (range(len(inked)), range(len(inked) - 1, -1, -1)):
        run = 0
        for y in order:
            if not inked[y]:
                break
            run += 1
        if run:
            ys = list(order)[:run + FRINGE]
            image[ys] = paper
            gray[ys] = int(round(float(np.mean(paper))))
            painted += run
    return painted


def rejected_strokes(image_bgr, x0, x1, spacing):
    """x of the full-height strokes inside the measure [x0, x1) that failed the barline rule
    (ink running on past the staff, a wavy stroke): where a barline may have been missed."""
    lo = max(0, round(x0 - WIDE_AROUND * spacing))
    staff = find_staff(cv2.cvtColor(image_bgr[:, lo:round(x1 + WIDE_AROUND * spacing)], cv2.COLOR_BGR2RGB))
    return [x + lo for x, reason in (staff.rejected if staff else ())
            if reason != 'gap inside staff' and x0 + WIDE_ENDS * spacing < x + lo < x1 - WIDE_ENDS * spacing]


def wide_measures(records, rows, proven=(), numbered=False):
    """Measures over WIDE x the video's median width, sorted into
    (signed, width only, one measure by its numbers, quiet); signed = [(record, why)].
    Long measures are common (a change of time signature, a run of short notes), so:
    - signed: a second sign of a missing barline: a full-height stroke inside it that failed the
      barline rule, or a second measure number. Always warned, one warning each.
    - width only: also over WIDE x the median of the other measures of its own row (a row of
      long measures is just a dense page). Warned together, in one warning.
    - by its numbers: the next measure carries the next number (`proven`).
    - quiet: not wide for its row; or, in a video that prints measure numbers (`numbered`), it
      holds no second number, as a measure with a missed barline would."""
    typical = float(np.median([r['staff_x1'] - r['staff_x0'] for r in records]))
    images = {row['row']: row['image_bgr'] for row in rows}
    signed, width_only, by_numbers, quiet = [], [], [], []
    for record in records:
        width = record['staff_x1'] - record['staff_x0']
        if width <= WIDE * typical:
            continue
        if record['measure'] in proven:
            by_numbers.append(record['measure'])
            continue
        x0 = record['row_x0'] + record['staff_x0']
        strokes = rejected_strokes(images[record['row']], x0, x0 + width, record['spacing'])
        others = [r['staff_x1'] - r['staff_x0'] for r in records if r['row'] == record['row'] and r is not record]
        if strokes:
            signed.append((record, 'a full-height stroke ' + ', '.join(f'{round(x - x0)}px' for x in strokes) +
                           ' into it failed the barline rule'))
        elif record.get('extra_blocks'):
            signed.append((record, 'it holds a second measure number'))
        elif numbered or (others and width <= WIDE * float(np.median(others))) or record is records[-1]:
            quiet.append(record['measure'])        # the piece's last measure: often long (a held final chord; user)
        else:
            width_only.append(record)
    return signed, width_only, by_numbers, quiet


def find_clef(rows):
    """The clef shared by the starts of most rows: (ink mask, x offset from the staff start), or None.

    Each row's start (from the staff start, top line to bottom line, staff lines and full-height
    barlines removed) is reduced to an ink mask; the clef is the first glyph run of the
    pixel-wise majority, if it is narrow and ends at a gap.
    """
    if len(rows) < 3:
        return None
    spacing = float(np.median([r['spacing'] for r in rows]))
    width, height = round(CLEF_SEARCH * spacing), round(5 * spacing) + 1
    windows = [w for w in (_start_ink(r, 0, width, height) for r in rows) if w is not None]
    if len(windows) < 3:
        return None
    majority = np.mean(windows, axis=0) >= CLEF_ROWS
    columns = np.flatnonzero(majority.any(axis=0))
    if not len(columns):
        return None
    gaps = np.flatnonzero(np.diff(columns) > CLEF_GAP * spacing)
    end = columns[gaps[0]] + 1 if len(gaps) else columns[-1] + 1
    if end - columns[0] > CLEF_MAX * spacing or (not len(gaps) and end + CLEF_GAP * spacing > width):
        return None
    x0, x1 = max(0, columns[0] - FRINGE), end + FRINGE
    return majority[:, x0:x1], int(x0)


def _start_ink(row, x_off, width, height, y_off=0):
    """Ink mask (height, width) at x_off from the row's staff start and y_off from its top line,
    with staff lines and full-height strokes (an opening barline) removed; None if out of range."""
    gray = row['gray']
    top, x0 = round(row['staff_lines'][0]) + y_off, row['staff_start'] + x_off
    if x0 < 0 or top < 0:
        return None
    window = gray[top:top + height, x0:x0 + width]
    if window.shape != (height, width):
        return None
    ink = window < INK
    for y in row['staff_lines']:
        ink[max(0, round(y) - top - 1):max(0, round(y) - top + 2)] = False
    ink[:, (window < INK).mean(axis=0) >= .9] = False
    return ink


def _mismatch(ink, clef):
    """Pixels of either mask with no pixel of the other within 1 px."""
    grow = np.ones((3, 3), np.uint8)
    near_ink = cv2.dilate(ink.astype(np.uint8), grow).astype(bool)
    near_clef = cv2.dilate(clef.astype(np.uint8), grow).astype(bool)
    return int((clef & ~near_ink).sum() + (ink & ~near_clef).sum())


def clef_at(row, clef):
    """(dx, dy) where this row has the clef (within CLEF_SHIFT px, under CLEF_MATCH of its ink
    differing), or None."""
    mask, x_off = clef
    height, width = mask.shape
    tries = [(dx, dy) for dx in range(-CLEF_SHIFT, CLEF_SHIFT + 1) for dy in (-1, 0, 1)]
    scored = [(_mismatch(ink, mask), dx, dy) for dx, dy in tries
              if (ink := _start_ink(row, x_off + dx, width, height, dy)) is not None]
    if not scored:
        return None
    miss, dx, dy = min(scored)
    return None if miss > CLEF_MATCH * mask.sum() else (dx, dy)


def remove_clef(row, clef, at):
    """Paint the clef's whole box, found at `at` = (dx, dy), as blank staff (its pale antialiased
    parts too); other ink in it would have failed the match."""
    mask, x_off = clef
    dx, dy = at
    ys = np.flatnonzero(mask.any(axis=1))
    image = row['image_bgr']
    top = round(row['staff_lines'][0]) + dy
    y0, y1 = max(0, top + ys[0] - FRINGE), min(image.shape[0], top + ys[-1] + FRINGE + 1)
    x0 = row['staff_start'] + x_off + dx
    image[y0:y1, x0:x0 + mask.shape[1]] = blank_staff(image)[y0:y1, None]


def clear_clef(rows):
    """Remove the TAB clef from every later row that has it. Returns (rows cleared, later rows
    without it, report note).

    A clef is printed at the start of the piece: the shared start ink only counts as the clef
    if the first row has it and so do CLEF_ROWS of the later rows. Otherwise it is music that
    recurs at row starts (V2: "(0)" with a tie on 4 of 10 rows) or a clef printed once; either
    way nothing is removed and nothing is warned.
    """
    clef = find_clef(rows)
    if clef is None:
        return [], [], 'kept as found'
    found = {row['row']: clef_at(row, clef) for row in rows}
    later = [row for row in rows[1:] if found[row['row']] is not None]
    if found[rows[0]['row']] is None or len(later) < CLEF_ROWS * (len(rows) - 1):
        matched = sorted(r for r, at in found.items() if at is not None)
        return [], [], f'not shared across rows (start ink matched rows {matched}); kept as printed'
    for row in later:
        remove_clef(row, clef, found[row['row']])
        row['gray'] = cv2.cvtColor(row['image_bgr'], cv2.COLOR_BGR2GRAY)
    cleared = [row['row'] for row in later]
    return cleared, sorted({r['row'] for r in rows[1:]} - set(cleared)), None


def _system_ink(row, x_off, width, y_off, height):
    """Ink mask (height, width) at x_off from the row's TAB staff start and y_off from its TAB top
    line, with both staves' lines removed. Pixels beyond the crop count as paper (a brace can run
    on past a tight crop)."""
    gray = row['gray']
    top, x0 = round(row['staff_lines'][0]) + y_off, row['staff_start'] + x_off
    ink = np.zeros((height, width), bool)
    a, b = max(0, top), min(gray.shape[0], top + height)
    c, d = max(0, x0), min(gray.shape[1], x0 + width)
    if a >= b or c >= d:
        return None
    ink[a - top:b - top, c - x0:d - x0] = gray[a:b, c:d] < INK
    for y in list(row['staff_lines']) + list(row['notation_lines']):
        ink[max(0, round(y) - top - 1):max(0, round(y) - top + 2)] = False
    return ink


def find_header(rows):
    """The notation header shared by every row's start: dict(mask, x_off, y_off) or None.

    Brace, system line, both clefs and the key signature sit at the same place at the start of
    every row (all but one when there are 5 or more rows). The header runs from the first shared
    column through glyph runs closer than HEADER_GAP notation spacings and ends at the first wider
    gap: music that opens several rows alike (I rows 1, 3, 7; J pages 1 and 3) lies beyond it. A time
    signature is printed once, so it is not shared and stays with measure 1.
    """
    if len(rows) < 3:
        return None
    spacing = float(np.median([r['spacing'] for r in rows]))
    n_spacing = float(np.median([r['notation_spacing'] for r in rows]))
    rise = float(np.median([r['staff_lines'][0] - r['notation_lines'][0] for r in rows]))
    y_off = -round(rise + 2 * spacing)
    height = round(rise + 9 * spacing)          # 2 spacings above the notation staff to 2 below the TAB
    x_off, width = -round(HEADER_BEFORE * spacing), round((HEADER_BEFORE + HEADER_SEARCH) * spacing)
    windows = [_system_ink(r, x_off, width, y_off, height) for r in rows]
    if any(w is None for w in windows):
        return None
    share = 1.0 if len(rows) < 5 else (len(rows) - 1) / len(rows)
    shared = np.mean(windows, axis=0) >= share - 1e-9
    columns = np.flatnonzero(shared.any(axis=0))
    if not len(columns):
        return None
    gaps = np.flatnonzero(np.diff(columns) > HEADER_GAP * n_spacing)
    end = columns[gaps[0]] + 1 if len(gaps) else columns[-1] + 1
    if not len(gaps) and end + HEADER_GAP * n_spacing > width:
        return None                                   # no gap: shared ink runs on (not a header)
    x0, x1 = max(0, columns[0] - FRINGE), min(width, end + FRINGE)
    return {'mask': shared[:, x0:x1], 'x_off': x_off + int(x0), 'y_off': y_off}


def header_at(row, header):
    """(dx, dy) where this row has the header (within CLEF_SHIFT px, under CLEF_MATCH of its ink
    differing), or None."""
    mask = header['mask']
    height, width = mask.shape
    tries = [(dx, dy) for dx in range(-CLEF_SHIFT, CLEF_SHIFT + 1) for dy in (-1, 0, 1)]
    scored = [(_mismatch(ink, mask), dx, dy) for dx, dy in tries
              if (ink := _system_ink(row, header['x_off'] + dx, width, header['y_off'] + dy, height)) is not None]
    if not scored:
        return None
    miss, _, dx, dy = min((m, abs(dx) + abs(dy), dx, dy) for m, dx, dy in scored)   # ties: least shift
    return None if miss > CLEF_MATCH * mask.sum() else (dx, dy)


def runs_on(row, end):
    """True if the row's header runs on past the shared one ending at column `end`: its first
    notation glyph after `end` starts within HEADER_GAP notation spacings (as close as the header's
    own glyphs) and has no TAB ink under it or within UNDER notation spacings after it. A note has
    its fret digit under it; a key signature (an added sharp) or a time signature has none."""
    gray, lines, spacing = row['gray'], row['notation_lines'], row['notation_spacing']
    y0, y1 = round(lines[0] - spacing / 2), round(lines[-1] + spacing / 2) + 1
    band = gray[y0:y1, end:end + round(4 * spacing)] < INK
    for y in lines:
        band[max(0, round(y) - y0 - 1):max(0, round(y) - y0 + 2)] = False
    columns = np.flatnonzero(band.any(axis=0))
    edge = np.flatnonzero(np.diff(columns) > 1)            # the first run of columns ends after edge[0]
    lead = columns[:edge[0] + 1] if len(edge) else columns
    if len(lead) and lead[0] == 0 and lead[-1] <= FRINGE:
        columns = columns[len(lead):]     # the header's own antialiased edge at `end`, not a glyph after it
    if not len(columns) or columns[0] > HEADER_GAP * spacing:
        return False
    gaps = np.flatnonzero(np.diff(columns) > spacing / 2)
    a, b = end + columns[0], end + (columns[gaps[0]] if len(gaps) else columns[-1]) + 1
    tab = row['staff_lines']
    # fingering printed left of a chord (I rows 3, 7) has the chord's digits a notehead further right
    under = gray[round(tab[0]) + 2:round(tab[-1]) - 1, max(0, a - round(spacing / 2)):b + round(UNDER * spacing)] < INK
    for y in tab:
        under[max(0, round(y) - round(tab[0]) - 2 - 1):max(0, round(y) - round(tab[0]) - 2 + 2)] = False
    under[:, under.mean(axis=0) >= .9] = False        # a barline
    return not under.any()


def strip_headers(rows, run_dir):
    """Notation mode: find the header and cut it off every row, so stage 5 can print it at the start
    of every line and nowhere else. Each row's header ink is painted as paper and its first measure
    starts after it (`cut_start`); the header is saved once, from the first row, as
    measures/header.png (shared ink and the staff lines under it only, so no measure number rides
    along) with header.json. Returns (note, warning or None). With no header, or a row whose start
    differs (a key or clef change), nothing is cut and headers stay as printed (warned)."""
    header = find_header(rows)
    if header is None:
        return 'not found; kept as printed', ('no notation header (brace, clefs, key) is shared by the starts of '
                                              'the rows; clefs and keys are kept as printed in each row.')
    found = {row['row']: header_at(row, header) for row in rows}
    width = header['mask'].shape[1]
    end = {row['row']: row['staff_start'] + header['x_off'] + at[0] + width - FRINGE
           for row in rows if (at := found[row['row']]) is not None}
    # the first row may print a time signature right after the header; a later row that runs on
    # has a key or time change there
    differ = [r for r, at in found.items() if at is None or (
        r != rows[0]['row'] and runs_on(next(row for row in rows if row['row'] == r), end[r]))]
    if differ:
        return 'kept as printed (rows differ)', (
            f'the notation header (clef, key) of rows {differ} differs from the other rows (a key or clef change?): '
            'no header is printed at line starts; each row keeps its own as printed, so a line starting '
            'mid-row has none.')
    mask = header['mask']
    height, width = mask.shape
    first = rows[0]
    for row in rows:
        dx, dy = found[row['row']]
        x0, y0 = row['staff_start'] + header['x_off'] + dx, round(row['staff_lines'][0]) + header['y_off'] + dy
        image, gray = row['image_bgr'], row['gray']
        paper = np.median(image[gray > 200], axis=0) if (gray > 200).any() else np.array([255, 255, 255])
        # In the row's frame: the header's ink, its antialiased fringe and its pale halo (any faint
        # component touching it, staff lines aside), so no speck is left to become a block.
        a, b = max(0, y0), min(gray.shape[0], y0 + height)
        grow = np.zeros((b - a, width), bool)
        grow[:] = mask[a - y0:b - y0]
        grow = cv2.dilate(grow.astype(np.uint8), np.ones((2 * FRINGE + 1,) * 2, np.uint8)).astype(bool)
        # dark ink mostly outside the header is not the header's, even where a speck the rows share lands
        # on it (row-start numbers 6, 10, 14... overlap at one place: blind 4 video 2)
        dark = gray[a:b, x0:x0 + width] < INK
        for y in list(row['staff_lines']) + list(row['notation_lines']):
            dark[max(0, round(y) - a - LINE_ROWS):max(0, round(y) - a + LINE_ROWS + 1)] = False
        count, pieces = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
        inside = np.bincount(pieces[grow], minlength=count)
        total = np.bincount(pieces.ravel(), minlength=count)
        foreign = np.flatnonzero((inside > 0) & (inside < FOREIGN * total))
        foreign = foreign[foreign > 0]
        if len(foreign):
            spot = cv2.dilate(np.isin(pieces, foreign).astype(np.uint8),
                              np.ones((2 * FRINGE + 1,) * 2, np.uint8)).astype(bool)
            grow &= ~spot
        faint = gray[a:b, x0:x0 + width] < np.median(gray) - PALE
        for y in list(row['staff_lines']) + list(row['notation_lines']):
            faint[max(0, round(y) - a - LINE_ROWS):max(0, round(y) - a + LINE_ROWS + 1)] = False
        _, parts = cv2.connectedComponents(faint.astype(np.uint8), connectivity=8)
        touched = np.unique(parts[grow & faint])
        grow |= np.isin(parts, touched[touched > 0])
        if row is first:
            keep = grow.copy()
            for y in list(row['staff_lines']) + list(row['notation_lines']):
                keep[max(0, round(y) - a - LINE_ROWS - 1):max(0, round(y) - a + LINE_ROWS + 2)] = True
            crop = image[a:b, x0:x0 + width]
            saved = np.where(keep[..., None], crop, paper.astype(np.uint8))
            cv2.imwrite(str(run_dir / 'measures' / 'header.png'), saved)
            (run_dir / 'header.json').write_text(json.dumps({
                'image': 'measures/header.png', 'width': width,
                'staff_lines': [round(y - a, 1) for y in row['staff_lines']],
                'notation_lines': [round(y - a, 1) for y in row['notation_lines']],
                'spacing': row['spacing']}, indent=1) + '\n')
        region = image[a:b, x0:x0 + width]
        region[grow] = paper
        row['gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        row['cut_start'] = x0 + width
    return f'cut from every row ({width}px) and printed at each line start', None


def staff_core(gray, band):
    """Pixels cut straight at barlines: the staff band, plus dark ink connected to it.

    Fret digits on the outer lines, stems and their beams, and ties drawn from a digit all touch
    the staff, so they are cut like the staff. Chord diagrams, chord names and measure numbers
    float free of it and are kept whole as annotation blocks.
    """
    _, components = cv2.connectedComponents((gray < INK).astype(np.uint8), connectivity=8)
    touching = np.unique(components[band[0]:band[1]])
    core = np.isin(components, touching[touching > 0])
    core[band[0]:band[1]] = True
    return core


def ledger_lines(gray, lines, spacing):
    """Mask of the notation staff's ledger lines: dark pixel-row runs LEDGER_MIN-LEDGER_MAX notation
    spacings long, within 2 px of a line position 1..LEDGER_STEPS spacings above or below the staff,
    thin at both ends (a beam or a notehead is taller there)."""
    ink = gray < INK
    height, mask = gray.shape[0], np.zeros(gray.shape, bool)

    def thin(y, x):
        column = ink[max(0, y - LEDGER_THIN):y + LEDGER_THIN + 1, x]
        return column.sum() <= LEDGER_THIN

    for k in range(1, LEDGER_STEPS + 1):
        for centre in (lines[0] - k * spacing, lines[-1] + k * spacing):
            for y in range(round(centre) - 2, round(centre) + 3):
                if not 0 <= y < height:
                    continue
                on = np.flatnonzero(ink[y])
                for run in np.split(on, np.flatnonzero(np.diff(on) > 1) + 1) if len(on) else []:
                    if LEDGER_MIN * spacing <= len(run) <= LEDGER_MAX * spacing and thin(y, run[0]) and thin(y, run[-1]):
                        mask[y, run] = True
    return mask


def system_core(gray, band, notation, spacing):
    """Notation mode: the straight-cut core of both staves. The TAB band and the notation band, and
    the notation staff's ledger lines, plus dark ink connected to any of them. Every notehead touches
    a staff line or a ledger line, so notes, stems, beams and ties between and beyond the staves are
    cut like the staff (a tie over a barline is split there, as in the TAB)."""
    lines = notation['lines']
    nband = (max(0, round(lines[0]) - CORE_PAD), min(gray.shape[0], round(lines[-1]) + CORE_PAD + 1))
    _, components = cv2.connectedComponents((gray < INK).astype(np.uint8), connectivity=8)
    seeds = np.zeros(gray.shape, bool)
    seeds[band[0]:band[1]] = seeds[nband[0]:nband[1]] = True
    seeds |= ledger_lines(gray, lines, notation['spacing'])
    touching = np.unique(components[seeds])
    core = np.isin(components, touching[touching > 0])
    core[band[0]:band[1]] = core[nband[0]:nband[1]] = True
    return core


def _blocks(gray, core, lines, spacing, also=None):
    """Label map of annotation blocks outside the staff core (0 = none) and each block's (center x, left, right).

    Dark ink near the staff only joins sideways, so two tie arcs or a tie and a slur mark stay
    apart. Further out, dark glyphs join in every direction, so a chord name, its markers and
    its diagram grid form one block. Pale ink (gray measure numbers) is grouped on its own and
    joins in every direction, so a number printed across a tie arc stays whole and apart from it.
    """
    height = gray.shape[0]
    # Gray digits can have parts darker than INK. Dark ink counts as dark only if it is at least
    # one spacing in area and has a pixel darker than DARK_CORE; the rest is pale, so a gray
    # number never splits into a dark digit and a pale one.
    # Pale pixels right next to real dark ink are that ink's antialiased edge, left to its FRINGE.
    count, specks, stats, _ = cv2.connectedComponentsWithStats((gray < INK).astype(np.uint8), connectivity=8)
    darkest = np.full(count, 255, np.int32)
    np.minimum.at(darkest, specks.ravel(), gray.ravel().astype(np.int32))
    real = (stats[:, cv2.CC_STAT_AREA] >= spacing) & (darkest < DARK_CORE)
    dark = real[specks] & (specks > 0)
    edge = cv2.dilate(dark.astype(np.uint8), np.ones((2 * FRINGE + 3,) * 2, np.uint8)).astype(bool)
    faint = gray < np.median(gray) - PALE
    pale = faint & ~edge
    dark &= ~core
    pale &= ~core
    near = np.zeros(height, bool)
    for staff in [lines] + ([also] if also else []):   # notation mode: near either staff
        near[max(0, round(staff[0] - NEAR * spacing)):min(height, round(staff[-1] + NEAR * spacing) + 1)] = True
    far = ~near
    size = max(3, round(BLOCK_JOIN * spacing))
    labels = np.zeros(gray.shape, np.int32)
    offset = 0
    square, sideways = np.ones((size, size), np.uint8), np.ones((1, size), np.uint8)
    for ink, rows, kernel in ((dark, near, sideways), (dark, far, square), (pale, near | far, square)):
        if ink is pale:
            pale_from = offset
        zone = ink & rows[:, None]
        joined = cv2.dilate(zone.astype(np.uint8), kernel) & rows[:, None].astype(np.uint8)
        count, zone_labels = cv2.connectedComponents(joined)
        labels[zone] = zone_labels[zone] + offset
        offset += count
    # A pale glyph printed across dark ink (a measure number over a tie arc) reclaims the
    # pixels it shares with that ink's edge, within its own bounding box.
    centers, numbers = {}, set()
    for label in np.unique(labels[labels > 0]):
        ys, cols = np.nonzero(labels == label)
        if label > pale_from:
            box = (slice(ys.min(), ys.max() + 1), slice(cols.min(), cols.max() + 1))
            labels[box][faint[box] & edge[box] & (labels[box] == 0)] = label
            if near[ys].all():
                numbers.add(int(label))   # pale glyph next to the staff: a measure number
        centers[int(label)] = (float(cols.mean()), int(cols.min()), int(cols.max()) + 1)
    return labels, centers, numbers


def _diagram_owners(labels, gray, centers, owner, cuts, spacing, numbers):
    """A block straddling a cut that holds chord diagrams over two or more measures (their chord
    names touch, so the names and diagrams form one block) goes to the measure with most of its
    diagrams. Changes `owner`; returns [(label, owner index, [other measure indices])]."""
    shared = []
    for label, (_, left, right) in centers.items():
        if label in numbers or not any(left < x0 < right for x0, _, _ in cuts[1:]):
            continue
        ys, xs = np.nonzero(labels[:, left:right] == label)
        mask = np.zeros((gray.shape[0], right - left), np.uint8)
        mask[ys, xs] = gray[ys, xs + left] < INK
        count, _, stats, middle = cv2.connectedComponentsWithStats(mask, connectivity=8)
        where = []
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_HEIGHT] >= DIAGRAM_H * spacing and stats[i, cv2.CC_STAT_WIDTH] >= DIAGRAM_W * spacing:
                x = middle[i][0] + left
                where.append(min(range(len(cuts)), key=lambda k: 0 if cuts[k][0] <= x < cuts[k][1]
                                 else min(abs(x - cuts[k][0]), abs(x - cuts[k][1]))))
        if len(set(where)) < 2:
            continue
        tally = {k: where.count(k) for k in set(where)}
        most = max(tally.values())
        best = owner[label] if tally.get(owner[label]) == most else min(k for k, n in tally.items() if n == most)
        owner[label] = best
        shared.append((label, best, sorted(set(where) - {best})))
    return shared


def _spanning(centers, owner, cuts, numbers, spacing):
    """Annotation blocks that span a barline: reaching more than SPAN_TOL spacings past it into
    the measure that does not own them. Smaller overhangs (a chord name or diagram peeking over
    the barline, 0.3-1.5 spacings in nine videos) still show whole at a line break, in the line's
    gutter. Measure numbers are left out: they sit over every barline and belong to one measure.
    Returns (indices k whose measure must stay on one line with the next, {k: [(left, right)]})."""
    keep_next, spanning = set(), {}
    tol = SPAN_TOL * spacing
    for label, k in owner.items():
        if label in numbers:
            continue
        _, left, right = centers[label]
        crossed = [j for j in range(len(cuts) - 1) if left < cuts[j + 1][0] < right
                   and (cuts[j + 1][0] - left if j + 1 == k else right - cuts[j + 1][0]) > tol]
        if crossed:
            keep_next.update(crossed)
            spanning.setdefault(k, []).append((left, right))
    return keep_next, spanning


def _row_start_block(labels, label, x0, notation):
    """A block above the notation staff starting within ROW_NUMBER_REACH notation spacings of the row
    start: where a row-start measure number is printed when the header stays as printed."""
    ys, xs = np.nonzero(labels == label)
    return ys.max() < notation['lines'][0] and xs.min() < x0 + ROW_NUMBER_REACH * notation['spacing']


def _box(labels, label, left):
    """[x0, y0, x1, y1] of a block, x in measure-image coordinates (the image starts at `left`)."""
    ys, xs = np.nonzero(labels == label)
    return [int(xs.min()) - left, int(ys.min()), int(xs.max()) + 1 - left, int(ys.max()) + 1]


def _number_owner(cuts, center, spacing):
    """Index of the measure starting nearest a measure number, or None if none is close."""
    k = min(range(len(cuts)), key=lambda k: abs(cuts[k][0] - center))
    return k if abs(cuts[k][0] - center) <= NUMBER_REACH * spacing else None


def _owner(cuts, center, left, right):
    """Index of the measure a block belongs to.

    A block that straddles a cut (a measure number or chord diagram printed over a barline)
    belongs to the measure starting there, unless under STRADDLE_RIGHT of its width is past
    the cut (e.g. a last-beat chord name just touching the barline). Otherwise: where its
    center is.
    """
    for k in range(1, len(cuts)):
        cut = cuts[k][0]
        if left < cut < right:
            return k if (right - cut) >= STRADDLE_RIGHT * (right - left) else k - 1
    return next((k for k, (x0, x1, _) in enumerate(cuts) if x0 <= center < x1),
                0 if center < cuts[0][0] else len(cuts) - 1)


def _ties_across(gray, bar, lines):
    """True if ink (not staff lines) touches both sides of the barline on the same rows."""
    left, right = bar['left'] - 3, bar['right'] + 1
    if left < 0 or right + 2 > gray.shape[1]:
        return False
    before = (gray[:, left:left + 2] < INK).any(axis=1)
    after = (gray[:, right:right + 2] < INK).any(axis=1)
    both = before & after
    both[:max(0, round(lines[0]) - 2)] = False
    both[round(lines[-1]) + 3:] = False
    for y in lines:
        both[max(0, round(y) - 2):round(y) + 3] = False
    return int(both.sum()) >= TIE_ROWS


def clear_edge_slivers(row):
    """Paint out a thin line the crop cut along its top or bottom edge. Edge band: the outer FRINGE
    pixel rows, or more where the border is thicker (rows from the edge that are ink over EDGE_LINE
    of the width, at most EDGE_ROWS; F's border is 3 rows). In it, a run of ink (down to PALE
    levels under the paper) at least EDGE_SLIVER spacings long, with mostly paper in the next row
    inward, is painted, unless it runs over FURNITURE of the width (then it is one page-furniture
    block, painted with the ink joined to it). clear_edge_rules only sees dark rows over most of
    the width; a pale panel border cut by the crop (G: gray 210-232 over 63-97% of row 0) is left,
    and as annotation blocks its pieces went whole to single measures (G m6: 470 px overhang).
    Returns the number of pixels painted."""
    gray, image = row['gray'], row['image_bgr']
    if gray.shape[0] <= 2 * (FRINGE + EDGE_ROWS):
        return 0
    paper_level = float(np.median(gray))
    ink = gray < paper_level - PALE
    paper = np.median(image[gray > 200], axis=0) if (gray > 200).any() else np.array([255, 255, 255])
    long = round(EDGE_SLIVER * row['spacing'])
    line = ink.mean(axis=1) >= EDGE_LINE
    height = gray.shape[0]
    deep = [next((n for n in range(EDGE_ROWS) if not line[y(n)]), EDGE_ROWS)
            for y in (lambda n: n, lambda n: height - 1 - n)]
    top, bottom = max(FRINGE, deep[0]), max(FRINGE, deep[1])
    if top + bottom >= height:
        return 0
    painted = 0
    for band, inward in ((range(top), top), (range(height - bottom, height), height - bottom - 1)):
        columns = np.zeros(gray.shape[1], bool)
        for y in band:
            on = np.flatnonzero(ink[y])
            for run in np.split(on, np.flatnonzero(np.diff(on) > 1) + 1) if len(on) else []:
                # a run over most of the width is one furniture block already (painted with what joins it)
                if long <= len(run) < FURNITURE * gray.shape[1] and ink[inward, run].mean() < .5:
                    columns[run] = True
        if columns.any():
            ys = np.array(list(band))[:, None]
            hit = np.zeros_like(ink)
            hit[ys, np.flatnonzero(columns)[None, :]] = ink[ys, np.flatnonzero(columns)[None, :]]
            # Ink joined to the sliver that stays within half a spacing of the edge goes with it;
            # anything reaching further in (a stem touching the border) keeps all but the band.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
            near = max(len(band) + 1, round(row['spacing'] / 2))
            for label in np.unique(labels[hit]):
                y0, h = stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_HEIGHT]
                if (y0 + h <= near) if band[0] == 0 else (y0 >= height - near):
                    hit |= labels == label
            painted += int(hit.sum())
            image[hit] = paper
            gray[hit] = int(round(float(np.mean(paper))))
    return painted


def edge_cut_ink(row):
    """Ink the crop cut along its top or bottom edge, left after the edge rules: [(edge, x0, x1)].
    A glyph touching the edge, wholly more than a spacing clear of the staves (and of a notation staff),
    with at least CUT_INK square spacings of ink and narrower than FURNITURE of the width: part of
    something printed outside the crop (blind test 5 video 3: the top of the next page's "Chorus"
    under row 5). Kept as printed and warned: a tight crop can also touch a real chord name."""
    gray = row['gray']
    height, width = gray.shape
    spacing = row['spacing']
    top = min(row['staff_lines'][0], *(row.get('notation_lines') or [row['staff_lines'][0]]))
    bottom = row['staff_lines'][-1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((gray < CUT_DARK).astype(np.uint8), connectivity=8)
    found = []
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        if area < CUT_INK * spacing ** 2 or w >= FURNITURE * width:
            continue
        if y == 0 and y + h < top - spacing:
            found.append(('top', int(x), int(x + w)))
        elif y + h == height and y > bottom + spacing:
            found.append(('bottom', int(x), int(x + w)))
    return found


def clear_edge_columns(row):
    """Paint out a panel's side border cut by the crop: columns at the left or right edge (at most
    EDGE_ROWS, contiguous from the edge) that are ink in FURNITURE of the rows above and below the
    staff band, and next to them its antialiased edge (ink in EDGE_LINE of those rows). A barline never runs on outside the staff, so it is never one. Single example
    (blind2 F: a 3 px gray border down the strip's right end went whole, above and below the
    staff, to the last measure as a 515 px overhang). Returns the number of columns painted."""
    gray, image = row['gray'], row['image_bgr']
    lines = row['staff_lines']
    outside = np.ones(gray.shape[0], bool)
    outside[max(0, round(lines[0]) - CORE_PAD):round(lines[-1]) + CORE_PAD + 1] = False
    if outside.sum() < row['spacing'] or gray.shape[1] <= 2 * EDGE_ROWS:
        return 0
    ink = gray[outside] < float(np.median(gray)) - PALE
    share = ink.mean(axis=0)
    paper = np.median(image[gray > 200], axis=0) if (gray > 200).any() else np.array([255, 255, 255])
    painted = 0
    for xs in (range(EDGE_ROWS), range(gray.shape[1] - 1, gray.shape[1] - 1 - EDGE_ROWS, -1)):
        xs = list(xs)
        core = next((n for n, x in enumerate(xs) if share[x] < FURNITURE), EDGE_ROWS)
        # then its paler antialiased edge: columns still ink in EDGE_LINE of those rows
        reach = next((n for n in range(core, EDGE_ROWS) if share[xs[n]] < EDGE_LINE), EDGE_ROWS) if core else 0
        run = xs[:reach]
        if run:
            hit = np.zeros(gray.shape, bool)
            hit[:, min(run):max(run) + 1] = True
            # ink joined to the border outside the staff band that stays within half a spacing of
            # the edge goes with it (where the top or bottom border met it: F's corner specks)
            loose = np.zeros(gray.shape, np.uint8)
            loose[outside] = ink
            count, labels, stats, _ = cv2.connectedComponentsWithStats(loose, connectivity=8)
            near = round(row['spacing'] / 2)
            for label in np.unique(labels[hit & (loose > 0)]):
                x0, w = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_WIDTH]
                if (x0 + w <= near) if xs[0] == 0 else (x0 >= gray.shape[1] - near):
                    hit |= labels == label
            image[hit] = paper
            gray[hit] = int(round(float(np.mean(paper))))
            painted += len(run)
    return painted


MARKER_FILL = .45     # page marker: a filled gray shape (circle, square, triangle, star...) at least this share of its box
MARKER_SIZE = (.5, 3)  # ...this many TAB spacings across, roughly square (aspect within MARKER_ASPECT)...
MARKER_ASPECT = 1.4
MARKER_PALE = 120      # ...with no pixel darker than this (gray, unlike numbers and noteheads)...
MARKER_REACH = 8       # ...within this many TAB spacings of the row start, above the staves...
MARKER_PLACE = 6       # ...at the same place (px) and size in MARKER_ROWS of the rows
MARKER_ROWS = .6


def _marker_candidates(row):
    """[(cx, cy, w, h, label mask)] of filled, square, uniformly gray blocks above the staves near the
    row start."""
    gray, spacing = row['gray'], row['spacing']
    top = round(min(row['staff_lines'][0], *(row.get('notation_lines') or [row['staff_lines'][0]])) - spacing / 2)
    if top <= 0:
        return []
    paper = float(np.median(gray))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((gray[:top] < paper - 40).astype(np.uint8), 8)
    out = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        piece = labels == i
        if (MARKER_SIZE[0] * spacing <= min(w, h) and max(w, h) <= MARKER_SIZE[1] * spacing
                and max(w, h) <= MARKER_ASPECT * min(w, h) and area >= MARKER_FILL * w * h
                and x + w / 2 - row['staff_start'] <= MARKER_REACH * spacing
                and int(gray[:top][piece].min()) >= MARKER_PALE):
            mask = np.zeros(gray.shape, bool)
            mask[:top] = piece
            out.append((x + w / 2, y + h / 2, w, h, mask))
    return out


def clear_page_markers(rows):
    """Page markers (blind 4 video 3: a gray circle, square, triangle... over each page's start, a
    different shape per page) are painted as paper: a filled, square, uniformly gray block near the
    row start, above the staves, at the same place and size in most rows (page-switching rows, 3 or
    more). Numbers and noteheads are dark, so never taken. Returns how many were painted."""
    if len(rows) < 3 or any(row.get('scroll') for row in rows):
        return 0
    found = [_marker_candidates(row) for row in rows]
    painted = 0
    for row, mine in zip(rows, found):
        for cx, cy, w, h, mask in mine:
            alike = sum(any(abs(cx - ox) <= MARKER_PLACE and abs(cy - oy) <= MARKER_PLACE
                            and abs(w - ow) <= MARKER_PLACE and abs(h - oh) <= MARKER_PLACE
                            for ox, oy, ow, oh, _ in other) for other in found)
            if alike < MARKER_ROWS * len(rows):
                continue
            grown = cv2.dilate(mask.astype(np.uint8), np.ones((2 * FRINGE + 1,) * 2, np.uint8)).astype(bool)
            gray, image = row['gray'], row['image_bgr']
            paper = np.median(image[gray > 200], axis=0) if (gray > 200).any() else np.array([255, 255, 255])
            image[grown] = paper
            gray[grown] = int(round(float(np.mean(paper))))
            painted += 1
    return painted


def build_measures(run_dir, report, layout='as played'):
    rows = json.loads((run_dir / 'rows.json').read_text())
    out_dir, debug_dir = run_dir / 'measures', run_dir / 'debug' / '4-cuts'
    for directory in (out_dir, debug_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob('*'):
            old.unlink()

    for row in rows:
        row['image_bgr'] = cv2.imread(str(run_dir / row['image']))
        row['gray'] = cv2.cvtColor(row['image_bgr'], cv2.COLOR_BGR2GRAY)
        row['staff_start'], row['staff_end'] = staff_extent(row['gray'], row['staff_lines'], row['spacing'])
    static = static_edge_ink(rows)             # before the edge rules go: they join this ink to the crop edge
    edge_rows = sum(clear_edge_rules(row) for row in rows)
    slivers_px = sum(clear_edge_slivers(row) for row in rows)
    edge_columns = sum(clear_edge_columns(row) for row in rows)
    for row in rows if static is not None else []:
        mask = static[row['row']]                  # painted here, before the staff core is found:
        gray, image = row['gray'], row['image_bgr']  # a bracket's foot can touch the staff's corner
        paper = np.median(image[gray > 200], axis=0) if (gray > 200).any() else np.array([255, 255, 255])
        image[mask] = paper
        gray[mask] = int(round(float(np.mean(paper))))
    markers = clear_page_markers(rows)
    cut_by_edge = [f"row {row['row']} {edge} x={x0}-{x1}" for row in rows for edge, x0, x1 in edge_cut_ink(row)]
    if cut_by_edge:
        report.warn('measures', f"ink touching the crop's top or bottom edge, away from the staff "
                    f"({'; '.join(cut_by_edge)}): something cut by the crop (another system's label) or page "
                    'furniture (a page number), kept as printed.')
    (run_dir / 'header.json').unlink(missing_ok=True)
    if rows and all(row.get('notation_lines') for row in rows):
        # Notation + TAB: the TAB clef is part of the system's header (brace, clefs, key).
        cleared, missing, clef_note = [], [], 'part of the notation header'
        header_note, warning = strip_headers(rows, run_dir)
        if warning:
            report.warn('measures', warning)
    else:
        header_note = None
        cleared, missing, clef_note = clear_clef(rows)
    if missing:
        report.warn('measures', f'TAB clef not recognised at the start of rows {missing}; left as it is.')

    records, moved_blocks, furniture, slivers = [], 0, 0, 0
    trailing_note = 'none'
    narrow = []
    opened = []
    for row in rows:
        image, gray = row['image_bgr'], row['gray']
        height, width = gray.shape
        lines, spacing = row['staff_lines'], row['spacing']
        band = (max(0, round(lines[0]) - CORE_PAD), min(height, round(lines[-1]) + CORE_PAD + 1))
        notation = {'lines': row['notation_lines'], 'spacing': row['notation_spacing']} \
            if row.get('notation_lines') else None
        core = system_core(gray, band, notation, spacing) if notation else staff_core(gray, band)
        paper = np.median(image[gray > 200], axis=0) if (gray > 200).any() else np.array([255, 255, 255])
        cuts = cut_row(row, width, row.get('cut_start', row['staff_start']))
        opening = len(cuts) > max(0, len(row['barlines']) - 1) and bool(row['barlines'])
        if opening:
            opened.append(row['row'])
        if not row['barlines']:
            report.warn('measures', f"row {row['row']}: no barlines to cut at; kept the whole row as one measure.")
        kept_clef = round(CLEF_MAX * spacing) if notation and 'cut_start' not in row else 0
        cuts, joined = join_slivers(cuts, gray, lines, spacing, opening, kept_clef)
        slivers += joined
        trailing, open_end = None, None
        if cuts and cuts[-1][2] and row['staff_end'] - cuts[-1][1] > TRAILING * spacing:
            from .structure import holds_notes, staff_ink
            x0, x1 = cuts[-1][1], row['staff_end']
            if holds_notes(staff_ink(gray, lines, x0, x1), spacing):
                # Music after the last barline is never dropped: it is a measure of its own (its closing
                # barline was not recognised, or the page is cut mid-measure), warned.
                open_end = len(cuts)
                cuts.append((x0, x1, None))
                edge = ' (to the crop edge)' if row['staff_end'] >= width - 2 else ''
                report.warn('measures', f"row {row['row']}: {x1 - x0}px of music after its last barline{edge}: kept "
                            'as a measure. Its closing barline was not recognised, or the page is cut mid-measure.')
            elif row is rows[-1]:
                # An empty measure at the very end, open at the crop edge: a cut of its own so its
                # number goes with it, then dropped (E printed "52" after its last barline).
                trailing = len(cuts)
                cuts.append((x0, x1, None))
                trailing_note = (f"row {row['row']}: the staff runs on {x1 - x0}px past the last barline with nothing "
                                 'but staff lines (an empty measure at the end); dropped')
        labels, centers, numbers = _blocks(gray, core, lines, spacing, notation and notation['lines'])
        staff_width = row['staff_end'] - row['staff_start']
        wide = {label for label, (_, l0, l1) in centers.items() if l1 - l0 > FURNITURE * staff_width}
        furniture += len(wide)
        centers = {label: extent for label, extent in centers.items() if label not in wide}
        numbers -= wide
        owner = {label: _owner(cuts, *extent) for label, extent in centers.items()}
        number_boxes = {}
        for label in numbers:   # a measure number goes with the measure it numbers
            k = _number_owner(cuts, centers[label][0], spacing)
            if k is not None:
                owner[label] = k
            ys, xs = np.nonzero(labels == label)
            number_boxes.setdefault(owner[label], []).append(
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        first = len(records) + 1
        shared = _diagram_owners(labels, gray, centers, owner, cuts, spacing, numbers)
        for label, k, others in shared:
            names = ', '.join(f'm{first + j}' for j in others)
            report.warn('measures', f'chord names over m{first + k} and {names} touch in the source, so they form '
                        f'one block: kept together with m{first + k}, which owns most of its diagrams; the chord '
                        f'name/diagram of {names} is printed in m{first + k}, across the barline.')
        keep_next, spanning = _spanning(centers, owner, cuts, numbers, spacing)
        median = float(np.median([x1 - x0 for x0, x1, _ in cuts]))
        extents = []
        outside = ~core
        ring = np.ones((2 * FRINGE + 1, 2 * FRINGE + 1), np.uint8)
        for k, (x0, x1, ending) in enumerate(cuts):
            if k == trailing:
                continue
            mine = [label for label, m in owner.items() if m == k]
            # The image includes each block's FRINGE (antialiased edge), so no glyph edge is trimmed.
            left = max(0, min([x0] + [centers[label][1] - FRINGE for label in mine]))
            right = min(width, max([x1] + [centers[label][2] + FRINGE for label in mine]))
            keep = np.zeros((height, width), bool)
            keep[:, x0:x1] = True                                               # staff, and ink touching it: straight cut
            # FRINGE px around each block covers its antialiased edge, which would otherwise be
            # left behind as a faint ghost of a neighbour's glyph.
            foreign = (labels > 0) & ~np.isin(labels, mine)
            theirs = cv2.dilate(foreign.astype(np.uint8), ring).astype(bool)
            ours = cv2.dilate(np.isin(labels, mine).astype(np.uint8), ring).astype(bool)
            keep &= ~(theirs & outside)                                         # ...minus other measures' blocks
            keep |= ours & outside                                              # plus its own blocks, whole
            keep &= ~(foreign & outside)                                        # a labelled pixel goes to its owner only
            measure = np.where(keep[..., None], image, paper.astype(np.uint8))[:, left:right]
            number = len(records) + 1
            name = f'm{number:03d}.png'
            cv2.imwrite(str(out_dir / name), measure)
            notes = []
            if left < x0 or right > x1:
                notes.append(f'annotation kept whole: overhangs staff by {x0 - left}px left, {right - x1}px right')
                moved_blocks += 1
            if x1 - x0 < NARROW * median:
                notes.append('narrow')
                narrow.append((name, median))
            if ending and k + 1 < len(cuts) and (_ties_across(gray, ending, lines) or (
                    notation and _ties_across(gray, ending, notation['lines']))):
                notes.append('tie/slur crosses the ending barline (split between measures)')
            records.append({'measure': number, 'row': row['row'], 'in_row': k + 1, 'image': f'measures/{name}',
                            'row_x0': left, 'row_x1': right, 'staff_x0': x0 - left, 'staff_x1': x1 - left,
                            'width': right - left, 'ending_strokes': len(ending['strokes']) if ending else 0,
                            'ending_final': bool(ending) and is_final(ending['strokes'], spacing),   # thin+thick
                            'ending_x0': (ending['left'] if ending else x1) - left,
                            'staff_lines': lines, 'spacing': spacing, 'notes': notes,
                            **({'notation_lines': notation['lines'], 'notation_spacing': notation['spacing']}
                               if notation else {}),
                            'number_boxes': [[bx0 - left, by0, bx1 - left, by1]
                                             for bx0, by0, bx1, by1 in sorted(number_boxes.get(k, []))],
                            **({'lead_blocks': [_box(labels, label, left)
                                                for label in mine if centers[label][2] <= x0]}
                               if k == 0 and 'cut_start' in row else {}),
                            **({'row_start_blocks': [_box(labels, label, left) for label in mine
                                                     if _row_start_block(labels, label, x0, notation)]}
                               if k == 0 and notation and 'cut_start' not in row else {}),
                            **({'open_end': True} if k == open_end else {}),
                            'keep_with_next': k in keep_next,
                            'spanning_blocks': [[l - left, r - left] for l, r in spanning.get(k, [])]})
            extents.append((left, right))
        _debug_cuts(debug_dir / f"row-{row['row']:02d}.jpg", image, cuts[:trailing], extents, band, first, row)

    if not records:
        report.warn('measures', 'No measures: no rows came out of stage 3. The PDF will be empty.')
        (run_dir / 'measures.json').write_text('[]\n')
        report.stage('measures', measures=0, rows=len(rows))
        return records
    # A missed barline merges two measures into one about twice as wide. Compare against the
    # whole video: a row has only a few measures, so its own median would include the wide one.
    # The number check runs first: a wide measure followed by the next printed number is one measure.
    from .numbers import safe_check
    seam = drop_seam_repeats(rows, records, run_dir, report)
    from . import replays
    replay_facts = replays.apply(rows, records, run_dir, report, layout,
                                 lambda rs: safe_check(rs, _Quiet(), run_dir))   # numbers only; real check next
    number_facts = safe_check(records, report, run_dir)
    proven = set(number_facts.get('proven_single', []))
    narrow_numbered = warn_narrow(narrow, records, proven, report)
    from . import structure
    structure.safe_check(rows, records, number_facts.get('sequence') == 'consistent', report)
    typical = float(np.median([r['staff_x1'] - r['staff_x0'] for r in records]))
    from .numbers import NUMBERED_VIDEO
    numbered = 'number_row' in number_facts and number_facts.get('numbered_share', 0) >= NUMBERED_VIDEO
    signed, width_only, wide_but_numbered, wide_quiet = wide_measures(records, rows, proven, numbered)
    for record, why in signed:
        record['notes'].append('wide: possible missing barline')
        report.warn('measures', f"measure {record['measure']} (row {record['row']}) is "
                    f"{record['staff_x1'] - record['staff_x0']}px wide vs a typical {typical:.0f}px and {why}: "
                    'a barline inside it was probably missed. Kept as one measure.')
    for record in width_only:
        record['notes'].append('wide: possible missing barline')
    if width_only:
        listed = ', '.join(f"{r['measure']} ({r['staff_x1'] - r['staff_x0']}px)" for r in width_only)
        report.warn('measures', f'wide measures: {listed}, over {WIDE}x the typical {typical:.0f}px and wide for '
                    'their row too, with no other sign of a missing barline. Probably long measures; a barline '
                    'may have been missed. Kept as they are.')
    last = records[-1]
    piece_end = ('final barline' if last['ending_final'] else 'double barline' if last['ending_strokes'] > 1
                 else 'single barline' if last['ending_strokes'] else 'no barline')
    (run_dir / 'measures.json').write_text(json.dumps(records, indent=1) + '\n')
    ties = [r['measure'] for r in records if any(n.startswith('tie') for n in r['notes'])]
    if ties:
        report.warn('measures', f'{len(ties)} ties/slurs cross a barline and are split between measures '
                    f'(after measures {ties}). Fine side by side; check them where a measure starts a new line.')
    report.stage('measures', measures=len(records), rows=len(rows),
                 double_or_final_endings=sum(r['ending_strokes'] > 1 for r in records),
                 piece_ending=piece_end,
                 trailing_empty_measure=trailing_note,
                 annotations_kept_whole_across_a_cut=moved_blocks,
                 narrow=[r['measure'] for r in records if 'narrow' in r['notes']] or 'none',
                 **({'narrow_but_numbered_in_sequence': narrow_numbered} if narrow_numbered else {}),
                 **({'strip_seam_repeats_dropped': seam} if seam else {}),
                 typical_width_px=round(typical),
                 possible_missing_barline=[r['measure'] for r in records if r['notes'] and any(
                     n.startswith('wide') for n in r['notes'])] or 'none',
                 wide_but_one_measure_by_its_numbers=wide_but_numbered or 'none',
                 wide_but_not_for_its_row_or_no_second_number=wide_quiet or 'none',
                 rows_without_opening_barline=opened or 'none',
                 page_furniture_blocks_removed=furniture,
                 empty_slivers_joined_to_the_next_measure=slivers,
                 crop_edge_rule_rows_painted=edge_rows, crop_edge_sliver_px=slivers_px,
                 crop_edge_columns_painted=edge_columns,
                 static_crop_edge_ink_px=int(next(iter(static.values())).sum()) if static is not None else 0,
                 tab_clef=clef_note or 'removed (blank staff) from rows ' + ', '.join(map(str, cleared)),
                 **({'notation_header': header_note} if header_note else {}),
                 **({'page_markers_painted': markers} if markers else {}),
                 **replay_facts,
                 dedupe=('none (page-switching video: repeats are real music)'
                         if not any(row.get('scroll') for row in rows) else
                         f"scrolling: strip seam repeats dropped: {'; '.join(seam)}" if seam else
                         'scrolling: none at the strip seams'))
    _measure_strip(run_dir, records)
    return records


def warn_narrow(narrow, records, proven, report):
    """Warn for the narrow measures ([(image name, row median)]), except the piece's first and last (a
    pickup, a short ending) and those whose own printed number is read and follows the previous one or
    leads to the next (`proven`: the numbers show m is one measure): a false barline splits a measure
    into pieces without a number of their own. Returns the narrow measures passed over for their numbers."""
    by_image = {r['image']: r for r in records}
    numbered = []
    for name, median in narrow:
        record = by_image.get(f'measures/{name}')
        if record is None or record['measure'] in (1, len(records)):
            continue
        if record['measure'] in proven or record['measure'] - 1 in proven:
            numbered.append(record['measure'])
            continue
        report.warn('measures', f"measure {record['measure']} (row {record['row']}) is "
                    f"{record['staff_x1'] - record['staff_x0']}px wide vs a row median of {median:.0f}px; check the "
                    'barlines around it. Kept.')
    return numbered


class _Quiet:
    """A report that keeps nothing: for a first reading of the measure numbers."""
    def stage(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass


def drop_seam_repeats(rows, records, run_dir, report, read=None):
    """Scrolling video, strip seams only (a join that found no overlap): when the new strip's first n
    measures repeat the previous strip's last n measure for measure (same width within SEAM_WIDTH
    spacings, staff ink within SEAM_MATCH, and the same printed numbers, all read), the repeats are
    dropped from the new strip, the measures renumbered and the drop noted. Anything less is kept
    (and the number check warns). Returns ['strip 2: m66-m69 = m62-m65 (numbers 62-65)', ...]."""
    if not any(row.get('scroll') for row in rows) or len({r['row'] for r in records}) < 2:
        return []
    from .numbers import safe_check
    from .structure import _mismatch, staff_ink
    (read or (lambda rs: safe_check(rs, _Quiet(), run_dir)))(records)   # numbers only; the real check runs later
    by_row = {}
    for record in records:
        by_row.setdefault(record['row'], []).append(record)
    rows_by_number = {row['row']: row for row in rows}

    def ink(record):
        row = rows_by_number[record['row']]
        return staff_ink(row['gray'], row['staff_lines'], record['row_x0'] + record['staff_x0'],
                         record['row_x0'] + record['staff_x1'])

    def same(a, b):
        if a.get('number') is None or a.get('number') != b.get('number'):
            return False
        wa, wb = a['staff_x1'] - a['staff_x0'], b['staff_x1'] - b['staff_x0']
        if abs(wa - wb) > SEAM_WIDTH * a['spacing']:
            return False
        ia, ib = ink(a), ink(b)
        w, h = min(ia.shape[1], ib.shape[1]), min(ia.shape[0], ib.shape[0])
        return _mismatch(ia[:h, -w:], ib[:h, -w:]) <= SEAM_MATCH     # aligned on the closing barline

    dropped, notes = set(), []
    order = sorted(by_row)
    for before, after in zip(order, order[1:]):
        old, new = by_row[before], by_row[after]
        for n in range(min(len(old), len(new)), 0, -1):
            if all(same(a, b) for a, b in zip(old[-n:], new[:n])):
                dropped.update(r['measure'] for r in new[:n])
                notes.append(f"strip {after}: m{new[0]['measure']}-m{new[n - 1]['measure']} = "
                             f"m{old[-n]['measure']}-m{old[-1]['measure']} (numbers {old[-n]['number']}-"
                             f"{old[-1]['number']})")
                break
    for r in records:
        r.pop('number', None), r.pop('number_score', None), r.pop('second_number', None)
    if not dropped:
        return []
    from .replays import drop_measures
    drop_measures(records, dropped, run_dir)
    for note in notes:
        report.warn('measures', f'{note}: the scroll join found no overlap there, but the new strip starts by '
                    'repeating the old one measure for measure, printed numbers included; the repeats were '
                    'dropped.')
    return notes


def _debug_cuts(path, image, cuts, extents, band, first, row):
    """The row with each measure's full extent tinted alternately; red = staff cut, orange = overhang."""
    shown = image.copy()
    tint = shown.copy()
    for k, (left, right) in enumerate(extents):
        color = (255, 220, 180) if k % 2 == 0 else (180, 255, 200)
        cv2.rectangle(tint, (left, 0), (right - 1, shown.shape[0] - 1), color, -1)
    shown = cv2.addWeighted(tint, .3, shown, .7, 0)
    for k, ((x0, x1, _), (left, right)) in enumerate(zip(cuts, extents)):
        for x in (x0, x1 - 1):
            cv2.line(shown, (x, band[0]), (x, band[1]), (0, 0, 255), 1)
        for x in {left, right - 1} - {x0, x1 - 1}:
            cv2.line(shown, (x, 0), (x, shown.shape[0]), (0, 140, 255), 1)
        cv2.putText(shown, f'm{first + k}', (x0 + 6, shown.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .9,
                    (0, 0, 200), 2, cv2.LINE_AA)
    header = np.full((36, shown.shape[1], 3), 255, np.uint8)
    cv2.putText(header, f"row {row['row']}: m{first}-m{first + len(cuts) - 1} | red = staff cut (ends on the barline's "
                "right edge) | orange = annotation overhang", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([header, shown]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def _measure_strip(run_dir, records, scale=.5, line_width=2400, gap=10):
    """debug/4-measures.jpg: every measure in order, separated by gray gaps, labelled m# (row r)."""
    tiles = []
    for record in records:
        image = cv2.imread(str(run_dir / record['image']))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        label = np.full((24, image.shape[1], 3), 255, np.uint8)
        flags = (' NARROW' if 'narrow' in record['notes'] else '') + \
            (' WIDE?' if any(n.startswith('wide') for n in record['notes']) else '')
        cv2.putText(label, f"m{record['measure']} (row {record['row']}){flags}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 255) if flags else (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([label, image]))
    height = max(t.shape[0] for t in tiles)
    line_width = max(line_width, max(t.shape[1] + gap for t in tiles))   # a measure wider than a line
    lines, current, width = [], [], 0
    for tile in tiles:
        tile = cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, gap, cv2.BORDER_CONSTANT, value=(150, 150, 150))
        if current and width + tile.shape[1] > line_width:
            lines.append(current)
            current, width = [], 0
        current.append(tile)
        width += tile.shape[1]
    lines.append(current)
    rendered = [cv2.copyMakeBorder(np.hstack(line), 0, gap, 0, line_width - sum(t.shape[1] for t in line),
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255)) for line in lines]
    cv2.imwrite(str(run_dir / 'debug' / '4-measures.jpg'), np.vstack(rendered), [cv2.IMWRITE_JPEG_QUALITY, 85])
