"""Stage 4b: check the measures found against the printed measure numbers. Never blocks the PDF.

Layer A (block level, no reading). Stage 4 hands every pale block near the staff to a measure
(`number_boxes`, measure-image coordinates). A printed measure number sits at the start of its
measure, over or just after the opening barline, so:

- the measure's number is a digit-shaped pale block centred within NUMBER_REACH spacings of
  the staff start (`number_box`); in a row's first measure, after the clef and time signature;
- the video's number row is the median height and vertical position of those blocks;
- another block of that height on that row, well inside the measure, is a second measure
  number: a barline between the two was probably missed;
- in a video where most measures are numbered, an unnumbered measure is a possible false
  barline (an arrow or stem accepted as one), or a pickup/count-in.

A number at the very end of a measure (over its ending barline) belongs to the next measure and
is ignored here (it is that measure's number when the next measure is in the same row).

Layer B (reading). Each number block is read by template matching against digit templates
(`tabvid/digits/<font>/<digit>.png`, harvested from runs whose count is known; a new font is new
templates, not code). One font is chosen per video (the one that reads the most blocks). Then
the printed numbers are compared with the measure order (see `sequence`). When numbers are read,
they replace layer A's guesses: a second block counts only if it reads as a number, and an
unnumbered measure is only a fact (the sequence shows whether a measure is extra).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

NUMBER_REACH = 1.5     # spacings from the staff start to the number's centre
FIRST_REACH = 7        # first measure of a row: the number follows the clef (<= 4) and time signature
MIN_HEIGHT = .3        # digit-shaped: height in spacings (a tie can clip part of a number)...
MAX_HEIGHT = 1.2
MAX_ASPECT = 3.5       # ...and at most this wide for its height (three digits)
ROW_TOLERANCE = .3     # spacings: same vertical position as the video's number row
HEIGHT_TOLERANCE = .25  # fraction of the typical number height
NUMBERED_VIDEO = .7    # share of numbered measures for unnumbered ones to be suspicious
MIN_NUMBERED = 3       # fewer numbered measures than this: the video prints no numbers
MIN_SYSTEM_NUMBERS = 2  # ...unless they are printed over the notation header, one per system (J: 2 pages)

DIGITS = Path(__file__).parent / 'digits'
GLYPH_H = 20           # templates and glyphs are compared at this height
OCCLUDED = 40          # gray levels darker than the video's number ink: a tie or line over the digit
SEEN = .6              # a glyph is compared only if this share of it is visible
WHOLE_FIT = 1.15       # a glyph that wide is still one digit if that digit is about as wide in the font
SPLIT = .95            # a glyph wider than this times its height is two touching digits
SHORT = .5             # glyphs whose ink spans less than this share of the window height are not digits
SPECK = .02            # glyphs with less ink than this times height^2 are specks (the dot of "sl.")
ON = .35               # ink level that counts as part of a glyph
ASPECT = 1.6           # glyph and template aspect ratios must agree within this factor
ACCEPT = .70           # best template score for a glyph...
MARGIN = .08           # ...and its lead over the next digit
UNRELIABLE = 5         # this many sequence problems, with under AGREE of the readings in step with the measure
AGREE = .25            # order, means the digits are misread (not that measures are wrong): one warning instead


def _digit_shaped(box, spacing):
    x0, y0, x1, y1 = box
    height = y1 - y0
    return MIN_HEIGHT * spacing <= height <= MAX_HEIGHT * spacing and x1 - x0 <= MAX_ASPECT * height


def _matches(box, top, spacing, typical, strict=True):
    """On the video's number row (compared at the box top: a tie arc can clip a number's bottom)
    and of its size (strict), or not taller than it (lenient)."""
    offset, height = typical
    tall = (box[3] - box[1]) / spacing
    same_row = abs((top - box[1]) / spacing - offset - height) <= ROW_TOLERANCE
    same_size = abs(tall - height) <= HEIGHT_TOLERANCE * height if strict else tall <= (1 + HEIGHT_TOLERANCE) * height
    return same_row and same_size


def find_numbers(records):
    """Set number_box on each record (or None); return the typical (bottom offset, height) or None.

    A measure's number is the digit-shaped pale block centred nearest its staff start, within
    NUMBER_REACH spacings. The first measure of a row is different: its number is printed after
    the TAB clef and any time signature, so there it is the leftmost block on the video's number
    row (learned from the other measures) within FIRST_REACH spacings. So is a later measure with
    no block at its start (it opens with a time signature), if the block has the numbers' size.
    """
    found = []
    for r in records:
        r['number_box'] = None
        if r['in_row'] == 1:
            continue
        spacing, top = r['spacing'], r['staff_lines'][0]
        near = [b for b in r.get('number_boxes', []) if _digit_shaped(b, spacing)
                and abs((b[0] + b[2]) / 2 - r['staff_x0']) <= NUMBER_REACH * spacing]
        if near:
            r['number_box'] = min(near, key=lambda b: abs((b[0] + b[2]) / 2 - r['staff_x0']))
            found.append(((top - r['number_box'][3]) / spacing, (r['number_box'][3] - r['number_box'][1]) / spacing))
    typical = tuple(float(v) for v in np.median(found, axis=0)) if found else None
    for r in records:
        if r['number_box'] or (r['in_row'] != 1 and typical is None):
            continue          # a later measure with no number at its start: printed after a time signature
        spacing, top = r['spacing'], r['staff_lines'][0]
        near = [b for b in r.get('number_boxes', []) if _digit_shaped(b, spacing)
                and -NUMBER_REACH * spacing <= (b[0] + b[2]) / 2 - r['staff_x0'] <= FIRST_REACH * spacing
                and (typical is None or _matches(b, top, spacing, typical, strict=r['in_row'] != 1))]
        if near:
            r['number_box'] = min(near, key=lambda b: b[0])
    for r in records:
        lead = r.get('lead_blocks') or r.get('row_start_blocks')
        if r['number_box'] is None and lead:
            boxes = [b for b in _merged(lead) if _digit_shaped(b, r['spacing'])]
            if boxes:                     # notation: the number printed over the header at the system start
                r['number_box'], r['system_number'] = min(boxes, key=lambda b: b[0]), True
    numbered = [r for r in records if r['number_box'] is not None]
    if len(numbered) < (MIN_SYSTEM_NUMBERS if numbered and all(r.get('system_number') for r in numbered)
                        else MIN_NUMBERED):
        return None
    if typical is None and all(r.get('system_number') for r in numbered):
        # notation, only system starts numbered: learn the number row from them
        typical = tuple(float(v) for v in np.median(
            [((r['staff_lines'][0] - r['number_box'][3]) / r['spacing'],
              (r['number_box'][3] - r['number_box'][1]) / r['spacing']) for r in numbered], axis=0))
    return typical


def _merged(boxes):
    """Overlapping boxes joined (a dark digit and its pale halo are two blocks)."""
    out = []
    for box in sorted(boxes):
        for i, other in enumerate(out):
            if box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]:
                out[i] = [min(box[0], other[0]), min(box[1], other[1]), max(box[2], other[2]), max(box[3], other[3])]
                break
        else:
            out.append(list(box))
    return out


def load_fonts(root=DIGITS):
    """{font: {digit: template (GLYPH_H rows, ink 0..1)}} from tabvid/digits/<font>/<digit>.png."""
    fonts = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        tpl = {int(f.stem): 1 - cv2.imread(str(f), 0).astype(np.float32) / 255 for f in folder.glob('[0-9].png')}
        if tpl:
            fonts[folder.name] = tpl
    return fonts


def video_ink(grays, boxes):
    """(paper, number ink, digit height) of a video, from its measures' number boxes."""
    paper = float(np.median([np.median(g) for g in grays]))
    ink = float(np.median([np.percentile(g[b[1]:b[3], b[0]:b[2]], 3) for g, b in zip(grays, boxes)]))
    height = int(np.percentile([b[3] - b[1] for b in boxes], 80))    # boxes clipped by a tie are shorter
    return paper, ink, height


def ink_window(gray, box, paper, ink, height):
    """Ink 0..1 over the box's columns, `height` rows down from its top (a tie can clip a number's
    box, never its top), and where that is known: not off the image, not under a darker stroke."""
    x0, y0, x1 = box[0], box[1], box[2]
    win = np.full((height, x1 - x0), paper, float)
    known = np.zeros(win.shape, bool)
    part = gray[y0:y0 + height, x0:x1].astype(float)
    win[:part.shape[0]], known[:part.shape[0]] = part, True
    covered = cv2.dilate((win < ink - OCCLUDED).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    known &= ~covered
    v = np.clip((paper - win) / max(1.0, paper - ink), 0, 1)
    v[~known] = 0
    return v, known


def glyphs(v, known, split=True):
    """Split a number window into glyphs at columns without ink: [(ink, known)] per glyph. With
    `split`, a glyph wider than SPLIT times the height is cut in two (touching digits)."""
    on = v > ON
    cols = np.concatenate([[False], on.any(axis=0), [False]])
    edges = np.flatnonzero(np.diff(cols.astype(int)))
    h, out = v.shape[0], []
    for s, e in zip(edges[::2], edges[1::2]):
        if on[:, s:e].sum() < SPECK * h * h:
            continue
        inked = np.flatnonzero(on[:, s:e].any(axis=1))
        if inked[-1] - inked[0] + 1 < SHORT * h:
            continue                            # not a digit: a brace's tip touching the number (blind 4 v1)
        if split and e - s > SPLIT * h:         # two digits touching: cut at the weakest middle column
            profile = v[:, s:e].sum(axis=0)
            third = (e - s) // 3
            cut = s + third + int(np.argmin(profile[third:e - s - third]))
            out += [(s, cut), (cut, e)]
        else:
            out.append((s, e))
    return [(v[:, s:e], known[:, s:e]) for s, e in out]


def _resize(a, width):
    return cv2.resize(a.astype(np.float32), (max(1, width), GLYPH_H), interpolation=cv2.INTER_AREA)


def glyph_score(glyph, template):
    """Normalised cross-correlation over the visible part, at the template's size; 0 when the
    aspect ratios disagree or too little of the glyph is visible."""
    v, known = glyph
    g_aspect, t_aspect = v.shape[1] / v.shape[0], template.shape[1] / GLYPH_H
    if max(g_aspect, t_aspect) / max(min(g_aspect, t_aspect), 1e-6) > ASPECT:
        return 0.0
    seen = _resize(known, template.shape[1]) > .99
    if seen.mean() < SEEN:
        return 0.0
    a, b = _resize(v, template.shape[1])[seen], template[seen]
    a, b = a - a.mean(), b - b.mean()
    norm = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / norm) if norm else 0.0


def read_window(v, known, font):
    """(number, score) read with one font's templates, or (None, best score) when any glyph is
    not clearly one digit or the block is not 1-3 glyphs (an "sl." mark, a tie end). A wide glyph
    is read as two touching digits; if that fails, as one (an italic 6 as wide as the window is
    high, its soft edges included: blind 4 video 2)."""
    number, score = _read_parts(glyphs(v, known), font)
    if number is None:
        whole = glyphs(v, known, split=False)
        if len(whole) != len(glyphs(v, known)):
            number_whole, score_whole = _read_parts(whole, font)
            # ...only if each glyph is as wide as its digit in the font (touching "33" is no wide "0")
            if number_whole is not None and all(
                    g[0].shape[1] * GLYPH_H / g[0].shape[0] <= WHOLE_FIT * font[int(d)].shape[1]
                    for g, d in zip(whole, str(number_whole))):
                return number_whole, score_whole
    return number, score


def _read_parts(parts, font):
    if not 1 <= len(parts) <= 3:
        return None, 0.0
    digits, worst = '', 1.0
    for glyph in parts:
        ranked = sorted(((glyph_score(glyph, t), d) for d, t in font.items()), reverse=True)
        if ranked[0][0] < ACCEPT or ranked[0][0] - ranked[1][0] < MARGIN:
            return None, round(ranked[0][0], 2)
        digits += str(ranked[0][1])
        worst = min(worst, ranked[0][0])
    return int(digits), round(worst, 2)


def read_numbers(records, run_dir, fonts):
    """Set `number`, `number_score` on each record (and `second_number` for a second block that
    reads as a number). Returns (font name or None, [(record, box, reading, score)] for the debug)."""
    for r in records:
        r['number'], r['number_score'] = None, None
    numbered = [r for r in records if r.get('number_box')]
    if not fonts or len(numbered) < (MIN_SYSTEM_NUMBERS if all(r.get('system_number') for r in numbered)
                                     else MIN_NUMBERED):
        return None, []
    gray = {r['measure']: cv2.imread(str(run_dir / r['image']), cv2.IMREAD_GRAYSCALE) for r in records}
    paper, ink, height = video_ink([gray[r['measure']] for r in numbered], [r['number_box'] for r in numbered])
    windows = {r['measure']: ink_window(gray[r['measure']], r['number_box'], paper, ink, height) for r in numbered}
    # One font per video: the one that reads the most blocks (then the higher total score).
    tallies = {}
    for name, font in fonts.items():
        reads = [read_window(*w, font) for w in windows.values()]
        tallies[name] = (sum(n is not None for n, _ in reads), sum(s for n, s in reads if n is not None))
    font = max(tallies, key=tallies.get)
    shown = []
    for r in numbered:
        r['number'], r['number_score'] = read_window(*windows[r['measure']], fonts[font])
        shown.append((r, r['number_box'], r['number'], r['number_score']))
    for r in records:
        for b in r.get('extra_blocks', []):
            n, score = read_window(*ink_window(gray[r['measure']], b, paper, ink, height), fonts[font])
            shown.append((r, b, n, score))
            if n is not None:
                r['second_number'] = n
    return font, shown


def sequence(records):
    """Compare the printed numbers with the measure order. Returns (misreads, problems, proven):
    misreads = [(measure, read)] single readings that disagree with two agreeing neighbours;
    problems = [message] where the numbers say measures are missing, extra or repeated;
    proven = measures whose next measure carries the next number (so they are one measure)."""
    reads = [(i, r['number'], r) for i, r in enumerate(records) if r.get('number') is not None]
    misreads, kept = [], []
    for k, (i, n, r) in enumerate(reads):
        if 0 < k < len(reads) - 1:
            (pi, pn, _), (ni, nn, _) = reads[k - 1], reads[k + 1]
            if pn - pi == nn - ni != n - i:
                misreads.append((r['measure'], n))
                continue
        kept.append((i, n, r))
    problems, proven = [], set()
    for (i1, n1, r1), (i2, n2, r2) in zip(kept, kept[1:]):
        steps, advance = i2 - i1, n2 - n1
        where = f"m{r1['measure']} (number {n1}) and m{r2['measure']} (number {n2})"
        if advance == steps:
            if steps == 1:
                proven.add(r1['measure'])
            continue
        if advance <= 0:
            problems.append(f'numbers go back from {n1} to {n2} between {where}: a repeated page or strip '
                            '(or the music repeats with its numbers reprinted).')
        elif advance > steps:
            problems.append(f'{advance - steps} measure(s) missing between {where}: a barline missed '
                            '(a measure holding two), or frames/pages dropped.')
        else:
            problems.append(f'{steps - advance} measure(s) too many between {where}: a false barline (strum '
                            'arrow, stem) split a measure, or a measure was duplicated.')
    return misreads, problems, proven


def check(records, report, run_dir=None, fonts=None):
    """Layer A and B checks; facts and warnings go to the report. Returns the facts (with
    `proven_single`: measures the numbers show are one measure each)."""
    typical = find_numbers(records)
    numbered = [r for r in records if r['number_box']]
    facts = {'measures_found': len(records), 'numbered_measures': len(numbered),
             'numbered_share': round(len(numbered) / len(records), 2) if records else 0}
    if typical is None or len(numbered) < (MIN_SYSTEM_NUMBERS if all(r.get('system_number') for r in numbered)
                                           else MIN_NUMBERED):
        facts['number_check'] = 'no printed measure numbers found; count check skipped'
        report.stage('numbers', **facts)
        return facts
    offset, height = typical or (0, 0)
    facts['number_row'] = f'{offset:.2f} spacings above the staff, {height:.2f} spacings tall'

    for r in records:
        r['extra_blocks'] = []
        spacing, top = r['spacing'], r['staff_lines'][0]
        for b in r.get('number_boxes', []):
            if b == r['number_box'] or not _digit_shaped(b, spacing) or (
                    r['number_box'] and b[0] < r['number_box'][0]):
                continue
            centre = (b[0] + b[2]) / 2
            if r['staff_x0'] + NUMBER_REACH * spacing < centre < r['staff_x1'] - NUMBER_REACH * spacing \
                    and _matches(b, top, spacing, typical):
                r['extra_blocks'].append(b)

    font, shown = (None, [])
    if run_dir is not None:
        font, shown = read_numbers(records, Path(run_dir), load_fonts() if fonts is None else fonts)
    read = [r for r in records if r.get('number') is not None]
    least = MIN_SYSTEM_NUMBERS if read and all(r.get('system_number') for r in read) else MIN_NUMBERED
    reading = font is not None and len(read) >= least

    doubled = []
    for r in records:
        if not r['extra_blocks'] or (reading and 'second_number' not in r):
            continue          # with reading on, a second block must read as a number to count
        doubled.append(r['measure'])
        b = r['extra_blocks'][0]
        what = f"number {r['second_number']}" if 'second_number' in r else 'measure number'
        report.warn('numbers', f"measure {r['measure']} (row {r['row']}) holds a second {what} "
                    f"{round((b[0] + b[2]) / 2 - r['staff_x0'])}px after its start: a barline between them was "
                    'probably missed.')
    facts['measures_with_two_numbers'] = doubled or 'none'

    unnumbered = [r['measure'] for r in records if not r['number_box']]
    facts['unnumbered_measures'] = unnumbered or 'none'
    if not reading:
        if unnumbered and len(numbered) >= NUMBERED_VIDEO * len(records):
            report.warn('numbers', f'measures {unnumbered} have no printed number while '
                        f'{len(numbered)}/{len(records)} do: a false barline (strum arrow, stem) may have split '
                        'a measure there, or it is a pickup/count-in.')
        if run_dir is not None:
            facts['numbers_read'] = f'{len(read)} (no digit templates matched; reading skipped)'
        report.stage('numbers', **facts)
        return facts

    misreads, problems, proven = sequence(records)
    values = [r['number'] for r in read]
    facts.update(font=font, numbers_read=f'{len(read)} of {len(numbered)} numbered measures '
                 f'({values[0]}..{values[-1]} in playback order)',
                 numbers_unread=[r['measure'] for r in numbered if r.get('number') is None] or 'none',
                 misreads=[f'm{m} read {n}' for m, n in misreads] or 'none',
                 printed_last=max(values), sequence='consistent' if not problems else f'{len(problems)} problem(s)')
    pairs = len(read) - len(misreads) - 1
    if len(problems) >= UNRELIABLE and pairs - len(problems) < AGREE * pairs:
        # Readings that almost never follow the measure order are misreadings (blind test 5 video 1: black
        # numbers seen only by their pale fringe, read as 1 or 3): one warning, nothing proven by them.
        facts['sequence'] = f'not used: {len(problems)} problem(s), {pairs - len(problems)} of {pairs} steps in order'
        report.warn('numbers', f'measure numbers could not be read reliably ({len(read)} of {len(numbered)} read, '
                    f'only {pairs - len(problems)} of {pairs} steps between readings follow the measure order): the '
                    'count was not checked against the printed numbers.')
        problems, proven = [], set()
    for message in problems:
        report.warn('numbers', message)
    facts['proven_single'] = sorted(proven)
    report.stage('numbers', **{k: v for k, v in facts.items() if k != 'proven_single'})
    if run_dir is not None and shown:
        _debug_numbers(Path(run_dir) / 'debug' / '4-numbers.jpg', shown)
    return facts


def _debug_numbers(path, shown):
    """Every number block, 3x, with its reading and score; red when unread."""
    tiles = []
    for r, b, n, score in shown:
        img = cv2.imread(str(path.parent.parent / r['image']))
        x0, y0, x1, y1 = b
        crop = img[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        tile = np.full((crop.shape[0] + 26, max(crop.shape[1], 130), 3), 255, np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        colour = (0, 0, 200) if n is None else (0, 120, 0)
        cv2.putText(tile, f"m{r['measure']}: {'?' if n is None else n} {score or 0:.2f}", (2, tile.shape[0] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)
        tiles.append(tile)
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1] + 6, cv2.BORDER_CONSTANT,
                                value=(255, 255, 255)) for t in tiles]
    per_row = max(1, 1800 // (w + 6))
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def safe_check(records, report, run_dir=None):
    """check(), but any failure is only a warning: the number check never stops a run."""
    try:
        return check(records, report, run_dir)
    except Exception as error:
        report.warn('numbers', f'measure-number check failed ({type(error).__name__}: {error}); skipped.')
        return {}
    finally:
        for r in records:
            r.pop('extra_blocks', None)


def _harvest_source(arg):
    """'RUN_DIR[:OFFSET[:FIRST-LAST]]' -> (run dir, offset, first, last): measures FIRST..LAST print
    their index plus OFFSET (an unnumbered pickup: -1; a page shown again restarts the count)."""
    parts = str(arg).split(':')
    offset = int(parts[1]) if len(parts) > 1 else 0
    first, last = (int(v) for v in parts[2].split('-')) if len(parts) > 2 else (1, 10 ** 6)
    return Path(parts[0]), offset, first, last


def harvest(font, run_dirs, out=DIGITS):
    """Write digit templates for `font` from runs whose measure k is printed as number k + OFFSET
    (counts confirmed by eye; see _harvest_source). Only blocks with nothing over them that split into
    as many glyphs as the number has digits are used; each digit's template is their mean at its
    median width."""
    per = {}
    for run_dir, offset, first, last in map(_harvest_source, run_dirs):
        records = json.loads((run_dir / 'measures.json').read_text())
        numbered = [r for r in records if r.get('number_box') and first <= r['measure'] <= last]
        gray = {r['measure']: cv2.imread(str(run_dir / r['image']), cv2.IMREAD_GRAYSCALE) for r in numbered}
        paper, ink, height = video_ink([gray[r['measure']] for r in numbered], [r['number_box'] for r in numbered])
        for r in numbered:
            printed = str(r['measure'] + offset)
            v, known = ink_window(gray[r['measure']], r['number_box'], paper, ink, height)
            parts = glyphs(v, known)
            if known.all() and len(parts) == len(printed):
                for d, (g, _) in zip(printed, parts):
                    per.setdefault(int(d), []).append(g)
    folder = out / font
    folder.mkdir(parents=True, exist_ok=True)
    for d, gs in sorted(per.items()):
        width = int(np.median([round(g.shape[1] * GLYPH_H / g.shape[0]) for g in gs]))
        template = np.mean([_resize(g, width) for g in gs], axis=0)
        cv2.imwrite(str(folder / f'{d}.png'), np.round(255 * (1 - template)).astype(np.uint8))
    return {d: len(gs) for d, gs in sorted(per.items())}


if __name__ == '__main__':
    # python -m tabvid.numbers harvest FONT RUN_DIR[:OFFSET[:FIRST-LAST]] ...
    if len(sys.argv) < 4 or sys.argv[1] != 'harvest':
        sys.exit('usage: python -m tabvid.numbers harvest FONT RUN_DIR[:OFFSET[:FIRST-LAST]] ...')
    print(harvest(sys.argv[2], sys.argv[3:]))
