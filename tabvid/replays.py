"""Stage 4d: replays, for a sheet "as written" (the printed score) or "as played" (the video's playback).

A page-switching video often shows a page again later: a repeat, a D.C. or D.S. jump, a Coda. As played
(the default) keeps every page as the video shows it. As written prints a replayed page only once.

A row replays an earlier row when
- it holds as many measures, and its black ink (dark, roughly gray; staff-line rows left out, 1 px
  tolerance) differs from the earlier row's in at most ROW_MATCH of it, with no difference the size of
  a digit (BLOB: colour fringes along a thick barline or the crop edge are thin strips), and
- its printed measure numbers are found and match the earlier row's: the same measures carry a number,
  each number's ink is the same (NUMBER_MATCH), at least one is read in both rows, and where both are
  read they read the same. Numbers found but none read: not confirmed, kept (warned).
A replayed row with no number of its own (the opening page: measure 1 has none) is dropped only when it is
the opening row again, or when the row right after it is a dropped replay of the row after its original.
Rows that match closely (up to PARTIAL) with matching numbers, but not exactly, are kept and warned.
Written-out repeats (same music, other numbers) never match. A video without printed numbers (or whose
numbers are not read, or read too unreliably for the number check to use them), or a scrolling video, is
always laid out as played (said in the report).

As played, the replays as written would drop are still found and noted in the report, so the reader can
choose 'as written' when the score prints the repeats or jumps (the tool cannot read jump markings).
The choice is saved per run (config.json "layout") and reused on re-runs.
"""
import json
from pathlib import Path

import cv2
import numpy as np

AS_PLAYED, AS_WRITTEN = 'as played', 'as written'
DARK = 140          # black ink: darker than this...
NEUTRAL = 60        # ...and roughly gray (max - min channel under this): a coloured highlight is not ink
LINE_ROWS = 2       # px each side of a staff line left out (its gray sits at the DARK edge)
ROW_MATCH = .01     # share of the two rows' ink that differs: an exact replay
PARTIAL = .06       # ...up to this, with matching numbers: a partial replay (kept, warned)
BLOB = .3           # spacings: a difference this wide and this tall (a digit, a notehead) is a real one
NUMBER_MATCH = .1   # share of a number's ink that may differ
NUMBER_PAD = 2      # px around a number box compared


def black_ink(image, lines=()):
    """Dark, roughly gray pixels, with the rows of the given staff lines left out."""
    image = image.astype(np.int16)
    ink = (image.mean(axis=2) < DARK) & (image.max(axis=2) - image.min(axis=2) < NEUTRAL)
    for y in lines:
        y = int(round(y))
        ink[max(0, y - LINE_ROWS):y + LINE_ROWS + 1] = False
    return ink


def mismatch(a, b):
    """Share of the ink of a and b that has no ink of the other within 1 px."""
    total = int(a.sum()) + int(b.sum())
    if not total:
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    near_a, near_b = (cv2.dilate(x.astype(np.uint8), kernel) > 0 for x in (a, b))
    return float((a & ~near_b).sum() + (b & ~near_a).sum()) / total


def digit_blobs(a, b, spacing):
    """How many differences between ink a and b (ink of one with no ink of the other within 1 px,
    joined over 1 px, and up and down over the staff-line rows left out: a fret digit sits on a line) are
    at least BLOB spacings wide and tall."""
    kernel = np.ones((3, 3), np.uint8)
    near_a, near_b = (cv2.dilate(x.astype(np.uint8), kernel) > 0 for x in (a, b))
    differ = cv2.dilate(((a & ~near_b) | (b & ~near_a)).astype(np.uint8), kernel)
    differ = cv2.dilate(differ, np.ones((2 * LINE_ROWS + 1, 1), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(differ, connectivity=8)
    wide = BLOB * spacing + 2                          # the joining dilation adds a pixel each side...
    tall = wide + 2 * LINE_ROWS                        # ...and the bridge LINE_ROWS above and below
    return int(sum(w >= wide and h >= tall for _, _, w, h, _ in stats[1:]))


def _numbers_match(old, new, run_dir):
    """'none' (neither row prints a number), 'match' (the same measures carry one, each with the same ink
    and, if both read, the same value; at least one read in both), 'unread' (the same ink, none read in
    both) or 'differ'."""
    pairs = [(a, b) for a, b in zip(old, new) if a.get('number_box') or b.get('number_box')]
    if not pairs:
        return 'none'
    read = False
    for a, b in pairs:
        if not (a.get('number_box') and b.get('number_box')):
            return 'differ'
        if a.get('number') is not None and b.get('number') is not None:
            if a['number'] != b['number']:
                return 'differ'
            read = True
        box = [min(a['number_box'][0], b['number_box'][0]) - NUMBER_PAD,
               min(a['number_box'][1], b['number_box'][1]) - NUMBER_PAD,
               max(a['number_box'][2], b['number_box'][2]) + NUMBER_PAD,
               max(a['number_box'][3], b['number_box'][3]) + NUMBER_PAD]
        inks = []
        for r in (a, b):
            image = cv2.imread(str(Path(run_dir) / r['image']))
            x0, y0 = max(0, box[0]), max(0, box[1])
            inks.append(black_ink(image[y0:box[3], x0:box[2]]))
        h, w = min(i.shape[0] for i in inks), min(i.shape[1] for i in inks)
        if mismatch(inks[0][:h, :w], inks[1][:h, :w]) > NUMBER_MATCH:
            return 'differ'
    return 'match' if read else 'unread'


def choose_layout(run_dir, given=None):
    """The run's layout: `given` (--as-written / --as-played) is saved in config.json ("layout") and used;
    with neither, the saved one, else as played. config.json is written only when the choice changes."""
    config_path = Path(run_dir) / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    saved = config.get('layout', AS_PLAYED)
    if given is None or given == saved:
        return saved
    config_path.write_text(json.dumps({**config, 'layout': given}, indent=1) + '\n')
    return given


def find(rows, records, run_dir, numbered_video):
    """Find the replays. Returns {'drop': [(row, original row)], 'partial': [(row, original, mismatch)],
    'unnumbered': [(row, original)] exact repeats with no numbers to confirm them (not dropped),
    'unread': [(row, original)] exact repeats whose numbers are found but not read (not dropped),
    'other_numbers': [(row, original)] same music, other printed numbers (written-out repeats)}."""
    found = {'drop': [], 'partial': [], 'unnumbered': [], 'unread': [], 'other_numbers': []}
    by_row = {}
    for r in records:
        by_row.setdefault(r['row'], []).append(r)
    rows = [row for row in rows if by_row.get(row['row'])]
    if len(rows) < 2:
        return found
    inks = {}
    for row in rows:
        image = cv2.imread(str(Path(run_dir) / row['image']))
        inks[row['row']] = black_ink(image, list(row['staff_lines']) + list(row.get('notation_lines') or []))
    first = rows[0]['row']
    exact, original = {}, {}
    for j, row in enumerate(rows):
        best = None
        for earlier in rows[:j]:
            a, b = inks[earlier['row']], inks[row['row']]
            if a.shape != b.shape or len(by_row[earlier['row']]) != len(by_row[row['row']]):
                continue
            share = mismatch(a, b)
            if share > PARTIAL:
                continue
            same = share <= ROW_MATCH and not digit_blobs(a, b, row['spacing'])
            if best is None or (same, -share) > (best[0], -best[1]):
                best = (same, share, earlier['row'])
        if best is None:
            continue
        same, share, old = best
        old = original.get(old, old)              # a replay of a replay: name the first showing
        numbers = _numbers_match(by_row[old], by_row[row['row']], run_dir) if numbered_video else 'none'
        if not same:
            if numbers == 'match':
                found['partial'].append((row['row'], old, round(share, 3)))
        elif numbers == 'differ':
            found['other_numbers'].append((row['row'], old))
        elif numbers == 'unread':
            found['unread'].append((row['row'], old))
            original[row['row']] = old
        else:
            exact[row['row']] = (old, numbers == 'match')
            original[row['row']] = old
    order = [row['row'] for row in rows]
    dropped = {}
    for k in range(len(order) - 1, -1, -1):        # backwards: an unnumbered row leans on the one after it
        name = order[k]
        if name not in exact:
            continue
        old, numbered = exact[name]
        if numbered:
            ok = numbered_video
        else:
            after = order[k + 1] if k + 1 < len(order) else None
            ok = numbered_video and (old == first or (after in dropped and order.index(dropped[after]) ==
                                                       order.index(old) + 1))
        if ok:
            dropped[name] = old
        elif not numbered:
            found['unnumbered'].append((name, old))
    found['drop'] = sorted(dropped.items())
    found['unnumbered'].sort()
    return found


def _groups(pairs):
    """[(9, 1), (10, 2), (11, 8)] -> [((9, 10), (1, 2)), ((11, 11), (8, 8))]: runs of consecutive rows
    replaying consecutive rows."""
    groups = []
    for row, old in pairs:
        if groups and row == groups[-1][1] + 1 and old == groups[-1][3] + 1:
            groups[-1][1], groups[-1][3] = row, old
        else:
            groups.append([row, row, old, old])
    return [((a, b), (c, d)) for a, b, c, d in groups]


def _span(a, b, word='row'):
    return f'{word} {a}' if a == b else f'{word}s {a}-{b}'


def describe(pairs, rows, by_row):
    """'rows 9-10 (86.5-104.5s, m35-m43) = rows 1-2; row 11 (105.0-108.0s, m44-m48) = row 8'."""
    seconds = {row['row']: row['seconds'] for row in rows}
    out = []
    for (a, b), (c, d) in _groups(pairs):
        measures = [r['measure'] for n in range(a, b + 1) for r in by_row.get(n, [])]
        out.append(f"{_span(a, b)} ({seconds[a][0]}-{seconds[b][1]}s, m{measures[0]}-m{measures[-1]}) = "
                   f"{_span(c, d)}")
    return '; '.join(out)


def apply(rows, records, run_dir, report, layout, numbers_read):
    """Find replays and, as written, drop them (measures renumbered, images renamed). `numbers_read`
    reads the printed numbers into the records (number_box, number) and returns the number facts.
    Returns the facts for the report. Saves replays.json (see save) in every case."""
    save(run_dir, layout, [], [])
    if any(row.get('scroll') for row in rows):
        if layout == AS_WRITTEN:
            report.warn('measures', 'as written: a scrolling video is always laid out as played (repeated '
                        'pages are only recognised in page-switching videos).')
        return {'layout': layout, 'replays': 'not looked for (scrolling video)'}
    number_facts = numbers_read(records)
    numbered_video = ('font' in number_facts and not str(number_facts.get('sequence', '')).startswith('not used'))
    found = find(rows, records, run_dir, numbered_video)
    for r in records:
        r.pop('number', None), r.pop('number_score', None), r.pop('second_number', None)
    by_row = {}
    for r in records:
        by_row.setdefault(r['row'], []).append(r)
    facts = {'layout': layout}
    save(run_dir, layout, found['drop'] if layout == AS_WRITTEN and numbered_video else [], found['drop'])
    if found['drop']:
        facts['replays'] = describe(found['drop'], rows, by_row)
    if found['unnumbered']:
        facts['replays_without_numbers'] = describe(found['unnumbered'], rows, by_row)
    if found['unread']:
        facts['replays_numbers_not_read'] = describe(found['unread'], rows, by_row)
    if found['other_numbers']:
        facts['same_music_other_numbers'] = describe(found['other_numbers'], rows, by_row)
    if found['partial']:
        facts['partial_replays'] = '; '.join(f'row {a} ~ row {b} ({s:.3f})' for a, b, s in found['partial'])
    if layout == AS_PLAYED:
        if found['drop']:
            report.warn('measures', f"this video replays pages: {facts['replays']}; choose 'as written' if the "
                        'score has repeat or D.C./D.S. markings (each replayed page is then printed once).')
        return facts
    if not numbered_video:
        if found['unnumbered'] or found['unread']:
            shown = '; '.join(facts[k] for k in ('replays_without_numbers', 'replays_numbers_not_read') if k in facts)
            report.warn('measures', f'as written: pages are shown again ({shown}), but no measure numbers are read '
                        'in this video to confirm they are replays, so the sheet is as played.')
        return facts
    if found['unread']:
        report.warn('measures', f"as written: {facts['replays_numbers_not_read']} show a page again, but its "
                    'measure numbers could not be read to confirm it; kept.')
    if found['unnumbered']:
        report.warn('measures', f"as written: {facts['replays_without_numbers']} show a page again without a "
                    'printed measure number to confirm it; kept.')
    for row, old, share in found['partial']:
        report.warn('measures', f'as written: row {row} ({rows_seconds(rows, row)}) nearly repeats row {old} '
                    f'with the same measure numbers, but not exactly ({share:.1%} of the ink differs); kept.')
    if found['drop']:
        dropped = {r['measure'] for n, _ in found['drop'] for r in by_row[n]}
        drop_measures(records, dropped, run_dir)
        report.warn('measures', f"as written: replayed pages printed once: {facts['replays']} (the same image, "
                    f'measure numbers included); {len(dropped)} measures dropped.')
        facts['replayed_measures_dropped'] = len(dropped)
    return facts


def save(run_dir, layout, dropped, found):
    """replays.json, for play along (tabvid/timing.py): `dropped` [[row, original row]] are the rows left out
    of the sheet as replays (their time spans light the original row's measures); `found` the replays found
    in either layout (as played they stay printed). Written on every run of this stage."""
    data = {'layout': layout, 'dropped': [list(p) for p in dropped], 'found': [list(p) for p in found]}
    (Path(run_dir) / 'replays.json').write_text(json.dumps(data) + '\n')


def rows_seconds(rows, name):
    row = next(row for row in rows if row['row'] == name)
    return f"{row['seconds'][0]}-{row['seconds'][1]}s"


def drop_measures(records, dropped, run_dir):
    """Remove the measures numbered in `dropped` (and their images); renumber the rest and rename their
    images in order; in_row is counted again."""
    out_dir = Path(run_dir) / 'measures'
    kept = [r for r in records if r['measure'] not in dropped]
    for r in records:
        if r['measure'] in dropped:
            (out_dir / Path(r['image']).name).unlink(missing_ok=True)
    for number, r in enumerate(kept, 1):
        if r['measure'] != number:
            name = f'm{number:03d}.png'
            (out_dir / Path(r['image']).name).rename(out_dir / name)
            r['measure'], r['image'] = number, f'measures/{name}'
    counts = {}
    for r in kept:
        counts[r['row']] = r['in_row'] = counts.get(r['row'], 0) + 1
    records[:] = kept
