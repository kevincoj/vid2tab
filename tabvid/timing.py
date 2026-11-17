"""Play along timing: when in the video each measure (and line) of a sheet is played.

Not a pipeline stage: a function of a finished run folder (the server calls it and caches timing.json;
`python -m tabvid.timing RUN` prints it). The PDF, CLI and re-run set do not change.

Inputs, all saved by the pipeline: frames/ (frames.json fps; frame i shows the cut video at
(i + 0.5) / fps less about one source frame: FFmpeg's fps filter, measured on Andantino and A Team, where
2 fps frame i equals the source frame at i / 2 + 0.233-0.244 s; rows.json and views.json seconds are i / fps,
so they get the same shift), config.json (crop, range), rows.json (each row's frames; a scrolling strip lists its views with
their offsets `x`), views.json (each view's frames), measures.json (each measure's row and x in it),
replays.json (rows left out as replays; older sheets: the `replays` fact in report.json) and
report.json (measures per line).

Every moment of the video is turned into a score position P: measure index i (0-based, in sheet order)
plus how far through it (by staff x). Measure i starts when P reaches i and ends when it reaches i + 1. A
replayed row (as written) uses the geometry of the row it replays, so P jumps back and its measures get a
second span; a cursor going back inside a row (a printed repeat) does the same.

- Cursor: many videos move a bar or line across the music (Andantino's blue band, Guitar Pro's blue
  line). Per frame, a narrow run of columns that are tinted (coloured, light) over most of the staff
  height; the video's cursor colour is the most common one among such runs; runs touching the crop's
  edges are left out. Where it is found in at least CURSOR_SHARE of the music frames, P comes from it.
- No cursor, page switching: a row is played from its first frame to the next row's first frame, each
  measure an equal share (the meter is constant; the measures' widths vary with note density).
- No cursor, scrolling: during hold k (view at offset x_k) the strip from x_k + a to x_(k+1) + a is
  played, a = where measure 1 starts in the first view.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

TINT = 18            # max - min channel: a coloured pixel
LIGHT = 110          # ...and lighter than this (dark coloured ink, e.g. a blue fret number, is not a cursor)
COVER = .6           # share of the staff height a cursor column is tinted
NARROW = 3.0         # spacings: a cursor is at most this wide (a measure highlight box is wider)
HUE_BIN = 12         # OpenCV hue units (0-180) per colour bin
CURSOR_SHARE = .5    # of the frames shown with music, the cursor is found in at least this share
STILL = 3            # px: a run back at the same place (in steps of this) ...
STILL_SHARE = .3     # ...in more than this share of its screen's frames does not move: not a cursor
EDGE = .25           # spacings: a cursor this close to a barline...
EDGE_SHARE = .5      # ...in more than this share of frames is the edge of a measure box (not used yet)
AGREE = 1.5          # a cursor's median measure length within this factor of the row/hold estimate
SPANNED = .8         # ...and giving at least this share of the measures a span
BACK = .5            # measures: P going back by more than this starts a new pass (repeat, replay)
SOURCE_FRAME = 1 / 60   # s: the fps filter's pick lands about one source frame before (i + 0.5) / fps
VERSION = 2          # bump to rebuild cached timing.json files


def _json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


# ---------------------------------------------------------------- inputs

def replays_of(run_dir):
    """[(row, original row)] dropped from the sheet as replays. replays.json, else (sheets made before it)
    the report's `replays` fact when the measures stage says measures were dropped as replays."""
    run_dir = Path(run_dir)
    saved = _json(run_dir / 'replays.json')
    if saved is not None:
        return [tuple(p) for p in saved.get('dropped', [])]
    facts = (_json(run_dir / 'report.json', {}) or {}).get('stages', {}).get('measures', {})
    if not facts.get('replayed_measures_dropped'):
        return []
    return parse_replays(str(facts.get('replays', '')))


def parse_replays(text):
    """'rows 9-12 (110.0-163.5s, m35-m51) = rows 1-4; row 14 (...) = row 8' -> [(9, 1), ..., (14, 8)]
    (the format of replays.describe)."""
    pairs = []
    for part in text.split(';'):
        found = re.match(r'\s*rows? (\d+)(?:-(\d+))? \(.*?\) = rows? (\d+)(?:-(\d+))?\s*$', part)
        if not found:
            continue
        a, b, c, d = found.groups()
        a, c = int(a), int(c)
        b, d = int(b or a), int(d or c)
        if b - a != d - c:
            continue
        pairs += [(a + k, c + k) for k in range(b - a + 1)]
    return pairs


class Run:
    """The parts of a run folder timing needs."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.config = _json(self.dir / 'config.json', {}) or {}
        self.fps = float((_json(self.dir / 'frames' / 'frames.json', {}) or {}).get('fps') or 2)
        self.rows = _json(self.dir / 'rows.json', []) or []
        self.views = {v['view']: v for v in _json(self.dir / 'views.json', []) or []}
        self.measures = _json(self.dir / 'measures.json', []) or []
        self.scrolling = any(r.get('scroll') for r in self.rows)
        self.replays = replays_of(self.dir)
        sheet = (_json(self.dir / 'report.json', {}) or {}).get('stages', {}).get('sheet', {})
        try:
            self.per_line = [int(n) for n in str(sheet.get('measures_per_line', '')).split()]
        except ValueError:
            self.per_line = []
        self.delay = 0.5 / self.fps - SOURCE_FRAME     # frame i / fps -> the moment it shows
        rng = self.config.get('range') or {}
        self.offset = float(rng.get('start') or 0)
        # measure geometry per row: [(index, start x, end x)] in row (strip) coordinates
        self.geometry = {}
        for i, m in enumerate(self.measures):
            x0 = m['row_x0'] + m.get('staff_x0', 0)
            x1 = m['row_x0'] + m.get('staff_x1', m.get('width', 0))
            self.geometry.setdefault(m['row'], []).append((i, x0, max(x1, x0 + 1)))

    def screens(self):
        """What the video shows, in time order: [{'row': geometry row, 'shift': x of the screen in the row,
        'frames': [frame indices], 'start': s, 'end': s}]. Page switching: one per row of rows.json (a
        replayed row uses its original's geometry). Scrolling: one per view in a strip."""
        original = dict(self.replays)
        out = []
        if self.scrolling:
            for row in self.rows:
                for place in row.get('views', []):
                    view = self.views.get(place['view'])
                    if view:
                        out.append({'row': row['row'], 'shift': place['x'], 'frames': view['frames'],
                                    'start': view['seconds'][0] + self.delay, 'end': view['seconds'][1] + self.delay})
        else:
            for row in self.rows:
                name = original.get(row['row'], row['row'])
                out.append({'row': name, 'shift': 0, 'frames': row['frames'], 'source_row': row['row'],
                            'start': row['seconds'][0] + self.delay, 'end': row['seconds'][1] + self.delay})
        out = [s for s in out if s['row'] in self.geometry]
        out.sort(key=lambda s: s['start'])
        return out

    def time(self, index):
        """Seconds of the cut video that frame `index` shows."""
        return index / self.fps + self.delay

    def band(self):
        """Rows of the crop the cursor must cover: top to bottom staff line (notation staff included)."""
        row = next((r for r in self.rows if r.get('staff_lines')), None)
        if not row:
            return None
        lines = list(row['staff_lines']) + list(row.get('notation_lines') or [])
        return int(min(lines)), int(max(lines)) + 1, float(row.get('spacing') or 10)

    def frame(self, index):
        import cv2
        image = cv2.imread(str(self.dir / 'frames' / f'{index:06d}.png'))
        if image is None:
            return None
        x0, y0, x1, y1 = self.config.get('crop') or (0, 0, image.shape[1], image.shape[0])
        return image[y0:y1, x0:x1]

    def position(self, row, x):
        """Score position P of x in a row (row coordinates), clamped to the row's measures."""
        geometry = self.geometry[row]
        first, last = geometry[0], geometry[-1]
        if x <= first[1]:
            return float(first[0])
        if x >= last[2]:
            return float(last[0] + 1)
        for index, x0, x1 in geometry:
            if x < x1:
                return index + max(0.0, (x - x0) / (x1 - x0))
        return float(last[0] + 1)


# ---------------------------------------------------------------- cursor

def runs_in(image, band):
    """Narrow tinted column runs over the staff band: [(x0, x1, hue bin)]. A run is a stretch of columns
    of one colour (a gray antialiased column of up to 2 px inside it is bridged); a cursor line next to a
    measure highlight of another colour is a run of its own."""
    import cv2
    top, bottom, spacing = band
    part = image[top:bottom]
    if part.size == 0:
        return []
    values = part.astype(np.int16)
    tinted = ((values.max(axis=2) - values.min(axis=2)) > TINT) & (values.mean(axis=2) > LIGHT)
    cols = np.flatnonzero(tinted.mean(axis=0) > COVER)
    if not len(cols):
        return []
    hue = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)[..., 0].astype(float)
    hue[~tinted] = np.nan
    bins = (np.nanmedian(hue[:, cols], axis=0) // HUE_BIN).astype(int)
    out, start = [], 0
    width = image.shape[1]
    for k in range(len(cols)):
        last = k + 1 == len(cols)
        if not last and cols[k + 1] - cols[k] <= 3 and bins[k + 1] == bins[k]:
            continue
        x0, x1 = int(cols[start]), int(cols[k])
        if x1 - x0 + 1 <= NARROW * spacing and x0 > 1 and x1 < width - 2:
            out.append((x0, x1, int(np.bincount(bins[start:k + 1]).argmax())))
        start = k + 1
    return out


def cursor_track(run, screens):
    """(samples, facts): samples [(t, P)] from the cursor, or [] when the video has none."""
    band = run.band()
    if not band:
        return [], {'cursor': 'no staff band'}
    found = []
    for screen in screens:
        for index in screen['frames']:
            image = run.frame(index)
            if image is not None:
                found.append((index, screen, runs_in(image, band)))
    if not found:
        return [], {'cursor': 'no frames'}
    found = _moving(found)
    counts = {}
    for _, _, runs in found:
        for hue in {h for _, _, h in runs}:
            counts[hue] = counts.get(hue, 0) + 1
    if not counts:
        return [], {'cursor': 'none found'}
    hue = max(counts, key=counts.get)
    share = counts[hue] / len(found)
    if share < CURSOR_SHARE:
        return [], {'cursor': f'none (a coloured column in {share:.0%} of frames)'}
    samples, last, on_edges = [], None, 0
    near = max(3.0, EDGE * band[2])
    for index, screen, runs in found:
        mine = [(x0 + x1) / 2 + screen['shift'] for x0, x1, h in runs if h == hue]
        if not mine:
            continue
        edges = [x for _, x0, x1 in run.geometry[screen['row']] for x in (x0, x1)]
        here = [run.position(screen['row'], x) for x in mine]
        k = min(range(len(here)), key=lambda j: abs(here[j] - last)) if last is not None else \
            min(range(len(here)), key=lambda j: here[j])
        on_edges += min(abs(mine[k] - e) for e in edges) <= near
        samples.append((run.time(index), here[k]))
        last = here[k]
    facts = {'cursor': f'found in {share:.0%} of frames', 'cursor_hue': hue * HUE_BIN}
    if samples and on_edges / len(samples) > EDGE_SHARE:
        facts['cursor'] = (f'not used: the coloured columns found ({share:.0%} of frames) sit on barlines in '
                           f'{on_edges / len(samples):.0%} of them (the edges of a measure highlight, not a cursor)')
        return [], facts
    return samples, facts


def _moving(found):
    """Leave out runs that stay put: a run of a colour at the same place (within STILL px) in more than
    STILL_SHARE of its screen's frames (at least 3) is part of the page (a coloured fringe along a barline
    or the crop edge, a panel border), not a cursor. A cursor resting on a long note for a frame or two
    stays in."""
    by_screen = {}
    for item in found:
        by_screen.setdefault(id(item[1]), []).append(item)
    out = []
    for items in by_screen.values():
        seen = {}
        for _, _, runs in items:
            for key in {(h, round((x0 + x1) / 2 / STILL)) for x0, x1, h in runs}:
                seen[key] = seen.get(key, 0) + 1
        limit = max(3, STILL_SHARE * len(items))

        def still(x0, x1, h):
            c = round((x0 + x1) / 2 / STILL)
            return max(seen.get((h, c + d), 0) for d in (-1, 0, 1)) > limit
        out += [(index, screen, [r for r in runs if not still(*r)]) for index, screen, runs in items]
    out.sort(key=lambda item: item[0])
    return out


# ---------------------------------------------------------------- without a cursor

def row_track(run, screens):
    """Page switching without a cursor: each screen from its first frame to the next screen's first frame,
    each measure an equal share of the time (the meter is constant), except a narrow one (a pickup, a
    short ending: narrower than the video's median measure), whose share is its width over the median.
    The first screen usually shows before the music starts and the last stays after it ends: they are
    played at the median pace of the others, the first ending and the last starting on its page change.
    Stacked staves: the rows of one page (the same frames) share its time, top to bottom."""
    widths = [x1 - x0 for g in run.geometry.values() for _, x0, x1 in g]
    median = float(np.median(widths)) if widths else 1.0
    pages = []                      # stacked staves: the rows shown together on one page, top to bottom
    for screen in screens:
        if pages and pages[-1][0]['frames'] == screen['frames']:
            pages[-1].append(screen)
        else:
            pages.append([screen])
    plan = []
    for k, page in enumerate(pages):
        geometry = [g for screen in page for g in run.geometry[screen['row']]]
        end = pages[k + 1][0]['start'] if k + 1 < len(pages) else page[0]['end'] + 1 / run.fps
        weights = [min(1.0, (x1 - x0) / median) for _, x0, x1 in geometry]
        plan.append([page[0]['start'], end, geometry, weights])
    paces = [(b - a) / sum(w) for a, b, _, w in plan[1:-1]]
    if paces:
        pace = float(np.median(paces))
        plan[0][0] = max(plan[0][0], plan[0][1] - pace * sum(plan[0][3]))
        if len(plan) > 1:
            plan[-1][1] = min(plan[-1][1], plan[-1][0] + pace * sum(plan[-1][3]))
    samples = []
    for start, end, geometry, weights in plan:
        t, total = start, sum(weights)
        for (index, _, _), weight in zip(geometry, weights):
            samples.append((t, float(index)))
            t += (end - start) * weight / total
        samples.append((end - 1e-3, geometry[-1][0] + 1 - 1e-3))
    return samples


def scroll_track(run, screens):
    """Scrolling without a cursor: hold k plays the strip from x_k + a to x_(k+1) + a. The play point a is
    half the overlap a scroll keeps (view width less the median step): in A Team and 7kq the cursor is
    150-250 px into the view just after each scroll, with overlaps of about 500 px. A step under half the
    median (a last small scroll to the end) belongs to the hold before it. The first hold plays from
    measure 1 and ends on its scroll, the last starts on its scroll and runs to the strip's end, both at
    the median pace of the others (the video shows them before and after the music)."""
    samples = []
    for row in sorted({s['row'] for s in screens}):
        mine = [s for s in screens if s['row'] == row]
        width = int(next((r.get('view_width') for r in run.rows if r['row'] == row), 0) or 0)
        steps = [b['shift'] - a['shift'] for a, b in zip(mine, mine[1:]) if b['shift'] > a['shift']]
        step = float(np.median(steps)) if steps else 0.0
        held = mine[:1]
        for screen in mine[1:]:
            if screen['shift'] - held[-1]['shift'] >= step / 2:
                held.append(screen)
            else:
                held[-1] = {**held[-1], 'end': screen['end']}
        a = max(run.geometry[row][0][1] - mine[0]['shift'], (width - step) / 2 if width > step else 0)
        samples += _hold_track(run, row, held, a)
    samples.sort()
    return samples


def _hold_track(run, row, mine, a):
    geometry = run.geometry[row]
    plan = []
    for k, screen in enumerate(mine):
        end = mine[k + 1]['start'] if k + 1 < len(mine) else screen['end'] + 1 / run.fps
        p0 = run.position(row, screen['shift'] + a) if k else float(geometry[0][0])
        p1 = run.position(row, mine[k + 1]['shift'] + a) if k + 1 < len(mine) else float(geometry[-1][0] + 1)
        plan.append([screen['start'], end, p0, max(p1, p0 + 1e-3)])
    paces = [(b - a_) / (p1 - p0) for a_, b, p0, p1 in plan[1:-1] if p1 > p0]
    if paces:
        pace = float(np.median(paces))
        plan[0][0] = max(plan[0][0], plan[0][1] - pace * (plan[0][3] - plan[0][2]))
        if len(plan) > 1:
            plan[-1][1] = min(plan[-1][1], plan[-1][0] + pace * (plan[-1][3] - plan[-1][2]))
    track = []
    for start, end, p0, p1 in plan:
        track += [(start, p0), (end - 1e-3, p1 - 1e-3)]
    return track


# ---------------------------------------------------------------- spans

def passes(samples):
    """Split the track where P goes back by more than BACK (a repeat or replay); inside a pass P only goes
    forward (a sample behind the pass so far is dropped as noise)."""
    out = []
    for t, p in samples:
        if out and p < out[-1][-1][1] - BACK:
            out.append([(t, p)])
        elif out and p < out[-1][-1][1]:
            continue
        elif out:
            out[-1].append((t, p))
        else:
            out.append([(t, p)])
    return out


def crossing(track, value, before=None):
    """Time P reaches `value` along a forward track (linear between samples). Before the first sample:
    extrapolated back at the track's own speed (not before `before`)."""
    (t0, p0) = track[0]
    if value <= p0:
        speed = _speed(track)
        t = t0 - (p0 - value) / speed if speed else t0
        return max(t, before) if before is not None else t
    for (ta, pa), (tb, pb) in zip(track, track[1:]):
        if pa <= value <= pb:
            return ta + (tb - ta) * ((value - pa) / (pb - pa) if pb > pa else 0)
    return None


def _speed(track):
    """Measures per second along a track (median of its steps)."""
    steps = [(pb - pa) / (tb - ta) for (ta, pa), (tb, pb) in zip(track, track[1:]) if tb > ta and pb > pa]
    return float(np.median(steps)) if steps else 0.0


def measure_spans(samples, count, speed=None):
    """Per measure index, its time spans [[start, end]] from the passes of the track."""
    spans = [[] for _ in range(count)]
    tracks = passes(samples)
    speed = speed or _speed(samples) or None
    for n, track in enumerate(tracks):
        before = tracks[n - 1][-1][0] if n else 0.0
        after = tracks[n + 1][0][0] if n + 1 < len(tracks) else None
        first, last = int(np.floor(track[0][1])), int(np.ceil(track[-1][1]))
        for i in range(max(0, first), min(count, last)):
            start = crossing(track, i, before) if i > track[0][1] or not speed else max(
                before, track[0][0] - (track[0][1] - i) / speed)
            end = crossing(track, i + 1)
            if end is None:      # the pass ends inside this measure: run on at the usual speed
                end = track[-1][0] + ((i + 1 - track[-1][1]) / speed if speed else 0)
                if after is not None:
                    end = min(end, after)
            if start is not None and end > start:
                spans[i].append([round(start, 2), round(end, 2)])
    return spans


def build(run_dir, use_cursor=True):
    """The timing of a run: {'version', 'offset' (add for the full video, e.g. YouTube), 'how',
    'measures': [{'n', 'spans': [[s, e], ...]}], 'lines': [{'measures': [first n, last n], 'spans'}],
    'facts'}. Times are seconds of the cut video (the local copy). 'how' None: no usable timing."""
    run = Run(run_dir)
    base = {'version': VERSION, 'offset': run.offset, 'layout': run.config.get('layout', 'as played')}
    screens = run.screens() if run.measures else []
    if not screens:
        return {**base, 'how': None, 'measures': [], 'lines': [], 'facts': {'why': 'no rows or measures'}}
    stacked = any('page' in r for r in run.rows)
    fallback = scroll_track(run, screens) if run.scrolling else row_track(run, screens)
    rough = measure_spans(fallback, len(run.measures))
    if not use_cursor:
        samples, facts = [], {'cursor': 'not used'}
    elif stacked:
        samples, facts = [], {'cursor': 'not looked for (stacked staves: rows share a page)'}
    else:
        samples, facts = cursor_track(run, screens)
    how = 'cursor'
    spans = measure_spans(samples, len(run.measures)) if len(samples) >= 2 else None
    if spans:
        mine, theirs = _median_length(spans), _median_length(rough)
        share = sum(bool(s) for s in spans) / len(spans)
        if not (mine and theirs and 1 / AGREE <= mine / theirs <= AGREE and share >= SPANNED):
            facts['cursor'] += (f' but not used: measures of {mine:.2f}s vs {theirs:.2f}s from the '
                                f'{"holds" if run.scrolling else "rows"}, {share:.0%} of measures reached')
            spans = None
    if not spans:
        how, samples, spans = ('views' if run.scrolling else 'rows'), fallback, rough
    filled = fill(spans)
    if filled:
        facts['filled'] = [run.measures[i]['measure'] for i in filled]
    measures = [{'n': m['measure'], 'spans': s} for m, s in zip(run.measures, spans)]
    lines, first = [], 0
    per_line = run.per_line if sum(run.per_line) == len(measures) else []
    for count in per_line:
        group = measures[first:first + count]
        lines.append({'measures': [group[0]['n'], group[-1]['n']], 'spans': _merge(s for m in group for s in m['spans'])})
        first += count
    facts.update(screens=len(screens), samples=len(samples), passes=len(passes(samples)),
                 replays=[list(p) for p in run.replays])
    return {**base, 'how': how, 'measures': measures, 'lines': lines, 'facts': facts}


def _median_length(spans):
    lengths = [b - a for s in spans for a, b in s]
    return float(np.median(lengths)) if lengths else 0.0


def fill(spans):
    """Give each measure without a span one: the gap between its neighbours' first spans, shared evenly by
    the measures in it; at the start or end, the median measure length before the next / after the last.
    Returns the indices filled."""
    empty = [i for i, s in enumerate(spans) if not s]
    if not empty or len(empty) == len(spans):
        return []
    length = _median_length(spans)
    k = 0
    while k < len(spans):
        if spans[k]:
            k += 1
            continue
        j = k
        while j < len(spans) and not spans[j]:
            j += 1
        before = spans[k - 1][0][1] if k else None
        after = spans[j][0][0] if j < len(spans) else None
        count = j - k
        if before is None:
            before = max(0.0, after - length * count)
        if after is None or after <= before:
            after = before + length * count
        step = (after - before) / count
        for n in range(count):
            spans[k + n] = [[round(before + step * n, 2), round(before + step * (n + 1), 2)]]
        k = j
    return empty


def _merge(spans, slack=1.0):
    """Join spans that touch or overlap (within `slack` seconds)."""
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + slack:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


INPUTS = ('rows.json', 'views.json', 'measures.json', 'replays.json', 'report.json', 'config.json',
          'frames/frames.json')


def code_version():
    """A short hash of this module's source: a change to the timing code rebuilds every cached timing.json
    (VERSION alone needed a hand bump)."""
    import hashlib
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]


def stamp(run_dir):
    run_dir = Path(run_dir)
    return [VERSION, code_version()] + [(run_dir / name).stat().st_mtime if (run_dir / name).exists() else None
                                        for name in INPUTS]


def load(run_dir):
    """timing.json, rebuilt when an input file or the timing code changed since it was made."""
    path = Path(run_dir) / 'timing.json'
    saved = _json(path)
    key = stamp(run_dir)
    if saved and saved.get('stamp') == key:
        return saved
    timing = {**build(run_dir), 'stamp': key}
    path.write_text(json.dumps(timing) + '\n')
    return timing


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m tabvid.timing', description=__doc__.split('\n')[0])
    parser.add_argument('run')
    parser.add_argument('--no-cursor', action='store_true', help='estimate without the cursor (for checks)')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    timing = build(args.run, use_cursor=not args.no_cursor)
    if args.json:
        print(json.dumps(timing, indent=1))
        return
    print(f"how: {timing['how']}  offset: {timing['offset']}  layout: {timing['layout']}  {timing['facts']}")
    for m in timing['measures']:
        print(f"m{m['n']:>3}  " + '  '.join(f'{a:7.2f}-{b:7.2f}' for a, b in m['spans']))


if __name__ == '__main__':
    sys.exit(main())
