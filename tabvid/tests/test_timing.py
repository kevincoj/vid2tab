"""Play along timing (tabvid/timing.py) on synthetic run folders: python -m pytest tabvid/tests/test_timing.py -q

A run folder is made by hand: frames with a staff and (optionally) a blue cursor band moving at a known pace,
rows.json / views.json / measures.json as the pipeline writes them. Real-video accuracy is in docs/history.md.
"""
import json

import cv2
import numpy as np
import pytest

from tabvid import replays, timing

FPS = 2
W, H = 600, 120
LINES = [40, 50, 60, 70, 80, 90]
BARS = [20, 160, 300, 440, 580]          # 4 measures of 140 px per row
PACE = 2.0                               # seconds per measure


def staff_image(width=W):
    image = np.full((H, width, 3), 250, np.uint8)
    for y in LINES:
        image[y] = 0
    return image


def draw_cursor(image, x):
    x = int(round(x))
    band = image[LINES[0]:LINES[-1] + 1, max(0, x - 8):x + 8].astype(int)
    band[...] = (band * [0.95, 0.85, 0.78]).astype(int)      # translucent blue (BGR)
    image[LINES[0]:LINES[-1] + 1, max(0, x - 8):x + 8] = band.clip(0, 255).astype(np.uint8)
    # light enough to be "tinted": raise the white parts' tint
    return image


def write_run(tmp_path, rows, frames, measures, views=None, extra=None):
    run = tmp_path / 'run'
    (run / 'frames').mkdir(parents=True)
    for i, image in enumerate(frames):
        cv2.imwrite(str(run / 'frames' / f'{i:06d}.png'), image)
    (run / 'frames' / 'frames.json').write_text(json.dumps({'fps': FPS, 'count': len(frames)}))
    width = frames[0].shape[1] if frames else W
    (run / 'config.json').write_text(json.dumps({'crop': [0, 0, width, H], 'range': {'start': 30.0, 'end': None},
                                                 **(extra or {})}))
    (run / 'rows.json').write_text(json.dumps(rows))
    (run / 'measures.json').write_text(json.dumps(measures))
    if views is not None:
        (run / 'views.json').write_text(json.dumps(views))
    return run


def row_measures(row, first, bars=BARS, base=0):
    return [{'measure': first + k, 'row': row, 'row_x0': base + a, 'staff_x0': 0, 'staff_x1': b - a, 'width': b - a}
            for k, (a, b) in enumerate(zip(bars, bars[1:]))]


def page_run(tmp_path, cursor=True, replay=False):
    """Rows 1 and 2 of 4 measures each, 8 s per row, the cursor going through at PACE s per measure; with
    `replay`, row 3 shows row 1 again (as written: dropped from the sheet, measures 1-4 played again)."""
    rows, frames, shown = [], [], [1, 2] + ([1] if replay else [])
    for n, _ in enumerate(shown, 1):
        indices = list(range(len(frames), len(frames) + 16))
        for i in range(16):
            t = i / FPS + 0.25          # the moment frame i shows (see timing.SOURCE_FRAME)
            image = staff_image()
            for x in BARS:
                image[LINES[0]:LINES[-1] + 1, x] = 0
            if cursor:
                draw_cursor(image, BARS[0] + 140 * t / PACE)
            frames.append(image)
        rows.append({'row': n, 'frames': indices, 'seconds': [indices[0] / FPS, indices[-1] / FPS],
                     'staff_lines': LINES, 'spacing': 10, 'image': f'rows/row-{n:02d}.png'})
    measures = row_measures(1, 1) + row_measures(2, 5)
    return write_run(tmp_path, rows, frames, measures)


def starts(timing_):
    return [[round(s[0], 1) for s in m['spans']] for m in timing_['measures']]


def test_cursor_gives_each_measure_its_start(tmp_path):
    run = page_run(tmp_path)
    out = timing.build(run)
    assert out['how'] == 'cursor' and out['offset'] == 30.0
    want = [8 * r + PACE * k for r in (0, 1) for k in range(4)]
    got = [m['spans'][0][0] for m in out['measures']]
    assert np.allclose(got, want, atol=0.3), got


def test_without_a_cursor_rows_share_their_time_evenly(tmp_path):
    run = page_run(tmp_path, cursor=False)
    out = timing.build(run)
    assert out['how'] == 'rows'
    got = [m['spans'][0][0] for m in out['measures']]
    assert np.allclose(got, [0.23 + v for v in (0, 2, 4, 6, 8, 10, 12, 14)], atol=0.1), got
    assert np.allclose(timing.build(run, use_cursor=False)['measures'][0]['spans'], out['measures'][0]['spans'])


def test_a_narrow_pickup_gets_a_shorter_share(tmp_path):
    run = page_run(tmp_path, cursor=False)
    measures = json.loads((run / 'measures.json').read_text())
    pickup = [{'measure': 1, 'row': 1, 'row_x0': 20, 'staff_x0': 0, 'staff_x1': 35, 'width': 35}]
    rest = [{**m, 'measure': m['measure'] + 1} for m in measures]
    (run / 'measures.json').write_text(json.dumps(pickup + rest))
    spans = [m['spans'][0] for m in timing.build(run)['measures']]
    assert spans[0][1] - spans[0][0] < 0.5 * (spans[2][1] - spans[2][0])


def test_as_written_a_replayed_row_lights_the_original_measures_again(tmp_path):
    run = page_run(tmp_path, replay=True)
    replays.save(run, 'as written', [(3, 1)], [(3, 1)])
    out = timing.build(run)
    assert out['facts']['replays'] == [[3, 1]] and out['facts']['passes'] == 2
    got = starts(out)
    assert [len(s) for s in got] == [2, 2, 2, 2, 1, 1, 1, 1]
    assert np.allclose([s[1] for s in got[:4]], [16, 18, 20, 22], atol=0.3)


def test_old_sheets_read_their_replays_from_the_report(tmp_path):
    run = tmp_path / 'old'
    run.mkdir()
    rows = [{'row': n, 'seconds': [n * 10.0, n * 10 + 9.5]} for n in range(1, 13)]
    by_row = {n: [{'measure': 4 * (n - 1) + k} for k in range(1, 5)] for n in range(1, 13)}
    text = replays.describe([(9, 1), (10, 2), (11, 3), (12, 8)], rows, by_row)
    report = {'stages': {'measures': {'replays': text, 'replayed_measures_dropped': 16}}}
    (run / 'report.json').write_text(json.dumps(report))
    assert timing.replays_of(run) == [(9, 1), (10, 2), (11, 3), (12, 8)]
    # as played the same fact is only a note: nothing was dropped
    report['stages']['measures'].pop('replayed_measures_dropped')
    (run / 'report.json').write_text(json.dumps(report))
    assert timing.replays_of(run) == []
    # replays.json wins once a run has one
    replays.save(run, 'as written', [(5, 2)], [(5, 2)])
    assert timing.replays_of(run) == [(5, 2)]


def scroll_run(tmp_path, cursor):
    """A strip of 16 measures (140 px) seen through a 600 px view that jumps 420 px, then a last small
    scroll of 40 px. The music (and the cursor) starts at measure 1 after a 2 s intro and moves 70 px/s
    (PACE s per measure, measure k from 0 starts at 2 + 2k s); each scroll comes when it is 90 px past the
    next view's left edge (half the 180 px overlap): at 9, 15, 21 and 27 s, the last one at 33 s."""
    bars = [20 + 140 * k for k in range(17)]
    strip = staff_image(2320)
    for x in bars:
        strip[LINES[0]:LINES[-1] + 1, x] = 0
    held = [(0, 0.0, 9.0)] + [(420 * k, 3.0 + 6 * k, 9.0 + 6 * k) for k in range(1, 5)] + [(1720, 33.0, 36.0)]
    views, frames, places = [], [], []
    for n, (shift, start, end) in enumerate(held, 1):
        indices = []
        for i in range(int(round(start * FPS)), int(round(end * FPS))):
            t = i / FPS + 0.25            # the moment frame i shows
            frame = strip[:, shift:shift + W].copy()
            x = 20 + 70 * (t - 2)
            if cursor and t >= 2 and shift + 8 <= x <= shift + W - 8:
                draw_cursor(frame, x - shift)
            indices.append(len(frames))
            frames.append(frame)
        views.append({'view': n, 'frames': indices, 'kept_frames': indices,
                      'seconds': [indices[0] / FPS, indices[-1] / FPS], 'staff_lines': LINES, 'spacing': 10})
        places.append({'view': n, 'x': shift, 'y': 0})
    rows = [{'row': 1, 'scroll': True, 'view_width': W, 'views': places, 'frames': list(range(len(frames))),
             'seconds': [0, (len(frames) - 1) / FPS], 'staff_lines': LINES, 'spacing': 10}]
    measures = row_measures(1, 1, bars)
    return write_run(tmp_path, rows, frames, measures, views)


def test_scrolling_with_a_cursor(tmp_path):
    out = timing.build(scroll_run(tmp_path, cursor=True))
    assert out['how'] == 'cursor'
    got = [m['spans'][0][0] for m in out['measures']]
    assert np.allclose(got, [2 + 2 * k for k in range(16)], atol=0.3), got


def test_scrolling_without_a_cursor_plays_from_the_middle_of_the_overlap(tmp_path):
    out = timing.build(scroll_run(tmp_path, cursor=False))
    assert out['how'] == 'views' and all(m['spans'] for m in out['measures'])
    got = [m['spans'][0][0] for m in out['measures']]
    assert np.allclose(got, [2 + 2 * k for k in range(16)], atol=0.35), got
    # the last 40 px scroll belongs to the hold before it: the last measure is not stretched over it
    last = out['measures'][-1]['spans'][0]
    assert last[1] - last[0] < 2 * PACE


def test_a_cursor_going_back_starts_a_new_pass():
    samples = [(0, 0.1), (1, 0.6), (2, 1.2), (3, 1.7), (4, 0.2), (5, 0.7), (6, 1.3)]
    assert [len(p) for p in timing.passes(samples)] == [4, 3]
    spans = timing.measure_spans(samples, 2)
    assert [len(s) for s in spans] == [2, 2]
    # a sample slightly behind is noise, not a repeat
    assert len(timing.passes([(0, 0.5), (1, 0.4), (2, 0.9)])) == 1


def test_a_run_without_rows_has_no_timing(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    out = timing.build(tmp_path)
    assert out['how'] is None and out['measures'] == []


def test_timing_is_cached_until_an_input_changes(tmp_path):
    run = page_run(tmp_path, cursor=False)
    first = timing.load(run)
    assert (run / 'timing.json').exists() and timing.load(run) == first
    measures = json.loads((run / 'measures.json').read_text())[:4]
    (run / 'measures.json').write_text(json.dumps(measures))
    import os
    os.utime(run / 'measures.json', (1, 1))
    assert len(timing.load(run)['measures']) == 4


def test_timing_is_rebuilt_when_the_timing_code_changes(tmp_path, monkeypatch):
    run = page_run(tmp_path, cursor=False)
    timing.load(run)
    saved = json.loads((run / 'timing.json').read_text())
    saved['measures'] = []                                  # what an older version of the code would have made
    (run / 'timing.json').write_text(json.dumps(saved))
    assert timing.load(run)['measures'] == []               # same code, same inputs: the cache is used
    monkeypatch.setattr(timing, 'code_version', lambda: 'changed')
    assert timing.load(run)['measures']                     # the code changed: rebuilt


@pytest.mark.parametrize('layout,numbered', [('as written', True), ('as played', True), ('as written', False)])
def test_stage_4d_saves_replays_json_in_every_layout(tmp_path, layout, numbered, monkeypatch):
    monkeypatch.setattr(replays, 'find', lambda *a: {'drop': [(3, 1)], 'partial': [], 'unnumbered': [],
                                                     'unread': [], 'other_numbers': []})
    monkeypatch.setattr(replays, 'drop_measures', lambda *a: None)

    class Report:
        def warn(self, *a):
            pass
    rows = [{'row': n, 'seconds': [n, n + 1]} for n in (1, 2, 3)]
    records = [{'measure': n, 'row': n} for n in (1, 2, 3)]
    facts = {'font': 'x', 'sequence': 'consistent'} if numbered else {}
    replays.apply(rows, records, tmp_path, Report(), layout, lambda rs: facts)
    saved = json.loads((tmp_path / 'replays.json').read_text())
    assert saved['found'] == [[3, 1]]
    assert saved['dropped'] == ([[3, 1]] if layout == 'as written' and numbered else [])
