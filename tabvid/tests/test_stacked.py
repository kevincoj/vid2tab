"""Stacked staves: two or more TAB staves one above the other in a frame (docs/multi-row-plan.md)."""
import cv2
import numpy as np

from tabvid.staff import find_staff, find_staves

SP = 20


def page(tops, bars, height=None, width=1000):
    """A gray page with one six-line TAB staff per top y; bars[k]: barline x positions of staff k."""
    height = height or tops[-1] + 5 * SP + 120
    img = np.full((height, width), 245, np.uint8)
    for top, xs in zip(tops, bars):
        for i in range(6):
            cv2.line(img, (xs[0], top + i * SP), (xs[-1], top + i * SP), 90, 1)
        for x in xs:
            img[top:top + 5 * SP + 1, x - 1:x + 2] = 20
        for x in range(xs[0] + 60, xs[-1] - 20, 90):           # fret digits on the second line
            cv2.putText(img, '3', (x, top + SP + 6), cv2.FONT_HERSHEY_SIMPLEX, .5, 20, 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def test_two_staves_found_top_to_bottom_with_their_own_barlines():
    rgb = page([80, 330], [(40, 500, 960), (40, 300, 640, 960)])
    staves = find_staves(rgb)
    assert len(staves) == 2
    assert staves[0].lines[0] < staves[1].lines[0]
    assert [round(b.x) for b in staves[0].barlines] == [40, 500, 960]
    assert [round(b.x) for b in staves[1].barlines] == [40, 300, 640, 960]
    assert staves[1].lines == find_staff(rgb).lines                # the lowest is find_staff's own


def test_three_staves():
    rgb = page([60, 290, 520], [(40, 500, 960), (40, 400, 960), (40, 300, 640, 960)])
    staves = find_staves(rgb)
    assert [round(s.lines[0]) for s in staves] == [60, 290, 520]
    assert [len(s.barlines) for s in staves] == [3, 3, 4]


def test_single_staff_is_find_staff():
    rgb = page([200], [(40, 330, 620, 960)])
    staves = find_staves(rgb)
    assert len(staves) == 1 and staves[0] == find_staff(rgb)


def test_row_of_chord_diagrams_is_not_a_second_staff():
    rgb = page([300], [(40, 330, 620, 960)])
    gray = rgb[..., 0].copy()
    for left in range(60, 940, 150):                              # fret lines at the staff's spacing
        for f in range(6):
            cv2.line(gray, (left, 60 + f * SP), (left + 45, 60 + f * SP), 20, 1)
        for s in range(6):
            cv2.line(gray, (left + 9 * s, 60), (left + 9 * s, 60 + 5 * SP), 20, 1)
    staves = find_staves(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    assert len(staves) == 1 and round(staves[0].lines[0]) == 300


def test_staff_too_close_above_is_not_another_row():
    rgb = page([80, 80 + 5 * SP + 3 * SP], [(40, 500, 960), (40, 300, 640, 960)])  # 3 spacings apart
    assert len(find_staves(rgb)) == 1


# --- splitting a page into rows (rows.split_page, build_rows) ---------------------------------------

import json

from tabvid import rows as R
from tabvid.report import Report

TOPS = (80, 360)                     # staff 1: 80-180, staff 2: 360-460; gap 180-360


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def test_stems_go_up_and_a_chord_name_goes_down():
    rgb = page(TOPS, [(40, 500, 960), (40, 300, 640, 960)])
    rgb[180:215, 299:301] = 20                                     # stem hanging from staff 1, beam at its foot
    rgb[213:216, 299:380] = 20
    cv2.putText(rgb, 'Am', (300, 330), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)   # chord name over staff 2
    staves = find_staves(rgb)
    bands, crossed, cuts = R.split_page(rgb, staves)
    assert crossed == 0 and len(bands) == 2 and 216 < cuts[0] < 300
    (upper, top0), (lower, top1) = bands
    assert (_gray(upper)[214 - top0, 300:380] < 100).all()         # the beam is in the upper row
    assert top1 < 310 and (_gray(lower)[310 - top1:335 - top1, 300:345] < 100).any()  # chord name: lower row
    assert top0 + upper.shape[0] < 300                              # the upper row ends above it


def test_ink_crossing_the_cut_goes_whole_to_the_side_holding_most_of_it():
    rgb = page(TOPS, [(40, 500, 960), (40, 300, 640, 960)])
    rgb[183:359, 700:702] = 20                                     # a line through the whole gap...
    rgb[300:345, 690:712] = 20                                     # ...mostly near staff 2
    staves = find_staves(rgb)
    bands, crossed, cuts = R.split_page(rgb, staves)
    assert crossed == 1
    (upper, top0), (lower, top1) = bands
    assert not (_gray(upper)[183 - top0:, 699:703] < 100).any()    # none of it left under the upper staff
    assert (_gray(lower)[183 - top1:345 - top1, 700] < 100).all()  # all of it in the lower row, whole


def _video(tmp_path, pages):
    """Frame files: each (image, count) page held for `count` frames."""
    frames = []
    for image, count in pages:
        for _ in range(count):
            path = tmp_path / 'frames' / f'{len(frames):06d}.png'
            path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            frames.append(path)
    return frames


def _rows(tmp_path, pages):
    frames = _video(tmp_path, pages)
    run = tmp_path / 'run'
    (run / 'debug').mkdir(parents=True)
    report = Report(run)
    h, w = pages[0][0].shape[:2]
    records = R.build_rows(frames, (0, 0, w, h), 2, run, report)
    return records, report


def test_pages_become_rows_top_to_bottom(tmp_path):
    a = page(TOPS, [(40, 500, 960), (40, 300, 640, 960)])
    b = page(TOPS, [(40, 330, 700, 960), (40, 480, 960)])
    c = page(TOPS[:1], [(40, 250, 960)], height=a.shape[0])           # a last page holding one row
    records, report = _rows(tmp_path, [(a, 4), (b, 4), (c, 2)])
    assert [(r['page'], r['slot'], r['measures']) for r in records] == \
        [(1, 0, 2), (1, 1, 3), (2, 0, 3), (2, 1, 2), (3, 0, 2)]
    assert len({cv2.imread(str(tmp_path / 'run' / r['image'])).shape for r in records}) == 1  # one row size
    assert records[0]['frames'] == records[1]['frames'] == [0, 1, 2, 3]
    assert 'on' in report.stages['rows']['stacked_staves']


def test_a_few_stacked_frames_are_warned_not_dropped_silently(tmp_path):
    single = page((300,), [(40, 330, 620, 960)], height=page(TOPS, [(40, 960)] * 2).shape[0])
    double = page(TOPS, [(40, 500, 960), (40, 300, 640, 960)])
    records, report = _rows(tmp_path, [(single, 8), (double, 2)])
    assert 'page' not in records[0]
    assert report.stages['rows']['stacked_staves'] == 'off'
    assert any('show two or more TAB staves' in w and '#8' in w or '4.0s' in w for _, w in report.warnings)


# --- rolling pages (fix plan 4): the lower row moves up, a new row comes in under it ---------------

A, B, C, D, E = (40, 500, 960), (40, 300, 640, 960), (40, 330, 700, 960), (40, 480, 960), (40, 250, 600, 960)


def _bars(records):
    return [tuple(round(b['x']) for b in r['barlines']) for r in records]


def test_rolled_rows_are_dropped_and_listed(tmp_path):
    h = page(TOPS, [A, B]).shape[0]
    mid = page((TOPS[0] - 120, TOPS[1] - 120), [B, C], height=h)       # caught mid-roll, one frame
    pages = [(page(TOPS, [A, B]), 4), (mid, 1), (page(TOPS, [B, C]), 4), (page(TOPS, [C, D]), 4),
             (page(TOPS, [A, E]), 4)]                                    # a jump: both rows new
    records, report = _rows(tmp_path, pages)
    assert _bars(records) == [A, B, C, D, A, E]
    facts = report.stages['rows']
    assert facts['rolling_pages'].startswith('on: 2 of 3')
    assert len(facts['rolled_rows_dropped']) == 2 and 'page 2 row 1 (= page 1 row 2' in facts['rolled_rows_dropped'][0]
    assert facts['dropped: page rolling'] == [4]


def test_pages_that_rarely_roll_keep_every_row(tmp_path):
    pages = [(page(TOPS, [A, B]), 4), (page(TOPS, [C, D]), 4), (page(TOPS, [D, E]), 4), (page(TOPS, [A, C]), 4)]
    records, report = _rows(tmp_path, pages)
    assert _bars(records) == [A, B, C, D, D, E, A, C]                   # 1 of 3 changes rolls: nothing dropped
    assert report.stages['rows']['rolling_pages'].startswith('off: 1 of 3')


def test_a_page_shown_only_under_a_dark_overlay_is_warned(tmp_path):
    a, b = page((200,), [A]), page((200,), [B])
    dark = (b.astype(int) * .7).astype(np.uint8)          # page B only under a dark overlay (paper 74 darker)
    records, report = _rows(tmp_path, [(a, 4), (dark, 3), (a, 4)])
    assert any('under a dark or coloured overlay' in w for _, w in report.warnings)
    (tmp_path / 'b').mkdir()
    records, report = _rows(tmp_path / 'b', [(a, 4), ((a.astype(int) * .7).astype(np.uint8), 3)])   # a fade of A
    assert not any('overlay' in w for _, w in report.warnings)
