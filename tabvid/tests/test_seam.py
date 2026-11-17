"""Scrolling video, strip seams: repeats dropped only when they match measure for measure, numbers
included; narrow measures with their own number in sequence are not warned.
python -m pytest tabvid/tests/test_seam.py -q"""
import json

import cv2
import numpy as np

from tabvid import measures
from tabvid.report import Report
from tabvid.tests.test_scroll import MEASURE, SP, TOP, strip

LINES = [TOP + i * SP for i in range(6)]


def setup(tmp_path, first, second, numbers):
    """Two strips (rows 1 and 2) of drawn measures with records as stage 4 makes them; `numbers`: the
    printed number read for each measure in order (None = unread)."""
    (tmp_path / 'measures').mkdir()
    rows, records = [], []
    for row_number, contents in ((1, first), (2, second)):
        gray = strip(contents)
        rows.append({'row': row_number, 'gray': gray, 'staff_lines': LINES, 'scroll': True})
        for k in range(len(contents)):
            n = len(records) + 1
            name = f'm{n:03d}.png'
            cv2.imwrite(str(tmp_path / 'measures' / name), gray[:, 20 + k * MEASURE:22 + (k + 1) * MEASURE])
            records.append({'measure': n, 'row': row_number, 'in_row': k + 1, 'image': f'measures/{name}',
                            'row_x0': 0, 'staff_x0': 20 + k * MEASURE, 'staff_x1': 22 + (k + 1) * MEASURE,
                            'spacing': SP})

    def read(rs):
        for r, number in zip(rs, numbers):
            r['number'] = number
    return rows, records, read


def test_a_new_strip_repeating_the_old_end_with_its_numbers_loses_the_repeats(tmp_path):
    rows, records, read = setup(tmp_path, [1, 2, 3, 4, 5], [4, 5, 6], [1, 2, 3, 4, 5, 4, 5, 6])
    report = Report(tmp_path)
    notes = measures.drop_seam_repeats(rows, records, tmp_path, report, read)
    assert notes == ['strip 2: m6-m7 = m4-m5 (numbers 4-5)']
    assert [(r['measure'], r['row'], r['in_row'], r['image']) for r in records] == [
        (1, 1, 1, 'measures/m001.png'), (2, 1, 2, 'measures/m002.png'), (3, 1, 3, 'measures/m003.png'),
        (4, 1, 4, 'measures/m004.png'), (5, 1, 5, 'measures/m005.png'), (6, 2, 1, 'measures/m006.png')]
    assert sorted(p.name for p in (tmp_path / 'measures').iterdir()) == [f'm00{i}.png' for i in range(1, 7)]
    kept = cv2.imread(str(tmp_path / 'measures' / 'm006.png'), 0)
    assert np.array_equal(kept, strip([4, 5, 6])[:, 20 + 2 * MEASURE:22 + 3 * MEASURE])   # measure "6" moved down
    assert all('number' not in r for r in records)          # the real number check reads again
    assert len(report.warnings) == 1 and 'repeats were dropped' in report.warnings[0][1]


def test_repeats_are_kept_unless_everything_matches(tmp_path):
    for second, numbers in (([4, 5, 6], [1, 2, 3, 4, 5, 4, None, 6]),     # one number unread
                            ([4, 5, 6], [1, 2, 3, 4, 5, 14, 15, 16]),     # the music repeats, numbers go on
                            ([4, 9, 6], [1, 2, 3, 4, 5, 4, 5, 6])):       # different music, same numbers
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        rows, records, read = setup(folder, [1, 2, 3, 4, 5], second, numbers)
        before = [r['measure'] for r in records]
        report = Report(folder)
        assert measures.drop_seam_repeats(rows, records, folder, report, read) == []
        assert [r['measure'] for r in records] == before and not report.warnings


def test_page_switching_videos_are_never_deduplicated(tmp_path):
    rows, records, read = setup(tmp_path, [1, 2, 3, 4, 5], [4, 5, 6], [1, 2, 3, 4, 5, 4, 5, 6])
    for row in rows:
        row['scroll'] = False
    assert measures.drop_seam_repeats(rows, records, tmp_path, Report(tmp_path), read) == []
    assert len(records) == 8


def test_narrow_measures_numbered_in_sequence_are_not_warned(tmp_path):
    # A Team m64 (a single strummed chord, 230 px vs 645) carries its own number 64 after 63: a real measure.
    records = [{'measure': n, 'row': 1, 'image': f'measures/m{n:03d}.png', 'staff_x0': 0,
                'staff_x1': 200 if n in (3, 5, 7) else 600} for n in range(1, 9)]
    narrow = [(f'm{n:03d}.png', 600) for n in (1, 3, 5, 7, 8)]
    report = Report(tmp_path)
    proven = {1, 2, 3, 4}                      # numbers read and in step from m1 to m5; m5 -> m6 not proven
    passed = measures.warn_narrow(narrow, records, proven, report)
    assert passed == [3, 5]                    # m5: its number follows m4's
    warned = [m for _, m in report.warnings]
    assert len(warned) == 1 and warned[0].startswith('measure 7 (row 1) is 200px wide')   # m1, m8: first, last
