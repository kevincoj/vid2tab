"""Structure check (stage 4c): python -m pytest tabvid/tests -q"""
import cv2
import numpy as np

from tabvid import structure
from tabvid.report import Report

TOP, SP = 60, 20
LINES = [TOP + i * SP for i in range(6)]


def row(number, bars, digits, width=900, staff_end=None):
    """A row dict as stage 4 holds it: staff lines to `staff_end`, barlines, fret digits (text, x, line)."""
    staff_end = width - 20 if staff_end is None else staff_end
    gray = np.full((220, width), 245, np.uint8)
    for y in LINES:
        gray[y, 20:staff_end] = 60
    for x in bars:
        gray[TOP:LINES[-1] + 1, x:x + 2] = 20
    for text, x, line in digits:
        cv2.putText(gray, text, (x, LINES[line] + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)
    return {'row': number, 'gray': gray, 'staff_lines': LINES, 'spacing': SP, 'staff_start': 20,
            'staff_end': staff_end,
            'barlines': [{'left': x, 'right': x + 1} for x in bars]}


def records(rows):
    out = []
    for r in rows:
        bars = r['barlines']
        for k, (a, b) in enumerate(zip(bars, bars[1:])):
            x0 = a['left'] if k == 0 else a['right'] + 1
            out.append({'measure': len(out) + 1, 'row': r['row'], 'row_x0': x0, 'staff_x0': 0,
                        'staff_x1': b['right'] + 1 - x0, 'spacing': SP})
    return out


def run(rows, tmp_path, consistent=False):
    report = Report(tmp_path)
    found = structure.check(rows, records(rows), consistent, report)
    return found, report


NOTES = [('3', 60, 1), ('5', 150, 3), ('7', 330, 2), ('0', 420, 4), ('2', 600, 0), ('1', 700, 5)]


def test_clean_rows_raise_nothing(tmp_path):
    rows = [row(1, (20, 280, 560, 878), NOTES), row(2, (20, 280, 560, 878), [(t, x + 9, 5 - l) for t, x, l in NOTES])]
    found, report = run(rows, tmp_path)
    assert not any(found.values()) and not report.warnings
    assert report.stages['structure']['structure_check'] == 'clean'


def test_music_past_the_last_barline_to_the_crop_edge_is_a_cut_row(tmp_path):
    cut = row(1, (20, 280, 560), NOTES, staff_end=900)          # staff and a digit run on to the edge
    ends = row(2, (20, 280, 560), NOTES[:4], staff_end=900)     # staff runs on, but empty: not cut
    found, report = run([cut, ends], tmp_path)
    assert found['cut_rows'] == [1]
    assert 'in no measure' in report.warnings[0][1]


def test_page_starting_with_the_end_of_the_previous_one_is_a_seam_repeat(tmp_path):
    first = row(1, (20, 300, 620), [('3', 60, 1), ('5', 150, 3), ('7', 400, 2), ('0', 520, 4)], width=660)
    # The next page starts 180 px before the old page's last barline: the tail of its last measure.
    tail = [(t, x - 440 + 20, l) for t, x, l in [('0', 520, 4)]]
    second = row(2, (20, 200, 500), tail + [('2', 260, 0), ('9', 380, 3)], width=660)
    second['gray'][TOP:LINES[-1] + 1, 20:22] = 245             # no opening barline: the staff starts at the cut
    second['gray'][LINES, 20:22] = 60
    found, _ = run([first, second], tmp_path)
    assert found['seam_repeats'] == [3]
    assert not run([first, second], tmp_path, consistent=True)[0]['seam_repeats']   # the numbers vouch for it


def test_row_repeating_the_previous_one_is_flagged_unless_numbers_are_consistent(tmp_path):
    rows = [row(1, (20, 280, 560, 878), NOTES), row(2, (20, 280, 560, 878), NOTES)]
    found, _ = run(rows, tmp_path)
    assert found['repeated_rows'] == [2]
    assert not run(rows, tmp_path, consistent=True)[0]['repeated_rows']


def test_sliver_with_only_a_time_signature_is_empty_but_a_short_measure_with_a_note_is_not(tmp_path):
    signature = row(1, (20, 70, 400, 878), NOTES[2:])
    for text, y in (('3', LINES[2] + 16), ('4', LINES[4] + 16)):   # tall digits, two spacings each
        cv2.putText(signature['gray'], text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 20, 4)
    pickup = row(2, (20, 70, 400, 878), [('3', 35, 1)] + NOTES[2:])
    found, _ = run([signature, pickup], tmp_path)
    assert found['empty_slivers'] == [1]


def test_a_failure_is_only_a_warning(tmp_path):
    report = Report(tmp_path)
    assert structure.safe_check([{'row': 1}], [{'row': 1}], False, report) == {}
    assert 'structure check failed' in report.warnings[0][1]
