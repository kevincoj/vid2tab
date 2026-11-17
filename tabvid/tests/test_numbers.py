"""Measure-number check, layers A and B: python -m pytest tabvid/tests -q"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from tabvid import numbers
from tabvid.report import Report

SP, TOP = 20, 100


def record(measure, in_row, boxes, width=400):
    return {'measure': measure, 'row': 1, 'in_row': in_row, 'staff_x0': 10, 'staff_x1': 10 + width,
            'staff_lines': [TOP + i * SP for i in range(6)], 'spacing': SP, 'number_boxes': boxes}


def number_at(x):
    return [x, TOP - 19, x + 12, TOP - 5]       # 14 px tall digit just above the staff


def test_two_numbers_in_one_measure_and_an_unnumbered_one(tmp_path):
    records = [record(1, 1, [number_at(60)]),
               record(2, 2, [number_at(5), number_at(200)]),        # a missed barline inside
               record(3, 3, [number_at(8)]),
               record(4, 4, [[150, TOP - 16, 175, TOP - 6]]),        # "sl." mid-measure: not a number
               record(5, 5, [number_at(6)])]
    report = Report(tmp_path)
    facts = numbers.check(records, report)
    assert facts['measures_with_two_numbers'] == [2]
    assert facts['unnumbered_measures'] == [4]
    assert len(report.warnings) == 2


def test_no_numbers_skips_the_check(tmp_path):
    report = Report(tmp_path)
    facts = numbers.safe_check([record(1, 1, []), record(2, 2, [])], report)
    assert 'skipped' in facts['number_check'] and not report.warnings


def test_a_failure_is_only_a_warning(tmp_path):
    report = Report(tmp_path)
    numbers.safe_check([{'broken': True}], report)
    assert report.warnings and 'skipped' in report.warnings[0][1]


def numbered(values):
    return [{'measure': i + 1, 'number': n} for i, n in enumerate(values)]


def test_sequence_finds_a_misread_extra_missing_and_repeated_measures():
    misreads, problems, proven = numbers.sequence(numbered([1, 2, 3, 9, 5, 6]))
    assert misreads == [(4, 9)] and not problems and {1, 2, 5} <= proven
    _, problems, _ = numbers.sequence(numbered([1, 2, None, None, 4, 5]))      # one measure too many
    assert len(problems) == 1 and 'too many' in problems[0]
    _, problems, proven = numbers.sequence(numbered([1, 2, 4, 5]))              # m2 holds 2 and 3
    assert len(problems) == 1 and 'missing' in problems[0] and 2 not in proven
    _, problems, _ = numbers.sequence(numbered([1, 2, 3, 4, 1, 2, 3, 4]))       # a page shown twice
    assert len(problems) == 1 and 'back' in problems[0]


def test_number_printed_after_a_time_signature_is_the_measures_own_number(tmp_path):
    # Single example (blind C m2, m37, m38): the measure opens with a time signature and its
    # number is printed after it, 2.6 spacings past the start: not "a second number".
    records = [record(1, 1, [number_at(60)]),
               record(2, 2, [number_at(5)]),
               record(3, 3, [number_at(60)]),                        # after a time signature
               record(4, 4, [number_at(8)]),
               record(5, 5, [number_at(5), number_at(200)]),         # a missed barline still shows two
               record(6, 6, [number_at(6)]),
               record(7, 7, [[60, TOP - 40, 72, TOP - 26]])]         # off the number row: not a number
    report = Report(tmp_path)
    facts = numbers.check(records, report)
    assert records[2]['number_box'] == number_at(60)
    assert facts['measures_with_two_numbers'] == [5]
    assert facts['unnumbered_measures'] == [7]


def _systems(numbered_rows):
    """Notation + TAB: rows of four measures, numbers printed only over the header at the start of
    the listed rows (recorded as lead blocks: a digit and its pale halo overlap)."""
    records = []
    for row in range(4):
        for k in range(4):
            r = dict(record(4 * row + k + 1, k + 1, []), row=row + 1)
            if k == 0 and row in numbered_rows:
                r['lead_blocks'] = [[-60, TOP - 219, -48, TOP - 203], [-61, TOP - 220, -47, TOP - 202]]
            records.append(r)
    return records


def test_notation_numbers_printed_only_at_system_starts_are_found():
    records = _systems({1, 2, 3})
    assert numbers.find_numbers(records) is not None
    assert [r['measure'] for r in records if r['number_box']] == [5, 9, 13]
    assert all(r['number_box'] == [-61, TOP - 220, -47, TOP - 202] for r in records if r['number_box'])
    assert numbers.find_numbers(_systems({1, 2})) is not None          # two systems are enough (J)
    assert numbers.find_numbers(_systems({1})) is None                 # one is no sequence


def test_lead_blocks_are_ignored_where_measures_print_their_own_numbers():
    records = _systems({1, 2, 3})
    for r in records:
        if r['in_row'] > 1:
            r['number_boxes'] = [number_at(6)]
    numbers.find_numbers(records)
    assert not any(r.get('system_number') for r in records if r['in_row'] > 1)


def test_row_start_numbers_are_found_when_the_header_stays_as_printed():
    """Blind 4: the header was not cut off (rows differ), so no lead blocks; the dark number printed over
    the clef is recorded as a row-start block and found the same way."""
    records = _systems({1, 2, 3})
    for r in records:
        if 'lead_blocks' in r:
            r['row_start_blocks'] = r.pop('lead_blocks')
    assert numbers.find_numbers(records) is not None
    assert [r['measure'] for r in records if r['number_box']] == [5, 9, 13]


def test_row_start_numbers_reveal_a_measure_lost_in_every_row():
    """Blind 4 video 2 before the fix: each row lost its last measure, so the numbers printed at the row
    starts (6, 10, 14, ...) ran ahead of the measure count by one more each row."""
    records = [{'measure': m, 'number': None} for m in range(1, 17)]
    for m, printed in ((6, 6), (9, 10), (12, 14), (15, 18)):       # rows of 3 found, 4 printed
        records[m - 1]['number'] = printed
    misreads, problems, _ = numbers.sequence(records)
    assert not misreads and len(problems) == 3
    assert all('1 measure(s) missing' in p for p in problems)


def test_a_short_mark_beside_the_digits_is_not_a_glyph():
    """Blind 4 video 1: the brace's top tip touches the row-start number's block (a comma-like mark
    low beside the digits); every digit spans the window's height, the tip does not."""
    import numpy as np
    v = np.zeros((20, 30), np.float32)
    v[1:19, 2:6] = 1                                    # a "1"
    v[1:19, 10:18] = 1                                  # a wide digit
    v[13:19, 22:25] = 1                                 # the tip: low and short
    parts = numbers.glyphs(v, np.ones_like(v, bool))
    assert len(parts) == 2


def test_harvest_source_takes_an_offset_and_a_range():
    from pathlib import Path
    assert numbers._harvest_source('runs/x') == (Path('runs/x'), 0, 1, 10 ** 6)
    assert numbers._harvest_source('runs/x:-1') == (Path('runs/x'), -1, 1, 10 ** 6)
    assert numbers._harvest_source('runs/x:-8:17-31') == (Path('runs/x'), -8, 17, 31)


def test_a_digit_as_wide_as_the_window_is_high_is_read_whole_when_split_halves_do_not_read():
    """Blind 4 video 2: an italic 6 (soft edges included) 16 px wide in a 16 px window was cut as two
    touching digits, and neither half is a digit. "10" still reads as two digits."""
    from tabvid import numbers as N
    font = N.load_fonts()['serif-italic']
    six = cv2.resize(font[6], (N.GLYPH_H, N.GLYPH_H), interpolation=cv2.INTER_LINEAR)   # square: wider than SPLIT
    assert six.shape[1] > N.SPLIT * six.shape[0]
    pad = np.zeros((N.GLYPH_H, 2), np.float32)
    v = np.hstack([pad, six, pad]); known = np.ones(v.shape, bool)
    assert len(N.glyphs(v, known)) == 2                       # cut as two...
    assert N.read_window(v, known, font)[0] == 6              # ...but read whole
    gap = np.zeros((N.GLYPH_H, 3), np.float32)
    ten = np.hstack([pad, font[1], gap, font[0], pad]); known = np.ones(ten.shape, bool)
    assert N.read_window(ten, known, font)[0] == 10


def test_two_touching_digits_are_not_read_whole_as_one_wide_digit():
    """V1 m33: a touching "33" as wide as it is high read whole as a "0" (0.75), which is far narrower."""
    from tabvid import numbers as N
    font = N.load_fonts()['serif-italic']
    zero = cv2.resize(font[0], (round(1.5 * font[0].shape[1]), N.GLYPH_H), interpolation=cv2.INTER_LINEAR)
    pad = np.zeros((N.GLYPH_H, 2), np.float32)
    v = np.hstack([pad, zero, pad]); known = np.ones(v.shape, bool)
    assert N.read_window(v, known, font)[0] is None


def _numbered(reads, monkeypatch, tmp_path):
    records = [{'measure': k + 1, 'row': 1, 'in_row': k + 2, 'spacing': 20, 'staff_lines': [100 + 20 * i for i in range(6)],
                'staff_x0': 10, 'staff_x1': 400, 'number_boxes': [[5, 70, 15, 84]]} for k in range(len(reads))]

    def read(records, run_dir, fonts):
        for r, n in zip(records, reads):
            r['number'] = n
        return 'drawn', []
    monkeypatch.setattr(numbers, 'read_numbers', read)
    report = Report(tmp_path)
    facts = numbers.check(records, report, run_dir=tmp_path, fonts={})
    return facts, [w for _, w in report.warnings]


def test_readings_out_of_step_everywhere_are_one_warning_not_many(monkeypatch, tmp_path):
    facts, warnings = _numbered([1, 1, 1, 3, 3, 3, 1, 1, 1, 3, 3, 3, 1, 1], monkeypatch, tmp_path)
    assert len(warnings) == 1 and 'could not be read reliably' in warnings[0]
    assert facts['proven_single'] == []


def test_a_real_gap_in_good_readings_is_still_warned(monkeypatch, tmp_path):
    facts, warnings = _numbered([1, 2, 3, 4, 9, 10, 11, 12, 13, 14], monkeypatch, tmp_path)
    assert len(warnings) == 1 and '4 measure(s) missing' in warnings[0]
