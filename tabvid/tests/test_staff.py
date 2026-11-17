"""Barline rule on drawn staves: python -m pytest tabvid/tests -q"""
import json

import cv2
import numpy as np
import pytest

from tabvid.measures import build_measures
from tabvid.report import Report
from tabvid.staff import find_staff

TOP, SP = 200, 22                      # top staff line y, line spacing
BOTTOM = TOP + 5 * SP


def staff_image(bars, width=None, double=()):
    """A white image with one six-line TAB staff and 3px barlines at `bars` (plus double-barline partners)."""
    width = width or bars[-1] + 40
    img = np.full((420, width), 245, np.uint8)
    for i in range(6):
        cv2.line(img, (bars[0], TOP + i * SP), (bars[-1], TOP + i * SP), 90, 1)
    for x in list(bars) + [x + 8 for x in double]:
        img[TOP:BOTTOM + 1, x - 1:x + 2] = 20
    return img


def diagram(img, x_string, gap):
    """Draw a 6-string chord-diagram grid whose third string sits at x_string, bottom `gap` px above the staff."""
    bottom = TOP - 1 - gap
    left = x_string - 2 * 8
    for s in range(6):
        cv2.line(img, (left + 8 * s, bottom - 45), (left + 8 * s, bottom), 20, 1)
    for f in range(6):
        cv2.line(img, (left, bottom - 9 * f), (left + 40, bottom - 9 * f), 20, 1)


def kept(img, x):
    staff = find_staff(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return any(abs(bar.x - x) < 4 for bar in staff.barlines), staff


def test_plain_barlines():
    ok, staff = kept(staff_image((40, 330, 620, 860)), 330)
    assert ok and len(staff.barlines) == 4


def test_stem_below_staff_rejected():
    img = staff_image((40, 330, 620, 860))
    img[TOP:BOTTOM + 3 * SP, 479:482] = 20          # stem through the staff and down to a beam
    ok, staff = kept(img, 480)
    assert not ok and len(staff.barlines) == 4


def test_arrow_above_staff_rejected():
    img = staff_image((40, 330, 620, 860))
    img[TOP - round(1.5 * SP):BOTTOM + 1, 199:202] = 20  # strum arrow poking above the staff
    ok, _ = kept(img, 200)
    assert not ok


@pytest.mark.parametrize('gap', [3, 6, 10, 15, 20, 26, 35])
def test_diagram_string_directly_above_barline_kept(gap):
    img = staff_image((40, 330, 620, 860))
    diagram(img, 330, gap)
    ok, _ = kept(img, 330)
    assert ok, f'barline lost with a diagram {gap}px above the staff'


def test_diagram_touching_staff_counts_as_extension():
    img = staff_image((40, 330, 620, 860))
    diagram(img, 330, 0)
    ok, _ = kept(img, 330)
    assert not ok  # documented limit: ink touching the staff is unbroken from the barline


def test_double_barline_is_one_barline_with_two_strokes():
    ok, staff = kept(staff_image((40, 330, 620, 860), double=(620,)), 624)
    bar = min(staff.barlines, key=lambda b: abs(b.x - 624))
    assert ok and len(bar.strokes) == 2 and bar.left <= 619 and bar.right >= 629


def test_missing_barline_warning(tmp_path):
    img = staff_image((40, 300, 560, 820, 1080, 1340))
    img[TOP:BOTTOM + 1, 819:822] = 245              # erase one barline: 260px measures with one 520px
    for i in range(6):
        img[TOP + i * SP, 819:822] = 90
    staff = find_staff(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    (tmp_path / 'rows').mkdir()
    cv2.imwrite(str(tmp_path / 'rows' / 'row-01.png'), img)
    (tmp_path / 'rows.json').write_text(json.dumps([{
        'row': 1, 'image': 'rows/row-01.png', 'staff_lines': list(staff.lines), 'spacing': staff.spacing,
        'barlines': [{'x': b.x, 'left': b.left, 'right': b.right, 'strokes': [list(s) for s in b.strokes]}
                     for b in staff.barlines]}]))
    (tmp_path / 'debug').mkdir()
    report = Report(tmp_path)
    records = build_measures(tmp_path, report)
    assert len(records) == 4
    assert [r['measure'] for r in records if any(n.startswith('wide') for n in r['notes'])] == [3]
    assert any('missed' in message for _, message in report.warnings)


def test_barline_just_under_the_limit_is_kept_and_reported():
    from tabvid.staff import extension_limit, near_limit
    img = staff_image([40, 300, 560])
    img[TOP - 2:TOP, 299:302] = 20                 # 2 px past the top line: limit is 2.2 px at SP 22
    staff = find_staff(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    assert extension_limit(SP) == pytest.approx(2.2)
    assert [round(b.x) for b in staff.barlines] == [40, 300, 560]
    assert near_limit(staff) == [(300.0, 2)]
    img[TOP - 4:TOP, 299:302] = 20                 # 4 px: a strum arrowhead
    staff = find_staff(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    assert [round(b.x) for b in staff.barlines] == [40, 560]


def _big_staff(height=318, spacing=40, top=50, width=1200):
    img = np.full((height, width), 245, np.uint8)
    for i in range(6):
        img[top + i * spacing] = 20
    for x in (300, 900):
        img[top:top + 5 * spacing + 1, x - 1:x + 1] = 20
    return img


def test_large_staff_in_a_tight_crop_is_found():
    """V6: 40 px spacing in a 318 px crop, over the vendored finder's 10%-of-height limit."""
    staff = find_staff(cv2.cvtColor(_big_staff(), cv2.COLOR_GRAY2RGB))
    assert staff is not None and staff.spacing == pytest.approx(40, abs=1)
    assert [round(b.x) for b in staff.barlines] == [300, 900]


def test_light_on_dark_staff_is_found_after_inversion(tmp_path):
    from tabvid.rows import load_crop
    dark = 255 - cv2.cvtColor(_big_staff(), cv2.COLOR_GRAY2BGR)          # light lines on a dark panel
    path = tmp_path / 'frame.png'
    cv2.imwrite(str(path), dark)
    box = (0, 0, dark.shape[1], dark.shape[0])
    normal = find_staff(load_crop(path, box, ()))
    assert normal is None or not normal.barlines          # at most the dark gaps, as a bare staff
    staff = find_staff(load_crop(path, box, (), inverted=True))
    assert staff is not None and len(staff.barlines) == 2


def test_faint_thin_barline_is_found_but_not_a_faint_stem_or_a_broken_dark_one():
    """Backed by one video (blind D, row 14: a 1 px barline at gray 148, 88% under the ink threshold)."""
    img = staff_image((40, 330, 620, 860))
    rng = np.random.default_rng(0)
    img[TOP:BOTTOM + 1, 480] = rng.choice([146, 148, 151, 153], BOTTOM + 1 - TOP)   # around INK: ~half under it
    ok, staff = kept(img, 480)
    assert ok and len(staff.barlines) == 5

    stem = staff_image((40, 330, 620, 860))
    stem[TOP:BOTTOM + 2 * SP, 480] = 150                     # as faint, but runs on below the staff
    assert not kept(stem, 480)[0]

    broken = staff_image((40, 330, 620, 860))                # V2 row 1: a dark stem between two digits,
    broken[TOP + 12:BOTTOM - 14, 480] = 86                   # dark over 88% of the staff
    broken[TOP:TOP + 12, 480] = 180
    broken[BOTTOM - 14:BOTTOM + 1, 480] = 170
    assert not kept(broken, 480)[0]


def test_wavy_arrow_spanning_the_staff_exactly_is_rejected_but_a_barline_touched_by_a_tie_is_kept():
    """Backed by one video (blind C): arpeggio arrows from the bottom line to the top line, head inside."""
    img = staff_image((40, 330, 620, 860))
    for y in range(TOP, BOTTOM + 1):                           # a 3 px wavy line, 1 px each way:
        x = 479 + round(np.sin((y - TOP) / 3))                 # column 480 is dark in every row
        img[y, x:x + 3] = 20
    ok, staff = kept(img, 480)
    assert not ok and len(staff.barlines) == 4
    assert any('wavy' in why for _, why in staff.rejected)

    tied = staff_image((40, 330, 620, 860))
    cv2.ellipse(tied, (330, TOP + 2 * SP + 4), (30, 8), 0, 0, 180, 20, 2)      # a tie crossing the barline
    cv2.putText(tied, '3', (334, TOP + 4 * SP + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)   # a digit touching it
    assert kept(tied, 330)[0]


def test_narrow_measure_warns_mid_piece_but_not_for_the_first_or_last_measure(tmp_path):
    img = staff_image((40, 140, 500, 860, 960, 1320, 1680, 1780))     # 100 px measures: first, fourth, last
    staff = find_staff(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    (tmp_path / 'rows').mkdir()
    cv2.imwrite(str(tmp_path / 'rows' / 'row-01.png'), img)
    (tmp_path / 'rows.json').write_text(json.dumps([{
        'row': 1, 'image': 'rows/row-01.png', 'staff_lines': list(staff.lines), 'spacing': staff.spacing,
        'barlines': [{'x': b.x, 'left': b.left, 'right': b.right, 'strokes': [list(s) for s in b.strokes]}
                     for b in staff.barlines]}]))
    (tmp_path / 'debug').mkdir()
    report = Report(tmp_path)
    records = build_measures(tmp_path, report)
    assert [r['measure'] for r in records if 'narrow' in r['notes']] == [1, 4, 7]
    narrow = [message for _, message in report.warnings if 'row median' in message]
    assert len(narrow) == 1 and narrow[0].startswith('measure 4 ')


def thick_staff(line_px, spacing, height=300):
    rgb = np.full((height, 1200, 3), 255, np.uint8)
    top = (height - 5 * spacing) // 2
    for k in range(6):
        y = top + k * spacing
        rgb[y - line_px // 2:y - line_px // 2 + line_px, 0:1200] = 0
    rgb[top:top + 5 * spacing + 1, 0:12] = 0                  # thick opening and closing bars
    rgb[top:top + 5 * spacing + 1, 1188:1200] = 0
    rgb[top:top + 5 * spacing + 1, 600:603] = 0
    return rgb, top


def test_staff_with_thick_lines_is_found():
    """Single example (blind2 H): 6-7 px staff lines at a 40 px spacing; the finder took any run
    over 5 px rows as "not a line" and found no staff in any frame."""
    rgb, top = thick_staff(7, 40)
    staff = find_staff(rgb)
    assert staff is not None and abs(staff.lines[0] - top) <= 4 and abs(staff.spacing - 40) <= 1
    assert len(staff.barlines) >= 2
    assert find_staff(thick_staff(12, 30)[0]) is None          # bands, not lines: still no staff


def _short_staff(final):
    """A 1900 px crop whose staff runs only 230 px (12%): an opening barline, a chord, an ending barline."""
    img = np.full((420, 1900), 245, np.uint8)
    for i in range(6):
        cv2.line(img, (80, TOP + i * SP), (310, TOP + i * SP), 90, 1)
    img[TOP:BOTTOM + 1, 80:82] = 20
    cv2.putText(img, '7', (140, TOP + 3 * SP + 6), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)
    img[TOP:BOTTOM + 1, 296:298] = 20                     # thin stroke...
    if final:
        img[TOP:BOTTOM + 1, 303:311] = 20                 # ...and a thick one: a final barline
    return img


def _row_measures(tmp_path, img):
    staff = find_staff(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    (tmp_path / 'rows').mkdir(exist_ok=True)
    (tmp_path / 'debug').mkdir(exist_ok=True)
    cv2.imwrite(str(tmp_path / 'rows' / 'row-01.png'), img)
    (tmp_path / 'rows.json').write_text(json.dumps([{
        'row': 1, 'image': 'rows/row-01.png', 'staff_lines': list(staff.lines), 'spacing': staff.spacing,
        'barlines': [{'x': b.x, 'left': b.left, 'right': b.right, 'strokes': [list(s) for s in b.strokes]}
                     for b in staff.barlines]}]))
    report = Report(tmp_path)
    return build_measures(tmp_path, report), report


def test_short_staff_is_kept_only_when_it_ends_on_a_final_barline(tmp_path):
    """K: the last page holds one measure over 12% of the crop width; lines that short are found only
    when nothing else is, and kept only if the staff ends on a final barline. Its staff starts at the
    opening barline, so no empty 'measure' is cut before it."""
    staff = find_staff(cv2.cvtColor(_short_staff(True), cv2.COLOR_GRAY2RGB))
    assert staff is not None and len(staff.barlines) == 2 and len(staff.barlines[-1].strokes) == 2
    assert find_staff(cv2.cvtColor(_short_staff(False), cv2.COLOR_GRAY2RGB)) is None
    records, report = _row_measures(tmp_path, _short_staff(True))
    assert len(records) == 1 and records[0]['ending_final']
    assert not any('missing its ending' in m for _, m in report.warnings)


def test_piece_ending_is_a_fact_and_a_double_barline_counts_as_an_ending(tmp_path):
    single, double = staff_image((40, 400, 760)), staff_image((40, 400, 760), width=820, double=(760,))
    final = staff_image((40, 400, 760), width=820)
    final[TOP:BOTTOM + 1, 766:774] = 20                   # thick stroke after the last thin one
    for img, ending in ((single, 'single barline'), (double, 'double barline'), (final, 'final barline')):
        records, report = _row_measures(tmp_path, img)
        assert report.stages['measures']['piece_ending'] == ending
        assert not any('ending' in m for _, m in report.warnings), 'the barline alone never warns'


def test_empty_unfinished_measure_after_the_last_barline_is_dropped_with_its_number(tmp_path):
    """E: the staff runs on past the last barline to the crop edge with nothing but staff lines and the
    next measure's number; it is not a measure, and its number no longer prints after the last barline."""
    img = staff_image((40, 400, 760), width=1100)
    for i in range(6):
        cv2.line(img, (760, TOP + i * SP), (1099, TOP + i * SP), 90, 1)   # staff on to the crop edge
    cv2.putText(img, '3', (757, TOP - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, 120, 1)   # its number at the barline
    records, report = _row_measures(tmp_path, img)
    assert len(records) == 2
    last = cv2.imread(str(tmp_path / records[-1]['image']), 0)
    assert last.shape[1] <= 760 - 400 + 4, 'the number of the dropped measure is not in the last measure'
    assert 'dropped' in report.stages['measures']['trailing_empty_measure']
    img = staff_image((40, 400, 760), width=1100)
    for i in range(6):
        cv2.line(img, (760, TOP + i * SP), (1099, TOP + i * SP), 90, 1)
    cv2.putText(img, '5', (900, TOP + 2 * SP + 6), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)   # a note there
    records, report = _row_measures(tmp_path, img)
    assert report.stages['measures']['trailing_empty_measure'] == 'none'


def test_music_after_the_last_barline_is_kept_as_a_measure_and_warned(tmp_path):
    """Blind 4: a closing barline that is not recognised left the row's last measure in no cut, and it
    was dropped without a word (the staff ended inside the crop, so the crop-edge check was silent)."""
    img = staff_image((40, 400, 760), width=1100)
    for i in range(6):
        cv2.line(img, (760, TOP + i * SP), (1000, TOP + i * SP), 90, 1)   # the staff ends inside the crop
    cv2.putText(img, '5', (880, TOP + 2 * SP + 6), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)   # a note there
    records, report = _row_measures(tmp_path, img)
    assert len(records) == 3 and records[-1].get('open_end') and not records[0].get('open_end')
    assert records[-1]['staff_x1'] - records[-1]['staff_x0'] >= 1000 - 762
    warned = [(stage, m) for stage, m in report.warnings if 'after its last barline' in m]
    assert len(warned) == 1 and warned[0][0] == 'measures' and 'crop edge' not in warned[0][1]
    from tabvid import structure
    from tabvid.measures import staff_extent
    import json
    rows = json.loads((tmp_path / 'rows.json').read_text())
    rows[0]['gray'] = cv2.imread(str(tmp_path / 'rows/row-01.png'), 0)
    rows[0]['staff_start'], rows[0]['staff_end'] = staff_extent(rows[0]['gray'], rows[0]['staff_lines'], SP)
    before = len(report.warnings)
    structure.check(rows, records, False, report)
    assert len(report.warnings) == before, 'the structure check does not warn about it again'
