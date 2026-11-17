"""Justified lines and measure numbers: python -m pytest tabvid/tests -q"""
import cv2
import numpy as np

from tabvid import sheet
from tabvid.measures import _blocks, staff_core, _number_owner, CORE_PAD

TOP, SP = 100, 20
LINES = [TOP + i * SP for i in range(6)]


def line_image(width=600):
    img = np.full((260, width), 255, np.uint8)
    for y in LINES:
        img[y] = 120
    img[TOP:LINES[-1] + 1, 10:12] = 20             # opening barline
    img[TOP:LINES[-1] + 1, width - 12:width - 10] = 20
    return img


def test_safe_columns_skip_digits_tie_ends_and_pale_numbers_but_take_a_tie_top():
    img = line_image()
    cv2.putText(img, '7', (100, LINES[2] + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)   # digit on a line
    cv2.ellipse(img, (250, TOP - 8), (40, 8), 0, 180, 360, 30, 2)                       # tie above the staff
    cv2.putText(img, '12', (400, TOP - 25), cv2.FONT_HERSHEY_SIMPLEX, .5, 190, 1)      # pale measure number
    safe, _ = sheet.safe_columns(img, LINES, 0, img.shape[1], SP)
    on_line = np.zeros(img.shape[0], bool)
    for y in LINES:
        on_line[y - 1:y + 2] = True
    glyph = (img < 220)[~on_line].any(axis=0)                 # columns with any non-staff-line ink
    glyph[205:296] = False                                     # the tie is checked on its own below
    clear = round(sheet.SAFE_CLEAR * SP)
    near_glyph = np.convolve(glyph, np.ones(2 * clear + 1), 'same') > 0
    assert not (safe & near_glyph).any()
    assert safe[235:266].all(), 'the flat top of a tie is widened...'
    assert not safe[205:217].any() and not safe[284:296].any(), '...its sloped ends are not'
    number = np.flatnonzero(glyph[390:440]) + 390
    assert not safe[number[0]:number[-1] + 1].any(), 'the gap inside "12" must not be widened'
    assert safe[150:190].all() and safe[480:560].all()


def test_safe_columns_take_beams_and_straight_lines_but_not_stems_or_slides():
    img = line_image(800)
    img[LINES[-1] + 30:LINES[-1] + 34, 100:300] = 20          # a beam...
    for x in (100, 200, 300):
        img[LINES[-1]:LINES[-1] + 34, x] = 20                 # ...with its stems
    img[TOP - 30, 400:700] = 40                               # a straight let-ring line
    cv2.line(img, (450, LINES[1] + 6), (480, LINES[1] - 6), 20, 2)   # a slide mark
    safe, _ = sheet.safe_columns(img, LINES, 0, img.shape[1], SP)
    clear = round(sheet.SAFE_CLEAR * SP)
    assert safe[110:190].all() and safe[210:290].all()        # beam between stems
    for x in (100, 200, 300):
        assert not safe[x - clear:x + clear + 1].any()        # stems are never widened
    assert safe[410:440].all() and safe[490:690].all()        # let-ring line
    assert not safe[450 - clear:480 + clear + 1].any()        # the slide mark is not


def test_justify_caps_each_gap_and_pads_the_rest_before_barlines():
    img = line_image(400)
    for x in range(60, 380, 16):                              # stems every 16px: gaps of 6 safe columns
        img[TOP - 20:LINES[-1] + 20, x] = 20
    line = {'canvas': img, 'staff_x0': 10, 'staff_x1': 390, 'lines': LINES, 'spans': [(10, 390, 388)],
            'notes': {}}
    out, x0, x1, info = sheet.justify(line, 1200, SP)
    assert x1 - x0 == 1200 and out.shape[1] == img.shape[1] + 820
    cap = sheet.GAP_GROW * SP
    gaps = 1 + (380 - 60) // 16                               # every gap reached its cap...
    assert info['barline_pad_px'] == round(820 - gaps * cap)  # ...and the rest went before the barline
    for y in LINES:
        assert (out[y, x0:x1] < 200).all()


def test_justify_reaches_exact_width_and_staff_lines_run_on():
    img = line_image()
    cv2.putText(img, '3', (200, LINES[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)
    line = {'canvas': img, 'staff_x0': 10, 'staff_x1': 590, 'lines': LINES, 'spans': [(10, 590, 578)],
            'notes': {}}
    out, x0, x1, info = sheet.justify(line, 800, SP)
    assert x1 - x0 == 800 and out.shape[1] == img.shape[1] + 220
    for y in LINES:                                  # every staff line is unbroken across the staff
        assert (out[y, x0:x1] < 200).all()
    assert info['padded'] == [] and abs(info['stretch'] - 800 / 580) < 1e-9


def test_measure_without_safe_columns_is_padded_before_its_barline():
    img = line_image(200)
    for x in range(20, 186, 4):                      # stems everywhere: no safe column
        img[TOP - 20:LINES[-1] + 20, x] = 20
    line = {'canvas': img, 'staff_x0': 10, 'staff_x1': 190, 'lines': LINES, 'spans': [(10, 190, 188)],
            'notes': {}}
    out, x0, x1, info = sheet.justify(line, 220, SP)
    assert info['padded'] == [0] and out.shape[1] == 240   # staff 180px -> 220px: 40px added
    assert (out[TOP - 20, 188:228] == 255).all()     # blank staff inserted at the barline
    assert (out[LINES[0], 188:228] == 120).all()     # with the staff lines running through it


def test_line_breaks_minimise_stretch_and_avoid_a_lonely_last_measure():
    widths = [100] * 7
    lines = sheet.break_lines(widths, 300)
    assert [len(l) for l in lines] in ([3, 2, 2], [2, 3, 2], [2, 2, 3])   # never 3 3 1


def test_gray_number_with_one_dark_digit_stays_one_block():
    img = line_image(800)
    gray = img.copy()
    x_cut = 400
    gray[TOP:LINES[-1] + 1, x_cut - 1:x_cut + 1] = 20
    cv2.putText(gray, '1', (x_cut - 14, TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, .5, 160, 1)
    cv2.putText(gray, '4', (x_cut + 1, TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, .5, 135, 2)   # darker, bigger digit
    band = (TOP - CORE_PAD, LINES[-1] + CORE_PAD + 1)
    labels, centers, numbers = _blocks(gray, staff_core(gray, band), LINES, SP)
    found = [l for l in numbers if centers[l][1] < x_cut + 20 and centers[l][2] > x_cut - 20]
    assert len(found) == 1, 'the two digits must form one number'
    cuts = [(10, x_cut, None), (x_cut, 790, None)]
    assert _number_owner(cuts, centers[found[0]][0], SP) == 1


def test_thick_lines_leave_safe_columns_and_padding_keeps_the_barline_edge():
    img = line_image(200)
    for y in LINES:
        img[y + 1] = 120                             # 2 px lines...
        img[y + 2] = 200                             # ...with an antialiased third row
    img[TOP:LINES[-1] + 1, 187] = 150                # antialiased left edge of the ending barline
    safe, _ = sheet.safe_columns(img, LINES, 10, 190, SP)
    assert safe.any(), 'a thick staff line must not hide every safe column'
    for x in range(20, 186, 4):                      # now stems everywhere: no safe column
        img[TOP - 20:LINES[-1] + 20, x] = 20
    line = {'canvas': img, 'staff_x0': 10, 'staff_x1': 190, 'lines': LINES, 'spans': [(10, 190, 188)],
            'notes': {}}
    out, _, _, info = sheet.justify(line, 220, SP)
    assert info['padded'] == [0]
    assert out[LINES[0] + 5, 227] == 150 and (out[LINES[0] + 5, 228:230] == 20).all()   # edge stays on its stroke


def test_dark_paper_measure_is_kept_uncleaned(tmp_path):
    """A fade-to-black measure: clean() would trim it; it must come back the same size, not raise."""
    img = np.full((260, 400, 3), 76, np.uint8)
    for y in LINES:
        img[y] = 30
    path = tmp_path / 'm1.png'
    cv2.imwrite(str(path), img)
    out, cleaned = sheet._clean_measure(path)
    assert out.shape == img.shape[:2]
    assert not cleaned


def test_failed_sheet_still_writes_a_pdf(tmp_path):
    from tabvid.__main__ import _sheet
    from tabvid.report import Report
    (tmp_path / 'measures.json').write_text('not json')
    (tmp_path / 'rows.json').write_text('[]')
    report = Report(tmp_path)
    pdf = _sheet(tmp_path, report, 'title')
    assert pdf.exists() and pdf.stat().st_size > 0
    assert any('stage 5 failed' in message for _, message in report.warnings)


def test_barline_padding_keeps_a_leaving_tie_on_its_note_and_a_number_whole():
    """Leftover width goes before the ending barline: a tie fragment leaving the measure stays on
    its note and runs on through the padding; a number printed over the barline moves whole."""
    img = line_image(400)
    for x in range(60, 330, 16):                              # dense stems: little safe room
        img[TOP - 5:LINES[-1] + 20, x] = 20
    cv2.putText(img, '12', (335, LINES[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, .5, 20, 2)
    cv2.ellipse(img, (388, LINES[1] + 2), (24, 8), 0, 200, 270, 20, 2)   # tie from "12" up to the barline
    cv2.putText(img, '9', (384, TOP - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, 90, 1)   # number over the barline
    line = {'canvas': img, 'staff_x0': 10, 'staff_x1': 390, 'lines': LINES, 'spans': [(10, 390, 388)],
            'notes': {}}
    out, x0, x1, info = sheet.justify(line, 900, SP)
    assert info['barline_pad_px'] >= 100 and x1 - x0 == 900
    bar = x1 - 2
    band = (out[LINES[1] - 10:LINES[1] + 2, :bar] < 120).any(axis=0)   # "12", then the tie, then the barline
    ink = np.flatnonzero(band[bar - 400:]) + bar - 400
    ink = ink[np.flatnonzero(np.diff(ink) > 20)[-1] + 1:]               # from the "12" on (stems before)
    gaps = np.diff(ink)
    assert ink[-1] >= bar - 1 and gaps.max() <= 10, 'tie on its note (gap as drawn) and running on to the barline'
    assert (gaps[np.argmax(gaps) + 1:] == 1).all(), 'the tie is unbroken through the padding'
    number = out[TOP - 22:TOP - 4, bar - 12:bar + 12]
    assert (number < 200).sum() > 15, 'the number moved with its barline'
    assert not (out[TOP - 22:TOP - 4, ink[0]:bar - 16] < 200).any(), 'and left nothing behind'


def test_measures_kept_together_never_get_a_line_break_between_them():
    widths = [100] * 6
    free = sheet.break_lines(widths, 300)
    assert [len(l) for l in free] == [3, 3]
    keep = [False, False, True, False, False, False]          # an annotation spans the m3|m4 barline
    lines = sheet.break_lines(widths, 300, keep=keep)
    assert not any(l[-1] == 2 for l in lines[:-1]) and sum(map(len, lines)) == 6
    wide = sheet.break_lines([200, 200, 100], 300, keep=[True, False, False])   # a group wider than a line
    assert wide[0] == [0, 1]


def test_pdf_is_named_after_the_title_and_the_title_is_kept_for_re_runs(tmp_path):
    import json
    from tabvid.report import Report
    run = tmp_path / 'my-run'
    run.mkdir()
    (run / 'measures.json').write_text('[]')
    (run / 'rows.json').write_text('[]')
    (run / 'sheet.pdf').write_text('old')                       # from before PDFs were named
    first = sheet.build_sheet(run, Report(run))                 # no title: the folder name
    assert first.name == 'my-run.pdf' and first.exists() and not (run / 'sheet.pdf').exists()
    titled = sheet.build_sheet(run, Report(run), 'Yesterday: Live / "Take 2"?', artist='The Beatles')
    assert titled.name == 'Yesterday Live Take 2.pdf' and titled.exists()
    assert not first.exists()                                   # one PDF per run folder
    config = json.loads((run / 'config.json').read_text())
    assert (config['title'], config['artist']) == ('Yesterday: Live / "Take 2"?', 'The Beatles')
    again = sheet.build_sheet(run, Report(run))                 # a re-run keeps the saved title
    assert again == titled and sheet.heading(run)[:2] == ('Yesterday: Live / "Take 2"?', 'The Beatles')


def test_title_and_artist_are_printed_on_page_one(tmp_path):
    from tabvid.vendor.render.pdf import build_pdf
    from reportlab.lib.pagesizes import letter
    strip = np.full((200, 1000), 255, np.uint8)
    plain, with_artist = tmp_path / 'a.pdf', tmp_path / 'b.pdf'
    build_pdf([(strip, 20)], str(plain), 'Song', letter, 29, 6)
    build_pdf([(strip, 20)], str(with_artist), 'Song', letter, 29, 6, subtitle='Somebody')
    import fitz

    def lines(path):
        page = fitz.open(str(path))[0]
        return [(span['text'], round(span['size']), round(span['origin'][1]))
                for block in page.get_text('dict')['blocks'] for line in block.get('lines', [])
                for span in line['spans']]

    assert [t for t, _, _ in lines(plain)] == ['Song']
    (title, title_size, title_y), (artist, artist_size, artist_y) = lines(with_artist)
    assert (title, artist) == ('Song', 'Somebody')
    assert artist_size < title_size and artist_y > title_y        # smaller, under the title


def test_left_overhang_wider_than_the_line_so_far_is_clipped_not_a_crash():
    """G m6: a crop-edge line handed whole to the second measure of a line gave it a 470 px left
    overhang; the paste position went negative and stage 5 crashed. Now the paste is clipped
    and the lost ink is noted."""
    def measure(number, in_row, overhang):
        staff = line_image(200)
        img = np.full((260, overhang + 200), 255, np.uint8)
        img[:, overhang:] = staff
        img[20, 5:overhang] = 60                     # ink far out to the left, above the staff
        return {'img': img, 'top': TOP, 'lines': [l - TOP for l in LINES], 'sx0': overhang,
                'sx1': overhang + 200, 'bar0': overhang + 188, 'row': 1, 'in_row': in_row, 'measure': number}
    first = measure(1, 1, 0)
    second = measure(2, 2, 470)
    second['lines'] = first['lines'] = LINES
    line = sheet.compose_line([first, second], SP)
    assert line['staff_x1'] - line['staff_x0'] == 400
    assert line['notes']['clipped'] and line['notes']['clipped'][0][0] == 2
    canvas = line['canvas']
    assert (canvas[LINES[0], line['staff_x0'] + 210:line['staff_x0'] + 380] < 200).all()   # second staff drawn


def _tempo(img, x, y):
    cv2.putText(img, '= 80', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)   # big text: wide gaps inside


def test_safe_columns_never_fall_inside_text_off_the_staff():
    img = line_image(600)
    _tempo(img, 200, TOP - 30)
    group = sheet.text_groups(img, LINES, SP)
    assert len(group) == 1
    gx0, _, gx1, _ = group[0]
    inner = (img[:TOP - 2, gx0:gx1] < 200).any(axis=0)
    assert not inner.all(), 'the drawn text has blank columns inside it (between glyphs)'
    safe, _ = sheet.safe_columns(img, LINES, 10, 590, SP)
    assert not safe[gx0 - 10:gx1 - 10].any()
    assert safe[:gx0 - 10 - SP].any() and safe[gx1 - 10 + SP:].any(), 'clear staff on both sides stays safe'


def test_barline_padding_never_opens_a_gap_inside_text_across_the_barline():
    """L m65: "= 80" printed across the ending barline; the 0 crossed it and moved with it, the 8 stayed.
    Text mostly before the barline stays whole over its measure; the padding goes in after it."""
    img = line_image(400)
    for x in range(16, 384, 4):                              # stems everywhere: padded before the barline
        img[TOP - 5:LINES[-1] + 20, x] = 20
    _tempo(img, 300, TOP - 30)                               # ends past the barline at 388
    (gx0, gy0, gx1, gy1), = sheet.text_groups(img, LINES, SP)
    assert gx0 < 388 < gx1
    line = {'canvas': img, 'staff_x0': 10, 'staff_x1': 390, 'lines': LINES, 'spans': [(10, 390, 388)],
            'notes': {}}
    out, x0, x1, info = sheet.justify(line, 500, SP)
    assert info['padded'] == [0]
    text = (out[gy0:gy1] < 200).any(axis=0)
    columns = np.flatnonzero(text)
    assert columns[-1] - columns[0] == gx1 - gx0 - 1, 'the text keeps its width: no gap opened inside it'


def test_black_digits_with_a_wide_warm_fringe_keep_their_strokes(tmp_path):
    """Blind 4 video 1 (also C, D, V4): black glyphs with dark warm chroma fringes 3+ px wide. clean()
    took the fringe for reddish UI and its dilated removal ate into the stroke; the erased ink now comes
    from a cleaning with those edges made neutral, and nothing else changes."""
    from tabvid.vendor.preprocess.cleaning import clean
    img = np.full((120, 200, 3), 250, np.uint8)
    for i in range(6):
        cv2.line(img, (0, 20 + 16 * i), (199, 20 + 16 * i), (90, 90, 90), 1)
    cv2.putText(img, '5', (60, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (5, 5, 5), 2)
    ink = (img.max(axis=2) < 80).astype(np.uint8)
    ring = cv2.dilate(ink, np.ones((7, 7), np.uint8)).astype(bool) & ~ink.astype(bool)
    img[ring] = (110, 60, 30)                                        # a dark warm fringe, 3 px wide
    cv2.rectangle(img, (150, 0), (153, 119), (40, 200, 230), -1)     # a bright cursor away from ink
    cv2.imwrite(str(tmp_path / 'm.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    old = clean(cv2.copyMakeBorder(img, sheet.BORDER, sheet.BORDER, sheet.BORDER, sheet.BORDER,
                                   cv2.BORDER_CONSTANT, value=(250,) * 3))[sheet.BORDER:-sheet.BORDER, sheet.BORDER:-sheet.BORDER]
    new, ok = sheet._clean_measure(tmp_path / 'm.png')
    core = ink.astype(bool)
    assert ok and (old[core] >= 128).sum() > 10, 'clean() alone erodes the stroke'
    assert (new[core] >= 128).sum() < (old[core] >= 128).sum() / 4
    assert (new[:, 148:156] == old[:, 148:156]).all(), 'a bright colour away from ink is cleaned as before'
    changed = new != old
    assert not changed[~cv2.dilate(ink, np.ones((7, 7), np.uint8)).astype(bool)].any(), 'nothing else changes'
