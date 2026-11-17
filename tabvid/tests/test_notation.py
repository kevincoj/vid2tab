"""Notation + TAB systems on drawn staves: python -m pytest tabvid/tests -q"""
import cv2
import numpy as np

from tabvid.staff import find_staff, notation_staff

TOP, SP = 300, 24                       # TAB top line y, TAB line spacing
BOTTOM = TOP + 5 * SP
N_SP = 16                               # notation line spacing
N_TOP = TOP - 200                       # notation top line y
N_BOTTOM = N_TOP + 4 * N_SP
BARS = (60, 460, 860, 1260)


def system(bars=BARS, through=(), notation=True, width=1320, height=480):
    """A notation staff over a TAB staff (gray image). Barlines per staff at `bars`; at `through`,
    one stroke from the notation top line down to the TAB bottom line."""
    img = np.full((height, width), 245, np.uint8)
    for i in range(6):
        cv2.line(img, (bars[0], TOP + i * SP), (bars[-1], TOP + i * SP), 90, 1)
    if notation:
        for i in range(5):
            cv2.line(img, (bars[0], N_TOP + i * N_SP), (bars[-1], N_TOP + i * N_SP), 90, 1)
    for x in bars:
        if x in through:
            img[N_TOP:BOTTOM + 1, x - 1:x + 2] = 20
            continue
        img[TOP:BOTTOM + 1, x - 1:x + 2] = 20
        if notation:
            img[N_TOP:N_BOTTOM + 1, x - 1:x + 2] = 20
    return img


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def test_notation_found_over_the_tab():
    img = system()
    found = notation_staff(rgb(img), find_staff(rgb(img)))
    assert found is not None and len(found.lines) == 5 and abs(found.spacing - N_SP) < 1
    assert abs(found.lines[0] - N_TOP) <= 1


def test_no_notation_on_tab_only():
    img = system(notation=False)
    assert notation_staff(rgb(img), find_staff(rgb(img))) is None


def test_diagram_fret_lines_are_not_a_staff():
    img = system(notation=False)
    for x in range(100, 1260, 110):                   # a row of chord diagrams: 5 fret lines each
        for f in range(5):
            cv2.line(img, (x, 100 + 9 * f), (x + 40, 100 + 9 * f), 20, 1)
        for s in range(6):
            cv2.line(img, (x + 8 * s, 100), (x + 8 * s, 136), 20, 1)
    assert notation_staff(rgb(img), find_staff(rgb(img))) is None


def test_notation_found_with_partners():
    img = system()
    img[N_TOP:N_BOTTOM + 1, 700] = 20                 # a 1 px stem spanning the notation staff exactly
    staff = find_staff(rgb(img))
    found = notation_staff(rgb(img), staff)
    assert found is not None and len(found.lines) == 5 and abs(found.spacing - N_SP) < 1
    assert all(p is not None for p in found.partners)
    assert found.loose == (700,)


def test_missing_partner():
    img = system()
    img[N_TOP:N_BOTTOM + 1, 459:462] = 245            # no notation stroke over the second TAB barline
    for i in range(5):
        img[N_TOP + i * N_SP, 459:462] = 90
    staff = find_staff(rgb(img))
    found = notation_staff(rgb(img), staff)
    assert found.partners[1] is None and sum(p is None for p in found.partners) == 1


def test_system_barline_through_both_staves():
    img = system(through=(460, 860))
    staff = find_staff(rgb(img))
    assert [round(b.x) for b in staff.barlines] == list(BARS)
    assert [b.system for b in staff.barlines] == [False, True, True, False]
    found = notation_staff(rgb(img), staff)
    assert all(p is not None for p in found.partners)


def test_stroke_stopping_in_the_gap_fails():
    img = system()
    img[N_BOTTOM + 40:TOP, 659:662] = 20              # leaves the TAB staff upward, stops between the staves
    img[TOP:BOTTOM + 1, 659:662] = 20
    staff = find_staff(rgb(img))
    assert all(abs(b.x - 660) > 4 for b in staff.barlines)
    assert any(abs(x - 660) < 4 and why == 'extends above staff' for x, why in staff.rejected)


def test_stroke_past_the_notation_top_fails():
    img = system()
    img[N_TOP - 30:BOTTOM + 1, 659:662] = 20          # through both staves and on above them (a playhead)
    staff = find_staff(rgb(img))
    assert all(abs(b.x - 660) > 4 for b in staff.barlines)


def test_tab_only_rule_unchanged_without_notation():
    img = system(notation=False)
    img[TOP - 150:BOTTOM + 1, 659:662] = 20           # a long stroke up from the TAB with no staff above
    staff = find_staff(rgb(img))
    assert all(abs(b.x - 660) > 4 for b in staff.barlines)


def ledger_note(img, x, y, half=10):
    """A whole note on a ledger line at (x, y): the ledger 2 * half + 12 px wide, an ellipse on it."""
    cv2.line(img, (x - half - 6, y), (x + half + 6, y), 30, 1)
    cv2.ellipse(img, (x, y), (half, 6), 0, 0, 360, 20, 3)


def test_ledger_notes_tied_over_a_barline_are_cut_with_the_staff():
    from tabvid.measures import CORE_PAD, staff_core, system_core
    img = system()
    y = N_BOTTOM + 2 * N_SP                           # second ledger line below the notation staff
    ledger_note(img, 400, y)
    ledger_note(img, 500, y)
    cv2.ellipse(img, (450, y + 4), (42, 10), 0, 0, 180, 20, 2)   # tie joining them under the barline at 460
    band = (TOP - CORE_PAD, BOTTOM + CORE_PAD + 1)
    notation = {'lines': [N_TOP + i * N_SP for i in range(5)], 'spacing': N_SP}
    tab_only, both = staff_core(img, band), system_core(img, band, notation, SP)
    assert not tab_only[y, 400] and not tab_only[y, 500]          # before: an annotation block over the barline
    assert both[y, 400] and both[y, 500] and both[y + 14, 450]    # now: cut straight, the tie split at 460


def test_beams_and_fingering_are_not_ledger_lines():
    from tabvid.measures import ledger_lines
    img = system()
    y = N_TOP - 2 * N_SP
    img[y - 4:y + 4, 200:300] = 20                    # a beam at a ledger position (thick)
    cv2.putText(img, '4', (600, y + 6), cv2.FONT_HERSHEY_SIMPLEX, .5, 20, 1)   # a fingering digit
    mask = ledger_lines(img, [N_TOP + i * N_SP for i in range(5)], N_SP)
    assert not mask[:, 200:300].any() and not mask[:, 595:620].any()


def header_rows(key=(100, 112), changed=None, count=4):
    """Rows of a notation + TAB system: brace, clef, a key of two sharps at `key` (a third sharp in row
    `changed`), then music that opens every row alike (a repeated phrase), then music that differs."""
    from tabvid.measures import staff_extent
    rows = []
    for n in range(1, count + 1):
        img = system()
        img[N_TOP - 10:BOTTOM + 10, 44:48] = 20                      # brace / system line
        cv2.ellipse(img, (75, N_TOP + 2 * N_SP), (10, 30), 0, 0, 360, 20, 3)        # clef
        cv2.putText(img, 'T', (66, TOP + 2 * SP), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)   # TAB clef
        for x in key + ((124,) if n == changed else ()):
            img[N_TOP + 4:N_TOP + 20, x:x + 6] = 20                 # sharps
        if n == 1:
            img[N_TOP + 2:N_TOP + 4 * N_SP - 2, 126:138] = 20       # time signature, printed once
        img[N_TOP + N_SP - 5:N_TOP + N_SP + 5, 170:184] = 20         # the same first note in every row
        img[N_TOP + 2 * N_SP - 5:N_TOP + 2 * N_SP + 5, 260 + 20 * n:274 + 20 * n] = 20   # then music differs
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lines = [TOP + i * SP for i in range(6)]
        rows.append({'row': n, 'image_bgr': bgr, 'gray': img.copy(), 'staff_lines': lines, 'spacing': SP,
                     'notation_lines': [N_TOP + i * N_SP for i in range(5)], 'notation_spacing': N_SP,
                     'staff_start': staff_extent(img, lines, SP)[0]})
    return rows


def test_header_ends_at_the_gap_before_music_repeated_at_row_starts(tmp_path):
    from tabvid import measures
    (tmp_path / 'measures').mkdir()
    rows = header_rows()
    header = measures.find_header(rows)
    end = rows[0]['staff_start'] + header['x_off'] + header['mask'].shape[1]
    assert 118 <= end <= 124                         # after the key (ends at 118), before the shared note at 170
    note, warning = measures.strip_headers(rows, tmp_path)
    assert warning is None and all(r['cut_start'] == end for r in rows)
    assert (rows[1]['gray'][N_TOP + 3:N_TOP + 14, 100:118] > 200).all()   # the key is gone from every row
    assert (rows[1]['gray'][N_TOP + N_SP - 4, 172:182] < 100).all()   # the music is not
    assert (tmp_path / 'measures' / 'header.png').exists() and (tmp_path / 'header.json').exists()


def test_a_row_with_a_different_header_keeps_every_header_as_printed(tmp_path):
    from tabvid import measures
    (tmp_path / 'measures').mkdir()
    rows = header_rows(changed=3)
    note, warning = measures.strip_headers(rows, tmp_path)
    assert warning and 'rows [3]' in warning
    assert not any('cut_start' in r for r in rows) and not (tmp_path / 'header.json').exists()


def test_line_breaks_count_the_header_on_every_line():
    from tabvid import sheet
    widths = [300, 280, 320, 300, 290, 310]
    assert sheet.break_lines(widths, 1000) == sheet.break_lines(widths, 1000, lead=0)
    with_header = sheet.break_lines(widths, 1000, lead=150)
    assert all(150 + sum(widths[i] for i in line) <= 1000 for line in with_header)
    assert len(with_header) > len(sheet.break_lines(widths, 1000))


def _measure(row, in_row, number, lead=()):
    """A measure for compose_line: TAB and notation staff, a barline at its end, a number left of it."""
    sx0 = 40 if lead else 0
    img = np.full((480, sx0 + 300), 255, np.uint8)
    for i in range(6):
        img[TOP + i * SP, sx0:] = 90
    for i in range(5):
        img[N_TOP + i * N_SP, sx0:] = 90
    img[TOP:BOTTOM + 1, -2:] = img[N_TOP:N_BOTTOM + 1, -2:] = 20
    for x0, y0, x1, y1 in lead:
        img[y0:y1, x0:x1] = 30
    return {'img': img, 'top': TOP, 'lines': [TOP + i * SP for i in range(6)],
            'nlines': [N_TOP + i * N_SP for i in range(5)], 'sx0': sx0, 'sx1': sx0 + 300, 'bar0': sx0 + 298,
            'row': row, 'in_row': in_row, 'measure': number, 'locks': [], 'lead': [list(b) for b in lead]}


def test_header_opens_every_line_and_a_row_start_mid_line_takes_its_number_over_its_barline():
    from tabvid import sheet
    header = {'img': np.full((480, 120), 255, np.uint8), 'top': TOP, 'width': 120,
              'lines': [TOP + i * SP for i in range(6)], 'nlines': [N_TOP + i * N_SP for i in range(5)]}
    header['img'][N_TOP:BOTTOM, 10:14] = 0                          # brace
    number = (5, 60, 15, 75)                                        # printed over the header at the row start
    first = _measure(1, 4, 4)
    opener = _measure(2, 1, 5, lead=[number])
    line = sheet.compose_line([first, opener], SP, header)
    canvas = line['canvas']
    assert (canvas[N_TOP:BOTTOM, 10:14] == 0).all() and line['staff_x0'] == 0
    assert line['locks'][0] == (0, 120)
    start = line['spans'][1][0]                                     # the opener's barline on this line
    ink = np.flatnonzero((canvas[60:75] < 100).any(axis=0))
    assert abs((ink[0] + ink[-1]) / 2 - start) <= 6                # the number now sits over its start
    alone = sheet.compose_line([opener], SP, header)                # opening a line: printed as at the source
    ink = np.flatnonzero((alone['canvas'][60:75] < 100).any(axis=0))
    assert ink[0] == 120 - 40 + 5


def _line(extra=None):
    """A composed notation + TAB line (three measures) for justify()."""
    img = system()
    if extra:
        extra(img)
    return {'canvas': img, 'staff_x0': BARS[0], 'staff_x1': BARS[-1] + 2,
            'lines': [TOP + i * SP for i in range(6)], 'nlines': [N_TOP + i * N_SP for i in range(5)],
            'spans': [(BARS[k] + 2, BARS[k + 1] + 2, BARS[k + 1] - 1) for k in range(3)], 'locks': []}


def _strokes(out, top, bottom):
    """Columns dark over 90% of a staff band: its barlines."""
    full = np.flatnonzero((out[top:bottom + 1] < 100).mean(axis=0) >= .9)
    return [int(run[0]) for run in np.split(full, np.flatnonzero(np.diff(full) > 1) + 1)] if len(full) else []


def _head_widths(out, y, x0, x1):
    """Widths of the ink runs in the notehead rows around y (the staff line through them left out,
    barlines too)."""
    band = out[y - 5:y + 6, x0:x1] < 100
    band[4:7] = False                                 # the staff line (and the tie under it is lower)
    head = band.any(axis=0) & ~(out[N_TOP:N_BOTTOM + 1, x0:x1] < 100).all(axis=0)
    xs = np.flatnonzero(head)
    return [len(r) for r in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)] if len(xs) else [0]


def _note(img, x, y, tie_to=None):
    cv2.ellipse(img, (x, y), (8, 6), -20, 0, 360, 20, -1)
    if tie_to:                                    # a tie from the note to the barline (leaving the measure)
        cv2.line(img, (x + 8, y + 7), (tie_to, y + 7), 20, 2)


def test_justify_widens_only_columns_clear_in_both_staves_and_both_staves_run_on():
    from tabvid import sheet
    line = _line(lambda img: [_note(img, 250, N_TOP + 2 * N_SP), _note(img, 650, N_TOP + N_SP)])
    every = sorted(line['lines'] + line['nlines'])
    safe, _ = sheet.safe_columns(line['canvas'], every, BARS[0], BARS[-1], SP)
    assert not safe[250 - BARS[0]] and not safe[650 - BARS[0]]      # a notehead over an empty TAB stretch
    for stretch in (1.2, 1.6):                                       # gaps only; then padding before barlines too
        target = round((line['staff_x1'] - line['staff_x0']) * stretch)
        out, x0, x1, info = sheet.justify(line, target, SP)
        assert x1 - x0 == target
        for y in every:                                              # both staves' lines run on through it all
            assert (out[y, x0:x1] < 200).all()
        assert _strokes(out, N_TOP, N_BOTTOM) == _strokes(out, TOP, BOTTOM)   # barlines stay one over the other
        assert _head_widths(out, N_TOP + 2 * N_SP, x0, x1)[0] <= 18   # the notehead is never widened
    assert info['barline_pad_px'] > 0


def test_padding_keeps_a_tie_leaving_its_note_on_the_note_and_running_to_the_barline():
    from tabvid import sheet
    y = N_TOP + N_SP
    line = _line(lambda img: [_note(img, 300 + 120 * k, y) for k in range(1)] + [_note(img, 420, y, tie_to=459)]
                 + [img.__setitem__((slice(N_TOP - 40, N_TOP - 36), slice(300, 440)), 20)]   # a beam over them
                 + [img.__setitem__((slice(N_TOP - 38, y), slice(x + 7, x + 9)), 20) for x in (300, 420)])  # stems
    target = round((line['staff_x1'] - line['staff_x0']) * 1.6)
    out, x0, x1, info = sheet.justify(line, target, SP)
    bar = _strokes(out, N_TOP, N_BOTTOM)[1]
    tie = out[y + 6:y + 9, :bar]
    assert (tie.min(axis=0)[bar - 30:bar] < 100).all()               # the tie runs on to the barline
    assert all(w <= 20 for w in _head_widths(out, y, x0, bar)[:2])  # noteheads whole, not torn or widened


def test_system_barlines_at_the_ends_of_the_system_count():
    """Blind 4: the closing barline drawn through both staves has notation lines on its left only (the
    system ends there); so does the opening one on its right. They count when the TAB staff ends there
    too. Each row had lost its last measure."""
    img = system(through=BARS)
    staff = find_staff(rgb(img))
    assert [round(b.x) for b in staff.barlines] == list(BARS)
    assert all(b.system for b in staff.barlines)
    edge = system(through=BARS, width=BARS[-1] + 6)          # the crop ends just past the closing barline
    assert [round(b.x) for b in find_staff(rgb(edge)).barlines] == list(BARS)


def test_one_sided_stroke_fails_where_the_tab_staff_runs_on():
    img = system(through=(460, 860), width=1500)
    for i in range(6):
        cv2.line(img, (BARS[-1], TOP + i * SP), (1460, TOP + i * SP), 90, 1)   # TAB runs on past 1260...
    img[TOP:BOTTOM + 1, 1458:1461] = 20
    img[N_TOP:N_BOTTOM + 1, BARS[-1] - 1:BARS[-1] + 2] = 245   # ...where a stroke ends the notation lines
    img[N_TOP:BOTTOM + 1, BARS[-1] - 1:BARS[-1] + 2] = 20
    staff = find_staff(rgb(img))
    assert all(abs(b.x - BARS[-1]) > 4 for b in staff.barlines)


def test_final_barline_through_both_staves_counts():
    """Blind 4 video 2: a thin + thick final barline through both staves. The lines run on between the
    two strokes, so the staff's end is looked for one to two spacings past the thin stroke."""
    img = system(through=BARS[:-1])
    end = BARS[-1]
    img[N_TOP:BOTTOM + 1, end - 12:end - 9] = 20               # thin stroke...
    img[N_TOP:BOTTOM + 1, end - 1:end + 6] = 20                # ...and thick, through both staves
    staff = find_staff(rgb(img))
    assert len(staff.barlines) == len(BARS) and staff.barlines[-1].system
    assert len(staff.barlines[-1].strokes) == 2


def test_notation_staff_with_a_row_of_beams_one_spacing_above_is_found():
    """Blind 4 video 3, last row: beams over notes one spacing above the notation staff fill a quarter
    of that row, an even sixth "line"; the vendored finder returned a six-line staff and the notation
    staff was missed (the row was laid out as TAB only)."""
    img = system()
    for x in (150, 420, 700, 980):                         # beams, 80 px each: 24% of the row
        img[N_TOP - N_SP - 1:N_TOP - N_SP + 2, x:x + 80] = 20
    staff = find_staff(rgb(img))
    found = notation_staff(rgb(img), staff)
    assert found is not None
    assert [round(y) for y in found.lines] == [N_TOP + i * N_SP for i in range(5)]


def test_the_headers_own_antialiased_edge_is_not_a_glyph_after_it(tmp_path):
    """Blind 4 videos 2 and 3: a 1 px column of the key's edge right at the header's end read as a
    glyph starting there with no TAB digit under it (a key change), so no header was printed."""
    from tabvid import measures
    (tmp_path / 'measures').mkdir()
    rows = header_rows()
    rows[2]['gray'][N_TOP + 4:N_TOP + 20, 118] = 120                   # row 3: the last sharp's soft edge
    note, warning = measures.strip_headers(rows, tmp_path)
    assert warning is None and all('cut_start' in r for r in rows)


def test_a_row_start_number_on_a_speck_the_rows_share_is_not_cut_with_the_header(tmp_path):
    """Blind 4 video 2: the numbers 6, 10, 14... printed over each row's header overlap in a few pixels,
    which became part of the shared header; painting it and its halo erased each row's number."""
    from tabvid import measures
    (tmp_path / 'measures').mkdir()
    rows = header_rows()
    for row, angle in zip(rows, (0, 45, 90, 135)):        # a different mark per row, crossing at one point
        dx, dy = np.cos(np.radians(angle)) * 12, np.sin(np.radians(angle)) * 12
        cv2.line(row['gray'], (round(60 - dx), round(N_TOP - 30 - dy)), (round(60 + dx), round(N_TOP - 30 + dy)), 20, 3)
        row['image_bgr'] = cv2.cvtColor(row['gray'], cv2.COLOR_GRAY2BGR)
    before = [(r['gray'][N_TOP - 45:N_TOP - 14, 45:76] < 100).sum() for r in rows]
    note, warning = measures.strip_headers(rows, tmp_path)
    assert warning is None and all('cut_start' in r for r in rows)
    assert all((r['gray'][N_TOP - 45:N_TOP - 14, 45:76] < 100).sum() == n for r, n in zip(rows, before))
    assert (rows[1]['gray'][N_TOP + 3:N_TOP + 14, 100:118] > 200).all()   # the key is still cut
