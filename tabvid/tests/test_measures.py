"""Stage 4 row starts and green residue: python -m pytest tabvid/tests -q"""
import cv2
import numpy as np

from tabvid import measures
from tabvid.rows import clean_merged
from tabvid.staff import Staff

TOP, SP = 60, 20
LINES = [TOP + i * SP for i in range(6)]


def row_image(clef=True, opening=True, width=700):
    img = np.full((220, width, 3), 245, np.uint8)
    for y in LINES:
        img[y, 20:width - 20] = 60
    if opening:
        img[TOP:LINES[-1] + 1, 20:22] = 20
    if clef:
        for k, letter in enumerate('TAB'):
            cv2.putText(img, letter, (30, TOP + 30 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    rng = np.random.default_rng(width)
    for x in range(90, width - 60, 50):
        cv2.putText(img, str(int(rng.integers(10))), (x, LINES[int(rng.integers(6))] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 2)
    img[TOP:LINES[-1] + 1, 400:402] = 20
    img[TOP:LINES[-1] + 1, width - 22:width - 20] = 20
    return img


def bar(x):
    return {'x': x + .5, 'left': x, 'right': x + 1, 'strokes': [[x, x + 1, x + .5]]}


def test_no_opening_barline_keeps_the_first_measure():
    gray = cv2.cvtColor(row_image(opening=False), cv2.COLOR_BGR2GRAY)
    start, _ = measures.staff_extent(gray, LINES, SP)
    assert abs(start - 20) <= 1
    row = {'barlines': [bar(400), bar(678)], 'spacing': SP}
    cuts = measures.cut_row(row, gray.shape[1], start)
    assert [(x0, x1) for x0, x1, _ in cuts] == [(start, 402), (402, 680)]
    with_opening = {'barlines': [bar(20), bar(400), bar(678)], 'spacing': SP}
    assert len(measures.cut_row(with_opening, gray.shape[1], start)) == 2   # unchanged when there is one


def clef_rows(has_clef, starts=None):
    rows = []
    for n, (clef, width) in enumerate(zip(has_clef, range(700, 2000, 20)), 1):
        image = row_image(clef=clef, width=width)
        if starts and starts[n - 1]:                       # music that recurs at some row starts
            cv2.putText(image, '(0)', (28, LINES[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 2)
            cv2.ellipse(image, (40, LINES[1] + 14), (14, 6), 0, 0, 180, (20, 20, 20), 2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows.append({'row': n, 'image_bgr': image, 'gray': gray, 'staff_lines': LINES, 'spacing': SP,
                     'staff_start': measures.staff_extent(gray, LINES, SP)[0]})
    return rows


def test_clef_removed_from_later_rows_only_where_it_matches():
    rows = clef_rows([True, True, False, True])
    before = rows[3]['image_bgr'].copy()
    cleared, missing, note = measures.clear_clef(rows)
    assert cleared == [2, 4] and missing == [3] and note is None
    after = rows[3]['image_bgr']
    assert (after[TOP + 3:LINES[-1] - 3, 25:75] >= 240).all() or \
        (np.delete(after[:, 25:75], [y + d for y in LINES for d in (-1, 0, 1)], axis=0) >= 240).all()
    assert (after[LINES[2], 25:75] < 100).all()                   # staff lines run through
    assert np.array_equal(after[:, 85:], before[:, 85:])          # music after the clef untouched


def test_music_recurring_at_row_starts_is_not_a_clef():
    """V2: the clef only on row 1; "(0)" with a tie opens 4 of the other rows. It must stay."""
    has_clef = [True] + [False] * 5
    starts = [False, True, True, False, True, True]
    rows = clef_rows(has_clef, starts)
    before = [r['image_bgr'].copy() for r in rows]
    cleared, missing, note = measures.clear_clef(rows)
    assert cleared == [] and missing == [] and 'kept as printed' in (note or 'kept as found')
    assert all(np.array_equal(r['image_bgr'], b) for r, b in zip(rows, before))


def test_green_cursor_left_on_a_staff_line_is_undone():
    img = row_image()
    img[LINES[1], 200:220] = (66, 174, 66)                        # per-channel max of black line and green cursor
    staff = Staff(tuple(LINES), SP, (), ())
    out, count = clean_merged(img, staff, [])
    assert count == 20 and (out[LINES[1], 200:220] == 66).all()
    whole = np.ones((25, 200), bool)
    out, _ = clean_merged(img, staff, [(100, 150, 300, 175, whole)])   # a mask box below the staff
    assert (out[LINES[5], 100:300] == 60).all() and (out[170, 100:300] == 245).all()


def test_mask_removes_a_translucent_overlay_and_recovers_the_music_under_it(tmp_path):
    """A logo over scrolling music: an opaque outline plus a translucent darkening watermark.
    The watermark is reversed (music under it recovered), the outline painted paper, and the
    rest of the box left exactly as it was."""
    from tabvid.rows import overlay_masks, unblend
    long = np.full((220, 2400, 3), 245, np.uint8)
    for y in LINES:
        long[y] = 60
    for x in range(40, 2400, 90):
        cv2.putText(long, 'P', (x, TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 2)
        long[TOP:LINES[-1] + 1, x + 40:x + 42] = 20
    frames, clean = [], []
    for k in range(12):
        under = long[:, 60 * k:60 * k + 700].copy()
        frame = under.copy()
        frame[8:48, 330:400] = (frame[8:48, 330:400] * .5).astype(np.uint8)   # translucent watermark
        cv2.rectangle(frame, (300, 5), (420, 50), (30, 30, 30), 2)            # opaque outline
        path = tmp_path / f'{k:03d}.png'
        cv2.imwrite(str(path), frame[..., ::-1])
        frames.append(path)
        clean.append(under)
    model, = overlay_masks(frames, (0, 0, 700, 220), [(290, 0, 430, 60)])
    l, t, r, b, opaque, alpha = model[:6]
    assert opaque[5 - t, 360 - l] and opaque[30 - t, 300 - l]          # the outline is opaque
    inked = np.min([c[8:48, 330:400].min(axis=2) for c in clean], axis=0) < 100
    watermark = alpha[8 - t:48 - t, 330 - l:400 - l]
    assert abs(np.median(watermark[inked]) - .5) < .05      # where ink passed under it: half
    assert (watermark[~inked] == 1).all()                    # never under ink: opaque, painted paper
    seen = 0
    for k in range(12):
        frame = cv2.imread(str(frames[k]))[..., ::-1].copy()
        out = unblend(frame, [model]).astype(int)
        inner = (slice(10, 46), slice(332, 398))
        diff = np.abs(out[inner] - clean[k][inner].astype(int)).max(axis=2)
        paper = clean[k][inner].min(axis=2) > 200
        assert diff[paper].max() <= 3, 'paper under it recovered'
        assert diff[~paper].max(initial=0) <= 12, 'music under it recovered'
        seen += (~paper).sum()
        assert (out[5, 300:421] == 245).all()                              # outline painted paper
        assert np.array_equal(out[60:], clean[k][60:].astype(int))         # elsewhere untouched
    assert seen > 100                                                      # music did pass under it


def test_time_signature_between_two_barlines_joins_the_next_measure():
    """Backed by one video (blind C, `| 3/4 ||:` after the pickup): 64 px holding only the time signature."""
    gray = cv2.cvtColor(row_image(clef=False, width=900), cv2.COLOR_BGR2GRAY)
    gray[TOP:LINES[-1] + 1, 80:82] = 20                                    # a second barline 3 spacings on
    gray[TOP + 1:LINES[-1], 22:80] = 245                                   # nothing between them...
    for y in LINES:
        gray[y, 22:80] = 60
    for text, y in (('3', LINES[2] + 16), ('4', LINES[4] + 16)):           # ...but a 3/4, two spacings a digit
        cv2.putText(gray, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 20, 4)
    cuts = [(20, 82, bar(80)), (82, 402, bar(400)), (402, 880, bar(878))]
    joined, count = measures.join_slivers(cuts, gray, LINES, SP)
    assert count == 1 and [(x0, x1) for x0, x1, _ in joined] == [(20, 402), (402, 880)]

    cv2.putText(gray, '5', (60, LINES[0] + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)   # a note: a short pickup
    assert measures.join_slivers(cuts, gray, LINES, SP) == (cuts, 0)


def edge_rows(tops, bracket=True, rule=True):
    """Page-switching rows whose page sits at a different height each time, with a system line
    running from the crop's top edge into the staff, a bracket foot under it and a progress bar."""
    rows = []
    for number, top in enumerate(tops, 1):
        img = row_image(clef=False)
        img = np.roll(img, top - TOP, axis=0)
        if bracket:
            img[:top, 20:22] = 20                                      # system line above the staff
            cv2.line(img, (0, top + 5 * SP + 12), (14, top + 5 * SP + 45), (20, 20, 20), 3)   # foot, cut by the crop's left edge
        if rule:
            img[:3] = (40, 120, 220)                                   # progress bar along the top edge
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows.append({'row': number, 'image_bgr': img, 'gray': gray, 'spacing': SP,
                     'staff_lines': [top + i * SP for i in range(6)]})
    return rows


def test_ink_at_the_crop_edge_in_every_row_is_painted_out_and_the_music_is_kept():
    # Single example (blind D): the crop cut through a bracket and a progress bar.
    rows = edge_rows([60, 54, 66, 60, 62])
    before = [row['gray'].copy() for row in rows]
    static = measures.static_edge_ink(rows)
    assert static is not None
    for row, old in zip(rows, before):
        top, mask = row['staff_lines'][0], static[row['row']]
        assert measures.clear_edge_rules(row) == 3                     # the bar, though the system line ties it to the staff
        assert (row['gray'][:5] > 200).all()
        ink = old < measures.INK
        assert mask[max(0, top - 60):top - 3, 20:22].all()             # the system line, at this row's own height
        foot = ink.copy()
        foot[:top + 5 * SP + 3] = False
        foot[:, 20:] = False
        assert foot.any() and mask[foot].all()                         # the whole foot
        staff = np.zeros_like(mask)
        staff[top - 2:top + 5 * SP + 3] = True
        assert not mask[staff].any()                                   # nothing inside the staff band
        assert not mask[12:, 40:].any()                                # nothing over the music (the bar is static too)


def test_rows_without_shared_edge_ink_are_left_alone():
    assert measures.static_edge_ink(edge_rows([60, 54, 66, 60], bracket=False, rule=False)) is None
    assert measures.static_edge_ink(edge_rows([60, 54])) is None       # too few rows to tell
    one = edge_rows([60, 60, 60, 60], bracket=False, rule=False)
    cv2.line(one[1]['image_bgr'], (0, 180), (14, 219), (20, 20, 20), 3)   # edge ink in one row only: music
    one[1]['gray'] = cv2.cvtColor(one[1]['image_bgr'], cv2.COLOR_BGR2GRAY)
    assert measures.static_edge_ink(one) is None
    assert measures.clear_edge_rules(one[0]) == 0


def wide_case(stroke=None, extra_blocks=()):
    """Three ordinary rows, one whose last measure (m8) is twice as wide (400-1178 px), and an ordinary
    last row (the piece's last measure is quiet when only wide)."""
    rows, records = [], []
    for number in (1, 2, 3, 4, 5):
        img = row_image(clef=False, width=1200 if number == 4 else 780)
        if number == 4 and stroke:
            img[TOP - stroke:LINES[-1] + 1, 790:792] = 20        # a barline with ink running on above it
        rows.append({'row': number, 'image_bgr': img})
        for x0, x1 in ((20, 400), (400, img.shape[1] - 20)):
            records.append({'measure': len(records) + 1, 'row': number, 'row_x0': x0, 'staff_x0': 0,
                            'staff_x1': x1 - x0, 'spacing': SP, 'notes': []})
    records[7]['extra_blocks'] = list(extra_blocks)
    return records, rows


def test_wide_measure_warning_fires_on_a_merged_measure_and_stays_quiet_on_long_ones():
    def names(found):
        signed, width_only, by_numbers, quiet = found
        return [r['measure'] for r, _ in signed], [r['measure'] for r in width_only], by_numbers, quiet

    # A barline missed without a trace: twice the typical width and twice its row's other measure.
    assert names(measures.wide_measures(*wide_case())) == ([], [8], [], [])
    # A barline that failed the barline rule (ink running on above it): warned with the reason.
    for reach in (10, 40):
        records, rows = wide_case(stroke=reach)
        found = measures.wide_measures(records, rows, numbered=True)
        assert names(found) == ([8], [], [], [])
        assert 'failed the barline rule' in found[0][0][1]
        assert abs(measures.rejected_strokes(rows[3]['image_bgr'], 400, 1180, SP)[0] - 790) <= 2
    # A numbered video: a merged measure holds two numbers; a long measure holds one (B m31).
    found = measures.wide_measures(*wide_case(extra_blocks=[[380, 20, 392, 34]]), numbered=True)
    assert names(found) == ([8], [], [], []) and 'second measure number' in found[0][0][1]
    assert names(measures.wide_measures(*wide_case(), numbered=True)) == ([], [], [], [8])
    # The numbers prove it is one measure.
    records, rows = wide_case(stroke=10)
    assert names(measures.wide_measures(records, rows, proven={8})) == ([], [], [8], [])
    # A row of long measures (a dense page; ZMj m46): wide for the video, not for its row.
    records, rows = wide_case()
    records[6]['staff_x1'] = 600
    assert names(measures.wide_measures(records, rows)) == ([], [], [], [8])
    # The piece's last measure, wide for its width only: quiet (a long closing measure; user).
    records, rows = wide_case()
    assert names(measures.wide_measures(records[:8], rows[:4])) == ([], [], [], [8])
    # ...but with a sign it is still warned.
    records, rows = wide_case(stroke=10)
    assert names(measures.wide_measures(records[:8], rows[:4])) == ([8], [], [], [])
    # A real barline (no ink beyond the staff) is a cut, not a sign.
    assert measures.rejected_strokes(row_image(clef=False), 20, 680, SP) == []


def test_a_pale_line_cut_along_the_crop_edge_is_painted_out():
    """Single example (blind2 G): the crop started on a pale panel border (gray ~215, broken into
    pieces), and its pieces went whole to single measures (m6: 470 px overhang)."""
    img = row_image(clef=False, width=900)
    img[0, 0:300] = 215                    # pieces of a pale border along the top edge
    img[0, 320:800] = 220
    img[0, 840:870] = 220                  # too short to tell from a cut glyph: kept
    img[0:40, 500:502] = 20                # a stem reaching the edge: only its outer rows go
    img[219, 100:700] = 225                # the same along the bottom edge
    row = {'image_bgr': img, 'gray': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 'spacing': SP}
    music = row['gray'][TOP - 5:].copy()
    assert measures.clear_edge_slivers(row) > 0
    gray = row['gray']
    assert (gray[0, :800] >= 240).all() and (gray[219, 100:700] >= 240).all()
    assert (gray[0, 840:870] < 235).all()
    assert (gray[measures.FRINGE:40, 500:502] < 50).all()
    assert (gray[TOP - 5:-measures.FRINGE] == music[:-measures.FRINGE]).all()

    full = row_image(clef=False, width=900)
    full[0, :] = 215                       # over most of the width: page furniture, left to that rule
    row = {'image_bgr': full, 'gray': cv2.cvtColor(full, cv2.COLOR_BGR2GRAY), 'spacing': SP}
    assert measures.clear_edge_slivers(row) == 0


def test_a_border_several_rows_thick_is_painted_whole():
    """Single example (blind2 F): a 3-row border along the crop's top edge; the old 2-row band saw
    its third row as ink "inward" and painted nothing."""
    img = row_image(clef=False, width=900)
    img[0:3, 0:400] = 215
    img[0:3, 420:900] = 220
    row = {'image_bgr': img, 'gray': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 'spacing': SP}
    assert measures.clear_edge_slivers(row) > 0
    assert (row['gray'][0:3] >= 240).all()


def test_a_side_border_is_painted_and_a_barline_at_the_edge_is_kept():
    """Single example (blind2 F): a gray panel border down the strip's right end went, above and
    below the staff, to the last measure as a 515 px overhang."""
    img = row_image(clef=False, width=700)
    img[:, 697:700] = 200                       # the border, top to bottom
    img[:, 696] = 230                           # its antialiased edge
    img[2:5, 690:696] = 230                     # a corner speck where the top border met it
    row = {'image_bgr': img, 'gray': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 'spacing': SP, 'staff_lines': LINES}
    assert measures.clear_edge_columns(row) == 4
    assert (row['gray'][:LINES[0] - 2, 690:] >= 240).all() and (row['gray'][LINES[-1] + 3:, 690:] >= 240).all()
    assert (row['gray'][TOP:LINES[-1] + 1, 678:680] < 50).all()    # the closing barline stays

    edge = row_image(clef=False, width=700)
    edge[TOP:LINES[-1] + 1, 696:700] = 20       # a closing barline right at the crop edge: staff only
    row = {'image_bgr': edge, 'gray': cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY), 'spacing': SP, 'staff_lines': LINES}
    assert measures.clear_edge_columns(row) == 0


def test_empty_opening_stretch_joins_the_first_measure_however_wide():
    """Blind 4 video 3: clef, key and 4/4 before a `||:` some 7 spacings in, with no opening barline.
    That stretch holds no notes, so it is not a measure: it joins the one after it."""
    gray = cv2.cvtColor(row_image(clef=False, width=900), cv2.COLOR_BGR2GRAY)
    gray[TOP + 1:LINES[-1], 0:200] = 245                                   # no opening barline...
    for y in LINES:
        gray[y, 0:200] = 60
    for text, y in (('4', LINES[2] + 16), ('4', LINES[4] + 16)):           # ...a 4/4 well inside the stretch
        cv2.putText(gray, text, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 20, 4)
    gray[TOP:LINES[-1] + 1, 198:204] = 20                                  # the thick stroke of ||:
    cuts = [(0, 204, bar(200)), (204, 402, bar(400)), (402, 880, bar(878))]
    assert 204 > measures.SLIVER * SP                                      # too wide to be a sliver
    assert measures.join_slivers(cuts, gray, LINES, SP) == (cuts, 0)
    joined, count = measures.join_slivers(cuts, gray, LINES, SP, opening=True)
    assert count == 1 and [(x0, x1) for x0, x1, _ in joined] == [(0, 402), (402, 880)]
    cv2.putText(gray, '5', (160, LINES[0] + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)   # a note: a real pickup
    assert measures.join_slivers(cuts, gray, LINES, SP, opening=True) == (cuts, 0)


def test_a_kept_tab_clef_is_not_a_note_in_the_opening_stretch():
    """Blind 4 video 3 (notation mode, header kept as printed): the TAB clef before the first `||:`
    has glyphs on the staff, but it is not music, so the empty opening stretch still joins."""
    gray = cv2.cvtColor(row_image(clef=True, opening=False, width=900), cv2.COLOR_BGR2GRAY)
    gray[TOP + 1:LINES[-1], 75:200] = 245                                  # nothing between clef and ||:
    for y in LINES:
        gray[y, 20:200] = 60
    gray[TOP:LINES[-1] + 1, 198:204] = 20
    cuts = [(20, 204, bar(200)), (204, 402, bar(400)), (402, 880, bar(878))]
    assert measures.join_slivers(cuts, gray, LINES, SP, opening=True) == (cuts, 0)   # T A B read as notes
    joined, count = measures.join_slivers(cuts, gray, LINES, SP, opening=True,
                                          clef=round(measures.CLEF_MAX * SP))
    assert count == 1 and [(x0, x1) for x0, x1, _ in joined] == [(20, 402), (402, 880)]


def marker_rows(shapes):
    """Rows with a gray page marker (a different shape per page, or None) over the row start, a dark
    measure number beside it."""
    rows = []
    for n, shape in enumerate(shapes, 1):
        img = row_image()
        if shape == 'square':
            img[16:40, 40:64] = 150
        elif shape == 'circle':
            cv2.circle(img, (52, 28), 12, (150, 150, 150), -1)
        elif shape == 'triangle':
            cv2.fillPoly(img, [np.array([[40, 40], [64, 40], [52, 16]])], (150, 150, 150))
        cv2.putText(img, str(n), (90, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        start = measures.staff_extent(gray, LINES, SP)[0]
        rows.append({'row': n, 'image_bgr': img, 'gray': gray, 'staff_lines': LINES, 'spacing': SP,
                     'staff_start': start})
    return rows


def test_page_markers_shared_by_most_rows_are_painted_and_numbers_kept():
    rows = marker_rows(['circle', 'square', 'triangle', 'circle'])
    numbers = [(r['gray'][20:42, 88:110] < 100).sum() for r in rows]
    assert measures.clear_page_markers(rows) == 4
    for row, number in zip(rows, numbers):
        assert (row['gray'][10:46, 34:70] > 230).all()                       # marker gone
        assert (row['gray'][20:42, 88:110] < 100).sum() == number            # number kept
    rows = marker_rows(['circle', None, None, None])                         # one row only: not a marker
    assert measures.clear_page_markers(rows) == 0 and (rows[0]['gray'][28, 52] < 200)


def test_a_key_before_a_near_opening_barline_stays_with_the_first_measure_once_the_header_is_cut():
    """Blind 4 video 3 page 3: `# ||:` right after the notation header. The opening barline is too
    near to make a measure of the stretch before it, so that stretch (the key) goes with measure 1."""
    row = {'barlines': [bar(100), bar(400), bar(700)], 'spacing': SP}
    assert measures.cut_row(row, 800, 80)[0][0] == 100                 # tab only: from the opening barline
    row['cut_start'] = 80
    assert [c[:2] for c in measures.cut_row(row, 800, 80)] == [(80, 402), (402, 702)]


def test_ink_cut_by_the_crop_edge_is_found_and_staff_ink_is_not():
    """Single example (blind test 5 video 3): the top of the next page's "Chorus" under row 5."""
    img = row_image(clef=False, width=900)
    h = img.shape[0]
    cv2.putText(img, 'Chorus', (300, h + 8), cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 20, 20), 2)   # cut by the bottom
    row = {'gray': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 'spacing': SP, 'staff_lines': LINES}
    found = measures.edge_cut_ink(row)
    assert found and all(edge == 'bottom' and 290 <= x0 and x1 <= 450 for edge, x0, x1 in found)
    assert measures.edge_cut_ink({**row, 'gray': cv2.cvtColor(row_image(clef=False, width=900),
                                                                cv2.COLOR_BGR2GRAY)}) == []
