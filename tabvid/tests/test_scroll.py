"""Scroll vote on drawn strips: python -m pytest tabvid/tests -q"""
import cv2
import numpy as np

from tabvid import scroll
from tabvid.report import Report
from tabvid.staff import find_staff

TOP, SP, MEASURE = 200, 22, 200        # top staff line y, line spacing, measure width
BOTTOM = TOP + 5 * SP


def strip(contents):
    """One long TAB staff, one measure per entry; equal entries draw identical measures."""
    width = MEASURE * len(contents) + 40
    img = np.full((420, width), 245, np.uint8)
    for i in range(6):
        cv2.line(img, (20, TOP + i * SP), (width - 20, TOP + i * SP), 90, 1)
    for m, content in enumerate(contents):
        x0 = 20 + m * MEASURE
        img[TOP:TOP + 5 * SP + 1, x0 - 1:x0 + 2] = 20
        rng = np.random.default_rng(content)
        for k in range(4):
            x, line = x0 + 25 + 40 * k, int(rng.integers(0, 6))
            cv2.putText(img, str(int(rng.integers(0, 10))), (x, TOP + line * SP + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)
    x_end = 20 + len(contents) * MEASURE
    img[TOP:TOP + 5 * SP + 1, x_end - 1:x_end + 2] = 20
    return img


def rows_from(views):
    out = []
    for n, view in enumerate(views, 1):
        staff = find_staff(cv2.cvtColor(view, cv2.COLOR_GRAY2RGB))
        out.append((n, view, staff, 5))
    return out


def test_scrolling_strip_votes_yes_and_median_picks_the_right_shift(tmp_path):
    # Measures 8-13 repeat a two-measure figure, so one join also matches two measures further on.
    contents = list(range(8)) + [100, 101] * 3 + list(range(20, 32))
    long = strip(contents)
    width, step = 5 * MEASURE + 30, 2 * MEASURE   # five measures in view, scroll by two
    views = [long[:, x:x + width] for x in range(0, long.shape[1] - width, step)]
    scrolls, joins, _ = scroll.vote(rows_from(views), Report(tmp_path))
    assert scrolls
    assert all(j.confident for j in joins)
    assert {j.chosen.dx for j in joins} == {step}


def test_page_switching_with_a_repeated_measure_votes_no(tmp_path):
    # Three pages of four measures; page 2 starts with the measure page 1 ended on (a repeat).
    pages = [strip([1, 2, 3, 4]), strip([4, 5, 6, 7]), strip([8, 9, 10, 11]), strip([12, 13, 14, 15])]
    scrolls, joins, _ = scroll.vote(rows_from(pages), Report(tmp_path))
    assert not scrolls


def test_same_view_split_in_two_is_not_counted(tmp_path):
    page = strip([1, 2, 3, 4])
    faded = np.clip(page.astype(int) + 40, 0, 255).astype(np.uint8)   # a fade-in frame of the same view
    scrolls, joins, _ = scroll.vote(rows_from([faded, page]), Report(tmp_path))
    assert joins[0].same_view is not None and not scrolls


def test_stitch_rebuilds_the_strip_and_checks_mid_scroll_frames(tmp_path):
    long = strip(list(range(12)))
    width, step = 5 * MEASURE + 30, 2 * MEASURE
    starts = list(range(0, long.shape[1] - width, step))
    held = [long[:, x:x + width] for x in starts]
    mid = long[:, starts[1] + 90:starts[1] + 90 + width]            # caught mid-scroll: covered
    stray = held[-1].copy()
    cv2.putText(stray, '9', (500, TOP + 2 * SP + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)  # ink no held view has
    order = [held[0], held[1], mid, *held[2:], stray]
    last = len(order)
    report = Report(tmp_path)
    rows = [(n, v, s, 1 if n in (3, last) else 5) for n, v, s, _ in rows_from(order)]
    scrolls, joins, held_numbers = scroll.vote(rows, report)
    views = {n: (cv2.cvtColor(v, cv2.COLOR_GRAY2RGB), s) for n, v, s, _ in rows}
    strips, dropped = scroll.stitch(views, joins, held_numbers, report)
    assert scrolls and list(dropped) == [3]                       # mid-scroll, between joined views: covered...
    assert 'between joined views 2 and 4' in dropped[3]
    assert [s.parts[0][0] for s in strips] == [1, last]           # ...the stray one at the end is tested and kept
    rebuilt = cv2.cvtColor(strips[0].rgb, cv2.COLOR_RGB2GRAY)
    assert np.array_equal(rebuilt, long[:, :rebuilt.shape[1]])


def test_one_px_stems_survive_a_1px_registration_error(tmp_path):
    """Two views of the same music, the second registered 1 px off: seam compositing keeps the
    stem, where lightest-pixel blending of the overlap erased it."""
    long = strip(list(range(6)))
    width = 3 * MEASURE + 30
    a = long[:, :width].copy()
    b = long[:, MEASURE:MEASURE + width].copy()
    x = 2 * MEASURE + 40                                   # a 1 px stem below the staff, in the overlap
    a[BOTTOM + 3:BOTTOM + 40, x] = 20
    b[BOTTOM + 3:BOTTOM + 40, x - MEASURE - 1] = 20        # the same stem, 1 px left in view b
    sa, sb = find_staff(cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)), find_staff(cv2.cvtColor(b, cv2.COLOR_GRAY2RGB))
    strip_ = scroll.Strip(parts=[(1, 0, 0), (2, MEASURE, 0)])
    scroll._place(strip_, {1: (cv2.cvtColor(a, cv2.COLOR_GRAY2RGB), sa), 2: (cv2.cvtColor(b, cv2.COLOR_GRAY2RGB), sb)})
    stem = cv2.cvtColor(strip_.rgb, cv2.COLOR_RGB2GRAY)[BOTTOM + 3:BOTTOM + 40, x - 1:x + 1]
    assert (stem < 100).any(axis=1).all()


def test_identical_bars_are_joined_at_the_shift_where_the_measure_numbers_agree(tmp_path):
    """Backed by one video (blind C): bars 22-24 are identical, so the ink also matches one measure
    short, where the larger overlap holds more barlines. The gray numbers tell the shifts apart."""
    def numbered(contents, with_numbers=True):
        img = strip(contents)
        for m in range(len(contents)):
            if with_numbers:
                cv2.putText(img, str(m + 1), (24 + m * MEASURE, TOP - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, 140, 1)
        return img

    contents = [1, 2, 7, 7, 7, 7, 7, 7, 3, 4, 5, 6]
    width, step = 5 * MEASURE + 30, 3 * MEASURE
    for with_numbers, expected in ((True, {step}), (False, {step - MEASURE, step - 2 * MEASURE})):
        long = numbered(contents, with_numbers)
        views = [long[:, x:x + width] for x in range(0, long.shape[1] - width + 1, step)]
        scrolls, joins, _ = scroll.vote(rows_from(views), Report(tmp_path))
        assert scrolls and {j.chosen.dx for j in joins} == expected   # without numbers: one and two measures short


def _with_clef(view):
    """A clef pinned over the left end of a view: a white box hiding the music, 'TAB' drawn in it."""
    out = view.copy()
    out[TOP - 60:BOTTOM + 40, :64] = 245
    for i in range(6):
        out[TOP + i * SP, :64] = 90
    for k, letter in enumerate('TAB'):
        cv2.putText(out, letter, (14, TOP + 30 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, .9, 20, 2)
    return out


def test_strip_paged_almost_a_full_view_with_a_pinned_clef_is_stitched(tmp_path):
    """Backed by one video (blind A): shifts of 84-90% of the width, a clef drawn over every page."""
    long = strip(list(range(40)))
    width, step = 5 * MEASURE + 30, 850                       # 83% of the view: 180 px of overlap
    views = [_with_clef(long[:, x:x + width]) for x in range(0, long.shape[1] - width, step)]
    rows = rows_from(views)
    assert not scroll.vote(rows, Report(tmp_path))[0]          # the clef hides the overlap's start

    zone = scroll.pinned_zone(views, SP, rows[0][2].lines)
    assert 64 <= zone <= 64 + 2 * SP
    scrolls, joins, held = scroll.vote(rows, Report(tmp_path), zone)
    assert scrolls and {j.chosen.dx for j in joins} == {step}

    staffs = {n: (cv2.cvtColor(v, cv2.COLOR_GRAY2RGB), s) for n, v, s, _ in rows}
    strips, _ = scroll.stitch(staffs, joins, held, Report(tmp_path), zone=zone)
    assert len(strips) == 1
    rebuilt = cv2.cvtColor(strips[0].rgb, cv2.COLOR_RGB2GRAY)
    end = rebuilt.shape[1]
    assert (rebuilt[:, 70:] == long[:, 70:end]).all()          # every later clef is gone, the music is back


def test_pages_that_all_start_with_a_clef_are_still_page_switching(tmp_path):
    pages = [_with_clef(strip([5 * k + i for i in range(5)])) for k in range(5)]
    rows = rows_from(pages)
    zone = scroll.pinned_zone(pages, SP, rows[0][2].lines)
    assert zone > 0 and not scroll.vote(rows, Report(tmp_path), zone)[0]
    plain = [strip([5 * k + i for i in range(5)]) for k in range(5)]
    assert scroll.pinned_zone(plain, SP, rows[0][2].lines) == 0  # an opening barline alone is no header


def test_closing_barline_over_an_opening_barline_is_not_an_overlap():
    """V1 rows 2->3: with the wider search, 92 px of overlap held one barline from each page and nothing else."""
    a, b = strip([1, 2, 3, 4, 5]), strip([6, 7, 8, 9, 10])
    for img in (a, b):                                          # no digits near the page edges
        img[TOP - 12:BOTTOM + 12, 22:120] = 245
        img[TOP - 12:BOTTOM + 12, -122:-22] = 245
        for i in range(6):
            img[TOP + i * SP, 22:-22] = 90
    pad = np.full((a.shape[0], 60), 245, np.uint8)             # the closing barline 80 px from the edge
    (_, ga, sa, _), (_, gb, sb, _) = rows_from([np.hstack([a, pad]), np.hstack([b, pad])])
    dx = ga.shape[1] - 100                                      # the two barlines on top of each other
    inks = [g < scroll.INK for g in (ga, gb)]
    for ink in inks:                                            # as candidates() does: staff lines are not ink
        for y in sa.lines:
            ink[round(y) - 1:round(y) + 2] = False
    edge = scroll._compare(*inks, sa.bars, sb.bars, dx, spacing=sa.spacing)
    assert edge.bars == 1 and edge.bars_ok and edge.mismatch == 0 and not edge.confident
    assert scroll._compare(*inks, sa.bars, sb.bars, dx).confident   # without the guard


def test_short_last_scroll_is_not_warned_but_a_short_one_mid_strip_is(tmp_path):
    long = strip(list(range(40)))
    width, step = 5 * MEASURE + 30, 2 * MEASURE

    def notes(starts):
        _, joins, _ = scroll.vote(rows_from([long[:, x:x + width] for x in starts]), Report(tmp_path))
        assert all(j.confident for j in joins)
        return [bool(j.note) for j in joins]

    regular = [k * step for k in range(6)]
    assert notes(regular + [regular[-1] + MEASURE // 2 + 30]) == [False] * 6          # the last scroll is short
    middle = regular[:3] + [x - step + MEASURE // 2 + 30 for x in regular[3:]]
    assert notes(middle) == [False, False, True, False, False]                        # a short one mid-strip


def test_barline_just_left_of_the_overlap_does_not_hide_its_music():
    """Single example (blind2 E): a barline a few px left of the overlap has x < 0 there; the
    blanking slice then ran to a negative end and erased the whole overlap, so a true join read
    as "no music"."""
    ink = np.zeros((100, 400), bool)
    ink[40:60, 50:350] = True                                     # plenty of music
    assert scroll._has_music(ink, [-15.0], SP)
    assert scroll._has_music(ink, [-15.0, 200.0], SP)
    assert not scroll._has_music(ink[:, :5], [-15.0], SP)        # nothing there: still no music


def test_double_barline_cut_by_the_view_edge_still_matches():
    """Single example (blind2 E views 8->9): view A shows only 2 px of a double barline's second
    stroke at its right edge, so the centre of the whole barline sits 3 px off the same barline
    in view B (tolerance 2 px). The first stroke's centre is whole in both."""
    full = strip([1, 2, 3, 4, 5, 6, 7, 8])
    x = 20 + 4 * MEASURE                                        # the barline after measure 4
    full[TOP:BOTTOM + 1, x - 1:x + 2] = 245
    full[TOP:BOTTOM + 1, x:x + 4] = 20                          # first stroke
    full[TOP:BOTTOM + 1, x + 15:x + 21] = 20                    # second stroke (thick)
    for i in range(6):
        full[TOP + i * SP, x - 1:x + 23] = 90
    width, start, dx = 700, x + 17 - 700, 450                   # A ends 2 px into the second stroke
    a, b = full[:, start:start + width], full[:, start + dx:start + dx + width]
    (_, ga, sa, _), (_, gb, sb, _) = rows_from([a, b])
    assert len(sa.barlines[-1].strokes) == 2 and abs(sa.barlines[-1].x - (sb.barlines[1].x + dx)) > scroll.BAR_TOL
    found, _ = scroll.candidates(ga, sa, gb, sb)
    assert found and found[0].confident and abs(found[0].dx - dx) <= 1


def test_view_fading_at_its_edge_still_joins_over_a_small_overlap():
    """Single example (blind2 F): the panel fades out over the last ~25 px of every view, so the
    new view's left edge lacks a digit the old view shows. With a small overlap and no barline in
    it (limit CLOSE, 5%), that alone failed half of F's joins and then the scroll vote."""
    width, dx = 600, 490                                        # 110 px (5 spacings) of overlap
    music = np.zeros((120, width + dx), bool)
    for x in (dx + 3, dx + 40, dx + 70, dx + 95, 100, 300):    # digit-sized blobs
        music[50:64, x:x + 10] = True
        music[52:62, x + 2:x + 8] = False
    a, b = music[:, :width].copy(), music[:, dx:dx + width].copy()
    b[:, :10] = False                                          # B fades out at its left edge
    plain = scroll._compare(a, b, [], [], dx)
    assert plain.mismatch > scroll.CLOSE                        # the whole overlap: fails on the faded digit
    joined = scroll._compare(a, b, [], [], dx, spacing=SP)
    assert joined.mismatch == 0 and joined.confident
    other = music[:, dx + 37:dx + 37 + width]                  # a wrong shift still fails
    assert not scroll._compare(a, other, [], [], dx, spacing=SP).confident


def test_a_tiny_last_scroll_is_found(tmp_path):
    # A Team: the last scroll moved 35 px (1.3 spacings, 2% of the width); the search started at 10% of
    # the width, so the last view became a strip of its own and its four measures were printed twice.
    long = strip(list(range(30)))
    width, step = 5 * MEASURE + 30, 2 * MEASURE
    starts = [k * step for k in range(5)]
    starts.append(starts[-1] + round(1.5 * SP))
    scrolls, joins, _ = scroll.vote(rows_from([long[:, x:x + width] for x in starts]), Report(tmp_path))
    assert scrolls and all(j.confident for j in joins)
    assert joins[-1].chosen.dx == round(1.5 * SP) and not joins[-1].note
