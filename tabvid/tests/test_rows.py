"""Stage 3 rows: python -m pytest tabvid/tests -q"""
import cv2
import numpy as np

from tabvid import rows as R
from tabvid.staff import find_staff

TOP, SP = 120, 22


def page(bars=(40, 300, 560, 820), digits=(('3', 100), ('5', 400), ('7', 650))):
    img = np.full((340, bars[-1] + 40, 3), 245, np.uint8)
    for i in range(6):
        cv2.line(img, (bars[0], TOP + i * SP), (bars[-1], TOP + i * SP), (90, 90, 90), 1)
    for x in bars:
        img[TOP:TOP + 5 * SP + 1, x - 1:x + 2] = 20
    for text, x in digits:
        cv2.putText(img, text, (x, TOP + 2 * SP + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 2)
    return img


def row_of(*images):
    return R.Row([R.View(i, i / 2, im, find_staff(im)) for i, im in enumerate(images)])


def test_same_page_split_by_a_neutral_cursor_is_merged():
    a = page()
    b = page()
    b[TOP + 30:TOP + 41, 304:306] = 60           # neutral playhead blob by barline 2 (22 px: same_row splits)
    other = page(digits=(('1', 120), ('2', 420), ('9', 700)))
    rows = [row_of(a, a), row_of(b, b), row_of(other, other)]
    merged = [R._merged_row(r, ())[:2] for r in rows]
    out, images, joined, _ = R._merge_same_views(rows, merged, ())
    assert joined == ['1+2']
    assert len(out) == 2 and len(out[0].views) == 4
    from tabvid.vendor.same_row import same_row
    assert not same_row(rows[0].views[0], rows[1].views[0])     # the split this repairs


def test_fade_frame_with_a_visible_staff_is_left_out(tmp_path):
    frames = []
    for i in range(4):
        img = page()
        if i == 3:                                  # last frame of a fade to black: paper at gray ~76
            img = (img.astype(np.float32) * .31).astype(np.uint8)
        path = tmp_path / f'{i:06d}.png'
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frames.append(path)
    box = (0, 0, img.shape[1], img.shape[0])
    rows, no_staff, fading, _ = R.group_rows(frames, box, 2)
    assert fading == [3] and not no_staff
    assert [v.index for r in rows for v in r.views] == [0, 1, 2]


def _chorded(digits=(('3', 100), ('5', 400), ('7', 650))):
    img = page(digits=digits)
    for name, x in (('Am', 60), ('G', 330), ('F', 600)):         # chord names above the staff
        cv2.putText(img, name, (x, 60), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
    return img


def test_one_frame_of_the_previous_page_sliding_out_is_dropped():
    """Backed by one video (blind B, frame #214): the page slides up, its chord names leave the crop."""
    a = _chorded()
    sliding = np.full_like(a, 245)
    sliding[:-52] = a[52:]                                       # the same page, 52 px higher
    sliding[40:280, 500:503] = (200, 120, 40)                    # the cursor a single frame cannot lose
    b = _chorded(digits=(('1', 120), ('2', 420), ('9', 700)))
    slide_view = row_of(sliding).views[0]
    assert R._is_slide(slide_view, row_of(a, a))
    assert not R._is_slide(slide_view, row_of(b, b))             # another page with the same barlines
    other = row_of(page(bars=(40, 250, 560, 820))).views[0]      # a real one-frame page: other barlines
    assert not R._is_slide(other, row_of(a, a))


def test_page_whose_opening_barline_fails_in_some_frames_is_one_row():
    # Single example (blind D rows 3 and 4): a system line runs on above the opening barline in
    # the first frames of a page, so the barline is rejected there and accepted afterwards.
    clean = page()
    early = page()
    early[TOP - 8:TOP, 39:42] = 20                          # the opening barline extends above the staff
    assert len(find_staff(early).bars) == len(find_staff(clean).bars) - 1
    assert any(abs(x - 40) <= 2 for x, _ in find_staff(early).rejected)
    other = page(digits=(('1', 120), ('2', 420), ('9', 700)))
    rows = [row_of(early, early), row_of(clean, clean), row_of(other, other)]
    merged = [R._merged_row(r, ())[:2] for r in rows]
    out, images, joined, _ = R._merge_same_views(rows, merged, ())
    assert joined == ['1+2']
    assert len(images[0][1].bars) == 4                      # merged by lightest pixel: the barline is whole
    # A barline with no stroke at all in the other row still keeps two rows apart.
    fewer = page(bars=(40, 300, 820))
    rows = [row_of(fewer, fewer), row_of(clean, clean)]
    merged = [R._merged_row(r, ())[:2] for r in rows]
    assert R._merge_same_views(rows, merged, ())[2] == []


def _views(*images):
    return R.Row(views=[R.View(i, i / 2, image, None) for i, image in enumerate(images)])


def test_note_drawn_red_while_played_merges_black():
    """F: a digit drawn red in the frames that play it and black in the others merges black, all
    over the crop (not only near a staff line); a moving cursor and a pale highlight leave no trace,
    and colour present in every frame (a still cursor) keeps the plain lightest pixel."""
    paper = np.full((60, 120, 3), 250, np.uint8)
    black, red = paper.copy(), paper.copy()
    black[10:30, 20:30] = 0                       # the digit, well away from any staff line
    red[10:30, 20:30] = (220, 60, 30)
    for frame, x in ((black, 60), (red, 80)):     # a moving green cursor line
        frame[:, x:x + 3] = (60, 200, 60)
    highlight = black.copy()
    highlight[:, 90:110] = (255, 230, 120)        # a pale yellow highlight band (luma > INK_LUMA)
    for frame in (black, red, highlight):
        frame[40:50, 100:110] = (30, 30, 200)     # blue ink in every frame
    merged = R.merge(_views(black, red, highlight))
    assert merged[10:30, 20:30].max() == 0        # the digit is black, not red
    assert merged[:, 55:90].min() >= 245          # no cursor residue (paper where it moved)
    assert (merged[:5, 90:110] >= 245).all()      # the highlight band is paper where a frame shows it
    assert (merged[40:50, 100:110] == (30, 30, 200)).all()


def test_highlight_band_present_in_every_frame_is_not_darkened():
    paper = np.full((40, 80, 3), 250, np.uint8)
    paper[:, 30:50] = (255, 230, 120)
    merged = R.merge(_views(paper.copy(), paper.copy()))
    assert (merged[:, 30:50] == (255, 230, 120)).all()


def _page(cover=False):
    """A page of four measures; with `cover`, an opaque framed picture over the last measure."""
    img = np.full((200, 800, 3), 250, np.uint8)
    for i in range(6):
        img[60 + 16 * i, 20:780] = 60
    for x in (20, 210, 400, 590, 779):
        img[60:141, x:x + 2] = 20
    for k, x in enumerate(range(60, 760, 45)):
        cv2.putText(img, str(k % 10), (x, 60 + 16 * (k % 6) + 5), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1)
    if cover:
        img[5:195, 560:790] = 250                                 # opaque white picture...
        cv2.rectangle(img, (560, 5), (790, 195), (90, 200, 90), 3)  # ...in a frame
        cv2.ellipse(img, (675, 100), (60, 80), 0, 0, 360, (40, 40, 40), 2)   # a drawing inside
    return img


def _rows_of(*images):
    rows = [R.Row(views=[R.View(i, i, image, find_staff(image))]) for i, image in enumerate(images)]
    return rows, [(image, find_staff(image)) for image in images]


def test_page_partly_covered_by_a_picture_then_clean_is_one_row():
    """I: the first page is shown with a picture of fingering hands over its last measure, then clean.
    The two are one row, from the clean copy; a different page with the same opening is not."""
    covered, clean = _page(cover=True), _page()
    rows, merged = _rows_of(covered, clean)
    out_rows, out_merged, joined, covers = R._merge_same_views(rows, merged, [])
    assert len(out_rows) == 1 and covers == ['1+2']
    assert (out_merged[0][0] == clean).all() and out_rows[0].views[0].dropped
    rows, merged = _rows_of(clean, covered)                       # the other order
    out_rows, out_merged, _, covers = R._merge_same_views(rows, merged, [])
    assert len(out_rows) == 1 and (out_merged[0][0] == clean).all()
    other = clean.copy()
    other[62:139, 600:770] = 250                                  # last measure's music differs...
    for i in range(6):
        other[60 + 16 * i, 600:770] = 60                          # ...on an unbroken staff
    for k, x in enumerate(range(605, 760, 22)):                 # other music on an unbroken staff
        cv2.putText(other, '58'[k % 2], (x, 60 + 16 * ((k * 3) % 6) + 5), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1)
        other[110:141, x + 4] = 20                                # with stems
    rows, merged = _rows_of(clean, other)
    assert len(R._merge_same_views(rows, merged, [])[0]) == 2


def test_frames_after_the_last_measure_showing_a_staff_are_flagged(tmp_path):
    """K before the fix: the last page's short staff was not taken, and its measure was lost silently.
    Frames after the last staff that still show staff lines are flagged; blank frames, fades and an
    outro without staff lines are not."""
    def frame(kind):
        img = np.full((300, 1900, 3), 245, np.uint8)
        if kind == 'short staff':
            for i in range(6):
                img[100 + 20 * i, 80:320] = 60
            img[100:201, 80:82] = 20
            cv2.putText(img, '7', (150, 166), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 2)
        elif kind == 'fade':
            img[:] = 120
            for i in range(6):
                img[100 + 20 * i, 80:1800] = 60
        elif kind == 'outro':
            cv2.putText(img, 'Thanks for watching', (500, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 30, 30), 4)
            cv2.circle(img, (1500, 150), 90, (60, 60, 60), -1)
        return img
    paths = []
    for i, kind in enumerate(['short staff', 'blank', 'fade', 'outro', 'short staff']):
        path = tmp_path / f'{i:06d}.png'
        cv2.imwrite(str(path), frame(kind))
        paths.append(path)
    box = [0, 0, 1900, 300]
    assert R.staff_after_end(paths, box, [], False, -1, 245.0) == [0, 4]
    assert R.staff_after_end(paths, box, [], False, 0, 245.0) == [4]
