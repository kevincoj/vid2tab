"""A staff whose top line is pale (blind test 5 video 3: gray 201-203 on 245 paper) is still found (fix plan 4, step 2)."""
import cv2
import numpy as np

from tabvid.staff import find_staff


def _staff(top_gray, noise=0):
    img = np.full((300, 1200), 245, np.uint8)
    for i in range(6):
        img[80 + 22 * i, 20:1180] = top_gray if i == 0 else 180
    for x in (20, 400, 790, 1179):
        img[80:80 + 5 * 22 + 1, x - 1:x + 1] = 30
    for x in range(60, 1150, 70):
        cv2.putText(img, '7', (x, 80 + 22 * 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, .5, 20, 1, cv2.LINE_AA)
    if noise:                                       # break up the pale line like compression does
        rng = np.random.default_rng(0)
        row = img[80, 20:1180].astype(int) + rng.integers(-noise, noise + 1, 1160)
        img[80, 20:1180] = np.clip(row, 0, 255)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def test_a_pale_top_line_is_found():
    staff = find_staff(_staff(208, noise=6))
    assert staff is not None and round(staff.lines[0]) == 80 and len(staff.barlines) == 4


def test_paper_only_is_still_no_staff():
    assert find_staff(np.full((300, 1200, 3), 245, np.uint8)) is None


# --- fix plan 4, video 5: a line broken by noise, staff gaps, music not read --------------------------

from tabvid.staff import staff_gaps, unread_lines


def test_a_line_with_a_one_pixel_piece_beside_it_is_still_a_staff():
    rgb = _staff(180)
    rgb[80 + 22 * 2 - 3, 20:1180] = 150                 # a stray 1 px run 3 px above the third line
    rgb[80 + 22 * 2 + 2:80 + 22 * 2 + 20, 300:900] = 60  # and a thick band just under it (a beam row)
    staff = find_staff(rgb)
    assert staff is not None and round(staff.lines[0]) == 80


def test_two_systems_side_by_side_break_the_staff():
    rgb = _staff(180)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    assert staff_gaps(gray, [80 + 22 * i for i in range(6)], 22) == []
    gray[:, 600:680] = 245                               # the staff stops and starts again
    assert staff_gaps(gray, [80 + 22 * i for i in range(6)], 22) == [(600, 680)]
    gray[:, 600:1195] = 245                              # a short last staff, a dark border at the edge
    gray[60:220, 1195:1199] = 20
    assert staff_gaps(gray, [80 + 22 * i for i in range(6)], 22) == []


def test_staff_lines_without_a_staff_are_flagged_and_paper_is_not():
    rgb = np.full((300, 1200, 3), 245, np.uint8)
    for y in (60, 84, 108, 131, 170):                    # lines the finder cannot fit as a staff
        rgb[y, 20:1180] = 60
    assert find_staff(rgb) is None and unread_lines(rgb, 24)
    assert not unread_lines(rgb, 30)                     # not at this video's staff spacing
    assert not unread_lines(np.full((300, 1200, 3), 245, np.uint8), 24)
