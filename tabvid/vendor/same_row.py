"""same_row and its chromatic-overlay helper, copied verbatim from the archived tabsheet/video.py."""
import cv2
import numpy as np


def _overlay(rgb):
    """Conservative chromatic ink veto. Preserve each original frame for review."""
    chromatic = np.ptp(rgb, axis=2) >= 28
    count, labels, stats, _ = cv2.connectedComponentsWithStats(chromatic.astype('uint8'))
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 16
    return keep[labels]


def same_row(a, b):
    """Match successive views with the same boundaries and agreement on clear ink."""
    if (a.rgb.shape != b.rgb.shape or len(a.bars) != len(b.bars)
            # tabvid edit: two frames with a staff and no barline (default=0) are left to the tests below
            or max((abs(x-y) for x, y in zip(a.bars, b.bars)), default=0) > 2
            or max(abs(x-y) for x, y in zip(a.lines, b.lines)) > 2):
        return False
    clear = ~(_overlay(a.rgb) | _overlay(b.rgb))
    if clear.mean() < .65:
        return False
    gray_a = cv2.cvtColor(a.rgb, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b.rgb, cv2.COLOR_RGB2GRAY)
    # Video compression can shift an otherwise identical gray staff rule
    # across the ink threshold. Strong note ink and white gaps still count.
    rule = np.zeros(gray_a.shape[0], dtype=bool)
    for y in (*a.lines, *b.lines):
        center = round(y)
        rule[max(0, center - 1):center + 2] = True
    clear &= ~(rule[:, None] & (gray_a >= 110) & (gray_a <= 150)
               & (gray_b >= 110) & (gray_b <= 150))
    ink_a = gray_a < 130
    ink_b = gray_b < 130
    if np.count_nonzero(ink_a & ink_b & clear) < .45 * max(1, (ink_a & clear).sum(), (ink_b & clear).sum()):
        return False
    kernel = np.ones((3, 3), np.uint8)
    different = ((ink_a & ~cv2.dilate(ink_b.astype('uint8'), kernel).astype(bool))
                 | (ink_b & ~cv2.dilate(ink_a.astype('uint8'), kernel).astype(bool))) & clear
    _, _, stats, _ = cv2.connectedComponentsWithStats(different.astype('uint8'))
    return different.sum() <= max(12, .025 * (ink_a & clear).sum()) and all(
        stats[1:, cv2.CC_STAT_AREA] < 18
    )
