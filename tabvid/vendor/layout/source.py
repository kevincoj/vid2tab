"""Analyze original coordinates without erasing overlays or reconstructing music.

This path proposes geometry only. Nothing here certifies that a measure is visible.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from .models import Staff, Barline


@dataclass(frozen=True)
class SourceConfig:
    horizontal_fraction: float = .04
    horizontal_coverage: float = .20
    ink_threshold: int = 210
    minimum_spacing: int = 4
    line_tolerance_fraction: float = .12
    system_gap_spacing: float = 10
    vertical_coverage: float = .85
    boundary_merge_spacing: float = .55
    color_difference: int = 20
    minimum_chroma_difference: int = 35
    pale_cursor_chroma: int = 16
    pale_cursor_hue_min: int = 65
    pale_cursor_hue_max: int = 140
    minimum_color_saturation: int = 16
    minimum_color_value: int = 60
    color_neighbor_spacing: float = .20
    colored_column_coverage: float = .30
    edge_safety_spacing: float = .25
    barline_staff_exclusion_spacing: float = .08
    barline_profile_radius_spacing: float = .5
    barline_center_residual_spacing: float = .04
    barline_slope_samples: int = 33
    barline_wavy_gap_fraction: float = .5
    # tabvid edit: thickest pixel-row run that counts as one staff line (default as before)
    max_line_rows: int = 5
    # tabvid edit: a line run this share of the spacing or less from a fitted line is a piece of it, not an interleaved
    # line (default 0: as before)
    split_line: float = 0


@dataclass(frozen=True)
class SourceSystem:
    bounds: tuple[int, int, int, int]
    staves: tuple[Staff, ...]
    barlines: tuple[Barline, ...]
    warnings: tuple[str, ...]
    staff_bounds: tuple[int, int, int, int]


DEFAULT = SourceConfig()


def _analysis_gray(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Normalize gradients for detection only. Returned music pixels always come from rgb.
    size = max(15, min(61, rgb.shape[1] // 40))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    return np.clip(gray.astype(np.float32) / np.maximum(background, 1) * 255, 0, 255).astype(np.uint8)


def _staffs(horizontal, config):
    height, width = horizontal.shape
    rows = np.flatnonzero((horizontal > 0).sum(axis=1) >= width * config.horizontal_coverage)
    if not len(rows):
        return []
    runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    centers = np.array([float(run.mean()) for run in runs if len(run) <= max(config.max_line_rows, height * .01)])
    candidates = {}
    for start in range(len(centers)):
        for second in range(start + 1, len(centers)):
            spacing = centers[second] - centers[start]
            if not config.minimum_spacing <= spacing <= height * .1:
                continue
            for count in (6, 5):
                expected = centers[start] + np.arange(count) * spacing
                indices = tuple(int(np.argmin(abs(centers - y))) for y in expected)
                # tabvid change: refit the spacing from the first and last matched lines.
                # One adjacent gap (e.g. 21px in a 21.7px staff) otherwise drifts past the
                # tolerance by the sixth line and a real staff is rejected.
                if len(set(indices)) == count:
                    fitted = (centers[indices[-1]] - centers[indices[0]]) / (count - 1)
                    expected = centers[start] + np.arange(count) * fitted
                    indices = tuple(int(np.argmin(abs(centers - y))) for y in expected)
                errors = abs(centers[list(indices)] - expected)
                if len(set(indices)) != count or errors.max() > max(1.5, spacing * config.line_tolerance_fraction):
                    continue
                # Interleaved lines indicate a harmonic of the real spacing, not a staff.
                between = ((centers >= expected[0]) & (centers <= expected[-1])).sum()
                if config.split_line:   # tabvid edit: pieces of a fitted line broken by noise are not extra lines
                    near = np.abs(centers[:, None] - centers[list(indices)][None, :]).min(axis=1) <= config.split_line * spacing
                    between -= int((near & (centers >= expected[0]) & (centers <= expected[-1])).sum()) - count
                if between > count:
                    continue
                candidates[indices] = min(candidates.get(indices, float('inf')), float(errors.mean() / spacing))
    selected, used = [], set()
    for indices, error in sorted(candidates.items(), key=lambda item: (-len(item[0]), item[1], item[0])):
        if used.intersection(indices):
            continue
        lines = tuple(float(centers[i]) for i in indices)
        selected.append(Staff(lines, float(np.median(np.diff(lines)))))
        used.update(indices)
    return sorted(selected, key=lambda staff: staff.lines[0])



def _straight_stroke(ink, staff, columns, config):
    """A long center column is insufficient: arpeggios have wavy stroke edges.

    Ignore horizontal staff rows, then measure each cross-section's center.
    Fit linear slope so a slightly tilted straight bar is still acceptable.
    This is detection evidence only; source music pixels are never changed.
    """
    rows=np.arange(round(staff.lines[0]),round(staff.lines[-1])+1)
    distance=np.min(abs(rows[:,None]-np.asarray(staff.lines)[None,:]),axis=1)
    rows=rows[distance>max(1,staff.spacing*config.barline_staff_exclusion_spacing)]
    if len(rows)<2:return False
    x=int(columns[np.argmax(ink[np.ix_(rows,columns)].mean(axis=0))])
    rows=rows[ink[rows,x]>0]
    if len(rows)<2:return False
    radius=max(3,round(staff.spacing*config.barline_profile_radius_spacing))
    left=max(0,x-radius);right=min(ink.shape[1],x+radius+1)
    window=ink[np.ix_(rows,np.arange(left,right))]>0
    center=x-left
    left_white=~window[:,:center+1][:,::-1]
    right_white=~window[:,center:]
    dl=np.where(left_white.any(axis=1),left_white.argmax(axis=1),left_white.shape[1])
    dr=np.where(right_white.any(axis=1),right_white.argmax(axis=1),right_white.shape[1])
    centers=x+(dr-dl)/2
    # Median pairwise slopes resist a notehead near either staff end. Ordinary
    # least squares lets that local glyph tilt the fit across the whole bar.
    sample=np.linspace(0,len(rows)-1,min(config.barline_slope_samples,len(rows))).round().astype(int)
    dy=rows[sample][None,:]-rows[sample][:,None]
    dc=centers[sample][None,:]-centers[sample][:,None]
    slopes=dc[dy>0]/dy[dy>0]
    slope=float(np.median(slopes)) if len(slopes) else 0.
    offset=float(np.median(centers-slope*rows))
    residual=centers-(slope*rows+offset)
    # Attached music or a player button can contaminate one portion of a bar.
    # A wavy arpeggio varies within most staff spaces, rather than merely
    # moving the profile once near the top or bottom of the staff.
    spreads=[]
    for a,b in zip(staff.lines,staff.lines[1:]):
        section=residual[(rows>a)&(rows<b)]
        if len(section)>=2:
            spreads.append(float(np.percentile(section,75)-np.percentile(section,25)))
    tolerance=max(1.,staff.spacing*config.barline_center_residual_spacing)
    return sum(spread>tolerance for spread in spreads)<=len(spreads)*config.barline_wavy_gap_fraction


def analyze_source(rgb, config=DEFAULT):
    """Return staff groups and original-source barline candidates, without pixel mutation."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('source must be a uint8 RGB image')
    h, w = rgb.shape[:2]
    gray = _analysis_gray(rgb)
    ink = (gray < config.ink_threshold).astype(np.uint8)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                     cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(w * config.horizontal_fraction)), 1)))
    staves = _staffs(horizontal, config)
    if not staves:
        return []
    groups = [[staves[0]]]
    for staff in staves[1:]:
        previous = groups[-1][-1]
        gap = staff.lines[0] - previous.lines[-1]
        if gap <= config.system_gap_spacing * max(previous.spacing, staff.spacing) and len(groups[-1]) < 3:
            groups[-1].append(staff)
        else:
            groups.append([staff])
    result = []
    # Saturated dark vertical ink can be a playhead; never redraw it as a barline.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    for index, group in enumerate(groups):
        top = 0 if index == 0 else int((groups[index-1][-1].lines[-1] + group[0].lines[0]) / 2)
        bottom = h if index + 1 == len(groups) else int((group[-1].lines[-1] + groups[index+1][0].lines[0]) / 2)
        spacing = float(np.median([staff.spacing for staff in group]))
        radius = max(6, round(spacing * config.color_neighbor_spacing))
        left_difference = cv2.absdiff(rgb,np.roll(rgb,radius,axis=1)).max(axis=2)
        right_difference = cv2.absdiff(rgb,np.roll(rgb,-radius,axis=1)).max(axis=2)
        # Local contrast on both sides distinguishes a thin cursor from a broad tint.
        chroma = rgb.max(axis=2).astype(np.int16)-rgb.min(axis=2)
        distinctive_color = ((chroma > config.minimum_chroma_difference) |
                             ((chroma > config.pale_cursor_chroma) &
                              (hsv[:,:,0] >= config.pale_cursor_hue_min) &
                              (hsv[:,:,0] <= config.pale_cursor_hue_max)))
        colored_ink = (distinctive_color &
                       (hsv[:,:,1] > config.minimum_color_saturation) &
                       (hsv[:,:,2] > config.minimum_color_value) &
                       (left_difference > config.color_difference) &
                       (right_difference > config.color_difference))
        scores, colored_scores = [], []
        for staff in group:
            y0, y1 = int(round(staff.lines[0])), int(round(staff.lines[-1])) + 1
            span = y1 - y0
            vertical = cv2.morphologyEx(ink[y0:y1], cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, int(span * config.vertical_coverage)))))
            scores.append(vertical.mean(axis=0))
            colored_scores.append(colored_ink[y0:y1].mean(axis=0))
        confidence = np.min(scores, axis=0)
        cursor = np.max(colored_scores, axis=0) >= config.colored_column_coverage
        cursor = cv2.dilate(cursor.astype(np.uint8).reshape(1,-1), np.ones((1,5),np.uint8))[0] > 0
        spacing = float(np.median([staff.spacing for staff in group]))
        edge = max(2, round(spacing * config.edge_safety_spacing))
        trusted_columns = (confidence >= config.vertical_coverage) & ~cursor
        trusted_columns[:edge] = False
        trusted_columns[-edge:] = False
        columns = np.flatnonzero(trusted_columns)
        merge = max(1, round(spacing * config.boundary_merge_spacing))
        bars = []
        rejected_wavy = False
        if len(columns):
            for run in np.split(columns, np.where(np.diff(columns) > merge)[0] + 1):
                if all(_straight_stroke(ink,staff,run,config) for staff in group):
                    bars.append(Barline(int(round(run.mean())), int(run[0]), int(run[-1]), float(confidence[run].max())))
                else:
                    rejected_wavy = True
        warnings = ['Candidate geometry: visibility, header and annotation extent require review']
        if rejected_wavy:
            warnings.append('Wavy vertical musical mark retained; not used as a measure boundary')
        if cursor.any():
            warnings.append('Colored vertical overlay detected; concealed music is not reconstructed')
        if not bars:
            warnings.append('No reliable original-source barlines detected')
        staff_rows = [int(round(y)) for staff in group for y in staff.lines]
        cols = np.flatnonzero(horizontal[staff_rows].any(axis=0))
        staff_bounds = (int(cols[0]),int(round(group[0].lines[0])),int(cols[-1])+1,int(round(group[-1].lines[-1]))+1)
        result.append(SourceSystem((0, top, w, bottom), tuple(group), tuple(bars), tuple(warnings),staff_bounds))
    return result
