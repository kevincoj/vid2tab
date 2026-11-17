"""Stage 2: find the TAB region, let the user confirm or redraw it, save it to config.json.

Then optional mask boxes: the user drags a box over each watermark or overlay inside the
crop; stage 3 paints them white in every frame before anything else (config.json "masks",
in frame coordinates). Asked once with the crop, again with --remask.

The guess: in ~10 evenly spaced frames, find a six-line staff, then grow a box
around it while rows/columns stay mostly light "paper". That keeps chord
diagrams above and rhythm stems below. The median box across frames wins.
"""
import json

import cv2
import numpy as np

from .vendor.layout.source import analyze_source

GUESS_SAMPLES = 10
PAPER_GRAY = 190       # a paper pixel is at least this light...
PAPER_CHROMA = 30      # ...and roughly gray (not the guitar or the performer)
PAPER_ROW = 0.6        # fraction of light pixels for a row/column to count as panel
GAP_ROWS = 4           # tolerate this many non-paper rows (e.g. a dense beam)
GAP_BELOW = .5         # ...below the staff, this many spacings if more: beams, stems and triplet brackets hang
                       # there (stack-gz2h: 5-6 px beams at a 16.5 px spacing stopped the box at y = 1040)
DIVIDER = .25          # a column under this fraction of paper pixels is a divider: the panel ends there
SIDE_RUN = 3           # sideways, past a gap, this many paper columns in a row before the panel goes on
TINT_FLOOR = 160       # sideways, a light pixel is paper if roughly gray or a pale tint (darkest channel at least
                       # this: a cursor highlight); a light-wood fretboard panel (darkest channel ~150) is not
BORDER_LINE = .5       # a row this share darker than the panel's paper (by PALE_STEP) is a border line...
PALE_STEP = 10
BORDER_ROWS = 8        # ...dropped from the top or bottom of the box, at most this many rows
INSET = 2              # then the top/bottom edge sits this many px inside the panel (unless at the frame edge)


def _read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _tab_staff(rgb):
    """The six-line staff to anchor on (the lowest one if several), or None."""
    staffs = [staff for system in analyze_source(rgb) for staff in system.staves if len(staff.lines) == 6]
    return max(staffs, key=lambda s: s.lines[0]) if staffs else None


def _grow(paper_fraction, start, stop, step, enough=PAPER_ROW, tolerate=GAP_ROWS, run=1):
    """Walk from start toward stop while the band stays paper; return the last paper index. Up to
    `tolerate` non-paper samples in a row are crossed; past a gap, paper counts again only after `run`
    paper samples in a row (sideways: the 1 px antialiased edge of a panel border is not the panel)."""
    edge, gap, seen = start, 0, run
    for i in range(start, stop, step):
        if paper_fraction[i] >= enough:
            seen += 1
            if seen >= run:
                edge, gap = i, 0
        else:
            seen = 0
            gap += 1
            if gap > tolerate:
                break
    return edge


def guess_box(rgb, inverted=False):
    """Light panel around the TAB staff in one frame, as (left, top, right, bottom), or None.

    inverted: look for light lines on a dark panel (the frame is inverted first)."""
    if inverted:
        rgb = 255 - rgb
    staff = _tab_staff(rgb)
    if staff is None:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    paper = (gray >= PAPER_GRAY) & (np.ptp(rgb, axis=2) <= PAPER_CHROMA)
    height, width = gray.shape
    rows = paper.mean(axis=1)
    top = _grow(rows, round(staff.lines[0]), -1, -1)
    bottom = _grow(rows, round(staff.lines[-1]), height, 1,
                   tolerate=max(GAP_ROWS, round(GAP_BELOW * staff.spacing))) + 1
    # Grow sideways from the staff too, so a separate panel beside the strip (a chord diagram box
    # behind a dark divider, a fretboard diagram) is not included. Top and bottom already exclude
    # the video; here a light pixel is paper if roughly gray or a pale tint (a cursor highlight is
    # still paper, light wood is not), and only a column that is mostly not paper stops it (not a
    # wavy arpeggio line or a beam).
    band = rgb[top:bottom]
    columns = ((gray[top:bottom] >= PAPER_GRAY)
               & ((np.ptp(band, axis=2) <= PAPER_CHROMA) | (band.min(axis=2) >= TINT_FLOOR))).mean(axis=0)
    staff_ink = _staff_columns(gray, staff)
    if not len(staff_ink):
        staff_ink = np.flatnonzero(gray[round(staff.lines[0])] < 150)
    if not len(staff_ink):
        top, bottom = _inside_border(gray, top, bottom, staff)
        return 0, top, width, bottom
    start = int(np.median(staff_ink))
    left = _grow(columns, start, -1, -1, DIVIDER, run=SIDE_RUN)
    right = _grow(columns, start, width, 1, DIVIDER, run=SIDE_RUN) + 1
    top, bottom = _inside_border(gray[:, left:right], top, bottom, staff)
    return left, top, right, bottom


LINE_CONTRAST = 15     # a staff line is this many gray levels darker than the paper midway between lines


def _staff_columns(gray, staff):
    """Columns where the staff runs: at least 5 of its 6 lines (darkest of the line's row and its
    neighbours) LINE_CONTRAST darker than the rows midway between lines. Pale lines count too (stack-gz2h:
    gray 170-200, so the top line alone, under 150, was only the fretboard's strings beside the panel)."""
    lines = [round(y) for y in staff.lines]
    if lines[0] < 1 or lines[-1] + 1 >= gray.shape[0]:
        return np.array([], int)
    dark = np.array([gray[y - 1:y + 2].min(axis=0) for y in lines], np.int16)
    mids = np.array([gray[(a + b) // 2] for a, b in zip(lines, lines[1:])], np.int16)
    paper = np.max(mids, axis=0)
    return np.flatnonzero(((paper[None] - dark) >= LINE_CONTRAST).sum(axis=0) >= 5)


def _inside_border(gray, top, bottom, staff):
    """Top and bottom moved a few px inside the panel: pale panel borders count as light, so the
    grown box can end exactly on one (G: the crop's first row was a gray border line, and its
    pieces went to single measures). Rows that are a line over most of the width are dropped,
    up to BORDER_ROWS, then INSET px more where the box does not reach the frame edge. Never
    into the staff. Left and right are left alone: a closing barline can sit at a panel's side."""
    paper = float(np.median(gray[top:bottom]))
    line = (gray < paper - PALE_STEP).mean(axis=1) >= BORDER_LINE
    first, last = round(staff.lines[0]) - INSET, round(staff.lines[-1]) + INSET + 1
    new_top = top
    while new_top < min(top + BORDER_ROWS, first) and line[new_top]:
        new_top += 1
    if top > 0:
        new_top = min(new_top + INSET, max(top, first))
    new_bottom = bottom
    while new_bottom > max(bottom - BORDER_ROWS, last) and line[new_bottom - 1]:
        new_bottom -= 1
    if bottom < gray.shape[0]:
        new_bottom = max(new_bottom - INSET, min(bottom, last))
    return new_top, new_bottom


def _draw(image, box, color, label=None):
    left, top, right, bottom = box
    cv2.rectangle(image, (left, top), (right - 1, bottom - 1), color, 4)
    if label:
        cv2.putText(image, label, (left + 8, max(30, top - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)


FONT, FONT_SCALE, FONT_THICK = cv2.FONT_HERSHEY_SIMPLEX, .65, 2
LINE_H = 28           # px per instruction line in the strip


def _wrap(text, width):
    """`text` split at spaces into lines that fit `width` px at the instruction font."""
    words, lines = text.split(' '), ['']
    for word in words:
        trial = f'{lines[-1]} {word}'.strip()
        if lines[-1] and cv2.getTextSize(trial, FONT, FONT_SCALE, FONT_THICK)[0][0] > width - 20:
            lines.append(word)
        else:
            lines[-1] = trial
    return lines


def with_strip(picture, note):
    """(picture with a black instruction strip stacked above it, strip height). The strip never
    covers the picture, so the whole frame stays visible."""
    lines = _wrap(note, picture.shape[1])
    strip = np.zeros((LINE_H * len(lines) + 12, picture.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(strip, line, (10, LINE_H * (i + 1)), FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)
    return np.vstack([strip, picture]), strip.shape[0]


def to_frame(roi, strip, scale, shape, offset=(0, 0)):
    """A box (x, y, w, h) dragged in a window made by with_strip, as (left, top, right, bottom) in
    frame coordinates: the strip is taken off, the box is clipped to the picture (`shape`: its
    height and width as shown), then scaled back and moved by `offset` (the shown picture's
    top-left corner in the frame). None if nothing of it lies on the picture."""
    x, y, w, h = roi
    x0, x1 = max(0, x), min(shape[1], x + w)
    y0, y1 = max(0, y - strip), min(shape[0], y + h - strip)
    if x1 <= x0 or y1 <= y0:
        return None
    left, top = offset
    return (left + round(x0 / scale), top + round(y0 / scale), left + round(x1 / scale), top + round(y1 / scale))


def _ask_user(rgb, guess):
    """Show the guess; Enter keeps it, dragging a new box replaces it. Returns (box, how)."""
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if guess:
        _draw(image, guess, (0, 200, 0))     # labelled in the strip, so nothing covers the frame
    note = ('GREEN BOX = GUESS   |   ' if guess else 'NO GUESS   |   ') + \
        'ENTER/SPACE = accept green box   |   drag a new box, then ENTER   |   C = keep guess'
    scale = min(1.0, 1400 / image.shape[1], 850 / image.shape[0])
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    window = 'tabvid: confirm TAB region'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    picture, strip = with_strip(shown, note)
    x, y, w, h = cv2.selectROI(window, picture, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)  # let macOS actually close the window
    box = to_frame((x, y, w, h), strip, scale, shown.shape) if w > 10 and h > 10 else None
    if box and box[2] - box[0] > 10 and box[3] - box[1] > 10:
        return box, 'user-drawn'
    return guess, 'guess-accepted'


def sample_guesses(frames):
    """(sampled frame indices, {index: box or None}, {index: box} found, inverted?) over GUESS_SAMPLES
    evenly spaced frames; the guess is the median of the found boxes."""
    picks = np.linspace(0, len(frames) - 1, min(GUESS_SAMPLES, len(frames))).round().astype(int)
    boxes = {int(i): guess_box(_read_rgb(frames[i])) for i in picks}
    found = {i: b for i, b in boxes.items() if b}
    if not found:   # maybe a dark panel with light lines
        inverted = {int(i): guess_box(_read_rgb(frames[i]), inverted=True) for i in picks}
        if any(inverted.values()):
            return picks, inverted, {i: b for i, b in inverted.items() if b}, True
    return picks, boxes, found, False


def initial_guess(frames):
    """The crop guess over GUESS_SAMPLES sampled frames, as the confirm window shows it: a dict with
    guess (left, top, right, bottom), shown (the frame index closest to the guess), boxes ({sampled
    index: box or None}), found and samples (counts), flipped (found on inverted frames) and
    warnings (texts for the report). Used by choose_crop and the web app."""
    picks, boxes, found, flipped = sample_guesses(frames)
    height, width = _read_rgb(frames[0]).shape[:2]
    warnings = []
    if found:
        guess = tuple(int(v) for v in np.median(list(found.values()), axis=0))
        shown = min(found, key=lambda i: sum(abs(a - b) for a, b in zip(found[i], guess)))
    else:
        guess, shown = (0, height // 2, width, height), int(picks[len(picks) // 2])
        warnings.append('No six-line staff found in sampled frames; guessed the bottom half of the frame.')
    if found and len(found) < len(picks) / 2:
        warnings.append(f'TAB staff found in only {len(found)}/{len(picks)} sampled frames; check the crop.')
    return {'guess': guess, 'shown': shown, 'boxes': boxes, 'found': len(found), 'samples': len(picks),
            'flipped': flipped, 'warnings': warnings}


def save_crop(run_dir, box, how, info):
    """Write the chosen box into config.json with the guess it came from (`info` from initial_guess).
    A new crop clears the saved polarity, so it is decided again."""
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.pop('inverted', None)   # a new crop: decide polarity again
    config.pop('polarity_samples', None)
    config.update(crop=[int(v) for v in box], crop_how=how, crop_guess=list(info['guess']),
                  crop_shown_frame=info['shown'],
                  crop_guess_frames={str(i): list(b) if b else None for i, b in info['boxes'].items()})
    config_path.write_text(json.dumps(config, indent=2) + '\n')


def save_masks(run_dir, masks):
    """Write the mask boxes (frame coordinates) into config.json."""
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config['masks'] = [[int(v) for v in m] for m in masks]
    config_path.write_text(json.dumps(config, indent=2) + '\n')


def choose_crop(frames, run_dir, report, *, recrop=False, accept_guess=False):
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if 'crop' in config and not recrop:
        report.stage('crop', box=config['crop'], how=config['crop_how'] + ' (saved)')
        return tuple(config['crop'])

    info = initial_guess(frames)
    if info['flipped']:
        report.stage('crop', guess='staff found on inverted frames (light lines on a dark panel)')
    for warning in info['warnings']:
        report.warn('crop', warning)
    guess = info['guess']
    rgb = _read_rgb(frames[info['shown']])
    box, how = (guess, 'guess-accepted (no window)') if accept_guess else _ask_user(rgb, guess)
    save_crop(run_dir, box, how, info)
    report.stage('crop', box=list(box), how=how, guess=list(guess),
                 staff_found_in=f"{info['found']}/{info['samples']} sampled frames")
    return box



def decide_polarity(frames, box, run_dir, report):
    """True if the TAB is light on dark (inverted), decided once per video, saved in config.json.

    In ~10 sampled frames, count crops with a staff and a barline as they are and inverted;
    inverted wins only if it finds more. Stage 3 then inverts every frame on load, so the rest of the pipeline sees
    dark lines on paper as usual.
    """
    from .staff import find_staff
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text())
    if 'inverted' not in config:
        left, top, right, bottom = box
        picks = np.linspace(0, len(frames) - 1, min(GUESS_SAMPLES, len(frames))).round().astype(int)
        crops = [np.ascontiguousarray(_read_rgb(frames[i])[top:bottom, left:right]) for i in picks]
        def ruled(rgb):   # a staff with a barline (the dark gaps between light lines can pass as a bare staff)
            staff = find_staff(rgb)
            return staff is not None and len(staff.barlines) > 0
        normal = sum(ruled(c) for c in crops)
        flipped = sum(ruled(np.ascontiguousarray(255 - c)) for c in crops) if normal < len(crops) / 2 else 0
        config['inverted'] = flipped > normal
        config['polarity_samples'] = {'normal': normal, 'inverted': flipped, 'of': len(crops)}
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    samples = config.get('polarity_samples', {})
    report.stage('crop', polarity='inverted (light TAB on a dark panel)' if config['inverted'] else 'normal',
                 polarity_samples=samples or 'saved')
    return config['inverted']


def _ask_masks(rgb, box):
    """Let the user drag any number of boxes inside the crop, then confirm them on a preview.

    Returns the boxes in frame coordinates. Repeated boxes (ENTER pressed twice) count once.
    """
    left, top, right, bottom = box
    image = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2BGR)
    scale = min(2.5, 1800 / image.shape[1], 800 / image.shape[0])
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    window = 'tabvid: mask boxes'

    while True:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        picture, strip = with_strip(shown, 'MASKS: drag a box over a watermark/overlay, ENTER to add it; '
                                           'repeat; ESC when done (ESC now = none)')
        picked = cv2.selectROIs(window, picture, showCrosshair=False, fromCenter=False)
        masks = []
        for roi in (picked if len(picked) else []):
            box_ = to_frame(roi, strip, scale, shown.shape, (left, top))
            if box_ and roi[2] > 2 and roi[3] > 2 and list(box_) not in masks:
                masks.append(list(box_))
        preview = shown.copy()
        for l, t, r, b in masks:
            x0, y0, x1, y1 = (round((v - o) * scale) for v, o in ((l, left), (t, top), (r, left), (b, top)))
            preview[y0:y1, x0:x1] = 255
            cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
        note = f'{len(masks)} mask box(es), painted white as shown. ENTER = save, R = redo'
        cv2.imshow(window, with_strip(preview, note)[0])
        key = 0
        while key not in (13, 10, ord('r'), ord('R')):
            key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        if key in (13, 10):
            return masks


def choose_masks(frames, box, run_dir, report, *, remask=False, accept_guess=False):
    """Mask boxes (frame coordinates) to paint white in every frame; saved in config.json."""
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text())
    if 'masks' in config and not remask:
        masks = config['masks']
    elif accept_guess and not remask:
        masks = []
    else:
        # A frame from mid-video: overlays that fade in after the start (V5's watermark) show.
        masks = _ask_masks(_read_rgb(frames[len(frames) // 2]), box)
        save_masks(run_dir, masks)
    report.stage('crop', masks=masks or 'none')
    return masks


def crop_debug(frames, box, run_dir):
    """debug/2-crop.jpg: one frame with the box; debug/2-crop-samples.jpg: the crop at 4 times."""
    config = json.loads((run_dir / 'config.json').read_text())
    guess = config.get('crop_guess')
    shown = config.get('crop_shown_frame', len(frames) // 2)
    image = cv2.imread(str(frames[shown]))
    if guess and list(guess) != list(box):
        _draw(image, guess, (0, 200, 255), 'guess')
    _draw(image, box, (0, 200, 0), f'crop {list(box)}')
    for mask in config.get('masks', []):
        _draw(image, mask, (0, 0, 255), 'mask')
    cv2.imwrite(str(run_dir / 'debug' / '2-crop.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    left, top, right, bottom = box
    picks = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    strips = []
    for i in picks:
        strip = cv2.imread(str(frames[i]))[top:bottom, left:right].copy()
        cv2.putText(strip, f'#{i}', (6, 24), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
        strips.append(cv2.copyMakeBorder(strip, 0, 6, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 255)))
    cv2.imwrite(str(run_dir / 'debug' / '2-crop-samples.jpg'), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 85])
