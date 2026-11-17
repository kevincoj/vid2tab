"""Crop guess: python -m pytest tabvid/tests -q"""
import cv2
import numpy as np

from tabvid import crop


def frame(border=True):
    """A 1280x720 frame: dark video above, a light TAB panel from y = 400 down to y = 700 with a
    pale gray border line on its first row, then dark again."""
    rgb = np.full((720, 1280, 3), 40, np.uint8)
    rgb[400:700] = 250
    if border:
        rgb[400, :] = 215                      # the panel border: light enough to count as paper
        rgb[699, :] = 215
    for k in range(6):
        rgb[480 + 20 * k, 40:1240] = 60          # six staff lines, spacing 20
    for x in (40, 640, 1238):
        rgb[480:581, x:x + 2] = 30
    for x in range(100, 1200, 60):
        rgb[440:452, x:x + 8] = 20             # chord names above the staff
    return rgb


def test_crop_guess_sits_inside_the_panel_border_not_on_it():
    # Single example (blind2 G): the guess began exactly on a pale panel border.
    left, top, right, bottom = crop.guess_box(frame())
    assert 401 + crop.INSET <= top <= 440            # past the border row, above the chord names
    assert 581 < bottom <= 699 - crop.INSET
    plain = crop.guess_box(frame(border=False))
    assert plain[1] == 400 + crop.INSET and plain[3] == 700 - crop.INSET


def test_crop_guess_at_the_frame_edge_is_not_moved_in():
    rgb = frame(border=False)
    rgb[700:] = 250                                    # the panel runs to the bottom of the frame
    assert crop.guess_box(rgb)[3] == 720


def gz_frame(beam=6, panel=True, line_gray=60):
    """stack-gz2h's layout: dark video above, a white TAB panel from y = 400 to the frame bottom with
    a light-wood fretboard panel on its left (x < 200, behind a 3 px dark border and a 1 px pale edge),
    staff lines at spacing 16, and beams `beam` px thick plus triplet brackets below the staff."""
    rgb = np.full((720, 1280, 3), 40, np.uint8)
    rgb[400:] = 255
    if panel:
        rgb[400:, :196] = (218, 194, 154)                          # light wood: gray ~197, darkest channel 154
        rgb[400:, 60:200:30] = 20                                  # strings
        rgb[400:, 196] = (216, 202, 174)                           # the border's pale antialiased edge
        rgb[400:, 197:200] = 20                                    # the border
    for k in range(6):
        rgb[480 + 16 * k, 210:1270] = line_gray
    for x in (210, 740, 1268):
        rgb[480:561, x:x + 2] = 30
    for x in range(240, 1240, 70):
        rgb[561:600, x:x + 2] = 20                                 # stems
        rgb[600:600 + beam, x:x + 50] = 20                         # a beam
        rgb[630:632, x:x + 50] = 20                                # a triplet bracket, ends hooked up
        rgb[624:632, x:x + 2] = 20
    return rgb


def test_beams_and_brackets_below_the_staff_are_inside_the_guess():
    # Single example (stack-gz2h): 5-6 px beams at a 16.5 px spacing stopped the box above them.
    assert crop.guess_box(gz_frame(beam=6))[3] == 720
    assert crop.guess_box(gz_frame(beam=10))[3] < 600              # a band over half a spacing still stops it


def test_the_guess_stops_at_a_fretboard_panel_beside_the_tab():
    left, _, right, _ = crop.guess_box(gz_frame())
    assert left == 200 and right == 1280                            # not into the wood, not on its border


def test_pale_staff_lines_anchor_the_guess_in_the_tab_panel():
    # stack-gz2h's lines are gray 170-200: the top line alone (< 150) was only the fretboard's strings.
    left, _, right, _ = crop.guess_box(gz_frame(line_gray=185))
    assert left == 200 and right == 1280


# Crop and mask windows: instructions sit in a strip above the frame, boxes map back to it.
NOTE = 'ENTER/SPACE = accept green box   |   drag a new box, then ENTER   |   C = keep guess'


def test_strip_is_stacked_above_and_never_covers_the_frame():
    frame = np.random.default_rng(0).integers(0, 255, (300, 500, 3), np.uint8)
    picture, strip = crop.with_strip(frame, NOTE)
    assert picture.shape == (300 + strip, 500, 3)
    assert (picture[strip:] == frame).all()                  # every frame pixel still shown
    assert picture[:strip].any()                             # the instructions are drawn


def test_long_instructions_wrap_to_the_window_width():
    narrow = np.full((100, 360, 3), 255, np.uint8)
    picture, strip = crop.with_strip(narrow, NOTE)
    lines = crop._wrap(NOTE, 360)
    assert len(lines) > 1 and strip == crop.LINE_H * len(lines) + 12
    assert all(cv2.getTextSize(l, crop.FONT, crop.FONT_SCALE, crop.FONT_THICK)[0][0] <= 340 for l in lines)
    assert ' '.join(lines).split() == NOTE.split()          # no word lost


def test_a_box_drawn_in_the_window_maps_to_frame_coordinates():
    scale, strip, shown = .5, 40, (400, 960)                 # a 1920x800 frame shown at half size
    box = (200, 100, 1000, 500)                              # frame coordinates
    roi = (100, 50 + strip, 400, 200)                        # the same box dragged in the window
    assert crop.to_frame(roi, strip, scale, shown) == box
    assert crop.to_frame(roi, strip, scale, shown, (30, 7)) == (230, 107, 1030, 507)   # mask: crop offset


def test_a_box_dragged_from_the_strip_or_past_the_edge_is_clipped_to_the_frame():
    scale, strip, shown = 1.0, 40, (300, 500)
    assert crop.to_frame((10, 20, 100, 120), strip, scale, shown) == (10, 0, 110, 100)   # bottom kept
    assert crop.to_frame((450, 250, 200, 200), strip, scale, shown) == (450, 210, 500, 300)
    assert crop.to_frame((10, 5, 100, 30), strip, scale, shown) is None                  # only on the strip


def test_guess_and_save_write_the_same_config_as_the_cli(tmp_path):
    # The web app saves a crop through initial_guess + save_crop/save_masks; choose_crop and
    # choose_masks (the CLI) go through the same functions, so config.json is the same either way.
    import json
    from tabvid.report import Report
    paths = []
    for i in range(4):
        path = tmp_path / f'f{i:03d}.jpg'
        cv2.imwrite(str(path), cv2.cvtColor(frame(), cv2.COLOR_RGB2BGR))
        paths.append(path)
    cli, web = tmp_path / 'cli', tmp_path / 'web'
    cli.mkdir(), web.mkdir()
    box = crop.choose_crop(paths, cli, Report(cli), accept_guess=True)
    crop.choose_masks(paths, box, cli, Report(cli), accept_guess=True)
    info = crop.initial_guess(paths)
    assert info['guess'] == tuple(box) and info['found'] == 4 and not info['warnings']
    crop.save_crop(web, info['guess'], 'guess-accepted (no window)', info)
    assert json.loads((cli / 'config.json').read_text()) == json.loads((web / 'config.json').read_text())
    crop.save_masks(web, [])   # the CLI with --accept-guess leaves masks unsaved; the web app saves its choice
    (web / 'config.json').write_text(json.dumps({**json.loads((web / 'config.json').read_text()),
                                                 'inverted': False, 'polarity_samples': {}}))
    crop.save_crop(web, (0, 0, 10, 10), 'user-drawn', info)
    saved = json.loads((web / 'config.json').read_text())
    assert saved['crop'] == [0, 0, 10, 10] and 'inverted' not in saved and saved['masks'] == []


def test_each_stage_prints_one_progress_line(tmp_path, capsys, monkeypatch):
    from tabvid import __main__ as cli
    cli._stage('rows')
    assert capsys.readouterr().out == 'Stage 3/5: rows\n'
