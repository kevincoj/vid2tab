"""As written / as played: a page shown again with the same printed numbers is printed once (as written);
written-out repeats and unnumbered videos are kept.
python -m pytest tabvid/tests/test_replays.py -q"""
import cv2
import numpy as np

from tabvid import replays
from tabvid.report import Report

SP, TOP, MEASURE, PER_ROW, HEIGHT = 16, 60, 200, 4, 200
LINES = [TOP + i * SP for i in range(6)]


def page(music, number):
    """A row image: a TAB staff with PER_ROW measures; `music` seeds the fret digits (the same seed draws
    the same music); `number` is printed over the first measure (None: no number)."""
    image = np.full((HEIGHT, PER_ROW * MEASURE + 20, 3), 250, np.uint8)
    for y in LINES:
        cv2.line(image, (10, y), (image.shape[1] - 10, y), (150, 150, 150), 1)
    rng = np.random.default_rng(music)
    for k in range(PER_ROW + 1):
        x = 10 + k * MEASURE
        cv2.line(image, (x, LINES[0]), (x, LINES[-1]), (0, 0, 0), 2)
    for k in range(PER_ROW):
        for slot in range(4):
            line = int(rng.integers(0, 6))
            cv2.putText(image, str(int(rng.integers(0, 10))), (30 + k * MEASURE + slot * 40, LINES[line] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
    if number is not None:
        cv2.putText(image, str(number), (14, TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def setup(folder, pages):
    """pages: [(music seed, printed number or None)] in playback order. Writes rows and measures as
    stages 3-4 leave them; returns (rows, records, read) where read() fills number_box/number as the
    number reader would (a number box only where a number is printed, read correctly)."""
    for sub in ('rows', 'measures'):
        (folder / sub).mkdir()
    rows, records, printed = [], [], {}
    for k, (music, number) in enumerate(pages, 1):
        image = page(music, number)
        cv2.imwrite(str(folder / 'rows' / f'row-{k:02d}.png'), image)
        rows.append({'row': k, 'image': f'rows/row-{k:02d}.png', 'staff_lines': LINES, 'spacing': SP,
                     'seconds': [10.0 * (k - 1), 10.0 * k - .5]})
        for m in range(PER_ROW):
            n = len(records) + 1
            name = f'm{n:03d}.png'
            x0 = 10 + m * MEASURE
            cv2.imwrite(str(folder / 'measures' / name), image[:, x0:x0 + MEASURE + 2])
            records.append({'measure': n, 'row': k, 'in_row': m + 1, 'image': f'measures/{name}'})
            printed[n] = number if m == 0 else None

    def read(rs):
        for r in rs:
            n = printed[r['measure']]
            r['number_box'] = [3, TOP - 24, 3 + 9 * len(str(n)), TOP - 10] if n is not None else None
            r['number'] = n
        return {'font': 'test', 'sequence': 'consistent'} if any(printed.values()) else {}
    return rows, records, read


def test_a_replayed_page_is_printed_once_as_written(tmp_path):
    # Pages 1-3 (m1-4 unnumbered, then "5", "9"), then a D.C.: pages 1-2 again, numbers included.
    rows, records, read = setup(tmp_path, [(1, None), (2, 5), (3, 9), (1, None), (2, 5)])
    report = Report(tmp_path)
    facts = replays.apply(rows, records, tmp_path, report, replays.AS_WRITTEN, read)
    assert facts['replays'] == 'rows 4-5 (30.0-49.5s, m13-m20) = rows 1-2'
    assert facts['replayed_measures_dropped'] == 8
    assert [(r['measure'], r['row'], r['in_row'], r['image']) for r in records] == [
        (n, (n - 1) // 4 + 1, (n - 1) % 4 + 1, f'measures/m{n:03d}.png') for n in range(1, 13)]
    assert sorted(p.name for p in (tmp_path / 'measures').iterdir()) == [f'm{n:03d}.png' for n in range(1, 13)]
    assert all('number' not in r for r in records)          # the real number check reads again
    assert len(report.warnings) == 1 and 'printed once' in report.warnings[0][1]


def test_as_played_keeps_the_replay_and_says_so(tmp_path):
    rows, records, read = setup(tmp_path, [(1, None), (2, 5), (3, 9), (1, None), (2, 5)])
    report = Report(tmp_path)
    facts = replays.apply(rows, records, tmp_path, report, replays.AS_PLAYED, read)
    assert len(records) == 20 and facts['layout'] == 'as played'
    assert len(report.warnings) == 1
    assert report.warnings[0][1].startswith("this video replays pages: rows 4-5 (30.0-49.5s, m13-m20) = rows 1-2; "
                                            "choose 'as written'")


def test_a_written_out_repeat_with_other_numbers_is_kept(tmp_path):
    # The music of page 2 printed again as page 3 with its own numbers (9, not 5): two passes in the score.
    rows, records, read = setup(tmp_path, [(1, None), (2, 5), (2, 9), (3, 13)])
    report = Report(tmp_path)
    facts = replays.apply(rows, records, tmp_path, report, replays.AS_WRITTEN, read)
    assert len(records) == 16 and 'replays' not in facts and not report.warnings
    assert facts.get('same_music_other_numbers', facts.get('partial_replays')) is None or \
        'row 3' in facts.get('same_music_other_numbers', '')


def test_an_unnumbered_video_falls_back_to_as_played(tmp_path):
    rows, records, read = setup(tmp_path, [(1, None), (2, None), (3, None), (1, None), (2, None)])
    report = Report(tmp_path)
    facts = replays.apply(rows, records, tmp_path, report, replays.AS_WRITTEN, read)
    assert len(records) == 20 and 'replays' not in facts
    assert facts['replays_without_numbers'] == 'rows 4-5 (30.0-49.5s, m13-m20) = rows 1-2'
    assert len(report.warnings) == 1 and 'so the sheet is as played' in report.warnings[0][1]


def test_a_page_shown_again_with_a_difference_is_kept_and_warned(tmp_path):
    rows, records, read = setup(tmp_path, [(1, None), (2, 5), (3, 9), (2, 5)])
    image = cv2.imread(str(tmp_path / 'rows' / 'row-04.png'))
    cv2.putText(image, '7', (640, LINES[2] + 5), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(tmp_path / 'rows' / 'row-04.png'), image)      # one extra fret digit on the replay
    report = Report(tmp_path)
    facts = replays.apply(rows, records, tmp_path, report, replays.AS_WRITTEN, read)
    assert len(records) == 16 and 'replays' not in facts
    assert facts['partial_replays'].startswith('row 4 ~ row 2')
    assert len(report.warnings) == 1 and 'nearly repeats row 2' in report.warnings[0][1]


def test_scrolling_videos_stay_as_played(tmp_path):
    rows, records, read = setup(tmp_path, [(1, None), (2, 5), (1, None), (2, 5)])
    for row in rows:
        row['scroll'] = True
    report = Report(tmp_path)
    facts = replays.apply(rows, records, tmp_path, report, replays.AS_WRITTEN, read)
    assert len(records) == 16 and facts['replays'].startswith('not looked for')
    assert len(report.warnings) == 1 and 'scrolling video is always laid out as played' in report.warnings[0][1]


def test_the_layout_is_saved_per_run_and_reused(tmp_path):
    import json
    (tmp_path / 'config.json').write_text(json.dumps({'crop': [0, 0, 10, 10]}))
    assert replays.choose_layout(tmp_path) == replays.AS_PLAYED
    assert replays.choose_layout(tmp_path, replays.AS_PLAYED) == replays.AS_PLAYED
    assert 'layout' not in json.loads((tmp_path / 'config.json').read_text())   # no flag change: file untouched
    assert replays.choose_layout(tmp_path, replays.AS_WRITTEN) == replays.AS_WRITTEN
    assert replays.choose_layout(tmp_path) == replays.AS_WRITTEN                 # a re-run with no flag
    assert json.loads((tmp_path / 'config.json').read_text()) == {'crop': [0, 0, 10, 10], 'layout': 'as written'}
    assert replays.choose_layout(tmp_path, replays.AS_PLAYED) == replays.AS_PLAYED
    assert replays.choose_layout(tmp_path) == replays.AS_PLAYED
