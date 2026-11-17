"""--start/--end: time parsing, the saved range, and the FFmpeg cut (on a drawn video)."""
import json

import cv2
import numpy as np
import pytest

from tabvid import intake
from tabvid.report import Report


def test_parse_and_format_times():
    assert intake.parse_time('12:30') == 750
    assert intake.parse_time('1:02:30') == 3750
    assert intake.parse_time('75') == 75 and intake.parse_time('7.5') == 7.5
    for bad in ('12:75', 'abc', '1:2:3:4', '-5'):
        with pytest.raises(ValueError):
            intake.parse_time(bad)
    assert intake.format_time(750) == '12:30' and intake.format_time(3750.5) == '1:02:30.5'


def test_range_is_saved_reused_replaced_and_cleared(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'crop': [0, 0, 10, 10]}))
    assert intake.video_range(tmp_path, None, None) is None             # never set: whole video
    assert intake.video_range(tmp_path, '12:30', '15:45') == {'start': 750, 'end': 945}
    (tmp_path / 'video.mp4').write_bytes(b'clip')
    (tmp_path / 'frames').mkdir()
    assert intake.video_range(tmp_path, None, None) == {'start': 750, 'end': 945}   # re-run: reused
    assert (tmp_path / 'video.mp4').exists() and (tmp_path / 'frames').exists()
    assert intake.video_range(tmp_path, '12:30', '15:45') == {'start': 750, 'end': 945}  # same: kept
    assert (tmp_path / 'video.mp4').exists()
    assert intake.video_range(tmp_path, '13:00', None) == {'start': 780, 'end': None}    # changed
    assert not (tmp_path / 'video.mp4').exists() and not (tmp_path / 'frames').exists()
    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['range'] == {'start': 780, 'end': None} and config['crop'] == [0, 0, 10, 10]
    assert intake.video_range(tmp_path, '0', None) is None              # --start 0: whole video again
    assert 'range' not in json.loads((tmp_path / 'config.json').read_text())
    with pytest.raises(ValueError):
        intake.video_range(tmp_path, '5:00', '4:00')


def test_range_change_never_deletes_the_source_video(tmp_path):
    intake.video_range(tmp_path, '0:10', '0:20')
    (tmp_path / 'video.mp4').write_bytes(b'clip')
    with pytest.raises(ValueError):
        intake.video_range(tmp_path, '0:05', '0:20', source=str(tmp_path / 'video.mp4'))
    assert (tmp_path / 'video.mp4').exists()


def _counting_video(path, seconds=10, fps=10):
    """Each second drawn at its own gray level (second k: 20 * k)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (64, 48))
    for i in range(seconds * fps):
        writer.write(np.full((48, 64, 3), 20 * (i // fps), np.uint8))
    writer.release()


def test_local_video_is_cut_to_the_range(tmp_path):
    source = tmp_path / 'source.mp4'
    _counting_video(source)
    run = tmp_path / 'run'
    run.mkdir()
    report = Report(run)
    section = intake.video_range(run, '0:03', '0:06', str(source))
    video = intake.get_video(str(source), run, report, section)
    assert abs(intake._seconds(video) - 3) < .2
    frames = intake.extract_frames(video, run, 2, report)
    levels = [round(float(cv2.imread(str(f)).mean()) / 20) for f in frames]
    assert levels == [3, 3, 4, 4, 5, 5]                                    # seconds 3, 4, 5 only
    assert report.stages['intake']['range'] == '0:03-0:06'
    assert not [w for _, w in report.warnings if 'range' in w]


def test_range_past_the_end_is_warned(tmp_path):
    source = tmp_path / 'source.mp4'
    _counting_video(source)
    run = tmp_path / 'run'
    run.mkdir()
    report = Report(run)
    section = intake.video_range(run, '0:08', '0:30', str(source))
    intake.get_video(str(source), run, report, section)
    assert any('past the end of the video' in w for _, w in report.warnings)


def _video_with_sound(path, seconds=10):
    """A counting picture (gray level = second) with a tone whose pitch rises each second: FFmpeg lavfi."""
    import subprocess
    tone = '+'.join(f'sin(2*PI*{200 + 100 * k}*t)*between(t,{k},{k + 1})' for k in range(seconds))
    subprocess.run([intake.ffmpeg_path(), '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f"color=c=black:s=160x120:d={seconds}:r=10,geq=lum='20*floor(T)':cb=128:cr=128",
                    '-f', 'lavfi', '-i', f"aevalsrc='{tone}':s=44100:d={seconds}",
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(path)], check=True)


def _pcm(path, start=0.0):
    import subprocess
    raw = subprocess.run([intake.ffmpeg_path(), '-v', 'error', '-ss', f'{start}', '-i', str(path), '-vn', '-ac', '1',
                          '-ar', '8000', '-f', 's16le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(float)


def _pitch(samples, at):
    """The strongest frequency (Hz) of 0.2 s of 8 kHz audio starting `at` seconds in."""
    part = samples[int(at * 8000):int(at * 8000) + 1600]
    return float(np.argmax(np.abs(np.fft.rfft(part))) * 8000 / len(part))


def test_play_copy_adds_the_sound_for_the_range_without_touching_the_video(tmp_path):
    """Play along: the kept video.* is the picture only (the frames come from it); play.mp4 joins the same video
    stream with the source's sound cut to the same range, so the kept copy is not silent."""
    source = tmp_path / 'source.mp4'
    _video_with_sound(source)
    run = tmp_path / 'run'
    run.mkdir()
    section = intake.video_range(run, '0:03', '0:06', str(source))
    video = intake.get_video(str(source), run, Report(run), section)
    assert not intake.has_sound(video)
    before = video.read_bytes()
    play = intake.make_play_copy(str(source), run, section, note=lambda text: None)
    assert play == run / 'play.mp4' and intake.has_sound(play)
    assert video.read_bytes() == before                                   # video.* untouched
    assert abs(intake._seconds(play) - intake._seconds(video)) < .05
    sound = _pcm(play)
    for second in (0, 1, 2):                                              # the range's seconds 3, 4, 5
        assert abs(_pitch(sound, second + .4) - (200 + 100 * (3 + second))) < 20
    a = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY).mean() for f in intake.extract_frames(video, run, 2, Report(run))]
    assert len(a) == 6
    # a new range removes the copy with sound with the video
    intake.video_range(run, '0:01', '0:02', str(source))
    assert not (run / 'play.mp4').exists() and not list(run.glob('video.*'))


def test_no_play_copy_when_the_source_has_no_sound(tmp_path):
    source = tmp_path / 'source.mp4'
    _counting_video(source)
    run = tmp_path / 'run'
    run.mkdir()
    intake.get_video(str(source), run, Report(run), None)
    assert intake.make_play_copy(str(source), run, None, note=lambda text: None) is None
    assert not (run / 'play.mp4').exists()


def test_a_failed_sound_download_is_tried_twice_more(tmp_path, monkeypatch):
    """YouTube's 403s come and go (Off My Face: failed once, then worked): two more tries before giving up."""
    monkeypatch.setattr(intake, 'RETRY_WAIT', 0)
    tries = []

    def flaky(fail):
        def work():
            tries.append(1)
            if len(tries) <= fail:
                raise RuntimeError('YouTube download failed: HTTP Error 403: Forbidden')
            return 'ok'
        return work
    notes = []
    assert intake._retry(flaky(2), notes.append) == 'ok' and len(tries) == 3 and len(notes) == 2
    tries.clear()
    with pytest.raises(RuntimeError, match='403'):
        intake._retry(flaky(3), notes.append)
    assert len(tries) == 3


BOT = ("ERROR: [youtube] b7gQAwZzfTs: Sign in to confirm you’re not a bot. This helps protect our community. "
       "Use --cookies-from-browser or --cookies for the authentication.")       # yt-dlp's words (2026-11-16)


def test_downloads_use_a_browsers_sign_in_only_when_asked(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, what, timeout):
        commands.append(command)
        (tmp_path / 'video.mp4').write_bytes(b'x')
    monkeypatch.setattr(intake, '_run', fake_run)
    intake._download('https://www.youtube.com/watch?v=abcdefghijk', tmp_path / 'video.%(ext)s')
    (tmp_path / 'video.mp4').unlink()
    intake._download('https://www.youtube.com/watch?v=abcdefghijk', tmp_path / 'video.%(ext)s', cookies='firefox')
    assert '--cookies-from-browser' not in commands[0]
    i = commands[1].index('--cookies-from-browser')
    assert commands[1][i + 1] == 'firefox' and commands[1].index('--') > i       # an option, before the URL
    with pytest.raises(ValueError):
        intake.cookie_args('netscape; rm -rf')


def test_the_bot_check_is_recognised_suggests_the_sign_in_and_is_not_retried(tmp_path, monkeypatch):
    assert intake.is_bot_check(BOT) and intake.is_bot_check("Sign in to confirm you're not a bot")
    assert not intake.is_bot_check('ERROR: HTTP Error 403: Forbidden')

    def blocked(command, what, timeout):
        raise RuntimeError(f'{what} failed: {BOT}')
    monkeypatch.setattr(intake, '_run', blocked)
    with pytest.raises(RuntimeError) as without:
        intake._download('https://www.youtube.com/watch?v=abcdefghijk', tmp_path / 'v.%(ext)s')
    assert '--cookies-from-browser' in str(without.value).splitlines()[-1]               # the hint
    with pytest.raises(RuntimeError) as with_it:
        intake._download('https://www.youtube.com/watch?v=abcdefghijk', tmp_path / 'v.%(ext)s', cookies='chrome')
    assert intake.BOT_HINT not in str(with_it.value)                                   # already on: no hint
    monkeypatch.setattr(intake, 'RETRY_WAIT', 0)
    tries = []

    def work():
        tries.append(1)
        raise RuntimeError(BOT)
    with pytest.raises(RuntimeError):
        intake._retry(work, lambda text: None)
    assert len(tries) == 1                                    # a bot check does not clear in 4 s


def test_the_sound_download_passes_the_sign_in_on(tmp_path, monkeypatch):
    seen = {}

    def fake_download(url, out, section=None, fmt=None, cookies=None):
        seen['cookies'] = cookies
        raise RuntimeError('stop here')
    monkeypatch.setattr(intake, '_download', fake_download)
    monkeypatch.setattr(intake, 'RETRY_WAIT', 0)
    (tmp_path / 'video.mp4').write_bytes(b'x')
    with pytest.raises(RuntimeError):
        intake.make_play_copy('https://www.youtube.com/watch?v=abcdefghijk', tmp_path, note=lambda t: None, cookies='safari')
    assert seen['cookies'] == 'safari'


def test_the_cli_flag_reaches_the_run_and_is_not_saved(tmp_path, monkeypatch):
    from tabvid import __main__ as cli
    assert cli.BROWSERS == intake.BROWSERS
    got = {}
    monkeypatch.setattr(cli, 'run', lambda *a: got.setdefault('args', a))
    cli.main(['https://youtu.be/abcdefghijk', '-o', str(tmp_path / 'r'), '--cookies-from-browser', 'brave'])
    assert got['args'][-1] == 'brave'
    with pytest.raises(SystemExit):
        cli.main(['x.mp4', '-o', str(tmp_path / 'r'), '--cookies-from-browser', 'lynx'])
