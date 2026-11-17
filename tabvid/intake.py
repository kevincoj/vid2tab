"""Stage 1: get the video (yt-dlp or a local file) and extract frames with FFmpeg.

Both steps are cached in the run folder: an existing video.* or a frames/
folder made at the same fps is reused on re-runs.

With --start/--end only that part of the video is fetched (yt-dlp section download; if that
fails or gives the wrong length, the whole video is downloaded and cut with FFmpeg). The range
is saved in config.json ("range") and reused on re-runs; a different range replaces the video
and frames.

Play along (web app) needs the video with its sound; video.* is the video stream only (what the frames come
from, never changed). play.mp4 is made beside it: video.*'s stream copied plus the sound, cut to the same range
(make_play_copy). New downloads make it right away; older runs get it the first time the web app needs it.
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np


def youtube_url(value):
    """Normalize a YouTube link to https://www.youtube.com/watch?v=ID (from tabsheet.video_acquisition)."""
    url = urlsplit(value)
    identity = ''
    if url.hostname == 'youtu.be':
        identity = url.path.removeprefix('/')
    elif url.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        if url.path == '/watch':
            identity = parse_qs(url.query).get('v', [''])[0]
        elif url.path.startswith(('/shorts/', '/live/')):
            identity = url.path.split('/')[-1]
    if url.scheme not in ('http', 'https') or not re.fullmatch(r'[A-Za-z0-9_-]{11}', identity):
        raise ValueError(f'Not a YouTube video URL: {value}')
    return f'https://www.youtube.com/watch?v={identity}'


RANGE_SLACK = 1.5   # seconds a section download may differ from the asked length


def parse_time(text):
    """'12:30' -> 750.0; also '1:02:30', '75', '75.5'."""
    parts = str(text).strip().split(':')
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f'Not a time: {text!r} (use 12:30, 1:02:30 or seconds)') from None
    if len(values) > 3 or any(v < 0 for v in values) or any(v >= 60 for v in values[1:]):
        raise ValueError(f'Not a time: {text!r} (use 12:30, 1:02:30 or seconds)')
    seconds = 0.0
    for v in values:
        seconds = seconds * 60 + v
    return seconds


def format_time(seconds):
    """750 -> '12:30'; 3750.5 -> '1:02:30.5'."""
    whole, rest = divmod(round(seconds, 2), 1)
    h, m, sec = int(whole) // 3600, int(whole) // 60 % 60, int(whole) % 60
    text = f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'
    return text + (f'{rest:.2f}'.rstrip('0')[1:] if rest else '')


def video_range(run_dir, start, end, source=None):
    """The part of the video to process: {'start': s, 'end': e or None} (seconds) or None (whole).
    --start/--end given replace the saved range (--start 0 alone = the whole video again); with
    neither, the range saved in config.json is reused. A changed range removes the run's video and
    frames, so they are made again for it (the crop and masks are kept)."""
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    saved = config.get('range')
    if start is None and end is None:
        return saved
    first = parse_time(start) if start is not None else 0.0
    last = parse_time(end) if end is not None else None
    if last is not None and last <= first:
        raise ValueError(f'--end ({end}) must be after --start ({start or 0})')
    wanted = None if first == 0 and last is None else {'start': first, 'end': last}
    if wanted != saved:
        own = [p.resolve() for p in run_dir.glob('video.*')]
        if source and '://' not in source and Path(source).expanduser().resolve() in own:
            raise ValueError('to change the range, give the YouTube URL or the original video file: '
                             "the run folder's video is the one being replaced")
        for old in [*run_dir.glob('video.*'), run_dir / PLAY]:
            old.unlink(missing_ok=True)
        shutil.rmtree(run_dir / 'frames', ignore_errors=True)
        if wanted:
            config['range'] = wanted
        else:
            config.pop('range', None)
        config_path.write_text(json.dumps(config, indent=1) + '\n')
    return wanted


def _range_text(section):
    return f"{format_time(section['start'])}-{format_time(section['end']) if section['end'] is not None else 'end'}"


def _seconds(video):
    capture = cv2.VideoCapture(str(video))
    return capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(capture.get(cv2.CAP_PROP_FPS), 1)


def trim(source, target, section):
    """Cut `section` out of `source` into `target` (an .mp4) with FFmpeg, re-encoded so it starts
    exactly at the asked time (a stream copy starts at the keyframe before it)."""
    command = [ffmpeg_path(), '-nostdin', '-v', 'error', '-y', '-ss', f"{section['start']:.3f}", '-i', str(source)]
    if section['end'] is not None:
        command += ['-t', f"{section['end'] - section['start']:.3f}"]
    command += ['-map', '0:v:0', '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'veryfast',
                '-pix_fmt', 'yuv420p', str(target)]
    _run(command, 'FFmpeg trim', 1800)


def _run(command, label, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'{label} timed out after {timeout}s') from exc
    if result.returncode:
        errors = [line for line in result.stderr.splitlines() if 'ERROR' in line or 'rror' in line]
        raise RuntimeError(f'{label} failed: {(errors or result.stderr.splitlines() or ["no output"])[-1]}')


def ffmpeg_path():
    found = shutil.which('ffmpeg')
    if found:
        return found
    if importlib.util.find_spec('imageio_ffmpeg'):
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    raise RuntimeError('FFmpeg not found. Install it, or: pip install imageio-ffmpeg')


# Opt-in (the web app's Settings, or --cookies-from-browser): yt-dlp reads this browser's YouTube cookies on
# this computer and sends them only to YouTube, which then treats the download as the signed-in user's (gets
# past "Sign in to confirm you're not a bot"). Off unless asked for.
BROWSERS = ('chrome', 'safari', 'firefox', 'edge', 'brave', 'chromium', 'opera', 'vivaldi')
_BOT_CHECK = re.compile(r"confirm you.{0,3}re not a bot", re.I)
BOT_HINT = ("YouTube asked to confirm this is not a bot. Using your browser's YouTube sign-in usually gets past "
            "this: --cookies-from-browser BROWSER (web app: Settings, \"Use my browser's YouTube sign-in for downloads\").")


def is_bot_check(text):
    """yt-dlp's error says YouTube wants a sign-in to prove this is not a bot."""
    return bool(_BOT_CHECK.search(text or ''))


def cookie_args(cookies):
    """yt-dlp arguments for a browser's YouTube sign-in (None: none)."""
    if not cookies:
        return []
    if cookies not in BROWSERS:
        raise ValueError(f'unknown browser {cookies!r} (one of {", ".join(BROWSERS)})')
    return ['--cookies-from-browser', cookies]


def _download(url, out, section=None, fmt='bv[height<=1080]/b[height<=1080]', cookies=None):
    """yt-dlp into out (a name pattern with %(ext)s); a section is cut by yt-dlp with FFmpeg. `cookies`: a
    browser whose YouTube sign-in yt-dlp uses (opt-in)."""
    command = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist', '--no-progress',
               '--no-warnings', '--match-filters', '!is_live', '-f', fmt, *cookie_args(cookies)]
    if section:
        end = f"{section['end']:.3f}" if section['end'] is not None else 'inf'
        command += ['--download-sections', f"*{section['start']:.3f}-{end}", '--force-keyframes-at-cuts',
                    '--ffmpeg-location', ffmpeg_path()]
    try:
        _run(command + ['-o', str(out), '--', url], 'YouTube download', 1800)
    except RuntimeError as error:
        if is_bot_check(str(error)) and not cookies:
            raise RuntimeError(f'{error}\n{BOT_HINT}') from None
        raise
    found = [p for p in out.parent.glob(out.name.replace('%(ext)s', '*')) if p.suffix not in ('.part', '.ytdl')]
    if len(found) != 1:
        raise RuntimeError('YouTube download did not produce exactly one video file')
    return found[0]


def _length_ok(video, section):
    """The clip is as long as asked (open-ended or past the video's end: any length)."""
    if section['end'] is None:
        return _seconds(video) > 0
    return abs(_seconds(video) - (section['end'] - section['start'])) <= RANGE_SLACK


def get_video(source, run_dir, report, section=None, cookies=None):
    """Return the run's video file, downloading or copying it only if not already present. With a
    section (from video_range) only that part is kept: yt-dlp's section download, else the whole
    video cut with FFmpeg."""
    existing = [p for p in run_dir.glob('video.*') if p.suffix not in ('.part', '.ytdl')]
    if section:
        report.stage('intake', range=_range_text(section), range_seconds=[section['start'], section['end']])
    if existing:
        report.stage('intake', video=existing[0].name, video_cached=True)
        return existing[0]
    how = None
    if '://' in source:
        url = youtube_url(source)
        if not importlib.util.find_spec('yt_dlp'):
            raise RuntimeError('yt-dlp is required for YouTube links: pip install yt-dlp')
        if section:
            print(f'Downloading {url} ({_range_text(section)}) ...')
            try:
                video = _download(url, run_dir / 'video.%(ext)s', section, cookies=cookies)
                if _length_ok(video, section):
                    how = 'yt-dlp section download'
                else:
                    print(f'Section download gave {_seconds(video):.1f}s; downloading the whole video to cut it ...')
                    video.unlink()
            except RuntimeError as error:
                print(f'Section download failed ({error}); downloading the whole video to cut it ...')
                for part in run_dir.glob('video.*'):
                    part.unlink()
            if how is None:
                full = _download(url, run_dir / 'whole.%(ext)s', cookies=cookies)
                video = run_dir / 'video.mp4'
                trim(full, video, section)
                full.unlink()
                how = 'whole video downloaded, cut with FFmpeg'
        else:
            print(f'Downloading {url} ...')
            video = _download(url, run_dir / 'video.%(ext)s', cookies=cookies)
        report.stage('intake', source=url, video=video.name, downloaded=True)
    else:
        local = Path(source).expanduser().resolve()
        if not local.is_file():
            raise RuntimeError(f'Video file not found: {local}')
        if section:
            video = run_dir / 'video.mp4'
            trim(local, video, section)
            how = 'cut with FFmpeg'
        else:
            video = run_dir / ('video' + local.suffix.lower())
            shutil.copyfile(local, video)
        report.stage('intake', source=str(local), video=video.name, downloaded=False)
    if section:
        length = _seconds(video)
        report.stage('intake', range_how=how, range_clip_seconds=round(length, 1))
        if section['end'] is not None and length < section['end'] - section['start'] - RANGE_SLACK:
            report.warn('intake', f"the range {_range_text(section)} runs past the end of the video: "
                                  f"the clip is {length:.1f}s, ending at {format_time(section['start'] + length)}.")
        if length <= 0:
            raise RuntimeError(f'Nothing in the range {_range_text(section)}: it starts after the end of the video')
    return video


PLAY = 'play.mp4'
RETRIES = 2            # a YouTube download that fails is tried again this many times (403s come and go)
RETRY_WAIT = 4.0       # seconds between tries


def _retry(work, note=print, what='the download'):
    """work() with RETRIES more tries RETRY_WAIT s apart; the last error is raised."""
    import time
    for attempt in range(RETRIES + 1):
        try:
            return work()
        except RuntimeError as error:
            if attempt == RETRIES or is_bot_check(str(error)):     # a bot check stays: no use trying again
                raise
            note(f'Downloading {what} failed ({error}); trying again ({attempt + 2} of {RETRIES + 1}) ...')
            time.sleep(RETRY_WAIT)


def has_sound(path):
    """True if the file has an audio stream."""
    result = subprocess.run([ffmpeg_path(), '-nostdin', '-hide_banner', '-i', str(path)], capture_output=True, text=True)
    return any('Stream #' in line and 'Audio:' in line for line in result.stderr.splitlines())


def make_play_copy(source, run_dir, section=None, note=print, cookies=None):
    """play.mp4 for play along: the run's video.* stream copied (the same pictures at the same times) with the
    sound of `source` (a YouTube link: its best audio, downloaded whole; a local file: its own audio track)
    from the range's start for the clip's length, as AAC. Returns the path, or None when the source has no
    sound (or the local file is gone). video.* and the frames are not touched. `note(text)`: progress."""
    run_dir = Path(run_dir)
    video = next((p for p in sorted(run_dir.glob('video.*')) if p.suffix not in ('.part', '.ytdl')), None)
    if video is None:
        raise RuntimeError('no video.* in the run folder')
    start = float((section or {}).get('start') or 0)
    length = _seconds(video)
    sound = None
    try:
        if '://' in source:
            note('Downloading the sound ...')

            def fetch():
                for old in run_dir.glob('sound.*'):          # what a failed try left
                    old.unlink()
                return _download(youtube_url(source), run_dir / 'sound.%(ext)s', fmt='ba[ext=m4a]/ba/b', cookies=cookies)
            sound = _retry(fetch, note, "the sound")
        else:
            local = Path(source).expanduser()
            if not local.is_file():
                return None
            sound = local
        if not has_sound(sound):
            return None
        note('Joining the sound to the video ...')
        part = run_dir / 'play.part.mp4'
        _run([ffmpeg_path(), '-nostdin', '-v', 'error', '-y', '-i', str(video),
              '-ss', f'{start:.3f}', '-t', f'{length:.3f}', '-i', str(sound),
              '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k',
              '-movflags', '+faststart', str(part)], 'FFmpeg join sound', 1800)
        target = run_dir / PLAY
        part.replace(target)
        return target
    finally:
        for old in [*run_dir.glob('sound.*'), run_dir / 'play.part.mp4']:
            old.unlink(missing_ok=True)


def extract_frames(video, run_dir, fps, report):
    """Sample frames at `fps` into frames/NNNNNN.png. Frame i is at about i / fps seconds."""
    frames_dir = run_dir / 'frames'
    meta_path = frames_dir / 'frames.json'
    if meta_path.exists() and json.loads(meta_path.read_text()).get('fps') == fps:
        frames = sorted(frames_dir.glob('*.png'))
        report.stage('intake', fps=fps, frames=len(frames), frames_cached=True)
        return frames
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()
    print(f'Extracting frames at {fps} fps ...')
    _run([ffmpeg_path(), '-nostdin', '-v', 'error', '-i', str(video), '-map', '0:v:0', '-an',
          '-vf', f'fps={fps}', '-start_number', '0', str(frames_dir / '%06d.png')], 'FFmpeg frame extraction', 1800)
    frames = sorted(frames_dir.glob('*.png'))
    meta_path.write_text(json.dumps({'fps': fps, 'count': len(frames)}) + '\n')
    capture = cv2.VideoCapture(str(video))
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(capture.get(cv2.CAP_PROP_FPS), 1)
    height, width = cv2.imread(str(frames[0])).shape[:2] if frames else (0, 0)
    report.stage('intake', fps=fps, frames=len(frames), video_seconds=round(duration, 1),
                 frame_size=f'{width}x{height}', frames_cached=False)
    if not frames:
        raise RuntimeError('FFmpeg produced no frames')
    if height < 480:
        report.warn('intake', f'Low resolution video ({width}x{height}); small TAB digits may be unreadable.')
    return frames


def contact_sheet(frames, fps, path, columns=8, thumb_width=240, offset=0):
    """One small thumbnail per frame, labelled with its time in the video (`offset`: the range's
    start), for a quick visual scan."""
    thumbs = []
    for index, frame in enumerate(frames):
        image = cv2.imread(str(frame))
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        label = f'#{index} {offset + index / fps:.1f}s'
        cv2.rectangle(thumb, (0, 0), (thumb_width, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    blank = np.full_like(thumbs[0], 255)
    thumbs += [blank] * (-len(thumbs) % columns)
    rows = [np.hstack(thumbs[i:i + columns]) for i in range(0, len(thumbs), columns)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
