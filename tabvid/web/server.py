"""The web app's server: standard library only, bound to 127.0.0.1. See docs/web-app.md.

Sheets live in their own folders under SHEETS (default sheets/), each a normal tabvid run folder plus
web.json (source, created time) and web-thumb.png. The pipeline runs as `python -m tabvid ...` in a
subprocess, one job at a time.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

PAGE = Path(__file__).with_name('index.html')
VIDEO_MIME = {'.mp4': 'video/mp4', '.m4v': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
               '.mkv': 'video/x-matroska'}
FONTS = {p.name: p for p in Path(__file__).with_name('fonts').glob('*.woff2')}   # served at /fonts/NAME (works offline)
ROOT = Path(__file__).resolve().parents[2]      # the repo: `python -m tabvid` runs from here
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}')
FPS = 2                                          # the CLI's default
VIDEO_TYPES = {'.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi'}   # local files the app takes
PDF_LIMIT = 500 << 20                            # an opened PDF may be this big
UPLOAD_NAME = 'source-video'                     # the uploaded file in the sheet folder (intake copies it to video.*)

# Steps shown while a job runs: (key, label). A video job is the first two.
STEPS = [('download', 'Downloading video'), ('frames', 'Reading frames'), ('crop', 'Finding the TAB'),
         ('rows', 'Reading rows'), ('measures', 'Cutting measures'), ('sheet', 'Laying out the sheet')]
# Output lines that move the progress on (the stage lines come from tabvid/__main__.py).
MARKERS = [('Downloading ', 'download'), ('Section download', 'download'), ('Extracting frames', 'frames'),
           ('Stage 2/5', 'crop'), ('Stage 3/5', 'rows'), ('Stage 4/5', 'measures'), ('Stage 5/5', 'sheet')]


def slug(text):
    """'Just the Way You Are' -> 'just-the-way-you-are' (ASCII letters and digits only, at most 60)."""
    import unicodedata
    plain = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '-', plain.lower()).strip('-')[:60].strip('-')


def _bot_check(text):
    from tabvid.intake import is_bot_check
    return is_bot_check(text)


class Job:
    """One `python -m tabvid ...` subprocess and what its output says about its progress.

    kind: 'video' (stage 1 only, --until intake) or 'sheet' (all stages; crop and masks saved)."""

    def __init__(self, folder, kind, command):
        self.folder, self.name, self.kind, self.command = folder, folder.name, kind, command
        self.step = 'download'
        self.started, self.ended = time.time(), None
        self.lines, self.warnings = [], 0
        self.outcome = None       # 'done', 'failed', 'cancelled' once ended
        self.cancelled = False
        self.duration = None      # the video's length in seconds, for the frames percentage
        log = folder / 'web.log'
        with log.open('a') as out:
            out.write(f'\n=== {time.strftime("%Y-%m-%d %H:%M:%S")} {kind}: {" ".join(command[3:])}\n')
        self.proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True,
                                     env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        with (self.folder / 'web.log').open('a') as log:
            for line in self.proc.stdout:
                log.write(line)
                log.flush()
                line = line.rstrip('\n')
                self.lines = (self.lines + [line])[-200:]
                for marker, step in MARKERS:
                    if line.startswith(marker):
                        self.step = step
                if line.lstrip().startswith('warning ['):
                    self.warnings += 1
        code = self.proc.wait()
        self.ended = time.time()
        self.outcome = 'cancelled' if self.cancelled else 'done' if code == 0 else 'failed'
        if self.cancelled and self.step == 'download':   # a half-made clip would be reused: remove it
            for video in self.folder.glob('video.*'):
                video.unlink(missing_ok=True)

    @property
    def running(self):
        return self.outcome is None

    def cancel(self):
        if not self.running:
            return
        self.cancelled = True
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)       # the pipeline and its yt-dlp / FFmpeg children
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.reader.join(5)

    def _detail(self):
        """(text, percent or None) for the current step: MB downloaded, share of frames read."""
        if self.step == 'download':
            if '://' not in self.command[4]:
                return '', None
            size = sum(p.stat().st_size for p in self.folder.glob('video*') if p.is_file())
            return (f'{size / 1e6:.0f} MB so far' if size >= 1e6 else 'Starting…'), None
        if self.step == 'frames':
            if self.duration is None:
                videos = [p for p in self.folder.glob('video.*') if p.suffix not in ('.part', '.ytdl')]
                if videos:
                    import cv2
                    capture = cv2.VideoCapture(str(videos[0]))
                    frames, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
                    capture.release()
                    self.duration = frames / fps if fps and frames else 0
            frames_dir = self.folder / 'frames'
            done = sum(1 for _ in frames_dir.glob('*.png')) if frames_dir.is_dir() else 0
            if self.duration:
                share = min(1.0, done / max(1, self.duration * FPS))
                return f'{round(share * 100)}%', share
            return (f'{done} frames' if done else ''), None
        return '', None

    def progress(self):
        keys = [k for k, _ in STEPS][:2 if self.kind == 'video' else None]
        current = keys.index(self.step) if self.step in keys else 0
        steps = []
        local = '://' not in self.command[4]
        for i, (key, label) in enumerate(STEPS[:len(keys)]):
            if key == 'download' and local:
                label = 'Copying the video file'
            if self.outcome == 'done' or i < current:
                state = 'done'
            elif i == current:
                state = 'active' if self.running else self.outcome
            else:
                state = 'todo'
            steps.append({'key': key, 'label': label, 'state': state})
        detail, share = self._detail() if self.running else ('', None)
        return {'name': self.name, 'kind': self.kind, 'running': self.running, 'outcome': self.outcome,
                'steps': steps, 'step': self.step, 'detail': detail, 'percent': share,
                'elapsed': round((self.ended or time.time()) - self.started), 'warnings_so_far': self.warnings,
                'log_tail': self.lines[-40:],
                # YouTube's "confirm you're not a bot" with no browser sign-in in use: the page suggests Settings
                'bot_check': self.outcome == 'failed' and '--cookies-from-browser' not in self.command
                             and any(_bot_check(line) for line in self.lines[-40:])}


# Words in a YouTube title that describe the video, not the song: "(Slow version)", "| Guitar TAB", ...
_VIDEO_WORDS = r'tab|tabs|tablature|tutorial|lesson|cover|slow|version|fingerstyle|guitar|sheet|score|how to play|' \
               r'easy|play ?along|chords|arrangement|arr\.|free|pdf|backing|official|video|4k|hd'


def clean_title(raw):
    """A video title -> a likely song title: 'Andantino in Am - F. Carulli (Slow version) | Classical Guitar TAB'
    -> 'Andantino in Am - F. Carulli'. Only removes what is clearly about the video; the user edits the rest."""
    title = re.split(r'\s+[|｜]\s+', raw.strip())[0]                       # "Song | Channel / Guitar TAB"
    title = re.sub(r'\s*[(\[【][^)\]】]*(?:' + _VIDEO_WORDS + r')[^)\]】]*[)\]】]', '', title, flags=re.I)
    title = re.sub(r'(?:\s+[-–—]|\s*:)\s+(?:[\w +&/.]*(?:' + _VIDEO_WORDS + r'))+[\w +&/.]*$', '', title, flags=re.I)  # "Song - Guitar Tab"
    title = re.sub(r'\s{2,}', ' ', title).strip(' -–—|:')
    return title or raw.strip()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


class App:
    """State shared by the request handlers: the sheets folder (and, later, the running job)."""

    def __init__(self, sheets, python=sys.executable):
        self.sheets = Path(sheets).resolve()
        self.sheets.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.job = None           # the running (or last) job
        self.lock = threading.Lock()
        self.sounds = {}          # play along: sheet name -> {'state', 'note', 'started'} of fetching its sound

    def shutdown(self):
        if self.job:
            self.job.cancel()

    # ---- jobs ----------------------------------------------------------------------------------

    def busy(self):
        return self.job is not None and self.job.running

    # ---- settings (the app's own, not a sheet's): settings.json in the sheets folder ----------------

    def settings(self):
        """{'youtube_cookies': None | a browser}: downloads use that browser's YouTube sign-in (opt-in)."""
        from tabvid.intake import BROWSERS
        saved = read_json(self.sheets / 'settings.json')
        browser = saved.get('youtube_cookies')
        return {'youtube_cookies': browser if browser in BROWSERS else None, 'browsers': list(BROWSERS)}

    def set_settings(self, data):
        """Change the settings from a POST body: (settings, None) or (None, error)."""
        from tabvid.intake import BROWSERS
        browser = data.get('youtube_cookies') or None
        if browser is not None and browser not in BROWSERS:
            return None, 'Choose one of the browsers listed.'
        saved = read_json(self.sheets / 'settings.json')
        write_json(self.sheets / 'settings.json', {**saved, 'youtube_cookies': browser})
        return self.settings(), None

    def cookies(self):
        return self.settings()['youtube_cookies']

    def _command(self, folder, extra):
        web = read_json(folder / 'web.json')
        command = [self.python, '-u', '-m', 'tabvid', web['source'], '-o', str(folder), '--title', web['title']]
        if web.get('artist'):
            command += ['--artist', web['artist']]
        # as written: a page shown again with the same numbers is printed once (saved in config.json too)
        command += ['--as-written' if web.get('layout') == 'as written' else '--as-played']
        if self.cookies():                           # Settings: the browser's YouTube sign-in (not saved per run)
            command += ['--cookies-from-browser', self.cookies()]
        return command + extra

    def start(self, folder, kind, extra=()):
        """Start a job for `folder`; False if another one is running."""
        with self.lock:
            if self.busy():
                return False
            if kind == 'video':
                extra = [*extra, '--until', 'intake']
            else:
                extra = [*extra, '--accept-guess']   # crop and masks are saved; never open a window
            self.job = Job(folder, kind, self._command(folder, list(extra)))
            return True

    def progress(self, folder):
        job = self.job if self.job and self.job.name == folder.name else None
        if job:
            return {**job.progress(), 'state': self.state(folder)}
        return {'name': folder.name, 'running': False, 'outcome': None, 'steps': [], 'state': self.state(folder)}

    def new_sheet(self, data):
        """Check a new-sheet form and make its folder: (folder, None) or (None, (status, error dict))."""
        from tabvid.intake import parse_time, youtube_url
        file_name = (data.get('file') or '').strip()
        if file_name:
            suffix = Path(file_name).suffix.lower()
            if suffix not in VIDEO_TYPES:
                return None, (HTTPStatus.BAD_REQUEST, {'field': 'url', 'error': 'That is not a video file the app '
                              'can read (' + ', '.join(sorted(VIDEO_TYPES)) + ').'})
            source = None
        else:
            try:
                source = youtube_url((data.get('url') or '').strip())
            except ValueError:
                return None, (HTTPStatus.BAD_REQUEST, {'field': 'url', 'error': 'That is not a YouTube video link.'})
        title, artist = (data.get('title') or '').strip(), (data.get('artist') or '').strip()
        if not title:
            return None, (HTTPStatus.BAD_REQUEST, {'field': 'title', 'error': 'Give the sheet a title.'})
        times = {}
        for field in ('start', 'end'):
            value = (data.get(field) or '').strip()
            if value:
                try:
                    times[field] = parse_time(value)
                except ValueError:
                    return None, (HTTPStatus.BAD_REQUEST, {'field': field, 'error': 'Use a time like 1:30 or 1:02:30.'})
        if 'end' in times and times['end'] <= times.get('start', 0):
            return None, (HTTPStatus.BAD_REQUEST, {'field': 'end', 'error': 'The end must be after the start.'})
        video_id = source.rsplit('=', 1)[1] if source else ''
        if source and not data.get('again'):
            same = [s for s in self.sheets_list() if s['video_id'] == video_id]
            if same:
                return None, (HTTPStatus.CONFLICT, {'error': 'same video', 'existing': same[0]})
        base = slug(title) or video_id
        name, n = base, 2
        while (self.sheets / name).exists() or (self.sheets / '.trash' / name).exists():
            name, n = f'{base}-{n}', n + 1
        folder = self.sheets / name
        folder.mkdir()
        if not source:
            source = str(folder / (UPLOAD_NAME + Path(file_name).suffix.lower()))
        write_json(folder / 'web.json', {'source': source, 'video_id': video_id, 'title': title, 'artist': artist,
                                         'start': data.get('start') or None, 'end': data.get('end') or None,
                                         'layout': _layout(data.get('layout')),
                                         'created': time.time(),
                                         **({'file': Path(file_name).name, 'upload': 'waiting'} if file_name else {})})
        return folder, None

    def _ytdlp_json(self, url):
        """yt-dlp's information about a video, without downloading it (raises on failure)."""
        from tabvid.intake import cookie_args
        done = subprocess.run([self.python, '-m', 'yt_dlp', '--ignore-config', '--no-playlist', '--skip-download',
                               '--dump-single-json', '--no-warnings', *cookie_args(self.cookies()), url],
                              cwd=ROOT, capture_output=True, text=True, timeout=45, stdin=subprocess.DEVNULL)
        if done.returncode:
            raise RuntimeError(((done.stderr or done.stdout).strip().splitlines() or ['yt-dlp failed'])[-1])
        return json.loads(done.stdout)

    def video_info(self, url):
        """For the details step: (info, None) or (None, (status, error dict)). Info has the suggested title
        and artist (the channel), the video's own title, length and thumbnail, and a sheet already made
        from the same video."""
        from tabvid.intake import youtube_url
        try:
            source = youtube_url((url or '').strip())
        except ValueError:
            return None, (HTTPStatus.BAD_REQUEST, {'field': 'url', 'error': 'That is not a YouTube video link.'})
        video_id = source.rsplit('=', 1)[1]
        same = [s for s in self.sheets_list() if s['video_id'] == video_id]
        info = {'url': source, 'video_id': video_id, 'existing': same[0] if same else None}
        try:
            data = self._ytdlp_json(source)
        except Exception as error:             # offline, private video, ...: the user types the details
            from tabvid.intake import is_bot_check
            return {**info, 'found': False, 'bot_check': is_bot_check(str(error)) and not self.cookies()}, None
        raw = data.get('title') or ''
        return {**info, 'found': True, 'video_title': raw, 'title': clean_title(raw),
                'artist': data.get('channel') or data.get('uploader') or '',
                'duration': data.get('duration'), 'thumbnail': data.get('thumbnail')}, None

    def receive(self, folder, stream, length):
        """Write an uploaded video (`length` bytes from `stream`) to the sheet's source file: True when complete."""
        web = read_json(folder / 'web.json')
        if not _receive_to(Path(web['source']), stream, length):
            return False
        write_json(folder / 'web.json', {**web, 'upload': 'done', 'size': length})
        return True

    def new_pdf(self, data):
        """A sheet for a PDF from elsewhere (Open a PDF): its folder, titled from the file name; the file
        follows (receive_pdf). (folder, None) or (None, (status, error dict))."""
        file_name = Path((data.get('name') or '').strip()).name
        if Path(file_name).suffix.lower() != '.pdf':
            return None, (HTTPStatus.BAD_REQUEST, {'error': 'That is not a PDF file.'})
        title = re.sub(r'\s+', ' ', re.sub(r'[_]+', ' ', Path(file_name).stem)).strip() or 'Sheet music'
        base = slug(title) or 'sheet'
        name, n = base, 2
        while (self.sheets / name).exists() or (self.sheets / '.trash' / name).exists():
            name, n = f'{base}-{n}', n + 1
        folder = self.sheets / name
        folder.mkdir(parents=True)
        write_json(folder / 'web.json', {'source': 'pdf', 'video_id': '', 'title': title, 'artist': '',
                                         'file': file_name, 'upload': 'waiting', 'created': time.time()})
        return folder, None

    def receive_pdf(self, folder, stream, length):
        """Write an opened PDF into its sheet folder: None when it is in place and opens with at least one
        page; else an error message, and the folder (only just made) is removed again."""
        web = read_json(folder / 'web.json')
        name = re.sub(r'[^\w .()&,+-]', '_', web['file']).strip() or 'sheet.pdf'
        target = folder / name
        problem = None
        if not _receive_to(target, stream, length):
            problem = 'The file stopped before the end.'
        else:
            try:
                import fitz
                with fitz.open(target) as doc:
                    pages = doc.page_count if doc.is_pdf else 0
            except Exception:
                pages = 0
            if not pages:
                problem = 'That file is not a PDF the app can open.'
        if problem:
            shutil.rmtree(folder, ignore_errors=True)
            return problem
        write_json(folder / 'config.json', {'title': web['title'], 'pdf': name})
        write_json(folder / 'web.json', {**web, 'upload': 'done', 'size': length, 'pages': pages})
        return None

    @staticmethod
    def opened_pdf(folder):
        """True for a sheet that is a PDF from elsewhere (Open a PDF): play mode only, nothing to make."""
        return read_json(folder / 'web.json').get('source') == 'pdf'

    def video_extra(self, folder):
        web = read_json(folder / 'web.json')
        extra = []
        for field in ('start', 'end'):
            if web.get(field):
                extra += [f'--{field}', web[field]]
        return extra

    def frames(self, folder):
        return sorted((folder / 'frames').glob('*.png'))

    def guess(self, folder):
        """crop.initial_guess for the sheet's frames, cached in web-guess.json (made again with new frames)."""
        from tabvid import crop
        cache, meta = folder / 'web-guess.json', folder / 'frames' / 'frames.json'
        if cache.exists() and cache.stat().st_mtime >= meta.stat().st_mtime:
            info = read_json(cache)
            info['boxes'] = {int(k): v for k, v in info['boxes'].items()}
            return info
        info = crop.initial_guess(self.frames(folder))
        info = {**info, 'guess': list(info['guess']), 'boxes': {int(k): list(v) if v else None for k, v in info['boxes'].items()}}
        write_json(cache, {**info, 'boxes': {str(k): v for k, v in info['boxes'].items()}})
        return info

    def crop_view(self, folder):
        """What the crop and mask screens need: frame size, moments to show, the guess, saved choices."""
        import cv2
        frames = self.frames(folder)
        info = self.guess(folder)
        config = read_json(folder / 'config.json')
        height, width = cv2.imread(str(frames[0])).shape[:2]
        fps = read_json(folder / 'frames' / 'frames.json').get('fps', FPS)
        offset = (config.get('range') or {}).get('start', 0)
        middle = len(frames) // 2                 # the CLI's mask frame: overlays that fade in show
        picks = sorted(set(info['boxes']) | {middle})
        return {'width': width, 'height': height, 'frames': len(frames),
                'moments': [{'index': i, 'time': offset + i / fps} for i in picks],
                'guess': info['guess'], 'shown': info['shown'], 'mask_frame': middle,
                'found': info['found'], 'samples': info['samples'], 'flipped': info['flipped'],
                'warnings': info['warnings'],
                'crop': config.get('crop') or info['guess'], 'masks': config.get('masks') or [],
                'saved': 'crop' in config}

    def save_choice(self, folder, data):
        """Save the crop and masks from the browser: None, or an error text."""
        from tabvid import crop
        import cv2
        frames = self.frames(folder)
        height, width = cv2.imread(str(frames[0])).shape[:2]

        def box(value, within):
            try:
                l, t, r, b = (int(round(float(v))) for v in value)
            except (TypeError, ValueError):
                return None
            L, T, R, B = within
            l, t, r, b = max(l, L), max(t, T), min(r, R), min(b, B)
            return [l, t, r, b] if r - l >= 2 and b - t >= 2 else None

        chosen = box(data.get('crop'), (0, 0, width, height))
        if chosen is None or chosen[2] - chosen[0] < 20 or chosen[3] - chosen[1] < 20:
            return 'The music area is too small.'
        masks = []
        for value in data.get('masks') or []:
            mask = box(value, chosen)
            if mask and mask not in masks:
                masks.append(mask)
        config = read_json(folder / 'config.json')
        if config.get('crop') != chosen:
            info = self.guess(folder)
            how = 'guess-accepted' if chosen == list(info['guess']) else 'user-drawn'
            crop.save_crop(folder, chosen, how, info)
        crop.save_masks(folder, masks)
        return None

    def accept_guess(self, folder):
        """Save the crop guess and no masks (until the crop screen exists)."""
        from tabvid import crop
        frames = sorted((folder / 'frames').glob('*.png'))
        info = crop.initial_guess(frames)
        crop.save_crop(folder, info['guess'], 'guess-accepted', info)
        crop.save_masks(folder, [])

    # ---- sheet folders ---------------------------------------------------------------------------

    def folder(self, name):
        """The sheet folder for `name`, or None if it is not a plain name of an existing folder."""
        if not NAME.fullmatch(name or '') or name.startswith('.'):
            return None
        path = self.sheets / name
        return path if path.is_dir() else None

    def pdf(self, folder):
        name = read_json(folder / 'config.json').get('pdf')
        path = folder / name if name else None
        return path if path and path.is_file() else None

    def state(self, folder):
        """running, needs-video (no frames yet), needs-crop (frames, no crop saved), unfinished (crop
        saved, no sheet), done, or failed (an error page or a sheet without measures)."""
        if self.busy() and self.job.name == folder.name:
            return 'running'
        if self.opened_pdf(folder):
            return 'done' if self.pdf(folder) else 'needs-pdf'
        config, report = read_json(folder / 'config.json'), read_json(folder / 'report.json')
        sheet = report.get('stages', {}).get('sheet', {})
        if self.pdf(folder) and sheet:
            return 'failed' if 'failed' in sheet or not report['stages'].get('measures', {}).get('measures') \
                else 'done'
        if not (folder / 'frames' / 'frames.json').exists():
            return 'needs-video'
        if 'crop' not in config or 'masks' not in config:
            return 'needs-crop'
        return 'unfinished'

    def summary(self, folder):
        """What the list and the sheet screen show about one sheet."""
        config, web = read_json(folder / 'config.json'), read_json(folder / 'web.json')
        report = read_json(folder / 'report.json')
        stages = report.get('stages', {})
        pdf = self.pdf(folder)
        from .plain import explain, measures_of
        warnings = explain(report.get('warnings', []), (config.get('range') or {}).get('start', 0))
        for warning in warnings:
            warning['measures'] = measures_of(warning, report.get('warnings', []))
        return {
            'name': folder.name,
            'title': config.get('title') or web.get('title') or folder.name,
            'artist': config.get('artist') or web.get('artist') or '',
            'source': web.get('source', ''),
            'file': web.get('file'),
            'video_id': web.get('video_id', ''),
            'created': web.get('created') or folder.stat().st_mtime,
            'updated': pdf.stat().st_mtime if pdf else folder.stat().st_mtime,
            'range': config.get('range'),
            'layout': _layout(web.get('layout')),
            'state': self.state(folder),
            'pdf': pdf.name if pdf else None,
            'pages': web.get('pages') if web.get('source') == 'pdf' else stages.get('sheet', {}).get('pages'),
            'measures': stages.get('measures', {}).get('measures'),
            'warnings': warnings,
            'counts': {level: sum(w['level'] == level for w in warnings) for level in ('problem', 'check', 'note')},
        }

    def page_image(self, folder, number, width):
        """web-page-N-W.png: page N (1-based) of the PDF, W px wide, made again when the PDF is newer."""
        pdf = self.pdf(folder)
        if not pdf:
            return None
        path = folder / f'web-page-{number}-{width}.png'
        if not path.exists() or path.stat().st_mtime < pdf.stat().st_mtime:
            import fitz
            with fitz.open(pdf) as doc:
                if not 1 <= number <= len(doc):
                    return None
                page = doc[number - 1]
                zoom = width / page.rect.width
                page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).save(path)
        return path

    def layout(self, folder):
        """Where things are on the PDF's pages, for the sheet screen: per page its size and lines (each line
        is one image in the PDF: its box in points), and per line its measures with their left and right
        edges as fractions of the line's width. The edges are estimated from the measures' widths (the
        layout spreads spare room over the line), so they are close, not exact. Per page also `gaps`: where
        the page can be split for a half turn without cutting a line (see _page_gaps). None without a PDF."""
        pdf = self.pdf(folder)
        if not pdf:
            return None
        import fitz
        if self.opened_pdf(folder):
            return {'pages': self._pdf_pages(folder, pdf), 'measures': False}
        pages = []
        with fitz.open(pdf) as doc:
            for page in doc:
                boxes = sorted((tuple(round(v, 1) for v in info['bbox']) for info in page.get_image_info()),
                               key=lambda b: b[1])
                pages.append({'width': page.rect.width, 'height': page.rect.height,
                              'lines': [{'box': list(b), 'measures': []} for b in boxes],
                              'gaps': _page_gaps(page, boxes)})
        sheet = read_json(folder / 'report.json').get('stages', {}).get('sheet', {})
        records = read_json(folder / 'measures.json', [])
        try:
            per_line = [int(n) for n in str(sheet.get('measures_per_line', '')).split()]
        except ValueError:
            per_line = []
        lines = [line for page in pages for line in page['lines']]
        if not per_line or len(per_line) != len(lines) or sum(per_line) != len(records):
            return {'pages': pages, 'measures': False}
        spacing = sheet.get('staff_spacing_px') or 0
        header = sheet.get('notation_header_px') or 0
        gutter = (sheet.get('gutters_px') or [0])[0] or 0
        first = 0
        for index, (line, count) in enumerate(zip(lines, per_line)):
            group = records[first:first + count]
            # estimated from the measures' widths, then moved onto the barlines found in the line image
            widths = [r.get('width', 0) * (spacing / r['spacing'] if spacing and r.get('spacing') else 1) for r in group]
            total = header + sum(widths) or 1
            edges = [header / total]
            for width in widths:
                edges.append(edges[-1] + width / total)
            image = folder / 'sheet' / f'line-{index + 1:02d}.png'
            edges = _snap(edges, _line_barlines(image))
            # the first measure starts at the staff (not the gutter), after the header, clef and time signature
            begin = _line_start(image, gutter, header, spacing, index == 0)
            if begin is not None and begin < edges[1]:
                edges[0] = begin
            for record, x0, x1 in zip(group, edges, edges[1:]):
                line['measures'].append({'n': record['measure'], 'x0': round(x0, 4), 'x1': round(x1, 4),
                                         **({'printed': record['number']} if record.get('number') else {})})
            first += count
        return {'pages': pages, 'measures': True}

    @staticmethod
    def local_video(folder):
        """The kept copy of the sheet's video (the part used, already cut), or None: play.mp4 (with its sound,
        see intake.make_play_copy) when there, else video.* (the video stream the frames came from: silent)."""
        play = folder / 'play.mp4'
        if play.is_file():
            return play
        return next((p for p in sorted(folder.glob('video.*'))
                     if p.is_file() and p.suffix.lower() in VIDEO_MIME), None)

    def sound(self, folder):
        """Getting the kept copy's sound: {'state': 'done' | 'running' | 'failed' | 'none' | 'impossible', 'note',
        'seconds'}. 'impossible': no source to get it from (a local file that is gone)."""
        if (folder / 'play.mp4').is_file():
            return {'state': 'done', 'note': ''}
        job = self.sounds.get(folder.name)
        if job:
            return {**job, 'seconds': round(time.time() - job['started'])}
        source = read_json(folder / 'web.json').get('source') or ''
        if '://' not in source and not Path(source).expanduser().is_file():
            return {'state': 'impossible', 'note': 'The original video file is no longer there, so its sound cannot be added.'}
        return {'state': 'none', 'note': ''}

    def fetch_sound(self, folder):
        """Start making play.mp4 for an older sheet whose kept copy is silent (once, in the background)."""
        with self.lock:
            now = self.sound(folder)
            if now['state'] not in ('none', 'failed'):
                return now
            job = {'state': 'running', 'note': 'Getting the sound ...', 'started': time.time()}
            self.sounds[folder.name] = job

        def work():
            from tabvid import intake
            try:
                source = read_json(folder / 'web.json').get('source') or ''
                section = read_json(folder / 'config.json').get('range')
                made = intake.make_play_copy(source, folder, section, note=lambda text: job.update(note=text),
                                             cookies=self.cookies())
                if made:
                    self.sounds.pop(folder.name, None)
                else:
                    job.update(state='failed', note='The video has no sound to add.')
            except Exception as error:
                youtube = '://' in (read_json(folder / 'web.json').get('source') or '')
                bot = youtube and intake.is_bot_check(str(error)) and not self.cookies()
                job.update(state='failed', detail=str(error), bot_check=bot,
                           note="YouTube asked to confirm you're not a bot, so the sound couldn't be downloaded. "
                                "Turn on \u201cUse my browser's YouTube sign-in for downloads\u201d in Settings, then try again."
                           if bot else "Couldn't download the sound from YouTube. Try again, or update Vid2Tab if it keeps happening."
                           if youtube else "Couldn't add the sound from the original video file.")
        threading.Thread(target=work, daemon=True).start()
        return self.sound(folder)

    def timing(self, folder):
        """Play along: when each measure and line is played (tabvid/timing.py, cached in timing.json until
        an input changes), plus what can play the video: the YouTube id (times + `offset`) and whether the
        downloaded copy `video.*` is there (times as they are). Only for finished Vid2Tab sheets; `how` is
        None when the sheet has no usable timing (the video still plays, without following)."""
        web = read_json(folder / 'web.json')
        local = self.local_video(folder)
        config = read_json(folder / 'config.json')
        out = {'video_id': web.get('video_id'), 'local_video': local.name if local else None,
               'local_sound': self.sound(folder)['state'] if local else None,
               'range': config.get('range') or {'start': 0, 'end': None}}
        try:
            from tabvid import timing
            data = timing.load(folder)
        except Exception as error:          # never break the sheet over timing
            data = {'how': None, 'offset': 0, 'measures': [], 'lines': [], 'facts': {'why': repr(error)}}
        data = {k: v for k, v in data.items() if k != 'stamp'}
        return {**data, **out}

    def _pdf_pages(self, folder, pdf):
        """Pages of an opened PDF: size, no lines, half-turn gaps read from the page images (its own images
        are not lines); kept in web-pages.json until the PDF changes."""
        cache = folder / 'web-pages.json'
        saved = read_json(cache, {})
        if saved.get('mtime') == pdf.stat().st_mtime and saved.get('pages'):
            return saved['pages']
        import fitz
        with fitz.open(pdf) as doc:
            pages = [{'width': page.rect.width, 'height': page.rect.height, 'lines': [], 'gaps': _page_gaps(page, [])}
                     for page in doc]
        write_json(cache, {'mtime': pdf.stat().st_mtime, 'pages': pages})
        return pages

    def sheets_list(self):
        folders = [p for p in self.sheets.iterdir() if p.is_dir() and not p.name.startswith('.')]
        items = [item for item in (self.summary(p) for p in folders) if item['state'] != 'needs-pdf']   # still arriving
        return sorted(items, key=lambda s: s['updated'], reverse=True)

    def thumbnail(self, folder):
        """web-thumb.png: page 1 of the PDF, 360 px wide, made again when the PDF is newer."""
        pdf, thumb = self.pdf(folder), folder / 'web-thumb.png'
        if not pdf:
            return None
        if not thumb.exists() or thumb.stat().st_mtime < pdf.stat().st_mtime:
            import fitz
            with fitz.open(pdf) as doc:
                page = doc[0]
                zoom = 360 / page.rect.width
                page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).save(thumb)
        return thumb


class Handler(BaseHTTPRequestHandler):
    app: App = None          # set by serve()
    port = 0
    server_version = 'tabvid'

    def log_message(self, fmt, *args):   # quiet: the terminal shows only errors
        if args and str(args[1])[:1] in '45':
            super().log_message(fmt, *args)

    # ---- safety --------------------------------------------------------------------------------

    def _local(self):
        """Only requests addressed to this server (no DNS rebinding) and, for POSTs, from its own page."""
        hosts = {f'127.0.0.1:{self.port}', f'localhost:{self.port}'}
        if self.headers.get('Host') not in hosts:
            return False
        origin = self.headers.get('Origin')
        return origin is None or origin in {f'http://{h}' for h in hosts}

    # ---- responses -----------------------------------------------------------------------------

    def _send(self, status, body=b'', kind='application/json', headers=(), cache='no-store'):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', cache)
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, data, status=HTTPStatus.OK):
        self._send(status, json.dumps(data).encode())

    def _error(self, status, message):
        self._json({'error': message}, status)

    def _file(self, path, kind, download_name=None, cache='no-store'):
        headers = []
        if download_name:
            ascii_name = download_name.encode('ascii', 'replace').decode().replace('"', "'")
            headers.append(('Content-Disposition',
                            f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(download_name)}'))
        self._send(HTTPStatus.OK, path.read_bytes(), kind, headers, cache)

    def _ranged(self, path, kind):
        """A file sent whole or, for a `Range: bytes=a-b` request, in part (206), streamed in chunks: the
        video player seeks with ranges."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        found = re.fullmatch(r'bytes=(\d*)-(\d*)', (self.headers.get('Range') or '').strip())
        if found and (found[1] or found[2]):
            if found[1]:
                start = int(found[1])
                end = min(int(found[2]), size - 1) if found[2] else size - 1
            else:                                   # bytes=-N: the last N bytes
                start = max(0, size - int(found[2]))
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'no-store')
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD':
            return
        with path.open('rb') as file:
            file.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = file.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):    # the player moved on to another range
                pass

    def _body(self):
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    # ---- routes --------------------------------------------------------------------------------

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._local():
            return self._error(HTTPStatus.FORBIDDEN, 'not a local request')
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        parts = [p for p in url.path.split('/') if p]
        app = self.app
        if not parts or parts == ['index.html']:
            return self._send(HTTPStatus.OK, PAGE.read_bytes(), 'text/html; charset=utf-8')
        if parts == ['api', 'sheets']:
            return self._json(app.sheets_list())
        if parts == ['api', 'settings']:
            return self._json(app.settings())
        if parts == ['api', 'video-info']:
            info, problem = app.video_info((query.get('url') or [''])[0])
            return self._json(problem[1], problem[0]) if problem else self._json(info)
        if len(parts) == 2 and parts[0] == 'fonts' and parts[1] in FONTS:
            return self._file(FONTS[parts[1]], 'font/woff2', cache='max-age=86400')
        if len(parts) >= 3 and parts[:2] == ['api', 'sheets']:
            folder = app.folder(parts[2])
            if folder is None:
                return self._error(HTTPStatus.NOT_FOUND, 'no such sheet')
            if len(parts) == 3:
                return self._json({**app.summary(folder), 'page_layout': app.layout(folder)})
            return self.get_more(folder, parts[3:], query)
        if len(parts) == 3 and parts[0] == 'files':
            folder = app.folder(parts[1])
            if folder is None:
                return self._error(HTTPStatus.NOT_FOUND, 'no such sheet')
            if parts[2] == 'sheet.pdf':
                pdf = app.pdf(folder)
                if not pdf:
                    return self._error(HTTPStatus.NOT_FOUND, 'no PDF yet')
                return self._file(pdf, 'application/pdf', pdf.name if query.get('download') else None)
            if parts[2] == 'thumb.png':
                thumb = app.thumbnail(folder)
                return self._file(thumb, 'image/png') if thumb else self._error(HTTPStatus.NOT_FOUND, 'no PDF yet')
            if parts[2] == 'video':
                video = app.local_video(folder)
                if not video:
                    return self._error(HTTPStatus.NOT_FOUND, 'no downloaded copy of the video')
                return self._ranged(video, VIDEO_MIME.get(video.suffix.lower(), 'application/octet-stream'))
            if parts[2] == 'report.md' and (folder / 'report.md').is_file():
                return self._file(folder / 'report.md', 'text/plain; charset=utf-8')
        if len(parts) == 4 and parts[0] == 'files' and parts[2] == 'page' and re.fullmatch(r'\d{1,3}\.png', parts[3]):
            folder = app.folder(parts[1])
            if folder is None:
                return self._error(HTTPStatus.NOT_FOUND, 'no such sheet')
            try:
                width = int((query.get('w') or ['1200'])[0])
            except ValueError:
                width = 1200
            width = min(2400, max(400, round(width / 200) * 200))     # a few sizes, so the cache stays small
            image = app.page_image(folder, int(parts[3][:-4]), width)
            return self._file(image, 'image/png') if image else self._error(HTTPStatus.NOT_FOUND, 'no such page')
        return self._error(HTTPStatus.NOT_FOUND, 'not found')

    def get_more(self, folder, rest, query):
        if rest == ['progress']:
            return self._json(self.app.progress(folder))
        if rest == ['timing']:
            if self.app.opened_pdf(folder) or self.app.state(folder) != 'done':
                return self._error(HTTPStatus.CONFLICT, 'Play along needs a finished sheet made from a video.')
            return self._json(self.app.timing(folder))
        if rest == ['sound']:
            return self._json(self.app.sound(folder))
        if rest == ['frames']:
            if not (folder / 'frames' / 'frames.json').exists():
                return self._error(HTTPStatus.CONFLICT, 'The video is not ready yet.')
            return self._json(self.app.crop_view(folder))
        if len(rest) == 2 and rest[0] == 'frame' and re.fullmatch(r'\d{1,6}\.jpg', rest[1]):
            return self._frame(folder, int(rest[1][:-4]), query)
        if rest == ['log']:
            log = folder / 'web.log'
            return self._send(HTTPStatus.OK, log.read_bytes() if log.exists() else b'', 'text/plain; charset=utf-8')
        return self._error(HTTPStatus.NOT_FOUND, 'not found')

    def do_PUT(self):
        """PUT /api/sheets/NAME/video: the video file for a sheet made from a local file; then "get video".
        PUT /api/pdfs/NAME/file: the file of an opened PDF."""
        if not self._local():
            return self._error(HTTPStatus.FORBIDDEN, 'not a local request')
        parts = [p for p in urlsplit(self.path).path.split('/') if p]
        if len(parts) == 4 and parts[:2] == ['api', 'pdfs'] and parts[3] == 'file':
            return self._put_pdf(parts[2])
        folder = self.app.folder(parts[2]) if len(parts) == 4 and parts[:2] == ['api', 'sheets'] and parts[3] == 'video' else None
        if folder is None:
            return self._error(HTTPStatus.NOT_FOUND, 'not found')
        web = read_json(folder / 'web.json')
        if 'file' not in web:
            return self._error(HTTPStatus.BAD_REQUEST, 'This sheet is made from a link, not a file.')
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._error(HTTPStatus.LENGTH_REQUIRED, 'No file was sent.')
        if not self.app.receive(folder, self.rfile, length):
            self.close_connection = True
            return self._error(HTTPStatus.BAD_REQUEST, 'The upload stopped before the end.')
        if not self.app.start(folder, 'video', self.app.video_extra(folder)):
            return self._error(HTTPStatus.CONFLICT, 'Another sheet is being made; continue this one later.')
        return self._json(self.app.progress(folder))

    def _put_pdf(self, name):
        folder = self.app.folder(name)
        if folder is None or not self.app.opened_pdf(folder):
            return self._error(HTTPStatus.NOT_FOUND, 'not found')
        if read_json(folder / 'web.json').get('upload') != 'waiting':
            return self._error(HTTPStatus.CONFLICT, 'This PDF is already in your sheets.')
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = 0
        if not 0 < length <= PDF_LIMIT:
            shutil.rmtree(folder, ignore_errors=True)
            return self._error(HTTPStatus.BAD_REQUEST, 'No file was sent.' if length <= 0 else 'That PDF is too big (over 500 MB).')
        problem = self.app.receive_pdf(folder, self.rfile, length)
        if problem:
            self.close_connection = True
            return self._error(HTTPStatus.BAD_REQUEST, problem)
        return self._json(self.app.summary(folder))

    def _frame(self, folder, index, query):
        """A frame as JPEG; ?w=240 for a thumbnail."""
        import cv2
        frames = self.app.frames(folder)
        if not 0 <= index < len(frames):
            return self._error(HTTPStatus.NOT_FOUND, 'no such frame')
        image = cv2.imread(str(frames[index]))
        width = int((query.get('w') or ['0'])[0] or 0)
        if 0 < width < image.shape[1]:
            image = cv2.resize(image, None, fx=width / image.shape[1], fy=width / image.shape[1], interpolation=cv2.INTER_AREA)
        ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'max-age=60')
        self.end_headers()
        self.wfile.write(data.tobytes())

    def do_POST(self):
        if not self._local():
            return self._error(HTTPStatus.FORBIDDEN, 'not a local request')
        parts = [p for p in urlsplit(self.path).path.split('/') if p]
        data = self._body()
        if data is None:
            return self._error(HTTPStatus.BAD_REQUEST, 'expected a JSON object')
        app = self.app
        if parts == ['api', 'settings']:
            settings, problem = app.set_settings(data)
            return self._error(HTTPStatus.BAD_REQUEST, problem) if problem else self._json(settings)
        if parts == ['api', 'pdfs']:                 # Open a PDF: the folder now, the file next (PUT .../file)
            folder, problem = app.new_pdf(data)
            return self._json(problem[1], problem[0]) if problem else self._json({'name': folder.name})
        if parts == ['api', 'sheets']:
            if app.busy():
                return self._error(HTTPStatus.CONFLICT, 'Another sheet is being made. Wait for it or cancel it.')
            folder, problem = app.new_sheet(data)
            if problem:
                return self._json(problem[1], problem[0])
            if data.get('file'):             # the page sends the file next (PUT .../video), then the job starts
                return self._json({'name': folder.name, 'upload': True})
            app.start(folder, 'video', app.video_extra(folder))
            return self._json({'name': folder.name})
        if len(parts) == 4 and parts[:2] == ['api', 'sheets']:
            folder = app.folder(parts[2])
            if folder is None:
                return self._error(HTTPStatus.NOT_FOUND, 'no such sheet')
            return self.post_action(folder, parts[3], data)
        return self._error(HTTPStatus.NOT_FOUND, 'not found')

    def post_action(self, folder, action, data):
        app = self.app
        if app.opened_pdf(folder) and action != 'delete':
            return self._error(HTTPStatus.BAD_REQUEST, 'An opened PDF can only be played or deleted.')
        if action == 'sound':             # play along: add the sound to the kept copy (not a sheet job)
            if app.state(folder) != 'done' or not app.local_video(folder):
                return self._error(HTTPStatus.CONFLICT, 'Play along needs a finished sheet made from a video.')
            return self._json(app.fetch_sound(folder))
        if action == 'cancel':
            if app.job and app.job.name == folder.name:
                app.job.cancel()
            return self._json(app.progress(folder))
        if action not in self.ACTIONS:
            return self._error(HTTPStatus.NOT_FOUND, 'not found')
        if action == 'delete':
            if app.busy() and app.job.name == folder.name:
                return self._error(HTTPStatus.CONFLICT, 'This sheet is being made. Cancel it first.')
            return self._delete(folder, data)
        if app.busy():
            return self._error(HTTPStatus.CONFLICT, 'Another sheet is being made. Wait for it or cancel it.')
        return self.ACTIONS[action](self, folder, data)

    def _continue(self, folder, data):
        """Carry on from where the sheet stopped: get the video, or make the sheet."""
        app, state = self.app, self.app.state(folder)
        if state == 'needs-video':
            app.start(folder, 'video', app.video_extra(folder))
        elif state == 'needs-crop':
            app.accept_guess(folder)
            app.start(folder, 'sheet')
        else:
            app.start(folder, 'sheet')
        return self._json(app.progress(folder))

    def _crop(self, folder, data):
        """Save the crop and masks chosen in the browser and make the sheet."""
        if not (folder / 'frames' / 'frames.json').exists():
            return self._error(HTTPStatus.CONFLICT, 'The video is not ready yet.')
        problem = self.app.save_choice(folder, data)
        if problem:
            return self._error(HTTPStatus.BAD_REQUEST, problem)
        self.app.start(folder, 'sheet')
        return self._json(self.app.progress(folder))

    def _remake(self, folder, data):
        """Make the sheet again with a changed title, artist, time range or layout (as played / as written). A new range fetches that part
        of the video first (the crop and masks are kept; the page then shows them for a check)."""
        from tabvid.intake import parse_time
        app = self.app
        web = read_json(folder / 'web.json')
        if 'title' in data:
            title = (data.get('title') or '').strip()
            if not title:
                return self._json({'field': 'title', 'error': 'Give the sheet a title.'}, HTTPStatus.BAD_REQUEST)
            web['title'] = title
        if 'artist' in data:
            web['artist'] = (data.get('artist') or '').strip()
        if 'layout' in data:
            web['layout'] = _layout(data.get('layout'))
        extra, kind = [], 'sheet'
        if 'start' in data or 'end' in data:
            times = {}
            for field in ('start', 'end'):
                value = (data.get(field) or '').strip()
                if value:
                    try:
                        times[field] = parse_time(value)
                    except ValueError:
                        return self._json({'field': field, 'error': 'Use a time like 1:30 or 1:02:30.'},
                                          HTTPStatus.BAD_REQUEST)
            if 'end' in times and times['end'] <= times.get('start', 0):
                return self._json({'field': 'end', 'error': 'The end must be after the start.'}, HTTPStatus.BAD_REQUEST)
            old = (read_json(folder / 'config.json').get('range') or {})
            new = {'start': times.get('start', 0.0), 'end': times.get('end')}
            if new != {'start': old.get('start', 0.0), 'end': old.get('end')}:
                kind = 'video'
                extra = ['--start', data.get('start') or '0'] + (['--end', data['end']] if data.get('end') else [])
            web['start'], web['end'] = data.get('start') or None, data.get('end') or None
        write_json(folder / 'web.json', web)
        app.start(folder, kind, extra)
        return self._json(app.progress(folder))

    def _delete(self, folder, data):
        """Move the sheet folder to sheets/.trash/ (a later sheet of the same name gets its own folder)."""
        trash = self.app.sheets / '.trash'
        trash.mkdir(exist_ok=True)
        target = trash / folder.name
        if target.exists():
            target = trash / f'{folder.name}-{time.strftime("%Y%m%d-%H%M%S")}'
        shutil.move(str(folder), str(target))
        return self._json({'trashed': target.name})

    ACTIONS = {'continue': _continue, 'crop': _crop, 'remake': _remake, 'delete': _delete}


def _receive_to(target, stream, length):
    """Write `length` bytes from `stream` to `target` (through a .part file): True when all arrived."""
    part = target.with_name(target.name + '.part')
    left = length
    with part.open('wb') as out:
        while left > 0:
            chunk = stream.read(min(left, 1 << 20))
            if not chunk:
                break
            out.write(chunk)
            left -= len(chunk)
    if left:
        part.unlink(missing_ok=True)
        return False
    part.replace(target)
    return True


def _page_gaps(page, boxes):
    """Where a page can be split for a half turn without cutting through music, as fractions of its height,
    top to bottom; the last one is just below the page's last music. A Vid2Tab page is split halfway
    between two of its lines (each line is one image). Any other page (fewer than two images: a PDF from
    elsewhere) is read from its image: see _system_gaps. [] when nothing is found."""
    height = page.rect.height
    if len(boxes) >= 2:
        gaps = [(a[3] + b[1]) / 2 for a, b in zip(boxes, boxes[1:])]
        return [round(y / height, 4) for y in gaps + [min(height, boxes[-1][3] + 4)]]
    import fitz
    import numpy as np
    zoom = 1000 / height
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    return [round(y / pix.height, 4) for y in _system_gaps(gray)]


def _system_gaps(gray):
    """Split rows of a grayscale page image between music systems: staff lines are rows inked (even lightly:
    thin gray lines of video-made sheets) across most of the width, grouped into staves by their spacing; two neighbouring staves belong to one system when a
    vertical stroke (a barline or the system's bracket line) joins them, otherwise the split goes in the
    middle of the emptiest rows between them. Then one row below the last staff's music. A page without
    staves (text) is split in the middle of each blank band of ~26 pt or more instead."""
    import numpy as np
    dark = gray < 200
    rows, width = dark.shape
    ink = dark.sum(axis=1)
    lines = []
    for r in np.flatnonzero((gray < 215).sum(axis=1) > 0.35 * width):
        (lines[-1].append(r) if lines and r - lines[-1][-1] <= 1 else lines.append([r]))
    centers = [(l[0] + l[-1]) / 2 for l in lines]
    if len(centers) >= 4:
        steps = np.diff(centers)
        space = float(np.median(steps))
        staves = [[centers[0]]]
        for c, d in zip(centers[1:], steps):
            (staves[-1].append(c) if d <= 1.8 * space else staves.append([c]))
        staves = [st for st in staves if len(st) >= 4]                 # 5-line notation, 4-6-line TAB
    else:
        staves = []
    if not staves:
        filled = np.flatnonzero(ink > max(1, width // 400))
        if not len(filled):
            return []
        gaps = [(a + b) / 2 for a, b in zip(filled, filled[1:]) if b - a > rows // 30]
        return gaps + [min(rows, filled[-1] + 3)]
    gaps = []
    for above, below in zip(staves, staves[1:]):
        top, bottom = int(above[-1]) + 2, int(below[0]) - 1
        if bottom - top < 3:
            continue
        joined = ((gray[top:bottom] < 215).mean(axis=0) >= 0.95).any()          # a stroke from one staff to the next
        if joined:
            continue
        band = ink[top:bottom]
        low = np.flatnonzero(band <= band.min() + max(1, width // 400))
        runs = [[low[0]]]
        for r in low[1:]:
            (runs[-1].append(r) if r - runs[-1][-1] <= 1 else runs.append([r]))
        run = max(runs, key=len)
        gaps.append(top + (run[0] + run[-1]) / 2)
    gaps.append(min(rows, staves[-1][-1] + 4 * space))                 # numbers and marks below the last staff
    return gaps


def _line_barlines(path):
    """Barline positions of one laid-out line (sheet/line-NN.png) as fractions of its width: columns dark
    from the top to the bottom line of the lowest staff (the TAB); a run of such columns, or a double or
    final barline, counts once. [] when the image is missing."""
    import cv2
    import numpy as np
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []
    dark = gray < 140
    width = dark.shape[1]
    rows = np.flatnonzero(dark.mean(axis=1) > 0.5)                 # staff line rows
    if len(rows) < 2:
        return []
    lines = [[rows[0]]]
    for r in rows[1:]:
        (lines[-1].append(r) if r - lines[-1][-1] <= 2 else lines.append([r]))
    tab = lines[-6:]
    top, bottom = tab[0][0], tab[-1][-1]
    if bottom - top < 10:
        return []
    bars = []
    for c in np.flatnonzero(dark[top:bottom + 1].mean(axis=0) >= 0.97):
        if bars and c - bars[-1][-1] <= max(4, width // 150):
            bars[-1].append(c)
        else:
            bars.append([c])
    return [(b[0] + b[-1]) / 2 / width for b in bars]


def _staff_rows(gray, spacing):
    """Line rows of the staves in a laid-out line image: ([rows of each TAB line] x 6, [rows of each notation
    line] x 5 or []). Staff lines are found as a comb: the 6 rows `spacing` apart (+-2 px each) whose weakest
    line is darkest across the image (digits, chords and light gray lines make single rows unreliable); a
    notation staff is a 5-line comb above it at its own spacing (0.4-1 TAB spacing) at least half as strong.
    ([], []) when no TAB is found."""
    import numpy as np
    share = (gray < 200).mean(axis=1)
    height = len(share)

    def comb(top, gap, count):
        """(weakest line's share, [row of each line]) for a comb from `top`."""
        rows = []
        for k in range(count):
            y = int(round(top + k * gap))
            window = share[max(0, y - 2):y + 3]
            if not len(window):
                return 0.0, []
            rows.append(max(0, y - 2) + int(window.argmax()))
        return float(min(share[r] for r in rows)), rows

    def lines_at(rows):
        """Each line's rows: the found row and its neighbours at least half as dark (thick lines)."""
        out = []
        for r in rows:
            line = [r]
            while line[0] > 0 and share[line[0] - 1] >= 0.5 * share[r]:
                line.insert(0, line[0] - 1)
            while line[-1] + 1 < height and share[line[-1] + 1] >= 0.5 * share[r]:
                line.append(line[-1] + 1)
            out.append(line[:4] if len(line) > 4 else line)
        return out

    if not spacing or height < 5 * spacing:
        return [], []
    best = max((comb(top, spacing, 6) + (top,) for top in range(0, int(height - 5 * spacing))), key=lambda c: (c[0], c[2]))
    if best[0] < 0.25:
        return [], []
    tab = lines_at(best[1])
    found = []
    for gap10 in range(int(4 * spacing), int(10 * spacing) + 1, 5):
        gap = gap10 / 10
        for top in range(0, int(best[2] - 4 * gap - 0.4 * spacing)):
            score, rows = comb(top, gap, 5)
            if rows and score >= 0.5 * best[0]:
                found.append((score, rows))
    notation = lines_at(max(found)[1]) if found else []
    return tab, notation


def _line_start(path, start_px, header_px, spacing, first_line):
    """Where the first measure of a laid-out line begins, as a fraction of the image width: at the staff's
    start (the opening barline; `start_px` is the left gutter, the same on every line), after the notation
    header (brace, clefs, key: `header_px`), and on the first line also after a TAB clef and a time signature
    printed there (the clef is kept only at the start of the piece; later lines start on blank staff).

    TAB glyphs: runs 0.9-1.6 spacings wide. The TAB clef: the first, within 1 spacing of the staff start,
    over 70% of the staff height (stacked letters, each 1 spacing or taller). A time signature: a vertical
    stroke of 1.2 spacings or more (staff lines bridged; fret digits are under 1.1), bold (thickest stroke 0.35 spacing or more: 0.43-0.52
    measured; fret digits, ties and parentheses 0.17-0.28), within 2.6 spacings of what came before.
    Notation clef, time or key signature: a glyph run in the notation staff at least 2 of its spacings tall with no
    TAB ink under it (every note has a fret number below it). None when the image is missing."""
    import cv2
    import numpy as np
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    width = gray.shape[1]
    dark = gray < 140
    x = start_px + header_px
    if not spacing:
        return x / width
    tab, notation = _staff_rows(gray, spacing)
    if not tab:
        return x / width

    def band(lines):
        top, bottom = lines[0][0], lines[-1][-1]
        ink = dark[top:bottom + 1].copy()
        removed = np.zeros(len(ink), bool)
        for line in lines:
            removed[max(0, line[0] - top - 2):line[-1] - top + 3] = True
        return ink, removed

    def stroke(has, removed):
        """Longest run of rows with ink; a removed staff line counts when ink touches it on both sides."""
        filled, n, i = has & ~removed, len(has), 0
        seen = filled.copy()
        while i < n:
            if removed[i]:
                j = i
                while j < n and removed[j]:
                    j += 1
                if (i == 0 or seen[i - 1]) and (j == n or seen[j]):
                    filled[i:j] = True
                i = j
            else:
                i += 1
        best = run = 0
        for v in filled:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    def glyphs(ink, removed, x0, x1, gap):
        cols = np.flatnonzero((ink & ~removed[:, None])[:, x0:x1].any(axis=0)) + x0
        runs = []
        for c in cols:
            if runs and c - runs[-1][1] <= gap:
                runs[-1][1] = c
            else:
                runs.append([c, c])
        return runs

    ink, removed = band(tab)
    sp = (tab[-1][0] - tab[0][0]) / 5
    # the opening barline (dark over the whole staff, at its start) is split off before glyphs are grouped
    full = (ink | removed[:, None]).mean(axis=0) >= 0.9
    bar = start_px
    while bar < len(full) and bar <= start_px + 0.3 * sp + 3 and (full[bar] or bar < start_px + 1):
        ink[:, bar] = False
        bar += 1
    tab_x = max(x, bar if full[start_px:bar].any() else start_px)
    clef = False
    for a, b in glyphs(ink, removed, start_px, x + int(7 * sp), max(2, 0.2 * sp)):
        if b < tab_x:                                   # inside the header
            continue
        part = ink[:, a:b + 1]
        has = part.any(axis=1)
        w = (b - a + 1) / sp
        if w < 0.15 and a <= tab_x + 1:                 # the barline's antialiased edge
            tab_x = b + 1
            continue
        if not 0.9 <= w <= 1.6:
            break
        tall = stroke(has, removed) / sp
        thick = 2 * cv2.distanceTransform((part & ~removed[:, None]).astype(np.uint8), cv2.DIST_L2, 3).max() / sp
        if not clef and a - tab_x <= 1.0 * sp and has[~removed].mean() >= 0.7 and tall >= 1.0:     # T A B, stacked
            clef = True
        elif thick >= 0.35 and tall >= 1.2 and a - tab_x <= 2.6 * sp:                             # time signature
            pass
        else:
            break
        tab_x = b + 1
    note_x = x
    if notation:
        n_ink, n_removed = band(notation)
        nsp = (notation[-1][0] - notation[0][0]) / 4
        tab_ink = ink & ~removed[:, None]
        tab_ink[:, :tab_x] = False                      # the TAB clef under a notation clef is not a note
        pad = int(0.3 * sp)
        for a, b in glyphs(n_ink, n_removed, start_px, x + int(5 * sp), max(2, 0.2 * nsp)):
            if b < note_x:
                continue
            if b - a + 1 <= 0.25 * nsp and a <= note_x + 0.3 * sp:     # the system's opening line
                note_x = b + 1
                continue
            under = tab_ink[:, max(0, a - pad):b + pad + 1].any()
            tall = stroke(n_ink[:, a:b + 1].any(axis=1), n_removed) >= 2.0 * nsp
            if not under and tall and b - a + 1 >= 0.7 * nsp:
                note_x = b + 1
                continue
            break
    return max(tab_x, note_x) / width


def _snap(edges, bars, reach=0.03):
    """Move each estimated measure edge to the nearest barline found within `reach` of the line width
    (the first edge, after the clef, stays an estimate)."""
    out = list(edges)
    for i in range(1, len(out)):
        near = min(bars, key=lambda b: abs(b - out[i]), default=None)
        if near is not None and abs(near - out[i]) <= reach and near > out[i - 1]:
            out[i] = near
    return out


def _layout(value):
    """'as written' (replayed pages printed once) or the default 'as played'."""
    return 'as written' if value == 'as written' else 'as played'


def make_server(sheets, port=8765, python=sys.executable):
    """A server on 127.0.0.1 at `port` or the next free one (up to 20 tries); port 0 picks any."""
    app = App(sheets, python)
    for candidate in ([0] if port == 0 else range(port, port + 20)):
        handler = type('BoundHandler', (Handler,), {'app': app})
        try:
            server = ThreadingHTTPServer(('127.0.0.1', candidate), handler)
        except OSError:
            continue
        handler.port = server.server_address[1]
        server.daemon_threads = True
        return server, app
    raise OSError(f'no free port from {port} to {port + 19}')
