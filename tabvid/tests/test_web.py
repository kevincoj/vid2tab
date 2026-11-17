"""Web app server: python -m pytest tabvid/tests/test_web.py -q

A real server on a free port, with sheet folders made by hand (no pipeline run)."""
import http.client
import json
import threading

import pytest

from tabvid.web.server import make_server


def make_pdf(path, pages=2):
    import fitz
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f'page {i + 1}')
    doc.save(path)


def make_sheet(sheets, name='song', title='My Song', artist='Someone', warnings=(), pdf=True):
    folder = sheets / name
    folder.mkdir(parents=True)
    pdf_name = f'{title}.pdf'
    if pdf:
        make_pdf(folder / pdf_name)
    (folder / 'config.json').write_text(json.dumps({'crop': [0, 0, 10, 10], 'title': title, 'artist': artist,
                                                    **({'pdf': pdf_name} if pdf else {})}))
    stages = {'measures': {'measures': 12}, 'sheet': {'pdf': pdf_name, 'pages': 2}} if pdf else {}
    (folder / 'report.json').write_text(json.dumps({'stages': stages, 'warnings': list(warnings)}))
    (folder / 'report.md').write_text('# report\n')
    (folder / 'web.json').write_text(json.dumps({'source': 'https://www.youtube.com/watch?v=abcdefghijk',
                                                 'video_id': 'abcdefghijk', 'created': 1}))
    return folder


@pytest.fixture
def server(tmp_path):
    srv, app = make_server(tmp_path / 'sheets', port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, app
    srv.shutdown()
    srv.server_close()


def request(srv, method, path, body=None, headers=None):
    port = srv.server_address[1]
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    sent = {'Host': f'127.0.0.1:{port}', **({'Content-Type': 'application/json'} if data else {}), **(headers or {})}
    conn.request(method, path, data, sent)
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    kind = response.getheader('Content-Type', '')
    return response.status, (json.loads(payload) if kind.startswith('application/json') else payload), response


def test_the_page_and_the_list_of_sheets(server):
    srv, app = server
    status, page, _ = request(srv, 'GET', '/')
    assert status == 200 and b'<title>Vid2Tab</title>' in page
    assert request(srv, 'GET', '/api/sheets')[1] == []
    make_sheet(app.sheets, warnings=[{'stage': 'sheet', 'message': 'line 4 is stretched 1.70x (over 1.4); '
                                      'no better line break was found.'}])
    make_sheet(app.sheets, 'half', title='Half', pdf=False)
    sheets = {s['name']: s for s in request(srv, 'GET', '/api/sheets')[1]}
    assert sheets['song']['title'] == 'My Song' and sheets['song']['pages'] == 2 and sheets['song']['state'] == 'done'
    assert sheets['song']['measures'] == 12 and len(sheets['song']['warnings']) == 1
    assert sheets['half']['state'] == 'needs-video' and sheets['half']['pdf'] is None


def test_fonts_are_served_for_offline_use(server):
    srv, _ = server
    status, body, response = request(srv, 'GET', '/fonts/fraunces.woff2')
    assert status == 200 and response.getheader('Content-Type') == 'font/woff2' and body[:4] == b'wOF2'
    assert request(srv, 'GET', '/fonts/../server.py')[0] == 404


def test_video_titles_are_cleaned_for_the_details_step():
    from tabvid.web.server import clean_title
    assert clean_title('Andantino in Am - F. Carulli (Slow version) | Classical Guitar TAB') == 'Andantino in Am - F. Carulli'
    assert clean_title('Ed Sheeran - The A Team (Fingerstyle Cover) [TAB]') == 'Ed Sheeran - The A Team'
    assert clean_title('Here, There and Everywhere (The Beatles) - Guitar Lesson + TAB') == 'Here, There and Everywhere (The Beatles)'
    assert clean_title('Hallelujah: Fingerstyle guitar') == 'Hallelujah'
    assert clean_title('Bach: Prelude in C') == 'Bach: Prelude in C'          # nothing about the video: kept
    assert clean_title('Yesterday') == 'Yesterday'


def test_video_info_suggests_title_and_artist_and_finds_an_earlier_sheet(server, monkeypatch):
    srv, app = server
    assert request(srv, 'GET', '/api/video-info?url=nonsense')[:2] == (400, {'field': 'url', 'error': 'That is not a YouTube video link.'})
    monkeypatch.setattr(type(app), '_ytdlp_json', lambda self, url: {
        'title': 'White Christmas - Fingerstyle Guitar Tab', 'channel': 'Shin Izumi', 'duration': 201, 'thumbnail': 'https://i.ytimg.com/x.jpg'})
    info = request(srv, 'GET', '/api/video-info?url=https://youtu.be/abcdefghijk')[1]
    assert info['found'] and info['title'] == 'White Christmas' and info['artist'] == 'Shin Izumi' and info['duration'] == 201
    assert info['existing'] is None
    make_sheet(app.sheets)                       # its video_id is abcdefghijk
    assert request(srv, 'GET', '/api/video-info?url=https://youtu.be/abcdefghijk')[1]['existing']['name'] == 'song'

    def offline(self, url):
        raise RuntimeError('no network')
    monkeypatch.setattr(type(app), '_ytdlp_json', offline)
    info = request(srv, 'GET', '/api/video-info?url=https://youtu.be/abcdefghijk')[1]
    assert info['found'] is False and info['url'] == 'https://www.youtube.com/watch?v=abcdefghijk'


def test_pdf_inline_download_thumbnail_and_report(server):
    srv, app = server
    make_sheet(app.sheets, title='Café Song')
    status, body, response = request(srv, 'GET', '/files/song/sheet.pdf')
    assert status == 200 and body[:4] == b'%PDF' and response.getheader('Content-Disposition') is None
    status, body, response = request(srv, 'GET', '/files/song/sheet.pdf?download=1')
    disposition = response.getheader('Content-Disposition')
    assert status == 200 and 'attachment' in disposition and "UTF-8''Caf%C3%A9%20Song.pdf" in disposition
    status, body, _ = request(srv, 'GET', '/files/song/thumb.png')
    assert status == 200 and body[:8] == b'\x89PNG\r\n\x1a\n' and (app.sheets / 'song' / 'web-thumb.png').exists()
    assert request(srv, 'GET', '/files/song/report.md')[1] == b'# report\n'


def test_only_sheet_folders_are_served(server):
    srv, app = server
    make_sheet(app.sheets)
    (app.sheets.parent / 'secret.txt').write_text('no')
    for path in ('/files/../secret.txt', '/files/..%2Fsecret.txt/sheet.pdf', '/api/sheets/..', '/files/song/config.json',
                 '/api/sheets/.trash', '/files/nope/sheet.pdf'):
        assert request(srv, 'GET', path)[0] == 404, path


def test_requests_from_elsewhere_are_refused(server):
    srv, app = server
    make_sheet(app.sheets)
    assert request(srv, 'GET', '/api/sheets', headers={'Host': 'evil.example:80'})[0] == 403
    assert request(srv, 'POST', '/api/sheets', {}, headers={'Origin': 'http://evil.example'})[0] == 403
    port = srv.server_address[1]
    assert request(srv, 'GET', '/api/sheets', headers={'Host': f'localhost:{port}'})[0] == 200


# ---- jobs, with a stub pipeline ---------------------------------------------------------------

STUB = r'''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args[:3] == ['-u', '-m', 'tabvid'], args
folder = Path(args[args.index('-o') + 1])
(folder / 'argv.json').write_text(json.dumps(args))
pause = float(os.environ.get('STUB_PAUSE', '0'))
if '--until' in args:
    print('Stage 1/5: intake')
    print('Downloading https://www.youtube.com/watch?v=abcdefghijk ...', flush=True)
    time.sleep(pause)
    if os.environ.get('STUB_FAIL'):
        print('RuntimeError: YouTube download failed: ERROR: Video unavailable')
        sys.exit(1)
    (folder / 'video.mp4').write_bytes(b'x')
    if '--start' in args:                    # intake saves the range in config.json
        start = sum(float(v) * 60 ** i for i, v in enumerate(reversed(args[args.index('--start') + 1].split(':'))))
        (folder / 'config.json').write_text(json.dumps({'range': {'start': start, 'end': None}}))
    print('Extracting frames at 2 fps ...')
    import cv2
    sys.path[:0] = [os.getcwd(), os.environ['STUB_TESTS']]   # the server runs us from the repo
    from test_crop import frame
    (folder / 'frames').mkdir(exist_ok=True)
    for i in range(4):
        cv2.imwrite(str(folder / 'frames' / f'{i:06d}.png'), cv2.cvtColor(frame(), cv2.COLOR_RGB2BGR))
    (folder / 'frames' / 'frames.json').write_text('{"fps": 2, "count": 4}')
    sys.exit(0)
config = json.loads((folder / 'config.json').read_text())
assert '--accept-guess' in args and 'crop' in config and 'masks' in config
for n, name in ((2, 'crop'), (3, 'rows'), (4, 'measures'), (5, 'sheet')):
    print(f'Stage {n}/5: {name}', flush=True)
    if n == 3:
        print('  warning [rows]: row-01: only one usable frame (3.0s), so a cursor could not be removed.', flush=True)
        time.sleep(pause)
title = args[args.index('--title') + 1]
import fitz
doc = fitz.open(); doc.new_page(); doc.save(folder / f'{title}.pdf')
config.update(pdf=f'{title}.pdf', title=title)
(folder / 'config.json').write_text(json.dumps(config))
(folder / 'report.json').write_text(json.dumps({'stages': {'measures': {'measures': 3}, 'sheet': {'pages': 1}},
    'warnings': [{'stage': 'rows', 'message': 'row-01: only one usable frame (3.0s), so a cursor could not be removed.'}]}))
print(f'PDF: {folder / (title + ".pdf")}')
'''


@pytest.fixture
def stub_server(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    script = tmp_path / 'stub.py'
    script.write_text(STUB)
    runner = tmp_path / 'python'
    runner.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    runner.chmod(0o755)
    monkeypatch.setenv('STUB_TESTS', str(Path(__file__).parent))
    srv, app = make_server(tmp_path / 'sheets', port=0, python=str(runner))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, app
    app.shutdown()
    srv.shutdown()
    srv.server_close()


def wait(srv, name, until=lambda p: not p['running'], timeout=20):
    import time
    end = time.time() + timeout
    while time.time() < end:
        p = request(srv, 'GET', f'/api/sheets/{name}/progress')[1]
        if until(p):
            return p
        time.sleep(0.1)
    raise AssertionError(f'timed out: {p}')


FORM = {'url': 'https://youtu.be/abcdefghijk', 'title': 'Just the Way You Are', 'artist': 'B. Joel',
        'start': '0:30', 'end': ''}


def test_a_new_sheet_from_link_to_pdf(stub_server):
    srv, app = stub_server
    status, data, _ = request(srv, 'POST', '/api/sheets', FORM)
    assert status == 200 and data['name'] == 'just-the-way-you-are'
    p = wait(srv, data['name'])
    assert p['outcome'] == 'done' and p['kind'] == 'video' and p['state'] == 'needs-crop'
    folder = app.sheets / data['name']
    argv = json.loads((folder / 'argv.json').read_text())
    assert argv[3] == 'https://www.youtube.com/watch?v=abcdefghijk'
    assert argv[argv.index('--start') + 1] == '0:30' and '--end' not in argv and argv[-2:] == ['--until', 'intake']
    assert argv[argv.index('--artist') + 1] == 'B. Joel'
    p = request(srv, 'POST', f'/api/sheets/{data["name"]}/continue', {})[1]
    p = wait(srv, data['name'])
    assert p['outcome'] == 'done' and p['state'] == 'done' and p['warnings_so_far'] == 1
    assert [s['state'] for s in p['steps']] == ['done'] * 6
    config = json.loads((folder / 'config.json').read_text())
    assert config['crop_how'] == 'guess-accepted' and config['masks'] == []
    argv = json.loads((folder / 'argv.json').read_text())
    assert '--start' not in argv and '--accept-guess' in argv and argv[argv.index('--title') + 1] == FORM['title']
    sheet = request(srv, 'GET', f'/api/sheets/{data["name"]}')[1]
    assert sheet['state'] == 'done' and sheet['pdf'] == 'Just the Way You Are.pdf'
    assert 'Stage 3/5: rows' in (folder / 'web.log').read_text()


def test_the_form_is_checked_before_anything_starts(stub_server):
    srv, app = stub_server
    for change, field in (({'url': 'https://vimeo.com/123'}, 'url'), ({'title': ' '}, 'title'),
                          ({'start': '1:75'}, 'start'), ({'start': '2:00', 'end': '1:00'}, 'end')):
        status, data, _ = request(srv, 'POST', '/api/sheets', {**FORM, **change})
        assert status == 400 and data['field'] == field, change
    assert list(app.sheets.iterdir()) == []


def test_same_video_twice_asks_first_and_names_do_not_clash(stub_server):
    srv, app = stub_server
    first = request(srv, 'POST', '/api/sheets', FORM)[1]['name']
    wait(srv, first)
    status, data, _ = request(srv, 'POST', '/api/sheets', FORM)
    assert status == 409 and data['existing']['name'] == first
    status, data, _ = request(srv, 'POST', '/api/sheets', {**FORM, 'again': True})
    assert status == 200 and data['name'] == 'just-the-way-you-are-2'


def test_one_job_at_a_time_and_cancel(stub_server, monkeypatch):
    srv, app = stub_server
    monkeypatch.setenv('STUB_PAUSE', '30')
    name = request(srv, 'POST', '/api/sheets', FORM)[1]['name']
    p = wait(srv, name, lambda p: p['running'] and p['step'] == 'download')
    assert p['steps'][0]['state'] == 'active'
    status, data, _ = request(srv, 'POST', '/api/sheets', {**FORM, 'url': 'https://youtu.be/bbbbbbbbbbb'})
    assert status == 409 and 'Another sheet' in data['error']
    request(srv, 'POST', f'/api/sheets/{name}/cancel', {})
    p = wait(srv, name, timeout=10)
    assert p['outcome'] == 'cancelled' and p['state'] == 'needs-video'
    assert not list((app.sheets / name).glob('video.*'))


def test_a_failed_download_says_so(stub_server, monkeypatch):
    srv, app = stub_server
    monkeypatch.setenv('STUB_FAIL', '1')
    name = request(srv, 'POST', '/api/sheets', FORM)[1]['name']
    p = wait(srv, name)
    assert p['outcome'] == 'failed' and p['state'] == 'needs-video'
    assert any('Video unavailable' in line for line in p['log_tail'])


def test_crop_screen_data_frames_and_saving_the_choice(stub_server):
    srv, app = stub_server
    name = request(srv, 'POST', '/api/sheets', FORM)[1]['name']
    wait(srv, name)
    view = request(srv, 'GET', f'/api/sheets/{name}/frames')[1]
    assert (view['width'], view['height'], view['frames']) == (1280, 720, 4) and not view['saved']
    assert view['crop'] == view['guess'] and view['masks'] == [] and view['mask_frame'] == 2
    assert [m['index'] for m in view['moments']] == [0, 1, 2, 3] and view['moments'][1]['time'] == 30.5   # from 0:30
    status, body, response = request(srv, 'GET', f'/api/sheets/{name}/frame/1.jpg?w=240')
    assert status == 200 and response.getheader('Content-Type') == 'image/jpeg' and body[:2] == b'\xff\xd8'
    import cv2, numpy as np
    assert cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR).shape[1] == 240
    assert request(srv, 'GET', f'/api/sheets/{name}/frame/9.jpg')[0] == 404
    status, data, _ = request(srv, 'POST', f'/api/sheets/{name}/crop', {'crop': [5, 5, 10, 10], 'masks': []})
    assert status == 400 and 'too small' in data['error']
    box = [30, 380, 1300, 710]                          # past the right edge: clipped to the frame
    masks = [[1000, 300, 1100, 420], [1000, 300, 1100, 420], [0, 0, 5, 5]]   # clipped to the crop; repeat and outside dropped
    status, _, _ = request(srv, 'POST', f'/api/sheets/{name}/crop', {'crop': box, 'masks': masks})
    assert status == 200
    wait(srv, name)
    config = json.loads((app.sheets / name / 'config.json').read_text())
    assert config['crop'] == [30, 380, 1280, 710] and config['crop_how'] == 'user-drawn'
    assert config['masks'] == [[1000, 380, 1100, 420]] and config['crop_guess'] == view['guess']
    view = request(srv, 'GET', f'/api/sheets/{name}/frames')[1]
    assert view['saved'] and view['crop'] == config['crop'] and view['masks'] == config['masks']
    # the same crop again keeps crop_how and the decided polarity; only the masks change
    (app.sheets / name / 'config.json').write_text(json.dumps({**config, 'inverted': False}))
    request(srv, 'POST', f'/api/sheets/{name}/crop', {'crop': config['crop'], 'masks': []})
    wait(srv, name)
    config = json.loads((app.sheets / name / 'config.json').read_text())
    assert config['crop_how'] == 'user-drawn' and config['inverted'] is False and config['masks'] == []


def made_sheet(srv, app):
    name = request(srv, 'POST', '/api/sheets', FORM)[1]['name']
    wait(srv, name)
    request(srv, 'POST', f'/api/sheets/{name}/crop', {'crop': [30, 380, 1250, 710], 'masks': [[100, 400, 200, 450]]})
    wait(srv, name)
    return name, app.sheets / name


def test_remake_with_a_new_title_keeps_video_crop_and_masks(stub_server):
    srv, app = stub_server
    name, folder = made_sheet(srv, app)
    status, data, _ = request(srv, 'POST', f'/api/sheets/{name}/remake', {'title': ' ', 'artist': ''})
    assert status == 400 and data['field'] == 'title'
    request(srv, 'POST', f'/api/sheets/{name}/remake', {'title': 'New Name', 'artist': ''})
    p = wait(srv, name)
    assert p['kind'] == 'sheet' and p['state'] == 'done'
    argv = json.loads((folder / 'argv.json').read_text())
    assert argv[argv.index('--title') + 1] == 'New Name' and '--artist' not in argv and '--start' not in argv
    config = json.loads((folder / 'config.json').read_text())
    assert config['crop'] == [30, 380, 1250, 710] and config['masks'] == [[100, 400, 200, 450]]
    assert request(srv, 'GET', f'/api/sheets/{name}')[1]['title'] == 'New Name'


def test_remake_with_a_new_range_fetches_that_part_first(stub_server):
    srv, app = stub_server
    name, folder = made_sheet(srv, app)                      # made from 0:30 to the end
    request(srv, 'POST', f'/api/sheets/{name}/remake', {'start': '0:30', 'end': ''})
    assert wait(srv, name)['kind'] == 'sheet'                # the same range: just made again
    request(srv, 'POST', f'/api/sheets/{name}/remake', {'start': '0:45', 'end': '1:30'})
    p = wait(srv, name)
    argv = json.loads((folder / 'argv.json').read_text())
    assert p['kind'] == 'video' and argv[argv.index('--start') + 1] == '0:45' and argv[argv.index('--end') + 1] == '1:30'
    (folder / 'config.json').write_text(json.dumps({**json.loads((folder / 'config.json').read_text()),
                                                   'range': {'start': 45.0, 'end': 90.0}}))
    request(srv, 'POST', f'/api/sheets/{name}/remake', {'start': '', 'end': ''})     # back to the whole video
    wait(srv, name)
    argv = json.loads((folder / 'argv.json').read_text())
    assert argv[argv.index('--start') + 1] == '0' and '--end' not in argv
    status, data, _ = request(srv, 'POST', f'/api/sheets/{name}/remake', {'start': '2:00', 'end': '1:00'})
    assert status == 400 and data['field'] == 'end'


def test_delete_moves_the_sheet_to_the_trash(stub_server, monkeypatch):
    srv, app = stub_server
    name, folder = made_sheet(srv, app)
    status, data, _ = request(srv, 'POST', f'/api/sheets/{name}/delete', {})
    assert status == 200 and not folder.exists() and (app.sheets / '.trash' / name / 'config.json').exists()
    assert request(srv, 'GET', '/api/sheets')[1] == []
    assert request(srv, 'GET', f'/api/sheets/{name}')[0] == 404
    name2 = request(srv, 'POST', '/api/sheets', FORM)[1]['name']      # same video: no "already made" question
    assert name2 == 'just-the-way-you-are-2'                          # the trashed name stays taken
    monkeypatch.setenv('STUB_PAUSE', '30')
    wait(srv, name2)
    request(srv, 'POST', f'/api/sheets/{name2}/remake', {'title': 'X'})
    wait(srv, name2, lambda p: p['running'])
    status, data, _ = request(srv, 'POST', f'/api/sheets/{name2}/delete', {})
    assert status == 409 and 'Cancel it first' in data['error']


def put(srv, path, data, headers=None):
    port = srv.server_address[1]
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    conn.request('PUT', path, data, {'Host': f'127.0.0.1:{port}', 'Content-Type': 'application/octet-stream',
                                     **(headers or {})})
    response = conn.getresponse()
    body = json.loads(response.read() or b'{}')
    conn.close()
    return response.status, body


def test_a_sheet_from_a_local_video_file(stub_server):
    srv, app = stub_server
    form = {'file': 'My Lesson.MOV', 'title': 'My Lesson', 'artist': '', 'start': '', 'end': ''}
    status, data, _ = request(srv, 'POST', '/api/sheets', {**form, 'file': 'notes.txt'})
    assert status == 400 and data['field'] == 'url' and 'video file' in data['error']
    status, data, _ = request(srv, 'POST', '/api/sheets', form)
    assert status == 200 and data == {'name': 'my-lesson', 'upload': True}
    folder = app.sheets / 'my-lesson'
    assert request(srv, 'GET', '/api/sheets/my-lesson/progress')[1]['running'] is False   # waits for the file
    payload = bytes(range(256)) * 5000                                                  # 1.28 MB, more than one chunk
    status, p = put(srv, '/api/sheets/my-lesson/video', payload)
    assert status == 200 and p['kind'] == 'video'
    source = folder / 'source-video.mov'
    assert source.read_bytes() == payload and not list(folder.glob('*.part'))
    p = wait(srv, 'my-lesson')
    assert p['outcome'] == 'done' and p['steps'][0]['label'] == 'Copying the video file'
    argv = json.loads((folder / 'argv.json').read_text())
    assert argv[3] == str(source) and argv[argv.index('--title') + 1] == 'My Lesson'
    web = json.loads((folder / 'web.json').read_text())
    assert web['file'] == 'My Lesson.MOV' and web['upload'] == 'done' and web['video_id'] == ''
    # a second file of the same name is not asked about (no video ID); the link sheets' check is unchanged
    assert request(srv, 'POST', '/api/sheets', form)[1]['name'] == 'my-lesson-2'


def test_uploads_only_for_file_sheets_and_from_the_page(stub_server):
    srv, app = stub_server
    name = request(srv, 'POST', '/api/sheets', FORM)[1]['name']
    wait(srv, name)
    status, data = put(srv, f'/api/sheets/{name}/video', b'xx')
    assert status == 400 and 'from a link' in data['error']
    request(srv, 'POST', '/api/sheets', {'file': 'a.mp4', 'title': 'A'})
    assert put(srv, '/api/sheets/a/video', b'')[0] == 411
    assert put(srv, '/api/sheets/a/video', b'xx', {'Origin': 'http://evil.example'})[0] == 403
    assert put(srv, '/api/sheets/../video', b'xx')[0] == 404


def test_as_written_is_chosen_on_the_form_and_switched_by_a_remake(stub_server):
    srv, app = stub_server
    name, folder = made_sheet(srv, app)                      # the form's default: as played
    argv = json.loads((folder / 'argv.json').read_text())
    assert '--as-written' not in argv and request(srv, 'GET', f'/api/sheets/{name}')[1]['layout'] == 'as played'
    request(srv, 'POST', f'/api/sheets/{name}/remake', {'layout': 'as written'})
    p = wait(srv, name)
    argv = json.loads((folder / 'argv.json').read_text())
    assert p['kind'] == 'sheet' and '--as-written' in argv and '--start' not in argv
    assert request(srv, 'GET', f'/api/sheets/{name}')[1]['layout'] == 'as written'
    request(srv, 'POST', f'/api/sheets/{name}/remake', {'layout': 'as played'})
    wait(srv, name)
    assert '--as-written' not in json.loads((folder / 'argv.json').read_text())
    status, data, _ = request(srv, 'POST', '/api/sheets', {**FORM, 'title': 'Written', 'layout': 'as written', 'again': True})
    assert status == 200 and json.loads((app.sheets / data['name'] / 'web.json').read_text())['layout'] == 'as written'


def test_warnings_name_their_measures_on_the_printed_sheet():
    from tabvid.web.plain import explain, measures_of
    raw = [{'stage': 'measures', 'message': 'as written: replayed pages printed once: rows 3-4 (1.0-2.0s, m10-m19) = rows 1-2 '
                                            '(the same image); 10 measures dropped.'},
           {'stage': 'measures', 'message': 'measure 25 (row 5) is 124px wide vs a row median of 396px; check the barlines '
                                            'around it. Kept.'},
           {'stage': 'numbers', 'message': '1 measure(s) too many between m4 (number 4) and m7 (number 6): a false barline '
                                           '(strum arrow, stem) split a measure, or a measure was duplicated.'}]
    found = {w['detail'][:10]: measures_of(w, raw) for w in explain(raw)}
    assert found['measure 25'] == [15]                 # numbered before the replay was dropped: shifted back by 10
    assert found['1 measure('] == [4, 5, 6, 7]         # "between measures 4 and 7": the whole stretch
    assert found['as written'] == []


def test_sheet_pages_as_images_with_measure_places(server):
    import fitz
    import numpy as np
    from PIL import Image
    srv, app = server
    folder = make_sheet(app.sheets)
    # a real two-page PDF: page 1 holds two lines (images), page 2 one; 12 measures as 5 + 4 + 3
    line = np.full((60, 400), 255, np.uint8)
    for y in range(10, 60, 10):
        line[y] = 0
    for x in (0, 80, 160, 240, 320, 399):
        line[10:51, x] = 0
    Image.fromarray(line).save(folder / 'line.png')
    with fitz.open() as doc:
        for boxes in ([(30, 60, 580, 140), (30, 150, 580, 230)], [(30, 30, 580, 110)]):
            page = doc.new_page(width=612, height=792)
            for box in boxes:
                page.insert_image(fitz.Rect(*box), filename=str(folder / 'line.png'))
        pdf = next(folder.glob('*.pdf'))
        doc.save(pdf)
    (folder / 'sheet').mkdir(exist_ok=True)
    for i in (1, 2, 3):
        (folder / 'sheet' / f'line-{i:02d}.png').write_bytes((folder / 'line.png').read_bytes())
    report = json.loads((folder / 'report.json').read_text())
    report.setdefault('stages', {})['sheet'] = {'measures_per_line': '5 4 3', 'staff_spacing_px': 10}
    (folder / 'report.json').write_text(json.dumps(report))
    (folder / 'measures.json').write_text(json.dumps([{'measure': n, 'width': 80, 'spacing': 10} for n in range(1, 13)]))
    info = request(srv, 'GET', '/api/sheets/song')[1]['page_layout']
    assert info['measures'] and [len(p['lines']) for p in info['pages']] == [2, 1]
    first = info['pages'][0]['lines'][0]
    assert first['box'][1] == 60 and [m['n'] for m in first['measures']] == [1, 2, 3, 4, 5]
    assert abs(first['measures'][1]['x0'] - 80.5 / 400) < 0.01          # on the drawn barline
    status, png, response = request(srv, 'GET', '/files/song/page/2.png?w=800')
    assert status == 200 and response.getheader('Content-Type') == 'image/png' and png[:4] == b'\x89PNG'
    assert request(srv, 'GET', '/files/song/page/3.png')[0] == 404


def test_half_turn_gaps_between_lines_or_between_systems(server):
    import fitz
    import numpy as np
    from PIL import Image
    srv, app = server
    folder = make_sheet(app.sheets)
    line = np.full((60, 400), 255, np.uint8)
    line[10:51:10] = 0
    Image.fromarray(line).save(folder / 'line.png')
    with fitz.open() as doc:
        # page 1: a Vid2Tab page, three lines (images); page 2: a PDF from elsewhere, two blocks of drawn lines
        page = doc.new_page(width=612, height=792)
        for top in (60, 300, 540):
            page.insert_image(fitz.Rect(30, top, 580, top + 80), filename=str(folder / 'line.png'))
        # page 2: a PDF from elsewhere: two systems, each a 5-line staff and a 6-line TAB staff joined by the
        # system's line at the left, a chord name and fret numbers between the systems
        page = doc.new_page(width=612, height=792)
        for top in (100, 420):
            staff = [top + 7 * i for i in range(5)] + [top + 60 + 8 * i for i in range(6)]
            for y in staff:
                page.draw_line((40, y), (570, y), width=0.8)
            page.draw_line((40, staff[0]), (40, staff[-1]), width=1.2)
            page.insert_text((60, top - 12), 'Am7', fontsize=11)
        page.insert_text((300, 240), '0 2 3', fontsize=9)
        doc.save(next(folder.glob('*.pdf')))
    pages = request(srv, 'GET', '/api/sheets/song')[1]['page_layout']['pages']
    assert pages[0]['gaps'] == [round(y / 792, 4) for y in (220, 460, 624)]   # between lines, below the last
    gaps = [g * 792 for g in pages[1]['gaps']]
    # one split between the systems (not between a staff and its TAB), in the empty rows, then below the last
    assert len(gaps) == 2 and 245 < gaps[0] < 395 and abs(gaps[1] - (420 + 100 + 4 * 7.6)) < 12


def put_pdf(srv, path, data):
    port = srv.server_address[1]
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    conn.request('PUT', path, data, {'Host': f'127.0.0.1:{port}', 'Content-Type': 'application/pdf'})
    response = conn.getresponse()
    body = json.loads(response.read() or b'{}')
    conn.close()
    return response.status, body


def test_open_a_pdf_joins_the_sheets_for_play_mode_only(server, tmp_path):
    srv, app = server
    make_pdf(tmp_path / 'x.pdf', pages=3)
    status, data, _ = request(srv, 'POST', '/api/pdfs', {'name': 'Bach_Prelude  in C.pdf'})
    assert status == 200 and data['name'] == 'bach-prelude-in-c'
    assert request(srv, 'GET', '/api/sheets')[1] == []                      # not listed until the file is in
    status, sheet = put_pdf(srv, '/api/pdfs/bach-prelude-in-c/file', (tmp_path / 'x.pdf').read_bytes())
    assert status == 200 and sheet['title'] == 'Bach Prelude in C' and sheet['source'] == 'pdf'
    assert sheet['state'] == 'done' and sheet['pages'] == 3 and sheet['pdf'] == 'Bach_Prelude  in C.pdf'
    listed = request(srv, 'GET', '/api/sheets')[1]
    assert [s['name'] for s in listed] == ['bach-prelude-in-c']
    layout = request(srv, 'GET', '/api/sheets/bach-prelude-in-c')[1]['page_layout']
    assert len(layout['pages']) == 3 and not layout['measures'] and all('gaps' in p and p['lines'] == [] for p in layout['pages'])
    assert (app.sheets / 'bach-prelude-in-c' / 'web-pages.json').exists()   # gaps kept until the PDF changes
    assert request(srv, 'GET', '/files/bach-prelude-in-c/thumb.png')[0] == 200
    assert request(srv, 'GET', '/files/bach-prelude-in-c/page/3.png?w=400')[0] == 200
    # nothing to make: only play and delete
    assert request(srv, 'POST', '/api/sheets/bach-prelude-in-c/remake', {'title': 'x'})[0] == 400
    assert put_pdf(srv, '/api/pdfs/bach-prelude-in-c/file', b'%PDF')[0] == 409     # already in
    assert request(srv, 'POST', '/api/sheets/bach-prelude-in-c/delete', {})[0] == 200
    # a second PDF of the same name gets its own folder; a file that is not a PDF leaves nothing behind
    assert request(srv, 'POST', '/api/pdfs', {'name': 'notes.txt'})[0] == 400
    name = request(srv, 'POST', '/api/pdfs', {'name': 'Bach_Prelude  in C.pdf'})[1]['name']
    assert name == 'bach-prelude-in-c-2'
    status, data = put_pdf(srv, f'/api/pdfs/{name}/file', b'not a pdf at all')
    assert status == 400 and 'not a PDF' in data['error'] and not (app.sheets / name).exists()


def test_play_along_timing_for_finished_video_sheets_only(server):
    srv, app = server
    folder = make_sheet(app.sheets)
    (folder / 'video.mp4').write_bytes(b'x')
    status, data, _ = request(srv, 'GET', '/api/sheets/song/timing')
    # a finished sheet without rows: no following, but the video can still play (YouTube or the copy)
    assert status == 200 and data['how'] is None and data['measures'] == []
    assert data['video_id'] == 'abcdefghijk' and data['local_video'] == 'video.mp4'
    assert (folder / 'timing.json').exists() and 'stamp' not in data
    make_sheet(app.sheets, name='unfinished', pdf=False)
    assert request(srv, 'GET', '/api/sheets/unfinished/timing')[0] == 409


def test_the_downloaded_video_is_served_in_ranges_for_seeking(server):
    srv, app = server
    folder = make_sheet(app.sheets)
    assert request(srv, 'GET', '/files/song/video')[0] == 404          # no copy kept
    data = bytes(range(256)) * 40                                       # 10240 bytes
    (folder / 'video.mp4').write_bytes(data)
    status, body, response = request(srv, 'GET', '/files/song/video')
    assert status == 200 and body == data and response.getheader('Accept-Ranges') == 'bytes'
    assert response.getheader('Content-Type') == 'video/mp4'
    status, body, response = request(srv, 'GET', '/files/song/video', headers={'Range': 'bytes=100-199'})
    assert status == 206 and body == data[100:200] and response.getheader('Content-Range') == 'bytes 100-199/10240'
    status, body, _ = request(srv, 'GET', '/files/song/video', headers={'Range': 'bytes=10000-'})
    assert status == 206 and body == data[10000:]
    status, body, _ = request(srv, 'GET', '/files/song/video', headers={'Range': 'bytes=-40'})
    assert status == 206 and body == data[-40:]
    assert request(srv, 'GET', '/files/song/video', headers={'Range': 'bytes=20000-'})[0] == 416
    timing = request(srv, 'GET', '/api/sheets/song/timing')[1]
    assert timing['local_video'] == 'video.mp4' and timing['range'] == {'start': 0, 'end': None}


def test_the_kept_copy_with_sound_is_preferred_and_its_sound_fetched_once(server, monkeypatch):
    """Older sheets kept the picture only: the sound is added once (intake.make_play_copy, in the background),
    then play.mp4 is what /files/NAME/video serves."""
    import time
    from tabvid import intake
    srv, app = server
    folder = make_sheet(app.sheets)
    (folder / 'web.json').write_text(json.dumps({'source': 'https://www.youtube.com/watch?v=abcdefghijk', 'title': 'My Song'}))
    (folder / 'video.mp4').write_bytes(b'silent')
    assert request(srv, 'GET', '/api/sheets/song/timing')[1]['local_sound'] == 'none'
    calls = []

    def fake(source, run_dir, section=None, note=print, cookies=None):
        calls.append(source)
        note('Downloading the sound ...')
        (run_dir / 'play.mp4').write_bytes(b'with sound')
        return run_dir / 'play.mp4'
    monkeypatch.setattr(intake, 'make_play_copy', fake)
    assert request(srv, 'POST', '/api/sheets/song/sound', {})[1]['state'] in ('running', 'done')
    for _ in range(50):
        if request(srv, 'GET', '/api/sheets/song/sound')[1]['state'] == 'done':
            break
        time.sleep(.05)
    assert request(srv, 'GET', '/api/sheets/song/sound')[1]['state'] == 'done'
    assert request(srv, 'POST', '/api/sheets/song/sound', {})[1]['state'] == 'done' and len(calls) == 1
    assert request(srv, 'GET', '/files/song/video')[1] == b'with sound'
    assert request(srv, 'GET', '/api/sheets/song/timing')[1]['local_video'] == 'play.mp4'


def test_a_sound_download_that_keeps_failing_says_so_plainly(server, monkeypatch):
    import time
    from tabvid import intake
    srv, app = server
    folder = make_sheet(app.sheets)
    (folder / 'web.json').write_text(json.dumps({'source': 'https://www.youtube.com/watch?v=abcdefghijk', 'title': 'My Song'}))
    (folder / 'video.mp4').write_bytes(b'silent')

    def fails(*a, **k):
        raise RuntimeError('YouTube download failed: ERROR: HTTP Error 403: Forbidden')
    monkeypatch.setattr(intake, 'make_play_copy', fails)
    request(srv, 'POST', '/api/sheets/song/sound', {})
    for _ in range(50):
        state = request(srv, 'GET', '/api/sheets/song/sound')[1]
        if state['state'] != 'running':
            break
        time.sleep(.05)
    assert state['state'] == 'failed' and state['note'].startswith("Couldn't download the sound from YouTube")
    assert '403' in state['detail']
    monkeypatch.setattr(intake, 'make_play_copy', lambda s, run_dir, *a, **k: (run_dir / 'play.mp4').write_bytes(b'x') and run_dir / 'play.mp4')
    request(srv, 'POST', '/api/sheets/song/sound', {})                       # Try again
    for _ in range(50):
        if request(srv, 'GET', '/api/sheets/song/sound')[1]['state'] == 'done':
            break
        time.sleep(.05)
    assert request(srv, 'GET', '/api/sheets/song/sound')[1]['state'] == 'done'


def test_no_sound_to_fetch_when_the_original_file_is_gone(server, tmp_path):
    srv, app = server
    folder = make_sheet(app.sheets)
    (folder / 'web.json').write_text(json.dumps({'source': str(tmp_path / 'gone.mp4'), 'title': 'My Song'}))
    (folder / 'video.mp4').write_bytes(b'silent')
    assert request(srv, 'GET', '/api/sheets/song/sound')[1]['state'] == 'impossible'


def _tab_line(width=900, spacing=24, left=40, clef=True, time=True, start_ink=None):
    """A white line image with a 6-line TAB staff from `left`, an opening barline, and optionally a TAB clef
    (three stacked letters), a bold 4/4 and some ink right after the start (`start_ink`: 'digits' or 'tie')."""
    import cv2
    import numpy as np
    top = 60
    image = np.full((top + 5 * spacing + 60, width), 255, np.uint8)
    for k in range(6):
        image[top + k * spacing] = 0
    image[top:top + 5 * spacing + 1, left:left + 3] = 0
    x = left + 3
    if clef:
        for k, letter in enumerate('TAB'):          # bold letters about 1.3 spacings tall, stacked over the staff
            cv2.putText(image, letter, (x + 8, top + int((1.45 + 1.65 * k) * spacing)), cv2.FONT_HERSHEY_SIMPLEX,
                        spacing / 19, 0, 3)
        x += 8 + spacing
    if time:
        for k in range(2):
            cv2.putText(image, '4', (x + 10, top + int((k + 1) * 2.5 * spacing) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        spacing / 12, 0, max(4, spacing // 4))
        x += 10 + int(1.2 * spacing)
    if start_ink == 'digits':          # a chord of fret numbers right at the start
        for k in range(6):
            cv2.putText(image, '0', (left + 12, top + k * spacing + 8), cv2.FONT_HERSHEY_SIMPLEX, spacing / 34, 0, 2)
    if start_ink == 'tie':             # tied fret numbers in parentheses
        for k in (0, 1):
            cv2.putText(image, '(0)', (left + 10, top + k * spacing + 8), cv2.FONT_HERSHEY_SIMPLEX, spacing / 34, 0, 2)
    for k in range(5):                 # music further on
        cv2.putText(image, '2', (left + 200 + 80 * k, top + (k % 6) * spacing + 8), cv2.FONT_HERSHEY_SIMPLEX, spacing / 34, 0, 2)
    return image


@pytest.mark.parametrize('clef,time', [(True, True), (True, False), (False, False)])
def test_a_lines_first_measure_starts_at_the_staff_after_clef_and_time_signature(tmp_path, clef, time):
    """Play along: the first measure of a line starts at the staff's opening barline (not at the gutter left for
    overhangs), after a TAB clef and a time signature printed there, so the measure ring, playhead and line tint
    do not begin in the margin (A Team m55)."""
    import cv2
    from tabvid.web.server import _line_start
    path = tmp_path / 'line-01.png'
    image = _tab_line(clef=clef, time=time)
    cv2.imwrite(str(path), image)
    start = _line_start(path, 40, 0, 24, True) * image.shape[1]
    ink = (image[60:60 + 5 * 24 + 1] < 140) & ((image[60:60 + 5 * 24 + 1] < 140).mean(axis=1) < 0.5)[:, None]
    glyph_end = max([c for c in range(44, 200) if ink[:, c].any()], default=43)
    if clef or time:
        assert glyph_end - 2 <= start <= glyph_end + 6
    else:
        assert 40 <= start <= 44


@pytest.mark.parametrize('ink', ['digits', 'tie'])
def test_fret_numbers_at_a_lines_start_are_music_not_a_clef(tmp_path, ink):
    import cv2
    from tabvid.web.server import _line_start
    path = tmp_path / 'line-02.png'
    image = _tab_line(clef=False, time=False, start_ink=ink)
    cv2.imwrite(str(path), image)
    assert 40 <= _line_start(path, 40, 0, 24, False) * image.shape[1] <= 44


def test_a_notation_lines_first_measure_starts_after_the_header(tmp_path):
    import cv2
    import numpy as np
    from tabvid.web.server import _line_start
    path = tmp_path / 'line-03.png'
    cv2.imwrite(str(path), np.full((200, 800), 255, np.uint8))
    assert _line_start(path, 10, 70, 24, False) == pytest.approx(80 / 800)


def test_a_clef_printed_on_the_notation_staff_of_every_line_is_skipped(tmp_path):
    """Etude in Am: notation + TAB without a separate notation header, the treble clef printed on every line after
    the system's opening line: the first measure starts after the clef (the TAB letters end earlier)."""
    import cv2
    import numpy as np
    from tabvid.web.server import _line_start
    spacing, nsp, left = 24, 16, 14
    image = np.full((420, 900), 255, np.uint8)
    for k in range(5):
        image[60 + k * nsp] = 0
    for k in range(6):
        image[240 + k * spacing] = 0
    image[50:240 + 5 * spacing + 10, left:left + 3] = 0                       # the system's opening line
    cv2.ellipse(image, (left + 30, 60 + 3 * nsp), (14, 18), 0, 0, 360, 0, 5)  # a clef: bold, taller than 2 spacings
    image[60 - 12:60 + 4 * nsp + 14, left + 34:left + 39] = 0
    for k, letter in enumerate('TAB'):
        cv2.putText(image, letter, (left + 8, 240 + int((1.45 + 1.65 * k) * spacing)), cv2.FONT_HERSHEY_SIMPLEX,
                    spacing / 19, 0, 3)
    cv2.circle(image, (left + 140, 60 + 2 * nsp), 6, 0, -1)                   # the first note and its fret number
    cv2.putText(image, '2', (left + 134, 240 + 2 * spacing + 8), cv2.FONT_HERSHEY_SIMPLEX, spacing / 34, 0, 2)
    path = tmp_path / 'line-06.png'
    cv2.imwrite(str(path), image)
    start = _line_start(path, left, 0, spacing, False) * image.shape[1]
    assert left + 44 <= start <= left + 60


BOT_ERROR = ("YouTube download failed: ERROR: [youtube] b7gQAwZzfTs: Sign in to confirm you’re not a bot. "
             "Use --cookies-from-browser or --cookies for the authentication.")


def test_the_youtube_sign_in_setting_is_off_by_default_kept_and_checked(server):
    srv, app = server
    make_sheet(app.sheets)
    got = request(srv, 'GET', '/api/settings')[1]
    assert got['youtube_cookies'] is None and 'firefox' in got['browsers']
    assert request(srv, 'POST', '/api/settings', {'youtube_cookies': 'firefox'})[1]['youtube_cookies'] == 'firefox'
    assert request(srv, 'GET', '/api/settings')[1]['youtube_cookies'] == 'firefox'
    assert request(srv, 'POST', '/api/settings', {'youtube_cookies': '--exec=evil'})[0] == 400
    assert request(srv, 'GET', '/api/settings')[1]['youtube_cookies'] == 'firefox'       # unchanged
    assert [s['name'] for s in request(srv, 'GET', '/api/sheets')[1]] == ['song']        # settings.json: not a sheet
    assert request(srv, 'POST', '/api/settings', {'youtube_cookies': 'chrome'}, headers={'Origin': 'http://evil.example'})[0] == 403
    assert request(srv, 'POST', '/api/settings', {'youtube_cookies': None})[1]['youtube_cookies'] is None


def test_the_sign_in_reaches_every_download(server, monkeypatch):
    import time
    from tabvid import intake
    srv, app = server
    folder = make_sheet(app.sheets)
    (folder / 'web.json').write_text(json.dumps({'source': 'https://www.youtube.com/watch?v=abcdefghijk', 'title': 'My Song'}))
    (folder / 'video.mp4').write_bytes(b'silent')
    assert '--cookies-from-browser' not in app._command(folder, [])
    request(srv, 'POST', '/api/settings', {'youtube_cookies': 'edge'})
    command = app._command(folder, [])
    assert command[command.index('--cookies-from-browser') + 1] == 'edge'               # making a sheet
    seen = {}
    monkeypatch.setattr(intake, 'make_play_copy', lambda *a, cookies=None, **k: seen.setdefault('cookies', cookies) and None)
    request(srv, 'POST', '/api/sheets/song/sound', {})                                  # the sound for play along
    for _ in range(50):
        if 'cookies' in seen:
            break
        time.sleep(.05)
    assert seen['cookies'] == 'edge'
    runs = []
    monkeypatch.setattr('subprocess.run', lambda command, **k: runs.append(command) or type('R', (), {'returncode': 1, 'stderr': 'x', 'stdout': ''})())
    app.video_info('https://www.youtube.com/watch?v=abcdefghijk')                        # reading the details
    assert runs and runs[0][runs[0].index('--cookies-from-browser') + 1] == 'edge'


def test_a_bot_check_suggests_the_setting(server, monkeypatch):
    import sys
    import time
    from tabvid import intake
    from tabvid.web import server as web
    srv, app = server
    folder = make_sheet(app.sheets)
    (folder / 'web.json').write_text(json.dumps({'source': 'https://www.youtube.com/watch?v=abcdefghijk', 'title': 'My Song'}))
    (folder / 'video.mp4').write_bytes(b'silent')

    def blocked(*a, **k):
        raise RuntimeError(BOT_ERROR)
    monkeypatch.setattr(intake, 'make_play_copy', blocked)
    request(srv, 'POST', '/api/sheets/song/sound', {})
    for _ in range(50):
        state = request(srv, 'GET', '/api/sheets/song/sound')[1]
        if state['state'] != 'running':
            break
        time.sleep(.05)
    assert state['bot_check'] and 'Settings' in state['note'] and 'not a bot' in state['note']
    # reading a video's details
    monkeypatch.setattr(app, '_ytdlp_json', blocked)
    assert app.video_info('https://www.youtube.com/watch?v=abcdefghijk')[0]['bot_check'] is True
    # making a sheet: the job's log says so
    job = web.Job(folder, 'video', [sys.executable, '-c', f'print({BOT_ERROR!r}); raise SystemExit(1)',   # shaped like
                                    '-m', 'https://www.youtube.com/watch?v=abcdefghijk'])          # [py, -u, -m, tabvid, SOURCE]
    for _ in range(100):
        if not job.running:
            break
        time.sleep(.05)
    assert job.progress()['bot_check'] is True
    # with the sign-in already on, the setting is not suggested again
    request(srv, 'POST', '/api/settings', {'youtube_cookies': 'chrome'})
    assert app.video_info('https://www.youtube.com/watch?v=abcdefghijk')[0]['bot_check'] is False
