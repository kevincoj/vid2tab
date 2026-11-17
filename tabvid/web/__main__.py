"""Start the tabvid web app: python -m tabvid.web [--port 8765] [--sheets sheets] [--no-open]"""
import argparse
import sys
import threading
import webbrowser

from .server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m tabvid.web', description=__doc__)
    parser.add_argument('--port', type=int, default=8765, help='port on 127.0.0.1 (the next free one if taken)')
    parser.add_argument('--sheets', default='sheets', help='folder for the sheets (default: sheets/)')
    parser.add_argument('--no-open', action='store_true', help='do not open the browser')
    args = parser.parse_args(argv)
    server, app = make_server(args.sheets, args.port)
    url = f'http://127.0.0.1:{server.server_address[1]}/'
    print(f'tabvid web app: {url}  (sheets in {app.sheets.resolve()}; Ctrl+C to stop)', flush=True)
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown() if hasattr(app, 'shutdown') else None
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
