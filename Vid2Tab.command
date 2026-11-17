#!/bin/bash
# Double-click to start Vid2Tab: the web app opens in your browser. Keep this window open while you use it;
# close it (or press Ctrl+C) to stop Vid2Tab. Your sheets are kept in the "sheets" folder next to this file.
cd "$(dirname "$0")" || exit 1
URL=http://127.0.0.1:8765/

if [ ! -x .venv/bin/python ]; then
    echo "First start: setting Vid2Tab up ..."
    ./install.sh || { echo; echo "Setup failed (see above). Press Return to close."; read -r; exit 1; }
fi

# Already running (another window)? Just open it.
if [ $# = 0 ] && curl -s --max-time 2 "$URL" 2>/dev/null | grep -q 'Vid2Tab'; then
    echo "Vid2Tab is already running: opening $URL"
    open "$URL"
    exit 0
fi

.venv/bin/python -m tabvid.web "$@"
status=$?
if [ $status != 0 ]; then
    echo; echo "Vid2Tab stopped with an error (see above). Press Return to close."; read -r
fi
exit $status
