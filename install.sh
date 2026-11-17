#!/bin/bash
# Vid2Tab setup: gets Python 3.13 (through uv), makes .venv, installs the requirements, checks FFmpeg and yt-dlp.
# Safe to run again: it reuses a working .venv and only installs what changed.
#   ./install.sh          what Vid2Tab needs
#   ./install.sh --dev    also the test tools (pytest)
set -u
cd "$(dirname "$0")" || exit 1

DEV=0
[ "${1:-}" = "--dev" ] && DEV=1
REQS=requirements.txt
[ $DEV = 1 ] && REQS=requirements-dev.txt

say()  { printf '%s\n' "$*"; }
fail() { printf '\n%s\n' "$*" >&2; exit 1; }

# 1. Python 3.13, provided by uv (https://docs.astral.sh/uv/): nothing to install by hand, and the same Python the
# pinned requirements were tested with. uv, its Python and its cache all live in .uv/ inside this folder (deleting
# the folder removes everything; nothing is added to your PATH or shell settings).
UV_VERSION=0.13.8
export UV_PYTHON_INSTALL_DIR="$PWD/.uv/python" UV_CACHE_DIR="$PWD/.uv/cache" UV_PYTHON_PREFERENCE=only-managed \
    PIP_CACHE_DIR="$PWD/.uv/cache/pip"
python_ok() {
    "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' >/dev/null 2>&1
}

if [ -x .venv/bin/python ] && python_ok .venv/bin/python && .venv/bin/python -m pip --version >/dev/null 2>&1; then
    say "Using the existing .venv ($(.venv/bin/python --version))."
else
    if [ ! -x .uv/bin/uv ]; then
        say "Getting uv $UV_VERSION (it provides Python) ..."
        curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
            | UV_UNMANAGED_INSTALL="$PWD/.uv/bin" sh >/dev/null
        [ -x .uv/bin/uv ] || fail "Could not download uv (no internet?). Check the connection and run this again."
    fi
    if [ -e .venv ]; then
        say "The old .venv does not work; making a new one."
        rm -rf .venv
    fi
    say "Getting Python 3.13 and making .venv ..."
    .uv/bin/uv venv --quiet --seed --python 3.13 .venv || fail "Could not make .venv with Python 3.13 (see above)."
    say "Using $(.venv/bin/python --version)."
fi

# 2. Requirements (again only when the requirements file changed since the last install).
STAMP=.venv/.vid2tab-installed
WANT="$REQS $(cat requirements.txt requirements-dev.txt 2>/dev/null | shasum | cut -c1-16)"
if [ "$(cat "$STAMP" 2>/dev/null)" = "$WANT" ]; then
    say "Requirements already installed."
else
    say "Installing the requirements (a few minutes the first time) ..."
    .venv/bin/python -m pip install --quiet --upgrade pip || fail "Could not update pip (no internet?)."
    .venv/bin/python -m pip install --quiet -r "$REQS" || fail "Installing the requirements failed (see above)."
    printf '%s\n' "$WANT" > "$STAMP"
fi

# 3. FFmpeg (a system one, else the copy that comes with imageio-ffmpeg) and yt-dlp.
FFMPEG=$(.venv/bin/python -c 'from tabvid.intake import ffmpeg_path; print(ffmpeg_path())' 2>/dev/null) \
    && "$FFMPEG" -version >/dev/null 2>&1 \
    || fail "FFmpeg does not run. Try: rm -rf .venv && ./install.sh"
say "FFmpeg: ok"
YTDLP=$(.venv/bin/python -m yt_dlp --version 2>/dev/null) || fail "yt-dlp does not run. Try: rm -rf .venv && ./install.sh"
JS=$(.venv/bin/python -m yt_dlp -v 2>&1 | sed -n 's/.*JS runtimes: //p')
case "$JS" in deno*) ;; *) fail "yt-dlp finds no JavaScript runtime (Deno), which YouTube needs. Try: rm -rf .venv && ./install.sh" ;; esac
say "yt-dlp: $YTDLP (JavaScript: $JS)"

chmod +x install.sh update.sh Vid2Tab.command "Update Vid2Tab.command" 2>/dev/null
say ""
say "Vid2Tab is ready. Double-click \"Vid2Tab.command\" to start it."
