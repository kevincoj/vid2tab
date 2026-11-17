#!/bin/bash
# Update Vid2Tab: pulls the latest code, reinstalls the requirements if they changed, and always updates yt-dlp
# (YouTube changes often; a new yt-dlp is the usual fix when downloads fail). Your sheets are not touched.
set -u

# All in a function, read whole before it runs: git pull may replace this file while it runs.
main() {
    cd "$(dirname "$0")" || exit 1

    fail() { printf '\n%s\n' "$*" >&2; exit 1; }

    [ -d .git ] || fail "This copy of Vid2Tab was not installed with git, so it cannot update itself.
Install it again as the README says (git clone ...), then move your \"sheets\" folder into the new copy."

    echo "Getting the latest Vid2Tab ..."
    before=$(git rev-parse HEAD)
    git pull --ff-only --quiet || fail "Could not update (no internet, or files in this folder were changed by hand).
To throw away changes made here and update anyway: git reset --hard origin/HEAD && ./update.sh
(Your sheets are safe: the sheets folder is not part of the code.)"
    after=$(git rev-parse HEAD)
    if [ "$before" = "$after" ]; then
        echo "The code was already up to date."
    else
        echo "Updated: $(git log --oneline "$before..$after" | wc -l | tr -d ' ') change(s)."
    fi

    # install.sh reinstalls only when the requirements changed (and makes .venv if it is missing).
    DEV=
    [ "$(cut -d' ' -f1 .venv/.vid2tab-installed 2>/dev/null)" = "requirements-dev.txt" ] && DEV=--dev
    ./install.sh $DEV || exit 1

    export PIP_CACHE_DIR="$PWD/.uv/cache/pip"
    echo "Updating yt-dlp and Deno (its JavaScript runtime for YouTube) ..."
    .venv/bin/python -m pip install --quiet --upgrade "yt-dlp[default,deno]" deno \
        || fail "Could not update yt-dlp (no internet?)."
    echo "yt-dlp: $(.venv/bin/python -m yt_dlp --version), Deno: $(.venv/bin/deno --version 2>/dev/null | head -1 | cut -d' ' -f2)"

    if curl -s --max-time 2 http://127.0.0.1:8765/ 2>/dev/null | grep -q 'Vid2Tab'; then
        echo
        echo "Vid2Tab is running with the old version: close its Terminal window and double-click Vid2Tab.command again."
    fi
    echo
    echo "Done."
}

main "$@"
exit
