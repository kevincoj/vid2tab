#!/bin/bash
# Double-click to update Vid2Tab (the code, its requirements and yt-dlp). Your sheets are not touched.
cd "$(dirname "$0")" || exit 1
./update.sh
echo
echo "Press Return to close this window."
read -r
