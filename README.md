# Vid2Tab

Vid2Tab turns a guitar TAB video (a YouTube link or a video file on your Mac) into a clean PDF sheet you can
print or play from. It reads the TAB the video already shows: no transcribing by ear, no guessing of notes.
The sheet is the video's own TAB, cleaned up, with each page or scroll shown once and laid out for paper and
screen.

Then you can play from it on screen: full screen pages that turn with a key or a Bluetooth page-turner pedal,
and **play along**, the video beside the sheet with the line being played tinted, pages turning by
themselves, click a measure to jump there, and drag across measures to loop them.

Everything runs on your Mac. You use it in your web browser, but nothing is uploaded anywhere.

**Use it for personal practice.** The TAB belongs to the people who made the video: credit them, and support
them (subscribe, buy their TAB books or Patreon sheets) if you learn from their work. You are responsible for
following [YouTube's Terms of Service](https://www.youtube.com/t/terms) when you download a video.

## What you need

- A Mac (Apple silicon or Intel) with macOS 12 or newer.
- An internet connection for the install. You do not need to install Python: the installer fetches its own
  copy (Python 3.13, through [uv](https://docs.astral.sh/uv/)) and keeps it inside the Vid2Tab folder.
- About 700 MB of disk space, plus room for the videos of your sheets.

## Install (once)

1. Open **Terminal** (press Cmd+Space, type "Terminal", press Return).
2. Copy this line, paste it into Terminal and press Return:

   ```sh
   git clone https://github.com/kevincoj/vid2tab.git ~/Vid2Tab && ~/Vid2Tab/install.sh
   ```

   If your Mac asks to install the "command line developer tools", click **Install**, wait for it to finish,
   then paste the line again.
3. Wait a few minutes until it says **Vid2Tab is ready**. You can close Terminal.

Vid2Tab is now in the **Vid2Tab** folder in your home folder (in Finder: Go > Home). Vid2Tab keeps everything
it installs (its Python included) in that folder. Running `install.sh` again is safe: it only fixes or
adds what is missing.

## Start

Double-click **Vid2Tab.command** in the Vid2Tab folder. A Terminal window opens and Vid2Tab opens in your
browser (at http://127.0.0.1:8765). **Keep the Terminal window open** while you use Vid2Tab; close it to stop
Vid2Tab. Double-clicking it again while it runs just opens the browser page.

Tip: drag Vid2Tab.command to the Dock (right side, next to the Trash) to start it with one click.

The first time, macOS may say it cannot check the file for malicious software. Then right-click (or
Control-click) Vid2Tab.command, choose **Open**, then **Open** again; macOS remembers it after that.

## Update

Double-click **Update Vid2Tab.command**. It gets the latest Vid2Tab, installs anything new it needs, and
always updates yt-dlp, the part that downloads from YouTube (YouTube changes often, so an update is the first
thing to try when downloads stop working). Your sheets are not touched. If Vid2Tab was running, close its
Terminal window and start it again afterwards.

## Make a sheet

1. **Paste a YouTube link** on the home screen (or choose a video file on your Mac).
2. **Check the details.** Vid2Tab fills in the title and artist from the video; change them if you like.
   The title is printed on page 1 and names the PDF. If the TAB is only in part of the video (say after a
   spoken intro), give a start and end time, so only that part is downloaded.
   **Repeats**: *As played* (the default) prints every page the video shows, in order. *As written* prints a page
   that comes back the same (a repeat, D.C., D.S.) only once, when Vid2Tab can tell for sure.
3. **Choose the music area.** Vid2Tab shows a frame of the video with its guess in green. If the box covers
   the TAB staff (with the chord names and numbers above it), go on; otherwise drag a new box around it.
   Other moments of the video are shown below, so you can check the box fits them all.
4. **Watermarks.** Drag a box over each logo or channel name that sits on the music. It is painted out in every
   frame; music under a see-through logo is recovered. No watermarks? Just go on.
5. **Make the sheet.** This takes from under a minute to a few minutes, depending on the video's length.

The sheet screen shows the pages, a **Download PDF** button and anything worth checking (below). You can
change the title, the music area, the part of the video or the repeats later and make it again; the video is
kept, so this is quick. Sheets are kept in the **sheets** folder inside the Vid2Tab folder.

Sheet music already as a PDF? **Open a PDF** on the home screen to play from it in play mode.

## What the notes on a sheet mean

Vid2Tab always makes a PDF, even when it is unsure about something. Instead of stopping, it lists what is worth
a look, in plain words, with the measure numbers:

- **Things to check** (amber): a spot that may be wrong. For example, a measure much wider or narrower than
  the others (two measures run together, or a barline missed), measure numbers that skip or go back (a page
  of the video may be missing or doubled), or a part of the video that was hard to read. Compare that spot
  with the video; most are fine.
- **Notes** (folded away): small things Vid2Tab did on purpose, such as a page the video showed twice that
  was printed once.
- **Problem** (red): the sheet could not be made, for example because no TAB staff was found in the music
  area. Choose the music area again, or check the start and end time.

**Full report** shows the technical details (useful when you report a problem).

## Play mode

**▶ Play** on a sheet opens it for playing: the pages fit the screen height (two side by side in a wide
window).

- **Turn pages:** Space, the arrow keys, Page Up / Page Down, or a Bluetooth page-turner pedal (they send
  these keys). Clicking the left or right edge of the page also turns.
- **F**: full screen. **Esc**: leave full screen, then back to the sheet.
- **Half turns** (toolbar): the first press brings in the top of the next page while you finish the bottom of
  this one.
- **Colours:** Auto, Light, Dark, or Night (light notes on a dark page).
- Vid2Tab remembers the page you were on.

## Play along

**Play along** in play mode's toolbar shows the video beside the sheet (above it in a tall window).

- **K** plays and pauses (Space keeps turning pages, so a pedal never stops the music). Speed: 0.5x, 0.75x,
  1x or others; slower speeds keep the pitch.
- **Follow** (on by default): the line being played is tinted and the pages turn by themselves. Videos with a
  moving cursor also get a ring around the measure and a playhead. Turning a page yourself pauses following
  until the music gets there (or click Follow).
- **Click a measure** to jump there (one second early, to count in).
- **Drag across measures** to loop them. "Pause between repeats" adds a short break before each repeat; the
  x on the loop's label removes it.
- The video can sit beside the sheet, or be hidden (the sound only) for videos from your Mac. Drag the divider
  to resize it.

The video plays in YouTube's own player when it can. If YouTube does not allow embedding (or you are offline),
the copy Vid2Tab kept plays instead.

Timing is good but not perfect: with a cursor in the video it is usually within a third of a second; without
one, about half a second, more on the first and last line.

## "Sign in to confirm you're not a bot"

Sometimes YouTube refuses downloads with this message, more often after many downloads in a short time.
Vid2Tab then says so and points to the fix:

1. Open **Settings** (top right) and turn on **Use my browser's YouTube sign-in for downloads**.
2. Choose the browser you use YouTube in, and make sure you are signed in to YouTube there.
3. Try again.

This is off unless you turn it on. When it is on, the downloader reads that browser's YouTube cookies on your
Mac, only while downloading, and sends them only to YouTube. macOS may ask for permission the first time
(Safari may need Terminal to have "Full Disk Access" in System Settings > Privacy & Security; Chrome asks for
your Mac password to unlock its keychain).

**"YouTube did not offer the video for download"** (in the details: "Requested format is not available") usually
passes too: wait a while and try again. If it keeps happening, run **Update Vid2Tab.command**, and try with the
browser sign-in turned on (or off, if it is on).

Other download errors: run **Update Vid2Tab.command** first.

## Privacy

- Vid2Tab runs entirely on your Mac. Your videos and sheets stay in the Vid2Tab folder.
- Vid2Tab collects no data: no accounts, no analytics, no tracking.
- It talks to YouTube only to read a video's title, to download a video you asked for, and to show the
  embedded player in play along. The embedded YouTube player is run by Google and covered by
  [Google's Privacy Policy](https://policies.google.com/privacy) (Vid2Tab uses YouTube's privacy-enhanced
  mode). Using YouTube means following [YouTube's Terms of Service](https://www.youtube.com/t/terms).
- The browser sign-in setting above is off unless you turn it on.

## Report a bad sheet

If a sheet comes out wrong, please open an issue at
[github.com/kevincoj/vid2tab/issues](https://github.com/kevincoj/vid2tab/issues) (a free GitHub account is needed) with:

- the **YouTube link** (and the start and end time, if you set one),
- a **screenshot** of the wrong spot on the sheet, and if you can, of the same spot in the video,
- what is wrong ("measures 12 and 13 run together", "page 3 is missing"),
- the notes Vid2Tab showed (or the text from **Full report**).

## For programmers

The pipeline is also a command line tool:

```sh
.venv/bin/python -m tabvid URL_OR_VIDEO -o runs/NAME [--title T] [--artist A] [--start 1:40] [--end 4:24]
.venv/bin/python -m tabvid.web [--port 8765] [--sheets sheets] [--no-open]
```

Tests: `./install.sh --dev`, then `.venv/bin/python -m pytest tabvid/tests -q`.

## License

[MIT](LICENSE). The fonts in `tabvid/web/fonts/` (Fraunces, Figtree) are under the SIL Open Font License
(the OFL files next to them).
