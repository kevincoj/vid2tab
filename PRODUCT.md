# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Guitarists, mainly fingerstyle players, who learn from YouTube TAB/lesson videos and want the scrolling
or page-flipping TAB as a clean sheet. They download Vid2Tab from GitHub and run it locally on a Mac. Many
have never used a command line, so the web app must make sense on its own.

## Product Purpose

Vid2Tab turns a guitar TAB video (YouTube link or local video file) into a printable PDF sheet: the user
marks the music area and any watermarks on a real frame, the pipeline builds the sheet, and the app lists
anything worth checking in plain words. Success: a sheet the player trusts enough to play from, made
without touching the terminal.

## Positioning

It reads the TAB the video already shows (no transcription by ear, no AI guessing of notes): the sheet is
the video's own engraving, cleaned, deduplicated and laid out for paper and screen.

## Operating Context

- Runs on the user's laptop: a local server opens one page in the browser; no account, nothing hosted.
- Sheets are used both printed (music stand) and on screen (iPad/laptop while playing); screen reading
  matters a lot.
- Planned: a "play mode" for reading while playing: full screen, one page fitted to screen height, page
  turns with Space/arrow keys (Bluetooth page-turner pedals), half-page turns, dark mode.
- Planned later: "play along": YouTube embedded player beside the sheet, current measure highlighted,
  automatic page turns. The sheet/play screen must leave room for a video panel.

## Capabilities and Constraints

- Screens: home (new sheet form + library), make (progress), crop (music area, then watermarks), sheet
  (PDF, things to check, remake options: area, time range, as played / as written).
- The pipeline, CLI and the printed PDF layout do not change for UI work.
- Terms: sheet, music area, watermarks, measures, "as played" / "as written", "things to check".

## Brand Commitments

- Name: Vid2Tab. Colours and logo are open (the old teal need not stay).
- Voice: plain words, no pipeline jargon.

## Evidence on Hand

Real sheets in `sheets/` (e.g. White Christmas, Sicilienne Op.78, Andantino in Am) with PDFs, thumbnails
and reports. No testimonials, users or download counts exist; none may be invented.

## Product Principles

1. The sheet is the product: every screen exists to get to a sheet you can play from.
2. Show, don't ask: decisions are made on real frames and real pages, not abstract options.
3. Always a result: a PDF plus honest, plain-word notes, never a dead end.
4. Built for the music stand: readable at arm's length, hands-free page turns.
