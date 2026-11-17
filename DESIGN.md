---
name: Vid2Tab
description: Turns a guitar TAB video into a clean sheet to print or play from.
colors:
  matcha: "#5c7041"
  matcha-text: "#4c5f35"
  matcha-wash: "#e3e8d3"
  on-matcha: "#ffffff"
  latte-ground: "#eeebdf"
  latte-band: "#e3dfc9"
  latte-surface: "#f7f5ec"
  cream-paper: "#fcfbf4"
  ink: "#2a2a20"
  soft-ink: "#5c5b4d"
  hairline: "#d8d3bf"
  hairline-strong: "#bdb79f"
  check-amber: "#94561a"
  check-wash: "#f2e4cf"
  problem-red: "#a3352a"
  problem-wash: "#f5dfda"
  crop-green: "#4fb34a"
  mask-orange: "#f08a24"
  dark-ground: "#1d1c17"
  dark-band: "#26251e"
  dark-surface: "#24231d"
  dark-field: "#2b2a22"
  dark-paper: "#e8e4d4"
  dark-ink: "#ece9dc"
  dark-soft-ink: "#b1ac9a"
  dark-hairline: "#3b392f"
  dark-hairline-strong: "#57543f"
  dark-matcha: "#b4c88e"
  dark-matcha-text: "#c0d29d"
  dark-matcha-wash: "#333a26"
  on-dark-matcha: "#1c2311"
  dark-check-amber: "#e8b06a"
  dark-problem-red: "#f0948a"
  night-paper: "#2a2923"
typography:
  display:
    fontFamily: "Fraunces, Georgia, serif"
    fontSize: "clamp(30px, 4vw, 42px)"
    fontWeight: 500
    lineHeight: 1.1
    letterSpacing: "-0.015em"
    fontVariation: "\"SOFT\" 100"
  headline:
    fontFamily: "Fraunces, Georgia, serif"
    fontSize: "28px"
    fontWeight: 500
    lineHeight: 1.15
    fontVariation: "\"SOFT\" 100"
  title:
    fontFamily: "Fraunces, Georgia, serif"
    fontSize: "19px"
    fontWeight: 500
    lineHeight: 1.15
    fontVariation: "\"SOFT\" 100"
  body:
    fontFamily: "Figtree, -apple-system, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Figtree, -apple-system, Segoe UI, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.5
  small:
    fontFamily: "Figtree, -apple-system, Segoe UI, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
rounded:
  page: "1px"
  inner: "4px"
  control: "6px"
  panel: "8px"
  round: "50%"
spacing:
  gutter: "clamp(16px, 4vw, 56px)"
  xs: "6px"
  sm: "12px"
  md: "22px"
  lg: "34px"
  xl: "72px"
components:
  button-primary:
    backgroundColor: "{colors.matcha}"
    textColor: "{colors.on-matcha}"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "42px"
    typography: "{typography.label}"
  button-secondary:
    backgroundColor: "{colors.cream-paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "42px"
    typography: "{typography.label}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.soft-ink}"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "42px"
  input-text:
    backgroundColor: "{colors.cream-paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "0 14px"
    height: "48px"
    typography: "{typography.body}"
  input-paste:
    backgroundColor: "{colors.cream-paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.panel}"
    padding: "0 20px"
    height: "64px"
  segmented:
    backgroundColor: "{colors.cream-paper}"
    rounded: "{rounded.control}"
    padding: "3px"
  segmented-on:
    backgroundColor: "{colors.matcha}"
    textColor: "{colors.on-matcha}"
    rounded: "{rounded.inner}"
    height: "34px"
  chip:
    backgroundColor: "{colors.cream-paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "2px 2px 2px 10px"
  panel:
    backgroundColor: "{colors.latte-surface}"
    rounded: "{rounded.panel}"
    padding: "22px"
  page:
    backgroundColor: "{colors.cream-paper}"
    rounded: "{rounded.page}"
---

# Design System: Vid2Tab

## Overview

**Creative North Star: "The Music Stand"**

Everything in Vid2Tab serves the page you play from. The app is a calm, warm room, latte-coloured ground in
a soft matcha tint, and the sheet is the one object that stands out of it: cream paper with a real page
shadow, lifted off the ground like a sheet clipped to a stand. Controls step back. There is one solid
matcha action per screen, the rest are outlined or bare, and anything that is not about the music (the
toolbar in play mode, the steps of making a sheet) gets out of the way of the page.

It is built for arm's length and busy hands. Targets are large (42 px buttons, a 64 px paste field), type
is plain and readable, and the words are the user's, not the pipeline's ("music area", "things to check").
The serif song titles give it a printed-music warmth. Everything else is a quiet sans.

Three colour themes share one set of tokens: Matcha latte (light), Matcha dark (the paper dimmed, never
inverted), and Night (Matcha dark with light notes on a dark page, for playing in a dim room). Auto follows
the Mac between light and dark. Print is always black on white.

**Key Characteristics:**
- Warm, low-contrast ground; the page is the brightest, most lifted thing on screen.
- One matcha accent for actions and the current choice; amber and red only for things to check and problems.
- Serif (Fraunces) for song titles and headings, sans (Figtree) for everything else.
- Large, flat controls with soft 6 px corners; no gradients, no glows.
- The sheet images are never recoloured beyond dimming (dark) or inverting (night).

## Colors

A warm, slightly green-tinted cream palette with one olive-matcha accent; dark mode is a deep olive-brown,
not black.

### Primary
- **Matcha** (`{colors.matcha}`): primary buttons, the selected option of a switch, step numbers, progress
  bars, focus rings, the logo. In dark and night it lightens to **Dark Matcha** (`{colors.dark-matcha}`)
  with near-black text on it (`{colors.on-dark-matcha}`).
- **Matcha Text** (`{colors.matcha-text}`): matcha used as text (links under warnings, "being made" status).
- **Matcha Wash** (`{colors.matcha-wash}`): pressed toggles, done steps, notices, input focus halo.

### Secondary
- **Check Amber** (`{colors.check-amber}`, wash `{colors.check-wash}`): "to check" status and warnings only.
- **Problem Red** (`{colors.problem-red}`, wash `{colors.problem-wash}`): errors, failed sheets, delete.

### Tertiary
- **Crop Green** (`{colors.crop-green}`) and **Mask Orange** (`{colors.mask-orange}`): only on video frames,
  for the music-area box and watermark boxes, where they must read over any picture.

### Neutral
- **Latte Ground** (`{colors.latte-ground}`): the page background.
- **Latte Band** (`{colors.latte-band}`): the top bar and the tinted band some screens continue under it.
- **Latte Surface** (`{colors.latte-surface}`): panels.
- **Cream Paper** (`{colors.cream-paper}`): sheet pages, covers, and the fill of fields and outlined buttons.
- **Ink** (`{colors.ink}`) and **Soft Ink** (`{colors.soft-ink}`): text and secondary text.
- **Hairline** (`{colors.hairline}`) and **Hairline Strong** (`{colors.hairline-strong}`): borders and
  dividers; strong for hover borders and step connectors.
- Dark equivalents: `{colors.dark-ground}`, `{colors.dark-band}`, `{colors.dark-surface}`,
  `{colors.dark-field}`, `{colors.dark-ink}`, `{colors.dark-soft-ink}`, `{colors.dark-hairline}`,
  `{colors.dark-hairline-strong}`; dark paper `{colors.dark-paper}`, shown at 90 % brightness.
- **Night Paper** (`{colors.night-paper}`): the page in Night; the black-on-white page image is inverted
  (`invert(.9) hue-rotate(180deg) sepia(.1)`) and laid on it with `lighten`, so notes come out warm off-white.

### Named Rules
**The One Matcha Rule.** One solid matcha button per screen (Next, Make the sheet, Play). Everything else
is outlined or quiet. Matcha marks the action or the current choice, never decoration.

**The Tokens Only Rule.** Every colour comes from a CSS custom property on `:root`, redefined for dark
(`prefers-color-scheme` when Auto, `[data-theme=dark]`) and Night (`[data-theme=night]`). New screens add
tokens, never hex values in rules (the crop/mask colours and the warning ring are the existing exceptions).

**The Paper Is Never Painted Rule.** Page images are multiplied onto the paper colour (light, dark) or
inverted onto it (night) through `--page-blend` and `--page-filter`; nothing else tints them. Warnings are
drawn as rings over the page, never by recolouring it.

## Typography

**Display Font:** Fraunces (with Georgia, serif), variable, `"SOFT" 100`, weight 500
**Body Font:** Figtree (with -apple-system, Segoe UI, sans-serif), 400 to 700

**Character:** a soft, slightly old-style serif for the names of songs and for headings, like the title
on printed music, against a friendly, plain geometric sans that stays out of the way. Both fonts are
served from `tabvid/web/fonts/` so the app works offline.

### Hierarchy
- **Display** (500, clamp(30px, 4vw, 42px), 1.1, -0.015em): the home question "What would you like to play?".
- **Headline** (500, 28 to 34px, 1.1 to 1.15): screen titles: the song on the sheet screen, the details and
  crop steps, "Your sheets".
- **Title** (500, 19 to 22px, 1.15): song titles on covers and the Continue strip, the play-mode toolbar title.
- **Body** (400, 16px, 1.5): running text; ledes in Soft Ink, at most 75ch.
- **Label** (600 to 700, 15px): buttons, field labels, step names, page tabs.
- **Small** (400 to 700, 14px): hints, statuses, metadata under covers, `details` summaries.

### Named Rules
**The Serif Is for Songs Rule.** Fraunces is for song titles and screen headings only; every control,
label and paragraph is Figtree. Small section headings inside forms ("Repeats") stay Figtree 700.

**The Tabular Numbers Rule.** Page counters, times and progress details use `font-variant-numeric:
tabular-nums` so they do not jitter as they change.

## Layout

Screens sit in a centred wrap (max 1440px) with a fluid side gutter (`{spacing.gutter}`). The top bar is
72px tall on the Latte Band; the home and sheet screens continue that band under it for their first block.
The library is a grid of covers (`minmax(168px, 1fr)`, 44px rows by 34px columns; 136px under 760px).
The sheet screen is the page (max 760px) beside a 340px sticky side column of things to check and remake
options; under 960px the side column drops below. The details step is a 560px picture beside the form.

Spacing is set per component from a loose rhythm of about 6, 12, 22, 34 and 72px rather than a strict
scale. One breakpoint does most of the work: 760px (stacked paste field, labels hidden on the step bar and
the colours switch).

**Play mode** is its own full-window layout: a toolbar strip on top and the pages in the rest, fitted to
its height (one page, or two side by side when they fit), 12px from the edges and 16px apart. The toolbar
never overlaps a page; in full screen it is hidden until the mouse moves, and the pages shrink to fit
below it while it shows.

### Named Rules
**The Page Is Never Covered Rule.** No bar, toast, button or badge may sit over the music. Toolbars take
their own strip and the pages are fitted below them; the small page count only shows where it has room
beside the pages.

## Elevation & Depth

The interface is flat: bands, panels and controls are separated by tone and 1 to 1.5px hairlines, not
shadows. Depth is reserved for paper. Pages and covers carry a layered page shadow that makes them look
lifted off the ground, and covers of longer sheets show one or two pages tucked behind, slightly rotated.

### Shadow Vocabulary
- **Page** (`0 1px 1px rgba(40,35,20,.10), 0 2px 4px rgba(40,35,20,.06), 0 16px 22px -14px rgba(40,35,20,.42)`;
  dark: `0 1px 2px rgba(0,0,0,.4), 0 16px 24px -14px rgba(0,0,0,.7)`): sheet pages, covers, play-mode pages.
- **Picture** (`0 1px 2px rgba(40,30,20,.12), 0 14px 26px -16px rgba(40,30,20,.5)`): the video thumbnail
  on the details step.

### Named Rules
**The Only Paper Floats Rule.** If it is not a page (or a picture of the video), it has no shadow.

## Shapes

Soft but not round: controls and fields have 6px corners (`{rounded.control}`), panels and the big paste
field 8px (`{rounded.panel}`), segment buttons and thumbnails 4px inside their 6px frame, and pages are
nearly square-cornered (`{rounded.page}`) like real paper. Circles are only for step numbers, progress
dots and status dots. Borders are 1.5px on controls and 1px on dividers.

## Components

Quiet and sturdy: big, easy targets, flat fills, soft corners, and only one thing on a screen asks to be
pressed.

### Buttons
- **Shape:** gently rounded (6px), 42px tall, 0 16px padding, Figtree 600 at 15px, icon (18px line icon)
  before the label with an 8px gap.
- **Primary:** Matcha fill, white label at 700 (`{components.button-primary}`); hover brightens by 6%.
  Larger where it ends a step: 52px (Make the sheet), 64px (Next beside the paste field).
- **Secondary:** Cream Paper fill, Hairline border, Ink label; hover darkens the border to Hairline Strong.
- **Quiet:** no fill or border, Soft Ink label; hover shows Ink and a Hairline border. Used in toolbars.
  Pressed toggles (`aria-pressed`) take the Matcha Wash with Matcha Text.
- **Link:** underlined text with a Hairline Strong underline, 4px offset, which turns to the text colour on hover.
- **Focus:** a 2.5px matcha outline, 2px offset, on every focusable thing.

### Segmented switch
A Cream Paper frame with a 1.5px Hairline border and 3px padding holding 34px buttons; the chosen one is a
Matcha fill. Used for every one-of-a-few choice: as played / as written, 1 page / 2 pages, and the
**colours switch** (Auto / Light / Dark / Night with icons), which is the same component in the top bar and
the play-mode toolbar and changes the theme app-wide; its labels hide on narrow windows, leaving icons.

### Chips
Cream Paper with a Hairline border, 6px corners, 14px 600 label and a 26px remove button; the active
watermark chip takes a Mask Orange border.

### Cards / Containers
- **Panel:** Latte Surface, 1px Hairline border, 8px corners, 22px padding.
- **Notice:** Matcha Wash (or Problem Wash) block, 6px corners, 12px 14px padding, no border.
- **Warning list:** plain items separated by hairlines; the selected one ringed on its page.

### Inputs / Fields
- **Style:** Cream Paper fill, 1.5px Hairline border, 6px corners, 48px tall, 16px text.
- **Paste field:** 64px tall, 18px text, 2px Matcha border, 8px corners: the one big input on the home screen.
- **Focus:** Matcha border plus a 3px Matcha Wash halo. **Error:** Problem Red border and a 14.5px 600 message below.
- **Choices:** custom 18px radio circles, Matcha dot when chosen, bold option name with a Soft Ink line under it.

### Navigation
The 72px top bar on Latte Band: the logo (a matcha pick with a play triangle) and "Vid2Tab" on the left,
the colours switch on the right. Screens add a quiet back link ("← Your sheets") rather than tabs. The
make-a-sheet flow shows a step bar: numbered circles joined by lines, current step in Matcha, done steps
in Matcha Wash.

### Sheet cover
A page thumbnail on Cream Paper with the page shadow, cropped to the top of page 1, one or two pages tucked
behind for longer sheets; it lifts 3px on hover. Below it: the song title in Fraunces, artist and page
count in Soft Ink, and a coloured status dot only when something needs attention.

### Play mode
The reading view: toolbar strip (back, title, page count with arrows, 1 page / 2 pages, the colours
switch, full screen) above pages fitted to the window height. Edges of the screen turn pages with a faint
arrow on hover; a small page count sits in a corner only where it fits beside the pages.

## Do's and Don'ts

### Do:
- **Do** take every colour from the tokens and define it for light, dark and night together.
- **Do** keep one solid matcha button per screen and make the rest outlined or quiet.
- **Do** give any bar its own strip and fit the pages below it (The Page Is Never Covered Rule).
- **Do** show pages through `--page-blend` and `--page-filter` so all three themes and print stay right.
- **Do** keep targets at least 40px tall in anything used while playing.
- **Do** use Fraunces for song titles and headings, Figtree for everything else.

### Don't:
- **Don't** invert the page in Dark: Dark dims the paper; inverting is Night's job.
- **Don't** let print or the downloaded PDF follow the theme: they are always black on white.
- **Don't** add shadows to panels, bars or buttons; only paper floats.
- **Don't** use amber or red for anything but things to check and problems.
- **Don't** overlay toolbars, toasts or badges on the music.
