"""Warnings from report.json in plain language, for the web app's sheet screen.

Each rule: (stage or None, pattern, level, sentence). The pattern is searched in the message; the
sentence is a function of the match. Levels: 'problem' (no usable sheet), 'check' (look at that spot),
'note' (small, folded away). A message no rule matches is shown as it is, as a 'check', so a new
warning in the pipeline is never hidden. Frame times in messages are seconds into the processed clip;
`offset` (the range's start) turns them into video times.
"""
import ast
import re


def _list(text):
    """'[2, 3, 4]' -> [2, 3, 4] (anything else: [])."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _join(items, limit=8):
    items = [str(i) for i in items]
    if len(items) > limit:
        items = items[:limit] + [f'{len(items) - limit} more']
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def _clock(seconds):
    seconds = int(round(seconds))
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def _times(spans, offset):
    """["'#7-#10 (3.5-5.0s, middle)'", ...] -> '0:04-0:05, 0:07 and ...' in video time."""
    out = []
    for span in spans:
        found = re.search(r'\(([\d.]+)-([\d.]+)s', str(span))
        if found:
            a, b = (_clock(offset + float(v)) for v in found.groups())
            out.append(a if a == b else f'{a}-{b}')
    return _join(out, 4) if out else 'some moments'


def _spans(text):
    """'rows 9-10 (86.5-104.5s, m35-m43) = rows 1-2; ...' -> ['(86.5-104.5s', ...] for _times."""
    return re.findall(r'\([\d.]+-[\d.]+s', text)


def _measures(items):
    items = list(items)
    return ('measure ' if len(items) == 1 else 'measures ') + _join(items)


def _row(label):
    """'row-07', 'row 7', 'view 3', 'strip 2' -> 'row 7' / 'part 3 of the scroll'."""
    kind, number = re.match(r'(row|view|strip)[- ]?0*(\d+)', label).groups()
    return f'row {number}' if kind == 'row' else f'part {number} of the scrolling video'


ROW = r'((?:row|view|strip)[- ]\d+)'

RULES = [
    # ---- problems: no usable sheet --------------------------------------------------------------
    (None, r'^stage (\d) failed \((\w+): (.*)\); wrote a one-page PDF', 'problem',
     lambda m, o: f'The sheet could not be made: something went wrong in step {m[1]} of 5. The PDF shows the error.'),
    ('rows', r'^No TAB staff found in any of the (\d+) frames', 'problem',
     lambda m, o: 'No TAB was found in the chosen music area. Adjust the area and make the sheet again.'),
    ('measures', r'^No measures: no rows came out', 'problem',
     lambda m, o: 'No measures were found, so the sheet is empty.'),
    ('sheet', r'^No measures to lay out', 'note',
     lambda m, o: 'The PDF holds a short note instead of music.'),

    # ---- measure count ----------------------------------------------------------------------------
    ('numbers', r'(\d+) measure\(s\) missing between m(\d+) \(number (\d+)\) and m(\d+) \(number (\d+)\)', 'check',
     lambda m, o: f'About {m[1]} measure{"s" if m[1] != "1" else ""} may be missing between measures {m[2]} and '
                  f'{m[4]}: the printed numbers jump from {m[3]} to {m[5]}. Compare with the video there.'),
    ('numbers', r'(\d+) measure\(s\) too many between m(\d+) \(number (\d+)\) and m(\d+) \(number (\d+)\)', 'check',
     lambda m, o: f'There may be {m[1]} measure{"s" if m[1] != "1" else ""} too many between measures {m[2]} and '
                  f'{m[4]} (printed {m[3]} to {m[5]}): a measure split in two, or shown twice.'),
    ('numbers', r'numbers go back from (\d+) to (\d+) between m(\d+) \(number \d+\) and m(\d+)', 'check',
     lambda m, o: f'The printed measure numbers go back from {m[1]} to {m[2]} at measure {m[4]}: a page may be '
                  'shown twice, or the music repeats with its numbers.'),
    ('numbers', r'^measure numbers could not be read reliably', 'note',
     lambda m, o: 'The printed measure numbers could not be read, so the number of measures was not checked.'),
    ('numbers', r'^measure (\d+) \(row \d+\) holds a second (.+?) \d+px after its start', 'check',
     lambda m, o: f'Measure {m[1]} shows two measure numbers: a barline between them was probably missed.'),
    ('numbers', r'^measures (\[.*?\]) have no printed number while', 'check',
     lambda m, o: f'{_measures(_list(m[1])).capitalize()} have no printed number while the others do: a measure '
                  'may have been split in two, or it is a pickup.'),
    ('numbers', r'^measure-number check failed', 'note',
     lambda m, o: 'The measure-number check was skipped after an error.'),

    # ---- measure widths ---------------------------------------------------------------------------
    ('measures', r'^measure (\d+) \(row \d+\) is (\d+)px wide vs a typical (\d+)px', 'check',
     lambda m, o: f'Measure {m[1]} is about {int(m[2]) / int(m[3]):.1f} times as wide as usual: a barline inside it '
                  'may have been missed, so it may hold two measures.'),
    ('measures', r'^wide measures: (.*?), over', 'note',
     lambda m, o: f'{_measures(re.findall(r"(\d+) \(", m[1])).capitalize()} '
                  f'{"is" if len(re.findall(r"(\d+) [(]", m[1])) == 1 else "are"} unusually long. Probably fine; '
                  'a barline may have been missed.'),
    ('measures', r'^measure (\d+) \(row \d+\) is \d+px wide vs a row median of \d+px', 'check',
     lambda m, o: f'Measure {m[1]} is very narrow: check the barlines around it (a stem or arrow may have been '
                  'taken for a barline).'),
    ('structure', r'^measure (\d+) \(row \d+\) is \d+px wide vs a typical \d+px and holds no notes', 'note',
     lambda m, o: f'Measure {m[1]} holds no notes (a time signature, or the cut-off end of a page).'),

    # ---- structure --------------------------------------------------------------------------------
    ('structure', r'^measure (\d+) \(start of row \d+\) and measure (\d+) \(end of row \d+\) show the same music', 'check',
     lambda m, o: f'Measures {m[2]} and {m[1]} show the same music, one of them cut short: the new page repeats the '
                  'end of the old one.'),
    ('structure', r'^row \d+ \(measures (\d+)-(\d+)\) repeats row \d+ measure for measure', 'check',
     lambda m, o: f'Measures {m[1]}-{m[2]} repeat the line before them measure for measure: the same page shown '
                  'twice, or the music repeats a whole line.'),
    ('structure', r'^row (\d+): the music runs on \d+px past its last barline to the crop edge', 'check',
     lambda m, o: f'Row {m[1]} runs past its last barline to the edge of the music area; that end is left out '
                  '(the page may be cut mid-measure).'),
    ('structure', r'^structure check failed', 'note',
     lambda m, o: 'A check for repeated pages was skipped after an error.'),

    # ---- rows: missing or doubtful music ----------------------------------------------------------
    ('rows', r'^frames (\[.*\]) \(\d+ frames\) show long horizontal lines but no TAB staff', 'check',
     lambda m, o: f'Some music may be missing: at {_times(_list(m[1]), o)} the video shows staff lines that were '
                  'not read. If the staff is cut off there, adjust the music area.'),
    ('rows', r'^frames (\[.*\]) after the last measure still show staff lines', 'check',
     lambda m, o: f'The sheet may be missing its ending: the video still shows staff lines after the last '
                  f'measure ({_times(_list(m[1]), o)}).'),
    ('rows', r'^frames (\[.*\]) show a page under a dark or coloured overlay', 'check',
     lambda m, o: f'A page shown only under a dark or coloured overlay ({_times(_list(m[1]), o)}) is missing '
                  'from the sheet.'),
    ('rows', r'^frames (\[.*\]) \(\d+ of \d+ frames with a staff\) show two or more TAB staves', 'note',
     lambda m, o: f'At {_times(_list(m[1]), o)} the video shows more than one TAB staff; only the lowest one was read.'),
    ('rows', ROW + r': the staff breaks off and starts again', 'check',
     lambda m, o: f'{_row(m[1]).capitalize()} has two pieces of staff side by side (a coda?), which is not '
                  'supported: measures may be joined across the gap.'),
    ('rows', ROW + r': fewer than two barlines found', 'check',
     lambda m, o: f'No barlines were found in {_row(m[1])}, so it became a single measure.'),
    ('measures', r'^row (\d+): no barlines to cut at', 'note',
     lambda m, o: f'Row {m[1]} was kept as one measure.'),
    ('measures', r'^row (\d+): \d+px of music after its last barline', 'check',
     lambda m, o: f'Row {m[1]} ends without a closing barline; its last part is kept as a measure (the page may be '
                  'cut mid-measure).'),
    ('rows', r'^row (\d+) -> (\d+): no confident overlap', 'check',
     lambda m, o: f'Rows {m[1]} and {m[2]} of the scrolling video could not be joined with confidence: check the '
                  'join between them.'),
    ('rows', r'^row (\d+) -> (\d+): .*; check the join', 'check',
     lambda m, o: f'Check the join between rows {m[1]} and {m[2]}: the video scrolled an unusual amount there.'),
    ('measures', r'^strip \d+: m\d+-m\d+ = m(\d+)-m(\d+) \(numbers (\d+)-(\d+)\): the scroll join found no overlap', 'note',
     lambda m, o: f'The video showed measures {m[3]}-{m[4]} twice where it scrolled; the copy was left out.'),
    ('rows', r'^view (\d+): not covered by the views around it', 'note',
     lambda m, o: f'Part {m[1]} of the scrolling video is kept as its own line.'),
    ('rows', r'^strip (\d+): no staff found on the stitched strip', 'check',
     lambda m, o: f'Part {m[1]} of the scrolling video was left out: no staff was found in it.'),
    ('rows', ROW + r': only one usable frame', 'note',
     lambda m, o: f'{_row(m[1]).capitalize()} was shown for only a moment, so a moving cursor or highlight may '
                  'still be visible.'),
    ('rows', ROW + r': every frame looked like a transition', 'note',
     lambda m, o: f'{_row(m[1]).capitalize()} looked like a fade in every frame; it was kept anyway.'),
    ('rows', r'^rows (\[.*?\]): no notation staff found over the TAB staff', 'note',
     lambda m, o: f'Rows {_join(_list(m[1]))} have no notation staff above the TAB (the other rows do); they are '
                  'printed as TAB only.'),
    ('rows', ROW + r': TAB barlines at x=.* have no barline in the notation staff', 'note',
     lambda m, o: f'In {_row(m[1])} a TAB barline has no matching barline in the notation above it; kept.'),
    ('rows', r'^(\d+) barlines run on past the staff within 1px', 'note',
     lambda m, o: f'{m[1]} barlines only just passed the barline test; if two measures look merged, look there.'),
    ('rows', r'^page (\d+): \d+ piece\(s\) of ink cross the cut between its staves', 'note',
     lambda m, o: f'Page {m[1]}: a few marks between its staves were given to the nearer staff.'),
    ('rows', r'^page (\d+), row (\d+) of (\d+): no staff found in its band', 'check',
     lambda m, o: f'Page {m[1]}: row {m[2]} of {m[3]} was not found and is left out.'),
    ('rows', r'^stacked staves in a scrolling video are not supported', 'check',
     lambda m, o: 'This video scrolls with several staves on screen, which is not fully supported: it was laid out '
                  'page by page.'),
    ('rows', r'^notation \+ TAB with stacked systems is not supported', 'check',
     lambda m, o: 'Notation + TAB with several systems on screen is not supported yet: a notation staff may sit '
                  'with the wrong TAB line.'),
    ('rows', r'^a notation staff sits over the TAB staff in \d+ of \d+ views, but the video scrolls', 'check',
     lambda m, o: 'This scrolling video has notation above the TAB; only the TAB is fully supported when '
                  'scrolling, so the notation is kept as it is.'),

    # ---- measures: kept as printed ----------------------------------------------------------------
    ('measures', r'^TAB clef not recognised at the start of rows (\[.*?\])', 'note',
     lambda m, o: f'The TAB clef at the start of rows {_join(_list(m[1]))} was not recognised and is left as printed.'),
    ('measures', r'^chord names over m(\d+) and (m[\d, m]+?) touch in the source', 'note',
     lambda m, o: f'The chord names of measures {m[1]} and {m[2].replace("m", "")} touch in the video, so they are '
                  f'printed together with measure {m[1]}.'),
    ('measures', r"^ink touching the crop's top or bottom edge", 'note',
     lambda m, o: 'Something is cut by the edge of the music area (a label or a page number); kept as printed. '
                  'If it is part of the music, make the area a little bigger.'),
    ('measures', r'^(\d+) ties/slurs cross a barline', 'note',
     lambda m, o: (f'{m[1]} ties or slurs cross a barline; ' if m[1] != '1' else 'A tie or slur crosses a barline; ') +
                  'where a measure starts a new line they may look cut.'),
    ('measures', r"^this video replays pages: (.*); choose 'as written'", 'note',
     lambda m, o: f'The video shows some pages again ({_times(_spans(m[1]), o)}). If the printed score has repeat '
                  'signs or D.C./D.S. markings, make it "as written" to print each page once.'),
    ('measures', r'^as written: replayed pages printed once: (.*) \(the same image', 'note',
     lambda m, o: f'As written: pages the video shows again ({_times(_spans(m[1]), o)}) are printed once.'),
    ('measures', r'^as written: pages are shown again \((.*)\), but no measure numbers are read', 'check',
     lambda m, o: f'The video shows some pages again ({_times(_spans(m[1]), o)}), but no measure numbers could be '
                  'read to confirm it, so the sheet is as played (every page as the video shows it).'),
    ('measures', r'^as written: (.*) show a page again, but its measure numbers could not be read', 'check',
     lambda m, o: f'A page shown again ({_times(_spans(m[1]), o)}) is kept: its measure numbers could not be read '
                  'to confirm it is the same page.'),
    ('measures', r'^as written: (.*) show a page again without a printed measure number', 'check',
     lambda m, o: f'A page shown again ({_times(_spans(m[1]), o)}) is kept: it has no measure number to confirm '
                  'it is the same page.'),
    ('measures', r'^as written: row \d+ \(([\d.]+-[\d.]+s)\) nearly repeats row \d+', 'check',
     lambda m, o: f'A page at {_times(["(" + m[1]], o)} looks like an earlier one with the same measure numbers, but '
                  'not exactly, so it is kept; check whether it is printed twice.'),
    ('measures', r'^as written: a scrolling video is always laid out as played', 'note',
     lambda m, o: 'Scrolling videos are always made as played (repeated parts are not recognised).'),
    ('measures', r'^the notation header \(clef, key\) of rows (\[.*?\]) differs', 'note',
     lambda m, o: f'The key or clef changes at rows {_join(_list(m[1]))}, so lines start without a clef and key.'),

    # ---- sheet layout -----------------------------------------------------------------------------
    ('sheet', r'^line (\d+) is stretched ([\d.]+)x', 'note',
     lambda m, o: f'Line {m[1]} is spread out more than usual (the music is dense there).'),
    ('sheet', r'^lines (\[.*?\]): an overhang wider than', 'note',
     lambda m, o: f'{"Line" if len(_list(m[1])) == 1 else "Lines"} {_join(_list(m[1]))}: a mark sticking out '
                  'before the first measure was cut at the margin.'),
    ('sheet', r'^overhang wider than the line before it, cut off: (.*); check the measure image', 'check',
     lambda m, o: f'A mark at the start of {_measures(re.findall(r"m(\d+)", m[1]))} was cut off; check '
                  f'{"it" if len(re.findall(r"m(\d+)", m[1])) == 1 else "them"} in the PDF.'),
    ('sheet', r'^m(\d+) and m(\d+) still touch outside the staff', 'check',
     lambda m, o: f'Measures {m[1]} and {m[2]} still touch above or below the staff; check that spot in the PDF.'),
    ('sheet', r'^measures (\[.*?\]): clean\(\) would have trimmed them', 'note',
     lambda m, o: f'{_measures(_list(m[1])).capitalize()} are printed exactly as captured (a dark background: a '
                  'fade or overlay?).'),

    # ---- video and crop ---------------------------------------------------------------------------
    ('intake', r'^the range .* runs past the end of the video: the clip is [\d.]+s, ending at ([\d:.]+)', 'note',
     lambda m, o: f'The end time is past the end of the video; the sheet goes to the end ({m[1]}).'),
    ('intake', r'^Low resolution video \((\d+)x(\d+)\)', 'check',
     lambda m, o: f'The video is low resolution ({m[1]}×{m[2]}); small TAB numbers may be hard to read.'),
    ('crop', r'^No six-line staff found in sampled frames', 'check',
     lambda m, o: 'No TAB staff was found automatically, so the music area was a guess.'),
    ('crop', r'^TAB staff found in only (\d+)/(\d+) sampled frames', 'note',
     lambda m, o: f'The TAB was found in only {m[1]} of {m[2]} moments of the video when guessing the music area.'),
]
RULES = [(stage, re.compile(pattern), level, sentence) for stage, pattern, level, sentence in RULES]
ORDER = {'problem': 0, 'check': 1, 'note': 2}


def explain_one(stage, message, offset=0):
    """{level, text, detail, stage, kind} for one warning (kind: the rule's index, to group alike ones)."""
    for kind, (rule_stage, pattern, level, sentence) in enumerate(RULES):
        if rule_stage not in (None, stage):
            continue
        found = pattern.search(message)
        if found:
            try:
                return {'level': level, 'text': sentence(found, offset or 0), 'detail': message, 'stage': stage,
                        'kind': kind}
            except (IndexError, ValueError, AttributeError):
                break        # a message that matched in shape but not in detail: show it as it is
    return {'level': 'check', 'text': message[:1].upper() + message[1:], 'detail': message, 'stage': stage,
            'kind': None, 'unmatched': True}


def explain(warnings, offset=0):
    """report.json's [{stage, message}] as [{level, text, detail, stage}], problems first."""
    out = [explain_one(w.get('stage', ''), w.get('message', ''), offset) for w in warnings]
    return sorted(out, key=lambda w: ORDER[w['level']])


# Stages that number measures before as-written replays are dropped (4d renumbers the rest).
_BEFORE_REPLAYS = {'rows', 'measures', 'numbers', 'structure'}


def _dropped(raw):
    """Measure numbers dropped as replays, from the 'replayed pages printed once' warning (pre-drop numbers)."""
    out = set()
    for w in raw:
        message = w.get('message', '')
        if 'replayed pages printed once' not in message:
            continue
        for part in message.split(';'):
            for a, b in re.findall(r'm(\d+)-m(\d+)', part.split(' = ')[0]):
                out.update(range(int(a), int(b) + 1))
    return out


def measures_of(warning, raw=()):
    """The sheet's measure numbers a warning is about (for 'show on the page'): read from its plain text,
    'between measures 14 and 23' as the whole range, and shifted past measures dropped as replays when the
    warning's stage numbered them before the drop. [] when it names none."""
    found = re.findall(r'\b[Mm]easures? (\d+(?:(?:, | and |-|–)\d+)*)', warning.get('text', ''))
    numbers = []
    for group in found:
        items = [int(n) for n in re.findall(r'\d+', group)]
        if re.search(r'between [Mm]easures ' + re.escape(group), warning['text']) and len(items) == 2 \
                or re.fullmatch(r'\d+[-–]\d+', group):
            items = list(range(items[0], items[-1] + 1))
        numbers += [n for n in items if n not in numbers]
    if 'replayed pages printed once' in warning.get('detail', '') or warning.get('stage') not in _BEFORE_REPLAYS:
        return numbers[:60]
    dropped = _dropped(raw)
    return [n - sum(d < n for d in dropped) for n in numbers if n not in dropped][:60]
