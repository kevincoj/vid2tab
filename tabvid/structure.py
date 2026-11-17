"""Stage 4c: structure check. Catches cut, repeated and empty measures without printed numbers.

Warnings only: nothing is dropped or merged (songs repeat measures on purpose), and it never
blocks a run.

1. Cut row: the staff and its notes run on past a row's last barline to the crop edge. That
   music is in no measure: the page is cut mid-measure, or the strip scrolls.
2. Seam repeat: a row's first measure is a fragment found whole at the end (or start) of the
   previous row's last measure: the new page starts with the end of the old one.
3. Repeated row: every measure of a row matches the previous row's, one for one.
4. Empty sliver: a measure much narrower than typical holding no notes (nothing, or only a
   time signature).

When the printed numbers were read and their sequence is consistent, 2 and 3 are skipped: the
numbers already show the count is right.
"""
import cv2
import numpy as np

INK = 150
CUT_GAP = 2        # spacings of staff past the last barline for a row to count as cut
NOTE_INK = .1      # square spacings of ink between the staff lines that count as a note (a digit is ~.15)
FRAGMENT = .8      # a measure narrower than this share of another can be a fragment of it
MATCH = .05        # share of ink that may differ between two matching measures
WIDTH_TOL = 2      # px: measures of a repeated row have the same widths
SLIVER = .35       # a measure under this share of the typical width...
TALL = 1.8         # ...holding only glyphs at least this many spacings tall (a time signature) is empty


def _mismatch(a, b):
    ring = np.ones((3, 3), np.uint8)
    near_a = cv2.dilate(a.astype(np.uint8), ring).astype(bool)
    near_b = cv2.dilate(b.astype(np.uint8), ring).astype(bool)
    total = a.sum() + b.sum()
    return float(((a & ~near_b).sum() + (b & ~near_a).sum()) / total) if total else 0.0


def staff_ink(gray, lines, x0, x1):
    """Ink between the top and bottom staff lines in columns [x0, x1), staff-line rows removed."""
    top, bottom = round(lines[0]), round(lines[-1])
    ink = gray[top:bottom + 1, x0:x1] < INK
    for y in lines:
        ink[max(0, round(y) - top - 1):round(y) - top + 2] = False
    return ink


def holds_notes(ink, spacing):
    """True if the staff ink holds a glyph that is not full height (a barline) and not as tall
    as a time signature: a fret digit, a rest, a tie."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(ink.astype(np.uint8), np.ones((5, 1), np.uint8)), connectivity=8)  # rejoin across line rows
    for left, top, width, height, area in stats[1:]:
        if height < TALL * spacing and area >= NOTE_INK * spacing ** 2:
            return True
    return False


def _measure_ink(row, record):
    x0 = record['row_x0'] + record['staff_x0']
    return staff_ink(row['gray'], row['staff_lines'], x0, record['row_x0'] + record['staff_x1'])


def _same_height(a, b):
    height = min(a.shape[0], b.shape[0])
    return a[:height], b[:height]


def check(rows, records, numbers_consistent, report):
    by_row = {}
    for record in records:
        by_row.setdefault(record['row'], []).append(record)
    rows = [row for row in rows if row['row'] in by_row]
    inks = {r['measure']: _measure_ink(row, r) for row in rows for r in by_row[row['row']]}
    typical = float(np.median([r['staff_x1'] - r['staff_x0'] for r in records]))
    found = {'cut_rows': [], 'seam_repeats': [], 'repeated_rows': [], 'empty_slivers': []}

    for row in rows:
        spacing, bars = row['spacing'], row['barlines']
        width = row['gray'].shape[1]
        if not bars or any(r.get('open_end') for r in by_row[row['row']]):
            continue                      # music after the last barline is a measure already (warned in stage 4)
        after = bars[-1]['right'] + 1
        reaches_edge = row['staff_end'] >= width - 2
        if reaches_edge and row['staff_end'] - after > CUT_GAP * spacing and \
                holds_notes(staff_ink(row['gray'], row['staff_lines'], after, row['staff_end']), spacing):
            found['cut_rows'].append(row['row'])
            report.warn('structure', f"row {row['row']}: the music runs on {row['staff_end'] - after}px past its "
                        'last barline to the crop edge and is in no measure: the page is cut mid-measure, or '
                        'the strip scrolls.')

    for record in records:
        staff_width = record['staff_x1'] - record['staff_x0']
        if staff_width < SLIVER * typical and not holds_notes(inks[record['measure']], record['spacing']):
            found['empty_slivers'].append(record['measure'])
            report.warn('structure', f"measure {record['measure']} (row {record['row']}) is {staff_width}px wide "
                        f'vs a typical {typical:.0f}px and holds no notes: not a measure (a time signature '
                        'between two barlines, or the cut-off end of a page).')

    if not numbers_consistent:
        for before, row in zip(rows, rows[1:]):
            old, new = by_row[before['row']], by_row[row['row']]
            last, first = inks[old[-1]['measure']], inks[new[0]['measure']]
            small, big = (first, last) if first.shape[1] < last.shape[1] else (last, first)
            if small.shape[1] < FRAGMENT * big.shape[1] and holds_notes(small, row['spacing']):
                w = small.shape[1]
                ends = [_mismatch(*_same_height(small, big[:, -w:])), _mismatch(*_same_height(small, big[:, :w]))]
                if min(ends) <= MATCH:
                    found['seam_repeats'].append(new[0]['measure'])
                    report.warn('structure', f"measure {new[0]['measure']} (start of row {row['row']}) and measure "
                                f"{old[-1]['measure']} (end of row {before['row']}) show the same music, one of "
                                'them cut short: the new page repeats the end of the old one.')
            if len(old) == len(new) and all(
                    abs(inks[a['measure']].shape[1] - inks[b['measure']].shape[1]) <= WIDTH_TOL and
                    _mismatch(*_same_height(*_same_width(inks[a['measure']], inks[b['measure']]))) <= MATCH
                    for a, b in zip(old, new)):
                found['repeated_rows'].append(row['row'])
                report.warn('structure', f"row {row['row']} (measures {new[0]['measure']}-{new[-1]['measure']}) "
                            f"repeats row {before['row']} measure for measure: the same page captured twice, or "
                            'the music repeats a whole line.')

    report.stage('structure', structure_check='clean' if not any(found.values()) else
                 {key: value for key, value in found.items() if value},
                 repeats_checked='no (numbers read, sequence consistent)' if numbers_consistent else 'yes')
    return found


def _same_width(a, b):
    width = min(a.shape[1], b.shape[1])
    return a[:, :width], b[:, :width]


def safe_check(rows, records, numbers_consistent, report):
    """check(), but any failure is only a warning."""
    try:
        return check(rows, records, numbers_consistent, report)
    except Exception as error:
        report.warn('structure', f'structure check failed ({type(error).__name__}: {error}); skipped.')
        return {}
