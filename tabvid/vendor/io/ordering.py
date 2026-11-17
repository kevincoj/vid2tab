"""Deterministic capture-time ordering, including macOS 12-hour filenames."""
from datetime import datetime
from pathlib import PurePosixPath
import re

CAPTURE = re.compile(r'(\d{4}-\d{2}-\d{2}) at (\d{1,2})[.:](\d{2})[.:](\d{2})\s*([AP]M)', re.I)


def capture_time(name):
    match = CAPTURE.search(name.replace('\u202f', ' ').replace('\u00a0', ' '))
    if not match:
        return None
    day, hour, minute, second, meridiem = match.groups()
    try:
        return datetime.strptime(f'{day} {hour}:{minute}:{second} {meridiem.upper()}',
                                 '%Y-%m-%d %I:%M:%S %p')
    except ValueError:
        return None


def natural_key(name):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r'(\d+)', name))


def ordered_names(names, order='time', mtimes=None):
    """Timestamped captures sort chronologically; other names sort naturally.

    With mixed naming, timestamped files come first; inspect the contact sheet.
    Modification time only breaks case-insensitive natural-name ties.
    """
    mtimes = mtimes or {}
    def key(name):
        timestamp = capture_time(PurePosixPath(name).name) if order == 'time' else None
        return (0 if timestamp else 1, timestamp or datetime.min,
                natural_key(name), mtimes.get(name, 0), name)
    return sorted(names, key=key)
