"""tabvid: YouTube TAB lesson -> printable PDF.

    python -m tabvid SOURCE -o runs/NAME [--fps 2] [--until STAGE] [--recrop] [--remask] [--accept-guess]
                                 [--title T] [--artist A] [--start 12:30] [--end 15:45] [--as-written]
    python -m tabvid build SCREENSHOTS.pdf -o out.pdf      (original screenshot pipeline)

SOURCE is a YouTube URL or a local video file. Re-runs reuse the run folder's
video and frames. Stages: intake, crop, rows, measures, sheet.
"""
import argparse
import sys
from pathlib import Path

from .report import Report

STAGES = ['intake', 'crop', 'rows', 'measures', 'sheet']


def _stage(name):
    """One progress line per stage ('Stage 3/5: rows'); the web app reads them."""
    print(f'Stage {STAGES.index(name) + 1}/{len(STAGES)}: {name}', flush=True)


def _stage(name):
    """One progress line per stage ('Stage 3/5: rows'); the web app reads them."""
    print(f'Stage {STAGES.index(name) + 1}/{len(STAGES)}: {name}', flush=True)


def _error_pdf(run_dir, report, stage, number, error, title=None, artist=None):
    """A one-page PDF with the error, and a warning: a failed stage still leaves a PDF."""
    import traceback
    from reportlab.lib.pagesizes import letter
    from .sheet import heading, _empty_pdf
    traceback.print_exc()
    title, artist, pdf = heading(run_dir, title, artist)
    what = 'The sheet could not be laid out' if stage == 'sheet' else f'Stage {number} ({stage}) failed'
    _empty_pdf(pdf, title, letter, f'{what}: {type(error).__name__}: {error}\n'
                                   'See report.md for the warnings.', artist)
    report.warn(stage, f'stage {number} failed ({type(error).__name__}: {error}); wrote a one-page PDF with the error.')
    report.stage('sheet', pdf=pdf.name, pages=1, lines=0, failed=str(error))
    return pdf


def _sheet(run_dir, report, title=None, artist=None):
    """Stage 5. Any failure still leaves a one-page PDF with the error, and a warning."""
    from .sheet import build_sheet
    try:
        return build_sheet(run_dir, report, title, artist=artist)
    except Exception as error:  # always produce a PDF
        return _error_pdf(run_dir, report, 'sheet', 5, error, title, artist)


def run(source, run_dir, fps, until, recrop=False, accept_guess=False, title=None, remask=False, artist=None,
        start=None, end=None, layout=None, cookies=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'debug').mkdir(exist_ok=True)
    report = Report(run_dir)
    from .replays import choose_layout
    layout = choose_layout(run_dir, layout)      # a flag is saved; without one the saved choice is reused
    try:
        _stage('intake')
        _stage('intake')
        from . import intake
        section = intake.video_range(run_dir, start, end, source)
        fresh = not any(p for p in run_dir.glob('video.*') if p.suffix not in ('.part', '.ytdl'))
        video = intake.get_video(source, run_dir, report, section, cookies=cookies)
        if fresh:                    # play along's copy with sound (web app); never stops or changes the run
            try:
                intake.make_play_copy(source, run_dir, section, cookies=cookies)
            except Exception as error:
                print(f'No copy with sound for play along ({error}); the web app will try again when needed.')
        frames = intake.extract_frames(video, run_dir, fps, report)
        intake.contact_sheet(frames, fps, run_dir / 'debug' / '1-frames.jpg',
                             offset=section['start'] if section else 0)
        if until == 'intake':
            return report
        _stage('crop')
        _stage('crop')
        from . import crop
        box = crop.choose_crop(frames, run_dir, report, recrop=recrop, accept_guess=accept_guess)
        masks = crop.choose_masks(frames, box, run_dir, report, remask=remask or recrop,
                                  accept_guess=accept_guess)
        inverted = crop.decide_polarity(frames, box, run_dir, report)
        crop.crop_debug(frames, box, run_dir)
        if until == 'crop':
            return report
        try:  # always produce a PDF: a failure in stage 3 or 4 leaves the error page
            stage = ('rows', 3)
            _stage('rows')
            _stage('rows')
            from .rows import build_rows
            build_rows(frames, box, fps, run_dir, report, masks, inverted)
            if until == 'rows':
                return report
            stage = ('measures', 4)
            _stage('measures')
            _stage('measures')
            from .measures import build_measures
            build_measures(run_dir, report, layout)
        except Exception as error:
            pdf = _error_pdf(run_dir, report, *stage, error, title, artist)
            print(f'PDF: {pdf}')
            return report
        if until == 'measures':
            return report
        _stage('sheet')
        _stage('sheet')
        pdf = _sheet(run_dir, report, title, artist)
        print(f'PDF: {pdf}')
    finally:
        report.write()
        print(f'Report: {run_dir / "report.md"}')


BROWSERS = ('chrome', 'safari', 'firefox', 'edge', 'brave', 'chromium', 'opera', 'vivaldi')   # = intake.BROWSERS


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'build':
        from .vendor.build_cli import main as build
        return build(argv[1:])
    parser = argparse.ArgumentParser(prog='python -m tabvid', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='YouTube URL or local video file')
    parser.add_argument('-o', '--out', type=Path, required=True, help='run folder (created or reused)')
    parser.add_argument('--fps', type=float, default=2, help='frames sampled per second (default 2)')
    parser.add_argument('--until', choices=STAGES, default=STAGES[-1], help='stop after this stage')
    parser.add_argument('--recrop', action='store_true', help='ask for the TAB region again')
    parser.add_argument('--remask', action='store_true', help='draw the mask boxes (watermarks, overlays) again')
    parser.add_argument('--accept-guess', action='store_true', help='use the guessed TAB region without a window')
    parser.add_argument('--title', help='title printed at the top of page 1; the PDF is named after it '
                        '(default: the title saved for this run, else the run folder name)')
    parser.add_argument('--artist', help='printed under the title, smaller')
    parser.add_argument('--start', help='process the video from this time on (12:30, 1:02:30 or seconds); '
                        'saved for re-runs; --start 0 alone goes back to the whole video')
    parser.add_argument('--end', help='process the video up to this time (default: its end); saved for re-runs')
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument('--as-played', dest='layout', action='store_const', const='as played',
                        help='every page as the video shows it, replays included (the default; saved for re-runs)')
    layout.add_argument('--as-written', dest='layout', action='store_const', const='as written',
                        help='a page the video shows again (same image, same printed measure numbers: a repeat, '
                        'D.C. or D.S.) is printed once, like the score; saved for re-runs')
    parser.add_argument('--cookies-from-browser', metavar='BROWSER', dest='cookies',
                        choices=BROWSERS,
                        help="downloads use this browser's YouTube sign-in (its cookies, read on this computer and "
                        "sent only to YouTube): gets past \"Sign in to confirm you're not a bot\". Not saved")
    parser.set_defaults(layout=None)
    args = parser.parse_args(argv)
    for value in (args.start, args.end):
        if value is not None:
            try:
                from .intake import parse_time
                parse_time(value)
            except ValueError as error:
                parser.error(str(error))
    run(args.source, args.out, args.fps, args.until, args.recrop, args.accept_guess, args.title, args.remask,
        args.artist, args.start, args.end, args.layout, args.cookies)
    return 0


if __name__ == '__main__':
    sys.exit(main())
