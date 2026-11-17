"""Screenshot PDF/folder -> cleaned sheet. Build-only copy of the original tabsheet/cli.py."""
import argparse
import os
from pathlib import Path

from .config import validate_layout
from .pipeline import process


def main(argv):
    ap = argparse.ArgumentParser(prog="tabvid build", description="Clean screenshot PDFs into printable music sheets.")
    ap.add_argument("inputs", nargs="+", help="input PDF(s), image(s), folders or ZIP archives")
    ap.add_argument("-o", "--out", help="output file (single input) or directory")
    ap.add_argument("--title", help="title printed on page 1 (default: text from the PDF)")
    ap.add_argument("--page", default="letter", choices=["letter", "a4"])
    ap.add_argument("--margin", type=float, default=0.4, help="page margin in inches")
    ap.add_argument("--gap", type=float, default=4.0, help="gap between strips in points")
    ap.add_argument("--rows", type=int, help="target rows per page; shrinks scale to fit")
    ap.add_argument("--pages", help="only use these input pages, e.g. 1-5 or 1,3-4")
    ap.add_argument("--split", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--no-decolor", action="store_true", help="keep colored overlays")
    ap.add_argument("--debug-dir", help="save every cleaned strip here")
    ap.add_argument("--order", default=None, choices=["time", "name", "manifest"])
    ap.add_argument("--credit", help="credit printed in the PDF footer")
    ap.add_argument("--source", help="source attribution printed in the PDF footer")
    args = ap.parse_args(argv)
    args._explicit = {a.split("=", 1)[0].lstrip("-").replace("-", "_") for a in argv if a.startswith("--")}
    try:
        validate_layout(args)
    except (ValueError, TypeError) as exc:
        ap.error(str(exc))

    planned = []
    for p in args.inputs:
        stem = os.path.splitext(os.path.basename(p))[0] + "_sheet.pdf"
        if args.out and len(args.inputs) == 1 and not os.path.isdir(args.out) and args.out.lower().endswith(".pdf"):
            out = args.out
        else:
            out = os.path.join(args.out or os.path.dirname(os.path.abspath(p)), stem)
        planned.append((p, out))
    sources = {Path(p).resolve() for p in args.inputs}
    outputs = [Path(out).resolve() for _, out in planned]
    if any(path in sources for path in outputs):
        ap.error("output would overwrite an input file")
    if len(set(outputs)) != len(outputs):
        ap.error("multiple inputs resolve to the same output file")
    for p, out in planned:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            process(p, out, args)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            ap.error(str(exc))
