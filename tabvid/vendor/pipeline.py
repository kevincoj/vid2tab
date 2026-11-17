from .config import CFG
import os
from PIL import Image
from reportlab.lib.pagesizes import A4, letter
from .io.sources import load_input
from .preprocess.cleaning import clean
from .layout.systems import split_systems, crop_to_staff
from .layout.staves import staff_spacing
from .render.pdf import build_pdf

def parse_pages(spec):
    if not spec:
        return None
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            pages.update(range(int(a), int(b) + 1))
        elif part:
            pages.add(int(part))
    return pages


def process(pdf_path, out_path, args):
    source = load_input(pdf_path, pages=parse_pages(args.pages), order=getattr(args, "order", "time"))
    strips, title = source.images, source.title
    import copy
    from .config import validate_layout
    args = copy.copy(args)
    for key in ("page", "margin", "gap", "rows", "split", "title", "credit", "source"):
        if key in source.metadata and key not in getattr(args, "_explicit", set()):
            setattr(args, key, source.metadata[key])
    validate_layout(args)
    if not strips:
        raise ValueError(f"no screenshot images found in {pdf_path}")
    cleaned, skipped, split_imgs = [], 0, 0
    for i, s in enumerate(strips):
        img = clean(s, decolor=not args.no_decolor)
        h, w = img.shape
        tall = (h / w > CFG.whole_page_aspect) if args.split == "auto" else (args.split == "always")
        if tall:
            pieces = split_systems(img, first_on_sheet=(len(cleaned) == 0))
            split_imgs += 1
        else:
            pieces = [img]
        for j, piece in enumerate(pieces):
            piece = crop_to_staff(piece)
            if piece is None or piece.shape[0] < CFG.minimum_row_height:
                skipped += 1
                continue
            cleaned.append((piece, staff_spacing(piece)))
            if args.debug_dir:
                os.makedirs(args.debug_dir, exist_ok=True)
                Image.fromarray(piece).save(os.path.join(args.debug_dir, f"strip_{i:03d}_{j}.png"))

    if not cleaned:
        raise ValueError(f"no music systems found in {pdf_path}")

    page_size = A4 if args.page.lower() == "a4" else letter
    title = args.title or title or os.path.splitext(os.path.basename(pdf_path))[0]
    pages, unit = build_pdf(cleaned, out_path, title, page_size,
                            args.margin * 72, args.gap, rows=args.rows,
                            credit=getattr(args, "credit", None), source=getattr(args, "source", None))
    no_sp = sum(1 for _, sp in cleaned if not sp)
    print(f"{os.path.basename(pdf_path)}: {len(strips)} images"
          f"{f' ({split_imgs} split into rows)' if split_imgs else ''} -> {len(cleaned)} rows"
          f"{f', {skipped} skipped' if skipped else ''}"
          f"{f', {no_sp} without staff-spacing' if no_sp else ''}"
          f" -> {pages} page(s)  => {out_path}")
