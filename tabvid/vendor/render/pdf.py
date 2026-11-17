import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def build_pdf(strips, out_path, title, page_size, margin_pt, gap_pt, title_pt=13, rows=None, credit=None, source=None,
              subtitle=None):  # tabvid edit: subtitle (the artist), printed smaller under the title
    """strips: list of (gray image, staff_spacing_px or None)."""
    pw, ph = page_size
    usable_w = pw - 2 * margin_pt
    footer = " | ".join(str(value) for value in (credit, source) if value)
    from reportlab.lib.utils import simpleSplit
    footer_lines = simpleSplit(footer, "Helvetica", 8, usable_w) if footer else []
    footer_reserve = len(footer_lines) * 10 + 6 if footer else 0
    if footer_reserve >= ph - 2 * margin_pt - 24:
        raise ValueError("credit/source footer leaves no room for music")
    bottom = margin_pt + footer_reserve
    usable_h = ph - 2 * margin_pt - footer_reserve

    spacings = [sp for _, sp in strips if sp]
    fallback = float(np.median(spacings)) if spacings else 1.0
    sp_list = [sp if sp else fallback for _, sp in strips]
    # size every strip in "staff-spacing units" so different zoom levels end up the same size
    w_units = [img.shape[1] / sp for (img, _), sp in zip(strips, sp_list)]
    h_units = [img.shape[0] / sp for (img, _), sp in zip(strips, sp_list)]
    unit = usable_w / max(w_units)  # pt per spacing unit (full page width for the widest strip)
    if rows:
        reserve = title_pt + 10 if title else 0
        unit_h = (usable_h - reserve - rows * gap_pt) / (rows * float(np.median(h_units)))
        unit = min(unit, unit_h)
    x_left = margin_pt + (usable_w - max(w_units) * unit) / 2  # center the block if narrower

    c = canvas.Canvas(out_path, pagesize=page_size)
    if title:
        c.setTitle(title)

    def draw_footer():
        if footer:
            c.setFont("Helvetica", 8)
            for index, line in enumerate(reversed(footer_lines)):
                c.drawString(margin_pt, margin_pt + 4 + index * 10, line)

    draw_footer()
    y = ph - margin_pt
    pages = 1
    if title:
        c.setFont("Helvetica-Bold", title_pt)
        c.drawString(margin_pt, y - title_pt, title)
        y -= title_pt + 10
        if subtitle:  # tabvid edit
            c.setFont("Helvetica", title_pt - 3)
            c.drawString(margin_pt, y - title_pt + 9, subtitle)
            y -= title_pt + 1

    for (img, _), wu, hu in zip(strips, w_units, h_units):
        w_pt, h_pt = wu * unit, hu * unit
        if y - h_pt < bottom and y < ph - margin_pt - 1:
            c.showPage()
            draw_footer()
            pages += 1
            y = ph - margin_pt
        target_w = int(w_pt / 72 * 300)  # ~300 dpi keeps the PDF small
        pil = Image.fromarray(img)
        if pil.width > target_w > 0:
            pil = pil.resize((target_w, max(1, int(pil.height * target_w / pil.width))),
                             Image.LANCZOS)
        c.drawImage(ImageReader(pil), x_left, y - h_pt, width=w_pt, height=h_pt)
        y -= h_pt + gap_pt
    c.save()
    return pages, unit
