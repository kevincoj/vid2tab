from ..config import CFG
import io
import numpy as np
import pymupdf
from PIL import Image

def extract_strips(pdf_path, min_w=CFG.min_image_width, min_h=CFG.min_image_height, pages=None):
    """Return (strips, title). strips = list of RGB uint8 arrays in reading order."""
    doc = pymupdf.open(pdf_path)
    strips = []
    title = None
    for pno, page in enumerate(doc):
        if pno == 0:
            text = page.get_text().strip().splitlines()
            if text:
                title = text[0].strip()
        if pages and (pno + 1) not in pages:
            continue
        items = []
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref")
            if not xref:
                continue
            x0, y0, x1, y1 = info["bbox"]
            try:
                data = doc.extract_image(xref)["image"]
                img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            except Exception as e:  # unreadable image object
                print(f"  ! page {pno + 1}: could not read image {xref}: {e}")
                continue
            h, w = img.shape[:2]
            if w < min_w or h < min_h:
                continue  # icons, bullets, tiny logos
            items.append((round(y0 / 4), x0, img))
        items.sort(key=lambda t: (t[0], t[1]))
        strips.extend(t[2] for t in items)
    return strips, title


# All input front-ends yield RGB arrays before the unchanged cleaning pipeline.
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import zipfile
from .ordering import ordered_names
from .manifest import read_manifest, apply_order

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp'}


@dataclass
class InputImages:
    images: list
    title: str | None = None
    names: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _eligible(name):
    path = PurePosixPath(name)
    return path.suffix.lower() in IMAGE_SUFFIXES and not any(p.startswith('.') or p == '__MACOSX'
                                                            for p in path.parts)


def _rgb(data):
    with Image.open(data) as image:
        # Honor phone-camera orientation without affecting original PDF extraction.
        from PIL import ImageOps
        return np.array(ImageOps.exif_transpose(image).convert('RGB'))


def load_input(path, pages=None, order=None):
    path = Path(path)
    if not path.exists():
        raise ValueError(f'input does not exist: {path}')
    if path.suffix.lower() == '.pdf' and path.is_file():
        images, title = extract_strips(str(path), pages=pages)
        return InputImages(images, title, [f'image_{i:03d}' for i in range(len(images))])
    if pages:
        raise ValueError('--pages applies only to PDF input')
    if path.is_dir():
        files = {p.relative_to(path).as_posix(): p for p in path.rglob('*') if p.is_file()
                 and _eligible(p.relative_to(path).as_posix())}
        manifest = path / 'tabsheet.yaml'
        metadata = read_manifest(manifest.read_text()) if manifest.exists() else {}
        names = apply_order(ordered_names(files, order or "time", {n: p.stat().st_mtime for n, p in files.items()}),
                            metadata, order)
        return InputImages([_rgb(files[n]) for n in names], metadata.get('title'), names, metadata)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            members = {}
            for info in archive.infolist():
                name = info.filename
                if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name:
                    raise ValueError(f'unsafe ZIP path: {name}')
                if not info.is_dir() and ( _eligible(name) or PurePosixPath(name).name == 'tabsheet.yaml'):
                    if name in members:
                        raise ValueError(f'duplicate ZIP path: {name}')
                    members[name] = info
            roots = {PurePosixPath(n).parts[0] for n in members}
            prefix = next(iter(roots)) + '/' if len(roots) == 1 and all('/' in n for n in members) else ''
            files = {n[len(prefix):]: info for n, info in members.items() if _eligible(n)}
            manifest = members.get(prefix + 'tabsheet.yaml')
            metadata = read_manifest(archive.read(manifest).decode('utf-8')) if manifest else {}
            names = apply_order(ordered_names(files, order or "time"), metadata, order)
            return InputImages([_rgb(io.BytesIO(archive.read(files[n]))) for n in names],
                               metadata.get('title'), names, metadata)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        if order == 'manifest':
            raise ValueError('--order manifest requires a folder or ZIP manifest')
        return InputImages([_rgb(path)], names=[path.name])
    raise ValueError(f'unsupported input: {path}; use PDF, folder, ZIP, PNG, JPEG or WEBP')
