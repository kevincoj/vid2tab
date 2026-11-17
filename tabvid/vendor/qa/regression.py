"""Compare every PDF page and embedded row geometry, independently of metadata."""
from pathlib import Path
import numpy as np
import pymupdf


def compare_pdfs(reference, candidate, dpi=120, max_error=0.0001):
    """MAE on a 0..1 scale; 0.0001 permits rounding, not missing musical content."""
    for path in (reference, candidate):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    with pymupdf.open(reference) as ref, pymupdf.open(candidate) as new:
        result = {'reference_pages': len(ref), 'candidate_pages': len(new),
                  'passed': len(ref) == len(new), 'pages': []}
        for index in range(min(len(ref), len(new))):
            rp, np_ = ref[index], new[index]
            geometry_equal = rp.rect == np_.rect
            a, b = [page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
                    for page in (rp, np_)]
            same_shape = (a.width, a.height) == (b.width, b.height)
            error = None
            if same_shape:
                av = np.frombuffer(a.samples, dtype=np.uint8).astype(np.int16)
                bv = np.frombuffer(b.samples, dtype=np.uint8).astype(np.int16)
                error = float(np.abs(av - bv).mean() / 255)
            rrows, nrows = rp.get_image_info(), np_.get_image_info()
            row_geometry_equal = len(rrows) == len(nrows) and all(
                np.allclose(r['bbox'], n['bbox'], atol=0.01, rtol=0)
                for r, n in zip(rrows, nrows))
            passed = geometry_equal and same_shape and error <= max_error and row_geometry_equal
            result['pages'].append({'page': index + 1, 'normalized_error': error,
                                    'reference_rows': len(rrows), 'candidate_rows': len(nrows),
                                    'row_geometry_equal': row_geometry_equal, 'passed': bool(passed)})
            result['passed'] = result['passed'] and bool(passed)
        return result
