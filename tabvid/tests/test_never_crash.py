"""Always a PDF: frames with a staff but no barline, and a stage that fails (fix plan 4, step 1)."""
import json
from types import SimpleNamespace

import cv2
import numpy as np

from tabvid.vendor.same_row import same_row


def _view(bars):
    rgb = np.full((120, 400, 3), 245, np.uint8)
    for i in range(6):
        rgb[10 + 20 * i, 20:380] = 60
    cv2.putText(rgb, '3', (100, 36), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
    return SimpleNamespace(rgb=rgb, bars=bars, lines=tuple(10 + 20 * i for i in range(6)))


def test_two_frames_without_barlines_are_compared_not_crashed():
    assert same_row(_view(()), _view(()))
    assert not same_row(_view(()), _view((200,)))


def test_a_failing_stage_3_still_writes_a_pdf(tmp_path, monkeypatch):
    from tabvid import __main__ as cli, rows
    source = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 2, (400, 120))
    for _ in range(4):
        writer.write(_view(()).rgb)
    writer.release()
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({'crop': [0, 0, 400, 120], 'crop_how': 'user-drawn', 'masks': [], 'inverted': False}))

    def boom(*args, **kwargs):
        raise RuntimeError('drawn failure')
    monkeypatch.setattr(rows, 'build_rows', boom)
    report = cli.run(str(source), run, 2, 'sheet', accept_guess=True)
    assert (run / 'run.pdf').exists()
    assert any('stage 3 failed' in w and 'drawn failure' in w for _, w in report.warnings)
