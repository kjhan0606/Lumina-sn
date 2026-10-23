"""inward 조건부 각도·광흡수율 합산과 끝점 분기를 확인한다."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_cmfgen_siii_all_minus as C


def test_all_minus_uses_terminal_linear_and_photo_weights(tmp_path, monkeypatch):
    paths = [tmp_path/name for name in ('geometry', 'compact', 'original',
                                        'finished_F', 'rates')]
    for path in paths:
        path.write_bytes(path.name.encode())
    n = 2544
    rays = [dict(observed=np.full(n, .1), delta=np.full(n, .1),
                 far_neighbor=dict()),
            dict(observed=np.full(n, .2), delta=np.full(n, .4),
                 far_neighbor=None)]
    monkeypatch.setattr(C, 'collect_minus', lambda _: dict(
        freq=np.linspace(2.99792458e18/750, 2.99792458e18/600, n)/1e15,
        native_j=np.ones(n), rays=rays,
        angle_weights=np.array([.2, .3]),
        geometry=paths[0], compact=paths[1]))
    monkeypatch.setattr(C, 'production_weight', lambda *_:
                        (np.full(n, 1./n), 2., 1., paths[2], paths[3]))
    terminal_calls = []

    def sweep(ray, pieces, mode, terminal_linear):
        terminal_calls.append((pieces, terminal_linear))
        return ray['observed'].copy() if pieces == 1 else ray['observed']+ray['delta']

    monkeypatch.setattr(C, 'sc_sweep', sweep)
    out = tmp_path/'result'
    C.analyze(tmp_path, tmp_path, paths[4], out, subdivisions=2)
    report = json.loads((out/'report.json').read_text())
    np.testing.assert_allclose(np.load(out/'inward_delta_J.npy'), .14)
    assert terminal_calls == [(1, False), (2, False), (1, True), (2, True)]
    assert report['original_bezier_baseline_max_abs'] == 0.
    np.testing.assert_allclose(report['inward_delta_J_photo_cm3_s'], .14)
    np.testing.assert_allclose(report['conditional_F_minus_J_photo_cm3_s'], .86)
