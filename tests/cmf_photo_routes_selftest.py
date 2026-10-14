"""실제 Si/S 원자료의 경로/문턱을 원본 Fortran 실행과 대조한다."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cmfgen_parser import parse_osc, parse_f_to_s, parse_phot
from cmfgen_photo_routes import (read_photo_tables, resolve_photo_routes,
                                level_photo_channels, PhotoTable, CM_TO_EV)


def rejected(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('invalid input was accepted')


def reference(executable, tables, upper, sl):
    with tempfile.TemporaryDirectory(prefix='route-reference-', dir='.') as tmp:
        root = Path(tmp).resolve()
        for i, table in enumerate(tables):
            (root / f'PHOTTEST_{chr(65+i)}').symlink_to(table.path.resolve())
        if upper is None:
            text = '1 0\nparent 1 0 1\n'
        else:
            text = f'{len(upper.levels)} 1\n'
            for i, level in enumerate(upper.levels):
                text += (f"{level['config']} {level['g']:.17g} "
                         f"{-level['E_cm'] * 2.99792458e-5:.17g} {int(sl[i])+1}\n")
        result = subprocess.run([str(executable)], input=text, text=True,
                                cwd=root, capture_output=True, check=True, timeout=20)
        rows = [line.split()[1:] for line in result.stdout.splitlines() if line.startswith('ROUTE')]
        assert rows, result.stdout + result.stderr
        return [[float(value) for value in row] for row in rows]


def check_case(atomic, executable, element, lower_stage, upper_stage, ftos, n_kept):
    lower_dir = atomic / element / lower_stage
    upper_dir = atomic / element / upper_stage
    tables = read_photo_tables(lower_dir / 'phot_data_A')
    lower = parse_osc(lower_dir / 'osc_data')
    upper = parse_osc(upper_dir / 'osc_data')
    sl = parse_f_to_s(upper_dir / ftos).sl_of_fl
    routes = resolve_photo_routes(tables, upper, sl)
    expected = reference(executable, tables, upper, sl)
    assert len(routes) == len(expected)
    for route, ref in zip(routes, expected):
        assert route.table + 1 == ref[0] and route.target_level + 1 == ref[1]
        np.testing.assert_allclose([route.excitation_cm1 * 2.99792458e-5,
                                   route.statistical_weight, route.scale], ref[3:], rtol=3e-13, atol=1e-15)
    channels = level_photo_channels(lower, tables, routes, n_kept)
    assert len(channels) == n_kept * len(routes)
    for table in range(len(tables)):
        assert abs(sum(r.scale for r in routes if r.table == table) - 1) < 1e-14
    rejected(lambda: resolve_photo_routes((PhotoTable(tables[0].path,
        replace(tables[0].data, final_state='no_such_state')),), upper, sl))
    rejected(lambda: resolve_photo_routes(tables, upper, sl[:-1]))
    # 목적 에너지 헤더를 바꿔도 active upper의 osc 에너지가 정본이다.
    changed = tuple(PhotoTable(t.path, replace(t.data, final_state_excit_eV=999)) for t in tables)
    assert resolve_photo_routes(changed, upper, sl) == routes
    if element == 'SIL':
        assert len(tables) == 2 and len(routes) == 2
        assert routes[1].members == (1,2,3)
        assert routes[1].statistical_weight == 9 and routes[1].scale == 1
        assert abs(routes[1].excitation_cm1 - 52724.69) < 1e-8
        # A 경로에서 명시적으로 0인 quartet도 B에는 실제 단면적이 있다.
        aa = next(c for c in channels if c.lower_level == 2 and c.route.table == 0)
        bb = next(c for c in channels if c.lower_level == 2 and c.route.table == 1)
        assert np.all(aa.entry.sigma_Mb == 0)
        assert np.any(bb.entry.sigma_Mb > 0) and bb.entry.cs_type == 20
        assert bb.threshold_eV > aa.threshold_eV + 6
        print(json.dumps({'SiII_quartet_level': 2, 'A_threshold_eV': aa.threshold_eV,
            'B_threshold_eV': bb.threshold_eV, 'A_explicit_zero': True,
            'B_max_raw_Mb': float(bb.entry.sigma_Mb.max()),
            'B_parent_excitation_eV': routes[1].excitation_cm1 * CM_TO_EV}))
        # 같은 실제 term을 개별 SL로 푼 경우: 1/9,3/9,5/9 분기(합계 1).
        split = np.arange(len(upper.levels))
        split_routes = resolve_photo_routes(tables, upper, split)
        split_ref = reference(executable, tables, upper, split)
        assert len(split_routes) == len(split_ref) == 4
        np.testing.assert_allclose([r.scale for r in split_routes], [1,1/9,3/9,5/9], rtol=1e-14)
        for r, ref in zip(split_routes, split_ref):
            np.testing.assert_allclose([r.excitation_cm1*2.99792458e-5,r.statistical_weight,r.scale],
                                       ref[3:],rtol=3e-13,atol=1e-15)
        terminal = resolve_photo_routes(tables, None)
        terminal_ref = reference(executable, tables, None, None)
        for r, ref in zip(terminal, terminal_ref):
            assert r.target_level is None
            np.testing.assert_allclose([r.excitation_cm1*2.99792458e-5,r.statistical_weight,r.scale],
                                       ref[3:],rtol=3e-13,atol=1e-15)
        with tempfile.TemporaryDirectory(prefix='route-missing-',dir='.') as tmp:
            first=Path(tmp)/'phot_data_A';first.symlink_to(tables[0].path.resolve())
            rejected(lambda: read_photo_tables(first))
    print(json.dumps({'ion': element+lower_stage,'files':len(tables),'routes':len(routes),
        'active_lower_levels':n_kept,'channels':len(channels),'fortran_reference':'PASS',
        'upper_anchors_0based':[r.target_level for r in routes],
        'scales':[r.scale for r in routes]}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--atomic',type=Path,required=True)
    ap.add_argument('--oracle',type=Path,required=True);args=ap.parse_args()
    check_case(args.atomic,args.oracle.resolve(),'SIL','II','III','f_to_s_ls',125)
    check_case(args.atomic,args.oracle.resolve(),'SUL','II','III','f_to_s_127',322)
    # 문자열 헤더의 True 및 단위 보존을 실제 파일의 사본으로 검사한다.
    with tempfile.TemporaryDirectory(prefix='phot-header-',dir='.') as tmp:
        source=args.atomic/'SIL/II/phot_data_A'
        text=source.read_text();assert 'False' in text
        path=Path(tmp)/'phot';path.write_text(text.replace('False','True',1))
        assert parse_phot(path).split_J is True
        assert parse_phot(path).cs_unit == 'Megabarns'
    print('CMF_PHOTO_ROUTES_PASS physical_rate_and_spectrum_validation=not_yet')


if __name__=='__main__':main()
