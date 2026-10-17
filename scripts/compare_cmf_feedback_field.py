"""S39 인구 섭동에 대한 실제 새 장의 국소 율 응답. 전체 SE 미분은 아니다."""
import csv
import json
import math
import sys
from pathlib import Path


def slopes(x, y):
    lo, zero, hi = (math.log(x[k]) for k in ('minus', 'zero', 'plus'))
    a, b, c = (math.log(y[k]) for k in ('minus', 'zero', 'plus'))
    if not lo < zero < hi:
        raise ValueError('섭동 방향 오류')
    return dict(central=(c-a)/(hi-lo), minus=(b-a)/(zero-lo), plus=(c-b)/(hi-zero))


def selftest():
    x = dict(minus=.0099, zero=.01, plus=.0101)
    result = slopes(x, {k: 2*v**.8 for k, v in x.items()})
    if any(abs(v-.8) > 1e-12 for v in result.values()):
        raise ValueError('알려진 로그 응답 재현 실패')
    print('FEEDBACK_ANALYSIS_SELFTEST_PASS analytic_power=.8')


def run(root):
    inputs, tables = {}, {}
    for mode in ('minus', 'zero', 'plus'):
        branch = root / mode
        inputs[mode] = json.loads((branch/'perturbation.json').read_text())
        tables[mode] = {}
        for field in ('stored', 'new'):
            with (branch/(field+'_field_bf.csv')).open() as f:
                tables[mode][field] = {(int(r['shell']), int(r['Z'])): r for r in csv.DictReader(f)}
        if tables[mode]['stored'].keys() != tables[mode]['new'].keys():
            raise ValueError('율 표 범위 불일치')
    keys = tables['zero']['new'].keys()
    if any(tables[k]['new'].keys() != keys for k in tables):
        raise ValueError('세 대조의 셸/원소 누락')
    x = {k: inputs[k]['new_ratio'] for k in inputs}
    rows = []
    for key in sorted(keys):
        for field in ('stored', 'new'):
            y = {}
            for mode in inputs:
                r = tables[mode][field][key]
                y[mode] = (float(r['photo_per_II_s'])+float(r['collision_per_II_s'])) / (
                    float(r['recomb_per_III_s'])+float(r['threebody_per_III_s']))
                if not y[mode] > 0 or not math.isfinite(y[mode]):
                    raise ValueError('비정상 열적 율비')
            rows.append(dict(shell=key[0], Z=key[1], field=field, thermal_ratio=y,
                dlogK_dlogS39ratio=slopes(x, y)))
    result = dict(status='LOCAL_CHARGE_PRESERVING_MATERIAL_TO_RADIATION_RATE_RESPONSE',
        goal_verdict='NOT_JUDGED', physical_history=False, full_SE_updates=0,
        full_coupled_Jacobian=False, nonthermal_included=False, perturbations=inputs, rows=rows)
    with (root/'feedback.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False);f.write('\n')
    print(json.dumps([r for r in rows if r['shell']==39], allow_nan=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    elif len(sys.argv) == 2:
        run(Path(sys.argv[1]))
    else:
        raise SystemExit(2)
