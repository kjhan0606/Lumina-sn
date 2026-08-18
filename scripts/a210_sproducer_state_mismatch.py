#!/usr/bin/env python3
"""DET-SPROD 실측: 생산자 자신의 beta·S 를 장부에서 역산하고 소비자 상태와 대조.

S2(덧셈 항등)가 정확히 0 이므로 Jbar = pct + plet 이 bit 로 성립한다.
따라서 beta_producer = pct / J_cont, S_producer = plet / (1 - beta_producer) 로
**생산자가 실제로 쓴 값**을 복원할 수 있다.  이것이 V2 판정의 직접 증거다.
"""
import argparse, json, math, re, statistics
from pathlib import Path
H, K_B, C = 6.62607015e-27, 1.380649e-16, 2.99792458e10
FIELD = re.compile(r'(\S+)=(\S+)')
MARKER = '[A2-10][LINE-SATURATION-ROW]'


def planck(nu, t):
    x = H * nu / (K_B * t)
    return 2.0 * H * nu ** 3 / C ** 2 / math.expm1(x) if x < 700.0 else 0.0


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stderr', type=Path)
    ap.add_argument('--te-k', type=float, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    rows = [dict(FIELD.findall(l)) for l in open(a.stderr, errors='replace')
            if MARKER in l]
    bp, sb, tc, skipped = [], [], [], 0
    for r in rows:
        try:
            jc = float(r['J_cont']); pct = float(r['producer_continuum_term'])
            plet = float(r['producer_local_emission_term'])
            nu = float(r['nu']); tau_c = float(r['tau_effective'])
        except (KeyError, ValueError):
            skipped += 1; continue
        if not (jc > 0.0):
            skipped += 1; continue
        beta = pct / jc
        if not (0.0 < beta < 1.0):
            skipped += 1; continue
        b_nu = planck(nu, a.te_k)
        if not (b_nu > 0.0):
            skipped += 1; continue
        bp.append(beta); sb.append((plet / (1.0 - beta)) / b_nu); tc.append(tau_c)
    rep = {
        'schema': 'lumina-a210-sproducer-state-mismatch-v1',
        'input': str(a.stderr), 'te_k': a.te_k,
        'rows': len(rows), 'evaluated': len(bp), 'skipped': skipped,
        'beta_producer': {'q10': q(bp, .1), 'median': q(bp, .5), 'q90': q(bp, .9)},
        'tau_consumer': {'q10': q(tc, .1), 'median': q(tc, .5), 'q90': q(tc, .9)},
        'S_producer_over_B_Te': {
            'min': min(sb), 'q10': q(sb, .1), 'median': q(sb, .5),
            'q90': q(sb, .9), 'max': max(sb),
            'stdev_log10': statistics.pstdev([math.log10(x) for x in sb if x > 0]),
            'rows_within_1pct_of_unity': sum(1 for x in sb if abs(x - 1.0) < 0.01)},
        'physical_values_modified': False,
        'floor': 0, 'cap': 0, 'clamp': 0, 'jitter': 0, 'repair': 0}
    a.output.write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')
    s = rep['S_producer_over_B_Te']
    print(f"evaluated={rep['evaluated']} skipped={skipped}  "
          f"S_prod/B: median={s['median']:.6f} stdev_log10={s['stdev_log10']:.2e} "
          f"within1pct={s['rows_within_1pct_of_unity']}/{rep['evaluated']}")


if __name__ == '__main__':
    main()
