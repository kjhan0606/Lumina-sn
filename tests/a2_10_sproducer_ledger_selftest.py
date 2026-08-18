#!/usr/bin/env python3
"""DET-SPROD 판정기 자기검사 — 양성 1 + 주입 결함 음성대조 6.

게이트는 주입 결함으로 FAIL 을 시연해야 PASS 자격이 있다(프로젝트 상설 규약).
"""
import importlib.util, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'sprod', ROOT / 'scripts' / 'a210_sproducer_ledger_check.py')
sprod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sprod)

TE = 10020.0


def row(nu=1.5e15, beta=0.25, jcont=3.0e-6, s_line=2.0e-5, **over):
    """생산식과 정확히 일치하는 정상 행 하나를 만든다."""
    omb = 1.0 - beta
    pct = beta * jcont
    plet = omb * s_line
    data = {
        'phase': 'REQUESTED_TE', 'rank': over.pop('rank', '1'),
        'line': over.pop('line', '100'),
        'nu': repr(nu), 'beta': repr(beta), 'one_minus_beta': repr(omb),
        'J_cont': repr(jcont),
        'producer_continuum_term': repr(pct),
        'producer_local_emission_term': repr(plet),
        'Jbar': repr(pct + plet),
        'producer_terms_defined': '1',
    }
    data.update({k: str(v) for k, v in over.items()})
    return data


def run(rows, te=TE):
    return sprod.check(rows, te)


def main():
    failures = []

    # --- 양성 ---
    base = [row(line='100', rank='1'), row(line='101', rank='2', beta=0.9)]
    ok = run(base)
    if ok['verdict'] != 'PASS':
        failures.append(f"positive: verdict={ok['verdict']} {ok.get('S2')}")
    if ok['S3']['evaluated_rows'] != 2:
        failures.append(f"positive: S3 rows={ok['S3']['evaluated_rows']}")

    # --- N1 S2 위반: Jbar 를 상대 1e-10 만큼 흔든다 ---
    bad = row()
    bad['Jbar'] = repr(float(bad['Jbar']) * (1.0 + 1e-10))
    r = run([bad])
    if r['verdict'] != 'FAIL' or r['S2']['pass']:
        failures.append('N1: perturbed Jbar not caught')

    # --- N2 producer_terms_defined=0 ⟹ 판정 불가 ---
    r = run([row(), row(line='102', rank='2', producer_terms_defined='0')])
    if r['verdict'] != 'UNDECIDABLE_PRODUCER_TERMS_MISSING':
        failures.append(f"N2: verdict={r['verdict']}")

    # --- N3 S2b 위반: 연속체 항이 beta*J_cont 와 어긋난다(합은 유지) ---
    bad = row()
    pct = float(bad['producer_continuum_term']) * 1.001
    plet = float(bad['Jbar']) - pct          # 합=Jbar 유지 ⟹ S2 는 통과해야
    bad['producer_continuum_term'] = repr(pct)
    bad['producer_local_emission_term'] = repr(plet)
    r = run([bad])
    if r['S2b']['pass']:
        failures.append('N3: S2b mismatch not caught')
    if not r['S2']['pass']:
        failures.append('N3: S2 must stay PASS (sum preserved)')

    # --- N4 합=0 인데 Jbar!=0 ---
    bad = row()
    bad['producer_continuum_term'] = '0.0'
    bad['producer_local_emission_term'] = '0.0'
    bad['Jbar'] = '1e-30'
    r = run([bad])
    if r['S2']['pass'] or r['S2']['zero_sum_violations'] != 1:
        failures.append('N4: zero-sum violation not caught')

    # --- N5 비유한/파손 필드 ---
    bad = row()
    bad['producer_local_emission_term'] = 'nan'
    r = run([bad])
    if r['S2']['pass'] or r.get('malformed_rows') != 1:
        failures.append('N5: malformed row not caught')

    # --- N6 행 0개 ---
    r = run([])
    if r['verdict'] != 'UNDECIDABLE_NO_ROWS':
        failures.append(f"N6: verdict={r['verdict']}")

    if failures:
        for f in failures:
            print('FAIL', f)
        return 1
    print('PASS a2_10_sproducer_ledger positive=1 negative_controls=6 '
          'floor=0 cap=0 clamp=0 jitter=0 repair=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
