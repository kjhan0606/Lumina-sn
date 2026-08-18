#!/usr/bin/env python3
"""DET-SPROD S2/S2b/S3 판정기 (오프라인, fail-closed).

사전등록 docs/RUNG_SPRODUCER_CAPTURE_2026-08-18.md 의 게이트를 그대로 집행한다.
  S2  : Jbar == producer_continuum_term + producer_local_emission_term
        (상대오차 <= 4e-16 = 덧셈 1회 2 ulp).  합이 0 인 행은 Jbar == 0 정확 항등.
  S2b : producer_continuum_term == beta * J_cont  (상대오차 <= 1e-12).
        beta*J_cont == 0 인 행은 producer_continuum_term == 0 정확 항등.
  S3  : [local_emission_term/(1-beta)] / B(T_e) 분포.  (1-beta)==0 행 제외·수 명기.

판정 불가 조건(fail-closed): producer_terms_defined=0 행이 1개라도 있으면 S2 UNDECIDABLE.
"""
import argparse, json, math, re, sys
from pathlib import Path

H = 6.62607015e-27
K_B = 1.380649e-16
C_LIGHT = 2.99792458e10
S2_TOL = 4e-16
S2B_TOL = 1e-12
FIELD = re.compile(r'(\S+)=(\S+)')
MARKER = '[A2-10][LINE-SATURATION-ROW]'


def planck(nu, te):
    x = H * nu / (K_B * te)
    return 2.0 * H * nu ** 3 / C_LIGHT ** 2 / math.expm1(x)


def parse_rows(path):
    rows, seen = [], set()
    with open(path, 'r', errors='replace') as handle:
        for line in handle:
            if MARKER not in line:
                continue
            row = dict(FIELD.findall(line))
            key = (row.get('phase'), row.get('rank'), row.get('line'))
            if key in seen:      # UNION-META 등 중복 방출 방어
                continue
            seen.add(key)
            rows.append(row)
    return rows


def num(row, name):
    text = row.get(name)
    if text is None or text == 'UNAVAILABLE':
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def check(rows, te_k):
    out = {'row_count': len(rows)}
    if not rows:
        out['verdict'] = 'UNDECIDABLE_NO_ROWS'
        return out

    undefined = [r for r in rows if r.get('producer_terms_defined') != '1']
    out['producer_terms_undefined_rows'] = len(undefined)
    if undefined:
        out['verdict'] = 'UNDECIDABLE_PRODUCER_TERMS_MISSING'
        out['first_undefined_line'] = undefined[0].get('line')
        return out

    s2_worst, s2_worst_line = 0.0, None
    s2_zero_sum, s2_zero_sum_violation = 0, 0
    s2b_worst, s2b_worst_line = 0.0, None
    s2b_zero, s2b_zero_violation, s2b_missing = 0, 0, 0
    s3_ratios, s3_excluded_beta1, s3_excluded_other = [], 0, 0
    malformed = 0

    for row in rows:
        jbar = num(row, 'Jbar')
        pct = num(row, 'producer_continuum_term')
        plet = num(row, 'producer_local_emission_term')
        beta = num(row, 'beta')
        omb = num(row, 'one_minus_beta')
        jcont = num(row, 'J_cont')
        nu = num(row, 'nu')
        if None in (jbar, pct, plet, beta, omb, nu):
            malformed += 1
            continue

        # --- S2 ---
        total = pct + plet
        if total == 0.0:
            s2_zero_sum += 1
            if jbar != 0.0:
                s2_zero_sum_violation += 1
        else:
            rel = abs(jbar / total - 1.0)
            if rel > s2_worst:
                s2_worst, s2_worst_line = rel, row.get('line')

        # --- S2b ---
        if jcont is None:
            s2b_missing += 1
        else:
            reference = beta * jcont
            if reference == 0.0:
                s2b_zero += 1
                if pct != 0.0:
                    s2b_zero_violation += 1
            else:
                rel = abs(pct / reference - 1.0)
                if rel > s2b_worst:
                    s2b_worst, s2b_worst_line = rel, row.get('line')

        # --- S3 ---
        if omb == 0.0:
            s3_excluded_beta1 += 1
        else:
            bnu = planck(nu, te_k)
            if bnu > 0.0 and math.isfinite(bnu):
                s3_ratios.append((plet / omb) / bnu)
            else:
                s3_excluded_other += 1

    out['malformed_rows'] = malformed
    out['S2'] = {
        'max_relative_deviation': s2_worst,
        'worst_line': s2_worst_line,
        'tolerance': S2_TOL,
        'zero_sum_rows': s2_zero_sum,
        'zero_sum_violations': s2_zero_sum_violation,
        'pass': (malformed == 0 and s2_worst <= S2_TOL
                 and s2_zero_sum_violation == 0),
    }
    out['S2b'] = {
        'max_relative_deviation': s2b_worst,
        'worst_line': s2b_worst_line,
        'tolerance': S2B_TOL,
        'zero_reference_rows': s2b_zero,
        'zero_reference_violations': s2b_zero_violation,
        'missing_j_cont_rows': s2b_missing,
        'pass': (s2b_worst <= S2B_TOL and s2b_zero_violation == 0
                 and s2b_missing == 0),
    }
    s3 = sorted(s3_ratios)
    if s3:
        n = len(s3)
        out['S3'] = {
            'te_k': te_k,
            'evaluated_rows': n,
            'excluded_one_minus_beta_zero': s3_excluded_beta1,
            'excluded_nonfinite_planck': s3_excluded_other,
            'q10': s3[int(0.10 * n)],
            'median': s3[n // 2],
            'q90': s3[min(n - 1, int(0.90 * n))],
            'min': s3[0],
            'max': s3[-1],
        }
    else:
        out['S3'] = {'te_k': te_k, 'evaluated_rows': 0,
                     'excluded_one_minus_beta_zero': s3_excluded_beta1,
                     'excluded_nonfinite_planck': s3_excluded_other}
    out['verdict'] = ('PASS' if out['S2']['pass'] and out['S2b']['pass']
                      else 'FAIL')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stderr', type=Path)
    ap.add_argument('--te-k', type=float, required=True,
                    help='생산자 상태의 shell-0 T_e [K] (S3 분모)')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    rows = parse_rows(args.stderr)
    report = check(rows, args.te_k)
    report['schema'] = 'lumina-a210-sproducer-ledger-v1'
    report['input'] = str(args.stderr)
    report['physical_values_modified'] = False
    report['floor'] = report['cap'] = report['clamp'] = 0
    report['jitter'] = report['repair'] = 0
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in
                      ('schema', 'row_count', 'verdict')}, sort_keys=True))
    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
