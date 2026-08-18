#!/usr/bin/env python3
"""SH-GRID B-6 — 격자 값 이동 장부 (오프라인 측정).

사전등록 docs/RUNG_GRID_CONTAINMENT_CONTRACT.md 의 B-6:
  "격자가 바뀌므로 값이 움직인다. **어디가 얼마나** 움직였는지 기재."
실패 조건이 아니라 측정이며, 이 수치가 곧 현재 격자의 재빈닝 오차다.

세 격자(계보는 git 실측):
  G0  3ca077d (2026-08-08 이전)  균일 log, 4000 빈, 리터럴 선언
  G1  2e26c2f (안 B 착지)        BF(1000빈, 1.5e14~3.0e16) 의 K=2 세분, j=[-1754,2112)
  G2  f6c2eb6 (현행)             BF(1234빈, 5.841e13~4.036e16) 의 K=2 세분, j=[-1398,2468)
"""
import json, math
from pathlib import Path

H = 6.62607015e-27
K_B = 1.380649e-16
C = 2.99792458e10


def planck(nu, t):
    x = H * nu / (K_B * t)
    if x > 700.0:
        return 0.0
    return 2.0 * H * nu ** 3 / C ** 2 / math.expm1(x)


def uniform_log_edges(nu_min, nu_max, n):
    dl = math.log(nu_max / nu_min) / n
    return [nu_min * math.exp(i * dl) for i in range(n + 1)], dl


def derived_edges(bf_min, bf_max, bf_bins, k, j_lo, j_hi):
    dl = math.log(bf_max / bf_min) / (k * bf_bins)
    return [bf_min * math.exp((j_lo + i) * dl) for i in range(j_hi - j_lo + 1)], dl


GRIDS = {}
e, d = uniform_log_edges(1.4402928950097124e12, 4.032418413741097e16, 4000)
GRIDS['G0_pre_B_4000'] = {'edges': e, 'dlog': d, 'commit': '3ca077d'}
e, d = derived_edges(1.5e14, 3.0e16, 1000, 2, -1754, 2112)
GRIDS['G1_plan_B_3866'] = {'edges': e, 'dlog': d, 'commit': '2e26c2f'}
e, d = derived_edges(5.8412785919616062e13, 4.0362581455823112e16, 1234, 2, -1398, 2468)
GRIDS['G2_current_3866'] = {'edges': e, 'dlog': d, 'commit': 'f6c2eb6'}


def rebin(src_edges, src_vals, dst_edges):
    """구간 겹침 가중 보존 재빈닝.  src 의 빈 적분량을 dst 로 정확 이송한다.
    dst 밖으로 나간 몫은 별도로 돌려준다(손실 계량)."""
    dst = [0.0] * (len(dst_edges) - 1)
    lost = 0.0
    j = 0
    for i in range(len(src_edges) - 1):
        a, b = src_edges[i], src_edges[i + 1]
        amount = src_vals[i] * (b - a)          # 빈 적분량
        if amount == 0.0:
            continue
        while j > 0 and dst_edges[j] > a:
            j -= 1
        while j < len(dst) and dst_edges[j + 1] <= a:
            j += 1
        placed = 0.0
        jj = j
        while jj < len(dst) and dst_edges[jj] < b:
            lo = max(a, dst_edges[jj])
            hi = min(b, dst_edges[jj + 1])
            if hi > lo:
                frac = (hi - lo) / (b - a)
                dst[jj] += src_vals[i] * (hi - lo)
                placed += amount * frac
            jj += 1
        lost += amount - placed
    dst = [dst[k] / (dst_edges[k + 1] - dst_edges[k]) for k in range(len(dst))]
    return dst, lost


def band_integral(edges, vals, lo, hi):
    total = 0.0
    for i in range(len(vals)):
        a, b = max(edges[i], lo), min(edges[i + 1], hi)
        if b > a:
            total += vals[i] * (b - a)
    return total


BANDS = [('FUV_918_1290A', C / 1290e-8, C / 918e-8),
         ('UV_1290_3000A', C / 3000e-8, C / 1290e-8),
         ('OPT_3000_10000A', C / 10000e-8, C / 3000e-8),
         ('IR_gt_10000A', 1.0e12, C / 10000e-8)]
PROBES = [('B_10020K', 10020.0), ('B_19059K', 19059.411196903675),
          ('B_5000K', 5000.0)]


def transition(name, src, dst):
    out = {'from': src, 'to': dst,
           'from_commit': GRIDS[src]['commit'], 'to_commit': GRIDS[dst]['commit']}
    se, de = GRIDS[src]['edges'], GRIDS[dst]['edges']
    out['geometry'] = {
        'src_bins': len(se) - 1, 'dst_bins': len(de) - 1,
        'src_range_hz': [se[0], se[-1]], 'dst_range_hz': [de[0], de[-1]],
        'src_dlog': GRIDS[src]['dlog'], 'dst_dlog': GRIDS[dst]['dlog'],
        'dst_covers_src': de[0] <= se[0] and de[-1] >= se[-1],
        'low_edge_ratio_dst_over_src': de[0] / se[0],
        'high_edge_ratio_dst_over_src': de[-1] / se[-1],
    }
    probes = {}
    for label, t in PROBES:
        sv = [planck(math.sqrt(se[i] * se[i + 1]), t) for i in range(len(se) - 1)]
        dv, lost = rebin(se, sv, de)
        tot_src = band_integral(se, sv, se[0], se[-1])
        entry = {'total_src': tot_src,
                 'total_dst': band_integral(de, dv, de[0], de[-1]),
                 'lost_absolute': lost,
                 'lost_fraction': (lost / tot_src) if tot_src > 0 else 0.0,
                 'bands': {}}
        for bname, blo, bhi in BANDS:
            s = band_integral(se, sv, blo, bhi)
            d_ = band_integral(de, dv, blo, bhi)
            entry['bands'][bname] = {
                'src': s, 'dst': d_,
                'relative_movement': ((d_ - s) / s) if s > 0 else None}
        probes[label] = entry
    out['probes'] = probes
    return out


def main():
    report = {'schema': 'lumina-sh-grid-b6-movement-v1',
              'gate': 'B-6', 'kind': 'MEASUREMENT_NOT_PASS_FAIL',
              'grids': {k: {'commit': v['commit'], 'bins': len(v['edges']) - 1,
                            'dlog': v['dlog'],
                            'range_hz': [v['edges'][0], v['edges'][-1]]}
                        for k, v in GRIDS.items()},
              'transitions': [
                  transition('T1_rung_change', 'G0_pre_B_4000', 'G1_plan_B_3866'),
                  transition('T2_post_bake', 'G1_plan_B_3866', 'G2_current_3866'),
                  transition('T3_net', 'G0_pre_B_4000', 'G2_current_3866')],
              'physical_values_modified': False,
              'floor': 0, 'cap': 0, 'clamp': 0, 'jitter': 0, 'repair': 0}
    out = Path('validation/sh_grid_b6/SH_GRID_B6_MOVEMENT.json')
    out.write_text(json.dumps(report, indent=2) + '\n')
    for tr in report['transitions']:
        g = tr['geometry']
        print(f"{tr['from']} -> {tr['to']}  covers_src={g['dst_covers_src']} "
              f"low_ratio={g['low_edge_ratio_dst_over_src']:.4f} "
              f"high_ratio={g['high_edge_ratio_dst_over_src']:.6f}")
        for label, p in tr['probes'].items():
            print(f"   {label}: lost_fraction={p['lost_fraction']:.6e}  " +
                  "  ".join(f"{b}={(v['relative_movement'] if v['relative_movement'] is not None else float('nan')):+.3e}"
                            for b, v in p['bands'].items()))
    print(f"-> {out}")


if __name__ == '__main__':
    main()
