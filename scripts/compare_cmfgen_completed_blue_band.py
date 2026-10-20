#!/usr/bin/env python3
"""같은59에서 완료된 blue 구간만으로 지지 전체가 덮인 ground 광율을 비교한다.

실패한 legacy run의 미완성 red record/부모 잔여값을 읽거나 새 whole로 쓰지 않는다.
네 ground 광율은 전체 적분 지지가 완료 구간 안에 있는 경우에만 계산한다.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

import compare_cmfgen_field_response as response
from prepare_cmfgen_field_owner_pair import verify_view
from prepare_cmfgen_mom_capture import sha


def prefix_field(run, seed, expected_nu):
    runtime = json.loads((run / 'seq_logs/whole-runtime.json').read_text())
    count = runtime['coverage']['count']
    inputs = json.loads((run / 'seq_logs/whole-inputs.json').read_text())
    if (runtime.get('diagnostic_complete') or runtime.get('returncode') != 95 or
            not 1 < count < len(expected_nu) or Path(inputs['seed']).resolve() != seed.resolve()):
        raise ValueError('기대하는 실패 run의 완료 prefix가 아님')
    if sha(run / 'SCRTEMP') != sha(seed / 'SCRTEMP'):
        raise ValueError('legacy 물질59 신원 변경')
    rows = list(csv.DictReader((run / 'WHOLE_SWEEP.csv').open()))
    if len(rows) != count or any(int(r['outer']) != 60 or int(r['index']) != i + 1 for i, r in enumerate(rows)):
        raise ValueError('prefix coverage 누락/중복')
    info = (run / 'EDDFACTOR_INFO').read_text().splitlines()[2].split()
    nd, recl, word = map(int, info[:3])
    if word != 8 or info[5] != 'T' or recl % 8 or recl < (nd + 1) * 8:
        raise ValueError('EDD direct-access 형식 불일치')
    with (run / 'EDDFACTOR').open('rb') as f:
        f.seek(14 * recl)
        # count 이후의 red/실패 record는 의도적으로 읽지 않는다.
        raw = f.read(count * recl)
    if len(raw) != count * recl:
        raise ValueError('완료 prefix record가 잘림')
    data = np.frombuffer(raw, dtype='<f8').reshape(count, recl // 8)
    if (not np.array_equal(data[:, nd] * 1e15, expected_nu[-count:][::-1]) or
            not np.allclose([float(r['freq_native']) for r in rows], data[:, nd], rtol=5e-8, atol=0)):
        raise ValueError('prefix 주파수/실제 EDD record 대응 불일치')
    field = data[::-1, :nd].copy()
    if not np.isfinite(field).all() or np.any(field < 0):
        raise ValueError('prefix 복사장 오류')
    return field, count


def supported_rate(weights, prefix, new_field):
    if np.any(weights[:-len(prefix)] != 0):
        raise ValueError('완료 prefix 밖에 광율 적분 지지가 있음')
    local = weights[-len(prefix):]
    return local @ prefix, local @ new_field[-len(prefix):]


def compare(seed, legacy, new_run, deck, out):
    if out.exists():
        raise ValueError('기존 blue 구간 비교 보존')
    loaded = verify_view(seed, deck)
    _, new, nu, nd = response.read_complete_pair(seed, new_run)
    old, count = prefix_field(legacy, seed, nu)
    if old.shape != (count, nd):
        raise ValueError('prefix 깊이 불일치')
    manifest = json.loads((deck / 'cmfgen_bf_channels_manifest.json').read_text())
    channels = response.bf.parse_cmfgen_bf_channels(deck / 'cmfgen_bf_channels.bin', manifest['n_levels'], manifest['n_freq_bins'], manifest)
    levels = list(csv.DictReader((deck / 'levels.csv').open()))
    smoother = response.bf.PhotSmoother.from_provenance(json.loads((deck / 'DECK_PROVENANCE.json').read_text())['phot_preprocessing'])
    links = response.bf.atomic.load_cmfgen_links(seed / 'atomic_links.txt')
    results = {}; output = []
    for key in sorted(response.bf.TARGETS):
        index = next(i for i, row in enumerate(levels) if
            (int(row['atomic_number']), int(row['ion_number']) + 1, int(row['level_number'])) == (*key, 0))
        osc = response.bf.parse_osc(links[key]['osc'])
        energy = float(osc.levels['E_cm'][0]) * response.bf.CM_TO_EV
        terms = response.terms_for_level(links[key]['phot'], channels, index, str(osc.levels['config'][0]),
            energy, float(osc.ionization_eV), key[0], float(osc.z_screen), smoother)
        weights, _ = response.linear_sigma_weights(nu, terms)
        a, b = supported_rate(weights, old, new)
        if np.any(a <= 0):
            raise ValueError('ground 광율 0의 비율은 별도 판독 필요')
        ratio = b / a
        results[str(key)] = dict(level_1based=1, entire_integration_support_covered=True,
            new_over_legacy_min_median_max=[float(ratio.min()), float(np.median(ratio)), float(ratio.max())])
        for d in range(nd):
            output.append([*key, 1, d + 1, a[d], b[d], ratio[d]])
    out.mkdir()
    with (out / 'ground_rates.csv').open('x', newline='') as f:
        writer = csv.writer(f); writer.writerow(['Z', 'stage', 'level_1based', 'depth_1based', 'legacy_gamma_s-1', 'formal_gamma_s-1', 'formal_over_legacy'])
        writer.writerows(output)
    source = Path(__file__).resolve(); shutil.copy2(source, out / source.name)
    files = [seed / 'SCRTEMP', legacy / 'EDDFACTOR', legacy / 'WHOLE_SWEEP.csv',
        legacy / 'seq_logs/whole-runtime.json', legacy / 'seq_logs/whole-inputs.json',
        new_run / 'EDDFACTOR', new_run / 'seq_logs/whole-runtime.json', deck / 'cmfgen_bf_channels.bin',
        deck / 'cmfgen_bf_channels_manifest.json', source, Path(response.__file__).resolve()]
    report = dict(scope='같은 물질59의 두 수송 경로; 전체 지지가 완료된 ground 네 준위의 광율만',
        material_record=59, completed_prefix_count=count, full_frequency_count=len(nu),
        legacy_full_field_complete=False, stale_red_records_consumed=False,
        smallest_completed_frequency_Hz=float(nu[-count]), ground_rates=results,
        source_equivalence=loaded['seed_atomic_sources'],
        full_ion_rate_or_SE_equivalence='NOT_CLAIMED', energy_pass='NOT_JUDGED', final_goal='NOT_JUDGED',
        input_sha256={str(p): sha(p) for p in files}, output_sha256=sha(out / 'ground_rates.csv'))
    with (out / 'blue-band.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(completed_prefix=count, ground_rates=results), ensure_ascii=False))


def compare_all_levels(seed, legacy, new_run, deck, out):
    if out.exists():
        raise ValueError('기존 전준위 blue 비교 보존')
    started = time.monotonic()
    loaded = verify_view(seed, deck)
    parent, new, nu, nd = response.read_complete_pair(seed, new_run)
    old, count = prefix_field(legacy, seed, nu)
    manifest = json.loads((deck / 'cmfgen_bf_channels_manifest.json').read_text())
    channels = response.bf.parse_cmfgen_bf_channels(deck / 'cmfgen_bf_channels.bin', manifest['n_levels'], manifest['n_freq_bins'], manifest)
    levels = list(csv.DictReader((deck / 'levels.csv').open()))
    lookup = {(int(r['atomic_number']), int(r['ion_number']) + 1, int(r['level_number']) + 1): i for i, r in enumerate(levels)}
    links = response.bf.atomic.load_cmfgen_links(seed / 'atomic_links.txt')
    models = response.bf.atomic.load_cmfgen_model_spec(seed / 'MODEL_SPEC', seed / 'atomic_links.txt')
    populations, _, _ = response.population_inputs(seed, models, links)
    smoother = response.bf.PhotSmoother.from_provenance(json.loads((deck / 'DECK_PROVENANCE.json').read_text())['phot_preprocessing'])
    # 앞의 세 행은 같은 완료 구간, 뒤의 두 행은 각각 실제 완성된 장이다.
    blue = np.concatenate((old, new[-count:], parent[-count:]), axis=1)
    full = np.concatenate((new, parent), axis=1)
    out.mkdir(); sums = {}; completed = 0
    with (out / 'levels.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Z', 'stage', 'level_1based', 'depth_1based', 'population59_cm3', 'entire_support_covered',
            'legacy_blue_gamma', 'formal_blue_gamma', 'parent_blue_gamma', 'formal_full_gamma', 'parent_full_gamma'])
        for key in sorted(populations):
            model = populations[key]; osc = model['osc']; totals = np.zeros((5, nd)); supported = 0
            for level in range(1, model['n'] + 1):
                if time.monotonic() - started > 250:
                    raise RuntimeError('250초 상한; 부분 준위 CSV는 전준위 결과가 아님')
                index = lookup[(*key, level)]
                energy = float(osc.levels['E_cm'][level - 1]) * response.bf.CM_TO_EV
                terms = response.terms_for_level(links[key]['phot'], channels, index, str(osc.levels['config'][level - 1]),
                    energy, float(osc.ionization_eV), key[0], float(osc.z_screen), smoother)
                weights, _ = response.linear_sigma_weights(nu, terms)
                local = (weights[-count:] @ blue).reshape(3, nd)
                all_nu = (weights @ full).reshape(2, nd)
                values = np.concatenate((local, all_nu))
                complete_support = not np.any(weights[:-count] != 0)
                supported += complete_support
                pop = model['population'][:, level - 1]; totals += values * pop
                for d in range(nd):
                    writer.writerow([*key, level, d + 1, pop[d], int(complete_support), *values[:, d]])
                completed += 1
            sums[str(key)] = dict(full_levels=model['n'], entire_support_levels=supported,
                population_sum_cm3=model['population'].sum(axis=1).tolist(),
                legacy_blue=totals[0].tolist(), formal_blue=totals[1].tolist(), parent_blue=totals[2].tolist(),
                formal_full=totals[3].tolist(), parent_full=totals[4].tolist())
    if completed != sum(m['n'] for m in populations.values()) or len(sums) != 4:
        raise ValueError('실제 전준위 coverage 미달')
    source = Path(__file__).resolve(); shutil.copy2(source, out / source.name)
    files = [seed / 'SCRTEMP', seed / 'EDDFACTOR', legacy / 'EDDFACTOR', legacy / 'WHOLE_SWEEP.csv',
        legacy / 'seq_logs/whole-runtime.json', legacy / 'seq_logs/whole-inputs.json',
        new_run / 'EDDFACTOR', new_run / 'seq_logs/whole-runtime.json', deck / 'cmfgen_bf_channels.bin',
        deck / 'cmfgen_bf_channels_manifest.json', source, Path(response.__file__).resolve()]
    report = dict(scope='850준위의 완료 blue 구간 광율과 양쪽 완성 장의 전체광율을 같은59 인구로 가중',
        material_record=59, completed_levels=completed, all_target_levels_complete=True, completed_prefix_count=count,
        smallest_completed_frequency_Hz=float(nu[-count]), ions=sums,
        warning='legacy_blue는 부분 주파수 예산이다. legacy red/전체율/전체 SE 동등성을 주장하지 않는다.',
        stale_red_records_consumed=False, input_sha256={str(p): sha(p) for p in files},
        output_sha256=sha(out / 'levels.csv'), elapsed_seconds=time.monotonic() - started, final_goal='NOT_JUDGED')
    with (out / 'blue-all-levels.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    compact = {}
    for key, ion in sums.items():
        a, b, p, full_new = [np.array(ion[k]) for k in ('legacy_blue', 'formal_blue', 'parent_blue', 'formal_full')]
        compact[key] = {name: [float(x.min()), float(np.median(x)), float(x.max())] for name, x in (
            ('same59_formal_over_legacy_blue', b / a), ('legacy59_over_parent58_blue', a / p), ('formal_blue_share_of_full', b / full_new))}
    print(json.dumps(dict(completed_levels=completed, elapsed_seconds=report['elapsed_seconds'], ratios=compact), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('seed', 'legacy', 'new-run', 'deck', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--all-target-levels', action='store_true')
    a = p.parse_args()
    function = compare_all_levels if a.all_target_levels else compare
    function(*(getattr(a, name).resolve() for name in ('seed', 'legacy', 'new_run', 'deck', 'out')))
