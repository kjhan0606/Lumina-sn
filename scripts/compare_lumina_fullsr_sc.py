#!/usr/bin/env python3
"""완성 SC 자체장만 기존 native90 Si/S 생산율 대조기에 전달한다."""
import argparse
import json
from pathlib import Path
import time

import lumina_native90_rates as R
from lumina_fullsr_sc_actual import BASE, BASE_SHA
import lumina_native_columns as N


def completed_entry(path, source_identity, intervals, iteration, previous=None):
    record = json.loads(Path(path).read_text())
    expected_input = 'ZERO_OWN_START' if previous is None else previous
    if (record.get('schema') != 'lumina-fullsr-sc-native90-v1' or
            record.get('complete_original_grid') is not True or record.get('next_frequency') != -1 or
            record.get('completed_frequencies') != 1115020 or
            record.get('source_identity') != source_identity or
            record.get('original_layout', {}).get('physical_intervals') != intervals or
            record.get('identity', {}).get('input_J') != expected_input or
            record.get('completed_actual_F_calls') != 1):
        raise ValueError('동일 물질의 전격자 완료/자체장 계보 불일치')
    j = record['J']
    if N.sha(j['path']) != j['sha256']: raise ValueError('완성 자체 J 변경')
    return dict(iteration=iteration, input_iteration=iteration-1, J=j,
        report=dict(completed_actual_F_calls=1, physical_intervals=intervals,
                    transport='FULLSR_INVARIANT_UPWIND_SC',
                    discretization=record['identity'], accepted_physical_field=False))


def compare(first, out, *, second=None, seconds=1200):
    if type(seconds) is not int or not 1 <= seconds <= 1200: raise ValueError('비교 내부1200초 한도')
    if Path(out).exists(): raise FileExistsError('기존 Si/S 비교 결과 보존')
    if N.sha(BASE) != BASE_SHA: raise ValueError('원 물질·계수 봉인 변경')
    c = json.loads(BASE.read_text()); deadline = time.monotonic()+seconds
    with N.open_sealed_native(c['config'], c['preparation'], c['preparation_sha256']) as source:
        if source.identity != c['source_identity']: raise ValueError('원 계수 신원 변경')
        intervals = 16172
        entries = [completed_entry(first, source.identity, intervals, 1)]
        if second is not None:
            entries.append(completed_entry(second, source.identity, intervals, 2, entries[0]['J']))
        # 기존 생산 PHOT/Milne·원90점 소비자를 재사용한다. 옛 수송기로 계산한 척하지 않는다.
        contract = dict(source_identity=source.identity, config_sha256=N.sha(c['config']),
                        plan_layout=dict(physical_intervals=intervals))
        result = R.compare(c['config'], source, contract, entries, out, deadline)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('first', type=Path); parser.add_argument('out', type=Path)
    parser.add_argument('--second', type=Path); parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(compare(**vars(args)), ensure_ascii=False))
