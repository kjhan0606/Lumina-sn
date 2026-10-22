#!/usr/bin/env python3
"""계측 켠 CMFGEN 장과 기존 장의 실제 수치 차이를 전주파수에서 판독한다."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_cmfgen_mom_capture import sha
from run_cmfgen_formal_primary import fields


def compare(previous, captured, out):
    if out.exists():
        raise FileExistsError('기존 장 비교 결과 보존')
    g_old, old = fields(previous)
    g_new, new = fields(captured)
    try:
        if len(old) != len(new) or len(old) != 197291 or not np.array_equal(g_old, g_new):
            raise ValueError('기하 또는 전주파수 차원 변경')
        index_equal = np.array_equal(old['index'], new['index'])
        freq_equal = np.array_equal(old['freq'], new['freq'])
        weight_equal = np.array_equal(old['weight'], new['weight'])
        if not all((index_equal, freq_equal, weight_equal)):
            raise ValueError('물리 주파수 격자·적분 가중치 변경')
        calls_mismatch = int(np.count_nonzero(old['calls'] != new['calls']))
        count_mismatch = np.zeros(8, dtype=np.int64)
        maximum_normalized = np.zeros(8)
        first_difference = None
        j_frequencies = np.empty(len(old))
        target = {}
        for begin in range(0, len(old), 1024):
            end = min(begin+1024, len(old))
            a = np.asarray(old['field'][begin:end])
            b = np.asarray(new['field'][begin:end])
            diff = abs(a-b)
            count_mismatch += np.count_nonzero(a != b, axis=(0, 2))
            scale = np.maximum(np.max(abs(a), axis=2), 1e-300)
            normalized = np.max(diff, axis=2)/scale
            maximum_normalized = np.maximum(maximum_normalized, np.max(normalized, axis=0))
            j_frequencies[begin:end] = normalized[:, 0]
            if first_difference is None and np.any(a != b):
                first_difference = int(begin+np.where(np.any(a != b, axis=(1, 2)))[0][0]+1)
            for index in (60714, 60715):
                if begin < index <= end:
                    local = index-1-begin
                    target[str(index)] = dict(frequency_Hz=float(old['freq'][index-1]*1e15),
                        maximum_J_relative=float(normalized[local, 0]),
                        old_native49_50_J=a[local, 0, 49:51].tolist(),
                        new_native49_50_J=b[local, 0, 49:51].tolist(),
                        calls_old=int(old['calls'][index-1]),
                        calls_new=int(new['calls'][index-1]))
        result = dict(scope=__doc__, records=len(old), geometry_equal=True,
                      frequency_equal=freq_equal, weights_equal=weight_equal,
                      calls_mismatch_count=calls_mismatch,
                      component_mismatch_counts=count_mismatch.tolist(),
                      component_max_frequency_normalized_difference=maximum_normalized.tolist(),
                      first_field_difference_frequency_index=first_difference,
                      J_median_frequency_max_relative=float(np.median(j_frequencies)),
                      J_p95_frequency_max_relative=float(np.percentile(j_frequencies, 95)),
                      J_max_frequency_max_relative=float(np.max(j_frequencies)),
                      J_frequency_max_relative_index=int(np.argmax(j_frequencies)+1),
                      target=target, native_nodes_zero_based=[49,50],
                      original_sha256=sha(previous), captured_sha256=sha(captured),
                      final_goal='NOT_MET')
        with out.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(result, ensure_ascii=False))
    finally:
        old._mmap.close()
        new._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'captured', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    compare(*(getattr(args, name).resolve() for name in ('previous', 'captured', 'out')))
