#!/usr/bin/env python3
"""공개 toy06의 이온 상단만 맞춘 대조 덱. 준위/판본의 완전 복제는 아니다.

원본 수정, 준위 인구 삭제, 단면적/분율 재정규화는 하지 않는다.
상단의 PHOT 바닥 상태/통계가중치는 별도 build_bf_channels가 원자료에서 읽는다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

TERMINAL = {14: 4, 16: 4, 20: 4, 26: 5, 27: 5, 28: 5}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def selected(row):
    z, q = int(row['atomic_number']), int(row['ion_number'])
    if z not in TERMINAL or q < 1:
        raise ValueError(f'unexpected original ion {(z, q)}')
    return q < TERMINAL[z]


def subset(source, dest, name, predicate, counts):
    kept = total = 0
    with (source / name).open(newline='') as fi, (dest / name).open('x', newline='') as fo:
        reader = csv.DictReader(fi)
        writer = csv.DictWriter(fo, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            if predicate(row):
                writer.writerow(row)
                kept += 1
    if not kept:
        raise ValueError(f'empty {name}')
    counts[name] = dict(original=total, kept=kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('destination', type=Path)
    a = ap.parse_args()
    source, dest = a.source.resolve(), a.destination.resolve()
    dest.mkdir()  # 기존 결과를 덮어쓰지 않는다.
    names = ['levels.csv', 'line_list.csv', 'ionization_energies.csv', 'active_ions.csv',
             'zeta_ions.csv', 'zeta_temps.csv', 'zeta_data.npy', 'atom_masses.csv',
             'coldata_cmfgen_manifest.csv', 'DECK_PROVENANCE.json']
    with (source / 'coldata_cmfgen_manifest.csv').open(newline='') as f:
        coll = list(csv.DictReader(f))
    kept_coll = [r for r in coll if int(r['ion0']) < TERMINAL[int(r['Z'])]]
    for row in kept_coll:
        if row['status'] == 'OK':
            name = row['out_bin']
            if Path(name).name != name or name.startswith('.'):
                raise ValueError('invalid collision filename')
            names.append(name)
    before = {n: digest(source / n) for n in names}
    counts = {}
    for name in ('levels.csv', 'line_list.csv', 'ionization_energies.csv', 'active_ions.csv'):
        subset(source, dest, name, selected, counts)
    with (source / 'zeta_ions.csv').open(newline='') as f:
        zeta = list(csv.DictReader(f))
    mask = np.array([selected(r) for r in zeta])
    matrix = np.load(source / 'zeta_data.npy', allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(zeta):
        raise ValueError('zeta shape/identity mismatch')
    subset(source, dest, 'zeta_ions.csv', selected, counts)
    with (dest / 'zeta_data.npy').open('xb') as f:
        np.save(f, matrix[mask], allow_pickle=False)
    subset(source, dest, 'coldata_cmfgen_manifest.csv',
           lambda r: int(r['ion0']) < TERMINAL[int(r['Z'])], counts)
    for name in names:
        if name in ('zeta_temps.csv', 'atom_masses.csv') or name.endswith('_cmfgen.bin'):
            shutil.copyfile(source / name, dest / name)
    # 하위 이온의 내부 준위 번호/SL 분류와 선 파라미터는 변경하지 않는다.
    with (dest / 'active_ions.csv').open(newline='') as f:
        active = list(csv.DictReader(f))
    expected = {(z, q) for z, top in TERMINAL.items() for q in range(1, top)}
    if {(int(r['atomic_number']), int(r['ion_number'])) for r in active} != expected:
        raise ValueError('incomplete retained ion ladder')
    if any(digest(source / n) != h for n, h in before.items()):
        raise ValueError('original input changed')
    provenance = json.loads((source / 'DECK_PROVENANCE.json').read_text())
    provenance['schema'] = 'toy06-ion-ceiling-control-v1'
    provenance['deck'] = str(dest)
    provenance['finalized'] = False  # BF 원자료 재생성과 실제 loader 확인이 뒤따른다.
    provenance['note'] = '공개 이온 상단 대조; Fe I 부재/준위 목록 차이 보존. 완전 공개 원자모형 아님.'
    provenance['ion_ceiling_control'] = dict(source=str(source), source_sha256=before,
        terminal_charge=TERMINAL, counts=counts,
        reference='https://arxiv.org/html/2209.11671#A2.T6',
        omitted_reference_ion='Fe I', lower_level_lists_unchanged=True)
    with (dest / 'DECK_PROVENANCE.json').open('x') as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(provenance['ion_ceiling_control'], ensure_ascii=False))


if __name__ == '__main__':
    main()
