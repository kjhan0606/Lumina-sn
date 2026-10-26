#!/usr/bin/env python3
"""CMFGEN의 충돌/연속항 인쇄 SE를 원시 witness와 원49점에서 비교한다."""

import argparse
from collections import defaultdict
from itertools import groupby
import json
import math
from pathlib import Path

from ptf_gated_metrics import read_variables
from run_cmfgen_lambda_probe import sha
from run_cmfgen_native_se_observation import read_witness


PHASES = ('STEQ ARRAY- Collisional Terms', 'STEQ ARRAY- Continuum Terms')
IONS = ('Si II', 'Si III', 'Si IV', 'S II', 'S III', 'S IV')


def printed_value(field):
    value = float(field.replace('D', 'E'))
    if not math.isfinite(value):
        raise ValueError('비유한 STEQ 인쇄값')
    if value == 0:
        return 0.0, 0.0
    return value, 0.5*10**(math.floor(math.log10(abs(value)))-4)


def parse(path, variables, nd, node):
    if not 0 <= node < nd:
        raise ValueError('원래점 주소 범위 밖')
    active = []
    for ion, members in groupby(variables, key=lambda var: var.ion):
        members = list(members)
        if all(var.terminal for var in members):
            continue
        if any(var.terminal for var in members):
            raise ValueError('원래 활성/terminal 변수 혼합: '+ion)
        active.append((ion, members))
    if not active:
        raise ValueError('활성 원자 이온 없음')
    sections = {}
    current = None
    with path.open(errors='replace') as stream:
        for line in stream:
            marker = next((m for m in PHASES if m in line), None)
            if marker:
                if marker in sections:
                    raise ValueError('중복 SE 단계')
                current = marker
                sections[current] = []
                continue
            if current is not None and line[1:5].strip().isdigit() and line[5:8] in ('** ', ' * '):
                sections[current].append(line)
    if set(sections) != set(PHASES):
        raise ValueError('필수 충돌/연속 SE 단계 누락')
    phases = {}
    nblocks = math.ceil(nd/10)
    for phase, lines in sections.items():
        if not lines or len(lines)%nblocks:
            raise ValueError(f'{phase}: 출력 행수/반경 블록 불일치')
        nrows = len(lines)//nblocks
        values = defaultdict(list)
        errors = defaultdict(list)
        reference_sizes = None
        for block in range(nblocks):
            rows = lines[block*nrows:(block+1)*nrows]
            starts = [i for i, line in enumerate(rows) if line[5:8] == '** ']
            if len(starts) != len(active) or starts[0] != 0:
                raise ValueError(f'{phase}: 활성 이온 블록 수 {len(starts)} != {len(active)}')
            sizes = [b-a for a,b in zip(starts, starts[1:]+[nrows])]
            if reference_sizes is None:
                reference_sizes = sizes
            elif sizes != reference_sizes:
                raise ValueError(f'{phase}: 반경별 이온 행수 변경')
            for (ion, members), start, size in zip(active, starts, sizes):
                nsl = len(members)
                if size < nsl+1:
                    raise ValueError(f'{phase}: {ion} SE 행이 SL+보존식보다 적음')
                for local, line in enumerate(rows[start:start+size], 1):
                    if int(line[1:5]) != local:
                        raise ValueError(f'{phase}: {ion} 내부 SE 주소 불연속')
                    if block != node//10 or local == size:
                        continue  # 마지막 행은 원소수 보존식이며 과정률이 아니다.
                    offset = node%10
                    field = line[8+12*offset:8+12*(offset+1)]
                    value, error = printed_value(field)
                    var = members[0]
                    key = (var.species, var.charge if local <= nsl else var.charge+1)
                    values[key].append(value)
                    errors[key].append(error)
        phases[phase] = ({key: math.fsum(items) for key, items in values.items()},
                         {key: math.fsum(items) for key, items in errors.items()})
    if phases[PHASES[0]][0].keys() != phases[PHASES[1]][0].keys():
        raise ValueError('두 SE 단계의 이온 지지 불일치')
    return phases


def summarize(path, witness, links, node):
    header, fields = read_witness(witness)
    variables = read_variables(links, header['nr'])
    if len(variables) != header['nr']:
        raise ValueError('원시 witness/준위 변수 수 불일치')
    phases = parse(path, variables, header['nd'], node)
    result = {}
    for ion in IONS:
        species, roman = ion.split()
        charge = {'II': 1, 'III': 2, 'IV': 3}[roman]
        rows = [i for i, var in enumerate(variables)
                if var.species == species and var.charge == charge]
        if not rows:
            raise ValueError('원 이온 행 없음: '+ion)
        coll_v, coll_e = phases[PHASES[0]]
        cont_v, cont_e = phases[PHASES[1]]
        key = (species, charge)
        coll = coll_v[key]
        cont = cont_v[key]
        coll_err = coll_e[key]
        cont_err = cont_e[key]
        raw = math.fsum(float(fields['raw'][i, node]) for i in rows)
        result[ion] = dict(levels=len(rows), collisions=coll,
                           collisions_print_bound=coll_err,
                           after_continuum=cont,
                           after_continuum_print_bound=cont_err,
                           continuum_increment=cont-coll,
                           continuum_increment_print_bound=coll_err+cont_err,
                           final_raw=raw, later_increment=raw-cont,
                           later_increment_print_bound=cont_err)
    return dict(schema='cmfgen-verbose-se70-stage-sums-v1', node_0based=node,
                source_file=str(path), source_sha256=sha(path),
                witness=str(witness), witness_sha256=sha(witness),
                links=str(links/'LEVEL_SL_STEQ_LINKS'),
                printed_precision='ES12.4; bounds cover decimal rounding only, not native floating summation',
                stage_sums=result, final_goal='NOT_JUDGED')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--node', type=int, default=49)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    run = args.run.resolve()
    runtime = json.loads((run/'seq_logs/verbose-se-runtime.json').read_text())
    if not runtime['reproduced']:
        raise ValueError('원71 재현 실패: 단계합을 권위 증거로 사용 불가')
    material = run/'material'
    result = summarize(material/'fort.19', material/'NATIVE_SE_BOUND_000071.bin',
                       material, args.node)
    dest = run/'seq_logs/verbose-se-node49.json'
    if args.verify_only:
        if json.loads(dest.read_text()) != result:
            raise ValueError('기존 단계합 보고서 재현 불일치')
    else:
        with dest.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
