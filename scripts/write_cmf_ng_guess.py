"""봉인된 감쇠 Ng 방향을 native reader로 검증할 미수렴 초기 추정으로 보존한다."""
import array
import hashlib
import json
from pathlib import Path
import runpy
import struct


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    # 같은 계산을 실행하고 전체 보존/비음수 검사를 그대로 사용한다.
    state = runpy.run_path('source/assess_damped.py')
    previous = json.loads(Path('previous_direction.json').read_text())
    current = json.loads(Path('damped_direction.json').read_text())
    for key in ('A', 'B', 'fortran_formula_relative', 'records', 'accepted', 'input_sha256'):
        if current[key] != previous[key]:
            raise ValueError(f'봉인된 후보와 다른 계산: {key}')
    if current['accepted'] is None or current['accepted']['alpha'] != .25:
        raise ValueError('계획과 다른 감쇠 후보')
    if not state['constraints'](state['candidate'], state['latest'])['admissible']:
        raise ValueError('유효하지 않은 후보')

    latest_path = state['env']['RUNS'][-1] / 'iterate_input.bin'
    raw = bytearray(latest_path.read_bytes())
    ns, ni, nl, ne, nb, generation, iteration = struct.unpack_from('<7Q', raw, 72)
    if iteration != 4 or (ns, ni, nl, ne, nb) != (113, 27, 19627, 6, 117606):
        raise ValueError('잘못된 원본 저장점')
    offset = 152 + 4 * (nl + 2 * ni) + 8 * (4 + ne) * ns
    values = array.array('d', state['candidate'])
    if len(values) != (1 + ni + nl) * ns or values.itemsize != 8:
        raise ValueError('후보 배열 범위 불일치')
    identity = dict(
        purpose='DAMPED_NG_NUMERICAL_GUESS_NOT_SOLUTION',
        parent_iteration=iteration, parent_identity=raw[8:72].decode(),
        parent_sha256=digest(latest_path), source_sha256=digest(Path('source.sha256')),
        direction_sha256=digest(Path('damped_direction.json')),
        A=current['A'], B=current['B'], alpha=current['accepted']['alpha'],
        J='unchanged_iteration_4_predictor_must_recompute',
        physical_history=False, converged=False,
    )
    with Path('guess_identity.txt').open('x') as stream:
        json.dump(identity, stream, indent=2, allow_nan=False)
        stream.write('\n')
    raw[8:72] = digest(Path('guess_identity.txt')).encode()
    struct.pack_into('<QQ', raw, 112, generation + 1, 0)
    raw[offset:offset + len(values) * 8] = values.tobytes()
    # 주파수별 J와 footer는 그대로 보존하며 새 해로 취급하지 않는다.
    with Path('guess_input.bin').open('xb') as stream:
        stream.write(raw)
    print('NG_GUESS_WRITTEN iteration=0 physical_history=0 converged=0 native_validation=PENDING')


if __name__ == '__main__':
    main()
