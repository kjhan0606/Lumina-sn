"""물질 갱신 전 완성 J를 새 수송의 초기 추정값으로만 연결한다.

물질응답·관측 결과 소비자는 여전히 동일 계수의 InputField를 사용한다.
여기서는 이전 준비 기록을 명시하고 같은 반경/주파수/epoch를 확인한다.
계수·경계·광선 이력·checkpoint를 이전 물질에서 가져오지는 않는다.
"""
from pathlib import Path

import numpy as np

from lumina_accepted_field_input import open_prepared
from lumina_fullsr_sc_actual import InputField


def validate_initial_arguments(input_j, input_j_sha256, initial_preparation,
                               initial_preparation_sha256):
    if ((input_j is None) != (input_j_sha256 is None) or
            (initial_preparation is None) != (initial_preparation_sha256 is None) or
            (initial_preparation is not None and input_j is None)):
        raise ValueError('초기 J/해시와 이전 물질 준비/해시 쌍이 필요하다')


def open_transport_trial(source, path, digest, *, initial_preparation=None,
                         initial_preparation_sha256=None):
    """기본 동일 물질 경로는 유지하고 명시한 이전 물질만 초기화에 허용한다."""
    validate_initial_arguments(path, digest, initial_preparation, initial_preparation_sha256)
    if path is None:
        raise ValueError('완성 자체 J의 명시 경로가 필요하다')
    if initial_preparation is None:
        return InputField(source, path, digest)
    previous_path = Path(initial_preparation).resolve()
    _, previous, _, _ = open_prepared(previous_path, initial_preparation_sha256)
    with previous:
        epoch = source.identity.get('epoch_seconds')
        if (epoch is None or previous.identity.get('epoch_seconds') != epoch or
                (previous.ns, previous.nf) != (source.ns, source.nf) or
                not np.array_equal(previous.radius, source.radius) or
                not np.array_equal(previous.frequency, source.frequency)):
            raise ValueError('물질 갱신 초기 J는 같은 반경/주파수/epoch에서만 재사용 가능')
        # 이전 물질의 완료 수송이라는 증거는 원래의 엄격한 소비자로 확인한다.
        field = InputField(previous, path, digest)
        try:
            for first in range(0, field.values.size, 131072):
                values = field.values.reshape(-1)[first:first+131072]
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError('물질 갱신 초기 J의 비유한/음수')
            field.identity = dict(role='OWN_COMPLETED_FIELD_INITIAL_TRIAL_FOR_NEW_MATERIAL',
                path=str(Path(path).resolve()), sha256=digest,
                initial_preparation=str(previous_path),
                initial_preparation_sha256=initial_preparation_sha256,
                origin_source_identity=previous.identity, target_source_identity=source.identity,
                self_consistent=False, accepted_field_for_target_material=False)
        except Exception:
            field.close()
            raise
    return field
