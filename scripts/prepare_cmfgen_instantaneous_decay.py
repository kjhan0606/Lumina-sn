#!/usr/bin/env python3
"""확인한 native 소스의 격리 복사본만 수정한다. 설치·빌드·실행은 하지 않는다.

시간 간격0의 isotope 변화와 현재 순간 붕괴출력을 구분한다. 원본 input70과
기존 zero-heat 결과는 보존하며 이 복사본을 기존 앵커로 재명명하지 않는다.
"""
import argparse
import hashlib
import json
from pathlib import Path


ORIGINAL_SHA256='5dd9944de92934a427ade7be70ac1440797c55e476a6de2947df82b06054be3a'
OLD=b'IF(DELTA_T .EQ. 0.0_LDP)THEN'
NEW=(b'IF(DELTA_T .EQ. 0.0_LDP .AND.\n'
     b'\t1           .NOT. INSTANTANEOUS_ENERGY_DEPOSITION)THEN')


def corrected_source(original):
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_SHA256:
        raise ValueError('검토한 원 native 소스와 다름')
    if original.count(OLD)!=3:raise ValueError('1-step/2-step/요약 분기 수 불일치')
    return original.replace(OLD,NEW)


def stage(source,out):
    original=source.read_bytes();patched=corrected_source(original)
    if out.exists():raise FileExistsError('기존 소스 보존')
    out.mkdir(parents=False)
    (out/'do_species_decays_v2.original.f').write_bytes(original)
    (out/'do_species_decays_v2.f').write_bytes(patched)
    report=dict(scope=__doc__,original_source=str(source.resolve()),original_sha256=ORIGINAL_SHA256,
        corrected_sha256=hashlib.sha256(patched).hexdigest(),guards_changed=3,
        source_original_unchanged=source.read_bytes()==original,installed_in_native=False,
        new_native_calculations=0)
    if not report['source_original_unchanged']:raise ValueError('원본 실행 중 변경')
    (out/'stage.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(stage(a.source,a.out)))
