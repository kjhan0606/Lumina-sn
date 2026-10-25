"""봉인 CMFGEN 원래점 J의 소량 읽기 대조. 자체 수송해로 쓰지 않는다.

전 파일/전 J를 매핑하거나 다시 해시하지 않는다. 원래 주파수는 디스크 이진
탐색으로 찾고 요청한 record의 J만 읽는다. 전체 해시는 이전 봉인 계획을 계승한다.
"""
from collections import OrderedDict
import json
import os
from pathlib import Path
import struct

import numpy as np

from lumina_native_columns import request, sha


class PrimaryReferenceColumns:
    def __init__(self, path, source, identity):
        self._fd = None; self._source = source
        self.ns, self.nf = source.ns, source.nf
        self.radius, self.frequency = source.radius, source.frequency
        self.identity = dict(identity, role='REFERENCE_INPUT_CONNECTION_ONLY_NOT_SELF_SOLUTION',
            full_field_hash_verification='NOT_PERFORMED',
            interpolation='linear J in log(nu), same original spatial nodes; no spatial projection')
        self.metadata_bytes_read = self.field_bytes_read = 0
        self._headers, self._j = OrderedDict(), OrderedDict()
        try:
            self._fd = os.open(path, os.O_RDONLY)
            raw = self._read(0, 40, metadata=True)
            precision, bits, ibits, ns, count, fields = struct.unpack('<6i', raw[16:])
            self.base = 40+32*ns; self.stride = 24+64*ns; self.count = count
            if (raw[:16].rstrip() != b'FORMALPRIMARYV1' or (precision, bits, ibits, fields) != (8, 64, 32, 8) or
                    ns != self.ns or not 2 <= count <= 1000000 or
                    os.fstat(self._fd).st_size != self.base+count*self.stride):
                raise ValueError('원래점 참조 J header/크기 오류')
            geometry = np.frombuffer(self._read(40, 32*ns, metadata=True), dtype='<f8').reshape(4, ns)
            if not np.isfinite(geometry).all() or not np.array_equal(geometry[0]*1e10, self.radius):
                raise ValueError('참조 J와 완성 native 계수의 원래 반경 불일치')
            if self._header(0) != self.frequency[-1] or self._header(count-1) != self.frequency[0]:
                raise ValueError('참조 J와 전역 미세 격자의 주파수 끝 불일치')
        except BaseException:
            self.close(); raise

    def _read(self, offset, count, metadata=False):
        if self._fd is None: raise ValueError('닫힌 원래점 참조 J')
        chunks = []; remaining = count
        while remaining:
            value = os.pread(self._fd, remaining, offset+count-remaining)
            if not value: raise ValueError('잘린 참조 J 요청 구간')
            chunks.append(value); remaining -= len(value)
            if metadata: self.metadata_bytes_read += len(value)
            else: self.field_bytes_read += len(value)
        return b''.join(chunks)

    @staticmethod
    def _keep(cache, key, value, count):
        cache[key] = value; cache.move_to_end(key)
        if len(cache) > count: cache.popitem(last=False)
        return value

    def _header(self, record):
        if record in self._headers:
            self._headers.move_to_end(record); return self._headers[record]
        index, calls, frequency, weight = struct.unpack('<iidd', self._read(
            self.base+record*self.stride, 24, metadata=True))
        if (index != record+1 or not 1 <= calls <= 20 or
                not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(weight) or weight <= 0):
            raise ValueError('참조 J의 요청 record/주파수 불일치')
        return self._keep(self._headers, record, frequency*1e15, 1024)

    def _values(self, record):
        if record in self._j:
            self._j.move_to_end(record); return self._j[record]
        values = np.frombuffer(self._read(self.base+record*self.stride+24, 8*self.ns), dtype='<f8')
        if not np.isfinite(values).all() or np.any(values < 0): raise ValueError('참조 J 요청 record의 비유한/음수 값')
        return self._keep(self._j, record, values, 64)

    def bracket(self, frequency):
        if not np.isfinite(frequency) or not self.frequency[0] <= frequency <= self.frequency[-1]:
            raise ValueError('참조 J 범위 밖의 주파수 요청')
        high, low = 0, self.count-1
        while low-high > 1:
            mid = (high+low)//2
            if self._header(mid) > frequency: high = mid
            else: low = mid
        red, blue = self._header(low), self._header(high)
        if not red <= frequency <= blue or not red < blue: raise ValueError('참조 J의 주파수 순서/브래킷 오류')
        weight = float((np.log(frequency)-np.log(red))/(np.log(blue)-np.log(red)))
        return low, high, weight

    def read(self, rows, columns):
        if self.closed or self._source.closed: raise ValueError('닫힌 참조 J/계수 연결')
        rows, columns = request(rows, columns, self.ns, self.nf)
        if len(columns) > 32: raise ValueError('참조 J 소량 요청은 최대32열이다')
        result = np.empty((len(rows), len(columns)))
        for column, index in enumerate(columns):
            low, high, weight = self.bracket(float(self.frequency[index]))
            a, b = self._values(low)[rows], self._values(high)[rows]
            result[:, column] = a if weight == 0 else b if weight == 1 else a+(b-a)*weight
        if not np.isfinite(result).all() or np.any(result < 0): raise ValueError('참조 J 보간의 수치 범위 오류')
        result.flags.writeable = False
        return result

    @property
    def closed(self): return self._fd is None

    def close(self):
        if self._fd is not None: os.close(self._fd); self._fd = None
        self._headers.clear(); self._j.clear()

    def __enter__(self): return self

    def __exit__(self, *args): self.close()


def open_sealed_reference(config_path, preparation_path, preparation_sha256, source):
    config_path, preparation_path = Path(config_path), Path(preparation_path)
    if sha(preparation_path) != preparation_sha256: raise ValueError('참조 연결의 봉인 준비 기록 불일치')
    preparation = json.loads(preparation_path.read_text()); config = json.loads(config_path.read_text())
    if sha(config_path) != preparation['input_sha256'][str(config_path)]: raise ValueError('참조 연결 config 불일치')
    plan_path = Path(config['plan'])
    if sha(plan_path) != config['plan_sha256'] or config['plan_sha256'] != preparation['input_sha256'][str(plan_path)]:
        raise ValueError('참조 연결의 봉인 계획 불일치')
    plan = json.loads(plan_path.read_text())
    if (source.identity.get('preparation_sha256') != preparation_sha256 or
            source.identity.get('material_record') != plan['reference_radiation_material_record'] or
            plan['reference_field_sha256'] != preparation['reference_field_sha256']):
        raise ValueError('계수와 참조 J의 물질 세대 불일치')
    field = PrimaryReferenceColumns(Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin', source,
        dict(material_record=plan['reference_radiation_material_record'], radiation_record=plan['reference_radiation_record'],
             reference_field_sha256=plan['reference_field_sha256'], plan_sha256=config['plan_sha256']))
    field.identity['full_field_hash_verification'] = 'INHERITED_SEALED_PLAN_NOT_REHASHED'
    return field
