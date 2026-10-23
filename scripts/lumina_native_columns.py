"""완성 native 계수와 동결 자체 J의 요청 열 조회. 수송/물질 풀이는 없다.

계수7평면 전체를 매핑하지 않는다. 합산 순서·signedχ·zeroη를 보존하며
전역 인덱스의 중복/순서를 그대로 반환한다. 전체 내용 해시는 상위 봉인 기록과
이번 읽기 범위를 구별해 표시한다.
"""
import json
import os
from pathlib import Path
import struct
from collections import OrderedDict

import numpy as np

from prepare_lumina_outer_input import read_native, sha
import lumina_outer_atmosphere as A


def request(rows, columns, ns, nf):
    rows, columns = np.asarray(rows), np.asarray(columns)
    if (rows.ndim != 1 or columns.ndim != 1 or not rows.size or not columns.size or
            rows.dtype.kind not in 'iu' or columns.dtype.kind not in 'iu' or
            np.any(rows < 0) or np.any(rows >= ns) or np.any(columns < 0) or np.any(columns >= nf) or
            rows.size*columns.size > 131072):
        raise ValueError('native 행/전역 열 요청 오류')
    return rows.astype(np.int64, copy=False), columns.astype(np.int64, copy=False)


class NativeSourceColumns:
    def __init__(self, path, frequency, radius, temperature, electron_density):
        self.path = Path(path); self._fd = None
        self.frequency = np.asarray(frequency)
        self.radius = np.array(radius, dtype='f8', copy=True)
        if self.radius.ndim != 1 or self.frequency.ndim != 1:
            raise ValueError('native 반경/주파수 차원 오류')
        self.ns, self.nf = len(self.radius), len(self.frequency)
        if (not 1 <= self.ns <= 512 or
                not np.isfinite(self.radius).all() or np.any(self.radius <= 0) or
                not np.all(np.diff(self.radius) < 0) or
                self.frequency.ndim != 1 or not 2 <= self.nf <= 2000000 or
                self.frequency.dtype.kind != 'f' or self.frequency.dtype.itemsize != 8):
            raise ValueError('native 반경/주파수 차원 오류')
        self.frequency = self.frequency.copy(); self.frequency.flags.writeable = False
        self.radius.flags.writeable = False
        self.metadata_bytes_read = self.coefficient_bytes_read = self.peak_request_elements = 0
        self.identity = dict(full_source_hash_verification='NOT_PERFORMED')
        try:
            self._fd = os.open(self.path, os.O_RDONLY)
            self.file_bytes = os.fstat(self._fd).st_size
            tag, ns, nf = struct.unpack('<8sII', self._read(0, 16, metadata=True))
            self.base = 16+8*(nf+2*ns)
            if (tag != b'LMPHYS01' or (ns, nf) != (self.ns, self.nf) or
                    self.file_bytes != self.base+7*ns*nf*8):
                raise ValueError('완성 native 계수의 header/길이 불일치')
            previous = None
            self.dlognu = float(np.log(self.frequency[1]/self.frequency[0]))
            if not np.isfinite(self.dlognu) or self.dlognu <= 0:
                raise ValueError('native 주파수 간격 오류')
            for begin in range(0, nf, 65536):
                count = min(65536, nf-begin)
                values = np.frombuffer(self._read(16+8*begin, 8*count, metadata=True), dtype='<f8')
                if (not np.array_equal(values, self.frequency[begin:begin+count]) or
                        not np.isfinite(values).all() or np.any(values <= 0) or
                        not np.all(np.diff(values) > 0)):
                    raise ValueError('봉인 주파수와 실제 계수의 ν 불일치')
                spacing = np.log(values[1:]/values[:-1])
                if (np.any(abs(spacing-self.dlognu) > 64*np.finfo(float).eps) or
                        (previous is not None and abs(np.log(values[0]/previous)-self.dlognu) > 64*np.finfo(float).eps)):
                    raise ValueError('native 주파수의 log-uniform 규약 불일치')
                previous = float(values[-1])
            for at, expected in enumerate((temperature, electron_density)):
                values = np.frombuffer(self._read(16+8*(nf+at*ns), 8*ns, metadata=True), dtype='<f8')
                expected = np.asarray(expected)
                if (expected.shape != (ns,) or not np.array_equal(values, expected) or
                        not np.isfinite(values).all() or np.any(values <= 0)):
                    raise ValueError('native 물질과 계수의 T/ne 또는 행 순서 불일치')
        except BaseException:
            self.close()
            raise

    def _read(self, offset, size, metadata=False):
        if self._fd is None:
            raise ValueError('닫힌 native 계수 입력')
        result = []; remaining = size
        while remaining:
            raw = os.pread(self._fd, remaining, offset+size-remaining)
            if not raw:
                raise ValueError('완성 native 계수의 요청 구간이 잘림')
            result.append(raw); remaining -= len(raw)
            if metadata:
                self.metadata_bytes_read += len(raw)
            else:
                self.coefficient_bytes_read += len(raw)
        return b''.join(result)

    def read(self, rows, columns):
        if self._fd is None:
            raise ValueError('닫힌 native 계수 입력')
        rows, columns = request(rows, columns, self.ns, self.nf)
        self.peak_request_elements = max(self.peak_request_elements, rows.size*columns.size)
        native, row_inverse = np.unique(rows, return_inverse=True)
        selected, inverse = np.unique(columns, return_inverse=True)
        starts = np.r_[0, np.flatnonzero(np.diff(selected) != 1)+1]
        ends = np.r_[starts[1:], len(selected)]
        parts = np.empty((7, len(native), len(selected)))
        for plane in range(7):
            for at, row in enumerate(native):
                for first, last in zip(starts, ends):
                    offset = self.base+8*((plane*self.ns+int(row))*self.nf+int(selected[first]))
                    parts[plane, at, first:last] = np.frombuffer(self._read(offset, 8*int(last-first)), dtype='<f8')
        if (not np.isfinite(parts).all() or any(np.any(parts[k] < 0) for k in (1, 2, 3, 5, 6))):
            raise ValueError('완성 계수의 비유한/음의 진η·FF·ES 입력')
        chi = (parts[6]+(parts[0]+parts[2]))+parts[4]
        eta = (parts[1]+parts[3])+parts[5]
        if not np.isfinite(chi).all() or not np.isfinite(eta).all():
            raise ValueError('완성 계수 합산의 수치 범위 초과')
        output = tuple(v[np.ix_(row_inverse, inverse)] for v in (chi, parts[6], eta))
        for values in output:
            values.flags.writeable = False
        return output

    def close(self):
        if self._fd is not None:
            os.close(self._fd); self._fd = None

    @property
    def closed(self):
        return self._fd is None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_sealed_native(config_path, preparation_path, preparation_sha256):
    """작은 봉인 메타데이터/물질/격자를 확인한다. 7평면 전체 해시는 재계산하지 않는다."""
    config_path, preparation_path = Path(config_path), Path(preparation_path)
    if sha(preparation_path) != preparation_sha256:
        raise ValueError('계수 준비 기록 신원 불일치')
    p = json.loads(preparation_path.read_text()); c = json.loads(config_path.read_text())
    if sha(config_path) != p['input_sha256'][str(config_path)]:
        raise ValueError('계수 준비와 config 불일치')
    seed_path = Path(c['seed'])/'material-seed.json'
    material = Path(c['seed'])/'native_material_nodes.bin'
    grid_path = Path(c['grid'])
    folder = Path(c['reuse_sources']['boundary_sources']['directory'])
    report_path = folder/'sources.json'
    for path, expected in ((seed_path, p['input_sha256'][str(seed_path)]),
                           (material, p['input_sha256'][str(material)]),
                           (grid_path, c['grid_sha256']),
                           (report_path, c['reuse_sources']['boundary_sources']['report_sha256'])):
        if sha(path) != expected:
            raise ValueError('native 계수 연결의 봉인 입력 변경')
    seed = json.loads(seed_path.read_text()); report = json.loads(report_path.read_text())
    if (seed['source_record'] != p['material_record'] or seed['plan_sha256'] != c['plan_sha256'] or
            seed['coordinate_role'] != 'CMFGEN outer-to-inner nodal values; NOT transfer-cell averages' or
            report['input_sha256'][str(material)] != seed['output_sha256']['native_material_nodes.bin'] or
            seed['output_sha256']['native_material_nodes.bin'] != p['input_sha256'][str(material)] or
            report['input_sha256'][str(grid_path)] != c['grid_sha256'] or
            p['input_sha256'][str(grid_path)] != c['grid_sha256'] or
            p['input_sha256'][str(report_path)] != c['reuse_sources']['boundary_sources']['report_sha256'] or
            report['output_sha256']['sources.bin'] != p['reused_source_binary_sha256'] or
            report['cells'] != seed['material_points'] or report['cells'] != p['native_nodes'] or
            report['frequencies'] != p['frequencies']):
        raise ValueError('native 계수의 물질 세대/좌표/격자 신원 불일치')
    physical, epoch = read_native(material)
    if epoch != seed['epoch_seconds'] or epoch != p['epoch_seconds']:
        raise ValueError('native 물질/계수 준비의 epoch 불일치')
    grid = np.load(grid_path, mmap_mode='r', allow_pickle=False)
    try:
        source = NativeSourceColumns(folder/'sources.bin', grid, physical[:, 0], physical[:, 1], physical[:, 2])
    finally:
        grid._mmap.close()
    source.identity = dict(material_record=seed['source_record'], epoch_seconds=epoch,
        coordinate_role=seed['coordinate_role'], source_sha256=p['reused_source_binary_sha256'],
        full_source_hash_verification='INHERITED_SEALED_REPORT_NOT_REHASHED',
        preparation_sha256=preparation_sha256, grid_sha256=c['grid_sha256'])
    return source


class FrozenNativeField:
    """명시한 native 반경/주파수의 읽기 전용 J. cell+aux 변환은 없다."""
    def __init__(self, path, source, radius, frequency):
        self.values = None
        if (not np.array_equal(np.asarray(radius), source.radius) or
                not np.array_equal(np.asarray(frequency), source.frequency)):
            raise ValueError('동결 J의 native 반경/주파수 좌표 불일치')
        try:
            self.values = np.load(path, mmap_mode='r', allow_pickle=False)
            if (not isinstance(self.values, np.memmap) or self.values.shape != (source.ns, source.nf) or
                    self.values.dtype != np.dtype('f8')):
                raise ValueError('동결 J의 native 차원/dtype 불일치; cell+aux 변환 없음')
        except BaseException:
            self.close()
            raise
        self.ns, self.nf = source.ns, source.nf
        self.radius = source.radius.copy(); self.radius.flags.writeable = False
        self.frequency = source.frequency

    def read(self, rows, columns):
        if self.values is None:
            raise ValueError('닫힌 동결 J 입력')
        rows, columns = request(rows, columns, self.ns, self.nf)
        result = self.values[np.ix_(rows, columns)]
        if not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError('동결 J 요청 값의 비유한/음수 입력')
        result.flags.writeable = False
        return result

    def close(self):
        if self.values is not None:
            if isinstance(self.values, np.memmap):
                self.values._mmap.close()
            elif hasattr(self.values, 'close'):
                self.values.close()
            self.values = None

    @property
    def closed(self):
        return self.values is None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def interior_coefficients(source, field, inner, outer, weight, columns):
    """χ·진η·ES·J를 따로 반경 보간한 뒤 ES×J를 조립한다."""
    if ((field.ns, field.nf) != (source.ns, source.nf) or not np.array_equal(field.radius, source.radius) or
            (field.frequency is not source.frequency and not np.array_equal(field.frequency, source.frequency)) or
            not np.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError('계수/동결 J 좌표 또는 보간 가중치 불일치')
    rows = np.array([inner, outer])
    chi, es, eta = source.read(rows, columns)
    j = field.read(rows, columns)
    mix = lambda values: (1-weight)*values[0]+weight*values[1]
    return mix(chi), mix(eta)+mix(es)*mix(j)


class NodalColumnProvider:
    """동결 입력에 묶인 반경별 계수 제공자. 조회만 하며 실제 F를 실행하지 않는다."""
    def __init__(self, geometry, source, field):
        if ((field.ns, field.nf) != (source.ns, source.nf) or
                not np.array_equal(field.radius, source.radius) or
                not np.array_equal(field.frequency, source.frequency) or
                not np.array_equal(geometry.native_radius, source.radius)):
            raise ValueError('native 계수/동결 J/수송 기하의 원래점 불일치')
        self.g, self._source, self._field = geometry, source, field
        self._profiles = OrderedDict()
        self.profile_cache_peak_entries = 0

    def _profile(self, columns):
        key = tuple(columns)
        if key in self._profiles:
            self._profiles.move_to_end(key)
            return self._profiles[key]
        # 기존 폐쇄관계에 실제 native 행의 요청 열만 준다. dummy 행은 없다.
        rows = np.arange(self._source.ns)
        chi, es, eta = self._source.read(rows, columns)
        j = self._field.read(rows, columns)
        ext = A.coefficients(self.g, chi, es, eta, j)
        magnitude = abs(ext['chi'][:self.g.added+1][::-1]); log_chi = np.zeros_like(magnitude)
        np.log(magnitude, out=log_chi, where=magnitude > 0)
        total_eta = ext['eta_total'][:self.g.added+1][::-1]; log_eta = np.full_like(total_eta, -np.inf)
        np.log(total_eta, out=log_eta, where=total_eta > 0)
        self._profiles[key] = log_chi, log_eta, np.sign(chi[0])
        if len(self._profiles) > 32:
            self._profiles.popitem(last=False)
        self.profile_cache_peak_entries = max(self.profile_cache_peak_entries, len(self._profiles))
        return self._profiles[key]

    def at(self, radius, columns):
        if self._source.closed or self._field.closed:
            raise ValueError('닫힌 native 계수/동결 J 연결')
        _, columns = request([0], columns, self._source.ns, self._source.nf)
        if len(columns) > 32 or not np.isfinite(radius) or not self.g.radius[-1] <= radius <= self.g.radius[0]:
            raise ValueError('반경/전역ν 블록 범위 오류')
        if radius > self.g.native_radius[0]:
            ascending = self.g.radius[:self.g.added+1][::-1]
            i = int(np.searchsorted(ascending, radius, side='left'))-1
            if not 0 <= i < len(ascending)-1:
                raise ValueError('외부 계수의 원래 절점 범위 오류')
            weight = float(np.log(radius/ascending[i])/np.log(ascending[i+1]/ascending[i]))
            log_chi, log_eta, sign = self._profile(columns)
            chi = sign*np.exp((1-weight)*log_chi[i]+weight*log_chi[i+1])
            eta = np.exp(log_eta[i] if weight == 0 else log_eta[i+1] if weight == 1 else
                         (1-weight)*log_eta[i]+weight*log_eta[i+1])
            return chi, eta
        if radius == self.g.native_radius[-1]:
            chi, es, eta = self._source.read([self._source.ns-1], columns)
            return chi[0], eta[0]+es[0]*self._field.read([self._source.ns-1], columns)[0]
        ascending = self.g.native_radius[::-1]
        i = int(np.searchsorted(ascending, radius))-1
        lo, hi = ascending[i:i+2]; inner = self._source.ns-1-i
        return interior_coefficients(self._source, self._field, inner, inner-1,
                                     float((radius-lo)/(hi-lo)), columns)
