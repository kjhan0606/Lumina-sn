"""완료한 온도별 bound 적분을 기존 고정온도 재인수로만 재사용한다.

새 온도는 여전히 전적분한다. occupation·pseudo 항과 재사용 불가 경로는
기존 생산자가 갱신한다. 실제 SE·열 평가나 평가 횟수를 생략하지 않는다.
"""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from lumina_model_material_rates import FixedTemperatureBoundRates, owned
from lumina_native_columns import sha

CAPACITY = 4


def extra_memory_reserve(routes, points, full_levels):
    # 기준 캐시 외3개와 교체 중 새 캐시1개: bound6+Milne1, T/ne와 fallback.
    # 퇴출 전 생성 시 역률4개 선택·절댓값과 비교 임시값도 별도 예약한다.
    # 읽기 복원 시 원 bound/pref와 참조 trial도 일시적으로 함께 존재한다.
    return CAPACITY*(7*routes*points*8+2*points*8+routes)+(20*routes+3*full_levels)*points*8


class TemperatureResponseCache:
    def __init__(self, response, *, out=None):
        self.response = response
        self.provider = response.provider
        self.seed = response.bound_cache
        self.seed_key = self._key(self.seed.temperature)
        self.entries = OrderedDict({self.seed_key: dict(cache=self.seed, reference_ne=None, saved=None, index=0)})
        self.out = None if out is None else Path(out)
        if self.out is not None:
            self.out.mkdir(exist_ok=False)
        self.evaluations = self.hits = self.new_tables = 0
        self.published_tables = []
        self.publication_errors = []
        self.external = {}
        self.restored_tables = []
        self.inputs = None
        self.layout = None
        self.required_temperature = None
        self.active = False

    @staticmethod
    def _key(temperature):
        value = np.asarray(temperature, dtype='f8')
        return value.shape, value.tobytes()

    def _remember(self, key, entry):
        self.entries[key] = entry
        self.entries.move_to_end(key)
        while len(self.entries) > CAPACITY:
            oldest = next(k for k in self.entries if k != self.seed_key)
            del self.entries[oldest]

    def _layout(self, *, fresh=False):
        # 함수 주소는 프로세스마다 다르다. 함수/원자료 신원은 상위 재개가 검증한다.
        if self.layout is None or fresh:
            p = self.provider; h = hashlib.sha256()
            h.update(repr((p.rows, p.models, p.dissolved)).encode())
            for r in p.routes:
                h.update(repr((r.Z, r.stage, r.lower, r.target, r.target_superlevel, r.phot_grid,
                    r.probability, r.target_g, r.target_energy_eV, r.threshold_Hz,
                    r.dissolution_primary, [(start, kind) for _, _, start, kind in r.models])).encode())
                for _, nodes, _, _ in r.models:
                    h.update(np.asarray(nodes, dtype='f8').tobytes())
            for a in (p.edge, p.z, p.primary, p.sigma_edge): h.update(a.tobytes())
            self.layout = h.hexdigest()
        return self.layout

    def register(self, receipts, *, inputs):
        """상위 열재개가 장·원 입력·물리 구현을 검증한 뒤 호출한다. 적분은 안 한다."""
        self.inputs = inputs
        for receipt in receipts:
            path = Path(receipt['path'])
            if sha(path) != receipt['sha256']:
                raise ValueError('보존 온도 적분 기록 변경')
            record = json.loads(path.read_text())
            if (record.get('state') != 'COMPLETE_BOUND_TABLE_NOT_MATERIAL_ACCEPTANCE' or
                    record.get('field_signature') != self.provider.field_signature or
                    record.get('route_layout_sha256') != self._layout() or
                    record.get('routes') != len(self.provider.routes) or
                    record.get('points') != self.provider.n_points or
                    record.get('rate_implementation_sha256') != sha(Path(__file__).with_name('lumina_model_material_rates.py'))):
                raise ValueError('보존 온도 적분의 장/원자 경로/물리 생산자 불일치')
            self.external[record['temperature_sha256']] = (record, receipt)
            inputs[str(path)] = receipt['sha256']

    def _restore(self, temperature, key):
        digest = hashlib.sha256(np.asarray(temperature, dtype='f8').tobytes()).hexdigest()
        if digest not in self.external: return
        record, receipt = self.external[digest]
        if sha(Path(receipt['path'])) != receipt['sha256']:
            raise ValueError('보존 온도 적분 등록 후 기록 변경')
        if (record['field_signature'] != self.provider.field_signature or
                record['route_layout_sha256'] != self._layout(fresh=True)):
            raise ValueError('보존 온도 적분 소비 전 장/원자 경로 변경')
        path = Path(record['bound']['path'])
        if sha(path) != record['bound']['sha256']:
            raise ValueError('보존 온도 bound 적분 본문 변경')
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != {'bound', 'temperature', 'reference_ne', 'milne_prefactor'}:
                raise ValueError('보존 온도 bound 배열 목록 불일치')
            arrays = {k:values[k] for k in values.files}
            if any(value.dtype != np.dtype('f8') for value in arrays.values()):
                raise ValueError('보존 온도 bound 배열 dtype 불일치')
            t, ne, pref = (arrays[k] for k in ('temperature', 'reference_ne', 'milne_prefactor'))
            if not np.array_equal(t, temperature):
                raise ValueError('보존 온도 적분의 실제 T 불일치')
            trial = self.provider.prepare({**self.response.seed, 'temperature':t, 'ne':ne},
                                           material_generation=self.response.generation)
            if (pref.shape != trial.milne_prefactor.shape or not np.isfinite(pref).all() or
                    np.any(pref < 0)):
                raise ValueError('보존 온도 Milne prefactor 형상/값 불일치')
            # 실제 적분 당시 pref를 복원한다. 정보 손실 경로는 기존 캐시가 직접 계산한다.
            restored = replace(trial, milne_prefactor=owned(pref))
            cache = FixedTemperatureBoundRates(self.provider, restored, arrays['bound'])
            if not np.allclose(pref[~cache.fallback], trial.milne_prefactor[~cache.fallback],
                               rtol=8*np.finfo('f8').eps, atol=0):
                raise ValueError('보존 온도 Milne prefactor 재구성 불일치')
        self.inputs[str(path)] = record['bound']['sha256']
        self._remember(key, dict(cache=cache, reference_ne=ne, saved=receipt, index=None))
        self.restored_tables.append(receipt)

    def _publish(self, entry):
        if self.out is None or entry['saved'] is not None:
            return entry['saved']
        cache = entry['cache']
        path = self.out/('bound-temperature-'+str(entry['index'])+'.npz')
        with path.open('xb') as stream:
            np.savez(stream, bound=cache.values, temperature=cache.temperature,
                     reference_ne=entry['reference_ne'], milne_prefactor=cache.pref)
        manifest = dict(state='COMPLETE_BOUND_TABLE_NOT_MATERIAL_ACCEPTANCE',
            field_signature=cache.field_signature, routes=len(self.provider.routes),
            route_layout_sha256=self._layout(),
            points=self.provider.n_points, temperature_sha256=hashlib.sha256(cache.temperature.tobytes()).hexdigest(),
            bound=dict(path=str(path.resolve()), sha256=sha(path)),
            implementation_sha256=sha(Path(__file__)),
            rate_implementation_sha256=sha(Path(__file__).with_name('lumina_model_material_rates.py')),
            parent_material_response_required=True, new_SE_solves=0, new_transport=0)
        receipt = path.with_suffix('.json')
        with receipt.open('x') as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
        entry['saved'] = dict(path=str(receipt.resolve()), sha256=sha(receipt))
        self.published_tables.append(entry['saved'])
        return entry['saved']

    def report(self):
        def size(entry):
            c = entry['cache']
            return sum(a.nbytes for a in (c.values, c.pref, c.temperature, c.fallback)) + (
                0 if entry['reference_ne'] is None else entry['reference_ne'].nbytes)
        return dict(evaluations=self.evaluations, cache_hits=self.hits,
            new_temperature_tables=self.new_tables, retained_temperatures=len(self.entries),
            capacity=CAPACITY, resident_array_bytes=sum(map(size, self.entries.values())),
            completed_bound_tables=self.published_tables+self.restored_tables,
            restored_temperature_tables=len(self.restored_tables),
            bound_publication_errors=self.publication_errors.copy())

    def require_completed(self, temperature):
        """같은 T 전하 보정에서 캐시 누락을 새 전적분으로 바꾸지 않는다."""
        key = self._key(temperature)
        if key not in self.entries:
            self._restore(temperature, key)
        if key not in self.entries:
            raise ValueError('고정 T 전하 보정의 완료 bound 자료 없음; 새 전적분 금지')
        self.required_temperature = key

    def evaluate(self, temperature, ne, *, save=None):
        if self.active or self.response.provider is not self.provider:
            raise ValueError('온도 적분 캐시는 같은 순차 물질 생산자에서만 사용한다')
        key = self._key(temperature)
        if self.required_temperature is not None and key != self.required_temperature:
            raise ValueError('고정 T 전하 보정의 온도 변경 금지')
        if key not in self.entries:
            self._restore(temperature, key)
        self.active = True
        hit = key in self.entries
        entry = self.entries.get(key)
        self.response.bound_cache = self.seed if entry is None else entry['cache']
        if hit:
            self.entries.move_to_end(key)
        original = self.provider.integrate
        had_override = 'integrate' in self.provider.__dict__
        captured = []

        def integrate(trial, **kwargs):
            ids = kwargs.get('route_indices')
            if self.required_temperature is not None and (ids is None or len(ids) == len(self.provider.routes)):
                raise ValueError('고정 T 전하 보정에서 새 전적분 금지')
            result = original(trial, **kwargs)
            if kwargs.get('route_indices') is None:
                if self._key(trial.state['temperature']) != key:
                    raise ValueError('완료 적분의 요청 온도 불일치')
                cache = FixedTemperatureBoundRates(self.provider, trial, result[0])
                self.new_tables += 1
                value = dict(cache=cache, reference_ne=trial.state['ne'].copy(), saved=None,
                             index=self.new_tables)
                self._remember(key, value)
                captured.append(value)
                # SE가 뒤에서 실패해도 완료된 비싼 적분은 별도 보존한다.
                # 선택적 캐시 보존 실패를 새 물리 실행 게이트로 만들지 않는다.
                try:
                    self._publish(value)
                except OSError as error:
                    self.publication_errors.append(dict(table_index=value['index'],
                        error=type(error).__name__+': '+str(error)))
            return result

        def record(data, metrics, trial, rates):
            self.evaluations += 1
            self.hits += int(hit)
            selected = captured[-1] if captured else entry
            if selected is None:
                raise ValueError('새 온도의 완료 bound 적분 누락')
            publication = selected['saved']
            metrics['full_BF_reintegration'] = bool(captured or
                metrics['bound_direct_routes'] == len(self.provider.routes))
            metrics['temperature_bound_cache'] = dict(hit=hit, new_table=bool(captured),
                temperature_sha256=hashlib.sha256(selected['cache'].temperature.tobytes()).hexdigest(),
                saved_bound_table=publication, **self.report())
            if save is not None:
                save(data, metrics, trial, rates)

        self.provider.integrate = integrate
        try:
            return self.response.evaluate(temperature, ne, save=record)
        finally:
            if had_override:
                self.provider.integrate = original
            else:
                del self.provider.integrate
            self.active = False
