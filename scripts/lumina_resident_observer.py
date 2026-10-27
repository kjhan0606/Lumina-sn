"""상주 전주파수 수송의 출사 광선을 관측자 스펙트럼에 연결한다.

고정 물질·입력 복사장의 유한 수송이다. 공동수렴이나 실제 스펙트럼
동등성을 인증하지 않으며, 초기화만으로 수송을 실행하지 않는다.
"""
import json
from pathlib import Path

import numpy as np

import lumina_observer_spectrum as observer_module
import lumina_resident_field as resident_module
from lumina_gpu_driver import check
from lumina_native_columns import sha
from lumina_observer_spectrum import ObserverDriverPlan, ObserverSpectrum
from lumina_resident_field import array_identity


class ResidentObserver:
    def __init__(self, field):
        plan = field.plan
        if not isinstance(plan, ObserverDriverPlan) or field.cursor or field.failed:
            raise ValueError('새 상주 수송과 관측 광선 계획 필요')
        starts = [(i, path) for i, (kind, path) in enumerate(field.sequence) if kind == 'zero']
        if [path for _, path in starts] != list(range(len(plan.paths))):
            raise ValueError('관측 광선의 수송 순서 불일치')
        self.field = field
        self.bounds = [i for i, _ in starts] + [len(field.sequence)]
        self.spectrum = ObserverSpectrum(plan.source.frequency, plan.observer_impact,
                                        plan.observer_core, plan.outer_radius, plan.observer_doppler)
        self.identity = dict(transport=field.identity, radius=plan.outer_radius,
            code_sha256=dict(observer=sha(Path(__file__)), transform=sha(Path(observer_module.__file__)),
                             resident=sha(Path(resident_module.__file__))),
            observer=array_identity(dict(frequency=self.spectrum.input_frequency,
                                         weights=self.spectrum.weights, doppler=self.spectrum.doppler)))
        self.failed = False

    def advance_path(self, *, deadline):
        k = self.spectrum.next_path
        if self.failed or k >= len(self.bounds)-1 or self.field.cursor != self.bounds[k]:
            raise ValueError('관측 광선의 완료 위치 불일치')
        try:
            self.field.advance(deadline=deadline, max_actions=self.bounds[k+1]-self.bounds[k])
            if self.field.cursor != self.bounds[k+1]:
                raise ValueError('부분 광선은 관측 결과에 포함하지 않는다')
            field = self.field
            check(field.gpu.lib.lumina_fullsr_chain_decode(field.nf, field.ray.address,
                  field.double_ray.address, field.conversions.address, field.rt.status))
            field.rt.sync()
            self.spectrum.add(k, field.rt.download(field.double_ray, field.nf))
        except Exception:
            self.failed = True
            raise

    def finish(self):
        if self.failed or self.field.failed or self.field.cursor != len(self.field.sequence):
            raise ValueError('미완료 수송을 스펙트럼으로 게시하지 않는다')
        result = self.spectrum.finish()
        result.update(physical_intervals=self.field.intervals,
                      source_identity=self.field.input.identity,
                      transport_identity=self.field.identity,
                      scattering_convergence_certified=False,
                      final_goal='NOT_MET', accepted_physical_spectrum=False)
        return result

    def checkpoint(self, root):
        k = self.spectrum.next_path
        if self.failed or self.field.failed or self.field.cursor != self.bounds[k]:
            raise ValueError('완성된 광선 경계에서만 저장할 수 있다')
        root = Path(root).resolve(); root.mkdir(exist_ok=False)
        self.field.checkpoint(root/'field')
        archive = root/'observer.npz'
        with archive.open('xb') as stream:
            np.savez(stream, total=self.spectrum.total, correction=self.spectrum.correction)
        metadata = dict(identity=self.identity, completed_paths=k, archive_sha256=sha(archive))
        with (root/'observer.json').open('x') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return metadata

    def restore(self, root):
        if self.failed or self.field.cursor or self.spectrum.next_path:
            raise ValueError('새 관측 수송에서만 복원할 수 있다')
        root = Path(root).resolve(); metadata = json.loads((root/'observer.json').read_text())
        k = metadata['completed_paths']
        if (metadata['identity'] != self.identity or type(k) is not int or
                not 0 <= k < len(self.bounds) or sha(root/'observer.npz') != metadata['archive_sha256']):
            raise ValueError('관측 저장 자료의 신원 또는 광선 위치 불일치')
        with np.load(root/'observer.npz', allow_pickle=False) as data:
            total, correction = data['total'], data['correction']
        if (total.shape != self.spectrum.total.shape or correction.shape != total.shape or
                total.dtype != np.dtype('f8') or correction.dtype != total.dtype or
                not np.isfinite(total).all() or not np.isfinite(correction).all() or np.any(total < 0)):
            raise ValueError('관측 저장 배열 오류')
        try:
            self.field.restore(root/'field')
            if self.field.cursor != self.bounds[k]:
                raise ValueError('관측과 수송의 완료 위치 불일치')
        except Exception:
            self.failed = True
            raise
        self.spectrum.total[:] = total; self.spectrum.correction[:] = correction
        self.spectrum.next_path = k
