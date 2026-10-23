"""실제 native 계수의 자체 J 재결합 경로. 물리 수송·수렴 시험은 아니다."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import lumina_outer_atmosphere as A
import lumina_native_columns as N
from lumina_compiled_input import CompiledInput


def test_actual_native90_rebind_reuses_static_source(tmp_path):
    config = os.environ.get('LUMINA_NATIVE_SOURCE_CONFIG')
    prep = os.environ.get('LUMINA_NATIVE_SOURCE_PREPARATION')
    digest = os.environ.get('LUMINA_NATIVE_SOURCE_PREPARATION_SHA256')
    if config is None and prep is None and digest is None:
        pytest.skip('실제 봉인 입력을 지정한 경우에만 실행')
    assert config and prep and digest
    library = tmp_path/'input.so'
    subprocess.run([os.environ.get('CC','cc'), '-O2', '-std=c11', '-Wall',
                    '-Wextra', '-Werror', '-fPIC', '-shared',
                    str(ROOT/'src/cmf_nodal_input.c'), '-lm', '-o', str(library)],
                   check=True, capture_output=True, timeout=60)
    with N.open_sealed_native(config, prep, digest) as source:
        preparation = json.loads(Path(prep).read_text())
        geometry_path = Path(prep).parent/'geometry.npz'
        assert N.sha(geometry_path) == preparation['output_sha256']['geometry.npz']
        with np.load(geometry_path, allow_pickle=False) as data:
            g = A.OuterGeometry(data['radius'].copy(), data['velocity'].copy(),
                data['sigma'].copy(), len(data['radius'])-len(data['native_radius']),
                tuple(data['reference_indices']), data['native_radius'].copy())
        assert source.ns == 90 and source.nf == 1115020

        class SourceRebindTrial:
            # 실제 수송 결과나 원 CMFGEN J가 아닌 비상수 재결합 전용 시험장.
            ns, nf = source.ns, source.nf
            radius, frequency = source.radius, source.frequency
            closed = False
            amplitude = 0.
            identity = dict(role='EXPLICIT_NATIVE90_SOURCE_REBIND_TRIAL_NOT_TRANSPORT')

            def read(self, rows, columns):
                rows, columns = N.request(rows, columns, self.ns, self.nf)
                return self.amplitude*1e-8*(.2+rows[:,None]/self.ns)*(
                    1.+.25*columns[None,:]/self.nf)

        trial = SourceRebindTrial()
        result = []
        with CompiledInput(library, g, source, trial,
                           slab_width=4, cache_rows=128, profiles=8) as provider:
            for wavelength in (300.,715.,1500.):
                center = int(np.argmin(abs(source.frequency-2.99792458e18/wavelength)))
                base = (center//4)*4
                assert base+4 <= source.nf
                integer = np.array([base+1,base+2], dtype='i8')
                columns = np.array([base+1.,base+1.375,base+2.])
                fraction = columns-(base+1)
                rows = np.arange(source.ns)
                chi, es, eta = provider.raw(rows, integer)
                coefficients_before = provider.report()['source_bytes']
                values = []
                for amplitude in (0.,1.):
                    trial.amplitude = amplitude
                    trial.identity = dict(role='EXPLICIT_NATIVE90_SOURCE_REBIND_TRIAL_NOT_TRANSPORT',
                                          amplitude=amplitude)
                    provider.bind(base,4)
                    actual_chi, actual_eta = provider.at(source.radius, columns)
                    expected_eta = eta+es*trial.read(rows, integer)
                    for actual, exact in ((actual_chi,chi), (actual_eta,expected_eta)):
                        expected = exact[:,0,None]+fraction*(exact[:,1,None]-exact[:,0,None])
                        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
                    assert provider.report()['source_bytes'] == coefficients_before
                    values.append(actual_eta)
                changed = int(np.count_nonzero(values[0] != values[1]))
                assert changed > 0
                result.append(dict(wavelength_A=wavelength, native_nodes=source.ns,
                    columns=columns.tolist(), changed_eta_values=changed,
                    static_coefficient_bytes_added_after_rebind=0,
                    field_role=trial.identity['role'], physical_F_calls=0))
        report = dict(native90_source_rebind=result, source_identity=source.identity,
            library_sha256=N.sha(library),
            code_sha256={str(p):N.sha(p) for p in (Path(__file__),
                ROOT/'scripts/lumina_compiled_input.py', ROOT/'src/cmf_nodal_input.c')},
            actual_model_F_calls=0, new_SE_RE_updates=0, final_goal='NOT_MET')
    (tmp_path/'native90-source-rebind.json').write_text(json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
