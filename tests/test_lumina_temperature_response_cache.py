"""온도별 재사용 연결을 실제 BF 커널로 검증한다. 새 SE/RT 시험은 아니다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_temperature_response_cache import TemperatureResponseCache, CAPACITY, extra_memory_reserve
from lumina_model_material_rates import FixedTemperatureBoundRates
from lumina_native_columns import sha
from test_model_material_rates import library, example
import lumina_temperature_response_cache as module


class RateResponse:
    """ModelResponse와 같은 적분/저장 호출만 사용한다. 개체수 해는 만들지 않는다."""
    def __init__(self, provider, state):
        self.provider = provider; self.state = self.seed = state; self.generation = 1; self.fail = False
        trial = provider.prepare(state, material_generation=1)
        bound, _ = provider.integrate(trial, material_generation=1, radiation_generation=7)
        self.bound_cache = FixedTemperatureBoundRates(provider, trial, bound)

    def evaluate(self, temperature, ne, *, save=None):
        self.generation += 1
        trial = self.provider.prepare({**self.state, 'temperature':temperature, 'ne':ne},
                                      material_generation=self.generation)
        fixed = np.array_equal(trial.state['temperature'], self.bound_cache.temperature)
        owner = self.bound_cache if fixed else self.provider
        bound, total = owner.integrate(trial, material_generation=self.generation, radiation_generation=7)
        if self.fail: raise RuntimeError('시험용 SE 이후 실패')
        metrics = dict(bound_reused_routes=self.bound_cache.last_reused_routes if fixed else 0,
            bound_direct_routes=self.bound_cache.last_direct_routes if fixed else len(bound))
        data = dict(bound=bound, temperature=trial.state['temperature'], ne=trial.state['ne'])
        if save is not None: save(data, metrics, trial, total)
        return data, metrics, trial, total


@pytest.fixture
def scenario(library, monkeypatch):
    provider, state = example(library)
    response = RateResponse(provider, state)
    original = provider.integrate; calls = []
    def spy(trial, **kwargs):
        calls.append(dict(full=kwargs.get('route_indices') is None,
                          temperature=trial.state['temperature'].copy()))
        return original(trial, **kwargs)
    monkeypatch.setattr(provider, 'integrate', spy)
    return response, state, calls, original, spy


def direct_check(response, actual, original):
    data, _, trial, total = actual
    expected = original(trial, material_generation=trial.material_generation,
                        radiation_generation=trial.radiation_generation)
    np.testing.assert_allclose(data['bound'], expected[0], rtol=5.e-12, atol=0)
    np.testing.assert_allclose(total, expected[1], rtol=5.e-12, atol=0)


def test_new_temperature_is_integrated_once_then_ne_changes_reuse_exact_bound(scenario):
    response, state, calls, original, spy = scenario
    cache = TemperatureResponseCache(response)
    t = state['temperature']*np.array([.91, 1.03, 1.06])
    for index, factor in enumerate((1., 1.001, .999, 1.0001)):
        actual = cache.evaluate(t, state['ne']*factor)
        direct_check(response, actual, original)
        metrics = actual[1]['temperature_bound_cache']
        assert metrics['hit'] == (index != 0)
        assert metrics['new_table'] == (index == 0)
        if index:
            assert actual[1]['bound_reused_routes'] > 0
            assert actual[1]['full_BF_reintegration'] is False
        assert response.provider.integrate is spy and not cache.active
    assert sum(c['full'] for c in calls) == 1
    assert cache.report()['cache_hits'] == 3
    assert cache.report()['new_temperature_tables'] == 1


def test_original_T_is_not_lost_after_other_temperatures_and_capacity_is_bounded(scenario):
    response, state, calls, original, _ = scenario
    cache = TemperatureResponseCache(response)
    seed = cache.seed.values.copy()
    for factor in (1.01, 1.02, 1.03, 1.04, 1.05):
        cache.evaluate(state['temperature']*factor, state['ne'])
        assert cache.report()['retained_temperatures'] <= CAPACITY
    before = sum(c['full'] for c in calls)
    actual = cache.evaluate(state['temperature'], state['ne']*1.001)
    direct_check(response, actual, original)
    assert sum(c['full'] for c in calls) == before
    np.testing.assert_array_equal(cache.seed.values, seed)
    assert cache.seed_key in cache.entries


def test_new_temperature_does_not_use_close_but_different_T_table(scenario):
    response, state, calls, _, _ = scenario
    cache = TemperatureResponseCache(response)
    t = state['temperature']*1.01
    cache.evaluate(t, state['ne'])
    changed = t.copy(); changed[1] = np.nextafter(changed[1], np.inf)
    cache.evaluate(changed, state['ne'])
    assert sum(c['full'] for c in calls) == 2


def test_persistence_keeps_real_bound_reference_ne_and_prefactor_not_a_solution(scenario, tmp_path):
    response, state, calls, original, _ = scenario
    cache = TemperatureResponseCache(response, out=tmp_path/'cache')
    t = state['temperature']*1.03
    actual = cache.evaluate(t, state['ne'])
    publication = actual[1]['temperature_bound_cache']['saved_bound_table']
    path = Path(publication['path']); assert sha(path) == publication['sha256']
    receipt = json.loads(path.read_text())
    assert receipt['state'] == 'COMPLETE_BOUND_TABLE_NOT_MATERIAL_ACCEPTANCE'
    assert receipt['parent_material_response_required']
    assert receipt['new_SE_solves'] == receipt['new_transport'] == 0
    assert sha(receipt['bound']['path']) == receipt['bound']['sha256']
    with np.load(receipt['bound']['path'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['bound'], actual[0]['bound'])
        np.testing.assert_array_equal(saved['temperature'], t)
        np.testing.assert_array_equal(saved['reference_ne'], state['ne'])
        np.testing.assert_array_equal(saved['milne_prefactor'], actual[2].milne_prefactor)
    subsequent = cache.evaluate(t, state['ne']*.999)
    direct_check(response, subsequent, original)
    assert subsequent[1]['temperature_bound_cache']['saved_bound_table'] == publication
    assert len(list((tmp_path/'cache').iterdir())) == 2


def test_failures_preserve_completed_integrals_without_publishing_a_material_solution(scenario, tmp_path):
    response, state, calls, original, spy = scenario
    cache = TemperatureResponseCache(response, out=tmp_path/'cache')
    response.fail = True
    with pytest.raises(RuntimeError): cache.evaluate(state['temperature']*1.01, state['ne'])
    assert response.provider.integrate is spy and not cache.active
    assert len(list((tmp_path/'cache').iterdir())) == 2
    assert cache.report()['evaluations'] == 0
    assert cache.report()['new_temperature_tables'] == 1
    receipt = cache.report()['completed_bound_tables'][0]
    saved = json.loads(Path(receipt['path']).read_text())
    assert saved['state'] == 'COMPLETE_BOUND_TABLE_NOT_MATERIAL_ACCEPTANCE'
    response.fail = False
    cache.evaluate(state['temperature']*1.02, state['ne'])
    actual = cache.evaluate(state['temperature']*1.01, state['ne'])
    direct_check(response, actual, original)
    assert actual[1]['temperature_bound_cache']['hit']
    assert len(list((tmp_path/'cache').iterdir())) == 4


def test_failed_integration_has_no_completed_table_and_restores_the_original_method(scenario, tmp_path):
    response, state, _, _, _ = scenario
    def failed(*args, **kwargs): raise RuntimeError('미완료 적분')
    response.provider.integrate = failed
    cache = TemperatureResponseCache(response, out=tmp_path/'cache')
    with pytest.raises(RuntimeError, match='미완료'):
        cache.evaluate(state['temperature']*1.01, state['ne'])
    assert response.provider.integrate is failed and not cache.active
    assert cache.report()['new_temperature_tables'] == 0
    assert not list((tmp_path/'cache').iterdir())


def test_optional_cache_output_failure_is_reported_but_does_not_block_material_evaluation(scenario, tmp_path, monkeypatch):
    response, state, _, original, _ = scenario
    cache = TemperatureResponseCache(response, out=tmp_path/'cache')
    def failed(*args, **kwargs): raise OSError('시험용 캐시 출력 실패')
    monkeypatch.setattr(module.np, 'savez', failed)
    actual = cache.evaluate(state['temperature']*1.01, state['ne'])
    direct_check(response, actual, original)
    info = actual[1]['temperature_bound_cache']
    assert info['saved_bound_table'] is None and info['evaluations'] == 1
    assert len(info['bound_publication_errors']) == 1
    assert not info['completed_bound_tables']


@pytest.mark.parametrize('fault', ['route', 'field'])
def test_field_and_route_checks_are_still_owned_by_existing_rate_cache(scenario, fault):
    response, state, _, _, spy = scenario
    cache = TemperatureResponseCache(response)
    if fault == 'route': response.provider.routes[0].probability *= .9
    else: response.provider.field_signature = 'different field'
    with pytest.raises(ValueError, match='원자 경로'):
        cache.evaluate(state['temperature'], state['ne'])
    assert response.provider.integrate is spy and not cache.active


def test_invalid_temperature_cannot_leave_an_active_context(scenario):
    response, _, _, _, spy = scenario
    cache = TemperatureResponseCache(response)
    with pytest.raises(ValueError): cache.evaluate(['invalid'], [1.])
    assert not cache.active and response.provider.integrate is spy


def test_class_method_is_restored_without_leaving_an_instance_override(scenario):
    response, state, _, _, _ = scenario
    del response.provider.integrate
    cache = TemperatureResponseCache(response)
    cache.evaluate(state['temperature']*1.01, state['ne'])
    assert 'integrate' not in response.provider.__dict__ and not cache.active


def test_existing_output_is_not_overwritten(scenario, tmp_path):
    tmp_path.joinpath('cache').mkdir()
    with pytest.raises(FileExistsError): TemperatureResponseCache(scenario[0], out=tmp_path/'cache')


def test_memory_reserve_covers_retained_arrays_and_construction_workspace(scenario):
    response, state, _, _, _ = scenario
    cache = TemperatureResponseCache(response)
    initial = cache.report()['resident_array_bytes']
    for factor in (1.01, 1.02, 1.03): cache.evaluate(state['temperature']*factor, state['ne'])
    assert cache.report()['resident_array_bytes']-initial < extra_memory_reserve(4, 3, 4)


def saved_then_reopened(library, tmp_path):
    provider, state = example(library)
    first = TemperatureResponseCache(RateResponse(provider, state), out=tmp_path/'old-cache')
    temperature = state['temperature']*1.02
    saved = first.evaluate(temperature, state['ne'])[1]['temperature_bound_cache']['saved_bound_table']
    new_provider, new_state = example(library)
    owner = RateResponse(new_provider, new_state)
    receiver = TemperatureResponseCache(owner, out=tmp_path/'new-cache')
    inputs = {}
    receiver.register([saved], inputs=inputs)
    return receiver, new_state, temperature, saved, inputs


def test_saved_table_is_reconstructed_in_a_new_provider_and_reused_without_full_integral(library, tmp_path):
    cache, state, t, receipt, inputs = saved_then_reopened(library, tmp_path)
    assert cache.report()['restored_temperature_tables'] == 0
    p = cache.provider; original = p.integrate; calls = []
    def spy(trial, **kwargs):
        calls.append(kwargs.get('route_indices') is None)
        return original(trial, **kwargs)
    p.integrate = spy
    actual = cache.evaluate(t, state['ne']*1.001)
    direct_check(cache.response, actual, original)
    assert not any(calls)
    assert actual[1]['temperature_bound_cache']['hit']
    assert cache.report()['new_temperature_tables'] == 0
    assert cache.report()['restored_temperature_tables'] == 1
    assert receipt in cache.report()['completed_bound_tables']
    record = json.loads(Path(receipt['path']).read_text())
    assert inputs[record['bound']['path']] == record['bound']['sha256']
    assert not list((tmp_path/'new-cache').iterdir())


def test_unrequested_saved_body_is_not_read_but_requested_corruption_is_rejected(library, tmp_path):
    cache, state, t, receipt, inputs = saved_then_reopened(library, tmp_path)
    record = json.loads(Path(receipt['path']).read_text())
    Path(record['bound']['path']).write_bytes(b'corrupt')
    cache.evaluate(t*1.01, state['ne'])
    assert cache.report()['restored_temperature_tables'] == 0
    assert record['bound']['path'] not in inputs
    with pytest.raises(ValueError, match='본문 변경'): cache.evaluate(t, state['ne'])
    assert not cache.active


@pytest.mark.parametrize('fault', ['route', 'field'])
def test_changes_after_registration_cannot_bypass_restore_identity(library, tmp_path, fault):
    cache, state, t, _, _ = saved_then_reopened(library, tmp_path)
    if fault == 'route': cache.provider.routes[0].probability *= .9
    else: cache.provider.field_signature = 'changed'
    with pytest.raises(ValueError, match='장/원자 경로 변경'): cache.evaluate(t, state['ne'])
    assert not cache.active and cache.report()['restored_temperature_tables'] == 0


def test_reference_ne_and_milne_prefactor_must_reconstruct_together(library, tmp_path):
    cache, state, t, receipt, _ = saved_then_reopened(library, tmp_path)
    path = Path(receipt['path']); record = json.loads(path.read_text())
    body = Path(record['bound']['path'])
    with np.load(body, allow_pickle=False) as values:
        arrays = {k:values[k] for k in values.files}
    arrays['reference_ne'] *= 1.01
    np.savez(body, **arrays); record['bound']['sha256'] = sha(body)
    path.write_text(json.dumps(record)); replacement = dict(path=str(path), sha256=sha(path))
    cache.register([replacement], inputs={})
    with pytest.raises(ValueError, match='prefactor 재구성'): cache.evaluate(t, state['ne'])
    assert not cache.active


def test_wrong_saved_header_is_not_a_valid_material_input(library, tmp_path):
    cache, _, _, receipt, _ = saved_then_reopened(library, tmp_path)
    bad = dict(receipt, sha256='0'*64)
    with pytest.raises(ValueError, match='기록 변경'): cache.register([bad], inputs={})


def test_charge_projection_requires_an_exact_completed_temperature_without_material_work(library,tmp_path):
    cache,state,t,_,_=saved_then_reopened(library,tmp_path)
    cache.require_completed(t)
    assert cache.report()['restored_temperature_tables']==1
    assert cache.report()['evaluations']==cache.report()['new_temperature_tables']==0
    changed=t.copy();changed[0]=np.nextafter(changed[0],np.inf)
    with pytest.raises(ValueError,match='새 전적분 금지'):cache.require_completed(changed)
    assert cache.report()['evaluations']==0


def test_projection_cannot_change_T_or_fall_back_to_a_whole_bound_integral(library,tmp_path):
    cache,state,t,_,_=saved_then_reopened(library,tmp_path)
    cache.require_completed(t)
    with pytest.raises(ValueError,match='온도 변경 금지'):cache.evaluate(t*1.001,state['ne'])
    with pytest.raises(ValueError,match='새 전적분 금지'):cache.evaluate(t,state['ne']*3.)
    assert not cache.active and cache.report()['evaluations']==0
