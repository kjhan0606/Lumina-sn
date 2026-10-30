"""실제 미추적 S 분석기를 메모리에서 Si T/ne 단일점 분석으로 변환한다."""
import hashlib
from pathlib import Path
import re

path=Path('scripts/compare_lumina_public_ne_response.py')
raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='ff5fb6b25e0a74f460e59b79cd778386f05792ec036e118f470e9c5e035eb0a6'
s=raw.decode()
s,count=re.subn(r'\b16\b','14',s)
assert count==11,count
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
replace("output['ledger'][1, i]", "output['ledger'][0, i]")
replace('S_III_over_II','Si_III_over_II')
replace('SINGLE_S_RESPONSE','SINGLE_SI_TNE_RESPONSE')
replace("public_ne = float(interp_columns(pub, [velocity], 3)[0])",
        "public_ne = float(interp_columns(pub, [velocity], 3)[0])\n    public_temperature = float(interp_columns(pub, [velocity], 1)[0])")
replace('def point(ne):', 'def point(ne, temperature=None):')
replace("temperature=state['temperature'][i:i+1], ne=np.array([ne]))",
        "temperature=np.array([state['temperature'][i] if temperature is None else temperature]), ne=np.array([ne]))")
replace("for generation, (name, ne_value) in enumerate((('current', output['ne'][i]), ('public_ne', public_ne)), 2):",
        "for generation, (name, ne_value, temperature_value) in enumerate((('current', output['ne'][i], state['temperature'][i]), ('public_T_ne', public_ne, public_temperature)), 2):")
replace('trial = provider.prepare(point(ne_value), material_generation=generation)',
        'trial = provider.prepare(point(ne_value, temperature_value), material_generation=generation)')
replace("full_trial = fallback_provider.prepare({**state, 'ne': full_ne}, material_generation=generation)",
        "full_temperature = state['temperature'].copy(); full_temperature[i] = temperature_value\n        full_trial = fallback_provider.prepare({**state, 'ne': full_ne, 'temperature': full_temperature}, material_generation=generation)")
replace('_, total = cache.integrate(trial, material_generation=generation, radiation_generation=1)',
        "if name == 'current':\n            _, total = cache.integrate(trial, material_generation=generation, radiation_generation=1)\n            reused, direct = cache.last_reused_routes, cache.last_direct_routes\n        else:\n            _, total = provider.integrate(trial, material_generation=generation, radiation_generation=1)\n            reused, direct = 0, len(ids)")
replace("current['ne'][i] = ne_value; current['occupation'] = output['occupation'].copy()",
        "current['ne'][i] = ne_value; current['temperature'][i] = temperature_value; current['occupation'] = output['occupation'].copy()")
replace("results.append(dict(case=name, ne=float(ne_value), fractions=fractions,",
        "results.append(dict(case=name, ne=float(ne_value), temperature_K=float(temperature_value), fractions=fractions,")
replace('bound_reused_routes=cache.last_reused_routes, bound_direct_routes=cache.last_direct_routes',
        'bound_reused_routes=reused, bound_direct_routes=direct')
exec(compile(s,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
