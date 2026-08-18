#!/usr/bin/env python3
"""SUB_PHOT_GEN validity prescan over the levels CMFGEN_FULL_LEVELS newly exposes.

Rules transcribed from /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/phot_prescan.py
(itself transcribed from newsubs/{sub_phot_gen,rdphot_gen_v2,rd_hyd_bf_data}.f),
imported here rather than re-typed.  The file selection / level cap / phot->level
matching is the BAKER's own code path (expand_atomic_data_cmfgen.parse_all_ions),
so what is scanned is exactly what the bake will evaluate.
"""
import importlib.util
import os
import sys

os.environ['CMFGEN_FULL_LEVELS'] = '1'
os.environ.setdefault('CMFGEN_SUPER_LEVELS', '1')
os.environ.setdefault('CMFGEN_EXACT_HYD', '1')
os.environ.setdefault('CMFGEN_VINTAGE_MATCH', 'phot')
os.environ.setdefault('CMFGEN_VINTAGE_PHOT_DROP', '28:2')

sys.path.insert(0, '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts')
import expand_atomic_data_cmfgen as B  # noqa: E402

spec = importlib.util.spec_from_file_location(
    'phot_prescan', '/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/phot_prescan.py')
PS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PS)

# (Z, stage) -> (old cap in the bakefix4b atom, label)
TARGETS = {
    (20, 3): (200, 'Ca III'), (20, 5): (200, 'Ca V'),
    (26, 4): (200, 'Fe IV'),  (26, 5): (200, 'Fe V'),
    (27, 4): (200, 'Co IV'),  (28, 4): (200, 'Ni IV'),
    (14, 5): (0, 'Si V'),     (26, 6): (0, 'Fe VI'),
    (27, 5): (0, 'Co V'),     (27, 6): (0, 'Co VI'),
    (28, 5): (0, 'Ni V'),     (28, 6): (0, 'Ni VI'),
}

B.ION_LEVEL_CAPS = {k: B.ION_LEVEL_CAPS[k] for k in TARGETS}
data = B.parse_all_ions()

print('\n' + '=' * 78)
print('PRESCAN over newly exposed levels')
print('=' * 78)
tot_viol = 0
rows = []
for (Z, stage), (old_cap, lab) in sorted(TARGETS.items()):
    d = data.get((Z, stage))
    if d is None:
        print(f'{lab:7s}: NOT PARSED')
        continue
    levs = d['levels']
    n_kept = d['n_kept']
    cfg_to_lvl, term_to_lvls = {}, {}
    for k in range(n_kept):
        cfg_to_lvl[B._norm_cfg(levs['config'][k])] = k
        term_to_lvls.setdefault(B._term_cfg(levs['config'][k]), []).append(k)

    phot = d['phot']
    E_ion = d['osc'].ionization_eV
    zion = float(d['osc'].z_screen)
    covered = set()
    viols = []
    tags = {}
    if phot is not None and phot.entries:
        for e in phot.entries:
            c = B._norm_cfg(e.config)
            tg = [cfg_to_lvl[c]] if c in cfg_to_lvl else \
                 term_to_lvls.get(B._term_cfg(e.config), [])
            new_tg = [t for t in tg if t >= old_cap]
            if not new_tg:
                continue
            covered.update(new_tg)
            v = []
            PS.check_term({'name': e.config, 'type': e.cs_type,
                           'params': list(e.sigma_Mb) if e.cs_type < 20 else
                           [x for p in zip(e.energy, e.sigma_Mb) for x in p],
                           'npnts': (len(e.sigma_Mb) if e.cs_type < 20
                                     else e.n_points),
                           'line': 0}, v, lab)
            for rule, src, det in v:
                viols.append((rule, src, det, e.config, len(new_tg)))
            # what the baker's certified evaluator actually does with it
            for t in new_tg[:1]:
                E_lvl = float(levs['E_cm'][t]) * 1.239841984e-4
                E_th = E_ion - E_lvl
                if E_th <= 0:
                    tags['E_thresh<=0'] = tags.get('E_thresh<=0', 0) + 1
                    continue
                nu_th = E_th * B.EV_TO_ERG / B.H_CGS
                if nu_th >= 3.0e16:
                    tags['nu_th>grid'] = tags.get('nu_th>grid', 0) + 1
                    continue
                nef = B._cmfgen_nef(Z, zion, nu_th)
                m = B._sigma_model(e.cs_type, e.energy, e.sigma_Mb, nu_th,
                                   zion=zion, nef=nef)
                tag = 'NONE(no sigma)' if m is None else m[3]
                tags[tag] = tags.get(tag, 0) + len(new_tg)
    n_new = n_kept - old_cap
    n_nophot = n_new - len(covered)
    rows.append((lab, Z, stage, old_cap, n_kept, n_new, len(covered),
                 n_nophot, len(viols)))
    tot_viol += len(viols)
    print(f'\n--- {lab} (Z={Z}, stage={stage}) osc={d["date"]} '
          f'levels {old_cap} -> {n_kept} (new {n_new})')
    print(f'    new levels with a phot entry: {len(covered)}   without: {n_nophot}')
    print(f'    baker evaluator tags on new levels: {dict(sorted(tags.items()))}')
    if viols:
        print(f'    *** {len(viols)} VALIDITY VIOLATIONS ***')
        for r in viols[:20]:
            print(f'      [{r[0]}] {r[3]!r} ({r[4]} lvl)  {r[1]}  {r[2]}')
    else:
        print('    validity: 0 violations')

print('\n' + '=' * 78)
print(f'{"ion":8s} {"Z":>3s} {"st":>3s} {"old":>5s} {"new":>5s} {"+lvl":>6s} '
      f'{"phot":>6s} {"nophot":>7s} {"viol":>5s}')
for r in rows:
    print(f'{r[0]:8s} {r[1]:3d} {r[2]:3d} {r[3]:5d} {r[4]:5d} {r[5]:6d} '
          f'{r[6]:6d} {r[7]:7d} {r[8]:5d}')
print(f'TOTAL VIOLATIONS = {tot_viol}')
