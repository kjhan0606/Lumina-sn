#!/usr/bin/env python3
"""Build the MA-RADRECOMB upper-ion photoionization TARGET map (B4/D1/M1 data gap).

dig_E1 proved LUMINA's deep-EUV field is a 99.3%-line / 0.7%-continuum Co III/Fe III
fluorescence lamp because the ion-changing macro-atom RADRECOMB *continuum* channel is
blocked: `cmfgen_sigma_bf.bin` stores per-level sigma_bf(nu) but DISCARDS the CMFGEN
phot_data "Final state in ion" header (the converter `expand_atomic_data_cmfgen.py`
writes it only to the browsable H5, never to the C-loadable binary). Without the target
identification the macro-atom cannot know which upper-ion LEVEL a photoionization lands
on, so the reverse radiative-recombination continuum photon cannot be emitted.

This script recovers that mapping WITHOUT touching cmfgen_sigma_bf.bin. For each lower
ion (Z, stage) it parses phot_data_A's final-state header, resolves the matching level
in the UPPER ion (Z, stage+1), and writes, per ref-global level index, the global index
of its photoionization target upper-ion level.

Established format: dig_B6 read S II's "Final state in ion 3s2_3p2_3Pe g=9"; the parser
is scripts/cmfgen_parser.parse_phot (fields final_state / final_state_excit_eV /
final_state_g). Survey (all 19apr23) confirms every target ion has a SINGLE
photoionization route to the UPPER-ION GROUND (Excitation energy of final state = 0.0):
  Fe III -> Fe IV 3d5_6Se(g6); Co III -> Co IV 3d6_5De(g25); S II -> S III 3p2_3Pe(g9);
so target = upper-ion ground for the mappable set. Fail-closed on any ion whose upper
stage is absent from the ref tree (e.g. Co IV -> Co V: Co V not tracked) or whose
final-state config/g/energy does not round-trip.

Output (NEW file, parallel to cmfgen_sigma_bf.bin, same n_levels ordering):
  <ref_dir>/ma_radrecomb_target.bin
    uint32 magic   = 0x4D415254 ('MART')
    uint32 version = 2
    int32  n_levels            (== cmfgen_sigma_bf.bin n_levels)
    int32  n_ions_mapped
    int32  n_routes
    int32  target_offset[n_levels + 1]  (CSR offsets by lower level)
    int32  target_level_idx[n_routes]   (global upper-ion target level)
    float64 target_probability[n_routes] (ARTIS allcont_probability equivalent)
  <ref_dir>/ma_radrecomb_target_manifest.csv   (audit / provenance)

Usage:  python3 scripts/build_ma_radrecomb_target.py <ref_dir>
"""
from __future__ import annotations
import sys, csv, struct
from pathlib import Path
from collections import defaultdict, OrderedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from cmfgen_parser import parse_phot, parse_osc  # validated parsers

CMFGEN_ROOT = ROOT / 'data' / 'atomic' / 'cmfgen'
CMFGEN_DIRS = {14:'SIL', 16:'SUL', 20:'CA', 26:'FE', 27:'COB', 28:'NICK'}
ROMAN = ['', 'I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII']
SYM   = {14:'Si', 16:'S', 20:'Ca', 26:'Fe', 27:'Co', 28:'Ni'}
_CM2EV = 1.239841984e-4

# ions dominating the s8-s12 deep-EUV budgets (task list), (Z, spectroscopic stage)
TARGET_IONS = [(26,2),(26,3),(26,4), (27,2),(27,3),(27,4), (28,2),(28,3),(28,4),
               (16,2),(16,3), (14,2),(14,3), (20,2),(20,3)]

MAGIC = 0x4D415254
VERSION = 2
_DATE_RE = __import__('re').compile(r'^\d{1,2}[a-z]{3}\d{2}$')
_MONTHS = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,
           'sep':9,'oct':10,'nov':11,'dec':12}


def _pick_latest(ion_dir: Path):
    import re
    dates = [d for d in ion_dir.iterdir() if d.is_dir() and _DATE_RE.match(d.name)]
    if not dates:
        return None
    def key(d):
        m = re.match(r'^(\d{1,2})([a-z]{3})(\d{2})$', d.name)
        if not m: return (0,0,0)
        dd, mo, yy = m.groups(); yr = int(yy); yr = 2000+yr if yr < 50 else 1900+yr
        return (yr, _MONTHS.get(mo,0), int(dd))
    return max(dates, key=key)


def _phot_path(date_dir: Path):
    for nm in ('phot_data_A','phot_data_a','phot_data_B','phot_data_b','phot_data'):
        p = date_dir / nm
        if p.exists(): return p
    c = list(date_dir.glob('phot_data*'))
    return c[0] if c else None


def _norm(s):  # config normalizer (matches expand_atomic_data_cmfgen._norm_cfg)
    return str(s).strip().lower()

def _term(s):  # strip trailing [J/2] bracket
    import re
    return re.sub(r'\[[^\[\]]*\]\s*$', '', _norm(s))


def load_ref_levels(ref_dir: Path):
    """levels.csv row order == ref-global level index (same as bake_sigma_bf_for_ref)."""
    per_ion = defaultdict(list)                 # (Z, stage) -> [(gidx, lvl_csv, E_eV, g)]
    lookup  = {}                                # (Z, stage, lvl_csv+1) -> gidx
    n = 0
    with open(ref_dir / 'levels.csv') as fp:
        for r in csv.DictReader(fp):
            Z = int(r['atomic_number']); ion_csv = int(r['ion_number'])
            lvl = int(r['level_number']); E = float(r['energy_eV'])
            g = int(round(float(r['g']))); stage = ion_csv + 1
            per_ion[(Z, stage)].append((n, lvl, E, g))
            lookup[(Z, stage, lvl + 1)] = n
            n += 1
    return n, per_ion, lookup


def load_atomic_sources(ref_dir: Path):
    """Load exact lower-phot/upper-osc choices when the deck records them."""
    path = ref_dir / 'atomic_vintage_manifest.csv'
    if not path.is_file():
        return {}
    required = {'atomic_number', 'ion_stage', 'osc_path', 'phot_path'}
    result = {}
    with open(path, newline='') as fp:
        reader = csv.DictReader(fp)
        missing = required - set(reader.fieldnames or ())
        if missing:
            sys.exit(f"ERROR: {path} missing columns {sorted(missing)}")
        for row in reader:
            key = (int(row['atomic_number']), int(row['ion_stage']))
            if key in result:
                sys.exit(f"ERROR: duplicate atomic source for {key} in {path}")
            result[key] = {
                'osc': Path(row['osc_path']) if row['osc_path'] else None,
                'phot': Path(row['phot_path']) if row['phot_path'] else None,
            }
    return result


def read_has_cmfgen(sigma_bin: Path, n_levels_expect: int):
    with open(sigma_bin, 'rb') as f:
        magic, ver = struct.unpack('<II', f.read(8))
        nlev, nb   = struct.unpack('<ii', f.read(8))
        f.read(16)  # nu_min, nu_max
        if magic != 0x434D4644 or nlev != n_levels_expect:
            sys.exit(f"ERROR: {sigma_bin} magic/n_levels mismatch "
                     f"(magic=0x{magic:08X} nlev={nlev} expect={n_levels_expect})")
        flag = f.read(nlev)
    return [flag[i] for i in range(nlev)]


def resolve_upper_target(Z, stage, per_ion, lookup, atomic_sources):
    """Return (target_gidx, diag) for lower ion (Z,stage) -> upper (Z,stage+1).

    Data-driven + fail-closed. CMFGEN phot_data_A records the target as a header triple
    (Final state in ion / Excitation energy / Statistical weight of ion). The whole file
    is a SINGLE route (verified across the target set) whose final state is the upper-ion
    GROUND TERM at excitation energy 0. The osc levels are J-split, so the ground TERM
    (phot 'Statistical weight of ion', e.g. Co IV 5De g=25) maps to the fine-structure
    ground multiplet whose lowest member (osc id 1, level_num 0) is the landing level.

    Validation (fail-closed):
      * excitation energy of final state must be ~0  (ground-term route),
      * sum of osc g over the ground fine-structure multiplet == phot 'g' (term identity),
      * the upper-ion ground (level_num 0) must survive the ref level cap.
    Note: parse_phot's numeric-only header regex cannot read the alphanumeric config
    STRING, but Excitation-energy and Statistical-weight are numeric and reliable."""
    selected = atomic_sources.get((Z, stage))
    eldir = CMFGEN_DIRS.get(Z)
    if selected is not None:
        pp = selected['phot']
    else:
        ion_dir = CMFGEN_ROOT / eldir / ROMAN[stage] if eldir else None
        if not ion_dir or not ion_dir.is_dir():
            return None, "no_cmfgen_dir"
        dd = _pick_latest(ion_dir)
        if dd is None:
            return None, "no_date_dir"
        pp = _phot_path(dd)
    if pp is None:
        return None, "no_phot_data"
    try:
        phot = parse_phot(pp)
    except Exception as e:
        return None, f"phot_parse_err:{e}"
    up_stage = stage + 1
    if not per_ion.get((Z, up_stage)):
        return None, f"upper_ion_absent_in_ref({SYM.get(Z,Z)} {ROMAN[up_stage]})"

    exc_eV = phot.final_state_excit_eV
    phot_g = phot.final_state_g
    if abs(exc_eV) > 0.05:
        # Excited-target route (none of the s8-s12 ions use one); fail-closed rather
        # than guess a J-level for a non-ground term.
        return None, f"nonground_final_state(exc={exc_eV:.3f}eV)"

    # Upper-ion osc: identify the ground term and sum g over its fine-structure multiplet.
    upper_selected = atomic_sources.get((Z, up_stage))
    if upper_selected is not None:
        upper_osc = upper_selected['osc']
    else:
        up_dir = CMFGEN_ROOT / eldir / ROMAN[up_stage]
        ddu = _pick_latest(up_dir) if up_dir.is_dir() else None
        if ddu is None:
            return None, "upper_osc_absent"
        upper_osc = ddu / 'osc_data'
    if upper_osc is None or not upper_osc.is_file():
        return None, "upper_osc_absent"
    try:
        osc = parse_osc(upper_osc)
    except Exception as e:
        return None, f"upper_osc_parse_err:{e}"
    if len(osc.levels) == 0:
        return None, "upper_osc_empty"
    ground_term = _term(osc.levels['config'][0])
    g_gs = float(osc.levels['g'][0])
    # leading contiguous run sharing the ground term = ground multiplet
    g_term_sum = 0.0
    for k in range(len(osc.levels)):
        if _term(osc.levels['config'][k]) != ground_term:
            break
        g_term_sum += float(osc.levels['g'][k])
    g_ok = (phot_g <= 0) or abs(g_term_sum - phot_g) < 0.5

    tgt_gidx = lookup.get((Z, up_stage, 1))   # upper-ion ground (level_num 0)
    if tgt_gidx is None:
        return None, "upper_ground_capped_out_of_ref"

    diag = (f"1route exc={exc_eV:.3f}eV phot_g={phot_g:g} ground='{ground_term}' "
            f"g_gs={g_gs:g} Sg(multiplet)={g_term_sum:g} -> upper ground gidx="
            f"{tgt_gidx} g_ok={g_ok}")
    if not g_ok:
        return None, "g_term_mismatch:" + diag
    return tgt_gidx, diag


def main():
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    ref_dir = Path(sys.argv[1]).resolve()
    if not (ref_dir / 'levels.csv').exists():
        sys.exit(f"ERROR: {ref_dir}/levels.csv not found")
    sigma_bin = ref_dir / 'cmfgen_sigma_bf.bin'
    if not sigma_bin.exists():
        sys.exit(f"ERROR: {sigma_bin} not found (target map must match its n_levels)")

    print(f"=== build MA-RADRECOMB target map: {ref_dir} ===")
    n_levels, per_ion, lookup = load_ref_levels(ref_dir)
    atomic_sources = load_atomic_sources(ref_dir)
    if atomic_sources:
        print(f"atomic source manifest: {len(atomic_sources)} ions")
    print(f"ref levels: {n_levels}")
    has_cmfgen = read_has_cmfgen(sigma_bin, n_levels)
    n_bf = sum(has_cmfgen)
    print(f"cmfgen_sigma_bf levels with data: {n_bf}")

    target = [-1] * n_levels
    # The surveyed CMFGEN phot_data files each contain exactly one final-state
    # route, hence p_target=1. Keeping the term explicit in v2 prevents the
    # runtime opacity from silently assuming unity when richer target data is
    # introduced.
    probability = [1.0] * n_levels
    manifest = []          # rows for the audit CSV
    n_ions_mapped = 0
    for (Z, stage) in TARGET_IONS:
        levs = per_ion.get((Z, stage))
        name = f"{SYM.get(Z,Z)} {ROMAN[stage]}"
        if not levs:
            manifest.append((Z, stage, name, 0, 0, -1, "ion_absent_in_ref"))
            print(f"  {name:8s}: absent in ref tree (skip)")
            continue
        n_ion_lev = len(levs)
        n_bf_lev = sum(1 for (g, _l, _E, _g) in levs if has_cmfgen[g])
        tgt_gidx, diag = resolve_upper_target(
            Z, stage, per_ion, lookup, atomic_sources)
        if tgt_gidx is None:
            manifest.append((Z, stage, name, n_ion_lev, n_bf_lev, -1,
                             "FAILCLOSED:" + diag))
            print(f"  {name:8s}: FAIL-CLOSED ({diag}); {n_bf_lev} bf levels left "
                  f"target=-1")
            continue
        # Assign target to every bf-carrying level of this lower ion.
        n_assigned = 0
        for (g, _l, _E, _g) in levs:
            if has_cmfgen[g]:
                target[g] = tgt_gidx
                n_assigned += 1
        n_ions_mapped += 1
        # cross-check: target's ion == (Z, stage+1)
        tref = per_ion.get((Z, stage+1))
        tgt_stage_ok = tgt_gidx in {gg for (gg, *_r) in tref} if tref else False
        manifest.append((Z, stage, name, n_ion_lev, n_bf_lev, tgt_gidx,
                         f"MAPPED n={n_assigned} tgt_ion_ok={tgt_stage_ok} | {diag}"))
        print(f"  {name:8s}: {n_ion_lev:5d} lev, {n_bf_lev:5d} bf -> "
              f"target gidx {tgt_gidx} ({n_assigned} assigned) | {diag}")

    # ---- Round-trip / fail-closed validation of the whole array ----
    # Every non -1 target must be a valid global index and belong to stage+1 of a
    # SOURCE ion that has that gidx as target.
    # Build gidx -> (Z, stage) for validation.
    gidx_ion = {}
    for (Z, stage), levs in per_ion.items():
        for (g, *_r) in levs:
            gidx_ion[g] = (Z, stage)
    bad = 0
    for g in range(n_levels):
        t = target[g]
        if t < 0:
            continue
        if t >= n_levels or not has_cmfgen[g]:
            bad += 1; target[g] = -1; continue
        Zs, ss = gidx_ion[g]                # source (lower) ion
        Zt, st = gidx_ion.get(t, (-9, -9))  # target (upper) ion
        if not (Zt == Zs and st == ss + 1):
            bad += 1; target[g] = -1
    n_mapped = sum(1 for t in target if t >= 0)
    if bad:
        print(f"  [validation] {bad} inconsistent entries scrubbed to -1 (fail-closed)")
    print(f"total mapped levels: {n_mapped} over {n_ions_mapped} ions")

    # Strict ARTIS target list in CSR form. The present CMFGEN survey contributes
    # one route per mapped lower level; the schema itself permits any number of
    # routes so the runtime can evaluate sum_t p_t*corrfactor_t without collapse.
    target_offset = [0]
    target_routes = []
    probability_routes = []
    for g, tgt in enumerate(target):
        if tgt >= 0:
            target_routes.append(tgt)
            probability_routes.append(probability[g])
        target_offset.append(len(target_routes))
    n_routes = len(target_routes)

    # ---- Write binary ----
    out_bin = ref_dir / 'ma_radrecomb_target.bin'
    with open(out_bin, 'wb') as f:
        f.write(struct.pack('<II', MAGIC, VERSION))
        f.write(struct.pack('<ii', n_levels, n_ions_mapped))
        f.write(struct.pack('<i', n_routes))
        f.write(struct.pack(f'<{n_levels + 1}i', *target_offset))
        f.write(struct.pack(f'<{n_routes}i', *target_routes))
        f.write(struct.pack(f'<{n_routes}d', *probability_routes))
    nbytes = 8 + 8 + 4 + 4 * (n_levels + 1) + 4 * n_routes + 8 * n_routes
    print(f"wrote {out_bin} ({nbytes} bytes)")

    # Round-trip: reload and compare.
    with open(out_bin, 'rb') as f:
        m, v = struct.unpack('<II', f.read(8))
        nl, ni = struct.unpack('<ii', f.read(8))
        nr, = struct.unpack('<i', f.read(4))
        offset_rt = list(struct.unpack(f'<{nl + 1}i', f.read(4 * (nl + 1))))
        target_rt = list(struct.unpack(f'<{nr}i', f.read(4 * nr)))
        probability_rt = list(struct.unpack(f'<{nr}d', f.read(8 * nr)))
    assert (m == MAGIC and v == VERSION and nl == n_levels and
            nr == n_routes and offset_rt == target_offset and
            target_rt == target_routes and probability_rt == probability_routes), \
        "ROUND-TRIP FAILED"
    print(f"round-trip OK (magic=0x{m:08X} v{v} n_levels={nl} n_ions={ni})")

    # ---- Manifest CSV ----
    out_csv = ref_dir / 'ma_radrecomb_target_manifest.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Z','stage','ion','n_levels','n_bf_levels','target_gidx','status'])
        for row in manifest:
            w.writerow(row)
    print(f"wrote {out_csv}")


if __name__ == '__main__':
    main()
