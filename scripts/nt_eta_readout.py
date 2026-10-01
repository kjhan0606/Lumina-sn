#!/usr/bin/env python3
"""DET-NT-ETA readout (prereg docs/PREREG_DET_NT_ETA_2026-10-01.md §2.3, §4, §5).

Subcommands
  gid      --ref R --test E --out DIR     pre-submit / post-run identity gate (G-ID)
  readout  --ref R --test E --out DIR     G-ID + G-RUN + §5 tables, aggregate and branch
  selftest --ref R --out DIR              §4.3 NC battery (synthetic inputs; R only read)

R = W5 run root (baseline), E = NT-ETA run root.  Branches: NT-SUPPORT, NT-IV-ONLY,
NT-REV, NT-UNDET, NT-X.  Cell outcomes: a (expected direction), b (NT-REV cell),
c (no registered response), d (undecidable).  No clamps, floors or substitutions:
a non-positive / non-finite population or a missing / duplicated row is undefined
and drives NT-X with its reason.
"""
import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w5_cont_readout import window_ratios  # frozen W5 pure helper (sha 7008f709)

C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
EV = 1.602176634e-12
ELEMS = (16, 14, 20)
SHELLS = (10, 15, 19, 24, 30)
AUTHS = ("stage", "levels")
THR34 = -0.5
THR23 = 0.30
GEN_K = {3: 6, 4: 7, 5: 8}
PATH_KEYS = ("LUMINA_MODEL_DIR", "LUMINA_CMFGEN_SIGMA_BF", "LUMINA_DEPOSITION_FILE",
             "LUMINA_FIXED_TE_PROFILE", "LUMINA_IONIZATION_REFERENCE_FILE",
             "LUMINA_TOPION_LEVELS_FILE", "LUMINA_PHYSICS_COMPARISON_DIR")
MUST_KEEP = {"LUMINA_PURE_CMFGEN_ITER": "6", "LUMINA_DET_TRANSACTIONAL": "1",
             "LUMINA_CMF_FINE_MGPU_DEVICES": "2", "LUMINA_CMF_FINE_CONT_NLTE_SOURCE": "1",
             "LUMINA_ION_NT": "1", "LUMINA_POP_DJDN": "1"}
W5_BIN = "7800484fb309addabfa5b2f6b879f3ab09b814db83d3c5db2f39debff3f1b390"
W5_BUILD = "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1"
W5_APPROVED = "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/APPROVED_SHA.txt"
SRC_ROOT = "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source"
CHI = {(16, 1): 23.3379, (14, 1): 16.3459}
PUB = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06"
PUB_SYM = {14: "si", 16: "s", 20: "ca"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def finite_pos(x):
    return isinstance(x, float) and math.isfinite(x) and x > 0.0


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ G-ID
def read_exports(p):
    d, dup = {}, []
    for ln in open(p):
        ln = ln.strip()
        if not ln.startswith("declare -x "):
            continue
        k, _, v = ln[len("declare -x "):].partition("=")
        v = v[1:-1] if len(v) >= 2 and v[0] == v[-1] == '"' else v
        if k in d:
            dup.append(k)
        d[k] = v
    return d, dup


def exports_delta(dr, de, rroot, eroot):
    """Reasons for any exports difference other than the registered one."""
    why = []
    for k in sorted(set(dr) | set(de)):
        vr, ve = dr.get(k), de.get(k)
        if k in PATH_KEYS:
            if vr is None or ve is None or not vr.startswith(rroot) or ve != eroot + vr[len(rroot):]:
                why.append(f"EXPORTS_PATH:{k}")
            continue
        if k == "LUMINA_NT_FULL":
            if (vr, ve) != ("1", "0"):
                why.append(f"EXPORTS_NT_FULL:{vr}->{ve}")
            continue
        if vr != ve:
            why.append(f"EXPORTS_EXTRA_DIFF:{k}")
    for k, v in MUST_KEEP.items():
        if de.get(k) != v:
            why.append(f"EXPORTS_MUST_KEEP:{k}")
    return why


def footer(run):
    p = Path(run) / "RUN_FOOTER.txt"
    if not p.exists():
        return {}
    return dict(ln.rstrip("\n").split("=", 1) for ln in open(p, errors="replace") if "=" in ln)


def manifest_bad(listing, base):
    bad = 0
    for ln in open(listing):
        if not ln.strip():
            continue
        h, p = ln.split(None, 1)
        if sha(Path(base) / p.strip().lstrip("*")) != h:
            bad += 1
    return bad


def content_reasons(root, nm):
    why = []
    if sha(root / "input/model/cmfgen_sigma_bf.bin") != (root / "input/sigma.sha256").read_text().strip():
        why.append(f"SIGMA_CONTENT:{nm}")
    n = manifest_bad(root / "input/deck.sha256", root / "input/model")
    if n:
        why.append(f"DECK_CONTENT:{nm}:{n}")
    n = manifest_bad(root / "input/topion.sha256", root / "input/global_atomic")
    if n:
        why.append(f"TOPION_CONTENT:{nm}:{n}")
    return why


def bins_reason(bins):
    return [] if set(bins.values()) == {W5_BIN} else ["BINARY_MISMATCH"]


def approved_reasons(rec, approved_text):
    why = []
    for item in approved_text.strip().split(","):
        p, h = item.split("=")
        if rec.get(p) != h:
            why.append(f"APPROVED_MISMATCH:{p}")
    return why


def fixed_te_reason(ft, nm):
    return [] if ft.get("fixed_te_loader_consumed") == "1" else [f"FIXED_TE_NOT_CONSUMED:{nm}"]


def gid(rroot, eroot, post_run):
    R, E = Path(rroot), Path(eroot)
    why, ev = [], {}
    bins = {"build_record": (Path(W5_BUILD) / "binary.sha256").read_text().split()[0],
            "build_file": sha(Path(W5_BUILD) / "lumina_cuda")}
    for nm, root in (("R", R), ("E", E)):
        bins[f"{nm}_file"] = sha(root / "input/lumina_cuda")
        bins[f"{nm}_record"] = (root / "input/binary.sha256").read_text().strip()
        bins[f"{nm}_rbi"] = json.load(open(root / "input/run_build_identity.json"))["binary"]["sha256"]
        st = dict(ln.rstrip("\n").split("=", 1) for ln in open(root / "STAGE_IDENTITY.txt") if "=" in ln)
        bins[f"{nm}_stage"] = st.get("run_binary_sha256", "").strip()
    ev["binary"] = bins
    why += bins_reason(bins)
    srcs = {nm: (root / "input/source_sha_at_build.txt").read_text() for nm, root in (("R", R), ("E", E))}
    if srcs["R"] != srcs["E"] or srcs["E"] != (Path(W5_BUILD) / "source_sha_at_build.txt").read_text():
        why.append("SOURCE_RECORD_MISMATCH")
    rec = {}
    for ln in srcs["E"].splitlines():
        h, p = ln.split(None, 1)
        rec[p.lstrip("*")] = h
    bad_src = [p for p, h in rec.items() if sha(Path(SRC_ROOT) / p) != h]
    if bad_src:
        why.append(f"SOURCE_TREE_MISMATCH:{len(bad_src)}")
    why += approved_reasons(rec, Path(W5_APPROVED).read_text())
    for f in ("deck.sha256", "topion.sha256", "sigma.sha256", "fixed_te_profile.txt", "run_build_identity.json"):
        if sha(R / "input" / f) != sha(E / "input" / f):
            why.append(f"INPUT_MISMATCH:{f}")
    why += content_reasons(R, "R") + content_reasons(E, "E")
    dr, dupr = read_exports(R / "input/resolved_lumina.exports")
    de, dupe = read_exports(E / "input/resolved_lumina.exports")
    if dupr or dupe:
        why.append("EXPORTS_DUPLICATE_KEY")
    why += exports_delta(dr, de, str(R), str(E))
    if post_run:
        for nm, root in (("R", R), ("E", E)):
            ft = footer(root)
            why += fixed_te_reason(ft, nm)
            ev[f"footer_{nm}"] = {k: ft.get(k) for k in ("job_id", "partition", "gpu_kind", "allocated_gpus",
                                                          "terminal_state", "model_rc", "node_list")}
    ev["runner_sha_E"] = sha(E / "job.slurm")
    return why, ev


# ------------------------------------------------------------------ G-RUN and inputs
def load_ion_iter(run, k):
    p = Path(run) / f"work/physics_comparison/physics_DET_iter{k:04d}.ion.csv"
    rows, dup = {}, 0
    with open(p) as fh:
        for r in csv.DictReader(fh):
            s = int(r["shell_id"])
            if s in rows:
                dup += 1
            rows[s] = r
    return rows, dup


def load_final_stages(run):
    out, dup = {}, 0
    with open(Path(run) / "work/lumina_ion_pops.csv") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["shell_id"]), int(r["Z"]), int(r["stage"]))
            if key in out:
                dup += 1
            out[key] = (fnum(r["n_ion"]), r["population_generation"])
    return out, dup


def footer_reasons(ft, slurm_state):
    why = []
    if ft.get("terminal_state") != "SUCCESS":
        why.append("FOOTER_NOT_SUCCESS")
    if ft.get("model_rc") != "0":
        why.append(f"MODEL_RC:{ft.get('model_rc')}")
    if ft.get("allocated_gpus") != "2" or ft.get("gpu_kind") not in ("H100", "A100"):
        why.append("GPU_IDENTITY")
    if ft.get("outer_iterations") != "6":
        why.append(f"OUTER_ITERATIONS:{ft.get('outer_iterations')}")
    if ft.get("fixed_te_loader_consumed") != "1":
        why.append("FIXED_TE_NOT_CONSUMED")
    if slurm_state is None or "COMPLETED" not in slurm_state.split():
        why.append(f"SLURM_STATE:{slurm_state}")
    if slurm_state is not None and "RESTARTS=0" not in slurm_state.split():
        why.append("SLURM_RESTARTS")
    return why


def te_reasons(run, profile_T):
    """Published T_e (k5 shell.csv) equals the fixed profile; manifests carry one te manifest hash."""
    why, hashes = [], set()
    W = Path(run) / "work"
    for k in range(6):
        mp = W / f"physics_comparison/physics_DET_iter{k:04d}.manifest.json"
        if mp.exists():
            hashes.add(json.load(open(mp)).get("te_manifest_sha256"))
    fp = W / "cmfgen_final_publication.json"
    if fp.exists():
        hashes.add(json.load(open(fp)).get("te_manifest_sha256"))
    if len(hashes) != 1 or None in hashes:
        why.append("TE_MANIFEST_HASHES")
    P = W / "physics_comparison/physics_DET_iter0005.shell.csv"
    if P.exists():
        for r in csv.DictReader(open(P)):
            s = int(r["shell_id"])
            if s not in profile_T or fnum(r["T_e_K"]) != profile_T[s]:
                why.append("TE_PUBLISHED_NE_PROFILE")
                break
    return why, sorted(h for h in hashes if h)


def grun(run, slurm_state=None):
    W = Path(run) / "work"
    ft = footer(run)
    why = footer_reasons(ft, slurm_state)
    prof = {}
    for ln in open(Path(run) / "input/fixed_te_profile.txt"):
        a = ln.split()
        if len(a) == 2:
            prof[int(a[0])] = float(a[1])
    tw, _ = te_reasons(run, prof)
    why += tw
    if (W / "physics_comparison/physics_DET_iter0006.manifest.json").exists():
        why.append("EXTRA_ITERATION_K6")
    for k in range(6):
        mp = W / f"physics_comparison/physics_DET_iter{k:04d}.manifest.json"
        if not mp.exists():
            why.append(f"MANIFEST_MISSING:k{k}")
            continue
        m = json.load(open(mp))
        if m.get("transaction_status") != "COMMITTED" or m.get("radiation_generation") != k + 2 \
                or m.get("population_generation") != k + 3 or m.get("te_lane") != "FIXED_T":
            why.append(f"MANIFEST_STATE:k{k}")
    fp = W / "cmfgen_final_publication.json"
    if not fp.exists():
        why.append("FINAL_PUBLICATION_MISSING")
    else:
        f = json.load(open(fp))
        if not (f.get("complete") is True and f.get("population_generation") == 8
                and f.get("required_population_generation") == 8):
            why.append("FINAL_PUBLICATION_STATE")
    log = Path(run) / "stderr.log"
    ok_epochs = 0
    if log.exists():
        for ln in open(log, errors="replace"):
            if ln.startswith("[cmf_fine][EXACT-MULTIGPU-EPOCH]") and " status=OK " in ln + " ":
                ok_epochs += 1
    if ok_epochs != 7:
        why.append(f"EXACT_SOLVE_OK:{ok_epochs}")
    for p in ("lumina_spectrum.csv", "nlte_levels_final.csv", "lumina_ion_channels.csv",
              "physics_comparison/physics_DET_iter0005.spectral.csv",
              "physics_comparison/physics_DET_iter0005.shell.csv"):
        if not (W / p).exists():
            why.append(f"FILE_MISSING:{p}")
    return why


def switch_witness(rroot, eroot):
    """Report-only (pre-audit R-1): NT switch consumption evidence."""
    def published(run):
        out = []
        for ln in open(Path(run) / "stderr.log", errors="replace"):
            if ln.startswith("[GAMMA][PUBLISHED]"):
                d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
                out.append((d.get("heating_manifest_sha256"), d.get("nonthermal_manifest_sha256")))
        return out
    def gnt(run):
        g, gmax = {}, -1
        with open(Path(run) / "work/lumina_ion_channels.csv") as fh:
            for r in csv.DictReader(fh):
                gm = int(r["gen"])
                if gm > gmax:
                    gmax, g = gm, {}
                if gm == gmax and r["Z"] == "16" and r["stage"] == "1":
                    g[int(r["shell"])] = fnum(r["gamma_nt"])
        return g
    pr, pe = published(rroot), published(eroot)
    gr, ge = gnt(rroot), gnt(eroot)
    ratios = {s: ge[s] / gr[s] for s in gr if s in ge and finite_pos(gr[s])}
    return {"heating_hash_same": bool(pr and pe and pr[-1][0] == pe[-1][0]),
            "nonthermal_hash_differs": bool(pr and pe and pr[-1][1] != pe[-1][1]),
            "gamma_nt_ratio_min": min(ratios.values()) if ratios else None,
            "gamma_nt_ratio_max": max(ratios.values()) if ratios else None,
            "shells_compared": len(ratios)}


# ------------------------------------------------------------------ §5 statistics
def y34_table(fr, fe):
    """15 cells; returns rows and reasons (undefined -> reason)."""
    rows, why = [], []
    for Z in ELEMS:
        for s in SHELLS:
            row = {"Z": Z, "shell": s}
            ok = True
            for nm, F in (("R", fr), ("E", fe)):
                n3, g3 = F.get((s, Z, 2), (float("nan"), None))
                n4, g4 = F.get((s, Z, 3), (float("nan"), None))
                row.update({f"nIII_{nm}": n3, f"nIV_{nm}": n4, f"gen_{nm}": f"{g3}/{g4}"})
                if not (finite_pos(n3) and finite_pos(n4)) or g3 != "8" or g4 != "8":
                    ok = False
                else:
                    row[f"y34_{nm}"] = math.log10(n4 / n3)
            if ok:
                row["delta34"] = row["y34_E"] - row["y34_R"]
            else:
                why.append(f"Y34_UNDEFINED:Z{Z}/sh{s}")
            rows.append(row)
    return rows, why


def y23_series(iters, Z, A, s):
    """y_k for k=3,4,5 or None with reason."""
    tag = "stage" if A == "stage" else "nlte"
    ys = {}
    for k in (3, 4, 5):
        r = iters[k].get(s)
        if r is None:
            return None, f"Y23_ROW_MISSING:k{k}"
        if r.get("population_generation") != str(GEN_K[k]):
            return None, f"Y23_GENERATION:k{k}"
        q1, q2 = fnum(r.get(f"Z{Z}_{tag}_q1_cm3")), fnum(r.get(f"Z{Z}_{tag}_q2_cm3"))
        if not (finite_pos(q1) and finite_pos(q2)):
            return None, f"Y23_NONPOSITIVE:k{k}"
        ys[k] = math.log10(q2 / q1)
    return ys, None


def cell_outcome(delta, D):
    """§5.2 four-way partition; delta finite, D >= 0."""
    T = max(THR23, 3.0 * D)
    if delta <= -T:
        return "a"
    if delta >= T:
        return "b"
    if abs(delta) < THR23 and 3.0 * D <= THR23:
        return "c"
    return "d"   # 3D > 0.30 and |delta| < 3D


def y23_cell(yr, ye):
    dr = abs(yr[5] - yr[3]) / 2.0
    de = abs(ye[5] - ye[3]) / 2.0
    D = max(de, dr)
    delta = (ye[4] + ye[5]) / 2.0 - (yr[4] + yr[5]) / 2.0
    return {"ybar_R": (yr[4] + yr[5]) / 2.0, "ybar_E": (ye[4] + ye[5]) / 2.0, "d_R": dr, "d_E": de,
            "D": D, "T": max(THR23, 3.0 * D), "delta23": delta, "outcome": cell_outcome(delta, D)}


def aggregate(outcomes):
    """Unique mode of the sulfur 10 cells; tie -> ('d', 'TIE')."""
    n = {o: outcomes.count(o) for o in "abcd"}
    top = max(n.values())
    modes = [o for o in "abcd" if n[o] == top]
    if len(modes) > 1:
        return "d", "AGGREGATE_TIE", n
    return modes[0], ("CELL_D_MODE" if modes[0] == "d" else "UNIQUE_MODE"), n


def branch(i34, agg):
    if not i34:
        return "NT-X"
    return {"a": "NT-SUPPORT", "b": "NT-REV", "c": "NT-IV-ONLY", "d": "NT-UNDET"}[agg]


# ------------------------------------------------------------------ §5.4 auxiliaries
def band_integral(pts, lo, hi):
    """Linear-interpolated edges + trapezoid in wavelength; full support required."""
    if not pts or pts[0][0] > lo or pts[-1][0] < hi:
        return None, "UNSUPPORTED"
    tot = 0.0
    for (l0, f0), (l1, f1) in zip(pts[:-1], pts[1:]):
        if l1 <= l0:
            return None, "NONMONOTONE"
        a, b = max(l0, lo), min(l1, hi)
        if b <= a:
            continue
        fa = f0 + (f1 - f0) * (a - l0) / (l1 - l0)
        fb = f0 + (f1 - f0) * (b - l0) / (l1 - l0)
        tot += 0.5 * (fa + fb) * (b - a)
    return tot, "OK"


def fuv(run):
    pts = []
    for r in csv.DictReader(open(Path(run) / "work/lumina_spectrum.csv")):
        pts.append((fnum(r["wavelength_angstrom"]), fnum(r["flux"])))
    pts.sort()
    if any(not (math.isfinite(l) and math.isfinite(f)) for l, f in pts):
        return {"status": "NONFINITE"}
    num, s1 = band_integral(pts, 912.0, 1500.0)
    den, s2 = band_integral(pts, 4000.0, 6000.0)
    if s1 != "OK" or s2 != "OK":
        return {"status": f"{s1}/{s2}"}
    if not (num > 0.0 and den > 0.0):
        return {"status": "NONPOSITIVE", "L912_1500": num, "L4000_6000": den}
    return {"status": "OK", "L912_1500": num, "L4000_6000": den, "F": num / den,
            "support_A": [pts[0][0], pts[-1][0]]}


def a_band(run, shell=24):
    P = Path(run) / "work/physics_comparison/physics_DET_iter0005"
    T = None
    for r in csv.DictReader(open(f"{P}.shell.csv")):
        if int(r["shell_id"]) == shell:
            T = fnum(r["T_e_K"])
    rows = [(fnum(r["nu_lo_Hz"]), fnum(r["nu_hi_Hz"]), fnum(r["J_nu"]))
            for r in csv.DictReader(open(f"{P}.spectral.csv")) if int(r["shell_id"]) == shell]
    w = window_ratios(rows, T)["blue_end-531"]
    return {"shell": shell, "T_e": T, **w}


def load_sigma(path):
    with open(path, "rb") as f:
        _, _, nlev, nfb = struct.unpack("<IIii", f.read(16))
        numin, numax = struct.unpack("<dd", f.read(16))
        has = f.read(nlev)
        f.read((-(32 + nlev)) % 8)
        import array
        sig = array.array("d")
        sig.frombytes(f.read(nlev * nfb * 8))
    return has, sig, nfb, numin, numax


GRID_RTOL = 1e-12   # float re-evaluation of log-uniform edges, recorded in the result


def grid_reason(rows, numin, numax, nfb):
    """Spectral bins must be the sigma table's log-uniform grid (same count and edges)."""
    if len(rows) != nfb:
        return f"GRID_COUNT:{len(rows)}!={nfb}"
    dl = (math.log(numax) - math.log(numin)) / nfb
    for b, (lo, hi, _) in enumerate(rows):
        elo, ehi = math.exp(math.log(numin) + b * dl), math.exp(math.log(numin) + (b + 1) * dl)
        if abs(lo - elo) > GRID_RTOL * elo or abs(hi - ehi) > GRID_RTOL * ehi:
            return f"GRID_EDGE:bin{b}"
    return None


def level_weights(L, T, lab):
    """LTE: g exp(-E/kT) normalised within the ion; NLTE: final level populations normalised."""
    w = [g * math.exp(-E * EV / (K_CGS * T)) for _, E, g, _ in L] if lab == "LTE" else [n for *_, n in L]
    sw = sum(w)
    return [x / sw for x in w]


def in_band_D(nu):
    lam = C_CGS / nu * 1e8
    return 912.0 <= lam < 1500.0


def gamma_band(L, rows, has, sig, nfb, chi, T, lab):
    """(Γ_D, Γ_all) with bin-centre classification and the given level weights."""
    gd = ga = 0.0
    for (gi, E, g, n), wi in zip(L, level_weights(L, T, lab)):
        if not has[gi]:
            continue
        nu_th = (chi - E) * EV / H_CGS
        for b, (lo, hi, J) in enumerate(rows):
            nu = math.sqrt(lo * hi)
            if nu < nu_th:
                continue
            gam = 4.0 * math.pi * sig[gi * nfb + b] * J / (H_CGS * nu) * (hi - lo)
            ga += wi * gam
            if in_band_D(nu):
                gd += wi * gam
    return gd, ga


def sulfur_outcomes(t23):
    return [r["outcome"] for r in t23 if r["Z"] == 16 and r.get("outcome") in ("a", "b", "c", "d")]


def d_band(run):
    """S II / Si II, sh10 & sh24, k5: share of photoionisation Γ in [912,1500) Å (LTE & NLTE weights)."""
    has, sig, nfb, numin, numax = load_sigma(Path(run) / "input/model/cmfgen_sigma_bf.bin")
    P = Path(run) / "work/physics_comparison/physics_DET_iter0005"
    Ts = {int(r["shell_id"]): fnum(r["T_e_K"]) for r in csv.DictReader(open(f"{P}.shell.csv"))}
    spec = {}
    for r in csv.DictReader(open(f"{P}.spectral.csv")):
        s = int(r["shell_id"])
        if s in (10, 24):
            spec.setdefault(s, []).append((fnum(r["nu_lo_Hz"]), fnum(r["nu_hi_Hz"]), fnum(r["J_nu"])))
    lev = {}
    for r in csv.DictReader(open(Path(run) / "work/nlte_levels_final.csv")):
        Z, q, s = int(r["Z"]), int(r["ion"]), int(r["shell"])
        if (Z, q) in CHI and s in (10, 24):
            lev.setdefault((Z, s), []).append((int(r["global_idx"]), fnum(r["E_eV"]), fnum(r["g"]), fnum(r["n_pop"])))
    out = {}
    for s in (10, 24):
        rows = spec[s]
        g = grid_reason(rows, numin, numax, nfb)
        if g:
            return {"status": g}
        T = Ts[s]
        for Z in (16, 14):
            chi = CHI[(Z, 1)]
            L = lev[(Z, s)]
            res = {}
            for lab in ("LTE", "NLTE"):
                gd, ga = gamma_band(L, rows, has, sig, nfb, chi, T, lab)
                if not (ga > 0.0 and math.isfinite(ga) and math.isfinite(gd)):
                    return {"status": f"GAMMA_UNDEFINED:Z{Z}/sh{s}/{lab}"}
                res[lab] = {"gamma_D": gd, "gamma_all": ga, "share_D": gd / ga}
            out[f"Z{Z}/sh{s}"] = res
    return {"status": "OK", "cells": out, "note": "k5 radiation gen7 x final gen8 weights (post-processing)"}


def pub_rows(Z, t="19.480"):
    rows, cur = [], None
    for ln in open(Path(PUB) / f"ionfrac_{PUB_SYM[Z]}_toy06_cmfgen.txt"):
        if ln.startswith("#TIME:"):
            cur = ln.split()[1]
            continue
        if ln.startswith("#") or not ln.strip():
            continue
        if cur == t:
            v = [float(x) for x in ln.split()]
            rows.append((v[0], v[1:]))
    return rows


def pub_gap34(run, rows34):
    """Report-only (pre-audit R-2): log10(x34_run / x34_pub), public row nearest the shell-centre velocity."""
    P = Path(run) / "work/physics_comparison/physics_DET_iter0005.shell.csv"
    vmid = {int(r["shell_id"]): 0.5 * (fnum(r["v_inner_cm_s"]) + fnum(r["v_outer_cm_s"])) / 1e5
            for r in csv.DictReader(open(P))}
    cache = {}
    for r in rows34:
        Z, s = r["Z"], r["shell"]
        rows = cache.setdefault(Z, pub_rows(Z))
        v, fr = min(rows, key=lambda x: abs(x[0] - vmid[s]))
        if len(fr) > 3 and fr[2] > 0 and fr[3] > 0 and "y34_E" in r:
            xp = fr[3] / fr[2]
            r["log10_x34_pub"] = math.log10(xp)
            r["gap_E"] = r["y34_E"] - math.log10(xp)
            r["gap_R"] = r["y34_R"] - math.log10(xp)
            r["pub_v_kms"] = v


# ------------------------------------------------------------------ commands
def cmd_gid(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    why, ev = gid(a.ref, a.test, post_run=False)
    (out / "gid_presubmit.json").write_text(json.dumps({"reasons": why, "evidence": ev}, indent=1))
    print(json.dumps({"G-ID": "PASS" if not why else "FAIL", "reasons": why}))
    return 0 if not why else 1


def cmd_readout(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    reasons, gate = [], {}
    why_id, ev = gid(a.ref, a.test, post_run=True)
    gate["G-ID"] = why_id
    st = json.load(open(a.slurm_states)) if a.slurm_states else {}
    gate["G-RUN"] = {"R": grun(a.ref, st.get("R")), "E": grun(a.test, st.get("E"))}
    gsel = json.load(open(a.selftest)) if a.selftest else []
    gate["G-READOUT"] = "PASS" if gsel and all(c.get("pass") for c in gsel) else "FAIL"
    if gate["G-READOUT"] != "PASS":
        reasons.append("G-READOUT:SELFTEST_NOT_PASSED")
    reasons += [f"G-ID:{w}" for w in why_id]
    reasons += [f"G-RUN:{k}:{w}" for k, v in gate["G-RUN"].items() for w in v]
    fr, dr = load_final_stages(a.ref)
    fe, de = load_final_stages(a.test)
    if dr or de:
        reasons.append("FINAL_STAGE_DUPLICATE")
    t34, w34 = y34_table(fr, fe)
    reasons += w34
    try:
        pub_gap34(a.ref, t34)
        pub_status = "OK"
    except Exception as exc:          # report-only column; never gates the branch
        pub_status = f"UNAVAILABLE:{type(exc).__name__}"

    iters = {}
    for nm, run in (("R", a.ref), ("E", a.test)):
        iters[nm] = {}
        for k in (3, 4, 5):
            rows, dup = load_ion_iter(run, k)
            if dup:
                reasons.append(f"ION_DUPLICATE:{nm}:k{k}")
            iters[nm][k] = rows
    t23 = []
    for Z in ELEMS:
        for A in AUTHS:
            for s in SHELLS:
                yr, er = y23_series(iters["R"], Z, A, s)
                ye, ee = y23_series(iters["E"], Z, A, s)
                row = {"Z": Z, "authority": A, "shell": s}
                if er or ee:
                    row["outcome"] = "UNDEFINED"
                    reasons.append(f"Y23:{Z}/{A}/sh{s}:{er or ee}")
                else:
                    row.update({f"y{k}_R": yr[k] for k in (3, 4, 5)})
                    row.update({f"y{k}_E": ye[k] for k in (3, 4, 5)})
                    row.update(y23_cell(yr, ye))
                t23.append(row)
    outcomes_S = sulfur_outcomes(t23)
    n_pass = sum(1 for r in t34 if r["Z"] == 16 and "delta34" in r and r["delta34"] <= THR34)
    i34 = n_pass >= 3
    agg, agg_why, counts = aggregate(outcomes_S) if len(outcomes_S) == 10 else ("d", "S_CELLS_INCOMPLETE", {})
    stat_branch = branch(i34, agg)
    full = "NT-X" if reasons else stat_branch
    aux = {"FUV": {"R": fuv(a.ref), "E": fuv(a.test)}}
    if aux["FUV"]["R"].get("status") == "OK" and aux["FUV"]["E"].get("status") == "OK":
        aux["FUV"]["delta_log10"] = math.log10(aux["FUV"]["E"]["F"]) - math.log10(aux["FUV"]["R"]["F"])
    else:
        full = "NT-X"
        reasons.append("FUV_UNDEFINED")
    aux["A_band_JB_sh24"] = {"R": a_band(a.ref), "E": a_band(a.test)}
    aux["D_band_gamma"] = {"R": d_band(a.ref), "E": d_band(a.test)}
    for nm in ("R", "E"):
        if aux["A_band_JB_sh24"][nm].get("status") not in ("OK", "LOG_ONLY", "ZERO_J") or \
                aux["D_band_gamma"][nm].get("status") != "OK":
            full = "NT-X"
            reasons.append(f"AUX_UNDEFINED:{nm}")
    aux["switch_witness_R1"] = switch_witness(a.ref, a.test)
    aux["public_IV_III_gap_column"] = pub_status
    ab = aux["A_band_JB_sh24"]
    if ab["R"].get("log10_R") is not None and ab["E"].get("log10_R") is not None:
        ab["delta_log10"] = ab["E"]["log10_R"] - ab["R"]["log10_R"]
    dbg = aux["D_band_gamma"]
    if dbg["R"].get("status") == "OK" and dbg["E"].get("status") == "OK":
        dbg["share_diff_E_minus_R"] = {c: {w: dbg["E"]["cells"][c][w]["share_D"] - dbg["R"]["cells"][c][w]["share_D"]
                                           for w in ("LTE", "NLTE")} for c in dbg["R"]["cells"]}
    x_reason = None
    if full == "NT-X":
        x_reason = "DATA_OR_GATE" if reasons else "PRIMARY_NOT_MET"
    res = {"schema": "NT_ETA_READOUT_V1", "ref": a.ref, "test": a.test,
           "branch": full, "statistic_branch": stat_branch, "nt_x_cause": x_reason,
           "reasons": reasons, "sulfur_shells_delta34_le_-0.5": n_pass, "I34": i34,
           "sulfur_aggregate": {"result": agg, "why": agg_why, "counts": counts},
           "cell_outcome_distribution_30": {o: sum(1 for r in t23 if r.get("outcome") == o) for o in "abcd"},
           "gates": gate, "identity_evidence": ev,
           "claim_limits": "6-iteration endpoint response; ETA=0.05 is a 1/20 NT-rate sensitivity value, not an SF fraction; "
                           "joint response does not isolate the IV-mediated path; non-detection is not NT irrelevance"}
    for nm, tab in (("population34.csv", t34), ("population23.csv", t23)):
        keys = sorted({k for r in tab for k in r})
        with open(out / nm, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(tab)
    (out / "auxiliary.json").write_text(json.dumps(aux, indent=1, default=str))
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    me = Path(__file__).resolve()
    consumed = []
    for root in (a.ref, a.test):
        W = Path(root) / "work"
        for rel in ["lumina_ion_pops.csv", "lumina_spectrum.csv", "nlte_levels_final.csv",
                    "cmfgen_final_publication.json", "lumina_ion_channels.csv"] + \
                   [f"physics_comparison/physics_DET_iter{k:04d}.{x}" for k in range(6)
                    for x in ("manifest.json", "ion.csv")] + \
                   ["physics_comparison/physics_DET_iter0005.spectral.csv",
                    "physics_comparison/physics_DET_iter0005.shell.csv"]:
            q = W / rel
            consumed.append({"path": str(q), "sha256": sha(q) if q.exists() else None})
        for rel in ("RUN_FOOTER.txt", "stderr.log", "input/model/cmfgen_sigma_bf.bin", "input/fixed_te_profile.txt"):
            q = Path(root) / rel
            consumed.append({"path": str(q), "sha256": sha(q) if q.exists() else None})
    gate_files = [x for x in (a.selftest, a.slurm_states) if x]
    (out / "evidence_manifest.json").write_text(json.dumps(
        {"readout": {"path": str(me), "sha256": sha(me)},
         "w5_helper": {"path": str(me.parent / "w5_cont_readout.py"), "sha256": sha(me.parent / "w5_cont_readout.py")},
         "consumed": consumed,
         "gate_evidence": [{"path": x, "sha256": sha(x)} for x in gate_files],
         "grid_rtol": GRID_RTOL}, indent=1))
    print(json.dumps({k: res[k] for k in ("branch", "statistic_branch", "nt_x_cause",
                                          "sulfur_shells_delta34_le_-0.5", "sulfur_aggregate", "reasons")},
                     default=str))
    return 0


# ------------------------------------------------------------------ selftest (§4.3)
def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []

    def case(name, got, want):
        ok = got == want
        res.append({"case": name, "pass": ok, "got": str(got), "want": str(want)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got} want {want}")

    nxt = math.nextafter
    # cell four-way partition at exact boundaries
    for D in (0.05, 0.10, 0.15):
        T = max(THR23, 3.0 * D)
        case(f"D={D} delta=-T -> a", cell_outcome(-T, D), "a")
        case(f"D={D} delta=+T -> b", cell_outcome(T, D), "b")
        inner = nxt(-T, 0.0)
        case(f"D={D} delta just inside -T", cell_outcome(inner, D), "c" if 3.0 * D <= THR23 else "d")
    D3 = 0.1
    while 3.0 * D3 > THR23:
        D3 = nxt(D3, 0.0)
    case("3D == 0.30 exactly representable", 3.0 * D3 == THR23, True)
    case("3D == 0.30, delta just inside 0.30 -> c", cell_outcome(nxt(-THR23, 0.0), D3), "c")
    case("3D == 0.30, delta = -0.30 -> a", cell_outcome(-THR23, D3), "a")
    D4 = nxt(D3, 1.0)
    case("3D just above 0.30, |delta|<3D -> d", cell_outcome(nxt(-THR23, 0.0), D4), "d")
    case("3D>0.30 below 3D -> d", cell_outcome(-0.40, 0.15), "d")
    case("old overlap counterexample D=0.15 delta=-0.60 -> a only", cell_outcome(-0.60, 0.15), "a")
    case("3D>0.30 alone does not force d when |delta|>=T", cell_outcome(0.46, 0.15), "b")
    # both-side drift and log mean
    yr = {3: 0.0, 4: 0.1, 5: 0.0}
    ye = {3: 1.0, 4: 0.2, 5: 0.0}          # test-side drift 0.5 -> D=0.5
    c = y23_cell(yr, ye)
    case("drift from test side enters D", (round(c["D"], 12), c["outcome"]), (0.5, "d"))
    yr = {3: 0.4, 4: 0.0, 5: 0.0}
    ye = {3: 0.0, 4: -0.4, 5: -0.4}        # reference drift 0.2 -> T=0.6
    case("drift from reference side enters D", y23_cell(yr, ye)["outcome"], "d")
    yr = {3: 0.0, 4: 0.0, 5: 0.0}
    ye = {3: -1.0, 4: -1.0, 5: -1.0}
    case("log mean (not arithmetic ratio mean)", round(y23_cell(yr, ye)["delta23"], 12), -1.0)
    # aggregate
    case("(4,3,2,1) -> a", aggregate(list("aaaabbbccd"))[0], "a")
    case("top tie -> d/TIE", aggregate(list("aaaabbbbcc"))[:2], ("d", "AGGREGATE_TIE"))
    case("d unique mode -> d/CELL_D_MODE", aggregate(list("ddddabcabc"))[:2], ("d", "CELL_D_MODE"))
    # primary boundary and full branch
    def t34_rows(deltas):
        fr, fe = {}, {}
        for Z in ELEMS:
            for i, s in enumerate(SHELLS):
                fr[(s, Z, 2)] = (1.0, "8"); fr[(s, Z, 3)] = (1.0, "8")
                d = deltas[i] if Z == 16 else 0.0
                fe[(s, Z, 2)] = (1.0, "8"); fe[(s, Z, 3)] = (10.0 ** d, "8")
        return y34_table(fr, fe)
    rows, why = t34_rows([-0.5, -0.5, -0.5, 0.0, 0.0])
    n = sum(1 for r in rows if r["Z"] == 16 and r["delta34"] <= THR34)
    case("delta34 exactly -0.5 counts; 3 shells -> I34", (n >= 3, why), (True, []))
    rows, _ = t34_rows([-0.5, -0.5, nxt(-0.5, 0.0), 0.0, 0.0])
    n = sum(1 for r in rows if r["Z"] == 16 and r["delta34"] <= THR34)
    case("2 shells -> not I34", n >= 3, False)
    case("primary not met + cell b -> NT-X", branch(False, "b"), "NT-X")
    for agg, want in (("a", "NT-SUPPORT"), ("b", "NT-REV"), ("c", "NT-IV-ONLY"), ("d", "NT-UNDET")):
        case(f"primary met + {agg} -> {want}", branch(True, agg), want)
    case("Si/Ca do not change sulfur aggregate",
         aggregate(list("cccccccccc"))[0], "c")
    # FUV independence
    pts = [(900.0, 900.0), (1600.0, 1600.0), (3900.0, 3900.0), (6100.0, 6100.0)]
    v, st = band_integral(pts, 912.0, 1500.0)
    case("linear spectrum f=lambda known integral 912-1500 (edges interpolated)",
         (round(v, 6), st), (round((1500.0 ** 2 - 912.0 ** 2) / 2.0, 6), "OK"))
    case("unsupported band rejected", band_integral([(1000.0, 1.0), (2000.0, 1.0)], 912.0, 1500.0)[1], "UNSUPPORTED")
    case("FUV -0.7 with c keeps NT-IV-ONLY", branch(True, "c"), "NT-IV-ONLY")
    # missing / identity faults
    fr, fe = {}, {}
    for Z in ELEMS:
        for s in SHELLS:
            for q in (2, 3):
                fr[(s, Z, q)] = (1.0, "8"); fe[(s, Z, q)] = (1.0, "8")
    fe[(24, 16, 3)] = (0.0, "8")
    case("non-positive IV -> undefined reason", y34_table(fr, fe)[1], ["Y34_UNDEFINED:Z16/sh24"])
    fe[(24, 16, 3)] = (float("nan"), "8")
    case("NaN IV -> undefined reason", y34_table(fr, fe)[1], ["Y34_UNDEFINED:Z16/sh24"])
    fe[(24, 16, 3)] = (1.0, "7")
    case("wrong generation -> undefined reason", y34_table(fr, fe)[1], ["Y34_UNDEFINED:Z16/sh24"])
    it = {k: {10: {"population_generation": str(GEN_K[k]), "Z16_stage_q1_cm3": "1", "Z16_stage_q2_cm3": "2"}} for k in (3, 4, 5)}
    case("y23 complete", y23_series(it, 16, "stage", 10)[1], None)
    it[4][10]["population_generation"] = "6"
    case("y23 wrong generation", y23_series(it, 16, "stage", 10)[1], "Y23_GENERATION:k4")
    it[4][10]["population_generation"] = "7"
    del it[5][10]
    case("y23 missing row", y23_series(it, 16, "stage", 10)[1], "Y23_ROW_MISSING:k5")
    rr, ee = "/R", "/E"
    base = {k: f"{rr}/input/x" for k in PATH_KEYS}
    base.update(MUST_KEEP)
    dr = dict(base, LUMINA_NT_FULL="1")
    de = {k: (v.replace(rr, ee, 1) if k in PATH_KEYS else v) for k, v in dr.items()}
    de["LUMINA_NT_FULL"] = "0"
    case("exports: only NT_FULL 1->0", exports_delta(dr, de, rr, ee), [])
    de2 = dict(de, LUMINA_ION_NT="0")
    case("exports: extra diff detected", "EXPORTS_EXTRA_DIFF:LUMINA_ION_NT" in exports_delta(dr, de2, rr, ee), True)
    de3 = dict(de, LUMINA_NT_FULL="1")
    case("exports: NT_FULL unchanged detected", any(w.startswith("EXPORTS_NT_FULL") for w in exports_delta(dr, de3, rr, ee)), True)
    de4 = dict(de)
    de4["LUMINA_MODEL_DIR"] = "/other/input/x"
    case("exports: path not a prefix transplant", "EXPORTS_PATH:LUMINA_MODEL_DIR" in exports_delta(dr, de4, rr, ee), True)
    # sol Q4: injected faults on the real gate / loader / aux paths
    yr = {3: 0.0, 4: 0.0, 5: 0.0}
    ye = {3: 0.0, 4: -2.0, 5: 0.0}
    c = y23_cell(yr, ye)
    case("log mean -1.0 (arithmetic ratio mean would give -0.297)", round(c["delta23"], 12), -1.0)
    case("D uses k3/k5 only (k4 excursion ignored)", c["D"], 0.0)
    tmp = out / "nc_tmp"
    (tmp / "work/physics_comparison").mkdir(parents=True)
    (tmp / "input").mkdir()
    with open(tmp / "work/physics_comparison/physics_DET_iter0004.ion.csv", "w") as fh:
        fh.write("population_generation,shell_id,Z16_stage_q1_cm3\n7,10,1\n7,10,1\n")
    case("duplicate ion row counted", load_ion_iter(tmp, 4)[1], 1)
    (tmp / "input/fixed_te_profile.txt").write_text("10 1.0000E+04\n")
    gr = grun(tmp, "COMPLETED RESTARTS=0")
    case("missing manifests/files/footer -> G-RUN reasons",
         all(any(w.startswith(p) for w in gr) for p in ("FOOTER_NOT_SUCCESS", "MANIFEST_MISSING:k0",
                                                        "FINAL_PUBLICATION_MISSING", "EXACT_SOLVE_OK:0",
                                                        "FILE_MISSING:lumina_spectrum.csv")), True)
    good = {"terminal_state": "SUCCESS", "model_rc": "0", "allocated_gpus": "2", "gpu_kind": "H100",
            "outer_iterations": "6", "fixed_te_loader_consumed": "1"}
    case("footer good", footer_reasons(good, "COMPLETED RESTARTS=0"), [])
    case("model_rc!=0 detected", footer_reasons(dict(good, model_rc="1"), "COMPLETED RESTARTS=0"), ["MODEL_RC:1"])
    case("slurm FAILED detected", footer_reasons(good, "FAILED RESTARTS=0"), ["SLURM_STATE:FAILED RESTARTS=0"])
    case("restart detected", footer_reasons(good, "COMPLETED RESTARTS=1"), ["SLURM_RESTARTS"])
    case("H200 rejected", footer_reasons(dict(good, gpu_kind="H200"), "COMPLETED RESTARTS=0"), ["GPU_IDENTITY"])
    case("iterations != 6 rejected", footer_reasons(dict(good, outer_iterations="15"), "COMPLETED RESTARTS=0"),
         ["OUTER_ITERATIONS:15"])
    with open(tmp / "work/physics_comparison/physics_DET_iter0005.shell.csv", "w") as fh:
        fh.write("shell_id,T_e_K\n10,10001\n")
    case("published T_e != fixed profile detected", "TE_PUBLISHED_NE_PROFILE" in te_reasons(tmp, {10: 1.0e4})[0], True)
    ln_, lx_ = 1.0e14, 1.0e16
    dl = (math.log(lx_) - math.log(ln_)) / 4
    grid = [(math.exp(math.log(ln_) + b * dl), math.exp(math.log(ln_) + (b + 1) * dl), 1.0) for b in range(4)]
    case("sigma grid matches", grid_reason(grid, ln_, lx_, 4), None)
    case("sigma grid count mismatch", grid_reason(grid[:3], ln_, lx_, 4), "GRID_COUNT:3!=4")
    bad = list(grid)
    bad[2] = (bad[2][0] * (1 + 1e-6), bad[2][1], 1.0)
    case("sigma grid edge shift detected", grid_reason(bad, ln_, lx_, 4), "GRID_EDGE:bin2")
    for nm, scale in (("R", 1.0), ("E", 10.0 ** -0.7)):
        d = tmp / f"fuv_{nm}/work"
        d.mkdir(parents=True)
        with open(d / "lumina_spectrum.csv", "w") as fh:
            fh.write("wavelength_angstrom,flux\n")
            for lam in (800.0, 1000.0, 1400.0, 1600.0, 3900.0, 6100.0):
                fh.write(f"{lam},{(scale if lam < 2000 else 1.0)}\n")
    fr_, fe_ = fuv(tmp / "fuv_R"), fuv(tmp / "fuv_E")
    dfuv = math.log10(fe_["F"]) - math.log10(fr_["F"])
    case("FUV -0.7 injected is reported and branch with c stays NT-IV-ONLY",
         (round(dfuv, 12), branch(True, "c")), (-0.7, "NT-IV-ONLY"))
    (tmp / "cr/input/model").mkdir(parents=True)
    (tmp / "cr/input/global_atomic").mkdir(parents=True)
    (tmp / "cr/input/model/cmfgen_sigma_bf.bin").write_bytes(b"x")
    (tmp / "cr/input/sigma.sha256").write_text("0" * 64)
    (tmp / "cr/input/model/a.csv").write_text("a")
    (tmp / "cr/input/deck.sha256").write_text("0" * 64 + "  a.csv\n")
    (tmp / "cr/input/global_atomic/t.csv").write_text("t")
    (tmp / "cr/input/topion.sha256").write_text("0" * 64 + "  t.csv\n")
    case("sigma/deck/topion content tampering detected", content_reasons(tmp / "cr", "R"),
         ["SIGMA_CONTENT:R", "DECK_CONTENT:R:1", "TOPION_CONTENT:R:1"])
    # closeout C1: Γ band boundary, level weights, all sulfur count distributions, Si/Ca non-interference
    lam_c = [911.9, 912.0, 1499.9, 1500.0]
    rows_t = []
    for lam in lam_c:
        nu = C_CGS / (lam * 1e-8)
        rows_t.append((nu * (1 - 1e-6), nu * (1 + 1e-6), 1.0))
    case("D band: bin centres 912.0 and 1499.9 in, 911.9 and 1500.0 out",
         [in_band_D(math.sqrt(lo * hi)) for lo, hi, _ in rows_t], [False, True, True, False])
    Lt = [(0, 0.0, 2.0, 3.0), (1, 1.0, 4.0, 1.0)]
    T_t = 1.0e4
    e1 = 4.0 * math.exp(-1.0 * EV / (K_CGS * T_t))
    case("LTE weights g·exp(-E/kT) normalised", [round(x, 12) for x in level_weights(Lt, T_t, "LTE")],
         [round(2.0 / (2.0 + e1), 12), round(e1 / (2.0 + e1), 12)])
    case("NLTE weights = populations normalised", level_weights(Lt, T_t, "NLTE"), [0.75, 0.25])
    has_t = [1, 0]
    nfb_t = len(rows_t)
    sig_t = [1e-18] * nfb_t + [5e-18] * nfb_t
    gd, ga = gamma_band(Lt, rows_t, has_t, sig_t, nfb_t, 1.0, T_t, "NLTE")
    exp_all = sum(4.0 * math.pi * 1e-18 * 1.0 / (H_CGS * math.sqrt(lo * hi)) * (hi - lo) for lo, hi, _ in rows_t) * 0.75
    exp_d = sum(4.0 * math.pi * 1e-18 * 1.0 / (H_CGS * math.sqrt(lo * hi)) * (hi - lo)
                for lo, hi, _ in rows_t if in_band_D(math.sqrt(lo * hi))) * 0.75
    case("Γ: has=0 level skipped, NLTE weight 0.75, D = 2 of 4 bins",
         (math.isclose(ga, exp_all, rel_tol=1e-12), math.isclose(gd, exp_d, rel_tol=1e-12)), (True, True))
    gdl, _ = gamma_band(Lt, rows_t, has_t, sig_t, nfb_t, 1.0, T_t, "LTE")
    case("Γ: LTE and NLTE weights give different D rates", gdl != gd, True)
    from itertools import product
    bad = 0
    n_dist = 0
    for na in range(11):
        for nb in range(11 - na):
            for nc in range(11 - na - nb):
                nd = 10 - na - nb - nc
                n_dist += 1
                cells = ["a"] * na + ["b"] * nb + ["c"] * nc + ["d"] * nd
                got = aggregate(cells)[0]
                cnt = {"a": na, "b": nb, "c": nc, "d": nd}
                top = max(cnt.values())
                modes = [k for k in "abcd" if cnt[k] == top]
                want = modes[0] if len(modes) == 1 else "d"
                bad += got != want
    case("all 286 sulfur count distributions: unique mode or tie -> d", (n_dist, bad), (286, 0))
    t_mix = [{"Z": 16, "outcome": "a"}] * 4 + [{"Z": 16, "outcome": "c"}] * 6 + \
            [{"Z": 14, "outcome": "a"}] * 10 + [{"Z": 20, "outcome": "a"}] * 10
    case("Si/Ca cells never enter the sulfur aggregate", aggregate(sulfur_outcomes(t_mix))[0], "c")
    case("binary mismatch detected", bins_reason({"a": W5_BIN, "b": "0" * 64}), ["BINARY_MISMATCH"])
    case("approved source mismatch detected",
         approved_reasons({"src/x.c": "aa"}, "src/x.c=bb"), ["APPROVED_MISMATCH:src/x.c"])
    case("FIXED_TE not consumed detected", fixed_te_reason({"fixed_te_loader_consumed": "0"}, "E"),
         ["FIXED_TE_NOT_CONSUMED:E"])
    # normal control: W5 vs W5 statistics -> I34 false -> NT-X (statistics only, not a G-ID pass)
    if a.ref:
        f, _ = load_final_stages(a.ref)
        t, w = y34_table(f, f)
        n = sum(1 for r in t if r["Z"] == 16 and "delta34" in r and r["delta34"] <= THR34)
        case("W5 vs W5: delta34=0, primary not met -> NT-X", (w, n, branch(n >= 3, "c")), ([], 0, "NT-X"))
        its = {k: load_ion_iter(a.ref, k)[0] for k in (3, 4, 5)}
        outs = []
        for A in AUTHS:
            for s in SHELLS:
                y, e = y23_series(its, 16, A, s)
                outs.append(y23_cell(y, y)["outcome"] if not e else e)
        case("W5 vs W5 y23 cells: delta=0 -> c or d only", all(o in ("c", "d") for o in outs), True)
    nfail = sum(1 for r in res if not r["pass"])
    (out / "readout_selftest.json").write_text(json.dumps(res, indent=1))
    print(f"\nNT-ETA NC: {len(res) - nfail}/{len(res)} -> {'PASS' if nfail == 0 else 'FAIL'}")
    return 0 if nfail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("gid", "readout"):
        p = sp.add_parser(nm)
        p.add_argument("--ref", required=True)
        p.add_argument("--test", required=True)
        p.add_argument("--out", required=True)
        if nm == "readout":
            p.add_argument("--selftest", required=True, help="readout_selftest.json (G-READOUT)")
            p.add_argument("--slurm-states", required=True,
                           help='JSON {"R": "COMPLETED RESTARTS=0", "E": "..."} recorded by the driver from sacct')
    p = sp.add_parser("selftest")
    p.add_argument("--ref")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"gid": cmd_gid, "readout": cmd_readout, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
