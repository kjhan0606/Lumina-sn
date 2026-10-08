#!/usr/bin/env python3
"""DET-COLL-OMEGA-OFFLINE readout (prereg docs/PREREG_DET_COLL_OMEGA_OFFLINE_2026-10-08.md).

Offline only: no production change, no production run, GPU 0.  On the stored R' material state (population
generation 7 at the W5 shells; generation 3 for the line 471957 / shell 10 counterexample) the readout evaluates,
for the fixed Si III / S III sample (the diag6 census keys, 510) and its one-step connected level pairs, the
collision rates of two sealed production prescriptions:
  COLL_FIX  - the per-line branch the stored runs consumed;
  OMEGA     - the LUMINA_OMEGA_CMFGEN branch: tabulated Upsilon (IGC1 tables, linear in T, tier 1, added by the
              general col_data pass), van Regemorter g-bar fallback (tier 2) or OMEGA_SET (tier 3) through the
              per-line loop, same-solve-unit table pairs skipped.
Rates are atomic level-pair rates before within-solve-unit fractions.  Every value is computed twice: by the
Python implementation here and by a CPU reference that compiles verbatim excerpts of the sealed C (loader, arm,
tier selection, per-line block, col_data pass) behind a small adapter; they must agree to 1e-12.

Outputs: sample.json, rates.csv, uncovered.csv, result.json, next_contract.json, manifest.json (analyze);
selftest.json (selftest).  Codes: 0 SURVIVES, 1 REFUTED, 4 UNRESOLVED, 5 NOT_IDENTIFIABLE, X inadmissible.

Subcommands
  selftest --production-source SRC --out DIR
  analyze  --prereg DOC --census CSV --prior-sample JSON --prior-result JSON --r-run RUN --production-source SRC
           --selftest DIR --out DIR
"""
import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_siiii_pop_readout as POP                                              # noqa: E402
import det_siiii_rates_readout as RATES                                          # noqa: E402
import det_seedmem_scope_readout as SEED                                         # noqa: E402

PREREG = "docs/PREREG_DET_COLL_OMEGA_OFFLINE_2026-10-08.md"
SHELLS = (10, 15, 19, 24, 30)
MAIN_GEN, COUNTER_GEN = 7, 3
H_PLANCK, K_B, EV_TO_ERG = 6.62607015e-27, 1.380649e-16, 1.602176634e-12        # lumina.h
COL_CONST = 8.629e-6                                                             # ARTIS_COL_CONST
OMCM_VR_PREF, OMCM_GBAR_ION, OMCM_FMIN, OMCM_SET, OMCM_HZ15 = 47.972, 0.2, 1.0e-5, 0.1, 1.0e15
CRIT_REF = 1e-12              # Python vs sealed C, positive rates (prereg 6.2)
CRIT_DB = 1e-12               # detailed balance, implementation consistency (prereg 5.2)
TWO = 2.0                     # H-SMALL (prereg 5.3)
BOUND_EPS = 1e-10             # |log10 R| within this of log10 2 -> UNRESOLVED (prereg 6.2)
PRIORITY = 0.20               # ordering threshold (prereg 5.3)
CODE = {"SURVIVES": 0, "REFUTED": 1, "UNRESOLVED": 4, "NOT_IDENTIFIABLE": 5}
TARGET = dict(SEED.TARGET)


def det_coll_omega_offline_rel(a, b):
    if a == b:
        return 0.0
    m = max(abs(a), abs(b))
    return abs(a - b) / m if m > 0 else math.inf


# ------------------------------------------------------------------ sealed C excerpts (prereg 8.2)
def det_coll_omega_offline_braces(lines, i0):
    """end index (inclusive) of the C block opened on line i0, counting braces outside strings / comments."""
    depth, started, in_block = 0, False, False
    for i in range(i0, len(lines)):
        s, j, in_str, in_chr = lines[i], 0, False, False
        while j < len(s):
            c = s[j]
            if in_block:
                if s.startswith("*/", j):
                    in_block, j = False, j + 2
                    continue
            elif in_str:
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    in_str = False
            elif in_chr:
                if c == "\\":
                    j += 2
                    continue
                if c == "'":
                    in_chr = False
            elif s.startswith("//", j):
                break
            elif s.startswith("/*", j):
                in_block, j = True, j + 2
                continue
            elif c == '"':
                in_str = True
            elif c == "'":
                in_chr = True
            elif c == "{":
                depth, started = depth + 1, True
            elif c == "}":
                depth -= 1
            j += 1
        if started and depth == 0:
            return i
    return None


EXCERPTS = (
    ("artis_consts", "lumina_plasma.c", "lines", "#define ARTIS_C_0", "#define ARTIS_GAUNT_UCROSS"),
    ("artis_parity_enabled", "lumina_plasma.c", "func", "int artis_parity_enabled(void) {", None),
    ("artis_col_rates", "lumina_plasma.c", "func", "static inline void artis_col_rates(double T_e, double n_e, double dE,",
     None),
    ("ion_has_realcoldata", "lumina_plasma.c", "func",
     "static int ion_has_realcoldata(const AtomicData *atom, int Z, int ion0) {", None),
    ("ma_ru", "lumina_plasma.c", "to_func_end",
     "typedef struct { int ntemp; const double *tgrid; const double *omega; } MaRuSrc;",
     "static inline double ma_ru_upsilon(const MaRuSrc *S, int t, double T_e) {"),
    ("omcm", "lumina_plasma.c", "to_func_end", "static int find_ion_pop_idx(AtomicData *atom, int Z, int ion_stage);",
     "void omega_cmfgen_arm(AtomicData *atom) {"),
    ("find_ion_pop_idx", "lumina_plasma.c", "func", "static int find_ion_pop_idx(AtomicData *atom, int Z, int ion_stage) {",
     None),
    ("vr_axelrod", "lumina_plasma.c", "lines", "#define VAN_REGEMORTER_COEFF", "#define AXELROD_OMEGA"),
    ("coll_fix_enabled", "lumina_plasma.c", "func", "int nlte_coll_fix_enabled(void) {", None),
    ("load_ion_coldata", "lumina_atomic.c", "func", "int load_ion_coldata(AtomicData *atom, const char *path) {", None),
    ("load_manifest", "lumina_atomic.c", "func", "int load_ion_coldata_manifest(AtomicData *atom, const char *ref_dir) {",
     None),
    ("line_block", "lumina_plasma.c", "line_block", None, None),
    ("coldata_pass", "lumina_plasma.c", "func",
     "if ((parity_on || omcm_on) && atom->ncol_ions > 0 && T_e > 0.0 && n_e > 0.0) {", None),
)


def det_coll_omega_offline_excerpt(src, spec):
    """cut one excerpt verbatim from a sealed source (unique markers; brace-matched blocks)."""
    name, fname, kind, a, b = spec
    text = (Path(src) / fname).read_text(errors="replace")
    lines = text.split("\n")

    def uniq(marker):
        hits = [i for i, ln in enumerate(lines) if ln.strip().startswith(marker.strip())]
        return hits[0] if len(hits) == 1 else None
    if kind == "lines":
        i0, i1 = uniq(a), uniq(b)
    elif kind == "func":
        i0 = uniq(a)
        i1 = None if i0 is None else det_coll_omega_offline_braces(lines, i0)
    elif kind == "to_func_end":
        i0, f0 = uniq(a), uniq(b)
        i1 = None if f0 is None else det_coll_omega_offline_braces(lines, f0)
    else:                                   # line_block: per-line collision dispatch of the CPU assembler
        k = [i for i, ln in enumerate(lines) if ln.strip() == "double ups = omega_cmfgen_line(atom, line, T_e, &tier);"]
        i0 = i1 = None
        if len(k) == 1:
            st = [i for i in range(k[0], max(0, k[0] - 40), -1)
                  if lines[i].strip().startswith("double dE = fabs(atom->level_energy_eV[upper_global] -")]
            en = [i for i in range(k[0], min(len(lines), k[0] + 200))
                  if lines[i].strip() == "double total_up   = R_absorb + C_up;"]
            if st and en:
                i0, i1 = st[0], en[0] - 1
    if i0 is None or i1 is None or i1 < i0:
        return {"name": name, "file": fname, "ok": False}
    body = "\n".join(lines[i0:i1 + 1]) + "\n"
    return {"name": name, "file": fname, "ok": True, "start_line": i0 + 1, "end_line": i1 + 1, "text": body,
            "sha256": hashlib.sha256(body.encode()).hexdigest(), "verbatim": body.rstrip("\n") in text}


CREF_ADAPTER = r'''
/* DET-COLL-OMEGA-OFFLINE CPU reference: verbatim excerpts of the sealed source behind a minimal adapter.
 * The adapter only provides arrays, scalars and the assembler locals; every formula / branch is the excerpt. */
#include "lumina.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static AtomicData A;
static NLTEConfig NL;
static FILE *g_out;
#include "x_artis_consts.inc"
#include "x_artis_parity_enabled.inc"
#include "x_artis_col_rates.inc"
#include "x_ion_has_realcoldata.inc"
#include "x_ma_ru.inc"
#include "x_vr_axelrod.inc"
#include "x_coll_fix_enabled.inc"
#include "x_omcm.inc"
#include "x_find_ion_pop_idx.inc"
#include "x_load_ion_coldata.inc"
#include "x_load_manifest.inc"

static double g_coll_floor = 0.0;
static void line_block(int line, int lower_global, int upper_global, double T_e, double n_e, int omcm_on,
                       double *out_up, double *out_down) {
    AtomicData *atom = &A;
    int ew_capture = 0;
    double coll_floor = g_coll_floor;
    int feiii_coldata_on = 0;
    (void)ew_capture; (void)coll_floor; (void)feiii_coldata_on;
#include "x_line_block.inc"
    *out_up = C_up; *out_down = C_down;
}

static double g_acm_sink = 0.0;
static int *g_solve_of = NULL;
static int g_entry = 0;
#define ACM(i, j) (g_acm_sink)
#define SOLVE_OF(fl) (g_solve_of[(fl)])
#define FRAC_OF(fl) (1.0)
#define nlte_ew_capture_transition(...) ((void)0)
#define DDC_COLL(a, b, cu, cd) fprintf(g_out, "T %d %d %d %d %d %d %.17g %.17g\n", g_entry, Zi, ioni, lo_ln, hi_ln, t, (cu), (cd))
static void coldata_pass(double T_e, double n_e, int ion_idx, int N) {
    AtomicData *atom = &A;
    NLTEConfig *nlte = &NL;
    int parity_on = 0, omcm_on = 1;
    int ion_idx_lo = ion_idx, ion_idx_hi = ion_idx;
#include "x_coldata_pass.inc"
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *in = fopen(argv[1], "r");
    g_out = fopen(argv[2], "w");
    if (!in || !g_out) return 2;
    char model[4096];
    if (fscanf(in, "MODEL %4095s COLL_FLOOR %lf", model, &g_coll_floor) != 2) return 2;
    int nlev;
    if (fscanf(in, " NLEV %d", &nlev) != 1) return 2;
    memset(&A, 0, sizeof(A)); memset(&NL, 0, sizeof(NL));
    A.n_levels = nlev;
    A.level_Z = calloc(nlev, sizeof(int)); A.level_ion = calloc(nlev, sizeof(int));
    A.level_num = calloc(nlev, sizeof(int)); A.level_energy_eV = calloc(nlev, sizeof(double));
    A.level_g = calloc(nlev, sizeof(int));
    for (int l = 0; l < nlev; l++)
        if (fscanf(in, "%d %d %d %lf %d", &A.level_Z[l], &A.level_ion[l], &A.level_num[l],
                   &A.level_energy_eV[l], &A.level_g[l]) != 5) return 2;
    /* element / ion-pop ladder in order of appearance (contiguous blocks required) */
    A.element_Z = calloc(nlev + 1, sizeof(int)); A.elem_ion_offset = calloc(nlev + 2, sizeof(int));
    A.ion_pop_stage = calloc(nlev + 1, sizeof(int)); A.level_offset = calloc(nlev + 2, sizeof(int));
    int ne_ = 0, nip = 0;
    for (int l = 0; l < nlev; l++) {
        if (l == 0 || A.level_Z[l] != A.level_Z[l - 1]) {
            for (int e = 0; e < ne_; e++) if (A.element_Z[e] == A.level_Z[l]) { fprintf(g_out, "E NONCONTIGUOUS_Z\n"); return 4; }
            A.element_Z[ne_] = A.level_Z[l]; A.elem_ion_offset[ne_] = nip; ne_++;
        }
        if (l == 0 || A.level_Z[l] != A.level_Z[l - 1] || A.level_ion[l] != A.level_ion[l - 1]) {
            A.ion_pop_stage[nip] = A.level_ion[l]; A.level_offset[nip] = l; nip++;
        }
    }
    A.elem_ion_offset[ne_] = nip; A.level_offset[nip] = nlev; A.n_elements = ne_;
    int ntg;
    if (fscanf(in, " NTARGET %d", &ntg) != 1) return 2;
    NL.global_to_nlte_level = malloc(nlev * sizeof(int));
    for (int l = 0; l < nlev; l++) NL.global_to_nlte_level[l] = -1;
    NL.nlte_to_global_level = malloc(nlev * sizeof(int));
    int *unit_of_nlte = malloc(nlev * sizeof(int));
    int nunits[64]; int off = 0;
    for (int k = 0; k < ntg; k++) {
        int Z, ion, n;
        if (fscanf(in, "%d %d %d %d", &Z, &ion, &n, &nunits[k]) != 4) return 2;
        NL.nlte_Z[k] = Z; NL.nlte_ion[k] = ion; NL.nlte_ion_level_offset[k] = off;
        for (int i = 0; i < n; i++) {
            int gl, u;
            if (fscanf(in, "%d %d", &gl, &u) != 2) return 2;
            NL.nlte_to_global_level[off] = gl; NL.global_to_nlte_level[gl] = off; unit_of_nlte[off] = u; off++;
        }
        NL.nlte_ion_level_offset[k + 1] = off;
    }
    NL.n_nlte_ions = ntg;
    int nl;
    if (fscanf(in, " NLINE %d", &nl) != 1) return 2;
    A.n_lines = nl;
    A.line_atomic_number = calloc(nl, sizeof(int)); A.line_ion_number = calloc(nl, sizeof(int));
    A.line_level_lower = calloc(nl, sizeof(int)); A.line_level_upper = calloc(nl, sizeof(int));
    A.line_f_lu = calloc(nl, sizeof(double)); A.line_A_ul = calloc(nl, sizeof(double));
    int *lg = calloc(nl, sizeof(int)), *ug = calloc(nl, sizeof(int));
    for (int l = 0; l < nl; l++)
        if (fscanf(in, "%d %d %d %d %lf %lf %d %d", &A.line_atomic_number[l], &A.line_ion_number[l],
                   &A.line_level_lower[l], &A.line_level_upper[l], &A.line_f_lu[l], &A.line_A_ul[l],
                   &lg[l], &ug[l]) != 8) return 2;
    int nent;
    if (fscanf(in, " NENTRY %d", &nent) != 1) return 2;
    int omcm = omega_cmfgen_enabled();
    fprintf(g_out, "C omcm=%d coll_fix=%d parity=%d\n", omcm, nlte_coll_fix_enabled(), artis_parity_enabled());
    if (omcm) {
        int nload = load_ion_coldata_manifest(&A, model);
        fprintf(g_out, "M %d\n", nload);
        if (nload < 0) { fclose(g_out); return 3; }
        omega_cmfgen_arm(&A);
    }
    g_solve_of = malloc((off > 0 ? off : 1) * sizeof(int));
    for (int e = 0; e < nent; e++) {
        int sid; double Te, ne;
        if (fscanf(in, "%d %d %lf %lf", &g_entry, &sid, &Te, &ne) != 4) return 2;
        for (int l = 0; l < nl; l++) {
            double cu = 0.0, cd = 0.0; int tier = -1; double ups = -1.0;
            line_block(l, lg[l], ug[l], Te, ne, omcm, &cu, &cd);
            if (omcm) ups = omega_cmfgen_line(&A, l, Te, &tier);
            fprintf(g_out, "L %d %d %.17g %.17g %d %.17g\n", g_entry, l, cu, cd, tier, ups);
        }
        if (omcm)
            for (int k = 0; k < ntg; k++) {
                for (int i = 0; i < off; i++) g_solve_of[i] = -1;
                for (int i = NL.nlte_ion_level_offset[k]; i < NL.nlte_ion_level_offset[k + 1]; i++)
                    g_solve_of[i] = unit_of_nlte[i];
                coldata_pass(Te, ne, k, nunits[k]);
            }
    }
    fprintf(g_out, "END\n");
    fclose(g_out);
    return 0;
}
'''


def det_coll_omega_offline_build_cref(src, out, fault=None):
    """write the excerpts and the adapter into the result path and compile them on the CPU (no production build)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rec = {"excerpts": [], "why": []}
    for spec in EXCERPTS:
        x = det_coll_omega_offline_excerpt(src, spec)
        if not x.get("ok") or not x.get("verbatim"):
            rec["why"].append(f"EXCERPT_NOT_FOUND:{spec[0]}")
            continue
        body = x["text"]
        if fault == "cref_tamper" and spec[0] == "omcm":
            body = body.replace("#define OMCM_FMIN       1.0e-5", "#define OMCM_FMIN       2.0e-5")
        (out / f"x_{spec[0]}.inc").write_text(body)
        rec["excerpts"].append({k: v for k, v in x.items() if k != "text"} |
                               {"written_sha256": hashlib.sha256(body.encode()).hexdigest()})
    for e in rec["excerpts"]:
        if e["written_sha256"] != e["sha256"]:
            rec["why"].append(f"EXCERPT_NOT_VERBATIM:{e['name']}")
    (out / "cref.c").write_text(CREF_ADAPTER)
    cmd = ["gcc", "-O2", "-std=gnu11", "-w", "-I", str(src), "-o", str(out / "cref"), str(out / "cref.c"), "-lm"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    rec.update({"cmd": " ".join(cmd), "rc": p.returncode, "stderr_tail": p.stderr[-3000:], "elapsed_s": time.time() - t0,
                "adapter_sha256": POP.sha(out / "cref.c"),
                "sources_sha256": {f: POP.sha(Path(src) / f) for f in ("lumina_plasma.c", "lumina_atomic.c", "lumina.h")},
                "binary": str(out / "cref")})
    if p.returncode != 0:
        rec["why"].append("CREF_BUILD_FAILED")
    return rec


# ------------------------------------------------------------------ inputs
def det_coll_omega_offline_levels(model_dir):
    lv = pd.read_csv(Path(model_dir) / "levels.csv", float_precision="round_trip")
    rows = [(int(r.atomic_number), int(r.ion_number), int(r.level_number), float(r.energy_eV), int(r.g))
            for r in lv.itertuples()]
    idx = {(z, q, n): i for i, (z, q, n, _, _) in enumerate(rows)}
    counts = {}
    for z, q, *_ in rows:
        counts[(z, q)] = counts.get((z, q), 0) + 1
    return {"rows": rows, "idx": idx, "counts": counts}


def det_coll_omega_offline_lines(model_dir, Zs, ion):
    """target-ion lines (line_id = 0-based data row, checked)."""
    cols = ["atomic_number", "ion_number", "level_number_lower", "level_number_upper", "line_id", "f_lu", "A_ul", "nu"]
    out, why, base = [], [], 0
    for ch in pd.read_csv(Path(model_dir) / "line_list.csv", usecols=cols, chunksize=1 << 20,
                          float_precision="round_trip"):
        idx = np.arange(base, base + len(ch))
        base += len(ch)
        m = ch.atomic_number.isin(Zs).values & (ch.ion_number.values == ion)
        for r, row in zip(ch[m].itertuples(), idx[m]):
            if int(r.line_id) != int(row):
                why.append("DECK_LINE_ID_NOT_ROW")
            out.append({"line_id": int(r.line_id), "Z": int(r.atomic_number), "ion": int(r.ion_number),
                        "lo": int(r.level_number_lower), "up": int(r.level_number_upper), "f_lu": float(r.f_lu),
                        "A_ul": float(r.A_ul), "nu": float(r.nu)})
    return out, sorted(set(why))


def det_coll_omega_offline_bin(path, nlev_ion, fault=None):
    """IGC1 table, independent Python reader with the production loader's checks (fail-closed)."""
    b = Path(path).read_bytes() if Path(path).exists() else None
    if b is None:
        return None, "TABLE_FILE_MISSING"
    if len(b) < 28:
        return None, "TABLE_HEADER_SHORT"
    magic, version, Z, ion0, ntr, ntp, nlr = struct.unpack("<IIiiiii", b[:28])
    if magic != 0x49474331 or version != 1:
        return None, "TABLE_MAGIC_VERSION"
    if ntr <= 0 or ntp <= 0 or ntp > 256:
        return None, "TABLE_COUNTS"
    if nlr != nlev_ion(Z, ion0):
        return None, f"TABLE_LEVEL_COUNT:{Z}:{ion0}:{nlr}"
    pos = 28
    if len(b) < pos + 8 * ntp:
        return None, "TABLE_TGRID_SHORT"
    tg = list(struct.unpack(f"<{ntp}d", b[pos:pos + 8 * ntp]))
    pos += 8 * ntp
    lo, hi, om = [], [], []
    rec = 8 + 8 * ntp
    nlev = nlev_ion(Z, ion0)
    for t in range(ntr):
        if len(b) < pos + rec:
            return None, f"TABLE_RECORD_SHORT:{t}"
        il, ih = struct.unpack("<ii", b[pos:pos + 8])
        if fault == "level_shift":
            il, ih = il + 1, ih + 1
        o = list(struct.unpack(f"<{ntp}d", b[pos + 8:pos + rec]))
        pos += rec
        if il < 0 or ih < 0 or il >= nlev or ih >= nlev:
            return None, f"TABLE_LEVEL_RANGE:{t}"
        lo.append(il)
        hi.append(ih)
        om.append(o)
    return {"Z": Z, "ion0": ion0, "n_trans": ntr, "n_temp": ntp, "tgrid": tg, "lo": lo, "hi": hi, "omega": om,
            "path": str(path), "sha256": hashlib.sha256(b).hexdigest()}, None


def det_coll_omega_offline_manifest(model_dir, counts, fault=None):
    """every status==OK manifest row is loaded (production fails the whole load on any failure)."""
    p = Path(model_dir) / "coldata_cmfgen_manifest.csv"
    if not p.exists():
        return {}, ["MANIFEST_MISSING"], []
    rows = p.read_text().split("\n")[1:]
    tables, why, listing = {}, [], []
    for ln in rows:
        f = ln.rstrip("\r").split(",")
        if len(f) < 4 or f[-1] != "OK":
            continue
        Z, ion0, b = int(f[1]), int(f[2]), f[-2]
        path = Path(b) if "/" in b else Path(model_dir) / b
        tab, err = det_coll_omega_offline_bin(path, lambda z, q: counts.get((z, q), 0), fault)
        listing.append({"Z": Z, "ion0": ion0, "bin": str(path), "error": err,
                        "sha256": None if tab is None else tab["sha256"]})
        if err:
            why.append(f"OMEGA_LOADER_FAILS:{Z}:{ion0}:{err}")
            continue
        if (tab["Z"], tab["ion0"]) != (Z, ion0):
            why.append(f"TABLE_HEADER_NOT_MANIFEST:{Z}:{ion0}")
            continue
        seen, contra, ident = {}, 0, 0
        for t, (a, b_) in enumerate(zip(tab["lo"], tab["hi"])):
            k = (min(a, b_), max(a, b_))
            if k in seen:
                if tab["omega"][seen[k]] != tab["omega"][t] or (tab["lo"][seen[k]], tab["hi"][seen[k]]) != (a, b_):
                    contra += 1
                else:
                    ident += 1
            seen[k] = t
        listing[-1]["identical_duplicates"], listing[-1]["contradictory_duplicates"] = ident, contra
        if contra:
            why.append(f"TABLE_CONTRADICTORY_DUPLICATE:{Z}:{ion0}:{contra}")
            continue
        tables[(Z, ion0)] = tab
    if not listing:
        why.append("MANIFEST_NO_OK_ROW")
    return tables, why, listing


def det_coll_omega_offline_dump(run, gen):
    """the generation-labelled physics dump (manifest checks; n_e, T_e by shell_id)."""
    d = Path(run) / "work/physics_comparison"
    for m in sorted(d.glob("physics_DET_iter*.manifest.json")):
        js = json.loads(m.read_text())
        if js.get("population_generation") != gen:
            continue
        why = []
        if js.get("transaction_status") != "COMMITTED":
            why.append("DUMP_NOT_COMMITTED")
        if js.get("te_lane") != "FIXED_T":
            why.append("DUMP_NOT_FIXED_T")
        s = pd.read_csv(d / js["shell_file"], float_precision="round_trip").set_index("shell_id")
        return {"manifest": str(m), "population_generation": gen, "radiation_generation": js.get("radiation_generation"),
                "te_generation": js.get("te_generation"), "ne": s.n_e_cm3.to_dict(), "Te": s.T_e_K.to_dict(),
                "files": [str(m), str(d / js["shell_file"])], "why": why}
    return {"why": [f"DUMP_GENERATION_ABSENT:{gen}"]}


# ------------------------------------------------------------------ production prescriptions (Python)
def det_coll_omega_offline_ex_e1x(x):
    """port of omcm_ex_e1x (CMFGEN subs/ex_e1x_fun.f; Abramowitz & Stegun 5.1.53 / 5.1.56)."""
    W = (-0.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857)
    A = (0.2677737343, 8.6347608925, 18.0590169730, 8.5733287401)
    B = (3.9584969228, 21.0996530827, 25.6329561486, 9.5733223454)
    if not x > 0.0:
        return 0.0
    if x <= 1.0:
        p = W[5]
        for i in range(4, -1, -1):
            p = p * x + W[i]
        return math.exp(x) * (p - math.log(x))
    num = den = 1.0
    for i in range(3, -1, -1):
        num = num * x + A[i]
        den = den * x + B[i]
    return num / den / x


def det_coll_omega_offline_gbar(x, zion):
    if zion <= 1:
        return 0.276 * det_coll_omega_offline_ex_e1x(x) if x <= 14.0 else 0.066 * (1.0 + 1.5 / x) / math.sqrt(x)
    g2 = 0.276 * det_coll_omega_offline_ex_e1x(x)
    return OMCM_GBAR_ION if OMCM_GBAR_ION > g2 else g2


def det_coll_omega_offline_interp(tab, t, T_e, fault=None):
    """tier-1 Upsilon: linear in T with the production clamp (ma_ru_upsilon); > 0 or -1."""
    tg, om = tab["tgrid"], tab["omega"][t]
    nt = len(tg)
    if nt < 2 or not T_e > 0.0:
        return -1.0, None
    if fault == "t10400":
        T_e = 10400.0
    ti = 0
    while ti < nt - 2 and T_e > tg[ti + 1]:
        ti += 1
    if fault == "swap_interval" and ti + 2 < nt:
        ti += 1
    den = tg[ti + 1] - tg[ti]
    frac = (T_e - tg[ti]) / den if den > 0.0 else 0.0
    frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
    if fault == "loglog" and om[ti] > 0 and om[ti + 1] > 0 and T_e > 0:
        lf = (math.log(T_e) - math.log(tg[ti])) / (math.log(tg[ti + 1]) - math.log(tg[ti]))
        lf = min(1.0, max(0.0, lf))
        ups = math.exp(math.log(om[ti]) + lf * (math.log(om[ti + 1]) - math.log(om[ti])))
    else:
        ups = om[ti] + frac * (om[ti + 1] - om[ti])
    return (ups if ups > 0.0 else -1.0), {"ti": ti, "frac": frac, "end_clamped": frac in (0.0, 1.0)}


def det_coll_omega_offline_fallback(f_lu, g_lo, dE, T_e, ion0, fault=None):
    fl = dE / (H_PLANCK * OMCM_HZ15)
    fmin_ok = (f_lu >= OMCM_FMIN) if fault == "f_boundary" else (f_lu > OMCM_FMIN)
    if not fmin_ok or not fl > 0.0 or not T_e > 0.0 or not g_lo > 0.0:
        return OMCM_SET, 3
    g = 1.0 if fault == "drop_glo" else g_lo
    flp = fl * fl if fault == "dE_power" else fl
    return OMCM_VR_PREF * det_coll_omega_offline_gbar(dE / (K_B * T_e), ion0 + 1) * f_lu * g / flp, 2


def det_coll_omega_offline_rates_from_ups(T_e, n_e, dE, g_lo, g_up, ups, fault=None):
    """artis_col_rates real-collision-strength branch (coll_str >= 0)."""
    if not (T_e > 0.0 and n_e > 0.0 and dE > 0.0 and g_lo > 0 and g_up > 0):
        return 0.0, 0.0
    s = math.sqrt(T_e)
    u = dE / (K_B * T_e)
    cd = n_e * COL_CONST * ups / g_up / s
    cu = n_e * COL_CONST * ups * math.exp(-u) / g_lo / s
    if fault == "swap_g":
        cu, cd = n_e * COL_CONST * ups * math.exp(-u) / g_up / s, n_e * COL_CONST * ups / g_lo / s
    if fault == "common_factor":
        cu, cd = 1.7 * cu, 1.7 * cd
    return cu, cd


def det_coll_omega_offline_python(levels, lines, tables, units, entries, fault=None):
    """Python evaluation of both prescriptions: per (entry, line) COLL_FIX and OMEGA line-loop rates with the tier,
    and per (entry, table transition) the col_data pass rates (same-solve-unit and non-positive cases recorded)."""
    L = levels["idx"]
    rows = levels["rows"]
    dm = {}
    for key, tab in tables.items():
        for t, (a, b) in enumerate(zip(tab["lo"], tab["hi"])):
            dm[(key, min(a, b), max(a, b))] = t                 # last record wins (production arm map)
    line_out, tab_out = {}, {}
    for e in entries:
        Te, ne = e["T_e"], e["n_e"]
        for li, ln in enumerate(lines):
            Z, q = ln["Z"], ln["ion"]
            lo_g, up_g = L.get((Z, q, ln["lo"])), L.get((Z, q, ln["up"]))
            if lo_g is None or up_g is None:
                line_out[(e["entry"], li)] = {"missing": "LEVEL_ABSENT"}
                continue
            E_l, g_l = rows[lo_g][3], rows[lo_g][4]
            E_u, g_u = rows[up_g][3], rows[up_g][4]
            fx_u, fx_d = RATES.det_siiii_rates_coll(ne, Te, E_l, E_u, g_l, g_u, ln["f_lu"])
            dE = abs(E_u - E_l) * EV_TO_ERG
            t = dm.get(((Z, q), min(ln["lo"], ln["up"]), max(ln["lo"], ln["up"])))
            ups, tier, info = -1.0, 3, None
            if t is not None:
                ups, info = det_coll_omega_offline_interp(tables[(Z, q)], t, Te, fault)
                if ups > 0.0:
                    tier = 1
            if tier != 1:
                ups, tier = det_coll_omega_offline_fallback(ln["f_lu"], g_l, dE, Te, q, fault)
            om_u = om_d = 0.0
            if tier != 1 or fault == "table_per_line":
                if Te > 0.0 and dE > 0.0 and g_l > 0 and g_u > 0:
                    om_u, om_d = det_coll_omega_offline_rates_from_ups(Te, ne, dE, g_l, g_u, ups, fault)
            if fault == "suppress_ion_fallback" and (Z, q) in tables:
                om_u = om_d = 0.0
            line_out[(e["entry"], li)] = {"fix_up": fx_u.mid, "fix_down": fx_d.mid, "om_up": om_u, "om_down": om_d,
                                          "tier": tier, "ups": ups, "table_t": t,
                                          "end_clamped": None if info is None else info["end_clamped"]}
        if fault == "omit_table_pass":
            continue
        for (Z, q), tab in tables.items():
            if (Z, q) not in units:
                continue
            u = units[(Z, q)]
            nlev = levels["counts"][(Z, q)]
            for t in range(tab["n_trans"]):
                a, b = tab["lo"][t], tab["hi"][t]
                k = (e["entry"], Z, q, t)
                if a >= nlev or b >= nlev or a not in u or b not in u:
                    tab_out[k] = {"skip": "LEVEL_NOT_IN_SOLVE_MAP"}
                    continue
                if u[a] == u[b] and fault != "same_unit_counted":
                    tab_out[k] = {"skip": "SAME_SOLVE_UNIT"}
                    continue
                om, _ = det_coll_omega_offline_interp(tab, t, Te, fault)
                if not om > 0.0:
                    tab_out[k] = {"skip": "OMEGA_NOT_POSITIVE"}
                    continue
                ga, gb = rows[L[(Z, q, a)]][4], rows[L[(Z, q, b)]][4]
                dE = (rows[L[(Z, q, b)]][3] - rows[L[(Z, q, a)]][3]) * EV_TO_ERG
                if not dE > 0.0:
                    tab_out[k] = {"skip": "DE_NOT_POSITIVE"}
                    continue
                inv = 1.0 / math.sqrt(Te)
                cu = ne * COL_CONST * inv / ga * om * math.exp(-dE / (K_B * Te))
                cd = ne * COL_CONST * inv / gb * om
                if fault == "common_factor":
                    cu, cd = 1.7 * cu, 1.7 * cd
                tab_out[k] = {"up": cu, "down": cd, "omega": om, "lo": a, "hi": b}
    return line_out, tab_out


def det_coll_omega_offline_run_cref(cref, model_dir, levels, lines, units_list, entries, omega_on, coll_floor, work):
    """run the CPU reference in a separate process with an explicit environment (no inherited LUMINA_* state)."""
    work = Path(work)
    inp = work / f"cref_input_{'omega' if omega_on else 'fix'}.txt"
    outp = work / f"cref_output_{'omega' if omega_on else 'fix'}.txt"
    with open(inp, "w") as f:
        f.write(f"MODEL {model_dir} COLL_FLOOR {coll_floor!r}\nNLEV {len(levels['rows'])}\n")
        for z, q, n, E, g in levels["rows"]:
            f.write(f"{z} {q} {n} {E!r} {g}\n")
        f.write(f"NTARGET {len(units_list)}\n")
        for (Z, q), mapping, nunit in units_list:
            f.write(f"{Z} {q} {len(mapping)} {nunit}\n")
            for gl, u in mapping:
                f.write(f"{gl} {u}\n")
        f.write(f"NLINE {len(lines)}\n")
        for ln in lines:
            lg = levels["idx"].get((ln["Z"], ln["ion"], ln["lo"]), -1)
            ug = levels["idx"].get((ln["Z"], ln["ion"], ln["up"]), -1)
            f.write(f"{ln['Z']} {ln['ion']} {ln['lo']} {ln['up']} {ln['f_lu']!r} {ln['A_ul']!r} {lg} {ug}\n")
        f.write(f"NENTRY {len(entries)}\n")
        for e in entries:
            f.write(f"{e['entry']} {e['shell']} {e['T_e']!r} {e['n_e']!r}\n")
    env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "LUMINA_NLTE_COLL_FIX": "1",
           "LUMINA_OMEGA_CMFGEN": "1" if omega_on else "0"}
    t0 = time.time()
    p = subprocess.run([str(cref), str(inp), str(outp)], capture_output=True, text=True, env=env)
    res = {"rc": p.returncode, "elapsed_s": time.time() - t0, "stdout_tail": p.stdout[-2000:],
           "stderr_tail": p.stderr[-2000:], "lines": {}, "table": {}, "header": None, "loaded": None,
           "input_sha256": POP.sha(inp)}
    if not outp.exists():
        return res
    for ln in outp.read_text().split("\n"):
        f = ln.split()
        if not f:
            continue
        if f[0] == "C":
            res["header"] = ln
        elif f[0] == "M":
            res["loaded"] = int(f[1])
        elif f[0] == "L":
            res["lines"][(int(f[1]), int(f[2]))] = (float(f[3]), float(f[4]), int(f[5]), float(f[6]))
        elif f[0] == "T":
            res["table"][(int(f[1]), int(f[2]), int(f[3]), int(f[6]))] = (float(f[7]), float(f[8]), int(f[4]),
                                                                          int(f[5]))
    res["output_sha256"] = POP.sha(outp)
    return res


def det_coll_omega_offline_compare(line_py, tab_py, fix_c, om_c):
    """Python vs sealed C: tiers, exact zeros and positive rates to 1e-12."""
    bad = []
    for k, v in line_py.items():
        if "missing" in v:
            continue
        cf, co = fix_c["lines"].get(k), om_c["lines"].get(k)
        if cf is None or co is None:
            bad.append(("LINE_ABSENT_IN_C", k))
            continue
        for a, b, nm in ((v["fix_up"], cf[0], "fix_up"), (v["fix_down"], cf[1], "fix_down"),
                         (v["om_up"], co[0], "om_up"), (v["om_down"], co[1], "om_down")):
            if (a == 0.0) != (b == 0.0) or det_coll_omega_offline_rel(a, b) > CRIT_REF:
                bad.append((f"LINE_{nm}", k))
        if v["tier"] != co[2]:
            bad.append(("LINE_TIER", k))
        elif det_coll_omega_offline_rel(v["ups"], co[3]) > CRIT_REF:
            bad.append(("LINE_UPS", k))
    cpos = {k: v for k, v in tab_py.items() if "up" in v}
    if set(cpos) != set(om_c["table"]):
        for k in set(cpos) ^ set(om_c["table"]):
            bad.append(("TABLE_SET", k))
    for k in set(cpos) & set(om_c["table"]):
        a, c = cpos[k], om_c["table"][k]
        if det_coll_omega_offline_rel(a["up"], c[0]) > CRIT_REF or det_coll_omega_offline_rel(a["down"], c[1]) > CRIT_REF:
            bad.append(("TABLE_RATE", k))
    return bad


# ------------------------------------------------------------------ pairs, classification, decision
def det_coll_omega_offline_db(cu, cd, g_lo, g_up, dE, T_e):
    if not (cu > 0.0 and cd > 0.0):
        return None
    return (cu / cd) / ((g_up / g_lo) * math.exp(-dE / (K_B * T_e))) - 1.0


def det_coll_omega_offline_pairs(sample, lines, tables, units, levels, entries, line_py, tab_py, fault=None):
    """sample pairs and one-step connected pairs per shell (denominators fixed before any rate is read)."""
    L, rows = levels["idx"], levels["rows"]
    by_pair = {}
    for li, ln in enumerate(lines):
        by_pair.setdefault((ln["Z"], min(ln["lo"], ln["up"]), max(ln["lo"], ln["up"])), []).append(li)
    line_of = {ln["line_id"]: li for li, ln in enumerate(lines)}
    tab_pairs = {}
    for (Z, q), tab in tables.items():
        if q != TARGET["ion"]:
            continue
        for t, (a, b) in enumerate(zip(tab["lo"], tab["hi"])):
            tab_pairs.setdefault((Z, min(a, b), max(a, b)), []).append(t)
    plan = []
    for Z, line, s in sample["keys"]:
        li = line_of.get(line)
        if li is None:
            plan.append({"set": "sample", "Z": Z, "line": line, "shell": s, "missing": "LINE_NOT_IN_DECK"})
            continue
        ln = lines[li]
        plan.append({"set": "sample", "Z": Z, "line": line, "shell": s,
                     "pair": (Z, min(ln["lo"], ln["up"]), max(ln["lo"], ln["up"]))})
    for s in sorted({k[2] for k in sample["keys"]}):
        sp = {p["pair"] for p in plan if p["set"] == "sample" and p["shell"] == s and "pair" in p}
        levs = {(p[0], p[1]) for p in sp} | {(p[0], p[2]) for p in sp}
        conn = set()
        for pr in list(by_pair) + list(tab_pairs):
            if ((pr[0], pr[1]) in levs or (pr[0], pr[2]) in levs) and pr not in sp:
                conn.add(pr)
        for pr in sorted(conn):
            u = units.get((pr[0], TARGET["ion"]), {})
            same = pr[1] in u and pr[2] in u and u[pr[1]] == u[pr[2]]
            plan.append({"set": "connected_same_unit" if same else "connected", "Z": pr[0], "line": None, "shell": s,
                         "pair": pr, "line_less": pr not in by_pair})
    if fault == "drop_connected_pair":
        plan = [p for p in plan if p["set"] != "connected"][:] + [p for p in plan if p["set"] == "connected"][1:]
    ent = {e["shell"]: e for e in entries if e["gen"] == MAIN_GEN}
    out = []
    for p in plan:
        rec = {k: v for k, v in p.items() if k != "pair"}
        if "pair" not in p:
            rec["status"] = "NOT_IDENTIFIABLE"
            out.append(rec)
            continue
        Z, a, b = p["pair"]
        rec.update({"lo": a, "hi": b})
        e = ent.get(p["shell"])
        la, lb = L.get((Z, TARGET["ion"], a)), L.get((Z, TARGET["ion"], b))
        if e is None or la is None or lb is None:
            rec["status"], rec["missing"] = "NOT_IDENTIFIABLE", "STATE_OR_LEVEL_ABSENT"
            out.append(rec)
            continue
        g_lo, g_up = rows[la][4], rows[lb][4]
        dE = abs(rows[lb][3] - rows[la][3]) * EV_TO_ERG
        fix_u = fix_d = om_line_u = om_line_d = om_tab_u = om_tab_d = 0.0
        tiers, lns = [], by_pair.get((Z, a, b), [])
        db_bad = []
        for li in lns:
            v = line_py.get((e["entry"], li), {})
            if "missing" in v or not v:
                rec["missing"] = "LINE_RATE_ABSENT"
                continue
            fix_u += v["fix_up"]
            fix_d += v["fix_down"]
            om_line_u += v["om_up"]
            om_line_d += v["om_down"]
            tiers.append(v["tier"])
            for cu, cd in ((v["fix_up"], v["fix_down"]), (v["om_up"], v["om_down"])):
                eps = det_coll_omega_offline_db(cu, cd, g_lo, g_up, dE, e["T_e"])
                if eps is not None and abs(eps) > CRIT_DB:
                    db_bad.append(eps)
        n_tab = 0
        for t in tab_pairs.get((Z, a, b), []):
            v = tab_py.get((e["entry"], Z, TARGET["ion"], t), {})
            if "up" in v:
                om_tab_u += v["up"]
                om_tab_d += v["down"]
                n_tab += 1
                eps = det_coll_omega_offline_db(v["up"], v["down"], g_lo, g_up, dE, e["T_e"])
                if eps is not None and abs(eps) > CRIT_DB:
                    db_bad.append(eps)
        om_u, om_d = om_line_u + om_tab_u, om_line_d + om_tab_d
        if fault == "denominator_floor":
            om_d = om_d if om_d > 0 else 1e-300
        for cu, cd in ((fix_u, fix_d), (om_u, om_d)):
            eps = det_coll_omega_offline_db(cu, cd, g_lo, g_up, dE, e["T_e"])
            if eps is not None and abs(eps) > CRIT_DB:
                db_bad.append(eps)
        comp = ("TABLE" if n_tab and not any(x in (2, 3) for x in tiers) else
                "VR" if 2 in tiers and not n_tab and 3 not in tiers else
                "SET" if 3 in tiers and not n_tab and 2 not in tiers else
                "NONE" if not n_tab and not tiers else "MIXED")
        rec.update({"n_lines": len(lns), "line_ids": ";".join(str(lines[i]["line_id"]) for i in lns),
                    "tiers": ";".join(str(x) for x in tiers), "n_table_records": n_tab, "omega_composition": comp,
                    "C_fix_up": fix_u, "C_fix_down": fix_d, "C_omega_line_up": om_line_u,
                    "C_omega_line_down": om_line_d, "C_omega_table_up": om_tab_u, "C_omega_table_down": om_tab_d,
                    "C_omega_up": om_u, "C_omega_down": om_d, "T_e": e["T_e"], "n_e": e["n_e"],
                    "detailed_balance": "FAIL" if db_bad else "OK", "db_worst": max((abs(x) for x in db_bad), default=0.0)})
        rec.update(det_coll_omega_offline_classify(fix_d, om_d, fix_u, om_u))
        if db_bad:
            rec["status"] = "X"
        out.append(rec)
    return out


def det_coll_omega_offline_classify(fix_d, om_d, fix_u, om_u):
    """H-SMALL on one pair: 1/2 <= R_ul <= 2 (prereg 5.3), boundary band 1e-10 in log10."""
    if fix_d == 0.0 and om_d == 0.0:
        return {"R_ul": None, "R_lu": None, "d": None, "status": "SURVIVES", "tag": "NO_CHANGE"}
    if fix_d == 0.0 or om_d == 0.0:
        return {"R_ul": None, "R_lu": None, "d": None, "status": "REFUTED", "tag": "ONE_SIDE_ZERO"}
    r = fix_d / om_d
    rl = (fix_u / om_u) if (fix_u > 0 and om_u > 0) else None
    d = math.log10(r)
    lim = math.log10(TWO)
    if abs(abs(d) - lim) <= BOUND_EPS:
        st = "UNRESOLVED"
    else:
        st = "SURVIVES" if abs(d) < lim else "REFUTED"
    return {"R_ul": r, "R_lu": rl, "d": d, "status": st, "tag": "TWO_SIDED"}


def det_coll_omega_offline_universal(states):
    s = set(states)
    if not s:
        return "NOT_IDENTIFIABLE"
    for st in ("X", "REFUTED", "NOT_IDENTIFIABLE", "UNRESOLVED"):
        if st in s:
            return st
    return "SURVIVES"


def det_coll_omega_offline_decide(rows, fault=None):
    """per-ion fractions and the registered ordering decision (prereg 5.3)."""
    ions = {}
    for set_name in ("sample", "connected"):
        for Z in TARGET["Z"]:
            sub = [r for r in rows if r["set"] == set_name and r["Z"] == Z]
            B = sum(1 for r in sub if r["status"] == "REFUTED")
            U = sum(1 for r in sub if r["status"] in ("UNRESOLVED", "NOT_IDENTIFIABLE"))
            X = sum(1 for r in sub if r["status"] == "X")
            N = len(sub)
            ions[f"{set_name}:{Z}"] = {"N": N, "B": B, "U": U, "X": X, "f_lo": (B / N) if N else None,
                                       "f_hi": ((B + U) / N) if N else None,
                                       "code": det_coll_omega_offline_universal([r["status"] for r in sub])}
    thr = 1.01 if fault == "threshold_shift" else PRIORITY      # injected: ordering threshold moved
    main = [ions[f"sample:{Z}"] for Z in TARGET["Z"]]
    conn = [ions[f"connected:{Z}"] for Z in TARGET["Z"]]
    if any(v["X"] for v in main + conn):
        dec = "X"
    elif any(v["f_lo"] is not None and v["f_lo"] >= thr for v in main):
        dec = "NEXT_CANDIDATE"
    elif all(v["f_hi"] is not None and v["f_hi"] < thr for v in main) and \
            any(v["f_lo"] is not None and v["f_lo"] >= thr for v in conn):
        dec = "CONNECTED_ONLY"
    elif all(v["f_hi"] is not None and v["f_hi"] < thr for v in main + conn):
        dec = "NOT_PRIORITIZED"
    else:
        dec = "UNDECIDED"
    return ions, dec


def det_coll_omega_offline_decide_check(rows):
    """independent re-derivation from the stored pair rows (validator)."""
    out = {}
    for set_name in ("sample", "connected"):
        for Z in TARGET["Z"]:
            st = [r["status"] for r in rows if r["set"] == set_name and r["Z"] == Z]
            out[f"{set_name}:{Z}"] = (len(st), st.count("REFUTED"), st.count("UNRESOLVED") + st.count("NOT_IDENTIFIABLE"),
                                      st.count("X"))
    m = [out[f"sample:{Z}"] for Z in TARGET["Z"]]
    c = [out[f"connected:{Z}"] for Z in TARGET["Z"]]
    if any(v[3] for v in m + c):
        return out, "X"
    f = lambda v, k: (v[1] + (v[2] if k else 0)) / v[0] if v[0] else None             # noqa: E731
    if any(f(v, 0) is not None and f(v, 0) >= PRIORITY for v in m):
        return out, "NEXT_CANDIDATE"
    if all(f(v, 1) is not None and f(v, 1) < PRIORITY for v in m) and any(f(v, 0) is not None and f(v, 0) >= PRIORITY for v in c):
        return out, "CONNECTED_ONLY"
    if all(f(v, 1) is not None and f(v, 1) < PRIORITY for v in m + c):
        return out, "NOT_PRIORITIZED"
    return out, "UNDECIDED"


def det_coll_omega_offline_quantiles(vals):
    """nearest-rank quantiles of the sorted d values (prereg 5.1)."""
    v = sorted(x for x in vals if x is not None and math.isfinite(x))
    if not v:
        return None
    nr = lambda p: v[max(0, math.ceil(p * len(v)) - 1)]                              # noqa: E731
    return {"n": len(v), "min": v[0], "p10": nr(0.10), "p50": nr(0.50), "p90": nr(0.90), "max": v[-1],
            "beyond_two": sum(1 for x in v if abs(x) > math.log10(TWO))}


# ------------------------------------------------------------------ analyze
def det_coll_omega_offline_analyze(census, prior_sample, prior_result, r_run, src, out, target=TARGET, fault=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    r_run = Path(r_run)
    M = r_run / "input/model"
    why = []
    # 1 fixed sample (prereg 2.1) and its agreement with the prior stage
    sample = SEED.det_seedmem_scope_sample(census, target, fault)
    why += sample["why"]
    ps = json.loads(Path(prior_sample).read_text()) if prior_sample and Path(prior_sample).exists() else None
    if ps is None or [tuple(k) for k in ps.get("keys", [])] != [tuple(k) for k in sample["keys"]]:
        why.append("SAMPLE_NOT_PRIOR_SAMPLE")
    pr = json.loads(Path(prior_result).read_text()) if prior_result and Path(prior_result).exists() else {}
    prior_branch = ((pr.get("collision_authority") or {}).get("R") or {}).get("branch")
    # 2 run identity, knobs, collision branch of the stored run
    ex = RATES.det_siiii_rates_exports(r_run)
    knobs = {k: ex.get(k) for k in ("LUMINA_NLTE_COLL_FIX", "LUMINA_NLTE_COLL_FLOOR", "LUMINA_OMEGA_CMFGEN",
                                    "LUMINA_ARTIS_PARITY", "LUMINA_NLTE_ASSEMBLE_GPU", "LUMINA_NLTE_METASTABLE_COLL",
                                    "LUMINA_SUPER_CUTOFF", "LUMINA_SUPER_LEVELS", "LUMINA_OMEGA_CMFGEN_SET")}
    on = lambda k: SEED.det_seedmem_scope_atoi(ex.get(k)) != 0                          # noqa: E731
    if not on("LUMINA_NLTE_COLL_FIX") or on("LUMINA_OMEGA_CMFGEN") or on("LUMINA_ARTIS_PARITY"):
        why.append("STORED_BRANCH_NOT_COLL_FIX")
    if on("LUMINA_NLTE_ASSEMBLE_GPU") or on("LUMINA_NLTE_METASTABLE_COLL") or ex.get("LUMINA_OMEGA_CMFGEN_SET"):
        why.append("STORED_CONFIG_NOT_REGISTERED")
    try:
        coll_floor = float(ex.get("LUMINA_NLTE_COLL_FLOOR", "0") or 0.0)
    except ValueError:
        coll_floor = math.nan
    if coll_floor != 0.0:
        why.append("COLL_FLOOR_NOT_ZERO")
    if prior_branch != "COLL_FIX_PER_LINE":
        why.append("PRIOR_COLLISION_AUTHORITY_NOT_COLL_FIX")
    # 3 deck, tables, solve units, stored states
    levels = det_coll_omega_offline_levels(M)
    lines_all, lwhy = det_coll_omega_offline_lines(M, target["Z"], target["ion"])
    why += lwhy
    lines = [l for l in lines_all if (l["Z"], l["ion"], l["lo"]) in levels["idx"] and (l["Z"], l["ion"], l["up"]) in levels["idx"]]
    lines_dropped = [l["line_id"] for l in lines_all if l not in lines]
    tables, twhy, listing = det_coll_omega_offline_manifest(M, levels["counts"], fault)
    why += twhy
    for Z in target["Z"]:
        if (Z, target["ion"]) not in tables:
            why.append(f"TARGET_TABLE_ABSENT:{Z}")
    deck = SEED.det_seedmem_scope_deck(M, target["Z"], target["ion"])
    so = (r_run / "stdout.log").read_text(errors="replace") if (r_run / "stdout.log").exists() else ""
    sl = SEED.det_seedmem_scope_sl(deck, ex, so, target["ion"])
    why += sl["why"]
    units = {(Z, target["ion"]): {l: u for (z, l), u in sl["unit"].items() if z == Z} for Z in target["Z"]}
    gen_main = MAIN_GEN + 1 if fault == "gen8" else MAIN_GEN
    dmain = det_coll_omega_offline_dump(r_run, gen_main)
    dctr = det_coll_omega_offline_dump(r_run, COUNTER_GEN)
    why += dmain["why"] + dctr["why"]
    if dmain.get("population_generation") != MAIN_GEN:
        why.append("MAIN_STATE_NOT_GENERATION_7")
    entries = []
    if "ne" in dmain:
        for s in target["shells"]:
            entries.append({"entry": len(entries), "shell": s, "gen": dmain["population_generation"],
                            "T_e": float(dmain["Te"][s]), "n_e": float(dmain["ne"][s])})
    if "ne" in dctr:
        cs = target["counter"][1]
        entries.append({"entry": len(entries), "shell": cs, "gen": COUNTER_GEN, "T_e": float(dctr["Te"][cs]),
                        "n_e": float(dctr["ne"][cs])})
    timing = {"inputs_s": time.time() - t0}
    base = {"schema": "DET_COLL_OMEGA_OFFLINE_RESULT_V1", "fault": fault, "knobs": knobs, "loader": listing,
            "lines_without_levels": lines_dropped}
    if why:
        res = base | {"validity": "X", "reasons": sorted(set(why))[:200]}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        return res
    # 4 Python evaluation and the sealed-C reference
    line_py, tab_py = det_coll_omega_offline_python(levels, lines, tables, units, entries, fault)
    cref = det_coll_omega_offline_build_cref(src, out / "cref", fault)
    units_list = []
    for Z in target["Z"]:
        u = units[(Z, target["ion"])]
        mapping = sorted((levels["idx"][(Z, target["ion"], l)], uu) for l, uu in u.items()
                         if (Z, target["ion"], l) in levels["idx"])
        units_list.append(((Z, target["ion"]), mapping, len(set(u.values()))))
    cfix = comg = None
    cmp_bad = []
    if not cref["why"]:
        cfix = det_coll_omega_offline_run_cref(cref["binary"], M, levels, lines, units_list, entries, False, coll_floor,
                                               out / "cref")
        comg = det_coll_omega_offline_run_cref(cref["binary"], M, levels, lines, units_list, entries, True, coll_floor,
                                               out / "cref")
        if cfix["rc"] != 0 or comg["rc"] != 0:
            cref["why"].append(f"CREF_RUN_RC:{cfix['rc']}:{comg['rc']}")
        else:
            cmp_bad = det_coll_omega_offline_compare(line_py, tab_py, cfix, comg)
    # 5 pair assembly, classification, decision
    rows = det_coll_omega_offline_pairs(sample, lines, tables, units, levels, entries, line_py, tab_py, fault)
    if cmp_bad:
        for r in rows:
            r["status"] = "X" if r["status"] != "NOT_IDENTIFIABLE" else r["status"]
    if fault == "drop_failed_rows":
        rows = [r for r in rows if r["status"] != "REFUTED"]
    if fault == "fill_missing_zero":
        for r in rows:
            if r["status"] == "NOT_IDENTIFIABLE":
                r["status"] = "SURVIVES"
    ions, decision = det_coll_omega_offline_decide(rows, fault)
    vwhy = list(cref["why"]) + [f"CREF_MISMATCH:{a}" for a, _ in cmp_bad[:50]]
    if cmp_bad:
        vwhy.append(f"CREF_MISMATCH_TOTAL:{len(cmp_bad)}")
    # validators: fixed denominators, recomputation of each stored state and of the decision
    n_sample = sum(1 for r in rows if r["set"] == "sample")
    if n_sample != len(sample["keys"]):
        vwhy.append(f"SAMPLE_ROW_ACCOUNTING:{n_sample}/{len(sample['keys'])}")
    for r in rows:
        if r.get("missing") and r["status"] != "NOT_IDENTIFIABLE":
            vwhy.append(f"MISSING_FILLED:{r['set']}:{r['Z']}:{r.get('line')}:{r['shell']}")
        if r.get("C_omega_down") is not None:
            for a, b, c in (("C_omega_down", "C_omega_line_down", "C_omega_table_down"),
                            ("C_omega_up", "C_omega_line_up", "C_omega_table_up")):
                if r[a] != r[b] + r[c]:
                    vwhy.append(f"PAIR_SUM_RECOMPUTE:{r['set']}:{r['Z']}:{r.get('lo')}:{r.get('hi')}:{r['shell']}")
        if r["status"] in ("X", "NOT_IDENTIFIABLE") or r.get("C_fix_down") is None:
            continue
        exp = det_coll_omega_offline_classify(r["C_fix_down"], r["C_omega_down"], r["C_fix_up"], r["C_omega_up"])["status"]
        if exp != r["status"]:
            vwhy.append(f"STATUS_RECOMPUTE:{r['set']}:{r['Z']}:{r.get('lo')}:{r.get('hi')}:{r['shell']}")
    plan_check = det_coll_omega_offline_plan_count(sample, lines, tables, units)
    got = {s: sum(1 for r in rows if r["set"] == "connected" and r["shell"] == s) for s in plan_check}
    if got != plan_check:
        vwhy.append("CONNECTED_DENOMINATOR")
    chk, dec2 = det_coll_omega_offline_decide_check(rows)
    if dec2 != decision:
        vwhy.append(f"DECISION_RECOMPUTE:{decision}!={dec2}")
    if any(r["status"] == "X" for r in rows):
        vwhy.append(f"X_ROWS:{sum(1 for r in rows if r['status'] == 'X')}")
    validity = "X" if vwhy else "VALID"
    if validity == "X":
        decision = "X"
    timing["compute_s"] = time.time() - t0 - timing["inputs_s"]
    # 6 outputs
    rate_rows = []
    for (ent, li), v in sorted(line_py.items()):
        e, ln = entries[ent], lines[li]
        rr = {"kind": "LINE", "entry": ent, "shell": e["shell"], "gen": e["gen"], "line_id": ln["line_id"], "Z": ln["Z"],
              "lo": ln["lo"], "up": ln["up"], "f_lu": ln["f_lu"]} | {k: v.get(k) for k in
                                                                       ("fix_up", "fix_down", "om_up", "om_down", "tier",
                                                                        "ups", "table_t", "end_clamped", "missing")}
        if cfix and comg:
            cf, co = cfix["lines"].get((ent, li)), comg["lines"].get((ent, li))
            if cf and co:
                rr.update({"c_fix_up": cf[0], "c_fix_down": cf[1], "c_om_up": co[0], "c_om_down": co[1], "c_tier": co[2],
                           "c_ups": co[3]})
        rate_rows.append(rr)
    for (ent, Z, q, t), v in sorted(tab_py.items()):
        e = entries[ent]
        rr = {"kind": "TABLE", "entry": ent, "shell": e["shell"], "gen": e["gen"], "Z": Z, "t": t,
              "lo": tables[(Z, q)]["lo"][t], "up": tables[(Z, q)]["hi"][t], "om_up": v.get("up"),
              "om_down": v.get("down"), "ups": v.get("omega"), "skip": v.get("skip")}
        if comg and (ent, Z, q, t) in comg["table"]:
            c = comg["table"][(ent, Z, q, t)]
            rr.update({"c_om_up": c[0], "c_om_down": c[1]})
        rate_rows.append(rr)
    for r in rows:
        rate_rows.append({"kind": "PAIR"} | r)
    pd.DataFrame(rate_rows).to_csv(out / "rates.csv", index=False)
    unc = [{"entry": r["entry"], "shell": r["shell"], "gen": r["gen"], "line_id": r["line_id"], "Z": r["Z"],
            "lo": r["lo"], "up": r["up"], "tier": r["tier"], "ups": r["ups"],
            "reason": ("NOT_IN_TABLE" if r["table_t"] is None else "TABLE_UPSILON_NOT_POSITIVE")}
           for r in rate_rows if r["kind"] == "LINE" and r.get("tier") in (2, 3)]
    unc += [{"entry": r["entry"], "shell": r["shell"], "gen": r["gen"], "Z": r["Z"], "lo": r["lo"], "up": r["up"],
             "table_t": r["t"], "reason": f"TABLE_NOT_APPLIED:{r['skip']}"} for r in rate_rows
            if r["kind"] == "TABLE" and r.get("skip")]
    pd.DataFrame(unc).to_csv(out / "uncovered.csv", index=False)
    smp = {"schema": "DET_COLL_OMEGA_OFFLINE_SAMPLE_V1", "authority": str(census), "authority_sha256": POP.sha(census),
           "prior_sample_sha256": POP.sha(prior_sample) if ps else None, "n_keys": len(sample["keys"]),
           "keys": sample["keys"], "counts": sample["counts"], "entries": entries,
           "connected_denominators": plan_check, "solve_units": {k: v for k, v in sl.items() if k not in ("unit", "size")},
           "states": {"main": {k: dmain.get(k) for k in ("manifest", "population_generation", "radiation_generation",
                                                         "te_generation")},
                      "counter": {k: dctr.get(k) for k in ("manifest", "population_generation", "radiation_generation",
                                                           "te_generation")}}}
    (out / "sample.json").write_text(json.dumps(smp, indent=1, default=str))
    dist = {}
    for set_name in ("sample", "connected", "connected_same_unit"):
        for Z in target["Z"]:
            sub = [r for r in rows if r["set"] == set_name and r["Z"] == Z]
            dist[f"{set_name}:{Z}"] = {
                "N": len(sub), "by_status": pd.Series([r["status"] for r in sub]).value_counts().to_dict() if sub else {},
                "by_composition": pd.Series([r.get("omega_composition") for r in sub]).value_counts().to_dict()
                if sub else {},
                "d_quantiles": det_coll_omega_offline_quantiles([r.get("d") for r in sub]),
                "one_side_zero": sum(1 for r in sub if r.get("tag") == "ONE_SIDE_ZERO"),
                "no_change": sum(1 for r in sub if r.get("tag") == "NO_CHANGE"),
                "unique_pairs": len({(r.get("lo"), r.get("hi"), r["shell"]) for r in sub}),
                "db_fail": sum(1 for r in sub if r.get("detailed_balance") == "FAIL")}
    named = {}
    for lid in (target["control"], target["counter"][0]):
        named[str(lid)] = [{k: r.get(k) for k in ("shell", "lo", "hi", "C_fix_down", "C_omega_down", "R_ul", "d", "status",
                                                  "omega_composition", "tiers")}
                           for r in rows if r["set"] == "sample" and r["line"] == lid]
    ctr_entry = next((e for e in entries if e["gen"] == COUNTER_GEN), None)
    counter_g3 = None
    if ctr_entry is not None:
        lid = target["counter"][0]
        li = next((i for i, l in enumerate(lines) if l["line_id"] == lid), None)
        if li is not None:
            v = line_py.get((ctr_entry["entry"], li), {})
            t_add = [(k, w) for k, w in tab_py.items() if k[0] == ctr_entry["entry"] and "up" in w and
                     (w["lo"], w["hi"]) in ((lines[li]["lo"], lines[li]["up"]), (lines[li]["up"], lines[li]["lo"]))]
            counter_g3 = {"generation": COUNTER_GEN, "T_e": ctr_entry["T_e"], "n_e": ctr_entry["n_e"],
                          "C_fix_down": v.get("fix_down"), "C_omega_line_down": v.get("om_down"), "tier": v.get("tier"),
                          "C_omega_table_down": sum(w["down"] for _, w in t_add),
                          "note": "comparison only: outside every denominator; does not lift or extend the prior "
                                  "two-level exclusion (prereg 1.3, 2.3)"}
    res = base | {"validity": validity, "reasons": vwhy[:200], "n_reasons": len(vwhy),
                  "H_SMALL": {k: v["code"] for k, v in ions.items()}, "ions": ions, "decision": decision,
                  "distributions": dist, "named_lines": named, "counter_generation3": counter_g3,
                  "cref": {"build": {k: v for k, v in cref.items() if k != "excerpts"}, "excerpts": cref["excerpts"],
                           "fix_run": None if cfix is None else {k: v for k, v in cfix.items() if k not in ("lines", "table")},
                           "omega_run": None if comg is None else {k: v for k, v in comg.items()
                                                                   if k not in ("lines", "table")},
                           "mismatches": len(cmp_bad)},
                  "timing_s": timing,
                  "claim_ceiling": "stored-state collision rates only; no population / ionisation / spectral effect, no "
                                   "physical correctness, no production promotion, no run authorisation"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    nxt = det_coll_omega_offline_next(rows, ions, decision, rate_rows, validity)
    (out / "next_contract.json").write_text(json.dumps(nxt, indent=1, default=str))
    return res


def det_coll_omega_offline_plan_count(sample, lines, tables, units):
    """connected denominators per shell, recomputed from the deck and the tables alone (validator)."""
    lp = {(l["Z"], min(l["lo"], l["up"]), max(l["lo"], l["up"])) for l in lines}
    tp = {(Z, min(a, b), max(a, b)) for (Z, q), t in tables.items() if q == TARGET["ion"] for a, b in zip(t["lo"], t["hi"])}
    line_of = {l["line_id"]: (l["Z"], min(l["lo"], l["up"]), max(l["lo"], l["up"])) for l in lines}
    out = {}
    for s in sorted({k[2] for k in sample["keys"]}):
        sp = {line_of[k[1]] for k in sample["keys"] if k[2] == s and k[1] in line_of}
        levs = {(p[0], p[1]) for p in sp} | {(p[0], p[2]) for p in sp}
        n = 0
        for pr in lp | tp:
            if pr in sp or not ((pr[0], pr[1]) in levs or (pr[0], pr[2]) in levs):
                continue
            u = units.get((pr[0], TARGET["ion"]), {})
            if pr[1] in u and pr[2] in u and u[pr[1]] == u[pr[2]]:
                continue
            n += 1
        out[s] = n
    return out


def det_coll_omega_offline_next(rows, ions, decision, rate_rows, validity):
    """prereg 5.4: the ordering decision with the evidence it consumes; X yields no decision."""
    ref = [r for r in rows if r["set"] == "sample" and r["status"] == "REFUTED"]
    comp = pd.Series([r.get("omega_composition") for r in ref]).value_counts().to_dict() if ref else {}
    lineless = [r for r in rows if r["set"] == "connected" and r.get("line_less")]
    dup = [r for r in rows if r.get("n_lines", 0) > 1 and r.get("n_table_records", 0) >= 1]
    unc = pd.Series([f"tier{r['tier']}" for r in rate_rows if r["kind"] == "LINE" and r.get("tier") in (2, 3)]
                    ).value_counts().to_dict()
    scope = ("collision prescription change: table + uncovered approximation + added connections + duplicate "
             "suppression" if comp.get("VR", 0) + comp.get("SET", 0) + comp.get("MIXED", 0) > comp.get("TABLE", 0)
             else "collision prescription change dominated by tabulated pairs (still includes the uncovered "
                  "approximation and the col_data additions)")
    text = {"NEXT_CANDIDATE": "after the NRB stationary-state check, propose the collision prescription change as the "
                              "preferred next physics rung",
            "CONNECTED_ONLY": "connected pathways change strongly but the target-transition evidence is weak: propose a "
                              "follow-up design that names the connected transitions first",
            "NOT_PRIORITIZED": "the rate evidence alone does not put the collision rung next",
            "UNDECIDED": "ordering undecided: name the missing / unresolved evidence before ordering",
            "X": "INADMISSIBLE: no ordering decision (validation failed)"}[decision]
    return {"schema": "DET_COLL_OMEGA_OFFLINE_NEXT_V1", "validity": validity, "decision": decision, "next": text,
            "ions": ions,
            "evidence": {"refuted_sample_keys_by_omega_composition": comp,
                         "line_less_connected_table_pairs": len(lineless),
                         "pairs_with_duplicate_lines_and_table": len(dup),
                         "uncovered_line_evaluations_by_tier": unc},
            "candidate_scope": scope if decision in ("NEXT_CANDIDATE", "CONNECTED_ONLY") else None,
            "lumina_vs_cmfgen_residuals": ["table interpolation linear in T (CMFGEN log-log)",
                                           "no SAME_N g-bar=0.7 (levels.csv carries no configuration label)",
                                           "uncovered approximation reaches only pairs with a deck line",
                                           "global OMEGA switch also loads other ions' tables and the RADEQ "
                                           "preparation (10400 K), not evaluated here"],
            "not_authorised": "no run, production change or collision-data activation follows from this table"}


# ------------------------------------------------------------------ fixtures (direct tests, prereg 8.2)
FX_SHELLS = (10, 15)
FX_NS = 16
FX_CUTOFF = 6
FX_LEVELS = (  # (Z, ion, level, E_eV, g)
    (14, 1, 0, 0.0, 2), (14, 1, 1, 5.0, 4), (14, 1, 2, 8.0, 2),
    (14, 2, 0, 0.0, 1), (14, 2, 1, 6.5, 3), (14, 2, 2, 6.6, 5), (14, 2, 3, 10.3, 3), (14, 2, 4, 12.0, 5),
    (14, 2, 5, 12.2, 7), (14, 2, 6, 15.0, 3), (14, 2, 7, 15.1, 5),
    (16, 2, 0, 0.0, 1), (16, 2, 1, 8.0, 3), (16, 2, 2, 11.0, 5), (16, 2, 3, 13.0, 3), (16, 2, 4, 14.0, 7),
    (16, 2, 5, 15.5, 5))
FX_LINES = (  # (Z, ion, lo, up, f_lu): row = line_id
    (14, 1, 0, 1, 0.4),
    (14, 2, 0, 1, 0.5), (14, 2, 0, 2, 0.3), (14, 2, 1, 3, 1.0e-5), (14, 2, 2, 4, 2.0e-5), (14, 2, 0, 5, 0.1),
    (14, 2, 6, 7, 0.2), (14, 2, 5, 6, 0.05), (14, 2, 3, 4, 0.2), (14, 2, 3, 4, 0.1), (14, 2, 3, 9, 0.1),
    (16, 2, 0, 1, 0.8), (16, 2, 1, 2, 0.05), (16, 2, 0, 3, 5.0e-6), (16, 2, 2, 4, 0.3))
FX_TABLES = {
    (14, 1): ([5000.0, 10000.0], [((0, 1), [1.1, 1.3])]),
    (14, 2): ([5000.0, 10000.0, 20000.0], [((0, 1), [1.0, 1.5, 2.0]), ((0, 5), [0.0, 0.0, 0.0]),
                                            ((6, 7), [0.5, 0.6, 0.7]), ((5, 6), [0.3, 0.35, 0.4]),
                                            ((3, 4), [2.0, 2.2, 2.4]), ((1, 2), [0.8, 0.9, 1.0]),
                                            ((2, 3), [0.2, 0.25, 0.3])]),
    (16, 2): ([5000.0, 8000.0, 10000.0, 20000.0], [((0, 1), [3.0, 3.2, 3.4, 3.6]), ((2, 5), [0.1, 0.12, 0.13, 0.15])]),
}
FX_SAMPLE = ((1, 10), (1, 15), (2, 10), (3, 10), (8, 10), (7, 15), (10, 15), (11, 10), (12, 15))
FX_TARGET = {"Z": (14, 16), "ion": 2, "beta_max": 1e-3, "counts": {14: 7, 16: 2}, "control": 1, "counter": (2, 10),
             "shells": FX_SHELLS, "forms": {}}
FX_STATE = {7: {10: (9000.0, 1.0e9), 15: (12000.0, 2.0e9)}, 3: {10: (25000.0, 1.5e9)}, 8: {10: (9100.0, 1.1e9),
                                                                                            15: (12100.0, 2.1e9)}}


def det_coll_omega_offline_write_bin(path, Z, ion0, nlev, tgrid, recs, truncate=False, nlev_header=None):
    b = struct.pack("<IIiiiii", 0x49474331, 1, Z, ion0, len(recs), len(tgrid), nlev if nlev_header is None else nlev_header)
    b += struct.pack(f"<{len(tgrid)}d", *tgrid)
    for (a, c), om in recs:
        b += struct.pack("<ii", a, c) + struct.pack(f"<{len(om)}d", *om)
    if truncate:
        b = b[:-5]
    Path(path).write_bytes(b)


def det_coll_omega_offline_fx(base, mutate=None):
    base = Path(base)
    run = base / "run"
    M = run / "input/model"
    M.mkdir(parents=True)
    (run / "work/physics_comparison").mkdir(parents=True)
    with open(M / "levels.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number,energy_eV,g,metastable,super_level,configuration\n")
        for Z, q, n, E, g in FX_LEVELS:
            f.write(f"{Z},{q},{n},{E!r},{g},0,{n},c{n}\n")
    E = {(Z, q, n): (En, g) for Z, q, n, En, g in FX_LEVELS}
    with open(M / "line_list.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number_lower,level_number_upper,line_id,wavelength,f_ul,f_lu,nu,B_lu,"
                "B_ul,A_ul,wavelength_cm\n")
        for i, (Z, q, lo, up, fl) in enumerate(FX_LINES):
            El, gl = E.get((Z, q, lo), (0.0, 1))
            Eu, gu = E.get((Z, q, up), (20.0, 1))
            nu = (Eu - El) * EV_TO_ERG / H_PLANCK
            lam = 2.99792458e10 / nu
            A = 0.6670 * (nu / 1e15) ** 2 * fl * gl / gu * 1e8
            f.write(f"{Z},{q},{lo},{up},{i},{lam * 1e8!r},{fl * gl / gu!r},{fl!r},{nu!r},1.0,1.0,{A!r},{lam!r}\n")
    cnt = {}
    for Z, q, *_ in FX_LEVELS:
        cnt[(Z, q)] = cnt.get((Z, q), 0) + 1
    man = ["ion,Z,ion0,osc,col,n_levels_osc,n_levels_ref,n_trans_source,n_temp,n_mapped,n_dropped,drop_reasons,"
           "omega_min,omega_max,omega_median,max_sumrule_err,cmfgen_quality_note,out_bin,status"]
    for (Z, q), (tg, recs) in FX_TABLES.items():
        fn = f"ige_col_{Z}_{q}_cmfgen.bin"
        det_coll_omega_offline_write_bin(M / fn, Z, q, cnt[(Z, q)], tg, recs)
        man.append(f"X,{Z},{q},o,c,{cnt[(Z, q)]},{cnt[(Z, q)]},{len(recs)},{len(tg)},{len(recs)},0,,0,1,0.5,0,,{fn},OK")
    man.append("Y,20,1,o,c,1,1,,,0,0,,,,,,,,SKIP: no table")
    (M / "coldata_cmfgen_manifest.csv").write_text("\n".join(man) + "\n")
    (M / "siii_col_cmfgen.csv").write_text("lower,upper,omega\n")
    (run / "input/resolved_lumina.exports").write_text(
        'declare -x LUMINA_NLTE_COLL_FIX="1"\ndeclare -x LUMINA_NLTE_COLL_FLOOR="0"\n'
        f'declare -x LUMINA_SUPER_LEVELS="1"\ndeclare -x LUMINA_SUPER_CUTOFF="{FX_CUTOFF}"\n'
        'declare -x LUMINA_NLTE_ASSEMBLE_GPU="0"\n')
    lumped = sum(1 for t in FX_LEVELS if t[2] >= FX_CUTOFF)
    (run / "stdout.log").write_text(
        f"  [ARTIS super-cutoff] K={FX_CUTOFF}: {lumped} levels lumped\n"
        f"    Z=14 ion=2: {cnt[(14, 2)]} levels\n    Z=16 ion=2: {cnt[(16, 2)]} levels\n"
        f"  [NLTE] Super-levels: ACTIVE (x FL -> y SL across ions)\n")
    for k, g in ((0, 3), (4, 7), (5, 8)):
        d = run / "work/physics_comparison"
        (d / f"physics_DET_iter{k:04d}.manifest.json").write_text(json.dumps(
            {"transaction_status": "COMMITTED", "te_lane": "FIXED_T", "population_generation": g,
             "radiation_generation": g - 1, "te_generation": g - 1, "shell_file": f"physics_DET_iter{k:04d}.shell.csv"}))
        with open(d / f"physics_DET_iter{k:04d}.shell.csv", "w") as f:
            f.write("shell_id,T_e_K,n_e_cm3\n")
            for s in range(FX_NS):
                Te, ne = FX_STATE[g].get(s, (10000.0 + 10 * s, 1.0e9))
                f.write(f"{s},{Te!r},{ne!r}\n")
    rows = [{"line_id": l, "shell": s, "Z": FX_LINES[l][0], "ion": 2, "beta2_E": 1e-6, "L2": 1, "D2": 1}
            for l, s in FX_SAMPLE]
    rows.append({"line_id": 4, "shell": 10, "Z": 14, "ion": 2, "beta2_E": 5e-3, "L2": 0, "D2": 0})
    rows.append({"line_id": 0, "shell": 10, "Z": 14, "ion": 1, "beta2_E": 1e-6, "L2": 0, "D2": 0})
    pd.DataFrame(rows).to_csv(base / "census.csv", index=False)
    keys = sorted((FX_LINES[l][0], l, s) for l, s in FX_SAMPLE)
    (base / "prior_sample.json").write_text(json.dumps({"keys": keys}))
    (base / "prior_result.json").write_text(json.dumps({"collision_authority": {"R": {"branch": "COLL_FIX_PER_LINE"}}}))
    if mutate:
        mutate(base, run)
    return base / "census.csv", base / "prior_sample.json", base / "prior_result.json", run


def det_coll_omega_offline_hp_tests():
    """prereg 6.2 small fixed inputs against a 60-digit reference: the e^x E1(x) port within its published accuracy
    (A&S 5.1.53 |eps| < 2e-7 on E1 + ln x, x <= 1; 5.1.56 |eps| < 5e-5 relative x e^x E1(x) bound used by CMFGEN as
    2e-8 absolute on x e^x E1), the f boundary (f = 1e-5 -> OMEGA_SET), continuity of the two branches at x = 1, and
    detailed balance of the real-collision-strength rates."""
    import decimal
    from decimal import Decimal as Dm
    ctx = decimal.Context(prec=60)
    out = []

    def e1(x):
        X = Dm(x)
        g = Dm("0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495")
        s, term, k = Dm(0), Dm(1), 1
        while True:
            term = ctx.multiply(term, ctx.divide(-X, Dm(k)))
            add = ctx.divide(term, Dm(k))
            s = ctx.add(s, add)
            if abs(add) < Dm("1e-55"):
                break
            k += 1
        return ctx.subtract(ctx.subtract(-g, ctx.ln(X)), s)
    for x in (0.01, 0.3, 0.999, 1.0, 1.001, 3.0, 10.0, 25.0):
        ref = ctx.multiply(ctx.exp(Dm(x)), e1(x))
        got = det_coll_omega_offline_ex_e1x(x)
        err = abs(Dm(got) - ref)
        bound = Dm(math.e) * Dm("2e-7") if x <= 1.0 else ctx.divide(Dm("2e-8"), Dm(x))
        out.append({"test": f"ex_e1x_{x}", "pass": bool(err <= bound), "abs_err": float(err), "bound": float(bound)})
    u_set = det_coll_omega_offline_fallback(1.0e-5, 1.0, 1.0e-11, 1.0e4, 2)
    u_vr = det_coll_omega_offline_fallback(math.nextafter(1.0e-5, 1.0), 1.0, 1.0e-11, 1.0e4, 2)
    out.append({"test": "f_boundary", "pass": u_set[1] == 3 and u_set[0] == OMCM_SET and u_vr[1] == 2,
                "at": u_set, "above": u_vr})
    lo, hi = det_coll_omega_offline_ex_e1x(1.0), det_coll_omega_offline_ex_e1x(math.nextafter(1.0, 2.0))
    out.append({"test": "branch_continuity_x1", "pass": abs(lo - hi) <= 2 * math.e * 2e-7, "diff": abs(lo - hi)})
    cu, cd = det_coll_omega_offline_rates_from_ups(1.0e4, 1.0e9, 2.0e-11, 3.0, 5.0, 0.7)
    eps = det_coll_omega_offline_db(cu, cd, 3.0, 5.0, 2.0e-11, 1.0e4)
    ref = ctx.multiply(ctx.divide(Dm(5.0), Dm(3.0)), ctx.exp(ctx.divide(Dm(-2.0e-11), ctx.multiply(Dm(K_B), Dm(1.0e4)))))
    out.append({"test": "detailed_balance_known", "pass": abs(eps) <= CRIT_DB and
                abs(Dm(cu / cd) - ref) / ref < Dm("1e-14"), "eps": eps})
    return out


def det_coll_omega_offline_selftest_cases(base, src):
    res = []

    def case(name, bundle, got, want, fault=None, results=()):
        ok = got == want
        res.append({"case": name, "bundle": bundle, "fault": fault, "pass": bool(ok), "got": str(got)[:600],
                    "want": str(want)[:600],
                    "runs": [{"validity": r.get("validity"), "analyze_rc_equivalent": 0 if r.get("validity") == "VALID" else 2,
                              "reasons": [str(x)[:160] for x in (r.get("reasons") or [])[:6]]} for r in results]})
        print(f"[{'PASS' if ok else 'FAIL'}] ({bundle}) {name}: got {str(got)[:220]} want {str(want)[:220]}", flush=True)

    def run(name, fault=None, mutate=None):
        census, ps, pr, rr = det_coll_omega_offline_fx(Path(base) / name / "in", mutate)
        r = det_coll_omega_offline_analyze(census, ps, pr, rr, src, Path(base) / name / "out", FX_TARGET, fault)
        return r, Path(base) / name / "out"

    def has(r, token):
        return r.get("validity") == "X" and any(token in str(w) for w in (r.get("reasons") or []))

    r0, o0 = run("clean")
    rt = pd.read_csv(o0 / "rates.csv")
    pairs = rt[rt.kind == "PAIR"]
    lines = rt[rt.kind == "LINE"]
    case("clean: valid; the sealed C reference agrees with Python on every line / table rate; tiers 1, 2, 3 all occur; "
         "detailed balance holds; the sample row count equals the census keys",
         "0 clean", (r0["validity"], r0["cref"]["mismatches"], sorted(set(lines.tier.dropna().astype(int))),
                     set(pairs.detailed_balance.dropna()), int((pairs.set == "sample").sum())),
         ("VALID", 0, [1, 2, 3], {"OK"}, len(FX_SAMPLE)), results=[r0])
    tab = rt[rt.kind == "TABLE"]
    case("clean: same-solve-unit table pair skipped, zero-Upsilon pair skipped and falls back to the line formula, "
         "line-less table pairs added to connected pairs, both lines of a duplicated pair summed with one table record",
         "5 additions", (set(tab[tab.lo == 6].skip.dropna()), set(tab[tab.lo == 0].skip.dropna()),
                         int(pairs[(pairs.set == "connected") & (pairs.line_less == True)].shape[0]) > 0,  # noqa: E712
                         sorted(set(pairs[(pairs.lo == 3) & (pairs.hi == 4)].n_lines.dropna().astype(int))),
                         sorted(set(pairs[(pairs.lo == 3) & (pairs.hi == 4)].n_table_records.dropna().astype(int)))),
         ({"SAME_SOLVE_UNIT"}, {"OMEGA_NOT_POSITIVE"}, True, [2], [1]))
    # 1 input linking
    def swap_tables(b, r):
        p = r / "input/model/coldata_cmfgen_manifest.csv"
        t = p.read_text().replace("ige_col_14_2_cmfgen.bin", "TMP").replace("ige_col_16_2_cmfgen.bin",
                                                                             "ige_col_14_2_cmfgen.bin").replace(
            "TMP", "ige_col_16_2_cmfgen.bin")
        p.write_text(t)
    r1a, _ = run("swap_tables", mutate=swap_tables)
    r1b, _ = run("level_shift", fault="level_shift")
    r1c, _ = run("drop_counter", fault="drop_counter")
    r1d, _ = run("gen8", fault="gen8")
    case("input linking: Si/S table files exchanged, table level index shifted by one, counterexample deleted, "
         "generation 7 replaced by 8 -> inadmissible (G-LINK)", "1 linking",
         (has(r1a, "OMEGA_LOADER_FAILS") or has(r1a, "TABLE_HEADER_NOT_MANIFEST"),
          has(r1b, "CREF_MISMATCH") or has(r1b, "OMEGA_LOADER_FAILS"), has(r1c, "COUNTER"),
          has(r1d, "MAIN_STATE_NOT_GENERATION_7")), (True, True, True, True), fault="mutations",
         results=[r1a, r1b, r1c, r1d])
    # 2 binary tables / loader
    def truncate(b, r):
        tg, recs = FX_TABLES[(14, 2)]
        det_coll_omega_offline_write_bin(r / "input/model/ige_col_14_2_cmfgen.bin", 14, 2, 8, tg, recs, truncate=True)

    def nlev_bad(b, r):
        tg, recs = FX_TABLES[(14, 2)]
        det_coll_omega_offline_write_bin(r / "input/model/ige_col_14_2_cmfgen.bin", 14, 2, 8, tg, recs, nlev_header=9)

    def missing(b, r):
        (r / "input/model/ige_col_16_2_cmfgen.bin").unlink()

    def contra(b, r):
        tg, recs = FX_TABLES[(14, 2)]
        det_coll_omega_offline_write_bin(r / "input/model/ige_col_14_2_cmfgen.bin", 14, 2, 8, tg,
                                         list(recs) + [((0, 1), [9.0, 9.0, 9.0])])
    r2 = [run(n, mutate=m)[0] for n, m in (("truncate", truncate), ("nlev_bad", nlev_bad), ("missing", missing),
                                           ("contra", contra))]
    case("loader: a truncated record, a level-count mismatch, a missing table and a contradictory duplicate fail with "
         "their cause", "2 loader",
         (has(r2[0], "TABLE_RECORD_SHORT"), has(r2[1], "TABLE_LEVEL_COUNT"), has(r2[2], "TABLE_FILE_MISSING"),
          has(r2[3], "TABLE_CONTRADICTORY_DUPLICATE")), (True, True, True, True), fault="mutations", results=r2)
    # 3 interpolation
    r3 = [run(f, fault=f)[0] for f in ("loglog", "swap_interval", "t10400")]
    case("interpolation: log-log instead of linear, the neighbouring T interval, 10400 K instead of the stored T "
         "-> Python disagrees with the sealed C (X)", "3 interpolation",
         tuple(has(r, "CREF_MISMATCH") for r in r3), (True, True, True), fault="loglog|swap_interval|t10400", results=r3)
    # 4 uncovered branch
    r4 = [run(f, fault=f)[0] for f in ("f_boundary", "drop_glo", "dE_power")]
    case("uncovered branch: the f boundary moved to f >= 1e-5, g_l dropped, the energy power changed -> C mismatch",
         "4 uncovered", tuple(has(r, "CREF_MISMATCH") for r in r4), (True, True, True),
         fault="f_boundary|drop_glo|dE_power", results=r4)
    # 5 additions and suppression
    r5 = [run(f, fault=f)[0] for f in ("table_per_line", "omit_table_pass", "suppress_ion_fallback")]
    case("additions: the table added again per line, only the selective zero kept for tier 1, the fallback suppressed "
         "for a whole covered ion -> C mismatch", "5 additions", tuple(has(r, "CREF_MISMATCH") for r in r5),
         (True, True, True), fault="table_per_line|omit_table_pass|suppress_ion_fallback", results=r5)
    # 6 solve units
    r6a, _ = run("same_unit_counted", fault="same_unit_counted")
    r6b, _ = run("drop_connected_pair", fault="drop_connected_pair")
    case("solve units: a same-unit table pair counted as a connection, a valid connected pair omitted -> X",
         "6 solve units", (has(r6a, "CREF_MISMATCH"), has(r6b, "CONNECTED_DENOMINATOR")), (True, True),
         fault="same_unit_counted|drop_connected_pair", results=[r6a, r6b])
    # 7 rates and detailed balance
    r7a, o7a = run("swap_g", fault="swap_g")
    r7b, o7b = run("common_factor", fault="common_factor")
    p7b = pd.read_csv(o7b / "rates.csv")
    case("detailed balance: exchanged statistical weights fail detailed balance; a common wrong factor on both rates "
         "keeps detailed balance but fails the C comparison", "7 detailed balance",
         (has(r7a, "X_ROWS") and has(r7a, "CREF_MISMATCH"),
          set(p7b[p7b.kind == "PAIR"].detailed_balance.dropna()), has(r7b, "CREF_MISMATCH")),
         (True, {"OK"}, True), fault="swap_g|common_factor", results=[r7a, r7b])
    # 8 zero / missing / decision
    ms = pairs[(pairs.set == "sample") & (pairs.line == 10)]
    r8 = [run(f, fault=f)[0] for f in ("fill_missing_zero", "denominator_floor", "drop_failed_rows", "threshold_shift")]
    case("zero / missing / decision: a key without levels is 5 in the clean run; filling it, adding a tiny value to "
         "a zero denominator, dropping refuted rows and moving the 20% threshold are rejected", "8 decision",
         (set(ms.status), has(r8[0], "MISSING_FILLED"), has(r8[1], "PAIR_SUM_RECOMPUTE"),
          has(r8[2], "SAMPLE_ROW_ACCOUNTING"), has(r8[3], "DECISION_RECOMPUTE"), r0["decision"] != "X"),
         ({"NOT_IDENTIFIABLE"}, True, True, True, True, True),
         fault="fill_missing_zero|denominator_floor|drop_failed_rows|threshold_shift", results=r8)
    r8b, o8b = run("x_decision", fault="drop_failed_rows")
    nx = json.loads((o8b / "next_contract.json").read_text())
    case("decision table: an inadmissible run carries no ordering decision", "8 decision",
         (nx["validity"], nx["decision"]), ("X", "X"))
    # sealed excerpt integrity
    r9, _ = run("cref_tamper", fault="cref_tamper")
    case("sealed excerpts: a modified excerpt is not verbatim and is refused", "C excerpts",
         has(r9, "EXCERPT_NOT_VERBATIM"), True, fault="cref_tamper", results=[r9])
    hp = det_coll_omega_offline_hp_tests()
    case("high-precision references (decimal 60 digits): e^x E1(x) port within its published accuracy, f boundary, "
         "branch continuity at x = 1, detailed balance of the rate form", "7 detailed balance",
         [t["test"] for t in hp if not t["pass"]], [])
    res[-1]["hp_tests"] = hp
    # 9 preservation
    try:
        det_coll_omega_offline_analyze(Path(base) / "clean/in/census.csv", Path(base) / "clean/in/prior_sample.json",
                                       Path(base) / "clean/in/prior_result.json", Path(base) / "clean/in/run", src, o0,
                                       FX_TARGET)
        pres = "WROTE"
    except FileExistsError:
        pres = "REFUSED_BEFORE_WRITING"
    case("preservation: an existing result path is refused before writing", "9 preservation", pres,
         "REFUSED_BEFORE_WRITING")
    return res


# ------------------------------------------------------------------ CLI
def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    tests = det_coll_omega_offline_selftest_cases(out / "fx", Path(a.production_source))
    fails = sum(1 for t in tests if not t["pass"])
    summary = {"schema": "DET_COLL_OMEGA_OFFLINE_SELFTEST_V1", "readout_sha256": POP.sha(Path(__file__).resolve()),
               "cases": tests, "failures": fails, "elapsed_s": time.time() - t0,
               "bundles_covered": sorted({t["bundle"] for t in tests}),
               "rc_meaning": "analyze_rc_equivalent = the rc cmd_analyze returns for that validity (0 VALID, 2 X); the "
                             "fixture analyses are function calls, not separate processes",
               "dependencies": {str(Path(m.__file__).resolve()): POP.sha(Path(m.__file__)) for m in (POP, RATES, SEED)}}
    (out / "selftest.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"failures": fails, "cases": len(tests), "elapsed_s": summary["elapsed_s"]}))
    return 0 if fails == 0 else 1


def cmd_analyze(a):
    out = Path(a.out)
    sj = json.loads((Path(a.selftest) / "selftest.json").read_text())
    if sj.get("readout_sha256") != POP.sha(Path(__file__).resolve()) or sj.get("failures") != 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": ["SELFTEST_NOT_THIS_READOUT_OR_FAILED"]}))
        return 2
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=False)
    res = det_coll_omega_offline_analyze(a.census, a.prior_sample, a.prior_result, a.r_run, Path(a.production_source),
                                         out / "analysis")
    R = Path(a.r_run)
    man = {"schema": "DET_COLL_OMEGA_OFFLINE_MANIFEST_V1",
           "contract": {"path": str(Path(a.prereg).resolve()), "sha256": POP.sha(a.prereg)},
           "readout": {"path": str(Path(__file__).resolve()), "sha256": POP.sha(Path(__file__).resolve())},
           "imports": {str(Path(m.__file__).resolve()): POP.sha(Path(m.__file__)) for m in (POP, RATES, SEED)},
           "imported_functions": {"det_siiii_pop_readout": ["sha"],
                                  "det_siiii_rates_readout": ["det_siiii_rates_coll", "det_siiii_rates_exports"],
                                  "det_seedmem_scope_readout": ["det_seedmem_scope_sample", "det_seedmem_scope_deck",
                                                                "det_seedmem_scope_sl", "det_seedmem_scope_atoi",
                                                                "TARGET"]},
           "selftest": {"path": str(Path(a.selftest) / "selftest.json"), "sha256": POP.sha(Path(a.selftest) / "selftest.json")},
           "census": {"path": a.census, "sha256": POP.sha(a.census)},
           "prior": {"sample": POP.sha(a.prior_sample), "result": POP.sha(a.prior_result)},
           "argv": sys.argv, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "inputs": {}, "outputs": {}}
    for rel in ("STAGE_IDENTITY.txt", "input/run_build_identity.json", "input/resolved_lumina.exports",
                "input/deck.sha256", "input/fixed_te_profile.txt", "stdout.log", "input/model/levels.csv",
                "input/model/line_list.csv", "input/model/coldata_cmfgen_manifest.csv",
                "work/physics_comparison/physics_DET_iter0004.manifest.json",
                "work/physics_comparison/physics_DET_iter0004.shell.csv",
                "work/physics_comparison/physics_DET_iter0000.manifest.json",
                "work/physics_comparison/physics_DET_iter0000.shell.csv"):
        p = R / rel
        if p.exists():
            man["inputs"][rel] = {"sha256": POP.sha(p), "bytes": p.stat().st_size}
    for row in (res.get("loader") or []):
        p = Path(row["bin"])
        if p.exists():
            man["inputs"][f"table:{p.name}"] = {"sha256": row["sha256"], "bytes": p.stat().st_size}
    if (out / "analysis").exists():
        for p in sorted((out / "analysis").iterdir()):
            if p.is_file():
                man["outputs"][p.name] = {"sha256": POP.sha(p), "bytes": p.stat().st_size}
    man["elapsed_s"] = time.time() - t0
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"validity": res["validity"], "reasons": res.get("reasons"), "decision": res.get("decision"),
                      "H_SMALL": res.get("H_SMALL")}, default=str)[:3000])
    return 0 if res["validity"] == "VALID" else 2


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--production-source", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("analyze")
    for k in ("--prereg", "--census", "--prior-sample", "--prior-result", "--r-run", "--production-source",
              "--selftest", "--out"):
        s.add_argument(k, required=True)
    a = p.parse_args()
    sys.exit({"selftest": cmd_selftest, "analyze": cmd_analyze}[a.cmd](a))


if __name__ == "__main__":
    main()
