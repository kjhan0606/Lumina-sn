#!/usr/bin/env python3
"""DET-SIIII-1206-MECH readout (prereg docs/PREREG_DET_SIIII_1206_MECH_2026-10-07.md).

Subcommands
  selftest --production-source DIR --out DIR
  analyze  --e-run DIR --r-run DIR --production-source DIR --selftest DIR --out DIR

Mechanism test of the Si III 1206.50 A resonance line (0->4) in the E' (NRB off)
and R' (NRB armed) runs, from the stored W5 iteration history (calls 1-7) and the
generation-7 capture:
  S  seed identity   : W5 call-1 Jbar vs the production Gaussian-profile mean of the
                       undiluted inner blackbody B_nu(T_inner) on the fine grid (prereg 3)
  H  seed retention  : D_seed = max_k |Jbar_k/Jbar_1 - 1| <= 1e-3 in E' (prereg 4.1)
  M  conditional map : one-step off-form prediction F_k(J) with the production
                       normalisation kappa and Delta B kept (prereg 4.2-4.3)
  X  excitation      : captured n_4/n_0 vs r_off(Jbar_1) (prereg 5)
  R  armed contrast  : R' shares the seed, is armed from call 2 and departs (prereg 4.4)
plus the thick-line census of prereg 2.2 / 6.3.  Every number is an interval
[lo, hi] with a middle value; the registered 1e-3 / 1e-2 / 0.99 / 1/4 are science
criteria applied to intervals, never numerical tolerances.
"""
import argparse
import json
import math
import os
import platform
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_siiii_pop_readout as POP                                             # noqa: E402
import det_siiii_rates_readout as RATES                                         # noqa: E402
from det_siiii_pop_readout import I, printed, recover, sha, REC, C, H, C_SOB   # noqa: E402

K_B, EV_TO_ERG, FOUR_PI = 1.380649e-16, 1.602176634e-12, 12.56637061435917295385057353311801153679
U = 2.0 ** -53
Z, ION = 14, 2
TARGET = {"name": "SiIII_1206", "row": 600693, "lo": 0, "up": 4, "main": 19, "shells": [10, 15, 19, 24, 30]}
CALLS = list(range(1, 8))
NDOP = 4.0
CRIT_SEED, CRIT_RET, CRIT_LOCAL, CRIT_RHO, CRIT_DEPART = 1e-3, 1e-3, 0.99, 0.25, 1e-2
CODE_NAME = {0: "MECHANISM_SURVIVES", 1: "SEED_NOT_BLACKBODY", 2: "SEED_RETENTION_REFUTED",
             3: "RETAINED_BUT_MECHANISM_INCOMPLETE", 4: "UNRESOLVED", 5: "NOT_IDENTIFIABLE"}
W5_NUM = ["nu_hz", "J_c", "J_c_err_upper", "beta", "L_l", "Jbar", "Jbar_err_upper", "tau_eff"]


def up(x):
    return math.nextafter(x, math.inf)


def dn(x):
    return math.nextafter(x, -math.inf)


def ivj(a):
    return None if a is None else a.j()


def det_1206mech_status(d, thr, sense="le"):
    """interval d against a science criterion: sense 'le' survives when d <= thr,
    'gt' survives when d > thr."""
    if d is None:
        return "NOT_IDENTIFIABLE"
    if not (math.isfinite(d.lo) and math.isfinite(d.hi)):
        return "NOT_IDENTIFIABLE"
    if sense == "le":
        return "SURVIVES" if d.hi <= thr else "REFUTED" if d.lo > thr else "UNRESOLVED"
    return "SURVIVES" if d.lo > thr else "REFUTED" if d.hi <= thr else "UNRESOLVED"


def det_1206mech_all(*st):
    if any(s == "REFUTED" for s in st):
        return "REFUTED"
    if any(s == "NOT_IDENTIFIABLE" for s in st):
        return "NOT_IDENTIFIABLE"
    if any(s == "UNRESOLVED" for s in st):
        return "UNRESOLVED"
    return "SURVIVES"


def det_1206mech_absdev(a, b):
    """|a/b - 1| as an interval (b > 0)."""
    q = a.div(b)
    if q is None:
        return None
    return (q - I.x(1.0)).abs()


# ------------------------------------------------------------------ inputs
def det_1206mech_meta(run):
    meta = POP.read_meta(Path(run) / "work/det_obs_capture")
    ex = RATES.det_siiii_rates_exports(run)
    out = {"t_exp": float.fromhex(meta["time_explosion"]), "T_inner": float.fromhex(meta["T_inner"]),
           "b_inner": float.fromhex(meta["inner_bb_scale"]),
           "vdop": float(ex.get("LUMINA_CMF_FINE_VDOP", "1.0e6")), "ppd": float(ex.get("LUMINA_CMF_FINE_PPD", "12")),
           "lam_hi": float(ex.get("LUMINA_CMF_FINE_LAMHI", "20000")),
           "nrb": ex.get("LUMINA_BB_NRB", "0") not in ("", "0"), "exports": ex,
           "material_generation": meta.get("material_generation"), "rad_generation": meta.get("rad_generation")}
    out["why"] = []
    if abs(out["b_inner"] - float(ex.get("LUMINA_INNER_BB_SCALE", "1.0"))) != 0.0:
        out["why"].append("LINK_INNER_BB_SCALE_META_VS_EXPORT")
    man = json.load(open(Path(run) / "work/det_obs_capture/manifest.json"))
    gens = man.get("generations", {})
    if gens.get("material") != 7 or gens.get("nlte_population_committed") != 7 or gens.get("radiation", 7) != 7:
        out["why"].append("LINK_CAPTURE_GENERATION")
    if man.get("transaction", "COMMITTED") != "COMMITTED":
        out["why"].append("LINK_CAPTURE_NOT_COMMITTED")
    if man.get("census", {}).get("mapped_B_fallback", 0) != 0:
        out["why"].append("LINK_CAPTURE_SOURCE_FALLBACK")
    if str(out["material_generation"]) != "7" or str(out["rad_generation"]) != "7":
        out["why"].append("LINK_CAPTURE_META_GENERATION")
    return out


def det_1206mech_w5_load(run):
    """whole W5 history: (line, shell) -> {call: row dict}; contradictory duplicates listed."""
    hist, contra = {}, []
    cols = ["call", "operator", "rad_generation", "material_generation", "shell", "line_id", "Z", "ion", "srce_chk",
            "status"] + W5_NUM
    for ch in pd.read_csv(Path(run) / "work/w5_lines.csv", usecols=cols, na_values=["NOT_APPLICABLE"],
                          dtype={"operator": str, "status": str, "srce_chk": str, "call": int, "shell": int,
                                 "line_id": int, "Z": int, "ion": int, "rad_generation": int,
                                 "material_generation": int}, float_precision="round_trip", chunksize=1 << 18):
        for r in ch.itertuples(index=False):
            d = r._asdict()
            k = (d["line_id"], d["shell"])
            h = hist.setdefault(k, {})
            if d["call"] in h:
                if h[d["call"]] != d:
                    contra.append((k, d["call"]))
                continue
            h[d["call"]] = d
    return hist, contra


def det_1206mech_jiv(row):
    """W5 Jbar as an interval [Jbar - err, Jbar + err]."""
    j, e = float(row["Jbar"]), float(row["Jbar_err_upper"])
    if not math.isfinite(j) or not math.isfinite(e):
        return None
    return I(dn(j - e), j, up(j + e))


def det_1206mech_check_row(row, call, gen_ok=True):
    """operator / generation / status admissibility of one W5 row; reasons."""
    why = []
    if row is None:
        return ["W5_CALL_MISSING"]
    if row["status"] not in ("OK", "OK_INIT_GAUSSIAN_PROFILE_MEAN"):
        why.append("W5_STATUS_" + str(row["status"]))
    if call == 1:
        if row["operator"] != "INIT_SHARED_GAUSSIAN" or row["rad_generation"] != 1 or row["material_generation"] != 0:
            why.append("W5_INIT_IDENTITY")
    else:
        if row["operator"] != "CMFGEN_NONOVERLAP_SOBOLEV" or row["rad_generation"] != call or \
                row["material_generation"] != call:
            why.append("W5_GENERATION_MISMATCH")
        for c in ("J_c", "beta", "L_l", "tau_eff"):
            if not math.isfinite(float(row[c])):
                why.append("W5_SOBOLEV_FIELD_MISSING")
                break
        else:
            b = RATES.det_siiii_rates_exponx(float(row["tau_eff"]))
            if b is None or b != float(row["beta"]):
                why.append("W5_BETA_LEDGER")
            if float(row["beta"]) * float(row["J_c"]) + float(row["L_l"]) != float(row["Jbar"]):
                why.append("W5_JBAR_LEDGER")
    return why


# ------------------------------------------------------------------ blackbody profile mean
def det_1206mech_planck(nu, T, b=1.0):
    """production cm_planck form (prefactor/expm1, Wien above x>50) as an interval:
    x carries 3u, expm1 relative error <= (x e^x/(e^x-1)) * 3u + 2u, prefactor 6u."""
    x = H * nu / (K_B * T)
    pre = (2.0 * H * nu * nu * nu) / (C * C)
    if x > 50.0:
        y = math.exp(-x)
        v = pre * y / (1.0 - y)
    else:
        v = pre / math.expm1(x)
    ex = x * math.exp(x) / math.expm1(x) if x < 700 else x
    k = 3.0 * ex + 2.0 + 6.0 + 2.0
    out = I.rel(v, k)
    return I(dn(out.lo * b), out.mid * b, up(out.hi * b)) if b != 1.0 else out


def det_1206mech_grid(vdop, ppd, lam_hi):
    """fine-grid parameters exactly as T2 lumina_cmfgen.c builds them."""
    dlognu = (vdop / C) / ppd
    profile_width = NDOP * vdop / C
    line_nu_lo = C / (lam_hi * 1.0e-8)
    support_nu_lo = line_nu_lo * (1.0 - profile_width)
    nu_lo = support_nu_lo * math.exp(-0.5 * dlognu)
    return {"dlognu": dlognu, "nu_lo": nu_lo, "vdop": vdop}


def det_1206mech_profile_mean(grid, nu_line, T, b_inner, fault=None):
    """Gaussian-weighted mean of B_nu(T) over the fine bins with |x| <= 4 (T2
    line_jbar_gaussian_discrete_shells weights exp(-x^2) dnu_i); interval = weighted
    combination of the per-bin Planck intervals (weights carry 8u each)."""
    dl, nu0 = grid["dlognu"], grid["nu_lo"]
    dnu_dop = nu_line * grid["vdop"] / C
    i_lo = int(math.floor(math.log(nu_line * (1 - NDOP * grid["vdop"] / C) / nu0) / dl - 0.5)) - 1
    i_hi = int(math.ceil(math.log(nu_line * (1 + NDOP * grid["vdop"] / C) / nu0) / dl - 0.5)) + 1
    num_lo = num_hi = num_mid = den_lo = den_hi = den_mid = 0.0
    used = 0
    for i in range(max(i_lo, 0), i_hi + 1):
        nu = nu0 * math.exp((i + 0.5) * dl)
        dnu = nu * dl
        x = (nu - nu_line) / dnu_dop
        if abs(x) > NDOP:
            continue
        w = math.exp(-x * x) * dnu if fault != "flat_weights" else dnu
        wi = I.rel(w, 8)
        B = det_1206mech_planck(nu, T, 1.0)
        c = wi * B
        num_lo, num_mid, num_hi = dn(num_lo + c.lo), num_mid + c.mid, up(num_hi + c.hi)
        den_lo, den_mid, den_hi = dn(den_lo + wi.lo), den_mid + wi.mid, up(den_hi + wi.hi)
        used += 1
    if used == 0:
        return None, 0
    m = I(num_lo, num_mid, num_hi).div(I(den_lo, den_mid, den_hi))
    if b_inner != 1.0:
        m = I(dn(m.lo * b_inner), m.mid * b_inner, up(m.hi * b_inner))
    return m, used


# ------------------------------------------------------------------ conditional off map
def det_1206mech_atom(run, row, lo, upl):
    ll = pd.read_csv(Path(run) / "input/model/line_list.csv", skiprows=lambda i: i > 0 and (i - 1) != row,
                     float_precision="round_trip")
    r = ll.iloc[0]
    lv = pd.read_csv(Path(run) / "input/model/levels.csv", float_precision="round_trip")
    lv = lv[(lv.atomic_number == Z) & (lv.ion_number == ION)].set_index("level_number")
    why = []
    if (int(r.atomic_number), int(r.ion_number), int(r.level_number_lower), int(r.level_number_upper)) != (Z, ION, lo, upl):
        why.append("LINK_TARGET_ROW")
    if POP.deck_sha_of(run, "line_list.csv") != sha(Path(run) / "input/model/line_list.csv"):
        why.append("LINK_LINE_LIST_NOT_SEALED")
    return {"nu": float(r.nu), "lam_cm": float(r.wavelength_cm), "f_lu": float(r.f_lu), "B_lu": float(r.B_lu),
            "B_ul": float(r.B_ul), "A_ul": float(r.A_ul), "g_l": float(lv.loc[lo, "g"]), "g_u": float(lv.loc[upl, "g"]),
            "E_l": float(lv.loc[lo, "energy_eV"]), "E_u": float(lv.loc[upl, "energy_eV"]), "why": why}


def det_1206mech_kappa(atom):
    """kappa = (h nu B_lu / 4pi) / (C_Sob f lam_cm nu / c) and Delta B = B_ul - a B_lu, as
    intervals (production operation order of line_net_einstein_opacity_ratio, 6u / 3u)."""
    nu, f, lam, Blu, Bul = atom["nu"], atom["f_lu"], atom["lam_cm"], atom["B_lu"], atom["B_ul"]
    sob = C_SOB * f * lam * nu / C
    ein = H * nu * Blu / FOUR_PI
    kappa = I.rel(ein / sob, 8)
    a = atom["g_l"] / atom["g_u"]
    dB = I.x(Bul) - I.x(a) * I.x(Blu)
    return kappa, dB, a


def det_1206mech_r_off(J, atom):
    """isolated-transition equilibrium ratio r = B_lu J / (A + B_ul J)."""
    num = I.x(atom["B_lu"]) * J
    den = I.x(atom["A_ul"]) + I.x(atom["B_ul"]) * J
    return num.div(den)


def det_1206mech_F(J, beta1, Jc1, atom, kappa, dB, fault=None):
    """F_k(J) = beta_{k+1} J_c,k+1 + (1-beta_{k+1}) kappa A J/(A + dB J); q_k = its derivative
    at fixed background; also the beta-only idealisation (kappa=1, dB=0)."""
    A = I.x(atom["A_ul"])
    omb = I.x(1.0) - beta1 if fault != "drop_beta" else I.x(1.0)
    kap = kappa if fault != "force_kappa1" else I.x(1.0)
    den = A + dB * J
    F = beta1 * Jc1 + omb * ((kap * A * J).div(den))
    q = omb * ((kap * A * A).div(den * den))
    F0 = beta1 * Jc1 + (I.x(1.0) - beta1) * J
    q0 = I.x(1.0) - beta1
    return F, q, F0, q0


# ------------------------------------------------------------------ history / predictions
def det_1206mech_history(hist, line, s, tag):
    rows = {c: hist.get((line, s), {}).get(c) for c in CALLS}
    why = []
    for c in CALLS:
        why += [f"{tag}:call{c}:{w}" for w in det_1206mech_check_row(rows[c], c)]
    J = {c: (det_1206mech_jiv(rows[c]) if rows[c] is not None else None) for c in CALLS}
    out = {"rows": rows, "J": J, "why": why, "missing_calls": [c for c in CALLS if J[c] is None]}
    out.update({"D_seed": None, "D_seed_call": None, "delta": {}, "endpoint": None, "f_local": {},
                "retention": "NOT_IDENTIFIABLE", "departure": "NOT_IDENTIFIABLE"})
    if J[1] is None:
        return out
    avail = [c for c in CALLS[1:] if J[c] is not None]
    if avail:
        devs = {c: det_1206mech_absdev(J[c], J[1]) for c in avail}
        D = I(max(d.lo for d in devs.values()), max(d.mid for d in devs.values()), max(d.hi for d in devs.values()))
        out["D_seed"], out["D_seed_call"] = D, max(avail, key=lambda c: devs[c].mid)
        out["delta"] = {c: (J[c + 1].div(J[c]) - I.x(1.0)) for c in CALLS[:-1] if J[c] is not None and J[c + 1] is not None}
        out["f_local"] = {c: I.x(float(rows[c]["L_l"])).div(J[c]) for c in avail}
        st = det_1206mech_status(D, CRIT_RET)
        # a confirmed counterexample on the available calls is kept even when other calls are missing
        out["retention"] = st if (st == "REFUTED" or not out["missing_calls"]) else "NOT_IDENTIFIABLE"
    if J[7] is not None:
        out["endpoint"] = J[7].div(J[1])
        out["departure"] = det_1206mech_status((out["endpoint"] - I.x(1.0)).abs(), CRIT_DEPART, "gt")
    return out


def det_1206mech_predict(hE, atom, kappa, dB, fault=None):
    """E' conditional one-step predictions for k = 1..6 (prereg 4.3)."""
    preds = []
    for k in CALLS[:-1]:
        rk, r1 = hE["rows"][k], hE["rows"][k + 1]
        if rk is None or r1 is None or hE["J"][k] is None or hE["J"][k + 1] is None:
            preds.append({"k": k, "status": "NOT_IDENTIFIABLE"})
            continue
        beta1 = I.rel(float(r1["beta"]), 2)
        Jc1 = I(dn(float(r1["J_c"]) - float(r1["J_c_err_upper"])), float(r1["J_c"]),
                up(float(r1["J_c"]) + float(r1["J_c_err_upper"])))
        Jk = hE["J"][k]
        F, q, F0, q0 = det_1206mech_F(Jk, beta1, Jc1, atom, kappa, dB, fault)
        if fault == "reuse_L":                      # forbidden by prereg 4.3: F = beta J_c + L_observed(k+1)
            F = beta1 * Jc1 + I.x(float(r1["L_l"]))
        Jn = hE["J"][k + 1]
        dhat = (F - Jk).div(Jk)
        dhat0 = (F0 - Jk).div(Jk)
        num = (Jn - F).abs()
        den = (F - Jk).abs()
        sep = den.lo > 0.0
        rho = num.div(den) if sep else None
        sep0 = (F0 - Jk).abs().lo > 0.0
        rho0 = (Jn - F0).abs().div((F0 - Jk).abs()) if sep0 else None
        st_q = det_1206mech_all("SURVIVES" if q.hi < 1.0 else "REFUTED" if q.lo >= 1.0 else "UNRESOLVED",
                                det_1206mech_status(q, 0.0, "gt"),
                                det_1206mech_status(I.x(1.0) - q, CRIT_RET))
        st_rho = det_1206mech_status(rho, CRIT_RHO) if sep else "UNRESOLVED"
        st_rho0 = det_1206mech_status(rho0, CRIT_RHO) if sep0 else "UNRESOLVED"
        preds.append({"k": k, "J_k": Jk, "J_next": Jn, "F": F, "q": q, "delta_hat": dhat, "rho": rho,
                      "F0": F0, "q0": q0, "delta_hat0": dhat0, "rho0": rho0, "predicted_change_separated": sep,
                      "status": det_1206mech_all(st_q, st_rho), "status_q": st_q, "status_rho": st_rho,
                      "status_beta_only": det_1206mech_status(rho0, CRIT_RHO) if sep0 else "UNRESOLVED"})
    return preds


def det_1206mech_capture_ratio(run, row, s, atom, t_exp):
    recs = POP.capture_records(Path(run) / "work/det_obs_capture", [row], {s})
    rec = recs.get((row, s))
    if rec is None or rec["mapped"] != 1:
        return None, "CAPTURE_NOT_CONSUMED"
    rv = recover(rec["tau"], rec["S"], atom["g_l"], atom["g_u"], atom["f_lu"], atom["lam_cm"], t_exp)
    if rv is None:
        return None, "CAPTURE_NOT_RECOVERABLE"
    return {"n_l": rv["n_l"], "n_u": rv["n_u"], "r": rv["nu_over_nl"], "tau": rec["tau"], "S": rec["S"]}, None


def det_1206mech_boltz(atom, T):
    dE = (atom["E_u"] - atom["E_l"]) * EV_TO_ERG
    x = dE / (K_B * T)
    v = atom["g_u"] / atom["g_l"] * math.exp(-x)
    return I.rel(v, 4.0 * x + 8.0)


def det_1206mech_Texc(atom, r):
    """T_exc = dE / (k ln(g_u n_l / (g_l n_u))) from the ratio interval (None if not > 0)."""
    dE = (atom["E_u"] - atom["E_l"]) * EV_TO_ERG
    g = atom["g_u"] / atom["g_l"]
    vals = []
    for v in (r.lo, r.mid, r.hi):
        if not (v > 0.0) or not (g / v > 1.0):
            return None
        vals.append(dE / (K_B * math.log(g / v)))
    return I(dn(min(vals) * (1 - 8 * U)), vals[1], up(max(vals) * (1 + 8 * U)))


def det_1206mech_logdist(q):
    """|ln q| as an interval for a positive ratio interval q (0 when q straddles 1)."""
    if q is None or not (q.lo > 0.0):
        return None
    ends = (abs(math.log(q.lo)), abs(math.log(q.hi)))
    lo = 0.0 if q.lo <= 1.0 <= q.hi else dn(min(ends) * (1 - 4 * U))
    return I(lo, abs(math.log(q.mid)), up(max(ends) * (1 + 4 * U)))


def det_1206mech_decide(S, Hs, M, X, R):
    """prereg 6.2 priority: H refuted ->2; H survives & S refuted ->1; H,S survive & any of
    M,X,R refuted ->3; all survive ->0; missing ->5; else 4."""
    if Hs == "REFUTED":
        return 2
    if Hs == "SURVIVES" and S == "REFUTED":
        return 1
    if Hs == "SURVIVES" and S == "SURVIVES" and "REFUTED" in (M, X, R):
        return 3
    if all(v == "SURVIVES" for v in (S, Hs, M, X, R)):
        return 0
    if "NOT_IDENTIFIABLE" in (S, Hs, M, X, R):
        return 5
    return 4


# ------------------------------------------------------------------ census
def det_1206mech_census(hE, hR, grid, T_inner, b_inner, target_key):
    """prereg 2.2 / 6.3: fixed population by the E' call-2 tau_eff; every excluded key is
    counted with its reason (excl); admissibility of the key's own rows enters the status."""
    rows, excl = [], {}

    def ex(reason, key, r):
        d = excl.setdefault(reason, {"n": 0, "by_shell": {}, "by_ion": {}})
        d["n"] += 1
        d["by_shell"][str(key[1])] = d["by_shell"].get(str(key[1]), 0) + 1
        if r is not None:
            ik = f"{int(r['Z'])}_{int(r['ion'])}"
            d["by_ion"][ik] = d["by_ion"].get(ik, 0) + 1
    for key, h in hE.items():
        r2 = h.get(2)
        if r2 is None:
            ex("E_CALL2_MISSING", key, h.get(1) or h.get(7)); continue
        if r2["status"] != "OK":
            ex("E_CALL2_" + str(r2["status"]), key, r2); continue
        if not math.isfinite(float(r2["tau_eff"])):
            ex("E_CALL2_TAU_NONFINITE", key, r2); continue
        tau2 = float(r2["tau_eff"])
        if tau2 <= 1.0:
            ex("THIN_TAU_LE_1", key, r2); continue
        cls = "H" if tau2 > 100.0 else "M"
        line, s = key
        nu = float(r2["nu_hz"])
        lam = C / nu * 1e8
        beta2 = float(r2["beta"])
        bbin = ("(0,1e-6]" if beta2 <= 1e-6 else "(1e-6,1e-4]" if beta2 <= 1e-4 else
                "(1e-4,1e-2]" if beta2 <= 1e-2 else "(1e-2,1]")
        Bm, used = det_1206mech_profile_mean(grid, nu, T_inner, b_inner)
        out = {"line": line, "shell": s, "Z": int(r2["Z"]), "ion": int(r2["ion"]), "lambda_A": lam,
               "uv": lam < 2000.0, "set": cls, "tau_eff_call2": tau2, "beta_call2": beta2, "beta_bin": bbin,
               "is_target": key == target_key, "B_profile_mid": None if Bm is None else Bm.mid}
        HH = {}
        for tag, hh in (("E", hE), ("R", hR)):
            hs = det_1206mech_history(hh, line, s, tag)
            HH[tag] = hs
            bad = bool(hs["why"])                      # mislinked rows: the key is inadmissible, not a science status
            J1 = hs["J"][1]
            dseed = det_1206mech_absdev(J1, Bm) if (J1 is not None and Bm is not None) else None
            out[f"{tag}_seed_dev_mid"] = None if dseed is None else dseed.mid
            out[f"{tag}_seed"] = "INADMISSIBLE" if bad else det_1206mech_status(dseed, CRIT_SEED)
            out[f"{tag}_D_seed_mid"] = None if hs["D_seed"] is None else hs["D_seed"].mid
            out[f"{tag}_D_seed_call"] = hs.get("D_seed_call")
            out[f"{tag}_retention"] = "INADMISSIBLE" if bad else hs["retention"]
            out[f"{tag}_endpoint_mid"] = None if hs["endpoint"] is None else hs["endpoint"].mid
            out[f"{tag}_departure"] = "INADMISSIBLE" if bad else hs.get("departure", "NOT_IDENTIFIABLE")
            fl = hs["f_local"]
            out[f"{tag}_f_local_min_mid"] = min(x.mid for x in fl.values()) if fl else None
            r7 = hh.get(key, {}).get(7)
            out[f"{tag}_tau_eff_call7"] = None if r7 is None else float(r7["tau_eff"])
            out[f"{tag}_missing_calls"] = ";".join(str(c) for c in hs["missing_calls"])
            out[f"{tag}_why"] = ";".join(hs["why"])
        JE, JR = HH["E"]["J"][1], HH["R"]["J"][1]
        der = det_1206mech_absdev(JE, JR) if (JE is not None and JR is not None) else None
        out["ER_seed_dev_mid"] = None if der is None else der.mid
        out["ER_seed"] = "INADMISSIBLE" if (HH["E"]["why"] or HH["R"]["why"]) else det_1206mech_status(der, CRIT_SEED)
        out["E_seed_and_retained"] = ("INADMISSIBLE" if "INADMISSIBLE" in (out["E_seed"], out["ER_seed"], out["E_retention"])
                                      else det_1206mech_all(out["E_seed"], out["ER_seed"], out["E_retention"]))
        rows.append(out)
    return rows, excl


CENSUS_COLS = ("E_seed", "E_retention", "E_seed_and_retained", "R_retention", "R_departure", "ER_seed")


def det_1206mech_aggregate(census, excl):
    """full status counts for every census column, overall and by ion / shell / beta bin."""
    agg = {"excluded": excl}
    df = pd.DataFrame(census)
    if df.empty:
        agg["note"] = "empty census"
        return agg

    def counts(sub):
        return {col: {k: int(v) for k, v in sub[col].value_counts().items()} for col in CENSUS_COLS}
    for cls in ("H", "M"):
        for uv in (True, False):
            sub = df[(df.set == cls) & (df.uv == uv)]
            key = f"{cls}_{'UV' if uv else 'opt'}"
            n = len(sub)
            a = {"n": n}
            if n:
                a.update(counts(sub))
                a["by_ion"] = {f"{z}_{i}": {"n": int(len(g)), **counts(g)} for (z, i), g in sub.groupby(["Z", "ion"])}
                a["by_shell"] = {int(s_): {"n": int(len(g)), **counts(g)} for s_, g in sub.groupby("shell")}
                a["by_beta_bin"] = {b: {"n": int(len(g)), **counts(g)} for b, g in sub.groupby("beta_bin")}
                surv = int((sub.E_seed_and_retained == "SURVIVES").sum())
                unres = int(sub.E_seed_and_retained.isin(["UNRESOLVED", "NOT_IDENTIFIABLE", "INADMISSIBLE"]).sum())
                a["universal_claim"] = "HOLDS" if surv == n else "REFUTED" if surv + unres < n else "UNRESOLVED"
                a["subset_count_bounds"] = [surv, surv + unres]
            agg[key] = a
    return agg


# ------------------------------------------------------------------ analyze
def det_1206mech_analyze(e_run, r_run, out, target=TARGET, fault=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    why = []
    mE, mR = det_1206mech_meta(e_run), det_1206mech_meta(r_run)
    why += [f"E:{w}" for w in mE["why"]] + [f"R:{w}" for w in mR["why"]]
    for k in ("t_exp", "T_inner", "b_inner", "vdop", "ppd", "lam_hi"):
        if mE[k] != mR[k]:
            why.append(f"LINK_META_E_R_DIFFER:{k}")
    if mE["nrb"] or not mR["nrb"]:
        why.append("CONFIG_NRB_STATES")
    atomE = det_1206mech_atom(e_run, target["row"], target["lo"], target["up"])
    atomR = det_1206mech_atom(r_run, target["row"], target["lo"], target["up"])
    why += [f"E:{w}" for w in atomE["why"]] + [f"R:{w}" for w in atomR["why"]]
    if any(atomE[k] != atomR[k] for k in ("nu", "f_lu", "B_lu", "B_ul", "A_ul", "g_l", "g_u", "E_l", "E_u")):
        why.append("LINK_ATOM_E_R_DIFFER")
    hE, cE = det_1206mech_w5_load(e_run)
    hR, cR = det_1206mech_w5_load(r_run)
    why += [f"E:W5_CONTRADICTORY_DUPLICATE:{k[0]}:{k[1]}:call{c}" for k, c in cE]
    why += [f"R:W5_CONTRADICTORY_DUPLICATE:{k[0]}:{k[1]}:call{c}" for k, c in cR]
    caps = {}
    for s in target["shells"]:
        for tag, h, run in (("E", hE, e_run), ("R", hR, r_run)):
            hrows = h.get((target["row"], s), {})
            for c, r in hrows.items():
                if int(r["Z"]) != Z or int(r["ion"]) != ION or float(r["nu_hz"]) != atomE["nu"]:
                    why.append(f"{tag}:LINK_W5_IDENTITY:{s}:call{c}")
                # mislinked rows (operator / generation / ledger / status) make the input inadmissible
                for w in det_1206mech_check_row(r, c):
                    if w != "W5_CALL_MISSING":
                        why.append(f"{tag}:LINK_{w}:{s}:call{c}")
                if c >= 2 and str(r["srce_chk"]) not in ("0", "0.0"):
                    why.append(f"{tag}:SRCE_CHK_APPLIED:{s}:call{c}")
            # cross-link: the call-7 tau_eff of the material generation 7 must equal the captured tau
            atom = atomE if tag == "E" else atomR
            cap, wcap = det_1206mech_capture_ratio(run, target["row"], s, atom, (mE if tag == "E" else mR)["t_exp"])
            caps[(tag, s)] = (cap, wcap)
            r7 = hrows.get(7)
            if cap is not None and r7 is not None and float(r7["tau_eff"]) != cap["tau"]:
                why.append(f"{tag}:LINK_TAU_W5_CALL7_VS_CAPTURE:{s}")
    if why:
        res = {"schema": "DET_SIIII_1206MECH_RESULT_V1", "validity": "X", "reasons": why}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        return res
    grid = det_1206mech_grid(mE["vdop"], mE["ppd"], mE["lam_hi"])
    kappa, dB, a = det_1206mech_kappa(atomE)
    TeE = np.fromfile(Path(e_run) / "work/det_obs_capture/Te.f64", "<f8")
    TeR = np.fromfile(Path(r_run) / "work/det_obs_capture/Te.f64", "<f8")
    Bc = det_1206mech_planck(atomE["nu"], mE["T_inner"], mE["b_inner"])
    Bm, used = det_1206mech_profile_mean(grid, atomE["nu"], mE["T_inner"], mE["b_inner"], fault)
    hist_rows, exc_rows, cells = [], [], {}
    for s in target["shells"]:
        HE = det_1206mech_history(hE, target["row"], s, "E")
        HR = det_1206mech_history(hR, target["row"], s, "R")
        cell_why = HE["why"] + HR["why"]
        # S: seed identity
        dE_ = det_1206mech_absdev(HE["J"][1], Bm) if HE["J"][1] is not None and Bm is not None else None
        dR_ = det_1206mech_absdev(HR["J"][1], Bm) if HR["J"][1] is not None and Bm is not None else None
        dER = det_1206mech_absdev(HE["J"][1], HR["J"][1]) if HE["J"][1] is not None and HR["J"][1] is not None else None
        S = det_1206mech_all(det_1206mech_status(dE_, CRIT_SEED), det_1206mech_status(dR_, CRIT_SEED),
                             det_1206mech_status(dER, CRIT_SEED))
        # H: retention in E'
        Hs = HE["retention"]
        # M: local-emission share and conditional map
        fl = [det_1206mech_status(HE["f_local"][c], CRIT_LOCAL, "gt") if c in HE["f_local"] else "NOT_IDENTIFIABLE"
              for c in CALLS[1:]]
        fl = [("SURVIVES" if (HE["f_local"][c].lo >= CRIT_LOCAL) else "REFUTED" if HE["f_local"][c].hi < CRIT_LOCAL
               else "UNRESOLVED") if c in HE["f_local"] else "NOT_IDENTIFIABLE" for c in CALLS[1:]]
        preds = det_1206mech_predict(HE, atomE, kappa, dB, fault)
        M = det_1206mech_all(*fl, *[p["status"] for p in preds])
        M0 = det_1206mech_all(*[p.get("status_beta_only", "NOT_IDENTIFIABLE") for p in preds])
        # X: excitation
        capE, wE = caps[("E", s)]
        capR, wR = caps[("R", s)]
        r_seed = det_1206mech_r_off(HE["J"][1], atomE) if HE["J"][1] is not None else None
        rB_in = det_1206mech_boltz(atomE, mE["T_inner"])
        rB_TeE, rB_TeR = det_1206mech_boltz(atomE, float(TeE[s])), det_1206mech_boltz(atomE, float(TeR[s]))
        r_offB = det_1206mech_r_off(Bm, atomE) if Bm is not None else None
        dX = det_1206mech_absdev(capE["r"], r_seed) if (capE and r_seed is not None) else None
        X = det_1206mech_status(dX, CRIT_SEED)
        temp = "NOT_IDENTIFIABLE"
        if capE and r_seed is not None:
            sT = det_1206mech_status(det_1206mech_absdev(capE["r"], rB_in), CRIT_SEED)
            din, dte = det_1206mech_logdist(capE["r"].div(rB_in)), det_1206mech_logdist(capE["r"].div(rB_TeE))
            if sT == "SURVIVES" and din is not None and dte is not None:
                temp = ("T_INNER" if din.hi < dte.lo else "T_E_CLOSER" if din.lo > dte.hi
                        else "TEMPERATURE_NOT_DISCRIMINATING")
            else:
                temp = sT
        # R: armed contrast
        armed_ok = mR["nrb"] and all(HR["rows"][c] is not None and str(HR["rows"][c]["srce_chk"]) in ("0", "0.0")
                                     for c in CALLS[1:])
        R = det_1206mech_all(det_1206mech_status(dER, CRIT_SEED), "SURVIVES" if armed_ok else "NOT_IDENTIFIABLE",
                             HR.get("departure", "NOT_IDENTIFIABLE"))
        code = det_1206mech_decide(S, Hs, M, X, R)
        # data absence (missing W5 call / capture record) can only lower a cell to NOT_IDENTIFIABLE;
        # a confirmed refutation (codes 1, 2, 3) is kept
        if (cell_why or wE or wR) and code in (0, 4):
            code = 5
        cells[str(s)] = {"shell": s, "role": "MAIN" if s == target["main"] else "COMPARISON", "code": code,
                         "name": CODE_NAME[code], "S": S, "H": Hs, "M": M, "M_beta_only": M0, "X": X, "R": R,
                         "temperature": temp, "D_seed": ivj(HE["D_seed"]), "D_seed_call": HE.get("D_seed_call"),
                         "R_D_seed": ivj(HR["D_seed"]), "R_endpoint": ivj(HR["endpoint"]),
                         "seed_dev_E": ivj(dE_), "seed_dev_R": ivj(dR_), "seed_dev_ER": ivj(dER),
                         "x_dev": ivj(dX), "reasons": cell_why + [w for w in (wE, wR) if w]}
        for tag, HH in (("E", HE), ("R", HR)):
            for c in CALLS:
                r = HH["rows"][c]
                J = HH["J"][c]
                p = next((p for p in preds if p["k"] == c - 1), None) if tag == "E" else None
                hist_rows.append({"state": tag, "shell": s, "call": c,
                                  "operator": None if r is None else r["operator"],
                                  "rad_generation": None if r is None else r["rad_generation"],
                                  "material_generation": None if r is None else r["material_generation"],
                                  "Jbar": None if r is None else float(r["Jbar"]),
                                  "Jbar_lo": None if J is None else J.lo, "Jbar_hi": None if J is None else J.hi,
                                  "J_c": None if r is None else float(r["J_c"]), "beta": None if r is None else float(r["beta"]),
                                  "L_l": None if r is None else float(r["L_l"]), "tau_eff": None if r is None else float(r["tau_eff"]),
                                  "f_local_mid": HH["f_local"][c].mid if c in HH["f_local"] else None,
                                  "seed_ratio_mid": None if (J is None or HH["J"][1] is None) else J.mid / HH["J"][1].mid,
                                  "delta_mid": HH["delta"][c - 1].mid if (c - 1) in HH["delta"] else None,
                                  "F_pred_mid": None if p is None or "F" not in p else p["F"].mid,
                                  "F_pred_lo": None if p is None or "F" not in p else p["F"].lo,
                                  "F_pred_hi": None if p is None or "F" not in p else p["F"].hi,
                                  "q_mid": None if p is None or "q" not in p else p["q"].mid,
                                  "q_lo": None if p is None or "q" not in p else p["q"].lo,
                                  "q_hi": None if p is None or "q" not in p else p["q"].hi,
                                  "delta_hat_mid": None if p is None or "delta_hat" not in p else p["delta_hat"].mid,
                                  "rho_mid": None if p is None or p.get("rho") is None else p["rho"].mid,
                                  "rho_hi": None if p is None or p.get("rho") is None else p["rho"].hi,
                                  "q0_mid": None if p is None or "q0" not in p else p["q0"].mid,
                                  "delta_hat0_mid": None if p is None or "delta_hat0" not in p else p["delta_hat0"].mid,
                                  "rho0_mid": None if p is None or p.get("rho0") is None else p["rho0"].mid,
                                  "pred_status": None if p is None else p["status"],
                                  "pred_status_beta_only": None if p is None else p.get("status_beta_only"),
                                  "B_profile_mid": None if Bm is None else Bm.mid, "B_centre_mid": Bc.mid})
        exc_rows.append({"shell": s, "role": cells[str(s)]["role"],
                         "r7_E_lo": None if not capE else capE["r"].lo, "r7_E_mid": None if not capE else capE["r"].mid,
                         "r7_E_hi": None if not capE else capE["r"].hi,
                         "r7_R_lo": None if not capR else capR["r"].lo, "r7_R_mid": None if not capR else capR["r"].mid,
                         "r7_R_hi": None if not capR else capR["r"].hi,
                         "r_seed_lo": None if r_seed is None else r_seed.lo, "r_seed_mid": None if r_seed is None else r_seed.mid,
                         "r_seed_hi": None if r_seed is None else r_seed.hi,
                         "r_off_Bprofile_mid": None if r_offB is None else r_offB.mid,
                         "r_boltz_Tinner_mid": rB_in.mid, "r_boltz_TeE_mid": rB_TeE.mid, "r_boltz_TeR_mid": rB_TeR.mid,
                         "Te7_E": float(TeE[s]), "Te7_R": float(TeR[s]), "T_inner": mE["T_inner"],
                         **{f"Texc_E_{k}": (None if not capE or det_1206mech_Texc(atomE, capE["r"]) is None
                                            else getattr(det_1206mech_Texc(atomE, capE["r"]), k)) for k in ("lo", "mid", "hi")},
                         **{f"Texc_R_{k}": (None if not capR or det_1206mech_Texc(atomR, capR["r"]) is None
                                            else getattr(det_1206mech_Texc(atomR, capR["r"]), k)) for k in ("lo", "mid", "hi")},
                         "x_dev_mid": None if dX is None else dX.mid, "X": X, "temperature": temp,
                         "n0_E_mid": None if not capE else capE["n_l"].mid, "n4_E_mid": None if not capE else capE["n_u"].mid,
                         "n0_R_mid": None if not capR else capR["n_l"].mid, "n4_R_mid": None if not capR else capR["n_u"].mid,
                         "single_line_inversion": True})
    census, excl = det_1206mech_census(hE, hR, grid, mE["T_inner"], mE["b_inner"], (target["row"], target["main"]))
    agg = det_1206mech_aggregate(census, excl)
    pd.DataFrame(hist_rows).to_csv(out / "history.csv", index=False)
    pd.DataFrame(census).to_csv(out / "census.csv", index=False)
    pd.DataFrame(exc_rows).to_csv(out / "excitation.csv", index=False)
    main = cells[str(target["main"])]
    res = {"schema": "DET_SIIII_1206MECH_RESULT_V1", "validity": "VALID", "reasons": [], "fault": fault,
           "target": {k: v for k, v in target.items()}, "main_code": main["code"], "main_name": main["name"],
           "cells": cells, "kappa": ivj(kappa), "Delta_B": ivj(dB), "a_gl_over_gu": a,
           "capture_authority": {"generations": "material 7 / nlte_population_committed 7 / radiation 7 (checked)",
                                 "transaction": "COMMITTED (checked)", "mapped_B_fallback": "0 (checked)",
                                 "tau_cross_link": "W5 call-7 tau_eff == captured tau, bitwise, every shell (checked)"},
           "B_centre": ivj(Bc), "B_profile": ivj(Bm), "profile_bins_used": used,
           "grid": grid, "T_inner": mE["T_inner"], "b_inner": mE["b_inner"], "t_exp": mE["t_exp"],
           "census": agg, "production_solution": "PRODUCTION_SOLUTION_NOT_TESTABLE (inherited, prereg 5.3)",
           "claim_ceiling": "seed identity / retention / conditional one-step map / captured excitation ratio; "
                            "no verdict on which form is physically right, NRB promotion, convergence, "
                            "UV2500 residual or sulfur causal share"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def cmd_analyze(a):
    out = Path(a.out)
    stj = json.load(open(Path(a.selftest) / "selftest.json"))
    if stj.get("readout_sha256") != sha(Path(__file__).resolve()) or stj.get("failures") != 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": ["SELFTEST_NOT_THIS_READOUT_OR_FAILED"]}, indent=1))
        print(json.dumps({"validity": "X"}))
        return 2
    res = det_1206mech_analyze(a.e_run, a.r_run, out)
    src = Path(a.production_source)
    T2_FILES = ("src/lumina_cmfgen.c", "src/lumina_plasma.c", "src/line_net_rate.c", "src/line_jbar.c",
                "src/line_jbar.h", "src/lumina.h")
    man = {"schema": "DET_SIIII_1206MECH_MANIFEST_V1", "prereg": "docs/PREREG_DET_SIIII_1206_MECH_2026-10-07.md",
           "readout": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
           "dependencies": {str(Path(POP.__file__).resolve()): sha(Path(POP.__file__).resolve()),
                            str(Path(RATES.__file__).resolve()): sha(Path(RATES.__file__).resolve())},
           "dependency_functions": ["det_siiii_pop_readout: I, printed, recover, capture_records, read_meta, deck_sha_of, sha",
                                    "det_siiii_rates_readout: det_siiii_rates_exports, det_siiii_rates_exponx"],
           "selftest": {"path": str(Path(a.selftest) / "selftest.json"), "sha256": sha(Path(a.selftest) / "selftest.json")},
           "production_source": {f: sha(src / f) for f in T2_FILES},
           "argv": sys.argv, "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
           "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "inputs": {}, "outputs": {}}
    for tag, run in (("E", a.e_run), ("R", a.r_run)):
        for rel in ("input/resolved_lumina.exports", "input/deck.sha256", "input/model/line_list.csv",
                    "input/model/levels.csv", "work/det_obs_capture/manifest.json", "work/det_obs_capture/meta.txt",
                    "work/det_obs_capture/Te.f64", "work/w5_lines.csv"):
            p = Path(run) / rel
            man["inputs"][f"{tag}:{rel}"] = {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
        lb = Path(run) / "work/det_obs_capture/lines.bin"
        man["inputs"][f"{tag}:work/det_obs_capture/lines.bin"] = {"path": str(lb), "bytes": lb.stat().st_size,
                                                                  "identity": "sealed by capture manifest.json; one record per shell read"}
    for p in sorted(out.iterdir()):
        man["outputs"][p.name] = sha(p)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"validity": res["validity"], "main": res.get("main_name"),
                      "cells": {k: v["name"] for k, v in res.get("cells", {}).items()}, "reasons": res.get("reasons")}))
    return 0 if res["validity"] == "VALID" else 2


# ------------------------------------------------------------------ fixture + exact truths
FX = {"row": 3, "lo": 0, "up": 4, "shells": [10, 15, 19, 24, 30], "main": 19, "name": "fx1206"}
FX_LEVELS = [(0, 1, 0.0), (1, 1, 6.537), (2, 3, 6.553), (3, 5, 6.585), (4, 3, 10.2763571337), (5, 5, 15.1527)]
FX_LINES = [(0, 2, 2.3e-5), (4, 5, 0.045), (2, 5, 3.0e-4), (0, 4, 1.7), (1, 4, 0.02)]   # row 3 = 0->4 target


def det_1206mech_fx_lam(lo, upl):
    return H * C / ((FX_LEVELS[upl][2] - FX_LEVELS[lo][2]) * EV_TO_ERG)


def det_1206mech_fx_atom(k):
    lo, upl, fl = FX_LINES[k]
    lam = det_1206mech_fx_lam(lo, upl)
    nu = C / lam
    gl, gu = FX_LEVELS[lo][1], FX_LEVELS[upl][1]
    A = 6.6702e15 * gl / gu * fl / (lam * 1e8) ** 2
    Bul = A * C * C / (2 * H * nu ** 3)
    Blu = Bul * gu / gl
    return {"nu": nu, "lam_cm": lam, "f_lu": fl, "B_lu": Blu, "B_ul": Bul, "A_ul": A, "g_l": float(gl), "g_u": float(gu),
            "E_l": FX_LEVELS[lo][2], "E_u": FX_LEVELS[upl][2], "why": []}


def det_1206mech_make_fixture(root, w5, pops, nrb, T_inner=10020.0, b_inner=1.0, Te=9000.0, cap_drop=None,
                              meta_scale=None, w5_contra=None, srce=0, init_rad=1, jc_err_rel=1e-9, bul_scale=1.0,
                              cap_unmapped=None, cap_gen=7, gen8_garbage=False, init_op="INIT_SHARED_GAUSSIAN",
                              swap_shells=None, w5_outside=()):
    """small run: w5 = {(line, shell): {call: (op, rad, mat, J_c, tau_eff, L_l, Jbar, Jbar_err)}} written
    verbatim except that the target line's call-7 tau_eff is set to the captured tau of that shell
    (the production cross-link); pops = {shell: (n0, n4)} -> capture tau, S of the target line."""
    root = Path(root)
    (root / "input/model").mkdir(parents=True)
    (root / "work/det_obs_capture").mkdir(parents=True)
    with open(root / "input/model/levels.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number,energy_eV,g,metastable,super_level,configuration\n")
        for lvl, g, E in FX_LEVELS:
            f.write(f"14,2,{lvl},{E!r},{g},0,0,x\n")
    with open(root / "input/model/line_list.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number_lower,level_number_upper,line_id,wavelength,f_ul,f_lu,nu,B_lu,B_ul,A_ul,wavelength_cm\n")
        for k, (lo, upl, fl) in enumerate(FX_LINES):
            at = det_1206mech_fx_atom(k)
            bul = at["B_ul"] * (bul_scale if k == FX["row"] else 1.0)
            f.write(f"14,2,{lo},{upl},{k},{at['lam_cm'] * 1e8!r},0,{fl!r},{at['nu']!r},{at['B_lu']!r},{bul!r},{at['A_ul']!r},{at['lam_cm']!r}\n")
    (root / "input/deck.sha256").write_text(f"{sha(root / 'input/model/line_list.csv')}  line_list.csv\n")
    ex = {"LUMINA_CMF_FINE_VDOP": "1000000", "LUMINA_CMF_FINE_PPD": "12", "LUMINA_CMF_FINE_LAMHI": "20000",
          "LUMINA_INNER_BB_SCALE": repr(b_inner) if b_inner != 1.0 else "1.0", "LUMINA_NLTE_SKIP_Z": ""}
    if nrb:
        ex["LUMINA_BB_NRB"] = "1"
    (root / "input/resolved_lumina.exports").write_text("".join(f'declare -x {k}="{v}"\n' for k, v in ex.items()))
    t_exp = 1683072.0
    bi = b_inner if meta_scale is None else meta_scale
    (root / "work/det_obs_capture/meta.txt").write_text(
        f"NS=50\ntime_explosion={t_exp.hex()}\nT_inner={T_inner.hex()}\ninner_bb_scale={float(bi).hex()}\n"
        f"material_generation=7\nrad_generation=7\n")
    (root / "work/det_obs_capture/manifest.json").write_text(json.dumps(
        {"generations": {"material": cap_gen, "nlte_population_committed": cap_gen, "radiation": 7},
         "transaction": "COMMITTED", "census": {"mapped_B_fallback": 0}}))
    np.full(50, Te).tofile(root / "work/det_obs_capture/Te.f64")
    at = det_1206mech_fx_atom(FX["row"])
    recs, cap_tau = [], {}
    for s, (n0, n4) in pops.items():
        if cap_drop == s:
            continue
        tau, S, *_ = POP.prod_tau_S(n0, n4, at["g_l"], at["g_u"], at["f_lu"], at["lam_cm"], t_exp)
        recs.append((FX["row"], s, tau, S, 0.0, 0 if cap_unmapped == s else 1))
        cap_tau[s] = tau
    recs.sort(key=lambda r: (r[0], r[1]))
    np.array(recs, dtype=REC).tofile(root / "work/det_obs_capture/lines.bin")
    if gen8_garbage:
        (root / "work/nlte_levels_final.csv").write_text("this file must never be read\n\x00garbage,,,\n")
    with open(root / "work/w5_lines.csv", "w") as f:
        f.write("call,operator,rad_generation,material_generation,shell,line_id,Z,ion,nu_hz,J_c,J_c_err_upper,beta,L_l,Jbar,Jbar_err_upper,tau_eff,srce_chk,status\n")
        for (line, s), calls in sorted(w5.items()):
            nu = det_1206mech_fx_atom(line)["nu"]
            s_out = s
            if swap_shells and line == FX["row"] and s in swap_shells:
                s_out = swap_shells[1] if s == swap_shells[0] else swap_shells[0]
            if (line, s) in w5_outside:
                f.write(f"2,CMFGEN_NONOVERLAP_SOBOLEV,2,2,{s_out},{line},14,2,{nu!r},,,,,,,,,OUTSIDE_FINE_SUPPORT\n")
                continue
            for c in sorted(calls):
                op, rad, mat, Jc, tau, Ll, J, Jerr = calls[c]
                if c == 1:
                    f.write(f"{c},{init_op},{rad if init_rad == 1 else init_rad},{mat},{s_out},{line},14,2,{nu!r},NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,{J!r},{Jerr!r},NOT_APPLICABLE,NOT_APPLICABLE,OK_INIT_GAUSSIAN_PROFILE_MEAN\n")
                else:
                    if c == 7 and line == FX["row"] and s in cap_tau:
                        tau = cap_tau[s]
                    beta = RATES.det_siiii_rates_exponx(tau)
                    jb = beta * Jc + Ll if J is None else J
                    f.write(f"{c},{op},{rad},{mat},{s_out},{line},14,2,{nu!r},{Jc!r},{Jc * jc_err_rel!r},{beta!r},{Ll!r},{jb!r},{beta * Jc * jc_err_rel!r},{tau!r},{srce},OK\n")
                    if w5_contra == (line, s, c):
                        f.write(f"{c},{op},{rad},{mat},{s_out},{line},14,2,{nu!r},{Jc * 2!r},{Jc * jc_err_rel!r},{beta!r},{Ll!r},{jb!r},{beta * Jc * jc_err_rel!r},{tau!r},{srce},OK\n")
    return root


class FI:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo, self.hi = lo, (lo if hi is None else hi)

    def __add__(s, o): return FI(s.lo + o.lo, s.hi + o.hi)
    def __sub__(s, o): return FI(s.lo - o.hi, s.hi - o.lo)

    def __mul__(s, o):
        c = (s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi)
        return FI(min(c), max(c))

    def div(s, o):
        c = (s.lo / o.lo, s.lo / o.hi, s.hi / o.lo, s.hi / o.hi)
        return FI(min(c), max(c))


def _fr(v):
    return Fraction(v)


def _dexp(x):
    from decimal import Decimal, localcontext
    with localcontext() as c:
        c.prec = 60
        v = Fraction((Decimal(x.numerator) / Decimal(x.denominator)).exp())
    e = Fraction(1, 10 ** 58)
    return FI(v * (1 - e), v * (1 + e))


def det_1206mech_truth_planck(nu, T):
    x = _fr(H) * _fr(nu) / (_fr(K_B) * _fr(T))
    pre = FI(2 * _fr(H) * _fr(nu) ** 3 / _fr(C) ** 2)
    ex = _dexp(x)
    return pre.div(ex - FI(_fr(1)))


def det_1206mech_truth_profile(grid, nu_line, T, b):
    dl, nu0 = _fr(grid["dlognu"]), _fr(grid["nu_lo"])
    vd = _fr(grid["vdop"])
    num, den = FI(_fr(0)), FI(_fr(0))
    # same bin membership rule as the production (|x| <= 4) evaluated exactly on the float grid points
    i_lo = int(math.floor(math.log(nu_line * (1 - 5 * grid["vdop"] / C) / grid["nu_lo"]) / grid["dlognu"])) - 2
    i_hi = int(math.ceil(math.log(nu_line * (1 + 5 * grid["vdop"] / C) / grid["nu_lo"]) / grid["dlognu"])) + 2
    for i in range(max(i_lo, 0), i_hi + 1):
        nu_f = grid["nu_lo"] * math.exp((i + 0.5) * grid["dlognu"])      # the production float bin centre
        x_f = (nu_f - nu_line) / (nu_line * grid["vdop"] / C)
        if abs(x_f) > NDOP:
            continue
        nu = _fr(nu_f)
        x = (nu - _fr(nu_line)) / (_fr(nu_line) * vd / _fr(C))
        w = _dexp(-x * x) * FI(nu * dl)
        num = num + w * det_1206mech_truth_planck(nu_f, T)
        den = den + w
    m = num.div(den)
    return FI(m.lo * _fr(b), m.hi * _fr(b))


def det_1206mech_truth_F(J, beta1, Jc1, at, kappa_exact=None):
    A, Blu, Bul = _fr(at["A_ul"]), _fr(at["B_lu"]), _fr(at["B_ul"])
    a = _fr(at["g_l"]) / _fr(at["g_u"])
    dB = Bul - a * Blu
    kap = kappa_exact if kappa_exact is not None else (_fr(H) * _fr(at["nu"]) * Blu / _fr(FOUR_PI)) / (
        _fr(C_SOB) * _fr(at["f_lu"]) * _fr(at["lam_cm"]) * _fr(at["nu"]) / _fr(C))
    J, b1, jc = _fr(J), _fr(beta1), _fr(Jc1)
    F = b1 * jc + (1 - b1) * kap * A * J / (A + dB * J)
    q = (1 - b1) * kap * A * A / (A + dB * J) ** 2
    return F, q, kap


def _dln(x):
    from decimal import Decimal, localcontext
    with localcontext() as c:
        c.prec = 60
        v = Fraction((Decimal(x.numerator) / Decimal(x.denominator)).ln())
    e = Fraction(1, 10 ** 58)
    return FI(v - abs(v) * e, v + abs(v) * e)


def det_1206mech_truth_Texc(at, r):
    """T_exc = dE / (k ln(g_u n_l / (g_l n_u))) exactly for the rational ratio r."""
    dE = (_fr(at["E_u"]) - _fr(at["E_l"])) * _fr(EV_TO_ERG)
    g = _fr(at["g_u"]) / _fr(at["g_l"])
    L = _dln(g / r)
    return FI(dE / (_fr(K_B) * L.hi), dE / (_fr(K_B) * L.lo))


SELFTEST_REQUIRED = (
    "link: clean fixture is VALID (main code 0) and INIT rows carry NOT_APPLICABLE columns",
    "link: shifted call / contradictory duplicate / non-INIT first call / shell swap make the input inadmissible (X)",
    "link: a missing call is NOT_IDENTIFIABLE for that shell only",
    "link: endpoint back at the seed with a mid-history excursion refutes retention; a missing call does not erase it",
    "link: census population follows the call-2 thickness; excluded keys are counted with reasons; mislinked keys are INADMISSIBLE",
    "seed: profile mean differs from the centre value; stored interval encloses the exact truth; flat weights do not",
    "seed: inner_bb_scale mismatch and a grid-parameter mismatch are detected",
    "seed: non-blackbody retained seed gives code 1",
    "map: stored F and q intervals enclose the exact truth with Delta B != 0 and kappa != 1; beta-only differs",
    "map: dropping beta, forcing kappa=1 or reusing the observed L breaks the exact-truth containment",
    "map: a predicted change not separated from 0 is UNRESOLVED (code 4)",
    "capture: inverted n4/n0 and T_exc intervals enclose the exact truth; seed-following success and failure; temperature discrimination",
    "capture: missing / unmapped record -> NOT_IDENTIFIABLE; generation-8 file is never read; capture generation 8 is inadmissible",
    "codes: all six codes are produced through the real analyze path; universal vs subset census claims",
    "codes: R' retained -> R refuted (code 3); R' departure criterion survives on the clean fixture",
)


def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []
    T = out / "fx"

    def case(name, got, want, inj=None, reason=None, rc=None):
        ok = bool(got == want)
        res.append({"case": name, "pass": ok, "got": str(got)[:300], "want": str(want)[:300], "injection": inj,
                    "reason": reason, "analysis_validity": rc})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {str(got)[:150]} want {str(want)[:150]}")

    def enc(lo, hi, t):
        return Fraction(lo) <= t.lo and t.hi <= Fraction(hi)

    at = det_1206mech_fx_atom(FX["row"])
    grid = det_1206mech_grid(1.0e6, 12.0, 20000.0)
    Tin = 10020.0
    Bprof, _ = det_1206mech_profile_mean(grid, at["nu"], Tin, 1.0)
    kap, dB, a_ = det_1206mech_kappa(at)

    def hist_retained(J1, tau, Jc, n_calls=7, override=None, atom=None, kappa=None, dBv=None):
        """E'-like history generated by the conditional off map J_{k+1} = F_k(J_k) (self-consistent),
        L_l = Jbar - beta J_c; override {call: Jbar} replaces single rows without moving the map."""
        h = {1: ("INIT_SHARED_GAUSSIAN", 1, 0, None, None, None, J1, 0.0)}
        beta = RATES.det_siiii_rates_exponx(tau)
        Jmap = J1
        for c in range(2, n_calls + 1):
            F, _, _, _ = det_1206mech_F(I.x(Jmap), I.x(beta), I.x(Jc), atom or at, kappa or kap, dBv if dBv is not None else dB)
            Jmap = F.mid
            J = override[c] if (override and c in override) else Jmap
            h[c] = ("CMFGEN_NONOVERLAP_SOBOLEV", c, c, Jc, tau, J - beta * Jc, None, 0.0)
        return h

    def hist_departing(J1, tau, Jc):
        h = {1: ("INIT_SHARED_GAUSSIAN", 1, 0, None, None, None, J1, 0.0)}
        for c in range(2, 8):
            beta = RATES.det_siiii_rates_exponx(tau)
            J = J1 * 0.3 ** (c - 1)
            h[c] = ("CMFGEN_NONOVERLAP_SOBOLEV", c, c, Jc, tau, J - beta * Jc, None, 0.0)
        return h

    J1, tau, Jc = Bprof.mid, 1.4e7, 3.0e-8
    r_seed = det_1206mech_r_off(I.x(J1), at).mid
    n0 = 1.0e8

    def mkw5(hE=None, extra=None):
        w = {}
        for s in FX["shells"]:
            w[(FX["row"], s)] = (hE(s) if callable(hE) else hE) if hE is not None else hist_retained(J1, tau, Jc)
            w[(1, s)] = hist_retained(J1 * 0.5, 50.0, Jc)          # medium line (1 < tau <= 100)
            w[(0, s)] = hist_departing(J1 * 0.2, 300.0, Jc)        # thick but departing (UV, 1892 A)
        if extra:
            w.update(extra)
        return w

    def mkw5R():
        w = {}
        for s in FX["shells"]:
            w[(FX["row"], s)] = hist_departing(J1, tau, Jc)
            w[(1, s)] = hist_departing(J1 * 0.5, 50.0, Jc)
            w[(0, s)] = hist_departing(J1 * 0.2, 300.0, Jc)
        return w

    def n0s(s):                       # shell-dependent ground density so that shell labels are distinguishable
        return n0 * (1.0 + 0.01 * s)
    popsE = {s: (n0s(s), n0s(s) * r_seed) for s in FX["shells"]}
    popsR = {s: (n0s(s), n0s(s) * r_seed / 150.0) for s in FX["shells"]}

    def run(name, wE=None, wR=None, pE=None, pR=None, ekw=None, rkw=None, fault=None):
        e = det_1206mech_make_fixture(T / name / "E", wE or mkw5(), pE or popsE, False, **(ekw or {}))
        r = det_1206mech_make_fixture(T / name / "R", wR or mkw5R(), pR or popsR, True, **(rkw or {}))
        return det_1206mech_analyze(e, r, T / name / "out", target=FX, fault=fault), T / name / "out"

    def hrow(o, state, shell, call):
        h = pd.read_csv(o / "history.csv", float_precision="round_trip")
        return h[(h.state == state) & (h.shell == shell) & (h.call == call)].iloc[0]

    # 0 clean
    rr, o = run("clean")
    hcsv = pd.read_csv(o / "history.csv", float_precision="round_trip")
    init = hcsv[hcsv.call == 1]
    case(SELFTEST_REQUIRED[0], (rr["validity"], rr["main_name"], bool(init.J_c.isna().all()), bool(init.tau_eff.isna().all())),
         ("VALID", "MECHANISM_SURVIVES", True, True), rc=rr["validity"])
    # 1 inadmissible inputs
    wsh = mkw5(); h = dict(wsh[(FX["row"], 19)]); h[4] = ("CMFGEN_NONOVERLAP_SOBOLEV", 5, 5) + h[4][3:]; wsh[(FX["row"], 19)] = h
    r1, _ = run("shift", wE=wsh)
    r2, _ = run("contra", ekw={"w5_contra": (FX["row"], 19, 3)})
    r3, _ = run("noinit", rkw={"init_op": "CMFGEN_NONOVERLAP_SOBOLEV"})
    r4, _ = run("swap", ekw={"swap_shells": (19, 24)})
    case(SELFTEST_REQUIRED[1], (r1["validity"], any("GENERATION_MISMATCH" in x for x in r1["reasons"]),
                                r2["validity"], any("CONTRADICTORY" in x for x in r2["reasons"]),
                                r3["validity"], any("INIT_IDENTITY" in x for x in r3["reasons"]),
                                r4["validity"], any("TAU_W5_CALL7_VS_CAPTURE" in x for x in r4["reasons"])),
         ("X", True, "X", True, "X", True, "X", True), inj="call 4 labelled 5 / duplicate call 3 / R' call 1 Sobolev / shells 19<->24 swapped")
    # 2 missing call
    wmis = mkw5(); h = dict(wmis[(FX["row"], 24)]); del h[5]; wmis[(FX["row"], 24)] = h
    r5, _ = run("miss", wE=wmis)
    case(SELFTEST_REQUIRED[2], (r5["validity"], r5["cells"]["24"]["code"], r5["cells"]["19"]["code"],
                                "E:call5:W5_CALL_MISSING" in r5["cells"]["24"]["reasons"]), ("VALID", 5, 0, True), rc=r5["validity"])
    # 3 excursion (+ missing call)
    wex = mkw5(); wex[(FX["row"], 19)] = hist_retained(J1, tau, Jc, override={4: J1 * 0.9})
    r6, _ = run("exc", wE=wex)
    wex2 = mkw5(); h = hist_retained(J1, tau, Jc, override={4: J1 * 0.9}); del h[6]; wex2[(FX["row"], 19)] = h
    r7, _ = run("excmiss", wE=wex2)
    case(SELFTEST_REQUIRED[3], (r6["cells"]["19"]["code"], r6["cells"]["19"]["H"], r6["cells"]["19"]["D_seed_call"], r6["cells"]["24"]["code"],
                                r7["cells"]["19"]["code"], r7["cells"]["19"]["H"]),
         (2, "REFUTED", 4, 0, 2, "REFUTED"), inj="call 4 Jbar at 0.9 seed, calls 5-7 back (+ call 6 dropped)", rc=r7["validity"])
    # 4 census membership, exclusions, inadmissible non-target key
    cen = pd.read_csv(o / "census.csv", float_precision="round_trip")
    wthin = mkw5()
    for s in FX["shells"]:
        h = dict(wthin[(0, s)])
        for c in range(3, 8):
            h[c] = ("CMFGEN_NONOVERLAP_SOBOLEV", c, c, Jc, 0.5, h[c][5], None, 0.0)
        wthin[(0, s)] = h
    h = dict(wthin[(1, 19)]); h[3] = ("CMFGEN_NONOVERLAP_SOBOLEV", 4, 4) + h[3][3:]; wthin[(1, 19)] = h   # mislinked non-target key
    wthin[(2, 10)] = {1: ("INIT_SHARED_GAUSSIAN", 1, 0, None, None, None, J1, 0.0)}                         # call-2 OUTSIDE key
    r8, o8 = run("thin", wE=wthin, ekw={"w5_outside": [(2, 10)]})
    cen8 = pd.read_csv(o8 / "census.csv", float_precision="round_trip")
    excl = r8["census"]["excluded"]
    case(SELFTEST_REQUIRED[4], (set(cen[cen.line == 0].set), set(cen8[cen8.line == 0].set), int((cen8.line == 0).sum()),
                                set(cen[cen.line == 1].set), r8["validity"],
                                cen8[(cen8.line == 1) & (cen8.shell == 19)].E_seed_and_retained.iloc[0],
                                excl.get("E_CALL2_OUTSIDE_FINE_SUPPORT", {}).get("n"), excl.get("E_CALL2_MISSING", {}).get("n")),
         ({"H"}, {"H"}, 5, {"M"}, "VALID", "INADMISSIBLE", 1, None), rc=r8["validity"])
    # 5 seed truths through analyze + flat-weights fault
    tp = det_1206mech_truth_profile(grid, at["nu"], Tin, 1.0)
    tc = det_1206mech_truth_planck(at["nu"], Tin)
    rf, _ = run("flat", fault="flat_weights")
    case(SELFTEST_REQUIRED[5], (enc(rr["B_profile"]["lo"], rr["B_profile"]["hi"], tp), enc(rr["B_centre"]["lo"], rr["B_centre"]["hi"], tc),
                                bool(rr["B_profile"]["mid"] != rr["B_centre"]["mid"]),
                                enc(rf["B_profile"]["lo"], rf["B_profile"]["hi"], tp)), (True, True, True, False),
         inj="flat_weights fault in the profile mean")
    # 6 scale / grid mismatch
    r9, _ = run("scale", ekw={"meta_scale": 0.5})
    e10 = det_1206mech_make_fixture(T / "nogrid" / "E", mkw5(), popsE, False)
    txt = (e10 / "input/resolved_lumina.exports").read_text().replace('declare -x LUMINA_CMF_FINE_PPD="12"\n', 'declare -x LUMINA_CMF_FINE_PPD="6"\n')
    (e10 / "input/resolved_lumina.exports").write_text(txt)
    r10_ = det_1206mech_make_fixture(T / "nogrid" / "R", mkw5R(), popsR, True)
    r10 = det_1206mech_analyze(e10, r10_, T / "nogrid" / "out", target=FX)
    case(SELFTEST_REQUIRED[6], (r9["validity"], any("INNER_BB_SCALE" in x for x in r9["reasons"]), r10["validity"],
                                any("ppd" in x for x in r10["reasons"])), ("X", True, "X", True), inj="meta scale 0.5 / E' ppd 6 vs R' 12")
    # 7 non-blackbody retained seed
    Jnb = J1 * 1.3
    rnb = det_1206mech_r_off(I.x(Jnb), at).mid
    r11, _ = run("nonbb", wE=mkw5(hist_retained(Jnb, tau, Jc)), wR={**mkw5R(), (FX["row"], 19): hist_departing(Jnb, tau, Jc)},
                 pE={s: (n0s(s), n0s(s) * rnb) for s in FX["shells"]})
    case(SELFTEST_REQUIRED[7], (r11["cells"]["19"]["code"], r11["cells"]["19"]["S"], r11["cells"]["19"]["H"]), (1, "REFUTED", "SURVIVES"),
         inj="seed 1.3 x blackbody, retained", rc=r11["validity"])
    # 8 map truths through analyze with Delta B != 0 (B_ul scaled in the fixture atom)
    at2 = dict(at, B_ul=at["B_ul"] * 1.01)
    kap2, dB2, _ = det_1206mech_kappa(at2)
    w2 = mkw5(hist_retained(J1, tau, Jc, atom=at2, kappa=kap2, dBv=dB2))
    r12, o12 = run("dB", wE=w2, ekw={"bul_scale": 1.01}, rkw={"bul_scale": 1.01})
    okF = True
    for c in range(2, 8):
        p = hrow(o12, "E", 19, c)
        Ft, qt, _ = det_1206mech_truth_F(hrow(o12, "E", 19, c - 1).Jbar, p.beta, p.J_c, at2)
        okF &= Fraction(p.F_pred_lo) <= Ft <= Fraction(p.F_pred_hi) and Fraction(p.q_lo) <= qt <= Fraction(p.q_hi)
    p2 = hrow(o12, "E", 19, 2)
    case(SELFTEST_REQUIRED[8], (r12["validity"], bool(okF), bool(r12["Delta_B"]["lo"] > 0), bool(abs(r12["kappa"]["mid"] - 1.0) > 1e-9),
                                bool(p2.delta_hat0_mid != p2.delta_hat_mid)), ("VALID", True, True, True, True), rc=r12["validity"])
    # 9 faults break the exact-truth containment (history perturbed at call 2 so that the observed
    # J_2 is not the map image of J_1; reusing the observed L would then reproduce J_2, not F_1(J_1))
    wpert = mkw5(hist_retained(J1, tau, Jc, override={2: J1 * (1 - 1e-4)}))
    outs = {}
    for fault in ("drop_beta", "force_kappa1", "reuse_L"):
        rfx, ofx = run("f_" + fault, wE=wpert, fault=fault)
        p = hrow(ofx, "E", 19, 2)
        Ft, qt, _ = det_1206mech_truth_F(hrow(ofx, "E", 19, 1).Jbar, p.beta, p.J_c, at)
        outs[fault] = bool(Fraction(p.F_pred_lo) <= Ft <= Fraction(p.F_pred_hi))
    rp, op = run("pert", wE=wpert)
    pc = hrow(op, "E", 19, 2)
    Ft, qt, _ = det_1206mech_truth_F(hrow(op, "E", 19, 1).Jbar, pc.beta, pc.J_c, at)
    case(SELFTEST_REQUIRED[9], (bool(Fraction(pc.F_pred_lo) <= Ft <= Fraction(pc.F_pred_hi)), outs["drop_beta"], outs["force_kappa1"], outs["reuse_L"]),
         (True, False, False, False), inj="fault injected in the conditional map", reason="containment of the exact F truth")
    # 10 tiny predicted change
    def tiny_hist(s):                 # tau = the captured tau of that shell (same on every call, incl. call 7)
        tau_c = POP.prod_tau_S(popsE[s][0], popsE[s][1], at["g_l"], at["g_u"], at["f_lu"], at["lam_cm"], 1683072.0)[0]
        beta_c = RATES.det_siiii_rates_exponx(tau_c)
        return hist_retained(J1, tau_c, J1 * (1.0 + (1.0 - kap.mid) * (1.0 - beta_c) / beta_c))
    r13, o13 = run("tiny", wE=mkw5(tiny_hist), ekw={"jc_err_rel": 1e-3})
    p = hrow(o13, "E", 19, 2)
    case(SELFTEST_REQUIRED[10], (p.pred_status, r13["cells"]["19"]["M"], r13["cells"]["19"]["code"], bool(math.isnan(p.rho_mid))),
         ("UNRESOLVED", "UNRESOLVED", 4, True), inj="J_c chosen so that beta (J_c - J) + (1-beta)(kappa-1) J ~ 0 with a 1e-3 J_c band", rc=r13["validity"])
    # 11 capture truths, X, temperature
    exc = pd.read_csv(o / "excitation.csv", float_precision="round_trip")
    e19 = exc[exc.shell == 19].iloc[0]
    tr = FI(Fraction(n0s(19) * r_seed) / Fraction(n0s(19)))
    tT = det_1206mech_truth_Texc(at, tr.lo)
    rX, _ = run("xfail", pE={s: (n0s(s), n0s(s) * r_seed * 1.02) for s in FX["shells"]})
    rT, _ = run("tsame", ekw={"Te": Tin}, rkw={"Te": Tin})
    case(SELFTEST_REQUIRED[11], (enc(e19.r7_E_lo, e19.r7_E_hi, tr), enc(e19.Texc_E_lo, e19.Texc_E_hi, tT), e19.X, rX["cells"]["19"]["X"],
                                 rX["cells"]["19"]["code"], rT["cells"]["19"]["temperature"], e19.temperature),
         (True, True, "SURVIVES", "REFUTED", 3, "TEMPERATURE_NOT_DISCRIMINATING", "T_INNER"), rc=rX["validity"])
    # 12 capture admissibility
    r14, _ = run("nocap", ekw={"cap_drop": 19})
    r15, _ = run("unmapped", ekw={"cap_unmapped": 19})
    r16, _ = run("gen8", ekw={"gen8_garbage": True}, rkw={"gen8_garbage": True})
    r17, _ = run("capgen8", ekw={"cap_gen": 8})
    case(SELFTEST_REQUIRED[12], (r14["cells"]["19"]["code"], "CAPTURE_NOT_CONSUMED" in r14["cells"]["19"]["reasons"], r14["cells"]["24"]["code"],
                                 r15["cells"]["19"]["code"], r16["validity"], r16["main_code"],
                                 r17["validity"], any("CAPTURE_GENERATION" in x for x in r17["reasons"])),
         (5, True, 0, 5, "VALID", 0, "X", True), inj="capture record dropped / mapped=0 / garbage gen-8 file / manifest generation 8")
    # 13 all six codes + census claims
    wun = mkw5(); h = dict(wun[(FX["row"], 19)])
    h[1] = ("INIT_SHARED_GAUSSIAN", 1, 0, None, None, None, J1 * (1 + 1e-3), J1 * 2e-3); wun[(FX["row"], 19)] = h
    r18, _ = run("unres", wE=wun)
    codes = {rr["cells"]["19"]["code"], r11["cells"]["19"]["code"], r6["cells"]["19"]["code"], rX["cells"]["19"]["code"],
             r14["cells"]["19"]["code"], r18["cells"]["19"]["code"]}
    agg = rr["census"]
    case(SELFTEST_REQUIRED[13], (codes == {0, 1, 2, 3, 4, 5}, agg["H_UV"]["universal_claim"], agg["H_UV"]["subset_count_bounds"][0] >= 5,
                                 r18["cells"]["19"]["S"], r18["cells"]["19"]["code"]), (True, "REFUTED", True, "UNRESOLVED", 4), rc=r18["validity"])
    # 14 R' contrast
    wRn = mkw5R(); wRn[(FX["row"], 19)] = hist_retained(J1, tau, Jc)
    r19, _ = run("rstay", wR=wRn)
    case(SELFTEST_REQUIRED[14], (r19["cells"]["19"]["R"], r19["cells"]["19"]["code"], r19["cells"]["24"]["R"], rr["cells"]["19"]["R"]),
         ("REFUTED", 3, "SURVIVES", "SURVIVES"), inj="R' retained at shell 19", rc=r19["validity"])
    names = [r["case"] for r in res]
    missing = [n for n in SELFTEST_REQUIRED if n not in names]
    fails = sum(1 for r in res if not r["pass"]) + len(missing)
    summary = {"schema": "DET_SIIII_1206MECH_SELFTEST_V2", "readout_sha256": sha(Path(__file__).resolve()),
               "dependencies": {str(Path(POP.__file__).resolve()): sha(Path(POP.__file__).resolve()),
                                str(Path(RATES.__file__).resolve()): sha(Path(RATES.__file__).resolve())},
               "production_source": str(a.production_source), "cases": res, "missing_required": missing,
               "n_cases": len(res), "failures": fails,
               "truth": "exact rational arithmetic on the fixture doubles; exp/ln at 60 digits bracketed 1e-58; interval containment"}
    (out / "selftest.json").write_text(json.dumps(summary, indent=1))
    print(f"selftest: {len(res) - fails + len(missing)}/{len(res)} pass, missing {len(missing)}, failures {fails}")
    return 0 if fails == 0 else 1


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--production-source", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("analyze")
    for k in ("--e-run", "--r-run", "--production-source", "--selftest", "--out"):
        s.add_argument(k, required=True)
    a = p.parse_args()
    sys.exit(cmd_selftest(a) if a.cmd == "selftest" else cmd_analyze(a))


if __name__ == "__main__":
    main()
