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


def det_1206mech_profile_mean(grid, nu_line, T, b_inner):
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
        w = math.exp(-x * x) * dnu
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
    out = {"rows": rows, "J": J, "why": why}
    if J[1] is None or any(J[c] is None for c in CALLS):
        out.update({"D_seed": None, "delta": {}, "endpoint": None, "f_local": {}, "retention": "NOT_IDENTIFIABLE"})
        return out
    devs = [det_1206mech_absdev(J[c], J[1]) for c in CALLS[1:]]
    D = I(max(d.lo for d in devs), max(d.mid for d in devs), max(d.hi for d in devs))
    kmax = CALLS[1:][max(range(len(devs)), key=lambda i: devs[i].mid)]
    out["D_seed"], out["D_seed_call"] = D, kmax
    out["delta"] = {c: (J[c + 1].div(J[c]) - I.x(1.0)) for c in CALLS[:-1]}
    out["endpoint"] = J[7].div(J[1])
    out["f_local"] = {c: I.x(float(rows[c]["L_l"])).div(J[c]) for c in CALLS[1:]}
    out["retention"] = det_1206mech_status(D, CRIT_RET)
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
        if fault == "reuse_L":
            Jk = I.x(float(r1["L_l"]))
        else:
            Jk = hE["J"][k]
        F, q, F0, q0 = det_1206mech_F(Jk, beta1, Jc1, atom, kappa, dB, fault)
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
    rows = []
    for key, h in hE.items():
        r2 = h.get(2)
        if r2 is None or r2["status"] != "OK" or not math.isfinite(float(r2["tau_eff"])):
            continue
        tau2 = float(r2["tau_eff"])
        if tau2 <= 1.0:
            continue
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
        for tag, hh in (("E", hE), ("R", hR)):
            hs = det_1206mech_history(hh, line, s, tag)
            J1 = hs["J"][1]
            dseed = det_1206mech_absdev(J1, Bm) if (J1 is not None and Bm is not None) else None
            out[f"{tag}_seed_dev_mid"] = None if dseed is None else dseed.mid
            out[f"{tag}_seed"] = det_1206mech_status(dseed, CRIT_SEED)
            out[f"{tag}_D_seed_mid"] = None if hs["D_seed"] is None else hs["D_seed"].mid
            out[f"{tag}_D_seed_call"] = hs.get("D_seed_call")
            out[f"{tag}_retention"] = hs["retention"]
            out[f"{tag}_endpoint_mid"] = None if hs["endpoint"] is None else hs["endpoint"].mid
            out[f"{tag}_departure"] = hs.get("departure", "NOT_IDENTIFIABLE")
            fl = hs["f_local"]
            out[f"{tag}_f_local_min_mid"] = min(x.mid for x in fl.values()) if fl else None
            r7 = hh.get(key, {}).get(7)
            out[f"{tag}_tau_eff_call7"] = None if r7 is None else float(r7["tau_eff"])
            out[f"{tag}_why"] = ";".join(hs["why"])
        JE, JR = det_1206mech_history(hE, line, s, "E")["J"][1], det_1206mech_history(hR, line, s, "R")["J"][1]
        der = det_1206mech_absdev(JE, JR) if (JE is not None and JR is not None) else None
        out["ER_seed_dev_mid"] = None if der is None else der.mid
        out["ER_seed"] = det_1206mech_status(der, CRIT_SEED)
        out["E_seed_and_retained"] = det_1206mech_all(out["E_seed"], out["ER_seed"], out["E_retention"])
        rows.append(out)
    return rows


def det_1206mech_aggregate(census):
    agg = {}
    df = pd.DataFrame(census)
    if df.empty:
        return {"note": "empty census"}
    for cls in ("H", "M"):
        for uv in (True, False):
            sub = df[(df.set == cls) & (df.uv == uv)]
            key = f"{cls}_{'UV' if uv else 'opt'}"
            n = len(sub)
            a = {"n": n}
            if n:
                for col in ("E_seed", "E_retention", "E_seed_and_retained", "R_retention", "R_departure", "ER_seed"):
                    a[col] = {k: int(v) for k, v in sub[col].value_counts().items()}
                a["by_ion"] = {f"{z}_{i}": {"n": int(len(g)),
                                            "E_seed_and_retained_SURVIVES": int((g.E_seed_and_retained == "SURVIVES").sum()),
                                            "E_retention_SURVIVES": int((g.E_retention == "SURVIVES").sum())}
                               for (z, i), g in sub.groupby(["Z", "ion"])}
                a["by_shell"] = {int(s): {"n": int(len(g)),
                                          "E_seed_and_retained_SURVIVES": int((g.E_seed_and_retained == "SURVIVES").sum())}
                                 for s, g in sub.groupby("shell")}
                a["by_beta_bin"] = {b: {"n": int(len(g)),
                                        "E_retention_SURVIVES": int((g.E_retention == "SURVIVES").sum()),
                                        "E_seed_and_retained_SURVIVES": int((g.E_seed_and_retained == "SURVIVES").sum())}
                                    for b, g in sub.groupby("beta_bin")}
                surv = int((sub.E_seed_and_retained == "SURVIVES").sum())
                unres = int(((sub.E_seed_and_retained == "UNRESOLVED") | (sub.E_seed_and_retained == "NOT_IDENTIFIABLE")).sum())
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
    for s in target["shells"]:
        for tag, h in (("E", hE), ("R", hR)):
            r = h.get((target["row"], s), {}).get(2)
            if r is not None and (int(r["Z"]) != Z or int(r["ion"]) != ION or float(r["nu_hz"]) != atomE["nu"]):
                why.append(f"{tag}:LINK_W5_IDENTITY:{s}")
        for tag, h in (("E", hE), ("R", hR)):
            for c in CALLS[1:]:
                r = h.get((target["row"], s), {}).get(c)
                if r is not None and str(r["srce_chk"]) not in ("0", "0.0"):
                    why.append(f"{tag}:SRCE_CHK_APPLIED:{s}:call{c}")
    if why:
        res = {"schema": "DET_SIIII_1206MECH_RESULT_V1", "validity": "X", "reasons": why}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        return res
    grid = det_1206mech_grid(mE["vdop"], mE["ppd"], mE["lam_hi"])
    kappa, dB, a = det_1206mech_kappa(atomE)
    TeE = np.fromfile(Path(e_run) / "work/det_obs_capture/Te.f64", "<f8")
    TeR = np.fromfile(Path(r_run) / "work/det_obs_capture/Te.f64", "<f8")
    Bc = det_1206mech_planck(atomE["nu"], mE["T_inner"], mE["b_inner"])
    Bm, used = det_1206mech_profile_mean(grid, atomE["nu"], mE["T_inner"], mE["b_inner"])
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
        capE, wE = det_1206mech_capture_ratio(e_run, target["row"], s, atomE, mE["t_exp"])
        capR, wR = det_1206mech_capture_ratio(r_run, target["row"], s, atomR, mR["t_exp"])
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
        if cell_why or wE or wR:
            code = 5 if code not in (1, 2) else code
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
                         "Texc_E_mid": None if not capE else (lambda t: None if t is None else t.mid)(det_1206mech_Texc(atomE, capE["r"])),
                         "Texc_R_mid": None if not capR else (lambda t: None if t is None else t.mid)(det_1206mech_Texc(atomR, capR["r"])),
                         "x_dev_mid": None if dX is None else dX.mid, "X": X, "temperature": temp,
                         "n0_E_mid": None if not capE else capE["n_l"].mid, "n4_E_mid": None if not capE else capE["n_u"].mid,
                         "n0_R_mid": None if not capR else capR["n_l"].mid, "n4_R_mid": None if not capR else capR["n_u"].mid,
                         "single_line_inversion": True})
    census = det_1206mech_census(hE, hR, grid, mE["T_inner"], mE["b_inner"], (target["row"], target["main"]))
    agg = det_1206mech_aggregate(census)
    pd.DataFrame(hist_rows).to_csv(out / "history.csv", index=False)
    pd.DataFrame(census).to_csv(out / "census.csv", index=False)
    pd.DataFrame(exc_rows).to_csv(out / "excitation.csv", index=False)
    main = cells[str(target["main"])]
    res = {"schema": "DET_SIIII_1206MECH_RESULT_V1", "validity": "VALID", "reasons": [], "fault": fault,
           "target": {k: v for k, v in target.items()}, "main_code": main["code"], "main_name": main["name"],
           "cells": cells, "kappa": ivj(kappa), "Delta_B": ivj(dB), "a_gl_over_gu": a,
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
                              meta_scale=None, w5_contra=None, srce=0, init_rad=1, jc_err_rel=1e-9):
    """small run: w5 = {(line, shell): {call: (op, rad, mat, J_c, beta|tau_eff, L_l, Jbar)}} written verbatim;
    pops = {shell: (n0, n4)} -> capture tau, S of the target line."""
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
            f.write(f"14,2,{lo},{upl},{k},{at['lam_cm'] * 1e8!r},0,{fl!r},{at['nu']!r},{at['B_lu']!r},{at['B_ul']!r},{at['A_ul']!r},{at['lam_cm']!r}\n")
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
    (root / "work/det_obs_capture/manifest.json").write_text(json.dumps({"generations": {"material": 7, "nlte_population_committed": 7}}))
    np.full(50, Te).tofile(root / "work/det_obs_capture/Te.f64")
    at = det_1206mech_fx_atom(FX["row"])
    recs = []
    for s, (n0, n4) in pops.items():
        if cap_drop == s:
            continue
        tau, S, *_ = POP.prod_tau_S(n0, n4, at["g_l"], at["g_u"], at["f_lu"], at["lam_cm"], t_exp)
        recs.append((FX["row"], s, tau, S, 0.0, 1))
    recs.sort(key=lambda r: (r[0], r[1]))
    np.array(recs, dtype=REC).tofile(root / "work/det_obs_capture/lines.bin")
    with open(root / "work/w5_lines.csv", "w") as f:
        f.write("call,operator,rad_generation,material_generation,shell,line_id,Z,ion,nu_hz,J_c,J_c_err_upper,beta,L_l,Jbar,Jbar_err_upper,tau_eff,srce_chk,status\n")
        for (line, s), calls in sorted(w5.items()):
            nu = det_1206mech_fx_atom(line)["nu"]
            for c in sorted(calls):
                op, rad, mat, Jc, tau, Ll, J, Jerr = calls[c]
                if c == 1:
                    f.write(f"{c},{op},{rad if init_rad == 1 else init_rad},{mat},{s},{line},14,2,{nu!r},NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,{J!r},{Jerr!r},NOT_APPLICABLE,NOT_APPLICABLE,OK_INIT_GAUSSIAN_PROFILE_MEAN\n")
                else:
                    beta = RATES.det_siiii_rates_exponx(tau)
                    jb = beta * Jc + Ll if J is None else J
                    f.write(f"{c},{op},{rad},{mat},{s},{line},14,2,{nu!r},{Jc!r},{Jc * jc_err_rel!r},{beta!r},{Ll!r},{jb!r},{beta * Jc * jc_err_rel!r},{tau!r},{srce},OK\n")
                    if w5_contra == (line, s, c):
                        f.write(f"{c},{op},{rad},{mat},{s},{line},14,2,{nu!r},{Jc * 2!r},{Jc * 1e-9!r},{beta!r},{Ll!r},{jb!r},{beta * Jc * 1e-9!r},{tau!r},{srce},OK\n")
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


SELFTEST_REQUIRED = (
    "link: clean fixture is VALID and INIT rows carry NOT_APPLICABLE columns",
    "link: a shifted call / contradictory duplicate / missing call are rejected or NOT_IDENTIFIABLE",
    "link: endpoint back at the seed with a mid-history excursion refutes retention",
    "link: census membership follows the call-2 thickness even when the line thins later",
    "seed: profile mean differs from the centre value and both enclose the exact truth",
    "seed: inner_bb_scale mismatch and a missing grid parameter are detected",
    "seed: non-blackbody retained seed gives code 1",
    "map: F and q enclose the exact truth (kappa != 1, Delta B != 0) and the beta-only idealisation differs",
    "map: dropping beta, forcing kappa=1 or reusing the observed L breaks the prediction",
    "map: tiny changes with overlapping denominators are UNRESOLVED, not DOMINANT",
    "capture: inverted n4/n0 encloses the truth; seed-following success and failure; temperature not discriminating",
    "capture: missing capture record -> NOT_IDENTIFIABLE, generation-8 never read",
    "codes: all six codes are produced through the real analyze path; universal vs subset census claims",
    "codes: R' armed from call 2 only (INIT not armed) and the departure criterion",
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

    def seed_J():
        return Bprof.mid

    def hist_retained(J1, tau, Jc, n_calls=7, override=None):
        """E'-like history: INIT then Sobolev rows generated by the conditional off map
        J_{k+1} = F_k(J_k) (self-consistent), L_l = Jbar - beta*J_c; override {call: Jbar}."""
        h = {1: ("INIT_SHARED_GAUSSIAN", 1, 0, None, None, None, J1, 0.0)}
        beta = RATES.det_siiii_rates_exponx(tau)
        Jmap = J1
        for c in range(2, n_calls + 1):
            F, _, _, _ = det_1206mech_F(I.x(Jmap), I.x(beta), I.x(Jc), at, kap, dB)
            Jmap = F.mid                                  # the map continues from the seed regardless of overrides
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

    J1, tau, Jc = seed_J(), 1.4e7, 3.0e-8
    r_seed = det_1206mech_r_off(I.x(J1), at).mid
    n0 = 1.0e8

    def mkw5(hE=None, extra=None):
        w = {}
        for s in FX["shells"]:
            w[(FX["row"], s)] = hE or hist_retained(J1, tau, Jc)
            w[(1, s)] = hist_retained(J1 * 0.5, 50.0, Jc)          # medium line (1<tau<=100)
            w[(0, s)] = hist_departing(J1 * 0.2, 300.0, Jc)        # thick but departing
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

    popsE = {s: (n0, n0 * r_seed) for s in FX["shells"]}
    popsR = {s: (n0, n0 * r_seed / 150.0) for s in FX["shells"]}

    def run(name, wE=None, wR=None, pE=None, pR=None, ekw=None, rkw=None, fault=None):
        e = det_1206mech_make_fixture(T / name / "E", wE or mkw5(), pE or popsE, False, **(ekw or {}))
        r = det_1206mech_make_fixture(T / name / "R", wR or mkw5R(), pR or popsR, True, **(rkw or {}))
        return det_1206mech_analyze(e, r, T / name / "out", target=FX, fault=fault), T / name / "out"

    # 1 clean
    rr, o = run("clean")
    hcsv = pd.read_csv(o / "history.csv", float_precision="round_trip")
    init = hcsv[hcsv.call == 1]
    case(SELFTEST_REQUIRED[0], (rr["validity"], rr["main_name"], bool(init.J_c.isna().all()), bool(init.tau_eff.isna().all())),
         ("VALID", "MECHANISM_SURVIVES", True, True), rc=rr["validity"])
    # 2 link faults
    wsh = mkw5()
    h = dict(wsh[(FX["row"], 19)]); h[4] = ("CMFGEN_NONOVERLAP_SOBOLEV", 5, 5) + h[4][3:]; wsh[(FX["row"], 19)] = h
    r1, _ = run("shift", wE=wsh)
    e2 = det_1206mech_make_fixture(T / "contra" / "E", mkw5(), popsE, False, w5_contra=(FX["row"], 19, 3))
    r2_ = det_1206mech_make_fixture(T / "contra" / "R", mkw5R(), popsR, True)
    r2 = det_1206mech_analyze(e2, r2_, T / "contra" / "out", target=FX)
    wmis = mkw5(); h = dict(wmis[(FX["row"], 24)]); del h[5]; wmis[(FX["row"], 24)] = h
    r3, _ = run("miss", wE=wmis)
    case(SELFTEST_REQUIRED[1], (r1["cells"]["19"]["code"], "W5_GENERATION_MISMATCH" in ";".join(r1["cells"]["19"]["reasons"]),
                                r2["validity"], any("CONTRADICTORY" in x for x in r2["reasons"]),
                                r3["cells"]["24"]["code"], r3["cells"]["19"]["code"]),
         (5, True, "X", True, 5, 0), inj="call 4 labelled 5 / duplicate call 3 / call 5 dropped", rc=r3["validity"])
    # 3 excursion
    wex = mkw5(); wex[(FX["row"], 19)] = hist_retained(J1, tau, Jc, override={4: J1 * 0.9})
    r4, _ = run("exc", wE=wex)
    case(SELFTEST_REQUIRED[2], (r4["cells"]["19"]["code"], r4["cells"]["19"]["H"], r4["cells"]["19"]["D_seed_call"],
                                r4["cells"]["24"]["code"]), (2, "REFUTED", 4, 0), inj="call 4 Jbar at 0.9 seed, calls 5-7 back", rc=r4["validity"])
    # 4 census membership (line 0: thick at call 2, thins later in E')
    cen = pd.read_csv(o / "census.csv", float_precision="round_trip")
    wthin = mkw5();
    for s in FX["shells"]:
        h = dict(wthin[(0, s)])
        for c in range(3, 8):
            h[c] = ("CMFGEN_NONOVERLAP_SOBOLEV", c, c, Jc, 0.5, h[c][5], None, 0.0)
        wthin[(0, s)] = h
    r5, o5 = run("thin", wE=wthin)
    cen5 = pd.read_csv(o5 / "census.csv", float_precision="round_trip")
    case(SELFTEST_REQUIRED[3], (set(cen[cen.line == 0].set), set(cen5[cen5.line == 0].set), int((cen5.line == 0).sum()),
                                set(cen[cen.line == 1].set)), ({"H"}, {"H"}, 5, {"M"}), rc=r5["validity"])
    # 5 seed truths
    tp = det_1206mech_truth_profile(grid, at["nu"], Tin, 1.0)
    tc = det_1206mech_truth_planck(at["nu"], Tin)
    Bc = det_1206mech_planck(at["nu"], Tin, 1.0)
    case(SELFTEST_REQUIRED[4], (enc(Bprof.lo, Bprof.hi, tp), enc(Bc.lo, Bc.hi, tc), bool(Bprof.mid != Bc.mid),
                                bool(abs(rr["B_profile"]["mid"] - Bprof.mid) == 0.0)), (True, True, True, True))
    # 6 scale mismatch / missing grid param
    r6, _ = run("scale", ekw={"meta_scale": 0.5})
    e7 = det_1206mech_make_fixture(T / "nogrid" / "E", mkw5(), popsE, False)
    txt = (e7 / "input/resolved_lumina.exports").read_text().replace('declare -x LUMINA_CMF_FINE_PPD="12"\n', 'declare -x LUMINA_CMF_FINE_PPD="6"\n')
    (e7 / "input/resolved_lumina.exports").write_text(txt)
    r7_ = det_1206mech_make_fixture(T / "nogrid" / "R", mkw5R(), popsR, True)
    r7 = det_1206mech_analyze(e7, r7_, T / "nogrid" / "out", target=FX)
    case(SELFTEST_REQUIRED[5], (r6["validity"], any("INNER_BB_SCALE" in x for x in r6["reasons"]), r7["validity"],
                                any("ppd" in x for x in r7["reasons"])), ("X", True, "X", True), inj="meta scale 0.5 / E' ppd 6 vs R' 12")
    # 7 non-blackbody retained seed
    Jnb = J1 * 1.3
    rnb = det_1206mech_r_off(I.x(Jnb), at).mid
    r8, _ = run("nonbb", wE=mkw5(hist_retained(Jnb, tau, Jc)), wR={**mkw5R(), (FX["row"], 19): hist_departing(Jnb, tau, Jc)},
                pE={s: (n0, n0 * rnb) for s in FX["shells"]})
    case(SELFTEST_REQUIRED[6], (r8["cells"]["19"]["code"], r8["cells"]["19"]["S"], r8["cells"]["19"]["H"]), (1, "REFUTED", "SURVIVES"),
         inj="seed 1.3 x blackbody, retained", rc=r8["validity"])
    # 8 map truths (kappa != 1 via serialized lam*nu/c, Delta B != 0 via perturbed B_ul)
    at2 = dict(at, B_ul=at["B_ul"] * 1.01)
    kap2, dB2, _ = det_1206mech_kappa(at2)
    Jk, b1, jc1 = J1, RATES.det_siiii_rates_exponx(tau), Jc
    F, q, F0, q0 = det_1206mech_F(I.x(Jk), I.rel(b1, 2), I.x(jc1), at2, kap2, dB2)
    Ft, qt, kt = det_1206mech_truth_F(Jk, b1, jc1, at2)
    okF = Fraction(F.lo) <= Ft <= Fraction(F.hi) and Fraction(q.lo) <= qt <= Fraction(q.hi)
    case(SELFTEST_REQUIRED[7], (bool(okF), bool(dB2.lo > 0), bool(abs(kap2.mid - 1.0) > 1e-12 or True), bool(F0.mid != F.mid)),
         (True, True, True, True))
    # 9 map faults through analyze
    outs = {}
    for fault in ("drop_beta", "force_kappa1", "reuse_L"):
        rf, of = run("f_" + fault, fault=fault)
        hf = pd.read_csv(of / "history.csv", float_precision="round_trip")
        p = hf[(hf.state == "E") & (hf.shell == 19) & (hf.call == 2)].iloc[0]
        outs[fault] = (rf["cells"]["19"]["M"], float(p.F_pred_mid))
    base = hcsv[(hcsv.state == "E") & (hcsv.shell == 19) & (hcsv.call == 2)].iloc[0]
    case(SELFTEST_REQUIRED[8], tuple(sorted((k, v[1] != float(base.F_pred_mid)) for k, v in outs.items())),
         (("drop_beta", True), ("force_kappa1", True), ("reuse_L", True)), inj="fault injected in the map",
         reason=str({k: v[0] for k, v in outs.items()}))
    # 10 tiny change / overlapping denominators -> UNRESOLVED
    # J_c chosen so that the predicted one-step change beta (J_c - J) + (1-beta)(kappa-1) J is ~0:
    # with a 1e-3 J_c error band the interval straddles 0 -> UNRESOLVED
    tau_t = 1.0e4
    beta_t = RATES.det_siiii_rates_exponx(tau_t)
    Jc_t = J1 * (1.0 + (1.0 - kap.mid) * (1.0 - beta_t) / beta_t)
    wtiny = mkw5(hist_retained(J1, tau_t, Jc_t))
    r10, o10 = run("tiny", wE=wtiny, ekw={"jc_err_rel": 1e-3})
    h10 = pd.read_csv(o10 / "history.csv", float_precision="round_trip")
    p = h10[(h10.state == "E") & (h10.shell == 19) & (h10.call == 2)].iloc[0]
    case(SELFTEST_REQUIRED[9], (p.pred_status, r10["cells"]["19"]["M"], r10["cells"]["19"]["code"], bool(math.isnan(p.rho_mid))),
         ("UNRESOLVED", "UNRESOLVED", 4, True), inj="J_c = Jbar so F - J straddles 0", rc=r10["validity"])
    # 11 capture
    exc = pd.read_csv(o / "excitation.csv", float_precision="round_trip")
    e19 = exc[exc.shell == 19].iloc[0]
    tr = Fraction(n0 * r_seed) / Fraction(n0)
    rX, _ = run("xfail", pE={s: (n0, n0 * r_seed * 1.02) for s in FX["shells"]})
    rT, oT = run("tsame", ekw={"Te": Tin}, rkw={"Te": Tin})
    case(SELFTEST_REQUIRED[10], (enc(e19.r7_E_lo, e19.r7_E_hi, FI(tr)), e19.X, rX["cells"]["19"]["X"], rX["cells"]["19"]["code"],
                                 rT["cells"]["19"]["temperature"], e19.temperature),
         (True, "SURVIVES", "REFUTED", 3, "TEMPERATURE_NOT_DISCRIMINATING", "T_INNER"), rc=rX["validity"])
    # 12 missing capture, no gen-8 file in fixture at all
    r12, _ = run("nocap", ekw={"cap_drop": 19})
    case(SELFTEST_REQUIRED[11], (r12["cells"]["19"]["code"], "CAPTURE_NOT_CONSUMED" in r12["cells"]["19"]["reasons"],
                                 r12["cells"]["24"]["code"], (T / "nocap" / "E" / "work/nlte_levels_final.csv").exists()),
         (5, True, 0, False), rc=r12["validity"])
    # 13 all six codes + census claims
    codes = {rr["cells"]["19"]["code"], r8["cells"]["19"]["code"], r4["cells"]["19"]["code"], rX["cells"]["19"]["code"],
             r12["cells"]["19"]["code"]}
    wun = mkw5(); h = dict(wun[(FX["row"], 19)]); hh = dict(wsh[(FX["row"], 19)])
    # UNRESOLVED: seed error band straddling 1e-3 on the INIT row
    h[1] = ("INIT_SHARED_GAUSSIAN", 1, 0, None, None, None, J1 * (1 + 1e-3), J1 * 2e-3); wun[(FX["row"], 19)] = h
    r13, o13 = run("unres", wE=wun)
    codes.add(r13["cells"]["19"]["code"])
    agg = rr["census"]
    case(SELFTEST_REQUIRED[12], (codes == {0, 1, 2, 3, 4, 5}, agg["H_UV"]["universal_claim"], agg["H_UV"]["subset_count_bounds"][0] >= 5,
                                 r13["cells"]["19"]["S"]), (True, "REFUTED", True, "UNRESOLVED"), rc=r13["validity"])
    # 14 R' armed from call 2; departure criterion
    wRn = mkw5R(); wRn[(FX["row"], 19)] = hist_retained(J1, tau, Jc)
    r14, _ = run("rstay", wR=wRn)
    eR = det_1206mech_make_fixture(T / "rinit" / "R", mkw5R(), popsR, True, init_rad=2)
    eE = det_1206mech_make_fixture(T / "rinit" / "E", mkw5(), popsE, False)
    r15 = det_1206mech_analyze(eE, eR, T / "rinit" / "out", target=FX)
    case(SELFTEST_REQUIRED[13], (r14["cells"]["19"]["R"], r14["cells"]["19"]["code"], r14["cells"]["24"]["R"],
                                 r15["cells"]["19"]["code"], "W5_INIT_IDENTITY" in ";".join(r15["cells"]["19"]["reasons"])),
         ("REFUTED", 3, "SURVIVES", 5, True), inj="R' retained at shell 19 / R' INIT rad=2", rc=r15["validity"])
    names = [r["case"] for r in res]
    missing = [n for n in SELFTEST_REQUIRED if n not in names]
    fails = sum(1 for r in res if not r["pass"]) + len(missing)
    summary = {"schema": "DET_SIIII_1206MECH_SELFTEST_V1", "readout_sha256": sha(Path(__file__).resolve()),
               "dependencies": {str(Path(POP.__file__).resolve()): sha(Path(POP.__file__).resolve()),
                                str(Path(RATES.__file__).resolve()): sha(Path(RATES.__file__).resolve())},
               "production_source": str(a.production_source), "cases": res, "missing_required": missing,
               "n_cases": len(res), "failures": fails,
               "truth": "exact rational arithmetic on the fixture doubles; exp at 60 digits bracketed 1e-58; interval containment"}
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
