#!/usr/bin/env python3
"""
mk_next_epoch.py  --  build epoch N+1 of a CMFGEN SN Ia time sequence from a
converged epoch N, using CMFGEN's OWN orthodox restart mechanism.

WHY THIS IS NOT A HAND-ROLLED HOMOLOGY
--------------------------------------
The first (t=2.0 d) epoch is a steady-state model whose structure was built by
mk_sn_hydro.py (homologous scaling of the toy06 profile + an approximate kappa).
Every SUBSEQUENT epoch must NOT be hand-scaled.  Verified in the 18jun25 source:

  * At the end of a converged model, cmfgen_sub.f:4519-4521 writes
    SN_HYDRO_FOR_NEXT_MODEL (out_sn_pops_v3.f): the CONVERGED R,V,sigma,T,rho,
    Natom,Ne,clump, the *real* Rosseland-mean opacity, and per-species/isotope
    mass fractions, all stamped with epoch N's age in the header.
  * rd_sn_data.f:232-252 re-reads that file for epoch N+1 and HOMOLOGOUSLY
    rescales it from the header age to the new VADAT [SN_AGE]:
        R  += (t_new-t_old)*V ;   rho,Natom,Ne /= volume-expansion factor.
    So the next epoch's SN_HYDRO_DATA is a *byte copy* of epoch N's
    SN_HYDRO_FOR_NEXT_MODEL; setting [SN_AGE] to the new age triggers the code's
    own scaling.  Ni->Co->Fe decay over the interval is handled by
    INC_RAD_DECAYS + NUC_DECAY_DATA.  Populations are inherited by regridding the
    previous <ion>OUT departure coefficients (renamed <ion>_IN) onto the new grid
    (set_new_model_estimates.f:166 REGRID_LOG_DC_V1, DC_METH=R -- not LTE).

FILE STAGING  (= com/drad_cpmod.sh)
  SN_HYDRO_FOR_NEXT_MODEL -> SN_HYDRO_DATA        (structure, homolog. rescaled)
  CUR_MODEL_DATA          -> OLD_MODEL_DATA        (prev pops, for D/Dt SE term)
  JH_AT_CURRENT_TIME      -> JH_AT_OLD_TIME        (prev J,H, for dJ/Dt term)
  JH_AT_CURRENT_TIME_INFO -> JH_AT_OLD_TIME_INFO
  GAMMAS                  -> GAMMAS_IN             (if present)
  <ion>OUT                -> <ion>_IN              (initial population guess)
  GREY_SCL_FACOUT         -> GREY_SCL_FAC_IN, T_OUT->T_IN, ... (all *OUT->*_IN)
  VADAT, MODEL_SPEC, IN_ITS, NUC_DECAY_DATA, run.sh, setup_links.sh copied.

VADAT KEY EDITS  (= misc/set_new_sn_mod.f, the distribution's next-epoch tool)
  Every step:   [SN_AGE]=<ladder age>  [TS_NO]=<n>  [FIX_T]=T  [LIN_INT]=F
                [DC_METH]=R
  First step to time-dependent (n>=2), and idempotently thereafter:
                [DO_DDT]=T  [INCL_DJDT]=T  [INC_AD]=T
                [USE_J_REL]=F  [INCL_REL]=F  [INCL_ADV_TRANS]=F
  The last three are MANDATORY: rd_control_variables.f:1101-1104 STOPs if
  USE_DJDT_RTE (which INCL_DJDT=T forces on) and USE_J_REL are both TRUE.  The
  DJDT solver carries its own relativistic transfer (USE_FORMAL_REL auto-on).

A fresh IN_ITS is written with DO_T_AUTO / DO_LAM_AUTO so the held T (FIX_T=T) and
the LAMBDA start release themselves automatically once the restart has relaxed.

This script performs *no physics fixes*.  It only stages files and rewrites the
documented keys.  It never writes inside the source epoch directory.
"""

import os, sys, re, glob, shutil, argparse, subprocess

# ----------------------------------------------------------------- config
# VADAT keyword -> new value.  "*" values apply every step; the DDT block applies
# for ts_no >= 2 (i.e. all next epochs).  Only keys already present in VADAT are
# touched (set_new_sn_mod.f semantics); missing keys fall back to code defaults
# (INCL_DJDT=T already forces USE_DJDT_RTE=T internally; LTE_EST defaults F).
VADAT_ALWAYS = {
    "FIX_T":   "T",     # hold regridded T; released by DO_T_AUTO once relaxed
    "LIN_INT": "F",     # NEW model -> regrid pops (not a scratch continuation)
    "DC_METH": "R",     # start from previous departure coefficients, not LTE
}
VADAT_DDT = {           # applied when ts_no >= 2
    "DO_DDT":         "T",
    "INCL_DJDT":      "T",
    "INC_AD":         "T",
    "USE_J_REL":      "F",   # mandatory: mutually exclusive with USE_DJDT_RTE
    "INCL_REL":       "F",
    "INCL_ADV_TRANS": "F",
    "USE_DJDT_RTE":   "T",   # only edited if the key exists (auto-on otherwise)
    "LTE_EST":        "F",   # only edited if the key exists
}

# IN_ITS for a warm restart (set_new_sn_mod.f default flavour).
IN_ITS_TEMPLATE = """{num_its}           [NUM_ITS]          !Number of iterations
T            [DO_LAM_IT]        !Do lambda iterations first?
T            [DO_LAM_AUTO]      !Auto-switch LAMBDA -> full when relaxed
T            [DO_T_AUTO]        !Release FIX_T once sufficiently converged
"""

SETUP_LINKS = "setup_links.sh"


def log(msg):
    print("[mk_next_epoch] " + msg, flush=True)


def edit_vadat(text, changes, applied):
    """Return VADAT text with the given [KEY]=value edits applied.
    Only rewrites lines whose [KEY] already exists; records changes in `applied`."""
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        for key, val in changes.items():
            if "[" + key + "]" in ln:
                # preserve the trailing '[KEY] ...comment' part exactly.
                kpos = ln.index("[" + key + "]")
                tail = ln[kpos:]
                old_tok = ln[:kpos].strip().split()[0] if ln[:kpos].strip() else ""
                newln = f"{val:<12} {tail}"
                if not newln.endswith("\n"):
                    newln += "\n"
                if old_tok != val:
                    applied.append((key, old_tok, val))
                lines[i] = newln
                break
    return "".join(lines)


def set_vadat_age_ts(text, age, ts_no, applied):
    """SN_AGE and TS_NO carry values, not T/F, so format them numerically."""
    ch = {"SN_AGE": f"{age:g}", "TS_NO": f"{ts_no:g}"}
    return edit_vadat(text, ch, applied)


def stage(src, dst, name_from, name_to, copies, missing, required=True):
    s = os.path.join(src, name_from)
    d = os.path.join(dst, name_to)
    if os.path.exists(s):
        shutil.copy2(s, d)
        copies.append((name_from, name_to))
        return True
    missing.append((name_from, "required" if required else "optional"))
    return False


def build(src, dst, age, ts_no, num_its, dry_run=False, standin=True):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.realpath(src) == os.path.realpath(dst):
        log("ERROR: src and dst are the same directory."); return 1
    if os.path.exists(dst) and os.listdir(dst) and not dry_run:
        log(f"ERROR: destination {dst} already exists and is non-empty."); return 1
    os.makedirs(dst, exist_ok=True)

    copies, missing = [], []

    # --- time-dependent SN state files (drad_cpmod.sh core) ---------------
    sn_ok = stage(src, dst, "SN_HYDRO_FOR_NEXT_MODEL", "SN_HYDRO_DATA",
                  copies, missing, required=True)
    if not sn_ok and dry_run and standin and os.path.exists(os.path.join(src, "SN_HYDRO_DATA")):
        # Pre-convergence stand-in so the age/header mechanics can still be tested.
        shutil.copy2(os.path.join(src, "SN_HYDRO_DATA"),
                     os.path.join(dst, "SN_HYDRO_DATA"))
        copies.append(("SN_HYDRO_DATA (STAND-IN for SN_HYDRO_FOR_NEXT_MODEL)",
                       "SN_HYDRO_DATA"))
    stage(src, dst, "CUR_MODEL_DATA", "OLD_MODEL_DATA", copies, missing, True)
    stage(src, dst, "JH_AT_CURRENT_TIME", "JH_AT_OLD_TIME", copies, missing, True)
    stage(src, dst, "JH_AT_CURRENT_TIME_INFO", "JH_AT_OLD_TIME_INFO",
          copies, missing, True)
    stage(src, dst, "GAMMAS", "GAMMAS_IN", copies, missing, required=False)

    # --- <ion>OUT -> <ion>_IN  (initial population guess) ----------------
    out_files = sorted(glob.glob(os.path.join(src, "*OUT")))
    for f in out_files:
        base = os.path.basename(f)
        new = base[:-3] + "_IN"            # <X>OUT -> <X>_IN
        shutil.copy2(f, os.path.join(dst, new))
        copies.append((base, new))
    if not out_files:
        missing.append(("*OUT (departure-coef files)", "required-at-convergence"))

    # --- static control / atomic-spec files ------------------------------
    for f in ("MODEL_SPEC", "NUC_DECAY_DATA", "run.sh", "monitor.sh",
              "mk_sn_hydro.py", "gen_atomic.py"):
        stage(src, dst, f, f, copies, missing, required=(f == "MODEL_SPEC"))

    # NUC_DECAY_DATA in the live dir is a symlink; copy its target contents.
    nd = os.path.join(src, "NUC_DECAY_DATA")
    if os.path.islink(nd):
        try:
            shutil.copy2(os.path.realpath(nd), os.path.join(dst, "NUC_DECAY_DATA"))
        except OSError:
            pass

    # --- VADAT: copy then rewrite keys -----------------------------------
    with open(os.path.join(src, "VADAT")) as f:
        vtext = f.read()
    applied = []
    vtext = set_vadat_age_ts(vtext, age, ts_no, applied)
    vtext = edit_vadat(vtext, VADAT_ALWAYS, applied)
    if ts_no >= 2:
        vtext = edit_vadat(vtext, VADAT_DDT, applied)
    with open(os.path.join(dst, "VADAT"), "w") as f:
        f.write(vtext)
    copies.append(("VADAT", "VADAT (edited)"))

    # --- fresh IN_ITS ----------------------------------------------------
    with open(os.path.join(dst, "IN_ITS"), "w") as f:
        f.write(IN_ITS_TEMPLATE.format(num_its=num_its))
    copies.append(("(generated)", "IN_ITS"))

    # --- atomic symlinks: copy setup_links.sh, retarget its cd, run it ----
    sl_src = os.path.join(src, SETUP_LINKS)
    if os.path.exists(sl_src):
        with open(sl_src) as f:
            sltext = f.read()
        sltext = re.sub(r"^cd\s+\S+.*$", "cd " + dst, sltext, flags=re.MULTILINE)
        sltext = re.sub(r"/gpfs/kjhan/cmfgen_runs/toy06_2d", dst, sltext)
        sl_dst = os.path.join(dst, SETUP_LINKS)
        with open(sl_dst, "w") as f:
            f.write(sltext)
        os.chmod(sl_dst, 0o755)
        copies.append((SETUP_LINKS, SETUP_LINKS + " (retargeted)"))
        if not dry_run:
            subprocess.run(["bash", sl_dst], check=False)

    # regenerate run.sh so it points at dst
    runsh = os.path.join(dst, "run.sh")
    with open(runsh, "w") as f:
        f.write(f"""#!/bin/bash
cd {dst}
ulimit -s unlimited 2>/dev/null
export OMP_NUM_THREADS=${{OMP_NUM_THREADS:-32}}
export OMP_STACKSIZE=512M
export OMP_PROC_BIND=close
export OMP_PLACES=cores
rm -f OUTGEN batch.log
nice -n 19 /gpfs/kjhan/cmfgen_src/cur_cmf/exe/cmfgen_dev.exe > batch.log 2>&1
echo "CMFGEN_EXIT=$?" >> batch.log
""")
    os.chmod(runsh, 0o755)

    # --- report ----------------------------------------------------------
    log(f"src = {src}")
    log(f"dst = {dst}")
    log(f"new SN_AGE = {age}   TS_NO = {ts_no}   NUM_ITS = {num_its}")
    log(f"staged {len(copies)} items:")
    for a, b in copies:
        log(f"    {a:42s} -> {b}")
    log(f"VADAT key edits ({len(applied)}):")
    for k, old, new in applied:
        log(f"    [{k}] {old!r} -> {new!r}")
    if missing:
        log("MISSING sources:")
        for a, why in missing:
            log(f"    {a:42s} ({why})")

    # --- validation ------------------------------------------------------
    ok = validate(dst, age, ts_no)
    if dry_run:
        log("DRY-RUN: file mechanics exercised; not launching.")
    return 0 if ok else 2


def validate(dst, age, ts_no):
    """Sanity-check the constructed epoch directory."""
    ok = True
    # 1. SN_HYDRO_DATA present & header age readable, new age > header age
    snd = os.path.join(dst, "SN_HYDRO_DATA")
    if os.path.exists(snd):
        hdr_age = None
        with open(snd) as f:
            for ln in f:
                if "Time(days) since explosion:" in ln:
                    hdr_age = float(ln.split(":")[1])
                    break
        if hdr_age is None:
            log("VALIDATE FAIL: SN_HYDRO_DATA has no age header"); ok = False
        elif age < hdr_age - 1e-9:
            log(f"VALIDATE FAIL: new age {age} < structure age {hdr_age} "
                "(cannot go back in time; rd_sn_data.f:517 would STOP)"); ok = False
        else:
            log(f"VALIDATE ok: SN_HYDRO_DATA age {hdr_age} -> rescale to {age} "
                f"(dt={age-hdr_age:.4g} d)")
    else:
        log("VALIDATE: SN_HYDRO_DATA absent (expected only until epoch converges)")
    # 2. VADAT consistency: USE_J_REL and USE_DJDT_RTE not both T
    from cmfgen_convergence import vadat_value
    ujr = vadat_value(dst, "USE_J_REL", str, "F")
    djdt = vadat_value(dst, "INCL_DJDT", str, "F")
    if ts_no >= 2 and str(ujr).startswith("T") and str(djdt).startswith("T"):
        log("VALIDATE FAIL: USE_J_REL=T and INCL_DJDT=T -> CMFGEN STOPs "
            "(rd_control_variables.f:1101)"); ok = False
    else:
        log(f"VALIDATE ok: USE_J_REL={ujr} INCL_DJDT={djdt} DO_DDT="
            f"{vadat_value(dst,'DO_DDT',str,'?')} DC_METH="
            f"{vadat_value(dst,'DC_METH',str,'?')} TS_NO="
            f"{vadat_value(dst,'TS_NO',str,'?')} SN_AGE="
            f"{vadat_value(dst,'SN_AGE',str,'?')}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="converged epoch N directory")
    ap.add_argument("dst", help="epoch N+1 directory to create")
    ap.add_argument("--age", type=float, required=True, help="new SN age (days)")
    ap.add_argument("--ts-no", type=int, required=True, help="new time-seq number")
    ap.add_argument("--num-its", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                    help="stage into dst but do not run setup_links / launch")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    rc = build(a.src, a.dst, a.age, a.ts_no, a.num_its, dry_run=a.dry_run)
    sys.exit(rc)


if __name__ == "__main__":
    main()
