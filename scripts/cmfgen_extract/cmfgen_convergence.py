#!/usr/bin/env python3
"""
cmfgen_convergence.py  --  robust convergence/termination detector for a CMFGEN run.

Used by seq_driver.sh to decide, hands-free, whether an epoch directory has
   CONVERGED  (build + launch the next epoch)
   RUNNING    (keep polling)
   CRASHED    (diverged/died -- STOP and report, do NOT auto-fix physics)
   FINISHED_NOT_CONVERGED  (ran out of NUM_ITS -- STOP and report)
   NO_RUN     (no run has started here yet)

HOW CMFGEN SIGNALS COMPLETION  (verified in source, 18jun25 release):
  * cmfgen_sub.f:4718-4720 -- when the max %% correction MAXCH < EPS  (EPS = VADAT
    keyword EPS_TERM, default 0.1 %%), the driver forces NUM_ITS_TO_DO=1, i.e. it
    does ONE final data-writing iteration and stops.  So CMFGEN EARLY-EXITS on
    convergence; it does not necessarily burn all NUM_ITS.
  * cmfgen_sub.f:4370-4375 -- if MAXCH > MAX_CHNG_LIM (VADAT MAX_CHNG, default
    1e100) it prints "bad initial population guesses" and STOPs = divergence.
    In practice the SN cold-start death is a J blow-up: comp_j_blank.f prints
    "Mean intensity blowing up" and STOPs.
  * The final (LST_ITERATION) block writes SN_HYDRO_FOR_NEXT_MODEL
    (cmfgen_sub.f:4519-4521), CUR_MODEL_DATA, RVTJ, the per-species POP* files
    (stamped 'Completion of Model:') and the <ion>OUT departure-coef files.
    A FRESH SN_HYDRO_FOR_NEXT_MODEL is therefore the reliable "the model finished
    and wrote next-epoch data" marker.
  * The per-iteration correction is printed to OUTGEN by solveba_v13.f:198-199:
        Maximum % increase at depth  NN is  X.XXE+YY  (TYPE)   --- iteration  K
        Maximum % decrease at depth  NN is  X.XXE+YY  (TYPE)   --- iteration  K
    MAXCH (the scalar the EPS test uses) = the "increase" value (solveba_v13.f:207).

CONVERGENCE TEST ADOPTED (documented, robust, conservative):
  CONVERGED  iff  ALL of:
     (1) the run has finished  (no live cmfgen_dev.exe in this dir AND
         batch.log carries CMFGEN_EXIT=0), and
     (2) SN_HYDRO_FOR_NEXT_MODEL exists and is newer than the run start, and
     (3) no death marker in OUTGEN/batch.log, and
     (4) the max %% correction (max of increase & decrease) is < CONV_PCT for the
         LAST TWO consecutive iterations recorded in OUTGEN.
  CONV_PCT default = 1.0 %% (env CMF_CONV_PCT).  The EPS_TERM value that CMFGEN
  itself used is also read from VADAT and reported; if the run early-exited,
  the final MAXCH is < EPS_TERM by construction.
Any ambiguous state resolves to a STOP-and-report status, never to "converged".
"""

import os, re, sys, glob, json, time

# ------------------------------------------------------------------ helpers
ITER_RE = re.compile(
    r"Maximum % (increase|decrease) at depth\s+\d+\s+is\s+([0-9.]+E[+-]?\d+)"
    r".*?---\s*iteration\s+(\d+)", re.IGNORECASE)

DEATH_MARKERS = [
    "blowing up",                      # comp_j_blank.f J blow-up
    "bad initial population guesses",  # cmfgen_sub.f:4371 MAX_CHNG death
    "cannot be TRUE at the same time", # USE_J_REL/USE_DJDT_RTE mis-config
    "is not recognized",               # DC_METH etc. bad keyword
    "Segmentation", "forrtl:", "Backtrace for this error",
    "Insufficient", "Unable to allocate", "STOP -",
]


def _read_tail(path, maxbytes=400_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            n = f.tell()
            f.seek(max(0, n - maxbytes))
            return f.read().decode("latin-1", "replace")
    except OSError:
        return ""


def running_pid(epoch_dir):
    """Return PID of a live cmfgen_dev.exe whose cwd is epoch_dir, else None.
    Read-only; never signals anything."""
    epoch_dir = os.path.realpath(epoch_dir)
    for pdir in glob.glob("/proc/[0-9]*"):
        try:
            with open(os.path.join(pdir, "comm")) as f:
                if "cmfgen" not in f.read():
                    continue
            cwd = os.path.realpath(os.path.join(pdir, "cwd"))
            if cwd == epoch_dir:
                return int(os.path.basename(pdir))
        except (OSError, ValueError):
            continue
    return None


def parse_iterations(outgen):
    """-> ordered list of dicts {iter, inc, dec, maxch} from OUTGEN."""
    txt = _read_tail(outgen, 4_000_000)
    per = {}
    for kind, val, it in ITER_RE.findall(txt):
        it = int(it); val = float(val)
        per.setdefault(it, {})[kind.lower()] = val
    out = []
    for it in sorted(per):
        inc = per[it].get("increase", float("nan"))
        dec = per[it].get("decrease", float("nan"))
        vals = [v for v in (inc, dec) if v == v]
        out.append({"iter": it, "inc": inc, "dec": dec,
                    "maxch": max(vals) if vals else float("nan")})
    return out


def vadat_value(epoch_dir, key, cast=float, default=None):
    path = os.path.join(epoch_dir, "VADAT")
    try:
        with open(path) as f:
            for ln in f:
                if "[" + key + "]" in ln:
                    tok = ln.strip().split()[0].replace("D", "E").replace("d", "e")
                    try:
                        return cast(tok)
                    except ValueError:
                        return tok
    except OSError:
        pass
    return default


def in_its_num(epoch_dir):
    path = os.path.join(epoch_dir, "IN_ITS")
    try:
        with open(path) as f:
            for ln in f:
                if "[NUM_ITS]" in ln:
                    return int(ln.strip().split()[0])
    except OSError:
        pass
    return None


def correction_sum_clean(epoch_dir, pct_cols=3):
    """CORRECTION_SUM: per-depth counts of variables changing > 100/10/1/.../%.
    Returns total count of variables changing by more than the pct_cols-th
    threshold (default 1%%) summed over all depths; 0 => fully relaxed."""
    path = os.path.join(epoch_dir, "CORRECTION_SUM")
    total = None
    try:
        with open(path) as f:
            total = 0
            for ln in f:
                parts = ln.split()
                if len(parts) >= 8 and parts[0].isdigit():
                    # depth  100%  10%  1%  0.1% ... (counts of vars ABOVE each)
                    total += sum(int(x) for x in parts[1:1 + pct_cols])
    except (OSError, ValueError):
        return None
    return total


# ------------------------------------------------------------------ main API
def classify(epoch_dir, conv_pct=None, run_start=0.0):
    epoch_dir = os.path.abspath(epoch_dir)
    if conv_pct is None:
        conv_pct = float(os.environ.get("CMF_CONV_PCT", "1.0"))
    eps_term = vadat_value(epoch_dir, "EPS_TERM", float, 0.1)
    info = {"dir": epoch_dir, "conv_pct": conv_pct, "eps_term": eps_term}

    outgen = os.path.join(epoch_dir, "OUTGEN")
    batch = os.path.join(epoch_dir, "batch.log")
    snhydro = os.path.join(epoch_dir, "SN_HYDRO_FOR_NEXT_MODEL")

    if not os.path.exists(outgen) and not os.path.exists(batch):
        info["status"] = "NO_RUN"
        return info

    pid = running_pid(epoch_dir)
    info["pid"] = pid

    iters = parse_iterations(outgen)
    info["n_iters"] = len(iters)
    info["last_iters"] = iters[-4:]
    info["num_its"] = in_its_num(epoch_dir)
    info["corr_sum_gt1pct"] = correction_sum_clean(epoch_dir)

    # batch exit code
    exit_code = None
    btxt = _read_tail(batch, 20000)
    m = re.search(r"CMFGEN_EXIT=(-?\d+)", btxt)
    if m:
        exit_code = int(m.group(1))
    info["exit_code"] = exit_code

    # death markers
    logtxt = _read_tail(outgen) + "\n" + _read_tail(batch)
    deaths = [d for d in DEATH_MARKERS if d.lower() in logtxt.lower()]
    info["death_markers"] = deaths

    snhydro_fresh = (os.path.exists(snhydro)
                     and os.path.getmtime(snhydro) >= run_start - 1)
    info["sn_hydro_for_next_fresh"] = snhydro_fresh

    # ---- decision tree -------------------------------------------------
    if pid is not None:
        info["status"] = "RUNNING"
        return info

    # process not alive.  Did it finish writing final outputs?
    finished = (exit_code is not None) or snhydro_fresh

    if deaths and not snhydro_fresh:
        info["status"] = "CRASHED"
        return info

    if not finished:
        # no live process, no exit stamp, no final outputs -> either just about
        # to write, or killed externally.  Treat as RUNNING unless OUTGEN is stale.
        try:
            stale = (time.time() - os.path.getmtime(outgen)) > 1800
        except OSError:
            stale = True
        info["status"] = "CRASHED" if stale else "RUNNING"
        return info

    # finished and wrote outputs -- converged or exhausted?
    last2 = [it["maxch"] for it in iters[-2:] if it["maxch"] == it["maxch"]]
    converged_metric = (len(last2) >= 1 and all(v < conv_pct for v in last2)
                        and (len(iters) >= 2))
    corr_ok = (info["corr_sum_gt1pct"] in (0, None))
    info["converged_metric"] = converged_metric

    if snhydro_fresh and converged_metric and corr_ok and not deaths:
        info["status"] = "CONVERGED"
    else:
        info["status"] = "FINISHED_NOT_CONVERGED"
    return info


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Classify a CMFGEN run directory.")
    ap.add_argument("dir")
    ap.add_argument("--conv-pct", type=float, default=None)
    ap.add_argument("--run-start", type=float, default=0.0,
                    help="epoch seconds; SN_HYDRO_FOR_NEXT_MODEL must be newer")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    info = classify(a.dir, a.conv_pct, a.run_start)
    if a.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print(info["status"])
        li = info.get("last_iters", [])
        if li:
            s = ", ".join(f"it{d['iter']}:{d['maxch']:.2e}%" for d in li)
            print("  last corrections:", s)
        if info.get("death_markers"):
            print("  DEATH:", "; ".join(info["death_markers"]))
        print(f"  eps_term={info.get('eps_term')}  conv_pct={info['conv_pct']}"
              f"  sn_hydro_next_fresh={info.get('sn_hydro_for_next_fresh')}"
              f"  corr>1%={info.get('corr_sum_gt1pct')}")
    # exit code: 0 CONVERGED, 2 RUNNING, 3 CRASHED, 4 FINISHED_NOT_CONVERGED, 5 NO_RUN
    sys.exit({"CONVERGED": 0, "RUNNING": 2, "CRASHED": 3,
              "FINISHED_NOT_CONVERGED": 4, "NO_RUN": 5}.get(info["status"], 1))


if __name__ == "__main__":
    main()
