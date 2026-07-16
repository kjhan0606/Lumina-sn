#!/usr/bin/env python3
"""Reader for LUMINA-SN gated packet event logs (lumina_events.bin).

Binary layout (little-endian):
  lumina_events.bin       : 32-byte header {magic "LUMEVT01"[8], u32 record_size,
                            20 reserved bytes} then N * record_size EventRec.
  EventRec (record_size=20): u32 pkt_id, i32 line_id, f32 nu_comov, f32 energy,
                            u8 etype, u8 shell, u8 iter, u8 pad.
  lumina_events_lines.bin : magic "LUMLIN01"[8] then per line
                            {f32 lambda_A, u16 Z, u16 ion}.
"""
import sys
import argparse
import numpy as np

EVENT_DTYPE = np.dtype([
    ("pkt_id",   "<u4"), ("line_id", "<i4"),
    ("nu_comov", "<f4"), ("energy",  "<f4"),
    ("etype",    "u1"),  ("shell",   "u1"),
    ("iter",     "u1"),  ("pad",     "u1"),
])  # itemsize == 20
LINE_DTYPE = np.dtype([("lam", "<f4"), ("Z", "<u2"), ("ion", "<u2")])  # 8 B
C_CM_S = 2.99792458e10
ETYPE_NAME = {1: "line-abs", 2: "line-emit", 3: "bf-abs", 4: "kpkt-ff",
              5: "kpkt-fb", 6: "escape", 7: "e-scatter", 8: "bf-reemit"}


def load_events(path):
    """Return a structured numpy array of EventRec (header stripped)."""
    with open(path, "rb") as f:
        head = f.read(32)
        if head[:8] != b"LUMEVT01":
            raise ValueError("bad magic in %s: %r" % (path, head[:8]))
        rsz = int(np.frombuffer(head[8:12], dtype="<u4")[0])
        if rsz != EVENT_DTYPE.itemsize:
            raise ValueError("record_size %d != expected %d" % (rsz, EVENT_DTYPE.itemsize))
        return np.frombuffer(f.read(), dtype=EVENT_DTYPE)


def load_lines(path):
    """Return (lambda_A, Z, ion) arrays indexed by line id."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"LUMLIN01":
            raise ValueError("bad magic in %s: %r" % (path, magic))
        rec = np.frombuffer(f.read(), dtype=LINE_DTYPE)
    return rec["lam"], rec["Z"].astype(np.int32), rec["ion"].astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description="summarize a LUMINA event log")
    ap.add_argument("events")
    ap.add_argument("lines", nargs="?", default=None)
    ap.add_argument("--etype", type=int, default=None, help="filter to one etype")
    ap.add_argument("--lambda-max", type=float, default=None, help="keep lambda < this (A)")
    args = ap.parse_args()

    ev = load_events(args.events)
    print("loaded %d events from %s" % (len(ev), args.events))
    if len(ev) == 0:
        return
    lam_ev = np.where(ev["nu_comov"] > 0, C_CM_S / ev["nu_comov"] * 1e8, 0.0)

    sel = np.ones(len(ev), dtype=bool)
    if args.lambda_max is not None:
        sel &= (lam_ev > 0) & (lam_ev < args.lambda_max)
    if args.etype is not None:
        sel &= ev["etype"] == args.etype
    ev_s, lam_s = ev[sel], lam_ev[sel]
    print("after filters: %d events" % len(ev_s))

    print("\ncounts by etype (energy-weighted):")
    print("  %-10s %12s %14s" % ("etype", "count", "sum(energy)"))
    for et in sorted(np.unique(ev_s["etype"])):
        m = ev_s["etype"] == et
        print("  %-10s %12d %14.4e"
              % (ETYPE_NAME.get(int(et), str(et)), int(m.sum()), ev_s["energy"][m].sum()))

    linem = ev_s["line_id"] >= 0
    if linem.any() and args.lines:
        lam_A, Z, ion = load_lines(args.lines)
        lid = ev_s["line_id"][linem]
        en = ev_s["energy"][linem]
        keys = {}
        for k, e in zip(lid, en):
            z, i = int(Z[k]), int(ion[k])
            slot = keys.setdefault((z, i, round(float(lam_A[k]), 1)), 0.0)
            keys[(z, i, round(float(lam_A[k]), 1))] = slot + float(e)
        top = sorted(keys.items(), key=lambda kv: -kv[1])[:20]
        print("\ntop-20 line events by total energy (Z, ion, line lambda_A):")
        print("  %-4s %-4s %12s %14s" % ("Z", "ion", "lambda_A", "sum(energy)"))
        for (z, i, lm), e in top:
            print("  %-4d %-4d %12.1f %14.4e" % (z, i, lm, e))
    elif linem.any():
        print("\n(top-20 line table skipped: pass the lines file to enable it)")

    print("\nper-shell histogram (count, sum energy):")
    for sh in sorted(np.unique(ev_s["shell"])):
        m = ev_s["shell"] == sh
        print("  shell %3d : %12d  %14.4e" % (int(sh), int(m.sum()), ev_s["energy"][m].sum()))


if __name__ == "__main__":
    main()
