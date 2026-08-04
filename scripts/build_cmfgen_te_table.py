#!/usr/bin/env python3
"""build_cmfgen_te_table.py -- F3-T TEMPERATURE-TABLE probe table builder.

Reads the PUBLISHED CMFGEN toy06 @19.48d electron-temperature profile T_e(v) and
resamples it onto LUMINA's 50 shell mid-velocities, producing a trivial CSV that the
C gate LUMINA_TE_TABLE pins as the per-shell electron temperature (WHOLE-STATE probe:
Saha/NLTE populations, emissivities, k-packet redistribution, cooling -- every T_e
consumer follows it).

Source: data/standart_data1/toy06/phys_toy06_cmfgen.txt, the "#TIME  19.480" block
        (columns: vel_mid[km/s], temp[K], rho, ne, natom). This is CMFGEN's own
        published output -- the same block the ionfrac/edep comparisons use.

Mapping:
  * velocity: CMFGEN's 100 vel_mid depths (1025..40325 km/s) -> Lumina's 50 shell mid-
    velocities (data/tardis_reference_toy06_19p48d/geometry.csv, 4264..39936 km/s);
    LINEAR interpolation in velocity (matches the design sanity targets exactly).
    Lumina shells beyond the outermost CMFGEN depth HOLD the outermost CMFGEN T_e.

Output: data/cmfgen_te_table_toy06_19p48d.csv  (shell_id,vel_mid_kms,T_e_K rows)

Sanity (design-registered): T(s0)~18760 K, T(s8)~10380 K.
"""
import sys, os, csv, argparse
import numpy as np

TARGET_TIME = 19.480


def parse_phys_block(path, t_target):
    """Return (vel_kms[], temp_K[]) for the '#TIME  <t_target>' block."""
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('#TIME:'):
            try:
                t = float(ln.split(':', 1)[1])
            except ValueError:
                i += 1; continue
            if abs(t - t_target) < 1e-6:
                # next line '#NVEL: N', then a column header, then N data rows
                nvel = int(lines[i + 1].split(':', 1)[1])
                v, T = [], []
                j = i + 3  # skip #NVEL and the '#vel_mid...' header
                while j < len(lines) and len(v) < nvel:
                    toks = lines[j].split()
                    if len(toks) >= 2 and not lines[j].startswith('#'):
                        v.append(float(toks[0])); T.append(float(toks[1]))
                    j += 1
                return np.array(v), np.array(T)
        i += 1
    raise KeyError(f"#TIME {t_target} block not found in {path}")


def lumina_shell_velocities(geom_csv):
    rows = list(csv.DictReader(open(geom_csv)))
    return np.array([(float(r['v_inner']) + float(r['v_outer'])) / 2.0 / 1e5
                     for r in rows])  # cm/s -> km/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phys', default='data/standart_data1/toy06/phys_toy06_cmfgen.txt')
    ap.add_argument('--geom', default='data/tardis_reference_toy06_19p48d/geometry.csv')
    ap.add_argument('--out',  default='data/cmfgen_te_table_toy06_19p48d.csv')
    a = ap.parse_args()

    vc, Tc = parse_phys_block(a.phys, TARGET_TIME)
    assert np.all(np.diff(vc) > 0), "CMFGEN velocity grid not strictly ascending"
    print(f"[phys] {a.phys} #TIME {TARGET_TIME}: {vc.size} depths "
          f"v={vc[0]:.0f}..{vc[-1]:.0f} km/s  T={Tc[0]:.0f}..{Tc[-1]:.0f} K")

    vmids = lumina_shell_velocities(a.geom)
    ns = vmids.size
    assert ns == 50, f"geometry has {ns} shells, expected 50"

    # LINEAR interp in velocity; HOLD outermost beyond CMFGEN max (np.interp clamps
    # to endpoints outside the range -> exactly the "hold last value" rule).
    Te = np.interp(vmids, vc, Tc)
    held = list(np.where(vmids > vc[-1])[0])
    print(f"[geom] Lumina shell mids: s0={vmids[0]:.0f} s8={vmids[8]:.0f} "
          f"s49={vmids[-1]:.0f} km/s;  held(beyond {vc[-1]:.0f}): {held}")

    print("\n=== SANITY (design targets: T(s0)~18760 K, T(s8)~10380 K) ===")
    print(f"  T(s0)={Te[0]:.1f} K   T(s8)={Te[8]:.1f} K   "
          f"T(s49)={Te[-1]:.1f} K")
    fails = []
    if not (18500 <= Te[0] <= 19000):
        fails.append(f"T(s0)={Te[0]:.1f} out of [18500,19000]")
    if not (10200 <= Te[8] <= 10600):
        fails.append(f"T(s8)={Te[8]:.1f} out of [10200,10600]")
    if fails:
        print("\n[ABORT] sanity gate FAILED:")
        for f in fails:
            print("   - " + f)
        sys.exit(1)
    print("[OK] sanity gates passed.")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        f.write("# CMFGEN toy06 @19.48d electron-temperature table for LUMINA_TE_TABLE (F3-T probe)\n")
        f.write("# columns: shell_id,vel_mid_kms,T_e_K\n")
        f.write(f"# source: {a.phys} #TIME {TARGET_TIME} (vel_mid,temp)\n")
        f.write("# linear-in-velocity interp to Lumina 50 shell mids; hold last beyond CMFGEN max\n")
        for s in range(ns):
            f.write(f"{s},{vmids[s]:.1f},{Te[s]:.2f}\n")
    print(f"\n[out] wrote {a.out} ({ns} shells, {os.path.getsize(a.out)} bytes)")


if __name__ == '__main__':
    main()
