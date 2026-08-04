#!/usr/bin/env python3
"""CMFGEN reference vs LUMINA emergent spectrum, DDC15 0.976d.
The pure-CMFGEN path emits NO spectrum (MC transport bypassed), so the
LUMINA spectrum is the MC blanket run 164032 (the run whose T_e(v) the
pure-CMFGEN module reproduces). Absolute units differ (CMFGEN F_lambda
~1e-4, LUMINA luminosity ~1e43), so each is normalized to unit integral
over the 3000-9000 A overlap to compare SPECTRAL SHAPE."""
import numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF  = f"{ROOT}/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"
MC   = f"{ROOT}/logs/ddc15_radeq_blanket_164032/lumina_spectrum.csv"
MCF  = f"{ROOT}/logs/ddc15_radeq_blanket_164032/lumina_spectrum_formal.csv"

ref = np.loadtxt(REF)
rw, rf = ref[:, 0], ref[:, 1]

def rd_csv(p):
    w, f = [], []
    r = csv.reader(open(p)); next(r)
    for row in r:
        w.append(float(row[0])); f.append(float(row[1]))
    return np.array(w), np.array(f)

mw, mf = rd_csv(MC)
fw, ff = rd_csv(MCF)

LO, HI = 3000.0, 9000.0
def norm(w, f):
    m = (w >= LO) & (w <= HI) & np.isfinite(f)
    area = np.trapz(f[m], w[m])
    return f / area if area > 0 else f

rfn, mfn, ffn = norm(rw, rf), norm(mw, mf), norm(fw, ff)

fig, ax = plt.subplots(1, 1, figsize=(13, 6.5))
ax.plot(rw, rfn, "k", lw=1.6, label="CMFGEN reference (DDC15 0.976d)")
ax.plot(mw, mfn, "#3898EC", lw=1.1, alpha=0.85, label="LUMINA MC (blanket 164032)")
ax.plot(fw, ffn, "#D97757", lw=1.0, alpha=0.7, label="LUMINA MC formal")
ax.set_xlim(2500, 10000)
ax.set_xlabel("wavelength [Angstrom]")
ax.set_ylabel("normalized flux (unit integral 3000-9000 A)")
ax.set_title("DDC15 0.976d: CMFGEN reference vs LUMINA emergent spectrum (shape)",
             fontsize=13, weight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)

# peak labels
rp = rw[np.argmax(np.where((rw >= LO) & (rw <= HI), rfn, 0))]
mp = mw[np.argmax(np.where((mw >= LO) & (mw <= HI), mfn, 0))]
ax.annotate(f"CMFGEN peak {rp:.0f}A", xy=(rp, rfn.max()),
            xytext=(rp+800, rfn.max()*0.95), fontsize=9)

out = f"{ROOT}/figures/2026-06-07_ddc15_cmfgen_vs_lumina_spectrum.png"
plt.tight_layout(); plt.savefig(out, dpi=130); print("saved", out)
print(f"CMFGEN peak={rp:.0f}A  LUMINA-MC peak={mp:.0f}A")
