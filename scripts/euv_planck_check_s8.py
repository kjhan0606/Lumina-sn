#!/usr/bin/env python3
"""euv_planck_check_s8.py -- OFFLINE super-Planckian audit at photosphere shell s8.

Unit-consistent Planck check. All four fields are erg/s/cm^2/Hz/sr:
  * mc_J, cs_J  : lumina_coevolve_field.csv  (cs.J / nlte_Jmc, both via
                  nlte_normalize_j_nu -> J_nu[b]=raw/(4pi V t dnu), plasma.c:173-174)
  * CMFGEN_J    : data/cmfgen_jtable_toy06_19p48d.bin (EDDFACTOR J_nu, cgs, same grid)
  * B_nu(T_e)   : planck_bnu(T_e, nu) cgs (plasma.c:1078)

Grid: NLTE_NU_MIN=1.5e14, NLTE_NU_MAX=3.0e16, NFB=1000, bin center nu=nu_min*exp((b+.5)*dln).
Prints a band table at s8 (T_e=11247.19 K).
"""
import csv, struct, sys
import numpy as np

RUN = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "logs/coevolve_consume_a10_kx_euv461"
FIELD = f"{RUN}/lumina_coevolve_field.csv"
JTAB  = "data/cmfgen_jtable_toy06_19p48d.bin"
S8 = 8
# T_e(s8) read per-run from plasma_state.csv (shell_id,W,T_rad,n_e,T_e); FB_COOL_KT etc. shift it.
def _read_te_s8(run):
    try:
        for row in csv.DictReader(open(f"{run}/lumina_plasma_state.csv")):
            if int(row["shell_id"]) == S8:
                return float(row["T_e"])
    except Exception:
        pass
    return 11247.193283
TE = _read_te_s8(RUN)

H = 6.62606957e-27
K = 1.3806488e-16
C = 2.99792458e10
CLAM = 2.99792458e18       # A*Hz

NU_MIN, NU_MAX, NFB = 1.5e14, 3.0e16, 1000
dln = np.log(NU_MAX/NU_MIN)/NFB
bctr = np.arange(NFB)
nu_ctr = NU_MIN*np.exp((bctr+0.5)*dln)
lam_ctr = CLAM/nu_ctr

def bnu(T, nu):
    x = H*nu/(K*T)
    out = np.where(x>500.0, 0.0, (2*H*nu**3/C**2)/(np.expm1(np.minimum(x,500.0))))
    return out

Bnu_s8 = bnu(TE, nu_ctr)

# --- read field CSV for s8 ---
mc = np.zeros(NFB); cs = np.zeros(NFB); lam = np.zeros(NFB); have=np.zeros(NFB,bool)
for r in csv.DictReader(open(FIELD)):
    if int(r['shell'])!=S8: continue
    b=int(r['bin']); cs[b]=float(r['cs_J']); mc[b]=float(r['mc_J']); lam[b]=float(r['wavelength_A']); have[b]=True
assert have.all(), f"missing bins: {(~have).sum()}"
# sanity: CSV wavelength must match analytic grid
assert np.allclose(lam, lam_ctr, rtol=1e-3), f"grid mismatch max={np.abs(lam-lam_ctr).max()}"

# --- read CMFGEN jtable ---
with open(JTAB,'rb') as f:
    magic,ver,nsh,nfb = struct.unpack('<4i', f.read(16))
    assert magic==0x4A544142 and nsh==50 and nfb==NFB, (hex(magic),nsh,nfb)
    tab = np.fromfile(f, dtype='<f8').reshape(nsh,nfb)
cmf = tab[S8]        # 0.0 in bins CMFGEN doesn't cover

def gmean(x):
    x = x[x>0]
    return float(np.exp(np.mean(np.log(x)))) if x.size else 0.0

BANDS = [("EUV 404-461A",404,461),("EUV 461-520A",461,520),("EUV 520-620A",520,620),
         ("opt 4800-5200A",4800,5200),("IR 9800-10200A",9800,10200)]

print(f"# s8  T_e={TE:.1f} K  W=0.0389  n_e=7.80e8   (all fields erg/s/cm^2/Hz/sr)")
print(f"{'band':16s} {'nbin':>4s} {'ncmf':>4s} | {'mc_J':>10s} {'cs_J':>10s} {'CMFGEN_J':>10s} {'B_nu(Te)':>10s} | "
      f"{'mc/Bnu':>9s} {'cs/Bnu':>9s} {'CMF/Bnu':>9s} {'mc/CMF':>8s} {'mc/cs':>8s}")
for name,lo,hi in BANDS:
    m = (lam_ctr>=lo)&(lam_ctr<=hi)
    nb = int(m.sum())
    ncmf = int((cmf[m]>0).sum())
    mc_b, cs_b, cmf_b, B_b = gmean(mc[m]), gmean(cs[m]), gmean(cmf[m]), gmean(Bnu_s8[m])
    def rr(a,b): return a/b if b>0 else float('nan')
    print(f"{name:16s} {nb:4d} {ncmf:4d} | {mc_b:10.3e} {cs_b:10.3e} {cmf_b:10.3e} {B_b:10.3e} | "
          f"{rr(mc_b,B_b):9.2e} {rr(cs_b,B_b):9.2e} {rr(cmf_b,B_b):9.2e} {rr(mc_b,cmf_b):8.2e} {rr(mc_b,cs_b):8.2e}")

# integrated EUV photon check: fraction of band J that is super-Planckian
print("\n# EUV 404-520A detail (per-band sums of J_nu*dnu, ~ energy density proxy):")
for name,lo,hi in [("404-520A",404,520)]:
    m=(lam_ctr>=lo)&(lam_ctr<=hi)
    dnu = nu_ctr[m]*dln
    print(f"  int mc_J dnu={np.sum(mc[m]*dnu):.3e}  int cs_J dnu={np.sum(cs[m]*dnu):.3e}  "
          f"int CMF dnu={np.sum(cmf[m]*dnu):.3e}  int B_nu dnu={np.sum(Bnu_s8[m]*dnu):.3e}")
    print(f"  ratio int(mc)/int(Bnu)={np.sum(mc[m]*dnu)/np.sum(Bnu_s8[m]*dnu):.3e}  "
          f"int(mc)/int(CMF)={np.sum(mc[m]*dnu)/max(np.sum(cmf[m]*dnu),1e-99):.3e}")
