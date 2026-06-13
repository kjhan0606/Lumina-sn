#!/usr/bin/env python3
"""Observer-frame formal integral (Doppler P-Cygni) from a pure-CMFGEN JDUMP.

The pure-CMFGEN cmfgen_write_spectrum integrates the COMOVING-frame binned
opacity at the same observer bin in every shell -> no Doppler shift between
shells -> no P-Cygni. Here we post-process the JDUMP (per shell,bin: nu, J,
chi_es, chi_abs, chi_line, chi_tot, S_fixed) by marching tangent rays in z and
Doppler-shifting the comoving (chi_tot, S) lookup by v_z = z/t_exp at each
segment, on a fine observer wavelength grid. Homologous: v = r/t, v_z = z/t.

Usage: formal_integral_obsframe.py <jobid> [out.csv]
"""
import sys, glob, numpy as np, pandas as pd

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
C = 2.99792458e10; H = 6.62607015e-27; KB = 1.380649e-16
T_EXP = 0.976 * 86400.0
T_INNER = 4434.0

def planck_nu(nu, T):
    x = np.clip(H*nu/(KB*T), 1e-9, 5e2)
    return 2*H*nu**3/C**2/np.expm1(x)

def main():
    jid = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/spectrum_obsframe_{jid}.csv"
    jd = pd.read_csv(glob.glob(f"{ROOT}/logs/ddc15_pc_phase3_*_{jid}/lumina_cmfgen_jnu.csv")[0])
    geo = pd.read_csv(f"{ROOT}/data/tardis_reference_ddc15_0p976d/geometry.csv")
    NS = int(jd.shell.max()) + 1
    nu_grid = jd[jd.shell == 0].sort_values("bin").nu.values            # ascending bin -> ascending nu
    NB = len(nu_grid)
    # per-shell comoving chi_tot and source S = S_fixed + (chi_es/chi_tot) J
    chi = np.zeros((NS, NB)); S = np.zeros((NS, NB))
    for s in range(NS):
        d = jd[jd.shell == s].sort_values("bin")
        ct = d.chi_tot.values.copy(); ct[ct <= 0] = 1e-99
        chi[s] = ct
        r = np.clip(d.chi_es.values/ct, 0, 1)
        S[s] = d.S_fixed.values + r*d.J.values
    r_in = geo.r_inner.values; r_out = geo.r_outer.values
    r_mid = 0.5*(r_in+r_out)
    Rmax = r_out[-1]; Rcore = r_in[0]

    # observer wavelength grid (fine), 2500-19000 A
    lam_A = np.linspace(2500.0, 19000.0, 2400)
    nu_obs = C/(lam_A*1e-8)                                              # Hz
    lnu = np.log(nu_grid)

    def chi_at(s, nu):    # log-interp comoving chi at (Doppler-shifted) nu
        return np.exp(np.interp(np.log(np.clip(nu, nu_grid[0], nu_grid[-1])), lnu, np.log(chi[s])))
    def S_at(s, nu):
        return np.interp(np.log(np.clip(nu, nu_grid[0], nu_grid[-1])), lnu, S[s])

    # impact parameters: core rays + one per shell r_mid
    n_core = 10
    p_core = Rcore*(np.arange(n_core)+0.5)/n_core
    p_ray = np.concatenate([p_core, r_mid])
    p_ray = np.sort(p_ray)

    Lnu = np.zeros_like(nu_obs)
    f_prev = np.zeros_like(nu_obs); p_prev = 0.0
    for p in p_ray:
        # z grid: shells crossed (r_mid > p), front+back symmetric
        sh = [s for s in range(NS) if r_mid[s] > p]
        if not sh:
            f_prev = np.zeros_like(nu_obs); p_prev = p; continue
        z = np.array([np.sqrt(r_mid[s]**2 - p**2) for s in sh])          # |z| per shell
        core = p < Rcore
        z_core = np.sqrt(max(Rcore**2 - p**2, 0.0)) if core else 0.0
        # build ordered segments from back (-z_max) to front (+z_max)
        zs = np.concatenate([-z[::-1], z]) if not core else np.concatenate([-z[::-1], [ -z_core, z_core], z])
        shorder = (sh[::-1] + sh) if not core else (sh[::-1] + [ -1, -1] + sh)
        # integrate I along z (back -> front)
        I = np.zeros_like(nu_obs)
        for i in range(len(zs)-1):
            za, zb = zs[i], zs[i+1]
            s = shorder[i] if i < len(shorder) else shorder[-1]
            if s == -1:    # inside the core: emit B(T_inner) at the front face
                if za < 0 <= zb or (za < z_core <= zb):
                    I = planck_nu(nu_obs, T_INNER)
                continue
            dz = zb - za
            if dz <= 0: continue
            zmid = 0.5*(za+zb)
            nu_cmf = nu_obs*(1.0 - zmid/(C*T_EXP))                       # blueshift front (z>0)
            dtau = chi_at(s, nu_cmf)*dz
            ex = np.exp(-np.clip(dtau, 0, 7e2))
            Sl = S_at(s, nu_cmf)
            I = I*ex + Sl*(1.0-ex)
        f = I*p
        Lnu += 0.5*(f_prev+f)*(p-p_prev)
        f_prev = f; p_prev = p
    Lnu *= 8.0*np.pi**2                                                  # erg/s/Hz
    L_lam = Lnu*C/(lam_A*1e-8)**2*1e-8                                   # erg/s/A
    pd.DataFrame({"wavelength_angstrom": lam_A, "flux": L_lam}).to_csv(out, index=False)
    print(out)

if __name__ == "__main__":
    main()
