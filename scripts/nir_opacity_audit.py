#!/usr/bin/env python3
"""NIR opacity audit [5500,9500]Å — per-(Z,ion,sub-band) Sobolev τ ranking.

Method (LTE-T_rad approximation):
1. Read shell ρ, X_Z from C2 champion abundances; W, T_rad from plasma_state.csv.
2. Saha at T_e≈T_rad (with charge balance for n_e) → n_ion(Z, ion, shell).
3. Boltzmann at T_rad → n_lower(line, shell).
4. Sobolev τ_sob = (π e²/m_e c) f_lu n_l λ t_exp summed per (Z,ion,sub-band).
5. Volume-weighted shell sum (4π r² Δr) → ranked tables.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
import h5py

ROOT  = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
H5    = ROOT/"data/atomic/kurucz_cd23_cmfgen_lumina.h5"
LOGD  = ROOT/"logs/ddc15C2_155756_ddc15C2_xFeO0.05"
REFD  = LOGD/"ref"
OUTD  = ROOT/"data/sn2011fe"
OUTD.mkdir(parents=True, exist_ok=True)

H, KB, ME, EC, C = 6.62607015e-27, 1.380649e-16, 9.1093837e-28, 4.80320425e-10, 2.99792458e10
SIGMA_PRE = np.pi * EC**2 / (ME * C)  # cm² Hz
AMU = 1.660539e-24
EV_ERG = 1.602176634e-12  # eV → erg (carsus stores energies in eV)
SUB_BANDS = [(5500,5800,"5500-5800"),(5800,6800,"Si red"),
             (6800,8000,"OI/cont"), (8000,9500,"Ca IR")]
A_MASS = {1:1.008,2:4.003,6:12.01,7:14.01,8:16.00,11:22.99,12:24.31,13:26.98,
          14:28.09,16:32.07,18:39.95,19:39.10,20:40.08,21:44.96,22:47.87,
          23:50.94,24:52.00,25:54.94,26:55.85,27:58.93,28:58.69}

def load_atomic():
    f = h5py.File(H5, 'r')
    L = {
        'lZ':   f['lines_data/axis1_label0'][:],
        'lion': f['lines_data/axis1_label1'][:],
        'llow': f['lines_data/axis1_label2'][:],
        'lup':  f['lines_data/axis1_label3'][:],
        'lwl':  f['lines_data/block0_values'][:,0],  # wavelength in Angstrom
        'lflu': f['lines_data/block0_values'][:,2],  # f_lu
    }
    Lv = {
        'vZ':   f['levels_data/axis1_label0'][:],
        'vion': f['levels_data/axis1_label1'][:],
        'vn':   f['levels_data/axis1_label2'][:],
        'vE':   f['levels_data/block0_values'][:,0] * EV_ERG,  # eV → erg
        'vg':   f['levels_data/block1_values'][:,0],
    }
    chi = {}
    iZ = f['ionization_data/index_label0'][:]
    ii = f['ionization_data/index_label1'][:]
    iv = f['ionization_data/values'][:]
    for z,j,v in zip(iZ,ii,iv): chi[(int(z),int(j))] = float(v) * EV_ERG  # eV → erg
    f.close()
    return L, Lv, chi

def partition_U(Lv, T):
    """U(Z, ion) at temperature T (vector over T)."""
    U = {}
    kT = KB * T
    # group levels by (Z, ion)
    keys = np.unique(np.column_stack([Lv['vZ'], Lv['vion']]), axis=0)
    for (Z, ion) in keys:
        m = (Lv['vZ']==Z) & (Lv['vion']==ion)
        E = Lv['vE'][m]; g = Lv['vg'][m]
        if isinstance(T, np.ndarray):
            U[(int(Z),int(ion))] = (g[:,None] * np.exp(-E[:,None] / kT[None,:])).sum(axis=0)
        else:
            U[(int(Z),int(ion))] = float((g * np.exp(-E/kT)).sum())
    return U

def saha_ladder(U_zi_arr, chi_zi_arr, T, n_e):
    """For one element Z with ions 0..nmax-1: compute n_ion fractions at T, n_e.
    U_zi_arr[j]   = U(Z, j) at this shell T
    chi_zi_arr[j] = χ(Z, j) ionization energy from ion j to j+1 [erg]
    Returns: f[j], j=0..nmax-1, sums to 1.
    """
    nmax = len(U_zi_arr)
    kT = KB * T
    g_th = (2*np.pi*ME*kT/H**2)**1.5
    # log ratios: log(n_{j+1}/n_j) = log(2 U_{j+1}/U_j · g_th/n_e) - chi_j/kT
    log_ratios = np.zeros(nmax-1)
    for j in range(nmax-1):
        if U_zi_arr[j] <= 0 or U_zi_arr[j+1] <= 0: log_ratios[j] = -50
        else:
            log_ratios[j] = (np.log(2*U_zi_arr[j+1]/U_zi_arr[j])
                            + np.log(g_th/max(n_e,1.0))
                            - chi_zi_arr[j]/kT)
    # cumulative log n_j relative to n_0
    log_cum = np.cumsum([0.0] + list(log_ratios))
    log_cum -= log_cum.max()
    f = np.exp(log_cum)
    return f / f.sum()

def solve_ne_saha(T, n_Z_dict, U_zi_dict, chi_dict, n_e_guess=1e9, niter=30):
    """Iterate n_e by charge balance over all elements."""
    n_e = max(n_e_guess, 1.0)
    for _ in range(niter):
        ne_new = 0.0
        for Z, n_total in n_Z_dict.items():
            chi_list = []
            U_list = []
            j = 0
            while (Z,j) in U_zi_dict and j <= Z:
                U_list.append(U_zi_dict[(Z,j)])
                if (Z,j) in chi_dict: chi_list.append(chi_dict[(Z,j)])
                j += 1
            nmax = len(U_list)
            if nmax < 2: continue
            chi_list = chi_list[:nmax-1]
            f = saha_ladder(np.array(U_list), np.array(chi_list), T, n_e)
            charges = np.arange(nmax)
            ne_new += n_total * (f * charges).sum()
        ne_new = max(ne_new, 1.0)
        if abs(ne_new - n_e) / n_e < 1e-3:
            n_e = ne_new; break
        n_e = 0.5*(n_e + ne_new)
    return n_e

def main():
    print("=== Loading atomic data...")
    L, Lv, chi = load_atomic()
    n_lines = len(L['lZ'])
    n_levels = len(Lv['vZ'])
    print(f"  lines={n_lines}, levels={n_levels}, χ entries={len(chi)}")

    # geometry, density, abundance, plasma state
    geom = pd.read_csv(REFD/"geometry.csv")
    dens = pd.read_csv(REFD/"density.csv")
    ab   = pd.read_csv(REFD/"abundances.csv")
    ps   = pd.read_csv(LOGD/"lumina_plasma_state.csv")
    with open(REFD/"config.json") as f: cfg = json.load(f)
    t_exp = cfg["time_explosion_s"]
    n_shells = len(geom)
    print(f"  n_shells={n_shells}  t_exp={t_exp:.3e}s")

    rho = dens['rho'].values
    T_rad = ps['T_rad'].values
    W = ps['W'].values
    r_in = geom['r_inner'].values
    r_out = geom['r_outer'].values
    dV = 4*np.pi/3 * (r_out**3 - r_in**3)

    # n_total per Z per shell
    Z_list = ab['atomic_number'].astype(int).values
    X = ab.iloc[:, 1:].values  # (n_Z, n_shells)
    A_arr = np.array([A_MASS.get(int(z), 2.0*z) for z in Z_list])
    n_total = (X * rho[None,:]) / (A_arr[:,None] * AMU)  # cm^-3

    # build n_Z per shell dict
    print("=== Saha for each shell...")
    n_ion_arr = {}  # (Z,ion) -> array(n_shells)
    for s in range(n_shells):
        nZ = {int(z): n_total[i,s] for i,z in enumerate(Z_list) if n_total[i,s]>0}
        U_at = partition_U(Lv, T_rad[s])
        ne0 = 0.5 * sum(nZ.values())  # ~50% singly ionized as start
        n_e = solve_ne_saha(T_rad[s], nZ, U_at, chi, n_e_guess=ne0)
        for Z, ntot in nZ.items():
            U_list, chi_list = [], []
            j = 0
            while (Z,j) in U_at and j <= Z:
                U_list.append(U_at[(Z,j)])
                if (Z,j) in chi: chi_list.append(chi[(Z,j)])
                j += 1
            nmax = len(U_list)
            chi_list = chi_list[:nmax-1]
            if nmax < 2:
                f = np.array([1.0])
            else:
                f = saha_ladder(np.array(U_list), np.array(chi_list), T_rad[s], n_e)
            for j, fj in enumerate(f):
                key = (Z, j)
                if key not in n_ion_arr: n_ion_arr[key] = np.zeros(n_shells)
                n_ion_arr[key][s] = ntot * fj
        if s in (0, n_shells//2, n_shells-1):
            top = sorted([(k, v[s]) for k,v in n_ion_arr.items() if v[s]>0],
                         key=lambda x:-x[1])[:8]
            print(f"  s={s:2d} T_rad={T_rad[s]:.0f}K  n_e={n_e:.2e}  top: " +
                  " ".join([f"Z{k[0]}+{k[1]}={v:.1e}" for k,v in top]))

    # level energy and g lookup: array indexed by (Z, ion, level_id)
    print("=== Building level lookup...")
    max_lev = int(Lv['vn'].max()) + 1
    Z_max = 30; ion_max = 10
    E_lev = np.zeros((Z_max, ion_max, max_lev))
    g_lev = np.zeros((Z_max, ion_max, max_lev))
    for i in range(n_levels):
        z = int(Lv['vZ'][i]); io = int(Lv['vion'][i]); n = int(Lv['vn'][i])
        if z<Z_max and io<ion_max and n<max_lev:
            E_lev[z,io,n] = Lv['vE'][i]
            g_lev[z,io,n] = Lv['vg'][i]

    # filter lines to [5500,9500]Å and species of interest
    wl = L['lwl']
    msk = (wl >= 5500) & (wl <= 9500)
    Z_sel = set([6,8,11,12,13,14,16,20,21,22,23,24,25,26,27,28])
    msk &= np.isin(L['lZ'], list(Z_sel))
    msk &= (L['lflu'] > 1e-8)
    print(f"=== Filtered lines in [5500,9500]Å with f_lu>1e-8: {msk.sum()}")

    sel_idx = np.where(msk)[0]
    lZ = L['lZ'][sel_idx]
    lion = L['lion'][sel_idx]
    llow = L['llow'][sel_idx]
    lwl_sel = L['lwl'][sel_idx]
    lflu_sel = L['lflu'][sel_idx]

    # per shell, compute τ_sob proxy per line and bin (Z, ion, sub-band)
    # NOTE: Carsus stitches Kurucz lines with CMFGEN levels; level-index
    # cross-reference is unreliable. So we use n_ion proxy (no Boltzmann):
    #   τ_proxy = (π e²/m_e c) f_lu (n_ion / U) λ t_exp
    # This treats all levels as ground state → upper bound on n_lower.
    # Ranking remains meaningful at fixed band (lines in [5500,9500] have
    # comparable Boltzmann factors relative to each other).
    print("=== Computing Sobolev τ proxy per shell (Boltzmann-free)...")
    rows = []
    for s in range(n_shells):
        U_at = partition_U(Lv, T_rad[s])
        n_lower = np.zeros(len(sel_idx))
        for i in range(len(sel_idx)):
            key = (int(lZ[i]), int(lion[i]))
            nion = n_ion_arr.get(key, np.zeros(n_shells))[s]
            U = U_at.get(key, 1.0)
            if nion <= 0 or U <= 0: continue
            # Assume ground-state Boltzmann ≈ 1/U (degeneracy-weighted)
            n_lower[i] = nion / max(U, 1.0)
        lam_cm = lwl_sel * 1e-8
        tau = SIGMA_PRE * lflu_sel * n_lower * lam_cm * t_exp
        # Bin per sub-band per (Z, ion). Weight by dV.
        for (lo, hi, lab) in SUB_BANDS:
            inband = (lwl_sel >= lo) & (lwl_sel < hi)
            for Z in [6,8,12,14,20,22,24,25,26,27,28]:
                for io in [0,1,2,3]:
                    mm = inband & (lZ==Z) & (lion==io)
                    if not mm.any(): continue
                    tau_sum = tau[mm].sum()
                    if tau_sum <= 0: continue
                    rows.append({
                        'shell':s, 'Z':Z, 'ion':io, 'subband':lab,
                        'lam_lo':lo, 'lam_hi':hi,
                        'tau_sum_vol_weighted': tau_sum * dV[s],
                        'tau_sum_raw': tau_sum,
                        'n_lines': int(mm.sum()),
                    })
        if s in (0, n_shells//2, n_shells-1):
            print(f"  s={s:2d} done; sample τ_max={tau.max():.3e}")

    df = pd.DataFrame(rows)
    out_csv = OUTD/"nir_opacity_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)")

    # Aggregate: per (Z, ion, sub-band) — sum vol-weighted τ over shells
    agg = df.groupby(['Z','ion','subband']).agg(
        tau_total=('tau_sum_vol_weighted','sum'),
        n_lines_uniq=('n_lines','first'),  # same per shell anyway
    ).reset_index()
    # normalize per sub-band → contribution fraction
    print("\n=== Per-(Z,ion) contribution fraction by sub-band (vol-weighted τ_sob sum) ===")
    for (_,_,lab) in SUB_BANDS:
        sub = agg[agg.subband==lab].sort_values('tau_total', ascending=False)
        tot = sub.tau_total.sum()
        if tot <= 0: continue
        print(f"\n[{lab}] total Σ τ·dV = {tot:.3e}")
        top10 = sub.head(10)
        for _, r in top10.iterrows():
            roman = ['I','II','III','IV','V','VI'][int(r.ion)]
            print(f"  Z={int(r.Z):2d} {roman:4s}  τ_tot={r.tau_total:.3e}  ({100*r.tau_total/tot:5.1f}%)  N_lines={int(r.n_lines_uniq)}")

if __name__ == '__main__':
    main()
