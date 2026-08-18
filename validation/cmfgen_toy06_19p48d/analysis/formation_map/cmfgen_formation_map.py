#!/usr/bin/env python3
"""CMFGEN per-shell SPECTRUM-FORMATION map from cmf_flux ETA_DATA / CHI_DATA.

Radial contribution function (task-specified integrand):
    dL_band(r) ~ SUM_{nu in band} eta_nu(r) * exp(-tau_nu(r->out)) * r^2 * dr
  eta_nu, chi_nu : CMF total emissivity / opacity [depth x CMF-freq] from ETA_DATA/CHI_DATA
  tau_nu(r->out) : radial optical depth from r outward to the OUTER boundary,
                   trapezoidal cumulative of chi_nu along R.
CF_cmf(band, depth) = dL_band(depth) / sum_depth dL_band(depth)  (fractional per band).

APPROXIMATIONS / CAVEATS (stated):
  * RADIAL (p=0-like) escape probability exp(-tau_radial); not the full (p,z) observer
    integral cmf_flux does for the emergent spectrum -- captures the formation REGION,
    band-integrated, not the exact emergent flux.
  * Band edges are CMF-frame (lam=c/NU_cmf), matching the Lumina event-log comoving side.
    Observer-frame Doppler (v/c ~ 1-13%) is small vs band widths.
  * eta is the TOTAL CMF emissivity (thermal + e-scattering source), i.e. where a
    photon last acquired its frequency -- the CMFGEN analogue of Lumina 'last emission'.
Files: ETA_DATA/CHI_DATA + *_INFO (EDDFACTOR direct-access fmt); RVTJ for R,V.
"""
import numpy as np, os, sys

RUNDIR = sys.argv[1] if len(sys.argv)>1 else "/gpfs/kjhan/cmfgen_runs/toy06_19.48d_cmfflux"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CLIGHT_A = 2997.92458  # lam_A = CLIGHT_A / FL(1e15 Hz)
BANDS = [(300,450),(450,918),(918,1290),(1290,2000),(2000,4500),(4500,7000),(7000,1e12),(1490,1650)]
BAND_LBL = ["300-450","450-918","918-1290","1290-2000","2000-4500","4500-7000","7000+","1490-1650"]

def read_info(info):
    L=open(info).read().splitlines(); v=L[2].split()
    return dict(ND=int(v[0]), RECL=int(v[1]), WORD=int(v[2]), little=(v[5]=='T'))

def read_edd_records(fname):
    """Return raw [nrec, ND+1] float64 records (data records only, 0-based 14..), ND."""
    info=read_info(fname+'_INFO'); ND=info['ND']; nwr=info['RECL']//info['WORD']
    dt='<f8' if info['little'] else '>f8'
    raw=np.fromfile(fname,dtype=dt); n=(raw.size//nwr)*nwr
    raw=raw[:n].reshape(-1,nwr)
    finish=raw[4,0]
    data=raw[14:]                     # records 15.. (0-based 14): ND values + NU
    return data, ND, finish

def parse_rvtj(text,label,ND):
    lines=text.splitlines()
    for i,ln in enumerate(lines):
        if ln.strip()==label:
            vals=[]; j=i+1
            while j<len(lines) and len(vals)<ND:
                try: vals+=[float(t) for t in lines[j].split()]
                except ValueError: break
                j+=1
            return np.array(vals[:ND])
    raise KeyError(label)

def main():
    eta_r, ND, fe = read_edd_records(os.path.join(RUNDIR,'ETA_DATA'))
    chi_r, ND2, fc = read_edd_records(os.path.join(RUNDIR,'CHI_DATA'))
    assert ND==ND2, (ND,ND2)
    nrec=min(eta_r.shape[0], chi_r.shape[0])
    eta_r=eta_r[:nrec]; chi_r=chi_r[:nrec]
    NU_e=eta_r[:,ND]; NU_c=chi_r[:,ND]
    good = np.isfinite(eta_r).all(1) & np.isfinite(chi_r).all(1) & (NU_e>0) & (np.abs(NU_e-NU_c)<1e-6*NU_e)
    eta=eta_r[good,:ND]; chi=chi_r[good,:ND]; NU=NU_e[good]        # [nf, ND]
    lam=CLIGHT_A/NU                                               # CMF Angstrom
    print(f"[eta/chi] ND={ND} nrec={nrec} good_freqs={good.sum()} FINISH eta={fe} chi={fc}")
    print(f"[freq] lam range {lam.min():.1f} .. {lam.max():.3e} A")

    rt=open(os.path.join(RUNDIR,'RVTJ')).read()
    # KEEP R in CMFGEN units of 10^10 cm: CHI_DATA opacity is per (10^10 cm) so
    # tau = INT chi dR with R in 10^10 cm. (r^2 dr weight is relative -> unit cancels
    # in the per-band normalization.) depth0=outer .. depthND-1=inner.
    R=parse_rvtj(rt,'Radius (10^10 cm)',ND)          # units of 10^10 cm
    V=parse_rvtj(rt,'Velocity (km/s)',ND)            # km/s, decreasing with depth
    print(f"[rvtj] R {R[0]:.3e}..{R[-1]:.3e} cm  V {V[0]:.0f}..{V[-1]:.0f} km/s")

    # radial tau from each depth OUTWARD to boundary (depth0). tau[:,0]=0.
    dR = R[:-1]-R[1:]                                 # >0, thickness between i-1 and i (len ND-1)
    tau = np.zeros_like(chi)                          # [nf, ND]
    # tau[:,i] = tau[:,i-1] + 0.5*(chi[:,i]+chi[:,i-1])*(R[i-1]-R[i])
    seg = 0.5*(chi[:,1:]+chi[:,:-1])*dR[None,:]       # [nf, ND-1]
    tau[:,1:] = np.cumsum(seg,axis=1)
    # shell volume weight r^2 dr  (central-difference thickness)
    dr_shell=np.empty(ND)
    dr_shell[1:-1]=0.5*(R[:-2]-R[2:]); dr_shell[0]=R[0]-R[1]; dr_shell[-1]=R[-2]-R[-1]
    vol = R**2 * dr_shell                             # [ND]
    escp = np.exp(-tau)                               # escape prob
    contr = eta * escp * vol[None,:]                  # [nf, ND] dL contribution per freq per depth

    # band-integrate over CMF wavelength
    E=np.zeros((len(BANDS),ND))
    for b,(lo,hi) in enumerate(BANDS):
        m=(lam>=lo)&(lam<hi)
        if m.any(): E[b]=contr[m].sum(0)
    band_tot=E.sum(1)
    CF=np.where(band_tot[:,None]>0,E/band_tot[:,None],0.0)

    # write CF(band, depth)
    with open(os.path.join(OUT,'cmfgen_CF_band_depth.csv'),'w') as f:
        f.write("band,depth,v_kms,CF_frac,dL_abs\n")
        for b in range(len(BANDS)):
            for i in range(ND):
                f.write(f"{BAND_LBL[b]},{i},{V[i]:.0f},{CF[b,i]:.5f},{E[b,i]:.4e}\n")

    print("\n=== CMFGEN band luminosity budget (CMF-frame radial CF) ===")
    tot=band_tot.sum()
    for b in range(len(BANDS)):
        print(f"  {BAND_LBL[b]:>10s}: dL={band_tot[b]:.4e} ({100*band_tot[b]/tot:5.1f}%)")

    # mean/median formation velocity per band + fraction in velocity zones matched to Lumina
    # zones: deep (v<3900 = below Lumina inner BC), forming (3900<=v<11908 = Lumina s0..s10),
    #        outer (v>=11908 = Lumina s>=11)
    print("\n=== CMFGEN mean/median formation velocity per band ===")
    with open(os.path.join(OUT,'cmfgen_formation_velocity.csv'),'w') as f:
        f.write("band,mean_v_kms,median_v_kms,frac_deep_lt3900,frac_forming_3900_11908,frac_outer_gt11908\n")
        for b in range(len(BANDS)):
            w=E[b]
            if w.sum()>0:
                mean_v=np.average(V,weights=w)
                srt=np.argsort(V); cw=np.cumsum(w[srt]); med_v=V[srt][np.searchsorted(cw,0.5*w.sum())]
                f_deep=w[V<3900].sum()/w.sum(); f_form=w[(V>=3900)&(V<11908)].sum()/w.sum(); f_out=w[V>=11908].sum()/w.sum()
            else: mean_v=med_v=f_deep=f_form=f_out=0
            f.write(f"{BAND_LBL[b]},{mean_v:.0f},{med_v:.0f},{f_deep:.4f},{f_form:.4f},{f_out:.4f}\n")
            print(f"  {BAND_LBL[b]:>10s}: mean_v={mean_v:6.0f} median_v={med_v:6.0f} km/s  deep<3900={100*f_deep:4.1f}%  forming={100*f_form:4.1f}%  outer>11908={100*f_out:4.1f}%")
    print(f"\n[wrote] {OUT}/cmfgen_CF_band_depth.csv, cmfgen_formation_velocity.csv")

if __name__=='__main__':
    main()
