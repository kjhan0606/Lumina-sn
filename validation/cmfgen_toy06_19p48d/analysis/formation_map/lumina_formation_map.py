#!/usr/bin/env python3
"""Lumina per-shell SPECTRUM-FORMATION map from the B-run corpse (event log).

For every escaping packet (etype=6) we recover the shell of its LAST photon-creating
emission (etype in {2 line-emit, 4 kpkt-ff, 5 kpkt-fb, 8 bf-reemit}) via a vectorized
grouped forward-fill over pkt_id, then bin escape ENERGY by (band, emission shell).

Frame: escape frequency is COMOVING (nu_comov) -> band edges are comoving/CMF-frame,
matching the CMFGEN ETA_DATA (CMF-frame) side for an apples-to-apples comparison.
Escapes with no logged emission (free-streamed photospheric/continuum packets) are
attributed to an 'innerBC' bucket (Lumina inner boundary, v<3900 km/s).

Caveat: the event log is capped at 128M events = ONE converged iteration (iter 11),
truncated when the cap was hit (134673 of ~200000 escapes captured) -- a large,
unbiased-in-time subsample; the FRACTIONAL CF is robust. The observer-frame band
budget from lumina_kromer_coevolve.csv (all iters) is written alongside as reference.
"""
import numpy as np, os, csv, json

RUN = "logs/coevolve_consume_a10_kx_gphall"
OUT = "validation/cmfgen_toy06_19p48d/analysis/formation_map"
C_CM_S = 2.99792458e10
EV_DT = np.dtype([("pkt_id","<u4"),("line_id","<i4"),("nu_comov","<f4"),("energy","<f4"),
                  ("etype","u1"),("shell","u1"),("iter","u1"),("pad","u1")])
EMIT_ETYPES = (2,4,5,8)   # photon-creating events
BANDS = [(300,450),(450,918),(918,1290),(1290,2000),(2000,4500),(4500,7000),(7000,1e12),(1490,1650)]
BAND_LBL = ["300-450","450-918","918-1290","1290-2000","2000-4500","4500-7000","7000+","1490-1650"]

def band_index(lam):
    idx = np.full(lam.shape, -1, np.int32)
    for i,(lo,hi) in enumerate(BANDS):
        idx[(lam>=lo)&(lam<hi)] = i
    return idx

def load_geometry():
    g = np.genfromtxt(os.path.join(RUN,"..","..","data","tardis_reference_toy06_19p48d","geometry.csv"),
                      delimiter=",", names=True)
    vmid = 0.5*(g["v_inner"]+g["v_outer"])/1e5   # km/s
    return vmid

def main():
    p = os.path.join(RUN,"lumina_events.bin")
    n = (os.path.getsize(p)-32)//EV_DT.itemsize
    ev = np.memmap(p, dtype=EV_DT, mode="r", offset=32, shape=(n,))
    pid   = np.asarray(ev["pkt_id"])
    et    = np.asarray(ev["etype"])
    sh    = np.asarray(ev["shell"]).astype(np.int32)
    en    = np.asarray(ev["energy"]).astype(np.float64)
    nu    = np.asarray(ev["nu_comov"]).astype(np.float64)
    print(f"[events] {n} events, iters {ev['iter'].min()}..{ev['iter'].max()}")

    # ---- vectorized grouped forward-fill: last emission shell per pkt before each event
    order = np.argsort(pid, kind="stable")     # group by pkt_id, keep temporal order within group
    pid_s = pid[order]; et_s = et[order]; sh_s = sh[order]
    idx   = np.arange(n)
    grp_start = np.empty(n, bool); grp_start[0]=True; grp_start[1:] = pid_s[1:]!=pid_s[:-1]
    gfirst = np.where(grp_start, idx, 0); np.maximum.accumulate(gfirst, out=gfirst)  # first idx of each group
    is_emit = np.isin(et_s, EMIT_ETYPES)
    last_emit_pos = np.where(is_emit, idx, -1); np.maximum.accumulate(last_emit_pos, out=last_emit_pos)
    valid = last_emit_pos >= gfirst            # an emission occurred earlier in THIS packet's group
    emit_shell_at = np.where(valid, sh_s[np.clip(last_emit_pos,0,None)], -1)  # sorted order

    # read the linkage at escape events (in sorted order), then restore original identity
    esc_mask_s = et_s == 6
    esc_shell   = emit_shell_at[esc_mask_s]            # emission shell (or -1 = innerBC)
    esc_globidx = order[esc_mask_s]                    # original event index of each escape
    esc_en      = en[esc_globidx]
    esc_lam     = np.where(nu[esc_globidx]>0, C_CM_S/nu[esc_globidx]*1e8, 0.0)  # comoving A
    print(f"[escapes] {esc_shell.size}  Esum={esc_en.sum():.4e}  linked={np.mean(esc_shell>=0)*100:.1f}%  innerBC={np.mean(esc_shell<0)*100:.1f}%")

    # ---- build CF(band, shell). shell axis: -1(innerBC), 0..49
    bi = band_index(esc_lam)
    nsh = 50
    E = np.zeros((len(BANDS), nsh+1))   # col 0 = innerBC, cols 1..50 = shells 0..49
    for b in range(len(BANDS)):
        mb = bi==b
        col = np.where(esc_shell[mb]<0, 0, esc_shell[mb]+1)
        np.add.at(E[b], col, esc_en[mb])
    band_tot = E.sum(axis=1)
    CF = np.where(band_tot[:,None]>0, E/band_tot[:,None], 0.0)  # fractional per band

    vmid = load_geometry()
    # write CF table (fraction of each band formed in each shell)
    with open(os.path.join(OUT,"lumina_CF_band_shell.csv"),"w") as f:
        f.write("band,shell,v_mid_kms,CF_frac,E_abs\n")
        for b in range(len(BANDS)):
            f.write(f"{BAND_LBL[b]},innerBC,<3900,{CF[b,0]:.5f},{E[b,0]:.4e}\n")
            for s in range(nsh):
                f.write(f"{BAND_LBL[b]},{s},{vmid[s]:.0f},{CF[b,s+1]:.5f},{E[b,s+1]:.4e}\n")

    # band energy budget (comoving-frame, event-log) + shell-collapsed summary
    print("\n=== Lumina band energy budget (event-log iter11, comoving-frame) ===")
    tot = band_tot.sum()
    for b in range(len(BANDS)):
        print(f"  {BAND_LBL[b]:>10s}: E={band_tot[b]:.4e} ({100*band_tot[b]/tot:5.1f}%)  innerBC={100*CF[b,0]:4.1f}%")

    # emission-weighted mean formation velocity per band (excluding innerBC)
    print("\n=== mean/median formation velocity per band (linked escapes only) ===")
    with open(os.path.join(OUT,"lumina_formation_velocity.csv"),"w") as f:
        f.write("band,mean_v_kms,median_v_kms,frac_innerBC,frac_forming_s0_s10,frac_outer_gt11\n")
        for b in range(len(BANDS)):
            mb=(bi==b)&(esc_shell>=0)
            vv=vmid[esc_shell[mb]]; ee=esc_en[mb]
            if ee.sum()>0:
                mean_v=np.average(vv,weights=ee)
                srt=np.argsort(vv); cee=np.cumsum(ee[srt]); med_v=vv[srt][np.searchsorted(cee,0.5*ee.sum())]
            else: mean_v=med_v=0
            frac_form=CF[b,1:12].sum(); frac_out=CF[b,12:].sum()
            f.write(f"{BAND_LBL[b]},{mean_v:.0f},{med_v:.0f},{CF[b,0]:.4f},{frac_form:.4f},{frac_out:.4f}\n")
            print(f"  {BAND_LBL[b]:>10s}: mean_v={mean_v:6.0f}  median_v={med_v:6.0f} km/s  innerBC={100*CF[b,0]:4.1f}%  s0-s10={100*frac_form:4.1f}%  s>=11={100*frac_out:4.1f}%")

    # observer-frame reference budget from kromer
    lam=[]; ek=[]
    with open(os.path.join(RUN,"lumina_kromer_coevolve.csv")) as fh:
        r=csv.reader(fh); next(r)
        for row in r: lam.append(float(row[0])); ek.append(float(row[6]))
    lam=np.array(lam); ek=np.array(ek); bik=band_index(lam)
    with open(os.path.join(OUT,"lumina_kromer_band_budget_observer.csv"),"w") as f:
        f.write("band,E_abs,frac\n")
        for b in range(len(BANDS)):
            m=bik==b; f.write(f"{BAND_LBL[b]},{ek[m].sum():.4e},{ek[m].sum()/ek.sum():.4f}\n")
    print(f"\n[wrote] {OUT}/lumina_CF_band_shell.csv, lumina_formation_velocity.csv, lumina_kromer_band_budget_observer.csv")

if __name__=="__main__":
    main()
