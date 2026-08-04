#!/usr/bin/env python3
"""Forensic query for the 404-461A (Fe III excited-edge) EUV band at the
photosphere (s8) in a LUMINA event log.

Stage 'summary'  : full-file column pass (no packet correlation).
Stage 'corr'     : packet-sampled trajectory correlation (emission shell/process
                   of photons bf-absorbed at s8 in the window).

Window: nu_comov in [6.503e15, 7.421e15] Hz  == 404-461 A.
Record schema: read_events.py (u4 pkt_id, i4 line_id, f4 nu_comov, f4 energy,
               u1 etype, u1 shell, u1 iter, u1 pad); 32-byte header.
"""
import sys, os, numpy as np

EVENT_DTYPE = np.dtype([
    ("pkt_id","<u4"),("line_id","<i4"),("nu_comov","<f4"),("energy","<f4"),
    ("etype","u1"),("shell","u1"),("iter","u1"),("pad","u1")])
LINE_DTYPE = np.dtype([("lam","<f4"),("Z","<u2"),("ion","<u2")])
C = 2.99792458e10           # cm/s ; lam_A = C/nu*1e8
ETN = {1:"line-abs",2:"line-emit",3:"bf-abs",4:"kpkt-ff",5:"kpkt-fb",
       6:"escape",7:"e-scat",8:"bf-reemit"}
ZN = {6:"C",8:"O",12:"Mg",13:"Al",14:"Si",16:"S",20:"Ca",21:"Sc",22:"Ti",
      23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni"}
ROMAN = {0:"I",1:"II",2:"III",3:"IV",4:"V",5:"VI"}

NU_LO, NU_HI = 6.503e15, 7.421e15     # 461A .. 404A
HDR = 32

def memmap_events(d):
    path = os.path.join(d, "lumina_events.bin")
    with open(path,"rb") as f:
        h = f.read(HDR); assert h[:8]==b"LUMEVT01", h[:8]
        rsz = int(np.frombuffer(h[8:12],dtype="<u4")[0]); assert rsz==20, rsz
    n = (os.path.getsize(path)-HDR)//20
    return np.memmap(path, dtype=EVENT_DTYPE, mode="r", offset=HDR, shape=(n,))

def load_lines(d):
    with open(os.path.join(d,"lumina_events_lines.bin"),"rb") as f:
        assert f.read(8)==b"LUMLIN01"
        return np.frombuffer(f.read(),dtype=LINE_DTYPE)

def summary(d):
    ev = memmap_events(d); n = len(ev)
    print(f"# file: {d}  records={n}")
    et  = np.asarray(ev["etype"])
    sh  = np.asarray(ev["shell"])
    nu  = np.asarray(ev["nu_comov"])
    lid = np.asarray(ev["line_id"])
    en  = np.asarray(ev["energy"])

    print("\n## etype distribution (whole file)")
    for e in sorted(np.unique(et)):
        m = et==e
        print(f"  {ETN.get(int(e),e):<10} n={int(m.sum()):>12}  E={en[m].sum():.4e}")

    inw = (nu>=NU_LO)&(nu<=NU_HI)
    print(f"\n## in-window 404-461A (nu in [{NU_LO:.3e},{NU_HI:.3e}]): n={int(inw.sum())}")
    print("   etype dist IN WINDOW:")
    for e in sorted(np.unique(et[inw])):
        m = inw&(et==e)
        print(f"     {ETN.get(int(e),e):<10} n={int(m.sum()):>11}  E={en[m].sum():.4e}")

    print("\n## line_id availability per etype (min/max/frac>=0) -- does bf carry ion?")
    for e in sorted(np.unique(et)):
        m = et==e
        l = lid[m]
        print(f"   {ETN.get(int(e),e):<10} line_id min={int(l.min()):>8} max={int(l.max()):>8}"
              f"  frac>=0={(l>=0).mean():.3f}")

    # per-shell tallies in window, split by emission vs absorption etypes
    print("\n## IN-WINDOW per-shell tallies (count | energy)")
    print("   shell | bf-abs(3) | line-abs(1) || line-emit(2) | kpkt-fb(5) | bf-reemit(8) | kpkt-ff(4) | escape(6) | e-scat(7)")
    for s in range(0, int(sh.max())+1):
        base = inw&(sh==s)
        cells=[]
        for e in (3,1,2,5,8,4,6,7):
            m = base&(et==e)
            cells.append(f"{int(m.sum()):>8}")
        print(f"   s{s:<3} " + " ".join(cells))
    print("\n   (energy-weighted, same columns)")
    for s in range(0, int(sh.max())+1):
        base = inw&(sh==s)
        cells=[]
        for e in (3,1,2,5,8,4,6,7):
            m = base&(et==e)
            cells.append(f"{en[m].sum():.2e}")
        print(f"   s{s:<3} " + " ".join(cells))

    # focus: bf-abs at s8 in window, and emission (2,5,8) deep vs local
    print("\n## FOCUS")
    m_abs8 = inw&(sh==8)&(et==3)
    print(f"   bf-abs@s8 in-window: n={int(m_abs8.sum())} E={en[m_abs8].sum():.4e}")
    emit = np.isin(et,[2,5,8])
    for label,shells in [("deep s0-s3",range(0,4)),("local s6-s10",range(6,11))]:
        mm = inw&emit&np.isin(sh,list(shells))
        print(f"   emission(2/5/8) {label}: n={int(mm.sum())} E={en[mm].sum():.4e}")
        for e in (2,5,8):
            m = inw&(et==e)&np.isin(sh,list(shells))
            print(f"       {ETN[e]:<10} n={int(m.sum()):>10} E={en[m].sum():.3e}")

def corr(d, stride, lines=None):
    """EXACT trajectory correlation (no sampling). Step 1: find every bf-abs at s8
    in the 404-461A window -> its packet ids. Step 2: gather the FULL event history
    of just those packets. Step 3: for each target row, ffill the immediately
    preceding EMISSION event of the same packet (etype 2 line-emit / 5 kpkt-fb /
    8 bf-reemit / 4 kpkt-ff) -> emission shell (D1) and process (D2). Single
    iteration (iter=11), so packet identity == pkt_id."""
    import pandas as pd
    ev = memmap_events(d)
    et_all = np.asarray(ev["etype"]); sh_all = np.asarray(ev["shell"])
    nu_all = np.asarray(ev["nu_comov"]); pkt_all = np.asarray(ev["pkt_id"])
    tgt_mask = (et_all==3)&(sh_all==8)&(nu_all>=NU_LO)&(nu_all<=NU_HI)
    tgt_pkts = np.unique(pkt_all[tgt_mask])
    print(f"# corr EXACT: bf-abs@s8 in-window targets={int(tgt_mask.sum())} in {len(tgt_pkts)} packets")
    sel = np.isin(pkt_all, tgt_pkts)          # full history of those packets
    idx = np.nonzero(sel)[0]
    print(f"#   gathered {len(idx)} events (full trajectories of target packets)")
    pk = pkt_all[idx]
    et = et_all[idx]
    sh = sh_all[idx]
    nu = nu_all[idx]
    en = np.asarray(ev["energy"])[idx].astype(np.float64)
    lid= np.asarray(ev["line_id"])[idx]
    # stable sort by pkt_id -> groups packets, preserves intra-packet file order
    order = np.argsort(pk, kind="stable")
    pk,et,sh,nu,en,lid = pk[order],et[order],sh[order],nu[order],en[order],lid[order]

    is_emit = np.isin(et, [2,5,8,4])
    df = pd.DataFrame({"pk":pk})
    df["em_sh"]  = np.where(is_emit, sh, np.nan)
    df["em_et"]  = np.where(is_emit, et, np.nan)
    df["em_nu"]  = np.where(is_emit, nu, np.nan)
    df["em_lid"] = np.where(is_emit, lid.astype(np.float64), np.nan)
    g = df.groupby("pk", sort=False)
    ff_sh  = g["em_sh"].ffill().to_numpy()
    ff_et  = g["em_et"].ffill().to_numpy()
    ff_nu  = g["em_nu"].ffill().to_numpy()
    ff_lid = g["em_lid"].ffill().to_numpy()

    tgt = (et==3) & (sh==8) & (nu>=NU_LO) & (nu<=NU_HI)
    nT = int(tgt.sum()); eT = float(en[tgt].sum())
    print(f"\n## TARGETS (EXACT, full population): bf-abs @ s8 in 404-461A window  n={nT}  E={eT:.4e}")
    if nT==0:
        return
    tsh, tet, tnu, tlid, ten = ff_sh[tgt], ff_et[tgt], ff_nu[tgt], ff_lid[tgt], en[tgt]

    inj = np.isnan(tsh)
    print(f"\n## D1 EMISSION-SHELL of s8-absorbed EUV photons  (n={nT}, E={eT:.3e})")
    print(f"   INJECTED/no-prior-emit (source packet, deepest BC): n={int(inj.sum())} "
          f"({int(inj.sum())/nT*100:.1f}%)  E={ten[inj].sum():.3e} ({ten[inj].sum()/eT*100:.1f}%)")
    bins = [("deep s0-s3",0,4),("mid s4-s5",4,6),("LOCAL s6-s10",6,11),
            ("s11-s15",11,16),("outer s16+",16,999)]
    for name,lo,hi in bins:
        m = (~inj) & (tsh>=lo) & (tsh<hi)
        print(f"   {name:<14}: n={int(m.sum()):>9} ({int(m.sum())/nT*100:5.1f}%)  "
              f"E={ten[m].sum():.3e} ({ten[m].sum()/eT*100:5.1f}%)")
    # per-shell fine for the interesting range
    print("   per-shell (non-injected):")
    for s in range(0,16):
        m = (~inj)&(tsh==s)
        if m.sum(): print(f"       s{s:<3} n={int(m.sum()):>8} ({int(m.sum())/nT*100:4.1f}%)  E={ten[m].sum():.3e}")

    print(f"\n## D2 EMISSION-PROCESS of s8-absorbed EUV photons (non-injected)")
    nn = ~inj
    for e in (5,8,2,4):
        m = nn&(tet==e)
        if m.sum():
            print(f"   {ETN[e]:<10} n={int(m.sum()):>9} ({int(m.sum())/nT*100:5.1f}%)  E={ten[m].sum():.3e} ({ten[m].sum()/eT*100:5.1f}%)")
    # cross: process x (deep vs local)
    print("   process x origin-region (count):")
    for e in (5,8,2,4):
        d03 = int((nn&(tet==e)&(tsh<4)).sum())
        loc = int((nn&(tet==e)&(tsh>=6)&(tsh<11)).sum())
        print(f"       {ETN[e]:<10} deep(s0-3)={d03:>8}  LOCAL(s6-10)={loc:>8}")
    # bb-origin ion identity
    if lines is not None:
        ln = load_lines(d); Zc, ionc = ln["Z"], ln["ion"]
        bb = nn&(tet==2)&(tlid>=0)
        if bb.sum():
            li = tlid[bb].astype(int); ee = ten[bb]
            agg={}
            for z,i,w in zip(Zc[li],ionc[li],ee):
                agg[(int(z),int(i))]=agg.get((int(z),int(i)),0.0)+float(w)
            print("   bb-origin ion palette (of s8-absorbed photons that came from a line):")
            for (z,i),w in sorted(agg.items(),key=lambda kv:-kv[1])[:8]:
                print(f"       {ZN.get(z,z):>3} {ROMAN.get(i,i):<4} E={w:.3e} ({w/ten[bb].sum()*100:4.1f}%)")

    # QC: is the ffilled emission frequency also in-window? (validates ordering)
    inw_emit = (~inj)&(ff_nu[tgt]>=NU_LO*0.90)&(ff_nu[tgt]<=NU_HI*1.05)
    print(f"\n## QC ordering: of non-injected targets, frac whose PRIOR-EMIT nu is in/near window "
          f"= {int(inw_emit.sum())}/{int(nn.sum())} = {int(inw_emit.sum())/max(1,int(nn.sum()))*100:.1f}%")

    # QC raw dump: full event sequence of a few target packets
    print("\n## QC raw trajectory dump (3 target packets; lam_A=C/nu*1e8):")
    tpk = pk[tgt]
    seen=set()
    lam = np.where(nu>0, C/nu*1e8, 0.0)
    for want in np.unique(tpk)[:3]:
        rows = np.nonzero(pk==want)[0]
        print(f"   --- packet {int(want)} ({len(rows)} events) ---")
        for r in rows:
            star = " <== TARGET(bf-abs@s8 EUV)" if tgt[r] else ""
            print(f"     et={ETN.get(int(et[r]),et[r]):<9} s{int(sh[r]):<3} "
                  f"lam={lam[r]:8.1f}A nu={nu[r]:.3e} E={en[r]:.2e} lid={int(lid[r]):>8}{star}")

    # supporting: provenance of the target PACKETS (did they ever visit deep s<=3?)
    grp = df.assign(sh=sh).groupby("pk")
    minsh = grp["sh"].min()
    print(f"\n## SUPPORT: provenance of the {len(tgt_pkts)} target packets")
    print(f"   packets that EVER visited deep s<=3 : {int((minsh<=3).sum())}/{len(minsh)} "
          f"({int((minsh<=3).sum())/len(minsh)*100:.0f}%)  "
          f"[even if born/visited deep, the s8-absorbed EUV photon is made locally]")

if __name__=="__main__":
    stage = sys.argv[1] if len(sys.argv)>1 else "summary"
    d = sys.argv[2] if len(sys.argv)>2 else "logs/coevolve_consume_a10_kx_tepop1"
    if stage=="summary":
        summary(d)
    elif stage=="corr":
        stride = int(sys.argv[3]) if len(sys.argv)>3 else 8
        corr(d, stride, lines=True)
