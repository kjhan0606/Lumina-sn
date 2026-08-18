#!/usr/bin/env python3
"""TASK B -- event-log forensics: who emits the deep red/NUV excess at s0-s2.

Reads the B-run EVENT_LOG (lumina_events.bin, LUMEVT01, 20-byte EventRec; schema
from scripts/read_events.py:1-25, commit ac8ef44) + lumina_events_lines.bin
(LUMLIN01, per-line {lam_A,Z,ion}). Tallies EMISSION and ABSORPTION energy into
the Task-A band edges, per shell group, per ion/channel; computes the net
spectral flow, the blueward-across-1290A up-conversion rate (s0-2 vs s7-8), and
the emission-weighted color vs thermal B(T_e).

COVERAGE CAVEATS (audited, restated -- not fabricated):
  * n=128,000,000 events == CAP128M (log SATURATED; a truncated packet sample).
  * single iteration (iter=11), the last co-evolve pass.
  * etype 7 (e-scatter) and 8 (bf-reemit) are UNLOGGED -> bound-free recombination
    RE-EMISSION (the CMFGEN thermal continuum source) is INVISIBLE here; the
    emission ledger = line-emit(2)+kpkt-ff(4)+kpkt-fb(5). Absorption = line-abs(1)
    +bf-abs(3). So this measures the LINE-FOREST reprocessing flow, which is exactly
    the one-way down-conversion suspect -- but any purely-bf thermal channel is out
    of frame and its absence must be stated in the verdict.
  * event 'energy' = packet energy (not absolutely calibrated to CMFGEN erg units);
    only RELATIVE band/shell/ion shares are claimable.

Read-only. No source edits, no commit.
"""
import numpy as np, csv, os
REPO = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
D    = f'{REPO}/logs/coevolve_consume_a10_kx_gphall'
OUT  = os.path.dirname(os.path.abspath(__file__))

EV = np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
               ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
LINE = np.dtype([('lam','<f4'),('Z','<u2'),('ion','<u2')])
C_A = 2.99792458e18   # A/s
ETN = {1:'line-abs',2:'line-emit',3:'bf-abs',4:'kpkt-ff',5:'kpkt-fb',6:'escape'}
EMIT = (2,4,5); ABSb = (1,3)
EDGES = [100,300,450,918,1290,2000,3000,4500,7000,10000,19933,1e12]
BLAB  = ['soft_100_300','EUV_300_450','xuv_450_918','FUV_918_1290','NUV_1290_2000',
         'UV_2000_3000','blue_3000_4500','opt_4500_7000','red_7000_10000',
         'NIR_10000_19933','beyond_19933']
ELEM = {6:'C',8:'O',12:'Mg',13:'Al',14:'Si',16:'S',20:'Ca',21:'Sc',22:'Ti',
        24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni'}
ROM = {0:'I',1:'II',2:'III',3:'IV',4:'V'}

# ---------- load (memmap; subset by shell to bound memory) ----------
mm = np.memmap(f'{D}/lumina_events.bin', dtype=EV, mode='r', offset=32)
n = len(mm)
et_all = np.array(mm['etype']); sh_all = np.array(mm['shell'])
print("="*90)
print("COVERAGE AUDIT")
print(f"  n_events = {n:,}   CAP128M = {n==128_000_000}")
print(f"  iter uniq = {list(np.unique(np.array(mm['iter'])))}  (single co-evolve iteration)")
import collections
hist = sorted(collections.Counter(et_all.tolist()).items())
print("  etype hist: " + ", ".join(f"{ETN.get(e,e)}={c:,}" for e,c in hist))
print(f"  UNLOGGED: e-scatter(7), bf-reemit(8) -> bf recomb continuum INVISIBLE")
print(f"  shells present: {sh_all.min()}..{sh_all.max()} ({len(np.unique(sh_all))})")
print("="*90)

# subset to shells 0-2 and 7-8 (all we need); fancy-index nu/energy/line_id
keep = np.isin(sh_all, [0,1,2,7,8])
idx = np.nonzero(keep)[0]
nu  = np.array(mm['nu'])[idx]
en  = np.array(mm['energy'])[idx]
lid = np.array(mm['line_id'])[idx]
et  = et_all[idx]; sh = sh_all[idx]
lam = np.where(nu>0, C_A/nu, 0.0)
del mm
# line ion tables
with open(f'{D}/lumina_events_lines.bin','rb') as f:
    assert f.read(8)==b'LUMLIN01'
    lrec = np.frombuffer(f.read(), dtype=LINE)
Lz = lrec['Z'].astype(np.int32); Lion = lrec['ion'].astype(np.int32)

def bandidx(lam_arr):
    return np.digitize(lam_arr, EDGES) - 1   # 0..len(BLAB)-1, -1/oob guarded below

def group_mask(g):
    return {'s0-2':(sh>=0)&(sh<=2), 's7-8':(sh>=7)&(sh<=8)}[g]

# ---------- band ledger per group ----------
bi = bandidx(lam)
ledger_rows = []
for g in ['s0-2','s7-8']:
    gm = group_mask(g)
    print(f"\n{'#'*80}\n# BAND LEDGER  group {g}  (n={int(gm.sum()):,} events)\n{'#'*80}")
    print(f"  {'band':>17} {'emitE':>11} {'absE':>11} {'netE(emit-abs)':>15} {'net?':>5}")
    emE_tot=0.0; abE_tot=0.0
    for b in range(len(BLAB)):
        bmask = gm & (bi==b)
        emE = float(en[bmask & np.isin(et,EMIT)].sum())
        abE = float(en[bmask & np.isin(et,ABSb)].sum())
        emE_tot+=emE; abE_tot+=abE
        net = emE-abE
        print(f"  {BLAB[b]:>17} {emE:>11.4e} {abE:>11.4e} {net:>+15.4e} {'SRC' if net>0 else 'sink':>5}")
        ledger_rows.append([g,BLAB[b],EDGES[b],EDGES[b+1],emE,abE,net])
    print(f"  {'TOTAL':>17} {emE_tot:>11.4e} {abE_tot:>11.4e} {emE_tot-abE_tot:>+15.4e}")

# ---------- top emitting / absorbing ions at s0-2, by band ----------
def ion_table(group, bands, etypes, label, topn=12):
    gm = group_mask(group)
    bandmask = np.isin(bi, bands)
    m = gm & bandmask & np.isin(et, etypes) & (lid>=0)
    lids = lid[m]; ens = en[m]
    z = Lz[lids]; io = Lion[lids]
    key = z*10 + io
    order = np.argsort(key)
    ks = key[order]; es = ens[order]
    uk, start = np.unique(ks, return_index=True)
    sums = np.add.reduceat(es, start)
    tot = sums.sum()
    top = sorted(zip(uk, sums), key=lambda kv:-kv[1])[:topn]
    print(f"\n  -- top {label} ions, {group}, bands {[BLAB[b] for b in bands]} "
          f"(total E={tot:.3e}) --")
    rows=[]
    for k,e in top:
        z=k//10; io=k%10; nm=f"{ELEM.get(z,z)} {ROM.get(io,io)}"
        print(f"       {nm:>8}  E={e:.3e}  ({e/tot*100:5.1f}%)")
        rows.append([group,label,nm,int(z),int(io),float(e),float(e/tot)])
    return rows, tot

ion_rows=[]
# the NUV pile (1290-2000, band idx 4) and red/NIR (7000-19933, bands 8,9) emitters at s0-2
ion_rows += ion_table('s0-2',[4],       EMIT, 'EMIT_NUVpile_1290_2000')[0]
ion_rows += ion_table('s0-2',[8,9],     EMIT, 'EMIT_red_7000_19933')[0]
ion_rows += ion_table('s0-2',[6,7],     EMIT, 'EMIT_optblue_3000_7000')[0]
# the FUV absorbers (who removes the blue): bands 2,3 (450-1290) at s0-2
ion_rows += ion_table('s0-2',[2,3],     ABSb, 'ABS_FUVxuv_450_1290')[0]
# and FUV emitters for contrast
ion_rows += ion_table('s0-2',[2,3],     EMIT, 'EMIT_FUVxuv_450_1290')[0]

# ---------- up-conversion across 1290A: blueward emission, s0-2 vs s7-8 ----------
print(f"\n{'#'*80}\n# UP-CONVERSION across 1290A (blueward emission into FUV/EUV)\n{'#'*80}")
print("  metric: emission (2,4,5) with lambda<1290A = energy placed BLUE of 1290;")
print("  net = that minus absorption(1,3) at lambda<1290; positive net = shell MANUFACTURES blue.")
blue = lam < 1290.0
uc_rows=[]
for g,nsh in [('s0-2',3),('s7-8',2)]:
    gm = group_mask(g)
    m_emit_blue = gm & blue & np.isin(et,EMIT)
    m_abs_blue  = gm & blue & np.isin(et,ABSb)
    m_kpkt_blue = gm & blue & np.isin(et,(4,5))
    n_emit=int(m_emit_blue.sum()); E_emit=float(en[m_emit_blue].sum())
    n_abs =int(m_abs_blue.sum());  E_abs =float(en[m_abs_blue].sum())
    n_kp  =int(m_kpkt_blue.sum()); E_kp  =float(en[m_kpkt_blue].sum())
    net = E_emit-E_abs
    print(f"\n  [{g}] ({nsh} shells)")
    print(f"     blueward EMISSION(<1290): n={n_emit:,}  E={E_emit:.4e}   (per-shell E={E_emit/nsh:.4e})")
    print(f"     blueward ABSORPTION(<1290): n={n_abs:,}  E={E_abs:.4e}   (per-shell E={E_abs/nsh:.4e})")
    print(f"     unambiguous kpkt-exit blueward(<1290): n={n_kp:,}  E={E_kp:.4e}")
    print(f"     NET blue (emit-abs) = {net:+.4e}   (per-shell {net/nsh:+.4e})  -> {'net SOURCE' if net>0 else 'net SINK'}")
    uc_rows.append([g,nsh,n_emit,E_emit,n_abs,E_abs,n_kp,E_kp,net,E_emit/nsh,net/nsh])
# ratios on same footing (per shell)
e02 = uc_rows[0][9]; e78 = uc_rows[1][9]   # per-shell blueward emission E
print(f"\n  RATIO (per-shell blueward EMISSION):  s7-8 / s0-2 = {e78/e02:.2f}x")
n78_02 = (uc_rows[1][2]/uc_rows[1][1])/(uc_rows[0][2]/uc_rows[0][1]) if uc_rows[0][2] else float('nan')

# ---------- emission color vs thermal B(T_e) at s0-2 ----------
print(f"\n{'#'*80}\n# EMISSION COLOR vs THERMAL B  (s0-2)\n{'#'*80}")
gm = group_mask('s0-2'); em = gm & np.isin(et,EMIT) & (lam>=100)&(lam<=20000)
lam_e = lam[em]; en_e = en[em]
# emission-weighted mean lambda
mean_lam = float((lam_e*en_e).sum()/en_e.sum())
# fraction of emission energy by band
def efrac(lo,hi):
    mm2=(lam_e>=lo)&(lam_e<hi); return float(en_e[mm2].sum()/en_e.sum())
def thermal_meanlam_and_fracs(T):
    # energy per bin on a fine grid over [100,20000]A: E ∝ Bnu dnu; Bnu=nu^3/(exp(hnu/kT)-1)
    lg = np.logspace(np.log10(100),np.log10(20000),4000)
    nug = C_A/lg
    h=6.62607015e-27; k=1.380649e-16
    Bnu = nug**3/np.expm1(h*nug/(k*T))
    # energy in each cell: |dnu| between grid points
    nu_edges = C_A/lg
    dnu = np.abs(np.gradient(nu_edges))
    E = Bnu*dnu
    ml = float((lg*E).sum()/E.sum())
    def fr(lo,hi):
        mm2=(lg>=lo)&(lg<hi); return float(E[mm2].sum()/E.sum())
    return ml, fr
mlT13, frT13 = thermal_meanlam_and_fracs(13120.)
mlT18, frT18 = thermal_meanlam_and_fracs(18760.)
print(f"  Lumina s0-2 emission-weighted mean lambda   = {mean_lam:7.1f} A")
print(f"  B(T_e=13120) energy-weighted mean lambda     = {mlT13:7.1f} A  (Lumina gas temp)")
print(f"  B(T_col=18760) energy-weighted mean lambda   = {mlT18:7.1f} A  (CMFGEN deep color)")
print(f"\n  {'band':>17} {'Lumina_emit':>12} {'B(13120)':>10} {'B(18760)':>10}")
for lo,hi,nm in [(100,1290,'<1290 FUV/EUV'),(1290,2000,'NUV pile'),(2000,4500,'2000-4500 valley'),
                 (4500,20000,'>4500 red/NIR')]:
    print(f"  {nm:>17} {efrac(lo,hi):>12.3f} {frT13(lo,hi):>10.3f} {frT18(lo,hi):>10.3f}")
frac_redward_of_thermalmean = float(en_e[lam_e>mlT13].sum()/en_e.sum())
print(f"\n  fraction of s0-2 emission energy REDWARD of B(13120) mean ({mlT13:.0f}A) = "
      f"{frac_redward_of_thermalmean:.3f}  (thermal would be ~0.5)")

# ---------- write CSVs ----------
with open(f'{OUT}/taskB_band_ledger.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['group','band','lo_A','hi_A','emitE','absE','netE'])
    w.writerows(ledger_rows)
with open(f'{OUT}/taskB_top_ions.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['group','role','ion','Z','ion_idx','E','frac_of_role'])
    w.writerows(ion_rows)
with open(f'{OUT}/taskB_upconversion.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['group','nshells','n_emit_blue','E_emit_blue','n_abs_blue',
                                 'E_abs_blue','n_kpkt_blue','E_kpkt_blue','net_blue',
                                 'E_emit_blue_per_shell','net_blue_per_shell'])
    w.writerows(uc_rows)
print(f"\n[out] {OUT}/taskB_band_ledger.csv, taskB_top_ions.csv, taskB_upconversion.csv")
