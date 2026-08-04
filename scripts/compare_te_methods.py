#!/usr/bin/env python3
"""Side-by-side T_e(v) comparison: Lumina methods vs CMFGEN/ARTIS references (toy06 19.48d)."""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
GEO = f"{ROOT}/data/tardis_reference_toy06_19p48d/geometry.csv"

def lumina_te(run):
    p = f"{ROOT}/logs/{run}/lumina_plasma_state.csv"
    if not os.path.exists(p): return None
    Te = {int(r['shell_id']): float(r['T_e']) for r in csv.DictReader(open(p))}
    return Te

def vmid():
    g = list(csv.DictReader(open(GEO)))
    return {int(r['shell_id']): (float(r['v_inner'])+float(r['v_outer']))/2/1e5 for r in g}

def ref_block(code, tt=19.48):
    p = f"{ROOT}/data/standart_data1/toy06/phys_toy06_{code}.txt"
    out={};cur=None;rows=[]
    for ln in open(p):
        s=ln.strip()
        if s.startswith('#TIME:'):
            if cur is not None: out[cur]=rows
            cur=float(s.split(':')[1]); rows=[]
        elif s.startswith('#'): continue
        elif s and cur is not None:
            try: rows.append([float(x) for x in s.split()])
            except: pass
    out[cur]=rows
    return sorted(out[min(out,key=lambda x:abs(x-tt))])

def interp(rows, vv):
    for i in range(len(rows)-1):
        if rows[i][0]<=vv<=rows[i+1][0]:
            return rows[i][1]+(rows[i+1][1]-rows[i][1])*(vv-rows[i][0])/(rows[i+1][0]-rows[i][0])
    return rows[-1][1] if vv>rows[-1][0] else rows[0][1]

V = vmid()
methods = [("escape (legacy)", "stage1_toy06_off"),
           ("transport-coupled (S2)", "stage1_toy06_s2"),
           ("transport+eps-fix (S2eps)", "stage1_toy06_s2eps")]
cmfgen = ref_block("cmfgen"); artis = ref_block("artis"); tardis = ref_block("tardis")

# table at diagnostic velocities
print(f"{'v[km/s]':>8} {'CMFGEN':>8} {'ARTIS':>8}", end="")
loaded=[(n,lumina_te(r)) for n,r in methods]
for n,_ in loaded: print(f" {n[:16]:>16}", end="")
print()
for s in [0,5,15,25,49]:
    vv=V[s]
    print(f"{vv:8.0f} {interp(cmfgen,vv):8.0f} {interp(artis,vv):8.0f}", end="")
    for n,Te in loaded:
        print(f" {(Te[s] if Te else 0):16.0f}", end="")
    print(f"   (shell {s})")

# figure
BG=(0x14/255,0x0D/255,0x44/255)
fig,ax=plt.subplots(figsize=(11,7)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
vc=[r[0] for r in cmfgen]; tc=[r[1] for r in cmfgen]
va=[r[0] for r in artis]; ta=[r[1] for r in artis]
ax.plot(vc,tc,'-',color='#FFC107',lw=2.5,label='CMFGEN (target)')
ax.plot(va,ta,'-',color='#D97757',lw=2.0,label='ARTIS')
vt=[r[0] for r in tardis]; tt2=[r[1] for r in tardis]
ax.plot(vt,tt2,'-',color='#B0AEA5',lw=2.0,label='TARDIS')
cols=['#707E9A','#3898EC','#4EC9B0']
for (n,Te),c in zip(loaded,cols):
    if not Te: continue
    vv=[V[s] for s in range(50)]; tt=[Te[s] for s in range(50)]
    ax.plot(vv,tt,'o-',color=c,ms=4,lw=1.8,label=f'Lumina {n}')
ax.set_xlabel('velocity [km/s]',color='#FAF9F5'); ax.set_ylabel('T_e [K]',color='#FAF9F5')
ax.set_title('toy06 19.48d: T_e methods vs StaNdaRT references',color='#FAF9F5')
ax.set_xlim(0,42000); ax.set_ylim(0,30000)
ax.tick_params(colors='#B0AEA5');
for sp in ax.spines.values(): sp.set_color('#707E9A')
ax.legend(facecolor=(0x20/255,0x18/255,0x58/255),edgecolor='#707E9A',labelcolor='#FAF9F5')
ax.grid(alpha=0.15)
out=f"{ROOT}/figures/2026-06-30_te_methods_vs_standart.png"
os.makedirs(f"{ROOT}/figures",exist_ok=True)
fig.savefig(out,dpi=130,facecolor=BG,bbox_inches='tight')
print(f"\nfigure -> {out}")
