#!/usr/bin/env python3
"""Identify WHICH ion/level carries the super-thermal optical S_l (job 166121),
and test the 'thick lines are also super-thermal' falsifier of physical-nebular.
Join line_id -> line_list (Z, ion, lower/upper level)."""
import numpy as np, pandas as pd
ROOT="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
DUMP=f"{ROOT}/logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_166135/lumina_sl_vs_B.csv"
LL=f"{ROOT}/data/tardis_reference_ddc15_0p976d/line_list.csv"

d=pd.read_csv(DUMP)
ll=pd.read_csv(LL,usecols=["atomic_number","ion_number","level_number_lower","level_number_upper"])
ll=ll.reset_index().rename(columns={"index":"line_id"})
d=d.merge(ll,on="line_id",how="left")
d=d[(d.Sl>0)&(d.B_Te>0)].copy()
d["Sl_over_B"]=d.Sl/d.B_Te

# (1) FALSIFIER: are optically THICK lines also super-thermal? (physical nebular
# only super-thermalizes thin sub-critical lines; thick lines MUST -> S_l~B)
print("=== (1) thick-line falsifier (tau>30) ===")
thk=d[d.tau>30]; opt_thk=thk[(thk.lambda_A>4000)&(thk.lambda_A<7000)]
print(f"  optical tau>30 lines: N={len(opt_thk)}  "
      f"frac S_l/B>10 = {np.mean(opt_thk.Sl_over_B>10):.3f}  "
      f"max S_l/B = {opt_thk.Sl_over_B.max():.2e}")
print("  -> physical nebular CANNOT super-thermalize trapped (beta<<1) lines\n")

# (2) which (Z,ion) carries the optical super-thermal forest?
print("=== (2) optical 4000-7000A super-thermal carriers by (Z,ion) ===")
opt=d[(d.lambda_A>=4000)&(d.lambda_A<7000)&(d.Sl_over_B>10)]
g=opt.groupby(["atomic_number","ion_number"]).agg(
    n=("Sl_over_B","size"), medSlB=("Sl_over_B","median")).sort_values("n",ascending=False)
print(g.head(10).to_string())

# (3) shell dependence: is it worst in COLD outer shells?
print("\n=== (3) optical super-thermal vs shell (Te) ===")
s=d[(d.lambda_A>=4000)&(d.lambda_A<7000)].groupby("shell").agg(
    Te=("Te","first"), medSlB=("Sl_over_B","median"), n=("Sl_over_B","size"))
for sh in [0,8,16,24,32,40,48]:
    if sh in s.index:
        r=s.loc[sh]; print(f"  shell{sh:2d}  Te={r.Te:6.0f}K  median S_l/B={r.medSlB:10.3g}  N={int(r.n)}")

# (4) lower-level dependence in a cold shell (ground-drain signature)
print("\n=== (4) optical S_l/B by LOWER level, cold shell (ground drain?) ===")
cold=d[(d.lambda_A>=4000)&(d.lambda_A<7000)&(d.shell.between(23,33))]
ll_g=cold.groupby("level_number_lower").agg(
    medSlB=("Sl_over_B","median"), n=("Sl_over_B","size")).head(12)
print(ll_g.to_string())
