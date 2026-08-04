#!/usr/bin/env python3
"""Per-shell offline adjudication: does switching the up-rate field from thermal (B(T_e),
~jbar_line_det) to the DETERMINISTIC binned cs.J lift b_k>1 across the line-forming shells?
If yes for s5-35 => a separated-version fix (deterministic blue-wing) is viable. If only at the
photosphere => co-evolution's MC-measured J_blue is required.

2-level b_k for a real Fe II UV line (lam2580, up=298 g=4, lo=87 g=6, A_ul=3.16e8) under
J = B(T_e) [Lumina thermal J-bar]  vs  J = cs.J(nu_line) [deterministic binned field].
Collisions (van Regemorter, Lumina's coeff). NOTE: binned cs.J is the MEAN field, a LOWER bound
on the true blue-wing incident J_blue; so b_k(det) here is conservative.
Usage: offline_bk_per_shell.py <field_csv> <plasma_csv>
"""
import sys, csv
import numpy as np
H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; EV=1.602176634e-12
field_csv=sys.argv[1] if len(sys.argv)>1 else "logs/coevolve_s01/lumina_coevolve_field.csv"
plasma_csv=sys.argv[2] if len(sys.argv)>2 else "logs/stage1_toy06_epay27/lumina_plasma_state.csv"

# Fe II 2580 line
A_ul=3.157e8; B_lu=9.096594470e9; B_ul=1.3644891706e10; nu=1.162037e15
g_lo=6.0; g_up=4.0; dE=(10.6290222511-5.8232247387)*EV; f_lu=0.21
VAN_REG=2.16e-6   # Lumina's VAN_REG_COEFF (plasma.c:1629)
def Bnu(nu,T): x=H*nu/(KB*T); return (2*H*nu**3/C**2)/np.expm1(np.clip(x,1e-9,700))

pl={int(r['shell_id']):r for r in csv.DictReader(open(plasma_csv))}
# field per shell
fld={}
for r in csv.DictReader(open(field_csv)):
    s=int(r['shell']); fld.setdefault(s,([],[]))
    fld[s][0].append(float(r['wavelength_A'])); fld[s][1].append(float(r['cs_J']))

def bk(Te,ne,J):
    lte=(g_up/g_lo)*np.exp(-dE/(KB*Te))
    C_ul=VAN_REG*ne*f_lu*0.2/(np.sqrt(Te)*g_up); C_lu=C_ul*lte
    ratio=(B_lu*J + C_lu)/(A_ul + B_ul*J + C_ul)
    return ratio/lte, C_ul/(A_ul+B_ul*Bnu(nu,Te))

lam_line=C/nu*1e8
print(f"Fe II {lam_line:.0f}A per-shell b_k: J=B(Te) [Lumina thermal] vs J=cs.J [deterministic binned]")
print(f"{'s':>3} {'Te':>6} {'ne':>9} | {'csJ/B(Te)':>9} | {'bk(thermal)':>11} {'bk(det csJ)':>11} {'lifted?':>7}")
lifted=[]
for s in sorted(fld):
    if s not in pl: continue
    Te=float(pl[s]['T_e']); ne=float(pl[s]['n_e'])
    lam=np.array(fld[s][0]); J=np.array(fld[s][1]); o=np.argsort(lam)
    csJ=float(np.interp(lam_line, lam[o], J[o]))
    Bte=Bnu(nu,Te)
    bk_th,_=bk(Te,ne,Bte)
    bk_det,cfrac=bk(Te,ne,csJ)
    lift = bk_det>1.15
    if 5<=s<=35: lifted.append(lift)
    if s%3==0 or s in (5,):
        print(f"{s:>3} {Te:6.0f} {ne:9.2e} | {csJ/Bte:9.3f} | {bk_th:11.4f} {bk_det:11.4f} {'YES' if lift else 'no':>7}")
print("\n=== VERDICT (line-forming s5-35) ===")
frac=100*np.mean(lifted) if lifted else 0
print(f"shells where deterministic cs.J lifts b_k>1.15: {frac:.0f}% of s5-35")
if frac>=80:
    print("  => deterministic binned field lifts b_k across the line-forming region:")
    print("     a SEPARATED-version fix (feed cs.J, not jbar_line_det, to mode-3) is VIABLE.")
elif frac>=30:
    print("  => deterministic field lifts b_k only in PART of the region (photosphere-weighted):")
    print("     separated fix partial; co-evolution's MC blue-wing needed for the outer shells.")
else:
    print("  => deterministic binned field does NOT lift b_k outside the photosphere:")
    print("     CO-EVOLUTION (MC-measured blue-wing J_blue) is REQUIRED. Separated fix insufficient.")
print("NOTE: binned cs.J is the MEAN field = a lower bound on the true blue-wing J_blue, so this")
print("is CONSERVATIVE; a fine-grid blue-wing sample or the (renormalized) MC jbar_line may lift more.")
