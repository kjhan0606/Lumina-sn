#!/usr/bin/env python3
"""
COIV FUNNEL TRACE — Step 2 + Step 3.
Reconstruct EXACTLY what compute_transition_probabilities() builds for the
macro-atom block of the upper level of the 1526.17A Co IV line (line_id 391357,
global macro level 22564) at shell s0, and the collisional-vs-radiative ratio.

Replicates src/lumina_plasma.c:1799-2164 with the B-run's resolved env:
  EWEIGHT=1, NEUTRAL_E=1, IDOWN_BETA=1, IDOWN_COLL=0 (unset), KPACKET=1,
  j_cap=0, j_floor=0, IUP_TRAD/JBLUE/BETA all off.
Co IV (Z=27 ion=3) is NOT NLTE -> nebular dilute-Boltzmann pops at T_rad,W.
"""
import numpy as np, csv, os

REF = "data/tardis_reference_toy06_19p48d"
LOG = "logs/coevolve_consume_a10_kx_gphall"
OUT = "validation/cmfgen_toy06_19p48d/analysis/coiv_funnel_trace"

# ---- constants (src/lumina.h) ----
SOBOLEV_COEFF = 2.6540281e-02
K_B   = 1.380649e-16        # erg/K
EV    = 1.602176634e-12     # erg/eV
H     = 6.62607015e-27      # erg s
C     = 2.99792458e10       # cm/s
VAN_REG_COEFF = 2.16e-6
AX_OMEGA      = 1.0
t_exp = 1.683072e6          # s (stdout line 105)

# ---- shell 0 plasma (lumina_plasma_state.csv row 0) ----
S = 0
W     = 0.2978587262
T_rad = 10470.093240
T_e   = 13119.874754
n_e   = 4.426076e9
beta_rad = 1.0/(K_B*T_rad)
inv_sqrt_Te = 1.0/np.sqrt(T_e)

# ---- ionization energies -> accumulated IP (neutral-ground ref) ----
ioniz = {}
with open(os.path.join(REF,"ionization_energies.csv")) as f:
    r=csv.DictReader(f)
    for row in r:
        ioniz[(int(row["atomic_number"]),int(row["ion_number"]))]=float(row["ionization_energy_eV"])
def accum_ip(Z, ion):
    s=0.0
    for i in range(ion):
        s += ioniz.get((Z,i),0.0)
    return s

# ---- levels.csv: global-index arrays (order = file order) ----
lev_Z=[]; lev_ion=[]; lev_num=[]; lev_E=[]; lev_g=[]; lev_meta=[]
with open(os.path.join(REF,"levels.csv")) as f:
    r=csv.DictReader(f)
    for row in r:
        lev_Z.append(int(row["atomic_number"])); lev_ion.append(int(row["ion_number"]))
        lev_num.append(int(row["level_number"])); lev_E.append(float(row["energy_eV"]))
        lev_g.append(int(row["g"])); lev_meta.append(int(row["metastable"]))
lev_Z=np.array(lev_Z); lev_ion=np.array(lev_ion); lev_num=np.array(lev_num)
lev_E=np.array(lev_E); lev_g=np.array(lev_g); lev_meta=np.array(lev_meta)
NLEV=len(lev_Z)
print("levels.csv global levels:", NLEV)

# map (Z,ion,level_num)->global idx
g_of={}
for gi in range(NLEV):
    g_of[(lev_Z[gi],lev_ion[gi],lev_num[gi])]=gi

# ---- line_list.csv (row index == line_id) ----
LL = np.genfromtxt(os.path.join(REF,"line_list.csv"), delimiter=",", names=True)
ln_Z   = LL["atomic_number"].astype(int)
ln_ion = LL["ion_number"].astype(int)
ln_lo  = LL["level_number_lower"].astype(int)
ln_up  = LL["level_number_upper"].astype(int)
ln_flu = LL["f_lu"]
ln_nu  = LL["nu"]
ln_Aul = LL["A_ul"]
ln_Blu = LL["B_lu"]
ln_lamcm = LL["wavelength_cm"]
NLINE=len(ln_Z)
print("lines:", NLINE)

# ---- ion populations at s0 ----
ion_pop={}
with open(os.path.join(LOG,"lumina_ion_pops.csv")) as f:
    r=csv.DictReader(f)
    for row in r:
        if int(row["shell_id"])!=S: continue
        ion_pop[(int(row["Z"]),int(row["stage"]))]=float(row["n_ion"])

# ---- nebular partition function per ion at s0 (matches compute_tau_sobolev) ----
# Z_part = sum_meta g e^{-Eβrad} + W sum_nonmeta g e^{-Eβrad}
def nebular_Zpart(Z, ion):
    m = (lev_Z==Z)&(lev_ion==ion)
    E=lev_E[m]; g=lev_g[m]; meta=lev_meta[m]
    b=E*EV*beta_rad
    ok=b<500.0
    wt=np.where(meta==1,1.0,W)
    return np.sum(wt[ok]*g[ok]*np.exp(-b[ok]))

# nebular n_lower for a line's lower level (non-NLTE ion)
def nebular_nlow(Z, ion, Elow_eV, glow, meta, Zpart, n_ion):
    b=Elow_eV*EV*beta_rad
    if b>=500.0: return 0.0
    wt=1.0 if meta else W
    return n_ion*wt*glow*np.exp(-b)/Zpart

Zpart_CoIV = nebular_Zpart(27,3)
nion_CoIV  = ion_pop[(27,3)]
print("Co IV: n_ion=%.4e  Zpart(nebular,s0)=%.4f"%(nion_CoIV, Zpart_CoIV))

# tau for a Co IV line at s0 (nebular)
def tau_coiv(line_id):
    Elow=lev_E[g_of[(27,3,ln_lo[line_id])]]
    glow=lev_g[g_of[(27,3,ln_lo[line_id])]]
    meta=lev_meta[g_of[(27,3,ln_lo[line_id])]]
    nlow=nebular_nlow(27,3,Elow,glow,meta,Zpart_CoIV,nion_CoIV)
    # stim corr
    Eup=lev_E[g_of[(27,3,ln_up[line_id])]]
    gup=lev_g[g_of[(27,3,ln_up[line_id])]]
    metau=lev_meta[g_of[(27,3,ln_up[line_id])]]
    bu=Eup*EV*beta_rad
    nup = nion_CoIV*(1.0 if metau else W)*gup*np.exp(-bu)/Zpart_CoIV if bu<500 else 0.0
    stim=1.0
    if nlow>0 and nup>0:
        stim=1.0-(glow*nup)/(gup*nlow)
        if stim<0: stim=0.0
    tau=SOBOLEV_COEFF*ln_flu[line_id]*ln_lamcm[line_id]*t_exp*nlow*stim
    return max(tau,1e-100), nlow, nup, stim

def beta_sob(tau):
    if tau<1e-6: return 1.0-0.5*tau
    if tau>500.0: return 1.0/tau
    return (1.0-np.exp(-tau))/tau

# ---- macro_atom references: block for global level 22564 ----
# read block_references column indexed by references_idx (global level)
ref_block=np.full(NLEV+1,-1,dtype=np.int64)
ntrans=0
with open(os.path.join(REF,"macro_atom_references.csv")) as f:
    r=csv.DictReader(f)
    for row in r:
        gi=int(row["references_idx"]); ref_block[gi]=int(row["block_references"])
# sentinel: total transitions
# load macro_atom_data length + arrays for the block only
G=22564
print("\n=== 1526.17A line 391357: Co IV lower=%d upper=%d (global upper=%d) ==="%(
    ln_lo[391357], ln_up[391357], G))
t1526,nl1526,nu1526,st1526=tau_coiv(391357)
print("tau(1526)=%.4g  beta=%.4g  A_ul=%.4g  n_low(lev50)=%.4g n_up(lev144)=%.4g stim=%.4f"%(
    t1526, beta_sob(t1526), ln_Aul[391357], nl1526, nu1526, st1526))

# We need block_start (of G) and block_end (of next occupied ref). ref_block for
# levels with zero transitions may be -1; use the loaded value for G and G+1.. up
# to first >=0. Simpler: read the whole macro_atom_data and slice by source_level_idx==G.
print("\nReading macro_atom_data.csv (7.7M rows) filtering source_level_idx==%d ..."%G)
blk=[]
with open(os.path.join(REF,"macro_atom_data.csv")) as f:
    r=csv.DictReader(f)
    for row in r:
        if int(row["source_level_idx"])!=G: continue
        blk.append((int(row["transition_type"]),
                    int(row["destination_level_idx"]),
                    int(row["lines_idx"])))
print("block size (source_level_idx==%d): %d transitions"%(G,len(blk)))

# ---- build rates exactly (EWEIGHT=1, NEUTRAL_E=1, IDOWN_BETA=1, no IUP here since count_up=0) ----
accip_CoIV = accum_ip(27,3)
print("accum_IP(Co IV)=%.4f eV  (added to internal-jump eweight)"%accip_CoIV)

rows=[]
for (ttype,dst,lid) in blk:
    Z=ln_Z[lid]; ion=ln_ion[lid]
    tau,_,_,_=tau_coiv(lid) if (Z==27 and ion==3) else (None,)*4 if False else (None,None,None,None)
    if Z==27 and ion==3:
        tau,_,_,_=tau_coiv(lid)
    else:
        tau=1e-100
    beta=beta_sob(tau)
    lam=C/ln_nu[lid]*1e8
    # kp_glo = line lower level (global)
    glo_gi=g_of.get((Z,ion,ln_lo[lid]),-1)
    if ttype==-1:
        rate=ln_Aul[lid]*beta
        rate*=H*ln_nu[lid]              # eweight emission: h nu
        chan="emit"
    elif ttype==0:
        rate=ln_Aul[lid]*beta           # IDOWN_BETA=1
        if glo_gi>=0:
            e_low=lev_E[glo_gi]+accip_CoIV
            rate*=e_low*EV
        else:
            rate=0.0
        chan="idown"
    else:
        rate=0.0; chan="iup"            # count_up=0 -> none expected
    rows.append((chan,ttype,lid,lam,tau,beta,ln_Aul[lid],ln_flu[lid],dst,rate))

tot=sum(x[9] for x in rows)
print("\nsum_rates(block)=%.6g   (n_emit=%d n_idown=%d)"%(
    tot, sum(1 for x in rows if x[0]=="emit"), sum(1 for x in rows if x[0]=="idown")))

# probabilities
rows2=[(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[9]/tot if tot>0 else 0.0) for x in rows]
rows2.sort(key=lambda z:-z[10])

print("\nTOP 15 EXIT CHANNELS from Co IV level 144 (upper of 1526.17A):")
print("%-6s %5s %10s %9s %9s %8s %6s %s"%("chan","lid","lam_A","tau","beta","p","dst","note"))
for x in rows2[:15]:
    note=""
    if 1490<=x[3]<=1650: note="<< in 1490-1650 pile"
    print("%-6s %5d %10.2f %9.3g %9.3f %8.4f %6d %s"%(x[0],x[2],x[3],x[4],x[5],x[10],x[8],note))

# P(re-emit into 1490-1650 complex) via EMISSION channels
p_emit_pile=sum(x[10] for x in rows2 if x[0]=="emit" and 1490<=x[3]<=1650)
p_emit_all =sum(x[10] for x in rows2 if x[0]=="emit")
p_idown_all=sum(x[10] for x in rows2 if x[0]=="idown")
p_emit_pile_frac_of_emit = p_emit_pile/p_emit_all if p_emit_all>0 else 0
# emission wavelength buckets
def bucket(lam):
    if lam<912: return "EUV<912"
    if lam<1290: return "FUV912-1290"
    if lam<1490: return "1290-1490"
    if lam<=1650: return "PILE1490-1650"
    if lam<2000: return "1650-2000"
    if lam<4500: return "2000-4500"
    return ">4500"
from collections import defaultdict
emit_by_band=defaultdict(float)
for x in rows2:
    if x[0]=="emit": emit_by_band[bucket(x[3])]+=x[10]
print("\nP(emission) total = %.4f ; P(internal-down) total = %.4f"%(p_emit_all,p_idown_all))
print("P(emit into 1490-1650 pile) = %.4f  (= %.1f%% of all EMISSION)"%(
    p_emit_pile, 100*p_emit_pile_frac_of_emit))
print("Emission probability by band:")
for b in ["EUV<912","FUV912-1290","1290-1490","PILE1490-1650","1650-2000","2000-4500",">4500"]:
    print("   %-14s %.4f"%(b, emit_by_band[b]))

# ---- Step 3: collisional vs radiative for level 144 ----
# kp_deact = sum over emission (ttype==-1) transitions of C_down (van Regemorter form)
# radiative sum_rates uses the BARE rate (A_ul*beta) w/o eweight for the p_kpacket
# competition? NO: in code p_kpacket uses sum_rates (the EWEIGHTED rates) + kp_deact.
# Compute both the bare radiative rate and kp_deact for transparency.
gup144=lev_g[G]
kp_deact=0.0
bare_rad=0.0
for (ttype,dst,lid) in blk:
    Z=ln_Z[lid]; ion=ln_ion[lid]
    tau,_,_,_=tau_coiv(lid) if (Z==27 and ion==3) else (1e-100,0,0,0)
    beta=beta_sob(tau)
    if ttype==-1:
        f_lu=ln_flu[lid]
        gup=gup144
        if f_lu>1e-10:
            C_down=VAN_REG_COEFF*n_e*f_lu*0.2*inv_sqrt_Te/gup
        else:
            C_down=8.63e-6*n_e*AX_OMEGA*inv_sqrt_Te/gup
        kp_deact+=C_down
        bare_rad+=ln_Aul[lid]*beta
    elif ttype==0:
        bare_rad+=ln_Aul[lid]*beta
print("\n=== Step 3: collisional vs radiative, Co IV level 144 (g=%d) ==="%gup144)
print("Sum C_down (collisional deexcitation)  = %.4g s^-1"%kp_deact)
print("Sum A_ul*beta (bare radiative)         = %.4g s^-1"%bare_rad)
print("p_kpacket ~ C/(C+R_bare)               = %.4e"%(kp_deact/(kp_deact+bare_rad)))
# the single 1526 line
C1526=VAN_REG_COEFF*n_e*ln_flu[391357]*0.2*inv_sqrt_Te/gup144
print("1526 line alone: C_down=%.4g  A_ul*beta=%.4g  ratio C/Ab=%.3e"%(
    C1526, ln_Aul[391357]*beta_sob(t1526), C1526/(ln_Aul[391357]*beta_sob(t1526))))

# save channel table
with open(os.path.join(OUT,"level144_exit_channels.csv"),"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["chan","ttype","line_id","lam_A","tau","beta","A_ul","f_lu","dest_global","rate","prob"])
    for x in rows2: w.writerow(x)
print("\nwrote level144_exit_channels.csv")
