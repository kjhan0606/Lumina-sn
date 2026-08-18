import csv,struct,math,numpy as np,collections
H=6.62607015e-27;KB=1.380649e-16;C=2.99792458e10;ME=9.1093837015e-28;EV=1.602176634e-12;PI=math.pi
BASE="data/tardis_reference_toy06_19p48d";RUN="logs/coevolve_consume_a10_kx_kpr5"
NU_MIN=1.5e14;NU_MAX=3.0e16;NFB=1000;DLN=math.log(NU_MAX/NU_MIN)/NFB
nu=NU_MIN*np.exp((np.arange(NFB)+0.5)*DLN);dnu=NU_MIN*np.exp((np.arange(NFB)+1)*DLN)-NU_MIN*np.exp(np.arange(NFB)*DLN)
lamA=2.99792458e18/nu
CHI={}
for r in csv.DictReader(open(f"{BASE}/ionization_energies.csv")): CHI[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])*EV
lZ=[];lI=[];lE=[];lg=[];ln=[]
for r in csv.DictReader(open(f"{BASE}/levels.csv")):
    lZ.append(int(r['atomic_number']));lI.append(int(r['ion_number']));lE.append(float(r['energy_eV'])*EV);lg.append(float(r['g']));ln.append(int(r['level_number']))
lZ=np.array(lZ);lI=np.array(lI);lE=np.array(lE);lg=np.array(lg);ln=np.array(ln)
with open(f"{BASE}/cmfgen_sigma_bf.bin",'rb') as f:
    struct.unpack('<IIii',f.read(16));struct.unpack('<dd',f.read(16))
    fl=np.frombuffer(f.read(len(lZ)),np.int8);pad=(8-(len(lZ)%8))%8;f.read(pad)
    sig=np.frombuffer(f.read(len(lZ)*NFB*8),np.float64).reshape(len(lZ),NFB)
hs=fl.astype(bool)
mc=np.full((50,NFB),1e-30);cs=np.full((50,NFB),1e-30)
with open(f"{RUN}/lumina_coevolve_field.csv") as f:
    rd=csv.reader(f);next(rd)
    for row in rd:
        s=int(row[0]);b=int(row[1]);cs[s,b]=float(row[3]);mc[s,b]=float(row[4])
gph=np.where(mc>0,mc,cs)
s=8;Z,ion=26,2;chi0=CHI[(Z,ion)]
idx=np.where((lZ==Z)&(lI==ion)&hs)[0]
J=gph[s]
# per-level R (ioniz) and threshold band
def contrib(T):
    kT=KB*T;U=float(np.sum(np.where(lE[idx]/kT<50,lg[idx]*np.exp(-np.minimum(lE[idx]/kT,50)),0)))
    gnd=0.0;exc=0.0;euv=0.0;opt=0.0
    for gl in idx:
        chil=chi0-lE[gl]
        if chil<=0:continue
        nu_th=chil/H;m=(nu>=nu_th)&(sig[gl]>0)&(J>0)
        if not m.any():continue
        R=float(np.sum(4*PI*sig[gl][m]*J[m]/(H*nu[m])*dnu[m]))
        x=lE[gl]/kT
        if x>=50:continue
        w=lg[gl]*math.exp(-x)/U;G=w*R
        if ln[gl]==0:gnd+=G
        else:exc+=G
        # band: where does this level ionize (threshold wavelength)
        lam_th=2.99792458e18/nu_th
        if lam_th<=912:euv+=G
        else:opt+=G
    return gnd,exc,euv,opt,gnd+exc
print(f"s8 Fe III Gph decomposition (kpr5 field). r34_crit for f(IV)=0.022 is Gph~7e-4/s; alpha~3.3e-11, ne=9.8e8")
print(f"{'T':>7} {'Gph_tot':>10} {'ground':>10} {'excited':>10} {'EUV(<912A)':>11} {'opt(>912A)':>11} {'gnd/tot':>7}")
for T in (12208,10383,9000):
    gnd,exc,euv,opt,tot=contrib(T)
    print(f"{T:>7} {tot:>10.3e} {gnd:>10.3e} {exc:>10.3e} {euv:>11.3e} {opt:>11.3e} {gnd/tot:>7.3f}")
