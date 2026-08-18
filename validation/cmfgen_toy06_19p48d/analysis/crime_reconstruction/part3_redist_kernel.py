#!/usr/bin/env python3
"""PART 3 measurement 2 -- measured redistribution kernel R(nu_in -> nu_out).
Pair each line-absorption (etype 1) with the SAME packet's NEXT emission
(etype 2/4/5) by chaining events by pkt_id in file order.

VALIDITY of packet-id chaining: the event ring buffer is filled by a global
monotonic atomicAdd(&d_event_count) (src/lumina_cuda.cu:2835); one packet is
followed by ONE GPU thread through its whole history, so successive d_event_record
calls for a given pkt_id land at strictly increasing file indices -> within a
pkt_id, file order == causal order. pkt_id is the packet loop index, unique within
one iteration; both logs are a single iteration (iter=11). => chaining is valid.
Method: stable-sort by pkt_id (preserves file order within id); forward-fill the
"governing absorption" band within each id (reset at id boundary); every emission's
governing absorption = its source absorption. Build (entry band x exit band) matrix
for absorptions at s0-2 (deep) and s7-8 (control), all-ion and Co-IV-entry only.
BLIND SPOTS: CAP128M saturation drops events -> some chains broken (unpaired
emissions/absorptions counted + reported); etype 8 bf-reemit unlogged -> bf-abs(3)
chains have no logged exit (so entry uses line-abs etype 1 only). Read-only.
"""
import numpy as np, csv, os
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/crime_reconstruction"
EV=np.dtype([('pkt_id','<u4'),('line_id','<i4'),('nu','<f4'),('energy','<f4'),
             ('etype','u1'),('shell','u1'),('iter','u1'),('pad','u1')])
LINE=np.dtype([('lam','<f4'),('Z','<u2'),('ion','<u2')])
C_A=2.99792458e18
EDGES=[100,300,450,918,1290,1490,1650,2100,4500,20000,1e12]
BLAB=['100-300','300-450','450-918','918-1290','1290-1490','1490-1650_CPLX',
      '1650-2100_VLY','2100-4500','4500-20000','>20000']
NB=len(BLAB)

def band_of(lam):
    b=np.digitize(lam,EDGES)-1
    b[(b<0)]= -1; b[b>=NB]=NB-1
    return b

def analyze(run):
    path=f"{REPO}/logs/coevolve_consume_a10_kx_{run}"
    mm=np.memmap(f"{path}/lumina_events.bin",dtype=EV,mode='r',offset=32)
    N=len(mm)
    pid=np.array(mm['pkt_id']); et=np.array(mm['etype']); sh=np.array(mm['shell'])
    nu=np.array(mm['nu']); lid=np.array(mm['line_id']); del mm
    lam=np.where(nu>0,C_A/nu,0.0); band=band_of(lam)
    # line ion table
    with open(f"{path}/lumina_events_lines.bin","rb") as f:
        assert f.read(8)==b'LUMLIN01'; lr=np.frombuffer(f.read(),dtype=LINE)
    Lz=lr['Z'].astype(np.int32); Lion=lr['ion'].astype(np.int32)

    idx=np.arange(N)
    order=np.argsort(pid,kind='stable')      # file order preserved within id
    pid_s=pid[order]; et_s=et[order]; sh_s=sh[order]; band_s=band[order]
    lid_s=lid[order]; orig=idx[order]
    is_abs =(et_s==1)                        # line absorption = macro-atom activation
    is_emit=np.isin(et_s,(2,4,5))            # any emission/exit
    # group starts (sorted by id -> contiguous)
    newg=np.empty(N,bool); newg[0]=True; newg[1:]=pid_s[1:]!=pid_s[:-1]
    posn=np.arange(N)
    gstart=np.where(newg,posn,-1); gstart=np.maximum.accumulate(gstart)
    # running index of most-recent absorption; invalidate if it precedes group start
    abspos=np.where(is_abs,posn,-1); runabs=np.maximum.accumulate(abspos)
    valid=runabs>=gstart
    gov=np.where(valid,runabs,-1)            # governing-absorption sorted-position per record
    # emissions with a valid governing absorption in the same packet+group
    sel=is_emit&(gov>=0)&(gov!=posn)
    gpos=gov[sel]
    in_band=band_s[gpos]; in_sh=sh_s[gpos]; in_lid=lid_s[gpos]
    out_band=band_s[sel]
    # matrices
    def matrix(shmask, coiv_only=False):
        m=shmask.copy()
        if coiv_only:
            good=(in_lid>=0)
            coiv=np.zeros(len(in_lid),bool)
            coiv[good]=(Lz[in_lid[good]]==27)&(Lion[in_lid[good]]==3)
            m=m&coiv
        M=np.zeros((NB,NB))
        ib=in_band[m]; ob=out_band[m]
        ok=(ib>=0)&(ob>=0)
        np.add.at(M,(ib[ok],ob[ok]),1.0)
        return M
    res={}
    for gname,shells in [('s0-2',[0,1,2]),('s7-8',[7,8])]:
        shmask=np.isin(in_sh,shells)
        res[(gname,'all')]=matrix(shmask,False)
        res[(gname,'coiv')]=matrix(shmask,True)
    # coverage
    npair=int(sel.sum()); nemit=int(is_emit.sum()); nabs=int(is_abs.sum())
    unpaired_emit=int((is_emit&~((gov>=0)&(gov!=posn))).sum())
    return res,dict(N=N,npair=npair,nemit=nemit,nabs=nabs,unpaired_emit=unpaired_emit)

def print_matrix(M,title):
    print(f"\n{title}  (rows=ENTRY band, cols=EXIT band; row-normalized %)")
    tot=M.sum()
    print(f"  total paired={int(tot):,}")
    # print row-normalized for the key entry rows
    hdr="  "+f"{'ENTRY\\EXIT':16}"+"".join(f"{b.split('_')[0][:7]:>8}" for b in BLAB)
    print(hdr)
    for i in range(NB):
        rs=M[i].sum()
        if rs<=0: continue
        row="  "+f"{BLAB[i]:16}"+"".join(f"{100*M[i,j]/rs:>8.1f}" for j in range(NB))
        print(row+f"   (n={int(rs):,})")

allcsv=[]
for run in ['gphall','stage4']:
    print("\n"+"#"*100); print(f"# RUN {run}"); print("#"*100)
    res,cov=analyze(run)
    print(f"coverage: N={cov['N']:,} line-abs={cov['nabs']:,} emit={cov['nemit']:,} "
          f"PAIRED={cov['npair']:,} unpaired_emit={cov['unpaired_emit']:,} "
          f"(pair rate={100*cov['npair']/max(cov['nemit'],1):.1f}% of emissions)")
    for gname in ['s0-2','s7-8']:
        print_matrix(res[(gname,'all')], f"[{run} {gname}] ALL-ION redistribution")
        print_matrix(res[(gname,'coiv')], f"[{run} {gname}] Co IV-ENTRY redistribution")
        for kind in ['all','coiv']:
            M=res[(gname,kind)]
            for i in range(NB):
                for j in range(NB):
                    if M[i,j]>0: allcsv.append([run,gname,kind,BLAB[i],BLAB[j],int(M[i,j])])
with open(f"{OUT}/part3_redist_kernel.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(['run','group','kind','entry_band','exit_band','count']); w.writerows(allcsv)
print(f"\n[out] {OUT}/part3_redist_kernel.csv")
