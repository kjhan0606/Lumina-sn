import csv,os,sys
D="data/standart_data1/toy06"
elems={'ca':20,'co':27,'fe':26,'ni':28,'o':8,'s':16,'si':14}
def blk(p,tt=19.48):
  out={};cur=None;rows=[]
  for ln in open(p):
    s=ln.strip()
    if s.startswith("#TIME:"):
      if cur is not None: out[cur]=rows
      cur=float(s.split(":")[1]); rows=[]
    elif s.startswith("#"): continue
    elif s and cur is not None:
      try: rows.append([float(x) for x in s.split()])
      except: pass
  out[cur]=rows; return sorted(out[min(out,key=lambda x:abs(x-tt))])
g={int(x['shell_id']):(float(x['v_inner'])+float(x['v_outer']))/2/1e5 for x in csv.DictReader(open("data/tardis_reference_toy06_19p48d/geometry.csv"))}
# Lumina ion pops
d={}
for x in csv.DictReader(open("lumina_ion_pops.csv")):
  s=int(x['shell_id']);Z=int(x['Z']);st=int(x['stage']);n=float(x['n_ion'])
  d.setdefault((s,Z),{})[st]=n
def lmean(s,Z):
  fe=d.get((s,Z),{}); tot=sum(fe.values())
  if tot<=0: return None,None
  frac=[fe.get(st,0)/tot for st in range(8)]
  return sum(st*frac[st] for st in range(8)), frac
# per element CMFGEN mean ion + Lumina mean ion at outer shells
shells=[25,30,32,33,36,40,49]
print(f"{'elem':>4} " + "".join(f"v{g[s]:5.0f}(L/C)".rjust(14) for s in shells))
print("mean ionization stage <ion>: L=Lumina C=CMFGEN, DIFF=L-C (>0 = Lumina OVER-ionized)")
overall={}
for e,Z in elems.items():
  cmf=blk(f"{D}/ionfrac_{e}_toy06_cmfgen.txt")
  def cmean(vv):
    row=None
    for i in range(len(cmf)-1):
      if cmf[i][0]<=vv<=cmf[i+1][0]: row=cmf[i][1:]; break
    if row is None: row=cmf[-1][1:]
    return sum(i*row[i] for i in range(len(row)))
  line=f"{e.upper():>4} "
  diffs=[]
  for s in shells:
    vv=g[s]; lm,_=lmean(s,Z); cm=cmean(vv)
    if lm is None: line+=f"{'--':>14}"; continue
    dif=lm-cm; diffs.append(dif)
    line+=f"{lm:5.2f}/{cm:.2f}({dif:+.2f})".rjust(14)
  print(line)
  if diffs: overall[e.upper()]=sum(diffs)/len(diffs)
print("\n=== OVER-IONIZATION RANKING (mean <ion> excess L-C over outer shells) ===")
for e,v in sorted(overall.items(),key=lambda x:-x[1]):
  tag="  <== OVER-IONIZED" if v>0.15 else ("  (under)" if v<-0.15 else "")
  print(f"  {e:>3}: {v:+.2f}{tag}")
