#!/usr/bin/env python3
"""Extract benchmark ion fractions at a given epoch+velocity from StaNdaRT ionfrac.
Usage: bench_ionfrac_at.py <file> <epoch_day> <vel_kms>
"""
import sys
fn=sys.argv[1]; ep=float(sys.argv[2]); vel=float(sys.argv[3])
lines=open(fn).read().splitlines()
# find block whose "#TIME:" matches ep closest
blocks=[]  # (time, start_idx)
for i,l in enumerate(lines):
    if l.strip().startswith('#TIME:'):
        try: t=float(l.split(':')[1].split()[0])
        except: continue
        blocks.append((t,i))
# nearest epoch
bt,bi=min(blocks,key=lambda x:abs(x[0]-ep))
# header line (#vel_mid...) then data until next '#' or blank
hdr=None; rows=[]
j=bi+1
while j<len(lines):
    s=lines[j].strip()
    if s.startswith('#vel_mid') or s.startswith('#vel'):
        hdr=s.split(); j+=1; continue
    if s.startswith('#TIME'): break
    if not s: j+=1; continue
    if s.startswith('#'): j+=1; continue
    parts=s.split()
    try: v=float(parts[0])
    except: j+=1; continue
    rows.append((v,[float(x) for x in parts[1:]]))
    j+=1
# nearest velocity
vr=min(rows,key=lambda x:abs(x[0]-vel))
frac=vr[1]
tot=sum(frac)
q=sum(i*f for i,f in enumerate(frac))/tot if tot>0 else float('nan')
print(f"{fn.split('/')[-1]}")
print(f"  epoch match: {bt}d (req {ep})   vel match: {vr[0]:.0f} km/s (req {vel})")
labs=hdr[1:] if hdr else [f"st{i}" for i in range(len(frac))]
print("  "+"  ".join(f"{l}={f:.3f}" for l,f in zip(labs,frac)))
print(f"  <q> = {q:.2f}   f(IV=st3) = {frac[3]/tot if len(frac)>3 else 0:.3f}")
