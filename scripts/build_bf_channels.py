#!/usr/bin/env python3
"""봉인 덱의 선택 이온에 실제 A/B 단면적과 목적 준위를 연결한다.

CBFC v2는 준위 없는 최상위 이온의 단일 경계 상태도 보존한다. X-ray 추가,
dielectronic line profile은 아직 별개이며 전체 물리 완비로 표시하지 않는다.
선택 실행은 기존 v1을 유지한다. 원본을 수정하지 않는다.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess
import numpy as np
from cmfgen_parser import parse_osc, parse_f_to_s
from cmfgen_photo_routes import read_photo_tables, resolve_photo_routes, level_photo_channels
import expand_atomic_data_cmfgen as baker

EV_HZ=1.602176634e-12/6.62607015e-27

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def control(path):
    values={}
    for line in path.read_text().splitlines():
        m=re.match(r'^\s*(\S+)\s+\[(\w+)\]',line)
        if m:values[m[2]]=m[1]
    return (float(values['SIG_GAU_KMS'].replace('D','E')),
            float(values.get('FRAC_SIG','0.25').replace('D','E')),
            float(values.get('CUT_ACC','0.02').replace('D','E')),
            values.get('ABV_EDGE','T').upper() in ('T','.TRUE.','TRUE'))

def smooth_entries(table,smoother,settings):
    width,spacing,accuracy,above=settings
    prior=table.data.smoothing_kms
    # RDPHOT_GEN_V2: 헤더가 없거나 이미 0.9*요청 폭 이상이면 재평활화하지 않는다.
    if prior is None or prior>=.9*width:return table
    if not np.isfinite(prior) or prior<0:raise ValueError('invalid previous smoothing width')
    width=np.sqrt(width*width-prior*prior)
    entries=[]
    for entry in table.data.entries:
        if entry.cs_type<20 or not np.any(entry.sigma_Mb):
            entries.append(entry);continue
        text=f'{len(entry.energy)} {width:.17g} {spacing:.17g} {accuracy:.17g} {int(above)}\n'
        text+=''.join(f'{x:.17g} {y:.17g}\n' for x,y in zip(entry.energy,entry.sigma_Mb))
        result=subprocess.run([str(smoother)],input=text,text=True,capture_output=True,check=True,timeout=20)
        lines=result.stdout.splitlines()
        if not lines or not lines[0].startswith('SMOOTHED '):
            raise ValueError(f'smoothing failed for {entry.config}: {result.stdout[:500]} {result.stderr[:500]}')
        count=int(lines[0].split()[1]);data=np.array([[float(x) for x in row.split()] for row in lines[1:]])
        if data.shape!=(count,2) or not np.all(np.isfinite(data)) or np.any(data[:,1]<0):
            raise ValueError('invalid smoothed data')
        entries.append(replace(entry,energy=data[:,0],sigma_Mb=data[:,1],n_points=count))
    return replace(table,data=replace(table.data,entries=entries,smoothing_kms=settings[0]))

def strict_model(entry,threshold,z,zion,nef_threshold):
    """원본 SUB_PHOT_GEN의 tab20/21 선형 보간/nu^-3 꼬리. 알려진 가짜 fit 금지."""
    p=np.asarray(entry.sigma_Mb,dtype=float);typ=entry.cs_type
    if not np.all(np.isfinite(p)):raise ValueError('nonfinite photo coefficients')
    if typ in (1,7) and p.size>=1 and p[0]==0:
        return lambda x:np.zeros(np.shape(x)),np.array([threshold]),threshold,'explicit_zero'
    if typ in (20,21):
        x=np.asarray(entry.energy,dtype=float)*threshold
        if len(x)<2 or len(x)!=len(p) or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0) or np.any(p<0):
            raise ValueError('invalid tabulated cross section')
        def fn(nu):
            nu=np.asarray(nu);out=np.zeros(nu.shape);active=nu>=threshold;f=nu[active]
            y=np.interp(f,x,p)
            below=f<x[0];y[below]=p[0]+(p[1]-p[0])/(x[1]-x[0])*(f[below]-x[0])
            tail=f>=x[-1];y[tail]=p[-1]*(x[-1]/f[tail])**3
            out[active]=y*1e-18;return out
        return fn,x,threshold,'tabulated_linear_nu_minus3'
    if typ in (2,3,8,9):
        model=baker._sigma_hydrogenic_exact(typ,p,threshold,zion,baker._cmfgen_nef(z,zion,nef_threshold))
    elif typ in (1,7):model=baker._sigma_model(typ,entry.energy,p,threshold)
    else:raise ValueError(f'unsupported SUB_PHOT_GEN cross-section type {typ}')
    if model is None or model[3]=='standin':raise ValueError(f'invalid physical fit {entry.config} type={typ}')
    return model

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--deck',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--control',type=Path,required=True);ap.add_argument('--smoother',type=Path,required=True)
    ap.add_argument('--hyd-data',type=Path,required=True)
    selection=ap.add_mutually_exclusive_group(required=True)
    selection.add_argument('--ion',action='append',help='Z:charge, e.g. 14:1')
    selection.add_argument('--all-active',action='store_true')
    args=ap.parse_args();deck=args.deck.resolve();output=args.output.resolve()
    terminal_path=output.with_suffix('.terminal.csv')
    if any(p.exists() for p in (output,output.with_suffix('.json'),terminal_path)):raise ValueError('output already exists')
    wanted={tuple(map(int,x.split(':'))) for x in (args.ion or [])}
    provenance=json.loads((deck/'DECK_PROVENANCE.json').read_text())
    links=Path(provenance['cmfgen_links'])
    if digest(links)!=provenance['cmfgen_links_sha256']:raise ValueError('changed atomic links')
    sources={}
    for line in links.read_text().splitlines():
        fields=shlex.split(line,comments=True)
        if len(fields)==4 and fields[:2]==['ln','-sf']:sources[fields[3]]=Path(fields[2])
    with (deck/'active_ions.csv').open() as f:active=list(csv.DictReader(f))
    with (deck/'levels.csv').open() as f:levels=list(csv.DictReader(f))
    with (deck/'ionization_energies.csv').open() as f:ionization={(int(r['atomic_number']),int(r['ion_number'])):float(r['ionization_energy_eV']) for r in csv.DictReader(f)}
    ions={(int(r['atomic_number']),int(r['ion_number'])):r for r in active}
    if args.all_active:wanted=set(ions)
    if not wanted<=ions.keys():raise ValueError('selected ion absent')
    index={};rows={}
    for i,r in enumerate(levels):
        key=(int(r['atomic_number']),int(r['ion_number']));rows.setdefault(key,[]).append(i)
        index[key+(int(r['level_number']),)]=i
    hashes={str(p):digest(p) for p in (links,deck/'active_ions.csv',deck/'levels.csv',deck/'ionization_energies.csv',args.control,args.smoother)}
    baker._HYD_BF_DIR=args.hyd_data.resolve();baker._HYD_BF=None
    for name in ('hyd_l_data.dat','gbf_n_data.dat'):
        p=args.hyd_data/name;hashes[str(p)]=digest(p)
    grid=provenance['frequency_grid'];nb=int(grid['n_freq_bins']);lo=float(grid['nu_min_hz']);hi=float(grid['nu_max_hz'])
    edges=lo*np.exp(np.arange(nb+1)*np.log(hi/lo)/nb);dnu=np.diff(edges)
    collected={};counts={};terminals={};settings=control(args.control)
    for key in sorted(wanted):
        upperkey=(key[0],key[1]+1)
        lowerrow,upperrow=ions[key],ions.get(upperkey)
        for row in (lowerrow,upperrow) if upperrow else (lowerrow,):
            for field in ('osc','f_to_s'):
                path=Path(row[field+'_path']);h=digest(path)
                if h!=row[field+'_sha256']:raise ValueError(f'changed source {path}')
                hashes[str(path)]=h
        lower=parse_osc(Path(lowerrow['osc_path']));upper=parse_osc(Path(upperrow['osc_path'])) if upperrow else None
        original=read_photo_tables(sources['PHOT'+lowerrow['model_label']+'_A'])
        for table in original:hashes[str(table.path)]=digest(table.path)
        tables=tuple(smooth_entries(t,args.smoother.resolve(),settings) for t in original)
        routes=resolve_photo_routes(tables,upper,parse_f_to_s(Path(upperrow['f_to_s_path'])).sl_of_fl if upperrow else None)
        if upper is None:
            if len(routes)!=1 or routes[0].excitation_cm1!=0:
                raise ValueError(f'unsupported multiple/excited terminal state {key}')
            terminals[upperkey]=dict(g=routes[0].statistical_weight,label=tables[0].data.final_state)
        first_route={r.table:r for r in reversed(routes)} # NEF는 같은 파일의 첫 경로로 정의된다.
        channels=level_photo_channels(lower,tables,routes,int(lowerrow['n_full']))
        for ch in channels:
            l=index[key+(ch.lower_level,)];u=index[upperkey+(ch.route.target_level,)] if upper else -1
            if levels[l]['configuration']!=str(lower.levels['config'][ch.lower_level]) or (upper and levels[u]['configuration']!=str(upper.levels['config'][ch.route.target_level])):
                raise ValueError('level identity mismatch')
            gu=float(levels[u]['g']) if upper else ch.route.statistical_weight
            excitation=float(levels[u]['energy_eV']) if upper else 0
            threshold=(ionization[key]+excitation-float(levels[l]['energy_eV']))*EV_HZ
            if abs(threshold/(ch.threshold_eV*EV_HZ)-1)>1e-8:raise ValueError('threshold identity mismatch')
            first=first_route[ch.route.table]
            nef_edge=(lower.ionization_eV+1.239841984e-4*(first.excitation_cm1-lower.levels['E_cm'][ch.lower_level]))*EV_HZ
            fn,nodes,start,tag=strict_model(ch.entry,threshold,key[0],lower.z_screen,nef_edge)
            sig=baker._bin_average_sigma(fn,nodes,edges,dnu,start)*ch.route.scale
            if not np.all(np.isfinite(sig)) or np.any(sig<0):raise ValueError('negative/nonfinite channel')
            collected.setdefault(l,[]).append((u,gu,threshold,sig,ch.route.table,ch.route.scale,tag))
            counts[tag]=counts.get(tag,0)+1
        print(f'BAKED Z={key[0]} charge={key[1]} levels={lowerrow["n_full"]} channels={len(channels)} terminal={upper is None}',flush=True)
    if args.all_active and len(collected)!=len(levels):raise ValueError('incomplete full-level coverage')
    offsets=[0];targets=[];upper_g=[];thresholds=[];sigma=[];metadata=[]
    for l in range(len(levels)):
        for u,gu,t,s,table,scale,tag in collected.get(l,[]):
            targets.append(u);upper_g.append(gu);thresholds.append(t);sigma.append(s)
            metadata.append(dict(lower=l,upper=u,upper_g=gu,table=table,scale=scale,threshold_hz=t,evaluator=tag))
        offsets.append(len(targets))
    if not targets:raise ValueError('no channels')
    for path,h in hashes.items():
        if digest(path)!=h:raise ValueError(f'input changed during bake: {path}')
    version=2 if args.all_active or terminals else 1
    with output.open('xb') as f:
        f.write(struct.pack('<IIiiidd',0x43424643,version,len(levels),len(targets),nb,lo,hi))
        for array,dtype in ((offsets,'<i4'),(targets,'<i4'),(thresholds,'<f8'),(sigma,'<f8')):
            f.write(np.asarray(array,dtype=dtype).tobytes())
        if version==2:f.write(np.asarray(upper_g,dtype='<f8').tobytes())
    if terminals:
        with terminal_path.open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['Z','ion_stage_0based','label','level_index','E_cm-1','g','provenance'])
            for (z,stage),term in sorted(terminals.items()):
                writer.writerow([z,stage,term['label'],0,0,term['g'],'CMFGEN_PHOT_terminal'])
    record=dict(schema=f'CBFC-v{version}',scope='raw photo tables; no added Xray/DR line profiles',
        ions=sorted(wanted),levels=len(levels),covered_levels=len(collected),channels=len(targets),bins=nb,
        terminal_catalog_sha256=digest(terminal_path) if terminals else None,
        smoothing=dict(width_kms=settings[0],fraction=settings[1],cut_accuracy=settings[2],above=settings[3]),
        input_sha256=hashes,output_sha256=digest(output),evaluator_counts=counts,routes=metadata)
    with output.with_suffix('.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('input_sha256','routes')}))

if __name__=='__main__':main()
