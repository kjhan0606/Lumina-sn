#!/usr/bin/env python3
"""층 1 재측정 #2b — 원본 osc 앵커링.
세 가지를 분리해 잰다:
 (1) deck f_lu  vs  source f      : 임포트 충실도 (기대 exact)
 (2) deck A_ul  vs  source A      : 구 감사가 잰 것
 (3) source A   vs  source f 유래 A: **원본 자신의 내부 무모순** = 잡음 바닥
(3)이 (2)의 하한이다. (3)이 크면 (2)의 불일치는 Lumina 결함이 아니다.
"""
import sys,json,numpy as np,pandas as pd
sys.path.insert(0,'/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts')
from cmfgen_parser import parse_osc
D='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos'
E_DECK=4.80320425e-10; ME=9.1093837015e-28; C=2.99792458e10
PRE=8*np.pi**2*E_DECK**2/(ME*C)     # 덱이 실제로 쓴 상수 (#2a 에서 역산)
IONS=[('Fe III',26,2,'FE/III/19apr23/osc_data'),('Co III',27,2,'COB/III/18oct00/coiii_osc.dat'),
      ('Co IV',27,3,'COB/IV/18oct00/coiv_osc.dat'),('S II',16,1,'SUL/II/19apr23/osc_data')]
lv=pd.read_csv(f'{D}/levels.csv',usecols=['atomic_number','ion_number','level_number','g'])
ll=pd.read_csv(f'{D}/line_list.csv',usecols=['atomic_number','ion_number','level_number_lower','level_number_upper','wavelength_cm','f_lu','A_ul'])
def stat(x):
    x=x[np.isfinite(x)]
    return None if x.size==0 else {'n':int(x.size),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),
        'max':float(x.max()),'over_1e-6':int((x>1e-6).sum()),'over_1e-4':int((x>1e-4).sum()),'over_1e-3':int((x>1e-3).sum())}
out={'schema':'lumina-layer1-remeasure-02b-v1','deck_prefactor_e':E_DECK,'ions':{}}
for name,Z,ion,rel in IONS:
    try: o=parse_osc(f'/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/{rel}')
    except Exception as ex: out['ions'][name]={'error':str(ex)[:80]}; continue
    t=o.transitions
    sub=ll[(ll.atomic_number==Z)&(ll.ion_number==ion)]
    if len(sub)==0: out['ions'][name]={'error':'no deck lines'}; continue
    # deck level_number 기준 확인: osc i,j 는 1-based
    off=1 if sub.level_number_lower.min()==0 else 0
    key_d=(sub.level_number_lower.values+off)*100000+(sub.level_number_upper.values+off)
    key_s=t['i'].astype(np.int64)*100000+t['j'].astype(np.int64)
    dm=pd.Series(np.arange(len(sub)),index=key_d)
    idx=dm.reindex(key_s).values
    m=np.isfinite(idx); idx=idx[m].astype(int)
    fs=t['f'][m].astype(float); As=t['A'][m].astype(float); lamA=t['lam_A'][m].astype(float)
    fd=sub.f_lu.values[idx]; Ad=sub.A_ul.values[idx]
    gl=lv[(lv.atomic_number==Z)&(lv.ion_number==ion)].set_index('level_number').g
    gi=gl.reindex(t['i'][m]-off).values.astype(float); gj=gl.reindex(t['j'][m]-off).values.astype(float)
    lam_cm=lamA*1e-8
    A_from_f = PRE*fs*(gi/gj)/lam_cm**2
    r1=np.abs(fd-fs)/np.maximum(np.abs(fs),np.abs(fd))
    r2=np.abs(Ad-As)/np.maximum(np.abs(As),np.abs(Ad))
    r3=np.abs(A_from_f-As)/np.maximum(np.abs(As),np.abs(A_from_f))
    out['ions'][name]={'source':rel,'source_transitions':int(len(t['i'])),'matched':int(m.sum()),
      'deck_lines':int(len(sub)),
      '1_deck_f_vs_source_f':stat(r1),'2_deck_A_vs_source_A':stat(r2),'3_source_A_vs_source_f_derived_A':stat(r3)}
print(json.dumps(out,indent=1))
