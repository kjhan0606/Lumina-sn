#!/usr/bin/env python3
"""층 1 재측정 #2a — 덱 내부 A_ul 항등식 (자기완결).
line_list.csv 의 A_ul 이 그 자신의 f_lu, wavelength, levels.csv 의 g 로부터
표준 관계식으로 재현되는가.  같은 파일 안의 값들이므로 기대는 ULP 급이다.
  A_ul = (8 pi^2 e^2)/(m_e c lambda^2) * (g_l/g_u) * f_lu
표준 수치형: A_ul = 6.6702e15 * (g_l/g_u) * f_lu / lambda_A^2  (문헌값 대신
물리상수에서 직접 조립해 자릿수 가정을 피한다)
"""
import numpy as np, pandas as pd, json, sys
D='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos'
E_ESU=4.80320471257e-10; ME=9.1093837015e-28; C=2.99792458e10
PRE=8*np.pi**2*E_ESU**2/(ME*C)          # * f_lu * (g_l/g_u) / lambda_cm^2

lv=pd.read_csv(f'{D}/levels.csv',usecols=['atomic_number','ion_number','level_number','g'])
key=(lv.atomic_number.values.astype(np.int64)<<40)|(lv.ion_number.values.astype(np.int64)<<32)|lv.level_number.values.astype(np.int64)
gmap=pd.Series(lv.g.values.astype(np.float64),index=key)

ll=pd.read_csv(f'{D}/line_list.csv',
    usecols=['atomic_number','ion_number','level_number_lower','level_number_upper',
             'wavelength_cm','f_lu','A_ul'])
kl=(ll.atomic_number.values.astype(np.int64)<<40)|(ll.ion_number.values.astype(np.int64)<<32)|ll.level_number_lower.values.astype(np.int64)
ku=(ll.atomic_number.values.astype(np.int64)<<40)|(ll.ion_number.values.astype(np.int64)<<32)|ll.level_number_upper.values.astype(np.int64)
gl=gmap.reindex(kl).values; gu=gmap.reindex(ku).values
lam=ll.wavelength_cm.values.astype(np.float64); f=ll.f_lu.values.astype(np.float64)
A=ll.A_ul.values.astype(np.float64)
ok=np.isfinite(gl)&np.isfinite(gu)&(gu>0)&(lam>0)
pred=np.full(A.shape,np.nan); pred[ok]=PRE*f[ok]*(gl[ok]/gu[ok])/lam[ok]**2
both=ok&np.isfinite(A)&(A>0)&(pred>0)
rel=np.full(A.shape,np.nan); rel[both]=np.abs(pred[both]-A[both])/np.maximum(np.abs(A[both]),np.abs(pred[both]))
r=rel[both]
out={'schema':'lumina-layer1-remeasure-02a-v1','deck':D,
 'total_lines':int(len(A)),'unmatched_level_g':int((~ok).sum()),
 'compared':int(both.sum()),
 'relative_error':{'max':float(np.nanmax(r)) if r.size else None,
   'median':float(np.nanmedian(r)) if r.size else None,
   'p95':float(np.nanpercentile(r,95)) if r.size else None,
   'p99.9':float(np.nanpercentile(r,99.9)) if r.size else None},
 'counts_over_threshold':{t:int((r>float(t)).sum()) for t in ['1e-12','1e-9','1e-6','1e-3','1e-1']},
 'formula':'A_ul = 8*pi^2*e^2/(m_e*c*lambda_cm^2) * (g_l/g_u) * f_lu',
 'constants':{'e_esu':E_ESU,'m_e_g':ME,'c_cms':C,'prefactor':PRE}}
print(json.dumps(out,indent=1))
