| 항목 | 판정(확인·반박·미결) | 실측 근거(명령과 출력) | 수정 요구 |
|---|---|---|---|
| 1. `SN_HYDRO_DATA` 파싱·블록 경계·순서 | 확인 | E1: 독립 파서에서 17개 블록 모두 정확히 700값. `Velocity`는 `35975→1025 km/s`, 간격 `-50`, 원본의 `[1000,36000]` 선택 700행을 뒤집은 값과 전부 일치. 운전석의 `np.argsort`는 이를 inner→outer 오름차순으로 올바르게 복원한다. | 없음. 다만 700점이 zone-centre임을 명시할 것. |
| 2. 속도·밀도 단위 | 확인 | E2: 생성기에서 `v_kms*1e5` 후 `geometry.csv`에 기록. Lumina 구조체도 `[cm/s]`. `r=v×1,683,072 s`는 상대오차 0. `density.csv:rho`와 SN 블록 모두 `g/cm^3`; raw 중심 보간 시 max 0.070026%, median 0.045561%. | §2.7에 `geometry: cm s^-1`, 두 밀도 모두 `g cm^-3`를 명시할 것. |
| 3. 원소·동위원소 매핑 | 확인 | E3: CMFGEN 소스가 `SIL/SUL/CAL/IRON/COB/NICK = Z 14/16/20/26/27/28`로 정의. `IRON/NICK` 원소 블록은 총 원소 질량분율이고 A=56 블록은 그 안의 동위원소 분해이지 별도 추가 질량이 아니다. 이 toy06은 stable IGE=0이어서 Fe·Co·Ni 원소 블록과 A56 블록이 수치상 정확히 같다. 실제 CMFGEN 출력에서도 `element−A56 max_abs=0`. | “IRON=안정 Fe”라고 쓰면 안 된다. “총 Fe이며, 이 모델에서는 stable IGE=0이라 전량 Fe56”으로 수정. 동위원소를 원소에 더하지 않는다고 명시할 것. |
| 4. 보간 방식의 강건성 | 반박 | E4: raw 700점 중심 선형보간에서는 3.33%지만, 독립 finite-volume `ρ dV` 평균에서는 `major max=60.457% (Co,s11)`, `s4 IME=6.97%`, `Co(s11)=1.31285e-3`. 순수 체적평균도 `major max=56.95%`. 더구나 CMFGEN 소스는 조성을 `log R`에서 내부 90-depth 격자로 다시 보간한다. 실제 `SN_HYDRO_FOR_NEXT_MODEL` 90점 조성을 공통 중심에 대조하면 `major=67.13%`, `max|ΔX|=2.3915e-3`; 질량가중하면 `major=78.08%`. | 중심점 결과를 물리적으로 강건한 상한처럼 사용하지 말 것. raw-centre, finite-volume `ρdV`, 실제 CMFGEN 90-depth 결과를 별도 표로 제시할 것. |
| 5. 제시된 다섯 수치 재현 | 확인 | E5: **raw 700점·선형-v·완전피복 셸 0–43**이라는 한정에서 `3.33416996%`, `1.0378781270e-3`, `63.21013345%`, floor `108`, 밀도 `0.070026446% / median 0.045561400%`로 재현. 전체 50셸에 같은 외삽을 적용하면 floor `126`, 밀도 max `74.53%`이므로 44셸 한정이 필수다. CMFGEN 실제 90-depth 조성 수치는 이와 다르다. | 다섯 수치마다 “700점 원입력의 중심보간, 완전피복 44셸만”을 붙일 것. 이를 “CMFGEN 내부에서 실제 사용된 90-depth 조성”이라 부르지 말 것. |
| 6. “조성은 5배 복사장 차를 만들 수 없으므로 원인 풀에서 제거” | 미결 | E6: s11 중심 IGE는 `Lumina=6.54066e-4`, `CMF=4.00750e-4`, 비 `1.6321`; 질량가중 Co는 이미 `1e-3`을 넘는다. IGE line forest의 국소 선불투명도·포화·ionization feedback에 대한 대조나 조성만 바꾼 통제 실행은 없다. 저장소 문서도 post-run 지표가 pending이라고 기록한다. | 원인 풀 제거와 “지난 코퍼스 유효” 결론을 철회하고 미결로 돌릴 것. 동일 geometry·density·T/ne seed에서 composition-only A/B 또는 line-opacity 기여/Jacobian 측정이 필요하다. |
| 7. Temperature / Electron density / Kappa / Sigma 누락 | 반박 | E7: CMF 입력 seed/Lumina deck seed의 완전피복 44셸 비는 `n_e: min 1.7066, median 1.9348, max 3.1470`; `T_CMF/(0.9T_rad): min 1.8332, median 2.2644, max 5.3016`. 특히 `n_e` 중앙비 1.9348은 캠페인 1.92배와 사실상 같다. `Kappa=σ_T n_e/ρ`는 `5.75e-9` 상대오차로 성립하지만 `rd_sn_data.f`에는 Kappa 판독 분기가 없어 CMFGEN이 먹지 않는다. Sigma는 전부 0이고 `PURE_HUB=T`에서 다시 0으로 설정된다. | T/ne seed 차이를 반드시 기록하고, seed와 수렴 후 값을 분리 대조할 것. Kappa는 “미대조 입력”이 아니라 “파일에 있으나 reader가 무시”로, Sigma는 “0 및 Hubble에서 재설정”으로 정정할 것. |

## E1. 블록 파싱과 outer→inner 순서

실행 명령:

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/SN_HYDRO_DATA')
L=p.read_text().splitlines()
labels=[]; counts=[]; current=None; n=0
for line in L[8:]:
    s=line.strip()
    if not s:
        continue
    try:
        vals=[float(x.replace('D','E')) for x in s.split()]
    except ValueError:
        if current is not None:
            labels.append(current); counts.append(n)
        current=s; n=0
    else:
        n += len(vals)
labels.append(current); counts.append(n)
for i,(lab,c) in enumerate(zip(labels,counts),1):
    print(f'{i:02d} {lab}: {c}')
print(f'blocks={len(labels)} unique_labels={len(set(labels))} all_700={all(c==700 for c in counts)}')
PY
```

출력:

```text
01 Radius grid (10^10cm): 700
02 Velocity (km/s): 700
03 Sigma (dlnV/dlnr-1): 700
04 Temperature (10^4 K): 700
05 Density (g/cm^3): 700
06 Atom density (/cm^3): 700
07 Electron density (/cm^3): 700
08 Kappa (cm^2/gm): 700
09 SIL mass fraction: 700
10 SUL mass fraction: 700
11 CAL mass fraction: 700
12 IRON mass fraction: 700
13 COB mass fraction: 700
14 NICK mass fraction: 700
15 NICK 56 mass fraction: 700
16 COB 56 mass fraction: 700
17 IRON 56 mass fraction: 700
blocks=17 unique_labels=17 all_700=True
```

원본 순서 교차검증 출력:

```text
velocity first,last=35975.0,1025.0
strictly_decreasing=True
dv_unique=[-50.0]
raw_rows=807 selected=700
raw_v_ascending=True
selected_matches_SN_v=True
max_abs_v=0.0e+00
```

따라서 파일은 outer→inner이고, 운전석의 다음 처리는 정확하다.

```python
order = np.argsort(v_cmf)
v_cmf = v_cmf[order]
```

운전석 블록 파서는 비수치 행을 새 라벨로 보고 다음 라벨 직전의 수치를 저장한다. 실제 데이터 구간에는 17개 라벨 외 비수치 행이 없고 `comp_diff.py:35-36`의 700값 assert도 전부 통과하므로 경계 오인은 없다.

## E2. 단위

실행 명령:

```bash
nl -ba scripts/build_toy06_epoch.py | sed -n '89,102p;143,150p;182,187p'
nl -ba src/lumina.h | sed -n '195,202p;353,359p'
head -n 3 data/tardis_reference_toy06_19p48d/geometry.csv
head -n 3 data/tardis_reference_toy06_19p48d/density.csv
```

핵심 출력:

```text
93     v = v_kms * 1e5                     # cm/s, invariant
96     rho_m = d[:, 10]                    # density @ t_model
101    r = v * (target_epoch_d * DAY)      # R = v t [cm]
149    rho_s = np.interp(v_cen, v, rho)
183    {"r_inner": r_edge[:-1], "r_outer": r_edge[1:],
184     "v_inner": v_edge[:-1], "v_outer": v_edge[1:]}
186    {"shell_id": sid, "rho": rho_s}

double *v_inner;   /* inner velocities [cm/s] */
double *v_outer;   /* outer velocities [cm/s] */
double *rho;       /* density [g/cm^3] */

shell_id,r_inner,r_outer,v_inner,v_outer
0,656398080000000.0,778925721600000.0,390000000.0,462800000.0

shell_id,rho
0,1.5687692791189745e-13
```

수치 identity:

```text
time_explosion_s=1683072.0
max_rel_abs(r-v*t)/r=0.000e+00
```

따라서 `geometry.csv`에 `1e-5`를 곱해 km/s로 바꾸는 것은 맞다. `density.csv`와 `SN_HYDRO_DATA`의 `Density (g/cm^3)`도 같은 단위다.

## E3. 원소와 동위원소 의미

CMFGEN 자체 원소 정의 확인 명령:

```bash
rg -n "AT_NO\\(ID\\)=14|AT_NO\\(ID\\)=16|AT_NO\\(ID\\)=20|AT_NO\\(ID\\)=26|AT_NO\\(ID\\)=27|AT_NO\\(ID\\)=28|SPECIES\\(ID\\)=" \
  /gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen.f
```

관련 출력:

```text
201: AT_NO(ID)=14.0 ... !Silicon
202: SPECIES(ID)='SIL'
211: AT_NO(ID)=16.0 ... !Sulpher
212: SPECIES(ID)='SUL'
231: AT_NO(ID)=20.0 ... !Calcium
232: SPECIES(ID)='CAL'
261: AT_NO(ID)=26.0 ... !Iron
262: SPECIES(ID)='IRON'
266: AT_NO(ID)=27.0 ... !Cobalt
267: SPECIES(ID)='COB'
271: AT_NO(ID)=28.0 ... !Nickel
272: SPECIES(ID)='NICK'
```

원본 StaNdaRT 헤더:

```bash
sed -n '44,52p' /gpfs/kjhan/cmfgen_runs/toy06_19.48d/snia_toy06_19.48d.dat
```

출력:

```text
# (14) X_Ni includes X_56Ni
# (15) X_Co = X_56Co
# (16) X_Fe includes 56Fe from 56Co decay
#idx ... X_56Ni X_Ni X_Co X_Fe X_Ca X_S X_Si ...
```

같은 파일은 다음도 선언한다.

```text
M(stable IGE) = 0.0000e+00 Msun
```

`mk_sn_hydro.py`는 원소 블록을 `X_Ni/X_Co/X_Fe`에서 만들고, 동위원소 블록을 별도로 기록한다. CMFGEN reader는 둘을 더하지 않는다. 기본 `USE_OLD_MF_SCALING=.FALSE.` 경로에서 동위원소 합과 원소 총량의 일관성을 검사한 뒤 부모 원소 population을 동위원소 합으로 대체한다.

```fortran
WRK=WRK+ISO(IS)%OLD_POP
...
T2=(POP_SPECIES(...)+T1)/(WRK(I)+T1)-1.0
...
POP_SPECIES(:,PAR(IP)%ISPEC)=WRK
```

실측 명령과 출력:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np
p=Path('/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/SN_HYDRO_DATA')
L=p.read_text().splitlines()
def b(s):
    a=[]
    for z in L[L.index(s)+1:]:
        if not z.strip(): continue
        try: a += list(map(float,z.split()))
        except ValueError: break
    return np.array(a)
for k in ('IRON','COB','NICK'):
    print(k, f'max_abs(element-A56)={np.max(abs(b(k+" mass fraction")-b(k+" 56 mass fraction"))):.3e}')
PY
```

```text
IRON max_abs(element-A56)=0.000e+00
COB max_abs(element-A56)=0.000e+00
NICK max_abs(element-A56)=0.000e+00
```

실제 CMFGEN의 90-depth 출력도 같다.

```text
IRON 90 90 max_abs(element-A56)=0.000e+00
COB  90 90 max_abs(element-A56)=0.000e+00
NICK 90 90 max_abs(element-A56)=0.000e+00
```

즉 Fe/Co/Ni 대조 자체의 원소 선택은 맞고 이중계산도 없었다. 단, 이 toy06에서만 stable IGE가 0이므로 총 Fe가 전량 Fe56인 것이다.

## E4. 보간·평균 방식

CMFGEN reader의 실제 보간 소스:

```bash
nl -ba /gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/rd_sn_data.f | sed -n '260,307p'
```

출력:

```text
260 ! interpolate from the HYDRO grid to the CMFGEN grid
266 LOG_R_HYDRO=LOG(R_HYDRO)
267 LOG_R=LOG(R)
269 T_HYDRO=LOG(T_HYDRO)
270 CALL MON_INTERP(T,...LOG_R,...T_HYDRO,...LOG_R_HYDRO,...)
272 ELEC_DEN_HYDRO=LOG(ELEC_DEN_HYDRO)
273 CALL MON_INTERP(ED,...)
281 WRK_HYDRO=LOG(DENSITY_HYDRO)
282 CALL MON_INTERP(DENSITY,...)
292 LIN_INTERP_RD_SN_DATA=.TRUE.
303 CALL LIN_INTERP(LOG_R,POP_SPECIES(...),...,LOG_R_HYDRO,POP_HYDRO(...),NX)
```

따라서 운전석의 선형-v 중심 보간은 CMFGEN의 실제 `linear X versus log R`와도 정확히 같지 않다. 미세격자라 중심 결과 차이는 작지만 0은 아니다.

```text
linear_v:
  major=3.33416996%
  abs=1.0378781270e-03
  trace=63.21013345%

linear_logv:
  major=3.33290741%
  abs=1.0374967413e-03
  trace=63.22965198%
```

독립 finite-volume 계산은 각 50 km/s 원자료 셀과 Lumina 셸의 구면 교차체적을 구하고 다음을 적용했다.

\[
X_s={\sum_j \rho_j X_j (v_{\rm hi}^3-v_{\rm lo}^3)\over
          \sum_j \rho_j (v_{\rm hi}^3-v_{\rm lo}^3)}
\]

출력:

```text
finite-volume rho*dV:
  major=60.456975% COB s11
  abs=2.648043210e-03 COB s8
  trace=100.000000% SIL s3

s4:
  SIL=2.40414133e-02 rel=6.968%
  SUL=1.52992026e-02 rel=6.968%
  CAL=4.37117782e-03 rel=6.969%
  COB=7.59017146e-01 rel=0.319%

s11:
  IRON=1.62038088e-04 rel=60.457%
  COB=1.31285070e-03 rel=60.457%
  NICK=1.79173958e-04 rel=60.457%

rho finite-volume:
  max=0.472388% s0
  median=0.090473%
```

밀도 없이 순수 구면체적 평균한 결과도 유의하게 다르다.

```text
major=56.9496067% COB s11
max_abs=4.15942913e-03 COB s5
s4 IME relative difference=11.158–11.159%
s11 Co=1.20589113e-03, relative difference=56.950%
```

더 중요한 것은 CMFGEN이 700점을 직접 계산격자로 쓰지 않는다는 점이다. 실제 출력:

```bash
sed -n '1,6p' /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/SN_HYDRO_FOR_NEXT_MODEL
```

```text
Number of data points:           90
Number of mass fractions:        28
Number of isotopes:               3
Time(days) since explosion:      19.4800000
```

이 90-depth 조성을 Lumina 중심에 다시 대조한 결과:

```text
CMF internal/output grid:
n=90 v=1024.971005..35975.288045

center:
  major=67.130722% COB s11
  max_abs=2.391483212e-03 COB s4
  trace=67.131700% IRON s11
  floor=105

mass-weighted:
  major=78.076927% COB s11
  max_abs=4.767972733e-03 COB s8
  trace=100.000000% SIL s3
  floor=102
```

따라서 3.33%는 raw 700점의 중심표본 비교값이지, CMFGEN 내부 유효 조성의 강건한 최대차가 아니다.

## E5. 다섯 수치의 독립 재현

독립 파서로 각 원소를 Lumina 셸 중심에 선형-v 보간하고, 완전피복 셸 `0..43`만 선택한 출력:

```text
major_max_rel=3.33416996%
  SIL s4 v=7176.0
  lum=2.2366093052e-02
  cmf=2.1644430938e-02

max_abs=1.0378781270e-03
  COB s4
  lum=7.6143624853e-01
  cmf=7.6247412666e-01

trace_max_rel=63.21013345%
  COB s11 v=12272.0
  lum=5.1914087574e-04
  cmf=3.1808127644e-04

floor_pairs=108

rho:
  max_rel=0.070026446% s40 v=33384.0
  lum=6.7141443866e-18
  cmf=6.7094460000e-18
  median=0.045561400%
```

따라서 반올림하면 운전석의 다섯 수치는 모두 재현된다.

하지만 범위 조건을 제거하면:

```text
all50 floor_pairs=126
rho first50:
  max_rel=74.534852345% s49
  median=0.050939861%
```

이는 `np.interp`가 무피복 외곽에서 끝값을 반복하기 때문이다. 따라서 §2.7의 다섯 수치는 모두 “완전피복 44셸만”이라는 조건에 종속된다.

## E6. 물리 결론

실측:

```text
s11 center IGE:
  Lumina=6.5406609798e-04
  CMF=4.0074966425e-04
  ratio=1.632106
  absdiff=2.5331643372e-04

s11 Co:
  Lumina=5.1914087574e-04
  CMF=3.1808127644e-04
  ratio=1.632101
```

여기에 finite-volume 결과에서는 CMF Co가 `1.31285e-3`까지 올라가 “미량” 범위를 벗어난다. Fe-group 원소는 작은 질량분율이라도 UV/EUV의 조밀한 선군을 지배할 수 있으며, 선불투명도는 질량분율에 단순 선형으로 비례하지 않는다. 포화, ionization balance, 재방출과 온도 피드백도 있다.

현재 자료에는 다음이 없다.

- 동일 geometry/density/T/ne에서 조성만 바꾼 통제 실행
- s11 Co/Fe/Ni의 band별 또는 line별 불투명도 기여
- 조성 변화에 대한 \(J_\nu\), \(u\), ion fraction의 응답계수
- 수렴 후 조성-only A/B 결과

실제로 `docs/TOY06_CMFGENCOMP_COMPOSITION.md`도 다음과 같이 기록한다.

```text
Spectra, ionization, and temperature are expected to change globally.
Post-run before/after metrics remain pending.
```

따라서 “3%가 5배를 만들 수 없다”와 “조성을 원인 풀에서 제거”는 현재 증거로 확인할 수 없다. 특히 s11 IGE 꼬리의 국소 선불투명도 영향은 **미결**이다.

## E7. 미대조 블록과 전자밀도

CMFGEN reader와 Lumina의 seed 소비 경로 확인 명령:

```bash
rg -n "Radius grid|Velocity|Sigma|Temperature|Density|Electron density|Atom density|Kappa|ass fraction" \
  /gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/rd_sn_data.f
rg -n "SN_T_OPT|PURE_HUB" /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/VADAT
nl -ba src/lumina_main.c | sed -n '143,150p'
nl -ba src/lumina_plasma.c | sed -n '2424,2434p;2541,2549p'
```

핵심 출력:

```text
rd_sn_data.f:
  Radius grid
  Velocity
  Sigma
  Temperature
  Density
  Electron density
  Atom density
  ass fraction
  [Kappa 분기 없음]

VADAT:
T            [PURE_HUB]
USE_HYDRO    [SN_T_OPT]   !T,Ne from SN_HYDRO_DATA

lumina_main.c:
143 /* Initialize n_electron from TARDIS reference */
144 plasma.n_electron = malloc(...)
146 plasma.n_electron[i] = opacity.electron_density[i];

lumina_plasma.c:
2430 double n_e = plasma->n_electron[s];
2543 /* damped update */
2544 n_e = 0.5 * (n_e_new + n_e_old);
2545 plasma->n_electron[s] = n_e;
```

독립 입력 비교는 CMFGEN reader와 같이 log값을 log-radius에서 보간했다.

```text
CMF_ne/Lumina_deck_ne full44:
  min=1.706572
  median=1.934786
  max=3.146986

ne s0  v=4264:
  CMF=5.074389e+09
  Lum=1.612460e+09
  ratio=3.146986

ne s8  v=10088:
  CMF=7.422317e+08
  Lum=3.729929e+08
  ratio=1.989935

ne s11 v=12272:
  CMF=3.370817e+08
  Lum=1.960231e+08
  ratio=1.719601
```

Temperature 비교는 Lumina의 최초 `T_e=0.9 T_rad`와 CMFGEN hydro T seed를 비교했다.

```text
CMF_Tseed/Lumina_Te_seed full44:
  min=1.833168
  median=2.264424
  max=5.301594

T s0:  CMF=18760 K, Lumina=9423 K, ratio=1.990867
T s8:  CMF=10383 K, Lumina=5664 K, ratio=1.833168
T s43: CMF=15846 K, Lumina=2989 K, ratio=5.301594
```

Kappa와 Sigma:

```text
Kappa identity:
  max_rel |Kappa - sigma_T*ne/rho| / Kappa = 5.749e-09
  CMF range=1.784996e-02..3.197618e-02 cm^2/g

CMF_Kappa/Lumina_seed_es_kappa full44:
  min=1.707651
  median=1.935601
  max=3.147293

Sigma unique=[0.0]
Lumina max_rel_abs(r-v*t)/r=0.000e+00
```

Kappa 블록은 생성기 내부 일관성은 있지만 CMFGEN reader가 소비하지 않는다. Sigma는 입력부터 0이며 `PURE_HUB` 경로가 다시 0으로 만든다. 반면 T와 \(n_e\)는 실제 초기조건으로 소비되고 양쪽 모두 이후 재계산된다. 그러므로 seed 차이와 수렴 후 차이를 분리해서 검증해야 한다. 특히 입력 \(n_e\) 중앙비 `1.9348`을 확인하지 않은 채 캠페인 `1.92×` 불일치를 조성 밖의 문제라고 단정할 수 없다.

**반려**