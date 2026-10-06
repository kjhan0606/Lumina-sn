# 사전등록 — DET-SIIII-POP: Si III 소비 준위 인구 관계와 공개 CMFGEN 이온 분율 대조

2026-10-07 · 기안  
정본 신설 예정: `docs/PREREG_DET_SIIII_POP_2026-10-07.md`  
상태: **사전감사 전 기안 · 실행 승인 판정 아님**

## §0. 목표·승인·주장 상한

### Q-GOAL — 최종목적에 직접 기여하는가?

선행 DET-OBS-LINEDECOMP는 봉인된 E′·R′의 출력 차이에서 Si III 1892.03 Å와 2542.58 Å의 매핑 선 주입이 주요 기여임을 확인했다. 그러나 그 결과만으로 원천함수를 정하는 **이온 내부 준위 분포**와 선 광학깊이를 정하는 **인구의 절대 규모**를 구별할 수 없다.

이 단의 목표는 다음을 한 묶음으로 얻는 것이다.

1. 주요 셸에서 두 전이의 **세대 7 소비값**으로 상·하 준위비와 복원 가능한 절대 준위밀도를 구한다.
2. **세대 8 저장값**으로 이온 내부 분포와 Si III 총수를 분리하여 보고하고, 세대 7 소비값과의 차이를 측정한다.
3. 공개 CMFGEN의 같은 에폭 자료와 속도를 맞춰 Si III 이온 분율·전자온도·전자밀도를 대조한다.
4. 자료로 식별되는 차이와 식별되지 않는 차이를 구분하여 후속 조사 결정을 돕는다.

이 단을 생략하면 세대가 다른 인구를 소비값의 원인으로 오인하거나, 이온 총수 변화와 준위비 변화를 혼동하여 수리 후보를 선택하게 된다.

**세대 7 전체 Si III 인구와 공개 CMFGEN 준위 인구는 현재 지정 입력에 없다.** 따라서 원래 질문의 완전한 해소를 미리 약속하지 않는다. 이 결손 때문에 남는 질문은 완료 판정에서 명시한다.

**기안자 자기답: PASS.** 사전감사자의 결론은 별도로 기록한다.

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

표적은 두 전이와 셸 17–30이다. 기존 캡처·최종 인구표·봉인 원자자료·공개 표만 읽는다.

추가하지 않는 작업은 다음과 같다.

- GPU 생산 런, 인구·율·복사장·온도 재계산.
- 새 수송 replay, 선행 광도 분해 재실행.
- 전체 원소의 인구 지도, 전이별 율 예산, 반복별 dump.
- 두 선에서 전체 Si III 준위계를 추정하는 역문제.
- 다른 코드들의 전수 비교와 새 CMFGEN 계산.
- 생산 코드 수정과 파일시스템 특수 코드.

새 검사는 **자료의 대응·권위 확인**과 **복원·분해·대조 산술 확인**으로 묶는다. 모든 산출은 준위구조 해석, 세대 구분 또는 CMFGEN 대조에 직접 소비한다.

**기안자 자기답: APPROPRIATE.**

### 승인 근거와 상태 정의

승인 근거:

`/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/evidence/USER_DECISION_2026-10-07.txt`

user 2026-10-07 「둘을 묶어서 진행」에 따라 준위 인구 관계 조사와 공개 CMFGEN 대조를 함께 기안한다.

E′·R′는 선행 계약의 NRB 비활성·활성 상태이며, 등록된 FIXED_TE·6회 반복 구성의 유한 응답이다. 새로운 수렴 상태나 NRB의 독립적인 인과효과로 재정의하지 않는다.

### 주장 상한

허용하는 결론은 **고정된 상태의 인구 관계와 대수적 귀속**, 그리고 **지정 공개 표에 대한 근접도**다.

다음은 판정하지 않는다.

- 준위 분포 차이를 만든 개별 방사·충돌·재결합·비열적 율의 원인.
- 생산 인구나 원천함수의 물리적 정오.
- NRB 생산 승격, 물리 수리 또는 새 계산 승인.
- 공개 CMFGEN 스펙트럼 잔차의 전이별 원인이나 잔차 감소.
- 세대 8의 일치를 근거로 한 세대 7 이온 총수의 동일성.
- 공개되지 않은 CMFGEN 준위 인구의 일치.

## §1. 기준선·입력·보존

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
L = /gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007
D = /gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007
T2 = C/source

EC = C/Eprime/run1
RC = C/Rprime/run1

새 도구 = D/tools
새 산출 = D/results/<실행 식별자>
새 집행 증거 = D/evidence
```

### 1.1 승계하는 판정

선행 폐합 기준은 DET-OBS-LINEDECOMP `feafa21`과 다음 기록이다.

- `docs/PREREG_DET_OBS_LINEDECOMP_2026-10-07.md`
- `docs/VERDICT_DET_OBS_LINEDECOMP_2026-10-07.md`
- `L/astra_closeout.md`, `L/astra_closeout2.md`, `L/astra_closeout3.md`
- `docs/VERDICT_DET_OBS_CAPTURE_2026-10-07.md`
- `docs/VERDICT_DET_OBS_WRITER_2026-10-02.md`

광도 분해 정본은 `L/results/f3/{E,R}/pairs.bin` 및 `L/results/f3/paired_b3/`다. 이 단에서는 표적 선정 근거로 연결하며 H·광도를 다시 계산하지 않는다.

캡처 무결성·소비 coverage·scalar 재생·선 목록 행 대응에 관한 선행 PASS를 승계한다. 새로 읽는 자료와 표적의 대응은 확인하되 선행 전체 시험을 반복하지 않는다.

### 1.2 읽을 입력

| 입력 | 경로·권위 |
|---|---|
| 소비값 | 각 런 `work/det_obs_capture/lines.bin` |
| 소비 세대·형식·fallback | 같은 디렉터리 `manifest.json`, `meta.txt` |
| 소비 물질·기하 | 같은 디렉터리 `Te.f64`, `ne.f64`, `r_inner.f64`, `r_outer.f64` |
| 전이·준위 신원 | 각 런 `input/model/line_list.csv`, `levels.csv` |
| 원소 총수·속도 | 같은 model 디렉터리의 `density.csv`, `abundances.csv`, `atom_masses.csv`, `geometry.csv`, `config.json` |
| 입력 봉인·생산 설정 | 각 런 `input/deck.sha256`, `input/resolved_lumina.exports`, `input/run_build_identity.json` |
| 최종 NLTE 인구 | 각 런 `work/nlte_levels_final.csv` |
| 최종 별도 이온 배열 | 각 런 `work/lumina_ion_pops.csv` |
| 최종 전자 상태·발행 신원 | 각 런 `work/lumina_plasma_state.csv`, `work/cmfgen_final_publication.json` |
| 공개 이온 분율 | `data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt` |
| 공개 물질 상태 | `data/standart_data1/toy06/phys_toy06_cmfgen.txt` |

캡처 manifest는 `nlte_population_committed=7`을 기록한다. `tau_required=tau_computed=14`는 opacity 갱신 표식이며 인구 세대 14로 읽지 않는다.

최종 발행 manifest는 인구 세대 8, 온도 세대 7을 기록한다. 온도 세대가 같다는 이유로 인구나 전자밀도의 동일성을 추정하지 않는다.

### 1.3 신원·보존

운전자는 소비한 입력의 경로·봉인 신원·도구 신원·실행 명령을 새 manifest에 연결한다. 새로 사용하는 최종 CSV와 공개 표는 실제 파일 해시를 기록한다. 이미 검증된 대형 캡처를 별도 목적으로 다시 전체 해시하거나 복제하지 않는다.

기존 입력·소스·실패 결과는 보존한다. 수정 후 실행에는 새 식별자를 사용하고 기존 결과를 덮어쓰지 않는다. D가 승인문 때문에 이미 존재하는 것은 허용한다.

## §2. 표적과 자료 대응

### 2.1 표적

| 전이 | 0-based 선 목록 데이터 행 | 원소·이온 | 하위→상위 준위 | 주요 셸 |
|---|---:|---|---|---|
| Si III 1892.03 Å | 908565 | Z=14, ion=2 | 0→2 | 22–30 |
| Si III 2542.58 Å | 1097363 | Z=14, ion=2 | 4→5 | 17–22 |

선 ID는 CSV 헤더를 제외한 **행 번호**다. CSV의 별도 `line_id` 열로 결합하지 않는다.

파장과 f의 표시값은 표적 설명용이다. 계산에는 봉인된 행의 전체 저장값을 사용한다. `ion=2`는 전하 단계 2, 즉 Si III다. 공개 표에서는 `si2`가 대응한다.

### 2.2 셸 범위와 결손

두 선 모두 셸 17–30을 읽는다. 그중 위 표의 주요 셸에서 각각의 주 결론을 내며, 나머지는 같은 셸에서 두 전이·하위 준위 분포를 비교하는 데만 사용한다. 결과를 본 뒤 주요 셸을 바꾸지 않는다.

각 상태·선·셸의 소비 레코드는 다음을 구분한다.

- 존재하고 `mapped=1`.
- 존재하지만 매핑되지 않음.
- sparse 소비 집합에 없음.

없는 레코드는 τ·S·인구 0이 아니다. 비소비 상태에는 `NOT_CAPTURED`를 기록한다.

준위의 g·E·식별자는 봉인 `levels.csv`와 생산 소스의 준위 매핑을 기준으로 연결한다. 최종 CSV의 `level_idx`가 원자자료의 `level_number`와 항상 같다고 가정하지 않고 `global_idx`·g·E의 대응도 확인한다.

### 2.3 권위 확인

생산 식의 권위는 **현재 작업 트리보다 T2의 실제 생산 판본**이다.

특히 다음을 확인한다.

- 두 상태의 epoch·셸 기하·원자자료 대응.
- 표적의 매핑과 NLTE τ 쓰기 권한.
- τ와 S가 같은 세대의 같은 상·하 준위 인구를 사용했음.
- source fallback이나 별도 변경이 표적 S에 적용되지 않았음.

지정 런의 exports에는 `LUMINA_NLTE_SKIP_Z=""`가 있고, 선행 캡처는 mapped fallback 0건을 확인했다. 이를 생산 함수의 분기와 연결하여 사용한다. `mapped=1`만으로 τ의 생산 권위까지 증명했다고 보지 않는다.

## §3. 세대 7 소비값에서의 복원

### 3.1 생산 소스에 근거한 식

T2 아래의 다음 함수를 권위로 삼는다.

| 소스 | 함수·역할 |
|---|---|
| `src/lumina_plasma.c` | `nlte_tau_line_authority`, `nlte_tau_line_shell_authorized_by`: 매핑·τ 권한 |
| `src/lumina_plasma.c` | `nlte_update_tau_sobolev_with_authority`: 같은 인구에서 τ·S 구성 |
| `src/opacity_publication.c` | `a208_signed_sobolev_counted`: signed Sobolev τ |
| `src/opacity_publication.c` | `a208_line_source_counted`: 인구 원천함수 |
| `src/lumina_cmfgen.c` | `det_obs_capture_pair`: 실제 선택된 `S_used` 저장 |
| `src/lumina.h` | `SOBOLEV_COEFF`, `H_PLANCK`, `C_SPEED_OF_LIGHT` |

해당 생산 판본에서

\[
a=\frac{g_l}{g_u},\qquad
A_\nu=\frac{2h\nu_l^3}{c^2},\qquad
K_l=C_{\rm Sob} f_{lu}\lambda_l t_{\rm exp},
\]

\[
d=n_l-a n_u,\qquad
\tau=K_l d,\qquad
S=A_\nu\frac{a n_u}{d}.
\]

여기서 \(C_{\rm Sob}\)는 소스의 `SOBOLEV_COEFF`다. 새 기본상수로 재계산한 값으로 교체하지 않는다. 파장·주파수 구성 순서도 생산 소스를 따른다. A 계수는 이 역산의 입력으로 사용하지 않는다.

### 3.2 복원량

유효한 인구 원천함수에서

\[
q=\frac{g_l n_u}{g_u n_l}
  =\frac{S}{A_\nu+S},
\qquad
\frac{n_u}{n_l}=\frac{g_u}{g_l}q.
\]

τ와 S의 생산 인구가 같음이 확인되면

\[
d=\frac{\tau}{K_l},
\]

\[
n_l=d\left(1+\frac{S}{A_\nu}\right),
\qquad
n_u=\frac{g_u}{g_l}\,d\frac{S}{A_\nu}.
\]

따라서 표적마다 다음을 산출한다.

- 소비한 τ·\(S_{\rm used}\).
- \(q\), \(n_u/n_l\).
- \(d=n_l-(g_l/g_u)n_u\).
- 복원 \(n_l,n_u\), 단위 \({\rm cm^{-3}}\).
- E′−R′ 차이와 정의 가능한 E′/R′ 비.
- 같은 셸에서 두 선이 모두 복원되면 \(n_4/n_0\).

\(n_4/n_0\)는 서로 다른 하위 준위의 상대 점유를 보여 준다. 전체 Si III 인구로 정규화한 점유율은 아니다.

계산은 \(n_l=d/(1-q)\)보다 위의 덧셈식을 기본으로 하여 \(q\approx1\)에서 불필요한 뺄셈 손실을 피한다. near-singular 결과를 clamp하거나 제한하지 않는다.

### 3.3 복원 적격성과 의미

주 복원은 캡처에서 소비된 양의 τ·양의 인구 원천함수에 적용한다. 비유한값·부적격 분모·권위 불일치·fallback은 인구로 역산하지 않는다.

- S만 인구 원천함수임이 확인되면 **준위비만** 복원할 수 있다.
- τ와 S의 인구 권위가 다르면 두 값을 결합한 절대 준위밀도를 만들지 않는다.
- 비소비 쌍의 τ나 S를 세대 8 값으로 채우지 않는다.
- 복원은 저장 소비값과 생산 산술에 대응하는 인구 추정이다. 원래 인구 배열의 비트 복원을 주장하지 않는다.

역산값을 같은 식에 다시 넣는 검사는 대수·구현 확인이다. 독립된 인구 관측과의 일치 시험으로 표시하지 않는다.

## §4. 세대 8의 사용과 이온 총수

### 4.1 두 인구 권위를 분리한다

T2의 `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/cmfgen_final_writer.c`의 함수 `cmfgen_final_writer`는 다음을 별도로 쓴다.

- `nlte_levels_final.csv:n_pop`: `NLTE_LEVEL`.
- `lumina_ion_pops.csv:n_ion`: `SAHA_STAGE`.
- 준위 CSV의 `n_ion_total`: 위의 별도 이온 배열에서 가져온 값.

따라서 `n_ion_total`을 NLTE 준위합으로 간주하지 않는다.

세대 8의 Si III 총수는 우선

\[
N_{\rm III,NLTE}^{(8)}
 =\sum_{i\in{\rm Si\,III}}n_i^{(8)}
\]

로 정의한다. 해당 이온의 전체 저장 준위를 한 번씩 합산하며 준위 누락·중복을 확인한다.

별도 이온 배열의 Si III 값은

\[
N_{\rm III,stage}^{(8)}
 =n_{\rm ion}^{(8)}(Z=14,\mathrm{stage}=2)
\]

로 보존한다. 두 값의 차이는 보고하며 강제로 일치시키지 않는다.

### 4.2 이온 분율의 분모

Si 원자핵 총수는 봉인 입력에서

\[
N_{\rm Si}
 =\frac{\rho X_{\rm Si}}{A_{\rm Si}m_u}
\]

로 구성한다. 근거는 T2의 `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c`의 함수 `nlte_pair_total_density`에 있는 원소 밀도 식이며 상수 `AMU`는 T2의 `src/lumina.h`를 따른다.

`density.csv`는 생산 로더가 직접 읽는 epoch의 밀도다. `t_model_d`를 보고 추가 팽창 보정을 적용하지 않는다. 운전자는 해당 고정 런에서 Si 조성·밀도의 소비 연결을 확인한다.

주 보고량은

\[
x_{\rm III,NLTE}^{(8)}
 =\frac{N_{\rm III,NLTE}^{(8)}}{N_{\rm Si}},
\qquad
x_{\rm III,stage}^{(8)}
 =\frac{N_{\rm III,stage}^{(8)}}{N_{\rm Si}}.
\]

공개 CMFGEN과의 주 비교에는 **NLTE 준위 인구를 설명하는 첫 번째 값**을 사용한다. 두 번째 값은 별도 이온 배열의 대조로 명확히 표시한다. 더 잘 맞는 쪽을 사후에 주 분율로 선택하지 않는다.

분모가 0이면 분율은 미정의다. 분율이 저장 정밀도를 넘어 [0,1] 밖이면 원값과 `POPULATION_NORMALIZATION_INCONSISTENT`를 남긴다. 잘라내거나 표본 단계합으로 재정규화하지 않는다. 이 경우 해당 NLTE 분율의 물리적 근접도 판정은 보류한다.

### 4.3 세대 8 내부 분해

\[
p_l^{(8)}=\frac{n_l^{(8)}}{N_{\rm III,NLTE}^{(8)}},
\quad
p_u^{(8)}=\frac{n_u^{(8)}}{N_{\rm III,NLTE}^{(8)}},
\quad
w^{(8)}=p_l^{(8)}-\frac{g_l}{g_u}p_u^{(8)}.
\]

세대 8의 양 끝점에서 \(\Delta z=z_E-z_R\), \(\bar z=(z_E+z_R)/2\)라 두면

\[
\Delta n_l
 =\underbrace{\Delta N_{\rm III}\,\bar p_l}_{\text{이온 총수 항}}
 +\underbrace{\bar N_{\rm III}\,\Delta p_l}_{\text{이온 내부 점유 항}},
\]

\[
\Delta d
 =\underbrace{\Delta N_{\rm III}\,\bar w}_{\text{이온 총수 항}}
 +\underbrace{\bar N_{\rm III}\,\Delta w}_{\text{이온 내부 점유·유도방출 항}}.
\]

이 두 항등식으로 하위 준위밀도와 순 흡수 인구 차이를 분해한다. \(N_{\rm Si,E}=N_{\rm Si,R}\)가 확인된 경우 이온 총수 항을 **이온화 분율 항**이라고도 부를 수 있다. 원소 밀도가 다르면 그 명칭을 사용하지 않는다.

### 4.4 세대 7과 세대 8의 관계

세대 8 준위값으로 두 전이의 \(q^{(8)},S^{(8)},d^{(8)},\tau^{(8)}\)를 생산 식에 따라 계산하고 세대 7 소비·복원량과 나란히 둔다.

- 수치 구간이 분리되면 `GENERATION_DIFFERENCE_RESOLVED`.
- 겹치면 `GENERATION_DIFFERENCE_UNRESOLVED`.

후자는 “세대가 동일하다”거나 “반복이 수렴했다”는 뜻이 아니다.

**세대 8 사용 허용 범위는 세대 8 자체의 분해와 공개 자료 대조다.** 두 전이의 S·τ 또는 네 준위가 세대 7과 맞더라도, 나머지 준위와 Si III 총수의 동일성이 보장되지 않는다. 따라서 세대 8 이온 분율·정규화 점유율을 세대 7 값으로 승격하지 않는다.

### 4.5 세대 7 이온 분율의 결손

두 전이에서 얻는 네 준위밀도만으로 전체 Si III 총수는 정해지지 않는다. 현재 입력에서는 다음과 같이 기록한다.

```text
Si III 전체 인구, 세대 7: NOT_AVAILABLE_SAME_GENERATION
Si III 이온 분율, 세대 7: NOT_AVAILABLE_SAME_GENERATION
세대 7 하위 준위밀도의 이온화/정규화 점유 분해: NOT_IDENTIFIABLE
```

네 준위의 합을 전체 이온 총수로 사용하지 않는다. 세대 8 값으로 결손을 메우거나 LTE·Boltzmann 분포를 가정하지 않는다. 이 단에서 전체 준위 복원이나 새 dump를 위한 생산 실행은 하지 않는다.

## §5. 공개 CMFGEN 대조

### 5.1 가능한 대조와 불가능한 대조

주 기준은 저장소에 있는 StaNdaRT toy06 CMFGEN의 **19.48 d 블록**이다.

| 물리량 | 대조 |
|---|---|
| Si III 이온 분율 | 공개 `si2`와 세대 8 분율 |
| \(T_e,n_e\) | 공개 물질 표와 캡처값·최종값을 구분하여 대조 |
| 같은 상·하 준위 인구·점유율·원천함수 | `NOT_AVAILABLE_PUBLIC_LEVEL_POPULATIONS` |
| 세대 7 Si III 전체 분율 | §4.5의 결손 때문에 직접 대조 불가 |

공개 이온 분율·온도·전자밀도로 CMFGEN 준위 인구를 만들어 내지 않는다. 자체 CMFGEN 런은 기준에서 제외한다.

### 5.2 속도 대응

Lumina 셸의 대표 속도는 봉인 기하의 산술 중점으로 고정한다.

\[
v_s=\frac{v_{\rm inner,s}+v_{\rm outer,s}}{2}.
\]

공개 표의 `vel_mid[km/s]`와 단위를 맞춘다. 셸 번호끼리 결합하지 않는다.

공개 19.48 d 블록에서 \(v_s\)를 끼우는 두 속도 행 사이를 **선형 속도·선형 물리량 보간**한다. 정확한 격자점에서는 그 값을 사용한다.

- 이온 표와 물질 표의 속도 격자는 각각 읽는다.
- 시간 보간은 하지 않는다.
- 속도 외삽은 하지 않는다.
- 사용한 양쪽 속도·원값·보간 가중치를 결과에 남긴다.
- 범위 밖은 `OUTSIDE_PUBLIC_VELOCITY_SUPPORT`로 표시한다.

이는 같은 속도에서의 표 비교다. 동일 질량좌표나 동일 광학깊이의 비교가 아니다. 수치 구간은 아래의 저장·산술 정밀도만 포함하며, 선형 보간의 물리적 정확도를 인증하지 않는다.

### 5.3 판정량

셸마다

\[
e_{X,s}
 =\left|x_{{\rm III,NLTE},X}^{(8)}(s)
       -x_{{\rm III,CMFGEN}}(v_s)\right|
\]

를 구한다. 0 분율을 다루기 위해 주 거리에는 로그를 사용하지 않는다.

주요 셸 집합 \(S\)마다

\[
D_X(S)=\frac{1}{|S|}\sum_{s\in S}e_{X,s}
\]

를 보고한다. 집합은 1892.03 Å의 22–30, 2542.58 Å의 17–22로 고정한다. 두 집합을 합쳐 중복 셸을 두 번 세는 종합 점수는 만들지 않는다.

등록된 주요 셸 중 하나라도 주 비교가 부적격이면 전체 집합의 승자를 내지 않는다. 사용 가능한 셸 표는 남기고 결손 셸을 명시한다.

전자온도와 전자밀도는 차이 및 양의 분모에서의 비를 보고한다. 이를 이온 분율 점수에 합산하거나 S 차이의 원인 점수로 사용하지 않는다.

### 5.4 다른 코드의 사용 여부

이 단에서는 다른 코드 분포를 **판정 기준이나 CMFGEN의 오차막대**로 사용하지 않는다. 코드 사이 분산은 독립적인 정답 검증이 아니며, 시간 격자·이온 단계·물리 설정을 추가로 맞춰야 하기 때문이다.

다른 코드 비교는 후속 후보로만 남긴다. 결과를 본 뒤 CMFGEN보다 유리한 기준을 선택하지 않는다.

## §6. 수치 구간과 분기

### 6.1 정밀도 처리

경험적인 상대오차 문턱을 새로 만들지 않는다.

- 캡처 binary64 값은 저장값을 정확히 읽는다.
- 생산 τ·S 구성의 반올림과 새 역산·분해·보간의 반올림을 구분하여 전파한다.
- 생산 인구 복원 구간은 §3의 실제 연산 순서에서 유도한다. 공유 중간량 \(d\)와 뺄셈의 오차를 빠뜨리지 않는다.
- 최종 writer의 `%.6e` 인구·전자밀도에는 마지막 출력 자릿수의 반 단위를 저장 불확실성으로 반영한다.
- 공개 표도 각 수치 토큰의 마지막 표시 자릿수에 따른 구간을 사용한다. 공개 자료의 clipping·floor가 확인된 값은 반올림 오차만으로 원래 물리값을 복원했다고 주장하지 않는다.
- 곱·나눗셈·절댓값·합에는 부호에 맞는 구간 산술을 적용하고 바깥 반올림한다.

연산식·상한의 유도와 구현은 실제 E′/R′ 결과 판독 전에 직접시험 기록에 고정한다. 관측 잔차에 맞춰 허용폭을 늘리지 않는다.

분모가 0을 포함하면 비를 만들지 않는다. 유한한 넓은 구간은 `UNRESOLVED`이며, 비유한 상한이나 잘못된 구간은 계산 부적격이다.

### 6.2 원천함수 차이의 판정

양의 인구 원천함수에서

\[
S=A_\nu\frac{q}{1-q}.
\]

동일 전이의 이온 총수를 일정 배수로 바꾸면서 내부 점유율을 유지하면 q와 S는 변하지 않는다.

따라서 다음처럼 판정한다.

| 결과 | 허용 결론 |
|---|---|
| 세대 7 q 차이가 수치적으로 분리됨 | `EXCITATION_RATIO_DIFFERENCE`: 소비 S 차이에 이온 내부 준위비 차이가 필요함 |
| q 차이가 0과 미분리 | `UNRESOLVED`: 같은 비라는 증명은 아님 |
| S가 인구 원천함수인지 확인 불가 | `SOURCE_POPULATION_LINK_UNAVAILABLE` |
| τ·S의 인구 권위 불일치 | 준위비 결과만 유지하고 절대 인구 결합 복원 보류 |

`EXCITATION_RATIO_DIFFERENCE`는 여기구조의 **상태 차이**를 뜻한다. 이온화 과정이 그 구조를 간접적으로 바꾸지 않았다는 인과적 배제는 아니다.

S 차이를 “이온화 몇 %, 여기 몇 %”로 분해하지 않는다. 같은 이온의 전체 정규화는 S 식에서 소거된다.

### 6.3 세대 8 밀도 차이의 판정

§4.3의 두 항을 부호와 함께 보고한다.

- 한 항의 절댓값 구간 하한이 다른 항의 상한보다 크면 해당 항이 `LARGER`.
- 두 항이 반대 부호로 분리되면 `CANCELLING`을 함께 표시한다.
- 크기 구간이 겹치면 `UNRESOLVED`.
- 총차이가 0과 겹치면 순차이의 방향을 확정하지 않는다.

작은 순차이를 분모로 한 기여율은 만들지 않는다. 이 분기를 세대 7 소비 광도나 선행 \(F_\tau\)의 분해로 재명명하지 않는다.

### 6.4 CMFGEN 근접도 분기

주요 셸 집합마다 다음을 적용한다.

- \(D_E^{hi}<D_R^{lo}\): `E_CLOSER`.
- \(D_R^{hi}<D_E^{lo}\): `R_CLOSER`.
- 그 밖의 유효한 구간: `UNRESOLVED`.
- 자료 결손·정규화 부적격·속도 범위 밖: `NOT_COMPARABLE`과 사유.

결론에는 반드시 **세대 8 NLTE 준위합의 Si III 분율, 등록 속도 보간과 거리 기준에서**라는 조건을 붙인다. 한 상태가 물리적으로 더 옳다거나 스펙트럼이 더 옳다는 뜻이 아니다.

## §7. 변경집합과 필수 산출

### 변경집합 — DET-SIIII-POP (표)

신규 Python 함수는 `det_siiii_pop_` 접두를 사용한다. 아래 파일은 신설 허용분이다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `scripts/det_siiii_pop_readout.py` | `det_siiii_pop_*` | 입력 연결, 선택 판독, 복원·세대 비교·이온 분율·공개 대조, 직접시험, 결과·manifest |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/tools/run_siiii_pop.slurm` | 오프라인 실행 | 환경·자원·명령·종료 상태 기록; §9 집행 조건 충족 후 작성 |

### 변경집합 끝

생산 `src/`, 생산 바이너리, 기존 판독기·replay·입력은 변경하지 않는다. 라이브러리를 import하면 의존 파일 신원을 기록한다.

필수 산출은 다음과 같다.

| 산출 | 내용 |
|---|---|
| `levels_g7.csv` | 상태·선·셸별 소비값, 복원량·구간·적격성·권위 |
| `levels_g8.csv` | 동일 표적의 저장 준위값, 내부 점유율, 세대 비교, 밀도 분해 |
| `ions_g8.csv` | 셸별 NLTE 준위합·별도 stage 총수·원소 총수·분율 |
| `cmfgen_compare.csv` | 속도 결합, 공개 원값·보간값, 상태별 대조·구간 |
| `checks.json` | 새 직접시험·음성대조·대수 폐합·자료 결손 |
| `result.json` | 주요 셸별 과학 분기, 세대·권위·공개자료 한계 |
| `manifest.json` | 입력·도구·환경·명령·산출 신원 |
| 판정문·종료감사 | 원래 질문의 해소 범위、미달 항목、후속 결정 |

결과표의 모든 행은 세대와 인구 권위를 명시한다. 미정의 수치는 JSON `null` 또는 빈 수치칸과 사유로 기록하며 0으로 채우지 않는다.

대형 `lines.bin`은 선택 레코드만 추출한다. 전체 문자열 CSV나 전체 캡처 사본을 만들지 않는다.

## §8. 최소 게이트·직접시험·음성대조

| 게이트 | 막는 실제 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 선·준위·셸·전하 단계 오결합, 세대 8을 7로 오기, NLTE와 stage 권위 혼동, 속도 단위·에폭 오류 | §1–§5의 대응·권위·결손 처리가 입력과 일치 |
| G-PHYS | 유도방출 누락, 잘못된 역산·분해, 저장 정밀도 누락, 미정의 비·근접도 오판 | §3–§6의 독립 작은 정답·구간·항등식과 일치 |

하나의 작은 인공 자료와 필요한 변형으로 시험한다.

1. **인구 정답:** 알려진 양의 \(n_l,n_u,g_l,g_u\)에서 독립 식으로 τ·S를 만들고 복원 구간이 원래 인구를 포함하는지 확인한다. 작은 \(d\) 사례도 포함한다. 유도방출 항 또는 g 비를 제거한 결함은 검출되어야 한다.
2. **구조·정규화 정답:** 모든 준위를 같은 배수로 바꾸면 S는 유지되고 τ는 같은 배수로 바뀌어야 한다. 내부 분포를 바꾸는 사례에서는 q가 변해야 한다. 두 분해항 누락과 부호 반전은 검출되어야 한다.
3. **권위·세대 음성대조:** 선 행 또는 이온 단계를 바꾸거나 세대 8을 세대 7로 선언한 입력, 별도 stage 총수를 NLTE 합으로 대체한 입력을 거부해야 한다.
4. **비교 경계:** 저장 반올림 구간이 겹치는 E/R을 승자로 분류하지 않는다. 0 포함 분모, 공개 속도 범위 밖, 빠진 주요 셸, E/R 교환을 확인한다.

정상 대조에는 동일 상태 비교를 포함한다. 음성대조는 작은 시험 자료에만 적용한다.

선행 수송 폐합을 새 시험으로 반복하지 않는다. 새 역산의 순·역 대입만으로 권위 검사가 통과했다고 처리하지 않는다.

기계 프리플라이트는 기존 도구를 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`

명령·문서 신원·rc를 기록한다. 기계 통과는 Q-GOAL·Q-LEAN 사전감사를 대신하지 않는다.

## §9. 실행 환경·비용·재현

### 9.1 실행 환경과 현재 제약

요청된 실행은 **a10 Slurm의 CPU 전용 오프라인 작업, GPU 요청 없음**이다. 로그인 노드에서는 파일 조회·소스 판독만 한다. 시험·기계 게이트·계산은 로그인 노드에서 실행하지 않는다.

운전자 실측에 따르면 syntax의 Slurm 제출 필터 `/etc/slurm/job_submit.lua`는 a40·h100·a100·a100_pcie·h200 파티션에 GPU 요청을 요구하며, a10은 GPU 요청 없는 CPU 작업을 받는다. 선행 DET-OBS-CAPTURE·DET-OBS-LINEDECOMP의 CPU 시험·판독도 이 경로로 완료됐다.

**a10 Slurm의 GPU 요청 없는 CPU 전용 실행 경로는 운전자 실측으로 확인됐다.** GPU를 임의로 요청하거나 로그인 실행으로 대체하지 않는다. 장소·자원 조건의 계약 변경이 필요하면 astra 개정으로 처리한다.

### 9.2 비용 상한

a10 Slurm의 GPU 요청 없는 CPU 전용 작업에 적용할 등록 한도다.

| 항목 | 상한 |
|---|---:|
| 동시 작업 | 1개 |
| 작업당 CPU | 4개 |
| 작업당 메모리 | 16 GiB |
| 시험·프리플라이트·본 판독·수정 재판독의 누적 할당 경과 | 60분 |
| 누적 CPU 예약량 | 4 core·h |
| GPU 예약·생산 실행 | 0 GPU·h · 0회 |
| 새 도구·시험·결과·로그 합계 | 256 MiB |

이 값은 **이번 기안의 지출 한도**이며 실측 예상시간이 아니다. 선행 캡처 쌍은 약 1.37 GB이고 이번에는 수송 없이 선택 레코드와 CSV를 읽는 작업으로 제한한다. 입력 복제는 하지 않는다.

실패 작업도 누적 비용에 포함한다. 자동 재제출은 하지 않는다. 원인이 특정된 판독기 오류는 한도 안에서 수정·재판독할 수 있다. 자원 확대는 user 결정 사항이다.

러너는 `module purge` 뒤 실제 사용할 모듈을 고정하고 Python·의존성 버전과 스레드 수를 기록한다. 각 명령의 실패를 전파하며 마지막 성공 문구만으로 완료를 판정하지 않는다.

### 9.3 계획 인터페이스

아래는 신설 도구의 계획 CLI이며 허용된 Slurm 작업 안에서만 실행한다.

```bash
python3 -B scripts/det_siiii_pop_readout.py selftest \
  --out "$D/results/$RUN_ID/selftest"

python3 -B scripts/det_siiii_pop_readout.py analyze \
  --e-run "$C/Eprime/run1" \
  --r-run "$C/Rprime/run1" \
  --production-source "$C/source" \
  --prior-paired "$L/results/f3/paired_b3" \
  --public-ion "$P/data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt" \
  --public-phys "$P/data/standart_data1/toy06/phys_toy06_cmfgen.txt" \
  --selftest "$D/results/$RUN_ID/selftest" \
  --out "$D/results/$RUN_ID/analysis"
```

manifest에는 실제 변수값·작업 디렉터리·전체 명령·작업 ID·종료 상태·할당 자원을 기록한다.

CLI 표현의 차이는 집행 기록으로 남길 수 있다. 물리량·세대 사용·분모·보간·수용 기준 변경은 CLI 차이로 처리하지 않는다.

## §10. 완료 기준과 처분

### 10.1 필수 완료 산출

다음을 갖춰 운전자가 판정한다.

1. G-LINK·G-PHYS와 기계 프리플라이트의 결과.
2. 두 전이의 주요 셸별 세대 7 소비 준위비·복원 가능한 절대밀도.
3. 세대 8 NLTE 총수·내부 점유·밀도 분해와 세대 차이.
4. 세대 8 Si III 분율 및 공개 CMFGEN 속도 대조.
5. 공개 준위 인구와 세대 7 이온 총수의 결손 표기.
6. 수치 미분리·자료 결손·권위 불일치의 구분.
7. 비용·보존·분장·이탈·재현 기록.
8. 운전자 판정과 astra 종료감사.

### 10.2 묶음 처분

- **`PASS_LIMITED` — 등록 판독 완료, 원질문 일부 미해소:** 가능한 산출과 수치 분기가 갖춰졌고 결손을 정확히 기록했다. 이 경우에도 **세대 7 절대밀도 변화의 완전한 이온화/점유 분해와 CMFGEN 준위 인구 대조는 미해소**임을 첫 판정 문단에 쓴다.
- **`REWORK`:** 새 도구의 연결·복원·정규화·구간·분기 오류로 등록 산출이 성립하지 않는다.
- **`BLOCKED`:** 필수 입력·생산 권위·실행 환경·자원 때문에 가능한 것으로 등록한 산출도 완료하지 못했다.

현재 지정 자료만으로 원래 두 후보를 완전히 해소한 **무조건부 PASS는 허용하지 않는다.** 자료에 없는 물리량을 새 계산 없이 얻은 것으로 표시하지 않는다.

유효한 계산의 `UNRESOLVED`는 오류가 아니다. 다만 자료 결손이나 환경 차단을 수치 미분리라고 부르지 않는다.

후속 제안에는 남은 질문과 필요한 자료를 특정한다. 제안 자체는 새 dump·율 분석·수리·생산 실행 승인이 아니다.

## §11. 집행 기록·분장 장부

| 업무 | 담당 | 이해관계·규칙 |
|---|---|---|
| 기안·개정·종료감사 | astra | 기안자가 자기 계약에 대한 판정문을 감사한다는 한계 명시 |
| Q-GOAL·Q-LEAN 사전감사 | Fable | 호출 실패 시 gpt-6.1-sol 대행; 실패와 실제 모델 신원 기록 |
| 구현·직접시험·기계 게이트·실행·판독·판정 | 운전자 | 구현자·판정자 겸임 |
| 범위·비용 확대·후속 물리 결정 | user | 현재 승인 밖 결정 |

아래 미집행 항목을 결과 없이 채우지 않는다. 집행 기록으로 계약의 수용 기준을 변경하지 않는다.

| 집행 항목 | 담당 | 상태·증거 |
|---|---|---|
| user 결정 | user | 2026-10-07 「둘을 묶어서 진행」, §0 승인문 |
| 본문 기안 | astra | `gpt-6-astra`, read-only, 본 문서 |
| 정본 파일화·기준선 연결 | 운전자 | ecc8553(기안 + 개정 1·2); 기준선 DET-OBS-LINEDECOMP feafa21 |
| 기계 프리플라이트 | 운전자 | lageunha: prereg·gate_referents·doc_canonical PASS(개정 2 후); 사전감사 발주서 dispatch PASS |
| Q-GOAL·Q-LEAN 사전감사 | Fable 또는 기록된 대행 | Fable 429(`/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/evidence/fable_preaudit_failed.txt`) → gpt-6.1-sol 대행 `/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/preaudit_sol.md`: 부합·적정·차단 0 (발주서 §0 경로 오기 1건, 기재) |
| CPU 전용 실행 환경 확인 | 운전자 | 확인됨: a10에서 GPU 요청 없이 선행 CPU 작업 완료, §9.1 |
| 오프라인 구현·도구 신원 | 운전자 | `scripts/det_siiii_pop_readout.py`, 러너 `/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/tools/run_siiii_pop.slurm`; 판독기 SHA `/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/results/a1/readout.sha256` |
| G-LINK·G-PHYS·음성대조 | 운전자 | 정본 a2: selftest 22/22(실제 분석 경로 시험 포함), 분석 VALID·사유 없음 (`/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/results/a2/`); a1(13/13)은 B1·B2 결함 전 판 |
| 세대 7 복원·세대 8 분해 | 운전자 | `/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/results/a2/analysis/levels_g7.csv`·`levels_g8.csv`·`ions_g8.csv`; 세대 비교 q·S·d·τ |
| 공개 CMFGEN 대조 | 운전자 | `/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/results/a2/analysis/cmfgen_compare.csv`; 두 집합 E_CLOSER |
| 실행 비용·보존·재현 기록 | 운전자 | a10 job 415089(a1)·415091(a2), 각 ~12 s, 4 CPU·16 GB; manifest.json |
| 목표 달성·미달 판정 | 운전자 | PASS_LIMITED `docs/VERDICT_DET_SIIII_POP_2026-10-07.md`; 구현자 겸임 |
| 종료감사 | astra | 1회차 `/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/astra_closeout.md`: 폐합 불가·차단 3(B1 실제 경로 시험, B2 세대 비교·q 상한·속도 정밀도, B3 §6 귀결) → 수리·a2 재판독; 2회차 대기 |
| 이탈·미해결·후속 후보 | 실제 담당 | 판정문 §4(8건)·§6 |

## §12. 기계 선언

아래 분기는 §6.4의 공개 CMFGEN 근접도에 적용한다. `v`는 판독기가 자료 적격성·구간 비교를 끝낸 뒤 내는 단일 코드다.

- `0`: E′가 가까움.
- `1`: R′가 가까움.
- `2`: 유효한 수치 미분리.
- `3`: 자료 결손·지원 범위·정규화 문제로 비교 불가.
- 그 밖·결손·복수 코드: 기계 입력 부적격.

각 결과에는 이유와 비교 세대·권위를 별도로 기록한다. 이 선언은 전체 물리 계산이나 묶음 완료를 검증하는 대체물이 아니다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SIIII-POP (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_siiii_pop_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/tools/run_siiii_pop\\.slurm)",
    "symbol": "det_siiii_pop_",
    "roots": [
      "scripts/det_siiii_pop_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/tools/run_siiii_pop.slurm"
    ],
    "expected_extra": [
      "scripts/det_siiii_pop_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/tools/run_siiii_pop.slurm"
    ]
  },
  "branches": {
    "regimes": [
      [0.0, 0.0],
      [1.0, 1.0],
      [2.0, 2.0],
      [3.0, 3.0],
      [9.0, 9.0]
    ],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2,3) for x in v)",
      "e": "sum(1 for x in v if x == 0)",
      "r": "sum(1 for x in v if x == 1)",
      "u": "sum(1 for x in v if x == 2)",
      "n": "sum(1 for x in v if x == 3)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "E_CLOSER", "predicate": "valid and e == 1"},
      {"name": "R_CLOSER", "predicate": "valid and r == 1"},
      {"name": "UNRESOLVED", "predicate": "valid and u == 1"},
      {"name": "NOT_COMPARABLE", "predicate": "valid and n == 1"}
    ],
    "adversarial_fixtures": [
      {"name": "e-closer", "mix": [[0,1]]},
      {"name": "r-closer", "mix": [[1,1]]},
      {"name": "overlap", "mix": [[2,1]]},
      {"name": "unavailable", "mix": [[3,1]]},
      {"name": "invalid", "mix": [[9,1]]},
      {"name": "missing", "mix": []},
      {"name": "extra", "mix": [[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path": "CLAUDE.md"},
    {"path": "docs/PREREG_DET_OBS_LINEDECOMP_2026-10-07.md"},
    {"path": "docs/VERDICT_DET_OBS_LINEDECOMP_2026-10-07.md"},
    {"path": "docs/VERDICT_DET_OBS_CAPTURE_2026-10-07.md"},
    {"path": "docs/VERDICT_DET_OBS_WRITER_2026-10-02.md"},
    {"path": "scripts/check_prereg_preflight.py"},
    {"path": "scripts/check_gate_referents.py"},
    {"path": "scripts/check_doc_canonical.py"},
    {"path": "scripts/check_dispatch_preflight.py"},
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c",
      "flags_existing": [
        "nlte_tau_line_authority",
        "nlte_tau_line_shell_authorized_by",
        "nlte_update_tau_sobolev_with_authority",
        "nlte_pair_total_density"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/opacity_publication.c",
      "flags_existing": [
        "a208_signed_sobolev_counted",
        "a208_line_source_counted"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "flags_existing": ["det_obs_capture_pair"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/cmfgen_final_writer.c",
      "flags_existing": ["cmfgen_final_writer", "SAHA_STAGE", "NLTE_LEVEL"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina.h",
      "flags_existing": [
        "SOBOLEV_COEFF",
        "H_PLANCK",
        "C_SPEED_OF_LIGHT",
        "AMU"
      ]
    },
    {
      "path": "data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt",
      "flags_existing": ["si2", "19.480"]
    },
    {
      "path": "data/standart_data1/toy06/phys_toy06_cmfgen.txt",
      "flags_existing": ["temp[K]", "ne[/cm^3]", "19.480"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/astra_closeout3.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/evidence/USER_DECISION_2026-10-07.txt"
    }
  ]
}
```

## 기안 메모

- 공유 문맥·선행 계약과 판정·종료감사·봉인 생산 소스·입력 헤더를 읽었다. 파일 수정·시험·계산·제출은 수행하지 않았다.
- `n_ion_total`과 NLTE 준위합의 권위가 다름을 writer 소스에서 확인했다. 실제 차이의 크기는 미계산이다.
- 세대 7 전체 이온 인구와 공개 CMFGEN 준위 인구의 결손은 수치 허용오차로 해소할 수 없다.
- 비용은 등록 상한이며 성능 실측이 아니다. a10 Slurm의 GPU 요청 없는 CPU 전용 실행 경로는 운전자 실측으로 확인됐다.