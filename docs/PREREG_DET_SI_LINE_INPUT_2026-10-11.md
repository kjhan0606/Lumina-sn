# 사전등록 — DET-SI-LINE-INPUT: Si 셸10 대표선의 생산 입력·소비 기록 정합 점검

2026-10-11 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_SI_LINE_INPUT_2026-10-11.md`  
상태: **기안. 구현·직접시험·기계 게이트·수치 재계산·판정·종료감사 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 직전 단은 Si 셸10의 모드 8 감도가 G1·G2에서 서로 다른 방향으로 나타남을 확인했지만, 그 선 입력이 생산식의 정상 출력인지 판별하지 않았다.

이번 단은 **직전 대표선 여섯 개에서 호출 36의 기록 입력과 봉인 생산식의 재계산값을 대조하는 표 하나**를 만든다. 다음 결정은 입력 결함의 수리를 별도 발주할지, 생산식 자체의 물리 가정을 조사할지, 부족한 원입력을 먼저 확보할지다. 생략하면 큰 광학깊이나 강한 펌핑을 곧바로 구현 결함으로 오인할 수 있다.

**Q-LEAN — 불필요한 측정·게이트를 더하는가? 자기답: APPROPRIATE.**

유지하는 것은 선·세대 대응, 생산식 재계산, 생산→소비 전달 대조, 현재 인구·연속장의 물리 대조다. 모두 주 표의 판정과 주장 한정에 소비된다.

새 정상해·복사전달·전체 선 재분석·추가 셸·추가 원소는 계산하지 않는다. 저장 산출 재검·쓰기 결함 모사·결정표 재계산 장치를 만들지 않는다. 직접시험은 새 코드의 음성 대조로 한정한다. 종료감사의 1차 대상은 **과학 표의 수치·대응·주장 한정**이다.

기안 조회에서 이미 드러난 한계는 숨기지 않는다. **재격자화된 비교 덤프와 생산 직후 선 원장은 미세 연속장 원배열을 대신하지 못한다.** 그 원배열이 확보되지 않은 J_c에는 완전 재계산 일치를 선언하지 않는다.

## §1. 권한·경로·기준선

사용자 결정은 「대표선 입력 정합 점검」이다. 기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사를 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra가 수행하며 기안자 이해관계를 명시한다.

| 기호 | 대상 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| V | `/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/results/run1` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| H | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src` |
| C | `/gpfs/kjhan/cmfgen_src/cur_cmf` |

직전 폐합 기준은 `VERDICT_DET_SI_BB_GROUPS_2026-10-11.md`의 7d13949 판본이다. 과거 판정·저작자는 보존한다.

S는 생산식 정본, H는 계측 위치와 호출 벡터의 의미를 확인하는 소스다. 실제 계측 실행의 연결은 T의 `STAGE_IDENTITY.txt`, `input/run_build_identity.json`, 기존 범위 증거와 대조한다. 두 소스 트리가 바이트 동일하다고 가정하지 않는다.

입력 파일·크기·SHA-256·코드 판본·명령·작업 번호는 `evidence.json`에 한 번 기록한다. 실제 소비 파일만 대상으로 하며 전체 트리를 해시하지 않는다.

## §2. 과학 질문·대상

> Si 셸10의 채택 호출 36이 소비한 대표선별 τ_eff·J_c·β·c·ℓ·J̄는, 해당 생산 시점의 인구·원자 자료·시간·연속장으로 봉인 생산식을 평가한 값과 일치하는가? 불일치가 있으면 입력 대응·생산 산술·생산→소비 전달 중 어디까지 특정할 수 있는가?

채택 키는 묶음 1·수용 n_e 반복 18·CE 패스 1·호출 36이며 기존 채택 함수로 재유도한다.

대표선은 직전 판정문 §3의 G1·G2 각 세 문장으로 고정한다. 이번 결과에 따라 다시 순위를 매기거나 교체하지 않는다.

| 묶음 | `aux` | Si II 하→상 `level_num` | 직전 표시 λ(Å) |
|---|---:|---|---:|
| G1 | 760321 | 1→7 | 1533.4 |
| G1 | 632403 | 1→10 | 1264.7 |
| G1 | 757019 | 0→7 | 1526.7 |
| G2 | 1676767 | 7→12 | 6348.9 |
| G2 | 1353542 | 6→12 | 3857.1 |
| G2 | 1679119 | 7→11 | 6373.1 |

문장 신원은 `(call,op)`다. `aux`만으로 중복을 지우지 않는다. `site=1`, 모드 8, 양 끝 이온·FL·전역 준위·`level_num`·SL을 함께 대조한다.

표의 파장은 식별용이다. 생산 재계산에는 원자 입력의 `wavelength_cm`·`nu`와 실제 사용 경로를 쓰며, 위 반올림 파장이나 준위 에너지 차에서 역산한 주파수로 대체하지 않는다.

G3·G4를 제외하는 이유는 이번 사용자 선택의 직접 표적이 G1·G2 대표선이기 때문이다. 이 범위에서 다른 선의 정합성을 일반화하지 않는다.

## §3. 잣대 1 — 생산·소비 세대와 원입력

**생산 시점과 채택 소비 시점은 다르다.** 확인한 계보는 다음과 같다.

1. `cmfgen_fine_jbar`가 물질12에서 선 입력을 생성한다.
2. 복사·선 세대12를 함께 공개한다.
3. R7에서 물질13을 구하며 호출 36은 그 복사·선 세대12를 소비한다.
4. `physics_DET_iter0010.manifest.json`은 물질13·복사12·온도12·불투명도23을 기록한다.

근거는 S의 `lumina_cmfgen.c` 약 11755–11860행의 생산→공개→R7→비교 덤프 순서, T의 `si10_calls.csv`·`si10_commit.csv`, `stderr.log`의 생산 호출12·물질12·복사12 기록이다. 서로 다른 종류의 세대 번호에 같은 숫자를 강제하지 않는다.

**물질13의 `FINAL_FL`을 생산 인구로 넣지 않는다.** 생산 인구 후보는 호출1의 `PRE_FL`이다. H의 `det_si_trace.h` 약 348행은 pair 조립 직전 인구를 기록하고, H의 `lumina_plasma.c` 약 22020행은 공개 인구의 작업 사본을 만든다. Si pair는 첫 pair이며 첫 CE 패스 전에 해당 Si 인구를 새 해로 바꾸지 않는 경로를 확인한다.

채택 조건은 다음을 함께 만족하는 것이다.

- 호출1이 묶음1·n_e 반복1·CE 패스0이고, 호출 벡터와 준위 대응이 완전하다.
- 호출1 `PRE_FL`과 같은 CE 행의 `old`가 일치한다.
- 그 이온별 합이 물질12인 `physics_DET_iter0009.ion.csv`의 **NLTE 준위 합**과 §7 오차 안에서 일치한다.
- 실제 실행 소스의 사본 생성부터 첫 Si 호출까지, 해당 인구를 변경하는 경로가 없다는 근거를 `evidence.json`에 적는다.

합계 일치만으로 준위별 동일성을 증명하지 않는다. 경로 근거가 없으면 생산 인구는 미확보이며 τ 재계산 판정을 유보한다. 호출36의 `PRE_FL`이나 사다리의 단계 인구로 대체하지 않는다.

원자 입력은 실제 로더가 읽은 T의 `input/model/line_list.csv`와 `levels.csv`다. S의 `lumina_atomic.c` 약 1330행이 선 계수를 읽는다. HDF5의 존재나 이름만으로 런타임 배열 신원을 인정하지 않는다. 실행 설정의 계수 배율·원소 제외·`SKIP_Z`·선 대응도 확인한다.

시간은 manifest의 `epoch_s=1683072`와 실행 기하 입력을 연결한다. 온도는 생산 시점의 고정 온도와 세대 근거를 확인한다. 인구 단위는 cm⁻³, 파장은 cm, 주파수는 Hz, 시간은 s다.

## §4. 잣대 2 — 생산식 재계산과 전달 대조

### 4.1 광학깊이와 국소 방출

S의 `lumina_plasma.c` 약 10128행은 선 인구 권위를 결정하고, 약 21420행은 허가된 NLTE 선의 광학깊이를 계산한다. 약 10250행의 상준위 인구 선택도 같은 권위 조건을 사용한다. 대표선의 모드 8과 빈 `LUMINA_NLTE_SKIP_Z` 기록을 실제 대응과 함께 확인한다.

\[
\tau_{\rm raw}=K f_{lu}\lambda t
\left(n_l-\frac{g_l}{g_u}n_u\right),\qquad
K=2.6540281\times10^{-2}.
\]

K는 S의 `lumina.h` 약 36행의 **생산 상수**다. 더 정밀한 기본상수로 바꾸어 생산 결함을 판정하지 않는다. 실제 평가 순서·부호·형식화는 봉인 함수에 맞춘다.

S의 `line_net_rate.c` 약 215행의 `line_net_sobolev_material`을 따라
\[
\eta_l=\frac{n_uA_{ul}h\nu_l}{4\pi}
\]
와 τ_eff·provenance를 평가한다. 등록 대표선은 양의 기록 τ_eff를 가진다. 생산식에는 음의 τ에 대한 `SRCE_CHK` 분기가 있으나 이를 새 보정으로 도입하거나 다른 분기에 대한 일반 검증으로 확대하지 않는다.

\[
\beta=\operatorname{EXPONX}(\tau_{\rm eff}),\quad
c=\beta J_c,\quad
\ell=\eta_l\frac{c_{\rm light}t}{\nu_l}
\frac{1-\beta}{\tau_{\rm eff}},\quad
\bar J=c+\ell.
\]

β와 동반 함수는 생산의 작은 τ 다항식·τ≥40 분기를 그대로 따른다. 0 나눗셈을 임의 작은 수로 피하지 않는다.

### 4.2 입사 연속장

S의 `lumina_cmfgen.c` 약 9770–9850행은 `fs.J`의 윤곽 평균을 `line_net_sobolev_radiation`에 전달한다. S의 `line_jbar.c` 약 69행의 정의는 다음과 같다.

\[
J_c^{\rm calc}=
\frac{\sum_{|x_i|\le4}e^{-x_i^2}\Delta\nu_iJ_{\nu,i}}
{\sum_{|x_i|\le4}e^{-x_i^2}\Delta\nu_i}},\qquad
x_i=\frac{\nu_i-\nu_l}{\nu_l v_D/c_{\rm light}}.
\]

실행 설정은 \(v_D=10^6\) cm s⁻¹이다. 선 중심의 BB 영역과 ±4 Doppler 지지 구간은 별도로 확인한다. 실제 선 주파수·격자 중심·폭·셸 순서·합산 순서를 보존한다.

**완전 재계산에는 생산 복사12의 `fs.nu`·`fs.dnu`·셸10 `fs.J`가 필요하다.** 기안 조회에서 이 원배열의 영속 파일은 확인되지 않았다. 확인된 `spectral.csv`는 1234개 비교 구간의 장이며, S의 `physics_comparison.c` 약 43행에서 적분 보존 구간상수 재격자화를 수행한다. 이를 미세 격자에 보간해도 원래 생산 입력의 복원이 아니다.

이번 등록 입력만으로 원배열이 없으면 `J_c`의 완전 재계산 열은 **미확보**로 남긴다. 복사전달 재실행·다른 런의 장 대입·τ나 J_c의 역산을 통한 원입력 대용은 하지 않는다.

### 4.3 생산→소비 전달

T의 `work/w5_lines.csv`는 생산 직후의 별도 원장이다. 생산 `call=12`, `rad_generation=12`, `material_generation=12`, 셸10·선 번호로 선택한다. 여기의 `call`을 trace 호출36과 같은 번호 체계로 읽지 않는다.

생산 원장의 J_c·β·ℓ·J̄·τ_eff를 호출36 기록과 대조한다. 원장의 J_c를 사용한 c 재계산은 **생산→소비 전달 및 하류 식 대조**이며, 미세 연속장으로부터의 J_c 재계산으로 표시하지 않는다.

기록 복사 3율도 독립 원자 계수로
\[
r_a=B_{lu}c,\qquad r_s=B_{ul}c,\qquad r_p=A_{ul}\beta
\]
를 대조한다. 기록율에서 A·B·인구를 역산하여 같은 기록을 검증하지 않는다.

## §5. 물리 대조와 CMFGEN 맥락

물리 대조는 생산식 일치 판정과 별도 열에 둔다.

- 호출36 `FINAL_FL`로 같은 부호 있는 Sobolev 식을 평가한 `tau_FL13`: 생산 시점과 현재 소비 인구의 차이를 보인다. 생산 τ와 달라도 곧바로 결함이 아니다.
- `physics_DET_iter0010.spectral.csv`에서 선 주파수가 속한 구간의 `J_nu`: 공개된 거친 장과 생산 J_c의 차이를 보인다. `[nu_lo,nu_hi)`를 쓰고 마지막 끝점만 포함한다. 외삽하지 않는다.
- \(B_\nu(T_e)=2h\nu^3/[c_{\rm light}^2\operatorname{expm1}(h\nu/kT_e)]\): J_c/Bν와 거친 Jν/Bν로 열적 규모를 제시한다. Bν를 기대 입사장으로 강제하지 않는다.

세대 차이·격자 평균 차이·물리적 비열성은 서로 다른 설명이다. 대조값과의 차이만으로 그중 하나를 원인으로 확정하지 않는다.

CMFGEN의 `sobjbar_sim.f` 약 90행은 `GAMH=CHIL*3.0E-10*R/V/FL`, 약 115행은 속도 구배에 따른 각도 의존 광학깊이를 사용한다. 약 150–190행은 연속 전달에서 구한 `AV`와 `EXPONX(GAM)`을 각도 적분한다. `exponx.f` 약 15행은 생산 β 분기의 근거다. `cmfgen_sub.f` 약 3550행에는 음의 Sobolev 광학깊이 처리 경로가 있다.

이는 식의 대응 근거다. Lumina의 비중첩·상사 팽창 Sobolev 처방과 미세장 가우스 평균을 공개 CMFGEN의 실제 선별 처리와 동일시하지 않는다. 공개 출력에서 여섯 선의 τ·입사장을 확보한 것은 아니다.

주 표의 공통 맥락에는 직전 결과를 **인용값**으로 싣는다: 공개 III/II=105.697(약 105.70), F0=3.559757, 공개 대비 −1.47264 dex, G1 독립 D8 이동 −0.80359 dex, G2 +0.10258 dex. 대응은 19.480 d, Lumina 11,544 km/s ↔ 공개 11,534.59 km/s다. F0를 채택 `FINAL_FL`의 비로 부르지 않는다.

## §6. 주 산출·기존 코드 재사용

정본은 `si10_line_input_table.csv`, 표시본은 같은 메모리 행으로 만든 `si10_line_input_table.md`다. 대표선당 한 행이며 공통 맥락은 표 설명에 둔다.

| 열 묶음 | 필수 내용 |
|---|---|
| 신원 | 묶음·`call`·`op`·`aux`·이온·상하 준위 대응·생산 파장과 주파수 |
| 원입력 | 생산·소비 세대, 인구 출처·n_l·n_u·g_l·g_u·f_lu·A·B·t·provenance |
| 생산식 대조 | τ_raw·τ_eff·β·ℓ·J_c·c·J̄의 기록값·재계산값·절대差·상대差·개별 판정 |
| 전달 대조 | 생산 원장 값·호출36 값·차이·전달 판정, 복사 3율 대조 |
| 물리 대조 | `tau_FL13`·거친 Jν·Bν·정의 가능한 비와 부호 있는 dex |
| 결론 | 선별 판정·불일치 구간·未解決 입력·근거 참조 |

위의 `未解決` 표기는 실제 산출에서 **미해결**로 쓴다. 미확보·해당 없음·정확한 0을 구분하며, 실패한 대표선도 행을 남긴다.

기존 판독기 네 개는 수정하지 않는다. `det_si_trace_readout.py`의 `read_trace`·`num`, `det_si_excitation_budget.py`의 `det_si_exc_adopt`, `det_si_bb_groups.py`의 `det_si_bb_groups_fl_info`·`det_si_bb_groups_assign`, `det_si_bb_counterfactual.py`의 `det_si_bb_mode`·`det_si_bb_exponx`를 재사용한다.

직전 `cmd_analyze`·정상해·수지·대표선 순위 계산은 호출하지 않는다. V의 기존 `line` 행은 대표 문장 연결에만 사용하며 T의 원기록과 확인한다. 새 코드는 선택 행 판독·원입력 연결·등록 식·차이·표 출력에 한정한다.

보조 산출은 표의 출처·미확보 사유·실행 신원을 담는 `evidence.json`, 실행·기계 검사 로그, 음성 대조 결과뿐이다.

## §7. 사전 허용오차·기대치

실수 대조는 결과를 보기 전에 다음으로 고정한다.

\[
D=|a-b|,\qquad M=\max(|a|,|b|).
\]
\(M>0\)이면 \(D/M\le10^{-10}\), \(M=0\)이면 두 값이 정확히 0일 때만 일치다. 절대差도 원단위로 남긴다. 분모에 임의 floor를 넣지 않는다.

인구의 이온합 대조도 같은 기준을 쓴다. 선·준위·셸·세대·영역·provenance 같은 이산 신원은 정확히 일치해야 한다. 생산 원장과 trace의 십진 기록은 원정밀도로 읽는다.

Sobolev 인구 차가 상쇄되는 경우에는 추가로
\[
S_\tau=|Kf\lambda t|\left(|n_l|+\left|g_ln_u/g_u\right|\right)
\]
와 \(10^{-10}S_\tau\)를 보고한다. 잔차가 이 규모 아래인데 일반 상대오차만 초과하면 **수치 분해능 부족**으로 유보하며 결함으로 확정하지 않는다.

동일 원입력의 생산식 재계산에는 복사전달의 물리적 오차 상한을 허용오차로 더하지 않는다. `J_c_err_upper`는 생산 장의 불확실성 설명값이며 큰 재계산 차이를 면제하지 않는다.

예상은 원입력·경로가 같다면 생산식과 전달값이 위 오차 안에서 일치한다는 것이다. τ의 크기나 J_c/Bν가 1에 가까워야 한다는 기대는 두지 않는다. `tau_FL13`의 이동·거친 장과 J_c의 차이·결함 유무와 위치는 예측하지 못한다.

기안 시점에 미세장 원배열이 미확보이므로 **J_c 완전 재계산이 미해결로 남을 가능성을 사전등록**한다. 나중에 전달 일치를 완전 재계산 일치로 승격하지 않는다.

## §8. 최소 변경집합·보존

### 변경집합 — DET-SI-LINE-INPUT (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_si_line_input.py` | 선택 행·세대·원자 대응, 등록 재계산·전달·물리 대조, 주 표, 새 코드 음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/tools/run_det_si_line_input_offline.slurm` | 제한 CPU 실행·입력 신원·비용 기록 |

### 변경집합 끝

생산 소스·바이너리·실행 입력·CMFGEN 소스 변경은 0이다. 수리·새 계측·생산 재실행은 이 단 밖이다.

출력은 B의 새 실행 하위 경로에만 쓴다. 기존 출력과 실패 산출을 보존한다. 일반 파일 읽기·쓰기만 사용한다. 하드링크·파일시스템 고유 기능·clamp/floor를 도입하지 않는다.

## §9. 직접시험·게이트

새 코드의 작은 합성 입력에 다음 결함을 주입하여 해당 대응·수식 검사가 거부함을 확인한다.

- 생산12 대신 소비13 인구 연결, 생산 호출 번호와 trace 호출 번호 혼용.
- 상하 준위·행 번호·이온 번호 오연결, Å/cm·일/s 단위 혼용.
- Sobolev 유도방출 항 누락, β 이중 적용, ℓ를 입사장에 합산.
- 가우스 가중치의 Δν 누락, 지지 구간 밖 값 사용, 거친 장을 생산 원배열로 표기.
- 미확보 입력을 0 또는 기록값으로 채워 일치 처리.

각 음성 대조에는 변조 전 유효 입력과 실제 변조가 수치·신원을 바꿨음을 함께 확인한다. 기존 코드의 시험을 복제하지 않는다.

| 게이트 | 막는 실재 실패 |
|---|---|
| G-IDENTITY | 다른 선·세대·인구 권위·원자 판본을 결합하여 정상값을 결함으로 판정하는 실패 |
| G-CALC | 단위·부호·가중치·β 적용 오류 또는 기록값 역산을 독립 재계산으로 보고하는 실패 |
| G-TABLE | 생산→소비 일치나 물리 대조 차이를 완전 정합·물리 정답으로 확대하는 실패 |
| G-RUN | GPU·생산 실행·비용 초과·미완료를 등록 오프라인 결과로 보고하는 실패 |

과학적 불일치 자체는 게이트 실패가 아니다. 계보가 검증된 불일치는 과학 결과다. 원입력 미확보를 정직하게 표시한 표도 유효한 부분 결과지만 전체 정합 확인은 아니다.

검사는 원입력과 메모리 결과에 적용한다. 기존 사전등록·참조·정본·발주·러너 위생·게이트 집행 검사는 기존 도구로 수행한다.

## §10. 비용·실행·재현

| 항목 | 상한 |
|---|---|
| GPU·생산 실행·정상해·복사전달 재실행 | 모두 0 |
| 실행 | a10 CPU Slurm, 작업명 `LUMINA_` 접두 |
| 작업당 | 1 task·1 CPU·4 GiB·벽시계 300 s |
| 누적 | 할당 벽시계 900 s·0.25 CPU core·h |
| 스레드 | OpenMP·BLAS 각 1 |
| 신규 저장 | 30 MiB |

직접시험·기계 검사·수정 재실행·감사 재계산을 누계에 포함한다. 입력 CSV는 필요한 행만 보존하는 순차 판독을 사용한다. 로그인 노드에서 수치 계산·해시 계산·시험을 실행하지 않는다.

러너는 `module purge` 후 사용 모듈을 명시하고 Python·수치 라이브러리 판본을 기록한다. Slurm 요청은 `--partition=a10 --ntasks=1 --cpus-per-task=1 --mem=4G --gres=gpu:0 --time=00:05:00 --exclude=syn102,syn03`이며 실제 GPU 할당 0을 확인한다.

작업 내부 인터페이스는 다음과 같다.

```bash
python3 -B scripts/det_si_line_input.py negative
python3 -B scripts/det_si_line_input.py analyze \
  --run /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1 \
  --baseline /gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/results/run1 \
  --out /gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/results/run1
```

재현은 같은 입력·코드·명령과 새 출력 경로로 수행한다. 상한에 도달하면 확보한 표와 미해결 사유를 남긴다. 미실행 재계산을 재현 통과로 적지 않는다.

## §11. 과학 분기·폐합·집행 기록

| 코드 | 분기 | 조건 |
|---:|---|---|
| 0 | `INPUT_DEFECT_FOUND` | 검증된 동일 입력·계보에서 허용오차 밖 생산식 또는 전달 불일치가 하나 이상 확인됨 |
| 1 | `INPUT_CONSISTENT` | 여섯 선 모두 생산 인구·원자·시간·미세장 입력이 확보되고 필수 생산식·전달 대조가 일치함 |
| 2 | `INDETERMINATE` | 확정 결함은 없으나 필수 원입력·계보·수치 분해능이 부족하여 전체 판별을 완료하지 못함 |

분기0이 일부 항목의 미해결과 공존하면 확정한 결함과 미해결 범위를 함께 적는다. 잘못 연결한 비교 입력을 발견한 것만으로 생산 결함을 선언하지 않는다. 허용된 한 세대 지연은 그 자체로 결함이 아니다.

분기1도 “등록 여섯 선이 봉인 생산식과 정합”이라는 뜻이다. 생산식의 물리적 정답·CMFGEN 동등성·자기일관 효과·스펙트럼 개선·생산 승격·철족 일반화를 뜻하지 않는다. 결함 수리는 별도 발주다.

폐합에는 여섯 행의 주 표, 운전자 판정, astra 종료감사가 필요하다. 분기2는 **전체 정합 판별 목표 미달**로 기록한다. 전달 대조만 완료한 것을 목표 달성으로 바꾸지 않는다.

종료감사는 수치·선 대응·생산 인구의 세대·J_c 대조의 독립성·CMFGEN 주장 범위를 우선 본다. 소스 열람과 저장 증거 대조를 실행 재현과 구분한다.

| 집행 항목 | 담당·현재 상태 |
|---|---|
| 기안 조회 | astra `[저자 실행]`: §1–5의 문서·소스·헤더·세대 로그 열람 |
| 수치 재계산·입력 해시 | 운전자 `[미실행]` |
| 프리플라이트·발주·러너·게이트 rc | 운전자 `[미실행]` |
| 구현·직접시험·등록 실행·비용 | 운전자 `[미실행]` |
| 판정 | 운전자 `[미실행]` |
| 종료감사·재현 상태 | astra `[미실행]`; 계약 저자 이해관계 명시 |

저자 조회 명령의 범위는 `cat`·`head`·`sed -n`·명시 파일의 `rg -n`이었다. 예: `sed -n '9770,9850p' "$S/lumina_cmfgen.c"`, `sed -n '348,400p' "$H/det_si_trace.h"`, `head -3 "$T/work/w5_lines.csv"`이며 기호는 §1을 따른다. 수치 일치 판정은 아직 수행하지 않았다.

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011`. 작업 번호·경과는 `B/JOBS.txt`·`B/evidence/sacct_offline_jobs.txt`, 게이트 집행은 `B/evidence/GATE_RECORD.json`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| user 결정 | user | 「대표선 입력 정합 점검」(`B/evidence/USER_DECISION_2026-10-11.txt`) |
| 기안 | astra(자기보고 「gpt-6-astra」) | 1609e79(개정 없음) |
| 기계 게이트(기안) | 운전자 | 기안 브리프 canonical(a10 419374), 기안 prereg·referents·canonical rc 0(419377) |
| 구현 | 운전자(Opus 5.5) · 코드 감사 없음(25-z) | a90ae08 — `scripts/det_si_line_input.py`(기존 판독기 무수정), 러너 `B/tools/run_det_si_line_input_offline.slurm` |
| 직접시험(§9) | 운전자 | a10 419380 실패(시험 장치 결함: 지지 구간 밖 주입값이 작아 변조가 10⁻¹⁰ 허용오차를 벗어나지 못함 — 게이트 결과 아님) → 밖 지점 J 를 크게 고침 → 419382 **17/17**(양성 대조 2·결함 주입 거부 15) |
| `[미실행]` 항목 집행 | 운전자 | 수치 재계산·입력 해시는 run1 에서, 프리플라이트·러너 위생·게이트 rc 는 위 작업에서, 비용은 sacct 로 집행 |
| 등록 계산 run1 | 운전자 | a10 419383(4 s) `B/results/run1/`: G-IDENTITY·G-CALC·G-TABLE·G-RUN 지적 0; 생산 인구 = 호출1 PRE_FL(CE old 와 동일, 물질12 NLTE 이온합과 상대차 ≤2.7×10⁻¹⁶); 대표선 6/6 생산식 재계산·전달 대조·복사 3율 상대차 0; **분기 2 INDETERMINATE** — J_c 완전 재계산 미확보(생산 미세 연속장 원배열 미보존) |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_SI_LINE_INPUT_2026-10-11.md` — INDETERMINATE(0f23727), 1회차 «기재로 족함» 3 반영, **폐합**(전체 정합 판별 목표 미달) |
| 종료감사 | astra(계약 저자 이해관계) | GPT-6 읽기 전용 검토: 과학 폐합 차단 0건. astra 세부 모델 신원 미확인. 저장 증거·소스 대조 완료, 실행 재현 BLOCKED(`B/astra_closeout.md`); «기재로 족함» 3 → 판정문 §2·§3·§4·§5 반영(판정문 §8) → **폐합** |
| 기계 게이트(판정·감사 발주) | 운전자 | 판정 roster·prereg preflight·referents·canonical rc 0(a10 419384), 감사 발주 dispatch·canonical rc 0(419385), 폐합본 4종 rc 0(419388) |
| 비용 | 운전자 | GPU 0; 오프라인 a10 8 작업 할당 98 s · 0.027 core·h / 900 s · 0.25 core·h(기안 게이트 2 작업 30 s·판정/감사/폐합 게이트 3 작업 62 s 포함); 신규 저장 63 KiB / 30 MiB |

**이탈(기재):** 주 CSV 에서 절대차·개별 판정·provenance 및 복사 3율의 값 쌍을 생략했다. 상대차와 원기록으로 추적 가능하며 이번 분기에는 영향이 없다(종료감사 1회차 지적 3). 표시본 `.md` 는 같은 메모리 행을 선별 소표로 나눴다.

## §12. 기계 선언

참조에는 실재 파일만 지정한다. 예정 신규 파일은 변경집합에만 둔다. 분기 fixture는 기존 프리플라이트용이며 새 결정 재계산 장치가 아니다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SI-LINE-INPUT (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_si_line_input\\.py|/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/tools/run_det_si_line_input_offline\\.slurm)",
    "symbol": "det_si_line_input",
    "roots": [
      "scripts/det_si_line_input.py",
      "/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/tools/run_det_si_line_input_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_si_line_input.py",
      "/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/tools/run_det_si_line_input_offline.slurm"
    ]
  },
  "branches": {
    "regimes": [[0,0],[1,1],[2,2],[9,9]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)"
    },
    "rules": [
      {"name":"INPUT_DEFECT_FOUND","predicate":"valid and c0 == 1"},
      {"name":"INPUT_CONSISTENT","predicate":"valid and c1 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"defect","mix":[[0,1]]},
      {"name":"consistent","mix":[[1,1]]},
      {"name":"indeterminate","mix":[[2,1]]},
      {"name":"invalid","mix":[[9,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_SI_BB_GROUPS_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_BB_GROUPS_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_CONSUMPTION_TRACE_2026-10-11.md"},
    {"path":"scripts/det_si_bb_groups.py","flags_existing":["det_si_bb_groups_fl_info","det_si_bb_groups_assign"]},
    {"path":"scripts/det_si_bb_counterfactual.py","flags_existing":["det_si_bb_mode","det_si_bb_exponx"]},
    {"path":"scripts/det_si_excitation_budget.py","flags_existing":["det_si_exc_adopt"]},
    {"path":"scripts/det_si_trace_readout.py","flags_existing":["read_trace"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/evidence/USER_DECISION_2026-10-11.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/w5_lines.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_vec.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/model/line_list.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/model/levels.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/physics_comparison/physics_DET_iter0010.manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_jbar.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/det_si_trace.h"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sobjbar_sim.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/subs/exponx.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f"}
  ]
}
```
