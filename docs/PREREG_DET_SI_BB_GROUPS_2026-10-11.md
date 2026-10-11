# 사전등록 — DET-SI-BB-GROUPS: Si 셸10 모드 8 감도의 선 묶음 분해

2026-10-11 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_SI_BB_GROUPS_2026-10-11.md`  
상태: **기안. 구현·직접시험·기계 게이트·계산·판정·종료감사 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 직전 단은 모드 8 전체에 대한 D8 처방이 Si 셸10 III/II를 약 −0.836 dex 움직임을 보였지만, 어느 선 집합을 다음 물리 점검 대상으로 삼을지는 가르지 못했다.

이번 단은 **사전에 고정한 선 묶음 각각의 독립 D8 효과를 CMFGEN 대비 표 하나로 제시**한다. 표에는 선택 준위 수지 변화와 해당 선들의 기록 입력을 연결한다. 다음 결정은 어느 묶음의 광학깊이·탈출확률·입사 연속장을 우선 점검할지다. 생략하면 전체 모드 감도를 특정 선의 문제로 성급히 귀속하거나 모드 8 전체를 불필요하게 조사하게 된다.

**Q-LEAN — 불필요한 측정·게이트를 더하는가? 자기답: APPROPRIATE.**

| 유지 항목 | 주 표의 소비처 |
|---|---|
| 채택 계보·원자료 재생·F0·전체 D8 대조 | 직전 감도를 같은 소비 상태에서 분해한다는 근거 |
| 등록 묶음 4개의 독립 개입 | Si II 내부 결합 부류와 Si III의 조건부 감도 구분 |
| 선택 준위 수지·선 입력 | 다음 물리 점검의 대상과 이유 |
| 비가산 잔차 | 독립 효과 합을 전체 효과의 기여율로 오인하지 않는 근거 |
| 입력 신원·수치 품질·비용 | 표의 유효성과 주장 범위 |

조합 탐색·매개변수 탐색·개별 선 정상해·추가 셸·추가 원소는 제외한다. 저장 산출 재검·쓰기 결함 모사·결정표 재계산 장치를 만들지 않는다. 직접시험은 새 코드의 음성 대조로 한정한다. 종료감사의 1차 대상은 **과학 표의 수치·대응·주장 한정**이다.

## §1. 권한·기준선·경로

사용자 결정은 「모드 8 감도의 선 묶음 분해」다. 기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사를 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra가 수행하며 계약 기안자 이해관계를 명시한다.

| 기호 | 대상 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| V | `/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/results/run1` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| 공개 자료 | `data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt` |

직전 폐합 기준은 `VERDICT_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md`의 1978324 판본이다. 그 단의 판정과 저작자는 보존하며 재판정하지 않는다.

T의 기록 문장이 과학 입력이고 V는 F0·F8 대조값의 출처다. 소비 파일·크기·SHA-256·코드 판본·명령·작업 번호는 `evidence.json`에 한 번 기록한다. 전체 트리를 해시하지 않는다.

## §2. 과학 질문·고정 조건

> 물질13·Si 셸10의 채택 소비 상태에서, 모드 8 문장의 복사 3율을 기록 β로 나누는 D8을 선 묶음별로 따로 적용하면 III/II와 선택 준위 수지는 얼마나 움직이며, 독립 효과의 합은 전체 D8 효과와 얼마나 다른가?

채택 키는 묶음 1·수용 n_e 반복 18·CE 패스 1·호출 36이다. 기존 채택 함수로 재유도한다.

연속 J장·T_e·n_e·CE 상대 인구·Si II+III 총량을 고정한다. 광학깊이·β·선원천함수·복사장·전자밀도를 새 정상해로 갱신하지 않는다. BB 충돌·모드 7 복사·BF·DR·CE·NT는 원래 율을 유지한다.

F0는 기록 FL 연산자 정상해, F8은 모든 모드 8 문장에 D8을 적용한 정상해다. F-G1부터 F-G4는 각각 **원래 F0 연산자에서 해당 묶음만** 바꾼 정상해다. 앞 묶음의 처방이나 해를 다음 묶음에 누적하지 않는다.

채택 `FINAL_FL`은 생산 소비 인구이며 F0와 같다고 간주하지 않는다. 이번 표의 개입 기준은 F0다.

## §3. 결과 전에 고정하는 묶음

대상 집합 U는 채택 호출의 `site=1`, `x9=8`인 문장 전체다. 모드는 정수 여부를 확인한 뒤 기존 판별 함수를 사용한다.

문장 신원은 `(call, op)`이며 선 신원은 이온·`aux`·상하 FL 번호다. 서로 다른 문장을 선 번호만으로 합치거나 삭제하지 않는다.

`lev_a`는 상 준위, `lev_b`는 하 준위다. 기존 조립기와 동일하게 `si10_slmap.csv`의 `fl_index`로 연결하고 `i_fl`, `global_level`, `level_num`, `upper_ion`, 에너지·g·SL을 함께 보존한다. 이 번호들을 서로 대체하지 않는다. 양 끝의 이온과 SL 대응도 확인한다.

| 묶음 | 결과와 무관한 소속 규칙 | 구분하려는 결합 |
|---|---|---|
| G1 | Si II이고 하 준위 `level_num ∈ {0,1}` | 바닥·미세구조 준위와 연결된 복사 결합 |
| G2 | Si II이고 하 준위 `level_num ∈ {6,7,9,10}` | 선택 중간 준위에서 더 높은 준위로 이어지는 복사 결합 |
| G3 | Si II이고 하 준위가 위 두 집합 밖 | 나머지 Si II 복사 결합 |
| G4 | Si III의 모드 8 문장 전체 | Si III 내부 복사 재분배가 만드는 효과 |

이 규칙은 상호 배타적이며 U를 완전히 덮는다. 상 준위는 소속 판정에 쓰지 않는다. 예를 들어 중간 준위에서 바닥으로 내려가는 선은 G1이다. 따라서 G2를 「중간 준위에 닿는 모든 선」이라고 부르지 않는다.

이온 밖 문장·대응 실패·중복 문장 신원은 미정의로 처리한다. 빈 묶음은 그대로 남겨 F0와 같은 무개입 행으로 표시하며 다른 묶음과 합치지 않는다.

등록 대조 건수는 모드 8 Si II 1169·Si III 1418이다. 따라서 G1+G2+G3은 1169, G4는 1418이어야 한다. 개별 G1–G3 건수는 소속 규칙에서 산출하며 결과에 맞춰 요구하지 않는다.

τ 부호별 건수와 에너지 범위는 각 묶음의 설명값이다. τ 절단값·에너지 경계·분위수로 묶음을 다시 나누지 않는다. 새 효과를 본 뒤 묶음을 바꾸는 것은 이 단 밖의 새 계약이다.

## §4. 처방과 해석의 잣대

site 1의 x1–x14는 직전 계약과 동일하다.

\[
(r_a,r_s,r_p,C_\uparrow,C_\downarrow,f_l,f_u,
\bar J,m,\beta,c,\ell,\tau_{\rm eff},p).
\]

모드 8의 기록 관계는 다음과 같다.

\[
c=\beta J_c,\qquad
r_a=B_{lu}c,\quad r_s=B_{ul}c,\quad r_p=A_{ul}\beta .
\]

묶음 g의 대상 문장에만 다음을 적용한다.

\[
(r_a',r_s',r_p')=(r_a,r_s,r_p)/\beta .
\]

나머지 문장은 무변경이다. 기록 β·τ_eff·c·ℓ·J̄는 보존하며 처방 β=1을 별도 열에 적는다. 자발율만 바꾸거나 J̄를 외부 펌프로 넣지 않는다. 입력 대조에는 기존 β 재구성·복사 원장·처방식 검사를 사용한다.

**이 처방은 기록 입사 연속장을 유지한 얇은 복사 결합 기준의 진단이며 CMFGEN 재현이 아니다.** 음의 τ에서는 β>1일 수 있으므로 복사율이 감소할 수 있다. 율의 증감은 인구·III/II의 상한이나 하한을 정하지 않는다.

이 단은 묶음별 유한 개입에 대한 조건부 반응을 가른다. 특정 선 입력이 틀렸는지, 개별 선이 전체 변화의 몇 퍼센트를 일으켰는지, 자기일관 계산에서 같은 효과가 나는지는 가르지 못한다.

## §5. 계산과 기존 함수 재사용

기존 파일은 수정하지 않는다. 새 판독기는 다음 함수를 가져다 쓰고, 직전 `cmd_analyze` 전체를 실행하지 않는다.

| 출처 | 재사용 |
|---|---|
| `det_si_trace_readout.py` | `read_trace`, `read_matrix`, `replay`, `det_si_exc_build_fl`, `backward`, `ratio`, `dump_row` |
| `det_si_excitation_budget.py` | `det_si_exc_adopt`, `det_si_exc_level_map`, `det_si_exc_budget`, `det_si_exc_public` |
| `det_si_bb_counterfactual.py` | `det_si_bb_mode`, `det_si_bb_ion`, `det_si_bb_check`, `det_si_bb_d8`, `det_si_bb_prescription_check`, `det_si_bb_build`, `det_si_bb_channel_sum_ok`, `det_si_bb_invariance`, `det_si_bb_solve` |

새 콜백은 `(call,op)`의 등록 소속만 검사해 대상이면 기존 D8 함수를 호출하고 아니면 `None`을 반환한다. 대상 부분집합에는 기존 D8 처방식 검사를 적용하고, 비대상 문장은 반환값 `None`을 확인한다. 전체 모드 8을 요구하는 검사를 부분집합 처방에 그대로 적용하지 않는다.

원율 분해 검사는 기존 훅이 대체 전에 수행한다. 모드 분리 채널·비개입 불변·채널 합·보존·정상해 검사는 기존 함수를 재사용한다. 빈 묶음은 F0의 계산 결과를 재사용한다.

정상해는 F0·F8·비어 있지 않은 묶음별 해, 최대 6개다. 선택 준위는 Si II `level_num` **0·1·6·7·9·10·42·44·52·124**를 유지한다.

정상해는 유한·비음, 닫힌 통신류 1개, 총량 상대오차와 열 보존 오차 \(10^{-12}\) 이하, 모든 행의 후방오차 유한 및 최대 \(10^{-8}\) 이하여야 한다. 원율·대체율·J_c도 유한해야 한다. 미정의 값·음수율·영 분모는 clamp/floor로 치환하지 않는다.

## §6. 주 산출 — CMFGEN 대비 표 하나

정본은 `si10_bb_groups_table.csv`, 표시본은 같은 메모리 행에서 만든 `si10_bb_groups_table.md`다. 행 종류별 표시용 소표는 허용한다.

| 행 종류 | 필수 내용 |
|---|---|
| 공개·F0·F8·F-G1–F-G4 | III/II, 공개 대비 dex, F0 대비 이동, 공개 거리 감소, 유효성 |
| 묶음 | 소속 규칙·문장 수·τ 부호별 수·효과 순위·분류 |
| 선택 준위×상태×채널·합계 | 준위 신원, \(f_i,b_i^{rel},I,O,B,D\), F0 대비 차, 정상해 잔차 |
| 비가산성 | III/II의 독립 이동 합·전체 이동·잔차, 준위 수지의 대응 잔차 |
| 모드 8 문장 전체 | 묶음·선 신원·기록 입력·대체율·후속 점검 표시 |
| 품질·범위 | 채택 키·고정량·후방오차·대조差·비용·신원 참조 |

\[
R=N_{\rm III}/N_{\rm II},\quad
d=\log_{10}(R/R_{\rm pub}),\quad
\Delta_g=\log_{10}(R_g/R_0),\quad
G_g=|d_0|-|d_g|.
\]

공개 대응은 19.480 d, Lumina 11,544 km/s ↔ 공개 11,534.59 km/s다. 계산은 기존 공개 판독기가 반환하는 원자료 비를 사용하며 반올림값 105.70을 입력하지 않는다.

\[
f_i=n_i/N_{\rm II},\qquad b_i^{rel}=f_i/f_i^{LTE},
\quad f_i^{LTE}=\frac{g_i e^{-E_i/kT_e}}{\sum_{j\in{\rm Si\,II}}g_j e^{-E_j/kT_e}}.
\]
\[
I_{ic}=\sum_{j\ne i}R^c_{j\to i}n_j,\quad
D_{ic}=\sum_{j\ne i}R^c_{i\to j},\quad
O_{ic}=D_{ic}n_i,\quad B_{ic}=I_{ic}-O_{ic}.
\]

I·O·B는 \(\mathrm{cm^{-3}\,s^{-1}}\), D는 \(\mathrm{s^{-1}}\)다. 채널은 직전 모드 7·8 복사 상향·하향 및 충돌·BF·DR·CE·NT 분류를 유지한다. F0 대비 차는 각 상태 인구로 평가한다. 비개입 채널의 I·O 변화는 그 채널 율의 변경을 뜻하지 않는다.

공개 준위별 수지는 미확보다. 준위 행에 공개값을 꾸며 넣지 않는다. 0·미확보·해당 없음·미정의를 구분한다.

**선 입력 행:** U 전체를 한 번씩 기록한다. 필수 열은 `(call,op)`, `aux`, 묶음, 이온, 상하 FL 번호·`level_num`·SL·에너지·g, τ_eff·β·c·ℓ·J̄·p, \(J_c=c/\beta\), 원 복사 3율·대체 3율이다. J_c는 기록 c와 β에서 복원한 값이라고 명시한다.

후속 점검용 대표선은 각 묶음에서 다음 점수 상위 3문장으로 고정한다.

\[
W_e=n_l^0|r_a'-r_a|+
n_u^0\bigl(|r_s'-r_s|+|r_p'-r_p|\bigr).
\]

동률은 `aux`, `op` 오름차순으로 정한다. W의 단위는 \(\mathrm{cm^{-3}\,s^{-1}}\)이며 F0 인구에서의 직접 율 교란 규모다. **III/II의 개별 선 기여도나 순수지 변화가 아니다.** 모든 입력 행을 남기므로 대표선 밖의 선도 누락되지 않는다. 대표선 선택은 새 정상해나 후속 개입을 발생시키지 않는다.

## §7. 비선형성·기대치·과학 분기

\[
\Delta_8=\log_{10}(R_8/R_0),\qquad
S=\sum_{g=1}^{4}\Delta_g,\qquad
\epsilon_{\rm nl}=\Delta_8-S.
\]

S·Δ₈·ε_nl을 부호 있는 dex로 모두 보고한다. \(|\epsilon_{\rm nl}|>0.01\) dex이면 이 분할에서 비가산성이 분리됐다고 적는다. 그 이하는 보고 분해능 내이며 가산 법칙의 입증이 아니다.

각 선택 준위·채널의 \(Q\in\{I,O,B,D\}\)에도
\[
\epsilon_Q=(Q_8-Q_0)-\sum_g(Q_g-Q_0)
\]
를 원단위로 기록한다. 별도 임계값이나 합격 조건은 만들지 않는다. 이 잔차로 특정 묶음 쌍의 상호작용을 식별하지 않는다. 조합 개입·순서별 분해·기여율 정규화는 하지 않는다.

F0와 F8은 V의 원정밀도 값 각각과 0.001 dex 이내로 일치해야 한다. 참고값은 F0=3.5597568125822305, F8=0.5187172546882665다. 불일치는 새 감도로 해석하지 않는다.

비음 τ의 D8 복사율은 감소하지 않고 음의 τ에서는 감소할 수 있다. **각 묶음의 III/II·선택 준위 인구·수지 변화의 부호·크기·효과 순위와 비가산 잔차의 부호·크기는 예측하지 못한다.** G1이나 G2가 지배한다는 기대를 수용 조건으로 삼지 않는다.

묶음은 \(|\Delta_g|>0.01\) dex이면 효과 분리, 나머지는 미분리다. 공개 접근은 \(G_g>0.01\), 이격은 \(G_g<-0.01\), 나머지는 거리 변화 미분리다. 모두 보고 분해능이며 물리 수용오차가 아니다.

유효 묶음의 \(|\Delta_g|\) 내림차순과 부호를 함께 제시한다. 「최대 독립 효과 묶음」이라고만 부르며 전체 효과의 주원인·독립 기여율로 확대하지 않는다.

| 코드 | 분기 | 조건 |
|---:|---|---|
| 0 | `GROUP_EFFECT_RESOLVED` | 필수 표·대조·해가 유효하고 하나 이상 \(|\Delta_g|>0.01\) |
| 1 | `NO_GROUP_EFFECT_RESOLVED` | 필수 표·대조·해가 유효하나 모든 \(|\Delta_g|\le0.01\) |
| 2 | `INDETERMINATE` | 필수 입력·해·대조·표가 미정의이거나 수치 불확실성으로 분류 불가 |

큰 전체 효과와 작은 개별 효과가 공존해도 유효 측정이다. 일부 실패 행이 있으면 유효 행은 보존하되 완전한 분해라고 보고하지 않는다. 작은 후방오차만으로 전방 정확도를 보증하지 않는다.

## §8. 최소 변경집합·보존

### 변경집합 — DET-SI-BB-GROUPS (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_si_bb_groups.py` | 소속 결정·부분집합 콜백·기존 함수 연결·비가산 잔차·주 표·새 코드 음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/tools/run_det_si_bb_groups_offline.slurm` | 제한 CPU 실행·신원·비용 기록 |

### 변경집합 끝

생산 소스·바이너리·입력·CMFGEN 소스 변경은 0이다. 기존 판독기 3개는 무수정 재사용한다. 필요성이 생겨도 조용히 변경집합을 늘리지 않는다.

출력은 B의 새 실행 하위 경로에만 쓴다. 기존 경로를 덮어쓰지 않고 실패 산출도 보존한다. 일반 파일 읽기·쓰기만 사용하며 하드링크를 사용하지 않는다.

보조 산출은 주 표가 참조하는 `evidence.json`, 실행·기계 검사 로그, 음성 대조 결과로 제한한다. 별도 선 순위 파일·행렬 덤프·범용 검증 장치를 만들지 않는다.

## §9. 직접시험·게이트

직접시험은 새 소속 판정·콜백·표 연결 코드의 다음 음성 대조로 한정한다. 작은 합성 입력에서 결함을 주입하고 거부를 확인한다.

- 상하 준위 뒤집기·FL 번호와 `level_num` 혼용·이온 오대응.
- 문장 중복·누락·복수 소속·등록 밖 소속.
- 비대상 변경·이전 묶음 누적·자발율만 변경·β 이중 나눗셈.
- 선 입력 행의 묶음 오연결·다른 상태의 수지를 현재 행에 연결하는 결함.

기존 정상해·기존 게이트의 시험을 새 배터리로 복제하지 않는다.

| 게이트 | 막는 실재 실패 |
|---|---|
| G-INPUT | 다른 호출·준위 번호·이온을 결합하거나 모드 8 문장을 누락·중복해 잘못된 묶음을 측정하는 실패 |
| G-RATE | 비대상·충돌율 변경이나 처방 누적으로 독립 D8이 아닌 효과를 보고하는 실패 |
| G-SOLVE | 비유일·비유한·음수·큰 잔차의 해 또는 다른 F0·F8을 감도 분해로 보고하는 실패 |
| G-TABLE | 공개 오대응·수지 상태 혼동·선 입력 오연결·비가산 효과의 기여율 오인을 표에 싣는 실패 |
| G-RUN | GPU 사용·생산 변경·미완료·비용 초과를 등록 실행으로 보고하는 실패 |

검사는 주 계산의 원자료·메모리 결과에 적용한다. 저장 표를 다시 읽어 검사하지 않는다. 기존 사전등록·참조·정본·발주·러너 위생 검사는 기존 도구로 집행하며 새 장치를 추가하지 않는다.

## §10. 비용·실행·재현

| 항목 | 상한 |
|---|---|
| GPU·생산 실행 | 모두 0 |
| 오프라인 | a10 CPU Slurm, 작업명 `LUMINA_` 접두 |
| 작업당 | 1 task·2 CPU·8 GiB·벽시계 300 s |
| 누적 | 할당 벽시계 900 s·0.5 CPU core·h |
| 수치 스레드 | BLAS·OpenMP 각 1 |
| 정상해 | 등록 계산에서 최대 6개; 조합·매개변수 탐색 없음 |
| 신규 저장 | 100 MiB 이하 |

직접시험·기계 검사·분석·수정 재실행·감사 재계산을 누적 비용에 포함한다. 로그인 노드에서는 파일 조회·편집·제출·결과 열람만 한다.

러너는 `module purge` 뒤 명시한 모듈을 로드하고 Python·NumPy 판본을 기록한다. 요청은 `--partition=a10 --ntasks=1 --cpus-per-task=2 --mem=8G --gres=gpu:0 --time=00:05:00 --exclude=syn102,syn03`이며 실제 GPU 할당 0을 확인한다.

작업 내부 인터페이스는 다음과 같다.

```bash
python3 -B scripts/det_si_bb_groups.py negative
python3 -B scripts/det_si_bb_groups.py analyze \
  --run /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1 \
  --baseline /gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/results/run1/evidence.json \
  --out /gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/results/run1
```

재현 시 같은 코드·입력·명령을 사용하되 새 출력 경로를 지정한다. 상한 내에서만 재계산한다. 미실행 재계산을 재현 통과로 기록하지 않는다.

## §11. 폐합·주장 상한·집행 기록

폐합에는 등록 묶음 전체의 결과·F0/F8 대조·선택 준위 수지·선 입력·비가산 잔차가 연결된 주 표와 운전자 판정, astra 종료감사가 필요하다. 중간 검사 통과만으로 폐합하지 않는다.

종료감사는 우선 표의 수치·공개 대응·준위와 선의 대응·비선형성 해석·주장 한정을 본다. 과학적 방향이나 큰 효과를 합격 조건으로 삼지 않는다.

생산 승격·물리적 정답·β 오류의 확정·자기일관 효과·스펙트럼 개선·다른 셸 및 철족 일반화를 주장하지 않는다. 개별 선 정상해를 풀지 않았으므로 대표선을 III/II 변화의 인과적 주범으로 부르지 않는다.

**[저자 실행]** `pwd`, 직전 계약·판정·사용자 결정의 `cat`/`sed`, 기존 함수의 `rg`/`sed`, 기록 CSV 머리글의 `head`로 프로젝트 경로·사용자 선택·재사용 함수·필드·직전 수치를 확인했다. `det_si_bb_counterfactual.py` 약 99–161행의 처방·조립·검사와 `det_si_trace_readout.py` 약 433행의 훅을 읽었다. 새 계산은 실행하지 않았다.

**[미실행]** 입력 해시·전체 채택 재생·묶음 건수·직접시험·기계 검사·정상해·주 표·비용 실측은 운전자가 §9–§10에 따라 집행한다.

| 항목 | 담당 | 상태·증거 |
|---|---|---|
| 사용자 결정 | 사용자 | B의 `evidence/USER_DECISION_2026-10-11.txt` |
| 기안·개정 | astra | 본문 기안; 개정 없음 |
| 구현·직접시험·게이트·실행 | 운전자 | 미집행 |
| 경량 사전감사·코드 감사 | 해당 없음 | §1 규약 적용 |
| 판정 | 운전자 | 미집행 |
| 종료감사 | astra | 미집행; 계약 저자 이해관계 명시 |
| 비용·이탈 | 운전자 | 최종 누계·미실행 항목 결과·중단 사유와 담당 기록 |

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011`. 작업 번호·경과는 `B/JOBS.txt`·`B/evidence/sacct_offline_jobs.txt`, 게이트 집행은 `B/evidence/GATE_RECORD.json`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| 기안 | astra(자기보고 「gpt-6-astra」) | ade71e9(개정 없음) |
| 기계 게이트(기안) | 운전자 | 기안 브리프 canonical(a10 419360), 기안 prereg·referents·canonical rc 0(419364) |
| 구현 | 운전자(Opus 5.5) · 코드 감사 없음(25-z) | 615d35e — `scripts/det_si_bb_groups.py`(기존 판독기 3개 무수정), 러너 `B/tools/run_det_si_bb_groups_offline.slurm` |
| 직접시험(§9) | 운전자 | a10 419366 **16/16**(양성 대조 4·결함 주입 거부 12), 러너 위생 rc 0 |
| `[미실행]` 항목 집행 | 운전자 | 입력 해시·채택 재생·묶음 건수·정상해·주 표는 run1 에서, 직접시험·기계 검사는 위 작업에서, 비용 실측은 sacct 로 집행 |
| 등록 계산 run1 | 운전자 | a10 419367(5 s) `B/results/run1/`: G-INPUT·G-RATE·G-SOLVE·G-TABLE·G-RUN 지적 0; 묶음 G1 62·G2 151·G3 956·G4 1418; F0·F8 = 기준(0 dex); **분기 0 GROUP_EFFECT_RESOLVED** — Δ G1 −0.804·G2 +0.103·G3 +0.032·G4 −0.001 dex, 비가산 잔차 −0.167 dex |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_SI_BB_GROUPS_2026-10-11.md` — GROUP_EFFECT_RESOLVED(acc6069), 1회차 «기재로 족함» 4 반영, **폐합** |
| 종료감사 | astra(계약 저자 이해관계; 자기보고 「GPT-6, 세부 식별자 확인 불가」) | 1회차 **폐합 가 «폐합 차단» 0**·«기재로 족함» 4(`B/astra_closeout.md`) → 판정문 §2·§5 반영(판정문 §8) → **폐합**; 독립 재계산은 재현 BLOCKED |
| 기계 게이트(판정·감사 발주) | 운전자 | 판정 roster·prereg preflight·referents·canonical rc 0(a10 419368), 감사 발주 dispatch·canonical rc 0(419369), 폐합본 4종 rc 0(419370) |
| 비용 | 운전자 | GPU 0; 오프라인 a10 7 작업 할당 98 s · 0.029 core·h / 900 s · 0.5 core·h(기안 게이트 2 작업 31 s·판정/감사/폐합 게이트 3 작업 61 s 포함); 신규 저장 2.0 MiB / 100 MiB |

**이탈(기재):** 과학 분기에 영향을 주는 이탈은 확인되지 않았다. 주 표가 §6 의 일부 등록 항목(처방 β=1 별도 열·묶음별 에너지 범위·합계 행의 F0 대비 차)을 생략했고, 영 채널 행과 네 잔차가 모두 0 인 비가산 행을 설명 없이 생략했다(종료감사 1회차 지적 3; 판정문 §8). 표시본 `.md` 는 같은 메모리 행을 행 종류별 소표로 나눴다(§6 허용).

## §12. 기계 선언

참조는 실재 파일만 지정한다. 예정 신규 파일은 변경집합에만 둔다. 분기 선언은 기존 프리플라이트의 분류 형식 검사이며 과학 판정을 재계산하는 장치가 아니다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SI-BB-GROUPS (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_si_bb_groups\\.py|/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/tools/run_det_si_bb_groups_offline\\.slurm)",
    "symbol": "det_si_bb_groups",
    "roots": [
      "scripts/det_si_bb_groups.py",
      "/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/tools/run_det_si_bb_groups_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_si_bb_groups.py",
      "/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/tools/run_det_si_bb_groups_offline.slurm"
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
      {"name":"GROUP_EFFECT_RESOLVED","predicate":"valid and c0 == 1"},
      {"name":"NO_GROUP_EFFECT_RESOLVED","predicate":"valid and c1 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"resolved","mix":[[0,1]]},
      {"name":"unresolved","mix":[[1,1]]},
      {"name":"indeterminate","mix":[[2,1]]},
      {"name":"invalid","mix":[[9,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md"},
    {"path":"scripts/det_si_bb_counterfactual.py","flags_existing":["det_si_bb_d8","det_si_bb_prescription_check","det_si_bb_build","det_si_bb_invariance","det_si_bb_solve"]},
    {"path":"scripts/det_si_trace_readout.py","flags_existing":["replay","det_si_exc_build_fl","bb_radiative_override"]},
    {"path":"scripts/det_si_excitation_budget.py","flags_existing":["det_si_exc_adopt","det_si_exc_level_map","det_si_exc_budget","det_si_exc_public"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/evidence/USER_DECISION_2026-10-11.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_ops.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_slmap.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/results/run1/evidence.json"},
    {"path":"data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt","flags_existing":["19.480","si1","si2"]}
  ]
}
```
