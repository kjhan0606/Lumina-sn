# 사전등록 — DET-JC-COARSE-CF: 순복사 브래킷 입사장의 거친 구간장 대입 반사실

2026-10-11 · astra 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_JC_COARSE_CF_2026-10-11.md`  
상태: **기안. 기존 문서·판독기·입력 메타데이터 조회 완료. 구현·계산·게이트·판정 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 이번 단은 채택 소비 상태의 고정 장에서 모드 8 입사장만 거친 구간장으로 대입했을 때 Si 셸10 III/II가 움직이는 양을 잰다.

그 결과는 이 입사장 대입 축을 후속 조사할지, 해당 처방의 영향이 작다는 근거로 우선순위를 낮출지를 결정한다. 생략하면 대표선의 큰 장 비율 차이를 이온화 효과로 간주하거나, 대표선 밖의 반응·상쇄를 놓칠 수 있다.

**Q-LEAN — 불필요한 측정·게이트를 더하는가? 자기답: APPROPRIATE.**

주 산출은 범위별 과학 표 하나다. 전체 모드 8, 기존 G1–G4 독립 대입, 기존 대표선 여섯 개의 공동 대입만 비교한다. 대표선 재선정·개별 선 정상해·새 준위 수지·불투명도 원인 분해는 하지 않는다.

기존 채택·선 대응·묶음 배정·행렬 조립·불변성·풀이기를 재사용한다. 새 수치 코드는 대입 처방과 그 음성 대조로 한정하며, 표 작성에 필요한 연결만 추가한다.

저장 산출 재검·쓰기 결함 모사·결정표 재계산 장치를 만들지 않는다. 직접시험은 새 코드의 음성 대조에 한정한다. 종료감사의 1차 대상은 **과학 표의 수치·대응·주장 한정**이다.

## §1. 권한·입력·고정 기준

사용자는 「J_c→거친 Jν 반사실」을 선택했다. 기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사는 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra가 수행한다(25-ac).

| 기호 | 대상 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/jccf_20261011` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| D | `/gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011/results/run3` |
| F | `/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/results/run1` |
| G | `/gpfs/kjhan/lumina/cmfgen_map/bbgrp_20261011/results/run1` |
| L | `/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/results/run1` |

주 입력은 T의 채택 기록, 원자 입력, 생산 원장, 비교 CSV와 manifest다. 채택 조건은 기존 판독기의 묶음1·전자밀도 반복18·CE 패스1·호출36, Si 셸10을 계승한다.

직전 `DET-JC-DEFINITION`의 분기 2·정의 판별 목표 미달·Δ_def 미식별을 그대로 보존한다. 이번 대입 결과로 그 판정을 덮어쓰지 않는다.

실제로 소비한 파일의 경로·크기·SHA-256, 실행 코드 판본, 명령·작업 번호를 보조 증거에 한 번 기록한다. 전체 트리 목록·해시는 만들지 않는다.

## §2. 과학 질문과 잣대

> 채택 호출36의 기록 FL 연산자에서 모드 8 순복사 브래킷의 입사장 \(J_c\)를 같은 셸·같은 복사 세대의 저장 거친 구간 \(J_\nu\)로 바꾸고, 나머지 율과 고정 조건을 유지하면 Si III/II는 F0와 공개 값에 대해 얼마나 움직이는가?

거친 장은 비교 CSV의 `J_nu`다. 원 방사장 구간값을 목표 구간과 겹치는 주파수 폭으로 적분한 뒤 목표 구간 폭으로 나눈 값이다.

\[
J_{\mathrm{coarse},b}
=\frac{\sum_k J_{\nu,k}\,
 |[\nu_{k,\mathrm{lo}},\nu_{k,\mathrm{hi}}]
 \cap[\nu_{b,\mathrm{lo}},\nu_{b,\mathrm{hi}}]|}
 {\nu_{b,\mathrm{hi}}-\nu_{b,\mathrm{lo}}}.
\]

이는 Hz에 대한 적분 보존 구간상수 재격자화다. 선 중심의 분해된 강도, 선 프로파일 가중 평균, log-ν 평균이 아니다. 저장 CSV를 다시 재격자화하지 않는다.

거친 장은 선을 제거한 입사장으로 생성·제공된 자료가 아니다. 원 거친 전달장의 선 숲 효과를 포함할 수 있으며, 대입 전에 자기 선·다른 선 기여를 차감하지 않는다. 비교 구간의 `chi_bb`를 해당 \(J_\nu\)의 선별 기여율로 읽지 않는다.

**세대 대응은 다음처럼 한정한다.**

- 비교 manifest: 인구13·복사12·불투명도23·방출23.
- 채택 소비 호출36: 복사12를 소비하는 인구13 상태.
- \(J_c\) 생산: 호출12·물질12·복사12.
- 따라서 같은 복사 세대의 저장장 대입은 가능하지만, 같은 물질·불투명도·격자에서 입사장 정의만 바꾼 장 쌍은 아니다.

같은 복사 세대라는 번호만으로 대응을 인정하지 않는다. T의 동일 실행 신원, 채택 호출의 `rad_generation`, 비교 manifest, 생산 원장의 연결을 함께 소비한다. 세대 불일치 시 다른 덤프를 임의로 고르지 않는다.

CMFGEN `SOBJBAR_SIM`의 입사 항은 광선별 \(\beta_k AV_k\)의 각도 적분이다. 거친 \(J_\nu\)는 그 `AV`나 유효 입사장과 동일하지 않다. 공개 모델의 해당 선들이 그 경로를 사용했다는 실행 대응도 확보되지 않았다.

이번 단은 **저장장 대입에 대한 고정 율계의 조건부 민감도**를 가른다. 입사장 정의의 옳고 그름, 자기 선 이중 계산, 미확정 비대응 선 차감, Δ_def의 크기는 가르지 못한다.

## §3. 대입 범위

모든 상태는 F0에서 독립적으로 조립한다. 앞 상태의 변경이나 인구를 다음 상태에 누적하지 않는다.

| 상태 | 대입 대상 | 선택 이유 |
|---|---|---|
| F0 | 없음 | 같은 풀이기의 기준 정상해 |
| C-ALL | 채택 Si 모드 8 전체 | 대표선 선택 편향 없이 주 질문에 답함 |
| C-G1 | Si II 하준위 `level_num` 0·1 | 기존 바닥항 출발 묶음 |
| C-G2 | Si II 하준위 6·7·9·10 | 기존 중간 준위 출발 묶음 |
| C-G3 | 나머지 Si II 모드 8 | G1·G2 밖의 반응 확인 |
| C-G4 | Si III 모드 8 전체 | 상위 이온 내 재분배 확인 |
| C-REP6 | 아래 여섯 진술 공동 대입 | 사용자 선택지의 대표선 실험과 전체 결과 연결 |

기존 묶음 건수는 G1=62, G2=151, G3=956, G4=1418, 전체=2587이다. 대상은 `aux`만으로 선택하지 않고 `(call,op)`와 선·이온·상하 준위 대응으로 확정한다.

C-REP6의 선 번호는 760321·632403·757019·1676767·1353542·1679119로 고정한다. 앞의 세 선은 G1, 뒤의 세 선은 G2다. 재순위화하거나 결과를 보고 교체하지 않는다.

묶음별 이동은 독립 유한 개입의 반응이다. 합산 기여율·전체 효과의 인과적 분해로 보고하지 않는다. 비가산성 전용 산출도 추가하지 않는다.

## §4. 대입 처방·불변 대상

유효 대상 진술 \(e\)에서 기록값으로 다음을 정의한다.

\[
J_{c,e}=c_e/\beta_e,\qquad
q_e=J_{\mathrm{coarse},b(e)}/J_{c,e},\qquad
c'_e=\beta_eJ_{\mathrm{coarse},b(e)}.
\]

\[
r'_{a,e}=q_er_{a,e},\qquad
r'_{s,e}=q_er_{s,e},\qquad
r'_{p,e}=r_{p,e},\qquad
\bar J'_e=c'_e+\ell_e.
\]

이는 \(B_{lu}=r_a/c\), \(B_{ul}=r_s/c\)를 고정한 \(J_c\) 대입이다. β를 다시 나누거나 자발율을 함께 확대하지 않는다.

기록 \(\beta,A,\tau_{\mathrm{eff}},\ell\), 충돌율, BF·DR·CE·NT, 모드 7 및 비대상 모드 8의 율은 불변이다. 온도·전자밀도·CE 상대 인구·Si 총량·다른 채널이 소비하는 복사장은 고정한다.

반사실 정상해의 인구는 변한다. 그 인구로 β·광학깊이·국소 항·전자밀도·다른 율을 갱신하지 않는다. 기록의 `f_lo`·`f_up`도 덮어쓰지 않는다.

기존 `bb_radiative_override` 훅에 복사 3율만 전달한다. \(\bar J'\)는 처방 설명용 파생량이며, 이를 전체장 처방의 Einstein 율식에 다시 넣어 국소 항을 이중 가산하지 않는다.

## §5. 정의되지 않는 대입과 경계 처리

선 주파수는 연결된 원자 입력의 원 주파수를 사용한다. 반올림 파장으로 구간을 찾지 않는다.

구간은 `[nu_lo_Hz,nu_hi_Hz)`로 선택하고, 전체 격자의 마지막 상단 끝점만 포함한다. 내부 경계에 정확히 놓인 중심은 높은 주파수 쪽 구간에 속한다.

선 프로파일이 여러 구간에 걸쳐도 이번 처방은 **중심 구간값 대입**으로 정의된다. 가우스 평균을 복원하지 않는다. 해당 여부는 기존 Doppler 폭을 사용해 표시하되, 경계 근처라는 이유로 새 평균·보간·외삽을 도입하지 않는다.

다음은 해당 진술의 대입을 미정의로 처리한다.

- 선·이온·상하 준위·생산/소비 연결의 누락·중복·불일치.
- 구간 누락·중첩·역전, 범위 밖 주파수, 단위·좌표계 불일치.
- 비유한 값, \(\beta\le0\), \(c<0\), 음의 거친 장.
- \(c=0\), 또는 \(J_c=c/\beta\)·\(q\)·대입율을 유한한 값으로 계산할 수 없음.
- 기록 율의 0이 아닌 값이 대입 과정에서 표현 한계로 0이 되는 등 수치 표현 실패.

\(c=0\)이면 \(J_c=0\) 자체는 알 수 있어도 기록율에서 B 계수와 비율 대입을 복원할 수 없다. 원자 B 계수로 우회하는 별도 처방은 이번 단에 추가하지 않는다.

반대로 \(c>0\), \(J_{\mathrm{coarse}}=0\)이면 \(q=0\)이며 흡수·유도율을 정확히 0으로 두는 유효 대입이다. 로그 장비율은 계산하지 않고 영값 건수를 표시한다.

음의 광학깊이는 제외 사유가 아니다. 기존 기록 조건을 만족하는 유한한 양의 β를 그대로 둔다.

**범위 안에 미정의 진술이 하나라도 있으면 그 범위의 정상해를 산출하지 않는다.** 정의된 부분만 바꾼 결과를 전체·묶음 결과로 표시하지 않는다. 다른 완전한 범위는 독립적으로 계산할 수 있다.

미정의 건수와 사유를 주 표에 표시한다. clamp/floor, 작은 양수 대체, β 상한, 임의 제외는 금지한다.

## §6. 주 산출과 수치 기준

정본은 `jc_coarse_cf_table.csv`, 표시본은 같은 표의 `jc_coarse_cf_table.md`다. 공개 행과 §3의 일곱 상태 행을 둔다.

| 열 | 내용 |
|---|---|
| 상태·범위 | 상태명, 대상 정의 |
| 적용 현황 | 예정·정의 가능·실제 대입·미정의 진술 수 |
| III/II | 해당 상태의 정상해 비 \(R\) |
| F0 대비 | \(\Delta=\log_{10}(R/R_0)\) |
| 공개 대비 위치 | \(d=\log_{10}(R/R_{\mathrm{pub}})\), 아래/위/일치 |
| 공개 접근량 | \(G=|d_0|-|d|\) |
| 장비율 분포 | 실제 대입 진술의 \(q\) 최솟값·중앙값·최댓값, \(q=0\)·\(q<1\)·\(q=1\)·\(q>1\) 건수 |
| 경계·판정 | 프로파일 구간 횡단 건수, 행 분류, 미정의·풀이 실패 사유 |

분포는 진술당 같은 가중치다. 같은 구간을 공유하는 진술도 각각 센다. 짝수 표본의 중앙값은 가운데 두 값의 산술평균이다. 미산출 상태의 분포는 공란으로 두고 적용 현황으로 이유를 밝힌다.

맥락 인용값은 F0 \(R_0=3.5597568125822305\), 공개 Si II=0.0092499·Si III=0.97769, 공개 III/II≈105.697이다. 공개 대비 계산에는 두 공개 분율의 비를 사용한다.

등록 대응은 19.480 d, Lumina 11,544 km/s ↔ 공개 11,534.59 km/s다. 채택 `FINAL_FL`의 III/II=3.563683은 F0와 구별되는 맥락값이며 별도 정상해로 재해석하지 않는다.

F0 재조립은 기존 기준 대비 0.001 dex 이내여야 한다. 각 정상해는 기존 풀이기의 유한·비음·총량 보존·닫힌 통신류 조건과 모든 행 후방오차 \(10^{-8}\) 이하를 만족해야 한다. III/II와 그 로그는 양의 이온 합에만 정의한다.

처방 수치 대조는 \(M=\max(|a|,|b|)>0\)이면 \(|a-b|/M\le10^{-12}\), \(M=0\)이면 정확한 0을 요구한다. 이산 신원은 정확히 일치해야 한다. 기존 채널 불변성 검사를 재사용한다.

보조 산출은 표에 소비되는 연결·처방값·미정의 사유·풀이 유효성·입력 신원을 담은 `evidence.json`, 실행 로그, 새 코드 음성 대조 결과뿐이다. 별도 선 순위·준위 수지 표를 만들지 않는다.

## §7. 사전 기대와 결과 해석

직전 여섯 대표선은 거친 장이 \(J_c\)보다 컸다. 같은 연결이 유지되면 C-REP6에서 흡수·유도율은 증가하고 자발율은 그대로일 것으로 예상한다.

전체 모드 8의 장비율 분포, 묶음별 III/II 이동의 부호·크기, 공개 접근 여부는 예측하지 못한다. 대표선에서 장이 커진다는 사실은 III/II 증가를 보장하지 않는다. 이전 β=1 처방의 결과를 이번 처방의 기대 부호로 옮기지 않는다.

행별로 \(|\Delta|\le0.01\) dex면 **이동 미분리**, 그보다 크면 **이동 분리**다. 공개 접근은 \(G>0.01\), 이격은 \(G<-0.01\), 나머지는 **공개 거리 변화 미분리**로 쓴다. 경계값은 미분리에 포함한다.

0.01 dex는 이 단의 보고 기준이며 통계적 유의성·물리적 허용오차가 아니다. 공개 값을 넘어서는 경우에도 부호만으로 접근이라 하지 않고 절대 로그 거리로 판정한다.

| 코드 | 전체 분기 | 조건 |
|---:|---|---|
| 0 | `RESOLVED_APPROACH` | 모든 등록 상태가 유효하고 C-ALL의 \(G>0.01\) dex |
| 1 | `NO_RESOLVED_APPROACH` | 모든 등록 상태가 유효하고 C-ALL의 \(G\le0.01\) dex |
| 2 | `INDETERMINATE` | 필수 대응·대입·풀이·실행 근거 중 부족한 항목이 있음 |

분기1에서도 C-ALL의 이동 크기와 묶음 반응을 반드시 구별한다. 전체 이동이 작고 일부 묶음 이동이 크면 상쇄 가능성을 남기며 “입사장 축 전체가 작다”고 결론내리지 않는다.

모든 대입 상태가 이동 미분리일 때만 **이 저장장 대입 처방의 영향이 작다**고 보고한다. 그래도 다른 입사장 처방·자기일관 효과의 부재를 뜻하지 않는다. 후속 우선순위는 사용자가 결정한다.

C-ALL이 유효해도 다른 필수 행이 미산출이면 전체 분기2다. 유효한 C-ALL 수치는 제한된 결과로 보존한다. 미정의·풀이 실패를 0 dex나 무효과로 바꾸지 않는다.

## §8. 최소 변경집합·보존

### 변경집합 — DET-JC-COARSE-CF (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_si_line_input.py` | `jccf`·`jccf-negative` 추가: 전체 대상 연결, 등록 대입 콜백, 기존 조립·풀이기 호출, 주 표 작성 |
| `/gpfs/kjhan/lumina/cmfgen_map/jccf_20261011/tools/run_det_jc_coarse_cf_offline.slurm` | 제한 CPU 실행·입력 신원·비용 기록 |

### 변경집합 끝

`det_si_trace_readout.py`의 대체 훅, `det_si_bb_counterfactual.py`의 채널 조립·불변성·풀이, `det_si_bb_groups.py`의 묶음 배정, `det_si_excitation_budget.py`의 채택·준위 대응을 무수정 재사용한다.

기존 `det_si_line_input.py`의 원자·원장 연결과 `det_si_jcdef_bin`을 재사용한다. 여섯 선에 한정된 선택을 전체 대상에 적용할 연결만 확장하며 기존 부명령 동작은 바꾸지 않는다.

생산 소스·바이너리·실행 입력·CMFGEN 소스 변경은 0이다. 신규 결과는 B의 새 실행 경로에만 쓴다. 기존·실패 산출은 보존하며 원자료를 덮어쓰지 않는다. 일반 파일 읽기·쓰기만 사용하고 하드링크를 만들지 않는다.

## §9. 직접시험·게이트

새 코드의 음성 대조는 유효 대조와 실제 결함 주입을 짝지어 다음에 한정한다.

- 다른 셸·복사 세대·선·준위 연결, 중복·누락 대상의 거부.
- 내부 경계·마지막 끝점의 등록 선택과 잘못된 구간 연결·중첩의 거부.
- \(c=0\)·β 비양수·비유한 장·음의 장 거부와 정확한 거친 영값 허용.
- 자발율 변경·β 중복 적용·한쪽 펌프율만 변경·비대상 변경·묶음 누적의 검출.

기존 풀이기·과학 분기·저장 파일을 별도 시험 장치로 복제하지 않는다.

| 게이트 | 막는 실재 실패 |
|---|---|
| G-INPUT | 다른 호출·선·셸·복사 세대·주파수 구간의 값을 연결하는 실패 |
| G-RATE | \(J_c\) 대입 외에 β·자발율·충돌·비대상 채널을 바꾸는 실패 |
| G-SOLVE | 다른 F0 또는 유효하지 않은 정상해를 III/II 결과로 사용하는 실패 |
| G-CLAIM | 부분 대입을 전체 결과로, 거친 장을 CMFGEN 입사장·수리 증거로 표시하는 실패 |
| G-RUN | GPU·생산 변경·미등록 재실행·비용 초과를 등록 결과에 섞는 실패 |

정직하게 표시한 대입 미정의는 구현 결함의 확정이 아니다. 해당 행 미산출과 분기2로 처리한다. 기존 기계 프리플라이트는 기존 도구로 집행하며 새 검사기를 추가하지 않는다.

## §10. 비용·실행·재현

| 항목 | 상한 |
|---|---|
| GPU·생산 실행·복사전달 | 모두 0 |
| 실행 위치 | a10 CPU Slurm, 작업명 `LUMINA_` 접두 |
| 작업당 | 1 task·2 CPU·8 GiB·벽시계 300 s |
| 누적 | 할당 벽시계 합 900 s·0.5 CPU core·h |
| 스레드·신규 저장 | OpenMP·BLAS 각 1, 50 MiB |

직접시험·기계 검사·수정 실행·감사 재계산을 누계에 포함한다. 로그인 노드에서는 파일 조회만 한다.

Slurm 요청은 `--partition=a10 --ntasks=1 --cpus-per-task=2 --mem=8G --gres=gpu:0 --time=00:05:00 --exclude=syn102,syn03`이다. 실제 GPU 할당 0을 확인한다. 러너는 `module purge` 후 사용할 모듈을 명시하고 Python·수치 라이브러리 판본을 기록한다.

작업 내부의 예정 인터페이스:

```bash
python3 -B scripts/det_si_line_input.py jccf-negative
python3 -B scripts/det_si_line_input.py jccf \
  --run /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1 \
  --out /gpfs/kjhan/lumina/cmfgen_map/jccf_20261011/results/run1
```

재현에는 같은 입력·코드·명령과 새 출력 경로를 사용한다. 별도 저장 산출 검증 모드는 만들지 않는다. 비용 상한 내 미완료이면 실제 미완료 항목을 기록하고 자원·과학 범위를 자동 확대하지 않는다.

## §11. 폐합·주장 상한·집행 기록

폐합에는 등록 범위의 과학 표, 운전자 판정, astra 종료감사가 필요하다. 분기2는 미달 목표와 부족한 대응·진술·상태를 명시한 기록 폐합으로만 허용한다.

종료감사는 표의 III/II·dex·공개 대응, 적용 범위·장비율 분포, 자발율 불변, 미정의 처리와 주장 한정을 우선한다. 기안자와 종료감사자가 같은 astra라는 이해관계를 밝힌다. 저장 증거 확인과 독립 실행 재현을 구분한다.

공개에 접근해도 생산 승격·물리적 정답·수리·자기일관 효과·스펙트럼 개선·철족 일반화를 주장하지 않는다. 이격해도 원 \(J_c\)의 정당성이 입증된 것은 아니다.

기안자는 공유 맥락, 현행 프로젝트 규약, 사용자 결정, 직전 계약·판정, 판독기, 비교 manifest와 재격자화 소스를 조회했다. 작업 디렉터리가 P임을 확인했다. 수치 계산·해시·파일 작성·시험·제출은 하지 않았다.

저자 조회에는 다음 명령을 사용했다. 두 조회는 정의·기록 확인이며 실행 검증이 아니다.

```bash
cat /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/physics_comparison/physics_DET_iter0010.manifest.json
sed -n '45,165p' scripts/det_si_bb_counterfactual.py
```

운전자는 이 절에 기계 프리플라이트 반환값, 구현 판본, 직접시험, 등록 계산, 누적 비용, 판정·종료감사, 중단 사유와 실제 담당을 추가한다. 실행 전 항목은 미집행으로 남긴다. 분장·집행 장부를 다른 문서에 복제하지 않는다.

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/jccf_20261011`. 작업 번호·경과는 `B/JOBS.txt`·`B/evidence/sacct_offline_jobs.txt`, 게이트 집행은 `B/evidence/GATE_RECORD.json`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| user 결정 | user | 「J_c→거친 Jν 반사실」(`B/evidence/USER_DECISION_2026-10-11.txt`) |
| 기안·개정 1 | astra(자기보고 「gpt-6-astra」·「GPT-6」) | 0b194a5 · 6d844de(OLD/NEW 1쌍, §1 일본어 문단 정리, 행 단위 1행 치환·의도 외 소실 0) |
| 기계 게이트(기안·개정) | 운전자 | 기안 브리프 canonical rc 0(a10 419445), 기안 prereg·referents·canonical rc 0(419452), 개정 브리프 dispatch·canonical rc 0(419456), 개정 후 prereg·referents·canonical rc 0(419457) |
| 자료 조건 확인 | 운전자 | 개정 1 발주 전 awk 집계 2건(풀이 없음): 호출 36 모드 8 진술 2587, c=0·c<0·β≤0·r_a=0 각 0, 셸10 비교 격자 1234 구간 5.84×10¹³–4.04×10¹⁶ Hz·J_nu=0 구간 0(a10 419450); 2587 선 모두 생산 원장 행 1개·status OK(a10 419454) — `B/evidence/precheck_*.out` |
| 구현 | 운전자(Opus 5.5) · 코드 감사 없음(25-z) | e65bc93 — `scripts/det_si_line_input.py` 에 `jccf`·`jccf-negative` 추가(기존 부명령·재사용 판독기 무수정), 러너 `B/tools/run_det_jc_coarse_cf_offline.slurm` |
| 직접시험(§9) | 운전자 | a10 419462 **36/36**, 러너 위생 rc 0 |
| 등록 계산 run1 | 운전자 | a10 419463(8 s) `B/results/run1/`: G-INPUT·G-RATE·G-SOLVE·G-CLAIM·G-RUN 지적 0, 미정의 0; **분기 0 RESOLVED_APPROACH** — C-ALL III/II 3.5598→3.7222(Δ +0.0194 dex, G +0.0194), C-G1 +0.0300(접근)·C-G2 −0.0174(이격)·C-G3 +0.0072·C-G4 +0.000004·C-REP6 +0.0025(뒤 셋 미분리) — 종료감사 1회차 «폐합 차단» 2(미정의 처리 구현 결함)로 대체, 보존 |
| 종료감사 반영 run2 | 운전자 | 판독기 수정 ee237f8(표현 실패 미정의, 진술 단위 중복·브래킷 조건·격자 지적 미정의, 범위 단위 미배정·등록 건수·대표선 해석 정지, manifest frame 검사), 음성 대조 a10 419471 **46/46**, a10 419472(8 s) `B/results/run2/`: 판독기 게이트 지적 0, 미정의 0·범위 정지 0; **분기 0 RESOLVED_APPROACH** — 수치·분류 run1 과 같음 |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_JC_COARSE_CF_2026-10-11.md` — 1차 a5535fd(run1) → 종료감사 1회차 반영(run2, 분기 0 유지) |
| 종료감사 | astra(계약 저자 이해관계; 자기보고 「GPT-6」, 실행 헤더 gpt-6-astra) | 1회차 **폐합 불가 «폐합 차단» 2**·«기재로 족함» 2(`B/astra_closeout.md`) → 반영(판정문 §8, run2); 2회차 **폐합 가 «폐합 차단» 0**·«기재로 족함» 1(`B/astra_closeout_r2.md`: 음성 대조 설명이 검사 범위보다 넓음) → 반영(536ba75 설명 문자열만, 음성 대조 a10 419492 46/46; 판정문 §9); 독립 재계산은 재현 BLOCKED |
| 폐합 | 운전자 | 분기 0 RESOLVED_APPROACH(run2) 유지 폐합(판정문 처분·§9) |
| 기계 게이트(판정·감사 발주) | 운전자 | 판정 roster(5/5)·prereg preflight·referents·canonical rc 0(a10 419465), 1회차 감사 발주 dispatch·canonical rc 0(419466), 1회차 반영 roster·prereg preflight·referents·canonical rc 0(419474), 2회차 감사 발주 dispatch·canonical rc 0(419475); 폐합 게이트는 `B/JOBS.txt` |
| 비용 | 운전자 | GPU 0; 오프라인 a10 15 작업 할당 151 s · 0.047 core·h / 900 s · 0.5 core·h(게이트 8 작업 131 s·자료 확인 2 작업 2 s 포함, 2회차 반영 음성 대조까지); 신규 저장 3.3 MiB / 50 MiB |

**이탈(기재):** ① 개정 1 발주 전 운전자가 자료 조건을 awk 로 두 번 집계했다(계약 밖 작업, 풀이 없음, 비용 누계 포함). ② 비용 집계 중 운전자가 로그인 노드에서 의미 없는 `python3 -c 1` 을 한 번 실행했다(연산 없음, 규약 위반으로 기재). ③ 종료감사 1회차 지적에 따라 판독기를 고쳐 run2 를 새 경로로 산출했다(run1 보존).

## §12. 기계 선언

참조에는 기존 실재 파일만 둔다. 예정 러너는 변경집합에만 둔다. 분기 fixture는 기존 프리플라이트의 문면 검사에만 사용하며 과학 결정표를 재계산하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-JC-COARSE-CF (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_si_line_input\\.py|/gpfs/kjhan/lumina/cmfgen_map/jccf_20261011/tools/run_det_jc_coarse_cf_offline\\.slurm)",
    "symbol": "det_si_line_input",
    "roots": [
      "scripts/det_si_line_input.py",
      "/gpfs/kjhan/lumina/cmfgen_map/jccf_20261011/tools/run_det_jc_coarse_cf_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_si_line_input.py",
      "/gpfs/kjhan/lumina/cmfgen_map/jccf_20261011/tools/run_det_jc_coarse_cf_offline.slurm"
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
      {"name":"RESOLVED_APPROACH","predicate":"valid and c0 == 1"},
      {"name":"NO_RESOLVED_APPROACH","predicate":"valid and c1 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"approach","mix":[[0,1]]},
      {"name":"no_approach","mix":[[1,1]]},
      {"name":"indeterminate","mix":[[2,1]]},
      {"name":"invalid","mix":[[9,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_JC_DEFINITION_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_JC_DEFINITION_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_LINE_INPUT_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_BB_GROUPS_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md"},
    {"path":"scripts/det_si_line_input.py","flags_existing":["det_si_line_input_match","det_si_jcdef_bin"]},
    {"path":"scripts/det_si_trace_readout.py","flags_existing":["det_si_exc_build_fl","bb_radiative_override"]},
    {"path":"scripts/det_si_bb_counterfactual.py","flags_existing":["det_si_bb_build","det_si_bb_invariance","det_si_bb_solve"]},
    {"path":"scripts/det_si_bb_groups.py","flags_existing":["det_si_bb_groups_assign"]},
    {"path":"scripts/det_si_excitation_budget.py","flags_existing":["det_si_exc_adopt"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/jccf_20261011/evidence/USER_DECISION_2026-10-11.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_ops.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/w5_lines.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/model/line_list.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/model/levels.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/resolved_lumina.exports"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/physics_comparison/physics_DET_iter0010.spectral.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/physics_comparison/physics_DET_iter0010.manifest.json"}
  ]
}
```
