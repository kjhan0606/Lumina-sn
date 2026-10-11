# 사전등록 — DET-SI-BB-COUNTERFACTUAL: Si 셸10 BB 복사 경계 처방의 고정 장 효과

2026-10-11 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md`  
상태: **기안. 구현·직접시험·기계 게이트·계산·판정·종료감사 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 직전 단에서 선택 Si II 준위의 주요 공급·배출이 BB 복사임을 확인했지만, 복사율의 크기만으로 III/II 차이의 원인을 판정할 수는 없다.

이번 단의 끝점은 **영역 밖 자발붕괴와 영역 안 탈출확률 처방을 각각 바꾼 고정 장 정상해·선택 준위 수지를 CMFGEN 대비 표 하나로 제시하는 것**이다. 이를 통해 다음 조사에서 BB 영역 경계 또는 영역 안 복사 결합을 우선 검토할 근거가 있는지 판단한다. 생략하면 큰 BB 수지를 원인 증거로 오인하거나, 실제 민감도 없이 생산 변경을 선택하게 된다.

**Q-LEAN — 불필요한 측정·게이트를 더하는가? 자기답: APPROPRIATE.**

| 유지 항목 | 주 표의 소비처 |
|---|---|
| 채택 계보·기록 재생·F0 | 같은 소비 상태에서의 개입이라는 근거 |
| 서로 누적하지 않는 처방 2개 | 영역 밖 붕괴와 영역 안 탈출확률의 구분 |
| 선택 준위 채널 수지 | III/II 이동에 수반되는 공급·배출 변화 |
| 공개 대응·입력 신원·수치 품질·비용 | 비교값의 의미와 유효 범위 |

처방의 조합·연속 매개변수 탐색·다른 셸·다른 원소는 제외한다. 저장 산출 재검 장치, 쓰기 결함 모사, 결정표 재계산 장치를 만들지 않는다. 직접시험은 새 코드의 음성 대조로 한정한다. 종료감사의 1차 대상은 **과학 표의 수치·대응·주장 한정**이다.

## §1. 권한·기준선·경로

사용자 선택은 「BB 복사 반사실 오프라인」이다. 기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사는 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra가 맡으며 계약 기안자 이해관계를 밝힌다.

| 기호 | 대상 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| E | `/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/results/run1` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| C | `/gpfs/kjhan/cmfgen_src/cur_cmf` |
| 공개 자료 | `data/standart_data1/toy06` |

직전 폐합 기준은 `VERDICT_DET_SI_EXCITATION_BUDGET_2026-10-11.md`의 9b45818 판본이다. 분기 2의 사유인 C1 미완결을 이번 단에서 해결하거나 재판정하지 않는다.

과학 입력은 T의 기록 문장·채택 계보·준위 대응이다. E는 F0 기준값과 기존 수지 정의의 출처이며 새 개입의 원율을 대신하지 않는다. 실제 소비 파일·크기·SHA-256·코드 판본·명령·작업 번호를 `evidence.json`에 한 번 기록한다. 전체 트리는 해시하지 않는다.

## §2. 과학 질문·고정 조건

> 물질13·Si 셸10의 채택 소비 상태에서, 영역 밖 BB의 자발붕괴를 제거하거나 영역 안 BB를 탈출확률 1의 처방으로 바꾸면, Si III/II와 선택 Si II 준위의 공급·배출은 각각 얼마나 움직이는가?

채택 키는 묶음 1·수용 n_e 반복 18·CE 패스 1·호출 36이다. 기존 채택 재유도 함수로 계보를 확인하며 마지막 행을 임의 선택하지 않는다.

연속 J장·T_e·n_e·CE 상대 인구·Si II+III 총량을 고정한다. BB 충돌과 BF·DR·CE·NT 등 비개입 율은 그대로 둔다. 직전 단의 C1·C2를 추가하지 않는다.

개입은 **기록된 Si II와 Si III 양쪽의 site 1 문장 전체 중 등록 모드에 해당하는 문장**에 적용한다. 선택 준위와 연결된 선만 골라 바꾸지 않는다. 기록 연산자에 없는 전이를 새로 만들지 않는다.

- **채택 SE:** 호출 36의 `FINAL_FL`; 생산 소비 인구의 참고 수지다.
- **F0:** 원래 기록 FL 연산자의 정상해다.
- **F7:** §4의 영역 밖 복사 제거만 적용한 정상해다.
- **F8:** §4의 영역 안 탈출확률 1만 적용한 정상해다.

F7과 F8은 각각 F0에서 출발한다. 광학깊이·선원천함수·복사장·전자밀도를 새 인구로 갱신하지 않는다. 채택 SE를 FL 정상해 또는 CE 감쇠 뒤 commit 인구와 혼동하지 않는다.

## §3. 먼저 정하는 잣대 — 소스 대응과 율식

**두 반사실 모두 진단 처방이다. CMFGEN의 공개 계산을 재현하는 처방으로 등록하지 않는다.**

| 대상 | 확인한 소스·루틴 | 대응의 범위 |
|---|---|---|
| 일반 선 순복사 | `momjbar.f`의 `MOMJBAR` | \(ZNET=1-\bar J\,CHIL/ETAL\); 전달로 구한 선 평균장과 순복사 브래킷의 관계 |
| Sobolev 선 처리 | `sobjbar_sim.f`의 `SOBJBAR_SIM` | 각도별 `EXPONX`와 입사장을 적분하고 동시 선의 기여를 반영 |
| 탈출확률 | `exponx.f`의 `EXPONX` | \((1-e^{-\tau})/\tau\)와 작은 \(\tau\)·큰 \(\tau\)의 소스 분기 |
| Lumina 모드 8 | `line_net_rate.c`의 `line_net_bracket_rates` | 기록 continuum 항에 B를 곱하고 자발율에 β를 곱함 |
| Lumina 모드 7 | `lumina_plasma.c`의 BB 생산 분기 | 영역 밖에서 \(\bar J=0\), 흡수·유도율 0, 자발율 A |

`momjbar.f` 약 296행, `sobjbar_sim.f` 약 160–192행, `exponx.f` 약 15–20행을 확인했다. `SOBJBAR_SIM`에는 각도 적분과 동시 선 처리가 있으므로 단일 기록 β의 변경을 CMFGEN 전체 선 처리의 이식이라고 부르지 않는다. 공개 toy06의 소스 판본·스위치 일치도 입증하지 않는다.

봉인 `lumina_plasma.c` 약 19204행의 생산 분기는 옛 진단용 분기 뒤에서 율을 다시 정한다. 옛 분기의 주석이나 clamp를 이번 처방으로 가져오지 않는다.

site 1의 x1–x14는 다음 순서다.

\[
(r_a,r_s,r_p,C_\uparrow,C_\downarrow,f_l,f_u,
\bar J,m,\beta,c,\ell,\tau_{\rm eff},p).
\]

\(c\)는 continuum 기여, \(\ell\)은 local-emission 기여이며 \(c+\ell=\bar J\)다. **c 자체를 감쇠 전 연속장으로 부르지 않는다.**

모드 8의 봉인 관계는 다음과 같다.

\[
c=\beta J_c,\qquad
r_a=B_{lu}c,\quad r_s=B_{ul}c,\quad r_p=A_{ul}\beta .
\]

\[
\beta=
\begin{cases}
1-\tau[1/2-\tau(1-\tau/4)/6],&|\tau|<10^{-3},\\
(1-e^{-\tau})/\tau,&|\tau|\ge10^{-3},\ \tau<40,\\
1/\tau,&\tau\ge40 .
\end{cases}
\]

`line_net_sobolev_radiation`은 \(c=\beta J_c\)와 국소 방출항을 별도로 구성한다. `line_net_bracket_rates`는 \(p\le3\), `srce_chk` 비트 부재, 기록 \(\bar J=c+\ell\), 유한한 양의 β를 요구한다.

통상 Einstein 관계는 \(g_lB_{lu}=g_uB_{ul}\), \(A_{ul}/B_{ul}=2h\nu^3/c_{\rm light}^2\)다. 다만 봉인 `lumina_atomic.c` 약 1340행은 A·B·ν를 CSV에서 각각 읽는다. 따라서 이번 계산은 이상 관계로 기록 계수를 덮어쓰지 않고 **소비된 율을 대수적으로 변환**한다.

모드 8에서 \(A_{ul}=r_p/\beta\), \(B_{lu}J_c=r_a/\beta\), \(B_{ul}J_c=r_s/\beta\)를 얻는다. \(c=0\)이어도 β가 양수이면 이 변환은 정의된다. B를 구하려고 0인 c로 나누지 않는다.

BB 영역은 `line_jbar.h`의 \(1.49896229\times10^{14}\)–\(2.99792458\times10^{16}\) Hz다. 연속 J 격자의 범위와 다르다. **모드 7은 연속 J가 물리적으로 0이라는 증거가 아니다.** 이번 단은 영역 밖 J를 보간·외삽하거나 영역을 확장하지 않는다.

## §4. 등록 처방 — 두 개만, 독립 적용

### D7 — 영역 밖 BB 복사 제거 → F7

모드 7의 모든 문장에서 다음으로 바꾼다.

\[
(r_a',r_s',r_p')=(0,0,0).
\]

모드 8과 모든 충돌율은 원래 값이다. 원래 모드 7의 \(r_a=r_s=0\), \(\beta=1\)을 확인한 뒤 자발율만 제거한다.

이것은 영역 밖 문장에 남아 있는 **일방향 자발붕괴의 민감도**를 재는 진단이다. 복사 결합 0은 해당 문장 율의 하한이며, 고정된 유한 입사장에 대한 완전 포획 극한의 유효 순복사율과 대응한다. 실제 영역 밖 선이 완전 포획된다는 가정의 채택은 아니다.

CMFGEN이 그 선을 누락하거나 자발방출을 끈다는 근거는 없다. F7을 CMFGEN 대응 처방 또는 III/II의 상한·하한으로 부르지 않는다.

### D8 — 영역 안 BB 탈출확률 1 → F8

모드 8의 모든 문장에서 다음으로 바꾼다.

\[
(r_a',r_s',r_p')=(r_a/\beta,\ r_s/\beta,\ r_p/\beta)
=(B_{lu}J_c,\ B_{ul}J_c,\ A_{ul}).
\]

모드 7은 원래 값이다. 기록 \(\tau_{\rm eff}\)와 β는 원입력으로 보존하고, 개입 β=1을 별도 항목으로 적는다.

이것은 **기록된 입사 연속장을 유지한 광학적으로 얇은 복사 결합 극한**이다. 자발율만 A로 바꾸거나, \(\bar J=c+\ell\)를 외부 펌프로 다시 넣지 않는다.

비음 광학깊이에서는 \(0<\beta\le1\)이므로 개입 전보다 세 복사율이 작아지지 않는다. 이는 해당 조건에서의 율 상한이지 인구·III/II의 상한이 아니다. 음의 광학깊이에서는 β가 1보다 클 수 있으므로 해당 문장은 얇은 기준 처방일 뿐 상한이라고 부르지 않는다. 부호별 문장 수를 주 표에 남긴다.

`EXPONX`의 얇은 극한과 수학적으로 대응하지만, CMFGEN이 모든 대상 선에 β=1을 쓰거나 공개 인구를 그 처방으로 얻었다는 뜻은 아니다.

## §5. 계산·기존 함수 재사용

`det_si_trace_readout.py`의 `read_trace`, `read_matrix`, `replay`, `det_si_exc_build_fl`, `gth`, `backward`, `ratio`를 재사용한다. `det_si_excitation_budget.py`의 `det_si_exc_adopt`, `det_si_exc_level_map`, `det_si_exc_solve`, `det_si_exc_budget`, `det_si_exc_public`을 가져다 쓴다.

직전 분석 명령 전체를 호출하거나 C1·C2 입력을 다시 처리하지 않는다.

`det_si_exc_build_fl`의 인자 끝에 선택적 `bb_radiative_override=None`을 추가한다. 콜백은 site 1의 새 복사율 세 값 또는 무변경을 반환한다. 기존 기록의 `v1/v2`와 원율 분해 검사는 **대체 전에** 수행한다. 대체율은 총 연산자·BB 연산자·복사 채널에 함께 반영하고 충돌율·기록 원본은 건드리지 않는다.

모드별 기준 수지는 기존 함수에 해당 site 1 문장 부분집합을 전달하여 얻는다. 정상해는 추가로 풀지 않는다. F0·F7·F8의 전체 정상해와 같은 채널 수지 함수에 소비한다.

정상해 유효 조건은 유한성·비음성·총량 보존·닫힌 통신류 1개·최대 성분별 후방오차 \(10^{-8}\) 이하다. 총량 상대오차와 열 보존 허용치는 \(10^{-12}\)다. 기존 함수를 사용하되 비유한 성분이 남은 해를 최대값 집계에서 제외하여 통과시키지 않는다.

모드 8의 β 재구성과 기록 β는 상대오차 \(10^{-12}\) 이하, \(\bar J=c+\ell\)은 봉인 함수와 같은 상대 기준으로 확인한다. 둘 다 0인 경우는 정확한 0으로 처리한다. 음수율·미정의 값·영 분모를 clamp/floor로 치환하지 않는다.

## §6. 주 산출 — CMFGEN 대비 표 하나

정본은 `si10_bb_counterfactual_table.csv`, 표시본은 같은 메모리 결과의 `si10_bb_counterfactual_table.md`다. 표시용 소표 분할은 허용하되 독립 계산 경로를 만들지 않는다.

| 행 종류 | 필수 내용 |
|---|---|
| 공개·채택 SE·F0·F7·F8 | III/II, 공개 대비 dex, F0 대비 이동, 공개 거리 감소, 유효 여부 |
| 처방 대응 | 진단 분류, 모드·이온별 대상 수, τ 부호별 수, CMFGEN 대응 범위 |
| 선택 준위×상태×채널 | \(f_i,b_i^{rel},I,O,B,D\), 유입·유출 몫과 F0 대비 변화 |
| 선택 준위×상태 합계 | 총 유입·유출·잔차, 정상해 여부 |
| 품질·범위 | 채택 키·고정량·후방오차·게이트·입력 신원 참조 |

선택 Si II `level_num`은 직전 단의 **0·1·6·7·9·10·42·44·52·124**를 유지한다. 채널은 BB 복사 상향·하향을 모드 7·8로 구분하고, 기존 충돌·BF·DR·CE·NT 분류를 유지한다.

\[
f_i=n_i/N_{\rm II},\quad
f_i^{LTE}=\frac{g_i e^{-E_i/kT_e}}{\sum_{j\in{\rm Si\,II}}g_j e^{-E_j/kT_e}},
\quad b_i^{rel}=f_i/f_i^{LTE}.
\]

\[
I_{ic}=\sum_{j\ne i}R^c_{j\to i}n_j,\quad
D_{ic}=\sum_{j\ne i}R^c_{i\to j},\quad
O_{ic}=D_{ic}n_i,\quad B_{ic}=I_{ic}-O_{ic}.
\]

I·O·B의 단위는 \(\mathrm{cm^{-3}\,s^{-1}}\), D는 \(\mathrm{s^{-1}}\)다. 변화는 각각의 정상해 인구로 평가한 값의 차이다. 비개입 채널도 인구 변화 때문에 I·O가 움직일 수 있다. 이를 해당 채널 율을 바꾼 효과로 서술하지 않는다.

\[
R=N_{\rm III}/N_{\rm II},\quad d=\log_{10}(R/R_{\rm pub}),\quad
\Delta_0=\log_{10}(R/R_{F0}),\quad G=|d_{F0}|-|d|.
\]

공개 대응은 19.480 d, Lumina 11,544 km/s ↔ 공개 11,534.59 km/s다. 계산에는 기존 판독 함수가 반환하는 원자료 비를 쓰며 표시값 105.70을 넣지 않는다.

공개 자료에는 이번 비교에 필요한 준위별 인구·채널 수지가 없다. 준위 행의 공개 비교 칸은 미확보로 표시한다. 0·미확보·해당 없음·미정의를 구분한다.

## §7. 기대치·과학 분기

F0는 직전 \(R=3.5597568125822305\)와 0.001 dex 이내로 일치해야 한다. 벗어나면 개입 효과로 해석하지 않고 채택·재생·FL 구성 문제로 처리한다.

기록 모드 분포의 대조값은 모드 7: Si II 671·Si III 221, 모드 8: Si II 1169·Si III 1418이다. 실제 채택 문장에서 다시 집계하며 불일치를 무시하거나 대상 수를 조정하지 않는다.

D7은 원래 인구에서 대상 문장의 자발 유출과 그 도착 준위 유입을 제거한다. D8은 비음 τ의 문장에서 세 복사율을 함께 증가시킨다. 그러나 두 처방 모두 네트워크 인구 재분배와 이온 간 결합을 거치므로 **선택 준위 인구와 III/II 이동의 부호·크기는 예측하지 못한다.**

BB 수지가 크다는 사실을 큰 III/II 이동의 예측으로 쓰지 않는다. 공개값 접근·LTE 분율 회복도 기대값으로 강제하지 않는다.

개별 처방은 \(G>0.01\) dex이면 접근, \(G<-0.01\)이면 이격, 나머지는 거리 변화 미분리다. 0.01 dex는 보고 분해능이며 물리 수용오차가 아니다.

| 코드 | 분기 | 조건 |
|---:|---|---|
| 0 | `TOWARD` | 표와 두 처방이 유효하고 하나 이상 접근 |
| 1 | `NO_RESOLVED_APPROACH` | 표와 두 처방이 유효하나 접근 처방 없음 |
| 2 | `INDETERMINATE` | 필수 입력·처방·해·수지가 미정의이거나 수치 문제로 분류 불가 |

작은 유효 효과는 분기 1에 포함한다. 작은 후방오차만으로 전방 정확도를 보증하지 않으며, 확인된 수치 불확실성이 분류 경계를 가로지르면 해당 분류를 유보한다. 일부 처방만 유효하면 그 행은 보존하되 완전 측정으로 보고하지 않는다.

## §8. 최소 변경집합·보존

### 변경집합 — DET-SI-BB-COUNTERFACTUAL (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_si_trace_readout.py` | `det_si_exc_build_fl`의 선택적 BB 복사율 대체 훅만 추가; 기본 호출 의미 유지 |
| `scripts/det_si_bb_counterfactual.py` | 등록 처방·입력 대응·기존 함수 연결·주 표·음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/tools/run_det_si_bb_counterfactual_offline.slurm` | 제한 CPU 실행·신원·비용 기록 |

### 변경집합 끝

`det_si_excitation_budget.py`는 가져다 쓰며 수정하지 않는다. 생산 소스·바이너리·입력·CMFGEN 소스 변경은 0이다. 범용 검증 프레임워크와 독립 시험 파일을 추가하지 않는다.

출력은 B의 새 실행 하위 경로에만 쓴다. 기존 출력은 덮어쓰지 않고 실패 산출도 보존한다. 일반 파일 읽기·쓰기만 사용하며 하드링크와 파일시스템 특수 기능을 사용하지 않는다.

보조 산출은 `evidence.json`, 실행·기계 게이트 로그, 음성 대조 결과로 제한한다. 처방별 대상 식별자와 미정의 사유는 주 표의 집계·유효성을 설명하는 증거로만 남긴다.

## §9. 직접시험·게이트

새 코드의 직접시험은 다음 음성 대조로 한정한다.

- 무개입 콜백에서 기존 연산자·채널이 변하는 결함을 검출한다.
- 모드 오지정·중복 적용·대상 누락·Si 이온 오대응을 주입하여 거부한다.
- 모드 7에 비영 흡수율, 모드 8에 0·비유한 β 또는 깨진 ledger를 주입하여 거부한다.
- D8에서 자발율만 나누거나 β를 두 번 나누는 결함을 처방식 대조로 검출한다.
- 충돌율 변경·채널 누락·차변 부호 오류를 주입하여 비개입 불변·보존 검사로 거부한다.

| 게이트 | 막는 실재 실패 |
|---|---|
| G-INPUT | 다른 소비 호출·이온·모드의 문장을 결합해 같은 상태의 결과로 보고하는 실패 |
| G-RATE | β 중복 적용·일방향 변경·충돌 오변경·비보존 행렬을 등록 처방으로 보고하는 실패 |
| G-SOLVE | 비유일·비유한·음수·큰 잔차의 해를 유효 III/II로 보고하는 실패 |
| G-TABLE | SE/F0 혼동·공개 오대응·율 경계를 인구 경계 또는 CMFGEN 재현으로 확대하는 실패 |
| G-RUN | GPU 사용·생산 변경·계산 미완료·비용 초과를 등록 실행으로 보고하는 실패 |

원자료 재생과 같은 메모리의 채널 합 대조는 주 계산 안에서 수행한다. 저장 표를 다시 읽어 검증하지 않는다. 기존 사전등록·참조·정본·발주 기계 검사는 기존 도구로 집행하고 별도 게이트 산출 체계를 만들지 않는다.

## §10. 비용·실행·재현

| 항목 | 상한 |
|---|---|
| 생산 실행·GPU | 모두 0 |
| 오프라인 | a10 CPU Slurm, 작업명 `LUMINA_` 접두, `--exclude=syn102,syn03` |
| 작업당 | 1 task·2 CPU·8 GiB·벽시계 300 s |
| 누적 | 할당 벽시계 900 s·0.5 CPU core·h |
| 수치 스레드 | BLAS·OpenMP 각 1 |
| 주 계산 | F0·F7·F8 각 1회; 수정 재실행은 누적 상한에 포함 |
| 신규 저장 | 100 MiB 이하 |

직접시험·기계 검사·분석·감사 재계산을 누적 비용에 포함한다. 로그인 노드에서는 파일 조회·편집·제출·결과 열람만 한다.

러너는 `module purge` 뒤 명시한 모듈을 로드하고 Python·NumPy 판본을 기록한다. 필수 요청은 `--partition=a10 --ntasks=1 --cpus-per-task=2 --mem=8G --gres=gpu:0 --time=00:05:00 --exclude=syn102,syn03`이며 실제 GPU 할당 0을 확인한다.

작업 내부 인터페이스는 다음과 같다.

```bash
python3 -B scripts/det_si_bb_counterfactual.py negative
python3 -B scripts/det_si_bb_counterfactual.py analyze \
  --run /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1 \
  --out /gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/results/run1
```

재현 시 같은 입력·코드 판본·명령을 사용하고 출력 경로만 새 실행 번호로 바꾼다. 반복 실행을 의무화하지 않는다. 비용 증액·처방 추가·영역 확장에는 계약 개정이 필요하다.

## §11. 완료·판정·주장 상한

완료 기록은 주 표, 입력·율식·게이트·비용 증거, 운전자 판정, astra 종료감사로 구성한다. 목표 달성은 두 처방의 유효한 측정이며 공개 접근 여부와 구분한다. 필수 측정이 빠진 상태를 기록상 폐합할 때는 미달 목표와 분기 2 사유를 명시한다.

종료감사는 과학 표의 수치 정의·소스 대응·모드 귀속·공개 비교·주장 한정을 우선 검토한다. 직전 단 전체 재감사와 코드 전면 감사를 반복하지 않는다. 직접 재계산한 항목과 저장 증거로 확인한 항목을 구분하며, 기안자 이해관계와 운전자의 구현·판정 겸임을 기재한다.

허용 결론은 **같은 소비 상태에서 두 BB 진단 처방이 만드는 조건부 인구·수지 변화**다. 큰 효과는 후속 조사 우선순위의 근거이고, 작은 효과는 이 두 처방의 효과가 작다는 근거다. 후자를 모든 BB 변경의 무효로 일반화하지 않는다.

복사 전달·되먹임·열/전하 폐합·시간 의존·자기일관 효과·스펙트럼을 계산하지 않는다. 생산 승격·물리적 정답·스펙트럼 개선·철족 일반화를 선취하지 않는다.

집행 기록은 운전자가 이 절에 추가하며 기안 문면·과거 판정·작성자 표기를 보존한다.

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011`. 작업 번호·경과는 `B/JOBS.txt`·`B/evidence/sacct_offline_jobs.txt`, 게이트 집행은 `B/evidence/GATE_RECORD.json`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| user 결정 | user | 「BB 복사 반사실 오프라인」(`B/evidence/USER_DECISION_2026-10-11.txt`) |
| 기안 | astra `gpt-6-astra`(자기보고 「GPT-6 계열, 세부 ID 확인 불가」) | 2f6fc92(개정 없음) |
| 기계 게이트(기안) | 운전자 | 기안 브리프 canonical(a10 419343), 기안 prereg·referents·canonical rc 0(419346) |
| 경량 사전감사 | 생략(25-ab: GPU 0·생산 변경 0) | — |
| 구현 | 운전자(Opus 5.5) · 코드 감사 없음(25-z) | ba32203 — `det_si_exc_build_fl` 의 선택적 `bb_radiative_override` 훅, 새 판독기, 러너 `B/tools/run_det_si_bb_counterfactual_offline.slurm` |
| 직접시험(§9) | 운전자 | a10 419351 **21/21**(양성 대조 5·무개입 불변 1·결함 주입 거부 15), 러너 위생 rc 0 |
| 등록 계산 run1 | 운전자 | a10 419352(3 s) `B/results/run1/`: G-INPUT·G-RATE·G-SOLVE·G-TABLE·G-RUN 지적 0; F0 = 직전 값(0 dex); census 등록값 일치; **분기 1 NO_RESOLVED_APPROACH** — F7 −0.0310 dex·F8 −0.8365 dex, 둘 다 이격 |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md` — 종료감사 대기 |
| 종료감사 | astra | 대기 |
| 비용 | 운전자 | GPU 0; 오프라인 a10 4 작업 할당 34 s · 0.010 core·h / 900 s · 0.5 core·h(기안 게이트 2 작업 31 s 포함); 신규 저장 219 KiB / 100 MiB |

**이탈(기재):**
1. §10 의 필수 요청은 그대로 썼다. 표시본 `.md` 는 같은 메모리 행을 행 종류별 소표로 나눴다(§6 허용).
2. 선택 준위×상태×채널 행의 「F0 대비 변화」는 I·O·D 의 차와 유입·유출 몫의 차(dI·dO·dD·d_in_share·d_out_share)로 적었고, 합계 행에 b_rel 의 F0 대비 비를 더했다.

## §12. 기계 선언

참조에는 실재 파일만 지정한다. 아래 분기 입력은 기존 사전등록 검사 형식이며 새 결정표 재계산 장치를 뜻하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SI-BB-COUNTERFACTUAL (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_si_(?:trace_readout|bb_counterfactual)\\.py|/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/tools/run_det_si_bb_counterfactual_offline\\.slurm)",
    "symbol": "det_si",
    "roots": [
      "scripts/det_si_trace_readout.py",
      "scripts/det_si_bb_counterfactual.py",
      "/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/tools/run_det_si_bb_counterfactual_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_si_trace_readout.py",
      "scripts/det_si_bb_counterfactual.py",
      "/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/tools/run_det_si_bb_counterfactual_offline.slurm"
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
      {"name":"TOWARD","predicate":"valid and c0 == 1"},
      {"name":"NO_RESOLVED_APPROACH","predicate":"valid and c1 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"toward","mix":[[0,1]]},
      {"name":"no_resolved_approach","mix":[[1,1]]},
      {"name":"indeterminate","mix":[[2,1]]},
      {"name":"invalid","mix":[[9,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_SI_EXCITATION_BUDGET_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_EXCITATION_BUDGET_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_CONSUMPTION_TRACE_2026-10-11.md"},
    {"path":"scripts/det_si_trace_readout.py","flags_existing":["replay","det_si_exc_build_fl","gth","backward"]},
    {"path":"scripts/det_si_excitation_budget.py","flags_existing":["det_si_exc_adopt","det_si_exc_solve","det_si_exc_budget","det_si_exc_public"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/bbcf_20261011/evidence/USER_DECISION_2026-10-11.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_ops.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_calls.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_commit.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_slmap.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_vec.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/results/run1/evidence.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c","flags_existing":["BB_EXCLUDED_OUTSIDE_DOMAIN","line_net_bracket_rates"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c","flags_existing":["line_B_lu","line_B_ul","line_A_ul"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c","flags_existing":["line_net_bracket_rates","line_net_cmfgen_exponx","line_net_sobolev_radiation"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_jbar.h","flags_existing":["LINE_JBAR_BB_NU_MIN_HZ","LINE_JBAR_BB_NU_MAX_HZ"]},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/subs/momjbar.f","flags_existing":["MOMJBAR","ZNET"]},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sobjbar_sim.f","flags_existing":["SOBJBAR_SIM","EXPONX"]},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/subs/exponx.f","flags_existing":["EXPONX"]},
    {"path":"data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt","flags_existing":["si1","si2","19.480"]}
  ]
}
```
