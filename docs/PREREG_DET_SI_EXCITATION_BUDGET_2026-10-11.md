# 사전등록 — DET-SI-EXCITATION-BUDGET: Si II 들뜬 준위 인구 수지와 CMFGEN 발자국 채널의 고정 장 효과

2026-10-11 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_SI_EXCITATION_BUDGET_2026-10-11.md`  
상태: **기안. 구현·직접시험·기계 게이트·계산·판정·종료감사 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 직전 단은 Si 셸10의 단계 해와 SE 해의 차이를 광전리 적용 인구에 연결했지만, 들뜬 준위 인구가 낮아지는 유입·유출 구조는 가르지 않았다.

이번 단의 관측 가능한 끝점은 **선택 준위의 인구 수지와 두 개별 채널 개입의 III/II 이동을 공개 CMFGEN 대비 표 하나로 제시하는 것**이다. 이를 통해 다음 생산 검토 대상으로 충돌 전리·3체 재결합 활성화 또는 표형 충돌강도 보간 변경을 검토할 수 있는지 판단한다. 효과가 작거나 공개에서 멀어져도 유효한 결과다. 이 단을 생략하면 LTE 인구를 강제하는 처방과 실제 율 수정의 근거를 구분하기 어렵다.

**Q-LEAN — 불필요한 측정·게이트를 더하는가? 자기답: APPROPRIATE.**

| 유지 항목 | 주 표의 소비처 |
|---|---|
| 채택 호출·준위 대응·기록 연산 재생 | 같은 소비 상태의 수지라는 근거 |
| 선택 준위의 채널별 유입·유출 | 낮은 인구를 유지하는 공급·배출 경로 |
| 기준 FL 정상해 | 채택 인구의 FL 불균형과 개입 효과의 분리 |
| 개별 개입 2개 | 각 채널 처방의 조건부 III/II 이동 |
| 공개 시각·속도·이온 분율 | 공개 대비 dex |
| 입력·식·실행 신원과 비용 | 표의 수치 출처와 범위 |

저장 산출 재검 장치, 쓰기 결함 모사, 결정표 재계산 장치, 전체 원소 조사, 조합 개입, 매개변수 탐색, 새 생산 계측은 만들지 않는다. 직접시험은 새 코드의 음성 대조로 한정한다. 종료감사의 1차 대상은 **과학 표의 수치·대응·주장 한정**이다.

## §1. 권한·기준선·경로

사용자 선택은 「들뜬 준위 인구 기전 오프라인」이다. 기안·개정·종료감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. 생산 변경 0·GPU 0이므로 경량 사전감사는 생략한다(25-ab). 코드 감사는 없다(25-z). astra는 종료감사에서 계약 기안자라는 이해관계를 명시한다.

| 기호 | 대상 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| A | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/analyze/analysis` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| C | `/gpfs/kjhan/cmfgen_src/cur_cmf` |
| 공개 자료 | `data/standart_data1/toy06` |

직전 폐합은 `VERDICT_DET_SI_CONSUMPTION_TRACE_2026-10-11.md`의 f4ef198 판본이다. 기존 판정·작성자·산출은 보존하며 재감사하지 않는다.

과학 입력은 T의 기록 원자료와 봉인 원자 자료다. A와 운전자 탐침은 기존 결과의 대응·선택 근거이며 새 계산의 원율을 대신하지 않는다. 실제 사용 파일·크기·SHA-256·소비 키·코드 판본·명령을 `evidence.json`에 한 번 기록한다. 전체 저장소나 데이터 트리를 해시하지 않는다.

## §2. 과학 질문·고정 조건

> 물질13·Si 셸10의 채택 소비 조건에서, 높은 준위와 중간 준위의 낮은 이온 내 SE/LTE 분율비는 어떤 유입·유출 수지로 유지되며, Lumina와 CMFGEN 사이에서 확인한 두 충돌 채널 처방을 각각 적용하면 III/II가 공개값 쪽으로 얼마나 움직이는가?

채택 키는 묶음 1·수용 n_e 반복 18·CE 패스 1·호출 36이다. `si10_calls.csv`와 commit 계보로 대응하며, 단순히 마지막 파일 행을 선택하지 않는다.

J·T_e·n_e·CE 상대 인구와 Si II+III 총량을 고정한다. 기록된 BB의 Jbar·β·유효 복사율, BF·DR·CE·NT 계수도 각 개입이 명시적으로 바꾸는 항 이외에는 고정한다. Si 자체 인구가 변해도 광학깊이·복사장·전자밀도·상대 원소를 다시 계산하지 않는다.

기준 상태는 구분한다.

- **채택 SE:** 호출 36의 `FINAL_FL`. 생산 소비 상태의 수지 진단 대상이다.
- **F0:** 기록 FL 연산자의 정상해. 개입 효과의 직접 대조군이다.
- **F1·F2:** 각각 한 개입만 적용한 FL 정상해다. 누적 적용하지 않는다.

직전 참고값은 채택 SE III/II≈3.563699, F0≈3.559757이다. 개별 FL 잔차가 큰 채택 SE를 FL 정상해로 부르지 않는다.

## §3. CMFGEN 발자국·율식 선정

「CMFGEN 발자국」은 **확인한 CMFGEN 소스에 해당 과정 또는 처방이 존재한다는 뜻**이다. 공개 toy06 계산이 같은 소스 판본·원자 자료·스위치를 사용했다는 증명은 아니다.

| 개입 | 소스 근거 | 이번 단의 선택 |
|---|---|---|
| C1: 충돌 전리·3체 재결합 | `SUBCOL_MULTI_V6`의 준위별 충돌 전리, `STEQ_MULTI_V4`의 LTE 역과정과 상위 이온 결합, `OMEGA_GEN_V3`의 문턱 단면적 기반 근사 | 봉인 Lumina 기존 가지의 율식을 사용하여 꺼진 가역 채널을 추가 |
| C2: 표형 충돌 여기·탈여기 | `OMEGA_GEN_V3`의 표형 Ω 로그 보간, `SUBCOL_MULTI_V6`의 상·하향 충돌율 | 기록된 Si II `COLDATA` 항의 Ω 보간만 CMFGEN 식으로 교체 |

`subcol_multi_v6.f` 약 132–157행에 직접 충돌 전리식이 있고, `steq_multi_v4.f` 약 33행은 상위 이온 바닥으로의 전리를 명시한다. 같은 파일 약 160행 이후에 LTE 인구를 통한 역과정이 들어 있다.

`omega_gen_v3.f` 약 115–149행은 표형 Ω의 로그 보간, 약 203–240행은 직접 전리의 근사 Ω를 정의한다. 봉인 `lumina_plasma.c` 약 19720–19760행은 일반 `col_data`의 선형 보간이고, 약 20033–20070행은 C1 가지다. `nlte_bf_collisional_enabled`는 같은 파일 약 463–472행에 있으며 O에서는 꺼져 있다.

### C1 — 기존 충돌 전리·3체 재결합 쌍

모든 Si II 준위 \(i\)와 Si III 바닥 \(g\)에 다음 율을 적용한다.

\[
u_i=\frac{h\nu_{{\rm th},i}}{kT_e},\qquad
C_{i\to g}=n_e\frac{1.55\times10^{13}}{\sqrt{T_e}}\,
0.2\,\sigma_{{\rm edge},i}\frac{e^{-u_i}}{u_i},
\]
\[
S_i=n_e\left(\frac{h^2}{2\pi m_ekT_e}\right)^{3/2}
\frac{g_i}{2g_g}e^{u_i},\qquad C_{g\to i}=C_{i\to g}S_i .
\]


율의 단위는 양방향 모두 입자당 \(\mathrm{s^{-1}}\)이다. 전리와 역과정은 **하나의 가역 채널**로 함께 추가한다. 한쪽만 켜는 개입은 하지 않는다.

\(\sigma_{\rm edge}\)는 T의 `cmfgen_sigma_bf.bin`에서 봉인 로더의 준위·주파수 대응과 pair의 첫 양의 bin 선택을 따른다. 광전리율을 J로 나누어 단면적을 추정하지 않는다. 기록 BF 항의 \(S_i\)와 문턱을 대조하여 입력 대응을 확인한다.

이 처방은 ARTIS 형식으로 구현된 Lumina 가지다. CMFGEN의 기본 근사와 과정·문턱 의존성이 대응하지만 CMFGEN 율의 정확한 재현이라고 부르지 않는다. 정밀 상수·문턱 단면적 평가·표형 직접 전리 자료의 차이는 남는다.

계산은 로그 표현 등으로 안정화할 수 있으나 clamp/floor를 넣지 않는다. 봉인 가지의 상한이나 지수 제한이 실제 대상에서 활성화될 조건이면 그 준위의 동등 처방은 미정의로 처리하고 C1 완전 계산을 주장하지 않는다.

### C2 — 표형 Ω 보간 처방 교체

기록에 존재하는 Si II `COLDATA` 전이에만 적용한다. 원자료는 T의 `ige_col_14_1_cmfgen.bin`이며, 단위·준위 번호는 봉인 로더와 대조한다.

\(T_a<T_e<T_b\), 양의 표값 \(\Omega_a,\Omega_b\)에 대해 다음을 사용한다.

\[
\Omega(T_e)=\Omega_a
\left(\frac{T_e}{T_a}\right)^{
\ln(\Omega_b/\Omega_a)/\ln(T_b/T_a)} .
\]

온도가 표의 절점과 정확히 같으면 그 절점 값을 사용한다. 표 범위 밖이거나 필요한 표값이 양수가 아니면 C2를 미정의로 둔다. 끝값 고정·외삽·양수 floor를 도입하지 않는다.

상·하향 율은 봉인 Lumina의 \(8.629\times10^{-6}\) 계수와 상세평형식을 유지하고 Ω만 교체한다. 따라서 CMFGEN의 \(8.63\times10^{-6}\)까지 바꾼 효과를 섞지 않는다.

기존 해당 충돌항을 제거하고 대체항을 넣으며, 복사 BB·비표형 충돌·다른 이온은 그대로 둔다. 새 충돌 전이나 SL 내부에서 생략된 전이는 추가하지 않는다. 두 보간값이 같으면 무효과도 정상 결과다.

준위 용해, 비표형 Ω 완성, 상위 이온 들뜬 상태로의 BF 결합은 이번 개입에서 제외한다. 기록 밖 입력·모형 선택을 늘리는 별도 문제이며, 미측정을 효과 0으로 해석하지 않는다.

## §4. 준위 선택·수지 정의

고정 선택은 Si II 준위 번호 **0·1·6·7·9·10·42·44·52·124**다. 0·1은 낮은 준위 대조, 6·7·9·10은 중간 준위, 42·44·52·124는 높은 준위 사례다. 결과를 보고 교체하지 않는다. 전체 높은 준위의 대표 범위라고 주장하지 않는다.

번호는 `level_num`을 뜻한다. `global_level`·`i_fl`·SL 번호·\(E_i\)·\(g_i\)를 함께 대응한다. 개입과 정상해는 선택 준위만이 아니라 기록된 전체 pair 차원에서 계산한다.

이온 내 분율과 상대 계수는 다음과 같다.

\[
f_i=n_i/N_{\rm II},\qquad
b_i^{\rm rel}=f_i/f_i^{\rm LTE},\qquad
f_i^{\rm LTE}=\frac{g_i e^{-E_i/kT_e}}
{\sum_{j\in{\rm Si\,II}}g_j e^{-E_j/kT_e}} .
\]

\(b_i^{\rm rel}\)은 절대 Saha 인구에 대한 출발계수가 아니다. LTE 분모는 같은 Si II 준위 집합·같은 T_e를 사용하며 사다리의 기록 분율과 대응한다.

열 생성자 \(A_{ij}=R_{j\to i}\)에 대해 채널 \(c\)별 수지는 다음과 같다.

\[
I_{ic}=\sum_{j\ne i}R^{c}_{j\to i}n_j,\quad
D_{ic}=\sum_{j\ne i}R^{c}_{i\to j},\quad
O_{ic}=D_{ic}n_i,\quad B_{ic}=I_{ic}-O_{ic}.
\]

\(I,O,B\)는 \(\mathrm{cm^{-3}\,s^{-1}}\), \(D\)는 \(\mathrm{s^{-1}}\)이다. 순수지만 제시해 큰 상쇄를 숨기지 않는다.

채널은 BB 복사 상향·하향, BB 충돌, BF 광전리·복사 재결합, DR, CE, NT, C1로 나눈다. 기록이 분리하지 않은 하위 항은 합산 명칭을 유지한다. 자발·유도 재결합을 기록보다 세밀하게 추정하지 않는다.

정상해에서 \(n_i=I_i/D_i\)와 다음 진단량을 사용한다.

\[
H_{ic}=I_{ic}-D_{ic}N_{\rm II}f_i^{\rm LTE}.
\]

\(H_{ic}\)는 **다른 준위와 상위 이온 인구를 해당 정상해에 고정하고 대상 준위만 LTE 분율값에 놓았을 때**의 공급−배출이다. 전체 원자를 LTE로 바꾼 실험이 아니다. 총합이 음수이면 그 고정 공급으로 LTE 수준의 대상 인구를 유지할 수 없음을 뜻한다.

채택 SE에는 실제 잔차를 병기하고 정상해 항등식을 강제하지 않는다. F0·F1·F2에서 유입·유출 몫과 \(H_{ic}\)를 비교하여 공급 부족·빠른 배출·상쇄를 서술한다. 채널 순위만으로 독립 인과 기여율을 부여하지 않는다.

## §5. 계산·판독기 재사용

`scripts/det_si_trace_readout.py`의 `read_trace`, `read_matrix`, `replay`, `gth`, `backward`, `ratio`, `public_si_ratio`를 가져다 쓴다. 기존 명령 전체를 호출해 직전 단의 분석 산출을 다시 쓰지 않는다.

현재 FL 구성은 `cmd_analyze` 내부에 있다. 약 622–674행의 로직을 `det_si_exc_build_fl`로 추출하고 기존 호출부도 이 함수를 사용하게 한다. 기록 순서·매핑·율·부호를 유지하며, 총 연산자와 채널별 항을 함께 반환한다. 새 판독기에 복사 구현하지 않는다.

채택 호출만 `replay`하여 기록 생성자와 대응한다. 채널별 FL 항의 합은 같은 메모리의 총 연산자와 대조한다. C1은 전이쌍을 추가하고, C2는 식별된 기존 항만 대체한다.

정상해는 GTH로 계산하고 같은 Si 총량으로 정규화한다. 유효 조건은 유한성·비음성·총량 보존·유일한 정상해 및 최대 성분별 후방오차 \(10^{-8}\) 이하이다. 영 분모는 별도 정의하며 작은 상수로 치환하지 않는다.

연산자의 닫힌 통신류로 정상해 유일성을 확인한다. 음수율·부호 오류·미정의 FL 항을 수정값으로 덮지 않는다. 작은 후방오차만으로 III/II 정확도를 보증하지 않으며, 수치 불확실성이 아래 0.01 dex 분류를 가로지르면 미분리로 둔다.

## §6. 주 산출 — CMFGEN 대비 표 하나

정본은 `si10_excitation_table.csv`, 표시본은 같은 메모리 결과를 표현한 `si10_excitation_table.md`다. 별도 재계산 경로를 두지 않는다.

| 행 종류 | 필수 수치·내용 |
|---|---|
| 공개 | 시각·속도·Si II·III 분율·III/II·준위 인구 미확보 |
| 채택 SE·F0·F1·F2 이온합 | III/II, 공개 대비 dex, 채택 SE 대비 이동, F0 대비 이동, 공개 거리 감소 |
| 선택 준위×상태×채널 | 준위 신원, \(f_i^{LTE},f_i,b_i^{rel},I,O,B,D,H\), 유입·유출 몫 |
| 선택 준위×상태 합계 | 총 유입·유출·잔차, 공급·배출 해석, 정상해 여부 |
| 개입 대응 | CMFGEN 루틴·선택 율식·대상 전이 수·적용 또는 미정의 사유 |
| 범위·수치 품질 | 소비 키·고정량·게이트 결과·후방오차·증거 식별자 |

해당 없는 칸과 미확보 칸을 0으로 쓰지 않는다.

\[
R=N_{\rm III}/N_{\rm II},\quad d=\log_{10}(R/R_{\rm pub}),
\]
\[
\Delta_{\rm SE}=\log_{10}(R/R_{\rm SE}),\quad
\Delta_0=\log_{10}(R/R_{F0}),\quad
G=|d_{F0}|-|d|.
\]

개입 귀속은 \(\Delta_0\)로 하고, 사용자가 요청한 기준 SE 대비 이동은 \(\Delta_{\rm SE}\)로 함께 보여 준다. \(G>0\)은 공개에 가까워짐이다.

공개 대응은 19.480 d, Lumina 11,544 km/s ↔ 공개 11,534.59 km/s다. 기존 공개 판독 함수를 사용하며 105.70은 표시값이다. 계산에는 공개 원자료의 비를 사용한다.

기안 때 확인한 toy06 자료의 `ionfrac_si_toy06_cmfgen.txt`는 이온 단계 분율, `phys_toy06_cmfgen.txt`는 온도·밀도·전자밀도·원자밀도를 담는다. 확인한 해당 묶음과 구형 형식 파일 목록에는 준위별 인구가 없다. **이번 공개 자료로는 Si II 준위별 CMFGEN 인구·채널 수지를 직접 대조할 수 없다.** 다른 미확인 공개 배포물까지 없다고 단정하지 않는다.

## §7. 기대치·과학 분기

F0는 직전 약 3.559757을 재현해야 한다. 차이가 0.001 dex를 넘으면 새 물리 효과로 해석하기 전에 호출·추출 대응 문제로 처리한다.

C1은 높은 준위에서 문턱이 낮아 직접 전리가 덜 억제되지만, 3체 유입도 함께 생긴다. 개별 준위 인구와 III/II 이동의 **부호·크기는 예측하지 못한다**. 다른 조건의 황 ≤0.01 dex 결과를 이번 Si의 상한으로 쓰지 않는다.

C2의 Ω 변화 부호는 온도와 표형 곡선에 달려 있다. 인구 재분배와 III/II 이동의 부호·크기를 예측하지 못하며, 공개값 접근을 기대값으로 강제하지 않는다.

각 유효 개입은 \(G>0.01\) dex이면 접근, \(G<-0.01\)이면 이격, 나머지는 거리 변화 미분리로 표시한다. 이는 수치 해석 분해능이며 물리 수용오차가 아니다.

단 전체 분기는 다음과 같다.

| 코드 | 분기 | 조건 |
|---:|---|---|
| 0 | `TOWARD` | 표·두 개입이 유효하고 하나 이상 공개 거리 감소가 0.01 dex 초과 |
| 1 | `NO_RESOLVED_APPROACH` | 표·두 개입이 유효하고 위 조건을 만족하는 개입이 없음 |
| 2 | `INDETERMINATE` | 필요한 수지·대응·개입 중 하나 이상 미정의 또는 수치 미분리로 결론 불가 |

한 개입의 접근이 다른 개입의 이격을 숨기지 않게 개별 분류를 남긴다. 유효한 작은 효과·반대 효과도 목표 달성이다. 필수 개입이 빠지면 완전 측정으로 폐합하지 않는다.

## §8. 최소 변경집합·보존

### 변경집합 — DET-SI-EXCITATION-BUDGET (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_si_trace_readout.py` | 기존 FL 구성의 함수 추출·채널 반환·기존 호출부 연결만 |
| `scripts/det_si_excitation_budget.py` | 입력 대응·수지·C1/C2·정상해·주 표·음성 대조 명령 |
| `/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/tools/run_det_si_excitation_offline.slurm` | 제한 CPU 실행·입력과 코드 신원·비용 기록 |

### 변경집합 끝

생산 소스·바이너리·입력·CMFGEN 소스 변경은 0이다. 새 범용 검증 프레임워크나 독립 시험 파일은 만들지 않는다.

출력은 B의 새 실행 하위 경로에만 쓴다. 기존 출력이 있으면 덮어쓰지 않는다. 일반 파일 읽기·쓰기와 필요한 소형 파일 복사만 사용하고 하드링크를 사용하지 않는다. 실패 결과는 보존한다.

보조 산출은 `evidence.json`, 실행 로그, 음성 대조 결과로 제한한다. 이들은 주 표의 입력·율식·미정의 사유·실행 범위를 설명한다.

## §9. 직접시험·게이트

새 코드 직접시험은 동일 구현 함수를 호출하는 소형 음성 대조로 한정한다.

- 잘못된 호출·준위 대응과 중복 C2 대상 주입을 거부한다.
- 음수율·비양수 Ω·표 범위 밖 온도·끊어진 정상해 계를 유효 결과로 통과시키지 않는다.
- C1 강도 0과 C2 원래 처방 선택에서 메모리 연산자 변화가 없어야 한다.
- 역율의 \(n_e\) 인자 누락 또는 전이 부호 오류를 주입하면 상세평형·보존 검사가 거부해야 한다.

파일 쓰기 실패 모사·저장 표 재독해·결정표 재계산 시험은 하지 않는다.

| 게이트 | 막는 실재 실패 |
|---|---|
| G-INPUT | 다른 소비 호출·준위·단면적·충돌표를 결합하여 같은 상태의 수지로 보고하는 실패 |
| G-RATE | 채널 중복·역과정 누락·보간 범위 위반·비보존 연산자를 물리 개입으로 보고하는 실패 |
| G-SOLVE | 비유일·비유한·음수·큰 잔차의 해를 유효 III/II로 보고하는 실패 |
| G-TABLE | 이온 내 LTE 분율과 절대 인구의 혼동, 공개 시각·속도 오대응, 채택 SE와 F0 혼동 |
| G-RUN | GPU 사용·생산 변경·계산 미완료·비용 초과를 등록 실행으로 보고하는 실패 |

기존 사전등록·참조·정본·발주 기계 검사는 기존 도구로 집행한다. 동일 입력·계산의 검사는 한 결과를 공유한다. 게이트별 별도 산출 체계는 만들지 않는다.

## §10. 비용·실행·재현 명령

| 항목 | 상한 |
|---|---|
| 생산 실행·GPU | 모두 0 |
| 오프라인 | a10, 작업명 `LUMINA_` 접두, `--exclude=syn102,syn03` |
| 작업당 | 1 task·2 CPU·8 GiB·벽시계 300 s |
| 누적 | 할당 벽시계 900 s·0.5 CPU core·h |
| 수치 스레드 | BLAS·OpenMP 각 1 |
| 주 계산 | F0·F1·F2 각 1회; 실패 수정 재실행도 누적 상한에 포함 |
| 신규 저장량 | 100 MiB 이하 |

모든 계산·직접시험·기계 검사는 Slurm에서 실행한다. 로그인 노드는 읽기·편집·제출·결과 열람만 한다. 러너는 `module purge` 후 등록한 모듈을 로드하고 실제 Python·NumPy 판본을 기록한다.

러너의 필수 요청은 `--partition=a10 --ntasks=1 --cpus-per-task=2 --mem=8G --gres=gpu:0 --time=00:05:00`이다. GPU 할당 0을 실행 기록에서 확인한다. 제출 조건이 수용되지 않으면 GPU 할당으로 대체하지 않는다.

작업 내부의 등록 인터페이스는 다음과 같다. 운전자는 구현 시 그대로 제공한다.

```bash
python scripts/det_si_excitation_budget.py negative
python scripts/det_si_excitation_budget.py analyze \
  --run /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1 \
  --out /gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/results/run1
```

재현은 같은 입력·코드 판본·명령을 사용하고 출력 경로만 새로운 실행 번호로 바꾼다. 재현 명령의 등록은 반복 실행 의무가 아니다. 상한을 늘리거나 추가 채널·조합을 계산하려면 계약 개정이 필요하다.

## §11. 완료·판정·주장 상한

완료 조건은 선택 준위 수지와 두 개입을 포함한 주 표, 필요한 게이트·비용 기록, 운전자 판정, astra 종료감사다. 종료감사는 수치의 정의·대응·채널 귀속·공개 비교·주장 한정을 우선 검토하며, 직전 단 전체나 코드 전면 감사를 반복하지 않는다.

운전자는 구현자와 판정자를 겸함을 밝힌다. astra는 계약 저자임을 밝히고 직접 재계산한 항목과 저장 증거로 확인한 항목을 구분한다. 감사 집행·반영·이탈 기록은 이 절에 이어 남긴다.

허용 결론은 **고정 소비 상태에서의 인구 수지 설명과 등록 개입의 조건부 효과**다. 공개와 가까워져도 생산 승격·물리적 정답·자기일관 효과·스펙트럼 개선·철족 일반화를 선취하지 않는다. J·T_e·n_e·CE 되먹임, 복사 전달, 열·전하 폐합, 시간 의존 효과는 측정하지 않는다.

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011`. 작업 번호·경과는 `B/JOBS.txt`·`B/evidence/sacct_offline_jobs.txt`, 게이트 집행은 `B/evidence/GATE_RECORD.json`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| user 결정 | user | 「들뜬 준위 인구 기전 오프라인」(`B/evidence/USER_DECISION_2026-10-11.txt`) |
| 기안·개정 1 | astra `gpt-6-astra`(자기보고 「GPT-6 계열, 세부 ID 확인 불가」) | bc60064 · 3406d99(OLD/NEW 2쌍, 비한국어 잔재 2곳 정리, 행 단위 −1 행·의도 외 소실 0) |
| 기계 게이트(기안·개정) | 운전자 | 기안 브리프 canonical(a10 419291), 기안 prereg·referents·canonical rc 0(419298), 개정 브리프(419299), 개정 후 rc 0(419300) |
| 경량 사전감사 | 생략(25-ab: GPU 0·생산 변경 0) | — |
| 구현 | 운전자(Opus 5.5) · 코드 감사 없음(25-z) | 16a05f7 — `det_si_exc_build_fl` 추출(누적 문장 불변, 채널·대체 훅·추가 항 반환), 새 판독기, 러너 `B/tools/run_det_si_excitation_offline.slurm` |
| 직접시험(§9) | 운전자 | a10 419325 실패(시험 장치 결함: 장난감 기록을 `repr(np.float64)` 로 써서 문자열 `np.float64(…)` 가 됨 — 게이트 결과 아님) → 고침 → 419326 **24/24**(양성 대조 7·무개입 불변 2·결함 주입 거부 15), 러너 위생 rc 0 |
| 등록 계산 run1 | 운전자 | a10 419327(3 s) `B/results/run1/`: G-INPUT·G-RATE·G-SOLVE·G-TABLE·G-RUN 지적 0; F0 = 저장 R_F(차 1.9×10⁻¹⁶ dex); C1 107/125 준위(18 준위 CMFGEN 단면적 행 없음 → §3 상 미정의), C2 406/406; **분기 2 INDETERMINATE**(C1 미완), 개입별 F1·F2 모두 거리 변화 미분리 |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_SI_EXCITATION_BUDGET_2026-10-11.md` — INDETERMINATE(7ae155a), 1회차 반영 44da3e8, **폐합** |
| 종료감사 | astra(계약 저자 이해관계; 자기보고 「GPT-6 계열, 세부 ID 확인 불가」) | 1회차 **폐합 불가 «폐합 차단» 2·«기재로 족함» 1**(`B/astra_closeout.md`) → 판정문 §2·§3·§4·§7·§9 전건 반영(판정문 §10) → 2회차 **폐합 가 «폐합 차단» 0·새 지적 0**(`B/astra_closeout_r2.md`) → **폐합**; 독립 재계산은 재현 BLOCKED |
| 기계 게이트(판정·감사 발주) | 운전자 | 판정 roster·prereg preflight·referents·canonical rc 0(a10 419334), 1회차 감사 발주 dispatch·canonical rc 0(419335), 2회차 발주·반영본 5종 rc 0(419338), 폐합본 4종 rc 0(419341) |
| 비용 | 운전자 | GPU 0; 오프라인 a10 11 작업 할당 181 s · 0.052 core·h / 900 s · 0.5 core·h(기안 게이트 4 작업 71 s·판정/감사/폐합 게이트 4 작업 105 s 포함); 신규 저장 191 KiB / 100 MiB |

**이탈(기재):**
1. §10 등록 인터페이스의 `python` 은 러너에서 `python3 -B` 로 실행했다(운전자 위생 H4). 인자·출력 경로는 등록 그대로다.
2. §10 의 필수 요청에 클러스터 상설 규약 `--exclude=syn102,syn03` 을 더했다.
3. 추출 함수의 채널 반환에서 BB 복사를 상향(흡수)·하향(자발+유도 방출) 두 채널로 나눴다(§4 「BB 복사 상향·하향」). 총 연산자 누적은 바뀌지 않으며 F0 재현이 이를 확인한다.
4. 직전 단 분석 명령 전체의 재실행·바이트 대조는 하지 않았다(§5 「기존 명령 전체를 호출해 직전 단의 분석 산출을 다시 쓰지 않는다」·Q-LEAN). 추출의 정합은 §7 F0 재현으로만 확인했다.
5. 표시본 `.md` 는 같은 메모리 행을 행 종류별 소표로 나눠 썼다(열은 그 종류에 값이 있는 것만).

## §12. 기계 선언

참조는 실재 파일만 지정한다. 분기 fixture는 기존 사전등록 검사 입력이며 새 결정표 재계산 장치가 아니다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SI-EXCITATION-BUDGET (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_si_(?:trace_readout|excitation_budget)\\.py|/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/tools/run_det_si_excitation_offline\\.slurm)",
    "symbol": "det_si_exc",
    "roots": [
      "scripts/det_si_trace_readout.py",
      "scripts/det_si_excitation_budget.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/tools/run_det_si_excitation_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_si_trace_readout.py",
      "scripts/det_si_excitation_budget.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/tools/run_det_si_excitation_offline.slurm"
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
    {"path":"docs/PREREG_DET_SI_CONSUMPTION_TRACE_2026-10-10.md"},
    {"path":"docs/VERDICT_DET_SI_CONSUMPTION_TRACE_2026-10-11.md"},
    {"path":"scripts/det_si_trace_readout.py","flags_existing":["replay","gth","backward","public_si_ratio"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/evidence/USER_DECISION_2026-10-11.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_calls.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_ops.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_slmap.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_vec.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_commit.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/si_trace/si10_ladder_levels.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/model/cmfgen_sigma_bf.bin"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/model/ige_col_14_1_cmfgen.bin"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c","flags_existing":["nlte_bf_collisional_enabled"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c","flags_existing":["load_cmfgen_sigma_bf"]},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/subcol_multi_v6.f","flags_existing":["SUBCOL_MULTI_V6"]},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/steq_multi_v4.f","flags_existing":["STEQ_MULTI_V4"]},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/omega_gen_v3.f","flags_existing":["OMEGA_GEN_V3"]},
    {"path":"data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt","flags_existing":["si1","si2","19.480"]},
    {"path":"data/standart_data1/toy06/phys_toy06_cmfgen.txt","flags_existing":["temp[K]","ne[/cm^3]"]}
  ]
}
```
