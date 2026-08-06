# OUT_E — Fable 인증: Codex D(OUT_D) 물리 평가의 인증/반박

판정자 주: 본 판정이 의존하는 OUT_D 인용 행은 전부 발췌본(`lumina/`, `artis/`)과
`paper_main.tex` 원문을 직접 재열람해 대조했다 [실측]. 운전석 1층 재작성 문서는
`docs/LAYER1_REPLAN_2026-08-07.md` 원문을 읽었다 [실측].
판정값은 `CERTIFIED` / `REFUTED` / `OVERSTATED` / `UNDECIDED` 넷만 쓴다. 수리 코드는 쓰지 않는다.

---

## 0. 판정 요약 (5줄)

1. [실측] ★핵심(발주서 §1): Codex D의 "두 장이 3% 일치했는데도 CMFGEN 대비 1.2–1.8 dex 틀린 사례" 인용은 **CERTIFIED** — `paper_main.tex:867-876`이 정확히 그 사례이고, 논문 자신이 그 일치를 "표본잡음·수송이산화 **배제**, 결함은 **두 팔이 공유하는 emissivity**에 있다"로 사용했다.
2. [판정] 운전석 1층 문서의 전제 "두 팔의 일치는 고리 안에서 얻어지는 **독립 잣대**"(`LAYER1_REPLAN_2026-08-07.md:143`)는 **OVERSTATED** — 일치는 팔-특이 오류(표본잡음·팔별 이산화·순서배선)에 대한 **차동(differential) 잣대**이지, 공통모드(원자자료·공유 source 조립·공유 population·동일 1000-bin 격자·동일 BC/덱 잔재)에는 원리적으로 맹목이다. 논문도 이를 "internal diagnostic"이라 부른다(`paper_main.tex:1044-1045`).
3. [판정] 아키텍처 분류(발주서 §2): "two-arm은 상위호환이 아니라 state 소유권 선택형 별도 아키텍처, 결정론 소유 시 MC-state 인과 되먹임 상실"은 **CERTIFIED** — frozen/shadow/feedback 3계층 gate 실측(`lumina/lumina_cuda.cu:10684-10690`, `8445-8453`, `9206-9223`).
4. [판정] 빠진 물리(발주서 §3): "ARTIS 비열적 excitation 명시 rate / Lumina는 ionization만"은 **CERTIFIED**(`artis/nltepop.cc:546-558` vs `lumina/lumina.h:703-707`); 결손 **크기**는 이 자료로 측정 불가 = **UNDECIDED** ([추정] 19.48d 광구기 고-x_e에서는 소항; 추가 실측: Lumina의 NT ionization 자체가 상수-η 처방으로 ARTIS Spencer-Fano보다 낮은 충실도, `lumina/lumina_plasma.c:15657-15668`).
5. [판정] 부트스트랩(발주서 §4): "ARTIS LTE 시대=물리시간 전진 / Lumina=한 epoch 내 반복"은 **CERTIFIED**(`artis/sn3d.cc:938-963` vs `lumina/lumina_plasma.c:15826-15838`); OUT_C 인정 사슬은 **초기조건 역할은 감당하되 시간기억 역할은 원리적으로 감당 불가** — 단 현 심판(CMFGEN 19.48d 고정 스냅숏, `paper_main.tex:776-786`) 프로토콜 안에서는 후자가 판정을 편향시키지 않는 **범위 경계**다.

---

## 1. ★발주서 §1 — "두 장의 일치"의 증명력

### 1.1 Codex D 해당 주장의 인증

| OUT_D 주장 | 판정 | 근거 (파일:행) |
|---|---|---|
| "두 장이 3% 안에서 일치했는데도 CMFGEN 대비 UV·trace photoionization이 1.2–1.8 dex 틀린 사례가 논문에 있다"(OUT_D §5) | **CERTIFIED** | [실측] `paper_main.tex:867-876` — "overdrives the photoionization of the trace stages by 1.2 to 1.8 dex … returns a mean intensity within 3 per cent of the sampled field over 600 to 3000 Å". |
| "같은 opacity·source를 공유하는 두 팔의 일치는 공통 source bias나 빠진 microphysics의 검증이 아니다"(OUT_D §5) | **CERTIFIED** | [실측] `paper_main.tex:876` — 논문 자신이 그 일치로 "the excess resides in the emissivity that **both solvers read**"를 도출; `paper_main.tex:884-887` — 수렴한 S=J 일치는 "the fixed point of coherent scattering rather than an **independent confirmation**". |
| "MC 복제(seed 2개)로는 공통 field 표현·source bias를 검출하지 못한다"(OUT_D §5) | **CERTIFIED** | [추정] 독립 seed는 분산만 직교화하고 공유 표현·조립은 두 실행에 동일하게 들어간다 — 표준 통계 논리이며 위 실측 사례와 정합. |

### 1.2 물리로 가른다: 일치가 증명하는 것 / 못하는 것

전제 [실측]: 두 팔은 설계상 하나의 plasma state를 공유하고(`paper_main.tex:245-256`),
비교되는 두 장은 **같은 1000-bin log-ν 격자** 위의 값이다 —
MC 추정자도, 결정론 cs.J도 같은 bin 인덱스로 비교된다(`lumina/lumina.h:533`,
`lumina/lumina_cuda.cu:9402-9424`: 동일 인덱스 `iuv`로 `cs.J`와 `nlte_Jmc`를 조회).

| 구분 | 내용 | 근거 |
|---|---|---|
| **증명하는 것** ① | [실측+추정] MC 표본잡음이 비교 대역에서 유의하지 않음 — 잡음은 두 팔에 독립이므로 일치가 상한을 준다. | `paper_main.tex:874-876` (배제 논증) |
| **증명하는 것** ② | [추정] 두 수송 커널(패킷 vs short-characteristic ray)의 **차이가 나는** 이산화 오차의 상한 — 단 두 팔이 입력을 **독립 조립**했을 때만. 논문의 3% 시험은 결정론 팔에 "captured opacity and emissivity of the same iteration"을 먹였으므로(`paper_main.tex:872-873`) 그 시험의 증명력은 **수송 커널로 국한**된다. | `paper_main.tex:872-876` |
| **증명하는 것** ③ | [실측] 순서/배선(ordering) 결함의 국소화 — coevolve shadow의 설계 목적 자체("proves the ordering gotcha is solved"). | `lumina/lumina_cuda.cu:8451-8453` |
| **증명 못하는 것** ① | [실측] 공통 원자자료 오류 — 두 장 모두 같은 `tau_sobolev`·단면적·A계수의 함수. | MC 소비 `lumina/lumina_cuda.cu:10102-10104`; 결정론 소비 `lumina/lumina_cmfgen.c:1795-1806` |
| **증명 못하는 것** ② | [실측] 공유 source-function closure — 사례 그 자체(coherent recirculation gain 5.2×10³, `paper_main.tex:877-889`); 결정론 line source는 기본 B_ν(T_e) fallback(`lumina/lumina_cmfgen.c:1755-1765`, `1804-1806`). | 좌동 |
| **증명 못하는 것** ③ | [실측] 공유 population/SE 오류 — state가 하나이므로 틀린 Fe III population은 두 장을 **같이** 움직인다. | `paper_main.tex:245-256` 설계 서술 |
| **증명 못하는 것** ④ | [실측] bin 내부 구조(좁은 UV edge·선 펌핑 파장) — 비교 자체가 동일 1000-bin 격자 위라 intra-bin 정보는 양쪽에서 동일하게 소실. MC 수송은 선-분해로 돌지만(`lumina/lumina_cuda.cu:3877-3953` 계열) J-비교는 binned 층에서 이루어진다. | `lumina/lumina.h:533`, `lumina/lumina_cuda.cu:9402-9424` |
| **증명 못하는 것** ⑤ | [실측] 공유 BC·격자·덱 잔재 — 같은 T_inner·shell·`transition_probabilities.npy` 상시 기본(OUT_A §3, OUT_C 1.2)이 두 팔에 동일 주입. | `lumina/lumina_atomic.c:1152-1165`(OUT_A 인용 확인) |

### 1.3 공통 편향(shared bias) 원천 열거 — 발주 요구

| # | 원천 | 두-장 잣대의 검출력 | 유일한 판정 경로 |
|---|---|---|---|
| 1 | 원자자료 (A_ul·f_lu·σ_bf·Υ·RR/DR) | [실측] 0 — 완전 공통모드 | CMFGEN 어긋남 지도 + 외부 앵커(NORAD/TOPbase/Badnell) = 캠페인 입력축 그대로 |
| 2 | opacity/emissivity 조립 (동일 tau 배열·binned expansion opacity·S_l closure) | [실측] 0 (frozen-replay 비교 시), [추정] 부분적 (독립 조립 + 팔별 closure를 다르게 둘 때만 >0) | 조립 provenance를 지도에 병기 + CMFGEN |
| 3 | population/SE state (공유 고정점) | [실측] 0 — 설계상 단일 state | CMFGEN population 대조(`paper_main.tex:811-843` 방식) |
| 4 | 주파수 이산화 (동일 1000-bin 격자) | [실측] 0 (J-지도 층), [추정] >0 (선-분해 line-J̄ 층을 비교할 때만 — 결정론 line-J̄가 선행 조건, OUT_C 2.2 조건 5) | 결정론 N_ν ladder 자체수렴 + CMFGEN |
| 5 | 공간격자·내부 BC(T_inner)·기하 | [실측] 0 — 동일 shell·동일 BC | 해상도 ladder + CMFGEN |
| 6 | 계약/위상(세대 장부·1-iter lag 규약) | [추정] 부분 — lag가 팔별로 다르면 검출, 공유 state를 오염시키면 은폐 | 이벤트로그/세대 감사 (고리 밖) |
| 7 | 덱 주입 잔재 (`transition_probabilities.npy`·seed n_e) | [실측] 0 — 두 팔 동일 주입 | K-FRESH 인식론 확장(OUT_C 1.2) + 덱 격리 감사 |

### 1.4 운전석 1층 문서 전제의 판정 — **OVERSTATED**

- [실측] 문서 원문(`LAYER1_REPLAN_2026-08-07.md:143-146`): "두 팔의 일치는 고리 안에서 얻어지는 **독립 잣대**다 — MC와 결정론이 같은 물리를 다른 방법으로 풀기 때문이다. … 고리 얽힘 항목 중 상당수가 '어긋남 지도'로 **직접 판정 가능**해진다."
- [판정] "다른 방법"이 참인 범위는 **수송 연산자**까지다. 잣대의 독립성은 **팔-특이 오류 채널에 한정**되고, 위 표의 공통모드 7원천에는 성립하지 않는다. 이는 사변이 아니라 논문 실측 사례가 증명한다: 두 장이 3% 일치하면서 둘 다 CMFGEN보다 UV ~12배 위였다(`paper_main.tex:867-883`).
- [판정] 캠페인 자신의 공리("닫힌 고리 안 역추적은 원리적으로 순환")와도 충돌한다: 두 장의 일치는 **고리 안에서** 얻어지므로, 캠페인 정의상 "고리가 소비하되 생산하지 않는 것"이 아니다. 같은 욕조에 담근 두 온도계의 일치는 고장 난 온도계를 잡지, 욕조가 틀린 것은 못 잡는다.
- [실측] 단, 방향 자체는 옳다: 논문도 그 일치를 잣대로 실제 성과를 냈다 — 다만 **배제/국소화 잣대**로 썼지 인증 잣대로 쓰지 않았다(`paper_main.tex:874-876`). "3층 항목 **상당수** 직접 판정"은 팔-특이 기전 항목에 한해 참이다.

### 1.5 3층 재작성 계획을 어떻게 고쳐야 하는가

1. [판정] 문구 교정: "독립 잣대" → "**팔-특이 오류에 대한 차동 잣대**". 어긋남 지도의 판정 권한을 항목별로 **사전등록**한다: 후보 기전이 팔-특이(표본잡음·MC 표현·결정론 이산화·순서)인 항목만 J^MC vs J^det 지도가 판정하고, 공통모드 항목(1.3 표의 7원천)은 종전대로 **CMFGEN 어긋남 지도 + 입력축 외부 앵커**에 남긴다 — 즉 캠페인의 고리 밖 프로그램은 축소되지 않는다.
2. [판정] 지도는 **3열**로: J^MC · J^det · J^CMFGEN. 두 팔이 서로 맞고 CMFGEN과 함께 어긋나는 셀×대역 = 공통모드 검출(OUT_D 시험 2와 동일 논리; 논문 §known이 이미 이 형식의 1회 실행).
3. [판정] 비교 provenance 명기: 결정론 팔이 MC capture를 재생(replay)한 비교인지, 공유 state에서 **독립 조립**한 비교인지를 지도에 기록한다 — 전자는 수송 커널만 시험한다(`paper_main.tex:872-873`의 교훈).
4. [판정] 음성 대조를 잣대 자신에 적용: 팔-특이 결함 주입(패킷 수 절감 등)으로 지도가 **잡는** 것을, 공통모드 결함 주입(A_ul 섭동 등)으로 지도가 **못 잡는** 것을 각각 시연해 잣대의 맹목 경계를 게이트로 확정한다(음성 대조 의무 규약의 그대로 적용).
5. [판정] source/형광 축에서 비교에 정보가 생기려면 두 팔의 closure를 **의도적으로 다르게** 두어야 한다(MC macro-atom vs 결정론 solved-population emissivity). 현 champion 기본(양팔 공유 B_ν fallback + coherent scattering, `lumina/lumina_cmfgen.c:1755-1765`)에서는 이 축이 공통이라 지도가 침묵한다.
6. [실측] 사실 정정 1건: 1층 문서 L1-3의 "한 런에서 두 장을 비교하는 진단은 **0건**"(`LAYER1_REPLAN_2026-08-07.md:97`)은 **CPU lane에 한해** 참이다(OUT_C의 조사 범위). GPU coevolve lane에는 동일 런 두-장 비교가 이미 존재한다 — `[COEVOLVE-COLOR]` 3-셸 색/진폭 비교 + 전 셸×빈 `lumina_coevolve_field.csv` 덤프(`lumina/lumina_cuda.cu:9392-9445`). 합류 비용이 낮다는 결론은 오히려 강화된다.

---

## 2. 발주서 §2 — 아키텍처 분류

| OUT_D 주장 | 판정 | 근거 (파일:행 + 한 줄) |
|---|---|---|
| 코드에는 4경로가 실재: 결정론 전용 / THEN_MC 동결 / shadow 무되먹임 / 선택적 MC 되먹임 | **CERTIFIED** | [실측] `lumina/lumina_cuda.cu:7889-7914`(PURE_CMFGEN 우회) · `10073-10085`("plasma FROZEN at the converged pure-CMFGEN state") · `7338-7346`+`8445-8453`("SHADOW buffer only (no state feedback)") · `9206-9223`(photoion_mc, lagged)+`9233-9244`(COEVOLVE_CONSUME jbar, lagged). |
| THEN_MC에서는 T_e·ionization·NLTE 재풀이를 통째로 생략 — state 고정점은 결정론 팔 단독, MC는 그 state에 조건부 분포 | **CERTIFIED** | [실측] `lumina/lumina_cuda.cu:10684-10690` "skip the entire T_e / ionization / coupled-Newton / NLTE re-solve" + `goto frozen_skip_plasma_solve`; `10746` NLTE 재풀이도 `!cmfgen_then_mc` 조건. |
| shadow는 결정론 J를 덮어쓰지 않는 진단이다 | **CERTIFIED** | [실측] `lumina/lumina_cuda.cu:9140-9146` — normalize 동안 포인터를 `nlte_Jmc`로 바꿔치기해 "deterministic nlte.J_nu is never overwritten". |
| ★"결정론이 state를 소유하면 MC-state 인과 되먹임을 잃는다" | **CERTIFIED** | [실측] 위 두 행의 구조적 귀결 — MC realization(희귀 UV 패킷 사슬 포함)이 population에 닿는 경로가 코드에서 절단돼 있다; [추정] 이는 분산 제거의 대가로 지불하는 인과 채널이며, ARTIS coevolution은 정확히 그 채널을 가진다(`artis/sn3d.cc:676-686`: 수송 estimator→다음 timestep state). |
| 되먹임 gate를 켜면 ARTIS형 coevolution이 설계 공간의 한 극한으로 포함된다 | **CERTIFIED** | [실측] gate 실재(`9206-9223`, `9233-9244`) + [추정] 소유율 α→1 극한 논증은 건전; 단 "포함"은 현재 기본 실행이 아니라는 OUT_D 자신의 단서(`7338-7346` 기본 OFF, `10080-10083` THEN_MC 아니면 return)와 함께 읽어야 한다. |
| 최종 분류: "상위호환이 아니라 state 소유권 선택형 별도 아키텍처" | **CERTIFIED** | [판정] 위 실측이 두 소유권 극한(동결=분산 0·인과 상실 / MC 소유=coevolution 회복·분산 회귀)을 모두 코드로 시연 — "상위호환" 서사보다 정확한 분류다. |

---

## 3. 발주서 §3 — 빠진 물리 (비열적 excitation)

| 쟁점 | 판정 | 근거 |
|---|---|---|
| "ARTIS에 명시적 비열적 excitation rate가 있다" | **CERTIFIED** | [실측] `artis/nltepop.cc:546-547` `NTC = nonthermal::nt_excitation_ratecoeff(...)` → `:557-558` `ntcoll_bb` 행렬에 적재; macro-atom internal-up에도 포함(`artis/macroatom.cc:130-132`). |
| "Lumina에는 nonthermal ionization만 확인된다" | **CERTIFIED** | [실측] `lumina/lumina.h:703-707` GammaDeposition = `heating_rate`+`nonthermal_ioniz_rate` 뿐; SE 소비 지점도 ionization 항만(`lumina/lumina_plasma.c:14306-14316`); 발췌 전체 grep에서 NT bound-bound excitation 채널 무매치. |
| 결손의 물리적 크기 | **UNDECIDED** (측정 없음) | [추정] 19.48d 광구기 IGE 코어의 높은 전리도(x_e≳0.1)에서 비열적 퇴적의 지배 채널은 열전자 가열이고 excitation 분율은 수 % 이하(Kozma–Fransson 채널 분할의 표준 물리) — 확인된 1.2–1.8 dex photoionization 편차·0.84 dex 평균 population 오차보다 한참 작은 교정일 것; nebular기·저-x_e로 갈수록 커진다. 제공 자료에 실측 상계가 없으므로 크기 판정은 유보. |
| (본 인증의 추가 실측) 결손의 실폭은 excitation 하나가 아니다 | — | [실측] Lumina NT ionization 자체가 상수-효율 처방: `nonthermal_ioniz_rate = pref×heating/ionpot`, `pref=ETA_NONTHERMAL` 기본(`lumina/lumina_plasma.c:15657-15668`) vs ARTIS Spencer-Fano 해(`artis/input.cc:1742-1744` `NT_SOLVE_SPENCERFANO`). [추정] 채널 분할 충실도 격차는 excitation 부재보다 먼저 판정 대상이 될 수 있다 — 대장 기재 권고. |

---

## 4. 발주서 §4 — 부트스트랩 대응

| 쟁점 | 판정 | 근거 |
|---|---|---|
| "ARTIS LTE 시대는 물리 시간이 실제 전진" | **CERTIFIED** | [실측] `artis/sn3d.cc:938-963` timestep당 반복 1회(`n_titer=1`)·`timestep++`; LTE 시대에도 직전 수송 J에서 `T_J`를 얻어 `T_e=T_R=T_J, W=1`(`artis/update_grid.cc:447-467`, `artis/input.cc:1737-1740`). |
| "Lumina bootstrap은 한 epoch 안의 반복" | **CERTIFIED** | [실측] epoch 간에는 homologous rescale뿐(`lumina/lumina_plasma.c:15826-15838` — r,ρ만 재척도, 상태 기억 없음); GREY_ITERS는 한 epoch 내 과도기 반복(`lumina/lumina_cuda.cu:1401-1424`, `8077-8092`). |
| "역할은 유사해도 같은 물리 시간연산자는 아니다" | **CERTIFIED** | [추정] ARTIS의 LTE 시대는 실제 초기 epoch의 근사 궤적(패킷 census가 시간 기억을 운반, `artis/rpkt.cc:507-640` 계열)이고, Lumina의 것은 정적 고정점의 초기추정 — 수렴 시 세척되어야 할 값이다. 범주가 다르다. |
| OUT_C 인정 사슬(seed T_e→LTE→tau→결정론 solve→무잡음 J→SE)이 이 차이를 감당하는가 | **초기조건 역할: CERTIFIED / 시간기억 역할: 감당 불가(단, 범위 경계)** | [판정] 사슬이 대체하는 것은 ARTIS "LTE 시대"의 **정적 문제용 역할**(첫 물질 상태 + 첫 장 공급)이고, 그것은 무잡음 장으로 더 깨끗하게 수행한다 — CMFGEN 자신의 반복 구조와 동형(OUT_C 2.2 판정 유지). [판정] 시간기억(∂/∂t 항·census·freeze-out)은 어떤 epoch-내 반복도 공급할 수 없으나, 이는 부트스트랩의 결함이 아니라 **정적-epoch 정식화의 속성**이며, 현 심판이 19.48d 고정 스냅숏(`paper_main.tex:776-786`)인 한 판정을 편향시키지 않는다. 나중 epoch·nebular 확장 시 재등장하는 **범위 경계**로 대장 기재. |
| (감액 단서) OUT_D §6 Lumina 열의 CPU-lane 행들 | **OVERSTATED** (서술 범위) | [실측] `lumina/lumina_main.c:253-265` 사망 지점(OUT_A §0·OUT_C 요약 1)이 있는 한 §6 표의 CPU-lane 부트스트랩 행들은 **현재 실행되지 않는 배선의 기술**이다. OUT_D는 이 도달 불능을 §6에서 언급하지 않았다 — 표 자체는 코드 사실로서 옳으나 "동작하는 대응물"로 읽히면 과대다. |

---

## 5. OUT_D 기타 주장 일괄 판정

| OUT_D 절 | 주장 요지 | 판정 | 근거 |
|---|---|---|---|
| 전제 | 논문 서술 vs 코드 4경로 분리 | **CERTIFIED** | [실측] §2 표에서 전 행 대조 완료. |
| §1 | "같은 문제를 향한다 ≠ 현재 같은 고정점을 가진다" | **CERTIFIED** | [실측] consistent/inconsistent split 구분은 논문 명시(`paper_main.tex:545-557`); 유한 해상도 차이 실측(`lumina/lumina.h:533`, `lumina/lumina_cmfgen.c:1514-1524` rays=NS+8, `artis/radfield.cc:716-731` multibin/dilute-BB). |
| §2 | 오차 분해표 (variance/표현 bias/공통 source bias/fallback bias) | **CERTIFIED** | [실측] 핵심 행 전부 원문 확인 — 특히 공통 source 행(`paper_main.tex:864-895`, `lumina/lumina_cmfgen.c:1755-1765`)과 ARTIS LTE fallback(`artis/nltepop.cc:1183-1191` — OUT_B/OUT_C 기확인). |
| §2 | coevolution 잡음은 raw-sum 누적이 아니라 state 기억 경유 | **CERTIFIED** | [실측] `artis/sn3d.cc:670-686` — update_grid가 소비 후 `zero_estimators()`; [추정] 수축률 ρ 논증은 건전, 논문의 두꺼운 한계 ρ→1(`paper_main.tex:566-576`)과 정합. |
| §3 | 선중첩: Lumina overlap 보정 = ±10 이웃·3 열폭·τ²/(τ+τ_ov) | **CERTIFIED** | [실측] `lumina/lumina_plasma.c:15758-15823` (스캔 창 `:15799-15801`). |
| §3 | 3D/시간의존: ARTIS만 실제 전진, Lumina는 1D 정적-epoch | **CERTIFIED** | [실측] `artis/sn3d.cc:938-963` vs `lumina/lumina_plasma.c:15826-15838`; 논문도 1D·미래 확장 명시(`paper_main.tex:1023-1048`). |
| §3 | MC 직접 photoionization estimator는 bin 내부 σJ/hν를 flight 중 표본 | **CERTIFIED** | [실측] `paper_main.tex:392-405` — γ_i 정의 + "resolves the frequency structure of the field inside each bin". |
| §4 | 비용 스케일링 표 | **CERTIFIED** (구조), 우열 판단은 OUT_D 스스로 유보 | [실측] O(CL) 조립(`lumina/lumina_cmfgen.c:1791-1807`)·ray 구조(`1514-1524`) 확인; [추정] wall-clock 우열 불가 판단은 옳은 유보. |
| §5 | event ancestry: Lumina는 packet-ID 이벤트, ARTIS 로그엔 packet ID 부재 | **CERTIFIED** (발췌 범위 한정) | [실측] `lumina/lumina_cuda.cu:6236-6237` `d_event_record(1,(unsigned int)p,...)` vs `artis/macroatom.cc:410-414` 로그 필드에 packet ID 없음. |
| §6 | 부트스트랩 표 | **CERTIFIED** + CPU-lane 도달 불능 단서(§4 참조) | 위 §4. |
| 말미 | "어느 경우에도 두 장의 일치만으로 정답 인증 불가, 최종 판정은 CMFGEN" | **CERTIFIED** | [실측] 논문 역할 분담(`paper_main.tex:788-800`) + 규약(★검증 기준=CMFGEN 단일)과 일치. |

REFUTED 판정 대상: **0건** — OUT_D의 [실측] 인용 중 원문과 어긋난 것을 찾지 못했다 [실측].

---

## 6. 1층 재작성 문서(`docs/LAYER1_REPLAN_2026-08-07.md`)에서 고쳐야 할 것

운전석이 고칠 항목 (본 인증의 판정 귀결):

1. **§2 3층(:143-148) 전제 문구 교정** — "두 팔의 일치는 … 독립 잣대" → "팔-특이 오류(표본잡음·팔별 이산화·순서배선)에 대한 **차동 잣대**; 공통모드(원자자료·공유 source 조립·공유 population·동일 bin 격자·공유 BC/덱 잔재)에는 맹목". 근거 사례를 본문에 박아라: `paper_main.tex:867-876`(3% 일치 + 1.2–1.8 dex 오류 공존).
2. **§2 3층 재작성 규칙 추가** — 3층 항목마다 "어긋남 지도가 판정 권한을 갖는가"를 **후보 기전의 팔-특이성 기준으로 사전등록**. 공통모드 항목은 CMFGEN 지도 + 입력축 외부 앵커(캠페인 원래 프로그램)에 잔류 — "상당수가 직접 판정 가능"의 '상당수'를 명단으로 확정하라.
3. **어긋남 지도 사양을 3열로** — J^MC · J^det · J^CMFGEN(셸×대역). 두 팔 상호 일치 + CMFGEN 공동 이탈 = 공통모드 검출이라는 판독 규칙 명기.
4. **비교 provenance 필드 신설** — 각 비교가 "frozen-replay(수송 커널만 시험)"인지 "독립 조립(조립까지 시험)"인지 지도에 기록 (`paper_main.tex:872-873` 교훈).
5. **잣대 음성 대조 게이트 추가** — 팔-특이 결함 주입은 잡히고 공통모드 결함 주입(예: A_ul 섭동)은 **안 잡히는** 것까지 시연해야 잣대 PASS (음성 대조 의무의 잣대 자신에의 적용).
6. **L1-3 사실 정정(:97)** — "두 장을 비교하는 진단 0건"을 "CPU lane 0건; GPU coevolve lane에는 `[COEVOLVE-COLOR]` + `lumina_coevolve_field.csv` 전장 덤프가 기존재(`lumina/lumina_cuda.cu:9392-9445`)"로 정정. L1-3 합류 설계는 이 기존 인터페이스를 계승 대상으로 명기.
7. **L1-3 사슬의 범위 경계 명기** — 사슬은 ARTIS "LTE 시대"의 **정적-epoch 역할**만 대체한다. 시간기억(∂/∂t·census·freeze-out)은 대체 대상이 아니며 `rescale_epoch`(`lumina/lumina_plasma.c:15826-15838`)는 시간의존이 아니다 — 현 심판(19.48d 고정 스냅숏) 프로토콜 안에서는 무해하나, nebular/다-epoch 주장으로 확장 시 재등장하는 경계로 대장 기재.
8. **L1-5 표에 1건 추가** — 비열적 채널: excitation 부재 + ionization 상수-η 처방(`lumina/lumina_plasma.c:15657-15668`) vs ARTIS Spencer-Fano. 처분은 규약대로 조용한 대장 기재(19.48d 광구기 해악은 [추정] 소항; nebular 확장 전 재판정).
9. **형광/source 축 비교 조건 명기** — 두-장 비교가 source 축에서 정보를 가지려면 팔별 closure를 다르게(MC macro-atom vs 결정론 solved-population emissivity) 두는 구성이어야 함; 공유 B_ν fallback 기본(`lumina/lumina_cmfgen.c:1755-1765`) 아래서는 그 축이 침묵함을 3층 항목 배정에 반영.
