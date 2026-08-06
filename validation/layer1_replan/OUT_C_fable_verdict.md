# OUT_C — Fable 판정: 비물리 지점 + 2-arm 구조 적합성

판정자 주: 모든 판정은 발췌본(`lumina/`, `artis/`)을 직접 재열람해 확인했다.
OUT_A·OUT_B 의 인용 행 중 본 판정이 의존하는 것은 전부 원문 대조를 마쳤다 [실측].
값은 `PHYSICAL` / `NON-PHYSICAL` / `UNDECIDED` 셋만 쓴다. 수리 코드는 쓰지 않는다.

---

## 0. 판정 요약 (5줄)

1. [실측] 현 스냅샷은 **CPU 경로의 어떤 구성으로도 첫 수송에 도달하지 못한다** — 루프 앞 K-FRESH 가 `POP_BF_STALE` 로 죽고(`lumina/lumina_main.c:263-265`), 이 지점은 pure-CMFGEN 분기(`lumina/lumina_main.c:346`)보다 앞이라 결정론 팔도 같이 죽는다.
2. [실측] 죽음의 근원은 클램프가 아니라 **절단**이다: 주석이 선언한 LTE-Saha 부트스트랩(`lumina/lumina_plasma.c:2347`)을 코드가 `(void)phi_neb` 로 폐기하고 stale 을 반환한다(`lumina/lumina_plasma.c:2643-2649`) — 반복 0 의 물질 공급자가 없다.
3. [실측] 설령 부트스트랩이 서도, A2-10 동세대 계약(`lumina/lumina_plasma.c:11983-11992`)과 루프 발행 순서(`lumina/lumina_main.c:545→641→685-702`)가 원리적으로 충돌하고, 비적격 시 `T_e_generation=0` 화(`lumina/lumina_main.c:648,655`)가 다음 plasma 를 죽인다 — iter≥1 도 전 구성 사망 [추정: 제어흐름 도출].
4. [판정] user 가설은 **수정 인정**: "LTE start 필요 = MC 전용" 전제는 기각(결정론 팔도 tau/χ 입력을 요구, `lumina/lumina_cmfgen.c:5117-5133`), 그러나 "seed→결정론 solve→무잡음 J→통계평형" 사슬은 **seed-T_e LTE 상태를 사슬의 첫 고리로 포함하면 성립**하며, 이는 ARTIS 방식이 아니라 **CMFGEN 자신의 구조**다.
5. [판정] 2-arm 은 미구현이다 — 두 팔 모두 실존하고 같은 canonical 계약에 발행하지만(`lumina/lumina_cmfgen.c:3433-3444` vs `lumina/lumina_main.c:545`), env 스위치로 **상호배타** 실행이며(`lumina/lumina_main.c:342-388`) 한 런에서 두 장을 비교하는 진단은 어디에도 없다.

---

## 1. 과제 1 — 지점별 판정

### 1.1 핵심 판정표 (ORDER 지정 5건 + 실측 추가 4건)

| # | 지점 | 판정 | 근거 (파일:행 + 한 줄) |
|---|---|---|---|
| A | 준위 없는 최상단 이온을 자료 결손으로 **거부한 구 계약** (`hi<=lo → POP_ATOMIC_MISSING`) | **NON-PHYSICAL** | [실측] `lumina/population_contract.c:93-105` 주석이 자백: "hi==lo 는 손상이 아니라 정상" — 최상단 이온은 실재 reservoir 이고 기준 배선(ARTIS)은 준위 1개·U=g₀ 를 준다(`artis/input.cc:1226-1234`, `artis/ltepop.cc:188-196`). 물리적 상태를 데이터 손상으로 오인한 계약이었다. |
| A′ | 그 수리인 **Z=1 임시 대입** (2026-08-07) | **NON-PHYSICAL** (단, 규약대로 격리된 부채) | [실측] `lumina/population_contract.c:105` `if(hi==lo){*out=1.0;...}` — 주석 스스로 "g=1(무구조·맨핵)에서만 정확한 임시 대입"이라 명시; 15개 top 은 맨핵이 아니므로 일반적으로 틀린 g 다. [추정] seed 온도(~1e4 K)에서 최상단 점유는 지수 억제되어 부트스트랩 단계 해악은 미미. 처분(조용한 대장 기재 + 소비자측 상한 게이트 + 정본 수리 방향 명기)은 규약에 부합 — 판정값은 물리 근거 유무 기준으로 NON-PHYSICAL. |
| B | `(void)phi_neb` — Saha 를 계산해 놓고 버리고 `POP_BF_STALE` | **NON-PHYSICAL** | [실측] `lumina/lumina_plasma.c:2646-2650` `(void)phi_neb; if(!r1_use){...return POP_BF_STALE;}` 가 주석 `lumina/lumina_plasma.c:2347` "Fail-closed to the B2 LTE-Saha pin (mirrors ARTIS's LTE start)" 과 정면 모순 — 선언된 물리(LTE start)가 배선에서 절단된 화석. [판정 세부] **루프 중** stale view 에서 조용히 Saha 로 갈아타지 않는 fail-closed 자체는 PHYSICAL 한 계약이다(가짜 값 제조 금지). 비물리인 것은 그 계약을 **반복 0 에도 적용해 초기조건을 삭제**한 것 — 반복 0 의 LTE 는 fallback 이 아니라 고정점 반복의 물리적 초기조건이다(CMFGEN·ARTIS 공통, `artis/update_grid.cc:584-586`). |
| C | n_e 반복: TARDIS-style damped update + 5% 수렴 | 감쇠 **PHYSICAL** / 5% 선언 **NON-PHYSICAL** | [실측] `lumina/lumina_plasma.c:2793-2800` `n_e=0.5*(new+old)`, `|new-old|/old<0.05 → 수렴 선언` — 감쇠 고정점은 수렴 시 무편향(수렴점에서 new=old)이므로 수치기법으로 무죄. [실측] 그러나 5% 는 주석이 자백하는 TARDIS 기본값 이식(`lumina/lumina_plasma.c:2745-2749`)이고, 이 문턱으로 **전하보존 잔차 최대 ~5% 를 기록 없이 커밋**한다(탈출 시 발행 n_e=감쇠평균, 이온 pop 은 직전 n_e 산물; 루프 뒤 재계산 `lumina/lumina_plasma.c:6718-6722` 후에도 charge-sum 검증 없음). CMFGEN 잣대 대비 근거 없는 상수 + 잔차 무장부 = 계보 화석. |
| D | K-FRESH 를 반복 루프 **앞**에 배치, 전 플라즈마 풀이 요구 | **PHYSICAL** (배치) — 단 전제 미배선으로 현재 도달 불능 | [실측] `lumina/lumina_main.c:253-254` "tau is solver-owned; deck NPY is only an epoch-validated seed and must be overwritten before transport or pure-CMFGEN consumes it" — 첫 소비자 전에 신선한 tau 를 강제하는 것은 ARTIS 의 수송-전-상태-갱신(`artis/sn3d.cc:647-679`)과 동형인 보존 요구다. 결함은 배치가 아니라 그 배치가 전제하는 반복 0 물질 공급자(지점 B)의 부재. seed-T_e 1세대 발행(안 B, `lumina/lumina_main.c:255-261`, `lumina/lumina_plasma.c:6581-6592`)은 이 사슬의 T_e 고리를 이미 끊어놨다 — population 고리만 남았다. |
| E | ★초기 T_e = `T_inner` 전 셸 복제 | **NON-PHYSICAL** (프로파일로서) | [실측] `lumina/lumina_atomic.c:1058-1061` `opacity->t_electrons[i]=config->T_inner` — 균질 팽창 외피에서 외곽 T_e 가 내부 경계온도와 같다는 프로파일은 성립하지 않고, 트윈 실측(가스가 자기 욕보다 2000-2600K 참)과도 상반된 방향의 씨앗이다. [추정] radeq 가 반복마다 T_e 를 재해결하므로 "seed 는 세척된다"는 방어가 가능하나, 유한 반복 + 감쇠 갱신에서 세척 보증은 어디에도 없고, 첫 LTE 이온화 상태·첫 tau·첫 J 가 전부 이 seed 에 걸린다. |
| E′ | 덱 `plasma_state.csv` 를 쓰지 않는 근거 | 계약 **PHYSICAL** / 적용 **비일관** | [실측] CONFIG-PREC 는 witness 를 "never published into PlasmaState"로 격리하고(`lumina/lumina_atomic.c:606-609`, `662-663`) w/t_rad 열만 읽는다(`lumina/lumina_atomic.c:668`) — t_electrons 열은 **아무도 읽지 않는다**. 자체런 검증에서 심판의 답(참조 상태)을 입력으로 되먹이지 않는 격리는 옳은 인식론이다. [실측] 그러나 같은 규칙이 n_e 에는 적용되지 않는다 — seed n_e 는 참조 덱 `electron_densities.csv` 를 그대로 복사한다(`lumina/lumina_main.c:146-149`). T_e 만 격리하고 n_e 는 주입하는 비일관 = 계보 혼재. [추정] 격리를 유지한 물리 seed 는 외부 자료 없이 기하만으로 구성 가능(희석인자 W(r) 스케일링 등) — 방향만 적는다. |
| F | `solve_radiation_field()` 전 인자 `(void)` 무연산 + `coupled_*` 스텁 | **NON-PHYSICAL** (화석) | [실측] `lumina/lumina_plasma.c:1054-1067` 전 인자 폐기, 호출부는 살아 있음(`lumina/lumina_main.c:605-609`); `coupled_set_fine_jnu`/`coupled_newton_solve_all` 도 전 인자 `(void)`(`lumina/lumina_plasma.c:12030-12056`). 해악은 계산이 아니라 **배선도 오독 유발**(노브 표면 동결 감사의 "아무 일도 안 하는 노브" 계열) — 죽은 의식은 제거 대상으로 대장 기재. |
| G | A2-10 동세대 계약 vs 루프 발행 순서 | 계약 **PHYSICAL** / 현 위상에서 **충족 불능** | [실측] `lumina/lumina_plasma.c:11987-11992` 는 `em.radfield_generation == 현재 view generation` 등 삼중 일치를 요구 — τ/χ/η 와 J 의 짝 일치는 에너지 균형의 정합 조건으로 물리적 근거가 있다(클램프 대장의 "formal τ/S 짝 비보장" 교훈과 동일 계열). [실측] 그러나 MC lane 순서는 commit(iter+1)→T_e→발행(iter+1 스탬프)이라 T_e 호출 시점의 발행물은 **항상 한 세대 뒤** — 계약이 원리적으로 충족될 수 없다(`lumina/lumina_main.c:545-546`, `641-644`, `685-702`). pure lane 은 아예 a209 emissivity 발행이 없다(`lumina/lumina_cmfgen.c` 전체에 a209 호출 부재; a208 만 `5159`). 분류: 이식 중 재배열 누락 — ARTIS 는 수송 전 갱신인데 T_e 를 수송 뒤로 옮기며 발행 시점을 같이 옮기지 않았다. |
| H | 비적격 시 `T_e_generation = 0` 화 | **NON-PHYSICAL** | [실측] `lumina/lumina_main.c:645-649` 주석 "Preserve the committed material temperature" 바로 아래에서 `plasma.T_e_generation = 0` — 보존을 선언하고 무효화를 실행하는 자기모순. 0 화는 다음 `compute_plasma_state` 를 `POP_INVALID_TE` 로 죽인다(`lumina/lumina_plasma.c:6646-6650` 계열; OUT_A §4 확인). [추정] 그 결과 radeq ON/OFF 어느 구성이든 iter≥1 플라즈마 블록은 `EXIT_FAILURE` — 지점 G 와 결합해 "iter>0 물리"가 전 구성에서 실행 불능. |

### 1.2 추가 실측 지점 (대장 기재용)

| 지점 | 판정 | 근거 |
|---|---|---|
| 초기 전이확률 = 덱 `transition_probabilities.npy`, dynamic 재계산 기본 OFF | **NON-PHYSICAL** (기본값으로서) | [실측] `lumina/lumina_atomic.c:1152-1165`(로드), `lumina/lumina_main.c:183-188`(기본 FROZEN), `705-710`(재계산은 opt-in). K-FRESH 가 tau 에 적용한 "덱 seed 는 소비 전 덮어써야 한다"는 인식론이 전이확률에는 적용되지 않는다 — 참조 상태 주입이 macro-atom 경로에 상시 잔류. E′ 와 같은 비일관 계열. |
| `LUMINA_OUTER_ION_BOOST` phi_neb 증폭 probe | PHYSICAL (계기로서) | [실측] `lumina/lumina_plasma.c:2636-2643` "Diagnostic, not a fix" 라 자기선언한 gated probe — 생산 기본 경로에서 무연산. 단 phi_neb 자체가 지금 `(void)` 폐기라 이 probe 는 현재 완전 사문 [추정]. |
| pure lane 의 `fout=0.5` 탈출분율 등 (`LUMINA_CMF_OVERLAP`) | UNDECIDED | [실측] `lumina/lumina_cmfgen.c:5257-5259` 하드코딩 0.5, env 로 조정 — 전부 gate OFF 기본의 falsifier 장치라 생산 판정 대상 아님; 생산 승격 시 재판정 요. |
| log-sum-exp 이온 사다리 (1e30 캡 대체) | **PHYSICAL** | [실측] `lumina/lumina_plasma.c:2670-2698` — 오버플로를 캡으로 "수리"하지 않고 로그 공간 정확 해로 푼다. 클램프 금지 원칙의 모범 구현, 이월 자산. |
| MC 추정자→canonical commit→checked view 사슬 | **PHYSICAL** | [실측] `lumina/lumina_main.c:545-559` — 세대·provenance·q-set hash 를 지닌 단일 장부 발행; ARTIS 의 estimator reduce(`artis/sn3d.cc:681-686`)와 등가이되 계약이 더 강하다. 이월 자산. |

### 1.3 MISSING 판별: 의도적 변경 vs 이식 누락 (ORDER 요구)

| OUT_B MISSING | 갈래 | 근거 |
|---|---|---|
| 복사장 없는 첫 population 공급자 (LTE start) | **의도적 변경(계약 강화)이 낳은 미완 이관 = 이식 누락** | [실측] 절단 자체는 의도다 — "The legacy nebular/Saha value above is a shadow diagnostic. The only physical supplier is the checked canonical BF view"(`lumina/lumina_plasma.c:2645-2646`). 그러나 대체 공급자(반복 0 선언적 LTE)는 지어지지 않았고, 주석 2347 은 여전히 옛 의도를 선언 중 — anti-fallback 이관이 초기조건까지 쓸어간 **과잉 적용**이며 seed-T_e 발행(안 B)이 그 수리의 절반만 완료한 상태. |
| SE 실패/저온의 LTE fallback (ARTIS `nltepop.cc:1183-1191`) | **의도적 변경 — 유지 지지** | [실측] Lumina 는 오류 반환+transaction abort(`lumina/lumina_plasma.c:15161-15188`). [판정] 루프 중 fallback 은 결함 은폐라서 fail-closed 유지가 옳다(정확해가 위반 불능인 refusal 은 클램프가 아님). 단 반복 0 초기조건과는 별개 사안임을 위 B 에서 갈랐다. |

---

## 2. 과제 2 — 2-arm 구조 적합성

### 2.1 현 배선이 2-arm 을 구현하는가 — **아니다 (스위치 구조)**

- [실측] 결정론 팔은 실존하며 완결이다: `cmfgen_run` 이 opacity 조립→ALI formal solve→canonical field commit→T_e→plasma→NLTE→tau 전 사슬을 돈다(`lumina/lumina_cmfgen.c:5112-5360`, 주석 "downstream solvers reused unchanged" `5310`).
- [실측] 두 팔은 같은 canonical 계약에 발행한다 — MC: `provenance=MC_PATH_LENGTH, statistic=ESTIMATOR_COUNT`(`lumina/lumina_main.c:545`); 결정론: `provenance=CMFGEN_REPLAY, statistic=DETERMINISTIC`(`lumina/lumina_cmfgen.c:3433-3443`). 소비자 게이트 `parity_field_built` 는 provenance 를 가리지 않는다(`lumina/lumina_plasma.c:2372-2377`).
- [실측] 그러나 실행은 상호배타다: `LUMINA_PURE_CMFGEN=1` 이면 MC 루프를 통째로 우회하고 종료(`lumina/lumina_main.c:342-388`), 기본이면 결정론 solve 는 한 번도 호출되지 않는다. 한 런에서 두 팔이 하나의 plasma state 를 **동시에** 공유·순환하는 배선도, 두 장(J^MC vs J^det)의 일치를 계산하는 진단도 없다 — `cmfgen_validate` 는 단일팔 해석적 극한 자가검사일 뿐(`lumina/lumina_cmfgen.c:2737-2760`).
- [판정] 논문 Fig.1 의 주장("두 솔버가 하나의 plasma state 를 공유하며 순환, 두 장의 일치 자체가 진단") 대비 현 배선은 **"MC 단일팔 + 덧댐"도 아니고 "완성된 2-arm"도 아닌, 두 대의 완결된 단일팔 기계가 스위치 뒤에 놓인 상태**다. 다만 공유 계약(canonical field·발행 장부)이 이미 있어 합류 비용은 구조적으로 낮다 [추정].

### 2.2 ★핵심 가설 판정 — **전제 기각, 결론은 수정 후 인정**

가설: "ARTIS 가 LTE start 를 필요로 하는 것은 MC 전용이기 때문 → Lumina 는 결정론 팔이 있으니 부트스트랩 = seed 상태에서 결정론 formal solve → 무잡음 J → 통계평형일 수 있다."

**전제 기각.** [실측] LTE start 의 필요는 잡음이 아니라 인과 순서에서 온다: 어떤 수송이든(MC든 formal 이든) 입력으로 opacity/source 를 요구하고, opacity 는 population 을 요구하며, t=0 에는 장이 없다. 결정론 팔도 예외가 아니다 — `cmfgen_run` 은 진입부터 `opac->tau_sobolev`·`bf->chi_bf` 를 검사·소비하고(`lumina/lumina_cmfgen.c:5117-5133`), `cmfgen_assemble` 이 그것으로 장을 조립한다(`5163`). 결정적 실측: 현 배선에서 pure lane 조차 MC 와 **같은 지점**(루프 앞 K-FRESH, `lumina/lumina_main.c:263-265` — pure 분기 346 보다 앞)에서 같은 이유(`POP_BF_STALE`)로 죽는다. "MC 전용"이 원인이라면 결정론 lane 은 살았어야 한다.

**결론 수정 인정.** [판정] 사슬을 한 고리 늘리면 성립한다:

> seed T_e → **LTE(Saha) 물질 상태** → solver-owned tau/χ → 결정론 formal solve → 무잡음 J_nu (generation 1) → rate-SE/NLTE 통계평형 → (이후 MC 팔 합류)

즉 결정론 팔이 대체하는 것은 **LTE start 가 아니라, ARTIS 의 "LTE 시대"(num_lte_timesteps 동안의 잡음 낀 MC 수송)** 다. LTE 는 기간(period)에서 순간(instant, 상태-0 한 번)으로 압축되고, 첫 rate-SE 부터 무잡음 장을 먹는다. [실측] 이 구조는 최종 심판 CMFGEN 자신의 반복 구조(formal solve + SE, 패킷 없음)와 동형이므로, ARTIS 모사가 아니라 심판 정합 방향이다.

**ABSENT 7건의 공급자 분해** (결정론 팔이 공급 가능한가):

| ABSENT 입력 (OUT_A §3) | 공급자 | 근거 |
|---|---|---|
| transported canonical J_nu view | **결정론 팔 가능** | [실측] `cmfgen_commit_jnu` 가 세대·validity 를 갖춘 canonical view 를 발행하고(`lumina/lumina_cmfgen.c:3400-3452`) `parity_field_built` 는 provenance 무검사(`lumina/lumina_plasma.c:2372-2377`) — 단 solve 입력(tau/χ)이 선행해야 함. |
| committed Z(T_e) | **어느 팔도 아님 — 장 불요** | [실측] partition 은 T_e+원자자료만 입력(`lumina/population_contract.c:112-121`); ABSENT 인 이유는 transaction 원자성 때문에 ion 단계 실패가 Z 까지 도매 롤백해서다(`lumina/lumina_plasma.c:6651-6675`). LTE start 가 서면 자동 해소. |
| solver-owned n_e·ion population | **LTE start 소관** | [실측] 반복 0 Saha(현재 `(void)` 폐기)가 공급자; 결정론 팔은 세대 1 이후 rate-SE 의 J 입력만 공급. |
| fresh LTE level population | **LTE start 파생** | [실측] committed ion·Z 에서 `compute_tau_sobolev` 내부 계산(`lumina/lumina_plasma.c:2931-2945`). |
| fresh solver-owned tau | **LTE start 후 K-FRESH 가 생산** | [실측] `lumina/lumina_plasma.c:6793-6806`. |
| explicit NLTE level population | **결정론 팔 가능 — 조건부** | [실측] `nlte_solve_all` 은 radiation view **와** line-J̄ view 의 동세대 OK 를 요구(`lumina/lumina_plasma.c:15161-15189`); 결정론 lane 은 line-J̄ commit 이 없어(전 파일 grep 무매치) 현재는 `POP_BB_STALE` 로 죽는다. |
| current-generation opacity/emissivity publication | **위상 수리 소관 (팔 무관)** | [실측] 지점 G — 발행을 field commit 후·T_e 전으로 옮겨야 동세대 삼중항이 존재 가능; pure lane 은 a209 자체가 부재. |

**사슬 성립의 최소 조건** (방향만, 수리 코드 아님):

1. [필수] 반복 0 물질 공급자: seed-T_e LTE Saha 가 `(void)` 폐기 대신 **provenance 스탬프를 단 선언적 공급자**로 실행 — 루프 중 fail-closed 계약은 그대로 두고 반복 0 에만 자격을 준다(fallback 아님, 초기조건임).
2. [필수] 첫 solve 입력의 자기소유: 덱 tau/전이확률 주입 없이 LTE 상태의 solver-owned tau/χ 로 첫 formal solve (K-FRESH 계약 유지, `lumina/lumina_main.c:253-254`).
3. [필수] 세대 단일 장부: 두 lane 모두 루프 인덱스(iter+1 하드코딩, `lumina/lumina_main.c:429,545` / `lumina/lumina_cmfgen.c:5185`) 대신 owner 의 computed+1 을 쓴다 — `radiation_field_begin_mc` 의 연속성 계약(`lumina/radiation_field.c:127-129`)이 이미 이것을 강제한다. 아니면 결정론 부트스트랩 뒤 MC 첫 begin 이 즉사한다.
4. [필수] 발행 위상: a208+a209 를 field commit 직후·T_e 호출 전으로 — 그래야 A2-10 동세대 계약이 충족 가능(지점 G). pure lane 에 a209 신설 포함.
5. [필수] 결정론 line-J̄: 같은 q-set hash·profile identity 로 line view 를 공급 — 없으면 SE 가 `POP_BB_STALE`(`lumina/lumina_plasma.c:15169-15177`).
6. [권장] top-ion partition 앵커(A′) 또는 seed 온도역 무해성의 실측 확인; seed T_e 프로파일의 물리화(E) — 사슬 성립엔 불요하나 수렴 품질에 걸림 [추정].

### 2.3 계측 배선도 — 현 계측은 단일팔 게이트뿐

- [실측] 현존 게이트는 전부 단일팔 신선도·세대 계약이다: A2-10 카운터(blocked_stale 등, `lumina/lumina_plasma.c:11985-11992`), tau assert(`6508-6528`), view 검사(`lumina/radiation_field.c:729-732`). 두 장의 일치를 쓰는 진단은 0건.
- [실측] 논문이 말한 "표시된 인터페이스"는 이미 존재한다 — canonical field 가 provenance_kind·statistic_kind 를 세대와 함께 실어 나르므로(`lumina/lumina_cmfgen.c:3433-3443`), 두 팔이 같은 반복에서 각자 commit 하면 view 층이 그대로 비교점이 된다.
- [판정] 2-arm 이 실현되면 계측의 정점은 "게이트 나열"에서 **동세대 J^MC vs J^det 어긋남 지도**(양×위치×크기; 셸×대역)로 바뀐다 — 이것이 보고 규약(CMFGEN 어긋남 지도)과 같은 형식이 되고, MC 잡음 바닥은 결정론 장이 즉석에서 준다. line-J̄ 까지 비교하려면 조건 5(동일 q-set identity)가 선행해야 한다 [추정].

---

## 3. 자산 처분 (이월 / 폐기)

| 이월 (확보된 물리·계약) | 근거 |
|---|---|
| canonical radiation-field owner: 세대 연속성·provenance·validity·q-set hash | [실측] `lumina/radiation_field.c:117-141,706-768` — 2-arm 합류의 기반. |
| population transaction 원자성(begin/commit/rollback) | [실측] `lumina/population_contract.c:211-217` — 부분 발행 차단은 유지 가치. |
| A2-08/09 signed publication + A2-10 동세대 계약(내용) | [실측] 지점 G — 계약은 옳고 위상만 틀렸다. |
| 결정론 팔 전체(assemble→ALI→commit) + 해석적 극한 자가검사 | [실측] `lumina/lumina_cmfgen.c:5112-5360, 2737-2760`. |
| rate-SE 폐합 기계(canonical view Γ 직적분, Seaton+3-body) | [실측] `lumina/lumina_plasma.c:2386-2439` 계열 — A-2 캠페인 산물. |
| log-sum-exp 이온 사다리, K-FRESH 개념, seed-T_e 1회 발행 계약 | [실측] 1.2 표 / 지점 D. |

| 폐기·격리 (대장 기재) | 근거 |
|---|---|
| `(void)phi_neb` 뒤의 사문 사슬: zeta 보간·ML 보정·twocomp lock·OUTER_ION_BOOST | [실측] 생산 경로에서 결과 무영향(`lumina/lumina_plasma.c:2646` 이후 전부 shadow) — 배선도 오염원. |
| `solve_radiation_field`·`coupled_*` 무연산 스텁과 그 호출부 | [실측] 지점 F. |
| n_e 5% 문턱·전하보존 무장부 (TARDIS 계보) | [실측] 지점 C — 잔차 장부화 전까지 신뢰 불가. |
| 덱 `transition_probabilities.npy` 상시 기본값 | [실측] 1.2 표 — K-FRESH 인식론과 모순. |
| 주석 화석 2건: `2347` "LTE-Saha pin fallback", `main.c:646` "Preserve" | [실측] 코드와 반대를 선언 — 판독자를 두 번 속였다(OUT_A·OUT_B 모두 반박에 행 소모). |

---

## 4. 가장 먼저 손대야 할 지점 3개 (물리가 한 조각씩 서는 순서)

1. **반복 0 물질 공급자 복원 (지점 B)** — `lumina/lumina_plasma.c:2643-2650` 의 절단을, seed-T_e LTE Saha 를 provenance 스탬프 단 **선언적 반복 0 공급자**로 세우는 방향으로 폐합한다(루프 중 fail-closed 는 유지).
   이유: 이 한 조각으로 ABSENT 7건 중 4건(Z·n_e/ion·LTE levels·tau)이 연쇄 해소되고, K-FRESH 가 살아나며, **두 팔 모두** 처음으로 첫 수송에 도달한다. 물리적으로도 이것이 유일한 정당한 초기조건이다(CMFGEN·ARTIS 공통). 다른 어떤 수리도 이것 없이는 실행조차 안 된다 [실측: K1 사망 지점].
2. **세대 장부의 위상 정합 (지점 G+H)** — 발행(a208/a209)을 field commit 직후·T_e 호출 전으로 옮길 수 있는 순서로 재배열하고, 비적격 시 `T_e_generation=0` 화 대신 커밋된 세대 보존(주석이 이미 선언한 의미대로)을 성립시킨다.
   이유: 1번이 서도 iter≥1 은 전 구성에서 죽는다 [추정: 제어흐름]. A2-10 계약(τ/χ/η–J 동세대)은 옳은 물리이므로 계약을 깎지 말고 위상을 맞추는 쪽이 "물리가 서는" 방향이다. 이 조각이 서야 radeq T_e — 즉 열 물리 — 가 처음으로 한 반복이라도 완주한다.
3. **결정론 부트스트랩 + 팔 합류의 최소 조건 (2.2 조건 3·4·5)** — 세대 단일 장부, pure lane 의 a209, 동일 q-set 의 결정론 line-J̄ 를 갖춰, "seed-LTE → 결정론 무잡음 J(세대 1) → rate-SE/NLTE" 사슬을 열고, 같은 반복에서 MC commit 과의 동세대 비교점을 확보한다.
   이유: 1·2 로 선 물리를 **잡음 없는 장 위에서** 처음부터 굴리게 되고(수정된 가설의 실현), 논문 Fig.1 의 "두 장의 일치=진단"이 처음으로 계측 가능해진다. 이것은 감사 편의가 아니라 심판(CMFGEN) 구조와의 정합이다.
