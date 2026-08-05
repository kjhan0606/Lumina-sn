# A2-05 구현 명세 read-only 검수 — 개정 8

- 검수일: 2026-08-06
- 검수 기준: `HEAD=bafd2bbdfbcdb7e84b6f573b37785e30185865f0`
- 범위: 사용자 제시 A2-05 명세, `ORDER_L0_JNU_OWNER_BY_CODEX.md`, A2-01 원장,
  A2-01~04 보고서와 커밋, `OUTSIDE_LOOP_POOL.md`, 현재 `src` 실물
- 작업 규율: 구현·기준 문서·코드 무변경. 이 검수 파일만 추가했다.
- 총평: **개정 필요(현 상태 구현 착수 반대)**. 정본 `RadiationField`를 CPU BF가 직접
  소비한다는 단일 계약은 ORDER와 맞지만, PRRR 채널 자격, 안전대, census 이관 집합,
  checked-read/validity, 음성 대조의 판정력이 닫히지 않았다. 아래 반박 1~7은 구현 전에
  계약으로 고쳐야 한다.

## 반박

1. **[BLOCKER] `*PRRR`이 photoion/collisional/spont/stim 네 채널을 분리해 준다는
   전제가 현 실물과 맞지 않는다.** ORDER §7 L-1bf의 문구는
   `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:400-410`에 실제로 존재하므로 “ORDER 인용” 자체는
   정확하다. 그러나 검증된 현 parser가 확인하는 PRRR header는 `Ion Density`,
   `Electron Density`, `<ion> Photoionization Rates`,
   `Radiative Recombination Coefficient for explicitly treated levels.`뿐이다
   (`scripts/oracle_compare_cmfgen.py:75-133`). 기존 비교기도 spontaneous/stimulated 분리는
   PRRR에 별도 label이 없다고 명시하여 unavailable 처리한다
   (`scripts/oracle_compare_cmfgen.py:403-408`), 실제 이전 gate 결과도 두 split이 0/24
   비교다(`docs/CODEX_GATEB_B15_TEST.md:95-98`). collisional ionization도 PRRR의 분리
   채널로 파싱되지 않는다. 따라서 현재 파일만으로는 명세의 “채널별 합계 ≤0.10” 중
   photoion 외 세 채널을 판정할 수 없고, stimulated 제거 음성 대조도 CMFGEN split
   원장에 대해 판정할 수 없다. source-certified 추가 export를 확보하지 않으면 이
   채널들은 `BLOCKED_MISSING_RATE_EXPORT`로 남겨야 하며, total RR coefficient를 임의로
   spont/stim으로 분해하면 안 된다.

2. **[BLOCKER] PRRR용 안전대를 무조건 `s0-s8`로 고정하면 S II s8 오염을 판정 분모에
   넣는다.** `OUTSIDE_LOOP_POOL.md:1212-1227`은 일반 런 교차 안전대를 s0-s8로
   선언하지만, PRRR 자체의 ion별 self-consistency 실측은 S II에 대해
   `gated shells inside the inconsistent band: ['s8']`을 기록한다
   (`validation/cmfgen_toy06_19p48d/analysis/rates_certification/run_log.txt:127-131`). 같은
   로그에서 Co III/Fe III/S III만 `NONE`이다(`:24-28`, `:73-77`, `:180-184`). Pool의
   `:1233-1244`는 특정 `NONE` 행을 일반화해 인용하여 이 예외를 놓쳤다. PRRR 판정은
   blanket s0-s8이 아니라 **ion별 `bandmask`와의 교집합**이어야 한다. 최소한 S II s8은
   기록 전용/분모 제외로 처리하고, 제외 사유·제외 전후 coverage를 남겨야 한다.

3. **[BLOCKER] A2-01 원장과 “이관 행 1:1 diff” 계약이 현재 BF 소비 변경과 1:1이
   아니다.** 원장의 A2-05 `REPLACE_SCALAR_RATE_READ`는 24행인데
   (`docs/A2_01_DISPOSITION_LEDGER.md:9-32`), 앞 8행과 여러 후반 행은 명백한 BB/line
   소비라 A2-06 범위다. BF로 이름 붙은 `:9160/:9162`, `:11943`, `:13672`는 각각
   BF **준위 population**을 `(W,T_rad)`로 만드는 경로이며, 정본 J 적분 소스 자체가
   아니다(`src/lumina_plasma.c:9152-9163,11937-11945,13666-13674`). 이것을 A2-05에서
   J로 “교체”하면 population 소유권 A2-07을 침범한다. `:11976`은 의도된 dilute-Planck
   비교 heating, `:12034`는 `B(T_rad)` 비교 진단이다(`:11971-11984,12031-12047`). 반대로
   실제 `bf_rate_estimator` CPU rate 소비는 원장 행에 없고 현재
   `src/lumina_plasma.c:2277-2299,2342-2344,5045-5067,16063-16079` 및
   `src/lumina_element_wide.c:597-613,1114-1137`에 존재한다. 따라서 명세는 먼저
   (a) A2-05에서 상태를 바꿀 정확한 원장 행, (b) 진단으로 유지할 행, (c) A2-06/A2-07로
   재배치할 잘못 stage된 행, (d) 원장에 누락된 estimator 소비 site를 구분해야 한다.
   이 정정 없이 “이관 행마다 1:1 diff”를 요구하면 정상 구현이 census gate에서 죽거나
   범위 밖 population/BB를 고치게 된다.

4. **[BLOCKER] “`bf_rate_estimator`를 읽는 rate 소비자 0”은 CPU-only 범위와 충돌한다.**
   GPU 생산·업로드·소비는 `src/lumina_cuda.cu:130,307,327-328,589-590,2199-2200,
   8021-8033` 등에 남아 있고 명세 자체가 GPU rate를 A2-13으로 미룬다. 전역 문자열/호출
   gate가 0을 요구하면 정상 A2-05 결과를 반드시 FAIL시킨다. 인수조건은
   “CPU production physics consumer 0”으로 좁히고, 허용되는 raw 통계 producer/lifecycle,
   출력 전용 진단, A2-13까지의 GPU 잔류를 각각 별도 목록과 카운터로 고정해야 한다.
   estimator를 legacy **rate fallback**으로 남기는 것은 ORDER §2.2와 L-1bf 금지를
   위반하므로 허용 목록에 넣을 수 없다.

5. **[BLOCKER] A2-04 정본에는 checked read API가 없는데 명세가 그 계약을 정의하지
   않는다.** 현재 공개 API는 owner lifecycle/accumulator/commit/validate/dump뿐이다
   (`src/radiation_field.h:184-201`). `radiation_field_commit()` 선언은 `:198-199`, 구현은
   현재 `src/radiation_field.c:418-489`다. A2-04 보고서의 구현행 `:402-403`은 현재 실물과
   16행 어긋난다. 더구나 기존 `radiation_field_validate_owner()`는 NULL 또는 disabled
   owner를 성공(0)으로 돌려서(`src/radiation_field.c:195-208`) rate용 checked reader로
   쓸 수 없다. A2-05는 read-only view의 성공 조건을 최소한 enabled, units, comoving
   frame, expected epoch/shell, `required==computed==현재 iteration generation`, canonical
   edge hash와 shape까지 명시해야 한다. 실패 시 rate 값 0을 반환하지 말고 rate validity를
   반환해야 한다. commit API 외 writer를 추가해서도 안 된다.

6. **[BLOCKER] `UNSAMPLED/OUT_OF_GRID` “상태 전파” 뒤 정상 MC CHAIN을 어떻게 처리할지
   없어 정상 산출을 전부 중단시킬 수 있다.** 현 정본 불변식은 `VALID>0`, sampled
   `EXACT_ZERO==0`, `UNSAMPLED/OUT_OF_GRID==0,count==0`으로 정확히 구현돼 있다
   (`src/radiation_field.h:62-68`, `src/radiation_field.c:210-234,388-415`). MC에서는
   `samples==0`이 정상적으로 UNSAMPLED가 된다(`src/radiation_field.c:270-287`). 명세는
   threshold 이상 구간에 missing bin 하나라도 있으면 전체 rate가 UNSAMPLED인지,
   coverage 상태인지, 그 결과를 SE/transition-probability/게이트가 `BLOCKED`로 받을지
   명시하지 않는다. 작은 값 대입은 금지지만 무조건 process abort도 정의된 계약이 아니다.
   `STALE`도 명세 §1.4에서 빠져 있다. 적어도 rate 결과형의
   `VALID/EXACT_ZERO/UNSAMPLED/OUT_OF_GRID/STALE`, 상태 결합 우선순위, downstream 동작,
   판정 분모 제외와 `BLOCKED_INSUFFICIENT_SAMPLING` 조건을 고정해야 한다.

7. **[BLOCKER] PRRR coefficient에 밀도를 적용하는 산식이 없어 §13 경로 23을 막지
   못한다.** 검증된 현 해석은 Γ=`sum(PR)/Ion Density`, total α는 이미
   `RR/(electron density*ion density)`로 출력된 `cm^3 s^-1` coefficient다
   (`scripts/oracle_compare_cmfgen.py:127-133,403-408`,
   `docs/GATE_B_PHASE1_5_CODEX_A_REPORT.md:108-115`). 명세의 “header 의미대로 분리”만으로는
   coefficient 비교인지 rate-flow 비교인지, `n_e`/upper-ion density를 어디서 정확히 한
   번 곱하는지 결정되지 않는다. channel 합계와 가중 `E_1`에 들어갈 동일 차원의 양을
   수식으로 등록해야 한다. 그렇지 않으면 밀도 0회/1회/2회 적용이 모두 구현 가능하다.

## 확인

1. ORDER §10의 A2-05 행은 현재 `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:653-673`에 있고,
   정확한 문구는 `:660`의 “직접 J_nu 적분, bf_rate_estimator 소비 제거; L-1bf PASS”다.
2. ORDER §6.4 CHAIN/ORACLE_INPUT 판정표 인용은 정확하다(`:356-372`). 특히
   `PASS/FAIL` 조합 네 개와 음성 대조 필수 조건이 명세에 올바르게 승계됐다.
3. ORDER §13 경로 2, 7, 23, 24 인용은 정확하다(`:748,753,769-770`). 다만 위 반박처럼
   경로 23의 실제 산식 gate와 경로 24의 독립 CMFGEN split oracle은 명세에 없다.
4. §9의 exact zero/missing 구분 인용은 정확하다. 실제 문구는
   `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:608-636`이고, A2-04 enum/validator도 그 네 상태와
   STALE을 보존한다.
5. A2-04의 헤더 계약은 실물과 맞는다. 정본 10필드는
   `src/radiation_field.h:90-103`, canonical 4000-bin union/edge는 `:7-15`, commit request는
   `:151-178`, commit 선언은 `:198-199`다. deterministic commit은 source bin-average와
   source validity를 받아 보존 overlap 적분하며 missing 상태를 값 0과 분리한다
   (`src/radiation_field.c:296-385`).
6. A2-04는 HEAD 커밋에서 실자료 L-0 artifact까지 포함해 폐합됐다.
   `validation/a2_04/a2_04_l0_replay_eddfactor.json`이 실재하고,
   `docs/OUTSIDE_LOOP_POOL.md:2033-2056`도 실 EDDFACTOR PASS 및 OUT_OF_GRID 전파 수리를
   기록한다. 반면 `docs/CODEX_IMPL_A2_04.md:10-11,288-290`의
   `PENDING_DRIVER_EXECUTION`은 HEAD 실물에 대해 낡았다. A2-05 회귀는 보고서의 pending
   문구가 아니라 현재 artifact/hash와 HEAD를 기준으로 해야 한다.
7. S07 수치 인용은 정확하다. `docs/CODEX_INPUT_ATOMIC_SUMMARY.md:129-132`는
   26,087/26,592와 fallback 505, `docs/CODEX_WAVE32_B_TEST_SUMMARY.md:8`은 Fe II 122준위
   Γ `+0.0158125051 dex`를 실제로 기록한다.

## 게이트 계약 누락

1. **CHAIN/ORACLE_INPUT 주입 경계:** CHAIN은 어느 generation의 canonical MC field를,
   ORACLE_INPUT은 A2-04 deterministic commit을 거친 CMFGEN J를 쓰는지, 양 lane에서
   population/`T_e`/`n_e`/cross-section을 무엇으로 고정하는지 없다. 바로 위 경계인 J만
   바꾸고 나머지는 동일 snapshot으로 고정해야 원인 분리가 된다.
2. **MC 통계 판정:** ORDER §6.3은 95% CI 반폭이 합격선의 1/3 이하일 때만 판정하도록
   한다(`ORDER...:354`). 명세에는 replicate/variance/seed/CI 산출과 부족 시 BLOCKED가
   없다. count만으로 물리오차 PASS를 내면 안 된다.
3. **matching universe와 coverage:** “매칭 이온”의 사전등록 목록, superlevel/full-level
   crosswalk, upper-ion target, threshold 허용오차, 누락 이온/준위가 f_cov 분모에 들어가는
   방식이 없다. `26,087/26,592` level-count coverage를 ORDER의 CMFGEN rate-flow coverage로
   대신할 수 없다. ORDER §6.3의 누적 기여 99.9% 활성 집합과 f_cov 분모를 그대로
   구현 계약으로 내려야 한다.
4. **부분 빈 적분:** “명시하라”만 있고 산식이 없다. canonical `J_b`를 빈 안에서 상수인
   빈 평균으로 두되 첫 빈은 `[max(nu_threshold,nu_lo),nu_hi]`에서 `sigma(nu)/(h nu)`를
   적분해야 한다. 빈 중심 sigma/nu를 전체 `Delta nu`에 곱하거나 단순 overlap 비율만
   곱하는 방식은 공명 구조와 `1/nu`를 보존하지 않는다. sigma 자체가 tabulated point인지
   bin average인지와 교차격자 적분법도 고정해야 한다.
5. **음성 대조 판정력:** threshold 한 빈 이동과 stimulated 제거는 선택한 witness가
   매끈한 sigma/J 또는 미미한 stimulated flow이면 정상적으로도 10% gate를 깨지 않을 수
   있다. 사전에 민감도 margin이 있는 ion-level-shell witness를 등록하고, 각 poison이
   어느 양성 metric을 FAIL시켜야 하는지와 verifier의 기대 rc(physics FAIL 관측 시 runner
   rc=0)를 명시해야 한다. `W B_nu`도 “모든 채널/모든 셀 실패”가 아니라 사전등록 metric
   실패 조건이 필요하다.
6. **legacy fallback 충돌:** 제약의 “legacy를 게이트 폴백으로 우선 검토”는 L-1bf의
   estimator 금지 및 owner 단일화와 충돌한다. 허용 가능한 것은 비교 전용 shadow/negative
   lane뿐이다. production checked read 실패 뒤 `(W,T_rad)`나 `bf_rate_estimator`로 계속하는
   fallback은 무증상 경로 2/10/12를 재도입하므로 명시적으로 금지해야 한다.
7. **double-normalization failure injection:** A2-04는 commit 입력 형식의 raw/normalized
   동시 제공만 거부했다. A2-05 rate 적분은 이미 정규화된 `J_nu`에 다시 `4pi V t Delta_nu`
   normalization을 적용하지 않는 별도 경로 7 injection이 필요하다. 단위/스케일 sentinel과
   정상 rate의 analytic fixture를 등록해야 한다.
8. **spont/stim 중복:** 구현 산출은 두 채널과 total을 모두 출력하되 total을 다시 개별
   rate에 더하지 않는 항등식 gate가 필요하다. 현재 명세는 “중복 금지”만 있고
   `total == spont + stim` 및 downstream matrix 합산 횟수 검사가 없다.
9. **L-1bb 상태:** A2-05 결과표에 L-1bb를 반드시
   `BLOCKED_MISSING_RATE_EXPORT`로 유지한다는 assertion과 이전/이후 status diff가 있어야
   한다. 단순 보고 문구만으로는 L-1bf runner가 공통 `L-1 PASS`를 써 거짓 승격하는 것을
   막지 못한다.

## classic 부채 6항목 실코드 매핑

1. **H07 — 위치 부정확/의미 갱신 필요.** 현재 legacy macro는
   `src/lumina.h:514-516`이지 `:512-514`가 아니다. A2-04가 include/member를 추가하며 두
   행 밀렸다. 더 중요한 점은 canonical field는 이미 `src/radiation_field.h:7-15`의
   4000-bin amended union이므로 “현 J_nu/BF 격자가 1000-bin”은 정본 전체가 아니라
   `NLTEConfig` legacy 경로에만 맞는다. A2-05 sweep은 legacy BF 발화와 canonical BF
   소비 전환을 구분해 기록해야 한다.
2. **S02 — 인용 범위가 설명/상수뿐이다.** `src/lumina_plasma.c:1072-1096`은 24-bin
   모델의 주석과 상수다. 실제 1000-bin `J_nu` overwrite는
   `nlte_build_perbin_dilute_field()` `:1168-1453`, production 호출은
   `src/lumina_cuda.cu:9114`다. A2-05에서 측정할 firing site는 이 본문/호출부다.
3. **S04 — 위치는 맞지만 진술은 default/fallback으로 한정해야 한다.** ML/Saha 본문은
   `src/lumina_plasma.c:2215-2225,2397-2529,2600-2730`에 있다. 동시에 parity+R1이면
   rate-SE를 쓰는 분기가 `:2233-2268,2519-2529,2624-2628`에 이미 존재한다. A2-05는 새
   Γ의 발화/영향만 재고 이 ion/charge-neutrality solver 자체는 A2-07에 남겨야 한다.
4. **S07 — 일부 위치가 fallback 산술을 가리키지 않는다.** loader fallback 선언
   `src/lumina_atomic.c:1891-1897`과 sigma0 선택 `src/lumina_plasma.c:7365-7373`은 맞지만,
   실제 CPU opacity Kramers 곡선은 `:7615-7626`, NLTE GEMM은
   `src/lumina_nlte_gemm.cu:217-225,372-383`, element-wide는
   `src/lumina_element_wide.c:537-590`이다. 제시된 `lumina_bf_gemm.cu:31`은 설명 주석이고
   `element_wide.c:518-526`은 fallback 카운터/guard일 뿐 곡선 계산이 아니다. A2-05 rate
   소비 site에는 `src/lumina_plasma.c:2329-2344,15997-16075`도 포함해야 한다.
5. **S09 — “모든 upper ion을 ground core 하나”는 현 트리에 대해 과도한 일반화다.**
   legacy pair assembler는 실제로 ground-only임을 `src/lumina_plasma.c:15929-15940,
   16155-16159`가 명시한다. 그러나 target-aware CSR/stim 경로는
   `:7522-7542,7592-7612`, element-wide target route는
   `src/lumina_element_wide.c:529-576`에 이미 있다. 제시된 `:3089-3104,6031-6069`는
   특정 spin-gated Milne 경로의 caveat/함수 전주이지 모든 BF 경로의 실소비 위치가
   아니다. sweep은 pair legacy와 target-aware lane을 분리해야 한다.
6. **S15 — 위치는 대체로 정확하나 opt-in/실제 산술을 밝혀야 한다.** top-stage gate는
   `src/lumina_plasma.c:16246-16332`, dilute-Boltzmann 고립행 anchor는
   `:16625-16667`에 있다. 전자는 `LUMINA_TOPSTAGE_IV` opt-in이고 `n_iv_eff=max(real,
   Saha)`(`:16321-16333`)라 단순 “고정 Saha reservoir”보다 복합적이다. A2-05에서는 발화와
   Γ 변화만 기록하고 reservoir/anchor 수리는 A2-07에 남겨야 한다.

## 명세에 추가할 최소 인수조건

1. PRRR 실물로 가능한 photo Γ와 total α만 `COMPARED`; collisional 및 spont/stim split은
   새 source-certified export 전까지 `BLOCKED_MISSING_RATE_EXPORT`.
2. PRRR 분모는 ion별 contamination mask를 적용하고 S II s8 예외를 회귀 fixture로 고정.
3. checked canonical read view와 rate validity 5상태, generation/frame/unit/edge 검증을
   공개 계약으로 추가.
4. CPU BF production consumer에서 estimator/legacy J/(W,T_rad) rate-source edge 0을 compiler
   callgraph와 runtime positive-consumption counter 양쪽으로 증명하되 GPU/진단 잔류는 별도.
5. A2-01 diff는 population·BB·진단 행을 잘못 완료 처리하지 않고, 실제 estimator 소비
   site 누락을 별도 원장 정정안으로 제시.
6. CHAIN/ORACLE_INPUT의 고정 입력, matching universe, CI, poison witness/기대 rc, channel
   단위와 밀도 적용식을 사전등록.
7. 위 조건 중 하나라도 충족되지 않으면 `PASS`가 아니라 해당 원인의 `BLOCKED` 또는
   `FAIL`; L-1bb 상태는 계속 `BLOCKED_MISSING_RATE_EXPORT`.
