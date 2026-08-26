# 사전등록 — DET-BITLOCK: bit-lockstep 인구의 구조적 생산자 귀속 (2026-08-26)

- **저자** = Fable (분담 개정14 ①, CLAUDE.md 상시 승인 ① — 사전등록). 작성 시점 HEAD
  `45d931d` [저자 실행: `git log -1 --format=%H`]. 이 문서는 저자가 커밋하지 않는다(운전석 몫).
- **승계 원천**: A210-ZERO-OPACITY 폐합(`4b61915`)의 승계 미결 #5
  (`docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md` §9), L6C 폐합의 «bit-동일 구조적 생산자
  귀속», 설계 연구 `docs/DESIGN_SEED_PREMISE_REPLACEMENT_2026-08-26.md` §2 **F6**(병행 권고,
  비차단 실측 확인).
- **런 등급: GPU 런 0회.** 판별 전부가 봉인 산출(read-only)·덱·소스·기존 추출물로 계산된다
  (§4 말미에 근거). 무거운 파스는 grammar-debug 단일 CPU(분 단위).
- 봉인 런루트 `<RR>` = `/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`
  — **읽기 전용**(`ZO_RUN_SEAL.*` 실존, 접촉은 전부 grep/파스). 저장소 **src 무접촉**
  (A210-Z2A 런 R-II 진행 중 — 트리 변조 태스크는 한 번에 하나).
- 오라클(CMFGEN 자체런) 정량 인용 **0**. 추측은 `[추정]`, 저자 직접 실행은 `[저자 실행]`.
  이 문서의 `[미실행]` 주장은 **0건**이다(사실 주장 전건을 저자가 실행했다 — 부록 A).
- 소스 행 번호는 전부 **HEAD `45d931d` 기준으로 저자가 직접 확인**했다. 봉인 바이너리 커밋
  `e500e29` 와의 대응은 §2-1 (오프셋 +48 규칙, 검증 명령 병기).

---

## §1. 계약 (하나)

**«봉인 census 가 확정한 hazard 준위쌍 221,382개(전건 등g·전건 `A208_EXACT_ZERO`)의
`n_lower ≡ n_upper` bit-동일 — 인구가 반복 간 ~1.4% 이동해도 유지 — 을 만드는 생산 기전을,
소스 감사와 봉인 산출 판독만으로(런 0회) 단일 기전에 귀속한다.»**

이 단은 **측정·귀속 단**이다. 수리하지 않고, 본선 물리 계약(A2-07 solve 의 J̄ 응답)을 재지
않는다(§8).

## §2. 저자 사전 실측 — 설계 입력 (전건 [저자 실행], 명령은 부록 A)

### 2-1. 좌표계: HEAD ↔ 봉인 바이너리

- `git diff --stat e500e29 45d931d -- src/` → **`src/lumina_plasma.c` 단 1개 파일**,
  94+/142− (= 순 −48행). hunk 전건이 `:13904`–`:15379`(구좌표 시작행; 구범위 상한 :15392) 구간 —
  **A2-10 진단 트랜잭션 영역(Z-2A 격리)뿐**이고, 이 단이 감사하는 생산자 부위(덱 로더·
  projection·solve·재분배·커밋)는 무접촉이다 [저자 실행: `git diff … | grep "^@@"`].
- 따라서 **구좌표(판정문 b7accae) = HEAD 행 + 48** (行 **>15392** 에 한함 — 마지막 hunk
  `@@ -15379,14 +15318,27 @@` 의 구범위 15379-15392 안은 +48 규칙 밖이다 [저자 실행: hunk
  헤더 재확인, r1]. 이 문서의 인용 행 중 그 구간에 드는 것은 없다 — 실해 0). 교차검증 4건 일치:
  19519→19471 · 19655→19607 · 16431→16383 · 21224→21176 [저자 실행].

### 2-2. 봉인 런 구성 (판별의 전제 — 전건 봉인 파일에서 실측)

| # | 실측 | 출처(봉인, read-only) |
|---|---|---|
| C1 | `LUMINA_SUPER_CUTOFF="100"` **실재** | `<RR>/input/resolved_lumina.exports` grep |
| C2 | `LUMINA_SUPER_LEVELS`·`LUMINA_NLTE_ELEMENT_WIDE` **부재** | 같은 파일, `grep -E "SUPER|ELEMENT_WIDE|EW_"` 매치 = C1 한 줄뿐 |
| C3 | `[ARTIS super-cutoff] K=100: 21919 levels lumped → per-ion ≤100 explicit + 1 super-level (set LUMINA_SUPER_LEVELS=1 to activate)` | `<RR>/stdout.log:129` |
| C4 | `[NLTE] Super-levels: off (identity) (13015 FL -> 1173 SL across ions)` — **모드는 identity 인데 레이아웃은 1,173 SL 로 붕괴돼 있다는 자기모순 배너** | `<RR>/stdout.log:236` |
| C5 | `[NLTE-GPU][FAIL-CLOSED-ROUTE]` 마커 **0건** ⟹ GPU 솔버 라우터 미사용(§2-3 ⑧과 결합 시 생산 solve = CPU 후보-번들 경로 단일) | `<RR>/stderr.log` grep -c = 0 |
| C6 | `[A2-INIT][SEED-MATERIAL] … population_generation=1->2` 1건 (iter0 물질 = 시드 NLTE 커밋분) | `<RR>/stderr.log:419` |

### 2-3. 소스 사슬 (전건 HEAD 직독 — 이것이 §3 H-FLAT 의 골격이다)

① **덱 로더의 컷오프 덮어쓰기는 무조건부다** — `src/lumina_atomic.c:1466-1480`:
   `getenv("LUMINA_SUPER_CUTOFF")`(:1468) 가 K>0 이면 `level_super[l] = (level_num[l]>=K ? K
   : level_num[l])`(:1473-1474) 로 **로드된 super 열을 통째 덮어쓴다**. 활성화 조건 검사는
   없고, 배너(:1476-1478)에 "set LUMINA_SUPER_LEVELS=1 to activate" 라는 **권고 문구만** 있다.
② **projection 의 SL 레이아웃은 super_mode 와 무관하게 덱에서 온다** —
   `src/lumina_plasma.c:16336-16347` (`nlte_ion_super_offset[i+1]` 누적이 `max_super+1` 기반,
   :16344), 매핑 루프 :16376-16393 (`g = offset+cursor++` :16383 · `sl = super_offset +
   atom->level_super[l]` :16384 · `fl_to_super[g]=sl` :16387). `super_mode` 는
   :16350-16351 에서 **불리언으로만** 세워진다: `super_requested && n_super_total <
   n_nlte_levels_total`; `super_requested = (LUMINA_SUPER_LEVELS !=0) || element_wide`
   (:16456-16457).
③ **`super_mode` 는 f 계산만 게이트하고 solve 구조는 게이트하지 않는다** — `super_mode` 의
   소비처는 :16350(설정)·:16484(배너)·:19642(빌드 인자) **뿐**이다 [저자 실행: grep 전수].
   `population_within_superlevel_build`(`src/population_contract.c:172-`) 는
   `!super_mode` 면 **`within_sl_frac ≡ 1.0`** (:199-201), super_mode 면 SL-앵커 상대
   Boltzmann `g·exp(-ΔE/kT)/Σ` (:202 이하). 이 재계산은 매 solve 전 의무·fail-closed
   (`nlte_precompute_within_sl_frac_checked` :19633-19656, 의무 호출 :19831-19837).
④ **solve 행렬 차원은 언제나 SL 수다** — `nlte_solve_ion_shell`(:19025) 의
   `N = nlte_ion_super_offset[hi+1] − super_start`(:19042), FL 수는 별도 `N_fl`(:19043).
   집계 매크로 `SOLVE_OF`/`FRAC_OF`(:16993-16994).
⑤ **재분배: `n_FL = b[SL] · f_FL · scale`** — 주석 :19386-19389("Identity mode: … f==1"),
   `xfl[i] = b[sl] * f`(:19395-19399, f 읽기 :19397), 총량 정규화 후 기록
   :19417-19435(대입행 :19418·:19422·:19433).
⑥ 헤더 계약 문면: *"When super_mode==0 every map is identity"* (`src/lumina.h:722-725`) —
   **①에 의해 위반 가능**한 전제다(컷오프 단독 설정 시 map 은 identity 가 아니다).
⑦ **후보(shadow) 레인도 같은 기전을 공유한다** — `src/nlte_population_candidate.c:226`
   (`candidate->nlte = *public_nlte` — `fl_to_super` 포인터·`super_mode`·offset 배열 승계),
   :267-268(인구·f 버퍼만 사유화), :385-391(같은 빌더에 `candidate->nlte.super_mode` 전달).
   후보 번들 solve 는 `nlte_solve_all_impl` 경유(`lumina_plasma.c:20641`) → per-pair
   `nlte_solve_ion_shell` 호출(:20043). 시드-물질 예측자(:16060, 커밋 :16202-16203,
   호출부 `lumina_cmfgen.c:7525`)도 같은 번들 기계다 ⟹ **iter0(P2)·iter1(P3) 공히 단일
   생산 구현**.
⑧ **GPU 솔버는 생산 경로가 아니다** — `nlte_solve_all_gpu`(`src/lumina_cuda.cu:1049`)는
   `[NLTE-GPU][FAIL-CLOSED-ROUTE] producer=A2-07-CPU-TRANSACTIONAL` 를 찍고(:1074-1078)
   CPU `nlte_solve_all` 로 위임(:1080-1081)하며, 레거시 cuBLAS 구현은 `#if 0`
   (:1083, "Forensic-only … never compiled"). C5(마커 0건)와 결합: 이 런에서 라우터 자체가
   호출되지 않았다.

### 2-4. 덱·산출 구조 (전건 [저자 실행])

| # | 실측 | 함의 |
|---|---|---|
| D-a | 커버 3이온 (Z,ion=1) 준위수/lumped(level_num≥100)/lumped 내 고유 g: **Fe 2599/2499/13 · Co 2558/2458/14 · Ni 1000/900/10** (lumped 합 5,857) [저자 실행: 부록 A D-a awk — r1 에서 `END` 전문 병기·재실행 대조] | 고유 hazard 참여 준위(lower 3,407·upper 5,376 — 명령·출력 = 부록 A «r1 추가: 수치 5종»)가 lumped 집합 크기 안에 든다 — H-FLAT 과 정합 |
| D-b | 앵커 쌍: (26,1,**100**) E=6.5194425068 eV g=8 덱 super=**26** ↔ (26,1,**2542**) E=15.7617685743 eV g=8 덱 super=**128** | 덱 원본 super 열 기준 **서로 다른 SL** — ①의 K-덮어쓰기(둘 다 level_num≥100 → lump 100)가 있어야만 같은 SL 이 된다. 에너지도 9.2423 eV 차 — «덱 축퇴 에너지» 가설(§3 H-DEG)은 앵커에서 즉사 |
| D-c | 앵커 line 669992 덱 행: lower 100 · upper 2542 · λ=1341.482626 Å = 1.3414826256e-5 cm · f_lu=1.409e-10 | ΔE↔λ 왕복 정합(12398.41984/9.2423260675 = 1341.48 Å) — 덱 내부 모순 없음. Z-O 실측 λ_cm 하한 1.3415e-5 와 일치 |
| D-d | `/gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv`: **221,382행**, sha256 = `ac7f3ca6def9738acc452121e844af08c4aba80bc83c90c14eb7683db7f75f01` — Z-O 판정문 기재값과 **일치**, 열 = line/Z/ion/lower_global/upper_global | 귀속 입력 목록의 무결 확인 |
| D-e | hazard ROW 프린터는 `n_upper`(%.17g)를 찍고(`lumina_plasma.c:14090-14102`), 선정 ROW 프린터는 `n_upper` **와 per-ion `lower_level`/`upper_level`** 을 찍는다(:14333-14352) | D2·D4 판별이 봉인 stderr 만으로 가능 |

## §3. 가설의 열거 — 무엇이 `n_l ≡ n_u` 를 bit 수준으로 만들 수 있는가

| # | 가설 | 판별 소스 위치 | 사전 상태 |
|---|---|---|---|
| **H-FLAT** (주도) | **컷오프-단독 반(半)상태의 평탄 lump**: C1+C2 로 레이아웃은 lump(①②④)인데 f 는 1.0(③) ⟹ 각 이온의 lumped 준위 **전원**이 `b[lump]·scale` 한 값을 받는다. 등g 는 τ=0 의 필요조건일 뿐(출판 대수 `n_l−(g_l/g_u)n_u`), 인구 평탄화 자체는 g 무관 | §2-3 사슬 전체; 수치 판별 §4 D1-D5 | 소스·구성 실측이 전건 정합. **수치 폐합은 미실행** — 이 단의 몫 |
| H-STAT | 통계 분배: f 가 에너지 없이 `g/Σg` 로 계산돼 `n ∝ g` (등g ⟹ 동일) | `population_contract.c:202-260` 직독 — 실물은 에너지 포함 Boltzmann. 수치로는 D2 가 (Z,iter)당 고유값 ≈ g-계급 수(≤14)를 낼 것 | 소스 문면상 근거 없음. D2 로 반증/채택 |
| H-BOLTZ | super_mode=1 의 정상 Boltzmann f 에서 우연히 동일 | C2·C4 로 super_mode=0 실측 + ΔE≥0.62 eV(λ≤2e-4 cm)에서 exp 비 ≤ e^-0.7 — bit-동일 불능 | 구성 실측으로 사전 배제(잔여: D2 가 다계급이면 재개) |
| H-COPY | 어느 단계가 준위쌍 단위로 `n_u ← n_l` 복사 | `nlte_level_populations` 기록자 전수(§4 A1) — 재분배(⑤)·커밋 외 쌍-단위 기록자 부재 확인. 수치로는 D4: **비-hazard**(등g 아님) lumped 행의 n_upper 가 lump 값과 같으면 복사가 아니라 평탄화 | D4 로 판별 |
| **H-CLAMP** (r1 추가 — 검수 지적 1(a)) | 이온-계급 상수를 부여하는 **다른** 기전(상대-플로어/클램프류 — 이 프로젝트에 클램프 대장 88항목 실존)이 lumped 집합에 한 값을 강제 | ★D2 d=1·D3 Z당 1 지문을 **H-FLAT 과 동일하게** 남긴다 ⟹ **수치 판별(D1-D5)로는 안 갈린다** — 가르는 것은 **A1 소스 감사 단독**(재분배(⑤)·후보 커밋 계열 밖 기록자·값-강제 경로의 존부) | 수치 소거 불능. **판별은 A1 에 걸려 있다**(§7 PB-A 는 A1 결합으로만 선언) |
| H-DEG | 덱이 두 준위를 축퇴(같은 E·g)로 등록 → LTE 류 초기화가 동일값, 이후 공통모드 | 덱 조회 | **앵커에서 기각**(D-b: ΔE=9.24 eV) — 전수 확인은 D1 에 포함(전 쌍 E 차 > 0) |
| H-MATSYM | 행렬 특이/대칭으로 해 성분이 우연 동일 | N=101 lump 행렬에서 서로 다른 SL 성분 221,382쌍 bit-동일은 조직적 강제 없이 불능 `[추정]`; D2 가 g-계급 횡단 동일값을 보이면 이 가설은 설 자리가 없다 | D2 로 소거 |
| H-PUBZERO | 출판 단계가 차이를 0 으로 절사 | Z-O 판정 §6-3 대수(`value==0 ⟺ n_l==n_u`, 언더플로 ~280자릿수 여유 배제)·§6-2 기록자 전수 — **승계 배제** | 기배제(승계) |
| H-SEED | 시드(P2)만의 성질이고 solve 는 소극 통과 | C6 + Z-O 실측 «반복 간 ~1.4% 이동 중 유지» — 유지가 재생산임은 실측; 시드·반복이 같은 구현(⑦)이므로 H-FLAT 에 흡수 | 흡수됨 |

**소거 구조**: H-BOLTZ·H-DEG·H-PUBZERO 는 구성·덱·승계 실측으로 이미 죽었거나 앵커에서
죽는다. 살아 있는 대립은 **H-FLAT vs H-STAT vs H-COPY** 이고, 셋은 §4 D2·D4 의 수치가
서로 다른 지문을 남긴다(고유값 1 vs ~g-계급 수 vs hazard-국한 동일). ★r1(검수 지적 1(a)):
**H-CLAMP 는 이 수치 소거 밖이다** — D2 d=1·D3 Z당 1 지문을 H-FLAT 과 동일하게 남기므로,
H-FLAT 과 H-CLAMP 를 가르는 것은 **A1 소스 감사 단독**이고, §7 PB-A 판정식이 그 결합을
명시한다.

### 3-1. ★고유 lower 3,407 / upper 5,376 구조가 가설을 어떻게 좁히는가

hazard 는 «쌍의 술어»지만 등호는 **추이적**이다: lower L 이 upper 235개와 각각 bit-동일이면
그 236개 준위가 **한 등가류**다. 221,382 변이 고유 준위 ≤8,783개(합집합 상한) 위에 얹혀
있으므로, 등가류는 쌍 단위가 아니라 **거대 블록**이다 — 221,382개의 독립 우연은 원리적으로
배제되고, 생산자는 **준위 공간의 구획(class)에 한 값을 주는 기전**이어야 한다. 그 구획이
무엇인가가 가설을 가른다:

- **H-FLAT**: 구획 = «이온의 lumped 집합 전체»(이온당 1류: Fe 2,499·Co 2,458·Ni 900 —
  D-a). hazard 는 그 안의 등g 선쌍만 τ=0 으로 **표본추출**한다(등g 아닌 lumped 쌍은
  τ=n(1−g_l/g_u)≠0 이라 hazard 밖 — Z-O §6-3 [시사] 의 그림 그대로). 예측: 고유 인구값 =
  (Z,iter)당 **1**.
- **H-STAT**: 구획 = «(이온, g)-계급»(이온당 ≤14류 — D-a). 예측: (Z,iter)당 고유값 =
  참여 g-계급 수.
- **H-COPY**: 구획 = 복사 사슬의 연결성분 — hazard 그래프 밖(등g 아닌 행)에서는 동일성이
  **없어야** 한다. D4 가 정확히 그 밖을 잰다.

관측된 재사용 최다(235/168회)·인덱스차 양수(9~2495)는 세 가설 모두와 모순 없다 — 선 목록의
성질이므로 판별력이 없고, 판별은 D2·D4 의 **값 수준**에서만 나온다. (이 절의 수치 5종 —
고유 lower **3,407** · 고유 upper **5,376** · 재사용 최다 **235/168** · 인덱스차 **9~2495** —
전건 [저자 실행]; 명령줄·출력 = 부록 A «r1 추가: 수치 5종», r1 재실행 대조 일치 — 검수
지적 3 반영.)

## §4. 판별 설계 (전부 오프라인·read-only — 런 0회)

### A. 소스 감사 (봉인 무접촉; 감리-B 재현 가능 형태로 등록)

| # | 감사 항목 | 방법 | 폐합 조건 |
|---|---|---|---|
| A1 | `nlte_level_populations` **기록자 전수** — 재분배(⑤)·후보 커밋 계열 밖의 기록자(특히 쌍-단위 복사 및 **이온-계급 상수 강제 — 클램프/플로어류(H-CLAMP)**) 부재 | `grep -n "nlte_level_populations" src/*.c src/*.cu` 전 매치의 좌변/우변 분류 + 각 기록자의 인덱스 구조 직독 | 기록자 목록표 완성, H-COPY·**H-CLAMP** 형 기록자 0 또는 발견 시 그 행 명시(발견 시 §7 판정 단계에서 PB-A 선언 금지) |
| A2 | §2-3 사슬 ①-⑧의 **행 단위 재확증** (HEAD + e500e29 오프셋 규칙) | 부록 A 의 명령 재실행 + `git diff e500e29 HEAD -- src/` hunk 무접촉 재확인 | 전건 일치 |
| A3 | 반(半)상태의 **계약 위반 특정**: `lumina.h:722-725` 문면("super_mode==0 ⟹ every map is identity") 대 ①의 무조건 덮어쓰기 — 위반 경로의 최소 재현 조건(env 2개의 조합표 4칸) 문서화 | 소스 직독 + 조합표 작성(측정이지 수리 아님) | 4칸 표 완성 |
| A4 | 시드(P2)와 반복(P3) 커밋이 같은 재분배를 지나는 **호출 사슬 도식** | :16060→:16202→`nlte_population_candidate.c`→:20641→:20043→:19386 | 사슬 도식 + 각 화살표의 행 번호 |

### D. 봉인 산출 수치 판별 (grammar-debug; 판독기 `scripts/analyze_bitlockstep_producer.py` 신설)

| # | 판별식 | 입력 | 가설별 예측 |
|---|---|---|---|
| **D1** | 221,382 쌍 **전건** 양 준위 `level_num ≥ 100`(K-덮어쓰기 규칙의 오프라인 재현: global→(Z,ion,level) 매핑은 levels.csv 행 순서 — Z-O 부록 B `map_mismatch=0` 승계) **그리고** 전 쌍 `E_u − E_l > 0` | TSV + `<RR>/input/model/levels.csv` | H-FLAT: 221,382/221,382. 반례 1건이라도 → PB-X(분할 밖). H-DEG 전수판: E 차 0 인 쌍 0 |
| **D2** | (Z,iter)별 hazard 행 `n_upper` 의 **고유 hex 값 수** `d(Z,iter)`; `dmax = max` (6그룹, shell=0) | `<RR>/stderr.log` ROW 스트림 (iter 경계 = SUMMARY 행 225081/449539 — Z-O 부록 승계) | **H-FLAT: 전 그룹 d=1 (dmax=1)** · H-STAT: d ≈ 참여 g-계급 수(≤14) · H-COPY/비붕괴: d ≫ 64 |
| **D3** | (Z)별 반복 간 상대이동 `n₁/n₀ − 1` 의 고유값 수 — H-FLAT 은 **Z당 정확 1**(공통모드의 실체가 `b[lump]·scale` 한 쌍의 이동임) | 동일 | H-FLAT: Z당 1 (전체 ≤3), 다중도 {Fe 106,502 · Co 105,685 · Ni 9,195}, min/med/max 가 Z-O 실측 1.1686e-2/1.4416e-2/1.9996e-2 와 일치해야 함(★**대조는 절대값 기준** — Ni 이동은 음수, §4-1 구현 유의; 불일치 = 파스 결함 또는 가설 파탄 — 이름 있는 FAIL) |
| **D4** | **hazard 밖 대조**: 선정 행(`[LINE-SATURATION-ROW]`, iter0 1,201·iter1 992 — 설계 연구 실측)에서 `upper_level ≥ 100` 인 행의 `n_upper` 를 D2 의 해당 (Z,iter) lump 값과 hex 비교 | 동일 stderr (선정 행은 per-ion level 을 직접 인쇄 — D-e) | H-FLAT: 전건 동일(등g 여부 무관) · H-COPY: hazard 밖에서 불일치. 표본 0이면 `D4_EMPTY` 라벨로 보고(조용한 생략 금지) |
| **D5** | 덱 전수 예측집합 정합: `P_deck` = {ion=1 선: 양 준위 lumped ∧ g_l==g_u} 를 덱에서 구성, **관측 221,382 ⊆ P_deck** 및 per-Z 관측/예측 수 보고(창 밖 잔여는 보고만 — 창 재현은 이 단 밖) | levels.csv + line_list.csv + TSV | H-FLAT: 포함 위반 0. 위반 1건 → PB-X |

**런 0회 근거**: D1-D5 의 입력은 전부 이미 존재하는 봉인 파일·추출물이고, A1-A4 는 소스
직독이다. 계산 최대치는 stderr 151 MB 1패스 + 덱 조인(수십만 행) — grammar-debug 등급.
PB-C/PB-X 로 착지해도 **이 단 안에서 런을 열지 않는다**(§7 처분).

### 4-1. D2·D3 사전 실측 (r1 기재, 2026-08-27 — **관측이지 판정이 아니다**)

코드 검수 회신(통과: 차단 0·지적 3 비차단)과 함께, **검수자(Fable 검수 컨텍스트)와 운전석이
각자 독립으로** D2·D3 을 계산했고 **전 수치가 일치**했다 — 두 계산 모두 **저자·판독기와
무관한 자체 파서**로 봉인 `<RR>/stderr.log` 를 1패스(2026-08-27, 운전석 경유 회신). 저자도
제3의 파서(awk 1패스 + python hex 변환)로 재계산해 일치를 확인했다 [저자 실행: 부록 A
«r1 추가: D2·D3 저자 재계산»]. 판독기 `analyze_bitlockstep_producer.py` 는 이 시점
**미구현** — 이 실측은 판독기 산출이 아니며 G-B 게이트·selftest 를 대체하지 않는다.

| Z | iter0 hex | iter1 hex | 고유값 | 행수(iter당) | rel = n₁/n₀−1 |
|---|---|---|---|---|---|
| 26 | `0x1.d1534dba20c26p-12` | `0x1.daa141c1395e6p-12` | 각 **1** | 106,502 | +1.999569e-02 |
| 27 | `0x1.9a2c1ea029ae1p-9` | `0x1.a015e376414d4p-9` | 각 **1** | 105,685 | +1.441628e-02 |
| 28 | `0x1.3799ea75acadep-10` | `0x1.33f5bf5d863f1p-10` | 각 **1** | 9,195 | **−1.168571e-02** |

- dmax = **1** · 다중도 {26: 106,502 · 27: 105,685 · 28: 9,195} = D3 사전등록 다중도와 정확
  일치 · Fe iter0 hex = Z-O 판정 **E-Z3 앵커** `n_upper` hex 와 동일
  [저자 실행: `grep "0x1.d1534dba20c26p-12" docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md`].
- ⟹ **D2 실측이 H-FLAT 예측(전 그룹 d=1)에 정확히 착지했다.** 이것은 **관측 기재**다 —
  분기 선언은 여전히 판정자 몫이고(§7: 판독기 라벨 + A1-A4 논리곱), d=1 은 «이온당 단일값»
  기전 **일반**의 지문이라 **H-FLAT 특정은 A1 이후에만** 선다(§3 H-CLAMP·§8-1 ②).
- ★**D3 구현 유의**: **Ni(Z=28) 이동값이 음수다**(−1.168571e-02). 판독기가 부호 있는
  고유값으로 세면 Z당 1 은 그대로 유지되나, **Z-O 기재 1.1686e-2 와의 대조는 절대값
  기준**이어야 한다(§4 D3 문면에 동일 반영).

## §5. 기대 변경집합

### 변경집합 표

| 경로 | 성격 | 내용 |
|---|---|---|
| `scripts/analyze_bitlockstep_producer.py` | **신설** (Codex 발주) | D1-D5 판독 + G-B 게이트 + `--selftest`(음성대조). 결과 라벨 `A210_BITLOCKSTEP_PRODUCER_RESULT status=… branch=…` **명시 인쇄**(§7) |

### 변경집합 끝

**src 변경 0 · env 변경 0 · 봉인 접촉 = 읽기 전용 grep/파스만.** 위 스크립트 1개가 전부다.
PF-1 은 이 표의 유일 행을 신설-예정(`expected_extra`)으로 선언해 만족시킨다 — 심볼
`A210_BITLOCKSTEP_PRODUCER_RESULT` 는 현 시점 `scripts/` 에 부재함을 저자가 확인했다
[저자 실행: `grep -rl … scripts/` rc=1].

## §6. 게이트 (각각 음성대조 필수 · 봉인 접촉 표기)

| 게이트 | 내용 | 음성대조 (주입 결함 → 이름 있는 FAIL 시연) | 봉인 접촉 |
|---|---|---|---|
| **G-B1** 판독기 selftest | 합성 픽스처(소형 ROW 스트림+덱)로 D1-D5 전 판별식 검증 | **NC-K0** 성한 픽스처 → PASS(branch=PB-A) 의무 · **NC-K1** ROW 1행 삭제 → `ROW_COUNT_MISMATCH` · **NC-K2** n_upper 1행 hex 변조 → dmax=2 로 branch 변동 시연 · **NC-K3** 쌍 1건을 level_num<100 으로 변조 → `MEMBERSHIP_VIOLATION`(→PB-X) · **NC-K4** TSV 행 변조 → `PAIRLIST_HASH_MISMATCH` · **NC-K5** SUMMARY 결손 → `INCOMPLETE_STREAM` | **없음** (합성만) |
| **G-B2** 실측 무결 앵커 | ROW 수 iter0/iter1 == 221,382 == TSV 행수 · TSV sha256 == `ac7f3ca6…f01` · 앵커 행(line=669992) 4필드 hex == Z-O 판정 E-Z3 값 | NC-K1/K4 가 겸함(같은 코드 경로) | **읽기 전용** |
| **G-B3** 소스 인용 실측 | §2-3·§9 인용 행 전건의 존재를 grep 스크립트로 재확증(감리-B 재현 대상) + `git diff e500e29 HEAD -- src/` hunk 가 :13904-15392 구간(구범위 상한 기준 — §2-1, r1 정정) 밖 0건 | **NC-K6** 고의 오행(존재하지 않는 패턴@행) 주입 → `CITATION_MISMATCH` | **없음** |

**개정16 ② 준수**: G-B2(봉인 접촉)는 **코드 검수의 "봉인·실물에 돌려도 되는가" 답변 후에만**
운전석이 실행한다. G-B1·G-B3(무접촉)은 검수와 병행 가능. 발주 전 운전석은 가동 중
트리-변조 태스크 0 을 확인한다(A210-Z2A R-II 는 런이지 트리 변조가 아님 — 확인 대상은
새 src 편집 태스크의 부재).

## §7. 기대치·분기 사전등록

판독기는 종료 시 **반드시 한 줄의 라벨을 명시 인쇄**한다(Z-O 검수 Q2-1 교훈 — 라벨 부재를
분기 증거로 쓰는 것 금지):
`A210_BITLOCKSTEP_PRODUCER_RESULT status={PASS|FAIL} branch={PB-A|PB-B|PB-C|PB-X} dmax=… membership=…/221382 …`
모든 실패 경로는 이름 있는 사유와 함께 **branch=PB-X** 를 명시 부여한다(fail-closed —
UNRESOLVED 침묵 금지).

★r1(검수 지적 1(b)): 판독기 라벨의 branch 는 **수치 분할(D1-D5·G-B)의 보고**다. **PB-A 의
최종 선언은 판정 단계에서 §4A(A1-A4) 폐합 조건 충족과의 논리곱으로만** 내린다 — 수치가
PB-A 여도 A1 이 계열 밖 의심 기록자를 발견하면 **판정 branch=PB-X**.

| 분기 | 판정식 (사전 고정) | 처분 |
|---|---|---|
| **PB-A** | D1 전건 ∧ **dmax ≤ 1.5**(정수이므로 =1) ∧ D3 Z당 1 ∧ D5 포함 위반 0 ∧ G-B 전건 PASS **∧ A1 이 재분배(⑤)·후보 커밋 계열 밖 별도 기록자 0 을 확인(H-COPY·H-CLAMP 형 부재) ∧ A2-A4 폐합 조건 전건 충족** (r1 — A1-A4 명시 결합) | **H-FLAT 확정** — 귀속 종결: 생산자 = «LUMINA_SUPER_CUTOFF 단독 설정의 반상태: lump 레이아웃(무조건 덮어쓰기) × f≡1.0(super_mode=0) × SL-차원 solve ⟹ 이온당 lumped 전원 단일값». 처분 = **조용한 대장 기재**(수리 제안 없음, §8) + 판정·감리 두 겹 후 폐합 |
| **PB-B** | 1.5 < dmax ≤ 64.5 | 소수-계급 붕괴(H-STAT 류) — H-FLAT 기각, 귀속은 «구조적이되 평탄 아님». f 산법 재감사(빌더의 실제 산출 덤프 경로 설계)를 **재등록** 후 속행. 런 없음 |
| **PB-C** | dmax > 64.5 | 값-붕괴 부재 — D4 로 H-COPY 계열 판별을 상세화하는 **후속 사전등록**. 런 없음 |
| **PB-X** (분할 밖) | D1 반례 ≥1 · D3/D5 정합 파탄 · ROW/TSV 무결 실패 · 파스 실패 — 전건 이름 있는 FAIL · **★r1: D 수치 전건 PASS 이더라도 A1 이 계열 밖 의심 기록자를 발견한 경우**(수치-소스 불일치 — PB-A 선언 금지, 판정 단계에서 배정) | 전제 자체의 재검(귀속 판정 내리지 않음). 봉인 재판독 설계로 재등록. 런 없음 |

보조 관측(비게이트, 판정에 기재만): D2 의 (Z,iter)별 d 전값 · D3 의 Z별 이동값 3개 ·
D4 표본 수와 결과 · D5 의 per-Z P_deck 크기(창-절사 잔여). **어느 분기든 이 단은 GPU 런을
발주하지 않는다.**

## §8. 처분 원칙 / 하지 않는 것

1. **수리하지 않는다.** PB-A 가 나와도 `LUMINA_SUPER_LEVELS` 를 켜거나 컷오프를 빼거나
   로더를 고치는 일은 이 단 밖이다 — 귀속 전 수리 금지 원칙의 사후 대칭: **귀속 후 수리는
   별도 단**(V5 권한·본선 영향 평가·bit-앵커 계보 단절 검토가 필요한 결정이다). 발견의
   처분은 조용한 대장 기재.
2. **본선 물리 계약(A2-07 solve 의 J̄ 응답)을 측정하지 않는다.** 그 계약은 L6C 재발주
   (F1/F1b)의 몫이다. 이 단은 lockstep 집합(τ=0 급)의 생산자만 다룬다 — 두 집합의 분리는
   설계 연구 §1-4 실측(동결 잔여 95행 등g 0%).
3. **Z-2A 와 간섭하지 않는다.** R-II 산출을 읽지 않고(이 단의 입력은 Z-1r census 봉인뿐),
   src 를 건드리지 않는다.
4. 스코프: shell=0 · ion=1 · Z∈{26,27,28} · 창 안 — Z-O census 의 스코프를 그대로
   승계한다. 창 밖·타 셸의 lockstep 존부는 **측정하지 않는다**(D5 는 포함성만).
5. 오라클 무의존 — 판정문·감리문 공히 CMFGEN 자체런 정량 인용 금지 승계.

### 8-1. 검수(2026-08-27)가 남긴 불확신 4건 — r1 기재 (이 단이 재도출하지 않는 것)

1. **`n_lower` 는 어디에도 인쇄되지 않는다** — 「n_l≡n_u」 전체가 출판 대수 경유 추론이다
   (Z-O §6-3 승계). 검수는 Z-O 의 τ 기록자 전수·언더플로 배제를 재도출하지 않았다(2겹 감리
   통과 산출의 승계 신뢰). **그 사슬이 틀렸다면 이 단 전체의 전제가 흔들린다.**
2. d=1 은 «이온당 단일값» 기전 **일반**의 지문이다 — H-FLAT **특정**은 소스 사슬(§2-3)과
   **A1** 에 걸려 있다(§3 H-CLAMP · §7 PB-A 결합).
3. C4 의 「13015 FL → 1173 SL」 산술 · 후보 레인 버퍼 사유화(§2-3 ⑦)의 완전성 ·
   `element_wide` 기록자의 런타임 게이팅은 재도출되지 않았다 — **A1/A2 가 덮을 몫**이다.
4. D5 는 포함성 단방향(관측 ⊆ 예측)만 게이트하고 충분성은 보고만이다 — §4 D5 의 저자 자기
   선언.

## §9. 승계 기재 (Z-O 감리-A 후속 권고 2건 — HEAD 행 저자 재확인)

① **global→nlte 단사성**: per-ion 단조 커서 `nlte_ion_level_offset[i] + cursor[i]++`
   (`src/lumina_plasma.c:16383` — 구좌표 16431), 소비처 `a.nlte_lo/up =
   global_to_nlte_level[…]`(:8925-8926)·음수 fail-closed(:8927). 같은 ion 안에서 서로 다른
   global 준위는 서로 다른 nlte 슬롯 ⟹ **같은 배열 원소 별칭 불능** — Z-O §6-2 별칭 배제
   (442,764행 `lower_global==upper_global` 0건) 승계. ★이 단과의 관계: 별칭 배제가 있어야
   «두 원소의 값 일치»가 실명제로 성립하고, H-FLAT 은 그 값 일치를 **한 SL 해의 두 사본**
   으로 설명한다 — 단사성(슬롯 구별)과 모순 없다(구별되는 슬롯에 같은 값이 쓰인 것).
② **K-FRESH 사슬** (판정된 τ = 같은 세대의 출판값 보증): `tau_sobolev_require_refresh`
   정의 :6893-6901 · `tau_sobolev_mark_computed` :6903-6911(불량 생산자 상태 abort) ·
   `tau_sobolev_assert_fresh` :6913-6931; NLTE 재출판 호출쌍 :19471/:19607(구좌표
   19519/19655); 후보 커밋 preflight `computed==required` fail-closed :20792-20799(구좌표
   20840-20847) · 출판 preflight :20877-20894 · 번들 preflight :21175-21178(구좌표
   21224-21226). ★이 단과의 관계: D2-D4 가 읽는 ROW `n_upper` 와 τ 를 만든 인구가 **같은
   세대**라는 전제가 이 사슬로 선다(Z-O §6-3 [한정] 승계).

## §10. 프리플라이트 선언 블록

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 표",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/analyze_bitlockstep_producer\\.py",
    "symbol": "A210_BITLOCKSTEP_PRODUCER_RESULT",
    "roots": ["scripts"],
    "expected_extra": ["scripts/analyze_bitlockstep_producer.py"]
  },
  "branches": {
    "metrics": {"dmax": "max(v)"},
    "rules": [
      {"name": "PB-A", "predicate": "dmax <= 1.5"},
      {"name": "PB-B", "predicate": "dmax > 1.5 and dmax <= 64.5"},
      {"name": "PB-C", "predicate": "dmax > 64.5"}
    ],
    "regimes": [[1.0, 1.49], [2.0, 64.0], [65.0, 6000.0]],
    "adversarial_fixtures": [
      {"name": "flat-lump-all-groups-one", "mix": [[1.0, 6]]},
      {"name": "g-class-collapse", "mix": [[1.0, 3], [13.0, 3]]},
      {"name": "no-collapse", "mix": [[1.0, 1], [2499.0, 5]]}
    ],
    "residual": "PB-X"
  },
  "references": [
    {"path": "src/lumina_atomic.c",
     "flags_existing": ["LUMINA_SUPER_CUTOFF", "ARTIS super-cutoff"]},
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["LUMINA_SUPER_LEVELS", "xfl[i] = b[sl] * f;",
                        "tau_sobolev_require_refresh", "tau_sobolev_mark_computed"]},
    {"path": "src/population_contract.c",
     "flags_existing": ["population_within_superlevel_build"]},
    {"path": "src/lumina_cuda.cu",
     "flags_existing": ["A2-07-CPU-TRANSACTIONAL"]},
    {"path": "/gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv"},
    {"path": "/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1/stdout.log",
     "flags_existing": ["Super-levels: off (identity)", "[ARTIS super-cutoff] K=100"]},
    {"path": "scripts/analyze_a210_zero_opacity_census.py",
     "flags_planned": ["A210_BITLOCKSTEP_PRODUCER_RESULT"]}
  ]
}
```

주: PF-2 의 `v` 는 D2 의 그룹별 고유값 수 벡터(6그룹)를 일반화한 스윕이다. `dmax` 단일 축
분할이라 겹침이 원리적으로 불능이고, 잔여 분기 PB-X 가 분할 밖(전제·무결 실패)을 받는다 —
실제 판독기의 PB-X 배정 규칙은 §7 문면이 정본이다. ★r1: §7 PB-A 의 A1-A4 결합은
**판정-수준 논리곱**(수치 밖 술어)이라 PF-2 의 수치 분할(`dmax` 단일 축)은 **불변**이다 —
겹침·공백 0 유지, 반영 후 재실행 rc=0(부록 A 말미).

## §11. 집행 기록 (분장 장부 **정본** — 개정16 ①; 판정문·감리문은 이 표를 지목만 한다)

| 단계 | 규약상 담당 | 실제 | 위반 |
|---|---|---|---|
| 사전등록 저작 (이 문서) | Fable (개정14 ①) | Fable — 본 문서, HEAD 45d931d | 없음 |
| 사전등록 프리플라이트 (저자 실행분) | 저자 (개정16 ③) | Fable — 부록 A 말미 rc=0 (r0·r1 공히) | 없음 |
| 발주 프리플라이트 (`check_dispatch_preflight.py`) + 트리-변조 0 확인 | 운전석 | (기입 대기) | |
| 판독기 구현 | Codex (개정14) | (기입 대기) | |
| 코드 검수 (첫 질문 = "봉인·실물에 돌려도 되는가", + "발주서가 사전등록 범위를 좁혔는가") | Fable (개정14 ②) | (기입 대기) | |
| 게이트 집행 (G-B1·G-B3 → 검수 후 G-B2) · 판독 실행 | 운전석 | (기입 대기) | |
| 판정 (판정문 저작) | Fable fresh (개정14 ③) | (기입 대기) | |
| 감리-A (물리·주장; 판정과 다른 fresh) | Fable fresh (개정14 ④·개정17) | (기입 대기) | |
| 감리-B (재현; read-only) | Codex (개정17) | (기입 대기) | |
| 커밋·대장 기재 | 운전석 | (기입 대기) | |

개정15 필수 4항목 자리: 발주 프리플라이트 rc = (대기) · `[미실행]` 주장 실행 결과 = **해당
없음(이 문서의 [미실행] 0건)** · 분장 장부 = 위 표 · 발주 중단 = (발생 시 기입).

## §12. 개정 이력

- **r0 (2026-08-26, 저자 Fable)** — 최초 등록, HEAD `45d931d`.
- **r1 (2026-08-27, 저자 Fable)** — 코드 검수 **통과**(차단 0 · 지적 3, 전건 비차단) 반영 +
  D2·D3 사전 실측 기재. 개정 시점 HEAD **동일 `45d931d`** [저자 실행: `git log -1`] — 행 번호
  인용 전건 유효. 개정15 형식 준수(`OLD:`/`NEW:` 작업 · 행 단위 diff 로 의도 외 소실 0 확인).
  **계약(§1)·가설의 판별식 수치·분기 분할(PF-2)은 불변** — 아래는 형식·정합·문구와 관측
  기재뿐이다.
  - 지적 1(a): §3 에 **H-CLAMP** 행 추가(수치 판별 불능 — A1 단독 판별 명시) + 소거 구조
    문단 보강 + §4 A1 폐합 조건에 H-CLAMP 형 포함.
  - 지적 1(b): §7 **PB-A 판정식에 A1-A4 명시 결합** + PB-X 에 «D 수치 전건 PASS ∧ A1 의심
    기록자 발견» 경로 명기 + 판독기 라벨(수치)과 판정 선언(논리곱)의 층 분리 문단 + §10 주
    (PF-2 수치 분할 무영향).
  - 지적 2: §2-1 오프셋 경계 **>15379 → >15392** 정정(마지막 hunk `@@ -15379,14 +15318,27 @@`
    의 구범위 15379-15392 는 +48 규칙 밖; 인용 행 실해 0) + §2-1 첫 불릿·G-B3 구간 상한 동반
    정합.
  - 지적 3: §3-1·§2-4 D-a 수치 5종(3,407 · 5,376 · 235/168 · 9~2495)과 D-a awk 에 **명령줄
    병기 + 재실행 대조**(부록 A «r1 추가» — awk `END` 전문 포함, 출력 전건 일치).
  - **§4-1 신설**: D2·D3 **사전 실측** — 검수자·운전석 독립 자체 파서 2벌 일치 + 저자 제3
    재계산 [저자 실행] 일치; dmax=1 · 다중도 정확 일치 · Fe₀ hex = E-Z3 앵커; ★Ni 이동
    음수의 D3 구현 유의(절대값 대조).
  - **§8-1 신설**: 검수가 남긴 불확신 4건 기재.
  - 프리플라이트 재실행 rc=0 (부록 A 말미).

---

## 부록 A. 저자 실행 원장 (전부 read-only; 로그인 노드는 grep/판독, 파스·프리플라이트는 grammar-debug)

`RR=/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`

```bash
# 좌표계 (§2-1)
git log -1 --format=%H                          # -> 45d931d0…
git diff --stat e500e29 45d931d -- src/         # -> src/lumina_plasma.c 만, 94+/142-
git diff e500e29 45d931d -- src/lumina_plasma.c | grep "^@@"   # -> 전 hunk 13904..15379 구간
# 오프셋 +48 교차검증 (구좌표 -> HEAD)
grep -n 'tau_sobolev_require_refresh(opacity, "nlte_update_tau_sobolev")' src/lumina_plasma.c  # -> 19471 (=19519-48)
grep -n "nlte_ion_level_offset\[i\] + cursor\[i\]++" src/lumina_plasma.c                       # -> 16383 (=16431-48)

# 봉인 구성 (§2-2)
grep -E "SUPER|ELEMENT_WIDE|EW_" $RR/input/resolved_lumina.exports   # -> LUMINA_SUPER_CUTOFF="100" 한 줄
grep -n "Super-levels:" $RR/stdout.log        # -> 236: off (identity) (13015 FL -> 1173 SL …)
grep -n "ARTIS super-cutoff" $RR/stdout.log   # -> 129: K=100: 21919 levels lumped …
grep -c "FAIL-CLOSED-ROUTE" $RR/stderr.log    # -> 0
grep -m1 -n "SEED-MATERIAL" $RR/stderr.log    # -> 419: … population_generation=1->2 …

# 소스 사슬 (§2-3; 발췌 — 전 항목 동일 요령의 grep/sed 직독)
grep -n 'getenv("LUMINA_SUPER_CUTOFF")' src/lumina_atomic.c       # -> 1468 (덮어쓰기 블록 1466-1480)
grep -n "atom->level_super\[l\] = K" src/lumina_atomic.c          # -> 1473
grep -n "if (!super_mode)" src/population_contract.c              # -> 199 (1.0 채움 199-201)
grep -n "int N    = nlte->nlte_ion_super_offset" src/lumina_plasma.c   # -> 19042
grep -n "xfl\[i\] = b\[sl\] \* f;" src/lumina_plasma.c            # -> 19398
grep -n "super_mode" src/lumina_plasma.c                          # -> 16350·16484·19642 (소비처 전수)
grep -n "candidate->nlte = \*public_nlte" src/nlte_population_candidate.c   # -> 226
grep -n "nlte_solve_all_impl\|nlte_solve_ion_shell" src/lumina_plasma.c | grep -E "20043|20641"  # -> 호출 사슬
sed -n '1049,1083p' src/lumina_cuda.cu                            # -> CPU 위임 + '#if 0' 레거시

# 덱·산출 (§2-4)
head -1 $RR/input/model/levels.csv   # -> …,level_number,energy_eV,g,metastable,super_level,…
awk -F, '$1==26&&$2==1&&($3==100||$3==2542)' $RR/input/model/levels.csv
#  -> 26,1,100,6.5194425068,8,0,26,… / 26,1,2542,15.7617685743,8,0,128,…
awk -F, '$5==669992{print;exit}' $RR/input/model/line_list.csv
#  -> 26,1,100,2542,669992,1341.482626,-1.409e-10,1.409e-10,…,1.3414826256e-05
awk -F, 'NR>1&&$2==1&&$1>=26&&$1<=28{t[$1]++;if($3>=100){l[$1]++;g[$1","$5]++}}END{for(z=26;z<=28;z++){n=0;for(k in g){split(k,p,",");if(p[1]==z)n++};print z,t[z],l[z],n}}' \
    $RR/input/model/levels.csv
#  -> 26 2599 2499 13 · 27 2558 2458 14 · 28 1000 900 10   (Z 준위수 lumped 고유g — r1: END 전문 병기·재실행 대조 일치)
wc -l /gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv        # -> 221382
sha256sum /gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv    # -> ac7f3ca6…f01 (일치)
grep -rl "A210_BITLOCKSTEP_PRODUCER_RESULT" scripts/                   # -> (없음, rc=1)
sed -n '14087,14102p;14333,14352p' src/lumina_plasma.c                 # -> ROW 프린터 필드 (D-e)
```

r1 추가 (2026-08-27 — 전건 [저자 실행], 파스는 grammar-debug, read-only):

```bash
# 지적 2 재확인: 마지막 hunk 구범위 -> +48 경계는 >15392
git diff e500e29 45d931d -- src/lumina_plasma.c | grep "^@@" | tail -1   # -> @@ -15379,14 +15318,27 @@

# 수치 5종 (§2-4 D-a 함의·§3-1; TSV 열 = line/Z/ion/lower_global/upper_global, 탭 구분·헤더 없음)
TSV=/gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv
awk -F'\t' '{a[$4];b[$5]}END{print length(a), length(b)}' $TSV                   # -> 3407 5376
awk -F'\t' '{c[$4]++}END{m=0;for(k in c)if(c[k]>m)m=c[k];print m}' $TSV          # -> 235
awk -F'\t' '{c[$5]++}END{m=0;for(k in c)if(c[k]>m)m=c[k];print m}' $TSV          # -> 168
awk -F'\t' '{d=$5-$4;if(mn==""||d<mn)mn=d;if(d>mx)mx=d}END{print mn, mx}' $TSV   # -> 9 2495

# D2·D3 저자 재계산 (§4-1; iter 경계 = stderr 행 225081/449539 의 SUMMARY — Z-O 부록 승계,
#  두 행 번호는 grep -n "LINE-SATURATION-SUMMARY" 로 재확인함)
awk '/\[A2-10\]\[LINE-SATURATION-ZERO-OPACITY-ROW\]/{itr=(NR<225081)?0:((NR<449539)?1:2);
  z=""; n=""; for(i=1;i<=NF;i++){if(substr($i,1,2)=="Z=")z=substr($i,3);
  else if(substr($i,1,8)=="n_upper=")n=substr($i,9)}; k=z" "itr; c[k]++; u[k" "n]++}
  END{for(k in c) print "COUNT",k,c[k]; for(k in u) print "VAL",k,u[k]}' $RR/stderr.log | sort
#  -> COUNT (Z iter n): 26 0/1 각 106502 · 27 0/1 각 105685 · 28 0/1 각 9195 (itr=2 키 없음)
#  -> VAL (Z,iter)당 정확 1개: 26,0 0.0004437688873013011 / 26,1 0.00045264235457262295 /
#     27,0 0.003129366625290828  / 27,1 0.0031744804538243829 /
#     28,0 0.0011886643746207774 / 28,1 0.0011747739839675887   (전건 다중도 = 행수)
# hex·상대이동: python3 float(v).hex() 및 n1/n0-1 -> §4-1 표와 동일 (Ni rel 음수 -1.168571e-02)

# §4-1 앵커 대조
grep -c "0x1.d1534dba20c26p-12" docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md     # -> 2 (E-Z3 행 포함)
```

프리플라이트 (grammar-debug, repo 루트):

```
$ python3 scripts/check_prereg_preflight.py docs/RUNG_BIT_LOCKSTEP_PRODUCER_2026-08-26.md --root .
[PREREG-PREFLIGHT][PASS] reason=changeset-consistent detail=files=1
[PREREG-PREFLIGHT][PASS] reason=branch-partition detail=trials=200000 rules=3 fixtures=3 residual=PB-X
[PREREG-PREFLIGHT][PASS] reason=references-resolved detail=refs=7
[PREREG-PREFLIGHT][PASS] reason=prereg-preflight detail=RUNG_BIT_LOCKSTEP_PRODUCER_2026-08-26.md
rc=0
```

(저자 실행, 2026-08-26, grammar-debug nested ssh — 기재 후 재실행으로 rc=0 재확인.
★r1 반영 후 2026-08-27 동일 명령 재실행 [저자 실행, grammar-debug]: **출력 4행 동일 · rc=0**
— PF-2 branch-partition PASS(trials=200000 · rules=3 · residual=PB-X) 그대로 ⟹ §7 의 A1-A4
결합은 판정-수준 논리곱이라 수치 분할에 겹침·공백을 만들지 않는다(§10 주·§12 r1).)
