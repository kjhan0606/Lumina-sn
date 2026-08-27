# 판정문 — DET-BITLOCK: bit-lockstep 인구의 구조적 생산자 귀속 (2026-08-27)

- **판정자**: Fable (fresh 컨텍스트, 분담 개정14 ③). 사전등록 저작·판독기 구현·검수·게이트
  집행과 별개 컨텍스트.
- **분장 장부 = 사전등록 §11** (`docs/RUNG_BIT_LOCKSTEP_PRODUCER_2026-08-26.md` r1, 커밋
  `491ce14` — 정본, 개정16 ①). 이 문서는 장부를 복제하지 않는다. 판정 칸 기입분:
  **판정 = Fable fresh (2026-08-27) · 위반 없음** — 단 §9 에 기재하는 특기 2건
  (A3·A4 집행 주체, 발주서 인용 정정).
- **계약(사전등록 §1)**: *봉인 census 가 확정한 hazard 준위쌍 221,382개의 `n_lower ≡
  n_upper` bit-동일을 만드는 생산 기전을, 소스 감사와 봉인 산출 판독만으로(런 0회) 단일
  기전에 귀속한다.* 측정·귀속 단이다 — 수리하지 않고, 본선 물리 계약(A2-07 solve 의 J̄
  응답)을 재지 않는다(§8).
- 모든 수치는 판정자가 직접 실측·재계산했다(부록 A·B — 전 명령 실행 확인 후 기재).
  봉인 런루트에는 아무것도 쓰지 않았고, `src/` 는 수정하지 않았다.

**입력 식별**: 봉인 런루트 `<RR>` =
`/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`
(read-only; `ZO_RUN_SEAL.{txt,sha,meta}` 실존, 지문 136행 [부록 A-1]). 봉인 바이너리 커밋
`e500e29` [`<RR>/ZO_RUN_SEAL.txt` `git_head=` 행]. 판독기 = `scripts/analyze_bitlockstep_producer.py`
(커밋 `a849306` = 판정 시점 HEAD). G-B2 운전석 산출 =
`/gpfs/kjhan/lumina/zo_artifacts/bitlock_result_20260826T161506Z.txt`. 소스 인용 행 번호는
HEAD `a849306` 기준 — `git diff --stat 45d931d a849306 -- src/` **공백**(사전등록 시점과 src
동일)이고, `git diff e500e29 HEAD -- src/` 는 `lumina_plasma.c` 단독·hunk 전건 구좌표
13904–15392 구간(A2-10 진단 영역)임을 판정자가 재확인했다 [부록 A-2] ⟹ 이 판정문의 인용
행은 **ROW 프린터 2곳을 제외하고** 전건 그 구간 밖이라 HEAD 행 번호 = 봉인 바이너리의 행
번호다. 예외인 ROW 프린터는 변경 hunk 구간과 겹쳐 HEAD·봉인의 행 번호가 다르다 — 부록 B
말미에 양 좌표를 구분 병기한다(감리-B NOTE #8(a) 정정).

---

## 1. 증거와 재계산 방법

| 증거 | 접근 |
|---|---|
| `<RR>/stderr.log` (봉인, 행-계측 정본; 지문 OK [부록 A-1]) | ROW/SUMMARY/선정행 전행 스트리밍 파스 — **판정자 자체 파서**(부록 B; 저자 awk·검수자·운전석 파서와 무관한 제4의 계산) |
| `<RR>/stdout.log` · `<RR>/input/resolved_lumina.exports` (지문 OK) | C1–C6 구성 재실측 [부록 A-3] |
| `<RR>/input/model/levels.csv` · `line_list.csv` (지문 OK) | D1·D5 전수 조인 + 덱-원생 super 층위 실측 (부록 B) |
| `/gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv` | sha256 재계산 + 전 행 파스 (부록 B) |
| `scripts/analyze_bitlockstep_producer.py` | 전문 직독 + G-B1/G-B2/G-B3 판정자 재실행 [부록 A-5·A-6] |
| `src/lumina_atomic.c`·`lumina_plasma.c`·`population_contract.c`·`lumina_cuda.cu`·`nlte_population_candidate.c`·`lumina_element_wide.c`·`lumina_cmfgen.c`·`lumina.h` | A1 기록자 전수 census + A2 사슬 행 단위 재확증 + A3·A4 집행 (§4, 부록 A-4) |

판독기 재계산 관계: G-B2 라벨의 수치는 판독기 산출이고, 부록 B 는 같은 봉인 파일을 판독기와
무관한 코드로 다시 계산한 것이다. 두 경로가 전 항목 일치했고(§2), §4-1 의 선행 3벌
(검수자·운전석·저자)과도 일치한다 — 동일 수치의 독립 계산이 도합 **4벌**이다.

---

## 2. 판별식 D1–D5 판정

| # | 등급 | 판정 | 판정자 재계산 (부록 B) | 판독기 (G-B2) |
|---|---|---|---|---|
| **D1** 소속·에너지 | 게이트 | **PASS** | map_bad=0 · membership_bad=0 · energy_bad=0 / 221,382 | membership=221382/221382 |
| **D2** (Z,iter)별 고유 hex | 게이트 | **PASS — H-FLAT 예측 정확 착지** | 전 6그룹 d=**1**, dmax=**1**; 행수 {26: 106,502 · 27: 105,685 · 28: 9,195} 양 반복 동일 | d2 전건 1, dmax=1 |
| **D3** Z별 이동 고유값 | 게이트 | **PASS** | Z당 정확 **1**; abs min/med/max = 1.1686e-2 / 1.4416e-2 / 1.9996e-2 (Z-O §6-1 실측과 일치) | 동일 |
| **D4** hazard 밖 대조 | **보조 관측 (비게이트 — 계약 §7·PB-A 판정식 밖)** | **전건 일치 — H-COPY 수치 기각 완결** (§2-1) | 1,586 표본 전건 lump 값과 hex 일치, mismatch 0 | D4_ALL_MATCH 1586/1586 |
| **D5** 덱 예측집합 포함성 | 게이트 | **PASS** | P_deck {26: 152,821 · 27: 159,064 · 28: 13,604} · 포함 위반 **0** | 동일 |

등급 정정(감리-A 지적 ①): D4 는 계약 §7 이 **보조 관측(비게이트, 판정에 기재만)** 으로
분류하고 **PB-A 판정식에도 없다** — 판독기도 그렇게 구현돼 있다(`check_d4` 는 어떤 FAIL 도
던지지 않고 라벨만 부여: `D4_ALL_MATCH` 등, 판정자 직독). 초판의 «게이트+관측» 표기는 보수
방향 격상이라 결과 무영향(D4_ALL_MATCH)이나 계약 등급 문면과 달라 계약대로 고쳤다.

6개 lump 값의 hex 전건이 §4-1 사전 실측과 bit-일치한다 [부록 B 출력: Fe₀
`0x1.d1534dba20c26p-12`(= Z-O E-Z3 앵커 `n_upper`) · Fe₁ `0x1.daa141c1395e6p-12` · Co₀
`0x1.9a2c1ea029ae1p-9` · Co₁ `0x1.a015e376414d4p-9` · Ni₀ `0x1.3799ea75acadep-10` · Ni₁
`0x1.33f5bf5d863f1p-10`].

**검수 지적 ② 처분 — D3 부호 있는 Z별 이동값 보충** (라벨은 절대값 통계만 인쇄한다;
§4-1 대로 판정문이 보충, 판정자 독립 재계산 [부록 B]):

| Z | rel = n₁/n₀ − 1 (부호 포함) | 다중도 |
|---|---|---|
| 26 (Fe) | **+1.999569e-02** | 106,502 |
| 27 (Co) | **+1.441628e-02** | 105,685 |
| 28 (Ni) | **−1.168571e-02** (음수 — §4-1 유의 그대로) | 9,195 |

### 2-1. ★D4 의 의미 — 표본 조성 실측이 말하는 것

D4 표본 1,586행의 조성을 판정자가 실측했다 [부록 B-2]:

- **hazard 안(τ=0·validity=2): 154행** · **hazard 밖(τ_raw≠0·validity=1): 1,432행**.
- hazard 밖 1,432행 중 **g_l ≠ g_u 인 행 1,061** — 예: line 484784 (Co, lower_level=7
  [명시 준위], upper_level=356 [lumped], g 3↔1, τ_raw=14.23).
- 이 1,432행 전건에서 `n_upper` 가 해당 (Z,iter) 의 lump 값과 **bit-일치**했다.

의미: 동일성은 **hazard 쌍의 성질이 아니라 lumped 집합 전체의 성질**이다. τ≠0 이고 등g 도
아닌 선의 상단 준위조차 — lumped 이기만 하면 — 같은 한 값을 갖는다. H-COPY(쌍-단위 복사:
hazard 그래프 밖에서는 동일성이 없어야 함)는 이 실측으로 수치 수준에서 죽는다. hazard =
{lumped 전원 단일값} ∩ {g_l=g_u} ∩ 창 이라는 §3-1 H-FLAT 의 그림 그대로다.

### 2-2. ★D5 포함성 — 예측 > 관측이 정합인 이유

P_deck 은 **창-무관 덱 전수** 예측(ion=1 선 중 양 준위 lumped ∧ 등g)이고, 관측 221,382 는
**뷰 창 안** 스캔이다(Z-O §5-2 ② 스코프 승계). 따라서 정합의 방향은 관측 ⊆ 예측이며 —
위반 0 [부록 B] — 초과분(Fe 46,319 · Co 53,379 · Ni 4,409)은 창 밖 잔여로 **보고만** 한다
(§4 D5 문면: 창 재현은 이 단 밖). 보조 관측: 예측은 Co(159,064) > Fe(152,821)인데 관측은
Fe > Co — 창 절사의 소산이라는 해석이 자연스러우나 이 단은 측정하지 않았다 [추정 아님 —
미측정 기재].

---

## 3. 게이트 재현 — G-B1·G-B2·G-B3 (전건 판정자 재실행)

| 게이트 | 판정 | 판정자 재현 실측 |
|---|---|---|
| **G-B1** selftest | **PASS** | rc=0 · `controls=NC-K0..NC-K6 failed=0` [부록 A-5]. 음성대조 7종 전건 이름 있는 FAIL 시연 + 복원 PASS |
| **G-B2** 본실행 | **PASS** | 판정자 재실행 rc=0, 결과 라벨이 운전석 산출 파일(`bitlock_result_20260826T161506Z.txt`)과 **자면 동일** [부록 A-6]. 봉인 지문: 이 판정이 의존하는 5파일(stderr·stdout·levels·line_list·exports) `sha256sum -c` 전건 OK [부록 A-1]; 136파일 전건 전·후 통과는 운전석 집행 기재의 승계 |
| **G-B3** 인용 실측 | **PASS (구멍 1건 기재)** | rc=0 · `citations=55 hunk_range=13904-15392` [부록 A-5] |

**판독기 동작 특성 2건 (관측 기재, 비차단)**:

1. **검수 지적 ① 처분 — G-B3 커버리지 구멍**: 사전등록 §2-3 ③ 의 `super_mode` 소비처
   3곳(:16350·:16484·:19642) 중 `lumina_plasma.c:19642` 가 판독기 CITATIONS(55건)에 없다
   — 판정자가 직접 확인했다: CITATIONS 전문 직독에 19642 부재 + `:19642` 실물은
   `nlte->super_mode,` (f-빌더 `nlte_precompute_within_sl_frac_checked` 의 인자,
   :19637–19646 판정자 직독). 존재-검사 보조 게이트의 구멍이므로 비차단 — 처분 = 조용한
   대장 기재(판독기 보강은 수리 계급, 이 단 밖).
2. **PB-B/PB-C 라벨의 실질 도달 불능**: 판독기는 D3 Z당 1 파탄을 이름 있는 FAIL
   (`D3_RELATIVE_MOVEMENT_MISMATCH` → branch=PB-X)로 fail-closed 한다(판독기
   `check_d2_d3` 직독). dmax>1 인 실데이터는 일반적으로 D3 도 깨지므로 판독기 라벨로
   PB-B/PB-C 에 도달하는 경로는 사실상 없다 — G-B1 NC-K2 가 이를 실연했다(변조 주입 →
   `dmax=2` + `branch=PB-X`). §7 PB-X 문면("D3 정합 파탄")과 정합하므로 결함이 아니라
   계급 차이의 기재다(라벨의 branch 는 수치 보고, 분기 배정은 §7 문면이 정본).

---

## 4. 소스 감사 A1–A4 — 판정자 전건 재확인·집행

### 4-1. A2: §2-3 사슬 ①–⑧ 행 단위 재확증 — **전건 일치** [부록 A-4]

| 항 | 재확증 실측 (HEAD `a849306` 직독) |
|---|---|
| ① 무조건 덮어쓰기 | `src/lumina_atomic.c:1468` `getenv("LUMINA_SUPER_CUTOFF")` · `:1470` `if (K > 0)` · `:1473-1474` `level_super[l] = (level_num[l]>=K ? K : level_num[l])` · `:1476-1478` 배너(활성화 조건 검사 없음 — 권고 문구만) |
| ② 레이아웃은 덱에서 | `src/lumina_plasma.c:16344-16346` super_offset 누적(max_super+1) · `:16383` `g = nlte_ion_level_offset[i] + cursor[i]++` · `:16384` `sl = super_offset + atom->level_super[l]` · `:16387` `fl_to_super[g] = sl` · `:16350-16351` `super_mode = super_requested && n_super_total < n_nlte_levels_total` · `:16456-16457` `super_requested = (LUMINA_SUPER_LEVELS != 0) || element_wide` |
| ③ super_mode 는 f 만 게이트 | 소비처 grep 전수(lumina_plasma.c 내): **:16350(설정)·:16484(배너)·:19642(빌드 인자) — 3곳뿐** [판정자 grep]. `population_contract.c:172` 빌더 · `:199-200` `if (!super_mode) work[i] = 1.0` ⟹ **within_sl_frac ≡ 1.0** · super_mode=1 분기는 에너지 포함 Boltzmann(`:248` `g[global_level] * exp(-relative_energy/...)` — H-STAT 의 소스 근거 부재 재확인) · 의무 호출 `:19831-19837`(실패 시 abort) |
| ④ solve 차원 = SL | `:19025` `nlte_solve_ion_shell` · `:19042-19043` `N = super_offset 차` / `N_fl` 별도 |
| ⑤ 재분배 | `:19386-19388` 주석 *"Identity mode: SL(FL)==FL nlte idx and f_FL==1, so xfl == b."* · `:19397-19398` `f = within_sl_frac[..]; xfl[i] = b[sl] * f` · 기록행 `:19418`·`:19422`(per-ion scale 분기)·`:19433`(공통 scale 분기) — scale 은 이온/쌍 총량 정규화(균일 곱, 비 보존) |
| ⑥ 헤더 계약 | `src/lumina.h:723-724` *"When super_mode==0 / every map is identity and the FL solve path is byte-identical to baseline"* · `:725` 필드 — ①에 의해 위반 가능(이 런에서 위반됨: §5) |
| ⑦ 후보 레인 공유 | `nlte_population_candidate.c:226` `candidate->nlte = *public_nlte` · `:267-268` 인구·f 버퍼만 사유화 · `:385-391` 같은 빌더에 `candidate->nlte.super_mode` 전달 · `lumina_plasma.c:20641` `nlte_solve_all_impl(&candidate->nlte, …)` · `:20043` per-pair `nlte_solve_ion_shell` 호출 |
| ⑧ GPU 솔버 비생산 | `src/lumina_cuda.cu:1049` `nlte_solve_all_gpu` · `:1074-1078` 진입 1회 마커 `[NLTE-GPU][FAIL-CLOSED-ROUTE]` · `:1080-1081` CPU `nlte_solve_all` 위임 · `:1083` `#if 0 /* Forensic-only legacy … */` — **판정자 실측: 이 `#if 0` 는 :2063 의 `#endif` 까지 닫히지 않는다** [부록 A-4] ⟹ cuda:1083–2063 전 구간 비컴파일 |

봉인 구성 C1–C6 도 전건 판정자가 재실측해 일치했다 [부록 A-3]: `LUMINA_SUPER_CUTOFF="100"`
단독(SUPER/ELEMENT_WIDE/EW grep 매치 1행뿐) · stdout:129 K=100 배너 · stdout:236 자기모순
배너(`off (identity) (13015 FL -> 1173 SL)`) · stderr FAIL-CLOSED-ROUTE 0건 · stderr:419
SEED-MATERIAL `population_generation=1->2`.

### 4-2. A1: `nlte_level_populations` 기록자 전수 census — **판정자 독자 수행**

`grep -n "nlte_level_populations" src/*.c src/*.cu` → **131 참조** [부록 A-4]. 전 매치를
좌변(값 기록)/우변(판독)/포인터·수명 관리로 분류했다. **값-기록자 전수**:

| 기록자 | 계급 | 이 런에서의 판정 |
|---|---|---|
| `lumina_plasma.c:19418`·`:19422`·`:19433` | **재분배 커밋 (⑤)** | **유일한 값-생산자.** f≡1.0 + 이온당 단일 lump SL ⟹ 이온·셸당 한 값 |
| `lumina_plasma.c:21275`·`:21383` | 후보 출판 커밋 (`memcpy(public ← candidate->level_population)`) | 값 이동(transport) — 이미 평탄한 값을 공표 |
| `lumina_plasma.c:20144`·`:20155` | 저장 블록 복원(save/restore 짝: 저장 :20017·:20023) | 기존 상태 복원 — 평탄화 불능 |
| `lumina_plasma.c:20169`·`:20182-20183` | CE 감쇠 혼합 `n = old + d·(new−old)` | 두 커밋 상태의 아핀 혼합 — 평탄 생성 불능(보존만). 역으로: iter1 값이 iter0 과 다른 평탄값이므로 새 solve 산출 자체가 평탄했다는 방증 |
| `lumina_plasma.c:2035`·`:2111` | zinert 정확-0 기록(비활성 원소 전용, `lumina_zinert_Z_inactive_or_absent` 게이트, 기록값 0.0) | hazard 행은 `n_upper>0` — 생산자 불능 |
| `lumina_element_wide.c:846-847`(뷰 복사)·`:2193-2220`(save/restore)·`:2473`(`x[local]*frac` 재분배) | element-wide 레인 | **이 런 비가동**: 마스터 노브 `LUMINA_NLTE_ELEMENT_WIDE`(element_wide.c:112) exports 부재(C2) + 봉인 stdout EW 마커 0건 [판정자 grep]. 가동됐어도 :2473 은 같은 `x·frac` 기전의 둘째 사례(별개 가설 아님) |
| `lumina_cuda.cu:10159`·`:10195`·`:10810-10812`·`:10845` | GPU 라우트 브래킷(save/perturb/restore — 전건 `nlte_solve_all_gpu` 호출부 :10143/:10168/:10801/:10990 블록 안) | **이 런 미실행**: 라우터 진입 1회 마커가 0건(C5) ⟹ 호출 0 |
| `lumina_cuda.cu:1585`(시드)·`:1601/:1603/:1611`(scale)·`:1614`(0.0)·`:1822/:1824/:1833`(xfl 기록)·`:1853`(복원)·`:1867-1869`(감쇠) | **`#if 0` 死코드** (cuda:1083–2063, 판정자 실측) | **비컴파일** — 어느 빌드에서도 실행 불능 |
| 포인터·수명(값 기록 아님): `lumina_plasma.c:16366`(calloc)·`:16418/:16425/:16613`(free)·`:19800/:20272/:20398`(트랜잭션 포인터 스왑/복원)·`:19808`(abort 매크로 내 포인터 복원)·`nlte_population_candidate.c:267`(사유 버퍼 지정)·`lumina_element_wide.c:813/:829` | — | — |

**⟹ A1 폐합: 재분배(⑤)·후보 커밋 계열 밖의 쌍-단위 복사(H-COPY 형)·이온-계급 상수
강제(H-CLAMP 형) 기록자 = 0.**

**A1 방법의 별칭 사각과 그 폐합 (감리-A 지적 ⑤ — 결론 불변)**: 계약 A1 의 등록 방법은
**식별자 grep** 이라 별칭 포인터 경유 기록(트랜잭션의 `work_level` 버퍼·`t->public_level`)은
그 검사 우주 밖이다. 감리-A 가 추적해 폐합했고 판정자가 행을 직독 재확인했다:
`population_contract.c:844`(transaction begin — `work_level=pop_copy(l,nl)` 전량 복제)·
`:848`(commit — `memcpy(t->public_level, t->work_level, …)` 전량 이송)·
`candidate->level_population` 은 `nlte_population_candidate.c:211` copy_values 생성 후 `:267`
별칭(`candidate->nlte.nlte_level_populations = candidate->level_population`) 경유로만 기록 —
전건 **원소별 논리 없는 전량 이송** ⟹ 계열 밖 기록자 0 결론 불변. 권고: 차기 사전등록의
A1 방법에 alias-폐합 한 줄 추가(사전등록 §11-2 에 기재됨).

**H-CLAMP 보강 소거 (판정자 추가 실측)** — 기록자 부재에 더해 세 겹:

1. 클램프류 노브는 **금지 레지스트리**에 있다: `nlte_forbidden_numeric_repairs[]`
   (`lumina_plasma.c:10522-10536` — `LUMINA_NLTE_LTE_FLOOR`·`FLOOR_MODE`·`FLOOR_REG`·
   `FLOOR_BKMAX`·`BK_CEIL`·`INV_CEIL`·`COLL_FLOOR`·`DR_FLOOR_CMS`·`STAGE4_BK_CAP`·
   `HRESP_CLAMP`·`J_CAP_FACTOR`·`J_FLOOR_FACTOR` 등), 검사 함수
   `nlte_reject_numeric_repairs`(:10538-)는 어느 하나라도 활성이면 `[NUMERIC-REPAIR][BLOCKED]
   … action=TERMINATE` 로 종료한다.
2. 봉인 exports 실측: 위 노브 전건 `"0"` 또는 부재 [부록 A-3] · 봉인 stderr
   `NUMERIC-REPAIR` 마커 **0건** · 런 완주(rc=0, Z-O E-Z1 승계) ⟹ 클램프류 수치수리는 이
   런에서 **기계적으로 배제**돼 있었다.
3. 라이브 solve 의 음수 처분은 클램프가 아니라 fail-closed 다:
   `lumina_plasma.c:19383` `return NLTE_ION_SOLVE_NEGATIVE_POPULATION`(:19340-19384 직독
   — 진단 기록 후 반환, 값 강제 없음).

### 4-3. A3: 계약 위반 특정 — env 2개 조합표 4칸 **[판정자 집행]**

A3 는 발주~게이트 단계에서 집행되지 않았다(집행 기록에 해당 항 없음). 소스 직독·봉인
실측만으로 완성 가능한 측정이므로 판정자가 집행한다(측정이지 수리 아님 — §4 A3 문면).

전제 실측: `level_super[]` 의 원천은 덱 `super_level` 열이다(`lumina_atomic.c:1442`; 열
부재 시 항등 초기화 `:1444-1446`). **이 덱의 원생 열 자체가 비항등**이다 — ion=1
Z∈{26,27,28} 준위 6,157행 중 **6,154행**이 `super_level ≠ level_number` [부록 B; 앵커 재확인:
(26,1,100)→super 26 · (26,1,2542)→super 128, 부록 A-7].

| | `LUMINA_SUPER_LEVELS` 부재/0 | `LUMINA_SUPER_LEVELS=1` |
|---|---|---|
| **`LUMINA_SUPER_CUTOFF` 부재/0** | level_super = 덱 열. 덱 열이 항등이면 계약 성립(항등 map + f≡1). **이 덱처럼 비항등이면 같은 위반 계급**(덱-원생 층위의 반상태) — 단 이 런의 유효 층위가 아님은 아래 층위 판별로 확정 | 정상 SL 모드(덱 층위): map 비항등 + super_mode=1 + f=Boltzmann — 헤더 계약(super_mode==0 조건부)의 범위 밖, 위반 아님 |
| **`LUMINA_SUPER_CUTOFF=K>0`** | ★**이 봉인 런 (위반 실물)**: 무조건 덮어쓰기(①)로 이온당 단일 lump + super_mode=0(②) + f≡1.0(③) ⟹ 반상태 | ARTIS 설계 의도(①의 배너 문구): lump 층위 + Boltzmann 재분배 — 위반 아님 |

**위반의 최소 재현 조건 = «level_super 비항등 ∧ super_mode==0»**. `LUMINA_SUPER_CUTOFF`
단독 설정은 그 충분조건의 실측 사례이고(이 런), 덱 열 비항등 + 두 노브 전부 부재도 같은
계급에 든다(좌상단 칸 — 이 런 밖, 존부는 런별 exports·덱의 문제로 **미결**).

**★층위 판별 (판정자 실측 — 이 런의 유효 층위가 K-덮어쓰기임의 수치 증명)**: 덱-원생
층위였다면 lumped(level_num≥100) 준위들이 이온당 **Fe 105 · Co 104 · Ni 40 개의 서로 다른
원생 SL 계급**으로 갈라져 있다 [부록 B — 예: 앵커 쌍은 원생 super 26 vs 128 로 다른 계급].
관측은 (Z,iter)당 고유값 **1**(D2) — 원생 계급 경계를 횡단하는 단일값이다. 이는 이온당
단일 lump(전 level_num≥100 → SL K)를 만드는 K-덮어쓰기 층위만이 준다. C3 배너(21,919
lumped — 덱 전수 level_number≥100 행수와 정확 일치 [부록 A-7])와 정합.

**C4 산술 재도출 (검수 §8-1 ③ 의 첫 항 해소)**: 13,015 = 봉인 stdout:204-235 의 NLTE 이온
인벤토리 합(Si II/III 125+147 · Ca II/III 62+232 · S II/III 322+256 · Fe II/III 2599+1500 ·
Co II/III 2558+3214 · Ni II/III 1000+1000) [판정자 산술]. 1,173 = **62(Ca II — 준위
62<100, 유일하게 미절단) + 11개 이온 × 101(명시 100 + lump 1)** [판정자 산술 — 정확 일치].

### 4-4. A4: P2/P3 단일 생산 구현 — 호출 사슬 도식 **[판정자 집행]**

A4 도 발주~게이트 단계에서 집행되지 않았다. 판정자가 각 화살표의 행을 직독으로 확증했다:

```
P2 (시드):  lumina_cmfgen.c:7525 lumina_init_seed_material_predictor 호출
              → lumina_plasma.c:16060 (정의)
              → :16202-16203 nlte_population_candidate_commit_seed_material
P3 (반복):  A2-07 트랜잭션 → 후보 번들
공통:       nlte_population_candidate.c:226 (public 사본) / :267-268 (버퍼 사유화)
              → lumina_plasma.c:20641 nlte_solve_all_impl(&candidate->nlte, …)
              → :20043 nlte_solve_ion_shell(…)
              → :19386-19434 재분배 (xfl[i]=b[sl]·f, 기록 :19418/:19422/:19433)
              → f 는 매 solve 전 의무 재계산 :19831-19837 → :19633-19656
                 (super_mode 인자 :19642) → population_contract.c:199-200 (f≡1.0)
출판:       lumina_plasma.c:21275 / :21383 (memcpy public ← candidate)
```

⟹ 시드(P2)와 반복(P3)이 **같은 재분배를 지난다** — iter0 값과 iter1 값이 다르면서 둘 다
평탄(D2·D3)인 관측과 정합: 평탄화는 화석이 아니라 매 solve 산출의 성질이다(관측 범위
iter0→iter1, Z-O §6-3 하향 문언 승계).

### 4-5. ★발주서(운전석 축 2)의 인용 정정 — 귀속 실체 불변, 자리 정정

운전석이 판정자에게 넘긴 기전 사슬은 마지막 두 링크와 「결함의 정확한 자리」를
**`lumina_cuda.cu:1657`(불변식 주석)·`:1795`(xfl)·`:1822`(기록)** 로 지목했다. 판정자
실측: 이 세 행은 전부 **`#if 0`(cuda:1083–2063) 死코드** 안에 있다 — 어느 빌드에도
컴파일되지 않는 레거시 트윈이다(⑧의 "Forensic-only" 블록 그 자체). 무죄 목록의
cuda:1585·:1614·:1853·:1869 도 같은 死블록이라 "무죄"가 아니라 **비존재**다.

**실물 기전의 자리** = 라이브 CPU 트랜잭션 경로: 위반된 불변식 문면은
`src/lumina.h:723-724`(계약)과 `src/lumina_plasma.c:19386-19388`(재분배 주석 — *"Identity
mode: SL(FL)==FL nlte idx and f_FL==1, so xfl == b"*)이고, 기록자는 `:19398` +
`:19418/:19422/:19433` 이다. `f_FL==1` 은 참이 됐고(③) `SL(FL)==FL` 이 거짓이 됐다(①②)
— 발주서가 cuda 주석으로 말한 논지 그대로이며, **사전등록 §2-3 ⑤ 는 처음부터 plasma 행을
지목했으므로 계약과 실물은 어긋난 적이 없다.** 어긋난 것은 발주서의 재서술이다(보조 정정:
축 2 의 `atomic.c:1468 if (K > 0)` 도 :1468 은 getenv 행, `if (K > 0)` 는 :1470). 귀속
결론에는 영향 없음 — §9 에 특기.

---

## 5. §7 판정식 결합 — 분기 선언

§7 (r1) PB-A = «D1 전건 ∧ dmax=1 ∧ D3 Z당 1 ∧ D5 포함 위반 0 ∧ G-B 전건 PASS ∧ **A1 이
계열 밖 기록자 0 확인(H-COPY·H-CLAMP 형 부재) ∧ A2–A4 폐합 조건 전건 충족**». 판독기
라벨(branch=PB-A, layer=NUMERIC_ONLY)은 앞 다섯 항의 보고이고, 결합은 판정자 몫이다.

| 항 | 충족 | 근거 |
|---|---|---|
| D1–D3·D5 (판정식의 수치 항 전건) | ✔ | §2 (판정자 재계산 + 판독기 일치). **D4 는 판정식 밖 보조 관측**(계약 §7) — D4_ALL_MATCH 는 §2-1 기재만 |
| G-B1·G-B2·G-B3 | ✔ | §3 (판정자 재현) |
| A1 (계열 밖 기록자 0) | ✔ | §4-2 (판정자 전수 census + 금지노브 3겹) |
| A2 (사슬 재확증) | ✔ | §4-1 (전건 행 단위) |
| A3 (4칸 표) | ✔ | §4-3 (판정자 집행) |
| A4 (사슬 도식) | ✔ | §4-4 (판정자 집행) |

### ⟹ **판정 branch = PB-A. H-FLAT 확정 — 귀속 종결.**

**귀속 문언**: `n_l ≡ n_u` (bit-동일)의 생산자 =
**«`LUMINA_SUPER_CUTOFF` 단독 설정의 반(半)상태»** —
(i) 덱 로더의 무조건 K-덮어쓰기(`lumina_atomic.c:1468-1478`)가 이온당 «명시 ≤100 + lump 1»
레이아웃을 만들고, (ii) `LUMINA_SUPER_LEVELS` 부재로 `super_mode=0`
(`lumina_plasma.c:16350-16351`·`:16456-16457`) ⟹ `within_sl_frac ≡ 1.0`
(`population_contract.c:199-200`), (iii) solve 는 SL 차원(`:19042`)이므로 재분배
`xfl[i] = b[sl]·1`(`:19398`)이 이온의 lumped 준위 **전원**에 `b[lump]` 한 값을 주고, 균일
총량 정규화(`:19418/:19422/:19433`)와 후보 커밋(`:21275/:21383`)이 그 평탄 구조를 보존
출판한다. hazard 221,382쌍은 그 평탄 집합 위에서 등g(전건 — Z-O §6-1 덱 전수 실측 승계)인
선쌍이 출판 대수 `n_l − (g_l/g_u)·n_u = 0` 으로 τ 정확-0 이 된 **표본추출**이다. 위반된
계약 문면 = `lumina.h:723-724`.

### 5-1. 실측/추론 경계 — 문장 단위

- **[실측 — 판정자 직접]** D1–D5 전 수치 · G-B 3게이트 · C1–C6 · A1 census 131참조 분류 ·
  A2 사슬 전 행 · A3 덱 열 비항등(6,154행)·층위 판별(원생 SL 계급 105/104/40 대 관측 d=1) ·
  A4 사슬 행 · 死코드 경계(1083–2063) · 금지노브 exports·마커 · D4 조성(154/1,432/1,061) ·
  C3=21,919·C4=13,015/1,173 산술 · TSV sha256 · 봉인 지문 5파일.
- **[실측 — 승계 (이 단이 재도출하지 않음)]** hazard 전건 등g · `A208_EXACT_ZERO` 전건 ·
  τ 기록자 전수·언더플로 배제 — Z-O 판정(2겹 감리 통과)의 §6 사슬.
- **[추론 — 승계 사슬과 실측의 결합]** «출판 시점 n_lower ≡ n_upper (bit-동일)» 자체.
  `n_lower` 는 어떤 산출물에도 인쇄되지 않는다 — 이 명제는 출판 대수 경유다(Z-O §6-3;
  검수 §8-1 ① 그대로: **그 사슬이 틀렸다면 이 단의 전제가 흔들린다** — 잔존 한정).
- **[확정 — 위 전제 하에서]** 생산 기전 = H-FLAT. "확정"의 실질: ⓐ 살아 있던 대안 전건
  소거 — H-STAT(D2 가 g-계급 수 아닌 1 + 소스 근거 부재)·H-COPY(D4 조성 — hazard 밖
  1,432행 일치)·H-CLAMP(A1 기록자 0 + 금지노브 3겹)·H-BOLTZ/H-DEG/H-PUBZERO/H-SEED/
  H-MATSYM(구성·덱·승계·D2 소거, 사전등록 §3 그대로) — ⓑ 지목 기전이 **by-construction
  충분**함의 소스 확인(f≡1 + 단일 lump ⟹ 단일값은 대수적 필연) — ⓒ 층위 지문 일치
  (관측 granularity = K-덮어쓰기 층위, 덱-원생 층위 기각).
- **[미측정 — 주장하지 않음]** 창 밖·타 셸·타 위상·타 이온의 lockstep 존부 · LTE 시드 뷰
  쪽 동형 현상 · 등g 아닌 lumped 쌍의 bit-동일(D4 표본 1,432행에서 실측됐으나 **전수**는
  아님 — 창·선정 밖 미측정).

---

## 6. 함의 — 어디까지 적을 수 있는가 (§8 경계 준수 판정)

이 발견의 문면상 함의는 «이 구성에서 커버 이온의 lumped 준위(level_num≥100)는 이온·셸당
한 값 — 그 구간의 NLTE 준위 구조가 사실상 부재»다. **기전의 사정거리는 커버 3이온보다
넓다(감리-A 지적 ④)**: by-construction 기전(①②③⑤)은 K=100 lump 를 갖는 **NLTE 인벤토리
11개 이온 전부**(Si II/III · Ca III · S II/III · Fe II/III · Co II/III · Ni II/III — Ca II 만
62준위로 유일하게 미절단)에 미친다 — 실측 근거는 C4 산술 **13,015 FL · 1,173 SL = 62 +
11×101** 이다(§4-3; 감리-A 재계산 일치). 다만 **측정된 것은 커버 3이온(Z∈{26,27,28}
ion=1)뿐**이므로 주장은 그 범위로 절제한다. §8 이 긋는 선 안에서 적을 수 있는 것과 없는 것:

1. **적는다 [실측]**: 봉인 L6(`sprim_l6_20260821T054111Z_probe`)·L6C
   (`l6c_20260822T154507Z_cover`)의 `input/resolved_lumina.exports` 에도
   `LUMINA_SUPER_CUTOFF="100"` 이 실재한다 [판정자 grep, 부록 A-8] — 같은 반상태가 그
   봉인 런들에도 있었다. **§8-3 문면 간극 기재(감리-A 지적 ③)**: 이 실측(L6·L6C exports
   읽기)은 계약 §8-3 괄호문(«이 단의 입력은 Z-1r census 봉인뿐»)의 문면 밖이다 — 의도
   (Z-2A 격리·src 무접촉)는 지켰고, 이 실측은 귀속 판정 논리곱(§5)에 쓰이지 않고 본 절
   함의·§8 미결 인계에만 쓰였다(Z-O 판정문 선례). 감리-A 가 실측을 재확인했다(두 exports
   공히 `LUMINA_SUPER_CUTOFF="100"`).
2. **적는다 [사실 관계 — 측정 아님]**: 따라서 그 계열 런에서 lumped 준위를 읽는 모든
   관측(예: 향후 L6C 형 «solve 의 J̄ 응답» 계측이 lumped 준위를 표적으로 삼는 경우)은 이
   반상태의 평탄 구조를 읽게 된다 — 잣대 오염의 소지가 **구조로서** 존재한다.
3. **넘지 않는다 [미결 인계]**: 어느 단의 어느 판정·관측이 실제로 lumped 준위 구조에
   의존했는지의 **전수 재감사는 이 단 밖**이다(측정 없이 단정하면 과대주장 — 예컨대 L6 의
   «준위비 3 ulp 동결» 관측은 표적이 ion IV·NLTE 테이블 밖이었으므로[8526d2c 귀속] 이
   기전과의 관계 자체가 별도 검증 대상이다). §8 미결 #3 으로 인계한다.
4. **하지 않는다 (§8-1·§8-2·§8-3)**: 수리 제안 없음(노브를 켜라/빼라/로더를 고쳐라 —
   전부 별도 단, V5 권한·bit-앵커 계보 단절 검토 필요). 본선 물리 계약 측정 없음. Z-2A
   R-II 산출은 읽지 않았다.

처분 = **조용한 대장 기재**(Z-O §6-4 의 기재를 본 판정이 귀속 완결로 갱신).

---

## 7. 선행 단 승계 판정 — Z-O §9 #5 는 닫혔는가

**닫혔다.** Z-O 판정 §9 #5(«bit-동일 구조적 생산자 귀속 — 입력 완비, 별도 단 설계 필요,
런 불요 가능성»)가 요구한 그 단이 본 단이고, 예견("solve 내부 소스 감사 예상 — 런 불요
가능성")대로 **런 0회**로 귀속이 종결됐다. Z-O 가 인계한 입력 3종이 전부 소비됐다:
준위쌍 목록(D1·D5 의 입력, sha256 일치) · 한 세대 보존 실측(D2·D3 의 판별 대상) · 전건
등g 관측(귀속 문언의 표본추출 항).

**§9 승계 기재 2건의 사용처** (행 전건 판정자 재확인 [부록 A-4]):

- **① global→nlte 단사성** (`:16383` 단조 커서 · `:8925-8927` 소비·fail-closed): 별칭
  배제가 있어야 «두 원소의 값 일치»가 실명제로 성립한다 — H-FLAT 은 그 일치를 **한 SL 해
  `b[lump]` 의 서로 다른 슬롯 사본**으로 설명하며 단사성과 모순 없다. 본 판정 §5 귀속
  문언의 전제로 사용됐다.
- **② K-FRESH 사슬** (`:6893-6931` 정의 · `:19471/:19607` 재출판 호출쌍 ·
  `:20792-20799`·`:20877-`·`:21175-21178` preflight): D2–D4 가 읽은 ROW `n_upper` 와
  판정된 τ 가 **같은 세대의 출판값**이라는 전제가 이 사슬로 선다 — §2 판별 전체의 전제로
  사용됐다.

---

## 8. 남은 미결과 다음 표적

| # | 미결 | 성격 |
|---|---|---|
| 1 | **본선 물리 계약 — A2-07 solve 의 J̄ 응답** (L6C 재발주 F1/F1b 몫) | 이 단은 측정하지 않았다(§8-2). Z-2A 착지 후 |
| 2 | Z-2A 구현·착지 · Z3/Z4 게이트 | Z-O §9 #1–#3 그대로 (이 단 무접촉) |
| 3 | **★반상태의 전제 재감사** — `LUMINA_SUPER_CUTOFF` 가 실렸던 런 전수와, lumped 준위 구조를 관측 잣대로 쓴 단의 전수 (§6-3). ★단 개설 시 대상은 커버 3이온이 아니라 **K=100 lump 를 갖는 NLTE 인벤토리 11이온 전부**다(§6 — 기전의 사정거리, 감리-A 지적 ④) | 신규 — 본 판정이 연 미결 |
| 4 | 반상태의 수리 단 개설 여부 (`SUPER_LEVELS` 활성/컷오프 제거/로더 계약 회복 — 셋 다 이 단 밖) | user 결정 대기 — V5·bit-앵커 계보 검토 필요(§8-1) |
| 5 | `n_lower` 비인쇄 — «n_l≡n_u» 의 추론 의존 잔존 (검수 §8-1 ①) | 승계 한정 유지. 직접 인쇄 계측은 수리 계급 — 제안 아님, 사실 기재 |
| 6 | 창 밖·타 셸·타 위상·등g 아닌 쌍의 lockstep 전수 | 미측정 유지 (§8-4) |
| 7 | 좌상단 칸(노브 2개 전부 부재 + 비항등 덱 열)의 실런 존부 | §4-3 이 연 미결 — #3 에 흡수 가능 |
| 8 | 판독기 CITATIONS 의 :19642 보강 · PB-B/C 라벨 도달성 문서화 (§3) | 비차단 — 다음 판독기 개정 시 |

---

## 9. 분장 장부 지목과 특기 (개정16 ①)

장부 정본 = **사전등록 §11**. 본 판정문은 판정 칸만 기입한다:
**판정 = Fable fresh, 2026-08-27, 본 문서.** 특기 2건(위반 아님·표류 기재):

1. **A3·A4 는 발주~게이트 단계에서 집행 주체가 비어 있었다** — §11 장부에 A-감사 행이
   없고, 운전석 집행분은 A1·A2 뿐이었다. 판정자가 §4-3·§4-4 로 집행했다(소스 직독 계급
   — 판정자 재확인 권한 안). 사전등록이 A-감사의 실행 주체를 지정하지 않은 것은 개정16 ③
   («저자가 실행 가능한 것은 저자가 실행») 관점의 표류 소지 — 대장 기재.
   **감리-A 판단(지적 ②)**: 이 자기집행은 **위반이 아니다** — 개정14 의 «집행=운전석»은
   런 발주·제출·계측 계급을 겨냥하고, 소스 직독·read-only 덱 판독은 Fable 계급이다.
   판정문이 미리 밝혔으므로 은폐도 없다. 자기채점 잔여는 감리-A 가 **A3 결정 실측을 전건
   독립 재현**해 해소했다(덱 원생 열 비항등 6,154/6,157 · 원생 SL 계급 {105/104/40} ·
   앵커 (100→super 26, 2542→super 128) · `deck_levels_ge100=21919`). **실제 결함은 사전등록
   §11 이 A-감사 집행 주체 행을 아예 두지 않은 공백**이다 — 차기 사전등록 개정 항목으로
   인계(사전등록 §11-2 에 기재됨).
2. **발주서(운전석 축 2)의 기전 자리 인용이 死코드였다** (§4-5) — cuda:1657/:1795/:1822 는
   `#if 0` 비컴파일 구간. 사전등록(§2-3 ⑤)은 처음부터 라이브 행을 지목했으므로 계약 무결
   — 발주서 재서술의 표류이며, 개정15 «뒷겹 재서술 금지»가 겨냥한 바로 그 계급이다.
   귀속 결론 불변.

---

## 부록 A — 재현 명령 (전건 판정자 실행 확인, 2026-08-27; 읽기 전용)

`RR=/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`

```bash
# A-1 봉인 지문 (의존 5파일; grammar-debug)
ls $RR/ZO_RUN_SEAL.*            # -> .meta .sha .txt
wc -l < $RR/ZO_RUN_SEAL.sha     # -> 136
grep "git_head" $RR/ZO_RUN_SEAL.txt   # -> git_head=e500e2995814f5f4f4eaf82984e48063a4aaf571
# grammar-debug 에서 ($RR 로 cd 후):
grep -E " (input/model/levels\.csv|input/model/line_list\.csv|input/resolved_lumina\.exports|stderr\.log|stdout\.log)$" \
    ZO_RUN_SEAL.sha > /tmp/dep.sha && sha256sum -c /tmp/dep.sha
#  -> 5행 전건 OK, rc=0

# A-2 좌표계
git log -1 --format=%H                          # -> a8493065b4aa2408506cbd3afaeb0c9ffd41d663
git diff --stat 45d931d a849306 -- src/ | wc -l # -> 0 (사전등록 시점과 src 동일)
git diff --name-only e500e29 HEAD -- src/       # -> src/lumina_plasma.c
git diff --stat e500e29 HEAD -- src/ | tail -1  # ->  1 file changed, 94 insertions(+), 142 deletions(-)
git diff e500e29 HEAD -- src/lumina_plasma.c | grep "^@@" | head -3   # -> 전 hunk 13904.. 구간

# A-3 봉인 구성 C1-C6 + 금지노브
grep -E "SUPER|ELEMENT_WIDE|EW_" $RR/input/resolved_lumina.exports  # -> LUMINA_SUPER_CUTOFF="100" 한 줄
sed -n '129p;236p' $RR/stdout.log      # -> K=100 배너 / off (identity) (13015 FL -> 1173 SL)
grep -c "FAIL-CLOSED-ROUTE" $RR/stderr.log      # -> 0 (rc=1)
grep -m1 -n "SEED-MATERIAL" $RR/stderr.log      # -> 419: … population_generation=1->2 …
grep -E "FLOOR|CEIL|CAP|CLAMP" $RR/input/resolved_lumina.exports    # -> 전건 ="0" (14행)
grep -c "NUMERIC-REPAIR" $RR/stderr.log         # -> 0 (rc=1)
grep -c "FLOOR_BKMAX" $RR/input/resolved_lumina.exports             # -> 0 (rc=1, 부재=비활성)
grep -ci "element.wide\|\[EW\]" $RR/stdout.log  # -> 0 (rc=1)

# A-4 소스 감사 (발췌 — 전 항목 동일 요령 grep/sed 직독; §4 표에 행별 결과)
grep -n "nlte_level_populations" src/*.c src/*.cu | wc -l           # -> 131
awk 'NR>=1083 && NR<=2100 {if ($0 ~ /^[[:space:]]*#[[:space:]]*if/) {d++; print NR": "$0}
     else if ($0 ~ /^[[:space:]]*#[[:space:]]*endif/) {print NR": depth_before="d; d--}}' src/lumina_cuda.cu
#  -> 1083: #if 0 /* Forensic-only legacy … */ (외곽, depth 1)
#  -> 1331/1333 · 1361/1363: 중첩 #ifdef _OPENMP 쌍 (depth 2 에서 폐합)
#  -> 2063: depth_before=1 #endif — 외곽 #if 0 를 닫는다 ⟹ cuda:1083–2063 전 구간 비컴파일 (중첩 계정 완료)
sed -n '1466,1480p' src/lumina_atomic.c          # -> ① 덮어쓰기 블록 (getenv 1468 · if(K>0) 1470)
grep -n "level_super" src/lumina_atomic.c | head -4
#  -> 1442: 덱 super_level 열 로드 / 1444-1446: 열 부재 시 항등 초기화
sed -n '16344,16351p;16383,16387p;16456,16457p' src/lumina_plasma.c  # -> ② 사슬
grep -n "super_mode" src/lumina_plasma.c         # -> 16350·16484·19642 (전수)
sed -n '199,200p' src/population_contract.c      # -> if (!super_mode) work[i] = 1.0
sed -n '19386,19398p;19417,19434p' src/lumina_plasma.c   # -> ⑤ 재분배 (주석·xfl·기록 3행)
sed -n '723,725p' src/lumina.h                   # -> ⑥ 계약 문면(:723-724) + super_mode 필드(:725)
sed -n '226p;267,268p;385,391p' src/nlte_population_candidate.c      # -> ⑦
sed -n '20641,20645p;20043,20046p' src/lumina_plasma.c               # -> ⑦ 호출 사슬
sed -n '1049p;1074,1081p' src/lumina_cuda.cu     # -> ⑧ 라우터 마커·CPU 위임
sed -n '10522,10560p' src/lumina_plasma.c        # -> 금지-노브 레지스트리 + TERMINATE 검사
grep -n "NLTE_ION_SOLVE_NEGATIVE_POPULATION" src/lumina_plasma.c | head -2   # -> 19383 (fail-closed)
grep -n "LUMINA_NLTE_ELEMENT_WIDE" src/lumina_element_wide.c | head -1       # -> 112 (마스터 노브)
sed -n '6893,6911p' src/lumina_plasma.c          # -> K-FRESH 정의
grep -n 'tau_sobolev_require_refresh(opacity\|tau_sobolev_mark_computed(opacity' src/lumina_plasma.c
#  -> 3221/3357 · 19471/19607 · 21853/21916
sed -n '20792,20799p;21175,21178p;8925,8927p' src/lumina_plasma.c    # -> preflight·단사성

# A-5 게이트 G-B1·G-B3 (grammar-debug, repo 루트에서)
python3 scripts/analyze_bitlockstep_producer.py --selftest
#  -> A210_BITLOCKSTEP_PRODUCER_SELFTEST status=PASS controls=NC-K0,...,NC-K6 failed=0 / rc=0
#  주(감리-B NOTE #9): --selftest 는 쓰기 가능 임시 디렉토리가 필요해 **감리 read-only
#  환경에서는 재현 불능 — 쓰기 가능 환경 필요**. 운전석이 두 노드(로그인·grammar-debug)에서
#  대리 실행해 동일 라벨·rc=0 을 확인했고, 감리-B 는 쓰기 없는 **정적 오라클**로 교차
#  검증했다 — 픽스처 바이트를 재구성해 계산한 pair-list sha256 이 정확히
#  57aaa0e8a51c18a46777306273104d4f5c00c2cc8900852a6c427942513ee784 이고, 정상 픽스처의
#  dmax=1 · membership=5/5 · D3 삼중항 · D4 6/6 · Pdeck 2/2/1 이 코드상 도출됨을 확인.
python3 scripts/analyze_bitlockstep_producer.py --check-citations
#  -> A210_BITLOCKSTEP_PRODUCER_CITATION_RESULT status=PASS citations=55 hunk_range=13904-15392 / rc=0

# A-6 G-B2 본실행 재현 (grammar-debug, repo 루트; 봉인 read-only, 수 분)
python3 scripts/analyze_bitlockstep_producer.py
#  -> rc=0, 라벨이 /gpfs/kjhan/lumina/zo_artifacts/bitlock_result_20260826T161506Z.txt 와 자면 동일:
#     A210_BITLOCKSTEP_PRODUCER_RESULT status=PASS branch=PB-A layer=NUMERIC_ONLY dmax=1
#     membership=221382/221382 … d4_samples=1586 d4_match=1586 d4_mismatch=0
#     pdeck=Z=26:152821,Z=27:159064,Z=28:13604 pairlist_sha256=ac7f3ca6…f75f01 reasons=NONE

# A-7 덱 앵커·C3 산술
awk -F, '$1==26&&$2==1&&($3==100||$3==2542){print $1","$2","$3","$4","$5","$6","$7}' \
    $RR/input/model/levels.csv
#  -> 26,1,100,6.5194425068,8,0,26  /  26,1,2542,15.7617685743,8,0,128
awk -F, 'NR>1 && $3>=100 {n++} END{print "deck_levels_ge100="n}' $RR/input/model/levels.csv
#  -> deck_levels_ge100=21919  (= C3 배너의 lumped 총수)
sed -n '204p' $RR/stdout.log                     # -> [NLTE] Total NLTE levels: 13015

# A-8 타 봉인 런 exports (§6-1; 읽기만)
grep -h "SUPER" \
  /gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover/input/resolved_lumina.exports \
  /gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe/input/resolved_lumina.exports
#  -> declare -x LUMINA_SUPER_CUTOFF="100"  (두 줄 — 두 런 모두)
```

주(재현 환경): 판정자는 grammar-debug 실행을 nested ssh 로 수행했다 —
`ssh grammar "ssh grammar-debug bash" <<'EOF' … EOF` (repo 루트 `cd` 후 실행; `/tmp` 는
노드-로컬이므로 스크립트는 stdin 파이프로 전달).

## 부록 B — 판정자 독립 재계산 (grammar-debug; stdin 파이프 실행; 읽기 전용)

파서·해시·조인 전부 판독기와 무관한 자체 구현. 실행 출력(2026-08-27):

```
tsv_sha256=ac7f3ca6def9738acc452121e844af08c4aba80bc83c90c14eb7683db7f75f01 pairs=221382
deck_native: ion1 Z26-28 super_ne_levelnum = 6154  native_max_super = {26: 130, 27: 133, 28: 58}
             distinct_native_super_among_lumped = {26: 105, 27: 104, 28: 40}
D1: map_bad=0 membership_bad=0 energy_bad=0 total=221382
D5: pdeck = {26: 152821, 27: 159064, 28: 13604}  inclusion_violations = 0
D2 unique-hex per (iter,Z): {(0,26):1,(0,27):1,(0,28):1,(1,26):1,(1,27):1,(1,28):1}  dmax=1
rows per (iter,Z): 전건 {26:106502, 27:105685, 28:9195}
D3 Z=26: unique=1 signed=+1.999569e-02(x106502)
D3 Z=27: unique=1 signed=+1.441628e-02(x105685)
D3 Z=28: unique=1 signed=-1.168571e-02(x9195)
D3 abs: min=1.1686e-02 med=1.4416e-02 max=1.9996e-02
lump hex: Fe0 0x1.d1534dba20c26p-12 · Fe1 0x1.daa141c1395e6p-12 · Co0 0x1.9a2c1ea029ae1p-9
          Co1 0x1.a015e376414d4p-9 · Ni0 0x1.3799ea75acadep-10 · Ni1 0x1.33f5bf5d863f1p-10
D4: samples=1586 match=1586 mismatch=0  selected_rows_total iter0=1201 iter1=992
anchor(line 669992, iter0): tau_validity=2 · n_upper hex=0x1.d1534dba20c26p-12 (= Z-O E-Z3)
```

조성 실측(B-2, 별도 1패스 — §2-1):

```
D4 composition: samples=1586 in_hazard=154 out_of_hazard=1432 out_gneq=1061
                out_tau_nonzero=1432 tau_validity_hist={'1': 1432, '2': 154}
예: (iter0, line 484784, Z=27, lower_level=7, upper_level=356, g 3↔1, tau_raw=14.23…, validity=1)
    (iter0, line 803797, Z=27, 39↔336, g 9↔7, tau_raw=4.257…)  (iter0, line 366507, Z=26, 8↔1780, g 4↔2)
```

파스 규약: iter 경계 = `[A2-10][LINE-SATURATION-SUMMARY]`(phase=REQUESTED_TE·target_ion=1)
2줄(stderr 행 225081/449539 — Z-O 부록 승계); hazard 행 = `[LINE-SATURATION-ZERO-OPACITY-ROW]`;
선정 행 = `[LINE-SATURATION-ROW]` 중 shell=0·ion=1·Z∈{26,27,28}·upper_level≥100.
ROW printer 는 HEAD `lumina_plasma.c:14090-14101`·`:14333-14352`, 봉인 `e500e29` 의 동등
printer 는 `:14078-14088`·`:14376-14400` 이며, 변경 hunk 와 겹치므로 양 좌표를 구분한다
(감리-B NOTE #8(a) 수정 문면 — 초판의 «변경구간 밖이므로 HEAD=봉인 행 번호» 설명은 틀렸었다.
판정자가 HEAD·봉인 양 좌표를 전건 직독 재확인 — 필드 실물: hazard 는 n_upper %.17g,
선정은 n_upper + per-ion lower_level/upper_level).

---

## 한 줄 결론

**분기 PB-A — H-FLAT 확정(판정식 수치 항 D1–D3·D5 전건 + D4 보조 관측 일치 + A1–A4 논리곱, 전건 판정자 재확인·집행)** ·
**확정**: 221,382쌍의 bit-lockstep 은 `LUMINA_SUPER_CUTOFF=100` 단독 설정의 반상태 —
무조건 K-덮어쓰기 lump 레이아웃 × `super_mode=0` 의 `f≡1.0` × SL-차원 solve — 가 이온당
lumped 준위 전원에 한 값을 주는 구조의 등g 표본추출이며(발주서가 지목한 cuda 행은 `#if 0`
死코드 — 실물 기전은 `lumina_plasma.c:19386-19434`, 귀속 실체 불변), hazard 밖 선정 행
1,432건(등g 아님 1,061 포함)의 bit-일치가 H-COPY 를, 기록자 전수·금지노브 3겹이 H-CLAMP 를
소거했고, Z-O §9 #5 승계 미결은 **런 0회로 닫혔다** · **미결**: 본선 물리 계약(A2-07 의
J̄ 응답)은 재지 않았고, `n_l≡n_u` 의 n_lower 비인쇄 추론 의존은 승계 한정으로 남으며,
같은 반상태가 실린 L6·L6C 계열 런들의 전제 재감사와 수리 단 개설은 별도 몫이다.

---

## 부록 C — 감리 스탬프 (개정17 두 겹)

| 겹 | 담당·계열 | 결과 |
|---|---|---|
| **감리-A** (물리·주장) | Fable fresh — 판정과 별개 컨텍스트(컨텍스트 분리) | **통과** · 차단 0 · 지적 5(전건 비차단, 본 개정판에 반영) |
| **감리-B** (재현) | Codex read-only — 계열 분리(개정17) | **통과** · 차단 0(재판정에서 1건 해소) · NOTE 2 · 재현 11항목 중 **재현됨 10 · 불일치 1 · 재현불능 1(→NOTE 강등)** |

**감리-A 특기**: A1 을 표본이 아니라 **전수 재수행**했다(참조 131건 분류) · 死코드 정정을
자체 전처리기-깊이 오토마톤으로 독립 확인했다(`lumina_cuda.cu` 死블록 = 정확히 1083–2063) ·
A1 의 별칭 포인터 사각을 스스로 찾아 폐합했다(§4-2) · A3 결정 실측을 전건 독립 재현해
자기채점 잔여를 해소했다(§9 특기 1).

**감리-A 지적 5건의 반영 위치**:

| # | 지적 | 반영 |
|---|---|---|
| ① | D4 등급 표기 격상(계약 §7 = 보조 관측·비게이트, PB-A 판정식 밖; 보수 방향이라 결과 무영향) | §2 표 등급 + 표 아래 주 + §5 표 행 + 한 줄 결론 |
| ② | 사전등록 §11 의 A-감사 집행 주체 행 공백(A3·A4 자기집행 자체는 위반 아님) | §9 특기 1 + 차기 개정 항목(사전등록 §11-2) |
| ③ | A-8 입력(L6/L6C exports)의 계약 §8-3 문면 간극(의도 준수·판정 논리곱 미사용) | §6-1 기재 |
| ④ | 함의의 이온 일반성 과소 기재(기전 사정거리 = 11이온, 측정 = 커버 3이온) | §6 서두 + §8 미결 #3 |
| ⑤ | A1 방법의 별칭 사각(감리-A 가 폐합 — 계열 밖 기록자 0 불변) | §4-2 + 차기 개정 항목(사전등록 §11-2) |

**감리-B NOTE 2건(#8·#9)의 반영 위치**: NOTE #8(a) ROW printer 좌표 설명 정정 = 서두
«입력 식별» + 부록 B 말미(감리-B 수정 문면 그대로, 판정자 양 좌표 직독 재확인) · NOTE #8(b)
`git diff --stat | tail -1` 명령 표기 정정 = 부록 A-2(판정자 실행 재확인) ·
NOTE #9(G-B1 selftest 의 read-only 재현 불능 — 대리 실행 + 정적 오라클 교차검증으로 해소)
= 부록 A-5 주석.

### ⟹ **폐합 조건(개정17: 감리 두 겹 모두 통과) 충족.**
