# V3 적대검증 — ARTIS 함수 단위 parity 감사 (docs/ARTIS_FUNCTION_PARITY_AUDIT_2026-07-31.md)

**검증일:** 2026-07-31 · **검증자:** Opus 적대검증(V3) · **모드:** read-only (본 파일만 신규 작성)
**Lumina:** `47bfa2001deb` / branch `thenmc-macroatom-fluorescence`
**ARTIS:** `../artis-ref` (= `/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref`)
**런타임 표본:** `logs/coevolve_consume_parity54/{stdout.log,stderr.log}`

기본 태도: 외부 감사문서와 Lumina 소스 주석 **둘 다** 불신. 모든 판정은 file:line 코드사슬 +
가능한 경우 parity54 실측치로만 근거한다.

---

## 0. parity54 환경 사실관계 (판정 전제, RUN FOOTER stdout.log:37855-37977, 119 vars)

| 변수 | 값 | 근거 |
|---|---|---|
| `LUMINA_ARTIS_PARITY` | **1** | :37953 |
| `LUMINA_MA_LINE_DESTRUCT` | **1** | :37922 |
| `LUMINA_KPACKET` | **1** | :37900 |
| `LUMINA_NLTE_ION_LOCK` | **1** | :37860 |
| `LUMINA_NLTE_LOCK_START_ITER` | **0** | :37915 |
| `LUMINA_NLTE_PER_ION_RESCALE` | **1** | :37958 |
| `LUMINA_CMF_BF_MILNE` | **2** | :37967 |
| `LUMINA_MA_RADRECOMB` | **1** | :37969 |
| `LUMINA_MA_REAL_UPSILON` | **1** | :37882 |
| `LUMINA_JBAR_MIN` / `LUMINA_JBAR_UNIFY` | 3 / 1 | :37973, :37877 |
| `LUMINA_NLTE_OPACITY_IONSTAGE` | **부재** | footer 전체 grep 카운트 = 0 |
| `LUMINA_IUP_BINFIELD` | **부재** | footer 전체 grep 카운트 = 0 |
| `LUMINA_NLTE_NO_ML_LOCK` | **부재** | footer 전체 grep 카운트 = 0 |

---

## 1. P0-1 — bf continuum 선택 + D6 배너

### (a) `compute_bf_opacity`가 per-continuum CDF를 버리고 (shell,bin)당 max-chi ion 하나만 남기는가 — **CONFIRMED**

- `src/lumina_plasma.c:6489-6491` — opacity 자체는 **합산**된다:
  `chi_contrib = n_level * sigma; bf->chi_bf[idx] += chi_contrib;` (합산은 정상)
- `src/lumina_plasma.c:6518-6522` — 그러나 사건 라우팅용으로는 argmax 하나만 남는다:
  ```c
  if (chi_contrib > best_chi[idx]) { best_chi[idx] = chi_contrib; best_ip[idx] = ip; }
  ```
  누적분포(CDF) 배열은 **생성 자체가 없다**.
- `src/lumina_plasma.c:6528-6540` — `activation_level[idx] = rr_act[ip] : ionized_ground[ip]`
  (idx = shell×n_freq_bins + bin). (shell,bin)당 정수 **하나**.
- 감사문서보다 **더 나쁜 사실 하나**: `best_ip`는 dominant **level**이 아니라 dominant **ion**이다.
  `rr_act[ip]`는 `src/lumina_plasma.c:6334-6339`에서 그 ion의 **첫 번째** mapped level 하나로
  ion 전체를 대표시킨다(`for (l...) if (ma_rr_target[l] >= 0) { rr_act[ip] = ...; break; }`).
  즉 lower level identity와 phixs target identity가 **둘 다** 소실된다.
- `src/lumina_cuda.cu:3326-3333` `d_bf_get_activation_level` — bin 인덱싱 후 `d_bf_act[shell*nb+bin]`
  정수 하나를 반환. CDF 소비 지점 없음.

**ARTIS 대조 (확인됨):** `artis-ref/rpkt.cc:414-426` — `chi_bf_sum` 누적분포에서
`chi_bf_rand = rng_uniform * chi_bf_inrest`로 실제 continuum `(element, ion, level, phixstargetindex)`를
추첨한다. (감사문서 인용 `rpkt.cc:405-445` → 실측 `407-446`, 오차 2줄, 실질 일치.)

### (b) GPU continuum-absorption 경로에 `nu_edge/nu` 이온화-vs-운동에너지 추첨이 없는가 — **CONFIRMED**

- `src/lumina_cuda.cu:5469-5479` 3-way continuum split (escatter / ff-heat / bf) → `cont_chan==2`
- `src/lumina_cuda.cu:5480-5487` bf 진입 → `act_level = d_bf_get_activation_level(...)`
- `src/lumina_cuda.cu:5496-5507` `act_level < 0`이면 **k-packet 여기 CDF**에서 임의 level 추첨
  (흡수 photon의 continuum/`nu_edge`와 무관한 fallback — 감사문서 지적 그대로)
- `src/lumina_cuda.cu:5508-5538` `act_level >= 0`이면 **무조건** `d_macro_atom_interaction`
  → 그 사이 어떤 난수도 `nu_edge/nu`와 비교되지 않는다.
- **저장소 전수조사:** `grep -n nu_edge src/*.c src/*.cu src/*.h` 전체 히트를 확인. `nu_edge`가 등장하는
  곳은 (i) opacity 빌드(`plasma.c:6446-6507`), (ii) recomb 토폴로지 edge 테이블
  (`plasma.c:2801-2876`), (iii) fb/radrecomb **방출** 주파수 샘플러
  (`cuda.cu:4468-4516`, `4758-4861`, `5568-5628`), (iv) NLTE 행렬 edge
  (`plasma.c:14487`, `nlte_gemm.cu:184-199`) 뿐. **흡수 사건 라우팅에서 `nu_edge/nu`를 난수와
  비교하는 코드는 CPU·GPU 어디에도 없다.**
- CPU 미러도 동일: `src/lumina_transport.c:558-580` — `bf_get_activation_level` → 있으면
  `macro_atom_event`, 없으면 `bf_absorption_event`. 분기 없음.
- 분기를 무장하는 게이트도 없음: `grep -rn "IONIZ_SPLIT|BF_SPLIT|EDGE_SPLIT|ioniz_vs|BF_KINETIC" src/ scripts/` → **0 hits**.

**ARTIS 대조 (확인됨):** `artis-ref/rpkt.cc:434-446`
```cpp
if (rng_uniform(pkt.number) < nu_edge / nu) { ... do_macroatom(... ion+1, get_phixsupperlevel(...)); }
else { pkt.type = TYPE_KPKT; }   // photoelectron kinetic energy -> thermal pool
```

### (c) D6 배너는 허위 capability 보고인가 — **CONFIRMED (3중 자기모순)**

- `src/lumina_cuda.cu:6142`
  `"  D6 bf event nu_edge/nu ioniz-vs-kinetic split + stim-recomb corr"`
  — **한정어가 전혀 없다.** 같은 배너 블록에서 미구현 항목은 모두 명시적으로 표기된다:
  `B2 ... = residual` (:6128), `B4 ... = residual` (:6130-6131), `D1 ... = residual` (:6134),
  `D3 ... = residual` (:6137-6138), `D4 **DISABLED** (no ARTIS analog)` (:6139),
  `E1 ... = residual` (:6144), `E3 **DISABLED**` (:6147), `M1 ... = residual` (:6154-6155).
  D6만 유일하게 **구현된 capability로 제시**된다.
- 같은 저장소가 두 곳에서 정반대를 말한다:
  - `src/lumina.h:96` — `"...and D6 nu_edge/nu split are **BLOCKED/deferred**"`
  - `docs/ARTIS_PARITY_GAP_AUDIT.md:45` — `"**D6 [APPROX]** ... Lumina level-map activation, **no split/corr**"`
- 배너는 capability 검사 없이 `if (artis_parity_enabled())` 하나로 무조건 출력된다
  (`cuda.cu:6118-6162`).
- **런타임 확증:** parity54 `stdout.log:184`에 이 줄이 실제로 출력됨 (D5는 :183).

### (d) bf event handler / `LUMINA_CMF_BF_MILNE`

- `src/lumina_plasma.c:6680-6707` `bf_absorption_event` — `sample_planck_frequency(T_rad)` 재방출.
  `src/lumina_cuda.cu:3407-3435` `d_bf_absorption_event` — 동일. 감사문서 서술 정확.
  (parity/macroatom 경로에서는 `act_level>=0`이면 이 handler를 타지 않으므로, 이 경로는
  act_level 미할당 bin 및 non-macroatom 모드용 legacy임 — 감사문서의 "더 단순하게"는 맞지만
  parity54의 주 경로는 아니다. **문맥 정정**.)
- `LUMINA_CMF_BF_MILNE`(parity54=2)는 `plasma.c:6231`에서 파싱되어 `eta_bf` **방출** 항만 만든다
  (`:6492-6516`). 사건 분기에는 관여하지 않으며, 그 주석 자체가 `:6505-6506`에서
  `"Stimulated recombination is dropped, consistent with the uncorrected legacy chi_bf"`라고 명시.

> **P0-1 판정: (a) CONFIRMED · (b) CONFIRMED · (c) CONFIRMED (상태보고 결함 확정) · (d) CONFIRMED(문맥 정정 1건).**

**2026-07-31 Wave 2 구현 추기:** 위 판정은 결함 발견 당시의 역사 기록으로
유지한다. 현 소스는 `LUMINA_FIX_BF_CONTINUUM_EVENT=1`에서
`(element,ion,lower-level,target)` route별 opacity 항을 누적해 continuum을
추첨하고, 선택 edge에 대해 별도 `nu_edge/nu` draw로 MA-vs-k를 나눈다.
`activation_level` argmax는 이 ON arm에서 생성·소비하지 않는다. D6 배너도
event/stim/k-packet의 실제 조합을 `ENABLED`/`PARTIAL`/`residual`로 구분한다.
게이트 OFF는 본 절이 감사한 legacy 동작을 보존한다.

---

## 2. P0-2 — stimulated recombination correction

### ARTIS 측 인용 검증 — **CONFIRMED (인용 정확)**

`artis-ref/rpkt.cc:737-757`:
```cpp
double corrfactor = 1.;
double modified_departure_ratio = get_cellcache(...).allcont_modified_departureratios[i];
... modified_departure_ratio = nnupperionlevel / nnlevel * clumpednne * modified_sahafact;
const double stimfactor = modified_departure_ratio * exp(-HOVERKB * (nu - nu_edge) / T_e);
corrfactor = std::max(0., 1 - stimfactor);
sigma_contr = sigma_bf * allcont_probability[i] * corrfactor;
```
`:765` `chi_bf_sum += nnlevel * sigma_contr;`
감사문서의 pseudo-code와 **수식·인자·라인번호 모두 일치**.

### Lumina 측 — **CONFIRMED (누락 확정, 저장소 전역)**

- `src/lumina_plasma.c:6489` — `double chi_contrib = n_level * sigma;` 보정계수 없음.
- `src/lumina_plasma.c:6496-6506` — 주석이 스스로 인정:
  `"Stimulated recombination is dropped, consistent with the uncorrected legacy chi_bf."`
- **다른 site 존재 여부 (요청 항목):**
  `grep -rn "stim_recomb|stimulated recomb|corrfactor|stimfactor|departure_ratio|departureratio" src/*.c src/*.cu src/*.h`
  → **0 hits**. Lumina에는 bf stim-recomb 보정이 **한 군데도 없다.**
- 유일한 `stim_corr`는 **bound-bound** `tau_sobolev`용이며 bf와 무관:
  `src/lumina_plasma.c:2285-2292`, `:15851-15858` (`1 - g_lo n_u/(g_up n_l)`).
- 부수 확인(감사 P0-2 전반부): `src/lumina_plasma.c:6348` `if (stage < 1) continue;` — neutral bf 전면 제외 사실 확인.

따라서 D6 배너의 `+ stim-recomb corr` 문구 역시 **허위**다(§1(c)와 동일 결론, 독립 근거).

> **P0-2 판정: CONFIRMED.**

---

## 3. P0-4 — 공유 pair의 save/restore·per-ion pin — **PARTLY (사실 전부 CONFIRMED, 성격규정은 감사문서 과잉주장)**

### 3.1 사실관계 — 전 항목 CONFIRMED, 그리고 **감사문서보다 강함**

**(i) per-ion 재고정** — `src/lumina_plasma.c:15533-15538`
```c
int lock = nlte_ion_lock_active(nlte->current_iter) ||
           nlte_per_ion_rescale_active() || pair_shares_slot;
```
`:15634-15655` — `lock`이면 lo/hi 각각을 `atom->ion_number_density[ip*n_shells+shell]`
(= 외부 ion-balance 소유자의 total)로 **개별 재정규화**. 행렬이 푼 ion split은 폐기되고
ion **내부** level 분포만 살아남는다. GPU 미러 `src/lumina_cuda.cu:1596-1610` 동일.

> **감사문서 대비 강화(NEW):** 감사문서는 이를 "공유 slot을 포함한 pair"의 문제로 서술했으나,
> parity54는 `LUMINA_NLTE_ION_LOCK=1` + `LUMINA_NLTE_LOCK_START_ITER=0`
> (`nlte_ion_lock_active`, `plasma.c:7177-7189`) + `LUMINA_NLTE_PER_ION_RESCALE=1`이
> **동시에** 켜져 있어 `lock`이 **모든 pair·모든 iteration에서 무조건 true**다.
> 즉 per-ion pin은 공유 pair 한정이 아니라 **전면 적용**이다.

**(ii) 공유 lo-ion 해의 폐기** — CPU `src/lumina_plasma.c:15995-16016`(save) / `:16027-16032`(restore),
GPU `src/lumina_cuda.cu:1633-1641`(restore). 뒤 pair가 계산한 공유 ion 값은 앞 pair 값으로 **복원**된다.
코드 주석(`:15987-15994`)이 의도를 명시하므로 "의도적 폐기"라는 감사문서 서술도 정확.

**(iii) `nlte_writeback_ion_stage`** — `src/lumina_plasma.c:2316-2317`
```c
const char *e = getenv("LUMINA_NLTE_OPACITY_IONSTAGE");
if (!e || e[0] != '1') return;
```
parity54 footer에 이 변수 **부재**(grep 카운트 0) ⇒ 함수는 **즉시 return**, 완전 무동작.
`:2321-2329`의 공유 pair skip 로직도 확인(generic, 인덱스 하드코딩 아님).

**(iv) NEW — 왜 outer iteration으로 수렴하지 않는지의 기전 (감사문서 미제시)**
`src/lumina_plasma.c:14484-14487`:
```c
double chi_eV = find_ioniz_energy(atom, Z_elem, nlte->nlte_ion[ion_idx_lo]);
```
pair 행렬의 photoion/recomb 블록은 **lo→hi 이온화만** 담는다. hi ion의 자기 이온화(→ 다음 stage) drain은 없다.
따라서 pair {O I, O II}가 확정하는 O II는 **O III로의 유출을 전혀 모른다**. 그리고 (ii)의 restore가
그 pair를 O II의 **최종 소유자**로 만든다. 결과: 공유 ion의 확정값은 구조적으로 **상위 stage에 눈먼 행렬**이
생산한다. ce_iter를 아무리 늘려도 element-wide 해로 수렴하지 않는다 — 다른 연산자의 부동점이다.
⇒ 감사문서의 결론("pairwise 반복이 element-wide 해로 수렴한다는 해석은 성립하지 않는다")은 **기전까지 확증**.

### 3.2 설계 vs 결함 판별 (요청된 CRITICAL adjudication)

`docs/B1_NLTE_ION_OWNER_DESIGN.md` §0을 읽으면 이것은 **알려진·문서화된·선언된 아키텍처**다:

> §0: "Committed ion owner = radeq_simul_all's nebular ladder ... The NLTE matrix runs AFTER
> and its writeback **RESCALES each ion back to the ladder's totals** ... The matrix's internal
> split (sum_lo/sum_hi) is **computed then thrown away**."
> §3 Stage 1 gate = `LUMINA_NLTE_ION_OWNER` — **Status: DESIGN (no source edited)**, 미구현.

그리고 parity 배너 자신이 잔여로 선언한다 — `src/lumina_cuda.cu:6127-6128`:
> `"B2 ionization closure = LTE Saha @T_e,W=1; ion split rate-SE (overlapping-pair per-ion pin = **residual**, needs B1 matrix)"`

| 구성요소 | 판정 |
|---|---|
| (i) per-ion pin (`lock` 분기) | **DESIGN** — B1 설계문서가 현행 아키텍처로 명시, 배너가 residual로 자기선언. D6와 달리 **허위보고 아님**. |
| (ii) 공유 lo-ion save/restore | **DESIGN 의도 + 구조적 결함** — 의도(앞 pair 보호)는 주석에 명시되나, 그 귀결(공유 ion을 상위 stage에 눈먼 행렬이 확정)은 §3.1(iv)대로 수렴 불가. **의도되지 않은 결과**. |
| (iii) `nlte_writeback_ion_stage` skip + early return | **DESIGN, 그리고 parity54에서 완전 무동작** — 게이트 부재로 함수 자체가 죽어 있어 "skip 로직" 논의는 parity54에 대해 **무의미(moot)**. |

**따라서 감사문서의 "확정 P0 결함" 일괄 규정은 (i)·(iii)에 대해 과잉주장이다.**
(i)은 선언된 잔여, (iii)은 비활성 코드다. 실제 결함으로 원장에 올릴 것은 **(ii) 하나**이며,
그것도 (i)의 전면 pin이 살아 있는 한 bulk에 대해서는 **가려진다**(pin이 어차피 split을 덮어씀).

> **P0-4 판정: PARTLY.** 사실관계 4/4 CONFIRMED(+2건 감사문서보다 강화), 성격규정은 REFUTED-in-part:
> (i)(iii)=선언된 설계/잔여, (ii)=진성 구조 결함.

---

## 4. P0-6 — `MA_LINE_DESTRUCT` 이중추첨 — **CONFIRMED (최고 확신도)**

### 4.1 두 추첨의 정확한 재구성

**추첨 1 (pre-roll, COLDEEXC).** `src/lumina_plasma.c` `compute_transition_probabilities`,
level `lev`의 transition block 루프:
- `:3818` `double kp_deact = 0.0;`
- `:4170-4177` 진입조건 `kpacket_mode && n_e>0 && T_e>0 && kp_glo[line_id]>=0 && kp_gup[line_id]>=0`
  **그리고 `ttype == -1`** (radiative deexcitation)
- `:4183-4197` `artis_col_rates(...) → C_down = cd` (parity에서 real-Omega 우선, `MA_REAL_UPSILON=1`)
- `:4240` **`kp_deact += parity_ma ? (C_down * dE) : C_down;`** (`dE = h*nu_line`)
- 소비: `:4526-4532`
  ```c
  double denom = sum_rates + kp_deact;
  double pkv = (denom > 0.0) ? (kp_deact / denom) : 0.0;
  opacity->p_kpacket[lev*n_shells + s] = pkv;
  ```
- 장치측 발화: `src/lumina_cuda.cu:4200-4208` `if (force_kpacket || (pk > 0.0 && d_rng_uniform(rng) < pk))`

**합산 도메인:** `kp_deact`는 `lev`의 block 안 **모든 `ttype==-1` transition**에 대한 `Σ C_down·hν`.
`sum_rates`는 `:4316-4319`에서 같은 block의 RADDEEXC(`A_ul·β·hν`, `:3832`+`:4151`) +
INTERNALDOWNSAME(`(A_ul·β [+C_down])·e_low`, `:3873-3874`,`:3881-3915`,`:4152-4162`) +
INTERNALUPSAME(`(B_lu·J + C_up)·e_cur`, `:4091-4132`) + recomb/iup(`:4335-4356`,`:4475-4477`).
`g_ctp_idown_coll`은 parity에서 자동 ON(`:3353`)이므로 내부하향에 C가 포함된다.

⇒ **`pk`는 구조적으로 정확히 ARTIS의 `P(COLDEEXC) = sum_coldeexc / rate_total`이다.**
코드 주석(`:4220-4239`)의 이 주장은 **검증 통과**.

**ARTIS 대조 (확인됨):** `artis-ref/macroatom.cc:87-110`
```cpp
sum_raddeexc            += R * epsilon_trans;      // R = rad_deexcitation_ratecoeff = A_ul * beta
sum_coldeexc            += C * epsilon_trans;
sum_internal_down_same  += (R + C) * epsilon_target;
```
`:114-135` INTERNALUPSAME `(R+C+NT)*epsilon_current`.
`:393-402` **단일 fair draw**: `partial_sum(levelrates)` → `rng_uniform * cumulative[MA_ACTION_COUNT-1]` → upper_bound.
`:421-430` COLDEEXC 당첨 ⇒ `pkt.type = TYPE_KPKT; end_packet = true;`
`:406-418` RADDEEXC 당첨 ⇒ `do_macroatom_raddeexcitation(...)` → `:198-207` **어느 선인지만** 하위 추첨
→ 방출 → `end_packet = true`. **두 번째 추첨은 없다.**
`artis-ref/macroatom.h:41-57` — `rad_deexcitation_ratecoeff = A_ul * beta`, `beta = (1-e^{-τ})/τ`.
⇒ **ARTIS의 fair draw에는 β가 이미 들어 있다.**

**추첨 2 (terminal ε).** `src/lumina_cuda.cu:4335-4357`, CDF 워크가 `ttype==-1`을 고른 직후:
```c
if (d_ma_line_destruct_on && d_ma_line_eps && d_transition_type[tid] == -1) {
    double eps_d = d_ma_line_eps[tid*n_shells + current_shell_id];
    if (eps_d > 0.0 && d_rng_uniform(rng) < eps_d) { force_kpacket = 1; current_type = 0; ... }
}
```
`eps` 생산자 — `src/lumina_plasma.c:4214-4218`:
```c
if (opacity->ma_line_eps) {
    double rad_bare = atom->line_A_ul[line_id] * beta;      /* == :3832의 그 값 */
    double denom_e  = C_down + rad_bare;
    opacity->ma_line_eps[tid*n_shells + s] = C_down / denom_e;
}
```

### 4.2 결정적 사실 — 두 추첨은 **같은 스칼라**를 쓴다

`:4214-4218`(ε 저장)과 `:4240`(kp_deact 누적)은
**같은 루프 반복 · 같은 `tid` · 같은 가드(`kpacket_mode && ttype==-1 && g_up>0`) · 같은 지역변수 `C_down`**
(`:4182-4202`에서 1회 계산)을 쓴다. 나아가:

- `kp_deact`의 **전 저장소 등장은 4곳뿐**(`plasma.c:3818` 선언, `:4240` 누적, `:4527-4528` 소비, 나머지는 주석).
  destruct 게이트가 켜졌을 때 `kp_deact`에서 해당 선의 `C_down`을 **빼는 코드는 없다**.
- `ma_line_eps` 할당은 게이트 종속(`:3698-3709`)이나, `kp_deact` 누적(`:4240`)은 **게이트와 무관하게 항상** 실행된다.

⇒ 작업지시에서 확인을 요청한 "**pre-roll이 선택된 선의 `C_down`을 제외하도록 분할되어 있을 가능성**"은
**명시적으로 REFUTED**. 두 도메인은 disjoint가 아니라 **동일**하다.

`src/lumina_cuda.cu:2101-2104`의 항변("mutually exclusive control-flow branches ... no collisional double-count")은
**제어흐름에 대해서는 참, 확률측도에 대해서는 거짓**이다. 두 분기가 순차적으로 배치되어도
전체 열화확률은 두 분기의 **합**이며, 두 분기가 같은 `C_down`을 소비하면 그 측도는 중복 계상된다.
게다가 `:4353`의 `force_kpacket = 1`은 `:4196/4208`에서 pk roll을 **우회하여 무조건** k-packet을 형성하므로,
파괴 경로는 pre-roll과 **완전히 동일한 sink**(ff/fb/thermal-CDF exit, `:4211-4318`)로 들어간다.
물리적으로 구분되는 별개 채널이 아니다.

### 4.3 2-준위 toy 산술 (요청 항목)

상태: 상준위 u, 하준위 l(=이온 바닥, `epsilon_target = 0`, 상향전이 없음).
`R ≡ A_ul·β`, `C ≡ C_ul`, `ε_trans = hν`, `r ≡ C/(A·β) = C/R`.

**(i) ARTIS fair draw**
```
RADDEEXC = R·hν ,  COLDEEXC = C·hν ,  INTERNALDOWNSAME = (R+C)·0 = 0
rate_total = (R+C)·hν
P_ARTIS(thermalize) = C/(R+C) ≡ p        P_ARTIS(photon) = R/(R+C) = 1-p
```

**(ii) Lumina pre-roll + ε**
```
sum_rates = R·hν  ,  kp_deact = C·hν
pk        = C·hν / (R·hν + C·hν) = C/(R+C) = p          ← ARTIS와 동일 ✔
ε         = C/(C + A·β) = C/(C+R)        = p            ← 같은 값이 재사용됨
P_LUMINA(thermalize) = p + (1-p)·p = 2p - p²
P_LUMINA(photon)     = (1-p)²
```

**같은가? 아니다.** 모든 `p ∈ (0,1)`에서 `2p - p² > p`. 초과분 `Δ = p(1-p)`, 비율 `= 2 - p`.

| `C/(A·β)` | `p` (=ARTIS) | `2p-p²` (=Lumina) | 초과 `Δ` | 배율 |
|---|---|---|---|---|
| 0.001 | 0.0009990 | 0.0019970 | 0.0009980 | **×1.99900** |
| 0.01  | 0.0099010 | 0.0197040 | 0.0098030 | **×1.99010** |
| 0.1   | 0.0909091 | 0.1735537 | 0.0826446 | **×1.90909** |

**결론: 복사지배(작은 C/Aβ) 영역 — 즉 SN Ia 광구의 허용선 대부분 — 에서 충돌열화 확률이 사실상 두 배가 된다.**
감사문서의 `p + (1-p)p = 2p-p²` 산술은 **정확**하다.

**물리적 해석 (감사문서 미제시, NEW):** `ε = C/(C+A·β)`는 two-level ALI의 열화 파라미터로,
"광자가 β 확률로 탈출하고 아니면 재흡수되어 재여기된다"는 경쟁을 **이미 담고 있다**.
macro-atom의 `A·β` 대 `C` fair draw는 **똑같은 경쟁의 등가 표현**이다.
두 개를 겹쳐 적용하는 것은 같은 탈출확률 물리를 **두 번 세는** 것이다.

### 4.4 실제 규모 — parity54 실측

- **파괴량** (`stdout.log:3778, 7195, 10656, 14121, 17585, 21048, 24509, 27598, 30491, 33343, 35652, 37727`):
  it0~it11 `destroyed/terminals` = 0.0035 / 0.0043 / 0.0080 / 0.0079 / 0.0067 / 0.0093 /
  **0.0102** / 0.0063 / 0.0031 / 0.0015 / 0.0009 / 0.0007.
  **전 iteration 합계: destroyed = 34,569,259 / terminals = 6,524,793,663 → 0.530%.**
- **ARTIS-fair 몫** (`stdout.log:445, 3873, 7329, ... , 35786`):
  `[KPACKET] mean p_kpacket: shell0 = 1.49e-04 ~ 3.04e-04, shell49 = 8.03e-07 ~ 9.07e-06`.
- **level 평균 pk** (`stderr.log`, `[KPD-FE]` 600개 (shell,iter) 레코드 **전부**):
  `pk_mean = 0.000`, `pk>0.9 = 0` — 예외 없음.
- 방문가중 상위 준위(`[KPD-FE2]`)에서만 `pk = 0.0021 ~ 0.1859`.

⇒ M1 수정(`C_down·dE` 가중, `plasma.c:4220-4239`)이 pk를 옛 funnel(`pk→1`)에서 `~1e-4` 수준으로 내린 결과,
**parity54에서 macro-atom 열화의 지배적 원천은 ARTIS에 존재하지 않는 terminal ε 추첨**이다.
level-평균 기준으로는 `0.0053 / ~2e-4 ≈ 25×`, 방문가중 상위준위 기준으로는 대체로 동급~수배.
정확한 배율은 `d_kpacket_count`가 출력되지 않아 단정할 수 없으나, **부호와 자릿수는 확정**이며
어떤 경우에도 이 채널 전체가 ARTIS 대비 순증(順增)이다.

- **극단 사례** (`stdout.log:447-449, 451-453`): 두꺼운/금지선에서 `β→0`이므로 `ε→1`.
  `Fe III forbidden 1989.0Å ε = 0.9447`, `Co III thick 118800Å ε = 0.9999`.
  이런 terminal은 **거의 확실히 파괴**된다 — ARTIS에는 대응 없음.

> **P0-6 판정: CONFIRMED.** 코드 주석 `cuda.cu:2101-2104`의 방어는 **REFUTED**.
> 분할설계(disjoint 도메인) 가설도 **REFUTED**(동일 `C_down`, 동일 가드, 동일 `tid`).
> ARTIS 비교 lane에서 `LUMINA_MA_LINE_DESTRUCT`는 **반드시 OFF**여야 한다.

---

## 5. P1-2 — IUP_JBLUE provenance — **CONFIRMED (4/4)**

| 주장 | 검증 |
|---|---|
| ARTIS `DETAILED_LINE_ESTIMATORS_ON = false` | **CONFIRMED** `artis-ref/artisoptions.h:74` (`constexpr bool ... = false`) |
| ARTIS `rad_excitation_ratecoeff`가 `Jb_lu`가 아니라 `radfield(nu)` 사용 | **CONFIRMED** `artis-ref/macroatom.cc:588-596` — `if (DETAILED_LINE_ESTIMATORS_ON && ...)` 블록은 constexpr false로 **사문(dead)**, 실제 반환은 `:596` `R_over_J_nu * radfield::radfield(nu_trans, nonemptymgi)` |
| Lumina parity 기본 `g_ctp_iup_jblue` ON | **CONFIRMED** `src/lumina_plasma.c:3408-3416` — `g_ctp_iup_jblue = (env_on \|\| artis_parity_enabled()) ? 1 : 0;` (parity54는 `LUMINA_IUP_JBLUE` 미설정이므로 **parity 게이트만으로** ON) |
| parity54 J-blue 사용률 84~91% | **CONFIRMED** `stdout.log` it0 `0.0%`(:3780) → it1 87.6% → it2 87.1% → it3 89.8% → it4 **91.0%** → it5 91.0% → it6 90.4% → it7 89.6% → it8 88.7% → it9 86.9% → it10 84.2% → it11 **83.9%**(:37729). 감사문서의 "84--91%"는 it11의 83.9%를 반올림한 것으로 실질 정확 |
| `LUMINA_IUP_BINFIELD`가 실제 ARTIS 소비자에 가깝고 parity54 미설정 | **CONFIRMED** `src/lumina_plasma.c:3440-3456` — 배너가 스스로 `"[ARTIS macroatom.cc:596 radfield(nu_trans) with artisoptions.h:74 DETAILED_LINE_ESTIMATORS_ON=false]"`라고 출처를 명시. parity54 footer grep 카운트 **0** |
| ARTIS `get_Jb_lu`에 contribution-count threshold 없음 | **CONFIRMED** `artis-ref/radfield.cc:650-653` — `return prev_Jb_lu_normed[nonemptymgi][jblueindex].value;` (assert 2개 외 조건 없음). vs Lumina `JBAR_MIN=3` 문턱(`plasma.c:3978-3985`) |

**부수 관찰(NEW):** `LUMINA_IUP_BINFIELD`의 배너 문구는 **Lumina 소스 자신이 BINFIELD를
ARTIS-충실 경로로 인정**하고 있음을 보여준다. 즉 이 불일치는 "몰라서"가 아니라 **선택**이다.
다만 그 선택이 parity 기본값이라는 사실은 배너 어디에도 잔여로 표기되지 않는다.

> **P1-2 판정: CONFIRMED.**

---

## 6. P1-1 — bin 기하 — **CONFIRMED (정량화 완료)**

### ARTIS (전부 확인)
- `artis-ref/artisoptions.h:62` `RADFIELDBINCOUNT = 24`
- `:66` `RADFIELDBINS_NU_MIN = CLIGHT/40000e-8` = 7.49481e13 Hz (40000 Å)
- `:68` `RADFIELDBINS_NU_MAX = CLIGHT/1085e-8` = 2.76307e15 Hz (1085 Å)
- `:70` `RADFIELDBINS_T_E_SUPERBIN_NU_MAX = CLIGHT/10e-8` = 2.99792e17 Hz (10 Å)
- `artis-ref/radfield.cc:62-63` `delta_nu = (NU_MAX - NU_MIN)/(RADFIELDBINCOUNT - 1)` = **1.168746e14 Hz (주파수 선형)**
- `:102-119` bin 0..22 = `NU_MIN + (i+1)·delta_nu`; bin 23 upper = superbin max
- `:126-132` `nu >= RADFIELDBINS_NU_MAX` ⇒ 무조건 superbin(index 23)
- `:765-768` **`if (binindex == RADFIELDBINCOUNT - 1) T_R_bin = grid::get_Te(...)`** — `T_R:=T_e`는 **superbin 단 하나**에만
- `:779-795` `W>1e4` rail 재시도

### Lumina (전부 확인)
- `src/lumina.h:491-493` `NLTE_N_FREQ_BINS 1000`, `NLTE_NU_MIN 1.5e14` (20000 Å), `NLTE_NU_MAX 3.0e16` (100 Å)
- `src/lumina_plasma.c:694` `ARTIS_RADFIELD_NC 24`
- `:917-925` `int c = (int)((long)f * NC / nb);` ⇒ fine **log-ν** 격자를 균등 그룹핑 ⇒ coarse도 **log-ν 균등**
  (bin당 ν 비 = **1.2492**)
- `:940-966` `if (tepin_on && nu_lo >= nu_superbin)` ⇒ **1085 Å보다 완전히 짧은 모든 coarse bin**이 `T_R:=T_e` 핀

### 경계 대조 (실산출)

| bin | ARTIS λ_lo → λ_hi (Å) | LUMINA λ_lo → λ_hi (Å) |
|---|---|---|
| 0 | 40000.0 → 15628.6 | 19986.2 → 15998.8 |
| 5 | 4547.0 → 3862.3 | 6604.1 → 5314.6 |
| 10 | 2410.5 → 2203.4 | 2193.8 → 1756.1 |
| 12 | 2029.1 → 1880.4 | 1413.2 → 1131.3 |
| 13 | 1880.4 → 1752.0 | 1131.3 → 905.6 |
| 14 | 1752.0 → 1639.9 | 905.6 → 728.8 |
| 20 | 1242.7 → 1185.3 | 240.8 → 193.8 |
| 22 | 1132.9 → 1085.0 | 155.1 → 124.2 |
| 23 | **1085.0 → 10.0 (superbin)** | 124.2 → 99.9 |

**광이온 edge 창 500-1100 Å:**
- **ARTIS: 내부 경계 0개.** 이 대역 전체가 bin 23(superbin, `T_R=T_e`) 안에 통째로 들어간다.
  1085 Å 바로 위쪽 경계는 1185.3, 1132.9, 1085.0 Å.
- **LUMINA: 내부 경계 5개** — **1131.3, 905.6, 728.8, 583.4, 467.0 Å.**
  1085 Å보다 완전히 짧은 coarse bin은 **c=14..23의 10개**, 전부 TEPIN 후보:
  905.6-728.8 / 728.8-583.4 / 583.4-467.0 / 467.0-375.8 / 375.8-300.8 /
  300.8-240.8 / 240.8-193.8 / 193.8-155.1 / 155.1-124.2 / 124.2-99.9 Å.

**NEW — 감사문서가 놓친 경계 오정렬 1건:** Lumina coarse bin 13 = **1131.3 → 905.6 Å**은
ARTIS의 1085 Å superbin 바닥을 **가로지른다**. `nu_lo < nu_superbin`이므로 TEPIN 조건에
걸리지 않아 **색온도 fit**을 받는다. 즉 ARTIS가 `T_R=T_e`로 처리하는 1085-905.6 Å 구간이
Lumina에서는 1131.3-1085 Å와 한 bin에 섞여 fit된다. 이는 bin 개수 차이가 아니라 **경계 위상 차이**다.

**범위 절단 (양쪽 다):**
- 적색: Lumina 19986 Å vs ARTIS 40000 Å ⇒ **2.00× 좁음** (NIR 결손)
- 청색: Lumina 99.9 Å vs ARTIS 10 Å ⇒ **10.0 - 99.9 Å 대역 전면 부재**

**런타임 확증:** `stdout.log:3783,...,35658` `[C1-SUPERBIN-TEPIN] it N: {92..149} coarse bins pinned
to T_R=T_e ... in {40..50} shells` ⇒ shell당 **약 3개** bin이 핀을 먹는다(ARTIS는 정확히 1개).
(10개 후보 중 나머지는 `J_c <= 0` 조기 탈출 `:937`로 zero-field 처리.)

> **P1-1 판정: CONFIRMED (+ 경계 오정렬 NEW 1건).**

---

## 7. NEW findings (감사문서에 없는 것)

1. **[N1 / P0-1]** bf activation의 소실은 continuum→level 뿐 아니라 **ion→level**까지다.
   `plasma.c:6334-6339`의 `rr_act[ip]`는 해당 ion의 **첫 번째** mapped level로 ion 전체를 대표시킨다.
   `best_ip`도 dominant **ion**이지 dominant **level**이 아니다.
2. **[N2 / P0-4]** parity54는 `ION_LOCK=1`+`LOCK_START_ITER=0`+`PER_ION_RESCALE=1` 삼중이라
   per-ion pin이 **공유 pair 한정이 아니라 전 pair·전 iteration 전면 적용**이다(`plasma.c:15537-15538`).
   따라서 감사문서가 (ii) save/restore에 돌린 비중은 실제로는 (i) pin이 대부분 흡수한다.
3. **[N3 / P0-4]** 수렴 불가의 **기전** 특정: pair 행렬의 이온화 블록이
   `find_ioniz_energy(..., ion_idx_lo)` 하나만 쓴다(`plasma.c:14484-14487`) ⇒ hi ion에 상위 drain 없음
   ⇒ C1 restore와 결합하면 공유 ion은 **영구히 상위 stage에 눈먼 행렬**이 확정한다.
4. **[N4 / P0-6]** `pk`는 실측 `~1e-4`(level 평균 `0.000`, `pk>0.9` 전무)까지 내려가 있어,
   parity54에서 macro-atom 열화의 **지배 채널이 ARTIS-부재 terminal ε**이다. 즉 이 오염은
   "2배"가 아니라 **채널 지배권 자체의 전복**이다. 누적 파괴량 34,569,259 / 6,524,793,663 = **0.530%**.
5. **[N5 / P0-6]** `ε = C/(C+A·β)`와 macro-atom의 `A·β` vs `C` fair draw는 **같은 two-level 탈출경쟁의
   등가 표현**이다(ARTIS `macroatom.h:41-57`에서 `R = A_ul·β` 확인). 두 번 적용은 β 물리의 이중계상.
6. **[N6 / P1-1]** Lumina coarse bin 13(1131.3-905.6 Å)이 ARTIS superbin 바닥(1085 Å)을 가로질러,
   ARTIS가 `T_R=T_e`로 다루는 대역 일부가 Lumina에서는 색온도 fit을 받는다.
7. **[N7 / 상태보고 일반]** D6는 배너 블록에서 **유일하게** 잔여/비활성 한정어가 없는 미구현 항목이다.
   같은 저장소의 `lumina.h:96`("BLOCKED/deferred")과 `docs/ARTIS_PARITY_GAP_AUDIT.md:45`("[APPROX] no split/corr")가
   배너와 정면 충돌한다 ⇒ 배너를 실제 capability 검사로 바꾸는 것은 **코드 한 줄이 아니라 상태보고 규약 문제**.
8. **[N8 / 문맥정정]** 감사문서 §2의 "CPU/GPU `bf_absorption_event`가 `B_nu(T_rad)`로 재방출"은 사실이나,
   parity54 macroatom 모드에서 그 handler는 `act_level < 0` **및** k-packet CDF fallback도 실패한 경우에만
   도달한다(`cuda.cu:5496-5508`). 주 경로의 결함은 "Planck 재방출"이 아니라 "**분기 부재**"다.

---

## 8. 판정 요약표

| 항목 | 판정 | 핵심 근거 |
|---|---|---|
| **P0-1(a)** continuum CDF 폐기, max-chi ion 고정 | **CONFIRMED** | `plasma.c:6518-6522`, `:6528-6540`, `:6334-6339`; ARTIS `rpkt.cc:414-426` |
| **P0-1(b)** GPU 경로에 `nu_edge/nu` 분기 부재 | **CONFIRMED** | `cuda.cu:5480-5538`; 전수 grep 0 hits; CPU 미러 `transport.c:558-580`; ARTIS `rpkt.cc:435-446` |
| **P0-1(c)** D6 배너 = 허위 capability 보고 | **CONFIRMED** | `cuda.cu:6142` 무한정어 vs `lumina.h:96` "BLOCKED/deferred" vs `ARTIS_PARITY_GAP_AUDIT.md:45` "[APPROX]"; parity54 `stdout.log:184` 출력 확인 |
| **P0-2** stim-recomb corr 누락 (+ARTIS 인용 정확) | **CONFIRMED** | ARTIS `rpkt.cc:737-757,765`; Lumina `plasma.c:6489`, 자백주석 `:6496-6506`; 저장소 전역 grep **0 hits** |
| **P0-4** 사실관계(pin/save-restore/writeback) | **CONFIRMED (+강화)** | `plasma.c:15533-15538`, `:15634-15655`, `:15995-16032`, `:2316-2329`; `cuda.cu:1596-1610`, `:1633-1641` |
| **P0-4** "확정 P0 결함" 성격규정 | **PARTLY / 과잉주장** | (i)pin=선언된 설계·잔여(`B1_NLTE_ION_OWNER_DESIGN.md` §0, `cuda.cu:6127-6128`), (ii)save/restore=진성 구조결함(`plasma.c:14484-14487` 기전), (iii)writeback=parity54에서 완전 무동작 |
| **P0-6** MA_LINE_DESTRUCT 이중추첨 | **CONFIRMED** | 동일 `C_down`(`plasma.c:4182-4202`)이 `:4214-4218`(ε)과 `:4240`(kp_deact) 양쪽; `kp_deact` 보정 site 부재; ARTIS `macroatom.cc:87-110,393-430` 단일 fair draw |
| **P0-6** 코드 주석 방어 `cuda.cu:2101-2104` | **REFUTED** | 제어흐름 배타 ≠ 측도 배타; `:4353 force_kpacket`이 `:4196/4208`을 우회해 **동일 sink**로 합류 |
| **P0-6** "분할설계(disjoint 도메인)" 가설 | **REFUTED** | 합산 도메인이 disjoint가 아니라 **동일**(같은 루프·같은 `tid`·같은 가드) |
| **P1-2** IUP_JBLUE provenance | **CONFIRMED** | `artisoptions.h:74`, `macroatom.cc:588-596`, `radfield.cc:650-653`; `plasma.c:3408-3416`, `:3440-3456`; parity54 83.9-91.0% |
| **P1-1** bin 기하 불일치 | **CONFIRMED (+정량화)** | `artisoptions.h:62,66,68,70` + `radfield.cc:62-63,102-132,765-768` vs `lumina.h:491-493` + `plasma.c:694,917-925,940-966`; 500-1100 Å 내부경계 ARTIS 0개 vs Lumina 5개; TEPIN 실측 ~3 bins/shell |

---

## 9. 원장 기재 권고

**확정 결함(defect)으로 기재:**
- **D-1** bf 흡수 사건의 continuum/level identity 소실 + `nu_edge/nu` 분기 부재 (P0-1 a·b)
- **D-2** D6 배너 허위 capability 보고 — 상태보고 결함 (P0-1 c). `lumina.h:96`·`ARTIS_PARITY_GAP_AUDIT.md:45`와 정면충돌.
- **D-3** `chi_bf` stimulated-recombination 보정 전역 부재 (P0-2)
- **D-4** `MA_LINE_DESTRUCT` 이중추첨 — 확률측도 중복 (P0-6). **parity54 전 런에 ON.**
- **D-5** 공유 lo-ion save/restore가 상위 stage에 눈먼 행렬을 최종 소유자로 만듦 (P0-4 (ii))

**2026-07-31 Wave 2 상태:** D-1과 D-2는 결함 원장에서는 역사 항목으로
유지하되, `LUMINA_FIX_BF_CONTINUUM_EVENT=1`에서 수리된 gated 상태다.
`LUMINA_KPACKET=0`이면 D-1의 kinetic arm이 first-class k-packet이 아니므로
런타임 배너가 `PARTIAL`로 제한한다. 기본 OFF에서는 위 legacy 결함이 의도대로
남는다.

**확정 parity 불일치(gap, 결함 아님)로 기재:**
- **G-1** per-ion pin에 의한 ion-split 폐기 (P0-4 (i)) — B1 설계문서·B2 배너가 이미 잔여로 선언
- **G-2** line 복사장 소비자 = per-line J-blue (ARTIS는 binned `radfield(nu)`) (P1-2)
- **G-3** 24-bin 기하: log-ν 24 bins + 다중 TEPIN vs 선형-ν 23 + 단일 EUV superbin; 양단 범위 절단 (P1-1)

**무효(moot)로 기재:**
- **M-1** `nlte_writeback_ion_stage`의 공유 pair skip — parity54에서 함수 자체가 즉시 return (P0-4 (iii))

---

**V3 VERDICT:** P0-1(a/b/c)·P0-2·P0-6·P1-1·P1-2 = 전부 CONFIRMED — 원장에 확정 결함 D-1·D-2·D-3·D-4 및 확정 parity 불일치 G-2·G-3로 기재. P0-4는 PARTLY: 코드 사실관계는 4/4 CONFIRMED하고 두 항목은 감사문서보다 강화(전 pair 전면 pin, hi-ion drain 부재 기전)되나 성격규정은 분해 필요 — 공유 lo-ion save/restore만 진성 결함(D-5)이고 per-ion pin은 B1 설계문서·B2 배너가 선언한 잔여(G-1), `nlte_writeback_ion_stage`는 parity54에서 무동작(M-1)이다. #4 산술 확정: 2-준위 toy에서 ARTIS = `p = C/(A·β+C)`, Lumina = `p + (1-p)p = 2p - p²`, 배율 `2-p` (C/Aβ = 0.001/0.01/0.1 → ×1.999/×1.990/×1.909) — 복사지배 영역에서 충돌열화가 사실상 두 배이며, 실측 parity54에서는 `pk ~ 1e-4`(level 평균 0.000)로 억눌린 반면 terminal ε가 terminal의 0.530%(34,569,259/6,524,793,663)를 파괴하여 **ARTIS-부재 채널이 macro-atom 열화의 지배 경로로 전복**되었다. `cuda.cu:2101-2104`의 "mutually exclusive" 방어와 "분할설계" 가설은 둘 다 REFUTED(같은 `C_down`·같은 `tid`·같은 가드, 보정 site 부재) — ARTIS 비교 lane에서 `LUMINA_MA_LINE_DESTRUCT`는 반드시 OFF.
