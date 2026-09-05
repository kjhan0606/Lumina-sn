# 판독 — d-proxy SL 수준 J̄–population 공변동, 승격 대 반상태 (2026-09-05, 런 0회)

**판독이지 판정이 아니다.** 분기 라벨·p-value 를 내지 않는다. 저자 = 운전석(Fable).
코딩 = gpt-5.6-sol (발주 `docs/ORDER_DPROXY_READOUT_{1,2}_2026-09-05.md`). 실행·재현 = 운전석(grammar-debug).

## 0. 한 줄

승격 모델(`LUMINA_SUPER_LEVELS=1`)에서도, 반상태(R-II)에서도, 반복 0→1 의 SL population 변화는
**이온 공통모드가 지배**하고 SL 수준 J̄ 공변동은 세 이온에서 부호가 일관되지 않는다.
승격은 공통모드의 **크기**를 10.7·28.3·9.5% → 3.3·0.7·4.3% (Fe·Co·Ni) 로 줄였고,
SL 별 변화의 **방향을 뒤집었다**(공통 231 짝에서 `rho(Δlog n_prom, Δlog n_semi) = −0.44`).

## 1. 입력 (전부 읽기 전용·기존 산출)

| 역할 | 루트 | 바이너리 | 노드 | 봉인 |
|---|---|---|---|---|
| 승격 | `/gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200` (331050) | `a4990a59…` sm_90 | syn104 H200 | 미봉인(post-seal FAIL은 검증기 오판, model rc=0). `stderr.log` sha `41355ff1…664d5c` |
| 승격 복제 | `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260831T040804Z_b1_k42_cell_1370680_33` (328645) | `39d452e4…` A100 | syn101 | stage seal |
| 반상태 | `/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1` (R-II) | `406712d` | A100 | `RII_RUN_SEAL.sha` 135/135 전·후 PASS |

모집단: `[A2-10][LINE-SATURATION-ROW]`, shell 0, Z=26/27/28 ion 1, `REQUESTED_TE` (T_e=10020 K 두 publication 동일, D3).

⚠**승격 대 반상태는 단일 변수 대조가 아니다**: 소스 판본(`406712d` 대 `364ee72`+provenance 패치)·refinement(36 대 42)·capture(1/1 대 0/0)도 다르다. 짝 수준 `rho(Δlog J̄_prom, Δlog J̄_semi) = +0.46` 이 두 런의 복사장 반복이 대체로 같은 방향임을 보이나, 아래 차이를 「승격 단독 효과」로 귀속하지 않는다.

## 2. 런 0회로 확정된 것 (등급 후보 C)

### 2-1. 플랫폼·패치 등가성
331050(H200, provenance 패치 바이너리) 대 328645(A100, 봉인 바이너리): ROW **685/685 행**, 9필드
(`n_upper Jbar tau_raw chi_raw signed_rate uncertainty Jbar_absolute_bound upper_global deck_scale`)
**17자리 문자열 동일 ⟹ double bit-일치**. env 차이는 `DPROXY_POPULATION_PROVENANCE=1` 대 `RADEQ_DIAG_LINE_CELL` 뿐.
⟹ 큐의 「A100 재런으로 플랫폼 대조」·「332010 재실행」은 수치적으로 새 정보가 없다 (둘 다 취소, user 지시).

### 2-2. runtime SL 구조 (provenance 실측)
runtime SL 98개 중 **다멤버 SL 은 둘**: Fe SL 465(관측 멤버 66 준위·91 짝), Co SL 869(21 준위·27 짝).
나머지 443행은 단독 anchor(`within_sl_frac=1`). 승격의 f-가중은 이 모집단에서 두 럼프에만 걸린다.

### 2-3. 승격 모델의 대수 정합 (D2)
runtime SL 안 `Δlog(n_upper/within_sl_frac)` 산포 최대 **1.8e-15** (SL 465), `within_sl_frac` 반복 간 변화 정확히 0.
⟹ `n_FL = N_SL·f_FL(T_e)` 가 bit 수준에서 성립. 반상태 R-II 는 `frac≡1`, 산포 0.

### 2-4. runtime SL 사상 이식 자기검증
331050 `upper_global → runtime_sl_id` 표를 331050 자신에 이식: provenance 필드와 685/685 일치. R-II 에 이식: 992 짝 중 288 사상 가능(704 는 표 밖 upper_global, 제외 계수).

## 3. 관측 (등급 B — 귀속 아님)

`X_s` = 덱 가중(`(g_u/g_l)A_ul/ν³`) Δlog J̄, `Y_s` = median Δlog N_SL, 블록 = (shell, Z, ion), midrank stratified Spearman.

| 블록 | SL | 승격 `Y` 평균±sd | 승격 `X` 평균 | 승격 ρ | 반상태 `Y` 평균±sd | 반상태 `X` 평균 | 반상태 ρ |
|---|---|---|---|---|---|---|---|
| Fe | 26 | +3.3%±1.1% | −0.7% | +0.50 (럼프 제외 +0.44) | +10.7%±2.1% | −4.0% | +0.79 |
| Co | 40/39 | +0.7%±0.4% | −0.1% | −0.01 | +28.3%±4.5% | −1.1% | −0.29 |
| Ni | 32 | +4.3%±1.1% | −0.1% | −0.23 | +9.5%±2.4% | −0.5% | −0.33 |
| pooled `rho_SL` | 98/97 | | | **+0.008** | | | **−0.129** (공통231: −0.086) |

- **공통모드가 지배**: 세 블록 모두 `Y` 평균이 SL 간 sd 의 3배 이상. 이는 이온 공통 성분이지 SL 별 응답이 아니다. (출처는 §8 에서 분해: `scale_ion` 이 **아니라** 여기 준위의 공통 departure 이동.)
- **럼프만 반대 방향**(승격): Fe 465 `Y=−2.0%, X=−5.6%`, Co 869 `Y=−0.4%, X=−0.6%` 가 각 블록 최저. 단독 SL 이 오른 만큼 럼프가 내린 재배분. 반상태에서는 SL 465 `Y=+2.0%`(Fe 최저이나 양).
- **Fe 블록만 양의 ρ 가 두 모델에서 반복**(+0.50/+0.79). Co·Ni 는 둘 다 음. 26점·permutation null 미계산 ⟹ 유의성 주장 없음.
- 짝 수준 ρ(승격) = +0.61 은 91멤버 럼프의 유사복제 아티팩트. SL 수준으로 올리면 소멸. 철회한 P-3 의 오류를 실물이 재확인.

## 4. 이 판독이 말하지 않는 것
J̄ 인과·FL 응답·수렴·스펙트럼 정확도·lumping 완전성·「승격 단독 효과」·유의성. 반복 1회(0→1)의 차분일 뿐이며 iter≥2 거동은 미측정.

## 5. 재현

```
# grammar-debug, 읽기 전용
cd /gpfs/kjhan/lumina/dproxy_audits/readout_331050_fable
python3 -B selftest_readout.py                       # positive 1/1 negative 8/8
python3 -B readout_dproxy_sl_331050.py --run-root <331050> --out-dir out_331050
python3 -B readout_dproxy_sl_331050.py --run-root <R-II> --out-dir rep/rii_c231 \
    --sl-map out_331050/sl_map.csv --semi-state --promoted-pairs out_331050/pairs.csv
```
스크립트 sha256 `02961bbd…9b9a6c`(판독기) · `019cd5a1…dd1ad5`(selftest); 저장소 사본 `scripts/dproxy_readout/`.
산출 sha256: `out_331050/readout.json` `8c40c1a6…`, `sl_map.csv` `a2d4149f…`, `readout_rii_all.json` `e4061bdc…`, `readout_rii_common231.json` `695cfd78…`.
grammar-debug 재현 산출과 Codex 샌드박스 산출: `sl_groups_*`·`contrast_common231.csv`·`pairs_rii_common231.csv` 해시 **일치**.

## 6. 분장 장부

| 단계 | 규약상 담당 | **실제** | 위반 |
|---|---|---|---|
| 명세(발주 1·2) | Fable | Fable(운전석) | — |
| 코딩 | Codex | gpt-5.6-sol | — |
| 실행·재현·해시 | 운전석 | 운전석(grammar-debug) | — |
| 판독 저작 | — | 운전석 | 판정 아님이므로 개정18 ① 미적용 |
| 판정·감리 | Fable fresh / Codex | **미발주** | user 결정 대기 |

## 7. 다음 (user 결정)
1. 이 관측에 분기 라벨이 필요하면 사전등록(귀무 D4 포함) → 판정 Fable fresh → 감리 두 겹. 필요 없으면 관측으로 대장 기재만.
2. ~~공통모드의 출처 분해~~ — **§8 에서 런 0회로 완료** (user 지시 09-05).

---

## 8. 공통모드 출처 분해 (2026-09-05 후속, 런 0회 · 배선 + 기존 산출)

**질문**: `n_FL = scale_ion · b[sl] · f(T_e)` 에서 반복 0→1 의 이온 공통모드(승격 +3.3/+0.7/+4.3%, 반상태 +10.7/+28.3/+9.5%)는 `scale_ion`(이온 분율)인가 `b`(여기 재정규화)인가.

### 8-1. 이온 밀도는 어느 산출물에도 없다
- `LINE-ION-OWNER`·`shell.csv`(30열)·ROW 어디에도 이온 수밀도 없음. writer `lumina_cuda.cu:8651·10281·11299` 는 `LUMINA_ION_POP_DUMP(_ITER)` 게이트이며 **어느 기존 런도 켜지 않았다**(전집 등록은 돼 있어 향후 1런으로 직접 측정 가능).
- `nlte_writeback_ion_stage`(`LUMINA_NLTE_OPACITY_IONSTAGE`) 도 두 런 stdout 0회. (⚠그 함수의 `1e-300` floor 둘은 클램프 대장 후보 — 이 런들에선 비활성.)

### 8-2. 그래서 배선으로 상한을 건다 — `scale_ion` 변화 ≲ 0.2% (구성상)
이온 분율 = `compute_ion_populations_shell` (`lumina_plasma.c:2742`) 의 rate-SE 비 `r = (Γ_phot + n_e C_ion)/(n_e α + n_e² C_3)` (`r1_rate_se_enabled()` 는 무조건 1, `:2559`). 입력:

| 입력 | 반복 0→1 변화 | 근거 |
|---|---|---|
| `Γ_phot` | < 1e-4 | 정준 view J(1234 bin) 적분(`nlte_bf_gamma_canonical`, A2-05). shell 0 J_ν ≥ 4 eV: bit-동일 161/704 bin, 나머지 \|Δlog J\| < 1e-4 (`physics_comparison/*.spectral.csv`) |
| 준위 가중 | 0 | `population_lte_level_fraction(T_e)` — **LTE 분율, NLTE population 불사용** (`:2612`) |
| `α(T_e)`, `C_ion(T_e)`, `C_3(T_e)` | 0 | T_e = 10020 K 고정 |
| `n_e` | **+0.1956%** | 3211601775.08 → 3217884585.65 (승격); R-II 도 +0.1956% |
| `Γ_nt` | 0 | `[GAMMA][PUBLISHED] generation=1` 1회, EXTERNAL_FILE |

⟹ `|Δlog r| ≤ Δlog n_e ≈ 0.2%`, 원소 총수 고정(`n_atom` 동일) ⟹ **`|Δlog scale_ion| ≲ 0.2%`**. 공통모드(≥0.7%)의 출처가 아니다.

### 8-3. 준위해는 이온 총량으로 정규화된다
`nlte_solve_ion_shell` (`:19430`, `nlte_solve_all_impl:20448` 에서 호출) 이 준위해 `xfl` 을 `scale = n_ion/Σxfl` 로 맞춘다(`:19823–19838`, 쌍 solve 는 `nlte_pair_total_density`). ⟹ **Σ_준위 n_FL = n_ion 강제**. 관측된 여기 준위(단독 SL)가 +3.6% 오르고 럼프가 −2.0% 내린 것(승격 Fe, 럼프 몫 24.8%)은 **이온 안 재배분**이며, 잔여는 관측 밖 준위(바닥·저준위)가 흡수한다.

### 8-4. 결론 (등급)
- **등급 C (구성상 확정)**: 공통모드는 `scale_ion` 이 아니다(상한 0.2%). 공통모드 = **여기 준위의 공통 departure 이동**(`b` 의 이온 공통 성분).
- **등급 B (관측)**: T_e 고정·연속 J 동결·n_e +0.2% 인 반복에서 그 이동을 만들 수 있는 입력은 **선 J̄(< 4 eV, 선별 −10%~+1.5%)** 뿐이다. 즉 population 은 J̄ 에 **응답하되 집단적으로**(행렬 결합) 응답하며, 각 SL 이 자기 선의 J̄ 를 따라가지는 않는다(§3 의 `rho_SL ≈ 0`). 승격은 그 집단 응답의 크기를 3–40배 줄였다(럼프 평탄 재분배가 사라져서로 읽히나, 단일 변수 대조 아님).
- **미측정**: 바닥·저준위의 실제 변화, 이온 분율의 실제 값(직접 측정은 `LUMINA_ION_POP_DUMP_ITER=1` 1런).
