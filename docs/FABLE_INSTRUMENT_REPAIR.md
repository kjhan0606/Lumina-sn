# FABLE_INSTRUMENT_REPAIR — 계기 수리 3건 (R-N4 · R-T2 · R-N2)

- 작성: fable (구현 설계자). 2026-08-02.
- 근거 문서: `docs/UV_CENSUS_CONSOLIDATION.md`, `docs/FABLE_UV_T3T4.md`, `docs/CODEX_UV_T5.md`.
  `docs/CODEX_STAGE32_ALI_DESIGN.md`는 **열람하지 않았다**(별 갈래).
- 모드: **설계 + 패치 작성 + 빌드 + 오프라인 fixture 실행**. 신규 모델/GPU 런 0, 커밋 0,
  생산 트리 수정 0 (패치는 `patches/instr_*.patch`로만 제출).
- 표기: **[F]** 본 작업 실측 · **[F-code]** 소스 경로 추적으로 확정한 코드 사실 · **[D]** 기존 문서 인용.
- 1차 실측 대상: `RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828` (parity59).

---

## 0. 세 줄 결론

> **R-N4 — 폐합.** 생산 writer가 세대 식별자 없이 한 경로를 매 MC pass마다 덮어쓰고 sha 사이드카를
> 함께 갱신하던 구조를 잘라냈다. 산출물은 `<path>.iter%03d`로 세대가 이름에 박히고, 이미 있는 세대는
> **덮어쓰지 않고 FATAL**이며, 소비자 6종 전부 `read_fluor_matrix`의 **키워드 필수 인자**
> `expected_iteration`을 통과하지 않으면 TypeError로 죽는다. 음성 대조(사이드카까지 함께 갱신되는
> 세대 교체)를 실제로 재현했고 `sha256sum -c`는 **여전히 OK**인데 소비자는 거부한다 [F].

> **R-T2 — 계기 신설 + 그 과정에서 T2 실험 설계 자체를 흔드는 사실 1건 발견.** 같은 세대에
> `(n_l, n_u, τ, χ_l 기여, ε_l, S_l^pop, S_l^used)`를 per-(line, shell)로 기록하는 LCMFLP01을
> **읽기 전용 replay**로 구현했다(생산 assemble 무수정). replay가 `cs->chi_line`을 **bitwise** 재현했는지를
> 매니페스트에 남겨 "같은 세대"가 주장이 아니라 검사 대상이 된다. **결정적 발견: parity59 설정
> (EPAY=2, HOTF=0, SMIN=5)에서 s≥5의 thin bin은 S_fixed가 `w_n·(bf_eta + χ_line,th·B(T_e))`로
> 통째로 재구성되어 assemble이 만든 η_line이 버려진다** [F-code]. s8 UV(600–3000Å) bin의 **최소 30.6%**,
> s16 이상은 **98.7–100%**가 그 상태다 [F]. η만 바꾸는 T2/B2-lane은 그 셀에서 원리적으로 무효이며,
> 새 덤프는 셀별 EPAY 처분(disposition) 열로 이것을 처음 가시화한다.

> **R-N2 — 편입 + 잣대 결함 2건 추가 등재.** ff-heat 활성화를 밴드 원장에 편입했고(bf 사이트와 동형),
> 그 과정에서 (a) **밴드 원장 `d_ma_fate_hist`가 co-evolve/pure-CMFGEN 레인에서 reset·download·print
> 어디에도 걸려 있지 않아 통째로 死物**이고(=N2 수리만으로는 검증불가 고아), (b) **선 활성화 카운트
> `EVCH_MA_ACT_BB`가 무-cap 센서스 어디에도 없어** T4가 활성화 클래스 비중을 CAP 41.2% 접두부에서만
> 잴 수 있었다는 것을 확인했다 [F-code]. 셋을 한 계약("모든 매크로원자 활성화 클래스는 무-cap 센서스에
> 계수된다")으로 묶어 수리했다.

---

## 1. 제출물

| 파일 | 내용 | 라인 |
|---|---|---|
| `patches/instr_rn4_generation_contract.patch` | 생산 writer 세대 스탬프/덮어쓰기 금지 + 소비자 6종 fail-closed 계약 + 음성 대조 | +239 |
| `patches/instr_rt2_linepop_dump.patch` | LCMFLP01 population-native line dump(읽기 전용 replay) + fail-closed 소비자 + 오프라인 fixture/배터리 + Makefile 타깃 | +1149 |
| `patches/instr_rn2_activation_census.patch` | ff-heat 밴드 원장 편입 + 활성화-클래스 무-cap 센서스 + co-evolve 레인 관측 경로 | +66 |
| `patches/instr_expected_changes.txt` | 세 rung의 기대 변경집합(사전등록) | — |

**적용 검증** [F]: 세 패치를 pristine 트리에 **임의 순서 3가지 전부**(`rn2 rt2 rn4`, `rn4 rn2 rt2`,
`rt2 rn4 rn2`) 적용 성공, 결과 트리가 검증된 작업 트리와 **바이트 동일**, 세 패치 모두 **역적용
(`patch -R --dry-run`) 깨끗**.

**빌드 검증** [F]:
- `nvcc -O2 -arch=sm_90 -std=c++14 -Xcompiler -fopenmp -DLUMINA_HAS_CUDA_BF_GEMM` 로 `lumina_cuda` **전체 링크 성공**(34.4 s, 2.77 MB).
- `gcc -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L` 로 `lumina_cmfgen.c` 경고 수 **baseline 25 → patched 25**(신규 경고 0).
- `make selftest_cmf_linepop_dump` 성공.

**신규 clamp/floor/fallback: 0.** 추가한 분기는 전부 (i) 게이트 판정, (ii) fail-closed FATAL,
(iii) 진단 누산이다. 물리량을 잘라내거나 대체하는 코드는 없다.

---

## 2. R-N4 — 계기 산출물의 세대 계약

### 2.1 진단 확정 (사고 재구성) [F, F-code]

parity59 env가 `LUMINA_FLUOR_MATRIX_DUMP=${RUN_DIR}/fluor_matrix_iter10`,
`LUMINA_PURE_CMFGEN_ITER=12`, `LUMINA_MC_COEVOLVE=1`이다 [F]. co-evolve 루프는
`cuda_fluor_matrix_dump(it)`를 **매 iteration 호출**하고(`cuda.cu:8694`), writer는 `g_fluor.path`
**한 경로**에 쓴다. 즉 운전자가 이름으로 표명한 계약(iter10)과 writer의 행위(it=0..11 전부 기록)가
애초에 어긋나 있었고, 마지막 기록 it=11이 최종 상태로 남았다.

탐지 실패의 세 층 [F-code]:
1. writer가 payload와 `.sha256` 사이드카를 **함께** 갱신 ⇒ `sha256sum -c`는 영구 PASS. **무결성은 지키지만 동일성은 못 지킨다.**
2. `read_fluor_matrix`가 헤더 `iteration`을 dict에 담기만 하고 **검증하지 않음**.
3. 소비자 계약이 **한 곳에만** 있었다 — `emiss_e12_preregister.py:68`의 `matrix.header["iteration"] != 10`.
   이건 **사전등록 시점 1회 검사**였고, 실제 소비 시점(E10 applicator·E12 diagnose·E13 audit)에는 재검사가 없다.
   *(FABLE_UV_T3T4 §10의 "소비자 3종 어디에도 계약 없음"을 이 지점에서 정정한다: 계약은 1/4개에 존재했으나
   사전등록 단계에만 있었고 적용 단계에는 없었다. 타이밍상 E-시리즈는 iter10을 읽은 것이 맞고 교체는 09:45,
   E13 종료 09:32 이후다 ⇒ 결론 무효화는 아니고 "정합이 우연"이라는 판정이 유지된다.)*

### 2.2 수리 ① 생산 writer (`src/lumina_cuda.cu`)

| 변경 | 내용 |
|---|---|
| `FluorMatrixHost` | `want_iter`(-1=전 pass), `allow_overwrite` 추가 |
| `cuda_fluor_matrix_init` | `LUMINA_FLUOR_MATRIX_ITER` 파싱(형식 위반 시 FATAL — CHIETA의 `LUMINA_CMF_FROZEN_CHIETA_ITER` 규약 그대로), `LUMINA_FLUOR_MATRIX_OVERWRITE` 파싱 |
| `cuda_fluor_matrix_dump` | ① 선택 iteration 아니면 스킵 로그 후 return 0 ② **출력 경로 = `<path>.iter%03d`** ③ 이미 존재하면 **FATAL**(override env 없으면) ④ sha 사이드카·배너 전부 새 경로 기준 |

설계 판단 2건을 명시한다.
- **세대 식별자를 파일명에 박는 쪽**과 **덮어쓰기 금지** 중 하나가 아니라 **둘 다** 넣었다.
  이름만 바꾸면 같은 런을 같은 디렉토리에 재실행할 때 다시 조용히 교체되고, 덮어쓰기 금지만 넣으면
  it=0의 산출물이 남고 it=10이 FATAL로 죽어 캡처가 불가능해진다.
- `LUMINA_FLUOR_MATRIX_OVERWRITE` 탈출구는 **clamp가 아니라 운영 스위치**다(재실행 편의). 기본은 거부이고,
  발동 시 경로가 배너에 찍힌다.

### 2.3 수리 ② 소비자 계약 (choke point 1곳)

`scripts/emiss_e11_fluor_matrix.py`의 `read_fluor_matrix`가 유일한 리더이므로 계약을 **거기에만** 걸었다.

```python
def read_fluor_matrix(path, *, expected_iteration, expected_sha256=None,
                      non_contract_override=False, ...)
CONTRACT_ITERATION = 10
```

- `expected_iteration`은 **기본값 없는 키워드 전용 인자** ⇒ 계약을 빠뜨린 소비자는 **TypeError로 즉사**한다
  (실측: `read_fluor_matrix() missing 1 required keyword-only argument: 'expected_iteration'` [F]).
  이것이 "소비자마다 검사를 복붙"보다 강한 이유: 새 소비자가 생겨도 계약을 우회할 수 없다.
- 계약 이탈(≠10)은 `non_contract_override`를 명시해야 하고, CLI는 `cmf_chieta_check.py`와 **같은 규약**으로
  `NON-CONTRACT` 출력 + **rc=2**, 실패는 rc=1이다.
- `expected_sha256`은 **사이드카와 독립**으로 바이트 동일성을 못박는다(사이드카는 공범이므로 자기증명 불가).
- 공용 CLI 헬퍼 `add_matrix_contract_args()` / `read_fluor_matrix_from_args()`를 추가하고,
  `emiss_e10_apply_redistribution.py`·`emiss_e12_preregister.py`·`emiss_e12_diagnose.py`·`emiss_e13_index_audit.py`에
  `--expected-matrix-iteration` / `--expected-matrix-sha256` / `--matrix-non-contract-override`를 붙였다.
  `emiss_t5_rank1.py`는 이미 있던 사후 검사(읽고 나서 비교)를 **리더 안으로 이동**시켰다 —
  "읽고 나서 검사"는 리더가 이미 파일을 소비한 뒤라 계약이 아니다.
- `emiss_e12_preregister.py`의 하드코딩 `iteration != 10`은 리더 계약으로 흡수해 제거했다(이중 진실 제거).
- 파생 fixture 왕복(`emiss_e13_index_audit.write_mirrored_matrix`, `emiss_t5_rank1` 대리 행렬)은
  부모 세대를 **명시적으로 상속**하고 `non_contract_override=True`를 단다 — 파생물이 계약을 세탁하지 못하게.

### 2.4 음성 대조 — 생산 사고의 정확한 재현 [F]

`scripts/emiss_e11_seeded_fixture.py`에 **generation-swap** 대조를 추가했다. payload를 다른 세대로
교체하면서 **사이드카도 같이 갱신**한다(생산 writer와 동일 행위).

```
sha256sum -c formal_matrix_generation_swap.bin.sha256   ->  OK      (사고 재현)
read_fluor_matrix(expected_iteration=10)                ->  FAIL: matrix generation mismatch:
                                                            header iteration=11, expected 10
read_fluor_matrix(expected_iteration=11, sha pin=이전)  ->  FAIL: matrix identity changed
read_fluor_matrix(expected_iteration=11) (override 없음) ->  FAIL: non_contract_override 필요
```

실제 아티팩트로도 확인 [F]:
```
$ python3 scripts/emiss_e11_fluor_matrix.py $RUN/fluor_matrix_iter10
FAIL: matrix generation mismatch: header iteration=11, expected 10 ...           rc=1
$ ... --expected-matrix-iteration 11 --matrix-non-contract-override
"contract_status": "NON-CONTRACT", "iteration": 11, "sha256": "08ff3312..."      rc=2
```

### 2.5 기대 변경집합 (사전등록)

| 대상 | ON(게이트 armed) | OFF(`LUMINA_FLUOR_MATRIX_DUMP` 미설정) |
|---|---|---|
| 산출 경로 | `…/fluor_matrix_iter10` → `…/fluor_matrix_iter10.iter000`…`.iter011` (또는 ITER 선택 시 1개) | 변화 없음(파일 없음) |
| stdout | `[FLUOR-MATRIX] armed path=….iter<NNN> … iter_select=N overwrite=0` 1줄 + 스킵 줄 | **변화 없음** |
| 전송/패킷 | **변화 없음** (writer는 커널 밖 호스트 코드) | 변화 없음 |
| 기존 소비자 CLI | 새 플래그 3개(기본값=계약) | — |
| rc 규약 | `emiss_e11_fluor_matrix.py`: 실패 2→**1**, NON-CONTRACT=**2** | — |

**주의 1건(운영):** rc 규약 변경으로 이 CLI를 rc==2로 판정하던 래퍼가 있으면 재조정 필요.
저장소 내 자동 소비처는 없음(문서의 수동 명령뿐) [F].

---

## 3. R-T2 — population-native χ+η 시험을 가능케 하는 덤프

### 3.1 왜 기존 계기로는 T2가 불가능했나 [F-code]

- LCMFCE01(χ,η 캡처)은 (shell, coarse bin) 단위다. 선 숲이 **합쳐진 뒤**의 값만 남고
  어떤 선이 얼마를 냈는지, 어떤 인구에서 나왔는지는 없다.
- E4/E5 B·B2 lane은 `cmfgen_assemble_impl`의 선 루프에서 **η만** 교체한다(`lumina_cmfgen.c:879`).
  χ는 `opac->tau_sobolev`에서 오므로 bitwise 동일 ⇒ **단일인자 시험이 아니다**.
- 그런데 χ는 이미 population-native다: NLTE 매핑된 선의 τ는
  `nlte_update_tau_sobolev`(`plasma.c:17057`)에서
  `τ = SOBOLEV_COEFF·f_lu·λ·t_exp·n_l·[1 − (g_l n_u)/(g_u n_l)]`로 쓰인다 [F-code].
  즉 T2가 실제로 필요한 것은 "χ를 population-native로 **바꾸는**" 것이 아니라
  **인구 → τ → χ_bin → η의 사슬을 per-line으로 노출**해서 오프라인에서 재조립·교체·검증하는 것이다.

### 3.2 설계 — 생산 assemble을 건드리지 않는 읽기 전용 replay

CHIETA 덤프 사이트(`cuda.cu:8028`, `it == LUMINA_CMF_FROZEN_CHIETA_ITER`) **바로 옆**에서 발화한다.
`cmfgen_assemble`(`cuda.cu:7957`)과 이 지점 사이에는 `cmfgen_solve_J`와 J-damping만 있고
둘 다 `chi_line`/`chi_line_th`/`chi_abs`/`chi_tot`를 쓰지 않는다 [F-code] ⇒ **같은 세대가 구조적으로 보장**된다.

핵심 설계 결정:

| 결정 | 이유 |
|---|---|
| assemble 내부 계측이 아니라 **외부 replay** | 생산 함수의 기계 코드가 ON/OFF 양쪽에서 그대로다. 핫 루프(2.58M lines × 50 shells)에 분기 하나도 추가하지 않는다 |
| replay가 `cs->chi_line`을 **bitwise 재현했는지 매니페스트에 기록** | "같은 세대"를 주장이 아니라 **왕복항등식**으로 만든다. 선 루프는 OMP 병렬이 아니므로(pragma 0건 [F-code]) 같은 순서·같은 식이면 bitwise 재현이 성립한다 |
| 선택 밖 파장의 선도 **χ 누산에는 포함**, 행 기록만 제외 | 그래야 왕복항등식이 성립한다. fixture가 이 성질을 직접 검사한다 |
| 선택(`LUMINA_CMF_LINEPOP_SHELLS`) **필수** | 전 셸 기록은 19,246,925 line-shell(=A-lane 매니페스트 `active_line_shell_count` [F])로 GB급 |
| 상한 초과 시 **절단이 아니라 FATAL** + 실측 행수/MiB 출력 | 조용한 절단은 잣대 오염 |
| `cont_only`/`frozen_morph`/`EPAY_TAUEFF>0` 상태는 **거부** | replay가 생산 숲과 달라지거나(앞 둘) 재현 불가능한 게이트(뒤 하나)이므로 오라벨 아티팩트를 만들지 않는다 |

### 3.3 LCMFLP01 v1 스키마

```
magic "LCMFLP01" | endian | version
iteration u64 | field_generation u64
n_shells u32 | n_bins u32 | n_sel u32 | n_lines_sel u32 | n_rows u64
t_exp f64 | lam_lo f64 | lam_hi f64
eps_phys u32 | src_nlte u32 | epay u32 | epay_smin u32
epay_taubin,epay_hotf,eps_floor,eps_cap,line_eps,eps_uv,line_gate  f64 ×7
selected shell ids                    u32 × n_sel
per-shell (T_e, T_rad, n_e, dr)       f64 × 4n_sel
nu[], dnu[]                           f64 × 2n_bins
chi_line_replay / chi_line_th_replay / eta_line_replay   f64 × 3·n_sel·n_bins
EPAY disposition                      u8  × n_shells·n_bins
line-static table (80 B)  { line_id, bin, Z, ion, g_lo, g_up, nlte_lo, nlte_up,
                            nu_l, lambda_cm, A_ul, f_lu, E_lo_eV, E_up_eV }
rows (76 B)               { line_slot, shell_slot, flags,
                            tau_used, tau_from_pops, n_lower, n_upper,
                            S_l_pop, S_l_used, eps_l, w }
```
`flags`: `NLTE_ION | POPS_DEFINED | SL_POP | SL_FALLBACK | STIM_CLAMPED | TAU_ROUNDTRIP`.
`tau_from_pops`는 `nlte_update_tau_sobolev`와 **같은 식**으로 재계산한 값이고 `TAU_ROUNDTRIP`은
그것이 실제 소비된 τ와 bitwise 일치하는지다 ⇒ **인구 ↔ τ 사슬의 왕복 검증이 행 단위로 남는다.**

사이드카 `.manifest.json`은 CHIETA와 같은 형식으로 `sha256`, `iteration`, `field_generation`,
`chi_line_roundtrip_bitwise`(+ `max_abs`), `chi_line_th_comparable`, `epay_disposition_counts`,
게이트 전량을 기록한다.

### 3.4 크기 추정과 상한 [F]

- s8, τ>1e-12, λ∈[1000,3000] Å 실측 행수 = **475,330** (`$RUN/cmf_fine_linedump_s8.csv` 624,245행 중) [F].
  600–1000 Å이 더해지므로 셸당 **0.5–0.9 M행**으로 본다.
- 행 76 B ⇒ **38–68 MB/shell**. 3셸(8,16,45) + line-static(union 1.0–1.5 M × 80 B = 80–120 MB) ⇒ **총 ~200–330 MB**.
- 기본 상한 `LUMINA_CMF_LINEPOP_MAXROWS=4,000,000`(= 행 304 MB). 초과 시 실측 행수와 MiB를 찍고 **FATAL**.
- 나머지 배열은 셸당 3×1000 f64 = 24 kB, disposition 50×1000 = 50 kB로 무시 가능.

### 3.5 **부수 최대 발견 — η_line은 s≥5 thin bin에서 버려진다** [F-code, F]

`cmfgen_assemble_impl`의 EPAY 재정규화(`lumina_cmfgen.c:1121-1172`):

```c
if (epay && s >= epay_smin && epay_tau_arr[s] < epay_tau) {
    int hot_regime = (Te > epay_hotf * plasma->T_rad[s]);
    if (epay >= 2 && acc_w > 0.0 && hot_regime) {
        for (b) { if (thick) continue;            /* legacy Kirchhoff */
                  w = bf_get_eta(...) + chi_line_th*B(Te);
                  cs->S_fixed[idx] = wn * w / chi_t; }   /* <-- eta_ln 소멸 */
```

parity59는 `EPAY=2, EPAY_SMIN=5, EPAY_TAUBIN=10, EPAY_HOTF=0` [F].
`HOTF=0` ⇒ `hot_regime = (Te > 0)` = **항상 참**. `EPAY_TAUEFF` 미설정 ⇒ `epay_tau_arr[s]=0 < 2.0` ⇒ **게이트 항상 통과**.
따라서 s≥5의 thin bin(=`(χ_abs+χ_line,th)·dr ≤ 10`)에서 **S_fixed는 assemble이 만든 η_line을 전혀 쓰지 않고**
`w_n·(bf Milne η + χ_line,th·B(T_e))`로 재구성된다.

동결 payload에서 잰 **확실히-thin** 하한(`χ_tot·dr ≤ 10` ⇒ `χ_abs+χ_line,th ≤ χ_tot`이므로 충분조건) [F]:

| shell | 전 bin thin | **UV 600–3000 Å bin thin** | 비고 |
|---|---|---|---|
| 0 | 19.5% | 0.0% | s<SMIN=5 ⇒ EPAY 미적용 |
| 3 | 38.7% | 9.5% | 동상 |
| 5 | 40.1% | 14.1% | 여기부터 EPAY 적용 |
| **8** | 45.1% | **≥30.6%** | E-시리즈/T5 기준 셸 |
| 16 | 65.8% | **≥98.7%** | |
| 25 | 68.4% | **100%** | |
| 49 | 100% | **100%** | |

**귀결.** (i) η만 교체하는 실험(E4 B-lane, E5 B2-lane, 그리고 "population-native η" 형태의 T2)은
s8 UV의 최소 30.6%, 외곽 전체에서 **원리적으로 무효**다 — 바꾼 값이 소비되지 않는다.
(ii) 그 셀에서 결정론 선 방출률은 `χ_line,th·B(T_e)`, 즉 **ε_l 가중 불투명도 × 국소 Planck**이며
S_l(인구비, 형광 담지)은 어디에도 들어가지 않는다. 이는 `docs/FABLE_UV_T3T4.md` §4.3의
"S_fixed는 T_e 열원"이라는 실측과 정확히 같은 말이고, 그 **기전을 코드 경로로 특정**한 것이다.
(iii) 그래서 LCMFLP01은 셀별 disposition(0=legacy / 1=thick-exempt / 2=rate-shape 대체 / 3=스칼라 재정규화)을
**1급 열로** 싣는다. 이 열 없이 오프라인 T2를 돌리면 "η를 바꿨는데 아무 일도 안 일어났다"를
물리 결론으로 오독하게 된다.

**한계 정직 기재:** 셸별 EPAY 정규화 스칼라 `w_n`은 덤프 시점에 재현 불가능하다
(`acc_abs`가 assemble 당시의 **lagged J**를 쓰는데 그 사이 `cmfgen_solve_J`가 J를 갱신했다).
매니페스트에 `epay_scale_not_reproducible: true`로 명기했고, 값이 필요하면 stdout의
`[CMF-EPAY] scale s0=… s25=… s38=… s49=…` 4개를 써야 한다. `w_n`은 셸 전체에 곱해지는 스칼라라
대역 간 상대 비교에서는 소거된다.

### 3.6 오프라인 검증 — 실행 결과 [F]

`make selftest_cmf_linepop_dump` + `python3 scripts/cmf_linepop_roundtrip_selftest.py`:

```json
{ "verdict": "PASS",
  "reference": {"bitwise": true, "rows": 4, "sha256": "a82edf4e…"},
  "seeded_replay_drift_refused": "chi_line round trip is not bitwise … (max_abs=3.08e-33)",
  "row_cap_refused":  "[CMF-LINEPOP][FAIL] selection yields 4 rows … > MAXROWS=3; narrow …",
  "generation_swap_refused": "generation mismatch: got iteration 11, expected 10",
  "generation_swap_override_status": "NON-CONTRACT",
  "payload_tamper_refused": "sidecar sha256 mismatch",
  "missing_selection_refused": "LUMINA_CMF_LINEPOP_SHELLS is required …",
  "out_of_range_shell_refused": "shell '7' out of [0,2)",
  "epay_rate_shape_cells": 4,
  "epay_taueff_refused": "LUMINA_CMF_EPAY_TAUEFF>0: … cannot be reproduced here" }
```

음성 대조 8종 전부 발화한다. 특히:
- **1 ulp 재현 드리프트**(`cs.chi_line`을 `nextafter`로 1 ulp 밀기)를 소비자가 거부한다 —
  왕복항등식이 장식이 아니라 게이트다.
- fixture는 λ 창 **밖**의 선 1개를 일부러 넣어, 그 선이 행에는 안 들어가면서 χ 누산에는 들어가는지를 검사한다.
- EPAY 대조: `SMIN=1`로 shell 1만 rate-shape 대체 ⇒ `eta_line_reaches_S_fixed_fraction = 0.5`로 잡힌다.

**OFF 중립성** [F-code]: 추가 코드 전량이
`if (dump_path && *dump_path)` → `if (it == wanted)` → `if (lp_path && *lp_path)` 3중 게이트 안에 있다.
게이트 미설정 시 `cmfgen_dump_line_populations`는 호출되지 않으며, `cmfgen_assemble_impl`·
`cmfgen_solve_J`·전송 커널의 코드는 **한 줄도 바뀌지 않았다**(패치 hunk가 그 함수들에 없다).

---

## 4. R-N2 — ff-heat 경로 센서스 편입

### 4.1 진단 정밀화 [F-code]

`docs/FABLE_UV_T3T4.md` N2는 "MA-FATE 센서스에 ff-heat 부재"였다. 소스 추적 결과 **두 개의 서로 다른 원장**이 있고 상태가 다르다:

| 원장 | 정체 | ff-heat 포함? | parity59에서 살아 있나? |
|---|---|---|---|
| `d_ma_fate_hist[8×8]` (진입밴드×출구밴드) | `d_ma_fate_record`(bf, `:6417`) / `_zi`(line, `:5626`) | **아니오** ← N2 | **아니오** — co-evolve 레인에 reset/download/print 전무 |
| `d_census_fate[shell][5]` → `lumina_census_ma_fate.csv` | `d_census_accumulate`의 fx 버킷 | 출구는 포함(`EVCH_KPKT_COLLEXC`→fx=0), **진입 클래스는 미분해** | 예 |

즉 N2를 문자 그대로 고쳐도 **읽는 사람이 없다**(검증불가 고아). 추가로:

> **신규 N6 [F-code]** — `d_ma_fate_hist`는 classic MC 루프(`cuda.cu:10029` reset, `:10141` aggregate,
> `:10689` print)에서만 다뤄진다. pure-CMFGEN/co-evolve 런은 `cuda.cu:9881`에서 `return 0`하므로
> **한 번도 초기화·다운로드·출력되지 않는다.** parity59 stdout에 `[MA-FATE]` 밴드표 0건 [F].

> **신규 N7 [F-code]** — 활성화 클래스의 **분모가 무-cap 센서스에 없다**.
> `EVCH_MA_ACT_BB`(선 흡수 = 활성화의 99.7%)는 emch/kx/hx/fx **어느 버킷에도 안 걸린다**.
> 그래서 T4는 활성화 클래스 비중을 CAP 41.2% 접두부에서만 잴 수 있었고 §6.1에서 6.5% 보정을 붙여야 했다 [D].

### 4.2 수리 — 한 계약, 세 조각

**계약: "모든 매크로원자 활성화 클래스는 무-cap 센서스에 계수되고, 활성화 밴드 원장은 그 레인에서 읽힌다."**

| # | 변경 | 위치 | 성격 |
|---|---|---|---|
| a | ff-heat 종료 직전 `d_ma_fate_record(comov_nu_ff, exit_nu)` | `cuda.cu` ff-heat 분기 끝 | bf 사이트(`:6417`)와 **동형**. RNG 추출 0, 패킷 상태 쓰기 0 |
| b | `d_census_act[shell][3]` = {bb, bf, ffh} + `lumina_census_ma_activation.csv` | `d_census_accumulate` / `cuda_census_dump_csv` / `cuda_census_reset` | **가산적**. 기존 열 무변경. **신규 이벤트 사이트 0** (세 채널은 이미 이벤트를 낸다) |
| c | co-evolve 레인에 `cuda_ma_fate_reset/zi_reset` + `macro_atom_fate_reset` + `download_and_aggregate` + `print` | `if (event_log_on)` 블록 안 (`:8624`, `:8963`) | N6 수리. **이벤트 로그 미armed면 stdout 무변화** |

기존 열을 건드리지 않은 이유는 코드 주석이 명시한 캠페인 규율이다 —
`cuda.cu:4542-4548`은 `EVCH_KPKT_COLLEXC_BB`를 kpkt-exit 히스토그램에서 **의도적으로 제외**하며
"collexc 열을 캠페인 중간에 조용히 재정의하는 것"을 금지한다. 새 신호는 **새 파일**로 낸다.

### 4.3 기대 변경집합 — 실측 사전등록 [F]

기존 `$RUN/lumina_events.bin`(400 M 레코드, 41.2% 접두부)을 `d_ma_fate_band_from_nu`의
호스트 복제로 replay해서 **추가될 카운트를 그대로 계산**했다(스크래치 `n2_ffheat_delta.py`).

```
HEAT_FF entries in the 41.2% prefix : 165
resolved entry->exit pairs          : 165   (미해결 0)
exit channels                       : {KPKT_COLLEXC: 165}   ← 전량 선광자 탈출
entry-band histogram                : [3, 2, 6, 2, 4, 11, 30, 107]
delta d_ma_fate_hist[entry,exit] 비영 셀:
  [0,7]+3 [1,7]+2 [2,0]+2 [2,7]+4 [3,7]+2 [4,6]+1 [4,7]+3 [5,0]+1 [5,6]+1 [5,7]+9
  [6,0]+3 [6,2]+1 [6,3]+1 [6,7]+25 [7,0]+14 [7,1]+1 [7,2]+2 [7,4]+1 [7,7]+89
```

- armed iteration 전체(무-cap)의 기대 총 delta = **403** (`$RUN/lumina_census_heating.csv` ff 열).
  기록 사이트가 무조건 실행되므로 정확히 403이다.
- 새 CSV `lumina_census_ma_activation.csv`의 기대값: `act_ff` 합 = **403**, `act_bf` 합 = **1,291,431**
  (heating 센서스와 항등), `act_bb` 합 = 신규 관측량(접두부 추정 199,492,039의 무-cap 대응값) [D/F].
- **부작용 없음의 근거:** 새 원장 3개(밴드 히스토그램·활성화 센서스·새 CSV) 외에
  `lumina_census_{kpkt_exit,heating,ma_fate,ma_fate_elem,emission}.csv`는 **바이트 불변**이어야 한다.
  기존 버킷 스위치에 손대지 않았으므로 이것이 회귀 판정 기준이다.

### 4.4 **부수 발견 N8 — 밴드 정의가 UV를 못 본다** [F]

`d_ma_fate_band_from_nu`의 band 7은 **λ ≥ 10000 Å과 λ < 1700 Å을 한 통에 넣는 catch-all**이다.
ff-heat 165건 실측:

| | 총 | λ<1700 Å | λ≥10000 Å |
|---|---|---|---|
| 진입 band-7 | 107 | 3 | 104 |
| **출구 band-7** | **137** | **137** | **0** |

출구 λ: min 699 Å, **중앙값 1526.5 Å**, max 8334 Å ⇒ **ff-heat 활성화의 83%(137/165)가 FUV 선광자로 나간다.**
그런데 밴드 원장은 이것을 진입 IR(band 7)→출구 band 7로 적어, **"열화된 IR 광자 → FUV 상향변환"이라는
바로 그 신호가 대각 성분으로 위장**된다. memory의 Axis-2(광구 FUV 초과 = 상향변환)와
"mc/cs 39×@1526 Å" Co IV 깔때기 전선에 직접 걸리는 잣대 결함이다.
**수리는 이번 범위 밖**(밴드 경계 변경은 parity 비교 대상인 `ma_fate_zihist.csv`를 재정의한다) —
아래 §7 대장에 등재한다.

---

## 5. 공통 규율 — 게이트 커버리지 감사

| rung | 추가 라인 | 게이트 밖(무조건 실행) | 판정 |
|---|---|---|---|
| R-N4 | src +62 / scripts +177 | 0 | 전량 `if (g_fluor.on)` 뒤. 호스트 코드, 커널 무관 |
| R-T2 | src +599 / scripts +544 | 0 | `dump_path` → `it==wanted` → `lp_path` 3중 게이트 |
| R-N2 | src +66 | **2줄** (ff-heat의 `d_ma_fate_record` 호출) | 아래 명시 |

**R-N2의 유일한 무게이트 hunk를 숨기지 않는다.** ff-heat 분기 끝의
`d_ma_fate_record(comov_nu_ff, ma_exit_comov_nu_ff)`는 bf 쌍둥이(`:6417`)와 마찬가지로 게이트가 없다.
근거와 범위:
- `d_ma_fate_record`는 device 전역에 `atomicAdd` 2회를 할 뿐이다. **RNG 추출 없음, 패킷 상태(r, mu, nu, energy, next_line_id) 쓰기 없음** ⇒ 전송 스트림·스펙트럼은 불변.
- 바뀌는 것은 오직 `d_ma_fate_hist` 내용이며, 그것이 **이 rung의 수리 대상**이다.
- classic MC 레인에서는 그 원장이 실제로 출력되므로 `macro_atom_fate_print` 표가 ff-heat 만큼 늘어난다(예상치 §4.3).
- 음성 대조: ARTIS-PARITY D5 ff-heat OFF(`chi_ff_heat=0` ⇒ `cont_chan`이 1이 되지 않음) 런에서는
  이 사이트가 **도달 불가** ⇒ 원장 delta 0. 게이트 없는 변경의 무해성이 이 대조로 판정된다.

**신규 clamp/floor/cap: 0** (세 패치 전수). 추가된 수치 처리는 없고, 새 상수는
`CENSUS_NACT=3`, `CMF_LINEPOP_ROW_BYTES=76`, `CMF_LINEPOP_LINE_BYTES=80`,
기본 `MAXROWS=4e6`, 기본 λ창 `[600,3000] Å` 뿐이다 — 전부 **버퍼/선택 파라미터**이지 물리량 제한이 아니다.

---

## 6. 전 검증 재현 명령

```bash
RT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828

# --- 패치 적용 (임의 순서 가능; 역적용도 깨끗) ---
cd $RT && for r in rn4 rt2 rn2; do patch -p1 --forward < patches/instr_${r}_*.patch; done

# --- 빌드 ---
make selftest_cmf_linepop_dump
nvcc -O2 -arch=sm_90 -std=c++14 -Xcompiler -fopenmp -DLUMINA_HAS_CUDA_BF_GEMM -Isrc \
  -o /tmp/lumina_cuda_instr src/lumina_cuda.cu src/lumina_bf_gemm.cu src/lumina_nlte_gemm.cu \
  src/lumina_nlte_assemble.cu src/lumina_cmf_solve.cu src/lumina_atomic.c src/lumina_plasma.c \
  src/lumina_element_wide.c src/lumina_cmfgen.c -lm -lcublas -Xcompiler -fopenmp

# --- R-N4: 음성 대조 + 실제 아티팩트 ---
python3 scripts/emiss_e11_seeded_fixture.py --out-dir /tmp/n4_fixture
sha256sum -c /tmp/n4_fixture/formal_matrix_generation_swap.bin.sha256      # OK (사고 재현)
python3 scripts/emiss_e11_fluor_matrix.py $RUN/fluor_matrix_iter10          # rc=1, 거부
python3 scripts/emiss_e11_fluor_matrix.py $RUN/fluor_matrix_iter10 \
        --expected-matrix-iteration 11 --matrix-non-contract-override       # rc=2, NON-CONTRACT

# --- R-T2: 왕복 + 음성 대조 8종 ---
python3 scripts/cmf_linepop_roundtrip_selftest.py                           # verdict PASS

# --- R-N2: 기대 변경집합 (기존 이벤트 로그 replay, ~25 s) ---
python3 <scratch>/n2_ffheat_delta.py $RUN/lumina_events.bin

# --- §3.5 EPAY thin 분율 (동결 payload) ---
python3 <scratch>/epay_thin.py
```

> 스크래치 스크립트(`n2_ffheat_delta.py`, `epay_thin.py`)는 생산 트리에 넣지 않았다.
> 전자는 상설 회귀 가치가 있으므로 **`scripts/`로 승격 검토**를 상신한다(§8-5).

---

## 7. 이번 작업에서 등재된 신규 항목

| # | 내용 | 좌표 | 처분 |
|---|---|---|---|
| **N6** | `d_ma_fate_hist`가 co-evolve/pure-CMFGEN 레인에서 reset/download/print 전무 ⇒ 원장 死物 | `cuda.cu:10029/10141/10689`(classic 전용), `:9881` return | **본 패치에서 수리**(R-N2 c) |
| **N7** | `EVCH_MA_ACT_BB`가 무-cap 센서스 어느 버킷에도 없음 ⇒ 활성화 분모가 CAP 접두부에만 존재 | `cuda.cu:4506-4601` | **본 패치에서 수리**(R-N2 b, 신규 CSV) |
| **N8** | `d_ma_fate_band_from_nu` band 7이 λ<1700 Å과 λ≥10000 Å을 합침 ⇒ IR→FUV 상향변환이 대각으로 위장. ff-heat 출구 83%가 λ<1700 Å(중앙값 1526.5 Å) | `cuda.cu:4106-4118` | **등재만.** 밴드 경계 변경은 `ma_fate_zihist.csv` 재정의 ⇒ 별도 결정 |
| **N9** | EPAY≥2 hot 분기가 s≥SMIN thin bin의 `S_fixed`를 재구성해 assemble η_line을 폐기. parity59 s8 UV의 ≥30.6%, s16+ 98.7–100% | `lumina_cmfgen.c:1136-1156` | **등재 + 계기화**(LCMFLP01 disposition 열). η-교체 실험 설계의 전제조건 |
| **N4 정정** | 소비자 계약이 "3종 어디에도 없음"이 아니라 `emiss_e12_preregister.py:68`에 **사전등록 단계 1회**만 있었음. 적용 단계 재검사 부재라는 결론은 유지 | — | 본 문서에 정정 기재 |

---

## 8. 상신

1. **R-N4를 즉시 채택하고 E-시리즈 재현 명령을 갱신할 것.** 현재 디스크의 행렬은 iteration 11이므로
   T5 재현은 `--expected-iteration 11 --matrix-non-contract-override`가 필요하고, 이는
   "우리는 계약 밖 세대를 쓰고 있다"를 **명시적으로 기록**한다. 이것 없이는 어떤 E-후속도 증거력이 없다 [D].
2. **N9를 먼저 판정할 것 — T2 발주보다 앞선다.** η만 바꾸는 시험이 s8 UV의 최소 30.6%에서 무효라면,
   T2의 올바른 형태는 "η 교체"가 아니라 **`χ_line,th`(=ε_l 가중)와 `w_n` 정규화가 결정론 UV 방출률을
   지배하는지"의 시험**이다. LCMFLP01은 그 재설계에 필요한 `ε_l`·`χ_l` 분해를 이미 싣는다.
3. **R-T2 덤프 발주는 셸 3개(8, 16, 45) × λ 600–3000 Å로 시작.** 예상 ~200–330 MB. `LUMINA_CMF_LINEPOP_DUMP`는
   기존 CHIETA 캡처 런에 **환경변수 1개 추가**로 붙고, 게이트 미설정 시 완전 무해하므로 다음 캡처에 동승시키면 된다.
4. **R-N2는 계측 부채 상환분으로 R-N4와 함께 묶어 처리.** 물리 무변경, 신규 파일 1개, 기존 CSV 불변.
5. **회귀 배터리 편입 2건:** (a) `scripts/cmf_linepop_roundtrip_selftest.py` (b) ff-heat 활성화 delta 계산기.
   memory의 "계측 부채가 진짜 반복비용"에 정확히 해당한다.
6. **N8은 UV 캠페인의 잣대 결함으로 별건 상신.** 밴드 재정의 시 기존 `ma_fate_zihist.csv` 비교 가능성이
   깨지므로, 새 밴드 세트를 **추가 열**로 내는 방식(N7과 같은 가산적 패턴)을 권한다.

---
---

# 부록: N3 수리 설계 (구현은 이번 범위 밖 — 설계와 영향 추정만)

## N3.1 결함의 정확한 형태 [F-code]

**분모(분배함수) — `compute_partition_functions`, `plasma.c:1896-1947`**
```c
double T_part = plasma->T_rad[s];  double W = plasma->W[s];
if (parity) { if (Te > 0.0) { T_part = Te; W = 1.0; } }   /* ARTIS-PARITY B3 */
...
Z_total = Z_meta(T_part) + W * Z_non_meta(T_part);        /* = Z_LTE(T_e), W=1 */
```

**분자(준위인구) — 소비처 4곳, 전부 `T_rad`/`W` 그대로**
| 좌표 | 소비 |
|---|---|
| `plasma.c:2636-2682` `compute_tau_sobolev` | 비-NLTE 선 τ |
| `plasma.c:7145-7150` `compute_bf_opacity` | `chi_bf` 전 연속체 |
| `plasma.c:8836-8846` `bf_rate_pop` | radeq bf 가열 인구 |
| `bf_gemm.cu:82-95` | 위의 GPU 사본 |

전부 `n_k = n_ion · w_k · g_k · e^{−E_k/kT_rad} / Z_part`, `w_k = 1`(metastable) 또는 `W`.

따라서
```
Σ_k n_k = n_ion · Z_neb(T_rad, W) / Z_LTE(T_e)  ≡  n_ion · R_norm(ip, s)
Z_neb(T_rad,W) = Σ_meta g e^{−E/kT_rad} + W Σ_nonmeta g e^{−E/kT_rad}
Z_LTE(T_e)     = Σ_meta g e^{−E/kT_e}   +   Σ_nonmeta g e^{−E/kT_e}
```
`R_norm = 1`은 `T_rad = T_e` **그리고** `W = 1`일 때뿐이다. parity59에서는 둘 다 거짓
(T_rad ≡ 10470.09 K 핀, W = 0.298→0.0108) [D] ⇒ **Σ_k n_k ≠ n_ion, 전 이온·전 셸.**

이것은 T_rad 핀과 **독립**이다: 핀을 실제 색온도로 고쳐도 `W ≠ 1`이고 `T_rad ≠ T_e`인 한 남는다.
(핀 수리는 T_rad를 T_e 쪽으로 당기므로 **완화**될 뿐이다.)

**부호는 이온·셸별로 갈린다.** `Z`는 온도에 단조 증가이므로
- `T_rad,pin < T_e`인 셸(s0 10470 vs 21228, s3 vs 15668, s8 vs 12004 [D])에서는
  `Z_neb ≤ Z_LTE(T_rad) < Z_LTE(T_e)` ⇒ **`R_norm < 1` 확정, 인구·τ·χ_bf 과소평가.**
- `T_rad,pin > T_e`인 셸(s16 vs 9334, s25 vs 8476)에서는 metastable 항은 커지지만
  non-metastable 항이 `W ≪ 1`로 눌리므로 **부호 미정** — 이온별 준위 구조(metastable 비중)로 갈린다.

## N3.2 수리안 3개와 선택

| 안 | 내용 | 판정 |
|---|---|---|
| **A. 정규화 이중화 (권장)** | 분배함수를 두 벌 유지: `partition_functions`(=Z_LTE(T_e), **Saha/이온화 전용**, B3의 원래 의도)와 신규 `partition_functions_level`(=`Z_neb(T_rad,W)`, **준위인구 전용**). 소비처 4곳이 후자를 쓴다 | **채택 권고.** by-construction으로 `Σ_k n_k = n_ion`이 **정확히** 성립. B3의 이온화 폐합은 손대지 않는다. 비용: `[n_ion_pops × n_shells]` 배열 1개(153×50×8 B = **61 kB**) + 같은 함수 안 루프 1회 추가 |
| B. 소비처도 T_e·W=1로 | `n_k = n_ion g e^{−E/kT_e}/Z_LTE(T_e)` | 정규화는 맞지만 **물리 변경**(비-NLTE 선 τ와 χ_bf에서 희석 효과 제거). 정규화 수리가 아니라 다른 결정 |
| C. 사후 재정규화 | `n_k /= R_norm` | A와 수치적으로 동일하나 소비처마다 잊을 수 있는 형태. **비권장** |

**A안의 검증 경로(현재 부재 — 이게 결함이 살아남은 이유):**
첫 assemble에서 이온·셸 전수로 `|Σ_k n_k / n_ion − 1| < 1e-12`를 확인하고 1회 배너로 찍는
불변식 검사를 붙인다. 위반 시 최대 위반 (ip, s, 값)을 출력하고 **fail-closed**.
`cmf_chieta`/`fluor_matrix`가 각각 `eta_decomposition_bitwise` / column-closure로 하는 것과 같은 규약이다.

## N3.3 영향 추정 — 어디에 얼마나

| 소비처 | 스케일링 | UV 진폭 영향 추정 | 근거 |
|---|---|---|---|
| 비-NLTE 선 τ (`plasma.c:2636`) | τ ∝ n_lower ∝ R_norm | **무시 가능.** UV(1700–3000 Å) Στ에서 비-NLTE 이온 몫 = **1.546e-5 (0.0015%)** | [D] FABLE_UV_T3T4 §5.1 |
| `chi_bf` (`plasma.c:7145`, `bf_gemm.cu:82`) | χ_bf ∝ R_norm (선형) | **직접적이나 진폭에는 1차 무감.** χ_bf 지배 대역은 B0(600–1000 Å; s16에서 χ_abs/χ_tot=0.609)뿐이고, 흡수 지배 대역은 국소 극한 `J → η_fixed/χ_abs → B(T_e)`라 χ_bf에 1차 무감 | [D] §4.3 |
| `bf_rate_pop` → radeq bf 가열 (`plasma.c:8836`) | 가열률 ∝ R_norm | **여기가 본체.** T_e 근 탐색의 가열 항이 통째로 R_norm배 밀린다 | [D] §2.1 A3/A4 |
| 광이온율 (`chi_bf` 소비 경로) | Γ_bf ∝ R_norm | 이온화 평형 전반. memory의 "n_e 1.92×" 미결 전선과 같은 층 | [D] |

**⇒ 예상 처분: UV 진폭 후보가 아니라 이온화·열 장부 항목**(FABLE_UV_T3T4 §5.2의 P1/P2/P3와 같은 칸).
이것은 T3에서 T_rad 핀이 받은 것과 **같은 성격의 판정**이며, 우연이 아니다 — 둘 다 같은 소비처 집합을 먹는다.

## N3.4 발주 전 필수 계량 (오프라인, 런 0)

수리 착수 전에 **`R_norm(ip, s)`를 실측**한다. 필요한 것은 전부 이미 디스크에 있다:
- 준위 데이터(E_eV, g, metastable) — `data/tardis_reference_toy06_19p48d*/`
- `T_e`, `T_rad`, `W` per shell — `$RUN/lumina_plasma_state.csv`
- 이온 밀도 — `$RUN/lumina_ion_pops.csv`

산출: 153 ion pop × 50 shell의 `R_norm` 표, `min/median/max`, 그리고
`Σ_ip n_ion·|R_norm − 1|` 기준의 **원소별 순위**.

**사전등록(측정 전):** 지배 이온(Fe/Co/Ni II–IV)의 `|R_norm − 1|` 중앙값이
- **< 5%** ⇒ 장부 등재만 하고 수리는 다른 국면과 묶어 처리,
- **5–50%** ⇒ A안 구현을 계측 부채 상환분으로 발주,
- **> 50%** ⇒ 이온화·열 전선(n_e 1.92×, b_k 2–20×)의 **후보 공범으로 승격**하고 T_e 근 탐색과 함께 재판정.

이 계량은 CPU 수 분이며, memory의 "★잣대부터 감사 — 'X=N배'는 가설(분모 실측 증명 먼저)"과
"틀린 값은 조용히 대장 기재 — 튜닝 금지" 양쪽에 맞는 순서다.
