# I20 공기파장 수리 — 계약 · 기대 변경집합 (착수 전 등록)

2026-08-06 (운전석). 근거 = `validation/layer1/L1_R5_VERDICT.md` (I20 확정).
**이 문서는 첫 src/scripts 편집 전에 작성됐다** — 기대 변경집합 사전등록 규율.

---

## 1. 수리 계약 (물리 계약 1개)

> **선 진동수는 준위 에너지에서, `f_lu`는 원본 osc의 f열에서, `A_ul`은 그 둘에서
> CMFGEN과 동일한 공식으로 산출한다. 원본 osc의 λ열과 A열은 소비하지 않는다.**

근거: CMFGEN 자신이 그렇게 한다. λ열·A열은 읽지도 않는다.

```
genosc_v6.f:205       FEDGE   = (E_ion - E_lev)·c·1e-15         ! 진공
genosc_v6.f:278-286   f 만 읽고 f·A·Lam 토큰 3개 건너뜀 → i-j
genosc_v6.f:313-317   EINA(J,I) = T1·f·g_i/g_j·(FEDGE_i-FEDGE_j)²
                      T1 = OPLIN/EMLIN·TWOHCSQ = 8π²e²/(m_e c³)·1e30
```

수리 후 Lumina 선자료:

```
ν      = (E_up − E_lo)·c                        [준위 에너지, 진공]
λ_vac  = c/ν
f_lu   = 원본 osc f열 (부호는 abs — genosc_v6.f:305-309 이 그렇게 한다)
A_ul   = 8π²e²/(m_e c³) · f_lu · (g_lo/g_up) · ν²
B_ul   = c²/(2hν³)·A_ul ,  B_lu = (g_up/g_lo)·B_ul       [기존 유지, 정확]
```

**이것은 근사 대체가 아니라 by-construction 정확 구현이다.** Edlén 공기→진공 변환을
덧붙이는 안은 기각한다 — 근사를 하나 더 얹고(어느 판? 어느 T·P?) A↔f 무모순도
해결하지 못한다.

### 전하 상수

production 덱은 **참값** `e = 4.803204713e-10 esu`를 쓴다
(SI-2019: `e[C]=1.602176634e-19` 정확, `c` 정확 ⟹ `e[esu]=e[C]·c/10`).
현행 `4.80320425e-10`(계보 불명)에서 교체. CMFGEN의 `4.80320427e-10`(CODATA-2006)도
참값이 아니므로 채택하지 않는다. **대조 검증에서만** CMFGEN 값을 써서 by-construction
일치를 증명한다(user 08-06 방침: 검증엔 CMFGEN 값, production은 참값).

## 2. 수리 지점 (뿌리 2곳)

| # | 파일:행 | 현재 | 수리 후 |
|---|---|---|---|
| 뿌리1 | `expand_atomic_data_cmfgen.py:676,686` | `lam ← t['lam_A']`(공기), `nu ← c/lam` | `nu ← (E_up−E_lo)·c`, `lam ← c/nu` |
| 뿌리1' | `expand_atomic_data_cmfgen.py:678` | `A ← t['A']`(원본 A열) | `A ← 8π²e²/(m_e c³)·f·(g_lo/g_up)·ν²` |
| 뿌리2 | `finalize_cmfgen_ref_npy.py:88-91` | `f_lu ← (m_e c/8π²e²)·λ²·(g_u/g_l)·A_ul` | **삭제** — 원본 f 유지 |

`:677`의 `fs.append(t['f'])`는 **이미 옳다**(원본 f열). 뿌리2가 그것을 덮어쓰고 있었다.

## 3. ★기대 변경집합 (판정 전 등록 — 이 밖의 변화는 회귀다)

### 3.1 변해야 하는 것

| 산출 | 기대 변화 | 근거 |
|---|---|---|
| `wavelength`, `wavelength_cm`, `nu` | 오염 45이온의 λ>2000Å **635,169선**에서 **+2.73~2.85e-4**(진공으로) | 굴절률 실측치 |
| " (그 외 선) | ≤ ~1e-8 (준위 에너지 반올림) | 비오염 이온 실측 vac_r 3–13e-9 |
| `f_lu` | 원본 f열로 복귀. 현행 대비 λ>2000Å 구간 **+5.6e-4**, 그 외 **~1e-5** | C1 실측 |
| `A_ul` | f 유래 재계산. 현행(=원본 A열) 대비 중앙 **~1.1e-5** | C4 실측 |
| `B_lu`, `B_ul`, `f_ul` | 위에서 유도되므로 연동 변경 | — |
| `line_id` | **재배열 가능** — ν 내림차순 정렬이 바뀐다 | `build_lines:696 argsort(-nu)` |
| `macro_atom_data.csv`, `transition_probabilities.npy`, `tau_sobolev.npy`, `line2macro_level_upper.npy` | line_id 재배열에 연동 | 같은 빌드에서 함께 생성되므로 내부 일관 |

### 3.2 변하면 안 되는 것 (음성대조)

| 산출 | 기대 |
|---|---|
| 선 개수 | **2,220,953 불변** (필터 조건 무변경) |
| `atomic_number, ion_number, level_number_lower/upper` 집합 | 불변 |
| `levels.csv` 전량 | **불변** (준위는 건드리지 않는다) |
| `abundances.csv`, `density.csv`, `geometry.csv` 등 조성·구조 | 불변 |
| 비오염 13이온의 λ | ≤1e-8 |

### 3.3 by-construction 인수 게이트 (이것이 PASS 조건)

| G | 검사 | 기대 |
|---|---|---|
| **G1** | 새 덱 `A_ul` vs `8π²e_CMFGEN²/(m_e c³)·f·(g_lo/g_up)·ν²` (CMFGEN 상수로 재계산) | **1.843e-7 ± 1e-12** — 오직 e 상수차. `(e_참/e_CMF)²−1 = 2×9.21e-8`. 상수를 맞추면 기계정밀도로 0이어야 한다(G1b) |
| **G1b** | 동일 검사를 `e = e_CMFGEN`으로 | **≤ 1e-14** (by-construction 일치 증명) |
| **G2** | 새 덱 `f_lu` vs 원본 osc f열 | **exact 0** |
| **G3** | 새 덱 `ν` vs 준위에너지 유래 ν | **≤ 1e-12** |
| **G4** | 공기 census 재실행: `deck_nu/levelenergy_nu − 1` | **전 이온 ≤ 1e-8**, 오염 이온 **0** |
| **G5** | 선 개수·이온별 선 수 | 구 덱과 동일 |
| **G6** | 음성대조: 뿌리1만 고치고 뿌리2 남기면 G2가 FAIL | FAIL 시연 필수 |

**G6는 음성 대조 의무다** — 게이트가 주입 결함으로 FAIL을 시연해야 PASS 자격이 있다.

## 4. 덱 불변 규율

`_ftos` 정본은 **덮어쓰지 않는다**. 새 덱
`data/tardis_reference_toy06_19p48d_sivcaiv_vac`을 새 드라이버로 생성한다
(`deck_regen_r4_ftos_driver.py` 선례). 0-G GEN-GUARD 유지.

## 5. 물리 영향 예고 (판정런 전 등록)

- **선 위치**: 오염 선 635,169개(덱 28.6%)가 **+82~85 km/s** 이동. SN Ia 선폭
  ~10⁴ km/s의 0.8%. narrow-band 잣대·P-Cygni 최소 위치에 직접.
- **선 세기**: τ ∝ f·λ 이므로 5.6e-4 수준 — 무시 가능.
- **바이트-parity**: **깨진다.** 회귀 기준선 재수립 필요.

## 6. 미해결 위험

`line_id` 재배열이 하류 고정 자산(픽스처·회귀 기준선·저장된 npy 캐시)을 무효화할 수
있다. 새 덱은 별도 디렉터리이므로 구 덱 기반 자산은 그대로 살아 있으나,
**새 덱으로 갈아타는 시점에 전량 회귀가 필요**하다.
