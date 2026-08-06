# 층 1 I3a — Co IV σ(ν) 판정

2026-08-06 (운전석 L1 실측). **검수 미실시.**
구 기록: *"Co IV 46,827/51,411 (91%) 불일치 — 잔류·최악. I1·I2c와 합쳐 Υ·A_ul·σ
세 축 전부 불일치하는 유일 이온"*.

---

## 1. 구 91%의 정체 — **교차-vintage 아티팩트**

Co IV 광이온 자료는 vintage 사이에 **표현 방식 자체가 다르다**:

| vintage | 준위 | 데이터 쌍 | 교차 타입 |
|---|---|---|---|
| `18oct00/phot_data.dat` | 407 | 1,221 | **전부 type 1** (Seaton 3파라미터) |
| `19apr23/phot_data_A` | 404 | 40,025 | **264 type 1 + 140 type 20**(표 형식 OP) |

파일 자신이 18oct00 을 *"Very crude photoionization cross-sections for CoIV"* 라고 적는다.
3파라미터 적합식과 40,025점 표를 대조하면 91% 불일치는 **보장된 결과**이지 발견이 아니다.
(같은 구조를 `expand_atomic_data_cmfgen.py:281-283` 이 Ni II 에 대해 이미 기록해 뒀다:
*"18oct00 is an all-Seaton table, which 19apr23 replaced with a 2166-point tabulated OP table"*.)

## 2. 동일 vintage 실측 — σ 는 **~1e-7 로 일치**

CMFGEN 이 σ 를 만드는 식을 원본에서 확인하고(`newsubs/sub_phot_gen.f:412-419`) 덱의
구운 값과 직접 대조했다.

```
CMFGEN  PHOT = CONV_FAC · A₀ · ( A₁ + (1−A₁)·RU ) · RU^A₂ ,  RU = EDGE/FREQ
Lumina  σ    = 1e-18   · s0 · ( β  + (1−β )·ru ) · ru^s_exp , ru = ν_th/ν
        EDGE = GS_EDGE(I) + EXC_FREQ  (sub_phot_gen.f:173; Co IV 는 EXC_FREQ=0)
```

형태·상수·edge 정의가 by-construction 일치한다. 실측:

| 덱 (vintage) | 준위 | 점 | 구운 σ vs **빈 평균** | vs **점 샘플** | 1e-6 초과 |
|---|---|---|---|---|---|
| `_ftos` (18oct00 = jnu4 런) | 1,000 | 398,612 | **1.029e-07** | 3.894e-07 | 1,000 = **준위당 1점** |
| `_ophys` (19apr23, type1 부분) | 557 | 243,132 | **1.029e-07** | 3.894e-07 | 557 = **준위당 1점** |

**두 덱 모두 자기 런의 자료로부터 CMFGEN 식을 ~1e-7 로 재현한다.**

### 준위당 1점의 정체 = 내 비교자의 임계빈 처리

1e-6 을 넘는 점이 **정확히 준위 수와 같다**. 최대 상대차가 0.9926/1.0 인 것이
증거다 — 한쪽이 거의 0 이라는 뜻이고, 이는 **ν_th 를 품은 빈**이다.
내 비교자는 그 빈을 통짜 사다리꼴로 평균해 ν<ν_th 구간(σ=0)을 함께 넣었고,
베이커는 A2-05 에서 확립한 **부분빈 정확 적분**을 쓴다.
⟹ **덱 결함이 아니라 비교자 결함이다.**

## 3. 판정

| 축 | 값 |
|---|---|
| posedness | WELL |
| outcome | **MATCH** — 동일 vintage 에서 ~1e-7 |
| kind | — (구 DIFFER 는 VINTAGE 교란) |
| disposition | **CLOSE**(불일치로서) |
| evidence_status | VALID (CMFGEN 소스 대조 + 양 덱 실측) |

**I1·I19 와 같은 결론이다**: 진술된 불일치는 덱과 런이 서로 다른 vintage 를 보던
잣대 문제였고, 덱-런 종속 방침(user 08-06)이 구조적으로 제거한다.

⟹ 구 기록의 *"I1·I2c와 합쳐 Υ·A_ul·σ 세 축 전부 불일치하는 유일 이온"* 이라는
Co IV 서사는 **세 축 중 두 축(Υ·σ)이 vintage 아티팩트로 해소**됐다. I2c(A_ul)는 미검.

## 4. 미결 — 이 측정이 **덮지 못한 것**

- ~~**type 20(표 형식) 140 준위 미검증**~~ → **소스 대조로 해소(2026-08-06)**:

  | 구간 | CMFGEN (`sub_phot_gen.f:393-406`) | Lumina (`expand:1229-1243`) | 판정 |
  |---|---|---|---|
  | 마디 사이 | `U` 선형보간, `U=FREQ/EDGE` | `np.interp(nu,…)` | **동등** (`nu_pts=energy·ν_th` 이므로 ν 선형 = U 선형) |
  | 마지막 마디 **위** | `CROSS_A(N)·(NU_NORM(N)/U)³` — **ν⁻³ 감쇠** | `right=sig_pts[-1]` — **상수 유지** | **상이** |
  | 첫 마디 **아래** | `CROSS_A[1]` | `left=0.0` | **상이** |

  경계 2건은 **의도적 미변경**이다 — `expand_atomic_data_cmfgen.py:1014-1018` 이
  *"NOT changed here (VERDICT 3.2, 'latent, not live') … Measured effect −0.24% (S II s0).
  Kept so this bake is a single-variable change"* 라고 이미 기재해 뒀다.
  ⚠ 이 −0.24% 는 **선행 certification 의 측정치를 인용**한 것이고 이번에 재측정하지
  않았다. 고빈도에서 Lumina σ 가 CMFGEN 보다 크므로 방향은 **Γ 과대**다.
- **I3 전체(다른 이온) 미재측정.** 구 I3 상위 기여는 Ni II 33.3% · Co III 27.1% ·
  S III 13.1% · S IV 5.8% · S V 5.7% 였다. Ni II 는 위 코드 주석이 이미
  18oct00 Seaton ↔ 19apr23 type-20 교체를 지목하므로 **같은 vintage 아티팩트일 가능성이
  높으나 실측 전에는 단정하지 않는다.**
- **임계빈 비교자 수정**: 부분빈을 반영한 비교자로 바꾸면 1e-6 초과가 0 이 되는지 확인.

## 5. 측정 자체의 기록

- 파서의 type-1 파라미터는 `PhotEntry.sigma_Mb` 에 들어간다(`params` 아님).
- phot 파일은 `Split J levels = False` 라 **term 단위**이고 덱은 J 분해다 —
  `[J]` 접미를 떼어 term 으로 매핑해야 한다. 첫 시도가 0 매칭이었던 원인.
