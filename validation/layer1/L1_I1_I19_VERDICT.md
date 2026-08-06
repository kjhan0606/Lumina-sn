# 층 1 I1(충돌강도 Υ) · I19(충돌표 상실) — 판정

2026-08-06 (운전석 L1 실측). **검수 미실시.**
계기: 덱-런 종속 방침(user 08-06) 적용 후 `_ophys` 덱이 7이온의 충돌표를 되찾자,
"그것이 진짜 자체 자료인가"가 미결로 남았다.

---

## 1. 실측 — 7이온의 충돌자료 계보

`_ophys` 덱(=O-PHYS 런 vintage 19apr23)이 수입한 7이온:

| 이온 | n_src | n_mapped | 출처 (파일 헤더 원문) |
|---|---|---|---|
| **Co IV** | 4,455 | 4,455 | **`Zha96_FeIII_col`** — *"Using FeIII values?"* (물음표 원본) |
| Co II | 105 | 105 | Storey, *Co II infrared forbidden lines* |
| Si IV | 378 | 378 | Liang/Whiteford/Badnell 2009, Na-sequence |
| Ca IV | 3 | 3 | Nahar 2023, CaIV |
| Ni II | 2,775 | 2,775 | Bautista 2002, NiII |
| Ni III | 2,485 | 2,485 | Ramsbottom + Storey 2023 |
| Ni IV | 1,225 | 1,128 | Fernández-Menchero, Smyth, Ramsbottom 2019 |

**⟹ 7이온 중 대용은 Co IV 하나뿐이다.** 나머지 6은 이온별 실제 출판 자료다.
(Ni IV 의 1,128/1,225 는 손실이 아니다 — `n_dropped=0`, 내역
`accum_repeats=93 · self_pairs=2 · bidir_slots=2` 로 병합·자기쌍 처리다.)

## 2. Co IV = Fe III 대용 — 실측 확증

```
Co IV rows                4,455        Fe III rows   22,139
무순서 쌍 공통             4,357        Co IV 에만     98
값 비트 동일               4,357        최대 절대차    0.000e+00
Co IV 커버리지            97.80%
```

구 감사 수치(4,357 동일 / 98 이름만 상이 / 최대 절대차 0)를 **독립 재현**했다.

**이것은 사고가 아니라 의도된 등전자 대용이다.** Co IV(Co³⁺)와 Fe III(Fe²⁺)는
둘 다 **3d⁶ 등전자**이고, 준위명도 `3d6_5De[3]` 처럼 공유한다. 원본 파일이
스스로 출처를 Zhang(1996) Fe III 로 적고 물음표까지 달아 뒀다.

### 측정 자체의 함정 (자기 기재)

1차 비교에서 **106쌍**만 일치해 구 감사와 어긋났다. 원인은 내 비교자였다 —
두 파일의 **전이쌍 표기 순서가 서로 반대**다:

```
Fe III :  3d6_5De[4] - 3d6_5De[3]    2.85E+00 2.29E+00 ...
Co IV  :  3d6_5De[3] - 3d6_5De[4]    2.85E+00 2.29E+00 ...   ← 같은 값
```

무순서 쌍(frozenset)으로 바꾸자 4,357 로 재현됐다. 파서 결손은 없었다
(선언=파싱: Fe III 22,139 / Co IV 4,455).

## 3. ★I1 판정 — 진술된 불일치는 **해소**, 물리 유보는 **강화**

풀의 I1 근거는 *"Lumina 4,455 대 **CMFGEN 런의 Co IV tabulated 전이 = 0개**"* 였다.
그 0 의 정체가 밝혀졌다:

```
jnu4 런 Co IV col :  COB/IV/18oct00/col_guess.dat  →  "0  !Number of transitions"
```

**vintage 아티팩트였다.** 덱-런 종속 방침 적용 후 실측:

| 덱 | Co IV Υ | 대응 런 | 상태 |
|---|---|---|---|
| `_vac` (jnu4 vintage) | **0** (SKIP) | jnu4 도 **0** | **identity** |
| `_ophys` (19apr23) | **4,455** | O-PHYS 도 같은 파일 | **identity** |

⟹ **Lumina 대 CMFGEN 불일치로서의 I1 은 해소된다.** 7이온 전부 동일하다.

**그러나 물리 유보는 남고, 오히려 성격이 나빠진다.** Co IV 의 Υ 가 Fe III 값이라는
사실은 **CMFGEN 도 공유**한다. 즉 이 결함은 **CMFGEN 대조로는 영원히 검출되지 않는다.**
ε = `C/(C+Aβ)` 이므로 Co IV 의 열화·산란 분기가 통째로 편향돼 있어도
두 코드가 나란히 틀린다.

| 축 | 값 |
|---|---|
| posedness | WELL |
| outcome | **MATCH (identity)** — 불일치 항목에서 제거 |
| kind | — |
| disposition | **CLOSE**(불일치로서) + **공유 맹점 대장에 신규 기재** |
| evidence_status | VALID (양방향 실측, 원본 헤더 자백) |

**신규 항목 제안 — 공유 맹점(shared blind spot)**: "두 코드가 같은 근사를 쓰기 때문에
대조로 검출 불가능한 것". Co IV Υ 가 첫 항목이다. 이 부류는 CMFGEN 대조가 아니라
**외부 독립 앵커**(NORAD/TOPbase/Badnell, 또는 등전자 계열 스케일링의 타당성 평가)로만
판정할 수 있다 — `feedback_no_unverifiable_orphans` 의 규율이 그대로 적용된다.

## 4. ★I19 판정 — **CLOSE**

풀은 I19 를 *"census 만으로는 개선/퇴행을 판정할 수 없다 — identity metric 과
physics-change metric 필요, 판정 보류"* 로 두었다. 그 두 metric 이 이제 필요 없다.

`_ftos` 에서 7이온 mapped 가 11,329 → 0 이 된 것은 **상실이 아니라 정합이었다** —
그 덱이 따르던 jnu4 런 자신이 그 이온들에 tabulated Ω 를 갖고 있지 않다
(`col_guess.dat` 가 0 전이 선언). 덱이 런보다 많이 갖고 있던 구 상태가 오히려 불일치였다.

덱-런 종속 방침 아래에서 질문 자체가 소멸한다: **덱은 언제나 자기 런과 같다.**

⟹ **disposition = CLOSE.** 단 §3 의 공유 맹점 항목은 별도로 살아 있다.

## 5. 부수 확인

- 구 감사가 "4,455 가 사라진 것은 identity 개선"이라 적은 것은 **jnu4 기준에서만 참**이다.
  O-PHYS 기준에서는 반대로 **있어야** identity 다. 절대적 판정이 아니라 런 상대적이다.
- 이 사실이 덱-런 종속 방침의 필요성을 사후 정당화한다 — 자료의 옳고 그름을 덱 홀로
  판정할 수 없고, **어느 런과 비교하는가**가 먼저 정해져야 한다.

## 6. 미결

- **공유 맹점 대장 신설**과 Co IV Υ 등재 (외부 앵커 조사 필요)
- 등전자 대용의 타당성 자체: Fe III → Co IV 스케일링이 Υ 에 얼마나 유효한가.
  이온 전하가 다르므로(+2 대 +3) 충돌 단면적은 통상 스케일링이 필요하다 —
  **스케일링 없이 값을 그대로 쓴 것**이 원본 파일의 상태다. 크기 평가 필요.
