# 층 1 재측정 #1 — 원자자료 계보(vintage) 대조

> ⚠ **이 문서의 §"이 결과가 바꾸는 것" 이하 결론부는 Fable L3 적대 검수에서
> OVERREACH 판정을 받았다. 확정 문장은 `L1_REMEASURE_VERDICT.md` 를 보라.**
> 측정 자체(27이온 동일 원본)는 CONFIRMED 다.

2026-08-06 (운전석 측정). 산출 = `validation/layer1/L1_VINTAGE_CROSSWALK.json`.
**개정 12 체제**(구현·측정=운전석, 검수=Fable) — 이 측정은 아직 **검수 전**이다.

---

## 왜 이것을 먼저 재는가

`docs/OUTSIDE_LOOP_POOL.md` 층 1 절의 1차 잣대 감사(2026-08-04)가 이렇게 경고한다:

> **층 1 수치 대부분이 구 덱(`_sivcaiv`)의 것이다.** `_ftos`에서는 분모가 통째로 바뀐다.
> ⟹ I2·I2a–I2d·I3·I3a–I3c·I17의 분모가 통째로 바뀌었다. **재측정 없이는 어느 것도
> 확정도 제거도 불가.**

재측정을 하려면 먼저 **무엇과 무엇을 비교하는지**를 확정해야 한다. Lumina 덱과 CMFGEN
런이 애초에 **같은 원본에서 왔는지**가 그 출발점이다.

## 측정

- 입력 A: `data/tardis_reference_toy06_19p48d_sivcaiv_ftos/atomic_vintage_manifest.csv`
  (이온별 osc/phot/col 원본 경로와 vintage를 기록)
- 입력 B: `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/*_F_OSCDAT` 의 `realpath`
- 방법: 이온별로 두 경로의 `/atomic/` 이하 상대경로를 문자열 대조

### 결과

```
same_source: 27   different_source: 0   compared: 27
```

**CMFGEN 런이 쓰는 27개 이온 전부가 Lumina `_ftos` 덱과 동일 원본 파일을 가리킨다.**

대표 예:

| 이온 | 양쪽 공통 원본 |
|---|---|
| Co III | `COB/III/18oct00/coiii_osc.dat` |
| Co IV | `COB/IV/18oct00/coiv_osc.dat` |
| Fe III | `FE/III/19apr23/osc_data` |
| Fe IV | `FE/IV/18oct00/feiv_osc.dat` |
| S II | `SUL/II/19apr23/osc_data` |
| S III | `SUL/III/3oct00/siiiosc_fin.dat` |

대조: 구 덱 `_sivcaiv`는 Co III에 `COB/III/19apr23/osc_data`를 썼다
(`validation/cmfgen_toy06_19p48d/analysis/rates_certification/run_log.txt` 의 `[C2]` 행).
즉 **`_ftos`는 계보 불일치를 해소한 덱이다.**

## ★이 결과가 바꾸는 것

**층 1 I2(A_ul)·I3(σ(ν)) 판정의 성격이 바뀐다.**

- 구 덱에서 잰 불일치(I2: 880,406선 중 75,075 · I3: 3,953,894점 중 1,233,529)는
  **서로 다른 vintage 사이의 비교**였다. 값이 다른 것이 당연할 수 있다.
- `_ftos`에서는 **같은 파일**에서 왔으므로, 남는 불일치는 **변환·임포트 결함**이다.
- 따라서 임계도 달라진다. 1차 감사가 지적한 "원자료 유효숫자 5자리인데 임계 `1e-6`은
  10× 엄격 — 반올림을 세고 있을 수 있다"는 **서로 다른 원본을 비교할 때의 문제**다.
  같은 원본 변환이라면 기대는 **exact 또는 ULP**이고, 벗어나면 그 자체가 결함이다.

## 미결 — 다음 측정 (#2)

계보가 같다는 것은 **선언**이다(manifest가 그 경로를 적고 있다). 실제로 그 파일에서
변환됐는지는 별개이며, 그것이 다음 측정이다.

**측정 #2 (변환 무결성)**: 각 이온의 원본 osc 파일을 파싱해 `(f, g, λ)`에서 `A_ul`을
CMFGEN 규약대로 계산하고 `line_list.csv`의 `A_ul`과 대조한다. 같은 원본이므로
기대는 exact/ULP이며, 벗어나는 선의 수와 크기가 I2의 **새 정본 수치**가 된다.
σ(ν)도 동형으로 phot 파일 기준 재측정한다(I3).

선례: `rates_certification`이 구 덱에서 같은 방식의 provenance 검사를 했고
`sum|D − C2|/sum|D| = 7.162e-15`를 얻었다("shipped binary IS the baker applied to that
vintage"). `_ftos`에서 이 검사를 다시 해야 한다.

## 측정 자체의 함정 (자기 기록)

v1 측정에서 **Si II를 S III 파일과 오매칭**했다(그리고 Si V ↔ S IV). CMFGEN 종 표기가
원소기호가 아니기 때문이다 — `Si→Sk`, `Ni→Nk`, `II→2`. v1은 `different_source: 2`를
보고했고 그것은 **측정 버그**였다. v2에서 CMFGEN 표기표를 명시해 해소했고, JSON의
`note` 필드에 남겼다.

교훈: 이름 규칙을 추정하지 말고 **실제 파일 목록을 먼저 나열**할 것
(`ls *_F_OSCDAT` = 27개가 정답 집합이었다).
