# E7 — UV 과잉의 새 표적 3종 분해

판정일: 2026-08-02 (Asia/Seoul)  
범위: 기존 payload/오라클의 산술 소비만 수행; 신규 런·수송 solve·clamp·commit 없음

## 0. 결론

**E6의 `S=eta_total/chi_total` 11.9773×는 물리적 고정 방출 초과열성이
아니라 `eta_coherent=chi_coherent*J_ours`로 자기 `J`를 다시 읽은 항이
만든 진단 순환이다.** s8 BALL에서 이 항은 `S_total`의 99.9809433%이고,
제거한 `S_fixed/CMFGEN`은 `0.00228248`이다. 즉 E6의 11.98× source-ratio는
완전히 폐합되지만, payload의 `J_producer`가 CMFGEN `J` 대비 크다는 독립 사실은
남는다. 순환 제거는 **E6의 source-ratio 해석을 정정**하지만 기존 수송장의 원인을 설명하지
않는다.

세 표적의 판독은 다음과 같다.

| 순위 | 표적 | 12×를 설명할 크기인가 | 판정 |
|---:|---|---|---|
| 1 | `eta_coherent` 자기-독해 | E6 `S_total` 11.98×의 99.981%를 산술 폐합 | **RESOLVED — 진단 순환** |
| 2 | UV opacity 결손 | s0→s8 `J` 감쇠의 차이는 B0/B1에서 10.3/9.59× 효과와 같은 크기 | **UNRESOLVED** — CMFGEN UV-bin `chi` 부재 |
| 3 | 상위준위 초열성 | s8 Fe III/Co III 대표 `b_u=2.32–4.86`; 12×에 못 미침 | **단독 설명 기각** |

물리적 `J` 과잉의 남은 원인은 **경계조건/산란 operator → EPAY post-shape
→ 빈 폭·선 투영** 순으로 다음 산술 표적으로 지정한다. 첫 항은 고정 source가
CMFGEN의 0.23%인 반면 저장된 source의 99.98%가 산란 `J`인 구조 때문이다.
EPAY는 payload가 post-EPAY 합계만 저장해 per-line 비교와 연결이 안 되고, 빈 투영은
1208 Å 빈 폭 6.40431 Å와 `dnu` 평균이 강한 선의 국소 방출률을 퍼뜨리기
때문이다. 이 세 후보는 이번 차터에서 테스트하지 않았다.

## 1. 입력·시점·산술 계약

권위 payload는
`/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.{A,B,B2}`이다.
A/B/B2는 50 shell × 1000 bin, iteration/generation 10/10, post-damping이며 `chi_total`,
`chi_coherent`, `J_producer`는 lane 간 bitwise 동일이다. payload SHA-256은 A
`ac62eae5...4011`, B `95b87203...4582`, B2 `775c9e84...9f5a`다. 전체 값은
`validation/emiss_e7/summary.json`에 있다.

산출물은 다음이다.

- `validation/emiss_e7/band_shell_fixed_opacity.csv`: 6 band × 50 shell = 300행의
  `S_fixed`, E6 `S_total`, opacity 범위, `tau_out`, CMFGEN `J`/MEANOPAC 매핑.
- `validation/emiss_e7/j_depth_indicators.csv`: s0→s8, s8→s20 `Delta ln J`.
- `validation/emiss_e7/line_departure_proxies.csv`: 선 20개 × s0/s8/s20 = 60행의
  `n_u`, absolute-within-ion LTE `b_u`, Sobolev tau 및 방출률 비 대리값.
- `validation/emiss_e7/summary.json`: 기계 판독 요약.

시점제약이 중요하다. iteration-10 payload는 iteration-9 끝의 plasma/population으로
조립됐지만 `lumina_levelpop.csv`, `lumina_ion_pops.csv`,
`lumina_plasma_state.csv`는 iteration-11 종료 시점이다. 따라서 payload 자체로
계산한 source/opacity 항등은 권위값이지만, `n_e*sigma_T`, per-line `b_u`, tau,
epsilon 비교는 **final-state proxy**다. exact capture-epoch per-line 비는
**UNRESOLVED**로 남겨두었다.

## 2. 순환 제거: `S_fixed=eta_fixed/chi_total`

payload가 저장한 항등은 bitwise로

```text
eta_total    = eta_fixed + eta_coherent
eta_coherent = chi_coherent * J_producer
S_total      = eta_total / chi_total
S_fixed      = eta_fixed / chi_total
```

이다. 따라서 E6의 `S_total` 비교는 `J_producer`를 source에 다시 넣은 좌표였다.
동일한 exact-edge overlap 가중, CMFGEN integral-preserving bin average로 다시 계산한
canonical s8은 다음과 같다.

| band [Å] | E6 `S_total/CMF` | `S_fixed/CMF` | total/fixed | coherent 분율 |
|---|---:|---:|---:|---:|
| B0 600–1000 | 33.7681 | 0.0206077 | 1,638.61 | 99.938973% |
| B1 1000–1500 | 32.3246 | 0.0120322 | 2,686.50 | 99.962777% |
| B2 1500–2000 | 7.37615 | 0.00452134 | 1,631.41 | 99.938703% |
| B3 2000–2500 | 6.91226 | 0.000515445 | 13,410.3 | 99.992543% |
| B4 2500–3000 | 16.2921 | 0.000807410 | 20,178.3 | 99.995044% |
| BALL 600–3000 | **11.9773** | **0.00228248** | **5,247.49** | **99.980943%** |

E6에서 “A source가 CMFGEN보다 11.98×”라고 읽힌 BALL의 고정 source는
실제로는 CMFGEN의 `1/438.12`이다. 나머지 99.980943%가 자기 `J`를
돌려 읽은 산란항이다.

E6와 동일한 `tau_out>=1` 파장 폭 90% 이상(thick90) shell 집합에서도
이 판독은 전 대역에서 같다.

| band | thick90 shell | `S_fixed/CMF` median [min,max] | E6 `S_total/CMF` median [min,max] |
|---|---:|---:|---:|
| B0 | 20 (s0–s19) | 0.005417 [0.000700,0.021788] | 1.9223 [0.04793,33.768] |
| B1 | 12 (s0–s11) | 0.004372 [0.000577,0.012032] | 7.4153 [2.843,32.325] |
| B2 | 14 (s0–s13) | 0.001881 [0.000941,0.004521] | 2.7296 [0.5985,7.376] |
| B3 | 13 (s0–s12) | 0.000647 [0.0000582,0.001162] | 5.6194 [2.160,10.973] |
| B4 | 11 (s0–s10) | 0.000888 [0.000416,0.001928] | 14.2148 [4.106,18.482] |
| BALL | 14 (s0–s13) | 0.001750 [0.000952,0.002388] | 6.2189 [2.124,11.977] |

전 300 band-shell 비율은 `band_shell_fixed_opacity.csv`가 권위표다. 위 결과는
“방출 자체가 너무 크다”를 반증하지만, 수송으로 이미 만들어진
`J_producer`의 기원을 반증하지는 않는다.

## 3. UV 불투명도 감사

### 3.1 전자산란 항등과 payload 이름의 함정

조립기의 순수 전자산란은 정의상 정확히

```text
sigma_T = 6.6524587e-25 cm^2
chi_e   = n_e * sigma_T            [cm^-1]
```

이다(`src/lumina_cmfgen.c:963`). 그러나 payload의 `chi_es`/이 보고서의
`chi_coherent`는 순수 `chi_e`가 아니다. 조립기가

```text
chi_coherent = chi_e + (chi_line - chi_line_th)
```

를 저장한다(`src/lumina_cmfgen.c:1051`). 즉 선의 coherent/scattering 못이 포함된
mixed field다. exact capture `n_e`가 payload에 없어 숫자 항등을 다시 검증할 수는
없지만, final-state `n_e*sigma_T` 대리값과 비교하면 BALL에서 mixed field가
s0/s8/s20에서 각각 216.08/471.14/442.24배다. 이것은 `chi_es` 필드를
전자산란으로 해석하면 안 된다는 독립 확인이다.

### 3.2 선 opacity의 산술 범위

`LCMFCE01`은 `chi_abs=chi_bf+chi_ff`와 `chi_line_th`를 저장하지 않는다.
따라서 선 전체를 유일하게 분리할 수는 없고, 다음 범위만 도출했다.

```text
chi_line,total = chi_total - chi_e - chi_bf - chi_ff
chi_line,coh(proxy) = chi_coherent(payload) - n_e(final)*sigma_T
lower line fraction = chi_line,coh(proxy) / chi_total
upper line fraction = (chi_total - n_e(final)*sigma_T) / chi_total
```

하한은 coherent line만, 상한은 bf/ff가 0이라는 극단에서 선 전체다. 단 `n_e`의
시점이 다르므로 이것은 exact capture bound가 아니라 **epoch-proxy 범위**다. 어떤 음수도
0으로 clamp하지 않았고 300행 전부에서 proxy는 양수였다.

| shell | `<chi_total>` BALL [cm^-1] | mixed coherent | `n_e sigma_T` proxy | 선 분율 하한–상한 | `<tau_out>` BALL |
|---:|---:|---:|---:|---:|---:|
| s0 | 6.74534e-13 | 6.65179e-13 | 3.07838e-15 | 98.1568–99.5436% | 599.660 |
| s8 | 2.40293e-13 | 2.34938e-13 | 4.98657e-16 | 97.5638–99.7925% | 135.376 |
| s20 | 1.38016e-14 | 1.03367e-14 | 2.33734e-17 | 74.7257–99.8306% | 25.2365 |

s8의 대역별 선 하한은 B0/B1/B2/B3/B4에서
97.069/99.330/99.638/99.378/97.147%이고, `tau_out` 평균은
222.927/49.907/67.068/33.995/10.053이다. 즉 우리 조립기 내부에서
**UV 선 opacity 자체가 없는 광범위 결손**은 반증된다. 다만 이는 선 강도,
이온 분포, 파장 형상이 CMFGEN과 같다는 뜻이 아니다.

### 3.3 CMFGEN과의 정량 비교: UNRESOLVED

CMFGEN `MEANOPAC`은 속도 depth별 Rosseland/flux mean과 전자산란 tau를 주지만
600–3000 Å frequency-bin opacity를 주지 않는다. 따라서 우리 `tau_out`과
MEANOPAC `tau_Ross`(s0/s8/s20 = 4.0854/0.20955/0.011978)를 비율로 나누어
“UV opacity 과잉/결손”으로 표현하지 않았다. 직접 CMFGEN UV `chi_nu` 비교와
`chi_bf` 분리는 둘 다 **UNRESOLVED**다.

가용한 간접 지표로 각 장의 depth 변화 `Delta ln J=ln[J(outer)/J(inner)]`를
비교했다.

| band | s0→s8 ours | s0→s8 CMF | CMF보다 덜 감쇠한 e-fold / factor | s8→s20 ours | s8→s20 CMF |
|---|---:|---:|---:|---:|---:|
| B0 | -4.371 | -6.707 | 2.336 / 10.34× | -4.532 | +2.100 |
| B1 | -2.387 | -4.647 | 2.260 / 9.59× | -3.431 | -0.122 |
| B2 | -2.444 | -3.495 | 1.051 / 2.86× | -3.287 | +1.122 |
| B3 | -2.169 | -2.782 | 0.614 / 1.85× | -3.330 | -0.923 |
| B4 | -1.079 | -2.457 | 1.378 / 3.97× | -3.049 | -2.151 |
| BALL | -1.959 | -3.237 | 1.278 / 3.59× | -3.197 | -0.0568 |

s0→s8의 B0/B1은 결손 blanketing이 만들 수 있는 10× 규모의 방향 지표다.
그러나 s8→s20에서는 우리 `J`가 오히려 훨씬 빠르게 줄고 CMFGEN은 B0/B2에서
증가한다. 기하학적 dilution, 분포 source, 경계조건이 엉켜 있어 이를 opacity로
유일 역산할 수 없다. 결론은 **UV opacity 결손 정량 = UNRESOLVED**이다.

## 4. 상위준위 초열성

### 4.1 정의와 대표 선

각 이온의 최종 저장 population을 보존한 within-ion LTE 기준을 새로 구성했다.

```text
Z_ion(T_e) = sum_k g_k exp(-E_k/kT_e)
n_u^LTE    = n_ion g_u exp(-E_u/kT_e) / Z_ion
b_u        = n_u / n_u^LTE
```

이것은 dump의 “자기 이온 ground 대비 b”와 다른 absolute-within-ion 값이다. 1208.743 Å
payload bin은 1205.545–1211.950 Å이며, s8 직접 `A_ul*n_u` 기여 상위 5개는
Co III 1206.020 Å 1개와 Si III 1206.500, 1206.555, 1207.517, 1210.455 Å
4개다. 추가 대표 선은 s8 기여 상위 Fe III
1122.521/1895.472/1914.068/1926.321/2079.655 Å, Co III
1760.356/1773.565/1782.967/1928.566/1940.146 Å다.

### 4.2 `b_u` 결과

아래는 선 집합 내 `min/median/max`다. 괄호의 `N`은 그 shell에서 `n_ion>0`이고
상위준위가 저장되어 `b_u`가 정의된 선 수다.

| 선 집합 | s0 `b_u` | s8 `b_u` | s20 `b_u` |
|---|---:|---:|---:|
| 1208 Å s8-top 5 | 0.498/0.498/0.498 (N=1) | 0.767/2.256/4.879 (N=5) | 5.584/11.493/121.129 (N=3) |
| Fe III UV top 5 | 0.501/0.700/0.709 (N=5) | 2.315/2.372/3.649 (N=5) | UNRESOLVED (`n_FeIII=0`) |
| Co III UV top 5 | 0.107/0.482/0.487 (N=5) | 2.606/3.301/4.860 (N=5) | UNRESOLVED (`n_CoIII=0`) |

s8의 대표 Fe III/Co III 초열성은 최대 3.65/4.86배, 즉 0.56/0.69 dex이다.
12×에 필요한 1.079 dex보다 작다. s20의 Fe III/Co III은 final archive에서
이온 population이 정확히 0이므로 LTE 비율을 억지로 만들지 않았다.

### 4.3 `A_ul n_u` 대 `epsilon B` 비

동일 선의 적분 방출 power를 생산 코드의 정식으로 계산했다.

```text
eta_Aul = (h nu / 4 pi) A_ul n_u
eta_epsB(epsilon=1) = (1-exp(-tau_S)) nu/(c t_exp) B_nu(T_e)
eta_Aul/eta_epsB(epsilon) = [eta_Aul/eta_epsB(1)] / epsilon
```

exact capture-epoch `n_u`, per-line tau, epsilon이 없으므로 아래는 final-state proxy이며,
epsilon=1일 때의 **하한** `min/median/max`다. 생산자는 `LINE_EPS_PHYS=1`이고
epsilon 설정 범위는 기존 1e-5–1이므로 실제 비는 아래 값에서 최대 1e5배 크다.
이 범위를 산출하기 위해 신규 clamp나 population 보정을 쓰지 않았다.

| 선 집합 | s0 ratio @ epsilon=1 | s8 ratio @ epsilon=1 | s20 ratio @ epsilon=1 |
|---|---:|---:|---:|
| 1208 Å s8-top | 263.99 (N=1) | 19.14/100.80/8.38e8 | 5.23/2.46e3/1.31e8 |
| Fe III UV top | 1.04e3/2.14e3/2.21e3 | 4.72e3/1.88e4/8.06e4 | UNRESOLVED |
| Co III UV top | 2.05e4/1.50e5/2.17e5 | 1.11e4/2.87e4/6.21e4 | UNRESOLVED |

이 비는 `b_u` 크기와 동일하게 움직이지 않는다. s0에서 Fe III/Co III
`b_u<1`이어도 비는 이미 1e3–2e5 이상이다. 즉 큰 비는 초열성 하나가 아니라
epsilon, escape probability, expansion normalization을 함께 바꾸는 **정식 좌표 차이**다.

1208 Å payload에서 s0 `eta_fixed_B2/eta_fixed_A=81,753.64`는 위 넓은 epsilon
범위와 차수상 양립하지만, post-EPAY 합계이며 다중 선/연속선을 포함하므로
개별 선 비의 폐합 증거는 아니다. 같은 셀의 `eta_total_B2/A`는 coherent 항으로
희석돼 22.4642다. 더 중요하게 s8과 s20의 1208 Å payload는
`eta_fixed_B2/A=eta_total_B2/A=1.0`이다. 따라서 s8의 `b_u` 초열성이
E6 12×를 만든 변경 좌표라고 볼 수 없다.

### 4.4 과이온화와의 동행성

기존 독립 rate-balance 오라클은 s8에서 S II→III +2.26 dex,
S III→IV +3.16 dex, Fe II→III +1.12 dex, Fe III→IV +2.37 dex의
과이온화를 보였다. s0 Fe II→III/Fe III→IV는 +0.59/+0.97 dex였다.

- s8에서 Fe III 대표 `b_u` 중앙값이 2.372(0.375 dex)로 1보다 크고
  Fe 과이온화와 같은 shell/ion에서 **방향은 동행**한다.
- 그러나 s0에서는 Fe III `b_u` 중앙값이 0.700인데도 Fe III→IV가
  +0.97 dex 과이온화다. 초열성은 과이온화의 필요조건이 아니다.
- s8 `b_u` 0.36–0.69 dex는 Fe/S rate-balance 1.12–3.16 dex보다 작고,
  Co에는 같은 shell의 독립 link-balance 표가 없다.

유효 쌍이 s0/s8 Fe 두 개에 불과하고 Co link table이 없으므로 상관계수나
유의성은 **UNRESOLVED**다. 정성적 결론은 “s8에서 동행하지만 shell 관통
원인은 아니며, 크기도 12×에 못 미친다”이다.

## 5. 종합 판독과 다음 표적

1. **E6 source 비교의 12×:** 자기-독해 항으로 완전 설명된다. 정정된
   `S_fixed`는 과잉이 아니라 438× 결핍이다. E6 target s0에서 관측한
   `eta_total_B2/A=22.46`을 A의 전체 UV 물리 source 원인으로 확대하는 해석은
   취소한다. E6 자체의 “B2 정식화가 과잉을 해소하지 않는다”는 lane 비교 판정은
   그대로 유효하다.
2. **물리적 `J_producer` 과잉:** 아직 UNRESOLVED다. 불투명도 필드에 선이
   97–99.8% 규모로 포함된 것은 확인되어 전체 선 누락은 기각되지만,
   CMFGEN UV-bin `chi_nu`가 없어 상대 결손을 숫자로 말할 수 없다. s0→s8
   감쇠율은 부족 blanketing과 같은 방향이지만 외곽에서 반전된다.
3. **초열성:** s8 대표 Fe III/Co III `b_u<=4.86`으로 단독 12×를 못
   만든다. `A_ul*n_u / epsilon B`가 1e3–1e5 이상인 것은 초열성이 아닌
   정식/소멸확률 차이가 지배한다.

따라서 다음 무-solve 포렌식의 우선순위는 다음이다.

1. 입사/inner-boundary `J`, coherent line-scattering operator, depth mapping의 별도 장부.
2. pre/post-EPAY per-bin `eta_fixed` shape과 셀별 scale 역산. 현 payload만으로는
   pre-EPAY 형상이 소실돼 **UNRESOLVED**.
3. 선별 적분 power→1000-bin projection의 `dnu` 규약과 1208 Å 빈 폭 민감도.

이 우선순위는 신규 수송 solve를 승인하는 것이 아니라, 현 산출물이 남긴
정보 공백을 기준으로 지정한 후속 측정 목록이다.

## 6. 재현 명령

아래 명령은 Python 문법 검사와 기존 파일 소비만 한다. `timeout`은 실행
상한일 뿐 수송 코드를 호출하지 않는다.

```bash
E7_RUN=/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766
E7_CMF=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4
E7_MODEL=data/tardis_reference_toy06_19p48d_sivcaiv

python3 -m py_compile scripts/emiss_e7_arithmetic.py
timeout 60s python3 scripts/emiss_e7_arithmetic.py \
  --run "$E7_RUN" \
  --cmf-run "$E7_CMF" \
  --model-dir "$E7_MODEL" \
  --out-dir validation/emiss_e7

jq '{fixed_source_canonical_s8,fixed_source_thick90,opacity_canonical_shells,
     attenuation_indicators,target_1208,line_departure_correlation,
     selected_line_category_stats}' validation/emiss_e7/summary.json

python3 - <<'PY'
import csv
rows = list(csv.DictReader(open('validation/emiss_e7/band_shell_fixed_opacity.csv')))
for r in rows:
    if r['shell'] in {'0', '8', '20'}:
        print(r)
PY

sha256sum \
  scripts/emiss_e7_arithmetic.py \
  validation/emiss_e7/band_shell_fixed_opacity.csv \
  validation/emiss_e7/j_depth_indicators.csv \
  validation/emiss_e7/line_departure_proxies.csv \
  validation/emiss_e7/summary.json
```

재현 시 SHA-256:

```text
e46178dfdab19a9a8e56ca2b4f9829cca94c1086c10b9fa75130e944ab3351e5  scripts/emiss_e7_arithmetic.py
113e35014f3cbba1c6515d10d57f3700674616ea070e9ceb123459b95a8839c2  validation/emiss_e7/band_shell_fixed_opacity.csv
8c59850a8bf33ad7eba5bab0e4a50ffa3f9a953313ae7dcf61e878f79d819a8c  validation/emiss_e7/j_depth_indicators.csv
047a3fa5b0bedfcb30e29c57db428e1a47c70cc5975098a6aa447947e78ff4d9  validation/emiss_e7/line_departure_proxies.csv
d3515b7e4de5ac07286c1fbc593578def109210d1f9320dec87966eb39797a0b  validation/emiss_e7/summary.json
```

산술은 `PASS arithmetic-only: 300 band-shell rows, 60 selected line-shell rows`로
종료했다. CMFGEN EDDFACTOR는 196,199 records(ND=90), FINISH=1이고 빈 적분
보존 오차는 `2.220e-16`이었다.
