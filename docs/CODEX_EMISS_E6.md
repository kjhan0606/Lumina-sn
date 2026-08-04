# E6 — 수송 우회 직접 측정: 방출률·선원함수 A/B/B2

판정일: 2026-08-02 (Asia/Seoul)  
입력: `/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.{A,B,B2}`  
소비기: `scripts/emiss_e6_direct_fields.py`  
산출물: `validation/emiss_e6/{summary.json,band_shell.csv,trip_1208_shells.csv}`

## 0. 판정

**E6 판독: 정식화 좌표 기각.** B2의 직접 source field는 UV에서 A의 CMFGEN 대비
과잉을 해소하지 않는다. 바뀌지 않는 셸·대역에서는 A와 같고, 바뀌는 곳에서는
`eta_B2 >= eta_A`이며 큰 폭으로 더 멀어진다.

- E5의 기준 셸 s8에서 `S_A/J_CMFGEN`은 B0/B1/B2/B3/B4/BALL 순서로
  `33.7681, 32.3246, 7.37615, 6.91226, 16.2921, 11.9773`이다.
  B2는 `35.9667, 32.3246, 7.37615, 6.91226, 16.2921, 11.9816`으로,
  B0와 BALL에서 오히려 증가하고 나머지는 정확히 유지된다.
- `tau_out>=1`인 파장 폭이 90% 이상인 thick90 셸들에서 BALL은 14개 셸 모두
  CMFGEN 쪽으로 이동하지 않았다. thick90 중앙값은 A가 `6.21894 x`, B2가
  `10.2799 x`이고 `S_B2/S_A` 중앙값은 `1.07646`이다.
- B0에는 A가 CMFGEN보다 작은 s15--s22에서 B2의 증가가 CMFGEN 방향인 부분 결과가
  있다. 그러나 이는 **A의 과잉을 줄인 결과가 아니며**, B0 thick90 전체 20개 중
  5개뿐이다. s0--s10의 A 과잉은 악화되고, A-deficit인 s11--s14도 B2가 CMFGEN을
  지나 크게 overshoot해 거리 기준으로 악화된다.
- 1208.743248 Å, target s0에서 `eta_B2/eta_A = S_B2/S_A = 22.46416481`이다.
  B2 emissivity는 급락한 것이 아니라 증가했다. 이 빈의 B2 `eta/이웃평균`은
  `1.16946`이고 source의 `S/이웃평균`은 `0.931381`; A의 source 값 `0.927111`보다
  더 가파른 새 dip도 아니다. “η 급감으로 source가 가팔라져 트립” 가설은 **반증**된다.
- 정확한 certified-negative recurrence의 비국소 수치 원인은 수송을 다시 풀거나
  내부 recurrence를 계측하지 않았으므로 **UNRESOLVED**다. 다만 E5와 합치면 트립은
  미정의분 삭제가 아니라 covered 정식화 intervention에 연관되고, E6는 그 연관이
  trip segment의 국소 η 급락으로 설명되지 않음을 추가로 확정한다.

신규 LUMINA/CMFGEN 런, 수송 solve, clamp/floor 추가, 커밋은 수행하지 않았다.

## 1. 입력과 산술 계약

### 1.1 payload 인증

E5의 fail-closed 3-lane validator를 그대로 재사용했다. 세 payload는 50 shell × 1000
bin, iteration/generation 10, post-damping이며 공통 assembly-state SHA-256은
`302a64e2e394d74a9bbbb92d26a0eb009dacda88145b10e42d3f0b280bf4b044`다.

| lane | payload SHA-256 |
|---|---|
| A | `ac62eae5bec6d6beaf06d513c2cef38386b365f6b7826cff61c4edc0f2a34011` |
| B | `95b87203673b69e6f3ef756361e11d8f9fb5d0a5e174be7d1762f9f31d184582` |
| B2 | `775c9e844b4000551caba7061be79ea8a7b4e25173139910ad447fc4b6689f5a` |

`chi_total`, `chi_coherent`, geometry, ν, `J_producer`는 bitwise 공통이고 η만 intervention
대상이다. 600--3000 Å의 세 lane `chi`, `eta_fixed`, `eta_coherent`, `eta_total`에서
negative/zero/nonfinite count는 모두 `0/0/0`이다. 따라서 비율의 부호 이상이나 0분모는
없다.

payload ν 중심은 공용 1000-bin 중심과 최대 상대오차 `1.22125e-15`로 일치한다.
payload `dnu`는 producer의 center-times-dlog 폭이라 exact log-bin edge 폭과
`1.16967e-6` 차이가 난다. 기존 CMFGEN 비교와 동일하게 band 적분에는 공용 exact edge의
겹침 폭을 썼고 이 차이를 보정값으로 payload에 되쓰지 않았다.

### 1.2 정의

각 shell `s`, band `q`에서 exact edge overlap `w_b`를 사용했다.

```text
E_X(s,q)       = sum_b eta_total_X(s,b) w_b
<eta_X>(s,q)   = E_X / sum_b w_b
S_X(s,b)       = eta_total_X(s,b) / chi_total(s,b)
<S_X>(s,q)     = sum_b S_X(s,b) w_b / sum_b w_b
tau_out(s,b)   = sum_{u=s}^{49} chi_total(u,b) Delta r_u
```

`E`의 단위는 `erg s^-1 cm^-3 sr^-1`, `<eta>`는 여기에 `Hz^-1`가 붙고, `S`와
CMFGEN `J_nu`는 `erg s^-1 cm^-2 Hz^-1 sr^-1`다. `thick90`은 band 폭의 90% 이상에서
`tau_out>=1`인 산술 표지다. 이는 Lumina `chi_total`의 outward radial 지표이며 CMFGEN
Rosseland tau로 위장하지 않는다. `band_shell.csv`에는 `tau_out` 평균과 `tau>=1`,
`tau>=10` 파장폭 분율을 모두 기록했다.

### 1.3 CMFGEN jnu4

기존 `w3_gamma_triple_compare.py`의 EDDFACTOR reader, RVTJ depth mapping, log-J velocity
interpolation, integral-preserving bin average를 import해 재사용했다. point interpolation은
쓰지 않았다.

- EDDFACTOR: 142,832,872 bytes, ND=90, good records=196,185, FINISH=1;
- RVTJ: 604,183 bytes;
- 44개 resolved shell에서 `sum(Jbar*dnu)/native integral`의 최대 `|ratio-1|`은
  `2.22045e-16`;
- CMFGEN 최대 RVTJ 속도는 `35975.288 km/s`다. 이를 넘는 Lumina s44--s49의
  `S/CMFGEN`은 외삽하지 않고 **UNRESOLVED**로 남겼다. 이 여섯 셸의 A/B/B2 내부
  emissivity/source 비율은 정상 산출했다.

## 2. 방출률 A/B/B2

### 2.1 shell별 변화 범위

아래 비율은 shell별 band-integrated `E_B2/E_A`다. `changed shells` 밖은 bitwise 또는
산술적으로 `1.0`이다. 모든 changed row가 증가이며 감소 row는 없다.

| band [Å] | changed shells | changed `E_B2/E_A` 범위 | max `|E_B2/E_B-1|` |
|---|---|---:|---:|
| B0 600--1000 | s0--s22 | 1.15849--6568.45 | `1.16803e-9` |
| B1 1000--1500 | s0--s4 | 46.3290--1098.74 | `1.46015e-8` |
| B2 1500--2000 | s0--s4 | 296.974--1330.82 | `4.33310e-9` |
| B3 2000--2500 | s0--s4 | 9.84371--64.6493 | `2.28263e-7` |
| B4 2500--3000 | s0--s4 | 1.42156--3.83234 | `1.12818e-5` |
| BALL 600--3000 | s0--s22 | 1.00256--733.897 | `2.17516e-8` |

B2는 B의 미정의 전이를 A값으로 유지하지만, post-EPAY payload에서 그 차이는 위 표처럼
매우 작다. 따라서 큰 B2/A 증가는 미정의 유지분이 아니라 B와 B2가 공유하는 covered
`A_ul*n_u` 정식화에서 온다.

`LCMFCE01`에는 continuum/line 완전 분해가 없으므로 **선 전체 절대 방출률은
UNRESOLVED**다. 임의로 `eta_fixed`를 모두 선으로 간주하지 않았다. 다만 lane 대수로
식별되는 두 항은 `band_shell.csv`에 별도 기록했다.

```text
covered formulation delta       = eta_fixed_B2 - eta_fixed_A
retained undefined contribution = eta_fixed_B2 - eta_fixed_B
```

manifest의 undefined 장부는 pre-EPAY이고 위 payload 차이는 post-EPAY이므로 두 epoch의
절대량을 섞지 않았다.

### 2.2 절대값 예시와 전체 표

전체 6 band × 50 shell = 300행의 `E_A`, `E_B`, `E_B2`, 평균 η와 비율은
`validation/emiss_e6/band_shell.csv`가 권위표다. E5 기준 s8의 band-integrated 절대값은
다음과 같다.

| band [Å] | `E_A` | `E_B` | `E_B2` | `E_B2/E_A` |
|---|---:|---:|---:|---:|
| 600--1000 | `5.694077203e-4` | `7.167656318e-4` | `7.167656318e-4` | 1.25879156 |
| 1000--1500 | `9.464015079e-3` | `9.464015079e-3` | `9.464015079e-3` | 1.00000000 |
| 1500--2000 | `1.401687271e-2` | `1.401687271e-2` | `1.401687271e-2` | 1.00000000 |
| 2000--2500 | `8.385157511e-3` | `8.385157511e-3` | `8.385157511e-3` | 1.00000000 |
| 2500--3000 | `4.659599353e-3` | `4.659599353e-3` | `4.659599353e-3` | 1.00000000 |
| 600--3000 | `3.709505237e-2` | `3.724241028e-2` | `3.724241028e-2` | 1.00397244 |

## 3. 선원함수 대 CMFGEN Jν

### 3.1 canonical s8

s8은 모든 band에서 `tau_out>=1` 파장폭 분율이 1.0이고 band-mean `tau_out`이
B0/B1/B2/B3/B4/BALL 순서로 `222.927, 49.9073, 67.0684, 33.9947, 10.0528,
135.376`이다.

| band [Å] | `<S_A>` | `<S_B2>` | `<J_CMFGEN>` | A/CMFGEN | B2/CMFGEN | B2/A |
|---|---:|---:|---:|---:|---:|---:|
| 600--1000 | `1.625079453e-6` | `1.730888740e-6` | `4.812470339e-8` | 33.7681 | 35.9667 | 1.06511 |
| 1000--1500 | `1.004967218e-4` | `1.004967218e-4` | `3.108990404e-6` | 32.3246 | 32.3246 | 1.00000 |
| 1500--2000 | `1.637225605e-4` | `1.637225605e-4` | `2.219621135e-5` | 7.37615 | 7.37615 | 1.00000 |
| 2000--2500 | `4.070636183e-4` | `4.070636183e-4` | `5.889006512e-5` | 6.91226 | 6.91226 | 1.00000 |
| 2500--3000 | `1.419536425e-3` | `1.419536425e-3` | `8.713018879e-5` | 16.2921 | 16.2921 | 1.00000 |
| 600--3000 | `1.479086329e-4` | `1.479615375e-4` | `1.234910069e-5` | 11.9773 | 11.9816 | 1.00036 |

따라서 A의 UV source 과잉은 s8에서 6.91--33.77배이고, B2는 이를 유의미하게 닫지
않는다.

### 3.2 thick90 shell 분포

범위와 중앙값은 각 band의 thick90 shell population에 대한 shell 통계다.

| band | thick90 shells | A/CMFGEN median [min,max] | B2/CMFGEN median [min,max] | B2/A median | toward count |
|---|---:|---:|---:|---:|---:|
| B0 | 20 (s0--s19) | 1.92230 [0.0479264,33.7681] | 14.8453 [0.0508206,15003.5] | 4.01999 | 5/20 |
| B1 | 12 (s0--s11) | 7.41533 [2.84317,32.3246] | 27.3849 [5.87590,9196.94] | 1.00000 | 0/12 |
| B2 | 14 (s0--s13) | 2.72956 [0.598493,7.37615] | 5.71979 [0.598493,3854.25] | 1.00000 | 0/14 |
| B3 | 13 (s0--s12) | 5.61943 [2.16039,10.9734] | 6.20388 [2.16039,241.691] | 1.00000 | 0/13 |
| B4 | 11 (s0--s10) | 14.2148 [4.10571,18.4824] | 16.2921 [14.1324,20.6912] | 1.00000 | 0/11 |
| BALL | 14 (s0--s13) | 6.21894 [2.12352,11.9773] | 10.2799 [2.56407,2027.84] | 1.07646 | 0/14 |

B0의 toward 5개는 s15--s19이며 A가 이미 CMFGEN의 0.0479--0.1095배인 deficit
구간에서 B2가 위로 움직인 경우다. s20--s22도 같은 방향이나 thick90 기준 밖이다.
정식화가 A의 UV 과잉을 해소한다는 판독에는 해당하지 않는다.

## 4. 1208.743248 Å 트립 해부

E5 metadata의 공통 좌표는 target radial s0, payload frequency 470, flattened ray 9,
segment 44, substep 0이다. 기하만 재구성하면 ray의 `mu=0.6408017754`, impact parameter
`5.509536193e14 cm`이고 segment midpoint는 `1.391312527e15 cm`이다. 이 위치의 radial
field interpolation bracket은 s5--s6이다.

| 위치 | `chi_total` | `eta_A` | `eta_B` | `eta_B2` | B2/A | A η/이웃평균 | B2 η/이웃평균 | A S/이웃평균 | B2 S/이웃평균 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target s0 | `3.804675718e-13` | `3.539365785e-16` | `7.950889631e-15` | `7.950889631e-15` | 22.4642 | 1.15558 | 1.16946 | 0.927111 | 0.931381 |
| segment bracket s5 | `1.807779397e-13` | `1.849165348e-17` | `1.849165348e-17` | `1.849165348e-17` | 1.00000 | 1.22380 | 1.22380 | 0.829551 | 0.829551 |
| segment bracket s6 | `1.597165984e-13` | `9.610911908e-18` | `9.610911908e-18` | `9.610911908e-18` | 1.00000 | 0.946192 | 0.946192 | 0.611723 | 0.611723 |

여기서 이웃평균은 payload index 469와 471의 산술평균이다. s0에서 B2/B는
`1.000000000002971`; B와 B2는 사실상 동일하다. s5와 s6에서는 A/B/B2가 정확히
동일하므로 segment 국소 source 형상은 intervention으로 바뀌지 않았다.

따라서 관측은 다음을 말한다.

1. target s0의 정식화 개입은 η와 S를 22.46배 **올린다**.
2. η는 이웃 평균보다 16.95% 높고, S의 6.86% dip은 A의 7.29% dip보다 가파르지 않다.
3. 실제 segment bracket의 기존 S dip은 크지만 세 lane 공통이므로 새 트립의 국소
   B2 η 급락 증거가 아니다.
4. covered intervention이 frequency-advection recurrence의 과거값/비국소 경로를 통해
   certified-negative를 유발하는 정확한 수치 과정은 이 산술 측정만으로 특정할 수 없어
   **UNRESOLVED**다.

## 5. 최종 판독 범위

캠페인 질문에 대한 답은 다음처럼 분리한다.

- **A의 UV source가 CMFGEN보다 큰가:** 그렇다. 특히 기준 s8의 다섯 sub-band에서
  6.91--33.77배, BALL에서 11.98배다. thick90 BALL도 전 셸에서 2.12--11.98배다.
- **B2가 CMFGEN 쪽으로 가까워지는가:** A 과잉 영역에서는 아니다. s8은 유지/악화,
  thick90 BALL은 14/14 모두 악화다. inner s0 BALL은 3.33786배에서 1388.64배로
  크게 악화된다.
- **정식화가 진범인가:** “A 과잉을 상당 부분 해소하는 대체 정식화”라는 의미에서는
  **아니다 — 좌표 기각**이다. B2가 큰 source 변화를 만들 수 있다는 점 자체는 보이나
  방향이 반대다.
- **부분 결과:** B0 외곽의 A-deficit shell에서는 위쪽 이동이 CMFGEN 방향이다. 이는
  파장·깊이 의존 혼합 효과지만 UV 과잉 해소 증거는 아니다.
- **남는 UNRESOLVED:** payload만으로 선 전체/continuum 절대 분해, RVTJ 범위 밖
  s44--s49의 CMFGEN 비교, certified-negative recurrence의 정확한 비국소 수치 원인.

## 6. 재현 명령

모든 E6 수치를 한 번에 재생성하는 명령이다. 수송 executable을 컴파일하거나 호출하지
않는다.

```bash
timeout 60s python3 scripts/emiss_e6_direct_fields.py \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10 \
  --cmf-run /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  --e5-verdict validation/emiss_e5/verdict.json \
  --out-dir validation/emiss_e6
```

입력 및 산출물 인증:

```bash
timeout 60s sha256sum \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.A \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.B \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.B2 \
  scripts/emiss_e6_direct_fields.py \
  validation/emiss_e6/summary.json \
  validation/emiss_e6/band_shell.csv \
  validation/emiss_e6/trip_1208_shells.csv

wc -l validation/emiss_e6/band_shell.csv \
      validation/emiss_e6/trip_1208_shells.csv

jq '{grid,cmfgen:.cmfgen.unresolved_shells_outside_RVTJ_grid,
     anomalies:.field_anomalies_600_3000,band_summary,trip}' \
  validation/emiss_e6/summary.json
```

현재 소비기/산출물 SHA-256은 다음과 같다.

```text
2b6b0d8e7632c1b30caf747d8544fd1f328216be092c956c4a3350f1b019c8c6  scripts/emiss_e6_direct_fields.py
a7a53922bff6c547729b5066cd30027b8a42e5bd1ab74f8fa3c2f5931f07a9a3  validation/emiss_e6/summary.json
f313ebb7e294d097073a3bd15027a48b630fc01cba32dc5d6988e77ce1f52499  validation/emiss_e6/band_shell.csv
6aeb28657bea63c43c0073d6e3ad6115f5c7bc68ce119865c127807e51985f8c  validation/emiss_e6/trip_1208_shells.csv
```
