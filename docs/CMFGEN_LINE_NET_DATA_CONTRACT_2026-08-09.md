# CMFGEN line-net 생산 데이터 계약 — 2026-08-09

## 판정

Lumina의 A2-10 line radiative-equilibrium 항은 더 이상 binned
`4 pi chi_bb J`와 direct Sobolev `4 pi eta_bb`를 서로 빼서 만들지 않는다.
생산 정본은 같은 line-cell과 같은 세대에서 계산한 signed net

```text
q_line = 4 pi (eta_line - chi_line Jbar_line)
```

또는 이에 정확히 동등한 CMFGEN direct bracket이다. 부호는 CMFGEN
`STEQ_T` 관례를 따른다.

```text
q_line > 0  : matter cooling
q_line < 0  : matter heating
```

현재 A2-09의 `n_upper A_ul h nu beta`는 line transport/emission 자료로는 남길 수
있지만 이 net rate의 대체물이 아니다. A2-10에서는 기존 binned line absorption과
escape emission을 제거한 뒤 line-resolved signed owner 하나만 사용한다. 두 표현을
동시에 더하는 것은 owner overlap으로 거부한다.

## 1. CMFGEN 단위와 유한 known answer

CMFGEN source가 정하는 단위 변환은 다음과 같다.

```text
OPLIN = 1e10 pi e^2 / (m_e c)
EMLIN = 1e25 h / (4 pi)
CMFGEN frequency unit = 1e15 Hz
CMFGEN number population = cm^-3

q_internal_raw    = ETAL_MAT * ZNET
q_internal_deck   = SCL * ETAL_MAT * ZNET
q_cgs             = q_internal * 4 pi 1e-10   [erg cm^-3 s^-1]
eta_int_cgs_per_sr = ETAL_MAT * 1e-10          [erg cm^-3 s^-1 sr^-1]
```

`new_main/subs/eval_temp_ddt_v2.f:237-248`와
`new_main/subs/eval_adiabatic_v3.f:143-170`도 radiative-equilibrium 내부 단위가
`1e10/(4 pi)`만큼 cgs보다 크다고 명시한다.

봉인한 O-PHYS finite witness:

| 항목 | 값 |
|---|---:|
| line/depth | `76887 / 90` (CMFGEN 1-based) |
| transition | `CoIV(3d5(6S)4p_5Po[3]-3d6_5De[4])` |
| `ZNET` | `1.4566200000000001e-3` |
| raw `ETAL_MAT` | `3.657273805148772e11` |
| raw integrated emissivity per sr | `36.57273805148772` cgs |
| raw `q_line` | `0.6694430052329409 erg cm^-3 s^-1` |
| deck scale | `0.997943` |
| deck-scaled `q_line` | `0.6680659609711768 erg cm^-3 s^-1` |

생산 O-PHYS parity의 비교 대상은 deck-scaled 값이다. raw 값은 식 자체의 물리
검증용으로 함께 보존한다. fixture:

```text
validation/a2_10/CMFGEN_LINE_NET_KNOWN_ANSWER_2026-08-09.json
SHA256 5a967bbbf6f374c69c6ae5fd63d420d1fadc002c04ddf2fbbef24192a81951a0
schema lumina-cmfgen-line-net-known-answer-v2
```

출력 정밀도 때문에 fixture의 CMFGEN 인쇄값 비교 허용치는 상대 `2e-4`다. 이는
음수나 상쇄를 0으로 만드는 생산 tolerance가 아니라 Fortran 출력 자릿수의 한계다.

## 2. line 집합: `Q_E` superset과 `Q_g` subset

### `Q_E`: energy/radiation cache 집합

`Q_E`는 등록 원자 line 중 canonical BB line-centre domain
`100--20000 Angstrom`에 있는 모든 선이다. NLTE mapping 유무, 현재 조성의 Z 유무,
line strength로 membership을 줄이지 않는다. line ID는 deck/global line index의
오름차순으로 고정하고 domain contract와 전체 ID 목록을 SHA-256으로 봉인한다.

### `Q_g`: population-rate graph 집합

현재 `LineJbarQSet`은 `nlte_line_map >= 0`인 BB rate graph 선만 모은 `Q_g`다.
새 계약에서 `Q_g`의 membership/hash는 그대로 보존하며 반드시 `Q_g subset Q_E`를
검증한다. population/SE 소비자는 `Q_g` line만 조회한다. energy 소비자는 `Q_E`의
active line을 조회한다.

### 단일 cache 결정

`RadiationFieldOwner`는 continuum `J_nu`와 `Q_E` line-`Jbar`를 한 번의 원자적
commit으로 발행한다. 겹치는 `Q_g`용 두 번째 수치 cache는 만들지 않는다.
`Q_g` hash는 rate graph 정체성으로 별도 보존하고, `Q_E` cache의 해당 line ID를
조회한다. 이 결정은 다음을 동시에 막는다.

- 같은 line-shell `Jbar`의 중복 메모리
- energy cache와 rate cache의 서로 다른 generation
- DET 팔에는 존재하지만 MC 팔에는 없는 line-energy field

DET fine solver는 현재 모든 in-window line의 private `jbar_line_det`를 계산하므로
`Q_E` publish가 가능하다. MC path-length estimator도 coevolution 재연결 전에 같은
`Q_E`를 누적해야 한다. 한 팔만 `Q_E`를 발행하는 상태에서는 양팔 parity나
coevolution 완료를 선언하지 않는다.

## 3. energy eligibility와 exact-zero 경계

cache membership과 실제 energy 소비는 구분한다. line-cell은 다음을 모두 만족할
때만 A2-10 signed energy에 들어간다.

1. line centre가 A2-09 production grid의 strict open interval
   `nu_edge[0] < nu_line < nu_edge[n_bins]`에 있다.
2. 해당 Z가 active composition에 있고 `opacity_skip_z[Z]`가 아니다.
3. population, signed tau/opacity, source/emissivity와 `Jbar`가 모두 같은 세대로
   유효하다.
4. sealed negative-opacity policy가 해당 cell에 성공적으로 적용됐다.

조성에서 빠진 Z 또는 명시적으로 skip한 Z는 기존 A2-08/A2-09와 동일하게
`EXACT_ZERO`다. domain 밖은 `OUT_OF_DOMAIN`이지 0이 아니다. `Q_E`에 없는 line,
unsampled `Jbar`, stale generation, hash/profile mismatch는 모두 distinct failure다.

금지된 fallback:

- binned `J_nu`를 line `Jbar` 대신 사용
- private raw `opac->jbar_line_det`를 checked view 없이 직접 사용
- 이전 generation의 line cache 사용
- miss/unsampled/out-of-domain을 0 또는 Planck 함수로 대체
- 음수 opacity, 음수 net rate, 작은 net rate를 clamp/floor/jitter로 수정

## 4. 동일 세대 transaction

한 line-net candidate가 읽는 다음 identity는 begin/end bracket에서 byte-equivalent로
고정돼야 한다.

- radiation required/computed/committed generation과 epoch
- `Q_E` hash, `Q_g` hash와 `Q_g subset Q_E` 증명
- line profile ID/hash 및 실제 producer profile parameter
- population required/computed/committed generation
- `T_e`, `n_e`, partition/within-superlevel generation
- A2-08 tau required/computed/publication generation
- opacity/emissivity generation과 shell geometry/frequency-edge hash
- atomic-model SHA-256와 grid/source manifest SHA-256

`J_nu`와 `Q_E Jbar`는 지금의 `radiation_field_commit()`처럼 하나의 transaction에서
검증 후 함께 공개한다. line-net ledger도 모든 line-cell이 성공한 뒤에만 공개한다.
도중 failure는 public radiation/material/temperature publication을 바꾸지 않는다.

## 5. 상쇄를 숨기지 않는 signed-net 산술

### 정본 계산

CMFGEN의 direct `ETAL_MAT*ZNET`가 이미 주어진 fixture에서는 그 곱을 직접 쓴다.
Lumina의 component 경로에서는 먼저 `chi_int*Jbar`를 별도 반올림한 뒤 빼지 않고
가능한 플랫폼에서 한 번의 fused 연산을 사용한다.

```text
emission_sr   = eta_int
absorption_sr = chi_int * Jbar
net_sr        = fma(-chi_int, Jbar, eta_int)
q_line        = 4 pi * deck_scale * net_sr
```

`exp(z)`, diagonal jitter, positivity floor, `max(q,0)`, 임의 epsilon은 사용하지
않는다. FMA는 double 입력에 대한 곱-차의 중간 반올림을 없애지만 물리 입력의
불확실성을 없애지는 않는다.

### sign qualification

각 cell은 적어도 다음을 함께 기록한다.

```text
emission_sr, absorption_sr, signed_q
absolute_input_uncertainty
cancellation_condition = (|emission_sr| + |absorption_sr|) / |net_sr|
```

MC `Jbar`는 cache의 standard error를 사용한다. DET `Jbar`는 exact-solve residual과
formal-solution qualification으로부터 명시적 absolute bound를 제공해야 한다.
그 bound를 만들 수 없으면 cancellation sign을 생산값으로 승인하지 않는다.

```text
if |signed_q| <= absolute_input_uncertainty:
    status = UNRESOLVED_CANCELLATION
    fail closed
```

이 비교는 음수를 허용하기 위한 tolerance가 아니다. 계산된 부호보다 입력 오차가
큰 cell을 물리적인 heating/cooling으로 오인하지 않게 거부하는 판정이다.

두 비영 성분이 산술적으로 정확한 0을 만들더라도 별도 exact-zero provenance가
없으면 `EXACT_ZERO`로 승격하지 않고 `UNRESOLVED_CANCELLATION`으로 둔다. 반대로
signed result가 유한하고 불확실성 bound보다 크면 음수도 정상적인 heating으로
보존한다.

line-cell signed 값은 먼저 보존하고 compensated signed sum과 absolute sum을 함께
누적한다. 모든 line을 emission/cooling과 absorption/heating 두 거대 합으로 먼저
나눈 뒤 마지막에 빼지 않는다. A2-10의 heating/cooling slot 분할은 검증된 shell
signed line sum 뒤에만 수행한다.

## 6. O-PHYS negative-opacity parity lane

봉인 deck:

```text
CHK_L_POS   = T
NEG_OPAC_OPT = SRCE_CHK
ALLOW_OL    = F
SCL_LN      = T
SCL_LN_FAC  = 0.5
```

CMFGEN `new_main/cmfgen_sub.f:3551-3580`은 `tau_sob < -0.5`일 때 원 signed
opacity를 진단용으로 보존하면서 `SOBJBAR_SIM`의 effective `CHIL=1`,
`CHIL_MAT=1/NUM_SIM_LINES`를 사용한다. `-0.5 <= tau < 0`은 그대로 두며
`EXPONX`가 `beta>1`을 계산한다. 따라서 parity lane은 이 threshold와 effective
transport material을 명시적으로 재현해야 한다. 모든 음수를 0 또는 양수로 바꾸는
정책이 아니다.

A2-08 public signed tau는 언제나 원값을 유지한다. effective parity opacity는
line-net/formal-solution transaction의 별도 typed view다. 완전 maser saturation
모델은 이 O-PHYS parity lane과 별도 계약이다.

## 7. 현재 남은 불일치

다음이 닫히기 전에는 finite fixture kernel PASS만으로 CMFGEN parity를 주장하지
않는다.

1. 현재 `cmfgen_fine_jbar()` line deposit은 `tau <= 1e-12`를 건너뛰므로 CMFGEN의
   `-0.5` threshold 및 `SRCE_CHK` effective material과 다르다.
2. Lumina Gaussian-profile `Jbar`가 CMFGEN `SOBJBAR_SIM`의 ray/quadrature `AV`와
   동일한 net bracket을 만드는지 검증되지 않았다.
3. O-PHYS deck의 density-dependent line scale `SCL`을 line별로 동일 적용해야 한다.
4. `ALLOW_OL=F` witness는 non-overlap 식을 허용하지만, overlap-on deck은
   simultaneous-line direct bracket 계약이 별도로 필요하다.

## 8. 단계별 승인 조건

1. 이 데이터 계약과 v2 finite fixture를 봉인한다.
2. pure signed-net kernel이 finite cooling/heating, exact-zero provenance,
   unresolved cancellation, nonfinite 입력을 구별한다.
3. `Q_E` owner/checked view를 구현하고 DET와 MC가 같은 membership/profile을
   원자적으로 commit한다.
4. A2-10 binned line owner를 line-resolved signed owner로 교체하고 전체 CPU/CUDA
   gate를 통과한다.
5. H200에서 동일 line/depth와 shell aggregate를 CMFGEN cgs finite 값으로 비교한
   뒤 DET master와 lagged MC-to-next-material coevolution barrier에 재연결한다.

## 9. 구현 진행 상태

pure kernel은 `src/line_net_rate.c/.h`에 구현했다.

- component path는 `fma(-chi_int,Jbar,eta_int)`를 사용한다.
- finite cooling과 finite heating을 부호 그대로 분할한다.
- typed exact-zero와 nonzero-component cancellation zero를 구별한다.
- 입력 uncertainty가 sign을 덮으면 signed 진단값은 보존하되 heating/cooling
  publication은 만들지 않고 `UNRESOLVED_CANCELLATION`을 반환한다.
- CMFGEN internal-to-cgs direct conversion은 v2 fixture의
  `0.6680659609711768`을 재현한다.
- `(1+2^-27)(1-2^-27)` FMA witness는 별도 곱셈이면 사라지는 `2^-54`의 finite
  positive net을 보존한다.

검증:

```text
selftest_line_net_rate = PASS
strict -Wall -Wextra -Werror -pedantic = PASS
ASan/UBSan = PASS
Makefile header closure = 29/29 PASS
CPU full link = PASS
OpenMP full link = PASS
```

`Q_E` membership builder와 subset proof도 `src/line_jbar.c/.h`에 구현했다.

- 기존 canonical `Q_g` hash known answer
  `ae6163fee5e036e2d751ba19559704401f6734338c413dbedc3b7517e97e1a30`은
  byte-identical로 유지됐다.
- synthetic `Q_E`는 mapped 여부와 무관하게 domain line을 모두 포함하며 hash
  `f781482b70a921a3e780e8ae8e111cabe41117d55c72e3aa1d1c5e3668ae1720`을
  재현한다.
- `Q_g subset Q_E`는 role/domain/profile/hash/line-ID/frequency를 모두 검사한다.
  missing line, seeded hash corruption과 동일 ID frequency mismatch를 서로 다른
  status로 fail-closed한다.
- 현재 exact-hyd O-PHYS `line_list.csv` offline census는 전체 2,783,436선,
  `Q_E=2,783,421`, invalid frequency 0, `Q_E` hash
  `846ff0e6f651a6f2f82cc1b736db823a894fdcea51671e4b295deacb37c0142d`다.
  50 shell 기준 public numeric cache는 약 3.629 GiB, MC accumulator는 약
  3.111 GiB다.

이 단계는 membership만 닫았다. public `RadiationFieldOwner`는 아직 기존 `Q_g`
cache를 발행하므로 다음 단계에서 `Q_E` schema/view와 DET commit을 교체한다.

public owner schema도 다음 단계까지 구현됐다.

- `LineJbarCache.set_kind`가 legacy `Q_g`와 production `Q_E` numeric membership을
  구별한다.
- owner는 `Q_g` ID/hash와 `Q_E` cache index의 sparse map만 별도로 소유한다.
  `Q_g`용 value/validity/count/SE slab은 존재하지 않는다.
- energy checked view는 `Q_E` hash를, rate checked view는 `Q_g` hash를 요구한다.
  rate view의 lookup은 sparse map을 통해 같은 `Q_E` numeric slab을 읽으며
  energy-only line을 `MISS`로 유지한다.
- legacy `radiation_field_line_jbar_view()`는 rate view의 호환 entry point다.
- seeded `Q_g`-not-subset-`Q_E` commit은 continuum, line generation, 이전 graph
  identity를 모두 보존하고 거부한다.
- 아직 sparse gather를 구현하지 않은 GPU upload는 `Q_E` 위 rate view를 연속
  배열로 오독하지 않도록 명시적으로 fail-closed한다.

A2-03/04/05/06, strict compile, ASan/UBSan, CPU/OpenMP full link와 sm_90 CUDA full
link가 통과했다. 이 검증 중 A2-04의 과거 `3900` edge literal과 Python 4000-bin
replay가 현재 3866-bin SH-grid와 어긋난 것을 발견해 canonical header 식으로
재결합했다. synthetic replay 최대 오차는 약 `1.2e-17`이고 5-band Planck
음성대조는 모두 의도대로 실패한다.

## 입력 provenance

```text
992fba38c8d786b880f345dd91b25103ad1028897c412a8d36bd281be8f7aa47  new_main/cmfgen.f
092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374  new_main/cmfgen_sub.f
7f97c601c7b861efb0cf93bab41c943c26ac3388a83fef937e2d9684fb53f3af  new_main/subs/eval_temp_ddt_v2.f
b9c148098f009fb2594d97ada99a05ccc1d413f52384d877f6e76602db1a286a  new_main/subs/eval_adiabatic_v3.f
b670330b5411831649b675edc19bea787d0f1d47fdcf66a8e03e04014e83301d  web/full_descr.tex
e533a503d00ce616982ac5346a0c219afb18a90352e5e1d2bb73dd9e2c3b59c8  O-PHYS LINEHEAT
3066c9e96069f64856fac4e25c8966aca48f6c29a4796c03d283a6161e5e8fee  O-PHYS NETRATE
```
