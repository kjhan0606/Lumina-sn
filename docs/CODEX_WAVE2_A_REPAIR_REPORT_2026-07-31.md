# Codex A Wave 2 수리 보고서

작성일: 2026-07-31  
범위: D-1 bf absorption continuum identity/energy split, Wave 1 C2
FB-MILNE binding, 비-bf B형 사건/MA 소비점  
금지 준수: git 명령 미사용, GPU 실행 미사용

## 1. 결과

- `LUMINA_FIX_BF_CONTINUUM_EVENT=1`에서 legacy `(shell,bin)` argmax
  `activation_level`을 만들거나 소비하지 않는다.
- `(element,ion,lower-level,target)` route별
  `n_lower * target_probability * sigma(nu) * corrfactor`를 누적해
  continuum을 추첨하고, 선택 edge의 `nu_edge/nu`로 MA와 k-packet
  thermal pool을 두 번째 난수로 나눈다.
- `LUMINA_FIX_BF_MULTI_EDGE=1`의 두 GPU free-bound 방출점은 초기화가
  끝난 동일 `sigma_bf` handle과 동일 `sigma_bf(nu)*nu^2` photon-number
  Milne sampler를 사용한다.
- 비-bf B형 사건/MA 직접 소비점은 C28과 C71을 독립 기본-OFF
  수리했다. C21 maser는 signed amplification transfer가 없어 잔여다.
- D6 배너는 event/stim/k-packet 실제 조합을 `ENABLED`, `PARTIAL`,
  `residual`로 구분한다.
- `make cuda` 성공. GPU kernel은 실행하지 않았다.

## 2. 항목별 diff 골자

### D-1 continuum CDF와 energy split

`BFOpacity`에 gate-on 전용 route identity와 per-shell 계수를 추가했다.
target map은 이 게이트가 독립적으로 로드한다. route weight는 opacity
producer가 실제 사용한 lower population과 target probability에서 만들고,
D-3가 함께 켜지면 target별 stimulated-recombination ratio도 같은 route에
저장한다.

생산 원자(26,592 levels)에서 `(shell,bin,route)` dense CDF는
`50*1000*26592*8 = 9.91 GiB`이므로 저장하지 않는다. 대신 다음 정확한
표현을 사용한다.

```text
route static: (Z, ion, lower, target, nu_edge, sigma source)
route shell coefficient: n_lower * p_target
event weight: coefficient * sigma(lower, bin) * corr(target, shell, bin)
CDF: event 시 route 순서로 exact cumulative sum
```

이는 top-K 절단이나 renormalization이 없는 전체 route CDF다. CPU mirror와
GPU가 같은 bin-center opacity 계수를 사용한다. channel opacity도 D-1 ON에서
free-free가 섞인 legacy `chi_bf`가 아니라 bf-only CDF endpoint를 사용한다.

GPU 사건은 첫 난수로 route를 뽑고 두 번째 난수로
`u < min(1,nu_edge/nu)`를 판정한다. 참이면 선택 target에서 macro-atom을
activate하고, 거짓이면 k-packet CDF로 thermal re-excitation한다.
`LUMINA_KPACKET=0`이면 Lumina에 first-class k-packet 상태가 없으므로 kinetic
arm은 legacy 보존 fallback을 쓰며 D6 배너가 `PARTIAL`이라고 명시한다.

### Wave 1 이관 — 두 방출점 Milne 통일

기존 실패 원인은 `cuda_set_kpkt_fb_edges()`가 BF-GEMM 초기화보다 먼저
호출되어 `bf_gemm_get_d_sigma_bf()`가 NULL을 반환한 순서 결함이었다.
multi-edge gate ON에서 BF opacity 초기화 뒤 `bf_gemm_init()`을 보장하고
edge tables와 sigma handle을 다시 bind한다. 두 방출점은 이미 공용 함수
`d_kpkt_fb_milne_frequency()`를 호출하므로 이 rebind로 둘 다 실제
sigma-weighted rejection arm에 들어간다. CMFGEN sigma가 없을 때만 정직하게
flat-sigma thermal-tail fallback이라고 배너에 표시한다.

### 비-bf B형 사건/MA

- C28 — `LUMINA_FIX_MA_J_UNCLAMP=1`: MA internal-up 소비점에서
  `J_CAP_FACTOR`/`J_FLOOR_FACTOR`를 우회한다. 비LTE `J_nu`는 `WB_nu`의
  상하 어느 쪽도 가능하며 Planck anchor는 물리 경계가 아니다.
- C71 — `LUMINA_FIX_MA_NO_LINE_THERM=1`: `LINE_THERM`이 요청돼도 선택된
  bound-bound transition frequency를 유지한다. `B_nu(T_e)` 재추첨은 별도
  thermal emissivity이며 선택된 MA radiative deactivation과 같은 사건이
  아니다.

미수리 12건과 C21 보류 근거는 `CODEX_CLAMP_PROVENANCE.md`에 보존했다.

## 3. ARTIS 대조

| ARTIS `rpkt.cc` | ARTIS 동작 | Lumina Wave 2 |
|---|---|---|
| 405-411 | bf channel 진입, absorption type 설정 | 기존 continuum channel 뒤 D-1 arm 진입 |
| 414-426 | `chi_bf_sum` upper-bound로 allcont 선택 | 전체 route opacity를 누적하고 최초 `CDF > u*chi_bf` route 선택 |
| 428-431 | element, ion, level, phixs target 보존 | route identity에 네 필드 보존; target을 MA activation에 직접 사용 |
| 434-443 | 별도 draw `< nu_edge/nu`이면 upper target MA | 선택 edge/실제 comoving photon frequency로 동일 draw |
| 444-446 | complement를 `TYPE_KPKT` | k-packet pool이 있으면 동일 thermal-pool routing |
| 733-765 | `sigma*p*corrfactor`, target별 departure | Wave 1 D-3 route ratio를 D-1 CDF가 그대로 소비 |

차이는 ARTIS가 cell cache의 `chi_bf_sum`을 직접 들고 있는 반면 Lumina는
고정 1000-bin opacity producer를 유지하므로 CDF가 bin-center sigma를
사용한다는 점이다. energy split의 `nu`는 실제 사건 comoving frequency다.

## 4. 게이트 표

| 게이트 | 기본 | ON 효과 | 의존성 / OFF 계약 |
|---|---:|---|---|
| `LUMINA_FIX_BF_CONTINUUM_EVENT` | 0 | 전체 continuum route CDF + `nu_edge/nu` | target map 독립 로드; 완전 kinetic sink는 `KPACKET=1`; OFF는 argmax/RNG 순서 유지 |
| `LUMINA_FIX_BF_STIM_RECOMB` | 0 | target별 net-opacity corrfactor | D-1과 독립; 함께 켜면 event CDF도 corrected opacity 사용 |
| `LUMINA_FIX_BF_MULTI_EDGE` | 0 | 두 GPU fb 방출점의 공용 sigma-weighted Milne | canonical explicit 0이 alias보다 우선; OFF는 re-init/rebind 없음 |
| `LUMINA_FIX_MA_J_UNCLAMP` | 0 | MA internal-up에서 diagnostic J cap/floor 제거 | OFF는 기존 factor와 산술 그대로 |
| `LUMINA_FIX_MA_NO_LINE_THERM` | 0 | `LINE_THERM` Planck 재추첨 억제 | OFF는 기존 `LINE_THERM`/ARTIS-parity 우선순위 그대로 |

## 5. oracle과 빌드

CPU single-thread frozen oracle, parity50 input, s0/s8/s43을 사용했다.
이 하니스의 strict/available 수량은 bf/ff/rates/thermal/state이며 packet
event fate는 실행하지 않는다.

기존 제출본과 현재 OFF 결과는 `available` 행 전체(즉 strict-compared의
상위집합)가 byte-identical했다.

| cell | eligible-superset SHA-256 |
|---|---|
| s0 | `beaac19b21bd5b9c0d8c7c81903a1c8c13c8f139ba05cf2e01c414f193678cfa` |
| s8 | `54f9fafad8da44602a419562a2ef37c9f0c726fdad6780c72e99df436e87d05f` |
| s43 | `b971a0381d4d6c8246979c3bb8d013290d65deac6985898795bee94894380804` |

전체 CSV의 현재 OFF SHA는 아래와 같고 D-1 ON도 각각 완전 동일했다.
새 게이트 네 개를 모두 명시적 `=0`으로 둔 실행도 unset/default OFF와 같은
세 SHA였다.

| cell | OFF = D-1 ON full CSV SHA-256 |
|---|---|
| s0 | `4789f13c89a3bb613e89cb23e836242285aae31bee6065b2631d61324eee1952` |
| s8 | `a4f1a146a313501a3eaf56232d2d7d3cd4f798425ebd8f426067292edb1538e2` |
| s43 | `c48d2619f160191d4a91e37334cf165d2fc312d2263635a281112523e70b72aa` |

Wave 1 제출 full-CSV SHA와의 차이는 수리와 무관한
`heating_MA_LINE_DESTRUCT,status=unavailable` note 한 줄뿐이다. 적격 범위
규약에 따라 이 비수치·비교불가 행은 효과 판정에서 제외했다.

D-1 ON이 frozen 수량을 바꾸지 않는 것은 예상 결과다. 이 수리는 opacity
값이 아니라 그 opacity로 발생한 packet event fate를 바꾼다. 따라서
“oracle ON 효과”는 이 하니스 범위에서 **0 byte**, 사건 효과는
**측정 불가**이며 GPU run 없이 수치 효과를 주장하지 않는다.

| ON 항목 | frozen-oracle 관측 가능 범위 | 이번 판정 |
|---|---|---|
| D-1 continuum event | bf/ff opacity·rates는 strict, packet fate는 없음 | strict/full CSV 0 byte; 사건 효과 측정 불가 |
| C28 MA J unclamp | MA transition probability/fate가 strict schema 밖 | 미측정; source 소비점과 기본-OFF만 검증 |
| C71 no line-thermal | CUDA packet emission 전용 | oracle 부적격 |
| multi-edge Milne rebind | CUDA 두 fb 방출점 전용 | oracle 부적격; 정적 공용-helper/bind 검증만 수행 |

`make cuda`는 exit 0이었다. 기존 `lumina_nlte_gemm.cu`의
`g_fgemm_nulo` unused warning 외 신규 warning/error는 없었다.

## 6. GPU 레지스터·성능 영향

`cuobjdump --dump-resource-usage`로 직전 `lumina_cuda.withParityAC`와 새
binary를 비교했다.

| arch | 이전 REG | Wave 2 REG | 변화 |
|---|---:|---:|---:|
| sm_80 | 98 | 102 | +4 |
| sm_86 | 98 | 102 | +4 |
| sm_90 | 92 | 92 | 0 |

stack/local spill 표시는 모두 0이다. gate OFF에서도 커널 parameter block은
늘었지만 event scan/RNG/global loads는 분기 밖이다.

gate ON 생산 크기(26,592 routes, 50 shells, 1000 bins, double sigma)는 약
224.26 MiB 추가 device memory다. 사건당 최악 26,592 route 선형 scan이므로
bf-event가 많은 런에서는 transport latency가 증가한다. 이는 9.91 GiB dense
CDF나 top-K 물리 절단을 피한 정확성 우선 구현이다. GPU 실행 금지 때문에
wall-time/occupancy 실측은 하지 않았다.

opacity producer도 ON에서는 BF-GEMM을 우회하고 target별 route coefficient와
corrfactor를 보존하는 CPU exact loop를 사용한다. 따라서 transport뿐 아니라
opacity 갱신 시간도 늘 수 있다. OFF에서는 기존 BF-GEMM 선택과 산술 경로를
그대로 유지한다.

## 7. “왜 생겼나” 5분류

1. **대표값 축약** — opacity 합을 보존하면서 사건 identity만 dominant ion으로
   줄여 lower level/target을 잃었다.
2. **에너지 장부 누락** — photon energy를 threshold ionization과
   photoelectron kinetic 부분으로 나누지 않고 전부 MA에 넣었다.
3. **초기화 순서 결함** — Milne 소비 pointer를 producer 초기화 전에 bind했다.
4. **진단 prior의 생산 침투** — C28/C71 falsifier가 실제 MA 확률/방출 사건을
   바꿀 수 있었다.
5. **backend 분기** — CPU/GPU, line-activated/bf-activated 방출점이 별도라
   같은 물리 수정이 일부 경로에만 도달했다.

## 8. Codex B/C 지침

Codex B:

1. GPU는 운전석 허가 전 실행하지 말고, 먼저 `make cuda`, oracle 2회,
   unset/명시-0/적대 gate 조합의 strict-row SHA를 반복한다.
2. CPU 소형 known-answer atom으로 두 continuum opacity 비 1:3의 선택 빈도,
   target identity, `nu_edge/nu` 두 번째 draw 경계를 검증한다.
3. GPU 허가 시 event census에 bf->MA와 bf->k 두 카운터를 추가해
   `N_MA/N_total ~= opacity-weighted <nu_edge/nu>`를 확인한다.
4. multi-edge ON에서 sigma handle 배너가 `BOUND`인지, 두 방출점의 동일
   `(edge,level,Te,seed)` 주파수 결과가 같은지 확인한다.
5. D-1 ON/OFF kernel wall-time, achieved occupancy, bf events당 route scan 수,
   device-memory 증가를 측정한다.

Codex C:

1. A 보고서 수치가 아니라 source에서 route probability/corrfactor/target의
   index 정합을 독립 추적한다.
2. `chi_bf`의 ff 혼입이 D-1 ON channel에서 제거됐는지, D5 ff channel과
   이중계상하지 않는지 검토한다.
3. `KPACKET=0`의 `PARTIAL` 배너와 fallback을 완전 D6로 오인하지 않는다.
4. C21을 “수리”로 세지 말고 signed maser amplification 부재를 잔여로 유지한다.
5. OFF 효과 판정은 oracle strict-compared 행으로만 하고 unavailable note,
   banner, route allocation 진단을 물리 효과로 세지 않는다.
