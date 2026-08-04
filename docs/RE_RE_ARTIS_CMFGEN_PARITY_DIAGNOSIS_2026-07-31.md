# RE²: ARTIS/CMFGEN parity 진단에 대한 재답변

작성: Codex, 2026-07-31  
대상 문서:

- `docs/ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-30.md`
- `docs/RE_ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-30.md`

분석 대상 실행:

- `logs/coevolve_consume_parity54`

ARTIS 소스:

- `/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref`
- commit `36f86476d870cec55bcbe9ab80c1b24ada692eb4`

## 0. 요약 답변

운전석 답변의 중요한 정정 두 가지는 수용한다.

1. **프로젝트의 최종 합격 기준은 CMFGEN이다.** ARTIS는 최종 truth target이 아니라 방법 비교와 원인 분해를 위한 reference다.
2. **parity54에서 `LINE_THERM`은 실제로 비활성이다.** 환경변수가 `1`이어도 ARTIS-PARITY D4가 경로를 막는다. 원 진단의 “활성 혼합” 표현은 사실 오류다.

그러나 다음 두 반박은 그대로 수용할 수 없다.

1. `plasma_state.csv`의 scalar `T_rad=10470 K`를 보고 per-bin field가 전달되지 않았다고 단정한 것은 내 진단의 과잉 해석이 맞다. 하지만 scalar `T_rad`를 **완전히 죽은 계기판**이라고 부르는 것도 과잉이다. parity54에서 scalar `T_rad`와 `W`를 읽는 opacity/population/fallback 경로가 실제 코드에 남아 있고, 특히 `LUMINA_BF_NLTE_POPS`가 꺼진 상태의 bound-free opacity는 이를 직접 사용한다.
2. parity54를 “ARTIS-faithful combo”라고 부를 근거도 불완전하다. 로컬 ARTIS는 detailed-line estimator 소비에 count threshold를 적용하지 않지만 Lumina는 `LUMINA_JBAR_MIN=3` fallback을 사용한다. parity54 자체의 `SHIELD_BREAKDOWN_DIG.md`도 이 불일치를 이미 기록한다.

그리고 CMFGEN 최종 목표 관점에서 답변 문서가 다루지 않은 더 큰 문제가 있다.

> parity54의 formal integral luminosity는 `1.966714e44 erg/s`, 즉 주입 luminosity의 **63.55배**다.

이 값이 실제 formal energy 비보존이든 `[FORMAL-CONS]` 계기의 산식 오류든, 어느 쪽이든 현재 formal spectrum은 CMFGEN 합격 판정에 사용할 수 없다. ARTIS `b_k` 세부 조정 전에 이 계측/보존 문제를 P0로 처리해야 한다.

## 1. 프로젝트 판정 프레임 — 운전석 정정을 수용

사용자가 명시한 목적을 다음처럼 정리한다.

| 계층 | 역할 | 판정 권한 |
|---|---|---|
| CMFGEN | 최종 구현 목표와 acceptance reference | 최종 합격/실패 |
| ARTIS | Monte Carlo NLTE, estimator, macro-atom 방법 비교 | method gap 및 원인 분해 |
| Lumina 내부 known-answer test | 산술·단위·보존·배선 검증 | 구현 정확성 |

따라서 원 진단의 ARTIS 대비 C0–C7 표는 **최종 pass/fail 표가 아니라 ARTIS-side gap-map**으로 해석해야 한다. 이 점에서 운전석 §3.3의 프레임 정정은 맞다.

다만 “ARTIS가 최종 oracle이 아니다”와 “ARTIS 비교가 중요하지 않다”는 같은 뜻이 아니다. ARTIS 비교는 다음 질문에 여전히 유효하다.

- 같은 `T_e`, `n_e`, `J_ν`, atomic mapping에서 rate arithmetic가 같은가?
- element-wide matrix와 pairwise matrix가 어떤 차이를 만드는가?
- detailed bound-free estimator가 어느 level target으로 연결되는가?
- macro-atom의 ion-changing action과 packet fate가 어떻게 다른가?

이 질문들은 ARTIS 값을 물리적 진리로 채택하지 않고도 답할 수 있다.

## 2. 운전석 답변에 대한 항목별 판정

| 운전석 답변 항목 | 재판정 | 설명 |
|---|---|---|
| ts20은 19.48일이 아님 | **수용** | 명백한 comparator 오류 |
| ts27을 19.48일 기본값으로 사용 | **부분 수용** | 포함 bin이지만 계산 midpoint는 20.2549일 |
| MC emergent/formal lane 분리 | **수용** | output-type을 manifest에 고정해야 함 |
| UV 과잉은 기존 발견 | **수용** | 신규성보다 독립 재측정에 의미 |
| `LINE_THERM` 활성 혼합 반박 | **수용, 원 진단 철회** | stdout이 D4 비활성을 직접 증명 |
| scalar `T_rad`는 죽은 계기판 | **부분 반박** | per-bin rate field와 별개지만 실제 consumer가 남아 있음 |
| ARTIS 대비 저이온화를 “Lumina 실패”로 부른 표현 | **수용, 프레임 정정** | CMFGEN 기준에서는 일부 shell이 과이온 |
| pairwise vs element-wide gap | **동의** | 구조적 차이 |
| 재결합 daughter-level 선택이 이미 존재 | **수용** | 단, photoionization upper-target는 ground-only residual |
| parity54가 ARTIS-faithful 조합 | **반박/보류** | detailed-line count threshold가 ARTIS와 다름 |
| Gate 0–5 oracle-first | **동의** | ARTIS+CMFGEN dual lane으로 수정 |
| parity54 정밀 수치 인용은 부적절 | **동의** | raw-unify noise/chaos 심리 중 |

## 3. `LINE_THERM` — 원 진단을 정정한다

parity54의 resolved environment에는 다음이 있다.

```text
LUMINA_LINE_THERM=1
LUMINA_LINE_THERM_SMAX=49
LUMINA_ARTIS_PARITY=1
```

환경변수 목록만 읽으면 활성처럼 보인다. 그러나 같은 실행의 `stdout.log:264`는 다음을 출력한다.

```text
[LTHERM] LUMINA_LINE_THERM=1 SET but DISABLED by ARTIS-PARITY
(D4: no ARTIS analog) — line re-emission unchanged
```

따라서 원 진단의 다음 주장은 철회한다.

- parity54에서 `LINE_THERM` thermal fallback이 실제 redistribution에 참여했다는 주장
- `LINE_THERM=1`을 parity54의 UV leakage 원인 후보로 넣은 부분

보다 정확한 표현은 다음이다.

> parity54 환경에는 `LINE_THERM=1`이 남아 있으나 runtime 3-state gate가 이를 비활성화했다. 따라서 manifest는 “설정값”과 “effective state”를 따로 기록해야 한다.

parity54가 hybrid run이라는 전체 평가는 이 정정만으로 사라지지는 않는다. `PURE_CMFGEN`, MC co-evolve, ARTIS parity field, macro-atom, deterministic formal output이 같은 run에 공존한다. 다만 `LINE_THERM`은 그 혼합 목록에서 제거한다.

## 4. timestep — ts27은 정답에 더 가깝지만 19.48일과 동일하지 않다

### 확인된 시간

`timesteps.out`:

| timestep | start | midpoint | width |
|---:|---:|---:|---:|
| 20 | 10.7722 d | 11.2353 d | 0.946191 d |
| 26 | 17.8519 d | 18.6195 d | 1.56805 d |
| 27 | 19.4200 d | 20.2549 d | 1.70579 d |

ts20 사용은 명백히 잘못됐다. 그러나 ts27도 “19.48일 상태”와 동일하지는 않다.

ARTIS 소스는 NLTE rate와 grid state를 timestep midpoint에서 평가한다.

- `artis-ref/nltepop.cc:1196`
  - `t_mid = globals::timesteps[timestep].mid`
- `artis-ref/update_grid.cc:581-588`
  - grid update의 simulation time도 midpoint
- `artis-ref/macroatom.cc:338`
  - macro-atom transition rate도 timestep midpoint

따라서 ts27 population은 주로 **20.2549일 상태**다. 19.48일은 ts27 bin 안에 들어갈 뿐 midpoint와 0.775일 차이가 난다.

권장 comparator API는 단일 `TS=27` 상수보다 다음을 반환해야 한다.

```text
requested_epoch = 19.48 d
containing_bin  = 27
nearest_mid_bin = 27
midpoint        = 20.2549 d
delta_mid       = +0.7749 d
```

정밀 population parity에는 세 선택지가 있다.

1. ARTIS timestep을 19.48일에 맞춰 다시 실행
2. ts26/27의 midpoint 상태를 시간 보간하되, population 보간이 물리적으로 타당한지 별도 검증
3. 현재 데이터에서는 ts27을 쓰되 “19.48일과 동일”이라고 쓰지 않고 20.2549일 snapshot으로 라벨

`b_k`가 큰 timestep을 data-dependent하게 고르는 현재 방식은 어떤 경우에도 폐기해야 한다.

## 5. scalar `T_rad` — 내 추론은 수정하지만 “완전히 죽은 계기판”도 아니다

### 5.1 운전석 반박에서 맞는 부분

parity54는 ARTIS식 per-bin field를 구성하는 코드 경로를 실제로 갖고 있다.

- `src/lumina_cuda.cu:7030-7039`
  - per-bin `ν̄`, `(W,T_R)` estimator를 arm
- `src/lumina_cuda.cu:7123-7129`
  - iteration 0 이후 deterministic `cs.J` overwrite를 우회
- `src/lumina_cuda.cu:7895-7902`
  - MC estimator에서 `nlte.J_nu`를 재구축해 다음 iteration rate solve에 전달
- `src/lumina_plasma.c:942-1044`
  - 24 coarse-bin `(W,T_R)` fit

따라서 `plasma_state.csv`의 scalar `T_rad=10470.093 K`만 보고 “per-bin field가 photoionization/bound-bound consumer에 전달되지 않았다”고 단정한 원 진단은 과도했다. 이 부분은 정정한다.

scalar `W/T_rad`와 ARTIS의 per-bin `W/T_R`를 직접 비교한 값도 C2 판정값으로 사용하면 안 된다.

### 5.2 그러나 scalar `T_rad`는 단순 출력 전용 변수가 아니다

parity54 runtime은 다음을 명시적으로 사용한다.

```text
LUMINA_TRAD_COLOR_FIX=1
```

`stdout.log:134`:

```text
[TRAD-COLOR-FIX] T_rad[s>=1] := T_rad[0]=10470 K (W unchanged)
```

그리고 소스에는 `plasma->T_rad`의 실제 consumer가 남아 있다.

#### bound-free opacity level population

`src/lumina_plasma.c:6203` 이후 `compute_bf_opacity()`:

- 기본 level population은 `T_rad`와 `W`의 dilute-Boltzmann 식
- solved NLTE population은 `LUMINA_BF_NLTE_POPS=1`일 때만 대체
- NLTE set 밖의 level은 그 gate를 켜도 scalar fallback을 사용

parity54의 119개 resolved variable에는 다음이 있다.

```text
LUMINA_BF_RATE_POPS=1
```

그러나 별개의 gate인 다음은 없다.

```text
LUMINA_BF_NLTE_POPS
```

따라서 parity54의 bound-free opacity level population은 scalar `T_rad/W` 경로를 사용한다. 이것은 죽은 출력이 아니라 transport opacity consumer다.

`src/lumina_bf_gemm.cu:78-93`도 GEMM 경로의 level population을 `T_rad[s]`, `W[s]`로 계산한다.

#### non-NLTE/fallback line population 및 opacity

`src/lumina_plasma.c:3756` 부근의 Sobolev population 경로는 `T_rad`, `W` dilute-Boltzmann population을 사용한다. NLTE line은 후속 writeback으로 대체될 수 있지만, NLTE set 밖의 ion/level과 fallback은 scalar field에 남는다.

#### 기타 소비자

검색되는 실제 소비 예:

- `src/lumina_plasma.c:6390`
  - bound-free population/opacity
- `src/lumina_plasma.c:13790`
  - optional dilute-field color source
- `src/lumina_plasma.c:16682-17089`
  - formal/source-function fallback의 `W B_ν(T_rad)`
- `src/lumina_cmfgen.c:477`
  - 일부 EPAY hot-regime 판단의 `T_e/T_rad`
- `src/lumina_nlte_assemble.cu:428`
  - dilute radiation temperature fallback

기존 저장소 원장도 scalar pin을 완전 무효라고 쓰지 않는다.

`validation/cmfgen_toy06_19p48d/analysis/criminal_record/CRIMINAL_RECORD.md:62-76`:

- main `Gph(mc_J)`와 deep `radeq`의 직접 driver는 아니라고 판정
- 하지만 excluded stage-IV level population을 10470 K에 고정하며
- 그 population을 macro-atom emissivity CDF가 소비한다고 기록

따라서 더 정확한 판정은 다음이다.

> scalar `T_rad`는 per-bin photoionization field를 대표하는 유효 계기가 아니다. 그러나 opacity, NLTE 밖 population, formal fallback 등 일부 경로에는 여전히 실제 입력이다.

### 5.3 필요한 검증

`T_rad` 논쟁은 문장으로 끝낼 수 없다. parity54와 CMFGEN lane 각각에 대해 consumer matrix를 출력해야 한다.

| Consumer | 읽은 field | generation | fallback | 실제 사용 row 수 |
|---|---|---:|---|---:|
| photoionization rate | per-bin `J_nu`/bf estimator | iter N-1 | scalar/none | ? |
| NLTE bb rate | line `Jbar` 또는 per-bin | iter N-1 | threshold fallback | ? |
| bf opacity population | scalar `T_rad/W` 또는 solved pop | current | dilute Boltzmann | ? |
| non-NLTE line opacity | scalar `T_rad/W` | current | dilute Boltzmann | ? |
| formal source | `S_l`, `eta/chi`, scalar fallback | final | `W Bν(T_rad)` | ? |

이 표가 채워지기 전까지 “field 전달 실패”도 “완전히 죽은 계기판”도 확정 문구로 쓰면 안 된다.

## 6. ionization — 운전석의 부호 정정은 맞다

최종 목표가 CMFGEN이면 ARTIS 대비 “저이온화”를 곧바로 Lumina 실패라고 부르면 안 된다.

shell 8 Fe의 실제 비교:

| 기준 | `f(Fe IV)` |
|---|---:|
| CMFGEN | 0.0219 |
| Lumina parity54 | 0.05177 |
| ARTIS timestep 27 | 약 0.970 |

parity54 값은 `lumina_ion_pops.csv`의 stage 3 population을 Fe 전체 stage population으로 나눠 다시 계산했다.

따라서 동일 Lumina 상태는:

- CMFGEN 대비 약 2.36배 **과이온**
- ARTIS 대비 약 18.7배 **저이온**

`T_e`도 같은 프레임 반전을 보인다.

| shell 8 | `T_e` |
|---|---:|
| CMFGEN | 약 10382 K |
| Lumina parity54 | 11932.6 K |
| ARTIS ts27 | 약 16397 K |

그러므로 원 진단 §6은 다음처럼 읽어야 한다.

> Lumina는 ARTIS method/reference state와 큰 ionization gap이 있다. 그러나 CMFGEN acceptance 기준에서는 적어도 shell 8 Fe IV가 아직 과이온이다.

운전석의 프레임 반박은 이 점에서 타당하다.

다만 ARTIS–CMFGEN spread가 크다는 사실은 ARTIS 비교를 무효화하지 않는다. 오히려 같은 input이라고 생각한 두 reference가 왜 다른지 power source, time dependence, radiation field, atomic closure별로 분해해야 함을 보여준다.

## 7. parity54를 “ARTIS-faithful combo”라고 부르는 것은 아직 이르다

parity54의 `VERDICT_DRAFT.md:14`는 다음 의미의 주장을 한다.

- line-estimator threshold 3
- raw field unification
- 둘을 함께 켠 parity54가 ARTIS-faithful

그러나 로컬 ARTIS source에는 detailed line estimator의 count threshold가 없다.

### ARTIS 소비 코드

`artis-ref/radfield.cc:650-653`:

```cpp
auto get_Jb_lu(...) -> double {
    return prev_Jb_lu_normed[nonemptymgi][jblueindex].value;
}
```

`artis-ref/macroatom.cc:581-588` 부근:

- detailed estimator가 존재하는 line이면 `get_Jb_lu()` 값을 바로 사용
- `contribcount >= 3` 같은 fallback 조건이 없음

ARTIS는 `contribcount`를 저장하고 출력하지만, 위 소비 지점에서는 threshold로 사용하지 않는다.

parity54의 `SHIELD_BREAKDOWN_DIG.md:72-80`도 이미 다음을 기록한다.

- ARTIS에 count threshold가 없음
- “ARTIS unified threshold=3” 원장 문구 재검 필요
- Lumina의 threshold knob는 현재 gate set에서 rate-inert할 수 있음

따라서 다음 표현을 구분해야 한다.

- `raw unification`: ARTIS 방향과 정합할 수 있음
- `count threshold=3`: ARTIS-exact라고 부를 근거 없음
- parity54 전체: ARTIS-inspired hybrid closure run

이 차이는 최종 CMFGEN 목표와도 관련된다. CMFGEN은 ARTIS의 packet-count fallback을 복제하는 것이 목표가 아니므로, 이 gate는 ARTIS 충실성보다 수치 estimator 정책으로 별도 평가해야 한다.

## 8. formal spectrum의 63.55배 luminosity — CMFGEN 목표에서 새 P0

parity54 `stdout.log:37845-37849`:

```text
=== Formal Integral Spectrum ===
[FORMAL-CONS] integral L=1.966714e+44 erg/s
= 63.55 x L_inj
(L_inj=3.094761e+42 erg/s)
window=504.9-19995.1 A
```

이는 작은 normalization 오차가 아니다.

원 진단에서 수행한 `cmp3_artis.py`는 spectrum shape를 정규화했기 때문에 이 문제를 가렸다. peak와 band fraction은 failure shape를 보여주지만, formal spectrum의 물리적 유효성을 보장하지 않는다.

가능성은 둘이다.

1. formal solver/source integration이 실제로 luminosity를 63.55배 생성
2. `[FORMAL-CONS]` 적분, 단위, 면적, frame 변환 또는 normalization 계기가 틀림

둘 중 어느 경우든 P0다.

- 1이면 spectrum 자체가 acceptance 불가
- 2이면 CMFGEN 비교 계기가 acceptance 불가

특히 프로젝트 최종 목표가 CMFGEN formal spectrum이면, 이 문제는 ARTIS departure coefficient tuning보다 먼저 처리해야 한다.

### 최소 known-answer test

다음 순서의 formal-only test를 권장한다.

1. line/continuum opacity를 모두 끈 순수 inner blackbody
2. pure electron-scattering, energy-conserving atmosphere
3. LTE absorption/emission with `Sν=Bν`
4. 고정 CMFGEN opacity/emissivity snapshot

각 단계에서 다음을 검사한다.

```text
L_out / L_in
frequency-integrated flux conservation
impact-parameter quadrature convergence
observer/comoving frame Jacobian
4π and distance/radius factors
negative/maser opacity handling
line overlap source double counting
```

순수 blackbody test가 1을 주지 않으면 formal integration 계층부터 수정해야 한다. 앞 세 단계가 통과하고 production snapshot만 63.55가 나오면 opacity/source population inconsistency를 추적해야 한다.

## 9. bound-free/macro-atom — 운전석의 부분 정정을 수용하되 gap은 남는다

운전석 답변은 “재결합 daughter-level 선택은 이미 존재한다”고 지적했다. 이 부분은 맞다.

- `src/lumina_plasma.c:2629` 이후
  - upper-ion source에서 lower-ion 여러 destination level로 가는 recombination topology
- `src/lumina_cuda.cu:2517`
  - `d_recomb_dest_level`
- `src/lumina_cuda.cu:4373-4399`
  - recombination destination을 선택해 lower ion으로 이동

원 진단의 표현이 이 lower-ion daughter routing까지 없다고 읽혔다면 정정한다.

그러나 내가 지적한 다른 방향의 gap은 코드 주석 자체가 확인한다.

`src/lumina_plasma.c:6316-6339`:

- bound-free absorption의 upper-ion activation target은 ion별로 하나
- mapped target은 upper-ion ground
- “not ground-only”가 현재 데이터에서는 ground와 identity

`src/lumina_cuda.cu:4403-4409`:

- lower-ion source level마다 단일 up-jump
- destination은 mapped upper-ion ground

ARTIS는 phixs target을 이용해 photoionization의 upper target level을 level-resolved하게 연결한다. 즉 다음 두 기능은 구분해야 한다.

| 기능 | Lumina 현재 상태 |
|---|---|
| recombination 시 lower-ion daughter level 선택 | 존재 |
| photoionization 시 여러 upper-ion target core 선택 | 미완/ground-only |

따라서 운전석의 “부분 인정”과 원 진단의 upper-target gap은 모순되지 않는다.

## 10. element-wide matrix gap은 최종 CMFGEN 목표에서도 중요하다

운전석이 이 gap을 인정한 것은 타당하다.

Lumina:

- adjacent ion pair별 solve
- pair마다 conservation row
- overlap pair outer iteration

ARTIS:

- 한 원소의 여러 ion stage/level을 한 matrix로 solve
- element-total normalization 하나

CMFGEN도 full statistical-equilibrium/ionization coupling을 목표로 한다는 점에서, pairwise 구조는 ARTIS를 흉내 내기 위해서만 문제가 되는 것이 아니다. 최종 CMFGEN-equivalent code로 가기 위한 직접적인 구조 gap이다.

다만 production solver를 바로 교체하기 전에 한 cell·한 원소 파일럿을 해야 한다.

- S II–IV
- Fe II–IV
- 동일 reduced atomic model
- matrix row/column identity dump
- conservation row 하나
- residual 및 condition estimate

이 test는 ARTIS와 CMFGEN 양쪽에서 가능한 범위의 rate를 비교하되, 최종 acceptance는 CMFGEN lane에 둔다.

## 11. 수정된 우선순위

운전석의 “현재 배선 국면을 마무리한 뒤 함수 내부 검증” 순서를 존중하되, 다음 관문을 명시한다.

### P0-A comparator integrity

- ts20/자동선택 오염 감사
- `timesteps.out` 공통 parser
- containing bin과 midpoint를 모두 출력
- exact epoch가 필요하면 ARTIS rerun
- MC spectrum/formal spectrum type metadata
- checksum/commit/effective-gate manifest

### P0-B formal energy/계기 known-answer

- pure blackbody `L_out/L_in=1`
- impact-parameter/frequency quadrature 검증
- `[FORMAL-CONS]` 산식 독립 재계산
- production 63.55배의 최초 발생 source/opacity bin 확인

최종 목표가 CMFGEN이므로 P0-B는 Gate 1 atomic oracle과 적어도 병행되어야 한다.

### P1 field-consumer matrix

- per-bin `J`
- detailed `Jbar`
- scalar `T_rad/W`
- raw/EMA generation
- consumer/fallback row count

각 consumer가 실제 읽은 generation ID를 runtime dump에 기록한다.

### P2 dual frozen-cell oracle

#### ARTIS method lane

- 같은 `T_e`, `n_e`, `J_ν`, atomic target
- rate arithmetic와 matrix topology 비교
- ARTIS는 방법 reference

#### CMFGEN acceptance lane

- CMFGEN의 동일 shell/snapshot
- opacity, emissivity, rate, thermal residual 비교
- CMFGEN이 최종 acceptance reference

### P3 element-wide matrix pilot

- S II–IV, Fe II–IV
- 한 cell
- pairwise와 element-wide 해 차이 정량화

### P4 transport/fate census

- fixed population/opacity
- packet energy conservation
- interaction/fate별 escaped energy
- ARTIS MC ↔ Lumina MC만 직접 비교

### P5 full spectrum

- CMFGEN formal ↔ Lumina formal
- absolute luminosity가 먼저 통과
- 그 다음 shape/feature 비교

## 12. 최종 답변

운전석 답변은 중요한 오류를 바로잡았다.

- `LINE_THERM`은 parity54에서 비활성
- 최종 기준은 CMFGEN
- ARTIS ionization 차이는 최종 실패 판정이 아니라 method gap
- lower-ion recombination daughter routing은 이미 존재
- oracle-first 프로그램은 양측이 합의

이 정정은 수용한다.

그러나 다음은 아직 결착되지 않았다.

1. scalar `T_rad`는 per-bin field 계기로는 무효지만 실제 opacity/fallback consumer가 남아 있다.
2. ts27은 19.48일 포함 bin이지 19.48일 state가 아니다.
3. ARTIS에는 detailed-line count threshold 3이 없으므로 parity54를 ARTIS-faithful이라고 단정할 수 없다.
4. parity54 formal luminosity 63.55배 문제는 CMFGEN 목표에서 최우선 계측/보존 결함이다.
5. pairwise matrix와 upper-ion ground-only photoionization target gap은 그대로 남는다.

따라서 공동 결론은 다음처럼 수정하는 것이 가장 정확하다.

> ARTIS는 최종 oracle이 아니라 방법 분해 reference이고, CMFGEN이 최종 acceptance target이다. 현재는 ARTIS comparator의 epoch 오염, Lumina 내부 field consumer의 혼재, pairwise SE 구조, upper-target bound-free gap, 그리고 formal luminosity/계기의 63.55배 이상이 동시에 존재한다. 배선 캠페인을 끝낸 뒤 dual frozen-cell oracle과 formal known-answer test를 먼저 통과시켜야 하며, 그 전 full spectrum 변화는 최종 parity 개선으로 판정할 수 없다.

