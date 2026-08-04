# ARTIS 함수 단위 parity 감사

작성일: 2026-07-31  
대상 Lumina commit: `47bfa2001deb`  
대상 ARTIS commit: `36f86476d870` (`../artis-ref`)  
주요 실행 표본: `logs/coevolve_consume_parity54/stdout.log`

## 0. 결론

현재 Lumina에는 ARTIS와 이름이나 주석은 비슷하지만 실제 상태 공간 또는 확률
분해가 다른 함수가 여럿 있다. 가장 먼저 고쳐야 할 것은 다음 여섯 묶음이다.

1. bound-free 흡수 사건이 실제 continuum을 확률적으로 고르지 않고, 주파수 bin의
   최대 opacity 기여 ion 하나로 고정된다.
2. ARTIS의 `nu_edge / nu` macro-atom 대 k-packet 분기가 구현되어 있지 않은데
   parity 배너는 D6가 구현됐다고 보고한다.
3. NLTE가 원소 전체 행렬이 아니라 겹치는 2-ion pair 행렬들의 순차 풀이이며,
   공유 ion 해를 다시 버리거나 기존 ion total에 재고정한다.
4. photoionization의 upper target level이 continuum별로 보존되지 않는다.
5. parity54에서 ARTIS에 없는 terminal line-destruction 추첨이 ARTIS식
   collisional-deactivation 추첨 위에 한 번 더 적용된다.
6. comparator인 `compare_bk_artis.py`의 두 번째 인자가 디렉토리와 timestep으로
   동시에 사용되어 정상 호출이 불가능하다.

따라서 현재 `LUMINA_ARTIS_PARITY=1`은 ARTIS parity 모드가 아니라 여러
ARTIS-inspired 보정과 Lumina/CMFGEN 실험 옵션이 섞인 hybrid 모드로 보는 것이
정확하다.

ARTIS는 이 감사의 white-box 방법 reference이다. 최종 물리 acceptance target은
여전히 CMFGEN이며, ARTIS와 다른 구현이 모두 잘못이라는 뜻은 아니다. 아래에서는
`확정 결함`, `확정 parity 불일치`, `의도적 근사`, `추가 검증 필요`를 구분한다.

## 1. 우선순위 요약

| 우선순위 | Lumina 함수/구간 | ARTIS 대응 | 판정 |
|---|---|---|---|
| P0 | `compute_bf_opacity`, `d_bf_get_activation_level`, GPU continuum-event branch | `calculate_chi_bf`, `rpkt_event_continuum` | continuum 선택 및 `nu_edge/nu` 분기 소실 |
| P0 | `compute_bf_opacity` | `calculate_chi_bf` | neutral bf 전부 제외, stimulated recombination correction 누락 |
| P0 | `nlte_solve_all`, `nlte_solve_all_gpu`, `nlte_solve_ion_shell` | `solve_nlte_pops_element` | pairwise SE와 element-wide SE의 구조 불일치 |
| P0 | `nlte_pair_total_density`, pair writeback/restore, `nlte_writeback_ion_stage` | `solve_nlte_pops_element`의 단일 원소 보존식 | 공유 ion 해가 고정·폐기되고 bulk ion fraction에 반영되지 않음 |
| P0 | `nlte_assemble_rate_matrix`, `build_recomb_topology`, `d_macro_atom_interaction` | `nltepop_matrix_add_ionisation`, `do_macroatom_ionisation` | continuum별 upper target topology 소실 |
| P0 | `d_macro_atom_interaction`의 `MA_LINE_DESTRUCT` | ARTIS macro-atom action fair draw | collisional deactivation을 추가로 한 번 더 추첨 |
| P0 | `compare_bk_artis.py:artis_bk`와 main | ARTIS 출력 비교 도구 | 인자 충돌, mean/median 및 level 범위 라벨 오류 |
| P1 | `nlte_build_perbin_dilute_field` | `radfield::select_bin`, `fit_parameters`, `radfield` | ARTIS와 다른 bin 경계와 EUV superbin |
| P1 | `compute_transition_probabilities`, `nlte_assemble_rate_matrix` | `rad_excitation_ratecoeff` | 실제 ARTIS build와 다른 line-field consumer |
| P1 | `d_update_base_estimators` 및 bf-rate 소비자 | `update_bfestimators`, `get_bfrate_estimator` | continuum-specific estimator를 generic frequency-bin moment로 대체 |
| P1 | `artis_col_rates` 호출부 | `col_excitation_ratecoeff`, `col_deexcitation_ratecoeff` | `f_lu`를 forbidden/collision-data metadata 대용으로 사용 |
| P2 | `d_ma_radrecomb_emit` | `select_continuum_nu` | 유한 횟수 rejection sampler의 bias 가능성 |

## 2. P0-1: bound-free 사건의 continuum 선택과 에너지 분해가 없다

### ARTIS

`artis-ref/rpkt.cc:405-445`의 `rpkt_event_continuum`은 다음 순서로 처리한다.

1. 현재 주파수에서 가능한 모든 continuum의 누적 opacity
   `phixslist.chi_bf_sum`을 만든다.
2. 그 누적분포에서 실제 흡수 continuum
   `(element, ion, lower level, phixs target)`을 추첨한다.
3. 선택된 continuum의 `nu_edge`에 대해 확률 `nu_edge / nu`로 upper target
   macro-atom을 활성화한다.
4. 나머지 확률 `1 - nu_edge / nu`는 photoelectron kinetic energy로 보아
   k-packet으로 보낸다.

### Lumina

- `src/lumina_plasma.c:6518-6538`의 `compute_bf_opacity`는 각
  `(shell, frequency bin)`에서 합산 opacity는 저장하지만, 사건 라우팅용으로는
  `best_chi`가 가장 큰 **ion 하나**만 `activation_level`에 남긴다.
- lower level과 continuum target의 누적분포는 전송되지 않는다.
- `src/lumina_cuda.cu:5480-5508`은 그 bin의 고정 activation level을 읽는다.
  해당 값이 있으면 `nu_edge / nu` 추첨 없이 항상 macro-atom으로 들어간다.
- 값이 없을 때만 임의의 k-packet excitation CDF에서 level을 뽑는다. 이것은
  흡수 photon의 continuum 또는 `nu_edge`와 관계없는 fallback이다.
- CPU의 `bf_absorption_event`와 GPU의 `d_bf_absorption_event`
  (`src/lumina_plasma.c:6680-6711`, `src/lumina_cuda.cu:3407-3437`)는 더 단순하게
  `B_nu(T_rad)`로 재방출한다.

이는 bin 해상도 오차가 아니라 사건 확률공간 자체가 다른 것이다. 작은 opacity의
continuum은 사건 라우팅에서 완전히 사라지고, photoelectron로 가야 할 에너지가
macro-atom으로 과도하게 들어갈 수 있다.

더 심각하게 `src/lumina_cuda.cu:6142`는
`D6 bf event nu_edge/nu ioniz-vs-kinetic split + stim-recomb corr`라고 출력하지만,
실제 GPU 사건 경로에는 이 분기가 없다. parity54 stdout에도 이 오배너가 출력됐다.

**판정: 확정 P0 구현 결함 및 상태 보고 결함.**

## 3. P0-2: `compute_bf_opacity`가 neutral bf와 stimulated correction을 누락한다

### Neutral bound-free

`src/lumina_plasma.c:6343-6348`은 `stage < 1`인 ion을 무조건 건너뛴다. 주석의
“no ionization from neutral ground to ion”은 물리적으로도 맞지 않는다. 현재 원자
모델은 O I 등 neutral stage를 포함하므로 neutral bf opacity와 neutral
photoionization을 동시에 제거한다.

ARTIS는 `nltepop_matrix_add_ionisation`과 `calculate_chi_bf`에서 neutral을 포함한
모든 유효 ionising level/continuum을 동일한 자료 구조로 순회한다
(`artis-ref/nltepop.cc:563-618`, `artis-ref/rpkt.cc:678-769`).

### Stimulated recombination correction

ARTIS `calculate_chi_bf`는 continuum마다

```text
corrfactor = max(0, 1 - modified_departure_ratio
                        * exp[-h(nu-nu_edge)/(kT_e)])
chi += n_lower * sigma_bf * target_probability * corrfactor
```

를 적용한다 (`artis-ref/rpkt.cc:733-765`).

Lumina는 `src/lumina_plasma.c:6477-6491`에서 단순히
`chi_contrib = n_level * sigma`를 더한다. `src/lumina_plasma.c:6496-6506`도
stimulated recombination을 버렸고 `chi_bf`가 uncorrected임을 명시한다.

따라서 D6 배너의 `stim-recomb corr` 역시 실제 opacity 구현과 맞지 않는다.

**판정: 확정 P0. CMFGEN target에서도 neutral continuum 누락은 별도 정당화가
어려운 결함이다.**

## 4. P0-3: NLTE solver가 ARTIS의 element-wide SE가 아니다

### ARTIS

`solve_nlte_pops_element`는 한 원소의 사용 ion stage와 level을 한 행렬에 넣는다
(`artis-ref/nltepop.cc:1165-1247`). bound-bound, bound-free, non-thermal,
autoionization을 같은 행렬에 더하고, 원소 전체 수밀도 하나로 정규화한다
(`:1249-1260`). 현재 ARTIS 설정은 `FORCE_SAHA_ION_BALANCE=false`이므로 ion
fraction도 이 rate-SE 해에서 나온다.

### Lumina

`nlte_solve_all`과 GPU 대응 경로는 II/III, III/IV 같은 인접 pair를 순서대로
`nlte_solve_ion_shell`에 보낸다 (`src/lumina_plasma.c:15933-16033`,
`src/lumina_cuda.cu:1040-1105`). parity gate는 iteration 수와 damping만 바꾸며
코드 자신도 element-wide matrix를 residual이라고 보고한다
(`src/lumina_plasma.c:15951-15959`).

`LUMINA_NLTE_NO_ML_LOCK=1`도 해결책이 아니다.
`nlte_pair_total_density`가 각 pair에 원소 전체 수밀도를 주기 때문에, 같은 원소의
겹치는 pair가 각각 전체 원소 수를 정규화 대상으로 삼게 된다
(`src/lumina_plasma.c:1421-1443`).

**판정: 확정 P0 구조 불일치. outer iteration을 늘려도 다른 연립방정식이 ARTIS
행렬이 되지는 않는다.**

## 5. P0-4: 겹치는 pair의 공유 ion 해가 고정되거나 버려진다

pairwise 구조 위에 다음 보정이 추가되어 있다.

- `nlte_solve_ion_shell`은 공유 slot을 포함한 pair이면 lower/upper ion을 기존
  `atom->ion_number_density` total에 각각 재정규화한다
  (`src/lumina_plasma.c:15534-15538`, `:15634-15655`).
- 뒤 pair가 앞 pair와 lower ion을 공유하면, 뒤 pair 풀이 전 population을 저장하고
  풀이 후 복원한다 (`src/lumina_plasma.c:15987-16031`,
  `src/lumina_cuda.cu:1633-1640`). 즉 뒤 pair가 계산한 공유 ion correction은
  의도적으로 폐기된다.
- `nlte_writeback_ion_stage`는 공유 pair를 모두 건너뛴다
  (`src/lumina_plasma.c:2321-2329`). Si II/III/IV, Fe II/III/IV,
  O I/II/III가 여기에 해당한다.
- parity54 effective config에는 `LUMINA_NLTE_OPACITY_IONSTAGE`도 없으므로 이
  writeback 함수는 시작 즉시 return한다.

결과적으로 pair matrix가 ion split을 계산하더라도 중요한 다단계 원소에서는 그
split이 bulk ion density와 opacity에 반영되지 않는다. `compute_ion_populations`
또는 별도 radiation-equilibrium ion solver가 소유한 ion total이 실제 bulk 상태로
남는다.

**판정: 확정 P0. “pairwise solve가 반복되어 element-wide 해로 수렴한다”는
해석은 현재 save/restore와 per-ion pin 때문에 성립하지 않는다.**

## 6. P0-5: bound-free upper target topology가 continuum별로 보존되지 않는다

ARTIS는 lower level마다 `nphixstargets`를 순회하여 target별 photoionization,
collisional ionization, recombination rate를 서로 다른 upper level에 연결한다
(`artis-ref/nltepop.cc:581-615`). macro-atom ionization도 rate-weighted target을
추첨한다 (`artis-ref/macroatom.cc:281-303`).

Lumina의 `nlte_assemble_rate_matrix`는 모든 lower level을 upper ion의
`ground_hi` 하나에 연결한다. 이 제한은 코드에도 residual로 명시되어 있다
(`src/lumina_plasma.c:14483-14502`).

`build_recomb_topology`가 여러 lower recombination destination을 만들기는 하지만,
photoionization 방향은 source level당 `iup_dest_level` 하나뿐이다
(`src/lumina_plasma.c:2886-2913`). GPU도 그 단일 destination으로만 이동한다
(`src/lumina_cuda.cu:4403-4420`).

`compute_bf_opacity`의 target 선택도 per-continuum이 아니다. 한 ion의 level을
훑다가 처음 발견한 `ma_rr_target` 하나를 ion 전체의 `rr_act`로 쓴다
(`src/lumina_plasma.c:6332-6339`).

**판정: 확정 P0 구조 불일치. recombination lower-destination 구현이 존재한다는
사실과 photoionization upper-target collapse는 구분해야 한다.**

## 7. P0-6: parity54의 `MA_LINE_DESTRUCT`가 ARTIS collisional draw를 중복한다

ARTIS는 macro-atom state에서 한 번의 fair draw로 다음 energy-flow action을 함께
경쟁시킨다.

- radiative deexcitation: `sum A_ul beta * epsilon_trans`
- collisional deexcitation: `sum C_ul * epsilon_trans`
- internal down: `sum (A_ul beta + C_ul) * epsilon_target`

근거는 `artis-ref/macroatom.cc:85-110`, 실제 선택은 `:392-429`이다.

Lumina parity 경로도 이미 `kp_deact += C_down * dE`를 만들고

```text
p_kpacket = kp_deact / (sum_rates + kp_deact)
```

로 ARTIS의 collisional-deactivation action을 추첨한다
(`src/lumina_plasma.c:4220-4240`, `:4516-4532`).

그런데 `LUMINA_MA_LINE_DESTRUCT=1`이면 radiative terminal line이 선택된 뒤 다시

```text
epsilon = C_down / (C_down + A_ul * beta)
```

를 뽑아 k-packet으로 보낸다
(`src/lumina_cuda.cu:4335-4356`).

단순히 radiative와 collisional 두 action만 있다고 하면 ARTIS의 올바른
collisional 확률은 `p=C/(A+C)`이다. 현재 두 추첨의 합성 확률은
`p + (1-p)p = 2p-p^2`가 되어 항상 더 크다. “pre-roll과 terminal branch가
control-flow상 동시에 실행되지 않으므로 double count가 아니다”라는
`src/lumina_cuda.cu:2101-2104`의 설명은 확률 measure의 중복을 해소하지 못한다.

parity54에서는 이 gate가 실제로 켜졌고 stdout은 device flag 1과 매 iteration
destroyed count를 기록한다. 이는 ARTIS에 없는 추가 thermalization이다.

**판정: 확정 P0 parity 오염. ARTIS 비교 lane에서는 이 gate를 끄고,
CMFGEN lane에서는 별도 two-level/ALI 수식으로 검증해야 한다.**

## 8. P0-7: `compare_bk_artis.py`는 현재 API로 정상 호출할 수 없다

`scripts/compare_bk_artis.py`의 사용법은 두 번째 인자를 ARTIS NLTE 디렉토리라고
정의한다 (`:8`, `:13`). 그러나 `artis_bk()`는 같은 `sys.argv[2]`를 정수
timestep으로 변환한다 (`:39`).

- 디렉토리를 주면 `int(directory)`에서 예외가 난다.
- timestep `27`을 주면 ARTIS 디렉토리가 문자열 `"27"`이 되어 데이터를 못 찾는다.

추가 오표기는 다음과 같다.

- `mn()`은 `numpy.mean`인데 마지막 문장은 `MEDIAN`이라고 쓴다 (`:66`, `:79`).
- 실제 범위는 level 1--4인데 출력은 `L1-7`이라고 쓴다 (`:34`, `:52`, `:79`).
- 주석은 아직 “highest S II median으로 timestep 선택”이라고 하지만 실제 코드는
  ts27 고정이다 (`:27-30`, `:37-39`).
- `artis_baseline_bk.py`의 usage는 기본 ts20이라 쓰지만 실제 기본은 ts27이고,
  ts27을 `19.49d`라고 출력한다. 실제 ARTIS midpoint는 20.2549 d이며 19.48 d는
  그 timestep bin에 포함된 epoch일 뿐이다 (`scripts/artis_baseline_bk.py:4-7`,
  `:25`).

**판정: 확정 P0 comparator 결함. 이 스크립트가 만든 기존 b-k verdict는
입력 manifest를 확인하기 전 acceptance 근거로 쓰면 안 된다.**

## 9. P1-1: `nlte_build_perbin_dilute_field`의 24-bin은 ARTIS 24-bin이 아니다

ARTIS 현재 설정은 다음과 같다
(`artis-ref/artisoptions.h:62-74`, `artis-ref/radfield.cc:102-145`).

- 24 bins
- 40000 Å부터 1085 Å까지 첫 23개를 **주파수 선형 간격**으로 분할
- 마지막 하나는 1085 Å부터 10 Å까지의 단일 EUV superbin
- 마지막 superbin의 `T_R`은 `T_e`로 고정

Lumina 기본 grid는 20000--100 Å의 1000개 log-frequency bins이다
(`src/lumina.h:491-493`). `nlte_build_perbin_dilute_field`는 fine-bin index를
`c = f * 24 / 1000`으로 묶으므로 24개 coarse bin도 log-frequency 간격이 된다
(`src/lumina_plasma.c:917-925`).

또한 1085 Å보다 짧은 영역을 하나의 superbin으로 적분하지 않고 여러 bin으로
쪼개 각자 `T_e` pin과 서로 다른 `W`를 준다 (`:940-966`). 코드는 이를 “faithful
generalisation”이라고 부르지만 rate field는 ARTIS와 달라진다. 범위도 ARTIS보다
red와 blue 양쪽에서 좁다.

**판정: 확정 P1 parity 불일치. 더 미세한 장이 CMFGEN 목적에는 유용할 수 있으나,
ARTIS parity 시험에는 exact-bin mode가 따로 필요하다.**

## 10. P1-2: line radiation-field consumer가 현재 ARTIS build와 다르다

현재 ARTIS는 `DETAILED_LINE_ESTIMATORS_ON=false`이다
(`artis-ref/artisoptions.h:74`). 따라서 `rad_excitation_ratecoeff`는 per-line
`Jb_lu`가 아니라 `radfield(nu_trans, cell)`을 사용한다
(`artis-ref/macroatom.cc:571-600`).

Lumina는 `LUMINA_ARTIS_PARITY=1`일 때 `g_ctp_iup_jblue`를 기본 ON으로 만든다
(`src/lumina_plasma.c:3408-3416`). parity54에서는 iteration 1 이후 macro-atom
up-rate line의 약 84--91%가 per-line J-blue를 사용했다. 이것은 현재 비교 중인
ARTIS executable의 consumer가 아니다.

`LUMINA_IUP_BINFIELD=1` 경로가 실제 ARTIS 설정에 더 가깝지만
(`src/lumina_plasma.c:3437-3448`), parity 기본값이 아니며 parity54에도 설정되지
않았다.

또한 ARTIS의 `get_Jb_lu`에는 contribution-count threshold가 없다
(`artis-ref/radfield.cc:650-653`). Lumina matrix와 macro-atom 소비자는 각각
`JBAR_MIN` 또는 자체 threshold/fallback 정책을 둔다. parity54의
`JBAR_UNIFY=1, JBAR_MIN=3`은 Lumina 내부 배선 통일에는 의미가 있지만 ARTIS
provenance는 없다.

**판정: 확정 P1 configuration/consumer 불일치.**

## 11. P1-3: bf estimator가 continuum-specific estimator가 아니다

ARTIS `update_bfestimators`는 현재 photon frequency에서 활성인 각각의 bf
continuum/target에 대해 `gamma_contr[continuum] * distance_e/nu`를 별도로
누적한다 (`artis-ref/radfield.cc:194-221`). 이후
`get_bfrate_estimator(element, ion, lower, target, cell)`가 그 continuum의
rate를 직접 반환한다 (`:828-868`).

Lumina `d_update_base_estimators`는 target을 모르는 generic frequency-bin
`sum(E * distance / nu)` 하나만 저장한다
(`src/lumina_cuda.cu:3263-3294`). 소비 시 bin-center cross section을 다시
곱한다.

fine bins가 충분하면 적분 근사로 사용할 수 있지만 다음 정보는 복구할 수 없다.

- 같은 lower level의 여러 upper target 확률
- bin 내부 cross-section 구조와 resonance
- Lumina frequency grid 밖의 기여

ARTIS 코드에는 estimator의 추가 Doppler factor에 대해 저자 자신의 TODO가 있으므로,
그 factor가 Lumina에 없다는 사실만으로 Lumina를 틀렸다고 판정하지는 않는다.

**판정: 확정 P1 방법 근사, Doppler factor 부분은 미판정.**

## 12. P1-4: collision helper보다 호출부의 metadata dispatch가 문제다

`artis_col_rates`의 permitted/forbidden 수식은 대체로 ARTIS 식을 충실히 옮겼다.
문제는 호출부가 ARTIS의 데이터 필드를 갖고 있지 않다는 점이다.

ARTIS는 각 transition의 `coll_str`과 `forbidden`을 읽는다. 양의 `coll_str`이면
실제 effective collision strength를 사용하고, 음수 sentinel일 때만
van Regemorter 또는 Axelrod fallback을 쓴다
(`artis-ref/macroatom.cc:685-770`).

Lumina parity 호출부는 대체로 `f_lu <= 1e-10`을 forbidden 판정의 대용으로 쓴다
(`src/lumina_plasma.c:14038-14051`, `:3888-3904`, `:4112-4130`). 따라서
oscillator strength가 작거나 0인 allowed transition, M1/E2 flag, 실제
collision-strength가 있는 transition을 ARTIS와 다르게 dispatch할 수 있다.
`LUMINA_MA_REAL_UPSILON`가 꺼진 경우에는 NLTE matrix와 macro-atom transport가
서로 다른 collision data tier를 쓸 가능성도 있다.

**판정: 확정 P1 data-model/dispatch 불일치. `artis_col_rates` 자체를 다시
튜닝하기 전에 transition metadata를 보존해야 한다.**

## 13. P2: `d_ma_radrecomb_emit` sampler는 ARTIS sampler와 동일하지 않다

ARTIS `select_continuum_nu`는 tabulated cross section을 포함한 Milne emissivity를
수치 적분하고, 누적 적분을 역으로 찾아 frequency를 고른다
(`artis-ref/ratecoeff.cc:496-544`).

Lumina `d_ma_radrecomb_emit`는 thermal exponential proposal에 대해
`sigma(nu) * nu^3` accept/reject를 최대 8회 수행하고, threshold 값으로 envelope를
잡는다 (`src/lumina_cuda.cu:4468-4516`).

두 가지 bias 가능성이 있다.

1. resonance 때문에 `sigma(nu) * nu^3`가 threshold 값보다 커지면 envelope가
   상한이 아니다.
2. 8회 모두 reject되어도 마지막 proposal을 채택한다.

Kramers `sigma ~ nu^-3`에서는 문제가 거의 보이지 않지만 실제 CMFGEN cross
section에는 별도 검증이 필요하다.

**판정: P2 추가 검증 필요. frozen-edge histogram을 ARTIS
`select_continuum_nu`와 직접 비교해야 한다.**

## 14. 이번 감사에서 “현재 결함”으로 다시 올리지 않은 항목

- `LINE_THERM`: parity54 환경에는 값이 있으나 effective device state는
  ARTIS parity에 의해 disabled다. active parity 차이로 다시 열지 않는다.
- collisional ionization/three-body recombination: 현재 matrix에는 parity gate
  아래 구현되어 있다. 남은 문제는 upper-target routing이다.
- electron scattering: 현재 ARTIS 설정은 dipole scattering이 아니며, Lumina의
  isotropic comoving 처리와 명백한 함수 parity 차이를 찾지 못했다.
- deterministic formal spectrum 대 ARTIS emergent MC spectrum: 방법 자체가 달라
  함수 일대일 parity 항목으로 분류하지 않았다. 다만 production formal energy
  non-conservation은 별도의 CMFGEN/formal P0 감사 대상이다.

## 15. 권장 수정 순서와 최소 acceptance test

### Gate A: comparator부터 봉인

1. `compare_bk_artis.py`를
   `compare_bk_artis.py <lumina_csv> --artis-dir DIR --timestep TS`처럼 분리한다.
2. 출력에 ARTIS commit, data directory, timestep index, timestep midpoint,
   requested epoch, level range, shell/cell selection, mean/median을 모두 기록한다.
3. empty ARTIS rows이면 성공 코드로 빈 verdict를 쓰지 말고 실패시킨다.

### Gate B: frozen-cell ARTIS method oracle

한 shell, 한 element, 작은 level set으로 다음을 독립 비교한다.

1. exact 24-bin `J_nu`: ARTIS bin 경계와 동일한가.
2. line rates: 같은 populations, `J_nu`, `T_e`, `n_e`에서 각
   `R_lu`, `R_ul`, `C_lu`, `C_ul`이 같은가.
3. continuum rates: lower/upper target별 `Gamma`, `alpha`, `C_ion`, `C_3b`.
4. element-wide matrix의 각 비대각 원소와 column sum.
5. bf event census: continuum ID 분포, macro-atom fraction
   `mean(nu_edge/nu)`, k-packet fraction.
6. macro-atom action census: RADDEEXC, COLDEEXC, INTERNALDOWN/UP,
   RADRECOMB의 확률.

### 구현 우선순위

1. continuum별 opacity CDF와 `nu_edge/nu` 사건 분기를 복원하고 D6 배너를
   실제 capability 검사로 바꾼다.
2. neutral bf 및 stimulated recombination correction을 복원한다.
3. S II--IV와 Fe II--IV부터 element-wide matrix pilot을 만든다.
4. atomic-data 변환기에 per-continuum upper target, target probability,
   forbidden flag, collision-strength source를 보존한다.
5. exact ARTIS bin/consumer mode를 별도 parity lane으로 만든다.
6. 위 frozen-cell test가 통과한 뒤에만 full spectrum을 ARTIS 방법 parity 또는
   CMFGEN acceptance의 증거로 사용한다.

## 16. 최종 판정

현재 ARTIS와의 가장 큰 차이는 상수 하나나 damping 값이 아니라 다음 상태
연결의 소실이다.

```text
continuum identity -> upper target -> nu_edge energy split -> macro-atom/k-packet
element identity   -> all ion stages -> one conservation row -> ion fraction
transition identity -> collision metadata -> rate/action probability
```

이 연결을 복원하지 않은 상태에서 `JBAR_MIN`, damping, temperature pin, empirical
thermalization gate를 조정하면 spectrum은 움직일 수 있지만 ARTIS 방법 parity가
개선됐다고 판정할 수 없다. CMFGEN과의 일치 역시 같은 이유로 우연한 보상에
머물 가능성이 높다.
