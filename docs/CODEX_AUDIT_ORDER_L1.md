# 발주서 검수표

판정은 발주서의 실행 가능성에 대한 `확인·반박·미결`이다. 물리 불일치 자체를 최종 판정한 것이 아니다.

## 1. §2 가설별 검수

| 항목 | 판정 | 실측 근거(명령과 출력) | 발주서 수정 요구 |
|---|---|---|---|
| §2 대상 수 | **반박** | `python3`로 §2 ID 추출 → `['I1','I2','I2a','I2b','I3','I3c','I4','I5','I6','I7','I8','I9','I17']`, `count=13` | “12개”를 **13개**로 고치고 정확한 대상 목록을 표제에 고정하라. |
| I1 Co IV Υ | **미결** | `col_guess.dat` → `0 !Number of transitions`, `1.0 !Scaling factor`, `0.1 !Value for OMEGA if f=0`.<br>`omega_gen_v3.f:158-203` → 비표 전이는 `EIN_A<=1e-5`이면 `OMEGA_SET`, 나머지는 oscillator/A 기반 근사식.<br>실행 후보 `cmfgen_dev.exe`: mtime `2026-07-15`, SHA `f2b9af…`; 결과 `batch.log`: `2026-07-18`, `CMFGEN_EXIT=0`. 그러나 실행 당시 바이너리 해시는 기록되지 않았다. | “CMFGEN 런의 입력표 0개”는 확인 가능하지만, **그 런이 현재 소스의 폴백을 사용했다는 확증은 없음**. 실행 당시 binary SHA/build commit 또는 런타임 collision-branch census를 선행조건으로 추가하라. |
| I2 `A_ul` 유효숫자 | **확인** | 연결 osc 전수 실측: Fe IV `A_sig={5:72223}`, S III `{5:6190}`, 대부분 이온 5자리, Si IV 4자리.<br>실제 Lumina `line_list.csv` 2,584,132개 분석 → `<=5 digits = 2584131/2584132`.<br>생성기 저장 형식은 `.6e`이나 원자료 정보량은 회복되지 않는다. | `r>1e-6`은 원자료의 약 `1e-4`(일부 `1e-3`) 상대 정밀도보다 엄격하다. 이를 물리 불일치 임계로 쓰지 말라. **동일 원본 변환 무결성은 exact/ULP**, 서로 다른 원본의 물리 비교는 양자화 구간을 반영한 별도 임계로 분리하라. |
| I2a Fe IV | **확인(단, epoch 조건부)** | 직접 집계: 구 덱 Fe IV `4,336선`, CMFGEN 원본 `72,223선`, 차 `67,887선`; `_ftos`는 `72,223선`.<br>저장된 Python 재현기 검색: `rg --glob '*.py' '880406|75075|strict.*A_ul'` → 출력 없음. | “남은 2선”의 line ID, 양쪽 원자료 행, 레벨 결합키, `f/gf/A` 정의를 명시하라. 또한 구 `_sivcaiv`와 `_ftos` 결과를 섞지 말고 **양 epoch에서 재대조**하라. |
| I2b Ni IV | **확인(단, epoch 조건부)** | 기존 산출물 검색 → `3,658/4,085 (89.5471%)`.<br>직접 집계: 구 덱 Ni IV 전체선 `4,199`, `_ftos` `72,898`; CMFGEN 원본도 `72,898`. | “σ는 동일, A만 다름”은 구 덱 판정이다. `_ftos`에서 line·level·σ가 모두 바뀌므로 I2b도 재실측해야 한다. I17만 새 덱으로 재는 현재 범위는 epoch 혼합이다. |
| I3 σ(ν) | **반박** | 현재 생성기에는 `_bin_average_sigma()`와 `BF_N_FREQ_BIN=1000`이 있다.<br>그러나 실소비 구 덱 바이너리: mtime `2026-07-28`, SHA `7135be…`; `bakefix2`는 `2026-07-29`; 현재 생성기는 mtime `2026-08-03 23:22`, git 상태 `M`이며 1,092행 미커밋 diff.<br>바이너리 헤더는 `version=1, nlevels=26592, nbin=1000, ν=[1.5e14,3e16]`만 저장한다. 평균법·생성 commit·옵션 필드는 없다. | “실소비 σ가 bin-averaged”라는 사실 전제를 삭제하라. **[추론]** mtime과 생성기 주석은 오히려 구 바이너리가 point-sampled였을 가능성을 지지한다. point-sample/average 두 재생성 가설과 바이트 또는 수치 대조로 bake semantics를 먼저 확정하라. |
| I3 CMFGEN σ 격자 | **확인** | `OUTGEN:91-93` → `The continuum will be evaluated at 15662 frequencies`, `Number of frequencies is: 196185`.<br>`cmfgen_sub.f`는 총 `NCF`를 순회하되 `NU_EVAL_CONT`가 바뀔 때만 새 cross-section을 계산한다. | CMFGEN σ 평가 격자를 `196,185`로 쓰지 말고 **15,662 continuum evaluation points**와 총 수송 격자 `196,185`를 구분하라. 공통 함수값 대조인지, bin 평균인지, 물리율 적분인지 하나를 사전 고정하라. |
| I3c Fe IV·Ni IV σ | **반박** | 구 덱 레벨: Fe IV `200`, Ni IV `200`.<br>`_ftos` 헤더/flag 직접 판독 → Fe IV `levels=1000, sigma_flag_1=1000`, Ni IV도 `1000/1000`.<br>R1 gate → sigma addressable `26592→31792`, present `26087→31237`, PASS. | “200/1,000만 결합 가능”은 구 epoch 상태다. I3c도 `_ftos`에서 임계 `1e-6/1e-9/1e-12`, zero/positive 네 상태를 다시 집계하라. I17에만 새 덱을 적용해서는 안 된다. |
| I4 슈퍼레벨 | **확인** | 캡처 env: `LUMINA_SUPER_CUTOFF=100`, `LUMINA_SUPER_LEVELS=1`.<br>stdout: `K=100: 21581 levels lumped`.<br>소스: `super=min(level_num,K)`로 로드된 `f_to_s`를 덮어씀. | 측정은 가능하다. 다만 “잘 정의된 설계 차이”를 자동으로 “수리 대상”으로 보내는 §1 exit가 잘못됐다. **B/ACCEPTED-DESIGN** 처분을 별도로 두라. |
| I5 재결합·DR | **미결** | CMFGEN `VADAT`: `F,F [DIE_CoIV]`.<br>현재 Lumina 소스 NLTE 행렬 경로는 `dr_lookup()` 후 `R_dr`을 조건 없이 추가한다. `LUMINA_FROZENIN_DR=0`은 별도 frozen-in 경로만 끈다.<br>그러나 캡처 바이너리와 현재 소스의 정확한 계보는 미확립이며 RR 전수 대조도 없다. | I5를 **I5a DR 설정**, **I5b level-resolved RR/Milne**로 분리하라. DR은 binary provenance를 붙이고, RR은 동일 `T_e`, `Jν`, target mapping, 격자·적분 규칙을 지정해야 한다. |
| I6 모델 덱 | **확인** | Lumina: `time_explosion_s=1683072`, geometry/density `50행`, abundance `8×50`.<br>CMFGEN: `Time=19.4800000 d`, `MODEL: 90 depth points`.<br>따라서 시간은 동일하고 공간·밀도·속도 배열은 남는다. | 조성은 I6에서 제거하되 I6을 시간/공간범위/속도/밀도로 분해하라. 서로 다른 영역은 공통 속도구간과 외삽 금지 규칙을 명시해야 한다. |
| I7 격자 | **반박** | Lumina `NLTE_N_FREQ_BINS=1000`은 NLTE `Jν`·bf opacity·σ bake용 로그 빈.<br>CMFGEN `196185`는 선 삽입 후 총 수송 주파수점이고, continuum/σ 실제 평가점은 `15662`. | `1,000 대 196,185` 행을 삭제하라. 비교하려면 `Lumina NLTE continuum bins 1000 ↔ CMFGEN NU_EVAL_CONT 15662` 또는 두 코드의 최종 적분 오차처럼 **동일 역할끼리** 정의하라. |
| I8 경계조건 | **미결** | Lumina stdout: `L_inj=3.094761e42`, `r_in=6.5640e14`, `T_inner=10020K`; deposition 포함 `L_total_in=1.088240e43`.<br>CMFGEN: `LSTAR=2.60e7 Lsun`, `DIF=T`, `IB_METH=DIFFUSION`.<br>같은 반지름에서의 CMFGEN `L(r)` 추출 명령·파일은 발주서에 없다. | `31.07` 비를 판정 입력에서 제외하라. 비교 속도좌표, `L=4πr²F`의 flux 정의, gamma deposition 포함 여부, CMFGEN 파일/열/단위, 보간법을 명시해야 실행 가능하다. |
| I9 수치 상수 | **미결** | Lumina: `eps_floor=1e-5`, `eps_cap=1`, 외부 반복 `12`, damping `0.5`.<br>CMFGEN: `NUM_ITS=4`, `NUM_LAM=2`, `ACC_F=1e-4`, `EPS_TERM=0.1%`.<br>이는 서로 적용 대상이 다른 묶음이다. “ε 대응물 없음”은 정확한 실행 바이너리 전체 의미론을 닫지 않고는 확증 불가하다. | I9를 ε clamp/외부 반복/Λ 반복/damping/종료조건/온도탐색으로 분해하라. **NO-COUNTERPART**를 판정값으로 추가하고, 이름이 비슷한 상수의 단순 수치 비교를 금지하라. |
| I17 커버리지 | **확인** | `_ftos` R1 gate: Fe IV `72223/72223 PASS`, Ni IV `72898/72898 PASS`; 27이온 rank identity 전부 PASS.<br>σ flag: Fe IV `1000/1000`, Ni IV `1000/1000`.<br>G7: 9개 파일 모두 `OK`.<br>단 저장된 `verification.log`는 `ERROR: R4 verifier contract failure: 'NoneType'...` 1행이다. 현재 read-only `gate_ftos`와 R1 gate는 PASS. | I17은 `_ftos`에서 **명목상 해소됨**을 판정할 수 있다. G7은 “조성 수리가 파일을 안 바꿈”만 증명하므로 충분하지 않다. I17 선행조건으로 `gate_ftos + verify_deck_r1_vintage`를 명령 그대로 넣고, stale 실패 로그의 처분도 명시하라. |

## 2. 판정 체계·함정·범위·작업량

| 항목 | 판정 | 실측 근거(명령과 출력) | 발주서 수정 요구 |
|---|---|---|---|
| §1 세 판정값의 완전성 | **반박** | §1 정의상 `WELL-POSED`는 곧 “수리 대상”, `ILL-POSED`는 규칙 의존, `ARTIFACT`는 잣대 오류다. 그런데 실측 I4는 잘 정의된 설계 차이, I9는 대응물 없음, I17은 새 epoch에서 해결됨, I5는 부분 판정이다. | 최소한 축을 분리하라: **posedness**=`WELL/ILL/UNVERIFIABLE`; **outcome**=`MATCH/DIFFER/NO-COUNTERPART/RESOLVED/PARTIAL`; **kind**=`BUG/DESIGN/DEFINITION/COVERAGE/PROVENANCE/NUMERIC`; **disposition**=`REPAIR/ACCEPT/DEFINE/REMEASURE`. |
| §3 네 필수 명시 | **반박** | 분모 함정→분자·분모, 좌표 함정→좌표·단위·정의, 표본 함정→표본·평균, I2 정밀도→임계·유효숫자는 대응한다. 그러나 오늘의 `n_e 1.92×` 사고는 **잘못된 oracle/epoch**였고 이 네 항목 어디에도 없다. I3 실제 바이너리 bake provenance 결손도 동일하다. | 다음을 추가하라: ① 권위 원본·실소비 경로·symlink target, ② binary/source SHA와 epoch, ③ 결합키와 중복 처리, ④ missing/zero/unsupported 네 상태, ⑤ 영분모·절대/상대 오차 규칙. |
| 누락된 기존 층 1 판정 | **반박** | 자동 추출: 대장 층 1 ID `18개`; §2 `13개`; 누락=`I2c,I2d,I3a,I3b,I10`. | `I2c Co IV`, `I2d Fe III 제거`, `I3a Co IV`, `I3b Fe III`, `I10 γ-deposition epoch 승계`를 포함하거나 제목·계약을 “선정 13건”으로 축소하라. 지금의 “층 1 각 불일치” 계약과 모순된다. |
| I11–I16 별건 처리 | **반박(부분 허용)** | 대장은 열거 완전성을 `UNRESOLVED`로 기록하고 I11–I16을 신규 누락으로 둔다. 그중 I15는 binary·환경, I16은 symlink이며 이번 비교 대상의 실소비 여부를 결정한다. | I11·I13·I14의 독립 물리 비교는 별건 가능하다. 그러나 **I15 binary/build/env와 I16 symlink**, I12의 level/line identity 부분은 이번 감사의 선행 provenance gate여야 한다. 모두를 후속으로 미루면 WELL-POSED 판정을 확증할 수 없다. |
| 단일 epoch 계약 | **반박** | 구 덱→`_ftos`: levels `26,592→31,792`, Fe IV lines `4,336→72,223`, Ni IV lines `4,199→72,898`, σ SHA도 변경. 그런데 발주서는 I17만 `_ftos`로 재측정하고 나머지는 구 캡처 수치를 유지한다. | 각 행에 `specimen/deck`, atomic SHA, binary SHA, CMFGEN run snapshot을 강제하라. 구 캡처 판정과 현재 `_ftos` 판정을 별도 열로 유지해야 한다. |
| 재현 가능성 | **반박** | `rg --glob '*.py' '880406|75075|3953894|1233529|r_sigma'` → 저장된 재현 스크립트 없음. 데이터는 구 σ `212,762,624 B`, `_ftos` σ `254,367,824 B`, line CSV는 각각 258만·222만 행이다. | 요약문 수치를 감사자가 처음부터 재구현하게 하지 말고 I2/I3 공통 read-only 재현기를 발주 입력으로 지정하라. 명령, version, 예상 분모와 checksum을 계약에 포함하라. |
| 작업량 판정 | **반박** | 13개 가설, 누락 5개, 두 atomic epoch, 880,406선·3,953,894점 재계산, I8 CMFGEN luminosity source trace, 대안 표본 규칙 2개씩을 한 산출물에 요구한다. 핵심 재현기도 없다. | 한 번의 감사로 발주하지 말고 아래 4건으로 분할하라. |

권장 분할:

1. **A — 표본·계보·판정어휘 게이트**: 정확한 deck/binary/source/SHA/symlink, missing·zero 규칙, 판정 축 확정. 이 단계 실패 시 후속 판정 금지.
2. **B — 원자 선·충돌·커버리지**: I1, I2–I2d, I4, I17. 유효숫자와 strict matching 재현기 포함.
3. **C — 광이온·재결합**: I3–I3c, I5. point/average bake 판별 후 공통 함수값 및 물리율 적분 대조.
4. **D — 모델·격자·경계·수치계약**: I6–I10. 공통 좌표와 대응물 유무를 먼저 확정.

반려