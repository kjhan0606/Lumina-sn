# 판정문 — A210-Z2A R-III·R-IV 게이트(G5·G6), 개정판 2

## 머리말

| 항목 | 내용 |
|---|---|
| 단 | A210-Z2A-DIAG-ISOLATION의 R-III·R-IV 잔여 게이트(G5·G6) |
| 판본 사슬 | 개정 전 판정문(sha256 `2953d271f686a8aac7e61076c763c54a6a77fb0e7380b936f80b584165b6f040`) → byte-보존본 `VERDICT_Z2A_RIII_RIV.md.rev1` → 이 파일(개정판 2) |
| 개정 사유·범위 | 감리-A의 «폐합 차단» 2건과 user의 개정 2 철회를 반영해 갈림 (d) 및 그 결론에 기대는 문면만 정정. G5·G6·(a)·(b)·(c)·분기 판정식은 변경하지 않음 |
| 계약 정본 — 판정 시점 | `ddf6701f` 커밋의 `docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md` blob (sha256 `8f5208ad98e140d1263800abe4778f85c4a69f1d47721d3300d9e9e0a6d9bb36`). 개정 전 판정문이 고정 인용한 값은 이 시점의 값으로 정확함 |
| 계약 정본 — 사후 판본 | 감리-B 시점에는 판정 뒤 §10 행 1건이 더해진 작업트리(sha256 `258f88ad9fdedc307eafb66541ebc688624ef8189437a8bd95ca764e0f246749`). 이 개정판 판정 시점에는 철회·무효 및 §10 귀속까지 반영된 작업트리(sha256 `71bb08c6e2561ac3f15a442639b2ba419e814c166d13080edbed3d02814374ff`) |
| 폐기·개정 전 보존본 | `docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md.rev1` (sha256 `f292cde8563bd8f069d3a62d0af8572c7de874a42b93182a8bdc7b2c42a1284d`) |
| 판정자 | Codex fresh 컨텍스트, 2026-08-29. 사전등록·구현·판독·선행 판정·감리에 관여하지 않음 |
| 판독 출처 | 운전석 계측 `/gpfs/kjhan/lumina/z2a_gates/Z2A_RIII_RIV_READOUT.md`. 그 문서의 PASS/FAIL·분기 라벨과 계약 지목은 받아쓰지 않고 계약 원문과 런·JSON에 재적용함 |
| 런 | `<RIII>=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/riii_20260827T233659Z_ion2`; `<RIV>=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/riv_20260827T233659Z_ion3` |
| 장부 | 장부 = 정본 §10(개정16 ①). 이 판정문은 장부를 복제하지 않음 |
| 쓰기 경계 | 이번 개정의 쓰기는 이 판정문과 요구된 `.rev1` byte-보존본뿐. 봉인·런루트·다른 게이트 산출물 쓰기 0. 부록의 기존 재실행 산출은 `verdict_scratch2/`에만 기록됐고 Python은 전부 `python3 -B` |

### ★정본 판본 시점과 개정 2 효력의 독립 판정

개정 전 판정이 읽은 정본은 확장자 없는 `.md`의 `ddf6701f` 커밋 blob이고 `.md.rev1`은 개정 2 전 보존본이다. 다음 네 축은 **그 판정 시점의 판본 관계**를 확정한다.

1. **Git 트리와 커밋**: `ddf6701f3f2805fe745c96a0a68b2c841024e5eb`는 기존 `.md` blob `578dd61…`을 `.rev1`로 similarity 100% copy하고, `.md`를 blob `ff9fd3…`으로 개정했다. 커밋 메시지도 `.rev1`을 “개정 전 보존”이라고 명시한다.
2. **문서 자기 서술**: 판정 시점 `.md` §13 개정 2는 `.rev1`을 “개정 전 판본 보존”이라 명명하고, §5.5 두 제출 시점의 교체와 §5.5-1 신설을 기록한다.
3. **행 단위 diff**: 판정 시점 `.rev1 → .md`는 추가 149행·삭제 3행, 네 hunk다. 삭제 3행은 §5.5 표 2행과 §10 사전등록 행 1행이고, 후자는 원문 전부를 보존한 채 개정 2 기록을 덧붙였다. §4 G5·G6와 §8 기대치·분기에는 delta가 없다.
4. **mtime·해시**: `.rev1` mtime은 2026-08-28 07:04:30 KST, 판정 시점 `.md`는 07:08:21 KST이며 해시도 위 Git blob 방향과 일치한다. mtime은 보조 근거이고 Git·자기 서술·diff가 주 근거다.

그러나 이 판본 관계는 개정의 **형식 효력**을 만들지 않는다. 상위 규약 `CLAUDE.md`의 「사전등록 개정의 형식」은 개정을 축자 `OLD:`/`NEW:` 쌍으로만 받도록 정한다. 개정 2는 §13에서 `OLD → NEW (요지)`로 약 90행의 §5.5-1 신설을 요약했으므로 이 요건을 충족하지 못해 무효다. 더구나 R-II 실패와 선행 「제출 불가」 판정을 확인한 뒤, 바로 그 FAILED R-II가 통과하도록 제출 술어를 구성했다. 이는 형식 위반과 별개의 결과 확인 뒤 골대 이동이다. user가 2026-08-29 개정 2를 철회했고, 현 정본은 §5.5 제출 시점을 구 문면 「R-II 런처 수용 PASS 후」로 복귀시키며 §5.5-1을 역사 기록으로만 보존한다.

개정 2의 **논거 자체**, 즉 구 조건이 봉인 census 자기입력에서도 실패한 판별력 0의 판독기 rc를 제출 자격의 필요조건으로 만든다는 지적은 기각되지 않았다. 다만 그 사실은 무효·철회된 개정에 규범 효력을 주지 않는다. 이 갈림의 시정은 이 판정 단의 대상이 아니다.

운전석은 1차 발주서에서 이 관계를 거꾸로 써 `.rev1`을 정본, `.md`를 비교판으로 지정했다. 1차 판정자 `VERDICT_Z2A_RIII_RIV.SUPERSEDED_wrong_contract.md`는 그 결함 있는 지시를 따랐고 특히 갈림 (d)에 폐기 조문을 적용했다. 이는 **1차 판정자의 잘못이 아니라 운전석 발주서 결함**이다. 1차 판정의 결론은 이 판정의 근거로 사용하지 않는다. 운전석 판독문 첫머리도 폐기본을 계약으로 지목하므로 그 라벨은 채택하지 않는다.

표기: **PASS**는 해당 등록 연언지가 실측으로 참임을 뜻한다. **미착지**는 등록 연언 전체가 성립하지 않음을 뜻한다. **계약 결함**은 실물이 내는 경로와 등록 경로가 어긋나 생긴 미착지 원인이다. 선행 R-II 폐합 판정과 ZI-A는 재판정하지 않는다.

## 1. 게이트별 판정 — 계약 §4 연언지와 1:1 대조

### 1-1. G5 — III Stage-4

| 실측 | 등록 요구 | 판정 |
|---|---|---|
| job 325913 `COMPLETED 0:0`; `model.rc` 정확 `0`; SUMMARY 2건 모두 `complete=1`; R7 iter1 커밋 1건; PHYSICS-COMPARISON iter1 COMMITTED 1건; RUN_FOOTER 첫 행 스키마 실재 | R-III 완주 | **PASS** |
| SUMMARY 정확 2건, 두 행 모두 `target_ion=2` | SUMMARY 2줄(`target_ion=2`) | **PASS** |
| `[A2-10][LINE-SATURATION-ROW]` 5,375건; 세 producer 필드 전부 보유 5,375건·결손 0. S2 판정기의 중복 키 제거 후 원장 `row_count=5264` | Stage-4 row이며 producer 필드 보유, 1건 이상 | **PASS** |
| 판정기 독립 재실행: S2 `pass=true`, `max_relative_deviation=0.0`, tolerance `4e-16`, zero-sum 위반 0, producer terms undefined 0 | S2: `Jbar == producer_continuum_term + producer_local_emission_term`, tol `4e-16` | **PASS** |
| ZERO-OPACITY-ROW 총 364,488건 = 반복당 182,244건. Z26 66,714 / Z27 265,084 / Z28 32,690. 고유 `(Z,line,lower_global,upper_global)` 182,244개가 각각 정확 2회 출현 | ion=2 hazard census 보고 | **PASS** |

**G5 = 충족.** §4의 모든 연언지가 성립한다.

### 1-2. G6 — IV byte 불변

| 실측 | 등록 요구 | 판정 |
|---|---|---|
| 좌·우 각 30,637 B, sha256 `e5843457e06c0252ff19ad17a6a683d389a40b0921f08928f0343cb0d18a921a` | `work/lumina_spectrum.csv` byte-동일 | **PASS** |
| 좌·우 각 30,637 B, sha256 `ff5e450c66b8d013720803ec1965a5400a5a469cc5540c9e5413ed37ffd546e3` | `work/lumina_spectrum_comoving.csv` byte-동일 | **PASS** |
| 좌·우 각 1,457 B, sha256 `e2dadf595a946c283a0fe9af5bf79a4bed5855c05ec1c1c0a1f5e9f50bd43dc0` | `work/lumina_plasma_state.csv` byte-동일 | **PASS** |
| 좌·우 모두 부재 | `work/physics_DET_iter0000/manifest.json` byte-동일 | **미착지 — 계약 결함** |
| 좌·우 모두 부재 | `work/physics_DET_iter0000/shell.csv` byte-동일 | **미착지 — 계약 결함** |
| 좌·우 모두 부재 | `work/physics_DET_iter0000/spectral.csv` byte-동일 | **미착지 — 계약 결함** |
| 좌·우 모두 부재 | `work/physics_DET_iter0001/manifest.json` byte-동일 | **미착지 — 계약 결함** |
| 좌·우 모두 부재 | `work/physics_DET_iter0001/shell.csv` byte-동일 | **미착지 — 계약 결함** |
| 좌·우 모두 부재 | `work/physics_DET_iter0001/spectral.csv` byte-동일 | **미착지 — 계약 결함** |
| SUMMARY 정확 2건, 모두 `zero_opacity_emitting_rows=0` | SUMMARY `zero_opacity_emitting_rows=0` | **PASS** |

등록 비교기 재실행은 9경로 중 3개만 읽어 `IDENTICAL`, 나머지 6개를 양쪽 `FileNotFoundError`로 기록해 `compared_files=3/9`, rc=2, `FAIL_CLOSED`를 냈다. 대칭적 부재에는 비교할 두 byte열이 없으므로 byte-동일도 byte-불일치도 아니다.

비게이트 보강 관측으로, 양 런이 실제 방출한 대응 경로는 `work/physics_comparison/physics_DET_iterNNNN.{manifest.json,shell.csv,spectral.csv}`이고 그 6쌍과 위 CSV 3쌍은 9/9 byte-동일했다. 그러나 판정자가 실제 경로로 계약을 바꾸어 읽을 수 없으므로 이 관측을 G6 PASS로 사용하지 않는다.

**G6 = 미착지, 원인 = 계약 경로 결함.** 존재하는 byte의 불일치는 0건이다.

### 1-3. 음성대조 자격

| 실측 | 등록 요구 | 판정 |
|---|---|---|
| NC-G5 입력은 성한 R-III 로그와 정확 3바이트가 다르고 전부 ASCII `1→0`이다. 재실행 결과 `producer_terms_undefined_rows=3`, `verdict=UNDECIDABLE_PRODUCER_TERMS_MISSING`, S2 키 부재, rc=1. 성한 입력의 등록 S2는 PASS | `producer_terms_defined=0` 행 주입 → S2 판정기 UNDECIDABLE fail-closed | **등록 형태로 시연 성립 — G5 PASS 자격 있음** |
| NC-G6 `pos.json`은 스크래치 3쌍의 동일 해시와 PASS를 기록한다. 그 뒤 `lumina_plasma_state.csv` 0-origin offset 728의 한 byte가 `+`(43)→`9`(57)로 바뀌었고, `neg.json` 및 독립 재실행은 해당 파일을 `DIFFERENT`, rc=1 FAIL로 지목했다. 현재 left-vs-left control은 rc=0 PASS | 스크래치 사본 1바이트 변조 → byte 비교기 FAIL | **등록 형태로 시연 성립 — G6의 PASS 자격 요건 자체는 있음** |

NC-G6의 right 사본은 음성 주입 후 상태라 `pos.json` 명령을 현재 right에 축자 재실행할 수 없다. 다만 `pos.json` 생성(23:53:18.356) → right 변조(23:53:18.396) → `neg.json` 생성(23:53:18.460)의 시간 순서, pos의 세 right 해시와 현 left 원본 해시 일치, 현 한-byte delta 및 neg 재현으로 양성→음성 시연 사슬을 확인했다. 음성대조 자격은 본게이트 6경로 부재를 대신하지 않는다.

## 2. 기대치 E-I5~E-I8

| 기대치 | 착지 판정 | 근거 |
|---|---|---|
| E-I5 | **착지 — PASS** | 완주·target_ion=2 SUMMARY 2건·Stage-4 row≥1·등록 S2 PASS. S2b는 이 연언에 없음 |
| E-I6 | **관측 착지** | ion=2 hazard 총 364,488행(반복당 182,244), Z별 66,714/265,084/32,690. 로그가 준위쌍 식별 전행을 보유하며 고유 키 182,244개가 양 반복에 각 1회 |
| E-I7 | **조건부 관측 착지** | `line=262210` 정확 2건(각 반복 1건), Z27 ion2, `lower_global=13428`, `upper_global=16541` 등 전 필드가 10020 K 런에 기록됨. 계약대로 **10020 K 스코프에만** 해당하며 S4-III 매칭-T 조건의 일반 미결을 해소하지 않음 |
| E-I8 | **미착지 — 계약 결함** | SUMMARY의 `zero_opacity_emitting_rows=0`·`sprobe_undefined_rows=0`은 양 반복 PASS. 그러나 G6 등록 9경로 중 6개가 양쪽 부재라 Tier1 연언 불성립 |

## 3. 갈림 4건 결론

### (a) G5의 S2 대 도구 종합 verdict

G5는 **충족**한다. 계약 §4 G5와 §8 E-I5가 등록한 것은 S2(tol `4e-16`)이고 재실행 S2는 `pass=true`, 최대 상대편차 `0.0`이다. 도구 소스는 종합 verdict를 `S2.pass ∧ S2b.pass`로 만들며, 계약 밖 S2b가 `max_relative_deviation=0.4236527496808573`, `missing_j_cont_rows=389`, tol `1e-12`로 실패해서 종합 `FAIL`이 됐다. S3는 `evaluated_rows=4875`, `excluded_one_minus_beta_zero=389`다.

따라서 §8-2 ZI-P의 `R-III S2 FAIL/UNDECIDABLE`은 성립하지 않는다. S2b 실패는 비게이트 발견으로만 조용히 기재하며 이 단에서 물리 결론·수리안·별도 구현 상태를 부여하지 않는다. S2의 항등이 물리 검증이라는 주장도 하지 않는다.

### (b) G6 선언 파일 6건의 대칭 부재

G6는 **미착지이고 원인은 계약 결함**이다. 열거된 9개 파일의 byte-동일은 연언인데 6개는 두 항 모두 없어서 연언을 만족하지 않는다. 동시에 어느 쪽 byte도 존재하지 않아 `R-IV 값 파일 byte 불일치`도 성립하지 않는다. 따라서 ZI-P로 승격하지 않는다. 비교기의 `FAIL_CLOSED`는 `DIFFERENT`가 아니라 읽을 파일 부재를 이름 있게 받은 결과다.

### (c) 봉인 L6 기준의 정당성

운전석이 좌항으로 쓴 다음 경로는 계약이 뜻한 **봉인 L6로 계보상 일의 확정**된다.

`/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe`

근거: ① 상류 계약 `docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md:99-102`가 IV의 봉인 L6를 이 절대경로로 직접 지목한다. ② 상류 판정 `docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md:30`도 같은 경로를 봉인 L6로 명명한다. ③ 등록 스테이저 `scripts/stage_a210_z2a_probe.sh:7`의 `SEALED_L6_ROOT`와 `<RIV>/Z2A_STAGE_SEAL.txt`의 `sealed_base`가 같은 경로다. ④ 그 루트의 `L6_STAGE_SEAL.txt`와 `L6_PROBE_VERDICT.txt`가 `A210_L6_PROBE_ACCEPT status=PASS`, job 321636, 동일 run_root를 기록한다.

현 정본 자체에 절대경로가 다시 적히지 않은 추적성 결손은 관측으로 남지만, 상류 등록물과 stage seal이 한 실물을 가리키므로 기준 미확정은 아니다. 이 결론은 G6 경로 결함을 해소하지 않는다.

### (d) 제출 자격 — 정본 §5.5와 철회된 §5.5-1 재판정

**이번 R-III·R-IV 제출은 자격 이탈이다.** 유효한 정본 조건은 개정 전 문면 **「R-II 런처 수용 PASS 후」**이고, 그 조건을 충족하는 실물이 없다.

| 항 | 유효한 요구·확인점 | 판정자 실측 | 판정 |
|---|---|---|---|
| R-II 종료 상태 | R-II 런처 수용 PASS가 선행해야 함 | `sacct -j 324557 ...` 결과는 `324557\|FAILED\|70:0`; `slurm-324557.out`은 정확 1행 `DET_FLIGHT_FATAL Z2A R-II analyzer failed`; `<RII>/z2a_probe.stdout`은 `A210_Z2A_PROBE_ANALYZER_RESULT status=FAIL branch=ZI-B ...` | **미충족** |
| 수용 PASS 실물 | `A210_Z2A_PROBE_ACCEPT status=PASS` 출력 실재 | `<RII>` 최상위 정규 산출물 전건에서 해당 문자열 0건. 문자열은 `input/job.slurm`의 미실행 출력 서식에만 있고 수용 산출물에는 없음 | **부재** |
| 모델 완주 실물 | 구 조건의 런처 수용 PASS를 대신할 수 없음 | `model.rc=0`, SUMMARY 2건 `complete=1`, R7·PHYSICS-COMPARISON iter1 커밋, RUN_FOOTER 스키마는 실재 | **사실이나 자격 효력 없음** |

감리-A의 두 차단 사유를 재적용한다. 첫째, 축자 `OLD:`/`NEW:` 쌍이 없는 개정 2는 상위 규약상 무효다. 둘째, 그 개정은 R-II 실패와 선행 「제출 불가」 확인 뒤 FAILED R-II가 통과하도록 구성된 결과 지향 골대 이동이다. user의 철회로 구 조건이 명시적으로 복귀했으며, 위 실측처럼 R-II는 FAILED로 종단했고 수용 PASS 실물은 없다. 따라서 무효 조항으로 PASS를 만들었던 개정 전 판정의 (d)는 오류이며 **제출 자격 이탈**로 정정한다.

철회된 §5.5-1의 사실 하위조건은 **8개가 아니라 9개**다: (i) 1개 + (ii)-1~5의 5개 + (iii)-1~3의 3개다. 개정 전 판정과 감리-B가 확인한 대로 이 9개 원자는 모두 사실이지만, 무효 조항의 사실 연언은 유효한 제출 자격을 만들지 못한다. §5.5-1의 NC-R14/R15 제출 전 기재 의무 역시 철회된 조항의 역사 기록이며, 현 정본 §10의 사후 집행 기록은 구 조건 미충족을 치유하지 않는다.

구 조건이 판별력 0인 판독기 rc를 필요조건으로 삼는다는 논거는 그대로 남는다. 그 논거의 존부와 이번 제출의 자격 이탈은 양립하며, 이 판정은 그 시정을 다루지 않는다.

## 4. 반복 0→1 응답 관측

운전석 판독문 §3은 다음 한정 아래 과대·과소 없이 맞다.

- 두 런 모두 FIXED_T이고 shell T_e는 10,020 K다. R-III의 전역 `LINE-ION-OWNER-SUMMARY` 8행과 R-IV의 8행은 전행 byte-동일이다.
- 전역 잣대에서 n_e는 shell 0~3 각각 `+0.195733383%`, `+0.195798347%`, `+0.195840475%`, `+0.195868608%`; signed rate는 `-8.829072646%`, `-9.334691662%`, `-9.898337163%`, `-10.543737486%`다. `signed_grouping_delta` 최대는 `2.03396327558280632e-16`이다.
- 소스의 owner 진단은 `LUMINA_A210_LINE_ION_OWNER_SHELLS`로 셸 범위를 받고 모든 ion slot을 합산한다(`src/lumina_plasma.c:13705-13844`). `LUMINA_A210_LINE_SATURATION_TARGET_ION`을 읽는 별도 saturation 경로(`:13957`)와 무관하다. 즉 이 전역 잣대는 `TARGET_ION`을 보지 않는다.
- 그러므로 두 팔의 동일한 전역 기울기는 측정 하나를 두 로그가 반복 기록한 것이고, R-IV는 “이온을 바꿔도 유지되는가”의 독립 확증이 아니다. 운전석의 **대조축 설계 오류** 기재는 정확하며 장부 후보다.
- 이온을 보는 SUMMARY에서 ion2는 `candidate_rows 624860→624860`, `selected_rows 2955→2420`(`-18.104906937%`), `total_scaled_emission 53.2279335647573468536→52.8790664054608953656`(원문 값, 약 `-0.655421%`), hazard/sprobe `182244→182244`다. ion3은 `candidate_rows 211887→211887`, `selected_rows 297→297`, emission `1.44173796616343670209e-07→1.43941912293423455153e-07`(약 `-0.160837%`), hazard/sprobe `0→0`다.
- ion3의 Z별 selected `156+20+121=297`은 이 런 내부 합으로만 확인했다. 이것이 **DET-A207-WIRING의 297과 동일 집합인지는 미확인**이며 사실로 승격하지 않는다.

## 5. 분기 확정과 근거

선행 R-II의 ZI-A는 폐합된 그대로 두고 이번 두 런에 §8-2를 적용한다.

| 등록 분기 | 판정식 대조 | 결론 |
|---|---|---|
| ZI-X | 어느 런이든 ISOLATED ≥1 | 두 런 모두 0, 불발 |
| ZI-D | 어느 런이든 미완주 | 두 job COMPLETED 0:0, model.rc=0, 완주 마커 전건, 불발 |
| ZI-P의 R-III 항 | S2 FAIL/UNDECIDABLE | 등록 S2 PASS, 불발 |
| ZI-P의 R-IV 항 | 값 파일 byte 불일치 | 존재하는 등록 3쌍과 실제 대응 6쌍에서 불일치 0. 등록 6쌍은 대칭 부재이므로 불일치가 아니며 불발 |
| ZI-A/ZI-B | R-II f_bit·E-I4 분할 | 선행 폐합 판정 영역, 재판정하지 않음 |

실측 `G6 선언 경로 6건 양쪽 부재`는 §8-2 분할에 없다. 이름 있는 분할 밖 라벨 **ZI-O-G6-DECLARED-PATH-ABSENT**를 부여하고 정지·보고한다. G6와 E-I8은 미착지로 남는다. 제출 자격 이탈은 §8-2의 어느 판정식에도 포함되지 않으므로 분기 판별량과 섞지 않는다. R-III·R-IV 완주, S2 PASS, 값 파일 불일치 0을 계약 원문에 재적용해도 이 라벨은 바뀌지 않는다.

운전석 결함·장부 귀속은 세 건이다: ① 1차 발주서의 정본/폐기본 역지정(1차 판정자 귀책 아님), ② TARGET_ION을 보지 않는 전역 잣대를 이온 대조축으로 삼은 설계 오류, ③ 무효 개정 위에서 이루어진 R-III·R-IV 제출의 자격 이탈. 철회된 §5.5-1의 NC-R14/R15 제출 전 기재 누락과 사후 집행 경위는 현 정본 §10의 역사 기록으로만 지목한다. 발견만 기재하며 수리안을 제시하지 않는다.

## 6. 하지 않은 것 — 계약 §7 경계 준수

- 선행 폐합 판정과 R-II ZI-A를 재판정하지 않았다.
- R-III Stage-4의 S_probe·Jbar를 본선 A2-07 물리 계약 측정으로 해석하지 않았다.
- S2b·d-proxy·Jbar 구동자 분리 분석이나 선행 Z2 제3원·Z3·Z4 자체 재집행을 하지 않았다.
- 실제 `physics_comparison` 경로를 G6 등록 경로로 대체하지 않았다.
- 계약·비교기·스테이저·런루트·봉인·기존 게이트 산출물을 고치지 않았다.
- 철회된 §5.5-1의 9개 사실 원자를 구 조건 대신 적용하지 않았다.
- G6 경로 결함, S2b 실패, 제출 자격 이탈, 판별력 0인 판독기 rc 문제에 대한 수리안을 제시하지 않았다.
- DET-A207-WIRING의 297과 R-IV의 297을 동일 집합으로 확인하거나 승격하지 않았다.
- 폐합을 선언하지 않는다. 폐합 정본은 장부이고 감리 두 겹이 남아 있다.

## 부록 — 판정자 실행 명령 원장

아래 명령은 저장소 `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn`에서 실행했다. Python 재실행은 전부 grammar-debug에서 `python3 -B`로 수행했고 출력은 `verdict_scratch2/`만 사용했다.

```bash
# 공통 약칭
RII=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1
RIII=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/riii_20260827T233659Z_ion2
RIV=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/riv_20260827T233659Z_ion3
L6=/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe

# A-1 정본 판별
stat -c '%n | size=%s | mtime=%y | inode=%i' \
  docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md{,.rev1}
sha256sum docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md{,.rev1}
git ls-files --stage docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md{,.rev1}
git log --follow --date=iso-strict --format='%H %ad %an %s' -- \
  docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md
git show --find-copies ddf6701f -- \
  docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md \
  docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md.rev1
git diff --no-index --numstat \
  docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md.rev1 \
  docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && \
  python3 -B scripts/check_prereg_preflight.py \
  docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md --root .'"

# A-2 런 종단·완주 마커
sacct -j 324557 -X -n -P -o JobIDRaw,State,ExitCode
sacct -j 325913,325914 -X -n -P \
  -o JobIDRaw,JobName,Submit,Start,End,State,ExitCode
for r in "$RIII" "$RIV"; do
  cat "$r/model.rc"
  rg -c -F '[A2-10][LINE-SATURATION-SUMMARY]' "$r/stderr.log"
  rg -F '[A2-10][LINE-SATURATION-SUMMARY]' "$r/stderr.log" | rg -c 'complete=1'
  rg -c -F 'LINE-SATURATION-ISOLATED' "$r/stderr.log" || true
  rg -c -F 'LINE-SATURATION-BLOCKED' "$r/stderr.log" || true
  rg -c -F 'event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1' "$r/stderr.log"
  rg -c -F '[PHYSICS-COMPARISON] lane=DET iter=1 status=COMMITTED' "$r/stderr.log"
  head -1 "$r/RUN_FOOTER.txt"
done

# A-3 G5 행·hazard·준위쌍·E-I7 독립 계수
awk 'index($0,"[A2-10][LINE-SATURATION-ROW]"){n++;a=index($0,"producer_continuum_term=")>0;\
b=index($0,"producer_local_emission_term=")>0;c=index($0,"producer_terms_defined=")>0;\
if(a&&b&&c)all++;else missing++}END{print n,all,missing+0}' "$RIII/stderr.log"
awk 'index($0,"[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]"){n++;z=ln=lo=up="";\
for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="Z")z=a[2];else if(a[1]=="line")ln=a[2];\
else if(a[1]=="lower_global")lo=a[2];else if(a[1]=="upper_global")up=a[2]}\
byz[z]++;c[z SUBSEP ln SUBSEP lo SUBSEP up]++;if(ln=="262210")hit++}\
END{for(k in c){u++;if(c[k]==1)o++;else if(c[k]==2)t++;else x++}\
print n,byz[26],byz[27],byz[28],u,o+0,t+0,x+0,hit+0}' "$RIII/stderr.log"
rg -n 'LINE-SATURATION-ZERO-OPACITY-ROW.*line=262210([[:space:]]|$)' "$RIII/stderr.log"

# A-4 G5·NC-G5 재실행 — 두 JSON은 기존 산출과 cmp rc=0
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && \
  python3 -B scripts/a210_sproducer_ledger_check.py --te-k 10020 \
  --output /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/g5_repro.json \
  /gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/riii_20260827T233659Z_ion2/stderr.log'"
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && \
  python3 -B scripts/a210_sproducer_ledger_check.py --te-k 10020 \
  --output /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/nc_g5_repro.json \
  /gpfs/kjhan/lumina/z2a_gates/nc_g5_injected.log'"
cmp /gpfs/kjhan/lumina/z2a_gates/g5_s2_riii.json \
  /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/g5_repro.json
cmp /gpfs/kjhan/lumina/z2a_gates/nc_g5.json \
  /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/nc_g5_repro.json
cmp -l "$RIII/stderr.log" /gpfs/kjhan/lumina/z2a_gates/nc_g5_injected.log
rg -n 'S2b|out\[.verdict.\]' scripts/a210_sproducer_ledger_check.py

# A-5 G6 등록 9경로 재실행(rc=2) — 기존 산출과 cmp rc=0
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && \
  python3 -B scripts/byte_parity_compare.py \
  --value-file work/lumina_spectrum.csv \
  --value-file work/lumina_spectrum_comoving.csv \
  --value-file work/lumina_plasma_state.csv \
  --value-file work/physics_DET_iter0000/manifest.json \
  --value-file work/physics_DET_iter0000/shell.csv \
  --value-file work/physics_DET_iter0000/spectral.csv \
  --value-file work/physics_DET_iter0001/manifest.json \
  --value-file work/physics_DET_iter0001/shell.csv \
  --value-file work/physics_DET_iter0001/spectral.csv \
  --output /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/g6_repro.json \
  /gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe \
  /gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/riv_20260827T233659Z_ion3'"
cmp /gpfs/kjhan/lumina/z2a_gates/g6_byte.json \
  /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/g6_repro.json

# A-6 NC-G6 재실행
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && \
  python3 -B scripts/byte_parity_compare.py \
  --value-file work/lumina_spectrum.csv \
  --value-file work/lumina_spectrum_comoving.csv \
  --value-file work/lumina_plasma_state.csv \
  --output /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/nc_g6_neg_repro.json \
  /gpfs/kjhan/lumina/z2a_gates/nc_g6/left \
  /gpfs/kjhan/lumina/z2a_gates/nc_g6/right'"
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && \
  python3 -B scripts/byte_parity_compare.py \
  --value-file work/lumina_spectrum.csv \
  --value-file work/lumina_spectrum_comoving.csv \
  --value-file work/lumina_plasma_state.csv \
  --output /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/nc_g6_control_repro.json \
  /gpfs/kjhan/lumina/z2a_gates/nc_g6/left \
  /gpfs/kjhan/lumina/z2a_gates/nc_g6/left'"
cmp /gpfs/kjhan/lumina/z2a_gates/nc_g6/neg.json \
  /gpfs/kjhan/lumina/z2a_gates/verdict_scratch2/nc_g6_neg_repro.json
cmp -l /gpfs/kjhan/lumina/z2a_gates/nc_g6/{left,right}/work/lumina_plasma_state.csv
sha256sum /gpfs/kjhan/lumina/z2a_gates/nc_g6/left/work/*.csv
jq -r '.value.files[]|[.path,.left_sha256,.right_sha256,.status]|@tsv' \
  /gpfs/kjhan/lumina/z2a_gates/nc_g6/pos.json

# A-7 실제 방출 대응 경로 9쌍과 선언 경로 부재
for rel in work/lumina_spectrum.csv work/lumina_spectrum_comoving.csv \
  work/lumina_plasma_state.csv \
  work/physics_comparison/physics_DET_iter0000.manifest.json \
  work/physics_comparison/physics_DET_iter0000.shell.csv \
  work/physics_comparison/physics_DET_iter0000.spectral.csv \
  work/physics_comparison/physics_DET_iter0001.manifest.json \
  work/physics_comparison/physics_DET_iter0001.shell.csv \
  work/physics_comparison/physics_DET_iter0001.spectral.csv; do
  cmp -s "$L6/$rel" "$RIV/$rel"
done
find "$L6/work" "$RIV/work" -maxdepth 3 -type f -printf '%p\n' | sort

# A-8 봉인 L6 계보
rg -n -F 'sprim_l6_20260821T054111Z_probe' \
  docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md \
  docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md \
  scripts/stage_a210_z2a_probe.sh
cat "$L6/L6_STAGE_SEAL.txt" "$L6/L6_PROBE_VERDICT.txt" "$RIV/Z2A_STAGE_SEAL.txt"

# A-9 구 조건 실측 + 철회된 §5.5-1의 역사적 하위조건 9개
sacct -j 324557 -X -n -P -o JobIDRaw,State,ExitCode
rg -n -F 'DET_FLIGHT_FATAL Z2A R-II analyzer failed' slurm-324557.out
rg -n -F 'A210_Z2A_PROBE_ANALYZER_RESULT status=FAIL' "$RII/z2a_probe.stdout"
find "$RII" -maxdepth 1 -type f -print0 | \
  xargs -0 rg -l -F 'A210_Z2A_PROBE_ACCEPT status=PASS'
wc -l "$RII/model.rc"; cat "$RII/model.rc"
rg -F '[A2-10][LINE-SATURATION-SUMMARY]' "$RII/stderr.log" | wc -l
rg -F '[A2-10][LINE-SATURATION-SUMMARY]' "$RII/stderr.log" | rg 'complete=1' | wc -l
rg -n -F 'event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1' "$RII/stderr.log"
rg -n -F '[PHYSICS-COMPARISON] lane=DET iter=1 status=COMMITTED' "$RII/stderr.log"
head -1 "$RII/RUN_FOOTER.txt"
nl -ba docs/VERDICT_A210_Z2A_DIAG_ISOLATION_2026-08-27.md | sed -n '377,386p'
nl -ba docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md | sed -n '499,518p'
git cat-file -t 4fc84fc
git show -s --format='%H%n%aI%n%cI%n%s' 4fc84fc ddf6701f

# A-10 반복 응답과 TARGET_ION 비소비
cmp <(rg -F '[A2-10][LINE-ION-OWNER-SUMMARY]' "$RIII/stderr.log") \
    <(rg -F '[A2-10][LINE-ION-OWNER-SUMMARY]' "$RIV/stderr.log")
sed -n '13690,13845p' src/lumina_plasma.c
rg -n 'LINE-ION-OWNER-SUMMARY|LINE_SATURATION_TARGET_ION' src/lumina_plasma.c
rg -F '[A2-10][LINE-ION-OWNER-SUMMARY]' "$RIII/stderr.log" | awk \
  '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="shell")s=a[2];\
  if(a[1]=="n_e_cm3")ne[s,++k[s]]=a[2]+0;if(a[1]=="line_order_signed_rate")sr[s,k[s]]=a[2]+0;\
  if(a[1]=="signed_grouping_delta"&&(a[2]+0)>mx)mx=a[2]+0}}\
  END{for(s=0;s<4;s++)printf "%d %.9f %.9f\n",s,100*(ne[s,2]/ne[s,1]-1),\
  100*(sr[s,2]/sr[s,1]-1);printf "max %.18g\n",mx}'
rg -F '[A2-10][LINE-SATURATION-UNION-ION-SUMMARY]' "$RIV/stderr.log"
```

분기 = ZI-O-G6-DECLARED-PATH-ABSENT
