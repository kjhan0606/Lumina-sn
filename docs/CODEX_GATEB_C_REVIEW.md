종합 판정은 **FAIL**입니다. 관측 프로브와 게이트 OFF 구조는 PASS지만, 동결 입력 재현·s45 대응·§3 커버리지가 명세를 충족하지 못합니다.

| 검토축 | 판정 | 핵심 |
|---|---|---|
| 1. 관측 전용성 | PASS | 프로브 자체는 생산 변수·행렬·인덱스를 변경하지 않음 |
| 2. 게이트 OFF 코드 불변 | PASS(정적) | 모든 프로브가 컴파일 타임 제거됨 |
| 3. CMFGEN 파서 | FAIL | T·속도는 정당하나 n_e의 cm⁻³ 단위 근거가 헤더에 없음 |
| 4. 셸↔depth 대응 | FAIL | s0/s8은 타당, s45는 CMFGEN 범위 밖인데 비교를 계속함 |
| 5. §3 수량표 커버리지 | FAIL | thermal 전부, 대부분의 bf rate와 상태·GENCOOL 비교가 결손 |

### 1. 관측 전용성 — PASS

- 신규 상태는 `g_oracle`에만 축적되며 생산 배열에는 쓰지 않습니다. bf/ff 프로브도 이미 계산된 `chi_contrib`, `S_l`, `R_bf`, `R_rec` 등을 읽습니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6697), [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6818), [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14964).
- bb/collision 프로브는 `total_up/down`이 형성된 뒤, 실제 행렬 갱신 전에 같은 로컬 값을 복사할 뿐입니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14355), [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14358), [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14393).
- 수식·분기 조건·생산 인덱스 변경은 발견하지 못했습니다.

단, `cooling_ff_grid`는 생산 thermal ledger 값을 관측한 것이 아니라 프로브가 `4πηΔν`를 새로 적분한 값입니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6822). 물리를 변경하지는 않지만, 명세 §1-4의 “생산 함수가 실제 산출한 냉각률”로 인정하기에는 약합니다.

### 2. 게이트 OFF byte 불변성 — PASS(코드 수준)

- observer 정의 전체와 모든 call-site가 `#ifdef LUMINA_FROZEN_ORACLE` 안에 있습니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:21), [src/lumina.h](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:926).
- 일반 CPU/GPU 타깃에는 매크로가 없고 전용 bench에만 정의됩니다: [Makefile](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/Makefile:34), [Makefile](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/Makefile:44), [Makefile](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/Makefile:64).
- 따라서 전처리 후 기존 계산 경로의 표현식·분기·메모리 쓰기는 그대로입니다.

남은 위험: 빌드·실행 금지 조건 때문에 실제 4종 CSV의 byte 비교는 인증하지 않았습니다. 판정은 정적 코드 수준에 한정됩니다.

### 중대 결함: “동결된 실제 생산 입력”이 재현되지 않음

1. 채택 설정에서 photoionization은 양수인 `bf_rate_estimator`를 우선 소비합니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14908). 그러나 하니스는 이를 전부 0으로 지웁니다: [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:324). 결과적으로 실제 생산 실행과 달리 모든 bin이 `J_bin` 적분 fallback으로 갑니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14912).

2. 기준 입력의 J-bar dump는 Si II/III만 기록하도록 설정돼 있습니다: [stdout.log](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:117). 하니스는 새 배열을 0으로 초기화한 뒤 기록된 행만 채웁니다: [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:225). 따라서 S/Fe/Co 선은 실제 동결 J-bar가 아니라 binned-J fallback을 사용합니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14148).

이는 생산 수식을 바꾼 것은 아니지만, oracle이 “그 동결 셀에서 생산 경로가 실제로 냈던 값”을 재현한다는 명세 목적에는 FAIL입니다.

### 3. CMFGEN 파서 — FAIL

PASS인 부분:

- RVTJ의 ND=90과 블록 열 순서는 실제 파일과 일치합니다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:4), [cmp_rvtj_T_ne_vs_published.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/cmp_rvtj_T_ne_vs_published.py:40).
- 속도는 헤더에 `km/s`가 명시돼 있고, 온도는 `10^4K`가 명시돼 있으므로 ×10⁴ 변환은 정당합니다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:26), [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:65), [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:217).
- PRRR의 `Depth index → Ion Density → <ion> Photoionization Rates` 구조는 실물과 일치합니다: [CoIIIPRRR](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CoIIIPRRR:1), [CoIIIPRRR](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CoIIIPRRR:13), [CoIIIPRRR](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CoIIIPRRR:16).
- Γ의 `sum(PR)/Ion Density`는 기존 source-certified parser의 정의와 일치합니다: [gamma_coiii_alllevel.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/gamma_coiii_alllevel/gamma_coiii_alllevel.py:136).
- α와 GENCOOL free-free 단위를 추측하지 않고 비교를 보류한 것은 올바릅니다: [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:225), [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:239).

FAIL인 부분:

- RVTJ 실물은 `Electron density`만 쓰고 단위를 명시하지 않습니다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:52). 그런데 비교자는 별도 source 근거 없이 identity `cm^-3`를 선언합니다: [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:305). 기존 reader도 값만 읽을 뿐 단위를 인증하지 않습니다: [cmp_rvtj_T_ne_vs_published.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/cmp_rvtj_T_ne_vs_published.py:47). 이는 엄격한 “추측 금지” 기준에서 근거 결손입니다.
- 새 PRRR parser는 검증된 parser처럼 super-level 수를 받아 정확히 N행을 읽지 않고, 다음 비수치 행까지 무제한 합산합니다: [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:97). 현재 실물은 빈 줄로 종료돼 맞지만 형식 변화에 fail-closed하지 않습니다.
- RVTJ와 PRRR가 같은 snapshot인지 검사하지 않습니다. 실제 depth 1의 n_e는 RVTJ `2.0043885E+05`, CoIIIPRRR `2.0153E+05`로 이미 다릅니다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:53), [CoIIIPRRR](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CoIIIPRRR:10).

### 4. 셸↔depth 속도 대응 — FAIL

- 셸 중심속도 `(v_inner+v_outer)/2`를 사용하고 RVTJ의 최근접 depth를 고르는 방식 자체는 타당합니다: [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:54), [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:175).
- s0=4264 km/s, s8=10088 km/s는 RVTJ 범위 안이며 근접점이 존재합니다: [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:2), [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:10).
- s45 중심은 37024 km/s인데 CMFGEN 최대는 약 35975 km/s입니다: [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:47), [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:27).

스크립트는 `in_cmfgen_range=False`를 기록하면서도 최근접 외곽 depth를 모든 수량의 정상 `compared` 행에 사용합니다: [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:183), [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:217). s45는 약 1049 km/s 떨어진 다른 셀로, “같은 셸 값” 비교로는 인정하기 어렵습니다.

### 5. 명세 §3 커버리지 — FAIL

- **bf:** 하니스가 조립하는 pair는 Si II/III, S II/III, Fe II/III, Co II/III뿐입니다: [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:352). Γ/α observer는 lower member만 `seen` 처리하므로 실제 산출되는 대상은 Si II·S II·Fe II뿐입니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:13844). Si III·S III·Fe III/IV·Co III Γ/α는 결손입니다.
- **bb/collisional:** Fe IV를 포함한 pair가 호출되지 않아 대표선·펌핑·C_ul/C_lu가 없습니다. 또한 대표선이 없을 때 `jbar_input_raw`와 `sobolev_beta` unavailable 행조차 쓰지 않아 조용히 누락됩니다: [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:154), [src/lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:181).
- **thermal:** 전 항목이 의도적으로 unavailable입니다: [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:303). 명세 핵심인 s0 가열 결손과 `MA_LINE_DESTRUCT` 위치를 관찰할 수 없습니다.
- **상태:** parser가 이미 PRRR `Ion Density`를 읽지만 n_ion/ion fraction 비교에는 사용하지 않습니다: [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:90), [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:250). b_k도 모두 미인증입니다.
- **GENCOOL:** 실물에는 bound-free cooling 단위와 collisional/free-free 블록이 존재하지만 비교자는 아무 항목도 파싱하지 않습니다: [GENCOOL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/GENCOOL:17), [GENCOOL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/GENCOOL:88), [GENCOOL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/GENCOOL:91).
- 결과적으로 수치 비교가 구현된 CMFGEN 앵커는 사실상 `T_e`, `n_e`, Γ 일부, Jν뿐입니다: [scripts/oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:217).

최종적으로, 이 구현은 안전한 observer scaffolding으로는 합격이지만 Gate B Phase-1 acceptance oracle로는 불합격입니다. 가장 큰 잔여 위험은 “미비교 행을 남겼다”는 형식적 완결성이 실제 동결 입력 재현과 CMFGEN 수량표 커버리지 결손을 가리고 있다는 점입니다.