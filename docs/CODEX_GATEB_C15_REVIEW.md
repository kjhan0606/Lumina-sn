종합 재판정은 **FAIL**입니다. 기존 FAIL 중 ① 단위 근거와 ② s43 속도 대응은 해소됐지만, ③ §3 실질 커버리지와 ④ 실제 생산값 재현성은 아직 FAIL입니다.

| 검토축 | 판정 | 핵심 근거 |
|---|---|---|
| ① n_e cm⁻³ 실증 | **PASS** | RVTJ writer가 `Electron density` 직후 `ED`를 쓰고, `ED` 선언이 `#/cm^3`임 |
| ② s43 대응 | **PASS** | s43 중심 35,568 km/s가 CMFGEN 범위 안이며 최근접 depth 차이는 −70.29 km/s |
| ③ 명세 §3 커버리지 | **FAIL** | 행·결손 사유는 완비됐지만 수치 비교 대상 대부분이 여전히 unavailable |
| ④ 동결 생산값 재현 | **FAIL** | C1/C2 iteration 정렬 불일치, J-bar 6개 이온 결손, thermal 등록 입력 미복원 |
| 관측 전용성 | **PASS** | 프로브는 생산 로컬을 읽어 `g_oracle`에만 기록 |
| 게이트 OFF 불변 | **PASS(정적)** | 매크로는 전용 bench에만 정의. 실행 금지로 4종 CSV 재검증은 미실시 |

### ① n_e 단위 — PASS

RVTJ 헤더 자체에는 여전히 단위가 없지만, 소스 연결이 명백해졌습니다.

- RVTJ writer: `Electron density` 다음 행에 동일 배열 `ED` 기록 — [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4421), [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4423)
- `ED` 선언 단위: `Electron density (#/cm^3)` — [mod_cmfgen.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/mod_cmfgen.f:211)
- 비교자는 writer/header/value/unit 선언을 모두 요구하고, n_e를 identity `#/cm^3`로 기록합니다 — [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:261), [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:447)

단, 저장된 `cmfgen_source_evidence.csv:2-4`의 물리 행 번호는 현재 소스 행 번호와 다릅니다. 소스 텍스트는 일치하지만 산출물 provenance는 갱신 전 상태입니다.

### ② s43 대응 — PASS

- s43 경계는 35,204–35,932 km/s이므로 중심은 35,568 km/s입니다. 다음 s44 중심은 36,296 km/s입니다 — [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:45), [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:46)
- CMFGEN 최대 속도는 35,975.288 km/s이므로 s43은 범위 안, s44는 범위 밖입니다 — [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:27)
- s43→depth 10은 35,497.710 km/s, Δv=−70.290 km/s입니다 — [shell_cmfgen_depth_map.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_5/shell_cmfgen_depth_map.csv:4)
- 비교자도 범위 밖 셸을 즉시 거부합니다 — [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:311)

다만 s43에서 RVTJ와 PRRR n_e가 약 0.188% 다르며 `different_snapshot_or_output_generation`으로 기록됩니다 — [cmfgen_snapshot_consistency.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_5/cmfgen_snapshot_consistency.csv:18). 속도 대응은 타당하지만 비교자는 이 snapshot 불일치를 공개만 하고 중단하지 않습니다.

### ③ §3 커버리지 — FAIL

형식적 행 커버리지는 개선됐습니다. 산출 불가 항목도 삭제하지 않고 사유 행을 냅니다. 그러나 저장 산출물 census는 `compared 79`, `context-only 9`, `Lumina unavailable 241`, `CMFGEN unavailable 217`입니다 — [oracle_vs_cmfgen.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_5/oracle_vs_cmfgen.md:15).

잔여 결손:

- **bf:** Γ·α가 실제 산출되는 것은 Si II·S II·Fe II뿐입니다. Si III·S III·Fe III/IV·Co III는 “assembled pair의 lower member가 아님”으로 unavailable입니다 — [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:142), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:157). α spont/stim은 CMFGEN 분리 앵커가 없고, chi/eta도 동일 단색 앵커가 없습니다 — [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:370), [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:397).
- **ff:** chi_ff·eta_ff는 CMFGEN 동일량 비교가 없고, 냉각률도 Lumina 방출 적분과 GENCOOL 순냉각의 비동일 `context_only`입니다 — [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:377).
- **bb:** 기준 dump 필터가 Si II/III뿐입니다 — [stdout.log](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:117). 따라서 S II/III·Fe II/III/IV·Co III의 대표 J·펌핑률은 unavailable입니다. s0은 Si도 양수 lower-population flow가 없어 전부 unavailable입니다 — [lumina_oracle_cell_s0.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_5/lumina_oracle_cell_s0.csv:50).
- **collisional:** s8/s43의 Si 두 이온만 Lumina 수치가 있고 나머지 여섯 이온은 결손입니다. Si 수치도 GENCOOL에는 일치하는 전이별 C_ul/C_lu가 없어 비교되지 않습니다 — [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:399).
- **thermal:** 순 bf, aggregate collisional, thermal net만 동일량 비교입니다. `MA_LINE_DESTRUCT`는 모든 셸에서 unavailable이며, 명세의 핵심 관찰 목표가 여전히 열리지 않습니다 — [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:284). deposition·photoion heating·adiabatic도 동일 CMFGEN 행이 없습니다 — [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:389).
- **상태:** n_e와 n_ion은 비교되지만 ion fraction은 CMFGEN 전 이온단계 census가 없어 unavailable입니다. b_k는 Lumina `levelN`을 CMFGEN OUT의 N번째 계수에 직접 대응시키며 별도의 준위 항등성 증명이 없습니다 — [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:354), [oracle_compare_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:358).

### ④ 동결 셀 생산값 재현성 — FAIL

원래의 “bf estimator 전부 0” 문제는 해소됐습니다. 1,000개 bin을 읽고 전 bin이 존재해야만 진행하며 — [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:251), [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:390) — 생산 소비 분기도 실제 양수 estimator를 사용합니다 — [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15033).

그러나 전체 재현은 실패합니다.

- C1, C2, J-bar loader가 각각 독립적으로 “최대 iter”를 선택하며 공통 iteration 일치 검사가 없습니다 — [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:189), [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:227), [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:269).
- 생산 경로에서 rate solve는 iteration 초반에 이전 C1/C2 필드를 소비하고 — [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7153), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7233) — 현재 iteration C1/C2는 transport 뒤 다음 iteration용으로 발행됩니다 — [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7925). 따라서 최종 C1/C2 block과 최종 population/J-bar block은 소비 시점이 한 iteration 어긋납니다.
- J-bar는 Si II/III만 동결됐으므로 나머지 여섯 이온의 생산 bb/collisional 값은 재현되지 않습니다.
- 생산 thermal 호출 전에 설정되는 tail color, tri-response, line-response 및 lagged photoion MC field를 하니스가 복원하지 않습니다 — [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7147), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7214), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7989). 하니스는 곧바로 thermal 생산 함수를 호출합니다 — [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:427).
- 저장된 2회 hash 일치는 하니스의 자기결정론만 증명합니다 — [smoke_sha256.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_5/smoke_sha256.txt:1). 생산 실행값과의 동일성을 증명하지는 않습니다.

### 관측 전용성·OFF

관측 프로브는 계산된 `chi_contrib`, `J_line`, rate locals를 읽어 `g_oracle`에만 누적합니다 — [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6831), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14483), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15098). Thermal도 로컬 `ne_probe`로 잔차만 평가하고 commit 전에 중단합니다 — [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:9868). **관측 전용성 PASS**입니다.

`LUMINA_FROZEN_ORACLE`은 전용 bench target에서만 정의되고 정상 CPU/GPU target에는 없습니다 — [Makefile](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/Makefile:34), [Makefile](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/Makefile:44), [Makefile](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/Makefile:64). **OFF 불변은 정적 PASS**입니다. 다만 요청상 실행하지 않았으므로 표준 4종 CSV의 동적 byte-identical 재인증은 이번 판정에 포함하지 않았습니다.