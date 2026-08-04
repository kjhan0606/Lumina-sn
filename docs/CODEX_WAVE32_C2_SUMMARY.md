# Codex C2 독립 소스 리뷰 결과

검토 범위는 A2 구현 소스와 관련 테스트로 한정했다. `CODEX_WAVE32_B2_*`는 열람하지 않았고, 파일 수정·빌드·모델/GPU 실행도 하지 않았다.

## 최종 판정

| 항목 | 판정 | 핵심 사유 |
|---|---|---|
| A2-1 EWPrivateView | **[FAIL]** | production `NLTEConfig` 격리는 대체로 성립하지만, private/commit 투영 코드가 중복되어 동등성이 구조적으로 보장되지 않고 중첩 OOM 경로도 닫히지 않음 |
| A2-2 commit pass-through | **[FAIL]** | boundary FAIL은 실제 commit을 막지만 frozen harness가 반환값을 버려 실패 실행도 프로세스 성공으로 끝날 수 있음 |
| A2-3 공유 bf 조건 | **[FAIL]** | 일반 pair/EW는 helper를 쓰지만 top-stage III→IV가 조건을 재복제하며 JEQB/C2를 우회; GPU 우회 기록도 frozen 전용 |
| A2-5 D3 5좌표 | **[FAIL]** | grid/cap/가중/coll-bf 4좌표는 정책대로이나 GPU/field 좌표가 미정렬 |
| A2-7 R7 writer | **[FAIL]** | binary schema는 대부분 정확하지만 η bitwise 감사가 실제 계산 없이 하드코딩됨 |
| 신규 clamp/floor/cap | **[PASS]** | A2 귀속 경로에서 신규 수치 clamp/floor/cap 발견 없음 |
| D6형 항등식 재발 | **[FAIL]** | D6 행렬 ledger는 개선됐지만 η sidecar 및 write-read-write 검사가 항등형이고 seeded-defect 음성대조가 없음 |

---

## 1. A2-1 EWPrivateView — **[FAIL]**

### 1.1 production `NLTEConfig` 무쓰기 경계 — **[PASS]**

Shadow 진입 시 production 포인터 대신 private config로 지역 포인터가 바뀐다([lumina_element_wide.c:1392](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1392>), [lumina_element_wide.c:1402](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1402>)).

Private view는 shallow copy 후 다음 layout-derived 상태를 모두 새 저장소로 교체한다.

- `n_nlte_ions`, Z/stage, level/super offset 및 `super_mode`: [lumina_element_wide.c:564](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:564>)-588
- `nlte_to_global_level`, `global_to_nlte_level`, `nlte_line_map`: [lumina_element_wide.c:589](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:589>)-593
- `nlte_level_populations`: [lumina_element_wide.c:593](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:593>)
- `fl_to_super`, `super_anchor_global`, `within_sl_frac`: [lumina_element_wide.c:595](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:595>)-600
- 모든 맵·anchor·population seed·fraction 계산도 private 포인터를 통해 수행: [lumina_element_wide.c:609](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:609>)-646

공유된 `J_nu`, `bf_rate_estimator`, `drainless_metastable`, `shell_tau` 등은 후보 assembler에서 읽기만 한다. Capture assembler는 물리 채널 조립 직후 반환하므로 이후 time-dependent/pin/floor/anchor/population 쓰기 구간에 도달하지 않는다([lumina_plasma.c:16197](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16197>)-16203).

Hot/cold 감사 중 population seed를 쓰는 코드도 `nlte` 지역 포인터가 private config로 이미 교체된 뒤다([lumina_element_wide.c:1487](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1487>)-1524). 따라서 `commit_requested=0`에서 production population buffer에 쓰는 경로는 발견하지 못했다.

### 1.2 해제 및 오류 경로 — **[FAIL]**

명시적 경로 자체는 잘 닫혀 있다.

- 부분 할당 실패 시 active를 세운 뒤 일괄 free: [lumina_element_wide.c:601](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:601>)-606
- missing slot/N≤0 조기 반환 시 free: [lumina_element_wide.c:1407](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1407>)-1417
- 정상/`cleanup_fail` 양쪽 free: [lumina_element_wide.c:1666](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1666>)-1679
- free 대상 일곱 포인터가 할당 대상과 일치: [lumina_element_wide.c:549](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:549>)-558

하지만 private 초기화가 호출하는 `nlte_precompute_within_sl_frac()`은 `Zsl`의 `malloc` 결과를 검사하지 않고 즉시 역참조한다([lumina_plasma.c:17033](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17033>)-17038). 이 중첩 OOM은 오류 반환이나 view 해제 없이 crash할 수 있으므로 “모든 에러 경로에서 해제”라는 강한 주장은 성립하지 않는다.

### 1.3 commit 레이아웃과 의미 동등성 — **[FAIL]**

현재 상수 배열 값은 서로 같다.

- Private 배열: [lumina_element_wide.c:535](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:535>)-542
- Commit 배열: [lumina_plasma.c:7681](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7681>)-7688

그러나 투영 코드는 공유되지 않는다. Private가 자체적으로 count/offset/map/super/anchor를 다시 구현한 반면([lumina_element_wide.c:571](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:571>)-646), commit 레인은 별도 `nlte_init()` 구현을 사용한다([lumina_plasma.c:14057](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14057>)-14164).

실제 차이 가능성도 있다. Private line map은 `atom->n_lines` 크기로 할당하지만([lumina_element_wide.c:592](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:592>)), commit 레인은 `opacity->n_lines`를 사용한다([lumina_plasma.c:14169](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14169>)-14172). 두 값의 동일성을 assert하지 않는다. 현재 데이터에서 같더라도 소스 불변식은 아니다.

---

## 2. A2-2 commit pass-through — **[FAIL]**

### 실제 commit 차단 — **[PASS]**

Gate는 실제 측정값에서 계산된다.

- topology/numerical/boundary gate 및 최종 `pass`: [lumina_element_wide.c:1606](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1606>)-1628
- `commit_performed = pass && commit_requested`: [lumina_element_wide.c:1629](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1629>)
- population write는 `if(commit_performed)` 내부에만 존재: [lumina_element_wide.c:1655](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1655>)-1664
- 실패 status는 pair baseline 실행을 막지 않음: [lumina_plasma.c:17181](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17181>)-17187
- candidate-only τ도 status가 정확히 1인 경우만 씀: [lumina_plasma.c:16968](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16968>)-16974

따라서 `boundary_gate_pass=0`인데 실제 commit이 수행되는 가짜 통과 경로는 없다.

`commit_blocked_by`도 세 gate의 실제 boolean을 우선순위로 분류한다([lumina_element_wide.c:1630](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1630>)-1633). 기록된 값 자체는 리터럴 성공값이 아니라 실측-derived 값이다.

### Pass-through 실패 전파 — **[FAIL]**

Frozen entry는 잘못된 env 또는 gate 실패에서 `-1`을 반환한다([lumina_element_wide.c:1695](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1695>)-1706). 그러나 실제 frozen harness는 두 호출 모두 `(void)`로 반환값을 폐기한다([bench_frozen_oracle.c:623](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:623>)-628). 따라서 다음이 가능하다.

- 잘못된 frozen commit 설정
- topology/numerical/boundary FAIL
- private-view OOM 반환

이들 모두 harness 전체 exit status에는 전달되지 않는다. 테스트가 diagnostics 내용을 별도로 확인하지 않으면 실패 EW 호출을 포함한 실행도 성공으로 취급할 수 있다.

기록 역시 필수 산출은 아니다. dump 파일 open 실패는 로그만 남기고 계속 진행한다([lumina_element_wide.c:1227](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1227>)-1233). 따라서 `blocked_by` 값은 정확하지만 “반드시 기록된다”는 보장은 없다.

성공적인 COMMIT=1 pilot/off-target 격리는 소스에 양성 fixture가 없어 **[UNRESOLVED]**다.

---

## 3. A2-3 공유 조건 단일화 — **[FAIL]**

일반 pair와 EW 경로는 공유 helper를 사용한다.

- 공유 field/GPU helper: [lumina_plasma.c:385](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:385>)-414
- 일반 pair 소비: [lumina_plasma.c:15646](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15646>)-15696
- EW 소비: [lumina_element_wide.c:347](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:347>)-416
- coll-bf 공유 helper 및 두 소비자: [lumina_plasma.c:417](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:417>)-420, [lumina_plasma.c:15814](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15814>), [lumina_element_wide.c:438](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:438>)

하지만 top-stage III→IV 소비자는 helper를 호출하지 않는다. `J_nu`를 직접 읽고 `(artis_parity_enabled() && bf_rate_estimator)` 조건을 다시 작성한다([lumina_plasma.c:15987](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15987>)-15999). 결과적으로:

- `LUMINA_C2_MATRIX_BF=1`은 이 경로에서 무시된다.
- `LUMINA_NLTE_BF_JEQB=1`도 무시된다.
- recombination의 stimulated `J`도 helper가 선택한 JEQB field와 달라질 수 있다.

GPU 우회 인지는 helper가 계산하지만([lumina_plasma.c:408](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:408>)-411), 카운터 증가는 `LUMINA_FROZEN_ORACLE` 및 `g_oracle.fp` 내부에만 있다([lumina_plasma.c:15650](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15650>)-15654). Production manifest/telemetry에는 GPU field bypass가 남지 않는다. 따라서 “숨겨지지 않고 기록”도 충족하지 못한다.

---

## 4. A2-5 D3 5좌표 — **[FAIL]**

| 좌표 | 판정 | 소스 판정 |
|---|---|---|
| Grid 검사 | **[PASS]** | EW는 grid 길이까지 일치해야 σ row를 쓰고 아니면 Kramers로 전환([lumina_element_wide.c:319](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:319>)-325). Pair도 동일 조건([lumina_plasma.c:15572](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15572>)-15573) |
| `1e30` cap 비복제 | **[PASS]** | Pair cap은 그대로 존재([lumina_plasma.c:15712](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15712>)-15729). EW는 `DBL_MAX` 초과를 거부·계수할 뿐 cap하지 않음([lumina_element_wide.c:427](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:427>)-433) |
| `within_sl_frac` 유지 | **[PASS]** | EW forward와 inverse 양쪽에 lower/upper fraction을 적용([lumina_element_wide.c:453](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:453>)-460). Pair inverse가 무가중인 차이는 실제 코드([lumina_plasma.c:15764](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15764>)-15776)와 manifest에 명시([lumina_element_wide.c:1651](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1651>)) |
| coll-bf 정렬 | **[PASS]** | 양쪽 모두 `nlte_bf_collisional_enabled()` 사용 |
| GPU/field 정렬 | **[FAIL]** | top-stage 경로가 helper를 우회하고, GPU bypass 기록이 frozen observer에만 존재 |

따라서 다섯 좌표 전체 acceptance는 FAIL이다.

---

## 5. A2-7 R7 writer — **[FAIL]**

### Binary v1 필드 대조 — **[PASS]**

다음은 설계 §5.3과 일치한다.

- magic, endian word, version, sizes, iteration/generation, flags/reserved, `t_exp`: [lumina_cmfgen.c:266](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:266>)-278
- native struct가 아닌 u32/u64/f64 little-endian 직렬화: [lumina_cmfgen.c:185](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:185>)-205
- `r_edge`, descending `nu`, positive `dnu`: [lumina_cmfgen.c:279](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:279>)-283
- `chi_total`, `chi_coherent`, `eta_fixed`, `eta_coherent`, `eta_total`, `J_producer` 순서: [lumina_cmfgen.c:284](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:284>)-303
- 모든 shell-major frequency array가 각자 동일한 descending 역순을 사용함
- 전체 binary의 SHA-256 sidecar: [lumina_cmfgen.c:310](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:310>)-338
- Python 검증기가 표준 `hashlib.sha256`로 독립 비교: [cmf_chieta_roundtrip_selftest.py:77](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmf_chieta_roundtrip_selftest.py:77>)-83

### η 분해 bitwise 감사 — **[FAIL]**

Writer는 `eta_total`을 `eta_fixed+eta_coherent`로 계산해 쓰지만([lumina_cmfgen.c:296](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:296>)-300), 실제 runtime 비교나 max-abs 누산은 전혀 없다. Sidecar에는 무조건 다음 상수를 쓴다.

- `"eta_decomposition_bitwise": true`
- `"eta_decomposition_max_abs": 0.0`

근거: [lumina_cmfgen.c:325](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:325>)-332.

오프라인 Python은 세 배열을 실제 비교하므로 그 시험 자체는 fail 가능하다([cmf_chieta_roundtrip_selftest.py:44](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmf_chieta_roundtrip_selftest.py:44>)-46). 그러나 runtime sidecar의 “감사 완료” 주장은 writer가 실측한 결과가 아니다.

### OFF 중립성 — **[PASS]**

환경변수가 없거나 빈 문자열이면 writer 호출, 파일 I/O, 상태 변경이 전혀 없다([lumina_cuda.cu:7544](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7544>)-7566). 호출 위치도 solve 및 optional damping 뒤다([lumina_cuda.cu:7514](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7514>)-7539).

### Fail-closed — **[FAIL]**

프로세스 제어는 대체로 fail-closed다.

- malformed/out-of-range iteration은 `EXIT_FAILURE`: [lumina_cuda.cu:7550](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7550>)-7559
- 선택 iteration의 writer 실패도 `EXIT_FAILURE`: [lumina_cuda.cu:7561](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7561>)-7564

다만 두 결함이 남는다.

1. 설계의 parity59 계약은 iter 10인데 소스는 `[0, pc_iter)`의 임의 iteration을 허용한다. 올바른 epoch는 launch 설정에 의존하므로 **[UNRESOLVED]**다.
2. Payload를 먼저 완전히 닫은 뒤 sidecar를 연다([lumina_cmfgen.c:259](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:259>)-320). Sidecar path/open/write 실패 시 이미 생성된 payload를 삭제하거나 quarantine하지 않는다. 실행은 실패하지만 불완전 artifact pair가 남을 수 있어 artifact-level fail-closed는 아니다.

실 production dump 및 Stage 3.1 consumer 입력 적합성은 실행 증거가 없으므로 **[UNRESOLVED]**다.

---

## 6. 신규 clamp/floor/cap 전수 — **[PASS]**

A2 귀속 신규 경로에서 수치 값을 잘라내는 새 clamp/floor/cap은 발견하지 못했다.

- EW의 `nstar_cap`이라는 이름은 실제 cap이 아니라 표현범위 초과 transition을 거부하고 계수하는 분기다([lumina_element_wide.c:427](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:427>)-433).
- Writer의 nonnegative/finite 검사는 입력 거부이며 값 수정이 아니다([lumina_cmfgen.c:246](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:246>)-256).
- Pair의 `1e30`은 기존 좌표로 유지되며 EW에 복제되지 않았다.

따라서 이 항목에 따른 추가 좌표 FAIL은 없다.

---

## 7. D6형 항등식 재발 — **[FAIL]**

### D6 행렬 residual 자체 — **[PASS]**

기존 단순 `+r/-r` 열합만 보는 방식보다 개선됐다.

- 행렬 update와 별도 `expected_outflow` ledger를 각각 누적: [lumina_element_wide.c:295](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:295>)-302
- 나중에 행렬 off-diagonal 합과 diagonal debit를 ledger와 독립 대조: [lumina_element_wide.c:1193](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1193>)-1208
- residual이 topology와 numerical gate 양쪽에 실제 투입됨: [lumina_element_wide.c:1621](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1621>)-1624

행렬의 debit 또는 off-diagonal write 한쪽만 손상시키면 FAIL할 수 있으므로 순수 항등식은 아니다. 다만 잘못된 target/channel/rate가 ledger와 행렬에 함께 전달되는 결함은 잡지 못한다.

### 재발 및 음성대조 — **[FAIL]**

- R7 writer의 η 감사는 실제 비교 없이 `true/0.0`을 기록하므로 D6형 공허 감사가 재발했다.
- `serialize(parse(raw)) == raw` 검사는 동일 parser가 읽은 값을 동일 schema로 다시 pack하는 구조다([cmf_chieta_roundtrip_selftest.py:73](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmf_chieta_roundtrip_selftest.py:73>)-76). 형식 완전성 검사는 되지만 field 의미나 잘못된 배열 교환을 검출하지 못한다.
- R1 시험은 정상 armed/unarmed 비교만 수행한다([wave32_r1_byte_invariant.py:78](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/wave32_r1_byte_invariant.py:78>)-104).
- R7 fixture도 정상 writer 호출 하나뿐이다([cmf_chieta_writer_fixture.c:11](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmf_chieta_writer_fixture.c:11>)-28).
- 소스·스크립트·Makefile 어디에도 matrix debit 손상, 잘못된 η, gate boolean 전복 등을 주입하는 seeded-defect hook/negative test가 없다.

따라서 신설 검사의 일부는 구조적으로 FAIL 가능하지만, 요구된 음성 결함 시연은 충족되지 않았고 R7에는 실제 항등형 감사가 남아 있다.

## 결론

A2 구현은 shadow production-field 격리와 R7 binary schema의 상당 부분을 제대로 구현했지만, 독립 승인 기준으로는 **전체 FAIL**이다. 차단 결함은 다음 다섯 가지다.

1. Private/commit 인덱서를 하나의 projection builder로 통합하지 않음.
2. Frozen harness가 EW 실패 반환값을 폐기함.
3. Top-stage bf 소비자가 공유 field helper를 우회함.
4. GPU field bypass가 production 계측에 남지 않음.
5. R7 η 감사가 하드코딩되고 seeded-defect 음성대조가 없음.