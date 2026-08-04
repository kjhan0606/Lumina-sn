# Wave 3 D-5 Codex A 구현 보고

기준 명세는 `docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md`이다. 이 보고서는 Stage 2A CPU gated path, s8 S/Fe 파일럿과 s0 Fe/s20 S shadow 실행 준비만 다룬다. §4 acceptance, CMFGEN parity, spectrum 개선은 판정하지 않는다.

## 1. 변경 골자

| 파일 | 구현 |
|---|---|
| `src/lumina_element_wide.c` | strict gate/parser, II–IV identity/indexer, target CSR projection, 7개 channel plane, 단일 보존행, equilibration/LU/refinement/condition 진단, 결정론 dump, fail-closed commit |
| `src/lumina.h` | element-wide layout/channel/API 계약과 pair/ion 최대치 분리 |
| `src/lumina_plasma.c` | S IV/Fe IV 인접 layout, 공통 pair rate producer capture hook, element-wide 캡처 중 legacy clamp/TOPSTAGE/CE/closure 우회, shadow/commit 호출 및 save-restore/damping 배제 |
| `src/lumina_atomic.c` | element-wide ON에서만 target CSR 강제 load |
| `src/lumina_cuda.cu` | element-wide ON CUDA binary를 CPU double reference path로 라우팅 |
| `bench_frozen_oracle.c` | frozen s0/s8/s43 fixture에서 label 기반 shadow 호출; 명시 ON에서만 s20 추가 시도; OFF fixture는 기존 권위 유지 |
| `Makefile` | CPU/CUDA/bench/selftest link에 새 모듈 추가 |

주요 구현 위치:

- gate/parser와 s0/s8/s20 범위: `src/lumina_element_wide.c:41-146`
- 원자적 `j -> i` 배치와 모든 target CSR route: `src/lumina_element_wide.c:191-279`
- projection/identity/checksum: `src/lumina_element_wide.c:301-427`
- pivoted LU, singular estimate, rank, ARTIS식 equilibration/refinement: `src/lumina_element_wide.c:428-712`
- element-wide orchestration, 보존행, gate, dump, commit: `src/lumina_element_wide.c:847-1023`
- 공통 producer capture와 legacy 보정 우회: `src/lumina_plasma.c:14523`, `src/lumina_plasma.c:15085-16130`
- 대상 pair/save-restore/damping 배제: `src/lumina_plasma.c:16991-17135`
- CUDA binary의 CPU reference 라우팅: `src/lumina_cuda.cu:997-1004`

## 2. gate 계약

| 명세 gate | 구현 | 기본값/제약 |
|---|---|---|
| `LUMINA_NLTE_ELEMENT_WIDE` | 동일 | 기본/미설정 0; 정확히 1만 ON |
| `LUMINA_NLTE_ELEMENT_WIDE_Z` | 동일 | ON이면 필수, `16,26` 부분집합만 허용 |
| `LUMINA_NLTE_ELEMENT_WIDE_SHELL` | 동일 | ON이면 필수; s8 파일럿, s0 Fe/s20 S shadow 허용 |
| `LUMINA_NLTE_ELEMENT_WIDE_COMMIT` | 동일 | 기본 0; s8에서만 1 허용 |
| `LUMINA_NLTE_ELEMENT_WIDE_DUMP` | 동일 | 기본 0 |
| dump directory | `LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR` | 선택; 기본 현재 디렉터리 |

미설정과 명시 0은 parser 첫 분기에서 함께 반환한다. 이 경로에는 새 allocation, target-map load, banner, dump, counter, RNG 소비가 없다.

## 3. 명세 조항별 준수

| 조항 | 상태 | 근거 |
|---|---|---|
| §0 범위/기본 OFF | 구현 | Stage 2A CPU, S/Fe, s0/s8/s20만 허용; default OFF |
| §1 II–IV canonical identity | 구현 | internal 1–3 ↔ spectroscopic II–IV; S/Fe 모두 `N=303` |
| §1.3 경계 active-set | fail-closed | s8 I/V 최대 fraction이 문턱 초과하여 commit 금지 |
| §2 공통 rate producer | 구현 | pair assembler 산술을 복제하지 않고 capture hook으로 재사용 |
| §3.2 행렬 identity | 구현 | column=source, row=target; inflow/동일 source diagonal을 한 함수에서 배치 |
| §3.3 bb/full→SL | 구현 | source within-SL fraction 사용, 두 번째 pair의 III bb 중복 캡처 배제 |
| §3.4 target CSR | 구현/fail-closed | 모든 route와 probability를 캡처; coverage 불완전 시 solve 시작 안 함 |
| §3.5 7 channel | 구현 | rad/coll/NT bb, rad/coll/NT bf, autoion/DR; 비활성 근거 manifest 기록 |
| §3.6 단일 보존행 | 구현 | row 0 하나, RHS=`n_element`; charge row 0개 |
| §3.7 conditioning | 구현 | 최대 10회 equilibration, partial-pivot LU, 최대 10회 refinement, rank/SVD estimate/rcond/pivot/residual dump |
| §3.8 writeback | 구현 | PASS+COMMIT만 SL 해를 within-SL fraction으로 FL 복원; floor/cap/damping 없음 |
| §5 dump | 구현 | identity/raw/normalized/equilibrated/solution/diagnostics/provenance/manifest 8개 역할 |
| §6 OFF/COMMIT | 구현 | shadow는 baseline 유지; commit 후보만 pair/save-restore/damping을 배제; 어느 gate 실패도 baseline fallback verdict |
| §7 A형 clamp | 구현 | element-wide capture에서 legacy floor/metacoll/DR floor/TOPSTAGE/CE/time/pin/anchor/repair 이전 반환; 전역 삭제 없음 |
| §8.7 역방향 축 준비 | 구현 | gate와 frozen fixture가 s0 Fe 및 s20 S shadow를 허용; commit은 거부 |

`COMMIT=1`에서도 target/identity/rank/condition/residual/boundary 중 하나라도 실패하면 element-wide population은 쓰지 않고 legacy pair path가 실행되며 verdict는 `EW_FAIL_FALLBACK_BASELINE`이다.

## 4. 자체 검증

검증 중 `git` 명령과 GPU 실행은 사용하지 않았다.

### 4.1 OFF oracle

최종 바이너리에서 미설정과 `LUMINA_NLTE_ELEMENT_WIDE=0`을 별도 프로세스로 실행했다. 양쪽 모두 exit 0, stderr 0 byte, EW artifact 0개이며 s0/s8/s43 CSV `cmp=0`이다.

| cell | SHA-256 |
|---|---|
| s0 | `4789f13c89a3bb613e89cb23e836242285aae31bee6065b2631d61324eee1952` |
| s8 | `a4f1a146a313501a3eaf56232d2d7d3cd4f798425ebd8f426067292edb1538e2` |
| s43 | `c48d2619f160191d4a91e37334cf165d2fc312d2263635a281112523e70b72aa` |

artifact: `/tmp/w3_final_off_u.eV5pJw`, `/tmp/w3_final_off_0.EfzGLF`.

### 4.2 s8 shadow 2회

두 실행 모두 exit 0이며 각 실행의 EW dump 16개(S 8개 + Fe 8개)와 oracle CSV 전부 `cmp=0`이다.

artifact: `/tmp/w3_final_s8_a.iIAJ6g`, `/tmp/w3_final_s8_b.Sla5Q2`.

| 항목 | S II–IV | Fe II–IV |
|---|---:|---:|
| N / raw rank | 303 / 302 | 303 / 302 |
| ion / SL coverage | 3/3, 303/303 | 3/3, 303/303 |
| full level | 898/898 | 4398/4398 |
| line | 18526/18526 | 672261/672261 |
| continuum/target | 574/581 | 4076/4198 |
| first bad lower global | 4036 | 11965 |
| graph component / zero row / zero column | 1 / 0 / 0 | 1 / 0 / 0 |
| legacy guard configured/firing | 2 / 0 | 2 / 0 |
| boundary fraction max | 1.9575712599488889e-5 | 7.9230440666087251e-5 |
| verdict | `EW_FAIL_SHADOW` | `EW_FAIL_SHADOW` |

모든 channel relative column sum은 `2.85e-16` 이하이다. target coverage와 boundary gate가 먼저 실패하므로 명세대로 population solve를 시작하지 않았다. 따라서 diagnostics의 `solve_attempted=0`, `kappa_2=inf`, residual=`inf`는 실패 은폐가 아니라 **not solved** 표기다. B는 `matrix_raw`/`matrix_normalized`에서 독립 condition을 산정할 수 있고, `matrix_equilibrated`/`solution`은 미시도 상태를 보존한다.

diagnostics SHA-256:

- S: `2c48ba74a893695d73de1175aeb0373f301296f9d7c3ffb062e99aa3fd7491e6`
- Fe: `a3271c6b71a066e64c7b4f1184d80293da85d31cb27f9aa2652adae42e780f0c`

### 4.3 역방향 shadow 준비 확인

- s0 Fe: exit 0, 8개 dump 생성, `N=303`, raw rank 302. 현재 target 4076/4198 및 boundary fraction 0.013749...로 `EW_FAIL_SHADOW`; artifact `/tmp/w3_shadow_fe_s0.QGoMNr`.
- s20 S: fixture/gate 호출까지 진입했으나 현재 `logs/coevolve_consume_parity50`에는 s20 Phase-1.6 frozen J/C1/C2 archive가 없어 `s20: frozen input load failed`로 exit 1, dump 0개. artifact `/tmp/w3_shadow_s_s20.BrJJ4U`. 입력을 추정하거나 s8 값을 복제하지 않고 fail-closed했다.

### 4.4 build

- `make -B bench_frozen_oracle`: exit 0.
- `make cuda`: exit 0. CUDA binary는 컴파일만 했으며 실행하지 않았다.

## 5. 미해결/후퇴 상태

1. 현재 model target CSR에 S 7개, Fe 122개 lower-level gap이 있다. fallback target이나 ground collapse를 만들지 않았고 solve를 금지했다.
2. 제외 I/V fraction이 `1e-8`을 초과한다. boundary rate/heating producer가 Stage 2A lane에 없으므로 `boundary_process_coverage=0`으로 fail-closed했다. Stage 2B active adjacent-stage 확장 전 solution acceptance 금지다.
3. 현재 parity50 archive에는 s20 frozen field가 없다. s20 방향축 산정 전 해당 cell의 consumer/producer iteration, J-bar, C1, C2를 같은 run에서 보존해야 한다.
4. coverage gate 때문에 equilibrated condition, refinement, population residual을 이번 smoke에서 산출하지 않았다. raw/normalized matrix와 identity/provenance는 보존됐다.
5. provenance는 nonzero별 집계와 CSR/direct route, probability 적용 위치, field generation, full→SL 적용 위치를 기록한다. B/C가 transition 단위 원자료 대조를 요구할 경우 producer-side 세부 trace를 별도 default-OFF 계기로 확장해야 한다.
6. CMFGEN parity, pair-wise 개선율, spectrum 개선, released-T acceptance는 선언하지 않는다.

## 6. B/C 인계 지침

Codex B:

1. 위 diagnostics checksum과 manifest의 atomic checksum을 먼저 대조한다.
2. `matrix_raw`의 7개 channel column sum, raw rank `N-1`, graph component를 독립 계산한다.
3. coverage gap lower identity 4036/11965에서 CSR 부재 원인을 원자자료와 비교한다. gap을 임의 target으로 보완하지 않는다.
4. `matrix_normalized`로 condition을 독립 산정하되 coverage/boundary FAIL을 PASS로 승격하지 않는다.
5. s0 Fe는 `Z=26,SHELL=0`, s20 S는 `Z=16,SHELL=20`, 항상 `COMMIT=0`으로 shadow 실행한다. s20은 먼저 해당 cell을 포함한 frozen archive를 제공해야 한다. 둘은 Gate B oracle이라고 부르지 않는다.
6. target 및 adjacent-stage coverage가 닫힌 뒤에만 pair-wise/ARTIS/element-wide §4 acceptance와 permutation 3종을 계산한다.

Codex C:

1. `src/lumina_plasma.c:15825`, `16045`, `16126`에서 TOPSTAGE_IV/CE/legacy closure가 capture lane에 도달하지 않는지 추적한다.
2. `src/lumina_plasma.c:17078-17135`에서 PASS+COMMIT 대상만 pair/save-restore/damping이 배제되고 FAIL은 baseline으로 복귀하는지 확인한다.
3. CSR probability가 `src/lumina_element_wide.c:216-279`에서 정확히 한 번 적용되고 III bb가 두 pair에서 중복되지 않는지 검사한다.
4. row 0 보존행 1개, charge row 0개, TOPSTAGE_IV call 0, clamp/fallback firing 0을 manifest/source 양쪽에서 대조한다.
5. 현재 적절한 판정은 acceptance PASS가 아니라 target/boundary로 중단된 `FAIL-TOPOLOGY`인지 독립 결정한다.
