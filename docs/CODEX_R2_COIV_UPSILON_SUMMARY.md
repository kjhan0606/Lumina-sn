# R2 — Co IV 충돌강도 Υ 대용 규명

읽기 전용 지시에 따라 `docs/CODEX_R2_COIV_UPSILON.md`는 생성하지 않았다. 아래가 해당 문서의 완결 내용이다. 파일 수정·commit·모델 런·GPU 사용은 없었다.

## 결론

새 `_links` 덱에서 Co IV `(Z=27, ion0=3; Co IV)`의 Fe III 대용표는 사라졌다.

- CMFGEN과 Lumina 모두 링크된 `COB/IV/18oct00/col_guess.dat`를 선택한다.
- 이 파일은 tabulated 전이 0개를 선언한다.
- 새 덱에도 Co IV tabulated Υ는 0개이며 `ige_col_27_3_cmfgen.bin`도 없다.
- 따라서 R2의 “Co IV에 Fe III 4,455전이를 잘못 적용”하던 데이터 불일치는 R1 판본 정렬로 해소됐다.
- 구 `19apr23` 문제의 원인은 Lumina 오매핑이 아니라, 원본 Co IV 파일 자체가 Fe III 값을 전용한 것이었다.

다만 “전체 충돌 처리의 수치 동일성”은 아직 성립하지 않는다. Lumina의 CMFGEN fallback 포트에는 `SAME_N` 미구현과 `8.629e-6` 대 CMFGEN `8.63e-6` 차이가 남아 있다.

## 1. 새 덱 재측정

실제 선택은 매니페스트에서 다음과 같다.

- Fe III `(26,2; Fe III)`: `FE/III/19apr23/col_data`, 22,139전이×20온도 — [atomic_vintage_manifest.csv:45](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_links/atomic_vintage_manifest.csv:45)
- Co IV `(27,3; Co IV)`: `COB/IV/18oct00/col_guess.dat`, 0전이 — [atomic_vintage_manifest.csv:52](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_links/atomic_vintage_manifest.csv:52)

| 항목 | Co IV `(27,3; IV)` | Fe III `(26,2; III)` |
|---|---:|---:|
| 원문 선언 전이 | 0 | 22,139 |
| Lumina mapped 전이 | 0 | 22,139 |
| 일치 건수 | 0/0 | — |
| 최대 절대차 | N/A — 비교값 없음 | — |
| 온도격자 | 9점, 2,000–1,000,000 K | 20점, 1,000–100,000 K |
| 온도격자 비트동일 | 아니오 | — |

Co IV 결과는 [coldata 매니페스트:52](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_links/coldata_cmfgen_manifest.csv:52)에 `SKIP: col_data declares 0 transitions`로 기록됐다. HDF5의 Co IV 그룹에도 `col`이 없고, Co IV 바이너리도 없다.

요청에 적힌 “Fe III 같은 판본 18oct00”은 실제 트리와 일치하지 않는다. 로컬 및 `/gpfs` 원자 트리에 `FE/III/18oct00`은 없으며, 벤치마크 링크가 지정한 Fe III 판본은 `19apr23`이다.

## 2. 다른 이온 대용 전수검사

새 덱의 활성 충돌표는 34이온, 총 106,091 mapped 전이다. 모든 활성표를 온도격자와 Υ 벡터 멀티셋으로 전 이온 쌍 비교했다.

전체 표가 다른 활성표의 정확한 부분집합인 관계는 하나뿐이다.

- Si I `(14,0; I)` ↔ S I `(16,0; I)`: 각각 11 mapped 전이의 Υ 멀티셋과 온도격자 비트동일, 값 최대차 0. 준위쌍 번호는 서로 다르다.

원문이 다른 이온 전용을 명시하는 활성 이온 전수 목록은 다음과 같다.

| 대상 이온 | 원문 선언 | source→mapped |
|---|---|---:|
| Al I `(13,0; I)` | Si II와 유사한 추정값. 파일 제목도 Si II | 1→6 |
| Si I `(14,0; I)` | C I 자료에 기반한 임의 추정 | 9→11 |
| S I `(16,0; I)` | C I 자료에 기반한 임의 추정 | 9→11 |
| Ti III `(22,2; III)` | Fe VII 충돌강도, 온도 `z²` scaling | 78→78 |
| Co III `(27,2; III)` | 구조가 유사한 Ni IV rates | 8→88 |

직접 근거는 [Al I col_data:4](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/AL/I/19apr23/col_data:4), [Si I col_data:5](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/SIL/I/19apr23/col_data:5), [S I col_data:5](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/SUL/I/19apr23/col_data:5), [Ti III col_data:16](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/TIT/III/19apr23/col_data:16), [Co III col_data:12](/gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/18oct00/col_data.dat:12)이다.

이는 CMFGEN 원본/링크가 사용하는 대용이므로 동일성 기준에서는 Lumina도 그대로 유지해야 한다.

## 3. 원인: 원본 대용이며 오매핑이 아님

구 `COB/IV/19apr23/col_data`는 파일 자체가 다음을 밝힌다.

- Co IV rates가 `Zha96_FeIII_col` 출처
- “Using FeIII values?”라는 주석
- 4,455전이×20온도 — [Co IV 19apr23 col_data:12](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/COB/IV/19apr23/col_data:12)

원문 재대조 결과도 이전 측정과 동일하다.

- Co IV 4,455개 전부가 Fe III 22,139개 값 멀티셋의 정확한 부분집합
- 온도격자 비트동일
- 최대 절대차 0
- 준위명+값까지 같은 전이 4,357개
- Fe III 근거: [FE III col_data:21](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/FE/III/19apr23/col_data:21)

반면 `18oct00`은 “당분간 0”, 기본 `OMEGA=0.1`, 선언 전이 0개다 — [Co IV 18oct00 col_guess.dat:12](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/COB/IV/18oct00/col_guess.dat:12).

Lumina 매핑 경로에도 교차 이온 오매핑은 없다.

- 링크 파서는 네 입력을 같은 `(Z, stage)`로 강제한다 — [expand_atomic_data_cmfgen.py:354](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:354)
- Co IV의 `cp`는 링크의 `col` 파일 그대로다 — [expand_atomic_data_cmfgen.py:454](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:454)
- 빌더는 `(Z, ion0)`별 `osc_path,col_path`를 매니페스트에서 읽는다 — [build_cmfgen_coldata_all.py:555](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_cmfgen_coldata_all.py:555)
- 선언 전이 0이면 바이너리를 만들지 않는다 — [build_cmfgen_coldata_all.py:246](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_cmfgen_coldata_all.py:246)
- 런타임은 `status==OK` 바이너리만 적재한다 — [lumina_atomic.c:1492](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1492)

## 4. CMFGEN의 Co IV 처리

`CoIV_COL_DATA` 링크는 존재한다.

- [atomic_links.txt:80](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt:80)  
  `CoIV_COL_DATA -> COB/IV/18oct00/col_guess.dat`

실제 CMFGEN 출력도 `CoIV_COL_DATA 0`이라고 기록한다 — [COLLISION_SUMMARY:142](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/COLLISION_SUMMARY:142).

CMFGEN reader는 `NUM_TRANS=0`이면 tabulated 자료 읽기를 끝내고 “approximate formulae only”로 돌아간다 — [gen_omega_rd_v2.f:227](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/gen_omega_rd_v2.f:227).

그 뒤 모든 Co IV bound-bound 쌍은 다음 처방을 사용한다 — [omega_gen_v3.f:151](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/omega_gen_v3.f:151).

- `f ≤ 1e-5`: `OMEGA_SET=0.1`
- 그 외 permitted transition: van Regemorter형  
  `Ω = 47.972 × OMEGA_SCALE × gbar × f × g_lo / FL`
- 이온의 `gbar = max(G1, 0.276 exp(x)E1(x))`
- `G1=0.7` if `SAME_N`, 아니면 `0.2`
- `OMEGA_SCALE=1.0`, `OMEGA_SET=0.1`

## 5. 동일성 판정과 수리안

판정은 두 층으로 나뉜다.

1. **Tabulated Υ 데이터 동일성: PASS**

   CMFGEN=0, Lumina=0이다. Co IV의 Fe III 4,455전이 제거는 이미 R1 링크 선택으로 달성됐다. Co IV 특례 삭제나 추가 파일 교체는 필요 없다.

2. **전체 Co IV 충돌 처리 수치 동일성: 아직 FAIL**

   Lumina의 fallback 포트는 다음 차이를 코드가 스스로 인정한다.

   - CMFGEN `SAME_N`의 `G1=0.7`을 구현하지 않고 항상 0.2 사용 — [lumina_plasma.c:605](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:605), [lumina_plasma.c:648](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:648)
   - 충돌률 상수 Lumina `8.629e-6` 대 CMFGEN `8.63e-6` — [lumina_plasma.c:528](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:528), [subcol_multi_v4.f:109](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/subcol_multi_v4.f:109)
   - `OMEGA_SET`이 전 이온 파일별 값이 아니라 전역 기본값/환경변수다 — [lumina_plasma.c:668](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:668)
   - CMFGEN 경로가 기본 OFF인 런타임 gate다 — [lumina_plasma.c:658](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:658)

수리는 다음과 같이 해야 한다.

- [lumina.h:385](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:385)와 [lumina_atomic.c:733](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:733): `levels.csv.configuration`을 런타임에 보존하거나, 동일한 결과의 per-line `SAME_N` sidecar를 적재한다.
- [lumina_plasma.c:648](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:648): CMFGEN [same_n.f:1](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/same_n.f:1)을 그대로 포트하고 `G1=0.7/0.2`를 분기한다.
- [lumina_plasma.c:500](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:500): CMFGEN 동일성 경로 전용 rate 함수에서 `8.63e-6`을 사용한다. ARTIS용 `8.629e-6` 전역 상수를 바꿔 다른 경로를 오염시키면 안 된다.
- [build_cmfgen_coldata_all.py:148](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_cmfgen_coldata_all.py:148): 전이 0 파일도 `OMEGA_SCALE`과 `OMEGA_SET`을 파싱해 이온별 policy sidecar/매니페스트에 기록한다.
- [lumina_atomic.c:1467](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1467): `status==OK` 표뿐 아니라 0-transition 이온의 fallback policy도 적재한다.
- [verify_deck_r1_vintage.py:316](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_r1_vintage.py:316): 신규 R2 gate를 추가해 Co IV 링크 경로, 선언 전이 0, mapped 0, 바이너리/HDF `col` 부재, `OMEGA_SCALE=1.0`, `OMEGA_SET=0.1`을 전부 강제한다.
- CMFGEN 비교 모드에서는 `LUMINA_OMEGA_CMFGEN`을 선택적 실험 gate가 아니라 필수·fail-closed 설정으로 만든다.

## UNRESOLVED

- Co IV tabulated 자료의 동일성은 확정됐지만, fallback 전 선의 비트/수치 동일성은 위의 소스상 차이 때문에 미달이다.
- Ni I `(28,0; I)` 파일 제목은 “Co II”이지만 본문은 다른 이온 전용이라고 명시하지 않는다. 21개 원문 전이가 모두 상수 1이고 69쌍으로 분할되지만, 실제 Co II 전용인지 단순 복사 헤더인지는 원본만으로 확정할 수 없다 — [Ni I col_data:4](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen/NICK/I/19apr23/col_data:4).
- `FE/III/18oct00`은 원자 트리에 없으므로 해당 판본과의 비교는 불가능하고, 실제 정본 링크인 `FE/III/19apr23`을 사용했다.