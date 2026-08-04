읽기 전용 지시를 지켜 파일·코드·덱을 수정하지 않았고, 모델/GPU 실행도 하지 않았습니다. 따라서 `docs/CODEX_EXTRA_IONS.md`는 생성하지 않았으며 아래가 그 문서에 들어갈 설계 보고입니다.

# CODEX_EXTRA_IONS — 「고리 밖 감사」

## 결론

- Lumina 덱: 59이온, FL 31,792, SL 12,335
- CMFGEN 활성 링크: 27이온
- Lumina 전용: 정확히 32이온, FL=SL=10,607
- 배타적 분류: `(a) 6`, `(b) 0`, `(c) 26`
- 32이온 모두 CMFGEN 원자파일은 존재하고 모두 `atomic_links.txt`에는 없다.
- `(c)` 26이온은 Lumina에서 불활성이 아니다. 해당 9원소 모두 양의 abundance를 받아 적어도 셸 0–29에서 이온화·선/연속 opacity 및 일부 NLTE 율 계산에 참여한다. 즉 형식적 위반이 아니라 물리적 오염이다.
- 32이온 격리 뒤 ftos 기준 논리적 최대 연립계는 `N=240`.
- 그러나 현 GPU allocator는 full-level offset으로 `N=4198`을 잡으므로 격리만 해서는 GPU 할당이 줄지 않는다. allocator 계약 수정이 함께 필요하다.
- 권고 창고 형식: 덱 내부의 물리적으로 분리된 `quarantine/` 하위 디렉터리 + 기계 판독 매니페스트. `active=0` 단독 방식은 금지한다.
- 활성 집합 게이트는 `Lumina_active = CMFGEN_linked`의 양방향 exact set/multiset identity로 설계한다.

근거가 되는 동일성 기준은 [ATOMIC_EQUIV_PLAN.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/ATOMIC_EQUIV_PLAN.md:6)의 무허용오차·전수검사 계약과 원자데이터 범위 정의다.

## 1. 32이온 전수표

`ion0`은 0-기반 원값이다. 32개 모두 `f_to_s`가 없어 `SL=FL`이다. 원자료는 [levels.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos/levels.csv:1)와 [atomic_vintage_manifest.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos/atomic_vintage_manifest.csv:1)에서 전수 집계했다.

| Z | ion0 | 분광표기 | FL | SL | 배타적 분류 |
|---:|---:|---|---:|---:|---|
| 6 | 0 | C I | 242 | 242 | (c) |
| 6 | 1 | C II | 338 | 338 | (c) |
| 6 | 2 | C III | 243 | 243 | (c) |
| 8 | 0 | O I | 199 | 199 | (c) |
| 8 | 1 | O II | 340 | 340 | (c) |
| 8 | 2 | O III | 343 | 343 | (c) |
| 12 | 0 | Mg I | 152 | 152 | (c) |
| 12 | 1 | Mg II | 80 | 80 | (c) |
| 12 | 2 | Mg III | 201 | 201 | (c) |
| 13 | 0 | Al I | 80 | 80 | (c) |
| 13 | 1 | Al II | 80 | 80 | (c) |
| 13 | 2 | Al III | 80 | 80 | (c) |
| 13 | 3 | Al IV | 201 | 201 | (c) |
| 14 | 0 | Si I | 493 | 493 | (a) |
| 16 | 0 | S I | 477 | 477 | (a) |
| 20 | 0 | Ca I | 184 | 184 | (a) |
| 21 | 0 | Sc I | 60 | 60 | (c) |
| 21 | 1 | Sc II | 500 | 500 | (c) |
| 21 | 2 | Sc III | 87 | 87 | (c) |
| 22 | 1 | Ti II | 600 | 600 | (c) |
| 22 | 2 | Ti III | 600 | 600 | (c) |
| 22 | 3 | Ti IV | 126 | 126 | (c) |
| 23 | 0 | V I | 1 | 1 | (c) |
| 24 | 0 | Cr I | 200 | 200 | (c) |
| 24 | 1 | Cr II | 600 | 600 | (c) |
| 24 | 2 | Cr III | 1,000 | 1,000 | (c) |
| 24 | 3 | Cr IV | 200 | 200 | (c) |
| 25 | 1 | Mn II | 600 | 600 | (c) |
| 25 | 2 | Mn III | 1,000 | 1,000 | (c) |
| 26 | 0 | Fe I | 300 | 300 | (a) |
| 27 | 0 | Co I | 200 | 200 | (a) |
| 28 | 0 | Ni I | 800 | 800 | (a) |
| **합계** |  | **32이온** | **10,607** | **10,607** | **a=6, b=0, c=26** |

분류별 FL/SL:

| 분류 | 이온 | FL=SL |
|---|---:|---:|
| (a) 데이터 존재·조성 존재·미링크 | 6 | 2,454 |
| (b) CMFGEN 데이터 부재 | 0 | 0 |
| (c) CMFGEN 조성 부재 | 26 | 8,153 |

### 분류 규칙과 중첩 처리

세 조건은 논리적으로 완전히 배타적이지 않다. `(c)` 26이온도 “원자파일은 있으나 링크되지 않음”이라는 사실만 보면 `(a)` 조건을 만족한다. 따라서 다음 우선순위로 배타화했다.

1. 실제 toy06 CMFGEN 조성에 원소가 없으면 `(c)`
2. 조성은 있으나 링크가 없고 원자파일이 있으면 `(a)`
3. 조성은 있으나 원자파일 자체가 없으면 `(b)`

독립 사실값은 다음과 같다.

- 원자파일 존재: 32/32
- `atomic_links.txt` 링크 부재: 32/32
- 실제 CMFGEN 조성 부재: 26/32
- CMFGEN 조성 존재하지만 해당 이온 미링크: 6/32
- 배포 데이터 자체 부재: 0/32

CMFGEN 원자 트리 `/gpfs/kjhan/cmfgen_21jun23/atomic/`에서 32개 모두 대응 `19apr23/osc_data`가 존재했다. Lumina 매니페스트에서도 모두 `selection_source=auto`이며, 활성 27개는 `selection_source=links`다.

## 2. CMFGEN 조성과 Lumina의 물리적 활성 여부

CMFGEN의 [MODEL_SPEC](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MODEL_SPEC:10)는 Si II–V, S II–V, Ca II–V, Fe II–VI, Co II–VI, Ni II–VI의 27개 `_ISF`만 선언한다. `atomic_links.txt`도 `_F_OSCDAT` 링크가 정확히 27개다.

실제 유체 조성은 [SN_HYDRO_DATA](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/SN_HYDRO_DATA:5)의 6개 질량분율뿐이다.

- Si
- S
- Ca
- Fe
- Co
- Ni

[VADAT](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/VADAT:714)의 범용 abundance 값은 `SN_HYDRO_DATA`로 override된다고 명시돼 있으므로 C/O/Mg/Al/Sc/Ti/V/Cr/Mn을 실제 런 조성으로 해석하면 안 된다.

반면 Lumina [abundances.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos/abundances.csv:2)는 `(c)` 원소 모두에 양의 값을 준다.

| 원소 | Lumina 질량분율 |
|---|---:|
| C | 0.02 |
| O | 0.0394 |
| Mg | 0.005 |
| Al | 0.0005 |
| Sc | 0.00003 |
| Ti | 0.0003 |
| V | 0.0002 |
| Cr | 0.003 |
| Mn | 0.0015 |

현재 로더는 활성 이온 필터 없이 `line_list.csv`와 `levels.csv` 전체를 읽고([lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:642)), 모든 원소 abundance로 이온 population table을 만든다([lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:817)). Plasma 계산 역시 모든 원소를 순회해 `n_element`를 계산한다([lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2452)).

또한 고정 NLTE 대상 배열에는 C II/III, O I/II/III, Mg II/III, Al II/III, Sc II/III, Ti II/III, Cr II/III, Mn II/III 등이 직접 들어 있다([lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7677)). Stage-IV 배열은 Al IV·Ti IV·Cr IV도 포함한다.

따라서:

- `(c)` 26이온은 물리적으로 불활성이 아니다.
- 다수는 NLTE 행렬에 직접 들어간다.
- NLTE 대상이 아닌 중성 이온도 line opacity, 이온화 분배 및 기타 bulk atomic 경로에 남는다.
- 결론은 `PHYSICAL_CONTAMINATION`, 단순 `FORMAL_ONLY`가 아니다.

### abundance 형상 결함

[config.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos/config.json:5)은 `n_shells=50`이지만 `abundances.csv`에는 셸 0–29만 있다. 현재 `strtod` 반복 로더 동작상 셸 30–49는 0으로 채워진다. 따라서 `(c)` 원소는 적어도 30개 셸에서 확실히 활성이다.

## 3. 창고 설계

### 대안 비교

| 방식 | 장점 | 위험/단점 | 판정 |
|---|---|---|---|
| 기존 표에 `active=0` 열 추가 | 이동이 없고 복원이 간단함 | 현재 로더는 열을 무시하고 전체를 읽으므로 fail-open. CSV·NPY·매크로원자·충돌자료 등 모든 소비자가 동일 필터를 구현해야 함 | 금지 |
| 별도 매니페스트만 두고 원자료는 혼재 | 변경량이 작고 정책 표현이 쉬움 | 필터 누락 하나로 선·준위·σ·Υ 일부가 새어 들어감. 파생 배열의 부분 누출 탐지가 어려움 | 단독 사용 금지 |
| 덱 내 물리적 `quarantine/` + 매니페스트 | 기존 비재귀 로더 경로와 물리적으로 분리됨. fail-closed 및 감사·복원이 쉬움 | 활성 뷰 재생성 도구와 전수 무결성 검사가 필요함 | **권고** |

### 권고 구조

```text
deck/
  active_ions.csv
  levels.csv
  line_list.csv
  ionization_energies.csv
  ...
  quarantine/
    manifest.json
    manifest.csv
    ions/
      Z006/ion000/...
      Z024/ion002/...
    elements/
      Z006/abundances.csv
      ...
```

핵심 원칙:

- 루트 파일은 활성 27이온만 포함하는 materialized active view다.
- `quarantine/`은 원본 레코드와 파생 데이터의 이온별 절편을 byte-preserving 형식으로 보존한다.
- `(c)` 원소는 abundance 행과 ionization ladder도 함께 격리한다.
- `(a)` 중성 이온은 준위·선뿐 아니라 중성 stage를 생성하는 ionization-boundary 자료까지 함께 투영해야 한다.
- 원자질량 같은 공용 표가 남더라도, 원소 열거는 `active_ions.csv`에서 유도해야 한다.

### 로더 계약

1. `active_ions.csv`는 수동 allowlist가 아니라 고정된 CMFGEN `MODEL_SPEC`와 `atomic_links.txt`에서 생성한다.
2. 로더는 덱 루트의 명시된 파일만 열며 재귀 glob을 금지한다.
3. 중앙 파일-open 계층에서 `/quarantine/` 소비 요청을 즉시 `[ATOMIC-ACTIVE-SET-LEAK]` fatal로 처리한다.
4. 모든 준위·선·광이온화·충돌·이온화에너지·zeta·macro-atom·reference/offset 자료가 활성 이온으로 역매핑되어야 한다.
5. 로드 직후, GPU/율 행렬 할당 전에 다음을 검사한다.

```text
active ∩ quarantined = ∅
active ∪ quarantined = preserved_original_59
loaded_ions = active_ions
모든 loaded row/object의 ion key ∈ active_ions
고아 level/line/offset/reference = 0
```

6. 매니페스트의 원본 SHA-256, 활성 뷰 SHA-256, 행 수와 실제 로드 결과가 하나라도 다르면 즉시 실패한다.
7. 테스트 fixture에는 의도적으로 읽을 수 없는 quarantine sentinel을 두어, 비활성 경로가 한 번이라도 소비되면 초기화 단계에서 실패하도록 한다.

### 사유 매니페스트

JSON을 정본으로, CSV를 사람이 읽는 색인으로 권고한다. 이온별 필수 필드 예시는 다음과 같다.

```json
{
  "schema_version": 1,
  "ion": {"Z": 24, "ion0": 2, "spectroscopic": "Cr III"},
  "status": "quarantined",
  "classification": {
    "primary": "c",
    "reason_code": "ELEMENT_ABSENT_FROM_CMFGEN_COMPOSITION",
    "precedence": "c>a>b"
  },
  "cmfgen": {
    "model_spec_present": false,
    "atomic_link_present": false,
    "atomic_data_exists": true,
    "composition_present": false,
    "atomic_paths": ["CHRO/III/19apr23/osc_data"],
    "evidence_hashes": {}
  },
  "lumina_before": {
    "abundance_min": 0.003,
    "abundance_max": 0.003,
    "nonzero_shells": 30,
    "physical_activity": "PHYSICAL_CONTAMINATION",
    "full_levels": 1000,
    "super_levels": 1000
  },
  "archive": {
    "content_hashes": {},
    "reversible": true
  },
  "restore_requirements": [
    "CMFGEN target set includes (24,2)",
    "CMFGEN composition includes Cr",
    "all bidirectional identity gates pass"
  ]
}
```

### 복원 경로

복원은 원본 덱을 직접 수정하지 않고 새 덱으로 승격한다.

1. 고정 CMFGEN 기준 런이 해당 `(Z,ion0)`을 링크하고 필요한 원소 조성을 포함하는지 확인한다.
2. quarantine SHA-256을 검증한다.
3. 해당 이온의 모든 연관 자료를 함께 승격한다.
4. 전역 level/line ID, offset, macro-atom mapping, NPY 배열 및 config 행 수를 재생성한다.
5. `active_ions.csv`를 새 기준 집합에서 다시 생성한다.
6. 매니페스트에는 삭제 대신 `restored` 이벤트를 append한다.
7. 양방향 동일성 게이트와 quarantine 누출 게이트를 모두 통과한 새 덱만 활성화한다.

현재 CMFGEN 기준이 해당 이온을 링크하지 않는 동안에는 복원하면 정확 일치 게이트가 의도대로 실패해야 한다.

## 4. NLTE `N`과 GPU 바이트

### `4198`의 실체

실제 CPU/GPU solve는 super-level offset으로 `N`을 정한다([CUDA solve](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1143), [CPU solve](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16685)).

하지만 초기 GPU allocator는 full-level offset으로 `max_N`을 계산한다([lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7719)).

- `4198 = Fe II FL 2698 + Fe III FL 1500`
- Fe II/III는 CMFGEN 활성 27이온에 포함된다.
- 따라서 32이온만 격리해도 현 allocator의 `4198`은 그대로다.
- ftos가 활성인 실제 논리적 현재 최대는 Cr II+III 또는 Mn II+III의 `600+1000=1600`.
- 32이온 격리 뒤 최대는 Fe II+III의 `135+105=240`.

즉 설계 목표는 `N=240`이지만, 이를 GPU 메모리에 반영하려면 allocator도 `active nlte_ion_super_offset`으로 계산해야 한다.

### 50셸 GPU 메모리

할당식은 [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:599)에 따라 다음과 같다.

```text
matrix = 50 × N² × 8
rhs    = 50 × N × 8
pivot  = 50 × N × 4
pointer arrays = 2 × 50 × 8
info   = 50 × 4
```

CUDA context/cuBLAS 내부 오버헤드는 제외했다.

| 상태 | N | matrix+rhs 출력값 | 나열된 GPU 버퍼 총계 |
|---|---:|---:|---:|
| 현 코드 | 4,198 | 7,050,960,800 B | 7,051,801,400 B = 6,725.122 MiB |
| 격리만 하고 allocator 미수정 | 4,198 | 7,050,960,800 B | 7,051,801,400 B |
| 현재 ftos 논리치와 allocator 일치 | 1,600 | 1,024,640,000 B | 1,024,961,000 B = 977.479 MiB |
| **32이온 격리 + 올바른 활성 SL allocator** | **240** | **23,136,000 B** | **23,185,000 B = 22.111 MiB** |

현 할당 대비 최종 절감:

- `7,028,616,400 B`
- 약 `304.15×` 축소

## 5. 양방향 동일성 게이트

### 기준 집합

```text
C_model = MODEL_SPEC의 모든 <ion>_ISF
C_link  = atomic_links.txt의 모든 <ion>_F_OSCDAT
L_active = Lumina가 실제 로드한 활성 이온
Q = quarantine manifest의 이온
```

먼저 `C_model = C_link`를 요구한다. 그다음:

```text
missing = C_link − L_active   # 결손
extra   = L_active − C_link   # 여분

PASS ⇔ missing = ∅ AND extra = ∅
```

추가 격리 불변식:

```text
L_active ∩ Q = ∅
L_active ∪ Q = 보존된 원본 59이온
```

매니페스트 선언만 비교하면 안 된다. 로더가 구성한 실제 runtime inventory를 덤프해 비교해야 한다.

### 준위 게이트

이온별 CMFGEN linked `osc_data`와 `f_to_s`를 정본으로 삼아 다음을 전수 비교한다.

- 준위 식별자 및 multiplicity
- configuration
- `g`
- `E`
- metastable 속성
- full-level 순서
- super-level membership

```text
CMF_level_keys − Lumina_active_level_keys != ∅ → FAIL_MISSING
Lumina_active_level_keys − CMF_level_keys != ∅ → FAIL_EXTRA
같은 key의 값 또는 raw round-trip 표현 불일치 → FAIL_VALUE
```

허용오차와 표본검사는 없다. hash는 가속·감사 수단일 뿐, 충돌 없는 전수 값 비교가 최종 판정이다.

### 선 게이트

선은 단순 `(lower,upper)` set이 아니라 중복도를 보존한 multiset으로 비교한다.

권고 키:

```text
(Z, ion0, lower_level_identity, upper_level_identity,
 CMF stable_line_id 또는 동일 전이 내 occurrence_index)
```

비교 값:

- `f_lu`
- `A_ul`
- `λ`
- 전이 방향
- 중복 multiplicity

각 방향을 별도 계산한다.

```text
CMF_lines − Lumina_active_lines → 결손 FAIL
Lumina_active_lines − CMF_lines → 잉여 FAIL
동일 key 값 불일치 → 값 FAIL
고아/중복도 불일치 → 구조 FAIL
```

따라서 사용자 제시 I18의 역방향 잉여 선 약 170만 건과 준위 15,606건은 각각 `FAIL_EXTRA_LINE`, `FAIL_EXTRA_LEVEL`로 잡힌다. 격리 이온은 `L_active`에 들어가지 않으므로 이 비교에서는 제외되며, 별도 archive-integrity 검사만 받는다.

같은 양방향 계약을 σ 전 점, Υ 전 항목, 원자 판본, `f_to_s`에도 적용해야 [ATOMIC_EQUIV_PLAN.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/ATOMIC_EQUIV_PLAN.md:60)의 전체 동일성 기준을 만족한다.

## UNRESOLVED

- `config.json`의 `n_lines=137,252`와 실제 `line_list.csv`의 2,220,953행이 불일치한다. 새 게이트는 이를 초기 cardinality FAIL로 처리해야 한다.
- `n_shells=50`인데 abundance 열은 30개뿐이다. 현재 물리적 결과는 뒤 20셸이 0이지만, 이것이 의도인지 생성기 결함인지는 확정되지 않았다.
- 32이온을 제외한 활성 뷰도 현재 집계상 FL 21,185/SL 1,728이며 CMFGEN `MODEL_SPEC` 집계 FL 20,749/SL 1,637과 다르다. 즉 quarantine만으로 전체 준위 동일성이 완성되지는 않는다.
- 사용자 제시 I18 역방향 수치는 이번 설계 감사에서 별도로 재실행·재인증하지 않았다. 구현될 게이트가 고정 CMFGEN 원자파일에서 전수 재산출해야 한다.
- `N=240` 메모리 이득은 allocator가 full-level 기준에서 활성 super-level 기준으로 바뀌어야 실현된다. 격리만 구현하면 GPU 메모리는 현재와 동일하다.