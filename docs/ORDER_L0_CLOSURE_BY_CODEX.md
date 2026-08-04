# 발주서 L0-CLOSE-R2 — 층 0 검증 폐포 완결

발주일: 2026-08-04  
발주 범위: 검증·판정·증거화. 구현과 생산 배포는 별도 승인 사항이다.  
정본: [OUTSIDE_LOOP_POOL.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/OUTSIDE_LOOP_POOL.md:75), [ORDER_CD_COMPOSITION_IDENTITY.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/ORDER_CD_COMPOSITION_IDENTITY.md), [CODEX_IMPL_D_READER.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_D_READER.md), [CODEX_IMPL_C_PRODUCTION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_C_PRODUCTION.md)

## 1. 「층 0 폐합」의 정의

층 0은 다음 여섯 조건이 동시에 성립할 때만 검증 폐합으로 판정한다.

1. 정본 덱의 조성·형상·행렬 차원·초기 플라스마 입력에 대해 생산자와 소비자의 계약이 하나로 확정되어 있다.
2. 모든 층 0 생산 경로가 실제 산출물을 생성해 보았고, 산출물이 독립 검증기를 통과했다. 정적 소스 검사만으로는 폐합하지 않는다.
3. 잘못된 입력은 경고 후 대체값으로 진행하지 않고, 소비 전에 실패한다. 특히 정확해가 위반할 수 있는 clamp/floor를 합법화하지 않는다.
4. 입력 변환의 물리 영향이 오프라인에서 정량되었거나, 오프라인으로 결판할 수 없는 항목은 기대치를 사전등록한 단 한 번의 판정런으로 처분되었다.
5. 각 양성 증거에 대응하는 음성 대조가 있으며, 음성 대조가 실제로 실패한다.
6. 증거에 소스·바이너리·입력·출력 해시와 실행 환경이 붙어 있어 같은 대상을 검증했다는 점이 확인된다.

대장에는 아래 두 필드를 분리 기록한다.

- `L0_VALIDATION_CLOSED`: 위 여섯 조건 충족 여부
- `L0_PRODUCTION_DEPLOYED`: 커밋, 생산 바이너리 설치, 생산 덱 적용 및 생산 런 여부

따라서 `L0_VALIDATION_CLOSED=yes`, `L0_PRODUCTION_DEPLOYED=no`는 가능한 상태다. 이 경우 층 0의 검증은 닫혔지만 생산 반영은 되지 않은 것이다. “생산에 고쳐졌다”라고 기록해서는 안 된다.

## 2. 미결 목록 재산정

운전석의 10건은 접수 목록으로는 맞다.

- 검증 잔여 5건: C2, D 빌드, H, K, T seed
- 범위 판정 3건: F, I, J
- 배포 잔여 2건: 미커밋, 생산 적용

그러나 폐합 계약으로는 부족하다. K가 서로 독립적인 두 계약으로 갈리고, 층 0 누락 2건이 있으며, G의 범위 판정도 별도 대장행으로 남겨야 한다.

최종 대장행은 총 14개다.

### 층 0 검증 계약 8개

1. C2-EXEC — 드라이버 5기 실재생성
2. D-BUILD — D가 포함된 생산 CUDA 바이너리의 격리 빌드
3. H-TRANSFORM — IGE floor와 6원소 재규격화의 정량·처분
4. K-SHAPE — `tau_sobolev.npy` 형상 계약과 fail-closed
5. K-FRESH — 첫 소비 전 Sobolev tau 신선도 계약
6. T-SEED — T seed 의존성 처분
7. Z-INERT — 입력 조성이 정확히 0인 원소의 하류 완전 비활성
8. GEN-GUARD — 정본 덱 덮어쓰기 방지 계약

### 범위 판정 4개

9. F — 격자 범위
10. G — 준위 절단·R3c 폴백·I18
11. I — `atom_masses.csv` 토폴로지
12. J — 조성 출처 통일

### 배포 대장행 2개

13. D·C1·C2 커밋
14. 생산 바이너리·덱·런 적용

추가로 감마 침적의 외부 deposition 입력 재계산/이중붕괴 가능성은 층 1 I10 미결로 이관한다. 위 14개에는 포함하지 않는다.

## 3. 항목별 폐합 요건

### 3.1 C2-EXEC — 드라이버 5기

다음 다섯 드라이버를 모두 일회성 격리 스테이지에서 실제 실행한다.

1. `deck_quarantine_driver.py`
2. `deck_regen_fullcov_driver.py`
3. `deck_regen_r1_vintage_driver.py`
4. `deck_regen_r4_ftos_driver.py`
5. `deck_regen_r4_offcontrol_driver.py`

폐합 조건:

- 정본 및 현재 생산 덱은 실행 전후 해시가 동일해야 한다.
- 스테이지의 각 출력 덱은 새 디렉터리여야 하며 기존 출력 재사용을 금한다.
- 각 `abundances.csv`는 정본 조성과 바이트 동일하고, 일반 파일이며, 심볼릭 링크가 아니어야 한다.
- 각 행렬은 기대 행 수와 50셸을 만족해야 한다.
- 각 `tau_sobolev.npy`는 해당 덱의 `n_lines × 50`이어야 한다.
- C1 expander가 다섯 경로에서 실제 호출됐다는 실행 증거를 남긴다.
- `sbatch_deck_r4_ftos.sh`는 `R4_OFF_DIR`이 명시된 경우 `SLURM_TMPDIR` 부재 때문에 중단하지 않아야 한다. 이 실행기 계약도 C2 안에서 검수한다.
- 드라이버가 “출력 경로가 이미 존재한다”로 거부한 것을 성공으로 세지 않는다.

음성 대조:

- 30열 조성 fixture를 각 드라이버 검증기에 넣어 모두 비영 종료해야 한다.
- 입력 덱과 출력 덱을 같은 경로로 지정하면 파일을 열기 전에 거부해야 한다.
- 스테이지 경로가 정본의 심볼릭 링크 또는 hard-link 별칭이면 거부해야 한다.

### 3.2 D-BUILD — 생산 바이너리 빌드

현재 루트의 `lumina_cuda`는 D 변경을 포함했다는 증거가 없다. [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:468)의 미커밋 변경을 그대로 복제한 격리 소스 트리에서 CUDA 전체 빌드를 수행한다.

폐합 조건:

- 빌드 전 D 소스 해시와 diff를 보존한다.
- 루트 바이너리를 덮어쓰지 않는다.
- 전체 CUDA 빌드가 성공하고 새 바이너리 해시를 남긴다.
- 새 소스로 D의 양성 게이트 19/19가 다시 통과한다.
- 구 판독기를 사용한 음성 대조는 0/19 통과해야 한다.
- FATAL 16건과 WARN 2건의 종류·대상·기대 종료코드를 대조표로 남긴다.
- missing file 경로에서도 배열 차원 변수가 초기화되지 않은 채 사용되지 않는지 포함해 검사한다.

이는 검증 빌드다. 생산 바이너리 설치는 13·14번 배포 항목이다.

### 3.3 H-TRANSFORM — floor 및 재규격화

문제 코드는 [/gpfs/kjhan/cmfgen_runs/toy06_19.48d/mk_sn_hydro.py](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/mk_sn_hydro.py)이다. IGE 4종에 `1e-10` floor를 주입하고 6원소 합으로 재규격화한다.

GPU 없이 결판 가능하다. 동일한 원시 CMFGEN 90-depth 입력과 700점 변환을 다음 세 갈래로 계산한다.

1. 원시 보간값
2. floor만 적용
3. floor와 재규격화 모두 적용

각 갈래에서 아래를 셸별·원소별로 산출한다.

- `ΔX_i`
- 6원소 합과 정규화 인자
- 원소별 적분 질량
- `Σ X_i/A_i`
- 현재 line list를 사용한 파장대별 `Σ(X_i/A_i) Σ|f_lu|λ` 선불투명도 대리량
- `X56Ni`와 원소 Ni/Co/Fe 변환의 일관성

현재 예비 실측을 정식 검증기의 사전 기대치로 고정한다.

- 원시 6원소 합: `0.999992 … 1.000012`
- floor 주입 총량: `1.416e-7`
- 원시→최종 최대 `|ΔX|`: `5.15e-6` 이하
- `ΣX/A` 최대 상대변화: `1.20e-5` 수준
- floor 단독 파장대 대리량 변화: 최대 약 `1.68e-9`
- floor+재규격화 변화: 최대 약 `1.20e-5`

정식 허용 상한은 각각 `6e-6`, `1.3e-5`, `2e-9`, `1.3e-5`로 사전등록한다. 초과 시 원인 재조사이지 기준 상향이 아니다.

판정:

- 물리효과가 작다는 것은 floor를 허용하는 근거가 아니다.
- floor는 금지 규율 위반이므로 제거 대상이다. 정확한 0은 0으로 유지해야 한다.
- 재규격화는 원시 합·인자를 숨기는 입력 clamp로 남겨서는 안 된다. 유지하려면 “출처 보존값”과 “명시적 보존 투영값”을 별도 산출하고 계약으로 선언해야 한다.
- 이 규모로 기존 2.53배·5.13배 차이를 설명할 수 없다는 것은 GPU 없이 폐합할 수 있다.

음성 대조:

- 정확한 0을 넣었을 때 출력도 정확한 0이어야 한다.
- 의도적으로 합이 틀린 행은 자동 보정이 아니라 명시적 실패 또는 별도 투영 산출로 나타나야 한다.

### 3.4 K-SHAPE — `tau_sobolev.npy` 형상

정본 파일은 `(2,565,342, 30)`이고 전 원소가 0이다. 실제 생산 소비자는 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:468)이다. 이 코드는 형상이 다르면 경고 후 `n_lines × n_shells` 0 배열로 대체하고 계속한다. 즉 현재 30열은 fail-open이다.

읽기·쓰기·진단 소비처를 전수 대장화한다.

- 생산 소비: `src/lumina_atomic.c`
- 출력/검증 언급: `src/lumina_main.c`
- Python 소비: `compare_tau_c_vs_python.py`, `debug_neutral_tau.py`, `validate_partition.py`, `validate_plasma.py`, `validate_tau_detail.py`, `validate_tau_impact.py`, `validate_reference.py`, `analyze_morphology_162212.py`, `verify_atomic_quarantine_identity.py`
- 생산자: `finalize_cmfgen_ref_npy.py`, `expand_atomic_data.py`, `export_tardis_reference.py`, prune 계열 및 드라이버 5기

폐합 조건:

- 정본 tau는 현재 line epoch와 50셸에 맞춰 생성되어야 한다.
- 행 수, 열 수, dtype, byte order, line-list 해시를 소비 전에 검증한다.
- missing, truncated, 30열, 다른 line epoch는 모두 FATAL이어야 한다.
- 경고 후 0 배열 재할당은 금지한다.
- “전 원소 0은 합법적인 초기 상태”라는 기존 검사 규칙은 K-FRESH가 입증되기 전까지 폐합 근거로 쓰지 않는다.

### 3.5 K-FRESH — 첫 소비 전 tau 재계산

현재 호출 순서에서는 첫 plasma refresh 전에 `opacity.tau_sobolev`를 읽는 경로가 있다. 따라서 50열로 고치기만 해도 첫 iteration이 stale 또는 all-zero tau를 사용할 수 있다.

폐합 조건:

- tau의 소유자를 덱과 solver 중 하나로 확정한다.
- solver 소유라면 첫 `cmfgen_assemble`·J-bar·transport 소비 전에 반드시 재계산한다.
- 덱 소유라면 파일은 실제 초기 플라스마 상태에서 계산된 값이어야 하고, epoch·line list·shell hash를 함께 검증한다.
- 첫 소비 지점에 generation counter 또는 계측을 두어 `computed_generation ≥ required_generation`을 증명한다.
- CPU call-chain harness로 첫 소비 전에 재계산됨을 증명할 수 있으면 GPU 런은 필요 없다.

음성 대조:

- 30열 파일
- 올바른 형상이지만 의도적인 nonzero sentinel을 넣은 stale 파일
- 다른 line-list epoch의 올바른 형상 파일
- missing/truncated NPY

어느 경우도 sentinel 또는 자동 0 대체값이 첫 소비자에 도달해서는 안 된다.

### 3.6 T-SEED — seed 의존성

0-C5의 `n_e` 처분 방법은 구조만 재사용할 수 있다. 즉 `seed → iteration trajectory → converged state → CMF oracle`를 비교한다. 그러나 기존 자료만으로 동일하게 기각할 수는 없다.

확인된 사실:

- 과거 기록의 `CMF T seed / 0.9·T_rad`는 최소 1.833, 중앙 2.264, 최대 5.302다.
- 현재 채택 캡처는 `T_RAD_COLOR_FIX=1`, `T_e/T_rad=1.0`이어서 과거 0.9 seed와 동일 조건이 아니다.
- 현재 seed 약 10,470 K에서 최종 `T_e/seed`는 최소 약 0.799, 중앙 0.947, 최대 2.027이다.
- 이는 solver가 seed에서 움직인다는 증거일 뿐, 최종해가 seed에 독립이라는 증거가 아니다.
- CMF와의 최종 T 비교도 일부 셸에서 충분히 가깝지 않다.

따라서 오프라인 분석 후에도 T seed는 판정런 없이 폐합할 수 없다. GPU 승인 후 한 배치에서 다음 세 lane을 동일 바이너리·입력·환경으로 수행한다.

1. 현 생산 seed
2. 과거 `0.9·T_rad` seed
3. CMF hydro T를 동일 셸에 매핑한 seed

사전등록 합격선:

- CMF 피복 셸 0–43의 최종 `T_e` lane 간 최대 상대차 `≤1%`, 중앙값 `≤0.2%`
- 최종 `n_e`도 최대 `≤1%`, 중앙값 `≤0.2%`
- 마지막 두 iteration의 `T_e` 최대 변화 `≤1%`
- NaN, 비정상 종료, lane별 clamp-hit 차이 없음
- 44–49셸은 별도 보고하며 0–43 통계에 섞지 않음

하나라도 실패하면 T seed는 폐합되지 않는다. 기준을 바꾸거나 후속 런을 자동 발주하지 않는다. 현재 존재하는 solver clamp의 정당성 자체는 층 1 I9이며, 이 실험에서는 seed만 분리하기 위해 고정하되 hit 수를 반드시 기록한다.

### 3.7 Z-INERT — 정확히 0인 원소

`atom_masses.csv`의 15원소 토폴로지 자체는 층 1이지만, 층 0 입력에서 C·O 등 정확히 0인 조성이 하류에서도 정확히 비활성이어야 한다는 것은 층 0 경계 계약이다.

현재 의심 경로:

- 상위 이온 단계의 `1e-300`
- dead NLTE species의 LTE-shape fallback
- Sobolev tau의 `1e-100`
- bound-free만 별도 문턱으로 차단되는 비대칭

폐합 조건:

- 입력 조성 0인 원소는 population, `n_ion`, line opacity, continuum opacity, emissivity, heating/cooling contribution이 모두 정확히 0이어야 한다.
- 활성 원소의 수치에는 변화가 없어야 한다.
- 0을 작은 양수로 바꾸는 floor와 fallback을 금지한다.
- 0원소 line이 transport 후보에 들어가지 않는다는 카운터 증거를 남긴다.

음성 대조로 inactive species에 phantom population을 주입하면 검증기가 반드시 실패해야 한다.

### 3.8 GEN-GUARD — 정본 덱 덮어쓰기 방지

[build_toy06_epoch.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_toy06_epoch.py:224)는 기본 출력이 정본 경로를 가리킬 가능성이 남아 있다.

폐합 조건:

- 출력 경로는 반드시 명시하도록 한다.
- 출력 생략, 입력=출력, 정본 경로, 정본의 symlink/realpath 별칭을 파일 개방 전에 거부한다.
- 성공 경로는 새 scratch 디렉터리만 허용한다.
- 정본 트리의 실행 전후 해시가 동일해야 한다.

## 4. 사전등록 게이트

모든 판정은 실행 전에 `L0_PREREG.md` 한 건에 동결한다.

각 항목에는 다음을 기록한다.

- 계약 ID와 대상 파일 목록
- 소스·입력·line list·설정·바이너리 SHA-256
- 실행 노드와 스케줄러 job ID
- 계산할 지표와 합격선
- 음성 대조와 기대 종료코드
- 제외 셸·제외 파장대와 이유
- 실패 시 다음 행동: `UNRESOLVED` 기록만 허용
- 결과를 본 뒤 기준·셸·파장대를 변경하지 않는다는 선언

GPU T-SEED는 이 문서가 동결되고 user가 승인한 뒤 한 번만 제출한다.

## 5. 음성 대조 총괄

다음 8종이 모두 실제 실패해야 한다.

1. D: 구 판독기 19개 fixture
2. C2: 30열 조성, 기존 출력 경로, 입력/출력 동일 경로
3. H: exact-zero와 의도적인 off-sum 조성
4. K-SHAPE: missing, truncated, 30열, wrong-line-epoch NPY
5. K-FRESH: 올바른 형상의 stale sentinel tau
6. T-SEED: 메타데이터가 다른 lane 또는 미수렴 trajectory
7. Z-INERT: 0원소 phantom population
8. GEN-GUARD: 정본 및 정본 별칭 출력

음성 대조가 검증기 자체 오류로 실패한 것과 계약 위반을 정확히 검출한 것을 구분해 로그에 남긴다.

## 6. 범위 밖 선언

- F 격자 범위는 층 1 I6/I7이다. 조성 덱 형상이 아니라 계산영역·외삽·공간 피복 문제다. 다만 기존 formation map에서 무피복 45–49셸 기여가 450–918 Å에서 약 5.138%, 1490–1650 Å에서 약 1.246%로 나타나므로 “작아서 무시”로 폐기해서는 안 된다. 이벤트 로그로 오프라인 측정 가능하지만 이는 마지막 방출 셸 점유율이지 셸 제거의 인과효과가 아니다.
- G 준위 절단·R3c collision fallback·I18은 원자데이터와 NLTE 모델 계약이므로 층 1이다. 운전석의 기존 층 0 범위밖 표에서 층 1 항목으로 정정한다.
- I의 15원소 토폴로지·성능·재색인은 층 1이다. 단, 0 조성의 정확한 비활성은 Z-INERT로 층 0에 남긴다.
- J 조성 출처 통일은 0-C8에서 효과 상한이 약 1.7%로 정밀화에 강등되었다. 층 0 폐합에 필요하지 않으며, 출처가 동일하다고 주장해서도 안 된다.
- D·C1·C2 커밋과 생산 설치·런은 배포 필드다. 검증 폐합과 분리한다.
- 외부 deposition 입력이 내부 Bateman 계산으로 다시 덮이거나 이중 붕괴하는 경로는 층 1 I10으로 이관하며 `해결 추정`으로 지우지 않는다.

## 7. 운전석 실행 명령

grammar-debug에서는 소규모 검사와 제출만 수행한다. `/usr/bin/time`은 사용하지 않는다.

### 7.1 현 상태 동결

```bash
set -euo pipefail
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd "$REPO"

git status --short
sha256sum \
  src/lumina_atomic.c \
  scripts/expand_atomic_data.py \
  scripts/deck_quarantine_driver.py \
  scripts/deck_regen_fullcov_driver.py \
  scripts/deck_regen_r1_vintage_driver.py \
  scripts/deck_regen_r4_ftos_driver.py \
  scripts/deck_regen_r4_offcontrol_driver.py \
  > /tmp/l0-close-source.sha256

python3 scripts/run_composition_c_gate.py
```

### 7.2 D 격리 전체 빌드

lageunha 또는 Slurm에서 수행한다. 예시는 lageunha이며 루트 바이너리를 건드리지 않는다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
BUILD=$(mktemp -d /gpfs/kjhan/lumina_runner2/l0-d-build.XXXXXX)

cp -a "$REPO/src" "$REPO/Makefile" "$BUILD/"
mkdir -p "$BUILD/scripts"
cp -a "$REPO"/scripts/composition_* "$BUILD/scripts/" 2>/dev/null || true

cd "$BUILD"
export OMP_NUM_THREADS=60
make -B cuda 2>&1 | tee build.log
sha256sum lumina_cuda src/lumina_atomic.c > build.sha256
printf '%s\n' "$BUILD"
EOF
```

빌드 디렉터리를 보존하고, 그 소스에 대해 D 19/19 및 구 판독기 0/19를 실행한다. 검증 스크립트가 루트 소스를 다시 참조하지 않는지 명령행과 로그로 확인한다.

### 7.3 C2 격리 재생성

```bash
set -euo pipefail
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
STAGE=$(mktemp -d /gpfs/kjhan/lumina_runner2/l0-c2.XXXXXX)

mkdir -p "$STAGE"/{data,logs,work}
cp -a "$REPO/scripts" "$STAGE/"
ln -s "$REPO/data/atomic" "$STAGE/data/atomic"
cp -al \
  "$REPO/data/tardis_reference_toy06_19p48d_sivcaiv" \
  "$STAGE/data/"

sha256sum "$REPO"/data/tardis_reference_toy06_19p48d*/abundances.csv \
  > "$STAGE/preexisting.sha256"

J1=$(sbatch --parsable --cpus-per-task=60 \
  --export=ALL,REPO_ROOT="$STAGE" \
  "$STAGE/scripts/sbatch_deck_regen_fullcov.sh")

J2=$(sbatch --parsable --dependency="afterok:$J1" --cpus-per-task=60 \
  --export=ALL,REPO_ROOT="$STAGE" \
  "$STAGE/scripts/sbatch_deck_regen_r1_vintage.sh")

J3=$(sbatch --parsable --dependency="afterok:$J2" --cpus-per-task=60 \
  --export=ALL,REPO_ROOT="$STAGE",R4_OFF_DIR="$STAGE/work/r4off-r4" \
  "$STAGE/scripts/sbatch_deck_regen_r4_ftos.sh")

J4=$(sbatch --parsable --dependency="afterok:$J3" --cpus-per-task=60 \
  --export=ALL,REPO_ROOT="$STAGE",R4_OFF_DIR="$STAGE/work/r4off-q" \
  "$STAGE/scripts/sbatch_atomic_quarantine.sh")

printf 'STAGE=%s\nJOBS=%s,%s,%s,%s\n' "$STAGE" "$J1" "$J2" "$J3" "$J4"
```

실제 wrapper 이름이 다르면 추정 실행하지 말고 `rg --files scripts | rg 'sbatch.*(regen|quarantine)'`로 확인해 대장 명령을 정정한다. `R4_OFF_DIR`을 줬는데도 `SLURM_TMPDIR`을 요구하면 C2 실행기 계약 실패로 기록한다.

### 7.4 H·F 오프라인 계산

H는 전용 검증기가 준비된 뒤 lageunha 또는 Slurm CPU에서 `OMP_NUM_THREADS=60`으로 수행한다. F도 GPU 없이 기존 이벤트 로그 파싱으로 수행할 수 있으나 층 1 증거로만 저장한다. 존재하지 않는 스크립트명을 임의 실행하지 말고, 검증기 경로·해시가 대장에 등록된 뒤 실행한다.

### 7.5 T-SEED

GPU 명령은 본 발주서에서 실행하지 않는다. 사전등록 문서와 세 lane manifest를 운전석이 검수하고 user의 명시적 승인을 받은 뒤 단일 Slurm batch로 제출한다.

## 8. 운전석 검수 항목

운전석은 다음을 반박 우선순위로 검수한다.

1. 14개 대장행의 열거와 산술이 실제 미결을 빠짐없이 나타내는가.
2. K-SHAPE와 K-FRESH가 별도 계약·별도 증거로 남았는가.
3. I의 토폴로지는 층 1, 0원소 불활성은 층 0으로 분리됐는가.
4. G가 층 1 원자데이터로 정정됐는가.
5. C2 드라이버 5기가 모두 실제로 실행됐는가.
6. C2 스테이지가 hard-link나 symlink를 통해 생산 덱을 변경하지 않았는가.
7. D 전체 빌드가 미커밋 D 소스를 포함했고, 기존 바이너리를 재검증한 것이 아닌가.
8. H의 floor와 재규격화 효과가 분리되어 있는가.
9. K에서 30열 경고 후 0 대체가 제거되고 첫 소비 전 신선도까지 증명됐는가.
10. T 비교가 과거 0.9 seed와 현재 color-fix seed를 혼동하지 않는가.
11. F의 formation share를 counterfactual spectral effect로 과장하지 않았는가.
12. 모든 증거에 해시, 노드, job ID, 종료코드가 있는가.
13. 음성 대조가 기대한 계약 위반 때문에 실패했는가.
14. 검증 폐합과 커밋·설치·생산 런을 별도 필드로 기록했는가.
15. `/usr/bin/time`을 호출한 명령이 없는가.
16. GPU T-SEED가 user 승인 전에 제출되지 않았는가.

## 9. 이 발주가 놓칠 수 있는 무증상 실패 경로

- 스테이지의 hard-link가 생산 파일과 inode를 공유해 출력이 원본을 바꾸는 경우
- 복사한 드라이버가 내부 hard-coded `ROOT`로 원 저장소를 다시 가리키는 경우
- 기존 출력 거부를 성공 실행으로 오인하는 경우
- Slurm wrapper가 `REPO_ROOT` 대신 제출 시점 경로를 사용하는 경우
- `R4_OFF_DIR`이 있는데도 `SLURM_TMPDIR` 선검사로 실패하는 경우
- 파이프라인에서 `tee`가 앞 명령의 실패를 가리는 경우
- 빌드는 성공했으나 HEAD 소스로 빌드되어 미커밋 D가 빠진 경우
- 검증 바이너리는 맞지만 생산에 설치되지 않은 사실을 누락하는 경우
- NPY missing 경로에서 행·열 변수가 초기화되지 않은 채 비교되는 경우
- 행·열은 맞지만 line-list epoch가 다른 tau가 통과하는 경우
- 30열 tau가 경고와 0 재할당으로 가려지고 첫 iteration이 그대로 진행되는 경우
- 정확히 0인 조성이 NLTE fallback이나 `1e-100` tau로 다시 살아나는 경우
- H 재규격화가 원시 합 오차와 floor 주입량을 함께 숨기는 경우
- 원소 Ni/Co/Fe와 `X56Ni`가 서로 다른 floor·붕괴 규칙을 타는 경우
- T seed 비교가 서로 다른 color correction, 고정 T, 외삽 셸 또는 바이너리를 섞는 경우
- 미수렴 최종 iteration을 seed 민감성으로 오판하는 경우
- F 이벤트 로그의 cap·누락 hash·파장계 변환 때문에 outer-shell 점유율이 편향되는 경우
- 외부 deposition profile을 읽은 뒤 내부 계산이 덮어써 이중 붕괴하는 경우
- 검증 결과는 통과했지만 정본 덱과 생산 바이너리는 전혀 바뀌지 않았는데 “생산 적용 완료”로 기록하는 경우

최종 폐합 판정은 1–8번 검증 계약이 모두 `PASS`, 9–12번 범위 판정이 대장에 확정되고, 음성 대조가 전부 기대대로 실패했을 때만 `L0_VALIDATION_CLOSED=yes`로 기록한다. 13–14번은 user 승인 전까지 `L0_PRODUCTION_DEPLOYED=no`로 유지한다.
---

## 10. 운전석 검수 결과 (2026-08-04, 자율 진행)

**판정: 조건부 수용.** 실행 명령 1건 반박, 신규 주장 2건 실측 확인, 정정 3건.

### 확인

| 항목 | 실측 |
|---|---|
| **Z-INERT 의심 실물** | `src/lumina_cmfgen.c:838` → `if (!(tau_pop > 1e-100)) tau_pop = 1e-100;` — **정확한 0을 비영으로 바꾸는 clamp**. 규약 판별식("정확해가 위반 가능한 가드")에 정확히 걸린다 |
| Z-INERT 무죄 매치 구분 | `1e-300` 매치 다수는 `fabs(S)+1e-300`(0분모 가드)·`-log(urand()+1e-300)`(log 0 방지)로 **성격이 다르다**. 정확해가 0인 자리를 덮는 것은 `:838` 하나 |
| 14개 대장행 산술 | 열거 확인: 검증 8(C2-EXEC·D-BUILD·H-TRANSFORM·K-SHAPE·K-FRESH·T-SEED·Z-INERT·GEN-GUARD) + 범위판정 4(F·G·I·J) + 배포 2 = **14** |
| 운전석 누락 2건 | **Z-INERT·GEN-GUARD는 운전석 10건 목록에 없었다.** Codex 신규 발견 |
| K 분할 | K-SHAPE(형상 계약)와 K-FRESH(첫 소비 전 신선도)는 **독립 계약이 맞다** — 형상만 고쳐도 첫 iteration이 stale/0 tau를 쓸 수 있다 |

### ★반박 — 실행 명령의 sbatch wrapper 이름 (§7.3)

```
ABSENT  scripts/sbatch_deck_regen_r4_ftos.sh   →  실제 scripts/sbatch_deck_r4_ftos.sh
ABSENT  scripts/sbatch_atomic_quarantine.sh    →  실제 scripts/sbatch_deck_atomic_quarantine.sh
EXISTS  scripts/sbatch_deck_regen_fullcov.sh
EXISTS  scripts/sbatch_deck_regen_r1_vintage.sh
```
⚠ `sbatch_deck_r1_vintage.sh` 와 `sbatch_deck_regen_r1_vintage.sh` 가 **둘 다 존재**한다.
C2-EXEC 전에 어느 것이 드라이버 5기 경로인지 확정해야 한다.

발주서 §7.3이 스스로 경고한 자리다("실제 wrapper 이름이 다르면 추정 실행하지 말고
`rg`로 확인해 대장 명령을 정정한다"). 운전석이 실행 담당이므로 반려 대신 정정한다.

### 운전석 추가 정정

1. §7.3의 `sbatch` 가 어느 클러스터인지 미명시. **LUMINA=syn / grammar 는 별개**이며
   `grammar exclude` 를 syn sbatch 에 붙이면 제출이 거부된다(규약). 덱 재생성은 CPU
   작업이므로 **grammar slurm 또는 lageunha**. 운전석이 확정한다.
2. §7.1의 `scripts/expand_atomic_data.py` — 실제 파일명은
   `scripts/expand_atomic_data_cmfgen.py` 일 가능성. 해시 대상 목록 실행 전 확인 필요.
3. 배포 2건에 대해 **user 승인이 이미 내려왔다**(2026-08-04:
   *"0층이 완료되면 커밋하고 푸시하도록"*). 단 조건부이므로
   `L0_VALIDATION_CLOSED=yes` 확정 후에만 집행한다.

### 검수 못 한 것 (미결로 남김)

- §3.3 H의 예비 실측 수치(원시합 `0.999992…1.000012`·floor 주입 `1.416e-7`·
  `max|ΔX| 5.15e-6` 등)를 운전석이 **독립 재현하지 않았다.** H 검증기 구현 후 대조한다.
- §3.6 T-SEED의 `T_e/seed min 0.799 median 0.947 max 2.027` 도 미재현.
- §6 F의 `450–918Å 5.138% / 1490–1650Å 1.246%` formation share 도 미재현.
- §3.5 K-FRESH의 "첫 plasma refresh 전에 `opacity.tau_sobolev` 를 읽는 경로가 있다"는
  **소스로 폐합 필요.** 구현 발주에 포함한다.

---

## 11. 운전석 실행 준비 실측 (2026-08-04 자율, C2-EXEC 선행)

### ★C2-EXEC 차단 결함 확인 — 발주서 §3.1이 지목한 그 자리

```bash
scripts/sbatch_deck_r4_ftos.sh:19
  : "${SLURM_TMPDIR:?SLURM_TMPDIR is required for the ephemeral OFF-control deck}"
scripts/sbatch_deck_r4_ftos.sh:20
  OFF_DECK="${R4_OFF_DIR:-$SLURM_TMPDIR/r4_ftos_offcontrol}"
```
**19행의 `:?` 확장이 20행보다 먼저 죽는다.** `R4_OFF_DIR`을 줘도 `SLURM_TMPDIR`
부재로 중단된다. 발주서 §3.1의 *"R4_OFF_DIR이 명시된 경우 SLURM_TMPDIR 부재 때문에
중단하지 않아야 한다"* 계약 **위반 확정**.

대조: `scripts/sbatch_deck_atomic_quarantine.sh:21`에는
*"job 400018 showed that SLURM_TMPDIR is not universal. R4_OFF_DIR is the…"* 주석과
함께 수정이 들어 있다. ⟹ **한쪽만 고쳐졌고 r4_ftos 쪽이 누락됐다.**

### wrapper 중복 확인

| 파일 | 크기 | 드라이버 | 출력 덱 |
|---|---|---|---|
| `sbatch_deck_r1_vintage.sh` | 2021 B | `deck_regen_r1_vintage_driver.py` | `..._links` |
| `sbatch_deck_regen_r1_vintage.sh` | 2018 B | `deck_regen_r1_vintage_driver.py` | `..._links` |

**주석 3바이트만 다른 중복본**이다. 기능 동일. C2-EXEC는 하나만 쓰고 다른 하나의
처분(삭제 또는 심링크)을 별건으로 남긴다.

### 발주서 §7.3 실행 명령 정정 (운전석 담당)

```
발주서 표기                          →  실제
sbatch_deck_regen_r4_ftos.sh         →  sbatch_deck_r4_ftos.sh        (단 위 결함 선수리 필요)
sbatch_atomic_quarantine.sh          →  sbatch_deck_atomic_quarantine.sh
sbatch_deck_regen_fullcov.sh         →  (동일, 실재)
sbatch_deck_regen_r1_vintage.sh      →  (동일, 실재. 중복본 존재)
```
`scripts/expand_atomic_data.py` → 실제 `scripts/expand_atomic_data_cmfgen.py` (§7.1 해시 목록).

⟹ **C2-EXEC 착수 전 선행 수리 1건**: `sbatch_deck_r4_ftos.sh`의 `SLURM_TMPDIR` 선검사를
`R4_OFF_DIR` 우선 순서로 바꾼다. 이는 실행기 계약 수리이며 **C2-EXEC의 일부**다.

---

## 12. C2-EXEC 실행 준비 실측 2 (운전석, 2026-08-04 07:36)

### sbatch 지시자 실측 — 발주서 §7.3 정정 4건

```
                              --partition  --exclude  출력경로
sbatch_deck_regen_fullcov.sh      없음       없음     logs/%x_%j.out          ← 저장소 상대
sbatch_deck_regen_r1_vintage.sh   없음       없음     /gpfs/.../slurm/%x_%j.out
sbatch_deck_r4_ftos.sh            없음       없음     /gpfs/.../slurm/%x_%j.out
sbatch_deck_atomic_quarantine.sh  없음       없음     /gpfs/.../slurm/%x_%j.out
공통: nodes=1 ntasks=1 cpus-per-task=8 mem=64G
--time: 8h / 8h / 16h / 20h  (의존 체인이면 예산 합 52h)
```

| # | 정정 |
|---|---|
| 1 | **`--partition` 미지정** — 제출 클러스터 기본 파티션으로 간다. 명시 필요 |
| 2 | **`--exclude` 없음** — 규약상 grammar 제출은 `--exclude=grammar072,grammar078,grammar080` 필수(072 /gpfs 기록불가·078/080 /home 마운트 불량) |
| 3 | **`fullcov`만 출력이 `logs/%x_%j.out`**(저장소 상대) — GPFS scratch 규약 위반이고 제출 cwd 의존. 나머지 3개와 불일치 |
| 4 | `--time` 합 52h — 실제 소요는 훨씬 짧다(`_fullcov` 덱 mtime 08-03 15:30→15:33). slurm 상한일 뿐 |

### 클러스터 선정 (운전석 결정)

```
grammar normal   idle 33 / alloc 75 / down grammar035 / drain grammar060
내 큐            없음 (대기 중인 3wm_*·dmo_* 는 user 의 다른 작업)
lageunha         load 1.76, 0 users   ← 여유
```
규약: *"대형 오프라인은 lageunha 직접 투척(OMP=60), 투척 전 uptime 으로 부하 확인"*.
부하 확인 결과 비어 있으므로 **C2-EXEC = lageunha 수동**. sbatch 스크립트의 `#SBATCH`
행은 bash 주석이므로 그대로 실행 가능하고, `REPO_ROOT`·`R4_OFF_DIR` 는 이미 지원된다.
`OMP_NUM_THREADS` 는 `${SLURM_CPUS_PER_TASK:-8}` 이므로 **명시적으로 60 을 export** 한다.

### 착수 조건

**Z-INERT(src 편집)가 끝난 뒤.** C2-EXEC 은 덱 디렉터리를 새로 만드는 트리 변조이므로
규약상 동시 진행 금지(`src-편집/트리-변조 태스크는 한 번에 1개`).
