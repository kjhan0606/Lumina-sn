## 결과

정정 발주 기준으로 준비를 완료했습니다. 새 덱은 지시대로 생성하지 않았고, GPU·모델·commit도 실행하지 않았습니다. `src/` 변경은 [패치 파일](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/standart_abundance_path_audit.patch)로만 제공하며 실제 트리에 적용하지 않았습니다.

### 30/50 결함 확증

결함은 사실입니다. 다만 경로별로 구분됩니다.

- 50열: `toy06_19p48d`, `toy06_19p48d_sivcaiv`
- 30열/50셸 결함: `sivcaiv_ftos`, `sivcaiv_fullcov`, `sivcaiv_links`

신규 덱의 기반인 `sivcaiv_ftos`도 결함 대상입니다.

현재 로더는 [lumina_atomic.c:819](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:819)에서 50셸을 0으로 할당한 뒤 [lumina_atomic.c:837](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:837)에서 CSV 폭과 무관하게 `strtod`를 50회 호출합니다. libc 실측:

```text
SHORT_ROW_CONFIRM nonzero=30 implicit_zero=20 pointer_stalls=20
FULL_ROW_CONTROL nonzero=50 implicit_zero=0 pointer_stalls=0
```

즉 결함 덱의 셸 30–49는 모든 원소가 0입니다. 전체 Lumina 로더 덤프는 금지된 실행 범위라 운전석용 [dump-only 스크립트](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_abundance_dump_only.sh)를 준비했습니다.

### 로더 → 물리 경로

핵심 경로는 다음과 같습니다.

1. 진입: [lumina_main.c:119](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_main.c:119), CUDA는 [lumina_cuda.cu:6888](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6888)
2. CSV → `atom->abundances`: [lumina_atomic.c:819](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:819)
3. 질량분율 → `n_element`: [lumina_plasma.c:2262](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2262), 반복 전자밀도 경로는 [lumina_plasma.c:2460](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2460)
4. plasma master: [lumina_plasma.c:6340](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6340)
5. Sobolev 불투명도: [lumina_plasma.c:2582](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2582)
6. BF 불투명도: [lumina_plasma.c:6878](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6878)
7. NLTE topology/projection: [lumina_plasma.c:14256](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14256)
8. macro-atom branching: [lumina_plasma.c:3498](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:3498)

중간 abundance 재정규화·클램프·기본값 대체는 없습니다. 따라서 입력 합이 1이 아니면 Lumina는 그대로 계산합니다.

추가로 두 경로 결함도 확인했습니다.

- 복사된 `tau_sobolev.npy`가 첫 반복에서 이전 조성 값으로 사용됨
- 외부 deposition을 읽어도 [lumina_cuda.cu:10596](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:10596)의 내부 붕괴 계산이 이후 덮어써 Ni/Co를 이중 붕괴시킴

패치에는 `LUMINA_REBUILD_INITIAL_PLASMA=1`과 외부 deposition 50셸 완전성·잠금 처리를 포함했습니다.

### abundance=0 하류

현재 무패치 상태에서는 0 종족이 완전히 제외되지 않습니다.

- 이온/NLTE/macro-atom 슬롯은 15원소 topology 기준으로 계속 할당됩니다.
- 상위 이온은 `1e-300` floor를 받을 수 있습니다.
- `LUMINA_NLTE_SKIP_DEAD` fallback은 `n_total==0`에서 LTE 모양 population을 남길 수 있습니다.
- BF는 [lumina_plasma.c:7151](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7151)의 `n_ion<1e-30`으로 차단됩니다.
- 선 불투명도는 일반 `1e-100` floor 때문에 exact zero가 아닙니다.
- `SIMUL_CAP_TOPION` 이전에 [lumina_plasma.c:10045](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10045)에서 `nel<=0` 사다리를 건너뜁니다.
- abundance 또는 `n_element`로 나누는 무방비 생산 경로는 찾지 못했습니다.

패치는 0 종족의 이온·NLTE population과 선 opacity/source를 exact zero로 만들고 카운터를 추가합니다. 실제 helper를 임시 패치 복사본에서 실행한 음성 대조 결과:

```text
[ABUNDANCE-ZERO-NLTE] excluded_ion_shell_slots=3 cleared_level_cells=5 residual_nonzero_before_clear=10 policy=EXACT_ZERO
ZERO_NLTE_FIXTURE excluded_slots=3 cleared_levels=5 residuals=10 PASS
```

활성 셸 sentinel은 그대로 유지되어 blanket wipe가 아님도 확인했습니다.

### 매핑·질량 보존

구현은 [standart_toy06_composition.py:217](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/standart_toy06_composition.py:217)에 있습니다.

- StaNdaRT 중심: 100–40,300 km/s
- 실측 source cell edge: 0–40,400 km/s
- Lumina 50셸: 3,900–40,300 km/s
- 매핑: `dmass × 교차구간 (v³_hi-v³_lo) / source-cell (v³_hi-v³_lo)`
- 같은 질량으로 각 종족·동위원소 numerator 계산
- 범위 밖 외삽·clamp·대체값 없음; coverage 부족은 FAIL

StaNdaRT 출력 반올림 오차는 native source cell에서 명시적으로 6원소 재정규화합니다. factor 범위는 `0.999993000049–1.000008800077`, 최대 보정 `8.80e-6`입니다. 매핑 후:

- 50셸 coverage: 전부 1
- 최대 `|ΣX−1|`: `2.22e-16`
- 심부 Ni/Co/Fe: `0.1083233 / 0.7937132 / 0.0979635`
- 외곽 Si/S/Ca: `0.55 / 0.35 / 0.10`

CMFGEN 2차 정본은 36,000 km/s까지만 있으므로 완전히 덮이는 공통 셸만 비교하고, 나머지는 `secondary_outside=[...]`로 기록하며 외삽하지 않습니다.

### 게이트와 산출물

7개 게이트는 [검증기](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_toy06_standart_deck.py:67)의 `gate1`–`gate7`에 구현했습니다. 합 허용치는 `2e-12`, CMF 교차확인은 `5e-5`, 질량 보존은 `2e-12 Msun`입니다. Gate 7은 immutable byte 비교 후 기존 R1/R4 검증기를 호출합니다.

주요 산출물:

- [덱 생성기](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_toy06_standart_deck.py)
- [7-gate 검증기](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_toy06_standart_deck.py)
- [sbatch](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_toy06_standart_deck.sh)
- [코드 경로 감사서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STANDART_COMPOSITION_PATH_AUDIT.md)
- [src 전용 패치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/standart_abundance_path_audit.patch)
- [fixture 실행기](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/selftest_toy06_standart.sh)

`REPO_ROOT`, `SLURM_TMPDIR` 부재 fallback, 지정 로그 경로, 검증기 마지막 실행과 `pipefail`을 반영했습니다.

### 운전석 명령

먼저 자기검사:

```bash
REPO_ROOT="$PWD" scripts/selftest_toy06_standart.sh
```

덱 생성·전체 검증 투척:

```bash
mkdir -p /gpfs/kjhan/lumina_runner2/slurm

sbatch \
  --export=ALL,REPO_ROOT="$PWD",CMFGEN_RUN=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  scripts/sbatch_toy06_standart_deck.sh
```

패치를 별도 작업복사본에 적용·CPU 빌드한 뒤 loader dump만 실행:

```bash
git apply --check patches/standart_abundance_path_audit.patch
git apply patches/standart_abundance_path_audit.patch
make -j4 lumina

REPO_ROOT="$PWD" \
LUMINA_BIN="$PWD/lumina" \
scripts/run_abundance_dump_only.sh \
  /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_standart
```

물리 실행 시에는 패치된 바이너리와 함께 `LUMINA_REBUILD_INITIAL_PLASMA=1`, `LUMINA_GAMMA_DEP=1`, `LUMINA_DEPOSITION_FILE=<deck>/deposition_cmfgen.csv`가 필수입니다.

### UNRESOLVED

- 외부 CMFGEN run이 이 workspace에 없어 실 SN_HYDRO Gate 4와 기존 R1/R4는 미실행입니다.
- production Lumina dump-only, CUDA 컴파일, GPU/모델 경로는 지시대로 미실행입니다.
- 패치는 임시 CPU 전체 빌드와 fixture를 통과했지만 실제 `src/`에는 미적용입니다.
- 9개 부재 원소의 atomic/macro topology 슬롯은 유지됩니다. 패치는 물리적 population/opacity만 exact zero로 만들며 reindex는 하지 않습니다.
- 새 덱 디렉터리는 덱 생성 금지에 따라 현재 존재하지 않습니다. 기존 5개 덱과 `validation/regression_ledger/`는 건드리지 않았습니다.