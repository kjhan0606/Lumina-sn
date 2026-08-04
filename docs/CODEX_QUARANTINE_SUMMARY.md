구현 스크립트와 문서를 준비했고 fixture 자기검사를 통과했습니다. GPU·모델·덱 생성·commit은 실행하지 않았으며 새 덱 경로도 아직 존재하지 않습니다. `src/`와 회귀 대장은 기존 변경을 그대로 보존했습니다.

### 구현 위치

- 창고 생성·32이온 매니페스트·사전등록: [deck_quarantine_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_quarantine_driver.py:189)
- 봉인 SHA-256·행 수 검증: [seal_atomic_quarantine.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/seal_atomic_quarantine.py:21)
- 로더 root-only/누출 fatal 계약: [atomic_quarantine_contract.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/atomic_quarantine_contract.py:81)
- 이온 양방향 게이트: [verify_atomic_quarantine_identity.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_atomic_quarantine_identity.py:158)
- 준위·R4 membership 게이트: [verify_atomic_quarantine_identity.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_atomic_quarantine_identity.py:277)
- 선 multiset·값 게이트: [verify_atomic_quarantine_identity.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_atomic_quarantine_identity.py:348)
- NPY·offset·macro·binary 고아 검사: [verify_atomic_quarantine_identity.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_atomic_quarantine_identity.py:405)
- 배치 및 기존 R1/R4 보존: [sbatch_deck_atomic_quarantine.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_deck_atomic_quarantine.sh:74)
- 전체 운전·복원 문서: [CODEX_ATOMIC_QUARANTINE.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_ATOMIC_QUARANTINE.md:7)

생성될 구조는 다음과 같습니다.

```text
..._sivcaiv_active/
  active_ions.csv
  <27-ion active atomic inputs>
  quarantine/
    DO_NOT_LOAD                  # mode 000 sentinel
    manifest.json               # 정본
    manifest.csv                # 32-ion 색인
    source_deck_snapshot/       # 원본 덱 byte-preserving snapshot
```

사전검사 결과는 `59 original / 27 active / 32 quarantine`, `(a,b,c)=(6,0,26)`, 격리 FL=`10,607`로 일치했습니다. 활성 NF/N_SL은 `MODEL_SPEC`와 27/27 exact prefix이며 논리적 최대 N=240입니다.

매니페스트에는 물리 변화와 예상 방향도 기록됩니다: C/O/Mg/Al/Sc/Ti/V/Cr/Mn의 양의-abundance opacity가 제거되므로 그 기여분만큼 더 투명해져야 하며, 반대 방향이면 기전 이해를 기각합니다.

### 음성 대조 실제 출력

```text
NEGATIVE 1 hidden extra ion: FAIL
  FAIL_EXTRA_ION Z=6,ion0=0
  [ATOMIC-ACTIVE-SET-LEAK] loaded quarantined ion Z=6,ion0=0

NEGATIVE 2 missing active ion: FAIL
  FAIL_MISSING_ION Z=16,ion0=1
  FAIL_ARCHIVE_PARTITION_MISSING Z=16,ion0=1

NEGATIVE 3 loader reads quarantine: FAIL
  [ATOMIC-ACTIVE-SET-LEAK] refused quarantine consumption: quarantine/DO_NOT_LOAD

FIXTURE SELF-CHECK: PASS (all three negative controls failed closed)
```

### 운전석 sbatch

`SLURM_TMPDIR`에 의존하지 않는 권장 명령입니다.

```bash
R4_OFF_DIR=/gpfs/kjhan/lumina_runner2/work/r4_off_atomic_quarantine_$(date +%Y%m%d_%H%M%S)

sbatch --export=ALL,REPO_ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn,R4_OFF_DIR="$R4_OFF_DIR" \
  scripts/sbatch_deck_atomic_quarantine.sh
```

로그는 `/gpfs/kjhan/lumina_runner2/slurm/`에 기록됩니다. 양방향 검증기가 마지막 명령이며 `pipefail`로 검증기 exit code가 job 결과가 됩니다.

### 복원

현재 덱을 수정하지 않고 새 덱으로 승격합니다.

1. 새 CMFGEN `MODEL_SPEC`·links·조성에 복원 이온이 포함됐는지 확인
2. archive SHA-256 전수 확인
3. 관련 준위·선·σ·Υ·이온화 경계·macro/reference를 함께 승격
4. 전역 ID·offset·NPY·`active_ions.csv` 재생성
5. 새 매니페스트에 `restored` 이벤트 append
6. R1/R4·양방향·누출 게이트를 모두 통과한 새 덱만 활성화

현재 기준으로 32이온을 복원하면 `FAIL_EXTRA_ION`이 정상 안전장치입니다.

### 남은 UNRESOLVED

- 실제 덱 생성, 전체 게이트 결과, 모델 전후 회귀 대장 비교는 운전석 실행 후 필요
- 선 `f_lu/A_ul/λ`의 기존 decimal serialization·finalizer 재유도는 exact-value 게이트에서 실패할 수 있음
- σ 전 점·Υ 전 항목의 CMFGEN 의미 값 비교는 R5 별건
- abundance 30열 대 `n_shells=50` 형상 결함은 미해결
- allocator는 손대지 않아 GPU 할당은 줄지 않음