구현과 로컬 검증을 완료했습니다.

- K-SHAPE: 두 NPY를 행·열·`<f8`·little endian·C-order·line-list SHA-256 계약으로 fail-closed 처리했습니다. 기존 경고 후 0 배열 재할당은 제거했습니다.
- K-FRESH: tau 소유자를 solver로 확정하고 CPU/CUDA 첫 소비 전에 강제 재계산하도록 했습니다. CPU harness에서 sentinel이 첫 소비자에 도달하지 않았고 generation `2/2`를 확인했습니다.
- 실행기: `R4_OFF_DIR`이 `SLURM_TMPDIR`보다 우선합니다.
- 음성대조: 30열, 잘못된 행 수, stale sentinel, 다른 line epoch, 양쪽 NPY의 missing/truncated/dtype/byte-order가 모두 기대대로 처리됐습니다.
- CPU 및 격리 CUDA 빌드 성공, 조성 회귀 9/9, `git diff --check` 통과.
- 덱·`/gpfs`·Z-INERT는 건드리지 않았고 commit/push도 하지 않았습니다.

전체 변경 27개, 실제 출력, 구현 줄번호, 소비처 영향, 운전석 명령과 잔여 위험은 [CODEX_IMPL_L0_K.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_L0_K.md)에 기록했습니다.

중요한 배포 차단점은 현재 production 후보 덱 4개 모두 `kshape_contract.txt`가 없다는 것입니다. C2 재생성 전에 새 binary를 배포하면 의도적으로 FATAL합니다.

또한 R1 wrapper 두 개는 현재 실측상 단순 3바이트 주석 중복이 아니며 기능 차이가 있습니다. 삭제하지 않았고, 안전한 정본 선정 권고를 보고서에 남겼습니다.