Z-INERT 계약 구현과 검증을 완료했습니다.

- 비활성 9원소의 line 353,770개: nonzero tau 0, transport 후보 0
- 활성 line 2,211,572개: legacy tau와 비트 차이 0
- 활성 population: ground 및 기존 `1e-300` 상위 단계 모두 비트 동일
- phantom population 음성대조: rc=1로 정상 실패
- CPU 전체 빌드 및 CUDA 전체 빌드 성공
- K-SHAPE/K-FRESH 회귀 통과
- D FATAL 16종 + WARN 2종 + 정본 control: 19/19 통과
- `git diff --check` 통과
- 덱·`/gpfs`·`lumina_atomic.c`·generation counter는 변경하지 않았습니다.
- commit/push하지 않았습니다.

구현·카운터 형식·수리/비수리 위치·실제 출력·운전석 명령·잔여 위험은 [CODEX_IMPL_L0_ZINERT.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_L0_ZINERT.md)에 기록했습니다.

주 검증 명령:

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
bash scripts/run_zinert_selftest.sh
```

남은 위험은 CUDA production packet 런을 수행하지 않았다는 점입니다. NVCC 전체 빌드는 통과했으며, 다음 운전 런에서 `LUMINA_ZINERT_AUDIT=1`로 `post-nlte-gpu` 카운터를 확인할 수 있습니다.