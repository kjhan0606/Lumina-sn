준비를 완료했습니다. 본 연산과 Slurm 제출, 커밋은 수행하지 않았습니다.

- [s31_ka2_oracle_hp.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/s31_ka2_oracle_hp.py): 전 단계 mpmath 80-digit, dense Nyström operator, 로그 특이점 subtraction, GMRES solve, 체크포인트 재개, 2048/4096 자기검증
- [s31_ka2_judge.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/s31_ka2_judge.py): 원문 rung10/11 문턱 재판정 및 binary64 oracle 비교
- [sbatch_s31_oracle_hp.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_s31_oracle_hp.sh): grammar CPU, 32 CPU, 32 GiB, `03:00:00`; 미제출
- [CODEX_STAGE31_IMPL6B.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_IMPL6B.md): JSON 규격, 알고리즘, smoke, 외삽, 한계까지 기록

`N=64` smoke 결과:

- wall: `1.36 s`, 핵심 산술: `0.984821 s`
- GMRES: 32회
- linear residual: `1.93475e-61`
- source residual: `4.50824e-61`
- 상태: `SMOKE`, acceptance 미판정
- 체크포인트 재개: 계산 없이 `0.19 s`에 성공

보수적 `O(N²)` 외삽은 2048 약 16.8분, 4096 약 67.2분, 합계 약 84분입니다. sbatch에는 2.14배 여유를 적용했습니다.

Judge 가짜 입력 시험은 PASS 경로 exit 0, Nref 문턱 실패 경로 exit 1, oracle 누락 exit 2로 정상 동작했습니다. 기존 solver JSON에 raw J 벡터가 없어 J 오차 두 항은 `legacy_binary64_basis`로 명시하며, HP pointwise 재비교를 수행한 것으로 위장하지 않습니다. `docs/s31_results/ka2_oracle_hp.json`은 아직 생성되지 않았습니다.