전체 결과는 [CODEX_DECK_REGEN.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_DECK_REGEN.md)에 기록했습니다.

- 자원 산정: 준위 36,355, edge 10,218,333.
- 보수 GPU 요구량: 110,569,158,168 byte = 105,446.97 MiB.
- H200 143,771 MiB에 적합하며 여유는 38,324.03 MiB.
- 덱 생성: CPU Slurm 제출이 컨트롤러 연결 차단으로 실패하여 미실행. 신규 디렉터리는 생성되지 않았고 기존 `_sivcaiv` 덱은 그대로입니다.
- 검증 게이트 4개: 신규 덱 부재로 모두 NOT RUN. 현행 vintage 경로상 S V·Co II의 45,255선 때문에 게이트 1 실패가 예상됩니다.
- Co IV Υ: 4,455/4,455개 Ω 행이 Fe III와 동일하여 대용 문제가 해소되지 않습니다. 별도 원자자료 수리가 필요합니다.
- 생성 배치와 읽기 전용 검증기는 각각 [sbatch_deck_regen_fullcov.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_deck_regen_fullcov.sh), [verify_deck_regen_fullcov.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_regen_fullcov.py)에 준비했습니다.

GPU·모델 런·commit은 하지 않았고 `src/` 및 회귀 대장은 수정하지 않았습니다.