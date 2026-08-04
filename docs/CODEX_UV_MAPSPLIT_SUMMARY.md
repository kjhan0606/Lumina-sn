구현을 완료했습니다. 실제 116만 행 집계는 지시대로 실행하지 않아 숫자는 아직 산출되지 않았습니다.

- 분석기: [uv_mapsplit_offline.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_mapsplit_offline.py:1)
- 전체 보고서 및 실행 대기 상태: [CODEX_UV_MAPSPLIT.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_UV_MAPSPLIT.md:1)
- 경량 자기검사: PASS
- 2회 byte-identical 검사: PASS
- 매핑 술어 결함 음성 대조: EXPECTED-CHANGE-OBSERVED
- `src/` 수정, commit, GPU/모델 실행: 없음

운전석 실행 명령 한 줄:

```bash
python3 scripts/uv_mapsplit_offline.py --linepop /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10 --outdir validation/uv_mapsplit --report docs/CODEX_UV_MAPSPLIT.md
```