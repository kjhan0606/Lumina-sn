3단계 보고와 판독 스크립트 보강을 완료했습니다.

- 전체 보고: [CODEX_UV_T2N9_STEP3.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_UV_T2N9_STEP3.md)
- forensic 모드: [uv_t2n9_offline.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py)

핵심 판정:

- nonpositive 28,949행은 writer 기록 조건상 모두 BALL 내부입니다.
- C는 대체 없이 구성 불가하므로 네 갈래 판정 전 `UNRESOLVED-FAIL-CLOSED`를 유지했습니다.
- 기존 산출물에는 0/음수·sentinel·실제 솔버 음수 분리가 없어 이를 정의 불가로 명시했습니다.
- N9: s>=5 rate-shape `0.7623111`, shell 8 BALL 에너지 `0.9956304`, B1–B4(s>=5) 전 셸 `1.0`, 열적 항등식 최대 오차 `2.22e-16`, 1 ULP로 PASS했습니다.
- 대장 기재용 `EPAY-REPLAY-001`, `LINEPOP-POPAUTH-001` 문안을 보고서에 작성했습니다.
- 문법 검사와 합성 self-test는 PASS했습니다. 무거운 판독·수송·GPU·모델 실행은 하지 않았습니다.

추가 판독 명령 한 줄:

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && python3 scripts/uv_t2n9_offline.py --forensics-only --linepop /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10 --outdir /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline
```

사전등록, `src/`, 검증 대장 원본은 수정하지 않았고 커밋도 하지 않았습니다.