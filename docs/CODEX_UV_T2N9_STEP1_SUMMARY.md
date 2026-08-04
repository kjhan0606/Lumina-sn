1단계를 완료했습니다.

- 전체 보고: [CODEX_UV_T2N9_STEP1.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_UV_T2N9_STEP1.md)
- 분석 스크립트: [uv_t2n9_offline.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py)
- 사전등록: [PREREG.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/uv_t2n9/PREREG.md)

핵심 요약:

- writer에서 little-endian, 152 B 헤더, 80 B line-static, 76 B row 배치를 확정했습니다.
- `rows×76=88,855,020 B` 외에 `601371×80=48,109,680 B` 선 정적 테이블 등이 있어 전체 예상 크기는 `137,151,032 B`입니다.
- 판독 경계는 BALL 기준 `|C/A−1|≤5%`, CMFGEN 수준 `[1/3,3]`으로 사전등록했습니다.
- A bitwise 재조립과 1-bit 주입 결함의 의도적 FAIL, 2회 byte 결정론 검사를 스크립트에 포함했습니다.
- 예상 자원은 wall 10분, RSS 2 GiB, OMP 4 threads입니다.

실행 명령:

```bash
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && python3 scripts/uv_t2n9_offline.py --linepop /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10 --chieta /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/chieta_iter10 --outdir /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline'"
```

미해결 항목은 grammar-debug 접속 차단으로 자기검사가 미실행된 점, C가 선택된 5개 셸에 국지화된 점, population coverage 및 EPAY scale 부재로 실행 시 fail-closed할 가능성입니다. 무거운 연산, 로그인 노드 연산, `src/` 수정, 커밋은 수행하지 않았습니다.