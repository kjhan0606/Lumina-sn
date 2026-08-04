구현과 재캡처 준비를 완료했습니다. 전체 내용은 [CODEX_EMISS_E5.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E5.md)에 정리했습니다.

핵심 결과:

- A/B/B2 3-lane 동시 캡처 추가
- B2는 covered 전이만 `A_ul·n_u`로 교체하고, 미정의분은 production A 방출률 유지
- manifest에 통제 정책, 유지 전이/cell 수·기여율 기록
- 미정의 A 기여를 pre-EPAY 기준 frequency-bin별·shell별 배열로 기록
- 판정기가 B 실패 후에도 B2를 독립 실행하고 partial verdict 생성
- 재캡처 deck에서 A/B/B2 계약과 장부 폐합 자동 검증

검증 결과:

- CPU fixture: PASS
- A/B/B2 payload 검사: PASS
- pre-E5 대비 A, B, seeded-B, undefined CSV byte 동일: PASS
- Python compile 및 stage31 strict build: PASS
- `make -B lumina_cuda`: PASS, 실행하지 않음
- rung 1–5 순차 적용 및 최종 파일 `cmp`: PASS
- 신규 clamp/floor, 모델/GPU run, commit: 모두 0

B2 fixture SHA-256은 `b85b5229...b9d044`, 재캡처용 binary SHA-256은 `22d56395...7af68`입니다.

패치 사다리는 [e5_expected_changes.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/e5_expected_changes.txt)에 있으며, 운전석 제출용 deck은 [sbatch_emiss_ab_capture.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_emiss_ab_capture.sh)입니다. 기존의 광범위한 작업트리 변경은 보존했고 커밋하지 않았습니다.