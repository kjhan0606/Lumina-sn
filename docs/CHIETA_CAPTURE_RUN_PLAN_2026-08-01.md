# χ,η 캡처 런 계획 (user 승인 ④, 5B 착지 시 발사)

목적: ①Stage 3.1 판별 벤치의 유일 입력(전 셸×전 주파수 동결 χ,η — LCMFCE01 binary v1) ②**s20 frozen-cell 오라클 입력 동시 캡처**(기지 계측 부채 폐합 — Wave 3 §4.4 s20 축 판정의 선행조건).

## 1. 바이너리 (사다리 순서: 5B 착지 → 빌드 → 인증 → 배포 → 제출)

- 현 트리(Wave-3.2 A~A6 + R7 + S31 greenfield) clean make. S31 파일은 런 바이너리에 무영향 확인(링크 여부 실측).
- **바이너리별 인증(규약)**: ①R1 byte-매트릭스 12/12 재실행(새 바이너리) ②R7 게이트 OFF-중립성(미설정=frozen replay 산출 byte-ident) ③R7 writer 왕복 fixture ④음성 대조 스팟(원장 주입 1건). 인증 로그 보고서 첨부.
- 클론 배포 선행(규약: 새 바이너리는 큐잉 전 클론 배포).

## 2. 런 구성 (parity59 정본 재현 + 캡처 게이트)

- env = parity59 RESOLVED CONFIG 전량 재현(결정론 재현은 parity25/26 자릿수 재현으로 기입증) + 추가:
  - `LUMINA_CMF_FROZEN_CHIETA_DUMP=<rundir>/chieta_iter10` + expected-iter env(producer **iter=10** — consumer 계약 scripts/cmf_chieta_check.py가 fail-closed 판정)
  - frozen-cell 오라클 캡처 셀에 **s20 추가**(기존 s0/s8 유지 — lumina_oracle_cell_s20.csv 산출)
- EW 게이트 비무장(D1 수리됐어도 캡처 런은 관측 전용 순수성 유지). EVENT_LOG=1(상설 규약).
- 제출: job-per-run 단발 sbatch, 파티션 h200→h100(full-NLTE 80GB — a40 제외), GPFS scratch, RUN FOOTER(argv 복제 규약).

## 3. 사전등록 기대

- 캡처 무관 산출(스펙트럼·기존 CSV)은 parity59와 자릿수 재현(결정론) — 어긋나면 새 바이너리 회귀 신호로 즉시 기재.
- chieta sidecar: iteration=10·post_damp flag·sha 일치·η 분해 max_abs 실측값(하드코딩 아님 — A4에서 실측화 완료).
- 착지 시 판정 배터리: consumer check RC 0 → Stage 3.1 판별 벤치 입력 승격 + Γ 삼중대조의 B-lane을 캡처 χ,η로 교차 재검(자유 검증).

## 4. 후속 연쇄

캡처 착지 → Stage 3.1 §7 판별 벤치(새 결정론 Jν vs MC 장 vs CMFGEN jnu4 — "수송 결함 vs χ,η 결함" 판별) + s20 EW shadow 측정(스펙 §4.4 잔여 축).
