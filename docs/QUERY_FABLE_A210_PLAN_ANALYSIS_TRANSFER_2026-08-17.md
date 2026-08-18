# Fable 총괄·계획 및 분석·평가 이관 — A2-10 non-census closure

사용자가 2026-08-17 KST에 총괄·계획과 분석·평가를 Fable로 이관했다.
Codex는 이후 코딩, 실행, tripwire/monitoring, 문서화, 커밋을 맡는다.
Fable은 아래 sealed 증거를 바탕으로 목표 전체의 계획과 물리 판정을 소유한다.

## 전체 목표와 절대 제약

목표는 K18 cancellation census와 K12→K18 검증을 완료하고, 물리적 이상이나
자원 충돌이 없으면 필요한 분기만 거쳐 A100×2 non-census gate를 통과시키는 것이다.

- 모든 물리값 floor/cap/clamp/jitter/repair와 음수 삭제·절댓값 치환·임의 scaling 금지.
- 물리적 음수/비유한값은 원인을 규명하고 fail-closed한다.
- rejected pre-core tau refresh는 되살리지 않는다.
- coevolution generation barrier를 보존한다.
- 수동 GPU는 선택 카드 외부 PID 또는 Slurm allocation이 생기면 우리 process group만
  종료하는 tripwire를 유지한다.
- aggregate deficit만으로 물리 원인을 확정하지 않는다.

## 현재 정본 증거

1. `validation/a2_10/A2_10_REFINEMENT_K12_K18_COMPARISON_2026-08-16.json`
   (SHA-256 `b6a0be0bc9ba0a44358e0f949578037f262f3ba7cb97a1002859c04453e62ec7`):
   K12 unresolved 19 → K18 unresolved 0, surviving 0, physical mutation/repair 0.
2. 중요 선행 물리 감사:
   `docs/FABLE_AUDIT_A210_K36_FINITE_COMPONENT_BRANCH_2026-08-17.md`.
   승인된 읽기 전용 분기는 shell-0 Fe/Co/Ni IV emission 90%를 운반하는 선의 Lumina
   `tau`, `Jbar/S`, `beta`와 CMFGEN depths 67/68 `1-ZNET` 대조다.
3. 현재 A100×2 K36 line-saturation run:
   `/gpfs/kjhan/lumina/a210_line_saturation_a100x2_nonoverlap_sobolev_k36/diag_20260816T224556Z_f9c2d1b826d5`
   binary SHA `f9c2d1b826d5205fa68f938c2affc2c6d9aa86772257fb95958ab9e65a95526c`.
4. R1 exact와 R6는 앞선 K36 owner 정본과 bit-exact다:
   45 iterations, residual `9.6662782724980344e-09 < 1e-8`, refinements 36,
   valid lines 2,180,286, valid cells 109,014,300, partial/unsampled 0.
5. R2 seed barrier와 material census도 정본과 bit-exact다:
   `r1_generation=1`, `te_generation=1->1`, `population_generation=1->2`,
   raw-negative/mild-negative/SRCE_CHK `4,246,581/4,246,577/4`, raw tau preserved,
   pre-core refresh 0, stage4 0, repair 0. R2 exact solve는 아직 실행 중이다.
6. 후처리는 자연 model rc=1과 REQUESTED_TE `RADEQ_NO_BRACKET` 뒤에만 순차 실행한다:
   V2 combined 90% summary/match → V3 roundoff-aware evidence classification →
   V4 Fe/Co/Ni IV 각각의 exact 90% coverage 검사.
7. 대상 emission 비중은 Co IV 82.87%, Fe IV 10.47%, Ni IV 6.65%여서 combined 90%
   prefix가 개별 90%를 보장하지 않는다. V4가 undercoverage이면 그 사실은 물리 원인이
   아니며, 같은 진단의 ion별 minimal 90% prefix union만 재실행하도록 사전등록했다.
8. K24/K30은 R1/R2와 proof는 통과했지만 shell 0--3 physical no-bracket으로
   `model.rc=1`; final non-census completion artifact는 아직 없다.

## Fable에 맡기는 총괄·계획 및 분석·평가

현재 실행을 변경하지 말고 다음을 압축해서 판정하라.

1. 전체 목표를 끝내기 위한 권위 있는 다음 5단계 계획. 각 단계의 입력 증거, PASS 조건,
   fail-closed 분기, Codex 실행 항목을 명시한다.
2. V4 PASS와 UNDERCOVERED 각각에서 정확히 다음 행동을 하나씩 지정한다.
3. V3/V4 결과가 나온 뒤 `tau`, `Jbar/S`, `1-beta`, CMFGEN `1-ZNET`으로
   radiation/Jbar와 opacity/lower-population/line-universe를 가르는 판정 규칙을
   산술 증거상한과 물리 tolerance를 혼동하지 않도록 명시한다.
4. 어떤 증거가 있어야 물리 코드 변경을 허가하는지, III-stage null control과 offline
   recomputation 및 preregistered negative control을 포함해 명시한다.
5. 최종 A100×2 non-census gate PASS를 주장하기 위한 최소 정본 artifact와 불변조건을
   열거한다.

계획·분석만 수행하고 파일 수정, 코드 작성, process 신호, 작업 제출은 하지 말라.
응답은 `VERDICT`, `CURRENT ASSESSMENT`, `NEXT FIVE STAGES`, `BRANCH RULES`,
`FINAL GATE EVIDENCE` 순서의 간결한 Markdown으로 작성하라.
