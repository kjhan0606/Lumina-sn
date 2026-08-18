# Fable 중요 판정 — 완전 CMFGEN 단열항과 all-shell trial 경계 (2026-08-08)

- model: `claude-fable-5`
- Claude Code CLI session: `ce79cefa-af58-4232-b300-e396a9018703`
- 질의: `docs/QUERY_FABLE_SH_RADEQ_COMPLETE_ADIABATIC_2026-08-08.md`

아래는 요청한 판정과 핵심 근거다.

```text
PHYSICS_MAPPING = ACCEPT
ALL_SHELL_TRANSACTION = REQUIRED
SIGNED_LEDGER_SPLIT = REQUIRED
STENCIL_AND_CENTER = ACCEPT
IMPLEMENT_NOW = VECTOR_PRODUCER_ONLY
```

## Fable 근거 요약

1. `eval_adiabatic_v3.f:230-249`의 `A/B/C/D`와 signed `WORK`를 cgs로 풀면
   Codex가 정리한 네 항과 일치한다.
2. `STEQ_T` 주경로는
   `UPDATE_BA_ST(BA_T,STEQ_T,INT_EN,TOT_ENERGY)`이며, diagnostic/EHB의
   `COL_EN` 혼재는 이 주경로와 별개다.
3. gradient 세 항이 인접 depth의 trial `T`, `n_e/n_atom`, `INT_EN`을 읽고,
   CMFGEN도 upper/lower off-diagonal band로 전 depth를 결합한다. committed neighbor를
   scalar trial에 섞는 우회는 A2-10 atomic-generation 계약 위반이다.
4. signed `WORK`가 음수가 될 수 있으므로 raw signed component를 보존하고
   `max(q,0)` cooling / `max(-q,0)` heating으로 나누어야 한다.
5. Lumina 배열을 안→밖으로 뒤집은 `neighbor(0)=1`, `neighbor(s>0)=s-1` stencil과
   `0.5*(inner+outer)` center는 CMFGEN의 단순 차분에 대응한다.
6. 현재 production은 `RADEQ_INCOMPLETE_ADIABATIC`으로 atomic rollback하므로 이를
   유지한 채 pure vector producer와 known-answer만 먼저 봉인해야 한다.

## Fable가 추가한 중요 경계

`cmfgen_sub.f:2941-2954`에서 `SN_MODEL .AND. DO_CO_MOV_DDT`이면
`EVAL_ADIABATIC_V3`가 아니라 이전 time-step population을 쓰는
`EVAL_TEMP_DDT_V2`가 실행된다. production 배선 전에 기준 실행의 `DO_CO_MOV_DDT`
설정을 실측해야 한다.

현재 toy06 19.48d 기준 덱은 다음과 같이 확인했다.

```text
/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/VADAT:119  T [INC_AD]
/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/VADAT:675  F [DO_DDT]
```

따라서 이 steady-state 기준 실행에는 V3가 동종 잣대다. 다른 time-dependent 기준
실행에는 자동 일반화하지 않으며 `DO_DDT=T`이면 별도 `EVAL_TEMP_DDT_V2` 계약 없이는
fail-closed한다.

## Codex 의견

판정에 동의한다. 현재 단계에서는 atomic/NLTE 생산 경로를 바꾸지 않고, V3의 signed
네 항을 전 shell candidate에서 계산하는 순수 함수와 단위 왕복 시험만 구현한다.
`CMFGEN_COMPLETE` ledger 승격과 `T_e` publication은 all-shell private atomic
transaction이 완성될 때까지 계속 금지한다.
