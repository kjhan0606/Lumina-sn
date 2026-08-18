[01:47:39] # 야간 자율주행 판정 — 2026-08-08 01:47
[01:47:39] 
[01:47:39] ## A. R7 MC 판정 대기
[01:47:39] 01:45 판정은 **무효**였다 — LUMINA_PURE_CMFGEN=0 이 하니스의 eval 에 덮여
[01:47:39] DET 를 두 번 돌리고 하나를 MC 라 불렀다(로그의 lane=DET 가 잡았다).
[01:47:39] 하니스에 T3_LANE 을 넣고 다시 돌린다.
[02:09:46]   MC 런 종료 rc=1
[02:09:46] 
[02:09:46] ```
[A2-10][SEED] bootstrap T_e published generation=1 n_shells=50 manifest=25b03b07ae22 reason=deck seed T_e — bootstrap before first transport (CPU)
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3
[02:09:46] ```
[02:09:46]   ★MC MATERIAL_PHASE_COMMITTED 없음
[02:09:46]   동세대 검사: rad/line/opacity/emissivity = { 1 }
      r7_mc_real.log lane=MC iter=0: 위상 view -> a208 -> a209
          [A2-10][BLOCKED] R7_MATERIAL_UPDATE_BLOCKED lane=MC iter=0  reason=RADEQ_TERM_MISSING te_generation_before=1 te_generation_after=1 te_manifest_preserved=1 generation_preserved=1 material_update=BLOCKED action=TERMINATE blocked_stale_delta=0 no_bracket_delta=0 missing_term_delta=1 schema_delta=0
      r7_mc_real.log lane=MC iter=0: PASS  [te=1 r=1 line=1 o=1 e=1 m=1]
    
    PUBLICATION_PHASE records=1 violations=0 verdict=PASS
[02:09:47] 
[02:09:47] **R7 MC = FAIL.**  사전등록에 없는 실패다 — R7 의 실제 실패다.
[02:09:47] 
[02:09:47] ## ★중단: R7 이 닫히지 않았다.  감마를 얹지 않는다(단독 귀속이 처음부터 깨진다). 2
[02:09:47] 다음 행동은 운전석이 깨어서 판단한다.
