# SH-RADEQ Fable `REVISE` closure evidence — 2026-08-08

## 코드 계약

- raw τ production writer 3개, CUDA 중복 writer 0개
- 모든 writer가 write 전 required generation 증가, write 후 computed generation 확정
- A2-09 raw τ 소비 시작/종료 generation bracket
- τ mutation + generation bump 음성대조 차단
- writer/reader 공용 NLTE authority predicate
- bulk τ/A2-09 공용 LTE line population routine, LTE/NLTE branch 자가검사
- signed τ 무클램프, nonfinite amplification은 이름 있는 candidate abort
- 707개 sub-min BF edge는 exact zero 금지, SH-GRID 재개방

## 검증 결과

```text
A2-07 population selftest: PASS
PASS A2_08_SELFTEST N1_N8=8/8 L4=BLOCKED_MISSING_CHI_DATA
PASS A2_09_SELFTEST N1_N8=8/8 L3=BLOCKED_MISSING_ETA_DATA L5=BLOCKED_MISSING_ETA_DATA
PASS A2_10_SELFTEST N1_N8=8/8 L6=BLOCKED_INCOMPLETE_ADIABATIC
[SH-RADEQ-0][STATIC][PASS]
[SH-RADEQ-0][NEGATIVE-CONTROL][PASS] injections=8 detected=8
[TAU-WRITER-CENSUS][PASS] writers=3 ... cuda_writers=0
[TAU-WRITER-CENSUS][NEGATIVE-CONTROL][PASS] injections=4 detected=4
[E-NE4][PASS]
[E-NE4][NEGATIVE-CONTROL][PASS] injections=4 detected=4
MAKEFILE_HEADERS declared=22 included=22 missing=0 stale=0 verdict=PASS
git diff --check: PASS
```

빌드:

- CPU: rc 0, SHA-256 `bb5237f00f809cd379ccf916c26769e547d9cc3b8bb74ca8ca4c8117bbfea82f`
- OpenMP: rc 0, SHA-256 `37cbf8572d615bb16a4c742cb7d10935c9822a93b1a014e2fa00a674b89f5ec2`
- full CUDA `sm_80` link: rc 0, SHA-256 `e1cbf03f50f8cf848b99ade8dc60d1b9ec328af33d26531c95a2a3c94f94e6f7`

실제 BF census:

```text
positive=24542 above=23835 below_or_at_all=707 default_active=707
cmfgen=707 kramers_fallback=0
lowest nu_edge=5.84852771e13 Hz
[BLOCKED] action=REOPEN_SH_GRID rc=3
```

## 경계

이는 로그인 노드의 코드/정적/빌드 폐합이다. 모델 flight는 수행하지 않았다. 완전 CMFGEN
단열항이 없으므로 기본 production T_e publication은 계속
`RADEQ_INCOMPLETE_ADIABATIC`으로 차단되어야 한다. 따라서 구현 `REVISE` 폐합과 전체
SH-RADEQ flight 허가는 서로 다른 판정이다.
