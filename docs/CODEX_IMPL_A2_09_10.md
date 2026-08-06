# CODEX 구현 보고 — A2-09 + A2-10 (개정 11)

기준 HEAD는 `8a9f861efb12e749fd47926b8e78be17d4ba8070`이다. 시스템 sandbox가
원 저장소 `.git` 쓰기를 막아, 같은 HEAD를 alternate object로 참조하는 로컬 Git
메타데이터에서 승인된 네 커밋을 선형 생성한다. push는 하지 않는다.

## 커밋 경계

| 순서 | 소속 | 커밋 | 내용 |
|---:|---|---|---|
| 1 | A2-09 seal | `36b84264da0dfdf6c2feb2827de47b628e28466e` | allowlist JSON+sha256만 봉인 |
| 2 | A2-09 구현 | `bf2af37fa11a18c2529559311a7a23fd0c41b60c` | eta/transition/CDF/old7897/selftest/gate |
| 3 | A2-10 seal | `540ebbd3bf66d29b7f36cf813301cf8c5df5d998` | Te changed-output allowlist JSON+sha256만 봉인 |
| 4 | A2-10 구현 | `SELF` (본 보고서를 포함한 최종 커밋; 작업트리 사후 증명에 실제 hash 기록) | term ledger/root/Te publication/L-6/최종 보고 |

## A2-09 처분과 구현

원장 3행은 checked Jbar transition, dilution/old-probability 제거,
`eta_reemit` CDF old7897 표본으로 각각 종결했다. E01–E21은
`validation/a2_09/A2_09_EMISSIVITY_CENSUS.json`에 21군 전량 분류했으며 unknown=0이다.
E18 formal은 `BLOCKED_A2_11`, GPU는 A2-15로 남겼다.

`CpuEmissivityPublication`은 comoving bin-edge grid에서 BB/BF/FF,
true/scattering/declared total, value/status/generation을 한 transaction으로 게시한다.
CDF는 `eta_reemit*Delta_nu`, monotone, 마지막 값 bitwise 1이며 generation/grid/channel
mask/summation order를 SHA-256으로 묶는다. old7897은 이 CDF에 한 RNG draw만 쓰고,
stale/empty이면 packet을 fail-closed한다. production Planck call, old CDF/probability,
last-channel fallback은 0이다.

### A2-09 seal 증거

- baseline: `8a9f861efb12e749fd47926b8e78be17d4ba8070`
- seal: `36b84264da0dfdf6c2feb2827de47b628e28466e`
- JSON blob: `0eec9c2bdff163da8ee811ebfcdbd59deadfe2bf`
- current/sidecar/sealed SHA-256:
  `0b0274a75df349eac5e60800967443b647ebd84458ade4a2ca703128ad517db3` (3중 일치)

### A2-09 게이트

- `make lumina`: rc 0.
- A2-03/04/05/06/07/08 및 A2-09 독립 selftest: PASS. A2-04 replay는
  `scripts/a2_04_l0_replay.py --self-test` 정식 wrapper rc 0으로 확인했다.
- A2-09 N1–N8: expected child rc `4,5,4,4,4,4,5,5`, wrapper rc 전부 0.
- A2-01 census: rc 0, rows=157, completed=20, unclassified=0.
- L-3/L-5 CHAIN·ORACLE_INPUT: `BLOCKED_MISSING_ETA_DATA`, child rc 3,
  `truth_f_cov=null`. 내부 PASS를 물리 PASS로 승격하지 않았다.
- 신규 TU는 Z-validator/tau/population/canonical 네 build에 직접 link했고 독립
  `a2-09-emissivity` case를 추가했다. 전체 배터리는 운전석 지시대로 실행하지 않는다.

## A2-10 처분과 구현

원장 `rate_radeq` 2행은 `T_rad/W` 스칼라 공급을 제거하고
generation-bound `A210TermLedger.checked_J_nu`로 1:1 이관했다. R01–R18은
18군 전량, 저작 시점 direct `T_e` writer는 14행 전량을 분류했고 unknown=0이다.
R18은 production 무기여 offline oracle로 유지했다.

`radeq_publication.c`/`.h`는 photoionization, line, free-free, Compton, gamma,
nonthermal, recombination, collisional, adiabatic 항을 heating/cooling 상쇄 전
별도 기록한다. 양의 유한 bracket에서 항별 `J_nu` 잔차의 root를 풀고,
no bracket/root/nonfinite/stale이면 Te/ne 부분 게시 없이 rollback한다.

line은 `radiative_line_included && collisional_or_escape_included` overlap을 먼저
`RADEQ_TERM_SCHEMA` rc 5로 거부한다. 모든 shell에서 owner의 수는 정확히 1이며,
`normalized_line_owner_closure<=1e-12`다. 실제 카운터 `line_owner_overlap_shells`/
`line_owner_closure_failures`를 둘 다 출력하고, L-6은 둘의 0을 필수로 요구한다.

A2-07 `population_te_manifest_sha256(const double*,size_t,char[65])` 선언·정의의
blob은 변경하지 않았다. Te 게시와 population stamp의 manifest는 bit-exact로
비교하며 geometry/solve epoch은 별도 `te_context_sha256` domain에 결박했다.

### A2-10 seal 증거

- baseline: `bf2af37fa11a18c2529559311a7a23fd0c41b60c`
- seal: `540ebbd3bf66d29b7f36cf813301cf8c5df5d998`
- JSON blob: `155c261f6cd9dc5d52a278b10a380085fad6ab61`
- sidecar blob: `26ef4d7106fc67ed959ae098059ab03791665cb7`
- current/sidecar/sealed SHA-256:
  `e8d3eb5745b50d465c97bf4db20858670da114681907e1830309943b8e59e5ba` (3중 일치, verifier rc 0)

### A2-10 게이트 산출물

- `make lumina`: rc 0. 기존 warning은 남았으나 신규 error는 0.
- A2-01 census: rows=157, completed=20, unclassified=0, rc 0.
- A2-10 census: R01–R18=18, direct writer=14, unknown=0,
  A2-07 signature unchanged=true, production `T_rad/W` read=0, rc 0.
- analytic root: `Te=5000 K`, `E_balance=0`, manifest exact=true,
  context separate=true, partial publish=0.
- owner: overlap_shells=0, closure_failures=0, max closure=0.
- N1–N8 child rc: `4,4,4,5,5,5,4,5`; wrapper rc는 모두 0.
- L-6 CHAIN·ORACLE_INPUT: `BLOCKED_FIXED_T_AND_MISSING_LINEHEAT`, child rc 3,
  `truth_f_cov=null`, `heat_residual_qualified=false`. analytic PASS를 L-6 PASS로 승격하지 않았다.
- Z-INERT: Z-validator/tau/population/canonical 네 hard-coded build에 A2-09/A2-10 TU를
  직접 link했고 A2-08/A2-09/A2-10 독립 case를 포함한 Z=9 선별 build/run rc 0.
  전체 D/K/Z/CP 배터리는 요청대로 실행하지 않았다.
- A2-03–A2-10 생산 target/selftest와 A2-04 replay wrapper 회귀: 전부 rc 0.
- 대장은 `validation/a2_10/A2_10_REGRESSION_LEDGER.jsonl` 정확히 1행이며,
  게이트 JSON SHA-256은 L-6 `07756aa9...b8181c`, census `884bef9a...bfe2`,
  selftest `76590b8b...51d`, term manifest `07931c21...7033`이다.

## 남은 위험과 A2-11 인계

ETA_DATA/INFO가 없어 L-3/L-5 coverage와 물리 metric은 계산할 수 없다. L-6도
released-T/LINEHEAT truth가 없어 물리 Te·term metric을 아직 인증할 수 없다. 대형 model에서
line source가 undefined거나 eta가 비유한이면 publication 전체가 정직하게 막힌다.
formal source division, signed amplification, observer spectrum은 수정하지 않았고 A2-11에
그대로 인계한다. A2-11은 signed opacity/emissivity publication을 generation/stamp
검증 후 formal source에 소비하고, line owner 배타성을 재해석하지 않아야 한다.
`.cu` diff는 0이며 GPU lifecycle/rate/emissivity는 A2-12+ 소유다.
