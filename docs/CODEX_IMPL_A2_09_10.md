# CODEX 구현 보고 — A2-09 + A2-10 (개정 11)

기준 HEAD는 `8a9f861efb12e749fd47926b8e78be17d4ba8070`이다. 시스템 sandbox가
원 저장소 `.git` 쓰기를 막아, 같은 HEAD를 alternate object로 참조하는 로컬 Git
메타데이터에서 승인된 네 커밋을 선형 생성한다. push는 하지 않는다.

## 커밋 경계

| 순서 | 소속 | 커밋 | 내용 |
|---:|---|---|---|
| 1 | A2-09 seal | `36b84264da0dfdf6c2feb2827de47b628e28466e` | allowlist JSON+sha256만 봉인 |
| 2 | A2-09 구현 | 구현 커밋에서 확정 | eta/transition/CDF/old7897/selftest/gate |
| 3 | A2-10 seal | A2-09 구현 뒤 생성 | Te changed-output allowlist만 봉인 |
| 4 | A2-10 구현 | 최종 구현 커밋 | term ledger/root/Te publication/L-6/최종 보고 |

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

## A2-10

A2-09 구현 커밋 이후 별도 seal을 만든 다음 이 절을 완성한다.

## 남은 위험과 A2-11 인계

ETA_DATA/INFO가 없어 L-3/L-5 coverage와 물리 metric은 계산할 수 없다. 대형 model에서
line source가 undefined거나 eta가 비유한이면 publication 전체가 정직하게 막힌다.
formal source division, signed amplification, observer spectrum은 수정하지 않았고 A2-11에
그대로 인계한다. `.cu` diff는 0이며 GPU lifecycle/rate/emissivity는 A2-12+ 소유다.
