# Fable 중요 판정 — A2-10 atomic trial core 추출 (2026-08-08)

- model: `claude-fable-5`
- Claude Code CLI session: `121c3f89-a44a-481b-b605-ff8502a14ee8`
- 질의: `docs/QUERY_FABLE_A210_ATOMIC_TRIAL_EXTRACTION_2026-08-08.md`

```text
TRIAL_POPULATION_ARCH = A
CORE_MUST_BE_SIDE_EFFECT_FREE = YES
EW_IN_TRIAL = SAME_CORE_CANDIDATE_VIEW
TAU_WRITEBACK_POSITION = POST_CORE_PRIVATE
LEGACY_WRAPPER_REUSES_CORE = REQUIRED
```

## 핵심 근거

- deep clone으로 기존 `nlte_solve_all`을 재호출하면 file-scope
  `g_ew_tau_authority`, tau generation ratchet, diagnostic/runtime-manifest 파일,
  publication 전역 counter를 격리하지 못한다.
- A2-10 전용 축약 solver는 명세의 “trial마다 A2-07 solver 실행”을 위반한다.
- scalar root와 committed neighbor 혼용은 이미 기각된 all-shell 계약 위반이다.
- EW는 기존 `EWPrivateView`를 가지고 있으므로 끄지 말고 같은 candidate population
  view에 연결해야 한다.
- tau writeback은 현 production처럼 CE 수렴 뒤에 두되 private slab에 수행해야 한다.
- legacy A2-07 wrapper와 A2-10 trial이 같은 core를 호출해야 두 solver의 독립 drift를
  구조적으로 막을 수 있다.

## first extraction 필수 private state

1. `T_e` vector와 trial `T_e_generation`.
2. ion/level/`n_e`/partition arrays.
3. `within_sl_frac` 배열. 현 transaction은 이 배열을 포함하지 않아 abort 뒤 새 trial
   값을 남기는 잠복 불일치가 있다.
4. `partition_stamp`, `within_sl_stamp`.
5. population error/status/counter와 required/committed generation ratchet.
6. candidate tau/source/validity slab과 tau generation view.
7. EW status/tau authority — file-scope global 교체 대신 post-core producer 인자.

## null/optional sink로 분리할 항목

- `lumina_rates_decomp.csv`와 `g_ddc` 진단.
- `nlte_levels_iter*.csv`와 static dump counter.
- EW dump/runtime manifest/`ew_dump_failed`.
- EW process-global instrumentation counters.
- CE pass/Jbar dump marker와 진행 stdout.
- A208/A209/A210 개별 publication global counter. bundle commit은 모든 후보를 먼저
  preflight한 뒤 실패 없는 swap/copy만 수행해야 한다.

## Codex 의견과 Fable 사용 경계

판정에 동의한다. 첫 구현은 public 객체를 잠시 pointer-swap하는 방식이 아니라,
`NLTEConfig/AtomicData/PlasmaState`의 shallow candidate struct가 private mutable arrays를
가리키게 하는 view부터 만든다. 그 다음 같은 view를 legacy와 trial core가 공유한다.

사용자 지시에 따라 이 판정 뒤 Fable 호출은 중단한다. 이후 구현·자가검사·일반 리뷰는
Codex가 수행하며, 물리식 또는 소유권 판정을 뒤집을 새 핵심 쟁점이 생기기 전에는
Fable token을 사용하지 않는다.
