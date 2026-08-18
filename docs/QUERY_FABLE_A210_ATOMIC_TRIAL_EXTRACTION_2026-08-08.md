# Fable 중요 판정 질의 — A2-07 core 추출인가 deep-clone 재호출인가

이 질의는 all-shell private atomic trial을 실제 기존 NLTE solver와 결박하는 중요 구현
경계다. 일반론으로 답하지 말고 아래 실물을 직접 읽어 판정해 달라. 파일 편집은 금지한다.

## 읽을 실물

- `src/lumina_plasma.c:15420-15975` (`nlte_solve_ion_shell` 및 rate assembly tail)
- `src/lumina_plasma.c:15970-16490` (`nlte_solve_all` begin/core/post/commit)
- `src/lumina_plasma.c:3042-3110` (`nlte_writeback_ion_stage`)
- `src/lumina_plasma.c:15748-15975` (`nlte_update_tau_sobolev`)
- `src/lumina_element_wide.c:2000-2465` (EW private/commit lane)
- `src/population_contract.c:250-270` (`PopulationTransaction`)
- `src/opacity_publication.c:119-149`, `src/emissivity_publication.c:71-81`
- `src/radeq_publication.c:16-23`
- `docs/SPEC_A2_09_10_V1.md:390-470`
- `docs/FABLE_VERDICT_SH_RADEQ_COMPLETE_ADIABATIC_2026-08-08.md`

## 실측한 문제

현 `nlte_solve_all`은 private population arrays로 pointer를 잠시 바꾸지만, core 뒤 commit
전에 다음 public/global side effect를 수행한다.

1. `nlte_writeback_ion_stage`가 ion population을 쓰고 bulk tau를 재조립한다.
2. `nlte_update_tau_sobolev`가 raw tau/source slab과 tau generation을 쓴다.
3. `g_ew_tau_authority`를 free/교체한다.
4. runtime manifest와 여러 diagnostic CSV를 쓸 수 있다.
5. `NLTEConfig` error/counter/stamp 및 파일-scope 진단 상태를 갱신한다.

따라서 구조체의 주요 배열만 deep-copy한 뒤 기존 `nlte_solve_all`을 trial마다 부르는
방식은 public pointer는 지켜도 global/file/generation side effect를 격리하지 못한다.
반대로 A2-10 전용 간이 Saha/LTE solver는 A2-07과 다른 population producer가 된다.

## Codex 권고안 A

기존 코드를 다음 세 층으로 분리한다.

```text
nlte_population_candidate_begin
  - trial T_e + private ion/level/ne/partition/within_sl buffers
  - immutable RF/Jbar/gamma/old committed population seed

nlte_population_solve_core(candidate_view)
  - partition + within-SL + pair/EW/CE solve만
  - opacity는 rate 입력으로 read-only
  - public generation/raw tau/files/global authority write 0
  - candidate population residual/charge residual/status를 반환

post-core private material producers
  - candidate ion writeback
  - candidate tau/opacity
  - candidate emissivity
  - internal energy + vector adiabatic + all-shell RE ledger

single bundle commit
  - 모든 preflight 뒤 T_e/pop/ne/partition/within-SL/tau/opacity/emissivity 게시
```

기존 `nlte_solve_all`도 같은 `solve_core`를 호출한 뒤 기존 production postprocess/commit을
하는 wrapper로 바꿔, A2-07과 A2-10이 동일 solver를 공유하게 한다. 진단은 core 내부
직접 파일쓰기 대신 optional sink를 받고, trial에서는 null sink를 쓴다. EW의 shadow
path와 population writer는 candidate view만 읽고 쓰도록 한다.

## 대안

- **B:** `NLTEConfig/AtomicData/PlasmaState/OpacityState`를 깊은 clone하고 기존
  `nlte_solve_all`을 그대로 호출한 뒤 결과만 취한다.
- **C:** A2-10 전용 축약 population solver를 새로 만든다.
- **D:** scalar shell root를 유지하고 neighbor는 직전 candidate/committed 값으로 둔다.

## 판정 요청

```text
TRIAL_POPULATION_ARCH = A | B | C | D | OTHER
CORE_MUST_BE_SIDE_EFFECT_FREE = YES | NO
EW_IN_TRIAL = SAME_CORE_CANDIDATE_VIEW | DISABLE | OTHER
TAU_WRITEBACK_POSITION = POST_CORE_PRIVATE | IN_CORE | OTHER
LEGACY_WRAPPER_REUSES_CORE = REQUIRED | OPTIONAL
```

권고안 A가 틀렸다면 구체적인 누락 state와 대체 경계를 소스 줄로 제시해 달라. A가
맞더라도 first extraction에서 반드시 private화해야 하는 mutable field/global/file sink를
짧은 체크리스트로 달라.
