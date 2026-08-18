# Stage-IV overlapping-pair ownership audit — 2026-08-09

범위: `LUMINA_NLTE_STAGE4=1`, `nlte_get_pairs()`의 23 pair와 CPU production
`nlte_solve_all_core()` / `nlte_solve_ion_shell()` 정적 경로.

## Census

| element | earlier pair | later pair | shared slot | final shared-level owner |
|---|---|---|---|---|
| Si | II–III | III–IV | III=1 | earlier II–III |
| Fe | II–III | III–IV | III=6 | earlier II–III |
| Co | II–III | III–IV | III=11 | earlier II–III |
| Ni | II–III | III–IV | III=14 | earlier II–III |
| Ti | II–III | III–IV | III=21 | earlier II–III |
| Cr | II–III | III–IV | III=24 | earlier II–III |
| Al | II–III | III–IV | III=27 | earlier II–III |
| O | I–II | II–III | II=36 | earlier I–II |

23 pair 중 16 pair call이 다른 pair와 slot을 공유하고, 7 pair call(Ca, S, C,
Mg, Sc, V, Mn)만 non-overlap이다.

## 실제 저장 규칙

1. 두 pair 중 하나라도 slot을 공유하면 `pair_shares_slot=1`이다.
2. 그러면 `single-total`/ion-lock env와 무관하게 full-level expansion에서
   `lock=1`이 되어 lower/upper 각각을 기존 `atom->ion_number_density` 총량으로
   rescale한다.
3. later pair는 solve 전 shared lower block을 저장하고 solve 후 그대로 복원한다.
   따라서 later `(III,IV)`가 계산한 III block은 최종 state에 남지 않는다.
4. `nlte_writeback_ion_stage()`도 slot-sharing pair 전체를 건너뛴다. 따라서
   upstream ion density의 II/III/IV 분율은 이 pair solve로 갱신되지 않는다.

결과적으로 overlapping Stage-IV pair에서 현재 rate solve가 소유하는 것은 각
upstream stage total 안의 level shape다. 전체 II/III/IV ion fraction은 하나의
결합 rate system이 소유하지 않는다. `single-total=1`에서 생산 GTH가 반환한
lower/upper stationary fraction도 최종 per-ion rescale 뒤에는 stage-total 권위가
아니다.

## Exact-zero 수리와의 관계

Fe III/IV upstream totals가 둘 다 exact zero인 job 251601 shell에서는 final
per-ion rescale 목표도 `0/0`이고 shared-III 이전 block도 exact zero다. 따라서
조립 전 exact-zero 반환은 기존 최종 소유권과 동일한 state를 직접 쓴다.

양의 finite Stage-IV 셀은 다르다. 다음 production 폐합은 아래 중 하나를 요구한다.

- 한 원소의 II/III/IV를 한 multi-stage generator에서 동시에 푼다.
- 이미 존재하는 element-wide solve가 stage-total final owner가 되도록 승격한다.
- 최소한 shared stage의 later-solve/earlier-solve 중 한 곳을 명시적인 단일 owner로
  정하고 conservation total을 최신 level totals에서 일관되게 전달한다.

어느 경우든 CMFGEN 동종 ion fraction과 finite level populations가 함께 맞아야 하며,
양팔 agreement만으로 통과시키지 않는다.

## 외부 잣대 고정

toy06 19.48 d CMFGEN ion-fraction 원본 SHA는 Fe
`86d21b536655a694da787ac4516964be1009a81602a843d8325cec64d5a402cd`,
Co `0e2c39e77641872cbe7c034f5ceb42199a05eb16a6dff5f4790059d4e9613c59`,
Ni `9b3d84f8ef9a800ca606d7987803f0aa62ef53f354317f1f0798809a9af39fd0`,
Si `671fcc3b6ceb6c765741f27771b1b062757ebbbce8f1890bd1b8078b18fb501d`다.
기존 Lumina shell midpoint 보간 표
`validation/cmfgen_toy06_19p48d/analysis/stage4_round2/fiv_depth_crossover.csv`
SHA는 `15b804a8e79d6fcbe89259f42fd6cfd38f3ca0c655a9e7a8b124c07c842d032c`다.
새 flight 비교는 이 역사 표의 Lumina 열을 재사용하지 않고 새 출력에서 다시 계산한다.
`scripts/regression_ledger.py --self-test --no-append`는 PASS했다.
