# Raw Sobolev τ writer census — 2026-08-08

## 결론

production `OpacityState.tau_sobolev` writer는 아래 3개뿐이다. 모두 첫 cell write 전에
`tau_sobolev_require_refresh`로 required generation을 올리고, 마지막 write 뒤
`tau_sobolev_mark_computed`로 computed generation을 맞춘다.

| writer | raw τ 대입 위치 수 | population/권한 | generation 경계 |
|---|---:|---|---|
| `compute_tau_sobolev` | 7 | 공유 `population_line_level_number_density(...LTE_TE...)` | 자체 require/mark |
| `nlte_update_tau_sobolev` | 2 | 공유 `nlte_tau_line_authority` + shell 권한 | 자체 require/mark |
| `apply_overlap_corrections` | 1 | 기존 τ에 overlap correction | 자체 require/mark |

CUDA NLTE solve는 raw τ를 직접 쓰지 않고 `nlte_update_tau_sobolev`를 호출한다.
종전 CUDA 중복 writer는 제거했다. 종전 post-NLTE 진단의 NaN/Inf→0 변이도 제거했고,
이제 비유한 committed τ가 하나라도 있으면 `[TAU-DIAG][FATAL]`로 종료한다.

`src/lumina_cmfgen.c`의 `opac.tau_sobolev` 대입은 `cmf_nlte_selftest` 내부의 지역 fixture로,
production `OpacityState` writer가 아니다. `tests/a2_08_signed_opacity_selftest.c` 대입도
독립 publication fixture다.

## 소비 불변식

A2-09는 대용량 line slab을 복사하지 않는 대신 다음 token view를 소비 시작과 종료에
각각 읽는다.

- raw τ required/computed generation
- A2-08 τ generation
- atom/A2-08 population generation
- plasma/A2-08 T_e generation
- NLTE population generation(0 또는 atom generation)
- A2-08 epoch와 요청 epoch

두 끝의 view가 다르거나 어느 한쪽의 내부 등식이 깨지면
`EMISS_TAU_MUTATED_DURING_CONSUME`로 private candidate를 폐기한다.

## 자동 gate

```text
$ make selftest-tau-writer-census
[TAU-WRITER-CENSUS][PASS] writers=3 compute_tau_sobolev=7 nlte_update_tau_sobolev=2 apply_overlap_corrections=1 cuda_writers=0
[TAU-WRITER-CENSUS][NEGATIVE-CONTROL][PASS] injections=4 detected=4
```

음성대조는 require 제거, mark 제거, 미등록 CPU writer 추가, 미등록 CUDA writer 추가를
각각 검출한다. A2-09 단위검사는 소비 중 raw τ 값을 바꾸고 정상 writer처럼 required,
computed, A2-08 τ token을 모두 올린 경우에도 end bracket이
`EMISS_STALE_OPACITY`를 반환함을 확인한다.
