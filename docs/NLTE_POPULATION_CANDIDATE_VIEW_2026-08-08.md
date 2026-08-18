# NLTE all-shell population candidate view — 2026-08-08

## 구현 범위

`src/nlte_population_candidate.c/.h`는 공개 NLTE/atomic/plasma 객체를 직접 바꾸지 않고
trial 계산이 사용할 shallow object view를 만든다. immutable map/table/radiation view는
빌리고, 다음 mutable material state는 private allocation으로 복제한다.

1. all-shell trial `T_e`
2. ion populations
3. NLTE level populations
4. electron density
5. partition functions
6. within-super-level fractions

객체 안의 partition/within-SL stamp, population error ledger, counter 및 generation token은
shallow copy의 scalar이므로 공개 객체와 분리된다. 이 API는 solver 실행이나 publication
commit을 허가하지 않는다.

## fail-closed 계약

- shell/layout/source pointer/generation/positive finite temperature를 allocation 전에 검사.
- 크기 곱셈 overflow를 거부.
- 일부 allocation 실패 시 후보 allocation만 해제.
- 성공·실패 어느 경우에도 공개 객체와 공개 배열은 byte-preserved.
- `nlte_population_candidate_free`는 반복 호출 가능.

## 검증

```text
NLTE-CANDIDATE status=PASS
private_arrays=6
stamps=PRIVATE errors=PRIVATE generations=PRIVATE
within_sl_rollback=PASS public_mutations=0 double_free=PASS
solver_core=NOT_YET_EXTRACTED
```

- normal selftest PASS.
- `-Wall -Wextra -Werror -pedantic` + ASan/UBSan/leak detection PASS.
- Makefile header census: declared=24, included=24, missing=0, stale=0.
- CPU/OpenMP/CUDA full compile+link PASS; model/GPU execution은 하지 않음.

## 남은 경계

현 `nlte_solve_all`은 CE core 안팎에서 diagnostic file, EW process-global counter,
runtime manifest, raw tau/source와 file-scope tau authority를 건드린다. 따라서 후보 view를
기존 wrapper에 넣어 호출하는 것만으로 atomic trial이 되지는 않는다. 다음 단계는 동일
A2-07 계산 코어에 optional side-effect sink를 주고, legacy wrapper와 trial wrapper가 그
코어를 함께 쓰도록 추출하는 것이다.
