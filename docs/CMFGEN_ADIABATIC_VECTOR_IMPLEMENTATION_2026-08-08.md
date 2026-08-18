# CMFGEN V3 signed vector producer 구현 보고 — 2026-08-08

상태: **vector producer 및 model-free 검증 완료 / production publication 차단 유지**.

## 1. 구현 경계

- `src/cmfgen_adiabatic.c/.h`에 steady homologous
  `EVAL_ADIABATIC_V3`의 네 signed cgs component를 구현했다.
- 출력은 `temperature_gradient`, `velocity_divergence`,
  `electron_fraction_gradient`, `internal_energy_gradient`, `signed_total`과
  `cooling=max(q,0)`, `heating=max(-q,0)`이다.
- inner→outer Lumina 배열에서 `neighbor(0)=1`, `neighbor(s>0)=s-1`을 쓴다.
- `r=v*epoch`가 상대 `1e-10` 안에서 성립하지 않으면
  `CMFGEN_ADIABATIC_NON_HOMOLOGOUS`로 거부한다.
- 모든 입력을 먼저 검사하고 private candidate를 완성한 뒤 memcpy한다. 어느 오류에서도
  caller output은 byte-preserved다.
- `src/lumina_plasma.c`의 scalar A2-10 residual에는 연결하지 않았다. 따라서
  `A210_ADIABATIC_ELECTRON_TRANSLATIONAL_ONLY/A210_INCOMPLETE`와 atomic rollback이
  그대로 남는다.

## 2. 시험

`tests/cmfgen_adiabatic_selftest.c`가 다음을 검사한다.

1. 네 항이 모두 0이 아닌 3-shell analytic cgs known answer.
2. CMFGEN `T/10^4 K`, `V/(km s^-1)`, `R/10^10 cm`,
   `SCALE=10^9 k_B/(4pi)`, 최종 `4e-10*pi`의 cgs 왕복.
3. constant `T`, electron fraction, internal energy에서 `3P/t`만 남는 극한.
4. 바깥으로 감소하는 내부에너지가 음의 `WORK`, 즉 heating으로 보존되는 경우.
5. non-homologous, nonmonotone radius, zero atom density, NaN temperature,
   negative internal energy, one shell, result overflow의 7개 음성대조.
6. 모든 음성대조에서 output byte-preserving rollback.

결과 marker:

```text
[CMFGEN-ADIABATIC][SELFTEST] status=PASS shells=3 components=4
signed_heating=PASS cmfgen_unit_roundtrip=PASS boundary_stencil=PASS
negative_controls=7 atomic_rollback=PASS
production_publication=BLOCKED_ALL_SHELL_TRANSACTION
```

ASan+UBSan 및 `-Werror -pedantic` 독립 빌드도 rc 0이다.

## 3. 회귀 및 링크

- 기존 A2-10: `N1_N8=8/8`, `L6=BLOCKED_INCOMPLETE_ADIABATIC`, rc 0.
- Makefile header census: `declared=23 included=23 missing=0 stale=0`, rc 0.
- CPU 전체 링크 rc 0:
  `d0d30ef877b0c7255905414deab29dc902f1932604d1122c951e667b83a8408b`.
- OpenMP 전체 링크 rc 0:
  `a919a4c65789fc3f5d5147879dbd6e49b2aeddb5139d42c9721504f0e3f57870`.
- CUDA sm_80/sm_86/sm_90 compile+link rc 0:
  `516889617c91dd42f86887e0a856deec673ee816ec805c2ffe42609c18477de4`.
- model/GPU flight는 실행하지 않았다.
- 기존 source warning은 남지만 신규 module warning은 0이다.
- `git diff --check`: PASS.

새 TU는 root CPU/CUDA build와 `run_gate_battery.py`의 Z-validator, Z-tau,
Z-population, Z-canonical 및 Z-a2-10 build에 명시적으로 연결했다. 전체 gate battery나
모델 실행은 이 구조 단계에서 수행하지 않았다.

## 4. 판정과 다음 경계

Fable 판정
`docs/FABLE_VERDICT_SH_RADEQ_COMPLETE_ADIABATIC_2026-08-08.md`의
`IMPLEMENT_NOW=VECTOR_PRODUCER_ONLY`를 지켰다. 이 producer의 PASS는 완전 atomic
trial이나 flight PASS가 아니다.

다음 단계는 전 shell `T_e` 후보 하나에서 population/partition/`n_e`,
opacity/emissivity, internal energy, vector adiabatic, RE residual을 같은 private token으로
만드는 transaction이다. 이것이 없으면 `CMFGEN_COMPLETE` 승격은 금지한다.
