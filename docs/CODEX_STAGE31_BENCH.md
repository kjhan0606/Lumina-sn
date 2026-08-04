# Codex A-S31 round 7 — 판별 벤치

상태: **UNRESOLVED**  
물리 판독: **UNRESOLVED-EXTRAP-POSITIVITY**

## 결론

요청된 수송 결함 대 χ,η 내용 결함의 이분 판정은 내릴 수 없다. 인증 payload 자체는 정상이나, 정본 KA solver의 one-sided radial face 외삽이 첫 sweep에서 음의 extinction을 만들고 fail closed했다. 이를 0으로 clamp하거나 boundary shell을 복제하면 acceptance와 입력 χ,η를 바꾸므로 수행하지 않았다.

실제 첫 실패는 `deterministic solve failed: LCMF_ENEGATIVE: chi radial reconstruction failed at outer face radial=50 frequency=331 ray=0 segment=0 substep=0 value=-1.4235983099001642e-20 interval=[0,0]`이다. 실행 시간은 0.062 s로, 장시간 계산 때문에 중단한 것이 아니다.

## 입력 독립 검증

- checker: `PASS: iteration=10 field_generation=10 post_damp=1 bytes=2416472`
- payload SHA-256: `94d75988034454f55fb6b130f04521f01c56f875cb22ef3a711850d7382ffa2f`; sidecar와 일치
- schema: 50 shell × 1000 bin, iter=10, generation=10, post_damp=1
- candidate/input ν grid max relative identity error: `1.221e-15`
- inner boundary: producer `cmf_solve_J`와 같은 explicit `Bν(T_inner=10020 K)` irradiation, amplitude scale 1.0; diffusion gradient를 추정하지 않음

### radial face 사전검사

| field | face | negative bins | 600–3000 Å negative bins | minimum | first bin / Å |
|---|---|---:|---:|---:|---:|
| chi_total | inner | 0 | 0 | 2.826566851e-14 | — |
| chi_total | outer | 40 | 36 | -2.692274598e-16 | 331 / 578.750 |
| eta_total | inner | 0 | 0 | 1.039793676e-36 | — |
| eta_total | outer | 37 | 37 | -1.143125249e-23 | 344 / 620.018 |

## s8 Jν 3중 대조

J_MC는 계약대로 sidecar payload의 `J_producer`다. CMFGEN은 RVTJ의 9610.017–10163.506 km/s 사이 log-J velocity interpolation 후 공통 1000-bin edge에 적분보존 평균했다. point interpolation은 쓰지 않았다.

| band [Å] | J_det/J_MC | J_det/J_CMFGEN | J_MC/J_CMFGEN | log10(MC/CMFGEN) | toward CMFGEN |
|---|---:|---:|---:|---:|---|
| B0 600–1000 | UNRESOLVED | UNRESOLVED | 33.764 | 1.52845 | UNRESOLVED |
| B1 1000–1500 | UNRESOLVED | UNRESOLVED | 32.3231 | 1.50951 | UNRESOLVED |
| B2 1500–2000 | UNRESOLVED | UNRESOLVED | 7.37579 | 0.867808 | UNRESOLVED |
| B3 2000–2500 | UNRESOLVED | UNRESOLVED | 6.91223 | 0.839618 | UNRESOLVED |
| B4 2500–3000 | UNRESOLVED | UNRESOLVED | 16.2922 | 1.21198 | UNRESOLVED |
| BALL 600–3000 | UNRESOLVED | UNRESOLVED | 11.9771 | 1.07835 | UNRESOLVED |

### available spectral norm

| band | median log10(MC/CMFGEN) | p10 | p90 | positive pairs | zero/excluded |
|---|---:|---:|---:|---:|---:|
| B0 | +1.489393 | +1.224499 | +1.658454 | 97 | 0 |
| B1 | +1.547098 | +1.077093 | +1.821144 | 76 | 0 |
| B2 | +0.828737 | +0.772125 | +1.060727 | 55 | 0 |
| B3 | +0.780508 | +0.739546 | +0.858911 | 42 | 0 |
| B4 | +0.984745 | +0.711919 | +1.507408 | 34 | 0 |
| BALL | +1.258953 | +0.771584 | +1.689085 | 304 | 0 |

J_det가 없으므로 candidate가 들어가는 spectral quantile은 UNRESOLVED다. 비교 가능한 baseline에서 J_producer는 CMFGEN보다 모든 사전등록 대역에서 높다; 이 사실만으로 수송/내용 원인을 분리할 수는 없다.

## Γ D-lane

기존 `w3_gamma_triple_compare.py`의 grid/C1/C2 loader, EDDFACTOR/RVTJ 적분보존 평균, within-SL fraction·σ·threshold·route 및 `4πσJ/(hν)` quadrature를 import해 재사용했다.

| target | Γ_MC B [s⁻¹] | Γ_CMFGEN C [s⁻¹] | Γ_det D [s⁻¹] | log10(B/C) | toward CMFGEN |
|---|---:|---:|---:|---:|---|
| Fe III C48 lump (idx 201) | 4.363858095e+02 | 2.807174861e+01 | UNRESOLVED | +1.191601 | UNRESOLVED |
| S II SL4 (idx 4) | 3.310150073e+01 | 4.748076950e-01 | UNRESOLVED | +1.843330 | UNRESOLVED |

Fe III와 S II의 baseline MC excess는 각각 위 log10(B/C)만큼 재확인됐다. D-lane은 J_det 부재로 계산하지 않았으며 J_producer나 C1 J를 대신 넣지 않았다.

## Acceptance와 판독

- sidecar/schema/checksum/epoch: **PASS**
- CMFGEN integral conservation: `1.000000000000000` (**PASS**)
- 6 band row와 2 Γ row의 baseline provenance: **PASS**
- candidate transport residual ≤1e-4, finite/nonnegative, clamp=0: **UNRESOLVED** (solve가 장 생성 전 fail closed)
- 최종 bench acceptance: **UNRESOLVED**

따라서 사전등록한 ‘CMFGEN 쪽 이동 → 수송 결함’과 ‘MC UV 과잉 재현 → χ,η 내용 결함’ 중 어느 가지도 선택하지 않는다. 현재 분해는 **입력 payload 정상 / shell-center field 정상 / boundary face reconstruction 부호 실패**다. 다음 조치는 acceptance를 완화하는 것이 아니라, 정본 설계의 boundary half-cell 계약(원설계 §3.1의 constant extension과 현재 KA 인증본의 one-sided extrapolation 중 어느 것이 parity bench 정본인지)을 별도 승인해 재실행하는 것이다.

## 재현 명령

```bash
sha256sum /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10
python3 scripts/cmf_chieta_check.py /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc scripts/stage31_cmf_field_driver.c \
  src/lumina_cmf_field.c -lm -o /tmp/stage31_cmf_field_driver
/tmp/stage31_cmf_field_driver /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10 /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10.manifest.json 8 16 10020 1 /tmp/stage31_jdet.tsv
python3 scripts/stage31_cmf_field_bench.py --frozen /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10 \
  --report docs/CODEX_STAGE31_BENCH.md \
  --status-json docs/s31_results/stage31_bench_round7.json
```

신규 모델/GPU run, 기존 `src` 수정, acceptance 변경, 커밋은 없었다.
