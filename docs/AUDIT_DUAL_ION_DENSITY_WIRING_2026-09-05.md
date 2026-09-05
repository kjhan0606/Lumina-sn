# 배선 감사 — 이온 단계 밀도의 이중 권위 (2026-09-05, 런 0회)

저자 = 운전석(Fable). 소스 = 작업본(HEAD `41d1e04` + 미커밋 provenance 패치; 인용 행은 `src/lumina_plasma.c` 작업본 기준).
대상 런 = 331050(승격)·R-II(반상태), 둘 다 `LUMINA_NLTE_ION_LOCK=0`·`PER_ION_RESCALE=0`·`NLTE_OPACITY_IONSTAGE` 미설정.
**감사이지 판정이 아니다.** 수리안을 내지 않는다(발견의 처분 = 기재).

## 0. 한 줄

한 런 안에 이온 II/III 밀도가 **둘** 있다. Saha 쪽 `atom->ion_number_density`(rate-SE 네뷸라 Saha, `:2742`)와 NLTE 준위해가 담은 Σ준위(쌍 총량은 Saha 합으로 정규화, II/III 분배는 행렬, `:19823`). 복사 수송(사상 선 τ·S·η, χ_bf 준위 항)은 **NLTE 분배**를, 전자 밀도·η_bf 재결합 항·비사상 선·쌍 총량은 **Saha 분배**를 쓴다. 두 분배는 반복마다 Fe II 기준 **+3.6%(승격)·+12.5%(반상태)** 어긋나며(판독 §9 S-역산), 되쓰기가 꺼져 있어 수렴해도 합쳐지지 않는다.

## 1. NLTE 집합 (331050 stdout 205–237)

쌍 = (II, III) × {Si, S, Ca, Fe, Co, Ni} = 12 이온, 13,015 준위. 사상 선 1,777,859 / 2,588,798. **IV 이상은 NLTE 밖**(Saha·LTE 전용). 비사상 810,939선의 주인은 IV+ 단계와 quarantine 원소.

## 2. 소비자 표

| 소비자 (함수 · 행) | 이온 II/III 밀도의 권위 | 비고 |
|---|---|---|
| 전자 밀도 `compute_electron_density` `:3048+44` | **Saha** `Σ charge·ion_number_density` | NLTE 행렬의 충돌·재결합 율이 받는 n_e 도 이것 |
| 쌍 총량 정규화 `nlte_solve_ion_shell` `:19823–19838`, `nlte_pair_total_density` `:1958` | **Saha** n_II+n_III 를 총량으로 | `lock` 거짓 ⟹ II/III 분배는 행렬 |
| II/III 분배 (준위해 안) | **NLTE 행렬** (준위별 광이온·재결합·충돌이온·3체, DDC 매크로 `pion/rec/cion/c3b` `:17420+`) | 판독 §9 의 이동 주체 |
| 사상 선 τ·S `compute_transition_probabilities` `:4075` (`gnl>=0` 분기 `+1146`), `nlte_update_tau_sobolev_with_authority` | **NLTE** 준위 pop | 하·상준위 모두 NLTE |
| 비사상 선 τ (같은 함수 `+1152·+1273`, `compute_tau_sobolev` `:7244`) | **Saha** n_ion × LTE(T_e) 분율 | IV+ 단계·quarantine 원소 |
| χ_bf 준위 항 `compute_bf_opacity_impl` `:8266` | **NLTE** pop (사상), 아니면 Saha×LTE (`:8274`) | 준위별 |
| η_bf Milne·유도재결합 `n_next_m` `:8235` | **Saha** n_{III} | ★한 과정(bf) 안에서 χ 는 NLTE·η 는 Saha — 혼합 |
| Saha 이온화의 Γ_phot `:2612` | canonical J × **LTE** 준위 분율 | NLTE pop 불사용 ⟹ Saha 분배는 NLTE 이동을 모른다 |
| CMF η_line·χ_line `cmfgen_assemble_impl` (`lumina_cmfgen.c:1789`) | `opac->tau_sobolev`·`line_source_S` ⟹ 사상 선은 **NLTE** | formal 스펙트럼이 보는 것 |
| A2-10 ROW `n_upper` (`a209_upper_population_for_tau`) | **NLTE_COMMITTED** (provenance 685/685) | 판독 1·2 의 입력 |
| helper `nlte_get_ion_density` `:10536` | NLTE 이온이면 **Σ NLTE pop**, 아니면 Saha | 호출처마다 권위가 갈린다 |
| 되쓰기 `nlte_writeback_ion_stage` `:3381` (`LUMINA_NLTE_OPACITY_IONSTAGE`) | 꺼짐 — stdout 0회 | 켜면 NLTE 분배가 Saha 배열로 이식됨 (그 안 `1e-300` floor 2건) |
| `radeq_simul_all` `:12098` 의 ION 완화 되쓰기 `:13245–13255` | 이 레인 **비활성으로 판단** (`[RADEQ][SIMUL]` 흔적 0; `LUMINA_SIMUL_ION_DAMP` 미설정; ⚠이름 grep 으로 호출처를 못 찾음 — `compute_radiative_equilibrium_te` 경유 추정, 미확인) | 켜지면 제3의 writer + `mixed > nel2` cap |

## 3. 함의 (관측 등급, 귀속 아님)

1. **반복 고정점의 정의가 둘로 갈린다.** Saha 분배는 (Γ_phot(LTE 가중, 동결 J), T_e, n_e) 만의 함수라 반복 간 ≲0.2% 변화(판독 §8-2). NLTE 분배는 <4 eV 장에 응답해 +3.6%/반복 움직인다(§9). 수렴 시점에 두 값이 같아질 기제가 없다(되쓰기 꺼짐, Γ_phot 이 NLTE 를 안 봄).
2. **전하 보존이 권위를 가로지른다.** n_e 는 Saha 분배로 닫히고, 복사 수송은 NLTE 분배를 본다. 어긋남의 크기 = 단계 II 몫 × 3.6% (승격) — II 가 소수 단계라 n_e 에는 작지만, 선 불투명도에는 그대로 3.6% 다.
3. **bf 과정 안 혼합**: χ_bf 는 NLTE 준위 pop, η_bf 의 재결합 항은 Saha n_{III}. 같은 bf 전이의 흡수와 방출이 다른 이온화 상태를 본다 ⟹ S_bf 가 정의상 B(T_e) 에서 벗어나는 인공 성분 가능(정량 미측정).
4. **CMFGEN 대조 관점**: CMFGEN 은 단계·준위를 한 행렬에서 풀고 하나의 n_e 로 닫는다. 이 이중 권위는 CMFGEN 발자국과의 구조적 차이이며, 「CMFGEN 재현」 목표에서 귀속 후보다.

## 4. 대장 후보 (조용히 기재)
- `compute_bf_opacity_impl:8225` `n_ion < 1e-30 → continue` (준위 건너뜀 guard).
- `radeq_simul_all:13253` `mixed > nel2` cap (이 레인 비활성).
- `nlte_writeback_ion_stage:3419-3420` `1e-300` floor ×2 (비활성).

## 5. 이 감사가 말하지 않는 것
수리 방향(되쓰기 ON, ION_LOCK, PER_ION_RESCALE 중 무엇이 CMFGEN 발자국인가)·정량 영향(스펙트럼)·수렴 거동. 모두 별도 사전등록 대상.

## 6. 분장 장부

| 단계 | 규약상 | **실제** | 위반 |
|---|---|---|---|
| 감사 설계·저작 | Fable | Fable(운전석) | — (판정 아님) |
| 실행 | 운전석 | 운전석(grep, 로그인 노드 비연산) | — |
| 판정·감리 | Fable fresh / Codex | 미발주 | user 결정 |
