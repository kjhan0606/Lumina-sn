# 설계 — F4(수렴) 런 0회 결론 + D-C/D-D 수리 방향의 CMFGEN 대조 설계 (2026-09-05)

저자 = Fable(운전석). **설계이지 사전등록이 아니다.** 사전등록은 계약 단위로 따로(Fable, 프리플라이트 통과 후).
근거 = `docs/PREREG_DUAL_AUTHORITY_2026-09-05.md` §1(CMFGEN 발자국)·`docs/AUDIT_DUAL_ION_DENSITY_WIRING_2026-09-05.md`·`docs/EVAL_FORK_DUAL_AUTH_2026-09-05.md`(Q7 F4).

## 0. F4 「수렴 먼저」 — 런 0회로 살펴본 결과

**질문**: 이온 II 총량의 반복당 이동(승격 +3.6%/반복, 반상태 +12.5%)은 2반복 스킴의 과도 상태인가, 감쇠하지 않는 표류인가.
**탐색** [저자 실행]: `/gpfs/kjhan/lumina` 깊이 4 의 전 런루트에서 `outer_iterations.txt` ≥ 3 은 48개(4 = 40, 20 = 8), 전부 08-09~16 의 `det1234_*`. 그 중 stderr 에 `iter=2 phase=view` 를 가진 것 **0** — 모두 iter 0 에서 사망(rc=1)이거나 R7 루프 미도달. 반상태·승격 어느 DET 런도 3회 이상 반복을 완주한 적이 없다.
**결론**: 감쇠는 **미측정**이다. 반복당 이동 한 점(0→1)만 있으므로 F4 는 런이 필요하다.
**F4 런 설계(후보, 사전등록 전)**: base 331050 과 동일 물리, `LUMINA_PURE_CMFGEN_ITER=4`(stager 는 현재 2 를 강제 `:436` ⟹ 모드 신설), 계측 = 반복별 S-역산 바닥 Δlog n_k (k=1,2,3) · Δlog N_SL · n_e. 분기 = 감쇠(|Δ₃/Δ₁| < 0.5) / 정체 / 발산. 비용 ≈ 2×base(5~6 h). **332463(단일 변수 대조) 착지 뒤** E-3 규칙(<1% 이면 F4 우선)에 따라 사전등록.

## 1. 발자국과 격차 (요약)

CMFGEN: 한 깊이점에서 전 종·전 단계·전 준위 + ED + T 를 **한 BA(Newton) 계**로 동시 갱신. 종 수 보존(abundance eq)·전하 보존(`STEQNE`)이 행렬의 행. 이온 단계 밀도는 준위 합이며 따로 없다.
Lumina: Saha 층(`compute_ion_populations_shell`)이 단계 총량·n_e 를 정하고, 준위 행렬은 쌍 총량만 받아 II/III 를 재분배하며, 되돌려주지 않는다. n_e 는 행렬 안에서 **고정 매개변수**(`nlte_assemble_rate_matrix_impl:+33 n_e = plasma->n_electron[shell]`), 보존 행은 SE 행 둘을 대체해 Σ 를 쌍 총량에 핀(`:17282`, `:19034`).

## 2. CMFGEN 대조의 실물 표적 (있는 것)

| 표적 | 위치 | 내용 |
|---|---|---|
| 공개 StaNdaRT 이온 분율 | `data/standart_data1/toy06/ionfrac_{fe,co,ni,si,s,ca}_toy06_cmfgen.txt` `#TIME: 19.480` | 100 속도구간 × 단계 0–5 |
| 공개 phys | `…/phys_toy06_cmfgen.txt` | T·ρ·n_e·n_atom per 속도구간 |
| 자체 CMFGEN 런 | `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/POPSIL POPSUL …`, `extract/parse_ionfrac.py` | 준위 population |
| Lumina Saha 단계 밀도 | `LUMINA_ION_POP_DUMP_ITER=1` → `lumina_ion_pops_iter%02d.csv` (`shell_id,Z,stage,n_ion`) | 전집 등록됨, 아직 어느 런도 안 켬 |
| Lumina NLTE 단계 총량 | `LUMINA_NLTE_LEVEL_DUMP`(`:20760`) 또는 S-역산(관측 준위만) | |

셸 대응 = 속도(`shell.csv v_inner/v_outer` ↔ `vel_mid`), 원소 밀도 = `ρ X_Z / m_Z`.

### 2-1. 첫 숫자 (런 0회, shell 0, 조건 불일치 명기)

| | Lumina 331050 (진단 레인) | CMFGEN 19.48d @3887 km/s |
|---|---|---|
| T | **10020 K (REQUESTED_TE 핀)** | **19411 K** |
| n_e | 3.21e9 | 5.81e9 |
| n_atom | 1.61e9 | 1.92e9 |
| 평균 전하 | 2.0 | 3.0 |
| Fe II 분율 | ≈ **2.2e-4** (S-역산 바닥 5.55e3 cm⁻³ × U/g₀ LTE 외삽 = 3.7e4; n_Fe = ρX_Fe/56u = 1.65e8) | **6.8e-12** |
| Fe III / IV | 미발행(로그 없음) | 2.2e-4 / **0.972** |
| 방출 주인(A2-10 LINE-ION-OWNER shell 0) | III 지배(10.5), II 5%, IV 3e-8 | — |

⟹ 7자릿수 차이는 **T_e 핀(10 kK 대 19 kK) 이 만든 것**이지 배선 격차의 크기가 아니다. **진단 레인은 배선 시험대이지 CMFGEN 상태가 아니다.** 이온화의 CMFGEN 대조는 T_e 를 CMFGEN 값에 맞춘 레인(7월 F3-T 쌍둥이: `LUMINA_FIXED_TE_PROFILE` 을 CMFGEN T(v) 로)에서만 의미가 있다.

## 3. 계약 사다리 (계약 1개 = 커밋 1개, offline-first)

| 단 | 계약(by-construction) | 불변식(게이트) | 음성대조 | 런 |
|---|---|---|---|---|
| **D-C-1 n_e 폐합** | commit 후 n_e = Σ_q q·n_q, NLTE 이온은 Σ준위, 비 NLTE 는 Saha | `\|Σ q n − n_e\|/n_e ≤ 1e-12`(commit 직후); 반복당 n_e 변화 계수 | Σ 에 준위 하나 누락 주입 → FAIL | CPU selftest(합성 2단계 원자) + 판정런 1 |
| **D-C-2 Saha 층이 NLTE 를 소비** | `compute_ion_populations_shell:2612` 의 준위 가중을 LTE 분율 → NLTE 준위 pop 으로 | 수렴 시 Saha 분배 = NLTE 분배(`writeback` 이 항등이 됨): `\|n_II^Saha/n_II^NLTE − 1\| ≤ ε` | 가중을 LTE 로 되돌리면 FAIL 시연 | CPU selftest + 판정런 1 |
| **D-D 단일 계** | 쌍 총량 = ρX/m − (비 NLTE 단계 Saha 합)(종 수 보존 행), ED 미지수 + 전하 보존 행, Newton 선형화(CMFGEN BA) | 종·전하 보존 잔차 = 0 by construction; 합성 원자에서 해석해와 일치 | 전하 행 제거 시 잔차 ≠ 0 시연 | GPU 솔버 차원 +1(cuBLAS batch N+1), 판정런 1 |

순서 후보: (i) D-C-1 → D-C-2 → D-D (증분, 각 단이 발자국 쪽으로 한 걸음) (ii) D-D 직행(정확, 큼). D-A(되쓰기 ON)는 D-C-2 의 부분집합이라 별도 단으로 두지 않는다.

## 4. CMFGEN 대조 측정 계획 (D-C-2 이후부터 의미)

1. 레인 = **쌍둥이 T_e**: `LUMINA_FIXED_TE_PROFILE` = CMFGEN `phys` 의 T(v) 를 Lumina 셸에 보간(단조·속도 정합 검증). J 는 Lumina 자체.
2. Lumina 산출 = `ION_POP_DUMP_ITER=1`(Saha) + 준위 덤프 Σ(NLTE) — D-C-2 이후 둘이 같아야 하며 그 차이 자체가 게이트.
3. 비교량 = 원소별 단계 분율 f_q(v), n_e(v). 잣대 = CMFGEN ionfrac·phys 19.48d, 속도 보간. 보고 형식 = **CMFGEN 어긋남 지도**(양×위치×크기×원인 확정/기각/미결) — 게이트 나열 금지.
4. 기대치는 사전등록에서: 외곽(> 8,000 km/s) 은 정상상태 근사 차이(CMFGEN 시간의존 항)로 갈릴 수 있어 별도 표기.

## 5. 비용·순서 제안 (결정은 user)

| | 코드 | 검증 | GPU |
|---|---|---|---|
| D-C-1 | 소(수십 행) | CPU selftest | 판정런 1 (~3 h) |
| D-C-2 | 소~중 | CPU selftest | 판정런 1 |
| D-D | 대(솔버 차원·Newton) | 합성 원자 해석해 | 판정런 1 + 수렴 확인 |
| F4 | stager 모드 1 | — | 1 (5~6 h) |

제안 순서: 332463 착지·E-3 판독 → (E-3 ≥1%) D-C-1 사전등록 / (<1%) F4 사전등록. 어느 쪽이든 CMFGEN 대조 런은 쌍둥이 T_e 레인으로만.

## 6. 이 설계가 말하지 않는 것
D-C/D-D 가 스펙트럼을 CMFGEN 에 가깝게 만든다는 것. 진단 레인의 숫자(§2-1)는 조건 불일치라 「Lumina 가 7자릿수 틀렸다」로 읽지 마라.
