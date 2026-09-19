# 설계 — Path B(완전 선형화) 아키텍처 정찰 (2026-09-19)

운전자(Claude) + 정찰(general-purpose). 근거 = CONT 폭주(`READOUT_CONT_NLTE_SUPERTHERMAL`)·감쇠 판별(`READOUT_DAMPING_DISCRIMINANT`, A 불가·B 필요). user 「B 직행·tensor core」.

## 1. 근인 배선 (file:line, 동결 트리)
- 경계 = `cmfgen_commit_jnu`(`lumina_cmfgen.c:3873`, 호출 `:8095`) → `nlte->radfield_view.J_nu`. 하류 읽기 전용.
- 인구 solve 가 J 소비: Γ_phot = Σ4πσ_bf·J/(hν)dν(simul `plasma.c:12967,12991`; nebular `parity_gamma_phot_checked:2612`→`nlte_bf_gamma_canonical:475`), 선율 R_lu=B_lu·J̄(`:20585`), η_bf(`bf_get_eta`, upper-ion n_i*).
- **디커플: `radeq_simul_all` 에서 sh.Gph 가 T_e 이분 전에 동결** — 이온화가 스스로 올라가도(중성 불투명도↓·EUV 얇아져 J↓ 이어야) **J 고정이라 복원 되먹임 없음** = 폭주.

## 2. ∂J/∂N — MVP 는 스칼라 사다리 선형화, 완전판은 행렬
- CMFGEN: `do_var_cont.f` 가 dCHI·dETA 를 BA 야코비안에 접어 dJ/dN(`:5,143-146`). = ∂χ/∂N·∂η/∂N 경유.
- Lumina 스칼라 비 사다리(`parity_rate_se_ratio_checked:2768` r=num/den)는 나눗셈 — **닫힌형 미분 가능**(에너지/전하 행 형제 미분 이미 존재 `:13819-13940`). ⟹ **스칼라 사다리를 J·n_e 에 대해 선형화 가능(이온단계 총량)**.
- ⚠**이중 인구 권위**: 단계 총량=스칼라 사다리 / 준위 분포=N×N rate 행렬(`nlte_solve_ion_shell:19621`). η_bf 는 둘 다 의존. 스칼라만 미분하면 행렬-해 준위와 불일치 ⟹ **완전 Path B = 셸별 rate 행렬(STEQ/BA)에 bf 소스 결합; 스칼라 선형화는 MVP 프로브**.

## 3. tensor core (정정)
기존 GEMM 3종 전부 **FP32 저장·TF32 tensor-core 조립**: bf_gemm(χ_bf=σ·N, `bf_gemm.cu:130`)·nlte_gemm(R_bf=Kᵀ·J, `:486`)·assemble(bb 행렬, **FP64** `nlte_assemble.cu`). ⟹ **tensor core = 야코비안 조립**(∂χ_bf/∂N=σ_bf 공짜). **N×N 해는 FP64**(뻣뻣·이미 0.4% 비용, TF32 위험). 정확 multi-GPU 솔버는 Λ\* 미노출(binned `formal_solve_bin` 만).

## 4. 단계 계획
- **① 오프라인 프로브(런0회)**: 대각 ∂Gph/∂N_ion 복원 항(=∫4πσ/(hν)·λ*·∂S/∂N·dν)을 CONT 발산 셸(S/Si/Ca, `lumina_ion_channels.csv` 계측)에 넣어 단일-셀 고정점이 유한(CMFGEN 근처)인지 반복. 유한 ⟹ MVP 정당; 발산 ⟹ 셸간 tri_* 필요.
- **② MVP**: `simul_ladder`/`radeq_simul_all`(`plasma.c:11646,12223`)에 대각 복원 항(env `LUMINA_POP_DJDN`, off=byte-동일), 기존 `lambda_star`(`:13659`) 재사용, **새 커널 0**. 스텝 제한 `LUMINA_POP_MAXCH`(기존 `SIMUL_ION_DAMP:13356` 자리). 한 함수 영역+2 게이트, 한 파일.
- **③ 완전 Path B**: 셸간 tri_* 결합·`coupled_newton_solve_all`(`plasma.c:16863` 스텁) 채우기·인구 rate 행렬 bf 결합·이중 권위 해소. 3-4 파일·야코비안 조립 커널. MVP 가 대각 부족일 때만.

## 5. 최대 위험 3
① 정확 multi-GPU 솔버 Λ\* 부재 → MVP 는 binned 챔피언 경로 먼저. ② 이중 권위 결합 뉴턴 진동 → 스텝 제한 필수·오프라인 프로브 선행. ③ tensor-core 정밀도 → 조립만, 해 FP64.
## 6. 다음
① 오프라인 프로브 즉시(런0회). 결과로 MVP(거의 국소) 대 완전 결합 갈림.
