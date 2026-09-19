# 운전자 기획 브리프 — DET-DJDN-MVP (대각 ∂J/∂N 커플링 MVP) (2026-09-19, Fable 기안용)

표적 = **DET-DJDN-MVP**. 근거 = `docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md`(설계 정찰, 608690e)·`docs/READOUT_DAMPING_DISCRIMINANT_2026-09-19.md`(A 불가·B 필요). user 「B 직행·tensor core」·「승인」(09-19).

## 배경 (설계 정찰 확정분)
근인 = `radeq_simul_all`(`lumina_plasma.c:12223`)에서 Γ_phot(`sh.Gph`)가 T_e 이분·이온화 사다리 전에 **동결** ⟹ 이온화가 스스로 올라가도 J·Γ_phot 복원 되먹임 없음 = 폭주(CONT). Lumina 에 ∂J 연산자(대각 `lambda_star` `:13659`·삼중대각 `tri_*`)와 결합뉴턴 스텁(`coupled_newton_solve_all:16863`) 이미 존재; 빠진 건 **∂J/∂N(인구) 채널**. `∂χ_bf/∂N = σ_bf` 는 이미 device tensor-core bf_gemm 행렬(공짜).

## 표적 (MVP — 대각만, 커플링 가설 시험)
`simul_ladder`/`radeq_simul_all` 의 이온화 balance 에 **대각 복원 항** 추가: 이온화가 오르면 Γ_phot 를 끌어내리는 첫차 항. 이게 **대각 커플링만으로 CONT 폭주를 유한 고정점으로 잡는가**를 시험한다. 잡히면 MVP 정당(거의 국소); 안 잡히면 셸간 tri_* 결합(더 큰 Path B) 필요.

## 수리안 (구현 Opus 4.8)
- env `LUMINA_POP_DJDN`(기본 0 = byte-동일·출력 무변화). ON 이면 이온화 사다리(`simul_ladder:11706` 고정점 또는 `simul_r1:11767` 뉴턴 잔차)에 대각 복원 항:
  `G_eff(N) ≈ G_lag + (∂G/∂J)·(∂J/∂S)·(∂S/∂N)·ΔN_ion`, ∂J/∂S = 셸-대각 `lambda_star[s,b]`(기존, `radeq_set_tri_response`), ∂χ_bf/∂N = σ_bf, ∂η_bf/∂N = n_i* 항. 이온화↑ ⟹ G↓(복원).
- **binned 챔피언 경로만**(정확 multi-GPU 솔버는 Λ\* 미노출). 새 커널 0(기존 lambda_star·σ_bf 재사용).
- 스텝 제한 `LUMINA_POP_MAXCH`(기존 `SIMUL_ION_DAMP:13356` 자리에 CMFGEN MAXCH 식 분율 캡). off=inert=byte-동일. **감쇠(폐기된 A)가 여기로 흡수**.
- **클램프 아님**: under-relaxation·복원 항은 고정점 불변(수렴 시 ΔN→0). floor/하드캡 신설 금지.
- selftest: DJDN off → byte-동일; on → 복원 항이 정확히 lambda_star·σ 곱; 음성대조 = 이온화 증가 픽스처에서 G_eff < G_lag(복원 부호 시연).

## 기대치·게이트 (Fable 정식화)
- G-BND: DJDN 켠 짧은 확인 런(CONT 소스 ON, ITER 4-6, 챔피언 binned)에서 **셸 12-24 S/Si/Ca II→III 비 r 궤적**. 분기: **MVP-A**(r 유한 고정점·CONT 반대 안정 ⟹ 대각 충분) / **MVP-B**(r 여전히 단조 상승 ⟹ 대각 불충분·셸간 필요) / MVP-X(미시연).
- base = DR-on 385315 폐합본. ITER·헤지·MAXCH 값은 결정 항목.
- 「구성상 참」 금지; 오프라인 판별은 A(순수 감쇠) 불가만 말했지 대각 커플링 충분성은 미예측 — 이 런이 첫 측정.

## 순서·상한
경량 사전감사(Q-GOAL·Q-LEAN) → 게이트 → **DR-on 폐합 후** Opus 4.8 구현 → Fable 코드 감사 → 빌드 → 확인 런 → 판정 → 종료감사. src 편집 1개. tensor core = 야코비안 조립(σ_bf, 이미 TF32)·N×N 해 FP64(이 MVP 는 N×N 해 무변경). 상한: 대각 커플링이 폭주를 잡는가까지; 완전 Path B(tri_*·결합뉴턴)는 MVP-B 시 별건.
