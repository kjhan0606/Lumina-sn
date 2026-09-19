# 운전자 기획 브리프 — A(인구 감쇠) 단계 (2026-09-19, Fable 계약 기안용)

표적 = **DET-POPDAMP-A**. 근거 = `docs/READOUT_DAMPING_DISCRIMINANT_2026-09-19.md`(오프라인 판별, 33bee9f). user 「A,B 모두 구현」.

## 배경
CONT 초열 폭주(`READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18`)는 lagged solve + **이온화 감쇠 0**(SIMUL 미가동, 인구 memcpy 덮어쓰기 `lumina_plasma.c:22236`/`:22240`)에서 났다. 오프라인 판별: 맵이 r 127→232 증폭, under-relaxation 정리상 감쇠는 **늦출 뿐 못 되돌림** ⟹ B(완전 선형화) 필요. 그럼에도 A 를 구현하는 이유 둘: ① **판별의 최종 in-vivo 확인**(감쇠 켜고 짧은 런 — 여전히 r 상승이면 B 확정) ② CMFGEN 식 **강건 층**(B 도 함께 씀).

## 수리안 (구현 Opus 4.8)
- env `LUMINA_POP_DAMP`(기본 1.0 = byte-동일). ON(w<1)이면 인구 커밋 `nlte_population_candidate_commit_bundle`(`lumina_plasma.c:22236` 이온·`:22240` 준위)에서 **`x_commit = w·x_candidate + (1-w)·x_public_prev`**. T_e 는 기존 radeq_damp 가 이미 감쇠하므로 이 단 범위 밖(이온·준위 인구만).
- **클램프 아님**(규약 판별식): under-relaxation 은 고정점 불변(수렴 시 candidate==public). 선례 `:13378`의 **하드 캡 `if(mixed>nel2)` 은 모방하지 마라**(그건 clamp) — 보존(n_ion≤n_element)이 필요하면 별도로 명시.
- selftest: w=1 → byte-동일; w<1 → 커밋값이 정확히 blend; 음성대조 = 발산 픽스처(candidate 가 public 대비 큰 이온화)에서 w<1 이 스텝을 축소함 시연.

## 기대치·게이트 (Fable 이 정식화)
- G-BND: 감쇠 켠 짧은 확인 런(w≈0.25, ITER 4-6, CONT 소스 ON)에서 **셸20 r(=f_III/f_II) 궤적**. 판별 규칙: r 이 반복마다 **여전히 단조 상승**하면 → 감쇠 불충분(B 확정, 분기 PD-B); r 이 꺾여 안정 고정점(f_II 회복)이면 → A 로 족함(PD-A, 오프라인 판별 반증).
- 오프라인 판별이 이미 PD-B 를 강하게 예측 — 이 런은 그 **확인**이지 새 예측 아님. 「구성상 참」 예측 금지.
- base = DR-on 385315 폐합본 위(깨끗한 재결합) 권장. ITER 수·헤지는 결정 항목.

## 순서
경량 사전감사(Q-GOAL·Q-LEAN) → 게이트 → **DR-on 폐합 후** Opus 4.8 구현 → Fable 코드 감사 → 빌드 → 확인 런 → 판정 → 종료감사. src 편집 1개 규칙 준수.
## 상한
A 가 폭주를 잡는가(오프라인은 불가 예측)까지. B 설계·tensor-core 는 별건. 확인 런 1회로 판별 in-vivo 확정.

## ★폐기 (user 2026-09-19 「A 를 할 필요가 있을까? B 로 바로 가야겠는데?」)
DET-POPDAMP-A 별도 단·확인 런 **취소**. 근거: 오프라인 판별의 under-relaxation 정리(감쇠는 M_r(r)>r 를 못 되돌림)가 강건해 in-vivo 재확인 불요. 감쇠(under-relaxation)는 폐기가 아니라 **B(완전 선형화)의 뉴턴-스텝 보정 상한으로 흡수**(CMFGEN 도 둘을 함께 씀). Fable A 계약 기안 중단(TaskStop). 표적 = B 직행.
