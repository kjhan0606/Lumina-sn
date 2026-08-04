# toy06 BC 재설계 — 점-광구 경계 → 확장-decay 경계

**작성 2026-07-14. 근원: 상속받은 TARDIS 점-광구 BC가 양 경계에서 실패.**

## 1. 확정된 근원 (못박음)

`scripts/build_toy06_epoch.py`가 toy06 참조를 TARDIS 점-광구로 생성:
- **광구 v_phot=3900 km/s, R_inner=6.564e14 cm** (line 125 `v_inner=v[i_phot]`).
- **T_inner=10020 K** = (L_inner/(4πR²σ))^0.25, L_inner=3.095e42 erg/s (line 140).
- **W = 0.5(1−√(1−(R_inner/r)²))** 점-광구 기하 희석 (line 194) — 검증: R_implied=2r√(W(1−W))가 s5-49 전셸 6.564e14 상수(std/mean 0.068).
- **T_rad = T_inner·W^0.25** (line 195, 에너지등가온도) → 이후 `LUMINA_TRAD_COLOR_FIX`가 전셸 T_rad=10470 고정.

Lumina 격자는 **v=4264 km/s(s0)에서 시작** — 광구 아래(<3900, CMFGEN 22-25kK 뜨거운 코어)는 **절단**되고 T_inner=10020 흑체로 대체.

## 2. 두 경계 결함 (속도-정정판)

**내부 BC (s0-4 모델셀):** T_inner=10020 흑체가 s0를 아래로 당김 → L/C 0.84(16% 냉, modest). Fe/Co III vs CMFGEN IV(과소이온화). 진짜 뜨거운 코어(<3900)는 미모델링. **효과 modest but IGE 형성층 오염.**

**외부 BC (far-edge s40-49):** W→0.003(점광원 희석) → 필드 기아 → RE 솔버 뜨거운 쌍안정근 낙하 → T_e 2-3.3× 과열(30-84kK vs C 13-25kK). **지배적 온도 결함.** 기존 task#15(무패킷 hot root, champion epay22)와 동일 대상의 BC-렌즈.

**중간(s5-39):** L/C 1.0-1.25, 양호 — 국소 decay+RE 지배, 경계서 멂 = BC-문제 서명.

**★별개(BC 아님): Ni-at-V 레일** s0-8 전역 4.0(냉·온 셸 무관) = T-무관 Ni 재결합/커버리지 데이터버그. 별도 트랙.

## 3. 목표 패러다임 (ARTIS/CMFGEN)

핵심 원리: **장·온도를 imposed(고정 T_inner + 점광원 W)가 아니라 derived(transport+decay)로.**
- ARTIS: 내부경계 온도 無; 전셸 decay deposit; T_e=열평형 root-find, T_R/W=복사장 estimator 매 iter 갱신; 외부=자유탈출.
- CMFGEN: 내부=τ_Ross=50 확산경계(광도 지정, T 부양); 외부=I⁻=0 자유탈출; 확장 decay(γ MC+Spencer-Fano).

## 4. 설계 (단계·게이트·falsifier)

각 게이트 기본 OFF, byte-identity 유지. 판정=이벤트로그 표준 + 3코드 대조.

**Stage A — 내부경계 광도 정합 (최소·최우선 falsifier):**
- 가설: T_inner=10020이 낮아 s0-4 IGE 과소이온화. 
- 테스트 게이트 `LUMINA_TINNER_SCALE=f`: T_inner→f·T_inner 스캔(f=1.3,1.6,1.9 → 13/16/19kK). 
- 예측: f 올리면 s0 Fe/Co III→IV 회복, s0 T_e→CMFGEN 18760 근접. **falsify: f 올려도 Fe/Co III 불변 = 내부BC 무죄, 딴 원인.**
- 정식 fix(통과시): L_inner를 광구 아래 실제 decay 적분으로 재계산(build_toy06_epoch 수정) 또는 격자를 <3900까지 확장+decay deposit.

**Stage B — 외부 필드 유도 (far-edge, 지배결함):**
- 가설: 점광원 W→0이 far-edge 필드 기아. 
- 게이트 `LUMINA_W_FROM_JFIELD=1`: dilute-Boltzmann W를 점기하 대신 **실제 J에서 유도**(W=J_bol/(σT_rad⁴/π)), 확장소스 반영. T_rad도 J의 색온도로.
- 예측: far-edge W 상승 → 필드 회복 → RE 냉각근 유지 → T_e 2-3.3×→<1.4×. **falsify: W 올려도 hot root 유지 = 솔버 근선택(EPAY 계열)이 진범, BC 아님.**
- 주의: 기존 hot-root 작업(epay22, TE_PIN, LOWEST-ROOT)과 상호작용 — 중복/충돌 점검 필수.

**Stage C — freeze 해제:** Stage A/B 통과 후 `TRAD_COLOR_FIX` off로 T_rad 유도. 단 원래 외곽 hot-strip 재발 감시(이게 freeze 도입 이유) — Stage B가 그걸 대체해야.

**별도 트랙 — Ni 데이터:** Ni IV/V 재결합률(frozenin_alpha_rr)·phot 커버리지·NLTE 단계 감사. BC와 독립, 병렬 가능.

## 5. 판정 기준
"고치면 스펙트럼 움직임"이 최종 falsifier — BC fix 후 3코드 T_e 프로파일 수렴 + IGE 이온화 정합 + emergent 스펙트럼(재고정 4기준). 단계별 사전등록 예측을 각 런에 명시.

## 6. 순서
Stage A(최소 1파라미터 스캔, 즉시) → 결과따라 B → C. Ni 트랙 병렬. 각 단계 이벤트로그 ON.
