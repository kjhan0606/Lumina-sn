# 형광 결함 공격 설계 (2026-07-06, post-ARTIS/SEDONA 대조)

## 확정된 진단 (다중 출처 수렴)

**증상**: emergent UV 42.9% (CMFGEN 23.8%), blue 5.8% (14.5%). 전 코드 중 최악.
**기제**: macro-atom이 흡수한 UV를 UV로 재방출 (UV-entry→UV-exit 99.5%, 형광수율 0.00%).
UV opacity는 충분 (τ_UV=106) — 흡수는 되나 광학으로 캐스케이드 안 함.

**교차코드 (StaNdaRT toy06, 동일 모델)**:
| code | UV% | blue% | 라인 처리 |
|---|---|---|---|
| CMFGEN | 23.8 | 14.5 | detailed RT |
| ARTIS | 14.6 | 15.0 | macro-atom + detailed rates (self-consistent MC J) |
| SEDONA(main) | 23.5 | 13.7 | 팽창 불투명도 + 2준위 ε 재분배 (+옵션 직접 MC 형광) |
| SEDONA-expansion | 28.5 | 14.0 | 팽창 + 2준위 ε (경량) |
| SEDONA-linetransfer | 18.1 | 15.2 | 직접 MC 형광 |
| TARDIS | 31.1 | 9.5 | macro-atom + dilute-bb rates |
| **LUMINA epay27** | **42.9** | **5.8** | macro-atom + frozen-hybrid |
→ **방법 무결**(ARTIS/SEDONA 형광 성공). Lumina-특이 결함. plasma도 CMFGEN급(ARTIS는 더 뜨거운데 UV 낮음 → 낮은 UV는 emergent RT 형광에서, plasma 온도 아님).

## 두 경쟁 가설 (증거 상충 — RUNG 0에서 판정)

- **STRUCTURE (구조 우세)**: 삼중검증 166121 — 결정론 S_l 경로에서 J_line/B(Te)≈1.0(필드 열적), super-thermal 전부가 상위준위 과점유(S_l/J_line 광학 median 3260×). U2 cascade walk: Si II exit 76-80% UV **J̄-independent**. → macro-atom 분기가 UV로 되돌림, 필드와 무관.
- **FIELD**: 2026-06-29 — Lumina 여기 rate가 binned-J→B(Te) 열화 → b_k=1. ARTIS는 W·B(ν,T_R) T_R decouple + 선=산란. → 필드에 super-thermal 펌프 없음.

두 가설은 공존 가능하나 **fix가 다름**. RUNG 0가 지배 root를 확정.

## Lucy ε-가중의 핵심 (구조 가설의 물리)

Lucy 2002 macro-atom: 상위준위 u에서
- p_emit(u→j) ∝ R_uj · **ε_uj=hν_uj** (방출광자 에너지) → **고주파(UV) 방출 선호**
- p_idn(u→j) ∝ R_uj · **ε_j=E_j** (하위준위 여기에너지) → 고여기 중간준위로 하강 선호

∴ p_emit/p_idn(u→j) = hν_uj/E_j. **바닥(E_j≈0)으로의 UV 전이는 ε_j→0 이라 p_idn≈0, UV 방출 강제.**
형광하려면 UV준위가 **중간준위로의 강한 전이**를 가져야(→ 캐스케이드). 원자데이터 불완전
또는 ε-가중 오구현이면 UV 재방출. **ARTIS transition_probabilities.npy로 white-box 대조 가능.**

## 공격 사다리

### RUNG 0 — 필드 vs 구조 판정 (오프라인, 무런 or 1 짧은 런)
결정론 Lucy 워크를 toy06 template + epay27 필드로 상위 UV운반체(Fe II/Co II/Ni II)에 대해 구성,
UV-exit 분율을 두 입력필드로 측정: (a) epay27 실제 필드, (b) 인공 super-thermal 펌프(T_R decouple).
- **게이트**: UV-exit(a)≈(b) [J̄-independent] → **STRUCTURE**. (b)≪(a) → **FIELD**.
- 병행: ARTIS transition_probabilities.npy vs Lumina 분기 white-box 대조 (UV준위의 p_emit/p_idn/
  중간준위 down-route 존재비). ε-가중 ARTIS-faithful 검증.

### RUNG 1 — 지배 root에 맞는 최소 fix
- **STRUCTURE 분기**: (i) Lucy ε-가중 audit vs artis macroatom.cc (ε_j vs ε_uj 배선 검증);
  (ii) UV준위→중간준위 전이가 원자데이터에 존재하나 outcompeted인지 vs 누락인지;
  (iii) 필요시 super-level 병합(상위준위 과점유 제거 — 166121 근원과 합류).
- **FIELD 분기**: macro-atom up-rate에 T_R-decoupled dilute-hot 펌프(ARTIS radfield 레시피).

### RUNG 2 — SEDONA-analog 독립 경로 (RUNG 1 정체시)
2준위 원자 ε 재분배 emergent (SEDONA-expansion analog). Lumina FINE_LINE_EPS 계보의 정공화.
28.5% 증명됨 = 경량 대안.

## Acceptance
UV≤30% (SEDONA-expansion급 우선, CMFGEN 24% 궁극), blue≥10%, plasma 무회귀,
**물리 항으로 명명된 단일 fix** (no 다중노브 튜닝). narrow-band+특징선 판정.

---

## RUNG 0 실행 결과 (2026-07-07, 오프라인 결정론 워크)

**단일사이클 Fe II 워크** (scripts/cascade_walk_fe2.py, shell 3, T_e=17683):
UV-exit vs 필드세기 k(=J̄/B(Te)): k=1→98.4%, k=2→97.2%, k=5→93.6%, k=10→88.4%,
k=30→76.3%. → **실제 필드(k~1-2)에서 UV 재방출 97-98%, 필드는 약한 레버.**
원인 = Lucy ε-가중 w_emit∝A_ul·hν(UV 방출) vs w_idn∝A_ul·E_l(바닥 E_l≈0 억제).

**원자데이터 동일성**: Lumina toy06 Fe II = ARTIS ddc10 Fe II **비트동일**(531662 라인,
파장·A_ul 동일, 둘 다 carsus/CMFGEN). → 단일사이클 분기가 ARTIS와 동일한데 ARTIS
emergent UV=14.6%. **∴ 형광은 전적으로 다중사이클 재처리에서.**

**다중사이클 모델** (scripts/cascade_multicycle.py, 실제 밴드별 τ): 단일사이클 98%→
**다중사이클 UV 29.8%**(CMFGEN 24% 근접). **재처리가 형광 기제** 확정. Lumina 실제
43% > 예측 30% = 재처리 부족.

**transport 검증**: cuda.cu:2369-2430 = 올바른 TARDIS full-forest 라인추적(모든 τ 누적
→ tau_event 교차 상호작용), 약한 라인 스킵 없음(MA-DUMP "skipped"는 진단 전용).
UV τ=약한 라인 숲(τ<1이 85%, 최강 1.4).

**근원 확정 (RUNG 0)**: macro-atom이 흡수 UV를 **위로 climbing 안 하고 바로 UV 방출** —
internal-up 펌프 부재. T_rad/T_e 프로파일: 외곽 형성층 s8-25에서 **T_rad>T_e**(dilute-hot
UV 펌프 존재), 그러나 epay27 use_j_nu=binned-J→B(Te) 열화로 펌프 사멸(2026-06-29 근원과
합류). ARTIS는 W·B(ν,T_R) T_R-decoupled 펌프 사용.

## RUNG 1 (실행중): epay29 = epay27 + LUMINA_MACROATOM_IUP_TRAD=1
internal-up rate를 binned-J → **B_lu·W·B(T_rad)** (ARTIS/TARDIS dilute-blackbody 펌프).
**예측**: UV 43%→~31%(TARDIS급). 확인시 펌프-부재 근원 확정 + 즉시 개선.
다음 rung = detailed rate(binned-MC 정공화, ARTIS 15% 목표).
