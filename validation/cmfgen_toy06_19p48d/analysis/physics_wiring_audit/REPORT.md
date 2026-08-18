# 물리 기작-배선 정합 감사 (2026-07-29; 운전석 스팟검증 N1·N2 통과)
전문은 세션 기록·원장 참조. 기준=parity42 RESOLVED CONFIG. 신규 위반 N1-N9 + 기지 K1-K20 지도.
## 신규 상위 5
- **N1[확정,오호출]** C2 상세 bf-추정기 배너("feeds the photoion R_bf")는 거짓 — NLTE_RATES_GEMM 기본 ON이라 행렬 R_bf=GEMM(K^T·J_C1 적합장, 250kK 레일·empty빈 포함) 소비. C2 실소비처는 MA iup_prob뿐(plasma.c:4174s; else분기 :14513s). 검증: R_bf ready ×12 + C2 armed 배너 공존.
- **N7[코드확정/인과잠정,d]** 행렬 준위별 Milne I_rec(:14513-14580) **무스핀게이트**(게이트 실장은 frozenin :5281-5309뿐) — Fe III metastable trap 준위군. 오프라인 A/B 가능.
- **N2[확정,b]** jbar 문턱 분열: 조립 JBAR_MIN=3(env) vs MA internal-up 하드코딩 count>=10(:3714) — 교차 3-9회 선은 인구·수송분기 비일관. (+N3: jbar EMA0.5 vs jblue raw)
- **N5[코드확정,d]** IUP-JBLUE stim 보정 인구=dilute-Boltz@T_rad 전셸핀(:3804-3820) — τ/S는 NLTE인데 stim만 성운·단일T. ARTIS 원배선 대조 미확인.
- **N6[구조확정]** binned 솔브 it>0 상태-사멸 — 판정 스펙트럼(formal/obs)은 cs·최종 τ/S, 상태 진화는 C1 MC장: **심판과 피고가 다른 장**.
기타: N4 MC 입력 세대혼식(χ_bf N−1 + τ N), N8 GEMM Z_eff 무 RATES_FIX 분기(잠복), N9 fb 지배에지 조회실패→공명 퇴화 무카운터.
## 지도 핵심
반복 = binned ALI(상태 무용) → radeq_simul(Kramers바닥Γ×lagged장 / frozenin α 비대칭짝 → T_e·ion 커밋) → 성운 τ 재기록[사례23: **in-config 유일 소비자=행렬 β**] → NLTE(GEMM R_bf·mode-3 β(τ_neb)·jbar EMA≥3·이온총량 락핀) → ctp/kpkt(pops N + jblue N−1 raw + χ_bf N−1 혼식) → MC → C1 발행.
활성 변형: **α 5원화**(frozenin[유일 게이트+DR]/recomb_per_level[무게이트]/kpkt-Kramers/행렬-Milne/DBFB) · Γ 2원화(Kramers바닥 vs GEMM) · **Z_eff 4관례** · NT 2처방 동시 · β 7변형(수치등가=무해) · S_l 폴백 3활성.
## 정정·확정 부수
- K4 정정[잠정]: DB_FB=1에서는 CAP_TOPION pair 열장부가 Wien 짝으로 닫힘 — 파괴는 DB_FB=0 else와 이온화 플럭스 장부 잔존.
- **K6 모순 발견: FROZENIN_DR=1 활성인데 코드 주석(plasma.c:79)은 "OFF가 정합구성"** — config-주석 모순.
- :8630 "does NOT call nlte_update_tau_sobolev" 주석은 구식(LINERES truthy 시 중복 호출 실재, 동일상수 무해).
- F1(x50)·축퇴분기·KPEMISS·mode-2: 이 config 휴면 확정. LINERES_JBAR=2 설정됨(스윕의 "미설정" 판독 정정).
## CMFGEN/ARTIS 대조
합법: 추정기 차기-이터 소비·0.5 감쇠·fine 최종1회(자백 축약). 결함: 소비자-선별 세대(사례23)·문턱/감쇠 분열(N2/N3)·커널 혼식(N4)·비네트워크 formal 짝(K5). CMFGEN은 α·Γ가 같은 σ의 DB 짝이라 K11 부류가 구조적으로 불가능.
