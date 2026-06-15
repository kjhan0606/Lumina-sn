# LUMINA-SN: top-stage(III) 연속체-앵커 부재 — fix 3안 비교설계 (2026-06-15)

## 확정된 메커니즘 (삼중검증 + rate-budget 확정)
DDC15 0.976d self-test. T_e/n_e는 gold와 0.5% 일치하나 emergent 스펙트럼 too-blue.
선소스 S_l이 광학서 super-thermal(median S_l/B≈1947). 주범 = **2가 이온(III) O III/S III/C III/Al III 들뜬 준위.**

NLTE 타깃(plasma.c:2835 NLTE_TARGET_Z/ION): 전 원소 최상위 = III(ion=2), **IV 없음.**
- III는 항상 페어 (II,III)의 **상위 이온** → bf 블록은 하위(II) 준위만 다룸. III는 ground_hi로만 등장.
- ⇒ **III 들뜬 준위는 어떤 bf 블록에도 안 나옴 = 연속체(광이온화/재결합) 결합 전무** (rate-budget rec 덤프로 확인: O III 들뜬준위 bf row 0개).
- III 들뜬 준위는 오직: II→III 광이온화(III ground로) + III 내부 bb선으로만 채워짐.

rate-budget (O III 광학선 3728개, sh8, n_e=1.7e9, T_e=3228K):
- R_spont(A_ul)=7.2e5, R_absorb(B_lu·J)=1.2e2, **C_up=9.5e-7, C_down=5.3e-3** → 충돌이 복사보다 **~1e8× 약함**, J_line=4.1e-7(차갑고 약함).
- ⇒ 순수 A_ul 복사-캐스케이드 분포, thermalize 앵커 전무 → super-thermal. (4434Å: n_up/n_lo=2.8e-2 vs 고립2준위 1.6e-4 = 캐스케이드로 175× 과점유.)

물리적 제약: sh8 n_e=1.7e9 ≪ O III 허용선 임계밀도(A_ul~3.6e8 → n_crit~1e12+). **이 밀도서 충돌은 O III를 thermalize 못 함 = 정상 물리** (충돌 강화는 비물리적 LTE 강제).

메커니즘 #1(재결합 Milne 자발항 누락 R_rec=R_bf·n*)은 별도로 이미 fix(plasma.c:6771-6883) — 하위 II와 차가운 외곽 drain 치유, 안전(T_e/n_e 불변). 천장(III)은 못 건드림.

## fix 3안 (비교·추천 요청)
- **A. IV 이온단계 추가**: O IV/Si IV/C IV…를 NLTE 타깃에 추가 → (III,IV)페어 생겨 III 들뜬준위에 bf 앵커.
- **B. 최상위 effective bf 앵커**: III 들뜬준위에 "맨핵+e" Saha 연속체로 가상 광이온화/재결합 부여(IV를 풀지 않고 연속체 결합만).
- **C. MC macroatom 스펙트럼**: 결정론 plasma 수렴 후 MC virtual-packet/macroatom로 선소스 생성(형광/캐스케이드 내장, 검증된 경로). plasma는 결정론 유지.

## 검증 질문 (각 안에 대해, 인용·근거)
1. **물리 충실도**: 실제 CMFGEN의 O III 들뜬 준위 점유를 가장 잘 재현하는 안은? 핵심: 이 가스상태(O 대부분 II/III, n_IV 작음 추정)서 CMFGEN의 O III 들뜬 준위는 **정말 thermal인가, 아니면 CMFGEN도 비-thermal인데 선수송이 다르게 처리하나?** super-thermal S_l이 정말 "틀린" 것인지부터.
2. **A안 유효성**: n_IV≈0이면 재결합 유입(∝n_IV)≈0 → A안이 정말 앵커가 되나? 아니면 **III 들뜬준위→IV 광이온화 OUT**이 n_IV와 무관하게 연속체 결합을 제공해 앵커하나(detailed-balance상)? A안이 부호상 thermalize시키나 sub-thermal로 드레인하나?
3. **B안 정당성**: III 위 "맨핵+e" effective 연속체가 물리적으로 방어 가능한가? σ_bf는 어떻게(수소근사 Z_eff? Kramers?). A안과 어떻게 다른가(B는 IV 인구 없이 연속체 결합만).
4. **C안**: macroatom 경로가 top-stage 캐스케이드를 결정론이 못하는 방식으로 올바르게 처리하나? 프로젝트 목표(스펙트럼)에 실용적 답인가? plasma 이온화는 여전히 결정론 reduced 모델인데 스펙트럼만 macroatom이면 정합적인가?
5. **추천**: 물리충실도·예상효과·구현비용/리스크 종합 1순위 + 근거. falsifier 테스트 제안.

원칙: 패치 금지, 근본·detailed-balance 복원, 구현 전 물리검증. 구체적 verdict.
</content>
