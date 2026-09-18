# 운전자 기획 브리프 — DR-on 확인 단 (2026-09-19, Fable 계약 기안용)

운전자(Claude) 작성. Fable 이 이 증거로 사전등록(계약)을 기안한다(개정25-b). 이 브리프는 계약이 아니다.

## 배경 (실측)
- **기전**: DR 이 조용히 꺼져 있었다 — `LUMINA_FROZENIN_DR` 는 경로만 켜고, `dr_alpha_eval` 이 `dr_boost_factor(source)=LUMINA_DR_BOOST_<SOURCE>` 를 곱하는데 계보 run5·CONT 런 exports 가 전 소스 0(coevolve 기본 상속). 정본 = `docs/READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18.md` §4, `[[feedback_dr_boost_zero_trap]]`.
- **오프라인 예측(운전자, 런 0회)**: DR_TABLE Badnell 계수로 `dr_alpha_eval` 을 mid-shell 쌍둥이 T_e 에서 평가. S III→II {16,2}: **α_DR/α_RR = 1.47/1.48/1.49** (sh12/16/20, T_e 10138/9386/8849K). RR 은 CONT 채널 덤프 `ne_alpha_rr/n_e`(소스 무관·T_e 속성). ⟹ DR-on 이 총재결합 ×2.48. base run5(안정 Kirchhoff 소스·DR off) f_II .017/.053/.102(r 57.0/17.7/8.79)에 적용 → **r/(1+1.48) ⟹ f_II 예측 .042/.123/.220**(sh12/16/20). CMFGEN .85/.80/.71 대비 **여전히 4–20× 부족** — 필요하나 불충분.

## 표적 (기획, user 2026-09-19 「DR 켜기 먼저」)
DR 을 켜서(부스트 노브) 안정 base 소스 위에서 **중간 셸 황 이온화가 예측대로 ~2.4× 이동하는지, DR-off 가 중간 셸 저재결합에 얼마를 기여했는지**를 측정. 초열 CONT 소스는 폐기(별건).

## 런 구성 (저비용·재빌드 불요)
- **바이너리 재사용**: 동결 트리 현 바이너리(sha 8ceafa36, `cuda_build4/lumina_cuda`) — CONT 게이트 OFF 면 base 와 byte-동일. 새 코드 0줄. env 만.
- exports: base run5 계보에서 `LUMINA_CMF_CONT_NLTE_SOURCE=0`(초열 소스 끔)·`LUMINA_CMF_FINE_UNMAPPED_EPS=1`(LTEUNMAP 수리 유지)·`LUMINA_FROZENIN_DR=1`·**`LUMINA_DR_BOOST_{BADNELL,NORAD,MAZZOTTA,AUTOSTRUCT}=1`**(DR 켬)·`LUMINA_ION_CHANNEL_DUMP=1`(DR 이제 nonzero 확인). ITER = run5 와 동일(2)로 base 직접 대조, 또는 8–10 으로 수렴점.
- 대조군 = run5(376251, DR off) 이미 있음 ⟹ **A/B 는 DR 토글 단독**.

## 기대치 (오프라인 예측 = 사전등록 후보)
- 채널 덤프 `ne_alpha_dr` **> 0**(부스트 이제 유효; base 는 0) — 음성대조.
- S f_II sh12/16/20 **상승 ≈ ×2.4**(.04/.12/.22 근방; 단 수렴·γ_phot 응답으로 편차 가능).
- CMFGEN(.71–.85) **미도달**(필요·불충분 확정).
- 내핵·외곽 부작용 최소.

## 감사·검증
- Q-GOAL: 이 단은 어긋남 지도의 무엇을 줄이는가 = 중간 셸 황 저재결합분(부분). Q-LEAN: 신규 계측 0(채널 덤프 재사용), 신규 게이트 최소. 경량 사전감사(Opus 4.8, Q-GOAL·Q-LEAN) 대상.
- ⚠Ca IV/V 는 DR_TABLE 항 부재(dr_lookup(20,3/4)=NULL) — 이 단 범위 밖(중간 셸 S/Si II↔III 만).
- ⚠오프라인 2.4× 는 1-step 율 추정 — 수렴값은 런이 정한다(offline-first: 예측→판정런 1회).
