# 인증 커버리지 전이온 확장 (2026-07-29 밤, Opus 추출) — **[확정: fable 적대 재검증 CONFIRM 8/8, 2026-07-29 23시대]**
**fable 스탬프**: 8청구항 전부 원시 파일 독립 재계산으로 재현(Γ_PRRR 6값 전자리·730.0·2361.6=730.0×3.235 분해·절단 몫 0.4723·Verner 3중 6.4570 자체구성 확인·NORAD 단일행 오배정·회귀 byte-identity·frac 3값). 차단 정정 0. **정밀 주석 3**: ①3.235=1000준위-합 구베이커 계수(꼬리 ×3.220·표본 ×1.005), 0.180=지상준위 순계수(표본 ×0.143·꼬리 ×1.259) ②"12×" 노드 간격=지상표 평균; 중앙값은 **79×**(적신호 강화) ③회귀 byte-identity는 regress 그룹=파일 전체·나머지 5이온=이온-행 단위. +경로 정정: 참조 levels.csv 정본 경로=data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat_sivcaiv/(toy06_19p48d_sivcaiv는 심링크 명).
운전석 스팟체크 통과 3건: NORAD (28,1) 오배정 소스 확인·Ni IV/Fe IV levels.csv 200행 실측·회귀 byte-identity. 수치 CSV는 본 디렉토리 하위에 전부 보존. 재현: `python3 certify_coverage.py --group {regress,new,absent,all} --bake {shipped,bakefix,bakefix2,bakefix4,bakefix4b}`.

## 핵심 (요지 — 세부 수치는 CSV가 정본)
1. **커버리지 9→21/27** (+부재 3이온 문서화). 한계는 추출측이었음(하드코딩 IONS 표; 참조런 27종 전부 PRRR/osc/f_to_s/phot 보유). **구조적 맹점 노출: s0에서 원소를 지배하는 이온 전원(Ni IV·Co IV·Fe IV·S IV·Ca III·Si IV)이 기존 미인증이었다.**
2. **회귀 증명**: 기존 9이온 전부 byte-identical(harness sha256 25d36f43 불변; Si IV 파서 심은 no-op 증명 동반).
3. **신규 결함 3분류**:
   - **[데이터] LUMINA 원자 준위 절단**: Ni IV/Co IV/Fe IV/Fe V=200/1000, Ca V=200/528, Ca III=200/232 — 결손 준위가 인구 <0.05%인데 **Γ의 20-95% 운반**(D/C: Ni IV 0.473·Fe IV 0.773·Fe V 0.207@s0). 인구-몫 잣대는 못 보는 부류 — 율 잣대만 봄. 수리=베이커 준위 캡 상향(외부 데이터 불요).
   - **[기계] 심부-Wien 구적 오차**: Bpt/Ag 0.16-0.29 (Ca IV/V·Fe V·Co IV, Γ~1e-15..1e-30 영역) — 공명 아님(Bav≈Bpt), 문턱-빈 배치 기전 추정. 이 모델 실무 가중 ≈0.
   - **[수리 확인] fit-type**: Si IV 1.51→0.99 (bakefix2 정확 평가기 효과; 단 Si IV는 A층 자체가 type2/3 40% 결손이라 사다리 중간층 맹목 — D/PRRR만 유효).
4. **Ni II 판정**: 2361× 재현·분해 = 구베이커 인플레(꼬리 상수외삽 ×3.235, 지상준위 점표본 ×0.180) × **정직 빈티지비 730×**. 18oct00=자기고백 "very crude hydrogenic placeholder"(README), 19apr23=Nahar&Bautista 2001 R-matrix(NB01). bakefix4(=18oct00)만 게이트 PASS(0.995@s0=대조군) — 단 이는 잣대-일관성이지 물리 우월 아님. 데이터-개정 효과 = Γ(Ni II) ×3.42@s0..×0.77@s8. **적신호: 19apr23 피크 1.36e-12 cm²(기하단면 +4자릿수), 공명 노드간격이 LUMINA 격자보다 12× 촘촘 — 출판 대조 감사 전 채택 금지.**
5. **외부 앵커**: S IV/S V/Si IV/Ca IV/Ca V = 4소스 σ_th ≤9% 합치(**검증 데이터 확정**; Ca IV/V 게이트 FAIL은 구적 탓, 데이터 무죄). **Co III 19apr23 지상=VY95 핏 그 자체**(CMFGEN type9 6.4570 = Cloudy 6.4570 = ARTIS 6.4567 Mb — Co III 빈티지 사가의 외부 확인). **ARTIS는 Fe-피크 지상준위서 독립 아님**(Verner 재구현, 4-5자리 일치). Badnell RR은 N>15(+18) 부재 — Ni III/Co II/III/Ca IV 커버 불가.
6. **★bycatch — NORAD Ni DR 오배정(코드 버그)**: raw_ni2.rrc.txt의 실물 과정=**Ni III+e→Ni II**(NB01)인데 parse_norad.py가 "Ni II→Ni I"로 재라벨 → plasma.c:6831 {28,1,...} 설치. Fe/Si 행은 정상(단일행 오류). 결과: 권위 있는 NORAD Ni III→II 총계(4.28e-12@1e4K)는 미사용, Ni II→I 슬롯에 Ni III 율 적용 중. **K6/DR 대장에 직결 — 수리 대상.**
7. **잔여 미검증 + 처방**: Si V/Fe VI/Co V/VI/Ni V/VI = LUMINA 원자 부재(데이터는 디스크에 有 — 베이커 추가만) / 절단 6이온 = 캡 상향 / Ni II σ = NORAD 사이트에서 NB01 σ 표 다운로드 필요 / Ca III-V 공명분해 σ = TOPbase / Co II/III 여기준위 = 문헌 조사 필요 / **Fe III Milne합 vs NORAD 7× 갭 = 미해결 플래그** / s8 = PRRR/POP 이터레이션 불일치 대역(기지 [snap]).

## 처분 (fable CONFIRMED 후 대장 기재)
- 준위 캡 상향 + 부재 이온 추가 = 베이커 후속(외부 데이터 불요, 오프라인)
- NORAD (28,1) 오배정 수리 = 소수리 배치 후속
- Ni II/Ca/Co 외부 데이터 다운로드 목록 = user 승인 후(네트워크)
- 앵커 B(modern 재런) 착지 시 전 표를 앵커 B로 재산출
