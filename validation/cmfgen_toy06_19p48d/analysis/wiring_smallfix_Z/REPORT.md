# withParityZ 소수리 배치 — 배선/계기 결함 3건 (2026-07-30, Opus 구현) — **[운전석(Fable) diff 검수 통과 — 생산 인증 = parity48 byte-identity 런 대기]**

**운전석 검수 스탬프 (2026-07-30 00:40대)**: ①W/X/Y 바이너리 md5 불변(498b7376/f18c53a1/3477fa7f)·Z 신규(sha256 5501be69) ②Z1 SPECS 재라벨+근거 주석·plasma.c 이관 주석·shadow 배치(AS→Mazzotta→NORAD 순, 채택 판단 아님 명기) 직접 확인 ③Z3 else-사이트 2곳 본문=atomicAdd 단독(RNG 무소비·패킷 상태 무접촉) 직접 확인 ④Z2 LTHERM active 배너 문자열 보존+3상태 헬퍼 호출 확인. 잔여 강제 장치 = parity48 물리 산출물 byte-identity(사소한 잠입 변경도 여기서 걸림).

기준 트리 = withParityY 상태의 미커밋 `src/`(그 위에 적층). 3건 외 변경 없음. 커밋 없음. 빌드 `make cuda`(nvcc 13.0.88, sm_80/86/90) exit 0, 신규 경고 0(수리 전 트리 스크래치 빌드와 경고 diff clean). 산출 `lumina_cuda.withParityZ` 3,719,864 B, sha256 5501be69…, md5 b02d1da5….

## Z1 — NORAD (28,1) Ni DR 오배정 [데이터 배선]
결함(coverage_extension/REPORT.md 항6): raw_ni2.rrc.txt 실물 = Ni III+e→Ni II(NB01)인데 parse_norad.py가 Ni II→Ni I로 재라벨 → DR_TABLE {28,1} 설치.
- 수리: ①parse_norad.py:33-40 SPECS 행 (28,1)→(28,2) + "파일 인덱스=ion_recombining" 규칙 주석 ②plasma.c {28,1} NORAD 삭제(:6857-6861 이관 표식) + AUTOSTRUCT·Mazzotta (28,2) **아래** 동일 계수 NORAD {28,2} 신설(:6996-7013, provenance+SHADOWED ON PURPOSE+채택 판단 아님 명기).
- 왕복 증명: 생성기 스크래치 재실행 — burgess_fits.txt 전체 diff=1행 1필드(`28 1`→`28 2`), 계수 12+오차 2 byte-identical; repo 산출물 미덮어씀(mtime 2026-05-12 불변). 8개 raw 파일 Process: 행 전수 grep = raw_ni2만 어긋남(단일행 오류 재현). fit α(1e4K)=4.2470e-12 vs 원시 4.2751e-12(0.66%, fit 오차 내).
- 기능 프로브(gcc 링크): dr_lookup(28,1) NORAD→**NULL**; dr_lookup(28,2) 전후 공히 **AUTOSTRUCT**(c1=1.6796e-07, E1=1.4518e+03); (26,1)/(26,2)/(14,1)/(14,2)/(27,2) 전부 동일.
- **생산 동일성 체인**(parity46 stdout 실측 기준): dr_lookup 호출처 전수=2곳. (a) frozen-in :5617 — FROZENIN_DR=0이라 스킵. **단 FROZENIN_DR=1이면 실변화**(Ni I α_DR 4.2470e-12→0 = 오배정 채널 제거가 본 수리의 실물; parity42/43/45 계보 재현 시 단일변수 재검토 필요). (b) NLTE :15005 — 인자 (Z, ion_hi=2) → 전후 공히 AS 반환+마스크 0. (c) NLTE Ni 쌍 실측 = (Ni II, Ni III)뿐 ⟹ **(28,1)은 NLTE 경로에서 애초 미조회**(발주문의 "(28,1)→NULL 전제"를 실측 정정 — (28,1) 소비 가능 통로는 frozen-in뿐이었음). ⟹ 현 기본 config 산출물 영향 0.
- 미해결: Ni1.csv 고아화 처분(생성기는 이제 Ni2.csv 산출), Ni III→II 소스 채택 판단(의도적 미수행).

## Z2 — H3 배너 오표기 [감시자 결함]
결함(clamp_census/partC_hidden.md H3): set-but-disabled를 "unset"으로 오표기(cuda.cu:6421 구), set/disabled 구분 불능.
- 수리: `banner_gate_off()` 헬퍼(:5856-5881) — unset / set-but-OFF-by-value / set-but-DISABLED-by-ARTIS-PARITY(D4) / set-but-DISABLED(gate off) 4상태+사유+동반 노브(SMAX) 부기. 적용 3곳: LTHERM(:6467-6478)·BSRC(:6573-6575)·KPR(:6691-6693, 동일 `!artis_parity_enabled()` 부류). **active 배너 문자열 3건 무변경.**
- 증명: 헬퍼 단독 컴파일 4케이스 표(H3 실사례 = "SET but DISABLED by ARTIS-PARITY (D4) …; SMAX=49 also set"), 바이너리 strings에 신규 포맷 검출.
- 범위 밖 기재: [TINCOL] "unset"은 parity 게이트 없는 값-기반이라 H3 부류 아님(미수리); plasma.c:8556 [TEHOLD] 진단이 LINE_THERM을 parity 체크 없이 읽음(상태 무변 진단, 미수리). 옛 문자열 언급처 전수=주석 3곳뿐(하니스 영향 없음).

## Z3 — N9 무카운터 [계기]
결함(physics_wiring_audit/REPORT.md:9): fb 지배에지 조회실패→공명 퇴화 무카운터.
- 근원: find_ioniz_energy 1e10 sentinel → dom_edge_nu=0 → kpacket_fb_nu[s]=0; 디바이스 4사이트 중 라인-활성 2 = 공명 퇴화, bf-활성 2 = **else 부재로 무동작** — 넷 다 무계수였음.
- 수리(계수만): 호스트 카운터+접근자(plasma.c:3050-3063, lumina.h:1103-1107, 실패 계수 :4620-4631 OMP critical); 디바이스 d_n_fb_edge_degen(:2155-2162)+증가 4사이트(:4814, :4836, :5580-5584, :5602-5603 — 뒤 둘은 신설 else, 본문=카운터뿐); 런 종료 1줄 `[FB-EDGE] dominant-edge lookup failures: host_shell_updates=… device_degenerate_fb_exits=…`(:9667-9680).
- 물리 무변경 논증: atomicAdd 4+호스트 카운터 1, RNG 소비 0, 신설 else=원래 빈 false 경로, 소비자=종료 printf 1.
- 미해결: 실런 계측치 없음(parity48이 첫 실측); 최종 퇴화만 계수(FB-MULTI 실패 후 단일에지 폴백 성공은 별건); host 마지막-실패 (Z,stage) 1건 보존(비영이면 히스토그램 확장 후보).

## 변경 파일:라인 전수
parse_norad.py:33-40(Z1) / lumina_plasma.c:3050-3063·4593·4612·4620-4631(Z3)·6824-6829·6857-6861·6996-7013(Z1) / lumina.h:1103-1107(Z3) / lumina_cuda.cu:2155-2162·2194-2199·4814·4836·5580-5584·5602-5603(Z3)·5856-5881·6467-6478·6573-6575·6691-6693(Z2)·9667-9680(Z3).

## 처분
- **생산 인증 = parity48**(러너1, parity46 완주 후 자동): config=parity46 정확 복제, 유일 diff=LUMINA_BIN Y→Z. 등록 기대 = 물리 산출물 전부 parity46과 byte-identical + stdout 추가([FB-EDGE] 1줄·3상태 배너 문구·값 미등록) — 어긋나면 Z 배치에 잠입 변경 존재로 판정, 3분할 재감사.
- FROZENIN_DR=1 계보(parity42/43/45) 재현 시 Z1이 실변수임을 대장에 플래그.
