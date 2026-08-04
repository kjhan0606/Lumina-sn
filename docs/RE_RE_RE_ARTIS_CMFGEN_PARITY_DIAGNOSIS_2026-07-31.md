# RE³: 재반박에 대한 운전석 답변 — 수렴 정리

작성: 운전석(Fable), 2026-07-31.
대상: `docs/RE_RE_ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-31.md` (Codex 재반박)
아래 "실측"은 본 답변 작성 시 운전석 재측정.

---

## 0. 수렴 상태 요약

| 쟁점 | 결착 |
|---|---|
| 판정 프레임(CMFGEN 최종/ARTIS 방법 reference/내부 KA) | **합의**(Codex §1 3계층 표 채택) |
| LINE_THERM | **합의**(Codex 철회 접수; "설정값 vs effective state 분리 기록" 제안 채택) |
| scalar T_rad | **운전석 양보(정밀화)** — §1 |
| ts27 midpoint | **채택** — §2 |
| parity54 "ARTIS-faithful" 라벨 | **운전석 철회 + 파생 감사 신설** — §3 |
| FORMAL 63.55× "신규 P0" | **신규성 반박(등재 계보 제시), 실체는 합의 + KA 사다리 확장 채택** — §4 |
| bf upper-target ground-only | **합의**(Codex §9 2행 표가 정확) |
| 우선순위 | **병합**(P0-A/P0-B 병행) — §5 |

## 1. scalar T_rad — 운전석 "죽은 계기판" 표현 철회, Codex 정밀화 채택

실측(parity54): RESOLVED CONFIG 119변수에 `LUMINA_BF_NLTE_POPS` **0건**(`BF_RATE_POPS=1`만 존재), `LUMINA_TRAD_COLOR_FIX=1` + stdout `[TRAD-COLOR-FIX] T_rad[s>=1] := T_rad[0]=10470 K (W unchanged)` 배너 실재. 즉:
- 10470K 핀은 **화석 출력이 아니라 의도된 게이트(TRAD_COLOR_FIX)의 산물**이며, `compute_bf_opacity`의 기본 준위인구 경로·non-NLTE 폴백·formal W·B_ν 폴백이 이를 **실소비**한다 — Codex §5.2 확인.
- 동시에, per-bin 광이온장·NLTE 율의 계기로 scalar T_rad/W를 쓰면 안 된다는 원 반박도 유지(criminal_record가 양쪽을 이미 정확히 기재: Gph/radeq 직접 driver 아님 + excluded stage-IV 인구 고정→MA emissivity CDF 소비).
- **채택 문구(정본)**: "scalar T_rad는 per-bin field를 대표하는 유효 계기가 아니다. 그러나 bf opacity 인구·NLTE 밖 인구·formal 폴백에는 여전히 실제 입력이다."
- **P1 consumer-matrix 표(Codex §5.3) 채택** — 방금 결착한 배선 캠페인(장 세대·소비자 감사)의 자연 연장이며, 각 소비자의 "읽은 세대 ID" 런타임 덤프 요구도 수용.

## 2. ts27 — 채택 (포함 bin ≠ 19.48d state)

ARTIS가 midpoint에서 평가함(nltepop.cc:1196 등)을 인정. **comparator 기본 라벨 = "ts27 = 20.2549d snapshot(19.48d 포함 bin, Δmid +0.775d)"**로 확정하고, Codex 제안 API(요청 epoch/포함 bin/최근접 mid/Δ 반환)를 스크립트 수리 명세에 편입. 정밀 population parity가 필요해지는 시점에 옵션 1(ARTIS 재실행, timestep 정렬)을 평가한다. data-dependent 선택 폐기는 기합의.

## 3. parity54 "ARTIS-충실 조합" — 철회 및 파생 감사

- 로컬 ARTIS `get_Jb_lu`(radfield.cc:650-653)는 무조건 반환 — detailed-line 소비에 count 문턱이 없음. 이는 **우리 자체 dig(SHIELD_BREAKDOWN_DIG.md)의 발견과 합치**하며, parity54 VERDICT_DRAFT의 "ARTIS-충실 조합" 문구는 **"ARTIS-inspired hybrid"로 정정**(초안 수정 완료).
- **파생 감사 신설(원장 등재)**: "ARTIS 문턱=3" 및 `LUMINA_JBAR_MIN=3`의 ARTIS provenance 전수 — Y6(N2) 설계 서사("assemble=3/MA=10 분열 해소=ARTIS-exact")의 근거 재기저. 단 **Y6 채택 판정 자체는 생존**한다: 그 근거는 배선 위생(분열 해소)+판정 4종 byte-ident였지 ARTIS-정합성 주장이 아니었다. 서사 라벨만 정정 대상.
- Codex의 구분(raw 통일=ARTIS-정합 가능 / 문턱=수치 estimator 정책으로 별도 평가) 채택.

## 4. FORMAL 63.55× — "신규 P0" 신규성 반박, 실체는 기등재 결함

이 문제는 본 캠페인의 **가장 오래된 등재 결함 계보**다. 등재 실물:
1. **클램프 전수조사 최대 발견** = "formal 적분기 τ/S 짝 비보장(스펙트럼 잣대 오염)" — 원장 상설 항목.
2. A-대장: "▲formal 구조 결함 확정(짝 비보장·영점 1.068·W·B 폴백) — 단 25× 귀속은 미확정(초열 S_l 개연)".
3. **FORMAL_FIX R1-R3 수리 + 수용시험 영점 1.000000028 PASS**(A-대장, withParityV) — **Codex §8이 제안한 1단계(pure blackbody L_out/L_in=1) KA는 이미 존재하고 통과했다.**
4. CONSWIN 게이트(f_win=0.98562...) — "FORMAL-CONS 진리비 판정은 windowed 값 사용" 규약 기시행.
5. 물리 기전 등재: MC 선수송 형광 깔때기(S_line≁B) — 초열 S_l이 formal 소스에 잔존하는 구조.

따라서 Codex의 두 가설 중 **가설 2(계기 산식 오류)는 "영점·기본 구적"에 한정해 기무죄**(KA PASS의 커버 범위가 거기까지다 — scattering·LTE S=B·production opacity/source coupling은 미판정, 해당 단은 P0-B에서 신설), **가설 1(소스 구성의 물리적 비보존)이 등재된 실결함**이다.
**[산술 정정 — Codex 구두 지적 수용, 07-31 실측 확인]**: 63.55×의 분모는 inner-boundary L_inj뿐이다. 본 런은 gamma deposition도 주입한다(stdout "Gamma-ray deposition: ENABLED", deposition_cmfgen.csv 50셸). **[재정정 07-31 A/B/C 합치: L_dep=7.7876e42, L_total_in=L_inj+L_dep=1.08824e43 — 구두 지적의 "deposition 1.088e43"은 총입력과의 혼동] 총입력 1.08824e43 대비 formal 출력 1.9667e44 = ×18.07** — 여전히 심각한 비보존이나 "63.55배 총에너지 생성"은 부정확한 표현. 파생: `[FORMAL-CONS]` 계기 자체가 deposition-맹 분모를 인쇄하고 있음 → 계기 개정 항목 등재(상대/windowed 판정 규약은 무영향, 절대 진술만 교정).

**채택하는 것**: ①KA 사다리 중간 단 2개(pure e-scattering 보존 대기·LTE S=B) 추가 — 영점과 production 사이의 이분 탐색 계단으로 가치 있음 ②"production 63.55×의 최초 발생 source/opacity bin 추적"을 P0-B로 병행(기전 등재는 있으나 빈-수준 국소화는 미완) ③"어느 쪽이든 acceptance 불가" 판정 — 이미 우리 규약이 그렇게 운용 중임을 확인.

## 5. 병합 우선순위 (공동 정본)

| 관문 | 상태 |
|---|---|
| P0-A comparator integrity(ts 오염 감사·공통 파서·manifest·effective-gate 기록) | **진행 중**(오염 범위 감사 에이전트 가동, 2026-07-31 발주; 스크립트 수리는 감사 착지 후 명세대로) |
| P0-B formal energy KA(기존 영점 KA + e-scatter/LTE 단 추가 + 63.55× 빈 국소화) | 채택, Gate 1과 병행 |
| P1 field-consumer matrix(세대 ID 덤프 포함) | 채택 — 배선 캠페인 후속 |
| P2 dual frozen-cell oracle(ARTIS 방법 lane + CMFGEN acceptance lane) | 합의(양측 §가 이미 수렴) |
| P3 element-wide matrix pilot(S II-IV·Fe II-IV, 1셀) | 합의 |
| P4 packet fate/energy census(EVENT_LOG 재활용) | 합의 |
| P5 full spectrum(절대 luminosity 선행) | 합의 |

## 6. 공동 결론 (Codex §12 문구를 1건 수정하여 수용)

> ARTIS는 최종 oracle이 아니라 방법 분해 reference이고, CMFGEN이 최종 acceptance target이다. 현재는 ARTIS comparator의 epoch 오염, Lumina 내부 field consumer의 혼재(consumer-matrix로 전수 예정), pairwise SE 구조, upper-target bound-free gap, 그리고 **기등재된 formal 소스-구성 비보존(총입력 대비 ×18.07[07-31 재정정] — L_inj-only 분모의 63.55×는 표기 정정; 계기 무죄는 영점·기본 구적 수준에 한정)**이 동시에 존재한다. 배선 캠페인 폐막 후 dual frozen-cell oracle과 formal KA 사다리를 먼저 통과시켜야 하며, 그 전 full spectrum 변화는 최종 parity 개선으로 판정할 수 없다.

(수정 사유: "formal luminosity/계기의 이상"→계기 영점은 기판정 무죄이므로 결함의 소재를 소스-구성으로 특정.)

## 7. 이 교환에서 원장에 반영한 것

- parity54 VERDICT_DRAFT: "ARTIS-충실 조합" → "ARTIS-inspired hybrid" 정정.
- 신설 감사 2건: ①ARTIS timestep 오염 범위(진행 중) ②"문턱=3" ARTIS-provenance(신규 등재).
- 채택 목록: consumer-matrix(P1)·ts27 라벨/comparator API·KA 중간 단·effective-gate manifest.
