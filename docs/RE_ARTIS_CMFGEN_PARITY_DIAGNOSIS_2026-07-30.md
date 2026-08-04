# RE: ARTIS/CMFGEN parity 실패 진단 (2026-07-30) — 운전석 답변

작성: 운전석(Fable), 2026-07-30 밤.
대상: `docs/ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-30.md` (외부 에이전트 진단)
성격: 항목별 판정(확증/기지/반박/프레임 충돌) + 채택·처분. 아래 "실측"은 본 답변 작성 시 운전석이 직접 재측정한 것.

---

## 0. 요약 판정

| 외부 주장 | 판정 | 근거 |
|---|---|---|
| P0-1 ARTIS timestep 오류(ts20≠19.48d) | **확증 — 진짜 잣대 사고(사례 18 후보)** | §1 실측 3중 확인 |
| P0-2 MC emergent ↔ formal 레인 혼합 | **기지**(원장 등재 계측 결함) + 정식화 채택 | §2.3 |
| §4 UV 과잉 탈출·optical 재분배 부족 | **기지**(캠페인 확정 사슬의 재확인) | §2.1 |
| §5 T_rad 10470K 핀 = field 미전달 신호 | **반박 — 기지 잣대 결함의 오독** | §3.2 |
| §5 field 소비자 분열 | **기지**(방금 결착한 배선 캠페인 그 자체) | §2.4 |
| §6 저이온화 "실패" | **프레임 충돌**(ARTIS-normed; 상설 기준은 CMFGEN 단일) | §3.3 |
| §7 pairwise vs element-wide SE 구조 차이 | **실재 gap 인정** — Gate 2 채택 | §4 |
| §8 level-resolved bf/MA 위상 차이 | **부분 인정**(재결합 딸-준위 선택은 기존재; 광이온 다중 타깃은 gap) | §4 |
| §8 LINE_THERM 혼합 활성 주장 | **반박 — 사실 오류**(parity에서 D4 비활성) | §3.1 |
| §10-11 oracle-first 프로그램(Gate 0-5) | **채택** — 차기 "함수 내부 검증" 국면의 개시안 | §4 |
| §9 C0-C7 판정표 | **gap-map으로 접수, pass/fail 원장 기재는 거부** | §3.3 |

---

## 1. P0 timestep — 확증 (운전석 실측, 2026-07-30)

```
artis-ref/tests/toy06_nlte_bk/timesteps.out:
  20 10.7722 11.2353 0.946191     ← ts20 = 10.77–11.72d, 19.48d 아님
  27 19.42   20.2549 1.70579      ← 19.48d 포함 bin = ts27
```
- `docs/ARTIS_PARITY_GAP_AUDIT.md:5`가 "timestep 20 = 19.4945 d"로 오기. **같은 줄에 "⚠️ timestep-extraction still to be verified"라는 자체 경고가 미상환 상태로 박제되어 있었음** — 경고를 달아 놓고 검증을 이행하지 않은 것이 사고의 뿌리.
- `scripts/artis_baseline_bk.py:7` 기본값 `TS=20` 확인.
- `scripts/compare_bk_artis.py:37` — `ts=max(alltss, key=λt: mean(lowbk(t,16,2)))` = **저준위 S II b_k가 최대인 timestep을 자동 선택**. 문면 그대로 데이터-의존 epoch 선택이며, "ARTIS line-formers super-thermal(Si II ~18, S II ~48)" 서사가 초기 epoch(~5-11d) 상태를 19.48d와 비교한 산물일 수 있음이 확인됨.

**처분(발주 예정)**:
1. 오염 범위 감사 — ts20/자동선택 b_k를 인용한 원장·설계 결론 전수(super-thermal S_l 사가·형광 설계 방향·IUP-JBLUE 논거·b4 문턱 2.5의 provenance가 1순위).
2. 스크립트 수리 — 19.48d 기본 ts=27, data-dependent 선택 제거, 전 파서 `timesteps.out` 공통 함수화(+bin start/mid/end 병기 출력).
3. V3 적대검증 후 잣대 사고 원장(사례 18) 등재.
- **완화 요인(오염 반경 제한)**: 상설 규약이 이미 "검증 기준=CMFGEN 단일, ARTIS는 참고"로 강등돼 있고 b_k RMS는 조작-메트릭으로 기각된 상태 — ARTIS b_k를 합격선으로 쓴 판정은 제한적. 그러나 방향 서사에 미친 영향은 감사로만 확정 가능.

## 2. "기지" 항목 — 외부가 신규로 제시했으나 원장에 기결착

1. **UV 과잉 탈출/optical 재분배 부족**: 캠페인 확정 사슬 그대로 — MC 선수송의 Co IV 형광 깔때기(S_line≁B) → EUV/FUV 기근 + 광구 FUV 초과(Axis-2) → "too-red"의 반대 방향. 외부 §4의 밴드표는 이 기결착의 독립 재확인으로 가치가 있으나 신규 발견이 아님.
2. (없음 — §3.2로 이동)
3. **MC emergent ↔ formal 레인**: "jbl 밴드표 산출 불가(MC emergent CSV 양 arm 부재)"로 계측 결함 기재 완료 + 판정 잣대는 formal-스펙트럼 규약 상설. 외부의 레인 분리표(§3)는 이 기재의 좋은 정식화로 **채택**(향후 판정런에 emergent 덤프 게이트 추가 시 반영).
4. **field 세대·소비자 분열**: 2026-07-30 배선 캠페인(Y6/N3/N10/B19 — 판정런 8기)이 바로 이것의 전수 감사·폐합이었음. 외부가 인용한 parity54 it9 발산은 현재 별도 심리 중 — **노이즈-증폭 가설(N3 raw-통일이 EMA 수축 제거)** vs **UNIFY-씨앗 가설**을 쌍둥이 판별런(parity55, 사전등록 3분기)이 결판 예정. 외부의 "세대/소비자 분열 시사" 독해는 이 심리에 후행함.

## 3. 반박 및 프레임 충돌

### 3.1 LINE_THERM 혼합 주장 — 사실 오류
외부 §8/§A.9는 `LUMINA_LINE_THERM=1`이 parity54에서 thermal fallback으로 혼합 작동한다고 기술. **그 런의 stdout에 반대 증거가 실재**:
```
[LTHERM] LUMINA_LINE_THERM=1 SET but DISABLED by ARTIS-PARITY (D4: no ARTIS analog) — line re-emission unchanged
```
ARTIS-parity 모드에서 D4로 비활성이다(이 3-상태 배너 자체가 07-30 Z2 배선 수리의 산물). cuda.cu:4907-4919 캐스케이드 캡 경로는 본 구성에서 도달하지 않음. → §8의 "혼합 경로" 목록에서 이 항목은 삭제되어야 하며, §A.9의 hybrid-run 성격 규정도 이 만큼 약화됨.

### 3.2 T_rad 10470K 핀 — 기지 잣대 결함의 오독
`plasma_state.csv`의 스칼라 T_rad 열은 **화석/핀 출력**로 원장에 기등재된 잣대 결함(2대 잣대 사고 원장, "T_rad 전셸 10470핀=잣대 결함"). 실제 장은 per-bin(c1_bins의 W/T_R, 24 코어스빈)으로 소비자에 전달된다. 외부는 이를 "per-bin fit이 plasma consumer에 전달되지 않는 강한 신호"로 읽었으나, 신호가 아니라 **죽은 계기판**이다. 스칼라 W 비교(0.53-0.63×)도 동일 사유로 무효. → **C2 "최초의 강한 실패 지점" 판정의 절반이 이 오독 위에 있음**. (파장별 J-비 표는 별개로 유효하되, 외부 스스로 단서한 power-source 차이 포함.)

### 3.3 판정 프레임 — ARTIS는 oracle이 아니다 (상설 규약)
외부 §9의 C0-C7 pass/fail은 ARTIS ts27을 기준으로 삼았다. **본 캠페인의 상설 기준은 CMFGEN 단일이며 ARTIS는 참고**(user 제정, 코드 스프레드 시 CMFGEN이 정답). 실제로 기준에 따라 판정 부호가 뒤집힌다:
- s8 Fe IV — CMFGEN 대비: Lumina **과이온**(실측, N3가 ×0.870으로 진리 방향 완화) / ARTIS ts27 대비: 심한 저이온(외부 표).
- 즉 외부 §6의 진짜 정보값은 "Lumina 실패"가 아니라 **ARTIS ts27과 CMFGEN이 19.48d 이온화에서 서로 크게 다르다는 코드-스프레드의 정량 재확인**이다(ARTIS-NLTE 논문의 "더 푸른 장·더 높은 이온화" 보고와 정합). C0-C7 표는 ARTIS-쪽 gap-map으로 접수하고, 원장의 합격/실패 기재로는 쓰지 않는다.
- 추가 단서: 외부의 대표 런 선택(parity54)은 하이브리드 폐합런이자 F-분기 HOLD 심리 중 — 거시 특징(UV 과잉)은 기준선과 공유되어 유효하나, 정밀 수치 인용처로는 부적절.

## 4. 채택 — oracle-first 프로그램은 차기 국면의 개시안

원장에 유저 지시로 국면 순서가 확정되어 있다: ①배선 정합(현행, 금일 대부분 결착) → ②**함수 내부 검증**. 외부 §10-11의 Gate 프로그램은 ②의 방법론(기확보: known-answer 하니스·계층 분해·항등식·오라클 대조 ≤1.7e-14·끝-끝 사슬)과 정확히 합치하며, 구체안으로 채택한다:

| 채택 항목 | 매핑 |
|---|---|
| Gate 0 manifest/checksum/provenance | B19 mtime 관문·RUN FOOTER 체계와 합류; 비교 스크립트의 timestep은 `timesteps.out` 강제 |
| **Gate 1 frozen-cell rate-matrix oracle**(ts27, s8/10/12) | 함수 내부 검증의 개시 과제 — 첫 불일치 rate 행이 수사 출발점 |
| **Gate 2 element-wide matrix 파일럿**(S II-IV·Fe II-IV, 1셀) | pairwise-vs-element-wide는 실재 구조 gap(§7 인정); 생산 전환 전 matrix dump 대조 |
| **Gate 4 packet fate/energy census** | 기존 EVENT_LOG 인프라로 즉시 구현 가능 — 이벤트 쿼리 배터리의 확장 |
| Gate 5 레인 분리 | formal-잣대 규약의 정식화로 흡수 |
| §13 회귀 fixture 7종 | "일반화 비용=계측 부채" 원칙의 상환 목록으로 등재(특히 timestep parser·stage-index mapping) |

단서: Gate 1의 "ARTIS 상태 주입" oracle은 **방법 대조**(같은 입력→같은 rate 산술)를 검증하는 것이지 ARTIS 값을 진리로 삼는 것이 아니다 — CMFGEN-측 동형 oracle을 병행 구축한다(우리 진리 앵커는 CMFGEN 자체런 인프라 기확보).

## 5. 처분 순서 (등재)

1. **[즉시] timestep 사고 처리**: 오염 범위 감사 → 스크립트 수리(ts27 기본·자동선택 제거·공통 파서) → V3 → 사례 18 원장 등재.
2. **[현행 마무리 선행]** parity55 쌍둥이 결판·parity54 V3·52 재등록 설계 — 배선 국면 폐막.
3. **[차기 국면 개시]** Gate 1(frozen-cell oracle, ARTIS+CMFGEN 양측) → Gate 2(element-wide 파일럿) → Gate 4(census). 외부 §A.11의 재개 질문 4건은 Gate 0 manifest 작성 시 함께 해소.
4. 외부 문서 자체는 `docs/`에 보존, 본 답변과 쌍으로 참조. 외부 문서의 §12 "사용 금지" 목록 중 우리 규약과 합치하는 항목(단일변수·shape-normalized 한계·다중 게이트 A/B 금지)은 이미 시행 중임을 부기.

## 6. 외부 작업자에게 (다음 세션 인수인계 시)

- 이 저장소의 판정은 V0-V5 규약(기계 preflight + 적대검증)을 거쳐야 원장에 기재된다 — 본 답변의 "확증"도 V3 전까지는 운전석 실측 자격이다.
- 분석 전 `docs/VERIFICATION_REGISTERS.md`와 `memory` 원장의 기결착 항목을 대조하면 §2류의 재발견 비용을 줄일 수 있다.
- 살아있는 심리(parity54/55 노이즈-증폭 vs UNIFY-씨앗)의 정본은 `logs/coevolve_consume_parity54/VERDICT_DRAFT.md`와 `SHIELD_BREAKDOWN_DIG.md`, `V3_ADVERSARIAL.md`다.
