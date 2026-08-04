# Stage 3 앞당김 차터 — 결정론 CMF 권위장 (user 승인 2026-08-01 새벽)

승인 문구: "Stage 3 앞당기는 것 승인. 병행 가능 여부 검토해서 진행해."
근거: Γ 삼중대조(docs/CODEX_W3_GAMMA_TRIPLE_COMPARE_2026-07-31.md) — s8 과이온화의 진범=동결 MC 장 내용(Γ 15–67× 과구동). 로드맵 정본=docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md §Stage 3 (:200-215).

## 1. 병행 가능성 검토 결과 (운전석)

**판정: 병행 가능 — 문서·설계·KA는 완전 병행, src 편집은 파일-소유권 분리로 준병행. 통합 지점만 직렬화.**

| 전선 | 편집 대상 | 충돌 분석 |
|---|---|---|
| Wave-3.2 수리 배치 (D1·D4·단방향 DR·D6/D7/D3 계기·D8[falsifier 판독 후]·II–V 창 확장) | `src/lumina_plasma.c` + `src/lumina_element_wide.c` | 이 배치가 두 파일 **소유** |
| Stage 3.1 (신규 CMF formal solver + KA 하니스) | **신규 파일** `src/lumina_cmf_field.c`(+헤더) + `scripts/` KA 러너 | greenfield — 기존 파일 무접촉 |
| Stage 3.2+ (producer/consumer 단일화 배선) | `src/lumina_plasma.c`·`src/lumina_cuda.cu` 통합 지점 | **Wave-3.2 머지 후에만 착수** (동시수정 금지 준수) |

- 검증런 동결과의 정합: Stage 3.1은 전부 오프라인(동결 χ,η 위 KA·벤치 대조) — 신규 모델 런 0.
- relT2 의존: Stage 3 **최종 acceptance**(released-T 앵커의 frozen 입력 위 Jν/Γ/moment 문턱)만 앵커 의존. KA 4종·Richardson·jnu4 벤치 대조는 앵커 불요 → 즉시 착수 가능.
- Codex 동시 인스턴스 규칙: src-편집 Codex A는 전선당 1개·파일 소유권 교차 금지. 현재 가동 중인 읽기성 인스턴스(D8 falsifier)와 무충돌.

## 2. Stage 3.1 범위 (첫 증분 — 이번 발주)

**목표: CPU 정본 결정론 CMF formal solver 골격 + KA 사다리 + "동결 χ,η 위 장 교체 후보" 정량 벤치.**

1. **Solver 골격**: comoving-frame formal solution, sequential-frequency(blue→red) formulation(로드맵 허용 대안), homologous frequency advection 포함, inner/outer BC(로드맵 §Stage 3), electron redistribution은 3.1에서 coherent 근사+구조 자리만(3.2에서 완전화).
2. **KA 사다리(사전등록, 각각 해석해 대조)**: ①pure absorption ②coherent scattering ③homologous redshift(선 하나의 P-Cygni/적색이동 보존) ④(구조 자리) redistribution. 각 KA는 h, h/2, h/4 Richardson 수렴 지수 동봉.
3. **판별 벤치(이번 증분의 핵심 산출물)**: parity59 동결 χ,η(불투명도·방출률)를 입력으로 새 solver의 Jν를 생산하고, 같은 셸에서 ①동결 MC 장 ②CMFGEN jnu4 J_nu와 3중 대조. **사전등록 기대(방향만)**: Γ-관련 대역(600–3000Å)에서 새 Jν가 MC 장보다 CMFGEN 쪽으로 이동(형광 깔때기는 MC 선수송 결함이므로 결정론 수송+동일 χ,η에서 완화 방향). 만약 새 Jν도 MC와 같은 UV 과잉을 재현하면 결함은 수송이 아니라 **χ,η(방출률) 자체** — 이것이 다음 국면을 가르는 무료 판별이다.
4. 산출: 신규 `src/lumina_cmf_field.c`(+`lumina_cmf_field.h`), KA 러너 스크립트, 벤치 보고서. 기존 파일 무수정(빌드 훅 1줄 제외 — Makefile).

Acceptance(3.1): KA ①②③ 해석해 상대오차 ≤1e-4(로드맵 transport residual 문턱 준용), Richardson 지수 ~2(스킴 차수), 벤치 3중 대조표+대역별 비율. floor/clamp 0(clamp 규율 준수 — 위생 클램프도 계량 없이 금지).

## 3. 실행 순서 (07-31 심야 시점)

1. Codex A' 발주: Stage 3.1 상세 설계 명세(수치 스킴·격자·BC·KA 해석해 정의) 초안 → 운전석 검수 → 구현 발주. [즉시]
2. D8 falsifier 착지 → Wave-3.2 수리 배치 명세 확정(D1 필수 포함) → Codex A 발주(plasma.c/element_wide.c 소유). [착지 시]
3. Wave-3.2가 B/C 사이클로 넘어간 뒤 Stage 3.1 구현 병행 개시(greenfield). 통합(3.2+)은 Wave-3.2 머지 후.
4. relT2 착지 → 앵커 인증 시 Stage 3 acceptance 입력(frozen 상태) 재캡처 계획 수립.

## 4. 원장 정정 동봉

parity26-diag의 "s12+ FUV 기근 HOLD(해악 미미)" 위해도 등급을 **"dex-급 이온화 오염 확정(Γ 삼중대조)"으로 재분류** — 가림막(pair Saha) 위에서 매긴 위해도였음. 캠페인 메모리·대장 반영.
