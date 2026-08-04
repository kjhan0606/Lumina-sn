# relT3 이중 독립 감사 종합 (운전석, 08-01 17:4x)

입력: docs/CODEX_RELT3_ANALYSIS.md + docs/FABLE_RELT3_ANALYSIS.md (상호 열람 금지 준수 확인).

## 1. 합치 (양측 독립 도달)

| 쟁점 | 판정 | 근거(각자 독립) |
|---|---|---|
| 종합 | **(c) 혼합** | 양측 동일 |
| % 잣대 | **붕괴 확정** | Codex: 거의 빈 고준위·terminal closure 지배 / fable: 반환 MAXCH=terminal 행 SOL≈1.09가 LIMIT 1.1 문턱을 매회 통과 → 1e7% 고정(기전 특정) |
| 물리 상태 | 전역 발산 아님 | n_e ≤2%/15it·질량가중 전하 ≤0.07%·심부 동결(fable) ≒ 10⁻⁴–10⁻³ 수준(Codex) |
| 완전 수렴도 아님 | 실물 미수렴 자유도 존재 | fable: d16–32 **캡-구속 정체**(Si III@d27 분율 0.50–0.60 무감쇠 왕복·Ca IV@d21 바닥 2.0× 크롤) / Codex: 비감쇠 진동·S 이온화 drift |
| it51 방향 | 방향=물리, 크기=수치 무효 | fable: **Ca V SL70(8g, 인구 5.6e-41)** 동정·far-outer 블록 cond 9.7e9·최소 특이벡터=Si V/Ca V 가족 / Codex: 동일 결론(조건화) |
| probe 폐기 | 정당 | fable: RE 잔차 d1 ×32 악화·심부 ×4000 이동 / Codex: 내부 광도 branch 이동 |
| terminal 행 | 수치 잔향(해악=제어 로직 경유) | fable 행렬 해부: 자기행=보존식(계수~1e-36)·자기열≈널 — MAXCH·auto-LAMBDA 오염이 유일 해악 경로 |

## 2. 상보 발견 (긴장 아님 — 관측량 상이)

- **Codex**: it51 full step 후 **내부 radiative luminosity(d=1) ×2.13**(6.19e10→1.32e11, it52 첫 평가에서 +100.7%) — 복사장 branch 이동, 이후 지속. 쌍안정성은 미입증(재현·되돌림 이력 없음).
- **fable**: 방출 스펙트럼 형상은 +3.7%만 이동·클립 23k건·심부(LAMBDA 불가촉) ×4000 이동. → 내부 flux 해와 관측 형상은 별개 지표(Codex도 명시).

## 3. 사전등록 프로브 2종 (상보 — 둘 다 it50 checkpoint 분기·저비용)

| | Codex P-lin | fable P-TF |
|---|---|---|
| 질문 | 광도 점프가 **매끈한 선형 응답인가 불연속 branch 전이인가** | trace 오염 제거 시 **LAMBDA가 실자유도를 실제 수렴시키는가** |
| 개입 | 단일 인자: MAX_LIN 1.05→1.01 (full 1회+LAMBDA 1회 평가) | trace-freeze LAMBDA(변수 게이트 1e-20×종총수) |
| 사전등록 기대 | 매끈=광도 +20%(8.0e10)·pop L1 0.0485% / 점프 유지=branch 전이 | E1–E3(수렴 기대)+반증 조건 명시 |
| 비용 | ~2.5h(BA 1회 포함) | ~1h(LAMBDA 5회) |

**운전석 권고: 두 프로브 병행**(별도 분기 디렉터리·상호 불간섭·합계 ~2.5h). P-lin은 branch 성격을, P-TF는 "잣대 정화 후 진짜 수렴 능력"을 각각 판별 — 두 결과의 조합이 relT3 노선의 최종 처분(수렴 재개 가능/branch 구조 실재/fixed-T 영구)을 결정한다.

## 4. 파생 등재
- 사례 후보: "반환 MAXCH의 LIMIT-문턱 고정 기전"(fable 특정) — 잣대 대장 추가.
- terminal 행 처방(fable 4종 중 변수-동결 최우선)은 P-TF가 겸검증.
