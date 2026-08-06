# A2-11 중간 마디 — 전량 회귀 + 계보 감사

감사 대상 HEAD = `068fb36` (A2-10 폐합 시점). 2026-08-06 작성(운전석).
3층 인수 프로토콜의 **마디**이므로 선별 회귀가 아니라 전량 회귀를 돌린다.

---

## 1. 원장 완전성

```
PASS A2_01_CENSUS rows=157 completed=20 unclassified=0
```

157행 전수 분류, 미분류 0, 이관 완료 20행. 체커는 `run_gate_battery.py`의
빌드 전 preflight로 편입돼 있어 앞으로 자동 실행된다(잣대 복구 커밋 `694d9cd`).

## 2. 회귀 전판 결과

| 항목 | 결과 |
|---|---|
| `make lumina` | PASS (error 0) |
| 게이트 배터리 36 케이스 (lageunha) | **PASS** |
| A2-01 census checker | PASS |
| selftest a2_05_bf_rate | PASS |
| selftest a2_06_line_jbar | PASS |
| selftest a2_06_dual_commit | PASS |
| selftest a2_07_population | PASS |
| L-0 replay (A2-04) | PASS (음성대조 5종 포함) |
| L-1bf 게이트 (A2-05) | PASS (음성대조 3종 포함) |

## 3. ★게이트 최종 상태 집계 — 하나의 패턴

| 게이트 | 단계 | 상태 | 막는 것 |
|---|---|---|---|
| L-1bf | A2-05 | **PASS** | — |
| L-1bb | A2-06 | BLOCKED | `NETRATE`/`TOTRATE` 부재 |
| L-2ion/level | A2-07 | self-check PASS / 물리 BLOCKED | 입력 manifest(진리 런) 부재 |
| L-4 | A2-08 | BLOCKED | `CHI_DATA` 부재 |
| L-3 · L-5 | A2-09 | BLOCKED | `ETA_DATA` 부재 |
| L-6 | A2-10 | BLOCKED | `LINEHEAT` 부재 + T 고정 |

**BLOCKED 5건이 전부 O-PHYS 산출물 부재 한 가지 원인이다.** 기계는 전부 구축·자기검증
완료됐고, 물리 판정만 진리 파일을 기다린다. 이것은 결함이 아니라 **의존 구조의 정직한
반영**이다 — 어느 것도 PASS로 세탁하지 않았다.

O-PHYS STAGE-1이 만들려는 것이 정확히 이 파일들이므로, 수렴 시 L-1bb·L-4·L-3·L-5가
한꺼번에 풀린다. L-6만 STAGE-2(free-T)를 추가로 요구한다.

## 4. O-PHYS 상태 — 정직한 기재

STAGE-1(T 고정)도 수렴하지 않고 있다. MAXCH 이력(최근 14 iteration, free-T·T고정 양쪽):

```
3.5E3 6.5E4 5.9E4 3.7E4 4.2E6 1.7E4 3.3E3 5.0E5 1.9E5 1.3E5 1.7E6 1.0E5 1.5E4 1.4E5
```

**3자릿수 폭으로 진동하며 감소 추세 없음.** 최대 변화 위치가 depth 18→24→51로 옮겨
다니는 것으로 보아 이온화 전선 이동 불안정이 계속된다(jnu4 덱 주석의 동종 사례 참조).

시도한 것과 결과:
1. 스텝 캡 완화(1.01/1.10 → 3.0/3.0, 수렴 실적 런 값): 교착은 풀렸으나 과도超 발생
2. T 고정 전환(STAGE-1): 전선 이동은 계속

**사전등록 판정 기준**: iteration 100까지 MAXCH < 1E+04 미달이면 런 설계 교체.
현재 iteration 84. 추가 파라미터 튜닝은 하지 않는다(수정-런-관찰 루프 금지 규약).

### 미결로 남기는 판단 (user 결정 사항)

수렴하지 않은 스냅샷을 **기계 게이트에 한해** 진리로 쓸 수 있는가?

- **찬성 논거**: L-1bb·L-4 같은 기계 게이트는 "CMFGEN 자기 상태를 고정한 채 Lumina 기계가
  CMFGEN 자기 rate/opacity를 재현하는가"를 묻는다. 이 산술에는 수렴이 무관하고,
  **자기 무모순**이면 족하다(어느 iterate에서든 rate는 그 populations·field로 계산된 값).
- **반대 논거**: L-2ion처럼 이온 분율 자체를 비교하는 게이트에는 미수렴 population이
  "정답"이 아니다. 또한 미수렴 산출물을 진리로 쓰는 관행은 잣대 오염 위험이 크다.
- **운전석 입장**: 단독 결정하지 않는다. 쓰려면 attestation에 미수렴 사실과 그때의
  MAXCH를 **공시**하고, 적용 게이트를 기계 게이트로 **한정**해야 한다.

## 5. 계보 감사 — 폐합 사슬

| 단계 | 커밋 | 계약 |
|---|---|---|
| A2-04 | `bafd2bb` | J_nu 정본 승격 (commit 초크포인트) |
| A2-05 | `d8b9870` | CPU bf 광이온율 → canonical view 직적분 |
| A2-06 | `ece5aef` | CPU bb rate → J̄ selective estimator + LineJbarCache |
| A2-07 | `3ddd95c` | population → 단일 Z(T_e) 정본 · transactional publish |
| 잣대 복구 | `694d9cd` | census 앵커 재결박 + 회귀 영구 편입 |
| A2-08 | `8a9f861` | CPU opacity → signed χ 게시 + 클램프 제거 |
| A2-09 | `36b8426`+`bf2af37` | CPU emissivity (seal + 구현) |
| A2-10 | `540ebbd`+`068fb36` | 복사평형 (seal + 구현) |

**계약 1개 = 커밋 1개** 규율 유지(A2-08 이후는 seal 커밋이 계약당 1개 추가 — 물리 출력
변경을 봉인하기 위한 구조적 요구).

## 6. 검수 체계 실효 기록

개정 11의 3층 검수(L1 실측 / L2 교차 / L3 Fable)에서 **L2가 잡은 것들**:

| 단계 | L2 적발 | 성질 |
|---|---|---|
| A2-06 | coarse 재적분이 상위 격자 개정 위반 · f_cov 순환 · SE 규약 오류 · 소비지점 9군 누락 | 아키텍처·잣대 |
| A2-08 | `element_wide.c` **파일 통째 누락** · `fabs(tau)` 부호 삼킴 | census |
| A2-09/10 | `lumina_main.c` 누락 · **이중계상 위험**(replace 규칙 부재) · te_manifest 자기모순 | census·물리 |
| A2-12 | `.cu` 3군 누락 · **CMF GPU의 CPU fallback**(계약 금지 사항) | census·계약 |
| 잣대 복구 | census 체커가 3커밋 동안 FAIL 방치 | 잣대 |

**명세 저작이 파일·심볼 목록을 스스로 좁히는 경향이 4연속 확인됐다**
(element_wide → lumina_main → .cu 3종). 남은 단계 검수의 최우선 항목으로 유지한다.

## 7. 마디 판정

- 원장 완전성: **PASS**
- 회귀 전판: **PASS**
- 계보 연속성: **PASS** (계약당 커밋 추적 가능)
- 물리 판정: **BLOCKED 5건** — 전부 O-PHYS 진리 부재, PASS 세탁 없음

**A2-12 이후 단계로 진행 가능.** 물리 판정은 O-PHYS 결과에 따라 A2-18에서 재집계한다.
