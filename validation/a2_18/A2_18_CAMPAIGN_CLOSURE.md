# A2-18 — A-2 캠페인 종결 판정 (중간 종결)

2026-08-06 작성(운전석). HEAD=`d47a596`.
**이 문서는 캠페인을 "완료"로 선언하지 않는다.** 무엇이 닫혔고 무엇이 왜 안 닫혔는지를
정직하게 적고, 층 1 진입 가부를 판정한다.

---

## 1. 단계별 폐합 상태

| 단계 | 계약 | 커밋 | 상태 |
|---|---|---|---|
| A2-00~03 | 원장 자격·census·격자 4000빈·스키마 | ~`606f23a` | **폐합** |
| A2-04 | J_ν 정본 승격 (commit 초크포인트) | `bafd2bb` | **폐합** |
| A2-05 | CPU bf 광이온율 → canonical view 직적분 | `d8b9870` | **폐합** (L-1bf PASS) |
| A2-06 | CPU bb rate → J̄ selective estimator + LineJbarCache | `ece5aef` | **폐합** (L-1bb BLOCKED) |
| A2-07 | population → 단일 Z(T_e)·transactional publish | `3ddd95c` | **폐합** (L-2 물리 BLOCKED) |
| — | 잣대 복구 (census 앵커 + 회귀 편입) | `694d9cd` | **폐합** |
| A2-08 | CPU opacity → signed χ 게시·클램프 제거 | `8a9f861` | **폐합** (L-4 BLOCKED) |
| A2-09 | CPU emissivity → Planck 표본 제거 | `36b8426`+`bf2af37` | **폐합** (L-3/L-5 BLOCKED) |
| A2-10 | 복사평형 → 항별 J_ν heating/cooling | `540ebbd`+`068fb36` | **폐합** (L-6 BLOCKED) |
| A2-11 | 중간 마디 (전량 회귀 + 계보 감사) | `9b73c04` | **폐합** |
| A2-12 | GPU lifecycle (세대 결박·업로드) | `4077537`+`3e9e317`+`7f72c0d` | **폐합 · GPU 검증 완료** |
| A2-13 | production CUDA 가 canonical view 소비 | `d47a596` | **폐합** |
| **A2-14/15** | GPU opacity·emissivity rate 이관 | — | **미완** — production API 부재 |
| **A2-16** | 스칼라 seed generation-0 격리 | `f5d646c`(증거) | **BLOCKED_UPSTREAM_NOT_CLOSED** |
| **A2-17** | 스칼라 생산자·잔여 소비 철거 | `2aaf6c7`(증거) | **BLOCKED_UPSTREAM_NOT_CLOSED** |

**16/18 폐합.** 남은 2단계(A2-14/15와 그에 의존하는 A2-16/17)는 정직하게 미완이다.

## 2. 게이트 최종 집계

| 게이트 | 상태 | 막는 것 |
|---|---|---|
| L-1bf (A2-05) | **PASS** | — |
| L-2 self-check (A2-07) | **PASS** | — |
| A2-12 GPU lifecycle | **PASS** (H200 실측) | — |
| A2-13~15 마이크로 오라클 | **bf·bb·conjunction·opacity·emissivity 전부 일치** | — |
| L-1bb (A2-06) | BLOCKED | `NETRATE`/`TOTRATE` |
| L-4 (A2-08) | BLOCKED | `CHI_DATA` |
| L-3 · L-5 (A2-09) | BLOCKED | `ETA_DATA` |
| L-6 (A2-10) | BLOCKED | `LINEHEAT` + free-T |
| A2-16/17 read-trace | BLOCKED | A2-14/15 미완 |

**PASS 세탁 0건.** BLOCKED는 전부 사유가 명시돼 있고, 그 사유는 두 종류뿐이다:
① O-PHYS 진리 파일 부재(4건) ② A2-14/15 production API 부재(2건).

### 특기: A2-12 GPU 음성대조 (slurm 218079/218009, H200)

9종 poison(CPU stale·cache generation·line ID shuffle·CPU/GPU 세대·partial upload·
invalid validity·fallback·upload bytes·reset generation)이 전부 **`physical_launches=0`**
으로 차단됐다 — GPU 커널이 뜨기 전에 잡혔다는 뜻으로, 개정 §5.4-5 계약의 실물 확인이다.

### 특기: A2-13~15 마이크로 오라클

`bf=True bb=True conjunction=True opacity=True emissivity=True` — **GPU rate 산술이 CPU
oracle 과 일치**한다. §5.4-7("bf/bb 한쪽만 맞는데 PASS 금지")은 충족됐다.
BLOCKED 사유는 **정확성이 아니라 production 배선 미완**이다.

## 3. 캠페인이 실제로 바꾼 것

셸당 스칼라 (W, T_rad) → **주파수분해 정본 `J_ν[4000빈]`**. 구체적으로:

- **소유권**: 정본 `RadiationField` + `LineJbarCache`, commit 초크포인트 2곳, checked view
  경유 소비만 허용, fallback(0·coarse·이전 세대) 전면 금지
- **rate**: bf Γ는 canonical view 직적분, bb J̄는 φ-가중 selective estimator(전역 빈
  재적분 금지 — 상위 격자 개정 준수)
- **population**: 단일 `Z(T_e)` 정본, transactional publish(부분 갱신 0)
- **opacity**: signed χ 게시 + 클램프/플로어 제거. 음수를 처리 못 하는 소비자는
  값을 바꾸지 않고 `BLOCKED_NEGATIVE_OPACITY_SEMANTICS`로 차단
- **GPU**: 세대 결박 lifecycle, fallback 금지(`BLOCKED_GPU_FALLBACK_FORBIDDEN`)

**측정된 물리 변화**: A2-05의 Γ_view/Γ_legacy = 0.987–0.998(전 이온·전 셸) —
threshold 부분빈 정확 적분과 4000빈 해상도의 효과.

## 4. 미완의 정확한 내용 (다음 착수점)

### 4.1 A2-14/15 — production API 부재
GPU opacity/emissivity rate가 canonical view를 소비하려면 production 쪽 API가 더
필요하다. 마이크로 오라클은 이미 통과했으므로 **산술이 아니라 배선 작업**이다.
현재 `GPU_*_NOT_MIGRATED` guard가 fail-closed로 유지되고 있어 조용한 회귀는 없다.

### 4.2 A2-16/17 — 위 의존
A2-17은 zero-consumer **최종** 철거라 모든 소비자 이관이 선행돼야 한다.
증거 커밋으로 현 상태만 기록해 뒀다.

### 4.3 O-PHYS — 진리 공급원
권위 노트 `/gpfs/kjhan/cmfgen_runs/OPHYS_RESUME_NOTE.txt`.
STAGE-1(T 고정)이 lageunha에서 진행 중이며, 최근 6 iteration에서 MAXCH가
`1.27E+07 → 6.33E+03`으로 3자릿수 이상 단조 감소했다(iteration 93).
수렴하면 **L-1bb·L-4·L-3·L-5가 한꺼번에 풀린다**. L-6은 STAGE-2(free-T) 필요.

## 5. ★층 1 진입 판정

**진입 가능.** 근거:

1. 층 1은 **입력축 감사**다 — Υ·A_ul·σ 등 "고리가 소비하되 생산하지 않는 것"을
   CMFGEN 원본과 직접 대조한다. 위 BLOCKED 6건은 전부 **고리 안쪽 물리 판정**이라
   독립도가 다르다(캠페인의 위상정렬 원칙).
2. A-2가 층 1에 필요한 것을 이미 공급했다: 정본 격자(I7 재기술의 근거),
   A_ul crosswalk 기준선, population 정본(I4 비교의 분모 고정), 클램프 카운터(I9 실측),
   살아 있는 원장(잣대 복구).
3. 층 1을 실제로 막는 것은 **`_ftos` 전면 재측정** 하나이며(구 덱 `_sivcaiv` 수치의
   분모가 통째로 바뀌었다), 이는 A-2와 무관하게 지금 착수 가능하다.

상세는 `docs/LAYER1_ENTRY_PREP.md`.

## 6. 검수 체계 결산 (개정 11, 3층)

L2 교차 검수(Codex, 저작/구현과 다른 프롬프트)가 잡은 것 — **L1 실측으로는 못 잡는 종류**:

| 단계 | 적발 | 성질 |
|---|---|---|
| A2-06 | coarse 재적분이 상위 격자 개정 위반 · f_cov 순환 · SE 규약 오류 | 아키텍처·잣대 |
| A2-08 | `element_wide.c` 파일 통째 누락 · `fabs(tau)` 부호 삼킴 | census |
| A2-09/10 | `lumina_main.c` 누락 · 이중계상 위험 · te_manifest 자기모순 | census·물리 |
| A2-12 | `.cu` 3군 누락 · CMF GPU의 CPU fallback(계약 금지) | census·계약 |
| 잣대 복구 | census 체커가 3커밋 동안 FAIL 방치 | 잣대 |

**교훈 1**: 명세 저작이 파일·심볼 목록을 스스로 좁히는 경향이 4연속 확인됐다.
**교훈 2**: "회귀 전판 PASS"는 회귀 목록 자체가 완전할 때만 의미가 있다.
**교훈 3**: 실행 검수(L1)와 읽기 검수(L2)는 서로 다른 결함을 잡는다 — 둘 다 필요하다.
