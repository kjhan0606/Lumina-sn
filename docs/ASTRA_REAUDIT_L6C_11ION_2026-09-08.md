# astra 재감사 — L6C·BIT-LOCKSTEP 의 「11이온 전제」 (gpt-6-astra, read-only, GPU 0회, 2026-09-08)

발주 = `/gpfs/kjhan/lumina/dproxy_audits/l6c_reaudit/ORDER_ASTRA_L6C_REAUDIT.md`. user 지시(09-08 「재감사 수행해」).

재감사 결과: **11이온 기전 전제는 유효하며, 범위·수치 기재 2건을 정정해야 합니다.** GPU 런 0회, 파일·봉인 수정 0회. 읽기 제한에 따라 본문으로 제출합니다.
저작자 표기: 지정 사전등록·설계·판정문은 Fable 저작으로 명시돼 있습니다. `CODEX_COVERAGE_SCOPE_SUMMARY.md`는 Codex 계열 보고서이며, 현 컨텍스트의 직접 저작물은 아닙니다.

아래 **대장**=[VERIFICATION_REGISTERS](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERIFICATION_REGISTERS.md), **R**=[L6C 재발주 정본](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/RUNG_L6C_REDISPATCH_2026-08-27.md), **V**=[L6C 판정 정본](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_L6C_REDISPATCH_2026-08-28.md), **BL**=[BITLOCK 판정](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_BIT_LOCKSTEP_PRODUCER_2026-08-27.md).
덱 실측 기준 **L**=[R-II levels.csv](/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1/input/model/levels.csv). 지정 4루트 모두 동일 SHA256: `b3645052d00dbd535de4f9b3687eda3da02f3dd0584ccef4079c8f9e700daf67`.

| Q | 답 | 근거 |
|---|---|---|
| Q1 | **YES** — 11이온 집합을 소스·봉인 덱에서 확인 | 소스 `lumina_plasma.c:10105–10112`의 31슬롯과 덱 교집합은 비어 있지 않은 **12슬롯**, 그중 절단 **11**. 목록은 BL:218–221·320–324와 일치. 측정 3이온은 **Fe II·Co II·Ni II**: 표적 필터 `:14420–14422`가 Z=26/27/28로 고정되고 exports:4가 ion=1. 원 사전등록:49–62는 III단 차단 전례 때문에 II단을 선택했다고 명시. |
| Q2 | **YES** — 11이온 전부 다중 준위 럼프 보유 | `lumina_atomic.c:1468–1474`가 전 준위를 `min(rank,100)`으로 덮어쓴다. `lumina_plasma.c:16755–16756,16789–16792`에서 사상은 `super_mode`와 독립; `population_contract.c:199–200`은 OFF에서 f=1, `:248–266`은 ON에서 Boltzmann 정규화. 재분배 `lumina_plasma.c:19801–19803`. **11 안에 singleton-only 이온 없음**; Ca II는 집합 밖. |
| Q3 | **PARTIAL** — 관측·귀속 경계는 유지했으나 범위 수치 오류 존재 | BL:301–312는 인구 동일성을 추론, 타 이온 lockstep을 미측정으로 구분. V:164–166·275–278도 **8이온 응답 미측정**을 명시한다. 8이온 응답을 관측 확정한 문장은 확인되지 않았다. 다만 **R:146–147의 ‘커버 3이온 기준 21,919준위’와 V:157의 BL 21,919준위**는 모집단을 혼동한다. «기재로 족함» |
| Q4 | **PARTIAL** — 구조·집계·일부 증인은 런 없이 측정 가능 | L과 `line_list.csv`로 럼프·등g·등E census 가능. 각 `stderr.log`에는 `LINE-ION-OWNER` **216행/런**의 이온별 순율·방출/흡수·부호/0 집계가 있다. R-II stderr:18은 비커버 **Co III** 증인(`τ=−0.0158854`, `n_upper=0.145528`). 그러나 8이온의 L6C 행별 응답 전수는 없다. H200 stderr:2204의 SL 기록은 **BLOCKED, actual_member_records=0**이므로 그 대체 증거가 아니다. «기재로 족함» |
| Q5 | **YES — 정정 2건, 강등 0건** | 대장:3의 전이규칙상 기존 구성·3이온 범위의 **DET-BITLOCK C / DET-L6C-COVER-R C 유지**. 8이온의 응답·정량 옳음은 **B 검증대상**이며 이번 구조 census로 C/A 승격되지 않는다. 두 수치 오류는 기존 귀속·분기 판정값을 바꾸지 않는다. |

실측 SL 수는 CSV 원생 `super_level` 수가 아니라 **봉인 K=100을 적용한 유효 사상**이다. 단계 괄호는 0기반 `ion_number`; 측정 여부는 기존 L6C/BITLOCK 행별 측정을 뜻한다.

| Z | 단계 | 준위 수 | SL 수 | 럼프 여부·멤버 수 | 측정 여부 | L 근거 행 |
|---:|---|---:|---:|---|---|---|
| 14 | Si II (1) | 125 | 101 | YES·25 | NO | 2–126 |
| 14 | Si III (2) | 147 | 101 | YES·47 | NO | 127–273 |
| 16 | S II (1) | 322 | 101 | YES·222 | NO | 538–859 |
| 16 | S III (2) | 256 | 101 | YES·156 | NO | 860–1115 |
| 20 | Ca III (2) | 232 | 101 | YES·132 | NO | 1538–1769 |
| 26 | Fe II (1) | 2,599 | 101 | YES·2,499 | YES | 2673–5271 |
| 26 | Fe III (2) | 1,500 | 101 | YES·1,400 | NO | 5272–6771 |
| 27 | Co II (1) | 2,558 | 101 | YES·2,458 | YES | 10772–13329 |
| 27 | Co III (2) | 3,214 | 101 | YES·3,114 | NO | 13330–16543 |
| 28 | Ni II (1) | 1,000 | 101 | YES·900 | YES | 19544–20543 |
| 28 | Ni III (2) | 1,000 | 101 | YES·900 | NO | 20544–21543 |

제외 대조: **Ca II=62 FL/62 SL, 럼프 0**(L:1476–1537). 따라서 전체는 **12슬롯·13,015 FL·1,173 SL**, 절단 집합은 **11이온·12,953 FL·1,111 SL**.
선목록 2,588,798행 독립 census: 양끝 럼프 **1,537,729**, 등g **539,149**, 등E **0**. 비커버 8이온 몫은 각각 **604,004 / 213,660 / 0**이며, 이는 덱 구조 계수다.
`CODEX_COVERAGE_SCOPE_SUMMARY.md:200`의 “11이온”은 **CMFGEN 대비 준위 비<0.5 집합**으로, 이번 절단 11이온과 다른 집합이다. «기재로 족함»

대장 정정 제안:
- **정정 1 — «기재로 족함»**: 대장:304–305의 `13,015 FL = 62+11×101`을 **`13,015 FL → 1,173 SL = 62+11×101; 실재 12슬롯 중 절단 11이온`**으로 정정. BITLOCK 귀속·C 등급 불변.
- **정정 2 — «기재로 족함»**: 대장 L6C 항(:396–415)에 R:146–147·V:157의 모집단 오류를 기록. **21,919는 전체 원자 덱의 rank≥100 계수**, NLTE 절단 11이온은 **11,853**, 측정 3이온은 **5,857**이다. `95/885`, `0/790`, C-M-S·C 등급 불변.

폐합 차단 지적 **0건**. 두 대장의 “비커버 8이온”은 **진단 미커버인 NLTE 이온**을 뜻하며, 기존 PC-2의 **NLTE 밖 LTE 공급 이온**과 구별해야 한다.
L6C_REAUDIT: 통과 | 정정 2