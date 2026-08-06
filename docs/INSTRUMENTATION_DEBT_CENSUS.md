# 계측·배선 부채 census (C1–C7)

2026-08-06/07 (운전석 기계 census). 착수 계기 = user *"아예, 계측 및 배선을 전체적으로 한번 볼까?"*

부류는 발명한 것이 아니라 **2026-08-06 하루의 실제 발견에서 유도**했다. 그날 물리 결함보다
계측·배선 결함이 압도적으로 많이 나왔고, 전부 *"목록을 좁혀서 놓쳤다가 전수로 다시 세니
나왔다"* 는 같은 패턴이었다.

## 방법 기록 — Codex 위임 실패와 전환

Codex 발주가 **4회 연속** 탐색 중 종료했다(exit 0, 최종 메시지 없음).
범위 8문항→3문항→7부류→**1부류**, 예산 리셋, `-o /tmp` 전부 무효.
`reasoning effort: xhigh` 로 도는데 저장소가 커서 탐색 단계에서 소진된다.

⟹ **이 저장소에서 Codex 는 개방 탐색을 완주하지 못한다.**
전환: **증거는 운전석이 기계적으로 모으고, 판정만 위임한다.**

---

## C1 기대값 고정

특정 덱/epoch 에 묶인 하드코딩 기준선이 그 사실을 표시하지 않은 채 검증에 쓰이는 곳.

| 실물 | 상태 |
|---|---|
| `tests/zinert_canonical_tau_fixture.c` + `scripts/run_zinert_selftest.py` | **수리됨** — 덱별 표 + 미등록 fail-closed. 핀이 **두 곳에 이중**이라 하나만 고치면 증상이 동일해 오진을 유발했다 |
| 64자리 sha256 리터럴을 든 python 검증 스크립트 **4개** | `a2_06_l1bb_gate.py` · `emiss_t5_rank1.py` · `uv_t2n9_offline.py` · `verify_trad_fix.py` — 기대값인지 기록값인지 **미판정** |

## C2 계약 강화 시 호출자 미이관 — **최대 항목**

정본 `validation/a2_16/C2_CALLER_MIGRATION_CENSUS.json`.
하드 거부 env **12종**, **영향 런처 157개**(앞선 집계 98은 과소였다).

| env | 설정 / 발동 |
|---|---|
| `LUMINA_NLTE_FLOOR_REG` | 146 / **71** |
| `LUMINA_FROZENIN` | 92 / **37** |
| `LUMINA_TE_TRAD_RATIO` | 78 / **78** |
| `LUMINA_NLTE_FLOOR_MODE` | 18 / 9 |
| `LUMINA_TRAD_COLOR_FIX` | 12 / 12 |
| 나머지 7종 | 각 0–2 |

처분 2갈래: **의미 중립 제거**(앞 2종, 죽은 노브) 대 **의도적 실행 불가 표시**(나머지 10종 —
실제 기능이었고 A2-07 이 물리적 판단으로 금지했다. 줄을 지우면 그 런의 의미가 바뀐다).

## C3 음성 픽스처가 실제 결함물에 의존

음성대조가 "진짜 결함을 가진 산출물"을 픽스처로 쓰면, 그 산출물이 고쳐지는 순간
대조가 **조용히 무력해진다**.

| 게이트 | data/ 참조 | 상태 |
|---|---|---|
| `run_gate_battery.py` | 5 | **1건 수리** — `--negative-deck` 을 결함 보유 덱에 고정 |
| `run_k_gate.py` | 2 | 미점검 |
| `run_composition_c_gate.py` | 4 | 미점검 |

## C4 생성 파이프라인 불완전

| 실물 | 상태 |
|---|---|
| `deck_regen_*_driver.py` **5개**가 `finalize` 이전에서 멈춤 → 39파일 누락(충돌자료 34 포함) | **수리됨** — `deck_regen.py` 하나로 통합, 전 단계 + 완전성/stale 게이트 |

## C5 계보 미기록

| 덱 | vintage manifest | provenance stamp |
|---|---|---|
| **`toy06_19p48d` (생산)** | **없음** | **없음** |
| `_sivcaiv` | 없음 | 없음 |
| `_sivcaiv_fullcov` | 없음 | 없음 |
| `_ftos` · `_links` · `_vac` | 있음 | 없음 |
| `_jnu4` · `_ophys` · `_ophys_exacthyd` | 있음 | 있음 |

**생산 덱에 원자자료 계보가 아예 없다.** user 08-03 동일성 교리 아래에서 대조군 자격의
전제를 충족하지 못한다.

부수: **회귀 대장에 덱 축이 없다** — 138행이 `binary_identifier.argv` 와 `model_geometry`
로만 덱을 간접 노출하며 스키마 필드가 아니다.

## C6 검증기 미실행

`validation/a2_*` 는 단계마다 산출물 2–9개가 있어 **기계적 고아는 없다.**
확인된 미실행은 **L-1bb · L-4 · L-3 · L-5 4종**이며 사유는 오라클 미인증이다
(user 결정으로 BLOCKED 유지). 게이트 **코드 경로 자체가 한 번도 실행된 적 없다**는
위험은 `A2_18_CAMPAIGN_CLOSURE.md` §2 에 기재돼 있다.

## C7 재현 스크립트 불일치

`schema` 를 만드는 스크립트가 저장소에 없는 산출물 **5건**:

| 산출물 | schema |
|---|---|
| `A2_18_GATE_ROLLUP.json` | `lumina-a2-18-campaign-closure-v1` |
| `A2_18_OPHYS_ORACLE_REFUSAL.json` | `lumina-a2-18-ophys-oracle-refusal-v1` |
| `L1_REMEASURE_02A_INTERNAL_AUL.json` | `lumina-layer1-remeasure-02a-v1` |
| `L1_REMEASURE_02B_SOURCE_ANCHOR.json` | `lumina-layer1-remeasure-02b-v1` |
| `L1_VINTAGE_CROSSWALK.json` | `lumina-layer1-vintage-crosswalk-v2` |

**★자기 기재**: 이 중 `A2_18_OPHYS_ORACLE_REFUSAL.json` 은 **내가 2026-08-06 에 인라인
`python3 -c` 로 만든 것**이다. R-8(재현 스크립트 격차)을 비판해 놓고 같은 결함을
재생산했다. 인라인 생성은 편하지만 재현 경로를 남기지 않는다.

---

## 총평

수리 완료 3부류(C1 일부·C4·C3 일부), **최대 미결은 C2(157 런처)**.
그 다음이 C5(생산 덱 계보 부재 + 대장 덱 축 부재)다.

**C6 를 제외한 모든 부류에서 "첫 집계가 좁았다"** — 오늘 네 번 반복된 실패 방식이다
(생산 덱 오인 · 6파일만 비교 · `--deck` 배선 · C2 집계 98→157).
전수로 세기 전에는 어떤 수치도 하한으로 취급한다.

## 미결 (판정 필요)

1. **C2 처분** — 2갈래 중 어느 쪽을, 어느 런처 집합에
2. **C5** — 생산 덱 계보 소급 기록 여부, 대장 스키마 v2(덱 축)
3. **C1 잔여** — sha256 리터럴 4개가 기대값인지 기록값인지
4. **C3 잔여** — `run_k_gate.py` · `run_composition_c_gate.py` 미점검
5. **C7** — 5건에 생성 스크립트 부여
