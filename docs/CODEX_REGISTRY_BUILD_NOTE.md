# Clamp 레지스트리 물리검증 조인 보고서

작성일: 2026-07-31  
정본: `docs/CLAMP_FIX_PRIORITY_REGISTRY.md`  
입력: Wave 1/2 B oracle, Gate B C 최종 적격범위, clamp census·provenance, ABS_STATE 51/54·60 overlay, 동등화 계획 v2

## 결론

94개 ID의 `oracle_verdict`, `band_link`, `fix_stage`, `priority`를 모두 채웠다. 우선순위는 **판정 잣대 오염 > 보존 위반 연루 > 인구 대체 > 기타**를 주축으로 하고, 같은 층에서는 B형 즉시성, A형 군집 승수, 직접 계량, 생산경로 여부 순으로 정했다.

- Gate B는 전체 582행 중 strict 99행(17.01%)과 명시 context 9행만 쓸 수 있다. 동일 행의 1%급 전후효과에는 적격이지만 전축·절대 CMFGEN 진리 판정은 아니다.
- Wave 1에서 C61의 `eta_bf`는 −4.31…−65.25% 변했으나 같은 단색 CMFGEN η 앵커가 없다. C59 multi-edge는 packet/GPU 경로라 oracle에서 0변화였다.
- Wave 2에서 C28과 C71은 각각 MA 확률 및 CUDA 방출 소비점이라 oracle에서 0변화였다. 이는 무효가 아니라 **oracle 범위 밖**이라는 판정이다.
- formal은 총입력 대비 ×18.07 비보존이고 출력의 2500–5000 Å 점유율이 71.80–72.49%다. parity60 일괄+D-4 OFF에서도 총량은 13.906배, 해당 대역은 71.74%이므로 단일 게이트 귀속은 금지한다.
- 현 CMFGEN 비교물은 fixed-T·미수렴 스냅숏이다. 따라서 상태·율·인구를 거치는 모든 간접 연결은 레지스트리에 **relT 착지 후 확정**으로 표시했다.

## priority 상위 20

| 순위 | ID | 채택 이유 | 수리 경로 한 줄 |
|---:|---|---|---|
| P001 | C09 | formal 잣대 직접 오염 + C09 군집 중심 | `FORMAL_FIX`로 formal fallback 일부 제거 가능; Wave3에서 network-out source provenance를 닫고 Stage 2·4 뒤 legacy fallback 제거, Stage 6 에너지 KA. |
| P002 | C22 | source=0 센티널이 C09 군집의 같은 상류 결함을 은폐 | Wave3에서 network-out source를 명시하고 Stage 2·4 폐합 뒤 zero sentinel·소비자 fallback 일괄 제거, Stage 6 τ/S KA. |
| P003 | C40 | C09 군집의 Te no-root/HOLD가 상태·선원함수 고정점을 대체 | Stage 4 global nonlinear solve에서 no-root·endpoint feedback을 수리하고 HOLD/500·1000 K 대체를 일괄 제거; relT 착지 후 확정. |
| P004 | C44 | C09 군집의 lagged/non-SE 인구가 음수 냉각 floor를 유발 | Stage 4에서 일관된 SE·RE residual을 만든 뒤 `COOL_NONNEG` 제거; line-cooling 부호·RE 장부는 relT 착지 후 확정. |
| P005 | C66 | C09 군집의 Jbar→population 폭주를 rollback으로 은폐 | Stage 3 권위장을 Stage 4 응답계에 결합한 뒤 rollback·영구 Jbar 차단을 함께 제거; relT 착지 후 확정. |
| P006 | C67 | formal τ/S·continuum provenance를 바꾸는 판정경로 | 기수리 `FORMAL_FIX`로 legacy 경로 제거 가능; Stage 6에서 e-scatter/LTE `S=B`/first-offending-bin/절대광도 KA를 통과한 뒤 기본화. |
| P007 | C68 | thick-line source를 `B(Te)`로 바꾸는 B형 falsifier | Wave3에서 production/default 불침범으로 격리하고 Stage 6 source-provenance KA 후 production 경로 제거. |
| P008 | C69 | IGE forest opacity를 없애 2500–5000 Å 잣대를 직접 변경 | Wave3에서 falsifier 전용 namespace·배너로 격리하고 Stage 6 opacity ledger 후 production 경로 제거. |
| P009 | C70 | Fe 창 source multiplier가 formal 잣대를 직접 변경 | Wave3에서 oracle falsifier로만 격리하고 Stage 6 source-energy ledger 후 production 소비를 0으로 봉인. |
| P010 | C72 | impact-ray 수가 formal 구적 오차상한 없이 잣대를 변경 | Stage 6 계측: `nimpact` 수렴표와 절대광도 오차상한을 만든 뒤 기본값을 정하며, 그 전에는 제거가 아니라 보류. |
| P011 | SC18 | nearest-neighbor comparator가 판정 잣대 자체 | Stage 0 계측: 보존 보간·속도/주파수 오차예산과 동일-row manifest를 만든 뒤 nearest-neighbor를 교체; relT 앵커에 재검증. |
| P012 | C52 | cap-hit packet 343개 에너지 삭제가 확인된 직접 비보존 | Stage 5에서 boundary/event 계수를 분리하고 cap-hit energy ledger를 보존시킨 뒤 truncation·force-escape 경로 제거. |
| P013 | C59 | single representative edge가 실제 multi-edge Milne 사건을 대체 | 기수리 Wave1 `FIX_BF_MULTI_EDGE`로 제거 가능; Stage 5 CPU/GPU event CDF·energy KA 통과 후 legacy single-edge 퇴역. |
| P014 | C61 | η가 최대 65.25% 변한 B형 process-predicate 불일치 | Wave1은 η 술어만 부분수리; Wave3/Stage 1에서 실제 target-term predicate를 통일한 뒤 `REC_SPINGATE` 대체 제거, relT 착지 후 확정. |
| P015 | C28 | `J≤aWB`, `J≥bWB` prior가 MA 확률 소비점을 오염 | 기수리 Wave2 `FIX_MA_J_UNCLAMP`로 제거 가능; Stage 5 fate 확률·energy KA 통과 후 MA 소비점의 cap/floor 우회 기본화. |
| P016 | C29 | 2000–3500 Å UV `J_nu` cap이 주 비보존 대역과 직접 겹침 | Wave3/Stage 3에서 권위 frequency-coupled field로 교체하고 UV cap 제거; rate·spectrum 판정은 relT 착지 후 확정. |
| P017 | C71 | 선택 line 사건을 Planck 재추첨해 packet energy/frequency를 변경 | 기수리 Wave2 `FIX_MA_NO_LINE_THERM`으로 제거 가능; Stage 5 event KA 후 `LINE_THERM`의 MA/line production 영향 제거. |
| P018 | C53 | MA cascade cap의 절단 오차와 보존 영향이 미계측 | Stage 5 계측: cap-depth 수렴, 미완 cascade energy, CPU/GPU fate census를 만든 뒤 오차상한에 따라 제거 또는 수치 guard로 재분류. |
| P019 | C06 | χ·η·Δτ 비음수화가 formal source/opacity 짝을 직접 바꿈 | Stage 6 first-offending-bin·signed contribution 계측 후 정의역 guard와 물리 음수항을 분리; production τ/S 보존식 밖 clip 제거. |
| P020 | C10 | EPAY 재척도·τ gate가 formal energy 장부를 직접 변경 | Stage 6에서 EPAY 지급·흡수·방출을 하나의 절대 ledger로 폐합하고 중복 재척도 제거; 보존 KA 전에는 계측 유지. |

상위 20 중 기수리 게이트로 legacy 제거가 가능한 것은 C67, C59, C28, C71이다. C09는 `FORMAL_FIX`가 일부 경로만 덮으므로 군집 전체 수리로 세지 않는다. C61도 Wave1이 η 술어만 고쳤으므로 부분수리다.

## A형 군집별 일괄제거 시점

| A형 상위 결함 군집 | 대상 ID | 최초 구조 수리 | 일괄제거 시점과 필수 관문 |
|---|---|---|---|
| NLTE network/source coverage·operator-split feedback | C09, C22, C40, C44, C66 | Stage 2 source coverage + Stage 3 권위장 + Stage 4 global response | Stage 4 residual·hot/cold·Jv 관문 뒤 state-side clamp 일괄 제거; C09/C22 formal legacy는 Stage 6 energy KA에서 최종 제거. **relT 착지 후 확정**. |
| NLTE singular/ill-conditioning·solver 실패 | C13, C14, C15, C16, C17, C19, C48 | Stage 1 full model projection + Stage 2 element-wide/global-charge solve | Stage 2B full-rank·conditioning·permutation·residual 관문 뒤 제거 후보화하고, Stage 4의 clamp/floor/freeze=0 acceptance에서 일괄 제거. **relT 착지 후 확정**. |
| 원자 자료·평가기 커버리지 | C08, C24, C25, C33, C34, C35, C49, SC13, SC21 | Stage 0 projection manifest + Stage 1 process graph | coverage 100%, LTE detailed balance, loader checksum 뒤 Kramers/gbar/Axelrod/skip/floor 일괄 제거. rate/pop 영향은 **relT 착지 후 확정**. |
| radiation-field sampling·provenance·geometry | C26, C27, C46, C60, C63 | Stage 3 single authoritative field + 3-grid CMF | field producer/consumer ID 100%, transport residual, Richardson 관문 뒤 binned/zero/refit/prior fallback 일괄 제거. **relT 착지 후 확정**. |
| D-5 upper-stage-blind·continuum drain 부재 | C64, C65 | Stage 2 element-wide S/Fe matrix와 metastable continuum 연결 | Stage 2B 원소보존·global charge·같은 해 관문 뒤 Boltzmann anchor와 `b_k` cap 동시 제거. **relT 착지 후 확정**. |
| 저온 DR resonance/K6 자료 결손 | C36 | Stage 1 ion별 저온 DR 데이터와 process predicate | 실제 데이터 coverage·detailed-balance KA 뒤 uniform α_DR floor 제거. 절대 이온화 영향은 **relT 착지 후 확정**. |
| transport interaction 계수·packet drop | C52 | Stage 5 event-level ledger | boundary/event count 분리, packet energy ≤0.1%, fate coverage 100% 뒤 legacy cap-hit drop을 제거. relT 비의존 event 관문. |

## 검산

- 레지스트리 데이터행: C01–C73 73행 + SC01–SC21 21행 = **94행**.
- ID 집합: 누락 0, 중복 0, 추가 0.
- join 4열: 공란 0.
- priority: P001–P094 각각 1회.
- 상위 20: P001–P020 연속, 레지스트리와 본 보고서 순서 일치.
- 수정 대상: `docs/CLAMP_FIX_PRIORITY_REGISTRY.md`, 본 보고서 두 파일뿐.
- `git` 명령 미사용.
