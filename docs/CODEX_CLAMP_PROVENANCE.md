완료했습니다.

- [레지스트리](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CLAMP_FIX_PRIORITY_REGISTRY.md:15)에 `intro_reason`, `upstream_bug` 열 추가
- [고고학 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CLAMP_INTRO_REASON_ARCHAEOLOGY.md:25) 작성
- 분류 총계: A 30 / B 16 / C 43 / D 5
- 위험군: A 22 / B 14 / C 8 / D 1
- 94행 유지, ID 누락·중복 0
- A형 30건의 상위 결함 전부 기입
- 기존 join 4열은 전부 공란 유지
- `git diff --check` 통과

수정 범위는 요청한 두 문서뿐입니다. 두 파일 모두 현재 Git 기준 미추적 상태입니다.

---

2026-07-31 Wave 1 bf 수리 결과는
`docs/CODEX_WAVE1_BF_REPAIR_REPORT_2026-07-31.md`에 기록했다. B형 16건 중
bf/free-bound 직접 항목은 C59와 C61로 판정했으며, C59는 표준
`LUMINA_FIX_BF_MULTI_EDGE` 게이트로 기존 multi-edge 물리 경로에 연결했고
C61은 요청 범위인 B18-ⓐ `eta_bf` 술어 일관성만 수리했다. 비-bf 14건은
Wave 1 범위 밖 목록으로 보존했다.

---

## 2026-07-31 Wave 2 — 비-bf B형 사건 경로 / macro-atom 소비점

B형 16건에서 Wave 1의 bf 직접 항목 C59/C61을 제외한 14건을 다시
소비점 기준으로 분류했다. 이 중 이번 Wave 2에서 실제 packet-event 또는
macro-atom 확률을 직접 바꾸는 항목은 C28과 C71 두 건이다. 두 수정은
각각 독립 `LUMINA_FIX_*` 기본-OFF 게이트이며, 진단 게이트 자체는 삭제하지
않는다.

| ID | 수리 게이트 | 수정 | 물리 근거 |
|---|---|---|---|
| C28 | `LUMINA_FIX_MA_J_UNCLAMP` | MA internal-up이 읽는 `J_line`에서 `J_CAP_FACTOR`/`J_FLOOR_FACTOR`를 모두 우회 | 비LTE 방사장은 국소 `W B_nu`보다 크거나 작을 수 있다. `J≤aWB`, `J≥bWB`는 pumping 가설용 prior이지 복사수송 항등식이 아니며, macro-atom radiative-up 확률은 표현된 장의 `B_lu J`를 소비해야 한다. |
| C71 | `LUMINA_FIX_MA_NO_LINE_THERM` | `LINE_THERM`이 요청돼도 MA/line이 선택한 방출 전이 주파수를 유지 | radiative deactivation의 photon frequency는 선택된 bound-bound 에너지 차로 정해진다. 사건 뒤 `B_nu(T_e)`를 재추첨하는 것은 별도 thermal emissivity이지 같은 MA 전이의 물리가 아니며 ARTIS `do_macroatom`에 해당 override가 없다. |

C21은 사건 경로와 인접하지만 이번에 수리했다고 표시하지 않는다. population
inversion의 올바른 처리는 “음의 MA 확률”이 아니라 signed line opacity와
maser amplification을 갖춘 transfer operator다. 현행 `max(0, ...)`만 제거하면
CDF가 음수가 되어 더 큰 오류가 되므로 구조 수리 전까지 잔여다.

나머지 비-bf B형 12건은 이번 사건/MA 소비점 범위 밖으로 보존한다.

- C21 — signed maser transfer 구조 필요
- C32 — RADEQ/충돌강도 producer; 실측 Υ 뒤 floor 문제
- C45 — line-response/열평형의 LTE upper-population ceiling
- C47 — 상태장 `T_rad`/TEPIN/W 대체
- SC05 — launcher 조성 floor
- SC12 — 원자 준위 수 cap
- SC15 — offline ETLA prototype upper-population cap
- SC16 — offline fluorescence prototype placeholder `gbar/f`
- C29 — photoionization-field UV `J_nu` cap
- C68 — formal-integral thick-line source clamp
- C69 — formal-integral IGE opacity 제거 falsifier
- C70 — formal source multiplier falsifier

따라서 비-bf B형 14건의 Wave 2 처분은 **수리 2(C28/C71), 구조 잔여
1(C21), 사건/MA 범위 밖 목록 11**이다. 위 12건 목록에는 구조 잔여 C21을
포함해 “미수리 총수”를 보존했다.
