A-2 캠페인 A2-06 구현 명세 (개정 8: 저작·구현=운전석, 검수=Codex). 기준 HEAD=d8b9870 (A2-05 폐합). A2-01~A2-05 는 폐합됐다 — 각 보고서와
커밋을 먼저 읽어라. A2-05 가 만든 bound-free 이관 패턴(원장 행 단위 이관 + 게이트)을
bound-bound 에 재적용하는 단계다.

## 0. 먼저 읽을 것

1. `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md`:
   - §10 A2-06 행: "CPU bound-bound rate — `J̄`, 상·하향률 직접 계산; L-1bb PASS"
   - **§7 L-1bb 게이트 전문**: 비교량 `J̄ · R_lu=B_lu J̄ · R_ul^stim=B_ul J̄ ·
     R_ul^sp=A_ul`; CMFGEN 파일 = `WRITE_RATES` 산출 `NETRATE`/`TOTRATE`;
     "net rate 로 두 성분을 역추정하지 않는다"; 합격선 = flow coverage ≥0.95 ·
     E_1 ≤ 0.10 각각 · E_sym P95 ≤ 0.25 · `A_ul` crosswalk ≤ 1e-10;
     음성 대조 3종 = line frequency 한 빈 이동 / stimulated 항 제거 / 상·하위 level 교환;
     ★**현재 스냅샷 상태: `BLOCKED_MISSING_RATE_EXPORT`**
   - §2.3 말미: line `J̄` 는 별도 복사장이 아니라 정본에서 명시 계산
     `J̄_lu = ∫ φ_lu(ν) J_ν dν`
   - §13 경로 3("jbar_line/j_blue 가 이전 iteration 값 유지"), 25(net 만 비교해 상쇄),
     26(level index 만으로 연결), 8(로그 빈 중심을 평균으로 가장)
2. `docs/A2_01_DISPOSITION_LEDGER.md` — bound-bound 계열 `REPLACE_SCALAR_RATE_READ`
   행들과 `jbar_line`/`j_blue` 관련 행. **행 단위 이관 + 원장 갱신 diff 제안**
   (A2-05 와 같은 방식).
3. `docs/CODEX_IMPL_A2_05.md` — bf 이관에서 확립된 validity 전파·부분 빈 규약을
   bb 프로파일 적분에 일관 적용하라.

## 1. A2-06 의 단일 계약

**CPU bound-bound 복사율의 소유권 이관.** 상·하향 선률이 `jbar_line`/`j_blue`
직접 소비가 아니라 **정본 `RadiationField.J_nu` 로부터 명시 계산한 `J̄_lu`** 를 쓴다.

1. `J̄_lu = ∫ φ_lu J_ν dν` 를 빈 평균 위 보존 적분으로 (프로파일 폭이 빈보다
   좁은 경우의 처리 명시 — §13 경로 8 금지)
2. `R_lu = B_lu J̄` · `R_ul^stim = B_ul J̄` · `R_ul^sp = A_ul` 분리 유지 (§13 경로 25)
3. `jbar_line`·`j_blue` 의 rate-소스 지위 제거 — §2.2 허용 객체로 강등.
   Sobolev/macro-atom 경로가 이들을 읽는 지점은 census 행 대응으로 이관
4. transition 연결은 원소·이온·level label·에너지·통계가중치로 (§6.2 — 주파수
   근접성·index 단독 금지, §13 경로 26)

### ★L-1bb 의 정직한 처분 — 이 단계의 인수 형태

CMFGEN 최종 게이트는 스냅샷에 NETRATE/TOTRATE 가 없어 **BLOCKED 다. PASS 를
만들지 마라** (§10 공통: BLOCKED 는 BLOCKED). 이 단계의 인수는:

- **wiring replay**: CMFGEN `J_ν`(EDDFACTOR 재빈, 안전대 셸)를 정본에 주입해
  `J̄`·상향률을 Lumina 원자자료로 계산 — 배선 검증 (L-1bf 의 보조 replay 와 동형)
- **자기 일관 게이트**: 이관 전후 고정 RNG 에서 rate 값 변화가 이관의 의도된
  변화(사전등록 allowlist)와 정확히 일치
- **음성 대조 3종** (위 §7 목록) — wiring replay 위에서 FAIL 시연
- `A_ul` crosswalk ≤ 1e-10 은 스냅샷 무관하므로 **지금 판정 가능** — 실행하라
- L-1bb 최종 판정 = `BLOCKED_MISSING_RATE_EXPORT` 로 기록, O-PHYS 요구명세
  (`A2_00_OPHYS_PROFILE.json`)에 WRITE_RATES 요건이 이미 있는지 확인하고 없으면
  추가를 제안하라

## 2. 제약

- src 편집 단계. CPU 만(GPU 는 A2-13~15). bf 경로(A2-05 산출)를 회귀로 보호.
- population/이온화 솔버는 건드리지 마라 — A2-07 이다. 이 단계는 rate 계산까지.
- 덱·`/gpfs` 불변. commit/push 금지. 로그인 노드 연산 금지(빌드 예외). `/usr/bin/time` 금지.
- 회귀: 전 게이트 + A2-05 L-1bf.

## 3. 자기 검수

1. J̄ 가 정본에서 계산되고 별도 저장장이 소비자 API 가 아닌가 (§13 경로 3).
2. 상·하향이 분리 비교되는가 — net 단독 비교가 없는가.
3. 프로파일-빈 폭 관계 처리가 명시돼 있는가.
4. L-1bb 를 PASS 로 적지 않았는가. BLOCKED 근거가 기록됐는가.
5. census 원장 diff 가 이관 행과 1:1 인가.
6. 보고서 파일명·줄번호를 이 세션에서 확인했는가.

## 4. 보고

`docs/CODEX_IMPL_A2_06.md`. 이관 행 대응표 · wiring replay 산출 경로 · A_ul crosswalk
결과 · 음성대조 3종 · 운전석 실행 명령(2노드 분할) · §11 행(BLOCKED 명기) ·
남은 위험과 A2-07 인계.

## 5. census A2-06 확인 목록 (발주 시점 주입 — 재배치 후 19행; 줄번호는 HEAD=bafd2bb 시점)

| src/lumina_plasma.c:4556 | W | local alias of plasma->W[s] | [rate] bound-bound dilute Planck pump | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4556 | T_rad | local alias of plasma->T_rad[s] | [rate] bound-bound Planck color | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4596 | W | local alias of plasma->W[s] | [rate] LTE comparison field amplitude | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4596 | T_rad | local alias of plasma->T_rad[s] | [rate] LTE comparison field color | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4701 | W | local alias of plasma->W[s] | [rate] line upward radiative rate | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4701 | T_rad | local alias of plasma->T_rad[s] | [rate] line upward radiative rate | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4879 | T_rad | local alias of plasma->T_rad[s] | [rate] Boltzmann fallback exponent in line rate | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4880 | W | local alias of plasma->W[s] | [rate] metastable dilution in line rate | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11908 | W | local alias of plasma->W[s] | [rate] line source fallback | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11908 | T_rad | local alias of plasma->T_rad[s] | [rate] line source fallback | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11915 | W | local alias of plasma->W[s] | [rate] bin field construction | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11915 | T_rad | local alias of plasma->T_rad[s] | [rate] bin field construction | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12093 | W | local alias of plasma->W[s] | [rate] lower-level radiative weight | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12100 | W | local alias of plasma->W[s] | [rate] upper-level radiative weight | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:13739 | W | local alias of plasma->W[s] | [rate] coupled lower-level weight | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:13743 | W | local alias of plasma->W[s] | [rate] coupled upper-level weight | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:13920 | W | local alias of plasma->W[s] | [rate_diagnostic] coupled-rate luminosity diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |
| src/lumina_plasma.c:13940 | W | local alias of plasma->W[s] | [rate_diagnostic] coupled-rate floor diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |
| src/lumina_plasma.c:14080 | T_rad | local alias of plasma->T_rad[s] | [rate_diagnostic] coupled-rate residual diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |

주의: :13920/:13940/:14080 3행은 rate_diagnostic (DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD) — 이관이 아니라 정본-세대 결박 진단 파생. 나머지 16행 = REPLACE_SCALAR_RATE_READ 이관. jbar_line/j_blue 소비 지점은 census 밖일 수 있으니 grep 전수로 재검증 (A2-05 교훈: 스펙 목록 밖 소비자 1개 실측 발견됨).
