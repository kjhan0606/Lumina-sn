# A2-06 잔여 구현 보고서

작성일: 2026-08-06  
적용 명세: `SPEC_A2_06_V5.md` > V4 > V3 > V2

## 결과

A2-06 CPU bound-bound 생산 소비를 checked `line_view`로 이관했다. 정상 항은
`R_lu=B_lu*Jbar`, `R_ul^stim=B_ul*Jbar`, `R_ul^sp=A_ul`로 분리하며, 비-OK view,
MISS, UNSAMPLED, out-of-domain은 legacy/coarse/0값 fallback 없이 해당 복사 기여만
차단하고 원인별 atomic 카운터를 올린다. JEQB류 falsifier와 출력 진단은 유지했다.
population/이온화 fallback은 변경하지 않고 V4 처분대로 A2-07/A2-08에 남겼다.

로컬 선행 게이트는 모두 PASS지만 분리 NETRATE/TOTRATE export가 없으므로 최종 L-1bb
상태는 명세대로 `BLOCKED_MISSING_RATE_EXPORT`이다. 이 상태를 PASS로 승격하지 않는다.

## 소비 이관 대응표

행번호는 구현 후 작업트리에서 직접 재확인했다.

| 명세 대상 | 구현 위치 | 생산 산식/처분 |
|---|---|---|
| census 6행(구 4556·4596·4701의 W/T_rad) | `src/lumina_plasma.c:4832-4836` | checked `Jbar`; `rate=B_lu*Jbar`, blocked이면 복사 상향항 0 |
| census-밖 4633 그룹 | `src/lumina_plasma.c:4698-4710`, 최종 `:4832-4836` | det/MC/coarse 사다리는 진단 shadow, 생산은 checked view |
| census-밖 4661 | `src/lumina_plasma.c:4726-4783`, 최종 `:4832-4836` | jblue/fallback은 falsifier·진단 shadow, 생산은 checked view |
| census-밖 4731 | `src/lumina_plasma.c:4796-4815`, 최종 `:4832-4836` | legacy net 계수는 shadow, 생산은 `B_lu*Jbar` |
| census-밖 10827 | `src/lumina_plasma.c:10914-10925` | simul ETLA: `B_lu*Jbar`, `A_ul+B_ul*Jbar` |
| census-밖 12182 | `src/lumina_plasma.c:12269-12280` | RADEQ ETLA: `B_lu*Jbar`, `A_ul+B_ul*Jbar` |
| census-밖 13823 | `src/lumina_plasma.c:13912-13923` | coupled ETLA: `B_lu*Jbar`, `A_ul+B_ul*Jbar` |
| census-밖 15238·15292·15361 | `src/lumina_plasma.c:15348-15556`, 최종 `:15566-15573` | legacy source/mode는 shadow; 최종 세 항을 canonical split으로 덮어씀 |

공통 소비 함수는 `src/lumina_plasma.c:514-561`이며, view 상태와 lookup 결과를 함께
검사한다. 카운터 필드는 `src/lumina.h:601-608`, 종료 보고는
`src/lumina_plasma.c:14802-14818`에 있다. 원본 configuration label SHA-256 결박은
`src/lumina_atomic.c:1259-1269`에서 로드하며 synthetic top-stage 확장과 해제도 함께
처리한다. 이 표의 상세 6+9행 및 A2-07 6행/A2-08 4행 재배치는
`docs/A2_01_DISPOSITION_LEDGER.md`의 A2-06 ADDENDUM에 기록했다.

## 게이트 구현과 산출물

게이트 드라이버는 `scripts/a2_06_l1bb_gate.py`, 8대역 C fixture는
`tests/a2_06_l1bb_fixture.c`이다. 현재 산출물은 다음과 같다.

| 산출물 | 현재 판정 |
|---|---|
| `validation/a2_06/A2_06_L1BB_GATE.json` | `BLOCKED_MISSING_RATE_EXPORT` |
| `validation/a2_06/A2_06_AUL_CROSSWALK.json` | PASS; Lumina 2,220,953선 전부 match, 최대 A_ul 상대오차 `1.7176632725861633e-16` |
| `validation/a2_06/A2_06_AUL_UNMATCHED.csv.gz` | unmatched 전체 목록; Lumina unmatched 0, level unmatched 0 |

현재 gate ledger의 세 정합 검사는 모두 PASS다.

- same-measure: generation/frame/normalization/q-set 결박과 원시 ledger hash 확인
- 8대역 projection closure: 최대·중앙 상대변화 모두 0
- fine closure: V5 고정 cohort SHA-256
  `0c029ca15116119e2d7af4693d76988b14a452e35cae3679522629230a6c3e69`, active 74건,
  최대 상대변화 `6.957887371969882e-4`, 중앙 `2.7450390836680265e-6`

V3 §3.4 음성대조 `A2_06_NEG_1`부터 `A2_06_NEG_9`까지 모두 기대한 FAIL을
관측했다. A_ul crosswalk는 Z·ion·양쪽 원본 label hash·통계가중치로 먼저 묶고,
에너지는 중복 후보의 tie-breaker로만 사용한다. 양쪽 0과 한쪽 0도 별도 판정한다.

## 검증 결과

대형 병렬 배터리는 실행하지 않았다.

| 명령 | 결과 |
|---|---|
| `make lumina` | PASS (exit 0) |
| `make selftest_a2_05_bf_rate selftest_a2_06_line_jbar selftest_a2_06_dual_commit` | 빌드 완료 |
| `./selftest_a2_05_bf_rate` | `A2_05_BF_RATE_SELFTEST PASS` |
| `./selftest_a2_06_line_jbar` | `A2_06_LINE_JBAR_SELFTEST PASS` |
| `./selftest_a2_06_dual_commit` | `A2_06_DUAL_COMMIT_SELFTEST PASS` |
| `python3 scripts/a2_06_l1bb_gate.py --out validation/a2_06 --aul-ledger validation/a2_06/A2_06_AUL_CROSSWALK.json` | exit 0, `BLOCKED_MISSING_RATE_EXPORT` |

## 운전석 실행 명령

현재 산출물을 재검증하는 명령:

```bash
python3 scripts/a2_06_l1bb_gate.py \
  --out validation/a2_06 \
  --aul-ledger validation/a2_06/A2_06_AUL_CROSSWALK.json
```

A_ul crosswalk까지 원자료에서 다시 만드는 명령:

```bash
python3 scripts/a2_06_l1bb_gate.py \
  --out validation/a2_06 \
  --aul-only
```

lageunha에서 분리 rate export를 얻은 뒤 최종 L-1bb 판정을 내리는 명령:

```bash
python3 scripts/a2_06_l1bb_gate.py \
  --out validation/a2_06 \
  --aul-ledger validation/a2_06/A2_06_AUL_CROSSWALK.json \
  --rate-ledger /path/to/A2_06_SEPARATED_RATE_LEDGER.json
```

rate ledger schema는 `lumina-a2-06-separated-rate-ledger-v1`이며 각 행에 최소
`n_lower`, `B_lu`, `B_ul`, `jbar_truth`, `jbar_lum`, `R_lu_cmf`,
`R_ul_stim_cmf`, `view_state`가 필요하다. gate가 V5의 99.9% truth-flow active prefix,
경계 동률 포함, `f_cov`, 채널별 E1, NumPy linear P95, false-positive를 직접 판정한다.

## 남은 위험

1. 분리 NETRATE/TOTRATE export가 아직 없어 L-1bb는 의도적으로 BLOCKED다.
2. A_ul truth-weight coverage 진단값은 `0.9811252941488394`다. V4 §4에 따라 coverage를
   판정에는 쓰지 않았고, Lumina 선은 전부 crosswalk되었으며 나머지는 삭제하지 않고
   gzip 목록에 기록했다.
3. 지정 selftest 재빌드 중 기존 `src/radiation_field.c:629`, `:631`의 C11 `strdup`
   선언 경고가 관측됐다. 사용자가 이미 구현·검증 완료로 고정한 파일이므로 이번 작업에서는
   수정하지 않았고, 세 selftest는 모두 PASS했다.
4. legacy source-selection 계산은 falsifier/진단 보존을 위해 명시적 shadow로 남아 있다.
   생산율은 shadow 종료 후 checked view 값으로 무조건 재설정되며 static read trace가 이를
   검사한다.
