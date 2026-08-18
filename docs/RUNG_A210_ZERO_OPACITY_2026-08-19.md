# 단 사전등록 — A210-ZERO-OPACITY (2026-08-19)

user 지시 "A210-ZERO-OPACITY 단 세워". 발단은 S4(III 음성대조) BLOCKED
(`docs/VERDICT_DET_SPROD_S4_III_2026-08-19.md`).
조사 중 **표적이 넓어졌다** — 아래 §1 이 그 경위이며, 계약을 그에 맞춰 정의한다.

## 계약 (하나)

> **진단 경로의 실패는 진단만 무효화하고, 물리 트랜잭션을 중단시키지 않는다.**

exact-zero 불투명도 행의 처분은 이 계약의 **구현 귀결**이다(별도 계약이 아니다).

## 1. 왜 표적이 넓어졌나 — 오프라인 기전 특정 (실측)

### 1-1. χ_eff 가 정확히 0 이 되는 경로 (`src/line_net_rate.c`)

```c
material->raw_integrated_opacity      = tau * nu / (c * t_exp);
material->effective_integrated_opacity = material->raw_integrated_opacity;
...
if (policy == CMFGEN_SRCE_CHK && tau < -0.5) {   /* 음수 큰 것만 대체 */
    material->effective_integrated_opacity = CMFGEN_INTERNAL_OPACITY_TO_CGS / n;
}
```
ν>0·t>0 이므로 **χ_eff == 0 ⟺ tau == 0 정확**. SRCE_CHK 는 `tau < −0.5` 에서만 발화하므로
tau==0 을 구제하지 않는다.

그리고 상류 계약은 그런 행을 **유효로 받아들인다**:
`a210_line_saturation_add` 입구가 `tau_validity==A208_VALID || tau_validity==A208_EXACT_ZERO`.
⟹ **"tau 는 정당하게 0" 이라고 등록해 놓고, 그 행에서 S 가 미정의라며 막는다.**

또한 `exact_zero_provenance = (n_upper==0.0 && tau==0.0)` 이므로
**tau==0 이면서 n_upper>0** 인 행은 그 표지를 얻지 못한다 — 방출은 있고 순 불투명도만 0인 상태다.

### 1-2. ★진단이 물리 트랜잭션을 중단시킨다 (이것이 더 무겁다)

```c
if(independent_capture && (!source_defined || !isfinite(source_function))){
    a210_line_saturation_blocked("INDEPENDENT_SPROBE_UNDEFINED", ...); return -1;
}
```
그리고 소비자:
```c
int saturation_add = a210_line_saturation_add(...);
if(saturation_add != 0){
    status = saturation_add==-2 ? RADEQ_NONFINITE : RADEQ_TERM_SCHEMA;
    first_bad_line=line; first_bad_shell=s; break;
}
```
⟹ **opt-in read-only 진단(`LUMINA_A210_INDEPENDENT_CAPTURE`)의 행 빌더 실패가
A2-10 물리 트랜잭션의 status 를 갈아치우고 루프를 끊는다.**

이는 진단 자신의 선언과 모순된다 — 그 경로가 찍는 문자열이
`interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 publication_authority=NONE` 이다.
값을 **바꾸지는** 않지만 **결과를 없앤다**. 실측 확증:
III 런에서 `phase=REQUESTED_TE status=RADEQ_TERM_SCHEMA valid=0`,
같은 런의 다른 3국면은 정상 완주(`RADEQ_NO_BRACKET`).
노브를 끄면 이 차단은 발생하지 않는다(코드상 `independent_capture &&` 가드).

### 1-3. 계급 — 같은 실수의 세 번째 자리

"정당하게 0" 과 "무효" 를 혼동한다. SH-GAMMA **NC3**, MC-EVT **OUT_OF_GRID** 에 이어 세 번째.

## 2. 미확정 (수리 전 반드시 실측할 것)

- **어느 행인가**: III 후보 51,807행 중 tau==0·n_upper>0 인 선의 목록·개수·이온.
  현재 로그는 `first_bad_line=262210` 하나만 주고 그 행의 tau·n_upper 를 찍지 않는다.
- **왜 tau==0 인가**: (a) A2-08 이 정당한 exact-zero 로 등록했는가,
  (b) 미기록(coverage 결손)이 0 으로 남았는가. **둘은 처분이 다르다.**
- **IV 에는 왜 없었나**: 후보 211,887행에 그런 선이 없었는가, 아니면 순서상 먼저
  다른 곳에서 끝났는가.

⚠ 이 셋을 모르면 수리안을 고를 수 없다. **측정이 먼저다.**

## 3. 단계

### Z-1 계측 (이 단의 첫 실행) — 기존 봉인 로그 재사용 불가 ⟹ 계측 추가 필요

★**런 노브 (감사 지적 1 — 틀리면 런 하나를 버린다)**
카운터는 `diag->active` 안에서만 증가하고 `diag->active` 는
`LUMINA_A210_LINE_SATURATION_DIAG` 가 켜야 선다(`lumina_plasma.c:13943,13970`).
⟹ **census 런은 `LUMINA_A210_LINE_SATURATION_DIAG=2` 를 켠 채
`LUMINA_A210_INDEPENDENT_CAPTURE` 만 끈다.** 둘 다 끄면 카운터가 0회 증가하고
SUMMARY 도 안 찍힌다. `scripts/stage_a210_line_saturation_diagnostic.sh:112` 는 둘 다 켜므로
그 스테이징을 그대로 쓰면 부분합만 얻는다.

★**카운터의 정확한 스코프 (감사 지적 3 — 대장 기재 필수 문구)**
이 수는 "스캔 전체" 가 **아니다**. **shell 0 · target ion 후보행 중** `tau==0 && n_upper>0` 인 수다.
빠지는 것: shell≠0 전부 · target ion 밖 전부 · 주파수 창 밖 · 비활성 Z ·
`UNRESOLVED_CANCELLATION`/`INVALID_INPUT` 셀 · **첫 차단 이후 전부**(노브 ON 런은 접두 부분합).
SUMMARY 가 여러 줄이면 각 줄은 독립 부분합이므로 **합산 금지**.
⟹ "III 전체에 tau==0 행이 N개" 로 읽으면 틀린다.
차단 시점에 그 행의 `line·Z·ion·tau_raw·tau_validity·n_upper·A_ul·emission_per_sr·
effective_integrated_opacity·exact_zero_provenance` 를 한 줄로 찍는다.
**판정 로직 불변**(차단은 그대로 하되 무엇을 막았는지 기록한다).
또한 스캔 전체에서 `tau==0 && n_upper>0` 행을 **세어** 요약에 싣는다(막기 전에 규모를 안다).

### Z-2 계약 수리 (Z-1 결과를 보고 결정 — 지금 고르지 않는다)
후보:
| 안 | 내용 | 조건 |
|---|---|---|
| **A** 진단 격리 | 진단 실패는 `diag` 만 무효화(`diag->active=0` + 사유 기록), 물리 루프 계속 | Z-1 이 "물리적으로 정당한 행" 을 보이면 **유력** |
| **B** 행 건너뛰기 | 그 행만 skip + **건너뛴 수 보고**(NC3 정신). 기존 `scaled_emission==0 → return 0` 과 대칭 | A 와 병행 가능 |
| **C** 차단 유지 | Z-1 이 "미기록 coverage 결손" 을 보이면 차단이 옳다 — 대신 **사유 이름을 바꾼다** | (b) 인 경우 |

★A 와 C 는 배타가 아니다: **진단이 물리를 죽이지 않는다(A)** 는 (b) 인 경우에도 유지되어야 하고,
coverage 결손은 **별도 단**으로 다뤄야 한다.

## 4. 게이트

| # | 조건 |
|---|---|
| **Z1** | Z-1 계측이 III 재현런에서 차단 행의 전 필드를 찍는다. **판정 로직 불변**(차단 시점·사유 동일) |
| **Z2** | `tau==0 && n_upper>0` 행 수가 요약에 실린다. IV·III 양쪽에서 그 수를 보고 |
| **Z3** | (수리 후) **III 가 Stage-4 row 를 낸다** — S2 덧셈 항등이 III 에서도 bit 로 성립하는가 |
| **Z4** | (수리 후) **IV byte 불변 — 값 파일 한정**. `scripts/byte_parity_compare.py` Tier1 로 IV 재현런의 값 산출물(`lumina_spectrum.csv` 등)을 봉인 IV 와 대조. ⚠**stderr 로그 전체에 걸면 자동 실패**한다 — Z-1 이 SUMMARY 줄에 `zero_opacity_emitting_rows=` 를 더했고 그것은 IV 에서도 바뀌기 때문(감사 지적 2). 로그를 대조하려면 그 토큰의 정규화 규칙을 **선언**하고 Tier2 census 에 남길 것 |
| **Z5** | 음성 대조: 진단 노브 **OFF** 에서 III 가 A2-10 을 통과하는가(= 차단이 진단 탓임을 시연) |

★**Z5 가 이 단의 NC3 다** — "진단이 원인" 이라는 주장을 주입 없이 시연한다.
★**Z4 는 오늘 만든 byte 비교기의 첫 실전**이다(08-08 의 R6-4 실패 모드를 되풀이하지 않는다).

## 5. 기대 변경집합 (Z-1 한정 — Z-2 는 결과 후 재등록)

- `src/lumina_plasma.c`: `a210_line_saturation_add` 의 차단 지점에 진단 출력 1개 추가
  + 스캔 요약에 카운터 1개. **판정 분기·반환값 불변.**
- 그 외 파일 변경 없음. 물리식 무접촉.

## 5.5 ★런 발주 경로 (2026-08-19 변경)

user 지시 **"syn101 수동 제출은 금지. 해당 노드는 정상 운영중."**
⟹ Z-1 의 두 런(진단 노브 ON/OFF)은 **slurm 으로 제출**한다. tripwire 수동런을 쓰지 않는다.
파티션 순서 h200→h100→a100(full-NLTE 는 80GB 필수라 a40 제외), `--time` 명시(백필 자격).
가드: `scripts/run_manual_det_with_tripwire.sh` 가 syn101 이면 무조건 거부하도록 박았다.

## 6. 판정 절차 (개정13)

사전등록·검수·판정·감리=**Fable**(감리는 독립 컨텍스트) / 코딩=**Codex** /
빌드·실행·대장·커밋=**운전석**.

## 6.5 감사 (2026-08-19, Fable 독립 컨텍스트)

Z-1 적용 diff 를 독립 감사에 넘겨 **승인(조건부)** 을 받았다.
- 코드 수정 불필요: word-diff 전수로 판정 로직·반환값·차단 시점 불변 확인,
  호출부 26곳 전수 갱신, 포맷/인자 22:22·7:7·7:7 손 계수 + `-Wformat=2` 일치.
- 손 이관본이 Codex 원안보다 두 곳 정확(실물 `%.21Lg` 보존, `isfinite(...)?1:0`).
- 조건 3건은 전부 코드 밖이며 위 §3·§4 에 반영했다(런 노브·Z4 문언·대장 문구).
- **권고(Z-2 착수 시)**: `[A2-10][ZERO-OPACITY-WITNESS]` 접두사를
  `LINE-SATURATION-` 네임스페이스 안으로 옮길 것. 차단을 걷어내는 순간
  `scripts/compare_a210_phase_baseline_streams.py:78-79` 가 이 줄을 phase 스트림에 담아
  조용히 레코드 수 불일치로 깨진다.
- **한계 기재**: Z-1 산출은 **수 1개 + 증인 행 1개**뿐이다. §2 가 요구한 "목록" 과
  "(a) 정당한 exact-zero 인가 (b) coverage 결손인가" 는 **그 한 행에 한해서만** 판별된다.
  Z-2 안 선택 근거로 삼기 전에 이 한계를 명시할 것.

## 7. 처분 원칙

이 단은 **측정 단**으로 시작한다. Z-1 결과 없이 Z-2 를 발주하지 않는다.
발견은 조용한 대장 기재이며, 클램프·대체값으로 증상을 덮지 않는다.
