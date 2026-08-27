# 사전등록 — DET-INSTRUMENT-REPAIR: 판독 증거의 판별력·귀속 수리 (2026-08-28)

- **단 이름**: DET-INSTRUMENT-REPAIR
- **저자**: Fable (사전등록, 분담 개정14 ① — CLAUDE.md 상시 승인)
- **상태**: 발주 전 (커밋 전 — 운전석 지시)
- **성격**: ★**계측 수리 단** — 물리 무접촉. GPU 런 **0회** (§9).
- **표기**: `[저자 실행]` = 저자가 직접 실행(명령 원장 §14) · `[추정]` = 실측 아님 ·
  `[미실행]` = 저자가 실행할 수 없는 주장 — **본 문서에 0건**.

---

## §0. 경위 (개정15 면제 절 — **이 절의 값은 요구가 아니다**)

폐합한 두 단이 **수리 금지 계약** 아래 계측 결함을 대장에 기재만 해 두었다:

- **Z-2A** (`docs/VERDICT_A210_Z2A_DIAG_ISOLATION_2026-08-27.md` §3, 대장
  `docs/VERIFICATION_REGISTERS.md:358-368`): 판독기 앵커 검사의 **판별력 0** — 판독기
  라벨 ZI-B 를 판정자가 기각하고 계약 문면으로 **ZI-A 재판정**(감리 두 겹 통과).
- **L6C-R** (`docs/VERDICT_L6C_REDISPATCH_2026-08-28.md` §7 **F-1**, 대장
  `docs/VERIFICATION_REGISTERS.md:426-436`): PR-0 sacct 증거의 **교차-클러스터
  provenance** — 게이트 실질·분기 무영향, provenance 만 오염. "차기 판독기 수정 발주 시
  반영 후보(클러스터 지정 또는 JobName conjunct — 설계는 그 단 몫)" — **이 단이 그 단이다**.

이 단은 그 두 결함(아래 A·B)을 고친다. 두 단의 판정은 **재판정하지 않는다**(§8).

---

## §1. 계약 (하나)

> **판독 증거의 판별력·귀속 계약**: 게이트 판독기가 소비하는 증거는
> (i) **성한 봉인 실물을 통과**시키고 주입 결함을 이름 있는 사유로 거부해야 하며
> (판별력 — 앵커 판별식은 반복별 등록값을 반복 인덱스로 적용한다),
> (ii) **그 증거가 주장하는 대상에서 온 것**이어야 한다
> (귀속 — 외부 채널(sacct) 레코드는 클러스터·작업명 귀속이 검증된 것만 수용하고,
> 검증 불능은 통과가 아니라 이름 있는 FAIL 이다 — fail-closed).

물리 계약이 아니다. 물리량·분기·판정식(타 단 소유)은 전부 무접촉(§8).

---

## §2. 결함 특정 (전 행 번호 [저자 실행] — `git status` 양 파일 clean, z2a 판독기
커밋 `406712d`(유일)·l6c 판독기 커밋 `9b08dae`. ★이 단은 C 소스를 인용하지 않으므로
전처리 死블록 규정(`lumina_plasma.c:1324-1693`·`lumina_cuda.cu:1083-2063`)은 해당 없음)

### 2-A. Z2A 판독기 앵커의 반복 스코프 — 판별력 0

`scripts/analyze_a210_z2a_probe.py` [저자 실행 직독]:

- `ANCHOR_LINE=669992`(`:28`) · `ANCHOR_BITS`(`:29-34`) = **iter0 값 고정** 4필드.
- `anchor_matches(raw_rows)`(`:188-210`, hex 비교 `:206`)가 전역 `ANCHOR_BITS` 를 참조.
- 반복 루프 `:255` (`for iteration, (actual, reference) in enumerate(...)`) 안의
  `:290-293` (`for raw_rows in (actual_zero, reference_zero):`)에서
  **양 반복 × run·census 양쪽**에 적용된다.
- 앵커 행의 반복 간 변화는 **정당하다** [저자 실행 — 봉인 read-only]:
  census `stderr.log:1242`(iter0)·`:226118`(iter1), R-II `stderr.log:1244`·`:226122` —
  두 쌍이 bit-동일이고, iter0→iter1 로 `n_upper` `0.0004437688873013011`→
  `0.00045264235457262295`, `emission_per_sr` `2.7310010625545125e-16`→
  `2.7856093265405119e-16`. hex 변환 [저자 실행]: iter0 `0x1.d1534dba20c26p-12`/
  `0x1.3adcec9caadc0p-52`, iter1 `0x1.daa141c1395e6p-12`/`0x1.4128acfd24836p-52`.
- ⟹ **봉인 census 자신을 넣어도 FAIL** — 판별력 0. 저자 재현 [저자 실행, §14 명령 C3]:
  `--mode R-II --stderr <ZO>/stderr.log --census-stderr <ZO>/stderr.log` →
  `status=FAIL branch=ZI-B blocking=['SPROBE_COUNTER_MISSING','ANCHOR_FIELD_MISMATCH']
  f_bit=1.0` rc=4 (4.5초). 판정문 §3-(b)와 동일 결과.
- 수리 전 기준선 [저자 실행, §14 명령 C4]: `--stderr <RII> --census-stderr <ZO>` →
  `FAIL ZI-B ['ANCHOR_FIELD_MISMATCH'] f_bit=1.0` rc=4 — 봉인 report M14 와 동일.
- ★**실해**: 이 오탐이 R-II 런처 수용 블록을 죽여 `sacct FAILED 70:0` 을 만들었다
  (`slurm-324557.out` 정확 1행 `DET_FLIGHT_FATAL Z2A R-II analyzer failed` — 판정문 M6;
  모델은 `model.rc=0`·04:34:36 완주). 판정자가 라벨을 기각하고 ZI-A 재판정. **E-I1 문면
  FAIL 로 남았고 R-III/R-IV 는 제출 불가**가 됐다(판정문 §5·§9).
- ★**selftest 가 못 잡은 이유**: 성한 픽스처가 `synthetic_zero_row()`(`:372`) **하나를
  양 반복에 재사용**(`synthetic_logs` `:402`, `for _ in range(2)` `:405`) — 반복 간
  진화가 없는 픽스처라 원리적 검출 불능. 현행 NC 는 NC-I0~I6 7종(`:426-441`, `:457`).

### 2-B. L6C-R 판독기 sacct 의 교차-클러스터 provenance

`scripts/analyze_l6c_redispatch.py` [저자 실행 직독]:

- `check_pr0_unsealed`(`:1120`)가 `:1135-1138` 에서
  `sacct -j 324557 -X -n -P -o JobIDRaw,State,ExitCode` 를 **클러스터 무지정**으로
  subprocess 호출. `RII_JOB_ID="324557"`(`:39`) ·
  `NONTERMINAL_SACCT_STATES`(`:71-73`) = {PENDING,RUNNING,REQUEUED,RESIZING,SUSPENDED}.
  파스 `:1143-1152` — 레코드 3토큰·계수≠1 → `RII_NOT_TERMINAL`(fail-closed 는 이미 있다).
- **실해(F-1)**: 운전석이 grammar-debug 에서 집행해 **grammar 클러스터의 동번호 무관
  작업** 레코드(`UPP_QE68_SCF`, Submit 2026-06-28)가 verdict JSON 에 기록됐다. syn 정본은
  `LUMINA_Z2A_II FAILED 70:0`. 게이트 분기는 불변이었다(진 레코드도 종단) — **결함은
  provenance 이지 판정이 아니다**(L6C-R 판정문 §7 F-1·부록 A-4).
- **저자 라이브 재현** [저자 실행, §14 명령 C7]: grammar-debug 에서 현행
  `check_pr0_unsealed(docs)` 호출 → **`status=PASS sacct_records=['324557|COMPLETED|0:0']`**
  — 오귀속 레코드가 지금도 침묵 수용된다.
- **클러스터 실측 4건** [저자 실행, §14 명령 C5·C6]:
  - syntax: `ClusterName=syntax` · `sacct -j 324557 -X -n -P -o
    JobIDRaw,JobName,State,ExitCode,Cluster` → `324557|LUMINA_Z2A_II|FAILED|70:0|syntax`.
  - grammar: `ClusterName=grammar` · 같은 명령 → `324557|UPP_QE68_SCF|COMPLETED|0:0|grammar`.
  - **교차 조회는 양방향 모두 빈 출력·rc=0**: syntax 에서 `-M grammar`, grammar 에서
    `-M syntax` — 둘 다 레코드 0건, 종료코드 0. (운전석 실측 「grammar 에서 `-M syn` 빈
    출력」의 확장 재확인 + 방향 대칭.)
- ★`NC-R14/R15`(`:1855-1893` 계열)는 `sacct_text` **텍스트 주입** 픽스처 — subprocess
  채널을 우회하므로 이 계급을 원리적으로 못 잡는다(대장 :435-436 명시와 합치).

---

## §3. 수리안과 판별 설계

### 3-A. 결함 A — 자매 판독기 해법의 **그대로 계승** (판정)

**방향**: `analyze_l6c_redispatch.py` 가 이미 구현·검증한 해법을 계승한다 —
`ANCHOR_BITS_BY_ITERATION`(그쪽 `:49-62`) + `anchor_ok(block, iteration)`(그쪽
`:439-459`, 반복 인덱스 검사 `:442-443`) + 반복-구별 픽스처(NC-R13, 그쪽 `:1819-1848`) +
census 자기입력 양성대조(`--census-selfcheck`/PR-3, 그쪽 `:1249-`·`:2036`). 그 해법은
L6C-R 단에서 게이트 전건 통과·판정·감리 두 겹까지 폐합됐다(`e11de26` 계열).

**기각한 대안 3건** (판별력 판별식):

1. **앵커를 iter0 에만 적용** (Z-O 판독기 전례 `analyze_a210_zero_opacity_census.py:216-218`):
   iter1 판별력을 **버린다**. Z2A 판독기가 검사를 iter1·census 로 넓힌 의도 자체는
   정당했고(bit-거울 확장), 틀린 것은 등록값의 스코프였다. 의도를 죽여 결함을 없애는
   것은 판별력 손실이다 — 기각.
2. **허용오차 비교**: 정확해(결정론 재현)가 위반할 수 없는 검사를 굳이 느슨하게 만들면
   섭동 런이 통과할 수 있다 — 「정확해가 위반 가능한 가드·표현은 잘못된 것」의 쌍대로,
   **정확해만 통과 가능한 검사를 무디게 만드는 것도 잘못**이다. clamp/floor 금지 규약의
   판별식 적용 — 기각.
3. **상대 대조만** (run≡census bit, 절대 등록값 폐지): census 자체 오염을 침묵 통과시킨다.
   절대 앵커는 상대 대조와 **독립 축**이다(L6C-R 사전등록 RD-6 근거 그대로) — 기각.

**반복별 등록값** (전원 [저자 실행] — 봉인 read-only·hex 재계산·자매 등록값 대조 3중):

| 필드 | iter0 | iter1 |
|---|---|---|
| `tau_validity` | `2` | `2` |
| `n_upper` | `0x1.d1534dba20c26p-12` | `0x1.daa141c1395e6p-12` |
| `A_ul` | `0x1.0b6516044e436p-1` | `0x1.0b6516044e436p-1` |
| `emission_per_sr` | `0x1.3adcec9caadc0p-52` | `0x1.4128acfd24836p-52` |

출처: 봉인 census `stderr.log:1242/:226118` + 봉인 R-II `:1244/:226122` (bit-동일 쌍,
§2-A) · 자매 판독기 `ANCHOR_BITS_BY_ITERATION:49-62` (동일값) ·
`docs/RUNG_L6C_REDISPATCH_2026-08-27.md` «RD-6 반복별 앵커 등록값» 표 ·
`docs/RUNG_BIT_LOCKSTEP_PRODUCER_2026-08-26.md:184`.

**수리 내용** (`scripts/analyze_a210_z2a_probe.py`):

1. `ANCHOR_BITS` → `ANCHOR_BITS_BY_ITERATION`(위 표, tuple). iter0 값은 기존 등록값
   **불변**.
2. `anchor_matches(raw_rows)` → `anchor_matches(raw_rows, iteration)`; 반복 인덱스 범위
   밖 → `ANCHOR_ITERATION_INVALID`(자매 계승). 호출부 `:290-293` 이 루프의 `iteration`
   을 전달. R-III/R-IV 경로(`analyze_r3_or_r4` — line=262210 존재 검사)는 **무접촉**.
3. `--census-selfcheck` 모드 신설(자매 PR-3 계승): census stderr **스트림만** 파스 —
   양 반복 앵커 행 = 등록값 bit 일치 ∧ SUMMARY 정확 2건·각 `complete=1` → rc=0, 위반 시
   이름 있는 사유 rc=4. ★`sprobe_undefined_rows` 는 요구하지 않는다(봉인 census 는 토큰
   이전 산출물 — R-II 모드 자기입력이 완전 PASS 할 수 없는 이유이자 이 모드가 필요한 이유).
4. selftest 픽스처 **반복-구별화**: 성한 픽스처의 iter0/iter1 앵커 행이 위 표의 값
   (십진 원문)을 각각 싣는다. **NC-I7** 신설: iter1 앵커 행에 iter0 값 주입 →
   `ANCHOR_FIELD_MISMATCH` FAIL, 복원 → PASS (자매 NC-R13 의 이식 — 원 결함이 selftest
   를 통과한 그 맹점을 제거). **NC-I8** 신설: `--census-selfcheck` 모드 자체의 성한
   픽스처 PASS ∧ 변조 FAIL 팔. 기존 NC-I0~I6 존속(NC-I2 등 십진 치환 픽스처는 첫
   출현(iter0)에 걸리므로 유효 유지 — 구현이 확인).

### 3-B. 결함 B — 후보 판정: **레코드 provenance 삼중 검증(기계) + 집행 노드(규율)**

**후보 열거와 판정**:

| 후보 | 판정 | 근거 |
|---|---|---|
| ① 클러스터 명시 `-M` | **채택 (부분)** | 클러스터 이름 실측: syn 측 `ClusterName=syntax` [저자 실행 scontrol]. ⚠**교차 조회는 안 된다** — `-M` 은 양방향 모두 빈 출력·rc=0 [저자 실행] ⟹ `-M syntax` 를 박으면 grammar 계열에서의 집행이 「오귀속 레코드」에서 「레코드 0건 → 기존 `RII_NOT_TERMINAL` fail-closed FAIL」로 바뀐다. 즉 ①은 조회를 되게 만드는 장치가 아니라 **오집행을 침묵에서 이름 있는 FAIL 로 바꾸는 장치**다 |
| ② `JobName` conjunct | **채택** | 레코드가 기대 작업(`LUMINA_Z2A_II`)인지 직접 검증 — 실해의 레코드(`UPP_QE68_SCF`)를 정확히 거부한다 |
| ③ `Cluster` 출력열 + 일치 검증 | **채택 (신규 후보)** | `-o` 에 `Cluster` 열 추가 — 레코드 자체가 귀속을 자백한다. `-M` 이 누락·오기되어도 로컬 DB 레코드의 `Cluster` 열이 등록값과 불일치 → 이름 있는 FAIL. 실측: syntax 레코드 `...|syntax`, grammar 레코드 `...|grammar` [저자 실행] |
| ④ 집행 노드 규율만 (도구 무변경) | **기각 (단독으로는)** | 규율은 이미 있었고(오프라인 티어 규약) **그 규율이 실해를 만들었다** — 운전석이 sacct 를 연산으로 분류해 grammar-debug 로 보냈다(대장 :433 「운전석 오류」). 규율 위반이 침묵 통과되는 구조가 결함이다. 단 규율은 ①~③의 **잔여**(아래)로서는 필수 — 병기 |

**수리 내용** (`scripts/analyze_l6c_redispatch.py`):

1. 등록 상수 신설: `RII_JOB_NAME="LUMINA_Z2A_II"` · `RII_CLUSTER="syntax"`
   (둘 다 [저자 실행] 실측 — §2-B).
2. `:1136` 명령 →
   `sacct -M syntax -j 324557 -X -n -P -o JobIDRaw,JobName,State,ExitCode,Cluster`.
3. 파스 5토큰: `JobIDRaw` 일치(기존) ∧ `JobName==RII_JOB_NAME` ∧ `Cluster==RII_CLUSTER`
   — 불일치 → **`SACCT_PROVENANCE_MISMATCH`**(이름 있는 신설 사유, 원문 레코드를 detail
   에). 레코드 0건·계수≠1 → 기존 `RII_NOT_TERMINAL` **유지** · sacct rc≠0 →
   `SACCT_UNAVAILABLE` **유지**. **조회 불능이 통과가 되는 경로는 없다** — fail-closed
   불변.
4. verdict JSON `sacct_records` 에 5열 원문 기록(provenance 증거의 자기 기술).
5. 기존 텍스트 주입 픽스처 전건 5토큰 갱신(`check_pr0(..., sacct_text=...)` 호출부
   `:1734,:1744,:1749,:1758,:1855,:1864,:1882,:1891,:1910,:1920` — 구현이 전수 확인).
6. **NC-R17** 신설(텍스트 주입 — 검사 로직): JobName 오기 주입·Cluster 오기 주입 각 →
   `SACCT_PROVENANCE_MISMATCH` FAIL / 성한 5열 → PASS.
7. **NC-R18** 신설(★실물 채널 모의 — PATH shim): selftest 가 스크래치에 가짜 `sacct`
   실행파일을 두고 PATH 선두 주입 후 `sacct_text=None` 로 호출 —
   (a) 실해 레코드 그대로(`324557|UPP_QE68_SCF|COMPLETED|0:0|grammar`) 방출 →
   `SACCT_PROVENANCE_MISMATCH` FAIL (b) 정본 5열 방출 → PASS (c) 빈 출력·rc=0 →
   `RII_NOT_TERMINAL` FAIL (d) shim 이 수신 argv 를 기록 → selftest 가 `-M syntax` 와
   5열 `-o` 문자열의 실재를 검증(**명령 표류 차단** — 등록 명령과 호출 명령의 이탈을
   기계로 잡는다). ⚠shim 디렉토리는 `TMPDIR` 존중(감리-B 환경 실측 `/tmp` EROFS —
   L6C-R 판정문 부록 A-4 NOTE; `/tmp` 는 노드-로컬이므로 selftest 는 단일 노드 완결).

**기계로 막는 것 vs 규율로만 되는 것** (판정):

- **기계**: (i) 오귀속 레코드의 수용 — 어느 노드에서 돌든 등록 삼중(JobID·JobName·Cluster)
  불일치는 이름 있는 FAIL (ii) 오노드 집행의 침묵 통과 — `-M syntax` 핀으로 타 클러스터
  집행은 빈 출력 → fail-closed FAIL (iii) 명령 표류 — NC-R18(d) argv 검증.
- **규율로만**: **정본 증거의 취득**(PASS 도달) — 교차 조회가 양방향 불능임이 실측이므로
  (§2-B), syn slurmdbd 에 붙은 노드(실측 성립: syntax 로그인)에서 집행해야만 PASS 가
  가능하다. 기계는 이를 대체할 수 없고, 위반을 **가시화**할 뿐이다. ⟹ §10 집행 조건에
  명기. (규약 「`squeue -u kjhan` 은 계정 범위」의 클러스터 판 전역 명문화는 user 몫 —
  이 단은 계약 내 명기까지만.)

**이 계급의 음성대조 — 원리 판정**: 텍스트 주입 단독으로는 **원리적으로 불가**
(subprocess 채널 우회 — 대장 명시와 저자 재확인). 가능한 것은 **두 겹**이고 둘 다 이
단이 등록한다: ① NC-R18 PATH-shim **실물 채널 모의**(selftest 상설 — 채널·argv·파서·
판정 로직을 잡는다; 못 잡는 잔여 = 「실행 노드의 실제 slurmdbd 귀속」, shim 은 진짜
sacct 이 아니므로) ② **라이브 교차-클러스터 음성대조**(G-B3, grammar-debug 실기 — 그
잔여를 역사적 실해 채널 그 자체로 잡는다; 단 상설 selftest 가 아니라 집행 게이트라
환경 의존이 한계다. grammar 측 동번호 레코드가 장래 소멸해도 빈 출력 → 이름 있는
FAIL 로 대조는 유효 — 기대 라벨 집합을 §6 에 사전등록). **selftest 단독 완결은 불가** —
한계로 기재한다.

---

## §4. 기대 변경집합 — PF-1 대상

### 변경집합 — IR (표)

| 파일 | 변경 성격 |
|---|---|
| `scripts/analyze_a210_z2a_probe.py` | §3-A 1~4 (반복별 앵커·시그니처·`--census-selfcheck`·NC-I7/I8·표지 심볼 `A210_INSTRUMENT_REPAIR`) |
| `scripts/analyze_l6c_redispatch.py` | §3-B 1~7 (등록 상수·`-M`/5열 명령·`SACCT_PROVENANCE_MISMATCH`·픽스처 5토큰·NC-R17/R18·표지 심볼 `A210_INSTRUMENT_REPAIR`) |

### 변경집합 끝

**이 밖의 변경 0**: `src/` 0 · 봉인 0 (read-only 열람만) · 타 스크립트 0 · 계약 문서
(본 문서) 외 docs 0. 산출물 거처: `validation/instrument_repair/` 신설(소형 JSON·로그
— 소스 아님, PF-1 밖). 구현은 두 파일에 표지 심볼 `A210_INSTRUMENT_REPAIR` 를 넣는다
(PF-1 의 grep 축 — 발주 전 0파일 [저자 실행], 구현 후 정확히 이 2파일).

---

## §5. 게이트 (각각 음성대조·봉인 접촉 표기)

★집행 순서 (개정16 ②): **코드 검수 Q0**(«이 산출물을 봉인·실물에 돌려도 되는가») 답
전에는 **봉인 무접촉 게이트만**(G-A1·G-B1·G-B2·G-B3) 집행 가능. G-A2~G-A5 는 Q0 답 후.
전 게이트 산출물은 봉인 밖(`validation/instrument_repair/`; 노드-로컬 `/tmp` 에 남기지
않는다).

| 게이트 | 판정식 | 음성대조 | 봉인 접촉 |
|---|---|---|---|
| **G-0 프리플라이트** | 본 문서 `check_prereg_preflight` rc=0 (§12 기재) + 발주 시 `check_dispatch_preflight` rc=0 | 게이트 자신의 `--selftest` (NC-P0·NC-D0 포함, 기존) | 무 |
| **G-A1 z2a selftest** | `--selftest` NC-I0~I8 전건 — 성한 반복-구별 픽스처 PASS ∧ 주입 전건 이름 있는 FAIL ∧ 복원 PASS | **NC-I7**(iter1 에 iter0 값 주입 → `ANCHOR_FIELD_MISMATCH`) — 원 결함의 재발 차단 검사 그 자체 | 무 (합성 픽스처만) |
| **G-A2 census 자기입력 양성대조** (PR-Z1) | `--census-selfcheck <ZO>/stderr.log` → **rc=0** — «봉인 census 자신을 넣으면 PASS 해야 한다» 의 직접 형식 | G-A4 가 이 게이트의 음성대조 | **유** — census read-only |
| **G-A3 결함 명령 쌍대** (PR-Z2) | 원 결함 시연 명령 그대로 `--mode R-II --stderr <ZO> --census-stderr <ZO>` → blocking **정확** `['SPROBE_COUNTER_MISSING']`(앵커 계열 0건·f_bit=1.0; 잔여 FAIL 은 census 의 토큰 이전 산출물성에 의한 구성적 잔여 — §3-A 3) | 수리 전 실측(§2-A: 같은 명령이 `ANCHOR_FIELD_MISMATCH` 동반)이 대조군 | **유** — census read-only |
| **G-A4 변조 사본 음성대조** | census stderr **스크래치 사본**의 iter1 앵커 `n_upper` 1자리 변조 → `--census-selfcheck` 이름 있는 FAIL(`ANCHOR_FIELD_MISMATCH`) | 이 게이트 자신 | **유** — 사본 생성 시 read-only (변조는 사본에만; `/gpfs` 스크래치 사용, 노드-로컬 `/tmp` 금지) |
| **G-A5 소급 재판독** (PR-Z3, §7) | `--mode R-II --stderr <RII>/stderr.log --census-stderr <ZO>/stderr.log` → **status=PASS · branch=ZI-A · f_bit=1.0 · blocking=[]** — report 는 봉인 밖 | 수리 전 기준선 [저자 실행, §2-A: `FAIL ZI-B ['ANCHOR_FIELD_MISMATCH']`] 이 대조군 | **유** — R-II·census read-only ×2 (봉인 report 무접촉·지문 무접촉) |
| **G-B1 l6c selftest** | `--selftest` NC-R0~R18 전건 (기존 R0~R16 존속 — 픽스처 5토큰 갱신 후에도 전건; 신설 R17·R18) | **NC-R17**(오귀속 주입)·**NC-R18**(실물 채널 shim — 실해 레코드 원문 재현 포함) | 무 (합성/스크래치; 기존 selftest 의 봉인 불접촉 계약 승계) |
| **G-B2 라이브 양성대조 (syntax)** | 수리된 PR-0 unsealed 반쪽을 syntax 에서 — PASS ∧ 레코드 정확 `324557\|LUMINA_Z2A_II\|FAILED\|70:0\|syntax` | G-B3 이 쌍대 | 무 (docs+sacct 만 — `check_pr0_unsealed` 는 봉인 무접촉, `:1120-1161` 실측 구조) |
| **G-B3 ★라이브 교차-클러스터 음성대조 (grammar-debug)** | 같은 호출을 grammar-debug 에서 — **이름 있는 FAIL 의무**. 1차 기대 = `RII_NOT_TERMINAL`(record_count=0; 근거 = `-M` 교차 빈 출력 rc=0 [저자 실행]) · 허용 집합 §6 · **PASS 는 즉시 수리 불성립(IR-F2)** | 이 게이트 자신 (역사적 실해 채널의 실기 재현 — 수리 전 같은 자리 실측: `PASS ['324557\|COMPLETED\|0:0']` [저자 실행, §2-B]) | 무 |

---

## §6. 기대치·분기 사전등록 (라벨 명시 부여 — UNRESOLVED 금지, Z-2A/L6C-R 판독기 전례 계승)

**상위 분할** (PF-2 기계 스윕 대상): 게이트 전건 충족 → **IR-A** / 그 외 → **IR-F**.

| 라벨 | 판정식 | 등급 | 처분 |
|---|---|---|---|
| **IR-A** (수리 성립) | G-0~G-B3 전건 충족 | 등급 1 — 전 기대가 실측·선례 기반 | 폐합 절차(판정→감리 A·B)로 |
| **IR-F1** (소급 불일치) | G-A5 산출이 PASS·ZI-A·f_bit=1.0·blocking=[] 과 다름 | 등급 1 (기대 이탈 시) | **BLOCKED** — 수리안 결함으로 처분(재작업)·user 에스컬레이션. ★Z-2A 판정(ZI-A)은 불변 — §7·§8 |
| **IR-F2** (라이브 음성 미발화) | G-B3 이 PASS | 등급 1 | 수리 불성립 — provenance 검사 미작동, 재작업 |
| **IR-F3** (게이트 실패 일반) | 그 밖의 G-* FAIL | 등급 1 | 사유 특정 후 재작업 (판정런 소모 없음 — GPU 0) |
| **IR-F4** (환경 차단) | sacct 조회 불능(syntax 에서도)·봉인 접근 불능·grammar-debug 불통 등 | 등급 2 — 환경 의존 | BLOCKED — 집행 기록에 사유·주인 기재 |
| **IR-X** (분할 밖 관측) | 위 어느 판정식에도 들지 않는 산출 | — | 라벨 명시 부여 후 정지·보고 (침묵 금지) |

**세부 기대치**:

- G-B3 허용 FAIL 사유 집합 = {`RII_NOT_TERMINAL`(1차 기대 — record_count=0),
  `SACCT_PROVENANCE_MISMATCH`, `SACCT_UNAVAILABLE`}. 1차 기대의 근거는 실측(`-M` 교차
  빈 출력 rc=0)이나, grammar-debug 로컬 slurm conf 의 경유 세부는 [추정]이므로 집합으로
  등록한다(등급 2). 집합 밖 사유 → IR-X.
- G-B2 기대 레코드는 등록 문자열 그대로(등급 1 — [저자 실행] §2-B). sacct State 는 종단
  상태라 불변 기대가 정당하다(L6C-R 계약 §14 전례).
- G-A2·G-A3·G-A5 기대(등급 1)의 근거: 수리 전 실측에서 앵커 사유를 제외한 **모든 검사가
  이미 통과**했다(§2-A — blocking 이 앵커(+구성적 sprobe)뿐, f_bit=1.0).

---

## §7. 소급 재판독 판정 (③) — **한다. 지위는 «계측 검증 관측», 판정 불변**

**판정**: 수리된 판독기로 봉인 R-II 를 **다시 읽는다** (G-A5).

- **읽는 근거**: 수리 주장이 합성 픽스처에만 얹히면 원 결함의 통과 경로(픽스처 현실성의
  구멍)를 그대로 반복한다. 수리된 계측은 **자기가 실패했던 바로 그 실물**에서 판별력을
  시연해야 한다. 봉인은 read-only 로만 열고(지문·봉인 report 무접촉), 새 report 는 봉인
  밖에 쓴다 — 자매 단이 같은 형식(PR-3 예외 접촉)을 이미 집행했다.
- **산출의 지위**: **관측(계측 검증의 양성대조)이다 — 판정을 바꾸지 않는다.** Z-2A 의
  ZI-A 는 판정자가 계약 §8-2 판정식 문면을 오탐 무관 1차 실측으로 채워 재판정했고 감리
  두 겹을 통과했다. G-A5 의 기대 산출(PASS·ZI-A)은 그 판정과의 **일치 확인**이다 —
  계측이 판정을 따라잡는 것이지 판정이 움직이는 것이 아니다.
- **불일치 시**(IR-F1): **수리안의 결함**으로 처분한다 — Z-2A 재판정이 아니다. BLOCKED
  후 user 에스컬레이션. (판정문이 오탐 제거 후 잔여 사유 0 을 실측으로 확정했으므로
  (§6 세부 기대치), 불일치는 수리 구현의 새 결함을 가리킨다 — fail-closed.)
- **E-I1 문면 FAIL·R-III/R-IV 제출 불가**(Z-2A 판정문 §5·§9)는 이 단이 **건드리지
  않는다** — 그 항목의 해소(런처 수용 실물 확보·R-III/R-IV 제출)는 GPU 런이 필요한 별도
  단이며, 이 수리는 그 단의 **전제를 회복**할 뿐이다(§13).

---

## §8. 처분 원칙 / 하지 않는 것

1. **물리 무접촉**: `src/` 0 · 물리량·판정식·분기 의미론 0 · 스펙트럼 0.
2. **타 단 재판정 0**: Z-2A ZI-A(계약 문면 재판정·감리 두 겹 통과)·L6C-R C-M-S·L6 분기
   B·A207·Z-O·BITLOCK 전부 불변. G-A5 는 §7 의 지위로만.
3. **반상태 수리는 이 단이 아니다** (user 결정 대기 — 무접촉).
4. **F-2**(σ 반복 스코프)·**F-3**(proxy≢capture)·Z-2A 대장 결함 2번(판정문 리터럴
   fail-open 벡터 — 본문 리터럴 제거로 1차 차단 상태)은 **이 단 밖** — 별도 단 후보로
   대장에 존속.
5. z2a 판독기의 «어떤 오류든 ZI-B» 확장(판정문 §2 가 관측한 구현 확장)은 **고치지
   않는다** — 분기 의미론 변경은 계측 수리가 아니고, 앵커 오탐 제거로 그 확장의 실해
   경로(오탐→ZI-B)는 소멸한다. 필요 시 별도 단.
6. 봉인 **읽기 전용** 절대(지문 파일·봉인 내 report 무접촉·하드링크 금지) ·
   `git add -A` 금지 · 커밋은 폐합 시 운전석(계약 하나=커밋 하나).
7. 발견의 처분 = 조용한 대장 기재 (수리 중 새 결함 발견 시에도 — 이 단의 범위를 늘리지
   않는다).
8. 두 파일 밖 수리 금지 · 등록 상수 밖 값 변경 금지 · clamp/floor/cap/jitter/repair 0.

---

## §9. GPU 런 판정 — **0회로 등록한다**

전 게이트가 합성 픽스처 selftest·봉인 read-only 파스(실측 4.5초/회)·sacct 조회·
grammar-debug 경량 python 이다. 모델 실행·물리 산출이 없고, 소급 재판독은 기존 봉인을
읽는다. 오프라인 3요건: 기전 특정 = 두 판정문 + 본 문서 §2 재현 [저자 실행] · 수리안
오프라인 검증 = 자매 해법의 폐합 실적 + §2 실측 · 기대치 = 본 문서. **판정런(GPU) 0.**

---

## §10. 집행 조건 (결함 B 의 규율 절반)

- **live sacct 를 포함하는 게이트·판독의 집행 노드 = syntax** (syn slurmdbd 귀속 실측,
  §2-B). grammar-debug 는 G-B3(음성대조)의 **의도된 오노드**로만 쓴다.
- selftest·판독(비 sacct)은 티어 규약대로 grammar-debug 가능 — 수리 후에는 오노드
  집행이 침묵 오귀속이 아니라 이름 있는 FAIL 로 나타난다(그것이 이 수리의 실질).
- 산출물은 봉인 밖 `validation/instrument_repair/`. `/tmp` 는 노드-로컬 — 게이트 산출
  금지. 스크래치 사본은 `/gpfs`.

---

## §11. 프리플라이트 선언 블록

⚠`references[].flags_planned` 는 **발주 전 시점 술어**다(지금 부재해야 함 — [저자 실행]
grep 0건 확인). 구현 착지 후 본 게이트를 재실행하면 정의상 FAIL 하며, 그것은 결함이
아니라 시점 술어의 만료다(발주 전 rc=0 기록이 판정 대상).

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — IR (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/[a-z0-9_\\-]+\\.py",
    "symbol": "A210_INSTRUMENT_REPAIR",
    "roots": ["scripts"],
    "expected_extra": ["scripts/analyze_a210_z2a_probe.py",
                       "scripts/analyze_l6c_redispatch.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.02], [0.98, 1.0]],
    "metrics": {
      "ok": "sum(1 for x in v if x >= 0.5)/len(v)"
    },
    "rules": [
      {"name": "IR-A", "predicate": "ok >= 1.0"},
      {"name": "IR-F", "predicate": "ok < 1.0"}
    ],
    "adversarial_fixtures": [
      {"name": "all-gates-pass", "mix": [[1.0, 10]]},
      {"name": "single-gate-fail-retro-mismatch", "mix": [[1.0, 9], [0.0, 1]]},
      {"name": "all-gates-fail", "mix": [[0.0, 10]]}
    ]
  },
  "references": [
    {"path": "scripts/analyze_a210_z2a_probe.py",
     "flags_existing": ["ANCHOR_BITS", "anchor_matches", "NC-I6"],
     "flags_planned": ["ANCHOR_BITS_BY_ITERATION", "census-selfcheck", "NC-I7",
                       "A210_INSTRUMENT_REPAIR"]},
    {"path": "scripts/analyze_l6c_redispatch.py",
     "flags_existing": ["JobIDRaw,State,ExitCode", "NC-R13", "ANCHOR_BITS_BY_ITERATION",
                        "RII_NOT_TERMINAL", "census_selfcheck"],
     "flags_planned": ["SACCT_PROVENANCE_MISMATCH", "RII_CLUSTER", "NC-R17", "NC-R18",
                       "JobName", "A210_INSTRUMENT_REPAIR"]},
    {"path": "docs/VERDICT_A210_Z2A_DIAG_ISOLATION_2026-08-27.md",
     "flags_existing": ["판별력 0", "ANCHOR_FIELD_MISMATCH"]},
    {"path": "docs/VERDICT_L6C_REDISPATCH_2026-08-28.md",
     "flags_existing": ["F-1", "UPP_QE68_SCF"]},
    {"path": "docs/RUNG_L6C_REDISPATCH_2026-08-27.md",
     "flags_existing": ["NC-R13", "0x1.daa141c1395e6p-12"]},
    {"path": "docs/VERIFICATION_REGISTERS.md",
     "flags_existing": ["판별력 0", "클러스터 판"]}
  ]
}
```

---

## §12. 집행 기록 (★장부 정본 — 판정문·감리문은 이 절을 지목만 한다)

**필수 4항목**:

1. 발주 프리플라이트(`check_dispatch_preflight`) rc: __ (운전석 기입)
2. `[미실행]` 주장 실행 결과: **본 사전등록의 [미실행] 주장 0건** — 전 사실 주장 저자
   실행(§14). 운전석 추가 실행분: __
3. 분장 장부: 아래 표 (검수 완료 후 기입)
4. 발주 중단 유무·사유·주인: __

**사전등록 프리플라이트 (G-0 전반부)**: `python3 scripts/check_prereg_preflight.py
docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md --root .` → **rc=0** [저자 실행, grammar-debug,
2026-08-28 — §14 명령 C9]

**분장 장부** (개정14·16·17 — 운전석 칸 비움):

| 단계 | 규약상 담당 | 실제 | 위반 |
|---|---|---|---|
| ① 사전등록 | Fable (개정14 ①) | Fable fresh — 본 문서 (2026-08-28) | 없음 |
| 발주서 작성·발주 행위 | 운전석 (앞겹=본 문서 그대로 첨부·뒷겹 재서술 금지) | | |
| 발주 프리플라이트 집행 | 운전석 (grammar-debug) | | |
| 구현 | Codex | | |
| ② 코드 검수 (Q0 선행 — 개정16 ②) | Fable fresh | | |
| 게이트 집행 — 봉인 무접촉 (G-A1·G-B1·G-B2·G-B3) | 운전석 (Q0 답 전 병행 가능) | | |
| 게이트 집행 — 봉인 접촉 (G-A2~G-A5) | 운전석 (★Q0 답 후에만) | | |
| ③ 판정 | Fable fresh | | |
| ④ 감리-A (물리·주장 — 개정17) | Fable fresh (①③ 과 다른 컨텍스트) | | |
| ★감리-A 발주 집행 주체 | 운전석 (Agent 위임 — CLAUDE.md 상시 승인 ④) | | |
| 감리-B (재현 — 개정17) | Codex read-only (판정문 실측 주장 재현 한정) | | |
| 폐합·커밋 | 운전석 (감리 A·B 둘 다 통과 후 — 개정17 폐합 조건) | | |

---

## §13. 승계 기재

- **결함 A 출처**: Z-2A 판정문 §3 (기전 5건 (a)~(e)) · 대장 `VERIFICATION_REGISTERS.md
  :358-368` («계측 결함 2건» 의 1번) — 수리 금지 계약 아래 기재분을 이 단이 인수.
- **결함 B 출처**: L6C-R 판정문 §7 F-1·부록 A-4 · 대장 `:426-436` («계측 결함 3건» 의
  F-1) — 「차기 판독기 수정 발주 시 반영 후보」의 그 단.
- **해법 계보**: 자매 판독기 개정 3~5 (`RUNG_L6C_REDISPATCH_2026-08-27.md` — RD-6
  반복별 앵커 등록값 표·NC-R13·PR-3) — A 는 이 해법의 이식이고, 반복별 등록값은 자매
  등록값과 bit-동일함을 저자가 재실측했다(§3-A).
- **폐합한 L6C-R 계약과의 관계**: 그 계약 §7 PR-0 ①-(i) 의 등록 명령 문면
  (`-o JobIDRaw,State,ExitCode`·클러스터 무지정)은 **역사 기록으로 존속**한다 — 이 단은
  그 단을 재판정하지 않으며, 수리된 판독기는 **차후 사용**에만 적용된다.
- **후속 회복**: 수리가 성립하면(IR-A) 후속 단이 「런처 수용 PASS 실물」·R-III/R-IV
  제출 가능성을 회복할 수 있다 — 그 발주는 이 단 밖(GPU 필요, user·운전석 몫).
- **잔여 대장 항목**(이 단이 인수하지 않음): F-2(σ 스코프)·F-3(proxy≢capture)·Z-2A
  대장 2번(fail-open 벡터)·«sacct FAILED 만 보면 오독» 잣대 함정(대장 :371) — 존속.

---

## §14. 저자 실행 명령 원장 (전건 2026-08-28)

| # | 명령 (요지) | 결과 |
|---|---|---|
| C1 | 두 판독기·두 판정문·대장 직독 (`sed -n`/`grep -n`) | §2 의 전 행 번호 확정; `git status` 양 파일 clean·커밋 `406712d`/`9b08dae` |
| C2 | 봉인 앵커 행 4곳 직독: census `stderr.log:1242/:226118` · R-II `:1244/:226122` (read-only `sed -n`) | 두 쌍 bit-동일·iter0≠iter1 (§2-A) |
| C3 | `python3 scripts/analyze_a210_z2a_probe.py --mode R-II --stderr <ZO>/stderr.log --census-stderr <ZO>/stderr.log --report <scratch>` | `FAIL ZI-B ['SPROBE_COUNTER_MISSING','ANCHOR_FIELD_MISMATCH'] f_bit=1.0` rc=4 · 4.5초 — 판별력 0 재현 |
| C4 | 같은 판독기, `--stderr <RII>/stderr.log --census-stderr <ZO>/stderr.log` | `FAIL ZI-B ['ANCHOR_FIELD_MISMATCH'] f_bit=1.0` rc=4 — 봉인 M14 동일 (G-A5 대조군) |
| C5 | syntax: `scontrol show config \| grep ClusterName` · `sacct -j 324557 -X -n -P -o JobIDRaw,JobName,State,ExitCode,Cluster` | `ClusterName=syntax` · `324557\|LUMINA_Z2A_II\|FAILED\|70:0\|syntax` |
| C6 | syntax: `sacct -M grammar -j 324557 ...` · `ssh grammar`: `scontrol ... ClusterName` · 로컬 sacct · `sacct -M syntax ...` | 교차 조회 양방향 빈 출력·rc=0 · `ClusterName=grammar` · `324557\|UPP_QE68_SCF\|COMPLETED\|0:0\|grammar` |
| C7 | grammar-debug (nested ssh): 현행 `check_pr0_unsealed(docs)` 호출 | `status=PASS sacct_records=['324557\|COMPLETED\|0:0']` — 결함 B 라이브 재현 (G-B3 수리 전 대조군) |
| C8 | `python3 -c` hex 변환 4값 · 선언 블록 앵커 문자열 grep 전건 · 계획 심볼 부재 grep (`A210_INSTRUMENT_REPAIR` 0파일 등) | §3-A 표·§11 블록 전건 확인 |
| C9 | grammar-debug: `check_prereg_preflight.py` 본 문서 | rc=0 (§12 기재) |

*(C3·C4 의 판독기 실행은 판독(비연산) — 로그인 노드 허용 범위; 판정자 선례와 동일 명령.
`<ZO>`/`<RII>` = Z-2A 판정문 §12 등록 경로.)*
