# 단 사전등록 — DET-SPRIM-L6: 생산자 원시 물질 장부 + 반복 1 population 의 LTE 이탈 판정 (2026-08-21)

저자 = Fable (분담 개정14: 사전등록=Fable). 발주서 앞겹 = **이 문서 원문 그대로**(재서술 금지).
갈림길 평가·후보 기각 사유 포함. HEAD `ba24977`, branch `thenmc-macroatom-fluorescence`.

## 0. 갈림길 판단 (겹 1 — 사전등록 저자 요약)

1. **user 가 지시한 "생산자 실제 S 직접 캡처"는 이미 집행·판정됐다**(DET-SPROD(IV), `34717ec`,
   판정 `docs/VERDICT_DET_SPROD_IV_2026-08-18.md`) — 참조된 메모리 인덱스가 3일 낡았다.
2. **이 갈림길이 옳은 질문인가**: 아니다 — 그 지시의 원 의도(NO_BRACKET 귀속)는
   `docs/FINDING_NOBRACKET_LTE_SEED_2026-08-19.md`(런 0회·오라클 무의존)로 이미 착지했고,
   표적은 "다음 캡처"가 아니라 **"STAGE-1(고정-T 레인)이 실제로 NLTE population 을 만드는가"**
   — 즉 DET-STAGE12 **L6** 으로 이동했다. L6 의 전제(DET-PHYSCMP 폐합)는 P-1 PASS +
   SH-A209-IDSEAL 전폐합으로 충족됐다.
3. **채택: `DET-SPRIM-L6`** — user 정정(원시량 η·τ_eff 장부)으로 잣대를 먼저 교정하고,
   그 잣대로 반복 1 의 LTE 이탈을 판정하는 **한 단**. 잣대 교정 없이 L6 를 재시도하면
   β 역산(증폭 7.6e4)과 χ_eff=0 특이점을 계승한다(잣대부터 감사).
4. 이 단은 **부수적으로** DET-STAGE12 L4 후반부("반복 1 도달")·IDSEAL B4(반복 간 신원 불변)·
   DET-SPROD 미결 2(S1(a) byte-불변 실행)를 회수한다.

## 1. 계약 (하나)

> **반복 1 의 생산자 선 물질이 원시량(η, τ_eff, provenance)으로 장부에 적히고,
> 그로부터 S_prod = η/χ_eff 가 역산·상쇄 없이 유도되어
> "고정-T 레인의 반복 1 population 이 LTE 를 벗어나는가"(DET-STAGE12 L6)가 판정된다.**

계측(원시 장부)과 계량(L6)은 한 계약이다 — 잣대 없이 판정은 성립하지 않고, 잣대는 이
판정런에서만 검증된다. 수리 아님·물리값 무접촉·기본 OFF.

## 2. 왜 이 단인가 — 경위와 기각한 후보

### 2-1. 경위

- user 지시 "생산자 실제 S 직접 캡처"는 **이미 집행됐다**: DET-SPROD(IV), 커밋 `34717ec`,
  판정 `docs/VERDICT_DET_SPROD_IV_2026-08-18.md`. 참조된 인덱스가 3일 낡았다.
- 그 지시의 원 의도(NO_BRACKET 의 물리 귀속)는 `docs/FINDING_NOBRACKET_LTE_SEED_2026-08-19.md`
  (런 0회·오라클 무의존)로 착지: NO_BRACKET 은 LTE 시드의 대수적 필연. 표적은
  "LTE 시드에 RE 근을 요구하는 것이 옳은가"로 이동 → user "두 개 다" → DET-STAGE12 개설.
- DET-STAGE12 L4 부분 PASS(사상 최초 `te_generation=1->2` 커밋,
  `docs/VERDICT_DET_STAGE12_L4_2026-08-20.md`).
  L6("반복 1 population 의 LTE 이탈")은 **미판정** — 커밋 직후 런이 죽었다. 그 죽음의 기전은
  DET-PHYSCMP P-1 이 특정(`physics_comparison.c:133`, `grid_manifest_sha256` 미기재)했고
  SH-A209-IDSEAL 이 수리·**전폐합**했다(`docs/VERDICT_A209_IDSEAL_2026-08-20.md`,
  `[PHYSICS-COMPARISON] lane=DET iter=0 status=COMMITTED`, model rc=0). **⟹ L6 의 전제 충족.**
- ★**발주 정정 (2026-08-21, user)**: 판정문 §3-2 의 후보명 `DET-BETAPROD`(β 적재)를 운전석
  발주서가 **검증 없이** 전달했고, user 가 "β 는 독립값이 아니다, S 가 근본"이라 지적했다.
  운전석 실측 + **본 사전등록 저자의 독립 검증**(§3-2)으로 확인: β 는 `exponx(τ_eff)` 의
  순수 함수다. 표적을 **원시량 장부**로 정정한다. 이 경위는 분장 장부 대상이다.

### 2-2. 기각한 후보

| 후보 | 기각 사유 |
|---|---|
| **DET-BETAPROD** (β 장부) | β=`exponx(τ_eff)` 순수 파생값(`src/line_net_rate.c:137-167` 실측). 파생 스칼라 계측은 증상(역산 잔차)을 겨눈다. 그 목적(S2b 역산 논쟁 종결)은 원시량 캡처의 **따름정리** — 재구성 항등 G4b 가 처음으로 "구성상 참"(DET-SPROD §3-1 한계)을 넘는다 |
| **S 자체 캡처** | S=η/χ_eff 는 χ_eff=0(inversion 경계)에서 발산. `exponx` 가 τ=0 유한극한을 뺄셈 전에 뽑는 설계(`line_net_rate.c:146-152` 주석)를 되돌리는 잣대다. 원시량이 아래층이다 |
| **A210-ZERO-OPACITY Z-1 재착지** | 새 단 아님 — 기존 사전등록(`docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md`)의 집행 잔무. 런 사망 원인 실측: `slurm-319967/8.err` = `DET_FLIGHT_FATAL A2-10 targeted gate requires A100 hardware: NVIDIA H200 NVL`(h200 배정, 템플릿 :154 가드) ⟹ a100 재제출 사안. 자유레인 4런 배터리는 전선 이동(고정레인)으로 **이 단 착지 후 재평가** — 이 단이 고정레인 census 를 부수 산출한다(§6 분기 D2) |
| **L2 (자유레인 byte 불변)** | 위생 게이트. 열린 물리 질문 0. 후순위 유지 |

### 2-3. "이 갈림길이 옳은 질문인가"

"다음 캡처 단이 무엇인가"는 옳은 질문이 아니다. 캠페인의 주된 경로(배선도/물리 검사)에서
열린 물리 질문은 하나다: **STAGE-1 이 실제로 NLTE population 을 만드는가.** 만들면
STAGE-2(자유-T) 재시도의 근거가 생기고, 못 만들면 그것이 다음 수리 표적이다. user 정정은
이 판정의 **잣대**를 정직하게 만든다 — 둘은 한 단이다.

## 3. 기전 오프라인 특정 (전부 이 세션 실측; 추측은 [추정] 표기)

### 3-1. L6 측정이 지금까지 불가능했던 이유 — 두 개의 스위치

1. **phase 게이트**: Stage-4 행 조립은 `diag->active = deterministic && phase=="REQUESTED_TE"`
   (`src/lumina_plasma.c:13971`)이고, phase 는 `a210_uniform_endpoint_phase`
   (`:13647-13663`)가 **시행 T 벡터 == `LUMINA_RADEQ_DIAG_TE_K`** 일 때만 `REQUESTED_TE` 를 준다
   (`a210_requested_diagnostic_te`, `src/radeq_publication.c:21-30`). 고정 레인의 시행 T =
   핀 프로파일(전셸 10020, `/gpfs/kjhan/lumina/te_profiles/seed_uniform_10020.txt`, 값 "10020"
   → strtod 정확 10020.0). 그런데 기존 세 런(l4/p1/idseal)은 전부
   `LUMINA_RADEQ_DIAG_TE_K=19059.411196903675`(IDSEAL RUN_FOOTER 실측) ⟹ phase=NULL ⟹ 행 0.
   **`LUMINA_RADEQ_DIAG_TE_K=10020` 으로 맞추면 소스 수정 없이 행이 발화한다**
   (10020 은 LOWER/UPPER/GEOMETRIC_MID 어느 것도 아님 — 기하 중앙 = √(3500·140000)=22135.9).
2. **반복 수**: DET 레인 반복 수 = `LUMINA_PURE_CMFGEN_ITER`(`src/lumina_cuda.cu:8050-8052`
   → `cmfgen_run(..., pc_iter)` `:8080-8085`). 루프 구조(`src/lumina_cmfgen.c:7382-7530`):
   pass0=init(시드 물질 예측자) → pass1=iter0(R6 생산 → R7/A2-10 → 커밋 gen 1→2 → snapshot
   iter0000) → **pass2=iter1(R6 가 커밋된 gen-2 물질로 재수송 → A2-10 → 커밋 gen 2→3)**.
   기존 런은 전부 `=1` ⟹ iter1 부재. 런처 템플릿(`scripts/run_det_convergence_2026-08-08.slurm`,
   IDSEAL input/job.slurm 과 **byte 동일** 실측)의 모드 게이트가 FLIGHT≥4 / TARGETED==1 /
   CENSUS==1 뿐 ⟹ **outer=2 를 허용하는 신규 모드가 필요하다**(런처=집행 인프라, src 아님).

### 3-2. 발주 정정의 독립 검증 — 원시량이 밑바닥이다 (전부 실측)

- 생산식(`src/line_net_rate.c`, `line_net_sobolev_radiation`): `continuum_term = β·J_cont`,
  `local_emission_term = η · (c·t/ν) · companion`, `jbar = continuum + local`,
  `companion=(1−β)/τ` (`exponx`). **S 는 형성되지 않는다.** 대입하면
  `local = (η/χ_eff)·(1−β) = S·(1−β)` — 항등이되, τ→0 에서 companion→1/2 로 유한
  (S→∞ 와 (1−β)→0 이 곱에서 상쇄). ⟹ **곱이 물리량이고 S 단독은 특이점을 가진다.**
- 물질(`line_net_sobolev_material`): `η = n_u·A_ul·hν/4π`, `χ_raw = τ·ν/(c·t)`,
  `τ_eff=τ`; srce_chk 갈래(τ<−0.5)도 `τ_eff = χ_eff·c·t/ν` 로 **같은 관계** 유지.
  ⟹ 독립 원시쌍은 **(η, τ_eff)** + provenance(srce_chk, exact_zero). χ_eff 는
  `τ_eff·ν/(c·t)` 로 정확 유도되므로 별도 적재는 정보 0 (본 문서가 그 대수를 사전등록).
- 상수: `LINE_NET_H_PLANCK=6.62607015e-27`, `LINE_NET_C_LIGHT=2.99792458e10`
  (`src/line_net_rate.c:7-8`), `LINE_NET_FOUR_PI`(`src/line_net_rate.h:14`).
- 기존 캡처(`src/lumina_cmfgen.c:6379-6385`)는 `radiation.continuum_term`·`local_emission_term`
  만 적는다(할당 `:6288-6301`, 플래그 `:6607`, 구조체 `src/lumina.h:295-298`). 캡처 지점의
  스코프에 `material.emission_per_sr`·`material.effective_tau`·`material.srce_chk_applied`·
  `material.exact_zero_provenance` 가 **이미 있다** — 확장은 같은 자리·같은 게이트다.
- DET-SPROD §3-3 의 방어 불가 사유였던 **증폭 7.6e4** 는 전적으로 β→1 에서
  `S = local/(1−β)` 역산 탓 — 원시량이면 **허용오차가 아니라 구성으로** 소멸한다.

### 3-3. 반복 1 이 측정하는 것

iter0 의 A2-10 bundle(`a210_production_bundle_ledger`, 고정 레인 분기
`src/lumina_plasma.c:15928-15966`)이 만든 NLTE 후보가 gen-2 로 커밋되고
(`nlte_population_candidate_commit_bundle` `:16004`), **iter1 의 R6 생산자가 그 물질을 읽는다**
(`sobolev_upper_population_cache`, `src/lumina_cmfgen.c:6349`). ⟹ iter1 행의
η^prod/χ_eff^prod = **커밋된 반복 1 물질의 선 소스**. LTE 시드에서는 S=B(FINDING 사슬 2:
1282/1282 가 1±1%, median 0.999993)였으므로, **S_prod/B(10020 K) 의 1 이탈 = L6 판정량**이다.
행은 shell 0 한정(`a210_line_saturation_add` 입구 `shell!=0` 반환) — L6 판정도 shell 0 한정.
행↔반복 귀속은 마커 브래킷으로 결정적:
행 블록은 해당 반복의 `[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=k` 줄보다
앞에 나온다(A2-10 solve 내부에서 인쇄). 반복당 bundle 1회 ⟹ SUMMARY 1개(`:15327`).

### 3-4. 알려진 위험 (수리하지 않고 분기로 등록)

`INDEPENDENT_SPROBE_UNDEFINED`(χ_eff==0·η>0 행)가 여전히 A2-10 트랜잭션 전체를 죽인다
(`src/lumina_plasma.c:14089-14093` 실측 — Z-1 계측은 WITNESS 인쇄+census 만 추가, 처분 불변).
iter0(IV 실측 후보 211,887행)에서는 미발화였으나 **iter1 의 NLTE population 에서는 미지**다.
이 결함의 수리는 A210-ZERO-OPACITY 의 계약이며 **이 단은 손대지 않는다** — 발화 시 분기 D2.

## 4. 기대 변경집합 (이 목록 밖 변경 = 실패) + V5 권한

> **개정 1 (2026-08-21, 저자=Fable)**: `src/lumina_atomic.c` 행 누락을 Codex 발주-즉시 신고로
> 적발, 추가했다(경위는 §13). 발주 1차가 `tests/` 행을 오독으로 탈락시킨 건은
> 운전석 위반으로 분장 장부에 기재된다 — 아래 표의 **5행 전부가 변경집합의 정식 구성원**이다.

### src/ — 계측 4파일 + tests/ 1파일 (물리값 경로 무접촉, 기본 OFF)

| 파일 | 변경 |
|---|---|
| `src/lumina.h` | OpacityState 에 `line_producer_eta`(double\*)·`line_producer_tau_eff`(double\*)·`line_producer_provenance`(uint8\*: bit0=srce_chk, bit1=exact_zero) 추가. 기존 두 항 배열·stride 필드는 불변 유지 |
| `src/lumina_cmfgen.c` | 같은 `sproducer_capture` 게이트 안에서 세 의무: ① 할당·센티널 초기화(`:6288` 블록) ② 캡처 대입(`:6379-6385` 블록) ③ **호출당 리셋**(`:4947-4952` 계열 — 기존 2배열과 동일하게 신규 3배열도 **free 후 NULL 대입**; NULL 대입 누락은 ④와 이중해제를 만든다 — 아래 게이트 불추가 판단 참조). 센티널: η·τ_eff = −1.0 (정당 범위 η≥0, τ_eff≥−0.5 ⟹ 충돌 없음 — srce_chk 가 τ<−0.5 를 항상 대체함을 §3-2 실측), provenance = 0xFF |
| `src/lumina_plasma.c` | `a210_line_saturation_add` 시그니처·행 구조체·ROW 인쇄 suffix 에 `producer_eta=`·`producer_tau_eff=`·`producer_srce_chk=`·`producer_exact_zero=`·`producer_raw_defined=` 추가(버퍼 확장 포함). 미정의/stride 불일치 시 기존 패턴 그대로 `UNAVAILABLE` fail-closed |
| ★`src/lumina_atomic.c` (**개정 1 추가**) | ④ **최종 teardown**: `free_opacity_state`(`:1225`)에 신규 3배열 `free` 추가 — 기존 `line_producer_continuum_term`·`local_emission_term` 해제 쌍(`:1233-1234` 계열) 바로 곁. 해제 경로는 저장소 전체에서 **이 둘뿐**(③ 리셋·④ teardown — `line_producer_` 접촉 파일 전수 grep 실측: 본 표의 src 4파일이 전부, `.cu` 0건). ③이 NULL 을 대입하므로 ④와의 관계는 이중해제가 아니라 순수 누수였다 — 이 행이 그 누수를 막는다 |
| ★`tests/` (해당 selftest 1파일) — **정식 구성원. 발주서가 이 행을 떨어뜨리면 그것이 곧 "사전등록 범위 좁힘" 위반이다** | 신규 필드 회귀 + 음성대조 **NC-C1~C3** (§5 P3 행이 인용하는 그 셋 — 이 행이 빠지면 P3 의 음성대조가 통째로 사라진다) |

`src/env_universe.h` **불변** — 신규 env 0. `LUMINA_A210_SPRODUCER_CAPTURE=1` 이 5배열 전부를
게이트하고 `=2` 거부(IV S1(b) 실측)는 유지된다.

**V5 권한 요청**: 이 단은 위 계측 확장을 위해 src 접촉 권한을 요청한다. 근거:
① user 발주 정정이 원시량 장부를 표적으로 명시 ② 잣대 교정 없는 L6 는 β 역산(증폭 7.6e4)과
χ_eff=0 특이점을 계승 — 잣대부터 감사 원칙 위반 ③ 물리값 무접촉·기본 OFF·byte-불변 게이트
P2 로 범위 봉쇄 ④ 전례 = DET-SPROD 캡처 단(같은 자리·같은 패턴·감리 통과).
**K-final 권한은 요청하지 않는다.**

### scripts/ — 런처·판독기

| 파일 | 변경 |
|---|---|
| `scripts/run_det_convergence_2026-08-08.slurm` | 신규 diagnostic_mode **`A210_L6_PROBE`**: outer_iterations **정확히 2** 요구, `LUMINA_CMF_FINE_MGPU_DEVICES==2`·A100 하드웨어 요구(TARGETED 와 동일), 사후게이트 = fatal_scan + `committed_count==2` + targeted 로그 게이트(`--expected-outer-iterations 2` — 개정 2 로 확장된 checker, 아래 행) + snapshot checker `--expected-iterations 2 --tail-transitions 0`(**수렴 주장 없음** — 스냅샷 실재·유효만) + `A210_L6_PROBE_ACCEPT` 토큰 |
| `scripts/check_a210_targeted_gate.py` (**개정 2 추가**) | `--expected-outer-iterations N`(기본 1 — 기본 경로 거동 불변, TARGETED 모드 호출부 무변경) 매개변수화. **감시 축소 0**: repair-토큰 스캔(floor/cap/clamp/jitter/repair)은 **로그 전체**(반복 1 포함)에서 유지. 정확-건수 구조 검사를 N 으로 일반화 — `[R7][PHASE] … COMMITTED` **N건**(k번째가 `iter=k`·`te_generation=k+1->k+2`), `[PHYSICS-COMPARISON] lane=DET` **N건**(k번째 `iter=k`), `[cmf_fine][EXACT-MULTIGPU-EPOCH]` **N+1건**(패스 수=N+1 — `src/lumina_cmfgen.c:7382` 루프 실측; N=1 에서 기존 기대 2 와 합치, 최종 확정은 P5), report 에 `expected_outer_iterations` 실값 기재. 반복 ≥1 구조 검사는 **신규 감시(추가)**다 |
| `scripts/stage_det_stage12_l6_probe.sh` (신설) | IDSEAL 스테이징 클론 + §7 env delta **만** 적용 + 스테이징 신선성 봉인(G1): job.slurm·checker·stager 를 repo HEAD 에서 신선 복사하고 sha256 을 RUN_FOOTER 에 기재, git provenance 신선 캡처, `python3 --version` 기재 (IDSEAL §3(b)·§6 최소 요건 — §L 수리 단이 미착지이므로 이 단이 그 최소치를 집행) |
| `scripts/analyze_det_stage12_l6.py` (신설) | 판독기: iter 마커 브래킷 → 행 파싱 → 재구성 항등(G4b) → S_prod/B 분포·분기(G5) → census(INVERSION_BOUNDARY·NEGATIVE_CHI·UNAVAILABLE) → 기계 verdict JSON. `--selftest` 에 NC-A1~A5 내장. B_ν·재구성 상수는 §3-2 의 소스 정의값 전사, exponx 는 `line_net_rate.c:137-167` 의 독립 파이썬 전사 |

### 기타

- `docs/RUNG_DET_SPRIM_L6_2026-08-21.md`(본 문서)·`validation/det_stage12/`(산출).
- **선행 정리(이 단의 변경집합 아님)**: 작업트리의 미커밋 드리프트
  `scripts/stage_a210_line_saturation_diagnostic.sh`(+`LUMINA_FIXED_TE_PROFILE` 관통 1줄 —
  DET-STAGE12 스테이징의 지연 착지)를 **스테이징 전에** 별도 커밋 또는 원복(변조 태스크 0 확인).
  ※운전석 집행 완료: `7057f31`.
- 커밋 규율: 사전등록 커밋(본 문서) → 계측·스크립트 커밋 **1개** → 판정문 커밋.
  검수·판정은 **커밋 접촉 파일과 이 절의 표가 1:1 로 일치**함을 확인한다(IDSEAL P7 방식) —
  ④(atomic 해제)와 ③의 NULL 대입이 diff 에 실존하는지가 그 대조의 명시 항목이다.

### 게이트 불추가 판단 (개정 1 — 이 결손이 드러낸 결함 계급에 대해)

teardown 누수·이중해제를 겨눈 **런타임 게이트는 추가하지 않는다.** 근거:

1. **누수는 어떤 등록 측정도 오염하지 못한다** — 판정런은 일회성 프로세스이고 모든 산출물
   (stderr·snapshot·RUN_FOOTER)은 teardown **전에** 기록된다. 계약(§1) 위해 = 0. P1~P5 가
   못 잡는 것은 사실이나, 잡을 가치가 있는 자리는 게이트가 아니라 **검수의 변경집합 1:1
   대조**다 — 위 "기타" 절이 그 확인을 명시 항목으로 못박았다.
2. **이중해제는 침묵 결함이 아니다** — ③이 NULL 대입을 빠뜨리면 ④에서 크래시(가시적 FATAL)로
   나타나며, 산출물은 이미 기록된 뒤다. 침묵 부패 경로가 없다.
3. **자격 있는 음성대조를 시연할 수 없다** — 이중해제 abort 는 C 표준이 보장하지 않는 glibc
   거동이고(주입 결함이 FAIL 을 **신뢰성 있게** 시연 못 하면 이 프로젝트 규약상 게이트 자격
   미달), leak-sanitizer 레인 신설은 빌드 기계 추가 = 이 개정이 금지하는 범위 확장이다.

## 5. 게이트 표 (각 행: 요구 / 증거 / ★음성대조)

### 오프라인 (런 전 — offline-first ②)

| # | 요구 | 증거 | ★음성대조 |
|---|---|---|---|
| **P1** | 빌드 CPU+GPU 두 타깃 에러 0·format 경고 0 | 로그 파일 보존(IDSEAL P1 형식) | (빌드 게이트 자체가 판별기) |
| **P2** | ★노브 미설정 시 **byte-불변** — 패치 전/후 바이너리로 A2-10 selftest 출력 Tier1 대조 (**DET-SPROD 미결 2 = S1(a) 를 이번에 실행**) | `scripts/byte_parity_compare.py` 보고서 보존 | 캡처 노브 **ON** 으로 같은 대조 → 신규 필드로 **차이 검출**(감도 시연) |
| **P3** | C selftest: 신규 필드 회귀 | 배터리 로그 | **NC-C1** stride 불일치 주입 → 행 `UNAVAILABLE` fail-closed / **NC-C2** `SPRODUCER_CAPTURE=2` → 거부 유지 / **NC-C3** provenance 비트 오염 주입 → 이름 있는 FAIL. 각각 주입 시 FAIL·제거 시 PASS 를 로그로 시연 |
| **P4** | 판독기 검증 | `--selftest` 로그 | **NC-A1** 위조 iter1 블록(S=B) → 분기 B 산출 / **NC-A2** τ_eff 1e-9 섭동 픽스처 → G4b FAIL / **NC-A3** iter1 R7 마커 삭제 → 귀속 BLOCKED(침묵 금지) / **NC-A4** χ_eff==0·η>0 행 → INVERSION_BOUNDARY census 로 분류되고 판정은 계속(**"정당한 0"≠"무효" — 4번째 반복 방지**) / **NC-A5** IDSEAL 봉인 stderr(행 0) → `NO_ROWS` fail-closed |
| **P5** | 런처 신규 모드·**확장 checker** 의 2-반복 거동 확정 | 봉인 로그에서 합성한 2-반복 픽스처(2번째 블록의 `iter`·세대 표지를 명시 규칙으로 재라벨 — 규칙은 픽스처 생성 스크립트에 기재; 단순 병치는 k번째=iter=k 검사에 걸려 불가)에 확장 checker(`--expected-outer-iterations 2`)·snapshot checker 실행 PASS + ★회귀 앵커: 봉인 IDSEAL stderr 에 **기본 인자** → 기존 checker 와 동일 verdict(기본 경로 불변 실증) | repair 토큰 오염 픽스처 → FAIL 시연 / ★매개변수 결합 시연: 봉인 IDSEAL stderr(1반복)에 `--expected-outer-iterations 2` → 건수 불일치 FAIL |

### 판정런 (1회)

| # | 요구 (기계 판정식) | 증거 | ★음성대조 |
|---|---|---|---|
| **G1** | 스테이징 신선성: RUN_FOOTER 의 job.slurm/checker/stager sha == repo HEAD blob sha; `input/git_head.txt` == 발주 HEAD; python3 버전 기재 | RUN_FOOTER·input/ | 검증 스크립트를 **IDSEAL 런 루트**에 적용 → FAIL (그 git_head 는 `dd9f7c18` 화석 — IDSEAL §6 실측) |
| **G2** | `[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1 … te_generation=2->3` 정확 1건 + `[PHYSICS-COMPARISON] lane=DET iter=1 status=COMMITTED` 1건 + `physics_DET_iter0001.{manifest.json,shell.csv,spectral.csv}` 실재 (★DET-STAGE12 L4 후반부 최초 충족) | stderr grep + 파일 실재 | 미도달 시 분기 D1/D2 (§6) — PASS 로 위장 불가(마커 부재는 판독기 BLOCKED, NC-A3) |
| **G3** | iter0·iter1 각 ROW ≥1, `producer_terms_defined=1`·`producer_raw_defined=1`·`independent_fields_defined=1` 인 행 100%, UNAVAILABLE=0 (위반 행 존재 시 미결 기재) | 판독기 census | IDSEAL stderr(DIAG_TE_K 불일치 → 행 0) → `NO_ROWS` (NC-A5) |
| **G4** | ★잣대 앵커(iter0): χ_eff≠0 행의 S_prod/B(10020 K) 중앙 ∈ [0.999, 1.001] — FINDING 사슬 2 의 고정레인 재확인. **G5 는 G4 통과를 전제**(잣대 먼저) | 판독기 | 앵커 실패 = 잣대/상수 결함 ⟹ L6 판정 금지·미결 기재 |
| **G4b** | ★재구성 항등(전 행): β_rec=exponx_py(τ_eff^prod), local_rec=η^prod·(c·t/ν)·companion_rec, cont_rec=β_rec·J_cont 에 대해 캡처 두 항·행 Jbar 와 상대편차 ≤1e-12 (Jbar=0 행은 census) — **뺄셈·역산 0 회**, "구성상 참" 한계(§3-1)를 처음 넘는 검증 | 판독기 | NC-A2 |
| **G5** | **L6 판정(계약 본체)**: iter1 의 χ_eff≠0 행 S_prod/B 분포 → §6 분기 A/A′/B/C 중 하나로 기계 판정 | 판독기 verdict JSON | 분기 정의가 상호 배타·전수(C 가 잔여 전부) + NC-A1 |
| **G6** | 부수(IDSEAL B4 회수): iter0000 vs iter0001 manifest 의 `grid_manifest_sha256` 동일 ∧ `te_manifest_sha256` 동일 | manifest 판독 | manifest 사본 1바이트 조작 → 검출 시연 |
| **G7** | 상류 무섭동: `[A2-10][LINE-COEFFICIENT-IDENTITY] phase=INTERIOR` 50줄이 IDSEAL 런과 byte 동일 (L4 §3-1 크로스빌드 방식) | diff | (결정론 자체가 판별기 — 불일치 = 즉시 FAIL) |

## 6. 기대치 사전등록 (빗나가면 그것이 정보다)

런 구성이 낼 수 있는 관측만 등재한다(IDSEAL 감리 R3 위생 — B4 사각 재발 금지).

| # | 기대 | 수치·범위 |
|---|---|---|
| **E1** | iter0 행 수 ≈1,282(union 선정 재현), UNAVAILABLE 0, S_prod/B 중앙 1±1e-4 (IV 0.999993 등급, stdev(log10) ~5e-7 등급) | G3·G4 |
| **E2** | 재구성 항등 전 행 ≤1e-12 | G4b |
| **E3** | iter1 도달(gen 2→3 커밋) | G2. 실패 시 D1/D2 |
| **E4** | iter1 분포 — **1순위 [추정]**: 분기 A (2준위: S=(1−ε)J̄+εB, J̄/B∈[0.71,1.0]@shell0 ⟹ 산란지배 선 S/B~0.7-1.0). A′·B 도 실질 확률 있는 대안 — 기대 드라마화 금지, 사전 확률 부여 안 함 | G5 |
| **E5** | 관측 등록(게이트 하중 0): iter1 의 τ_eff^prod vs 소비자 tau_effective(매칭 세대) — IV 의 2e5× 불일치·소비자 초열 S(1.9e5·B, DET-SPROD §3-5 미화해)가 매칭-T 에서 어떻게 되는지 원시량 수준에서 최초 판독 | 보고서 |
| **E6** | 자원: 경과 ≈ 2×1:56 + 여유(`--time=10:00:00`) [추정]; 호스트 메모리 +≈1.85GB (2,180,286선×50셸: double 2배열 872MB×2 + uint8 109MB — 기존 4배열 3.5GB 위에) [계산] | — |

**분기 (상호 배타·전수 — 어느 쪽이든 단은 계량으로 착지)**

정의: `f_super` := iter1 의 χ_eff≠0 행 중 S/B>10 인 행의 비율.
평가 순서(이중 안전): D2 → D1 → A′ → A → B → C, 첫 일치가 verdict. 단 아래 판정식은
순서 없이도 상호배타가 되도록 배제조건을 명문화했다(A↔A′ 는 f_super 10% 문턱으로,
A↔B·A′↔B 는 산술로 배타: B 는 0.01 초과·S/B>10 행을 ≤1%<10% 로 강제).
★분기는 라벨이지 필터가 아니다 — 어느 분기가 발화하든 verdict JSON 은 f_super 와
분포 전량(분위수·census)을 보고한다.

| 분기 | 판정식 (χ_eff≠0 행) | 함의 |
|---|---|---|
| **A** | **f_super<10%** ∧ ≥10% 행이 \|S/B−1\|>0.01 ∧ q50(S/B)∈[0.5,1.0) | STAGE-1 목적(NLTE population) 달성 실증 → 자유-T 재시도 근거. f_super>0 이면 그 행들은 census 로 기재(대장 후보)하되 인증은 유지된다 |
| **A′** | **f_super ≥ 10%** (다른 조건과 무관하게 A 에 우선) | 초열 이탈 — IV 소비자 초열(DET-SPROD §3-5, 1.9e5·B 미화해)이 매칭-T 에서 재현 ⟹ 후보 단: candidate solve 감사. ★우선 사유: 두 자릿수 초열 비율은 실물 다준위 형광일 수도 결함일 수도 있어 **귀속 전에는 A 의 인증(자유-T 재시도 근거)을 발행할 수 없다** — 분포의 나머지가 A 형태여도 그 관측은 verdict 에 병기만 한다 |
| **B** | ≥99% 행이 \|S/B−1\|≤1e-3 | population 공급이 복사장 미반영 ⟹ 후보 단: A2-07 배선 감사 |
| **C** | 그 외 | 미결 기재 + 분포 전량 보고 |
| **D1** | iter1 미도달, 이름 있는 차단 | 차단 사유·자리 = 발견(그 사유가 다음 단). 산출물은 보존된다(IDSEAL 전례: 사후게이트 사망 ≠ 증거 소멸) |
| **D2** | `INDEPENDENT_SPROBE_UNDEFINED` 차단 | WITNESS 행 = **고정레인 zero-opacity census** ⟹ A210-ZERO-OPACITY 수리의 우선순위 증거. 단은 부분 착지(계측 성립·L6 미판정), L6 재시도는 Z-O 수리 후 |

## 7. 판정런 구성 (발주서 뒷겹과의 대조 기준 — 좁힘 검출용)

기준 = IDSEAL RUN_FOOTER(봉인). **resolved env diff 가 아래 5건과 정확히 일치**해야 한다:

| env/구성 | IDSEAL | 이 단 | 사유 |
|---|---|---|---|
| `LUMINA_PURE_CMFGEN_ITER` (+`outer_iterations.txt`+argv[3]) | 1 | **2** | §3-1 반복 수 |
| `diagnostic_mode.txt` | A210_TARGETED_GATE | **A210_L6_PROBE** | 런처 모드 |
| `LUMINA_RADEQ_DIAG_TE_K` | 19059.411196903675 | **10020** | §3-1 phase 스위치 (핀과 strtod 정확 일치) |
| `LUMINA_A210_SPRODUCER_CAPTURE` | 0 | **1** | 5배열 캡처 |
| `LUMINA_A210_INDEPENDENT_CAPTURE` | 0 | **1** | J_cont(재구성 G4b 필요) |

그 외 전부 불변(`LUMINA_FIXED_TE_PROFILE`=seed_uniform_10020, `LINE_SATURATION_DIAG=2`,
`TARGET_ION=3`, sigma sha `90d04042…` 등). 덱·`/gpfs` 정본 불변.
제출: **slurm, partition `a100` 한정, `--gres=gpu:2`, `--mem` 명시**, job-name `LUMINA_` 접두,
OWNER.txt(`DO_NOT_CANCEL`). syn101 수동 제출 금지. grammar 용 `--exclude` 부착 금지(별개 클러스터).
소형 오프라인(P2~P5)은 grammar-debug(nested ssh) — 로그인 노드 연산 금지.

## 8. 이 단이 모르는 것 (추측으로 메우지 않는다)

1. **iter1 이 실제로 도달하는가** — physics_comparison 의 반복 ≥2 거동은 미시험(B4 사각),
   SPROBE 차단 위험(§3-4), 그 밖의 미지 가드. 분기 D1/D2 가 받는다.
2. **분기 A/A′/B 의 사전 확률** — 부여하지 않는다. E4 의 정량은 2준위 [추정]이다.
3. **iter1 행 수** — union 선정이 population 의존이라 iter0 과 다를 수 있다. 행 매칭은
   line id 교집합으로만.
4. **E5 의 정량 기대** — 생산/소비 두 코드 경로(`cmf_fine_line_material` vs
   `line_net_sobolev_material`)가 다르므로 없음. 관측만.
5. **targeted checker 의 2-반복 로그 거동** — P5 에서 오프라인 확정. 빗나가면 런처 계열
   변경으로 개정 기재 후 진행(런 발주 전).
6. **candidate solve(A2-07) 내부가 어떤 rates 로 도는지** — 이 단은 출력만 계량한다.
7. **a100 2-반복 wallclock·정확한 RAM 증분** — [추정]/[계산], 실측으로 대체된다.

## 9. 분장 장부 (집행 후 운전석이 "실제"·"위반" 을 채운다 — 규약상 담당만 적는 것 금지)

| 단계 | 규약상 담당 (개정14) | **실제** | 위반 |
|---|---|---|---|
| 갈림길 평가·사전등록(본 문서) | Fable | | |
| 발주(앞겹=본 문서 원문 첨부, 재서술 금지) | 운전석 | | |
| 코딩(§4 변경집합) | Codex | | |
| 코드 검수(고정질문에 *"발주서가 사전등록의 범위를 좁혔는가"* 포함) | Fable | | |
| 빌드·오프라인 게이트 P1~P5·스테이징·제출 | 운전석 | | |
| 판독기 실행·계측 패킷 | 운전석 | | |
| 판정(판정문 저작) | Fable (fresh) | | |
| 판정 감리 | Fable (★판정과 **다른** fresh 컨텍스트·고정질문 4) | | |
| 감리 반영·대장·커밋 | 운전석 | | |

## 10. 판정 절차

- 판정 = Fable **fresh 컨텍스트**(본 사전등록 + 봉인 산출물 + 운전석 패킷 제공; 판정 하중
  항목은 판정자가 직접 재실측 — IDSEAL 전례).
- 감리 = **또 다른 fresh Fable**(자기 채점 금지), 고정질문 4.
- 폐합 전 감리 필수. 판정문은 §6 분기 중 어느 것이 발화했는지 축자 기재하고,
  분기 밖 결과는 폐합 금지·미결 기재.
- 오라클 규율: 이 단의 어떤 게이트·기대치도 CMFGEN 런 수치를 인용하지 않는다
  (B_ν 는 물리 상수 함수, 앵커는 자체 봉인 로그).
- clamp/floor/cap 0. 판독기의 census 분류(INVERSION_BOUNDARY·NEGATIVE_CHI)는 물리값을
  바꾸지 않는 분석 분류이며 전량 보고된다 — 조용한 탈락 금지.

## 11. 이 단이 하지 않는 것

- `INDEPENDENT_SPROBE_UNDEFINED` 차단의 **수리**(A210-ZERO-OPACITY 계약 — 발화 시 D2 로
  증거만 넘긴다).
- 자유-T 레인 접촉·L2·핀 온도 선택·수렴 주장(snapshot checker 는 `--tail-transitions 0`).
- CMFGEN 정량 대조(잣대 미수렴 — 별도 단).
- S2b 구판(β 역산) 재판정 — G4b 가 원시량으로 대체한다.

---

## 12. 운전석 기록 — user 결정 (2026-08-21, 발주 직전)

§2-2 의 기각 후보 ②(A210-ZERO-OPACITY Z-1 재착지)는 **방향 결정**이라 user 에게 올렸다.
**user 결정: 미룬다.** 사전등록 권고대로 이 단 착지 후 재평가한다.

근거(사전등록 §2-2·§6 D2): Z-1 은 새 단이 아니라 기존 사전등록의 집행 잔무이고(h200 배정
사망 — a100 재제출 사안), **이 단이 고정레인 zero-opacity census 를 부수 산출**하므로
그 실측을 보고 Z-O 수리의 우선순위를 정하는 편이 낫다. 다만 그 차단 결함은 **미수리 상태로
이 단의 위험**이며 분기 D2 가 받는다 — 발화하면 그 WITNESS 행 자체가 Z-O 의 우선순위 증거다.

⚠**보류는 소멸이 아니다.** Z-1 은 `docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md` 의 미착지
집행분으로 살아 있다. 이 단 폐합 시 판정문이 재평가 결과를 기재한다.

---

## 13. 발주 1차 — 즉시 중단 (2026-08-21). ★운전석의 좁힘이 실측으로 드러났다

Codex 는 **한 줄도 쓰지 않고 중단**했다(작업트리 clean, `git status --short` 무출력).
발주서가 *"어긋남을 발견하면 구현을 멈추고 보고하라 — 그 어긋남 자체가 보고 대상이다"* 를
지시했고 그대로 집행했다. 신고 2건은 **둘 다 실물**이다.

### 13-1. ★운전석 잘못 — 발주서가 사전등록을 좁혔다

발주서 뒷겹이 *"구현 대상 = 사전등록 §4 의 `src/` 4파일 + `scripts/` 3파일"* 이라 적었다.
§4 소제목 *"src/ — 계측 4파일"* 을 **src 파일 4개**로 오독해, 표의 4번째 행인 **`tests/`
selftest 를 범위에서 떨어뜨렸다.** NC-C1~C3 이 거기 살므로 **음성대조가 통째로 빠지는**
좁힘이다 — "게이트는 주입 결함으로 FAIL 을 시연해야 PASS 자격" 규약을 발주가 무력화한 셈.

⟹ **개정14 가 코드 검수 고정질문으로 *"발주서가 사전등록의 범위를 좁혔는가"* 를 넣어 둔
바로 그 위반이다.** 검수까지 가기 전에 Codex 가 먼저 잡았다. 두 겹 발주(앞겹=원문 첨부)가
설계대로 작동해 **좁힘이 두 문서의 불일치로 보이게** 된 사례로 기재한다.

### 13-2. 사전등록 §4 의 결손 — `src/lumina_atomic.c` 부재 (운전석 실측으로 확인)

해제 경로가 **둘**이고 역할이 다르다:

| 경로 | 자리 | 역할 |
|---|---|---|
| `cmfgen_fine_jbar` | `src/lumina_cmfgen.c:4918` (해제 `:4947-4952`) | **호출당 리셋** — free 후 NULL 대입 |
| `free_opacity_state` | `src/lumina_atomic.c:1225` (해제 `:1233-1234`) | ★**최종 teardown** |

§4 표는 앞엣것만 적었다. 신규 3배열을 뒤엣것에 넣지 않으면 **최종 teardown 누수**다
(cmfgen 이 NULL 을 대입하므로 이중해제는 아니다 — 순수 누수). 소제목 *"계측 4파일"* 과
표의 실제 행 수(src 3 + tests 1)도 불일치한다.

⟹ 사전등록 저자(Fable)에게 **§4 최소 개정**을 발주했다. 계약(§1)·게이트 판정식·기대치(§6)·
§7 런 구성은 **불변** — 개정이 범위를 넓히면 그것이 다음 사고다.

### 13-3. 처분

- 13-1 은 **운전석 위반**으로 §9 분장 장부의 "위반" 열에 기재한다(집행 후 기입).
- 13-2 는 사전등록 개정 후 **재발주**. 재발주서 뒷겹은 파일 수를 세지 않고 **§4 표를 지목**한다
  — 숫자를 옮겨 적는 행위 자체가 이번 좁힘의 기전이었다.

---

## 14. 발주 2차 — 또 중단 (2026-08-21). ★개정이 행 하나를 **바꿔치웠고** 운전석이 못 잡았다

Codex 는 2차에서도 **한 줄도 쓰지 않고 중단**했다. 신고: *"§4 표에 `src/lumina_plasma.c`
가 없다. 그러나 생산자 배열을 A2-10 행으로 전달하는 코드(`:15132`)·행 구조(`:13877`)·
두 인쇄 경로(`:14298`·`:14467`)가 전부 그 파일에 있다. 신규 3배열을 `lumina.h`·
`lumina_cmfgen.c` 에 추가해도 **판독기에 도달하지 않는다.**"* — **사실이다.**

### 14-1. 기전 — 개정이 추가가 아니라 치환이었다

| | §4 표의 src 행 |
|---|---|
| 원판 `8046986` | `lumina.h` · `lumina_cmfgen.c` · **`lumina_plasma.c`** |
| 개정 1 `ef2ff32` | `lumina.h` · `lumina_cmfgen.c` · **`lumina_atomic.c`** |

행 수가 3으로 **불변**이다 — `atomic` 이 `plasma` **자리에 들어갔다.** 그런데 개정문 자신의
산문은 *"src 4 + tests 1"*, *"`line_producer_` 접촉 파일 전수 grep 실측: **본 표의 src
4파일**이 전부"* 라고 말한다. 표(3)와 산문(4)이 **문서 내부에서 모순**이며, 이를 정합시키는
읽기는 하나뿐이다: `plasma.c` 가 표에 있어야 한다.

### 14-2. ★운전석의 검수 실패 (이쪽이 더 무겁다)

개정을 절 단위로 통째 교체하면서 **새 텍스트를 읽는 한 경로로만** 확인했다. 원판 표와의
**행 단위 대조를 하지 않았다.** 커밋 메시지에 *"tests/ 행은 정식 구성원임을 표에 명기했다"*
라고 적었으니 tests 행은 봤고 — **사라진 행은 보지 않았다. 없는 것을 찾는 눈이 없었다.**

이것은 GR-7b 에서 내가 요구한 잣대의 위반이다. 그 단의 검수는 27행 사상을 **두 독립 경로**
(PREREG 명단 경유 / 판정문 표 직접)로 대조해 mismatch 0 을 보였다. 나는 같은 요구를 내
문서 교체에는 적용하지 않았다. **절 통째 교체는 결손을 숨기는 개정 형식이다.**

### 14-3. 처분

- **복원**: `src/lumina_plasma.c` 행을 `8046986` 원문 그대로 되살렸다(byte 동일 확인).
  저자의 산문이 요구하는 유일한 읽기이므로 **설계 결정이 아니라 되돌리기**다 — 새 문장을
  쓰지 않았다. §4 표는 이제 **src 4 + tests 1 = 5행**이고 개정문 산문과 정합한다.
- **규칙 신설**: 이후 이 문서의 **절 단위 개정은 행 단위 diff 로 검증한다**
  (`git diff <이전> <이후> -- <문서>` 의 `-`/`+` 행을 눈으로 대조). 통째 교체를 신뢰하지
  않는다.
- §9 분장 장부의 "위반" 열에 **운전석 2건**(§13-1 좁힘, §14-2 검수 실패)을 기재한다.

### 14-4. 기재 — 계약 결함 3건을 전부 Codex 가 잡았다

| 회차 | 결함 | 누구의 것 |
|---|---|---|
| 1차 | `tests/` 탈락 (음성대조 통째 소실) | 운전석 발주서 |
| 1차 | `lumina_atomic.c` 부재 (teardown 누수) | 사전등록 |
| 2차 | `lumina_plasma.c` 치환 소실 (판독기 미도달) | 개정 + **운전석 검수** |

세 번 다 **구현 착수 전**에 잡혔고 세 번 다 작업트리는 clean 이었다. 두 겹 발주(앞겹=사전등록
원문 첨부)가 설계대로 작동한 결과다 — 발주가 계약을 좁히거나 계약이 스스로 모순되면
**두 문서의 불일치로 드러난다**. 코더에게 "계약과 어긋나면 멈추고 보고하라"를 준 것이
이 단에서 세 번 값을 했다.

---

## 15. 발주 3차 — 또 중단 (2026-08-21). 이번엔 **게이트 설계** 결함 2건

Codex 는 3차에서 **코드를 쓰다가 되돌리고** 중단했다(작업트리 clean). 신고 2건 모두 실물이며
운전석이 기계로 검증했다.

### 15-1. 신고 ④ — G5 분기 A·A′ 가 상호 배타가 아니었다

§6 이 *"분기 (상호 배타·전수)"* 라 선언했으나 판정식이 그렇지 않았다. **운전석 재계산**:

```
분포 60% at S/B=0.8 · 30% at 1.0 · 10% at 11
  q50=0.8  이탈행 70%  S/B>10 행 10%
  A = (70%>=10%) and (0.5<=0.8<1.0) = True     A' = (10%>=10%) = True   ⟹ 동시 참
```

⟹ 판독기가 **유일한 verdict JSON 을 만들 수 없다**. G5 는 계약 본체(§1)의 판정이므로
이대로면 단이 착지할 수 없었다. 가상의 분포도 아니다 — DET-SPROD §3-5 의 소비자 초열
(1.9e5·B)이 미화해로 남아 있어 A 와 A′ 가 섞인 분포가 나올 실질 가능성이 있다.

**개정 2-1~2-3 으로 해소**: `f_super` 정의 신설 + A 에 `f_super<10%` 배제조건 + A′ 우선.
저자의 우선 근거: *두 자릿수 초열 비율은 실물 다준위 형광일 수도 결함일 수도 있어
(parity26 전례: J/B 22-59 형광 초열은 실물) **귀속이 인증에 선행**해야 한다* — A 가 발행할
인증("자유-T 재시도 근거")을 미귀속 관측이 오염하게 두지 않는다.

**운전석 기계 검증**: Codex 반례 → `(A,A′,B)=(False,True,False)`. 무작위 20만 분포에서
**두 분기 동시 참 0건**.

### 15-2. 신고 ⑤ — 2-반복 로그는 기존 targeted checker 를 통과할 수 없었다

§4 가 신규 모드의 사후게이트로 *"targeted 로그 게이트(**동일 인자**)"* 를 요구했으나
**성립 불가**였다. `scripts/check_a210_targeted_gate.py` 실측:

```python
:53   if len(found) != count: raise GateError(...)        # ★정확 N건 강제
:126  indexed_lines(..., "[cmf_fine][EXACT-MULTIGPU-EPOCH]", 2)
:321  indexed_lines(..., "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED", 1)  + "iter=0" 고정
:329  indexed_lines(..., "[PHYSICS-COMPARISON] lane=DET", 1)                  + "iter=0" 고정
:355  "expected_outer_iterations": 1
```
2-반복 런은 세 자리 전부 건수 불일치다. 게다가 G2 는 `iter=1` 커밋을 요구하므로 **계약이
스스로 "checker 가 거부할 로그"를 요구**하고 있었다.

**개정 2-4~2-6 으로 해소 (선택지 ⓐ)**: checker 를 `--expected-outer-iterations N`(기본 1)로
매개변수화해 변경집합에 1행 추가. 저자가 ⓑ(iter0-prefix 한정 검사)를 기각한 사유 —
*이 단의 표적인 **반복 1 을 repair-토큰 감시에서 제외**하는 감시 축소*.

★**감시 변화 공시 (축소 0)**: 제외되는 로그 줄 없음 — repair-토큰 스캔은 로그 전체 유지.
반복 ≥1 의 구조 검사(k번째 commit = `iter=k` · 세대 `k+1->k+2`)는 **신규 감시(추가)**다.

### 15-3. 개정 형식 — 절 통째 교체를 금지했다

§14 의 사고(개정 1 이 행 하나를 소리 없이 치환) 때문에 이번 개정은 **`OLD:`/`NEW:` 쌍 6개**
로만 받았다. 운전석이 기계로 적용하고 **행 단위 diff** 로 검증:

```
삭제된 행 4 = 개정 2-2·2-3·2-4·2-6 의 교체 대상 그 자체  ⟹ 의도 외 소실 0
§4 src 표: lumina.h · lumina_cmfgen.c · lumina_plasma.c · lumina_atomic.c · tests/  (5행 유지)
§4 scripts 표: slurm · check_a210_targeted_gate.py · stager · analyzer            (4행)
```

### 15-4. 누적 — 계약 결함 5건, 전부 구현 착수 전

| 회차 | 결함 | 주인 |
|---|---|---|
| 1차 | `tests/` 탈락 (음성대조 소실) | 운전석 발주서 |
| 1차 | `lumina_atomic.c` 부재 (teardown 누수) | 사전등록 |
| 2차 | `lumina_plasma.c` 치환 소실 (판독기 미도달) | 개정 1 + **운전석 검수** |
| 3차 | A·A′ 비배타 (verdict 불능) | 사전등록 |
| 3차 | targeted checker 2-반복 불가 (계약 자기모순) | 사전등록 |

결함이 **얕은 데서 깊은 데로** 이동하고 있다: 파일 범위 → 문서 정합 → 게이트 설계.
세 번 다 작업트리 clean 이었고 판정런은 한 번도 소모되지 않았다. offline-first 의
"런 발주 3요건"이 값을 하는 자리다 — 이 다섯 건이 런 뒤에 드러났다면 각각 판정런 1회씩이다.

---

## 16. 기계 프리플라이트 선언 (개정 3, 2026-08-21) — 계약이 스스로를 검사한다

`scripts/check_prereg_preflight.py` 가 이 블록을 읽어 발주 **전에** 계약을 검사한다.
선언이 없으면 **거부**(fail-closed) — 조용한 건너뛰기 금지.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### src/ — 계측 4파일 + tests/ 1파일",
    "table_end": "`src/env_universe.h` **불변**",
    "path_pattern": "src/[a-z_]+\\.[ch]|tests/",
    "symbol": "line_producer_",
    "roots": ["src"],
    "expected_extra": ["tests/"]
  },
  "branches": {
    "regimes": [[0.3, 0.99], [0.999, 1.001], [1.01, 3.0], [10.5, 60.0]],
    "metrics": {
      "f_super": "sum(1 for x in v if x>10)/len(v)",
      "dev": "sum(1 for x in v if abs(x-1)>0.01)/len(v)",
      "q50": "median(v)",
      "near": "sum(1 for x in v if abs(x-1)<=1e-3)/len(v)"
    },
    "rules": [
      {"name": "A'", "predicate": "f_super >= 0.10"},
      {"name": "A",  "predicate": "f_super < 0.10 and dev >= 0.10 and 0.5 <= q50 < 1.0"},
      {"name": "B",  "predicate": "near >= 0.99"}
    ],
    "adversarial_fixtures": [
      {"name": "codex-counterexample-2026-08-21", "mix": [[0.8, 60], [1.0, 30], [11.0, 10]]}
    ],
    "residual": "C"
  },
  "references": [
    {"path": "scripts/check_a210_targeted_gate.py",
     "flags_existing": ["--expected-devices", "--expected-refinements"],
     "flags_planned": ["--expected-outer-iterations"]},
    {"path": "scripts/run_det_convergence_2026-08-08.slurm", "flags_existing": []},
    {"path": "scripts/byte_parity_compare.py", "flags_existing": []},
    {"path": "src/line_net_rate.c", "flags_existing": ["line_net_cmfgen_exponx"]}
  ]
}
```

**이 선언이 검사하는 것과 각각이 잡았을 실제 결함**:

| 검사 | 판정식 | 잡았을 결함 |
|---|---|---|
| **PF-1** 변경집합 ↔ 심볼 grep | 양방향 차집합 = 공집합 | ②`lumina_atomic.c` 부재 · ③`lumina_plasma.c` 치환 소실 |
| **PF-2** 분기 분할 | 혼합-가중치 스윕 + 회귀 픽스처에서 동시참 0·공백 0 | ④A·A′ 비배타 |
| **PF-3** 참조 실존 | 경로 실존 · `flags_existing` 수용 · `flags_planned` **미수용** | (⑤의 절반 — 계획 플래그가 이미 있으면 잡는다) |

★**PF-3 의 정직한 한계**: ⑤(2-반복 로그가 checker 를 못 통과)는 **잡지 못한다.** 그것은
정적 참조 검사가 아니라 **의미 검사**(정확-건수 강제의 함의)이기 때문이다. 이 게이트는
5건 중 **3건**을 발주 전에 잡는다 — 전부가 아니다. 과대 주장하지 않는다.
