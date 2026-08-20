# 판정 — SH-A209-IDSEAL: A2-09 방출률 발행체의 신원 봉인

날짜 2026-08-20 · 사전등록 `docs/RUNG_A209_IDENTITY_SEAL_2026-08-20.md`(커밋 `7917cd3`, 12:37) ·
수리 커밋 `2dc2817`(13:46) · 판정런 slurm **321104**(syn101 · A100×2 · 제출 16:05 · 실행 19:13–21:09,
경과 1:55:59 [실측 sacct]) · 판정자 **Fable**(분담 개정14 — 이 판정문은 Fable 이 썼다).

증거: 운전석 계측 패킷(68줄) · 게이트 보존 `/gpfs/kjhan/lumina/gates/sh_a209_idseal_20260820T044201Z/` ·
런 루트 `/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/idseal_20260820T044703Z_a209/`.
★판정 하중이 걸리는 항목(B1·B2·B3·게이트 오탐·생략 게이트)은 패킷을 믿지 않고 **판정자가 전부
직접 재실측**했다 — 운전석이 이 건에서 두 번 오독한 전력(순간 스냅샷 "GPU 0%" → "CPU 병목",
둘 다 정정됨, `CLASSIC_DEBT_CENSUS.md` 8절+정정 절)이 그 이유다.

---

## ★분장 장부 (명목이 아니라 실제 집행자)

| 단계 | 규약상 담당 (개정14) | **실제** | 위반 |
|---|---|---|---|
| 사전등록 (`7917cd3`) | Fable | Fable | ✅ |
| 발주 (worktree `wt_a209`, PREREG 원문 첨부) | 운전석 | 운전석 | ✅ |
| 코딩 (§3 변경집합 10수정+1신설) | Codex | Codex | ✅ (⚠집행 기록 초판을 Codex 가 선기입 — 검수 R2 가 적발, 운전석 재실측으로 교체됨. 사전등록 문서에 경위 기재) |
| 코드 검수 | Fable | Fable (판정=인정, R1–R5) | ✅ |
| 빌드·오프라인 게이트·제출 | 운전석 | 운전석 | ⚠규약 배정은 준수. 집행 결함 2건: (1) **낡은 검사기 스테이징** → 판정런 FAILED 종료(오탐, §3)·(2) **낡은 provenance 캡처 스테이징**(§6 신규 발견) |
| 판정 (이 문서) | Fable | **Fable** | ✅ (직전 단의 위반이 재발하지 않음) |
| 판정 감리 | Fable (fresh 컨텍스트) | **미실시** — 이 문서 이후 | — |
| 감리 반영·대장·커밋 | 운전석 | 대기 | — |

---

## 1. 판정: **PASS — 전폐합** (계약 범위 내; §4 에 범위 명시, §5 에 미결 명시)

사전등록 §6 기대치 대조. 각 행의 "자문"(다른 가설도 같은 값을 내는가)을 그대로 적용했다.

| # | 사전등록한 기대 | 실측 | 증거력 판정 |
|---|---|---|---|
| **B1** | `site=133` 0줄, iter=0 에 `[PHYSICS-COMPARISON][FATAL]` 없음 | [실측 판정자 grep] `site=133` **0건**·`FATAL` **0건**·stderr:253 `[PHYSICS-COMPARISON] lane=DET iter=0 status=COMMITTED` | ✅ 적중. 단 사전등록대로 **증거력 낮음** — hex64 상수 채움 가짜 수리도 적중한다. B2 와 결합해서만 의미 |
| **B2** | manifest 의 `grid_manifest_sha256` == 그 런의 spectral CSV edge 에서의 **독립 재계산** | [실측 판정자, 3중 — §1-1] `49a7fa33…` **일치** | ✅ **핵심 증거 성립.** 상수 채움·타 해시 복사·직렬화 오류 전부 배제됨 |
| **B3** | `[A2-09][IDENTITY]` 존재, 세 해시 hex64, 그 atomic == manifest(온도측) atomic | [실측 판정자] stderr:18/32/39 — gen 1·2·3 발행, 세 해시 전부 hex64, atomic `b779a811…` == manifest `b779a811…` **문자 단위 일치** | ✅ 적중, **중간 증거력**. 자문의 잔여 가설(이전 런 로그에서 베낀 상수)은 §1-2 의 diff 보강으로 배제 |
| **B4** | 반복 간 `grid_manifest_sha256` 불변 | 판정런이 `outer_iterations=1`(RUN_FOOTER [실측]) — 꼬리 반복이 **구조적으로 없어** 비교 불가 | **미측정으로 둔다** (성립 확인도 아님 — 확인할 반복이 없다). 사전등록이 스스로 증거력 0 이라 했으므로 어느 쪽이든 판정 하중 없음. IDENTITY 3세대의 값 동일은 [실측]이나 상수 채움도 같은 관측을 내므로 증거로 세지 않는다 |
| — | `EMISS_IDENTITY_*` 거부 발화 | [실측 판정자 grep] **0건** (stderr·stdout) — 결박 검증이 생산 3회 발행 전부에서 통과 | 분기표 3행 미발화 (§4) |

산출물 3종 실재 [실측]: `physics_DET_iter0000.{manifest.json(1,377B), shell.csv(24KB), spectral.csv(16.5MB)}`.
`model.rc=0` [실측]. 이 구성 계열(det_stage12_fixed_te_a100x2_k36)의 세 런 중 DET 스냅샷을 낸 것은
이 런뿐 [실측 — l4·p1 런의 `work/physics_comparison/` 에 DET manifest 0건]. "결정론 팔 사상 최초"
전 역사 전수는 확인하지 않았다 [모름] — 다만 스냅샷 계측 자체가 08-18 이후 신설이므로 정황상 참 [추정].

### 1-1. B2 — 판정자 직접 재실측 세 겹 [전부 실측]

1. **정본 verifier 재실행**: `scripts/verify_a209_grid_manifest.py`(sha256 `a915c1d9…` — 스테이징
   사본과 동일 [실측]) 를 판정런 manifest+CSV 에 실행 →
   `A209_GRID_MANIFEST_VERIFY PASS n_shells=50 n_bins=1234 grid_manifest_sha256=49a7fa33…` rc=0.
2. **판정자 자신의 제3 구현**: verifier 코드를 쓰지 않고 부록 A-1 명세만으로 별도 작성한 파이썬으로
   CSV 에서 edge 복원(50셸 전부 동일·연속·엄격 순증가·전부 양수 확인) 후 재해시 → `49a7fa33…` 일치.
   상수 채움이라면 SHA-256 역상 불가로 CSV 와 일치할 수 없다 — B2 자문의 "verifier 가 C 의 재서술이면
   공허" 위험이 판정자 독립 구현으로 추가 봉쇄됨.
3. **source 해시의 부록 A-2 재계산**: 도메인 문자열+`u64_be(0x7)` 를 판정자가 hashlib 로 직접 계산 →
   `d14856a5…` == IDENTITY 줄의 `source_manifest_sha256`. 직렬화 관례(도메인 문자열·big-endian u64)가
   생산 경로에서 **byte 단위로** 성립함의 별도 확인.

보강(운전석 실측, 보존): C 도메인 문자열만 v1→vX 주입 시 `P4_APPENDIX_A=FAIL`·복구 시 PASS
(`NC_driver_demo.log`) — C↔Python 이 각자 부록 A 사본을 든 독립 구현임의 실험적 입증.

### 1-2. B3 의 잔여 가설 배제 [실측 판정자 diff 판독]

커밋 `2dc2817` 의 `src/lumina_plasma.c` diff: atomic 은 `atom->partition_stamp.atomic_model_sha256`
의 65바이트 **memcpy** 이며, 복사 전 3중 결박(`status==POP_OK`·`computed_population_generation ==
c.population_generation`·hex64)이 사전등록 §2-Q3 문언 그대로 있다. 재해시 없음(이중권위 회피).
manifest 의 atomic 은 소비자가 **온도측**(`te->atomic_model_sha256`)에서 쓴다
[실측 `physics_comparison.c:513`] — 따라서 B3 의 일치는 em(발행체)↔te(온도 발행체)가 같은
partition stamp 를 물었다는 실측 관측이다. 값 `b779a811…` 은 직전 판정런(320568)의 온도측 실측과도
동일 [실측 — 직전 판정문 인용 로그] — 결정론 구성 간 교차 일관.

---

## 2. 게이트 대조표 (사전등록 §4 의 번호 **그대로**)

| # | 사전등록 문언 | 실측 | 증거 계급 | |
|---|---|---|---|---|
| **P1** | CPU+GPU 두 타깃 에러 0·format 경고 0, **로그 파일 보존** | 로그 `P1_build_cpu.log`(21.5KB)·`P1_build_gpu.log`(3.8KB) 영속 보존. [실측 판정자 판독] "error" 매치는 양쪽 각 1건이며 둘 다 **파일명 `cmf_error_envelope.c`**(컴파일 명령줄) — 실제 에러 0. format 경고 0. ⚠비-format 경고 69건(CPU)·19건(GPU) 존재하나 전부 선재 부류(unknown-pragmas·unused 등)이고 **이 단 접촉 파일에는 0건** [실측] | 판정자 로그 판독 | ✅ (직전 단 P1 의 "운전석 주장" ⚠가 이번엔 보존 로그로 해소) |
| **P2** | 오프라인 회귀 6종 PASS | `P2_regression.log`(04:43:58Z): 6/6 PASS. 구조화 산출 `validation/a2_09/A2_09_SELFTEST.json`(13:44, status=PASS·N1–N8 8/8·NC1–NC4 4/4) 부합 [실측 판정자 판독] | 운전석 실측(보존 로그+JSON) — **판정자 미재현**(로그인 노드 실행 금지) | ✅ |
| **P3** | NC1~NC4 시연 (FAIL 2·감도 2) | 같은 로그에 4/4. [실측 판정자 코드 판독] C selftest 의 판정식은 사전등록 §5 그대로 — `EMISS_IDENTITY_GRID_INVALID`·`EMISS_IDENTITY_ATOMIC_UNSEALED` **정확 strcmp**, NC1 은 1-ULP 격자에 다른 해시+헬퍼 재호출 일치, NC4 는 mask 0x7/0x3 상이, OR 술어 없음 | 운전석 실측 + 판정자 코드 판독 — 미재현 | ✅ |
| **P4** | C↔Py 부록 A 합치 | `P4_APPENDIX_A=PASS` + 독립성 실험(NC_driver_demo) + **판정자 자신의 생산 산출물 재계산**(§1-1, 픽스처 수준을 넘는 확인) | 운전석 실측 + 판정자 실측 | ✅ |
| **P5** | 판정런 B1·B2·B3 적중 | §1 — 전부 적중, 전부 판정자 재실측 | 판정자 실측 | ✅ |
| **P6** | 물리값 무접촉 | (a) [실측 판정자 diff 판독] 커밋의 신규 코드는 해시 계산·hex64 검증·결박 검증·stderr 1줄·카운터뿐 — eta/cdf/샘플러 수치 라인 0 접촉, 기존 발행 조건 변경은 신원 검증 추가분뿐. 검수(Fable)도 동일 확인. (b) 기존 수치 케이스 N1–N8 8/8 문언 불변 PASS | 판정자 diff 판독 + 운전석 실측 | ✅ |
| **P7** | 커밋 접촉 파일 = §3 목록과 정확 일치 | [실측 판정자 `git show --name-status 2dc2817`] 11파일(M 10+A 1) — §3 의 11행과 1:1. 접촉 금지 4종(`physics_comparison.c`·checker 2종·모든 `.cu`) 무접촉 | 판정자 실측 | ✅ |

바이너리 신원 사슬 [실측 판정자]: 게이트 빌드 `binaries.sha256`(lumina_cuda `75664b75…`) →
스테이징 `input/binary.sha256` → 실파일 `sha256sum input/lumina_cuda` → `RUN_FOOTER.binary_sha256` —
네 값 동일. job.slurm 이 deck·sigma·binary·topion 을 sha256 봉인 검증함 [실측 판독].

---

## 3. ★`DET_FLIGHT_ACCEPT`(정확히는 수락 토큰) 미취득의 처분 — **재런 불요**

### 사실 [전부 실측 판정자]

- `model.rc=0`(모델 성공 — 이 구성에서 처음). slurm `sacct`: **FAILED**, ExitCode 70,
  `slurm-321104.err` = `DET_FLIGHT_FATAL A2-10 targeted log gate failed`.
- 죽은 자리: `job.slurm:241-249` — 모델 종료 **후**의 사후 로그 게이트. 이 targeted 모드에서
  취득했어야 할 토큰은 `A210_TARGETED_GATE_ACCEPT`(`TARGETED_GATE_VERDICT.txt`)다 — `DET_FLIGHT_ACCEPT`
  는 비-targeted 분기의 토큰 [실측 job.slurm 판독; 패킷의 명명은 이 점에서 부정확하나 실질 동일].
- **오탐 입증 (판정자 재실행)**: 같은 stderr.log 에 스테이징 사본(sha `4b3c3fcc…`)을 job.slurm 인자
  그대로 돌리면 `A210_TARGETED_GATE_FAIL nonzero numerical repair field cap=64` rc=4, 저장소 정본
  (sha `dffd4d0d…`)은 `A210_TARGETED_GATE_PASS devices=2 refinements=36 floor=0 cap=0 clamp=0
  jitter=0 repair=0` rc=0. 두 판본의 diff 는 **예외 조항 하나**(`a00b991`, 08-18): unqualified
  `cap` 이 `[cmf_fine][EXACT-MULTIGPU-EPOCH]` 줄에 있으면 제외 — 그 줄의 `cap=64` 는
  `LUMINA_CMF_FINE_ALI=64`(RUN_FOOTER [실측]) 솔버 최대 반복수이고 실반복 45·52 회, residual 둘 다
  tolerance 1e-08 미만, 같은 줄의 floor/clamp/jitter 전부 0 [실측 stderr:12,23]. **물리값 캡이 아니다.**
- 귀속: 운전석의 스테이징 실수(L4 런 input 통복사) — `docs/GATE_RECOVERY_INVENTORY_2026-08-18.md` §L.

### die 가 생략한 것은 둘뿐이고, 둘 다 회수됐다 [실측 판정자]

| 생략분 | 성격 | 회수 |
|---|---|---|
| 스냅샷 수렴 게이트 (`check_det_convergence.py`, job.slurm:250-258) | 보존 산출물의 순수 함수 | **판정자가 job.slurm 인자 그대로 오프라인 집행** — 스테이징 사본==저장소 정본(sha `0513f20f…` 동일 [실측]) → `DET_CONVERGENCE_CONVERGED iterations=1` rc=0, status=CONVERGED, manifest 불변량에 `grid_manifest_sha256=49a7fa33…` 포함 |
| `A210_TARGETED_GATE_ACCEPT` 토큰 + exit 0 | 기록 행위 (측정 아님) | 회수 불가·불필요 — 아래 조건 (a) |

### 판정

**재런 불요. 정본 검사기 재실행으로 족하며, 그 재실행은 이 판정에서 판정자가 이미 집행했다.**

근거: (1) 죽은 게이트는 모델 실행에 영향 없는 사후 로그 판독이고 [실측 job.slurm — 검사기는 실행
전 `-x` 만 검사], 모델은 rc=0 로 끝났다. (2) 판정에 필요한 모든 측정(B1–B3·targeted 게이트·스냅샷
게이트)이 보존 산출물 위에서 재실행·재계산됐다. (3) 재런은 새 정보를 하나도 낳지 않는다 — 열린
질문이 없는 런 발주는 offline-first 규약 위반이고, A100×2 약 2시간+대기열의 낭비다.

조건: **(a)** slurm FAILED 기록과 수락 토큰 미취득 사실은 지우지 않는다 — 런 디렉토리에 토큰을
소급 생성(위조)하지 않으며, 원장에는 "낡은 사본으로 FAILED, 정본으로 PASS(판정자 재실행)"를
그대로 적는다. **(b)** 다음 판정런 전에 §L 처분(스테이징 검사기 sha 봉인)을 별도 단으로 처리하거나,
최소한 스테이징이 저장소 정본을 신선 복사하고 그 sha 를 RUN_FOOTER 에 기재한다 — §6 의 신규 발견이
이 요구를 더 강화한다.

---

## 4. 폐합 — 사전등록 §6 철회·분기표의 축자 적용

| 분기 행 | 발화했는가 [실측] |
|---|---|
| site=133 이 여전히 `grid_manifest_sha256_valid=0` → 철회 | 미발화 (site=133 0줄) |
| site=133 이 다른 필드로 발화 → 보류 | 미발화 |
| 정상 경로 `EMISS_IDENTITY_*` 거부 → 발견·미폐합 | 미발화 (0건) |
| B2 불일치 → 폐합 금지 | 미발화 (3중 일치) |
| 후속 게이트에서 죽어 **산출물 0** → 부분폐합(B2 미회수) | **전제 불성립** — 후속 게이트에서 죽은 것은 맞으나 산출물 3종이 전부 나왔고 B2 가 회수·검증됐다 |

★마지막 행의 미묘함(패킷 §3)의 처리: 그 행이 부분폐합을 명한 **이유**는 "B2 미회수, 다음 단에서
회수"다 — 죽음 자체가 아니라 핵심 증거의 부재가 강등 사유였다. 실제로는 스냅샷이 게이트 **앞**에서
이미 원자적으로 기록됐고(3파일 일괄 rename) B2 는 회수됐으므로, 강등 사유가 존재하지 않는다.
행을 문언 그대로도, 유추로도 적용할 수 없고 적용할 필요도 없다.

⟹ 어떤 분기 행도 발화하지 않았고, P1~P7 전부 PASS, 성패 기준(§1: NC1+B2 = "격자를 바꾸면
바뀐다")이 생산 경로에서 성립한다. **전폐합.**

**전폐합이 뜻하는 범위** (이것을 넘겨 쓰지 말 것): 계약 문장 하나 — "발행체가 커밋 시점에 세 신원
필드를 스스로 계산해 봉인하고, 봉인할 수 없으면 이름 있는 사유로 발행을 거부한다" — 가 생산
경로에서 성립함. 그 이상(아래 §5)은 이 폐합에 포함되지 않는다.

---

## 5. 미결로 남는 것 (전폐합 범위 **밖** — 사전등록이 선언한 한계와 판정이 추가 확인한 것)

1. **atomic 해시 정의역 한계**: 선 목록(A_ul·ν·소속) 미봉인 — 정본 헬퍼 자체의 한계, 온도 발행체와
   공유(부록 B-2). 대장 기재 후보.
2. **source v1 의 정보량**: 사실상 `channel_mask` 하나(나머지는 관례 상수) — 사전등록이 정직하게
   선언한 한계(§2-Q2). SPEC 원 의도는 여전히 모른다(부록 B-1).
3. **writer 측 거부 경로**(`EMISS_IDENTITY_ATOMIC_STAMP_INVALID`) 는 시연된 적 없다 — 단위 시험
   미주입(사전 선언된 커버리지 구멍, §5), 판정런에서도 미발화(정상 경로이므로 당연). 코드 실존과
   문언 일치는 [실측 diff]이나 **발화 시연은 없다.**
4. **B4 미측정**: 반복 간 불변량은 outer_iterations≥2 인 런이 나올 때까지 미확인.
5. **em↔te atomic 동일성의 소비자측 강제 부재**(부록 B-4) — 이번 일치는 관측이지 강제가 아니다.
   후속 단 후보.
6. **te-측 손 채움 픽스처 잔존**(부록 B-3)·**R3**(pub 불변 검사가 얕은 memcmp)·**R4**(거부 후 후보
   신원 필드 부분 변경 — 읽기 금지 주의)·**R5**(`check_a209_source_failclosed.py` stale, 선재 결함 —
   §L 계급, 별도 단).
7. **P2·P3 는 판정자가 재현하지 않았다**(로그인 노드 실행 금지) — 보존 로그·JSON·코드 판독으로
   대조했고 판정 하중은 판정자 실측(B1–B3·P7·오탐 입증)에 있으나, 재현이 아닌 것은 아닌 것이다.
8. **133 너머 물리의 옳음**: 이 단은 신원 봉인 계약만 폐합한다. 스냅샷 **내용물**(J/χ/η 값)의
   CMFGEN 대조는 아무것도 판정되지 않았다 — 그것이 다음 측정 단(A2-10 Stage-4 J/O 귀속)의 일이다.

---

## 6. ★판정 과정의 신규 발견 — 스테이징 provenance 캡처도 낡았다 (같은 계급 **4번째**)

[실측 판정자] 세 런(l4·p1·idseal)의 `input/git_head.txt` 가 **전부 동일값 `dd9f7c18`**
(DET-STAGE12 커밋, 08-19 21:11)이고, **mtime 이 나노초까지 동일**(08-19 21:12:18 — 단일 캡처를
`cp -p` 로 복제한 사본). 그러나 p1 런의 실코드는 `ccfaab1`, 이 판정런은 `2dc2817` 이다.
`git_status.txt`·`git_diff.stat` 도 같은 시점의 화석이다. ⟹ **런 산출물 안의 git provenance 기록은
세 런 모두 실제 비행 코드를 기술하지 않는다.** 이 판정런의 코드 신원은 OWNER.txt(수기)와 §2 의
바이너리 sha 사슬, 그리고 낡은 트리(dd9f7c18)로는 원리적으로 낼 수 없는 런타임 출력
(`[A2-09][IDENTITY]` — 2dc2817 신설)으로 별도 입증되므로 **이 단의 판정에는 영향이 없다.**
처분: §L 수리 단의 범위에 포함할 것(검사기 sha 봉인 + provenance **신선 캡처** 의무).

부수 확인 [실측]: 게이트 보존 경로의 `orphan_gate_status.log` 내 FAIL 4건(event-measure-check·
line-saturation·sh-radeq-source·tau-writer-census)은 전부 GR 캠페인이 §F·§I 에 선재 부채(앵커
노후 등)로 문서화한 것들이며, tau-writer-census 가 지목한 `lumina_plasma.c:19521` 은 이 단 diff
(9100–9147)와 무관하다. **이 단 소행 아님.**

---

## 7. 판정 결론

- **SH-A209-IDSEAL: PASS, 전폐합.** B1·B2·B3 적중(전부 판정자 재실측), B4 미측정(증거 0 — 세지
  않음), P1~P7 전부 PASS, 철회·분기 행 발화 0. 성패 기준 "격자를 바꾸면 바뀐다"가 생산 경로에서
  성립 — B2 는 판정자 3중 독립 검증(정본 verifier·판정자 제3 구현·A-2 재계산)으로 확정.
- **수락 토큰 미취득은 오탐**(낡은 스테이징 검사기, 내용 diff 와 양판본 재실행으로 판정자가 입증) —
  **재런 불요**, 생략된 스냅샷 게이트까지 판정자가 오프라인 집행해 CONVERGED. 단 FAILED 기록은
  지우지 않고, 스테이징 봉인 수리를 다음 판정런의 전제로 삼는다.
- 신규 발견 1건(§6): provenance 캡처 화석화 — §L 과 같은 계급 4번째. 별도 단으로.

### 운전석이 다음에 할 일

1. 이 판정문의 **감리 발주** — fresh 컨텍스트 Fable, 고정질문 3 (판정과 같은 컨텍스트 금지).
2. 감리 통과 후: 판정문 커밋, 3대 검증 대장·회귀 대장 기재 (B 대장 전이 규칙대로 — 이 단은
   경로 2개 이상[NC 생산 체인 + B2 독립 재계산]이므로 A 후보).
3. §L 수리 단 발의(사전등록부터): 스테이징 검사기·판정 스크립트 sha 봉인 + provenance 신선 캡처
   (§6 발견 포함). **다음 판정런 전 처리.**
4. `check_a209_source_failclosed.py` stale(R5)·GPU 부하 불균형(census 8절 — hot spot 실측 특정
   먼저)은 각각 별도 단 후보로 대장에 유지 — 이 단에서 고치지 않는다.
5. 본선 복귀: 신원 봉인이 열어 준 결정론 스냅샷 위에서 A2-10 Stage-4 생산자 실제 S 직접 캡처
   (`docs/VERDICT_A210_STAGE4_JO_2026-08-18.md` 의 다음 단) — 우회를 끝내고 배선도/물리 검사의
   주된 경로로.
