# 단 사전등록 — DET-L6C-COVER: 커버 이온 재선정 L6 — A2-07 solve 권한 안에서의 J̄ 응답 판정 (판정런 1회) (2026-08-22)

저자 = Fable (분담 개정14: 사전등록=Fable). 발주서 앞겹 = **이 문서 원문 그대로**(재서술 금지).
표적은 **user 결정(2026-08-22)** — 기각된 대안(STAGE4 승격 런) 포함 갈림길 평가는 불요, 형상만
저자가 정했다. HEAD `8526d2c`, branch `thenmc-macroatom-fluorescence`. 본 문서의 소스 인용은
전부 **HEAD `8526d2c` blob 기준**이며(저자 실측: `src/`·`tests/` 작업트리 clean, tracked dirty 는
`validation/a2_09,10/*.json` 2건뿐; `git diff 5d711d06..8526d2c -- src tests Makefile` **공집합**),
봉인 런루트(`sprim_l6_20260821T054111Z_probe`·`idseal_20260820T044703Z_a209`)는 **read-only 로만**
접근했다. **부호 관례(감리 권고 (b) 반영): 이 단의 모든 잔차·대조량은 `resid := pred − meas`
하나로 통일한다.**

## 0. 표적 수령과 형상 판단 (겹 1 — 저자 요약)

1. **user 결정**: "이미 NLTE 해 집합 안에 있는 이온에서 'A2-07 solve 가 선 복사장 J̄ 에 옳게
   응답하는가'를 묻는다" (사유: 원래의 물리 질문에 더 가깝다). 이는 A207 판정 §9 "없는 것" 1번
   — *"A2-07 solve 가 커버된 이온에서 J̄ 에 옳게 응답하는지 — 판정하지 않음"* — 을 정면 표적으로
   삼는 것이다.
2. **형상**: 봉인 L6 구성에서 **유일 물리-무영향 delta 하나**(A2-10 진단 선택자
   `LUMINA_A210_LINE_SATURATION_TARGET_ION` 3→**1**)로 판정런 1회. 측정 기계(원시량 장부·재구성
   항등·마커 브래킷·판독 기계)는 DET-SPRIM-L6 이 검증 완료한 것을 계승한다.
3. **"옳게"의 조작화 (과대 주장 봉쇄)**: 이 단이 판정하는 것은 ① 도달(커버 이온 생산자 선
   소스가 solve 산출에 귀속되는가 — 동결 클래스와의 판별) ② 응답의 실재(J̄ 신호 라벨 하의 이탈
   구조), 그리고 ③ 부호·규모 정합은 **관측(하중 0)** 이다. 다준위 전 정량의 옳음은 오라클
   (CMFGEN 런 INELIGIBLE) 없이 판정 불능 — §8 에 등재한다. solve 자신의 SE 폐합(R1/R2 exact
   residual < 1e-8 등급, 봉인 실측 9.67e-9)은 기존 런 게이트가 이미 강제한다.

## 1. 계약 (하나)

> **A2-10 계측 표적을 봉인 구성의 NLTE 권한 안(커버 이온, ion=1·Z∈{26,27,28})으로 재선정한
> 판정런 1회로, 반복 1 의 커버 이온 생산자 선 소스가 LTE-공급 동결 클래스
> (|S₁/S₀−1| ≤ 1e-13, L6/A207 교정 실측 ≤6.7e-16)와 구별되게 A2-07 solve 산출에 응답하는지를
> 기계 분기로 판정한다.**

수리 아님·src/tests 접촉 0·물리값 무접촉·봉인 무변조(read-only)·덱/`/gpfs` 정본 불변.
⟹ **V5 원상 유지 — source edit·K-final 권한을 요청하지 않는다.** 런처·stager·판독기는 집행
인프라이지 src 가 아니다(L6 §3-1 전례의 같은 분류).

## 2. 설계 쟁점 6건의 판단 (발주서가 이 판단을 좁히면 그것이 위반이다)

### 2-1. TARGET_ION 핀 = 등록 노브 재설정 (계약 변경 분리 안 함)

- `LUMINA_A210_LINE_SATURATION_TARGET_ION` 은 진단 선택자다: 파서 `src/lumina_plasma.c:13953-13965`
  (기본 3, 0..10 수용), 소비는 A2-10 진단 층뿐. **물리 경로 무영향은 실측이다** —
  `docs/VERDICT_DET_SPROD_S4_III_2026-08-19.md` §1: III(ion=2) vs IV(ion=3) 런에서 R1/R2 solve·
  캡처·브래킷 4국면 전부 **동일**. 이 단은 그 무영향을 게이트 R7 로 **이 구성에서 재검증**한다.
- A207 §2-2 의 "재선정은 별도 사전등록 사안" 은 **이 문서가 그 별도 사전등록**임으로 충족된다.
  A2-10 선정 알고리즘(PER_ION_UNION·90% 목표, `:14294`·`:14312`)은 불변 — 바뀌는 것은 선택자
  값 하나다.
- ★**핵심 위험 = S4-III 차단 전례**: target_ion=2 는 `INDEPENDENT_SPROBE_UNDEFINED`
  (χ_eff==0·η>0 행에서 A2-10 트랜잭션 전체 사망, `:14111-14139`)로 전면 차단됐다. 그 수리
  (A210-ZERO-OPACITY)는 **미착지·user 보류** — 이 단은 손대지 않는다. 처분:
  **표적 = ion=1(분광 II) 단일** (근거 §2-2), 발주 전제 = 오프라인 hazard census PC-2, 잔여
  위험(반복 1 의 NLTE 산출이 만드는 χ==0)은 분기 D2 가 받는다. **회피는 수리가 아니다** — Z-O
  는 존속하며 PC-2 실측(ion=2 hazard 행 수 포함)은 Z-O 사전등록의 입력으로 병기된다.

### 2-2. 표적 이온 선택 — ion=1, 기각 사유 명시

| 후보 | 판단 |
|---|---|
| **ion=2 (III)** | **기각** — S4-III 실측 차단 전례(측정 자체가 미수리 결함에 물려 있음). Z-O 수리 선행 없이는 판정런이 D2 로 소모될 [추정 강] 위험 |
| **ion=1 (II)** | **채택** — base 31슬롯에 (26,1)(27,1)(28,1) 실재(`:9791-9798`), 차단 전례 0, 봉인 슬롯 실측 Z=26:2599·Z=27:2558·Z=28:1000 준위. shell 0 선 방출의 사실상 전부가 NLTE-SE 소유(LINE-OWNER-FORENSIC 0.99999998 — II·III 가 그 구성원) |
| **ion=0 / 타 Z** | 기각 — A2-10 표적 필터가 Z∈{26,27,28} 하드코딩(`:14027-14030`)이라 Z 는 선택 대상이 아님; ion=0 은 Fe/Co/Ni 슬롯에 없음 |

ion=1 의 약점(정직 기재): shell 0·10020 K 에서 II 는 지배 이온화 단계가 아니고 그 선 진동수의
J̄/B 이탈 크기는 **미지**다. 신호 부재 시에도 동결 판별(도달 질문)은 성립하며, J̄-응답 판정만
라벨 σ 로 강등된다(§6).

### 2-3. 분기 도달 가능성의 게이트화 (쟁점 2 — L6 §3-8·A207 §8 발견 1 의 처방)

*"측정 행이 피측정 기전의 권한 안에 있는가"* 를 3중으로 게이트화한다:
① **PC-1(런 전, 오프라인)** — HEAD blob 테이블 사상 사전 증명 + ★SKIP-마스크 교집합
   판정식(`skip_tau` 도 권한 사슬의 일부다 — §3-2) + 봉인 실물 594행 NC,
② **R5(런 행 전수)** — f_mapped = 1.0 정확,
③ **성공 분기 판정식이 R3 census 를 명시 conjunct 로 포함**(감리 비차단 권고 (a) 축자 반영).
추가로 **신호 전제**(J̄/B 이탈의 실재)를 라벨 σ 로 분기 해석에 넣는다 — 권한과 신호가 다
갖춰져야 null 이 정보가 된다는 것이 L6→A207 사슬의 교훈이다.

### 2-4. 행 수 기대식 (쟁점 3 — E1 계급 재발 방지)

**행 수 기대치를 등록하지 않는다.** 행 수는 가중 분포의 함수이지 불변량이 아니고
(`선정 = 총가중 90% 도달까지`, 가중 `scaled_emission=result->emission_per_sr×rate_factor`
`:14079`), 이 단의 delta(TARGET_ION)는 **후보 집합 자체를 바꾼다**(S4-III 실측: ion=2 후보
51,807 vs ion=3 후보 211,887). 등록하는 것은 선정 불변량(selected_fraction, census 전량 보고)과
최소 표본 하한(§6 E1)뿐이다. 행 수·후보 수·총가중은 관측(E5)으로만 기재한다.

### 2-5. 판정런 필요성 (쟁점 5 — 오프라인 폐합 불능을 소스로 닫음)

봉인 L6 런에 커버 이온 선-단위 데이터는 **없다**: ① ROW 장부는 표적 필터를 통과한 행만 인쇄
(`:14045` — 전부 ion=3) ② `LINE-COEFFICIENT-IDENTITY` 는 셸 단위 집계(`:15338` 의 셸 루프 —
50셸×2반복=100줄, 선 분해 없음) ③ snapshot 은 shell/spectral CSV ④ SPRODUCER 캡처 배열
(109,014,300 셀)은 메모리 전용이고 내보내기 경로는 ROW suffix 뿐. ⟹ **판정런 1회.**
단 빌드는 0회다: src 가 `5d711d06` 이래 불변(공집합 diff, 저자 실측)이므로 **봉인 L6 바이너리
(`input/lumina_cuda`, sha `b9a30a81ebea57f9fa857d192107dd85aeb04ab1308f27b1a68cf45f1a69af99`)를
그대로 재사용**한다(봉인 원본은 read-only 복사만; PC-5 가 발주 HEAD 에서 재검증).
실측 표기(개정15 §3-3): ① 표적 필터·suffix 인쇄 `sed -n '14040,14050p;13944,13952p'
src/lumina_plasma.c` [저자 실행] ② 셸 집계 `sed -n '15290,15375p' src/lumina_plasma.c`
[저자 실행] ③④ snapshot 파일명·캡처 배열 메모리 상주 = L6 판정 G2·L6 사전등록 §4 폐합 인용;
diff 공집합 = `git diff --stat 5d711d06..8526d2c -- src tests Makefile` 무출력 [저자 실행];
바이너리 sha `b9a30a81…` 는 L6 판정 §1 기재 인용 **[미실행]** — 발주 전 운전석 실행:
`sha256sum $L6/input/lumina_cuda` → `b9a30a81ebea57f9fa857d192107dd85aeb04ab1308f27b1a68cf45f1a69af99`
일치 확인(대용량 해시라 로그인 노드 비연산 범위 밖 — grammar-debug 에서).
★**[운전석 실행 2026-08-22]**: grammar-debug 에서 실행 — 실측 `b9a30a81ebea57f9fa857d192107dd85aeb04ab1308f27b1a68cf45f1a69af99` = 기재값 **일치**. 이 문서의 `[미실행]` 주장은 0 이 됐다.

### 2-6. W5 미결 오프셋 (쟁점 6)

**건드리지 않는다.** `pred−meas = +7.265e-6 ± 3.3e-7` 은 ion IV 행의 LTE-공급 서명 미세구조
관측이고, 이 단의 iter1 커버 행은 LTE 예측의 시험대가 아니다. 대장 존속(§8-7).

## 3. 기전 오프라인 특정 (근거를 가른다 — 저자 실측·폐합 인용·[미실행]; 추측은 [추정])

★개정15 §3-3 준수: 봉인·실행 상태 주장마다 실측 명령을 병기한다. 표기 3종 — **[저자 실행]** =
저자가 이 명령을 실제 실행(명령줄이 곧 재현 절차) / **폐합 인용** = 이미 폐합된 판정문·사전등록의
실측 인용(재실행 의무 없음 — 원장이 근거) / **[미실행]** = 저자가 실행하지 않은 주장 —
★운전석이 발주 전 전건 실행한다. 소스 명령은 전부 HEAD `8526d2c` 작업트리 기준(전제 실측:
`git status --short src tests` 무출력 ∧ `git diff --stat HEAD -- src tests` 무출력 [저자 실행]).
약칭: `$L6` = `/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe` ·
`$IDS` = 같은 부모의 `idseal_20260820T044703Z_a209` (접근은 전부 read-only).

1. **표적 필터**: `a210_line_saturation_target`(`src/lumina_plasma.c:14027-14030`) —
   `ion==target_ion ∧ Z∈{26,27,28}`(Z 하드코딩). env 파서 `:13953-13965`.
   실측: `sed -n '13953,13965p;14027,14030p' src/lumina_plasma.c` [저자 실행].
2. **커버리지**: base 31슬롯 `NLTE_TARGET_Z/ION`(`:9791-9798`)에 (26,1)(27,1)(28,1) 실재.
   ★**사상 전수성은 구성상**이다: `nlte_build_projection` 의 준위 사상 루프(`:16413-16429`)는
   대상 (Z,ion) 의 **전 준위를 절단 없이** 사상하고(용량 컷 없음), 선 사상 루프(`:16430-16439`)
   도 (Z,ion) 일치 전 선을 사상한다 ⟹ 커버 이온의 어떤 선도 준위-단위 unmapped 가 될 수 없다.
   authority 사슬(`:8903-8951`): mapped ∧ !skip_tau ∧ pair-owned ∧ 세대 정합 → NLTE 소비.
   봉인 env 에 `LUMINA_NLTE_STAGE4`·`ELEMENT_WIDE`·`SUPER_LEVELS` 부재(A207 W1 실측 — W1 의
   금지 목록은 이 셋뿐이다). ★SKIP_Z 는 부재가 아니다: 봉인 base·L6 양 루트 exports 와 L6
   RUN_FOOTER 에 `LUMINA_NLTE_SKIP_Z="14"` 실재 [운전석 실측 2026-08-22 · 저자 재확인]. 의미론
   (저자 소스 실측): 파서 `:8845-8856`(미설정·빈 값=공집합 `:8849`, 구분자 콤마/공백/탭,
   토큰별 atoi, 0<z<100, 256B 절단), 적용점 `:8922`(`skip_tau`)·`:8945-8951`(`uses_nlte` 가
   `!skip_tau` 요구) ⟹ SKIP_Z 에 든 Z 는 mapped 여도 NLTE 소비가 꺼져 동결-클래스 공급 경로로
   회귀한다. 14=Si ∉ {26,27,28} ⟹ 표적 권한과 교집합 공집합. 따라서 요구는 "부재"가 아니라
   **파스 집합 ∩ {26,27,28} = ∅** 다(PC-1 판정식, ★user 결정 2026-08-22 선택지 2). 별개 마스크
   `LUMINA_OPACITY_SKIP_Z`(`:3139-3153` 파스·`:3257` tau 0화)는 봉인 양 루트 exports 부재
   실측(=공집합)이며 PC-1 이 같은 판정식으로 함께 검사한다.
   실측 명령: 테이블·사상 루프 `sed -n '9780,9815p;16380,16460p' src/lumina_plasma.c` +
   `grep -n global_to_nlte_level src/lumina_plasma.c` [저자 실행]; SKIP 두 마스크
   `grep -h SKIP_Z $L6/input/resolved_lumina.exports $IDS/input/resolved_lumina.exports` →
   `declare -x LUMINA_NLTE_SKIP_Z="14"` 2건뿐(OPACITY 계열 0건) ∧ `grep SKIP_Z
   $L6/RUN_FOOTER.txt` → `LUMINA_NLTE_SKIP_Z=14` [저자 실행 2026-08-22]; STAGE4/EW/SUPER 부재
   = A207 W1 폐합 인용 + 저자 재실측 `grep -cE
   'LUMINA_(NLTE_STAGE4|NLTE_ELEMENT_WIDE|SUPER_LEVELS)=' $L6/RUN_FOOTER.txt` → **0** [저자 실행].
3. **solve 의 J̄ 소비 (커버 이온)**: production split `:17401-17425` 이
   `nlte_bb_jbar_canonical`(`:583`)만 소비 — A207 §6(a) 폐합 인용 + 저자 재실측:
   `sed -n '17401,17425p' src/lumina_plasma.c` → "every ordinary path consumes only the
   checked line view" 주석·`nlte_bb_jbar_canonical` 소비 문면 실재 [저자 실행 2026-08-22].
4. **판정량의 공급 사슬**: iter k 의 `producer_eta/tau_eff` = R6 fine 이 읽은 **gen-(k+1)
   pre-commit 물질**(iter0=시드, iter1=iter0 solve 커밋분) — L6 사전등록 §3-3 실측 계승.
   ⟹ **d := |S₁/S₀−1| = |(η₁/η₀)/(τ₁/τ₀)−1|** (같은 ν·t 라 정확 항등)가 공급 경로 판별량이다.
   LTE-공급 경로는 이 양을 대수적 항등으로 ≤ ulp 급으로 동결시킨다(L6 실측 ≤6.66e-16, A207 W4
   한계 1e-14·여유 15×). 반복 선형 solve 산출이 해석적 LTE 값과 전 행 ulp 일치할 수는 사실상
   없다 [추정 강 — solve residual 등급 1e-8 실측 근거]. 문턱 **1e-13** = 실측 동결의 150×,
   [추정] 응답 등급(≥1e-9)의 1/10⁴.
   근거 구분: 공급-지연 구조(iter0=시드/iter1=커밋분)·6.66e-16·1e-14(여유 15×)·solve residual
   9.67e-9 는 **폐합 인용**(L6 사전등록 §3-3 / L6 판정 부록 A / A207 §3 W4 / S4-III §1);
   d 항등은 대수(S=η/χ_eff ∧ χ_eff=τ_eff·ν/(c·t) — L6 사전등록 §3-2 의 소스 정의 인용).
5. **차단 위험**: `INDEPENDENT_SPROBE_UNDEFINED` 전면 차단(`:14111-14139`, WITNESS 인쇄 후
   트랜잭션 사망)은 미수리 존속. 발화 조건 = χ_eff==0 ∧ 방출>0 ∧ independent_capture. iter0 의
   물질은 LTE-at-10020 [추정: 시드=LTE, FINDING 사슬]이므로 **iter0 위험은 오프라인 재현 가능**
   — 덱(`line_list.csv`+`levels.csv`, 봉인 read-only)에 `build_lte_level_density_cache` +
   `a208_signed_sobolev`(`:3219-3355`) 전사 산법을 적용하는 census 가 PC-2 다. [추정] 기전 =
   에너지 축퇴 준위쌍(LTE 에서 n_l g_u = n_u g_l 정확 상쇄 — T 무관). iter1(NLTE 산출) 위험은
   오프라인 예측 불능 — 분기 D2.
   실측: 차단·WITNESS·census 문면 `sed -n '14040,14140p' src/lumina_plasma.c` [저자 실행],
   LTE 경로 `sed -n '3219,3400p' src/lumina_plasma.c` [저자 실행], 덱 파일 실재
   `ls $L6/input/model/` → `line_list.csv`·`levels.csv` 확인 [저자 실행]; S4-III 차단
   (ion=2 51,807 후보·III 발화/IV 무발화)은 폐합 인용.
6. **선정층**: PER_ION_UNION 90% 목표(`:14294`·`:14312`), Z-1 census 카운터(`:14075`).
   실측: `grep -n 'target_fraction\*ion_total\|PER_ION_UNION' src/lumina_plasma.c` [저자 실행].
7. **집행 인프라의 ion=3 핀 2곳(변경 필요 근거)**: 런처 모드 가드
   `scripts/run_det_convergence_2026-08-08.slurm:177`(`TARGET_ION==3` 아니면 die), stager 는
   base 봉인 검증 `scripts/stage_det_stage12_l6_probe.sh:117`(base=3 확인 — **유지**)에 delta
   재작성 기계(`:169-174`·`:212-213`·`:249` resolved_env_delta_count=4)가 TARGET_ION 을 다루지
   않음. 실측: `grep -n TARGET_ION scripts/run_det_convergence_2026-08-08.slurm
   scripts/stage_det_stage12_l6_probe.sh` + `sed -n '100,130p;160,260p'
   scripts/stage_det_stage12_l6_probe.sh` [저자 실행].

## 4. 기대 변경집합 (이 목록 밖 변경 = 실패) + V5

**src/ 접촉 0 · tests/ 접촉 0 · env_universe 불변 · 신규 env 0 · 덱/`/gpfs` 정본 불변 ·
봉인 런루트 무변조.** ⟹ V5 원상 유지, 권한 요청 없음. 빌드 0회(§2-5).

### scripts/ — 집행 2파일 확장 + 신설 2파일

| 파일 | 변경 |
|---|---|
| `scripts/run_det_convergence_2026-08-08.slurm` | 신규 diagnostic_mode **A210_L6C_PROBE**: 기존 A210_L6_PROBE 블록과 동일 요구(outer=2·A100·MGPU 2·CAPTURE 2종·DIAG=2·수렴 주장 없음) 단 ion 가드만 `TARGET_ION==1` + 스테이징 provenance 파일(`l6c_target_ion.txt`) 일치 요구, 사후게이트·수용 토큰은 `A210_L6C_PROBE_ACCEPT`. **기존 A210_L6_PROBE 블록 무변경**(기본 경로 불변) |
| `scripts/stage_det_stage12_l6_probe.sh` | 선택 인자 `--l6c-target-ion 1`(허용값 1 만·엄격 파서): 재작성 규칙·assert 목록에 `LUMINA_A210_LINE_SATURATION_TARGET_ION="1"` 추가, `resolved_env_delta_count=5`, diagnostic_mode=A210_L6C_PROBE, `input/l6c_target_ion.txt` 기재. **무인자 기본 경로 = 기존과 byte 동일**(base 봉인 검증 `:117` 유지 — base 는 여전히 3). 쓰기는 전부 기존 temp+`mv` 3함수 경유(§18-1 ① 계승) |
| `scripts/analyze_det_stage12_l6c.py` | 신설 판독기(자립형): R7 마커 브래킷·ROW 파스·재구성 항등(R4b)·iter0 앵커(R4)·행 전수 census+사상(R5, base 테이블은 `git show <발주HEAD>:src/lumina_plasma.c` 파스)·d 분포·ff·σ 라벨·분기(§6)·verdict JSON 전량 보고. `--selftest` 에 NC-B1~B5 내장(전부 스크래치 사본 — 봉인 원본 불접촉) |
| `scripts/audit_l6c_cover_precondition.py` | 신설 사전 게이트 도구: **PC-1**(테이블 사상 증명 + ★봉인 실물 594행 × base/ION4 사상 NC **내장** — A207 §10-2 보완 후보 반영) + ★**SKIP-마스크 교집합 판정식**(§5 PC-1 등록분 — 두 마스크 공용 단일 파스·교집합 루틴) + **PC-2**(zero-χ hazard census: 덱 read-only × LTE(10020) 전사 산법, ion∈{1,2,3} 각각; 앵커 2종 §5) + verdict JSON. `--selftest` 에 NC-P1~P4(§5 등록 4종 전부 — P4=SKIP 주입) |

### 변경집합 끝

- 기타 산출(패턴 밖): `docs/RUNG_DET_L6C_COVER_2026-08-22.md`(본 문서),
  `validation/det_stage12/l6c_cover/`(PC verdict·판독 보고), 판정문.
- 커밋 규율: 사전등록 커밋(본 문서) → 도구 커밋 **1개**(scripts 4파일 + 오프라인 산출) →
  판정런 → 판정문 커밋. 검수·판정은 커밋 접촉 파일과 위 표의 1:1(초과 0·미달 0)을 확인한다.
- 실행 환경: 소형 오프라인 전부 **grammar-debug**(nested ssh) — 로그인 노드 연산 금지.

## 5. 게이트 표 (각 행: 요구 / 증거 / ★음성대조)

### 오프라인 (런 전 — offline-first ②; 전부 grammar-debug)

| # | 요구 (기계 판정식) | 증거 | ★음성대조 |
|---|---|---|---|
| **PC-1 커버리지 사전 증명** | 발주 HEAD blob 파스: (26,1)(27,1)(28,1) ∈ `NLTE_TARGET_Z/ION` ∧ 준위 사상 루프 무절단 문면(`:16413-16429`) 실재 ∧ 봉인 base env 에 STAGE4/EW/SUPER 부재 재확인 ∧ ★SKIP-마스크 판정식(user 결정 선택지 2): 소스 파스 규칙 전사(`:8845-8856`·`:3139-3153` — 미설정·빈 값=공집합, 콤마/공백/탭 구분, 토큰별 atoi, 0<z<100)의 **공용 단일 루틴**으로 `LUMINA_NLTE_SKIP_Z`·`LUMINA_OPACITY_SKIP_Z` 를 각각 파스한 집합 S 에 대해 **S ∩ {26,27,28} = ∅**(봉인 실측 기대: NLTE="14"→{14}·OPACITY 부재→∅, 둘 다 PASS; 위반 시 `SKIPZ_TARGET_OVERLAP` 이름 있는 FAIL) ∧ **봉인 실물 594행(ion=3) × base 사상 = 0/594 · × ION4 = 594/594**(A207 판별기를 실물로 내장 재실증) | PC verdict JSON | **NC-P1**: 같은 사상 검사를 ion=3 표적으로 실행 → `COVERAGE_ABSENT` 이름 있는 FAIL(0/594) — 도달 불능 표적이 게이트에 걸림을 시연 / **NC-P2**: 테이블 스크래치 사본에서 (26,1) 행 제거 → FAIL / **NC-P4**: env 스크래치 사본에 `LUMINA_NLTE_SKIP_Z="14,26"` 주입 → `SKIPZ_TARGET_OVERLAP` FAIL(다중 토큰 파스 동시 시연)·원복 시 PASS — 두 마스크가 공용 단일 루틴을 지나므로(요구 셀에 등록) 이 1건이 신규 판정식의 FAIL 가지를 덮는다 |
| **PC-2 zero-χ hazard census** | 덱(`line_list.csv`·`levels.csv`, 봉인 read-only) × LTE(10020) 전사 산법으로 Z∈{26,27,28} 별 χ_eff==0∧η>0 후보 행 수: **ion=1 = 0 이 발주 전제**. 앵커 ①: ion=2 ≥ 1 [추정 — S4-III 실측 차단의 재현] ② ion=3 = 0 (L6 census 0 재현). 앵커 ①빗나감(=0) 시: 기전 모델 미검증으로 **이름 있는 발견** 기재하되 ion=1 진행은 유지(차단 전례 0 + D2 안전망). **ion=1 > 0 이면 발주 중단** — 실측을 user 갈림길(Z-O 선행 여부)에 회부 | census 표(행 목록 전량) | **NC-P3**: 합성 축퇴 준위쌍 주입 → hazard 검출·제거 시 0 |
| **PC-3 스테이징 delta 봉쇄** | `--l6c-target-ion 1` 스테이징의 resolved env diff: vs IDSEAL base = **정확 5**(L6 등록 4 + TARGET_ION 3→1) + diagnostic_mode 파일; vs 봉인 L6 런 = **정확 1**(TARGET_ION) + diagnostic_mode 파일. base byte-seal(기존 ②기계) PASS | stager 로그·diff | 무인자 스테이징 → 기존과 **byte 동일**(기본 경로 불변 실증) / delta 1건 누락 주입 → 기존 seal FAIL(§18-1 ② 재사용) |
| **PC-4 판독기 검증** | `--selftest` rc=0 | selftest 로그 | **NC-B1** 봉인 L6 stderr(실물 동결 클래스) → **C-F 발화** 시연(주입 결함으로 결함-분기 FAIL 시연 — 음성대조 의무의 본체) / **NC-B2** 비표적 위조행 → R3 census FAIL / **NC-B3** η 1e-9 섭동 → R4b FAIL / **NC-B4** 사상 위조(테이블에서 (26,1) 제거) → f_mapped<1 검출 / **NC-B5** R7 마커 삭제 → BLOCKED(침묵 금지) |
| **PC-5 바이너리 동일성** | `git diff <발주HEAD> 5d711d06 -- src tests Makefile` **공집합** ∧ 재사용 바이너리 sha = `b9a30a81…` (봉인 L6 기재값) | diff·sha 로그 | 검사기를 IDSEAL 화석 HEAD `dd9f7c18…` 에 적용 → 비공집합 FAIL 시연 |

### 판정런 (1회)

| # | 요구 (기계 판정식) | 증거 | ★음성대조 |
|---|---|---|---|
| **R1 신선성** | RUN_FOOTER 의 job.slurm/checker/stager/analyzer_l6c/precondition sha = 발주 HEAD blob sha ∧ `git_head.txt` = 발주 HEAD ∧ binary sha = `b9a30a81…` | RUN_FOOTER·input/ | 검증을 봉인 L6 루트에 적용 → git_head 불일치 FAIL(L6 G1 NC 계승) |
| **R2 커밋 마일스톤** | `R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1 … te_generation=2->3` 정확 1건 + `[PHYSICS-COMPARISON] lane=DET` 2건 구조(`check_a210_targeted_gate.py --expected-outer-iterations 2` 재사용 — 무변경) | checker 보고 | 상설 NC (ii)/(iii) 기존 16+4종 유지 |
| **R3 행 전수 census** | iter0·iter1 각 ROW ≥1, 전 행 Z∈{26,27,28}∧ion=1, 비표적 행 **0**, `producer_terms/raw/independent_fields_defined` 100%, UNAVAILABLE=0, line-id 교집합 ≥**30** | 판독기 census | NC-B2 |
| **R4 잣대 앵커** | iter0 매치 행 S_prod/B(10020 K) 중앙 ∈ [0.999, 1.001] [추정 — 시드=LTE]. 실패 = ANCHOR_FAIL → D3, **분기 판정 금지·미결**(빗나감 자체가 시드 공급 경로의 발견) | 판독기 | 앵커 밖 픽스처 → FAIL |
| **R4b 재구성 항등** | 전 행: β_rec=exponx_py(τ_eff^prod) 재구성으로 캡처 두 항·Jbar 상대편차 ≤1e-12, 뺄셈·역산 0회(L6 G4b 기계 계승) | 판독기 | NC-B3 |
| **R5 권한 사상** | **f_mapped = 1.0 정확**(양 반복 전 행, base 테이블 사상 — A207 f_mapped=0 판별기의 거울) | 판독기 | NC-B4 |
| **R6 판정 (계약 본체)** | 매치 행 d 분포 → §6 분기 (판정식이 R2~R5·R7 PASS 를 명시 conjunct 로 포함 — 감리 권고 (a)) | verdict JSON 전량 | NC-B1(동결 → C-F)·solver-response 픽스처(→ C-R) |
| **R7 상류 무섭동 = TARGET_ION 물리-무영향 재검증** | 봉인 L6 런과의 마커-계열 byte 대조(★phase 라벨 조건 없음 — L6 G7 문면 결함의 개정 집행): 양 반복의 `[cmf_fine]` R1/R2 exact-solve 줄 + `LINE-COEFFICIENT-IDENTITY` 전 100줄이 봉인 L6 과 **byte 동일** [추정 강 — 동일 바이너리·동일 env(선택자 제외)·DET 레인 결정론 실측 계보]. FAIL = TARGET_ION 의 물리 개입 발견 → D3·판정 금지(그 자체가 중대 발견) | diff | 비교자 사본 1바이트 변조 → 검출(G6 계급) |

clamp/floor/cap 0. 판독기의 census 분류(INVERSION_BOUNDARY·NEGATIVE_CHI·EXACT_ZERO)는 물리값을
바꾸지 않는 분석 분류이며 전량 보고 — 조용한 탈락 금지("정당한 0"≠"무효", L6 검수 확증 계승).
음성대조는 전부 스크래치 사본 — 봉인 파일 쓰기 접촉 0.

## 6. 기대치 사전등록 (빗나가면 그것이 정보다) + 분기

| # | 기대 | 수치·범위 |
|---|---|---|
| **E1** | 행 실재: 각 반복 ≥30 매치 [추정]. **행 수 점 기대는 등록하지 않는다**(§2-4 — 이 단의 delta 가 후보 집합·가중 분포를 바꾼다) | R3 |
| **E2** | f_mapped = 1.0 정확(양 반복) | R5 |
| **E3** | iter0 앵커 중앙 1±1e-3 [추정 — 시드 LTE] | R4 |
| **E4** | 1순위 [추정] **C-R**, d 등급 ≥1e-9(solve residual 9.67e-9 등급 근거). C-F·C-M 도 실질 확률 — 사전 확률 부여 안 함, 기대 드라마화 금지 | R6 |
| **E5** | 관측(하중 0): ① J̄₁/B 분포(σ 라벨 문턱 q10<0.95 [추정]) ② `resid := pred−meas` 관례로 (S/B−1) vs (J̄/B−1) 부호 일치율·Pearson(2준위 관측 — 정합/부정합 병기만) ③ C-R 발화 시 d 서명이 단일-T 재조정으로 붕괴하는지 검사(붕괴하면 불리 정황 병기) ④ candidate_rows·selected_rows·총가중·selected_fraction(참고 대역 [0.900,0.905] — 두 봉인 런 실측 0.90008/0.90020) ⑤ zero_opacity_emitting_rows census(PC-2 예측과 대조) | 보고서 |
| **E6** | 자원: 경과 ≈ 봉인 L6 실측 04:34:10 등급 [추정], `--time=10:00:00`; MaxRSS ≈ 37.1 GiB 등급(행 수 변화 영향은 row 구조체 수십 MB [계산]) | — |
| **E7** | R7 byte 동일 [추정 강] | R7 |

**분기 (상호 배타·전수 — 어느 쪽이든 단은 착지)**

정의: 매치 행 i 에 대해 `d_i := |(η₁ᵢ/η₀ᵢ)/(τ₁ᵢ/τ₀ᵢ) − 1|` (= |S₁/S₀−1|, §3-4 항등);
`ff := frac(d ≤ 1e-13)`; `f_mapped` := R5; σ := (q10(J̄₁/B) < 0.95 ? SIGNAL : WEAK-SIGNAL).
평가 순서: D2 → D1 → D3 → W-V → C-F → C-R → C-M(잔여). 판정식은 순서 없이도 상호배타.

| 분기 | 판정식 | 함의 |
|---|---|---|
| **C-R** | f_mapped=1.0 ∧ R2·R3·R4·R4b·R5·R7 전부 PASS ∧ **ff ≤ 0.10** | **응답 실증** — 커버 이온 생산자 선 소스는 solve 산출이다(동결 클래스 이탈). σ=SIGNAL 이면 J̄-응답 관측(E5②③) 유효 병기; σ=WEAK-SIGNAL 이면 "쓰기 실증, J̄-응답 미판정" 으로 절단해 기재 |
| **C-F** | f_mapped=1.0 ∧ R2·R3·R4·R4b·R5·R7 전부 PASS ∧ **ff ≥ 0.99** | **커버 이온에서도 동결 재현** ⟹ 후보 단: A2-07 커밋→생산자 소비 사슬 감사(솔브 산출이 생산자에 미도달). ★이 함의는 **후보 라벨**이지 실증이 아니다(L6 분기 B 의 교훈 그대로) |
| **C-M** | f_mapped=1.0 ∧ 게이트 PASS ∧ 0.10 < ff < 0.99 | 부분 동결 — 미결 기재 + 행별 구조 전량 보고(어느 행이 동결인지의 census 자체가 다음 단 입력) |
| **W-V** | R3 비표적 행 >0 ∨ f_mapped < 1.0 | 권한 전제 위반 = **발견**(PC-1 의 구성상 증명과 모순 — 구성 이해의 결함). 계측 보존·미결 |
| **D1** | iter1 미도달, 이름 있는 차단 | 차단 사유·자리 = 발견. 산출물 보존 |
| **D2** | `INDEPENDENT_SPROBE_UNDEFINED` 차단 | PC-2 census 의 반증 + WITNESS 행 = Z-O 우선순위 증거. 부분 착지(계측 성립·판정 미결), 재시도는 Z-O 처분 후 |
| **D3** | 그 외 이름 있는 차단(R4 ANCHOR_FAIL·R7 FAIL·표본<30·파스 불능 등) | 사유 = 발견. 미결 기재, 폐합 금지 |

분기는 라벨이지 필터가 아니다 — verdict JSON 은 ff·d 분포 전량·σ·census 를 어느 분기에서든 보고한다.

## 7. 판정런 구성 (발주서 뒷겹과의 대조 기준 — 좁힘 검출용)

기준 = IDSEAL RUN_FOOTER(봉인). resolved env 값 delta 가 **정확히 아래 5건**이어야 한다:

| env/구성 | IDSEAL | 이 단 | 사유 |
|---|---|---|---|
| `LUMINA_PURE_CMFGEN_ITER`(+outer_iterations.txt+argv) | 1 | 2 | L6 계승 |
| `LUMINA_RADEQ_DIAG_TE_K` | 19059.411196903675 | 10020 | L6 계승 |
| `LUMINA_A210_SPRODUCER_CAPTURE` | 0 | 1 | L6 계승 |
| `LUMINA_A210_INDEPENDENT_CAPTURE` | 0 | 1 | L6 계승 |
| ★`LUMINA_A210_LINE_SATURATION_TARGET_ION` | 3 | **1** | **이 단의 유일 신규 delta** |

- diagnostic_mode 파일 = `A210_L6C_PROBE`. 그 외 전부 봉인 계승(`FIXED_TE_PROFILE`
  seed_uniform_10020, `LINE_SATURATION_DIAG=2`, sigma sha `90d04042…` 등). **vs 봉인 L6 런 =
  env 값 delta 정확 1 + 모드 파일.**
- 바이너리 = 봉인 L6 `input/lumina_cuda` 재사용(sha 고정, §2-5). 덱·`/gpfs` 정본 불변.
- 제출: slurm, partition **`a100` 한정**, `--gres=gpu:2`, `--mem` 명시(봉인 L6 job.slurm 과
  동일 인자), job-name `LUMINA_l6c_cover`, run root 에 OWNER.txt(project·목적·`DO_NOT_CANCEL`).
  **syn101 수동 제출 전면 금지. grammar 용 `--exclude` 부착 금지**(별개 클러스터).
- 소형 오프라인(PC-1~5·판독) = grammar-debug(nested ssh). 로그인 노드 연산 금지.

## 8. 이 단이 모르는 것 (추측으로 메우지 않는다)

1. **ion=1 선 진동수에서의 J̄/B 이탈 크기** — σ 라벨이 받는다. WEAK-SIGNAL 이면 J̄-응답
   판정은 미결로 남고 후속 갈림길(표적/shell 확장) 입력이 된다.
2. **d 의 절대 규모** — ≥1e-9 는 [추정](solve residual 근거). 분기 문턱(1e-13)과 4자릿수
   이상 떨어져 있어 분기 자체는 강건하다.
3. **iter1 의 SPROBE 위험**(NLTE 산출의 χ==0) — 오프라인 예측 불능, D2.
4. **시드가 커버 이온에서 정확 LTE(10020)인가** — R4 앵커가 실측한다.
5. **행 수·후보 수·교집합 크기·총가중** — 관측만(§2-4).
6. **다준위 전 정량의 옳음** — 오라클 없이 판정 불능(§0-3). C-R 은 "도달+응답+관측 정합"까지다.
7. **W5 공통 오프셋의 원인**(§2-6) — 불접촉, 대장 존속.
8. **PC-2 기전 모델([추정] 축퇴 상쇄)의 완전성** — 앵커 2종으로만 검증; 앵커 빗나감은 이름
   있는 발견.

## 9. 이 단이 하지 않는 것

- src/tests/env_universe 접촉 일체(V5 원상 유지 — 권한 요청 0). 빌드(바이너리 재사용).
- Z-O/Z-1 수리(존속, user 보류 — PC-2 실측은 그 사전등록의 입력으로 병기만).
- STAGE4/EW/SUPER 승격 · ion=2/3 재선정 · shell 0 밖 확장 · 핀 온도 변경.
- L6·A207 판정의 재판정(둘 다 폐합 유효). W5 오프셋 귀속. 수렴 주장(snapshot checker
  `--tail-transitions 0` 유지). CMFGEN 정량 인용(오라클 INELIGIBLE — B_ν 는 물리상수 함수).
- 봉인 런루트 쓰기 접촉 일체.

## 10. 분장 장부 (집행 후 운전석이 "실제"·"위반"을 채운다 — 규약상 담당만 적는 것 금지)

| 단계 | 규약상 담당 (개정14) | **실제** | 위반 |
|---|---|---|---|
| 표적 결정 | user (2026-08-22) | | |
| 형상·사전등록(본 문서) | Fable | | |
| 발주(앞겹=본 문서 원문 첨부, 재서술 금지) | 운전석 | | |
| 코딩(§4 변경집합) | Codex | | |
| 코드 검수(고정질문에 *"발주서가 사전등록의 범위를 좁혔는가"* 포함) | Fable | | |
| 오프라인 게이트 PC-1~5·스테이징·제출 | 운전석 | | |
| 판독기 실행·계측 패킷 | 운전석 | | |
| 판정(판정문 저작) | Fable (fresh) | | |
| 판정 감리 | Fable (★판정과 **다른** fresh 컨텍스트·고정질문 4) | | |
| 감리 반영·대장·커밋 | 운전석 | | |

## 11. 판정 절차

- 판정 = Fable **fresh 컨텍스트**(본 사전등록 + 봉인 산출물 경로 + 도구 산출 제공). 판정 하중
  항목은 판정자가 직접 재실측 — 특히 **f_mapped·d 분포·ff·R7 byte 대조·PC-2 앵커·SKIP-마스크
  교집합(PC-1 — 이 단에서 결함 이력이 있는 판정식)**은 판정자
  자체 경로(도구 import 없이)로 재현할 것(L6·A207 전례).
- 감리 = **또 다른 fresh Fable**(자기 채점 금지), 고정질문 4. 감리 고정질문에 ① §2-3 의 권한
  게이트 3중화가 실제로 집행됐는지 ② C-F/C-R 함의가 후보/실증 절단을 지켰는지 ③ §8-8 앵커
  처분의 타당성을 포함하기를 권고한다.
- 폐합 전 감리 필수. 판정문은 §6 분기 중 발화분을 축자 기재, 분기 밖 결과는 폐합 금지·미결.
- clamp/floor/cap 0 · 오라클 인용 0 · 봉인 무변조 실측(전후 지문)을 판정문이 기재한다.

## 12. 기계 프리플라이트 선언 — 계약이 스스로를 검사한다

`scripts/check_prereg_preflight.py` 가 발주 **전에** 검사한다(fail-closed). PF-1 은 §4 표와
표지 심볼 `A210_L6C_PROBE` 의 양방향 1:1 을(신설 2파일·확장 2파일 전부가 심볼을 가진다 —
발주 시점(구현 전)은 expected_extra 가 계획 경로를 공급하고, 도구 커밋 후에는 grep 실물이
같은 집합을 낸다 ⟹ 모든 커밋 경계에서 green), PF-2 는 ff 1차원 분할(이 단의 분기 판별량이
실제로 하나다 — D/W-V 는 A207 전례와 같은 분할 밖 이름 있는 차단 계급)을, PF-3 은 §3 인용
앵커의 실존을 강제한다. **PF-3 의 정직한 한계**: R7 의 byte-동일 의미 검사와 PC-2 산법의
옳음은 정적 검사 밖이다 — 각각 판정런 게이트와 앵커 2종이 잡는다. 과대 주장하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### scripts/ — 집행 2파일 확장 + 신설 2파일",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/[a-z0-9_\\-]+\\.(?:py|sh|slurm)",
    "symbol": "A210_L6C_PROBE",
    "roots": ["scripts"],
    "expected_extra": ["scripts/run_det_convergence_2026-08-08.slurm",
                       "scripts/stage_det_stage12_l6_probe.sh",
                       "scripts/analyze_det_stage12_l6c.py",
                       "scripts/audit_l6c_cover_precondition.py"]
  },
  "branches": {
    "regimes": [[1e-16, 1e-14], [1e-10, 1e-4], [0.001, 2.0]],
    "metrics": {
      "ff": "sum(1 for x in v if x <= 1e-13)/len(v)"
    },
    "rules": [
      {"name": "C-R", "predicate": "ff <= 0.10"},
      {"name": "C-F", "predicate": "ff >= 0.99"}
    ],
    "adversarial_fixtures": [
      {"name": "sealed-l6-frozen-class", "mix": [[6.7e-16, 297]]},
      {"name": "solver-response-class", "mix": [[1e-9, 297]]},
      {"name": "partial-freeze", "mix": [[6.7e-16, 150], [1e-9, 147]]},
      {"name": "exact-ten-percent-boundary", "mix": [[6.7e-16, 30], [1e-9, 270]]}
    ],
    "residual": "C-M"
  },
  "references": [
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["a210_line_saturation_target",
                        "LUMINA_A210_LINE_SATURATION_TARGET_ION",
                        "NLTE_TARGET_ION",
                        "nlte_build_projection",
                        "LUMINA_NLTE_SKIP_Z",
                        "INDEPENDENT_SPROBE_UNDEFINED"]},
    {"path": "scripts/run_det_convergence_2026-08-08.slurm",
     "flags_existing": ["A210_L6_PROBE"]},
    {"path": "scripts/stage_det_stage12_l6_probe.sh",
     "flags_existing": ["resolved_env_delta_count"]},
    {"path": "scripts/check_a210_targeted_gate.py",
     "flags_existing": ["--expected-outer-iterations"]},
    {"path": "scripts/analyze_det_stage12_l6.py", "flags_existing": []},
    {"path": "docs/VERDICT_DET_SPRIM_L6_2026-08-22.md",
     "flags_existing": ["0.607"]},
    {"path": "docs/VERDICT_DET_A207_WIRING_2026-08-22.md",
     "flags_existing": ["ION4"]},
    {"path": "docs/VERDICT_DET_SPROD_S4_III_2026-08-19.md",
     "flags_existing": ["INDEPENDENT_SPROBE_UNDEFINED"]}
  ]
}
```
