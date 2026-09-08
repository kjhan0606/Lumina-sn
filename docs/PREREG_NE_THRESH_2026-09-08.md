<!-- 저작: astra(gpt-6-astra) 기안 (개정21 계획 담당). 운전석은 파일화·프리플라이트·커밋만 수행하고 본문을 고치지 않았다.
     발주 = /gpfs/kjhan/lumina/dproxy_audits/nethresh/ORDER_ASTRA_PLAN_NETHRESH.md · 원문 = /gpfs/kjhan/lumina/dproxy_audits/nethresh/ASTRA_PLAN_NETHRESH.md -->
# 사전등록 기안 — Saha 층 n_e 수렴 문턱 (DET-NE-THRESH, 2026-09-08)

**런 0회로 답할 수 있는 것 / 런이 필요한 것:** 봉인 소스의 정지 기전·대수, 보존된 상태가 충분한 경우의 전하 사상 F와 F′, 문턱별 재생·독립 근 대조·OFF 함수 배터리는 오프라인으로 확인할 수 있다. 실제 수송과 결합한 새 상태의 수렴, 공개 population·스펙트럼 변화, 전체 OFF 증분 동일성은 런이 필요하다. 기존 로그의 n_e·전하합만으로 F′를 복원할 수는 없다. 필요한 고정 상태가 없으면 `INPUT_PENDING`이며, 자료획득런을 오프라인으로 세거나 판정런으로 기전 탐색을 대신하지 않는다.

저자 = **astra(본 응답)**. 상태 = **기안·미집행**. 이번 작업은 조회·해시 확인만 수행했다. 파일 생성·코드 변경·계산런·프리플라이트 실행 없음. 문턱의 실제 선택과 기대치 봉인은 아래 오프라인 절차 뒤 완료한다.

## 1. 계약 — 물리 계약 1개

**계약:** 현재 원자모형·복사장·열역학 상태가 정하는 Saha/rate-SE 전하중성 방정식 `n_e = Fθ(n_e) = Σ q·n_q(n_e;θ)`의 **반복 정지 문턱만** 수리한다. 「지금은 CMFGEN 발자국 그대로」를 적용하여 원자자료·율·이온 사다리·전하 규약·NLTE 재분배·온도·수송 격자를 유지한다. CMFGEN 일치 자체는 이 계약의 판정 대상이 아니다.

현행 `x → m=(x+Fθ(x))/2`, 채택 전 `abs(Fθ(x)−x)/x < 0.05`, 최대 100회를 유지하되, ON에서는 `0.05`만 사전 봉인한 `τ*`로 교체한다. 새 키 `LUMINA_SAHA_NE_THRESH` 미설정·`0`은 기존 경로이며, ON은 양의 유한 `τ*<0.05`; 그 밖 설정은 명시적 실패다. `LUMINA_NLTE_NE_CLOSURE=1`은 R-ACQ·OFF·ON 공통이다. 계측 전용 키 `LUMINA_SAHA_NE_TRACE`는 미설정·문자열 `0`이면 비활성, 문자열 `1`이면 기록·격리 종단 탐침만 활성화하며 문턱·감쇠·100회·실패 전파를 바꾸지 않는다. 빈 문자열·그 밖 값은 실행 전 명시적 실패다. 계측 활성 조건은 TRACE=1 또는 유효한 THRESH ON이며, 둘 다 켜면 중복 기록·탐침 없이 THRESH의 문턱을 적용한다. 각 키의 잘못된 값은 다른 키로 무시하지 않는다. R-ACQ는 THRESH 미설정·TRACE=1, OFF는 두 키 미설정, ON은 THRESH=τ*·TRACE 미설정으로 고정한다.

100회 미수렴은 기존 `NE_FAIL` 및 후보 거부로 전파한다. 실패값 발행·마지막 값 성공 처리·자동 한도 증가·다른 해법으로의 fallback은 허용하지 않는다. 정확해를 배제하는 clamp/floor를 추가하지 않는다. 문턱과 반복 한도는 계산의 성공·실패를 정할 뿐 물리해의 허용 범위를 정하지 않는다.

**잔차 구분:** `δ=(Fθ(x)−x)/x`, `r_old=abs(Fθ(x)−m)/m`, `r_m=ln(Fθ(m)/m)`를 별도 기록한다. `m≠x`이면 구간 할선 `s=(Fθ(m)−Fθ(x))/(m−x)`에 대해 **정확히** `r_m=log1p[δ(1+s)/(2+δ)]`; `δ=0`이면 같은 결정론 사상에서 `r_m=0`이다. `δ(1+F′θ(x))/2`는 선형근사이며, 측정한 점 미분도 구간 전체의 엄밀 상한은 아니다.

## 2. 소스 상태 — 봉인 identity 기준 확인

기준 런 `<BASE>` = `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on`(**333531**). 기준 소스 `<SRC>` = `/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth`.
`<BASE>/input/current_source_identity.json`의 source_files **5/5**와 `<SRC>` 파일 해시 일치를 본 저자가 확인했다. git head는 `364ee72ba17dbd944bacc4a7c3420693500b9c88`; HEAD만을 소스 identity로 쓰지 않는다.
봉인 패치 `<BASE>/input/current_source.patch` SHA256 = `916eb7bb0a5ef41a876b13f59f71245a2cc49490d0f40fc94f1070a72e0ab9a9`.
`<SRC>/src/lumina_plasma.c` SHA256 = `90fd73cc51cae34649533fc8ed87f84a39e08521b4fdd6fb8a57b107b959210d`.
`<BASE>/input/lumina_cuda`와 `<SRC>/lumina_cuda` SHA256 = `ca721b07cb4fef632b1a0a0b2d3c7b11cdaac0e95f74e6cc56b131a7c5449059`.
이 identity의 `:3048–3144`에서 감쇠·문턱·100회·실패 경로를 재확인했다. 이온 분율은 `compute_ion_populations_shell(:2742)`, rate-SE 비는 `parity_rate_se_ratio_checked(:2689)`이며 bootstrap 분기는 별도다. 후보는 `:21052`의 n_e 반복 뒤 `:21063`에서 이온 분율을 재평가한다.
새 작업본은 이 봉인 상태의 별도 사본에서 만든다. 현재 로컬 미커밋 변경은 섞지 않는다. `<EVID>`는 운전석이 새로 지정할 절대 증거 루트이며, `<OFF>`·`<ON>`도 새 절대 런루트로 실행 전에 고정한다.

## 3. 런 설계 — R-ACQ 자료획득 1회 → 오프라인 자격 검증·봉인 → OFF 대조 1회 → ON 판정 1회

**R-ACQ 자료획득·G-ACQ:** 운전석이 새 절대 런루트 `<ACQ>`=`/gpfs/kjhan/lumina/dproxy_audits/nethresh/racq_rev5/run`, 새 증거루트 `<ACQ-EVID>`=`/gpfs/kjhan/lumina/dproxy_audits/nethresh/racq_rev5/evidence`를 사용하며 기존 경로가 있으면 덮어쓰지 않고 중단한다. 333531의 물리 exports·입력 digest·K42·ion1·T_e 핀·2반복·50셸을 계승하고 경로 치환 외 exports 차이는 TRACE=1뿐이다. R-1의 두 호출처·전 solve·경로 식별자·17자리 궤적·추가 궤적 평가 0·종단 탐침 의무를 동일 적용하고 산출은 `<ACQ-EVID>` 아래에 둔다. G-C의 staging 원문·종료코드·전후 seal 검증·shared_hardlinks=0과 G-D의 완주·오류 0·publication 2·registry 338·provenance·150행 조건을 R-ACQ에도 요구한다. **G-ACQ 물리 동일성**은 동일 봉인 판독기로 BASE와 ACQ에서 추출한 `<ACQ-EVID>/baseline/` 및 `acq/`의 `epochs.txt` 3행, `pairs.csv` 338짝 키와 네 수치열, `closure.txt` 150행, `spectrum.csv` 파장·flux가 각각 byte-동일해야 한다. 새 계측 행은 이 물리 산출 비교에 포함하지 않되 원문은 보존한다. 모든 종단 탐침마다 production 배열·캐시·BF 카운터·bootstrap 전역 및 탐침이 접근하는 나머지 가변 상태를 주소·padding을 제외한 명시적 필드 목록으로 봉인하고, 탐침 직전과 복원 직후 전 필드 byte 비교 결과·해시를 `acq/probe_restore.csv`에 남긴다. 종단 탐침 목록과 비교 목록은 키 전건 일치·결손/중복 0이어야 하며, 출력 동일성만으로 복원을 인정하지 않는다. 상태 누출·물리값 1 byte 변조·복원 행 누락 음성대조가 각각 실패해야 한다. 결손은 INPUT_PENDING, 불일치는 REWORK로 보존하고 O-1/O-2·OFF·ON을 중단한다. R-ACQ에는 NT-A/NT-R을 부여하지 않는다. 이는 기존 0.05 궤적의 자격 표적·고정 상태 획득이며, 그 뒤 기존 O-2 선택 규칙으로 τ*를 정하고 ON 전에 봉인하므로 ON 판정의 순환 사용이 아니다. ON을 본 뒤 표적·τ*·기대치를 바꾸면 NT-X이며 자동 재런은 없다.

**O-0 입력 충분성:** `<EVID>/offline/state_manifest.json`에 bootstrap 및 SEED/R7 관련 전 셸의 상태 출처·시점·해시를 등록한다. θ는 T_e·밀도·조성·원자자료·분배함수·BF/복사장 generation·NT 입력·활성 env·분기 상태 등 F의 모든 의존성을 포함한다. 기존 입력으로 정확히 재구성 가능한 상태와 보존되지 않은 상태를 구분하며, 누락을 LTE 대체나 로그 맞춤 계수로 메우지 않는다.

**O-1 기전 특정·F′ 측정:** O-0 원본은 변경하지 않는다. 입력 14/14 보존과 생산 F의 CPU 호출 경로 부재를 구분한다. 독립 고정밀 사다리는 부트스트랩의 네뷸라 Saha `phi_neb`와 부트스트랩 종료 후 rate-SE를 모두 구현하고, 종료 후 `!r1_use`는 생산과 동일하게 즉시 실패시킨다. 자격 검증 표적은 rate-SE에 한하여 봉인 장부의 SEED gen2·R7 gen3·R7 gen4 각각 shell 0–49, 총 150쌍이며, 같은 θ에서 전건 `abs(F_oracle(m)−F_record(m))/abs(F_record(m))≤1e-10`을 요구한다. **`phi_neb` 자격 표적은 G-ACQ PASS 산출에서 획득한다.** `<ACQ>/stderr.log`의 `[A2-07][NE-TRAJ]` 전건을 `<ACQ-EVID>/acq/trajectory.csv`로 추출하고 `solve_index.csv`·`solve.csv`·`terminal.csv`와 대조한다. 키는 `(call, 호출처, shell, epoch, θ해시, bootstrap, r1_use, 실제경로, 반복번호, 점종류)`이며 bootstrap 활성·r1_use=0·실제경로=phi_neb인 모든 solve의 모든 `(x_k,F(x_k))`와 종단 `(m,F(m))`을 `phi_neb_targets.csv`에 등록한다. 반복 수에서 기대 목록을 생성하여 누락·중복·여분 0을 요구하고 로그 진단 횟수나 8,100회를 solve 수로 대신하지 않는다. θ 전 의존성·초기값·원문 행 위치·실제 절대경로·SHA256·전건 키 목록을 `state_manifest.json`에 봉인하며 기존 O-0 원본은 보존한다. 이 보충 manifest와 원본의 합집합을 이후 오프라인 상태 모집단으로 사용한다. 각 표적의 동일 θ·n_e에서 전건 `abs(F_oracle−F_record)/abs(F_record)≤1e-10`을 요구하며 생산값≤0·비유한·경로 혼합·불일치는 REWORK, 상태·표적 결손은 INPUT_PENDING이다. 실제 파일·해시·전건 목록을 후속 계약 봉인으로 확정하고 두 경로 자격 검증을 통과하기 전에는 F′ 측정·O-2를 열지 않는다. 기존 bootstrap 최대 residual 한 줄과 rate-SE 150쌍은 이 표적을 대신하지 않는다. 이후 F′는 경로별 고정 θ의 x·m·n*에서 각각 측정하며 기존 차분·안정성 기준을 전부 적용한다. 입력·장부·구현 해시, 상태 대응, 원수치·오차를 `<EVID>/offline/oracle_validation.csv`에 보존한다. 표적 맞춤 계수는 금지한다. 결손은 INPUT_PENDING, 불일치는 REWORK이며 통과 전 O-1/O-2 계산을 진행하지 않는다. 통과 후 θ를 고정하고 n_e만 바꾼다. 매 평가 전 배열·캐시·카운터·bootstrap 상태를 복원하거나 격리 프로세스를 사용한다. 평가 순서를 뒤집어 F가 동일한지도 확인한다.
각 상태의 기존 마지막 x·채택점 m·독립 근 n*에서 수행한다. x가 직접 보존되지 않았으면 봉인된 초기값·θ·생산 연산 순서·0.05 문턱·100회 한도로 기존 반복을 재생하고, 채택점 m도 상대오차 1e-10 이내로 재현해야 x를 사용한다. 초기값·상태 대응 결손은 INPUT_PENDING, 재현 불일치는 REWORK다. 검증된 세 점 각각에서 `n±=n·exp(±h)`, `D_h=[F(n+)−F(n−)]/(n+−n−)`를 계산한다. `h={1e-2,1e-3,1e-4,1e-5,1e-6}` 전부와 좌·우 차분을 `<EVID>/offline/gain.csv`에 보존한다.
인접 h 두 쌍 이상에서 차분 간 차이가 `1e-4·max(1,abs(D_h))` 이내인 구간을 요구한다. 불안정·분기 전환은 `GAIN_UNRESOLVED`; F′의 부호나 `−1<F′≤0`을 가정하지 않는다. `(1+F′)/2`는 국소 감쇠 이득 측정치로만 보고한다. O-3 전에는 검증된 오프라인 F의 반복 궤적으로 할선 항등식을 확인한다. **R-1 뒤에는 G-E에 따라 ON 전 반복점·종단점의 생산/독립 F 값과 동일 구간 할선을 비교한다.** 할선 일치를 점 미분 일치나 미관측점의 보증으로 해석하지 않는다. 동일 호출·셸·θ의 연속점에서 `x_(k+1)=m_k`를 확인한 뒤 `s_k=(F(x_(k+1))−F(x_k))/(x_(k+1)−x_k)`를 오프라인 계산한다. 동일점은 할선을 정의하지 않고 §1의 δ=0 조건과 부동소수점 정체를 구분한다. 할선을 점 미분으로 대체하지 않으며 기존 h·좌우 차분·안정성 기준은 유지한다.

**O-2 수리안 검증:** O-1의 **두 경로 자격 검증을 모두 통과한** 오프라인 F로, O-0에 등록한 두 경로의 전 고정 상태에서 `τ∈{5e-3,5e-4,5e-5,5e-6,5e-7,5e-8}`를 전부 재생한다. 한 경로의 표적·상태 결손도 `INPUT_PENDING`이며 다른 경로의 통과로 대체하지 않는다. τ 재생은 생산 double 연산 순서·감쇠·정지 판정을 재현하고, 독립 근은 별도 고정밀 계산 경로로 구한다. 150쌍 일치는 표본 검증이며 미분·전역 동일성·bootstrap 및 다른 θ의 검증을 대신하지 않는다. 해당 상태의 의존성·분기 대응이 입증되지 않으면 제외하지 않고 INPUT_PENDING으로 중단한다. 별도 고정밀 구현이 동일 율·정규화 사다리로 F를 계산하고, 부호가 확인된 구간에서 전하중성 근 n*를 구한다. 구간은 수치 근 찾기용이며 물리적 cap이 아니다. 근 미확보·비유일성 미해결은 실패다.
수치 분해능 잣대 `ε=1e-6`은 기존 분류 문턱 `1e-3`보다 세 자릿수 작게 택한 **설계 기준이며 예측·물리 정확도 보증이 아니다**. **두 경로의** 전 재생 상태 모두에서 `abs(r_m)≤ε`, `abs(ln(m/n*))≤ε`, 100회 이내 성공을 만족하는 **가장 느슨한 공통 τ 하나를 τ***로 택한다. 경로별 τ는 허용하지 않는다. O-3의 상태별 F′·잔차·근 차이·반복 수에는 경로를 명시하고 경로별 최대값과 전체 최대값을 함께 봉인한다. E-2는 ON의 두 경로 전 solve에 같은 ε를 적용하며 전체 최대값으로 판정한다. 후보가 없으면 해법·한도를 바꾸지 않고 `REWORK`다.

**O-3 기대치 봉인:** `<EVID>/offline/expectations.json`에 τ*·상태별 F′ 안정 구간·잔차·근 차이·반복 수·최대값의 셸/시점·입력/도구 해시를 기록하고 계약 개정으로 커밋한다. F′와 실제 궤적을 측정하기 전에는 τ*나 잔차 감소 배수를 확정하지 않는다. **사전 봉인 대상** = O-0~O-2 상태·오프라인 F′·재생 궤적·τ*·기대치·구현 해시와 **G-E 비교 규칙·허용오차**이며 ON 자료를 요구하지 않는다. ON 의 새 θ·생산 궤적·종단 탐침·독립 계산과 비교 결과는 **사후 검증 대상**으로 별도 보존하고 사전 봉인을 갱신하지 않는다. 자료 부족이면 `INPUT_PENDING`이다. R-ACQ 착수에는 §2 identity·획득 입력/경로/저장량 봉인·계측 구현 코드 검수·해당 직접 selftest와 `<EVID>/design_audit_rev5.md`·`<EVID>/preflight_rev5.log`의 R-ACQ 범위 PASS를 요구한다. 미획득 표적의 자격 검증·τ*·기대치만 R-ACQ 선행조건에서 제외하며 G-0 전체 PASS로 표기하지 않는다. OFF·ON에는 G-ACQ PASS와 기존 G-0~G-E의 해당 선행조건, 두 경로 자격 검증·O-3 봉인을 모두 요구한다. G-0의 현행 개정 감사 파일은 위 rev5 두 파일이며, 획득 전 설계/구현 해시와 획득 후 표적·τ*·기대치 봉인 해시에 대한 단계별 PASS를 별도 보존한다. 이 지정은 개정 4 및 과거 이력의 GPU 발주 금지·감사 버전 지정만 대체하고 이전 PASS를 승계하지 않는다. 기존 감사·preflight·O-0·부록·차단 기록은 원본 보존한다. G-ACQ 미충족은 최종 분기의 g_fail에도 포함하며 판정 게이트·기대치·허용오차는 완화하지 않는다.

**R-0 OFF:** 새 바이너리·새 키 미설정, 기준은 **333531의 폐합 ON 상태**다. G-A 통과 후에만 R-1을 연다. 새 키 `0`의 동등성은 별도 프로세스 오프라인 배터리로 검증한다.
**R-1 ON:** OFF와 같은 바이너리·물리 exports에서 새 키 `τ*`만 추가한다. K42·ion1·T_e 핀·2반복·50셸을 계승한다. ON의 **두 호출처**(`compute_plasma_state:7302`·후보 파이프라인 `:21196`, O-0 부록 기준)를 모두 계측하여 모든 n_e solve에 호출 ID·**호출처**·셸·θ identity·**bootstrap 상태**·**`r1_use`**·**실제 사다리 경로**·x·F(x)·m·F(m)·반복 수·상태를 기록한다. 호출 수는 기존 진단 인쇄로 추정하지 않고 ON의 `NE-TRAJ` call 카운터로 확정한다. 이 식별자를 solve_index·solve·trajectory·종단 탐침에 보존하고, 공개 commit이 있으면 연결하며 **bootstrap solve도 누락하지 않는다**. ON과 R-ACQ에서 매 반복의 호출 ID·셸·θ identity·반복 번호·기존 x·F(x)·m을 round-trip 가능한 17자리로 기록하여 각각 `<EVID>/on/trajectory.csv`와 `<ACQ-EVID>/acq/trajectory.csv`로 추출한다. **궤적 기록을 위한 추가 F 평가·수치 부동소수 연산은 0**이며, 할선·잔차 계산은 판독기가 수행한다. OFF 미설정·0에서는 기록 경로에 진입하지 않는다. 마지막 채택점은 다음 반복이 없으므로 F(m)를 낡은 F(x)나 연속점 할선으로 대신하지 않는다. 동일 θ의 기존 후보 재평가가 있으면 연결하고, 없는 solve는 GPU 실행 중 확보한 사설 상태에서 생산 F(m)를 평가한다. 배열·캐시·**BF 카운터·bootstrap 전역**을 포함한 격리를 검증하며 production 상태를 바꾸지 않는다. 이는 궤적 기록의 추가 평가 0과 별개인 기존 탐침 의무다.
**비용·보존:** 오프라인은 grammar-debug 단일 CPU, 총 2 CPU-hour·16 GiB·신규 증거 10 GiB 상한; 초과 예상 시 중단 보고한다. R-ACQ·OFF·ON 각 GPU 런은 A100×2·32 CPU·64 GiB·8시간이며, 총 3런 합계 최대 48 GPU-hour이다. 자동 재런은 없고 오프라인 총 2 CPU-hour·16 GiB·신규 증거 10 GiB 상한은 유지한다. 신규 저장량은 기준 런 실측 크기와 계측 예상량을 합쳐 staging 전에 등록한다. 기존 봉인·산출물은 보존하며 rollback은 새 패치 비채택이다.

## 4. 기대치 — 구성상 도출과 측정의 구분

- **E-1 / 구성상 도출:** 성공한 ON solve의 채택 전 잔차 `<τ*`, 감쇠식 및 할선 항등식 성립. 작동 검사이며 물리 예측으로 세지 않는다. 기존 NLTE 폐합의 같은 합 재계산 residual도 이 부류다.
- **E-2 / 측정 기반 검증가설:** O-3에서 봉인한 수리안이 실제 ON의 전 solve에서도 채택점 잔차와 독립 근 차이를 각각 `≤ε`로 유지하는가. 코드는 이 두 조건으로 정지하지 않으므로 자동 보장이 아니다. 실패는 기대 불일치로 남기며 사후 τ 조정은 금지한다.
- **E-3 / 측정, 크기·부호 기대 없음:** 공개 commit의 Saha 잔차·NLTE 재분배·합계, 각 항의 우세 셸 수. solve와 commit 사이 θ가 달라졌으면 별도 상태로 판독한다.
- **E-4 / 측정, 크기·부호 기대 없음:** ON/OFF population 비와 formal 스펙트럼 대역별 중앙·최소·최대, 전역 최대 절댓값. 직전 +0.783%·−2.325%·0.4274%를 다음 응답의 상한으로 사용하지 않는다.

### 게이트 명부 (표)

| 게이트 | 판정 및 피연산자 | 실패 처분 |
|---|---|---|
| G-0 사전 준비 | `<EVID>/identity.json`이 §2 원본과 일치; `<EVID>/offline/state_manifest.json` 의존성 결손 0; `<EVID>/preflight.log`·`<EVID>/design_audit.md`에 PASS | 미집행/PENDING |
| G-1 오프라인 | `<EVID>/offline/gain.csv` 미해결 0; `<EVID>/offline/sweep.csv`와 `<EVID>/offline/oracle.csv`로 O-2 충족; `<EVID>/offline/expectations.json`이 선택 규칙·해시와 일치 | REWORK 또는 INPUT_PENDING |
| G-2 음성대조·격리 | `<EVID>/selftest/results.json`에서 성한 fixture PASS; 낡은 F(x)를 F(m)로 오기·전하종 누락·θ 변경·탐침 상태 누출·NaN·100회 실패 은폐·OFF 오진입·장부 결손 주입 각각 지정 실패; **기존 rate-SE 150점과 생산 항등식을 유지한 채 ON 중간점 Q만 변조한 fixture**·P/Q 경로 키 교환/누락 fixture·bootstrap solve 누락 fixture·부트스트랩 종료 후 `!r1_use`를 성공 처리한 fixture·`phi_neb` 자격 표적 결손에도 오프라인 계산을 진행한 fixture·할선만 허용오차를 초과한 fixture·재생 반복 수 불일치는 NT-X, G 전건 정상인 E-2 초과는 NT-R; 값 오차 `1e-10`·할선 오차 `1e-4·max(1,abs(s_P))`·E-2 `1e-6` 각각 **경계 동일값은 통과하고 바로 위 표현 가능값은 지정 실패**; `<EVID>/selftest/off_bytes.json`에서 기존/미설정/0 출력 배열 byte-동일 | 코드 REWORK |
| G-A OFF 증분 | `<EVID>/baseline/epochs.txt` 대 `<EVID>/off/epochs.txt`의 epoch 3행 문자열 동일; 두 디렉터리 `pairs.csv`의 338짝 키와 n_upper_0·n_upper_1·Jbar_0·Jbar_1 문자열 동일; `closure.txt`의 150행 문자열 동일; `spectrum.csv`의 파장·flux 문자열 동일. 각각 `<BASE>`·`<OFF>`에서 동일 봉인 판독기로 추출; `<OFF>/stderr.log` 새 문턱 계측 0행 | ON 발주 금지 |
| G-B 단일 변수·봉인 | `<OFF>/input/resolved_lumina.exports` 대 `<ON>/input/resolved_lumina.exports`는 `<EVID>/path_map.json`에 열거한 경로 치환 후 새 키 τ* 추가만 허용; 양팔 `input/binary.sha256` 동일; 양팔 `SUPERLEVEL_B_STAGE_SEAL.txt`의 입력 digest는 `<BASE>/SUPERLEVEL_B_STAGE_SEAL.txt`와 동일 | 무효 |
| G-C staging 증거 | `<EVID>/staging/OFF.stdout.log`·`ON.stdout.log` 원문과 종료코드 보존; §2 기준 seal의 sealed_base 아래 `RII_RUN_SEAL.sha`, sealed_census 아래 `ZO_RUN_SEAL.sha`를 해당 루트에서 `sha256sum -c`하여 staging 전·완료 후 전건 OK; 양팔 seal shared_hardlinks=0 | 무효 |
| G-D 완주·커버리지 | 양팔 `model.rc`=0; `<EVID>/off/readout.json`·`on/readout.json` 오류 0·publication 2·registry 338 포함·provenance 전건 OK; `<EVID>/on/solve_index.csv`와 `solve.csv` 호출/셸 키 완전 일치; `<EVID>/on/closure.csv`는 **공개 commit 3개(SEED gen2 · R7 gen3 · R7 gen4) 각각 shell 0–49 완비 = 150행** | 무효 |
| G-E 계측 진실성 | `solve.csv` 로 E-1 재계산; `trajectory.csv` 는 solve_index 의 전 호출·셸·반복을 결손·중복 0 으로 포함하고 동일 θ 의 연속점·할선 항등식을 충족한다. **생산값 P** = 모든 반복점의 17자리 `F(x_k)` 및 격리 검증된 종단 `F(m)`, **독립값 Q** = 두 경로 자격 검증을 마친 사전 봉인 오프라인 구현을 생산 P와 정확히 같은 θ·n_e·**bootstrap 상태·`r1_use`·실제 사다리 경로**에 평가한 값. P/Q의 호출 ID·호출처·셸·epoch·상태 해시·**경로 키**를 전건 대조하며 결손·불일치·**경로 혼합은 NT-X**다. 기존 연속점·종단 구간 할선·독립 double 재생·독립 근에도 같은 경로 대응을 요구하고 **경로 경계를 가로지르는 할선은 허용하지 않는다**. 경로별 비교 전건과 전체 판정을 보존하며 기존 값·할선·재생·E-2 허용오차는 유지한다. 전건 `abs(Q−P)/abs(P) ≤ 1e-10` 을 요구하며 `P≤0`·누락·비유한·θ/epoch 혼합은 실패. 서로 다른 연속점과 마지막 `x→m` 구간 각각에서 `s_P=(P_b−P_a)/(b−a)`·`s_Q=(Q_b−Q_a)/(b−a)` 를 계산하고 전건 `abs(s_Q−s_P) ≤ 1e-4·max(1,abs(s_P))` 를 요구한다; 동일점은 할선을 생략하되 δ=0 과 부동소수점 정체를 구분한다. 같은 θ·초기값·봉인 τ* 로 **독립 double 재생**하여 생산과 반복 수·성공/실패 상태가 같고 전 `x_k`·`m` 의 상대오차 ≤1e-10 이어야 한다. 원수치·점/구간 키·오차·전건 판정을 `<EVID>/on/replay.csv` 에 보존한다. E-2 는 생산 종단값의 `abs(ln(P(m)/m))` 와 `<EVID>/on/oracle.csv` 의 독립 근에 대한 `abs(ln(m/n*))` 로 계산한다. | 비교·재생·증거 조건 하나라도 실패/미확보면 **NT-X**; G 전건 PASS 에서만 E-2 최대값 하나라도 >1e-6 이면 **NT-R**, 둘 다 ≤1e-6 이면 **NT-A**; 사후 τ 조정·자동 재런 금지 |

### 게이트 명부 끝

### 분기 (표)

| 분기 | 조건 | 뜻 |
|---|---|---|
| NT-X | G-0~G-E 중 실패·미충족 존재 | 무효 또는 미집행; INPUT_PENDING·AUDIT_PENDING·구현/집행/증거 실패를 명명 |
| NT-R | G 전건 PASS, `<EVID>/on/metrics.json`의 E-2 잔차 또는 근 차이 최대값 `>ε` | 오프라인 기대가 실제 상태로 전이되지 않음; 원인 분류 후 후속 결정 |
| NT-A | G 전건 PASS, 같은 경로의 두 최대값 모두 `≤ε` | 봉인 범위에서 문턱 수리의 수치 목적 충족; E-3·E-4는 별도 측정 보고 |

### 분기 끝

기계 분기 입력 `<EVID>/branch_tokens.json`은 각 게이트 FAIL/미충족→0, PASS→1, E-2 개별 측정 충족→1·초과→2로 변환한다. 결손은 0이며 빈 입력은 NT-X다. 변환과 실제 임계값 경계의 음성대조는 판독기 selftest가 담당하고, 아래 JSON은 분기의 배타성·완전성을 검사한다.

### 변경집합 — SAHA_NE_THRESH (표)

| 경로 | 변경 |
|---|---|
| `src/lumina_plasma.c` | THRESH 문턱 선택과 TRACE 계측 활성화를 분리; ON/R-ACQ 사설 계측·종단 탐침·전건 복원 증거; OFF 기존 연산·로그 유지; 감쇠·100회·물리 사상 불변. 같은 변경집합의 `src/env_universe.h`·`validation/instrumentation_debt/ENV_UNIVERSE.json`에 두 키를 생성기로 등록하고, `tests/saha_ne_thresh_selftest.c`에 키 조합·잘못된 값·G-ACQ 누출/결손 음성대조를 추가한다. `scripts/saha_ne_thresh_replay.py`는 R-ACQ 표적 추출·보충 manifest·G-ACQ byte 비교·분기 입력 및 `--manifest <ACQ-EVID>/state_manifest.json --acquire <BASE> <ACQ>` 재현 인터페이스를, `scripts/stage_superlevel_b.sh`는 R-ACQ 모드·TRACE 단독 차이·세 런 동일 binary identity 봉인·획득 staging 원문 보존을 추가한다. 기존 각 행의 의무는 유지하며 추가 파일이 없어 선언 블록 `expected_extra`는 그대로 둔다. |
| `src/env_universe.h` | 전집 생성기로 새 키 등록 |
| `validation/instrumentation_debt/ENV_UNIVERSE.json` | 같은 전집 재생성 산출 |
| `tests/saha_ne_thresh_selftest.c` | 신설: OFF 별도 프로세스 byte 배터리·합성 F의 알려진 미분/근·실패 전파·탐침 격리 검사 |
| `scripts/saha_ne_thresh_replay.py` | 신설: **경로별** 상태 검증·rate-SE 150쌍 및 **별도 봉인 `phi_neb` 표적**의 oracle 자격 검증·**표적 결손 INPUT_PENDING**·두 경로 생산 double 반복 재현·경로별 F′ 차분 분석·두 경로 독립 고정밀 사다리/근·**공통 τ** 선택·P/Q 경로 대응 및 혼합 거부·**ON 전 반복점/종단점 생산·독립 F 및 구간 할선 비교**·전 궤적 재생 대조·생산 종단 잔차 산출·G-E 전건 증거·분기 변환 및 G-2 음성대조 |
| `scripts/stage_superlevel_b.sh` | 새 OFF/ON 모드·바이너리/계약 identity·새 키 단일 차이·staging stdout 원문 보존; 봉인 내부 목적지 거부 유지 |

### 변경집합 끝

## 5. 주장 상한

NT-A는 **이 원자모형과 봉인된 실행 범위에서 정지 문턱의 수치 잔차를 제어했다**는 뜻이다. 전하 방정식의 물리적 완전성·CMFGEN 일치·수송/열평형 수렴·스펙트럼 오차 상한을 뜻하지 않는다. F′ 표본은 전역 수렴 증명이 아니며, 독립 근도 유일성이 확인된 범위에 한정한다. 기존 CPU χ/η·bf·열원장과 폐합 n_e의 시대 차이는 계승 미결이다.

## 6. 분장 장부

| 단계 | 규약상 담당 — 본 발주·개정21 | 실제 | 위반·독립성 |
|---|---|---|---|
| 사전등록 기안 | astra | **astra, 본 응답 저자** | 조회·identity 확인만 완료 |
| 코딩·직접 selftest | **운전석**(개정22) | 미집행 | 구현자 = 운전석 |
| 코드 검수 | **astra**(개정22) | 미집행 | 구현자와 **다른 계열** ⟹ 이 축의 자기 검수는 소멸 |
| 판정·마감감사 | astra | 미집행; 이후 본 계약의 판정·마감감사도 astra 담당 | 자기 계획 판정·감사이며 독립 모델 감사로 표기 금지 |
| 파일화·프리플라이트·빌드·staging·제출·판독·커밋 | 운전석 | 미집행 | 원문 로그와 재현 명령 보존 |
| 실측 주장 전건 재현·불일치 장부 | 운전석 | 미집행 | 동일 source/binary/input/evidence 해시와 결정적 출력으로 reproduced/not-reproduced/blocked 기록 |

재현 명령 인터페이스는 신설 판독기의 `--manifest <EVID>/offline/state_manifest.json --offline`, `--manifest <EVID>/identity.json --compare <BASE> <OFF> <ON>`, `--selftest`로 고정한다. 운전석은 실제 절대경로·실행 위치·도구 해시·명령·종료코드를 `<EVID>/reproduce.log`에 보존한다. 아직 실행하지 않은 검수·판정·감사·재현의 PASS를 이 기안에 기재하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — SAHA_NE_THRESH (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests|scripts|validation)/[A-Za-z0-9_/.]+\\.(?:c|h|py|sh|json)",
    "symbol": "SAHA_NE_THRESH",
    "roots": ["src", "tests", "scripts", "validation"],
    "expected_extra": ["src/lumina_plasma.c", "src/env_universe.h", "validation/instrumentation_debt/ENV_UNIVERSE.json", "tests/saha_ne_thresh_selftest.c", "scripts/saha_ne_thresh_replay.py", "scripts/stage_superlevel_b.sh"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    "metrics": {"g_fail": "sum(1 for x in v if x < 0.5)", "miss": "sum(1 for x in v if x > 1.5)", "count": "len(v)"},
    "rules": [
      {"name": "NT-X", "predicate": "count == 0 or g_fail > 0"},
      {"name": "NT-R", "predicate": "count > 0 and g_fail == 0 and miss > 0"},
      {"name": "NT-A", "predicate": "count > 0 and g_fail == 0 and miss == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "all-pass", "mix": [[1.0, 10]]},
      {"name": "expectation-miss", "mix": [[1.0, 9], [2.0, 1]]},
      {"name": "invalid-before-miss", "mix": [[0.0, 1], [2.0, 1]]},
      {"name": "missing-input", "mix": [[0.0, 1]]},
      {"name": "empty", "mix": []}
    ],
    "residual": "NT-X"
  },
  "references": [
    {"path": "src/lumina_plasma.c", "flags_existing": ["compute_electron_density", "compute_ion_populations_shell", "parity_rate_se_ratio_checked", "NE_FAIL"]},
    {"path": "scripts/derive_env_universe.py", "flags_existing": ["getenv"]},
    {"path": "scripts/check_gate_referents.py", "flags_existing": ["OPERAND_VAGUE"]},
    {"path": "docs/PREREG_DC1_NE_CLOSURE_2026-09-06.md"},
    {"path": "docs/VERDICT_DC1_NECLOSE_R2_2026-09-08.md"},
    {"path": "docs/VERIFICATION_REGISTERS.md"},
    {"path": "docs/CLASSIC_DEBT_CENSUS.md"},
    {"path": "CLAUDE.md"}
  ]
}
```

## 개정 1 (2026-09-08, astra 사전감사 1회차 반영 — 기재 4, 차단 0)
- G-D 문면 「SEED·R7 두 commit」 → **공개 commit 3개(SEED gen2·R7 gen3·R7 gen4) 150행**(직전 단 실측).
- 감사가 남긴 기재 셋은 계약 문면 변경 없이 여기 기록한다: ① 이 감사는 **자기 계획 자기 감사**이므로 독립 감사 PASS 로 읽지 않는다(개정21 계열 분리 소멸) ② `<EVID>` 는 현재 `identity.json`·`preflight.log`·`design_audit.md` 만 있고 state manifest·치환표·판독기 검증은 미완 — 계약이 `INPUT_PENDING` 으로 이미 처리 ③ **비용 상한(오프라인 2 CPU-hour)은 중단 기준이지 완료 가능성의 측정이 아니다** — 생산 F 가 준위별 BF 적분(24,542준위·1234빈)을 반복하므로 O-1/O-2 의 실제 소요는 미시연. 초과 시 중단 보고.
- ★감사가 확인한 대수(계약 §1 의 할선 항등식): `m−x = xδ/2`, `F(m) = x(1+δ) + sxδ/2` ⟹ `F(m)/m = 1 + δ(1+s)/(2+δ)`. 기존 로그 잔차(`compute_electron_density` 의 출력)는 `|δ|/(2+δ)` 로 **F(m) 재평가 잔차와 다른 양**이다.
- ★감사가 지목한 격리 위험(구현 시 대상): 후보 배열 복사만으로는 **BF 카운터·bootstrap 전역**이 격리되지 않는다 — O-1 의 「복원 또는 격리 프로세스」는 이 둘을 포함해야 한다.
행 단위 diff 로 의도 외 소실 0 확인 [저자 실행].


## 개정 2 (2026-09-08, O-0 실측 반영 — astra 기안, OLD/NEW 10쌍)
**경위**: 운전석이 O-0 을 실행해 (a) F 의 입력 **14/14 보존**(덱·J_nu 50셸×1234빈·γ 침적·env 전집) (b) 검증 표적 **(m,F(m)) 150쌍** (c) **생산 F 의 CPU 호출 경로 부재**(모델 초기화 하네스가 변경집합에 없음)를 확인했다. 증거 = `<EVID>/offline/state_manifest.json`(원본 보존, 개정으로 고치지 않음).
**선택한 경로**: 하네스를 새로 만들지 않고 **O-2 의 독립 고정밀 사다리를 150쌍으로 먼저 자격 검증**(≤1e-10)한 뒤 그것을 오프라인 F 로 쓴다. F′ 는 그 F 로 오프라인 측정하고, **ON 생산 궤적의 할선**으로 교차검증한다. 궤적 기록은 이미 계산되는 값의 기록이므로 **추가 F 평가 0**이며 OFF 는 진입하지 않는다(G-A 「OFF 새 계측 0행」 유지).
**승계 금지**(⚠개정 3 이 대체 — 현행 G-0 추가 필수조건은 `design_audit_rev3`·`preflight_rev3` 의 PASS **뿐**이다): 기존 사전감사·preflight PASS 를 개정 2 의 실행 경로에 승계하지 않는다 — `design_audit_rev2.md`·`preflight_rev2.log` 를 G-0 에서 함께 요구한다.
**운전석 정정 1건**: 기안 #10 이 코드 검수를 운전석으로 적었으나 **개정22 는 코드 검수 = astra** 다. 코딩·직접 selftest = 운전석, 코드 검수 = astra 로 적용했다(자기 코드 자기 검수 소멸).
행 단위 diff 로 의도 외 소실 0 확인 [저자 실행].


## 개정 3 (2026-09-08, 개정 2 사전감사 «폐합 차단» 1 해소 — astra 기안, OLD/NEW 5쌍)
**차단 내용**: 「ON 생산 궤적으로 교차검증」이라 적었으나 계약이 지정한 검사(G-E 궤적 항등식)는 **생산 값끼리** 구성되므로 독립 사다리가 틀려도 성립한다 ⟹ 표본 밖에서 잘못된 오프라인 F′·τ* 가 통과할 여지.
**해소**: G-E 가 **생산 P 대 독립 Q 를 모든 반복점·종단점에서** `≤1e-10`, **구간 할선**을 `≤1e-4·max(1,|s_P|)`, **독립 double 재생**의 반복 수·상태·궤적 일치까지 요구하고, 실패 처분(NT-X)과 기대 불일치(NT-R)를 분리한다. 사전 봉인 대상(O-0~O-2·τ*·비교 규칙)과 사후 검증 대상(ON 자료)을 나눠 순환을 없앴다. G-2 음성대조에 「생산 항등식은 유지한 채 Q 만 변조」 반례와 경계 동일값·바로 위 값 검사를 넣었다.
**승인 범위**: 개정 3 사전감사(`<EVID>/design_audit_rev3.md`)·preflight(`preflight_rev3.log`) PASS 가 G-0 의 추가 필수조건. 개정 2 감사의 **차단 기록 보존**, 이전 PASS 승계 금지.
행 단위 diff 로 의도 외 소실 0 확인 [저자 실행].


## 개정 3 정정 (2026-09-08, 운전석 적용 오류 — 이탈 기재)
개정 3 기안(astra)의 NEW 는 「기존 증거는 **보존**하고 `design_audit_rev3.md`·`preflight_rev3.log` 의 PASS 를 G-0 추가 필수조건으로 요구, 개정 2 감사의 **차단 기록은 보존**」이었다. 운전석이 이를 옮겨 적으면서 **rev2 의 PASS 까지 필수조건으로 넣었고**, 개정 2 감사의 판정이 `통과 불가|차단 1`(보존 대상)이므로 **G-0 이 원리적으로 닫히지 않는 문면**이 됐다. 개정 3 사전감사가 이를 «폐합 차단»으로 잡았다(정당).
**처분**: 기안 원문대로 복원(위 문단). 이것은 계약 설계 변경이 아니라 **운전석의 옮겨 적기 오류 정정**이며, 이탈은 지우지 않고 여기 남긴다 — 개정15 가 지목한 「뒷겹 재서술은 표류한다」와 같은 계급이 **개정 적용**에서 재발한 것이다.


## 개정 4 (2026-09-08, O-0 부록 실측 반영 — astra 기안, OLD/NEW 8쌍)
**경위**: 운전석 계측(`docs/O0_SOLVE_PATH_SURVEY_NETHRESH_2026-09-08.md`)이 사다리의 **두 경로**(부트스트랩 `phi_neb` / 이후 rate-SE)와 두 호출처를 드러냈고, 폐합 장부 150쌍이 **rate-SE 표본뿐**임을 보였다.
**결과**: 오프라인 사다리는 두 경로를 모두 구현하고, 자격 검증도 경로별로 받는다. **`phi_neb` 경로의 표적이 봉인 산출물에 없으므로 이 단은 현재 `INPUT_PENDING`** 이며, 표적을 계약 개정으로 지정해 같은 1e-10 을 통과하기 전에는 τ* 선택·기대치 봉인·GPU 발주가 열리지 않는다. 계측은 두 호출처·bootstrap 상태·`r1_use`·경로 키를 전부 남기고, G-E 는 **경로 혼합을 NT-X 로** 거부한다.
행 단위 diff 로 의도 외 소실 0 확인 [저자 실행]. 기안 NEW 는 **축자 복사**했다(개정 3 정정의 재발 방지).


## 개정 5 (2026-09-08, 자료획득런 등록 — astra 기안, OLD/NEW 8쌍, user 09-08 「1번」 선택)
`phi_neb` 자격 표적을 얻기 위해 **R-ACQ 자료획득런 1회**를 정식 등록했다. 계측 전용 키 **`LUMINA_SAHA_NE_TRACE`**(문턱 불변, 기록·종단 탐침만), 세 런의 키 조합 고정(R-ACQ = TRACE=1 / OFF = 둘 다 미설정 / ON = THRESH=τ*), **G-ACQ 물리 동일성**(epochs 3행·338짝 4열·closure 150행·spectrum 을 base 와 byte-동일; 새 계측 행은 비교에서 제외하되 원문 보존)과 **탐침 복원 증거**(필드 목록 byte 비교를 `probe_restore.csv` 에, 출력 동일성만으로 복원 인정 금지)를 게이트로 두었다. R-ACQ 에는 분기 라벨을 주지 않으며, τ* 는 그 산출로 정해 **ON 전에 봉인**하므로 순환이 아니다. GPU 예산 3런·48 GPU-hour.
행 단위 diff 로 의도 외 소실 0 확인 [저자 실행]. 기안 NEW **축자 복사**.
