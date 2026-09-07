# 판정 — DET-DC1-NECLOSE (D-C-1: NLTE 커밋 뒤 전자 밀도 폐합, OFF 333303 · ON 333531) — Fable fresh, 2026-09-08

저자 = Fable fresh(판정 전용 컨텍스트, 운전석 아님). 잣대 = `docs/PREREG_DC1_NE_CLOSURE_2026-09-06.md` @ HEAD(`376dd03`) **개정 1·2 + 집행 기록, 단일**. 증거 = 판독 `docs/READOUT_DC1_333303_333531_2026-09-07.md`(사실만 인용), 런루트 원본, `docs/VERDICT_DUAL_AUTH_2_R2_2026-09-06.md`(E-2b 도출 입력), `docs/ASTRA_CODE_REVIEW_DC1_R2_2026-09-06.md`. 인용 문서 5건 `check_doc_canonical` rc=0 [저자 실행, grammar-debug]. 전 수치는 본 저자가 **독립 재도출**했다(스크립트·산출 = `/gpfs/kjhan/lumina/dproxy_audits/dc1/verdict_probe/`: `dc1_verdict_probe.py`·`stage3.py`·`run_readouts.sh`·`out_off/`·`out_on/`·`probe_stage*.txt`·`seal_*.log`). 런루트·repo 무접촉.

## 0. 착지 사실 (재현)
- `sacct -j 333303,333531`: `LUMINA_DC1OFF`/`LUMINA_DC1ON`, a100 syn102, `cpu=32,gres/gpu=2,mem=64G`, Elapsed 02:58:54 / 02:58:58, 둘 다 `FAILED 70:0`. `model.rc` = 0·0. Slurm 70 의 실체 = post-model `DPROXY_PROVENANCE_RUN_SEAL_FAIL reason=unsealed complete marker`(ON stderr 2304행) — base 331050 도 같은 계급(2204행) ⟹ 「seal 오판」 주장 확인. `fatal_scan.log` 0행·`[A2-07][n_e][FATAL]` 0·`UNRESOLVED_CANCELLATION|RADEQ_SIGN_MISMATCH` 0 ⟹ NC-X 사유 없음.
- 증거 무접촉: 런루트에서 `model.rc` 보다 새 파일 = post-model 산출 3건뿐(mtime = 잡 End 23:37:57). ON `stderr.log` sha256 `0911fcf83a9b91ad…bf1f14c` = 판독 인용값. 판독 산출 23:40 → 커밋 `376dd03` 23:41:30.
- `[DPROXY][SL-POP-COMPLETE] status=BLOCKED reason=INVALID_COMMITTED_VIEW` 는 base·OFF·ON 셋 모두 동일(선재 계급, readout 은 ROW 경로로 판독) — 이 단의 변수가 아니다.

## 1. 게이트 (사전등록 문면 그대로, 전건 저자 재실행)
| 게이트 | 판정 | 재현값 (독립 도출) |
|---|---|---|
| G-A | **PASS** | exact epoch 3행 base·OFF 문자열 동일(`45/9.6662782724980344e-09 · 53/7.8976358675842993e-09 · 53/8.5618986192583131e-09`); readout 을 **재생성**(`out_off/`, 운전석 산출과 `cmp` 동일) → base `out_331050/pairs.csv` 대 `n_upper_0,n_upper_1,Jbar_0,Jbar_1` **338/338 문자열 동일**. 대조: ON 은 338/338 **전건 상이**(4열 모두) |
| G-B | **PASS** | sorted diff = 런루트 경로 치환 7쌍(`CMFGEN_SIGMA_BF·DEPOSITION_FILE·IONIZATION_REFERENCE_FILE·MODEL_DIR·PHYSICS_COMPARISON_DIR·TOPION_LEVELS_FILE`) + `71a72 > declare -x LUMINA_NLTE_NE_CLOSURE="1"` **1행**, OFF 에 키 부재 |
| G-C | **PASS** | 두 루트 `SUPERLEVEL_B_STAGE_SEAL.txt` digest 4종 동일(levels `b3645052…`·line_list `cbd81c4f…`·registry `c96f74f0…`·binary `ca721b07…`, `RUN_FOOTER binary_sha256` 동일). `RUN_SEAL_PASS` 2행은 **스테이징 stdout 이 어디에도 보존돼 있지 않아** 원문 재현 불가 → 대체 재검: 봉인 루트 두 곳에서 `sha256sum -c` 오늘 재실행 = R-II **135/135 OK**·census **136/136 OK**(rc 0) — 현재 시점 무결이지 스테이징 시점 로그가 아님(§6 #3) |
| G-D | **PASS** | rc 0·0; publication 2(경계 ON 480/2317, OFF 430/2217); `provenance_status=OK` **685/685** 두 런; pub0 347행 ⊇ registry 338, pub1 338, 공통 짝 **338 ≥ 330**; readout validation 0 |
| G-E | **PASS** | ON `[A2-07][NE-CLOSURE]` **150행** = SEED gen 2 (50) + R7 gen 3 (50) + R7 gen 4 (50), commit 마다 shell 0–49 완비; FATAL **0**; OFF **0행**; `residual=` 최대 **0.000e+00**; `dlog_total−dlog_saha_residual−dlog_resplit` 최대 **2.43e-16**; 인쇄된 `n_e_saha·q_saha·q_closed` 로 세 로그를 재계산해 차이 **0** |
| G-F | **PASS** | 같은 하준위 공유 선 간 Δlog n_l 산포 최대 ON **1.21e-9**·OFF **1.21e-9** ≤ 1e-8 (중앙 7.8e-11) |

## 2. 폐합 장부 분해 (ON, 150행 — E-3 측정)
| commit | `dlog_total` 중앙 [min,max] | `dlog_saha_residual` 중앙 / max\|·\| | `dlog_resplit` 중앙 / max\|·\| | \|total\|>1e-3 셸 |
|---|---|---|---|---|
| SEED gen 2 | +6.77e-3 [−2.33e-2, +7.83e-3] | **+6.76e-3 / 2.33e-2** | −1.8e-7 / 1.40e-4 (shell 10) | **49/50** (shell 33 만 ≤) |
| R7 gen 3 | +2.51e-4 [−1.21e-5, +1.50e-3] | +2.8e-5 / **1.50e-3** | +3.2e-5 / 4.50e-4 (shell 13) | **6** = shell 44–49 |
| R7 gen 4 | +8.7e-5 [−5.06e-5, +2.84e-4] | −8.6e-6 / 2.19e-4 | +1.6e-5 / 4.90e-4 | **0** |
- gen 3 물질 셸 6개의 항별 값: shell 44 total +1.0425e-3 = saha +1.0436e-3 + resplit **−1.1e-6** … shell 49 +1.5020e-3 = +1.5025e-3 + (−5.1e-7). 6셸 전부 \|saha\|/\|total\| = **1.00**, resplit 최대 1.11e-6. **resplit 단독으로 1e-3 을 넘는 셸은 150행 중 0**, saha 단독으로 넘는 셸 = 49(SEED)+6(gen 3).
- SEED 부호: shell 0–32 는 + (n_e 상향, shell 0 +0.783%), shell 33–49 는 − (최대 shell 40 **−2.33%**, n_e 2.8e5) — Si/S/Ca 지배 외곽에서 Saha 층이 n_e 를 2% 과대 유지하고 있었다. 되먹임 뒤 gen 3 외곽 잔차 +1.0–1.5e-3, gen 4 ≤2.2e-4 — 폐합이 Saha 층의 반복을 한 걸음 대신 밟는 꼴.
- 구성상 상한 재검: `q_saha` 는 `compute_electron_density` **뒤에** 채택 n_e 로 `compute_ion_populations` 를 다시 돈 값(worktree `:21052–21063`)이라 `dlog_saha_residual = ln(F(n_e)/n_e)` = Saha 전하 사상의 1회 적용 잔차. 5% 문턱(`:3105`)이 근사-선형에서 주는 상한 ≈ +0.024/−0.026; 실측 최대 2.33e-2 — **문턱 바로 밑**. E-2a 의 0.05 는 참이되 느슨한 상한이다.

## 3. 기대치 대조
- **E-1** residual ≤1e-12: 150/150(최대 0) ✔(구성상). **E-2a** \|saha\| ≤0.05: 150/150 ✔, 최대 2.33e-2.
- **E-2b** shell 0 \|resplit\|: SEED 4.88e-6 · gen 3 **5.94e-7** · gen 4 2.88e-6 — 전건 ≤1e-4 ✔, NC-N 아님. 도출 대조: 도출이 쓴 09-06 비(R7 iter 0 상태)에 대응하는 gen 3 에서 ΔQ = 5.94e-7×3.224e9 = **+1.9e3 cm⁻³** — 도출값 +1.5e3 과 부호·자릿수 일치(30%). SEED 는 +1.57e4(10×, 다른 population 상태) — 「≲1e-5」 상한 안이지만 점추정과는 다르다.
- **E-1b**(측정) SEED resplit 중앙 −1.8e-7·최대 1.40e-4. **E-4**(재실행) iter 0 `n(ON)/n(OFF)` 바닥 Fe **1.003734**·Co **1.003785**·Ni **1.003751**, 준위 산포 2.6e-4/2.7e-5/9.2e-5; iter 1 1.001727/1.001829/1.001694. 세 이온·전 준위가 같은 비(+0.37%, +0.18%). 바닥 Δlog iter0→1 ON Fe +3.42%·Co +0.30%·Ni +3.89%(OFF +3.62/+0.49/+4.09).
- **E-5**(`spectrum_contrast.py` selftest PASS 후 재실행, JSON 운전석 산출과 `cmp` 동일; 1234 bin·격자 bit-동일·비양수 0) 중앙: [0,2) **+0.13%** · [2,4) **+0.24%** · [4,6) **+0.28%** · [6,12) **+0.12%** · [12,∞) **−0.01%**(최소 −0.43%). 전 대역 중앙 <0.3%.

## 4. 분기 (개정 2 분기표 문면)
G-A~G-F 전건 PASS, E-2a·E-2b 일치 ⟹ NC-X·NC-N 아님. NC-A 문면 「전 셸·**전 R7 commit** \|dlog_total\| ≤ 1e-3」: R7 gen 3 shell 44–49 가 **1.04–1.50e-3 > 1e-3** ⟹ NC-A **불성립**(SEED 를 세지 않아도). NC-B 「어느 셸·commit 이든 > 1e-3」 ⟹ **성립**. 경계 검사: 최소 초과 shell 44 = 1.0425e-3, 문턱 대비 +4.25%, 값은 결정론 17자릿수 — 모호할 여지 없음. 선언 블록 규칙(`g_fail==0 and broken==0 and material>0 → NC-B`)과 일치.
★**분기 = NC-B**. 단 문면이 요구하는 「두 성분 보고」의 답: 물질 셸을 만든 항은 **전량 Saha 잔차**(6셸 모두 비율 1.00)이고 **NLTE 재분배 항은 150행 어디서도 ≤4.9e-4, shell 0 ≤4.9e-6** 이다. NC-B 의 뜻 「폐합이 어딘가에서 물리량」은 **Saha 층의 자기 잔차를 폐합이 걷어낸다는 의미로만** 충족된다 — 이 계약의 신규 내용(NLTE 분배를 전하 보존이 보게 함)은 이 시대에서 **수치적으로 불활성**(NC-A 계급 크기)이다. 라벨은 문면대로 NC-B 이나, 「NLTE 재분배가 n_e 에 물리량」으로 읽으면 안 된다.

## 5. 등급 · 주장 상한 (§5)
| 주장 | 등급 | 상한 |
|---|---|---|
| 폐합 스위치 in-vivo 작동(OFF = base bit-동일, ON 은 gen-2 epoch 부터 상이 + 150행 + 전 준위 공통 비) | **C** | 독립 경로 2(epoch 문자열·population 실측) |
| **지배 효과 = Saha 층 수렴 여유**(seed shell 0 +0.78%, 외곽 −2.3%; gen 3 외곽 +1.0–1.5e-3) | **C**(존재·자릿수) / **B**(셸별 값) | 경로 2 = NE-CLOSURE 장부 + 선재 `[A2-07][n_e] charge-conservation residual max=1.586e-02`(bootstrap, base·OFF·ON 동일 stdout 175/176행) — 같은 상태는 아니나 같은 계급(1e-2)의 독립 증인. 셸별 값은 장부 1경로 |
| NLTE 재분배 항 shell 0 ≤4.9e-6(gen 3 +5.9e-7 ⟺ ΔQ +1.9e3 ≈ 도출 +1.5e3) | **C** | 장부 + 09-06 비 도출. 바깥 셸(≤4.9e-4)은 **B**(장부 1경로) |
| E-4 population ×1.0037(iter 0)·×1.0018(iter 1), 이온·준위 공통 | **B** | S-역산 1경로; 산포 ≤2.6e-4. 「n_e +0.78% 의 전파」는 단일 변수 대조에 의한 귀속이지 기전 시연이 아님(비 0.48×Δln n_e 는 관측) |
| E-5 스펙트럼 중앙 <0.3% | **B** | 두 상태의 차이 크기이지 오차 크기가 아님 |
**말하지 않는 것**: 「전하 보존이 옳게 닫혔다」 — Saha 층 5% 문턱과 Saha 단계(I·IV+·quarantine)의 LTE 가중 Γ_phot 이 그대로다; CMFGEN 대조 없음; 반복 2회·T_e 핀·K42·shell 0 Fe/Co/Ni II 에 한정. 폐합 n_e 와 같은 commit 의 CPU χ/η·bf 는 폐합 전 값(개정 2 §1 두 시대 공존) — 이 런은 그 크기를 E-4·E-5 로 잰 것이지 고친 것이 아니다.
**다음 계약 후보(사전등록이 지목한 5% 문턱)에 대한 함의**: 확립된 것은 「D-C-1 폐합 효과의 >99.9%(물질 셸 기준)가 Saha 층 자기 잔차」라는 **귀속**(C)이며, 그 잔차의 크기 상한이 이 런으로 **측정**됐다 — seed n_e ±2.3%(외곽 −, 내부 +0.8%), 되먹임 1회 뒤 ≤1.5e-3, 2회 뒤 ≤2.2e-4; population ≤0.4%, 스펙트럼 ≤0.3%. 5% 문턱을 조이는 계약의 **사전등록 기대치는 이 숫자를 넘을 수 없다**(이 시대·이 관측량에서). 「5% 문턱이 물리적으로 틀렸다」는 이 런에서 **나오지 않는다** — 잔차의 부호·크기가 셸마다 다르다는 사실만 나온다.

## 6. 운전석 이탈 (계급 병기)
| # | 이탈 | 주인 | 계급 |
|---|---|---|---|
| 1 | NC-A 문면은 「전 R7 commit」, NC-B 는 「어느 commit 이든」 — SEED 의 지위가 두 행에서 비대칭. 이번 결과에선 라벨 불변(gen 3 만으로 NC-B) | 사전등록 저자 | «기재로 족함» |
| 2 | E-2a 상한 0.05 는 근사-선형 구성상 상한(≈0.025)의 2배 — 참이되 느슨; 「측정」 결과 2.33e-2 가 실은 문턱 포화라는 정보를 문면이 못 담음 | 사전등록 저자 | «기재로 족함» |
| 3 | G-C 「스테이징 stdout 의 RUN_SEAL_PASS 2행」의 원문이 런루트·dc1 감사 디렉토리 어디에도 보존되지 않음(집행 기록의 「census 136·R-II 135 PASS」는 전해 들은 값). 봉인 루트 재검(135/136 OK)으로 실체는 확인 | 운전석(집행) | «기재로 족함» — 다음 단부터 스테이징 stdout 을 런루트 곁에 보존 |
| 4 | 판독 §2 소제목 「E-1 (ON, 바닥 준위 Δlog n iter0→1)」 — 이 사전등록의 E-1 은 residual 게이트이고 바닥 Δlog 는 등록된 기대치가 아님(DUAL-AUTH 의 E-1 을 옮겨 씀) | 판독 저자(운전석) | «기재로 족함» |
| 5 | 판독 「E-2b ✔ (도출값 ≲1e-5 와 일치)」 — SEED 4.9e-6 은 상한 안이나 점추정(+1.5e3 ⟹ 4.7e-7)의 10×; 「일치」는 gen 3(5.9e-7)에서만 참 | 판독 저자 | «기재로 족함» |
| 6 | 판독 E-3 「그 셸의 resplit ≤4.5e-4」 — 4.5e-4 는 gen 3 전 셸 최대(shell 13)이고 shell 44–49 는 ≤1.1e-6(과소 진술, 방향은 안전) | 판독 저자 | «기재로 족함» |
| 7 | 두 런 Slurm `FAILED 70`(post-model seal 오판, base 와 동일 계급) — 판독은 사실대로 기재 | 선재(seal 스크립트) | «기재로 족함» |
«폐합 차단» 계급: **0건**. 발주·단일 변수·봉인·자원·판독 수치: 이탈 없음(판독의 모든 수치가 독립 재도출과 일치).

## 7. 분장 장부
장부 = **사전등록 §6 지목**(복제하지 않음). 본 판정 칸: 판정 = Fable fresh(본 문서), 운전석 아님. 마감감사(astra)·감리 = 미발주 — 폐합은 두 겹 모두 «폐합 차단» 0 이어야 한다(개정18 ③).

VERDICT: NC-B (R7 gen 3 shell 44–49 |dlog_total| 1.04–1.50e-3 > 1e-3, 전량 Saha 잔차 항; NLTE 재분배 항은 150행 ≤4.9e-4·shell 0 ≤4.9e-6 = NC-A 계급; G-A~G-F 전건 PASS, E-2a·E-2b 일치) | 폐합 가능 — 마감감사(astra) + 감리 두 겹 모두 «폐합 차단» 0 을 조건으로 | 다음 행동: 폐합 효과 귀속이 Saha 층 5% 문턱이므로 사전등록이 지목한 그 계약 후보를 대장에 기재하고 사전등록 여부는 user 결정 — 기대치 상한은 본 §5 측정값(seed n_e ±2.3%·population ≤0.4%·스펙트럼 ≤0.3%)을 넘을 수 없다
