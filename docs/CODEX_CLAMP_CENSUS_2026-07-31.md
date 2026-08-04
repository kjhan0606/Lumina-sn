## 조사 결론

정적 재조사 결과, 현재 트리에는 논리 메커니즘 기준 **94개**의 floor/cap/damping/대체가 있습니다. 동일 로직의 CPU/GPU 미러와 반복 생성된 스크립트는 한 행으로 합쳤고 모든 발생 파일을 병기했습니다.

- 조사 범위: `src` 17개 파일, 39,630행 + `scripts`의 `.py/.sh` 751개
- 원시 검색: source `1e-30`류 148행, `fmin/fmax` 11행; scripts `np.clip/maximum/minimum` 108행, `min/max` 633행
- 빌드·프로그램·수치 스크립트·git 실행 없음
- 기존 88대장은 비교에만 사용했으며 현재 줄번호와 로직을 다시 판독함
- `docs/CODEX_TUNING_CLAMP_CONSULT.md`의 핵심 3문을 적용:
  1. 정확해가 경계를 위반할 수 없는가?
  2. 최종 고정점을 보존하는가?
  3. 쌍대항·보존법칙을 함께 유지하는가?
- `조건부`는 정적 코드만으로 위 3문 중 하나 이상을 증명할 수 없거나, 충분한 반복·정제·오차상계가 전제인 경우입니다.

### 총계

| 유형 | 정당 | 조건부 | 위험 | 합계 |
|---|---:|---:|---:|---:|
| floor | 5 | 9 | 9 | 23 |
| cap | 5 | 8 | 13 | 26 |
| damping | 0 | 5 | 0 | 5 |
| 대체·fallback | 1 | 16 | 23 | 40 |
| **합계** | **11** | **38** | **45** | **94** |

`Ctr`: Y=발화 카운터 존재, P=부분·제한 카운터, N=없음.

## 전수표 — 현재 C/CUDA/header

| ID | file:line | 대상 물리량 | 유형 | 게이트 | 3문 분류 | Ctr | 기존 88 대응 |
|---|---|---|---|---|---|---|---|
| C01 | `lumina_main.c:58-61`; `lumina_cuda.cu:3446-49,4551-65,4782,4872-4910,5196-99,5599,5654,5677,5805`; `lumina_plasma.c:7015-18` | RNG `log(0)` | floor | 없음 | 정당 | N | 기존 A48/H21 |
| C02 | `lumina_cmfgen.c:37-41`; `lumina_nlte_assemble.cu:115`; `lumina_plasma.c:1025,1037,2687,4116,5787,8199,8265,8427,8431,16845-51` | Planck/지수 underflow | cap | 없음 | 정당 | N | 기존 A18/A37/C12/F4 |
| C03 | `lumina_plasma.c:2680-81,10171-72`; `lumina_cmfgen.c:121,132,150,161` | escape probability 점근식 | 대체 | 없음 | 정당 | N | 기존 A39/H22 |
| C04 | `lumina_plasma.c:488-495,7921-22,14767-74,14858-59`; `lumina_radeq_col_pairs.h:67-70` | Ω(T), ζ 등 표 범위 밖 끝값 | 대체 | 경로별 | 조건부 | N | 기존 H23/H25 |
| C05 | `lumina_cuda.cu:5047,5091,5127`; `lumina_plasma.c:16878` | 확률·sqrt 정의역 | floor | 없음 | 정당 | N | 기존 H20/H25 |
| C06 | `lumina_cmfgen.c:368,377,608,633,923,1038,1046,1135-37,1220-39,1872,1880` | χ, η, Δτ 비음수화 | floor | 없음 | 조건부 | N | 기존 C9/M6 |
| C07 | `lumina_cmfgen.c:1950,2145-46`; `lumina_cuda.cu:6383,6402,8425` | ε·확률 `[0,1]` | cap | env | 정당 | N | 기존 C3/C4 |
| C08 | `lumina_cmfgen.c:177-234` | `eps_floor=1e-5`, 미등록 ε→1 | 대체 | `LINE_EPS_PHYS` | 위험 | N | 기존 C4 |
| C09 | `lumina_cmfgen.c:227,1277-84,2359,2398`; `lumina_plasma.c:10944,17143-91` | 미해결 선원함수→`B`/`WB`/0 | 대체 | 경로별 | 위험 | P | 기존 F1/C1/C2, FORMAL_FIX에서 변경 |
| C10 | `lumina_cmfgen.c:269-518` | EPAY 열방출 재척도·τ 게이트 | 대체 | `LUMINA_CMF_EPAY*` | 위험 | P | 변경 C7, `TAUEFF` 추가 |
| C11 | `lumina_cmfgen.c:739-797`; `lumina_cmf_solve.cu:220` | ALI 분모·음수/NaN J | floor | ALI 경로 | 조건부 | N | 기존 C8/C9/P5 |
| C12 | `lumina_cmfgen.c:1121,1197`; `lumina.h:144-151` | 적분 substep·고정 작업공간 | cap | 없음/env | 조건부 | N | 기존 P6/P7 |
| C13 | `lumina_cuda.cu:1543-52`; `lumina_plasma.c:16045-68` | 음수 NLTE 인구→`1e-30` | 대체 | 기본 경로 | 위험 | N | 기존 A1 |
| C14 | `lumina_cuda.cu:1473-1543`; `lumina_plasma.c:15996-16068` | LTE-relative repair/floor | 대체 | `LTE_FLOOR/LTE_REPAIR` | 위험 | N | 변경 A3 |
| C15 | `lumina_cuda.cu:987-1002,1510-24`; `lumina_plasma.c:16000-41` | FLOORM LTE floor | floor | `FLOOR_MODE=1` | 위험 | N | 기존 A4 |
| C16 | `lumina_cuda.cu:1460-70,1560-75` | `b_k` ceiling | cap | `BK_CEIL` | 위험 | N | 기존 A5 |
| C17 | `lumina_cuda.cu:1270-1323`; `lumina_plasma.c:15913-16134` | INV/grey/residual 실패→Boltzmann | 대체 | 복수 env | 조건부 | P | 기존 A2/A6/H7 |
| C18 | `lumina_cuda.cu:1116`; `lumina_plasma.c:15920` | 희박 ion pair skip→fallback | 대체 | `SKIP_DEAD` | 조건부 | N | 기존 A7 |
| C19 | `lumina_cuda.cu:1424-35`; `lumina_plasma.c:15776-84` | BK_PARTIAL 참조 인구 | floor | `BK_PARTIAL` | 조건부 | N | 기존 A10 |
| C20 | `lumina_cuda.cu:1798`; `lumina_plasma.c:2524-2606,16307`; `lumina_cuda.cu:9541` | Sobolev τ zero sentinel | floor | 없음 | 조건부 | P | 기존 A12/M6 |
| C21 | `lumina_cuda.cu:1794`; `lumina_plasma.c:2600,4400,16302` | inversion/maser 흡수율→0 | floor | 경로별 | 조건부 | N | 기존 A13/H12 |
| C22 | `lumina_cuda.cu:1807`; `lumina_plasma.c:16318` | `S_l` 분모 컷→0→소비자 fallback | 대체 | 없음 | 위험 | N | 기존 A14 |
| C23 | `lumina_bf_gemm.cu:83-93`; `lumina_plasma.c:1830-53,4116-18,5787,8197-99` | bf 하준위 인구·분배함수 | 대체 | 대부분 없음 | 위험 | N | 기존 A18/A19/A37/H24 |
| C24 | `lumina_nlte_gemm.cu:186-90,379`; `lumina_atomic.c:961-65`; `lumina_plasma.c:12694,14922-32` | 미등록 σ_bf→Kramers | 대체 | 데이터 의존 | 위험 | Y | 기존 A16/L6 |
| C25 | `lumina_nlte_gemm.cu:182` | χ 부재→`1e10 eV` | 대체 | 없음 | 조건부 | N | 기존 A17 |
| C26 | `lumina_plasma.c:3249-53,3688-3713,14209`; `lumina_cuda.cu:8499,8562` | MC J̄ crossing 미달→binned J | 대체 | `JBAR_MIN` | 조건부 | Y | 기존 A21/M9 |
| C27 | `lumina_nlte_assemble.cu:123`; `lumina_plasma.c:13677,13775`; `lumina_cmfgen.c:1089,1303` | 빈/창밖 J→`1e-30`, 0, fallback | floor | 경로별 | 위험 | N | 기존 A24/A25/P9/L9 |
| C28 | `lumina_plasma.c:3441-63` | J→factor·`WB` 상하한 | cap | `J_CAP/​FLOOR_FACTOR` | 위험 | N | 신규(H9는 과거 88 이후) |
| C29 | `lumina_plasma.c:13706-68` | UV Jν→`W_cap Bν` | cap | `J_NU_UV_CAP` | 위험 | Y | 신규(H9 이후) |
| C30 | `lumina_plasma.c:8914,10028,10679-85,13296`; `lumina_cuda.cu:7153,8104-39` | Te/ion/J 반복 감쇠 | damping | 복수 env | 조건부 | N | 기존 A32 |
| C31 | `lumina_cuda.cu:816-49,8108-39,8169-81` | AA J̄/Jblue raw·EMA 통일 | damping | `JBAR_DAMP_UNIFY=1/2` | 조건부 | N | **신규 AA** |
| C32 | `lumina_plasma.c:7871-95,8048-68` | Υ 하한 | floor | `RADEQ_OMEGA_FLOOR` | 위험 | N | **변경 A29**: parity 기본 off, CMFGEN tier와 상호배타 |
| C33 | `lumina_plasma.c:502-680,7878-8080` | 미등록 Ω→vR/`OMEGA_SET` | 대체 | `OMEGA_CMFGEN` | 조건부 | Y | 신규 |
| C34 | `lumina_plasma.c:350-480,8061-68,14420-14717`; `lumina_nlte_assemble.cu:209-10` | gbar/Axelrod/forbidden Ω | floor | parity·mode | 위험 | P | 기존 A20/A26/A29/L7 |
| C35 | `lumina_plasma.c:14094-108,14456-65`; `lumina_nlte_assemble.cu:419-21` | `C_down≥εA`, DB로 C_up 재생성 | floor | `NLTE_COLL_FLOOR` | 조건부 | N | 신규 |
| C36 | `lumina_plasma.c:15446-64` | 전 이온쌍 α_DR 하한 | floor | `DR_FLOOR_CMS` | 위험 | N | 신규 |
| C37 | `lumina_plasma.c:2134-38,2285-95,2438-44,8389-99,11461-64,11844-45,15089` | 이온비·연쇄곱 `1e28/1e30` | cap | 경로별 | 위험 | N | 기존 A27/A38 |
| C38 | `lumina_plasma.c:2305,2356,2453,2662-63,6166,6206-21,8331,11897,11916` | n_ion·n_e 비정상→양의 상수 | 대체 | 경로별 | 위험 | N | 변경/확장 A38 |
| C39 | `lumina_plasma.c:8219-31,9989,11344,13300` | Te `[0.5,2]×Told` | damping | `TE_STEP_CLAMP` | 조건부 | N | 기존 A31 |
| C40 | `lumina_plasma.c:9950-78,11258-81,11306-39,13285` | Te bracket/HOLD/500·1000 K floor | 대체 | solver별 | 위험 | Y | 변경 A30 |
| C41 | `lumina_plasma.c:9787-88,11148-59,12787-98` | line cooling contribution cull | cap | `RADEQ_LINE_CULL` | 조건부 | N | 기존 A36 |
| C42 | `lumina_plasma.c:10193-201` | H-response trust region | damping | `HRESP_CLAMP` | 조건부 | N | 기존 A35 |
| C43 | `lumina_plasma.c:12068-74,12301-11,13009,13254-96` | S42 증폭·Newton 15% step·line search | damping | coupled solver | 조건부 | P | 신규(H11 이후) |
| C44 | `lumina_plasma.c:10153-62,10691-716` | 음수 선냉각→0 | floor | `COOL_NONNEG` | 위험 | N | 신규 |
| C45 | `lumina_plasma.c:10261-77,10474-76` | 상준위≤LTE, η_lag≥0 | cap | line-response 경로 | 위험 | N | 신규 |
| C46 | `lumina_plasma.c:1246-90,1520` | W>1e4/비유한→TR refit 또는 장 0 | 대체 | radiation-field 경로 | 위험 | N | 기존 A43 |
| C47 | `lumina_atomic.c:364-77`; `lumina_plasma.c:869,1240-90` | T_rad 고정·TEPIN·W cap | 대체 | `TRAD_COLOR_FIX` 등 | 위험 | P | 기존 A44/L5/H8 |
| C48 | `lumina_atomic.c:723-39` | `SUPER_CUTOFF` 이상 준위 lump·LTE 내부분배 | cap | `SUPER_CUTOFF` | 위험 | Y | 신규(H2 이후) |
| C49 | `lumina_atomic.c:435-55,961-65,1058,1122,1222` | 로더 불일치→0/Kramers/ground/Axelrod | 대체 | 데이터 의존 | 위험 | P | 기존 L1/L2/L6/L7 |
| C50 | `lumina_atomic.c:902-22` | multiplicity `[0,127]` 표현범위 | cap | spin table | 정당 | N | 신규(H25 이후) |
| C51 | `lumina.h:144-51,471-84,1252`; `lumina_atomic.c:1142,1238-43`; `lumina_plasma.c:8732` | shell/col-ion/collpair/network 정적 크기 | cap | 컴파일·loader | 조건부 | P | 신규 H14-H19/H25 |
| C52 | `lumina_cuda.cu:2724-64,5342,5856-90,6297-321` | packet interaction 절단·에너지 drop/force escape | cap | `MAX_INTERACTIONS` | 위험 | Y | 기존 M1 |
| C53 | `lumina_cuda.cu:4226,4491,6331-36`; `lumina_transport.c:419-49` | MA 내부 cascade | cap | `MA_INTERNAL_CAP` | 위험 | Y | 기존 M2 |
| C54 | `lumina_cuda.cu:2755,5342,6321-26`; `lumina_transport.c:523` | total step·CPU loop | cap | env/경로 | 조건부 | N | 기존 M3/M5 |
| C55 | `lumina.h:144-51`; `lumina_cuda.cu:4025,7035-53` | census/event-log 저장량 | cap | 계기 gate | 정당 | Y | **변경 M11**: 기본 400M→32M records |
| C56 | `lumina_cuda.cu:5158,7577-84,7625,8954`; `lumina_main.c:813-28` | vpacket τ, injection τ, SED·광선 수 | cap | 복수 env | 조건부 | P | 기존 F3/F5/F6/M7/M8/H17/H20 |
| C57 | `lumina_cuda.cu:3486-3504`; `lumina_main.c:41-67` | Planck sampler 반복 실패→대역 uniform | 대체 | 경로 | 조건부 | N | 신규 H10/H21 |
| C58 | `lumina_plasma.c:6450-68`; `lumina_cuda.cu:3358-61` | bf 격자 밖 0·마지막 빈 유지 | 대체 | 없음 | 조건부 | N | 신규 H4 |
| C59 | `lumina_plasma.c:4896-5055,6760-6925`; `lumina_cuda.cu:4781-4910,5599-5805` | k-packet fb Kramers 확률·대표 에지 | 대체 | `KPKT_FB_MULTI` | 위험 | Y | 신규 H1; Z에서 edge-failure counter 추가 |
| C60 | `lumina_plasma.c:14945-54,15014-54` | C1 GEMM→C2 estimator/빈별 fallback | 대체 | `C2_MATRIX_BF` | 조건부 | Y | **신규 Y3** |
| C61 | `lumina_plasma.c:2803-936,14962,15092-108,15311-73` | spin-forbidden 재결합율→0 | 대체 | `REC_SPINGATE` | 조건부 | P | **신규 Y4** |
| C62 | `lumina_plasma.c:3688-3713` | MA J̄ 문턱 10→`JBAR_MIN` | 대체 | `JBAR_UNIFY` | 조건부 | N | **신규 Y6** |
| C63 | `lumina_plasma.c:371-405,2018-2453,12694` | rate-SE field 선택·0/0 prior·ratio caps | 대체 | `RATES_FIX` | 조건부 | P | 신규 |
| C64 | `lumina_plasma.c:15594-694` | bb-isolated·top-stage 행을 Boltzmann anchor로 교체 | 대체 | `FLOOR_REG`, `TOPSTAGE_THERMALIZE` | 위험 | P | 신규 |
| C65 | `lumina_plasma.c:9415-24,9487-90` | stage-IV 하준위 `b_k≤1000` | cap | `STAGE4_BK_CAP` | 위험 | N | 신규 |
| C66 | `lumina_cuda.cu:9370-82` | `S/B>100`이면 pops rollback·J̄ 영구차단 | 대체 | THEN-MC/JBAR pops | 위험 | Y | 신규 H6 |
| C67 | `lumina_plasma.c:16914-40,17028-91`; `lumina.h:1154-86` | formal ray·continuum·τ/S provenance 경로 교체 | 대체 | `FORMAL_FIX` | 조건부 | Y | **변경 F1/F2** |
| C68 | `lumina_plasma.c:16942-45,17195-97` | thick-line `S_l→B(Te)` | 대체 | `FI_CLAMP_SL` | 위험 | N | 신규 |
| C69 | `lumina_plasma.c:16957-59,17206-12` | IGE forest opacity 제거 | 대체 | `FI_FOREST_NOBLANK` | 위험 | P | 신규 |
| C70 | `lumina_plasma.c:16234-41,16321-30` | Fe 창내 `S_l *= X` | 대체 | `FLUOR_ORACLE_X` | 위험 | Y | 신규/oracle falsifier |
| C71 | `lumina_cuda.cu:6539-61`; `scripts/run_coevolve_s01.sh:54` | line re-emission→`B(Te)` | 대체 | `LINE_THERM` | 위험 | N | 변경 H3: Z에서 배너만 수정 |
| C72 | `lumina_main.c:813-21`; `lumina_cuda.cu:9778-84`; `lumina.h:1178-80` | formal impact-ray 해상도 | cap | `CMF_NIMPACT` | 조건부 | N | **변경 F5**: 고정 100→env, 기본 50 |
| C73 | `lumina_cmf_selftest.c:307-51,512,955,1077,1440`; `cmf_pcygni_b1.c:199` | selftest 잔차·분모 정의역 | floor | test 전용 | 정당 | N | 신규/검증 전용 |

## 전수표 — scripts의 수치 처리

| ID | file:line | 대상 | 유형 | 게이트 | 3문 분류 | Ctr | 88 대응 |
|---|---|---|---|---|---|---|---|
| SC01 | `analyze_jnu_sed.py:20`; `euv_planck_check_s8.py:44`; `formal_integral_obsframe.py:21,87`; `frozen_in_milne_prototype.py:98`; `offline_bk_per_shell.py:23`; `patch_transprob_aul_weighted.py:57`; `expand_atomic_data_cmfgen.py:643`; `offline_macroatom_calc.py:25`; `lte_inversion_F1.py:108`; `cascade_walk_fe2.py:63`; `cascade_multicycle.py:52` | Planck/exp 정의역 | cap | 없음 | 정당 | N | 신규(script) |
| SC02 | `score_blondin_fscl_sn2002bo.py:76,119,145`; `diag_sl_vs_jline.py:37,58,62`; `compare_narrowband.py:76-77`; `analyze_nlte_matrix_svd.py:35-37,73`; `finalize_cmfgen_ref_npy.py:110`; 다수 plot/ratio | 로그·비율 분모 | floor | 없음 | 조건부 | N | 신규 |
| SC03 | `plot_hst_pcygni_map.py:53,58`; `single_fe2_line_pcygni.py:59,66`; `compare_ne_vs_cmfgen.py:92`; `build_toy06_epoch.py:194` | sqrt·기하학 범위 | floor | 없음 | 정당 | N | 신규 |
| SC04 | `build_ddc15_epoch.py:74`; `build_ddc15_initial_epoch.py:131`; `build_ddc15_real_composition.py:144` | isotope mass fraction→비음수 | floor | 없음 | 위험 | N | 신규 |
| SC05 | 아래 별도 전개한 86개 `slurm_*.sh` | 외곽 Fe 질량분율 `X_Fe≥5e-4` | floor | launcher 분기 | 위험 | N | 신규 |
| SC06 | `analyze_ddc15_F1_oskip.py:61`; `G1:60`; `H1:60`; `H1b:65`; `H1p:64`; `H2:66`; `I1/J1/K1/K1b:62`; `phase_D_si_red_validation.py:70` | continuum normalization≥peak 1% | floor | 없음 | 위험 | N | 신규 |
| SC07 | `frozen_in_milne_prototype.py:215`; `frozen_in_multistage_prototype.py:141`; `frozen_in_ode_test.py:104` | ODE ion fraction `[0,1]` | cap | prototype | 조건부 | N | 신규 |
| SC08 | `empirical_pcygni_ml.py:122,151` | kernel width≥0.01, τ≤8 | cap | 없음 | 위험 | N | 신규 |
| SC09 | `formal_integral_obsframe.py:38,50,52,87` | scattering fraction·ν endpoint·τ | 대체 | 없음 | 조건부 | N | 신규 |
| SC10 | `g2_inverse_regression.py:74,168,170,185,203,216`; `g1_jacobian_sensitivity.py:86-87` | 회귀 파라미터 feasible bounds | cap | 명시 범위 | 조건부 | N | 신규 |
| SC11 | `score_nw.py:133`; `check_mode_equivalence.py:90` | χ·검증 tolerance | cap | CLI/MC | 위험 | N | 신규 |
| SC12 | `expand_atomic_data_cmfgen.py:62-155,425`; `bake_coiii_real_sigma.py:78` | 원자 준위 수 | cap | config | 위험 | N | 신규 |
| SC13 | `expand_atomic_data_cmfgen.py:406-25,739-775,1004-54,1128`; `build_cmfgen_coldata_all.py:455-95` | 데이터 부재→cap/Kramers/skip | 대체 | 데이터 의존 | 위험 | P | 신규 |
| SC14 | `build_dr_cob3.py:196`; `parse_adasdr_adf09.py:85,107,134,137` | DR fit coefficient·weight 양수화 | floor | fit 경로 | 조건부 | N | 신규 |
| SC15 | `offline_cell_balance.py:220,226-28` | ETLA 상준위≤LTE | cap | prototype | 위험 | N | 신규 |
| SC16 | `offline_fluor_field_test.py:64,69,72,82,92,96` | gbar=0.2, f≥1e-6, pops≥0 | 대체 | prototype | 위험 | N | 신규 |
| SC17 | `validate_plasma.py:130-248`; `debug_neutral_tau.py:128,244,263`; `per_ion_tau_attr_si2_6355.py:107-54` | production floor의 검증 미러 | 대체 | 검증 전용 | 조건부 | N | 신규 |
| SC18 | `oracle_compare_cmfgen.py:348-51,408,446` | CMFGEN depth/frequency nearest-neighbor | 대체 | report-only | 조건부 | N | **신규 AB/oracle** |
| SC19 | `check_mode_equivalence.py:90`; `mode_convergence_telemetry.py:147` | MC noise tolerance·ratio denom | floor | 진단 | 조건부 | N | 신규 |
| SC20 | `compare_smooth_baseline.py:42`; `analyze_bfdark.py:77`; `plot_jbmap_pump_location.py:21` | 표시용 log/semilogy floor | floor | plot 전용 | 정당 | N | 신규 |
| SC21 | `build_cmfgen_coldata_all.py:18,455,486,495`; `expand_atomic_data_cmfgen.py:425`; `bake_coiii_real_sigma.py:78` | cap 위 전이·비양수 Ω 데이터 제거 | 대체 | 데이터 생성 | 위험 | P | 신규 |

### SC05 86개 발생 위치

모두 같은 `X_Fe = max(5e-4, ...)` 물리 조성 floor입니다.

```text
slurm_ddc15_223_ionlock_smoke.sh:146
slurm_ddc15_223_perionresc_smoke.sh:141
slurm_ddc15_A1_eps_fine.sh:141
slurm_ddc15_A1b_eps_dense.sh:141
slurm_ddc15_A1c_eps_top.sh:139
slurm_ddc15_A1d_e1p00_mcvar.sh:142
slurm_ddc15_A1e_e0p70_mcvar.sh:143
slurm_ddc15_A2_s2W.sh:156
slurm_ddc15_C2_xFeOuter.sh:129
slurm_ddc15_D1_Linner.sh:130
slurm_ddc15_F1_oskip.sh:131
slurm_ddc15_FI_ablation.sh:149
slurm_ddc15_FI_prod.sh:143
slurm_ddc15_G1_xFeInner.sh:134
slurm_ddc15_H1_epsUV.sh:134
slurm_ddc15_H1b_epsUV_knee.sh:131
slurm_ddc15_H1p_production.sh:135
slurm_ddc15_H2_epsUVred.sh:132
slurm_ddc15_H2p_redonly.sh:147
slurm_ddc15_H3_fate_attribution.sh:128
slurm_ddc15_I1_NiII_UVidown.sh:133
slurm_ddc15_J1_SiII_UVidown.sh:135
slurm_ddc15_K1_NiII_Aul.sh:136
slurm_ddc15_K1b_NiII_Aul_strong.sh:135
slurm_ddc15_KL1_stack.sh:142
slurm_ddc15_L1_SiII_Aul.sh:137
slurm_ddc15_M1_FeII_Aul.sh:136
slurm_ddc15_N1_KLM_stack.sh:142
slurm_ddc15_O1_CoCr_stack.sh:141
slurm_ddc15_P1_FeCo_push.sh:143
slurm_ddc15_P2_FeCo_stack.sh:142
slurm_ddc15_Q1_eps_uv_on_stack.sh:145
slurm_ddc15_Q1b_lambdamin_iron3_red.sh:144
slurm_ddc15_Q1c_lambdamin_wide.sh:141
slurm_ddc15_R1_eps_uv_2step.sh:149
slurm_ddc15_R2_aszeta.sh:150
slurm_ddc15_S1_siII_opt_aul.sh:149
slurm_ddc15_S2_siII_opt_finer.sh:147
slurm_ddc15_T1_s1_h2_stack.sh:151
slurm_ddc15_U1_feII_opt_aul.sh:152
slurm_ddc15_U1_ni2_opt.sh:151
slurm_ddc15_V1_c2_h1b_prod.sh:140
slurm_ddc15_W1_ca2_boost.sh:147
slurm_ddc15_X1_u1f005_ca2.sh:142
slurm_nlte3_diag.sh:138
slurm_nlte3fix_femerge.sh:153
slurm_nlte3fix_femerge_bare.sh:146
slurm_nlte3fix_femerge_combo.sh:145
slurm_nlte3fix_femerge_drop.sh:152
slurm_nlte3fix_femerge_optscan.sh:143
slurm_nlte3fix_femerge_probez.sh:150
slurm_nlte3fix_femerge_struct.sh:151
slurm_nlte3fix_femerge_sweep134.sh:145
slurm_nlte3fix_femerge_sweep34.sh:148
slurm_nlte3fix_optA.sh:145
slurm_nlte3fix_optAp.sh:148
slurm_nlte3fix_optC.sh:154
slurm_nlte3fix_optD.sh:153
slurm_nlte_o_prod.sh:144
slurm_nlte_o_recal.sh:145
slurm_nlte_o_recal_prod.sh:155
slurm_nlte_o_recal_seed.sh:149
slurm_nlte_o_seed.sh:148
slurm_nlte_o_smoke.sh:144
slurm_o_triplet_prod.sh:147
slurm_o_triplet_smoke.sh:147
slurm_plain_ddc15_sn2002bo.sh:153
slurm_v1_epoch_bracket.sh:157
slurm_v2_hst_epoch_bracket.sh:140
slurm_v32_4epoch.sh:145
slurm_v3_4epoch_w5frozen.sh:146
slurm_v3_de_l_sweep.sh:134
slurm_v3_epsir_sweep.sh:133
slurm_v3_vinner_sweep.sh:133
slurm_v4_ablation.sh:164
slurm_v4_inversion.sh:134
slurm_v4_probe.sh:133
slurm_v4_smoke.sh:143
slurm_viLzeta_grid.sh:158
slurm_w1_p37_retune.sh:145
slurm_w2_logL_red5_diag.sh:148
slurm_w3_nir_damp.sh:148
slurm_w4_ni2_nir_push.sh:146
slurm_w5_stack_closer.sh:146
slurm_zeta_clean_mcvar.sh:161
slurm_zeta_clean_smoke.sh:158
```

## 세대별 신규·변경·삭제

| 세대 | 판정 |
|---|---|
| X | `FORMAL_CONS_WINDOW`는 진단 분모 정정이며 물리 clamp가 아니므로 제외. 물리 상태 변경 없음. |
| Y3 | C60 `C2_MATRIX_BF` 신규: rate-field 생산자 교체. |
| Y4 | C61 `REC_SPINGATE` 신규: spin-forbidden recombination을 0으로 대체. |
| Y6 | C62 `JBAR_UNIFY` 신규: MA와 matrix의 crossing 문턱 통일. |
| Z | 물리 clamp 신설 없음. dominant-edge failure counter 추가, H3 배너 수정, Ni DR 라벨 정정. |
| AA | C31 `JBAR_DAMP_UNIFY` 신규. raw 통일은 고정점 보존 가능성이 있으나 finite-iteration 결과를 바꾸므로 조건부. |
| AB/oracle | `LUMINA_FROZEN_ORACLE`은 `#ifdef` observer뿐이며 피드백 없음. SC18 nearest-grid 대체만 report 처리에 존재. |
| 이후 현재 | C33 OMEGA-CMFGEN, C35 coll-floor, C36 DR-floor, C63 RATES_FIX, C67 FORMAL_FIX, C68-C70 formal/oracle falsifier가 추가됨. |

명시적으로 확인된 변경·삭제:

- 기존 A29 Ω floor는 현재 parity 경로에서 **unset=disabled**이며 `OMEGA_CMFGEN=1`과 동시에 켜지면 무시됩니다. 단 비-parity vR 경로의 별도 `om_floor` 기본 1.0은 남아 있어 완전 삭제는 아닙니다.
- 기존 F5 `n_impact=100` 고정은 삭제되고 `LUMINA_CMF_NIMPACT`, 기본 50으로 변경됐습니다.
- 기존 M11 event log 기본 400M은 현재 32M records로 변경됐습니다.
- FORMAL_FIX ON에서는 F1의 `S≤0→WB`가 사라지고, provenance가 없는 선은 0, nebular-owned 선은 τ와 일치하는 fused source로 바뀝니다.
- [lumina_plasma.c:16178](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16178)의 “`max(nlte,nebular)`” 설명은 **사문화 주석**입니다. 실제 코드는 [16309](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16309)에서 `tau_nlte`를 직접 대입하므로 활성 clamp가 아닙니다.

## 위험 상위 목록

물리량 대체형을 우선했습니다.

1. **NLTE 해의 사후 대체군 C13-C17/C64**  
   음수 인구→`1e-30`, LTE repair/FLOORM, `b_k` cap, 실패→Boltzmann, isolated/top-stage Boltzmann anchor입니다. \(n_i\)를 직접 바꿔 고정점과 선원함수·입자분배를 함께 바꾸며, 일부는 정규화 후 작은 floor가 증폭됩니다.

2. **선원함수 대체 C08-C10/C22/C67-C71**  
   미등록 ε→1, `S_l→B/​WB/0`, EPAY, 분모 컷, thick-line thermalization, forest 제거, Fe source multiplier, LINE_THERM입니다. 흡수율과 쌍을 이루지 않는 경우 보존법칙을 직접 위반합니다.

3. **k-packet free-bound 대체 C59**  
   Kramers 확률과 단일 대표 에지가 실제 다중 continuum Milne 분포를 대체합니다. Z에서 failure count는 생겼지만 에너지 가중 편향 카운터는 없습니다.

4. **수송 절단 C52/C53**  
   interaction cap은 packet 에너지를 삭제하거나 강제 escape시키며, MA cap은 cascade를 공명 재방출로 바꿉니다. 건수 카운터는 있으나 C52의 채널·파장별 에너지 편향이 핵심입니다.

5. **충돌·재결합 발명값 C32/C34/C36**  
   Ω floor, Axelrod/gbar forbidden floor, 전 이온 공통 α_DR floor는 실제 rate를 하한값으로 교체합니다. 특히 Ω floor는 과거 실측상 광범위 발화한 전력이 있습니다.

6. **복사장 직접 절단 C27-C29/C46-C47**  
   Jν를 `1e-30`, `factor×WB`, `W_cap B`로 바꾸거나 W rail/TRAD pin을 적용합니다. 계산된 장이 기준 Planck장으로 대체되므로 고정점 불보존입니다.

7. **원자 해상도 절단 C48/SC12/SC21**  
   super-level lump, level cap, cap 위 전이 drop은 사건 공간 자체를 제거합니다. 메모리 제약은 이해되지만 물리 오차상계가 별도로 필요합니다.

8. **scripts 조성 floor SC04/SC05**  
   음수 isotope clip과 86개 launcher의 `X_Fe≥5e-4`는 입력 조성을 바꿉니다. 특히 SC05는 실험 종류와 무관하게 광범위 복제돼 있습니다.

9. **offline 모델 대체 SC15/SC16**  
   ETLA no-pumping cap과 placeholder gbar/f oscillator floor는 prototype 결과를 물리 검증값처럼 사용할 경우 위험합니다.

10. **진단 목적함수 절단 SC06/SC08/SC11**  
    continuum norm floor, τ≤8, χ clip은 생산 물리는 바꾸지 않지만 최적값·판정 순서를 바꿀 수 있습니다.

## 발화 카운터 부재

위험 분류 중 카운터가 전혀 없는 핵심 항목:

- 소스: `C08, C13-C16, C22-C23, C27-C28, C32, C36-C38, C44-C46, C64-C65, C68, C71`
- 부분 계측만 있는 항목: `C09-C10, C34, C47, C49, C64, C69`
- scripts: `SC04-SC12, SC14-SC20`은 모두 발화 카운터가 없습니다. `SC13/SC21`도 manifest·상태 메시지는 있으나 물리 가중 손실 카운터가 아닙니다.

특히 우선 추가가 필요한 카운터는 다음입니다.

- negative-pop/LTE/FLOORM/BK cap: 원인별 건수 + 인구·방출 가중량
- `S_l→B/WB/0`: provenance별 선 수 + \(S(1-e^{-\tau})\) 에너지
- Ω/DR/collision floors: 전이 수가 아니라 냉각·율 가중 증분
- J cap/floor: rate·파장·shell 가중 영향
- atomic level/transition caps: 제거된 \(A_{ul}\), f-value, opacity, cooling 합
- composition floor: 수정 전후 질량과 재정규화량

## 제외한 위양성

| 부류 | 예 | 제외 사유 |
|---|---|---|
| 배열·인덱스 경계 | `DDC_CAP`, `searchsorted` bin clip, binary-search `min/max`, `shell<256` | 메모리 안전·인덱스 선택이며 수치 물리량을 자르지 않음 |
| 동적 배열 capacity | `cmf_pcygni_b1.c:136-145`, `radeq_col_pairs_bench.c:85-88` | 저장공간 한계/bench 전용 |
| 통계 집계 | `max_error`, `argmin`, `best=min(...)`, plot y-limit | 관측값 선택·표시이며 floor/cap 아님 |
| 파장·속도 범위 검증 | `oracle_compare_cmfgen.py:349-351`의 outside→exception | fail-closed; 대체 없음. 단 nearest-neighbor 자체는 SC18로 포함 |
| 주석만 존재 | `lumina_plasma.c:16178`의 nebular max, 여러 “floor decision” 설명 | 실행문과 불일치하거나 설계 설명뿐 |
| 계기 버킷 | JBLUE ±3 dex histogram clamp | 물리 소비값이 아닌 진단 축 포화 |
| 파일 자체 무코드 | `src/lumina_cuda.c` | 0행 |
| 선언만 있는 header | `lumina_cmfgen.h`의 API·sanity 설명 | 실행 가능한 경계 처리 없음 |