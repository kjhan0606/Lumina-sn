# 현재 작업 계획 — 2026-08-09

운영 승인 규칙(2026-08-09): 현재 배선·smoke 단계의 구현, 로컬 회귀, CPU/CUDA
빌드, H200 pilot/full 판정은 중간 단계승인을 요청하지 않고 자율 진행한다. 이 범위가
모두 닫힌 뒤에만 CMFGEN same-identity finite 비교와 coevolution을 포함한 다음 국면
계획을 사용자에게 제시하고 승인을 받는다. Fable은 핵심 모호성에만 제한한다.

마지막 갱신: H200 flight `251966`에서 line-net 차단의 원시 원인을 전역
infinity-norm Jbar bound의 극단적 cell-local 과대평가로 확정; 국소 오차봉투 설계 착수

```text
IMPLEMENTATION_CLOSURE = ACCEPT
SH_GRID = ASSET_COMPLETE_PHOTO_ACCURACY_OPEN
LOW_BAND_DISCRETISATION = ACCEPT
CMFGEN_FINITE_GAMMA = ACCEPT_23_CELLS
UPPER_SIV_PHOTO = ACCURACY_FAIL_EFFECT_NEGLIGIBLE
PHYSICAL_FLIGHT = JOB_251622_FAILED_NO_BRACKET__LINE_RATE_NOT_CMFGEN_EQUIVALENT
DIAGNOSTIC_FLIGHT = JOB_251596_COMPLETE__STAGE4_SI_III_IV_MATRIX
REPAIR_FLIGHT = JOB_251599_COMPLETE__GENERATOR_GTH_STAGE4_PASS
FE_RANK_FLIGHT = JOB_251601_COMPLETE__EXACT_ZERO_PAIR_PROVEN
ZERO_TOTAL_REPAIR = JOB_251622_COMPLETE__LOWER_MID_UPPER_EXACT_ZERO
NEGATIVE_TAU_ENERGY = FAIL_CLOSED__CMFGEN_ETAL_ZNET_REQUIRED
LOCAL_VALIDATION = D19_K7_Z12_CP4_PASS__CUDA_A54D2600
FABLE = EXP_GENERATOR_VERDICT_CAPTURED_THEN_REFROZEN
CMFGEN_LINE_NET_FIXTURE = V2_SEALED__0.6680659609711768_CGS
LINE_NET_DATA_CONTRACT = SEALED__QE_SUPERSET_QG_SUBSET__ATOMIC_COMMIT
LINE_NET_KERNEL = PASS__FINITE_0.66806596097117665__FMA_2POW_MINUS54
LINE_ENERGY_SET = PASS__OPHYS_QE_2783421__LUMINA_ACTIVE_QE_2180286__QG_SUBSET_FAIL_CLOSED
LINE_JBAR_OWNER_SCHEMA = PASS__ONE_QE_CACHE__SPARSE_QG_VIEW
DET_QE_PUBLICATION = PASS__ATOMIC_QE_CACHE__QG_RATE_VIEW__QE_ENERGY_VIEW
GPU_QG_FROM_QE = H200_PASS__SPARSE_GATHER__BYTE_ATTESTED
MC_QE_PUBLICATION = PASS__PATH_LENGTH_QE__QG_SUBSET__NO_VARIANCE_CLAMP
```

Coevolution 승계 경계:

- 최종 구조는 인수인계서의 양팔 coevolution 그대로다. 동일 물질상태를 DET-CMFGEN
  팔과 MC 팔이 공유하고, 반복 `it`의 갱신 상태로 만든 MC 복사장이 반복 `it+1`의
  전체 물질상태에 되먹임된다. 결정론 loop A가 master이며 frozen `THEN_MC` 후처리는
  coevolution으로 간주하지 않는다.
- 현재 `LUMINA_DET_TRANSACTIONAL=1` flight는 이 고리에 재연결하기 전 공유
  NLTE/RADEQ 생산자를 고리 밖에서 단독 검증하는 단계다. 그래서 제출기가
  `LUMINA_MC_COEVOLVE*`를 의도적으로 제거한다. DET 수렴 뒤
  `LUMINA_MC_COEVOLVE=1` 경로, 두 팔 generation barrier, lagged MC→next-iteration
  feedback을 다시 검증한다.
- 양팔 일치는 공통 원자자료·population·격자·closure 오류에 눈먼 차동 잣대다.
  따라서 CMFGEN의 동종 finite 출력과의 외부 비교를 별도 잣대로 계속 유지한다.

가장 최근 완료:

- [x] 24,542×1234 CMFGEN sigma fresh bake 및 원자 승격. SHA
  `90d04042c17bcc5f2c7c521b65a9bb0f824179d79493f82ad40deaa7185cc3ad`.
- [x] Si V ground threshold는 새 grid 안이며 CMFGEN row가 실제로 등록된다.
- [x] CPU/GPU photo-rate 소비자는 full-bin-average sigma 질량을 physical threshold
  위 active support로 재배치하고 canonical `K=2` J와 결합한다.
- [x] 저주파 707 level×90 depth 재검증: global rate `6.962e-5`, ion rate
  `1.573e-4`, weighted-L1 rate `2.455e-4`; 보조 discretisation PASS.
- [x] 현재 O-PHYS `*PRRR` 직접 출력의 유한 Γ를 생산 CPU BF rate 함수로 재현.
  판정 23셀, CMFGEN 범위 `3.362e-7–1.723e2 s^-1`; Fe III s0
  `172.2797588↔170.8093667 s^-1`, Co III s0 `16.51833371↔16.44964614`,
  S III s0 `19.07608805↔18.97556829`.
- [x] 상한 Si V Wien-tail은 sigma 정확 재구성·Milne `1.421%`지만 photo-rate
  상대오차 `61.95%`다. 극소 absolute effect `1.688e-22/epoch`는 영향도
  `NEGLIGIBLE`일 뿐 정확도 PASS가 아니다.
- [x] sharp-edge 이후 CPU/OpenMP full link, 핵심 selftest, header 28/28,
  `git diff --check`, CUDA sm_80/86/90 fresh link PASS.
- [x] CUDA SHA
  `69407eec7ae088101872115fd663d84d584c66d4310613245aff34c86f86dc48`.
- [x] 전체 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그 SHA
  `a2a2498ea9a05fb2447ff917ac20bdde5131ef63446f33ee64043427f7e5ebcf`.
- [x] H200 `syn104` job `251513` 제출·실행. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T142233Z_69407eec7ae0`.
- [x] 1234-bin exact-sliding 45회, residual `9.8975866364090719e-09`
  (`tol=1e-8`) PASS; R6 판정 대상 1,391,131선 100% 유효.
- [x] job `251513` 종료: H200 `syn104`, `FAILED 70:0`, elapsed 18분37초,
  MaxRSS 79,472,484 KiB. 공개 Te/세대 byte 보존 PASS.
- [x] 이전 `POP_BF_OOG`는 소실. 새 최초 차단은 Si I–II pair level-SE.
  3500 K shell 4에서 63 음수, min `-0.0061327292 cm^-3`,
  `|min|/max=2.24548e-10`, backward error `6.09233e-17`.
- [x] 140000 K 동일 pair/shell은 `NONFINITE`/inversion sanity gate에서 차단;
  linear rank 202, backward error `7.08856e-17`.
- [x] 기본값을 바꾸지 않는 CPU trial matrix/RHS 온도별 dump hook 구현.
  CPU/OpenMP full link 및 A2-07/candidate/tau/A2-10 selftest PASS.
- [x] 진단 CUDA binary SHA
  `09b2f7bc4f05d4266fbf51d650268e73e2ac32744579ee690bf3a412a2ffac54`;
  `MATDUMP-CPU` 포함, sm_80/sm_86/sm_90, `make -q` PASS.
- [x] 신규 전체 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS.
  로그 `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_nlte_matdump_cpu_2026-08-08.log`,
  SHA `5b46939e1763b3b9b9bab6f5e6ba5a37fdd36ce988973018d82931ee665f72a9`.
- [x] H200 diagnostic job `251514` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T145122Z_09b2f7bc4f05`.
- [x] job `251514` lower 3500 K post-lock 202×202 행렬 수집. SHA
  `e213a115f852df1cb84f562d733d559b90a7e8d78294a739cfc1a7a342cba1f7`.
- [x] lower 균형화 후 `cond2=2.77248e6`; 200 SE 행 상쇄배율 중앙
  `8.07821e15`, 최대 `2.15746e18`.
- [x] 80 decimal digit solve도 63개 음수, min `-0.00613272916244`로
  double과 일치. 따라서 solve 단계 roundoff는 반증됐지만, 조립 중
  이미 입력 `A`에 들어간 상쇄는 아직 열려 있다. 결과 JSON SHA
  `03d02e5e0eb87994ea0a810ec7727a07b0d5ff9e1ffdaa3c9e045d9636e2075e`.
- [x] job `251514` upper 140000 K post-lock 행렬 SHA
  `809c3da5a570fc41b55fed422288a2a2b86cdbf6a1542f86c0514c50d804076f`.
  실제는 finite 해이며 ground min `-4.59538371271e-22`, max
  `1.80727324864e-14`, 음수 10개로 inversion sanity gate가 `NONFINITE`로
  분류했다. 80자리에서도 같은 부호; JSON SHA
  `5a4d3d69b13e9e5d1f21d5e2da23481d5085276ad74d1171138fe4daea86cee6`.
- [x] job `251514` 종료: `FAILED 70:0`, H200 `syn104`, elapsed 18분23초,
  MaxRSS 79,472,200 KiB. stdout/stderr/footer SHA는 각각
  `a62652e93d35af8dbc022acdf116db63336f9a7b5e2f45cad7077403cdb48ba6`,
  `ce950c873eecd12170a72a83da7efc23e14e16dc439593964f3fed61b2ae7e34`,
  `02243777ca15505f2f429cc7bf4f5afe9c1f9ac831254ff4300c5c37bac9b419`.
- [x] prelock raw-rate/postlock 동시 dump와 열합 직접 복원 분석기 구현.
  CPU full link·관련 selftest PASS; CUDA SHA
  `69ce0c8465acaa29fe85b7dc68932d4235a59f3f7a1da103a75bda426b98f367`,
  sm_80/86/90 PASS.
- [x] prelock 계측 신규 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS.
  로그 SHA `39904ede64fb0480d34a03e08b5e68fd292da1e9283e4f25d3fcc933d7dc391a`.
- [x] prelock/postlock H200 job `251515` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T151200Z_69ce0c8465ac`.
- [x] 사용자 요청에 따른 Fable 단발 중요 판정: `n=exp(z)`는 부적합성을 숨기는
  양수성 강제이므로 기각. `exp(tQ)`/uniformization은 전체 원소 총량 하나만
  보존하므로 두 stage-total lock의 생산 대체제가 아니며, repaired `Q` 자체의
  정상 이온비를 구하는 독립 진단기로 채택. 정본
  `docs/FABLE_VERDICT_NLTE_EXP_GENERATOR_2026-08-09.md`.
- [x] prelock 분석기에 단일 원소총량으로 정규화한 repaired `Q`의 정상상태와
  uniformization `P=I+Q/Lambda` 공리 검사를 추가. 80자리 경로는 float64
  대각을 재사용하지 않고 imported off-diagonal을 다중정밀도에서 정확히 재합산.
  4-state 합성 generator에서 양수 정상상태·비음수 `P`·열합 오차 0 및 의도한
  stage-ratio mismatch `1.4328493648x` 검출 PASS.
- [x] job `251515` 종료: `FAILED 70:0`, H200 `syn104`, elapsed 18분23초,
  MaxRSS 79,470,188 KiB. 물리 실패는 재현됐지만 pre/post 파일이 0개라 계측
  차단으로 판정. stdout/stderr/footer SHA는 각각
  `45c6f6e6b0f9d1f6dec2f18d5e578e400ab5db490f18a0961f898090c2069483`,
  `76d502c5506cdb60913980f62eef5815bbefb07072d4e34dd5550014b3ba9e69`,
  `a39786808c184039d26f4ab105d44c6ea47907836644fc914c582888d8fcbe2e`.
- [x] 원인은 private A2-10 candidate의 일반 diagnostic effect=0에 prelock hook을
  잘못 묶은 계측 권한 오류. 일반 side effect를 열지 않고 명시적
  `LUMINA_NLTE_MATDUMP=1`만 허용하는 `FORENSIC_MATRIX` effect bit로 분리.
- [x] 권한 수리 후 CPU full link·관련 candidate/A2-07/A2-10 selftest PASS,
  전체 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그 SHA
  `42df41872a617777a085088984feeae687397450344ae4f0c74b46fd368dc8cb`.
- [x] CUDA SHA
  `0f87386c38283e3e0685d24bc94cb71568a3537b845515b9c75d4beefea76ae8`,
  sm_80/86/90 및 `make -q cuda` PASS. 별도 OpenMP 강제 full link도 PASS.
- [x] forensic 재비행 job `251516` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T153746Z_0f87386c3828`.
- [x] job `251516` lower 3500 K pre/post-lock 동시 수집. pre SHA
  `3534b840571157183a569e9d864502ae662eb94df66764e0d28f6e9e49cbe246`,
  post SHA는 job 251514와 동일한
  `e213a115f852df1cb84f562d733d559b90a7e8d78294a739cfc1a7a342cba1f7`.
- [x] lower pre-lock은 RHS 비영 0·음수 off-diagonal 0인 homogeneous generator.
  raw 열합 오차 최대 `2.90283e-6`, direct-diagonal delta 최대 `2.86102e-6`로
  assembly cancellation은 실재하지만, 복원 후에도 double/float-diagonal 80자리/
  exact-generator 80자리 모두 음수 63개, min 각각 `-0.00613272915189`,
  `-0.00613272915211`, `-0.00613272915752`. 원인 가설에서 탈락.
- [x] lower repaired `Q`의 단일-total 정상상태는 전 준위 양수, Si I 총량
  `8217.81598204` 대 lock `91306.8308711`; generator Si I/II 비는 lock 비의
  `0.0897304249x`다. 즉 lock은 자체 평형보다 약 `11.1445x` Si I-rich.
  stage-lock↔level-SE 구조 불일치 확정. JSON SHA
  `1cbbb641ba59c797d928b9de5a408346c493c479981522892c2551ef4c024494`.
- [x] lower locked population을 exact-sum `Q`에 되넣은 순 stage flow는
  Si I→Si II `9.64801788667e5 cm^-3 s^-1`; Si I target 대비 손실률
  `10.5665893719 s^-1`. 두 삭제 행이 이 동일·반대 유량을 숨기며 전체 원소
  net은 수치 0이다.
- [x] job `251516` upper 140000 K pre/post-lock 동시 수집. pre SHA
  `2e1db767978731a7b6c12ee824d19f11900663a8d34b94f4f3093c11d8fff394`,
  post SHA `809c3da5a570fc41b55fed422288a2a2b86cdbf6a1542f86c0514c50d804076f`.
  pre-lock은 lower와 동일하게 homogeneous nonnegative-offdiagonal generator.
- [x] upper 대각 exact 복원 후에도 80자리 음수 10개, ground
  `-4.595383712714619e-22`; assembly cancellation 가설 탈락. 양수 `Q`
  정상상태 Si I 총량 `2.33488100116e-17` 대 lock `1.15809679792e-23`,
  generator stage 비가 lock의 `2.01867746287e6x`. locked 순 Si II→Si I
  flow `3.40260402941e-16 cm^-3 s^-1`, Si I target 대비
  `2.93809985099e7 s^-1`. JSON SHA
  `f7f55e34d35e9bbee987afa665c619c65d717fce6822d1f6f2cec5d0a51cb254`.
- [x] job `251516` 종료: `FAILED 70:0`, elapsed 18분24초,
  MaxRSS 79,473,076 KiB. stdout/stderr/footer SHA는 각각
  `a17d2476fb4161a5329e8d7dc64d6f8a53a4d1cbe6489f2e401eb3a46edb183f`,
  `40ea48a26c6ec3596a1a50f33198ff54c3f9241b9b7828df3fb5ea2ed0ce654f`,
  `461adb6fd1b943f880ec6405da536be966afced0dc3869b77f2cdb6ce0f287ba`.
- [x] 기존 solver에는 combined element-total 한 행으로 `Q`가 stage partition을
  정하는 경로가 이미 있다. reference launcher가 `ION_LOCK=1`과
  `PER_ION_RESCALE=1`을 무조건 재강제하던 것을 default=1 보존·명시적 caller
  override 허용으로 바꿔 sealed lock-off A/B가 가능하게 했다.
- [x] launcher override 계약 default=`1/1`, explicit=`0/0` 및 전체 battery
  D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그 SHA
  `747d3f83a04a36c9d46b2c4139bd6e7d7f978ef60f7b5fdc5a9b6b2b28cebf10`.
- [x] 최초 single-total 제출 job `251518`은 submitter의 inherited `LUMINA_*`
  sanitization 뒤 staged 값이 `1/1`로 되돌아간 것을 제출 직후 검출, 28초에
  취소했다. 물리 결과로 사용하지 않는다.
- [x] submitter에 fail-closed `DET_SINGLE_TOTAL=1` control을 추가. reference 환경
  해석·오염 제거 뒤 정확히 `0/0`을 적용하고, staged exports가 기대값과 다르면
  `sbatch` 전에 거부한다.
- [x] 올바른 single-total A/B job `251519` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T160253Z_0f87386c3828`;
  staged `ION_LOCK=0`, `PER_ION_RESCALE=0`, `single_total=1` 확인.
- [x] job `251519` lower 3500 K single-total post 행은 정확히 한 행(201)만
  202개 계수 `1.0`, RHS `27523967.342142675`로 바뀌었다. 해는 double/80자리
  모두 음수 0개, min 각각 `8.64750554722e-9`, `8.64750554612e-9`이며
  Si I=`8217.81593150`, Si II=`27515749.5262112`로 repaired `Q` 정상상태를
  재현했다. JSON SHA
  `ef2e53d7012e26699b4e38c2f545e331307ebcf6a0afa55af37e36f691c6e876`.
- [x] 따라서 두 stage-total lock의 중복 권위가 Si I-II 음수 해의 원인이며,
  combined element-total 한 행은 양수 finite 해를 실제로 복원한다는 A/B가 성립했다.
- [x] 다음 최초 차단은 active 조성표에 행이 없는 C의 보존된 NLTE pair가
  `INVALID_LAYOUT`으로 떨어지는 Z-inert 경계였다. 원소 미등재를 all-zero row와
  동등한 inactive 상태로 닫고, 조성 offset 조회가 못 찾는 잔존 ion-catalogue 슬롯은
  이 Z-inert 경로에서만 보조 검색해 exact zero로 만든다.
- [x] 조성 미등재 Z의 level/ion exact-zero, active pair 불변, invalid pair fail-closed
  회귀를 `selftest_nlte_candidate_tau`에 추가해 PASS. `git diff --check` PASS.
- [x] 직접 영향 CPU 회귀군 강제 재빌드·실행 PASS:
  `selftest_nlte_population_candidate`, `selftest_nlte_candidate_adiabatic`,
  `selftest_nlte_candidate_tau`, `selftest_a2_07_population`,
  `selftest_a2_10_radeq`(N1–N8 8/8). 기존 단일 미정의-static 경고 외 신규 실패 0.
- [x] Z-inert active-only 보완 후 CPU/OpenMP full link PASS. 전체 battery
  D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_single_total_zinert_absent_2026-08-09.log`,
  SHA `410e7f8138b8dbfb933dc6b731c751dd36e0484ea7553e901266bc7edde83e31`.
- [x] CUDA sm_80/86/90 fresh link PASS. 새 `lumina_cuda` SHA
  `25d690395052cd0c892dab93dfbf71d70f2eceb005993f6a32c70a67034e44ca`;
  cubin 3종 확인 및 `make -q cuda` PASS.
- [x] 구 바이너리 single-total job `251519` 종료: `FAILED 70:0`, elapsed
  19분08초, MaxRSS 79,481,148 KiB. lower와 upper 모두 Si I-II solve 자체는
  양수였고 이후 공통 C `INVALID_LAYOUT`에서 차단됐다. stdout/stderr/footer SHA는
  각각 `5bd261e9ecc62994907d8f299bfb806b5ff21ca5519804a46f692ad6d8bff49f`,
  `019a8a88ed6a1802b32fc749b7588c1a35d030ff0499944deb71ea811824bd6a`,
  `52bf787d88c6aaac69f18d9eee9a04f5040d4e0d5c73282c3a7dbd358293d555`.
- [x] upper 140000 K single-total도 한 행(201)만 전 계수 `1.0`, RHS
  `1.8548130497484562e-14`로 교체. double/80자리 음수 0개, min
  `3.28310056120e-26`; Si I=`2.33488100056e-17`,
  Si II=`1.85247816875e-14`. JSON SHA
  `88cc79da54a7689463b6d4fc5e7ed5d08533b4a1ece657b3eca0382d5415fe92`.
- [x] 수정 CUDA SHA로 H200 job `251523` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T162634Z_25d690395052`;
  staged `ION_LOCK=0`, `PER_ION_RESCALE=0`, `single_total=1`, `MATDUMP=1`,
  binary SHA `25d690395052cd0c892dab93dfbf71d70f2eceb005993f6a32c70a67034e44ca`
  확인. H200 `syn104`에서 실행 중.
- [x] job `251523` lower에서 Si solve와 C pair exact-zero는 통과했다
  (`solve_stage=NONE`, `ion_status=OK`). 새 최초 차단은 post-population
  tau/source typed validation의 `NLTE_CANDIDATE_OPACITY_FAILED`였다.
- [x] 소스 계약 추적으로 실제 `Inf`가 아니라 unowned/absent-Z 선의
  `line_source_validity=0`(calloc 미초기화 enum)이 포괄 `POP_NONFINITE` 라벨로
  거부되는 경로를 확인했다. source owner가 매 세대 모든 선을
  `0/A208_UNSAMPLED`로 초기화하고 mapped 선만 교체하도록 수리했다.
- [x] active-only 조성에서 아예 없는 Z는 pair뿐 아니라 bulk tau, NLTE source,
  direct A209 emissivity까지 `EXACT_ZERO`로 일관되게 단락한다. 첫 invalid
  line/shell/Z/ion/value/status를 내는 fail-only 진단도 추가했다.
- [x] absent-Z line을 실제 두 번째 선으로 넣은 candidate 회귀에서 private
  population→signed tau/source→A208→A209 publication 전체 PASS, active 선 값과
  public byte 불변 PASS. `git diff --check` PASS.
- [x] job `251523` 종료: `FAILED 70:0`, elapsed 20분23초,
  MaxRSS 79,472,016 KiB. lower/upper 모두 `solve_stage=NONE`, `ion_status=OK`로
  Si 및 C pair 경계를 통과했고 같은 old source-validity 차단에서 종료.
  stdout/stderr/footer SHA는 각각
  `89091a5faaf32704dfbdef1978d3ec5d702fc1c187e04da51c298d6a3d67b395`,
  `4622ec818031fee6103325ded660914de8eb3935f692345e3b3d62c433e345cf`,
  `26b31592620e509c6bd56ffd1ada7972c029ddad1535ab5baf4a6412f2517f8b`.
- [x] source typed-initialization/A209 Z-inert 수리 후 직접 CPU 회귀군 PASS:
  `selftest_nlte_population_candidate`, `selftest_nlte_candidate_tau`,
  A2-08 N1–N8 8/8, A2-09 N1–N8 8/8, A2-10 N1–N8 8/8.
- [x] CPU/OpenMP full link PASS. 전체 battery D 19/19, K 7/7, Z 12/12,
  CP 4/4 PASS. full-topology inactive 353,770선 exact-zero, active tau byte 차이 0.
  로그 `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_single_total_zinert_source_2026-08-09.log`,
  SHA `f4d6294fec161f4352d9e0744c2971efaddfb1832a36c8f5574a4d3cd0cdaa55`.
- [x] CUDA sm_80/86/90 fresh link PASS. `lumina_cuda` SHA
  `fbf20eafd8414f8411422c16bcdc3177a57cef6b9101af23af31ed6805013441`;
  `make -q cuda` PASS.
- [x] source/A209 수리 SHA로 H200 job `251528` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T165331Z_fbf20eafd841`;
  staged `ION_LOCK=0`, `PER_ION_RESCALE=0`, `single_total=1`, `MATDUMP=1` 확인.
- [x] job `251528` 종료: `FAILED 70:0`, elapsed 20분33초,
  MaxRSS 79,472,432 KiB. lower/upper 모두 pair solve 양수, C exact-zero,
  tau/source, A208/A209와 BF를 통과했다. 최초 차단은
  `NLTE_CANDIDATE_INTERNAL_ENERGY_FAILED / POP_ATOMIC_MISSING`였다.
  stdout/stderr/footer SHA는 각각
  `9f785f9995c8c07694f52e25a8ce06f7dc4e2adecba201741a17fa8ad4b6953d`,
  `896cb5e7c7c05200dd8597ab33dd55f1d7757faba3643bb9c77594d1c88deb53`,
  `6abec4f9200643586a7f57244bccbc164d30a4b11a3ff212f936446c0a17ef74`.
- [x] active-only 덱은 6개 원소의 population/rate ladder를 stage II부터 시작하므로
  regular ionization table에 q=0이 없다. CMFGEN neutral-ground 내부에너지만 그
  여섯 scalar가 필요함을 확인했다. 중성 population/rate를 복원하지 않고 별도
  `data/atomic/ionization_reference.csv`에 q=0 여섯 행만 봉인했다. SHA
  `660141ef55e5b9028f6a8f374a28a149bcf81e67f1f888fc0e157397aea4fa9a`.
- [x] regular deck row 우선, 누락 링크만 reference fallback, 양쪽 누락 시 output
  byte rollback을 단위 known-answer로 검증. candidate 실패는 상세
  `ATOMIC_INTERNAL_ENERGY_*` status를 출력한다. strict env universe는 501개로
  기계 재생성했다.
- [x] active canonical loader 실측: regular ionization 27, reference 6,
  ion populations 기존 33, mapped levels 24,542. active tau FNV64
  `6c53c2f89ad53e47` 불변. CPU/OpenMP full link와 직접 회귀군 PASS.
- [x] 최종 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_single_total_energy_reference_final_2026-08-09.log`,
  SHA `b2f7f2b985f98919dccb622b8d533d41bdb3866d39b09767627da193db827442`.
- [x] CUDA sm_80/86/90 fresh link 및 `make -q cuda` PASS. binary SHA
  `8a724067e4b2c161a52c4915234200827cef95a4cb5dd91732178c85a551852a`.
- [x] sealed H200 job `251574` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T220622Z_8a724067e4b2`;
  staged reference SHA 일치, `ION_LOCK=0`, `PER_ION_RESCALE=0`,
  `single_total=1`, `MATDUMP=1` 확인.
- [x] job `251574` 종료: H200 `syn104`, `FAILED 70:0`, elapsed 20분34초,
  MaxRSS 79,469,968 KiB. q=0 reference와 lower/upper 양수 pair, C exact-zero,
  tau/source, A208/A209, BF는 통과했다. 양 endpoint 공통 최초 차단은
  `ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE`였다. stdout/stderr/footer SHA는
  각각 `11e277597217ad15a2b52593906ed378d1e3568fce44431d71bcb70d46e0ec57`,
  `f0c24603df06529768d7b1e10a83a25219fd94277d1887dea1b5a7db9e9a7809`,
  `9bf465964e596e156ccb878a6011916e3d98613ceca8fb5c5e7d07cb73f3f2d5`.
- [x] 이 차단은 수치 음수가 아니라 owner 계약 문제다. single-total SE가 tracked
  stage partition을 결정하는데 internal energy가 각 stage를 이전 upstream
  ionization estimate와 다시 같게 요구했다. fully tracked stage는 실제 level 합을
  사용하고, untracked stage는 기존 ion-population closure를 유지하며, 모든 stage를
  합친 원소별 nuclei total은 반드시 upstream element total과 닫히게 수리했다.
- [x] upstream estimate가 stage를 exact-zero로 두어도 fully mapped SE가 finite
  stage를 만들 수 있는 경계를 별도 known-answer로 검증했다. stage별 재분배 허용,
  원소총량 위반 거부, output byte rollback 모두 PASS.
- [x] 직접 internal-energy/candidate/tau 회귀, `git diff --check`, 전체 battery
  D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 최종 gate 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_single_total_element_closure_upstream_zero_2026-08-09.log`,
  SHA `5afe76c8c0299aa844fd005659e72e3a1772c83ed8210f2a3f2f6c908af57843`.
- [x] CUDA sm_80/86/90 fresh link와 `make -q cuda` PASS. 새 binary SHA
  `56a8c2caaaf2aca6d02d16ed218bf9716dc6587f0160088b38fb65894c91d740`.
- [x] 원소별 closure H200 job `251575` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T223400Z_56a8c2caaaf2`;
  H200 `syn104` 배정, staged binary/reference SHA 일치,
  `ION_LOCK=0`, `PER_ION_RESCALE=0`, `single_total=1`, `MATDUMP=1` 확인.
- [x] job `251575` exact-sliding 45회 residual
  `9.8975866364090719e-09 < 1e-8`, negative recurrence 0; R6 대상
  1,391,131/1,391,131선 100% 유효. lower 3500 K는 두 번의 private
  bundle/ledger를 차단 없이 완료해 upper 140000 K 행렬까지 진입했다. 따라서
  lower population→tau/source→BF→A208/A209→원소별 internal energy→complete
  adiabatic 경로는 finite/valid다. 성공 numeric dump는 아직 미추출이다.
- [x] job `251575` 종료: H200 `syn104`, `FAILED 70:0`, elapsed 20분28초,
  MaxRSS 79,470,600 KiB. lower/upper 모두 complete bundle/ledger를 통과한 뒤
  최초 차단은 `RADEQ_NO_BRACKET`; 공개 Te/세대/material은 보존됐다.
  stdout/stderr/footer SHA는 각각
  `f399f8285acdeeb77a9faa1345ca205b54b3d8088f65cd70609936c8b7652a22`,
  `74139b5904ed1ed826b434b3ad4fb4b049af178975d0435190f3585c5b824b51`,
  `fb69868172c93766ea03b104fa4e295054b08d113b30d872611db4f186792eb4`.
- [x] 기존 vector no-bracket 경로가 원인 shell/항을 출력하지 않던 계측 부채를
  수리했다. 최초 shell, same-positive/same-negative/endpoint-zero 개수,
  lower/upper H/C/residual과 7 heating·6 cooling 항을 fail-only로 기록한다.
- [x] `LUMINA_RADEQ_DIAG=1`일 때 성공한 uniform lower/upper endpoint마다 shell별
  finite `T_e`, `n_e`, `n_atom`, energy density, `u_atom`, CMFGEN 단열 네 항과
  signed total, H/C/residual/e_balance를 보존하도록 추가했다. publication 권한이나
  solver 판정은 바꾸지 않는다. A2-10 vector no-bracket rollback과 직접 회귀 PASS.
- [x] finite endpoint/no-bracket 진단 후 CPU/OpenMP 강제 full link와 전체 battery
  D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. gate 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_endpoint_finite_no_bracket_diag_2026-08-09.log`,
  SHA `f510902ddb7dae48eccb1f18abf93c95ca8191262f4c0218543aa155d25c3b66`.
- [x] CUDA sm_80/86/90 fresh link, cubin 3종과 `make -q cuda` PASS. binary SHA
  `d39421fc85e4937bd1dcf7617907cb097911e17d74a6b7d6ba51907ea22f3de6`.
- [x] finite endpoint H200 diagnostic job `251579` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T230243Z_d39421fc85e4`;
  staged binary/reference SHA 일치, `ION_LOCK=0`, `PER_ION_RESCALE=0`,
  `single_total=1`, 불필요한 `MATDUMP`는 off. H200 `syn104` 실행 중;
  strict env 70/70, reference 6행, ion population 33슬롯 확인.
- [x] job `251579` lower 3500 K finite endpoint 50/50행 확보. `u_atom` 범위
  `4.0436137036146466e-11–4.3612193216074651e-11 erg/atom`, energy density
  `6.0490930890533921e-7–6.5201433745875573e-2 erg cm^-3`. residual은
  음수 12/양수 38/zero 0 shell. shell 0은 H=`6.8538971392572312e-2`,
  C=`3.975597529184093`, residual=`-3.9070585577915207`; shell 49는
  H=`1.4885316011811185e-5`, C=`4.9128395509610727e-6`,
  residual=`9.9724764608501123e-6`. finite 비영 물리값 재현 성립.
- [x] upper 140000 K도 finite endpoint 50/50행. `u_atom` 범위
  `3.1986364333207505e-10–4.225384547902891e-10 erg/atom`, energy density
  `5.8606877073752323e-6–5.1576559179691106e-1 erg cm^-3`; residual은
  음수 47/양수 3/zero 0 shell.
- [x] no-bracket은 15 shell: inner 0–11은 lower/upper 모두 cooling 우세,
  outer 47–49는 모두 heating 우세. 최초 shell 0은 lower residual
  `-3.9070585577915207`, upper `-1.4534167321473593e8 erg cm^-3 s^-1`.
  upper H=`2.1129639178304013e-3`, C=`1.4534167321684888e8`; cooling은 사실상
  `line_emit=1.4534167321534827e8`가 독점한다. bracket을 근거 없이 확장하지 않는다.
- [x] job `251579` 종료: H200 `syn104`, `FAILED 70:0`, elapsed 20분02초,
  MaxRSS 79,470,612 KiB. stdout/stderr/footer SHA는 각각
  `7f236928664a5da824dfb8e459ae1fa532247ed48f7b440a8623f5c9259bd29a`,
  `ff9004c4ba374d1b1964516c512fe57a7390c6a644e04b96728bd00424ab813f`,
  `4822337afa6c6f12467031e0980ef45b36bf04ddfaf5c9ad378503cc5f26b659`.
- [x] A209 line-emission 식의 `4pi*dnu` 왕복은 차원이 맞음을 확인. signed Sobolev
  계약은 `tau<0`에서 beta를 지수 증폭하며 clamp하지 않도록 사전등록돼 있다.
  추측 수정 대신 endpoint/shell별 total line emit, negative-tau emit fraction,
  `[-1e-6,0)`, `[-1e-2,-1e-6)`, `[-1,-1e-2)`, `<-1` count와 최대 기여
  `(line,Z,ion,levels,tau,beta,n_upper,A_ul,nu)`를 출력하는 forensic을 추가했다.
  진단 allocation 실패는 물리 publication을 바꾸지 않고 경고만 남긴다.
- [x] signed-tau forensic 이후 CPU/OpenMP 강제 full link와 전체 battery
  D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. gate 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_signed_tau_forensic_2026-08-09.log`,
  SHA `1b7ec3d1623b792b7e2d79251236e01d749eac0e2f54c76afe0adfb45b861e67`.
- [x] CUDA sm_80/86/90 fresh link, cubin 3종과 `make -q cuda` PASS.
  binary SHA `6d201447c69885384910ff0ba3086e574ab4bd4f5b2e221f1653d5075e76bcad`;
  `git diff --check` PASS.
- [x] signed-tau forensic H200 job `251580` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260808T233301Z_6d201447c698`;
  staged binary SHA 일치, `single_total=1`, `MATDUMP=0`, outer iterations 4.
- [x] job `251580` H200 `syn104` 배정·실행. H200 NVL 139.8 GB,
  strict env 70/70 known·unknown 0, `ION_LOCK=0`, `PER_ION_RESCALE=0`,
  `RADEQ_DIAG=1` 확인.
- [x] job `251580` 종료: H200 `syn104`, `FAILED 70:0`, elapsed 20분0초,
  MaxRSS 79,469,536 KiB. exact-sliding 45회 residual
  `9.8975866364090719e-09 < 1e-8`, negative recurrence 0, R6
  1,391,131/1,391,131선 100% valid. lower/upper signed-tau forensic 각 50/50행.
  stdout/stderr/footer SHA는 각각
  `20953b4506d1104055cac11727576dc02aed15f871ac3f70327acbbc32d104f1`,
  `2a1df949f1cd744648f2d5ee6f6fc74de77f7ec3dce53f5b926c8459eb72eb5f`,
  `777b5b5aaff2bcf0e92c9cae768a80ece66bb32042079f62fd382ab82fa6a5e9`.
- [x] lower 3500 K shell 0의 음의 tau 선 기여는
  `0.83924247611282132 / 3.975403493583793 = 0.21110875348058097`.
  최대 기여 Co III line의 `tau=-0.0130056`, `beta=1.006531`로
  약한 증폭이며 폭발적 maser가 아니다.
- [x] upper 140000 K shell 0의 음의 tau 기여는
  `9.6265934939489869e-13 / 1.4534167321524096e8 = 6.6234e-21`.
  따라서 upper line cooling 폭증의 signed-tau/지수 beta 가설은 실증 기각.
- [x] upper 최대 기여는 staged canonical row `14065`, Co VI
  `3d4_1Ie[6] -> 3d3(2H)4f_1Ko[7]`, 176.857818 A. H200은
  `n_upper=1.5045550849571085e5 cm^-3`, `tau=4.78814e-42`, `beta=1`,
  `A_ul=1.063668e11 s^-1`, 한 선 방출 `1.7974903388460749e6`.
  CMFGEN `COB/VI/19apr23/osc_data:5787`의 176.858 A, `A=1.0637e11`과 일치.
- [x] Co VI는 CMFGEN 정본에서 1000 full level/41 superlevel NLTE 원자지만
  현 Lumina NLTE target에는 없어 A209가 LTE@140000 K population을 쓴다.
  level 620 LTE fraction `1.6010500069801094e-4`, Co VI density
  `9.3973022603771818e8 cm^-3`로 canonical 74,553선을 독립 합산하면
  optically-thin `1.0265482477122988e8 erg cm^-3 s^-1`, 전체 line emit의
  `70.6300006738%`를 재현한다. 따라서 최초 구조 원인은
  **untracked high-ion LTE excitation을 A210이 확정 thermal cooling으로 소비**하는 경계다.
- [x] endpoint 동일부호가 내부 root 부재를 증명하지 못하는 문제를 닫기 위해,
  `LUMINA_RADEQ_DIAG=1` no-bracket 경로에 한 번의 완전한 private
  population→tau→BF→A208→A209→adiabatic 기하중간 온도 vector trial을 추가했다.
  3500–140000 K의 기하중간은 `22135.943621178667 K`로 CMFGEN 실제 Te 표의
  `8855.96–24600.30 K` 범위 안이다. shell별 midpoint H/C/residual/line emit과
  lower–mid, mid–upper bracket 여부를 기록하지만 bracket·publication·최종
  `RADEQ_NO_BRACKET` 판정은 바꾸지 않는다.
- [x] vector callback 정확히 3회(lower/upper/geometric-mid), 반환 rc=4,
  기존 공개 Te/세대 byte 보존을 직접 selftest로 고정. A2-10 N1–N8 8/8,
  `git diff --check` PASS.
- [x] A209가 실제 `n_upper`를 선택한 동일 authority predicate를 밖에서
  재추론하지 않고 그대로 반환해, 각 line-shell emit을 `NLTE_SE`,
  `LTE_UNMAPPED`, `LTE_MAPPED_UNOWNED` 세 상호배타적 소유권으로 분해한다.
- [x] lower/upper뿐 아니라 새 `GEOMETRIC_MID`에서도 shell별 ownership 합,
  fraction, closure와 Z/ion 상위 5개 및 remainder를 출력한다. 이는 진단 배열일
  뿐 `eta_bb`나 publication을 바꾸지 않으며 allocation 실패도 기존 물리
  판정을 바꾸지 않는다. candidate tau 전체 private bundle과 A2-09 직접 회귀 PASS.
- [x] midpoint/ownership 계측 이후 CPU 및 OpenMP full link PASS. 새 경고 없이
  기존 경고만 재현했고 `git diff --check` PASS.
- [x] 전체 gate battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_interior_owner_forensic_2026-08-09.log`,
  SHA `bb912e6da837b841f45fecfff1705cf387d0ebeda1775bac2634c70f40234fc8`.
- [x] CUDA sm_80/86/90 fresh link, cubin 3종과 `make -q cuda` PASS.
  binary SHA `91dfc87a9c4585fde3d38dc2f29a921e67c4d65d551da6e04d42c432b9d7dd75`.
- [x] midpoint/ownership H200 job `251594` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T001035Z_91dfc87a9c45`;
  staged binary SHA 일치, `single_total=1`, `MATDUMP=0`, outer iterations 4.
- [x] job `251594` H200 `syn104` 배정·실행 시작. staged binary SHA는
  `91dfc87a...d7dd75` exact match, strict env와 `RADEQ_DIAG=1`,
  `ION_LOCK=0`, `PER_ION_RESCALE=0` 확인.
- [x] job `251594` lower 3500 K 원장 완결: signed-tau 50/50,
  line-owner 50/50, Z/ion top5+remainder 300/300, endpoint finite 50/50.
  shell 0 line emit `3.975403493583793`의 `99.9999999999895%`가 실제
  `NLTE_SE`이며 LTE-unmapped는 `4.18776e-13`, mapped-unowned는 0이다.
  상위 contributor는 Co III `62.8306%`, Fe III `34.8672%`, Ni III
  `1.15535%`; lower inner cooling은 high-ion LTE가 아니라 NLTE ledger다.
  outer shell 49도 NLTE_SE `99.9999729080%`이고 S III가 `99.9632%`다.
- [x] job `251594` upper 140000 K 원장 완결: signed-tau 50/50,
  line-owner 50/50, Z/ion top5+remainder 300/300, endpoint finite 50/50.
  전 50개 shell에서 line emit의 `LTE_UNMAPPED` fraction이 정확히 출력
  `1.0`이며, `NLTE_SE` fraction은 `4.88982e-31–2.72550e-19`,
  `LTE_MAPPED_UNOWNED` 최대도 `6.35192e-19`에 불과하다. ownership closure는
  전 shell 0이다. 즉 upper 온도 방정식의 선원장은 사실상 전부 현 SE target
  밖의 LTE 준위가 소유한다.
- [x] upper shell 0 line emit은 `1.4534167321524096e8 erg cm^-3 s^-1`이고
  Co VI `70.6300007%`, Co V `18.59865496%`, Ni VI `10.17104315%`,
  Ni V `0.57960828%`, Fe VI `0.02039647%`다. 기존 문제선 line 14065
  (Co VI, lower 21→upper 620, `A=1.063668e11 s^-1`,
  `nu=1.695104e16 Hz`)은 `n_upper=150455.5085 cm^-3`, emit
  `1.7974903388e6 erg cm^-3 s^-1`로 여전히 최대 단일선이다. shell 0
  A210 residual은 `-1.4534167321473593e8`; endpoint 부호 판정은
  no-bracket 15개(같은 음수 12, 같은 양수 3, endpoint zero 0)를 재현했다.
- [x] job `251594` geometric midpoint `22135.943621178667 K` 원장 완결:
  signed-tau 50/50, line-owner 50/50, Z/ion top5+remainder 300/300.
  line emit은 shell 0 `2333.7730562942347`에서 shell 49
  `0.43016019873497624 erg cm^-3 s^-1`까지 모두 finite다. 음의-tau emit
  fraction 최대는 shell 0의 `1.63377e-6`으로 midpoint 과냉각의 원인이 아니다.
- [x] midpoint line emit도 전 shell에서 LTE-unmapped fraction
  `0.9999876235–0.999999999985`, NLTE-SE fraction
  `1.98088e-12–1.23765e-5`다. shell 0은
  Co IV `74.4285%`, Fe IV `13.6424%`, Ni IV `11.9275%`; shell 47–49는
  S V가 각각 약 `99.78%`를 차지한다. 따라서 midpoint residual도 현 SE가
  사실상 소유하지 않는 고이온 LTE 선냉각에 의해 결정된다.
- [x] no-bracket 15개 중 shell 47–49는 midpoint residual이 음수라
  lower–mid와 mid–upper 양쪽에 각각 bracket이 생겼다. 그러나 shell 0–11은
  lower/mid/upper residual이 모두 음수여서 `still_same_sign=12`다.
  한 점 adaptive bracket 승격은 전 vector를 풀 수 없으므로 기각하고
  `RADEQ_NO_BRACKET`, 공개 Te/generation byte 보존, fail-closed rc=4를 유지한다.
- [x] job `251594` 종료: H200 `syn104`, elapsed `00:29:55`, expected
  `FAILED 70:0`, MaxRSS `79,473,576 KiB`. stdout/stderr SHA는 각각
  `82b281b57bee7ec34d0e4b34ba81efb38707b0be57b2e7976c511c2852206071`,
  `b721f795b8896996da38600dfc519005b6bf5811a2e11c4b5617fac2aeaae0a6`.
- [x] high-ion asset census: active deck에는 S IV/V, Fe IV/V/VI, Co IV/V/VI,
  Ni IV/V/VI의 finite level·line·IP가 모두 있다. 대표적으로 Fe
  IV/V/VI=`1000/1000/2000` levels와 `72223/72213/185392` lines,
  Co IV/V/VI=`1000/1000/1000` levels와 `69803/75923/75118` lines,
  Ni IV/V/VI=`1000/1000/1000` levels와 `73012/76049/79169` lines다.
- [x] collision census: Fe/Co/Ni IV는 CMFGEN collision table asset이 있으나
  Fe/Co/Ni V–VI 대부분은 원본 `col_data`가 0 transitions라 manifest가
  CMFGEN 근사식 경로 필요를 명시한다(Fe VI 3160 mapped transition은 예외).
  따라서 V/VI 슬롯 단순 추가는 아직 물리 폐합이 아니다.
- [x] sealed DET submitter에 기본 OFF인 `DET_NLTE_STAGE4=0/1` control을 추가.
  invalid value는 staging 전 `rc=70`, ON arm은 resolved env에 정확히
  `LUMINA_NLTE_STAGE4=1`, OFF arm은 변수가 없음으로 검증한다. `bash -n` PASS.
- [x] Stage-IV 단일변수 A/B job `251595` 제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T004545Z_91dfc87a9c45`;
  binary SHA `91dfc87a...d7dd75`, sigma SHA `90d04042...c3ad`,
  `single_total=1`, `ION_LOCK=0`, `PER_ION_RESCALE=0`, `STAGE4=1`,
  `RADEQ_DIAG=1`.
- [x] job `251595` H200 `syn104` 배정 및 Stage-IV wiring PASS: 38 slots,
  23 pairs, `16076 FL→1537 SL`, mapped lines `1993694/2588798`.
  Fe/Co/Ni IV는 각각 1000 levels, Si IV는 61 levels로 실제 활성화됐다.
  Ti/Cr/Al IV 슬롯은 active deck에서 0 levels이므로 기존 empty-slot/skip 경계를
  finite lower trial에서 계속 검증한다.
- [x] Stage-IV 공개 preflight PASS: population, tau/source, A208/A209가 모두
  valid하고 A210 private trial에 진입. exact-sliding 45회 residual
  `9.8975866364090719e-9 < 1e-8`; 새 Q-set `1603732`선은 100% valid/Jbar
  sampled이며 all-line coverage는 `61.948904%`다.
- [x] Stage-IV lower private trial은 line ledger 전 Si III-IV pair에서
  fail-closed했다: `T=3500 K`, shell 4, pair slots `1:2`, 162차 solve,
  음수 29개, min `-1.2754336776440669e-11 cm^-3`, vector max
  `2.6341314943753902e7`, 상대크기 `4.84195144e-19`. rank 출력은 162,
  equilibration 10회/refinement 2회, backward error `6.89247e-17`이다.
  작은 backward error만으로 부호를 버리지 않으며 floor/clamp는 금지한다.
  lower signed/owner/finite 원장은 0/50으로 Stage-IV A/B는 아직 불합격이다.
- [x] 동일 binary/deck/Stage-IV/single-total 조건의 matrix-capture job `251596`
  제출. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T005908Z_91dfc87a9c45`;
  `MATDUMP=1`, target `Z=14`, lower ion0=`2`, shell=`4`가 sealed env에
  정확히 기록됐다.
- [x] job `251595` 종료: H200 `syn104`, `FAILED 70:0`, elapsed `18:21`,
  MaxRSS `79,476,176 KiB`, comparison commit 0. lower는 위 미세 음수,
  upper 140000 K도 같은 Si III-IV pair가 shell 1에서 `ASSEMBLY_FAILED`로
  차단됐다. A210은 `RADEQ_TERM_SCHEMA`로 종료했고 Te generation/manifest는
  정확히 보존됐다. stdout/stderr/footer/resolved-env SHA는 각각
  `e0adce1d...6c0ef`, `79f3b485...54d5`, `c1642efb...fc9f`,
  `782ba3f5...c0ab`.
- [x] matrix job `251596`가 H200 `syn104`에서 실행 시작. 기준 arm과 동일한
  binary/sigma 및 Stage-IV/single-total 봉인을 재확인했다.
- [x] `scripts/analyze_nlte_roundoff.py`를 기존 two-stage-lock 출력 불변으로
  유지하면서 single-combined-total 자동 판별, exact-generator diagonal 재합산,
  80자리 solve, directed SCC/closed-class census로 확장. 기존 job 251516 dump는
  two-lock `changed_rows=2`를 재현했고, 합성 4-state/two-closed-class 검출 PASS.
- [x] CMFGEN `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern`은 `FIX_T=T`인
  고정온도 진단 run임을 `RVTJ`/`VADAT`/`MODEL` 세 파일에서 교차 확인했다.
  따라서 이는 자유 온도근의 정답이 아니며, 주어진 유한 온도에서의 NLTE
  population snapshot으로만 사용한다. `SCL_LN=T`, `SCL_LN_FAC=0.5`도 기록하되
  발동 조건이 확인되지 않은 상태에서 Lumina line cooling에 0.5를 곱하지 않는다.
- [x] 로컬 CMFGEN 2025-06-18 source의 `cmfgen_sub.f`와
  `update_ba_for_line.f`를 직접 추적해 `SCL_LN_FAC=0.5`의 의미를 닫았다.
  이는 line cooling을 0.5배 하는 계수가 아니라, superlevel 평균 에너지차/실제
  line 에너지로 계산한 `SCL_FAC`을 채택할지 정하는 허용편차다:
  `abs(SCL_FAC-1)>0.5`이면 오히려 `SCL_FAC=1`로 되돌린다. 따라서 0.5 blanket
  보정은 물리적으로도 코드상으로도 기각한다. 실제 line별 scaling 검증에는
  현재 누락된 동세대 `LINEHEAT`가 필요하다.
- [x] Lumina shell 0의 `v=4264 km/s`를 CMFGEN RVTJ depth 67/68
  (`4394.1823505/4092.3914953 km/s`) 사이에서 선형 보간하면
  `T_e=18760.319890793377 K`다. `POPCOB` CoSIX 90 depth x 1000 level을
  depth-major로 직접 합산·보간한 유한값은 Co VI 전체
  `0.047515604155321307 cm^-3`, CMFGEN level 621
  `3d3(2H)4f_1Ko[7]`은 `1.7073108691247696e-22 cm^-3`, 상준위 fraction은
  `3.5931582886830803e-21`이다. 이 수동 해석은 저장소의 독립 정식
  `scripts/cmfgen_extract/parse_pops.py`로 다시 읽어 depth 67/68 각각
  `(n_ion,n_621)=(0.0102032004458591,1.1141509e-23)`,
  `(0.096701425295826343,3.8110468e-22)`가 bit-for-print 일치했다.
- [x] 이에 비해 Lumina upper endpoint 140000 K의 동일 Co VI ledger는 이온
  `9.3973022603771818e8 cm^-3`, zero-based level 620
  `1.5045550849571085e5 cm^-3`, LTE fraction `1.6010500069801094e-4`다.
  서로 온도가 다르므로 동일-T 정확도 비가 아니라 **유한 구조 대조**이지만,
  CMFGEN NLTE fraction/Lumina LTE fraction=`2.2442511308316177e-17`로
  현 high-ion LTE line ledger가 CMFGEN NLTE ledger와 동종이 아님을 직접 확인한다.
  source SHA: `POPCOB=ef537e29...a96e16`, `RVTJ=ab7d0275...e028`,
  `VADAT=f516c90a...2238`, `GENCOOL=b7428b72...b1b`,
  `STEQ_VALS=2bb05580...9fd`.
- [x] 이 finite 비교의 자격을 재감사했다. CMFGEN modern 마지막 iteration은
  `FIX_T=T`일 뿐 아니라 population solve도 미수렴이며, 원본 `OUTGEN`의 반환
  `MAXCH=1e7%`, `STEQ_VALS`의 큰 correction이 남아 있다. 또한 GENCOOL↔RVTJ는
  T/r/v가 출력 정밀도에서 맞지만 electron density가 90 depth 중 49개에서 최대
  `4.512%` 어긋난다는 기존 manifest가 있다. 따라서 위 Co VI 값은 **실제 유한
  snapshot 비교/context evidence**이지 정확도 PASS용 oracle이 아니다. 현 LTE
  고이온 ledger와 CMFGEN식 NLTE snapshot이 현격히 다르다는 구조 가설을 지지하지만,
  수렴 CMFGEN 동일-T 재현을 대신하지 않는다.
- [x] 더 최신 O-PHYS STAGE-1
  `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys`도 감사했다. 동세대
  `POPCOB`와 2.73 GB `LINEHEAT`는 존재하지만 iteration 103의 population 변화가
  증가 `9.03e4%`, 감소 `1.07e6%`, 반환 `MAXCH=1e7%`이고 formal moment solve도
  excessive-iteration 오류를 냈다. 따라서 O-PHYS도 유한 context 산출물일 뿐
  accuracy oracle이 아니다. source SHA는 `RVTJ=893775d9...f715`,
  `POPCOB=c21f9820...8d4a`, `GENCOOL=8bad1c95...5d98`,
  `LINEHEAT=e533a503...59c8`, `OUTGEN=12717f50...4451`이다.
- [x] O-PHYS `LINEHEAT`에서 문제의 동일 Co VI 전이를 직접 찾았다:
  CMFGEN line 9901, `3d3(2H)4f_1Ko[7]-3d4_1Ie[6]`,
  `nu=1.6951044e16 Hz`. 실제 line별 `SCL_FAC=1.00872`이므로 0.5 blanket
  scale 가설은 이 전이 자체에서도 반증된다. 세 90-depth 배열을 정확히 분리했고
  depth 67/68 첫 배열은 `-5.2265e-15/-1.6462e-13`, v=4264 보간은
  `-7.3983456e-14`다. 다만 LINEHEAT는 인라인 단위가 없고 run도 비수렴이므로
  이 배열값은 cgs accuracy 비교에 사용하지 않는다. 같은 O-PHYS POPCOB의
  Co VI 전체/upper population도 depth 67에서
  `0.00919528394145/1.1129905e-23`, depth 68에서
  `0.087328387298/3.7364096e-22 cm^-3`로 유한함을 독립 parser로 확인했다.
- [x] job `251596` 종료: H200 `syn104`, expected `FAILED 70:0`, elapsed
  `18:20`, MaxRSS `79,476,600 KiB`. lower/upper pre/post 162차 행렬을 모두
  회수했다. lower pre/post SHA는
  `c5362106fb60d74a7d9c759fea9ba6b510084a0f1b40e1c9ced78c99c4e37a20` /
  `75660481e8e2ad04463aff29f1b8d0849f830794b964be9621f8b9ba8f7f65f2`,
  upper는
  `030daedbb55a85a34784555baefcffdffb7408650f7940dc0357609f3da8e8fb` /
  `f8b909abc83b62e7e4961f753a60e28ce8eb0062de5851cbb1db336d60a2c412`다.
  stdout/stderr SHA는 `f023a255...d148` / `84a705d3...a7674`다.
- [x] 두 preconstraint 행렬 모두 RHS=0, 음수 off-diagonal=0, 162 state 전체가
  하나의 SCC이자 하나의 closed class인 irreducible homogeneous generator다.
  lower raw 열합 상대오차 최대 `4.5782e-16`, upper `3.4583e-16`이며 대각
  절대 상쇄오차는 각각 최대 `1.9231e-6`, `1.7454e-6`다.
- [x] lower post 행렬의 일반 고정밀 solve는 11개 음수, min
  `-1.2695790142e-12`; float64 대각을 재합산해도 24개 음수가 남는다. 그러나
  imported 양의 off-diagonal을 80자리에서 정확 합산한 generator 정상상태는
  음수 0, min `1.4276152500892796e-16`, Si III/IV 합
  `26458377.684286512 / 974287.3570324968 cm^-3`다. 즉 물리적 음수가 아니라
  큰 outflow 대각합의 반올림과 일반 LU forward sign 전파다. lower JSON SHA
  `81e25bc9d710d87ee6099d0b06240ef7422ec258f89785e77609788714ea6966`.
- [x] upper는 기존 double부터 음수 0이며, exact-generator 80자리 해도 min
  `3.2013992855783896e-18`, Si III/IV 합
  `1.7119317692731112e-6 / 1.4283550387855277e-5 cm^-3`로 일치한다. upper
  JSON SHA
  `0980ffd8dc0821bf9f03a6870ddaa5aa8d654ea8387cb346de79cefe2e0209b2`.
- [x] long-double continuous-time GTH 커널을 population 계약에 추가했다.
  대각을 해에 쓰지 않고 nonnegative off-diagonal만 상태축소하므로 clamp/floor/
  `exp(z)` 양수강제가 아니다. 음의 off-diagonal은 적용 거부, reducible two-class
  generator는 `POP_RANK_INCOMPLETE`로 fail-closed, 40-decade 합성 정상분포는
  strictly-positive known answer를 재현했다. A2-07 selftest PASS.
- [x] 실제 job `251596` dump를 새 C 커널에 직접 입력한 결과 lower/upper 모두
  exact-generator 80자리 해와 출력 정밀도까지 일치했다. 성분별 잔차는 각각
  `6.89849e-17`, `7.78523e-17`; 입력 행렬 byte는 보존한다.
- [x] 생산 CPU pair solve는 single-total의 preconstraint 행렬을 별도 보존하고
  `one positive all-ones normalization row + zero generator RHS + recognized
  generator`일 때만 GTH를 쓴다. time-dependent RHS, b_k transform, two-stage
  lock, anchor/negative off-diagonal은 적용 대상이 아니다. 기존 general dense
  solve는 비생성자 fallback으로 보존한다.
- [x] `population_error_count` 전역 before/after 차이를 한 shell의 assembly
  status로 해석하던 OpenMP 경합을 제거했다. assembler의 직접 return만 local
  assembly 판정에 사용한다. job `251596` 양 endpoint의 허위
  `ASSEMBLY_FAILED`와 정확히 대응한다.
- [x] 수리 후 `git diff --check`, A2-07 GTH known-answer, private population,
  candidate adiabatic/tau, A2-10 N1–N8, CPU/OpenMP full link PASS. 기존 경고 외
  신규 컴파일 경고는 없다.
- [x] CUDA sm_80/86/90 fresh link PASS. 새 `lumina_cuda` SHA
  `a403d49d8f610b7a6d1f94999209b3c3eb01e3b38823aa6370b12f7d5e35c32b`;
  cubin 30개에서 세 arch를 확인했고 `make -q cuda`, header 28/28 PASS.
- [x] 전체 gate battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_generator_gth_stage4_2026-08-09.log`,
  SHA `2947c8b0e86ed96dd2c12030f6d82c57ac3ef8808f24b802b3522c494271bf5b`.
- [x] GTH 수리 sealed H200 job `251599` 제출·실행. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T013602Z_a403d49d8f61`;
  H200 `syn104`, binary/sigma SHA 일치, `single_total=1`, `STAGE4=1`,
  `ION_LOCK=0`, `PER_ION_RESCALE=0`, `MATDUMP=1`, target Si III-IV shell 4를
  staged manifest에서 재검증했다.
- [x] job `251599` lower target에서 생산 GTH 진입 확인. 3500 K Si III-IV
  shell 4는 total `27432665.041319009`, min
  `1.4276152500892798e-16`, max `26341314.923965145`, 음수 0이며
  input-column 상대오차 `9.15758e-16`, exact-generator residual
  `6.89849e-17`이다. 이전 미세 음수와 허위 `ASSEMBLY_FAILED`가 모두 소실됐다.
- [x] lower의 새 최초 차단은 그 뒤 pair 4 Fe III-IV, shell 44의
  `RANK_INCOMPLETE`다. general fallback 출력 rank `102`, equilibration 10회,
  refinement 0으로 Si 상쇄 문제와는 별개의 topology/연결성 결손이다. Fe target
  pre/post 캡처가 다음 진단 대상이다.
- [x] job `251599` upper target도 생산 GTH PASS. 140000 K Si III-IV shell 4는
  total `1.5995482157128386e-5`, min `3.2013992855783896e-18`, max
  `1.4282074894562855e-5`, 음수 0이며 input-column 상대오차
  `6.9168827899629077e-16`, exact-generator residual
  `7.7852327912861914e-17`이다. upper에서도 허위 `ASSEMBLY_FAILED`는 없다.
- [x] upper의 다음 최초 차단도 Fe III-IV pair 4이며 shell 46에서 동일하게
  `RANK_INCOMPLETE`, rank `102/202`, equilibration 10회, refinement 0이다.
  따라서 Si 수리는 endpoint 독립적으로 성립하고 Fe 결손은 재현 가능한 별도
  구조 문제다. `202`는 runtime `SUPER_CUTOFF=100`이 Fe III/IV 각각을
  101 state로 투영한 값이며, 원본 CMFGEN superlevel `105+63=168`과 구분한다.
- [x] job `251599` 봉인 완료: H200 `syn104`, expected `FAILED 70:0`, elapsed
  `18:59`, MaxRSS `79,485,196 KiB`. stdout/stderr/footer/resolved-env SHA는
  `c044a68b30ea297ac95f0d4813d1da4330b79920c89103a1f833d34b0d7ca368`,
  `cc051e5399e23371016d8c9f89317052eea2bab39cf7371b0a0a18d62d63d2d2`,
  `731b5773de55dfc6b30ef0cd083c32b2c2271a442216836310c2a57bc0115690`,
  `c1f7315a9d1d4467b49c1e0aa5406a5f91a0d12f524a5af838f7cdfbcec75e32`다.
  lower pre/post matrix SHA는 `c5362106...e37a20` / `75660481...f65f2`,
  upper는 `030daedb...e8fb` / `f8b909ab...c412`로 job `251596`과 정확히
  동일하다. staged binary SHA도 `a403d49d...c32b` exact match다.
- [x] 동일 binary/deck/single-total/Stage-IV 조건의 Fe rank matrix-capture job
  `251601` 완료. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T020237Z_a403d49d8f61`;
  H200 `syn104`, expected `FAILED 70:0`, elapsed `19:06`, MaxRSS
  `79,482,924 KiB`. target은 `Z=26`, lower ion0=`2`, shell=`44`다.
  staged binary SHA `a403d49d...c32b`, sigma SHA `90d04042...c3ad`,
  `ION_LOCK=0`, `PER_ION_RESCALE=0`, `single_total=1`, `STAGE4=1`,
  `MATDUMP=1`을 재검증했다. stdout/stderr/footer/resolved-env SHA는
  `4fe9c245573a55df4819c18623e716c28e81492ca9c2c979186a5546091329d4`,
  `1e86e822221ab83d34fab5bcf5f6253a03e09c734e985ee5f2951603f943891f`,
  `6ec9a4b8e21ff4dfc12efac780f5b0ba575f076bf8a3634e083b1e01710c9c5a`,
  `4f3965d63207a0f31f2bff07707825433537a488557bb250a428c83240290094`다.
- [x] `251601` lower/upper shell 44 prelock RHS와 single-total normalization
  RHS는 모두 비트 단위 `0.0`이다. 양 행렬 모두 N=`202`, directed edge=`3082`,
  SCC=`1×202`, closed class=`1`, zero-in/out state=`0`, 음의 off-diagonal=`0`인
  완전 연결 homogeneous generator다. 따라서 `rank 102`는 결손 물리율이 아니라
  exact-zero 총량을 일반 dense solve로 보낸 분기 오류다. lower/upper 분석 JSON
  SHA는 `c8828dd7a2fd45454e44efcfbbea958634e1b8016bb74be093dd10bbcdeb75b2`,
  `dd25e877f039ac5cd099d051595020e5c4d07408f51c4911049e2665b8ee261b`다.
- [x] 동일 lower pre-generator를 실제 C GTH에 넣어 total=`1`과 `1e20` 모두
  rc=0, 202개 finite positive를 재현했다. total=`1` residual
  `7.79310569895104e-17`, min `4.235764078220075e-15`, max
  `0.9204718511819489`다. GTH/kernel/topology는 정상이다. 단, overlapping pair는
  최종 저장 때 upstream per-ion total로 rescale하고 lower shared stage를 앞 pair
  값으로 복원하므로, 이 결과는 multi-stage ion fraction 소유권까지 증명하지 않는다.
- [x] `nlte_zero_total_pair_exact_zero`를 production pair solve의 조립 전 경계에
  추가했다. 오직 `conservation_total == 0.0`일 때만 두 stage의 해당 shell
  population/ion density를 정확히 0으로 만든다. threshold·floor·clamp는 없으며,
  positive subnormal `1e-300`은 그대로 solve 경로에 남고 음수/비유한/누락 layout은
  fail-closed다. private candidate transaction과 기존 overlapping pair restore는
  그대로 유지한다. 판정 원장은 `docs/NLTE_ZERO_TOTAL_PAIR_REPAIR_2026-08-09.md`다.
- [x] exact-zero/neighbor-invariance/subnormal/negative selftest, CPU/OpenMP full
  link, A2-10 N1-N8, `git diff --check`, 전체 gate D 19/19·K 7/7·Z 12/12·CP
  4/4 PASS. gate log SHA는
  `61d123ab23a0808feeb1023ce540c225cbf931c227d2908a250ffca623a0ee80`다.
- [x] fresh CUDA SHA
  `3e38b9cd0750d4d36e1fac0b94c39b94bb84eca029522aa42c305360a7c26a98`,
  sm_80/86/90, `make -q cuda=0`, header 28/28를 봉인했다.
- [ ] 수정 H200 job `251622` 실행 중. run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T022526Z_3e38b9cd0750`;
  동일 deck/sigma/single-total/Stage-IV, target Fe III-IV shell 44다. 목적은
  `[PAIR-EXACT-ZERO]` production 분기와 양 endpoint에서 Fe 뒤 최초 차단점을
  확인하는 것이다.
- [x] job `251622` lower 3500 K에서 Fe III-IV shell 44
  `[PAIR-EXACT-ZERO]`를 두 CE pass에 실제 기록했다. 구형 binary의
  `RANK_INCOMPLETE/ASSEMBLY_FAILED`는 재발하지 않았고, lower 50/50 shell 모두
  `[A2-10][ENDPOINT-FINITE]`로 `n_e`, atomic energy density, 네 signed adiabatic
  합, heating/cooling/residual을 유한하게 생산했다. residual sign은 shell 0–11
  negative, 12–49 positive이며 upper와 짝지어 bracket을 판정해야 한다.

다음 5단계:

1. job `251622` upper 140000 K 50/50 finite와 exact-zero 분기를 확인한다.
2. lower/upper residual sign으로 shell별 bracket/no-bracket을 분류한다.
3. 최초 새 차단 또는 root-solve 상태를 판정한다.
4. 실행 원장·stdout/stderr/footer/resolved-env·staged binary SHA를 봉인한다.
5. finite Stage-IV final-owner ion fraction을 CMFGEN 원본과 직접 비교한다.

Stage-IV ownership 정적 census는
`docs/STAGE4_PAIR_OWNERSHIP_AUDIT_2026-08-09.md`에 봉인했다. 8개 shared stage
모두 earlier pair가 최종 shared-level owner이고, 16개 overlapping pair call은
upstream per-ion total로 rescale된다. H200 finite cell에서 이 규칙의 동적 결과와
CMFGEN ion fraction을 대조해야 4단계가 완료된다.

완료된 현재 구조:

- [x] CMFGEN `EVAL_ADIABATIC_V3` 대응 signed 네 단열항과 원자 내부에너지.
- [x] 한 trial Te vector의 private population→tau→BF→A208→A209→adiabatic bundle.
- [x] 전 shell 동시 vector residual/root candidate와 수렴점 최종 replay.
- [x] replay ledger byte 일치 후 Te/ne/population/tau/BF/A208/A209/A210 단일 commit.
- [x] MC·순수 CMFGEN 루프의 commit 직후 중복 plasma/NLTE solve 제거.
- [x] 실패 preflight에서 공개 material/publication byte 불변 회귀시험.
- [x] A2-10 N1–N8, 단열·내부에너지·candidate·single-commit selftest PASS.
- [x] CPU/OpenMP 및 CUDA sm_80/sm_86/sm_90 compile+link PASS; binary 실행 안 함.
- [x] D 19/19, K 7/7, Z 12/12, CP 4/4 전체 gate battery PASS.
- [x] A2-17 include-order 의존 제거와 bit-identical canonical edge fixture 복구.
- [x] 세대·단위·4π·부호를 선언하는 DET/MC 물리 비교 dump + manifest.
- [x] shell-volume×frequency-width 적분보존 comparator와 음성대조 4종.
- [x] SH-GRID 상한 자산 폐합: 1234 bins,
  `[5.8412785919616062e13,4.0362581455823112e16] Hz`, 구 dlog bit 동일.
- [x] canonical radiation field 3866 bins, offset `[-1398,2468]`, BF↔canonical 왕복 오차 0.
- [x] active 덱 24,542-level CMFGEN σ를 원본 phot data에서 재평가; 1000-bin padding/first-fill 0.
- [x] GPU injection CDF의 1024-bin clamp 제거; CPU/OpenMP/CUDA sm_80/86/90 compile+link PASS.
- [x] 새 하한 아래 default-active BF threshold 0; 종래 전체 gate battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS.
- [x] type 2/3/8/9 stand-in을 CMFGEN `SUB_PHOT_GEN` exact evaluator로 교체; 7,266행 변경.
- [x] canonical σ SHA `90d04042c17bcc5f2c7c521b65a9bb0f824179d79493f82ad40deaa7185cc3ad` 봉인.
- [x] full-bin average의 부분 임계빈 이중 절단 제거; actual-frequency sharp edge는 유지.
- [x] 신규 저주파 band 707 level×90 depth 동일-snapshot rate/eta 폐합 PASS;
  상한 Si V photo-rate 정확도는 별도 OPEN.
- [x] active 덱 D 19/19, K 7/7, Z 12/12, CP 4/4 및 CUDA sm_80/86/90 PASS.

현재 판정:

- 예전 `BLOCKED_INCOMPLETE_ADIABATIC`은 legacy 음성대조 fixture의 예상 결과이지,
  production A210의 현재 차단 사유가 아니다.
- production transaction은 구조적으로 닫혔다.
- SH-GRID 저주파 신규 band의 rate/eta discretisation은 닫혔다. 상한 Si V photo-rate는
  현재 Wien-tail snapshot에서 정확도 미폐합이며 영향만 negligible이다. 이를 CMFGEN
  정확도 일치로 부르지 않는다.
- 실제 CMFGEN/ARTIS 전체 물리 일치 주장은 아직 금지한다. 다음 증거는 DET 수렴 flight다.
- active 덱 quarantine 전체 검증은 SH-GRID 형상/Q1·Q2·Q3·Q5를 통과했다. Q4의
  2,588,793 line-value 불일치는 기존 I20 직렬화 대 CMFGEN 원값 문제로 격자와 독립이다.
- vector solver는 모든 shell을 같은 trial vector로 평가하지만 componentwise bracket을 쓴다.
  강결합 비단조 문제의 보편적 Newton solver라는 주장은 하지 않는다.

CMFGEN/ARTIS 비교 가능 시점:

1. 지금: known-answer와 동일 입력 replay에 대한 단위·부호·원자성 비교.
2. 비교 dump 연결 직후: 고정 snapshot의 `u_atom`, 단열 네 항, `chi_nu`, `eta_nu`,
   A210 `H-C` 비교. 이는 solver 수렴 판정이 아니다.
3. SH-GRID migration과 DET 수렴 뒤: shell별 `T_e`, `n_e`, 이온분율 및 에너지 잔차.
4. MC 수렴 뒤: `J_nu`, heating estimators, packet energy와 spectrum. ARTIS는 현재
   출력에 exact adiabatic accumulator가 없어 그 항은 직접 대조하지 않는다.

방금 완료한 단계 — A210 BF-OOG runtime provenance와 상한 전수 조사:

- [x] Slurm job `251512`, H200 `syn104`, `FAILED 70:0`, elapsed 5분58초.
- [x] run root:
  `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T132730Z_6db79943e140`.
- [x] exact sliding 45회/residual `9.8975819498016093e-09`와 R6 Q_g
  1,391,131선 전 영역은 다시 PASS했다.
- [x] 양 endpoint의 첫 실패는 동일하다: shell 0, Si V→VI, global level 333,
  level 0, `166.7674257419 eV`, `4.0324184137410968e16 Hz`, `74.3455731128 Å`.
- [x] BF state는 `OUT_OF_GRID`, 기존 1178-bin baked CMFGEN sigma row는 없어 Kramers
  `sigma0=7.91e-20`을 선택했다. `w_miss=0`은 kernel overlap 자체가 없기 때문이다.
- [x] 활성 abundance 원소 전 stage와 level을 조사한 결과 현 100 Å 상한 밖 양의
  threshold는 이 Si V ground 한 건뿐이다.

방금 완료한 단계 — SH-GRID 상한 설계 판정:

- [x] 기존 A2 frequency union은 limiting BF edge를
  `4.032418413741097e16 Hz`로 이미 등록했다.
- [x] canonical radiation field의 sealed 상한
  `4.0362581455823112e16 Hz`는 구 dlog의 정확히 56칸 위이며 Si V edge보다
  `9.5221563023573097e-4` 높다.
- [x] 1178→1234 bins로 확장하면 dlog가 bit 동일하고 `K*N=2468=J_HI`가 되어
  canonical absolute grid/3866 bins/edge identity는 바뀌지 않는다.
- [x] ARTIS 10 Å superbin 외삽이나 OOG exact-zero는 채택하지 않는다. 기존 등록
  canonical support에 BF/NLTE 소비자를 맞추는 최소 수리다.
- [x] fresh rebake는 linked CMFGEN phot source에 있는 Si V ground row를 찾아냈다.
  기존 bake의 `has_cmfgen=0`은 source 부재가 아니라 old upper-edge skip이었다.
- [x] 첫 prefix 승격 gate에서 sigma baker의 별도-log 차감 산식이 runtime ratio-log
  산식과 3 ULP 어긋난 기존 결함을 발견했다. 새 baker를 runtime 산식으로 통일하고,
  prefix는 support 동일·최대 상대변화 `≤1e-7`·물리 폐합으로 검증한다.
- [x] 구현·자산 승격·flight 조건을
  `docs/SH_GRID_UPPER_CLOSURE_CONTRACT_2026-08-08.md`에 사전등록했다.

방금 완료한 단계 — A210 trial-consistent private ionization owner:

- [x] A210 후보가 공용 `n_e`와 ion-stage totals를 trial `T_e`에 그대로 복제하던
  누락을 확인했다.
- [x] 공용 ion-stage 경로의 canonical BF 소유자를 명시적 `NLTEConfig *` 인자로
  분리해 public path는 공용 owner, candidate path는 candidate-local owner를 쓴다.
- [x] candidate partition/within-SL을 trial `T_e`에서 먼저 만들고, 같은 rate ladder로
  private `n_e`와 ion totals를 재계산한 뒤 기존 두 ion-lock 행에 공급한다.
- [x] 이 substage는 generation을 올리지 않으며 clamp/floor/pin/fallback을 추가하지 않는다.
- [x] 재계산 전후 최대 `n_e`/ion 변화와 전하잔차를 candidate-local 원장에 남긴다.
- [x] one-ion known answer에서 ion total `1,2 -> 12,24`, public ion/ne byte 불변,
  candidate population/tau selftest PASS.

방금 완료한 단계 — trial-ionization 정적/회귀 gate:

- [x] 첫 전체 battery가 public bootstrap의 null BF owner 회귀 한 건을 검출했다.
  candidate는 explicit owner를 필수로 유지하고, public iteration-0 Saha bootstrap만
  종전처럼 null owner를 허용하도록 수리했다.
- [x] 재실행 결과 D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a210_trial_consistent_ionization_v2_2026-08-08.log`.
- [x] CPU/OpenMP, candidate population/tau/adiabatic, A2-07/A2-10, exact sliding PASS.
- [x] Makefile header census 28/28, `git diff --check`, CUDA fat binary
  sm_80/sm_86/sm_90 compile+link PASS.
- [x] 새 H200 binary SHA:
  `0b36500896cea5125c7f7de4eee1e38b968954bd41408bdd26dc8512115b3e81`.

완료한 trial-consistent ionization preflight:

- Slurm job `251510`, H200 `syn104`, `FAILED 70:0`, elapsed 6분00초.
- run root: `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T131344Z_0b36500896ce`.
- exact sliding 45회/residual `9.8975819498016093e-09`와 R6 Q_g 1,391,131선
  전 영역은 PASS했다.
- lower `3500 K`와 upper `140000 K` 모두 shared level-SE 이전의 candidate
  `IONIZATION` 단계에서 `POP_BF_OOG`로 fail-closed했다.
- 이는 기존 bootstrap ion totals를 복제하던 경로가 감추던 첫 trial rate-SE coverage
  공백이다. 현재 진단은 shell/pair/level을 특정하지 못하므로 값을 대입하거나 grid를
  임의 확장하지 않고 BF provenance를 먼저 추가한다.

방금 완료한 단계 — candidate ionization BF provenance:

- [x] 실패 shell·element·Z·ion-pop pair·ion stage와 trial `T_e/n_e`, ionization
  energy를 candidate-local 원장에 추가했다.
- [x] 실패 global level/level number/level energy, BF validity state, CMFGEN σ 보유 여부,
  row 최대 σ, threshold frequency, missing-weight를 함께 기록한다.
- [x] public 경로는 진단 포인터 `NULL`이며 값·counter·출력에 변화가 없다.
- [x] candidate population/tau, A2-07, CPU/OpenMP, CUDA sm_80/86/90,
  header 28/28, `git diff --check` PASS.
- [x] D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a210_bf_oog_provenance_2026-08-08.log`.
- [x] 새 diagnostic binary SHA:
  `6db79943e1405dd044ede26268323dd253b6e788f12b7f6e64f122baf727ad06`.

방금 완료한 단계 — Q_g coverage census:

- [x] current Q_g = 1,777,859선; current 1000–4000 A valid = 533,172선과 smoke #5가 정확히 일치.
- [x] A2-02C BB domain 100–20000 A 안 = 1,391,131선(78.247544%); 그중 창 때문에 미표본 = 857,959선.
- [x] BB domain 밖 = 386,728선(21.752456%). raw mapped set의 최대 파장은 1.5625e9 A.
- [x] 계약상 Q_g는 `BB_IN_DOMAIN` 교집합이어야 하나 production 호출은 domain mask `NULL`임을 확인.
- [x] 근거: `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/QG_LINE_JBAR_COVERAGE_AUDIT_2026-08-08.md`.
- [x] bare BB profile-support fine mesh 예상 1,906,171 bins. causal blue reservoir까지
  포함한 production mesh는 2,013,113 bins; 기존 host 7배열 약 5.250 GiB.

방금 완료한 단계 — fine-Jbar solver 판독:

- [x] CPU는 causal blue→red이나 현 `ADV_SPLIT` 고정점 자체가 production Courant≈874에서 부정확.
- [x] GPU lagged advection은 24 ALI로 미수렴; full BB-support device layout 약 195.478 GiB로 H200 초과.
- [x] `ALAM=0` static limit는 물리 변경이므로 생산 수리안에서 제외.
- [x] 기존 exact sliding characteristic harness: 498,721 bins, 25 ALI, residual 8.297e-10, 62.8 s.
- [x] production causal mesh의 현 exact CPU peak 예상 총 109.972 GiB; syn104 host RAM
  4,128,416 MiB로 가능.
- [x] `cmfgen_fine_jbar`가 solve rc를 버리는 fail-open 결함 확인; status-returning producer가 필요.
- [x] 근거: `docs/FINE_JBAR_SOLVER_AUDIT_2026-08-08.md`.

방금 완료한 단계 — runtime BB_IN_DOMAIN graph:

- [x] CPU와 transactional CUDA의 Q_g caller가 null mask 대신 닫힌 100–20000 A
  line-centre mask를 생성·검증한다.
- [x] Q_g = 1,391,131선, `BB_EXCLUDED_OUTSIDE_DOMAIN` = 386,728선으로 정적 원장과 일치한다.
- [x] Q hash는 A2-02C domain-contract hash `3278062cf80281ff...`를 포함하고, profile
  hash는 정본 `f8572907be3ad2e97...`를 사용한다.
- [x] 제외 선은 NLTE 소유권을 잃지 않는다. J-driven absorption/stimulated edge만
  rate graph에서 제외하고 spontaneous `A_ul`·collision은 보존한다.
- [x] 구조적 제외 `excluded_out_of_domain`과 잘못된 OOG cache 요청 `blocked_oog`를
  별도 계수한다.
- [x] A2-06 line-Jbar/dual-commit, A2-07 population selftest PASS; CPU 및 CUDA
  sm_80/sm_86/sm_90 compile+link PASS. CUDA SHA
  `ffec0bf59730473f862c7de6e06283083fc44ccc94cf89d6941bd73b5e070cc2`.

방금 완료한 단계 — exact drifting-characteristic 생산자와 커밋 게이트:

- [x] July 독립 harness의 direct `O(beta)`와 sliding `O(1)/bin` 연산자를
  `cmf_exact_characteristic_solve()` 생산 모듈로 이식했다.
- [x] sliding/direct 자가시험 최대 상대차 `9.152e-16`; 양쪽 모두 17회 수렴,
  최종 residual `6.557e-12`. cap exhaustion과 음의 opacity 음성대조도 PASS.
- [x] 기본 cap 64, tolerance `1e-8`; allocation/nonfinite/negative recurrence/
  non-convergence는 모두 명시적 status이며 extraction·R6 publication 전에 차단된다.
- [x] 100–20000 A line-ID domain의 ±4 Doppler support와 blue→red causal upstream을
  분리했다. canonical 74.274847-A blue reservoir가 geometry별 최대 drift를 덮는지
  runtime precheck하며, default fine mesh는 2,013,113 bins다.
- [x] 생산자는 성공 시에만 exact iter/cap/residual/tolerance와 fine-grid stamp를 남긴다.
  R6는 profile/domain hash, residual, 전체 Q profile support, 모든 shell의 VALID 또는
  EXACT_ZERO를 모두 만족하지 않으면 fail-closed한다.
- [x] profile-support endpoint 통과/한-ULP red truncation 실패 회귀시험,
  Makefile header census 28/28, `git diff --check`, CPU full build, CUDA
  sm_80/sm_86/sm_90 link PASS.
- [x] strict env universe 500/500, active 덱 battery D 19/19·K 7/7·Z 12/12·CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_exact_sliding_2026-08-08.log`.
- [x] 현 CUDA binary SHA:
  `42fe86d0ade0a9b3a21c4cf5d807810e98c734c2646fb009641da2b296c6779a`.

완료한 exact transactional preflight:

- Slurm job `251486`, 4 iterations, H200 `syn104`, `FAILED 70:0`, elapsed 6분16초.
- run root: `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T113433Z_42fe86d0ade0`.
- fine 계약은 `100–20000 A`, `v_D=10 km/s`, `ppd=12`, cap 64,
  tolerance `1e-8`로 resolved env와 함께 SHA 봉인됐다.
- exact sliding은 45회에 residual `9.8975819498016093e-09 < 1e-8`로 수렴했고,
  negative recurrence는 0이다.
- R6는 Q_g 1,391,131선 전부, 69,556,550 line-shell cell을 VALID로 발행했다.
  partial/unsampled/exact-zero cell은 모두 0이다. 따라서 DET-R6 전 영역 runtime
  증명은 완료됐다.
- 새 차단은 A210 private population이다. lower `10 K`와 upper `1e7 K`가 모두
  `NLTE_CANDIDATE_SOLVE_FAILED / POP_SOLVE_FAILED`이며 공개 T_e와 generation은 보존됐다.

방금 완료한 단계 — A210 private population 진단 원장:

- [x] 실패를 precondition/transaction/thermodynamic/workspace/EW/pair/CE/publish로 분리했다.
- [x] pair 실패에는 CE pass, pair index/slot, Z/ion, 가장 작은 실패 shell과
  assembly/rank/linear/nonfinite/negative 상태를 남긴다.
- [x] CE 실패에는 iteration/cap, 최악 ion/shell, max relative change와 threshold를 남긴다.
- [x] 진단은 private candidate 소유이며 public NLTE/population 객체에는 추가 상태가 없다.
- [x] reset/정상 이름/invalid-enum 음성대조, A2-07/A2-10/candidate tau/adiabatic 시험 PASS.
- [x] CPU/OpenMP 및 CUDA sm_80/sm_86/sm_90 link PASS; header census 28/28,
  `git diff --check` PASS.
- [x] active battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a210_candidate_diag_2026-08-08.log`.
- [x] 진단 H200 binary SHA:
  `b6469e8e73aec211f3f89fdeb4e0d0bbc06b36fee3cb8f48cf1df4e68b75cafc`.

완료한 A210 진단 preflight:

- Slurm job `251496`, 4 iterations, H200 `syn104`, `FAILED 70:0`, elapsed 6분21초.
- run root: `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T115445Z_b6469e8e73ae`.
- exact/R6 결과는 job 251486과 동일하게 PASS했다.
- lower `10 K`: `PAIR_SOLVE/NEGATIVE_POPULATION`, CE pass 1, pair 0,
  Si II–III, shell 4.
- upper `1e7 K`: `PAIR_SOLVE/INVALID_LAYOUT`, CE pass 1, pair 6,
  inactive C II–III, shell 0.
- 따라서 실패는 CE non-convergence가 아니라 범위 밖 저온 음수해와 비활성 topology의
  잘못된 layout 자격 요구 두 가지다.

방금 완료한 단계 — 물리 bracket과 Z-inert pair 수리:

- [x] 신규 A210의 근거 없던 `10–1e7 K`를 저장소의 ARTIS-mirror 열평형 정본과 같은
  `3500–140000 K`로 교체하고 public contract 상수와 selftest에 봉인했다.
- [x] abundance가 전 shell에서 0인 원소는 rate/layout/rank 검사 전에 candidate-local
  ion/level population을 exact zero로 만들고 성공시킨다. 활성 원소와 잘못된 pair는
  각각 byte-unchanged/blocked 음성대조로 확인했다.
- [x] endpoint pin/floor/fallback은 추가하지 않았다. vector solver의 완전 ledger,
  shell별 sign change, root residual과 단일 commit 조건은 그대로다.
- [x] candidate tau/Z-inert, A2-10 N1–N8, CPU/OpenMP와 CUDA sm_80/sm_86/sm_90 PASS.
- [x] active battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a210_bracket_zinert_2026-08-08.log`.
- [x] 새 H200 binary SHA:
  `2353a77b3bcf85b9ddc952048df89f6e76f5b9eb238c7a03ea1d12d854c46b79`.

완료한 bracket/Z-inert 재검증 preflight:

- Slurm job `251497`, 4 iterations, H200 `syn104`, `FAILED 70:0`, elapsed 6분12초.
- run root: `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T121210Z_2353a77b3bcf`.
- exact/R6는 PASS했고 inactive C의 invalid-layout 실패는 사라졌다. 따라서 Z-inert
  exact-zero 수리는 runtime에서도 유효하다.
- lower `3500 K`: `PAIR_SOLVE/NEGATIVE_POPULATION`, CE pass 1, pair 0,
  Si II–III, shell 17.
- upper `140000 K`: 같은 Si II–III 음수해, shell 10.
- 현재 차단은 bracket 밖 endpoint나 비활성 topology가 아니라 활성 Si rate matrix의
  음수 population이다. floor/clamp 없이 raw 해의 크기와 제약을 계측한다.

방금 완료한 단계 — Si II–III 음수 population 수치 원장:

- [x] 실패한 raw super-level 해의 가장 음수인 준위와 전체 음수 개수, 벡터
  min/max/sum/absolute scale을 candidate-local 진단에 추가했다.
- [x] anchor global level, level number, energy, statistical weight와 trial `T_e/n_e`를
  함께 기록한다.
- [x] 실제 ion-lock 상태, pair total, Si II/III target density와 solve-vector의
  이온별 합을 기록하므로 pinned public ionization과 trial 해의 양립성을 판별할 수 있다.
- [x] 해를 clamp/floor/rescale하지 않으며 실패 시 public object 불변 계약도 유지한다.
- [x] CPU/OpenMP, CUDA fat binary sm_80/sm_86/sm_90, header census 28/28,
  `git diff --check` PASS.
- [x] candidate population/tau/adiabatic selftest와 active battery
  D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a210_si_negative_detail_2026-08-08.log`.
- [x] 진단 H200 binary SHA:
  `76a9d3a9e2c37c40c07a18d5747d679f737f2643ea6655fc5e7a97ef68939493`.

완료한 Si II–III endpoint 수치 판별 flight:

- Slurm job `251501`, 4 iterations, H200 `syn104`, `FAILED 70:0`, elapsed 6분03초.
- run root: `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T123019Z_76a9d3a9e2c3`.
- exact sliding 45회/residual `9.8975819498016093e-09`와 R6 전 Q-domain은 PASS했다.
- lower `3500 K`: Si II solve level 100 (`E=15.671165 eV`, `g=8`) 한 개가
  `-1.1958320185e-7`; vector max `2.5627434584e7` 대비 `4.6662182e-15`다.
- upper `140000 K`: 같은 준위 한 개가 `-1.6274056223e-5`; vector max
  `1.4475599465e8` 대비 `1.1242406e-13`다.
- Si II solve sum 대 target 상대차는 lower 약 `1.26e-7`, upper 약 `1.56e-8`이고
  Si III sum은 target과 출력 정밀도 내에서 일치한다. material-scale 음수나 trial
  ionization 불일치가 아니라 현 unscaled Gaussian solve의 numerical tail로 판정한다.
- 음수 허용치나 clamp/floor를 두지 않는다. 행/열 equilibration과 원방정식 iterative
  refinement로 raw 해 자체와 residual을 수리한다.

방금 완료한 단계 — rate-equation equilibration/refinement 수리:

- [x] 모든 pair-wise dense SE solve를 ARTIS식 row/column 2-norm equilibration,
  partial-pivot LU, long-double 원방정식 residual의 최대 10회 iterative refinement로
  교체했다.
- [x] 변환은 `A'=RAC`, `b'=Rb`, `x=Cy`이며 matrix/RHS 입력 byte를 보존한다.
  population clamp/floor/pin/anchor/fallback은 없다.
- [x] original `Ax=b`의 componentwise backward error가 `1e-12`를 넘으면 solve를
  성공으로 발행하지 않는다.
- [x] 14-decade detailed-balance known answer는 raw solution 전부 strictly positive,
  singular duplicate-column 음성대조는 `POP_RANK_INCOMPLETE`로 fail-closed했다.
- [x] CPU/OpenMP, CUDA fat binary sm_80/sm_86/sm_90, header census 28/28,
  `git diff --check` PASS.
- [x] active battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a210_equilibrated_refined_2026-08-08.log`.
- [x] 새 H200 binary SHA:
  `32ed2b4a4b529c946d04e24398dcbccb61ba0bf82ce3947822d8cace76094cc9`.

완료한 equilibrated/refined endpoint 반증 flight:

- Slurm job `251507`, 4 iterations, H200 `syn104`, `FAILED 70:0`, elapsed 6분05초.
- run root: `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T124647Z_32ed2b4a4b52`.
- exact/R6는 재차 PASS했다.
- lower `3500 K`: rank `202/202`, equilibration 10회, refinement 2회,
  backward error `2.2971e-15 -> 7.1462e-17`, pivot growth `0.999975`.
  Si II/III solve sum은 target과 출력 정밀도까지 일치하지만 level 100은
  `-1.1958271737e-7`로 남았다.
- upper `140000 K`: rank `202/202`, backward error `1.2761e-15 -> 7.7552e-17`,
  pivot growth `0.999973`; 같은 level은 `-1.6274060242e-5`로 남았다.
- [폐기된 이전 판정] refinement 후에도 음수가 남았다는 이유로
  "조립된 ion-locked system의 정확한 미세 음수해"라고 단정했다. 이는
  backward error가 작다는 사실만으로 forward sign과 조립 중 파국적
  상쇄까지 배제한 과한 추론이다. job `251514`의 조건수·고정밀 해와
  추가 raw-rate 보존 잠차로 재판정한다.

현재 flight 봉인:

- Slurm job `250512`: `FAILED 70:0`, 40초. model rc=1, comparison dump 0개.
- run root: `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T093658Z_1336efa41c7d`.
- binary SHA: `1336efa41c7d72ad6355523aa9a88f77d3ec6add8602803839c3a81d4ea92b0d`.
- exact-Hyd σ SHA: `c65d6fcf12c61952855599b317562f45a1b3cc816ce66cc801f2a7a3f14d2675`.
- active-only 55파일을 GPFS에 복사 후 source manifest로 재검증했다. quarantine은 stage하지 않았다.
- 시작 로그: NVIDIA H200 NVL 139.8 GB, env surface 61/61 known, unknown 0, strict fatal gate ON.
- 250512 차단 원인: isolated work에서 상대경로 `data/atomic/topion_levels.csv`가 보이지 않아
  level-less top ion 6개가 `POP_ATOMIC_MISSING`으로 거부됨. catalog 자체는 로컬에 존재하고
  `scripts/check_topion_catalog.py` C1-C8 PASS(15 ions, 7,242 levels). gate를 끄지 않고 두 catalog를
  별도 물리 입력으로 stage·SHA 검증하는 수리를 적용했다.
- diagnostic job `250534`: `FAILED 70:0`, 13분43초, model rc=1, comparison dump 0.
  H200 `syn104`, strict 61/61 known, top-ion 3,853 mapped levels와 6개 level-less ion
  POP_OK, bootstrap population, solver-owned Sobolev τ, `K-FRESH`, CMFGEN assemble까지 통과했다.
  종료 원인은 CUDA의 과거 pure loop가 canonical radiation/line view와 A208/A209를 발행하지
  않고 A210을 직접 호출한 single-owner 우회였다. 물리식 실패가 아니라 driver 위상 결함이다.
- provenance 수리: top-ion thermodynamic membership의 `(ion_index,E_cm,g)` 전체를
  `atomic_model_sha256` v2에 포함하고 값 변경/partial-input 음성대조를 추가했다.
- 숨은 상대경로 수리: `LUMINA_TOPION_LEVELS_FILE`을 추가하고 strict env universe를 498개로 재생성했다.
- single-owner 수리: `LUMINA_DET_TRANSACTIONAL=1`인 봉인 flight는 CUDA의 과거 실험 루프를
  우회하고 canonical `cmfgen_run()` 한 곳만 사용한다. strict env universe는 499개다.
- tau 성능 수리: 2,588,798 line마다 LTE 준위 population을 반복 재구성하지 않고 24,542×50
  고유 `(level,shell)` 값을 같은 계약 함수로 cache한다. canonical active tau bit difference 0.
- 새 CUDA sm_80/86/90 binary SHA:
  `613d66ac814330b299392e98ddc91f9d38e2e498ac9474f5dc41b31a0de76bf2`.
- 수리 후 active 덱 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_transactional_2026-08-08.log`.
- transactional smoke job `250656`, 4 iterations, run root
  `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T100451Z_613d66ac8143`.
  1분07초에 fail-closed: CUDA main이 canonical RadiationFieldOwner와 frozen Q_g를
  초기화하지 않아 iter0 R6 commit이 owner precondition에서 거부됐다. material/dump 0개.
- transactional 모드의 CUDA init에 CPU와 동일한 owner+Q_g를 추가하고, R6 조기거부를
  이유와 세대로 출력하도록 보강했다. A2-06 line-Jbar와 dual-commit selftest PASS.
- 새 CUDA binary SHA:
  `77bd048dd9a4c4dfb0b6fcf93700c2219fd798b7fc935f1cae5a6bb7ae4c8c72`.
- 재봉인 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_owner_qg_2026-08-08.log`.
- transactional smoke #2 job `250658`, 4 iterations, run root
  `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T101342Z_77bd048dd9a4`.
- smoke #2는 H200 `syn104`에서 1분08초, `FAILED 70:0`. owner generation 0과
  Q_g 1,777,859선 구축, R6 generation 1 dual-view commit, A208 generation 1까지
  통과했고 A209가 `rc=5`로 fail-closed했다. material commit/comparison dump는 0개다.
- A209 차단은 stale generation이 아니다(네 stale counter 모두 0). BF producer의
  `eta_bf`가 BF+FF 합인데 A208/A209가 FF를 다른 격자 중심 표현·연산 결합으로
  재구성해 BF=0 근방에서 음의 잔차가 가능했던 산술 정체성 결함이다.
- 수리: BF producer의 등록 중심 `nu_min*exp((b+0.5)*dlog)`과 FF 연산 순서를
  A208/A209가 그대로 공유한다. clamp/floor/fallback은 추가하지 않았다. 실패 시 최초
  `(shell,bin,nu,eta_bf+ff,chi_ff,B,eta_ff,eta_bf)`를 항상 출력한다.
- 수리 후 A2-08, A2-09, private candidate tau/single-commit selftest PASS.
- CUDA sm_80/86/90 재링크 PASS. 새 binary SHA:
  `08855ba67281630632c133c32ffd1a02ebc7dc3f19e4a61ea2631fbcb993a099`.
- 재봉인 active battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a209_ff_identity_2026-08-08.log`.
- transactional smoke #3 job `250765`, 4 iterations, run root
  `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T102346Z_08855ba67281`.
- smoke #3는 R6/A208 뒤 A209 full-line 합산에 진입해 FF 차단 제거를 확인했으나,
  6분04초 동안 LTE 준위를 line×shell마다 재탐색했다. material commit 0 상태에서
  새 cache binary로 대체하기 위해 운영자가 `CANCELLED`했다.
- A209도 τ writer와 동일한 정본 population 계약으로 24,542×50 LTE level density를
  한 번 계산해 재사용한다. private candidate tau/single-commit, A2-07 회귀 PASS.
- cached CUDA sm_80/86/90 binary SHA:
  `e2b71bbbf9b81aeeedb0d2a30cb9afe2b6ec5be340701aa2283e358bf5b60f7a`.
- cache 재봉인 active battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a209_cache_2026-08-08.log`.
- transactional smoke #4 job `250870`, 4 iterations, run root
  `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T103025Z_e2b71bbbf9b8`.
- smoke #4 실측: 약 1분13초에 R6 generation 1→A208 `o=1`→A209 `e=1`을
  통과하고 `[A2-10][PRE]`에 진입했다. 직접 Sobolev emissivity cancellation cell은 0이다.
- smoke #4는 1분15초에 `RADEQ_TERM_SCHEMA`로 fail-closed했다. stale/bracket/root가
  아니라 lower/upper private trial 자격 단계이며, 공개 T_e 값·manifest·generation은 보존됐다.
- A210에 candidate status, population status, bundle 단계 flags, trial T 범위와 최초
  invalid ledger shell/status/equation/adiabatic 값을 출력하는 관측 전용 진단을 추가했다.
- 진단 CUDA SHA `d48f893ce642faf85c72cf93ab2810c57459f35f7de78365adfcf9e16a4b9a76`;
  active battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_det_a210_diag_2026-08-08.log`.
- A210 schema diagnostic smoke #5 job `250995`, run root
  `/gpfs/kjhan/lumina/det_convergence/det1178_20260808T103716Z_d48f893ce642`.
- smoke #5 판정: lower=10 K와 upper=1e7 K 모두
  `NLTE_CANDIDATE_SOLVE_FAILED / POP_BB_UNSAMPLED`. 이는 A210 ledger schema 실패가
  아니라 R6 partial line-Jbar를 실제 SE 소비자가 정직하게 거부한 것이다.
- smoke #5의 pre-repair R6 coverage는 Q_g 1,777,859선 중 VALID 533,172,
  UNSAMPLED 1,244,687이었다. 새 graph에서는 Q_g 1,391,131선 중 같은 VALID 533,172,
  in-domain UNSAMPLED 857,959가 예상된다. A209는 통과하고 SE만 차단하므로 N6-4
  계약은 그대로 작동한다.

DET 자산 조사 판정:

- `validation/.../detjbar_convergence`는 2026-07의 Jbar 이산화 harness이며 새 production DET 수렴 flight가 아니다.
- `scripts/run_coevolve_s01.sh`는 물리 env의 기계적 원천으로만 사용한다. 저장소 루트 출력과 과거 실험 분기는 재사용하지 않는다.
- 현재 `cmfgen_run`은 고정 outer-iteration driver다. 반복 횟수 완료를 수렴으로 간주하지 않고 comparison dump의 마지막 연속 변화량으로 별도 판정한다.
- H200 partition은 사용 가능하고 현재 사용자 잡은 없다. login node에서는 binary를 실행하지 않는다.

최근 전체 배터리: `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/gate_battery_rebaseline.log`, verdict=PASS.
저주파 폐합: `validation/sh_grid_low_band_exacthyd_canonical_2026-08-08/manifest.json`, verdict=PASS.
SH-GRID 판정: `docs/SH_GRID_MIGRATION_VERDICT_2026-08-08.md`.

터미널 고정 보기:

```bash
watch -n 2 cat docs/CURRENT_PLAN.md
```

Fable 예산 정책: 추가 호출은 동결한다. 로컬 계약과 시험으로 해결할 수 없는 새로운 핵심
물리 모순이 생긴 경우에만 사용자에게 먼저 사유를 보고하고 사용한다.

## 2026-08-09 최신 단계 — endpoint 실증과 CMFGEN line-energy 경계

- [x] job `251622` 종료: H200 `syn104`, `FAILED 70:0`, model rc=1,
  elapsed 29분18초, MaxRSS 79,478,728 KiB, MaxVM 173,829,548 KiB.
- [x] run root:
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T022526Z_3e38b9cd0750`.
- [x] Fe III–IV shell 44 exact-zero pair는 lower `3500 K`, upper `140000 K`,
  geometric-mid `22135.943621178667 K`에서 모두 `[PAIR-EXACT-ZERO]`로 통과했다.
  floor/tolerance/clamp는 없다.
- [x] lower/upper endpoint ledger 각 50개는 모두 finite. endpoint sign은
  bracket 35 shells(12–46), same-negative 12(0–11), same-positive 3(47–49).
  midpoint에서 47–49만 interior bracket을 보였고 0–11은 계속 음수였다.
- [x] job 251622 stdout/stderr/footer SHA는 각각
  `c5ef55c5d83f982d8bdd8f60ce3cb91dcd2ae5b849da58b915712d8d14865118`,
  `957c3989b91bb83980389cfc3d4693fad1f5eba06353f9a391fc26f5b649d203`,
  `2ef1b33a852d267b2d5401e0f3622303f6920255c9b08ae2b4a287ff8d6f3f1f`.
- [x] CMFGEN 원문 audit 결과, Sobolev RE line 항은 standalone
  `n_u A_ul h nu beta`가 아니라 radiation-coupled `ETAL_MAT*ZNET`이다.
  O-PHYS deck은 `CHK_L_POS=T`, `NEG_OPAC_OPT=SRCE_CHK`를 사용한다.
- [x] lower shell 0 line forensic은 negative-tau emission fraction
  `0.9999979824`; 지배 Co II line은 `tau=-23.4290436825`,
  `beta=6.3877781636e8`. 현재 큰 cooling은 CMFGEN 동종 물리량이 아니다.
- [x] A2-08 signed tau는 원형 보존하고, A2-10 private candidate가 line energy를
  소비하기 직전에 in-grid active line의 `tau<0`를 tolerance 없이
  `RADEQ_SIGN_MISMATCH`로 거부한다. 가장 음의 identity와 전체 count를 기록한다.
- [x] 직접 회귀군과 전체 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS.
  로그:
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_negative_tau_energy_guard_2026-08-09.log`,
  SHA `606b9f4b80d46c4317f41b0865425bb979ce5509fbc47ab1f3a52230f63b007b`.
- [x] fresh CUDA sm_80/sm_86/sm_90 link, `make -q cuda`, `git diff --check` PASS.
  binary SHA `a54d2600542a53c002deaf15e1a172cce93df9e94ffc2e19fa8638f4aac218ae`.
- [x] 정본 audit:
  `docs/CMFGEN_NEGATIVE_SOBOLEV_RADEQ_AUDIT_2026-08-09.md`.
- [x] 이 단계에서는 Fable을 호출하지 않았다. CMFGEN source와 sealed deck이 핵심
  모순을 직접 해소했으므로 희소 토큰을 보존했다.
- [x] CMFGEN RE 내부 단위를 source에서 cgs로 복원했다:
  `q_cgs=q_internal*4*pi*1e-10`. O-PHYS line 76887/depth 90의 raw finite
  net은 `0.6694430052329409`, deck-scaled production 값은
  `0.6680659609711768 erg cm^-3 s^-1`다.
- [x] `LINEHEAT` 2.73 GB와 `NETRATE` 1.09 GB를 streaming으로 교차해 finite
  known-answer v2 fixture를 봉인했다. fixture SHA
  `5a967bbbf6f374c69c6ae5fd63d420d1fadc002c04ddf2fbbef24192a81951a0`,
  extractor SHA
  `169be7e55c1502bd944a4aab270f021eebce0c70e77f7602c7477aac6ac1433e`.
- [x] energy용 모든 BB-domain line `Q_E`와 population rate graph `Q_g subset Q_E`를
  분리하고, continuum과 단일 `Q_E Jbar` cache를 한 atomic commit으로 발행하는
  데이터 계약을 확정했다. 중복 `Q_g` numeric cache와 generation 이중화를 금지한다.
- [x] 상쇄 계약은 `fma(-chi,Jbar,eta)` signed cell을 먼저 보존하고, MC SE/DET
  formal-solution bound가 sign보다 크면 `UNRESOLVED_CANCELLATION`으로 거부한다.
  exp 변환, jitter, floor, clamp는 없다.
- [x] 정본 계약:
  `docs/CMFGEN_LINE_NET_DATA_CONTRACT_2026-08-09.md`.
- [x] `src/line_net_rate.c/.h` pure kernel 구현. component 경로는
  `fma(-chi_int,Jbar,eta_int)`로 signed rate를 먼저 계산하며 clamp/floor/jitter가
  없다.
- [x] finite cooling/heating, typed exact-zero, 비영 성분의 exact cancellation,
  uncertainty-covered sign, NaN과 상충 exact-zero provenance 음성대조 PASS.
- [x] CMFGEN v2 fixture를 `0.66806596097117665 erg cm^-3 s^-1`로 재현했다.
  fixture 인쇄 정본 `0.6680659609711768`과 double 정밀도에서 일치한다.
- [x] large-minus-large FMA witness
  `(1+2^-27)(1-2^-27)`는 별도 곱의 0 대신 finite `2^-54 =
  5.5511151231257827e-17`을 보존했다.
- [x] strict `-Werror -pedantic`, ASan/UBSan, Makefile header 29/29,
  CPU/OpenMP full link와 `make -q OMP=1 all` PASS. 기존 source warning 외 새
  kernel warning 0.
- [x] `LineJbarSetKind`와 `line_jbar_eset_build()` 구현. 기존 canonical `Q_g`
  sample hash `ae6163fe...`는 byte-identical로 유지하고, synthetic `Q_E` hash
  `f781482b...`를 known answer로 봉인했다.
- [x] `line_jbar_qset_subset_of_eset()`은 role/domain/profile/hash/line-ID/frequency를
  검사한다. missing line, seeded hash corruption, same-ID frequency mismatch
  음성대조 PASS.
- [x] exact-hyd O-PHYS `line_list.csv` 453 MB offline census: 전체 2,783,436선,
  `Q_E=2,783,421`, invalid frequency 0, hash
  `846ff0e6f651a6f2f82cc1b736db823a894fdcea51671e4b295deacb37c0142d`.
  50 shell cache 약 3.629 GiB, MC accumulator 약 3.111 GiB.
- [x] A2-06 line-Jbar normal/strict/ASan-UBSan, dual-commit, header 29/29,
  CPU/OpenMP full link PASS. 아직 public owner는 `Q_g`; 이번 단계는 membership만
  닫았다.
- [x] `LineJbarCache.set_kind`와 owner-internal `Q_g` ID/hash→`Q_E` index sparse
  map 구현. 별도 `Q_g` numeric value/validity/count/SE slab은 만들지 않는다.
- [x] energy checked view는 `Q_E` hash, rate checked view는 `Q_g` hash를 요구한다.
  4-line `Q_E` 위 `Q_g={42,900}` sparse map `{1,3}` known answer에서 rate view는
  energy-only line 11을 `MISS`, energy view는 정상 lookup했다.
- [x] `Q_g`에 없는 line 901을 넣은 generation 2 commit은 field/cache generation과
  기존 graph line 900을 byte 보존하고 거부했다. GPU sparse gather 전에는
  연속 memcpy 오독을 막는 명시적 fail-closed guard가 작동한다.
- [x] A2-03/04/05/06 호환, strict compile, ASan/UBSan, CPU/OpenMP full link,
  sm_90 CUDA full link PASS. sm_90 binary SHA
  `0165a297dbbe5a7417c065da14256165c59f913da170df8760447769c3900dac`.
- [x] 호환 회귀 중 A2-04의 stale edge `3900`과 Python `4000` bins를 발견했다.
  canonical SH-grid 식의 3866 bins로 수리했고 replay 최대 오차 약 `1.2e-17`,
  Planck 5-band 음성대조 전부 expected-fail PASS.
- [x] CPU와 CUDA transactional DET 초기화에 전 BB-domain `Q_E`를 만들고
  `Q_g subset Q_E` role/domain/profile/hash/line/frequency 검증을 fail-closed로
  연결했다. CPU MC는 다음 전용 단계 전까지 기존 `Q_g` producer를 유지한다.
- [x] `cmfgen_commit_jnu()`는 fine solver의 private `jbar_line_det` 전 `Q_E`를
  continuum과 한 atomic commit으로 발행한다. population은 sparse `Q_g` rate
  view, 이후 line-energy는 동일 cache의 `Q_E` energy view를 사용한다.
- [x] 전 `Q_E` profile support/line identity/모든 shell coverage를 검사하고,
  로그에 Q_g/Q_E count와 두 hash를 함께 남긴다. 누락·sentinel·nonfinite·stale
  상태는 공개 전에 거부한다.
- [x] A2-03/05/06와 line-net 회귀, CPU full link, sm_90 CUDA full link,
  `git diff --check` PASS. 현 binary SHA는 CPU
  `52793e5b9f06e2b5903e8c5c5a16f062b7a5e2312351bbe2eba397231d7bea1c`,
  sm_90 CUDA
  `c743ecd59e6bb0d5fffc8807372e8554027a40257b5acb80ae8539df8f17ce67`.
- [x] GPU mirror는 rate graph hash를 cache hash와 혼동하지 않고, `Q_E` cache의
  `cache_index`로 Jbar/validity/count/SE를 compact `Q_g` 순서에 gather한다.
  contiguous base memcpy 경로는 없다.
- [x] gather buffer는 후보 상태에만 존재하며 H2D 후 전 바이트 D2H attestation이
  일치해야 READY가 된다. device 공개 line count와 readiness shape도 `Q_g` count에
  결박했다.
- [x] H200 job `251754`, node `syn104`, NVIDIA H200 NVL 143771 MiB,
  `COMPLETED 0:0`, elapsed 9초. synthetic `Q_E={11,17,23,29}`에서
  `Q_g={17,29}`와 Jbar `{3,4,7,8}e-12` device 재독출 PASS; N1--N9 모두
  expected nonzero PASS. stdout SHA
  `4962098e3b0142787ce18ba9ed7e058c0aea48285a754e4265761f6917df7fa5`,
  stderr empty SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- [x] production sm_90 CUDA full link SHA
  `98ad6e9f6fd1957759a5fbd12db7ffca4be3027722f184f6eeff05e6e0ea5a53`;
  mirror source SHA
  `0119a8123f859967e4852bfa7bc932afad02c5832e011afe11edd62cac2904d2`.
- [x] CPU MC segment estimator의 membership을 `Q_g`에서 전 `Q_E`로 확장하고,
  `sum/sumsq/count` accumulator도 `Q_E*n_shells`로 일치시켰다. size 곱셈 overflow는
  allocation 전에 fail-closed한다.
- [x] MC commit은 numeric line IDs/hash로 `Q_E`, rate graph IDs/hash로 `Q_g`를
  따로 전달해 continuum과 한 transaction으로 발행한다. commit 뒤 같은 generation의
  sparse rate view와 energy view를 모두 재검증한다.
- [x] MC variance numerator는 `fma(-sum/N,sum,sumsq)` signed 결과를 보존한다.
  음수이면 0 clamp/floor 없이 전체 commit을 거부하고 이전 field/cache generation을
  byte 보존한다. synthetic `sum=2, sumsq=0, N=100`의 `-0.04` 음성대조 PASS.
- [x] A2-03/05/06, strict, ASan/UBSan, CPU full link, sm_90 CUDA full link,
  header 29/29, `git diff --check` PASS. CPU SHA
  `f7af09fbc5c54e09281613a5f33822684fe600f8f56f5f719a872e0967d0a8db`,
  CUDA SHA
  `83f0f2c705bdccb7389dcb8d2d79f270cde3d5cf94590becb069bc5472e3894a`.

현재 판정:

- exact-zero pair repair와 유한 endpoint 생산은 runtime에서 성립했다.
- job 251622의 residual/cooling은 line energy 식이 다르므로 CMFGEN finite 재현으로
  인정하지 않는다.
- [x] A2-10 line-energy 정식 owner를 binned `chi_bb/eta_bb` 차분에서 전 `Q_E`
  line-resolved `fma(eta-chi*Jbar)` owner로 교체했다. 기존 binned line 성분은
  `NAN/UNSAMPLED`이어도 새 owner의 ledger가 성립하는 음성대조를 통과했다.
- [x] deterministic fine producer는 `tau*nu/(c*t)`와
  `n_upper*A_ul*h*nu/(4*pi)`를 각각 Gaussian 적분량으로 직접 deposit한다.
  `tau<-0.5`는 CMFGEN 원문 `SRCE_CHK`; `[-0.5,0)`은 signed raw opacity를
  보존한다. 기존 `(1-exp(-tau))*S_l` 경로는 Q_E context가 없는 legacy
  diagnostic에만 남는다.
- [x] `SCL_LN` statistical-weight super-level average-energy scale과 density/range
  deck rule을 구현했고, CMFGEN finite fixture `0.66806596097117665`를 유지했다.
- [x] DET 오차는 `tolerance*Jbar` 추정에서 formal fixed-point 절대 오차상한으로
  교체했다. 마지막 infinity-norm absolute change와
  `max(chi_es/chi_tot)<1` 수축상한으로 계산하며, 상한이 유한하지 않으면 Q_E
  commit/A2-10이 fail-closed한다. mild signed line의 formal solve 자체와
  energy publication 자격을 분리한 음성대조를 통과했다.
- [x] 수치 floor/cap/clamp/jitter 금지 원칙을 재감사했다. line_eps 범위값을
  조용히 자르지 않으며, fine-BF true absorption 또는 total source 조립이 음수/
  비유한이면 최초 shell/bin/nu와 성분을 기록하고 대체 없이 차단한다. Planck
  Wien tail도 임의 `x>700 -> 0` floor 대신 동치인 `exp(-x)` 식을 사용한다.
- [x] CPU, OpenMP, sm_90 CUDA full link, strict/ASan/UBSan line-net+exact solver,
  Makefile header 29/29, `git diff --check` PASS. binary SHA는 CPU
  `d52cd4fe30046f7b3448b8625d1ee910dd990519a43ffede7195b42ca74b0fe0`,
  sm_90 CUDA
  `07155fd015fc244f2b1bdd9fc4e823ae434c7a0a0b4a8e178961f277a1700810`.
- [x] Q_E line material의 upper population을 scalar 반복 검색 대신 한 번의 checked
  bulk population cache로 만든다. line×shell 원시값은 바꾸지 않으며, 실제 flight는
  signed/raw-negative/mild-negative/SRCE_CHK census와 최악 line/shell의 population
  inversion 성분을 항상 기록한다. `floor=0 clamp=0 jitter=0`이다.
- [x] 전체 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_line_net_owner_bulk_2026-08-09.log`,
  SHA `a90578daff5be63c5e28e565ef567f1d8faa38c172e79aaa8b910347de4cf063`.
- [x] 이 단계에서도 Fable은 호출하지 않았다. CMFGEN 원문과 로컬 contract/test로
  모순을 해소했으므로 희소 토큰을 보존했다.
- [x] H200 job `251916`은 생산 launcher/footer에 수치 수리 knob
  (`HRESP_CLAMP=1`, `NLTE_INV_CEIL=1e4`, `NLTE_LTE_FLOOR=1`,
  `TE_STEP_CLAMP=1`)가 남아 있음을 확인한 즉시 45초에 취소했다. 이 job의
  물리 결과는 전부 무효이며 어떤 판정에도 채택하지 않는다.
- [x] 물리적으로 비음수인 population을 푼 뒤 `x<0 -> 1e-30`으로 바꾸던 옛 CUDA
  solver와 singular/nonfinite/inversion 때의 Boltzmann 대체 경로는 생산 build에서
  분리했다. GPU entry는 현재 transactional CPU generator/GTH solver로 fail-closed
  route하며, pure-CMFGEN loop의 두 번째 population solve도 제거해 A2-10 trial
  bundle만 population/atom generation의 단일 owner가 되게 했다.
- [x] 생산 A2-07/A2-10 진입점은 수치 repair/fallback 환경변수가 하나라도 nonzero면
  `[NUMERIC-REPAIR][BLOCKED]`로 즉시 종료한다. submit helper와 compute-node
  preflight도 16개 knob가 문자열 `0`인지 이중 검증한다. `SUPER_CUTOFF=100`은
  solve 뒤 값을 자르는 cap이 아니라 atomic super-level topology를 정하는 구조적
  모델 해상도이므로 별도 봉인한다.
- [x] 무보정 전체 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_no_numeric_repair_2026-08-09.log`,
  SHA `3015a330a20db50d4d1774b66ebec2a453fed5596ed587c73a14cabab19e2362`.
  최신 CPU SHA `593e553228fc06f6d8a79d0bba2750f906740e36b8c848fee96bfccba1911225`,
  sm_90 CUDA SHA `e8daa7e343473b3168093830354d981dc888366fa82fa23bd2203fe013060de6`;
  `make -q lumina_cuda` PASS.
- [x] 무보정 flight `251932`, run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T050756Z_e8daa7e34347`는
  H200 `syn104`에서 12초 만에 strict env preflight로 종료했다. 새 금지 knob
  `LUMINA_NLTE_FALLBACK_TE=0`가 `env_universe`에 빠져 있었고 reference data를
  읽기 전에 `BLOCKED_UNKNOWN_ENV`가 발생했으므로 물리 결과는 없다.
- [x] solver나 값을 바꾸지 않고 금지 knob 이름만 strict env 사전에 등록했다.
  재검증 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_no_numeric_repair_envdict_2026-08-09.log`,
  SHA `bef755df27b19fee8cf2666dbe4bb1bf9d2fa43e2301ca1e38a9cd742e87fa20`.
  최신 CPU SHA `074e1ba2fd6e00e1952a793a7f64d20db7414604ad40386bd6242fd822909d50`,
  sm_90 CUDA SHA `561a745197df999261123d55d7179e2797f0b9a00d2cb24d34120b7b7c589b29`;
  `make -q lumina_cuda`, `git diff --check` PASS.
- [x] resolved env의 추가 이름 감사를 수행했다. `NLTE_COLL_FIX=1`은 solve 결과의
  음수를 덮는 수치 fix가 아니라 허용선 van Regemorter/금지선 Axelrod `Omega=1`
  충돌강도 물리 근사 선택이다. `ION_LOCK=0`, `PER_ION_RESCALE=0`이고, 수치
  floor/cap/clamp/fallback 16종은 전부 exact zero다.
- [x] H200 job `251933`, run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T051322Z_561a745197df`
  종료. node `syn104`, `FAILED 70:0`, elapsed 19분26초, MaxRSS
  79,768,224 KiB, strict env 81/81, staged binary SHA
  `561a745197df999261123d55d7179e2797f0b9a00d2cb24d34120b7b7c589b29`,
  16개 numerical-repair knob exact zero를 확인했다. A2-08/09와 50-shell signed
  tau forensic은 통과했으나 A2-10 lower/upper가 `RADEQ_NONFINITE`로 차단됐고,
  Te generation/manifest는 byte 보존됐다.
- [x] 정확한 원인은 물리량 NaN이나 메모리 부족이 아니라 atomic topology 검사다.
  runtime view는 33 ion 중 level-less 6개를 합법적으로 갖지만,
  `a210_super_average_energy_eV()`가 모든 interval에 `end > begin`을 요구하여 첫
  빈 interval `end == begin`에서 `NULL`을 반환했다. staged `levels.csv` 24,542개는
  `g<=0`, 음수 super-level, 비유한/음수 energy가 모두 0건이며 27개 nonempty ion
  group과 정확히 일치한다.
- [x] 빈 interval만 건너뛰고 reversed/out-of-range interval과 잘못된 level 값은
  계속 fail-closed하도록 수정했다. 향후 실패는 ion slot, begin/end, level,
  energy, g 또는 allocation bytes/errno를 직접 기록한다. floor/cap/clamp/jitter나
  물리값 대체는 없다.
- [x] 수정 후 CPU full link와 전체 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS.
  로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_level_less_ion_topology_2026-08-09.log`,
  SHA `bd7c7371f3616a948a7d9e961dbedf911e107e0b4ee88d24509b74a60cb4117c`.
  fresh CUDA sm_80/sm_86/sm_90 link SHA
  `942c6972fc9281d5f6fa33697de19e035f3b6aa1929442c95ffe6e188492333e`;
  `make -q lumina_cuda`, `git diff --check` PASS.
- [x] H200 job `251949`, run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T054153Z_942c6972fc92`
  종료. node `syn104`, `FAILED 70:0`, elapsed 19분32초, MaxRSS
  79,769,740 KiB다. staged SHA 일치, strict env 81/81, unknown=0,
  single-total=1, Stage-IV=1, MATDUMP=0, 수치 repair 16종 exact zero다.
  level-less 6 ion은 모두 `POP_OK`; exact sliding은 45회에 residual
  `9.666269975044044e-09 < 1e-8`, absolute error bound
  `7.95680597520651e-06`, 음수 recurrence 0으로 통과했다. R6는
  `Q_g=1,603,732`, `Q_E=2,180,286`, valid `Q_E=2,180,286`, 총
  109,014,300 cells, partial/unsampled 0이다.
- [x] A2-10은 population/topology가 아니라 line-net의 cancellation 자격에서
  fail-closed했다. lower 최초 증거는 line 15/shell 4, upper는 line 17794/shell 26이며
  둘 다 `RADEQ_SIGN_MISMATCH`와 `UNRESOLVED_CANCELLATION`이다. Te generation과
  manifest는 보존됐고 population failure/negative, trial-bundle 차단,
  super-average 차단은 없다. stdout/stderr/footer SHA는 각각
  `3b9da6a2bf5786c5e3c7cb8e1281006aec8ff7e989f04a5024f78982a13d49b9`,
  `7691c0bf95b85e30366f2a4883e78ff706f20376d34f7af923b141950cbc80ba`,
  `2019fa81c0bfcad30222589b861ec926c24b5be1f37eaf8809219d792a985b24`다.
- [x] 다음 비행에서 원인을 값 대체 없이 판별하도록 최초 실패 cell의 line/shell/
  Z/ion/level, raw tau, raw/effective chi, eta, Jbar와 SE·bound, absorption,
  FMA net, signed rate, propagated uncertainty와 cancellation 조건을 한 줄로 보존한다.
  또한 invalid callback을 무조건 `blocked_schema`로 세던 진단 오류를 고쳐 실제
  `RADEQ_SIGN_MISMATCH` counter를 보존한다. 새 selftest는 candidate와 출력 byte가
  보존되고 `blocked_sign=1`, `blocked_schema=0`임을 검증한다. 계산 경로와 물리값은
  바꾸지 않았다.
- [x] 수정 후 전체 battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_line_net_sign_provenance_2026-08-09.log`,
  SHA `7a1a32cfecef8223a38cd3cf860021fdca37fd4355db71a6d536fcdda350097d`.
  최종 CPU/CUDA SHA는 각각
  `d68640da39ea97938c4ca79957768157104f16c92df4c73748e2f3e27231f425`,
  `1591473a355150586d5aabb0657d4c297cb9b4836d1c0e7708d1b4f889e60a1e`;
  `make -q lumina_cuda`, `git diff --check` PASS.
- [x] H200 job `251966`, run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T060943Z_1591473a3551`
  종료. node `syn104`, `FAILED 70:0`, elapsed 19분54초, MaxRSS
  79,769,736 KiB다. staged CUDA SHA는 로컬과 일치하고, compute-node strict env
  81/81, unknown=0, env SHA
  `32cd5210b15ba0575e775cb793d22c4261657c2647dab4643a5808867f1cd653`,
  수치 repair 16종 exact zero다. exact sliding과 R6 coverage는 job `251949`와
  bit-for-print 동일하게 통과했다.
- [x] lower 최초 raw cell은 line 15/shell 4다. `Jbar=2.9609774309735141e-51`,
  emission `9.6057446027311036e-164`, absorption/net
  `2.413426102348906e-60`/`-2.413426102348906e-60`,
  `cancellation_condition=1`이다. 즉 큰수-큰수 상쇄가 아니다. 모든 cell에 동일하게
  넣은 전역 `Jbar_bound=7.95680597520651e-06`이 실제 Jbar보다
  `2.6872227704182334e45`배 커서 signed-rate uncertainty도 raw rate보다 같은 비율로
  커졌다.
- [x] upper 최초 raw cell은 line 17794/shell 26이다.
  `Jbar=5.1118328552902324e-32`, emission/net per sr
  `2.0671935418107471e-15`/`2.0671935418107471e-15`,
  `cancellation_condition=1`; 전역 bound는 Jbar의 `1.5565465852374293e26`배이고
  signed-rate uncertainty는 raw cooling의 `1.643185435005021`배다. 두 endpoint
  모두 population/roundoff/cancellation 실패가 아니라 valid하지만 지나치게 거친
  전역 fixed-point bound의 cell별 재사용 때문에 자격을 잃었다.
- [x] 실패 counter provenance 수정도 flight에서 확인했다. 최종 R7 reason은
  `RADEQ_SIGN_MISMATCH`, `schema_delta=0`, Te generation/manifest와 material
  generation은 보존됐다. stdout/stderr/footer SHA는 각각
  `0cf53214cb2096417c67abb55611b2f4fe0a787cbc7d5f87ac81e1ad68ab923e`,
  `9b5e1bfb8e9352d9426ebb179dd3ef5dfa49455a53e472dc5aab79866f65aaef`,
  `6a2604439715bf65a6450e9f45e0914a3bc95312daa7c4a15243d45c058b5ac0`.
- [x] 이 사안은 109,014,300 line-cell의 물리 승인 방식을 바꾸는 핵심 알고리듬
  선택이므로 희소 Fable 판정을 1회만 사용했다. 판정은 양의 affine operator
  `J=b+KJ`에서 residual `r`와 componentwise supersolution
  `u >= |r| + K u`를 directed-rounding으로 한 번 검증하면
  `|J*-J| <= u`, 따라서 비음 normalized line profile `w`에 대해
  `|delta Jbar| <= w^T u`라는 것이다. 반복은 후보 생성일 뿐이며 최종 부등식만
  증명이다. 별도 adjoint 1억 회와 local diagonal `|r_i|/(1-q_i)`는 기각했다.
- [x] 자체 판정으로 Fable의 `u` 예상 크기는 acceptance 조건으로 채택하지 않았다.
  비국소 K 전파 때문에 실제 크기는 계산해야 한다. 또한 이 증명은 연속 물리해가
  아니라 현재 이산 formal operator의 fixed point만 인증하며, 실제 solve와 동일한
  floating operator, K의 성분별 비음성, residual/Ku의 상향 반올림이 필수다.
- [x] 물리값을 입력으로 받지 않는 generic `cmf_error_envelope` verifier를 구현했다.
  전역 supersolution에서 `|r|+Ku`로 아래 방향 정련하되, 후보가 componentwise
  non-increasing이고 다시 supersolution 검증을 통과한 뒤에만 교체한다. 2x2
  known answer `[1,1]`, one-ULP 축소, zero-iteration, 비국소 coupling을 누락한
  local-diagonal 근사, 음의 K, operator failure, `1+2^-54` round-to-nearest false
  pass를 모두 봉인했다. strict/ASan/UBSan PASS; selftest/ASan SHA는
  `9de272da4d67599b5f0b698dbb23ede33b9116d229dee793d3e039a070701dc1`,
  `b67dbd8b935d3498d04e2beb3a802559f59732946c9f49991f8d8581d5fb9a57`.
- [x] generic verifier 추가 후 header closure 30/30과 전체 battery
  D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그 SHA
  `fcf6190028cd7cd8c3b66cf9f8d9cd6cb36ba02568a422276abc9e60d3dac0d9`.
- [x] production iteration 안에 중복돼 있던 ray propagation과 angular reconstruction을
  `exact_formal_sweep()` 단일 owner로 추출했다. solver와 향후 K/error path가 별도
  formal operator를 재구성하지 않게 한 동작불변 리팩터링이다. small-grid
  sliding/direct는 17/17회 수렴, 최대 상대차 `9.152e-16`; 전체 battery
  D19/K7/Z12/CP4 PASS. 로그 SHA
  `04389869652cdddcef55b9f31c1ccd26deb5dd9a8b882ece38d50564107a052a`.
- [x] 논문 `docs/paper/lumina_paper.tex`에 shipout foreground `DRAFT` watermark를
  넣고 PDF를 재생성했다. 11 pages/11 markers이며 page 1, 6, 11을 렌더링해 전면
  figure page에서도 보임을 확인했다. TeX/PDF SHA는 각각
  `1c7d6c14405da70f780ca74061937bbfbd9b7e55a47aea45ca5ade9dc9211fc1`,
  `f4477f84c5d1451432b90fa3ad6704d77cc50f8ada76f3d5684189ee8471ad0d`.
- [x] line universe를 바로잡았다. 외부 O-PHYS CMFGEN offline line list의
  `Q_E=2,783,421`은 이번 staged Lumina active deck의 target이 아니다. 이번 deck은
  전체 2,588,798 lines, in-domain `Q_E=2,180,286`; 최초 109,014,300 line-shell
  material census는 raw-negative=0, mild-negative=0, SRCE_CHK=0이며
  floor/clamp/jitter=0이다. 두 universe의 비교는 동일 identity 교집합에서만 한다.
- DET 단독 flight는 여전히 coevolution 고리 밖 검증이다. 최종 구조는 DET loop A
  master와 반복 `it`의 MC field를 `it+1` shared material에 넣는 lagged feedback이다.
- [x] subtraction recurrence의 interval 폭발을 감수하지 않고, nonnegative affine
  transform monoid를 두 stack으로 합성하는 `POSITIVE_SLIDING` formal mode를 별도로
  구현했다. outgoing old-large term을 빼지 않으며 amortized O(1)이다. 독립 direct
  oracle과 최대 상대차 `7.322e-16`(동일 tolerance), tight direct oracle과
  `1.557e-12`이고 음수 recurrence는 0이다. 기존 production `SLIDING` mode를 몰래
  바꾸지 않았으며 error-envelope 요청은 positive mode가 아니면 입력/출력을 byte
  보존하고 거부한다.
- [x] 동일 positive formal object의 source-dependent multiply/add를 outward 평가하는
  lower/upper apply를 연결했다. 곱셈은 `fma` residual 자체의 subnormal underflow를
  신뢰하지 않고 proof bound만 무조건 한 ULP 바깥으로 이동한다. physical `J`, opacity,
  emissivity에는 floor/cap/clamp가 전혀 적용되지 않는다. 이 상계는 고정된 binary64
  이산 operator의 roundoff+iteration error만 인증하고 continuum/discretization
  error를 주장하지 않는다.
- [x] 최종 iterate에서 outward `|F(J)-J|`, zero-boundary scattering `K`, global
  supersolution seed, 5회 local downward refinement와 마지막 componentwise
  `u >= |r|+Ku` 재검증을 단일 solver API에 연결했다. small-grid에서 모든 local
  bound가 finite/nonnegative이고 tight direct 고정점과의 관측 차이를 덮었으며,
  최대 관측오차/local-bound 비는 `5.506e-01`이다. 1/4 OpenMP thread bit-for-print,
  strict/ASan/UBSan PASS; normal/strict SHA
  `819ed9dba93dc1d60ae8941967e5d43f05d928c4c62d1b8ce4dbee17a5aa0cd4`,
  sanitizer SHA
  `a5542163185c34dfbd13aebff4231057d0f2d3e12e6f47f7e9f2c98dcee8d866`.
- [x] componentwise solver 연결 후 header closure 30/30, `git diff --check`, 전체
  battery D 19/19, K 7/7, Z 12/12, CP 4/4 PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_componentwise_solver_2026-08-09.log`,
  SHA `7ec31e0981e117168006e29905d6750a885233214ded04476f90405627e45935`.

- [x] verified component envelope `u[s,nu]`를 production Gaussian profile의 바로
  그 support와 normalization으로 적분하는 단일 owner
  `line_jbar_gaussian_discrete_shells()`를 구현했다. 물리 Jbar는 기존 nearest
  accumulation과 bit-identical이고, 오차 분자는 위로, 양의 정규화 분모는 아래로
  directed rounding하여 각 line/shell의 `delta_Jbar <= w^T u`를 저장한다. profile
  support가 fine grid에서 하나라도 잘리면 허용하지 않는다.
- [x] small-grid long-double 독립 합산 oracle, 기존 물리 Jbar bit identity, zero-error,
  negative error 거부, truncated-support 거부를 봉인했다. production은 solver envelope를
  8회 정련하지만, 이는 증명 상계만 줄이며 물리 J를 변경하지 않는다.
- [x] `OpacityState::jbar_line_det_error_upper`를 sole local owner로 두고 deterministic
  radiation commit의 canonical `se`에 그대로 싣는다. 음수 local bound는 dual commit
  전체를 원자적으로 거부하며, A2-10은 canonical `jbar.se`가 owner local bound와 exact
  equality일 때만 소비한다. local array/verified flag가 없으면 R6/A2-10 모두
  fail-closed한다.
- [x] 전역 `jbar_line_det_exact_absolute_error_bound`를 승인식에서 완전히 제거했다.
  정적 census의 남은 5개 참조는 구조체 저장, 초기화, 생산자 대입, R6 blocked/identity
  진단 출력뿐이다. 따라서 이전 flight처럼 하나의 전역값을 109,014,300 cells에
  uncertainty로 재사용할 경로가 없다.
- [x] 마지막 local-only 상태에서 component/profile/dual-commit의 normal·strict·
  ASan/UBSan이 모두 PASS했다. component oracle의 최대 actual/local-bound ratio는
  `5.504e-01`; strict/sanitizer SHA는 component
  `f9b65acad03d05f1978b23adcf717818c2e476117d47ffc299dd9e3bbc47dbe5` /
  `526ee71461c41f786b794645b88db7838c05201e217e5601b010d18b4d6c4fe2`, profile
  `f0fe71c17ab38921532ebc568599efabdcf018fd746272c85abe026db1c86d47` /
  `094e369db842f79a61299f5faa666c638036fcd788a945d4f8c7e5266f66eb17`, dual commit
  `aca1613e5bc3b250a873d6632de59b072d5681f8fda508a45daf9dcca9bdff64` /
  `0acce20eb180866ad2aeb85ce1bbb13cda8437c5b4c4010535a7deef3522a591`다.
- [x] final CPU OpenMP/CUDA sm80·86·90 전체 링크, header closure 30/30,
  `git diff --check`, battery D19/K7/Z12/CP4가 PASS했다. CPU/CUDA SHA는
  `a1d3f7d31a8ea8bb2386746e72f007d12894232b90d8f7fec9d5383b41b70c26` /
  `f57e55e0cd71a6213d83e65dad3e80dca9b77c4ac2e8d9ca9b75a2247d0846fd`, battery log는
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_local_only_final_2026-08-09.log`,
  SHA `00e7257558b6e5fbdf45c686cd8e4ea8264118cc3b75a6e93ca726e9863c0c29`다.
- [x] full H200 flight job `251976`을 20 iterations, CUDA SHA
  `f57e55e0cd71a6213d83e65dad3e80dca9b77c4ac2e8d9ca9b75a2247d0846fd`로
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T073623Z_f57e55e0cd71`에
  제출했다. 동일 SHA의 4-iteration backfill pilot job `251977`도 제출했으며, 이전
  실측 peak RSS 약 76 GiB에 대해 128 GiB, 8 CPU, 4시간으로 요청했다. 두 job 모두
  아직 `PENDING/Priority`; login node에서는 model을 실행하지 않았다.
- [x] `scripts/check_det_local_jbar.py`를 추가했다. iteration마다 positive-sliding
  status, residual qualification, verified component envelope, 정확히 8 refinements,
  R6 component/profile interval, canonical generation, `Q_E=2,180,286`과
  109,014,300 valid cells를 검사하며 BLOCKED marker가 하나라도 있으면 fail한다.
  synthetic positive/poison negative가 PASS했고, 이전 global-bound flight `251966`은
  line 15/shell 4 blocked evidence 때문에 의도대로 FAIL한다.
- [x] successful A2-10 endpoint가 침묵하던 계측 공백을 닫았다. diagnostic 전용
  `[A2-10][LINE-NET-CELL-FINITE]`가 LOWER/UPPER의 line 15/shell 4와
  line 17794/shell 26에 대해 raw tau/chi/eta/Jbar, certified local bound,
  absorption/net/signed rate/uncertainty/cancellation condition을 보존한다. 계산,
  sign 판정, publication에는 영향을 주지 않으며 `clamp=floor=jitter=0`을 명시한다.
- [x] local-Jbar checker도 위 네 endpoint key가 iteration마다 정확히 한 번 존재하고,
  모든 값이 finite이며 `Jbar_local_bound >= 0`, `|signed_rate| > uncertainty`,
  status가 finite heating/cooling이고 repair가 0인지 검사한다. synthetic positive와
  negative-bound poison이 PASS했다.
- [x] endpoint 계측 상태에서 CPU/CUDA 전체 링크, A2-10 selftest, header closure 30/30,
  `git diff --check`, battery D19/K7/Z12/CP4가 PASS했다. CPU/CUDA SHA는
  `a7017e351df1313b3688114ae888aaba6482043ff091940f7fc8a9de0ce966be` /
  `6ed1034562a978d1a159b1536b8f1863e897a5a71bbc8914166da31f9ebb2732`, checker SHA는
  `c78b38e4575672ffe948968c4fcbcdc180bb6d493c986651e70ae703be05562e`, battery log
  SHA는 `7aa96e69af540b1fbbdde46ce1cb5e4226a33260a8e76b3d06de99529535401d`다.
- [x] 아직 실행되지 않은 구형 diagnostic pilot `251977`만 취소하고 새 SHA의
  4-iteration H200 diagnostic pilot `251978`을
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T080842Z_6ed1034562a9`에
  제출했다. 8 CPU/128 GiB/4시간이며 현재 `PENDING/Priority`; 20-iteration
  독립 flight `251976`은 보존했다.
- [x] scheduler가 제출 시각이 빠른 full flight `251976`을 diagnostic pilot보다
  먼저 선택하지 않도록 `251976`을 user hold했다. 이는 취소가 아니며 staged input과
  run root를 그대로 보존하는 가역적 순서 조정이다. pilot `251978` 완료·판정 후에만
  release한다. 현재 H200 8장이 모두 사용 중이어서 `251978`은 `PENDING/Priority`다.
- [x] `syn104`를 read-only로 직접 감사했다. Slurm은 H200 8/8장을 할당 중이고,
  `nvidia-smi` 순간 GPU utilization은 `100,66,100,35,100,100,35,43%`; 모든 GPU에
  실제 CUDA process와 3.2--80.2 GiB device allocation이 있었다. fake/idle card는
  없으므로 다른 job allocation을 Slurm 밖에서 점유하지 않는다. pilot `251978`은
  정상 queue에서 첫 합법적 H200 반환을 기다린다.
- [x] accelerator 범위를 재감사했다. `lumina_cuda`는 sm80/sm90 fatbin이며 H100
  NVL은 95,830 MiB/CC 9.0, A100 SXM4는 81,920 MiB/CC 8.0라 80,000 MiB gate를
  만족한다. 향후 batch는 `h200,h100,a100` multi-partition과 실제 GPU name/memory/
  compute capability 봉인을 지원한다. bash/shellcheck와 `sbatch --test-only` PASS.
  다만 test-only 예상 시작은 새 H200 8/18, H100 8/24, A100 12/12이고 기존 H200
  pilot `251978`은 8/16이므로, 현재 pilot은 교체하지 않는다. smoke는 세 GPU를
  허용하되 최종 reference는 실제 architecture를 명시하고 H200과 분리 판정한다.

다음 5단계:

1. [진행 중] diagnostic pilot `251978`의 H200 배정을 기다려 SHA/env/GPU 봉인과
   model 기동을 확인한다. full flight `251976`은 대기열에 그대로 보존한다.
2. component/profile envelope의 verified flag, refinement 수, min/max와 이전 두
   endpoint line 15/shell 4 및 line 17794/shell 26의 finite local bound를 수집한다.
3. 전체 `Q_E=2,180,286`, 50 shells, 109,014,300 cells가 local-qualified canonical
   Jbar로 publication되고 A2-10 signed line-net owner가 이를 소비하는지 확인한다.
4. raw negative/floor/cap/clamp/jitter/부분 publication이 0이고, 실패 시 raw 원인과
   verifier 상태를 보존했는지 flight 산출물과 SHA로 봉인한다.
5. flight가 통과한 뒤 동일 line identity의 finite CMFGEN 물리량 비교를 시작하고,
   DET loop A master + lagged MC(`it -> shared material it+1`) coevolution barrier로
   진행한다.

## 2026-08-10 — 4×A40 exact multi-GPU direct prototype

- [x] production과 분리된 `cmf_exact_multigpu_direct_solve()`를 구현했다. 66-ray
  production geometry를 contiguous ray range로 나누고 각 경계의 다음 ray 하나를
  halo로 재계산한다. peer access, Unified Memory, memory pooling은 사용하지 않는다.
- [x] device별 partial `J(shell,nu)`를 큰 impact-parameter block부터 작은 block까지
  host에서 고정 순서로 합산한다. 모든 실패와 cap exhaustion에서 caller `J`는
  byte-unchanged이고 negative/nonfinite는 그대로 차단한다.
- [x] A40 4장 Slurm job `252098` PASS. CPU direct 대비 최대 상대차 `6.805e-16`,
  one-GPU 대비 `5.856e-16`, 2/3/4장 partition 전수 최대 `3.075e-16`; 반복수는
  모두 20회다. 두 실행 byte-identical, compute-sanitizer 0 errors,
  repair/floor/cap/clamp/jitter=0이다.
- [x] run root
  `/gpfs/kjhan/lumina/cmf_multigpu_prototype/mgpu_20260809T150124Z_57b252862954`,
  staged binary SHA
  `57b2528629545a7761ef096e7dabc181ab83faae42d8a83dbe864930b73c5ea0`.
- [x] exact allocation model에서 production grid direct prototype의 max/device는
  1장 103.237 GiB, 2장 54.281 GiB, 4장 29.803 GiB, 8장 17.564 GiB다.
  따라서 direct prototype만 보면 4×A40과 8×A10이 들어가지만, full Lumina CUDA
  state와 positive-envelope는 아직 포함하지 않은 수치다.
- [x] CPU positive/direct selftest 1/4 thread PASS, header closure 31/31,
  strict sm86 prototype compile, CPU 및 sm86 production link, `git diff --check` PASS.
- [x] sm80/sm86/sm90 production fatbin link PASS, local SHA
  `0dbb8fc0bd1d1e89d020d4f3d0e3b4a3dc8e3d01c25137dc2ffb0537492cda8c`.
  prototype은 production source/dispatch를 건드리지 않았으므로 queued H200 pilot의
  staged SHA는 바꾸지 않았다.
- [x] 전체 D19/K7/Z12/CP4 gate battery PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_multigpu_prototype_2026-08-10.log`,
  SHA `9f7aeb6e7f57b3ae30b245dfa9f66cd0377b133976b9e03b3bb696ca8735d8de`.
- [x] 상세 원장:
  `docs/CMF_EXACT_MULTIGPU_PROTOTYPE_2026-08-10.md`.

다음 5단계:

1. [기존 순서 유지] H200 diagnostic pilot `251978`을 그대로 보존하고 실제 full
   Lumina CPU MaxRSS/GPU VRAM peak를 수집한다.
2. direct oracle이 아니라 subtraction-free positive affine-transform monoid를
   ray-sharded CUDA에 이식한다. production 최대 drift≈47,649 bins에서 `O(beta)`
   direct를 사용하는 것은 금지한다.
3. multi-GPU lower/upper directed sweep와 고정순서 outward reduction을 구현하고
   componentwise `u >= |r| + K u`를 최종 재검증한다.
4. 기존 NLTE/transport device state와 shard가 한 A40에 공존 가능한지 실측한다.
   불가능하면 lifetime 분리 또는 별도 owner sharding으로 해결하며 paging/fallback은
   사용하지 않는다.
5. small-grid direct oracle → reduced grid → 4×A40 full-grid smoke 순으로 통과한 뒤에만
   production dispatch와 finite CMFGEN 비교 계획을 제시한다.

## 2026-08-10 — 4×A40 positive monoid + directed envelope

- [x] `cmf_exact_multigpu_positive_solve()`에 CPU와 같은 subtraction-free affine
  transform 및 two-stack reverse composition 순서를 이식했다. 각 segment work는
  beta와 무관한 `O(n_bins)`이고 production dispatch에는 연결하지 않았다.
- [x] `cmf_exact_multigpu_apply_positive_bounds()`가 source, segment, angular
  reconstruction, shard partial, 고정순서 host 합산을 모두 lower/upper directed
  binary64로 평가한다. lower/nearest/upper는 전 cell ordering 확인 후 함께 publish된다.
- [x] `cmf_exact_multigpu_positive_solve_envelope()`가 zero fixed source/zero inner
  boundary의 동일 upper operator로 `K*u`를 계산하고 componentwise
  `u >= |F(J)-J| + K*u`를 검증한다. 실패 시 J와 error field가 모두 byte-unchanged다.
- [x] nearest-positive job `252101`, directed-bound job `252102`, 최종 supersolution
  job `252103`이 모두 4×A40 `syn06`에서 PASS했다. 최종 run root는
  `/gpfs/kjhan/lumina/cmf_multigpu_prototype/mgpu_20260809T153049Z_fadc17e7377f`,
  binary SHA는
  `fadc17e7377f6e0ed4f7a7188466e5da141fd9ba1f4f51aab34a5b1edb4a9d30`이다.
- [x] CPU positive 대비 `6.199e-16`, 1-GPU 대비 `4.349e-16`, 2/3/4 partition
  최대 `3.661e-16`; directed nearest는 CPU 대비 `5.905e-16`, partition 최대
  `4.348e-16`이다. 전 cell `lower<=nearest<=upper`, 최대 상대 폭 `6.954e-15`다.
- [x] residual upper max `9.487e-20`, refined local envelope
  `[2.604e-20,1.021e-19]`, independent direct observed/envelope 최대 비율
  `3.546e-01`로 전 cell coverage가 PASS했다.
- [x] 두 실행 byte-identical, compute-sanitizer 0 errors,
  repair/floor/cap/clamp/jitter=0. output manifest SHA
  `06391eeb5bf46de5303e0fc3d7a85adc64067d4ee53d78f37e83140a906e179a`,
  footer SHA `ed7a79c60887fedc2167ad98e0c3a2253b396a6c39464d416461f54cdaafe749`.
- [x] 보수적 max positive window 47,649를 포함한 production-grid prototype
  allocation은 4×A40에서 max/device `31.354 GiB`, 8×A10에서 `19.092 GiB`다.
  이는 full Lumina NLTE/transport device state를 포함하지 않는다.
- [x] CPU exact 1/4-thread, strict sm86 compile, CPU/sm86 link, header closure 31/31,
  `git diff --check` PASS. 전체 D19/K7/Z12/CP4 battery도 PASS했고 로그는
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_multigpu_positive_envelope_2026-08-10.log`,
  SHA `f6f5f474a0331f6dc47ba2f8b34bb03043585070913002d9220aa5f4a55d7491`다.
- [x] production source/dispatch 변경 없이 sm80/sm86/sm90 fatbin link를 다시 봉인했다.
  local SHA는
  `b6a89050d93e366304ee8f024eaba28cd2722bd26d9035fa531205e36250d700`이며,
  H200 pilot `251978`의 기존 staged SHA는 그대로 보존한다.
- [x] 상세 원장:
  `docs/CMF_EXACT_MULTIGPU_PROTOTYPE_2026-08-10.md`.

다음 5단계:

1. [기존 순서 유지] H200 diagnostic pilot `251978`을 보존하고 full Lumina의 실제
   CPU MaxRSS 및 GPU VRAM peak를 수집한다.
2. 매 `K*u`마다 device allocation을 반복하는 prototype을 persistent multi-device
   context로 바꾸되 transactional publication과 고정 reduction 순서를 보존한다.
3. 현재 one-thread-per-ray sequential frequency walk의 production 성능을 측정하고,
   two-stack binary64 grouping을 바꾸지 않는 범위에서 frequency 병렬화를 설계한다.
4. 기존 NLTE/transport state와 positive shard의 A40 공존을 실측한다. 부족하면
   lifetime 분리 또는 owner sharding으로 해결하고 paging/fallback은 금지한다.
5. reduced grid → 4×A40 full grid positive/envelope smoke를 봉인한 뒤에만 production
   dispatch와 same-identity finite CMFGEN 비교 단계의 승인을 요청한다.

## 2026-08-10 — persistent 4×A40 componentwise envelope

- [x] envelope solve 안에서 반복되던 매 `K*u` device allocation/free를
  `PersistentBoundContext` 한 번의 생성으로 교체했다. geometry, static coefficient,
  boundary, shard partial 및 work buffer를 재사용하며 고정 host reduction 순서와
  transactional publication은 그대로다.
- [x] selftest가 persistent context 초기화 횟수와 실제 operator 호출 수를 독립적으로
  검사한다. job `252105`에서 `persistent_contexts/bounds/upper=1/15/12`로 PASS했다.
- [x] 4×A40 `syn06` job `252105`는 `COMPLETED 0:0`, 13초였다. run root는
  `/gpfs/kjhan/lumina/cmf_multigpu_prototype/mgpu_20260809T154516Z_175bd7c3845a`,
  staged binary SHA는
  `175bd7c3845ab625426e746af63638ae4cda528293bd9381fc8d59f23fa18c72`다.
- [x] 새 counter token만 제거하면 이전 job `252103`의 모든 출력 수치와 byte-identical이다.
  CPU positive 대비 `6.199e-16`, 1-GPU 대비 `4.349e-16`, partition 최대
  `3.661e-16`, observed/envelope 최대 비율 `3.546e-01`이 그대로다.
- [x] 같은 바이너리 두 실행은 byte-identical(SHA
  `1eaffc08889585d9f8f5a3288260ee1297f7013d89c92f0f2daf028a35c5ca46`)이고,
  compute-sanitizer는 0 errors다. repair/floor/cap/clamp/jitter는 모두 0이다.
- [x] CPU exact 1/4-thread, strict CUDA-host/pedantic C compile, CPU/sm86 link,
  header closure 31/31, `git diff --check`가 PASS했다. sm80/sm86/sm90 cubin을 모두
  포함한 재봉인 fatbin SHA는
  `0bf6cd95140be6c0fd9e87885a031482716b14371c3632d97523b5912ad180fe`다.
- [x] 전체 D19/K7/Z12/CP4 battery PASS. 로그는
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_multigpu_persistent_envelope_2026-08-10.log`,
  SHA `2587726c364a2fdf4083a94aa699164244b0cb36b645b6d4274e05fc3b780272`다.
- [x] production source/dispatch는 아직 연결하지 않았고 H200 diagnostic `251978`과
  user-held full job `251976`의 staged 입력도 변경하지 않았다.
- [x] 상세 원장:
  `docs/CMF_EXACT_MULTIGPU_PROTOTYPE_2026-08-10.md`.

다음 5단계:

1. [기존 순서 유지] H200 diagnostic `251978`이 시작되면 full Lumina의 CPU MaxRSS와
   GPU VRAM peak를 수집하고 기존 staged SHA를 검증한다.
2. production에 가까운 reduced-grid fixture로 persistent 4×A40 wall time, device별
   peak VRAM, context 재사용 횟수를 계측한다.
3. one-thread-per-ray sequential frequency walk의 병목을 분해하되 two-stack binary64
   grouping과 directed rounding 순서는 바꾸지 않는 frequency 병렬화 후보를 만든다.
4. 기존 NLTE/transport device state와 positive shard를 A40에서 함께 실측한다. 부족하면
   lifetime 분리 또는 owner sharding만 사용하며 paging/fallback은 금지한다.
5. reduced grid에서 CPU/1GPU/4GPU directed envelope 일치 → 4×A40 full-grid smoke를
   봉인한 뒤 production dispatch와 same-identity finite CMFGEN 비교로 넘어간다.

## 2026-08-10 — production-shaped 8k split A40 flight

- [x] toy06 실제 50-shell geometry, 66 rays, production dlognu
  `2.7797007933179339e-6`, 총 drift `47649.254516728804`, max positive window
  9,108을 고정하고 frequency slice만 8,192 bins로 줄인 reduced contract를 만들었다.
  finite coefficient는 synthetic이며 CMFGEN 물리 비교가 아님을 출력과 footer에
  명시했다.
- [x] 독립인 1-GPU와 4-GPU solve를 별도 Slurm job으로 동시에 실행하고, 각 `J`와
  componentwise error-upper를 self-describing binary로 저장하는 split mode 및 사후
  전-cell 비교기를 구현했다. 계산식, iteration cap, tolerance, 물리값은 바꾸지 않았다.
- [x] common run root
  `/gpfs/kjhan/lumina/cmf_multigpu_reduced_split/split_20260810T001144Z_7fa25bf4e3e1`,
  binary SHA
  `7fa25bf4e3e1763f18c324f1244377c43f6d3b0676732055ec2d2bdf238dc8b7`로
  4-GPU job `252352`가 `COMPLETED 0:0`했다. solve `1711.016270583 s`, 13 iterations,
  context/bounds/upper `1/7/4`, peak VRAM `415/421/415/415 MiB`다.
- [x] 1-GPU job `252351`의 solve는 `3047.908084386 s`, 같은 13 iterations와
  `1/7/4`, peak 755 MiB로 정상 결과를 썼다. 이후 4-trace를 하드코딩한 VRAM 요약기가
  1 trace를 거부해 job은 `FAILED 70:0`이다. 계산 실패로 숨기지 않고 driver 결함으로
  분리했으며, 요약기에 `--expected-devices`를 추가해 15,234 samples를 재검증했다.
- [x] 409,600/409,600 cells가 one/four combined envelope를 통과했다. max relative
  difference `1.6222759467960672e-15`, max envelope ratio
  `5.4656493245645533e-05`, repair/floor/cap/clamp/jitter=0이다. four-GPU finite J는
  `[8.8332793258264307e-08,2.4965020783600907e-05]`지만 synthetic이므로 finite
  CMFGEN 재현 판정은 아니다.
- [x] cross-node operational speedup은 `1.781343717642`, max per-card VRAM 분산비는
  `1.793349168646`이다. aggregate VRAM은 4-GPU 쪽이 더 크므로 총메모리 절감으로
  주장하지 않는다. one/four result SHA는
  `37c709e591972631260efe83f710856c39c2c3506d81c6c5663050765b5a49d9` /
  `8adde94b741fda31a1febf79b515ddd82f70eb89c3fef993b326e0f00ceb4fa8`다.
- [x] 32k job `252346`은 33:07에도 첫 one-GPU solve 내부여서 성능 blocker 증거를
  보존하고 취소했다. 직렬 8k job `252350`도 split이 안정화된 뒤 30:17에 취소해
  놀던 A40 3장을 반납했다. timeout 보험 `252360`은 원본 수치 결과 통과 뒤 6:04에
  취소했다.
- [x] full D19/K7/Z12/CP4 및 serial/parallel equivalence PASS. 로그
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_multigpu_reduced_split_2026-08-10.log`,
  SHA `79453c756f21d4c5bc5571176a966db147d5840361103c7fd84b129badc7aa28`다.
  production dispatch와 H200 staged input은 변경하지 않았다.

다음 5단계:

1. equal-ray-count contiguous partition의 실제 path-work 불균형을 수치화하고,
   angular halo/reduction identity를 보존하는 weighted contiguous 경계를 구현한다.
2. weighted partition이 1/4-GPU componentwise envelope `409600/409600`을 그대로
   통과하는지 reduced 8k에서 재비행하고 device utilization/벽시계를 비교한다.
3. 한 ray thread가 전체 frequency를 순차 처리하는 핵심 병목에 대해 two-stack
   affine monoid의 정확한 binary64 grouping과 directed rounding을 보존하는
   within-ray parallel scan 설계를 만든다.
4. H200 diagnostic `251978`이 시작되면 full Lumina CPU MaxRSS/GPU VRAM peak를
   봉인하고, A40 shard와 기존 NLTE/transport state의 coexistence 가능성을 판정한다.
5. reduced-grid 최적화와 full-state 메모리 증거가 모두 통과한 뒤 4×A40 full-grid
   smoke를 제출한다. 그 다음에만 same-identity finite CMFGEN 물리량 비교로 넘어간다.

## 2026-08-10 — weighted contiguous A40×4 폐합

- [x] equal ray 경계 `[0,16,33,49,66]`의 computed work
  `849/729/408/136`을 확인하고, active ray/segment 누적량으로 경계
  `[0,10,20,34,66]`을 만들었다. 새 computed work는 `550/535/570/496`, owned
  min/max는 `490/539`다. direct/positive/apply-bounds/persistent-context가 같은
  경계를 사용한다.
- [x] device별 lower<=nearest<=upper 검사와 실패 device/shard/segment/bin 계측을
  추가했다. `sqrt(fmax(0,1-p^2/r^2))` 수치 clamp는 제거했고 analytic mu-squared가
  음수/비유한이면 원값을 숨기지 않고 실패한다. floor/cap/clamp/jitter/repair는 0이다.
- [x] 1024-bin 실패를 알고리듬 문제로 덮지 않고 물리 GPU까지 추적했다. 같은 binary가
  `syn06`에서 실패하고 `syn07`에서 통과했으며, device 순서 변경 시 실패가 UUID
  `GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6`을 따라 이동했다. 그 UUID를 제외한
  `syn06` 대체 4장 job `252390`은 PASS했다. compute-sanitizer는 0 errors였으므로
  해당 UUID를 이 workload에서 quarantine한다.
- [x] 최종 clamp-free SHA
  `1a0480fe321b89c4036ded02b9809a2363024cd67c030012795b6e3fcd9a7a31`의
  diagnostic `252395`와 selftest/memcheck `252394`가 `syn07`에서 PASS했다.
- [x] 최종 8k jobs `252396/252397`은 각각 `371.818018868/371.813001576 s`, 두
  result가 byte-identical(SHA
  `aa43bb667c8602691ce89f1169ed014a90474d759a48c0f68b364e2eb7e57b9b`)이다.
  peak VRAM은 `373/373/399/515 MiB`; finite J는
  `[8.8332793258264307e-08,2.4965020783600907e-05]`다.
- [x] 기존 독립 1-GPU 결과와 409,600/409,600 cells가 combined envelope를 통과했다.
  max relative `1.5503706747130466e-15`, envelope ratio
  `1.0930758301523256e-04`, numerical repairs 0이다. coefficient는 synthetic이므로
  finite CMFGEN 물리량 재현은 아직 아니다.
- [x] 전체 D19/K7/Z12/CP4 및 serial/parallel equivalence PASS. 최종 로그 SHA는
  `cb31e365d95a4a09dd7b9b4871116b53e2a37365854e8311ce52f07fc2fd2c5d`다.
  production dispatch와 H200 staged input은 변경하지 않았다.

다음 5단계:

1. 동일 binary의 equal/weighted runtime A/B로 partition 자체의 wall-time 효과를
   분리한다. 현재 weighted 구현은 correctness candidate로 유지한다.
2. two-stack binary64 grouping과 directed rounding 순서를 그대로 보존하는 within-ray
   parallel scan 설계를 만든다.
3. small-grid direct oracle에서 scan의 lower/nearest/upper와 byte 결정성을 먼저 증명한다.
4. H200 diagnostic `251978`에서 full Lumina CPU MaxRSS/GPU VRAM을 수집하고 A40 shard
   state와 coexistence를 판정한다.
5. full-grid A40 smoke 뒤 same-identity finite CMFGEN 비교로 진입한다.

## 2026-08-10 — 동일 binary equal/weighted A/B 폐합

- [x] 기존 public API의 weighted 기본값은 보존하고, 실험용 explicit partition enum/API와
  `CMF_MGPU_REDUCED_PARTITION=equal|weighted` benchmark mapping을 추가했다. solver core가
  환경변수를 직접 읽지 않으며 production dispatch도 건드리지 않았다.
- [x] 동일 binary SHA
  `f9f84912ee5dd84c5cb449d9cca186835a41b57f68e5e4cb215f1ad4759a34eb`, 동일
  `syn07` UUID/순서에서 jobs `252398/252399`와 `252400/252401` 두 A/B를 완료했다.
  equal은 `458.240888931/458.235787507 s`, weighted는
  `371.797192502/371.819500103 s`다.
- [x] 평균 `458.238338219/371.8083463025 s`; weighted speedup
  `1.232458450102x`, wall-time 감소 `18.861362026674%`로 partition 자체의 효과를
  분리했다.
- [x] mode별 반복 결과가 각각 byte-identical이다. equal SHA는
  `5db7dd2f801190e45826f9548abe86c6a5b78d796440f8790f6e766439e439ec`, weighted
  SHA는 `aa43bb667c8602691ce89f1169ed014a90474d759a48c0f68b364e2eb7e57b9b`다.
- [x] equal/weighted 차이 max relative `8.7441847530776422e-16`, max absolute
  `1.3552527156068805e-20`, combined envelope `409600/409600`, ratio
  `1.0755378417557555e-04`로 PASS했다. numerical repair/floor/cap/clamp/jitter=0이다.
- [x] equal peak VRAM `415/421/415/415 MiB`, weighted
  `373/373/399/515 MiB`다. weighted는 시간은 이기지만 single-card 메모리는 더 크므로
  메모리 절감으로 주장하지 않는다.
- [x] D19/K7/Z12/CP4 및 serial/parallel equivalence PASS. 로그 SHA는
  `e0c4f0e596db822862d89319873aec84ff773dadc616073c0fb2a24cecc0d319`, 통합 ledger는
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/cmf_multigpu_partition_ab_final_2026-08-10.log`다.
- [x] 판정: weighted contiguous segment-work partition을 default candidate로 유지한다.
  값은 finite synthetic transport이며 finite CMFGEN 물리량 재현은 아직 아니다.

다음 5단계:

1. 현 two-stack affine monoid의 binary64 grouping과 directed rounding 순서를 보존하는
   exact within-ray parallel scan 명세를 작성한다.
2. small-grid direct oracle에서 lower/nearest/upper enclosure와 반복 byte 결정성을 먼저
   증명한다.
3. H200 diagnostic `251978`이 시작되면 CPU MaxRSS/GPU VRAM을 봉인하고 full state와
   A40 shard state의 coexistence를 판정한다.
4. 증명된 scan만 prototype에 구현하고 reduced CPU/1GPU/4GPU gate를 재실행한다.
5. A40 full-grid smoke 뒤 same-identity finite CMFGEN 물리량 비교로 진입한다.

## 2026-08-10 — exact within-ray scan 명세 폐합

- [x] 정본 reverse composition의 node 순서를
  `multiply(T_B,T_A) → multiply(T_B,E_A) → add(E_B,attenuated)`로 고정했다.
  lower/nearest/upper 모두 node별 rounding 결과가 다음 node operand가 된다.
- [x] 일반 Blelloch/Hillis–Steele/tree scan을 기각했다. 세 transform의 explicit hex
  witness에서 괄호 변경 결과가 lower/nearest/upper 모두 1 ulp 갈라지므로 현재
  binary64 operator를 모든 입력에서 bitwise 재현할 수 없다.
- [x] exact 후보로 canonical two-stack transfer-epoch replay를 명세했다. W회 pop마다
  queue가 raw value로 transfer된다는 점을 이용해 boundary-back Q, transferred-front F,
  new-back P의 serial fold 괄호는 그대로 두고 세 chain과 epoch만 병렬 스케줄한다.
- [x] `scripts/verify_cmf_exact_epoch_formula.py`에서 serial push/transfer/pop과 epoch 공식을
  독립 구현해 6,588/6,588 lower/nearest/upper aggregate pairs가 bit-identical임을 확인했다.
  SHA는 `3f5601b99cbc7f9a5013e4c4867fa1d0fe8bab58186440f13bef7915d7fa82b2`다.
- [x] reduced 8k max W=9,108은 worst segment가 한 epoch뿐이고, full 2,013,113 bins /
  W=47,649는 최대 43 epochs임을 분리했다. small-grid 증명 전에는 성능 향상을 주장하지
  않는다.
- [x] G0 nonassociation부터 G7 sanitizer/hygiene까지 aggregate bits, logical node mapping,
  segment bits, full-sweep envelope, scheduling invariance, transactional failure의 승인
  기준을 `docs/CMF_EXACT_WITHIN_RAY_SCAN_SPEC_2026-08-10.md`에 봉인했다.
- [x] Fable에는 이 핵심 쟁점 하나만 질의했으나 CLI가 `Exceeded USD budget (0.5)`로
  응답 없이 종료했다. 재질의하지 않았으며 Fable 판정을 받았다고 기록하지 않는다.
- [x] GPU kernel과 production dispatch, DET-loop-A coevolution master, H200 staged input은
  변경하지 않았다. 수치 repair/floor/cap/clamp/jitter도 추가하지 않았다.

다음 5단계:

1. G0/G1을 C/CUDA small-grid selftest로 구현해 aggregate T/E와 canonical logical-node
   mapping을 bitwise 봉인한다.
2. W=0 direct-bin과 한-epoch Q/F/P CUDA prototype으로 segment lower/nearest/upper
   bit identity를 증명한다.
3. multi-epoch 및 epoch-batch scheduling을 추가해 full small sweep, 1/2/4 GPU 결정성,
   fail-closed 계약을 통과한다.
4. compute-sanitizer와 memory model 통과 후에만 reduced 8k A40×4 timing을 실행한다.
5. H200 full-state memory와 합쳐 full-grid 가능성을 판정한 뒤 smoke 및 same-identity
   finite CMFGEN 비교로 이동한다.

## 2026-08-10 — exact epoch scan G0/G1 C/CUDA 폐합

- [x] production dispatch 밖에 `tests/cmf_exact_epoch_scan_selftest.cu`를 추가하고 CPU
  serial two-stack reference, CUDA Q/F/P transfer-epoch kernel, output ownership mapping을
  구현했다. reverse compose는 정본의 multiply/multiply/add와 directed rounding node
  순서를 그대로 쓴다.
- [x] lower/nearest/upper 비결합 witness가 CPU에서 계속 갈라지고 CUDA의 left/right 여섯
  pair가 CPU와 bitwise 일치했다. tree 재결합이 조용히 들어오면 G0가 실패한다.
- [x] 2,196 base × 3 mode = 6,588 cases에서 153,972 CPU/CUDA aggregate `(T,E)` pair와
  153,972 `(epoch,offset,boundary,Q/F/P fold index)` output mapping이 전부 bit-identical이다.
  모든 bin에서 lower≤nearest≤upper도 통과했다.
- [x] sm_80/sm_86/sm_90 fatbin SHA는
  `7024d38fae8e51acb1c418e06c29b9e2cebbae04f7878bd0008b82dc8aa74be3`다.
  A40 `syn07` job `252405`는 `COMPLETED 0:0`; 두 full run stdout/stderr가 byte-identical,
  full log SHA는 `61e762a15960bf44b921f6275290b6488f4065ca0d15f204bd640e09b78c0db5`다.
- [x] sanitizer-smoke는 666 mode cases/29,007 aggregate pair를 실행했고
  compute-sanitizer 0 errors, leak 0 bytes다. numerical repair/floor/cap/clamp/jitter=0이다.
- [x] ledger는
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/cmf_exact_epoch_scan_g01_2026-08-10.log`,
  SHA `12d0be3de592a48e90a900e5205591a6d99b05e4963663bdad0bc15e5a9812cc`다.
- [x] 판정 범위는 G0/G1뿐이다. output mapping은 봉인했지만 primitive마다
  `(input bits,parent,output bits)`를 대조하는 G2, segment 계산 G3, 성능 및 finite CMFGEN
  비교는 아직 통과하지 않았다. production dispatch, DET-loop-A coevolution master,
  H200 staged input은 변경하지 않았다.

다음 5단계:

1. G2 canonical logical-node trace를 구현해 serial과 epoch Q/F/P의 primitive 입력 bits,
   parent, 출력 bits를 전 node에서 대조한다.
2. G3 `W=0` direct-bin 및 one/multi-epoch segment CUDA path를 만들고 upstream zero/nonzero,
   `phi` 0.5 양쪽, repeated high index, W/warp/block 경계를 mode별 bitwise 봉인한다.
3. G4 full small sweep에서 direct oracle/serial positive/epoch positive finite J와
   lower≤nearest≤upper, componentwise envelope coverage를 닫는다.
4. G5/G6 block size·epoch batch·1/2/4 GPU 반복 결정성과 invalid/nonfinite/allocation
   failure의 정확한 logical provenance 및 transactional output 보존을 검증한다.
5. G7 sanitizer/memory model 뒤 reduced 8k A40×4 timing을 실행한다. 그 후 H200 memory,
   full-grid smoke, same-identity finite CMFGEN 물리량 비교 순서로 간다.

## 2026-08-10 — exact epoch scan G2/G3 폐합

- [x] G2 serial queue의 Q/F/P/G 합성을 mul_T/mul_E/add_E primitive로 풀어 mode/epoch/
  chain/node/operand bits/result bits를 기록했다. CUDA replay 338,472/338,472 records가
  bitwise 일치했고, 재계산 Q와 직전 P alias도 통과했다.
- [x] G3 production-shaped CUDA serial segment와 epoch segment를 3,726 mode cases에서
  비교했다. upstream zero/nonzero, phi half 경계, W=0/1/2와 warp/block/n 주변,
  high-index repetition을 포함한 139,788/139,788 values가 bit-identical이다.
- [x] 첫 실패는 W=0 일반 경로가 identity aggregate application을 생략한 1-ulp 구조
  결함이었다. directed `1×intensity`는 실제 node이므로 누락 node를 복원했다. tolerance,
  floor, cap, clamp 또는 결과 수리는 사용하지 않았다.
- [x] A40 job `252408`은 `COMPLETED 0:0`, 반복 byte-identical, compute-sanitizer 0 errors,
  leak 0 bytes다. binary SHA는
  `f89d6a7b1784201487001900c8831c9286e259b9f01b3b999db040981db80f86`다.
- [x] ledger SHA는
  `89d753da92736bad6f59c04333caa5614673a482f1bc5bd90e4892f656e125bb`다.
  production dispatch와 H200 staged input은 아직 변경하지 않았다.

다음 5단계:

1. G4 direct/serial/epoch full small sweep와 componentwise envelope를 닫는다.
2. G5 block size 및 epoch batch cardinality 결정성을 검증한다.
3. G5 1/2/4 A40 shard/reduction 결과와 반복 byte identity를 검증한다.
4. G6 invalid/nonfinite/allocation 실패 provenance와 transactional output 보존을 검증한다.
5. G7 sanitizer/memory model 뒤 reduced 8k A40×4 통합·timing으로 이동한다.

## 2026-08-10 — exact epoch G4–G7 및 production prototype 폐합

- [x] G4 full small sweep는 3 sweeps/9 directed mode sweeps에서 serial/epoch J
  3,456개가 bit-identical이고 direct oracle 1,152/1,152개를 enclosure했다. A40
  `syn07` job `252411`은 반복 byte identity와 compute-sanitizer 0 errors를 통과했다.
- [x] G5는 block `32/64/128/256`, epoch batch `1/2/7/all`의 1,206 schedule run,
  347,328 values를 bitwise 봉인했다(job `252425`). 별도 A40×4 job `252426`은
  1/2/4-device canonical-ray reduction digest `1667cc4c2584f887`을 반복 재현했다.
- [x] G6는 invalid mode/index/workspace, injected allocation/CUDA failure, 실제 NaN을
  포함한 7 cases에서 실패 6건의 public output을 byte 보존하고 성공 1건만 publish했다.
  NaN provenance는 `epoch=0,Q,node=0,source_index=3`이다. job `252427`, sanitizer
  0 errors/leak 0이다.
- [x] `src/cmf_exact_multigpu.cu/.h`에 명시적 epoch solve/apply-bounds/envelope API를
  추가했다. 기존 public direct/positive/bounds/envelope API는 schedule null인 종전 serial
  경로를 그대로 쓴다. 새 경로는 작은 W direct replay와 큰 W global workspace epoch를
  분리하고 block/batch/replay threshold를 report에 기록한다.
- [x] production API selftest job `252432`는 solve, lower/nearest/upper bounds, persistent
  componentwise envelope를 serial과 bitwise 일치시켰다. schedule matrix와 invalid schedule
  transactional 보존도 통과했고 반복 byte-identical, sanitizer 0 errors다. binary SHA는
  `122c5a04d42efeaccd991c090c11b8d7e485b51593c415eff9ca71fa653862e1`이다.
- [x] production-shaped 1,024-bin sanitizer job `252431`은 51,200 cells에서 ordering
  failure 0, compute-sanitizer 0 errors다.
- [x] reduced 8k A40×4 epoch job `252433`은 36.502375973 s에 13 iterations를 완료했다.
  기존 weighted serial 371.813001576 s 대비 `10.185994518577x`, wall time
  `90.182598290464%` 감소다. 409,600-cell result는 기존 serial과 `cmp` byte-identical,
  공통 SHA는 `aa43bb667c8602691ce89f1169ed014a90474d759a48c0f68b364e2eb7e57b9b`다.
- [x] finite J 범위는
  `[8.8332793258264307e-08,2.4965020783600907e-05]`, error-upper 범위는
  `[2.9241481245511831e-17,1.5167598257518997e-16]`이다. epoch peak VRAM
  `373/373/397/517 MiB`는 serial `373/373/399/515 MiB`와 사실상 같으므로 메모리 절감을
  주장하지 않는다. repair/floor/cap/clamp/jitter는 0이다.
- [x] `syn06` job `252429`의 차이는 기존 quarantine UUID
  `GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6` 포함 시에만 재현됐고, 신뢰 노드
  `syn07`의 `252430/252431/252432/252433`은 모두 통과했다. 해당 카드는 계속 제외한다.
- [x] 통합 ledger는
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/cmf_exact_epoch_g47_production_2026-08-10.log`,
  SHA `337e90efe7dafe516e905d08d3fe672422ca756be2024039d3013eebfbcb96f9`다.
- [x] 이 폐합은 finite synthetic-coefficient transport 재현이다. same-identity finite
  CMFGEN 물리량 비교는 아직 수행하지 않았고, production dispatch와 DET-loop-A
  coevolution owner는 바꾸지 않았다.

다음 5단계:

1. H200 diagnostic `251978`에서 full Lumina CPU MaxRSS/GPU VRAM을 수집하고 epoch shard
   state와 공존 가능한지 합산이 아닌 실측으로 판정한다.
2. explicit epoch API를 production owner가 선택할 integration gate를 만들되 기본 serial
   dispatch는 full-grid 검증 전까지 유지한다.
3. 신뢰 A40 네 장에서 full 2,013,113-bin smoke를 실행하고 transactional failure 및
   finite J 범위를 봉인한다.
4. 동일 input identity와 물리량 정의를 고정한 CMFGEN fixture를 준비해 finite J/flux 등
   첫 실질 비교를 수행한다.
5. 비교가 통과한 뒤에만 production dispatch 승격 여부를 판정한다.

## 2026-08-10 — compact full-grid와 실제 production CMF 계수 A/B 폐합

- [x] A40×4 full 2,013,113-bin/100,655,650-cell jobs `252438/252443`의
  `result.bin`은 cross-binary byte-identical이다. SHA-256은
  `dcda52e5a97cbc92e95522ba92406ad54706354bcbee8fd9511acf70bf0e028c`, repeat
  시간은 `895.507151513 s`, finite synthetic J는
  `[5.9347823429553942e-08,9.7311679856602645e-05]`다.
- [x] production fine owner에 `LUMINA_CMF_FINE_MGPU_DEVICES`를 연결했다. unset/0은
  CPU positive-sliding, 양수는 정확히 그 수의 visible GPU를 요구한다. 실패 시 같은
  attempt의 CPU fallback은 없다.
- [x] `LUMINA_CMF_FINE_MGPU_AB=1`은 동일 조립 상태를 private CPU와 GPU buffer에서
  각각 풀고, 전 셀 finite/nonnegative, max relative J, combined directed envelope를
  통과해야만 GPU 결과를 R6에 넘긴다. 결과 수리용 floor/cap/clamp/jitter는 없다.
- [x] 최종 binary SHA
  `9549e375aeaf439aace587eb4b02b42051b2cac1c3d8910c46d6e767aea08f8b`의 job
  `252447`은 owner 반복 byte identity, transactional negative input, invalid config,
  compute-sanitizer 0 errors를 통과했다.
- [x] sealed production deck job `252448`에서 CPU/GPU 모두 45회 수렴했다. finite J는
  CPU `[8.4086208255147163e-82,1.9072381379446642e-4]`, A40×4
  `[8.408620825514714e-82,1.9072381379446645e-4]`; max relative 차이
  `3.1710829615213259e-15`, combined-envelope ratio max
  `0.25924739579810846 < 1`로 100,655,650셀 전부 PASS다.
- [x] R6는 Q 1,603,732/E 2,180,286, valid E 전부, partial/unsampled 0이다. peak host
  RSS는 `84,688,916 KiB`, production coexistence VRAM은
  `38545/21245/22569/21341 MiB`다.
- [x] exact/R6 scope report 뒤 범위 밖 A2-10 census가 네 idle A40를 계속 점유해 job을
  `01:54:53`에 취소했다. Slurm CANCELLED는 exact 실패가 아니다. 후속 관측은 기존
  line15/shell4가 `OK_HEATING`, line15/shell10이 `UNRESOLVED_CANCELLATION`이다.
- [x] 이는 외부 CMFGEN executable과의 독립 비교가 아니라 CMFGEN-derived sealed deck에서
  production이 조립한 동일 물리 계수의 CPU owner↔A40×4 재현이다. synthetic-only gap은
  닫지만 외부 코드 비교를 했다고 쓰지 않는다.
- [x] 통합 ledger SHA는
  `50d1811ac41dec475816f7bdf567c3276096d628e4db5916ba21a58085bac0c4`다.

판정: **multi-GPU exact production 기능 구현과 실제 finite-value 동일-input 검증은
완료**다. 기본값은 portable CPU serial이며 positive device 요청에서 multi-GPU가 켜진다.

다음 5단계(새 승인 대상):

1. exact solve/envelope/host reduction phase timer와 device별 work/utilization을 계측한다.
2. 동일 production shape에서 H200 1장과 A40×4를 같은 수렴 조건으로 직접 비교한다.
3. validation 전용 exit-after-R6 경계를 추가해 무관한 A2-10이 GPU allocation을 잡지 않게 한다.
4. 외부 CMFGEN 또는 ARTIS의 동일 정의 finite J/flux fixture와 독립 코드 비교를 수행한다.
5. 위 결과 뒤 multi-GPU default와 DET-loop-A 다중 outer iteration 승격을 판정한다.

## 2026-08-10 — 승인된 후속 5단계 실행 중

- [x] `CMFMultiGPUReport`에 fixed-point initialization/source/H2D/device sweep/
  D2H/host reduction/convergence와 envelope setup/bounds/residual/verify/refine/
  publication/cleanup 타이머를 추가했다. device별 ray 구간, owned/computed segment
  work, allocation bytes도 production log에 발행한다.
- [x] CPU/GPU 수렴 상대오차 분모에 남아 있던 `+1e-30`을 제거했다. 분모가 정확히 0일
  때 `absolute==0`이면 0, 아니면 infinity로 판정한다. 이는 물리값 수리가 아니라
  convergence diagnostic이었지만, numerical floor 금지 계약을 문자 그대로 적용했다.
- [x] validation 전용 `LUMINA_VALIDATION_EXIT_AFTER_R6=1`을 추가했다. single outer
  iteration + positive multi-GPU device request + CPU/GPU A/B일 때만 허용하며 R6 commit 뒤
  R7/A2-10/spectrum 전에 정상 종료한다.
- [x] `LUMINA_CMF_FINE_EXTERNAL_FIXTURE=1`은 50 shell × 8 registered wavelength의 실제
  fine-bin-centre `J_nu`와 midpoint velocity를 read-only CSV로 발행한다. 독립
  `EDDFACTOR/RVTJ` parser는 같은 scalar `J_nu`를 같은 frequency/velocity 좌표에서
  log-bilinear 평가한다. historical CMFGEN run과 common-state identity는 아직 인증되지
  않았으므로 결과는 `PASS_COMPARISON_NOT_PARITY`만 허용한다.
- [x] 최종 fat binary SHA는
  `653f94b7f9916fdb879fab7e04f53f04a2be337f248ff2d1094f424380e78637`이며 owner job
  `252489`가 4초, rc=0으로 통과했다.
- [ ] fair pair root는
  `/gpfs/kjhan/lumina/cmf_exact_gpu_fair/pair_20260810T084359394144656Z_p2535909_653f94b7f991`다.
  A40×4 `252493`은 `01:49:18`, rc=0으로 완료했고 H200×1 `252492`는 Priority 대기다.
  H200 8장은 모두 실제 process와 nonzero utilization이 확인되어 수동 침범하지 않았다.
- [x] 2026-08-12부터 특별한 장치 제약이 없으면 H200/H100/A100 후보를 동시에 제출한다.
  동일 sealed binary/input의 H100×2 `266000`은 `01:21:20`, A100×2 `266001`은
  `01:25:09`에 rc=0으로 완료했다.
  full-grid 실측 footprint 때문에 143,771 MiB H200은 1장, 95,830 MiB H100과
  81,920 MiB A100은 각각 2장으로 잡았다. 단일 저메모리 카드에 OOM을 감수해 넣지 않는다.
  세 lane의 binary/deck/atomic hash가 서로 같음을 확인했다.
- [x] H100/A100 모두 45회 수렴, max relative J
  `3.1676961234913679e-15`, combined-envelope ratio max
  `0.24242408517713512`, exit-after-R6 PASS다. GPU owner는 H100
  `1178.412445307 s`, A100 `1458.743644238 s`; floor/cap/clamp/jitter/repair는 0이다.
- [x] 2026-08-14 queue audit에서 stale user-held duplicate `251976`을 취소했다. A100와
  `syn101`에 고정돼 9월 10일로 밀린 `273935`는 `det4_accel`로 이름을 바꾸고
  H200/H100/A100 generic 1-GPU 후보로 정상화했다. CF4 `276126`의 모순된
  `TresPerTask=cpu:32`도 실제 `CPUs/Task=16`과 같은 `cpu:16`으로 수정했다. 두 작업의
  scheduler 예상 시작은 8월 16일 15:00이다.
- [x] A40×4는 45회, residual `9.6662779267286222e-09 < 1e-8`로 수렴했다. CPU↔GPU
  max relative `3.1710829615213259e-15`, combined-envelope ratio max
  `0.25924739579810846`; 100,655,650셀 전부 PASS다. exit-after-R6가 R7/A2-10/spectrum
  `NOT_RUN`을 확인했고 floor/cap/clamp/jitter/repair는 0이다.
- [x] A40 fixture 400행 중 CMFGEN domain 내부 352행을 독립 `EDDFACTOR/RVTJ` parser와
  비교했다. median `log10(Lumina/CMFGEN)=-0.015039090168600878`, 범위
  `[-8.000450208246109,0.8002509060020172]`, verdict는
  `PASS_COMPARISON_NOT_PARITY`다. common physical-state identity가 미인증이므로 parity를
  주장하지 않는다.
- [ ] dependency finalizer `252494`는 두 lane 종료 뒤 timing/utilization 비교와 외부
  CMFGEN finite-J 결과를 한 report에 자동 봉인한다.
- [x] 현재 checkpoint ledger는
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/cmf_exact_gpu_fair_a40_external_2026-08-11.log`다.

남은 순서:

1. `252492` H200×1 lane의 8월 16일 배정과 결과를 검증한다.
2. H200/H100/A100/A40 phase timing을 한 비교 report에 봉인한다.
3. GPU 종류별 병목과 장치 수 대비 성능을 판정한다.
4. external comparison을 same-physical-state identity fixture로 강화한다.
5. 그 증거 뒤에만 default multi-GPU와 DET-loop-A 승격 여부를 판정한다.

## 2026-08-15 — DET ownership correction and corrected flight

- [x] 사용자는 이 검증 묶음 전체를 승인했다. 아래 단계 사이에 개별 승인을 다시
  요청하지 않고, 외부 작업을 침범하지 않는 범위에서 다음 판정까지 자율 진행한다.

- [x] `syn101`의 Slurm `IDLE+PLANNED` gap에서 tripwire를 붙여 최종 binary
  `653f94b7f991` flight와 historical diagnostic `6ed1034562a9` mirror를 각각
  A100 GPU 5/4에서 수동 실행했다. 둘 다 Slurm/foreign-PID trip이 아니라 model
  자체 rc=1로 iteration 0에서 fail-closed했다.
- [x] 두 실행은 모두 잘못 봉인된 `single_total=0`, `ION_LOCK=1`,
  `PER_ION_RESCALE=1`이었다. 같은 Si II--III shell 4 pair에서 lower 3500 K
  population `-0.0061392332709293129`(relative `2.2478694603818218e-10`, 63개)와
  upper 140000 K population `-4.5839458986018107e-22`(relative
  `2.5363871044793723e-8`, 10개)를 두 binary가 동일하게 재현했다. comparison
  commit은 0이며 이를 수렴 결과로 사용하지 않는다.
- [x] 이 실패는 새 roundoff 가설이 아니다. 기존 80자리/A/B 원장이 확정한 것처럼
  homogeneous element generator에 두 stage-total 행을 중복 강제한 owner 계약
  위반이다. 값의 floor/clamp/cap/jitter, exp 변수화, diagonal regularization은
  적용하지 않았다.
- [x] 관련 population/candidate/A2-10 selftest를 현재 tree에서 다시 실행해 PASS했다.
  corrected sealed job `302652`를 H200/H100/A100 multi-partition에 제출했다. run root는
  `/gpfs/kjhan/lumina/det_convergence_single_total/det1234_20260815T050821Z_653f94b7f991`,
  binary SHA `653f94b7f9916fdb879fab7e04f53f04a2be337f248ff2d1094f424380e78637`,
  `single_total=1`, `ION_LOCK=0`, `PER_ION_RESCALE=0`, 4 iterations, 1 GPU,
  8 CPU, 128 GiB, 4 h다. deck/atomic/resolved-env seal은 모두 PASS했다.
- [ ] `302652`는 현재 `PENDING/Priority`, scheduler start estimate 미정이다. 확인 시점에
  `syn101` GPU 0--7 모두 다른 세션의 실제 Python process가 있어 수동 침범하지 않았다.

- [x] 2026-08-15 15:14 KST 재점검에서 `syn101` GPU 7
  (`GPU-e545e65b-27fe-058e-807a-0db7ba59d55f`)이 process 0, memory 0 MiB,
  utilization 0%이고 node allocation도 없음을 확인했다. `302652`를 먼저 user hold한 뒤
  같은 sealed run root를 GPU 7에서 tripwire supervisor로 수동 기동했다. 실제
  `lumina_cuda` PID/PGID와 A100 선택, SHA/env seal을 확인한 뒤 Slurm 복제 job만
  취소했다. supervisor는 node에 실제 Slurm allocation이 생기거나 GPU 7에 foreign
  process가 나타나면 우리 child process group만 종료한다.
- [ ] 수동 flight는 원자자료 load 중이다. job id는
  `manual_syn101_20260815T061421Z_1260096`, binary PID는 `1260502`다. 아직 iteration
  commit은 없으며 결과 판정 전이다.

- [x] 17:36 KST의 `eu-stack` audit로 위 작업의 장기 무출력 구간이 전처리가 아니라
  `cmf_error_envelope_refine -> exact_apply_scattering_upper ->
  exact_formal_sweep_bound` CPU exact owner임을 확인했다. sealed env에
  `LUMINA_CMF_FINE_MGPU_DEVICES`가 없어 GPU 7은 NLTE allocation 17.3 GiB만 유지하고,
  8 CPU thread가 100,655,650-cell positive error-envelope를 계산하고 있었다.
  기존 24-CPU baseline 약 3,559 s와 비교하면 현재 2시간 이상은 예상 가능한 배선
  결과지만, 이를 outer iteration마다 네 번 반복하는 것은 비효율적이다.
- [x] DET submit helper에 fail-closed `DET_CMF_FINE_MGPU_DEVICES`와
  `DET_STAGE_ONLY`를 추가했고 manual tripwire는 복수 GPU, CPU thread 수, 독립 cpuset을
  지원한다. single-GPU 동작은 그대로 유지하며 multi-GPU 요청 수와 staged GRES 수가
  다르면 staging 전에 거부한다.
- [x] 같은 binary/deck/atomic identity의 A100x2 corrected flight를 stage-only로 만들고
  `syn101` GPU 1,2와 CPU 8-31에서 병렬 기동했다. run root는
  `/gpfs/kjhan/lumina/det_convergence_single_total_mgpu/det1234_20260815T084331Z_653f94b7f991`,
  supervisor PID `1949640`, binary PID `1950102`, `MGPU_DEVICES=2`,
  `single_total=1`, lock `0/0`, OMP 24다. GPU 1/2의 실측 allocation은
  54,215/38,919 MiB이고 두 장 모두 100% utilization으로 exact multi-GPU owner에
  진입했다. 이후 중복 CPU-only GPU 7 flight는 supervisor `operator_stop`으로 자기
  process group만 종료했으며 GPU 7은 0 MiB로 회수됐다.
- [x] A100x2 iteration 0 exact owner는 45회, residual
  `9.6662782724980344e-09 < 1e-8`, floor/clamp/jitter 0으로 완료했다. caller time은
  `1443.352471113 s`, device allocation은 38,738,428,324 / 40,356,205,424 bytes다.
  R6 generation 1도 `Q_g=1,391,131`, `Q_E=2,180,286`, valid `Q_E=2,180,286`,
  partial/unsampled 0으로 통과했다.
- [x] flight는 population 음수가 아니라 이미 등록된 A2-10 cancellation gate에서
  fail-closed했다. lower witness line 15/shell 10은 signed rate
  `-9.7849550420208522e-58`, uncertainty `6.9194822653875239e-57`; upper witness
  line 1279130/shell 18 (S II)은 signed rate `1.0940582993553446e-40`, uncertainty
  `1.6426383122528282e-40`, cancellation condition `44095.870032182931`이다.
  R7은 rc=4, model rc=1이며 Te/material generation은 보존됐다. 값 repair는 없다.
  이는 Handover 17--18에 남은 `UNRESOLVED_CANCELLATION` 과제의 production 재현이다.

남은 순서:

1. lower/upper 두 cancellation witness의 Jbar/profile error 구성요소를 분해한다.
2. tolerance와 envelope refinement를 값 수리 없이 강화했을 때 sign 인증 가능성을 판정한다.
3. 필요한 경우 중요한 이 판정만 Fable에 질의하고 독립 의견을 기록한다.
4. selftest와 targeted one-iteration A100x2 재현 뒤 4-iteration flight로 승격한다.
5. 통과 뒤 local-Jbar/H200 diagnostic 및 same-physical-state CMFGEN 비교로 진행한다.

## 2026-08-15 — A2-10 cancellation 전수 census와 증명 정밀도 분리

- [x] Fable에 이 중요 쟁점의 세부계획을 단발 위임했다. 두 번의 `$1` 제한 호출은
  응답 없이 종료됐고, 도구를 막은 압축 호출에서만 `VERDICT: REVISE`를 받았다.
  원문과 Codex 보류점은 `docs/FABLE_PLAN_A210_CANCELLATION_2026-08-15.md`에 있다.
- [x] 실제 stderr를 읽는 `check_a210_cancellation_witnesses.py`를 추가했다. rate,
  uncertainty, `4pi*deck_scale`, absorption/net identity를 Decimal 100자리로 재계산했고
  최대 상대오차 `1.2246806772541337e-13 < 1e-12`로 PASS했다. 필요한 대칭 Jbar bound
  축소비는 lower line15/shell10 `7.07155243501095`, upper line1279130/shell18
  `1.501417532521737`이다. repair counter는 0이다.
- [x] `LUMINA_A210_CANCELLATION_CENSUS=1` opt-in 진단을 구현했다. 기본 모드는 기존처럼
  첫 실패에서 즉시 차단한다. census는 independent line-cell을 끝까지 검사하지만
  unresolved/invalid cell을 physical shell sum에 넣지 않고, 하나라도 있으면 publication을
  동일하게 fail closed한다. phase별 ratio histogram, first witness, complete flag와
  `physical_values_modified=0`을 기록한다.
- [x] census summarizer는 per-cell identity를 다시 검산하고 summary count/bin 합계,
  nonfinite 0, repair 0을 요구해 CSV/JSON을 만든다. incomplete scan과 uncertainty 변조
  음성대조를 모두 rc=4로 거부했다.
- [x] k=8 실제 census를 A100 GPU 1/2에서 완료했다. sealed root는
  `/gpfs/kjhan/lumina/a210_cancellation_census_mgpu/det1234_20260815T144043Z_e1f71f5ceacf`,
  binary SHA `e1f71f5ceacf903b58100a345146ebb62ec9a153333fee6eac47eb3a88c15595`,
  1 outer iteration, single-total/lock `1/0/0`, OMP 24다. LOWER/UPPER 각각
  109,014,300셀을 검사했고 unresolved는 132/92, invalid는 0/0이다. CSV/JSON SHA는
  `0264d7ef00d2a01cd40f3bfa580944f8` /
  `8825b337cf326ce4fa6a0b825c79ae29de2c48035f65a6ebfb1fbe7d792e3757`이며 repair는 0이다.
- [x] 다음 refinement-only 실험을 위해
  `LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS`를 strict `1..64` proof-only 설정으로 노출했다.
  기본값은 8이고 invalid 0/65/text는 staging 전에 rc=70이다. 물리 J는 바뀌지 않는다.
  후보 census 비교기는 physical eta/chi/J/net의 bit identity, unresolved subset,
  non-increasing bound를 강제하며 네 음성대조를 통과했다. 다음 binary SHA는
  `8d09f31330da13218673f2b574c6adb605dc4c715676e193d0eae71e87c4a227`이다.
- [x] one-sided 제안의 전제를 감사했다. 현재 대칭 `|J*-J|<=e`만으로 lower witness의
  양의 하한은 얻을 수 없다. 가능한 별도 증명은 positive affine transport의
  source-only lower sweep `F_lower(0)<=J*`이며 설계와 금지 경계를
  `docs/A210_ONE_SIDED_JBAR_PROOF_DESIGN_2026-08-15.md`에 기록했다. census 전에
  production에는 넣지 않는다.
- [x] cancellation condition `1.1474300902143982e6`인 실제 셀에서 offline Decimal
  곱셈·뺄셈과 producer binary64 `fma(-chi,Jbar,eta)`가 다른 연산임을 확인했다.
  판정기를 FMA 계약 그대로 고치고 해당 셀을 회귀 fixture에 추가했다. 물리값과
  허용오차는 바꾸지 않았다.
- [x] k=10 refinement-only run은
  `/gpfs/kjhan/lumina/a210_cancellation_census_mgpu_k10_fixed/det1234_20260815T153850Z_8d09f31330da`에서
  tripwire 충돌 없이 완료됐다. LOWER/UPPER unresolved는 각 30개, invalid/repair는 0이다.
  k=8의 224행 중 164행을 해결했고 surviving 60행의 physical eta/chi/J/net/rate는
  bit-exact다. 모든 bound는 감소했으며 k=10/k=8 비는 `0.1027955847`--`0.4316098784`다.
  비교 report는
  `validation/a2_10/A2_10_REFINEMENT_K8_K10_COMPARISON_2026-08-16.json`이다.
- [x] k=12 refinement-only run은
  `/gpfs/kjhan/lumina/a210_cancellation_census_mgpu_k12/det1234_20260815T162808Z_8d09f31330da`에서
  tripwire 충돌 없이 완료됐다. LOWER/UPPER unresolved는 12/7, invalid/repair는 0이다.
  k=10의 60행 중 41행을 해결했고 surviving 19행의 physical eta/chi/J/net/rate는
  bit-exact다. 모든 bound는 감소했으며 k=12/k=10 비는
  `0.2850493542`--`0.4219825377`이다. 비교 report는
  `validation/a2_10/A2_10_REFINEMENT_K10_K12_COMPARISON_2026-08-16.json`, SHA는
  `39719e08f98e370733e9d3cb2b47d8ccec228b734a5cc08caf3e2c9a9a668d7f`다.
- [x] 수축이 계속되고 plateau 증거가 없으므로 k=14/16을 생략하고 k=18
  refinement-only run을
  `/gpfs/kjhan/lumina/a210_cancellation_census_mgpu_k18/det1234_20260816T002655Z_8d09f31330da`에서
  시작했다. syn101 A100 GPU 1/2, CPU 0-23, supervisor/model PID
  `94862/95222`이며 2초 tripwire와 30초 monitor가 동작한다. exact solve는 45회,
  residual `9.6662782724980344e-09`, refinements 18, raw negative 0으로 완료됐고
  component error 최대는 k=12의 `5.583228050015298e-08`에서
  `3.9690870292479115e-09`로 줄었다. R6 valid cell은 109,014,300개 전부이며 현재
  R7 endpoint census를 완료했다. LOWER/UPPER 각각 109,014,300셀, unresolved/invalid는
  모두 0이고 physical modification/repair는 0이다. zero-closure에서 기존 wrapper가
  production INTERIOR solve로 계속 들어가는 진단 종료조건 결함이 있어 UPPER summary
  직후 operator-stop으로 정확히 두 endpoint scope만 봉인했다. k=12→18 비교는
  19→0, physical identity bit-exact PASS다. CSV/JSON SHA는
  `738d06da379020b4d097b531de7d67fbbdfdb08027ccace7462162a55fe5ede0` /
  `e2ca23e41a1bf12be22f352e08353ad613b428d6fe4c6b2a8cc90c6a71f795fb`다.
- [x] census가 0 unresolved로 닫힌 뒤 실행할 non-census 1-iteration gate 모드를
  `A210_TARGETED_GATE`로 분리했다. model rc 0, fatal/blocked 0, A100 exact owner 2/2,
  R6 전수 valid, R7 material commit, physics snapshot commit, 모든 repair field 0을
  fail-closed로 요구한다. 판정기 양성대조와 5개 음성대조, 단일 snapshot 검증이 PASS했다.
  sealed root는
  `/gpfs/kjhan/lumina/a210_targeted_gate_a100x2_k18/det1234_20260816T005508Z_8d09f31330da`다.
  k=12→18 comparison PASS와 unresolved 0을 확인하고 fresh resource audit 뒤
  10:33 KST syn101 GPU 1/2, CPU 0-23에서 시작했다. supervisor/model PID는
  `280914/281254`이며 2초 tripwire가 동작한다.
- [x] 첫 non-census gate는 exact/R6까지 k=18 reference와 bit-exact였지만 R7에서
  `RADEQ_NO_BRACKET`으로 정상 fail-closed했다. reference comparison은 4개 record,
  physical modification/repair 0으로 PASS했고 현재 report SHA는
  `bf1c3e863ce8c72042f83674484af2f2035bd31340a071b8990fccb00cff5544`다. endpoint는
  42개 shell이 동부호였고(`same_positive=38`, `same_negative=4`), 기하중간 trial은
  38개를 뒤집었지만 shell 0--3은 닫지 못했다. model rc=1/R7 rc=4이며 tripwire나
  외부 자원 충돌은 없었다.
- [x] 이 실패는 수치 bracket 수리가 아니라 coevolution 상태 소유권 결함으로 특정했다.
  같은 line 15/shell 4에서 LOWER/UPPER trial population은
  `1.8145350373112681e-163`/`0.46957666793645558`로 바뀌는데 bulk Sobolev tau는 양쪽
  모두 `1.6167540330683487e-09`였다. private candidate가 public tau slab을 복제한 뒤
  mapped NLTE 선만 덮어써 active-unmapped LTE/nebular 선이 이전 공개 Te에 남은 것이
  원인이다. 따라서 residual이 한 material state의 함수가 아니었다.
- [x] 중요한 coevolution 소유권 판정만 Fable에 요청했다. 판정은 candidate-local
  `trial LTE/nebular bulk tau rebuild -> optional 기존 overlap correction -> NLTE authority
  overlay` 순서가 물리 의존성 복원이며 floor/cap/repair가 아니라는 것이다. pre-core
  rate matrix의 lagged tau는 별도 rung으로 분리하고 이번 수정에는 섞지 않는다.
- [x] `nlte_population_candidate_produce_tau_source()`에 위 순서를 구현했다. 직접
  selftest는 active-unmapped tau의 trial known answer/Te 민감도, 동일 상태 byte identity,
  stale-public 음성대조, inactive exact-zero, mapped overlay, signed negative 보존,
  public mutation 0을 고정했다. 관련 candidate/A2-10/gate/reference 회귀, CPU link,
  CUDA build, `git diff --check`가 PASS했다. 새 `lumina_cuda` SHA는
  `2135362336fd7311143afe47834e8d1c5bc358a483e9c4570dc6fcfa7a49bf11`이다.
- [x] 수정 binary의 A100x2 non-census 재실행을
  `/gpfs/kjhan/lumina/a210_targeted_gate_a100x2_tau_refresh_k18/det1234_20260816T033242Z_2135362336fd`
  에서 시작했다. 12:35 KST 현재 syn101 GPU 1/2, CPU 24-47,
  supervisor/model PID `634565/634927`, 2초 tripwire다. GPU 0의 기존 PID `545207`은
  건드리지 않았다. 정확히 같은 input/deck이며 floor/cap/clamp/jitter/repair는 0이다.
  실수로 기본 submit script의 `--help`가 job `312918`을 제출했으나 allocation 전에 즉시
  취소했고 실행 이력은 없다. unused provenance root는
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260816T033138Z_2135362336fd`다.
- [x] 수정 재실행 exact/R6는 45회, residual `9.6662782724980344e-09`, R6 valid
  `109,014,300/109,014,300`으로 기존 sealed k=18과 4-record bit-exact PASS했다. 최종
  reference report SHA는
  `6bc093375bce28d4a237c893808b2448cc4fa1178e6a33598b5d72133ccdb5cb`이고
  candidate stderr SHA는
  `f8bfaccdf91abb4dbd7acc123bce73ee8443413bd686039d33727e10c4ebd439`다.
- [x] stale tau 수정은 실제 endpoint에서 실증됐다. line 15/shell 4의 tau는 이전처럼
  양쪽 `1.6167540330683487e-09`가 아니라 LOWER `2.3937729582934353e-14`, UPPER
  `12.441552300624615`다. 각 endpoint의 `n_upper`는 기존 LOWER/UPPER trial 값과
  일치하며 public state publication은 없었다. UPPER 50 shell ledger는 모두 finite다.
- [x] corrected LOWER가 새 cancellation witness line 2026622/shell 10을 드러냈다.
  signed rate `2.3587290463683075e-50`, k=18 uncertainty
  `2.442858638972677e-50`, ratio `1.0356673407375478`이라 publication을 정확히
  fail-closed했다. `RADEQ_SIGN_MISMATCH`, R7 rc=4, model rc=1, wrapper rc=70이며
  Te/material generation은 보존됐다. tripwire 개입, external conflict, physical repair는 0이다.
- [x] corrected-material k=18 cancellation census를
  `/gpfs/kjhan/lumina/a210_tau_refresh_cancellation_census_k18/det1234_20260816T041828Z_2135362336fd`
  에서 완료했다. exact/R6 4-record는 sealed k=18 reference와 bit-exact이고, LOWER/UPPER
  각각 `109,014,300`셀을 완전 검사했다. LOWER unresolved/invalid는 `6/0`, UPPER는
  `0/0`이며 LOWER max uncertainty/|rate|는 `11.975441127875897`이다. ratio bin은
  `<=2:4`, `2--10:1`, `10--100:1`, `>100:0`이고 physical modification/repair는 0이다.
  model rc=1의 의도된 fail-closed와 wrapper rc=0, tripwire 충돌 0을 봉인했다. census
  CSV/JSON SHA는 `2df0a7d9853a697a2fa0da778e400c51ec80e7f337532f60be412b98366f5ca3` /
  `796f5c2e5f017608a694108f23eb8270f424119db987d0f6a49c405c174cef32`이다.
- [x] 여섯 witness의 producer binary64 FMA/rate/uncertainty identity를 offline judge로
  재현했다. bound ratio는 `1.03566734`, `11.9754411`, `1.46303651`, `4.69274236`,
  `1.30480041`, `1.18930584`이며 repair는 0이다. report는
  `validation/a2_10/A2_10_TAU_REFRESH_CANCELLATION_WITNESS_AUDIT_K18_2026-08-16.json`,
  SHA는 `1c73f51a821c78496171cf63cc0e51fce6f9b2b9934c04b4f7dd7021ea51416b`이다.
- [x] max ratio `11.9754`에는 기존 +2 refinement 수축으로 k=20이 부족하다. 기존
  k=12--18의 +6 component-envelope 최대 수축 약 `0.071`을 근거로 k=24 closure census를
  `/gpfs/kjhan/lumina/a210_tau_refresh_cancellation_census_k24/det1234_20260816T052635Z_2135362336fd`
  에 stage하고 14:27 KST syn101 GPU 1/2, CPU 24-47에서 supervisor/model PID
  `923312/923632`로 시작했다. binary/deck은 동일하고 refinement만 24이며 모든 repair
  knob는 0이다. exact는 45회, residual `9.6662782724980344e-09`, R6 valid
  `109,014,300/109,014,300`, raw negative/repair 0으로 끝났다. component error 최대는
  k=18 `3.9690870292479115e-09`에서 `2.6699338943402563e-10`으로 `0.06727`배 줄었다.
- [x] k=24 LOWER/UPPER가 각각 `109,014,300`셀을 complete scan했고 둘 다
  unresolved/invalid `0/0`, physical modification/repair 0이다. UPPER complete 직후
  알려진 zero-closure INTERIOR 진입 전에 operator-stop이 작동했고 INTERIOR record는 0이다.
  stderr/CSV/JSON SHA는 `9b117c3ba6bb3140c71b1582253aeedbab4dc639246fccd71047ad983aa4b331` /
  `738d06da379020b4d097b531de7d67fbbdfdb08027ccace7462162a55fe5ede0` /
  `bc48dee81ab1d33302a0c08bae17381fc9fd41693d95a6536f7b1501060ae0b1`이다.
- [x] k=18--24 refinement-only comparison은 baseline 6, candidate 0, resolved 6,
  새 unresolved 0, repair 0으로 PASS했다. report는
  `validation/a2_10/A2_10_TAU_REFRESH_REFINEMENT_K18_K24_COMPARISON_2026-08-16.json`,
  SHA는 `eeeff76a315fd1bd671d2977caddb7e7758409ea4e116929f0b78839fa3ee62c`이다.
- [ ] 동일 k=24 A100x2 non-census gate를
  `/gpfs/kjhan/lumina/a210_targeted_gate_a100x2_tau_refresh_k24/det1234_20260816T062824Z_2135362336fd`
  에 stage했다. census env가 없고 binary/deck, single-total, stage4=0은 동일하며 모든
  repair knob는 0이다. 15:29 KST syn101 GPU 1/2, CPU 24--47에서 supervisor/model PID
  `1081096/1081458`로 시작했고 2초 tripwire가 동작한다.

남은 순서:

1. k=24 non-census exact/R6가 census reference와 동일한 물리 해인지 봉인한다.
2. R7 LOWER/UPPER bracket과 모든 shell의 finite ledger를 확인한다.
3. R7 material commit, physics snapshot commit, model rc=0, repair 0을 봉인한다.
4. gate PASS 뒤 4-iteration flight의 자원/판정 계약을 stage한다.
5. CMFGEN same-identity finite 비교 대상을 구체화하고 실행 순서를 제시한다.

## 2026-08-16 16:54 KST — k=24 gate fail-closed 및 pre-core tau 원인 A/B

- [x] k=24 non-census gate의 exact/R6는 census reference와 bit-identical이다. exact는
  45회, residual `9.6662782724980344e-09`, component error 최대
  `2.6699338943402563e-10`, R6 valid `109,014,300/109,014,300`이다. 최종 reference
  report는 `validation/a2_10/A2_10_TARGETED_TAU_REFRESH_K24_REFERENCE_COMPARISON_2026-08-16.json`,
  SHA는 `79989b63969bd8ec99fe2096599d1684e2ac227c6f6acd9d5f3a93c2169c80ed`다.
- [x] cancellation은 실제 non-census endpoint에서도 닫혔다. LOWER/UPPER 50 shell
  ledger는 finite이고 line 15/shell 4 tau는 각각
  `2.3937729582934353e-14` / `12.441552300624615`다. 그러나 shell 0--3의 R7 residual은
  `3500 K`와 `140000 K`에서 모두 음수다. shell 0은 `-5.9087607939292193` /
  `-149142448.23160025`, 기하중간 `22135.943621178667 K`에서도
  `-5012.5758`이므로 `RADEQ_NO_BRACKET`으로 정상 fail-closed했다. model rc=1,
  R7 rc=4, public generation 보존, repair 0이다.
- [x] line owner 분해 결과 LOWER shell 0--3은 Co III/Fe III NLTE SE 방출이 거의 전부,
  기하중간은 Co IV/Fe IV/Ni IV active-unmapped LTE 방출이 약 99.7%, UPPER는
  Co/Ni/Fe V--VI active-unmapped LTE 방출이 사실상 전부다. 이는 수치 정밀도 문제가
  아니라 후보 물질 상태의 온도 의존 문제다.
- [x] 데이터 흐름에서 별도 lag를 확인했다. candidate opacity view가 공개 tau를 복제하고
  `nlte_solve_all_impl()`의 mode-3 SE matrix가 그 tau로 `beta*J_inc`를 조립한 뒤에야
  post-core trial tau를 만든다. 즉 post-core closure는 해결됐지만 pre-core population은
  공개 상태 tau를 소비한다.
- [x] 원인 판별 전용 opt-in `LUMINA_A210_PRECORE_TAU_REFRESH=1`을 추가했다. trial
  T_e/trial ionization의 LTE Sobolev seed만 SE 직전에 공급하며 mode 3 이외의 source
  consumer는 fail-closed한다. 이것은 최종 population--tau fixed point가 아니며 로그에도
  `population_tau_fixed_point=0`을 명시한다. public mutation/repair는 0이다. 직접
  known-answer/isolation selftest, A2-10, candidate, targeted checker 회귀와 CUDA build,
  `git diff --check`가 PASS했다. binary SHA는
  `4c7ecd6f4e537fc2c25dad5d6b431ac62caac8da78bdc1dcca86e809bd2898ba`다.
- [ ] A/B root는
  `/gpfs/kjhan/lumina/a210_precore_tau_seed_ab_k24/det1234_20260816T075214Z_4c7ecd6f4e53`다.
  Slurm job `312974`는 예상 시작이 8월 18일이라 A100 전용으로 고치고 user-hold했다.
  16:53 KST syn101 allocation 0, GPU 1/2 idle을 확인한 뒤 GPU 1/2, CPU 24--47에서
  supervisor/child PID `1293154/1293185`로 수동 시작했다. 2초 tripwire는 Slurm
  allocation 또는 selected-GPU foreign PID가 생기면 우리 process group만 종료한다.
- [ ] 기존 k=24 targeted gate는 `single_total=1`, ion-lock/per-ion-rescale `0/0`인
  반면 위 seed-on root는 `single_total=0`, ion-lock/per-ion-rescale `1/1`임을 sealed
  env diff에서 발견했다. 따라서 둘의 R7 차이는 tau seed 단독 인과효과로 판정할 수 없다.
  실행 중인 seed-on 결과는 버리지 않고 보조 증거로 보존한다. 정확한 대조군은 같은 새
  binary/deck/refinement와 같은 `single_total=0`, lock/rescale `1/1`에서 seed만 끈
  `/gpfs/kjhan/lumina/a210_precore_tau_seed_ab_baseline_k24/det1234_20260816T081459Z_4c7ecd6f4e53`다.
  normalized env diff는 오직 `LUMINA_A210_PRECORE_TAU_REFRESH=1` 한 줄이며 binary SHA,
  deck manifest SHA, topion manifest SHA가 동일하다. 17:15 KST syn101 GPU 6/7,
  CPU 0--23에서 supervisor/child/model PID `1352221/1352286/1352621`로 시작했다.
  카드 5는 사용하지 않았고, 별도 2초 tripwire가 같은 충돌 조건에서 이 대조군 process
  group만 종료한다.
- [ ] 위 `single_total=0` 쌍은 seed 단독효과를 엄밀히 분리하지만 최종 census/gate의
  `single_total=1` 구성과 다르다. 최종 실패에 직접 귀속하기 위한 새-binary
  `single_total=1` pair도 봉인했다. seed-on은
  `/gpfs/kjhan/lumina/a210_precore_tau_seed_ab_st1_on_k24/det1234_20260816T082314Z_4c7ecd6f4e53`,
  seed-off는
  `/gpfs/kjhan/lumina/a210_precore_tau_seed_ab_st1_off_k24/det1234_20260816T082328Z_4c7ecd6f4e53`다.
  두 root의 55-file deck, 3-file topion, binary, sigma, 모든 sealed control/env를 실제
  SHA로 검증했고 차이는 seed flag 하나뿐이다. seed-on은 17:24 KST syn101 GPU 0/3,
  CPU 48--63에서 supervisor/child/model `1381199/1381244/1381915`로 시작했다.
  seed-off는 stage-only READY이며 GPU가 반환되는 즉시 같은 자원 폭으로 실행한다.
  카드 5는 계속 제외하며 각 실행은 독립 2초 tripwire를 사용한다.
- [x] 독립 CMF exact GPU fair job `252492`는 H200x1에서 `01:17:54`, rc=0으로 끝났다.
  같은 fat binary의 H200x1--A40x4 비교, exit-after-R6, repair 0이 모두 PASS했다.
  H200 CPU--GPU max relative J는 `3.3605860545056784e-15`, finite J 범위는
  `8.40862082551471e-82`--`1.9072381379446642e-4`다. 외부 CMFGEN J_nu 352 finite
  point 비교는 PASS지만 same-state identity가 아니므로 parity 주장은 하지 않는다.
  final verdict SHA는 `170d4d6cc35820e0770f485d1bdcb6b6145da6f45b2f0a5f4c62d5626d9d625d`다.
  별도 H200 4-iteration job `251978`은 syn104에서 계속 실행 중이다.

남은 순서:

1. 설정이 정확히 같은 `single_total=0` 및 최종-gate `single_total=1` A/B의 exact/R6
   identity와 tripwire 무충돌을 확인한다.
2. LOWER/UPPER/기하중간 shell 0--3 residual을 각각의 matched pair 안에서 정량 비교한다.
3. lagged tau가 원인으로 입증되면 population--tau fixed-point 수렴 계약을 먼저 작성한다.
4. fixed-point를 값 보정 없이 구현하고 단위/selftest 뒤 k=24 targeted gate를 재실행한다.
5. gate rc=0 뒤 4-iteration과 CMFGEN same-state finite 비교로 진입한다.

## 2026-08-16 17:45 KST — pre-core tau 가설 기각 및 line-rate identity 감사로 전환

- [x] 위의 pre-core lag 서술을 production 경로에 적용한 것은 잘못이었다. 실제
  `nlte_solve_all_impl()`은 legacy mode-1/2/3 계산 뒤 `Production split`에서 모든 정상
  경로의 `J_line`, `R_absorb`, `R_stim`, `R_spont`를 A2-06 canonical line view와
  `jd_beta=1.0`으로 다시 쓴다. 따라서 현재 gate의 SE population은 pre-core tau seed를
  소비하지 않는다. 앞 절의 고정점 분기는 이 후속 증거로 명시적으로 기각한다.
- [x] seed-on `single_total=0` 실행은 exact/R6 뒤
  `PRECORE-TAU-SEED-BLOCKED reason=UNSUPPORTED_RATE_CONSUMER jbar_mode=UNSET` 및
  `POP_FORBIDDEN_FALLBACK`으로 production 경계에서 정상 차단됐다. root는
  `/gpfs/kjhan/lumina/a210_precore_tau_seed_ab_k24/det1234_20260816T075214Z_4c7ecd6f4e53`다.
  exact 45회, residual `9.6662782724980344e-09`, component error 최대
  `2.6699338943402563e-10`, valid `109,014,300`, repair 0은 유지됐지만 R7 비교값은
  생성되지 않았으므로 물리 A/B로 해석하지 않는다.
- [x] matched baseline과 `single_total=1` seed-on은 exact 도중 operator-stop했고,
  seed-off는 READY stage만 남겼다. 세 실행은 모두 `YIELDED` 또는 미실행이며 GPU는 전부
  반환됐다. held Slurm duplicate `312974`도 실행 없이 취소됐다. 자동 launcher/finalizer와
  감시 세션은 종료했다.
- [x] `docs/A210_POPULATION_TAU_FIXED_POINT_CONTRACT_2026-08-16.md`는
  `REJECTED / DO NOT IMPLEMENT`로 봉인했다. 값 floor/cap/clamp/jitter/repair는 여전히 0이며,
  이 기각 때문에 어떤 물리값도 변경하지 않았다.
- [ ] 현재 검증 대상은 shell 0--3 no-bracket의 실제 지배항인 signed line cooling이다.
  A2-06 canonical SE 전이율
  `n_l B_lu J - n_u(B_ul J + A_ul)`과 R7 private line-energy의
  `eta(beta,tau) - chi Jbar`가 같은 방사장 정의·단위·부호·escape 확률을 쓰는지 소스와
  per-line 항등식으로 감사한다. signed-net 원장이 냉각일 때 `H_line_abs=0`인 것은 raw
  흡수 부재가 아니라 net owner 표현이므로 오류로 간주하지 않는다.
- [ ] H200 4-iteration job `251978`은 syn104에서 독립 실행 중이다. 17:44 KST 기준
  `RUNNING 02:42:39/04:00:00`, MaxRSS `83,908,236 KiB`, 누적 read
  `20,040,003,648 B`로 실제 계산 중이며 중단하지 않는다.

남은 순서:

1. canonical SE와 R7 line-energy의 항별 식·단위·부호·`beta/Jbar` 정의를 대조한다.
2. 동일성이 성립하면 no-bracket 원인을 population/atomic-rate/thermodynamic owner 쪽으로
   좁히고, 불일치가 입증되면 먼저 known-answer 회귀와 수정 계약을 작성한다.
3. 물리식 수정이 필요한 경우에만 값 보정 없이 구현하고 CPU/selftest/CUDA build를 통과한다.
4. A100x2 k=24 non-census gate에서 exact/R6 identity, 50-shell bracket, commit, repair 0을
   다시 봉인한다.
5. H200 `251978` 결과를 수거하고 gate rc=0 뒤에만 4-iteration 및 CMFGEN same-state finite
   비교로 진입한다.

## 2026-08-16 18:19 KST — Sobolev--Einstein 공통모드 불일치와 정밀 진단 재시작

- [x] canonical SE의 raw bound-bound energy flow와 R7 direct bracket은 동일한 `Jbar`를
  사용할 때 대수적으로 같은 식이어야 한다. 실제 R7 흡수계수는
  `SOBOLEV_COEFF*f_lu*(wavelength_cm*nu/c)`이며 SE 쪽은
  `h*nu*B_lu/(4*pi)`다. beta 누락이나 signed-owner 표기 자체는 원인이 아니다.
- [x] 런타임 `SOBOLEV_COEFF=0.026540281`은 I20에서 이미 채택한 정확 상수의 값
  `0.026540088545744744`보다 `7.251454904544374e-6` 크다. 전체 2,588,798선의
  `B_lu` implied coefficient는 exact 상수 주위 `-1.4083e-6`--`+1.4145e-6`, 평균
  `-6.0910e-11`; 덱 `wavelength_cm*nu/c-1`은 `-4.9851e-7`--`+4.9929e-7`다.
  따라서 실제 Einstein/runtime-transport 비는 `-9.1450e-6`--`-5.3648e-6`, 평균
  `-7.251441834474044e-6`다. invalid line은 0이다.
- [x] 읽기 전용 전수검증기 `scripts/check_sobolev_einstein_identity.py`와 known-answer
  `tests/sobolev_einstein_identity_selftest.py`를 추가했다. C selftest도 직렬화된
  wavelength를 포함한 변환식을 검사하며 모두 PASS했다. 계약은
  `docs/A210_SOBOLEV_EINSTEIN_IDENTITY_CONTRACT_2026-08-16.md`다. 봉인된 정적 report는
  `validation/a2_10/A2_10_SOBOLEV_EINSTEIN_STATIC_AUDIT_2026-08-16.json`, SHA
  `e2ba5cffe897c01f4a677111b69aceb0beabd5898a9a475fe7883bc8938395f6`다.
- [x] 덱 생성기의 `write_line_list_csv`와 정적 macro-atom internal-up 가중치가 I20/SE의
  `A*c^2/(2*h*nu^3)`가 아니라 `A*c^2/(8*pi*h*nu^3)`를 쓰는 별도 source defect를 찾았다.
  finalizer는 line-list B만 고친다. 현재 gate는 `LUMINA_DYNAMIC_TRANSPROB=1`이므로 이번
  R7 잔차의 직접 원인은 아니지만 새 source-of-truth 덱에서는 함께 수리해야 한다.
- [x] 최초 진단은 stage 절대경로 오류로 model 진입 전 실패했다. 다음 두 실행은 정확한
  원인식 확정 전에 operator-stop했다. 특히
  `/gpfs/kjhan/lumina/a210_line_coefficient_identity_k24/det1234_20260816T090105Z_8183a9bd5f6b`
  는 diagnostic ratio에서 `wavelength_cm*nu/c`를 1로 가정했으므로 결과 판정에서 제외했고
  18:12 KST 정상 중단하여 GPU 5/6을 반환했다.
- [x] 수정 진단 binary SHA는
  `1b4ffc8a6ac74c0f0677a0e6ba26b21672ab23d21a9abeb947da714ee425e679`다. 새 sealed root
  `/gpfs/kjhan/lumina/a210_line_coefficient_identity_k24/det1234_20260816T091800Z_1b4ffc8a6ac7`
  를 18:18 KST syn101 GPU 5/6, CPU 24--47에서 supervisor/child
  `1533251/1533316`으로 시작했다. 2초 tripwire, refinements 24, single-total 1,
  physical mutation/repair 0이다. 하지만 `SRCE_CHK`를 보존하면서 전역 정확 상수와
  선별 직렬화 오차의 동적 기여를 분리하려면 세 번째 합이 필요함을 실행 3분에 발견해
  18:22 KST operator-stop했다. 이 root도 물리 판정에 쓰지 않으며 GPU는 반환됐다.
- [ ] 최종 진단은 같은 셀 순회에서 `current`, `exact-global-K`, `Einstein-per-line` 세 합과
  각각의 delta를 동시에 누적한다. C/Python known-answer와 CUDA build가 PASS했고 binary
  SHA는 `c2b667007c6a1ae1453530c1dfa33e7877e877dea703bfb3f2f511598cb7dd6c`다. 새 sealed root
  `/gpfs/kjhan/lumina/a210_line_coefficient_identity_k24/det1234_20260816T092600Z_c2b667007c6a`
  를 18:25 KST syn101 GPU 5/6, CPU 24--47에서 supervisor/child
  `1551018/1551082`로 시작했다. 2초 tripwire와 모든 repair=0을 유지한다.
- [x] H200 job `251978`은 17:46 KST에 `FAILED`, elapsed `02:45:25`로 끝났다. 이는
  구 binary/config(`6ed103...`, refinements 8, ion-lock/rescale 1/1)의 별도 4-iteration
  시도이며 LOWER/UPPER population 후보가 각각 작은 음수를 내어 fail-closed했다.
  현재 gate의 k=24 single-total 구성이나 새 line-identity 진단 결과로 해석하지 않는다.

남은 순서:

1. 수정 A100 진단의 exact/R6가 sealed k=24 reference와 동일함을 확인한다.
2. LOWER/MID/UPPER shell별 기존 signed rate와 Einstein-consistent finite rate를 봉인한다.
3. 전역 상수 오차와 덱 직렬화 오차가 no-bracket에 기여한 양을 분리한다.
4. 인과가 입증될 때만 계약의 source-of-truth 수리를 구현하고 새 덱/회귀를 만든다.
5. A100x2 k=24 non-census gate, H200 교차검증, CMFGEN same-state finite 비교를 수행한다.

## 2026-08-16 19:40 KST — 계수 인과 분기 종료, PUBLIC_SEED 진단 실행

- [x] 최종 계수-동일성 run
  `/gpfs/kjhan/lumina/a210_line_coefficient_identity_k24/det1234_20260816T092600Z_c2b667007c6a`
  가 자연 종료했다. model rc=1/R7 rc=4와 wrapper rc=70은 네 shell의 물리
  `RADEQ_NO_BRACKET`을 그대로 fail-closed한 결과다. stderr SHA는
  `98ccfee05d2776439200972e513cca611dad033c039fee49c15fb5417e3568d2`다.
- [x] exact/R6 네 record는 sealed k=24 gate와 bit-exact다. exact 45회, residual
  `9.6662782724980344e-09`, refinements 24, component max
  `2.6699338943402563e-10`, valid cells `109,014,300`, partial/unsampled 0이다.
- [x] current/exact-global-K/Einstein-per-line 모두 endpoint bracket이 `46/50`이다.
  recovered/lost shell은 모두 0이다. shell 0의 LOWER residual은 각각
  `-5.9087607939292193/-5.909094644249742/-5.909095131091645`, UPPER는
  `-149142448.23160025/-149142448.23414654/-149142448.2341374`다. 따라서
  7.25 ppm Sobolev 상수 및 CSV 직렬화 불일치는 실재하지만 현재 no-bracket의 원인은
  아니다. production 상수와 덱은 바꾸지 않았다.
- [x] 기하중간 `22135.943621178667 K`에서도 shell 0--3 residual은
  `-5012.5757976049363`, `-3900.4473678152158`, `-3032.8033250590665`,
  `-2359.4368970253981`이고 interior bracket은 0/4다. dynamic causal report는
  `validation/a2_10/A2_10_LINE_COEFFICIENT_DYNAMIC_CAUSAL_AUDIT_2026-08-16.json`,
  SHA `007ae633536d890b551858986f2506d9d321550efb1e6035a2076fb5f9ed47cf`다.
- [x] 실제 O-PHYS CMFGEN `LINEHEAT` 789,775선의 signed 총합을 깊이 67/68에서
  각각 `0.0015538702032496805`와 `0.0017695388318788178 erg cm^-3 s^-1`로
  재현했다. 취소 condition은 `41.8098265/42.7720363`이고 repair 0이다. 이는 finite
  scale 증거이며 아직 matched-state parity가 아니다. report는
  `validation/a2_10/A2_10_CMFGEN_LINEHEAT_FINITE_DEPTHS_2026-08-16.json`, SHA
  `03244b15b349ee18dc808860654c0d1716d5d3a73a8c9e949de66d38e0998d45`다. 온도는
  CMFGEN `RVTJ`에서 직접 읽은 `18842.362/19345.529 K`다.
- [x] 공개 seed private evaluation을 기존 `old_te_attempts`와 분리해
  `diagnostic_seed_trials`로 계측했다. solver result, bracket, publication candidate를
  수정하지 않는다. A2-10 N1--N8와 parser selftest가 PASS했고 새 binary SHA는
  `8d7a389e5ee82bad25c15f155c63e17780dfc80784f840bf35050e104d415c3f`다.
- [ ] PUBLIC_SEED run은
  `/gpfs/kjhan/lumina/a210_public_seed_identity_k24/det1234_20260816T104009Z_8d7a389e5ee8`
  이다. 19:40 KST syn101 GPU 5/6, CPU 24--47에서 supervisor/child/model
  `1742409/1742473/1742875`로 시작했다. 2초 tripwire, single-total 1,
  refinements 24, repair knobs 0이다.
- [ ] 19:40 KST의 직접 `squeue --me`에는 kjhan H200 실행/대기 작업이 없다. H200
  `251978`은 17:46 FAILED, `252492`는 16:19 COMPLETED다. 별도 세션에서 새 ID가
  나타나면 그 ID를 기준으로 다시 병행 모니터한다.

남은 순서:

1. PUBLIC_SEED run의 exact/R6 bit identity와 공개 seed 10,020 K 내부 부호를 봉인한다.
2. 숨은 bracket이면 publication을 건드리지 않는 제한된 deterministic scan을 설계한다.
3. CMFGEN depth 또는 Lumina trial을 matched-temperature/state로 맞춰 signed line total을 비교한다.
4. 원인이 입증된 분기만 production 구현하고 모든 repair counter 0을 확인한다.
5. census와 A100x2 non-census gate를 재실행한 뒤 H200 교차검증으로 간다.

## 2026-08-16 20:10 KST — CMFGEN 동일 전이 finite 기준선과 병렬 line-cell 진단

- [x] A2-09 forensic `total_line_emit`과 A2-10 `scaled_emission`의 큰 차이는 동일
  observable이 아님을 소스 식으로 확정했다. A2-09는
  `n_u A_ul h nu beta_esc(tau)`인 escape/transport emission이고 A2-10은 CMFGEN
  direct net bracket `4*pi*SCL*(eta-chi*Jbar)`의 intrinsic `eta=n_u A_ul h nu/(4*pi)`다.
  따라서 이 두 총합의 비를 no-bracket 결함 근거로 쓰지 않는다.
- [x] 활성 덱 line `233521`은 CMFGEN O-PHYS line `76887`,
  `CoIV(3d5(6S)4p_5Po[3]-3d6_5De[4])`, `606.784334 A`와 frequency 및 full-level
  index가 정확히 대응한다. CMFGEN depth 67/68에서 이 한 전이의 signed rate는
  `4.174045663265543e-05/7.06142063932685e-05 erg cm^-3 s^-1`,
  `Jbar/S=0.999998687164/0.999998559733`, cell cancellation condition은
  `1.5234185284102506e6/1.3886304134809726e6`이다. 이는 거의 완전한 finite
  emission--absorption cancellation이며 0 출력 비교가 아니다.
- [x] 기준 artifact는
  `validation/a2_10/A2_10_CMFGEN_MAPPED_LINE_76887_2026-08-16.json`, SHA
  `0ee73a3d779263664199d77bba1466b71e4a249dbc864bef8ed85a2e99ebb73d`다.
  LINEHEAT/NETRATE/RVTJ 및 활성 line-list SHA를 모두 기록했고, state parity가 아님을
  명시했다. physical mutation 및 floor/cap/clamp/jitter/repair는 0이다.
- [x] 같은 velocity의 CMFGEN depth 67/68 선형 좌표에서 mass density는
  `1.570773538958886e-13 g cm^-3`, Lumina shell 0 입력은
  `1.5687692791189745e-13 g cm^-3`로 비가 `1.0012776001331676`이다. 질량밀도는 거의
  맞지만 CMFGEN `n_e=5.087765219e9 cm^-3`와 Lumina 입력 `1.612459940e9 cm^-3`, 온도와
  방사장은 아직 같지 않으므로 parity를 선언하지 않는다.
- [x] `LUMINA_RADEQ_DIAG_LINE_CELL=LINE:SHELL` 요청형 read-only 진단을 추가했다.
  잘못된 line/shell은 schema failure로 거부하며 물리값을 바꾸지 않는다. strict env
  universe에도 등록했다. CUDA build SHA는
  `6a44c34c558f0157170958aa9ed5b858d76b0856fddf8907b36ca81f35dcc0f3`; A2-10,
  line-net selftest가 PASS했다.
- [x] 비교기 `scripts/compare_a210_cmfgen_mapped_line.py`와 positive 1/negative 4
  selftest를 추가했다. signed rate, `Jbar/S`, cancellation condition을 그대로 비교하며
  tolerance에 의한 sign 승인이나 repair를 하지 않는다.
- [ ] mapped-line root
  `/gpfs/kjhan/lumina/a210_cmfgen_mapped_line_k24/det1234_20260816T110100Z_6a44c34c558f`
  를 20:04 KST syn101 GPU 4/7, CPU 0--23에서 supervisor/child/model
  `1803710/1803787/1804117`로 시작했다. 기존 PUBLIC_SEED GPU 5/6과 자원 집합이
  겹치지 않으며 두 작업 모두 2초 tripwire와 repair 0을 유지한다.

남은 순서:

1. PUBLIC_SEED exact/R6와 10,020 K shell 0--3 residual을 수거한다.
2. mapped line의 LOWER/UPPER/PUBLIC_SEED/GEOMETRIC_MID `eta`, `chi*Jbar`, `Jbar/S`를 수거한다.
3. CMFGEN depth 67/68의 같은 전이와 finite 규모·상쇄구조를 비교해 material/Jbar 중 원인을 가른다.
4. 입증된 원인 경로만 값 보정 없이 수정하고 selftest/CUDA build를 통과한다.
5. k=24 census와 A100x2 non-census gate를 재실행하고 필요할 때만 H200 교차검증한다.

## 2026-08-17 00:02 KST — R2 non-overlap Sobolev 구현·감사 완료, A100×2 gate 시작

- [x] seed-material R2 실패를 물질 결함이 아니라 연산자 불일치로 확정했다. 같은 Fe III
  1100→1296 전이는 CMFGEN finite 출력에서도 41/90 depth에서 population-inverted다.
  정본 forensic은
  `validation/a2_10/A2_10_A100X2_SEED_MATERIAL_R2_NEGATIVE_OPACITY_FORENSIC_2026-08-16.json`,
  SHA `3f9e92e9ac72a320b061591429b602d9d66b43ed46833205d05907a4bcfca648`다.
- [x] parity lane의 R2 이후 line 연산자를 CMFGEN non-overlap Sobolev로 교체했다.
  exact fine solver에는 continuum만 들어가고, registered Gaussian profile은 continuum
  `J_cont` 표본 정의에만 남는다. 각 line/shell은
  `Jbar=beta*J_cont+eta*(c*t/nu)*(1-beta)/tau_eff`를 소비한다. R1은 Fable이 동시에
  요구한 sealed k24 seed-predictor input의 bit identity를 위해 명시적인 init-only shared
  operator로 유지한다. 선택은 pass 기반이며 line sign 기반이 아니다.
- [x] CMFGEN EXPONX 세 분기와 zero-tau 소거 안전 companion을 구현했다. 실제 Fe III
  line `2164811`, shell 0의 raw tau/population/A/nu/time을 고정한 known-answer,
  mild-negative/zero/positive direct-bracket identity, NaN/negative-J defect injection이
  모두 PASS했다. `nextafter`는 continuum input error의 proof bound에만 적용되며 물리값을
  바꾸지 않는다.
- [x] R2 gate judge는 raw-negative `4,246,581`, mild-negative `4,246,577`, typed
  policy `4`, 총 `109,014,300` finite Jbar cell을 pre-register했다. material census와
  runtime application이 다르면 fail-closed한다. raw tau/population은 보존된다.
- [x] CPU/OpenMP, CUDA sm80/sm86/sm90, A2-06 dual commit, A2-08/09/10,
  cancellation/refinement/targeted checker 회귀가 PASS했다. 최종 staged binary SHA는
  `1b8d19fa61014ec2735f11ed23321dbfc604ffa55deaaf41c0d34a4d854be46c`다.
- [x] 중요 코드만 Fable에 감사 요청했고 `VERDICT: APPROVE`, A100 전 필수 변경 0을
  받았다. 감사 기록은
  `docs/FABLE_AUDIT_A210_NONOVERLAP_SOBOLEV_CODE_2026-08-16.md`다.
- [ ] A100×2 non-census gate root는
  `/gpfs/kjhan/lumina/a210_targeted_gate_a100x2_nonoverlap_sobolev_k24/det1234_20260816T150135Z_1b8d19fa6101`다.
  00:02 KST syn101 GPU 5/6, CPU 24--47에서 supervisor/child
  `2439806/2439837`로 시작했다. 시작 직전 두 GPU는 memory/utilization `0/0`; GPU 0의
  타 process와 분리했다. 2초 tripwire는 선택 GPU 외부 PID 또는 Slurm allocation이
  생기면 우리 process group만 종료한다.

남은 순서:

1. R1 exact/R6 네 record가 sealed k24 reference와 bit-exact인지 확인한다.
2. R2 census가 pre-registered `4,246,581/4,246,577/4`를 재현하고 이전 negative-total
   fail point를 지나 finite Jbar `109,014,300`개를 발행하는지 확인한다.
3. R7 50-shell bracket, material commit, physics snapshot, repair 0을 판정한다.
4. 성공 로그·checker·tripwire·binary/deck hash를 정본 artifact로 봉인한다.
5. 실패하면 실패한 최초 물리/계약 witness만 추적하고 값 보정 없이 다음 분기를 정한다.

## 2026-08-17 00:17 KST — 장시간 gate 자동 후속판정과 최종 감사 준비

- [x] R1/R2 장시간 실행이 로그인과 분리되어도 판정이 이어지도록 post-gate monitor를
  붙였다. PID는 `1327361`, 로그는 실행 root의
  `manual_control/post_gate_monitor.log`다. gate PASS 뒤에만 R1 첫 occurrence를 sealed
  k24 projection과 비교한다. 고정한 비교기 SHA는
  `79a49f2ea1db29aa58b4873b632b4d16b66e3abce4a1b1a22eca796aa2506d22`이며, 실행 전
  SHA가 달라지면 fail-closed한다. gate 실패/yield이면 비교하지 않는다.
- [x] `scripts/compare_a210_targeted_reference.py`가 multi-pass 전체 stderr에서 명시한
  occurrence를 선택하도록 보강했다. positive, occurrence selection, negative 5개와
  `py_compile`, `git diff --check`가 PASS했다. 최종 stderr가 닫힌 뒤 occurrence 0의 R1
  네 record와 sealed reference를 비교하므로 report의 candidate SHA도 최종 로그를
  가리킨다.
- [x] `scripts/finalize_a210_nonoverlap_gate.py`와
  `tests/a2_10_nonoverlap_completion_selftest.py`를 추가했다. K12/K18 원 CSV hash와
  unresolved `19->0`, gate/snapshot/R1 reports, binary/deck/top-ion hash, two-A100
  tripwire natural completion, coevolution generation `[1,2]`, rejected pre-core refresh,
  모든 repair 환경/로그/구조화 필드를 한 번에 fail-closed 감사한다. positive 1개와
  repair-env/K18-row/yield/log-mutation/operator 음성 대조군 5개가 PASS했다.
- [x] K12/K18 원본 CSV SHA를 다시 계산해 report와 일치함을 확인했다. K12 CSV는
  header+19행, K18 CSV는 header만 있으며 미해결은 정확히 0이다.
- [ ] A100x2 run은 00:17 KST 약 15분째 R1 exact solve 중이다. GPU 5/6은 계산 부하를
  유지하고, Slurm allocation/외부 선택-GPU PID/tripwire yield는 없다.

남은 순서:

1. R1/R2와 wrapper가 자연 종료할 때까지 tripwire를 유지한다.
2. 자동 R1 comparison이 final stderr SHA에서 bit-exact PASS했는지 확인한다.
3. targeted gate/snapshot reports와 최초 실패·비정상 표식 부재를 직접 감사한다.
4. 새 completion auditor로 실제 run과 K12/K18 원자료를 하나의 정본 JSON에 봉인한다.
5. 문서와 regression ledger를 갱신하고 요구사항별 증거 감사 후에만 이 목표를 닫는다.

## 2026-08-17 01:51 KST — K24 국소 proof-bound 실패와 K30 최소 rung

- [x] K24 A100x2 run은 R1과 R2 exact/Sobolev publication까지 정상 통과했다. R2는
  52회에서 residual `8.1222406993212508e-09 < 1e-8`, refinements 24,
  `109,014,300` Jbar cell 전부 finite였다. raw-negative/mild-negative/SRCE_CHK는
  `4,246,581/4,246,577/4`이고 물리값을 수정하지 않았다.
- [x] R7 endpoint에서 두 국소 부호만 proof bound 안에 남아 publication이 의도대로
  fail-closed했다. LOWER `line=1154618 shell=5`는 bound/required
  `1.2461589272572855`, UPPER `line=894169 shell=27`은
  `1.7838554807658646`이다. model rc=1, R7 rc=4, tripwire wrapper rc=70이며 T_e
  generation/manifest는 보존됐다. floor/cap/clamp/jitter/repair는 0이다.
- [x] 두 witness를 binary64 FMA와 Decimal proof arithmetic으로 독립 재구성했고 identity
  오차는 모두 0이다. 임의 tolerance가 아니라 현재 certified Jbar bound가 필요한 값보다
  각각 24.6%, 78.4% 넓다는 뜻이다.
- [x] 중요한 구조 판단만 Fable에 요청했다. 판정은 `YES — K30 is the justified minimum
  next rung`이며 정본은 `docs/FABLE_VERDICT_A210_K30_PROOF_RUNG_2026-08-17.md`다.
  K30 실패 시 K36으로 기계적으로 가지 말고 국소 bound 성분을 분해한다.
- [x] completion auditor를 `--expected-refinements`로 일반화했다. K24/K30 positive와
  기존 음성 대조군 5개가 PASS했다. detached post-gate reference monitor도 재사용 가능한
  `scripts/monitor_a210_targeted_reference.sh`로 봉인했다.
- [ ] K30 root는
  `/gpfs/kjhan/lumina/a210_targeted_gate_a100x2_nonoverlap_sobolev_k30/det1234_20260816T164749Z_1b8d19fa6101`다.
  01:48 KST syn101 GPU 5/6, CPU 24--47에서 supervisor/child/model
  `2721372/2721409/2721697`로 시작했다. binary/deck/물리 설정은 K24와 같고 proof
  refinements만 30이다. 2초 tripwire와 post-gate monitor PID `1655887`이 동작 중이다.

남은 순서:

1. K30 R1 physical/exact/R6가 sealed k24 reference와 bit-exact인지 확인한다.
2. K30 R2 exact/Sobolev census와 109,014,300 finite Jbar cell을 확인한다.
3. LOWER/UPPER 두 witness의 국소 bound가 실제로 required 아래인지 확인한다.
4. R7 commit, physics snapshot, tripwire natural completion과 repair 0을 확인한다.
5. 최종 completion auditor를 `--expected-refinements 30`으로 실행해 목표를 봉인한다.

### 2026-08-17 01:59 KST — K30 국소 witness 보존판 재시작

- [x] 첫 K30 root `.../det1234_20260816T164749Z_1b8d19fa6101`은 실행 5분 전
  operator-stop했다. 수치/물리 실패가 아니며 최종 판정에서 제외한다. GPU는 정상 반환됐고
  post monitor도 `NO_GATE_PASS`로 닫혔다.
- [x] Fable의 국소 증거 요건을 충족하도록 K24 최초 실패 두 셀을 endpoint finite witness
  목록에 추가했다. 이는 `fprintf` 계측만 추가하며 물리 분기와 값에는 관여하지 않는다.
- [x] K24 witness 정본은
  `validation/a2_10/A2_10_NONOVERLAP_K24_PROOF_WITNESSES_2026-08-17.json`, SHA
  `06f81208f5c2f896390af4df273b0754dc0590cd3eea0e089a3446091a64735c`다.
  completion auditor는 K30의 두 셀에서 K24 `eta/chi/Jbar/signed_rate` bit identity,
  FMA/uncertainty identity, local bound < required를 모두 요구한다.
- [x] CUDA binary SHA `d5f769212cf6ac4b40aa86515fb9a550626336fd591ada8748ea113f29ee2075`.
  line-net, A2-06, A2-08/09/10, targeted gate/reference, completion positive K24/K30 및
  음성 대조군 6개가 PASS했다.
- [ ] 최종 K30 root는
  `/gpfs/kjhan/lumina/a210_targeted_gate_a100x2_nonoverlap_sobolev_k30/det1234_20260816T165757Z_d5f769212cf6`다.
  01:58 KST syn101 GPU 5/6, CPU 24--47, supervisor/child/model
  `2745715/2745746/2746040`으로 시작했다. proof-mode post monitor PID는 `1690109`다.
- [x] K24와 K30은 refinement 수와 certified error envelope가 의도적으로 다르므로 R1
  비교기를 proof-only mode로 확장했다. signed-material census, exact convergence/domain,
  R6 line identity/coverage의 물리·solver 필드는 bit-exact여야 하고, 허용 차이는
  `refinements 24->30`과 `component_error/profile_error`의 수축뿐이다. 하한은 증가하지
  않고 모든 상한은 엄격히 감소해야 한다. 비교기 SHA는
  `047c2f0b49c6b89b694dd65832c6fb3f8a8fe047fd1b80347bbd0b55d82a16ba`다.
- [x] strict/proof positive, occurrence selection, 독립 음성 대조군 8개의 reference
  selftest와 K24/K30 completion positive 및 음성 대조군 7개가 PASS했다. 기존 monitor는
  계산 process group과 독립적으로 종료했고, 새 monitor만 `24 30` proof 인자를 받아
  동작한다. monitor의 TERM trap도 active marker를 지운 뒤 확실히 종료하도록 고쳤다.
- [x] completion auditor는 proof-mode에서 허용되는 다섯 change key의 정확한 집합,
  `24->30`, 각 유한 envelope의 비증가 하한·엄격 감소 상한·상한비를 독립 재검산한다.
  auditor SHA는 `0c30c3e06672b635efed4751c97f330f50b0b9b96ff9f30731157ab8385722d1`다.
  R1 monitor PASS 뒤에만 이 auditor를 실행하는 completion monitor PID `1703863`도
  시작했다. K12/K18 report SHA `b6a0be0b...`, K24 witness SHA `06f81208...`, auditor
  SHA를 모두 launch 시점에 고정하며 최종 report도 SHA와 함께 기록한다.
- [x] 02:44 KST K30 R1 exact/R6가 발행됐다. iterations/residual/domain hash와
  signed-material/R6 coverage를 포함한 모든 비-proof field는 sealed K24 R1과
  bit-exact다. component upper는 `2.6699338943402563e-10 ->
  1.7997420448285821e-11`(ratio `0.06740773802091832`), profile upper는
  `2.6695397784909486e-10 -> 1.7994672967902888e-11`(ratio
  `0.06740739775780757`)로 수축했다. 하한은 bit-exact이고 repair는 0이다.
- [ ] 모델은 R1 publication 뒤 R2 coevolution 단계로 계속 실행 중이다. transient
  failure scan의 2건은 실제 수치 cap이 아니라 solver iteration metadata `cap=64`와
  `exact_cap=64`를 잡은 오탐임을 원문으로 확인했다.
- [x] R2 seed-material commit은 `r1_generation=1`, `te_generation=1->1`,
  `population_generation=1->2`, manifest/publication preserved를 발행했다. non-overlap
  signed material은 `line_shells/exact_zero/raw_negative/mild_negative/srce_chk =
  22,866,166/86,148,134/4,246,581/4,246,577/4`를 정확히 재현했고 raw tau를 그대로
  보존했다. floor/cap/clamp/jitter/repair는 0이며 R2 exact solve가 A100 두 장에서
  시작됐다.

남은 순서:

1. K30 R1 physical/exact/R6의 비-proof 필드가 sealed k24 reference와 bit-exact이고,
   허용된 proof envelope만 수축하는지 확인한다.
2. K30 R2 exact/Sobolev census와 109,014,300 finite Jbar cell을 확인한다.
3. LOWER/UPPER 두 witness의 local bound/required가 모두 1 미만인지 확인한다.
4. R7 commit, physics snapshot, tripwire natural completion과 repair 0을 확인한다.
5. proof witness baseline을 포함한 completion auditor로 최종 artifact를 봉인한다.

## 2026-08-17 04:35 KST — K30 종료: endpoint proof 폐합과 물리적 no-bracket 분리

- [x] 최종 K30 root
  `/gpfs/kjhan/lumina/a210_targeted_gate_a100x2_nonoverlap_sobolev_k30/det1234_20260816T165757Z_d5f769212cf6`
  는 04:09:32 KST에 자연 종료했다. `syn101` GPU 5/6과 CPU 24--47을 사용했고,
  시작 전 두 GPU는 memory/utilization `0/0`이었다. 실행 중 선택 GPU의 외부 PID,
  Slurm allocation, tripwire `YIELD/COLLISION`은 없었다. 모델은 물리적 no-bracket을
  fail-closed해 `model.rc=1`, wrapper child rc=70, `manual_control/FAILED`로 닫혔다.
  stdout/stderr SHA는 각각
  `850250192270d8de4f0998fedbc07ba7f720b85b8d1563a142080a8fc9b93ebe`와
  `7033453d7e6be2363ef17d99149cd22cf5936438905e8517e86d1ad8d7e29b3c`다.
- [x] K30 R1/R2는 K24와 모든 비-proof 물리·solver field가 bit-exact다. R1/R2
  reference 정본은 각각
  `validation/a2_10/A2_10_NONOVERLAP_K30_R1_PROOF_REFERENCE_2026-08-17.json`
  (SHA `bb0df7dd5fd7b5a0fcf35d4e9f393cacd042d9dde29eba66afd3e7f197627f2e`)와
  `validation/a2_10/A2_10_NONOVERLAP_K30_R2_PROOF_REFERENCE_2026-08-17.json`
  (SHA `8e41a2410c4f53965b7c969a477ea9d3fdc20c22ca567c7f9b124e895ccb3638`)다.
  R2 exact는 `52/8.1222406993212508e-09/30`, Sobolev Jbar는
  `109,014,300`개 전부 finite이고 signed/raw-negative/mild-negative/SRCE_CHK는
  `22,866,166/4,246,581/4,246,577/4`다. raw tau와 coevolution generation barrier는
  보존됐고 모든 물리 repair는 0이다.
- [x] K24의 두 endpoint witness는 K30에서 물리값 bit identity를 유지하면서
  bound/required가 LOWER `0.060311609497464516`, UPPER
  `0.05771346719097574`로 모두 1 미만이 됐다. 정본
  `validation/a2_10/A2_10_NONOVERLAP_K30_PROOF_WITNESSES_2026-08-17.json`의 SHA는
  `f6344bee66b4e3e6a4e850ff0c09d71ee75e4cdad1fcae6b3d33574e079c18da`다.
- [x] endpoint line identity 150개(LOWER/UPPER/PUBLIC_SEED 각 50 shell)가 완전하며,
  current/exact-constant/Einstein counterfactual bracket count는 모두 `46/50`이다.
  PUBLIC_SEED도 같은 네 shell만 no-bracket이고 나머지 interior bracket 손실/회복은 0이다.
  정본 `validation/a2_10/A2_10_NONOVERLAP_K30_LINE_IDENTITY_2026-08-17.json`의 SHA는
  `fc62524fdd786b882970ae6d4990673907f0d278ece1060daa30c11b07db1307`다.
- [x] 실제 물리 분기는 shell 0--3의 no-bracket이다. PUBLIC_SEED `10020 K`에서 residual은
  각각 `-3.4648553111533551/-2.7104785016110249/-2.1168582690802782/
  -1.653197660559971`이고 line cooling이 지배한다. 이는 endpoint proof 오류나 0값 비교가
  아니며 R7은 Te generation/manifest와 material publication을 보존하고 종료했다.
- [x] 선택적인 `GEOMETRIC_MID` callback은 물리 residual을 내기 전에
  `line=894169 shell=11`의 proof-only `UNRESOLVED_CANCELLATION`에서 중단됐다.
  uncertainty/abs(rate)는 `6.764121495492545`다. 따라서 이를 네 shell의 물리적
  same-sign 증거로 사용하지 않는다. 코드상 이 uncertainty는 정확히
  `beta * continuum_j_absolute_uncertainty`이며 local-emission proof 성분은 0이므로,
  non-contracting bound 성분이 발견된 것은 아니다.
- [x] 위 분리를 fail-closed 감사한 정본은
  `validation/a2_10/A2_10_NONOVERLAP_K30_NO_BRACKET_BRANCH_2026-08-17.json`
  (SHA `c8bbb907076bf1f71e06ba557d9bf99fbd7943fd2ce673412c947b370aefe6ad`)다.
- [x] generic repair scan이 exact solver metadata `cap=64`를 물리 cap으로 오인하던 문제를
  고쳤다. 오직 `[cmf_fine][EXACT-MULTIGPU-EPOCH] cap`만 solver iteration limit로
  제외하며, 다른 모든 `cap` 관측은 계속 fail-closed한다. targeted/completion positive와
  독립 음성 대조군 및 line-identity selftest가 모두 PASS했다.

다음 다섯 단계:

1. K30 종료·분기 artifact와 원 로그 해시를 인수인계서에도 봉인한다.
2. CMFGEN depth 67/68과 Lumina shell 0의 T, n_e, density, line-owner 정의를 명시 대조한다.
3. 물리합을 바꾸지 않는 A2-10 이온별 signed-rate owner 진단을 완전 callback 뒤에만 낸다.
4. CMFGEN 보간 온도 `19059.411196903675 K`를 private diagnostic callback으로만 평가한다.
5. proof가 부족하면 contracting bound 근거로 필요한 최소 rung만 사용하고, A100x2
   tripwire 실행에서 population/n_e/Jbar 중 실제 원인을 가른 뒤 gate 재진입을 결정한다.

### 2026-08-17 04:44 KST — matched-temperature owner 진단 구현과 K36 실행

- [x] CMFGEN `LINEHEAT` 789,775개를 transition ion label별로 streaming 재집계했다.
  depth 67/68의 전체 signed/absolute 합은 기존 finite reference와 각각 bit-exact다.
  두 depth 모두 abs(signed ion total) 상위는 Co IV, Co III, Ni IV(`NkIV` 원표기),
  Fe IV, Ni III, Fe III 순이다. depth 67의 cgs signed total은 각각
  `9.6405240948949696e-4`, `2.6095467820823359e-4`, `1.285994761910633e-4`,
  `9.619802280420502e-5`, `7.2672289522517924e-5`, `2.9367699069063052e-5`다.
  정본은 `validation/a2_10/A2_10_CMFGEN_LINEHEAT_ION_OWNERS_2026-08-17.json`,
  SHA `7befb84bcc065c9fc809d14ea8642719456222305abdda64a12b441fa4c649a5`다.
- [x] K30 GEOMETRIC_MID proof stop의 propagated bound를 코드와 원 로그에서 독립
  분해했다. `tau=2.9469856021858767e-6`, `beta=0.9999985265086464`이고, Jbar bound는
  오직 `beta * continuum_j_absolute_uncertainty`다. 추가 non-contracting propagated
  성분은 0이며 line-rate uncertainty도 `|chi|*Jbar_bound*4pi*deck_scale`로 bit-exact
  재현된다. 단, operator가 beta/companion/Jbar 산술 전체 roundoff enclosure를 주장하지
  않는 범위는 그대로 명시했다. 정본
  `validation/a2_10/A2_10_NONOVERLAP_K30_SOBOLEV_BOUND_DECOMPOSITION_2026-08-17.json`,
  SHA `0cc684d6d0c63c3bce00b24341f51e6546cec5fcee144711b8125ecb261cab70`다.
- [x] no-bracket 뒤 `LUMINA_RADEQ_DIAG_TE_K`의 finite positive uniform T를 open bracket
  안에서만 한 번 private callback으로 평가한다. public Te/n_e/material/bracket/result는
  변경하지 않는다. invalid/out-of-bracket은 callback 없이 진단 거부한다.
- [x] `LUMINA_A210_LINE_ION_OWNER_SHELLS`로 요청한 inner shell에 대해 A2-10
  `signed/absolute/uncertainty/scaled emission/scaled absorption`을 ion-slot별 long-double로
  모은다. 전체 line universe와 shell proof가 모두 성공한 뒤에만 complete record를 내며,
  부분 scan은 출력하지 않는다. generic solver, full CPU/CUDA build, CMFGEN owner fixture,
  Lumina owner summary positive+3 negative, line-net/targeted/reference/completion/census 회귀가
  모두 PASS했다. CUDA SHA는
  `3ec4239f2d7641d753d8d5daa0df8a04558ab7937fde8ff3d8411035da1ccb72`다.
- [x] K30에서 요구된 bound 분해 조건이 충족됐으므로 blind escalation이 아니라 지정-T
  callback 완주만을 위한 K36 최소 rung을 stage했다. root는
  `/gpfs/kjhan/lumina/a210_line_owner_a100x2_nonoverlap_sobolev_k36/diag_20260816T194201Z_3ec4239f2d76`다.
  requested T는 `19059.411196903675 K`, owner shell은 0--3, physical mutation/publication
  authority는 0/NONE이다.
- [ ] 04:42:49 KST syn101 GPU 5/6, CPU 24--47에서 2초 tripwire로 실행을 시작했다.
  두 GPU preflight는 memory/utilization `0/0`; syn101 Slurm allocation과 선택 GPU 외부
  PID는 없었다. GPU 0의 별도 Python 작업과 물리 카드가 분리돼 있다. detached owner
  monitor는 staged summarizer SHA
  `430c388b2c2a0cc71e83e75328548e7d51898e784626bfe9fe8e3171bbf9e2ac`를 고정했다.

남은 순서:

1. tripwire 충돌·Slurm allocation을 계속 감시하면서 K36 R1/R2 exact publication을 확인한다.
2. REQUESTED_TE callback이 완전하면 shell 0--3 ion owner closure report를 봉인한다.
3. incomplete이면 최초 proof witness만 수거하고 물리합 없이 최소 추가 proof 여부를 판정한다.
4. Lumina shell 0 owner와 CMFGEN depth 67/68 owner를 같은 signed observable에서 비교하되
   n_e/Jbar/population이 아직 unmatched임을 명시한다.
5. 실제 원인에 따라 population/n_e/radiation 분기 하나만 선택해 non-census gate 재진입을
   준비한다.

### 2026-08-17 04:55 KST — requested-T/ion-owner 중요 감사

- [x] Fable이 요청 온도 callback의 no-bracket 반환·public Te/n_e·publication·generation
  불변성과 ion-owner의 complete-only 경계를 읽기 전용으로 감사했다. 판정은 `APPROVE`,
  필수 수정은 0이다. 정본은
  `docs/FABLE_AUDIT_A210_REQUESTED_TE_ION_OWNER_2026-08-17.md`, SHA
  `12933775a3bc016eaa1d8560f3efcad5ed0eef823bfb712d0d588bf648a50e03`다.
- [x] 선택적 방어 강화로 postprocessor가 ion-group 합과 원래 line-order signed 합의
  차이를 long-double 연산·출력 roundoff 상한 안에서 직접 검증한다. 이 상한은 증거
  검증에만 쓰며 물리값/부호를 승인·대체·repair하지 않는다. positive+4 independent
  negative selftest가 PASS했고 현재 script SHA는
  `aa2c72beab7946332a3e4158d53f2d0398b32bfd050df9eaceffacc5d258b27e`다.
- [ ] 04:52 KST K36 model은 syn101 GPU 5/6에서 R1 exact solve 중이었다. 두 카드 모두
  100% compute이며 선택 카드 외부 PID, Slurm allocation, tripwire yield는 없었다.
- [x] 종료 후 strict closure monitor PID `2095503`을 분리 실행했다. supervisor가 자연
  종료한 뒤에만 K30→K36 R1 proof-only identity, 강화된 requested-T owner closure,
  CMFGEN depth 67/68 온도정렬 비교를 순차 실행한다. 앞 단계 실패·script/input SHA drift·
  tripwire collision이면 뒤 단계를 만들지 않는다. monitor SHA는
  `42f01fdd2169a1a4a13d1d12fa81d3c9189bb6c07723272ea1c536069239ce69`이고,
  positive+SHA-drift negative 통합 selftest와 전체 owner target이 PASS했다.
- [x] CMFGEN `LINEHEAT`와 `NETRATE` 789,775개를 line identity로 전수 결합해 각 ion의
  signed net/scaled emission/scaled absorption을 복원했다. raw header의 음수 값은 실제
  scale이 아니라 `(E_lower-E_upper)/nu`; CMFGEN 원문과 VADAT의 `SCL_LN=T`,
  `SCL_LN_FAC=0.5`에 따라 `abs(raw-1)>0.5`이면 유효 scale은 정확히 1이다. skip/abs/
  repair는 없다. depth 67/68 net과 absolute sum은 기존 finite 정본과 bit-exact다.
- [x] shell-0 속도 보간점에서 CMFGEN signed net은
  `1.6469023429593025e-3 erg cm^-3 s^-1`, serialized-ledger component 진단은 emission/
  absorption `514.2714201090474/514.2697732067045`다. 큰 수의 차 구조를 명시적으로
  보존한다. 성분은 LINEHEAT 5자리/NETRATE 7자리 직렬화 한계가 있어 원 net을 대체하지
  않는다. 정본 SHA는
  `c26adc5478d2328896db59356b75d68a4bc108b639ca0d95ee69b5d784ad8da0`다.
  line별 `emission-absorption` 재구성도 depth 67/68에서 원 signed net과 각각
  `9.01e-7/4.68e-7` internal 차이이며 명시적 binary64 ULP 상한
  `2.0879e-4/3.4409e-4` 안에서 폐합했다.
- [x] Lumina requested-T owner의 동일 세 성분과 비교하되 T만 같고 n_e/pop/Jbar가
  unmatched임을 강제하는 비교기를 추가했다. extractor/comparator SHA는
  `26d70b138319141759b112a83a6cc4da2c9f74863306bee449e3e4ccc9a0a7bd` /
  `9dcdd66873021b318a3c930e419eca7b6a248d12e756b3040f41b77a6052a723`이고,
  owner 전체 selftest target이 PASS했다. 비교기는 이제 requested-T의 exact binary64
  일치, 두 CMFGEN depth의 bit-exact finite reference와 cellwise component closure,
  전 입력의 zero-repair 표식을 fail-closed로 요구한다. temperature mismatch,
  unsealed closure, repair marker 독립 음성 대조군도 모두 거부한다.
- [x] strict owner monitor가 PASS한 뒤에만 component comparison을 자동 생성하는 별도
  read-only monitor를 붙였다. script SHA는
  `f96dc6f903eaf033a94afbe2d5efc308cdff9121f5905aa3845620a98138a91c`, PID는
  `2169657`이다. 모델이나 supervisor에는 signal을 보내지 않으며 pinned comparator,
  dependency, CMFGEN component/finite SHA가 달라지면 중단한다. positive+SHA-drift
  negative 통합 selftest가 PASS했다.

### 2026-08-17 05:33 KST — K36 R1 exact publication

- [x] K36 R1은 45회에서 residual `9.6662782724980344e-9 < 1e-8`로 정상
  publication했다. 두 A100 분할, line universe `109,014,300` cell, material census와
  R6 identity/coverage는 K30 R1과 bit-exact다.
- [x] K30→K36 provisional proof-only comparison은 PASS했다. 허용된 변화는 refinement
  `30->36`과 strictly contracted component/profile upper bound뿐이다. upper-bound ratio는
  `0.07810201563450851 / 0.07810201563450851 / 0.07809452796649147`이며
  floor/cap/clamp/jitter/repair는 0이다. 최종 candidate stderr SHA는 실행 자연 종료 뒤
  strict monitor가 다시 계산해 봉인한다.
- [x] R2 seed-material generation barrier는 `r1_generation=1`, population generation
  `1->2`, Te generation/publication 불변으로 닫혔다. R2 census는 K30과 동일한
  line/exact-zero/raw-negative/mild-negative/SRCE_CHK
  `22,866,166/86,148,134/4,246,581/4,246,577/4`를 재현하고 raw population/tau를
  보존했다. floor/cap/clamp/jitter/repair는 0이다.
- [x] 06:29 KST R2 exact solve는 52회, residual
  `8.1222406993212508e-9 < 1e-8`로 publication했다. K30과 physical/solver field는
  bit-exact이고 K36 component/profile upper bound는
  `1.2329864181309809e-12 / 1.4367670712763223e-12`로 더 수축했다. Sobolev operator는
  `109,014,300` Jbar cell 전부 finite, census `4,246,581/4,246,577/4`를 재현했다.
  선택 GPU/Slurm collision, YIELD, floor/cap/clamp/jitter/repair는 없다.
- [ ] R6 generation 2 publication까지 완료됐고 모델은 R7 endpoint/requested-T diagnostic
  callback 구간에 진입했다. 요청-T는 geometric-mid보다 먼저 평가되므로 뒤쪽 별도
  geometric-mid proof stop과 소유자 진단을 혼동하지 않는다.

## 2026-08-17 08:04 KST — K36 owner 폐합과 line-saturation 진단 실행

- [x] K36 owner run은 07:16:08 KST에 요청-T callback까지 완주한 뒤 물리적
  `RADEQ_NO_BRACKET`을 fail-closed하여 `model.rc=1`, wrapper rc=70으로 자연 종료했다.
  `REQUESTED_TE` owner 108 records와 K30→K36 proof contraction은 각각 PASS이며,
  선택 GPU 5/6의 외부 PID·Slurm allocation·tripwire YIELD/COLLISION은 없었다.
- [x] 최초 자동 CMFGEN owner monitor는 `FeSIX`를 `FeS`+`IX`로 잘못 나눈 라벨 parser
  때문에 후처리에서 차단됐다. 물리 실행 실패가 아니다. longest valid element token과
  stage suffix를 함께 검사하도록 고친 뒤 owner/component finite 비교를 다시 봉인했다.
  정본 SHA는 owner `4f3238bf091bc2d209fde33d8faac5113e6852e41e2ad0d1593ae7c2bc09b4f0`,
  component `d8f65d4c0b8214fae2d047421b0dfb4065291887bd606b39b25a58f682f5775e`다.
  둘 다 temperature 외 n_e/pop/Jbar가 unmatched인 진단이며 parity/원인 판정이 아니다.
- [x] 중요 물리 분기만 Fable에 감사시켰다. aggregate absorption deficit만으로 Jbar 결함을
  확정할 수 없고, shell-0 Co/Fe/Ni IV emission 상위 최소 90%의 `tau`, `Jbar/S`,
  `beta`, CMFGEN depths 67/68 `1-ZNET`을 같은 전이로 비교하는 읽기 전용 분기 하나만
  승인됐다. 정본은 `docs/FABLE_AUDIT_A210_K36_FINITE_COMPONENT_BRANCH_2026-08-17.md`,
  SHA `4d4fbdbc3d2f69802e775616b8544ba6cbcd8cc89f2327897d298465f53bd9f0`다.
- [x] 진단은 모든 대상 양의-emission line을 먼저 수집한 뒤 정확한 내림차순 최소 prefix로
  90%를 선택한다. 물리 producer/publication은 이 배열을 읽지 않으며 allocation,
  provenance, 비유한값, incomplete scan은 모두 fail-closed한다. 요약기·CMFGEN matcher의
  양성/독립 음성대조, line-net 직접 계산, owner 비교기 전체가 PASS했다.
- [ ] A100x2 진단 root는
  `/gpfs/kjhan/lumina/a210_line_saturation_a100x2_nonoverlap_sobolev_k36/diag_20260816T224556Z_f9c2d1b826d5`다.
  07:46:55 KST syn101 GPU 5/6, CPU 32--55에서 시작했고 binary SHA는
  `f9c2d1b826d5205fa68f938c2affc2c6d9aa86772257fb95958ab9e65a95526c`다.
  08:02 KST model/supervisor와 원격 linear-time postprocessor가 모두 정상이며 충돌은 없다.
- [x] 후처리의 `Jbar/S-(1-beta)` 상쇄 판정에 Jbar 물리 오차상한뿐 아니라 별도 binary64
  평가·직렬화 증거상한을 더했다. 이는 물리 허용오차나 값 보정이 아니다. 그 상한을 포함해
  implied external-continuum component가 음/비음/미결정인지 분리하며, 물리 일치 fixture,
  음의 witness, 1-ULP 경계 음성대조가 모두 PASS했다. 비교기 SHA는
  `d48e9efbf5b656f43b9c7b1664e30dd8ea81d23a1da340e7c61de7c27795ad63`다.

다음 다섯 단계:

1. 현재 A100x2 run의 R1/R2 exact publication, R6, 요청-T 자연 종료와 tripwire 무충돌을 확인한다.
2. 원격 v2 후처리 결과를 검증한 뒤 강화된 현 비교기로 독립 재처리해 두 결과의 공통 증거를 봉인한다.
3. 상위 90%에서 optically-thick `Jbar/S < 1-beta`인지, thin-tau/CMFGEN saturation인지 가른다.
4. 후자이면 같은 full-level CMFGEN POP와 Lumina tau-implied lower population을 읽기 전용으로 대조해
   opacity/lower-population/line-universe 중 정확한 식 하나를 국소화한다.
5. III-stage null control·offline recomputation·사전등록 음성대조를 통과한 분기만 구현하고 A100x2
   non-census gate에 재진입한다.

### 2026-08-17 08:12 KST — 순차 roundoff-aware 후처리 봉인

- [x] 기존 V2 후처리가 NETRATE를 먼저 한 번만 읽도록 유지하고, V2 PASS 및 그 summary SHA를
  확인한 뒤에만 현재 roundoff-aware 비교기를 실행하는 독립 V3 bundle을 봉인했다. bundle은
  `$RUN_ROOT/postprocess_roundoff_v3`, comparator SHA는
  `d48e9efbf5b656f43b9c7b1664e30dd8ea81d23a1da340e7c61de7c27795ad63`다.
  산술 상한은 증거 분류에만 사용하고 물리 허용오차로 사용하지 않으며, physical mutation과
  floor/cap/clamp/jitter/repair 표식은 모두 0이다.
- [x] syn101 원격 V3 monitor PID `3700813`을 시작했다. V2 verdict가 없으면 대기하고, V2가
  BLOCKED/YIELDED이거나 manifest/NETRATE/summary SHA가 다르면 V3를 실행하지 않는다. 이로써
  두 postprocessor가 큰 NETRATE를 동시에 읽는 경로를 닫았다.
- [ ] 08:12 KST 물리 model PID `3637018`은 GPU 5/6에서 25분째 실행 중이다. supervisor와 V2/V3
  monitor가 모두 살아 있고 선택 GPU 외부 PID, Slurm allocation, YIELD/COLLISION은 없다.
  stdout/stderr가 초기 exact solve 동안 정지해 있는 것은 이전 K36과 같은 계산 구간이며 종료
  증거로 해석하지 않는다.

### 2026-08-17 08:19 KST — Fe/Co/Ni IV 개별 90% coverage 방어

- [x] K36 owner 총량에서 대상 방출 비중은 Co IV `82.87%`, Fe IV `10.47%`, Ni IV
  `6.65%`다. 따라서 세 이온을 합친 내림차순 90% prefix는 합계 조건을 만족하면서도 특정
  이온의 진단선을 90% 미만으로 남길 수 있다. 현재 물리 실행은 변경하지 않고, 같은 stderr의
  complete shell-0 owner 총량에 선택 row를 결합해 Fe/Co/Ni IV 각각의 coverage를 검사하는
  fail-closed checker를 추가했다.
- [x] checker는 saturation/owner report가 동일 절대경로·동일 SHA의 stderr를 사용하고,
  각 이온 identity가 `(Z, ion_stage, ion_label)=(26|27|28,3,4)`이며, owner 합과 saturation
  총합이 long-double 평가·21자리 직렬화 상한 안에서 폐합할 때만 coverage를 계산한다.
  상한은 증거 검증 전용이고 물리 허용오차가 아니다. 각 이온의 `selected/owner >= 0.9`를
  정확히 요구하며, combined PASS/per-ion FAIL 음성대조를 포함한 positive+4 negative가
  PASS했다. checker SHA는
  `97443670f61917aae44d66fc27df7674f1e03730f793d75e78183335b7f7d8bb`다.
- [x] V3 PASS 뒤에만 같은-run owner summary와 per-ion coverage를 만드는 V4 bundle을
  `$RUN_ROOT/postprocess_per_ion_coverage_v4`에 봉인하고 syn101 monitor PID `3722215`를
  시작했다. 세 이온이 모두 90% 이상이면 재실행 없이 계속하고, 하나라도 미달이면 그 사실을
  물리 원인으로 오독하지 않고 per-ion 90% prefix union 진단 재실행만 요구한다.
- [x] 최종 completion artifact와 과거 후보를 다시 전수 확인했다. K24/K30은 R1/R2 exact,
  coevolution generation, Sobolev coverage까지 정상이나 R7 이후 물리적 no-bracket으로
  `model.rc=1`이어서 completion auditor의 `model.rc=0`, R7/comparison commit 조건을
  충족하지 못했다. 따라서 기존 PASS를 재사용할 수 있는 숨은 non-census completion은 없으며,
  현재 line-saturation 원인 분기는 최종 gate 전에 실제로 필요한 미완료 단계다.
- [x] V4 monitor 자체도 실제 subprocess 통합 fixture로 검증했다. per-ion PASS,
  combined-PASS/per-ion-UNDERCOVERED 정상 결과, bundle SHA drift 차단, 불완전 V3 verdict 차단을
  모두 재현했으며 line-saturation 전체 selftest target이 PASS했다. monitor selftest SHA는
  `cf8202b0686b9c2078e9f92bab8cd41b96da821f518da27da64e96088cccd9e0`다.

### 2026-08-17 08:42 KST — line-saturation K36 R1 publication 확인

- [x] 08:37:16 KST에 R1 exact publication이 자연스럽게 발행됐다. `devices=2/2`,
  `iterations=45`, `residual=9.6662782724980344e-09 < 1e-8`, `refinements=36`,
  domain/canonical-edge hash와 두 GPU ray 분할은 앞선 K36 owner 정본과 바이트 단위로
  동일하다. timing을 제외한 signed-material census/policy와 exact record의 차이는 0이다.
- [x] 뒤이어 R6가 `valid_lines=2,180,286`, `partial_lines=0`, `unsampled_lines=0`,
  `valid_cells=109,014,300`을 발행했다. q/e set hash, profile hash, component/profile
  envelope도 앞선 K36과 바이트 단위로 동일하다. nonfinite, collision, blocked 및
  nonzero floor/clamp/jitter/repair 표식은 없다.
- [x] 실행 binary, resolved environment, active deck 55개, top-ion 3개와 V2/V3/V4
  postprocess manifest를 현재 파일에서 다시 검증해 모두 PASS했다. pre-core tau refresh와
  stage4는 계속 0이고, 진단은 publication authority가 없는 읽기 전용 경로다.
- [ ] 모델 PID `3637018`과 supervisor/V2/V3/V4 monitor는 살아 있으며 R2로 진행 중이다.
  syn101 GPU 5/6에 외부 PID 또는 Slurm allocation은 없고 tripwire도 활성 상태다.

다음 다섯 단계:

1. R2 exact/Sobolev publication과 coevolution generation barrier를 K36 정본과 대조한다.
2. REQUESTED_TE complete-only line/owner 진단과 자연 `RADEQ_NO_BRACKET` 종료를 확인한다.
3. 순차 V2→V3→V4 verdict, source-log SHA, 개별 Fe/Co/Ni IV 90% coverage를 감사한다.
4. 개별 coverage가 부족할 때만 per-ion minimal 90% prefix union 진단을 재실행한다.
5. 완전한 `tau`, `Jbar/S`, `beta`, CMFGEN `1-ZNET` 증거로 필요한 물리 분기 하나만 선택한다.

### 2026-08-17 09:23 KST — 총괄·계획 및 분석·평가 Fable 이관과 gate 정의 정정

- [x] 사용자 지시에 따라 총괄·계획과 분석·평가를 Fable로 이관했다. Codex는 코딩,
  실행, tripwire/monitoring, 문서화, 커밋을 담당한다. Fable은 여러 실행 증거를 묶어
  단계계획과 물리 귀속을 판정하고, Codex는 그 판정 경계 안에서만 구현한다.
- [x] Fable 1차 계획은 K18 closure, 현재 K36 완주, V2→V3→V4, per-ion undercoverage의
  표본추출 의미, J/O 규칙과 III-stage null control을 승인했다. 정본은
  `docs/FABLE_PLAN_ANALYSIS_TRANSFER_A210_2026-08-17.md`, SHA
  `6c5eb0228a7ff6a42a861fca69720cf099e7bbafda9d2acb13233926966a79b5`다.
- [x] 1차 계획이 K36 diagnostic의 자연 `model.rc=1 + RADEQ_NO_BRACKET`을 final
  non-census PASS 증거로 잘못 합친 범주 오류를 실제 completion auditor 계약과 대조했다.
  Fable에 즉시 정정을 요청했고, `model.rc=0`, targeted verdict PASS, R7 material commit,
  physics-comparison commit, child rc 0을 요구하는 실제 계약과 충돌함을 Fable이 인정했다.
- [x] 정정 정본은
  `docs/FABLE_PLAN_ANALYSIS_TRANSFER_CORRECTION_A210_2026-08-17.md`, SHA
  `e1021f549e03ea73609902f4d2cbdca05c01620edc99dd21847c6423d34744f2`다.
  K36은 K24/K30처럼 Stage-4 증거 생산 diagnostic으로만 봉인하고 final PASS에서 제외한다.
  미결은 K-final 발주 자격이 없으므로 gate PASS를 막는다.
- [x] 수정된 총괄 계획은 `K36 diagnostic 완주 → V2/V3/V4 및 필요 시 per-ion union →
  Fable J/O 귀속 → 귀속 확정 뒤 Codex 국소 구현/offline recomputation/negative control과
  Fable 평가 → 별도 K-final auditor PASS`다. 어떤 단계에서도 수치적 물리값 repair나
  pre-core tau refresh를 허용하지 않는다.

다음 다섯 단계:

1. 진행 중 K36 R2 exact와 REQUESTED_TE diagnostic 자연 종료를 tripwire와 함께 봉인한다.
2. V2→V3→V4를 순차 감사하고 UNDERCOVERED일 때만 ion별 최소 90% prefix union을 재실행한다.
3. Fable이 per-line tau/Jbar/S/beta/1-ZNET과 III-stage null control로 J/O 귀속을 판정한다.
4. 귀속 확정 뒤에만 Codex가 파일:라인 국소 구현, offline bracketing 회복, negative control을 수행한다.
5. 별도 A100×2 K-final에서 `model.rc=0`, R7/comparison commit, completion auditor PASS를 봉인한다.

### 2026-08-17 09:35 KST — K36 R2 exact/R6 publication 확인

- [x] 09:33:56 KST에 R2 exact publication이 자연 발행됐다. `devices=2/2`,
  `iterations=52`, `residual=8.1222406993212508e-09 < 1e-8`, `refinements=36`,
  component envelope와 domain/canonical-edge hash는 앞선 K36 owner 정본과 바이트 단위로
  동일하다. timing record만 실행시간 차이를 가지며 물리·solver record 차이는 0이다.
- [x] `CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0` operator는 Jbar `109,014,300`개를 모두 finite로
  발행했다. raw-negative/mild-negative/SRCE_CHK `4,246,581/4,246,577/4`, raw-preserved 1,
  floor/cap/clamp/jitter/repair 0을 재현했다.
- [x] R6 generation 2도 `valid_lines=2,180,286`, partial/unsampled 0,
  `valid_cells=109,014,300`이며 identity/coverage record가 K36 정본과 바이트 단위로
  동일하다. pre-core tau refresh, nonfinite, BLOCKED, collision 표식은 없다.
- [ ] 모델은 R7 endpoint와 REQUESTED_TE complete-only line-saturation/owner 진단으로
  진입했다. 이 run의 의도된 rc=1은 진단 완료이며 Fable 정정 판정에 따라 final gate
  PASS로 사용하지 않는다.

다음 다섯 단계:

1. LOWER/UPPER/REQUESTED_TE 전수 scan과 자연 rc=1 종료, tripwire 무충돌을 봉인한다.
2. V2→V3→V4 결과와 source-log SHA를 순차 감사한다.
3. V4 UNDERCOVERED일 때만 ion별 최소 90% prefix union을 같은 state에서 재실행한다.
4. 완전한 대조표를 Fable에 보내 J/O 귀속과 III-stage null control을 평가받는다.
5. 귀속 확정 뒤에만 물리 구현/offline 검증/negative control을 거쳐 별도 K-final을 발주한다.

### 2026-08-17 10:08 KST — K36 tripwire YIELD와 무오염 재실행

- [x] 최초 line-saturation run은 LOWER와 UPPER를 모두 완료했다. UPPER의
  `LINE-NET-CELL-FINITE`, owner, owner-summary, endpoint 166개 record는 앞선 K36 owner
  정본과 SHA `07c953baf27b7e9a7c312b8db1cdd80c9ebec1bce41df462a1dc76afe00f5501`로
  바이트 단위 동일하며 BLOCKED/nonzero repair는 없다.
- [x] 09:58:10 KST에 `syn101`의 Slurm heterogeneous job 예약이 순간 allocation으로
  전환되자 tripwire가 약속대로 해당 process group만 종료했다. `manual_control/YIELDED`가
  남았고 REQUESTED_TE line-saturation summary 전이므로 이 partial run은 V2/V3/V4 또는
  물리 판정에 사용하지 않는다. V2는 YIELDED, V3/V4는 선행 verdict 부재로 fail-closed했다.
- [x] 동일 binary SHA `f9c2d1b826d5205fa68f938c2affc2c6d9aa86772257fb95958ab9e65a95526c`와
  동일 owner base로 새 Slurm root를 봉인하고 A100x2 job `313057`을 제출했다. 큐 예상이
  8월 29일이어서 중복 실행 방지를 위해 현재 사용자 hold 상태로 보존한다. Slurm 제어기는
  배정 GPU의 선행 compute PID를 검사해 충돌 시 계산 전에 차단하며 SHA는
  `d66ea89908700f99974b223f4f31ae41fb7ec2cf7b7deaf38211e182858cb99d`다.
- [ ] 실제 allocation이 0이고 GPU 5/6이 빈 것을 확인한 뒤 별도 manual retry root
  `/gpfs/kjhan/lumina/a210_line_saturation_a100x2_nonoverlap_sobolev_k36/manual_retry_20260817T010714Z_f9c2d1b826d5`
  를 10:07:37 KST에 시작했다. GPU 5/6, CPU 32--55, 2초 tripwire이며 V2/V3/V4 monitor는
  각각 한 인스턴스만 대기한다. partial output을 재사용하거나 이어붙이지 않았다.

다음 다섯 단계:

1. manual retry의 R1/R2/LOWER/UPPER를 기존 K36 정본과 대조하고 tripwire를 계속 감시한다.
2. REQUESTED_TE complete-only 진단과 자연 `RADEQ_NO_BRACKET` 종료를 새 root에서 봉인한다.
3. V2→V3→V4를 순차 감사하고 UNDERCOVERED일 때만 per-ion 90% union을 재실행한다.
4. 완전한 J/O 관측량과 III-stage null control을 Fable에 보내 귀속을 판정받는다.
5. 귀속 확정 뒤에만 국소 물리 구현과 별도 A100x2 K-final gate를 수행한다.

### 2026-08-17 10:16 KST — tripwire 조회 오탐 방어와 guard3 재출발

- [x] 최초 YIELD의 `running_jobs`가 비어 있고 예약 heterogeneous job이 실제
  RUNNING/ALLOCATED가 된 이력이 없음을 확인했다. 기존 tripwire는 단 한 번의 빈
  `scontrol` 응답도 allocation으로 간주했으므로 scheduler 조회 순간 실패가 장시간
  계산을 오탐 종료할 수 있었다.
- [x] 실제 RUNNING job, nonzero CPUAlloc/AllocTRES, 선택 GPU의 외부 PID는 계속 즉시
  YIELD한다. 오직 `scontrol` 조회 불능만 2초 간격 3회 연속 재확인한 뒤 fail-closed하도록
  분리했다. 현재 node parser는 `syn101`을 `CLEAR`로 판정했고 `bash -n`과 diff 검사가
  PASS했다. tripwire SHA는
  `07a8671b4c94bf87ab7e3a475d6ad418d7acd8f61b2d9b2606d555e3af485c2a`다.
- [x] 10:07 retry는 정확한 PID/root를 확인한 뒤 `operator_stop`으로 해당 process group만
  종료했다. GPU 5/6은 0 MiB로 반환됐고 partial output은 어떤 비교에도 사용하지 않는다.
- [ ] 새 정본 후보 root는
  `/gpfs/kjhan/lumina/a210_line_saturation_a100x2_nonoverlap_sobolev_k36/manual_retry_guard3_20260817T011506Z_f9c2d1b826d5`다.
  10:15:28 KST GPU 5/6, CPU 32--55에서 시작했으며 model PID는 `4040285`다. 시작 전
  두 GPU는 모두 0 MiB이고 V2/V3/V4 monitor가 각각 하나씩 대기한다.

### 2026-08-17 11:07 KST — guard3 K36 R1 strict publication

- [x] guard3 R1은 11:05:54 KST에 45회, residual
  `9.6662782724980344e-09 < 1e-8`, refinements 36으로 publication했다. 두 A100 분할과
  domain/canonical-edge hash, component/profile bound는 이전 K36 owner 정본과 동일하다.
- [x] occurrence 0의 signed-material census, exact epoch, R6 identity, R6 coverage 네
  record를 `compare_a210_targeted_reference.py` strict mode로 대조해 차이 0 PASS를
  확인했다. R6는 valid lines `2,180,286`, valid cells `109,014,300`, partial/unsampled 0이다.
- [x] R1 시점까지 실제 Slurm allocation·선택 GPU 외부 PID·tripwire YIELD와
  BLOCKED/nonzero floor/cap/clamp/jitter/repair는 모두 0이다. 모델은 자연스럽게 R2
  population/seed-material 준비 구간으로 계속 진행 중이다.
- [x] 11:10:56 KST R1→R2 seed-material barrier가 `r1_generation=1`, Te generation
  `1->1`, population generation `1->2`, Te manifest/publication preserved 1로 닫혔다.
  R2 census/policy는 line/exact-zero/raw-negative/mild-negative/SRCE_CHK
  `22,866,166/86,148,134/4,246,581/4,246,577/4`를 재현했다. seed barrier와 누적 두
  census/policy stream 모두 이전 K36 정본과 바이트 단위 동일하고 raw material 보존 및
  floor/cap/clamp/jitter/repair 0이다.

다음 다섯 단계:

1. R1→R2 seed-material generation barrier와 R2 signed census를 이전 K36 정본과 대조한다.
2. R2 exact/Sobolev/R6 occurrence 1을 strict 비교한다.
3. LOWER/UPPER/REQUESTED_TE와 자연 `RADEQ_NO_BRACKET` 종료를 봉인한다.
4. V2→V3→V4 결과를 감사하고 완전한 증거만 Fable의 J/O 귀속 판정에 보낸다.
5. 귀속 확정 뒤에만 국소 물리 구현과 별도 A100x2 K-final gate를 수행한다.

### 2026-08-17 12:04 KST — guard3 K36 R2 strict publication

- [x] guard3 R2는 12:02:26 KST까지 exact/R6 generation 2를 출판했다. exact solve는
  52회, residual `8.1222406993212508e-09 < 1e-8`이며, two-A100 domain/edge hash와
  deterministic line universe를 보존했다.
- [x] occurrence 1의 signed-material census, exact epoch, R6 identity, R6 coverage 네
  record를 sealed K36 owner stderr occurrence 1과 strict 비교했다. 결과는
  `STRICT_BIT_EXACT`, differences 0, physical-values-modified false, repair 0 PASS다.
  실행 중인 candidate 전체 stderr SHA는 계속 변하므로 현재 report는 provisional이며,
  자연 종료 뒤 같은 비교를 최종 로그에 다시 수행한다.
- [x] Sobolev line operator는 `CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0`, Jbar
  `109,014,300` cell 전부 finite, generation-2 valid line `2,180,286`, partial/unsampled
  0을 재현했다. signed census `22,866,166/86,148,134/4,246,581/4,246,577/4`와
  raw material을 그대로 보존했고 floor/cap/clamp/jitter/repair는 모두 0이다.
- [x] 선택 GPU 외부 PID, 실제 syn101 Slurm allocation, tripwire YIELD/COLLISION,
  BLOCKED 표시는 없다. 모델은 LOWER/UPPER/REQUESTED_TE 진단으로 자연 진행 중이다.

다음 다섯 단계:

1. LOWER와 UPPER branch publication 및 reference identity를 확인한다.
2. REQUESTED_TE owner closure와 자연 `RADEQ_NO_BRACKET` 종료를 봉인한다.
3. 종료 후 occurrence 0/1 strict 비교를 최종 stderr SHA로 다시 생성한다.
4. V2→V3→V4를 순차 감사하고 완전한 증거만 Fable의 J/O 귀속 판정에 보낸다.
5. 귀속 확정 뒤에만 국소 물리 구현과 별도 A100x2 K-final gate를 수행한다.

### 2026-08-17 13:32 KST — V4 union 정정·구현 감사와 A100x2 판정런 시작

- [x] guard3는 12:49 KST에 자연 `model.rc=1`, `RADEQ_NO_BRACKET`, generation/publication
  보존으로 종료했다. V2/V3는 각각 PASS했고 V4는 Fe/Co/Ni IV coverage
  `0.738388/0.935772/0.710002`로 사전등록된 `UNDERCOVERED` 분기에 들어갔다.
- [x] 현 binary가 combined 90% prefix 929개만 직렬화하고 per-ion mode가 없어서
  literal 동일 SHA union 재실행이 불가능함을 확인했다. Fable은 기존 산출물 복원과 union
  생략을 기각하고, 동일 물리 baseline/state를 strict 비교로 입증하는 diagnostic-only
  새 SHA를 유일한 적법 경로로 승인했다.
- [x] 새 mode 2는 Fe/Co/Ni IV 각각 scaled emission 0.9 최초 도달 최소 prefix의 union만
  기록한다. 최소성은 큰 수 차감 없이 마지막 행 추가 직전 누적값으로 증명한다. 기존
  mode 1 guard3 재요약은 원본 V2 JSON과 SHA
  `4de6f38cf721aff5dc267b4e81b6482e874c99002b43a7534820226b820f6e20`로 byte-identical이다.
  행 삭제·scaled-emission 섭동·교집합 row 섭동 음성 대조는 모두 rc=4를 재현했다.
- [x] Fable 구현 감사가 `APPROVE`를 내렸다. 새 fat binary SHA는
  `14b199f12d29246e1b7c7173d0ef9e8a9254ba4e8614cc0a838e85c200ad8825`이며 물리
  producer/publication 소비자는 0, floor/cap/clamp/jitter/repair는 0이다.
- [ ] 새 run root
  `/gpfs/kjhan/lumina/a210_line_saturation_per_ion_union_a100x2_nonoverlap_sobolev_k36/diag_20260817T042955Z_14b199f12d29`
  를 동일 owner base와 mode `1->2`만 바꿔 봉인했다. 13:31:58 KST syn101 GPU 5/6,
  CPU 32--55에서 2초 tripwire로 시작했으며 model PID는 `386580`이다. Slurm allocation은
  0이고 GPU4의 다른 세션 작업과 GPU/CPU가 겹치지 않는다.

다음 다섯 단계:

1. R1/R2 exact/R6와 seed-material barrier를 guard3 occurrence 0/1과 strict 비교한다.
2. LOWER/UPPER/REQUESTED_TE 공통 baseline과 자연 `RADEQ_NO_BRACKET` 종료를 봉인한다.
3. V2/V3/V4에서 per-ion prefix 최소성·coverage ≥0.9·owner closure를 확인한다.
4. guard3와 새 union의 세 ion 교집합 row 전수를 byte-identical 비교한다.
5. 모든 계보 검증 PASS 뒤에만 Fable의 Stage 4 J/O 귀속 평가로 진행한다.

### 2026-08-17 14:24 KST — per-ion union A100x2 R1 strict publication

- [x] mode 2 판정런의 R1은 14:22:33 KST에 45회, residual
  `9.6662782724980344e-09 < 1e-8`로 publication했다. exact epoch의 domain/edge hash,
  R6 line identity/coverage, component/profile error bound는 guard3 occurrence 0과
  화면상 및 구조화 비교 모두 동일하다.
- [x] occurrence 0의 signed-material census, exact epoch, R6 identity, R6 coverage를
  `compare_a210_targeted_reference.py` strict mode로 비교해 `BIT_EXACT`, differences 0,
  physical-values-modified false, repair 0 PASS를 얻었다. candidate stderr는 실행 중이므로
  현재 report SHA `b481ee92c046f9987397cb06f2c7d172fe2005d59a8d70f510f42db2631de64d`는
  provisional이며 자연 종료 후 최종 SHA로 다시 생성한다.
- [x] R1 이전 external gamma publication도 generation 1, epoch 1683072와 heating/nonthermal
  manifest SHA가 guard3와 byte-identical이다. signed-material census는 raw negative 0이고
  floor/clamp/jitter/repair 0을 보존한다.
- [x] 14:27:28 KST R1→R2 seed-material barrier가 `r1_generation=1`, Te generation
  `1->1`, population generation `1->2`, Te manifest/publication preserved 1로 닫혔다.
  R1/R2 census·policy를 포함한 다섯 record의 filtered SHA는 guard3와 동일한
  `6e18f2bcdaf11aee596c39bbc5e94eb4b10dd676a14ad4b5e05c52b37e8ef857`이다.
  R2 raw negative `4,246,581`건은 삭제·수선 없이 보존되고 floor/cap/clamp/jitter/repair는 0이다.
- [x] syn101 GPU 5/6은 R1까지 100% 계산 중이며 GPU4의 다른 세션과 겹치지 않는다.
  Slurm allocation, 외부 PID, tripwire YIELD/COLLISION은 없다.

다음 다섯 단계:

1. R2 exact/Sobolev/R6 occurrence 1을 strict 비교한다.
2. 자연 `RADEQ_NO_BRACKET` 종료와 V2→V3→V4 결과를 봉인한다.
3. mode 1/2 교집합 및 전체 phase baseline stream을 byte-identical 비교한다.
4. 완전한 증거를 Fable의 Stage-4 J/O 귀속 평가에 넘긴다.
5. 귀속 결과에 따라 read-only 국소화 또는 허가된 물리 구현 뒤 non-census gate를 판단한다.

### 2026-08-17 15:20 KST — per-ion union A100x2 R2 strict publication

- [x] mode 2 판정런 R2는 15:18:34 KST에 52회, residual
  `8.1222406993212508e-09 < 1e-8`로 publication했다. exact epoch의 domain/edge hash와
  R6 generation 2 line universe, profile/component bounds는 guard3 occurrence 1과 동일하다.
- [x] occurrence 1의 signed-material census, exact epoch, R6 identity, R6 coverage를
  strict 비교해 `BIT_EXACT`, differences 0, physical-values-modified false, repair 0 PASS를
  얻었다. 실행 중 candidate stderr SHA는
  `a3bf6bb4b2c9e60ebe56494e1e4766bb5bf493c9e9a687faa62dc1c3fbf78ca3`, provisional
  report SHA는 `2777c827bd589b206ed7044bf8ac7ba7fd1adef5f845e8deeafc5721ec861692`다.
  자연 종료 뒤 최종 stderr SHA로 다시 생성한다.
- [x] R2 직후 syn101 Slurm allocation은 0, held job 313057은 `JobHeldUser`, 선택 GPU의
  외부 PID와 tripwire YIELD/COLLISION은 0이다. GPU exact allocation 해제 뒤 모델은
  LOWER/UPPER/REQUESTED_TE 진단 구간으로 자연 진행한다.

다음 다섯 단계:

1. LOWER/UPPER/REQUESTED_TE 공통 baseline과 자연 `RADEQ_NO_BRACKET` 종료를 확인한다.
2. 자동 V2→V3→V4에서 per-ion coverage와 prefix 최소성을 봉인한다.
3. mode 1/2 교집합 row 전수와 전체 phase stream을 byte-identical 비교한다.
4. 최종 stderr SHA로 R1/R2 strict report를 다시 생성하고 계보를 봉인한다.
5. 증거 완결 뒤 Fable Stage-4 J/O 귀속 판정과 허가된 다음 분기로 진행한다.

### 2026-08-17 Stage4 J/O read-only audit and sealed-state offline census

- [x] Fable 판정은 J: selected IV 1,282/1,282 (Fe 482, Co 426, Ni 374), O 0이다.
  per-line III evidence는 아직 없으며, K-final 또는 source edit 권한은 부여되지 않았다.
- [x] Codex 소스 감사는 producer `lumina_cmfgen.c:6188-6274`의 bulk upper-population
  cache와 consumer `lumina_plasma.c:14481-14740`의 trial-candidate `n_upper`가
  분리되어 있음을 확인했다. line view는 radiation generation만 결박하고 population
  generation을 운반하지 않는다. 이는 J 후보를 구조적으로 지지하지만 단독 확정은 아니다.
- [x] 봉인 stderr SHA `07dc0366951dce4bc19d2832acb503ab74981a31db94914ba3d10822343e173c`에서
  1,282 saturation row를 오프라인 검사했다. 모든 row의 현재 Jbar/S/beta는 finite이고
  `Jbar=beta*Jcont+(1-beta)*S`의 대수적 복원은 가능했지만, 독립적인 `J_cont`와
  `S_probe` 필드가 0/1,282건뿐이다. 결과는
  `validation/a2_10/A2_10_STAGE4_JBAR_OFFLINE_2026-08-17.json`의
  `INSUFFICIENT_INDEPENDENT_FIELDS`이며, rc=0 예측으로 과장하지 않는다.
- [ ] 다음은 III per-line negative-control과 독립 Jcont/Sprobe를 보존하는 read-only
  capture 설계/실행이다. 이 두 입력이 봉인되기 전 source edit 및 최종 non-census gate는 금지한다.

### 2026-08-17 17:36 KST — 목표 이온 배선 후 5단계 재검증

- [x] 기본 IV(`target_ion=3`) stage bundle을 새 binary로 재생성했다. 새 stage root는
  `/gpfs/kjhan/lumina/a210_line_saturation_dynamic_target_a100x2_nonoverlap_sobolev_k36/diag_20260817T173500Z_iv3`이며, mode 2와 target contract가 봉인됐다.
- [x] 대체 target ion=`2` stage contract와 export를 생성하고, 기존 sealed union stderr를
  필드 경계 보존 방식으로 변환한 synthetic target-2 log를 현재 summarizer로 검증했다.
  `TARGET2_SUMMARY PASS`, target ion 2, selected rows 1,282이다.
- [x] 실제 A2-10 mode-2 A100x2 실행은 sealed run
  `/gpfs/kjhan/lumina/a210_line_saturation_per_ion_union_a100x2_nonoverlap_sobolev_k36/diag_20260817T042955Z_14b199f12d29`를 사용했다. `model.rc=1`, 자연 `RADEQ_NO_BRACKET`, selected rows 1,282이며 물리 실행은 재사용 가능한 sealed state로 확인됐다. 별도 held Slurm job 313057은 변경하지 않았다.
- [x] 현재 repo comparator로 CMFGEN NETRATE finite transition 1,282건을 모두 매칭했다.
  verdict는 `FINITE_TRANSITION_MATCH_DIAGNOSTIC_NOT_STATE_PARITY_NOT_CAUSE_CLAIM`이다.
- [x] 새 coverage/intersection 감사가 모두 PASS했다. Fe/Co/Ni IV coverage는 각각
  `0.9002542817714418`, `0.9000529441915179`, `0.9000952343951808`이고, mode 1/2
  shared row 738건은 strict byte-identical이다. floor/cap/clamp/jitter/repair는 0이다.

새 감사 산출물은 위 stage root의 `current_summary.json`, `current_cmfgen_comparison.json`,
`current_coverage.json`, `current_intersection.json`이다. 다음은 독립 Jcont/Sprobe를
보존하는 III-stage negative-control capture이며, 그 전에는 source edit와 K-final gate를
진행하지 않는다.
### 2026-08-17 — Stage-4 independent `J_cont`/`S_probe` capture (in flight)

- Added opt-in `LUMINA_A210_INDEPENDENT_CAPTURE=1` to the CMFGEN fine-grid
  producer.  After the normal line-inclusive exact solve has been profiled,
  the same owner solver is run again with the untouched continuum arrays only:
  `chi_tot=chi_es+chi_abs`, `S_fixed=chi_abs*B/(chi_es+chi_abs)`.  The second
  field is profile-averaged into read-only `jbar_line_det_continuum` arrays;
  production Jbar/rates are not replaced.
- A2-10 rows now carry numeric `J_cont`, its propagated profile error bound,
  and `S_probe` (line-material emission/effective-opacity source, computed
  without Jbar) only when the independent capture is complete.  Otherwise the
  fields remain explicitly unavailable and Stage-4 stays insufficient.
- CPU/CUDA builds and all four existing A2-10 parser/coverage/intersection
  self-tests pass.  New sealed run submitted: job `313169`,
  `/gpfs/kjhan/lumina/a210_line_saturation_independent_jcont_a100x2_nonoverlap_sobolev_k36/diag_20260817T085348Z_stage4`.
- Fable review was attempted for this physical-independence design but the
  local Claude CLI is not authenticated (`Not logged in`); no external verdict
  is being fabricated.  Awaiting the real finite-value capture before any
  K-final/causal claim.
- The new capture/target-ion knobs were added by rerunning the generated
  environment-universe derivation; the first manual launch therefore failed
  closed on the stale allow-list and was not a physical result.  CUDA was
  rebuilt and the corrected run is now executing manually on `syn101` A100
  GPUs 0/1 under `syn101_tripwire`; pre-existing GPU-4 Python PID is the only
  baseline process and no external PID has triggered termination.
