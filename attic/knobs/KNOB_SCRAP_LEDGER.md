# 창고 대장 — 노브 스크랩

생성 `scripts/knob_scrap_ledger.py`. **생존이 예외, 스크랩이 기본.**

| 부류 | 수 | 뜻 |
|---|---|---|
| S-CONTRACT | 1 | 0층 계약이 요구 — 확증된 자산 |
| S-INPUT | 7 | 경로·자원 지정 — 노브가 아니라 입력 |
| P-VERDICT | 47 | **판정런이 넘김 — 개별 물리 판정 대상** |
| SCRAP-CLAMP | 42 | 이름이 clamp/floor/ceil — 규약상 물리가 아니다 |
| SCRAP-FOSSIL | 218 | 과거 런처만 설정 — 실패의 화석층 |
| SCRAP-DEAD | 104 | 아무도 설정 안 함 — 분기가 밟히지 않는다 |

합계 **419**

## P-VERDICT (개별 판정 대상 — 여기만 사람이 판정한다)

| 노브 | 사이트 | 파일 |
|---|---|---|
| `LUMINA_BF_OPACITY` | 4 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_BF_RATE_POPS` | 1 | lumina_plasma.c |
| `LUMINA_CMFGEN_ALI_ITER` | 5 | lumina_cmfgen.c, lumina_cuda.cu |
| `LUMINA_CMFGEN_FROZEN_ALI` | 1 | lumina_cuda.cu |
| `LUMINA_CMFGEN_FROZEN_CONT` | 1 | lumina_cuda.cu |
| `LUMINA_CMFGEN_LINE_EPS_PHYS` | 2 | lumina_cmfgen.c |
| `LUMINA_CMFGEN_SIGMA_BF` | 2 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_CMF_BF_MILNE` | 1 | lumina_plasma.c |
| `LUMINA_CMF_DEP_SOURCE` | 1 | lumina_cmfgen.c |
| `LUMINA_CMF_EPAY_TAUBIN` | 2 | lumina_cmfgen.c |
| `LUMINA_CMF_LINERES_JBAR` | 5 | lumina_cmfgen.c, lumina_cuda.cu, lumina_plasma.c |
| `LUMINA_CMF_SOLVE_GPU` | 1 | lumina_cmfgen.c |
| `LUMINA_COUPLED_NEWTON` | 4 | lumina_cuda.cu |
| `LUMINA_DIFFUSE_INNER_BC` | 2 | lumina_cuda.cu |
| `LUMINA_DIP_TRACE` | 2 | lumina_cuda.cu |
| `LUMINA_DYNAMIC_TRANSPROB` | 4 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_ENERGY_BUDGET` | 2 | lumina_cuda.cu |
| `LUMINA_ETLA_ALLOW_HEAT` | 1 | lumina_plasma.c |
| `LUMINA_FROZENIN_DR` | 3 | lumina_plasma.c |
| `LUMINA_GAMMA_DEP` | 4 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_INNER_BB_SCALE` | 1 | lumina_cmfgen.c |
| `LUMINA_J_DAMP` | 1 | lumina_cuda.cu |
| `LUMINA_KPACKET` | 5 | lumina_cuda.cu, lumina_plasma.c |
| `LUMINA_LINE_INTERACTION` | 2 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_MACROATOM_EWEIGHT` | 1 | lumina_plasma.c |
| `LUMINA_MACROATOM_IDOWN_BETA` | 1 | lumina_plasma.c |
| `LUMINA_MACROATOM_NEUTRAL_E` | 1 | lumina_plasma.c |
| `LUMINA_MAX_INTERACTIONS` | 2 | lumina_cuda.cu |
| `LUMINA_NLTE` | 4 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_NLTE_ASSEMBLE_GPU` | 1 | lumina_cuda.cu |
| `LUMINA_NLTE_COLL_FIX` | 1 | lumina_plasma.c |
| `LUMINA_NLTE_GREY_ITERS` | 1 | lumina_cuda.cu |
| `LUMINA_NLTE_GREY_TAU` | 1 | lumina_cuda.cu |
| `LUMINA_NLTE_ION_LOCK` | 2 | lumina_plasma.c |
| `LUMINA_NLTE_LOCK_START_ITER` | 1 | lumina_plasma.c |
| `LUMINA_NLTE_PER_ION_RESCALE` | 2 | lumina_plasma.c |
| `LUMINA_NLTE_SKIP_Z` | 2 | lumina_cuda.cu, lumina_plasma.c |
| `LUMINA_NLTE_START_ITER` | 4 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_PURE_CMFGEN` | 3 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_PURE_CMFGEN_ITER` | 2 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_RADEQ_DAMP` | 1 | lumina_plasma.c |
| `LUMINA_RADEQ_FB_RATE` | 1 | lumina_plasma.c |
| `LUMINA_RADEQ_TE` | 4 | lumina_cuda.cu, lumina_main.c |
| `LUMINA_RADEQ_VR_STD` | 1 | lumina_plasma.c |
| `LUMINA_SIMUL_NESTED` | 1 | lumina_plasma.c |
| `LUMINA_SUPER_CUTOFF` | 2 | lumina_atomic.c, lumina_element_wide.c |
| `LUMINA_TAU_BY_ION` | 2 | lumina_plasma.c |

## SCRAP-CLAMP (규약 위반 후보 — 전량 확인)

| 노브 | 사이트 | 파일 |
|---|---|---|
| `LUMINA_CAP_FORCE_ESCAPE` | 2 | lumina_cuda.cu |
| `LUMINA_CAP_REAL_ONLY` | 2 | lumina_cuda.cu |
| `LUMINA_CMFGEN_EPS_CAP` | 3 | lumina_cmfgen.c |
| `LUMINA_CMFGEN_EPS_FLOOR` | 3 | lumina_cmfgen.c |
| `LUMINA_CMF_EPAY_SMIN` | 2 | lumina_cmfgen.c |
| `LUMINA_CMF_FINE_SL_CLAMP` | 2 | lumina_cmfgen.c |
| `LUMINA_CMF_FINE_TAUMIN` | 1 | lumina_cmfgen.c |
| `LUMINA_CMF_OBS_TAUMIN` | 2 | lumina_cmfgen.c |
| `LUMINA_COEVOLVE_TAU_FLOOR` | 1 | lumina_cuda.cu |
| `LUMINA_DETFLUOR_SL_CEIL` | 3 | lumina_cuda.cu |
| `LUMINA_DR_FLOOR_CMS` | 1 | lumina_plasma.c |
| `LUMINA_EVENT_LOG_CAP` | 2 | lumina_cuda.cu |
| `LUMINA_EVENT_LOG_LAMBDA_MAX` | 2 | lumina_cuda.cu |
| `LUMINA_FIX_MA_J_UNCLAMP` | 1 | lumina_plasma.c |
| `LUMINA_FI_CLAMP_SL` | 1 | lumina_plasma.c |
| `LUMINA_HRESP_CLAMP` | 1 | lumina_plasma.c |
| `LUMINA_JBAR_MIN` | 3 | lumina_cuda.cu, lumina_plasma.c |
| `LUMINA_J_CAP_FACTOR` | 1 | lumina_plasma.c |
| `LUMINA_J_FLOOR_FACTOR` | 1 | lumina_plasma.c |
| `LUMINA_J_NU_UV_CAP` | 1 | lumina_plasma.c |
| `LUMINA_J_NU_UV_CAP_LAMBDA_MAX` | 1 | lumina_plasma.c |
| `LUMINA_J_NU_UV_CAP_LAMBDA_MIN` | 1 | lumina_plasma.c |
| `LUMINA_J_NU_UV_W_CAP` | 1 | lumina_plasma.c |
| `LUMINA_KPEMISS_BSRC_PHOT_WFLOOR` | 1 | lumina_cuda.cu |
| `LUMINA_KPEMISS_BSRC_WFLOOR` | 1 | lumina_cuda.cu |
| `LUMINA_KPEMISS_FB_OTS_NUMIN` | 1 | lumina_cuda.cu |
| `LUMINA_LINE_THERM_SMAX` | 5 | lumina_cuda.cu, lumina_plasma.c |
| `LUMINA_MA_CAP_EMIT` | 4 | lumina_cuda.cu |
| `LUMINA_MA_INTERNAL_CAP` | 2 | lumina_cuda.cu |
| `LUMINA_NLTE_BK_CEIL` | 1 | lumina_cuda.cu |
| `LUMINA_NLTE_COLL_FLOOR` | 2 | lumina_nlte_assemble.cu, lumina_plasma.c |
| `LUMINA_NLTE_FLOOR_BKMAX` | 1 | lumina_cuda.cu |
| `LUMINA_NLTE_FLOOR_MODE` | 1 | lumina_cuda.cu |
| `LUMINA_NLTE_FLOOR_REG` | 1 | lumina_plasma.c |
| `LUMINA_NLTE_INV_CEIL` | 1 | lumina_plasma.c |
| `LUMINA_NLTE_LTE_FLOOR` | 1 | lumina_cuda.cu |
| `LUMINA_RADEQ_OMEGA_FLOOR` | 2 | lumina_plasma.c |
| `LUMINA_SIMUL_CAP_TOPION` | 1 | lumina_plasma.c |
| `LUMINA_STAGE4_BK_CAP` | 1 | lumina_plasma.c |
| `LUMINA_TE_STEP_CLAMP` | 1 | lumina_plasma.c |
| `LUMINA_UVOPT_EMIT_LAM_MAX` | 1 | lumina_plasma.c |
| `LUMINA_UVOPT_EMIT_LAM_MIN` | 1 | lumina_plasma.c |
