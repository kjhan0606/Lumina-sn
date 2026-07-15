# CMFGEN 빌드·세팅·런 가이드 (LUMINA-SN 검증 표준 인프라)

**작성 2026-07-15.** 검증 기준=CMFGEN 확정(user)에 따른 자체 런 인프라의 전 지식.
1차 자료: `/gpfs/kjhan/cmfgen_runs/toy06_2d/README.txt`(런처·픽스사이클 원본),
빌드/셋업 에이전트 보고. 이 문서가 마스터 요약.

---

## 1. 자산 위치

| 자산 | 경로 |
|---|---|
| 소스 (18jun25 릴리스) | `/gpfs/kjhan/cmfgen_src/cur_cmf/` |
| 실행파일 (80개, 빌드완료) | `/gpfs/kjhan/cmfgen_src/cur_cmf/exe/` |
| 원자데이터 (21jun23, 4.4GB) | `/gpfs/kjhan/cmfgen_21jun23/atomic/` |
| 문서 (Lowell 가이드 등) | `/gpfs/kjhan/cmfgen_doc/` |
| toy06 모델 디렉토리 | `/gpfs/kjhan/cmfgen_runs/toy06_2d/` |
| toy06 공식 입력 | `data/standart_data1/input_models/snia_toy06_2d.dat` (807존, t=2d) |
| 검증 진리 (공개 결과) | `data/standart_data1/toy06/{phys,ionfrac_*,spectra}_toy06_cmfgen.txt` |
| SN_HYDRO_DATA 포맷 예제 | `data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d` |

## 2. 빌드 (완료 상태; 재빌드 시 절차)

- **컴파일러**: gfortran 13.2.0 (`/opt/ohpc/pub/compiler/gcc/13.2.0/bin/gfortran`). ifx 미사용.
- **소스 패치 0** — 전부 `Makefile_definitions`만 수정 (원본 `Makefile_definitions.orig_bak`):
  1. `INSTALL_DIR` → `/gpfs/kjhan/cmfgen_src/cur_cmf/` (HOST=crc 하드코딩 우회)
  2. gfortran13 legacy 플래그: `-fallow-argument-mismatch -fallow-invalid-boz -std=legacy -fno-range-check` (FG/FD/FFREE/FFRED에)
  3. `-ffpe-trap=invalid` 제거 (무해한 NaN 연산에 SIGFPE 방지)
  4. BLAS/LAPACK → 동봉 `libmy_blas.a`/`libmy_lapack.a` (MKL 불요; 단 번들 LAPACK은 dgetrf 계열 9루틴뿐 — 부족 시 `module load intel/mkl/2025.3`)
  5. `X11LIB` → `-L/usr/lib64`
  6. **PGPLOT 스텁**: `pgplot_stub/pgplot_stub.f` (no-op 49루틴) — 시스템 PGPLOT 부재 우회. 그래픽만 무동작, **EDDFACTOR/J_nu 데이터 경로는 온전**.
- **빌드 명령**: `nice -n 19 make -j1 all` (톱레벨; `-j1` 필수 — ar 아카이브 레이스).
- 필수 4종: `cmfgen_dev.exe`(본체) / `cmf_flux.exe`(관측자계 플럭스) / `dispgen.exe` / **`plt_jh.exe`(J_nu 추출)**.

## 3. SN Ia 모델 세팅 레시피 (Lowell 가이드엔 없음 — 소스에서 확립)

파서·리더: `rd_control_variables.f`, `rd_sn_data.f`, `set_rv_hydro_model_v3.f`.

### 3.1 SN_HYDRO_DATA (구조 입력; 포맷=rd_sn_data.f + DDC15 예제)
- 헤더 키: `Number of data points:` / `Number of mass fractions:` / `Number of isotopes:` / `Time(days) since explosion:`
- 섹션(격자 **외곽→내부** 순): Radius(10^10cm), Velocity(km/s), Sigma(dlnV/dlnr−1),
  Temperature(10^4K), Density, Atom density, Electron density, **Kappa(cm²/g)**(격자빌더가 τ_Ross용으로 사용),
  원소별 `<SPEC> mass fraction` 블록, 동위원소별 `<SPEC> <A> mass fraction` 블록.
- **decay-부모 종은 원소·동위원소 질량분율 20% 이내 일치 필수** (전량 0인 외곽도 1e-10 플로어로 맞출 것 — "inconsistent total/isotope" 에러 방지).
- toy06 생성기: `mk_sn_hydro.py` (v∈[1000,36000] km/s 절단, 700존).

### 3.2 VADAT SN 필수 블록
```
11        [VEL_LAW]        ← SN 모드 (R격자를 SN_HYDRO_DATA의 τ_Ross 스케일로)
2.0       [SN_AGE]         ← SN_HYDRO_DATA의 시각과 일치
T         [PURE_HUB]
T         [INC_RAD_DECAYS] ← NUC_DECAY_DATA 읽음 (Ni56→Co56→Fe56 체인 파일)
F         [TRT_NON_TE]     ← 첫 모델 단순화(decay 열화); 완전물리는 T
USE_HYDRO [SN_T_OPT]       ← T/Ne를 hydro 파일에서
LOCAL     [GAMRAY_TRANS]   ← 2d엔 γ 국소 포획 OK (GAMRAY_PARAMS 불요)
LTE       [DC_METH]        ← 신규 모델 콜드스타트 (departure 파일 불요)
F         [DO_DDT]         ← 첫 epoch=정상상태; 시퀀스부턴 T
T         [REL_OBS] [REL_CMF] [USE_J_REL] [INCL_REL] [INCL_ADV_TRANS] ← SN이 강제
F         [AUTO_ADD]       ← 신규 모델(이전 epoch 파일 없음)
F         [IT_ON_T]        ← 그레이-T 초기화 스킵 (상대론 MOM_JREL_GREY 2차 발산 회피)
```
- 신버전 필수 키 (없으면 파싱 에러): `INC_PEN, VTURB_MIN/MAX, GLOBAL_PROF, OPAC_LIMS, DOP_LIM, VOIGT_LIM, CHK_NG, IB_METH`.
- 주석 행도 `!` 접두 필수.

### 3.3 MODEL_SPEC / 원자데이터 링크
- 이온 포함: `[<IonID>_ISF] NV,NS,NF` 행. IonID 명명: Si=`Sk`, Ni=`Nk`, S=`S`, II→`2`, VI→`SIX` 등.
- 심볼릭 링크 4종/이온: `<IonID>_F_OSCDAT`, `PHOT<IonID>_A`, `<IonID>_F_TO_S`, `<IonID>_COL_DATA` (`setup_links.sh`).
- **NF 자동 캡**: OP phot 파일에 이온화 한계 위 자동이온화 준위 포함 → `SET_CONT_FREQ` 음수 주파수 에러. NF를 0.98×E_ion 아래로 캡 (`gen_atomic.py`).
- osc↔f_to_s 정합 필수 (다른 버전 섞으면 super-level 링크 mismatch — SkIII 사례: 19apr23 osc_data+f_to_s_ls 세트로 통일).
- toy06 선택: Si/S/Ca III–V + Fe/Co/Ni III–VI (21 full ion + 6 bare limit), SL≈1228, ND=90(빌더 정착 ~65→90), NC=15, NP=105.

## 4. ★신규 SN 모델 안정화 사다리 (3련속 사망 부검, 근원 확증)

**증상**: iteration 2의 J-루프서 `comp_j_blank.f` "Mean intensity blowing up" (|RJ(1)|>1e30).

| Round | 처방 | 결과 |
|---|---|---|
| R1 | 외곽 격자 세분화 (N_OB_INS 3→8, ND 70→90) — 코드 권장 | **null** (동일 사망) |
| R2 | N_TYPE N_ON_J→G_ONLY (모멘트 폐쇄 교체) | **null** (동일 사망) |
| R3 | T_EXC 0.4→1.2 (웜스타트) + MAX_LAM 캡 | **더 빨리 사망** — T_EXC 1.2가 iter1 필드 자체를 파괴(여기준위 부스트→반전→음의 불투명도) |
| **R4** | **MAX_LAM 1e10→10** 단독 (T_EXC 0.4 원복) | **★성공** |

**확증된 근원**: `fiddle_pop_corrections_v2.f`의 인구 증가 캡=MAX_LAM(기본 1e10)
→ LTE 콜드스타트 iter 1에서 인구 ×10¹⁰ 점프 → 병적 불투명도 → iter 2 J 폭파.
**fix = MAX_LAM=10** (per-depth 캡). 수렴 추세: 요구보정 e43→e35 (−8 dex/iter).
LIMIT 옵션(solveba_v13.f:129)은 감소만 캡(110%) — 증가 캡이 별도(MAX_LAM)라는 게 함정.

**교훈**: 신규 SN 콜드스타트는 (1) MAX_LAM=10, (2) T_EXC=0.4(웜스타트 금지), (3) IT_ON_T=F.
그리드·폐쇄 스킴은 무죄였음.

## 5. 런/감시/재시작 절차

```bash
cd /gpfs/kjhan/cmfgen_runs/toy06_2d
bash setup_links.sh     # 링크 재생성 (101개, 깨진 링크 0 확인)
bash run.sh             # nice -19, OMP=32, → batch.log / 진단은 OUTGEN
```
- **진행 확인**: `grep -c "Current great iteration count" OUTGEN`; 보정 추이 `grep "Maximum %" OUTGEN | tail`; 셸별 수렴 `CORRECTION_SUM`.
- **우아한 정지**: IN_ITS의 `NUM_ITS=0` (현 iteration 마치고 종료).
- **체크포인트**: `SCRTEMP`(+POINT1/POINT2) — 그대로 두고 재실행하면 이어달림. NUM_ITS 소진 시 늘려서 재실행.
- **완전 재시작**(입력 변경 후): `rm -f SCRTEMP POINT1 POINT2 EDDFACTOR* STEQ_VALS CORRECTION_SUM`.
- 무해 경고: `MOM_J_REL_V9: excessive iteration count`(초기 iter 소수 주파수), `Unable to open EDDFACTOR_INFO`(첫 iter — 새로 계산함).
- iter당 비용: ND=90·NCF~180k에서 ~40-60분 (OMP 실효 ~2.5/32코어 — STEQ/EDDFACTOR 구간 병렬 저활용; NCF coarsen이 최대 가속 레버 `FRAC_SP/AMP_FAC/MAX_BF/dV_LEV`).

## 6. J_nu 추출 (검증 표준의 핵심 산출물)

- CMFGEN은 fine CMF 주파수격자의 J_ν를 **`EDDFACTOR`**(direct-access, `EDDFACTOR_INFO` 헤더)에 기록 (`comp_j_blank.f`가 씀).
- **`plt_jh.exe`**: EDDFACTOR(+_INFO) + `RVTJ`(r/T/V 격자) + `SCRTEMP`을 읽어 깊이별 J(λ) 또는 주파수별 J(depth)를 **ASCII로 내보냄** (Y단위 NAT/Flam/FNU 선택).
- 대화형 도구 — 배치는 옵션 시퀀스 파이프 or `.sve` 커맨드 파일.
- 연속체-only J는 `EDD_CONT`; 관측자계 스펙트럼은 `cmf_flux.exe`(OBS_FRAME_INPUT).

## 7. 시간 시퀀스 (2.0d → 19.48d 공개 epoch 사다리)

1. 현 epoch 수렴 후 `SN_HYDRO_FOR_NEXT_MODEL` 출력 확보.
2. 다음 epoch 디렉토리: `DO_DDT=T`, `TS_NO` 증가, `SN_AGE=<다음>`, `INCL_DJDT=T`,
   시작 인구 = 이전 epoch (LTE 아님 — SCRTEMP/이전 출력 체인).
3. 공개 epoch 사다리(TIMES): 2.0 → 2.02 → 2.05 → … → **19.48** (28스텝).
4. 19.48d 도달 시 판정: 공개 phys/ionfrac/spectra 정합(falsifier) → 통과 시 J_nu/Γ/α/b_k 추출 → Lumina 3자대조.

## 8. 기지 리스크

- 번들 LAPACK 최소(9루틴) — 새 옵션이 다른 루틴 요구 시 MKL로 전환.
- PGPLOT 스텁 — 화면/PS 그래픽 불가(데이터 추출은 가능).
- NICK/IV 최신 dir에 tar.gz 동봉 — 압축본 아닌 풀린 dated dir을 가리킬 것.
- Co/Ni DR: CMFGEN은 Co IV→III을 명시 DIE(`DIECoIII_*`)로, S/Si/Fe는 공명-σ로 암묵 처리, Ni IV/V DR은 CMFGEN에도 없음("very crude" fit) — Lumina 대조 시 참조 (`docs/../memory: reference_cmfgen_source_available.md`).
