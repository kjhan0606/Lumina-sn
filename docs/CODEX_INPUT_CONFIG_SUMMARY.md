읽기 전용 지시에 따라 `docs/CODEX_INPUT_CONFIG.md`는 생성하지 않았다. 아래가 그 파일에 들어갈 보고서 본문이다.

# 입력축 B 감사 — capture 188932 / parity59

## -o 요약

| 항목 | 판정 | 한 줄 근거 |
|---|---|---|
| I6 모델 덱 | **잔류** | `t_exp`만 Δ=0. 공간 범위·속도·밀도·조성 배열이 다르며, 캡처는 발주서의 덱이 아니라 `_sivcaiv` 덱을 소비했다. |
| I7 격자 | **잔류** | ν: 1,000 대 196,185, 공간: 50 셸 대 90 depth, 각도: 58 대 105 rays. |
| I8 경계조건 | **잔류** | Lumina는 `Bν(10020 K)` 내부 BC, CMFGEN은 `DIFFUSION`; 내부 광도는 약 31.1배 차이. CMFGEN 외부 BC의 정확한 코드값은 **UNRESOLVED**. |
| I9 수치 상수 | **잔류** | Lumina 고유 ε clamp는 CMFGEN 대응물 없음. 외부 반복 12 대 4이며 damping·임계·반복 계약도 동일하지 않다. |
| 입력 열거 완전성 | **UNRESOLVED** | 파일 63개와 설정 환경 131개는 닫혔으나, 캡처 바이너리와 정확히 대응하는 소스 스냅샷이 없어 모든 하드코딩 상수의 전수성은 증명되지 않았다. |
| I1–I9 누락 | **잔류** | 초기 `n_e/W/T_rad/τ/transition-probability`, 원자 구조·macro-atom topology, 패킷/RNG, 연산자 게이트, 바이너리·실행환경, symlink 해석이 대장 밖이다. |

## 기준 런 식별

실제 argv는 다음이었다.

```text
./lumina_cuda.withParityAH \
  data/tardis_reference_toy06_19p48d_sivcaiv \
  100000 12 spectrum nlte
```

출처: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:134).

따라서 발주서에 적힌 `data/tardis_reference_toy06_19p48d/`는 기준 런의 소비 경로가 아니다. 실제 `LUMINA_MODEL_DIR`도 `_sivcaiv`이다: [PARITY59_INSTR.env](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:92), [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:22).

실행 바이너리:

```text
/gpfs/kjhan/lumina_runner2/lumina_cuda.withParityAH
size   = 3,986,176 bytes
mtime  = 2026-08-02 12:33:01 +0900
sha256 = bcb1292707d33d324763b0ca9132087fc5081416801b59b8a08389b5b312dc44
```

현재 `src/lumina_cuda.cu`, `src/lumina_cmfgen.c`, `src/lumina_plasma.c`는 이 바이너리보다 나중에 수정됐다. 아래 코드 줄은 값과 경로를 대조하는 보조 증거이며, 캡처 바이너리의 빌드 계보 증거는 아니다.

## I6 — 모델 덱

| 필드 | Lumina가 소비한 값과 출처 | CMFGEN 대응값과 출처 | 직접 대조 | 판정 |
|---|---|---|---|---|
| 시간 | `1,683,072 s = 19.48 d`; [config.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/config.json:2) | `19.4800000 d`; [SN_HYDRO_DATA](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/SN_HYDRO_DATA:7), [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:68) | Δ=`0 s` | **제거** |
| 공간·속도 구조 | 50 셸; `r=6.5639808e14…6.78278016e15 cm`, `v=3900…40300 km/s`; [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:2), [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:51) | 90 depth points; `r=1.7251e14…6.0549e15 cm`, `v=1024.971…35975.288 km/s`; [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:6), [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:13), [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:26) | inner `r` 3.805×, outer `r` 1.120×; velocity 끝점 Δ=`+2875.029`, `+4324.712 km/s` | **잔류** |
| 밀도 | `ρmax=1.5687693e-13`, `ρmin=6.9807609e-19 g cm⁻³`; [density.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/density.csv:2), [density.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/density.csv:51) | `ρmax=4.8021e-13`, `ρmin=2.7413e-18`; [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:221) | 범위 끝점 Lumina/CMFGEN=`0.3267`, `0.2547`; 좌표 범위도 다름 | **잔류** |
| 조성 | 6종×50 셸. 외곽 `(Si,S,Ca)=(0.55,0.35,0.10)`, 나머지 0; 내측 `(Ni,Co,Fe)=(0.1083233103,0.7937131696,0.0979635201)`; [abundances.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/abundances.csv:2) | 6종×700 입력점; [SN_HYDRO_DATA](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/SN_HYDRO_DATA:4). 외곽 Fe/Co/Ni=`1e-10`; 내측 `(Ni,Co,Fe)=(0.108320650,0.793714762,0.0979645878)`; [SN_HYDRO_DATA](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/SN_HYDRO_DATA:1318), [SN_HYDRO_DATA](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/SN_HYDRO_DATA:1437), [SN_HYDRO_DATA](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/SN_HYDRO_DATA:1556) | 배열 크기 50 대 700. 외곽 최대 절대차 `1e-10`; 내측 최대 절대차 `2.6603e-6`. 좌표 끝점은 서로 다름 | **잔류** |

I6 판정: **잔류**.

## I7 — 격자

| 필드 | Lumina | CMFGEN | 직접 대조 | 판정 |
|---|---|---|---|---|
| ν 빈 수 | `1000`; [lumina.h](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:507), 캡처 확인 [linepop manifest](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10.manifest.json:7) | `196185`; [CONT_FREQ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/CONT_FREQ:2), [MOD_SUM](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MOD_SUM:7) | Δ=`195185` | **잔류** |
| ν 범위 | nominal edges `1.5e14…3.0e16 Hz`; bin centers `1.50398e14…2.99206e16 Hz`; [lumina.h](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:508) | `3.49897e12…1.0e18 Hz`; [CONT_FREQ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/CONT_FREQ:4), [CONT_FREQ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/CONT_FREQ:196188) | CMFGEN이 저주파로 42.87×, 고주파로 33.33× 더 연장 | **잔류** |
| ν 간격 | 일정한 `Δlnν=ln(200)/1000=0.005298317`, 인접비 `1.005312`; [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1453) | 비균일; 예: `dV=200.13,124.08,524.65…km/s`; [CONT_FREQ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/CONT_FREQ:4) | 간격 규칙 다름 | **잔류** |
| 공간 격자 | 50 셸, 위 I6 경계 | 90 depth points, 위 I6 경계 | 개수 및 경계 다름 | **잔류** |
| 각도 | 8 core + 셸당 tangent 50 = 58 rays; core `p=r_in(k+0.5)/8`; [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1504) | 15 core, 105 total; [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:6) | total Δ=`47`, core Δ=`7` | **잔류** |
| CMFGEN 각도 node/weight 전열 | — | 해당 배열이 지정 파일에 없음 | 직접 배열 대조 불가 | **UNRESOLVED** |

CMFGEN 내부 기록에는 `RVTJ:NCF=166152`도 존재한다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:7). `CONT_FREQ`/`MOD_SUM`의 `196185`와 Δ=`30033`이다. 어느 쪽을 택해도 Lumina의 1000과는 다르다.

I7 판정: **잔류**.

## I8 — 경계조건

Lumina:

- 입력 `T_inner=10020 K`, 요청 광도 `3.0927255108e42 erg s⁻¹`: [config.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/config.json:3).
- 실제 결정론적 내부 BC는 `LUMINA_INNER_BB_SCALE=1`인 `Bν(T_inner)`: [PARITY59_INSTR.env](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:68), [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2560).
- 해당 반지름과 온도로 기록된 `L_inj=3.094761e42 erg s⁻¹`; 요청 광도 대비 `+0.0658%`: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:38113).
- 외부 입사 세기 `I=0`: [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2451).

CMFGEN:

- `LSTAR=2.60e7 Lsun`: [VADAT](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/VADAT:11), [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:58).
- 내부 BC는 `DIF=T`, `IB_METH=DIFFUSION`, continuum/line thick BC=`T`: [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:123), [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:134), [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:169).
- CMFGEN에 독립적인 `T_inner` 입력은 없다. 가장 안쪽 고정 온도는 `23021 K`지만 경계 색온도와 동일 필드가 아니다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:65).
- `Lsun=3.828e33 erg s⁻¹` 변환 시 `LSTAR≈9.9528e40 erg s⁻¹`; Lumina 요청 광도/CMFGEN=`31.074`.
- 지정 파일에는 CMFGEN 외부 입사 BC의 명시 필드가 없다: **UNRESOLVED**.

I8 판정: **잔류**.

## I9 — 수치 상수

| 묶음 | Lumina 유효값 | CMFGEN 대응 입력 | 판정 |
|---|---|---|---|
| 선 ε clamp | `eps_floor=1e-5`, `eps_cap=1.0`; 환경 override 미설정, 물리 ε 사용=`1`; [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:691), [PARITY59_INSTR.env](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:27). 판정런 189116에서 `31,353,733/37,586,850=83.4%` 행 변경; [정본](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/OUTSIDE_LOOP_POOL.md:32) | 대응 line-ε clamp 없음 | **잔류** |
| 외부 반복 | `PURE_CMFGEN_ITER=12`; [PARITY59_INSTR.env](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:111), 실제 12; [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:267) | `NUM_ITS=4`; [IN_ITS](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/IN_ITS:1) | Δ=`8`, Lumina/CMFGEN=`3` | **잔류** |
| 방사장 반복 | ALI 8회; `ALI_TOL=1e-3` 기본이나 tri-ALI가 꺼져 있어 조기 종료 미사용; [PARITY59_INSTR.env](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:24), [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2532) | `NUM_LAM=2`, `ACC_F=1e-4`; [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:324), [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:348) | 반복 Δ=`6`; 임계 적용 대상 다름 | **잔류** |
| damping | `J=0.5`, `Jbar=0.5`, `RADEQ=0.5`, `CN=0.5`, transition/W/Trad=`0.5`; [PARITY59_INSTR.env](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:41), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6876) | 동일한 `f` 필드 없음. `MAX_LIN=3`, `MAX_LAM=3`은 step cap; [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:312) | 대응물 없음/값 다름 | **잔류** |
| CE | `tol=1e-2`, damping=`1`, 최대 20회; [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1027), 캡처 확인 [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:38027) | 직접 대응 외부 CE sweep 입력 없음 | 대응물 없음 | **잔류** |
| 전체 수렴 | Lumina 외부 12회 고정; coupled-Newton 내부 임계 `relT,reln<1e-5`; [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:13988) | `EPS_TERM=0.1% = 1e-3`, `BA_CHK=1e-4`, `FIX_BA=0.05`; [VADAT](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/VADAT:629) | 수치 및 적용 범위 다름 | **잔류** |
| 온도 탐색 | `[3500,140000] K`, 24구간 탐색, 최대 45회 이분법, 폭 `2 K`, step clamp `[0.5,2]×Told`; [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10644), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10673), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8924) | `FIX_T=T`, hydro 온도 입력; [VADAT](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/VADAT:622) | 대응물 없음 | **잔류** |
| 패킷/상호작용 cap | packets=`100000`, interaction=`50000`, total steps=`2000000`, MA internal=`5000`; [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:259) | 결정론적 CMFGEN에 대응물 없음 | 대응물 없음 | **잔류** |

I9 판정: **잔류**.

## 입력 전집

### 1. 실제 소비 데이터 파일 — 63개

주 loader는 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:274), 원자자료 loader는 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:642), 추가 입력 분기는 [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6892)와 [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7737)에 있다.

고유 파일 19개:

```text
geometry.csv
config.json
electron_densities.csv
plasma_state.csv
density.csv
line_list.csv
tau_sobolev.npy
transition_probabilities.npy
macro_atom_references.csv
macro_atom_data.csv
line2macro_level_upper.npy
levels.csv
ionization_energies.csv
zeta_ions.csv
zeta_temps.csv
zeta_data.npy
atom_masses.csv
abundances.csv
level_multiplicity.csv
```

추가 파일 5개:

```text
cmfgen_sigma_bf.bin
ma_radrecomb_target.bin
feiii_col_zhang.bin
coldata_cmfgen_manifest.csv
deposition_cmfgen.csv
```

manifest가 실제 연 일반 충돌강도 파일 39개는 다음 `(Z,ion0)` 전집의
`ige_col_<Z>_<ion0>_cmfgen.bin`이다.

```text
(6,0) (6,1) (6,2)
(8,0) (8,1) (8,2)
(12,0) (12,1)
(13,0) (13,1) (13,2)
(14,0) (14,1) (14,2) (14,3)
(16,0) (16,1) (16,2) (16,3) (16,4)
(20,0) (20,1) (20,3)
(21,0) (21,1)
(22,1) (22,2)
(24,1) (24,2)
(26,0) (26,1) (26,3)
(27,1) (27,2) (27,3)
(28,0) (28,1) (28,2) (28,3)
```

`(26,2)`는 `feiii_col_zhang.bin`과 중복되어 열리지 않았다. 캡처 집계는 `40 OK → 39 loaded, 1 skipped`: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:245).

### 2. 경로·symlink 전집

실제 `_sivcaiv` 디렉터리는 단일 덱이 아니다.

- `geometry/config/density/abundances/plasma/deposition` 등은 비‑`sivcaiv` toy06 덱을 가리킨다.
- `levels/line_list/τ/macro-atom/zeta/ionization/mass`는 `tardis_reference_cmfgen_superlev_ionfix_ddc15strat_sivcaiv`를 가리킨다.
- `cmfgen_sigma_bf.bin`은 `data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin`을 가리킨다.
- 39개 일반 충돌강도 표와 `ma_radrecomb_target.bin`은 `_sivcaiv` 디렉터리의 실파일이다.

따라서 symlink target과 그 바이트도 입력이다.

### 3. 환경·argv 전집

- 바이너리가 본 환경변수: 정확히 129개. 전집은 [PARITY59_INSTR.env:15](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:15)부터 134 및 138–146의 모든 `export` 행이다. 캡처도 `129 vars`를 기록했다: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:4).
- 바이너리 dump에 없는 OpenMP 입력: `OMP_PLACES=cores`, `OMP_PROC_BIND=close`; [PARITY59_INSTR.env](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:10).
- 합계: 설정 환경 131개.
- `CHIETA_EXPECTED_ITER`, `CHIETA_ORACLE_CELLS`, `RUN_DIR`은 launcher 측 입력이다. 바이너리에는 확장된 출력 경로와 iteration `10`이 전달됐다.
- 그 밖의 `LUMINA_*` 변수가 미설정이라는 사실도 입력 조건이다.
- argv 입력은 덱 경로, `100000`, `12`, `spectrum`, `nlte`의 5개다.
- 출력 대상이 사전에 존재하지 않는다는 파일시스템 상태도 overwrite가 꺼진 dump 분기의 입력 조건이다.

### 4. 코드 고정 입력 — 확인된 범위

- 물리상수: `c`, `σT`, `h`, `kB`, `σSB`, π, Sobolev 계수, eV 변환; [lumina.h](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:17).
- CMF 주파수 grid `1000`, `1.5e14`, `3e16`; [lumina.h](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:507).
- core rays `8`; [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1504).
- ε clamp `1e-5…1`; [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:691).
- ALI 기본 임계 `1e-3`, 최소 pass `1`; [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2532).
- config damping `0.5`, hold `3`; [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6876).
- CE 임계·damping·cap; [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1027).
- packet/step/MA cap 기본값; [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7106).
- 온도 bracket·탐색수·이분법 임계·step clamp; [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10644).
- ARTIS radiation estimator의 24 coarse bins는 실제 캡처에 기록됨: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:342).

캡처 바이너리에 들어간 모든 literal과 위 현재 소스 literal의 동일성은 **UNRESOLVED**.

## I1–I9를 초과한 누락 항목

다음은 현재 I1–I9 명칭에 포함되지 않은 독립 입력이다.

1. **초기 상태**
   - `electron_densities.csv`
   - `plasma_state.csv`의 `W`, `T_rad`
   - 초기 `tau_sobolev.npy`
   - 초기 `transition_probabilities.npy`
   - `T_e/T_rad` seed와 `TRAD_COLOR_FIX`

2. **원자 구조·line identity**
   - level energy, `g`, metastable
   - ionization energy
   - atomic mass
   - `f_lu`, `B_lu`, `B_ul`, line ν·wavelength
   - line의 Z/ion/lower/upper 연결

3. **macro-atom topology**
   - block references
   - transition type
   - destination level
   - line-to-macro-level mapping
   - recombination target mapping

4. **통계·실행 계약**
   - 패킷 수 `100000`
   - RNG seed `23111963`
   - line interaction mode `macroatom`
   - NLTE 시작 iteration `2`
   - OpenMP threads/placement
   - GPU solver 선택과 실행 바이너리

5. **연산자·물리 게이트**
   - 129개 환경 설정 전집과 나머지 gate의 미설정 상태
   - `BF/FF`, deposition injection, coevolve, J/Jbar 소비 모드, macro-atom/k-packet 선택 등

6. **경로 계보**
   - 발주 덱과 실제 `_sivcaiv` 덱의 차이
   - symlink target 전집
   - 캡처 시점의 파일 바이트/digest

7. **코드·빌드 계보**
   - 실제 source commit
   - compile flags, CUDA/toolchain, linked-library 버전
   - 하드코딩 상수의 바이너리별 전집

`γ-deposition`은 I1–I9 밖이지만 정본에는 이미 I10으로 등재돼 있으므로 새 누락은 아니다.

## UNRESOLVED와 결판 요건

| 대상 | 판정 | 결판 요건 |
|---|---|---|
| 캡처 바이너리의 모든 하드코딩 상수 | **UNRESOLVED** | SHA `bcb129…c44`를 만든 정확한 source tree/commit, compile command와 flags, 또는 바이너리에서 생성한 상수·gate manifest |
| 캡처 시점 63개 입력의 바이트 동일성 | **UNRESOLVED** | 캡처 시점 SHA-256 목록 또는 immutable snapshot; symlink 자체와 해석된 target 모두 필요 |
| CMFGEN 외부 BC | **UNRESOLVED** | 해당 CMFGEN 빌드의 outer-boundary transfer routine 또는 런타임 BC trace |
| CMFGEN 각도 node/weight 전열 | **UNRESOLVED** | 105-ray impact parameter와 quadrature weight dump 또는 생성 코드 |
| CMFGEN 광도의 정확한 erg/s 비 | **UNRESOLVED** | 해당 CMFGEN 빌드가 사용한 `Lsun` 상수. 통상값 기준 비는 약 `31.074`이며 판정은 바뀌지 않는다. |

입력 열거 완전성의 최종 판정은 **UNRESOLVED**다. 파일 경로·활성 환경·argv는 닫혔지만, 캡처 바이너리의 정확한 소스 계보와 캡처 시점 입력 파일 digest가 없으므로 “모든 파일·모든 gate·모든 hardcode”의 완전 제거 조건을 충족하지 못한다.