읽기 전용 규율에 따라 `docs/CODEX_POOL_ENUM.md`는 생성하지 않았습니다. 아래가 그 파일에 들어갈 전체 보고 내용입니다. 수정·commit·모델 실행·GPU 사용은 하지 않았습니다.

## `-o` 요약

- **I11 — 잔류:** Lumina는 50-shell 초기 `n_e/W/T_rad/τ/transition probability`를 읽으며, CMFGEN에는 `W·T_rad·MC transition probability` 동형 입력이 없다.
- **I12 — 잔류:** Lumina의 26,592-level/7,752,396-edge macro-atom topology에 대한 CMFGEN 대응물은 없다. 원자·superlevel 구조 수도 다르다.
- **I13 — 잔류:** Lumina는 100,000 packets, seed `23111963`, SplitMix64+xoshiro256**를 사용한다. CMFGEN에는 packets/RNG 대응물이 없다.
- **I14 — UNRESOLVED:** 설정된 129+2개 환경값은 닫혔지만, 미설정 gate의 바이너리 기본값 전집은 소스 스냅샷 부재로 닫히지 않는다. 확인된 연산자 설정만으로도 CMFGEN과 동일하지 않다.
- **I15 — UNRESOLVED:** 캡처 ELF 해시는 확정됐지만 build-id·DWARF·빌드 로그·소스 스냅샷이 없어 정확한 소스 상태를 특정할 수 없다.
- **I16 — 잔류:** Lumina 41개와 CMFGEN 126개 symlink는 모두 해석되지만, 두 실행이 소비한 실제 target 집합은 서로 다르다.
- **전집:** 63개 모델 파일 + argv/config + 설정 환경 131개 + 바이너리/동적 실행환경 + symlink target + 하드코딩 기본값·수치/물리 상수.
- **I1–I16 초과 확정 항목:** 없음. 다만 “없음”을 완결 증명할 수는 없다. 캡처 바이너리의 미설정 gate 기본값과 하드코딩 상수 전집이 `UNRESOLVED`다.
- **전체 판정:** **열거 완전성 `UNRESOLVED`**.

# I11 — 초기 상태

### Lumina 실제 소비값

| 입력 | 실제 값 | 출처/소비 지점 |
|---|---|---|
| `n_e` | 50개. shell 0 `1.6124599403931708e9`, shell 49 `1.387104996744542e4 cm⁻³` | [electron_densities.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/electron_densities.csv:2), 현 트리 소비 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:358), 캡처 확인 [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:143) |
| `W` | 50개. `0.2978587261676735 → 0.0023898974006010265` | [plasma_state.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/plasma_state.csv:2), 소비 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:364) |
| `T_rad` | 파일값 `10470.09324 → 3133.59439 K`; 그러나 `LUMINA_TRAD_COLOR_FIX=1`이므로 실제 초기 배열은 50개 모두 `10470.093240032314 K` | 변환 코드 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:377), 캡처 [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:144) |
| 초기 `T_e` | 초기화 시 위 `T_rad`와 `T_e/T_rad=1.0`에서 생성 | 초기화 [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7056), 환경값 [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:38124) |
| `τ` | `(2,584,132, 50)`, float64, 전 원소 0 (`129,206,600` zeros) | 로드 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:465), 캡처 shape [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:148) |
| transition probability | `(7,752,396, 50)`, float64, min `8.532093070253922e-35`, max `1`, zero/non-finite 0 | 로드 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:482), 캡처 [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:149) |

캡처는 `DYNAMIC_TRANSPROB=1`이었다. 현 트리에서는 각 coevolve iteration에서 확률을 다시 계산하고 GPU에 올린다: [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8409). 따라서 파일 확률은 읽히는 초기 입력이지만 transport 직전 갱신 경로가 존재한다. 캡처 바이너리에도 `compute_transition_probabilities` 심볼이 있다.

### CMFGEN 대응물

CMFGEN에는 90-depth `n_e`, 온도, `J`가 있다. 현재 `RVTJ`에는:

- `n_e`: `2.5324408e4 … 1.8325991e10 cm⁻³`, [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:52)
- 온도: `1.6497639e4 … 2.3021000e4 K`, [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:65)
- `J` moment: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:156)

그러나 이 `RVTJ`는 실행 완료 시 덮어쓴 결과이며, jnu4 시작 record 62의 별도 immutable snapshot은 확인되지 않았다.

- `W`: 대응물 없음.
- `T_rad`라는 dilute-blackbody 상태 변수: 대응물 없음.
- Lumina `tau_sobolev.npy` 입력 배열: 대응물 없음.
- MC macro-atom transition-probability 배열: 대응물 없음.

**대조:** 공간 수부터 50 대 90이며 동형 배열이 아니다.  
**판정: 잔류.**

# I12 — 원자 구조와 macro-atom topology

### Lumina

캡처가 실제 로드한 구조:

- levels: `26,592`
- bound-bound lines: `2,584,132`
- macro levels: `26,592`
- macro transitions: `7,752,396`
- transition type `-1/0/+1`: 각각 정확히 `2,584,132`
- `line2macro_level_upper`: `2,584,132`, 범위 `1…26,591`
- superlevel: `21,581` levels lumped; NLTE 경로는 `21,038 FL → 2,828 SL`

근거는 [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:146), [macro_atom_data.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/macro_atom_data.csv:2), 소비 코드는 [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:495)다.

### CMFGEN

CMFGEN `MODEL`의 명시 이온 27행 합은:

- full levels `N_F=20,749`
- superlevels `N_S=1,637`
- 각 이온의 `F_TO_S` mapping 사용

실제 이온별 수는 [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:16), 설정은 [MODEL_SPEC](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL_SPEC:10)에 있다.

CMFGEN에는 `F_OSCDAT`, `F_TO_S`, rate-equation topology는 있지만 Lumina/TARDIS의 macro-atom activation/deactivation edge graph와 transition-probability table은 **없다**.

**대조:** 원자·superlevel cardinality가 다르고 macro-atom topology는 대응물 자체가 없다. CMFGEN-only/Lumina-only line support도 이 항목의 atomic support 차이에 포함된다.  
**판정: 잔류.**

# I13 — packet 수와 RNG

### Lumina

- config 기본값: packets `200000`, iterations `20`, seed `23111963`, [config.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/config.json:7)
- argv override: packets `100000`, iterations `12`, [sbatch_instr_capture.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_instr_capture.sh:65)
- 캡처 실제값: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:267)
- RNG: SplitMix64로 seed 확장 후 xoshiro256**, [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3278)
- packet seed: `SplitMix64(base_seed XOR p*0x9e3779b97f4a7c15)`, [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5969)
- iteration seed: `23111963 + iteration*1,000,000`, [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8777)

### CMFGEN

해당 실행은 결정론적 CMF transfer/rate-equation 계산이다.

- packet 수: 대응물 없음.
- RNG 및 seed: 대응물 없음.
- 반복: `NUM_ITS=4`, `DO_LAM_IT=T`, [IN_ITS](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/IN_ITS:1)

**판정: 잔류.**

# I14 — 연산자 gate

캡처 footer가 기록한 설정값은 129개이며 [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:38123)–[종료 표식](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:38255)에 전부 열거돼 있다. 여기에 footer가 출력하지 않은 `OMP_PLACES=cores`, `OMP_PROC_BIND=close`가 더해져 명시 환경은 131개다.

루프 연산자에 직접 들어가는 확인값은 다음과 같다.

- 모드: `ARTIS_PARITY=1`, `PURE_CMFGEN=1`, `PURE_CMFGEN_ITER=12`, `MC_COEVOLVE=1`, `CONSUME=1`, `INJECT=2`, `NLTE=1`, `NLTE_START_ITER=2`, `LINE_INTERACTION=macroatom`.
- CMF: `ALI_ITER=8`, `FROZEN_ALI=60`, `FROZEN_CONT=1`, `FINE_ALI=20000`, `ADV_SPLIT=1`, `LINERES_JBAR=2`, `SOLVE_GPU=1`.
- 결합: `COUPLED_LAMBDA_STAR=1`, `JNU_PHOTOION=1`, `JNU_LSTAR=0`, `COUPLED_NEWTON=0`, `COUPLED_TDEP=1`.
- opacity/source: `BF_OPACITY=1`, `BF_RATE_POPS=1`, `CMF_BF_MILNE=2`, `CMF_DEP_SOURCE=1`, `CMF_EPAY=2`, `LINE_EPS_PHYS=1`.
- plasma/NLTE: `ION_LOCK=1`, `PER_ION_RESCALE=1`, `GREY_ITERS=2`, `GREY_TAU=2`, `FINAL_RESOLVE=1`, `LTE_FLOOR=0`, `FLOOR_REG=0`.
- macro-atom: `DYNAMIC_TRANSPROB=1`, `KPACKET=1`, `MA_RADRECOMB=1`, `MA_REAL_UPSILON=1`, `MA_LINE_DESTRUCT=1`, `MACROATOM_EWEIGHT=1`, `IDOWN_BETA=1`, `NEUTRAL_E=1`.
- damping/clamp: `J_DAMP=0.5`, `COEVOLVE_JBAR_DAMP=0.5`, `CN_DAMP=0.5`, `RADEQ_DAMP=0.5`, `TE_STEP_CLAMP=1`, `HRESP_CLAMP=1`.
- 경계/state: `DIFFUSE_INNER_BC=1`, `INNER_BB_SCALE=1`, `TRAD_COLOR_FIX=1`, `TE_TRAD_RATIO=1`, `SUPER_LEVELS=1`, `SUPER_CUTOFF=100`.
- explicit off: `COUPLED_NEWTON=0`, `RADEQ_LINE_RE=0`, `RADEQ_COOL_ESCAPE=0`, `RADEQ_COOL_NLTE_ONLY=0`, `RADEQ_COOL_NONNEG=0`, 네 `DR_BOOST_*=0`, `FROZENIN=0`, `FROZENIN_DR=0`, `NLTE_ASSEMBLE_GPU=0`.

Dump·trace·event-log 경로는 환경 전집에는 포함되지만 물리 연산자 전집에서는 분리했다.

CMFGEN 대응 설정은 `4`회의 pure Lambda iteration, fixed temperature, diffusion inner boundary다: [IN_ITS](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/IN_ITS:1), [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:123), [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/OUTGEN:142).

### 완결 실패

- 캡처 ELF 문자열에서 고유 `LUMINA_*` 토큰은 465개다.
- 설정 환경에 없던 토큰은 345개다.
- 현 트리는 고유 `getenv("…")` literal 461개지만 캡처 이후 변경됐다.
- 문자열은 메시지·진단명도 포함하므로 465개를 gate 전집으로 간주할 수 없다.
- 미설정 gate가 선택한 기본값을 전부 복원하려면 정확한 캡처 소스나 완전한 binary data-flow 역분석이 필요하다.

따라서 확인된 설정만으로 CMFGEN과 동일하지 않지만, I14 자체의 열거 완전성은 닫히지 않는다.  
**판정: UNRESOLVED.**

# I15 — 바이너리와 실행환경

### Lumina 캡처

- ELF: `/gpfs/kjhan/lumina_runner2/lumina_cuda.withParityAH`
- SHA-256: `bcb1292707d33d324763b0ca9132087fc5081416801b59b8a08389b5b312dc44`
- size: `3,986,176`
- mtime: `2026-08-02 12:33:01.872074051 +0900`
- ELF64 x86-64, dynamically linked, not stripped
- GPU: `NVIDIA H200 NVL`, `143771 MiB`, host `syn104`, [slurm log](/gpfs/kjhan/lumina_runner2/slurm/instr_capture_188932.out:1)
- launcher module: CUDA `13.0.2`, [sbatch_instr_capture.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_instr_capture.sh:23)
- embedded CUDA compiler string: `release 13.0, V13.0.88`
- compiler strings: GCC `11.5.0`, `13.2.0`, `8.5.0`
- OMP: threads `16`, places `cores`, bind `close`

### CMFGEN 대응물

현재 경로의 executable은 `/gpfs/kjhan/cmfgen_src/cur_cmf/exe/cmfgen_dev.exe`, 현 SHA-256 `f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1`, x86-64 dynamically linked, debug info 포함이다. CMFGEN 실행 시점에 이 SHA를 기록한 증명 파일은 확인되지 않았다. 실행 설정은 `OMP_NUM_THREADS=16` 및 지정 core 집합으로 남아 있다: [run_jnu4.info](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/run_jnu4.info:1).

### 바이너리–소스 대응 가능성

현재 증거로는 **정확한 소스 상태를 특정할 수 없다**.

- `.symtab/.strtab`은 있어 함수 존재와 주소는 확인 가능하다.
- `src/lumina_cuda.cu`, 함수명, gate명, CUDA/GCC 문자열은 남아 있다.
- GNU build-id 없음.
- DWARF/debug line table 없음.
- git commit/hash 문자열 없음.
- 컴파일 명령이나 빌드 로그 없음.
- 배포 경로에 해당 ELF의 소스 snapshot 없음.
- 현재 `lumina_cuda.cu`, `lumina_plasma.c`, `lumina.h`, `Makefile`은 바이너리 mtime 이후인 2026-08-03에 수정됐다.
- 현재 git HEAD는 `47bfa200…`이지만 worktree는 광범위하게 수정돼 있어 HEAD가 캡처 소스를 나타내지 않는다.

심볼·문자열·disassembly는 후보 구현을 제한할 뿐, 동일 심볼과 문자열을 갖는 여러 소스 상태를 구별하지 못한다. 외부 build log/source archive가 새로 발견되지 않는 한 이 캡처에 대해서는 항구적 제약이다.

정확한 CUDA driver, 커널/OS, 실행 당시 DSO 해시도 캡처되지 않았다. 현재 `ldd` 결과는 실행 당시 증거가 아니다.

**판정: UNRESOLVED.**

# I16 — symlink 해석

### Lumina

캡처 경로:

- `lumina_cuda.withParityAH → /gpfs/kjhan/lumina_runner2/lumina_cuda.withParityAH`
- `data → /gpfs/kjhan/lumina_runner2/data → /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data`
- 최종 model dir: `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv`

model dir의 symlink 41개는 모두 해석되며 dangling link는 0개다. 최종 target 분포:

- 28개: `data/tardis_reference_toy06_19p48d`
- 12개: `data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat_sivcaiv`
- 1개: `data/atomic`

즉 `_sivcaiv` 디렉터리는 단일 copied snapshot이 아니라 세 target 계열의 composite view다.

### CMFGEN

`toy06_19.48d_jnu4`의 top-level symlink는 126개, dangling 0개다. 주 target은 `/gpfs/kjhan/cmfgen_21jun23/atomic/...`의 `F_OSCDAT`, `F_TO_S`, collision/photoionization 자료와 `/gpfs/kjhan/cmfgen_src/cur_cmf/txt_files/...`다.

**대조:** 양쪽 모두 해석은 성공하지만 실제 target 집합은 동일하지 않다.  
**판정: 잔류.**

# 전집 증명

## 1. 루프 밖 external file 전집: 63개

고정 24개:

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
cmfgen_sigma_bf.bin
ma_radrecomb_target.bin
feiii_col_zhang.bin
coldata_cmfgen_manifest.csv
deposition_cmfgen.csv
```

manifest가 지정하고 실제 로드한 generic collision table 39개:

```text
ige_col_6_0_cmfgen.bin   ige_col_6_1_cmfgen.bin   ige_col_6_2_cmfgen.bin
ige_col_8_0_cmfgen.bin   ige_col_8_1_cmfgen.bin   ige_col_8_2_cmfgen.bin
ige_col_12_0_cmfgen.bin  ige_col_12_1_cmfgen.bin
ige_col_13_0_cmfgen.bin  ige_col_13_1_cmfgen.bin  ige_col_13_2_cmfgen.bin
ige_col_14_0_cmfgen.bin  ige_col_14_1_cmfgen.bin  ige_col_14_2_cmfgen.bin
ige_col_14_3_cmfgen.bin
ige_col_16_0_cmfgen.bin  ige_col_16_1_cmfgen.bin  ige_col_16_2_cmfgen.bin
ige_col_16_3_cmfgen.bin  ige_col_16_4_cmfgen.bin
ige_col_20_0_cmfgen.bin  ige_col_20_1_cmfgen.bin  ige_col_20_3_cmfgen.bin
ige_col_21_0_cmfgen.bin  ige_col_21_1_cmfgen.bin
ige_col_22_1_cmfgen.bin  ige_col_22_2_cmfgen.bin
ige_col_24_1_cmfgen.bin  ige_col_24_2_cmfgen.bin
ige_col_26_0_cmfgen.bin  ige_col_26_1_cmfgen.bin  ige_col_26_3_cmfgen.bin
ige_col_27_1_cmfgen.bin  ige_col_27_2_cmfgen.bin  ige_col_27_3_cmfgen.bin
ige_col_28_0_cmfgen.bin  ige_col_28_1_cmfgen.bin  ige_col_28_2_cmfgen.bin
ige_col_28_3_cmfgen.bin
```

Fe III generic 행은 `feiii_col_zhang.bin`과 중복이라 skip됐다. 실제 로그도 `40 OK → 39 loaded, 1 duplicate skip`을 기록한다: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:255).

## 2. 나머지 leaf input 전집

- argv: model path, `100000`, `12`, `spectrum`, `nlte`
- 설정 환경: footer 129개 + `OMP_PLACES`, `OMP_PROC_BIND`
- executable와 ELF interpreter/DSO/CUDA driver/GPU/OpenMP runtime
- 위 63개 파일의 최종 symlink target
- 하드코딩 기본값·수치 상수·물리 상수

현 트리에서 확인되는 하드코딩 하한은 다음과 같다.

- 물리상수: `c`, `σ_T`, `h`, `k_B`, `σ_SB`, `π`, Sobolev coefficient, eV/amu/electron mass, [lumina.h](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:19)
- 주파수 격자: `1000`, `1.5e14…3.0e16 Hz`, [lumina.h](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:507)
- spectrum 기본값: `500…20000 Å`, 2000 bins, [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7436)
- CUDA block: 256 threads
- transport 기본 caps: interactions `100000`, total steps `2000000`, macro internal `5000`; 캡처는 interactions만 `50000`으로 override
- CMF line epsilon 기본값: floor `1e-5`, cap `1.0`, [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:691)
- RNG mixing/rotation constants
- collision/recombination/DR/gamma-decay coefficient tables
- ion/stage layouts, fixed table capacities 및 convergence tolerances

이 목록은 **현 소스 기준 하한**이며 캡처 바이너리의 전집 증명은 아니다.

## 3. I1–I16 초과 항목

확인된 63개 파일과 명시 환경을 의미별로 다시 분류하면 모두 기존 항목에 들어간다.

- collision tables → I1
- `A_ul`/line list → I2/I12
- photoionization → I3
- superlevels → I4/I12
- ionization/recombination/zeta/target tables → I5
- geometry/density/composition/time/config → I6
- grids → I7/I9
- boundary/luminosity → I8
- constants/defaults → I9
- deposition → I10
- initial arrays → I11
- topology → I12
- packet/RNG → I13
- gates → I14
- executable/runtime → I15
- path resolution → I16

따라서 **새로운 의미 범주로 확정된 I17은 없다**. CMFGEN-only line support는 별도 외부 입력 범주가 아니라 I12의 atomic-support 차이에 포함된다.

하지만 다음 두 집합이 닫히지 않아 “초과분 없음”은 증명되지 않았다.

1. 캡처 바이너리의 미설정 환경 gate가 선택한 기본값 전집.
2. 캡처 바이너리가 실제 포함한 모든 loop-consumed 하드코딩 상수와 compile-time table 전집.

# 최종 판정

파일 63개, 설정 환경 131개, symlink target, I11–I13의 주요 값은 열거됐다. 그러나 캡처 ELF에 대응하는 정확한 소스 상태가 없으므로 미설정 gate의 기본값과 하드코딩 상수에 대한 전수성은 증명할 수 없다.

**입력 열거 완전성: `UNRESOLVED`.**