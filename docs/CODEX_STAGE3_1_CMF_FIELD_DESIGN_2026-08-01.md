# Codex A' — Stage 3.1 결정론 CMF 장 상세 설계 명세 초안

상태: **DRAFT / 구현 전 설계**  
작성 기준: 2026-08-01 차터 §2, `CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md` Stage 3  
이번 산출물의 변경 범위: 이 문서만. `src/`, `scripts/`, `Makefile`은 수정하지 않았다.

## 0. 결정 요약과 범위 경계

Stage 3.1의 CPU 정본은 구면 대칭 homologous flow에서 impact-parameter ray를 따라 푸는 **blue-to-red sequential-frequency CMF formal solver**로 한다. 공간 적분은 선형 short-characteristics(SC), 주파수 미분은 균일 `ln nu` 격자의 2차 후방차분(BDF2), 각도 적분은 각 평가 반지름의 양의 반구 Gauss-Legendre `mu` 절점을 p-ray로 변환하는 방법을 채택한다. 모든 상태와 산술은 `double`이다.

3.1이 제공하는 것은 다음뿐이다.

1. 주어진 `chi_nu`, `eta_nu`의 formal solution과 `J_nu`(선택 진단으로 `H_nu`, `K_nu`).
2. coherent isotropic scattering KA를 위한 가속 없는 Lambda fixed-point 반복.
3. pure absorption, coherent scattering, homologous redshift KA와 3-grid Richardson 보고.
4. 입력이 확보될 때 parity59 frozen field 및 CMFGEN `jnu4`와의 오프라인 판별 벤치.

다음은 3.1 범위 밖이다.

- VEF(moment equation으로 `J`를 보정), ALI/MALI, Krylov/preconditioner: Stage 3.2.
- electron redistribution의 실제 kernel: 3.2. 3.1에는 API enum과 `UNSUPPORTED` 반환 자리만 둔다.
- producer/consumer 배선 및 기존 plasma/transport 코드 수정: Wave-3.2 머지 뒤 Stage 3.2+.
- GPU/OMP 권위화: CPU 단일 스레드 정본을 통과한 뒤.

이 경계는 차터의 “3.1은 formal solver+J, VEF/ALI는 3.2”와 로드맵의 CPU-first 및 frequency-coupled 요구를 동시에 만족한다(`docs/STAGE3_CMF_FIELD_CHARTER_2026-08-01.md:20-29`, `docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:200-214`).

## 1. 실물 조사 결과와 설계 근거

### 1.1 CMFGEN 원본에서 확인한 방법

아래는 `/gpfs/kjhan/cmfgen_src/cur_cmf/` 실물에서 확인한 사항이다.

- `plane/fg_j_cmf_v13.f:162-209`는 한 주파수에서 CMF Eddington factors를 구하며, 1차 주파수 차분과 이전의 더 푸른 주파수장을 명시한다. 입력은 `ETA`, `CHI`, `ESEC`, 속도/반지름, p-ray와 `dLOG_NU`이다(`:203-209`, `:229-260`).
- 같은 파일은 새 주파수가 반드시 감소해야 한다고 검사하고 이전 주파수 intensity를 넘긴다(`:991-1011`). 즉 blue-to-red가 구현상의 선택이 아니라 expanding monotonic flow의 인과 순서다.
- homologous case에서 ray advection 계수는 `GAM = (v/cr)[1 + sigma mu^2]`이고 `sigma=dln(v)/dln(r)-1`; 따라서 `sigma=0`, `v/r=1/t_exp`이면 모든 ray에서 `GAM=1/(c t_exp)`이다(`plane/fg_j_cmf_v13.f:946-975`).
- `plane/rel_variables.f:13-38,73-82`가 formal equation의 실제 이산 변수를 직접 적고 있다. `chi_prime=chi+3b`, `alpha=(nu/dnu)b`, `chi_tau=chi_prime+alpha`, `source_prime=(eta+alpha I_prev)/chi_tau`이다.
- CMFGEN SC 구현은 Olson-Kunasz/Hauschildt 방법이라고 명시하며 outer incoming intensity를 0으로 둔다(`plane/solve_cmf_formal_v3.f:9-20,83-112`). 작은 optical depth에는 급수식을 쓰고, 선형/포물 source 보간으로 inward/outward sweep을 한다(`:120-169`, `:242-280`).
- inner boundary 선택은 `ZERO_FLUX`, `HOLLOW`, `GRAY`, `DIFFUSION`이며 diffusion은 `I_plus=B_nu+mu dB/dtau`이다(`plane/solve_cmf_formal_v3.f:17-27,216-227`). FG 쪽도 outer incident를 0으로 명시한다(`plane/fg_j_cmf_v13.f:1119-1126`).
- production orchestration은 coherent scattering emissivity를 포함해 `FG_J_CMF_V13`으로 Eddington factors를 계산한 뒤, 그 factors로 `MOM_J_CMF_V11`의 moment solution을 갱신한다(`/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f:600-660,741-769`). `MOM_J_CMF_V11` 자체도 이전 blue frequency 의존과 coherent-scattering 처리를 명시한다(`/gpfs/kjhan/cmfgen_src/cur_cmf/plane/mom_j_cmf_v11.f:90-121,164-177`).

따라서 Stage 3.1의 대응 관계는 다음과 같다.

| CMFGEN | Stage 3.1 | 이후 단계 |
|---|---|---|
| `FG_J_CMF_V13` p-ray formal solve, blue-frequency 저장 | p-ray SC + sequential `ln nu` sweep | 유지 |
| `FG_J_CMF_V13`의 1차 `nu` upwind | 2차 BDF2로 승격 | 필요 시 비균일 격자 일반화 |
| `FG`가 Eddington factors 산출 | `J/H/K` 진단 적분만 | Stage 3.2 VEF 입력 |
| `MOM_J_CMF_V11` moment solve와 FG/MOM 반복 | 없음 | Stage 3.2 |
| coherent `ESEC*J` | 가속 없는 source fixed point | Stage 3.2 ALI/MALI |
| redistribution emissivity | enum/API 자리만 | Stage 3.2 kernel |

즉 3.1은 CMFGEN의 formal half를 독립 검증 가능한 CPU kernel로 만드는 증분이지, CMFGEN의 FG/MOM/VEF 전체 반복을 축약해 동등하다고 주장하는 증분이 아니다.

### 1.2 저장소 기존 구현의 재사용 판정

현재 저장소에는 이미 서로 다른 목적의 formal/CMF 코드가 있다.

- `src/lumina_cmfgen.h:29-84`의 `CMFGENState`는 1000-bin `chi_es`, `chi_abs`, `chi_line`, `chi_tot`, `S_fixed`, `J`와 p-ray를 보유한다. `src/lumina_cmfgen.c:58-105`는 50 shell + 8 core ray와 균일 log-frequency 중심을 만든다.
- 정적 binned solver의 `formal_solve_bin`은 p-ray inward/outward SC 및 inner `B_nu`, outer zero를 쓴다(`src/lumina_cmfgen.c:540-625`). 그러나 음의 `dtau`를 0으로 바꾸는 clamp가 있고(`:607-610`, `:631-635`), 주파수 비결합 binned formal이다.
- frequency-coupled `cmf_solve_J`는 blue-to-red outer loop와 tangent-ray SC를 이미 갖는다(`src/lumina_cmfgen.c:1533-1547,1661-1674`). 그러나 1차 upwind이고, 선택 경로가 capped operator-split advection을 사용하며(`:1554-1566,1694-1713`), 고정 횟수 scattering-source 반복과 결합돼 있다. 기존 코드 주석 자체가 capped pass를 임시 수정으로 설명한다.
- fine producer는 log-frequency fine mesh에서 continuum와 Gaussian line profile을 조립한 뒤 이 solver를 호출한다(`src/lumina_cmfgen.c:2111-2183,2419-2459`). 하지만 source/opacity clamp와 환경변수 기반의 실험 분기가 섞여 있고, 완성된 `chi,eta`를 parity59 산출물로 내보내지 않는다.
- observer spectrum formal integrator는 p-ray emergent luminosity용이며(`src/lumina.h:1284-1301`), shell별 `J_nu` 권위 producer나 sequential-frequency CMF solve가 아니다.
- `lumina_fi_impact_ray`는 photospheric core와 외부 annulus를 정확히 타일하는 유용한 quadrature 아이디어를 담지만 emergent disk integral용 midpoint p-grid이다(`src/lumina.h:1245-1282`). shell별 `J`의 `mu` quadrature로 그대로 쓸 수 없다.

판정은 다음과 같다.

- **개념/공식 재사용:** `Geometry`의 cgs 반지름, log-frequency grid convention, p-z 교차 계산, `expm1` 기반 small-`tau` 산술, 기존 CMF state의 opacity split 명칭.
- **코드 직접 재사용 안 함:** `formal_solve_bin`, static `cmf_solve_J`, fine producer 내부 kernel. 이유는 static visibility, source-iteration/실험 분기 결합, 1차/capped advection, clamp 규율 위반, 검증 가능한 독립 API 부재다.
- **Stage 3.2 adapter 후보:** 새 solver input을 `CMFGENState`로부터 채우는 얇은 adapter. 3.1에는 넣지 않는다.

## 2. 연속 방정식과 선택한 이산화

### 2.1 Stage 3.1이 푸는 방정식

`O(v/c)` 구면 CMF transfer에서 homologous flow `v=r/t_exp`를 넣으면, 한 straight p-ray의 signed path coordinate `s`에 대해

```text
d I_nu / ds - a d I_nu / d ln(nu) + 3 a I_nu = eta_nu - chi_nu I_nu,
a = 1 / (c t_exp)  [cm^-1].                                      (1)
```

`I_nu`, `J_nu`, `eta_nu`는 comoving-frame 양이다. `chi_nu`는 total extinction `[cm^-1]`, `eta_nu`는 total true/frozen emissivity `[erg s^-1 cm^-3 Hz^-1 sr^-1]`이다. Equation (1)의 `3a`와 이전-frequency source는 CMFGEN `rel_variables.f:23-38,75-81`의 `chi+3b` 및 `alpha I_prev`와 일치한다.

일반 monotonic `v(r)`의 angle-dependent `b(r,mu)`는 3.1 API에 넣지 않는다. v1은 `r`와 `t_exp`로 `v(r)=r/t_exp`를 정의하는 homologous 전용 solver다. frozen-file loader는 별도 geometry CSV의 `v_inner/v_outer`와 `r_inner/r_outer/t_exp` identity를 relative `1e-12`로 확인하고 불일치하면 `LCMF_EHOMOLOGY`로 실패한다. 일반 velocity array를 조용히 homologous로 투영하지 않는다.

### 2.2 주파수 격자와 BDF2

`x=ln(nu)`를 쓰고 배열 순서는 `k=0`이 가장 푸른 주파수, 이후 `nu[k-1] > nu[k]`로 고정한다. 균일 `ln nu` 격자를 **권장 수준이 아니라 v1 필수 계약**으로 한다.

```text
Delta x = ln(nu[k-1]/nu[k]) > 0, constant to relative 1e-12.
```

이 선택의 이유는 다음과 같다.

1. homologous Doppler shift는 `Delta ln nu`가 거리와 선형이라 advection coefficient가 모든 주파수에서 같다.
2. parity59 C2가 이미 1000-bin log grid `1.5e14--3.0e16 Hz`를 사용한다(`scripts/w3_gamma_triple_compare.py:183-195`).
3. 선 폭/적색이동을 상대 해상도로 제어할 수 있고 Richardson의 `h/2`가 명확하다.
4. 비균일 BDF2 계수와 positivity 분석을 3.1에서 동시에 도입하지 않는다.

`k>=2`에서

```text
dI/dx |_k = (-3 I_k + 4 I_{k-1} - I_{k-2}) / (2 Delta x) + O(Delta x^2).
```

따라서 한 주파수 plane은 다음 static-looking formal equation이 된다.

```text
dI_k/ds + chi_eff,k I_k = eta_eff,k,

chi_eff,k = chi_k + 3a + 3a/(2 Delta x),
eta_eff,k = eta_k + (2a/Delta x) I_{k-1} - (a/(2 Delta x)) I_{k-2}. (2)
```

- `k=0`: CMFGEN의 `INIT`와 같이 frequency coupling을 끄고 static formal equation을 푼다. 이것이 유한 주파수 영역의 blue inflow plane 정의다.
- `k=1`: 1차 implicit upwind `chi_eff=chi+3a+a/Delta x`, `eta_eff=eta+(a/Delta x)I_0`로 한 번 bootstrap한다. 정확한 `k=0`에서 한 step의 오차는 `O(Delta x^2)`이므로 이후 BDF2의 전역 2차를 훼손하지 않는다.
- `k>=2`: Equation (2).

BDF2의 `-I_{k-2}` 때문에 `eta_eff` 또는 해가 음수가 될 수 있다. 이를 0으로 자르지 않는다. 발생 시 해당 `(ray,segment,k)`와 세 항을 기록하고 `LCMF_ENEGATIVE`로 종료하여 **UNRESOLVED**로 판정한다. 향후 positivity-preserving 2차 scheme은 별도 설계 변경 없이는 대체할 수 없다.

### 2.3 공간 격자와 short-characteristics

입력 radial mesh는 연속 shell interface `r_edge[0..Nr]`와 shell-center `r_ctr[i]`를 갖는다. `chi`, `eta`는 `[Nr][Nnu]` shell-center 값이다. 각 p-ray에서 모든 교차 shell center와 physical boundary를 z-node로 만들고

```text
z = +/-sqrt(r^2-p^2),  ds = |z_d-z_u|.
```

node의 `chi_eff`와 `eta_eff`는 인접 radial center 사이의 **선형 r 보간**으로 얻는다. 양 끝은 가장 가까운 center에서 boundary까지 one-sided linear extrapolation하지 않고 center 값으로 반-cell constant extension한다. 이 boundary half-cell은 KA refinement에 포함되며, 필요하면 Stage 3.2에서 ghost-cell physical reconstruction으로 교체한다.

각 segment에서 `Q=eta_eff/chi_eff`, `Delta tau=0.5(chi_eff,u+chi_eff,d) ds`를 사용하고 선형 SC를 적용한다.

```text
I_d = I_u exp(-Delta tau) + psi_u Q_u + psi_d Q_d,
psi_d = 1 - (1-exp(-Delta tau))/Delta tau,
psi_u = (1-exp(-Delta tau)) - psi_d.                            (3)
```

`Delta tau -> 0`은 `expm1`과 해석 급수로 평가한다. 이는 상태값을 자르는 clamp가 아니라 동일 함수의 cancellation-free 평가다. `chi_eff=0`인 진공 segment는 별도 exact branch `I_d=I_u+0.5(eta_u+eta_d)ds`를 쓴다. `chi<0`, `eta<0`, non-finite input은 즉시 실패한다.

**SC를 long-characteristics보다 선택하는 이유:** 동일 ray를 boundary부터 끝까지 추적하되 각 이웃 node의 선형 source를 쓰면 비용이 `O(Nray Nr Nnu)`이고 향후 local Lambda/VEF 계수를 만들 수 있다. 각 evaluation point마다 boundary부터 재적분하는 naive LC는 중복 작업만 늘리고 local response coefficient를 주지 않는다. 누적형 LC라면 이 격자에서는 사실상 같은 segment recursion이므로 별도 LC 구현의 이득이 없다. CMFGEN 원본도 Olson-Kunasz SC를 사용한다(`solve_cmf_formal_v3.f:9-12`).

### 2.4 p-ray 및 각도 quadrature

기존의 “shell tangent + 소수 core ray” 한 벌을 모든 radius에서 재정규화하지 않는다. `J(r_i)`마다 양의 반구 `Nmu`-point Gauss-Legendre 절점 `(mu_m,w_m)`을 `[0,1]`에 만들고

```text
p_{i,m} = r_ctr[i] sqrt(1-mu_m^2).
```

인 ray를 생성한다. 같은 p가 여러 shell에서 중복돼도 v1은 합치지 않는다. 각 ray는 전 대기를 inward/outward로 한 번 풀고 자신을 요청한 `(i,m)` crossing의 두 intensity를 반환한다. 이 구성은 tangent singularity를 p 적분으로 직접 다루지 않으며 각 shell에서 정확한 Gauss rule을 보장한다.

```text
J_i,k = 1/2 sum_m w_m [I_plus(i,m,k)+I_minus(i,m,k)]
H_i,k = 1/2 sum_m w_m mu_m [I_plus-I_minus]      (diagnostic)
K_i,k = 1/2 sum_m w_m mu_m^2 [I_plus+I_minus]    (diagnostic).     (4)
```

여기서 `[0,1]` Gauss weight의 합은 1이다. ray 수는 `Nr*Nmu`, intensity history는 각 ray/node/direction에 최근 두 frequency plane만 둔다. `Nr=50`, `Nmu=16`이면 정본 벤치는 CPU 단일 스레드로 충분하며 deterministic reduction order는 `(i,m)` 사전순으로 고정한다.

## 3. 경계조건

### 3.1 outer boundary

모든 `p<r_edge[Nr]` ray의 inward leg 시작에서

```text
I_minus(r_outer,mu,nu) = I_outer_inc(mu,nu),
default I_outer_inc = 0.                                         (5)
```

로 둔다. production/benchmark에서는 무입사만 허용한다. KA의 manufactured boundary가 필요할 때만 callback으로 nonzero 값을 줄 수 있고 결과 metadata에 `outer_bc=manufactured`를 쓴다. CMFGEN의 outer zero와 대응한다(`fg_j_cmf_v13.f:1119-1126`, `solve_cmf_formal_v3.f:106-113`).

### 3.2 inner boundary와 turning ray

- `p < r_edge[0]`인 core ray는 inner surface에서 outward intensity를 정한다.
- `p >= r_edge[0]`인 tangent ray는 물질 내부의 turning point에서 inward intensity를 그대로 outward upstream으로 넘긴다. 인위적 반사 source를 더하지 않는다.

production inner BC enum은 다음 두 가지다.

1. `LCMF_BC_DIFFUSION`

   ```text
   I_plus(r_in,mu,nu) = B_nu(T_inner) + mu dB_nu/dtau_nu.          (6)
   ```

   입력 계약에 `B_inner[Nnu]`와 `dB_dtau_inner[Nnu]`를 요구한다. 값이 제공되지 않으면 `dB/dtau=0`으로 추정하지 않고 실패한다. 단, 명시적 `LCMF_BC_IRRADIATION`을 선택하면 아래 계약을 쓴다.

2. `LCMF_BC_IRRADIATION`

   ```text
   I_plus(r_in,mu,nu) = I_inner_inc(mu,nu),                        (7)
   ```

   callback 또는 `[Nmu][Nnu]` table을 받는다. blackbody backlight는 호출자가 `B_nu(T_inner)`를 채운 irradiation의 한 사례다.

Equation (6)은 CMFGEN `DIFFUSION`의 `B+mu dB/dtau`와 직접 대응한다(`solve_cmf_formal_v3.f:216-227`). `HOLLOW`, point-source, thick-outer extension은 Stage 3.1 production enum에 넣지 않는다.

## 4. C API 및 파일 구성

### 4.1 신규 파일

구현 증분은 다음 파일만 새로 만든다.

- `src/lumina_cmf_field.h`: 독립 입력/출력 struct, enum, public functions.
- `src/lumina_cmf_field.c`: grid 검증, p-ray builder, SC kernel, frequency sweep, plain scattering iteration, residual 및 CSV/binary reader.
- `scripts/stage31_cmf_ka_driver.c`: C solver를 호출하는 작은 fixture executable.
- `scripts/run_stage31_cmf_ka.py`: h/h2/h4 실행, exact/reference 평가, Richardson/acceptance JSON+Markdown 생성.
- `scripts/stage31_cmf_field_bench.py`: frozen input, candidate J, MC/CMFGEN field 및 Gamma 비교.

기존 파일은 수정하지 않는다. 유일한 예외는 구현 승인 시 `Makefile`에 standalone KA target 한 줄을 추가하는 것이다. production `SOURCES`에는 3.1에서 연결하지 않는다.

### 4.2 public data contract

헤더의 최소 형태는 아래와 같다. 명칭은 구현 시 고정하고 struct에 숨은 environment default를 두지 않는다.

```c
typedef double (*LCMFBoundaryFn)(void *ctx, double p_cm, double mu,
                                 double nu_hz);

typedef enum {
    LCMF_BC_DIFFUSION = 1,
    LCMF_BC_IRRADIATION = 2
} LCMFInnerBC;

typedef enum {
    LCMF_SCAT_NONE = 0,
    LCMF_SCAT_COHERENT = 1,
    LCMF_SCAT_REDISTRIBUTION = 2  /* 3.1 returns LCMF_EUNSUPPORTED */
} LCMFScatterMode;

typedef struct {
    size_t nr, nnu;
    const double *r_edge;       /* nr+1, cm, strict ascending */
    const double *nu;           /* nnu, Hz, strict descending, uniform ln nu */
    const double *chi_total;    /* nr*nnu, cm^-1 */
    const double *eta_fixed;    /* nr*nnu, cgs transfer emissivity */
    const double *chi_coherent; /* nr*nnu, cm^-1; may be NULL => zero */
    double t_exp_s;
    LCMFInnerBC inner_bc;
    const double *B_inner;
    const double *dB_dtau_inner;
    LCMFBoundaryFn inner_irradiation; /* required for LCMF_BC_IRRADIATION */
    LCMFBoundaryFn outer_irradiation; /* NULL in production => exactly zero */
    void *boundary_ctx;
} LCMFInput;

typedef struct {
    size_t n_mu;
    size_t max_source_iter;
    double source_rtol;
    int compute_hk;
} LCMFOptions;

typedef struct {
    double *J, *H, *K;          /* nr*nnu; H/K optional */
    double transport_resid_linf;
    double source_resid_linf;
    uint64_t clamp_count;       /* must stay exactly zero */
    uint64_t negative_count;    /* any nonzero => failure */
    size_t source_iterations;
} LCMFResult;

int lumina_cmf_field_solve(const LCMFInput *, const LCMFOptions *, LCMFResult *);
int lumina_cmf_field_residual(const LCMFInput *, const LCMFResult *, double *linf);
```

Index는 `q=i*nnu+k`, frequency descending이다. public API의 모든 floating scalar/array는 `double`; `float`, mixed precision, TF32, implicit cast buffer는 금지한다. allocator는 overflow-checked `size_t` 곱을 사용한다. global/static mutable solver state와 OpenMP pragma를 두지 않는다.

입력 validator는 `0 <= chi_coherent <= chi_total`, `chi_total>=0`, `eta_fixed>=0`, `t_exp_s>0`, 연속·증가하는 `r_edge`, 감소하는 `nu`를 전 cell에서 확인한다. 위반값을 보정하지 않고 첫 index와 원값을 오류 record에 남긴다.

### 4.3 coherent scattering in 3.1

`chi_coherent>0`이면 iteration `m`에서

```text
eta_total^(m) = eta_fixed + chi_coherent J^(m)                    (8)
```

를 만들어 전체 frequency sweep을 다시 한다. 초기값은 `J^0=0`; iteration은 고정된 순서로 수행하고 damping하지 않는다. 수렴 판정은

```text
max |J^(m+1)-J^m| / (J_ref + |J^(m+1)|) <= source_rtol,
J_ref = max(max|B_inner|, max_{chi_total>0}|eta_fixed/chi_total|)
        * DBL_EPSILON.
```

이다. `J_ref`는 진단 norm의 0분모 방지값이며 물리 상태를 변경하는 floor가 아니다. 기본 KA `source_rtol=1e-11`; production frozen-total-emissivity bench는 아래 §5.3에 따라 `chi_coherent=0`으로 실행한다. ALI/VEF가 없으므로 optically very thick, albedo~1 문제의 느린 수렴은 `LCMF_ENOCONV`로 정직하게 노출한다.

### 4.4 독립 transport residual

SC update 식 자체를 다시 대입한 machine-zero를 residual로 보고하지 않는다. 각 ray segment midpoint에서

```text
R = (I_d-I_u)/ds - a D_x I_mid + (chi_mid+3a)I_mid - eta_mid
Rhat = |R| / (|(I_d-I_u)/ds| + |a D_x I_mid| + |(chi_mid+3a)I_mid|
              + |eta_mid| + R_scale).                            (9)
```

를 계산한다. `D_x`는 solution용 BDF와 독립적인 centered/midpoint reconstruction, `R_scale=DBL_EPSILON*max(term scale)`이다. `linf Rhat`를 저장한다. physical array를 변경하지 않는다.

## 5. parity59 frozen 입력 계약 실측

### 5.1 존재하는 산출물

실측 run directory는

```text
/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59/
```

이다. 관련 파일은 다음과 같다.

| 파일 | 실측 크기/행 | 실제 schema 및 의미 |
|---|---:|---|
| `lumina_c1_bins.csv` | 880,945 B / 14,400 data rows | `iter,shell,bin,lam_lo_A,lam_hi_A,J_bin,W,T_R,mode`; 12 iter x 50 shell x 24 coarse bands |
| `lumina_c2_bfr_dump.csv` | 30,634,495 B / 600,000 rows | `iter,shell,bin,nu_mid,J_raw,bfr,j_nu_count`; 12 x 50 x 1000, `nu=1.503979e14--2.99206e16 Hz` |
| `cmf_fine_linedump_s8.csv` | 104,094,984 B / 793,505 rows | `line_id,shell,lambda_A,nu,J_fine,J_binned,S_l,B,SoverB,Jbin_over_Jfine,tau_sob,Sl_times_esc`; line sample only |
| `lumina_plasma_state.csv` | 2,739 B / 50 rows | `shell_id,W,T_rad,n_e,T_e` |
| `lumina_spectrum_formal.csv` | 51,000 B | `wavelength_angstrom,flux`; emergent spectrum, local field 아님 |

`lumina_c1_bins.csv`와 C2의 exact producer 설명은 `src/lumina_plasma.c:1171-1215,1385-1421`에 있다. C2 `J_raw`는 raw MC per-Hz field이고 `bfr`는 normalized photoionization-rate density이지 opacity나 emissivity가 아니다. `cmf_fine_linedump` 생산 지점은 `src/lumina_cmfgen.c:2568-2626`이며 선택 shell의 line diagnostic이다.

geometry는

```text
data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv
```

이고 실제로 50 rows, `r_inner/r_outer [cm]`, `v_inner/v_outer [cm s^-1]`를 가진다. s0은 `6.5639808e14--7.789257216e14 cm`, s49 outer는 `6.78278016e15 cm`; config symlink가 가리키는 `data/tardis_reference_toy06_19p48d/config.json`은 `t_exp=1,683,072 s`, `T_inner=10,020 K`이다. `r=v*t_exp`가 파일 정밀도에서 성립한다. loader는 geometry와 config의 값을 읽고 이 identity를 다시 검증한다.

CMFGEN 비교 실물은 `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/{RVTJ,EDDFACTOR,EDDFACTOR_INFO}`이고 각각 604,183 B, 142,832,872 B, 284 B이다. 기존 검증은 ND=90, good frequency record 196,185, `3.499e12--1.000e18 Hz`, `FINISH=1`을 확인했다(`docs/CODEX_W3_GAMMA_TRIPLE_COMPARE_2026-07-31.md:30-43,63-65`).

### 5.2 판정: frozen `chi,eta` 직접 복원 불가능 — UNRESOLVED

**UNRESOLVED-INPUT-1:** parity59 directory에는 전 shell x 전 frequency의 total `chi_nu` 및 `eta_nu`가 없다. C1/C2는 radiation-field/rate estimator이고, fine line dump는 s8/s45/s49의 line location sample이라 continuum, 전 shell, 전 frequency emissivity를 복원할 수 없다. `lumina_spectrum_formal.csv`도 surface integral이다.

메모리 안에는 coarse state가 존재한다. `CMFGENState`는 `chi_es`, `chi_abs`, `chi_line`, `chi_tot`, `S_fixed`, `J`를 가진다(`src/lumina_cmfgen.h:29-60`), CUDA deterministic loop가 매 iteration `cmfgen_assemble` 후 `cmfgen_solve_J`를 호출한다(`src/lumina_cuda.cu:7488-7515`). 그러나 parity59에서는 그것을 파일로 보존하지 않았다. 기존 `LUMINA_CMFGEN_JDUMP`도 활성화되지 않았고, 활성화돼도 출력 정밀도가 `chi %.4e`, `J %.6e`라 `1e-4` acceptance의 입력 정본으로 부족하다(`src/lumina_cmfgen.c:932-952`).

따라서 이 문서만으로 KA 구현은 즉시 가능하지만, parity59 `chi,eta` 판별 벤치는 입력 capture 전까지 PASS/FAIL이 아니라 **UNRESOLVED**다. C1/C2나 line dump로 `eta`를 추측해 벤치를 진행하면 안 된다.

### 5.3 Wave-3.2 계기 항목: 최소 dump 확장

Wave-3.2 instrumentation batch에 `W32-CMF-FROZEN-CHIETA`를 추가한다. 수정 지점은 둘 중 coarse authoritative state를 가장 작게 건드리는 곳이다.

1. `src/lumina_cmfgen.c`에 `cmfgen_dump_frozen_chieta(const CMFGENState*, const Geometry*, int iter, const char *path)`를 추가한다.
2. CUDA pure-CMFGEN/co-evolve loop에서 `cmfgen_solve_J`와 optional J damping block이 모두 끝난 직후(현재 `src/lumina_cuda.cu:7538` 다음), 최종 field epoch에 호출한다. parity59 비교 계약은 consumer iter 11이 읽은 producer iter 10이므로 **iter=10, post-damping flag와 field generation을 header에 기록**해야 한다(`docs/CODEX_W3_GAMMA_TRIPLE_COMPARE_2026-07-31.md:37-42`).
3. CSV가 아니라 little-endian binary v1로 다음을 전부 `float64`로 쓴다.

```text
magic[8]="LCMFCE01", endian=0x01020304, version=1
uint64 nr, nnu, iteration, field_generation
uint32 flags (post_damp, coherent_frozen, frequency_descending), reserved=0
double t_exp_s
double r_edge[nr+1]
double nu[nnu]                    # Hz, descending in dump contract
double dnu[nnu]
double chi_total[nr*nnu]          # cm^-1
double chi_coherent[nr*nnu]       # cm^-1
double eta_fixed[nr*nnu]          # chi_total*S_fixed
double eta_coherent[nr*nnu]       # chi_coherent*J used by frozen state
double eta_total[nr*nnu]          # eta_fixed+eta_coherent, redundant audit column
double J_producer[nr*nnu]
sha256 payload checksum in sidecar manifest
```

고정 header와 각 array는 위 순서대로 padding 없이 field-by-field little-endian으로 직렬화한다. native C struct를 통째로 `fwrite`하지 않는다. `dnu`는 frequency 순서와 무관하게 양의 bin width이며, 현재 ascending `CMFGENState` 배열은 writer가 모든 frequency-dependent array를 함께 역순으로 바꾼다.

현재 state에서 `eta_fixed=chi_tot*S_fixed`, `eta_coherent=chi_es*J`로 직접 산출 가능하다(`src/lumina_cmfgen.c:419-423`, `src/lumina_cmfgen.h:55-60`). dump 시 `max |eta_total-(eta_fixed+eta_coherent)|`가 bitwise 0인지 기록한다.

Stage 3.1의 **수송 단독 판별**은 `chi_total`과 이미 수렴/동결된 `eta_total`을 입력하고 `chi_coherent=0`으로 둔다. 그래야 새 solver와 기존 MC/CMFGEN field의 차이가 수송 이산화에 한정되며 scattering source를 새로 반복해 `eta`를 바꾸지 않는다. coherent scattering 재수렴은 KA2와 Stage 3.2 결합 시험의 별도 질문이다.

fine producer 자체의 `chi,eta`가 판별 대상이라면 `src/lumina_cmfgen.c:2419-2459`의 `fs` solve 직후 같은 schema를 쓰되 `nnu=NF`로 dump한다. 이는 수백 MB 이상이므로 coarse v1 capture를 최소 필수로 하고 fine capture는 별도 flag로 한다.

## 6. KA 사전등록

공통으로 h/h2/h4는 모든 활성 독립 격자를 동시에 2배 세분한다. `e_h=||u_h-u_exact||_2/||u_exact||_2`, 관측 차수는

```text
p_obs = log2( ||u_h-u_h2||_2 / ||u_h2-u_h4||_2 ).                (10)
```

로 계산한다. exact가 0인 cell은 max-norm에서 characteristic intensity scale로 정규화한다. source-iteration tolerance와 reference quadrature error는 예상 `h4` truncation error의 1/20 이하로 둔다. 모든 seed, grid, norm, 제외 cell 수를 JSON에 기록한다.

### 6.1 KA1 — pure absorption, 주어진 S와 tau

설정은 radius `R`의 full sphere, `chi=chi0>0`, scattering 0, 외부 전 방향 무입사, advection off(`a=0`)이다. SC가 상수 source를 segment마다 정확히 적분해 Richardson 오차가 roundoff로 사라지는 것을 막기 위해 smooth quadratic source

```text
S(r)=S0[1+q(r/R)^2], q=0.5, eta(r)=chi0 S(r)
```

를 쓴다. 임의 점 `(r,mu)`에서 upstream vacuum boundary까지 거리는

```text
l(r,mu) = r mu + sqrt(R^2-r^2(1-mu^2)),
tau(r,mu) = chi0 l(r,mu).                                        (11)
```

ray를 upstream 방향으로 거리 `s`만큼 거슬렀을 때 `r_s^2=r^2+s^2-2r mu s`이므로 exact intensity는 닫힌 형태로

```text
I_exact = chi0 S0 { M0 + q/R^2 [r^2 M0 - 2r mu M1 + M2] },
M0 = [1-exp(-T)]/chi0,
M1 = [1-exp(-T)(1+T)]/chi0^2,
M2 = [2-exp(-T)(T^2+2T+2)]/chi0^3,
T = chi0 l.                                                       (12)
```

따라서

```text
J_exact(r) = 1/2 integral_-1^1 I_exact(r,mu) dmu.                (13)
```

Equation (13)은 `mpmath` 80-digit adaptive quadrature로 계산한다. `T<<1`인 `M_n`도 고정밀 oracle에서 평가하므로 cancellation이 없다. 추가 low-level segment gate는 임의 piecewise-constant `(S_j,Delta tau_j)`에 대해

```text
I_N = I_0 exp(-sum tau_j)
      + sum_j S_j(1-exp(-Delta tau_j)) exp(-sum_{q>j}Delta tau_q). (14)
```

를 비교한다.

기본 grids는 `(Nr,Nmu)=(32,8),(64,16),(128,32)`, test optical depths `chi0 R={1e-3,1,100}`이다.

Acceptance:

- h4의 `I` 및 `J` 상대 L2 오차 각각 `<=1e-4`; max scaled error `<=3e-4`.
- 세 tau case 모두 `1.8 <= p_obs <= 2.2`(optically thin case는 roundoff plateau 전에 측정).
- Equation (9) transport residual `<=1e-4`.
- outer incoming 및 center symmetry 오차 `<=1e-12`.
- clamp/negative/non-finite count 모두 0.

### 6.2 KA2 — coherent scattering, homogeneous spherical atmosphere

elementary closed form이 없는 문제를 “해석해”라고 속이지 않는다. 정확한 비교 대상은 vacuum-boundary homogeneous sphere의 **정확한 Fredholm integral equation**이고, 독립 고정밀 Nyström solve를 oracle로 쓴다.

```text
chi=chi0,
S(r)=epsilon B0 + (1-epsilon)J(r),

J(r) = Lambda[S](r),
Lambda[S](r) = chi0/(2r) integral_0^R r' S(r')
  { E1(chi0|r-r'|) - E1(chi0(r+r')) } dr'.                     (15)
```

`r=0`은 극한 `Lambda[S](0)=chi0 integral_0^R S(r')exp(-chi0 r')dr'`를 쓴다. `E1(x)=integral_x^infinity exp(-t)/t dt`. Equation (15)는 3-D formal integral을 각도에 대해 정확히 적분한 식이다.

oracle은 80-digit arithmetic, Gauss-Legendre Nyström + `r=r'` logarithmic singularity subtraction, `Nref=2048/4096` 두 해의 상대차 `<1e-9`를 요구한다. test parameters는 `chi0 R=1`, `epsilon=0.2`, `B0=1`; production solver는 `eta_fixed=epsilon chi0 B0`, `chi_coherent=(1-epsilon)chi0`를 받는다. grids는 KA1과 같다.

Acceptance:

- h4 `J` 대 Equation (15) oracle 상대 L2 `<=1e-4`, max scaled error `<=3e-4`.
- `1.7 <= p_obs <= 2.3`(Fredholm singular kernel의 max-norm은 차수 판정에 쓰지 않음).
- source fixed-point residual `<=1e-10`, transport residual `<=1e-4`.
- energy ledger `L_thermal = L_escape + 4pi integral epsilon*chi0*J dV`의 상대 closure `<=1e-4`.
- max iterations 내 수렴, clamp/negative/non-finite count 0.

### 6.3 KA3 — homologous redshift, 단일 선 보존

`chi=eta=0`, `a=1/(ct_exp)>0`인 radial p=0 characteristic 길이 `L`에 narrow Gaussian line profile을 inner irradiation으로 주입한다. `A=L/(ct_exp)`라 하면 Equation (1)의 exact characteristic solution은

```text
nu_out = nu_in exp(-A),
I_out(nu) = exp(-3A) I_in(nu exp(A)),
I_nu/nu^3 = constant along matched characteristic.               (16)
```

Gaussian은 `ln nu`에서 중심 `x0`, 폭 `sigma_x=0.04`, domain은 양쪽 8 sigma와 shift A를 포함하고 `A=0.1`로 둔다. exact profile을 grid cell-average하여 point-sampling alias를 제거한다. 보존 진단은

```text
centroid: <ln nu>_out - <ln nu>_in = -A,
invariant area: exp(3A) integral I_out dlnnu
                = integral I_in dlnnu.                            (17)
```

이다. grids는 `(Ns,Nnu)=(32,128),(64,256),(128,512)`이고 frequency-domain 양 끝의 exact profile이 peak의 `1e-12` 미만인지 먼저 검사한다.

Acceptance:

- h4 exact shifted profile 상대 L1/L2 각각 `<=1e-4`.
- centroid shift absolute error `<=1e-4`, invariant-area relative error `<=1e-4`.
- profile L2의 `1.8 <= p_obs <= 2.2`, transport residual `<=1e-4`.
- peak가 domain boundary에 닿은 cell 0, clamp/negative/non-finite count 0.

### 6.4 redistribution 자리

`LCMF_SCAT_REDISTRIBUTION` enum과 callback signature만 선언한다. 호출 시 성공한 것처럼 coherent로 낮추지 말고 `LCMF_EUNSUPPORTED`를 반환한다. 이 구조 gate 자체는 빌드/ABI test만 하며 Stage 3.1 acceptance KA 수에는 넣지 않는다. 로드맵 최종 redistribution KA는 Stage 3.2에서 닫는다.

## 7. parity59 판별 벤치

### 7.1 입력 epoch와 shell

- field producer `iter=10`, consumer `iter=11`, lag=1을 고정한다.
- 중심 shell은 s8. geometry midpoint는 `10088.0 km s^-1`이고 CMFGEN RVTJ의 `9610.017--10163.506 km s^-1` 사이 log-J interpolation weight는 `0.863582`이다(`docs/CODEX_W3_GAMMA_TRIPLE_COMPARE_2026-07-31.md:28-35`).
- candidate/MC/CMFGEN 세 field는 모두 Lumina 1000-bin edges에 **bin-integral 보존 평균**한다. point interpolation은 금지한다.
- J ratio 표의 `J_MC`는 C2의 `J_raw`(iter 10, s8)다. Gamma 표의 MC 기준은 실제 consumer와 동일하게 positive mask에서는 `bfr`, fallback에서는 C1 reconstructed J를 쓰는 기존 `B`이다. 두 물리량을 같은 것으로 혼동하지 않는다.
- frozen dump의 payload checksum, iteration, pre/post damping flag가 기대값과 다르면 fail closed 한다.

### 7.2 J ratio table

대역은 사전 고정한다.

```text
B0  600--1000 A
B1 1000--1500 A
B2 1500--2000 A
B3 2000--2500 A
B4 2500--3000 A
BALL 600--3000 A
```

각 field `X`의 band scalar는 `J_X(B)=integral_B J_nu dnu / integral_B dnu`. 표에는

```text
J_new/J_MC, J_new/J_CMFGEN, J_MC/J_CMFGEN,
log10 of each ratio,
d_new=|log10(J_new/J_CMFGEN)|,
d_MC=|log10(J_MC/J_CMFGEN)|,
toward_CMFGEN = (d_new < d_MC)
```

를 싣는다. 추가 spectral norm은 band별 `median`, `p10/p90` of `log10(J ratio)`와 positive-pair/zero-bin count다. band integral이 양수이면 개별 0-bin이 있어도 integrated ratio는 유효하다. per-bin log quantile은 두 field가 모두 양수인 bin에서만 계산하고 제외 수를 명시한다. band integral이 0이거나 값이 negative/non-finite이면 floor로 대체하지 않고 해당 ratio를 UNRESOLVED로 한다.

### 7.3 Gamma 재계산 연결

`scripts/stage31_cmf_field_bench.py`는 `scripts/w3_gamma_triple_compare.py`를 module로 import하여 다음 실측 코드를 재사용한다.

- 1000-bin grid 및 C1/C2 loader: `:171-223`.
- CMFGEN EDDFACTOR/RVTJ parsing, velocity interpolation, integral-preserving bin average: `:320-456`.
- sigma/threshold/route/SL fraction과 `4pi sigma J/(h nu)` quadrature: `:530-588`.

기존 script를 수정해 새 CLI를 얹지 않는다. 새 runner가 candidate `J_new[1000]`를 `rate_replay(..., Jcmf=J_new)`에 넘겨 Fe III C48 lump와 S II SL4의 `Gamma_new`를 만들고, 기존 `Gamma_MC(B)` 및 `Gamma_CMFGEN(C)`와 함께 다음을 보고한다.

```text
Gamma_new/Gamma_MC, Gamma_new/Gamma_CMFGEN,
log10 ratios, |log Gamma_new/Gamma_CMFGEN| < |log Gamma_MC/Gamma_CMFGEN| 여부.
```

기존 측정값은 MC가 CMFGEN보다 Fe III에서 1.191977 dex, S II에서 1.826934 dex 높다(`docs/CODEX_W3_GAMMA_TRIPLE_COMPARE_2026-07-31.md:10-17`). 이것은 회귀 입력이지 새 solver의 목표값으로 강제하지 않는다.

### 7.4 판독과 벤치 acceptance

벤치는 방향을 미리 “PASS”로 강제하지 않는 falsifier다.

- `J_new`와 두 Gamma가 CMFGEN 쪽으로 이동: MC transport-field 결함 가설 지지.
- `J_new`가 MC UV excess를 재현: frozen `chi,eta` 자체가 다음 조사 대상.
- band별 방향이 갈림: opacity/emissivity와 transport의 wavelength-dependent 혼합 원인, **UNRESOLVED-MIXED**.

어느 물리 판독도 구현 실패는 아니다. 벤치 산출 acceptance는 다음의 완전성/수치 계약이다.

- s8의 6-band ratio table과 두 target Gamma table이 모두 존재.
- candidate input/solution grid identity max relative error `<=1e-12`.
- CMFGEN native-to-bin integral conservation `<=1e-12`(기존 script도 이를 검사한다: `:434-449`).
- candidate transport residual `<=1e-4`; 모든 field/rate가 finite and nonnegative.
- clamp count 0. 0/negative/non-finite를 floor로 숨긴 band 0.
- 방향 판정은 결과 그대로 기록하며 불리한 결과를 acceptance failure로 바꾸지 않는다.

## 8. 전체 acceptance matrix

| Gate | 수치 문턱 | 실패 처리 |
|---|---|---|
| input/schema | dims/order/unit/checksum/epoch exact; log-grid rel `<=1e-12` | fail closed / UNRESOLVED |
| KA1 | h4 relative `<=1e-4`, `p=1.8--2.2`, residual `<=1e-4` | FAIL |
| KA2 | h4 relative `<=1e-4`, `p=1.7--2.3`, source residual `<=1e-10`, transport `<=1e-4`, energy `<=1e-4` | FAIL |
| KA3 | profile `<=1e-4`, centroid abs `<=1e-4`, invariant `<=1e-4`, `p=1.8--2.2`, residual `<=1e-4` | FAIL |
| determinism | 동일 executable/input의 J binary hash 3회 동일 | FAIL |
| precision | solver와 dump 전부 double | FAIL |
| clamp/floor/freeze | physical state clamp/floor count exactly 0 | 발견 즉시 UNRESOLVED+좌표 노출 |
| parity bench | 6 bands + 2 Gamma rows + provenance, residual `<=1e-4` | 입력 없으면 UNRESOLVED; 방향은 판독 |

explicit `min/max`로 physical `I,J,chi,eta,S,tau`를 고치지 않는다. 입력 검증, quadrature bound, loop index clamp는 물리 clamp가 아니지만 별도 validation branch로 분리한다. overflow/underflow, BDF2 음수, non-convergence를 위생 clamp로 덮지 않는다.

## 9. 구현 단계와 규모/게이트

규모는 review 포함 person-day(PM)와 대략적 신규 LOC다. 합계 예상은 **9--14 PM, C 1.2--1.8 kLOC + Python 0.7--1.1 kLOC**이다.

| 단계 | 구현 내용 | 예상 | 다음 단계 진입 gate |
|---|---|---:|---|
| 0. 골격 | header/API, checked allocation, grid validator, p-ray cache, binary input/manifest reader, deterministic diagnostics | 2--3 PM / C 400--600 | strict compile flags, malformed-schema tests, ASan/UBSan, double-only audit |
| 1. KA1 | static SC, two-leg sphere, GL `mu`, J/H/K, independent residual, exact absorption runner | 1.5--2 PM / C 250--350 + Py 150 | KA1 전 문턱, 3-run hash, clamp 0 |
| 2. KA2 | `eta_fixed+chi_s J` fixed point, Fredholm oracle, energy ledger | 2--3 PM / C 180--280 + Py 220 | KA2 전 문턱, no hidden damping/ALI |
| 3. KA3 | k0/k1/BDF2 history, redshift profile/invariant diagnostics | 2--3 PM / C 250--350 + Py 180 | KA3 전 문턱; advection-off가 KA1 binary result를 바꾸지 않음 |
| 4. 벤치 | frozen dump loader, s8 solve, band ratios, `w3_gamma_triple_compare.py` module reuse, report | 1.5--3 PM / C 100--200 + Py 250--400 | dump provenance + residual + 6-band/2-Gamma completeness |

단계 4는 `UNRESOLVED-INPUT-1`이 닫히기 전에는 loader/schema/report dry-run까지만 완료로 표시한다. 이를 우회하기 위한 새 model run이나 `chi,eta` 추정은 이 WBS에 없다.

## 10. UNRESOLVED register

1. **UNRESOLVED-INPUT-1 (blocking for parity bench):** parity59 산출물에 total `chi_nu,eta_nu`가 없다. §5.3 dump capture가 필요하다.
2. **UNRESOLVED-EPOCH-1:** capture가 producer iter 10의 pre-damping인지 post-damping인지 현재 파일만으로 고정할 수 없다. dump header와 consumer가 실제 읽은 generation을 맞춰야 한다.
3. **UNRESOLVED-FINE-1:** coarse 1000-bin expansion-opacity `chi,eta`가 600--3000 A 판별에 충분한지, fine Doppler-grid dump가 필요한지는 coarse 결과의 resolution study 뒤 결정한다. coarse를 fine truth로 선언하지 않는다.
4. **UNRESOLVED-BDF2-POSITIVITY:** physical frozen source에서 BDF2 effective emissivity/solution 음수가 생기는지 입력 없이는 알 수 없다. clamp하지 않고 최초 좌표와 항을 보고한다.
5. **UNRESOLVED-KA2-TERMINOLOGY:** coherent spherical atmosphere는 elementary closed form이 없다. 이 명세는 exact Fredholm equation + independently converged high-precision oracle를 “정확해”로 사전등록했다. review가 elementary manufactured solution만을 요구하면 KA2 fixture를 재승인해야 한다.
6. **UNRESOLVED-REDISTRIBUTION:** 구조 자리만 있으며 로드맵 최종 redistribution acceptance는 3.2 소유다.

이 항목 중 1--3은 Wave-3.2 계기/입력 capture 작업에 편입하고, 4는 Stage 3.1 구현 첫 physical-input run의 판독 항목으로 남긴다.
