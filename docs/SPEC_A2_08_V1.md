# A2-08 구현 명세 V1 — signed CPU opacity와 generation-bound source 소비

- 개정: 9
- 저작: Codex
- 검수: fable
- 구현: Codex
- 기준 HEAD: `3ddd95c0de20abea3284ca326ce41b7968d4b26d`
- 단계: A-2 캠페인의 A2-08 하나
- 최종 물리 상태: **`BLOCKED_MISSING_CHI_DATA`**

## 0. 규범 우선순위와 이번 단계의 한 계약

이 명세는 다음 존재 파일을 현재 HEAD에서 직접 읽어 작성했다.

- `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md`
- `docs/A2_01_DISPOSITION_LEDGER.md`
- `docs/SPEC_A2_05_V2.md`, `docs/CODEX_REVIEW_A2_05.md`,
  `validation/a2_05/A2_05_CLOSURE.md`
- `docs/SPEC_A2_06_V2.md`부터 `docs/SPEC_A2_06_V5.md`,
  `docs/CODEX_IMPL_A2_06.md`, `validation/a2_06/A2_06_CLOSURE.md`
- `docs/SPEC_A2_07_V1.md`, `docs/CODEX_IMPL_A2_07.md`
- `docs/A2_00_OPHYS_PROFILE.json`, `docs/ORDER_OPHYS_RUN_BY_CODEX.md`,
  `docs/CHIETA_CAPTURE_RUN_PLAN_2026-08-01.md`

발주문에 적힌 `docs/CODEX_IMPL_A2_05.md`는 현재 저장소에 없다. 따라서 그 이름을
근거로 가장하지 않고, 실제 존재하는 A2-05 명세·검수·폐합 문서를 근거로 삼는다.

상위 ORDER의 단계표는 A2-08을 CPU `χ_ν`, A2-09를 CPU `η_ν`로 나누지만, A2-06
폐합은 line-source·blanketed-heating·CMF formal/source 소비를 명시적으로 A2-08로
재배치했고 이번 개정 발주는 CPU BF 재방출 소비까지 A2-08로 확정했다. 그러므로 이
개정의 단일 계약은 다음과 같다.

> **A2-08은 A2-07의 generation-bound population과 정본 `RadiationField` /
> `LineJbarCache`에서 유도한 checked view로 signed CPU opacity와 그 opacity에 붙은
> source 소비를 한 세대로 게시한다. `(W,T_rad)`, raw `J`, raw `jbar`, 숫자 sentinel,
> Planck 직접 표본은 이 call graph에서 소비할 수 없다.**

이는 두 개의 독립 단계를 합치는 것이 아니다. `χ`·line source·heating field·재방출
CDF가 서로 다른 generation이나 부호 규약을 보지 못하게 하는 하나의
`opacity/source publish` 계약이다. A2-09는 여전히 전체 `η_ν` 성분의 물리식,
macro-atom 재분배/전이확률, L-3·L-5 물리 PASS를 소유한다. A2-08은 그중 이미
재배치된 **source 소비 인터페이스와 BF 재방출 sampler 배선**만 닫는다.

## 1. 범위, 성공 정의, 비범위

### 1.1 반드시 구현할 범위

1. CPU `χ_ν`를 `es`, `bb`, `bf`, `ff`, `total`로 분리하고 signed 값으로 게시한다.
2. BF는 A2-07 population·partition·`T_e`·`n_e`와 A2-05의 보존 부분-빈 단면적
   적분을 사용한다. radiation-dependent stimulated rate/audit는 checked
   `RadiationFieldView.J_nu`를 사용한다.
3. BB는 동일한 공개 population과 정규화 profile로 계산한다. 두 Sobolev 생산식의
   stimulated-emission clamp와 `1e-100` floor를 없애고 하나의 signed helper로 합친다.
4. FF와 전자산란은 A2-07이 게시한 `n_e` 및 `T_e`를 읽는다. 복제된
   `OpacityState.electron_density`와 미사용 `t_electrons`를 CPU 물리 소유자로 남기지
   않는다.
5. A2-06이 넘긴 line-source fallback 2행과 blanketed-heating field 2행을 checked
   `RadiationFieldView`/`LineJbarView`로 바꾼다.
6. `src/lumina_cmfgen.c:3153,3159`의 formal/source raw line-field read를 없앤다.
7. CMFGEN replay commit이 `CMFGEN_REPLAY` provenance의 line block을 `J_ν`와 같은
   원자적 commit에 게시하도록 production 배선을 닫는다.
8. `bf_absorption_event`의 `sample_planck_frequency(T_rad)`를 checked, bin-integrated
   re-emission source CDF 소비로 바꾼다.
9. 등록 대역의 셸별 `χ` 성분 적분, 부호 interval, self-closure를 저장해 CHI truth가
   도착하면 재실행 없이 즉시 비교할 수 있게 한다.
10. 원장 `opacity_rate` 9행과 `opacity` 3행, A2-06 재배치 4행, A2-09에서
    재배치하는 emissivity 1행 및 원장 밖 소비를 현행 줄번호로 1:1 처분한다.

### 1.2 성공 정의

A2-08 구현 성공은 다음 두 상태를 동시에 정직하게 기록하는 것이다.

- 내부 계약: signed component publish, self-closure `<=1e-10`, source/replay wiring,
  static census, 필수 negative controls와 전 회귀가 PASS.
- L-4 물리: `CHI_DATA(_INFO)` 부재 때문에 CHAIN과 ORACLE_INPUT 모두
  **`BLOCKED_MISSING_CHI_DATA`**, gate rc 3. `NEG_OPAC`이나 `MEANOPAC`으로 승격 금지.

상위 ORDER `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:663`의 “L-4 PASS”는 최종 인수조건이지
현재 없는 truth를 만들어 내라는 지시가 아니다. 같은 문서 `:475-488`의 L-4 전문과
이번 발주가 우선한다.

### 1.3 비범위

- CUDA/GPU `χ`: A2-14. 이번 단계에서 `.cu`를 수정하거나 GPU 결과를 PASS시키지 않는다.
- 전체 analytic `η_ν` component closure, macro-atom 목적지/재분배 확정: A2-09.
- `T_e` 에너지 방정식과 heating/cooling balance: A2-10. A2-08은 그 방정식에 들어가는
  field construction만 바꾼다.
- CPU formal transfer 전반과 observer spectrum: A2-11. 단, A2-08 source raw read와
  signed-opacity handoff에서 음수를 자르는 행은 이번 계약상 제거한다.
- native J seed와 scalar 구조체 완전 삭제: A2-16/A2-17.
- 덱, `/gpfs` 원본, CMFGEN source 수정.

## 2. 현행 코드에서 확인한 결함

### 2.1 opacity/rate의 dead scalar shadow

`src/lumina_plasma.c:2624-2701`은 `T_rad`, `W`로 nebular `phi_neb`를 완전히 계산하지만
`:2714-2733`에서 이를 `(void)phi_neb`로 버리고 checked BF supplier만 사용한다. 원장
`opacity_rate` 9행은 새 J 근사식을 발명하는 대신 이 죽은 생산 shadow를 제거해야 한다.
output-only 비교가 필요하면 별도 진단 함수로 옮기고 production call graph에서
도달 불가능하게 한다.

### 2.2 signed χ를 파괴하는 실제 지점

현재 다음 값 변경을 확인했다.

| 현재 위치 | 현행 동작 | A2-08 처분 |
|---|---|---|
| `src/lumina_plasma.c:2985-2995` | inversion stim을 0, `tau`를 `1e-100`으로 floor | signed population difference로 교체 |
| `src/lumina_plasma.c:7071-7076` | BF event contribution을 `fmax(0,1-stim)` 및 양수만 반환 | signed net χ와 nonnegative event measure 분리 |
| `src/lumina_plasma.c:7773-7797` | BF grid corrfactor를 `fmax(0,1-stim)` | clamp 없이 signed BF component 게시 |
| `src/lumina_plasma.c:17514-17525` | NLTE tau의 inversion을 0, `1e-100` floor | 위와 같은 단일 helper 사용 |
| `src/lumina_cmfgen.c:833-838` | line-pop dump가 같은 clamp/floor를 재계산 | helper 결과/status를 읽고 round-trip |
| `src/lumina_cmfgen.c:2009-2021` | BF/FF 음수를 0으로 바꿔 source assembly | checked components를 그대로 결합 |
| `src/lumina_cmfgen.c:2990-2998` | formal `chi0`, `dtau` 음수를 0 | finite signed transfer; 지수 표현범위 초과만 명시 실패 |
| `src/lumina_cmfgen.c:3073-3097` | `chi_tot/es/abs`, `dtau` 음수를 0 | signed handoff 보존; abs/zero 금지 |

`fabs`는 오차 분모·진단에만 허용된다. 물리 적분 입력, optical depth, component publish,
event channel identity를 바꾸는 데 사용할 수 없다.

### 2.3 source와 replay의 실제 gap

- `src/lumina_plasma.c:12064-12078`은 line source가 없거나 `<=0`이면
  `W B_ν(T_rad)`를 쓰고 blanketed field의 바탕도 `W B_ν(T_rad)`로 만든다.
- 같은 heating 적분은 `src/lumina_plasma.c:12122`에서 legacy `nlte->J_nu`를 직접 읽는다.
- `src/lumina_plasma.c:8035-8044`는 BF 흡수 뒤 `T_rad` Planck 분포를 직접 표본한다.
- `src/lumina_cmfgen.c:3153-3168`은 `jbar_line_det`, `jbar_line`, 기하학적
  `W_d B(T_inner)`를 순차 fallback으로 사용한다.
- MC commit은 `src/lumina_main.c:535`에서 line block까지 게시하지만 replay commit
  `src/lumina_cmfgen.c:3390-3446`은 request의 line block을 모두 0으로 둔 채
  `RadiationFieldView`만 refresh한다.
- `OpacityState.line_source_S`는 `src/lumina.h:217`에서 `<=0 => fallback`이라는 숫자
  sentinel 계약이다. producer와 consumer가 raw pointer를 공유해 generation과
  EXACT_ZERO/missing을 구분하지 못한다.

### 2.4 electron-density 복제 실측

현행 `src`에서 `opacity->electron_density`, `opacity.electron_density`, `t_electrons`의
텍스트 hit는 18개다. CPU mirror write는 다음 세 곳이며 “동기화 6곳”은 현재 코드와
일치하지 않는다.

- opacity seed → plasma: `src/lumina_main.c:147`, `:712`
- committed plasma → opacity: `src/lumina_plasma.c:6791-6793`

GPU의 seed copy `src/lumina_cuda.cu:7088`은 A2-12/A2-14로 남긴다. CPU 물리 read는
`src/lumina_transport.c:562`, `src/lumina_plasma.c:4460,5273,5666,5740,18671,18774,19020`
및 CMF source의 fallback 계열이다. 구현자는 낡은 “6”을 맞추기 위해 가짜 행을 만들지
말고 현재 3 CPU write와 모든 CPU read를 census한다.

## 3. 자료형과 소유권

### 3.1 한 세대의 `CpuOpacitySourceOwner`

구현 이름은 저장소 관례에 맞게 조정할 수 있으나 의미는 다음과 같아야 한다.

```text
CpuOpacitySourceOwner {
    required_generation
    committed_generation
    epoch, shell_geometry_hash, frequency_edge_hash, atomic_model_hash
    radiation_generation, line_jbar_generation
    population_generation, partition_generation, within_sl_generation
    tau_generation, te_generation, ne_generation

    chi_es[s,b]
    chi_bb[s,b]
    chi_bf[s,b]
    chi_ff[s,b]
    chi_total[s,b]
    chi_validity[component,s,b]

    line_source[line,s]
    line_source_validity[line,s]
    heating_field[s,b]
    heating_field_validity[s,b]
    reemission_weight[s,b]
    reemission_validity[s,b]
    reemission_cdf[s,b+1]
}
```

주파수 edge는 정본 `RadiationField`의 4000-bin edge를 빌리고, 값은 공이동계 빈 평균이다.
기존 `BFOpacity` 1000-bin 배열을 새 정본으로 선언하지 않는다. 필요한 transport 호환
projection은 위 owner의 generation-bound 파생 view이며 보존 재빈해야 한다.

모든 candidate 배열과 status를 비공개 작업 버퍼에서 만든 뒤 전체 검증에 성공했을
때만 한 번 publish한다. 어느 shell/component/source가 실패해도 새 generation의
일부 배열을 공개하지 않는다. 실패 후 public pointer와 committed generation은 이전
세대 그대로다.

### 3.2 checked read view

consumer는 owner 배열을 직접 받지 않고 다음을 검사하는 checked view만 받는다.

- epoch와 shell shape
- frequency-edge hash와 단위 `cm^-1`, frame `comoving`
- required/committed generation
- RadiationField, LineJbarCache, population, partition, within-SL, `T_e`, `n_e`, tau stamp
- line query-set hash와 profile id/hash
- component/source validity

raw `cs->J`, `nlte->J_nu`, `opacity->jbar_line`, `opacity->jbar_line_det`,
`opacity->line_source_S`, `BFOpacity.chi_bf`를 production consumer API로 넘기지 않는다.
진단 dump도 checked view를 받은 뒤 직렬화한다.

### 3.3 validity 상태

최소 상태는 다음을 서로 다른 enum 값으로 둔다.

```text
VALID
EXACT_ZERO
UNSAMPLED
OUT_OF_GRID
MISS
STALE_GENERATION
QHASH_MISMATCH
PROFILE_MISMATCH
INVALID_POPULATION
INVALID_PARTITION
INVALID_TE
INVALID_NE
NONFINITE
SOURCE_CANCELLATION_SINGULAR
EVENT_MEASURE_UNAVAILABLE
FORBIDDEN_FALLBACK
```

`VALID`와 `EXACT_ZERO`만 수치 입력이다. `EXACT_ZERO`는 실제 0이고 missing sentinel이
아니다. A2-05의 weighted-missing 규약은 BF cross-section 적분에서 그대로 적용한다.
적분 구간의 `w_miss <=1e-3`만 VALID로 남길 수 있고 그 값을 기록한다. STALE은 즉시
실패하며 UNSAMPLED/OOG/MISS는 0으로 대입하지 않는다. source는 `S<=0`, `-1`, NULL
같은 숫자/포인터 상태를 validity로 재해석하지 않는다.

## 4. component 식과 수치 규약

### 4.1 electron scattering

각 shell/bin에서

\[
\chi^{es}_{s,b}=n_{e,s}\sigma_T
\]

이다. `n_e`는 A2-07 population transaction과 같은 generation의
`PlasmaState.n_electron` checked view에서 온다. finite `n_e>=0`이어야 하며 정확한 0은
`EXACT_ZERO`다. 덱에서 읽은 `OpacityState.electron_density`는 초기 seed도 plasma
loader로 직접 넘기고 CPU owner가 되지 않는다.

### 4.2 bound-bound와 signed Sobolev

선 `l`의 기본 차이는

\[
D_l=n_l B_{lu}-n_u B_{ul}
    =B_{lu}\left(n_l-{g_l\over g_u}n_u\right)
\]

이다. `D_l<0`인 population inversion은 유효한 negative opacity다. `D_l`을 먼저
계산해 `n_l(1-ratio)` 형태의 0·무한대 cancellation을 피한다. `n_l=0,n_u>0`도 정확한
음수다. 두 현행 tau 생산자는 이 단일 helper와 A2-07 공개 population accessor를
사용한다.

\[
\tau_l=K_{Sob}\,f_{lu}\lambda_l t_{exp}
       \left(n_l-{g_l\over g_u}n_u\right)
\]

`tau=0`은 정확한 0, `tau<0`은 유효한 maser/inversion 값이다. NaN을 잡기 위해
`1e-100`으로 바꾸지 않고 `NONFINITE`로 실패한다. K-FRESH의
`tau_sobolev_require_refresh`, `tau_sobolev_mark_computed`,
`tau_sobolev_assert_fresh` 호출과 required/computed/first-consumer generation은
그대로 유지한다.

스펙트럼 component는 정규화된 공이동 profile로 보존 적분한다.

\[
\bar\chi^{bb}_{s,b}={1\over\Delta\nu_b}
 \sum_l\int_{\nu_{b,-}}^{\nu_{b,+}}
 {h\nu_l\over4\pi}D_{l,s}\phi_l(\nu)\,d\nu .
\]

profile id/hash는 A2-06 cache와 같아야 하고 `∫phi dnu=1`을 `<=1e-12`로 selftest한다.
빈 중심에 선 전체를 꽂거나 Planck 표본으로 line strength를 만들지 않는다.

`LUMINA_OPACITY_SKIP_Z`는 production에서 설정 자체가 `FORBIDDEN_FALLBACK`이다.
channel-drop negative child에서만 허용하며 marker를 출력한다. Z-INERT 원소의 정확한
0 population/tau는 계속 bitwise 0이어야 한다.

### 4.3 bound-free

각 continuum route의 net coefficient는 A2-07이 게시한 lower/upper population,
partition, `n_e`, `T_e`, target probability와 tabulated cross section으로 계산한다.
현행 `src/lumina_plasma.c:7650-7797`의 ARTIS/Milne lineage를 유지하되 다음을 강제한다.

- `n<1e-30` 같은 물리 floor로 valid population을 건너뛰지 않는다. EXACT_ZERO만 0이다.
- stimulated bracket `1-r_mod exp[-h(nu-nu_edge)/(kT_e)]`은 자르지 않는다.
- 여러 target route는 등록 확률로 합하며 누락 route는 fallback target을 발명하지 않고
  coverage/status에 남긴다.
- radiation-dependent stimulated recombination rate와 gate audit는
  `4π J_b/(h nu)`를 곱할 때 checked `RadiationFieldView.J_nu`를 쓴다. `J`는 incident
  rate에 들어가며 opacity coefficient 자체에 다시 곱해 이중 계산하지 않는다.
- scalar `(W,T_rad)` 또는 그것으로 fit한 color/dilution을 사용하지 않는다.

packet event 선택은 signed net `chi_bf`를 확률로 억지 정규화하지 않는다. 같은 route
분해에서 spontaneous/gross absorption의 nonnegative event measure와 induced-emission
measure를 별도로 보존하고, 전자가 event CDF를 소유한다. 둘의 차이만 net opacity다.
net이 음수라는 사실 자체는 오류가 아니며 `EVENT_MEASURE_UNAVAILABLE`을 발생시키지
않는다. 원자 route 정보가 없어 gross measure를 구성할 수 없을 때만 event consumer를
명시적으로 차단하고 signed L-4 component publish는 그대로 유지한다.

A2-05의 부분-빈 규약을 그대로 사용한다. threshold 빈은

\[
{1\over\Delta\nu_b}\int_{\max(\nu_{th},\nu_{b,-})}^{\nu_{b,+}}
 \sigma(\nu)\,C_{stim}(\nu,T_e,pop)\,d\nu
\]

를 tabulated `sigma`의 구간별 선형 해석으로 적분한다. edge 아래는 정확한 0이다.
Kramers route가 원자자료 계약상 허용된 경우에만 provenance와 count를 남기고 사용한다.
point sample, edge를 bin 시작으로 내림, 인접 bin 복사, σ 보정계수 흡수는 금지한다.

### 4.4 free-free

FF는 `T_e`, `n_e`, 공개 ion population/charge를 사용해 현행 stimulated factor
`-expm1(-h nu/kT_e)`를 포함한다. `n_i≈n_e`, `Z²≈1` fallback은 production에서
금지한다. 모든 활성 ion의 `Z_i² n_i` 합을 같은 population generation에서 계산한다.
`chi_ff`는 BF 배열에 섞지 않는다.

### 4.5 total과 self-closure

candidate total은 각 cell에서 고정 순서

```text
total = ((es + bb) + bf) + ff
```

로 한 번 계산한다. gate는 저장된 네 component를 long-double 또는 보상합으로 독립
재합산한다.

\[
E_{close}={|\chi_{total}-\Sigma_c\chi_c|\over
\max(\Sigma_c|\chi_c|,\mathrm{DBL\_MIN})}.
\]

모든 component와 total이 exact zero이면 `E_close=0`이다. 모든 shell/bin과 셸·대역
적분에서 `max E_close <=1e-10`이어야 한다. 성분 하나가 missing인데 total만 있는 상태는
closure PASS가 아니라 invalid publish다.

## 5. line source, heating field, 재방출 소비

### 5.1 line source는 값과 상태를 분리한다

population-native line emissivity와 opacity는

\[
\eta_l={h\nu_l\over4\pi}n_u A_{ul}\phi_l,\qquad
\chi_l={h\nu_l\over4\pi}(n_lB_{lu}-n_uB_{ul})\phi_l
\]

로 같은 population generation에서 만든다. `chi_l!=0`일 때만 `S_l=eta_l/chi_l`을
materialize한다. `chi_l=0, eta_l!=0`이면 무한 source를 숫자 cap/fallback으로 바꾸지
않고 `SOURCE_CANCELLATION_SINGULAR`로 표시하며 consumer는 `(eta,chi)` 형식을 사용한다.
negative finite `S_l`도 값 자체로는 valid할 수 있으므로 `<=0`을 missing으로 쓰지 않는다.

명시적 scattering source mode에서는

\[
S_l=(1-\epsilon_l)\bar J_l+\epsilon_l B_{\nu_l}(T_e)
\]

를 사용한다. `bar J_l`은 checked `LineJbarView`, 물질 thermal term은 `T_e`다.
population-native와 scattering mode 사이의 자동 fallback은 없다. 선택한 mode와
`epsilon` provenance를 manifest에 고정하고 필수 view가 invalid면 publish를 막는다.

### 5.2 A2-06 query set과 replay line block

A2-06의 `Q_g`는 A2-08에서 다음 합집합으로 한 번 확장한다.

```text
Q_A2_08 = Q_rate ∪ Q_line_source ∪ Q_heating ∪ Q_reemission ∪ Q_cmf_formal
```

reachable production consumer의 line id를 결과를 보기 전에 정렬·deduplicate하고
query-set hash를 만든다. “강한 선만” 같은 결과 의존 필터나 missing line의 coarse-J
복원은 금지한다.

MC는 기존 packet estimator로 이 집합을 누적한다. CMFGEN replay는 fine deterministic
field와 동일 profile을 보존 적분해 `line_jbar`, validity, line id, Q hash, profile
id/hash를 채운다. `src/lumina_cmfgen.c:3420-3445`의 request는 J와 line block을 한 번에
commit하고, 성공 뒤 `RadiationFieldView`와 `LineJbarView`를 둘 다 refresh한다. 어느
후보든 실패하면 둘 다 이전 generation이다. `CMFGEN_REPLAY` provenance를 유지한다.

`src/lumina_cmfgen.c:3153,3159`와 같은 formal/source lookup은 checked
`line_jbar_lookup`만 쓴다. MISS/UNSAMPLED/OOG/STALE일 때 `Jv`, raw array,
`W_d B(T_inner)`로 내려가지 않는다.

### 5.3 blanketed heating field

`src/lumina_plasma.c:12045-12078`의 base field는 같은 generation의
`RadiationFieldView.J_nu[s,b]`다. line이 없는 exact-zero bin은 그대로 `J_b`이며
`W B(T_rad)`가 아니다. signed tau를 건너뛰지 않고

\[
\beta(\tau)={-\operatorname{expm1}(-\tau)\over\tau},\quad \beta(0)=1
\]

을 사용한다. line weights와 source는 signed 상태를 보존한다. 합산 가중치가 정확히
상쇄되어 `Sbar`가 정의되지 않으면 Planck fallback 대신
`SOURCE_CANCELLATION_SINGULAR`다. 표현 가능한 finite negative tau는 반드시 증폭을
포함한 signed 식으로 계산한다. 지수 표현범위를 넘는 경우에만 값을 0/상한으로 바꾸지
않고 명시적 overflow 상태로 실패한다.

heating 적분의 `src/lumina_plasma.c:12122` raw `nlte->J_nu`도 checked radiation view로
바꾼다. `H_photo_dilute` 같은 output-only 옛-field 비교량은 production 해법에 합쳐지지
않는 별도 diagnostic shadow로만 남길 수 있고, 파일:행·도달 root를 allowlist한다.
A2-08은 `T_e` root solve나 cooling 항의 식을 바꾸지 않는다.

### 5.4 BF 재방출 sampler와 A2-09 경계

`bf_absorption_event`는 analytic `sample_planck_frequency`를 호출하지 않는다. 같은
generation에서 게시한 `reemission_weight[s,b]`의 빈 적분
`weight_b = eta_reemit,b * Delta_nu_b`로 정규화한 CDF를 이진 탐색하고, 선택한 빈
안에서는 manifest에 고정한 piecewise-constant density로 정확히 한 번 표본한다.

- `eta_reemit`은 A2-08에서 새 전체 emissivity 정본이라고 부르지 않는다.
- A2-08 builder는 현재 BF/free-bound source producer를 checked wrapper로 감싸고,
  radiation-dependent 항은 RadiationField, line/scattering 항은 LineJbarCache,
  population 항은 A2-07 view에 결박한다.
- 음수/nonfinite weight는 0으로 자르지 않고 publish 실패다.
- 전체 weight가 exact zero면 `EXACT_ZERO`이며 이전 CDF/Planck로 fallback하지 않는다.
- packet 방향과 주파수에 소비하는 RNG draw 수와 순서를 baseline/poison manifest에
  기록한다. 진단은 추가 RNG를 소비하지 않는다.

A2-09는 이 좁은 source view의 producer를 완전한 `eta_es/eta_bb/eta_fb/eta_ff/total`
정본으로 승격하고 L-3/L-5를 판정한다. A2-08은 L-5 PASS를 주장하지 않는다. 원장
emissivity 1행은 sampler 소비의 귀속만 A2-08로 재배치한다.

### 5.5 CMF EPAY scalar split

원장 opacity 2행인 `src/lumina_cmfgen.c:908,2144`의 `T_e > hotf*T_rad` heuristic은
삭제한다. dump와 production이 서로 다시 계산하지 않도록 하나의 source disposition
enum을 publish한다.

rate-shape source가 valid하고 `epay>=2`, `acc_w>0`인 셀은 그 명시적 source를 쓰며,
invalid이면 scalar hot/cold branch로 회피하지 않고 상태를 전파한다. absorbed-power
book의 field는 checked RadiationField view다. 기존 “near LTE이면 legacy chi*B shape”
휴리스틱을 보존하려면 `(T_e,T_rad)` 비가 아니라 실제 component/source detailed-balance
residual을 이름 붙인 순수 진단으로 사전등록하고 검수 승인을 받아야 한다. 결과를 본 뒤
threshold를 맞추는 것은 금지한다.

## 6. `n_e`, `T_e`, population 소유권

1. CPU seed loader는 deck `n_e`를 `PlasmaState.n_electron` 초기값으로 직접 읽는다.
2. A2-07 population transaction 이후에는 그 committed `n_e`만 CPU owner다.
3. CPU opacity, transport, CMF source/formal의 전자산란·FF는 checked material view를
   받는다. `plasma ? plasma->n_electron : opacity->electron_density` fallback은 금지한다.
4. `OpacityState.electron_density`를 ABI 때문에 잠시 남기면 `const` non-owning alias와
   generation stamp여야 하며 독립 allocation/copy/update가 없어야 한다. alias identity와
   generation mismatch는 fatal이다.
5. `OpacityState.t_electrons`는 현재 alloc/init/free만 있고 `T_rad`로 초기화된다. CPU
   필드를 제거하고 모든 matter-temperature 소비는 checked `PlasmaState.T_e`를 쓴다.
6. formal opacity thermal width `src/lumina_plasma.c:18305,18314`는 radiation field가
   아니라 물질 `T_e`가 옳다. `USE_MATTER_TEMPERATURE`로 1:1 처분한다.
7. CUDA의 device mirror와 upload는 건드리지 않고 A2-12/A2-14 allowlist에 남긴다.

## 7. 원장 1:1 처분표

구현은 `docs/A2_01_DISPOSITION_LEDGER.md` 원본 157행을 다시 쓰지 않고 A2-08 addendum을
추가한다. 고정 migration id, old 위치, 구현 직전 위치, 구현 후 위치, disposition이
각각 정확히 하나여야 한다.

### 7.1 `opacity_rate` 9행

| 고정 ID | 현행 witness | 최종 처분 |
|---|---|---|
| `A2-01:old2435:T_rad` | `src/lumina_plasma.c:2624` | dead nebular shadow 제거 |
| `A2-01:old2437:W` | `src/lumina_plasma.c:2626` | dead nebular shadow 제거 |
| `A2-01:old2498:T_rad` | `src/lumina_plasma.c:2695` | zeta shadow 제거 |
| `A2-01:old2499:T_rad` | `src/lumina_plasma.c:2696-2699` | Te/Trad shadow 제거 |
| `A2-01:old2500:W` | `src/lumina_plasma.c:2697-2699` | dilution shadow 제거 |
| `A2-01:old2501:T_rad` | `src/lumina_plasma.c:2697-2699` | ratio shadow 제거 |
| `A2-01:old2502:W` | `src/lumina_plasma.c:2699` | non-meta dilution shadow 제거 |
| `A2-01:old2503:T_rad` | `src/lumina_plasma.c:2700` | ML shadow를 output-only 진단으로 격리 또는 제거 |
| `A2-01:old2504:W` | `src/lumina_plasma.c:2701` | two-component shadow를 output-only 진단으로 격리 또는 제거 |

이 9행의 물리 successor는 이미 `src/lumina_plasma.c:2714-2733`의 checked BF rate와
A2-07 population이다. `T_color[J]` 같은 새 압축 scalar로 치환하지 않는다.

### 7.2 `opacity` 3행

| 고정 ID | 현행 위치 | 최종 처분 |
|---|---|---|
| `A2-01:old908:T_rad` | `src/lumina_cmfgen.c:908` | common source disposition enum 소비 |
| `A2-01:old2144:T_rad` | `src/lumina_cmfgen.c:2144` | 같은 enum; checked J/source book 사용 |
| `A2-01:old18010:T_rad` | `src/lumina_plasma.c:18305,18314` | checked `T_e`, `USE_MATTER_TEMPERATURE` |

### 7.3 A2-06 재배치 4행

| 고정 ID | 현행 위치 | 최종 처분 |
|---|---|---|
| `A2-06:old11908:W` | `src/lumina_plasma.c:11999-12000,12064-12066` | checked line source; W taint 0 |
| `A2-06:old11908:T_rad` | 같은 위치 | checked line source; Trad taint 0 |
| `A2-06:old11915:W` | `src/lumina_plasma.c:12073-12078` | base=`RadiationFieldView.J_nu` |
| `A2-06:old11915:T_rad` | 같은 위치 | base=`RadiationFieldView.J_nu` |

### 7.4 emissivity 1행의 단계 재배치

| 고정 ID | 원장 상태 | 현행 위치 | 개정 9 처분 |
|---|---|---|---|
| `A2-01:old7897:T_rad` | A2-09 `REPLACE_PLANCK_REEMISSION_SOURCE` | `src/lumina_plasma.c:8043-8044` | A2-08 checked reemission CDF consumer |

addendum에는 “A2-09 → A2-08, revision-9 order”를 명시한다. A2-09는 전체 eta producer와
L-3/L-5를 계속 소유한다.

### 7.5 원장 밖 필수 처분

| 현재 witness | 처분 |
|---|---|
| `src/lumina_cmfgen.c:3153,3159` | checked LineJbarCache lookup; fallback 0 |
| `src/lumina_cmfgen.c:3390-3446` | replay J+line atomic commit 및 두 view refresh |
| `src/lumina_plasma.c:7548-7797` | A2-07 population + signed BF + 부분-빈 적분 |
| `src/lumina_plasma.c:2985-2995,17514-17525` | 단일 signed line-opacity helper |
| `src/lumina_plasma.c:12122` | checked RadiationField read |
| `src/lumina.h:217` 및 모든 production `line_source_S` read | status-bearing source view |
| CPU `electron_density` mirror/read 전부 | A2-07 checked `n_e` owner |

## 8. 소비지점 전수 census와 allowlist

새 `a2_08` static census는 단순 문자열 0건만 보지 않고 call graph와 taint를 함께 본다.
production root는 최소 다음이다.

```text
compute_bf_opacity / bf continuum event selector
compute_tau_sobolev / nlte_update_tau_sobolev
CMF opacity/source assembly and EPAY disposition
line-source producer and every production consumer
blanketed-heating field builder and H_photo input
bf_absorption_event
CMFGEN replay commit and CMF formal/source line lookup
CPU transport/formal opacity handoff
```

합격 조건은 이 root들에서 다음 read/taint가 0인 것이다.

```text
plasma->W, plasma->T_rad
OpacityState.t_electrons, mutable OpacityState.electron_density
nlte->J_nu, cs->J as post-commit consumer
opacity->jbar_line, opacity->jbar_line_det
numeric-sentinel opacity->line_source_S
sample_planck_frequency in a reemission root
abs/max(0)/positive-only conversion of chi or tau
```

현재 전역 `plasma->W[`/`plasma->T_rad[` direct hit는 55개다. 전역 0을 요구하지 않는다.
A2-10 energy solve, A2-11 formal/output, A2-13/14 GPU, A2-17 lifecycle, output-only
diagnostic은 파일:행, 함수, root, 심볼, 비물리 이유, 후속 단계가 있는 좁은 allowlist로
남긴다. 파일 전체·행 범위 wildcard·“diagnostic” 한 단어 허용은 금지한다.

현행 `line_source_S`의 alloc/free/comment를 제외한 producer/consumer도 모두 분류한다.
특히 `src/lumina_cmfgen.c:785-789,1297,1795,3855,4174,4274,4313,4464,4557,5215`와
`src/lumina_plasma.c:12064-12066,15604-15607,17499-17542,18689-18690,19084-19085`를
목록 밖 소비자로 놓치지 않는다. gate/debug writer도 raw pointer가 아니라 checked
snapshot을 읽게 한다.

## 9. fallback 금지, 설정, 카운터와 `nlte_free`

### 9.1 production에서 금지하는 fallback/config

- `W B_nu(T_rad)`, `B_nu(T_inner)` line/source fallback
- raw/coarse/이전 generation J 또는 Jbar fallback
- invalid/negative source를 `B_nu(T_e)`로 자동 교체
- BF/FF missing atomic data를 임의 Kramers/`n_i≈n_e`로 대체
- missing population을 LTE, Saha, 이전 population, 0으로 대체
- `LUMINA_OPACITY_SKIP_Z`, source clamp, stimulated-emission disable
- `LUMINA_J_NU_UV_CAP` 등 scalar-derived field cap이 A2-08 input을 바꿈
- `chi<0 -> 0`, `fabs(chi)`, `tau<epsilon -> epsilon`
- invalid source CDF에서 Planck sampler로 복귀

설정되면 조용히 무시하지 않고 `FORBIDDEN_FALLBACK`과 rc 5로 종료한다. 각 poison은
별도 child process에서만 한 개를 켠다. static `getenv()` cache로 같은 process의 poison을
바꾸지 않는다.

### 9.2 필수 카운터

thread-local/atomic reduction 뒤 `nlte_free`에서 정확히 한 줄
`[A2-08][OPACITY-SOURCE]`로 다음을 보고한다.

```text
generation_required, generation_committed
shells_attempted, shells_published, cells_attempted, cells_published
es_terms, bb_terms, bf_terms, ff_terms
exact_zero_es, exact_zero_bb, exact_zero_bf, exact_zero_ff
negative_bb, negative_bf, negative_total, negative_intervals
blocked_stale, blocked_unsampled, blocked_oog, blocked_miss
blocked_profile, blocked_qhash, blocked_population, blocked_te, blocked_ne
source_line_terms, source_heating_bins, source_reemission_bins
source_exact_zero, source_cancellation_singular
closure_failures, nonfinite_failures, event_measure_unavailable
fallback_attempts, scalar_read_attempts, raw_view_attempts
partial_publish_attempts, ne_alias_mismatches
replay_line_blocks_attempted, replay_line_blocks_committed
```

정상 internal PASS lane은 blocked/fallback/scalar/raw/partial/alias/closure/nonfinite가 모두
0이고 attempted=published, required=committed다. negative component count는 0을 요구하지
않으며 fixture에서 `>0`이어야 한다. 실패 주입 lane은 원인 counter가 `>0`, 새 publish가
0이어야 한다. A2-05/06/07 기존 summary와 generation 불변식도 함께 검사한다.

### 9.3 종료코드

| rc | 의미 |
|---:|---|
| 0 | 모든 요청 gate PASS, 또는 negative wrapper가 기대 child FAIL을 확인 |
| 2 | 사용법, I/O, schema, manifest, hash, parser 오류 |
| 3 | truth/upstream 부족의 `BLOCKED_*`; 현재 L-4 정상 최종 rc |
| 4 | 계산은 완료됐으나 metric/self-closure/negative expectation FAIL |
| 5 | forbidden fallback, stale/raw read, partial publish 등 계약 위반 |

모든 JSON은 `status`, `reason_code`, `child_rc`, `wrapper_rc`를 가진다. internal-only
selftest PASS와 physical L-4 BLOCKED를 한 status로 합치지 않는다.

## 10. L-4 gate 사전등록

### 10.1 산출물과 schema

구현은 최소 다음 artifact를 `validation/a2_08/` 아래 남긴다.

```text
A2_08_OPACITY_COMPONENTS.npz
A2_08_OPACITY_COMPONENTS_MANIFEST.json
A2_08_COMPONENT_INTEGRALS.csv
A2_08_NEGATIVE_INTERVALS.csv
A2_08_SOURCE_VIEW.json
A2_08_STATIC_CENSUS.json
A2_08_SELFTEST.json
A2_08_L4_GATE.json
A2_08_REGRESSION_LEDGER.jsonl
```

component manifest는 edge, shell boundary, units, frame, component inclusion, signed/net
규약, summation order, 모든 generation/hash, validity counts를 가진다. CSV는 각 shell과
다섯 등록 대역에 대해 `total/es/bb/bf/ff/sum_components/closure`, signed integral,
absolute-contribution integral, negative-bin count를 기록한다.

등록 대역은 wavelength로 고정한다.

```text
450-918 A
918-1290 A
1290-2000 A
2000-10000 A
10000-25000 A
```

주파수 방향으로 변환할 때 edge 순서를 명시하고 보존 적분한다.

### 10.2 CHAIN과 ORACLE_INPUT

| lane | 즉시 상류 입력 | 목적 | 현재 상태 |
|---|---|---|---|
| CHAIN | current committed RF/LineJbar + A2-07 population/Te/ne | 전체 배선 | `BLOCKED_MISSING_CHI_DATA` |
| ORACLE_INPUT | 동일 checked 인터페이스에 같은 generation의 CMF J/Jbar/population/Te/ne를 commit | opacity/source 층 분리 | `BLOCKED_MISSING_CHI_DATA` |

ORACLE_INPUT도 raw 배열을 직접 대입하지 않는다. CMF population/line crosswalk 또는
generation proof가 없으면 더 구체적인 upstream BLOCKED를 함께 기록하되, CHI 부재를
가리지 않는다. CHAIN은 A2-05/06/07의 실제 status를 승계하며 하류 internal closure가
upstream BLOCKED를 PASS로 바꾸지 않는다.

### 10.3 truth-side universe와 `f_cov`

CHI_DATA가 도착하면 Lumina 결과를 열기 전에 writer schema로 total/es/bb/bf/ff의
포함 의미, units, frame, depth/frequency order, net/stimulated sign convention을 고정한다.
truth component absolute contribution을 내림차순 누적해 99.9% 활성 집합을 만들고
경계 동률을 모두 포함한다.

\[
f_{cov}={\sum_{active\cap matched}|\chi_C|\Delta\nu
\over\sum_{active}|\chi_C|\Delta\nu}.
\]

stale/unsampled/OOG/unmatched는 분자에서 빠질 수 있지만 truth 분모에서는 빠지지 않는다.
Lumina가 잘 계산한 항만으로 분모를 다시 만들지 않는다. 합격선은 다음과 같다.

- component truth coverage `f_cov >=0.95`
- 공통 셸별 signed total `E_1 <=0.15`, 분모 `sum Delta_nu |chi_C|`
- 다섯 대역 각각 signed `E_B <=0.15`
- component-sum/total self-closure `<=1e-10` CPU
- CMFGEN negative-active interval의 sign mismatch 0

MC 영향을 받는 CHAIN metric의 95% CI half-width는 합격폭의 1/3 이하여야 한다.
통계오차를 물리오차에서 빼지 않는다.

### 10.4 O-PHYS 요건 확인과 보강

`docs/A2_00_OPHYS_PROFILE.json:16-17`은 이미 `CHI_DATA`, `CHI_DATA_INFO`를 exact file로
요구하고 `:68-69`는 units/frame attestation을 요구한다. `:89-91`은
MEANOPAC/NEG_OPAC 대체를 명시적으로 금지한다. `docs/ORDER_OPHYS_RUN_BY_CODEX.md:67`은
`CMF_FLUX_PARAM: T [WR_ETA]`의 formal writer 경로를 확인한다.

다만 현재 profile은 component inclusion 의미까지 요구하지 않는다. 구현 시 profile과
attestation validator에 다음 CHI schema 필드를 추가해야 한다.

```text
record_schemas.CHI_DATA.writer_revision
record_schemas.CHI_DATA.record_layout
record_schemas.CHI_DATA.depth_order
record_schemas.CHI_DATA.frequency_order
record_schemas.CHI_DATA.sign_convention
record_schemas.CHI_DATA.total_includes
record_schemas.CHI_DATA.components.electron_scattering
record_schemas.CHI_DATA.components.bound_bound
record_schemas.CHI_DATA.components.bound_free
record_schemas.CHI_DATA.components.free_free
record_schemas.CHI_DATA.normalization
```

writer source로 이 의미가 증명되지 않으면 파일이 있어도 `BLOCKED_CHI_SCHEMA`다.
`NEG_OPAC`은 negative interval 교차진단에만 쓰고 값 원장은 CHI_DATA다.

## 11. 음성 대조와 expected FAIL

모든 poison은 baseline과 별도 child process, 한 poison만, 고유 marker를 사용한다.
marker 미발화, child rc 불일치, poisoned PASS, baseline FAIL은 wrapper 전체 FAIL이다.

### 11.1 L-4 필수 3종

| ID | poison | marker | 지금의 독립 witness/기대 FAIL | CHI 도착 뒤 기대 FAIL | child/wrapper rc |
|---|---|---|---|---|---|
| N1 | stimulated correction 제거 | `A2_08_NEG_STIM_OFF` | analytic inversion/BF fixture의 signed coefficient·rate identity FAIL | total/component E1 또는 negative sign FAIL | `4/0` |
| N2 | 한 BF edge를 canonical bin 하나 이동 | `A2_08_NEG_BF_EDGE_SHIFT` | partial-edge closed-form 및 edge digest FAIL | witness band EB 또는 cell error FAIL | `4/0` |
| N3 | 한 opacity channel 제거 | `A2_08_NEG_CHANNEL_DROP` | required-component manifest/fixture total FAIL | coverage/E1/EB FAIL | `4/0` |

N3는 total도 같이 다시 합해 closure만 통과하는 우회를 막기 위해 required-component
presence와 analytic total oracle를 함께 검사한다.

### 11.2 source/signed 계약 추가 음성 대조

| ID | poison | marker | 기대 거부 | child/wrapper rc |
|---|---|---|---|---|
| N4 | negative χ를 abs/0 | `A2_08_NEG_CHI_CLAMP` | negative interval sign digest FAIL | `4/0` |
| N5 | BF reemission을 old Planck sampler로 | `A2_08_NEG_PLANCK_REEMIT` | source-CDF distribution/consumer census FAIL | `4/0` |
| N6 | raw `jbar_line` 또는 CMF fallback | `A2_08_NEG_RAW_JBAR` | checked-view/static/generation FAIL | `5/0` |
| N7 | RF/line/pop stamp 하나 stale | `A2_08_NEG_STALE_SOURCE` | publish 0, stale counter >0 | `5/0` |
| N8 | replay commit에서 line block 제거 | `A2_08_NEG_REPLAY_LINELESS` | atomic dual-view invariant FAIL | `5/0` |

각 poison은 실제 값이 바뀐 witness id, before/after hash, 발화 count를 JSON에 남긴다.

## 12. 필수 selftest

새 CPU selftest는 최소 다음을 포함한다.

1. ES 단위와 exact-zero `n_e`.
2. two-level BB의 normal, exact-zero, inversion 세 경우와 signed tau bit round-trip.
3. profile integral 및 bin 경계에 걸친 선의 보존 부분 적분.
4. hydrogenic/piecewise-linear BF closed form, threshold partial bin, edge OOG.
5. BF stimulated bracket이 음수가 되는 fixture와 clamp poison.
6. FF의 `T_e`, `n_e`, ion-charge 합 및 invalid material view.
7. 각 cell/대역의 total-component closure `<=1e-10`.
8. VALID/EXACT_ZERO/UNSAMPLED/OOG/MISS/STALE/profile/Qhash 상태 전파.
9. 중간 shell 실패 시 opacity/source partial publish 0.
10. line source population/scattering mode, `chi=0 eta!=0` singular 상태, negative source.
11. heating base가 J이고 no-line bin에서 bitwise J와 같음; scalar poison FAIL.
12. reemission CDF 정규화, exact-zero total, fixed-seed distribution, RNG draw-count 불변.
13. replay J+line block 성공과 J/line 후보 각각의 failure injection 원자성.
14. `n_e` alias identity/generation 및 mutable mirror poison.
15. K-FRESH stale rejection과 signed tau 이후 generation 보존.
16. Z-INERT exact-zero가 component/source에서 phantom contribution을 만들지 않음.
17. 카운터 합계와 `nlte_free` summary 정확히 한 줄.
18. N1-N8 marker와 rc.

## 13. 회귀 전판과 배터리 Z 배선

구현 보고서는 실제 CLI를 각 script의 `--help`와 source에서 다시 확인하고 명령, rc,
artifact hash를 기록한다. 현재 존재하는 회귀 대상은 다음이다.

1. `make lumina`.
2. `python3 scripts/a2_01_census_contract.py check` 및 A2-08 addendum checker.
3. `scripts/a2_02_resolution_ladder.py`, `scripts/a2_02c_frequency_union.py`,
   `scripts/a2_02c_segment_replay.py`의 기존 implementation/negative gates.
4. A2-03 `selftest_a2_03_radiation_field`, `selftest_a2_03_producer_parity_fixture`,
   `scripts/a2_03_byte_parity.py`, `scripts/a2_03_callgraph_audit.py`.
5. A2-04 `selftest_a2_04_commit`, `selftest_a2_04_replay_commit`,
   `scripts/a2_04_l0_replay.py`, `scripts/a2_04_classic_debt_sweep.py`.
6. A2-05 `selftest_a2_05_bf_rate`, `scripts/a2_05_l1bf_gate.py`,
   `scripts/a2_05_chain_lane.py`; 기존 eligibility/BLOCKED를 보존.
7. A2-06 `selftest_a2_06_line_jbar`, `selftest_a2_06_dual_commit`,
   `scripts/a2_06_l1bb_gate.py`; `BLOCKED_MISSING_RATE_EXPORT`를 세탁하지 않음.
8. A2-07 `selftest_a2_07_population`, `scripts/a2_07_population_census.py`,
   `scripts/a2_07_population_gate.py`, `scripts/a2_07_classic_sweep.py`.
9. A2-08 static/selftest/internal closure/negative/L-4 BLOCKED assertion.
10. `python3 scripts/run_gate_battery.py` 전 36 case와 serial/parallel equivalence.

새 translation unit을 만들면 Makefile target만 추가해서 끝내지 않는다. 현재
`scripts/run_gate_battery.py`의 Z는 hard-coded source list를 네 번 가진다.

- `Z-validator`: `scripts/run_gate_battery.py:121-127`
- `Z-tau`: `:129-135`
- `Z-population`: `:137-144`
- `Z-canonical`: `:146-155`

새 CPU opacity/source TU를 **네 compile command 모두**에 link한다. 이어
`scripts/run_zinert_selftest.py`에 A2-08 component/source Z fixture 인자를 추가하고
definitions/result row를 추가한다. `scripts/run_gate_battery.py:22`의 Z expected row도
6에서 새 실제 수로 올린다. `cpu_link_sources()`를 쓰는 K/CP가 우연히 link된다는 사실은
Z 4곳 배선을 대신하지 않는다.

기존 canonical Z baseline은 다음을 유지한다.

```text
active_lines=2211572
active_tau_bit_differences=0
active_tau_fnv64=1cfbc8dba0b0f23f
```

단, signed-tau 물리식 때문에 active nonzero tau bytes가 의도적으로 바뀌면 구현 전에
changed-output allowlist에 line ids/inversion status를 등록하고 새 hash의 원인 diff를
제시해야 한다. Z-INERT의 exact-zero 및 active line set은 변하면 안 된다. 결과를 본 뒤
baseline을 무조건 갱신하지 않는다.

## 14. 단계 시작 전 changed-output allowlist

구현자는 값을 계산하기 전에 다음 좁은 범주와 exact identity를 manifest에 등록한다.

- stimulated inversion이 있던 `(line,shell)`의 signed tau/BB χ.
- BF stimulated correction이 0으로 잘렸던 `(route,shell,bin)`의 BF/total χ.
- component 분리 때문에 기존 mixed `chi_bf`/`eta_bf` layout에서 바뀌는 CPU host buffer.
- line-source fallback이 발화하던 `(line,shell)`의 source status/value.
- blanketed heating field와 그 직접 `H_photo_blanket` 입력.
- BF absorption event의 re-emitted frequency와 `next_line_id`; 방향 RNG는 불변.
- CMF replay의 line cache generation/hash/view status.
- electron-density pointer ownership과 CPU transport/formal ES read provenance.
- 새 component/source diagnostic artifact.

“전체 spectrum”, “모든 opacity”, “physics changed”는 허용목록이 아니다. A2-09/10/11
observable 변화는 이번 단계가 임의로 허용하지 않는다. 예상 밖 packet interaction,
bolometric luminosity, population, `T_e`, spectrum 변화는 원인 분석 전 회귀 FAIL이다.

## 15. 구현 순서

1. A2-08 migration-id/static census와 pre-change JSON을 먼저 고정한다.
2. checked material view(`T_e`,`n_e`, population stamps)와 opacity/source status를 정의한다.
3. single signed line-opacity helper를 만들고 두 Sobolev producer와 dump를 이관한다.
4. ES/BF/FF/BB component builder와 A2-05 partial-bin kernel 재사용을 구현한다.
5. transactional owner/view, closure, counters, `nlte_free` summary를 구현한다.
6. `n_e` CPU owner를 plasma transaction으로 단일화하고 `t_electrons`를 제거한다.
7. A2-08 query-set union과 replay line block atomic commit/view refresh를 구현한다.
8. line source checked view를 만들고 모든 production raw consumer를 이관한다.
9. blanketed heating base와 direct H-photo J read를 checked view로 이관한다.
10. reemission source CDF consumer를 이관하고 A2-09 경계를 manifest에 기록한다.
11. EPAY scalar split과 CMF formal/source direct reads를 제거한다.
12. signed handoff의 clamp/floor를 제거하고 unsupported event/overflow를 명시 실패로 만든다.
13. selftest, N1-N8, component artifact, L-4 gate의 BLOCKED assertion을 구현한다.
14. 원장 addendum과 `docs/CODEX_IMPL_A2_08.md`를 작성한다.
15. Makefile, Z 4곳, `run_zinert_selftest`, 2-node driver를 배선하고 전 회귀를 실행한다.

각 단계에서 빌드가 깨진 중간 상태를 다음 단계의 fallback으로 감추지 않는다.

## 16. 운전석 명령과 실행 제약

구현자는 다음 두 driver를 만들고 `set -euo pipefail`, source/input hash, 환경변수,
명령별 rc, artifact SHA-256, scratch 경로를 기록한다.

### 16.1 grammar-debug — build/static/deterministic

```bash
ssh grammar "ssh grammar-debug 'bash /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_a2_08_grammar_debug.sh'"
```

driver는 build, census, A2-03~08 CPU selftest, replay atomicity, internal closure,
negative wrappers를 실행한다. 로그인 노드에서는 빌드만 허용하고 test binary를 실행하지
않는다.

### 16.2 lageunha — full CPU battery와 lane

```bash
ssh lageunha uptime
ssh lageunha 'bash /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_a2_08_lageunha.sh'
```

driver는 36-case battery, serial/parallel equivalence, A2-05/06/07 상태, A2-08
CHAIN/ORACLE_INPUT wiring, N1-N8, classic sweep를 실행한다. CPU thread 수를 명시하고
oversubscription을 금지한다. `/gpfs` 입력은 read-only이고 결과는 workspace의
`validation/a2_08/`에 쓴다.

어느 노드에서도 `/usr/bin/time`을 쓰지 않는다. GPU/syn 실행, 덱 수정, `/gpfs` 수정,
commit, push는 금지한다.

## 17. §11 단계 회귀 대장

`validation/a2_08/A2_08_REGRESSION_LEDGER.jsonl`에는 정확히 한 JSON object를 남기고
`docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:689-710`의 필드를 모두 포함한다.

```text
stage_id=A2-08
contract=SPEC_A2_08_V1
source_tree_hash
input_manifest_hash
oracle_id
node
command
exit_status
new_layer_status={
  INTERNAL_OPACITY_SOURCE,
  L4:{CHAIN:BLOCKED_MISSING_CHI_DATA,ORACLE_INPUT:BLOCKED_MISSING_CHI_DATA}
}
all_previous_layer_statuses
negative_control_status={stim_off,bf_edge_shift,channel_drop,chi_clamp,
                         planck_reemit,raw_jbar,stale_source,replay_lineless}
coverage={truth_f_cov:null,reason:BLOCKED_MISSING_CHI_DATA,internal_component_presence}
metric_values={cell_closure,band_closure,negative_intervals,source_cdf_norm,
               replay_atomicity}
changed_output_allowlist
guard_hits
fallback_hits
rng_seed
mc_confidence
artifact_paths
driver_signoff
```

`metric_values`가 실측됐더라도 `truth_f_cov:null`인 상태에서 L-4를 PASS로 쓰지 않는다.
`driver_signoff`는 Codex 구현자와 fable 검수 상태를 분리한다.

## 18. 구현 보고서 필수 목차

`docs/CODEX_IMPL_A2_08.md`는 최소 다음을 포함한다.

1. 기준/최종 HEAD와 source-tree diff; commit/push하지 않았다는 기록.
2. 원장 migration id 1:1 대응표와 현행 파일:행.
3. 목록 밖 consumer census, allowlist, production forbidden read 0.
4. component 식·단위·frame·grid·generation schema.
5. 셸/빈/대역 self-closure 실제 최대값과 worst identity.
6. negative χ interval과 clamp/floor 제거 증거.
7. source validity, fallback 0, reemission CDF/RNG 보고.
8. replay line-block production wiring과 dual-view atomicity.
9. `n_e`/`T_e` owner와 CPU mirror 처분.
10. `[A2-08][OPACITY-SOURCE]` 실제 한 줄 및 counter 불변식.
11. N1-N8 poison/marker/witness/기대·실제 metric/child·wrapper rc.
12. CHAIN/ORACLE_INPUT 표, truth-side `f_cov=null`, L-4
    `BLOCKED_MISSING_CHI_DATA`.
13. 2-node 실제 명령, node, rc, artifact hash.
14. 전 회귀와 battery Z 배선 4곳+`run_zinert_selftest` 증거.
15. §11 정확히 한 행과 changed-output allowlist diff.
16. A2-09 인계: full eta/L3/L5, transition/redistribution, source-view promotion.

## 19. fable 자기검수 체크리스트

- [ ] A2-08 production call graph에서 `plasma->W`/`plasma->T_rad` read·taint가 0이다.
- [ ] raw `nlte->J_nu`, `cs->J`, `jbar_line*`, numeric `line_source_S` fallback이 0이다.
- [ ] 원장 9+3+4+1행이 migration id별 정확히 한 처분을 가진다.
- [ ] 원장 밖 line-source consumer와 replay line block을 모두 census했다.
- [ ] A2-07 population/partition/within-SL/Te/ne stamp가 모두 checked된다.
- [ ] EXACT_ZERO와 invalid/missing이 다른 상태다.
- [ ] 두 Sobolev 경로가 한 signed helper를 쓰고 K-FRESH를 보존한다.
- [ ] BF/BB/CMF handoff 어디에서도 negative χ/tau를 abs/0/floor하지 않는다.
- [ ] event probability와 signed net opacity가 분리됐고 negative net 자체는 valid다.
- [ ] total/es/bb/bf/ff self-closure 최대가 실제 `<=1e-10`이다.
- [ ] 등록 5대역 component 적분과 negative intervals가 artifact에 있다.
- [ ] CHI_DATA writer component schema가 없으면 `BLOCKED_CHI_SCHEMA`다.
- [ ] MEANOPAC/NEG_OPAC를 CHI truth로 쓰지 않았다.
- [ ] N1-N8이 각자 고유 marker, 실제 witness, 기대 child/wrapper rc를 가진다.
- [ ] 정상 lane fallback/raw/scalar/partial-publish/alias mismatch count가 0이다.
- [ ] `nlte_free` summary가 정확히 한 줄이고 기존 A2-05/06/07 summary를 깨지 않는다.
- [ ] 새 TU가 Z build 4곳과 `run_zinert_selftest`에 모두 연결됐다.
- [ ] 전 36-case battery와 A2-03~07 전판이 실제 status를 보존한다.
- [ ] L-4를 PASS로 적지 않고 `BLOCKED_MISSING_CHI_DATA`, rc 3으로 기록했다.
- [ ] A2-09 경계를 full eta/L3/L5로 명시했다.

## 20. A2-09 인계

A2-08이 완료되면 A2-09에 다음 checked 기반을 넘긴다.

- 동일 grid/generation의 signed `chi` components와 source-view validity.
- population/LineJbar/RadiationField에 결박된 line source와 reemission CDF consumer.
- BF/free-bound narrow source producer provenance와 미완전 channel coverage.
- eta에 섞여 있던 기존 BF/FF 배열의 분리 부채.
- macro-atom route/transition probability와 full emissivity component를 닫아야 할 목록.
- L-3/L-5가 계속 `BLOCKED_MISSING_ETA_DATA`인 실제 상태.

A2-09는 A2-08 source consumer를 다시 raw array나 Planck sampler로 되돌릴 수 없다.
반대로 A2-08 내부 source-CDF PASS를 L-3/L-5 물리 PASS로 승격할 수 없다.

## 21. 저작 시 실측

측정 시각의 repository HEAD는
`3ddd95c0de20abea3284ca326ce41b7968d4b26d`였다. 이 절의 줄번호는 그 작업트리에서
`nl -ba`와 `rg -n`으로 다시 확인했다.

1. `docs/A2_01_DISPOSITION_LEDGER.md`는 157행, 미분류 0이며 A2-07 addendum까지 있다.
2. A2-08 원장 역할은 `opacity_rate` 9행(`:71-79`), `opacity` 3행(`:144-146`)이다.
   emissivity 1행은 `:165`에서 아직 A2-09로 적혀 있어 개정 9 addendum 재배치가 필요하다.
3. A2-06 재배치 4행은 같은 원장 `:229-232`, CMF direct read 2곳은 `:242`에 있다.
4. current scalar direct hit는 `src` C/H에서 55개다. A2-08 핵심은
   `src/lumina_plasma.c:2624,2626,8043,11999-12000,12066,12073-12078,18305-18314`와
   `src/lumina_cmfgen.c:908,2144`다.
5. raw CMF source read는 여전히 `src/lumina_cmfgen.c:3153,3159`; replay commit은
   `:3390-3446`에서 line block을 채우지 않는다. MC commit은 `src/lumina_main.c:535`에서
   line block을 채운다.
6. signed 값을 바꾸는 핵심 hit는
   `src/lumina_plasma.c:2989,2994,7073,7076,7786,17518,17523`과
   `src/lumina_cmfgen.c:834,838,2010,2019,2992-2994,3077-3096`이다.
7. `electron_density/t_electrons` 텍스트 hit는 18개다. CPU mirror write는
   `src/lumina_main.c:147,712`, `src/lumina_plasma.c:6793`의 3곳이다.
8. `OpacityState.line_source_S`는 `src/lumina.h:217`에서 여전히 `<=0` fallback sentinel로
   선언되어 있고 production read가 plasma/CMF에 다수 남아 있다.
9. `scripts/run_gate_battery.py:22`는 D19/K7/Z6/CP4, 총 36행을 기대한다. Z hard-coded
   compile list는 `:121-155`의 4곳이고 `scripts/run_zinert_selftest.py`는 6 case다.
10. `docs/A2_00_OPHYS_PROFILE.json`은 CHI_DATA/INFO를 exact file로 요구하지만 현재
    workspace의 `docs`/`validation`에는 이름이 정확히 `CHI_DATA`, `CHI_DATA_INFO`인
    파일이 0개다. 따라서 저작 시 L-4 상태는 `BLOCKED_MISSING_CHI_DATA`다.
11. O-PHYS profile은 CHI units/frame을 요구하고 MEANOPAC/NEG_OPAC 대체를 금지하지만,
    component inclusion schema는 아직 필수 field가 아니다.
12. 이 명세 저작 중 `src`와 덱은 수정하지 않았고, 산출물은 이 파일 하나다.
