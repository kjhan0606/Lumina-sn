#ifndef LUMINA_BF_RATE_JNU_H
#define LUMINA_BF_RATE_JNU_H

/* A2-05 (SPEC_A2_05_V2): bound-free photoionization rate integrated directly
 * on the canonical RadiationField view.
 *
 *   Gamma = 4*pi * Integral_{nu_th}^{nu_max} J_nu * sigma(nu) / (h*nu) dnu
 *
 * J_nu is a bin average (constant within a bin); sigma is tabulated and
 * integrated piecewise-linearly against 1/(h*nu) inside each overlap interval,
 * so the threshold partial bin uses [max(nu_th, nu_lo), nu_hi] exactly (gate
 * contract 4).  Validity propagates per R6: within the integration range the
 * sigma-weighted contribution share of non-VALID bins (w_miss) decides the
 * result state; no value substitution ever happens.
 */

#include <stddef.h>
#include <stdint.h>

#include "radiation_field.h"

typedef enum {
    BF_RATE_VALID = 1,
    BF_RATE_EXACT_ZERO = 2,
    BF_RATE_UNSAMPLED = 3,
    BF_RATE_OUT_OF_GRID = 4,
    BF_RATE_STALE = 5
} BfRateValidityState;

typedef struct {
    double gamma;              /* s^-1; meaningful only when state==VALID or EXACT_ZERO */
    BfRateValidityState state;
    double w_miss;             /* sigma-weighted non-VALID contribution share */
    uint64_t sample_count;     /* summed estimator counts over judged bins */
    double gamma_poisson_var;  /* delta-method Var(gamma) from per-bin counts
                                * (ORDER 6.3 CI); exactly 0 for deterministic
                                * commits, whose VALID bins carry count==0 */
} BfRateResult;

/* True only when a result carries a finite, nonnegative rate whose missing
 * sigma-weighted contribution is within the declared R6 tolerance. */
int bf_rate_result_is_consumable(const BfRateResult *result);

/* 점값 전용: 전역 quadrature_Hz를 그대로 소비한다. bin 평균이 아니다.
 * forward는 하위 FL 입자당, inverse는 상위 SL 입자당 계수다.
 * PHOT 확률은 이 적분 밖에서 한 번만 적용한다. */
typedef struct {
    double photo, spontaneous, stimulated; /* s^-1 */
    double heat_photo, heat_spontaneous, heat_stimulated; /* erg s^-1 */
} BfPointRates;

int bf_rate_point_contract(size_t n, const double *frequency_Hz,
    const double *quadrature_Hz, const double *J_nu,
    const double *forward_sigma, const double *inverse_sigma,
    const double *inverse_eta, double threshold_Hz, BfPointRates *out);

/* 같은 Milne 물리식을 경로당 한 번만 정규화해 전ν 블록에서 직접 적분한다.
 * inverse_sigma = forward_sigma * milne_prefactor *
 * exp[-inverse_kT_Hz * (frequency_Hz-threshold_Hz)].
 * 중간 stimulated/eta 배열을 만들지 않으며 기존 점값 C 적분과 동등해야 한다. */
int bf_rate_point_contract_milne(size_t n, const double *frequency_Hz,
    const double *quadrature_Hz, const double *J_nu,
    const double *forward_sigma, double threshold_Hz,
    double milne_prefactor, double inverse_kT_Hz, BfPointRates *out);

/* 여러 원래점의 같은 주파수 블록을 한 σ 평가로 소비한다. 모든 율은
 * route-major [n_routes*n_points], J는 point-major [n_points*n_frequency],
 * bound_sigma는 [n_routes*n_frequency]다. 확률/FL분율은 밖에서 한 번 적용한다.
 * evaluation_Hz는 σ/용해 평가점이며 열적 지수·열량에는 actual frequency를 쓴다.
 * 폭32/kT 제한은 지수의 블록 표현만을 위한 것으로 주파수를 버리지 않는다.
 * 실패 시 두 출력 배열은 보존된다. material_b/a/x는 기존 용해 fit의 점별 값.
 * dissolved=0이면 해당 세 벡터를 NULL로 줄 수 있다. */
typedef struct {
    size_t n_frequency, n_routes, n_points;
    const double *frequency_Hz, *evaluation_Hz, *quadrature_Hz, *J_nu;
    const double *bound_sigma, *threshold_Hz, *milne_prefactor, *inverse_kT_Hz;
    int dissolved;
    const uint8_t *dissolution_primary;
    const double *sigma_edge, *z_screen, *material_b, *material_a, *material_x;
} BfMaterialRateBlock;

#if defined(__GNUC__)
#define BF_MATERIAL_API __attribute__((visibility("default")))
#else
#define BF_MATERIAL_API
#endif
BF_MATERIAL_API int bf_rate_material_block_contract(const BfMaterialRateBlock *block,
    BfPointRates *bound, BfPointRates *total);

/* 같은 구현의 용해항만 적분한다. bound_sigma는 NULL이어도 되며 첫 출력은0.
 * forward/역률 모두 새 trial의 b/a/x와 Milne prefactor를 소비한다.
 * 기존 물질에서 total-bound를 빼서 새 용해항으로 재사용하지 않는다. */
BF_MATERIAL_API int bf_rate_material_pseudo_block_contract(const BfMaterialRateBlock *block,
    BfPointRates *zero_bound, BfPointRates *pseudo);

/* 원자 경로 순서의 불변 snapshot. T/ne가 바뀌면 재생산해야 하며
 * stale snapshot을 bin 경로로 대체하지 않는다. 배열은 caller 소유다. */
typedef struct {
    size_t n_levels, n_routes, n_shells;
    uint64_t radiation_generation, material_generation;
    /* occupation 모형에서는 partition stamp의 실제 물질 hash를 값으로 복사.
     * mutable stamp의 문자열 포인터를 빌려 stale 율이 함께 갱신되지 않게 한다. */
    char material_model_sha256[65];
    const int *offset;             /* n_levels+1 */
    const int *target;             /* n_routes; upper SL의 실제 FL anchor */
    const int *terminal_ion;       /* n_routes; target=-1일 때 실제 ion-pop index */
    const double *probability, *threshold_Hz, *sigma_edge;
    const int *phot_grid;          /* 원 PHOT grid; CI 경로 선택용 */
    const double *saha_lower_per_upper; /* route-major; 점유확률 포함 */
    const double *temperature, *electron_density; /* n_shells */
    const uint8_t *shell_ready;   /* NULL=전체; 0=미계산 원래점, exact zero 아님 */
    const BfPointRates *rates;     /* route-major [n_routes*n_shells] */
} BfPointRateView;

/* 확률만 한 번 적용한 물리율. FL→SL fraction은 행렬 조립 때 적용한다.
 * terminal은 가짜 FL이 아니라 실제 ion-pop scalar로 식별한다. */
typedef struct {
    int target_level, target_ion;
    BfPointRates radiative;
    double collisional_ionization, three_body_recombination;
} BfPointTransition;

int bf_rate_point_get(const BfPointRateView *view, size_t route, size_t shell,
    uint64_t generation, double temperature, double electron_density,
    BfPointRates *out);

/* 빈 CSR 행은 실제 0개 경로이며 Kramers fallback을 뜻하지 않는다. */
int bf_rate_point_row(const BfPointRateView *view, size_t n_levels,
    size_t n_shells, size_t lower, uint64_t material_generation,
    size_t *begin, size_t *end);

/* sigma tabulation: photo cross-section points (nu ascending, cm^2), zero
 * below nu_threshold.  Matches the existing tabulated interpretation used by
 * the A2-02C builder bf kernel (4*pi*sigma/(h*nu), zero below threshold). */
typedef struct {
    size_t n_points;
    const double *nu;          /* Hz, ascending */
    const double *sigma;       /* cm^2 */
    double nu_threshold;       /* Hz */
} BfCrossSection;

/* w_miss tolerance below which missing bins are ignored (R6). */
#define BF_RATE_W_MISS_TOLERANCE 1.0e-3

int bf_rate_gamma_from_view(const RadiationFieldView *view, size_t shell,
                            const BfCrossSection *sigma, BfRateResult *out);

/* Read one bin from the same published field used by the BF rate integral.
 * Returns 0 for VALID/EXACT_ZERO, 1 for a non-consumable bin (out is zero),
 * and -1 for malformed arguments or inconsistent values. */
int bf_rate_view_jnu(const RadiationFieldView *view, size_t shell, size_t bin,
                     double *out);

/* 다른 격자의 번호를 쓰지 않고 활성 주파수 구간의 log-weighted J를 읽는다.
 * 완전 coverage/유효성을 요구하며 실패 시 반환값을 소비해서는 안 된다.
 * 반응은 기존 getter와 동일하다: 0=소비가능, 1=불가용, -1=잘못된 인자. */
int bf_rate_view_log_band_jnu(const RadiationFieldView *view, size_t shell,
                              double nu_lo, double nu_hi, double *out);

/* Legacy-grid adapter shared by production (nlte_bf_gamma_canonical) and the
 * L-1bf gate fixture, so both run byte-identical arithmetic: the per-bin
 * sigma_row (or the Kramers sigma_0*(nu_th/nu_c)^3 evaluation at legacy bin
 * centres) is re-encoded as a bin-constant step tabulation over the nfb-bin
 * log grid and integrated against the canonical view.  A stored full-bin
 * average in the physical threshold bin is rescaled onto its active
 * [nu_th,nu_hi] support while preserving integral sigma*dnu.  node_nu and
 * node_sigma are caller scratch of at least 2*nfb entries each. */
int bf_rate_gamma_legacy_grid(const RadiationFieldView *view, size_t shell,
                              int nfb, double nu_min, double d_log_nu,
                              const double *sigma_row, double sigma_0,
                              double nu_thresh,
                              double *node_nu, double *node_sigma,
                              BfRateResult *out);

/* Log-frequency kernel weight for a full-bin-average CMFGEN sigma row.
 * If nu_thresh cuts the bin, the stored average is first renormalized onto
 * [nu_thresh, nu_hi], exactly as bf_rate_gamma_legacy_grid does, then
 * integrated against dnu/nu. Returns zero for an inactive/invalid bin. */
double bf_rate_sigma_log_weight(double sigma_bin_average, double nu_lo,
                                double nu_hi, double nu_thresh);

#endif
