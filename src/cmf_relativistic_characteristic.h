#ifndef LUMINA_CMF_RELATIVISTIC_CHARACTERISTIC_H
#define LUMINA_CMF_RELATIVISTIC_CHARACTERISTIC_H

typedef enum {
    CMF_REL_OK=0,
    CMF_REL_INVALID_INPUT=1,
    CMF_REL_NONFINITE=2,
    CMF_REL_NEEDS_RED_UPWIND=3
} CMFRelStatus;

/* 원본 DERIVS/CHARACTERISTICS_V2의 정상 상대론 특성선 계수.
 * r,s는 cm, beta=v/c, mu는 comoving 방향 코사인이다.
 * dlog(nu)/ds=-b, dI_nu/ds=eta-(chi+3b)I_nu이며 주파수 결합은 별도다.
 * 이것만으로 전체 특성선·각도 적분·경계·이류 수송을 구현한 것은 아니다. */
typedef struct {
    double gamma;
    double dr_ds;
    double dmu_ds;
    double b_cm_inverse;
} CMFRelCharacteristic;

CMFRelStatus cmf_rel_characteristic_coefficients(
    double radius_cm,double mu,double beta,double dbeta_dr_cm_inverse,
    CMFRelCharacteristic *result);

/* 원본 REL_VARIABLES의 blue-upwind 결합을 raw 계수로 반환한다.
 * chi_eff=chi+(nu_dnu+3)b, eta_eff=eta+nu_dnu*b*I_blue.
 * nu_dnu는 호출자가 정한 실제 이산 계수다. 원본 formal은1/dlog(nu)를
 * 공급한다. eta/chi를 만들지 않으므로 chi_eff=0과 음수도 보존한다.
 * b<0이고 주파수 결합이 있으면 반대쪽 upwind가 필요하며 강제 수정하지 않는다. */
typedef struct {
    double chi_effective_cm_inverse;
    double eta_effective;
    double blue_coupling_cm_inverse;
} CMFRelFrequencyTerms;

CMFRelStatus cmf_rel_frequency_raw_terms(
    double chi_cm_inverse,double eta,double nu_dnu,double b_cm_inverse,
    double intensity_blue,CMFRelFrequencyTerms *result);

/* 검증된 homologous beta=r/(ct)의 실험실 직선 p,z 재매개화.
 * ds=D dz, D=gamma(1-z/(ct)), mu=(z/r-beta)/(1-z/(ct)).
 * from_z<=to_z가 광자 진행 방향이며 delta_log_nu=log(D_from/D_to).
 * ds와 주파수 이동은 구간의 해석 적분이다. P0 chi/eta 자체의 공간오차를
 * 제거하거나 전체 상대론 RTE/경계 동등성을 증명하는 함수는 아니다. */
typedef struct {
    double comoving_length_cm;
    double delta_log_nu;
    double mu_from;
    double mu_to;
} CMFRelHomologousSegment;

CMFRelStatus cmf_rel_homologous_segment(
    double impact_cm,double from_z_cm,double to_z_cm,double epoch_seconds,
    CMFRelHomologousSegment *result);

#endif
