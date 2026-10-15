#ifndef CMF_GAMMA_TRANSPORT_H
#define CMF_GAMMA_TRANSPORT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* 동결된 구형 셸의 감마선 수송. 방출률은 좌표 시간/부피당 광자수이고
 * 선 에너지는 국소 공이동계 값이다. age=0은 정적 해석시험 전용이다.
 * 실제 동형팽창은 r/(c*age)<0.5 범위. 전자에는 속박 전자도 포함한다.
 * 양전자 운동 에너지·Spencer–Fano 열화는 이 모듈 밖에서 처리한다. */
typedef struct {
    int n_shells,n_lines;
    double homologous_age_s;
    const double *radius_edges_cm;       /* n_shells+1, 중심0 필수 */
    const double *total_electrons_cm3;   /* n_shells */
    const double *sum_number_Z5_cm3;     /* n_shells, sum(n_element Z^5) */
    const double *line_energy_MeV;       /* n_lines */
    const double *photon_emissivity;     /* n_lines*n_shells, photons/s/cm^3 */
} CMFGammaInput;
typedef struct {
    int n_shells;
    uint64_t packets,scatterings,absorptions,escapes,null_events,boundary_crossings;
    double emitted_comoving_erg_s,emitted_lab_erg_s;
    double deposited_comoving_erg_s,deposited_lab_erg_s,escaped_lab_erg_s;
    double deposited_comoving_standard_error;
    double lab_energy_relative_error;
    double *storage,*deposition_erg_cm3_s,*deposition_standard_error,*positive_packet_count;
    /* 0표본 셸의0분산은 물리적0 또는 유한 신뢰구간을 뜻하지 않는다. */
} CMFGammaResult;
double cmf_gamma_klein_nishina_cm2(double energy_MeV);
double cmf_gamma_photoelectric_cm1(double energy_MeV,double sum_number_Z5_cm3);
void cmf_gamma_result_free(CMFGammaResult*);
/* 결과 객체는 첫 사용 전0초기화. 실패시 기존 결과 보존. seed와 packet 수로 재현된다. 광자 도중의 에너지
 * cutoff/국소 강제침적/음수 보정은 없다. 사건 한도 초과는 실패로 반환한다. */
int cmf_gamma_transport(const CMFGammaInput*,uint64_t packets,uint64_t seed,CMFGammaResult*);
#ifdef __cplusplus
}
#endif
#endif
