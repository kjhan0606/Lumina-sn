#ifndef CMF_NONTHERMAL_H
#define CMF_NONTHERMAL_H
#ifdef __cplusplus
extern "C" {
#endif
/* Spencer–Fano 차분형의 보존적 하강 에너지 수송. 경로마다 실제 lower
 * population을 받는다. ion은1개의 secondary를 만들고 excitation은 만들지
 * 않는다. Auger cascade/준위 연결은 호출자 몫이며 이 함수가 꾸며내지 않는다. */
enum { CMF_NT_EXCITATION=0, CMF_NT_IONIZATION=1 };
typedef struct {
    int kind;
    double threshold_eV,lower_population_cm3;
    const double *cross_section_cm2; /* n_energy, 같은 에너지 노드 */
} CMFNonthermalRoute;
typedef struct {
    int n_energy,n_routes;
    const double *energy_eV; /* 양의 증가 노드, 마지막에1eV 에너지 주입 */
    const double *coulomb_loss_eV_cm; /* n_energy;0은 해석시험에서 허용 */
    const CMFNonthermalRoute *routes;
} CMFNonthermalInput;
typedef struct {
    int n_energy,n_routes;
    double *storage,*path_cm_per_eV,*events_per_eV,*coefficient_cm3_per_eV;
    double heating_fraction,ionization_fraction,excitation_fraction;
    double ionizations_per_eV,thermal_electrons_per_eV;
    double energy_error,electron_number_error;
} CMFNonthermalResult;
/* 첫 사용 전0초기화. 결과는 실패시 보존. 분율 재정규화/음수 clipping 없음.
 * coefficient * 침적[eV/cm3/s]가 lower particle당 rate[s^-1]이다. */
int cmf_nonthermal_solve(const CMFNonthermalInput*,CMFNonthermalResult*);
void cmf_nonthermal_free(CMFNonthermalResult*);
double cmf_nonthermal_coulomb(double energy_eV,double free_electrons_cm3);
double cmf_nonthermal_arnaud(double energy_eV,double threshold_eV,const double abcd[4]);
double cmf_nonthermal_bethe(double energy_eV,double transition_eV,double f_lu,int ion_charge);
double cmf_nonthermal_omega(double energy_eV,double transition_eV,double omega,double lower_g);

enum { CMF_NT_MAX_TAB=128, CMF_NT_MAX_ROUTES=1024 };
typedef struct {
    int Z,charge,n,l,n_tab,source_line;
    double threshold_eV,abcd[4],tab_energy[CMF_NT_MAX_TAB],tab_sigma[CMF_NT_MAX_TAB];
    char terms[256]; /* 원본 lower(*),upper,& 표시 보존; 아직 원자 연결 아님 */
} CMFNonthermalAtomicRoute;
typedef struct { int n_routes; CMFNonthermalAtomicRoute *routes; } CMFNonthermalAtomicData;
int cmf_nonthermal_atomic_read(CMFNonthermalAtomicData*,const char *path);
void cmf_nonthermal_atomic_free(CMFNonthermalAtomicData*);
double cmf_nonthermal_atomic_sigma(const CMFNonthermalAtomicRoute*,double energy_eV);
#ifdef __cplusplus
}
#endif
#endif
