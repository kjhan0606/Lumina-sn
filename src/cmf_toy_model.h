#ifndef LUMINA_CMF_TOY_MODEL_H
#define LUMINA_CMF_TOY_MODEL_H
#include "cmf_ejecta.h"
#include "line_jbar.h"
#ifdef __cplusplus
extern "C" {
#endif

/* 실제 toy06 시간 운전의 초기 물질 소유자. 처음0초기화, 복사/이동 금지:
 * NLTE의 Q/E 포인터는 이 객체의 멤버를 참조한다. prepare는1회 bootstrap이다.
 * Saha/Boltzmann predictor이지 정상 NLTE 또는 물리 시간 이력이 아니다. */
typedef struct {
    CMFToyEjecta ejecta;
    Geometry geometry;
    PlasmaState plasma;
    AtomicData atom;
    OpacityState opacity;
    NLTEConfig nlte;
    BFOpacity bf;
    LineJbarQSet qset;
    LineJbarESet eset;
    double *shell_storage;
    int prepared;
} CMFToyModel;
int cmf_toy_model_prepare(CMFToyModel*,const char *ejecta_path,const char *atomic_dir,
                          int n_shells,const double mass_amu[CMF_TOY_ELEMENTS]);
void cmf_toy_model_free(CMFToyModel*);
int cmf_toy_model_write_seed(const CMFToyModel*,const char *csv_path);
int cmf_toy_bootstrap_main(int argc,char **argv);
int cmf_toy_initial_main(int argc,char **argv);
#ifdef __cplusplus
}
#endif
#endif
