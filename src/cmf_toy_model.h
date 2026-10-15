#ifndef LUMINA_CMF_TOY_MODEL_H
#define LUMINA_CMF_TOY_MODEL_H
#include "cmf_ejecta.h"
#include "line_jbar.h"
#include "cmf_time_epoch.h"
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
/* 초기 공간 해상도 시험용 선택 경로. 기본 API와 생산 기본값은 바꾸지 않는다. */
int cmf_toy_model_prepare_refined(CMFToyModel*,const char *ejecta_path,const char *atomic_dir,
                          int base_shells,const double mass_amu[CMF_TOY_ELEMENTS],double temperature_ratio);
void cmf_toy_model_free(CMFToyModel*);
int cmf_toy_model_write_seed(const CMFToyModel*,const char *csv_path);
/* 초기 반복 전용: 물리 이력/H/수렴 표지가 없다. identity는 runner가 동일
 * binary/input/설정에서 만든 SHA256이다. read는 새 prepare 소유자만 받는다. */
int cmf_toy_initial_iterate_write(const CMFToyModel*,const char *path,const char *identity,int iteration);
int cmf_toy_initial_iterate_read(CMFToyModel*,const char *path,const char *identity,int *iteration);
int cmf_toy_model_write_material(const CMFToyModel*,const char *csv_path,const char *status);
/* 새로 prepare한 소유자에 실제 과거를 연결한다. 현재 반복 초기값일 뿐
 * 수렴한 새 시점은 아니다. projected는 모델보다 오래 살아야 한다.
 * 과거는 불변. 갱신 후 실패시 m->prepared=0이므로 모델을 버리고 재구축한다. */
int cmf_toy_model_begin_epoch(CMFToyModel*,const CMFPhysicalTimeState*,
    const CMFEjectaEpoch*,NLTEPhysicalHistory *projected);
int cmf_toy_bootstrap_main(int argc,char **argv);
int cmf_toy_initial_main(int argc,char **argv);
int cmf_toy_step_main(int argc,char **argv);
#ifdef __cplusplus
}
#endif
#endif
