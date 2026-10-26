#ifndef LUMINA_NLTE_MODEL_MATERIAL_BRIDGE_H
#define LUMINA_NLTE_MODEL_MATERIAL_BRIDGE_H

#include "nlte_model_sum.h"
#include "nlte_population_candidate.h"

/* 실제 원자모형의 소유 객체와 private trial. 공개 solver/물질 게시와 분리한다.
 * BF/BB/충돌의 물리식은 기존 일반 원소 조립기를 그대로 사용한다. */
typedef struct NLTEModelMaterialBridge NLTEModelMaterialBridge;
typedef struct NLTEModelMaterialElement NLTEModelMaterialElement;
#if defined(__GNUC__)
#define LUMINA_MATERIAL_API __attribute__((visibility("default")))
#else
#define LUMINA_MATERIAL_API
#endif

LUMINA_MATERIAL_API NLTEModelMaterialBridge *nlte_model_material_bridge_open(const char *atomic_directory,
    const char *native_nodes,const char *occupation,const char *occupation_parameters);
LUMINA_MATERIAL_API void nlte_model_material_bridge_free(NLTEModelMaterialBridge *bridge);
LUMINA_MATERIAL_API int nlte_model_material_bridge_ion_index(
    const NLTEModelMaterialBridge *bridge,int Z,int charge);

/* columns는 원 native 파일과 같은 순서, SL 인구는 column-major [nc,ns].
 * 원소총수는 불변이고 ne는 off-root일 수 있다. thermo는 C에서 다시 만든다.
 * 주어진 occupation/분율은 C 파생값과 비교하는 독립 생산자 증거다. */
LUMINA_MATERIAL_API int nlte_model_material_bridge_prepare(NLTEModelMaterialBridge *bridge,
    uint64_t trial_id,const double *temperature,const double *electron_density,
    const int32_t *columns,size_t n_columns,const double *superlevel,
    const double *occupation,const double *fraction,
    const BfPointRateView *bf);

/* BB view는 이 원소의 전 활성 roster여야 하며 call 동안 살아 있어야 한다.
 * 그 완결성·공통 J의 신원은 봉인 생산자가 보장한다. */
LUMINA_MATERIAL_API NLTEModelMaterialElement *nlte_model_material_bridge_element_begin(
    NLTEModelMaterialBridge *bridge,uint64_t trial_id,int Z,int node,
    const NLTEModelBBView *bb);
LUMINA_MATERIAL_API int nlte_model_material_bridge_element_push(NLTEModelMaterialElement *element,
    uint64_t trial_id,const int32_t *lower,const int32_t *upper,
    const double *omega,size_t count);
LUMINA_MATERIAL_API int nlte_model_material_bridge_push_points(
    NLTEModelMaterialElement *const *elements,uint64_t trial_id,
    const int32_t *lower,const int32_t *upper,const double *omega,
    size_t n_points,size_t n_pairs);
/* result_population은 원 native column 순서의 [nc,ns] private 출력이다.
 * 실패 시 이 원소의 결과를 게시하지 않는다. ledger: BF photon,BB photon,
 * BF electron,직접 CI 냉각,BB+용해 냉각,입력 SE 최대 scaled 잔차,후보 SE 잔차.
 * 외부열원/FF/단열항이 없으므로 ledger를 전체 RE로 해석하면 안 된다. */
LUMINA_MATERIAL_API int nlte_model_material_bridge_element_finish(NLTEModelMaterialElement *element,
    uint64_t trial_id,double *result_population,size_t population_count,double ledger[7]);
LUMINA_MATERIAL_API int nlte_model_material_bridge_finish_points(
    NLTEModelMaterialElement *const *elements,uint64_t trial_id,size_t n_points,
    double *result_population,size_t population_count,double *ledger);
LUMINA_MATERIAL_API void nlte_model_material_bridge_element_free(NLTEModelMaterialElement *element);
/* 전원소·전점 후보가 완성됐을 때만 charge 잔차와 실제 내부에너지를 만든다.
 * energy는 [ns,3]: nuclei density, energy density, energy/nucleus다.
 * 기존 BF/BB 입력을 소비하는 phase는 닫으며 공개 물질은 바꾸지 않는다. */
LUMINA_MATERIAL_API int nlte_model_material_bridge_finalize(
    NLTEModelMaterialBridge *bridge,uint64_t trial_id,const double *population,
    size_t population_count,double *energy,double *charge_residual);

#endif
