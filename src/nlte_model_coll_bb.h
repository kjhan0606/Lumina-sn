#ifndef LUMINA_NLTE_MODEL_COLL_BB_H
#define LUMINA_NLTE_MODEL_COLL_BB_H

#include "nlte_model_se.h"

/* primary PHOT의 FL LTE / 다음 이온 ground-SL LTE 비를 소유한다.
 * PHOT 분기확률은 용해 충돌에 다시 곱하지 않는다. */
typedef struct {
    int Z, shell;
    size_t n_levels, n_variables;
    uint64_t material_generation;
    char material_model_sha256[65];
    double temperature_K, electron_density_cm3;
    int *ground_row;
    double *saha, *threshold_Hz;
} NLTEModelDissolutionContext;

int nlte_model_dissolution_init(NLTEModelDissolutionContext *out,
    const NLTEModelElement *element,const NLTEConfig *nlte,
    const AtomicData *atom,const PlasmaState *plasma);
void nlte_model_dissolution_free(NLTEModelDissolutionContext *context);

/* 원소 하나·원 물질 한 점의 전 FL 쌍 충돌강도. 이온별 FL 순서대로
 * (i,j), i<j의 사전식 전체 목록이어야 한다. 표/van Regemorter/OMEGA_SET
 * 중 어느 경로로 강도를 얻었는지는 caller의 봉인 생산 기록에 남긴다. */
typedef struct {
    int Z;
    size_t n_levels, n_shells, native_shell, n_pairs;
    uint64_t material_generation;
    char material_model_sha256[65];
    double temperature_K, electron_density_cm3;
    const int32_t *lower, *upper;
    const double *omega;
    const NLTEModelDissolutionContext *dissolution;
} NLTEModelCollisionBBView;

/* CMFGEN SUBCOL_MULTI_V6의 cross-SL bound-bound 부분을 투영한다.
 * 선택적 dissolution 컨텍스트가 있으면 same-SL을 포함한 전 FL쌍의
 * 용해 이온화와 상세균형 역률도 더한다. 직접 CI/C3는 BF 블록 소유다.
 * collisional_heat는 가스 냉각 부호(+ 들뜸, - 탈들뜸) 계수다. */
typedef struct {
    int Z, shell;
    uint64_t material_generation;
    size_t n_variables, pairs, cross_super_pairs, intra_super_pairs;
    double *collisional, *collisional_heat;
    size_t dissolution_pairs, intra_super_dissolution_pairs;
    double *dissolution, *dissolution_heat;
} NLTEModelCollisionBBBlock;

int nlte_model_element_collision_bb_assemble(NLTEModelCollisionBBBlock *out,
    const NLTEModelElement *element, const NLTEConfig *nlte,
    const AtomicData *atom, const PlasmaState *plasma,
    const NLTEModelCollisionBBView *view);
void nlte_model_collision_bb_block_free(NLTEModelCollisionBBBlock *block);

/* 한 trial의 원소 블록을 순서대로 받아 완결 뒤에만 게시한다.
 * borrowed element/atom/nlte/plasma/용해 컨텍스트는 finish까지 불변이어야 한다.
 * trial_id는 서로 다른 후보의 블록 혼합을 거부하는 caller 소유 신원이다. */
typedef struct NLTEModelCollisionBBStream NLTEModelCollisionBBStream;
int nlte_model_collision_bb_stream_begin(NLTEModelCollisionBBStream **stream,
    const NLTEModelElement *element,const NLTEConfig *nlte,
    const AtomicData *atom,const PlasmaState *plasma,
    const NLTEModelCollisionBBView *metadata,uint64_t trial_id);
int nlte_model_collision_bb_stream_push(NLTEModelCollisionBBStream *stream,
    uint64_t trial_id,const int32_t *lower,const int32_t *upper,
    const double *omega,size_t count);
int nlte_model_collision_bb_stream_finish(NLTEModelCollisionBBStream *stream,
    uint64_t trial_id,NLTEModelCollisionBBBlock *out);
void nlte_model_collision_bb_stream_free(NLTEModelCollisionBBStream *stream);

#endif
