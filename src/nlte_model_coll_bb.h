#ifndef LUMINA_NLTE_MODEL_COLL_BB_H
#define LUMINA_NLTE_MODEL_COLL_BB_H

#include "nlte_model_se.h"

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
} NLTEModelCollisionBBView;

/* CMFGEN SUBCOL_MULTI_V6의 cross-SL bound-bound 부분만 투영한다.
 * same-SL/용해 이온화, CI/C3, 전체 RE와 새 SE 해는 포함하지 않는다.
 * collisional_heat는 가스 냉각 부호(+ 들뜸, - 탈들뜸) 계수다. */
typedef struct {
    int Z, shell;
    uint64_t material_generation;
    size_t n_variables, pairs, cross_super_pairs, intra_super_pairs;
    double *collisional, *collisional_heat;
} NLTEModelCollisionBBBlock;

int nlte_model_element_collision_bb_assemble(NLTEModelCollisionBBBlock *out,
    const NLTEModelElement *element, const NLTEConfig *nlte,
    const AtomicData *atom, const PlasmaState *plasma,
    const NLTEModelCollisionBBView *view);
void nlte_model_collision_bb_block_free(NLTEModelCollisionBBBlock *block);

#endif
