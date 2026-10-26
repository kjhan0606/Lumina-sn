#ifndef LUMINA_NLTE_MODEL_SE_H
#define LUMINA_NLTE_MODEL_SE_H

#include "lumina.h"

/* 원소 하나의 실제 SL+선언 terminal 좌표. 원90점 주소는 유지한다.
 * initial_population은 잔차 평가의 시작 인구이지 새 SE 해가 아니다. */
typedef struct {
    int Z, shell;
    size_t n_variables, n_bound, n_terminals, n_super_total, n_ion_pops;
    int *super_to_row, *terminal_to_row;
    double *initial_population;
    double total_nuclei;
    uint64_t material_generation;
} NLTEModelElement;

int nlte_model_element_init(NLTEModelElement *element,const NLTEConfig *nlte,
    const AtomicData *atom,const PlasmaState *plasma,int Z,int shell);
void nlte_model_element_free(NLTEModelElement *element);

/* BF 블록만의 물리율이다. BB/CE/DR/NT/FF/전체 RE가 완성됐다는 뜻이 아니다.
 * radiative/collisional은 pre-constraint column-major generator이고,
 * radiative_heat는 복사 BF 전자 운동에너지 계수(erg s^-1/입자)만 담는다.
 * CI/C3 열량과 전체 RE ledger는 아직 포함하지 않는다. */
typedef struct {
    int Z, shell;
    uint64_t material_generation, radiation_generation;
    size_t n_variables, routes, terminal_routes;
    double *radiative, *collisional, *radiative_heat;
} NLTEModelBFBlock;

int nlte_model_element_bf_assemble(NLTEModelBFBlock *out,
    const NLTEModelElement *element,const NLTEConfig *nlte,
    const AtomicData *atom,const PlasmaState *plasma);
void nlte_model_bf_block_free(NLTEModelBFBlock *block);

#endif
