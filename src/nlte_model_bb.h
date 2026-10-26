#ifndef LUMINA_NLTE_MODEL_BB_H
#define LUMINA_NLTE_MODEL_BB_H

#include "nlte_model_se.h"

/* 동일 복사장·물질의 한 원래점에서 완성된 모든 native 복사 선. 배열은 caller
 * 소유이고 n_lines 길이다. 준비하지 않은 점은 0율이 아니라 거부한다.
 * 수치율은 원 FL 입자당이며 조립 때 within-SL 비율을 한 번 적용한다. */
typedef struct {
    size_t n_levels, n_lines, n_shells, native_shell;
    uint64_t radiation_generation, material_generation;
    char material_model_sha256[65];
    const int32_t *lower, *upper;
    const double *frequency_Hz, *absorption, *stimulated, *spontaneous;
    const uint8_t *status; /* 1=VALID,2=EXACT_ZERO; 나머지는 거부 */
} NLTEModelBBView;

/* 원소별 radiative generator와 복사 선의 h*nu 가스 열량만 담는다.
 * same-SL 선은 population generator에는0, 열량에는 남긴다.
 * collisional BB, BF/pseudo, 자유-자유와 전체 RE는 포함하지 않는다. */
typedef struct {
    int Z, shell;
    uint64_t material_generation, radiation_generation;
    size_t n_variables, lines, intra_super_lines;
    double *radiative, *radiative_heat;
    double *native_thermal_heat; /* 선택적 native 열장부; radiative_heat와 구분 */
} NLTEModelBBBlock;

int nlte_model_element_bb_assemble(NLTEModelBBBlock *out,
    const NLTEModelElement *element,const NLTEConfig *nlte,
    const AtomicData *atom,const PlasmaState *plasma,
    const NLTEModelBBView *view);
/* 기존 BB view ABI를 유지하는 별도 진입점. caller가 같은 trial의 SCL_LN과
 * POP_ATOM 분기를 평가한 선 계수와 정책 신원을 소유한다. NULL이면 미생산.
 * 물리율/광자 에너지는 바꾸지 않는다. */
int nlte_model_element_bb_assemble_with_native_thermal(NLTEModelBBBlock *out,
    const NLTEModelElement *element,const NLTEConfig *nlte,
    const AtomicData *atom,const PlasmaState *plasma,
    const NLTEModelBBView *view,const double *native_thermal_factor);
void nlte_model_bb_block_free(NLTEModelBBBlock *block);

#endif
