#ifndef LUMINA_NLTE_MODEL_SUM_H
#define LUMINA_NLTE_MODEL_SUM_H

#include "nlte_model_se.h"
#include "nlte_model_bb.h"
#include "nlte_model_coll_bb.h"
#include "population_contract.h"

/* 같은 원소·원래점·물질 세대와 같은 BF/BB 복사 세대의 네 물리 블록을 합한다.
 * fixed_residual=A*n_initial은 입력 물질의 잔차다. 이 API는 입력 BF의
 * 전주파수 완결성, 모든 원소의 원자 채널, 전하/RE 및 자기일관성을 인증하거나
 * 새 인구를 게시하지 않는다. 해 판정은 상위 완결성 계약 뒤에만 가능하다. */
typedef struct {
    int Z, shell;
    uint64_t material_generation, radiation_generation;
    size_t n_variables;
    double *generator, *fixed_residual;
} NLTEModelCombinedGenerator;

int nlte_model_element_generator_combine(NLTEModelCombinedGenerator *out,
    const NLTEModelElement *element,const NLTEModelBFBlock *bf,
    const NLTEModelBBBlock *bb,const NLTEModelCollisionBBBlock *collision_bb);
void nlte_model_combined_generator_free(NLTEModelCombinedGenerator *combined);

/* 같은 원소·물질·복사 세대의 고정장 정상 SE 후보를 private 배열에만 푼다.
 * 이 호출은 입력 블록의 전주파수/전채널 완결성을 인증하지 않으며, 전하·RE·
 * 수송 공동수렴이나 공개 물질 게시도 하지 않는다. 실패 시 출력은 불변이다. */
PopulationStatus nlte_model_element_fixed_field_candidate(
    const NLTEModelCombinedGenerator *combined,const NLTEModelElement *element,
    double *candidate,PopulationGeneratorSolveDiagnostic *diagnostic);

#endif
