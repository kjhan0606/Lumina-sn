/* 네 물리 블록의 동일 원소/세대 합산과 기존 인구 잔차. 합성 입력만 쓴다. */
#include "nlte_model_sum.h"
#include <assert.h>
#include <stdio.h>

static int near(double a,double b)
{
    return isfinite(a) && fabs(a-b)<2.e-13*fmax(1.,fabs(b));
}

int main(void)
{
    double initial[]={3.,7.,1.};
    NLTEModelElement element={.Z=16,.shell=49,.material_generation=9,
        .n_variables=3,.initial_population=initial,.total_nuclei=11.};
    double bf_rad[]={-.8,.8,0.,.75,-1.95,1.2,0.,2.5,-2.5};
    double bf_col[]={-.1,0.,.1,0.,0.,0.,.2,0.,-.2};
    double bb_rad[]={-.4,.4,0.,.3,-.3,0.,0.,0.,0.};
    double col_bb[]={0.,0.,0.,0.,-.5,.5,0.,.6,-.6};
    NLTEModelBFBlock bf={.Z=16,.shell=49,.material_generation=9,
        .radiation_generation=5,.n_variables=3,.radiative=bf_rad,.collisional=bf_col};
    NLTEModelBBBlock bb={.Z=16,.shell=49,.material_generation=9,
        .radiation_generation=5,.n_variables=3,.radiative=bb_rad};
    NLTEModelCollisionBBBlock col={.Z=16,.shell=49,.material_generation=9,
        .n_variables=3,.collisional=col_bb};
    NLTEModelCombinedGenerator combined={0};
    assert(!nlte_model_element_generator_combine(&combined,&element,&bf,&bb,&col));
    assert(combined.Z==16 && combined.shell==49 && combined.n_variables==3 &&
           combined.material_generation==9 && combined.radiation_generation==5);
    double expected[]={-1.3,1.2,.1,1.05,-2.75,1.7,.2,3.1,-3.3};
    double residual[]={3.65,-12.55,8.9};
    for(int i=0;i<9;i++)assert(near(combined.generator[i],expected[i]));
    for(int i=0;i<3;i++)assert(near(combined.fixed_residual[i],residual[i]));
    double solved[3];PopulationGeneratorSolveDiagnostic diagnostic;
    assert(population_generator_stationary_gth(combined.generator,3,11.,solved,&diagnostic)==POP_OK);
    assert(diagnostic.generator_recognized &&
           diagnostic.exact_generator_componentwise_residual<1.e-12);
    double private_candidate[]={-1.,-1.,-1.};
    assert(nlte_model_element_fixed_field_candidate(&combined,&element,
        private_candidate,&diagnostic)==POP_OK);
    for(int i=0;i<3;i++)assert(near(private_candidate[i],solved[i]));
    assert(diagnostic.generator_recognized &&
           diagnostic.exact_generator_componentwise_residual<1.e-12);
    combined.material_generation++;
    assert(nlte_model_element_fixed_field_candidate(&combined,&element,
        private_candidate,&diagnostic)!=POP_OK);
    combined.material_generation--;
    for(int i=0;i<3;i++)assert(near(private_candidate[i],solved[i]));
    double saved_generator[9];memcpy(saved_generator,combined.generator,sizeof(saved_generator));
    memset(combined.generator,0,sizeof(saved_generator));
    assert(nlte_model_element_fixed_field_candidate(&combined,&element,
        private_candidate,&diagnostic)==POP_RANK_INCOMPLETE);
    for(int i=0;i<3;i++)assert(near(private_candidate[i],solved[i]));
    memcpy(combined.generator,saved_generator,sizeof(saved_generator));
    nlte_model_combined_generator_free(&combined);
    bb.radiation_generation++;
    assert(nlte_model_element_generator_combine(&combined,&element,&bf,&bb,&col)!=0 &&
           !combined.generator);
    bb.radiation_generation--;
    col.shell++;
    assert(nlte_model_element_generator_combine(&combined,&element,&bf,&bb,&col)!=0 &&
           !combined.generator);
    col.shell--;
    bf_rad[0]=-.7;
    assert(nlte_model_element_generator_combine(&combined,&element,&bf,&bb,&col)!=0 &&
           !combined.generator);
    bf_rad[0]=-.8;
    col.collisional=NULL;
    assert(nlte_model_element_generator_combine(&combined,&element,&bf,&bb,&col)!=0 &&
           !combined.generator);
    puts("MODEL_COMBINED_GENERATOR_SELFTEST PASS synthetic_only private_fixed_field_candidate=PASS physical_SE_RE=NOT_SOLVED");
    return 0;
}
