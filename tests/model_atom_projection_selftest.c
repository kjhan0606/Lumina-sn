/* 실제 원자모형 등록에서 상위 이온 누락과 terminal 혼입을 확인한다. */
#include "lumina.h"

static int check(int yes, const char *message) {
    if (!yes) fprintf(stderr, "[MODEL-PROJECTION][FAIL] %s\n", message);
    return yes ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2) {
        AtomicData actual = {0}; OpacityState op = {0}; NLTEConfig model = {0};
        /* 이 원자 view의 조성은 명시적으로 한 점만 저장한 조건부 입력이다.
         * 물리 쉘 격자 템플릿으로 쓰지 않고 원자 등록만 검사한다. */
        if (check(load_atomic_data(&actual,argv[1],1)==0,"actual atomic load")) return 1;
        op.n_lines=actual.n_lines;
        if (check(nlte_init_model_atom(&model,&actual,&op,1)==0,
                  "actual model projection")) return 1;
        if (check(model.n_nlte_levels_total==actual.n_levels,
                  "actual full-level coverage")) return 1;
        for (int l=0;l<actual.n_levels;l++)
            if (check(model.global_to_nlte_level[l]>=0,"actual full-level missing")) return 1;
        for (int l=0;l<actual.n_lines;l++)
            if (check(model.nlte_line_map[l]>=0,"actual line missing")) return 1;
        printf("[MODEL-PROJECTION][ACTUAL] bound_ions=%d full_levels=%d superlevels=%d lines=%d\n",
               model.n_nlte_ions,model.n_nlte_levels_total,model.n_super_total,actual.n_lines);
        nlte_free(&model); free_atomic_data(&actual);
        return 0;
    }
    enum { NI = 12, NL = 8, NS = 2, NLINE = 4 };
    int ion_z[NI] = {28,28,27,27,26,26,20,20,16,16,14,14};
    int stages[NI] = {5,6,5,6,5,6,4,5,4,5,4,5};
    int offsets[NI+1] = {0,1,1,2,2,3,3,4,4,5,5,8,8};
    int level_z[NL] = {28,27,26,20,16,14,14,14};
    int level_stage[NL] = {5,5,5,4,4,4,4,4};
    int level_num[NL] = {0,0,0,0,0,0,1,2};
    int super[NL] = {0,0,0,0,0,0,0,1};
    int metastable[NL] = {0};
    double energy[NL] = {0,0,0,0,0,0,1,2};
    int line_z[NLINE] = {14,20,26,28};
    int line_stage[NLINE] = {4,4,5,5};
    int line_upper[NLINE] = {1,0,0,0};
    AtomicData atom = {0}; OpacityState opacity = {0}; NLTEConfig nlte = {0};
    atom.n_ion_pops=NI; atom.n_levels=NL; atom.n_lines=NLINE;
    atom.ion_pop_Z=ion_z; atom.ion_pop_stage=stages; atom.level_offset=offsets;
    atom.level_Z=level_z; atom.level_ion=level_stage; atom.level_num=level_num;
    atom.level_super=super; atom.level_metastable=metastable;
    atom.level_energy_eV=energy;
    atom.line_atomic_number=line_z; atom.line_ion_number=line_stage;
    atom.line_level_upper=line_upper; opacity.n_lines=NLINE;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)==0,"initialization")) return 1;
    if (check(nlte.n_nlte_ions==6,"every bound ion, no terminal") ||
        check(nlte.n_nlte_levels_total==NL,"all full levels") ||
        check(nlte.n_super_total==7,"all superlevels") ||
        check(nlte.super_mode,"superlevel reconstruction active")) return 1;
    const int expected_z[6]={14,16,20,26,27,28};
    for (int i=0;i<6;i++)
        if (check(nlte.nlte_Z[i]==expected_z[i],"contiguous sorted projection")) return 1;
    for (int l=0;l<NL;l++)
        if (check(nlte.global_to_nlte_level[l]>=0,"unmapped full level")) return 1;
    for (int l=0;l<NLINE;l++)
        if (check(nlte.nlte_line_map[l]>=0,"unmapped high-ion line")) return 1;
    nlte_free(&nlte);
    offsets[NI]=NL-1;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "inconsistent level ownership accepted")) return 1;
    offsets[NI]=NL; level_stage[0]=4;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "wrong ion identity accepted")) return 1;
    level_stage[0]=5; super[0]=-1;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "negative superlevel accepted")) return 1;
    puts("[MODEL-PROJECTION][PASS] full-level=8 bound-ion=6 terminal=6 excluded; identity failures rejected");
    return 0;
}
