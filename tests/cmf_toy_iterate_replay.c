/* 기존 native reader/writer의 전체 수치 상태 왕복. 새 물리 풀이가 아니다. */
#include "cmf_toy_model.h"
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"ITERATE_REPLAY_FAIL line=%d %s\n",__LINE__,#x);goto done;}}while(0)
int main(int argc,char **argv){
    if(argc!=5)return 2;
    const double mass[]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};int rc=1,it=-1;
    CHECK(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    CHECK(!cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&it));
    CHECK(it==0&&m.plasma.n_shells==113&&m.opacity.n_fine==117606&&
        m.nlte.n_nlte_levels_total==19627&&!m.plasma.physical_history&&m.plasma.initial_steady_nlte==1);
    CHECK(!cmf_toy_initial_iterate_write(&m,"state_roundtrip.bin",argv[4],it));
    CHECK(!cmf_toy_model_write_material(&m,"material_roundtrip.csv","NUMERICAL_STATE_ROUNDTRIP_ONLY; NOT_CONVERGED; NOT_PHYSICAL_HISTORY"));
    printf("ITERATE_REPLAY_READ_DONE shells=%d levels=%d frequencies=%d iteration=%d physical_history=0 RTE_solved=0 SE_solved=0 goal=NOT_JUDGED\n",
        m.plasma.n_shells,m.nlte.n_nlte_levels_total,m.opacity.n_fine,it);rc=0;
done:
    cmf_toy_model_free(&m);return rc;
}
