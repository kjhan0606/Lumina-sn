/* 실제 인구 갱신 함수의 항등 연산과 기존 감쇠 산술을 직접 검사한다. */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
extern void nlte_population_damp_values_selftest(double*,const double*,size_t,double);
#define CHECK(x) do {if(!(x)){fprintf(stderr,"POPULATION_DAMP_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
int main(void){
    enum {N=6};
    const double old[N]={1,1e30,0,1e-100,DBL_MAX,1e-20};
    const double next[N]={1e-30,1e-10,0,1e-300,DBL_MAX/2,1e20};
    double current[N],preserved[N],expected[N];
    memcpy(preserved,old,sizeof old);memcpy(current,next,sizeof next);
    volatile double a=old[0],b=next[0],weight=1;
    double legacy=a+weight*(b-a);
    CHECK(legacy==0&&b>0);
    nlte_population_damp_values_selftest(current,old,N,1);
    CHECK(!memcmp(current,next,sizeof next)&&!memcmp(old,preserved,sizeof old));
    for(int pass=0;pass<2;pass++){
        double damping=pass?.5:0;
        memcpy(current,next,sizeof next);
        for(int k=0;k<N;k++)expected[k]=old[k]+damping*(next[k]-old[k]);
        nlte_population_damp_values_selftest(current,old,N,damping);
        CHECK(!memcmp(current,expected,sizeof current)&&!memcmp(old,preserved,sizeof old));
        for(int k=0;k<N;k++)CHECK(isfinite(current[k])&&current[k]>=0);
    }
    nlte_population_damp_values_selftest(NULL,NULL,0,.5);
    nlte_population_damp_values_selftest(NULL,NULL,0,1);
    puts("POPULATION_DAMP_PASS identity_byte_exact=1 old_cancellation_reproduced=1 legacy_half_zero_unchanged=1 source_unchanged=1 production_function=1");
    return 0;
}
