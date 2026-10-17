#include "cmf_anderson_multi.h"
#include <assert.h>
#include <stdio.h>

int main(void){
    CMFAndersonMulti a={0};
    assert(cmf_anderson_multi_init(&a,SIZE_MAX,4)==-1);
    assert(cmf_anderson_multi_init(&a,3,4)==0);
    double scale[3]={1e-100,1,1e100},x[3]={0},g[3];int finished=0;
    for(int k=0;k<30;k++){
        double norm=0;
        for(int i=0;i<3;i++){
            g[i]=scale[i]+.99*x[i];
            norm=fmax(norm,x[i]?fabs(g[i]/x[i]-1):INFINITY);
        }
        if(k&&norm<1e-12){
            for(int i=0;i<3;i++)assert(fabs(g[i]/(100*scale[i])-1)<1e-10);
            finished=1;break;
        }
        cmf_anderson_multi_next(&a,x,g,norm);
    }
    assert(finished);cmf_anderson_multi_free(&a);
    assert(cmf_anderson_multi_init(&a,1,2)==0);
    a.count=1;a.next=1;a.g[0]=1;a.r[0]=.9;
    x[0]=1;g[0]=2;
    cmf_anderson_multi_next(&a,x,g,1);
    assert(x[0]==2&&!a.proposed&&a.rejected==1);
    a.count=1;a.next=1;a.g[0]=3;a.previous_norm=1;a.proposed=1;
    x[0]=8;g[0]=25;
    cmf_anderson_multi_next(&a,x,g,100);
    assert(x[0]==3&&!a.proposed&&a.count==0);
    cmf_anderson_multi_free(&a);
    assert(cmf_anderson_multi_init(&a,3,4)==0);
    for(int i=0;i<3;i++)x[i]=g[i]=0;
    for(int k=0;k<7;k++)cmf_anderson_multi_next(&a,x,g,0);
    for(int i=0;i<3;i++)assert(x[i]==0);
    assert(a.rank_losses>0);cmf_anderson_multi_free(&a);
    setenv("LUMINA_CMF_ANDERSON_DEPTH","garbage",1);assert(cmf_anderson_depth()==0);
    unsetenv("LUMINA_CMF_ANDERSON_DEPTH");assert(cmf_anderson_depth()==1);
    puts("CMF_ANDERSON_MULTI_HELPER_OK rank_deficiency=1 rollback=1 negative_candidate_rejected=1 exact_zero=1 dynamic_range=1 allocation_overflow=1");
    return 0;
}
