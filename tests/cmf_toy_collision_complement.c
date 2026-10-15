/* 실제 원자/불변 수치 상태에서 누락 충돌의 조립만 확인한다. NLTE/RTE 해가 아니다. */
#include "cmf_toy_model.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
extern int nlte_cmf_missing_collisions_selftest(NLTEConfig *,const AtomicData *,int,
    int,int,int,int,double,double,double *,uint64_t *);
#define CHECK(x) do{if(!(x)){fprintf(stderr,"ACTUAL_COLLISION_FAIL line=%d\n",__LINE__);rc=1;goto done;}}while(0)
int main(int argc,char **argv){
    if(argc!=5)return 2;
    const double mass[]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};double *matrix=NULL,*weight=NULL;int rc=0,iteration=-1;
    CHECK(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    CHECK(!cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&iteration));
    NLTEConfig *n=&m.nlte;const AtomicData *a=&m.atom;int ns=m.plasma.n_shells,s=43,tested=0;
    CHECK(ns==113&&iteration==2&&m.plasma.initial_steady_nlte&&!m.plasma.physical_history);
    double T=m.plasma.T_e[s],ne=m.plasma.n_electron[s];CHECK(T==5000&&ne>0);
    for(int ion=0;ion<n->n_nlte_ions;ion++){
        int first=n->nlte_ion_level_offset[ion],last=n->nlte_ion_level_offset[ion+1];
        if(last-first<2)continue;
        int start=n->nlte_ion_super_offset[ion],N=n->nlte_ion_super_offset[ion+1]-start;
        CHECK(N>0);
        matrix=calloc((size_t)N*N,sizeof(double));weight=calloc((size_t)N,sizeof(double));CHECK(matrix&&weight);
        uint64_t pairs=0;
        CHECK(!nlte_cmf_missing_collisions_selftest(n,a,ion,s,ns,start,N,T,ne,matrix,&pairs));
        for(int fl=first;fl<last;fl++){
            int gl=n->nlte_to_global_level[fl],sl=n->fl_to_super[fl]-start;
            weight[sl]+=a->level_g[gl]*exp(-a->level_energy_eV[gl]*EV_TO_ERG/(K_BOLTZMANN*T));
        }
        double worst_column=0,worst_balance=0;long double flow=0,balance_error=0;
        for(int from=0;from<N;from++){
            long double sum=0,scale=0;
            for(int to=0;to<N;to++){
                double x=matrix[(size_t)from*N+to];CHECK(isfinite(x)&&(from==to?x<=0:x>=0));
                sum+=x;scale+=fabs(x);
                if(to>from){
                    long double left=(long double)x*weight[from];
                    long double right=(long double)matrix[(size_t)to*N+from]*weight[to];
                    flow+=left+right;balance_error+=fabsl(left-right);
                }
            }
            if(scale>0)worst_column=fmax(worst_column,(double)(fabsl(sum)/scale));
        }
        if(flow>0)worst_balance=(double)(balance_error/flow);
        CHECK(worst_column<1e-12&&worst_balance<1e-12);
        if(n->nlte_Z[ion]==27&&n->nlte_ion[ion]==4){
            CHECK(pairs==391663);
            int ground=-1,upper=-1;
            for(int fl=first;fl<last;fl++){
                int gl=n->nlte_to_global_level[fl];
                if(a->level_num[gl]==0)ground=n->fl_to_super[fl]-start;
                if(a->level_num[gl]==25)upper=n->fl_to_super[fl]-start;
            }
            CHECK(ground>=0&&upper>=0&&ground!=upper&&matrix[(size_t)upper*N+ground]>0);
            printf("ACTUAL_CO_V missing_pairs=%llu level25_SL=%d ground_SL=%d aggregate_down_s=%.17g aggregate_up_s=%.17g\n",
                (unsigned long long)pairs,upper,ground,matrix[(size_t)upper*N+ground],matrix[(size_t)ground*N+upper]);
        }
        printf("ACTUAL_COLLISION Z=%d charge=%d levels=%d SL=%d missing_pairs=%llu conservation=%.17g LTE_balance=%.17g\n",
            n->nlte_Z[ion],n->nlte_ion[ion],last-first,N,(unsigned long long)pairs,worst_column,worst_balance);
        free(matrix);matrix=NULL;free(weight);weight=NULL;tested++;
    }
    CHECK(tested>10);
    puts("ACTUAL_COLLISION_COMPLEMENT_PASS NLTE_solved=0 RTE_solved=0 physical_history=0");
done:
    free(matrix);free(weight);cmf_toy_model_free(&m);return rc;
}
