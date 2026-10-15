/* 같은 실제 원자 자료에서 BF 연산자의 모든 bin을 비교한다.
 *4셸의 거친 초기 predictor이며 수렴 SN 결과가 아니다. */
#include "cmf_toy_model.h"
#include <stdio.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define CHECK(x) do{if(!(x)){fprintf(stderr,"CONTINUUM_FAIL line=%d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char **argv){
    if(argc!=4||atoi(argv[3])!=4)return 2;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};double *saved=NULL;FILE *fp=NULL;int rc=0;
    CHECK(!cmf_toy_model_prepare(&m,argv[1],argv[2],4,mass));
    size_t cells=(size_t)4*m.bf.n_freq_bins;
    saved=malloc(4*cells*sizeof(double));CHECK(saved);
    bf_set_nlte_pops(&m.nlte);
    double *arrays[4]={m.bf.chi_bf,m.bf.eta_bf,m.bf.event_chi_bf,m.bf.event_chi_bf_comparison};
    for(int threads=1;threads<=2;threads++){
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        double started=seconds();compute_bf_opacity(&m.bf,&m.atom,&m.plasma,4);
        printf("CONTINUUM_TIMING threads=%d seconds=%.9g shells=4 channels=%d bins=%d\n",
               threads,seconds()-started,m.atom.bf_channels->n_channels,m.bf.n_freq_bins);
        for(int s=0;s<4;s++)CHECK(m.bf.eta_milne_ok[s]);
        for(int k=0;k<4;k++){
            for(size_t i=0;i<cells;i++)CHECK(isfinite(arrays[k][i])&&((k!=1&&k!=2)||arrays[k][i]>=0));
            if(threads==1)memcpy(saved+(size_t)k*cells,arrays[k],cells*sizeof(double));
            else CHECK(!memcmp(saved+(size_t)k*cells,arrays[k],cells*sizeof(double)));
        }
    }
    fp=fopen("continuum.bin","wbx");CHECK(fp);
    CHECK(fwrite(saved,sizeof(double),4*cells,fp)==4*cells);
    {int closed=fclose(fp);fp=NULL;CHECK(!closed);}
    puts("CMF_CONTINUUM_SELFTEST_PASS all_bins=1 one_two_threads_identical=1 SN_comparison=0");
done:
    if(fp)fclose(fp);
    bf_set_nlte_pops(NULL);free(saved);cmf_toy_model_free(&m);return rc;
}
