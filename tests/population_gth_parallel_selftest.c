/* 같은 GTH의 직렬/행 병렬 해와 실패 보존을 비교한다. SN 물리해는 아니다. */
#include "population_contract.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"GTH_PARALLEL_FAIL line=%d %s\n",__LINE__,#x);exit(1);}}while(0)
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

int main(int argc,char **argv){
    REQUIRE(argc==2);
    FILE *out=fopen(argv[1],"wx");REQUIRE(out);
    omp_set_dynamic(0);
    const size_t sizes[2]={512,768};
    for(int test=0;test<2;test++){
        size_t n=sizes[test];double total=7.25e7;
        double *a=calloc(n*n,sizeof(double)),*before=malloc(n*n*sizeof(double));
        double *w=malloc(n*sizeof(double)),*serial=malloc(n*sizeof(double));
        double *parallel=malloc(n*sizeof(double));
        REQUIRE(a&&before&&w&&serial&&parallel);
        long double sum=0;
        for(size_t i=0;i<n;i++){w[i]=pow(10.,-(test?40.:20.)*i/(n-1));sum+=w[i];}
        for(size_t source=0;source<n;source++)for(size_t dest=0;dest<n;dest++){
            if(source==dest)continue;
            double rate=w[dest]*(1.+((source+dest)%7)/7.);
            a[source*n+dest]=rate;a[source*n+source]-=rate;
        }
        memcpy(before,a,n*n*sizeof(double));
        PopulationGeneratorSolveDiagnostic ds,dp;
        omp_set_num_threads(1);double start=seconds();
        REQUIRE(population_generator_stationary_gth(a,n,total,serial,&ds)==POP_OK);
        double one=seconds()-start;
        omp_set_num_threads(2);start=seconds();
        REQUIRE(population_generator_stationary_gth(a,n,total,parallel,&dp)==POP_OK);
        double two=seconds()-start;
        REQUIRE(!memcmp(serial,parallel,n*sizeof(double)));
        REQUIRE(!memcmp(&ds,&dp,sizeof(ds)));
        REQUIRE(!memcmp(a,before,n*n*sizeof(double)));
        double worst=0;
        for(size_t i=0;i<n;i++){
            double expected=(double)((long double)w[i]*total/sum);
            worst=fmax(worst,fabs(serial[i]/expected-1));
        }
        REQUIRE(worst<2e-12);
        REQUIRE(fwrite(serial,sizeof(double),n,out)==n);
        REQUIRE(fwrite(&ds,sizeof(ds),1,out)==1);
        int nested_status=POP_SOLVE_FAILED;
#pragma omp parallel num_threads(2) shared(nested_status,parallel,dp)
        {
#pragma omp single
            nested_status=population_generator_stationary_gth(a,n,total,parallel,&dp);
        }
        REQUIRE(nested_status==POP_OK&&!memcmp(serial,parallel,n*sizeof(double)));
        REQUIRE(!memcmp(&ds,&dp,sizeof(ds)));
        memset(a,0,n*n*sizeof(double));
        REQUIRE(population_generator_stationary_gth(a,n,total,parallel,&dp)==POP_RANK_INCOMPLETE);
        REQUIRE(!memcmp(serial,parallel,n*sizeof(double)));
        a[1]=-1;
        REQUIRE(population_generator_stationary_gth(a,n,total,parallel,&dp)==POP_SOLVE_FAILED);
        REQUIRE(!memcmp(serial,parallel,n*sizeof(double)));
        a[1]=NAN;
        REQUIRE(population_generator_stationary_gth(a,n,total,parallel,&dp)==POP_NONFINITE);
        REQUIRE(!memcmp(serial,parallel,n*sizeof(double)));
        printf("GTH_PARALLEL_PASS n=%zu seconds_1=%.9g seconds_2=%.9g known_relative=%.9g byte_equal=1 nested=preserved invalid_output=preserved\n",n,one,two,worst);
        free(a);free(before);free(w);free(serial);free(parallel);
    }
    REQUIRE(!fclose(out));return 0;
}
