/* 작은 동일 fixture의 전체 결과를 내보내 전체/묶음 버퍼를 비교한다. */
#include "cmf_exact_sliding.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv)
{
    enum{NS=19,NB=160,N=NS*NB};
    if(argc!=2)return 2;
    FILE *fp=fopen(argv[1],"wbx");if(!fp)return 2;
    double ri[NS],ro[NS],nu[NB],ct[N],ce[N],source[N],j[N],error[N];
    double bs[NB],br[NB],bg[NB];
    for(int s=0;s<NS;s++){ri[s]=2e14+s*1e13;ro[s]=2e14+(s+1)*1e13;}
    for(int b=0;b<NB;b++){
        nu[b]=1e14*exp((b+.5)*.001);
        bs[b]=2e-8*(1+.1*cos(.2*b));br[b]=.2;bg[b]=2e-9;
    }
    CMFExactInnerBoundary bc={CMF_EXACT_INNER_DIFFUSION,NAN,NAN,bs,br,bg};
    for(int raw=0;raw<=1;raw++)for(int direct=0;direct<=1;direct++){
        for(int s=0;s<NS;s++)for(int b=0;b<NB;b++){
            int i=s*NB+b;
            ct[i]=1e-15*(1+.2*sin(.17*b+.3*s));ce[i]=.1*ct[i];
            source[i]=1e-8*(1+.2*cos(.11*b+.2*s));
            if(raw){source[i]*=ct[i];if(i%17==0)ct[i]=-2e-16;}
            j[i]=error[i]=0;
        }
        CMFExactMode mode=direct?CMF_EXACT_MODE_DIRECT_REFERENCE:CMF_EXACT_MODE_POSITIVE_SLIDING;
        CMFExactReport report;
        CMFExactStatus status=raw?
            cmf_exact_cell_solve_raw_sources(NS,NB,.001,nu,ri,ro,1e6,&bc,ct,ce,source,
                j,direct?NULL:error,3,100,1e-12,mode,&report):
            cmf_exact_cell_solve(NS,NB,.001,nu,ri,ro,1e6,&bc,ct,ce,source,
                j,direct?NULL:error,3,100,1e-12,mode,&report);
        if(status!=CMF_EXACT_OK||(!direct&&!report.componentwise_error_envelope_verified))return 1;
#ifdef CMF_EXACT_TEST_FULL_CELL_BUFFER
        if(report.ray_buffer_slots!=report.segment_slots)return 1;
#else
        if(report.ray_buffer_slots>=report.segment_slots||report.ray_buffer_slots>16*(NS+1))return 1;
#endif
        fprintf(stderr,"CELL_BUFFER raw=%d direct=%d allocated_slots=%zu geometry_slots=%zu iterations=%d K_bound=%.17g\n",
            raw,direct,report.ray_buffer_slots,report.segment_slots,report.iterations_used,
            report.max_scattering_ratio);
        /* 버퍼 크기만 의도적으로 다르다. 나머지 모든 report 바이트와 J/error를 비교한다. */
        report.ray_buffer_slots=0;
        if(fwrite(&report,sizeof(report),1,fp)!=1||fwrite(j,sizeof(double),N,fp)!=N||
           fwrite(error,sizeof(double),N,fp)!=N)return 1;
    }
    if(fclose(fp))return 1;
    puts("CELL_BUFFER_ALL_OUTPUTS_WRITTEN physical_equivalence=NOT_JUDGED");
    return 0;
}
