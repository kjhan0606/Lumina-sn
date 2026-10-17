/* 봉인 생산 소스의 내부 상계 연산자를 그대로 호출한다. 생산 소스 수정 없음. */
#include "cmf_exact_multigpu.cu"
#include <cstdio>
#include <cstdint>
#include <cstring>
int main(int argc,char **argv){
    if(argc!=2)return 2;
    setvbuf(stdout,nullptr,_IOLBF,0);
    FILE *f=std::fopen(argv[1],"rb");if(!f)return 2;
    char magic[8];uint64_t shape[2];double p[4];
    if(fread(magic,1,8,f)!=8||memcmp(magic,"CMFOP01",8)||
       fread(shape,8,2,f)!=2||shape[0]!=113||shape[1]!=303316||fread(p,8,4,f)!=4){fclose(f);return 2;}
    const int ns=(int)shape[0],nb=(int)shape[1];const size_t cells=(size_t)ns*nb;
    std::vector<double> nu(nb),ri(ns),ro(ns),chi(cells),scatter(cells);
    if(fread(nu.data(),8,nb,f)!=(size_t)nb||fread(ri.data(),8,ns,f)!=(size_t)ns||
       fread(ro.data(),8,ns,f)!=(size_t)ns||fread(chi.data(),8,cells,f)!=cells||
       fread(scatter.data(),8,cells,f)!=cells||fgetc(f)!=EOF||ferror(f)){fclose(f);return 2;}
    if(fclose(f))return 2;
    for(int k=0;k<4;k++)if(!std::isfinite(p[k])||(k<3?p[k]<=0:p[k]<0))return 2;
    for(size_t i=0;i<cells;i++)if(!std::isfinite(chi[i])||!std::isfinite(scatter[i])||scatter[i]<0)return 2;
    PersistentBoundContext context;context.emissivity=true;
    const CMFMultiGPUEpochSchedule schedule={128,4096,32};
    auto status=context.initialize(ns,nb,p[0],nu.data(),ri.data(),ro.data(),p[1],p[2],p[3],
        chi.data(),scatter.data(),1,CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS,&schedule);
    printf("OPERATOR_INITIALIZE status=%s\n",cmf_multigpu_status_name(status));
    if(status!=CMF_MGPU_OK)return 1;
    std::vector<double> zero(cells,0.),weight(cells,1.),lower(cells),nearest(cells),upper(cells);
    status=context.apply_bounds(zero.data(),weight.data(),true,lower.data(),nearest.data(),upper.data());
    printf("OPERATOR_K1 status=%s\n",cmf_multigpu_status_name(status));
    if(status!=CMF_MGPU_OK)return 1;
    size_t worst=0;double max_lower=0,max_nearest=0;
    for(size_t i=0;i<cells;i++){
        if(upper[i]>upper[worst])worst=i;
        max_lower=std::max(max_lower,lower[i]);max_nearest=std::max(max_nearest,nearest[i]);
    }
    printf("OPERATOR_K1_NORM lower=%.17g nearest=%.17g upper=%.17g shell=%zu bin=%zu nu=%.17g chi=%.17g scatter=%.17g\n",
        max_lower,max_nearest,upper[worst],worst/(size_t)nb,worst%(size_t)nb,nu[worst%nb],chi[worst],scatter[worst]);
    /* 아래 weight는 오차 노름 후보일 뿐 물리 J가 아니다. */
#ifdef CMF_OPERATOR_WEIGHTED_SEED_TEST
    /* 생산 실패 보고의 최대 잔차를 모든 성분에 준 보수적 시험 입력이다.
     * 실제 잔차 벡터/새 J의 재생이나 물리해 검증으로 표시하지 않는다. */
    std::vector<double> residual(cells,7.6103000637428586e-11);
    MultiGPUEnvelopeContext ec={&context,zero.data(),cells};
    double q=INFINITY;size_t steps=0;CMFEnvelopeReport er;
    auto es=cmf_error_envelope_weighted_seed(cells,residual.data(),upper.data(),4,
        multigpu_apply_scattering_upper,&ec,weight.data(),&q,&steps);
    printf("OPERATOR_WEIGHTED_SEED status=%s norm_upper=%.17g steps=%zu residual_upper=%.17g\n",
        cmf_error_envelope_status_name(es),q,steps,residual[0]);
    if(es!=CMF_ENVELOPE_OK)return 1;
    es=cmf_error_envelope_verify(cells,residual.data(),weight.data(),
        multigpu_apply_scattering_upper,&ec,&er);
    printf("OPERATOR_WEIGHTED_VERIFY status=%s margin=%.17g\n",cmf_error_envelope_status_name(es),er.minimum_margin);
    if(es!=CMF_ENVELOPE_OK)return 1;
    es=cmf_error_envelope_refine(cells,residual.data(),weight.data(),2,
        multigpu_apply_scattering_upper,&ec,&er);
    printf("OPERATOR_WEIGHTED_REFINE status=%s iterations=%zu min=%.17g max=%.17g\n",
        cmf_error_envelope_status_name(es),er.iterations_completed,
        *std::min_element(weight.begin(),weight.end()),*std::max_element(weight.begin(),weight.end()));
    if(es!=CMF_ENVELOPE_OK)return 1;
#else
    for(int step=1;step<=4;step++){
        for(size_t i=0;i<cells;i++){
            weight[i]=std::nextafter(1.+upper[i],INFINITY);
            if(!std::isfinite(weight[i]))return 1;
        }
        status=context.apply_round(zero.data(),weight.data(),true,1,upper.data());
        if(status!=CMF_MGPU_OK){printf("OPERATOR_WEIGHT status=%s step=%d\n",cmf_multigpu_status_name(status),step);return 1;}
        double q=0;worst=0;
        for(size_t i=0;i<cells;i++){
            double ratio=std::nextafter(upper[i]/weight[i],INFINITY);
            if(ratio>q){q=ratio;worst=i;}
        }
        printf("OPERATOR_WEIGHT step=%d norm_upper=%.17g contraction_proved=%d shell=%zu bin=%zu\n",step,q,q<1,worst/(size_t)nb,worst%(size_t)nb);
        if(q<1)break;
    }
#endif
    puts("OPERATOR_PROBE_DONE new_RTE=0 new_physical_solution=0");return 0;
}
