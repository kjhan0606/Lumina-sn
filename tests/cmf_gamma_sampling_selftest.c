#include "cmf_gamma_source.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"GAMMA_SAMPLING_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
int cmf_gamma_transport_reference(const CMFGammaInput*,uint64_t,uint64_t,CMFGammaResult*);
static double volume(double a,double b){return (4*M_PI/3)*(b-a)*(b*b+a*b+a*a);}
static int same(const CMFGammaResult *a,const CMFGammaResult *b){
    return a->n_shells==b->n_shells&&a->packets==b->packets&&a->scatterings==b->scatterings&&
        a->absorptions==b->absorptions&&a->escapes==b->escapes&&a->null_events==b->null_events&&
        a->boundary_crossings==b->boundary_crossings&&
        a->emitted_comoving_erg_s==b->emitted_comoving_erg_s&&a->emitted_lab_erg_s==b->emitted_lab_erg_s&&
        a->deposited_comoving_erg_s==b->deposited_comoving_erg_s&&a->deposited_lab_erg_s==b->deposited_lab_erg_s&&
        a->escaped_lab_erg_s==b->escaped_lab_erg_s&&a->deposited_comoving_standard_error==b->deposited_comoving_standard_error&&
        a->lab_energy_relative_error==b->lab_energy_relative_error&&
        !memcmp(a->storage,b->storage,(size_t)3*a->n_shells*sizeof(double));
}
static int compare(const CMFGammaResult *a,const CMFGammaResult *b,const char *name){
    CHECK(a->n_shells==b->n_shells&&a->emitted_comoving_erg_s==b->emitted_comoving_erg_s);
    double sig=hypot(a->deposited_comoving_standard_error,b->deposited_comoving_standard_error);
    CHECK(fabs(a->deposited_comoving_erg_s-b->deposited_comoving_erg_s)<=6*sig+1e-12*a->emitted_comoving_erg_s);
    double max_z=0;
    for(int s=0;s<a->n_shells;s++){
        double delta=fabs(a->deposition_erg_cm3_s[s]-b->deposition_erg_cm3_s[s]);
        double se=hypot(a->deposition_standard_error[s],b->deposition_standard_error[s]);
        CHECK(se>0?delta<=6*se:delta==0);
        if(se>0)max_z=fmax(max_z,delta/se);
    }
    printf("GAMMA_SAMPLING_COMPARE name=%s max_cell_combined_SE=%.17g total_combined_SE=%.17g\n",name,max_z,
           sig>0?fabs(a->deposited_comoving_erg_s-b->deposited_comoving_erg_s)/sig:0);
    return 0;
}
static int synthetic(void){
    double edges[]={0,1e12,2e12},ne[]={0,0},z5[]={0,0},energy[]={1,3};
    double emission[]={1,1e-6,.1,3e-7};
    CMFGammaInput in={2,2,0,edges,ne,z5,energy,emission};CMFGammaResult a={0},b={0},c={0};
    const uint64_t count=100000;
    CHECK(!cmf_gamma_transport(&in,count,41,&a));
    CHECK(!cmf_gamma_transport_reference(&in,count,41,&b)&&same(&a,&b));
    CHECK(!cmf_gamma_transport_shell_mixture(&in,count,41,0,&c)&&same(&a,&c));
    CHECK(!cmf_gamma_transport_shell_mixture(&in,count,42,.5,&b));
    double p=volume(edges[0],edges[1])*(emission[0]*energy[0]+emission[2]*energy[1]);
    double outer=volume(edges[1],edges[2])*(emission[1]*energy[0]+emission[3]*energy[1]);
    p/=p+outer;double q=.5*p+.25,se=sqrt((p*p/q+(1-p)*(1-p)/(1-q)-1)/count);
    CHECK(b.escapes==count&&b.deposited_comoving_erg_s==0&&b.lab_energy_relative_error==0);
    CHECK(fabs(b.escaped_lab_erg_s/b.emitted_comoving_erg_s-1)<6*se);
    CHECK(!cmf_gamma_transport_shell_mixture(&in,count,42,.5,&c)&&same(&b,&c));
    printf("GAMMA_SAMPLING_VACUUM expected=1 sampled=%.17g analytic_SE=%.17g default_reference_byte_equal=1\n",b.escaped_lab_erg_s/b.emitted_comoving_erg_s,se);
    /* 극히 약한 외부 소스의 알려진 국소 흡수 극한. 방출이0인 셸을 채우지 않는다. */
    in.n_lines=1;z5[0]=z5[1]=1e8/(1e12*cmf_gamma_photoelectric_cm1(1,1));
    CHECK(!cmf_gamma_transport(&in,count,43,&a));
    CHECK(!cmf_gamma_transport_shell_mixture(&in,count,44,.5,&b));
    outer=emission[1]*energy[0]*1.602176634e-6;
    CHECK(fabs(b.deposition_erg_cm3_s[1]-outer)<6*b.deposition_standard_error[1]);
    CHECK(b.positive_packet_count[1]>10000&&a.positive_packet_count[1]<100);
    CHECK(b.deposition_standard_error[1]<.02*outer);
    printf("GAMMA_SAMPLING_WEAK_SOURCE expected=%.17g mixture=%.17g SE=%.17g positive_original=%.0f positive_mixture=%.0f\n",
           outer,b.deposition_erg_cm3_s[1],b.deposition_standard_error[1],a.positive_packet_count[1],b.positive_packet_count[1]);
    /* 서로 다른 입사 에너지/방출률과 실제 Compton 및 도플러 변환을 함께 검사. */
    in.n_lines=2;emission[1]=.2;emission[3]=.04;
    ne[0]=ne[1]=1/(cmf_gamma_klein_nishina_cm2(1)*1e12);z5[0]=0;
    z5[1]=1/(1e12*cmf_gamma_photoelectric_cm1(1,1));
    for(int moving=0;moving<=1;moving++){
        in.homologous_age_s=moving?edges[2]/(.1*2.99792458e10):0;
        CHECK(!cmf_gamma_transport(&in,count,45+moving,&a));
        CHECK(!cmf_gamma_transport_reference(&in,count,45+moving,&c)&&same(&a,&c));
        CHECK(!cmf_gamma_transport_shell_mixture(&in,count,47+moving,.5,&b));
        CHECK(!compare(&a,&b,moving?"moving":"static"));
        CHECK(fabs(b.lab_energy_relative_error)<1e-11&&b.scatterings>0&&b.absorptions>0);
        CHECK(!cmf_gamma_transport_shell_mixture(&in,count,49+moving,1,&c));
        CHECK(!compare(&a,&c,moving?"moving_uniform_source":"static_uniform_source"));
    }
    CMFGammaResult saved=b;double cells[6];memcpy(cells,b.storage,sizeof cells);
    CHECK(cmf_gamma_transport_shell_mixture(&in,count,1,-.1,&b));
    CHECK(cmf_gamma_transport_shell_mixture(&in,count,1,1.1,&b));
    CHECK(cmf_gamma_transport_shell_mixture(&in,count,1,NAN,&b));
    ne[0]=-1;CHECK(cmf_gamma_transport_shell_mixture(&in,count,1,.5,&b));ne[0]=0;
    CHECK(!memcmp(&saved,&b,sizeof b)&&!memcmp(cells,b.storage,sizeof cells));
    for(int k=0;k<4;k++)emission[k]=0;
    CHECK(!cmf_gamma_transport_shell_mixture(&in,count,1,.5,&b)&&b.emitted_comoving_erg_s==0);
    /* 방출은 안쪽뿐이지만 수송에 의한 바깥 침적은 남아야 한다. */
    emission[0]=1;in.homologous_age_s=0;ne[1]=0;z5[0]=0;
    CHECK(!cmf_gamma_transport_shell_mixture(&in,count,2,.5,&b));
    CHECK(b.deposition_erg_cm3_s[0]==0&&b.deposition_erg_cm3_s[1]>0);
    cmf_gamma_result_free(&a);cmf_gamma_result_free(&b);cmf_gamma_result_free(&c);return 0;
}
static int actual(const char *model,const char *nuclear,uint64_t packets){
    const double masses[]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyEjecta m={0};CMFGammaNuclearData nuclear_data={0};
    CHECK(!cmf_toy_ejecta_read_refined(&m,model,100,masses,1.10,256)&&m.n_shells==113);
    CHECK(!cmf_gamma_nuclear_read(&nuclear_data,nuclear));
    for(int epoch=0;epoch<2;epoch++){
        double days=epoch?19.48:2;CMFToyGammaSource source={0};CMFGammaResult r[3]={{0}};
        CHECK(!cmf_toy_gamma_source_build(&source,&m,&nuclear_data,days*86400));
        const uint64_t seeds[]={UINT64_C(2026101502),UINT64_C(2026101513),UINT64_C(2026101527)};
        for(int mode=0;mode<3;mode++){
            double mixture=mode?.5:0;
            CHECK(!cmf_gamma_transport_shell_mixture(&source.input,packets,seeds[mode],mixture,&r[mode]));
            if(mode)CHECK(!compare(&r[0],&r[mode],"actual_vs_original"));
            char name[128];snprintf(name,sizeof name,"gamma_sampling_%.2fd_%d.csv",days,mode);
            FILE *fp=fopen(name,"wx");CHECK(fp);
            fprintf(fp,"shell,vlo_cm_s,vhi_cm_s,gamma_emission,gamma_deposition,gamma_standard_error,positron_local,positive_packet_count\n");
            for(int s=0;s<m.n_shells;s++)fprintf(fp,"%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.0f\n",s,
                m.velocity_edges[s],m.velocity_edges[s+1],source.epoch.gamma_emission[s],r[mode].deposition_erg_cm3_s[s],
                r[mode].deposition_standard_error[s],source.epoch.positron_emission[s],r[mode].positive_packet_count[s]);
            CHECK(!fclose(fp));
            printf("GAMMA_SAMPLING_ACTUAL days=%.17g mode=%d mixture=%.17g seed=%llu packets=%llu emitted=%.17g deposited=%.17g SE=%.17g lab_error=%.17g\n",
                   days,mode,mixture,(unsigned long long)seeds[mode],(unsigned long long)packets,r[mode].emitted_comoving_erg_s,
                   r[mode].deposited_comoving_erg_s,r[mode].deposited_comoving_standard_error,r[mode].lab_energy_relative_error);
            fflush(stdout);
        }
        CHECK(!compare(&r[1],&r[2],"actual_independent_mixture_seeds"));
        for(int k=0;k<3;k++)cmf_gamma_result_free(&r[k]);cmf_toy_gamma_source_free(&source);
    }
    cmf_toy_ejecta_free(&m);return 0;
}
int main(int argc,char **argv){
    CHECK(!synthetic());
    if(argc==4){char *end;uint64_t count=strtoull(argv[3],&end,10);CHECK(!*end&&count>=2);CHECK(!actual(argv[1],argv[2],count));}
    else CHECK(argc==1);
    puts("GAMMA_SAMPLING_SELFTEST_OK production_unchanged=1 final_spectrum=0");return 0;
}
