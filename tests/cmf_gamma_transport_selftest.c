#include "cmf_gamma_source.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"GAMMA_TEST_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
static double vol(double a,double b){return (4*M_PI/3)*(b-a)*(b*b+a*b+a*a);}
static int synthetic(void){
    const double sigma=6.6524587321e-25;
    /* 미분 KN 식을 독립 Simpson 적분한 값과 총단면 비교. */
    for(int t=0;t<10;t++){
        double E=pow(10.,-8+t),x=E/.51099895069;long double integral=0;int n=100000;
        for(int j=0;j<=n;j++){
            long double mu=-1+2.L*j/n,k=1/(1+x*(1-mu));
            integral+=(j==0||j==n?1:j%2?4:2)*k*k*(k+1/k-1+mu*mu);
        }
        double expected=(double)(integral*(2.L/n)/3*3/8*sigma);
        CHECK(fabs(cmf_gamma_klein_nishina_cm2(E)/expected-1)<1e-9);
    }
    CHECK(isnan(cmf_gamma_klein_nishina_cm2(0))&&isnan(cmf_gamma_photoelectric_cm1(1,-1)));
    double edge[3]={0,1e12,2e12},ne[2]={0,0},z5[2]={0,0},E=1,emiss[2]={1,0};
    CMFGammaInput in={1,1,0,edge,ne,z5,&E,emiss};CMFGammaResult r={0};
    CHECK(!cmf_gamma_transport(&in,100000,5,&r));
    CHECK(r.escapes==r.packets&&r.absorptions==0&&r.scatterings==0&&r.deposited_comoving_erg_s==0);
    CHECK(r.escaped_lab_erg_s==r.emitted_comoving_erg_s&&r.lab_energy_relative_error==0);
    cmf_gamma_result_free(&r);
    for(int t=0;t<3;t++){
        double tau=pow(10.,t-1),expected=3/(4*tau)*(1-1/(2*tau*tau)+(1/tau+1/(2*tau*tau))*exp(-2*tau));
        z5[0]=tau/(edge[1]*cmf_gamma_photoelectric_cm1(E,1));
        CHECK(!cmf_gamma_transport(&in,200000,100+t,&r));
        double actual=r.escaped_lab_erg_s/r.emitted_lab_erg_s;
        double se=sqrt(expected*(1-expected)/r.packets);
        CHECK(fabs(actual-expected)<6*se&&r.scatterings==0&&r.absorptions>0);
        CHECK(fabs(r.deposition_erg_cm3_s[0]*vol(0,edge[1])/r.deposited_comoving_erg_s-1)<1e-11);
        CHECK(fabs(r.deposition_standard_error[0]*vol(0,edge[1])/r.emitted_comoving_erg_s-se)<.03*se);
        printf("GAMMA_ABSORPTION tau=%g expected=%.17g escaped=%.17g sigma=%.17g lab_error=%.17g\n",tau,expected,actual,se,r.lab_energy_relative_error);
        cmf_gamma_result_free(&r);
    }
    z5[0]=0;ne[0]=1/(cmf_gamma_klein_nishina_cm2(E)*edge[1]);
    CHECK(!cmf_gamma_transport(&in,100000,9,&r));
    CHECK(r.scatterings>10000&&r.absorptions==0&&r.escapes==r.packets&&r.deposited_comoving_erg_s>0);
    CHECK(fabs(r.deposited_lab_erg_s/r.deposited_comoving_erg_s-1)<1e-13);
    printf("GAMMA_COMPTON deposited_fraction=%.17g scatterings=%llu lab_error=%.17g\n",
       r.deposited_comoving_erg_s/r.emitted_comoving_erg_s,(unsigned long long)r.scatterings,r.lab_energy_relative_error);
    cmf_gamma_result_free(&r);
    in.n_shells=2;ne[0]=0;z5[1]=1/(edge[1]*cmf_gamma_photoelectric_cm1(E,1));
    CHECK(!cmf_gamma_transport(&in,100000,10,&r));
    CHECK(r.deposition_erg_cm3_s[0]==0&&r.deposition_erg_cm3_s[1]>0&&emiss[1]==0);
    CMFGammaResult duplicate={0};CHECK(!cmf_gamma_transport(&in,100000,10,&duplicate));
    CHECK(!memcmp(r.storage,duplicate.storage,6*sizeof(double))&&r.escaped_lab_erg_s==duplicate.escaped_lab_erg_s);
    CMFGammaResult saved=r;double before[6];memcpy(before,r.storage,sizeof before);
    ne[0]=-1;CHECK(cmf_gamma_transport(&in,100000,10,&r)!=0);
    CHECK(!memcmp(&saved,&r,sizeof r)&&!memcmp(before,r.storage,sizeof before));ne[0]=0;
    cmf_gamma_result_free(&r);cmf_gamma_result_free(&duplicate);
    in.homologous_age_s=edge[2]/(.1*2.99792458e10);ne[0]=ne[1]=1/(sigma*edge[1]);
    CHECK(!cmf_gamma_transport(&in,100000,11,&r));
    CHECK(r.scatterings>0&&r.absorptions>0&&r.null_events>0&&r.deposited_comoving_erg_s>0);
    CHECK(fabs(r.deposited_lab_erg_s/r.deposited_comoving_erg_s-1)>1e-5);
    CHECK(fabs(r.lab_energy_relative_error)<1e-11);
    printf("GAMMA_MOVING deposited_cmf=%.17g deposited_lab=%.17g lab_error=%.17g null_events=%llu\n",
      r.deposited_comoving_erg_s,r.deposited_lab_erg_s,r.lab_energy_relative_error,(unsigned long long)r.null_events);
    cmf_gamma_result_free(&r);return 0;
}
static int actual(const char *model,const char *nuclear,uint64_t packets){
    CMFGammaNuclearData d={0};CHECK(!cmf_gamma_nuclear_read(&d,nuclear));
    CHECK(d.n_lines==21&&d.decay.positron_kinetic_erg[0]==0&&d.decay.positron_kinetic_erg[1]>0);
    printf("GAMMA_NUCLEAR lines=%d Ni_gamma_MeV=%.17g Co_gamma_MeV=%.17g Co_positron_MeV=%.17g declared_Q=%.17g/%.17g\n",
       d.n_lines,d.decay.gamma_energy_erg[0]/1.602176634e-6,d.decay.gamma_energy_erg[1]/1.602176634e-6,
       d.decay.positron_kinetic_erg[1]/1.602176634e-6,d.declared_total_MeV[0],d.declared_total_MeV[1]);
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    for(int ns=100;ns<=200;ns+=100){
        CMFToyEjecta m={0};CHECK(!cmf_toy_ejecta_read(&m,model,ns,masses));
        for(int k=0;k<2;k++){
            double epoch=(k?19.48:2)*86400;
            CMFToyGammaSource source={0};CMFGammaResult r={0};
            CHECK(!cmf_toy_gamma_source_build(&source,&m,&d,epoch));
            CHECK(!cmf_gamma_transport(&source.input,packets,20261015+ns+k,&r));
            double analytic=0,positron=0,tally=0,outer=0;
            for(int s=0;s<ns;s++){
                double V=vol(source.input.radius_edges_cm[s],source.input.radius_edges_cm[s+1]);
                analytic+=source.epoch.gamma_emission[s]*V;positron+=source.epoch.positron_emission[s]*V;
                tally+=r.deposition_erg_cm3_s[s]*V;
                if(source.epoch.gamma_emission[s]==0)outer+=r.deposition_erg_cm3_s[s]*V;
            }
            CHECK(fabs(r.emitted_comoving_erg_s/analytic-1)<1e-12);
            CHECK(fabs(tally/r.deposited_comoving_erg_s-1)<1e-11&&positron>0);
            CHECK(fabs(r.lab_energy_relative_error)<1e-11);
            if(k==1)CHECK(outer>0&&r.escapes>0);
            char filename[128];snprintf(filename,sizeof filename,"gamma_%d_%.2fd.csv",ns,epoch/86400);
            FILE *fp=fopen(filename,"wx");CHECK(fp);
            fprintf(fp,"# MC_COMPTON_PHOTOELECTRIC; not SF heating; nuclear identity not public-verified\n"
                "# zero-tally shells are not physical exact zeros; their sample variance is not an upper limit\n"
                "shell,vlo_cm_s,vhi_cm_s,gamma_emission,gamma_deposition,gamma_standard_error,positron_local,positive_packet_count\n");
            for(int s=0;s<ns;s++)fprintf(fp,"%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.0f\n",s,
                m.velocity_edges[s],m.velocity_edges[s+1],source.epoch.gamma_emission[s],r.deposition_erg_cm3_s[s],
                r.deposition_standard_error[s],source.epoch.positron_emission[s],r.positive_packet_count[s]);
            CHECK(!fclose(fp));
            printf("GAMMA_TOY shells=%d days=%.17g emitted_cmf=%.17g deposited_cmf=%.17g positron=%.17g escape_lab=%.17g "
                "deposition_no_radioactivity=%.17g standard_error=%.17g lab_error=%.17g packets=%llu scatters=%llu nulls=%llu\n",ns,epoch/86400,
                analytic,tally,positron,r.escaped_lab_erg_s,outer,r.deposited_comoving_standard_error,r.lab_energy_relative_error,
                (unsigned long long)r.packets,(unsigned long long)r.scatterings,(unsigned long long)r.null_events);
            cmf_gamma_result_free(&r);cmf_toy_gamma_source_free(&source);
        }
        cmf_toy_ejecta_free(&m);
    }
    return 0;
}
static int source_choices(const char *model,const char *nuclear,uint64_t packets){
    CMFGammaNuclearData raw={0},completed={0};CHECK(!cmf_gamma_nuclear_read(&raw,nuclear));
    CHECK(!cmf_gamma_nuclear_complete_declared_Q(&completed,&raw));
    CHECK(!memcmp(raw.energy_MeV,completed.energy_MeV,sizeof raw.energy_MeV));
    for(int k=0;k<2;k++){
        CHECK(raw.completion_scale[k]==1&&completed.completion_scale[k]>1);
        CHECK(fabs((completed.decay.gamma_energy_erg[k]+completed.decay.positron_kinetic_erg[k])/
                   (raw.declared_total_MeV[k]*1.602176634e-6)-1)<1e-14);
    }
    CMFGammaNuclearData saved=completed;
    CHECK(cmf_gamma_nuclear_complete_declared_Q(&completed,&completed)!=0&&!memcmp(&saved,&completed,sizeof saved));
    CMFGammaNuclearData malformed=raw;
    malformed.declared_total_MeV[1]=-1;
    CHECK(cmf_gamma_nuclear_complete_declared_Q(&completed,&malformed)!=0&&!memcmp(&saved,&completed,sizeof saved));
    malformed=raw;malformed.decay.gamma_energy_erg[1]*=2;
    CHECK(cmf_gamma_nuclear_complete_declared_Q(&completed,&malformed)!=0&&!memcmp(&saved,&completed,sizeof saved));
    CHECK(cmf_gamma_nuclear_read(&completed,"intentionally_absent_nuclear_input")!=0&&!memcmp(&saved,&completed,sizeof saved));
    printf("GAMMA_COMPLETION_WEIGHTS Ni=%.17g Co=%.17g observation_used=0\n",completed.completion_scale[0],completed.completion_scale[1]);
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyEjecta m={0};const int ns=100;const double epoch=19.48*86400;
    CHECK(!cmf_toy_ejecta_read(&m,model,ns,masses));
    for(int choice=0;choice<4;choice++){
        const CMFGammaNuclearData *data=choice/2?&completed:&raw;
        CMFDecaySourceMode mode=choice%2?CMF_DECAY_INTERVAL_AVERAGE:CMF_DECAY_ENDPOINT;
        double previous=choice%2?17.71*86400:epoch;
        const char *names[4]={"raw_endpoint","raw_average","completed_endpoint","completed_average"};
        CMFToyGammaSource source={0};CMFGammaResult r={0};
        CHECK(!cmf_toy_gamma_source_build_at(&source,&m,data,previous,epoch,mode));
        CHECK(!cmf_gamma_transport(&source.input,packets,202610151,&r));
        double positron=0,emission=0;
        char name[128];snprintf(name,sizeof name,"gamma_variant_%s.csv",names[choice]);
        FILE *fp=fopen(name,"wx");CHECK(fp);
        fprintf(fp,"# nuclear source control; not a fitted normalization or spectrum\n"
                   "shell,vlo_cm_s,vhi_cm_s,gamma_emission,gamma_deposition,gamma_standard_error,positron_local,positive_packet_count\n");
        for(int s=0;s<ns;s++){
            double V=vol(source.input.radius_edges_cm[s],source.input.radius_edges_cm[s+1]);
            positron+=V*source.epoch.positron_emission[s];emission+=V*source.epoch.gamma_emission[s];
            fprintf(fp,"%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.0f\n",s,m.velocity_edges[s],m.velocity_edges[s+1],
                    source.epoch.gamma_emission[s],r.deposition_erg_cm3_s[s],r.deposition_standard_error[s],
                    source.epoch.positron_emission[s],r.positive_packet_count[s]);
        }
        CHECK(!fclose(fp)&&fabs(emission/r.emitted_comoving_erg_s-1)<1e-12);
        printf("GAMMA_SOURCE_CONTROL name=%s emitted=%.17g gamma_deposition=%.17g positron=%.17g total=%.17g standard_error=%.17g lab_error=%.17g\n",
               names[choice],emission,r.deposited_comoving_erg_s,positron,r.deposited_comoving_erg_s+positron,
               r.deposited_comoving_standard_error,r.lab_energy_relative_error);
        cmf_gamma_result_free(&r);cmf_toy_gamma_source_free(&source);
    }
    cmf_toy_ejecta_free(&m);return 0;
}
int main(int argc,char **argv){
    CHECK(!synthetic());
    if(argc==4||argc==5){char *end;unsigned long long packets=strtoull(argv[3],&end,10);CHECK(!*end&&packets>=2);
        if(argc==5){CHECK(!strcmp(argv[4],"sources"));CHECK(!source_choices(argv[1],argv[2],packets));}
        else CHECK(!actual(argv[1],argv[2],packets));
    }else CHECK(argc==1);
    puts("GAMMA_TRANSPORT_SELFTEST_OK spectrum_compared=0 SF_implemented=0");return 0;
}
