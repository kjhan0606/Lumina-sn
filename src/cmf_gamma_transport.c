#include "cmf_gamma_transport.h"
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* KN 미분단면의 적분 및 광전흡수 근사:
 * Wilk et al.2019, MNRAS487,1218 Eq6/7 및 Hillier/Dessart2012 부록.
 * 원본 소스 복사가 아니라 해당 식의 독립 구현이다. */
static const double sigma_T=6.6524587321e-25,me_MeV=.51099895069;
static const double c_light=2.99792458e10,MeV_erg=1.602176634e-6;
static const double pi=3.1415926535897932384626433832795;
static double uniform(uint64_t *state){
    uint64_t z=(*state+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    z^=z>>31;
    return ((double)(z>>12)+.5)*0x1p-52;
}
double cmf_gamma_klein_nishina_cm2(double E){
    if(!(E>0)||!isfinite(E))return NAN;
    long double x=E/me_MeV,value;
    if(x<.1L){
        /* 작은 x에서 폐형식의 상쇄를 피한다. 8점 Gauss 적분은 x<0.1에서
         * 매끄러운 KN 각분포에만 사용하며 물리 단면을 Thomson으로 바꾸지 않는다. */
        static const double node[4]={.183434642495649805,.525532409916328986,
                                    .796666477413626740,.960289856497536232};
        static const double weight[4]={.362683783378361983,.313706645877887287,
                                      .222381034453374471,.101228536290376259};
        value=0;
        for(int j=0;j<4;j++)for(int sign=-1;sign<=1;sign+=2){
            long double mu=sign*node[j],k=1/(1+x*(1-mu));
            value+=(3.L/8)*weight[j]*k*k*(k+1/k-1+mu*mu);
        }
    }else{
        long double y=1+2*x,l=log1pl(2*x);
        value=(3.L/4)*((1+x)/(x*x*x)*(2*x*(1+x)/y-l)+l/(2*x)-(1+3*x)/(y*y));
    }
    return (double)(sigma_T*value);
}
double cmf_gamma_photoelectric_cm1(double E,double z5){
    if(!(E>0)||!isfinite(E)||!(z5>=0)||!isfinite(z5))return NAN;
    const double alpha=7.2973525693e-3;
    return 8*sqrt(2.)*sigma_T*pow(alpha,4)*z5*pow(me_MeV/E,3.5);
}
void cmf_gamma_result_free(CMFGammaResult *r){if(r){free(r->storage);memset(r,0,sizeof(*r));}}
static double beta_at(double r,double age){return age==0?0:r/(c_light*age);}
static double doppler(double r,double mu,double age){
    double beta=beta_at(r,age);return (1-beta*mu)/sqrt((1-beta)*(1+beta));
}
static double volume(double a,double b){return (4*pi/3)*(b-a)*(b*b+a*b+a*a);}

int cmf_gamma_transport_shell_mixture(const CMFGammaInput *in,uint64_t np,uint64_t seed,
                                     double mixture,CMFGammaResult *out){
    if(!in||!out||np<2||np>UINT64_C(1000000000)||in->n_shells<1||in->n_lines<1||
       !in->radius_edges_cm||!in->total_electrons_cm3||!in->sum_number_Z5_cm3||
       !in->line_energy_MeV||!in->photon_emissivity||
       !(in->homologous_age_s>=0)||!isfinite(in->homologous_age_s)||
       !(mixture>=0&&mixture<=1)||!isfinite(mixture))return -1;
    size_t ns=in->n_shells,nl=in->n_lines;
    if(ns>SIZE_MAX/nl/sizeof(double)||ns>SIZE_MAX/(4*sizeof(double)))return -1;
    double *cdf=malloc(ns*nl*sizeof(double)),*work=calloc(4*ns,sizeof(double));
    double *source_power=mixture>0?calloc(ns,sizeof(double)):NULL;
    double *weight=mixture>0?calloc(ns*nl,sizeof(double)):NULL;
    CMFGammaResult result={0};result.n_shells=(int)ns;result.packets=np;
    result.storage=calloc(3*ns,sizeof(double));
    if(!cdf||!work||!result.storage||(mixture>0&&(!source_power||!weight)))goto fail;
    result.deposition_erg_cm3_s=result.storage;result.deposition_standard_error=result.storage+ns;
    result.positive_packet_count=result.storage+2*ns;
    double *vol=work,*per_packet=vol+ns,*mean=per_packet+ns,*m2=mean+ns;
    const double *edges=in->radius_edges_cm;double age=in->homologous_age_s;
    if(edges[0]!=0)goto fail;
    for(size_t s=0;s<ns;s++){
        if(!(edges[s+1]>edges[s])||!isfinite(edges[s+1])||
           !(in->total_electrons_cm3[s]>=0)||!isfinite(in->total_electrons_cm3[s])||
           !(in->sum_number_Z5_cm3[s]>=0)||!isfinite(in->sum_number_Z5_cm3[s]))goto fail;
        vol[s]=volume(edges[s],edges[s+1]);if(!(vol[s]>0)||!isfinite(vol[s]))goto fail;
    }
    if(beta_at(edges[ns],age)>=.5)goto fail;
    double luminosity=0;
    for(size_t l=0;l<nl;l++){
        if(!(in->line_energy_MeV[l]>0)||!isfinite(in->line_energy_MeV[l]))goto fail;
        for(size_t s=0;s<ns;s++){
            double rate=in->photon_emissivity[l*ns+s];
            if(!(rate>=0)||!isfinite(rate))goto fail;
            double power=rate*vol[s]*in->line_energy_MeV[l]*MeV_erg;
            luminosity+=power;
            if(source_power)source_power[s]+=power;
            if(!isfinite(luminosity))goto fail;
            cdf[l*ns+s]=luminosity;
        }
    }
    result.emitted_comoving_erg_s=luminosity;
    if(luminosity==0){cmf_gamma_result_free(out);*out=result;free(cdf);free(work);free(source_power);free(weight);return 0;}
    double proposal_total=luminosity;
    if(mixture>0){
        size_t active=0;
        for(size_t s=0;s<ns;s++)if(source_power[s]>0)active++;
        if(!active)goto fail;
        proposal_total=0;
        /* Baes et al.2016의 composite source biasing. 에너지별 방출 비는
         * 유지하고 셸 선택만 바꾼다. 관측 tally를 이용한 사후 정규화가 아니다. */
        for(size_t l=0;l<nl;l++)for(size_t s=0;s<ns;s++){
            size_t index=l*ns+s;
            double power=in->photon_emissivity[index]*vol[s]*in->line_energy_MeV[l]*MeV_erg;
            double proposal=source_power[s]>0?
                (1-mixture)*power+mixture*(luminosity/active)*(power/source_power[s]):0;
            if(!(proposal>=0)||!isfinite(proposal)||(power>0&&!(proposal>0)))goto fail;
            weight[index]=proposal>0?power/proposal:0;
            proposal_total+=proposal;cdf[index]=proposal_total;
        }
        if(!(proposal_total>0)||!isfinite(proposal_total))goto fail;
        /* CDF 누적의 반올림까지 포함한 제안 분포의 전체 정규화. */
        for(size_t k=0;k<ns*nl;k++){
            weight[k]*=proposal_total/luminosity;
            if(!(weight[k]>=0)||!isfinite(weight[k]))goto fail;
        }
    }
    long double lab_emitted=0,lab_deposited=0,lab_escaped=0,cmf_deposited=0;
    double total_mean=0,total_m2=0;
    for(uint64_t packet=0;packet<np;packet++){
        memset(per_packet,0,ns*sizeof(double));
        /* 독립 표본. 아래 Welford에는 제안 분포에서 실제 얻은 가중 침적을 넣는다. */
        double pick=uniform(&seed)*proposal_total;size_t lo=0,hi=ns*nl;
        while(lo<hi){size_t mid=lo+(hi-lo)/2;if(pick<cdf[mid])hi=mid;else lo=mid+1;}
        if(lo==ns*nl)goto fail;
        double packet_weight=weight?weight[lo]:1;
        if(!(packet_weight>0)||!isfinite(packet_weight))goto fail;
        int shell=(int)(lo%ns);double E0=in->line_energy_MeV[lo/ns];
        double ri=edges[shell],ro=edges[shell+1];
        double r=cbrt(ri*ri*ri+uniform(&seed)*(ro*ro*ro-ri*ri*ri));
        double mu_c=2*uniform(&seed)-1,beta=beta_at(r,age),gamma=1/sqrt((1-beta)*(1+beta));
        double mu=(mu_c+beta)/(1+beta*mu_c),E=E0*gamma*(1+beta*mu_c);
        lab_emitted+=packet_weight*(E/E0);
        int finished=0;
        for(int event=0;event<1000000;event++){
            if(shell<0||shell>=(int)ns||!(r>=edges[shell]&&r<=edges[shell+1])||
               !(mu>=-1&&mu<=1)||!(E>0)||!isfinite(E))goto fail;
            double z=r*mu,impact=r*sqrt((1-mu)*(1+mu));
            int inward=mu<0&&impact<edges[shell];
            double boundary=inward?edges[shell]:edges[shell+1];
            if(impact>boundary)goto fail;
            double zend=sqrt((boundary-impact)*(boundary+impact));if(inward)zend=-zend;
            double distance=zend-z;if(!(distance>=0)||!isfinite(distance))goto fail;
            double muend=zend/boundary,D0=doppler(r,mu,age),D1=doppler(boundary,muend,age);
            /* D는 이 속도 범위에서 ray를 따라 감소한다. 각 셸 구간의 엄밀한
             * majorant로 null-collision을 하므로 opacity를 중앙값으로 고정하지 않는다. */
            if(!(D1>0)||D1>D0*(1+16*DBL_EPSILON))goto fail;
            double majorant=D0*(in->total_electrons_cm3[shell]*cmf_gamma_klein_nishina_cm2(E*D1)+
                               cmf_gamma_photoelectric_cm1(E*D1,in->sum_number_Z5_cm3[shell]));
            if(!(majorant>=0)||!isfinite(majorant))goto fail;
            double flight=majorant>0?-log(uniform(&seed))/majorant:INFINITY;
            if(flight>=distance){
                r=boundary;mu=muend;result.boundary_crossings++;
                shell+=inward?-1:1;
                if(shell==(int)ns){lab_escaped+=packet_weight*(E/E0);result.escapes++;finished=1;break;}
                continue;
            }
            z+=flight;r=hypot(z,impact);mu=z/r;
            double D=doppler(r,mu,age),Ec=E*D;
            double scatter=in->total_electrons_cm3[shell]*cmf_gamma_klein_nishina_cm2(Ec);
            double absorb=cmf_gamma_photoelectric_cm1(Ec,in->sum_number_Z5_cm3[shell]);
            double opacity=D*(scatter+absorb);
            if(!(opacity>0)||!isfinite(opacity)||opacity>majorant*(1+64*DBL_EPSILON))goto fail;
            if(uniform(&seed)*majorant>opacity){result.null_events++;continue;}
            if(uniform(&seed)*(scatter+absorb)<absorb){
                per_packet[shell]+=packet_weight*(Ec/E0);cmf_deposited+=packet_weight*(Ec/E0);
                lab_deposited+=packet_weight*(E/E0);
                result.absorptions++;finished=1;break;
            }
            double cosine=0,k=0;int accepted=0;
            for(int attempt=0;attempt<100000;attempt++){
                cosine=2*uniform(&seed)-1;k=1/(1+(Ec/me_MeV)*(1-cosine));
                double probability=.5*k*k*(k+1/k-1+cosine*cosine);
                if(!(probability>=0&&probability<=1))goto fail;
                if(uniform(&seed)<probability){accepted=1;break;}
            }
            if(!accepted)goto fail;
            beta=beta_at(r,age);gamma=1/sqrt((1-beta)*(1+beta));
            mu_c=(mu-beta)/(1-beta*mu);
            if(!(mu_c>=-1&&mu_c<=1))goto fail;
            double after_mu=mu_c*cosine+sqrt((1-mu_c)*(1+mu_c)*(1-cosine)*(1+cosine))*cos(2*pi*uniform(&seed));
            double after_E=Ec*k*gamma*(1+beta*after_mu);
            double deposit=packet_weight*((Ec/E0)*(1-k));
            per_packet[shell]+=deposit;cmf_deposited+=deposit;lab_deposited+=packet_weight*((E-after_E)/E0);
            E=after_E;mu=(after_mu+beta)/(1+beta*after_mu);result.scatterings++;
        }
        if(!finished)goto fail;
        double packet_total=0;
        for(size_t s=0;s<ns;s++){
            packet_total+=per_packet[s];
            if(per_packet[s]>0)result.positive_packet_count[s]++;
            double delta=per_packet[s]-mean[s];mean[s]+=delta/(packet+1);
            m2[s]+=delta*(per_packet[s]-mean[s]);
        }
        double delta=packet_total-total_mean;total_mean+=delta/(packet+1);
        total_m2+=delta*(packet_total-total_mean);
    }
    for(size_t s=0;s<ns;s++){
        result.deposition_erg_cm3_s[s]=luminosity*mean[s]/vol[s];
        result.deposition_standard_error[s]=(luminosity/vol[s])*sqrt(m2[s]/np/(np-1));
        if(!isfinite(result.deposition_erg_cm3_s[s])||!isfinite(result.deposition_standard_error[s]))goto fail;
    }
    result.emitted_lab_erg_s=(double)(lab_emitted/np*luminosity);
    result.deposited_comoving_erg_s=(double)(cmf_deposited/np*luminosity);
    result.deposited_lab_erg_s=(double)(lab_deposited/np*luminosity);
    result.escaped_lab_erg_s=(double)(lab_escaped/np*luminosity);
    result.deposited_comoving_standard_error=luminosity*sqrt(total_m2/np/(np-1));
    if(!isfinite(result.emitted_lab_erg_s)||!isfinite(result.deposited_comoving_erg_s)||
       !isfinite(result.deposited_lab_erg_s)||!isfinite(result.escaped_lab_erg_s)||
       !isfinite(result.deposited_comoving_standard_error))goto fail;
    result.lab_energy_relative_error=(double)((lab_emitted-lab_deposited-lab_escaped)/lab_emitted);
    if(!isfinite(result.lab_energy_relative_error)||fabs(result.lab_energy_relative_error)>1e-11)goto fail;
    cmf_gamma_result_free(out);*out=result;free(cdf);free(work);free(source_power);free(weight);return 0;
fail:
    cmf_gamma_result_free(&result);free(cdf);free(work);free(source_power);free(weight);return -1;
}

int cmf_gamma_transport(const CMFGammaInput *in,uint64_t np,uint64_t seed,CMFGammaResult *out){
    return cmf_gamma_transport_shell_mixture(in,np,seed,0,out);
}
