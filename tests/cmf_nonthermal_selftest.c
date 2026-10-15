#include "cmf_nonthermal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void close_to(double a,double b,double tol){assert(isfinite(a)&&isfinite(b)&&fabs(a-b)<=tol);}
static void check(const CMFNonthermalResult *r){
    close_to(r->heating_fraction+r->ionization_fraction+r->excitation_fraction,1,2e-11);
    assert(r->heating_fraction>=0&&r->ionization_fraction>=0&&r->excitation_fraction>=0);
    assert(fabs(r->electron_number_error)<1e-12);
    for(int i=0;i<r->n_energy;i++)assert(isfinite(r->path_cm_per_eV[i])&&r->path_cm_per_eV[i]>=0);
}
static void primitives(void){
    const double h[4]={22.8,-12,1.9,-22.6};
    close_to(cmf_nonthermal_arnaud(13.6,13.6,h),0,0);
    double U=1000/13.6,t=1-1/U;
    close_to(cmf_nonthermal_arnaud(1000,13.6,h),1e-14*(22.8*t-12*t*t+1.9*log(U)-22.6*log(U)/U)/(U*13.6*13.6),1e-30);
    assert(isnan(cmf_nonthermal_arnaud(-1,13.6,h)));
    assert(cmf_nonthermal_coulomb(10,1e8)>0);
    assert(cmf_nonthermal_coulomb(1000,1e8)>0);
    close_to(cmf_nonthermal_coulomb(1000,0),0,0);
    close_to(cmf_nonthermal_bethe(5,10,1,1),0,0);
    assert(cmf_nonthermal_bethe(1000,10,1,1)>0);
    close_to(cmf_nonthermal_bethe(10,10,1,0),0,0);
    close_to(cmf_nonthermal_omega(5,10,1,2),0,0);
    assert(cmf_nonthermal_omega(1000,10,1,2)>0);
    /* 문헌 Eq9의 평균 secondary 에너지는 적분의 독립 sanity 기준이다. */
    double J=.6*13.6,upper=(1000-13.6)/2;
    double mean=J*log1p(upper*upper/(J*J))/(2*atan(upper/J));
    close_to(mean,21.53,.01);
    printf("NT_PRIMITIVES_OK secondary_mean_H_1keV=%.17g\n",mean);
}
static double excitation(int n){
    /* E_in=15,dE=10: 최대1회 여기. 상수 L,k의 정확한 확률
     * P=1-exp[-k(E_in-dE)/L]. 값은 회수/에너지 주입[eV]로 비교한다. */
    double *p=calloc((size_t)3*n,sizeof(double));assert(p);
    double *E=p,*L=p+n,*s=p+2*n;
    /* 문턱이 격자에 놓이도록 n은3의 배수. 문턱 위치의 왕복 오차를
     * upwind 해법의 수렴 오차와 섞지 않는다. */
    assert(n%3==0);
    for(int i=0;i<n;i++){E[i]=15.*(i+1)/n;L[i]=2;s[i]=E[i]>10?.3:0;}
    CMFNonthermalRoute q={CMF_NT_EXCITATION,10,1,s};
    CMFNonthermalInput in={n,1,E,L,&q};CMFNonthermalResult r={0};
    assert(cmf_nonthermal_solve(&in,&r)==0);check(&r);
    double probability=r.events_per_eV[0]*15,exact=1-exp(-.3*5/2);
    printf("NT_EXCITATION n=%d P=%.17g analytic=%.17g error=%.17g energy_error=%.17g\n",
           n,probability,exact,fabs(probability-exact),r.energy_error);
    cmf_nonthermal_free(&r);free(p);return fabs(probability-exact);
}
static void simple(void){
    double E[]={1,2,4,8,16},L[]={2,3,4,5,6},s[]={0,0,0,1,1};
    CMFNonthermalInput in={5,0,E,L,NULL};CMFNonthermalResult r={0};
    assert(cmf_nonthermal_solve(&in,&r)==0);check(&r);close_to(r.heating_fraction,1,1e-15);
    close_to(r.thermal_electrons_per_eV,1./16,1e-16);
    CMFNonthermalRoute ion={CMF_NT_IONIZATION,5,1,s};in.n_routes=1;in.routes=&ion;
    assert(cmf_nonthermal_solve(&in,&r)==0);check(&r);
    assert(r.ionization_fraction>0&&r.thermal_electrons_per_eV>1./16);
    CMFNonthermalResult saved=r;ion.threshold_eV=.5;
    assert(cmf_nonthermal_solve(&in,&r)!=0);assert(!memcmp(&saved,&r,sizeof r));
    ion.threshold_eV=5;s[0]=-1;
    assert(cmf_nonthermal_solve(&in,&r)!=0);assert(!memcmp(&saved,&r,sizeof r));s[0]=0;
    E[1]=E[0];assert(cmf_nonthermal_solve(&in,&r)!=0);E[1]=2;
    cmf_nonthermal_free(&r);
    double a=excitation(999),b=excitation(3999);assert(b<a&&b<.001);
    puts("NT_ANALYTIC_AND_TRANSACTION_OK");
}
/* 독립 적분형 SF(Eq2)의 상삼각 사다리꼴 해. 생산 하강수송의 child
 * 보간/전자재분배 함수를 쓰지 않는다. 이 해의 유한격자 에너지 오차도
 * 정규화하지 않고 출력한다. Li/Hillier/Dessart2012 식2,3,5,9. */
static double hat_integral(const double *g,int n,int j,double lo,double hi){
    double value=0;
    if(j){
        double a=fmax(lo,g[j-1]),b=fmin(hi,g[j]);
        if(b>a)value+=(b-a)*(a+b-2*g[j-1])/(2*(g[j]-g[j-1]));
    }
    if(j+1<n){
        double a=fmax(lo,g[j]),b=fmin(hi,g[j+1]);
        if(b>a)value+=(b-a)*(2*g[j+1]-a-b)/(2*(g[j+1]-g[j]));
    }
    return value;
}
static double integral_oracle(const CMFNonthermalInput *in,double *balance){
    int n=in->n_energy;double *work=calloc((size_t)2*n,sizeof(double));assert(work);
    double *y=work,*w=work+n;
    for(int j=0;j<n;j++)w[j]=((j+1<n?in->energy_eV[j+1]:in->energy_eV[j])-
                             (j?in->energy_eV[j-1]:in->energy_eV[j]))/2;
    for(int i=n-1;i>=0;i--){
        double Ei=in->energy_eV[i],diag=in->coulomb_loss_eV_cm[i];
        double rhs=1/in->energy_eV[n-1];
        for(int j=i;j<n;j++){
            double Ej=in->energy_eV[j],element=0;
            for(int k=0;k<in->n_routes;k++){
                const CMFNonthermalRoute *q=&in->routes[k];
                double rate=q->lower_population_cm3*q->cross_section_cm2[j];if(rate==0)continue;
                double probability;
                if(q->kind==CMF_NT_EXCITATION){
                    element+=rate*hat_integral(in->energy_eV,n,j,Ei,Ei+q->threshold_eV);
                    continue;
                }
                else{
                    double J=.6*q->threshold_eV,U=(Ej-q->threshold_eV)/2,norm=atan(U/J);
                    double a=Ej-Ei-q->threshold_eV;
                    double p_primary=a<=0?1:a>=U?0:1-atan(a/J)/norm;
                    double p_secondary=Ei>=U?0:1-atan(Ei/J)/norm;
                    probability=p_primary-p_secondary;
                }
                double quad=j==i?(j+1<n?(in->energy_eV[j+1]-Ej)/2:0):w[j];
                element+=rate*probability*quad;
            }
            if(j==i)diag+=element;else rhs-=element*y[j];
        }
        assert(diag>0);y[i]=rhs/diag;assert(y[i]>=0&&isfinite(y[i]));
    }
    double ion=0,heat=in->energy_eV[0]*y[0]*in->coulomb_loss_eV_cm[0],exc=0;
    for(int j=0;j<n;j++){
        double E=in->energy_eV[j];heat+=y[j]*w[j]*in->coulomb_loss_eV_cm[j];
        for(int k=0;k<in->n_routes;k++){
            const CMFNonthermalRoute *q=&in->routes[k];
            double count=y[j]*w[j]*q->cross_section_cm2[j]*q->lower_population_cm3;
            if(q->kind==CMF_NT_IONIZATION){
                ion+=count*q->threshold_eV;
                if(count>0){
                    double J=.6*q->threshold_eV,U=(E-q->threshold_eV)/2;
                    double top=fmin(U,in->energy_eV[0]),norm=atan(U/J);
                    heat+=count*J*log1p(top*top/(J*J))/(2*norm);
                    if(E-q->threshold_eV<2*in->energy_eV[0]){
                        double lo=fmax(0,E-q->threshold_eV-in->energy_eV[0]);
                        double prob=(atan(U/J)-atan(lo/J))/norm;
                        double moment=J*log((J*J+U*U)/(J*J+lo*lo))/(2*norm);
                        heat+=count*((E-q->threshold_eV)*prob-moment);
                    }
                }
            }else{
                exc+=count*q->threshold_eV;
                if(E-q->threshold_eV<in->energy_eV[0])heat+=count*(E-q->threshold_eV);
            }
        }
    }
    *balance=heat+ion+exc-1;free(work);return ion;
}
static CMFNonthermalResult actual_mix(const CMFNonthermalAtomicData *a,int n,double xe){
    int selected[16],nr=0;
    for(int j=0;j<a->n_routes;j++)if((a->routes[j].Z==14||a->routes[j].Z==16)&&
        (a->routes[j].charge==1||a->routes[j].charge==2)){
        assert(nr<16);selected[nr++]=j;
    }
    assert(nr>=6);
    double *p=calloc((size_t)(nr+4)*n,sizeof(double));assert(p);
    double *E=p,*L=p+n,*sigma=p+2*n;
    CMFNonthermalRoute *q=calloc(nr+2,sizeof(*q));assert(q);
    for(int i=0;i<n;i++){
        E[i]=exp(log(.1)+log(20000)*i/(n-1));
        L[i]=cmf_nonthermal_coulomb(E[i],xe*1e8);assert(isfinite(L[i]));
    }
    E[n-1]=2000;
    for(int k=0;k<nr;k++){
        const CMFNonthermalAtomicRoute *r=&a->routes[selected[k]];
        q[k]=(CMFNonthermalRoute){CMF_NT_IONIZATION,r->threshold_eV,2.5e7,sigma+(size_t)k*n};
        for(int i=0;i<n;i++){sigma[(size_t)k*n+i]=cmf_nonthermal_atomic_sigma(r,E[i]);assert(isfinite(sigma[(size_t)k*n+i]));}
    }
    /* 임의 lower 인구의 성분 시험: 실제 SN 준위 인구/선 목록 아님. */
    for(int k=0;k<2;k++){
        q[nr+k]=(CMFNonthermalRoute){CMF_NT_EXCITATION,8.+3*k,2.5e7,sigma+(size_t)(nr+k)*n};
        for(int i=0;i<n;i++)sigma[(size_t)(nr+k)*n+i]=cmf_nonthermal_bethe(E[i],8.+3*k,.5,1);
    }
    CMFNonthermalInput in={n,nr+2,E,L,q};CMFNonthermalResult r={0};
    assert(cmf_nonthermal_solve(&in,&r)==0);check(&r);
    printf("NT_SI_S_MIX n=%d ne_over_fixed_ion_density=%.9g heat=%.17g ion=%.17g exc=%.17g "
           "ion_per_eV=%.17g energy_error=%.17g number_error=%.17g SN_population=0\n",
           n,xe,r.heating_fraction,r.ionization_fraction,r.excitation_fraction,r.ionizations_per_eV,
           r.energy_error,r.electron_number_error);
    if(xe==1){
        double balance,oracle=integral_oracle(&in,&balance);
        printf("NT_INTEGRAL_ORACLE n=%d integral_ion=%.17g cascade_ion=%.17g difference=%.17g integral_energy_error=%.17g\n",
               n,oracle,r.ionization_fraction,fabs(oracle-r.ionization_fraction),balance);
        if(n>=512)assert(fabs(oracle-r.ionization_fraction)<.005&&fabs(balance)<.01);
    }
    free(q);free(p);return r;
}
static void actual(const char *path){
    CMFNonthermalAtomicData a={0};assert(cmf_nonthermal_atomic_read(&a,path)==0);
    assert(a.n_routes>300);int tab=0,co_tab=0,si=0,sulfur=0;
    for(int i=0;i<a.n_routes;i++){
        CMFNonthermalAtomicRoute *r=&a.routes[i];
        if(r->n_tab){tab++;if(r->Z==27)co_tab++;for(int k=0;k<r->n_tab;k++)close_to(cmf_nonthermal_atomic_sigma(r,r->tab_energy[k]),r->tab_sigma[k],1e-28);}
        if(r->Z==14&&r->charge==1&&r->n==3&&r->l==1){si++;close_to(r->threshold_eV,16.3,1e-12);assert(strstr(r->terms,"3s2_1Se"));}
        if(r->Z==16&&r->charge==1)sulfur++;
    }
    assert(tab==15&&co_tab==6&&si==1&&sulfur>0);
    CMFNonthermalAtomicData saved=a;assert(cmf_nonthermal_atomic_read(&a,"missing-nt-file")!=0);
    assert(!memcmp(&a,&saved,sizeof a));
    printf("NT_ACTUAL_ATOMIC_DATA routes=%d tabulated=%d SiII=%d SII=%d mapped_to_NLTE=0\n",a.n_routes,tab,si,sulfur);
    CMFNonthermalResult r[3];int grid[]={128,256,512};
    for(int k=0;k<3;k++)r[k]=actual_mix(&a,grid[k],1);
    double d0=fabs(r[1].ionization_fraction-r[0].ionization_fraction),d1=fabs(r[2].ionization_fraction-r[1].ionization_fraction);
    printf("NT_GRID_CHANGE ion_128_256=%.17g ion_256_512=%.17g\n",d0,d1);
    assert(d1<d0&&d1<.005);
    CMFNonthermalResult weak=actual_mix(&a,256,.01),strong=actual_mix(&a,256,100);
    assert(weak.heating_fraction<r[1].heating_fraction&&r[1].heating_fraction<strong.heating_fraction);
    assert(weak.ionizations_per_eV>strong.ionizations_per_eV);
    cmf_nonthermal_free(&weak);cmf_nonthermal_free(&strong);
    for(int k=0;k<3;k++)cmf_nonthermal_free(&r[k]);
    cmf_nonthermal_atomic_free(&a);
}
int main(int argc,char **argv){
    primitives();simple();if(argc==2)actual(argv[1]);else assert(argc==1);
    puts("CMF_NONTHERMAL_SELFTEST_OK NLTE_connected=0 spectrum_compared=0");return 0;
}
