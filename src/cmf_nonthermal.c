#include "cmf_nonthermal.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static const double pi=3.14159265358979323846;

double cmf_nonthermal_coulomb(double E,double ne){
    if(!(E>0)||!(ne>=0)||!isfinite(E)||!isfinite(ne))return NAN;
    if(ne==0)return 0;
    const double ev=1.602176634e-12,me=9.1093837139e-28,q=4.803204712570263e-10,hbar=1.054571817e-27;
    double erg=E*ev,omega=sqrt(4*pi*ne*q*q/me),v=sqrt(2*erg/me);
    double arg=E<14?me*v*v*v/(.577215665*q*q*omega):2*erg/(hbar*omega);
    double L=ne*2*pi*q*q*q*q/erg*log(arg)/ev;
    return L>=0&&isfinite(L)?L:NAN;
}
double cmf_nonthermal_arnaud(double E,double I,const double a[4]){
    if(!(E>0)||!(I>0)||!isfinite(E)||!isfinite(I)||!a)return NAN;
    for(int k=0;k<4;k++)if(!isfinite(a[k]))return NAN;
    if(E<=I)return 0;
    double z=(E-I)/I,u=1+z,t=z/u,l=log1p(z);
    double s=1e-14*(a[0]*t+a[1]*t*t+a[2]*l+a[3]*l/u)/(u*I*I);
    return s>=0&&isfinite(s)?s:NAN; /* 원본 음수 단면적을0으로 바꾸지 않는다. */
}
double cmf_nonthermal_bethe(double E,double d,double f,int charge){
    if(!(E>0)||!(d>0)||!(f>=0)||charge<0||!isfinite(E)||!isfinite(d)||!isfinite(f))return NAN;
    if(E<d||f==0)return 0;
    double x2=E/d-1,g;
    if(charge>0&&x2<=1.2212243*1.2212243)g=.2;
    else if(x2<=.64){double x=sqrt(x2);g=.074*x*(1+x);}
    else if(x2<=36){double x=sqrt(x2);g=-.0745397+x*(.232715-.00647558*x);}
    else g=.105+log(x2)/3.6276;
    double a0=5.29177210544e-9;
    return 8*pi*pi*a0*a0*13.6*13.6/sqrt(3.)*f*g/(d*E);
}
double cmf_nonthermal_omega(double E,double d,double omega,double g){
    if(!(E>0)||!(d>0)||!(omega>=0)||!(g>0)||!isfinite(E)||!isfinite(d)||!isfinite(omega)||!isfinite(g))return NAN;
    const double a0=5.29177210544e-9;
    return E<d?0:pi*a0*a0*13.6*omega/(g*E);
}
void cmf_nonthermal_free(CMFNonthermalResult *r){
    if(r){free(r->storage);memset(r,0,sizeof(*r));}
}
static int point(const double *grid,int n,long double E,long double weight,
                 long double *dest,long double *heat,long double *thermal){
    if(E<0||E>grid[n-1]||weight<0)return -1;
    if(E<grid[0]){*heat+=weight*E;*thermal+=weight;return 0;}
    int lo=0,hi=n-1;
    while(hi-lo>1){int mid=lo+(hi-lo)/2;if(grid[mid]<=E)lo=mid;else hi=mid;}
    if(n==1){dest[0]+=weight;return 0;}
    long double w=(E-grid[lo])/(grid[hi]-grid[lo]);
    dest[lo]+=weight*(1-w);dest[hi]+=weight*w;return 0;
}
/* 각 구간의 확률과 에너지1차 모멘트를 해석 적분한다. 보간 확률은
 * 양수이고 두 전자의 수와 에너지를 보존한다. Opal: J=0.6I. */
static void probability(long double a,long double b,long double J,long double norm,
                        long double *P,long double *M){
    *P=atanl((b-a)*J/(J*J+a*b))/norm;
    *M=J*log1pl((b-a)*(a+b)/(J*J+a*a))/(2*norm);
}
static int ion_products(const double *g,int n,long double E,long double I,
                        long double *dest,long double *heat,long double *thermal){
    long double total=E-I,upper=total/2,J=.6L*I,norm=atanl(upper/J);
    if(!(total>0)||!(norm>0))return -1;
    for(int primary=0;primary<2;primary++){
        long double offset=primary?total:0;
        /* k=-1: 최저 노드보다 작은 전자는 모든 전이 문턱 아래다. */
        for(int k=-1;k<n-1;k++){
            long double elo=k<0?0:g[k],ehi=g[k+1];
            long double a=primary?offset-ehi:elo,b=primary?offset-elo:ehi;
            if(a<0)a=0;
            if(b>upper)b=upper; /* 분포 지지집합과 구간의 교집합 */
            if(!(b>a))continue;
            long double P,M;probability(a,b,J,norm,&P,&M);
            long double energy=primary?offset*P-M:M;
            if(k<0){*heat+=energy;*thermal+=P;}
            else{
                long double wl=(ehi*P-energy)/(ehi-elo),wh=(energy-elo*P)/(ehi-elo);
                if(wl<0||wh<0)return -1;
                dest[k]+=wl;dest[k+1]+=wh;
            }
        }
    }
    long double count=*thermal,sum=*heat;
    for(int k=0;k<n;k++){count+=dest[k];sum+=g[k]*dest[k];}
    if(fabsl(count-2)>2e-12L||fabsl(sum-total)>2e-12L*E)return -1;
    return 0;
}
int cmf_nonthermal_solve(const CMFNonthermalInput *in,CMFNonthermalResult *out){
    if(!in||!out||in->n_energy<2||in->n_energy>100000||in->n_routes<0||
       !in->energy_eV||!in->coulomb_loss_eV_cm||(in->n_routes&&!in->routes))return -1;
    int n=in->n_energy,nr=in->n_routes;
    if((size_t)nr>(SIZE_MAX/sizeof(double)-n)/2)return -1;
    for(int i=0;i<n;i++){
        if(!(in->energy_eV[i]>0)||!isfinite(in->energy_eV[i])||
           (i&&!(in->energy_eV[i]>in->energy_eV[i-1]))||
           !(in->coulomb_loss_eV_cm[i]>=0)||!isfinite(in->coulomb_loss_eV_cm[i]))return -1;
    }
    for(int r=0;r<nr;r++){
        const CMFNonthermalRoute *q=&in->routes[r];
        if((q->kind!=CMF_NT_EXCITATION&&q->kind!=CMF_NT_IONIZATION)||!q->cross_section_cm2||
           !(q->threshold_eV>=in->energy_eV[0])||!isfinite(q->threshold_eV)||
           !(q->lower_population_cm3>=0)||!isfinite(q->lower_population_cm3))return -1;
        for(int i=0;i<n;i++)if(!(q->cross_section_cm2[i]>=0)||!isfinite(q->cross_section_cm2[i])||
           (in->energy_eV[i]<q->threshold_eV&&q->cross_section_cm2[i]!=0)||
           (q->kind==CMF_NT_IONIZATION&&in->energy_eV[i]==q->threshold_eV&&q->cross_section_cm2[i]!=0))return -1;
    }
    CMFNonthermalResult a={.n_energy=n,.n_routes=nr};
    a.storage=calloc((size_t)n+2*(size_t)nr,sizeof(double));
    long double *work=calloc((size_t)3*n+nr,sizeof(long double));
    if(!a.storage||!work){free(work);cmf_nonthermal_free(&a);return -1;}
    a.path_cm_per_eV=a.storage;a.events_per_eV=a.storage+n;a.coefficient_cm3_per_eV=a.events_per_eV+nr;
    long double *incoming=work,*dest=work+n,*one=work+2*n,*events=work+3*n;
    long double heat=0,terminal=0;
    incoming[n-1]=1.L/in->energy_eV[n-1];
    for(int i=n-1;i>=0;i--){
        double E=in->energy_eV[i],delta=E-(i?in->energy_eV[i-1]:0);
        long double coul=(long double)in->coulomb_loss_eV_cm[i]/delta;
        long double denom=coul,heat_coefficient=coul*delta,thermal_coefficient=i?0:coul;
        memset(dest,0,(size_t)(i+1)*sizeof(*dest));
        if(i)dest[i-1]=coul;
        for(int r=0;r<nr;r++){
            const CMFNonthermalRoute *q=&in->routes[r];
            long double rate=(long double)q->lower_population_cm3*q->cross_section_cm2[i];
            if(rate==0)continue;
            long double h=0,t=0;
            memset(one,0,(size_t)(i+1)*sizeof(*one));
            int rc=q->kind==CMF_NT_IONIZATION?
                ion_products(in->energy_eV,i+1,E,q->threshold_eV,one,&h,&t):
                point(in->energy_eV,i+1,(long double)E-q->threshold_eV,1,one,&h,&t);
            if(rc||one[i]>=1)goto fail;
            denom+=rate*(1-one[i]);heat_coefficient+=rate*h;thermal_coefficient+=rate*t;
            for(int k=0;k<i;k++)dest[k]+=rate*one[k];
        }
        if(!(denom>0)){if(incoming[i]==0)continue;goto fail;}
        long double path=incoming[i]/denom;
        if(path<0||!isfinite((double)path))goto fail;
        a.path_cm_per_eV[i]=(double)path;
        heat+=path*heat_coefficient;terminal+=path*thermal_coefficient;
        for(int r=0;r<nr;r++){
            long double coeff=path*in->routes[r].cross_section_cm2[i];
            a.coefficient_cm3_per_eV[r]+=(double)coeff;
            events[r]+=coeff*in->routes[r].lower_population_cm3;
        }
        for(int k=0;k<i;k++)incoming[k]+=path*dest[k];
    }
    long double ion=0,exc=0,ioncount=0;
    for(int r=0;r<nr;r++){
        a.events_per_eV[r]=(double)events[r];
        if(in->routes[r].kind==CMF_NT_IONIZATION){ion+=events[r]*in->routes[r].threshold_eV;ioncount+=events[r];}
        else exc+=events[r]*in->routes[r].threshold_eV;
        if(!isfinite(a.events_per_eV[r])||!isfinite(a.coefficient_cm3_per_eV[r]))goto fail;
    }
    a.heating_fraction=(double)heat;a.ionization_fraction=(double)ion;a.excitation_fraction=(double)exc;
    a.ionizations_per_eV=(double)ioncount;a.thermal_electrons_per_eV=(double)terminal;
    a.energy_error=(double)(heat+ion+exc-1);
    a.electron_number_error=(double)(terminal-ioncount-1.L/in->energy_eV[n-1]);
    if(!isfinite(a.energy_error)||fabsl(heat+ion+exc-1)>1e-10L||
       fabsl(terminal-ioncount-1.L/in->energy_eV[n-1])>1e-10L*(terminal+ioncount))goto fail;
    free(work);cmf_nonthermal_free(out);*out=a;return 0;
fail:
    free(work);cmf_nonthermal_free(&a);return -1;
}
