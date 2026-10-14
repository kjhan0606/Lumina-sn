#ifndef CMF_OBSERVER_RAY_H
#define CMF_OBSERVER_RAY_H
#include "cmf_observer.h"
#include <math.h>
#ifdef __CUDACC__
#define OBS_HD __host__ __device__
#else
#define OBS_HD
#endif
#define OBS_C 2.99792458e10

static OBS_HD double obs_q(double p, double z, double ict) {
    const double u=z*ict, w=p*ict;
    return (1-u)/sqrt(1-u*u-w*w);
}
static OBS_HD double obs_bb(double nu, double temp) {
    if (temp==0) return 0;
    const double x=6.62607015e-27*nu/(1.380649e-16*temp);
    /* 큰 x에서도 nu^3/expm1(x)의 inf/inf가 생기지 않는 표현. */
    return (2*6.62607015e-27/OBS_C/OBS_C)*nu*nu*nu*
           exp(-x)/(-expm1(-x));
}

/* 하나의 단조 Doppler 구간. 각 fine-bin 경계의 위치를 해석적으로 구한다.
 * χobs=qχ, ηobs=(χS+χes J)/q². 공변식의 3a 감쇠를 여기서 중복하지 않는다. */
static OBS_HD double obs_leg(const CMFObserverInput *in, int shell,
        double p, double z0, double z1, double nu, double ict, int rising,
        double I) {
    if (!(z1>z0)) return I;
    double q0=obs_q(p,z0,ict), q1=obs_q(p,z1,ict);
    int b=(int)floor((log(nu/in->nu_min)+log(q0))/in->dlognu);
    const double wp=p*ict;
    for (int visit=0; visit<in->nf+3; ++visit) {
        if (b<0 || b>=in->nf) return NAN;
        double end=z1;
        int crossing=0;
        if (ict>0) {
            const double qe=(in->nu_min/nu)*exp((b+(rising?1:0))*in->dlognu);
            crossing=rising ? qe<q1 : qe>q1;
            if (crossing) {
                const double disc=qe*qe*(1-wp*wp)-wp*wp;
                if (disc<0) return NAN;
                const double rad=qe*sqrt(disc);
                const double u=rising ? (1+rad)/(1+qe*qe)
                    : ((1-qe)*(1+qe)+qe*qe*wp*wp)/(1+rad);
                end=u/ict;
                /* 경계 반올림으로 시작점과 같거나 조금 앞이면 다음 bin으로.
                 * 입력의 물리량을 clamp하거나 범위 밖 값을 외삽하지 않는다. */
                if (end<=z0) { b+=rising?1:-1; continue; }
                if (end>z1) return NAN;
            }
        }
        const size_t k=(size_t)shell*in->nf+b;
        const double ds=(end-z0)/in->substeps;
        for (int j=0;j<in->substeps;++j) {
            const double q=obs_q(p,z0+(j+0.5)*ds,ict);
            const double tau=q*in->chi[k]*ds;
            const double fixed=in->source_is_emissivity?in->source[k]:in->chi[k]*in->source[k];
            const double eta=(fixed+in->es[k]*in->J[k])/(q*q);
            const double phi=tau==0?1:-expm1(-tau)/tau;
            I=I*exp(-tau)+eta*ds*phi;
        }
        if (!isfinite(I) || I<0) return NAN;
        if (!crossing) return I;
        z0=end;
        b+=rising?1:-1;
    }
    return NAN;
}

static OBS_HD double obs_segment(const CMFObserverInput *in, int s,
        double p, double a, double b, double nu, double ict, double I) {
    if (ict==0) return obs_leg(in,s,p,a,b,nu,ict,0,I);
    const double w=p*ict, turn=(1-w*w)/ict;
    /* 상대론적 큰 p에서도 q의 극소점을 건너뛰지 않는다. */
    if (a<turn && turn<b) {
        I=obs_leg(in,s,p,a,turn,nu,ict,0,I);
        return obs_leg(in,s,p,turn,b,nu,ict,1,I);
    }
    return obs_leg(in,s,p,a,b,nu,ict,a>=turn,I);
}

static OBS_HD double cmf_observer_ray(const CMFObserverInput *in,
        double p, double nu) {
    const double ict=in->t_exp==0?0:1/(OBS_C*in->t_exp);
    const int core=p<in->ri[0];
    double I=0;
    if (core) {
        const double z=sqrt((in->ri[0]-p)*(in->ri[0]+p));
        const double q=obs_q(p,z,ict);
        I=in->inner_scale*obs_bb(q*nu,in->T_inner)/(q*q*q);
    } else {
        for(int s=in->ns-1;s>=0;--s) {
            if(p>=in->ro[s])continue;
            double zo=sqrt((in->ro[s]-p)*(in->ro[s]+p));
            double zi=p<in->ri[s]?sqrt((in->ri[s]-p)*(in->ri[s]+p)):0;
            I=obs_segment(in,s,p,-zo,-zi,nu,ict,I);
        }
    }
    for(int s=0;s<in->ns;++s) {
        if(p>=in->ro[s])continue;
        double zo=sqrt((in->ro[s]-p)*(in->ro[s]+p));
        double zi=p<in->ri[s]?sqrt((in->ri[s]-p)*(in->ri[s]+p)):0;
        I=obs_segment(in,s,p,zi,zo,nu,ict,I);
    }
    return I;
}
#undef OBS_HD
#endif
