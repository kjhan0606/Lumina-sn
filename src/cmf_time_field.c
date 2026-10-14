#include "cmf_time_field.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define C_LIGHT 2.99792458e10

void cmf_time_radiation_trial_free(CMFTimeRadiationTrial *t){
    if(t){free(t->J);free(t->H);memset(t,0,sizeof(*t));}
}

static double relative(double a,double b){
    double scale=fmax(fabs(a),fabs(b));return scale?fabs(a-b)/scale:0;
}
static double row_residual(const long double *v,int n){
    long double sum=0,norm=0;
    for(int i=0;i<n;i++){sum+=v[i];norm+=fabsl(v[i]);}
    return norm?(double)(fabsl(sum)/norm):0;
}

/* 원래 시간식/면식으로 잔차를 구한다. 소거한 삼중대각 RHS를 재사용하지 않는다. */
static double equation_residual(const CMFTimeMomentInput *p,const double *J,const double *H){
    int ns=p->n_shells,nb=p->n_bins;double worst=0;
    long double inv=1.0L/(C_LIGHT*(p->time_new-p->time_old));
    long double dilution=powl(p->time_old/p->time_new,3);
    for(int k=0;k<nb;k++){
        long double drift=p->frequency_edges[k+1]/
            (p->frequency_edges[k+1]-p->frequency_edges[k])/(C_LIGHT*p->time_new);
        if(H[k]!=p->inner_H[k])return INFINITY;
        for(int s=0;s<ns;s++){
            size_t i=(size_t)s*nb+k;long double l=p->radius_edges[s],r=p->radius_edges[s+1];
            long double vol=(r-l)*(r*r+r*l+l*l)/3;
            long double v[]={inv*J[i],-inv*dilution*p->old_J[i],
                r*r*H[i+nb]/vol,-l*l*H[i]/vol,drift*J[i],
                k+1<nb?-drift*J[i+1]:0,
                (p->chi_total[i]-p->chi_scattering[i])*J[i],-p->eta_fixed[i]};
            worst=fmax(worst,row_residual(v,8));
        }
        for(int s=1;s<=ns;s++){
            size_t face=(size_t)s*nb+k,il=face-nb;
            double rl=.5*(p->radius_edges[s-1]+p->radius_edges[s]);
            double rr=s<ns?.5*(p->radius_edges[s]+p->radius_edges[s+1]):p->radius_edges[s];
            double fl=p->eddington[il],fr=s<ns?p->eddington[face]:p->outer_f[k];
            double logq=.25*((3-1/fl)+(3-1/fr))*log(rr/rl);
            double ext=p->chi_total[il];long double jr;
            if(s<ns){double w=(p->radius_edges[s]-rl)/(rr-rl);
                ext=(1-w)*ext+w*p->chi_total[face];jr=J[face];}
            else if(p->outer_h[k]>0)jr=H[face]/p->outer_h[k];
            else {if(H[face]!=0)return INFINITY;continue;}
            long double v[]={ext*H[face],inv*H[face],-inv*dilution*p->old_H[face],
                drift*H[face],k+1<nb?-drift*H[face+1]:0,
                fr*exp(logq)*jr/(rr-rl),-fl*exp(-logq)*J[il]/(rr-rl)};
            worst=fmax(worst,row_residual(v,7));
        }
    }
    return worst;
}

CMFTimeMomentStatus cmf_time_field_solve(const CMFTimeMomentInput *p,
    const double *initial_J,CMFTimeFormalApply formal,void *context,
    int cap,double tol,double *new_J,double *new_H,double *delta,CMFTimeFieldReport *report){
    CMFTimeFieldReport rep={0};rep.iteration_cap=cap;rep.tolerance=tol;
    rep.relative_update=rep.closure_update=rep.equation_residual=INFINITY;
    if(report)*report=rep;
    if(!p||!initial_J||!formal||!new_J||!new_H||!delta||!report||cap<2||
       new_J==new_H||new_J==delta||new_H==delta||
       !(tol>0&&tol<1)||!isfinite(tol)||p->n_shells<1||p->n_bins<1||
       !p->old_J||!p->old_H||!p->eta_fixed||!p->inner_H||
       !p->radius_edges||!p->frequency_edges||!p->chi_total||!p->chi_scattering||
       !(p->time_old>0)||!(p->time_new>p->time_old)||!isfinite(p->time_new))return CMF_TIME_INVALID;
    size_t ns=p->n_shells,nb=p->n_bins;
    if(ns+1>SIZE_MAX/nb||(ns+1)*nb>SIZE_MAX/(12*sizeof(double)))return CMF_TIME_INVALID;
    size_t cells=ns*nb,faces=(ns+1)*nb;
    for(size_t i=0;i<cells;i++)if(!(initial_J[i]>=0)||!isfinite(initial_J[i]))return CMF_TIME_INVALID;
    double *mem=calloc(8*cells+faces+5*nb,sizeof(double));
    if(!mem)return CMF_TIME_ALLOCATION;
    double *iter=mem,*next=iter+cells,*h=next+cells,*mj=h+faces,*mh=mj+cells,*mk=mh+cells;
    double *f=mk+cells,*priorf=f+cells,*change=priorf+cells;
    double *oj=change+cells,*oh=oj+nb,*ok=oh+nb,*ho=ok+nb,*fo=ho+nb;
    CMFFormalMoments m={mj,mh,mk,oj,oh,ok,NULL};
    memcpy(iter,initial_J,cells*sizeof(double));
    CMFTimeMomentInput work=*p;work.eddington=f;work.outer_h=ho;work.outer_f=fo;
    CMFTimeMomentStatus status=CMF_TIME_NOT_CONVERGED;
    for(int it=1;it<=cap;it++){
        if(formal(context,iter,&m)){status=CMF_TIME_INVALID;break;}
        rep.closure_update=it==1?INFINITY:0;
        for(size_t i=0;i<cells;i++){
            if(!(mj[i]>0)||!(mk[i]>0)||!isfinite(mj[i])||!isfinite(mk[i])||mk[i]>mj[i]){
                status=CMF_TIME_INVALID;goto done;
            }
            f[i]=mk[i]/mj[i];
            if(it>1)rep.closure_update=fmax(rep.closure_update,relative(f[i],priorf[i]));
        }
        for(size_t b=0;b<nb;b++){
            if(!(oj[b]>0)||!(ok[b]>0)||!isfinite(oj[b])||!isfinite(oh[b])||
               !isfinite(ok[b])||oh[b]<0||ok[b]>oj[b]){status=CMF_TIME_INVALID;goto done;}
            double nh=oh[b]/oj[b],nf=ok[b]/oj[b];
            if(it>1){rep.closure_update=fmax(rep.closure_update,relative(nh,ho[b]));
                rep.closure_update=fmax(rep.closure_update,relative(nf,fo[b]));}
            ho[b]=nh;fo[b]=nf;
        }
        status=cmf_time_moment_step(&work,next,h);if(status!=CMF_TIME_OK)break;
        rep.iterations=it;rep.relative_update=0;
        for(size_t i=0;i<cells;i++){
            change[i]=fabs(next[i]-iter[i]);
            rep.relative_update=fmax(rep.relative_update,relative(next[i],iter[i]));
        }
        if(it>1&&rep.relative_update<tol&&rep.closure_update<tol){
            rep.equation_residual=equation_residual(&work,next,h);
            if(!isfinite(rep.equation_residual)||rep.equation_residual>1e-10){status=CMF_TIME_NONFINITE;break;}
            memcpy(new_J,next,cells*sizeof(double));memcpy(new_H,h,faces*sizeof(double));
            memcpy(delta,change,cells*sizeof(double));rep.converged=1;status=CMF_TIME_OK;break;
        }
        memcpy(iter,next,cells*sizeof(double));memcpy(priorf,f,cells*sizeof(double));
        status=CMF_TIME_NOT_CONVERGED;
    }
done:
    *report=rep;free(mem);return status;
}
