#include "cmf_time_moment.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LIGHT_C 2.99792458e10

/* q is the integrating factor for dK/dr+(3-1/f)K/r:
 * dlog(q)/dlog(r)=3-1/f. Only local ratios are formed. */
static double half_log_q(double left, double right, double fl, double fr)
{
    return 0.25*((3.0-1.0/fl)+(3.0-1.0/fr))*log(right/left);
}

CMFTimeMomentStatus cmf_time_moment_step(const CMFTimeMomentInput *p,
                                        double *new_J, double *new_H)
{
    if (!p || !new_J || !new_H || new_J==new_H || p->n_shells<1 || p->n_bins<1 ||
        !(p->time_old>0) || !(p->time_new>p->time_old) || !isfinite(p->time_new) ||
        !p->radius_edges || !p->frequency_edges || !p->chi_total ||
        !p->chi_scattering || !p->eta_fixed || !p->eddington || !p->old_J ||
        !p->old_H || !p->inner_H || !p->outer_h || !p->outer_f) return CMF_TIME_INVALID;
    const size_t ns=(size_t)p->n_shells, nb=(size_t)p->n_bins;
    if ((ns+1)>SIZE_MAX/nb || (ns+1)*nb>SIZE_MAX/sizeof(double) ||
        ns+1>SIZE_MAX/(8*sizeof(double))) return CMF_TIME_INVALID;
    const size_t cells=ns*nb, faces=(ns+1)*nb;
    const double *re=p->radius_edges, *nu=p->frequency_edges;
    if (!(re[0]>=0) || !(nu[0]>0)) return CMF_TIME_INVALID;
    for (size_t s=0;s<ns;s++)
        if (!isfinite(re[s]) || !isfinite(re[s+1]) || !(re[s+1]>re[s]))
            return CMF_TIME_INVALID;
    for (size_t b=0;b<nb;b++) {
        double h=p->outer_h[b], f=p->outer_f[b];
        if (!isfinite(nu[b]) || !isfinite(nu[b+1]) || !(nu[b+1]>nu[b]) ||
            !isfinite(p->inner_H[b]) || !isfinite(h) || !isfinite(f) ||
            !(h>=0 && h<=1) || !(f>0 && f<=1) || h*h>f+1e-14)
            return CMF_TIME_INVALID;
    }
    for (size_t i=0;i<cells;i++) {
        double ct=p->chi_total[i], cs=p->chi_scattering[i], eta=p->eta_fixed[i];
        double f=p->eddington[i], j=p->old_J[i];
        if (!isfinite(ct) || !isfinite(cs) || !isfinite(eta) || !isfinite(j) ||
            !isfinite(f) || !(ct>=cs && cs>=0) || !(eta>=0) || !(j>=0) ||
            !(f>0 && f<=1)) return CMF_TIME_INVALID;
    }
    for (size_t i=0;i<faces;i++) if (!isfinite(p->old_H[i])) return CMF_TIME_INVALID;

    double *jout=malloc(cells*sizeof(double)), *hout=malloc(faces*sizeof(double));
    double *scratch=malloc(8*(ns+1)*sizeof(double));
    if (!jout || !hout || !scratch) {
        free(jout);free(hout);free(scratch);return CMF_TIME_ALLOCATION;
    }
    double *a=scratch, *b=a+ns+1, *constant=b+ns+1, *diagonal=constant+ns+1;
    double *upper=diagonal+ns+1, *rhs=upper+ns+1, *mid=rhs+ns+1, *volume=mid+ns+1;
    const double inverse_cdt=1.0/(LIGHT_C*(p->time_new-p->time_old));
    const double radius_ratio=p->time_old/p->time_new;
    const double dilution=radius_ratio*radius_ratio*radius_ratio;
    CMFTimeMomentStatus status=CMF_TIME_OK;
    if (!(inverse_cdt>0) || !isfinite(inverse_cdt) || !(dilution>0)) {
        status=CMF_TIME_NONFINITE;goto cleanup;
    }
    for (size_t s=0;s<ns;s++) {
        mid[s]=0.5*re[s]+0.5*re[s+1];
        volume[s]=(re[s+1]-re[s])*(re[s+1]*re[s+1]+re[s]*re[s+1]+re[s]*re[s])/3;
        if (!(mid[s]>0) || !(volume[s]>0) || !isfinite(volume[s])) {
            status=CMF_TIME_NONFINITE;goto cleanup;
        }
    }
    for (int ib=p->n_bins-1;ib>=0;ib--) {
        const size_t k=(size_t)ib;
        /* Bin-integrated -nu*dJ/dnu: red face uses J_k, blue face J_(k+1).
         * Sum over frequency leaves expansion work plus the red-edge escape. */
        const double drift=nu[k+1]/(nu[k+1]-nu[k])/(LIGHT_C*p->time_new);
        const double memory=inverse_cdt*dilution;
        a[0]=b[0]=0; constant[0]=p->inner_H[k];
        for (size_t face=1;face<ns;face++) {
            size_t il=(face-1)*nb+k, ir=face*nb+k, ih=face*nb+k;
            double dr=mid[face]-mid[face-1];
            double w=(re[face]-mid[face-1])/dr;
            double extinction=(1-w)*p->chi_total[il]+w*p->chi_total[ir]+inverse_cdt+drift;
            double logq=half_log_q(mid[face-1],mid[face],p->eddington[il],p->eddington[ir]);
            a[face]=p->eddington[il]*exp(-logq)/(extinction*dr);
            b[face]=p->eddington[ir]*exp(logq)/(extinction*dr);
            constant[face]=(memory*p->old_H[ih]+(k+1<nb ? drift*hout[ih+1] : 0))/extinction;
        }
        a[ns]=b[ns]=constant[ns]=0;
        if (p->outer_h[k]>0) {
            size_t il=(ns-1)*nb+k, ih=ns*nb+k;
            double dr=re[ns]-mid[ns-1], f=p->eddington[il];
            double extinction=p->chi_total[il]+inverse_cdt+drift;
            double logq=half_log_q(mid[ns-1],re[ns],f,p->outer_f[k]);
            double denominator=extinction*dr+p->outer_f[k]*exp(logq)/p->outer_h[k];
            a[ns]=f*exp(-logq)/denominator;
            constant[ns]=dr*(memory*p->old_H[ih]+(k+1<nb ? drift*hout[ih+1] : 0))/denominator;
        }
        for (size_t s=0;s<ns;s++) {
            size_t i=s*nb+k;
            double left=re[s]*re[s]/volume[s], right=re[s+1]*re[s+1]/volume[s];
            double lower=-left*a[s];
            diagonal[s]=p->chi_total[i]-p->chi_scattering[i]+inverse_cdt+drift
                        +right*a[s+1]+left*b[s];
            upper[s]=-right*b[s+1];
            rhs[s]=p->eta_fixed[i]+memory*p->old_J[i]
                   +(k+1<nb ? drift*jout[i+1] : 0)-right*constant[s+1]+left*constant[s];
            if (s) {
                double factor=lower/diagonal[s-1];
                diagonal[s]-=factor*upper[s-1];
                rhs[s]-=factor*rhs[s-1];
            }
            if (!(diagonal[s]>0) || !isfinite(diagonal[s]) || !isfinite(rhs[s])) {
                status=CMF_TIME_NONFINITE;goto cleanup;
            }
        }
        for (int is=p->n_shells-1;is>=0;is--) {
            size_t s=(size_t)is, i=s*nb+k;
            jout[i]=(rhs[s]-(s+1<ns ? upper[s]*jout[i+nb] : 0))/diagonal[s];
            if (!isfinite(jout[i])) {status=CMF_TIME_NONFINITE;goto cleanup;}
            if (jout[i]<0) {status=CMF_TIME_NEGATIVE;goto cleanup;}
        }
        hout[k]=constant[0];
        for (size_t face=1;face<=ns;face++) {
            size_t i=face*nb+k;
            hout[i]=a[face]*jout[i-nb]+constant[face]-(face<ns ? b[face]*jout[i] : 0);
            if (!isfinite(hout[i])) {status=CMF_TIME_NONFINITE;goto cleanup;}
        }
    }
    memcpy(new_J,jout,cells*sizeof(double));memcpy(new_H,hout,faces*sizeof(double));
cleanup:
    free(jout);free(hout);free(scratch);return status;
}
