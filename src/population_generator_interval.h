#ifndef LUMINA_POPULATION_GENERATOR_INTERVAL_H
#define LUMINA_POPULATION_GENERATOR_INTERVAL_H
/* Outward-rounded fixed-binary64-rate nominal root certification.
 * This does not certify rate uncertainty or coupled physical convergence. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#if defined(__x86_64__) || defined(__i386__)
#include <fpu_control.h>
#endif

typedef struct { long double lo, hi; } PGIInterval;
static inline PGIInterval pgi_exact(long double x) { return (PGIInterval){x,x}; }
static inline PGIInterval pgi_add(PGIInterval a,PGIInterval b) {
    if(a.hi==0)return b;
    if(b.hi==0)return a;
    volatile long double low=a.lo+b.lo,high=a.hi+b.hi;
    return (PGIInterval){low==0?0:nextafterl(low,0),nextafterl(high,INFINITY)};
}
static inline PGIInterval pgi_multiply(PGIInterval a,PGIInterval b) {
    if(a.hi==0||b.hi==0)return pgi_exact(0);
    volatile long double low=a.lo*b.lo,high=a.hi*b.hi;
    return (PGIInterval){low==0?0:nextafterl(low,0),nextafterl(high,INFINITY)};
}
static inline PGIInterval pgi_divide(PGIInterval a,PGIInterval b) {
    volatile long double low=a.lo/b.hi,high=a.hi/b.lo;
    return (PGIInterval){low==0?0:nextafterl(low,0),a.hi==0?0:nextafterl(high,INFINITY)};
}
static inline int pgi_valid(PGIInterval a) {
    return a.lo>=0&&a.lo<=a.hi&&isfinite(a.lo)&&isfinite(a.hi);
}

static inline int pgi_precision(void) { return LDBL_MANT_DIG; }
static inline int pgi_environment(void) {
    #if defined(__FAST_MATH__)
    return 3;
#endif
    if(FLT_RADIX!=2||LDBL_MANT_DIG<64||fegetround()!=FE_TONEAREST)return 1;
#if (defined(__x86_64__) || defined(__i386__)) && LDBL_MANT_DIG == 64
    fpu_control_t word;
    _FPU_GETCW(word);
    if((word & 0x300U)!=0x300U)return 2;
#endif
    return 0;
}

static inline int pgi_stationary(const double *column_generator,size_t n,
                                 long double *lower,long double *upper) {
    if(!column_generator||!lower||!upper||n<2||n>1000||pgi_environment())return 1;
    PGIInterval *work=(PGIInterval*)calloc(n*n,sizeof(*work)),*denominator=(PGIInterval*)calloc(n,sizeof(*denominator));
    PGIInterval *p=(PGIInterval*)calloc(n,sizeof(*p));
    if(!work||!denominator||!p){free(work);free(denominator);free(p);return 2;}
    int status=0;
    for(size_t source=0;source<n;source++)for(size_t dest=0;dest<n;dest++){
        if(source==dest)continue;
        double value=column_generator[source*n+dest];
        if(!isfinite(value)||value<0){status=3;goto done;}
        work[source*n+dest]=pgi_exact((long double)value);
    }
    for(size_t source=0;source<n;source++){
        double diagonal=column_generator[source*n+source];
        long double outflow=0;
        for(size_t dest=0;dest<n;dest++)if(dest!=source)outflow+=work[source*n+dest].lo;
        long double scale=fmaxl(fabsl((long double)diagonal),outflow);
        if(!isfinite(diagonal)||diagonal>0||
           (scale>0&&fabsl((long double)diagonal+outflow)/scale>1.e-12L)){
            status=11;goto done;
        }
    }
    for(size_t reduced=n;reduced-->1;){
        PGIInterval scale=pgi_exact(0);
        for(size_t dest=0;dest<reduced;dest++)scale=pgi_add(scale,work[reduced*n+dest]);
        if(!(scale.lo>0)||!pgi_valid(scale)){status=4;goto done;}
        denominator[reduced]=scale;
        for(size_t source=0;source<reduced;source++){
            PGIInterval factor=pgi_divide(work[source*n+reduced],scale);
            if(!pgi_valid(factor)){status=5;goto done;}
            for(size_t dest=0;dest<reduced;dest++){
                if(source==dest)continue; /* self loops do not enter CTMC balance */
                PGIInterval value=pgi_add(work[source*n+dest],pgi_multiply(factor,work[reduced*n+dest]));
                if(!pgi_valid(value)){status=6;goto done;}
                work[source*n+dest]=value;
            }
        }
    }
    p[0]=pgi_exact(1);
    for(size_t state=1;state<n;state++){
        PGIInterval incoming=pgi_exact(0);
        for(size_t source=0;source<state;source++)
            incoming=pgi_add(incoming,pgi_multiply(p[source],work[source*n+state]));
        if(!(incoming.lo>0)||!pgi_valid(incoming)){status=7;goto done;}
        p[state]=pgi_divide(incoming,denominator[state]);
        if(!pgi_valid(p[state])){status=8;goto done;}
    }
    PGIInterval total=pgi_exact(0);
    for(size_t i=0;i<n;i++)total=pgi_add(total,p[i]);
    if(!(total.lo>0)||!pgi_valid(total)){status=9;goto done;}
    for(size_t i=0;i<n;i++){
        PGIInterval normalized=pgi_divide(p[i],total);
        if(!pgi_valid(normalized)){status=10;goto done;}
        lower[i]=normalized.lo;upper[i]=normalized.hi;
    }
done:
    free(work);free(denominator);free(p);return status;
}

static inline int pgi_group_sums(const long double *lower,const long double *upper,
                                 size_t n,const int *groups,size_t ng,
                                 long double *sum_lower,long double *sum_upper) {
    if(pgi_environment()||!lower||!upper||!groups||!sum_lower||!sum_upper||!ng)return 1;
    for(size_t g=0;g<ng;g++){
        PGIInterval sum=pgi_exact(0);
        for(size_t i=0;i<n;i++){
            if(groups[i]<0||(size_t)groups[i]>=ng)return 2;
            if(!pgi_valid((PGIInterval){lower[i],upper[i]}))return 3;
            if((size_t)groups[i]==g)sum=pgi_add(sum,(PGIInterval){lower[i],upper[i]});
        }
        if(!pgi_valid(sum))return 3;
        sum_lower[g]=sum.lo;sum_upper[g]=sum.hi;
    }
    return 0;
}

static inline int pgi_candidate_error(const long double *lower,const long double *upper,
                                       const double *candidate,size_t n,double total,
                                       long double *probability_error,long double *relative_error) {
    if(pgi_environment()||!lower||!upper||!candidate||!probability_error||!relative_error||
       !isfinite(total)||!(total>0))return 1;
    for(size_t i=0;i<n;i++){
        PGIInterval p={lower[i],upper[i]};
        if(!pgi_valid(p)||!(p.lo>0)||!isfinite(candidate[i])||candidate[i]<0)return 2;
        PGIInterval density=pgi_multiply(p,pgi_exact((long double)total));
        if(!pgi_valid(density)||!(density.lo>0))return 3;
        volatile long double left=(long double)candidate[i]-density.lo;
        volatile long double right=density.hi-(long double)candidate[i];
        long double error=nextafterl(fmaxl(fabsl(left),fabsl(right)),INFINITY);
        PGIInterval absolute=pgi_divide(pgi_exact(error),pgi_exact((long double)total));
        PGIInterval relative=pgi_divide(pgi_exact(error),pgi_exact(density.lo));
        if(!pgi_valid(absolute)||!pgi_valid(relative))return 4;
        probability_error[i]=absolute.hi;relative_error[i]=relative.hi;
    }
    return 0;
}

/* Only the original candidate is checked. No renormalization or root replacement.
 * Maxima and strict decisions remain long double; printed decimals are not proof. */
static inline int pgi_certify_candidate(const double *matrix,size_t n,const double *candidate,
                                       double total,const int *groups,size_t ng,
                                       long double *sum_lower,long double *sum_upper,
                                       long double *maximum_probability_error,
                                       long double *maximum_relative_error) {
    if(!matrix||!candidate||!groups||!sum_lower||!sum_upper||
       !maximum_probability_error||!maximum_relative_error||n<2||n>1000||!ng)return 1;
    *maximum_probability_error=INFINITY;*maximum_relative_error=INFINITY;
    long double *storage=(long double*)calloc(4*n,sizeof(long double));
    if(!storage)return 2;
    long double *lo=storage,*hi=lo+n,*absolute=hi+n,*relative=absolute+n;
    int status=pgi_stationary(matrix,n,lo,hi);
    if(status)goto done;
    status=pgi_group_sums(lo,hi,n,groups,ng,sum_lower,sum_upper);
    if(status){status+=20;goto done;}
    status=pgi_candidate_error(lo,hi,candidate,n,total,absolute,relative);
    if(status){status+=40;goto done;}
    *maximum_probability_error=0;*maximum_relative_error=0;
    for(size_t i=0;i<n;i++) {
        if(absolute[i]>*maximum_probability_error)*maximum_probability_error=absolute[i];
        if(relative[i]>*maximum_relative_error)*maximum_relative_error=relative[i];
    }
    if(*maximum_probability_error>nextafterl(1.e-12L,0)||
       *maximum_relative_error>nextafterl(1.e-10L,0))status=60;
done:
    free(storage);return status;
}

#endif
