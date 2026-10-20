/* Shared production/offline implementation; no duplicated arithmetic. */
#include "population_generator_interval.h"
int generator_interval_precision(void) { return pgi_precision(); }
int generator_interval_environment(void) { return pgi_environment(); }
int generator_interval_stationary(const double *matrix,size_t n,long double *lo,long double *hi) {
    return pgi_stationary(matrix,n,lo,hi);
}
int generator_interval_group_sums(const long double *lo,const long double *hi,size_t n,
                                 const int *groups,size_t ng,long double *gl,long double *gh) {
    return pgi_group_sums(lo,hi,n,groups,ng,gl,gh);
}
int generator_interval_candidate_error(const long double *lo,const long double *hi,
                                       const double *candidate,size_t n,double total,
                                       long double *absolute,long double *relative) {
    return pgi_candidate_error(lo,hi,candidate,n,total,absolute,relative);
}
int generator_interval_certify_candidate(const double *matrix,size_t n,const double *candidate,
                                         double total,const int *groups,size_t ng,
                                         long double *gl,long double *gh,long double *absolute,
                                         long double *relative) {
    return pgi_certify_candidate(matrix,n,candidate,total,groups,ng,gl,gh,absolute,relative);
}
