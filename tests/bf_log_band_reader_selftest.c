/* 최소 직접시험: 번호가 아니라 주파수 겹침을 소비하는지 확인한다. */
#include "bf_rate_jnu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    const double edges[]={1,2,4,8,16};
    double j[]={1,1,1,1};
    RadiationFieldValidityState valid[]={RADIATION_FIELD_VALID,RADIATION_FIELD_VALID,
                                        RADIATION_FIELD_VALID,RADIATION_FIELD_VALID};
    RadiationFieldView v;
    memset(&v,0,sizeof(v));
    v.n_shells=1;v.n_bins=4;v.frequency_bin_edges=edges;v.J_nu=j;v.validity=valid;
    double value;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==0 && fabs(value-1)<1e-14);
    j[0]=10000;j[1]=3;j[2]=6;j[3]=10000;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==0 && fabs(value-4.5)<1e-14);
    j[0]=1e100;j[3]=1e100;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==0 && fabs(value-4.5)<1e-14);
    double expected=(3*log(4./3)+6*log(6./4))/log(2.);
    assert(bf_rate_view_log_band_jnu(&v,0,3,6,&value)==0 && fabs(value-expected)<1e-14);
    j[1]=6;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==0 && fabs(value-6)<1e-14);
    valid[0]=RADIATION_FIELD_UNSAMPLED;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==0 && fabs(value-6)<1e-14);
    valid[1]=RADIATION_FIELD_UNSAMPLED;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==1);
    valid[1]=RADIATION_FIELD_STALE;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==1);
    valid[1]=RADIATION_FIELD_OUT_OF_GRID;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==1);
    valid[1]=RADIATION_FIELD_EXACT_ZERO;valid[2]=RADIATION_FIELD_EXACT_ZERO;j[1]=j[2]=0;
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,&value)==0 && value==0);
    assert(bf_rate_view_log_band_jnu(&v,0,.5,8,&value)==1);
    assert(bf_rate_view_log_band_jnu(&v,0,2,32,&value)==1);
    assert(bf_rate_view_log_band_jnu(&v,0,8,8,&value)==-1);
    assert(bf_rate_view_log_band_jnu(&v,1,2,8,&value)==-1);
    assert(bf_rate_view_log_band_jnu(&v,0,NAN,8,&value)==-1);
    assert(bf_rate_view_log_band_jnu(&v,0,2,8,NULL)==-1);
    puts("FREQUENCY_LOG_BAND_DIRECT_TESTS_PASS baseline_wrong_index_not_used");
    return 0;
}
