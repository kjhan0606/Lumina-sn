#ifndef LUMINA_CMF_CELL_GEOMETRY_H
#define LUMINA_CMF_CELL_GEOMETRY_H
#include <math.h>

enum { CMF_CELL_CORE_RAYS=16, CMF_CELL_OUTER_RAYS=16 };

/* CPU/CUDA host가 같은 실제 셸 경계·광선·각도 가중을 사용한다. */
static inline int cmf_cell_grid_valid(int ns,const double *ri,const double *ro) {
    if(ns<=0 || !ri || !ro)return 0;
    for(int s=0;s<ns;++s)
        if(!(ri[s]>=0) || !(ro[s]>ri[s]) || !isfinite(ri[s]) ||
           !isfinite(ro[s]) || (s && ri[s]!=ro[s-1]))return 0;
    return 1;
}

static inline int cmf_cell_ray_count(int ns) {
    return ns+CMF_CELL_CORE_RAYS+CMF_CELL_OUTER_RAYS;
}

static inline void cmf_cell_fill_rays(int ns,const double *ri,const double *ro,double *p) {
    int nr=0;
    for(int k=0;k<CMF_CELL_CORE_RAYS;++k)
        p[nr++]=ri[0]*(double)k/CMF_CELL_CORE_RAYS;
    for(int k=0;k<CMF_CELL_OUTER_RAYS;++k)
        p[nr++]=ri[0]+(ro[ns-1]-ri[0])*(double)k/CMF_CELL_OUTER_RAYS;
    for(int s=0;s<ns;++s)p[nr++]=0.5*(ri[s]+ro[s]);
    for(int i=1;i<nr;++i) {
        double v=p[i];int j=i;
        while(j>0 && p[j-1]>v){p[j]=p[j-1];--j;}
        p[j]=v;
    }
}

static inline double cmf_cell_mu(double p,double r) {
    double q=p/r;
    return sqrt((1-q)*(1+q));
}

/* 한 반구의 dmu 사다리꼴 가중. 코어 limb를 가로질러 보간하지 않는다. */
static inline double cmf_cell_angle_weight(int k,int nr,const double *p,double r,double core) {
    if(p[k]>r)return 0;
    int is_core=p[k]<core;
    double limb=cmf_cell_mu(core,r),mu=cmf_cell_mu(p[k],r);
    double top=is_core?1:limb,bottom=is_core?limb:0;
    if(k>0 && (p[k-1]<core)==is_core)
        top=0.5*(mu+cmf_cell_mu(p[k-1],r));
    if(k+1<nr && p[k+1]<=r && (p[k+1]<core)==is_core)
        bottom=0.5*(mu+cmf_cell_mu(p[k+1],r));
    return top-bottom;
}

/* 동일한 각도 구간의 정확한 integral(mu^m dmu). m=1,2는 H/K용이다.
 * 등방 강도는 격자와 무관하게 K/J=1/3, 바깥 반구는 H/J=1/2를 준다. */
static inline double cmf_cell_angle_moment_weight(int k,int nr,const double *p,
                                                double r,double core,int moment) {
    if(p[k]>r)return 0;
    int is_core=p[k]<core;
    double limb=cmf_cell_mu(core,r),mu=cmf_cell_mu(p[k],r);
    double top=is_core?1:limb,bottom=is_core?limb:0;
    if(k>0 && (p[k-1]<core)==is_core)
        top=0.5*(mu+cmf_cell_mu(p[k-1],r));
    if(k+1<nr && p[k+1]<=r && (p[k+1]<core)==is_core)
        bottom=0.5*(mu+cmf_cell_mu(p[k+1],r));
    double width=top-bottom;
    if(moment==1)return width*(top+bottom)/2;
    if(moment==2)return width*(top*top+top*bottom+bottom*bottom)/3;
    return width;
}

/* 광선은 ro>p인 모든 셸을 통과한다. 중심에 닿지 않는 접선 셸의
 * 전체 광로는 outer 반구간에 두며 inner 반구간은 정확히0이다. */
static inline void cmf_cell_half_lengths(double ri,double ro,double p,double *outer,double *inner) {
    double zh=sqrt((ro-p)*(ro+p));
    double zl=p<ri?sqrt((ri-p)*(ri+p)):0;
    double rm=0.5*(ri+ro);
    double zm=p<=rm?sqrt((rm-p)*(rm+p)):zl;
    *outer=zh-zm;
    *inner=zm-zl;
}
#endif
