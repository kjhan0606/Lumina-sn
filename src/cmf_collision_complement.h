#ifndef LUMINA_CMF_COLLISION_COMPLEMENT_H
#define LUMINA_CMF_COLLISION_COMPLEMENT_H
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* 선/실표가 이미 소유한 쌍은 mask로 제외한다. 나머지는 CMFGEN의
 * f=0 처방이며 인구 floor, 추가 복사선 또는 임의 metastable 보강이 아니다. */
typedef struct {
    int label, solve;
    double energy_eV, g, fraction;
} CMFCollisionLevel;
typedef int (*CMFCollisionEmit)(void *,int,int,double);

static inline int cmf_collision_label(const CMFCollisionLevel *levels,int n,int label){
    int lo=0,hi=n;
    while(lo<hi){int mid=lo+(hi-lo)/2;if(levels[mid].label<label)lo=mid+1;else hi=mid;}
    return lo<n&&levels[lo].label==label?lo:-1;
}

static inline int cmf_collision_complement(int n,const CMFCollisionLevel *levels,
    const unsigned char *owned,int dimension,double T,double ne,double omega,
    CMFCollisionEmit emit,void *context,uint64_t *pair_count){
    if(n<0||dimension<1||!emit||!(T>0)||!isfinite(T)||ne<0||!isfinite(ne)||
       omega<0||!isfinite(omega)||(n&&(!levels||!owned))||
       (n&&(size_t)n>SIZE_MAX/(size_t)n))return -1;
    /* 잘못된 입력에서 부분 행렬을 만들지 않도록 전 준위를 먼저 확인한다. */
    for(int i=0;i<n;i++){
        const CMFCollisionLevel *l=levels+i;
        if(l->label<0||(i&&l->label<=levels[i-1].label)||l->solve<0||l->solve>=dimension||
           l->energy_eV<0||!isfinite(l->energy_eV)||!(l->g>0)||!isfinite(l->g)||
           l->fraction<0||l->fraction>1||!isfinite(l->fraction))return -1;
    }
    double scale=ne*8.629e-6/sqrt(T)*omega;
    if(!isfinite(scale))return -1;
    uint64_t count=0;
    for(int j=1;j<n;j++)for(int i=0;i<j;i++){
        const CMFCollisionLevel *lo=levels+i,*hi=levels+j;
        if(owned[(size_t)i*n+j]||lo->solve==hi->solve)continue;
        if(lo->energy_eV>hi->energy_eV){const CMFCollisionLevel *swap=lo;lo=hi;hi=swap;}
        double de=hi->energy_eV-lo->energy_eV;
        double down=scale/hi->g*hi->fraction;
        double up=scale/lo->g*exp(-de*1.602176634e-12/(1.380649e-16*T))*lo->fraction;
        if(!isfinite(up)||!isfinite(down)||up<0||down<0)return -1;
        if(emit(context,hi->solve,lo->solve,up)||emit(context,lo->solve,hi->solve,down))return -1;
        count++;
    }
    if(pair_count)*pair_count=count;
    return 0;
}
#endif
