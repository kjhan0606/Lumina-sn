#include "cmf_collision_complement.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"COLLISION_COMPLEMENT_FAIL line=%d\n",__LINE__);return 1;}}while(0)
typedef struct {double a[16];int dimension,calls;} Matrix;
static int add(void *v,int to,int from,double rate){
    Matrix *m=v;m->a[from*m->dimension+to]+=rate;
    m->a[from*m->dimension+from]-=rate;m->calls++;return 0;
}
int main(void){
    CMFCollisionLevel levels[4]={{0,0,0,2,1},{2,1,1,4,1},{7,2,3,6,1},{9,3,3,8,1}};
    unsigned char mask[16]={0};mask[1]=1;mask[2*4+3]=1;
    Matrix m={{0},4,0};uint64_t pairs=99;
    const double T=6000,ne=1e10,omega=.1,kT=1.380649e-16*T/1.602176634e-12;
    CHECK(cmf_collision_label(levels,4,7)==2&&cmf_collision_label(levels,4,3)==-1);
    CHECK(!cmf_collision_complement(4,levels,mask,4,T,ne,omega,add,&m,&pairs));
    CHECK(pairs==4&&m.calls==8&&m.a[1]==0&&m.a[4]==0&&m.a[11]==0&&m.a[14]==0);
    double w[4];for(int i=0;i<4;i++)w[i]=levels[i].g*exp(-levels[i].energy_eV/kT);
    for(int i=0;i<4;i++){
        double sum=0;for(int j=0;j<4;j++)sum+=m.a[i*4+j];CHECK(fabs(sum)<1e-12);
        for(int j=i+1;j<4;j++){
            double a=m.a[i*4+j]*w[i],b=m.a[j*4+i]*w[j],scale=fmax(a,b);
            CHECK(scale==0||fabs(a-b)/scale<3e-15);
        }
    }
    /* 같은 SL 안의 전이는 소유자의 Boltzmann 투영에 포함된다. */
    levels[3].solve=2;levels[2].fraction=6./14;levels[3].fraction=8./14;
    memset(&m,0,sizeof m);m.dimension=4;
    CHECK(!cmf_collision_complement(4,levels,mask,4,T,ne,omega,add,&m,&pairs)&&pairs==4);
    double sl_weight[3]={w[0],w[1],w[2]+w[3]};
    for(int i=0;i<3;i++)for(int j=i+1;j<3;j++){
        double a=m.a[i*4+j]*sl_weight[i],b=m.a[j*4+i]*sl_weight[j];
        CHECK(fmax(a,b)==0||fabs(a-b)/fmax(a,b)<3e-15);
    }
    double snapshot[16];memcpy(snapshot,m.a,sizeof snapshot);int calls=m.calls;
    levels[3].fraction=-1;
    CHECK(cmf_collision_complement(4,levels,mask,4,T,ne,omega,add,&m,&pairs)!=0);
    CHECK(!memcmp(snapshot,m.a,sizeof snapshot)&&calls==m.calls);levels[3].fraction=8./14;
    CHECK(cmf_collision_complement(4,levels,mask,4,-T,ne,omega,add,&m,&pairs)!=0);
    CHECK(cmf_collision_complement(4,levels,mask,4,T,-ne,omega,add,&m,&pairs)!=0);
    CHECK(cmf_collision_complement(4,levels,mask,4,T,ne,NAN,add,&m,&pairs)!=0);
    memset(mask,1,sizeof mask);memset(&m,0,sizeof m);m.dimension=4;
    CHECK(!cmf_collision_complement(4,levels,mask,4,T,ne,omega,add,&m,&pairs)&&pairs==0&&m.calls==0);
    /* 에너지 순서와 label 순서는 달라도 된다. 퇴화 준위의 정·역률도 유한하다. */
    CMFCollisionLevel equal[2]={{0,0,3,2,1},{1,1,3,4,1}};unsigned char none[4]={0};
    memset(&m,0,sizeof m);m.dimension=2;
    CHECK(!cmf_collision_complement(2,equal,none,2,T,ne,omega,add,&m,&pairs));
    CHECK(pairs==1&&m.a[1]>0&&m.a[1]==2*m.a[2]);
    equal[0].energy_eV=5;
    memset(&m,0,sizeof m);m.dimension=2;
    CHECK(!cmf_collision_complement(2,equal,none,2,T,ne,omega,add,&m,&pairs));
    CHECK(fabs(m.a[2]/m.a[1]-.5*exp(-2/kT))<1e-15);
    puts("COLLISION_COMPLEMENT_PASS missing_pairs_only=1 detailed_balance=1 SL_projection=1 invalid_preserved=1 degenerate_levels=1");
    return 0;
}
