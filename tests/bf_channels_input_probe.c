/* 실제 봉인 준위/채널 입력을 production loader와 Milne 적분으로 확인한다.
 * Planck 장 직접시험이며 자체 수송/스펙트럼 결과가 아니다. */
#include "lumina.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(c) do{if(!(c)){fprintf(stderr,"input probe failed line=%d\n",__LINE__);return 1;}}while(0)
int main(int argc,char **argv){
    REQUIRE(argc==4);AtomicData a={0};char *line=NULL;size_t cap=0;
    FILE *f=fopen(argv[1],"r");REQUIRE(f);REQUIRE(getline(&line,&cap,f)>0);
    while(getline(&line,&cap,f)>0)a.n_levels++;
    rewind(f);REQUIRE(getline(&line,&cap,f)>0);
    a.level_Z=calloc(a.n_levels,sizeof(int));a.level_ion=calloc(a.n_levels,sizeof(int));
    a.level_g=calloc(a.n_levels,sizeof(int));a.level_energy_eV=calloc(a.n_levels,sizeof(double));
    REQUIRE(a.level_Z&&a.level_ion&&a.level_g&&a.level_energy_eV);
    int witness=-1;
    for(int i=0;i<a.n_levels;i++){
        REQUIRE(getline(&line,&cap,f)>0);int num;
        REQUIRE(sscanf(line,"%d,%d,%d,%lf,%d",a.level_Z+i,a.level_ion+i,&num,a.level_energy_eV+i,a.level_g+i)==5);
        if(a.level_Z[i]==14&&a.level_ion[i]==1&&num==2)witness=i;
    }
    fclose(f);f=fopen(argv[2],"r");REQUIRE(f);REQUIRE(getline(&line,&cap,f)>0);
    while(getline(&line,&cap,f)>0)a.n_ionization++;
    rewind(f);REQUIRE(getline(&line,&cap,f)>0);
    a.ioniz_Z=calloc(a.n_ionization,sizeof(int));a.ioniz_ion=calloc(a.n_ionization,sizeof(int));
    a.ioniz_energy_eV=calloc(a.n_ionization,sizeof(double));
    REQUIRE(a.ioniz_Z&&a.ioniz_ion&&a.ioniz_energy_eV);
    for(int i=0;i<a.n_ionization;i++){
        REQUIRE(getline(&line,&cap,f)>0);
        REQUIRE(sscanf(line,"%d,%d,%lf",a.ioniz_Z+i,a.ioniz_ion+i,a.ioniz_energy_eV+i)==3);
    }
    fclose(f);free(line);REQUIRE(!load_bf_channels(&a,argv[3]));
    BfChannels *c=a.bf_channels;int nonzero=0;double worst=0,witness_up[2]={0},witness_down[2]={0};
    const double T=1e4,ne=1e8,step=log(c->nu_max/c->nu_min)/c->n_bins;
    for(int l=0;l<a.n_levels;l++)for(int r=c->offset[l];r<c->offset[l+1];r++){
        int u=c->upper[r];double up=0,down=0;
        for(int b=0;b<c->n_bins;b++){
            BfChannelBin v;
            REQUIRE(!bf_channel_bin(c->nu_min*exp(b*step),c->nu_min*exp((b+1)*step),c->threshold[r],
                c->sigma[(size_t)r*c->n_bins+b],T,ne,a.level_g[l],a.level_g[u],1,1,0,1,&v));
            up+=v.ionization;down+=v.recombination;
        }
        if(up>0){
            nonzero++;double ratio=ne*pow(H_PLANCK*H_PLANCK/(2*M_PI_VAL*M_ELECTRON*K_BOLTZMANN*T),1.5)*
                a.level_g[l]/(2.0*a.level_g[u])*exp(H_PLANCK*c->threshold[r]/(K_BOLTZMANN*T));
            double err=fabs(up*ratio/down-1);if(err>worst)worst=err;REQUIRE(err<2e-12);
        }else REQUIRE(down==0);
        if(l==witness){int k=r-c->offset[l];REQUIRE(k<2);witness_up[k]=up;witness_down[k]=down;}
    }
    REQUIRE(witness>=0&&c->offset[witness+1]-c->offset[witness]==2);
    REQUIRE(witness_up[0]==0&&witness_down[0]==0&&witness_up[1]>0&&witness_down[1]>0);
    printf("[BF-INPUT] PASS channels=%d nonzero=%d max_Planck_balance_error=%.17g\n",c->n_channels,nonzero,worst);
    printf("[BF-INPUT] SiII_level2 T=10000 ne=1e8 A_Gamma=%.17g B_Gamma=%.17g A_Rrec=%.17g B_Rrec=%.17g spectrum=NOT_TESTED\n",
        witness_up[0],witness_up[1],witness_down[0],witness_down[1]);
    free_bf_channels(c);free(a.level_Z);free(a.level_ion);free(a.level_g);free(a.level_energy_eV);
    free(a.ioniz_Z);free(a.ioniz_ion);free(a.ioniz_energy_eV);return 0;
}
