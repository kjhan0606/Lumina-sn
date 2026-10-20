#include "lumina_cmfgen.h"

static int small(void) {
    double es[2]={4,4},bf[2]={-1,2},eta_bf[2]={7,9};
    double ff[2]={1,1},eta_ff[2]={2,2},bb[2]={1,-1},eta_bb[2]={3,3};
    double ce[2],ca[2],cl[2],ct[2],sf[2],j[2]={123,456};
    CMFGENState cs={0};cs.n_shells=1;cs.n_bins=2;
    cs.chi_es=ce;cs.chi_abs=ca;cs.chi_line=cl;cs.chi_tot=ct;cs.S_fixed=sf;cs.J=j;
    if (cmfgen_assemble_physical_coefficients(&cs,es,bf,eta_bf,ff,eta_ff,bb,eta_bb)) return 1;
    if (ct[0]!=5 || ct[1]!=6 || sf[0]!=12./5 || sf[1]!=14./6 || j[0]!=123 || j[1]!=456) return 1;
    double before[2];memcpy(before,sf,sizeof(before));bf[1]=-20;
    if (!cmfgen_assemble_physical_coefficients(&cs,es,bf,eta_bf,ff,eta_ff,bb,eta_bb) ||
        memcmp(before,sf,sizeof(before))) return 1;
    double raw_chi[2],raw_eta[2];
    if(cmfgen_sum_raw_coefficients(2,es,bf,eta_bf,ff,eta_ff,bb,eta_bb,raw_chi,raw_eta)||
       raw_chi[0]!=5||raw_chi[1]!=-16||raw_eta[0]!=12||raw_eta[1]!=14)return 1;
    memcpy(before,raw_chi,sizeof(before));eta_bf[1]=-1;
    if(!cmfgen_sum_raw_coefficients(2,es,bf,eta_bf,ff,eta_ff,bb,eta_bb,raw_chi,raw_eta)||
       memcmp(before,raw_chi,sizeof(before))||raw_eta[1]!=14)return 1;
    puts("PHYSICAL_COEFFICIENTS synthetic=PASS signed_opacity=KEPT eta=RAW J=UNCHANGED");
    return 0;
}

int main(int argc,char **argv) {
    if (small()) return 1;
    if (argc==1) return 0;
    if (argc!=2) return 2;
    FILE *fp=fopen(argv[1],"rb");char magic[8];uint32_t ns,nf;
    if (!fp || fread(magic,1,8,fp)!=8 ||
        (memcmp(magic,"LMBFCO01",8) && memcmp(magic,"LMPHYS01",8)) ||
        fread(&ns,4,1,fp)!=1 || fread(&nf,4,1,fp)!=1 || ns<1 || nf<1 || ns>1000 || nf>10000000) return 1;
    int complete=!memcmp(magic,"LMPHYS01",8);
    size_t n=(size_t)ns*nf;
    if (n>SIZE_MAX/sizeof(double)/14) return 1;
    double *nu=malloc((size_t)nf*sizeof(double)),*te=malloc(ns*sizeof(double)),*ne=malloc(ns*sizeof(double));
    double *bf=malloc(n*sizeof(double)),*eta=malloc(n*sizeof(double)),*es=malloc(n*sizeof(double));
    double *ff=calloc(n,sizeof(double)),*ff_eta=calloc(n,sizeof(double));
    double *bb=calloc(n,sizeof(double)),*bb_eta=calloc(n,sizeof(double));
    double *storage=calloc(5*n,sizeof(double));
    if (!nu || !te || !ne || !bf || !eta || !es || !ff || !ff_eta || !bb || !bb_eta || !storage ||
        fread(nu,8,nf,fp)!=nf || fread(te,8,ns,fp)!=ns || fread(ne,8,ns,fp)!=ns ||
        fread(bf,8,n,fp)!=n || fread(eta,8,n,fp)!=n) return 1;
    if (complete && (fread(ff,8,n,fp)!=n || fread(ff_eta,8,n,fp)!=n ||
        fread(bb,8,n,fp)!=n || fread(bb_eta,8,n,fp)!=n || fread(es,8,n,fp)!=n)) return 1;
    if (fgetc(fp)!=EOF) return 1;
    fclose(fp);
    if(!complete)for(size_t s=0;s<ns;s++)for(size_t b=0;b<nf;b++)es[s*nf+b]=ne[s]*6.6524587e-25;
    CMFGENState cs={0};cs.n_shells=(int)ns;cs.n_bins=(int)nf;
    cs.chi_es=storage;cs.chi_abs=storage+n;cs.chi_line=storage+2*n;
    cs.chi_tot=storage+3*n;cs.S_fixed=storage+4*n;
    /* 요청 표본의 계수 소비 시험이며 수송 실행이 아니다. */
    if(complete){
        if(cmfgen_sum_raw_coefficients(n,es,bf,eta,ff,ff_eta,bb,bb_eta,cs.chi_tot,cs.S_fixed))return 1;
    }else if(cmfgen_assemble_physical_coefficients(&cs,es,bf,eta,ff,ff_eta,bb,bb_eta))return 1;
    size_t negative=0,nonthermal=0,negative_total=0;
    for (size_t i=0;i<n;i++) {
        double true_eta=(eta[i]+ff_eta[i])+bb_eta[i];
        if(complete){
            if(cs.chi_tot[i]!=(es[i]+(bf[i]+ff[i]))+bb[i]||cs.S_fixed[i]!=true_eta)return 1;
        }else if(cs.chi_abs[i]!=bf[i]+ff[i]||cs.chi_line[i]!=bb[i]||
                 cs.S_fixed[i]!=true_eta/cs.chi_tot[i])return 1;
        if(cs.chi_tot[i]<0)negative_total++;
        if (bf[i]<0) negative++;
        double nu0=nu[i%nf],temp=te[i/nf];
        double B=2*6.62607015e-27*nu0*nu0*nu0/(2.99792458e10*2.99792458e10)/
            expm1(6.62607015e-27*nu0/(1.380649e-16*temp));
        if (bf[i]>0 && B>0 && (eta[i]/bf[i]/B>2 || eta[i]/bf[i]/B<.5)) nonthermal++;
    }
    printf("PHYSICAL_COEFFICIENTS actual_samples=%zu raw_BF_eta=EXACT negative_BF_preserved=%zu non_Planck_source=%zu negative_TOTAL_preserved=%zu components=%s RT_executions=0\n",
        n,negative,nonthermal,negative_total,complete?"BF_FF_BB_ES_RAW_ETA":"BF_ES_ONLY");
    free(nu);free(te);free(ne);free(bf);free(eta);free(es);free(ff);free(ff_eta);free(bb);free(bb_eta);free(storage);
    return 0;
}
