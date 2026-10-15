#include "cmf_gamma_source.h"
#include "cmf_decay.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int isotope(const char *name,double mass){
    if(!isfinite(mass)||floor(mass+.5)!=56)return -1;
    return !strcmp(name,"NICK")?0:!strcmp(name,"COB")?1:-1;
}
int cmf_gamma_nuclear_read(CMFGammaNuclearData *out,const char *path){
    if(!out||!path)return -1;
    FILE *fp=fopen(path,"r");if(!fp)return -1;
    CMFGammaNuclearData d={0};int declared[2]={0},read_lines[2]={0},rc=-1;
    char line[2048],name[32],daughter[32],order,extra;
    double mass,half,dmass,Q;
    while(fgets(line,sizeof line,fp)){
        if(!strchr(line,'\n')&&!feof(fp))goto done;
        int count=0;
        if(sscanf(line,"%31s %lf %lf %31s %lf %lf %c %d",name,&mass,&half,daughter,&dmass,&Q,&order,&count)==8){
            int k=isotope(name,mass);if(k<0)continue;
            if(declared[k]||!(half>0)||!isfinite(half)||!(Q>0)||!isfinite(Q)||count<1||
               count>CMF_GAMMA_MAX_LINES||(k==0?(strcmp(daughter,"COB")||order!='F'):
               (strcmp(daughter,"IRON")||order!='S')))goto done;
            d.decay.lambda[k]=log(2.)/(half*86400);d.declared_total_MeV[k]=Q;declared[k]=count;
        }else if(sscanf(line,"%31s %lf %c",name,&mass,&extra)==2){
            int k=isotope(name,mass);if(k<0)continue;
            if(!declared[k]||read_lines[k]||d.n_lines+declared[k]>CMF_GAMMA_MAX_LINES)goto done;
            if(!fgets(line,sizeof line,fp)||!strstr(line,"E (MeV) Probability"))goto done;
            for(int i=0;i<declared[k];i++){
                double E,P;
                if(!fgets(line,sizeof line,fp)||sscanf(line,"%lf %lf %c",&E,&P,&extra)!=2||
                   !(E>0)||!isfinite(E)||!(P>=0)||!isfinite(P))goto done;
                int l=d.n_lines++;d.parent[l]=k;d.energy_MeV[l]=E;d.probability[l]=P;
                d.decay.gamma_energy_erg[k]+=E*P*1.602176634e-6;
            }
            double kinetic;
            if(!fgets(line,sizeof line,fp)||!strstr(line,"tot ekin positron")||
               sscanf(line,"%lf",&kinetic)!=1||!(kinetic>=0)||!isfinite(kinetic))goto done;
            d.decay.positron_kinetic_erg[k]=kinetic*1.602176634e-6;read_lines[k]=1;
        }
    }
    if(ferror(fp)||!read_lines[0]||!read_lines[1]||d.n_lines<2)goto done;
    for(int k=0;k<2;k++)if(!(d.decay.gamma_energy_erg[k]>0)||!isfinite(d.decay.gamma_energy_erg[k]))goto done;
    d.completion_scale[0]=d.completion_scale[1]=1;
    *out=d;rc=0;
done:
    fclose(fp);return rc;
}
int cmf_gamma_nuclear_complete_declared_Q(CMFGammaNuclearData *out,const CMFGammaNuclearData *in){
    if(!out||!in||in->n_lines<2||in->n_lines>CMF_GAMMA_MAX_LINES)return -1;
    CMFGammaNuclearData d=*in;
    for(int k=0;k<2;k++){
        if(in->completion_scale[k]!=1)return -1;
        double wanted=in->declared_total_MeV[k]*1.602176634e-6-in->decay.positron_kinetic_erg[k];
        if(!(wanted>0)||!isfinite(wanted)||!(in->decay.gamma_energy_erg[k]>0))return -1;
        d.completion_scale[k]=wanted/in->decay.gamma_energy_erg[k];
        d.decay.gamma_energy_erg[k]=0;
    }
    for(int l=0;l<d.n_lines;l++){
        int k=d.parent[l];if(k<0||k>1||!(d.energy_MeV[l]>0)||!(d.probability[l]>=0))return -1;
        d.probability[l]*=d.completion_scale[k];
        d.decay.gamma_energy_erg[k]+=d.energy_MeV[l]*d.probability[l]*1.602176634e-6;
        if(!isfinite(d.decay.gamma_energy_erg[k]))return -1;
    }
    for(int k=0;k<2;k++){
        double wanted=in->declared_total_MeV[k]*1.602176634e-6-in->decay.positron_kinetic_erg[k];
        if(!(d.decay.gamma_energy_erg[k]>0)||fabs(d.decay.gamma_energy_erg[k]/wanted-1)>1e-12)return -1;
    }
    *out=d;return 0;
}
void cmf_toy_gamma_source_free(CMFToyGammaSource *s){
    if(s){free(s->storage);cmf_ejecta_epoch_free(&s->epoch);memset(s,0,sizeof(*s));}
}
int cmf_toy_gamma_source_build(CMFToyGammaSource *out,const CMFToyEjecta *m,
    const CMFGammaNuclearData *d,double epoch){
    return cmf_toy_gamma_source_build_at(out,m,d,epoch,epoch,CMF_DECAY_ENDPOINT);
}
int cmf_toy_gamma_source_build_at(CMFToyGammaSource *out,const CMFToyEjecta *m,
    const CMFGammaNuclearData *d,double previous,double epoch,CMFDecaySourceMode mode){
    if(!out||!m||!d||m->n_shells<1||d->n_lines<1||d->n_lines>CMF_GAMMA_MAX_LINES)return -1;
    size_t ns=m->n_shells,nl=d->n_lines;
    if(ns>(SIZE_MAX/sizeof(double)-1)/(3+nl))return -1;
    CMFToyGammaSource s={0};
    if(cmf_ejecta_epoch_build(&s.epoch,m,&d->decay,previous,epoch,mode))return -1;
    s.storage=calloc((3+nl)*ns+1,sizeof(double));if(!s.storage)goto fail;
    double *edge=s.storage,*ne=edge+ns+1,*z5=ne+ns,*emiss=z5+ns;
    static const int Z[6]={28,27,26,20,16,14};
    for(size_t i=0;i<=ns;i++)edge[i]=m->velocity_edges[i]*epoch;
    for(size_t i=0;i<ns;i++){
        double rate[2];
        if(mode==CMF_DECAY_ENDPOINT){
            for(int k=0;k<2;k++)rate[k]=d->decay.lambda[k]*s.epoch.number_density[k*ns+i];
        }else{
            double initial[3],prior[3],now[3],counts[2],unused[2];
            for(int k=0;k<3;k++)initial[k]=m->number_per_g[k*ns+i];
            if(cmf_decay_chain_step(d->decay.lambda[0],d->decay.lambda[1],previous-m->epoch_s,initial,prior,unused)||
               cmf_decay_chain_step(d->decay.lambda[0],d->decay.lambda[1],epoch-previous,prior,now,counts))goto fail;
            for(int k=0;k<2;k++)rate[k]=counts[k]/(epoch-previous)*s.epoch.density[i];
        }
        for(int k=0;k<6;k++){
            double number=s.epoch.number_density[k*ns+i];
            ne[i]+=Z[k]*number;z5[i]+=pow((double)Z[k],5)*number;
        }
        for(size_t l=0;l<nl;l++){
            int k=d->parent[l];if(k<0||k>1||!(d->probability[l]>=0)||!isfinite(d->probability[l]))goto fail;
            emiss[l*ns+i]=rate[k]*d->probability[l];
        }
    }
    s.input=(CMFGammaInput){(int)ns,(int)nl,epoch,edge,ne,z5,d->energy_MeV,emiss};
    cmf_toy_gamma_source_free(out);*out=s;return 0;
fail:
    cmf_toy_gamma_source_free(&s);return -1;
}
