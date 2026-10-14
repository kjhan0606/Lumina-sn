#include "cmf_ejecta.h"
#include "cmf_decay.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int toy_Z[CMF_TOY_ELEMENTS]={28,27,26,20,16,14};
static const int toy_column[CMF_TOY_ELEMENTS]={12,14,15,16,17,18};
static double volume(double lo,double hi,double t){
    double r=lo*t,R=hi*t;
    return (4.0*M_PI_VAL/3.0)*(R-r)*(R*R+R*r+r*r);
}
static double header_value(const char *line,const char *key){
    const char *p=strstr(line,key);if(!p||(p=strchr(p,'='))==NULL)return NAN;
    char *end;double x=strtod(p+1,&end);return end==p+1?NAN:x;
}
void cmf_toy_ejecta_free(CMFToyEjecta *m){if(m){free(m->storage);memset(m,0,sizeof(*m));}}
void cmf_ejecta_epoch_free(CMFEjectaEpoch *e){if(e){free(e->storage);memset(e,0,sizeof(*e));}}

int cmf_toy_ejecta_read(CMFToyEjecta *out,const char *path,int requested,const double masses[6]){
    if(!out||!path||!masses||requested<0)return -1;
    for(int i=0;i<6;i++)if(!(masses[i]>0)||!isfinite(masses[i]))return -1;
    FILE *fp=fopen(path,"r");if(!fp)return -1;
    CMFToyEjecta m={0};double (*rows)[21]=NULL;
    double msun=NAN,dv=NAN,declared=NAN,tday=NAN;char line[4096];int n=0,rc=-1;
    m.model_halflife_s[0]=m.model_halflife_s[1]=NAN;
    m.model_decay_energy_MeV[0]=m.model_decay_energy_MeV[1]=NAN;
    while(fgets(line,sizeof line,fp)){
        if(!strchr(line,'\n')&&!feof(fp))goto done;
        char *p=line;while(*p==' '||*p=='\t')p++;
        if(*p=='#'){
            if(strstr(p,"MSUN ="))msun=header_value(p,"MSUN");
            if(strstr(p,"nzones ="))declared=header_value(p,"nzones");
            if(strstr(p,"dvel ="))dv=header_value(p,"dvel")*1e5;
            if(strstr(p,"tend ="))tday=header_value(p,"tend");
            if(strstr(p,"THALF_56NI ="))m.model_halflife_s[0]=header_value(p,"THALF_56NI")*86400;
            if(strstr(p,"THALF_56CO ="))m.model_halflife_s[1]=header_value(p,"THALF_56CO")*86400;
            if(strstr(p,"EDECAY_56NI ="))m.model_decay_energy_MeV[0]=header_value(p,"EDECAY_56NI");
            if(strstr(p,"EDECAY_56CO ="))m.model_decay_energy_MeV[1]=header_value(p,"EDECAY_56CO");
            continue;
        }
        if(*p=='\n'||!*p)continue;
        if(!rows){
            if(!(declared>=1)||declared>8192||declared!=floor(declared))goto done;
            rows=malloc((size_t)declared*sizeof(*rows));if(!rows)goto done;
        }
        if(n>=(int)declared)goto done;
        for(int k=0;k<21;k++){
            char *end;errno=0;rows[n][k]=strtod(p,&end);
            if(end==p||errno||!isfinite(rows[n][k]))goto done;
            p=end;
        }
        while(*p==' '||*p=='\t'||*p=='\r'||*p=='\n')p++;
        if(*p)goto done;n++;
    }
    if(ferror(fp)||n!=declared||n<=0||!(msun>0)||!isfinite(msun)||
       !(dv>0)||!isfinite(dv)||!(tday>0)||!isfinite(tday)||requested>n)goto done;
    for(int k=0;k<2;k++)if(!(m.model_halflife_s[k]>0)||!isfinite(m.model_halflife_s[k])||
        !(m.model_decay_energy_MeV[k]>=0)||!isfinite(m.model_decay_energy_MeV[k]))goto done;
    m.n_source_shells=n;m.n_shells=requested?requested:n;m.epoch_s=tday*86400;
    memcpy(m.mass_amu,masses,sizeof m.mass_amu);
    if(!isfinite(m.epoch_s))goto done;
    for(int i=0;i<n;i++){
        const double *r=rows[i];double v=(i+.5)*dv,V=volume(i*dv,(i+1)*dv,m.epoch_s);
        if(r[0]!=i+1||fabs(r[1]*1e5/v-1)>5e-5||!(r[2]>0)||!(r[10]>0)||!(r[11]>0)||
           fabs(r[9]/(v*m.epoch_s)-1)>5e-5||!(V>0)||!isfinite(V))goto done;
        /* 순수 Ni56 붕괴 사슬과 Ca/S/Si로 구성된 toy 입력 전용이다. */
        if(r[4]!=0||r[7]!=0||r[8]!=0||r[19]!=0||r[20]!=0||r[12]!=r[13])goto done;
        double sum=0;
        for(int k=0;k<6;k++){
            double X=r[toy_column[k]];if(X<0||X>1)goto done;
            sum+=X;
        }
        m.max_input_fraction_error=fmax(m.max_input_fraction_error,fabs(sum-1));
        m.max_input_density_error=fmax(m.max_input_density_error,fabs(r[2]*msun/(V*r[10])-1));
        if(m.max_input_fraction_error>5e-5||m.max_input_density_error>2e-4||r[5]<0||r[5]>1)goto done;
        m.initial_ni56_mass_g+=r[2]*msun*r[5];
    }
    {
        size_t ns=m.n_shells;
        m.storage=calloc(9*ns+1,sizeof(double));if(!m.storage)goto done;
        m.velocity_edges=m.storage;m.mass_g=m.storage+ns+1;
        m.initial_temperature=m.mass_g+ns;m.number_per_g=m.initial_temperature+ns;
        for(int s=0;s<m.n_shells;s++){
            int lo=(int)((long long)s*n/m.n_shells),hi=(int)((long long)(s+1)*n/m.n_shells);
            long double M=0,V=0,rad=0,number[6]={0};
            m.velocity_edges[s]=lo*dv;m.velocity_edges[s+1]=hi*dv;
            for(int i=lo;i<hi;i++){
                long double dm=rows[i][2]*msun,vol=volume(i*dv,(i+1)*dv,m.epoch_s);
                M+=dm;V+=vol;rad+=vol*powl(rows[i][11],4);
                for(int k=0;k<6;k++)number[k]+=dm*rows[i][toy_column[k]]/(masses[k]*AMU);
            }
            m.mass_g[s]=(double)M;m.initial_temperature[s]=(double)powl(rad/V,.25L);
            if(!(M>0)||!isfinite(m.mass_g[s])||!(m.initial_temperature[s]>0)||!isfinite(m.initial_temperature[s]))goto done;
            for(int k=0;k<6;k++){
                m.number_per_g[k*ns+s]=(double)(number[k]/M);
                if(!(m.number_per_g[k*ns+s]>=0)||!isfinite(m.number_per_g[k*ns+s]))goto done;
            }
        }
    }
    cmf_toy_ejecta_free(out);*out=m;memset(&m,0,sizeof m);rc=0;
done:
    fclose(fp);free(rows);cmf_toy_ejecta_free(&m);return rc;
}

int cmf_ejecta_epoch_build(CMFEjectaEpoch *out,const CMFToyEjecta *m,
    const CMFDecaySourceData *d,double previous,double epoch,CMFDecaySourceMode mode){
    if(!out||!m||!d||m->n_shells<=0||!m->velocity_edges||!m->mass_g||!m->number_per_g||
       !(m->epoch_s>0)||!isfinite(m->epoch_s)||!isfinite(epoch)||!isfinite(previous)||previous<m->epoch_s||epoch<previous||
       (mode!=CMF_DECAY_ENDPOINT&&mode!=CMF_DECAY_INTERVAL_AVERAGE)||
       (mode==CMF_DECAY_INTERVAL_AVERAGE&&epoch==previous))return -1;
    for(int k=0;k<2;k++)if(!(d->lambda[k]>0)||!isfinite(d->lambda[k])||
       !(d->gamma_energy_erg[k]>=0)||!isfinite(d->gamma_energy_erg[k])||
       !(d->positron_kinetic_erg[k]>=0)||!isfinite(d->positron_kinetic_erg[k]))return -1;
    size_t ns=m->n_shells;
    if(ns>SIZE_MAX/(9*sizeof(double)))return -1;
    CMFEjectaEpoch e={0};e.n_shells=m->n_shells;e.epoch_s=epoch;e.previous_epoch_s=previous;e.source_mode=mode;
    e.storage=malloc(9*ns*sizeof(double));if(!e.storage)return -1;
    e.density=e.storage;e.number_density=e.density+ns;e.gamma_emission=e.number_density+6*ns;
    e.positron_emission=e.gamma_emission+ns;
    for(size_t s=0;s<ns;s++){
        double old[3],prior[3],now[3],counts[2],unused[2];
        if(!(m->velocity_edges[s]>=0)||!(m->velocity_edges[s+1]>m->velocity_edges[s])||
           !isfinite(m->velocity_edges[s+1])||!(m->mass_g[s]>0)||!isfinite(m->mass_g[s]))goto fail;
        for(int k=0;k<3;k++)old[k]=m->number_per_g[k*ns+s];
        if(cmf_decay_chain_step(d->lambda[0],d->lambda[1],previous-m->epoch_s,old,prior,unused)||
           cmf_decay_chain_step(d->lambda[0],d->lambda[1],epoch-previous,prior,now,counts))goto fail;
        double vol=volume(m->velocity_edges[s],m->velocity_edges[s+1],epoch);
        e.density[s]=m->mass_g[s]/vol;
        if(!(vol>0)||!isfinite(vol)||!(e.density[s]>0)||!isfinite(e.density[s]))goto fail;
        for(int k=0;k<6;k++){
            e.number_density[k*ns+s]=(k<3?now[k]:m->number_per_g[k*ns+s])*e.density[s];
            if(!(e.number_density[k*ns+s]>=0)||!isfinite(e.number_density[k*ns+s]))goto fail;
        }
        e.gamma_emission[s]=e.positron_emission[s]=0;
        for(int k=0;k<2;k++){
            double rate=(mode==CMF_DECAY_ENDPOINT?d->lambda[k]*now[k]:counts[k]/(epoch-previous))*e.density[s];
            e.gamma_emission[s]+=rate*d->gamma_energy_erg[k];
            e.positron_emission[s]+=rate*d->positron_kinetic_erg[k];
        }
        if(!isfinite(e.gamma_emission[s])||!isfinite(e.positron_emission[s]))goto fail;
    }
    cmf_ejecta_epoch_free(out);*out=e;return 0;
fail:
    cmf_ejecta_epoch_free(&e);return -1;
}

int cmf_ejecta_apply_epoch(const CMFToyEjecta *m,const CMFEjectaEpoch *e,
    Geometry *g,PlasmaState *p,AtomicData *a){
    if(!m||!e||!g||!p||!a||m->n_shells<=0||m->n_shells!=e->n_shells||g->n_shells!=m->n_shells||
       p->n_shells!=m->n_shells||!g->r_inner||!g->r_outer||!g->v_inner||!g->v_outer||!p->rho||
       !m->velocity_edges||!e->number_density||!e->density||!a->element_Z||
       !a->element_mass_amu||!a->abundances||a->n_elements<=0||!(e->epoch_s>=m->epoch_s)||!isfinite(e->epoch_s))return -1;
    int index[6];size_t ns=m->n_shells;
    for(int k=0;k<6;k++){
        index[k]=-1;
        if(!(m->mass_amu[k]>0)||!isfinite(m->mass_amu[k]))return -1;
        for(int j=0;j<a->n_elements;j++)if(a->element_Z[j]==toy_Z[k]){
            if(index[k]>=0)return -1;index[k]=j;
        }
        if(index[k]<0)return -1;
    }
    for(size_t s=0;s<ns;s++){
        if(!(e->density[s]>0)||!isfinite(e->density[s])||!(m->velocity_edges[s]>=0)||
           !(m->velocity_edges[s+1]>m->velocity_edges[s])||!isfinite(m->velocity_edges[s+1]*e->epoch_s))return -1;
        for(int k=0;k<6;k++){
            double X=e->number_density[k*ns+s]*(m->mass_amu[k]*AMU)/e->density[s];
            if(!(X>=0)||!isfinite(X))return -1;
        }
    }
    g->time_explosion=e->epoch_s;
    memset(a->abundances,0,(size_t)a->n_elements*ns*sizeof(double));
    for(int k=0;k<6;k++){
        a->element_mass_amu[index[k]]=m->mass_amu[k];
        for(size_t s=0;s<ns;s++)a->abundances[(size_t)index[k]*ns+s]=
            e->number_density[k*ns+s]*(m->mass_amu[k]*AMU)/e->density[s];
    }
    for(size_t s=0;s<ns;s++){
        g->v_inner[s]=m->velocity_edges[s];g->v_outer[s]=m->velocity_edges[s+1];
        g->r_inner[s]=g->v_inner[s]*e->epoch_s;g->r_outer[s]=g->v_outer[s]*e->epoch_s;p->rho[s]=e->density[s];
    }
    return 0;
}
