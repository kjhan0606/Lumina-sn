#include "cmf_nonthermal.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void cmf_nonthermal_atomic_free(CMFNonthermalAtomicData *a){
    if(a){free(a->routes);memset(a,0,sizeof(*a));}
}
static int table_row(FILE *fp,double *values,int n,double scale,int *line){
    char buf[16384],*p,*end;
    if(!fgets(buf,sizeof buf,fp)||!strchr(buf,'\n'))return -1;
    ++*line;p=buf;
    for(int i=0;i<n;i++){
        values[i]=strtod(p,&end)*scale;if(p==end||!(values[i]>0)||!isfinite(values[i]))return -1;p=end;
    }
    while(isspace((unsigned char)*p))p++;
    return *p?-1:0;
}
int cmf_nonthermal_atomic_read(CMFNonthermalAtomicData *out,const char *path){
    if(!out||!path)return -1;
    FILE *fp=fopen(path,"r");if(!fp)return -1;
    CMFNonthermalAtomicData a={0};a.routes=calloc(CMF_NT_MAX_ROUTES,sizeof(*a.routes));
    if(!a.routes){fclose(fp);return -1;}
    char buf[4096];int line=0;
    while(fgets(buf,sizeof buf,fp)){
        line++;if(!strchr(buf,'\n')&&!feof(fp))goto fail;
        char *p=buf;while(isspace((unsigned char)*p))p++;if(!*p||*p=='!')continue;
        if(a.n_routes==CMF_NT_MAX_ROUTES)goto fail;
        CMFNonthermalAtomicRoute *r=&a.routes[a.n_routes];int bound,used=0;
        int parsed=sscanf(p,"%d %d %d %d %lf %lf %lf %lf %lf %n",&r->Z,&bound,&r->n,&r->l,
                          &r->threshold_eV,&r->abcd[0],&r->abcd[1],&r->abcd[2],&r->abcd[3],&used);
        if(parsed!=9){
            if(sscanf(p,"%d %d %d %d %lf %d %n",&r->Z,&bound,&r->n,&r->l,
                      &r->threshold_eV,&r->n_tab,&used)!=6||r->n_tab<2||r->n_tab>CMF_NT_MAX_TAB)goto fail;
        }
        r->source_line=line;r->charge=r->Z-bound;
        if(r->Z<1||r->Z>118||bound<1||bound>r->Z||r->n<1||r->l<0||r->l>=r->n||
           !(r->threshold_eV>0)||!isfinite(r->threshold_eV))goto fail;
        for(int k=0;k<4;k++)if(!isfinite(r->abcd[k]))goto fail;
        char *label=p+used;size_t len=strcspn(label,"\r\n");
        while(len&&isspace((unsigned char)label[len-1]))len--;
        if(len>=sizeof r->terms)goto fail;
        memcpy(r->terms,label,len);r->terms[len]=0;
        if(r->n_tab){
            if(table_row(fp,r->tab_energy,r->n_tab,r->threshold_eV,&line)||
               table_row(fp,r->tab_sigma,r->n_tab,1,&line))goto fail;
            for(int j=0;j<r->n_tab;j++)if(!(r->tab_energy[j]>r->threshold_eV)||
               (j&&!(r->tab_energy[j]>r->tab_energy[j-1])))goto fail;
        }
        a.n_routes++;
    }
    if(ferror(fp)||!a.n_routes)goto fail;
    fclose(fp);cmf_nonthermal_atomic_free(out);*out=a;return 0;
fail:
    fclose(fp);cmf_nonthermal_atomic_free(&a);return -1;
}
double cmf_nonthermal_atomic_sigma(const CMFNonthermalAtomicRoute *r,double E){
    if(!r||!(E>0)||!isfinite(E)||!(r->threshold_eV>0))return NAN;
    if(!r->n_tab)return cmf_nonthermal_arnaud(E,r->threshold_eV,r->abcd);
    if(r->n_tab<2||r->n_tab>CMF_NT_MAX_TAB)return NAN;
    if(E<r->tab_energy[0])return 0; /* 원본 표의 하한 아래는 정의된0 */
    int i=0;while(i+2<r->n_tab&&E>r->tab_energy[i+1])i++;
    double w=log(E/r->tab_energy[i])/log(r->tab_energy[i+1]/r->tab_energy[i]);
    double s=exp(log(r->tab_sigma[i])+w*log(r->tab_sigma[i+1]/r->tab_sigma[i]));
    return s>=0&&isfinite(s)?s:NAN;
}
