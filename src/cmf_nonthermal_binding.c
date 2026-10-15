#include "cmf_nonthermal_binding.h"
#include <limits.h>
typedef struct {
    CMFNonthermalTransition target;
    int raw;
    double threshold,auger,weight;
} Binding;
void cmf_nonthermal_binding_free(CMFNonthermalResolvedModel *m){
    if(m){free(m->transitions);free(m->channels);free(m->ion_sigma);memset(m,0,sizeof(*m));}
}
int cmf_nonthermal_binding_build(const AtomicData *a,const CMFNonthermalAtomicData *raw,
    const CMFNonthermalExcitations *exc,const char *csv,CMFNonthermalResolvedModel *out){
    return cmf_nonthermal_binding_build_policy(a,raw,exc,csv,CMF_NT_SIGMA_SAME_EXCESS,out);
}
int cmf_nonthermal_binding_build_policy(const AtomicData *a,const CMFNonthermalAtomicData *raw,
    const CMFNonthermalExcitations *exc,const char *csv,int policy,CMFNonthermalResolvedModel *out){
    if(!a||!raw||!exc||!csv||!out||raw->n_routes<1||raw->n_routes>CMF_NT_MAX_ROUTES||!raw->routes||
       exc->view.n_levels!=a->n_levels||exc->view.n_ions!=a->n_ion_pops||
       exc->view.n_energy<2||exc->view.n_channels<0||!exc->energy||!exc->channels||!exc->transitions||
       (policy!=CMF_NT_SIGMA_SAME_EXCESS&&policy!=CMF_NT_SIGMA_SOURCE_KINETIC))return -1;
    FILE *fp=fopen(csv,"r");if(!fp)return -1;
    CMFNonthermalResolvedModel m={0};m.cross_section_policy=policy;Binding *b=NULL;size_t count=0,capacity=0;
    char line[512];int seen[CMF_NT_MAX_ROUTES]={0},routes=0;
    if(!fgets(line,sizeof line,fp)||sscanf(line,"# resolved_routes=%d unresolved_routes=%d missing_ions=%d",
       &m.n_resolved_routes,&m.n_unresolved_routes,&m.n_missing_ions)!=3||m.n_resolved_routes<1||
       m.n_resolved_routes>raw->n_routes||m.n_unresolved_routes<0||m.n_missing_ions<0)goto fail;
    if(!fgets(line,sizeof line,fp)||strcmp(line,"source_line,lower_global,upper_global,target_Z,target_charge,threshold_eV,auger_eV,branch_weight\n")){
        /* Python csv의 CRLF도 같은 헤더다. */
        if(strcmp(line,"source_line,lower_global,upper_global,target_Z,target_charge,threshold_eV,auger_eV,branch_weight\r\n"))goto fail;
    }
    while(fgets(line,sizeof line,fp)){
        int source,lo,up,z,q,used=0,index=-1;double threshold,auger,w;
        if(!strchr(line,'\n')||sscanf(line,"%d,%d,%d,%d,%d,%lf,%lf,%lf%n",&source,&lo,&up,&z,&q,&threshold,&auger,&w,&used)!=8||
           (strcmp(line+used,"\n")&&strcmp(line+used,"\r\n"))||lo<0||lo>=a->n_levels||up< -1||up>=a->n_levels||
           !(threshold>0)||!isfinite(threshold)||!(auger>=0&&auger<threshold)||!isfinite(auger)||
           !(w>0&&w<=1)||!isfinite(w))goto fail;
        for(int k=0;k<raw->n_routes;k++)if(raw->routes[k].source_line==source){if(index!=-1)goto fail;index=k;}
        if(index<0||!(raw->routes[index].threshold_eV>0)||!isfinite(raw->routes[index].threshold_eV)||
           a->level_Z[lo]!=raw->routes[index].Z||a->level_ion[lo]!=raw->routes[index].charge||
           z!=raw->routes[index].Z||q!=raw->routes[index].charge+1+(auger>0))goto fail;
        CMFNonthermalTransition t={lo,up,-1};
        if(up>=0){if(a->level_Z[up]!=z||a->level_ion[up]!=q)goto fail;}
        else for(int ip=0;ip<a->n_ion_pops;ip++)if(a->ion_pop_Z[ip]==z&&a->ion_pop_stage[ip]==q&&
             a->level_offset[ip]==a->level_offset[ip+1]){if(t.upper_ion!=-1)goto fail;t.upper_ion=ip;}
        double delta;int electrons;
        if(cmf_nonthermal_transition_energy(a,&t,&delta,&electrons)||fabs((threshold-auger)/delta-1)>1e-10)goto fail;
        if(count==capacity){
            size_t next=capacity?2*capacity:64;
            if(next>INT_MAX||next>SIZE_MAX/sizeof(*b))goto fail;
            Binding *more=realloc(b,next*sizeof(*b));if(!more)goto fail;b=more;capacity=next;
        }
        b[count++]=(Binding){t,index,threshold,auger,w};
        if(!seen[index]){routes++;seen[index]=1;}
    }
    if(ferror(fp)||routes!=m.n_resolved_routes||count>(size_t)INT_MAX-exc->view.n_channels)goto fail;
    /* 각 원본 경로/하위 준위의 분기 확률을 검증한다. 이 합을1로 보정하지 않는다. */
    for(size_t k=0;k<count;k++){
        double sum=0;
        for(size_t j=0;j<count;j++)if(b[j].raw==b[k].raw&&b[j].target.lower_global==b[k].target.lower_global){
            sum+=b[j].weight;
            if(j!=k&&b[j].target.upper_global==b[k].target.upper_global&&b[j].target.upper_ion==b[k].target.upper_ion)goto fail;
        }
        if(fabs(sum-1)>1e-12)goto fail;
    }
    size_t ne=exc->view.n_energy,nc=count+exc->view.n_channels;
    if(nc>SIZE_MAX/sizeof(*m.channels)||nc>SIZE_MAX/sizeof(*m.transitions)||count>SIZE_MAX/ne/sizeof(double))goto fail;
    m.channels=malloc(nc*sizeof(*m.channels));m.transitions=malloc(nc*sizeof(*m.transitions));
    m.ion_sigma=malloc((count?count:1)*ne*sizeof(double));
    if(!m.channels||!m.transitions||!m.ion_sigma)goto fail;
    memcpy(m.channels,exc->channels,(size_t)exc->view.n_channels*sizeof(*m.channels));
    memcpy(m.transitions,exc->transitions,(size_t)exc->view.n_channels*sizeof(*m.transitions));
    for(size_t k=0;k<count;k++){
        Binding *x=&b[k];size_t j=k+exc->view.n_channels;
        m.transitions[j]=x->target;
        m.channels[j]=(CMFNonthermalRoute){CMF_NT_IONIZATION,x->threshold,0,m.ion_sigma+k*ne,x->auger};
        double shift=raw->routes[x->raw].threshold_eV-x->threshold;
        m.max_threshold_shift_eV=fmax(m.max_threshold_shift_eV,fabs(shift));
        for(size_t i=0;i<ne;i++){
            double argument=exc->energy[i]+(policy==CMF_NT_SIGMA_SAME_EXCESS?shift:0);
            double s=exc->energy[i]<=x->threshold?0:cmf_nonthermal_atomic_sigma(&raw->routes[x->raw],argument)*x->weight;
            if(!(s>=0)||!isfinite(s))goto fail;
            m.ion_sigma[k*ne+i]=s;
        }
    }
    m.n_ion_channels=(int)count;m.view=exc->view;m.view.n_channels=(int)nc;
    m.view.channels=m.channels;m.view.transitions=m.transitions;
    fclose(fp);free(b);cmf_nonthermal_binding_free(out);*out=m;return 0;
fail:
    fprintf(stderr,"NT_BINDING_BUILD_FAIL parsed_channels=%zu resolved=%d expected=%d\n",count,routes,m.n_resolved_routes);
    fclose(fp);free(b);cmf_nonthermal_binding_free(&m);return -1;
}
