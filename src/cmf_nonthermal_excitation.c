#include "cmf_nonthermal_excitation.h"
#include <limits.h>
void cmf_nonthermal_excitation_free(CMFNonthermalExcitations *x){
    if(x){free(x->transitions);free(x->channels);free(x->energy);free(x->sigma);memset(x,0,sizeof(*x));}
}
typedef struct { double f,A,omega; unsigned char line,collision; } Pair;
int cmf_nonthermal_excitation_build(const AtomicData *a,const double *E,int ne,
                                   int limit,CMFNonthermalExcitations *out){
    if(!a||!E||!out||ne<2||ne>100000||limit<1||a->n_levels<1||a->n_ion_pops<1||
       a->n_lines<0||!a->level_offset||!a->level_num||!a->level_Z||!a->level_ion||
       !a->level_energy_eV||!a->level_g||!a->ion_pop_Z||!a->ion_pop_stage||
       (a->n_lines&&(!a->line_atomic_number||!a->line_ion_number||!a->line_level_lower||
                    !a->line_level_upper||!a->line_f_lu||!a->line_A_ul))||
       a->ncol_ions<0||a->ncol_ions>LUMINA_MAX_COL_IONS)return -1;
    for(int i=0;i<ne;i++)if(!(E[i]>0)||!isfinite(E[i])||(i&&E[i]<=E[i-1]))return -1;
    if((size_t)limit>SIZE_MAX/(size_t)a->n_levels/sizeof(Pair))return -1;
    Pair *pairs=calloc((size_t)limit*a->n_levels,sizeof(*pairs));
    if(!pairs)return -1;
    int ion[119][119];memset(ion,-1,sizeof ion);
    CMFNonthermalExcitations x={0};x.lower_limit=limit;x.minimum_threshold_eV=INFINITY;
    int count=0,record=-1;const char *phase="level IDs";
    if(a->level_offset[0]!=0||a->level_offset[a->n_ion_pops]!=a->n_levels)goto fail;
    for(int ip=0;ip<a->n_ion_pops;ip++){
        int z=a->ion_pop_Z[ip],q=a->ion_pop_stage[ip],lo=a->level_offset[ip],hi=a->level_offset[ip+1];
        if(z<1||z>118||q<0||q>118||ion[z][q]!=-1||lo<0||hi<lo||hi>a->n_levels)goto fail;
        ion[z][q]=ip;
        for(int gl=lo;gl<hi;gl++)if(a->level_num[gl]!=gl-lo||a->level_Z[gl]!=z||a->level_ion[gl]!=q||
            !(a->level_g[gl]>0)||!isfinite(a->level_energy_eV[gl]))goto fail;
    }
    phase="line pairs";
    for(int l=0;l<a->n_lines;l++){
        record=l;
        int z=a->line_atomic_number[l],q=a->line_ion_number[l];
        if(z<1||z>118||q<0||q>118||ion[z][q]<0)goto fail;
        int ip=ion[z][q],base=a->level_offset[ip],nl=a->level_offset[ip+1]-base;
        int lo=a->line_level_lower[l],hi=a->line_level_upper[l]; /* ion-local IDs */
        if(lo<0||hi<0||lo>=nl||hi>=nl||!isfinite(a->line_f_lu[l])||a->line_f_lu[l]<0||
           !isfinite(a->line_A_ul[l])||a->line_A_ul[l]<0)goto fail;
        if(lo>=limit||a->level_energy_eV[base+hi]<=a->level_energy_eV[base+lo])continue;
        Pair *p=&pairs[(size_t)lo*a->n_levels+base+hi];
        if(p->line&&(p->f!=a->line_f_lu[l]||p->A!=a->line_A_ul[l]))goto fail;
        p->line=1;p->f=a->line_f_lu[l];p->A=a->line_A_ul[l];
    }
    phase="collision pairs";
    for(int c=0;c<a->ncol_ions;c++){
        record=c;
        int z=a->col_ion_Z[c],q=a->col_ion_stage[c],nt=a->col_ion_n_temp[c],nr=a->col_ion_n_trans[c];
        if(z<1||z>118||q<0||q>118||ion[z][q]<0||nt<1||nr<0||!a->col_ion_tgrid[c]||
           (nr&&(!a->col_ion_lo[c]||!a->col_ion_hi[c]||!a->col_ion_omega[c])))goto fail;
        for(int t=0;t<nt;t++)if(!(a->col_ion_tgrid[c][t]>0)||!isfinite(a->col_ion_tgrid[c][t])||
            (t&&a->col_ion_tgrid[c][t]<=a->col_ion_tgrid[c][t-1]))goto fail;
        int ip=ion[z][q],base=a->level_offset[ip],nl=a->level_offset[ip+1]-base;
        for(int j=0;j<nr;j++){
            int lo=a->col_ion_lo[c][j],hi=a->col_ion_hi[c][j];
            double omega=a->col_ion_omega[c][(size_t)j*nt+nt-1];
            if(lo<0||hi<0||lo>=nl||hi>=nl||!(omega>=0)||!isfinite(omega))goto fail;
            if(lo>=limit||a->level_energy_eV[base+hi]<=a->level_energy_eV[base+lo])continue;
            Pair *p=&pairs[(size_t)lo*a->n_levels+base+hi];
            if(p->collision&&p->omega!=omega)goto fail;
            p->collision=1;p->omega=omega;
        }
    }
    /* 같은 두 준위의 f/Ω를 한 채널로 합친다. 존재하지 않는 충돌강도는0,
     * 약한 A<=1000/s에서 실제 Ω가 있으면 그 값을 선택한다. */
    phase="cross sections";record=-1;
    for(int pass=0;pass<2;pass++){
        int k=0;
        for(int ip=0;ip<a->n_ion_pops;ip++){
            int base=a->level_offset[ip],nl=a->level_offset[ip+1]-base;
            for(int lo=0;lo<nl&&lo<limit;lo++)for(int hi=0;hi<nl;hi++){
                Pair *p=&pairs[(size_t)lo*a->n_levels+base+hi];
                int bethe=p->A>1000||p->omega==0;
                if((bethe&&p->f==0)||(!bethe&&p->omega==0))continue;
                double delta=a->level_energy_eV[base+hi]-a->level_energy_eV[base+lo];
                if(!(delta>0)||!isfinite(delta)||k==INT_MAX)goto fail;
                if(pass){
                    x.transitions[k]=(CMFNonthermalTransition){base+lo,base+hi,-1};
                    x.channels[k]=(CMFNonthermalRoute){CMF_NT_EXCITATION,delta,0,x.sigma+(size_t)k*ne,0};
                    for(int i=0;i<ne;i++){
                        double s=bethe?cmf_nonthermal_bethe(E[i],delta,p->f,a->ion_pop_stage[ip]):
                                      cmf_nonthermal_omega(E[i],delta,p->omega,a->level_g[base+lo]);
                        if(!(s>=0)||!isfinite(s))goto fail;
                        x.sigma[(size_t)k*ne+i]=s;
                    }
                    x.n_bethe+=bethe;x.n_omega+=!bethe;x.n_subthermal+=delta<E[0];
                    x.minimum_threshold_eV=fmin(x.minimum_threshold_eV,delta);
                }
                k++;
            }
        }
        if(!pass){
            count=k;
            if((size_t)count>SIZE_MAX/(size_t)ne/sizeof(double))goto fail;
            x.transitions=calloc(count?count:1,sizeof(*x.transitions));
            x.channels=calloc(count?count:1,sizeof(*x.channels));
            x.energy=malloc((size_t)ne*sizeof(double));
            x.sigma=calloc(count?(size_t)count*ne:1,sizeof(double));
            if(!x.transitions||!x.channels||!x.energy||!x.sigma)goto fail;
            memcpy(x.energy,E,(size_t)ne*sizeof(double));
        }else if(k!=count)goto fail;
    }
    x.view=(CMFNonthermalBoundModel){a->n_levels,a->n_ion_pops,ne,count,x.energy,x.transitions,x.channels};
    free(pairs);cmf_nonthermal_excitation_free(out);*out=x;return 0;
fail:
    fprintf(stderr,"NT_EXCITATION_BUILD_FAIL phase=%s record=%d\n",phase,record);
    free(pairs);cmf_nonthermal_excitation_free(&x);return -1;
}
