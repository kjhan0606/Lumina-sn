/* 실제 조립 입력을 한 셸에서만 원소별로 분해한다. 생산 배열은 읽기 전용이다. */
#include "bf_channels.h"
#include "cmf_free_free.h"
static int origin_index(int Z){
    const int elements[]={14,16,20,26,27,28};
    for(int k=0;k<6;k++)if(elements[k]==Z)return k;
    return -1;
}
static int origin_capture(CMFToyModel *m,int nf,const double *freq,
                         const double *chi,const double *eta){
    const int s=39,ns=m->plasma.n_shells;
    const AtomicData *a=&m->atom;const NLTEConfig *n=&m->nlte;
    const BfChannels *p=a->bf_channels;int nb=p->n_bins,rc=-1;
    double T=m->plasma.T_e[s],ne=m->plasma.n_electron[s],time=m->geometry.time_explosion;
    double *coarse=calloc((size_t)14*nb,sizeof(double));
    double *line=calloc((size_t)12*nf,sizeof(double));
    double *pop=malloc((size_t)a->n_levels*sizeof(double));FILE *f=NULL;
    if(!coarse||!line||!pop)goto done;
    for(int gl=0;gl<a->n_levels;gl++){
        int fl=n->global_to_nlte_level[gl];if(fl<0)goto done;
        pop[gl]=n->nlte_level_populations[(size_t)fl*ns+s];
    }
    double dlog=m->bf.d_log_nu,z2ni=0;
    for(int ip=0;ip<a->n_ion_pops;ip++)
        z2ni+=(double)a->ion_pop_stage[ip]*a->ion_pop_stage[ip]*a->ion_number_density[(size_t)ip*ns+s];
    for(int l=0;l<p->n_levels;l++){
        int k=origin_index(a->level_Z[l]);if(k<0)goto done;
        for(int r=p->offset[l];r<p->offset[l+1];r++){
            int u=p->upper[r],ip=p->upper_ion[r];
            double gu=u>=0?a->level_g[u]:p->upper_g[r];
            double nu=u>=0?pop[u]:a->ion_number_density[(size_t)ip*ns+s];
            for(int b=0;b<nb;b++){
                double sigma=p->sigma[(size_t)r*nb+b];if(sigma==0)continue;
                double lo=p->nu_min*exp(b*dlog),hi=p->nu_min*exp((b+1)*dlog);
                BfChannelBin v;
                if(bf_channel_bin(lo,hi,p->threshold[r],sigma,T,ne,a->level_g[l],gu,pop[l],nu,0,0,&v))goto done;
                coarse[(size_t)(2*k)*nb+b]+=v.chi_net;
                coarse[(size_t)(2*k+1)*nb+b]+=v.eta;
            }
        }
    }
    double coarse_error=0;
    for(int b=0;b<nb;b++){
        double nu=p->nu_min*exp((b+.5)*dlog),cf,ef;
        if(cmf_free_free(nu,T,ne,z2ni,&cf,&ef))goto done;
        coarse[(size_t)12*nb+b]=cf;coarse[(size_t)13*nb+b]=ef;
        long double c=0,e=0,sc=0;
        for(int k=0;k<7;k++){c+=coarse[(size_t)(2*k)*nb+b];sc+=fabs(coarse[(size_t)(2*k)*nb+b]);e+=coarse[(size_t)(2*k+1)*nb+b];}
        double actual_c=m->bf.chi_bf[(size_t)s*nb+b],actual_e=m->bf.eta_bf[(size_t)s*nb+b];
        double ec=sc>0?(double)(fabsl(c-actual_c)/sc):fabs(actual_c);
        double ee=e>0?(double)(fabsl(e-actual_e)/e):fabs(actual_e);
        coarse_error=fmax(coarse_error,fmax(ec,ee));
    }
    if(coarse_error>1e-10)goto done;
    const double vdop=5e6,fdlog=m->opacity.dlognu_fine;
    double flo=freq[0]*exp(-.5*fdlog);
    for(int l=0;l<a->n_lines;l++){
        int k=origin_index(a->line_atomic_number[l]);if(k<0)goto done;
        int gu=n->nlte_line_global[2*(size_t)l+1];if(gu<0)goto done;
        double nu=a->line_nu[l],dop=nu*vdop/C_SPEED_OF_LIGHT;
        int center=(int)floor(log(nu/flo)/fdlog);
        int half=(int)ceil(4*dop/(nu*fdlog))+1;
        LineNetSobolevMaterial material;
        if(line_net_sobolev_material(pop[gu],a->line_A_ul[l],nu,
            m->opacity.tau_sobolev[(size_t)l*ns+s],time,LINE_NET_NEGATIVE_OPACITY_NONE,1,&material))goto done;
        double c=material.effective_integrated_opacity/(1.7724538509055160*dop);
        double e=material.emission_per_sr/(1.7724538509055160*dop);
        int begin=center-half,end=center+half;if(begin<0)begin=0;if(end>=nf)end=nf-1;
        for(int b=begin;b<=end;b++){
            double x=(freq[b]-nu)/dop;if(fabs(x)>4)continue;
            double profile=exp(-x*x);
            line[(size_t)(2*k)*nf+b]+=c*profile;
            line[(size_t)(2*k+1)*nf+b]+=e*profile;
        }
    }
    double fine_error=0;
    for(int b=0;b<nf;b++){
        double x=log(freq[b]/p->nu_min)/dlog-.5;int i=(int)floor(x);double w=x-i;
        if(i<0){i=0;w=0;}if(i>=nb-1){i=nb-2;w=1;}
        long double c=ne*SIGMA_THOMSON,e=0,sc=c;
        for(int k=0;k<7;k++){
            double cc=(1-w)*coarse[(size_t)(2*k)*nb+i]+w*coarse[(size_t)(2*k)*nb+i+1];
            c+=cc;sc+=fabs(cc);
            e+=(1-w)*coarse[(size_t)(2*k+1)*nb+i]+w*coarse[(size_t)(2*k+1)*nb+i+1];
        }
        if(freq[b]<p->nu_min){
            double cf,ef;if(cmf_free_free(freq[b],T,ne,z2ni,&cf,&ef))goto done;
            c=ne*SIGMA_THOMSON+cf;sc=c;e=ef;
        }
        for(int k=0;k<6;k++){c+=line[(size_t)(2*k)*nf+b];sc+=fabs(line[(size_t)(2*k)*nf+b]);e+=line[(size_t)(2*k+1)*nf+b];}
        double ec=sc>0?(double)(fabsl(c-chi[b])/sc):fabs(chi[b]);
        double ee=e>0?(double)(fabsl(e-eta[b])/e):fabs(eta[b]);
        fine_error=fmax(fine_error,fmax(ec,ee));
    }
    printf("ORIGIN_REPRODUCTION coarse=%.17g fine=%.17g shell=%d\n",coarse_error,fine_error,s);
    if(fine_error>1e-10)goto done;
    f=fopen("origin_shell39.bin","wx");if(!f)goto done;
    uint64_t shape[]={nb,nf};double par[]={p->nu_min,dlog,T,ne,z2ni};
    int ok=fwrite("ORIGIN01",1,8,f)==8&&fwrite(shape,8,2,f)==2&&fwrite(par,8,5,f)==5&&
        fwrite(coarse,8,(size_t)14*nb,f)==(size_t)14*nb&&
        fwrite(line,8,(size_t)12*nf,f)==(size_t)12*nf;
    if(fclose(f))ok=0;f=NULL;if(!ok)goto done;rc=0;
done:
    if(f)fclose(f);free(coarse);free(line);free(pop);return rc;
}
