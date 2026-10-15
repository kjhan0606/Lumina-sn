#include "cmf_ejecta.h"
#include "nlte_population_candidate.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"EJECTA_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
static int near(long double a,long double b,long double tol){return fabsl(a-b)<=tol*fmaxl(fabsl(a),fabsl(b));}
static long double cell_volume(const CMFToyEjecta *m,int s,double t){
    long double r=m->velocity_edges[s]*t,R=m->velocity_edges[s+1]*t;
    return 4*acosl(-1.L)/3*(R-r)*(R*R+R*r+r*r);
}
static void sums(const CMFToyEjecta *m,long double sum[8]){
    memset(sum,0,8*sizeof(long double));
    for(int s=0;s<m->n_shells;s++){
        sum[0]+=m->mass_g[s];sum[1]+=cell_volume(m,s,m->epoch_s)*powl(m->initial_temperature[s],4);
        for(int k=0;k<6;k++)sum[2+k]+=m->mass_g[s]*m->number_per_g[k*m->n_shells+s];
    }
}
static long double kinetic_energy(const CMFToyEjecta *m){
    long double sum=0;
    for(int s=0;s<m->n_shells;s++){
        long double l=m->velocity_edges[s],h=m->velocity_edges[s+1];
        long double mean_v2=.6L*(powl(h,5)-powl(l,5))/(h*h*h-l*l*l);
        sum+=.5L*m->mass_g[s]*mean_v2;
    }
    return sum;
}

static int refinement(const CMFToyEjecta *full,const CMFToyEjecta *base,const double *mass){
    CMFToyEjecta fine={0},disabled={0};long double expected[8];sums(full,expected);
    CHECK(!cmf_toy_ejecta_read_refined(&disabled,"snia_toy06_2d.dat",100,mass,0,256));
    CHECK(disabled.n_shells==base->n_shells);
    CHECK(!memcmp(disabled.storage,base->storage,(size_t)(9*base->n_shells+1)*sizeof(double)));
    const double ratios[2]={1.10,1.05};int previous_count=base->n_shells;
    for(int r=0;r<2;r++){
        CHECK(!cmf_toy_ejecta_read_refined(&fine,"snia_toy06_2d.dat",100,mass,ratios[r],256));
        CHECK(fine.n_shells>base->n_shells&&fine.n_shells>=previous_count);
        previous_count=fine.n_shells;
        long double actual[8];sums(&fine,actual);
        for(int k=0;k<8;k++)CHECK(near(expected[k],actual[k],2e-15));
        for(int s=0;s<=base->n_shells;s++){
            int found=0;
            for(int j=0;j<=fine.n_shells;j++)if(base->velocity_edges[s]==fine.velocity_edges[j])found++;
            CHECK(found==1);
        }
        int i=0;double worst=1;
        for(int s=0;s<fine.n_shells;s++){
            CHECK(i<full->n_shells&&fine.velocity_edges[s]==full->velocity_edges[i]);
            double tmin=INFINITY,tmax=0;
            while(i<full->n_shells&&full->velocity_edges[i+1]<=fine.velocity_edges[s+1]){
                tmin=fmin(tmin,full->initial_temperature[i]);
                tmax=fmax(tmax,full->initial_temperature[i]);i++;
            }
            CHECK(i>0&&full->velocity_edges[i]==fine.velocity_edges[s+1]);
            CHECK(tmax/tmin<=ratios[r]*(1+2e-15));
            CHECK(fine.initial_temperature[s]>=tmin*(1-2e-15)&&fine.initial_temperature[s]<=tmax*(1+2e-15));
            worst=fmax(worst,tmax/tmin);
        }
        CHECK(i==full->n_shells);
        double *saved=fine.storage;
        CHECK(cmf_toy_ejecta_read_refined(&fine,"snia_toy06_2d.dat",100,mass,ratios[r],100));
        CHECK(fine.storage==saved);
        printf("EJECTA_T_REFINEMENT_PASS base=100 shells=%d T_ratio_limit=%.17g actual=%.17g mass_nuclei_T4_conserved=1 base_edges_retained=1 kinetic_error=%.9Lg\n",
               fine.n_shells,ratios[r],worst,kinetic_energy(&fine)/kinetic_energy(full)-1);
    }
    const double invalid[4]={1,-1,NAN,INFINITY};double *saved=fine.storage;
    for(int i=0;i<4;i++){
        CHECK(cmf_toy_ejecta_read_refined(&fine,"snia_toy06_2d.dat",100,mass,invalid[i],256));
        CHECK(saved==fine.storage);
    }
    cmf_toy_ejecta_free(&fine);cmf_toy_ejecta_free(&disabled);return 0;
}

static int projection(const CMFToyEjecta *m,const CMFEjectaEpoch *now,const CMFEjectaEpoch *initial){
    enum{NI=12,NE=6};int ns=m->n_shells;
    int el[NE]={28,27,26,20,16,14},eo[NE+1]={0,2,4,6,8,10,12},Z[NI],stage[NI],off[NI+1],map[NI],g[NI];
    double E[NI]={0},chi[NE]={10,20,30,40,50,60};int istage[NE]={0};
    for(int i=0;i<NI;i++){Z[i]=el[i/2];stage[i]=i%2;off[i]=map[i]=i;g[i]=1;}off[NI]=NI;
    double *buf=calloc((size_t)ns*(4+4*NI+NE),sizeof(double));CHECK(buf);
    double *T=buf,*rho=T+ns,*electron=rho+ns,*radius=electron+ns,*pop=radius+ns;
    double *part=pop+NI*ns,*within=part+NI*ns,*preserved=within+NI*ns,*after=preserved+NI*ns;
    for(int s=0;s<ns;s++){
        T[s]=m->initial_temperature[s];rho[s]=initial->density[s];
        for(int k=0;k<NE;k++){
            pop[(2*k)*ns+s]=.4*initial->number_density[k*ns+s];
            pop[(2*k+1)*ns+s]=.6*initial->number_density[k*ns+s];
            electron[s]+=pop[(2*k+1)*ns+s];
            after[k*ns+s]=now->number_density[k*ns+s]*initial->density[s]/now->density[s];
        }
    }
    for(int i=0;i<NI*ns;i++)within[i]=1;
    memcpy(preserved,pop,NI*ns*sizeof(double));
    AtomicData a={0};NLTEConfig n={0};PlasmaState p={0};
    a.n_elements=NE;a.element_Z=el;a.elem_ion_offset=eo;a.n_ion_pops=a.n_levels=NI;
    a.ion_pop_Z=Z;a.ion_pop_stage=stage;a.level_Z=Z;a.level_ion=stage;a.level_g=g;
    a.level_offset=off;a.level_energy_eV=E;a.ion_number_density=pop;a.partition_functions=part;
    a.n_ionization=NE;a.ioniz_Z=el;a.ioniz_ion=istage;a.ioniz_energy_eV=chi;a.population_committed_generation=1;
    n.n_nlte_levels_total=n.n_super_total=NI;n.n_nlte_ions=NI;
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=map;n.nlte_level_populations=pop;
    n.within_sl_frac=within;n.population_committed_generation=1;
    p.n_shells=ns;p.rho=rho;p.T_e=T;p.n_electron=electron;p.T_e_generation=1;
    PopulationAtomicView v={0};v.n_ions=v.n_levels=NI;v.level_offset=off;
    v.energy_eV=E;v.g=g;v.level_Z=Z;v.level_ion=stage;
    CHECK(population_partition_build(&v,T,ns,1,1,part,&a.partition_stamp)==POP_OK);
    NLTEPhysicalHistory old={0},projected={0};
    CHECK(nlte_physical_history_capture(&old,&n,&a,&p,m->epoch_s)==NLTE_CANDIDATE_OK);
    CHECK(nlte_physical_history_decay_projection(&projected,&old,&n,&a,now->epoch_s,after)==NLTE_CANDIDATE_OK);
    p.physical_history=&projected;memcpy(rho,now->density,ns*sizeof(double));
    double A[4]={0},solution[2];
    for(int s=0;s<ns;s++)for(int k=0;k<NE;k++){
        CHECK(nlte_physical_history_solve_element(&n,&a,&p,s,now->epoch_s,2*k,2,-1,A,solution)==POP_OK);
        CHECK(near(solution[0]+solution[1],now->number_density[k*ns+s],2e-13));
    }
    CHECK(!memcmp(old.level_density,preserved,NI*ns*sizeof(double)));
    nlte_physical_history_free(&old);nlte_physical_history_free(&projected);free(buf);
    puts("EJECTA_PHYSICAL_HISTORY_PASS real_toy_composition=1 synthetic_ion_fractions=1 actual_time_consumer=1");return 0;
}

int main(void){
    /* 로컬 핵 질량의 명시적 선택. 공개 CMFGEN 핵파일과 동일하다고 주장하지 않는다.
     * 방출 에너지는 합성값이며 실제 감마선 침적을 시험하지 않는다. */
    double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyEjecta full={0},m={0},finer={0};
    CHECK(!cmf_toy_ejecta_read(&full,"snia_toy06_2d.dat",0,mass));
    CHECK(!cmf_toy_ejecta_read(&m,"snia_toy06_2d.dat",100,mass));
    CHECK(!cmf_toy_ejecta_read(&finer,"snia_toy06_2d.dat",200,mass));
    CHECK(!refinement(&full,&m,mass));
    CHECK(full.n_shells==807&&m.n_source_shells==807&&m.epoch_s==172800&&m.velocity_edges[0]==0);
    long double sf[8],sc[8],sd[8];sums(&full,sf);sums(&m,sc);sums(&finer,sd);
    for(int k=0;k<8;k++)CHECK(near(sf[k],sc[k],2e-15)&&near(sf[k],sd[k],2e-15));
    CHECK(near(sf[0]/1.989e33,.99990L,1e-5));
    double *saved=m.storage;CHECK(cmf_toy_ejecta_read(&m,"snia_toy06_2d.dat",900,mass)&&saved==m.storage);
    printf("EJECTA_REMAP_PASS source=%d coarse=%d fine=%d mass_Msun=%.17Lg Ni0_Msun=%.17g density_rounding=%.9g fraction_rounding=%.9g\n",
        full.n_shells,m.n_shells,finer.n_shells,sf[0]/1.989e33,m.initial_ni56_mass_g/1.989e33,
        m.max_input_density_error,m.max_input_fraction_error);
    long double kin=kinetic_energy(&full),coarse_error=kinetic_energy(&m)/kin-1,fine_error=kinetic_energy(&finer)/kin-1;
    printf("EJECTA_REMAP_RESOLUTION kinetic_erg=%.17Lg error100=%.9Lg error200=%.9Lg\n",kin,coarse_error,fine_error);
    CHECK(fabsl(fine_error)<fabsl(coarse_error));
    CMFDecaySourceData d={{log(2)/m.model_halflife_s[0],log(2)/m.model_halflife_s[1]},
                         {2e-6,4e-6},{0,2e-7}};
    CMFEjectaEpoch init={0},now={0},average={0};double t=19.48*86400,dt=t-m.epoch_s;
    CHECK(!cmf_ejecta_epoch_build(&init,&m,&d,m.epoch_s,m.epoch_s,CMF_DECAY_ENDPOINT));
    CHECK(!cmf_ejecta_epoch_build(&now,&m,&d,m.epoch_s,t,CMF_DECAY_ENDPOINT));
    CHECK(!cmf_ejecta_epoch_build(&average,&m,&d,m.epoch_s,t,CMF_DECAY_INTERVAL_AVERAGE));
    for(int s=0;s<m.n_shells;s++){
        int ns=m.n_shells;double Ni=m.number_per_g[s],Co=m.number_per_g[ns+s];
        double ni=Ni*exp(-d.lambda[0]*dt),co=Co*exp(-d.lambda[1]*dt)+
            Ni*d.lambda[0]/(d.lambda[1]-d.lambda[0])*(exp(-d.lambda[0]*dt)-exp(-d.lambda[1]*dt));
        CHECK(near(init.number_density[s],Ni*init.density[s],2e-15));
        CHECK(near(now.number_density[s],ni*now.density[s],3e-14));
        CHECK(near(now.number_density[ns+s],co*now.density[s],3e-14));
        double oldsum=0,newsum=0;for(int k=0;k<3;k++){oldsum+=init.number_density[k*ns+s]/init.density[s];newsum+=now.number_density[k*ns+s]/now.density[s];}
        CHECK(near(oldsum,newsum,3e-15));
        CHECK(near(now.gamma_emission[s],now.density[s]*(2e-6*d.lambda[0]*ni+4e-6*d.lambda[1]*co),3e-14));
        double dn=Ni-ni,dc=now.number_density[2*ns+s]/now.density[s]-m.number_per_g[2*ns+s];
        CHECK(near(average.gamma_emission[s]/now.density[s]*dt,2e-6*dn+4e-6*dc,3e-14));
        CHECK(near(average.positron_emission[s]/now.density[s]*dt,2e-7*dc,3e-14));
        CHECK(near(now.density[s]/init.density[s],powl(m.epoch_s/t,3),3e-15));
    }
    saved=now.storage;CHECK(cmf_ejecta_epoch_build(&now,&m,&d,0,t,CMF_DECAY_ENDPOINT)&&now.storage==saved);
    CHECK(cmf_ejecta_epoch_build(&now,&m,&d,t,t,CMF_DECAY_INTERVAL_AVERAGE)&&now.storage==saved);
    CHECK(!projection(&m,&now,&init));
    int ns=m.n_shells,el[6]={14,16,20,26,27,28},order[6]={5,4,3,2,1,0};
    double amass[6]={28.0855,32.065,40.078,55.845,58.933194,58.6934};
    double *arrays=calloc((size_t)ns*11,sizeof(double));CHECK(arrays);
    Geometry g={0};g.n_shells=ns;g.r_inner=arrays;g.r_outer=arrays+ns;g.v_inner=arrays+2*ns;g.v_outer=arrays+3*ns;
    PlasmaState p={0};p.n_shells=ns;p.rho=arrays+4*ns;
    AtomicData a={0};a.n_elements=6;a.element_Z=el;a.element_mass_amu=amass;a.abundances=arrays+5*ns;
    CHECK(!cmf_ejecta_apply_epoch(&m,&now,&g,&p,&a));
    CHECK(g.r_inner[0]==0&&g.time_explosion==t&&a.element_mass_amu[5]==mass[0]);
    for(int k=0;k<6;k++)for(int s=0;s<ns;s++){
        double n=a.abundances[k*ns+s]*p.rho[s]/(a.element_mass_amu[k]*AMU);
        CHECK(near(n,now.number_density[order[k]*ns+s],5e-15));
    }
    double held=p.rho[0],held_input=now.density[0];now.density[0]=-1;
    CHECK(cmf_ejecta_apply_epoch(&m,&now,&g,&p,&a)&&p.rho[0]==held&&g.time_explosion==t);
    now.density[0]=held_input;
    puts("EJECTA_EPOCH_PASS double_decay=0 nuclei_conserved=1 endpoint_and_interval_energy=1 mass_bridge=1 deposition_implemented=0");
    free(arrays);cmf_ejecta_epoch_free(&init);cmf_ejecta_epoch_free(&now);cmf_ejecta_epoch_free(&average);
    cmf_toy_ejecta_free(&full);cmf_toy_ejecta_free(&m);cmf_toy_ejecta_free(&finer);return 0;
}
