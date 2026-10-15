/* 실제 원본 hydro의12100--12500km/s,8개 셀에 기존 Saha 공급자를 적용한다.
 * 전체 ejecta/NLTE/수송 계산이 아니며 공개 인구나 온도를 입력으로 쓰지 않는다. */
#include "cmf_toy_model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"NATIVE_CELLS_FAIL line=%d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
int main(int argc,char **argv){
    if(argc!=4||atoi(argv[3])!=8)return 2;
    enum {NS=8};
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    const int elements[6]={28,27,26,20,16,14};
    CMFToyEjecta ejecta={0};CMFToyModel view={0};
    AtomicData a={0};PlasmaState p={0};OpacityState o={0};Geometry g={0};
    double ri[NS],ro[NS],rho[NS],T[NS],ne[NS]={0};int rc=0,opened=0;
    CHECK(!cmf_toy_ejecta_read(&ejecta,argv[1],0,masses));
    CHECK(ejecta.n_shells==807&&ejecta.epoch_s==172800);
    CHECK(load_atomic_structure(&a,argv[2],NS)==0);
    p.n_shells=g.n_shells=o.n_shells=NS;p.initial_steady_nlte=1;
    p.rho=rho;p.T_e=T;p.n_electron=ne;g.r_inner=ri;g.r_outer=ro;
    g.time_explosion=ejecta.epoch_s;
    o.n_lines=a.n_lines;o.line_list_nu=a.line_nu;o.electron_density=ne;o.t_electrons=T;
    size_t cells=(size_t)a.n_lines*NS;
    o.tau_sobolev=calloc(cells,sizeof(double));o.line_source_S=calloc(cells,sizeof(double));
    o.tau_validity=calloc(cells,sizeof(*o.tau_validity));
    o.line_source_validity=calloc(cells,sizeof(*o.line_source_validity));
    CHECK(o.tau_sobolev&&o.line_source_S&&o.tau_validity&&o.line_source_validity);
    int selected=0;
    for(int i=0;i<ejecta.n_shells;i++){
        if(ejecta.velocity_edges[i]<12100e5||ejecta.velocity_edges[i+1]>12500e5)continue;
        CHECK(selected<NS);int s=selected++;
        ri[s]=ejecta.velocity_edges[i]*ejecta.epoch_s;
        ro[s]=ejecta.velocity_edges[i+1]*ejecta.epoch_s;
        double volume=(4*M_PI_VAL/3)*(ro[s]-ri[s])*(ro[s]*ro[s]+ri[s]*ro[s]+ri[s]*ri[s]);
        rho[s]=ejecta.mass_g[i]/volume;T[s]=ejecta.initial_temperature[i];
        for(int e=0;e<a.n_elements;e++){
            int k=0;while(k<6&&elements[k]!=a.element_Z[e])k++;CHECK(k<6);
            a.element_mass_amu[e]=masses[k];
            a.abundances[e*NS+s]=ejecta.number_per_g[k*ejecta.n_shells+i]*masses[k]*AMU;
            ne[s]+=a.element_Z[e]*rho[s]*ejecta.number_per_g[k*ejecta.n_shells+i];
        }
    }
    CHECK(selected==NS);
    CHECK(!lumina_publish_seed_te(&p,"original hydro cells; initial Saha diagnostic only"));
    CHECK(!lumina_bootstrap_window_open("original hydro cell diagnostic"));opened=1;
    CHECK(!lumina_prepare_solver_owned_tau(&a,&p,&o,g.time_explosion,"native-cell Saha comparison"));
    lumina_bootstrap_window_close();opened=0;
    double charge_error=0,nucleus_error=0;
    for(int s=0;s<NS;s++){
        double charge=0;
        for(int e=0;e<a.n_elements;e++){
            double total=0,expected=rho[s]*a.abundances[e*NS+s]/(a.element_mass_amu[e]*AMU);
            for(int ip=a.elem_ion_offset[e];ip<a.elem_ion_offset[e+1];ip++){
                double value=a.ion_number_density[ip*NS+s];CHECK(value>=0&&isfinite(value));
                total+=value;charge+=value*a.ion_pop_stage[ip];
            }
            if(expected>0)nucleus_error=fmax(nucleus_error,fabs(total/expected-1));else CHECK(total==0);
        }
        CHECK(charge>0);charge_error=fmax(charge_error,fabs(ne[s]/charge-1));
    }
    CHECK(charge_error<1e-8&&nucleus_error<1e-10);
    view.prepared=1;view.atom=a;view.plasma=p;view.geometry=g;
    CHECK(!cmf_toy_model_write_seed(&view,"toy_initial_material.csv"));
    printf("NATIVE_CELLS_SAHA_PASS cells=%d velocity_edges=12100,12500 charge_error=%.17g nuclei_error=%.17g NLTE_solved=0 radiation_solved=0\n",
        NS,charge_error,nucleus_error);
done:
    if(opened)lumina_bootstrap_window_close();
    o.line_list_nu=o.electron_density=o.t_electrons=NULL;
    free_opacity_state(&o);a210_publication_free(&p.te_publication);
    free_atomic_data(&a);cmf_toy_ejecta_free(&ejecta);
    return rc;
}
