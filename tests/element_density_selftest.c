/* 동위원소 상태의 고정 총수와 자연 원자량 환산의 분리를 확인한다. */
#include "lumina.h"

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"[ELEMENT-DENSITY][FAIL] line=%d\n",__LINE__); return 1; } } while(0)

static PopulationStatus close_state(PopulationElementDensityView *view,
                                     double *ions, double *ne) {
    int iz[]={27,27,27},iq[]={1,2,3},ez[]={27},eo[]={0,3};
    int nz[]={27,27},nq[]={1,2},lo[]={0,1,2};
    double level[]={20,30},mass[]={NAN},x[]={NAN},rho[]={NAN};
    double residual;int worst;
    return population_nlte_stage_charge_close_with_density(
        3,iz,iq,1,ez,eo,mass,x,rho,AMU,2,nz,nq,lo,2,1,level,
        ions,ne,1,&residual,&worst,view);
}

int main(int argc, char **argv) {
    double inventory[]={100,0,200,10};
    PopulationElementDensityView view={POP_ELEMENT_TOTAL_EXPLICIT_NUMBER,2,2,inventory};
    double mass[]={58.933194,28.0855},x[]={1,0,1,1},rho[]={1e-10,1e-20};
    double n=-1;
    REQUIRE(population_element_density_get(&view,2,2,0,0,mass,x,rho,AMU,&n)==POP_OK && n==100);
    mass[0]=NAN;x[0]=NAN;rho[0]=NAN;
    REQUIRE(population_element_density_get(&view,2,2,0,0,mass,x,rho,NAN,&n)==POP_OK && n==100);
    REQUIRE(population_element_density_get(&view,2,2,0,1,NULL,NULL,NULL,NAN,&n)==POP_EXACT_ZERO && n==0);
    view.number_density=NULL;n=-1;
    REQUIRE(population_element_density_get(&view,2,2,0,0,mass,x,rho,AMU,&n)==POP_ATOMIC_MISSING && n==-1);
    view.number_density=inventory;
    REQUIRE(population_element_density_get(&view,2,1,0,0,mass,x,rho,AMU,&n)==POP_ATOMIC_MISSING);
    inventory[0]=NAN;
    REQUIRE(population_element_density_get(&view,2,2,0,0,mass,x,rho,AMU,&n)==POP_NONFINITE);
    inventory[0]=100;
    view=(PopulationElementDensityView){POP_ELEMENT_TOTAL_EXPLICIT_NUMBER,1,1,inventory};
    double ions[]={10,20,50},ne[]={200};
    REQUIRE(close_state(&view,ions,ne)==POP_OK);
    REQUIRE(ions[0]==20 && ions[1]==30 && ions[2]==50 && ne[0]==230 && inventory[0]==100);
    double saved_ions[3];memcpy(saved_ions,ions,sizeof(ions));double saved_ne=ne[0];
    inventory[0]=99;
    REQUIRE(close_state(&view,ions,ne)==POP_SOLVE_FAILED);
    REQUIRE(!memcmp(saved_ions,ions,sizeof(ions)) && ne[0]==saved_ne && inventory[0]==99);
    view.number_density=NULL;
    REQUIRE(close_state(&view,ions,ne)==POP_ATOMIC_MISSING);
    REQUIRE(!memcmp(saved_ions,ions,sizeof(ions)) && ne[0]==saved_ne);
    view.source=POP_ELEMENT_TOTAL_FROM_MASS;
    mass[0]=2;x[0]=0.5;rho[0]=8*AMU;
    REQUIRE(population_element_density_get(&view,2,2,0,0,mass,x,rho,AMU,&n)==POP_OK && n==2);
    if (argc==2) {
        AtomicData atom={0};int z[]={14,27};atom.n_elements=2;atom.element_Z=z;
        REQUIRE(load_element_number_density_profile(&atom,argv[1],2)==0);
        REQUIRE(atom.element_density.source==POP_ELEMENT_TOTAL_EXPLICIT_NUMBER);
        REQUIRE(atom.element_density.number_density[0]==100 &&
                atom.element_density.number_density[1]==0 &&
                atom.element_density.number_density[2]==200 &&
                atom.element_density.number_density[3]==10);
        free(atom.element_density.number_density);
    }
    puts("[ELEMENT-DENSITY][PASS] explicit inventory independent of mass/X/rho; zero retained; missing/stale/nonconserving rejected atomically; legacy unchanged");
    return 0;
}
