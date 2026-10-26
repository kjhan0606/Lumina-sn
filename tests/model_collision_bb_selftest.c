#include "nlte_model_coll_bb.h"

static int near(double actual,double expected)
{
    return isfinite(actual) && fabs(actual-expected)<=1.e-12*fmax(1.e-30,fabs(expected));
}

int main(void)
{
    int offsets[]={0,3},ion_z[]={16};
    int z[]={16,16,16},ion[]={2,2,2},g[]={2,4,6};
    int global[]={0,1,2},super[]={0,0,1},to_row[]={0,1};
    double energy[]={0.,1.,3.},w[]={1.,0.8,0.5},fraction[]={0.75,0.25,1.};
    double te[]={10000.},ne[]={1.e8};
    int32_t lower[]={0,0,1},upper[]={1,2,2};
    double omega[]={0.1,0.2,0.3};
    AtomicData atom={0};NLTEConfig nlte={0};PlasmaState plasma={0};
    atom.n_levels=3;atom.n_ion_pops=1;atom.level_offset=offsets;
    atom.ion_pop_Z=ion_z;atom.level_Z=z;atom.level_ion=ion;
    atom.level_energy_eV=energy;atom.level_g=g;
    atom.model_level_occupation=w;atom.model_occupation_n_shells=1;
    atom.population_committed_generation=1;atom.partition_stamp.status=POP_OK;
    memset(atom.partition_stamp.atomic_model_sha256,'a',64);
    nlte.model_atom_projection=1;nlte.n_nlte_levels_total=3;nlte.n_super_total=2;
    nlte.global_to_nlte_level=global;nlte.fl_to_super=super;
    nlte.within_sl_frac=fraction;nlte.population_committed_generation=1;
    plasma.n_shells=1;plasma.T_e=te;plasma.n_electron=ne;
    NLTEModelElement layout={.Z=16,.shell=0,.n_variables=2,.n_super_total=2,
        .n_ion_pops=1,.material_generation=1,.super_to_row=to_row};
    NLTEModelCollisionBBView view={.Z=16,.n_levels=3,.n_shells=1,
        .native_shell=0,.n_pairs=3,.material_generation=1,
        .temperature_K=10000.,.electron_density_cm3=1.e8,
        .lower=lower,.upper=upper,.omega=omega};
    memset(view.material_model_sha256,'a',64);
    NLTEModelCollisionBBBlock block={0};
    if(nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view) ||
       block.pairs!=3 || block.cross_super_pairs!=2 || block.intra_super_pairs!=1)return 1;
    const double kT=8.617333262145e-5*te[0];
    const double factor=8.63e-6*ne[0]/sqrt(te[0]);
    const double f02=0.75*factor*0.2*exp(-3./kT)/2.*0.5;
    const double f12=0.25*factor*0.3*exp(-2./kT)/4.*(0.5/0.8);
    const double r02=factor*0.2/6.,r12=factor*0.3/6.;
    const double ev=1.602176634e-12;
    if(!near(block.collisional[1],f02+f12) ||
       !near(block.collisional[2],r02+r12) ||
       !near(block.collisional[0],-(f02+f12)) ||
       !near(block.collisional[3],-(r02+r12)) ||
       !near(block.collisional_heat[0],ev*(3*f02+2*f12)) ||
       !near(block.collisional_heat[1],-ev*(3*r02+2*r12))){
        fprintf(stderr,"matrix %.17g %.17g %.17g %.17g expected %.17g %.17g %.17g %.17g\n",
            block.collisional[0],block.collisional[1],block.collisional[2],block.collisional[3],
            -(f02+f12),f02+f12,r02+r12,-(r02+r12));
        fprintf(stderr,"heat %.17g %.17g expected %.17g %.17g\n",
            block.collisional_heat[0],block.collisional_heat[1],
            ev*(3*f02+2*f12),-ev*(3*r02+2*r12));
        return 2;
    }
    nlte_model_collision_bb_block_free(&block);
    view.n_pairs=2;
    if(!nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view) ||
       block.collisional || block.n_variables)return 3;
    view.n_pairs=3;view.temperature_K+=1.;
    if(!nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view))return 4;
    view.temperature_K=te[0];upper[0]=2;
    if(!nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view))return 5;
    upper[0]=1;view.material_model_sha256[0]='b';
    if(!nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view))return 6;
    view.material_model_sha256[0]='a';
    atom.n_levels=1;offsets[1]=1;nlte.n_nlte_levels_total=1;
    nlte.n_super_total=1;layout.n_variables=1;layout.n_super_total=1;
    view.n_levels=1;view.n_pairs=0;view.lower=NULL;view.upper=NULL;view.omega=NULL;
    if(nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view) ||
       block.n_variables!=1 || block.pairs!=0 || block.collisional[0]!=0. ||
       block.collisional_heat[0]!=0.)return 7;
    nlte_model_collision_bb_block_free(&block);
    puts("MODEL_COLLISION_BB synthetic PASS");return 0;
}
