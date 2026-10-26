#include "nlte_model_coll_bb.h"

static int near(double actual,double expected)
{
    return isfinite(actual) && fabs(actual-expected)<=1.e-12*fmax(1.e-30,fabs(expected));
}

static int stream_cases(const NLTEModelElement *e,const NLTEConfig *n,
    const AtomicData *a,PlasmaState *p,const NLTEModelCollisionBBView *view)
{
    NLTEModelCollisionBBBlock expected={0},actual={0};
    if(nlte_model_element_collision_bb_assemble(&expected,e,n,a,p,view))return 1;
    for(size_t chunk=1;chunk<=2;chunk++){
        NLTEModelCollisionBBStream *s=NULL;
        if(nlte_model_collision_bb_stream_begin(&s,e,n,a,p,view,19))return 2;
        for(size_t at=0;at<view->n_pairs;at+=chunk){
            size_t count=view->n_pairs-at;if(count>chunk)count=chunk;
            if(nlte_model_collision_bb_stream_push(s,19,view->lower+at,view->upper+at,view->omega+at,count))return 3;
        }
        if(nlte_model_collision_bb_stream_finish(s,19,&actual) ||
           actual.pairs!=expected.pairs ||
           memcmp(actual.collisional,expected.collisional,e->n_variables*e->n_variables*sizeof(double)) ||
           memcmp(actual.collisional_heat,expected.collisional_heat,e->n_variables*sizeof(double)))return 4;
        if(view->dissolution &&
           (memcmp(actual.dissolution,expected.dissolution,e->n_variables*e->n_variables*sizeof(double)) ||
            memcmp(actual.dissolution_heat,expected.dissolution_heat,e->n_variables*sizeof(double))))return 5;
        nlte_model_collision_bb_stream_free(s);nlte_model_collision_bb_block_free(&actual);
    }
    for(int defect=0;defect<5;defect++){
        NLTEModelCollisionBBStream *s=NULL;
        if(nlte_model_collision_bb_stream_begin(&s,e,n,a,p,view,20) ||
           nlte_model_collision_bb_stream_push(s,20,view->lower,view->upper,view->omega,1))return 6;
        int rc;
        if(defect==0){ /* 미완결 */
            rc=nlte_model_collision_bb_stream_finish(s,20,&actual);
        }else if(defect==1){ /* 중복 */
            rc=nlte_model_collision_bb_stream_push(s,20,view->lower,view->upper,view->omega,1);
        }else if(defect==2){ /* 다른 후보의 신원 */
            rc=nlte_model_collision_bb_stream_push(s,21,view->lower+1,view->upper+1,view->omega+1,1);
        }else if(defect==3){ /* 블록 사이 T 변경 */
            double saved=p->T_e[e->shell];p->T_e[e->shell]+=1.;
            rc=nlte_model_collision_bb_stream_push(s,20,view->lower+1,view->upper+1,view->omega+1,1);
            p->T_e[e->shell]=saved;
        }else{ /* 좋은 앞 블록 뒤의 비유한 값 */
            double bad=NAN;
            rc=nlte_model_collision_bb_stream_push(s,20,view->lower+1,view->upper+1,&bad,1);
        }
        if(!rc || !nlte_model_collision_bb_stream_finish(s,20,&actual) ||
           actual.n_variables || actual.collisional || actual.collisional_heat)return 7;
        nlte_model_collision_bb_stream_free(s);
    }
    nlte_model_collision_bb_block_free(&expected);return 0;
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
    if(stream_cases(&layout,&nlte,&atom,&plasma,&view))return 17;
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
    /* same-SL 쌍도 다음 이온으로 용해 이온화를 전달한다. */
    int ground_row[]={2,2,2};double saha[]={.03,.01,.02},threshold[]={4.e15,3.e15,2.e15};
    NLTEModelDissolutionContext dissolution={.Z=16,.shell=0,.n_levels=3,.n_variables=3,
        .material_generation=1,.temperature_K=te[0],.electron_density_cm3=ne[0],
        .ground_row=ground_row,.saha=saha,.threshold_Hz=threshold};
    memset(dissolution.material_model_sha256,'a',64);
    view.dissolution=&dissolution;layout.n_variables=3;
    if(stream_cases(&layout,&nlte,&atom,&plasma,&view))return 18;
    if(nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view) ||
       block.dissolution_pairs!=3 || block.intra_super_dissolution_pairs!=1)return 8;
    const double q01=factor*.1*exp(-1./kT)/2.*(1.-.8);
    const double q02=factor*.2*exp(-3./kT)/2.*(1.-.5);
    const double q12=factor*.3*exp(-2./kT)/4.*(1.-.5/.8);
    if(!near(block.dissolution[2],.75*(q01+q02)+.25*q12) ||
       !near(block.dissolution[6],.03*(q01+q02)+.01*q12))return 9;
    /* LTE 행별 상세균형/열량0. 고의로 역률을 틀리면 반드시 깨진다. */
    const double lte[]={.04,.02,1.};long double heat=0.;
    for(int i=0;i<3;i++){
        long double net=0.,flow=0.;
        for(int j=0;j<3;j++){
            long double term=block.dissolution[j*3+i]*lte[j];net+=term;flow+=fabsl(term);
        }
        if(fabsl(net)>1.e-12L*flow)return 10;
        heat+=lte[i]*block.dissolution_heat[i];
    }
    const double expected_heat=6.62607015e-27*(.75*4.e15*(q01+q02)+.25*3.e15*q12);
    if(!near(block.dissolution_heat[0],expected_heat) || fabsl(heat)>1.e-28L)return 11;
    double wrong_net=block.dissolution[0]*lte[0]+1.01*block.dissolution[6]*lte[2];
    if(fabs(wrong_net)<1.e-4*fabs(block.dissolution[6]))return 12;
    nlte_model_collision_bb_block_free(&block);
    w[1]=w[2]=1.;
    if(nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view) ||
       block.dissolution_pairs)return 13;
    for(int i=0;i<9;i++)if(block.dissolution[i]!=0.)return 14;
    nlte_model_collision_bb_block_free(&block);
    w[0]=.5;w[1]=.8;
    if(!nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view))return 15;
    w[0]=1.;w[1]=.8;w[2]=.5;
    dissolution.material_generation=2;
    if(!nlte_model_element_collision_bb_assemble(&block,&layout,&nlte,&atom,&plasma,&view))return 16;
    dissolution.material_generation=1;view.dissolution=NULL;layout.n_variables=2;
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
