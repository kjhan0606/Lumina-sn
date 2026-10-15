/* 실제 원자/선택 격자의 수치 재시작 I/O. J는 합성 추정치이고 SN 수렴 시험이 아니다. */
#include "cmf_toy_model.h"
#include <stdio.h>
#include <unistd.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"ITERATE_FAIL line=%d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
int main(int argc,char **argv){
    if(argc!=4||atoi(argv[3])!=4)return 2;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    const char *id="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const char *wrong="1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    CMFToyModel m={0};CMFPhysicalTimeState history={0};double *saved=NULL;int rc=0,iteration=-1;
    {
        const char *setting=getenv("LUMINA_TOY_GRID_T_RATIO");
        char *saved_setting=setting?strdup(setting):NULL;
        CHECK(!setting||saved_setting);
        const char *invalid[]={"","1","-2","1.05junk","nan","inf"};
        for(int i=0;i<6;i++){
            CHECK(!setenv("LUMINA_TOY_GRID_T_RATIO",invalid[i],1));
            CHECK(cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],4,mass));
            CHECK(!m.prepared&&!m.shell_storage&&!m.ejecta.storage);
        }
        if(saved_setting){CHECK(!setenv("LUMINA_TOY_GRID_T_RATIO",saved_setting,1));free(saved_setting);}
        else CHECK(!unsetenv("LUMINA_TOY_GRID_T_RATIO"));
    }
    CHECK(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],4,mass));
    size_t ns=m.geometry.n_shells,jcells=3*ns;
    m.opacity.n_fine=3;m.opacity.nu_lo_fine=1e14;m.opacity.dlognu_fine=.1;
    m.opacity.jnu_fine=malloc(jcells*sizeof(double));CHECK(m.opacity.jnu_fine);
    for(size_t i=0;i<jcells;i++)m.opacity.jnu_fine[i]=i*1e-7;
    size_t nlevels=ns*m.nlte.n_nlte_levels_total,nions=ns*m.atom.n_ion_pops;
    saved=malloc((ns+nlevels+nions+jcells)*sizeof(double));CHECK(saved);
    memcpy(saved,m.plasma.n_electron,ns*8);memcpy(saved+ns,m.atom.ion_number_density,nions*8);
    memcpy(saved+ns+nions,m.nlte.nlte_level_populations,nlevels*8);
    memcpy(saved+ns+nions+nlevels,m.opacity.jnu_fine,jcells*8);
    CHECK(!cmf_toy_initial_iterate_write(&m,"iterate.bin",id,7));
    CHECK(cmf_toy_initial_iterate_write(&m,"iterate.bin",id,8)); /* 기존 파일 보존 */
    CHECK(cmf_physical_time_state_read(&history,"iterate.bin",&m.nlte,&m.atom,&m.geometry));
    CHECK(!history.material.storage&&!history.radiation_storage);
    free(m.opacity.jnu_fine);m.opacity.jnu_fine=NULL;
    CHECK(cmf_toy_initial_iterate_read(&m,"iterate.bin",wrong,&iteration));
    CHECK(m.prepared&&iteration==-1&&!m.opacity.jnu_fine);
    m.plasma.T_e[0]*=2;
    CHECK(cmf_toy_initial_iterate_read(&m,"iterate.bin",id,&iteration));
    m.plasma.T_e[0]*=.5;
    CHECK(m.prepared&&iteration==-1&&!m.opacity.jnu_fine);
    {
        double boundary=m.geometry.r_outer[0];m.geometry.r_outer[0]*=1.001;
        CHECK(cmf_toy_initial_iterate_read(&m,"iterate.bin",id,&iteration));
        CHECK(m.prepared&&iteration==-1&&!m.opacity.jnu_fine);
        m.geometry.r_outer[0]=boundary;
    }
    /* 누락 꼬리도 거부하고 아직 소유자를 변경하지 않는다. */
    {
        FILE *in=fopen("iterate.bin","rb"),*out=fopen("truncated.bin","wbx");CHECK(in&&out);
        unsigned char b[4096];size_t count;long length;
        CHECK(!fseek(in,0,SEEK_END)&&(length=ftell(in))>8&&!fseek(in,0,SEEK_SET));
        long left=length-8;
        while(left>0){count=(size_t)left<sizeof b?(size_t)left:sizeof b;
            CHECK(fread(b,1,count,in)==count&&fwrite(b,1,count,out)==count);left-=(long)count;}
        CHECK(!fclose(in)&&!fclose(out));
    }
    CHECK(cmf_toy_initial_iterate_read(&m,"truncated.bin",id,&iteration));
    CHECK(m.prepared&&iteration==-1&&!m.opacity.jnu_fine);
    CHECK(cmf_toy_initial_iterate_start(&m,"iterate.bin",wrong,NULL,&iteration));
    CHECK(cmf_toy_initial_iterate_start(&m,"iterate.bin",wrong,wrong,&iteration));
    CHECK(cmf_toy_initial_iterate_start(&m,"iterate.bin","invalid",id,&iteration));
    CHECK(cmf_toy_initial_iterate_start(&m,"iterate.bin",wrong,"invalid",&iteration));
    CHECK(cmf_toy_initial_iterate_start(&m,"truncated.bin",wrong,id,&iteration));
    CHECK(m.prepared&&iteration==-1&&!m.opacity.jnu_fine);
    memset(m.nlte.nlte_level_populations,0,nlevels*8);m.plasma.n_electron[0]*=.5;
    CHECK(!cmf_toy_initial_iterate_start(&m,"iterate.bin",id,NULL,&iteration));
    CHECK(iteration==7&&m.prepared&&!m.plasma.physical_history&&m.plasma.initial_steady_nlte);
    CHECK(!m.nlte.radiation_field.field.generation.computed_generation);
    CHECK(!memcmp(saved,m.plasma.n_electron,ns*8)&&!memcmp(saved+ns,m.atom.ion_number_density,nions*8));
    CHECK(!memcmp(saved+ns+nions,m.nlte.nlte_level_populations,nlevels*8));
    CHECK(!memcmp(saved+ns+nions+nlevels,m.opacity.jnu_fine,jcells*8));
    CHECK(!tau_sobolev_assert_fresh(&m.opacity,"restart selftest"));
    CHECK(cmf_toy_initial_iterate_read(&m,"iterate.bin",id,&iteration)); /* 게시 전에도 재복원 금지 */
    /* 이 시험의 정적 원자 bootstrap은 프로세스당 한 번이다. 아직 복사장을
     * 게시하지 않은 동일 소유자의 J 추정만 비우고 별도 읽기 경로를 검사한다. */
    free(m.opacity.jnu_fine);m.opacity.jnu_fine=NULL;
    iteration=-1;
    CHECK(!cmf_toy_initial_iterate_start(&m,"iterate.bin",wrong,id,&iteration));
    CHECK(iteration==0&&m.prepared&&!m.plasma.physical_history&&m.plasma.initial_steady_nlte);
    CHECK(!m.nlte.radiation_field.field.generation.computed_generation);
    CHECK(!memcmp(saved,m.plasma.n_electron,ns*8)&&!memcmp(saved+ns,m.atom.ion_number_density,nions*8));
    CHECK(!memcmp(saved+ns+nions,m.nlte.nlte_level_populations,nlevels*8));
    CHECK(!memcmp(saved+ns+nions+nlevels,m.opacity.jnu_fine,jcells*8));
    CHECK(!cmf_toy_initial_iterate_write(&m,"new_guess.bin",wrong,0));
    free(m.opacity.jnu_fine);m.opacity.jnu_fine=NULL;
    CHECK(cmf_toy_initial_iterate_read(&m,"new_guess.bin",id,&iteration));
    CHECK(!cmf_toy_initial_iterate_read(&m,"iterate.bin",id,&iteration));
    CHECK(iteration==7); /* 원본의 신원/반복 횟수는 바뀌지 않는다. */
    puts("INITIAL_GUESS_PASS distinct_ids=1 new_iteration_zero=1 original_iteration_seven=1 old_id_rejected_for_new_output=1 physical_history=0 radiation_unpublished=1");
    printf("CMF_INITIAL_ITERATE_PASS shells=%zu actual_atomic_material=1 J_synthetic=1 byte_roundtrip=1 identity_T_geometry_truncation_rejected=1 invalid_grid_rejected=1 physical_history_rejected=1 physical_convergence=0\n",ns);
done:
    free(saved);cmf_physical_time_state_free(&history);cmf_toy_model_free(&m);return rc;
}
