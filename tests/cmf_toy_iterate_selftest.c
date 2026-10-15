/* 실제 원자4셸의 수치 재시작 I/O. J는 합성 추정치이고 SN 수렴 시험이 아니다. */
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
    CHECK(!cmf_toy_model_prepare(&m,argv[1],argv[2],4,mass));
    m.opacity.n_fine=3;m.opacity.nu_lo_fine=1e14;m.opacity.dlognu_fine=.1;
    m.opacity.jnu_fine=malloc(12*sizeof(double));CHECK(m.opacity.jnu_fine);
    for(int i=0;i<12;i++)m.opacity.jnu_fine[i]=i*1e-7;
    size_t nlevels=(size_t)4*m.nlte.n_nlte_levels_total,nions=(size_t)4*m.atom.n_ion_pops;
    saved=malloc((4+nlevels+nions+12)*sizeof(double));CHECK(saved);
    memcpy(saved,m.plasma.n_electron,4*8);memcpy(saved+4,m.atom.ion_number_density,nions*8);
    memcpy(saved+4+nions,m.nlte.nlte_level_populations,nlevels*8);
    memcpy(saved+4+nions+nlevels,m.opacity.jnu_fine,12*8);
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
    memset(m.nlte.nlte_level_populations,0,nlevels*8);m.plasma.n_electron[0]*=.5;
    CHECK(!cmf_toy_initial_iterate_read(&m,"iterate.bin",id,&iteration));
    CHECK(iteration==7&&m.prepared&&!m.plasma.physical_history&&m.plasma.initial_steady_nlte);
    CHECK(!m.nlte.radiation_field.field.generation.computed_generation);
    CHECK(!memcmp(saved,m.plasma.n_electron,4*8)&&!memcmp(saved+4,m.atom.ion_number_density,nions*8));
    CHECK(!memcmp(saved+4+nions,m.nlte.nlte_level_populations,nlevels*8));
    CHECK(!memcmp(saved+4+nions+nlevels,m.opacity.jnu_fine,12*8));
    CHECK(!tau_sobolev_assert_fresh(&m.opacity,"restart selftest"));
    CHECK(cmf_toy_initial_iterate_read(&m,"iterate.bin",id,&iteration)); /* 게시 전에도 재복원 금지 */
    puts("CMF_INITIAL_ITERATE_PASS actual_atomic_material=1 J_synthetic=1 byte_roundtrip=1 identity_T_truncation_rejected=1 physical_history_rejected=1 physical_convergence=0");
done:
    free(saved);cmf_physical_time_state_free(&history);cmf_toy_model_free(&m);return rc;
}
