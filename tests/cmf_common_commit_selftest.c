/* 합성 장으로 실제 R6 게시/소비 경계를 시험한다. 별 모형의 수렴 증거가 아니다. */
#include "lumina_cmfgen.h"
#include "line_jbar.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"COMMON_COMMIT_FAIL line=%d: %s\n",__LINE__,#x); return 1; } } while(0)

static int exercise(double vdop, int bind_profile, int expect_rejected) {
    enum { NS=2, NL=3, NB=128 };
    double vi[NS]={1e8,2e8},vo[NS]={2e8,3e8};
    double nu[NL]={LINE_JBAR_BB_NU_MIN_HZ,6e14,LINE_JBAR_BB_NU_MAX_HZ};
    int map[NL]={0,-1,1}; uint8_t mask[NL]={1,0,1};
    double j[NS*NB],jb[NS*NL],err[NS*NL];
    for(int i=0;i<NS*NB;++i)j[i]=1e-6;
    for(int i=0;i<NS*NL;++i){jb[i]=(i+1)*1e-6;err[i]=1e-18;}
    LineJbarQSet q={0};LineJbarESet e={0};
    REQUIRE(line_jbar_qset_build(&q,NL,nu,map,mask)==0);
    REQUIRE(line_jbar_eset_build(&e,NL,nu)==0);
    NLTEConfig n={0};n.line_qset=&q;n.line_eset=&e;
    REQUIRE(radiation_field_owner_init(&n.radiation_field,NS)==0);
    if(bind_profile)REQUIRE(cmfgen_bind_common_profile(&n,vdop)==0);
    LineJbarQSet before_q=q;LineJbarESet before_e=e;
    REQUIRE(cmfgen_bind_common_profile(&n,2e6)!=0);
    REQUIRE(memcmp(&q,&before_q,sizeof q)==0 && memcmp(&e,&before_e,sizeof e)==0);
    Geometry g={0};g.n_shells=NS;g.v_inner=vi;g.v_outer=vo;g.time_explosion=1e6;
    CMFGENState c={0};c.n_shells=NS;c.n_bins=NB;c.J=j;
    c.nu_min=LUMINA_RADFIELD_NU_MIN_HZ;c.nu_max=LUMINA_RADFIELD_NU_MAX_HZ;
    c.d_log_nu=log(c.nu_max/c.nu_min)/NB;
    OpacityState o={0};o.n_shells=NS;o.n_lines=NL;o.line_list_nu=nu;
    o.jbar_line_det=jb;o.jbar_line_det_error_upper=err;
    o.jbar_line_det_operator=CMF_FINE_LINE_OPERATOR_COMMON_GAUSSIAN;
    o.jbar_line_det_vdoppler_cms=vdop;o.jbar_line_det_ndoppler=4;
    o.jbar_line_det_exact_converged=1;o.jbar_line_det_exact_iterations=3;
    o.jbar_line_det_exact_iteration_cap=64;o.jbar_line_det_exact_tolerance=1e-8;
    o.jbar_line_det_exact_residual=1e-12;o.jbar_line_det_exact_max_scattering_ratio=.1;
    o.jbar_line_det_error_envelope_verified=1;o.jbar_line_det_error_refinement_iterations=1;
    o.jbar_line_det_component_error_min=o.jbar_line_det_profile_error_min=1e-18;
    o.jbar_line_det_component_error_max=o.jbar_line_det_profile_error_max=1e-18;
    o.jbar_line_det_grid_n_bins=1000;
    o.jbar_line_det_grid_nu_min=c.nu_min;o.jbar_line_det_grid_nu_max=c.nu_max;
    int rc=cmfgen_commit_jnu(&c,&n,&g,&o,1);
    if(expect_rejected) {
        REQUIRE(rc!=0);
        REQUIRE(n.radiation_field.field.generation.computed_generation==0);
    } else {
        REQUIRE(rc==0 && n.line_view_status==LINE_JBAR_VIEW_OK);
        REQUIRE(n.line_energy_view_status==LINE_JBAR_VIEW_OK);
        LineJbarValue value;
        REQUIRE(line_jbar_lookup(&n.line_energy_view,1,1,&value)==0);
        REQUIRE(value.jbar==jb[3]);
        REQUIRE(n.radiation_field.line_profile_id==q.profile_id);
        REQUIRE(strcmp(n.radiation_field.line_profile_hash_storage,q.profile_hash)==0);
        REQUIRE(cmfgen_bind_common_profile(&n,vdop)!=0);
        REQUIRE(memcmp(&q,&before_q,sizeof q)==0 && memcmp(&e,&before_e,sizeof e)==0);
        o.jbar_line_det_vdoppler_cms=vdop==1e6?5e6:1e6;
        REQUIRE(cmfgen_commit_jnu(&c,&n,&g,&o,2)!=0);
        REQUIRE(n.radiation_field.field.generation.computed_generation==1);
        REQUIRE(line_jbar_lookup(&n.line_energy_view,1,1,&value)==0 && value.jbar==jb[3]);
        o.jbar_line_det_vdoppler_cms=vdop;
        q.profile_hash[0]=e.profile_hash[0]='x';
        REQUIRE(cmfgen_commit_jnu(&c,&n,&g,&o,2)!=0);
        REQUIRE(n.radiation_field.field.generation.computed_generation==1);
        q=before_q;e=before_e;
        if(vdop==5e6) {
            double narrow=4e6/2.99792458e10;
            o.jbar_line_det_grid_nu_min=nu[0]*(1-narrow);
            o.jbar_line_det_grid_nu_max=nu[2]*(1+narrow);
            REQUIRE(cmfgen_commit_jnu(&c,&n,&g,&o,2)!=0);
            REQUIRE(n.radiation_field.field.generation.computed_generation==1);
            LineJbarPacketPartial partial={0};
            REQUIRE(line_jbar_segment_add(&q,&partial,0,nu[0],nu[1],1,1,1)!=0);
        }
    }
    printf("synthetic_R6_commit vdop=%.0f rc=%d expected_rejection=%d profile_id=%llu hash=%s\n",
        vdop,rc,expect_rejected,(unsigned long long)q.profile_id,q.profile_hash);
    radiation_field_owner_free(&n.radiation_field);
    line_jbar_qset_free(&q);line_jbar_qset_free(&e);
    return 0;
}

int main(void) {
    if(exercise(1e6,1,0)||exercise(5e6,1,0)||exercise(5e6,0,1))return 1;
    puts("COMMON_PROFILE_COMMIT_SELFTEST_PASS");
    return 0;
}
