#define _POSIX_C_SOURCE 200809L

#include "physics_comparison.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int failures;
#define CHECK(expr,name) do { if (!(expr)) { \
    fprintf(stderr,"PHYSICS_COMPARISON_SELFTEST_FAIL %s line=%d\n",name,__LINE__); \
    ++failures; } } while (0)

static void fill_hash(char value[65], char digit)
{
    memset(value,digit,64);
    value[64]='\0';
}

static int file_has(const char *path, const char *needle)
{
    FILE *stream=fopen(path,"r");
    if(!stream) return 0;
    char line[4096]; int found=0;
    while(fgets(line,sizeof(line),stream))
        if(strstr(line,needle)){found=1;break;}
    fclose(stream); return found;
}

static int diagnostic_reason_equals(const char *diagnostic,
                                    const char *expected)
{
    const char prefix[] = "[PHYSICS_COMPARISON][BLOCKED] reason=";
    const char *start = strstr(diagnostic, prefix);
    size_t length;

    if (!start)
        return 0;
    start += strlen(prefix);
    length = strcspn(start, " \r\n");
    return length == strlen(expected) && strncmp(start, expected, length) == 0;
}

static PhysicsComparisonStatus capture_snapshot_diagnostic(
        const char *output_directory,
        const PhysicsComparisonSnapshotInput *input,
        char *diagnostic, size_t diagnostic_capacity)
{
    FILE *capture = tmpfile();
    int stderr_fd, saved_stderr_fd;
    PhysicsComparisonStatus status;

    if (!capture || diagnostic_capacity == 0) {
        if (capture)
            fclose(capture);
        if (diagnostic_capacity)
            diagnostic[0] = '\0';
        return (PhysicsComparisonStatus)-1;
    }
    stderr_fd = fileno(stderr);
    saved_stderr_fd = dup(stderr_fd);
    if (saved_stderr_fd < 0 || dup2(fileno(capture), stderr_fd) < 0) {
        if (saved_stderr_fd >= 0)
            close(saved_stderr_fd);
        fclose(capture);
        diagnostic[0] = '\0';
        return (PhysicsComparisonStatus)-1;
    }

    status = physics_comparison_snapshot_write(output_directory, input);
    fflush(stderr);
    if (dup2(saved_stderr_fd, stderr_fd) < 0)
        status = (PhysicsComparisonStatus)-1;
    close(saved_stderr_fd);

    rewind(capture);
    diagnostic[fread(diagnostic, 1, diagnostic_capacity - 1, capture)] = '\0';
    fclose(capture);
    return status;
}

static PhysicsComparisonStatus capture_dump_diagnostic(
        const char *lane, int iteration, const Geometry *geometry,
        const AtomicData *atom, const PlasmaState *plasma,
        const OpacityState *opacity, const NLTEConfig *nlte,
        char *diagnostic, size_t diagnostic_capacity)
{
    FILE *capture = tmpfile();
    int stderr_fd, saved_stderr_fd;
    PhysicsComparisonStatus status;

    if (!capture || diagnostic_capacity == 0) {
        if (capture)
            fclose(capture);
        if (diagnostic_capacity)
            diagnostic[0] = '\0';
        return (PhysicsComparisonStatus)-1;
    }
    stderr_fd = fileno(stderr);
    saved_stderr_fd = dup(stderr_fd);
    if (saved_stderr_fd < 0 || dup2(fileno(capture), stderr_fd) < 0) {
        if (saved_stderr_fd >= 0)
            close(saved_stderr_fd);
        fclose(capture);
        diagnostic[0] = '\0';
        return (PhysicsComparisonStatus)-1;
    }

    status = physics_comparison_dump_if_requested(
        lane, iteration, geometry, atom, plasma, opacity, nlte);
    fflush(stderr);
    if (dup2(saved_stderr_fd, stderr_fd) < 0)
        status = (PhysicsComparisonStatus)-1;
    close(saved_stderr_fd);

    rewind(capture);
    diagnostic[fread(diagnostic, 1, diagnostic_capacity - 1, capture)] = '\0';
    fclose(capture);
    return status;
}

int main(void)
{
    enum { NS=2, NB=2, RNB=4, CELLS=NS*NB };
    char directory[256];
    snprintf(directory,sizeof(directory),"/tmp/lumina-physics-compare-%ld",
             (long)getpid());

    double ri[NS]={10,20},ro[NS]={20,30};
    double vi[NS]={1,2},vo[NS]={2,3};
    double te_value[NS]={5000,6000},ne[NS]={10,20};
    double natom[NS]={100,200},uatom[NS]={1.0e-12,2.0e-12};
    double target_edge[NB+1]={1,2,4};
    double rf_edge[RNB+1]={1,1.5,2,3,4};
    double rf_j[NS*RNB]={2,4,6,8,10,12,14,16};
    RadiationFieldValidityState rf_valid[NS*RNB];
    for(size_t i=0;i<NS*RNB;i++)rf_valid[i]=RADIATION_FIELD_VALID;
    RadiationFieldView radiation={NS,RNB,rf_edge,rf_j,rf_valid,NULL,9};

    double es[CELLS]={1,1,1,1},bb[CELLS]={2,2,2,2};
    double bf[CELLS]={3,3,3,3},ff[CELLS]={4,4,4,4};
    double chit[CELLS]={10,10,10,10};
    A208Validity chi_status[4*CELLS];
    for(size_t i=0;i<4*CELLS;i++)chi_status[i]=A208_VALID;
    CpuOpacityPublication op={0};
    op.generation_required=op.generation_committed=11;
    op.radiation_generation=9;op.population_generation=7;op.te_generation=5;
    op.n_shells=NS;op.n_bins=NB;op.frequency_edges=target_edge;
    op.chi_es=es;op.chi_bb=bb;op.chi_bf=bf;op.chi_ff=ff;
    op.chi_total=chit;op.chi_validity=chi_status;

    double ebb[CELLS]={1,1,1,1},ebf[CELLS]={2,2,2,2};
    double eff[CELLS]={3,3,3,3},et[CELLS]={6,6,6,6};
    EmissivityStatus cell_status[CELLS],component_status[5*CELLS];
    for(size_t i=0;i<CELLS;i++)cell_status[i]=EMISS_OK;
    for(size_t i=0;i<5*CELLS;i++)component_status[i]=EMISS_EXACT_ZERO;
    for(size_t c=0;c<3;c++)for(size_t i=0;i<CELLS;i++)
        component_status[c*CELLS+i]=EMISS_OK;
    CpuEmissivityPublication em={0};
    em.required_emissivity_generation=em.committed_emissivity_generation=11;
    em.radfield_generation=9;em.population_generation=7;
    em.opacity_generation=11;em.te_generation=5;
    em.n_shells=NS;em.n_bins=NB;em.nu_edge=target_edge;
    em.eta_bb=ebb;em.eta_bf=ebf;em.eta_ff=eff;em.eta_true_total=et;
    em.cell_status=cell_status;em.component_status=component_status;
    CHECK(a209_grid_manifest_sha256(target_edge,NB,
                                    em.grid_manifest_sha256)==0,
          "emissivity-grid-hash");

    A210TermLedger ledger[NS];memset(ledger,0,sizeof(ledger));
    RadeqStatus shell_status[NS]={RADEQ_OK,RADEQ_OK};
    RadeqStatus residual_status[NS]={RADEQ_OK,RADEQ_OK};
    for(size_t s=0;s<NS;s++){
        ledger[s].adiabatic_model=A210_ADIABATIC_CMFGEN_COMPLETE;
        ledger[s].adiabatic_temperature_gradient=0.1;
        ledger[s].adiabatic_velocity_divergence=0.2;
        ledger[s].adiabatic_electron_fraction_gradient=-0.1;
        ledger[s].adiabatic_internal_energy_gradient=0.4;
        ledger[s].adiabatic_signed_total=0.6;
        ledger[s].heating[A210_PHOTO]=2.0;
        ledger[s].heating[A210_LINE_ABS]=1.0;
        ledger[s].cooling[A210_RECOMB]=1.0;
        ledger[s].cooling[A210_ADIABATIC]=0.6;
        ledger[s].sum_heating=3.0;ledger[s].sum_cooling=1.6;
        ledger[s].residual=1.4;
    }
    ElectronTemperaturePublication tep={0};
    tep.te_lane=A210_TE_LANE_FREE_T;
    tep.re_root_required=1;
    tep.required_te_generation=tep.committed_te_generation=5;
    tep.radfield_generation=9;tep.population_generation=7;
    tep.opacity_generation=11;tep.emissivity_generation=11;tep.n_shells=NS;
    tep.ledger=ledger;tep.shell_status=shell_status;
    tep.residual_status=residual_status;
    fill_hash(tep.atomic_model_sha256,'a');
    fill_hash(tep.te_manifest_sha256,'c');
    double velocity_edges[NS+1]={1,2,3};
    CHECK(a210_geometry_sha256(velocity_edges,NS+1,tep.geometry_sha256)==RADEQ_OK,
          "geometry-hash");

    PhysicsComparisonSnapshotInput input={
        "DET",0,10,NS,ri,ro,vi,vo,te_value,ne,natom,uatom,
        &radiation,&op,&em,&tep
    };
    CHECK(physics_comparison_snapshot_write(directory,&input)==
          PHYSICS_COMPARISON_OK,"positive-write");
    char manifest[512],shell[512],spectral[512];
    snprintf(manifest,sizeof(manifest),"%s/physics_DET_iter0000.manifest.json",directory);
    snprintf(shell,sizeof(shell),"%s/physics_DET_iter0000.shell.csv",directory);
    snprintf(spectral,sizeof(spectral),"%s/physics_DET_iter0000.spectral.csv",directory);
    CHECK(file_has(manifest,"\"transaction_status\": \"COMMITTED\""),
          "manifest-commit");
    CHECK(file_has(manifest,"\"te_lane\": \"FREE_T\""),
          "manifest-te-lane");
    CHECK(file_has(manifest,"\"radiative_integral_factor\": 12.566370614359172"),
          "manifest-four-pi");
    CHECK(file_has(shell,"q_ad_signed_total"),"shell-schema");
    CHECK(file_has(spectral,"0,0,1,2,3,"),"integral-rebin-known-answer");

    em.te_generation=6;
    input.iteration=1;
    CHECK(physics_comparison_snapshot_write(directory,&input)==
          PHYSICS_COMPARISON_STALE_GENERATION,"generation-negative");
    char missing[512];
    snprintf(missing,sizeof(missing),"%s/physics_DET_iter0001.manifest.json",directory);
    CHECK(access(missing,F_OK)!=0,"generation-no-manifest");
    em.te_generation=5;
    ledger[0].adiabatic_signed_total=-0.6;
    input.iteration=2;
    CHECK(physics_comparison_snapshot_write(directory,&input)==
          PHYSICS_COMPARISON_INVALID_VALUE,"sign-negative");
    snprintf(missing,sizeof(missing),"%s/physics_DET_iter0002.manifest.json",directory);
    CHECK(access(missing,F_OK)!=0,"sign-no-manifest");

    /* M5 negative controls: invalid temperature publication metadata must
     * remove the staged manifest, shell, and spectral files. */
    ledger[0].adiabatic_signed_total=0.6;
    tep.te_lane=A210_TE_LANE_UNSET;
    input.iteration=3;
    CHECK(physics_comparison_snapshot_write(directory,&input)==
          PHYSICS_COMPARISON_IO_ERROR,"unset-lane-negative");
    snprintf(manifest,sizeof(manifest),"%s/physics_DET_iter0003.manifest.json",directory);
    snprintf(shell,sizeof(shell),"%s/physics_DET_iter0003.shell.csv",directory);
    snprintf(spectral,sizeof(spectral),"%s/physics_DET_iter0003.spectral.csv",directory);
    CHECK(access(manifest,F_OK)!=0,"unset-lane-no-manifest");
    CHECK(access(shell,F_OK)!=0,"unset-lane-no-shell");
    CHECK(access(spectral,F_OK)!=0,"unset-lane-no-spectral");

    tep.te_lane=A210_TE_LANE_FREE_T;
    tep.pinned_shells=NS;
    input.iteration=4;
    CHECK(physics_comparison_snapshot_write(directory,&input)==
          PHYSICS_COMPARISON_IO_ERROR,"free-t-fixed-field-leak-negative");
    snprintf(manifest,sizeof(manifest),"%s/physics_DET_iter0004.manifest.json",directory);
    snprintf(shell,sizeof(shell),"%s/physics_DET_iter0004.shell.csv",directory);
    snprintf(spectral,sizeof(spectral),"%s/physics_DET_iter0004.spectral.csv",directory);
    CHECK(access(manifest,F_OK)!=0,"free-t-fixed-field-leak-no-manifest");
    CHECK(access(shell,F_OK)!=0,"free-t-fixed-field-leak-no-shell");
    CHECK(access(spectral,F_OK)!=0,"free-t-fixed-field-leak-no-spectral");

    /* P-4: exercise every INVALID_ARGUMENT guard site.  The assertion checks
     * the exact reason token captured from stderr; no aggregate OR predicate
     * can make a wrong reason pass. */
    {
        char diagnostic[4096];
        PhysicsComparisonStatus status;
        Geometry geometry = {0};
        AtomicData atom_state = {0};
        PlasmaState plasma_state = {0};
        OpacityState opacity_state = {0};
        NLTEConfig nlte_state = {0};

        input.lane = NULL;
        status = capture_snapshot_diagnostic(
            directory, &input, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-site-99-status");
        CHECK(diagnostic_reason_equals(
                  diagnostic, "COMPARISON_INPUT_INVALID"),
              "p4-site-99-reason");

        input.lane = "DET";
        op.frequency_edges = NULL;
        status = capture_snapshot_diagnostic(
            directory, &input, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-site-112-status");
        CHECK(diagnostic_reason_equals(
                  diagnostic, "COMPARISON_PUBLICATION_LAYOUT_INVALID"),
              "p4-site-112-reason");
        op.frequency_edges = target_edge;

        char saved_grid_hash_digit = em.grid_manifest_sha256[0];
        em.grid_manifest_sha256[0] = '!';
        status = capture_snapshot_diagnostic(
            directory, &input, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-site-133-status");
        CHECK(diagnostic_reason_equals(diagnostic, "COMPARISON_HASH_INVALID"),
              "p4-site-133-reason");
        em.grid_manifest_sha256[0] = saved_grid_hash_digit;

        status = capture_snapshot_diagnostic(
            NULL, &input, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-site-255-status");
        CHECK(diagnostic_reason_equals(diagnostic, "SNAPSHOT_INPUT_MISSING"),
              "p4-site-255-reason");

        op.n_bins = 0;
        status = capture_snapshot_diagnostic(
            directory, &input, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-site-258-status");
        CHECK(diagnostic_reason_equals(
                  diagnostic, "SNAPSHOT_BIN_OR_SHELL_INVALID"),
              "p4-site-258-reason");
        op.n_bins = NB;

        geometry.n_shells = 2;
        plasma_state.n_shells = 2;
        setenv("LUMINA_PHYSICS_COMPARISON_DIR", directory, 1);

        status = capture_dump_diagnostic(
            "DET", 0, NULL, &atom_state, &plasma_state, &opacity_state,
            &nlte_state, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-dump-geometry-status");
        CHECK(diagnostic_reason_equals(diagnostic, "DUMP_GEOMETRY_MISSING"),
              "p4-dump-geometry-reason");

        status = capture_dump_diagnostic(
            "DET", 0, &geometry, NULL, &plasma_state, &opacity_state,
            &nlte_state, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-dump-atom-status");
        CHECK(diagnostic_reason_equals(diagnostic, "DUMP_ATOM_MISSING"),
              "p4-dump-atom-reason");

        status = capture_dump_diagnostic(
            "DET", 0, &geometry, &atom_state, NULL, &opacity_state,
            &nlte_state, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-dump-plasma-status");
        CHECK(diagnostic_reason_equals(diagnostic, "DUMP_PLASMA_MISSING"),
              "p4-dump-plasma-reason");

        status = capture_dump_diagnostic(
            "DET", 0, &geometry, &atom_state, &plasma_state, NULL,
            &nlte_state, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-dump-opacity-status");
        CHECK(diagnostic_reason_equals(diagnostic, "DUMP_OPACITY_MISSING"),
              "p4-dump-opacity-reason");

        status = capture_dump_diagnostic(
            "DET", 0, &geometry, &atom_state, &plasma_state, &opacity_state,
            NULL, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-dump-nlte-status");
        CHECK(diagnostic_reason_equals(diagnostic, "DUMP_NLTE_MISSING"),
              "p4-dump-nlte-reason");

        geometry.n_shells = 1;
        plasma_state.n_shells = 1;
        status = capture_dump_diagnostic(
            "DET", 0, &geometry, &atom_state, &plasma_state, &opacity_state,
            &nlte_state, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-dump-small-shell-status");
        CHECK(diagnostic_reason_equals(
                  diagnostic, "DUMP_SHELL_COUNT_TOO_SMALL"),
              "p4-dump-small-shell-reason");

        geometry.n_shells = 2;
        plasma_state.n_shells = 3;
        status = capture_dump_diagnostic(
            "DET", 0, &geometry, &atom_state, &plasma_state, &opacity_state,
            &nlte_state, diagnostic, sizeof(diagnostic));
        CHECK(status == PHYSICS_COMPARISON_INVALID_ARGUMENT,
              "p4-dump-mismatch-status");
        CHECK(diagnostic_reason_equals(
                  diagnostic, "DUMP_SHELL_COUNT_MISMATCH"),
              "p4-dump-mismatch-reason");

        unsetenv("LUMINA_PHYSICS_COMPARISON_DIR");
    }

    remove(manifest);remove(shell);remove(spectral);rmdir(directory);
    if(failures)return 1;
    printf("PHYSICS_COMPARISON_SELFTEST PASS generation=ATOMIC rebin=INTEGRAL "
           "four_pi=DECLARED adiabatic_sign=CHECKED\n");
    return 0;
}
