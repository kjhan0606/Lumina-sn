#include "lumina.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_asset(const char *path, double nu_min, int trailing)
{
    FILE *stream = fopen(path, "wb");
    uint32_t magic = 0x434D4644u, version = 1u;
    int32_t n_levels = 1, n_freq = NLTE_N_FREQ_BINS;
    int8_t flag = 1;
    unsigned char pad[7] = {0};
    double sigma = 1.0;
    if (!stream) return -1;
    if (fwrite(&magic, sizeof(magic), 1, stream) != 1 ||
        fwrite(&version, sizeof(version), 1, stream) != 1 ||
        fwrite(&n_levels, sizeof(n_levels), 1, stream) != 1 ||
        fwrite(&n_freq, sizeof(n_freq), 1, stream) != 1 ||
        fwrite(&nu_min, sizeof(nu_min), 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MAX}, sizeof(double), 1, stream) != 1 ||
        fwrite(&flag, sizeof(flag), 1, stream) != 1 ||
        fwrite(pad, sizeof(pad), 1, stream) != 1) {
        fclose(stream);
        return -1;
    }
    for (int b = 0; b < NLTE_N_FREQ_BINS; ++b)
        if (fwrite(&sigma, sizeof(sigma), 1, stream) != 1) {
            fclose(stream);
            return -1;
        }
    if (trailing) fputc(0, stream);
    return fclose(stream);
}

static void clear_asset(AtomicData *atom)
{
    free(atom->cmfgen_has_sigma);
    free(atom->cmfgen_sigma_bf);
    free(atom->cmfgen_bf_channel_offset);
    free(atom->cmfgen_bf_channel_target);
    free(atom->cmfgen_bf_channel_probability);
    free(atom->cmfgen_bf_channel_phot_grid);
    free(atom->cmfgen_bf_channel_target_energy_eV);
    free(atom->cmfgen_bf_channel_target_g);
    free(atom->cmfgen_bf_channel_sigma_edge);
    free(atom->cmfgen_bf_channel_sigma);
    atom->cmfgen_has_sigma = NULL;
    atom->cmfgen_sigma_bf = NULL;
    atom->cmfgen_bf_channel_offset = NULL;
    atom->cmfgen_bf_channel_target = NULL;
    atom->cmfgen_bf_channel_probability = NULL;
    atom->cmfgen_bf_channel_phot_grid = NULL;
    atom->cmfgen_bf_channel_target_energy_eV = NULL;
    atom->cmfgen_bf_channel_target_g = NULL;
    atom->cmfgen_bf_channel_sigma_edge = NULL;
    atom->cmfgen_bf_channel_sigma = NULL;
    atom->cmfgen_bf_channel_version = 0;
    memset(atom->cmfgen_bf_recombination_policy,0,sizeof(atom->cmfgen_bf_recombination_policy));
    memset(atom->cmfgen_bf_policy_model_sha256,0,sizeof(atom->cmfgen_bf_policy_model_sha256));
    atom->cmfgen_loaded = 0;
    atom->cmfgen_bf_channels_loaded = 0;
}

static int write_two_level_sigma(const char *path)
{
    FILE *stream = fopen(path, "wb");
    uint32_t magic = 0x434D4644u, version = 1u;
    int32_t n_levels = 2, n_freq = NLTE_N_FREQ_BINS;
    int8_t flags[2] = {1, 0};
    unsigned char pad[6] = {0};
    double lower_sigma = 2.0, upper_sigma = 0.0;
    if (!stream) return -1;
    if (fwrite(&magic, 4, 1, stream) != 1 ||
        fwrite(&version, 4, 1, stream) != 1 ||
        fwrite(&n_levels, 4, 1, stream) != 1 ||
        fwrite(&n_freq, 4, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MIN}, 8, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MAX}, 8, 1, stream) != 1 ||
        fwrite(flags, sizeof(flags), 1, stream) != 1 ||
        fwrite(pad, sizeof(pad), 1, stream) != 1) {
        fclose(stream);
        return -1;
    }
    for (int b = 0; b < NLTE_N_FREQ_BINS; ++b)
        if (fwrite(&lower_sigma, 8, 1, stream) != 1) { fclose(stream); return -1; }
    for (int b = 0; b < NLTE_N_FREQ_BINS; ++b)
        if (fwrite(&upper_sigma, 8, 1, stream) != 1) { fclose(stream); return -1; }
    return fclose(stream);
}

static int write_channel_asset(const char *path, double probability, int trailing)
{
    FILE *stream = fopen(path, "wb");
    uint32_t magic = 0x434D4643u, version = 1u;
    int32_t n_levels = 2, n_freq = NLTE_N_FREQ_BINS, n_routes = 1;
    int32_t offsets[3] = {0, 1, 1}, targets[1] = {1}, grids[1] = {0};
    double sigma = 2.0;
    if (!stream) return -1;
    if (fwrite(&magic, 4, 1, stream) != 1 ||
        fwrite(&version, 4, 1, stream) != 1 ||
        fwrite(&n_levels, 4, 1, stream) != 1 ||
        fwrite(&n_freq, 4, 1, stream) != 1 ||
        fwrite(&n_routes, 4, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MIN}, 8, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MAX}, 8, 1, stream) != 1 ||
        fwrite(offsets, sizeof(int32_t), 3, stream) != 3 ||
        fwrite(targets, sizeof(int32_t), 1, stream) != 1 ||
        fwrite(&probability, sizeof(double), 1, stream) != 1 ||
        fwrite(grids, sizeof(int32_t), 1, stream) != 1) {
        fclose(stream);
        return -1;
    }
    for (int b = 0; b < NLTE_N_FREQ_BINS; ++b)
        if (fwrite(&sigma, 8, 1, stream) != 1) { fclose(stream); return -1; }
    if (trailing) fputc(0, stream);
    return fclose(stream);
}

static int write_terminal_channel_asset(const char *path,
                                        double target_energy, double target_g)
{
    FILE *stream = fopen(path, "wb");
    uint32_t magic = 0x434D4643u, version = 2u;
    int32_t n_levels = 2, n_freq = NLTE_N_FREQ_BINS, n_routes = 1;
    int32_t offsets[3] = {0, 1, 1}, target = -1, grid = 0;
    double probability = 1.0, sigma = 2.0;
    if (!stream) return -1;
    if (fwrite(&magic, 4, 1, stream) != 1 ||
        fwrite(&version, 4, 1, stream) != 1 ||
        fwrite(&n_levels, 4, 1, stream) != 1 ||
        fwrite(&n_freq, 4, 1, stream) != 1 ||
        fwrite(&n_routes, 4, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MIN}, 8, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MAX}, 8, 1, stream) != 1 ||
        fwrite(offsets, sizeof(int32_t), 3, stream) != 3 ||
        fwrite(&target, sizeof(target), 1, stream) != 1 ||
        fwrite(&probability, sizeof(probability), 1, stream) != 1 ||
        fwrite(&grid, sizeof(grid), 1, stream) != 1 ||
        fwrite(&target_energy, sizeof(target_energy), 1, stream) != 1 ||
        fwrite(&target_g, sizeof(target_g), 1, stream) != 1) {
        fclose(stream);
        return -1;
    }
    for (int b = 0; b < NLTE_N_FREQ_BINS; ++b)
        if (fwrite(&sigma, sizeof(sigma), 1, stream) != 1) {
            fclose(stream);
            return -1;
        }
    return fclose(stream);
}

static int write_edge_channel_asset(const char *path, double sigma_edge)
{
    FILE *stream = fopen(path, "wb");
    uint32_t magic = 0x434D4643u, version = 3u;
    int32_t n_levels = 2, n_freq = NLTE_N_FREQ_BINS, n_routes = 1;
    int32_t offsets[3] = {0, 1, 1}, target = 1, grid = 0;
    double probability = 1.0, target_energy = 0.0, target_g = 1.0;
    double sigma = 2.0;
    if (!stream) return -1;
    if (fwrite(&magic, 4, 1, stream) != 1 ||
        fwrite(&version, 4, 1, stream) != 1 ||
        fwrite(&n_levels, 4, 1, stream) != 1 ||
        fwrite(&n_freq, 4, 1, stream) != 1 ||
        fwrite(&n_routes, 4, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MIN}, 8, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MAX}, 8, 1, stream) != 1 ||
        fwrite(offsets, sizeof(int32_t), 3, stream) != 3 ||
        fwrite(&target, sizeof(target), 1, stream) != 1 ||
        fwrite(&probability, sizeof(probability), 1, stream) != 1 ||
        fwrite(&grid, sizeof(grid), 1, stream) != 1 ||
        fwrite(&target_energy, sizeof(target_energy), 1, stream) != 1 ||
        fwrite(&target_g, sizeof(target_g), 1, stream) != 1 ||
        fwrite(&sigma_edge, sizeof(sigma_edge), 1, stream) != 1) {
        fclose(stream);
        return -1;
    }
    for (int b = 0; b < NLTE_N_FREQ_BINS; ++b)
        if (fwrite(&sigma, sizeof(sigma), 1, stream) != 1) {
            fclose(stream);
            return -1;
        }
    return fclose(stream);
}

static int write_superlevel_channel_asset(const char *path, double sigma_edge,
                                           double target_energy)
{
    FILE *stream = fopen(path, "wb");
    uint32_t magic = 0x434D4643u, version = 4u;
    int32_t n_levels = 2, n_freq = NLTE_N_FREQ_BINS, n_routes = 1;
    int32_t offsets[3] = {0, 1, 1}, target = 1, grid = 0;
    double probability = 1.0, target_g = 4.0;
    double sigma = 2.0;
    if (!stream) return -1;
    if (fwrite(&magic, 4, 1, stream) != 1 ||
        fwrite(&version, 4, 1, stream) != 1 ||
        fwrite(&n_levels, 4, 1, stream) != 1 ||
        fwrite(&n_freq, 4, 1, stream) != 1 ||
        fwrite(&n_routes, 4, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MIN}, 8, 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MAX}, 8, 1, stream) != 1 ||
        fwrite(offsets, sizeof(int32_t), 3, stream) != 3 ||
        fwrite(&target, sizeof(target), 1, stream) != 1 ||
        fwrite(&probability, sizeof(probability), 1, stream) != 1 ||
        fwrite(&grid, sizeof(grid), 1, stream) != 1 ||
        fwrite(&target_energy, sizeof(target_energy), 1, stream) != 1 ||
        fwrite(&target_g, sizeof(target_g), 1, stream) != 1 ||
        fwrite(&sigma_edge, sizeof(sigma_edge), 1, stream) != 1) {
        fclose(stream);
        return -1;
    }
    for (int b = 0; b < NLTE_N_FREQ_BINS; ++b)
        if (fwrite(&sigma, sizeof(sigma), 1, stream) != 1) {
            fclose(stream);
            return -1;
        }
    return fclose(stream);
}

static int write_policy_channel_asset(const char *path,int bad)
{
    if(write_superlevel_channel_asset(path,7e-18,0.0)!=0)return -1;
    FILE *f=fopen(path,"r+b"); if(!f)return -1;
    uint32_t version=5,magic=0x43525035u,pver=1,count=bad==3?0:bad==4?2:1;
    unsigned char model[32],phot[32];memset(model,1,32);memset(phot,2,32);
    int32_t p[6]={14,1,2,0,0,0};
    if(bad==1)p[3]=1;
    if(bad==2)p[2]=3;
    if(bad==5)memset(model,0,32);
    if(bad==6)p[0]=26;
    if(bad==7)memset(phot,0,32);
    if(bad==10){p[1]=2;p[2]=3;}
    int ok=fseek(f,4,SEEK_SET)==0 && fwrite(&version,4,1,f)==1 && fseek(f,0,SEEK_END)==0 &&
        fwrite(&magic,4,1,f)==1 && fwrite(&pver,4,1,f)==1 && fwrite(&count,4,1,f)==1 && fwrite(model,1,32,f)==32;
    for(uint32_t i=0;ok && i<count;i++)
        ok=fwrite(p,4,6,f)==6 && fwrite(phot,1,bad==8?16:32,f)==(size_t)(bad==8?16:32);
    if(bad==9)fputc(0,f);
    return fclose(f)==0 && ok?0:-1;
}

int main(void)
{
    char path[256], sigma_path[256], channels_path[256], bad_channels_path[256];
    char terminal_channels_path[256], bad_terminal_path[256];
    char edge_channels_path[256], bad_edge_channels_path[256];
    AtomicData atom;
    memset(&atom, 0, sizeof(atom));
    atom.n_levels = 1;
    snprintf(path, sizeof(path), "/tmp/lumina-sh-grid-loader-%ld.bin",
             (long)getpid());

    if (write_asset(path, NLTE_NU_MIN, 0) != 0 ||
        load_cmfgen_sigma_bf(&atom, path) != 0 || !atom.cmfgen_loaded ||
        atom.cmfgen_n_freq_bins != NLTE_N_FREQ_BINS ||
        atom.cmfgen_nu_min != NLTE_NU_MIN || atom.cmfgen_nu_max != NLTE_NU_MAX) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL positive\n");
        remove(path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);

    if (write_asset(path, 1.5e14, 0) != 0 ||
        load_cmfgen_sigma_bf(&atom, path) == 0 || atom.cmfgen_loaded) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL stale-range\n");
        remove(path);
        clear_asset(&atom);
        return 1;
    }
    if (write_asset(path, NLTE_NU_MIN, 1) != 0 ||
        load_cmfgen_sigma_bf(&atom, path) == 0 || atom.cmfgen_loaded) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL trailing-byte\n");
        remove(path);
        clear_asset(&atom);
        return 1;
    }
    remove(path);
    clear_asset(&atom);

    snprintf(sigma_path, sizeof(sigma_path), "/tmp/lumina-bf-channel-sigma-%ld.bin",
             (long)getpid());
    snprintf(channels_path, sizeof(channels_path), "/tmp/lumina-bf-channels-%ld.bin",
             (long)getpid());
    snprintf(bad_channels_path, sizeof(bad_channels_path),
             "/tmp/lumina-bf-channels-bad-%ld.bin", (long)getpid());
    snprintf(terminal_channels_path, sizeof(terminal_channels_path),
             "/tmp/lumina-bf-terminal-channels-%ld.bin", (long)getpid());
    snprintf(bad_terminal_path, sizeof(bad_terminal_path),
             "/tmp/lumina-bf-terminal-channels-bad-%ld.bin", (long)getpid());
    snprintf(edge_channels_path, sizeof(edge_channels_path),
             "/tmp/lumina-bf-edge-channels-%ld.bin", (long)getpid());
    snprintf(bad_edge_channels_path, sizeof(bad_edge_channels_path),
             "/tmp/lumina-bf-edge-channels-bad-%ld.bin", (long)getpid());
    int level_Z[2] = {14, 14}, level_ion[2] = {1, 2};
    double level_energy[2] = {0.0, 0.0};
    int level_g[2] = {2, 1};
    int level_super[2] = {0, 0};
    atom.n_levels = 2;
    atom.level_Z = level_Z;
    atom.level_ion = level_ion;
    atom.level_energy_eV = level_energy;
    atom.level_g = level_g;
    atom.level_super = level_super;
    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_channel_asset(channels_path, 1.0, 0) != 0 ||
        load_cmfgen_bf_channels(&atom, channels_path) != 0 ||
        !atom.cmfgen_bf_channels_loaded || atom.cmfgen_bf_channels_n_routes != 1 ||
        atom.cmfgen_bf_channel_offset[0] != 0 ||
        atom.cmfgen_bf_channel_offset[1] != 1 ||
        atom.cmfgen_bf_channel_offset[2] != 1 ||
        atom.cmfgen_bf_channel_target[0] != 1 ||
        atom.cmfgen_bf_channel_probability[0] != 1.0 ||
        atom.cmfgen_bf_channel_phot_grid[0] != 0 ||
        atom.cmfgen_bf_channel_sigma[NLTE_N_FREQ_BINS - 1] != 2.0) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-channel-positive\n");
        remove(sigma_path); remove(channels_path); remove(bad_channels_path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);

    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_superlevel_channel_asset(edge_channels_path, 7.0e-18, 0.0) != 0 ||
        load_cmfgen_bf_channels(&atom, edge_channels_path) != 0 ||
        atom.cmfgen_bf_channel_version != 4 ||
        atom.cmfgen_bf_channel_target_g[0] != 4.0) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-channel-v4-superlevel-positive\n");
        remove(sigma_path); remove(edge_channels_path);
        clear_asset(&atom);
        return 1;
    }
    setenv("LUMINA_CMFGEN_BF_CHANNELS","0",1);
    int legacy_disabled=load_cmfgen_bf_channels_from_env(&atom,edge_channels_path);
    unsetenv("LUMINA_CMFGEN_BF_CHANNELS");
    if(legacy_disabled!=0){fprintf(stderr,"SH_GRID_LOADER_SELFTEST_FAIL legacy-off\n");return 1;}
    clear_asset(&atom);

    for(int bad=0;bad<=10;bad++) {
        if(write_two_level_sigma(sigma_path)!=0 || load_cmfgen_sigma_bf(&atom,sigma_path)!=0 ||
           write_policy_channel_asset(edge_channels_path,bad)!=0)return 1;
        int loaded=load_cmfgen_bf_channels(&atom,edge_channels_path)==0;
        if(loaded!=(bad==0) || (loaded && (atom.cmfgen_bf_channel_version!=5 ||
           atom.cmfgen_bf_recombination_policy[0][0]!=1))) {
            fprintf(stderr,"SH_GRID_LOADER_SELFTEST_FAIL v5-policy bad=%d\n",bad);return 1;
        }
        if(loaded) {
            setenv("LUMINA_CMFGEN_BF_CHANNELS","0",1);
            int disabled=load_cmfgen_bf_channels_from_env(&atom,edge_channels_path);
            unsetenv("LUMINA_CMFGEN_BF_CHANNELS");
            if(disabled==0){fprintf(stderr,"SH_GRID_LOADER_SELFTEST_FAIL v5-policy-bypass\n");return 1;}
        }
        clear_asset(&atom);
    }

    /* Exercise the actual text/binary precision mismatch, and reject a
     * metadata change larger than serialization roundoff. */
    level_energy[1] = 1.2345678901;
    for (int bad = 0; bad < 2; bad++) {
        double raw_energy = 1.23456789012345 + (bad ? 1e-7 : 0.0);
        if (write_two_level_sigma(sigma_path) != 0 ||
            load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
            write_superlevel_channel_asset(edge_channels_path,7e-18,raw_energy) != 0) {
            fprintf(stderr,"SH_GRID_LOADER_SELFTEST_FAIL decimal-energy-fixture\n");
            return 1;
        }
        int loaded = load_cmfgen_bf_channels(&atom,edge_channels_path) == 0;
        if (loaded == bad) {
            fprintf(stderr,"SH_GRID_LOADER_SELFTEST_FAIL decimal-energy bad=%d\n",bad);
            clear_asset(&atom); return 1;
        }
        clear_asset(&atom);
    }
    level_energy[1] = 0.0;

    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_edge_channel_asset(edge_channels_path, 7.0e-18) != 0 ||
        load_cmfgen_bf_channels(&atom, edge_channels_path) != 0 ||
        !atom.cmfgen_bf_channels_loaded ||
        !atom.cmfgen_bf_channel_sigma_edge ||
        atom.cmfgen_bf_channel_sigma_edge[0] != 7.0e-18) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-channel-v3-edge-positive\n");
        remove(sigma_path); remove(channels_path); remove(bad_channels_path);
        remove(edge_channels_path); remove(bad_edge_channels_path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);
    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_edge_channel_asset(bad_edge_channels_path, -1.0e-18) != 0 ||
        load_cmfgen_bf_channels(&atom, bad_edge_channels_path) == 0 ||
        atom.cmfgen_bf_channels_loaded) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-channel-v3-edge-negative\n");
        remove(sigma_path); remove(channels_path); remove(bad_channels_path);
        remove(edge_channels_path); remove(bad_edge_channels_path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);
    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_channel_asset(bad_channels_path, 0.5, 0) != 0 ||
        load_cmfgen_bf_channels(&atom, bad_channels_path) == 0 ||
        atom.cmfgen_bf_channels_loaded) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-channel-bad-probability\n");
        remove(sigma_path); remove(channels_path); remove(bad_channels_path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);
    remove(sigma_path); remove(channels_path); remove(bad_channels_path);

    /* In the atomic level table Si/S V is ion_number=4; its separate
     * ion-population ladder represents the terminal VI reservoir as stage 5. */
    level_ion[0] = 4;
    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_terminal_channel_asset(terminal_channels_path, 0.0, 6.0) != 0 ||
        load_cmfgen_bf_channels(&atom, terminal_channels_path) != 0 ||
        !atom.cmfgen_bf_channels_loaded || atom.cmfgen_bf_channels_n_routes != 1 ||
        atom.cmfgen_bf_channel_target[0] != -1 ||
        atom.cmfgen_bf_channel_target_energy_eV[0] != 0.0 ||
        atom.cmfgen_bf_channel_target_g[0] != 6.0 ||
        atom.cmfgen_bf_channel_sigma[NLTE_N_FREQ_BINS - 1] != 2.0) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-terminal-si-v2-positive\n");
        remove(sigma_path); remove(channels_path); remove(bad_channels_path);
        remove(terminal_channels_path); remove(bad_terminal_path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);

    level_Z[0] = 16;
    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_terminal_channel_asset(terminal_channels_path, 0.0, 2.0) != 0 ||
        load_cmfgen_bf_channels(&atom, terminal_channels_path) != 0 ||
        !atom.cmfgen_bf_channels_loaded || atom.cmfgen_bf_channel_target[0] != -1 ||
        atom.cmfgen_bf_channel_target_g[0] != 2.0) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-terminal-s-v2-positive\n");
        remove(sigma_path); remove(channels_path); remove(bad_channels_path);
        remove(terminal_channels_path); remove(bad_terminal_path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);

    level_Z[0] = 14;
    level_ion[0] = 3;
    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_terminal_channel_asset(bad_terminal_path, 0.0, 6.0) != 0 ||
        load_cmfgen_bf_channels(&atom, bad_terminal_path) == 0 ||
        atom.cmfgen_bf_channels_loaded) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-terminal-wrong-stage\n");
        remove(sigma_path); remove(channels_path); remove(bad_channels_path);
        remove(terminal_channels_path); remove(bad_terminal_path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);

    level_ion[0] = 4;
    if (write_two_level_sigma(sigma_path) != 0 ||
        load_cmfgen_sigma_bf(&atom, sigma_path) != 0 ||
        write_terminal_channel_asset(bad_terminal_path, 0.0, 0.0) != 0 ||
        load_cmfgen_bf_channels(&atom, bad_terminal_path) == 0 ||
        atom.cmfgen_bf_channels_loaded) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL bf-terminal-invalid-stat-weight\n");
        remove(sigma_path); remove(channels_path); remove(bad_channels_path);
        remove(terminal_channels_path); remove(bad_terminal_path);
        clear_asset(&atom);
        return 1;
    }
    remove(sigma_path); remove(channels_path); remove(bad_channels_path);
    remove(terminal_channels_path); remove(bad_terminal_path);
    remove(edge_channels_path); remove(bad_edge_channels_path);
    clear_asset(&atom);
    printf("SH_GRID_LOADER_SELFTEST PASS bins=%d stale_range=REJECTED "
           "trailing_byte=REJECTED bf_channels=LOADED bad_probability=REJECTED "
           "bf_terminal_si_s_v2=LOADED bf_edge_v3=LOADED negative_edge=REJECTED "
           "wrong_stage=REJECTED invalid_g=REJECTED\n",
           NLTE_N_FREQ_BINS);
    return 0;
}
