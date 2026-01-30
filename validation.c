/**
 * LUMINA-SN Validation Framework
 * validation.c - Implementation of trace recording and comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "physics_kernels.h"
#include "rpacket.h"
#include "validation.h"

/* ============================================================================
 * VALIDATION TRACE MANAGEMENT
 * ============================================================================ */

ValidationTrace *validation_trace_create(int64_t packet_index, int64_t capacity) {
    ValidationTrace *trace = (ValidationTrace *)malloc(sizeof(ValidationTrace));
    if (!trace) return NULL;

    trace->packet_index = packet_index;
    trace->n_snapshots = 0;
    trace->capacity = capacity;
    trace->snapshots = (PacketSnapshot *)malloc(capacity * sizeof(PacketSnapshot));

    if (!trace->snapshots) {
        free(trace);
        return NULL;
    }

    return trace;
}

void validation_trace_free(ValidationTrace *trace) {
    if (trace) {
        if (trace->snapshots) {
            free(trace->snapshots);
        }
        free(trace);
    }
}

void validation_trace_record(ValidationTrace *trace, const RPacket *pkt,
                             InteractionType itype, double distance) {
    if (!trace || trace->n_snapshots >= trace->capacity) {
        return;
    }

    PacketSnapshot *snap = &trace->snapshots[trace->n_snapshots];
    snap->step_number = trace->n_snapshots;
    snap->r = pkt->r;
    snap->mu = pkt->mu;
    snap->nu = pkt->nu;
    snap->energy = pkt->energy;
    snap->shell_id = pkt->current_shell_id;
    snap->status = pkt->status;
    snap->interaction_type = (int64_t)itype;
    snap->distance = distance;

    trace->n_snapshots++;
}

int validation_trace_write_binary(const ValidationTrace *trace, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return -1;

    /* Header */
    fwrite(&trace->packet_index, sizeof(int64_t), 1, fp);
    fwrite(&trace->n_snapshots, sizeof(int64_t), 1, fp);

    /* Snapshots */
    for (int64_t i = 0; i < trace->n_snapshots; i++) {
        const PacketSnapshot *snap = &trace->snapshots[i];
        fwrite(&snap->step_number, sizeof(int64_t), 1, fp);
        fwrite(&snap->r, sizeof(double), 1, fp);
        fwrite(&snap->mu, sizeof(double), 1, fp);
        fwrite(&snap->nu, sizeof(double), 1, fp);
        fwrite(&snap->energy, sizeof(double), 1, fp);
        fwrite(&snap->shell_id, sizeof(int64_t), 1, fp);
        fwrite(&snap->status, sizeof(int64_t), 1, fp);
        fwrite(&snap->interaction_type, sizeof(int64_t), 1, fp);
        fwrite(&snap->distance, sizeof(double), 1, fp);
    }

    fclose(fp);
    return 0;
}

int validation_trace_write_csv(const ValidationTrace *trace, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return -1;

    fprintf(fp, "step,r,mu,nu,energy,shell_id,status,interaction_type,distance\n");

    for (int64_t i = 0; i < trace->n_snapshots; i++) {
        const PacketSnapshot *snap = &trace->snapshots[i];
        fprintf(fp, "%ld,%.15e,%.15f,%.15e,%.15e,%ld,%ld,%ld,%.15e\n",
                (long)snap->step_number, snap->r, snap->mu, snap->nu,
                snap->energy, (long)snap->shell_id, (long)snap->status,
                (long)snap->interaction_type, snap->distance);
    }

    fclose(fp);
    return 0;
}

ValidationTrace *validation_trace_load_binary(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    int64_t packet_index, n_snapshots;
    if (fread(&packet_index, sizeof(int64_t), 1, fp) != 1 ||
        fread(&n_snapshots, sizeof(int64_t), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    ValidationTrace *trace = validation_trace_create(packet_index, n_snapshots);
    if (!trace) {
        fclose(fp);
        return NULL;
    }

    for (int64_t i = 0; i < n_snapshots; i++) {
        PacketSnapshot *snap = &trace->snapshots[i];
        if (fread(&snap->step_number, sizeof(int64_t), 1, fp) != 1 ||
            fread(&snap->r, sizeof(double), 1, fp) != 1 ||
            fread(&snap->mu, sizeof(double), 1, fp) != 1 ||
            fread(&snap->nu, sizeof(double), 1, fp) != 1 ||
            fread(&snap->energy, sizeof(double), 1, fp) != 1 ||
            fread(&snap->shell_id, sizeof(int64_t), 1, fp) != 1 ||
            fread(&snap->status, sizeof(int64_t), 1, fp) != 1 ||
            fread(&snap->interaction_type, sizeof(int64_t), 1, fp) != 1 ||
            fread(&snap->distance, sizeof(double), 1, fp) != 1) {
            validation_trace_free(trace);
            fclose(fp);
            return NULL;
        }
        trace->n_snapshots++;
    }

    fclose(fp);
    return trace;
}

int64_t validation_compare_traces(const ValidationTrace *trace_c,
                                   const ValidationTrace *trace_py,
                                   double tolerance) {
    int64_t mismatches = 0;
    int64_t n_compare = (trace_c->n_snapshots < trace_py->n_snapshots)
                        ? trace_c->n_snapshots : trace_py->n_snapshots;

    for (int64_t i = 0; i < n_compare; i++) {
        const PacketSnapshot *c = &trace_c->snapshots[i];
        const PacketSnapshot *py = &trace_py->snapshots[i];

        double r_err = fabs(c->r - py->r) / fabs(py->r + 1e-100);
        double mu_err = fabs(c->mu - py->mu) / (fabs(py->mu) + 1e-100);
        double nu_err = fabs(c->nu - py->nu) / fabs(py->nu + 1e-100);

        if (r_err > tolerance || mu_err > tolerance || nu_err > tolerance) {
            mismatches++;
        }
    }

    /* Count length mismatch as additional errors */
    if (trace_c->n_snapshots != trace_py->n_snapshots) {
        mismatches += labs(trace_c->n_snapshots - trace_py->n_snapshots);
    }

    return mismatches;
}

/* ============================================================================
 * TRACED PACKET LOOP (for validation)
 * ============================================================================ */

void single_packet_loop_traced(RPacket *pkt, const NumbaModel *model,
                               const NumbaPlasma *plasma,
                               const MonteCarloConfig *config,
                               Estimators *estimators,
                               ValidationTrace *trace) {
    /*
     * Same as single_packet_loop but records state after each interaction.
     * Used for validation against Python implementation.
     */

    /* Initialize line search position */
    rpacket_initialize_line_id(pkt, plasma, model);

    /* Apply relativistic corrections to initial state */
    if (config->enable_full_relativity) {
        ENABLE_FULL_RELATIVITY = 1;
        double beta = pkt->r / (model->time_explosion * C_SPEED_OF_LIGHT);
        double inv_doppler = get_inverse_doppler_factor(
            pkt->r, pkt->mu, model->time_explosion);
        pkt->nu *= inv_doppler;
        pkt->energy *= inv_doppler;
        pkt->mu = (pkt->mu + beta) / (1.0 + beta * pkt->mu);
    } else {
        ENABLE_FULL_RELATIVITY = 0;
        double inv_doppler = get_inverse_doppler_factor(
            pkt->r, pkt->mu, model->time_explosion);
        pkt->nu *= inv_doppler;
        pkt->energy *= inv_doppler;
    }

    /* Main transport loop */
    while (pkt->status == PACKET_IN_PROCESS) {

        /* Find next interaction */
        double distance;
        int delta_shell;
        InteractionType itype = trace_packet(
            pkt, model, plasma, config, estimators, &distance, &delta_shell);

        /* Process based on interaction type */
        switch (itype) {

            case INTERACTION_BOUNDARY:
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                move_packet_across_shell_boundary(
                    pkt, delta_shell, model->n_shells);
                break;

            case INTERACTION_LINE:
                pkt->last_interaction_type = 2;
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                line_scatter(pkt, model->time_explosion,
                            config->line_interaction_type, plasma);
                break;

            case INTERACTION_ESCATTERING:
                pkt->last_interaction_type = 1;
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                thomson_scatter(pkt, model->time_explosion);
                break;
        }

        /* Record state after interaction */
        if (trace) {
            validation_trace_record(trace, pkt, itype, distance);
        }
    }
}
