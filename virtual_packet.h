/**
 * LUMINA-SN Virtual Packet Implementation (TARDIS-compatible)
 * virtual_packet.h - Header for virtual packet spectrum synthesis
 *
 * This implements the TARDIS virtual packet technique for observer spectra.
 * Virtual packets are spawned at each r-packet interaction and traced toward
 * the observer, with energy attenuated by exp(-tau) along the path.
 *
 * Key difference from LUMINA rotation:
 *   - Rotation: transform escaped packets at the end
 *   - Virtual packets: spawn toward observer at EACH interaction
 */

#ifndef VIRTUAL_PACKET_H
#define VIRTUAL_PACKET_H

#include <math.h>
#include <stdint.h>
#include "simulation_state.h"  /* For SimulationState, ShellState */
#include "physics_kernels.h"

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define VPACKET_TAU_RUSSIAN 10.0      /* Russian roulette threshold */
#define VPACKET_SURVIVAL_PROB 0.1     /* Survival probability for Russian roulette */
#define VPACKET_MAX_SHELLS 100        /* Maximum shells to traverse */

/* ============================================================================
 * VIRTUAL PACKET STRUCTURE
 * ============================================================================ */

typedef struct {
    double r;               /* Radius [cm] */
    double mu;              /* Direction cosine */
    double nu;              /* Frequency [Hz] */
    double energy;          /* Energy [erg] */
    int64_t current_shell;  /* Current shell index */
    int64_t next_line_id;   /* Next line in sorted list */
    int status;             /* 0=in_process, 1=escaped, 2=absorbed */
} VirtualPacket;

/* ============================================================================
 * VIRTUAL PACKET COLLECTION (stores spectrum contributions)
 * ============================================================================ */

typedef struct {
    double *nus;            /* Frequencies of collected v-packets */
    double *energies;       /* Energies of collected v-packets */
    int64_t n_packets;      /* Number of packets collected */
    int64_t capacity;       /* Allocated capacity */

    /* Spectrum binning */
    double nu_min;
    double nu_max;
    int64_t n_bins;
    double *spectrum;       /* Binned spectrum [erg/s/Hz] */
    int64_t *counts;        /* Counts per bin */
} VPacketCollection;

/* ============================================================================
 * FUNCTION DECLARATIONS
 * ============================================================================ */

/**
 * Initialize virtual packet collection
 */
void vpacket_collection_init(VPacketCollection *coll, int64_t capacity,
                             double nu_min, double nu_max, int64_t n_bins);

/**
 * Free virtual packet collection
 */
void vpacket_collection_free(VPacketCollection *coll);

/**
 * Reset collection for new iteration
 */
void vpacket_collection_reset(VPacketCollection *coll);

/**
 * Add a virtual packet to the collection
 */
void vpacket_collection_add(VPacketCollection *coll, double nu, double energy);

/**
 * Spawn virtual packets from r-packet position (TARDIS trace_vpacket_volley)
 *
 * This is the key function that creates virtual packets directed toward
 * the observer and traces them through the ejecta.
 *
 * @param r_pkt_r       R-packet radius
 * @param r_pkt_mu      R-packet direction cosine
 * @param r_pkt_nu      R-packet frequency
 * @param r_pkt_energy  R-packet energy
 * @param shell_id      Current shell index
 * @param next_line_id  Current line index
 * @param state         Simulation state
 * @param coll          Collection to store results
 * @param n_vpackets    Number of virtual packets to spawn
 */
void spawn_vpacket_volley(double r_pkt_r, double r_pkt_mu,
                          double r_pkt_nu, double r_pkt_energy,
                          int shell_id, int64_t next_line_id,
                          const SimulationState *state,
                          VPacketCollection *coll,
                          int n_vpackets);

/**
 * Trace a single virtual packet to escape
 *
 * @param vpkt          Virtual packet to trace
 * @param state         Simulation state
 * @return              Total optical depth accumulated
 */
double trace_vpacket(VirtualPacket *vpkt, const SimulationState *state);

/**
 * Trace virtual packet within single shell
 *
 * @param vpkt          Virtual packet
 * @param shell         Current shell
 * @param state         Simulation state
 * @param d_boundary    Output: distance to boundary
 * @param delta_shell   Output: shell change direction
 * @return              Optical depth in this shell
 */
double trace_vpacket_in_shell(VirtualPacket *vpkt,
                              const ShellState *shell,
                              const SimulationState *state,
                              double *d_boundary, int *delta_shell);

/**
 * Get binned spectrum from collection
 */
void vpacket_collection_get_spectrum(const VPacketCollection *coll,
                                     double *wavelength, double *flux,
                                     int64_t *n_points);

/**
 * Write virtual packet spectrum to file
 */
int vpacket_spectrum_write_csv(const VPacketCollection *coll,
                               const char *filename, double t_exp);

#endif /* VIRTUAL_PACKET_H */
