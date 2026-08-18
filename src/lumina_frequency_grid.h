#ifndef LUMINA_FREQUENCY_GRID_H
#define LUMINA_FREQUENCY_GRID_H

/* Single authority for the current NLTE/BF frequency grid.  Keep these
 * constants in a dependency-light header so radiation_field.h is valid when
 * included without the monolithic lumina.h. */
/* SH-GRID (2026-08-08): extend the old 1000-bin grid downward by exactly
 * 178 and upward by exactly 56 old logarithmic spacings.  The stored endpoints
 * make log(NU_MAX/NU_MIN)/1234 bit-identical to the old
 * log(3.0e16/1.5e14)/1000 spacing.  The upper edge is also the already sealed
 * canonical radiation-field edge at 74.2748474421 A and strictly contains the
 * limiting active Si V threshold at 74.3455731128 A. */
#define NLTE_N_FREQ_BINS  1234
#define NLTE_NU_MIN       5.8412785919616062e13
#define NLTE_NU_MAX       4.0362581455823112e16

#endif
