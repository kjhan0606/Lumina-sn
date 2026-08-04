#!/bin/bash
# Known-answer (Saha) self-test of the ionization path — CPU, minutes, no GPU.
#
#   make selftest_ioniz_saha
#   bash scripts/run_ioniz_saha_selftest.sh [outdir]
#
# Runs the identity  q = Gamma_phot/alpha_rec  ==  q_saha  under J_nu = B_nu(T_e)
# through the PRODUCTION radeq_simul_all Gph assembly / frozenin_alpha_rr /
# coupled_photoion_rate_jnu / simul_ladder, once per photoionization ROUTE.
#
# LUMINA_RADEQ_LINE_CULL=1e40 throws away the ETLA line table AFTER the probe has
# already printed (the probe sits in the Gph block, before the line table is
# built), so it only removes the minutes of post-probe radeq work.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-logs/ioniz_saha_selftest}
mkdir -p "$OUT"

run() { tag=$1; shift
  env "$@" LUMINA_IONIZ_SELFTEST=1 LUMINA_RADEQ_SIMUL=1 \
      LUMINA_RADEQ_LINE_CULL=1e40 \
      ./selftest_ioniz_saha > "$OUT/$tag.log" 2>&1
  echo "$tag exit=$?"
}

# A: detailed-balance route (all-level CMFGEN sigma both sides) — must give 1.0
run A_alllevel  LUMINA_GPH_SIGMA_CMFGEN=1 LUMINA_GPH_ALLLEVEL=1
# B: legacy default route (ground level, Kramers sigma)
run B_ground_kramers LUMINA_GPH_SIGMA_CMFGEN=0 LUMINA_GPH_ALLLEVEL=0
# C: ground level, CMFGEN sigma
run C_ground_cmfgen  LUMINA_GPH_SIGMA_CMFGEN=1 LUMINA_GPH_ALLLEVEL=0
# D: the parity39/40 champion gate set
run D_champion  LUMINA_GPH_SIGMA_CMFGEN=1 LUMINA_GPH_ALLLEVEL=1 \
                LUMINA_GPH_ALLLEVEL_NLTE=1 LUMINA_ALPHA_SPINGATE=1 \
                LUMINA_FROZENIN_DR=1 LUMINA_ARTIS_PARITY=1 \
                LUMINA_SIMUL_CAP_TOPION=1
echo "logs in $OUT"
