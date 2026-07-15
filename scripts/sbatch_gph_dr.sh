#!/bin/bash
#SBATCH --job-name=a10_kx_gphdr
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:30:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# DR ARM: LUMINA_FROZENIN_DR=1 adds dielectronic recombination (Badnell/NORAD
# DR_TABLE) to frozenin_alpha_rr -> simul r=G/(n_e*alpha). Physical-completeness
# term (all reference codes include DR); measured in vivo on top of MODE arm.
#   MODE=ground : D1 = A(ground-only Gph) + DR
#   MODE=all    : D2 = B(all-level Boltzmann Gph) + DR
# DR coverage: S(16,1-4) Si(14,1-4) Ca(20,1-2) Fe(26,1-5); Co(27) & Ni(28) only
# to ion_recomb 2 -> Co IV->III and Ni IV/V->III/IV have NO DR (asymmetry watch).
# Offline pre-registration (dr_alpha_eval coefficients, T_e 10.5-16kK):
#   alpha_DR: S III->II ~3.3e-12 (~2x total alpha), Si III->II ~7.5e-12 (~6x),
#   Fe IV->III ~4.7e-12 (~3x). Predictions:
#  D1 vs A: S/Si f(III) roughly halves (still II-dominant), Fe core f(IV)
#     0.10 -> ~0.03-0.05, spectrum narrow-corr within +-0.03 of 0.474 (null-ish).
#  D2 vs B: S III rail NOT rescued (needs ~500x, DR gives ~2x -> III >=99%);
#     Fe core f(IV) 0.92 -> ~0.6-0.8 (DR pulls Fe III while Co/Ni unchanged ->
#     palette asymmetry Fe III vs Co IV). If instead S II recovers >30%,
#     self-consistent amplification >> offline 2x = real coupling discovery.
#  Falsifier role: kills/confirms "DR explains ARTIS(S II)<->CMFGEN(S III@s8)
#     spread" and quantifies the missing physical term for the CMFGEN-standard
#     comparison once our own CMFGEN toy06 run lands.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
MODE=${MODE:-ground}
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  MODE=$MODE+DR  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_FROZENIN_DR=1
[ "$MODE" = "all" ] && export LUMINA_GPH_ALLLEVEL=1

TAG="a10_kx_gphdr_${MODE}"
mkdir -p logs/coevolve_consume_${TAG}
rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="$TAG"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_${TAG}/$f"
done
echo "${TAG} DONE -> logs/coevolve_consume_${TAG}/"
