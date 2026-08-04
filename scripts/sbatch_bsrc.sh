#!/bin/bash
#SBATCH --job-name=a10_kx_bsrc
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# FORK B -- PER-LINE THERMAL SOURCE for the stage-IV iron-peak forest
#           (the surviving repair)                    (PREPARE ONLY -- driver submits)
# ============================================================================
# Exact clone of the B-run recipe (a10_kx_gphall MODE=all -- same env as
# sbatch_tincol.sh's B-run block) PLUS the source-function lever ONLY:
#     export LUMINA_LINE_BSRC=1  LUMINA_LINE_BSRC_MODE=1
# Binary: lumina_cuda.withBsrc (carries jtable/tetab/tincol/ltherm/stage4/bsrc
#         gates; ALL env-OFF by default; byte-identical to the B-run when
#         LUMINA_LINE_BSRC unset).
# NO stage4, NO macroatom-BF, NO line-therm: ISOLATE the per-line source lever.
#
# WHAT CHANGES (color only, energy conserved):
#   Every re-emission of a FLAGGED line (stage-IV iron-peak: Fe IV 26,3 /
#   Co IV 27,3 / Ni IV 28,3 -- 12,576 lines, non-NLTE => cs uses the B(T_e)
#   fallback source) redraws its COMOVING frequency from Planck(T_e[shell]) at
#   ALL shells. This makes the MC re-emission of exactly those lines reproduce
#   the deterministic-cs treatment (S -> B(T_e)) instead of the resonant
#   epsilon_eff~1e-10 scattering fixed point that piles Co IV UV. Packet
#   energies, weights, L_inner and the T_inner controller are bit-unchanged;
#   T_e stays FREE to evolve.  MODE=1 = full thermal (default). MODE=2 =
#   redshift-only guard (accept the Planck draw only if it lands redward of
#   1290 A comoving; blueward draws keep the resonant re-emit, protecting the
#   deep EUV/FUV pump) -- energy-neutral, same 5-uniform draw either way.
#
# WHY (crime_reconstruction VERDICT, 2026-07-19):
#   The deep-shell (v<7000) mechanism is an MC scattering fixed point S~=J with
#   epsilon_eff~1e-10; exit-channel repairs are dimensionally dead. The ONLY
#   lever is the per-line source function the packets sample. Deep Co IV /
#   Fe IV / Ni IV are outside the NLTE/SE set, so S_l is never written and the
#   MC recycles their lines resonantly => mc_J/cs_J = 39x at 1526 A while the cs
#   already matches CMFGEN's smooth ~B(T_e) deep field. Fork B is the direct
#   MC-side fix: sample Planck(T_e) for exactly those non-NLTE lines.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES (verbatim -- do not move the goalposts after the run):
#   PASS = (1) mc_J/cs_J @1526A s0 collapses 39.0x -> <=3x (ltherm floor 0.57x)
#              AND Co IV share of deep 1290-2000 pile emission falls from 84.0%
#              toward continuum levels;
#          (2) energy guard: u_bol(s0) within +-0.1 dex of the B-run 400 (color
#              redistribution only);
#          (3) pile u-frac(1290-2000, s0) 0.514 -> <=0.40 and toward CMFGEN 0.32;
#          (4) EUV guard: deep 450-1290A escape drops <=0.3 dex vs B-run (if
#              violated and (1)-(3) pass => rerun with MODE=2);
#   Directional: T_e(s0) rises from 13120 (LTHERM reached 14080 while also
#              killing EUV; Fork B should climb with less EUV damage);
#              FUV(918-1290, s0) rises; f(FeIV) s0 does NOT crater below 0.5
#              (LTHERM cratered to 0.294 by killing the funnel pump -- Fork B
#              keeps NLTE Fe III/Co III emission alive, argue expected behavior).
#   NULL = (1) fails => the reconstruction's frame is wrong at the
#              implementation level -- check [BSRC] counters and flagged-line
#              coverage first.
#   Wiring (check FIRST on any null):
#     [BSRC] flagged 12576 lines (Fe IV 4336, Co IV 4041, Ni IV 4199), mode=1
#            (once, at init)
#     [BSRC] it NN: thermalized=<N> (mode=1)                (once per iter;
#            N==0 with the gate ON prints a *** WARNING *** and is a wiring
#            no-op, NOT a physics null.)
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withBsrc binary as ./lumina_cuda for run_coevolve_s01.sh -------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
BS_BIN=lumina_cuda.withBsrc
[ -x "$BS_BIN" ] || { echo "FATAL: $BS_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[bsrc] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$BS_BIN" lumina_cuda
echo "[bsrc] installed $BS_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

# --- B-run (a10_kx_gphall MODE=all) environment, verbatim ----------------------
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
# etype 7 (electron scatter) is opt-in via this env; etype 8 (legacy bf re-emit)
# is already logged unconditionally under EVENT_LOG=1 in this binary. Both make
# the crime_reconstruction's bf/escatter blind spots visible.
export LUMINA_EVENT_LOG_ESCATTER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
# B-run had LUMINA_NLTE_BK_CEIL ABSENT (b_k cap OFF). Clear any inherited value.
unset LUMINA_NLTE_BK_CEIL
# ISOLATE the source-function lever: keep every other repair gate OFF even if
# something is inherited from the environment.
unset LUMINA_LINE_THERM LUMINA_NLTE_STAGE4 LUMINA_MACROATOM_BF LUMINA_TINNER_COLOR

# --- Fork B per-line thermal source (the ONLY new levers) -----------------------
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

TAG="a10_kx_bsrc"
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
echo "[bsrc] verify wiring: grep '\[BSRC\]' logs/coevolve_consume_${TAG}/stdout.log"
