#!/bin/bash
#SBATCH --job-name=ddc15init
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# MISSION: faithfully reproduce the DDC15 INITIAL EPOCH (0.976 d) from the public
# hydro file, and compare to DDC15's own emergent spectrum
# (DDC15_spec_2500_25500_interp5_000.976d.dat). Most controlled test of the
# pipeline: same model, same epoch, structure taken AS-IS (no decay/scaling/
# regridding). Reference built by scripts/build_ddc15_initial_epoch.py.
#
# The photosphere is cool (T~4434 K, peak ~6595 A) so this is a near-blackbody +
# weak-line spectrum (O/Si/Mg/S/Ca outer layers) — if LUMINA can't reproduce the
# model's own 0.976 d spectrum here, the divergence is internal (transport/atomic).
#
#   sbatch --export=ALL,LINE_INT=scatter   scripts/slurm_ddc15_initial.sh
#   sbatch --export=ALL,LINE_INT=macroatom scripts/slurm_ddc15_initial.sh

LINE_INT=${LINE_INT:-scatter}
BINNEDJ=1
DIFFUSE_BC=1
N_PKT=200000
N_ITER=10
# #5 cap-semantics fix (default OFF = legacy drop-on-cap, for clean A/B baseline)
CAP_REAL_ONLY=${CAP_REAL_ONLY:-0}
CAP_FORCE_ESCAPE=${CAP_FORCE_ESCAPE:-0}
# k-packet + collisional thermal pool (default OFF). Requires DYNAMIC_TRANSPROB=1.
KPACKET=${KPACKET:-0}
# task #4: self-consistent / radiative-equilibrium T_e (default OFF = ratio 0.9*T_rad).
# SELF_TE and RADEQ_TE both now route to the full radiative-equilibrium balance.
SELF_TE=${SELF_TE:-0}
RADEQ_TE=${RADEQ_TE:-0}
# under-relaxation factor for the radeq T_e solve (damps MC-J_nu shell-to-shell noise)
RADEQ_DAMP=${RADEQ_DAMP:-0.5}
# spectrum-synthesis mode (argv[4]): "spectrum"=real-only (default, unrecognized->real),
# "rotation"=real+observer-Doppler-reweighted (lumina_spectrum_rotation.csv), "all"=real+virtual+rotation
SPEC_MODE=${SPEC_MODE:-spectrum}
# cap-scaling sensitivity sweep: interaction cap (default 200, harness override of compile-default 100000)
MAX_INT=${MAX_INT:-200}
# 56Ni/56Co decay non-thermal ionization + gamma heating (default OFF). CMFGEN includes it.
GAMMA_DEP=${GAMMA_DEP:-0}
# time-dependent (frozen-in) ionization: backward-Euler dn/dt term in NLTE rate matrix
# (option A, single-epoch, Dt=t_exp, fully-ionized t=0 IC). Default OFF = steady-state.
TIMEDEP_ION=${TIMEDEP_ION:-0}
RUN_TAG="ddc15init_${LINE_INT}_cro${CAP_REAL_ONLY}_cfe${CAP_FORCE_ESCAPE}_kp${KPACKET}_ste${SELF_TE}_rte${RADEQ_TE}_mi${MAX_INT}_gd${GAMMA_DEP}_td${TIMEDEP_ION}_sm${SPEC_MODE}"

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/${DDC15_REF:-tardis_reference_ddc15_0p976d}"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
# BIN_SUFFIX lets a diagnostic run point at an alternately-named binary (e.g. _capdiag)
# without disturbing the canonical _macap binaries used by the running sweep.
BIN_SUFFIX=${BIN_SUFFIX:-macap}
if echo "$GPU_NAME" | grep -qiE "H100|H200"; then
    BIN="$ROOT/lumina_cuda_binnedj_h100_${BIN_SUFFIX}"
else
    BIN="$ROOT/lumina_cuda_binnedj_a40_${BIN_SUFFIX}"
fi

label="paperDDC15init_${RUN_TAG}"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15-INITIAL-EPOCH (0.976d) run  LINE_INT=$LINE_INT  N_PKT=$N_PKT N_ITER=$N_ITER ==="
echo "Host: $(hostname)  GPU: $GPU_NAME"
echo "Binary: $BIN  Ref: $REF  Time: $(date)"
python3 -c "import json; c=json.load(open('$REF/config.json')); print(f'  t_exp={c[\"time_explosion_s\"]:.0f}s ({c[\"time_explosion_s\"]/86400:.3f}d)  L={c[\"luminosity_inner_erg_s\"]:.3e}  T_inner={c[\"T_inner_K\"]}K  n_shells={c[\"n_shells\"]}')"

env LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_SKIP_Z=14 \
    LUMINA_NLTE_START_ITER=2 \
    LUMINA_NLTE_FLOOR_REG=1 \
    LUMINA_NLTE_INV_CEIL=1e4 \
    LUMINA_LINE_INTERACTION=$LINE_INT \
    LUMINA_MAX_INTERACTIONS=$MAX_INT \
    LUMINA_GAMMA_DEP=$GAMMA_DEP \
    LUMINA_TIMEDEP_ION=$TIMEDEP_ION \
    LUMINA_CAP_REAL_ONLY=$CAP_REAL_ONLY \
    LUMINA_CAP_FORCE_ESCAPE=$CAP_FORCE_ESCAPE \
    LUMINA_KPACKET=$KPACKET \
    LUMINA_SELF_CONSISTENT_TE=$SELF_TE \
    LUMINA_RADEQ_TE=$RADEQ_TE \
    LUMINA_RADEQ_DAMP=$RADEQ_DAMP \
    LUMINA_BINNED_J_ESTIMATOR=$BINNEDJ \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_NLTE_LEVEL_DUMP=1 \
    LUMINA_DIFFUSE_INNER_BC=$DIFFUSE_BC \
    LUMINA_ENERGY_BUDGET=1 \
    LUMINA_MA_FATE_ZIHIST=${MA_FATE_ZIHIST:-0} \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$SPEC_MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo ""
echo "--- transprob mode + interaction cap ---"
grep -E "Transition probs:|\[CAP\]|truncat" stdout.log | head -8
echo ""
echo "--- k-packet thermal pool (KPACKET=$KPACKET) ---"
grep -E "\[KPACKET\]" stdout.log | head -14
echo ""
echo "--- energy budget per iteration ---"
grep -E "E-BUDGET" stdout.log
echo ""
echo "--- T_inner / T_rad trajectory ---"
grep -E 'T_inner:|T_inner final|T_rad' stdout.log | tail -14
echo "Done: $(date)"
