#!/bin/bash
#SBATCH --job-name=ddc15_mc_macro
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# PHASE 3 (option-2 endgame): feed the DETERMINISTIC J_nu directly into the
# ionization rate, replacing the nebular-Saha phi_neb closure.
#
# Mechanism (already wired in lumina_cuda.cu:3040 pure-CMFGEN block):
#   cmfgen_solve_J  -> deterministic J_nu -> nlte.J_nu (cmfgen_write_jnu)
#   coupled_newton_solve_all (LUMINA_COUPLED_NEWTON=1) re-solves {n_i,n_e,T_e}
#   per shell; with LUMINA_COUPLED_JNU_PHOTOION=1 the photoionization rate is
#   Gamma = Int(4*pi*J_nu/h/nu)*sigma_bf dnu (coupled_photoion_rate_jnu) using
#   the deterministic J_nu, instead of Gamma = alpha*phi_neb.
#
# KNOWN RISK: the thin-UV J/S diagonal-ALI floor (Defect A) lives in nlte.J_nu;
# integrating it through sigma_bf can over-ionize the outer. The B3-1 diagonal-
# Lambda* blend (LUMINA_COUPLED_JNU_LSTAR=1 + LUMINA_COUPLED_LAMBDA_STAR=1)
# replaces the blown thin-UV J with W*B(T_e) where the gas is thick to its own
# bf continuum. Run two arms (LSTAR=0 / LSTAR=1) and A/B vs baseline 164941.
#
# Toggle the mitigation arm with env JNU_LSTAR=0|1 at submit time.

module load cuda/13.0.2 2>/dev/null || true

N_PKT=${N_PKT:-500000}
N_ITER=${N_ITER:-15}
SPEC_MODE=${SPEC_MODE:-spectrum}
JNU_LSTAR=${JNU_LSTAR:-0}        # 0 = bare J_nu photoion; 1 = + diagonal-Lambda* blend
LSTAR=${LSTAR:-$JNU_LSTAR}       # RADEQ/Newton T_e radiation response (Phase-1 faithful Lambda*); defaults to JNU_LSTAR
LINE_RE=${LINE_RE:-0}            # 0 = collisional bound-bound cooling; 1 = Option-2 integral-RE line term
TE_RATIO=${TE_RATIO:-0.9}        # T_e/T_rad seed+fallback constant (perturbation test: 0.7 / 1.0)
JNU_PHOTOION=${JNU_PHOTOION:-1}  # 0 = phi_neb closure (control arm: falsify the J->Gamma chain)
FROZENIN=${FROZENIN:-0}          # 1 = frozen-in freeze-out owns the outer shells (+ per-ion rescale)

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/${DDC15_REF:-tardis_reference_ddc15_0p976d}"
BIN="$ROOT/lumina_cuda"

label="ddc15_pc_phase3_jnul${JNU_LSTAR}_radls${LSTAR}_linere${LINE_RE}_ratio${TE_RATIO}_pi${JNU_PHOTOION}_fz${FROZENIN}"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d PURE-CMFGEN PHASE 3: J_nu -> ionization rate ==="
echo "Host: $(hostname)  Time: $(date)"
echo "Binary: $BIN  Ref: $REF  JNU_LSTAR=$JNU_LSTAR  LINE_RE=$LINE_RE"
ls -l "$BIN"

    LUMINA_CMFGEN_ALI_ITER=${LUMINA_CMFGEN_ALI_ITER:-8} \
    LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_SKIP_Z=14 \
    LUMINA_NLTE_START_ITER=2 \
    LUMINA_NLTE_FLOOR_REG=1 \
    LUMINA_NLTE_INV_CEIL=1e4 \
    LUMINA_RADEQ_TE=1 \
    LUMINA_RADEQ_DIAG=1 \
    LUMINA_RADEQ_COOL_ESCAPE=0 \
    LUMINA_RADEQ_COOL_NONNEG=0 \
    LUMINA_RADEQ_COOL_NLTE_ONLY=1 \
    LUMINA_COUPLED_NEWTON=1 \
    LUMINA_COUPLED_JNU_PHOTOION=$JNU_PHOTOION \
    LUMINA_FROZENIN=$FROZENIN \
    LUMINA_NLTE_PER_ION_RESCALE=$FROZENIN \
    LUMINA_COUPLED_JNU_LSTAR=$JNU_LSTAR \
    LUMINA_COUPLED_LAMBDA_STAR=$LSTAR \
    LUMINA_COUPLED_TDEP=1 \
    LUMINA_RADEQ_LINE_RE=$LINE_RE \
    LUMINA_TE_TRAD_RATIO=$TE_RATIO \
    LUMINA_LINE_INTERACTION=macroatom \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_DIFFUSE_INNER_BC=1 \
    LUMINA_ENERGY_BUDGET=1 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" virtual nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo ""
echo "--- [CMFGEN] driver lines ---"
grep -E "\[CMFGEN\] iter" stdout.log | tail -20
echo ""
echo "--- plasma_state head ---"
head -6 lumina_plasma_state.csv 2>/dev/null
echo "Done: $(date)"
