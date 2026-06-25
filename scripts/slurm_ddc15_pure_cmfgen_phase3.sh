#!/bin/bash
#SBATCH --job-name=ddc15_pc_phase3
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
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

N_PKT=${N_PKT:-1000}
N_ITER=${N_ITER:-8}
SPEC_MODE=${SPEC_MODE:-spectrum}
JNU_LSTAR=${JNU_LSTAR:-0}        # 0 = bare J_nu photoion; 1 = + diagonal-Lambda* blend
LSTAR=${LSTAR:-$JNU_LSTAR}       # RADEQ/Newton T_e radiation response (Phase-1 faithful Lambda*); defaults to JNU_LSTAR
LINE_RE=${LINE_RE:-0}            # 0 = collisional bound-bound cooling; 1 = Option-2 integral-RE line term
TE_RATIO=${TE_RATIO:-0.9}        # T_e/T_rad seed+fallback constant (perturbation test: 0.7 / 1.0)
JNU_PHOTOION=${JNU_PHOTOION:-1}  # 0 = phi_neb closure (control arm: falsify the J->Gamma chain)
FROZENIN=${FROZENIN:-0}          # 1 = frozen-in freeze-out owns the outer shells (+ per-ion rescale)
THEN_MC=${THEN_MC:-0}            # 1 = after pure-CMFGEN converges plasma, run MC macroatom for the emergent spectrum
MAX_INT=${MAX_INT:-100000}       # per-packet interaction cap for the THEN_MC pass (std-MC uses 200; 100000 over-interacts -> too blue)

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

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

env LUMINA_PURE_CMFGEN=1 \
    LUMINA_PURE_CMFGEN_ITER=$N_ITER \
    LUMINA_CMFGEN_ALI_ITER=${LUMINA_CMFGEN_ALI_ITER:-8} \
    LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=${SIGMA_BF:-$REF/cmfgen_sigma_bf.bin} \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_SKIP_Z=${SKIP_Z:-14} \
    LUMINA_NLTE_START_ITER=2 \
    `# ORTHODOX cold-Te NLTE closure (2026-06-24): the rate matrix is genuinely` \
    `# rank-deficient at cold/dilute shells, so the super-thermal S_l artifact is` \
    `# cured by FLOOR_REG=0 LTE_FLOOR=1 (replace the Boltzmann@T_RAD anchor of` \
    `# isolated levels with a physical LTE@T_e level-zeroing floor). Validated:` \
    `# optical 0% residual super-thermal emissivity. ADOPTED AS DEFAULT 2026-06-25:` \
    `# A/B 169732(legacy) vs 169733(orthodox) -> optical max S_l/B 8.7e6 -> 1.00,` \
    `# 30212 super-thermal lines -> 0, freqres peak 7234 -> 6830 (gold 6790),` \
    `# grn/nir 0.26 -> 0.38, plasma no-regress (T_e 0.98, n_e dex 0.18).` \
    LUMINA_NLTE_FLOOR_REG=${FLOOR_REG:-0} \
    LUMINA_NLTE_INV_CEIL=${INV_CEIL:-1e4} \
    LUMINA_RADEQ_TE=${RADEQ_TE:-1} \
    LUMINA_RADEQ_DIAG=1 \
    LUMINA_RADEQ_COOL_ESCAPE=0 \
    LUMINA_RADEQ_COOL_NONNEG=0 \
    LUMINA_RADEQ_COOL_NLTE_ONLY=1 \
    LUMINA_COUPLED_NEWTON=${NEWTON:-1} \
    LUMINA_COUPLED_JNU_PHOTOION=$JNU_PHOTOION \
    LUMINA_FROZENIN=$FROZENIN \
    LUMINA_NLTE_PER_ION_RESCALE=$FROZENIN \
    LUMINA_COUPLED_JNU_LSTAR=$JNU_LSTAR \
    LUMINA_COUPLED_LAMBDA_STAR=$LSTAR \
    LUMINA_COUPLED_TDEP=${TDEP:-1} \
    LUMINA_RADEQ_LINE_RE=$LINE_RE \
    LUMINA_TE_TRAD_RATIO=$TE_RATIO \
    LUMINA_LINE_INTERACTION=${LINE_INTERACT:-macroatom} \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_DIFFUSE_INNER_BC=1 \
    LUMINA_ENERGY_BUDGET=1 \
    LUMINA_TOPSTAGE_ANCHOR=${TOPSTAGE:-0} \
    LUMINA_TOPSTAGE_IV=${TOPIV:-0} \
    LUMINA_TOPSTAGE_IV_ZONLY=${TOPIV_Z:-0} \
    LUMINA_TOPSTAGE_THERMALIZE=${TSTH:-0} \
    LUMINA_TOPSTAGE_DEPARTURE=${TSTH_DEP:-0} \
    LUMINA_CMFGEN_THEN_MC=${THEN_MC:-0} \
    LUMINA_MAX_INTERACTIONS=$MAX_INT \
    LUMINA_MACROATOM_EWEIGHT=${EWEIGHT:-0} \
    LUMINA_NLTE_COLL_FLOOR=${COLL_FLOOR:-0} \
    LUMINA_NLTE_BK_CEIL=${BK_CEIL:-0} \
    LUMINA_NLTE_LTE_FLOOR=${LTE_FLOOR:-1} \
    LUMINA_NLTE_COLL_FIX=${COLL_FIX:-1} \
    LUMINA_NLTE_ION_LOCK=${ION_LOCK:-1} \
    LUMINA_NLTE_LOCK_START_ITER=${LOCK_START:-0} \
    LUMINA_NLTE_FALLBACK_TE=${FALLBACK_TE:-1} \
    LUMINA_NLTE_RESID_CHECK=${RESID_CHECK:-0} \
    LUMINA_NLTE_RESID_TOL=${RESID_TOL:-1e-3} \
    LUMINA_NLTE_LTE_NCRIT=${LTE_NCRIT:-1e8} \
    LUMINA_CMFGEN_FROZEN_MORPH=${FROZEN_MORPH:-0} \
    LUMINA_CMFGEN_FROZEN_EPS=${FROZEN_EPS:-0} \
    LUMINA_CMFGEN_FROZEN_ALI=${FROZEN_ALI:-60} \
    LUMINA_CMFGEN_FROZEN_CONT=${FROZEN_CONT:-1} \
    LUMINA_CMFGEN_FROZEN_FEATURE_ONLY=${FEATURE_ONLY:-0} \
    LUMINA_CMFGEN_FEATURE_Z=${FEATURE_Z:-8,12,14,16,20} \
    LUMINA_TOPSTAGE_TAU=${TSTH_TAU:-10} \
    LUMINA_MALI=${LUMINA_MALI:-0} \
    LUMINA_NLTE_JBAR_POPS=${JBAR_POPS:-0} \
    LUMINA_JBAR_POPS_DAMP=${JBAR_POPS_DAMP:-0.3} \
    LUMINA_FLUOR_ORACLE_X=${FLUOR_ORACLE_X:-1} \
    LUMINA_SL_DUMP=${LUMINA_SL_DUMP:-0} \
    LUMINA_LEVELPOP_DUMP=${LEVELPOP_DUMP:-0} \
    LUMINA_FI_CLAMP_SL=${FI_CLAMP:-0} \
    LUMINA_FI_CONT_OPACITY=${FI_CONT:-0} \
    LUMINA_CMF_OBS_CONTONLY=${OBS_CONTONLY:-0} \
    LUMINA_CMF_CONTONLY=${CONTONLY:-0} \
    LUMINA_FI_FOREST_NOBLANK=${FOREST_NOBLANK:-0} \
    LUMINA_FI_LEDGER=${FI_LEDGER:-0} \
    LUMINA_NLTE_BUDGET_DUMP=${LUMINA_NLTE_BUDGET_DUMP:-0} \
    LUMINA_BUDGET_Z=${LUMINA_BUDGET_Z:-8} \
    LUMINA_BUDGET_STAGE=${LUMINA_BUDGET_STAGE:-2} \
    LUMINA_BUDGET_SHELL=${LUMINA_BUDGET_SHELL:-8} \
    LUMINA_CMF_LINERES_JBAR=${LINERES_JBAR:-0} \
    LUMINA_CMF_LINERES_CONSUME=${CONSUME:-0} \
    LUMINA_CMF_FINE_DIAG=${FINE_DIAG:-0} \
    LUMINA_CMF_FINE_LINEDUMP=${LINEDUMP:-0} \
    LUMINA_NLTE_JEQB=${JEQB:-0} \
    LUMINA_NLTE_BK_PARTIAL=${BK_PARTIAL:-0} \
    LUMINA_NLTE_MATDUMP=${MATDUMP:-0} \
    LUMINA_NLTE_MATDUMP_PATH=${MATDUMP_PATH:-lumina_nlte_matrix.bin} \
    LUMINA_POP_Z=${POP_Z:-8} \
    LUMINA_POP_ION=${POP_ION:-1} \
    LUMINA_POP_SHELL=${POP_SHELL:-24} \
    LUMINA_CMF_FINE_LAMLO=${LAMLO:-3000} \
    LUMINA_CMF_FINE_LAMHI=${LAMHI:-3200} \
    LUMINA_CMF_FINE_VDOP=${VDOP:-1e6} \
    LUMINA_CMF_FINE_PPD=${PPD:-12} \
    LUMINA_CMF_FINE_ALI=${FINE_ALI:-16} \
    LUMINA_CMF_FINE_EMERGENT=${FINE_EMERGENT:-0} \
    LUMINA_CMF_FINE_EMERGENT_OBS=${FINE_EMERGENT_OBS:-0} \
    LUMINA_CMF_FINE_OBS_NOBS=${FINE_OBS_NOBS:-3000} \
    LUMINA_CMF_FINE_OBS_SCATTER=${FINE_OBS_SCATTER:-0} \
    LUMINA_CMF_FINE_SL_CLAMP=${FINE_SL_CLAMP:-0} \
    LUMINA_CMF_FINE_TAUMIN=${FINE_TAUMIN:-1e-12} \
    LUMINA_CMF_FINE_CONTONLY=${FINE_CONTONLY:-0} \
    LUMINA_SUPER_LEVELS=${SUPER_LEVELS:-0} \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$SPEC_MODE" nlte \
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
