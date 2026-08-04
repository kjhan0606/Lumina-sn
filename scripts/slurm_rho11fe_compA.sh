#!/bin/bash
#SBATCH --job-name=rho11fe_A
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-1
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# Path (A): Mazzali ρ-11fe-style composition probe at physical baseline.
# Geometry/density: physical ref unchanged (v_in=10400, t_exp=17.2d, L=1.3e43).
# Composition: ρ-11fe-like — Si peak 0.55 at inner shells (v=10.4-12.5k km/s),
# narrow IME shell, O-dominated outer envelope (>16k km/s), Fe-group only as
# 56Ni decay product in inner shells, no 56Ni above 14k km/s.
# Array 0: + LUMINA_NLTE_SKIP_Z=14 (Si LTE for fair compare with prior SkipSi)
# Array 1: full NLTE on Si (test whether composition fix alone is enough)

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_2011fe_physical"
BIN="$ROOT/lumina_cuda_h100_cefix"

N_PKT=50000
N_ITER=8

VARIANTS=("skipSi" "fullNLTE")
variant=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
label="rho11fe_${variant}"

work_root="$ROOT/logs/rho11feA_${SLURM_ARRAY_JOB_ID}_${label}"
mkdir -p "$work_root"
cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$SRC_REF"/*; do
    bn=$(basename "$f")
    ln -sf "$f" "$REF_DIR/$bn"
done
# Overrides we'll regenerate:
rm -f "$REF_DIR/abundances.csv"

python3 <<PYEOF
import json, numpy as np, pandas as pd
src = "$SRC_REF"
dst = "$REF_DIR"

geom = pd.read_csv(f"{src}/geometry.csv")
v_mid_kms = (geom["v_inner"].values + geom["v_outer"].values) / 2 / 1e5
n_shells = len(v_mid_kms)

# ρ-11fe-style composition profile (Mazzali+2014, Blondin DDC25-like)
# Inner ~10-13k km/s: Si-dominant IME shell (X_Si ≈ 0.55)
# 13-16k km/s: IME → O transition
# >16k km/s: O+C envelope (X_O > 0.75)
# Fe-group: only as 56Ni decay product, vanishing above ~14k km/s

def rho11fe_profile(v):
    # Silicon plateau then ramp
    if   v < 12500: X_Si = 0.55
    elif v < 16000: X_Si = 0.55 - 0.50*(v-12500)/3500
    else:           X_Si = 0.02
    # Oxygen ramp up
    if   v < 11000: X_O = 0.18
    elif v < 16000: X_O = 0.18 + 0.60*(v-11000)/5000
    else:           X_O = 0.78
    # Carbon: trace inner, rises in outer envelope
    if   v < 12000: X_C = 0.02
    elif v < 17000: X_C = 0.02 + 0.11*(v-12000)/5000
    else:           X_C = 0.13
    # S, Ca tied to Si (IME ratios from solar Si:S:Ca ≈ 1:0.13:0.045)
    X_S  = X_Si * 0.13
    X_Ca = X_Si * 0.045
    # Mg roughly flat ~0.04
    X_Mg = 0.04 if v < 16000 else 0.02
    # Fe-group: decay tail above 56Ni edge ~8500 km/s
    if   v < 11000: X_Fe = 0.10
    elif v < 15000: X_Fe = 0.10 * np.exp(-(v-11000)/1500.0)
    else:           X_Fe = 1e-3
    if   v < 11000: X_Ni = 5e-3
    elif v < 14000: X_Ni = 5e-3 * np.exp(-(v-11000)/1000.0)
    else:           X_Ni = 1e-5
    if   v < 11000: X_Co = 3e-3
    elif v < 14000: X_Co = 3e-3 * np.exp(-(v-11000)/1000.0)
    else:           X_Co = 1e-6
    # Light trace (Al, Sc, Ti, V, Cr, Mn)
    X_Al = 4e-3
    X_Sc = 1e-5
    X_Ti = 1e-4
    X_V  = 5e-5
    X_Cr = X_Fe * 0.05
    X_Mn = X_Fe * 0.03
    return {6:X_C, 8:X_O, 12:X_Mg, 13:X_Al, 14:X_Si, 16:X_S,
            20:X_Ca, 21:X_Sc, 22:X_Ti, 23:X_V, 24:X_Cr, 25:X_Mn,
            26:X_Fe, 27:X_Co, 28:X_Ni}

Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
X = np.zeros((len(Z_LIST), n_shells))
for s, v in enumerate(v_mid_kms):
    prof = rho11fe_profile(v)
    for i, Z in enumerate(Z_LIST):
        X[i, s] = prof[Z]
# Normalize per-shell (Σ X = 1)
for s in range(n_shells):
    X[:, s] /= X[:, s].sum()

cols = ["atomic_number"] + [str(s) for s in range(n_shells)]
rows = []
for i, Z in enumerate(Z_LIST):
    rows.append([Z] + list(X[i]))
pd.DataFrame(rows, columns=cols).to_csv(f"{dst}/abundances.csv", index=False)

# Diagnostic print
print(f"ρ-11fe abundance profile (Path A) — {n_shells} shells")
print(f"{'sh':>3} {'v_mid':>7}   {'X_O':>6} {'X_C':>6} {'X_Si':>6} {'X_S':>6} {'X_Ca':>6} {'X_Fe':>7} {'X_Ni':>7}")
for s in [0, 3, 6, 9, 12, 15, 20, 29]:
    v = v_mid_kms[s]
    p = {Z: X[Z_LIST.index(Z), s] for Z in Z_LIST}
    print(f"{s:>3d} {v:>7.0f}   {p[8]:>6.3f} {p[6]:>6.3f} {p[14]:>6.3f} "
          f"{p[16]:>6.3f} {p[20]:>6.3f} {p[26]:>7.1e} {p[28]:>7.1e}")
PYEOF

echo "=== ρ-11fe Path (A) [${label}]  v_in=10400 physical  +variant=${variant} ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"

ENV_CMD=(
  "LUMINA_BF_OPACITY=1"
  "LUMINA_CMFGEN_SIGMA_BF=$ROOT/data/atomic/cmfgen_sigma_bf.bin"
  "LUMINA_DYNAMIC_TRANSPROB=1"
  "LUMINA_NLTE_START_ITER=5"
)
if [ "$variant" = "skipSi" ]; then
  ENV_CMD+=("LUMINA_NLTE_SKIP_Z=14")
fi

env "${ENV_CMD[@]}" "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -6
echo "Done: $(date)"
