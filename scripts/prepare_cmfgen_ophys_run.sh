#!/usr/bin/env bash
# Prepare (but never submit) the O-PHYS CMFGEN run directory.
set -euo pipefail
umask 027

readonly BASE=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern
readonly TARGET=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys
readonly CMF_ROOT=/gpfs/kjhan/cmfgen_src/cur_cmf
readonly REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn

[[ -d "$BASE" ]] || { echo "missing base: $BASE" >&2; exit 2; }
[[ ! -e "$TARGET" ]] || { echo "refusing to alter existing target: $TARGET" >&2; exit 3; }

stage=$(mktemp -d /gpfs/kjhan/cmfgen_runs/.toy06_19p48d_ophys.prepare.XXXXXX)
case "$stage" in
  /gpfs/kjhan/cmfgen_runs/.toy06_19p48d_ophys.prepare.*) ;;
  *) echo "unsafe staging path: $stage" >&2; exit 4 ;;
esac
cleanup() { [[ -d "$stage" ]] && rm -rf -- "$stage"; }
trap cleanup EXIT

# Inputs, documented local atomic repairs, and the minimum compatible restart.
inputs=(
  VADAT IN_ITS MODEL_SPEC SN_HYDRO_DATA MODEL SPECIES_MASSES
  LEVEL_SL_STEQ_LINKS model_spec_isf.txt atomic_links.txt setup_links.sh
  PROVENANCE.txt INTENDED_DIFF_MANIFEST.txt MODEL_SPEC.base_reference
  RUNTIME_ESTIMATE.txt PHOT_PRESCAN.txt PREFLIGHT.txt SIGMA_REPAIR_CHECK.txt
  gen_atomic_base_reference.py gen_atomic_modern.py mk_atomic_local_repairs.py
  mk_sn_hydro.py phot_prescan.py preflight_lib.py preflight_run.py
  sigma_repair_check.py snia_toy06_19.48d.dat
)
restart=(SCRTEMP POINT1 POINT2 EDDFACTOR EDDFACTOR_INFO CONT_FREQ CUR_MODEL_DATA STEQ_VALS)
for name in "${inputs[@]}" "${restart[@]}"; do
  [[ -e "$BASE/$name" ]] || { echo "missing required base object: $name" >&2; exit 5; }
  cp -a -- "$BASE/$name" "$stage/"
done
cp -a -- "$BASE/atomic_local" "$stage/"
mkdir -p "$stage/seq_logs/captures" "$stage/seq_logs/tools"

# setup_links.sh has two deliberate run-local phot_data repairs. Retarget only
# those paths to this new clone; every other link remains byte-for-byte lineage.
sed -i "s|$BASE/atomic_local|$TARGET/atomic_local|g; s|cd $BASE|cd $TARGET|" \
  "$stage/setup_links.sh"

# O-PHYS begins with the first post-it40 continuation: T is released immediately.
# The 40 fixed-T base iterations already provide the requested Lambda precondition.
python3 - "$stage/VADAT" <<'PY'
from pathlib import Path
import re, sys

p = Path(sys.argv[1])
text = p.read_text()

def replace(key: str, value: str, comment: str) -> None:
    global text
    pat = re.compile(rf"^.*\[{re.escape(key)}\].*$", re.MULTILINE)
    matches = pat.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one [{key}], found {len(matches)}")
    text = pat.sub(f"{value:<13}[{key}]          ! {comment}", text)

replace("FIX_T", "F", "O-PHYS: solve electron temperature at every depth")
replace("FIX_T_AUTO", "F", "O-PHYS: no automatic depth freeze")
replace("MAX_LIN", "1.01D0", "O-PHYS 1-percent full-linearization trust radius")
replace("MAX_LAM", "1.10D0", "O-PHYS 10-percent Lambda safety-step cap")
replace("NUM_LAM", "2", "retain parent/rel-T safety-step cadence")

for key, value, comment in (
    ("MAX_dT", "1.0D-02", "O-PHYS 1-percent temperature-correction cap"),
    ("WRITE_RATES", "T", "NETRATE/TOTRATE/EWDATA/LINEHEAT on final iteration"),
    ("WRITE_JH", "T", "retain JH_AT_CURRENT_TIME explicitly"),
):
    if re.search(rf"\[{key}\]", text):
        replace(key, value, comment)
    else:
        text += f"\n{value:<13}[{key}]          ! {comment}\n"
p.write_text(text)
PY

# One normal continuation allocation. Capture mode later replaces NUM_ITS with 1.
cat >"$stage/IN_ITS" <<'EOF'
80           [NUM_ITS]                  ! O-PHYS free-T continuation budget
F            [DO_LAM_IT]                ! request a full coupled solve first
EOF

# cmf_flux controls are copied from the proven full-key base and narrowed only
# by disabling the unrequired depth-flux file; WR_ETA remains true.
cp -a -- /gpfs/kjhan/cmfgen_runs/toy06_19.48d_cmfflux/CMF_FLUX_PARAM \
  "$stage/CMF_FLUX_PARAM"
sed -i 's/^T[[:space:]]*\[WR_FLUX\]/F            [WR_FLUX]/' "$stage/CMF_FLUX_PARAM"
sed -i 's/^T[[:space:]]*\[COMP_F\].*$/F            [COMP_F]           ! Reuse converged O-PHYS Eddington factors/' "$stage/CMF_FLUX_PARAM"
grep -Eq '^T[[:space:]]+\[WR_ETA\]' "$stage/CMF_FLUX_PARAM"
grep -Eq '^F[[:space:]]+\[WR_FLUX\]' "$stage/CMF_FLUX_PARAM"
grep -Eq '^F[[:space:]]+\[COMP_F\]' "$stage/CMF_FLUX_PARAM"
cat >"$stage/CMF_FLUX_STDIN" <<'EOF'
RVTJ         [RVTJ]
1.0          [MASS]
F            [ONLY_OBS_LINES]
EOF

cp -a -- "$REPO/scripts/submit_cmfgen_ophys.slurm" "$stage/"
cp -a -- "$REPO/scripts/package_cmfgen_ophys_attestation.py" "$stage/"
cp -a -- "$REPO/scripts/cmfgen_oracle_contract.py" "$stage/seq_logs/tools/"
cp -a -- "$REPO/docs/A2_00_OPHYS_PROFILE.json" "$stage/seq_logs/tools/"

cat >"$stage/PROVENANCE_OPHYS.txt" <<'EOF'
O-PHYS run root: /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys
Parent: /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern, completed fixed-T iteration 40.
Purpose: temperature-solved, freeze-zero, diagnostic-complete CMFGEN oracle.
Parent PROVENANCE.txt remains authoritative for the 19apr23 atomic selection,
80 links, f_to_s choices, and the two run-local phot_data repairs.
Only the staged clone is changed: FIX_T=F; conservative MAX_LIN/MAX_LAM;
WRITE_RATES=T; WRITE_JH=T; WR_ETA=T; WR_FLUX=F; COMP_F=F; IN_ITS continuation budget.
The parent run, CMFGEN source, executable, and canonical atomic tree are untouched.
EOF

cat >"$stage/RUNTIME_OPHYS_ESTIMATE.txt" <<'EOF'
Slurm wall clock: 72:00:00, OMP=16, one exclusive 256-GiB grammar node.
Basis: the fixed-T 40-iteration parent completed in about 11 h; released-T
linearizations and safety retries can cost materially more, while cmf_flux has
previously required about 1.5 h. 72 h covers an 80-iteration continuation plus
formal-output margin. It is a scheduling envelope, not a convergence claim.
EOF

# Freeze-zero input audit. FIX_BA is a matrix reuse threshold, not a physical
# ion/level freeze and is intentionally excluded; all population FIX_* keys must 0.
python3 - "$stage/VADAT" <<'PY'
from pathlib import Path
import re, sys

bad = []
for line in Path(sys.argv[1]).read_text().splitlines():
    m = re.search(r"^\s*([^!\s]+).*\[(FIX_[A-Za-z0-9]+)\]", line)
    if not m:
        continue
    value, key = m.groups()
    if key in {"FIX_T", "FIX_T_AUTO", "FIX_NE", "FIX_IMP"}:
        if value.upper() not in {"F", ".FALSE."}:
            bad.append((key, value))
    elif key == "FIX_BA":
        continue
    else:
        try:
            if float(value.replace("D", "E").replace("d", "e")) != 0.0:
                bad.append((key, value))
        except ValueError:
            bad.append((key, value))
if bad:
    raise SystemExit(f"nonzero/unknown freeze controls: {bad}")
PY

# Materialize all generic/atomic symlinks while the staging path is temporary.
# setup_links names TARGET in its cd command, so temporarily substitute stage.
sed -i "s|cd $TARGET|cd $stage|" "$stage/setup_links.sh"
bash "$stage/setup_links.sh"
sed -i "s|cd $stage|cd $TARGET|" "$stage/setup_links.sh"

while IFS= read -r -d '' link; do
  if [[ ! -e "$link" ]]; then
    case "$(basename "$link")" in
      PHOTSIII_A) [[ "$(readlink "$link")" == "$TARGET/atomic_local/SUL/III/19apr23/phot_data_A" ]] && continue ;;
      PHOTCo2_A) [[ "$(readlink "$link")" == "$TARGET/atomic_local/COB/II/19apr23/phot_data_A" ]] && continue ;;
    esac
    echo "broken symlink: $link -> $(readlink "$link")" >&2
    exit 6
  fi
done < <(find "$stage" -maxdepth 1 -type l -print0)

# Reject undeclared external population/freeze vectors.
if find "$stage" -maxdepth 1 -type f \
    \( -name 'XzV_IN' -o -name 'XzV_IN_*' -o -name 'POP*_IN' \) -print -quit | grep -q .; then
  echo "unexpected external population/freeze vector" >&2
  exit 7
fi

mv -- "$stage" "$TARGET"
trap - EXIT
echo "prepared only: $TARGET"
echo "no Slurm job was submitted"
