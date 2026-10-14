#!/bin/bash
#SBATCH --job-name=CMF_COMMON1
#SBATCH --partition=a100,h100,h200,a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-physics-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-physics-%j.err
# 새 공통 물리 실행. 형식해/fixture 통과나6회 반복 완료는 목표 달성이 아니다.
set -euo pipefail
umask 027
common_build=${1:?cluster build directory required}
common_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
common_original=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1
common_model=$common_original/input/model
common_sigma=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/atomic-stage-422722/cmfgen_sigma_bf.bin
common_base_env=/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/ada_20261014/run1/exports.env
common_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-physics-${SLURM_JOB_ID}
common_bin=$common_build/lumina_cuda
mkdir "$common_out"
mkdir "$common_out/work"
common_start=$(date +%s)
finish() {
    local rc=$?
    trap - EXIT
    printf 'model_and_output_rc=%d elapsed_seconds=%d job=%s\n' "$rc" \
      "$(( $(date +%s)-common_start ))" "$SLURM_JOB_ID" > "$common_out/RUN_FOOTER.txt"
}
trap finish EXIT
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
grep -Fxq 'variant=cluster managed=0' "$common_build/variant.txt"
sha256sum -c "$common_build/binary.sha256" > "$common_out/binary_verify.log"
( cd "$common_src"; sha256sum -c "$common_build/source.sha256" ) > "$common_out/source_verify.log"
( cd "$common_model"; sha256sum -c "$common_original/input/deck.sha256" ) > "$common_out/deck_verify.log"
[[ $(sha256sum "$common_sigma") == "ee7d689e14f5426f6d1dc28fa31a0d5727b9eac41d0157bdcb2b8c0d6e9a0635  $common_sigma" ]]
[[ $(sha256sum "$common_base_env") == "923c3ad35f0c5df373f74ac0189bf8010e4485b419df189c902d821698d9c71a  $common_base_env" ]]
while read -r common_key; do unset "$common_key"; done < <(compgen -v LUMINA_ || true)
source "$common_base_env"
export LUMINA_CMFGEN_SIGMA_BF=$common_sigma
export LUMINA_PHYSICS_COMPARISON_DIR=$common_out/work/physics_comparison
export LUMINA_CMF_COMMON_FIELD=1 LUMINA_CMF_DEP_SOURCE=0 LUMINA_PURE_CMFGEN_ITER=6
export LUMINA_CMF_FINE_MGPU_DEVICES=1
# Sobolev 전용 추적은 공통 Gaussian producer에 적용하지 않는다.
export LUMINA_A210_LINE_SATURATION_DIAG=0 LUMINA_A210_SPRODUCER_CAPTURE=0
export LUMINA_A210_INDEPENDENT_CAPTURE=0 LUMINA_DET_OBS_CAPTURE=0 LUMINA_DET_OBS_WRITER=0
[[ -z ${LUMINA_FIXED_TE_PROFILE+x} && $LUMINA_NLTE_NE_CLOSURE == 1 &&
   $LUMINA_CMF_FINE_VDOP == 1000000 && $LUMINA_CMF_FINE_PPD == 12 &&
   $LUMINA_MODEL_DIR == "$common_model" ]]
env | LC_ALL=C sort | grep '^LUMINA_' > "$common_out/env_lumina_effective.txt"
printf 'build=%s\nsource=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$common_build" "$common_src" \
 "${CUDA_VISIBLE_DEVICES:-UNSET}" > "$common_out/identity.txt"
cp "$common_build/commit.txt" "$common_build/binary.sha256" "$common_out/"
cd "$common_out/work"
# 같은 실행 바이너리의 소형 GPU owner 확인. 다른 모델/CPU로 자동 대체하지 않는다.
LUMINA_CMF_EXACT_OWNER_SELFTEST=1 "$common_bin" > "$common_out/owner.log" 2> "$common_out/owner.err"
printf 'COMMON1_START time=%s root=%s\n' "$(date -Is)" "$common_out"
( ulimit -f 1048576; exec "$common_bin" "$common_model" 1 2 spectrum nlte ) \
 > "$common_out/stdout.log" 2> "$common_out/stderr.log"
python3 - "$common_out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); work=root/'work'
obs=json.loads((work/'lumina_spectrum_common_obs.json').read_text())
final=json.loads((work/'cmfgen_final_publication.json').read_text())
material=json.loads((work/'physics_comparison/physics_DET_iter0005.manifest.json').read_text())
assert obs['material_iteration']==5
assert obs['material_generation']==final['population_generation']==material['population_generation']
assert obs['radiation_generation']==material['radiation_generation']+1
assert obs['outer_convergence_verified'] is False
print('COMMON1_OUTPUT_IDENTITY_OK population_generation=',obs['material_generation'],
      'radiation_generation=',obs['radiation_generation'],'physical_agreement=NOT_YET_EVALUATED')
PY
printf 'COMMON1_FINISH time=%s physical_agreement=NOT_YET_EVALUATED\n' "$(date -Is)"
