#!/bin/bash
#SBATCH --job-name=CMF_TOY_INITIAL
#SBATCH --partition=a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-%j.err
set -euo pipefail
umask 027
initial_build=${1:?cluster build directory required}
initial_input=${2:?verified initial-prepare input directory required}
initial_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
initial_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-initial-${SLURM_JOB_ID}
mkdir "$initial_out"
mkdir "$initial_out/input" "$initial_out/work"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
grep -Fxq 'variant=cluster managed=0' "$initial_build/variant.txt"
(cd "$initial_build/source"; sha256sum -c "$initial_build/source.sha256") > "$initial_out/build_source_verify.log"
sha256sum -c "$initial_build/binary.sha256" > "$initial_out/binary_verify.log"
# 실행되는 것은 위에서 검증한 봉인 binary/source다. 이후 작업 트리 편집이
# 이전 실행의 재현을 막지 않도록 현재 작업 트리와의 동일성은 요구하지 않는다.
(cd "$initial_input/input"; sha256sum -c "$initial_input/input.sha256") > "$initial_out/original_input_verify.log"
# 복사 대상은 이전 sealed 입력 manifest의 단일 디렉터리 파일만이다.
while read -r initial_hash initial_file; do
 [[ "$initial_file" != */* && "$initial_file" != .* ]]
 cp "$initial_input/input/$initial_file" "$initial_out/input/$initial_file"
done < "$initial_input/input.sha256"
cp "$initial_input/input.sha256" "$initial_build/commit.txt" "$initial_build/binary.sha256" "$initial_out/"
cp "$initial_src/scripts/run_cmf_toy_initial.slurm" "$initial_out/runner.slurm"
sha256sum "$initial_out/runner.slurm" > "$initial_out/runner.sha256"
(cd "$initial_out/input"; sha256sum -c "$initial_out/input.sha256") > "$initial_out/input_verify.log"
while read -r initial_key; do unset "$initial_key"; done < <(compgen -v LUMINA_ || true)
export LUMINA_PURE_CMFGEN=1 LUMINA_CMF_COMMON_FIELD=1 LUMINA_CMF_BF_MILNE=2
export LUMINA_SUPER_LEVELS=1 LUMINA_NLTE_COLL_ION=1 LUMINA_OMEGA_CMFGEN=1
export LUMINA_NLTE_NE_CLOSURE=1 LUMINA_CMF_FINE_VDOP=5000000 LUMINA_CMF_FINE_PPD=3
export LUMINA_CMF_FINE_ALI=768 LUMINA_CMF_FINE_TOL=1e-8 LUMINA_CMF_FINE_PHOTOION=1
export LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=42 LUMINA_CMF_FINE_MGPU_DEVICES=1
export LUMINA_BF_CHANNELS_FILE=$initial_out/input/bf_channels.bin
export LUMINA_TOPION_LEVELS_FILE=$initial_out/input/bf_channels.terminal.csv
export LUMINA_IONIZATION_REFERENCE_FILE=$initial_out/input/ionization_reference.csv
env | LC_ALL=C sort | grep '^LUMINA_' > "$initial_out/env_lumina_effective.txt"
printf 'mode=ALL_THERMAL_INITIAL_CONTROL\nsource=%s\nbuild=%s\ninput=%s\n' \
 "$initial_build/source" "$initial_build" "$initial_out/input" > "$initial_out/identity.txt"
cd "$initial_out/work"
# 2GiB/file 상한. 중간 raw dump는 생성하지 않고 수렴한 물리 이력만 저장한다.
(ulimit -f 2097152; exec "$initial_build/lumina_cuda" --cmf-toy-initial \
 "$initial_out/input/snia_toy06_2d.dat" "$initial_out/input" \
 "$initial_out/input/NUC_DECAY_DATA_ni56_co56_fe56" 100 200000 128) \
 > "$initial_out/initial.log" 2> "$initial_out/initial.err"
sha256sum initial_state.bin initial_converged_material.csv initial_spectrum.csv initial_spectrum.json \
 > "$initial_out/output.sha256"
echo 'INITIAL_COMPLETE: 2-day all-thermal control only; final spectrum goal not achieved'
