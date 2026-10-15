#!/bin/bash
#SBATCH --job-name=CMF_TOY_NATIVE
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-native-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-native-%j.err
set -euo pipefail
umask 027
[[ $# == 2 ]]
native_build=$1
native_baseline=$2
native_root=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
[[ "$native_build" =~ ^[0-9]+$ && "$native_baseline" =~ ^[0-9]+$ ]]
native_build=$native_root/common-build-$native_build
native_baseline=$native_root/toy-material-$native_baseline
native_out=$native_root/toy-native-${SLURM_JOB_ID}
mkdir "$native_out"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while read -r native_key; do unset "$native_key"; done < <(compgen -v LUMINA_ || true)
export LUMINA_PURE_CMFGEN=1 LUMINA_CMF_COMMON_FIELD=1 LUMINA_CMF_BF_MILNE=2
export LUMINA_SUPER_LEVELS=1 LUMINA_NLTE_COLL_ION=1 LUMINA_OMEGA_CMFGEN=1
export LUMINA_BF_CHANNELS_FILE=$native_baseline/input/bf_channels.bin
export LUMINA_TOPION_LEVELS_FILE=$native_baseline/input/bf_channels.terminal.csv
export LUMINA_IONIZATION_REFERENCE_FILE=$native_baseline/input/ionization_reference.csv
(cd "$native_baseline/input"; sha256sum -c "$native_baseline/input.sha256") > "$native_out/input_verify.log"
sha256sum -c "$native_build/binary.sha256" > "$native_out/binary_verify.log"
cp "$native_build/lumina_cuda" "$native_build/commit.txt" "$native_out/"
sha256sum "$native_out/lumina_cuda" > "$native_out/binary.sha256"
env | LC_ALL=C sort | grep '^LUMINA_' > "$native_out/env_lumina_effective.txt"
cd "$native_out"
./lumina_cuda --cmf-toy-bootstrap "$native_baseline/input/snia_toy06_2d.dat" \
 "$native_baseline/input" 100 toy_initial_material.csv > prepare.log 2> prepare.err
cmp toy_initial_material.csv "$native_baseline/toy_initial_material.csv"
echo 'NATIVE_TOY_BOOTSTRAP_BYTE_IDENTICAL shells=100 spectrum_compared=0'
