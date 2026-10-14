#!/bin/bash
#SBATCH --job-name=CMF_OBSERVER_TEST
#SBATCH --partition=a10,a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/observer-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/observer-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
observer_src=${3:-/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015}
observer_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/observer-${SLURM_JOB_ID}
mkdir "$observer_out"
cd "$observer_src"
observer_files=(src/cmf_observer.h src/cmf_observer_ray.h src/cmf_observer.c
 src/cmf_observer.cu tests/cmf_observer_selftest.c scripts/run_cmf_observer_selftest.slurm)
git rev-parse HEAD > "$observer_out/commit.txt"
mkdir "$observer_out/source"
cp --parents "${observer_files[@]}" "$observer_out/source/"
cd "$observer_out/source"
sha256sum "${observer_files[@]}" > "$observer_out/source.sha256"
if [[ ${1:-cpu} == gpu ]]; then
    module load cuda/13.0.2
    nvcc -O2 -arch=sm_86 -std=c++14 -DOBSERVER_TEST_GPU -Isrc \
      tests/cmf_observer_selftest.c src/cmf_observer.c src/cmf_observer.cu \
      -lm -o "$observer_out/test" > "$observer_out/build.log" 2>&1
else
    gcc -O2 -Wall -Wextra -Werror -std=c11 -fopenmp -Isrc \
      tests/cmf_observer_selftest.c src/cmf_observer.c -lm \
      -o "$observer_out/test" > "$observer_out/build.log" 2>&1
fi
sha256sum -c "$observer_out/source.sha256" > "$observer_out/verify.log"
cd "$observer_out"
./test > test.log 2> test.err
if [[ ${1:-cpu} == gpu ]]; then
    compute-sanitizer --tool memcheck --error-exitcode 99 ./test > memcheck.log 2>&1
    if [[ ${2:-} == bench ]]; then ./test --bench > benchmark.log 2> benchmark.err; fi
fi
echo "result=$observer_out"
