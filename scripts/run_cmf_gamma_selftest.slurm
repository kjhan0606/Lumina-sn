#!/bin/bash
#SBATCH --job-name=CMF_GAMMA_TRANSPORT
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gamma-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gamma-%j.err
set -euo pipefail
umask 027
gamma_packets=${1:-200000}
[[ "$gamma_packets" =~ ^[0-9]+$ && "$gamma_packets" -ge 2 && "$gamma_packets" -le 2000000 ]]
gamma_mode=${2:-all}
gamma_args=()
case "$gamma_mode" in all) ;; sources) gamma_args=(sources) ;; *) exit 2 ;; esac
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 python/3.11.14
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
gamma_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
gamma_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gamma-${SLURM_JOB_ID}
mkdir "$gamma_out"
mkdir "$gamma_out/source" "$gamma_out/input"
cd "$gamma_src"
gamma_files=(src/*.h src/cmf_gamma_transport.c src/cmf_gamma_source.c src/cmf_ejecta.c src/cmf_decay.c
 tests/cmf_gamma_transport_selftest.c scripts/run_cmf_gamma_selftest.slurm scripts/compare_cmf_gamma.py)
cp --parents "${gamma_files[@]}" "$gamma_out/source/"
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/input_models/snia_toy06_2d.dat "$gamma_out/input/"
cp /gpfs/kjhan/cmfgen_21jun23/atomic/rad_decay_data/NUC_DECAY_DATA_ni56_co56_fe56 "$gamma_out/input/"
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06/edep_toy06_cmfgen.txt \
 /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06/lbol_edep_toy06_cmfgen.txt "$gamma_out/input/"
git rev-parse HEAD > "$gamma_out/commit.txt"
(cd "$gamma_out/input"; sha256sum -- * > "$gamma_out/input.sha256")
cd "$gamma_out/source"
sha256sum "${gamma_files[@]}" > "$gamma_out/source.sha256"
gamma_units=(tests/cmf_gamma_transport_selftest.c src/cmf_gamma_transport.c src/cmf_gamma_source.c src/cmf_ejecta.c src/cmf_decay.c)
gcc -O2 -std=gnu11 -Isrc "${gamma_units[@]}" -lm -o "$gamma_out/gamma_test" > "$gamma_out/build.log" 2>&1
(cd "$gamma_out"; ./gamma_test input/snia_toy06_2d.dat input/NUC_DECAY_DATA_ni56_co56_fe56 "$gamma_packets" "${gamma_args[@]}" > test.log 2> test.err)
(cd "$gamma_out"; python3 source/scripts/compare_cmf_gamma.py "${gamma_args[@]}" > comparison.log 2> comparison.err)
gcc -O1 -g -std=gnu11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 "${gamma_units[@]}" -lm -o "$gamma_out/gamma_san" > "$gamma_out/san-build.log" 2>&1
(cd "$gamma_out"; ASAN_OPTIONS=detect_leaks=1 ./gamma_san > san.log 2> san.err)
sha256sum -c "$gamma_out/source.sha256" > "$gamma_out/source_verify.log"
(cd "$gamma_out/input"; sha256sum -c "$gamma_out/input.sha256" > "$gamma_out/input_verify.log")
