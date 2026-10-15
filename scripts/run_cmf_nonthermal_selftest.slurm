#!/bin/bash
#SBATCH --job-name=CMF_NONTHERMAL
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/nonthermal-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/nonthermal-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 python/3.11.14
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nt_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
nt_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/nonthermal-${SLURM_JOB_ID}
mkdir "$nt_out"
mkdir "$nt_out/source" "$nt_out/source/src" "$nt_out/source/tests" "$nt_out/source/scripts" "$nt_out/input"
nt_files=(src/cmf_nonthermal.h src/cmf_nonthermal.c src/cmf_nonthermal_atomic.c tests/cmf_nonthermal_selftest.c scripts/run_cmf_nonthermal_selftest.slurm scripts/map_cmf_nonthermal.py)
for nt_file in "${nt_files[@]}"; do cp "$nt_src/$nt_file" "$nt_out/source/$nt_file"; done
cp /gpfs/kjhan/cmfgen_21jun23/atomic/misc/arnaud_rothenflug_23jun15 "$nt_out/input/arnaud_rothenflug_23jun15"
cp /gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model/levels.csv /gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model/ionization_energies.csv "$nt_out/input/"
cp /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/bf-input-423197/bf_channels.terminal.csv "$nt_out/input/"
git -C "$nt_src" rev-parse HEAD > "$nt_out/commit.txt"
(cd "$nt_out/input"; sha256sum -- * > "$nt_out/input.sha256")
cd "$nt_out/source"
sha256sum "${nt_files[@]}" > "$nt_out/source.sha256"
nt_units=(tests/cmf_nonthermal_selftest.c src/cmf_nonthermal.c src/cmf_nonthermal_atomic.c)
gcc -O2 -std=gnu11 -Wall -Wextra -Isrc "${nt_units[@]}" -lm -o "$nt_out/nt_test" > "$nt_out/build.log" 2>&1
(cd "$nt_out"; stdbuf -oL ./nt_test input/arnaud_rothenflug_23jun15 > test.log 2> test.err)
(cd "$nt_out"; python3 source/scripts/map_cmf_nonthermal.py > mapping.log 2> mapping.err)
gcc -O1 -g -std=gnu11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 "${nt_units[@]}" -lm -o "$nt_out/nt_san" > "$nt_out/san-build.log" 2>&1
(cd "$nt_out"; ASAN_OPTIONS=detect_leaks=1 ./nt_san input/arnaud_rothenflug_23jun15 > san.log 2> san.err)
sha256sum -c "$nt_out/source.sha256" > "$nt_out/source_verify.log"
(cd "$nt_out/input"; sha256sum -c "$nt_out/input.sha256" > "$nt_out/input_verify.log")
