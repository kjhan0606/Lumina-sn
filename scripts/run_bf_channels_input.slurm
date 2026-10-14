#!/bin/bash
#SBATCH --job-name=CMF_BF_CHANNEL_INPUT
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/bf-input-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/bf-input-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
while read -r bf_key; do unset "$bf_key"; done < <(compgen -v LUMINA_ || true)
while read -r bf_key; do unset "$bf_key"; done < <(compgen -v CMFGEN_ || true)
bf_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
bf_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/bf-input-${SLURM_JOB_ID}
bf_deck=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model
mkdir "$bf_out" "$bf_out/source" "$bf_out/reference" "$bf_out/hyd"
cd "$bf_src"
bf_files=(scripts/build_bf_channels.py scripts/cmfgen_photo_routes.py scripts/cmfgen_parser.py
 scripts/expand_atomic_data_cmfgen.py scripts/cmfgen_smooth_driver.f90 scripts/run_bf_channels_input.slurm
 tests/bf_channels_input_probe.c src/lumina_atomic.c src/*.h)
cp --parents "${bf_files[@]}" "$bf_out/source/"
git rev-parse HEAD > "$bf_out/commit.txt"
cd "$bf_out/source"
sha256sum "${bf_files[@]}" > "$bf_out/source.sha256"
bf_reference=/gpfs/kjhan/cmfgen_src/cur_cmf
cp "$bf_reference/tools/sm_phot_v3.f" "$bf_reference/newsubs/cut_points_v3.f" \
 "$bf_reference/newsubs/sub_phot_gen.f" "$bf_reference/newsubs/rdphot_gen_v2.f" \
 "$bf_reference/new_main/mod_subs/rd_control_variables.f" "$bf_out/reference/"
cp /gpfs/kjhan/cmfgen_21jun23/atomic/HYD/I/5dec96/hyd_l_data.dat \
 /gpfs/kjhan/cmfgen_21jun23/atomic/HYD/I/5dec96/gbf_n_data.dat "$bf_out/hyd/"
cp /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/VADAT "$bf_out/VADAT"
cd "$bf_out"
sha256sum reference/* hyd/* VADAT > reference.sha256
gfortran -O2 -fcheck=all -c source/scripts/cmfgen_smooth_driver.f90 -o smooth-driver.o > build.log 2>&1
gfortran -O2 -fcheck=all -ffixed-line-length-none -c reference/sm_phot_v3.f reference/cut_points_v3.f >> build.log 2>&1
gfortran smooth-driver.o sm_phot_v3.o cut_points_v3.o -o smooth >> build.log 2>&1
/home/kjhan/miniconda3/bin/python3 source/scripts/build_bf_channels.py --deck "$bf_deck" \
 --output "$bf_out/bf_channels.bin" --control VADAT --smoother "$bf_out/smooth" \
 --hyd-data "$bf_out/hyd" --ion 14:1 --ion 16:1 > bake.log 2> bake.err
gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer \
 -ffunction-sections -fdata-sections -Isource/src source/tests/bf_channels_input_probe.c \
 source/src/lumina_atomic.c -Wl,--gc-sections -lm -o input-probe >> build.log 2>&1
ASAN_OPTIONS=detect_leaks=1 ./input-probe "$bf_deck/levels.csv" "$bf_deck/ionization_energies.csv" \
 bf_channels.bin > input-probe.log 2> input-probe.err
sha256sum -c reference.sha256 > reference_verify.log
cd "$bf_out/source"
sha256sum -c "$bf_out/source.sha256" > "$bf_out/source_verify.log"
