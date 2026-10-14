#!/bin/bash
#SBATCH --job-name=CMF_PHOTO_ROUTES
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/photo-routes-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/photo-routes-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
route_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
route_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/photo-routes-${SLURM_JOB_ID}
mkdir "$route_out" "$route_out/source" "$route_out/atomic"
cd "$route_src"
route_files=(scripts/cmfgen_parser.py scripts/cmfgen_photo_routes.py
 scripts/run_cmf_photo_routes_selftest.slurm tests/cmf_photo_routes_selftest.py
 tests/cmf_photo_route_oracle.f90)
cp --parents "${route_files[@]}" "$route_out/source/"
git rev-parse HEAD > "$route_out/commit.txt"
cd "$route_out/source"
sha256sum "${route_files[@]}" > "$route_out/source.sha256"
route_atomic=/gpfs/kjhan/cmfgen_21jun23/atomic
for route_spec in SIL/II/phot_data_A SIL/II/phot_data_B SIL/II/osc_data \
 SIL/III/osc_data SIL/III/f_to_s_ls SUL/II/phot_data_A SUL/II/osc_data \
 SUL/III/osc_data SUL/III/f_to_s_127; do
 route_dir=${route_spec%/*}
 route_name=${route_spec##*/}
 mkdir -p "$route_out/atomic/$route_dir"
 cp "$route_atomic/$route_dir/19apr23/$route_name" "$route_out/atomic/$route_spec"
done
cp /gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/get_num_phot_routes_v1.f "$route_out/reference.f"
cd "$route_out"
sha256sum reference.f atomic/SIL/II/* atomic/SIL/III/* atomic/SUL/II/* atomic/SUL/III/* > inputs.sha256
gfortran -O2 -fcheck=all -c source/tests/cmf_photo_route_oracle.f90 -o adapter.o > build.log 2>&1
gfortran -O2 -fcheck=all -ffixed-line-length-none -c reference.f -o reference.o >> build.log 2>&1
gfortran adapter.o reference.o -o oracle >> build.log 2>&1
/home/kjhan/miniconda3/bin/python3 source/tests/cmf_photo_routes_selftest.py \
 --atomic "$route_out/atomic" --oracle "$route_out/oracle" > test.log 2> test.err
sha256sum -c inputs.sha256 > input_verify.log
cd "$route_out/source"
sha256sum -c "$route_out/source.sha256" > "$route_out/source_verify.log"
