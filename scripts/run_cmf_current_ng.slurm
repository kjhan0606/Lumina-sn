#!/bin/bash
#SBATCH --job-name=CMF_CURRENT_NG_ASSESS
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/current-ng-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/current-ng-%j.err
set -euo pipefail
umask 027
ng_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
ng_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
ng_prior=$ng_base/ng-damped-424693
ng_out=$ng_base/current-ng-${SLURM_JOB_ID:?}
mkdir "$ng_out" "$ng_out/source"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 python/3.11.14
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$ng_src"
[[ "$(git rev-parse --show-toplevel)" == "$ng_src" ]]
cp "$ng_prior/source/original.py" "$ng_prior/source/assess_damped.py" scripts/assess_cmf_current_ng.py "$ng_out/source/"
cp "$ng_prior/ngaccel.so" "$ng_out/"
scontrol write batch_script "$SLURM_JOB_ID" "$ng_out/source/runner.slurm"
git rev-parse HEAD > "$ng_out/source_commit.txt"
cd "$ng_out"
sha256sum -c - <<'HASHES'
d55f681425d1eb627b5a5a26bedbaf788ad5b4ff6d1125ec094b6f331b49f66c  source/original.py
1093f38e9ccd31f7b78a9d48e5251a2d7e554225262ac32e6ed34002808d2a76  source/assess_damped.py
4286e89173b21925a28da5495e1c6b5dac2f8e5d2fe0d9dc7d14cfb95b96994f  ngaccel.so
HASHES
sha256sum source/* ngaccel.so source_commit.txt > source.sha256
(cd "$ng_base/toy-material-426303/input"; sha256sum -c ../input.sha256) > input-verify.log
python3 source/assess_cmf_current_ng.py > assessment.log
sha256sum -c source.sha256 > source-verify.log
sha256sum current_ng.json assessment.log input-verify.log > output.sha256
printf 'CURRENT_NG_ASSESS_DONE candidate_written=0 production_changed=0 physical_goal=NOT_JUDGED\n'
