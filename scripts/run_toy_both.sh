#!/bin/bash
#SBATCH --job-name=toyboth
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/toy/%x_%j.out
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; cd $ROOT
T=$ROOT/${TOYDIR}; OUT=logs/toy/${TAG}; mkdir -p $OUT
COMMON="LUMINA_PURE_CMFGEN=1 LUMINA_PURE_CMFGEN_ITER=6 LUMINA_BF_OPACITY=1 LUMINA_CMFGEN_SIGMA_BF=$T/cmfgen_sigma_bf.bin LUMINA_NLTE_COLL_FIX=1 LUMINA_NLTE_ION_LOCK=1 LUMINA_FIXED_TE_PROFILE=$T/te_profile.txt LUMINA_FIXED_NE_PROFILE=$T/ne_profile.txt"
# (1) pure-CMFGEN
mkdir -p $OUT/cmf; cd $OUT/cmf
env $COMMON $ROOT/lumina_cuda $T 5000 6 virtual nlte > stdout.log 2>stderr.log
cp lumina_spectrum.csv ../cmf_spectrum.csv 2>/dev/null; cp lumina_spectrum_formal.csv ../cmf_formal.csv 2>/dev/null
cd $ROOT
# (2) THEN_MC macro-atom
mkdir -p $OUT/mc; cd $OUT/mc
env $COMMON LUMINA_CMFGEN_THEN_MC=1 LUMINA_LINE_INTERACTION=macroatom $ROOT/lumina_cuda $T 200000 6 virtual nlte > stdout.log 2>stderr.log
cp lumina_spectrum.csv ../mc_spectrum.csv 2>/dev/null
echo "done TAG=$TAG"
