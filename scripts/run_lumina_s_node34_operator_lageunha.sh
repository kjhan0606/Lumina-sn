#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
parent="$1"
test -d "$parent"
test ! -e "$parent/source"
mkdir "$parent/source" "$parent/build"
git -C "$repo" archive HEAD src scripts data validation/cmfgen_toy06_19p48d/analysis | tar -x -C "$parent/source"
for name in src/nlte_model_material_bridge.c src/nlte_model_material_bridge.h scripts/compare_lumina_s_node34_operator.py; do
  cp -- "$repo/$name" "$parent/source/$name"
done
cp -- "$repo/data/atomic/ionization_reference.csv" "$parent/source/data/atomic/ionization_reference.csv"
cd "$parent/source"
sha256sum src/nlte_model_material_bridge.c src/nlte_model_material_bridge.h scripts/compare_lumina_s_node34_operator.py data/atomic/ionization_reference.csv > "$parent/source-sha256.txt"
core=(lumina_plasma lumina_element_wide nlte_model_se nlte_model_bb nlte_model_coll_bb population_contract opacity_publication emissivity_publication radeq_publication line_net_rate cmfgen_adiabatic bf_rate_jnu radiation_field seed_capability atomic_internal_energy nlte_population_candidate line_jbar lumina_atomic nlte_model_material_bridge)
objects=()
for name in "${core[@]}"; do
  gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_OPERATOR_PROBE -fopenmp -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections -Isrc -c "src/$name.c" -o "$parent/build/$name.o"
  objects+=("$parent/build/$name.o")
done
gcc -shared -fopenmp -Wl,--gc-sections -Wl,-z,defs -o "$parent/build/bridge-probe.so" "${objects[@]}" -lm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LUMINA_ARTIS_PARITY=0 LUMINA_NLTE_BF_COLLISION=1
export LUMINA_TOPION_LEVELS_FILE="$parent/source/data/atomic/topion_levels.csv"
export LUMINA_IONIZATION_REFERENCE_FILE="$parent/source/data/atomic/ionization_reference.csv"
export PYTHONPATH="$parent/source/scripts"
python scripts/compare_lumina_s_node34_operator.py --library "$parent/build/bridge-probe.so" --out "$parent/result"
