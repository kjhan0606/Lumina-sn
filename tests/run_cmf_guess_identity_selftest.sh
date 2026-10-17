#!/usr/bin/env bash
set -euo pipefail
# 셸 신원 검사만 한다. 계산/모듈 로드/생산 출력/GPU 접근은 없다.
gi_src=$(cd -- "$(dirname -- "$0")/.." && pwd)
source <(sed -n '/^check_refinement_guess_identity()/,/^}/p' "$gi_src/scripts/run_cmf_toy_initial.slurm")
source <(sed -n '/^check_accelerated_guess_identity()/,/^}/p' "$gi_src/scripts/run_cmf_toy_initial.slurm")
gi_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
gi_old=$gi_base/baseline-425609-20261017.45Maft/iterate_identity.txt
gi_one=$gi_base/first-field-425639-20261017.oUMXEZ/iterate_identity.txt
gi_new=$gi_base/ada-evolution-20261017-0522/initial/iterate_identity.txt
gi_tmp=$(mktemp -d /tmp/cmf-guess-identity.XXXXXX)
# 이 시험이 만든 명시적 작은 fixture만 제거한다.
trap 'rm -f -- "$gi_tmp/contract" "$gi_tmp/duplicate" "$gi_tmp/missing" "$gi_tmp/optimized" "$gi_tmp/accelerated" "$gi_tmp/aa_depth" "$gi_tmp/aa_duplicate" "$gi_tmp/aa_contract"; rmdir -- "$gi_tmp"' EXIT
gi_cases=0
reject() {
 if check_refinement_guess_identity "$@" >/dev/null 2>&1; then
  printf 'unexpected acceptance: %s\n' "$*" >&2
  exit 1
 fi
 gi_cases=$((gi_cases+1))
}
accept() { check_refinement_guess_identity "$@"; gi_cases=$((gi_cases+1)); }
accept "$gi_one" "$gi_new" 42 0
accept "$gi_old" "$gi_new" 42 1
reject "$gi_old" "$gi_new" 42 0
reject "$gi_one" "$gi_new" 42 1
reject "$gi_old" "$gi_new" 1 1
reject "$gi_new" "$gi_old" 42 1
reject "$gi_new" "$gi_new" 42 1
reject "$gi_old" "$gi_new" 42 2
sed 's/^outer_tol=.*/outer_tol=0.1/' "$gi_new" > "$gi_tmp/contract"
sed '/^LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=/p' "$gi_new" > "$gi_tmp/duplicate"
sed '/^LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=/d' "$gi_new" > "$gi_tmp/missing"
reject "$gi_old" "$gi_tmp/contract" 42 1
reject "$gi_old" "$gi_tmp/duplicate" 42 1
reject "$gi_old" "$gi_tmp/missing" 42 1
gi_binary=$(cut -d' ' -f1 "$gi_base/common-build-425672/binary.sha256")
[[ "$gi_binary" =~ ^[0-9a-f]{64}$ ]]
sed "1s/.*/$gi_binary/" "$gi_new" > "$gi_tmp/optimized"
accept "$gi_old" "$gi_tmp/optimized" 42 1
sed '/^LUMINA_CMF_ANDERSON1=/a LUMINA_CMF_ANDERSON_DEPTH=4' "$gi_tmp/optimized" > "$gi_tmp/accelerated"
aa_accept() { check_accelerated_guess_identity "$@"; gi_cases=$((gi_cases+1)); }
aa_reject() {
 if check_accelerated_guess_identity "$@" >/dev/null 2>&1; then
  printf 'unexpected accelerated acceptance: %s\n' "$*" >&2; exit 1
 fi
 gi_cases=$((gi_cases+1))
}
aa_accept "$gi_old" "$gi_tmp/accelerated"
aa_accept "$gi_new" "$gi_tmp/accelerated"
aa_reject "$gi_tmp/optimized" "$gi_tmp/accelerated"
aa_reject "$gi_old" "$gi_tmp/optimized"
aa_reject "$gi_tmp/accelerated" "$gi_new"
sed 's/^LUMINA_CMF_ANDERSON_DEPTH=4$/LUMINA_CMF_ANDERSON_DEPTH=2/' "$gi_tmp/accelerated" > "$gi_tmp/aa_depth"
sed '/^LUMINA_CMF_ANDERSON_DEPTH=/p' "$gi_tmp/accelerated" > "$gi_tmp/aa_duplicate"
sed 's/^outer_tol=.*/outer_tol=0.1/' "$gi_tmp/accelerated" > "$gi_tmp/aa_contract"
aa_reject "$gi_old" "$gi_tmp/aa_depth"
aa_reject "$gi_old" "$gi_tmp/aa_duplicate"
aa_reject "$gi_old" "$gi_tmp/aa_contract"
printf 'GUESS_IDENTITY_SELFTEST_OK cases=%s production_files_changed=0 GPU=0\n' "$gi_cases"
