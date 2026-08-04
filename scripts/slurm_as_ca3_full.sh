#!/bin/bash
#SBATCH --job-name=as_ca3_full
#SBATCH --partition=a10,a40,a100,a100_pcie
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# AUTOSTRUCTURE Ca III FULL-PHYSICS DR (Ar-like, Z=20).
# Layer-2 over-ionization fix per #299/#298 R_bf/R_rec diagnosis.
# Ca II→III over-ionization 99.996% (158870 ion dump); Mazzotta1998
# single resonance @ 35.5 eV gives alpha_DR(8000K) ~ 1.5e-30 (effectively 0).
# Need low-n Rydberg resonances captured by NMAX=15 LMAX=7 DRR grid.
#
# MXCONF=8: ground 3p^6 + 5 single excitations (3p->3d,4s,4p,4d,4f)
#         + 2 double excitations (3p^4 3d^2, 3p^4 3d 4s).
# MXCCF=15: full single-from-ground correlation + select doubles.
# Output: oic -> o1 (symlink) -> adasdr -> adf09 -> Burgess 5-term fit.

cd /home/kjhan/local/autostructure/runs/ca3_dr_full

echo "=== AS Ca III FULL-PHYSICS DR (Ar-like Z=20) ==="
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "deck:"; cat das | head -40; echo "..."

/home/kjhan/local/autostructure/aslm.x < das > as_stdout.log 2> as_stderr.log
rc_as=$?
echo "AS exit=$rc_as  $(date)"
echo "Files: $(ls -la oic olg ols 2>&1 | head -5)"

if [ -f oic ] && [ $rc_as -eq 0 ]; then
    ln -sf oic o1
    echo "--- AS done, oic ready for adasin construction ---"
    echo "oic size: $(ls -l oic | awk '{print $5}') bytes"
    echo "ols size: $(ls -l ols | awk '{print $5}') bytes"
    echo "--- oic header (target levels) ---"
    head -40 oic
fi

echo "Done: $(date)"
