"""Y2 C2 bf-rate dump: shared loader + field reconstruction.

FORMAL certification: the input is the CLEAN, COMPLETE 12-iteration parity46
run (logs/coevolve_consume_parity46/), iters 0-11, 600,000 dump rows.
Final iteration index = 11 (= array index -1 throughout).

Everything here is derived from code, not assumed:
  grid            src/lumina.h:491-493   NLTE_N_FREQ_BINS=1000, 1.5e14..3.0e16 Hz
  coarse ladder   src/lumina_plasma.c:694  ARTIS_RADFIELD_NC=24, c = f*NC/nb
  J_C1 rebuild    src/lumina_plasma.c:1049-1067 (the array the GEMM later stages)
  C2 consume      src/lumina_plasma.c:14605-14636 (sigma*bfr, else pref*J_bin)
  GEMM K          src/lumina_nlte_gemm.cu:217-229 (K = sigma*4pi/(h*nu)*dnu)
  GEMM J          src/lumina_nlte_gemm.cu:416-419 (stages nlte->J_nu = J_C1)
"""
import os
import struct
import numpy as np
import pandas as pd

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
MODEL = os.path.join(ROOT, "data/tardis_reference_toy06_19p48d_sivcaiv")
_CERT = os.path.join(
    ROOT, "validation/cmfgen_toy06_19p48d/analysis/y2_c2bfr_certification_formal")
# Defaults reproduce FORMAL_REPORT.md verbatim (clean parity46, iters 0-11).
# The PRELIMINARY numbers live in ../y2_c2bfr_certification/ (killed-partial,
# iters 0-10) and are NOT reproduced by this copy of the scripts.
RUN = os.environ.get("Y2_RUN", os.path.join(ROOT, "logs/coevolve_consume_parity46"))
OUT = os.environ.get("Y2_OUT", _CERT)
STDOUT_LOG = os.environ.get("Y2_STDOUT", os.path.join(RUN, "stdout.log"))
IT_FINAL = 11          # clean run: LUMINA_PURE_CMFGEN_ITER=12 -> iters 0..11

# --- physical constants, verbatim from src/lumina.h / lumina_plasma.c ---
H_PLANCK = 6.62607015e-27
C_LIGHT = 2.99792458e10
K_B = 1.380649e-16
EV_TO_ERG = 1.602176634e-12

NB = 1000                 # NLTE_N_FREQ_BINS
NU_MIN = 1.5e14           # NLTE_NU_MIN
NU_MAX = 3.0e16           # NLTE_NU_MAX
NC = 24                   # ARTIS_RADFIELD_NC
DLOG = np.log(NU_MAX / NU_MIN) / NB

# fine-bin geometry (lumina_plasma.c:1102-1105)
_lo = NU_MIN * np.exp(np.arange(NB) * DLOG)
_hi = NU_MIN * np.exp((np.arange(NB) + 1) * DLOG)
NU_MID = NU_MIN * np.exp((np.arange(NB) + 0.5) * DLOG)
DNU = _hi - _lo
LAM_MID_A = C_LIGHT / NU_MID * 1e8
# coarse index of each fine bin: (int)((long)f * NC / nb)
CIDX = (np.arange(NB) * NC) // NB


def planck_bnu(T, nu):
    """B_nu in erg/cm2/s/Hz/sr; mirrors planck_bnu() overflow guard."""
    T = np.asarray(T, dtype=float)
    nu = np.asarray(nu, dtype=float)
    x = H_PLANCK * nu / (K_B * np.where(T > 0, T, 1.0))
    out = np.zeros(np.broadcast(T, nu).shape, dtype=float)
    ok = (T > 0) & (x < 700.0) & (x > 0)
    xs = np.where(ok, x, 1.0)
    val = (2.0 * H_PLANCK * nu**3 / C_LIGHT**2) / np.expm1(xs)
    return np.where(ok, val, out)


def load_dumps():
    """-> c2 arrays [n_iter, n_shell, NB] and the c1 table."""
    c2 = pd.read_csv(os.path.join(RUN, "lumina_c2_bfr_dump.csv"))
    c1 = pd.read_csv(os.path.join(RUN, "lumina_c1_bins.csv"))
    iters = np.sort(c2["iter"].unique())
    shells = np.sort(c2["shell"].unique())
    ni, ns = len(iters), len(shells)
    # rows are written in (iter, shell, bin) order by the writer loop -> reshape
    assert len(c2) == ni * ns * NB, (len(c2), ni, ns, NB)
    for col in ("iter", "shell", "bin"):
        pass
    order = np.lexsort((c2["bin"].values, c2["shell"].values, c2["iter"].values))
    c2 = c2.iloc[order].reset_index(drop=True)
    shp = (ni, ns, NB)
    J_raw = c2["J_raw"].values.reshape(shp)
    bfr = c2["bfr"].values.reshape(shp)
    cnt = c2["j_nu_count"].values.reshape(shp)
    nu_mid_file = c2["nu_mid"].values.reshape(shp)[0, 0]
    return dict(iters=iters, shells=shells, J_raw=J_raw, bfr=bfr, cnt=cnt,
                nu_mid_file=nu_mid_file, c1=c1)


def build_JC1(c1, iters, shells):
    """Reconstruct nlte->J_nu exactly as lumina_plasma.c:1049-1067 leaves it.

    Returns (J_C1[ni,ns,NB], mode[ni,ns,NC], W[ni,ns,NC], TR[ni,ns,NC]).
    Callers must pass J_raw for the 'degen' branch (handled in fill_degen).
    """
    ni, ns = len(iters), len(shells)
    W = np.zeros((ni, ns, NC))
    TR = np.zeros((ni, ns, NC))
    mode = np.empty((ni, ns, NC), dtype=object)
    it_of = {v: i for i, v in enumerate(iters)}
    sh_of = {v: i for i, v in enumerate(shells)}
    for r in c1.itertuples(index=False):
        i, s, c = it_of[r.iter], sh_of[r.shell], r.bin
        W[i, s, c] = r.W
        TR[i, s, c] = r.T_R
        mode[i, s, c] = r.mode
    return W, TR, mode


def assemble_JC1(W, TR, mode, J_raw):
    """Fine-bin J the GEMM stages. Branch order = lumina_plasma.c:1053-1067."""
    Wf = W[:, :, CIDX]
    TRf = TR[:, :, CIDX]
    modef = mode[:, :, CIDX]
    B = planck_bnu(TRf, NU_MID[None, None, :])
    J = np.where((Wf > 0) & (TRf > 0), Wf * B, 0.0)
    # LUMINA_C1_DEGEN_FALLBACK=1 in this run -> degen coarse bins publish raw
    J = np.where(modef == "degen", J_raw, J)
    return J


# ------------------------------------------------------------------ atomic
def load_levels():
    lv = pd.read_csv(os.path.join(MODEL, "levels.csv"))
    lv["gidx"] = np.arange(len(lv))
    return lv


def load_ioniz():
    io = pd.read_csv(os.path.join(MODEL, "ionization_energies.csv"))
    return {(int(r.atomic_number), int(r.ion_number)): float(r.ionization_energy_eV)
            for r in io.itertuples(index=False)}


class SigmaBF:
    """Random-access reader for cmfgen_sigma_bf.bin (lumina_atomic.c:951-1035)."""

    def __init__(self, path):
        self.f = open(path, "rb")
        magic, version, n_lev, n_freq = struct.unpack("<IIii", self.f.read(16))
        assert magic == 0x434D4644 and version == 1, (hex(magic), version)
        self.nu_min, self.nu_max = struct.unpack("<dd", self.f.read(16))
        self.n_lev, self.n_freq = n_lev, n_freq
        self.has = np.frombuffer(self.f.read(n_lev), dtype=np.int8).copy()
        off = 32 + n_lev
        pad = (-off) % 8
        self.data_off = off + pad
        self.f.seek(self.data_off)

    def row(self, gidx):
        self.f.seek(self.data_off + gidx * self.n_freq * 8)
        return np.frombuffer(self.f.read(self.n_freq * 8), dtype="<f8").copy()


def sigma_path():
    return os.path.join(MODEL, "cmfgen_sigma_bf.bin")


# --------------------------------------------------------------- rate kernels
def R_bf_paths(sigma, nu_thresh, J_C1, bfr, apply_thresh_to_gemm=False):
    """Both consumer integrals for ONE level, vectorised over (..., NB).

    C2   (lumina_plasma.c:14605-14636): sum over bins with nu_mid>=nu_thresh and
         sigma>0 of  sigma*bfr  where bfr>0, else pref*J_C1.
    GEMM (lumina_nlte_gemm.cu:217-229 + 427-443): sum over bins with sigma>0 of
         pref*J_C1.  NOTE the CMFGEN-sigma branch of the K build has NO
         nu_bin<nu_thresh cut (only the Kramers branch does), so the faithful
         GEMM integral omits it; apply_thresh_to_gemm=True gives the variant.
    """
    pref = 4.0 * np.pi * sigma / (H_PLANCK * NU_MID) * DNU
    ok_sig = sigma > 0.0
    above = NU_MID >= nu_thresh
    m_c2 = ok_sig & above
    m_gm = ok_sig & (above if apply_thresh_to_gemm else True)
    use_bfr = bfr > 0.0
    c2 = np.sum(np.where(m_c2, np.where(use_bfr, sigma * bfr, pref * J_C1), 0.0),
                axis=-1)
    gm = np.sum(np.where(m_gm, pref * J_C1, 0.0), axis=-1)
    # split diagnostics
    c2_mc = np.sum(np.where(m_c2 & use_bfr, sigma * bfr, 0.0), axis=-1)
    c2_fb = np.sum(np.where(m_c2 & ~use_bfr, pref * J_C1, 0.0), axis=-1)
    return c2, gm, c2_mc, c2_fb
