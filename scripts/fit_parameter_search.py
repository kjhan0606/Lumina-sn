#!/usr/bin/env python3
"""
Multi-dimensional parameter fitting for SN 2011fe using LUMINA-SN.

Searches an 11D physical parameter space with broken power-law density:
  (log_L, v_inner, log_rho_0, density_exp, T_e_ratio,
   v_core, v_wall, X_Fe_core, X_Si_wall, v_break, density_exp_outer)

Broken power-law density profile:
  v < v_break:  rho = rho_0 * (v / v_inner)^density_exp
  v >= v_break: rho = rho_break * (v / v_break)^density_exp_outer
  (continuous at v_break)

3-zone composition model:
  Core  (v_inner → v_core): Fe-rich (X_Fe_core), Si=0.05, S=0.05, Ca=0.03, O=filler
  Wall  (v_core  → v_wall): Si-rich (X_Si_wall), Fe=0.05, S=0.05, Ca=0.03, O=filler
  Outer (v_wall  → v_outer): O-rich, Si=0.02, Fe=0.01, S=0.02, Ca=0.01

Uses coarse-to-fine strategy:
  Phase 1: 200 Latin Hypercube samples, 20K packets x 5 iters  (~45 min)
  Phase 2: Top-20 refinement, 100K packets x 10 iters          (~30 min)
  Phase 3: Top-3 production, 500K packets x 20 iters            (~24 min)

Usage:
  python3 scripts/fit_parameter_search.py           # Full 3-phase search
  python3 scripts/fit_parameter_search.py --test     # Single validation run
  python3 scripts/fit_parameter_search.py --phase1   # Phase 1 only
"""
import os
import sys
import json
import time
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== Constants =====
SIGMA_SB = 5.6704e-5       # Stefan-Boltzmann (erg/cm^2/s/K^4)
T_EXP = 1641600.0          # Time since explosion (19 days in seconds)
V_OUTER = 25000.0           # Outer velocity (km/s)
N_SHELLS = 30
DENSITY_EXPONENT = -7
C_LIGHT = 2.99792458e10    # cm/s

# Truly fixed abundance fractions (same in all zones)
FIXED_SPECIES = {
    27: 0.05,   # Co
    28: 0.10,   # Ni
    6:  0.02,   # C
}
FIXED_SPECIES_SUM = sum(FIXED_SPECIES.values())  # 0.17

# Fixed S and Ca per zone (not free parameters)
ZONE_S  = {'core': 0.05, 'wall': 0.05, 'outer': 0.02}
ZONE_CA = {'core': 0.03, 'wall': 0.03, 'outer': 0.01}

# Element ordering in abundances.csv (atomic numbers)
ELEMENT_ORDER = [6, 8, 14, 16, 20, 26, 27, 28]

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINARY = PROJECT_ROOT / "lumina"
REF_DIR = PROJECT_ROOT / "data" / "tardis_reference"
OBS_FILE = PROJECT_ROOT / "data" / "sn2011fe" / "sn2011fe_observed_Bmax.csv"
TARDIS_SPEC = PROJECT_ROOT / "data" / "sn2011fe" / "tardis_spectrum.csv"
TMPDIR_BASE = Path("/tmp/lumina_fit")

# Files to symlink (everything except what we regenerate)
REGEN_FILES = {'config.json', 'geometry.csv', 'density.csv', 'abundances.csv',
               'electron_densities.csv', 'plasma_state.csv'}


# ===== Data classes =====
@dataclass
class ModelParams:
    log_L: float       # log10(luminosity in erg/s), range [42.8, 43.15]
    v_inner: float     # inner velocity (km/s), range [8000, 13000]
    log_rho_0: float   # log10(rho_0 in g/cm^3), range [-13.5, -12.7]
    density_exp: float  # density power-law exponent (inner), range [-10, -4]
    T_e_ratio: float    # T_e/T_rad ratio, range [0.7, 1.0]
    v_core: float       # core/wall boundary velocity (km/s), range [10000, 16000]
    v_wall: float       # wall/outer boundary velocity (km/s), range [14000, 22000]
    X_Fe_core: float    # Fe mass fraction in core zone, range [0.2, 0.8]
    X_Si_wall: float    # Si mass fraction in wall zone, range [0.2, 0.7]
    v_break: float      # density slope break velocity (km/s), range [12000, 20000]
    density_exp_outer: float  # outer density slope (v > v_break), range [-14, -6]

    @property
    def L_erg_s(self):
        return 10**self.log_L

    @property
    def rho_0(self):
        return 10**self.log_rho_0

    @property
    def v_inner_cm_s(self):
        return self.v_inner * 1e5

    @property
    def T_inner_estimate(self):
        """Stefan-Boltzmann estimate for initial T_inner."""
        R_inner = self.v_inner_cm_s * T_EXP
        return (self.L_erg_s / (4 * np.pi * SIGMA_SB * R_inner**2))**0.25

    def density_at_v(self, v_mid):
        """Compute broken power-law density at velocity v_mid (km/s)."""
        rho_0 = self.rho_0
        v_min = self.v_inner
        if v_mid < self.v_break:
            return rho_0 * (v_mid / v_min) ** self.density_exp
        else:
            # Continuity at v_break
            rho_break = rho_0 * (self.v_break / v_min) ** self.density_exp
            return rho_break * (v_mid / self.v_break) ** self.density_exp_outer

    def zone_abundances(self, zone):
        """Return (X_Si, X_Fe, X_S, X_Ca, X_O) for a given zone."""
        if zone == 'core':
            X_Si, X_Fe = 0.05, self.X_Fe_core
        elif zone == 'wall':
            X_Si, X_Fe = self.X_Si_wall, 0.05
        else:  # outer
            X_Si, X_Fe = 0.02, 0.01
        X_S = ZONE_S[zone]
        X_Ca = ZONE_CA[zone]
        X_O = 1.0 - FIXED_SPECIES_SUM - X_Si - X_Fe - X_S - X_Ca
        return X_Si, X_Fe, X_S, X_Ca, X_O

    def is_valid(self):
        # Zones must not overlap, with >=1000 km/s gap
        if self.v_core + 1000 >= self.v_wall:
            return False
        # Core boundary must be above inner boundary
        if self.v_core <= self.v_inner:
            return False
        # Wall boundary must be below outer boundary
        if self.v_wall >= V_OUTER:
            return False
        # Density break must be within ejecta
        if self.v_break <= self.v_inner + 1000:
            return False
        if self.v_break >= V_OUTER - 1000:
            return False
        # Oxygen filler must be positive in all zones
        for zone in ('core', 'wall', 'outer'):
            *_, X_O = self.zone_abundances(zone)
            if X_O < 0.03:
                return False
        return True


@dataclass
class FitResult:
    params: ModelParams
    rms: float = 999.0
    si_depth: float = 0.0
    si_velocity: float = 0.0
    si_wave_min: float = 0.0
    runtime: float = 0.0
    converged: bool = False
    spectrum_wave: np.ndarray = field(default_factory=lambda: np.array([]))
    spectrum_flux: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary_dict(self):
        p = self.params
        *_, X_O_core = p.zone_abundances('core')
        *_, X_O_wall = p.zone_abundances('wall')
        *_, X_O_outer = p.zone_abundances('outer')
        return {
            'log_L': p.log_L,
            'v_inner': p.v_inner,
            'log_rho_0': p.log_rho_0,
            'density_exp': p.density_exp,
            'T_e_ratio': p.T_e_ratio,
            'v_core': p.v_core,
            'v_wall': p.v_wall,
            'X_Fe_core': p.X_Fe_core,
            'X_Si_wall': p.X_Si_wall,
            'v_break': p.v_break,
            'density_exp_outer': p.density_exp_outer,
            'X_O_core': X_O_core,
            'X_O_wall': X_O_wall,
            'X_O_outer': X_O_outer,
            'rms': self.rms,
            'si_depth': self.si_depth,
            'si_velocity': self.si_velocity,
            'si_wave_min': self.si_wave_min,
            'runtime': self.runtime,
            'converged': int(self.converged),
        }


# ===== Latin Hypercube Sampling =====
def latin_hypercube(n_samples, param_ranges, rng=None):
    """Generate Latin Hypercube samples in 11D parameter space.

    Args:
        n_samples: Number of samples
        param_ranges: List of (min, max) tuples for each parameter
        rng: numpy random generator

    Returns:
        List of ModelParams
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ndim = len(param_ranges)
    max_attempts = n_samples * 20

    # Generate LHS: stratified random sampling
    result = np.zeros((n_samples, ndim))
    for d in range(ndim):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            lo = perm[i] / n_samples
            hi = (perm[i] + 1) / n_samples
            result[i, d] = lo + rng.random() * (hi - lo)

    # Scale to parameter ranges
    for d in range(ndim):
        pmin, pmax = param_ranges[d]
        result[:, d] = pmin + result[:, d] * (pmax - pmin)

    # Convert to ModelParams, rejecting invalid (zone overlap, O<0)
    valid = []
    for i in range(n_samples):
        p = ModelParams(
            log_L=result[i, 0],
            v_inner=result[i, 1],
            log_rho_0=result[i, 2],
            density_exp=result[i, 3],
            T_e_ratio=result[i, 4],
            v_core=result[i, 5],
            v_wall=result[i, 6],
            X_Fe_core=result[i, 7],
            X_Si_wall=result[i, 8],
            v_break=result[i, 9],
            density_exp_outer=result[i, 10],
        )
        if p.is_valid():
            valid.append(p)

    # Fill rejected slots with random valid samples
    attempts = 0
    while len(valid) < n_samples and attempts < max_attempts:
        vals = [rng.uniform(lo, hi) for lo, hi in param_ranges]
        p = ModelParams(
            log_L=vals[0], v_inner=vals[1], log_rho_0=vals[2],
            density_exp=vals[3], T_e_ratio=vals[4],
            v_core=vals[5], v_wall=vals[6],
            X_Fe_core=vals[7], X_Si_wall=vals[8],
            v_break=vals[9], density_exp_outer=vals[10],
        )
        if p.is_valid():
            valid.append(p)
        attempts += 1

    return valid[:n_samples]


# ===== Fitter class =====
class LuminaFitter:
    def __init__(self, obs_file=OBS_FILE, ref_dir=REF_DIR, binary=BINARY, feature_weight=5.0):
        self.ref_dir = Path(ref_dir)
        self.binary = Path(binary)
        self.obs_file = Path(obs_file)
        self.feature_weight = feature_weight

        # Load observed spectrum
        if self.obs_file.exists():
            obs = np.genfromtxt(str(self.obs_file), delimiter=',', names=True)
            self.obs_wave = obs['wavelength_angstrom']
            self.obs_flux = obs['flux_erg_s_cm2_angstrom']
        else:
            print(f"WARNING: Observed spectrum not found: {self.obs_file}")
            self.obs_wave = None
            self.obs_flux = None

        # Load TARDIS reference spectrum
        if TARDIS_SPEC.exists():
            ts = np.genfromtxt(str(TARDIS_SPEC), delimiter=',', names=True)
            self.tardis_wave = ts['wavelength_A']
            self.tardis_flux = ts['flux_erg_s_A']
        else:
            self.tardis_wave = None
            self.tardis_flux = None

        # Common grid for comparison
        self.grid = np.arange(3500, 9001, 5.0)

        # Precompute normalized observed spectrum on grid
        if self.obs_wave is not None:
            obs_i = np.interp(self.grid, self.obs_wave, self.obs_flux)
            opt = (self.grid >= 4000) & (self.grid <= 7000)
            peak = obs_i[opt].max()
            self.obs_norm = obs_i / peak if peak > 0 else obs_i
        else:
            self.obs_norm = None

        # Build feature weight array
        self.weights = self._build_weight_array()

        # List reference files for symlinking
        self.ref_files = [f.name for f in self.ref_dir.iterdir() if f.is_file()]

        # Ensure tmpdir base exists
        TMPDIR_BASE.mkdir(parents=True, exist_ok=True)

    def _build_weight_array(self):
        """Build per-grid-point weight array from FEATURE_WINDOWS and feature_weight."""
        W = self.feature_weight
        weights = np.ones(len(self.grid))
        for name, (lo, hi, tier) in FEATURE_WINDOWS.items():
            mask = (self.grid >= lo) & (self.grid <= hi)
            if tier == 1:
                tier_weight = W
            else:
                tier_weight = (W + 1.0) / 2.0
            # Use max so overlapping windows get the higher weight
            weights[mask] = np.maximum(weights[mask], tier_weight)
        return weights

    def create_model_dir(self, params, tag=""):
        """Create temporary reference directory with modified parameters."""
        dirname = f"model_{tag}_{id(params):x}"
        temp_dir = TMPDIR_BASE / dirname
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Symlink unchanged files
        for fname in self.ref_files:
            if fname not in REGEN_FILES:
                src = self.ref_dir / fname
                dst = temp_dir / fname
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)

        # Generate modified files
        self._write_config_json(temp_dir, params)
        self._write_geometry_csv(temp_dir, params)
        self._write_density_csv(temp_dir, params)
        self._write_abundances_csv(temp_dir, params)
        self._write_electron_densities_csv(temp_dir, params)
        self._copy_plasma_state(temp_dir, params)

        return temp_dir

    def _write_config_json(self, dirpath, params):
        cfg = {
            "time_explosion_s": T_EXP,
            "T_inner_K": params.T_inner_estimate,
            "luminosity_inner_erg_s": params.L_erg_s,
            "n_shells": N_SHELLS,
            "n_lines": 137252,
            "n_packets": 200000,
            "n_iterations": 20,
            "seed": 23111963,
            "v_inner_min_cm_s": params.v_inner_cm_s,
            "v_outer_max_cm_s": V_OUTER * 1e5,
            "T_e_T_rad_ratio": params.T_e_ratio,
        }
        with open(dirpath / "config.json", 'w') as f:
            json.dump(cfg, f, indent=2)

    def _write_geometry_csv(self, dirpath, params):
        v_min = params.v_inner  # km/s
        dv = (V_OUTER - v_min) / N_SHELLS
        with open(dirpath / "geometry.csv", 'w') as f:
            f.write("shell_id,r_inner,r_outer,v_inner,v_outer\n")
            for i in range(N_SHELLS):
                vi = (v_min + i * dv) * 1e5       # cm/s
                vo = (v_min + (i + 1) * dv) * 1e5  # cm/s
                ri = vi * T_EXP                     # cm
                ro = vo * T_EXP                     # cm
                f.write(f"{i},{ri},{ro},{vi},{vo}\n")

    def _write_density_csv(self, dirpath, params):
        v_min = params.v_inner  # km/s
        dv = (V_OUTER - v_min) / N_SHELLS
        with open(dirpath / "density.csv", 'w') as f:
            f.write("shell_id,rho\n")
            for i in range(N_SHELLS):
                vi = v_min + i * dv
                vo = v_min + (i + 1) * dv
                v_mid = (vi + vo) / 2.0
                rho = params.density_at_v(v_mid)
                f.write(f"{i},{rho}\n")

    def _write_abundances_csv(self, dirpath, params):
        """Write per-shell abundances using 3-zone stratification model.

        Zone assignment based on shell midpoint velocity:
          Core:  v_inner → v_core
          Wall:  v_core  → v_wall
          Outer: v_wall  → v_outer
        """
        v_min = params.v_inner  # km/s
        dv = (V_OUTER - v_min) / N_SHELLS

        # Build per-shell abundance arrays
        # abundances[z][shell] = mass fraction
        abundances = {z: np.zeros(N_SHELLS) for z in ELEMENT_ORDER}

        for i in range(N_SHELLS):
            vi = v_min + i * dv
            vo = v_min + (i + 1) * dv
            v_mid = (vi + vo) / 2.0

            # Determine zone
            if v_mid < params.v_core:
                zone = 'core'
            elif v_mid < params.v_wall:
                zone = 'wall'
            else:
                zone = 'outer'

            X_Si, X_Fe, X_S, X_Ca, X_O = params.zone_abundances(zone)

            for z in ELEMENT_ORDER:
                if z == 8:     # Oxygen (filler)
                    abundances[z][i] = X_O
                elif z == 14:  # Silicon
                    abundances[z][i] = X_Si
                elif z == 26:  # Iron
                    abundances[z][i] = X_Fe
                elif z == 16:  # Sulfur
                    abundances[z][i] = X_S
                elif z == 20:  # Calcium
                    abundances[z][i] = X_Ca
                else:
                    abundances[z][i] = FIXED_SPECIES[z]

        with open(dirpath / "abundances.csv", 'w') as f:
            header = "atomic_number," + ",".join(str(i) for i in range(N_SHELLS))
            f.write(header + "\n")
            for z in ELEMENT_ORDER:
                vals = ",".join(f"{abundances[z][i]}" for i in range(N_SHELLS))
                f.write(f"{z},{vals}\n")

    def _write_electron_densities_csv(self, dirpath, params):
        """Write initial electron densities (scaled from reference)."""
        # Scale n_e proportional to density ratio
        ref_density = np.genfromtxt(str(self.ref_dir / "density.csv"),
                                    delimiter=',', names=True)
        ref_ne = np.genfromtxt(str(self.ref_dir / "electron_densities.csv"),
                               delimiter=',', names=True)

        # For each shell, scale n_e by (new_rho / ref_rho) at equivalent position
        v_min_new = params.v_inner
        dv_new = (V_OUTER - v_min_new) / N_SHELLS

        with open(dirpath / "electron_densities.csv", 'w') as f:
            f.write("shell_id,n_e\n")
            for i in range(N_SHELLS):
                vi = v_min_new + i * dv_new
                vo = v_min_new + (i + 1) * dv_new
                v_mid = (vi + vo) / 2.0
                rho_new = params.density_at_v(v_mid)
                # Scale from innermost reference shell
                rho_ref_0 = float(ref_density['rho'][0])
                ne_ref_0 = float(ref_ne['n_e'][0])
                ne = ne_ref_0 * (rho_new / rho_ref_0)
                f.write(f"{i},{ne}\n")

    def _copy_plasma_state(self, dirpath, params):
        """Write initial plasma state (geometric W, scaled T_rad)."""
        with open(dirpath / "plasma_state.csv", 'w') as f:
            f.write("shell_id,W,T_rad\n")
            v_min = params.v_inner
            dv = (V_OUTER - v_min) / N_SHELLS
            T_inner = params.T_inner_estimate
            for i in range(N_SHELLS):
                vi = v_min + i * dv
                vo = v_min + (i + 1) * dv
                v_mid = (vi + vo) / 2.0
                # Geometric dilution: W = 0.5 * (1 - sqrt(1 - (R_inner/R)^2))
                r_ratio = v_min / v_mid
                W = 0.5 * (1.0 - np.sqrt(1.0 - r_ratio**2))
                # T_rad scales roughly as T_inner * (v_inner/v_mid)^0.5
                T_rad = T_inner * (v_min / v_mid)**0.45
                f.write(f"{i},{W},{T_rad}\n")

    def run_model(self, params, n_packets, n_iters, tag=""):
        """Run LUMINA with given parameters and return FitResult."""
        t0 = time.time()
        result = FitResult(params=params)

        # Create temporary reference directory
        temp_dir = self.create_model_dir(params, tag=tag)

        # Run from a temporary working directory to isolate output files
        work_dir = TMPDIR_BASE / f"work_{tag}_{id(params):x}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                [str(self.binary), str(temp_dir), str(n_packets), str(n_iters), "rotation"],
                capture_output=True, timeout=600, cwd=str(work_dir),
                text=True,
            )
            result.converged = (proc.returncode == 0)

            # Read rotation spectrum
            spec_file = work_dir / "lumina_spectrum_rotation.csv"
            if not spec_file.exists():
                # Fallback to real spectrum
                spec_file = work_dir / "lumina_spectrum.csv"

            if spec_file.exists():
                spec = np.genfromtxt(str(spec_file), delimiter=',', names=True)
                result.spectrum_wave = spec['wavelength_angstrom']
                result.spectrum_flux = spec['flux']

                # Compute RMS vs observed
                result.rms = self.compute_rms(result.spectrum_wave, result.spectrum_flux)

                # Measure Si II 6355 feature
                si = self._measure_si_feature(result.spectrum_wave, result.spectrum_flux)
                if si is not None:
                    result.si_depth = si['depth_peak']
                    result.si_velocity = si['v_abs']
                    result.si_wave_min = si['wave_min']
            else:
                if proc.returncode != 0:
                    # Print last few lines of stderr for debugging
                    stderr_lines = proc.stderr.strip().split('\n')
                    print(f"  LUMINA failed (rc={proc.returncode}): "
                          f"{stderr_lines[-1] if stderr_lines else 'no stderr'}")

        except subprocess.TimeoutExpired:
            print(f"  LUMINA timed out (600s)")
        except Exception as e:
            print(f"  Error running LUMINA: {e}")
        finally:
            # Cleanup
            self.cleanup(temp_dir)
            self.cleanup(work_dir)

        result.runtime = time.time() - t0
        return result

    def compute_rms(self, model_wave, model_flux):
        """Compute normalized RMS vs observed spectrum."""
        if self.obs_norm is None or len(model_wave) == 0:
            return 999.0

        # Convert model flux from erg/s/cm to erg/s/A (bin width ~10A for 2000 bins over 500-20000)
        model_flux_A = model_flux / 1e8

        # Interpolate model onto common grid
        model_i = np.interp(self.grid, model_wave, model_flux_A)

        # Normalize to peak in 4000-7000 A
        opt = (self.grid >= 4000) & (self.grid <= 7000)
        peak = model_i[opt].max()
        if peak <= 0:
            return 999.0
        model_n = model_i / peak

        # RMS where observed flux > 5% of peak (exclude noise floor)
        mask = (self.obs_norm > 0.05)
        if mask.sum() == 0:
            return 999.0

        w = self.weights[mask]
        return np.sqrt(np.sum(w * (model_n[mask] - self.obs_norm[mask])**2) / np.sum(w))

    def _measure_si_feature(self, wave, flux):
        """Measure Si II 6355 P-Cygni feature."""
        if len(wave) == 0:
            return None

        flux_A = flux / 1e8
        # Interpolate to fine grid
        grid = np.arange(5500, 7001, 2.0)
        fi = np.interp(grid, wave, flux_A)

        # Normalize
        opt = (grid >= 6500) & (grid <= 7000)
        if fi[opt].max() <= 0:
            return None
        fn = fi / fi[opt].max()

        # Continuum (red side)
        cont_mask = (grid >= 7000) & (grid <= 7500)
        if cont_mask.sum() == 0:
            # Use last point
            F_cont = fn[-1]
        else:
            F_cont = np.median(fn[cont_mask]) if cont_mask.sum() > 0 else fn[-1]

        # Blue trough
        blue_mask = (grid >= 5700) & (grid <= 6250)
        if blue_mask.sum() == 0:
            return None
        idx_min = np.argmin(fn[blue_mask])
        F_min = fn[blue_mask][idx_min]
        wave_min = grid[blue_mask][idx_min]

        # Red peak
        red_mask = (grid >= 6250) & (grid <= 6600)
        if red_mask.sum() == 0:
            return None
        idx_peak = np.argmax(fn[red_mask])
        F_peak = fn[red_mask][idx_peak]

        depth_peak = 1.0 - F_min / F_peak if F_peak > 0 else 0
        v_abs = 3e5 * (6355.0 - wave_min) / 6355.0

        return {
            'depth_peak': depth_peak,
            'v_abs': v_abs,
            'wave_min': wave_min,
            'F_min': F_min,
            'F_peak': F_peak,
        }

    def cleanup(self, dirpath):
        """Remove temporary directory."""
        try:
            if dirpath.exists():
                shutil.rmtree(dirpath, ignore_errors=True)
        except Exception:
            pass


# ===== Phase functions =====
PARAM_RANGES = [
    (42.80, 43.15),    # log_L
    (8000, 13000),     # v_inner (km/s)
    (-13.5, -12.7),    # log_rho_0
    (-10, -4),         # density_exp (inner slope)
    (0.7, 1.0),        # T_e_ratio
    (10000, 16000),    # v_core (km/s)
    (14000, 22000),    # v_wall (km/s)
    (0.2, 0.8),        # X_Fe_core
    (0.2, 0.7),        # X_Si_wall
    (12000, 20000),    # v_break (km/s)
    (-14, -6),         # density_exp_outer
]
PARAM_NAMES = ['log_L', 'v_inner', 'log_rho_0', 'density_exp', 'T_e_ratio',
               'v_core', 'v_wall', 'X_Fe_core', 'X_Si_wall', 'v_break', 'density_exp_outer']

# Feature windows for weighted RMS (wavelength in Angstrom)
# Tier 1 = THE SN Ia diagnostic, Tier 2 = major diagnostics
# Tier weights: tier 1 = feature_weight, tier 2 = (feature_weight+1)/2, everything else = 1.0
FEATURE_WINDOWS = {
    'Si II 6355':  (5850, 6300, 1),   # THE SN Ia diagnostic
    'Ca II H&K':   (3600, 3950, 2),   # Outer Ca abundance, velocity
    'S II W':      (5150, 5550, 2),   # Temperature indicator
    'Si II 5972':  (5550, 5850, 2),   # Temperature indicator
    'Ca II IR':    (7800, 8600, 2),   # Outer Ca, velocity
}


def phase1_coarse(fitter, n=100):
    """Phase 1: Coarse scan with Latin Hypercube Sampling."""
    print("\n" + "=" * 70)
    print("PHASE 1: COARSE SCAN (20K packets x 5 iters)")
    print("=" * 70)

    samples = latin_hypercube(n, PARAM_RANGES)
    print(f"Generated {len(samples)} valid LHS samples")

    results = []
    for i, params in enumerate(samples):
        print(f"\n  [{i+1:3d}/{n}] L={params.log_L:.3f} v={params.v_inner:.0f} "
              f"rho={params.log_rho_0:.2f} exp={params.density_exp:.1f} Te={params.T_e_ratio:.2f} "
              f"vc={params.v_core:.0f} vw={params.v_wall:.0f} "
              f"Fe_c={params.X_Fe_core:.2f} Si_w={params.X_Si_wall:.2f} "
              f"vb={params.v_break:.0f} exp_o={params.density_exp_outer:.1f}",
              end="", flush=True)

        result = fitter.run_model(params, n_packets=20000, n_iters=5, tag=f"p1_{i}")
        results.append(result)

        print(f"  -> RMS={result.rms:.4f} Si_d={result.si_depth:.1%} "
              f"v_Si={result.si_velocity:.0f} ({result.runtime:.1f}s)")

    # Sort by RMS
    results.sort(key=lambda r: r.rms)
    return results


def phase2_refine(fitter, phase1_results, n_top=20):
    """Phase 2: Refine top candidates with more packets."""
    print("\n" + "=" * 70)
    print(f"PHASE 2: REFINEMENT (100K packets x 10 iters, top-{n_top})")
    print("=" * 70)

    top = phase1_results[:n_top]
    results = []
    for i, prev in enumerate(top):
        params = prev.params
        print(f"\n  [{i+1:3d}/{n_top}] L={params.log_L:.3f} v={params.v_inner:.0f} "
              f"rho={params.log_rho_0:.2f} exp={params.density_exp:.1f} Te={params.T_e_ratio:.2f} "
              f"vc={params.v_core:.0f} vw={params.v_wall:.0f} "
              f"Fe_c={params.X_Fe_core:.2f} Si_w={params.X_Si_wall:.2f} "
              f"vb={params.v_break:.0f} exp_o={params.density_exp_outer:.1f} "
              f"(P1 RMS={prev.rms:.4f})", end="", flush=True)

        result = fitter.run_model(params, n_packets=100000, n_iters=10, tag=f"p2_{i}")
        results.append(result)

        print(f"  -> RMS={result.rms:.4f} Si_d={result.si_depth:.1%} "
              f"v_Si={result.si_velocity:.0f} ({result.runtime:.1f}s)")

    results.sort(key=lambda r: r.rms)
    return results


def phase3_production(fitter, phase2_results, n_top=3):
    """Phase 3: Production-quality spectra for best candidates."""
    print("\n" + "=" * 70)
    print(f"PHASE 3: PRODUCTION (500K packets x 20 iters, top-{n_top})")
    print("=" * 70)

    top = phase2_results[:n_top]
    results = []
    for i, prev in enumerate(top):
        params = prev.params
        print(f"\n  [{i+1}/{n_top}] L={params.log_L:.3f} v={params.v_inner:.0f} "
              f"rho={params.log_rho_0:.2f} exp={params.density_exp:.1f} Te={params.T_e_ratio:.2f} "
              f"vc={params.v_core:.0f} vw={params.v_wall:.0f} "
              f"Fe_c={params.X_Fe_core:.2f} Si_w={params.X_Si_wall:.2f} "
              f"vb={params.v_break:.0f} exp_o={params.density_exp_outer:.1f} "
              f"(P2 RMS={prev.rms:.4f})", end="", flush=True)

        result = fitter.run_model(params, n_packets=500000, n_iters=20, tag=f"p3_{i}")
        results.append(result)

        print(f"  -> RMS={result.rms:.4f} Si_d={result.si_depth:.1%} "
              f"v_Si={result.si_velocity:.0f} ({result.runtime:.1f}s)")

    results.sort(key=lambda r: r.rms)
    return results


# ===== Plotting =====
def save_results_csv(results, filename):
    """Save results to CSV file."""
    filepath = PROJECT_ROOT / filename
    if not results:
        return
    keys = list(results[0].summary_dict().keys())
    with open(filepath, 'w') as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            d = r.summary_dict()
            f.write(",".join(str(d[k]) for k in keys) + "\n")
    print(f"Saved: {filepath}")


def plot_sensitivity(results, output="fit_sensitivity.png"):
    """11-panel scatter: RMS vs each parameter."""
    filepath = PROJECT_ROOT / output
    fig, axes = plt.subplots(3, 4, figsize=(22, 12))
    fig.suptitle('11D Broken Density Parameter Sensitivity (RMS vs. each parameter)',
                 fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()

    for idx, pname in enumerate(PARAM_NAMES):
        ax = axes_flat[idx]
        vals = [getattr(r.params, pname) for r in results]
        rms_vals = [r.rms for r in results]

        # Color by RMS (lower = better = greener)
        sc = ax.scatter(vals, rms_vals, c=rms_vals, cmap='RdYlGn_r', s=20, alpha=0.7,
                        edgecolors='k', linewidths=0.3)
        ax.set_xlabel(pname, fontsize=10)
        ax.set_ylabel('RMS' if idx % 4 == 0 else '', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark best point
        best_idx = np.argmin(rms_vals)
        ax.scatter([vals[best_idx]], [rms_vals[best_idx]], marker='*', s=200,
                   c='blue', edgecolors='k', zorder=10)

    # Hide the 12th subplot (3x4 grid, 11 params)
    axes_flat[11].set_visible(False)

    plt.colorbar(sc, ax=axes_flat[10], label='RMS', shrink=0.8)
    plt.tight_layout()
    plt.savefig(str(filepath), dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_best_fit(best_result, fitter, output="fit_best_spectrum.png", feature_weight=None):
    """Plot best-fit spectrum vs observed + TARDIS reference."""
    filepath = PROJECT_ROOT / output
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={'height_ratios': [3, 3, 1.5]})
    fig.suptitle('Best-Fit LUMINA Model vs SN 2011fe Observation', fontsize=14, fontweight='bold')

    grid = fitter.grid

    # Get normalized spectra
    obs_n = fitter.obs_norm

    # Best model
    if len(best_result.spectrum_wave) > 0:
        model_flux_A = best_result.spectrum_flux / 1e8
        model_i = np.interp(grid, best_result.spectrum_wave, model_flux_A)
        opt = (grid >= 4000) & (grid <= 7000)
        peak = model_i[opt].max()
        model_n = model_i / peak if peak > 0 else model_i
    else:
        model_n = np.zeros_like(grid)

    # TARDIS reference
    tardis_n = None
    if fitter.tardis_wave is not None:
        tardis_i = np.interp(grid, fitter.tardis_wave, fitter.tardis_flux)
        opt = (grid >= 4000) & (grid <= 7000)
        peak_t = tardis_i[opt].max()
        tardis_n = tardis_i / peak_t if peak_t > 0 else tardis_i

    p = best_result.params

    # Panel 1: Full spectrum
    ax = axes[0]
    if obs_n is not None:
        ax.plot(grid, obs_n, 'k-', alpha=0.8, linewidth=1.2, label='Observed (Pereira+2013)')
    if tardis_n is not None:
        ax.plot(grid, tardis_n, 'b-', alpha=0.5, linewidth=0.8, label='TARDIS SN 2011fe')
    ax.plot(grid, model_n, 'r-', alpha=0.7, linewidth=1.0,
            label=f'LUMINA best-fit (RMS={best_result.rms:.4f})')
    ax.set_xlim(3500, 9000)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Full Optical Spectrum')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Stats box
    _, _, _, _, X_O_core = p.zone_abundances('core')
    _, _, _, _, X_O_wall = p.zone_abundances('wall')
    _, _, _, _, X_O_outer = p.zone_abundances('outer')
    fw_label = f" (w={feature_weight:.1f})" if feature_weight is not None and feature_weight != 1.0 else ""
    stats = (f"log L = {p.log_L:.3f}\n"
             f"v_inner = {p.v_inner:.0f} km/s\n"
             f"log rho_0 = {p.log_rho_0:.3f}\n"
             f"density_exp = {p.density_exp:.1f}\n"
             f"v_break = {p.v_break:.0f} km/s\n"
             f"exp_outer = {p.density_exp_outer:.1f}\n"
             f"T_e/T_rad = {p.T_e_ratio:.2f}\n"
             f"--- Zones ---\n"
             f"Core (<{p.v_core:.0f}): Fe={p.X_Fe_core:.2f} O={X_O_core:.2f}\n"
             f"Wall (<{p.v_wall:.0f}): Si={p.X_Si_wall:.2f} O={X_O_wall:.2f}\n"
             f"Outer: Si=0.02 Fe=0.01 O={X_O_outer:.2f}\n"
             f"RMS = {best_result.rms:.4f}{fw_label}\n"
             f"Si II depth = {best_result.si_depth:.1%}")
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Si II detail
    ax = axes[1]
    mask = (grid >= 5000) & (grid <= 7500)
    if obs_n is not None:
        ax.plot(grid[mask], obs_n[mask], 'k-', alpha=0.8, linewidth=1.2, label='Observed')
    if tardis_n is not None:
        ax.plot(grid[mask], tardis_n[mask], 'b-', alpha=0.5, linewidth=0.8, label='TARDIS')
    ax.plot(grid[mask], model_n[mask], 'r-', alpha=0.7, linewidth=1.0, label='LUMINA best-fit')
    ax.axvline(6355, color='green', linestyle='--', alpha=0.3, label='Si II 6355 rest')
    for v in [10000, 12000, 15000]:
        w = 6355 * (1 - v / 3e5)
        ax.axvline(w, color='orange', linestyle=':', alpha=0.3)
        ax.text(w, 0.12, f'{v // 1000}k', fontsize=7, color='orange', ha='center')
    ax.set_xlim(5000, 7500)
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Si II 6355 Region Detail')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Residuals
    ax = axes[2]
    if obs_n is not None:
        res_model = model_n - obs_n
        ax.plot(grid, res_model, 'r-', alpha=0.7, linewidth=0.8,
                label=f'LUMINA - Obs (RMS={best_result.rms:.4f})')
        if tardis_n is not None:
            mask_fit = obs_n > 0.05
            rms_tardis = np.sqrt(np.mean((tardis_n[mask_fit] - obs_n[mask_fit])**2))
            res_tardis = tardis_n - obs_n
            ax.plot(grid, res_tardis, 'b-', alpha=0.5, linewidth=0.8,
                    label=f'TARDIS - Obs (RMS={rms_tardis:.4f})')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.fill_between(grid, -0.05, 0.05, color='green', alpha=0.1, label='5% band')
    ax.set_xlim(3500, 9000)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Observed')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(filepath), dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_si_detail(results, fitter, output="fit_best_siII.png"):
    """Si II region detail for top-3 models."""
    filepath = PROJECT_ROOT / output
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Si II 6355: Top Models vs Observation', fontsize=14, fontweight='bold')

    grid = fitter.grid
    mask = (grid >= 5500) & (grid <= 7000)

    if fitter.obs_norm is not None:
        ax.plot(grid[mask], fitter.obs_norm[mask], 'k-', linewidth=1.5, label='Observed', alpha=0.9)

    if fitter.tardis_wave is not None:
        tardis_i = np.interp(grid, fitter.tardis_wave, fitter.tardis_flux)
        opt = (grid >= 4000) & (grid <= 7000)
        tardis_n = tardis_i / tardis_i[opt].max()
        ax.plot(grid[mask], tardis_n[mask], 'b-', linewidth=1.2, label='TARDIS', alpha=0.6)

    colors = ['red', 'darkorange', 'green']
    for i, r in enumerate(results[:3]):
        if len(r.spectrum_wave) == 0:
            continue
        model_flux_A = r.spectrum_flux / 1e8
        model_i = np.interp(grid, r.spectrum_wave, model_flux_A)
        opt = (grid >= 4000) & (grid <= 7000)
        peak = model_i[opt].max()
        if peak <= 0:
            continue
        model_n = model_i / peak
        p = r.params
        ax.plot(grid[mask], model_n[mask], color=colors[i], linewidth=1.2, alpha=0.7,
                label=f'#{i+1} RMS={r.rms:.4f} (L={p.log_L:.2f} v={p.v_inner:.0f})')

    ax.axvline(6355, color='green', linestyle='--', alpha=0.3)
    for v in [10000, 12000, 15000]:
        w = 6355 * (1 - v / 3e5)
        ax.axvline(w, color='orange', linestyle=':', alpha=0.3)
        ax.text(w, 0.12, f'{v // 1000}k', fontsize=7, color='orange', ha='center')

    ax.set_xlim(5500, 7000)
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Normalized Flux')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(filepath), dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


# ===== Main =====
def main():
    parser = argparse.ArgumentParser(description="LUMINA-SN parameter fitting for SN 2011fe")
    parser.add_argument('--test', action='store_true',
                        help='Single validation run with default params (10K pkts)')
    parser.add_argument('--phase1', action='store_true',
                        help='Run Phase 1 only')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Number of LHS samples for Phase 1 (default: 200)')
    parser.add_argument('--binary', type=str, default=None,
                        help='Path to LUMINA binary (default: lumina; use lumina_cuda for GPU)')
    parser.add_argument('--refine', type=str, default=None,
                        help='Refine around a model from fit_results_final.csv (1-indexed row, e.g. --refine 3)')
    parser.add_argument('--refine-samples', type=int, default=50,
                        help='Number of LHS samples for refinement (default: 50)')
    parser.add_argument('--feature-weight', type=float, default=5.0,
                        help='Feature weight W: Si II 6355 = W×, major features = (W+1)/2×, '
                             'continuum = 1×. Use 1.0 for uniform RMS. (default: 5.0)')
    args = parser.parse_args()

    # Determine binary
    binary = Path(args.binary) if args.binary else BINARY
    if not binary.exists():
        print(f"ERROR: LUMINA binary not found at {binary}")
        print("Build with: make (CPU) or make cuda (GPU)")
        sys.exit(1)

    # Check observed data
    if not OBS_FILE.exists():
        print(f"ERROR: Observed spectrum not found: {OBS_FILE}")
        sys.exit(1)

    fitter = LuminaFitter(binary=binary, feature_weight=args.feature_weight)

    # Print feature weighting info
    fw = args.feature_weight
    tier2_w = (fw + 1.0) / 2.0
    print(f"Feature weighting: Si II 6355 = {fw:.1f}x, major features = {tier2_w:.1f}x, continuum = 1.0x")

    if args.test:
        # Single validation run with default parameters
        print("=" * 70)
        print("TEST MODE: Single run with default parameters")
        print("=" * 70)

        params = ModelParams(
            log_L=42.975,
            v_inner=10000,
            log_rho_0=-13.097,
            density_exp=-7.0,
            T_e_ratio=0.9,
            v_core=12000,
            v_wall=18000,
            X_Fe_core=0.5,
            X_Si_wall=0.4,
            v_break=15000,
            density_exp_outer=-10.0,
        )
        print(f"\nParameters:")
        print(f"  log_L = {params.log_L} -> L = {params.L_erg_s:.4e} erg/s")
        print(f"  v_inner = {params.v_inner} km/s")
        print(f"  log_rho_0 = {params.log_rho_0} -> rho_0 = {params.rho_0:.4e} g/cm^3")
        print(f"  density_exp = {params.density_exp} (inner, v < v_break)")
        print(f"  v_break = {params.v_break} km/s")
        print(f"  density_exp_outer = {params.density_exp_outer} (outer, v > v_break)")
        print(f"  T_e/T_rad ratio = {params.T_e_ratio}")
        print(f"  v_core = {params.v_core} km/s, v_wall = {params.v_wall} km/s")
        print(f"  Core: Fe={params.X_Fe_core:.2f}, Si=0.05, S=0.05, Ca=0.03, O={params.zone_abundances('core')[4]:.2f}")
        print(f"  Wall: Si={params.X_Si_wall:.2f}, Fe=0.05, S=0.05, Ca=0.03, O={params.zone_abundances('wall')[4]:.2f}")
        print(f"  Outer: Si=0.02, Fe=0.01, S=0.02, Ca=0.01, O={params.zone_abundances('outer')[4]:.2f}")
        print(f"  T_inner (S-B estimate) = {params.T_inner_estimate:.1f} K")

        result = fitter.run_model(params, n_packets=10000, n_iters=5, tag="test")

        print(f"\nResult:")
        print(f"  RMS vs observed: {result.rms:.4f}")
        print(f"  Si II depth: {result.si_depth:.1%}")
        print(f"  Si II velocity: {result.si_velocity:.0f} km/s")
        print(f"  Converged: {result.converged}")
        print(f"  Runtime: {result.runtime:.1f}s")

        if len(result.spectrum_wave) > 0:
            plot_best_fit(result, fitter, output="fit_test_spectrum.png",
                         feature_weight=fitter.feature_weight)
            print("\nTest passed! Pipeline is functional.")
        else:
            print("\nTest FAILED: no spectrum produced.")
            sys.exit(1)
        return

    # ===== Refine mode =====
    if args.refine is not None:
        total_t0 = time.time()
        row_idx = int(args.refine) - 1  # 1-indexed to 0-indexed

        # Load center point from fit_results_final.csv
        final_csv = PROJECT_ROOT / "fit_results_final.csv"
        if not final_csv.exists():
            print(f"ERROR: {final_csv} not found. Run full search first.")
            sys.exit(1)
        import csv
        with open(final_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if row_idx >= len(rows):
            print(f"ERROR: Row {args.refine} not found (only {len(rows)} rows)")
            sys.exit(1)
        r = rows[row_idx]
        center = ModelParams(
            log_L=float(r['log_L']), v_inner=float(r['v_inner']),
            log_rho_0=float(r['log_rho_0']), density_exp=float(r['density_exp']),
            T_e_ratio=float(r['T_e_ratio']), v_core=float(r['v_core']),
            v_wall=float(r['v_wall']), X_Fe_core=float(r['X_Fe_core']),
            X_Si_wall=float(r['X_Si_wall']),
            v_break=float(r.get('v_break', 15000)),
            density_exp_outer=float(r.get('density_exp_outer', -10)),
        )

        print("=" * 70)
        print(f"REFINE MODE: Local search around model #{args.refine}")
        print("=" * 70)
        print(f"  Center: L={center.log_L:.3f} v={center.v_inner:.0f} "
              f"rho={center.log_rho_0:.2f} exp={center.density_exp:.1f} "
              f"Te={center.T_e_ratio:.2f}")
        print(f"  Zones: vc={center.v_core:.0f} vw={center.v_wall:.0f} "
              f"Fe_c={center.X_Fe_core:.2f} Si_w={center.X_Si_wall:.2f}")
        print(f"  Density: vb={center.v_break:.0f} exp_o={center.density_exp_outer:.1f}")
        print(f"  Original RMS={float(r['rms']):.4f} Si_depth={float(r['si_depth']):.1%}")

        # Define narrow ranges: ±15% around center (clamped to global bounds)
        refine_ranges = []
        for i, pname in enumerate(PARAM_NAMES):
            val = getattr(center, pname)
            glob_lo, glob_hi = PARAM_RANGES[i]
            span = glob_hi - glob_lo
            delta = span * 0.15  # ±15% of global range
            lo = max(glob_lo, val - delta)
            hi = min(glob_hi, val + delta)
            refine_ranges.append((lo, hi))
            print(f"    {pname:12s}: {val:.3f}  -> [{lo:.3f}, {hi:.3f}]")

        n_ref = args.refine_samples
        print(f"\n  Generating {n_ref} LHS samples in narrow range...")

        # Phase R1: Local LHS, 100K packets x 10 iters
        print(f"\n{'=' * 70}")
        print(f"REFINE PHASE 1: {n_ref} local samples (100K packets x 10 iters)")
        print(f"{'=' * 70}")
        samples = latin_hypercube(n_ref, refine_ranges, rng=np.random.default_rng(77))
        print(f"  Generated {len(samples)} valid samples")

        ref_results = []
        for i, params in enumerate(samples):
            print(f"\n  [{i+1:3d}/{n_ref}] L={params.log_L:.3f} v={params.v_inner:.0f} "
                  f"rho={params.log_rho_0:.2f} exp={params.density_exp:.1f} Te={params.T_e_ratio:.2f} "
                  f"vc={params.v_core:.0f} vw={params.v_wall:.0f} "
                  f"Fe_c={params.X_Fe_core:.2f} Si_w={params.X_Si_wall:.2f} "
                  f"vb={params.v_break:.0f} exp_o={params.density_exp_outer:.1f}",
                  end="", flush=True)
            result = fitter.run_model(params, n_packets=100000, n_iters=10, tag=f"ref_{i}")
            ref_results.append(result)
            print(f"  -> RMS={result.rms:.4f} Si_d={result.si_depth:.1%} "
                  f"v_Si={result.si_velocity:.0f} ({result.runtime:.1f}s)")

        ref_results.sort(key=lambda x: x.rms)
        save_results_csv(ref_results, "fit_results_refine_phase1.csv")

        print(f"\nRefine Phase 1 complete. Top-5:")
        for i, result in enumerate(ref_results[:5]):
            p = result.params
            print(f"  #{i+1}: RMS={result.rms:.4f} Si_d={result.si_depth:.1%} "
                  f"L={p.log_L:.3f} v={p.v_inner:.0f} vc={p.v_core:.0f} vw={p.v_wall:.0f} "
                  f"Fe_c={p.X_Fe_core:.2f} Si_w={p.X_Si_wall:.2f} "
                  f"vb={p.v_break:.0f} exp_o={p.density_exp_outer:.1f}")

        # Phase R2: Production for top-3
        print(f"\n{'=' * 70}")
        print(f"REFINE PHASE 2: Top-3 production (500K packets x 20 iters)")
        print(f"{'=' * 70}")
        ref_prod = []
        for i, prev in enumerate(ref_results[:3]):
            params = prev.params
            print(f"\n  [{i+1}/3] (R1 RMS={prev.rms:.4f})", end="", flush=True)
            result = fitter.run_model(params, n_packets=500000, n_iters=20, tag=f"refp_{i}")
            ref_prod.append(result)
            print(f"  -> RMS={result.rms:.4f} Si_d={result.si_depth:.1%} "
                  f"v_Si={result.si_velocity:.0f} ({result.runtime:.1f}s)")

        ref_prod.sort(key=lambda x: x.rms)
        save_results_csv(ref_prod, "fit_results_refine_final.csv")

        # Plots
        best = ref_prod[0]
        plot_best_fit(best, fitter, output="fit_refine_spectrum.png",
                     feature_weight=fitter.feature_weight)
        plot_si_detail(ref_prod, fitter, output="fit_refine_siII.png")

        print(f"\n{'=' * 70}")
        print("REFINEMENT COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total time: {time.time()-total_t0:.0f}s ({(time.time()-total_t0)/60:.1f} min)")
        p = best.params
        _, _, _, _, X_O_c = p.zone_abundances('core')
        _, _, _, _, X_O_w = p.zone_abundances('wall')
        _, _, _, _, X_O_o = p.zone_abundances('outer')
        print(f"\nBest refined parameters:")
        print(f"  log_L       = {p.log_L:.4f}")
        print(f"  v_inner     = {p.v_inner:.0f} km/s")
        print(f"  log_rho_0   = {p.log_rho_0:.4f}")
        print(f"  density_exp = {p.density_exp:.2f} (inner, v < v_break)")
        print(f"  v_break     = {p.v_break:.0f} km/s")
        print(f"  exp_outer   = {p.density_exp_outer:.2f} (outer, v > v_break)")
        print(f"  T_e/T_rad   = {p.T_e_ratio:.3f}")
        print(f"  Core  (<{p.v_core:.0f}): Fe={p.X_Fe_core:.3f} O={X_O_c:.3f}")
        print(f"  Wall  (<{p.v_wall:.0f}): Si={p.X_Si_wall:.3f} O={X_O_w:.3f}")
        print(f"  Outer (>{p.v_wall:.0f}): O={X_O_o:.3f}")
        print(f"\n  RMS: {best.rms:.4f}  Si II depth: {best.si_depth:.1%}  "
              f"v_Si: {best.si_velocity:.0f} km/s")
        print(f"\n  vs original #{args.refine}: RMS={float(r['rms']):.4f} "
              f"Si_depth={float(r['si_depth']):.1%}")
        return

    # ===== Full 3-phase search =====
    total_t0 = time.time()

    # Phase 1
    p1_results = phase1_coarse(fitter, n=args.n_samples)
    save_results_csv(p1_results, "fit_results_phase1.csv")
    plot_sensitivity(p1_results, "fit_sensitivity.png")

    print(f"\nPhase 1 complete. Top-5:")
    for i, r in enumerate(p1_results[:5]):
        p = r.params
        print(f"  #{i+1}: RMS={r.rms:.4f} L={p.log_L:.3f} v={p.v_inner:.0f} "
              f"rho={p.log_rho_0:.2f} exp={p.density_exp:.1f} Te={p.T_e_ratio:.2f} "
              f"vc={p.v_core:.0f} vw={p.v_wall:.0f} "
              f"Fe_c={p.X_Fe_core:.2f} Si_w={p.X_Si_wall:.2f} "
              f"vb={p.v_break:.0f} exp_o={p.density_exp_outer:.1f}")

    if args.phase1:
        print(f"\nPhase 1 only. Total time: {time.time()-total_t0:.0f}s")
        return

    # Phase 2
    p2_results = phase2_refine(fitter, p1_results, n_top=20)
    save_results_csv(p2_results, "fit_results_phase2.csv")

    print(f"\nPhase 2 complete. Top-5:")
    for i, r in enumerate(p2_results[:5]):
        p = r.params
        print(f"  #{i+1}: RMS={r.rms:.4f} L={p.log_L:.3f} v={p.v_inner:.0f} "
              f"rho={p.log_rho_0:.2f} exp={p.density_exp:.1f} Te={p.T_e_ratio:.2f} "
              f"vc={p.v_core:.0f} vw={p.v_wall:.0f} "
              f"Fe_c={p.X_Fe_core:.2f} Si_w={p.X_Si_wall:.2f} "
              f"vb={p.v_break:.0f} exp_o={p.density_exp_outer:.1f}")

    # Phase 3
    p3_results = phase3_production(fitter, p2_results, n_top=3)
    save_results_csv(p3_results, "fit_results_final.csv")

    # Final plots
    best = p3_results[0]
    plot_best_fit(best, fitter, output="fit_best_spectrum.png",
                 feature_weight=fitter.feature_weight)
    plot_si_detail(p3_results, fitter, output="fit_best_siII.png")

    print(f"\n{'=' * 70}")
    print("FITTING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {time.time()-total_t0:.0f}s ({(time.time()-total_t0)/60:.1f} min)")
    print(f"\nBest-fit parameters:")
    p = best.params
    print(f"  log_L       = {p.log_L:.4f}  (L = {p.L_erg_s:.4e} erg/s)")
    print(f"  v_inner     = {p.v_inner:.0f} km/s")
    print(f"  log_rho_0   = {p.log_rho_0:.4f}  (rho_0 = {p.rho_0:.4e} g/cm^3)")
    print(f"  density_exp = {p.density_exp:.2f}  (inner, v < v_break)")
    print(f"  v_break     = {p.v_break:.0f} km/s")
    print(f"  exp_outer   = {p.density_exp_outer:.2f}  (outer, v > v_break)")
    print(f"  T_e/T_rad   = {p.T_e_ratio:.3f}")
    print(f"  v_core      = {p.v_core:.0f} km/s")
    print(f"  v_wall      = {p.v_wall:.0f} km/s")
    _, _, _, _, X_O_c = p.zone_abundances('core')
    _, _, _, _, X_O_w = p.zone_abundances('wall')
    _, _, _, _, X_O_o = p.zone_abundances('outer')
    print(f"  Core  (<{p.v_core:.0f} km/s): Fe={p.X_Fe_core:.3f} Si=0.050 S=0.050 Ca=0.030 O={X_O_c:.3f}")
    print(f"  Wall  (<{p.v_wall:.0f} km/s): Si={p.X_Si_wall:.3f} Fe=0.050 S=0.050 Ca=0.030 O={X_O_w:.3f}")
    print(f"  Outer (>{p.v_wall:.0f} km/s): Si=0.020 Fe=0.010 S=0.020 Ca=0.010 O={X_O_o:.3f}")
    print(f"\nMetrics:")
    print(f"  RMS vs observed: {best.rms:.4f}")
    print(f"  Si II depth: {best.si_depth:.1%}")
    print(f"  Si II velocity: {best.si_velocity:.0f} km/s")

    print(f"\nOutput files:")
    print(f"  fit_results_phase1.csv  ({args.n_samples} rows, 11D broken density)")
    print(f"  fit_results_phase2.csv  (20 rows)")
    print(f"  fit_results_final.csv   (3 rows)")
    print(f"  fit_sensitivity.png     (11 panels)")
    print(f"  fit_best_spectrum.png")
    print(f"  fit_best_siII.png")


if __name__ == '__main__':
    main()
