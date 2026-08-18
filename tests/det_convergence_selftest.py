#!/usr/bin/env python3
"""Synthetic positive and negative controls for check_det_convergence.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "scripts/check_det_convergence.py"
NITER = 5
NSHELL = 2
NBIN = 3

SHELL_FIELDS = [
    "shell_id", "r_inner_cm", "r_outer_cm", "v_inner_cm_s", "v_outer_cm_s",
    "T_e_K", "n_e_cm3", "n_atom_cm3", "u_atom_erg",
    "q_ad_temperature_gradient", "q_ad_velocity_divergence",
    "q_ad_electron_fraction_gradient", "q_ad_internal_energy_gradient",
    "q_ad_signed_total", "q_ad_heating", "q_ad_cooling", "photo_heat",
    "line_abs_heat", "ff_abs_heat", "compton_heat", "gamma_heat",
    "nonthermal_heat", "recomb_cool", "line_emit_cool", "coll_line_cool",
    "ff_emit_cool", "compton_cool", "sum_heating", "sum_cooling", "residual",
]
SPECTRAL_FIELDS = [
    "shell_id", "bin_id", "nu_lo_Hz", "nu_hi_Hz", "J_nu", "chi_es_cm1",
    "chi_bb_cm1", "chi_bf_cm1", "chi_ff_cm1", "chi_total_cm1", "eta_bb",
    "eta_bf", "eta_ff", "eta_true_total",
]


def write_fixture(directory: Path, jump: bool = False) -> None:
    for iteration in range(NITER):
        factor = 1.0 + 1.0e-3 * iteration
        if jump and iteration == NITER - 1:
            factor *= 1.1
        shell_name = f"physics_DET_iter{iteration:04d}.shell.csv"
        spectral_name = f"physics_DET_iter{iteration:04d}.spectral.csv"
        manifest_name = f"physics_DET_iter{iteration:04d}.manifest.json"
        with (directory / shell_name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SHELL_FIELDS)
            writer.writeheader()
            for shell in range(NSHELL):
                row = {field: 0.0 for field in SHELL_FIELDS}
                row.update({
                    "shell_id": shell,
                    "r_inner_cm": 1.0e14 * (shell + 1),
                    "r_outer_cm": 1.0e14 * (shell + 2),
                    "v_inner_cm_s": 1.0e8 * (shell + 1),
                    "v_outer_cm_s": 1.0e8 * (shell + 2),
                    "T_e_K": (9000.0 + 100.0 * shell) * factor,
                    "n_e_cm3": (1.0e8 + 1.0e7 * shell) * factor,
                    "n_atom_cm3": 2.0e8 + 1.0e7 * shell,
                    "u_atom_erg": (2.0e-3 + 1.0e-4 * shell) * factor,
                    "q_ad_temperature_gradient": 0.20 * factor,
                    "q_ad_velocity_divergence": 0.30 * factor,
                    "q_ad_electron_fraction_gradient": -0.05 * factor,
                    "q_ad_internal_energy_gradient": 0.10 * factor,
                    "q_ad_signed_total": 0.55 * factor,
                    "q_ad_heating": 0.0,
                    "q_ad_cooling": 0.55 * factor,
                    "sum_heating": 100.0 * factor,
                    "sum_cooling": 100.0 * factor,
                    "residual": 1.0e-8,
                })
                writer.writerow(row)
        with (directory / spectral_name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SPECTRAL_FIELDS)
            writer.writeheader()
            for shell in range(NSHELL):
                for bin_id in range(NBIN):
                    writer.writerow({
                        "shell_id": shell,
                        "bin_id": bin_id,
                        "nu_lo_Hz": 1.0e14 + bin_id * 1.0e13,
                        "nu_hi_Hz": 1.0e14 + (bin_id + 1) * 1.0e13,
                        "J_nu": (1.0 + shell + bin_id) * factor,
                        "chi_es_cm1": 0.1 * factor,
                        "chi_bb_cm1": 0.2 * factor,
                        "chi_bf_cm1": 0.3 * factor,
                        "chi_ff_cm1": 0.4 * factor,
                        "chi_total_cm1": 1.0 * factor,
                        "eta_bb": 0.2 * factor,
                        "eta_bf": 0.3 * factor,
                        "eta_ff": 0.5 * factor,
                        "eta_true_total": 1.0 * factor,
                    })
        manifest = {
            "schema": "LUMINA_PHYSICS_COMPARISON_V1",
            "transaction_status": "COMMITTED",
            "code": "LUMINA",
            "lane": "DET",
            "iteration": iteration,
            "epoch_s": 1683072.0,
            "n_shells": NSHELL,
            "n_bins": NBIN,
            "atomic_model_sha256": "a" * 64,
            "geometry_sha256": "b" * 64,
            "grid_manifest_sha256": "c" * 64,
            "radiation_generation": iteration + 1,
            "population_generation": iteration + 1,
            "te_generation": iteration + 1,
            "opacity_generation": iteration + 1,
            "emissivity_generation": iteration + 1,
            "shell_file": shell_name,
            "spectral_file": spectral_name,
        }
        (directory / manifest_name).write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def run(directory: Path, expected_iterations: int = NITER,
        tail_transitions: int = 3) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--dump-dir", str(directory),
         "--expected-iterations", str(expected_iterations),
         "--tail-transitions", str(tail_transitions),
         "--expected-bins", str(NBIN)],
        text=True, capture_output=True, check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="det-convergence-selftest-") as temporary:
        root = Path(temporary)

        positive = root / "positive"
        positive.mkdir()
        write_fixture(positive)
        result = run(positive)
        assert result.returncode == 0, (result.stdout, result.stderr)
        report = json.loads((positive / "det_convergence_report.json").read_text())
        assert report["status"] == "CONVERGED"

        nonconverged = root / "nonconverged"
        nonconverged.mkdir()
        write_fixture(nonconverged, jump=True)
        result = run(nonconverged)
        assert result.returncode == 2, (result.stdout, result.stderr)
        report = json.loads((nonconverged / "det_convergence_report.json").read_text())
        assert report["status"] == "NOT_CONVERGED"
        assert report["transitions"][-1]["failures"]

        incomplete = root / "incomplete"
        incomplete.mkdir()
        write_fixture(incomplete)
        (incomplete / f"physics_DET_iter{NITER - 1:04d}.manifest.json").unlink()
        result = run(incomplete)
        assert result.returncode == 3, (result.stdout, result.stderr)
        report = json.loads((incomplete / "det_convergence_report.json").read_text())
        assert report["status"] == "INPUT_ERROR"

        snapshot = root / "snapshot"
        snapshot.mkdir()
        write_fixture(snapshot)
        for iteration in range(1, NITER):
            for suffix in ("manifest.json", "shell.csv", "spectral.csv"):
                (snapshot / f"physics_DET_iter{iteration:04d}.{suffix}").unlink()
        result = run(snapshot, expected_iterations=1, tail_transitions=0)
        assert result.returncode == 0, (result.stdout, result.stderr)
        report = json.loads((snapshot / "det_convergence_report.json").read_text())
        assert report["status"] == "CONVERGED"
        assert report["transitions"] == []

    print(
        "DET_CONVERGENCE_SELFTEST PASS positive=1 nonconverged=1 "
        "input_error=1 single_snapshot=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
