#!/usr/bin/env python3
"""Commit CMFGEN J_nu and enforce the A2 L-0 positive/negative gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.chain_replay_parity59.common import (  # noqa: E402
    parse_rvtj_block,
    read_eddfactor,
    read_info,
)

# Mirror src/lumina_frequency_grid.h + src/radiation_field.h.  The former
# 4000-bin literals predated SH-GRID and made the binary fixture reject the
# input byte count after the canonical owner moved to 3866 bins.
NLTE_N_BINS = 1234
NLTE_NU_MIN = 5.8412785919616062e13
NLTE_NU_MAX = 4.0362581455823112e16
REFINEMENT_K = 2
J_LO = -1398
J_HI = 2468
N_BINS = J_HI - J_LO
DLOG = math.log(NLTE_NU_MAX / NLTE_NU_MIN) / (REFINEMENT_K * NLTE_N_BINS)
NU_MIN = NLTE_NU_MIN * math.exp(J_LO * DLOG)
NU_MAX = NLTE_NU_MIN * math.exp(J_HI * DLOG)
C_A_S = 2.99792458e18
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
C_CGS = 2.99792458e10
BANDS = (
    ("EUV", 450.0, 918.0),
    ("FUV", 918.0, 1290.0),
    ("UV", 1290.0, 2000.0),
    ("OPT", 2000.0, 10000.0),
    ("IR", 10000.0, 25000.0),
)
VALID, EXACT_ZERO, UNSAMPLED, OUT_OF_GRID = 1, 2, 3, 4
EXPECTED_EDD_BYTES = 142_832_872


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_edges() -> np.ndarray:
    edges = np.empty(N_BINS + 1, dtype=np.float64)
    ratio_log = math.log(NLTE_NU_MAX / NLTE_NU_MIN)
    for b in range(N_BINS + 1):
        j = J_LO + b
        if 0 <= j <= REFINEMENT_K * NLTE_N_BINS and \
                j % REFINEMENT_K == 0:
            bf_edge = j // REFINEMENT_K
            if bf_edge == 0:
                edges[b] = NLTE_NU_MIN
            elif bf_edge == NLTE_N_BINS:
                edges[b] = NLTE_NU_MAX
            else:
                edges[b] = NLTE_NU_MIN * math.exp(
                    bf_edge * ratio_log / NLTE_N_BINS)
        else:
            edges[b] = NLTE_NU_MIN * math.exp(j * DLOG)
    edges[0], edges[-1] = NU_MIN, NU_MAX
    return edges


def geometry(deck: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with (deck / "geometry.csv").open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["shell_id"]))
    require([int(row["shell_id"]) for row in rows] == list(range(len(rows))),
            "geometry shell ids are not contiguous")
    v_inner = np.asarray([float(row["v_inner"]) for row in rows])
    v_outer = np.asarray([float(row["v_outer"]) for row in rows])
    return v_inner, v_outer, 0.5e-5 * (v_inner + v_outer)


def map_velocity(j_depth: np.ndarray, native_v: np.ndarray,
                 target_v: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(native_v)
    velocity = native_v[order]
    require(velocity[0] <= target_v <= velocity[-1],
            f"target velocity {target_v} outside RVTJ")
    right = int(np.searchsorted(velocity, target_v, side="left"))
    if right < velocity.size and velocity[right] == target_v:
        values = j_depth[:, order[right]].copy()
        return values, np.isfinite(values) & (values >= 0.0)
    require(0 < right < velocity.size, "RVTJ bracket failure")
    left = right - 1
    weight = (target_v - velocity[left]) / (velocity[right] - velocity[left])
    a, b = j_depth[:, order[left]], j_depth[:, order[right]]
    positive = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    zero = (a == 0.0) & (b == 0.0)
    values = np.zeros(a.size)
    values[positive] = np.exp((1.0 - weight) * np.log(a[positive]) +
                              weight * np.log(b[positive]))
    return values, positive | zero


def piecewise_linear_average(x: np.ndarray, y: np.ndarray,
                             sample_valid: np.ndarray,
                             target_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(np.all(np.diff(x) > 0.0), "native frequency not increasing")
    require(y.shape == x.shape == sample_valid.shape, "native shape mismatch")
    dx = np.diff(x)
    cumulative = np.concatenate((
        np.asarray([0.0]), np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    ))

    def primitive(q: np.ndarray) -> np.ndarray:
        index = np.searchsorted(x, q, side="right") - 1
        index = np.clip(index, 0, x.size - 2)
        delta = q - x[index]
        slope = (y[index + 1] - y[index]) / (x[index + 1] - x[index])
        return cumulative[index] + y[index] * delta + 0.5 * slope * delta * delta

    inside = (target_edges[:-1] >= x[0]) & (target_edges[1:] <= x[-1])
    average = np.zeros(N_BINS)
    average[inside] = (primitive(target_edges[1:][inside]) -
                       primitive(target_edges[:-1][inside])) / np.diff(target_edges)[inside]
    state = np.full(N_BINS, OUT_OF_GRID, dtype=np.int32)
    for b in np.flatnonzero(inside):
        left = max(0, int(np.searchsorted(x, target_edges[b], side="right") - 1))
        right = min(x.size - 1,
                    int(np.searchsorted(x, target_edges[b + 1], side="left")))
        if not np.all(sample_valid[left:right + 1]):
            average[b] = 0.0
            state[b] = UNSAMPLED
        elif average[b] > 0.0:
            state[b] = VALID
        elif average[b] == 0.0:
            state[b] = EXACT_ZERO
        else:
            raise GateError("conservative integration produced negative J")
    return average, state


def from_eddfactor(cmf_dir: Path, shell_v: np.ndarray,
                   edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    edd = cmf_dir / "EDDFACTOR"
    info_path = cmf_dir / "EDDFACTOR_INFO"
    rvtj = cmf_dir / "RVTJ"
    info = read_info(info_path)
    require(info == {"ND": 90, "RECL": 728, "WORD": 8, "little": True},
            f"EDDFACTOR_INFO schema mismatch: {info}")
    require(edd.stat().st_size == EXPECTED_EDD_BYTES,
            f"EDDFACTOR bytes {edd.stat().st_size} != {EXPECTED_EDD_BYTES}")
    j_depth, fl, nd = read_eddfactor(edd)
    require(nd == 90 and j_depth.shape == (196_185, 90),
            f"EDDFACTOR payload shape {j_depth.shape}")
    raw = np.memmap(edd, mode="r", dtype="<f8", shape=(14, 91))
    finish = float(raw[4, 0])
    del raw
    require(finish == 1.0, f"EDDFACTOR FINISH_REC={finish}")
    nu = np.asarray(fl) * 1.0e15
    order = np.argsort(nu)
    nu, j_depth = nu[order], np.asarray(j_depth[order])
    native_v = parse_rvtj_block(rvtj, "Velocity (km/s)", nd)
    expected = np.zeros((shell_v.size, N_BINS))
    state = np.zeros((shell_v.size, N_BINS), dtype=np.int32)
    for shell, velocity in enumerate(shell_v):
        mapped, valid = map_velocity(j_depth, native_v, float(velocity))
        expected[shell], state[shell] = piecewise_linear_average(
            nu, mapped, valid, edges
        )
    return expected, state, {
        "mode": "DIRECT_EDDFACTOR",
        "EDDFACTOR": str(edd),
        "EDDFACTOR_sha256": sha256(edd),
        "EDDFACTOR_INFO_sha256": sha256(info_path),
        "RVTJ_sha256": sha256(rvtj),
        "FINISH_REC": finish,
        "valid_frequency_records": int(nu.size),
    }


def from_fine_npz(path: Path, edges: np.ndarray,
                  n_shells: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        source_edges = np.asarray(data["nu_edges_hz"])
        source_j = np.asarray(data["j_nu"][:n_shells])
        source_state = np.asarray(data["j_state"][:n_shells])
    expected = np.zeros((n_shells, N_BINS))
    state = np.zeros((n_shells, N_BINS), dtype=np.int32)
    for shell in range(n_shells):
        for b in range(N_BINS):
            lo, hi = edges[b], edges[b + 1]
            first = int(np.searchsorted(source_edges, lo, side="right") - 1)
            last = int(np.searchsorted(source_edges, hi, side="left"))
            if first < 0 or last >= source_edges.size:
                state[shell, b] = OUT_OF_GRID
                continue
            indices = np.arange(first, last, dtype=np.int64)
            left = np.maximum(source_edges[indices], lo)
            right = np.minimum(source_edges[indices + 1], hi)
            overlap = np.maximum(right - left, 0.0)
            active = overlap > 0.0
            indices, overlap = indices[active], overlap[active]
            states = source_state[shell, indices]
            if not indices.size or np.any(~np.isin(states, [VALID, EXACT_ZERO])):
                state[shell, b] = UNSAMPLED
                continue
            expected[shell, b] = float(np.sum(source_j[shell, indices] * overlap) /
                                       (hi - lo))
            state[shell, b] = VALID if expected[shell, b] > 0.0 else EXACT_ZERO
    return expected, state, {
        "mode": "HASH_BOUND_A2_02C_FINE_NPZ",
        "path": str(path),
        "sha256": sha256(path),
    }


def planck_bin_average(edges: np.ndarray, temperature: float,
                       dilution: float) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(16)
    midpoint = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1:] - edges[:-1])
    nu = midpoint[:, None] + half[:, None] * nodes[None, :]
    x = H_CGS * nu / (K_CGS * temperature)
    denominator = np.expm1(np.minimum(x, 709.0))
    bnu = 2.0 * H_CGS * nu ** 3 / (C_CGS ** 2 * denominator)
    bnu[x > 709.0] = 0.0
    return dilution * 0.5 * np.sum(bnu * weights[None, :], axis=1)


def band_weights(edges: np.ndarray, lam_lo: float, lam_hi: float) -> np.ndarray:
    nu_lo, nu_hi = C_A_S / lam_hi, C_A_S / lam_lo
    return np.maximum(np.minimum(edges[1:], nu_hi) -
                      np.maximum(edges[:-1], nu_lo), 0.0)


def metrics(expected: np.ndarray, committed: np.ndarray,
            state: np.ndarray, edges: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    widths = np.diff(edges)
    for shell in range(expected.shape[0]):
        active = np.isin(state[shell], [VALID, EXACT_ZERO])
        denominator = float(np.sum(widths[active] * expected[shell, active]))
        require(denominator > 0.0, f"shell {shell} has zero L-0 denominator")
        e1 = float(np.sum(widths[active] * np.abs(
            committed[shell, active] - expected[shell, active])) / denominator)
        positive = active & (expected[shell] > 0.0) & (committed[shell] > 0.0)
        p95 = float(np.percentile(np.abs(np.log10(
            committed[shell, positive] / expected[shell, positive])), 95))
        bands: dict[str, float] = {}
        for name, lo, hi in BANDS:
            weight = band_weights(edges, lo, hi)
            mask = (weight > 0.0) & active
            denom = float(np.sum(weight[mask] * expected[shell, mask]))
            require(denom > 0.0, f"shell {shell} band {name} denominator zero")
            bands[name] = float(abs(np.sum(weight[mask] * (
                committed[shell, mask] - expected[shell, mask]))) / denom)
        rows.append({"shell": shell, "E_1": e1, "band_E_B": bands,
                     "P95_log10_dex": p95})
    return rows


def replay_commit(binary: Path, work: Path, expected: np.ndarray,
                  state: np.ndarray, v_inner: np.ndarray, v_outer: np.ndarray,
                  epoch: float) -> tuple[np.ndarray, np.ndarray, str]:
    source = work / "replay.in"
    output = work / "replay.out"
    with source.open("wb") as stream:
        stream.write(b"A204IN01")
        stream.write(struct.pack("=QQd", expected.shape[0], 1, epoch))
        stream.write(np.asarray(v_inner, dtype="=f8").tobytes())
        stream.write(np.asarray(v_outer, dtype="=f8").tobytes())
        stream.write(np.asarray(expected, dtype="=f8").tobytes())
        stream.write(np.asarray(state, dtype="=i4").tobytes())
    proc = subprocess.run([str(binary), str(source), str(output)], cwd=work,
                          text=True, capture_output=True, check=False)
    require(proc.returncode == 0, f"commit fixture rc={proc.returncode}: {proc.stderr}")
    raw = output.read_bytes()
    require(raw[:8] == b"A204OUT1", "commit output magic")
    n_shells, generation = struct.unpack_from("=QQ", raw, 8)
    require(n_shells == expected.shape[0] and generation == 1,
            "commit output generation/shape")
    offset = 24
    cells = expected.size
    committed = np.frombuffer(raw, dtype="=f8", count=cells, offset=offset).copy()
    offset += cells * 8
    committed_state = np.frombuffer(raw, dtype="=i4", count=cells,
                                    offset=offset).copy()
    require(offset + cells * 4 == len(raw), "commit output trailing bytes")
    return committed.reshape(expected.shape), committed_state.reshape(state.shape), proc.stdout.strip()


def negative_control(expected_s0: np.ndarray, edges: np.ndarray,
                     deck: Path) -> dict[str, Any]:
    with (deck / "plasma_state.csv").open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["shell_id"]))
    dilution = float(rows[0]["W"])
    energy_temperature = np.asarray([float(row["T_rad"]) for row in rows])
    w = np.asarray([float(row["W"]) for row in rows])
    color = energy_temperature / np.power(w, 0.25)
    require(float(np.ptp(color)) <= 1.0e-8,
            "deck intrinsic color is not shell-constant")
    temperature = float(color[0])
    require(abs(temperature - 14172.549003) <= 0.01,
            f"deck intrinsic color {temperature} != fossil witness 14172.549003")
    injected = planck_bin_average(edges, temperature, dilution)
    errors: dict[str, float] = {}
    for name, lo, hi in BANDS:
        weight = band_weights(edges, lo, hi)
        denominator = float(np.sum(weight * expected_s0))
        require(denominator > 0.0, f"negative band {name} denominator zero")
        errors[name] = float(abs(np.sum(weight * (injected - expected_s0))) /
                             denominator)
    require(all(value > 0.10 for value in errors.values()),
            f"Planck negative did not fail all five bands: {errors}")
    return {
        "injection": "deck W[0] * B_nu(T_rad_color)",
        "W_s0": dilution,
        "T_rad_color_K": temperature,
        "band_E_B": errors,
        "failed_bands": [name for name, value in errors.items() if value > 0.10],
        "verdict": "EXPECTED_FAIL_OBSERVED_ALL_5",
    }


def synthetic(shells: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    centre = np.sqrt(edges[:-1] * edges[1:])
    shape = 1.0e-4 * (1.0 + 0.3 * np.sin(7.0 * np.log(centre / centre[0])))
    values = np.vstack([(1.0 + 0.01 * shell) * shape for shell in range(shells)])
    state = np.full(values.shape, VALID, dtype=np.int32)
    # 2026-08-06 driver fix: exercise every validity state the real EDDFACTOR
    # input produces, so a state-collapsing commit fails the self-test too
    # (the OUT_OF_GRID->UNSAMPLED collapse was only caught on real data).
    values[:, :8] = 0.0;  state[:, :8] = OUT_OF_GRID
    values[:, -8:] = 0.0; state[:, -8:] = OUT_OF_GRID
    values[:, 100:104] = 0.0; state[:, 100:104] = EXACT_ZERO
    values[:, 200:202] = 0.0; state[:, 200:202] = UNSAMPLED
    return values, state, {"mode": "SYNTHETIC_SELF_TEST"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    binary = args.fixture.resolve()
    deck = args.deck.resolve()
    require(binary.is_file() and deck.is_dir(), "fixture or deck absent")
    edges = canonical_edges()
    v_inner_all, v_outer_all, shell_v_all = geometry(deck)
    n_shells = 9 if args.self_test else 44
    v_inner, v_outer = v_inner_all[:n_shells], v_outer_all[:n_shells]
    if args.self_test:
        expected, state, provenance = synthetic(n_shells, edges)
    elif args.fine_npz:
        expected, state, provenance = from_fine_npz(
            args.fine_npz.resolve(), edges, n_shells
        )
    else:
        expected, state, provenance = from_eddfactor(
            args.cmf_dir.resolve(), shell_v_all[:n_shells], edges
        )
    config = json.loads((deck / "config.json").read_text())
    epoch = float(config["time_explosion_s"])
    with tempfile.TemporaryDirectory(prefix="a2_04_l0_", dir=args.scratch_root) as temporary:
        committed, committed_state, marker = replay_commit(
            binary, Path(temporary), expected, state, v_inner, v_outer, epoch
        )
    require(np.array_equal(state, committed_state), "commit changed validity states")
    rows = metrics(expected, committed, state, edges)
    require(all(row["E_1"] <= 0.10 and row["P95_log10_dex"] <= 0.15 and
                all(value <= 0.10 for value in row["band_E_B"].values())
                for row in rows), "positive L-0 threshold failure")
    negative = negative_control(expected[0], edges, deck)
    return {
        "schema": "lumina-a2-04-l0-replay-v1",
        "oracle_qualification": "CMFGEN_SNAPSHOT_REPLAY=ELIGIBLE; PHYSICAL_ORACLE=INELIGIBLE",
        "source": provenance,
        "commit_marker": marker,
        "canonical_bins": N_BINS,
        "wiring_replay_shells": list(range(n_shells)),
        "safe_physics_shells": list(range(min(9, n_shells))),
        "thresholds": {"E_1_max": 0.10, "band_E_B_max": 0.10,
                       "P95_log10_dex_max": 0.15},
        "positive_rows": rows,
        "positive_summary": {
            "max_E_1": max(row["E_1"] for row in rows),
            "max_band_E_B": max(value for row in rows for value in row["band_E_B"].values()),
            "max_P95_log10_dex": max(row["P95_log10_dex"] for row in rows),
            "verdict": "PASS",
        },
        "negative_control": negative,
        "guard_hits": 0,
        "fallback_hits": 0,
        "verdict": "PASS",
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--deck", type=Path,
                        default=ROOT / "data/tardis_reference_toy06_19p48d")
    parser.add_argument("--cmf-dir", type=Path,
                        default=Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"))
    parser.add_argument("--fine-npz", type=Path)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    try:
        payload = run(args)
        rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
        return 0
    except (GateError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"A2_04_L0_REPLAY FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
