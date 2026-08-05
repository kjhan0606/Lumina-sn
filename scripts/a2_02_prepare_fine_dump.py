#!/usr/bin/env python3
"""Build the A2-02 fine-bin aggregate from existing, immutable dumps.

The program is an offline packer.  It neither launches Lumina nor writes to the
CMFGEN run, the capture directory, or the selected atomic deck.  EDDFACTOR is
decoded by the already exercised chain-replay reader; this file only adds the
A2-02 validity states and conservative bin-average packing around that reader.

Exit codes:
  0  aggregate and hash-bound input manifest written
  2  input/schema/provenance/packing failure
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse, rather than reproduce, the working direct-access and RVTJ parsers.
from validation.chain_replay_parity59.common import (  # noqa: E402
    parse_rvtj_block,
    read_eddfactor,
    read_info,
)
from cmf_chieta_check import check_artifact  # noqa: E402


SCHEMA = "lumina-a2-02-resolution-input-v1"
EDD_EXPECTED_ND = 90
EDD_EXPECTED_NCF = 196_185
EDD_HEADER_RECORDS = 14
EDD_EXPECTED_RECORDS = EDD_HEADER_RECORDS + EDD_EXPECTED_NCF
EDD_EXPECTED_RECL = 728
EDD_EXPECTED_BYTES = EDD_EXPECTED_RECORDS * EDD_EXPECTED_RECL
CMFD_MAGIC = 0x434D4644
CMFD_VERSION = 1
H_CGS = 6.62607015e-27
FOUR_PI = 4.0 * math.pi
EV_TO_ERG = 1.602176634e-12

MEASURED = np.uint8(1)
EXACT_ZERO = np.uint8(2)
UNSAMPLED = np.uint8(3)
OUT_OF_RANGE = np.uint8(4)
PROFILE_SHELLS = (0, 8, 9, 43)

DEFAULT_EDD = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR")
DEFAULT_CHIETA = Path(
    "/gpfs/kjhan/lumina_runner2/scratch/"
    "chieta_capture_parity59_188605/chieta_iter10"
)
DEFAULT_DECK = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
DEFAULT_LEDGER = ROOT / "docs/A2_02_FREQUENCY_UNION.json"
DEFAULT_TEMPLATE = ROOT / "docs/A2_02_RESOLUTION_INPUT_TEMPLATE.json"
DEFAULT_OUTPUT = ROOT / "validation/a2_02/a2_02_fine_bin_averages.npz"
DEFAULT_MANIFEST = ROOT / "validation/a2_02/A2_02_RESOLUTION_INPUT.json"


class BuildError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"top-level JSON is not an object: {path}")
    return value


def resolve_repo(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=path.name + ".", suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def source_edges_from_centers_widths(centers_desc: np.ndarray,
                                     widths_desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the exact logarithmic LCMFCE01 edges, in ascending order."""
    centers = np.asarray(centers_desc, dtype=np.float64)[::-1]
    widths = np.asarray(widths_desc, dtype=np.float64)[::-1]
    require(centers.ndim == widths.ndim == 1 and centers.size >= 2,
            "CHIETA frequency arrays must be nonempty 1-D arrays")
    require(np.all(np.isfinite(centers)) and np.all(centers > 0.0) and
            np.all(np.diff(centers) > 0.0),
            "CHIETA centers are not finite, positive, and descending on disk")
    require(np.all(np.isfinite(widths)) and np.all(widths > 0.0),
            "CHIETA has a non-positive dnu")
    ratios = centers[1:] / centers[:-1]
    ratio = float(np.exp(np.mean(np.log(ratios))))
    require(np.max(np.abs(ratios / ratio - 1.0)) <= 2.0e-13,
            "CHIETA frequency centers are not one logarithmic grid")
    edges = centers[0] / math.sqrt(ratio) * ratio ** np.arange(centers.size + 1)
    require(np.max(np.abs(np.sqrt(edges[:-1] * edges[1:]) / centers - 1.0)) <= 3.0e-13,
            "CHIETA center-to-edge round trip failed")
    # 2026-08-05 driver fix: the stored dnu IS the geometric edge width (the
    # three alternative definitions deviate by 2.7e-3), but the writer stored
    # it at ~1e-6 relative precision (measured max 1.17e-6 on chieta_iter10)
    # while the centers form an exact log grid (uniformity 8.9e-16).  The
    # centers-derived edges above are therefore canonical; the stored dnu is
    # only a cross-check at its own measured precision.
    require(np.max(np.abs(np.diff(edges) / widths - 1.0)) <= 3.0e-6,
            "CHIETA dnu-to-edge round trip failed")
    return edges, centers


def piecewise_constant_rebin(values: np.ndarray, source_edges: np.ndarray,
                             target_edges: np.ndarray) -> np.ndarray:
    """Conservative subdivision of source bin averages; zero off support.

    The merged A2-02 grid contains every source edge.  Consequently no target
    bin crosses a source discontinuity and copying the enclosing bin average is
    the exact overlap integral, without subtracting two large cumulative sums.
    """
    values = np.asarray(values, dtype=np.float64)
    require(values.shape[-1] + 1 == source_edges.size,
            "piecewise-constant source width mismatch")
    locations = np.searchsorted(target_edges, source_edges)
    require(np.all(locations < target_edges.size) and
            np.array_equal(target_edges[locations].view(np.uint64),
                           source_edges.view(np.uint64)),
            "target grid does not contain every piecewise-constant source edge")
    flat = values.reshape((-1, values.shape[-1]))
    midpoint = np.sqrt(target_edges[:-1] * target_edges[1:])
    index = np.searchsorted(source_edges, midpoint, side="right") - 1
    valid = (index >= 0) & (index < values.shape[-1]) & (
        target_edges[:-1] >= source_edges[0]
    ) & (target_edges[1:] <= source_edges[-1])
    index = np.clip(index, 0, values.shape[-1] - 1)
    result = np.zeros((flat.shape[0], target_edges.size - 1), dtype=np.float64)
    result[:, valid] = flat[:, index[valid]]
    return result.reshape(values.shape[:-1] + (target_edges.size - 1,))


def load_chieta(path: Path) -> dict[str, Any]:
    checked = check_artifact(path)
    header = checked.header
    nr, nnu, t_exp = int(header[3]), int(header[4]), float(header[9])
    arrays = [np.asarray(value, dtype=np.float64) for value in checked.arrays]
    require(nr == 50 and nnu == 1000,
            f"A2-02 requires the parity59 50x1000 capture, got {nr}x{nnu}")
    edges, centers = source_edges_from_centers_widths(arrays[1], arrays[2])
    r_edge = arrays[0]
    require(r_edge.size == nr + 1 and np.all(np.diff(r_edge) > 0.0),
            "CHIETA radial edges are invalid")
    chi = arrays[3].reshape(nr, nnu)[:, ::-1]
    eta = arrays[7].reshape(nr, nnu)[:, ::-1]
    require(np.all(np.isfinite(chi)) and np.all(chi >= 0.0) and
            np.all(np.isfinite(eta)) and np.all(eta >= 0.0),
            "CHIETA total chi/eta contains an invalid value")
    return {
        "nr": nr, "nnu": nnu, "t_exp": t_exp, "r_edge": r_edge,
        "edges": edges, "centers": centers, "chi": chi, "eta": eta,
        "sidecar": Path(str(path) + ".manifest.json"),
    }


def load_geometry(path: Path, chieta: dict[str, Any]) -> np.ndarray:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == chieta["nr"],
            f"geometry rows {len(rows)} != CHIETA nr {chieta['nr']}")
    rows.sort(key=lambda row: int(row["shell_id"]))
    require([int(row["shell_id"]) for row in rows] == list(range(chieta["nr"])),
            "geometry shell ids must be exactly 0..49")
    r_inner = np.asarray([float(row["r_inner"]) for row in rows])
    r_outer = np.asarray([float(row["r_outer"]) for row in rows])
    v_inner = np.asarray([float(row["v_inner"]) for row in rows])
    v_outer = np.asarray([float(row["v_outer"]) for row in rows])
    require(np.array_equal(r_inner.view(np.uint64), chieta["r_edge"][:-1].view(np.uint64)) and
            np.array_equal(r_outer.view(np.uint64), chieta["r_edge"][1:].view(np.uint64)),
            "deck geometry radial edges are not bitwise identical to CHIETA")
    velocity = 0.5 * (v_inner + v_outer) / 1.0e5
    velocity_from_capture = (
        0.5 * (chieta["r_edge"][:-1] + chieta["r_edge"][1:]) /
        chieta["t_exp"] / 1.0e5
    )
    require(np.max(np.abs(velocity / velocity_from_capture - 1.0)) <= 2.0e-15,
            "geometry velocity and CHIETA r/t shell mapping disagree")
    return velocity


def load_eddfactor(edd: Path) -> dict[str, Any]:
    info = read_info(Path(str(edd) + "_INFO"))
    require(info == {"ND": 90, "RECL": 728, "WORD": 8, "little": True},
            f"unexpected EDDFACTOR_INFO schema: {info}")
    size = edd.stat().st_size
    require(size == EDD_EXPECTED_BYTES,
            f"EDDFACTOR size {size} != 14+196185 records x 728 = {EDD_EXPECTED_BYTES}")
    j_depth, fl, nd = read_eddfactor(edd)
    require(nd == EDD_EXPECTED_ND and j_depth.shape == (EDD_EXPECTED_NCF, nd) and
            fl.shape == (EDD_EXPECTED_NCF,),
            f"EDDFACTOR valid payload is {j_depth.shape}, expected (196185,90)")
    dtype = "<f8" if info["little"] else ">f8"
    first = np.memmap(edd, mode="r", dtype=dtype,
                      shape=(EDD_HEADER_RECORDS, EDD_EXPECTED_RECL // 8))
    finish = float(first[4, 0])
    del first
    require(finish == 1.0, f"EDDFACTOR FINISH record is {finish!r}, expected 1.0")
    nu = np.asarray(fl, dtype=np.float64) * 1.0e15
    order = np.argsort(nu)
    nu = nu[order]
    j_depth = np.asarray(j_depth[order], dtype=np.float64)
    require(np.all(np.isfinite(nu)) and np.all(nu > 0.0) and np.all(np.diff(nu) > 0.0),
            "EDDFACTOR frequencies are not finite, positive, and unique")
    return {"nu": nu, "j_depth": j_depth, "nd": nd, "finish": finish,
            "records": EDD_EXPECTED_RECORDS, "info": info}


def velocity_map(native_j: np.ndarray, native_velocity: np.ndarray,
                 target_velocity: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """State-aware form of the chain-replay log-J velocity interpolation."""
    order = np.argsort(native_velocity)
    velocity = native_velocity[order]
    if target_velocity < velocity[0] or target_velocity > velocity[-1]:
        return (np.zeros(native_j.shape[0], dtype=np.float64),
                np.zeros(native_j.shape[0], dtype=bool),
                {"status": "OUT_OF_RANGE"})
    right = int(np.searchsorted(velocity, target_velocity, side="left"))
    if right < velocity.size and velocity[right] == target_velocity:
        values = native_j[:, order[right]].copy()
        valid = np.isfinite(values) & (values >= 0.0)
        return values, valid, {"v0": float(velocity[right]),
                               "v1": float(velocity[right]), "weight": 0.0}
    require(0 < right < velocity.size, "internal RVTJ bracketing failure")
    left = right - 1
    v0, v1 = float(velocity[left]), float(velocity[right])
    weight = (target_velocity - v0) / (v1 - v0)
    a = native_j[:, order[left]]
    b = native_j[:, order[right]]
    both_positive = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    both_zero = (a == 0.0) & (b == 0.0)
    valid = both_positive | both_zero
    values = np.zeros(native_j.shape[0], dtype=np.float64)
    values[both_positive] = np.exp(
        (1.0 - weight) * np.log(a[both_positive]) + weight * np.log(b[both_positive])
    )
    # A zero/nonzero or negative bracket is deliberately not linearly filled.
    return values, valid, {"v0": v0, "v1": v1, "weight": float(weight)}


def rebin_j_shell(native_nu: np.ndarray, values: np.ndarray, sample_valid: np.ndarray,
                  target_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrate piecewise-linear J exactly between native sample frequencies."""
    mid = np.sqrt(target_edges[:-1] * target_edges[1:])
    index = np.searchsorted(native_nu, mid, side="right") - 1
    in_range = (target_edges[:-1] >= native_nu[0]) & (
        target_edges[1:] <= native_nu[-1]
    )
    index = np.clip(index, 0, native_nu.size - 2)
    x0, x1 = native_nu[index], native_nu[index + 1]
    fraction0 = (target_edges[:-1] - x0) / (x1 - x0)
    fraction1 = (target_edges[1:] - x0) / (x1 - x0)
    y0 = values[index] + fraction0 * (values[index + 1] - values[index])
    y1 = values[index] + fraction1 * (values[index + 1] - values[index])
    segment_valid = sample_valid[index] & sample_valid[index + 1]
    valid = in_range & segment_valid
    average = 0.5 * (y0 + y1)
    average[~valid] = 0.0
    tiny_negative = valid & (average < 0.0) & (
        np.abs(average) <= 16.0 * np.finfo(np.float64).eps * np.maximum(np.abs(y0), np.abs(y1))
    )
    average[tiny_negative] = 0.0
    require(np.all(average[valid] >= 0.0), "conservative J integration produced negative J")
    state = np.full(average.size, UNSAMPLED, dtype=np.uint8)
    state[valid & (y0 == 0.0) & (y1 == 0.0)] = EXACT_ZERO
    state[valid & ~((y0 == 0.0) & (y1 == 0.0))] = MEASURED
    return average, state


class SigmaDeck:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            raw = stream.read(32)
            require(len(raw) == 32, f"short CMFD header: {path}")
            magic, version, self.nlevels, self.nfreq, self.numin, self.numax = struct.unpack(
                "<IIiidd", raw
            )
            self.has = np.frombuffer(stream.read(self.nlevels), dtype=np.int8).copy()
        require(magic == CMFD_MAGIC and version == CMFD_VERSION and
                self.nlevels > 0 and self.nfreq > 0 and 0.0 < self.numin < self.numax,
                f"invalid CMFD header: magic={magic:#x} version={version}")
        padding = (8 - self.nlevels % 8) % 8
        self.offset = 32 + self.nlevels + padding
        expected = self.offset + 8 * self.nlevels * self.nfreq
        require(path.stat().st_size == expected,
                f"CMFD size {path.stat().st_size} != schema size {expected}")
        self.edges = np.geomspace(self.numin, self.numax, self.nfreq + 1)
        self.data = np.memmap(path, mode="r", dtype="<f8", offset=self.offset,
                              shape=(self.nlevels, self.nfreq))


def read_levels_and_thresholds(deck: Path, sigma: SigmaDeck) -> tuple[list[dict[str, str]], np.ndarray]:
    ionization: dict[tuple[int, int], float] = {}
    with (deck / "ionization_energies.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            ionization[(int(row["atomic_number"]), int(row["ion_number"]))] = float(
                row["ionization_energy_eV"]
            )
    with (deck / "levels.csv").open(newline="") as stream:
        levels = list(csv.DictReader(stream))
    require(len(levels) == sigma.nlevels,
            f"levels.csv rows {len(levels)} != CMFD nlevels {sigma.nlevels}")
    thresholds = np.full(len(levels), np.nan, dtype=np.float64)
    for index, row in enumerate(levels):
        key = (int(row["atomic_number"]), int(row["ion_number"]))
        delta_ev = ionization.get(key, float("nan")) - float(row["energy_eV"])
        if math.isfinite(delta_ev) and delta_ev > 0.0:
            thresholds[index] = delta_ev * EV_TO_ERG / H_CGS
    return levels, thresholds


def select_bf_levels(sigma: SigmaDeck, thresholds: np.ndarray, count: int,
                     union_lo: float, union_hi: float) -> np.ndarray:
    eligible = np.zeros(sigma.nlevels, dtype=bool)
    for start in range(0, sigma.nlevels, 512):
        stop = min(start + 512, sigma.nlevels)
        block = np.asarray(sigma.data[start:stop])
        threshold = thresholds[start:stop, None]
        positive_above = np.any(
            np.isfinite(block) & (block > 0.0) & (sigma.edges[1:][None, :] > threshold),
            axis=1,
        )
        eligible[start:stop] = (sigma.has[start:stop] == 1) & positive_above
    eligible &= np.isfinite(thresholds) & (thresholds < union_hi) & (
        thresholds < sigma.numax
    )
    indices = np.flatnonzero(eligible)
    require(indices.size >= count,
            f"only {indices.size} positive thresholded CMFD rows, need {count}")
    low = max(union_lo, float(np.min(thresholds[indices])))
    high = min(union_hi, float(np.max(thresholds[indices])))
    targets = np.geomspace(low, high, count)
    chosen: list[int] = []
    available = set(map(int, indices))
    logs = np.log(thresholds[indices])
    for target in targets:
        for position in np.argsort(np.abs(logs - math.log(target))):
            candidate = int(indices[position])
            if candidate in available:
                chosen.append(candidate)
                available.remove(candidate)
                break
    require(len(chosen) == count and len(set(chosen)) == count,
            "BF stratified selection did not produce unique rows")
    return np.asarray(sorted(chosen), dtype=np.int64)


LINE_FIELDS = (
    "atomic_number", "ion_number", "level_number_lower", "level_number_upper",
    "line_id", "nu",
)


def compact_line(row: dict[str, str], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "Z": int(row["atomic_number"]), "ion": int(row["ion_number"]),
        "lower": int(row["level_number_lower"]),
        "upper": int(row["level_number_upper"]),
        "catalog_id": row["line_id"], "nu": float(row["nu"]),
    }


def scan_line_count(path: Path, union_lo: float, union_hi: float) -> int:
    count = 0
    previous = math.inf
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(LINE_FIELDS) <= set(reader.fieldnames),
                f"line_list.csv missing required columns {LINE_FIELDS}")
        for row in reader:
            nu = float(row["nu"])
            require(math.isfinite(nu) and nu > 0.0, f"line row {count}: invalid nu")
            require(nu <= previous, f"line_list.csv not descending at row {count}")
            require(union_lo <= nu <= union_hi,
                    f"line row {count}: nu={nu} outside ledger line union")
            previous = nu
            count += 1
    require(count > 1, "line_list.csv has fewer than two rows")
    return count


def select_lines(path: Path, total: int, union_lo: float, union_hi: float,
                 log_count: int, rank_count: int) -> list[dict[str, Any]]:
    rank_targets = set(map(int, np.rint(np.linspace(0, total - 1, rank_count))))
    log_targets = list(np.geomspace(union_hi, union_lo, log_count))
    log_position = 0
    chosen: dict[int, dict[str, Any]] = {}
    previous: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    with path.open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            current = compact_line(row, index)
            if index in rank_targets:
                chosen[index] = current
            while log_position < len(log_targets) and current["nu"] <= log_targets[log_position]:
                options = [current] if previous is None else [previous, current]
                target_log = math.log(log_targets[log_position])
                best = min(options, key=lambda item: (
                    abs(math.log(item["nu"]) - target_log), item["index"]
                ))
                chosen[best["index"]] = best
                log_position += 1
            previous = current
            last = current
    require(last is not None, "empty line list during selection")
    while log_position < len(log_targets):
        chosen[last["index"]] = last
        log_position += 1
    selected = [chosen[index] for index in sorted(chosen)]
    require(len(selected) >= max(log_count, rank_count),
            f"line strata collapsed to only {len(selected)} unique lines")
    return selected


def build_bf_kernel(sigma: SigmaDeck, levels: list[dict[str, str]],
                    thresholds: np.ndarray, selected: np.ndarray,
                    edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    shell_ids: list[int] = []
    identifiers: list[str] = []
    lo, hi = edges[:-1], edges[1:]
    midpoint = np.sqrt(lo * hi)
    source_index = np.searchsorted(sigma.edges, midpoint, side="right") - 1
    source_valid = (source_index >= 0) & (source_index < sigma.nfreq) & (
        lo >= sigma.numin
    ) & (hi <= sigma.numax)
    source_index = np.clip(source_index, 0, sigma.nfreq - 1)
    for global_level in selected:
        gl = int(global_level)
        row = levels[gl]
        threshold = float(thresholds[gl])
        source_row = np.asarray(sigma.data[gl])
        require(np.all(np.isfinite(source_row)) and np.all(source_row >= 0.0),
                f"CMFD selected row {gl} contains a nonfinite/negative cross section")
        cross_section = source_row[source_index]
        lower = np.maximum(lo, threshold)
        integral = np.zeros(lo.size, dtype=np.float64)
        active = source_valid & (hi > threshold) & np.isfinite(cross_section) & (
            cross_section > 0.0
        )
        integral[active] = (
            FOUR_PI / H_CGS * cross_section[active] *
            np.log(hi[active] / lower[active])
        )
        kernel = integral / (hi - lo)
        require(np.any(kernel > 0.0), f"thresholded BF row {gl} has no support")
        for shell in PROFILE_SHELLS:
            rows.append(kernel)
            shell_ids.append(shell)
            identifiers.append(
                f"bf:s{shell}:Z{row['atomic_number']}:i{row['ion_number']}:"
                f"l{row['level_number']}:g{gl}"
            )
    return np.asarray(rows), np.asarray(shell_ids, dtype=np.int64), np.asarray(identifiers)


def build_line_profiles(lines: list[dict[str, Any]], edges: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.zeros((len(lines) * len(PROFILE_SHELLS), edges.size - 1), dtype=np.float64)
    shells: list[int] = []
    identifiers: list[str] = []
    out_row = 0
    for line in lines:
        index = int(np.searchsorted(edges, line["nu"], side="right") - 1)
        index = min(max(index, 0), edges.size - 2)
        require(edges[index] <= line["nu"] <= edges[index + 1],
                f"line {line['index']} not bracketed by fine grid")
        for shell in PROFILE_SHELLS:
            rows[out_row, index] = 1.0 / (edges[index + 1] - edges[index])
            shells.append(shell)
            identifiers.append(
                f"line:s{shell}:Z{line['Z']}:i{line['ion']}:"
                f"l{line['lower']}:u{line['upper']}:id{line['catalog_id']}:"
                f"row{line['index']}"
            )
            out_row += 1
    return rows, np.asarray(shells, dtype=np.int64), np.asarray(identifiers)


def build_edges(union_lo: float, union_hi: float, edd_nu: np.ndarray,
                chieta_edges: np.ndarray, sigma_edges: np.ndarray,
                selected_thresholds: np.ndarray) -> np.ndarray:
    pieces = [
        np.asarray([union_lo, union_hi]),
        edd_nu[(edd_nu > union_lo) & (edd_nu < union_hi)],
        chieta_edges[(chieta_edges > union_lo) & (chieta_edges < union_hi)],
        sigma_edges[(sigma_edges > union_lo) & (sigma_edges < union_hi)],
        selected_thresholds[(selected_thresholds > union_lo) &
                            (selected_thresholds < union_hi)],
    ]
    edges = np.unique(np.concatenate(pieces).astype(np.float64))
    require(edges[0] == union_lo and edges[-1] == union_hi and
            edges.size >= 16_001 and np.all(np.diff(edges) > 0.0),
            "merged fine grid does not cover the ledger union with >=16000 bins")
    return edges


def build_manifest(template_path: Path, ledger_path: Path, output: Path,
                   input_hashes: dict[str, str], ancillary: list[dict[str, Any]],
                   npz_hash: str, arrays: dict[str, np.ndarray],
                   bf_selected: int, line_selected: int) -> dict[str, Any]:
    manifest = load_json(template_path)
    require(manifest.get("schema") == SCHEMA, "input template schema mismatch")
    ledger_hash = sha256_file(ledger_path)
    require(manifest.get("consumer_union_ledger", {}).get("sha256") == ledger_hash,
            "template frequency-ledger hash is stale")
    manifest["consumer_union_ledger"] = {
        "path": str(ledger_path.relative_to(ROOT)), "sha256": ledger_hash,
    }
    manifest["fine_dump"] = {"path": str(output), "sha256": npz_hash}
    provenance = manifest["provenance"]
    provenance["new_gpu_run"] = False
    provenance["existing_dump_ids"] = list(input_hashes)
    provenance["existing_dump_sha256"] = input_hashes
    provenance["packing_method"] = (
        "offline conservative bin-average export; EDDFACTOR piecewise-linear "
        "overlap integral; CHIETA/CMFD piecewise-constant overlap integral; no point samples"
    )
    provenance["notes"] = (
        "Built only from the two existing dumps plus hash-bound RVTJ/INFO/deck metadata. "
        "CHIETA is zero-padded outside its captured frequency support because npz_contract "
        "has no chi/eta validity array; j_state never uses that convention."
    )
    provenance["ancillary_inputs"] = ancillary
    provenance["builder"] = {
        "path": "scripts/a2_02_prepare_fine_dump.py",
        "sha256": sha256_file(Path(__file__).resolve()),
        "fine_bins": int(arrays["nu_edges_hz"].size - 1),
        "shells": list(map(int, arrays["shell_id"])),
        "bf_threshold_strata": bf_selected,
        "line_unique_stratified_sample": line_selected,
        "profile_shells": list(PROFILE_SHELLS),
        "j_state_counts": {
            "MEASURED": int(np.count_nonzero(arrays["j_state"] == MEASURED)),
            "EXACT_ZERO": int(np.count_nonzero(arrays["j_state"] == EXACT_ZERO)),
            "UNSAMPLED": int(np.count_nonzero(arrays["j_state"] == UNSAMPLED)),
            "OUT_OF_RANGE": int(np.count_nonzero(arrays["j_state"] == OUT_OF_RANGE)),
        },
    }
    return manifest


def self_test() -> None:
    source_edges = np.geomspace(1.0, 16.0, 9)
    centers_desc = np.sqrt(source_edges[:-1] * source_edges[1:])[::-1]
    widths_desc = np.diff(source_edges)[::-1]
    rebuilt, _ = source_edges_from_centers_widths(centers_desc, widths_desc)
    require(np.max(np.abs(rebuilt / source_edges - 1.0)) < 1.0e-13,
            "self-test CHIETA edge reconstruction")
    target = np.unique(np.concatenate([source_edges, np.geomspace(1.0, 16.0, 33)]))
    values = np.arange(1.0, 9.0)[None, :]
    rebinned = piecewise_constant_rebin(values, source_edges, target)
    require(abs(np.sum(rebinned * np.diff(target)) -
                np.sum(values * np.diff(source_edges))) < 1.0e-12,
            "self-test conservative rebin")
    native_nu = np.asarray([1.0, 2.0, 4.0, 8.0])
    j, state = rebin_j_shell(native_nu, np.asarray([0.0, 0.0, 2.0, 4.0]),
                             np.ones(4, dtype=bool),
                             np.asarray([0.5, 1.0, 2.0, 4.0, 8.0, 16.0]))
    require(list(state) == [UNSAMPLED, EXACT_ZERO, MEASURED, MEASURED, UNSAMPLED] and
            j[1] == 0.0, "self-test J state distinction")
    print("A2_02_PREPARE_SELFTEST PASS conservative=1 states=4 point_sample=0")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edd", type=Path, default=DEFAULT_EDD)
    parser.add_argument("--rvtj", type=Path,
                        help="default: RVTJ beside --edd")
    parser.add_argument("--chieta", type=Path, default=DEFAULT_CHIETA)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bf-strata", type=int, default=24)
    parser.add_argument("--line-log-strata", type=int, default=16)
    parser.add_argument("--line-rank-strata", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    edd_path = resolve_repo(args.edd)
    rvtj_path = resolve_repo(args.rvtj) if args.rvtj else edd_path.parent / "RVTJ"
    chieta_path = resolve_repo(args.chieta)
    deck = resolve_repo(args.deck)
    ledger_path = resolve_repo(args.ledger)
    template_path = resolve_repo(args.template)
    output = resolve_repo(args.output)
    manifest_path = resolve_repo(args.manifest)
    require(args.bf_strata > 0 and args.line_log_strata > 1 and
            args.line_rank_strata > 1, "sample stratum counts must be positive")
    require(args.force or (not output.exists() and not manifest_path.exists()),
            "output or manifest already exists; use --force for an atomic replacement")
    for path in (edd_path, Path(str(edd_path) + "_INFO"), rvtj_path, chieta_path,
                 Path(str(chieta_path) + ".manifest.json"), ledger_path,
                 template_path, deck / "geometry.csv", deck / "levels.csv",
                 deck / "ionization_energies.csv", deck / "line_list.csv",
                 deck / "cmfgen_sigma_bf.bin"):
        require(path.is_file(), f"required input absent: {path}")

    ledger = load_json(ledger_path)
    require(ledger.get("schema") == "lumina-a2-02-frequency-union-v1",
            "frequency ledger schema mismatch")
    union_lo = float(ledger["union"]["nu_min_hz"])
    union_hi = float(ledger["union"]["nu_max_hz"])

    print("[1/7] validating LCMFCE01 capture and geometry", flush=True)
    chieta = load_chieta(chieta_path)
    shell_velocity = load_geometry(deck / "geometry.csv", chieta)

    print("[2/7] reading EDDFACTOR with chain-replay parser", flush=True)
    edd = load_eddfactor(edd_path)
    native_velocity = parse_rvtj_block(rvtj_path, "Velocity (km/s)", edd["nd"])
    in_rvtj = (shell_velocity >= np.min(native_velocity)) & (
        shell_velocity <= np.max(native_velocity)
    )
    require(np.array_equal(np.flatnonzero(in_rvtj), np.arange(44)),
            f"RVTJ mapping valid shells are {np.flatnonzero(in_rvtj).tolist()}, expected s0..s43")

    print("[3/7] validating CMFD thresholds and selecting BF strata", flush=True)
    sigma = SigmaDeck(deck / "cmfgen_sigma_bf.bin")
    levels, thresholds = read_levels_and_thresholds(deck, sigma)
    selected_bf = select_bf_levels(sigma, thresholds, args.bf_strata, union_lo, union_hi)

    print("[4/7] scanning line-list strata (two streaming passes)", flush=True)
    line_path = deck / "line_list.csv"
    line_total = scan_line_count(line_path, union_lo,
                                 float(ledger["consumers"][1]["nu_max_hz"]))
    selected_lines = select_lines(
        line_path, line_total, union_lo,
        float(ledger["consumers"][1]["nu_max_hz"]),
        args.line_log_strata, args.line_rank_strata,
    )

    edges = build_edges(union_lo, union_hi, edd["nu"], chieta["edges"], sigma.edges,
                        thresholds[selected_bf])
    print(f"[5/7] packing shell fields on {edges.size - 1} conservative bins", flush=True)
    chi = piecewise_constant_rebin(chieta["chi"], chieta["edges"], edges)
    eta = piecewise_constant_rebin(chieta["eta"], chieta["edges"], edges)
    j = np.zeros((chieta["nr"], edges.size - 1), dtype=np.float64)
    state = np.full(j.shape, OUT_OF_RANGE, dtype=np.uint8)
    brackets: list[dict[str, Any]] = []
    for shell, velocity in enumerate(shell_velocity):
        mapped, sample_valid, bracket = velocity_map(
            edd["j_depth"], native_velocity, float(velocity)
        )
        bracket = {"shell": shell, "velocity_km_s": float(velocity), **bracket}
        brackets.append(bracket)
        if bracket.get("status") == "OUT_OF_RANGE":
            continue
        j[shell], state[shell] = rebin_j_shell(edd["nu"], mapped, sample_valid, edges)
    require(np.all(np.isin(state, [MEASURED, EXACT_ZERO, UNSAMPLED, OUT_OF_RANGE])),
            "internal unknown j_state")
    require(np.all(state[:44] != OUT_OF_RANGE) and np.all(state[44:] == OUT_OF_RANGE),
            "s0..s43/s44..s49 RVTJ state boundary was not preserved")

    print("[6/7] packing thresholded BF kernels and stratified line profiles", flush=True)
    bf_kernel, bf_shell, bf_id = build_bf_kernel(
        sigma, levels, thresholds, selected_bf, edges
    )
    line_profile, line_shell, line_id = build_line_profiles(selected_lines, edges)
    arrays = {
        "nu_edges_hz": edges.astype(np.float64, copy=False),
        "shell_id": np.arange(chieta["nr"], dtype=np.int64),
        "j_nu": j, "j_state": state, "chi_nu": chi, "eta_nu": eta,
        "bf_kernel": bf_kernel, "bf_shell_id": bf_shell, "bf_id": bf_id,
        "line_profile": line_profile, "line_shell_id": line_shell,
        "line_id": line_id,
    }
    require(set(arrays) == {
        "nu_edges_hz", "shell_id", "j_nu", "j_state", "chi_nu", "eta_nu",
        "bf_kernel", "bf_shell_id", "bf_id", "line_profile", "line_shell_id", "line_id",
    }, "NPZ key set is not exactly the 12-array contract")
    require(np.allclose(np.sum(line_profile * np.diff(edges), axis=1), 1.0,
                        rtol=0.0, atol=2.0e-15),
            "line profile normalization failed")

    print("[7/7] atomically writing NPZ, hashes, and input manifest", flush=True)
    atomic_npz(output, arrays)
    npz_hash = sha256_file(output)
    dump_hashes = {
        str(edd_path): sha256_file(edd_path),
        str(chieta_path): sha256_file(chieta_path),
    }
    ancillary_paths = [
        Path(str(edd_path) + "_INFO"), rvtj_path,
        Path(str(chieta_path) + ".manifest.json"),
        deck / "geometry.csv", deck / "levels.csv", deck / "ionization_energies.csv",
        deck / "line_list.csv", deck / "cmfgen_sigma_bf.bin",
    ]
    ancillary = [{"path": str(path), "sha256": sha256_file(path)}
                 for path in ancillary_paths]
    manifest = build_manifest(
        template_path, ledger_path, output, dump_hashes, ancillary, npz_hash,
        arrays, len(selected_bf), len(selected_lines),
    )
    manifest["provenance"]["builder"]["rvtj_shell_brackets"] = brackets
    manifest["provenance"]["builder"]["line_list_rows"] = line_total
    atomic_json(manifest_path, manifest)
    print(
        f"A2_02_PREPARE PASS rc=0 bins={edges.size - 1} shells=50 "
        f"bf_rows={bf_kernel.shape[0]} line_rows={line_profile.shape[0]} "
        f"npz={output} npz_sha256={npz_hash} manifest={manifest_path}",
        flush=True,
    )


def main() -> int:
    args = arguments()
    try:
        if args.self_test:
            self_test()
            return 0
        run(args)
        return 0
    except (BuildError, OSError, ValueError, KeyError, struct.error) as exc:
        print(f"A2_02_PREPARE_FAIL {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
