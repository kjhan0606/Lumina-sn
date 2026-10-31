#!/usr/bin/env python3
"""완료된 계수에 CMFGEN 연속소광 하한과 SRCE_CHK를 읽기 전용 적용한다.

복사장 수송이나 물질 갱신을 하지 않는 계수공간 반사실 분석이다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
from numpy.lib import format as npformat


ARRAY_NAMES = (
    "bf_chi", "ff_chi", "bb_chi", "es_chi",
    "bf_eta", "ff_eta", "bb_eta",
)
C_CGS = 2.99792458e10
ETA_FLOOR = 1.0e-280


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def open_npy_member(archive: zipfile.ZipFile, name: str, expected_shape: tuple[int, ...]):
    stream = archive.open(name + ".npy")
    version = npformat.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = npformat.read_array_header_1_0(stream)
    elif version in ((2, 0), (3, 0)):
        shape, fortran_order, dtype = npformat.read_array_header_2_0(stream)
    else:
        stream.close()
        raise ValueError(f"지원하지 않는 NPY 버전: {name} {version}")
    if tuple(shape) != expected_shape or fortran_order or np.dtype(dtype) != np.dtype("<f8"):
        stream.close()
        raise ValueError(f"배열 신원 불일치: {name} {shape} {dtype}")
    return stream


def read_exact(stream, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise ValueError("NPY 배열이 예기치 않게 끝남")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def model_and_process_contract(report: dict) -> tuple[str, str]:
    model = report["process_selection"]["model"]
    model_path = Path(model["path"])
    actual_model_sha = sha256(model_path)
    if actual_model_sha != model["sha256"]:
        raise ValueError("계수 생산 MODEL 신원 불일치")
    model_text = model_path.read_text(errors="replace")
    if not re.search(r"^\s*SRCE_CHK\s+\[NEG_OPAC_OPT\]", model_text, re.MULTILINE):
        raise ValueError("생산 MODEL의 NEG_OPAC_OPT=SRCE_CHK를 확인하지 못함")

    inactive = report["inactive_processes"]
    if (inactive.get("xrays") != "MODEL F" or
            inactive.get("rayleigh") != "MODEL F" or
            inactive.get("additional_opacity") != "MODEL F/default F" or
            inactive.get("two_photon", {}).get("active_in_model") != 0):
        raise ValueError("현재 분석의 continuum process 축약이 성립하지 않음")
    if report["free_free"].get("policy") != "NATIVE_UPPER_IONS":
        raise ValueError("자유-자유 계수 정책이 봉인된 native 선택과 다름")
    return actual_model_sha, str(model_path.name)


def analyze(source_dir: Path, output: Path | None) -> dict:
    source_dir = source_dir.resolve(strict=True)
    manifest_path = source_dir / "sources.json"
    archive_path = source_dir / "sources.npz"
    report = json.loads(manifest_path.read_text())
    model_sha, model_name = model_and_process_contract(report)

    cells = int(report["cells"])
    frequencies = int(report["frequencies"])
    if cells != 90 or frequencies != 1_115_020:
        raise ValueError(f"예상하지 않은 generation15 격자: {cells}x{frequencies}")
    expected_npz_sha = report["output_sha256"]["sources.npz"]
    actual_npz_sha = sha256(archive_path)
    if actual_npz_sha != expected_npz_sha:
        raise ValueError("sources.npz 생산 해시 불일치")

    shape = (cells, frequencies)
    byte_count = frequencies * np.dtype("<f8").itemsize
    totals = {
        "raw_negative_pairs": 0,
        "continuum_floor_pairs": 0,
        "srce_source_branch_pairs": 0,
        "srce_scattering_branch_pairs": 0,
        "native_total_changed_pairs": 0,
    }
    band_totals = {key: 0 for key in totals}
    raw_min = math.inf
    raw_max = -math.inf
    max_abs_delta = -1.0
    max_abs_delta_record = None
    max_delta_over_es = 0.0

    with zipfile.ZipFile(archive_path) as archive:
        required = {name + ".npy" for name in (*ARRAY_NAMES, "nu_Hz")}
        if not required.issubset(set(archive.namelist())):
            raise ValueError("필수 계수 배열 누락")
        nu_stream = open_npy_member(archive, "nu_Hz", (frequencies,))
        nu = np.frombuffer(read_exact(nu_stream, byte_count), dtype="<f8").copy()
        if not np.isfinite(nu).all() or np.any(nu <= 0.0):
            raise ValueError("주파수 격자가 유효하지 않음")
        nu_stream.close()
        band = ((C_CGS * 1.0e8 / 750.0) <= nu) & (nu <= (C_CGS * 1.0e8 / 600.0))

        streams = {name: open_npy_member(archive, name, shape) for name in ARRAY_NAMES}
        try:
            for cell in range(cells):
                row = {
                    name: np.frombuffer(read_exact(streams[name], byte_count), dtype="<f8")
                    for name in ARRAY_NAMES
                }
                if any(not np.isfinite(values).all() for values in row.values()):
                    raise ValueError(f"비유한 계수: cell={cell}")

                # COMP_OPAC: chi_noscat >= 1e-3 ESEC, chi_cont=chi_noscat+chi_scat.
                chi_noscat_raw = row["bf_chi"] + row["ff_chi"]
                chi_scat = row["es_chi"]
                chi_noscat = np.maximum(chi_noscat_raw, 1.0e-3 * chi_scat)
                chi_cont = chi_noscat + chi_scat
                chi_raw = (chi_scat + chi_noscat_raw) + row["bb_chi"]
                chi_pre_srce = chi_cont + row["bb_chi"]

                # ETA_CONT is continuum emission; ETA also includes bound-bound emission.
                eta_cont = np.maximum(row["bf_eta"] + row["ff_eta"], ETA_FLOOR)
                eta_total = eta_cont + row["bb_eta"]
                if np.any(eta_total < 0.0):
                    raise ValueError(f"음수 총 emissivity: cell={cell}")
                source_limit = 0.1 * eta_total * chi_noscat / eta_cont
                source_branch = (chi_pre_srce < chi_cont) & (chi_pre_srce < source_limit)
                scatter_branch = (~source_branch) & (chi_pre_srce < 0.1 * chi_scat)
                chi_native = np.where(
                    source_branch,
                    source_limit,
                    np.where(scatter_branch, 0.1 * chi_scat, chi_pre_srce),
                )
                delta = chi_native - chi_raw
                changed = delta != 0.0
                floor = chi_noscat_raw < 1.0e-3 * chi_scat
                negative = chi_raw < 0.0

                current = {
                    "raw_negative_pairs": negative,
                    "continuum_floor_pairs": floor,
                    "srce_source_branch_pairs": source_branch,
                    "srce_scattering_branch_pairs": scatter_branch,
                    "native_total_changed_pairs": changed,
                }
                for key, mask in current.items():
                    totals[key] += int(np.count_nonzero(mask))
                    band_totals[key] += int(np.count_nonzero(mask & band))

                raw_min = min(raw_min, float(np.min(chi_raw)))
                raw_max = max(raw_max, float(np.max(chi_raw)))
                abs_delta = np.abs(delta)
                local_max_index = int(np.argmax(abs_delta))
                local_max = float(abs_delta[local_max_index])
                if local_max > max_abs_delta:
                    max_abs_delta = local_max
                    max_abs_delta_record = {
                        "cell_0based": cell,
                        "frequency_index_0based": local_max_index,
                        "wavelength_A": float(C_CGS * 1.0e8 / nu[local_max_index]),
                        "chi_raw": float(chi_raw[local_max_index]),
                        "chi_native_counterfactual": float(chi_native[local_max_index]),
                        "continuum_floor_applied": bool(floor[local_max_index]),
                        "srce_source_branch": bool(source_branch[local_max_index]),
                        "srce_scattering_branch": bool(scatter_branch[local_max_index]),
                    }
                ratio = abs_delta / np.maximum(np.abs(chi_scat), 1.0e-300)
                max_delta_over_es = max(max_delta_over_es, float(np.max(ratio)))
        finally:
            for stream in streams.values():
                stream.close()

    expected_range = report["chi_total_min_max"]
    if totals["raw_negative_pairs"] != int(report["total_negative_samples"]):
        raise ValueError("재구성 raw chi 음수 표본이 생산 manifest와 다름")
    if not math.isclose(raw_min, float(expected_range[0]), rel_tol=2.0e-12, abs_tol=1.0e-300):
        raise ValueError("재구성 raw chi 최솟값이 생산 manifest와 다름")
    if not math.isclose(raw_max, float(expected_range[1]), rel_tol=2.0e-12, abs_tol=1.0e-300):
        raise ValueError("재구성 raw chi 최댓값이 생산 manifest와 다름")

    result = {
        "schema": "lumina-native-aggregate-opacity-policy-v1",
        "state": "COMPLETE_READ_ONLY_COEFFICIENT_COUNTERFACTUAL",
        "input_identity": {
            "sources_npz_sha256": actual_npz_sha,
            "producer_model_sha256": model_sha,
            "producer_model_name": model_name,
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "grid": {"cells": cells, "frequencies": frequencies},
        "native_policy": {
            "continuum_absorption_floor": "chi_noscat=max(bf_chi+ff_chi, 1e-3*chi_es)",
            "srce_source_branch": "chi<chi_cont and chi<0.1*eta*chi_noscat/eta_cont",
            "srce_scattering_branch": "else if chi<0.1*chi_es",
            "process_scope": "the producer manifest records X-rays, Rayleigh, additional opacity, and two-photon inactive",
        },
        "counts": totals,
        "counts_600_750_A": band_totals,
        "raw_chi_range": {"min": raw_min, "max": raw_max},
        "maximum_absolute_chi_change": max_abs_delta,
        "maximum_absolute_change_over_electron_scattering": max_delta_over_es,
        "maximum_change_location": max_abs_delta_record,
        "opacity_modified_on_disk": False,
        "transport_runs": 0,
        "material_or_rate_updates": 0,
        "does_not_test": [
            "transported J or its effect on Si/S populations",
            "NEG_OPAC_FAC feedback into CMFGEN's bound-bound rate linearisation",
            "full CMFGEN equation or solution equivalence",
        ],
        "final_goal": "NOT_MET",
    }
    if output is not None:
        output = output.resolve()
        with output.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sources_dir, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
