#!/usr/bin/env python3
"""완료 계수에 현 CMFGEN source worktree의 미커밋 연속소광 patch를 반사실 적용한다.

이것은 stock/committed CMFGEN 정책 판독도, 복사장 수송도, 물질 갱신도 아니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
from numpy.lib import format as npformat


ARRAY_NAMES = (
    "bf_chi", "ff_chi", "bb_chi", "es_chi",
    "bf_eta", "ff_eta", "bb_eta",
)
C_CGS = 2.99792458e10
ETA_FLOOR = 1.0e-280
EXPECTED_CMFGEN_COMMIT = "dd1660af039e10f8f8c61d892c4e226e1cd6438c"
EXPECTED_COMMITTED_COMPOPAC_SHA256 = "46aa68419d3bd6ef359d43ac95e8de6ee19ab0e33f52f1a61dca17431d21ea41"
EXPECTED_WORKTREE_COMPOPAC_SHA256 = "c065c49f1e156f11d04385b57b8869bd1d366eaaa1887e54f7753222d63bb99a"
EXPECTED_CONTROL_OPTIONS_SHA256 = "1f065997994a61681abc8e34426aa4f4cb95f5aaaaa8ec4518f27a9bbedc8e88"
EXPECTED_CMFGEN_SUB_SHA256 = "092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_cmfgen_source(source_tree: Path) -> dict:
    source_tree = source_tree.resolve(strict=True)
    commit = subprocess.run(
        ["git", "-C", str(source_tree), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != EXPECTED_CMFGEN_COMMIT:
        raise ValueError("CMFGEN committed source revision changed")

    comp_opac = source_tree / "new_main/mod_subs/comp_opac.f"
    rd_control = source_tree / "new_main/mod_subs/rd_control_variables.f"
    cmfgen_sub = source_tree / "new_main/cmfgen_sub.f"
    committed_comp_opac = subprocess.run(
        ["git", "-C", str(source_tree), "show", "HEAD:new_main/mod_subs/comp_opac.f"],
        check=True, capture_output=True,
    ).stdout
    hashes = {
        "comp_opac_committed_sha256": sha256_bytes(committed_comp_opac),
        "comp_opac_worktree_sha256": sha256(comp_opac),
        "rd_control_variables_sha256": sha256(rd_control),
        "cmfgen_sub_sha256": sha256(cmfgen_sub),
    }
    expected = {
        "comp_opac_committed_sha256": EXPECTED_COMMITTED_COMPOPAC_SHA256,
        "comp_opac_worktree_sha256": EXPECTED_WORKTREE_COMPOPAC_SHA256,
        "rd_control_variables_sha256": EXPECTED_CONTROL_OPTIONS_SHA256,
        "cmfgen_sub_sha256": EXPECTED_CMFGEN_SUB_SHA256,
    }
    if hashes != expected:
        raise ValueError("CMFGEN source files differ from the recorded stock/worktree pair")
    return {
        "git_commit": commit,
        **hashes,
        "comp_opac_worktree_differs_from_committed": True,
        "binary_build_identity": "NOT_VERIFIED",
    }


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


def policy_row_summary(row: dict, band: np.ndarray, nu: np.ndarray, cell: int,
                       apply_continuum_floor: bool) -> dict:
    chi_noscat_raw = row["bf_chi"] + row["ff_chi"]
    chi_scat = row["es_chi"]
    patch_floor = chi_noscat_raw < 1.0e-3 * chi_scat
    if apply_continuum_floor:
        chi_noscat = np.maximum(chi_noscat_raw, 1.0e-3 * chi_scat)
    else:
        chi_noscat = chi_noscat_raw
    chi_cont = chi_noscat + chi_scat
    chi_raw = (chi_scat + chi_noscat_raw) + row["bb_chi"]
    chi_pre_srce = chi_cont + row["bb_chi"]

    eta_cont = np.maximum(row["bf_eta"] + row["ff_eta"], ETA_FLOOR)
    eta_total = eta_cont + row["bb_eta"]
    if np.any(eta_total < 0.0):
        raise ValueError(f"음수 총 emissivity: cell={cell}")
    source_limit = 0.1 * eta_total * chi_noscat / eta_cont
    source_branch = (chi_pre_srce < chi_cont) & (chi_pre_srce < source_limit)
    scatter_branch = (~source_branch) & (chi_pre_srce < 0.1 * chi_scat)
    chi_adjusted = np.where(
        source_branch,
        source_limit,
        np.where(scatter_branch, 0.1 * chi_scat, chi_pre_srce),
    )
    delta = chi_adjusted - chi_raw
    changed = delta != 0.0
    applied_floor = patch_floor if apply_continuum_floor else np.zeros_like(patch_floor)
    masks = {
        "raw_negative_pairs": chi_raw < 0.0,
        "below_worktree_floor_threshold_pairs": patch_floor,
        "continuum_floor_applied_pairs": applied_floor,
        "srce_source_branch_pairs": source_branch,
        "srce_scattering_branch_pairs": scatter_branch,
        "opacity_changed_from_raw_pairs": changed,
    }
    counts = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    band_counts = {name: int(np.count_nonzero(mask & band)) for name, mask in masks.items()}
    abs_delta = np.abs(delta)
    max_index = int(np.argmax(abs_delta))
    return {
        "counts": counts,
        "counts_600_750_A": band_counts,
        "maximum_absolute_chi_change": float(abs_delta[max_index]),
        "maximum_absolute_change_over_electron_scattering": float(
            np.max(abs_delta / np.maximum(np.abs(chi_scat), 1.0e-300))
        ),
        "maximum_change_location": {
            "cell_0based": cell,
            "frequency_index_0based": max_index,
            "wavelength_A": float(C_CGS * 1.0e8 / nu[max_index]),
            "chi_raw": float(chi_raw[max_index]),
            "chi_counterfactual": float(chi_adjusted[max_index]),
            "continuum_floor_applied": bool(applied_floor[max_index]),
            "srce_source_branch": bool(source_branch[max_index]),
            "srce_scattering_branch": bool(scatter_branch[max_index]),
        },
    }


def analyze(source_dir: Path, cmfgen_source_tree: Path, output: Path | None) -> dict:
    source_dir = source_dir.resolve(strict=True)
    source_context = verify_cmfgen_source(cmfgen_source_tree)
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
    count_names = (
        "raw_negative_pairs",
        "below_worktree_floor_threshold_pairs",
        "continuum_floor_applied_pairs",
        "srce_source_branch_pairs",
        "srce_scattering_branch_pairs",
        "opacity_changed_from_raw_pairs",
    )
    variants = {
        "committed_stock": {
            "apply_continuum_floor": False,
            "counts": dict.fromkeys(count_names, 0),
            "counts_600_750_A": dict.fromkeys(count_names, 0),
            "maximum_absolute_chi_change": -1.0,
            "maximum_absolute_change_over_electron_scattering": 0.0,
            "maximum_change_location": None,
        },
        "uncommitted_worktree_floor_patch": {
            "apply_continuum_floor": True,
            "counts": dict.fromkeys(count_names, 0),
            "counts_600_750_A": dict.fromkeys(count_names, 0),
            "maximum_absolute_chi_change": -1.0,
            "maximum_absolute_change_over_electron_scattering": 0.0,
            "maximum_change_location": None,
        },
    }
    raw_min = math.inf
    raw_max = -math.inf

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

                chi_raw = (row["es_chi"] + (row["bf_chi"] + row["ff_chi"])) + row["bb_chi"]
                raw_min = min(raw_min, float(np.min(chi_raw)))
                raw_max = max(raw_max, float(np.max(chi_raw)))
                for variant in variants.values():
                    measured = policy_row_summary(
                        row, band, nu, cell, variant["apply_continuum_floor"]
                    )
                    for name in count_names:
                        variant["counts"][name] += measured["counts"][name]
                        variant["counts_600_750_A"][name] += measured["counts_600_750_A"][name]
                    variant["maximum_absolute_change_over_electron_scattering"] = max(
                        variant["maximum_absolute_change_over_electron_scattering"],
                        measured["maximum_absolute_change_over_electron_scattering"],
                    )
                    if measured["maximum_absolute_chi_change"] > variant["maximum_absolute_chi_change"]:
                        variant["maximum_absolute_chi_change"] = measured["maximum_absolute_chi_change"]
                        variant["maximum_change_location"] = measured["maximum_change_location"]
        finally:
            for stream in streams.values():
                stream.close()

    expected_range = report["chi_total_min_max"]
    for variant in variants.values():
        if variant["counts"]["raw_negative_pairs"] != int(report["total_negative_samples"]):
            raise ValueError("재구성 raw chi 음수 표본이 생산 manifest와 다름")
    if not math.isclose(raw_min, float(expected_range[0]), rel_tol=2.0e-12, abs_tol=1.0e-300):
        raise ValueError("재구성 raw chi 최솟값이 생산 manifest와 다름")
    if not math.isclose(raw_max, float(expected_range[1]), rel_tol=2.0e-12, abs_tol=1.0e-300):
        raise ValueError("재구성 raw chi 최댓값이 생산 manifest와 다름")

    result = {
        "schema": "lumina-cmfgen-opacity-policy-variants-counterfactual-v1",
        "state": "COMPLETE_READ_ONLY_POLICY_VARIANT_CENSUS",
        "input_identity": {
            "sources_npz_sha256": actual_npz_sha,
            "producer_model_sha256": model_sha,
            "producer_model_name": model_name,
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "cmfgen_source_context": source_context,
        "grid": {"cells": cells, "frequencies": frequencies},
        "policy_definitions": {
            "committed_stock": {
                "continuum_absorption": "raw bf_chi+ff_chi; committed comp_opac.f has no chi floor",
                "srce_source_branch": "chi<chi_cont and chi<0.1*eta*chi_noscat/eta_cont",
                "srce_scattering_branch": "else if chi<0.1*chi_es",
            },
            "uncommitted_worktree_floor_patch": {
                "continuum_absorption_floor": "chi_noscat=max(bf_chi+ff_chi, 1e-3*chi_es)",
                "srce_source_branch": "chi<chi_cont and chi<0.1*eta*chi_noscat/eta_cont",
                "srce_scattering_branch": "else if chi<0.1*chi_es",
            },
            "process_scope": "the producer manifest records X-rays, Rayleigh, additional opacity, and two-photon inactive",
        },
        "policy_variants": {
            name: {
                "counts": value["counts"],
                "counts_600_750_A": value["counts_600_750_A"],
                "maximum_absolute_chi_change": value["maximum_absolute_chi_change"],
                "maximum_absolute_change_over_electron_scattering": value[
                    "maximum_absolute_change_over_electron_scattering"
                ],
                "maximum_change_location": value["maximum_change_location"],
            }
            for name, value in variants.items()
        },
        "raw_chi_range": {"min": raw_min, "max": raw_max},
        "opacity_modified_on_disk": False,
        "transport_runs": 0,
        "material_or_rate_updates": 0,
        "does_not_test": [
            "source identity of the CMFGEN executable used for any existing reference run",
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
    parser.add_argument("--cmfgen-source-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        analyze(args.sources_dir, args.cmfgen_source_tree, args.output),
        ensure_ascii=False, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
