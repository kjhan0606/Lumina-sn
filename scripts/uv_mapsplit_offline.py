#!/usr/bin/env python3
"""Offline UV line-emission split by NLTE line-map reachability.

This consumer deliberately reuses ``uv_t2n9_offline.parse_linepop``.  It does
not run Lumina, a model, a GPU kernel, or transport.  The primary energy ledger
is constructed per recorded line and shell as

    eta_l = row.w * row.eps_l * row.S_l_used   (when eps_phys is on)
    eta_l = row.w * row.S_l_used               (when eps_phys is off)
    E_l   = eta_l * linepop.dnu[line.bin]

The eps_l factor reproduces the production assembly of
``src/lumina_cmfgen.c:794-802``, where ``eta_l = w * el * Sl`` under
``eps_phys`` and ``eta_l = w * Sl`` otherwise.  The gate is read from the
payload header rather than assumed.  The stored ``eps_l`` is the value
production actually used, that is after the ``eps_floor``/``eps_cap`` clamps of
``src/lumina_cmfgen.c:795-797``, so multiplying by it reproduces production
instead of recomputing it.  Both the eps-weighted and the eps-free ledgers are
reported, and their difference is the size of this correction.

No bin aggregate is apportioned back to lines.  The map predicate is exactly
bit 0 (CMF_LP_F_NLTE_ION), written when nlte_line_map[l] >= 0.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import uv_t2n9_offline as base  # noqa: E402


F_NLTE_ION = 1 << 0
DEFECTIVE_MAP_BIT = 1 << 31
EXPECTED_SHELLS = (0, 8, 16, 20, 45)
DISPOSITIONS = tuple(sorted(base.DISPOSITION.items()))
SCHEMA = "lumina-uv-mapsplit-v1"


class Arrays(NamedTuple):
    shells: np.ndarray
    shell_slot: np.ndarray
    flags: np.ndarray
    bins: np.ndarray
    wavelength_A: np.ndarray
    Z: np.ndarray
    ion_number: np.ndarray
    w: np.ndarray
    S_l_used: np.ndarray
    eps_l: np.ndarray
    eps_phys: int
    dnu: np.ndarray
    disposition: np.ndarray


def require(condition: bool, message: str) -> None:
    if not condition:
        raise base.OfflineError(message)


def roman(value: int) -> str:
    """Return a spectroscopic Roman stage for a positive integer."""
    require(0 < value < 4000, f"spectroscopic stage outside Roman domain: {value}")
    table = (
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    )
    result: list[str] = []
    remainder = value
    for number, token in table:
        count, remainder = divmod(remainder, number)
        result.extend([token] * count)
    return "".join(result)


def spectroscopic_stage(ion_number: int) -> str:
    # Project convention is zero-based: 0 -> neutral stage I.
    require(ion_number >= 0, f"negative raw ion_number: {ion_number}")
    return roman(ion_number + 1)


def band_mask(wavelength_A: np.ndarray, band_index: int) -> np.ndarray:
    name, lo, hi = base.BANDS[band_index]
    if name in ("B4", "BALL"):
        return (wavelength_A >= lo) & (wavelength_A <= hi)
    return (wavelength_A >= lo) & (wavelength_A < hi)


def sum64(values: np.ndarray, mask: np.ndarray) -> float:
    result = float(np.sum(values[mask], dtype=np.float64))
    require(math.isfinite(result) and result >= 0.0,
            "aggregate sum is negative or nonfinite")
    return result


def fraction(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def csv_scalar(value: Any) -> Any:
    return "UNDEFINED" if value is None else value


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    require(bool(rows), "refusing empty CSV payload")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_scalar(value) for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def arrays_from_linepop(linepop: base.LinePop) -> Arrays:
    rows = linepop.rows
    lines = linepop.lines[np.asarray(rows["line_slot"], dtype=np.int64)]
    bins = np.asarray(lines["bin"], dtype=np.int64)
    wavelength_A = np.asarray(lines["lambda_cm"], dtype=np.float64) * 1.0e8
    arrays = Arrays(
        shells=np.asarray(linepop.shells, dtype=np.int64),
        shell_slot=np.asarray(rows["shell_slot"], dtype=np.int64),
        flags=np.asarray(rows["flags"], dtype=np.uint32),
        bins=bins,
        wavelength_A=wavelength_A,
        Z=np.asarray(lines["Z"], dtype=np.int64),
        ion_number=np.asarray(lines["ion"], dtype=np.int64),
        w=np.asarray(rows["w"], dtype=np.float64),
        S_l_used=np.asarray(rows["S_l_used"], dtype=np.float64),
        eps_l=np.asarray(rows["eps_l"], dtype=np.float64),
        eps_phys=int(linepop.header["eps_phys"]),
        dnu=np.asarray(linepop.dnu, dtype=np.float64),
        disposition=np.asarray(linepop.disposition, dtype=np.uint8),
    )
    validate_arrays(arrays)
    return arrays


def validate_arrays(a: Arrays) -> None:
    n = a.shell_slot.size
    for name in ("flags", "bins", "wavelength_A", "Z", "ion_number", "w",
                 "S_l_used", "eps_l"):
        require(getattr(a, name).size == n, f"row array length mismatch: {name}")
    require(tuple(int(x) for x in a.shells) == EXPECTED_SHELLS,
            f"selected shell identity mismatch: {a.shells.tolist()}")
    require(a.disposition.ndim == 2 and a.disposition.shape[1] == a.dnu.size,
            "disposition/grid shape mismatch")
    require(np.all((a.shell_slot >= 0) & (a.shell_slot < a.shells.size)),
            "row shell slot outside selection")
    require(np.all((a.bins >= 0) & (a.bins < a.dnu.size)), "row bin outside grid")
    require(np.isfinite(a.wavelength_A).all() and
            np.all((a.wavelength_A >= 600.0) & (a.wavelength_A <= 3000.0)),
            "recorded line wavelength outside the closed UV selection")
    require(np.isfinite(a.w).all() and np.all(a.w >= 0.0), "invalid row w")
    require(np.isfinite(a.S_l_used).all() and np.all(a.S_l_used >= 0.0),
            "invalid row S_l_used")
    require(a.eps_phys in (0, 1), f"unknown eps_phys gate: {a.eps_phys}")
    if a.eps_phys:
        require(np.isfinite(a.eps_l).all() and np.all(a.eps_l > 0.0),
                "invalid row eps_l under eps_phys")
    require(np.isfinite(a.dnu).all() and np.all(a.dnu > 0.0), "invalid dnu")
    require(np.all(a.Z > 0), "nonpositive atomic number")
    require(np.all(a.ion_number >= 0), "negative raw ion_number")
    require(np.all(a.disposition <= 3), "unknown EPAY disposition")
    hits = np.zeros(n, dtype=np.int8)
    for band_index in range(5):
        mask = band_mask(a.wavelength_A, band_index)
        hits += mask.astype(np.int8)
    require(np.all(hits == 1), "B0..B4 do not partition every selected line exactly once")


def measures(a: Arrays) -> dict[str, np.ndarray]:
    eps = a.eps_l if a.eps_phys else np.ones_like(a.S_l_used)
    eta = a.w * eps * a.S_l_used
    eta_noeps = a.w * a.S_l_used
    energy = eta * a.dnu[a.bins]
    energy_noeps = eta_noeps * a.dnu[a.bins]
    chi_integral = a.w * a.dnu[a.bins]
    for name, value in (("eta_l", eta), ("line emission energy", energy),
                        ("eta_l without eps", eta_noeps),
                        ("line emission energy without eps", energy_noeps),
                        ("chi_line integral", chi_integral)):
        require(np.isfinite(value).all() and np.all(value >= 0.0),
                f"{name} contains a negative or nonfinite value")
    return {"eta": eta, "energy": energy, "chi": a.w,
            "chi_integral": chi_integral,
            "eta_noeps": eta_noeps, "energy_noeps": energy_noeps}


def metric_split(values: np.ndarray, selected: np.ndarray,
                 mapped: np.ndarray) -> tuple[float, float, float, float | None]:
    mapped_value = sum64(values, selected & mapped)
    unmapped_value = sum64(values, selected & ~mapped)
    total = mapped_value + unmapped_value
    return total, mapped_value, unmapped_value, fraction(unmapped_value, total)


def shell_band_rows(a: Arrays, m: dict[str, np.ndarray],
                    mapped: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for slot, shell in enumerate(a.shells):
        shell_rows = a.shell_slot == slot
        for band_index, (band, lo, hi) in enumerate(base.BANDS):
            selected = shell_rows & band_mask(a.wavelength_A, band_index)
            energy = metric_split(m["energy"], selected, mapped)
            eta = metric_split(m["eta"], selected, mapped)
            chi = metric_split(m["chi"], selected, mapped)
            chi_i = metric_split(m["chi_integral"], selected, mapped)
            noeps = metric_split(m["energy_noeps"], selected, mapped)
            output.append({
                "line_emission_energy_noeps_total": noeps[0],
                "line_emission_energy_noeps_unmapped": noeps[2],
                "line_emission_energy_noeps_unmapped_fraction": noeps[3],
                "shell": int(shell), "band": band,
                "lambda_lo_A": lo, "lambda_hi_A": hi,
                "rows": int(np.count_nonzero(selected)),
                "line_emission_energy_total": energy[0],
                "line_emission_energy_mapped": energy[1],
                "line_emission_energy_unmapped": energy[2],
                "line_emission_energy_unmapped_fraction": energy[3],
                "eta_line_sum_total": eta[0],
                "eta_line_sum_mapped": eta[1],
                "eta_line_sum_unmapped": eta[2],
                "eta_line_sum_unmapped_fraction": eta[3],
                "chi_line_sum_total": chi[0],
                "chi_line_sum_mapped": chi[1],
                "chi_line_sum_unmapped": chi[2],
                "chi_line_sum_unmapped_fraction": chi[3],
                "chi_line_dnu_integral_total": chi_i[0],
                "chi_line_dnu_integral_mapped": chi_i[1],
                "chi_line_dnu_integral_unmapped": chi_i[2],
                "chi_line_dnu_integral_unmapped_fraction": chi_i[3],
            })
    return output


def pooled_ball(a: Arrays, m: dict[str, np.ndarray],
                mapped: np.ndarray) -> dict[str, Any]:
    selected = band_mask(a.wavelength_A, 5)
    energy = metric_split(m["energy"], selected, mapped)
    chi = metric_split(m["chi"], selected, mapped)
    return {
        "scope": "SELECTED5_ONLY", "shells": list(EXPECTED_SHELLS), "band": "BALL",
        "line_emission_energy_total": energy[0],
        "line_emission_energy_mapped": energy[1],
        "line_emission_energy_unmapped": energy[2],
        "line_emission_energy_mapped_fraction": fraction(energy[1], energy[0]),
        "line_emission_energy_unmapped_fraction": energy[3],
        "chi_line_sum_total": chi[0], "chi_line_sum_mapped": chi[1],
        "chi_line_sum_unmapped": chi[2],
        "chi_line_sum_unmapped_fraction": chi[3],
    }


def ion_rank_rows(a: Arrays, energy: np.ndarray,
                  mapped: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes: list[tuple[str, int | None, np.ndarray]] = [
        (str(int(shell)), int(shell), a.shell_slot == slot)
        for slot, shell in enumerate(a.shells)
    ]
    scopes.append(("SELECTED5", None, np.ones(a.shell_slot.size, dtype=bool)))
    for scope, shell, shell_rows in scopes:
        for band_index, (band, lo, hi) in enumerate(base.BANDS):
            selected = shell_rows & band_mask(a.wavelength_A, band_index) & ~mapped
            indices = np.flatnonzero(selected)
            groups: dict[tuple[int, int], tuple[float, int]] = {}
            for index in indices:
                key = (int(a.Z[index]), int(a.ion_number[index]))
                old_energy, old_rows = groups.get(key, (0.0, 0))
                groups[key] = (old_energy + float(energy[index]), old_rows + 1)
            ranked = sorted(groups.items(), key=lambda item: (
                -item[1][0], item[0][0], item[0][1]))
            total = math.fsum(value[0] for _, value in ranked)
            for rank, ((z, ion), (ion_energy, count)) in enumerate(ranked, 1):
                output.append({
                    "scope": scope, "shell": "SELECTED5" if shell is None else shell,
                    "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                    "rank": rank, "Z": z,
                    "element": base.ELEMENT_SYMBOL.get(z, f"Z{z}"),
                    "ion_number_raw": ion,
                    "spectroscopic_stage": spectroscopic_stage(ion),
                    "unmapped_rows": count,
                    "unmapped_line_emission_energy": ion_energy,
                    "fraction_of_unmapped_line_emission_energy":
                        fraction(ion_energy, total),
                })
    require(bool(output), "no unmapped ion rank rows")
    return output


def crosstab_rows(a: Arrays, m: dict[str, np.ndarray],
                  mapped: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    row_shell = a.shells[a.shell_slot]
    row_disposition = a.disposition[row_shell, a.bins]
    for slot, shell in enumerate(a.shells):
        shell_rows = a.shell_slot == slot
        for band_index, (band, lo, hi) in enumerate(base.BANDS):
            band_rows = shell_rows & band_mask(a.wavelength_A, band_index)
            for map_name, map_mask in (("mapped", mapped), ("unmapped", ~mapped)):
                for code, disposition in DISPOSITIONS:
                    selected = band_rows & map_mask & (row_disposition == code)
                    output.append({
                        "shell": int(shell), "band": band,
                        "lambda_lo_A": lo, "lambda_hi_A": hi,
                        "mapping": map_name,
                        "epay_disposition_code": code,
                        "epay_disposition": disposition,
                        "rows": int(np.count_nonzero(selected)),
                        "line_emission_energy": sum64(m["energy"], selected),
                        "eta_line_sum": sum64(m["eta"], selected),
                        "chi_line_sum": sum64(m["chi"], selected),
                        "chi_line_dnu_integral": sum64(m["chi_integral"], selected),
                    })
    return output


def negative_control(a: Arrays, m: dict[str, np.ndarray],
                     canonical_mapped: np.ndarray) -> dict[str, Any]:
    defective_mapped = (a.flags & np.uint32(DEFECTIVE_MAP_BIT)) != 0
    selected = band_mask(a.wavelength_A, 5)
    canonical = metric_split(m["energy"], selected, canonical_mapped)
    defective = metric_split(m["energy"], selected, defective_mapped)
    canonical_chi = metric_split(m["chi"], selected, canonical_mapped)
    defective_chi = metric_split(m["chi"], selected, defective_mapped)
    changed = (canonical[2] != defective[2] or
               canonical_chi[2] != defective_chi[2])
    require(changed, "seeded mapping-predicate defect did not change the aggregate")
    return {
        "status": "EXPECTED-CHANGE-OBSERVED",
        "injection": "replace CMF_LP_F_NLTE_ION bit 0 with nonexistent bit 31",
        "canonical_mapped_rows": int(np.count_nonzero(canonical_mapped)),
        "defective_mapped_rows": int(np.count_nonzero(defective_mapped)),
        "canonical_unmapped_line_emission_energy": canonical[2],
        "defective_unmapped_line_emission_energy": defective[2],
        "canonical_unmapped_chi_line_sum": canonical_chi[2],
        "defective_unmapped_chi_line_sum": defective_chi[2],
    }


def aggregate(a: Arrays, provenance: dict[str, Any]) -> dict[str, Any]:
    m = measures(a)
    mapped = (a.flags & np.uint32(F_NLTE_ION)) != 0
    shell_band = shell_band_rows(a, m, mapped)
    ion_rank = ion_rank_rows(a, m["energy"], mapped)
    crosstab = crosstab_rows(a, m, mapped)
    headline = pooled_ball(a, m, mapped)
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "provenance": provenance,
        "definitions": {
            "mapping_predicate": "(flags & (1 << 0)) != 0",
            "writer_predicate": "nlte->nlte_line_map && nlte->nlte_line_map[l] >= 0",
            "eps_phys": int(a.eps_phys),
            "eta_l": ("w * eps_l * S_l_used" if a.eps_phys else "w * S_l_used"),
            "eta_l_production_site": "src/lumina_cmfgen.c:1371-1395 (eps_phys branch)",
            "eps_l_epoch": "post-clamp as recorded by the writer; production applied "
                           "eps_floor/eps_cap before serialisation",
            "line_emission_energy": "eta_l * dnu[line.bin]",
            "line_emission_energy_noeps": "w * S_l_used * dnu[line.bin]; eps_l omitted, "
                                          "diagnostic contrast only, not the production form",
            "chi_line": "w",
            "thin_numerator": "dump w is authoritative; writer used tau when tau <= 1e-6",
            "undefined_fraction": "null in JSON and UNDEFINED in CSV when denominator is zero",
            "bin_total_apportionment": False,
            "clamp": 0, "floor": 0, "cap": 0, "fallback": 0,
            "substitution": 0,
        },
        "ion_label_convention": {
            "ion_number_base": 0,
            "spectroscopic_stage": "Roman(ion_number_raw + 1)",
            "examples": {"0": "I", "1": "II", "2": "III", "3": "IV", "4": "V"},
        },
        "headline_selected_five_shells": headline,
        "shell_band": shell_band,
        "unmapped_ion_ranks": ion_rank,
        "epay_crosstab": crosstab,
        "negative_control": negative_control(a, m, mapped),
        "determinism": {
            "same_input_two_complete_aggregations": 2,
            "all_csv_json_markdown_payloads_byte_identical": True,
        },
        "scope_limit": "Only shells 0,8,16,20,45 were captured; no all-shell generalization.",
    }


def report_text(result: dict[str, Any]) -> str:
    h = result["headline_selected_five_shells"]
    d = result["definitions"]
    rows = [row for row in result["shell_band"] if row["band"] == "BALL"]
    cross = result["epay_crosstab"]
    thick = math.fsum(row["line_emission_energy"] for row in cross
                      if row["band"] == "BALL" and row["mapping"] == "unmapped"
                      and row["epay_disposition"] == "thick_exempt")
    replaced = math.fsum(row["line_emission_energy"] for row in cross
                         if row["band"] == "BALL" and row["mapping"] == "unmapped"
                         and row["epay_disposition"] == "rate_shape_replaced")
    disposition_conclusion = (
        "미매핑 에너지는 thick_exempt와 rate_shape_replaced 양쪽에 존재한다. "
        "따라서 매핑 여부와 EPAY 처분은 표본 전체에서는 독립 축이다. 단, 한 셀의 "
        "disposition은 단일 값이므로 같은 셀에서 두 EPAY 처분이 겹치지는 않는다."
        if thick > 0.0 and replaced > 0.0 else
        "미매핑 에너지가 thick_exempt와 rate_shape_replaced 양쪽 모두에서 양수라는 "
        "증거는 이 표본에서 성립하지 않는다. 교차표의 0을 그대로 유지하며 일반화하지 않는다."
    )
    lines = [
        "# UV NLTE map split", "", "## 결론", "",
        ("선택된 5개 셸을 합친 BALL(600–3000 Å)에서 매핑 선의 방출 에너지 몫은 "
         f"`{h['line_emission_energy_mapped_fraction']!r}`, 미매핑 선—즉 이 정의에서 "
         f"ALI가 도달할 수 없는 선—의 몫은 `{h['line_emission_energy_unmapped_fraction']!r}`이다."),
        "", (f"여기서 에너지는 행별로 `eta_l = {d['eta_l']}`를 만든 뒤 "
             "`eta_l * dnu[line.bin]`를 합한 값이며, 이는 "
             f"`{d['eta_l_production_site']}`의 조립식과 같은 형태다"
             f"(덤프 header `eps_phys={d['eps_phys']}`). `eps_l`은 production이 "
             "eps_floor/eps_cap을 적용한 뒤의 값이다. bin 총 `eta_line`을 opacity 몫으로 "
             "재배분하지 않았다. `*_noeps` 열은 `eps_l`을 뺀 대조값이며 production 형태가 "
             "아니다."), "", "## 셸별 BALL", "",
        "| shell | mapped energy | unmapped energy | unmapped fraction | unmapped chi fraction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['shell']} | {row['line_emission_energy_mapped']!r} | "
            f"{row['line_emission_energy_unmapped']!r} | "
            f"{row['line_emission_energy_unmapped_fraction']!r} | "
            f"{row['chi_line_sum_unmapped_fraction']!r} |")
    lines += [
        "", "B0–B4를 포함한 전체 셸×밴드 수치는 동봉 CSV/JSON의 `shell_band`에 있다.",
        "", "## 판정 규약", "",
        "- production 조립식은 `src/lumina_cmfgen.c:1369-1395`이다. 특히 얇은 선의 "
        "분자는 `tau <= 1e-6`일 때 `tau`이며, 본 분석은 이를 이미 반영한 dump `w`를 "
        "그대로 소비한다.",
        "", "- writer는 `src/lumina_cmfgen.c:822-824`에서 "
        "`nlte->nlte_line_map && nlte->nlte_line_map[l] >= 0`일 때만 bit 0 "
        "`CMF_LP_F_NLTE_ION`을 세운다(`src/lumina_cmfgen.c:528-534`). 따라서 "
        "`flags & 1`이 매핑 판정 술어이고, bit가 없는 행이 미매핑이다.",
        "", "- `ion_number`는 0-기반이다. `src/lumina_plasma.c:7672-7684`가 O I/O II/O III를 "
        "각각 원값 0/1/2로 명시한다. CSV는 `ion_number_raw`와 "
        "`spectroscopic_stage=Roman(ion_number_raw+1)`을 함께 기록한다.",
        "", "## EPAY 교차표", "", disposition_conclusion, "",
        f"선택 5셸 BALL 미매핑 에너지 중 thick_exempt 합은 `{thick!r}`, "
        f"rate_shape_replaced 합은 `{replaced!r}`이다. 상세 교차표는 "
        "`uv_mapsplit_epay_crosstab.csv`에 있다.",
        "", "주의: artifact disposition은 `src/lumina_cmfgen.c:904-919`의 기록값이다. "
        "기존 독립 리뷰 `docs/CODEX_STAGE32_RUNG1_REVIEW.md:F1`이 지적했듯 writer의 "
        "rate-shape 재구성에는 production 분기의 `acc_w > 0` 조건이 빠져 있다. 따라서 "
        "이 표는 payload에 기록된 EPAY disposition과의 교차이며 branch-site 관측으로 "
        "과대해석하지 않는다.",
        "", "## 무결성 및 자기검사", "",
        f"- schema `{result['provenance']['schema']}`, iteration "
        f"`{result['provenance']['iteration']}`, field_generation "
        f"`{result['provenance']['field_generation']}`, SHA-256 "
        f"`{result['provenance']['sha256']}`를 fail-closed로 검증했다.",
        "", "- 같은 입력을 두 번 집계·직렬화해 모든 산출물의 byte identity를 확인했다.",
        "", "- 매핑 bit 0 대신 존재하지 않는 bit 31을 읽도록 술어 결함을 주입했고 "
        "집계 변경을 관측했다(`negative_control.status=EXPECTED-CHANGE-OBSERVED`).",
        "", "- clamp/floor/cap/fallback/대체를 적용하지 않았다. 분모가 0인 분율은 "
        "JSON `null`, CSV `UNDEFINED`로 기록한다.",
        "", "## 범위 제한", "",
        "이 결론은 capture에 들어 있는 셸 0, 8, 16, 20, 45에만 해당한다. 전 셸 "
        "분포로 일반화하지 않는다.", "", "## 재현 명령", "",
        "```bash",
        ("python3 scripts/uv_mapsplit_offline.py --linepop "
         "/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10 "
         "--outdir validation/uv_mapsplit --report docs/CODEX_UV_MAPSPLIT.md"),
        "```", "",
    ]
    return "\n".join(lines)


def payloads(result: dict[str, Any]) -> dict[str, bytes]:
    return {
        "uv_mapsplit_shell_band.csv": csv_bytes(result["shell_band"]),
        "uv_mapsplit_unmapped_ion_rank.csv": csv_bytes(result["unmapped_ion_ranks"]),
        "uv_mapsplit_epay_crosstab.csv": csv_bytes(result["epay_crosstab"]),
        "uv_mapsplit.json": json_bytes(result),
        "CODEX_UV_MAPSPLIT.md": report_text(result).encode("utf-8"),
    }


def fixture_arrays() -> Arrays:
    return Arrays(
        shells=np.asarray(EXPECTED_SHELLS, dtype=np.int64),
        shell_slot=np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64),
        flags=np.asarray([F_NLTE_ION, 0, F_NLTE_ION, 0, F_NLTE_ION, 0], dtype=np.uint32),
        bins=np.asarray([0, 1, 1, 2, 3, 4], dtype=np.int64),
        wavelength_A=np.asarray([700.0, 1200.0, 1499.0, 1800.0, 2300.0, 2800.0]),
        Z=np.asarray([8, 14, 26, 26, 28, 20], dtype=np.int64),
        ion_number=np.asarray([0, 1, 2, 1, 3, 4], dtype=np.int64),
        w=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        S_l_used=np.asarray([2.0, 3.0, 5.0, 7.0, 11.0, 13.0]),
        eps_l=np.asarray([0.25, 0.5, 0.75, 1.25, 1.5, 2.0]),
        eps_phys=1,
        dnu=np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
        disposition=np.tile(np.asarray([1, 2, 1, 2, 0], dtype=np.uint8), (50, 1)),
    )


def self_test() -> dict[str, Any]:
    fixture = fixture_arrays()
    fixture_noeps = fixture._replace(eps_phys=0)
    validate_arrays(fixture)
    validate_arrays(fixture_noeps)
    measures_eps = measures(fixture)
    measures_noeps = measures(fixture_noeps)
    require(measures_noeps["energy"].tobytes() ==
            measures_noeps["energy_noeps"].tobytes(),
            "eps_phys=0 fixture energy was not bitwise identical to noeps")
    require(measures_eps["energy"].tobytes() !=
            measures_eps["energy_noeps"].tobytes(),
            "eps_phys=1 fixture energy did not differ from noeps")
    provenance = {"schema": "LCMFLP01-v1-fixture", "iteration": 10,
                  "field_generation": 10, "sha256": "fixture"}
    first = payloads(aggregate(fixture, provenance))
    second = payloads(aggregate(fixture, provenance))
    require(first == second, "fixture repeat was not byte-identical")
    result = aggregate(fixture, provenance)
    require(result["negative_control"]["status"] == "EXPECTED-CHANGE-OBSERVED",
            "fixture predicate negative control did not fire")
    require(spectroscopic_stage(0) == "I" and spectroscopic_stage(4) == "V",
            "ion label convention self-test failed")
    return {
        "status": "PASS", "repeat_payloads_byte_identical": True,
        "payload_count": len(first),
        "mapping_predicate_negative_control": "EXPECTED-CHANGE-OBSERVED",
        "eps_phys_0_energy_bitwise_identical_to_noeps": True,
        "eps_phys_1_energy_differs_from_noeps": True,
        "ion_labels_0_through_4": [spectroscopic_stage(x) for x in range(5)],
        "clamp": 0, "floor": 0, "cap": 0, "fallback": 0, "substitution": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linepop", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            print(json.dumps(self_test(), indent=2, sort_keys=True))
            return 0
        if args.linepop is None or args.outdir is None or args.report is None:
            parser.error("--linepop, --outdir, and --report are required")
        linepop = base.parse_linepop(args.linepop.resolve())
        provenance = {
            "path": str(linepop.path), "schema": linepop.manifest["schema"],
            "iteration": linepop.header["iteration"],
            "field_generation": linepop.header["field_generation"],
            "sha256": linepop.manifest["sha256"], "rows": linepop.header["rows"],
            "selected_shells": [int(x) for x in linepop.shells],
            "lambda_window_A": linepop.header["lambda_window_A"],
        }
        arrays = arrays_from_linepop(linepop)
        first_result = aggregate(arrays, provenance)
        second_result = aggregate(arrays, provenance)
        first = payloads(first_result)
        second = payloads(second_result)
        require(first == second, "same-input repeat aggregation was not byte-identical")
        outdir = args.outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        for name, content in first.items():
            if name != "CODEX_UV_MAPSPLIT.md":
                (outdir / name).write_bytes(content)
        report = args.report.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(first["CODEX_UV_MAPSPLIT.md"])
        print(json.dumps({
            "schema": SCHEMA, "status": "PASS",
            "headline_selected_five_shells":
                first_result["headline_selected_five_shells"],
            "repeat_outputs_byte_identical": True,
            "negative_control": first_result["negative_control"]["status"],
            "report": str(report), "outdir": str(outdir),
        }, indent=2, sort_keys=True))
        return 0
    except (base.OfflineError, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED-FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
