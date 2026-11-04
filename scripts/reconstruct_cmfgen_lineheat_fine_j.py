#!/usr/bin/env python3
"""Read-only fixed-state CMFGEN line-heat counterfactual on a fine J grid.

The native line identity, populations, and source expression remain fixed.
For selected ions/depths this replaces the reconstructed native profile mass
and profile-averaged J with Q=1 and the same line profile integrated over the
fine EDDFACTOR J field. It does not rerun CMFGEN, solve SE/RE, or mutate data.

The native frequency grid and line centers are printed values, so this is an
approximate reconstruction, not a capture of internal LINE_QW_SUM/TRANS_INFO.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import zip_longest
from pathlib import Path
from typing import Sequence

import numpy as np

from compare_lumina_native_grid_heat import (
    fine_grid_trapezoid_weights,
    native_trapuneq_weights,
    parse_native_cont_freq_text,
)
from extract_cmfgen_line_net_fixture import (
    CMFGEN_RE_INTERNAL_TO_CGS,
    Header,
    NetHeader,
    lineheat_header,
    netrate_header,
    vectors,
)
from summarize_cmfgen_lineheat_ion_owners import transition_owner


DEFAULT_IONS = ("Si II", "Si III", "Si IV", "S III", "Co IV", "Fe IV", "Co V", "Ni IV")
DEFAULT_WIDTH_FRACTION = 4.506845517e-5


def compatible(line: Header, net: NetHeader) -> bool:
    scale = max(abs(line.frequency_field), 1.0)
    return (
        line.line_id == net.line_id
        and line.transition == net.transition
        and line.lower == net.lower
        and line.upper == net.upper
        and abs(line.frequency_field - net.frequency_field) <= 1.0e-9 * scale
    )


def profile_slice(
    frequency_hz: np.ndarray,
    integration_weights_hz: np.ndarray,
    center_hz: float,
    width_fraction: float,
) -> tuple[slice, np.ndarray, float]:
    """Return local weighted Gaussian samples and normalized profile mass."""
    if (
        frequency_hz.ndim != 1
        or integration_weights_hz.shape != frequency_hz.shape
        or len(frequency_hz) < 2
        or not np.isfinite(frequency_hz).all()
        or not np.isfinite(integration_weights_hz).all()
        or not math.isfinite(center_hz)
        or center_hz <= 0.0
        or not math.isfinite(width_fraction)
        or width_fraction <= 0.0
        or np.any(np.diff(frequency_hz) <= 0.0)
    ):
        raise ValueError("invalid profile-integration grid or width")
    sigma_hz = center_hz * width_fraction
    first = max(0, int(np.searchsorted(frequency_hz, center_hz - 9.0 * sigma_hz)) - 1)
    last = min(
        len(frequency_hz),
        int(np.searchsorted(frequency_hz, center_hz + 9.0 * sigma_hz)) + 1,
    )
    local_frequency = frequency_hz[first:last]
    weighted_profile = integration_weights_hz[first:last] * np.exp(
        -((local_frequency - center_hz) / sigma_hz) ** 2
    )
    raw_mass = float(np.sum(weighted_profile, dtype=np.float64))
    mass = raw_mass / (math.sqrt(math.pi) * sigma_hz)
    if not math.isfinite(raw_mass) or raw_mass <= 0.0 or not math.isfinite(mass):
        raise ValueError("invalid sampled line profile mass")
    return slice(first, last), weighted_profile, mass


def reweighted_line_heat(
    native_heat: float,
    native_znet: float,
    native_mass: float,
    native_jbar: float,
    fine_jbar: float,
    fine_mass: float,
) -> tuple[float, float]:
    """Return Q=1-only and Q=1+fine-J heat in native LINEHEAT sign/units."""
    values = (native_heat, native_znet, native_mass, native_jbar, fine_jbar, fine_mass)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite line-heat counterfactual operand")
    if native_znet == 0.0 or native_mass <= 0.0 or native_jbar <= 0.0 or fine_mass <= 0.0:
        raise ValueError("degenerate line-heat counterfactual operand")
    alpha = (native_mass - native_znet) / (native_mass * native_jbar)
    q_one = native_heat / native_mass
    q_one_fine_j = (native_heat / native_znet) * fine_mass * (1.0 - alpha * fine_jbar)
    if not math.isfinite(q_one) or not math.isfinite(q_one_fine_j):
        raise ValueError("nonfinite reweighted line heat")
    return q_one, q_one_fine_j


def profile_mass_printing_interval(frequency_hz,center_hz,width_fraction,*,digits=8):
    """인쇄 좌표 반올림 상자의 Q 범위. native binary Q의 직접 포획이 아니다.

    양의 사다리꼴 가중치와 Gaussian의 구간별 최솟값/최댓값을 합한다.
    하한에는 확실히6 Doppler 폭 안인 점만 포함하고, 상한에는 전 격자를 쓴다.
    가중치의 하한0은 구간 산술의 양수 조건이며 실제 데이터 보정이 아니다.
    """
    nu=np.asarray(frequency_hz,dtype='f8');width=center_hz*width_fraction
    if (nu.ndim!=1 or len(nu)<2 or not np.isfinite(nu).all() or np.any(nu<=0) or
            np.any(np.diff(nu)<=0) or not math.isfinite(center_hz) or center_hz<=0 or
            not math.isfinite(width_fraction) or width_fraction<=0 or not math.isfinite(width) or width<=0 or
            type(digits) is not int or digits<2):
        raise ValueError('Q 반올림 구간의 격자/폭/정밀도 오류')
    epsilon=.5*10.**(np.floor(np.log10(nu))-digits+1)
    low,high=nu-epsilon,nu+epsilon
    wl=np.r_[(low[1]-high[0])*.5,(low[2:]-high[:-2])*.5,(low[-1]-high[-2])*.5]
    wh=np.r_[(high[1]-low[0])*.5,(high[2:]-low[:-2])*.5,(high[-1]-low[-2])*.5]
    xl,xh=(low-center_hz)/width,(high-center_hz)/width
    near=np.where((xl<=0)&(xh>=0),0.,np.minimum(xl*xl,xh*xh))
    far=np.maximum(xl*xl,xh*xh)
    norm=1.77245385095516*width
    inside=(xl>=-6)&(xh<=6)
    qlow=np.sum(np.maximum(wl,0)*np.exp(-far)*inside/norm,dtype=np.longdouble)
    qhigh=np.sum(wh*np.exp(-near)/norm,dtype=np.longdouble)
    return float(qlow),float(qhigh)


def _compensated_add(total: np.ndarray, correction: np.ndarray, index: int, value: float) -> None:
    """Kahan-add one scalar to a long-double depth accumulator."""
    y = np.longdouble(value) - correction[index]
    updated = total[index] + y
    correction[index] = (updated - total[index]) - y
    total[index] = updated


def read_eddfactor_depths(
    path: Path, info_path: Path, depth_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Read only requested depth columns from the sealed EDDFACTOR records."""
    info = info_path.read_text(encoding="ascii", errors="strict").splitlines()
    if len(info) < 3:
        raise ValueError("EDDFACTOR_INFO is truncated")
    fields = info[2].split()
    if (
        len(fields) < 6
        or list(map(int, fields[:5])) != [90, 728, 8, 1, 4]
        or fields[5].upper() != "T"
    ):
        raise ValueError("EDDFACTOR layout differs from the sealed 90-depth record")
    size = path.stat().st_size
    if size % 728 != 0 or size // 728 <= 14:
        raise ValueError("EDDFACTOR record size/count is invalid")
    nfreq = size // 728 - 14
    headers = np.memmap(path, dtype="<f8", mode="r", offset=0, shape=(14, 91))
    data = np.memmap(path, dtype="<f8", mode="r", offset=14 * 728, shape=(nfreq, 91))
    try:
        if headers[4, 0] != 1.0:
            raise ValueError("EDDFACTOR FINISH header is not complete")
        frequency = np.asarray(data[:, 90], dtype=np.float64)
        order = np.argsort(frequency)
        frequency = frequency[order] * 1.0e15
        j_by_frequency = np.empty((nfreq, len(depth_indices)), dtype=np.float64)
        for output_depth, source_depth in enumerate(depth_indices):
            j_by_frequency[:, output_depth] = np.asarray(
                data[order, source_depth], dtype=np.float64
            )
    finally:
        headers._mmap.close()
        data._mmap.close()
    if (
        not np.isfinite(frequency).all()
        or np.any(frequency <= 0.0)
        or np.any(np.diff(frequency) <= 0.0)
        or not np.isfinite(j_by_frequency).all()
        or np.any(j_by_frequency < 0.0)
    ):
        raise ValueError("EDDFACTOR frequency or selected J columns are invalid")
    return frequency, j_by_frequency


def reconstruct(
    lineheat_path: Path,
    netrate_path: Path,
    cont_freq_path: Path,
    eddfactor_path: Path,
    eddfactor_info_path: Path,
    *,
    depth_indices: Sequence[int],
    ions: Sequence[str],
    width_fraction: float = DEFAULT_WIDTH_FRACTION,
) -> dict:
    if not depth_indices or len(set(depth_indices)) != len(depth_indices):
        raise ValueError("depth indices must be nonempty and unique")
    if any(index < 0 or index >= 90 for index in depth_indices):
        raise ValueError("depth index outside 90-point capture")
    if not ions or len(set(ions)) != len(ions):
        raise ValueError("ion selection must be nonempty and unique")

    ncf_nu, _, ncf_metadata = parse_native_cont_freq_text(
        cont_freq_path.read_text(encoding="ascii", errors="strict")
    )
    ncf_weights = native_trapuneq_weights(ncf_nu)
    fine_nu, fine_j = read_eddfactor_depths(
        eddfactor_path, eddfactor_info_path, depth_indices
    )
    fine_weights = fine_grid_trapezoid_weights(fine_nu)
    ncf_j = np.vstack(
        [np.interp(ncf_nu, fine_nu, fine_j[:, depth]) for depth in range(len(depth_indices))]
    )

    ion_index = {ion: index for index, ion in enumerate(ions)}
    depth_index = {depth: index for index, depth in enumerate(depth_indices)}
    shape = (len(ions), len(depth_indices))
    totals = {
        name: np.zeros(shape, dtype=np.longdouble)
        for name in ("native", "q_one", "q_one_fine_j", "absolute_q_delta", "absolute_fine_j_delta")
    }
    compensations = {name: np.zeros(shape, dtype=np.longdouble) for name in totals}
    line_counts = np.zeros(len(ions), dtype=np.int64)
    excluded = np.zeros(shape, dtype=np.int64)
    paired_records = 0

    line_records = vectors(lineheat_path, 90, lineheat_header)
    net_records = vectors(netrate_path, 90, netrate_header)
    for pair_index, pair in enumerate(zip_longest(line_records, net_records), 1):
        line_record, net_record = pair
        if line_record is None or net_record is None:
            raise ValueError("LINEHEAT/NETRATE record counts differ")
        line_header, line_values = line_record
        net_header, znet_values = net_record
        if not isinstance(line_header, Header) or not isinstance(net_header, NetHeader):
            raise ValueError("unexpected CMFGEN line-ledger header")
        if not compatible(line_header, net_header):
            raise ValueError(f"LINEHEAT/NETRATE identity mismatch at pair {pair_index}")
        paired_records += 1
        _, owner = transition_owner(line_header.transition)
        selected_ion = ion_index.get(owner)
        if selected_ion is None:
            continue

        center_hz = net_header.frequency_field * 1.0e15
        ncf_slice, ncf_profile, native_mass = profile_slice(
            ncf_nu, ncf_weights, center_hz, width_fraction
        )
        ncf_weight_sum = float(np.sum(ncf_profile, dtype=np.float64))
        native_jbar = (ncf_j[:, ncf_slice] @ ncf_profile) / ncf_weight_sum
        fine_slice, fine_profile, fine_mass = profile_slice(
            fine_nu, fine_weights, center_hz, width_fraction
        )
        fine_weight_sum = float(np.sum(fine_profile, dtype=np.float64))
        fine_jbar = (fine_profile @ fine_j[fine_slice, :]) / fine_weight_sum

        line_counts[selected_ion] += 1
        for output_depth, depth in enumerate(depth_indices):
            native_heat = float(line_values[depth])
            _compensated_add(
                totals["native"][selected_ion],
                compensations["native"][selected_ion],
                output_depth,
                native_heat,
            )
            try:
                q_one, q_one_fine_j = reweighted_line_heat(
                    native_heat,
                    float(znet_values[depth]),
                    native_mass,
                    float(native_jbar[output_depth]),
                    float(fine_jbar[output_depth]),
                    fine_mass,
                )
            except ValueError:
                excluded[selected_ion, output_depth] += 1
                continue
            _compensated_add(totals["q_one"][selected_ion], compensations["q_one"][selected_ion], output_depth, q_one)
            _compensated_add(totals["q_one_fine_j"][selected_ion], compensations["q_one_fine_j"][selected_ion], output_depth, q_one_fine_j)
            _compensated_add(totals["absolute_q_delta"][selected_ion], compensations["absolute_q_delta"][selected_ion], output_depth, abs(q_one-native_heat))
            _compensated_add(totals["absolute_fine_j_delta"][selected_ion], compensations["absolute_fine_j_delta"][selected_ion], output_depth, abs(q_one_fine_j-native_heat))

    factor = CMFGEN_RE_INTERNAL_TO_CGS
    rows = []
    for ion, i in ion_index.items():
        for depth, j in depth_index.items():
            rows.append({
                "ion": ion,
                "depth_one_based": depth + 1,
                "line_records": int(line_counts[i]),
                "counterfactual_excluded": int(excluded[i, j]),
                "native_lineheat_cgs_cooling_positive": float(totals["native"][i, j] * factor),
                "q_one_cgs_cooling_positive": float(totals["q_one"][i, j] * factor),
                "q_one_fine_j_cgs_cooling_positive": float(totals["q_one_fine_j"][i, j] * factor),
                "sum_abs_q_delta_cgs": float(totals["absolute_q_delta"][i, j] * factor),
                "sum_abs_q_fine_j_delta_cgs": float(totals["absolute_fine_j_delta"][i, j] * factor),
            })
    return {
        "schema": "cmfgen-fixedj-lineheat-q1-finej-counterfactual-v1",
        "physical_recalculation": False,
        "paired_line_records": paired_records,
        "selected_ions": list(ions),
        "selected_depths_one_based": [depth + 1 for depth in depth_indices],
        "width_fraction": width_fraction,
        "ncf_points": int(len(ncf_nu)),
        "fine_points": int(len(fine_nu)),
        "native_frequency_metadata": ncf_metadata,
        "units": "erg cm^-3 s^-1; positive means cooling",
        "method": "native ZNET=Q*(1-alpha*Jbar); infer alpha from printed native ZNET, then reweight to fine-grid Jbar and Q=1",
        "limitations": [
            "CONT_FREQ and NETRATE centers are printed coordinates, not exact internal native arrays",
            "native qsum and interval map are reconstructed; this is not exact LINE_QW_SUM",
            "fixed-state thermal counterfactual only; no SE, RE, transport, or ionization closure",
        ],
        "line_heat_by_ion_depth": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineheat", type=Path, required=True)
    parser.add_argument("--netrate", type=Path, required=True)
    parser.add_argument("--cont-freq", type=Path, required=True)
    parser.add_argument("--eddfactor", type=Path, required=True)
    parser.add_argument("--eddfactor-info", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(1, 91)), help="1-based CMFGEN depths; default: all 90")
    parser.add_argument("--ions", nargs="+", default=list(DEFAULT_IONS), help="normalized ion names")
    parser.add_argument("--width-fraction", type=float, default=DEFAULT_WIDTH_FRACTION)
    args = parser.parse_args(argv)
    result = reconstruct(
        args.lineheat,
        args.netrate,
        args.cont_freq,
        args.eddfactor,
        args.eddfactor_info,
        depth_indices=[depth - 1 for depth in args.depths],
        ions=args.ions,
        width_fraction=args.width_fraction,
    )
    json.dump(result, __import__("sys").stdout, indent=2, sort_keys=True)
    __import__("sys").stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
