#!/usr/bin/env python3
"""MC 침적을 공개19.48일과 비교한다. 목표 자료는 소스/정규화에 쓰지 않는다."""
import bisect
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


def reference(path, epoch):
    lines = path.read_text().splitlines()
    times = next(line.split(":", 1)[1].split() for line in lines if line.startswith("#TIMES"))
    column = [float(x) for x in times].index(epoch) + 1
    rows = [[float(x) for x in line.split()] for line in lines if line and not line.startswith("#")]
    return [row[0] for row in rows], [row[column] for row in rows]


def volume_average(x, y, lower, upper):
    """공개 q(v)의 선형 보간을 v²dv로 적분. 정의역 밖은 외삽하지 않는다."""
    if lower < x[0] or upper > x[-1]:
        return None
    knots = [lower] + [v for v in x if lower < v < upper] + [upper]
    total = 0.0
    for a, b in zip(knots, knots[1:]):
        i = min(bisect.bisect_right(x, (a + b) / 2) - 1, len(x) - 2)
        slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        qa = y[i] + slope * (a - x[i])
        # 이동 좌표 u=v-a로 적분해 큰 v의 차감 상쇄를 피한다.
        h = b - a
        total += qa * (a * a * h + a * h * h + h**3 / 3)
        total += slope * (a * a * h * h / 2 + 2 * a * h**3 / 3 + h**4 / 4)
    return total / ((upper - lower) * (upper * upper + upper * lower + lower * lower) / 3)


def main():
    epoch = 19.48
    profile = Path("input/edep_toy06_cmfgen.txt")
    lightcurve = Path("input/lbol_edep_toy06_cmfgen.txt")
    x, y = reference(profile, epoch)
    public_total = next(float(row[2]) for line in lightcurve.read_text().splitlines()
                        if not line.startswith("#") and (row := line.split()) and float(row[0]) == epoch)
    output = {"epoch_day": epoch, "fitted_normalization": False, "spectrum_compared": False,
              "nuclear_identity_verified": False, "public_total_erg_s": public_total,
              "reference_velocity_domain_km_s": [x[0], x[-1]], "models": [],
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (profile, lightcurve)}}
    if sys.argv[1:] == ["sources"]:
        cases = [(name, Path(f"gamma_variant_{name}.csv")) for name in
                 ("raw_endpoint", "raw_average", "completed_endpoint", "completed_average")]
    else:
        assert len(sys.argv) == 1
        cases = [(f"raw_endpoint_{ns}", Path(f"gamma_{ns}_{epoch:.2f}d.csv")) for ns in (100, 200)]
    for name, path in cases:
        with path.open() as stream:
            rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
        ns = len(rows)
        assert ns in (100, 200)
        compared, total, common_model, common_reference, l1 = [], 0.0, 0.0, 0.0, 0.0
        for row in rows:
            lo, hi = float(row["vlo_cm_s"]) / 1e5, float(row["vhi_cm_s"]) / 1e5
            volume = 4 * math.pi / 3 * (epoch * 86400 * 1e5)**3 * (hi - lo) * (hi * hi + lo * hi + lo * lo)
            value = float(row["gamma_deposition"]) + float(row["positron_local"])
            total += volume * value
            ref = volume_average(x, y, lo, hi)
            if ref is None:
                continue
            common_model += volume * value
            common_reference += volume * ref
            l1 += abs(value - ref) * volume
            compared.append({"shell": int(row["shell"]), "vlo_km_s": lo, "vhi_km_s": hi,
                             "model": value, "reference_volume_average": ref,
                             "ratio": value / ref if ref else None,
                             "sample_se": float(row["gamma_standard_error"]),
                             "positive_packet_count": int(row["positive_packet_count"])})
        record = {"case": name, "shells": ns, "total_gamma_plus_positron_erg_s": total,
                  "total_over_public": total / public_total, "common_domain_cells": len(compared),
                  "common_domain_integral_ratio": common_model / common_reference,
                  "common_domain_relative_l1": l1 / common_reference,
                  "zero_tally_cells": sum(int(row["positive_packet_count"]) == 0 for row in rows),
                  "cells": compared}
        output["models"].append(record)
        output["sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        print("GAMMA_PUBLIC_COMPARISON", json.dumps({k: v for k, v in record.items() if k != "cells"}))
    with Path("gamma_public_comparison.json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
