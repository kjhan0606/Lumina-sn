#!/usr/bin/env python3
"""원본 덱을 보존하고 J_nu 규약의 Einstein B 열만 고친 별도 뷰를 만든다.

물리율 적합이나 인구 보정이 아니다. 기존 A,nu,g로부터 A/B=2h nu^3/c^2를
복원한다. 다른 모든 CSV 열은 문자열 그대로 보존하며 원본은 수정하지 않는다.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def repair(deck, out):
    deck, out = deck.resolve(), out.absolute()
    if out.exists() or out.is_symlink():
        raise ValueError("기존 출력 보존")
    levels_path, lines_path = deck / "levels.csv", deck / "line_list.csv"
    weights = {}
    with levels_path.open() as stream:
        for row in csv.DictReader(stream):
            key = tuple(int(row[k]) for k in ("atomic_number", "ion_number", "level_number"))
            if key in weights or not float(row["g"]) > 0:
                raise ValueError("중복 준위 또는 유효하지 않은 통계가중치")
            weights[key] = float(row["g"])
    out.mkdir()
    # 알려진 소형 원자 덱 한 디렉터리만 열거한다. 재귀 복제는 하지 않는다.
    for source in deck.iterdir():
        if source.is_file() and source.name != "line_list.csv":
            (out / source.name).symlink_to(source)
    count = 0
    census = dict(J_nu=0, legacy_8pi=0, exact_zero=0)
    ratio_min, ratio_max, worst_db = math.inf, 0.0, 0.0
    original_other, repaired_other = hashlib.sha256(), hashlib.sha256()
    with lines_path.open(newline="") as source, (out / "line_list.csv").open("x", newline="") as dest:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            other = [row[k] for k in reader.fieldnames if k not in ("B_lu", "B_ul")]
            original_other.update(json.dumps(other, separators=(",", ":")).encode() + b"\n")
            Z, ion = int(row["atomic_number"]), int(row["ion_number"])
            gl = weights[(Z, ion, int(row["level_number_lower"]))]
            gu = weights[(Z, ion, int(row["level_number_upper"]))]
            A, nu, old_up, old_down = (float(row[k]) for k in ("A_ul", "nu", "B_lu", "B_ul"))
            if not all(math.isfinite(v) and v >= 0 for v in (A, nu, old_up, old_down)) or nu == 0:
                raise ValueError("유효하지 않은 Einstein 계수/주파수")
            new_down = A * C_CGS**2 / (2 * H_CGS * nu**3)
            new_up = new_down * gu / gl
            if A == 0:
                if old_up != 0 or old_down != 0:
                    raise ValueError("영 A와 비영 B가 혼합됨")
                census["exact_zero"] += 1
            else:
                if old_up == 0 or old_down == 0:
                    raise ValueError("양 A와 영 B가 혼합됨")
                ratio = new_down / old_down
                # 원래 CSV는 7자리 유효숫자다. 규약 식별에만 반올림 오차 허용.
                label = next((name for name, expected in (("J_nu", 1.0), ("legacy_8pi", 4 * math.pi))
                              if abs(ratio / expected - 1) <= 3e-6), None)
                if label is None or abs(old_up * gl / (old_down * gu) - 1) > 3e-6:
                    raise ValueError("설명되지 않는 Einstein B 규약/가중치 불일치")
                census[label] += 1
                ratio_min, ratio_max = min(ratio_min, ratio), max(ratio_max, ratio)
                worst_db = max(worst_db, abs(new_up * gl / (new_down * gu) - 1))
            row["B_lu"], row["B_ul"] = format(new_up, ".17g"), format(new_down, ".17g")
            writer.writerow(row)
            other_after = [row[k] for k in reader.fieldnames if k not in ("B_lu", "B_ul")]
            repaired_other.update(json.dumps(other_after, separators=(",", ":")).encode() + b"\n")
            count += 1
    if count == 0 or original_other.digest() != repaired_other.digest():
        raise ValueError("빈 선 목록 또는 B 이외 열 변경")
    manifest = dict(scope="Einstein B convention only; no population or physical-fit change",
                    rows=count, old_conventions=census,
                    correction_factor_range=[ratio_min, ratio_max] if ratio_max else None,
                    max_g_weighted_DB_relative_error=worst_db,
                    unchanged_non_B_columns_sha256=original_other.hexdigest(),
                    source_deck=str(deck), output_view=str(out),
                    source_sha256={str(p): sha(p) for p in (levels_path, lines_path)},
                    output_line_list_sha256=sha(out / "line_list.csv"),
                    script_sha256=sha(Path(__file__)), final_goal="NOT_JUDGED")
    (out / "einstein_B_repair.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(repair(args.deck, args.out), indent=2))


if __name__ == "__main__":
    main()
