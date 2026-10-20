import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import expand_atomic_data_cmfgen as atomic
from repair_cmfgen_einstein_view import C_CGS, H_CGS, repair
from summarize_cmfgen_einstein_response import input_population, row_flow, summarize_full_response
from prepare_cmfgen_owner_response import write_input


def fixture(path):
    levels = [(14, 2, 0, 0, 2, 0, 0, "ground"),
              (14, 2, 1, 5, 6, 0, 1, "excited")]
    lines = {"Z": np.array([14]), "ion": np.array([2]), "lo": np.array([0]),
             "up": np.array([1]), "f_lu": np.array([.1]), "nu": np.array([1e15]),
             "A_ul": np.array([1e8]), "lam": np.array([C_CGS / 1e15 * 1e8])}
    atomic.write_levels_csv(levels, path / "levels.csv")
    atomic.write_line_list_csv(lines, levels, path / "line_list.csv")
    return levels, lines


def test_producer_planck_einstein_detailed_balance(tmp_path):
    fixture(tmp_path)
    row = next(csv.DictReader((tmp_path / "line_list.csv").open()))
    A, nu, Blu, Bul = (float(row[k]) for k in ("A_ul", "nu", "B_lu", "B_ul"))
    for T in (5000, 10000, 50000):
        x = H_CGS * nu / (1.380649e-16 * T)
        J = (2 * H_CGS * nu**3 / C_CGS**2) / math.expm1(x)
        ratio = Blu * J / (A + Bul * J)
        assert ratio == pytest.approx(3 * math.exp(-x), rel=2e-6)
        # 두 B 사이의 가중치 항등식만 보면 4*pi 결함을 검출하지 못한다.
        bad_ratio = (Blu / (4 * math.pi)) * J / (A + (Bul / (4 * math.pi)) * J)
        assert abs(bad_ratio / (3 * math.exp(-x)) - 1) > .5


def test_macro_seed_uses_same_J_nu_convention(tmp_path):
    levels, lines = fixture(tmp_path)
    ma, mr = tmp_path / "macro.csv", tmp_path / "references.csv"
    atomic.write_macro_atom(lines, levels, {(14, 3, 1): 0, (14, 3, 2): 1}, ma, mr)
    up = next(r for r in csv.DictReader(ma.open()) if r["transition_type"] == "1")
    x = H_CGS * 1e15 / (atomic.K_CGS * 10000)
    expected = 1e8 * 3 * .5 / math.expm1(x)
    assert float(up["transition_probability"]) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("legacy", [False, True])
def test_repair_preserves_original_and_non_B_columns(tmp_path, legacy):
    deck = tmp_path / "deck"
    deck.mkdir()
    fixture(deck)
    line_path = deck / "line_list.csv"
    if legacy:
        rows = list(csv.DictReader(line_path.open()))
        for row in rows:
            for key in ("B_lu", "B_ul"):
                row[key] = format(float(row[key]) / (4 * math.pi), ".17g")
        with line_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    original = line_path.read_bytes()
    out = tmp_path / "view"
    result = repair(deck, out)
    assert line_path.read_bytes() == original
    assert result["old_conventions"]["legacy_8pi" if legacy else "J_nu"] == 1
    a = next(csv.DictReader(line_path.open()))
    b = next(csv.DictReader((out / "line_list.csv").open()))
    assert {k: v for k, v in a.items() if k not in ("B_lu", "B_ul")} == {
        k: v for k, v in b.items() if k not in ("B_lu", "B_ul")}
    assert (out / "levels.csv").resolve() == (deck / "levels.csv").resolve()
    with pytest.raises(ValueError, match="기존 출력"):
        repair(deck, out)


def test_repair_rejects_unexplained_B(tmp_path):
    deck = tmp_path / "deck"
    deck.mkdir()
    fixture(deck)
    path = deck / "line_list.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["B_ul"] = str(float(rows[0]["B_ul"]) * 2)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Einstein B"):
        repair(deck, tmp_path / "bad")


def test_saved_input_and_flow_accounting(tmp_path):
    path = tmp_path / "owner.bin"
    profiles = np.array([[1e-14, 1e4, 1e7, 1e8, 2e8], [2e-14, 2e4, 2e7, 2e8, 3e8]])
    populations = np.array([[3., 7.], [4., 6.]])
    write_input(path, np.array([1e10, 1e20]), profiles,
                [(14, 1, 0), (14, 5, 0)], populations, np.ones((2, 2)), 1e6)
    identity = [dict(internal_stage="1", sl_id="0"), dict(internal_stage="5", sl_id="0")]
    for depth in (1, 2):
        pop, profile = input_population(path, depth, 14, identity)
        assert np.array_equal(pop, populations[depth - 1])
        assert np.array_equal(profile, profiles[depth - 1])
    A, pop = np.array([[-2., 3.], [2., -3.]]), np.array([4., 5.])
    for i in range(2):
        flow = row_flow(A, pop, i)
        assert flow["net"] == pytest.approx((A @ pop)[i])
        assert flow["inflow"] >= 0 and flow["outflow"] >= 0


def test_full_depth_summary_keeps_coverage_and_fixed_inputs(tmp_path):
    before, after = tmp_path / "before.csv", tmp_path / "after.csv"
    fields = ["depth_1based", "Z", "stage", "Te_K", "ne_cm3", "n_element", "pre_fraction",
              "candidate_fraction", "commit_performed", "candidate_valid", "kappa_2", "raw_rank"]
    for path, fractions in ((before, [.2, .7, .1, 0, 0]), (after, [.12, .78, .1, 0, 0])):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for d in range(1, 91):
                for Z in (14, 16):
                    for i, stage in enumerate(range(2, 7)):
                        writer.writerow(dict(depth_1based=d, Z=Z, stage=stage, Te_K=10000,
                            ne_cm3=1e7, n_element=1000, pre_fraction=[.1, .8, .1, 0, 0][i],
                            candidate_fraction=fractions[i], commit_performed=int(Z == 16),
                            candidate_valid=1, kappa_2=1e6, raw_rank=246))
    summary = summarize_full_response(before, after)
    assert summary["rows"] == 900
    assert summary["per_element"]["16"]["committed"] == 90
    assert summary["per_element"]["14"]["committed"] == 0
    metric = summary["per_element"]["14"]["stages"]["2"]
    assert metric["before"]["abs_fraction_error_median_p95_max"] == pytest.approx([.1]*3)
    assert metric["after"]["abs_fraction_error_median_p95_max"] == pytest.approx([.02]*3)
    # 누락 행을 통계에서 조용히 제외하지 않는다.
    after.write_text("\n".join(after.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="누락"):
        summarize_full_response(before, after)
