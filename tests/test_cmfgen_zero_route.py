#!/usr/bin/env python3
"""문자열 헤더와 명시적 영 광전리 경로의 보존 회귀시험.

생산 덱은 읽기만 한다. 작은 임시 덱은 계산 노드 작업 경로에 둔다.
"""
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cmfgen_parser as P
import expand_atomic_data_cmfgen as B


class ZeroRouteTests(unittest.TestCase):
    def test_text_header(self):
        self.assertEqual(P._parse_header_value("3s_3p_3Po/3s3p3Po !Final state in ion"),
                         ("3s_3p_3Po/3s3p3Po", "Final state in ion"))
        self.assertEqual(P._parse_header_value("Megabarns !Cross-section unit"), ("Megabarns", "Cross-section unit"))
        self.assertEqual(P._parse_header_value("False !Split J levels"), ("False", "Split J levels"))
        self.assertEqual(P._parse_header_value(" 1.0D+07 !Default autoionization rate"),
                         ("1.0D+07", "Default autoionization rate"))
        self.assertEqual(P._parse_header_value("! comment"), (None, ""))
        self.assertEqual(P._parse_header_value("   ! comment"), (None, ""))

    def test_raw_phot_header_and_numeric_body(self):
        raw = ROOT / "data/atomic/cmfgen/SIL/II/19apr23"
        a, b = P.parse_phot(raw / "phot_data_A"), P.parse_phot(raw / "phot_data_B")
        self.assertEqual(a.final_state, "3s2_1Se/3s21Se")
        self.assertEqual(b.final_state, "3s_3p_3Po/3s3p3Po")
        self.assertEqual((a.final_state_g, b.final_state_g), (1, 9))
        self.assertEqual((a.cs_unit, a.split_J), ("Megabarns", False))
        self.assertEqual(a.entries[0].cs_type, 20)
        self.assertEqual(a.entries[0].n_points, 1487)
        self.assertEqual(a.entries[0].sigma_Mb[0], 1.414)

    def test_known_zero_distinct_from_unsupported(self):
        for params in (np.array([0.0]), np.array([0.0, 1.0, 3.0])):
            model = B._sigma_model(1, np.array([]), params, 1e15)
            self.assertIsNotNone(model)
            fn, knots, start, tag = model
            self.assertEqual(tag, "explicit_zero")
            np.testing.assert_array_equal(fn(np.array([5e14, 1e15, 2e15])), np.zeros(3))
        self.assertIsNone(B._sigma_model(1, np.array([]), np.array([1.0]), 1e15))
        self.assertIsNone(B._sigma_model(999, np.array([]), np.array([0.0]), 1e15))
        fn, _, _, tag = B._sigma_model(1, np.array([]), np.array([2.0, 1.0, 3.0]), 1e15)
        self.assertEqual(tag, "seaton")
        np.testing.assert_allclose(fn(np.array([1e15, 2e15])), [2e-18, 2.5e-19], rtol=1e-15, atol=0)

    def test_bake_retains_zero_presence(self):
        # One real forbidden Si II level and one tabulated ground level.
        raw = ROOT / "data/atomic/cmfgen/SIL/II/19apr23"
        osc, phot = P.parse_osc(raw / "osc_data"), P.parse_phot(raw / "phot_data_A")
        ion = {(14, 2): {"osc": osc, "levels": osc.levels, "n_kept": 5, "phot": phot}}
        lookup = {(14, 2, 1): 0, (14, 2, 3): 1}
        # Explicitly stay on GPFS, never the shared source/data directory.
        with tempfile.TemporaryDirectory(prefix="zero-route-", dir=os.environ["SI_ROUTE_TEST_TMP"]) as td:
            path = Path(td) / "sigma.bin"
            B.bake_sigma_bf_grid(ion, {}, [None, None], lookup, path)
            with path.open("rb") as f:
                _, _, nl, nf = struct.unpack("<IIii", f.read(16))
                f.read(16)
                flags = f.read(nl)
                f.seek(32 + nl + (-nl) % 8)
                values = np.frombuffer(f.read(), dtype="<f8").reshape(nl, nf)
            self.assertEqual(flags, bytes([1, 1]))
            self.assertGreater(values[0].max(), 0)
            np.testing.assert_array_equal(values[1], np.zeros(nf))

    def test_actual_si_positive_rows_unchanged(self):
        # Existing diagnostic verifies the exact 125-row input with no writer.
        from diagnose_si_bf_provenance_20261014 import main
        from contextlib import redirect_stdout
        import json
        previous = sys.argv
        try:
            sys.argv = ["provenance", "--bin-average", "--run",
                        "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1"]
            stream = io.StringIO()
            with patch.object(B, "SIGMA_EXACT_HYD", True), redirect_stdout(stream):
                main()
            report = json.loads(stream.getvalue())
        finally:
            sys.argv = previous
        self.assertEqual(report["compatible_rows_1e_8"], 125)
        self.assertEqual(report["saved_positive"], 107)

    def test_sulfur_term_matching_and_existing_positive_rows(self):
        import pandas as pd
        from dataclasses import replace
        raw = ROOT / "data/atomic/cmfgen/SUL/II/19apr23"
        osc, phot = P.parse_osc(raw / "osc_data"), P.parse_phot(raw / "phot_data_A")
        run = Path("/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/model")
        all_rows = pd.read_csv(run / "levels.csv")
        rows = all_rows[(all_rows.atomic_number == 16) & (all_rows.ion_number == 1)]
        n = len(rows)
        self.assertEqual(n, 322)
        self.assertEqual(list(rows.level_number), list(range(n)))
        np.testing.assert_array_equal(rows.configuration, osc.levels["config"][:n])
        lookup = {(16, 2, i + 1): i for i in range(n)}
        with (run / "cmfgen_sigma_bf.bin").open("rb") as f:
            _, _, nl, nf = struct.unpack("<IIii", f.read(16))
            nu_min, nu_max = struct.unpack("<dd", f.read(16))
            flags = np.frombuffer(f.read(nl), dtype="u1")
        saved = np.memmap(run / "cmfgen_sigma_bf.bin", mode="r", dtype="<f8",
                          offset=32 + nl + (-nl) % 8, shape=(nl, nf))[rows.index]
        previous = flags[rows.index]
        self.assertEqual(int(previous.sum()), 315)
        log_step = np.log(nu_max / nu_min) / nf
        nu_edges = nu_min * np.exp(np.arange(nf + 1) * log_step)
        with tempfile.TemporaryDirectory(prefix="s-term-route-", dir=os.environ["SI_ROUTE_TEST_TMP"]) as td:
            path = Path(td) / "sigma.bin"
            ion = {(16, 2): {"osc": osc, "levels": osc.levels, "n_kept": n, "phot": phot}}
            with patch.object(B, "SIGMA_EXACT_HYD", True):
                B.bake_sigma_bf_grid(ion, {}, [None] * n, lookup, path)
            with path.open("rb") as f:
                f.seek(32)
                actual_flags = f.read(n)
                f.seek(32 + n + (-n) % 8)
                values = np.frombuffer(f.read(), dtype="<f8").reshape(n, nf)
            self.assertEqual(actual_flags, bytes([1] * n))
            positive = previous == 1
            relative = []
            tail_changed = 0
            for i in np.flatnonzero(positive):
                cfg = B._norm_cfg(osc.levels["config"][i])
                entries = ([e for e in phot.entries if B._norm_cfg(e.config) == cfg]
                           if phot.split_J else
                           [e for e in phot.entries
                            if B._term_cfg(e.config) == B._term_cfg(cfg)])
                threshold = (osc.ionization_eV -
                             float(osc.levels["E_cm"][i]) * 1.239841984e-4) * B.EV_TO_ERG / B.H_CGS
                table_ends = []
                for entry in entries:
                    if entry.cs_type not in B._SIGMA_TABULATED:
                        continue
                    model = B._sigma_model(
                        entry.cs_type, entry.energy, entry.sigma_Mb, threshold,
                        zion=float(osc.z_screen),
                        nef=B._cmfgen_nef(16, float(osc.z_screen), threshold))
                    if model is not None:
                        table_ends.append(float(model[1][-1]))
                row_scale = max(float(np.max(abs(saved[i]))),
                                float(np.max(abs(values[i]))), 1e-300)
                row_diff = abs(values[i] - saved[i])
                relative.append(float(np.max(row_diff)) / row_scale)
                changed_bins = row_diff > 1e-8 * row_scale
                if table_ends:
                    # Saved rows used a constant high-frequency table tail. The
                    # corrected CMFGEN nu^-3 tail may differ only in bins whose
                    # upper edge extends beyond at least one raw table endpoint.
                    before_tail = nu_edges[1:] <= min(table_ends)
                    self.assertLessEqual(float(np.max(row_diff[before_tail], initial=0.0)) /
                                         row_scale, 1e-8)
                    self.assertFalse(np.any(changed_bins & before_tail))
                    tail_changed += int(np.any(changed_bins))
                else:
                    self.assertLessEqual(float(np.max(row_diff)) / row_scale, 1e-8)
            self.assertGreater(tail_changed, 0)
            self.assertTrue(np.all(np.max(values[~positive], axis=1) > 0))
            print(f"S_BAKE_REGRESSION=positive_rows:{int(positive.sum())},new_rows:{int((~positive).sum())},tail_rows:{tail_changed},max_rel:{max(relative):.17g}", flush=True)
            # SPLIT_J=True requires exact names, never implicit redistribution.
            term = "3s2_3p2(3Pe)6s_4Pe"
            entry = next(e for e in phot.entries if e.config == term)
            ion[(16, 2)]["phot"] = replace(phot, split_J=True, entries=[entry])
            with patch.object(B, "SIGMA_EXACT_HYD", True):
                B.bake_sigma_bf_grid(ion, {}, [None] * n, lookup, path)
            with path.open("rb") as f:
                f.seek(32)
                exact_flags = f.read(n)
            self.assertEqual([i for i, present in enumerate(exact_flags) if present], [121])

    def test_sulfur_headers_and_missing_route_inventory(self):
        # Inventory only: no S population or ionization-ratio claim is made.
        import pandas as pd
        import bake_cmfgen_sigma_bf_carsus as BA
        import det_si_trace_readout as TR
        from diagnose_si_routes_20261014 import entries_by_level
        run = Path("/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/model")
        levels = pd.read_csv(run / "levels.csv")
        levels["global_idx"] = np.arange(len(levels))
        manifest = pd.read_csv(run / "atomic_vintage_manifest.csv")
        with (run / "cmfgen_sigma_bf.bin").open("rb") as f:
            magic, version, nl, nf = struct.unpack("<IIii", f.read(16))
            f.read(16)
            flags = f.read(nl)
        self.assertEqual((magic, version, nl), (0x434D4644, 1, len(levels)))
        inventory = {}
        for ion, roman, target, g in ((1, "II", "3s2_3p2_3Pe", 9), (2, "III", "3s2_3p_2Po", 6)):
            raw = ROOT / "data/atomic/cmfgen/SUL" / roman / "19apr23"
            source = manifest[(manifest.atomic_number == 16) & (manifest.ion_number == ion)].iloc[0]
            # S III uses a recorded local correction, not the unpatched copy.
            osc_path, phot_path = Path(source.osc_path), Path(source.phot_path)
            osc, phot = P.parse_osc(osc_path), P.parse_phot(phot_path)
            self.assertEqual((phot.final_state, phot.final_state_g), (target, g))
            rows = levels[(levels.atomic_number == 16) & (levels.ion_number == ion)]
            mapping = BA.match_levels_to_cmfgen(rows, osc.levels)
            self.assertEqual(len(mapping), len(rows))
            entries = entries_by_level(phot, mapping)
            missing = []
            for r in rows.itertuples():
                if flags[r.global_idx]:
                    continue
                es = entries[r.global_idx]
                zero = bool(es) and all(e.cs_type == 1 and len(e.sigma_Mb) > 0 and
                                        np.all(e.sigma_Mb == 0.0) for e in es)
                raw_level = next(v for v in osc.levels if v["config"] == mapping[r.global_idx])
                edge = (osc.ionization_eV - float(raw_level["E_cm"]) * 1.239841984e-4) * BA.EV_TO_ERG / BA.H_CGS
                models = [B._sigma_model(e.cs_type, e.energy, e.sigma_Mb, edge,
                                        zion=osc.z_screen, nef=B._cmfgen_nef(16, osc.z_screen, edge)) for e in es]
                missing.append({"level": int(r.level_number), "config": mapping[r.global_idx],
                                "energy_eV": float(r.energy_eV),
                                "raw_types": [e.cs_type for e in es], "explicit_zero": zero,
                                "model_tags": [m[3] if m is not None else None for m in models],
                                "descending_steps": [int(np.count_nonzero(np.diff(e.energy) < 0)) for e in es],
                                "threshold_Hz": edge})
                if zero:
                    for e in es:
                        model = B._sigma_model(e.cs_type, e.energy, e.sigma_Mb, 1e15)
                        self.assertEqual(model[3], "explicit_zero")
                        np.testing.assert_array_equal(model[0](np.array([1e15, 2e15])), np.zeros(2))
            inventory[roman] = {"mapped_levels": len(mapping), "missing_saved_flag": missing,
                               "phot_path": str(phot_path), "phot_sha256": TR.sha(phot_path),
                               "repo_phot_same_bytes": TR.sha(phot_path) == TR.sha(raw / "phot_data_A")}
        print("S_ROUTE_INVENTORY=" + json.dumps(inventory, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
