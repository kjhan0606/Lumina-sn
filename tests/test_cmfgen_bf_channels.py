#!/usr/bin/env python3
"""회귀시험: Si/S CMFGEN 도착 채널과 총 연속 단면적 보존."""
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import deck_regen_r1_vintage_driver as D
import expand_atomic_data_cmfgen as B


class CmfgenBFChannelTests(unittest.TestCase):
    def test_model_spec_nf_joins_oscillators_in_link_order(self):
        with tempfile.TemporaryDirectory(prefix="cmfgen-model-spec-") as td:
            root = Path(td) / "atomic"
            links_path = Path(td) / "atomic_links.txt"
            model_spec_path = Path(td) / "MODEL_SPEC"
            sources = [
                root / "SUL/III/19apr23/osc_data",
                root / "COB/II/19apr23/osc_data",
            ]
            links_path.write_text("\n".join(
                f"ln -sf {source} {target}" for source, target in zip(
                    sources, ("SIII_F_OSCDAT", "Co2_F_OSCDAT"), strict=True)) + "\n")
            model_spec_path.write_text(
                "79,79,256 [SIII_ISF]\n134,134,2558 [Co2_ISF]\n")

            selected = B.load_cmfgen_model_spec(model_spec_path, links_path)

            self.assertEqual(selected[(16, 3)]["n_full"], 256)
            self.assertEqual(selected[(16, 3)]["n_super"], 79)
            self.assertEqual(selected[(27, 2)]["n_full"], 2558)
            self.assertEqual(selected[(27, 2)]["n_super"], 134)
            model_spec_path.write_text(
                "79,79,256 [SIII_ISF]\n134,134,2558 [CoIII_ISF]\n")
            with self.assertRaisesRegex(ValueError, "does not match oscillator link"):
                B.load_cmfgen_model_spec(model_spec_path, links_path)

    def test_parse_all_ions_uses_run_local_phot_overrides(self):
        with tempfile.TemporaryDirectory(prefix="cmfgen-phot-overrides-") as td:
            root = Path(td)
            atomic_root = root / "atomic_local"
            link_map = {}
            expected = {}
            for key, element_dir, roman, env_name in (
                    ((16, 3), "SUL", "III", "LUMINA_SIII_PHOT_PATH"),
                    ((27, 2), "COB", "II", "LUMINA_COII_PHOT_PATH")):
                date_dir = atomic_root / element_dir / roman / "19apr23"
                date_dir.mkdir(parents=True)
                osc_path = date_dir / "osc_data"
                default_phot = date_dir / "phot_data_A"
                col_path = date_dir / "col_data"
                ftos_path = date_dir / "f_to_s_test"
                override_phot = date_dir / "repaired_phot_data_A"
                for path in (osc_path, default_phot, col_path, ftos_path,
                             override_phot):
                    path.touch()
                link_map[key] = {"osc": osc_path, "phot": default_phot,
                                 "col": col_path, "f_to_s": ftos_path}
                expected[env_name] = override_phot.resolve()

            auto_only = atomic_root / "SUL/I/19apr23"
            auto_only.mkdir(parents=True)
            for name in ("osc_data", "phot_data_A", "col_data"):
                (auto_only / name).touch()

            levels = np.zeros(4, dtype=[("ID", "i4"), ("g", "f8"),
                                        ("E_cm", "f8"), ("config", "U16")])
            levels["ID"] = np.arange(1, 5)
            levels["g"] = 1.0
            levels["config"] = ["lev0", "lev1", "lev2", "lev3"]
            transitions = np.array([(2, 4, -500.0)],
                                   dtype=[("i", "i4"), ("j", "i4"),
                                          ("lam_A", "f8")])
            osc = SimpleNamespace(n_levels=4, levels=levels,
                                  transitions=transitions, n_transitions=1)
            parsed_phot_paths = []
            manifest_path = root / "atomic_vintage.csv"
            model_spec_map = {
                (16, 3): {"n_super": 2, "n_group": 2, "n_full": 3,
                          "label": "SIII"},
                (27, 2): {"n_super": 1, "n_group": 1, "n_full": 2,
                          "label": "Co2"},
            }
            level_sl_map = {
                ("SIII", 1): {"superlevel": 1, "configuration": "lev0"},
                ("SIII", 2): {"superlevel": 1, "configuration": "lev1"},
                ("SIII", 3): {"superlevel": 2, "configuration": "lev2"},
                ("Co2", 1): {"superlevel": 1, "configuration": "lev0"},
                ("Co2", 2): {"superlevel": 1, "configuration": "lev1"},
            }

            def fake_parse_ftos(_path):
                return SimpleNamespace(n_levels=4, n_super=2,
                                       sl_of_fl=np.array([0, 0, 1, 1]),
                                       format_name="fixture",
                                       format_basis="test mapping")

            def fake_parse_phot(path):
                parsed_phot_paths.append(Path(path).resolve())
                return SimpleNamespace(entries=[])

            with (patch.dict(os.environ, {name: str(path)
                                          for name, path in expected.items()}),
                  patch.object(B, "CMFGEN_ROOT", atomic_root),
                  patch.object(B, "CMFGEN_DIRS", {16: "SUL", 27: "COB"}),
                  patch.object(B, "CMFGEN_LINK_MAP", link_map),
                  patch.object(B, "ION_LEVEL_CAPS",
                               {(16, 1): 1, (16, 3): 1, (27, 2): 1}),
                  patch.object(B, "CMFGEN_MODEL_SPEC_PATH", root / "MODEL_SPEC"),
                  patch.object(B, "CMFGEN_MODEL_SPEC_MAP", model_spec_map),
                  patch.object(B, "CMFGEN_LEVEL_SL_STEQ_LINKS_MAP", level_sl_map),
                  patch.object(B, "LINK_FTOS_ENABLED", True),
                  patch.object(B, "SUPER_LEVEL_ENABLED", False),
                  patch.object(B, "parse_osc", return_value=osc),
                  patch.object(B, "parse_f_to_s", side_effect=fake_parse_ftos),
                  patch.object(B, "parse_phot", side_effect=fake_parse_phot),
                  patch.object(B, "parse_col",
                               return_value=SimpleNamespace(entries=[]))):
                parsed = B.parse_all_ions()
                B.write_atomic_vintage_manifest(parsed, manifest_path)

            self.assertEqual(set(parsed_phot_paths), set(expected.values()))
            self.assertNotIn((16, 1), parsed,
                             "auto-discovered ions must not extend a sealed MODEL_SPEC")
            self.assertEqual(parsed[(16, 3)]["provenance"]["phot_path"],
                             expected["LUMINA_SIII_PHOT_PATH"])
            self.assertEqual(parsed[(27, 2)]["provenance"]["phot_path"],
                             expected["LUMINA_COII_PHOT_PATH"])
            self.assertEqual(parsed[(16, 3)]["provenance"]["phot_override_env"],
                             "LUMINA_SIII_PHOT_PATH")
            self.assertEqual(parsed[(27, 2)]["provenance"]["phot_override_env"],
                             "LUMINA_COII_PHOT_PATH")
            self.assertEqual(parsed[(16, 3)]["n_kept"], 3)
            self.assertEqual(parsed[(27, 2)]["n_kept"], 2)
            self.assertEqual(parsed[(16, 3)]["provenance"]["active_superlevels"], 2)
            self.assertEqual(parsed[(27, 2)]["provenance"]["active_superlevels"], 1)
            self.assertEqual(len(parsed[(16, 3)]["trans"]), 0)
            manifest = manifest_path.read_text()
            self.assertIn("phot_override_env", manifest)
            self.assertIn("LUMINA_SIII_PHOT_PATH", manifest)
            self.assertIn("LUMINA_COII_PHOT_PATH", manifest)
            self.assertIn("model_spec_nf", manifest)
            self.assertIn(",3,2,", manifest)

    def test_deck_source_identity_hashes_linked_files_and_replaces_old_provenance(self):
        with tempfile.TemporaryDirectory(prefix="cmfgen-source-identity-") as td:
            root = Path(td)
            sources = {}
            for kind in ("osc", "f_to_s", "phot", "col"):
                path = root / kind
                path.write_bytes((kind + "-payload").encode())
                sources[kind] = path
            manifest_path = root / "atomic_vintage_manifest.csv"
            fields = ["atomic_number", "ion_stage", "ion", "selection_source",
                      "model_spec_nf", "active_superlevels", "osc_path",
                      "f_to_s_path", "phot_path", "col_path"]
            with manifest_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "atomic_number": 16, "ion_stage": 3, "ion": "S III",
                    "selection_source": "links", "model_spec_nf": 256,
                    "active_superlevels": 79,
                    **{f"{kind}_path": str(path)
                       for kind, path in sources.items()},
                })
            identity_path = root / "atomic_source_identity.json"
            identity = D.write_atomic_source_identity(manifest_path, identity_path)
            self.assertEqual(identity["n_active_ions"], 1)
            self.assertEqual(identity["n_unique_source_files"], 4)
            for kind, path in sources.items():
                self.assertEqual(
                    identity["ions"][0]["sources"][kind]["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertIn("DECK_PROVENANCE.json", D.REBUILT)

            links = root / "atomic_links.txt"
            model_spec = root / "MODEL_SPEC"
            level_map = root / "LEVEL_SL_STEQ_LINKS"
            override = root / "repaired_phot"
            links.write_text("links\n")
            model_spec.write_text("MODEL_SPEC\n")
            level_map.write_text("LEVEL_MAP\n")
            override.write_text("override\n")
            output = root / "deck"
            output.mkdir()
            (output / "atomic_source_identity.json").write_bytes(
                identity_path.read_bytes())
            source_identity = dict(identity)
            smoothing_library = root / "smoothing.so"
            smoothing_library.write_bytes(b"compiled-CMFGEN-smoothing-fixture")
            smoothing = SimpleNamespace(
                library=smoothing_library,
                sources=[{"path": str(sources["osc"])}],
                provenance=lambda: {"library": {"path": str(smoothing_library)}})
            D_NEW, D_OLD = D.NEW, D.OLD
            try:
                D.NEW = output
                D.OLD = root / "template"
                D.write_deck_provenance(
                    type("Module", (), {
                        "ATOMIC_LOCAL_OVERRIDES": [
                            {"path": str(override),
                             "tail": "SUL/III/phot_data_A"}
                        ],
                        "HYD_BF_DATA_FILES": [sources["osc"], sources["phot"]],
                        "PHOT_SMOOTHER": smoothing,
                    })(),
                    links, model_spec, level_map, 50, source_identity)
            finally:
                D.NEW, D.OLD = D_NEW, D_OLD
            stamp = json.loads((output / "DECK_PROVENANCE.json").read_text())
            self.assertEqual(stamp["schema"], "lumina-deck-provenance-v1")
            self.assertFalse(stamp["finalized"])
            self.assertEqual(stamp["line_einstein_convention"], "J_nu")
            self.assertEqual(stamp["cmfgen_links_sha256"],
                             hashlib.sha256(links.read_bytes()).hexdigest())
            self.assertEqual(stamp["active_ions"]["S III"],
                             {"model_spec_nf": 256, "active_superlevels": 79})
            self.assertEqual(stamp["atomic_local_overrides"][0]["sha256"],
                             hashlib.sha256(override.read_bytes()).hexdigest())
            self.assertEqual(
                [row["sha256"] for row in stamp["hydrogenic_fit_data"]],
                [hashlib.sha256(sources[k].read_bytes()).hexdigest()
                 for k in ("osc", "phot")])
            self.assertEqual(stamp["env"]["CMFGEN_EXACT_HYD"],
                             os.environ.get("CMFGEN_EXACT_HYD", ""))
            copied_library = output / "cmfgen_phot_smoothing.so"
            self.assertEqual(copied_library.read_bytes(), smoothing_library.read_bytes())
            self.assertEqual(stamp["phot_preprocessing"]["library"], {
                "path": str(copied_library),
                "sha256": hashlib.sha256(copied_library.read_bytes()).hexdigest()})

    def test_exact_hyd_mode_rejects_unusable_fit_without_standin(self):
        with (patch.object(B, "SIGMA_EXACT_HYD", True),
              patch.object(B, "_sigma_hydrogenic_exact", return_value=None)):
            with self.assertRaisesRegex(ValueError, "stand-in fallback forbidden"):
                B._sigma_model(2, np.array([]), np.array([1.0, 0.0, 0.0]),
                               1.0e15, zion=1.0, nef=1.0)

    def test_exact_hyd_type2_ground_state_matches_known_answer(self):
        edge = B._NU_INF_INF * 1.0e15 / (1.0 + 5.48597e-4)
        nef = B._cmfgen_nef(1, 1.0, edge)
        with patch.object(B, "SIGMA_EXACT_HYD", True):
            model = B._sigma_model(
                2, np.array([]), np.array([1.0, 0.0, 0.0]), edge,
                zion=1.0, nef=nef)
        self.assertEqual(model[3], "hyd2")
        sigma = float(model[0](np.array([edge]))[0])
        self.assertAlmostEqual(sigma / 6.3034e-18, 1.0, places=4)

    def test_phot_override_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="cmfgen-missing-phot-") as td:
            missing = Path(td) / "missing_phot_data_A"
            with patch.dict(os.environ,
                            {"LUMINA_SIII_PHOT_PATH": str(missing)}):
                with self.assertRaisesRegex(FileNotFoundError,
                                            "LUMINA_SIII_PHOT_PATH"):
                    B._apply_phot_path_override((16, 3), Path("default_phot"))

    def test_tabulated_tail_matches_cmfgen_nu_minus_three(self):
        model = B._sigma_model(20, np.array([1.0, 2.0]),
                               np.array([4.0, 2.0]), 1.0e15)
        self.assertIsNotNone(model)
        fn, _, _, _ = model
        values = fn(np.array([0.9e15, 1.5e15, 4.0e15]))
        np.testing.assert_allclose(values, [0.0, 3.0e-18, 2.5e-19], rtol=1e-14, atol=0)

    def test_route_bake_matches_cmfgen_superlevel_target_and_alias_order(self):
        with tempfile.TemporaryDirectory(prefix="cmfgen-bf-routes-") as td:
            root = Path(td)
            phot_a_path = root / "phot_data_A"
            phot_b_path = root / "phot_data_B"
            phot_a_path.touch()
            phot_b_path.touch()

            def entry(scale):
                return SimpleNamespace(config="lower", cs_type=20,
                                       energy=np.array([1.0, 2.0]),
                                       sigma_Mb=np.array([scale]))

            phot_a = SimpleNamespace(n_routes=2, final_state="ground",
                                     final_state_g=1.0, split_J=False,
                                     entries=[entry(1.0)])
            phot_b = SimpleNamespace(n_routes=2, final_state="triplet/other",
                                     final_state_g=4.0, split_J=False,
                                     entries=[entry(2.0)])
            lower = {
                "osc": SimpleNamespace(ionization_eV=10.0, z_screen=2.0),
                "levels": {"config": np.array(["lower[0]"]),
                           "E_cm": np.array([0.0]), "g": np.array([2.0])},
                "n_kept": 1,
                "phot": phot_a,
                "provenance": {"phot_path": phot_a_path},
            }
            upper = {
                "osc": SimpleNamespace(ionization_eV=20.0, z_screen=2.0),
                "levels": {"config": np.array(["ground[0]", "triplet[0]",
                                                "triplet[1]", "other[0]"]),
                           "E_cm": np.array([0.0, 8065.5439, 16131.0878,
                                             24196.6317]),
                           "g": np.array([1.0, 1.0, 3.0, 5.0])},
                "n_kept": 4,
                "ftos": SimpleNamespace(sl_of_fl=np.array([0, 1, 1, 2])),
                "phot": None,
                "provenance": {"phot_path": None},
            }
            ion_data = {(14, 2): lower, (14, 3): upper}
            level_lookup = {(14, 2, 1): 0, (14, 3, 1): 1,
                            (14, 3, 2): 2, (14, 3, 3): 3,
                            (14, 3, 4): 4}
            sigma_path = root / "cmfgen_sigma_bf.bin"
            routes_path = root / "cmfgen_bf_channels.bin"

            def sigma_model(_type, _energy, params, nu_th, **_kwargs):
                value = float(params[0]) * 1.0e-18
                return (lambda nu, v=value: np.full(np.shape(nu), v)), \
                       np.array([nu_th]), nu_th, "tab"

            with (patch.object(B, "BF_CHANNEL_IONS", {(14, 2)}),
                  patch.object(B, "BF_NU_MIN", 1.0e15),
                  patch.object(B, "BF_NU_MAX", 5.0e15),
                  patch.object(B, "BF_N_FREQ_BIN", 4),
                  patch.object(B, "parse_phot", side_effect=lambda p:
                               phot_a if Path(p).name == "phot_data_A" else phot_b),
                  patch.object(B, "_sigma_model", side_effect=sigma_model),
                  patch.object(B, "_bin_average_sigma",
                               side_effect=lambda fn, _nodes, _edges, widths, _start:
                               np.full(len(widths), float(fn(np.array([2.0e15]))[0])))):
                n_baked, n_covered = B.bake_sigma_bf_grid(
                    ion_data, ion_data, [None] * 5, level_lookup,
                    sigma_path, route_path=routes_path)
                model_path = root / "MODEL"
                model_path.write_text("F,F [DIE_test_lower]\nF,F [DIE_test_upper]\n")
                policy_models = {(14, 2): {"label": "test_lower"},
                                 (14, 3): {"label": "test_upper"}}
                policy_links = {key: {"phot": phot_a_path} for key in policy_models}
                v5_path = root / "v5_channels.bin"
                B.bake_sigma_bf_grid(ion_data, ion_data, [None]*5, level_lookup,
                    root / "v5_sigma.bin", route_path=v5_path,
                    policy_context=(model_path, policy_models, policy_links))
                v5_manifest = json.loads((root / "v5_channels_manifest.json").read_text())
                self.assertEqual(v5_manifest["schema"], "lumina-cmfgen-bf-channels-v5")
                self.assertEqual(len(v5_manifest["recombination_policy"]["ions"]), 2)
                original = routes_path.read_bytes()
                self.assertEqual(v5_path.read_bytes()[8:len(original)], original[8:])
                self.assertEqual((root / "v5_sigma.bin").read_bytes(), sigma_path.read_bytes())

            self.assertEqual((n_baked, n_covered), (2, 1))
            with routes_path.open("rb") as stream:
                magic, version, n_levels, n_bins, n_routes, nu_min, nu_max = \
                    struct.unpack("<IIiiidd", stream.read(36))
                offsets = np.frombuffer(stream.read(4 * (n_levels + 1)), dtype="<i4")
                targets = np.frombuffer(stream.read(4 * n_routes), dtype="<i4")
                probs = np.frombuffer(stream.read(8 * n_routes), dtype="<f8")
                grids = np.frombuffer(stream.read(4 * n_routes), dtype="<i4")
                target_energy = np.frombuffer(stream.read(8 * n_routes), dtype="<f8")
                target_g = np.frombuffer(stream.read(8 * n_routes), dtype="<f8")
                edge_sigma = np.frombuffer(stream.read(8 * n_routes), dtype="<f8")
                route_sigma = np.frombuffer(stream.read(8 * n_routes * n_bins),
                                             dtype="<f8").reshape(n_routes, n_bins)
            self.assertEqual((magic, version, n_levels, n_bins, n_routes),
                             (B.BF_CHANNEL_MAGIC, 4, 5, 4, 2))
            self.assertEqual((nu_min, nu_max), (1.0e15, 5.0e15))
            np.testing.assert_array_equal(offsets, [0, 2, 2, 2, 2, 2])
            np.testing.assert_array_equal(targets, [1, 2])
            np.testing.assert_allclose(probs, [1.0, 1.0], rtol=0, atol=1e-15)
            np.testing.assert_array_equal(grids, [0, 1])
            np.testing.assert_allclose(target_energy, [0.0, 1.0])
            np.testing.assert_allclose(target_g, [1.0, 4.0])
            np.testing.assert_allclose(edge_sigma, [1e-18, 2e-18], rtol=1e-14)
            np.testing.assert_allclose(route_sigma[:, 0], [1e-18, 2e-18], rtol=1e-14)

            with sigma_path.open("rb") as stream:
                stream.seek(32 + n_levels + (-n_levels) % 8)
                sigma_total = np.frombuffer(stream.read(), dtype="<f8").reshape(n_levels, n_bins)
            np.testing.assert_allclose(sigma_total[0], np.full(n_bins, 3e-18), rtol=1e-14)
            manifest = json.loads((root / "cmfgen_bf_channels_manifest.json").read_text())
            self.assertEqual(manifest["schema"], "lumina-cmfgen-bf-channels-v4")
            self.assertEqual(manifest["n_channels"], 2)
            self.assertEqual(len(manifest["ions"]["Z14_stage2"]["sources"]), 2)

    def test_terminal_phot_target_is_preserved_without_inventing_a_level(self):
        with tempfile.TemporaryDirectory(prefix="cmfgen-bf-terminal-") as td:
            root = Path(td)
            phot_path = root / "phot_data_A"
            phot_path.touch()

            for Z, stage, final_state, g in (
                    (14, 5, "2s2_2p5_2Po", 6.0),
                    (16, 5, "2p6_3s_2Se/3s_2Se", 2.0)):
                phot = SimpleNamespace(
                    n_routes=1, final_state=final_state,
                    final_state_excit_eV=0.0, final_state_g=g,
                    split_J=False,
                    entries=[SimpleNamespace(config="lower", cs_type=20,
                                             energy=np.array([1.0, 2.0]),
                                             sigma_Mb=np.array([3.0]))])
                lower = {
                    "osc": SimpleNamespace(ionization_eV=10.0, z_screen=5.0),
                    "levels": {"config": np.array(["lower[0]"]),
                               "E_cm": np.array([0.0]), "g": np.array([2.0])},
                    "n_kept": 1, "phot": phot,
                    "provenance": {"phot_path": phot_path},
                }
                sigma_path = root / f"sigma-{Z}.bin"
                routes_path = root / f"routes-{Z}.bin"

                def sigma_model(_type, _energy, params, nu_th, **_kwargs):
                    value = float(params[0]) * 1.0e-18
                    return (lambda nu, v=value, edge=nu_th:
                            v * np.asarray(nu, dtype="f8") / edge), \
                           np.array([nu_th]), nu_th, "tab"

                with (patch.object(B, "BF_CHANNEL_IONS", {(Z, stage)}),
                      patch.object(B, "BF_NU_MIN", 1.0e15),
                      patch.object(B, "BF_NU_MAX", 5.0e15),
                      patch.object(B, "BF_N_FREQ_BIN", 4),
                      patch.object(B, "parse_phot", return_value=phot),
                      patch.object(B, "_sigma_model", side_effect=sigma_model),
                      patch.object(B, "_bin_average_sigma",
                                   side_effect=lambda fn, _nodes, _edges, widths, _start:
                                   np.full(len(widths), float(fn(np.array([2.0e15]))[0])))):
                    baked, covered = B.bake_sigma_bf_grid(
                        {(Z, stage): lower}, {(Z, stage): lower}, [None],
                        {(Z, stage, 1): 0}, sigma_path, route_path=routes_path)

                self.assertEqual((baked, covered), (1, 1))
                with routes_path.open("rb") as stream:
                    magic, version, n_levels, n_bins, n_routes, _, _ = \
                        struct.unpack("<IIiiidd", stream.read(36))
                    offsets = np.frombuffer(stream.read(8), dtype="<i4")
                    targets = np.frombuffer(stream.read(4), dtype="<i4")
                    probability = np.frombuffer(stream.read(8), dtype="<f8")
                    phot_grid = np.frombuffer(stream.read(4), dtype="<i4")
                    target_energy = np.frombuffer(stream.read(8), dtype="<f8")
                    target_g = np.frombuffer(stream.read(8), dtype="<f8")
                    edge_sigma = np.frombuffer(stream.read(8), dtype="<f8")
                self.assertEqual((magic, version, n_levels, n_bins, n_routes),
                                 (B.BF_CHANNEL_MAGIC, 4, 1, 4, 1))
                np.testing.assert_array_equal(offsets, [0, 1])
                np.testing.assert_array_equal(targets, [-1])
                np.testing.assert_allclose(probability, [1.0])
                np.testing.assert_array_equal(phot_grid, [0])
                np.testing.assert_allclose(target_energy, [0.0])
                np.testing.assert_allclose(target_g, [g])
                np.testing.assert_allclose(edge_sigma, [3.0e-18 * (1.0 + 1.0e-14)],
                                           rtol=1e-14)

    def test_terminal_phot_target_rejects_unsupported_multiroute_or_weight(self):
        for routes, g in ((2, 6.0), (1, 0.0)):
            phot = SimpleNamespace(n_routes=routes, final_state="ground",
                                   final_state_excit_eV=0.0,
                                   final_state_g=g)
            with self.assertRaises(ValueError):
                B._phot_terminal_target_component(phot, 14, 6)

    def test_selected_real_si_s_grids_resolve_to_cmfgen_superlevels(self):
        # Exercise the exact sealed CMFGEN atom: links select raw sources,
        # MODEL_SPEC selects active NF, and the linked F_TO_S prefix must
        # reproduce its active superlevel cardinality.
        if B.CMFGEN_LINKS_PATH is None or B.CMFGEN_MODEL_SPEC_PATH is None:
            self.skipTest("set CMFGEN_LINKS and CMFGEN_MODEL_SPEC to the sealed CMFGEN run")
        if not os.environ.get("LUMINA_SIII_PHOT_PATH") or \
                not os.environ.get("LUMINA_COII_PHOT_PATH"):
            self.skipTest("set both sealed run-local repaired PHOT paths")
        route_keys = {(14, 2), (14, 3), (14, 4), (14, 5),
                      (16, 2), (16, 3), (16, 4), (16, 5)}
        selected_keys = route_keys | {(27, 2)}
        with patch.object(B, "ION_LEVEL_CAPS", {key: None for key in selected_keys}):
            parsed = B.parse_all_ions()
        ion_data = {key: parsed[key] for key in route_keys}

        for key in selected_keys:
            self.assertEqual(parsed[key]["n_kept"],
                             B.CMFGEN_MODEL_SPEC_MAP[key]["n_full"])
            self.assertEqual(parsed[key]["provenance"]["active_superlevels"],
                             B.CMFGEN_MODEL_SPEC_MAP[key]["n_super"])
        for key, env_name in (((16, 3), "LUMINA_SIII_PHOT_PATH"),
                              ((27, 2), "LUMINA_COII_PHOT_PATH")):
            selected = parsed[key]["provenance"]["phot_path"]
            self.assertEqual(selected, Path(os.environ[env_name]).resolve())
            self.assertEqual(selected, B.CMFGEN_LINK_MAP[key]["phot"].resolve())

        level_lookup = {}
        level_owner = []
        for (Z, stage), data in sorted(ion_data.items()):
            for i in range(data["n_kept"]):
                level_lookup[(Z, stage, i + 1)] = len(level_owner)
                level_owner.append((Z, stage))
        with tempfile.TemporaryDirectory(prefix="real-bf-routes-") as td:
            root = Path(td)
            sigma_path = root / "cmfgen_sigma_bf.bin"
            routes_path = root / "cmfgen_bf_channels.bin"
            with patch.object(B, "SIGMA_EXACT_HYD", True):
                B.bake_sigma_bf_grid(ion_data, ion_data, [None] * len(level_owner),
                                     level_lookup, sigma_path, route_path=routes_path)

            with routes_path.open("rb") as stream:
                magic, version, n_levels, n_bins, n_routes, nu_min, nu_max = \
                    struct.unpack("<IIiiidd", stream.read(36))
                offsets = np.frombuffer(stream.read(4 * (n_levels + 1)), dtype="<i4")
                targets = np.frombuffer(stream.read(4 * n_routes), dtype="<i4")
                probabilities = np.frombuffer(stream.read(8 * n_routes), dtype="<f8")
                phot_grids = np.frombuffer(stream.read(4 * n_routes), dtype="<i4")
                target_energy = np.frombuffer(stream.read(8 * n_routes), dtype="<f8")
                target_g = np.frombuffer(stream.read(8 * n_routes), dtype="<f8")
                edge_sigma = np.frombuffer(stream.read(8 * n_routes), dtype="<f8")
                route_sigma = np.frombuffer(stream.read(8 * n_routes * n_bins),
                                            dtype="<f8").reshape(n_routes, n_bins)
            self.assertEqual((magic, version, n_levels, n_bins),
                             (B.BF_CHANNEL_MAGIC, 4, len(level_owner), B.BF_N_FREQ_BIN))
            self.assertEqual((nu_min, nu_max), (B.BF_NU_MIN, B.BF_NU_MAX))
            self.assertEqual(edge_sigma.shape, (n_routes,))
            self.assertTrue(np.all(np.isfinite(edge_sigma) & (edge_sigma >= 0.0)))
            self.assertGreater(np.count_nonzero(edge_sigma), 0)
            self.assertEqual(int(offsets[-1]), n_routes)
            manifest = json.loads(routes_path.with_name(
                "cmfgen_bf_channels_manifest.json").read_text())
            for lower, (Z, stage) in enumerate(level_owner):
                begin, end = int(offsets[lower]), int(offsets[lower + 1])
                if (Z, stage) not in B.BF_CHANNEL_IONS:
                    continue
                self.assertGreater(end, begin,
                                   f"CMFGEN PHOT route omitted active lower level {Z}:{stage}:{lower}")
                expected_grids = {
                    int(row["phot_grid"])
                    for row in manifest["ions"][f"Z{Z:02d}_stage{stage}"]["channels_by_grid"]
                }
                self.assertEqual(set(map(int, phot_grids[begin:end])), expected_grids,
                                 f"PHOT grid coverage mismatch for {Z}:{stage}:{lower}")
                for route in range(begin, end):
                    target = int(targets[route])
                    if (Z, stage) in B.BF_TERMINAL_IONS:
                        self.assertEqual(target, -1)
                        self.assertEqual(float(target_energy[route]),
                                         parsed[(Z, stage)]["phot"].final_state_excit_eV)
                        self.assertEqual(float(target_g[route]),
                                         parsed[(Z, stage)]["phot"].final_state_g)
                    else:
                        self.assertEqual(level_owner[target], (Z, stage + 1))
                    self.assertGreater(probabilities[route], 0.0)
                for grid in np.unique(phot_grids[begin:end]):
                    mask = phot_grids[begin:end] == grid
                    self.assertAlmostEqual(float(probabilities[begin:end][mask].sum()),
                                           1.0, places=12)
                    upper_superlevels = []
                    upper_ftos = parsed.get((Z, stage + 1), {}).get("ftos")
                    for route in range(begin, end):
                        if int(phot_grids[route]) != int(grid) or targets[route] < 0:
                            continue
                        upper_local = int(targets[route]) - level_lookup[(Z, stage + 1, 1)]
                        upper_superlevels.append(
                            int(upper_ftos.sl_of_fl[upper_local]) if upper_ftos is not None
                            else upper_local)
                    self.assertEqual(len(upper_superlevels), len(set(upper_superlevels)),
                                     f"duplicate target superlevel for {Z}:{stage}:{grid}")

            si2_rows = [l for l in range(len(level_owner))
                        if level_owner[l] == (14, 2) and
                        np.count_nonzero(phot_grids[offsets[l]:offsets[l + 1]] == 1) == 1]
            self.assertTrue(si2_rows, "Si II phot_data_B did not resolve one CMFGEN 3Po superlevel route")
            l = si2_rows[0]
            begin, end = int(offsets[l]), int(offsets[l + 1])
            b_route = begin + int(np.flatnonzero(phot_grids[begin:end] == 1)[0])
            self.assertEqual(int(targets[b_route]), level_lookup[(14, 3, 2)])
            self.assertEqual(float(target_g[b_route]), 9.0)
            self.assertEqual(float(probabilities[b_route]), 1.0)

            for lower in range(len(level_owner)):
                if level_owner[lower] != (16, 2):
                    continue
                begin, end = int(offsets[lower]), int(offsets[lower + 1])
                self.assertEqual(end - begin, 1,
                                 "S II -> S III ground term must be one CMFGEN SL route")
                self.assertEqual(int(targets[begin]), level_lookup[(16, 3, 1)])
                self.assertEqual(float(target_g[begin]), 9.0)

            for lower in range(len(level_owner)):
                if level_owner[lower] != (16, 3):
                    continue
                begin, end = int(offsets[lower]), int(offsets[lower + 1])
                self.assertEqual(end - begin, 1,
                                 "S III -> S IV alternate final states must coalesce by CMFGEN SL")
                self.assertEqual(float(target_g[begin]), 6.0)

            with sigma_path.open("rb") as stream:
                stream.seek(32 + n_levels + (-n_levels) % 8)
                total_sigma = np.frombuffer(stream.read(), dtype="<f8").reshape(n_levels, n_bins)
            for lower in range(n_levels):
                begin, end = int(offsets[lower]), int(offsets[lower + 1])
                if begin == end:
                    continue
                weighted = np.sum(route_sigma[begin:end] *
                                  probabilities[begin:end, None], axis=0)
                np.testing.assert_allclose(weighted, total_sigma[lower], rtol=1e-13, atol=1e-300)

            self.assertEqual(set(manifest["ions"]),
                             {"Z14_stage2", "Z14_stage3", "Z14_stage5",
                              "Z16_stage2", "Z16_stage3", "Z16_stage5"})
            self.assertEqual([Path(x["path"]).name for x in
                              manifest["ions"]["Z14_stage2"]["sources"]],
                             ["phot_data_A", "phot_data_B"])
            self.assertTrue(all(len(x["sha256"]) == 64 for ion in manifest["ions"].values()
                                for x in ion["sources"]))


if __name__ == "__main__":
    unittest.main()
