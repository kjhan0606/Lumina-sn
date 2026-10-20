import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cmfgen_recombination_policy import build_policy, encode_policy, validate_tail
from compare_cmfgen_bf_superlevel_rates import parse_cmfgen_bf_channels
from summarize_cmfgen_recombination_response import parse_native_se_tail


def fixture(tmp_path):
    model, phot = tmp_path / "MODEL", tmp_path / "phot_data_A"
    model.write_text("F,F [DIE_run_label]\nF,F [DIE_run_iv]\n")
    phot.write_bytes(b"sealed phot source")
    models = {(14, 2): dict(label="run_label"), (14, 4): dict(label="run_iv")}
    links = {key: dict(phot=phot) for key in models}
    policy = build_policy(model, models, links, {}, models)
    return model, phot, models, links, policy


def test_boundary_identity_from_model_and_non_channel_IV(tmp_path):
    model, phot, _, _, policy = fixture(tmp_path)
    assert policy["model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert [(r["lower_ion_number"], r["upper_ion_number"]) for r in policy["ions"]] == [(1, 2), (3, 4)]
    assert [r["model_key"] for r in policy["ions"]] == ["DIE_run_label", "DIE_run_iv"]
    assert policy["ions"][0]["phot_sources"][0]["sha256"] == hashlib.sha256(phot.read_bytes()).hexdigest()
    raw = encode_policy(policy)
    assert len(raw) == 44 + 2*56
    validate_tail(raw, policy)
    with pytest.raises(ValueError):
        validate_tail(raw[:-1], policy)
    with pytest.raises(ValueError):
        validate_tail(raw, None)


@pytest.mark.parametrize("value", ["T,F", "F,T", "T,T", "", "F,F [DIE_run_label]\nF,F"])
def test_enabled_missing_duplicate_source_selection_rejected(tmp_path, value):
    model, _, models, links, _ = fixture(tmp_path)
    model.write_text(value + " [DIE_run_label]\nF,F [DIE_run_iv]\n")
    with pytest.raises(ValueError):
        build_policy(model, models, links, {}, models)


@pytest.mark.parametrize("bad", ["auto", "allow", "upper", "duplicate", "hash", "schema"])
def test_policy_cannot_silently_enable_or_lose_provenance(tmp_path, bad):
    *_, policy = fixture(tmp_path)
    modified = copy.deepcopy(policy)
    if bad == "auto": modified["ions"][0]["die_auto"] = 1
    if bad == "allow": modified["ions"][0]["external_dr_fit_allowed"] = 1
    if bad == "upper": modified["ions"][0]["upper_ion_number"] = 3
    if bad == "duplicate": modified["ions"][1] = modified["ions"][0]
    if bad == "hash": modified["ions"][0]["phot_sources_sha256"] = "a"*64
    if bad == "schema": modified["schema"] = "unknown"
    with pytest.raises(ValueError):
        encode_policy(modified)


def test_v5_rate_reader_requires_matching_policy_tail(tmp_path):
    *_, policy = fixture(tmp_path)
    path = tmp_path / "channels.bin"
    raw = bytearray(struct.pack("<IIiiidd", 0x434D4643, 5, 1, 1, 1, 1e15, 2e15))
    raw += struct.pack("<2ii di ddd d", 0, 1, -1, 1., 0, 0., 6., 1e-18, 2e-18)
    raw += encode_policy(policy)
    manifest = dict(schema="lumina-cmfgen-bf-channels-v5", n_levels=1, n_freq_bins=1,
                    n_channels=1, recombination_policy=policy)
    path.write_bytes(raw)
    parsed = parse_cmfgen_bf_channels(path, 1, 1, manifest)
    assert parsed["sigma"].item() == 2e-18
    bad_manifest = dict(manifest)
    del bad_manifest["recombination_policy"]
    with pytest.raises(ValueError):
        parse_cmfgen_bf_channels(path, 1, 1, bad_manifest)


def test_native_se_reader_uses_last_epoch_bound_rows_not_extra_equations():
    models = {(14,3):dict(label="SkIII",n_super=2)}
    def block(a,b):
        return ("\n STEQ ARRAY\n\n" + f" 1** {a} {a}\n 2 * {b} {b}\n"
                " 3 * 9.0000E+20 9.0000E+20\n 4 * 8.0000E+20 8.0000E+20\n\n CORRECTIONS\n")
    text = block("9.0000E+00","9.0000E+00")+block("1.2000E+01","-2.0000E+00")
    result = parse_native_se_tail(text,models,2,[(14,3,2)])[0]
    assert result["bound_SE_sum"] == 10
    assert result["bound_SE_sum_abs"] == 14
    assert result["decimal_print_rounding_sum_bound"] == pytest.approx(.00055)
    with pytest.raises(ValueError,match="block"):
        parse_native_se_tail(text,models,20,[(14,3,2)])
