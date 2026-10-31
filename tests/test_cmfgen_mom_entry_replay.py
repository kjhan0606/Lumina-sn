import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_cmfgen_mom_capture as analysis


PHASE = (78, 1234, 4, 0, 1)
TARGET = dict(frequency_native=9.39898883, outer_iteration=78,
              frequency_absolute_tolerance=5e-9)


def _array(values):
    return {"values": np.asarray(values, dtype=float)}


def _states(h_return=(0.0, 0.0, 0.0), formal_j=(1.0, 1.0, 1.0)):
    formal = dict(
        HN_DEF_ON_NODES=1,
        ND_STORE=3,
        JNU_STORE=_array(formal_j),
        HNU_STORE=_array((0.0, 0.0, 0.0)),
        KNU_STORE=_array((1.0/3.0,)*3),
        NNU_STORE=_array((0.0, 0.0, 0.0)),
        KMID_ON_J_SAVE=_array((1.0/6.0,)*3),
    )
    entry = dict(phase=PHASE,
        args=dict(FREQ=TARGET["frequency_native"], INNER_BND_METH="DIFFUSION"),
        modules=dict(ray=formal))
    final = dict(phase=PHASE,
        modules=dict(
            jrel=dict(
                FREQ_SAVE=TARGET["frequency_native"],
                GAM_RSQJNU_SAVE=_array((1.0, 1.0, 1.0)),
                GAM_RSQHNU_SAVE=_array(h_return),
            ),
            ray=formal.copy()))
    return entry, final


def _call():
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    raw = np.ones(3)
    return dict(phase=PHASE, frequency=TARGET["frequency_native"],
        iterations={1: dict(ND=3, TRI0=tri, RAW0=raw,
                            ERR0=(1e-9, (raw.copy(), raw.copy(), raw.copy())))})


def _trace_flux_build(build, *, trace_flux=True, entry_only=False):
    build.mkdir()
    parent = build.parent / (build.name + "-parent")
    parent.mkdir()
    parent_identity = parent / "parent-build-identity.json"
    parent_identity.write_text("{}")
    inventory = build / "state_inventory.json"
    inventory.write_text("{}")
    replay_exe = build / "replay_mom.exe"
    replay_exe.write_bytes(b"replay executable")
    source = build / "source.f"
    source.write_text("source")
    generator = Path(analysis.__file__).with_name("prepare_cmfgen_mom_capture.py")
    record = dict(capture_target=TARGET, trace_flux=trace_flux,
        entry_only=entry_only, first_repair=False, parent=str(parent),
        parent_manifest_sha256=analysis.sha(parent_identity),
        generator_sha256=analysis.sha(generator),
        input_sha256={str(source.resolve()): analysis.sha(source)},
        output_sha256={str(replay_exe.resolve()): analysis.sha(replay_exe),
                       str(inventory.resolve()): analysis.sha(inventory)})
    (build / "capture-build.json").write_text(json.dumps(record))
    return record


def test_entry_replay_reports_formal_support_and_returned_boundary_condition():
    entry, final = _states()
    result = analysis.entry_replay_summary(entry, final, _call(), TARGET)
    assert result["formal_angular_support"]["available"]
    assert result["formal_angular_support"]["minimum"] > 0.0
    assert result["returned_mom_conservative_support"]["max_ratio"] == 0.0
    assert result["returned_mom_conservative_support"]["violating_edges_1based"] == []
    assert result["returned_mom_conservative_support"]["final_J_matches_last_trace"]
    assert result["replay_iterations"][0]["linear_residual_before"] == 0.0


def test_entry_replay_reports_unrealizable_return_without_calling_it_a_gate():
    entry, final = _states(h_return=(1.0, 0.0, 0.0))
    result = analysis.entry_replay_summary(entry, final, _call(), TARGET)
    assert result["returned_mom_conservative_support"]["max_ratio"] == pytest.approx(3.0)
    assert result["returned_mom_conservative_support"]["violating_edges_1based"] == [1]
    assert result["final_goal"] == "NOT_JUDGED"


def test_entry_replay_rejects_wrong_target_frequency():
    entry, final = _states()
    call = _call()
    call["frequency"] += 1e-7
    with pytest.raises(ValueError, match="phase 또는 주파수"):
        analysis.entry_replay_summary(entry, final, call, TARGET)


def test_entry_only_file_analysis_accepts_missing_pre_call_trace(tmp_path, monkeypatch):
    run, replay, build = (tmp_path / name for name in ("run", "replay", "build"))
    run.mkdir(); replay.mkdir()
    _trace_flux_build(build)
    (run / "MOM_CAPTURE.bin").write_bytes(b"entry")
    (replay / "MOM_TRACE.bin").write_bytes(b"trace")
    (replay / "MOM_FINAL.bin").write_bytes(b"final")
    (replay / "MOM_H_TRACE.bin").write_bytes(b"htrace")
    entry, final = _states()
    monkeypatch.setattr(analysis, "capture",
        lambda path, inventory: entry if Path(path).name == "MOM_CAPTURE.bin" else final)
    monkeypatch.setattr(analysis, "trace", lambda path: [_call()])
    output = tmp_path / "entry-analysis.json"

    report = analysis.analyze_entry_only(run, replay, build, output)

    assert report["entry_trace_absent"] is True
    assert report["formal_angular_support"]["available"] is True
    assert json.loads(output.read_text())["final_goal"] == "NOT_JUDGED"


def test_trace_flux_build_verifies_inputs_outputs_parent_and_generator(tmp_path):
    build = tmp_path / "build"
    _trace_flux_build(build)
    record = analysis.verify_entry_replay_build(build)
    assert record["trace_flux"] is True

    replay_exe = build / "replay_mom.exe"
    replay_exe.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="파일/해시 불일치"):
        analysis.verify_entry_replay_build(build)


@pytest.mark.parametrize("kwargs", [dict(trace_flux=False), dict(entry_only=True)])
def test_entry_replay_rejects_wrong_build_mode(tmp_path, kwargs):
    build = tmp_path / "build"
    _trace_flux_build(build, **kwargs)
    with pytest.raises(ValueError, match="trace-flux build 필요"):
        analysis.verify_entry_replay_build(build)


def test_entry_only_reports_missing_replay_mode_outputs(tmp_path):
    run, replay = tmp_path / "run", tmp_path / "replay"
    run.mkdir(); replay.mkdir()
    (replay / "MOM_FINAL.bin").write_bytes(b"final")
    build = tmp_path / "build"
    _trace_flux_build(build)
    with pytest.raises(ValueError, match="REPLAY mode"):
        analysis.analyze_entry_only(run, replay, build, tmp_path / "report.json")
