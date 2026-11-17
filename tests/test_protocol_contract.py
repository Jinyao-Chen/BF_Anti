# tests/test_protocol_contract.py
"""Docs-level protocol contract: AUX discovery, conformance CLI, motor A/B."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from msp import MSP_CMD
from onboard.config import FcConfig
from onboard.fc.mode import (
    BOX_PERMANENT_ID_ARM,
    BOX_PERMANENT_ID_MSPOVERRIDE,
    discover_aux,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeLink:
    """Answers (cmd, page) queries; discover_aux tries v1 then v2."""

    def __init__(self, pages):
        self.pages = dict(pages)
        self.calls = []

    def query(self, cmd, is_v2=False, timeout=None, payload=b""):
        self.calls.append((cmd, is_v2, payload))
        key = (cmd, payload)
        if key in self.pages:
            return self.pages[key]
        return self.pages.get((cmd, b""))


def _discovery_link(box_ids, box_names, ranges):
    link = FakeLink({
        (MSP_CMD.BOXIDS, b"\x00"): bytes(box_ids),
        (MSP_CMD.BOXNAMES, b"\x00"): box_names,
        (MSP_CMD.MODE_RANGES, b""): ranges,
    })
    return link


def test_discover_aux_resolves_permanent_ids_to_full_channels():
    ids = [0, 1, 2, 3, 26, 27, 28, 50]
    names = b"ARM;ANGLE;HORIZON;ALTHOLD;FAILSAFE;AIR MODE;POS HOLD;MSP OVERRIDE;"
    ranges = (bytes([0, 0, 32, 48]) + bytes([1, 1, 32, 48])
              + bytes([50, 6, 32, 48]) + bytes([26, 2, 32, 48])
              + bytes([0, 0, 0, 0]))
    cfg = FcConfig(aux_engage=7)
    discovery = discover_aux(_discovery_link(ids, names, ranges), cfg=cfg)

    assert discovery.arm_channel == 4          # AUX1
    assert discovery.angle_channel == 5        # AUX2
    assert discovery.override_channel == 10    # AUX7
    assert discovery.engage_channel == 7       # pilot switch from config
    assert discovery.override_range == (1700, 2100)
    assert discovery.config_snippet() == {
        "fc": {"aux_arm": 4, "aux_angle": 5, "aux_override": 10, "aux_engage": 7}
    }
    assert discovery.name_for(BOX_PERMANENT_ID_MSPOVERRIDE) == "MSP OVERRIDE"


def test_discover_aux_falls_back_to_box_index_for_old_firmware():
    ids = [50, 1]
    names = b"MSP OVERRIDE;ANGLE;"
    ranges = bytes([0, 5, 32, 48]) + bytes([1, 1, 32, 48])
    discovery = discover_aux(_discovery_link(ids, names, ranges), cfg=FcConfig())
    # Older firmware wrote the box index (0) instead of permanent ID 50.
    assert discovery.override_channel == 9    # AUX6
    assert discovery.angle_channel == 5
    assert discovery.arm_channel is None
    assert any("ARM mode range" in warning for warning in discovery.warnings)


def test_discover_aux_warns_when_override_box_missing():
    ids = [0, 1]
    names = b"ARM;ANGLE;"
    ranges = bytes([0, 0, 32, 48]) + bytes([1, 1, 32, 48])
    discovery = discover_aux(_discovery_link(ids, names, ranges))
    assert discovery.override_channel is None
    assert any("BOXMSPOVERRIDE" in warning for warning in discovery.warnings)
    assert "aux_override" not in discovery.config_snippet()["fc"]


def _run_cli(args):
    return subprocess.run([sys.executable] + args, cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=60)


def test_protocol_conformance_help_and_offline(tmp_path):
    help_run = _run_cli(["tools/protocol_conformance.py", "--help"])
    assert help_run.returncode == 0
    assert "--allow-rc" in help_run.stdout

    out = tmp_path / "offline.json"
    run = _run_cli(["tools/protocol_conformance.py", "--offline", "--out", str(out)])
    assert run.returncode == 0, run.stderr
    report = json.loads(out.read_text())
    framing = [c for c in report["checks"] if c["id"] == "framing_crc"][0]
    assert framing["status"] == "pass"
    assert report["summary"]["fail"] == 0
    assert report["flight_clearance"] is False


def test_fw_motor_ab_and_fc_identify_help():
    for script in ("tools/fw_motor_ab.py", "scripts/fc_identify.py"):
        run = _run_cli([script, "--help"])
        assert run.returncode == 0, "%s --help failed: %s" % (script, run.stderr)
        assert "usage" in run.stdout.lower()


def _load_tool(relative_path, name):
    path = os.path.join(REPO_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recording(tool, sequence, samples):
    return {
        "kind": tool.RECORD_KIND,
        "version": 1,
        "sequence": dict(sequence, sha256=tool.sequence_hash(sequence)),
        "samples": samples,
    }


def test_motor_ab_compare_tolerance_and_sequence_hash():
    tool = _load_tool("tools/fw_motor_ab.py", "fw_motor_ab_test")
    sequence = tool.normalise_sequence({
        "steps": [
            {"hold_s": 0.5, "channels": [1500, 1500, 1100, 1500] + [1000, 1500, 1500, 1000]},
            {"hold_s": 0.5, "channels": [1500, 1500, 1200, 1500] + [1000, 1500, 1500, 1000]},
        ]
    })
    samples = [
        {"t": 0.0, "step": 0, "motors": [1000, 1000, 1000, 1000]},
        {"t": 0.25, "step": 0, "motors": [1010, 1010, 1010, 1010]},
        {"t": 0.5, "step": 1, "motors": [1100, 1100, 1100, 1100]},
        {"t": 0.75, "step": 1, "motors": [1110, 1110, 1110, 1110]},
    ]
    base = _recording(tool, sequence, samples)
    same = tool.compare_recordings(base, base, tolerance=25.0)
    assert same["passed"] is True
    assert same["max_diff"] == 0.0

    changed_samples = [dict(sample) for sample in samples]
    changed_samples[2] = {"t": 0.5, "step": 1, "motors": [1200, 1100, 1100, 1100]}
    changed = _recording(tool, sequence, changed_samples)
    strict = tool.compare_recordings(base, changed, tolerance=25.0)
    assert strict["passed"] is False
    assert strict["max_diff"] == pytest.approx(100.0)
    loose = tool.compare_recordings(base, changed, tolerance=150.0)
    assert loose["passed"] is True

    other_sequence = dict(sequence, steps=[])
    mismatch = _recording(tool, other_sequence, samples)
    result = tool.compare_recordings(base, mismatch, tolerance=25.0)
    assert result["passed"] is False
    assert any("sha256" in reason for reason in result["reasons"])
    ignored = tool.compare_recordings(base, mismatch, tolerance=25.0,
                                      ignore_sequence_hash=True)
    assert any("sha256" in reason for reason in ignored["reasons"]) is False


def test_motor_ab_compare_cli_exit_codes(tmp_path):
    tool = _load_tool("tools/fw_motor_ab.py", "fw_motor_ab_cli_test")
    sequence = tool.normalise_sequence({"steps": [
        {"hold_s": 0.5, "channels": [1500, 1500, 1100, 1500] + [1000, 1500, 1500, 1000]},
    ]})
    samples_a = [
        {"t": 0.0, "step": 0, "motors": [1000, 1000, 1000, 1000]},
        {"t": 0.25, "step": 0, "motors": [1050, 1050, 1050, 1050]},
    ]
    samples_b = [
        {"t": 0.0, "step": 0, "motors": [1000, 1000, 1000, 1000]},
        {"t": 0.25, "step": 0, "motors": [1150, 1050, 1050, 1050]},
    ]
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(_recording(tool, sequence, samples_a)))
    path_b.write_text(json.dumps(_recording(tool, sequence, samples_b)))

    fail = _run_cli(["tools/fw_motor_ab.py", "compare", str(path_a), str(path_b),
                     "--tolerance", "25"])
    assert fail.returncode == 1, fail.stdout + fail.stderr
    assert "passed=False" in fail.stdout

    ok = _run_cli(["tools/fw_motor_ab.py", "compare", str(path_a), str(path_b),
                   "--tolerance", "200"])
    assert ok.returncode == 0, ok.stdout + ok.stderr
