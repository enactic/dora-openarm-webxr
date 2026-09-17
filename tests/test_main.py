# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import numpy as np
import pytest

from dora_openarm_webxr import main


class _FakeNode:
    def __init__(self):
        self.outputs = []

    def send_output(self, output_id, value, metadata=None):
        self.outputs.append((output_id, value))

    def ids(self):
        return [output_id for output_id, _value in self.outputs]

    def value(self, wanted_id):
        for output_id, value in self.outputs:
            if output_id == wanted_id:
                return value
        raise KeyError(wanted_id)


class _FakeTransport:
    # _process_frame only speaks to the transport when a calibration run
    # completes, which these tests never trigger; a do-nothing stub just
    # keeps that path from needing the real server.
    def send_control(self, payload):
        pass


@pytest.fixture
def node(monkeypatch):
    fake = _FakeNode()
    monkeypatch.setattr(main, "node", fake)
    monkeypatch.setattr(main, "webrtc_server", _FakeTransport())
    return fake


IDENTITY_POSE = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
    "qw": 1.0,
}


def test_frame_outputs(node):
    state = main._ConnectionState()
    main._process_frame(
        {
            "type": "frame",
            "sequence": 1,
            "pose_reference": dict(IDENTITY_POSE),
            "pose_right": dict(IDENTITY_POSE),
            "trigger_right": 0.5,
            "grip_left": 0.25,
            "joystick_right": [0.0, 0.0, 0.5, 0.25],
            "button_a": True,
        },
        state,
    )
    ids = node.ids()
    assert "vr_receive_times" in ids
    assert "pose_reference" in ids
    assert "pose_right" not in ids
    assert "trigger_right" in ids
    assert "grip_left" in ids
    assert "button_a" in ids
    assert node.value("button_a")[0].as_py() is True
    assert node.value("trigger_right")[0].as_py() == 0.5
    assert node.value("grip_left")[0].as_py() == 0.25
    # The thumbstick is the second axis pair, and the y sign is flipped
    # to keep the convention the downstream nodes were written against.
    assert node.value("joystick_x_right")[0].as_py() == 0.5
    assert node.value("joystick_y_right")[0].as_py() == -0.25
    main._publish_poses(state)
    main._publish_poses(state)
    assert node.ids().count("pose_right") == 2
    assert node.ids().count("vr_receive_times") == 1
    assert node.ids().count("button_a") == 1


def test_pose_needs_reference(node):
    # Hand poses are made relative to the viewer pose, so a frame
    # without one publishes the trigger but no pose.
    state = main._ConnectionState()
    main._process_frame(
        {
            "type": "frame",
            "sequence": 1,
            "pose_right": dict(IDENTITY_POSE),
            "trigger_right": 0.5,
        },
        state,
    )
    main._publish_poses(state)
    ids = node.ids()
    assert "pose_right" not in ids
    assert "trigger_right" in ids


def test_stale_frame_dropped(node):
    # The "xr" channel is unordered and never retransmits, so a frame
    # can arrive after a newer one was already published. Publishing it
    # would drag the arms back to a pose the operator has left.
    state = main._ConnectionState()
    main._process_frame({"type": "frame", "sequence": 2, "button_a": True}, state)
    published = len(node.outputs)

    main._process_frame({"type": "frame", "sequence": 1, "button_a": True}, state)
    assert len(node.outputs) == published

    main._process_frame({"type": "frame", "sequence": 2, "button_a": True}, state)
    assert len(node.outputs) == published

    main._process_frame({"type": "frame", "sequence": 3, "button_a": False}, state)
    assert len(node.outputs) > published


def test_frame_without_sequence_processed(node):
    state = main._ConnectionState()
    main._process_frame({"type": "frame", "button_a": True}, state)
    main._process_frame({"type": "frame", "button_a": True}, state)
    assert node.ids().count("button_a") == 2


def test_quit_button(node, monkeypatch):
    # A configured quit button shuts the node down straight from the
    # controller. The quit command is not sent here: it goes out when
    # the dora loop ends, so an error exit says it too.
    monkeypatch.setattr(main, "_QUIT_BUTTONS", ("a", "b"))
    monkeypatch.setattr(main, "_running", True)
    state = main._ConnectionState()
    main._process_frame(
        {"type": "frame", "sequence": 1, "button_b": False, "button_x": True}, state
    )
    assert main._running
    main._process_frame({"type": "frame", "sequence": 2, "button_b": True}, state)
    assert not main._running
    assert "command" not in node.ids()
    # The press still reaches the dataflow like any other button.
    assert node.ids().count("button_b") == 2


def test_quit_command_sent_on_exit(node, monkeypatch):
    # With --quit-button on, every exit publishes "quit" on the command
    # output, so a dora-openarm-quitter node ends the rest of the
    # dataflow even when this node died of an error.
    monkeypatch.setattr(main, "_QUIT_BUTTONS", ("a",))
    main._send_quit_command()
    assert node.value("command")[0].as_py() == "quit"


def test_quit_command_needs_quit_button(node, monkeypatch):
    # A dataflow that never asked for --quit-button keeps its own idea
    # of when to stop.
    monkeypatch.setattr(main, "_QUIT_BUTTONS", ())
    main._send_quit_command()
    assert "command" not in node.ids()


def test_quit_command_send_failure_swallowed(node, monkeypatch, capsys):
    # A dataflow already torn down can refuse the send; that must not
    # replace whatever error is already on its way out of main().
    class _RefusingNode:
        def send_output(self, output_id, value, metadata=None):
            raise RuntimeError("event stream closed")

    monkeypatch.setattr(main, "node", _RefusingNode())
    monkeypatch.setattr(main, "_QUIT_BUTTONS", ("a",))
    main._send_quit_command()
    assert "cannot send quit command" in capsys.readouterr().out


def test_session_start_resets_state(node, monkeypatch):
    # A new session must not inherit the last one's smoother history or
    # frame numbering: a reconnecting browser starts its "sequence" over.
    old_state = main._ConnectionState()
    main._process_frame({"type": "frame", "sequence": 100, "button_a": True}, old_state)
    monkeypatch.setattr(main, "_state", old_state)
    main._on_session_start()
    assert main._state is not old_state
    assert main._state.last_sequence == -1
    assert main._state.latest_frame is None
    assert node.ids() == ["vr_receive_times", "button_a", "status"]
    main._process_frame({"type": "frame", "sequence": 1, "button_b": True}, main._state)
    assert "button_b" in node.ids()


def _hand_frame(sequence=1):
    return {
        "sequence": sequence,
        "pose_reference": dict(IDENTITY_POSE),
        "pose_right": dict(IDENTITY_POSE),
        "pose_left": dict(IDENTITY_POSE),
        "trigger_right": 0.5,
        "trigger_left": 0.5,
    }


@pytest.mark.parametrize("missing", ["pose_right", "trigger_right", "pose_reference"])
def test_invalid_target_suspends_and_recovers(node, monkeypatch, missing):
    clock = [1.0]
    monkeypatch.setattr(main.time, "perf_counter", lambda: clock[0])
    state = main._ConnectionState()
    main._publish_poses(state)
    assert not node.outputs
    frame = _hand_frame()
    main._process_frame(frame, state)
    main._publish_poses(state)
    smoother = state.smoothers["right"]
    previous = smoother.p_prev.copy()
    invalid = _hand_frame(2)
    del invalid[missing]
    clock[0] = 10.0
    main._process_frame(invalid, state)
    main._publish_poses(state)
    assert node.ids().count("pose_right") == 1
    assert node.ids().count("pose_left") == (1 if missing == "pose_reference" else 2)
    assert smoother.t_prev == 10.0
    np.testing.assert_array_equal(smoother.dp_prev, np.zeros(3))
    np.testing.assert_array_equal(smoother.p_prev, previous)

    frame["sequence"] = 3
    frame["pose_right"]["x"] = 1.0
    clock[0] += 0.002
    main._process_frame(frame, state)
    main._publish_poses(state)
    assert 0 < np.linalg.norm(smoother.p_prev - previous) <= 0.002001
    previous = smoother.p_prev.copy()
    clock[0] += 0.002
    main._publish_poses(state)
    assert 0 < np.linalg.norm(smoother.p_prev - previous) <= 0.002001
    assert node.ids().count("vr_receive_times") == 3


@pytest.mark.parametrize("timeout", [0.0, 0.05])
def test_pose_timeout_uses_accepted_frame_receive_time(node, monkeypatch, timeout):
    clock = [1.0]
    monkeypatch.setattr(main.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(main, "_POSE_TIMEOUT", timeout)
    state = main._ConnectionState()
    main._process_frame(_hand_frame(), state)
    main._publish_poses(state)
    clock[0] = 2.0
    main._process_frame(_hand_frame(), state)  # Duplicate sequence is not a refresh.
    main._publish_poses(state)
    assert state.received_at == 1.0
    assert node.ids().count("pose_right") == (1 if timeout else 2)
    assert (state.latest_frame is None) == bool(timeout)
    assert state.smoothers["right"].t_prev == 2.0


def test_calibration_samples_are_receive_driven(node, monkeypatch):
    monkeypatch.setattr(main, "_CALIBRATION_ENABLED", True)
    state = main._ConnectionState()
    frame = _hand_frame()
    frame["button_y"] = True
    main._process_frame(frame, state)
    for _ in range(5):
        main._publish_poses(state)
    assert "pose_right" not in node.ids()
    assert len(state.pivot_calibration._samples) == 1
    assert node.ids().count("trigger_right") == 1


def test_session_end_clears_cached_targets(node, monkeypatch):
    state = main._ConnectionState()
    monkeypatch.setattr(main, "_state", state)
    main._process_frame(_hand_frame(), state)
    main._publish_poses(state)
    main._on_session_end()
    main._publish_poses(state)
    assert state.latest_frame is None
    assert node.ids().count("pose_right") == 1


def _pose(position, rotation):
    return dict(
        zip(
            ("x", "y", "z", "qx", "qy", "qz", "qw"),
            [*position, *rotation.as_quat()],
            strict=True,
        )
    )


def _adjust(pose, reference):
    return np.asarray(
        main._adjust_pose(pose, reference, main.OneEuroPoseSmoother(), 1.0)
    )


def test_neck_mode_keeps_target_fixed_when_head_rotates_about_pivot(monkeypatch):
    monkeypatch.setattr(main, "_POSE_MODE", "neck")
    pivot = np.array([0.2, 1.4, -0.1])
    hand_position = np.array([0.4, 1.2, -0.5])
    hand_rotation = main.Rotation.from_euler("xyz", [15, -20, 35], degrees=True)
    expected_position = (
        main._ROBOT_ROTATION.apply(hand_position - pivot) + main._FRAME_OFFSET_CELL
    )
    expected_rotation = main._ROBOT_ROTATION * hand_rotation * main._CONTROLLER_TO_EE
    for angles in ([0, 0, 0], [25, 60, -15]):
        head_rotation = main.Rotation.from_euler("xyz", angles, degrees=True)
        head_position = pivot - head_rotation.apply(main._NECK_PIVOT_OFFSET)
        result = _adjust(
            _pose(hand_position, hand_rotation), _pose(head_position, head_rotation)
        )
        np.testing.assert_allclose(result[:3], expected_position, atol=1e-6)
        assert abs(result[3:] @ expected_rotation.as_quat()[[3, 0, 1, 2]]) == (
            pytest.approx(1.0, abs=1e-6)
        )


def test_relative_mode_follows_reference_frame_and_ignores_neck(monkeypatch):
    monkeypatch.setattr(main, "_POSE_MODE", "relative")
    monkeypatch.setattr(main, "_NECK_PIVOT_OFFSET", np.array([10.0, 20.0, 30.0]))
    relative_position = np.array([0.2, -0.3, -0.4])
    relative_rotation = main.Rotation.from_euler("xyz", [20, 40, -10], degrees=True)
    expected_position = (
        main._ROBOT_ROTATION.apply(relative_position) + main._FRAME_OFFSET_CELL
    )
    expected_rotation = (
        main._ROBOT_ROTATION * relative_rotation * main._CONTROLLER_TO_EE
    )
    for angles, head_position in (([0, 0, 0], [0, 0, 0]), ([25, 60, -15], [1, 2, 3])):
        head_rotation = main.Rotation.from_euler("xyz", angles, degrees=True)
        hand_position = head_rotation.apply(relative_position) + head_position
        result = _adjust(
            _pose(hand_position, head_rotation * relative_rotation),
            _pose(head_position, head_rotation),
        )
        np.testing.assert_allclose(result[:3], expected_position, atol=1e-6)
        assert abs(result[3:] @ expected_rotation.as_quat()[[3, 0, 1, 2]]) == (
            pytest.approx(1.0, abs=1e-6)
        )


@pytest.mark.parametrize(
    ("mode", "calibrate", "error"),
    [
        (None, False, None),
        ("relative", False, None),
        ("relative", True, "--calibration requires pose.mode=neck"),
        ("invalid", False, "pose.mode must be 'neck' or 'relative'"),
    ],
)
def test_pose_mode_configuration(node, monkeypatch, capsys, mode, calibrate, error):
    argv = ["webxr", "--offer", "test", "--answer-port", "1234"]
    if calibrate:
        argv.append("--calibration")
    monkeypatch.setattr(sys, "argv", argv)
    for name in ("args", "_POSE_MODE", "_CALIBRATION_ENABLED", "_NECK_PIVOT_FILE"):
        monkeypatch.setattr(main, name, getattr(main, name))
    monkeypatch.setattr(main.video, "configure", lambda args: None)
    pose = {} if mode is None else {"mode": mode}
    monkeypatch.setattr(main.video, "view_configuration", lambda: {"pose": pose})
    neck_reads = []
    monkeypatch.setattr(
        main, "_configure_neck_pivot", lambda *args: neck_reads.append(args)
    )
    monkeypatch.setattr(main.dora, "Node", lambda: node)

    async def run():
        pass

    monkeypatch.setattr(main, "_main_async", run)
    if error:
        with pytest.raises(SystemExit) as exc:
            main.main()
        assert exc.value.code == 2
        assert error in capsys.readouterr().err
    else:
        main.main()
        assert main._POSE_MODE == (mode or "neck")
        assert len(neck_reads) == (0 if mode == "relative" else 1)
