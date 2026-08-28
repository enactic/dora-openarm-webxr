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
    assert "pose_right" in ids
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
    assert node.ids() == ["vr_receive_times", "button_a", "status"]
    main._process_frame({"type": "frame", "sequence": 1, "button_b": True}, main._state)
    assert "button_b" in node.ids()
