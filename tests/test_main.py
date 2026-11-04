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

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

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


@pytest.mark.parametrize("ending", [None, {"type": "STOP"}, "error"])
def test_dora_reader_does_not_wait_for_video_pool(node, monkeypatch, ending):
    readers = []

    def read(timeout):
        assert timeout == 0.1
        readers.append(threading.current_thread())
        if ending == "error":
            raise RuntimeError("read failed")
        return ending

    node.is_empty = lambda: True
    node.next = read
    monkeypatch.setattr(main, "_serving", lambda: True)
    monkeypatch.setattr(main, "_QUIT_BUTTONS", ("a",))
    monkeypatch.setattr(main, "server", None)
    monkeypatch.setattr(main, "_running", True)

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        release = threading.Event()
        busy = loop.run_in_executor(None, release.wait)
        task = asyncio.create_task(main._main_dora())

        async def wait_for_read():
            while not readers:
                await asyncio.sleep(0.001)

        try:
            await asyncio.wait_for(wait_for_read(), 2)
            assert not busy.done()
        finally:
            release.set()
            await busy
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        if ending == "error":
            with pytest.raises(RuntimeError, match="read failed"):
                task.result()
        else:
            task.result()

    asyncio.run(run())
    assert node.value("command")[0].as_py() == "quit"
    assert not main._running
    assert len(readers) == 1
    assert readers[0].name.startswith("webxr-dora")
    assert not readers[0].is_alive()


def test_cancelled_dora_reader_keeps_loop_responsive_and_joins(node, monkeypatch):
    release = threading.Event()
    readers, released = [], []
    node.is_empty = lambda: True
    monkeypatch.setattr(main, "_serving", lambda: True)
    monkeypatch.setattr(main, "_QUIT_BUTTONS", ("a",))
    monkeypatch.setattr(main, "server", None)
    monkeypatch.setattr(main, "_running", True)

    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def read(timeout):
            assert timeout == 0.1
            readers.append(threading.current_thread())
            loop.call_soon_threadsafe(started.set)
            released.append(release.wait(2))

        node.next = read
        task = asyncio.create_task(main._main_dora())
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            # This coroutine must resume while cleanup is still joining the reader.
            assert not task.done()
            assert node.value("command")[0].as_py() == "quit"
            assert not main._running
        finally:
            release.set()
            if not task.cancelling():
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)

    asyncio.run(run())
    assert released == [True]
    assert len(readers) == 1
    assert not readers[0].is_alive()


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("offer", [None, "offer"])
def test_video_pool_configuration_and_shutdown(monkeypatch, workers, offer):
    pools, threads = [], []

    def create_pool(**kwargs):
        pools.append(kwargs)
        return ThreadPoolExecutor(**kwargs)

    async def hosted():
        threads.append(await asyncio.to_thread(threading.current_thread))

    monkeypatch.setattr(
        main, "args", SimpleNamespace(video_workers=workers, offer=offer)
    )
    monkeypatch.setattr(main, "_state", None)
    monkeypatch.setattr(main, "webrtc_server", None)
    monkeypatch.setattr(main, "ThreadPoolExecutor", create_pool)
    monkeypatch.setattr(main, "_main_hosted", hosted)
    monkeypatch.setattr(main, "_main_webrtc_only", hosted)
    asyncio.run(main._main_async())
    assert pools == (
        [{"max_workers": workers, "thread_name_prefix": "webxr-video"}]
        if workers
        else []
    )
    assert not threads[0].is_alive()


def test_negative_video_workers_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["webxr", "--video-workers", "-1"])
    monkeypatch.setattr(main, "args", None)
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2
    assert "--video-workers must be non-negative" in capsys.readouterr().err


# pose.clutch: the node's side of dora_openarm_webxr.clutch, which has its own
# tests. These use a fake dora node and a fake clock, and touch no hardware.

_CLUTCH_INITIAL = {
    "initial_right": [0.2, -0.15, -0.2, 0.7071068, 0.0, -0.7071068, 0.0],
    "initial_left": [0.2, 0.15, -0.2, 0.7071068, 0.0, -0.7071068, 0.0],
}


@pytest.fixture
def clutches(monkeypatch):
    hands = main._build_clutches(_CLUTCH_INITIAL)
    monkeypatch.setattr(main, "_CLUTCHES", hands)
    monkeypatch.setattr(main, "_POSE_TIMEOUT", 0.2)
    return hands


@pytest.fixture
def clock(monkeypatch):
    now = [1.0]
    monkeypatch.setattr(main.time, "perf_counter", lambda: now[0])
    return now


def _clutch_frame(sequence, grip_right=0.0, grip_left=0.0, x=0.0):
    frame = _hand_frame(sequence)
    frame["grip_right"] = grip_right
    frame["grip_left"] = grip_left
    frame["pose_right"]["x"] = x
    return frame


def _tick(state, clock, frame=None):
    node_outputs = main.node.outputs
    node_outputs.clear()
    clock[0] += 0.004
    if frame is not None:
        main._process_frame(frame, state)
    main._publish_poses(state)
    return {
        output_id: np.array(value.field("pose")[0].values)
        for output_id, value in node_outputs
        if output_id in ("pose_right", "pose_left")
    }


def _engage(state, clock, sequence=1, grip_right=1.0, grip_left=1.0, x=0.0):
    _tick(state, clock, _clutch_frame(sequence, x=x))
    return _tick(state, clock, _clutch_frame(sequence + 1, grip_right, grip_left, x))


def test_clutch_off_leaves_the_default_path_alone(node, clock):
    assert main._CLUTCHES == {}
    state = main._ConnectionState()
    assert set(_tick(state, clock, _hand_frame())) == {"pose_right", "pose_left"}


def test_clutch_first_engagement_publishes_the_initial_targets(node, clutches, clock):
    state = main._ConnectionState()
    outputs = _engage(state, clock, x=3.0)
    np.testing.assert_allclose(
        outputs["pose_right"][:7], _CLUTCH_INITIAL["initial_right"], atol=1e-6
    )
    np.testing.assert_allclose(
        outputs["pose_left"][:7], _CLUTCH_INITIAL["initial_left"], atol=1e-6
    )
    # Open: the trigger has been at rest, and the gripper starts open.
    assert outputs["pose_right"][7] == pytest.approx(np.deg2rad(-45.0), abs=1e-6)


def test_clutch_publishes_nothing_until_authorized(node, clutches, clock):
    state = main._ConnectionState()
    assert _tick(state, clock, _clutch_frame(1, 1.0, 1.0)) == {}
    assert _tick(state, clock, _clutch_frame(2, 1.0, 1.0)) == {}


def test_clutch_publishes_nothing_while_both_hands_are_released(node, clutches, clock):
    state = main._ConnectionState()
    _engage(state, clock)
    assert _tick(state, clock, _clutch_frame(3, x=0.5)) == {}
    assert "pose_right" not in node.ids()
    # The buttons, triggers and grips keep flowing.
    assert "grip_right" in node.ids()


def test_clutch_regrip_continues_from_the_published_target(node, clutches, clock):
    state = main._ConnectionState()
    _engage(state, clock)
    for sequence in range(3, 30):
        moved = _tick(
            state, clock, _clutch_frame(sequence, 1.0, 1.0, x=(sequence - 2) * 0.002)
        )
    held = moved["pose_right"]

    assert _tick(state, clock, _clutch_frame(30, x=2.0)) == {}
    regripped = _tick(state, clock, _clutch_frame(31, 1.0, 1.0, x=2.0))
    np.testing.assert_array_equal(regripped["pose_right"], held)
    # And it stays there while the hand stays still.
    still = _tick(state, clock, _clutch_frame(32, 1.0, 1.0, x=2.0))
    np.testing.assert_allclose(still["pose_right"], held, atol=1e-6)


def test_clutch_step_is_bounded_by_the_speed_limit_after_a_regrip(
    node, clutches, clock
):
    state = main._ConnectionState()
    _engage(state, clock)
    _tick(state, clock, _clutch_frame(3))
    _tick(state, clock, _clutch_frame(4, 1.0, 1.0))
    last = None
    for sequence in range(5, 25):
        # A hand moving far faster than the limit.
        outputs = _tick(
            state, clock, _clutch_frame(sequence, 1.0, 1.0, x=(sequence - 4) * 0.5)
        )
        if last is not None:
            step = np.linalg.norm(outputs["pose_right"][:3] - last)
            assert 0.0 < step <= main._MAX_LINEAR_SPEED * 0.004 + 1e-5
        last = outputs["pose_right"][:3]


def test_clutch_released_hand_holds_while_the_other_moves(node, clutches, clock):
    # Bimanual IK needs both targets, so this is what lets one hand keep working.
    state = main._ConnectionState()
    _engage(state, clock)
    held = main._CLUTCHES["left"].target.position.copy()
    for sequence in range(3, 8):
        frame = _clutch_frame(sequence, 1.0, 0.0, x=sequence * 0.001)
        frame["pose_left"]["x"] = sequence * 0.1
        outputs = _tick(state, clock, frame)
        np.testing.assert_allclose(outputs["pose_left"][:3], held, atol=1e-6)
    assert (
        np.linalg.norm(outputs["pose_right"][:3] - _CLUTCH_INITIAL["initial_right"][:3])
        > 0.0
    )


def test_clutch_one_hand_engages_with_the_other_never_authorized(node, clutches, clock):
    state = main._ConnectionState()
    outputs = _engage(state, clock, grip_right=1.0, grip_left=0.0)
    assert set(outputs) == {"pose_right", "pose_left"}
    np.testing.assert_allclose(
        outputs["pose_left"][:7], _CLUTCH_INITIAL["initial_left"], atol=1e-6
    )


@pytest.mark.parametrize(
    "corrupt",
    [
        {"pose_right": None},
        {"pose_right": {"x": 0.0}},
        {"grip_right": None},
        {"grip_right": 2.0},
        {"trigger_right": None},
        {"pose_reference": None},
    ],
)
def test_clutch_unusable_input_faults_the_hand(
    node, clutches, clock, monkeypatch, corrupt
):
    monkeypatch.setattr(main, "_POSE_MODE", "relative")
    state = main._ConnectionState()
    _engage(state, clock)
    held = main._CLUTCHES["right"].target
    frame = {**_clutch_frame(3, 1.0, 1.0, x=1.0), **corrupt}
    _tick(state, clock, frame)
    assert main._CLUTCHES["right"].state is main.clutch.ClutchState.AWAIT_RELEASE
    assert main._CLUTCHES["right"].target is held
    # Good input with the grip still held does not bring it back.
    _tick(state, clock, _clutch_frame(4, 1.0, 1.0, x=1.0))
    assert main._CLUTCHES["right"].state is main.clutch.ClutchState.AWAIT_RELEASE


def test_clutch_neck_mode_does_not_need_the_head(node, clutches, clock):
    state = main._ConnectionState()
    _engage(state, clock)
    frame = _clutch_frame(3, 1.0, 1.0)
    del frame["pose_reference"]
    assert set(_tick(state, clock, frame)) == {"pose_right", "pose_left"}


def test_clutch_stale_link_stops_and_needs_a_new_squeeze(node, clutches, clock):
    state = main._ConnectionState()
    _engage(state, clock)
    assert _tick(state, clock, _clutch_frame(3, 1.0, 1.0))
    clock[0] += 0.3
    assert _tick(state, clock) == {}
    for hand in main._CLUTCHES.values():
        assert hand.state is main.clutch.ClutchState.AWAIT_RELEASE
    assert _tick(state, clock, _clutch_frame(4, 1.0, 1.0)) == {}
    assert _tick(state, clock, _clutch_frame(5)) == {}
    assert set(_tick(state, clock, _clutch_frame(6, 1.0, 1.0))) == {
        "pose_right",
        "pose_left",
    }


def test_clutch_reconnect_with_the_grip_held_does_not_resume(
    node, clutches, clock, monkeypatch
):
    state = main._ConnectionState()
    monkeypatch.setattr(main, "_state", state)
    _engage(state, clock)
    moved = _tick(state, clock, _clutch_frame(3, 1.0, 1.0, x=0.01))["pose_right"]

    # A new session with no tick in between, so nothing saw a gap.
    main._on_session_end()
    main._on_session_start()
    state = main._state
    assert _tick(state, clock, _clutch_frame(1, 1.0, 1.0, x=2.0)) == {}
    assert _tick(state, clock, _clutch_frame(2, x=2.0)) == {}
    resumed = _tick(state, clock, _clutch_frame(3, 1.0, 1.0, x=2.0))["pose_right"]
    np.testing.assert_allclose(resumed, moved, atol=1e-6)


def test_clutch_late_frame_cannot_reengage(node, clutches, clock):
    state = main._ConnectionState()
    _engage(state, clock)
    _tick(state, clock, _clutch_frame(5))
    main._process_frame(_clutch_frame(4, 1.0, 1.0), state)
    assert _tick(state, clock) == {}


@pytest.mark.parametrize("mode", ["neck", "relative"])
def test_clutch_motion_matches_the_absolute_mapping(
    node, clutches, clock, monkeypatch, mode
):
    # The check on the frames: with a still head, an engaged clutch has to move
    # the target exactly as the absolute mapping would, for any controller
    # motion, because all it changes is where that motion starts from.
    monkeypatch.setattr(main, "_POSE_MODE", mode)
    head_rotation = main.Rotation.from_euler("yx", [0.4, -0.2])
    head = _pose([0.1, 1.6, -0.2], head_rotation)
    start = _pose([0.2, 1.1, -0.4], main.Rotation.from_euler("zyx", [0.3, 0.2, 0.1]))
    moved = _pose([0.3, 1.0, -0.6], main.Rotation.from_euler("zyx", [-0.2, 0.6, 0.4]))
    # No filtering, so that this compares the mappings alone.
    monkeypatch.setattr(main, "_MAX_LINEAR_SPEED", 0.0)
    monkeypatch.setattr(main, "_MAX_ANGULAR_SPEED", 0.0)
    monkeypatch.setattr(
        main.OneEuroPoseSmoother, "smooth", lambda self, t, pose: pose.copy()
    )

    def absolute(pose):
        return np.asarray(_adjust(pose, head))

    state = main._ConnectionState()

    def frame(sequence, grip, right):
        return {
            **_hand_frame(sequence),
            "pose_reference": head,
            "pose_right": right,
            "grip_right": grip,
            "grip_left": 0.0,
        }

    _tick(state, clock, frame(1, 0.0, start))
    _tick(state, clock, frame(2, 1.0, start))
    published = _tick(state, clock, frame(3, 1.0, moved))["pose_right"]
    initial = np.array(_CLUTCH_INITIAL["initial_right"])

    expected_position = initial[:3] + absolute(moved)[:3] - absolute(start)[:3]
    np.testing.assert_allclose(published[:3], expected_position, atol=1e-5)

    def rotation(wxyz):
        return main.Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]])

    increment = rotation(absolute(moved)[3:7]) * rotation(absolute(start)[3:7]).inv()
    expected_rotation = increment * rotation(initial[3:7])
    assert (rotation(published[3:7]) * expected_rotation.inv()).magnitude() < 1e-5


@pytest.mark.parametrize(
    ("configuration", "calibrate", "error"),
    [
        ({"initial_right": _CLUTCH_INITIAL["initial_right"]}, False, "initial_left"),
        ({**_CLUTCH_INITIAL, "initial_left": [0.0, 0.0, 0.0]}, False, "pose.clutch"),
        ({**_CLUTCH_INITIAL, "initial_left": [0.0] * 7}, False, "not a valid pose"),
        ("yes", False, "pose.clutch"),
        (_CLUTCH_INITIAL, True, "--calibration cannot be used with pose.clutch"),
    ],
)
def test_clutch_configuration_is_rejected_at_startup(
    node, monkeypatch, capsys, configuration, calibrate, error
):
    argv = ["webxr", "--offer", "test", "--answer-port", "1234"]
    if calibrate:
        argv.append("--calibration")
    monkeypatch.setattr(sys, "argv", argv)
    for name in ("args", "_POSE_MODE", "_CALIBRATION_ENABLED", "_NECK_PIVOT_FILE"):
        monkeypatch.setattr(main, name, getattr(main, name))
    monkeypatch.setattr(main, "_CLUTCHES", {})
    monkeypatch.setattr(main.video, "configure", lambda args: None)
    monkeypatch.setattr(
        main.video, "view_configuration", lambda: {"pose": {"clutch": configuration}}
    )
    monkeypatch.setattr(main, "_configure_neck_pivot", lambda *args: None)
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2
    assert error in capsys.readouterr().err


@pytest.mark.parametrize(("given", "used"), [(0.0, 0.2), (0.5, 0.5)])
def test_clutch_always_has_a_pose_timeout(node, monkeypatch, given, used):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "webxr",
            "--offer",
            "test",
            "--answer-port",
            "1234",
            "--pose-timeout",
            str(given),
        ],
    )
    for name in ("args", "_POSE_MODE", "_CALIBRATION_ENABLED", "_NECK_PIVOT_FILE"):
        monkeypatch.setattr(main, name, getattr(main, name))
    monkeypatch.setattr(main, "_CLUTCHES", {})
    monkeypatch.setattr(main, "_POSE_TIMEOUT", 0.0)
    monkeypatch.setattr(main.video, "configure", lambda args: None)
    monkeypatch.setattr(
        main.video, "view_configuration", lambda: {"pose": {"clutch": _CLUTCH_INITIAL}}
    )
    monkeypatch.setattr(main, "_configure_neck_pivot", lambda *args: None)
    monkeypatch.setattr(main.dora, "Node", lambda: node)

    async def run():
        pass

    monkeypatch.setattr(main, "_main_async", run)
    main.main()
    assert main._POSE_TIMEOUT == used
    assert set(main._CLUTCHES) == {"right", "left"}
