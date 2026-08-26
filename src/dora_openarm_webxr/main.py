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

"""WebXR server node for OpenArm teleoperation.

This dora-rs node talks WebRTC with a VR device such as Meta Quest 3 or
PICO 4. Controller poses arrive on an unreliable data channel and head
camera frames leave on video tracks; :mod:`.webrtc` owns that half.
For each frame received from the device, it converts the controller
pose from WebXR coordinates into the OpenArm workspace, smooths it with
a One Euro filter, and publishes the pose, trigger, joystick and button
state as dora-rs outputs.

The published poses are expressed in the scene's ``arm_origin`` site
frame (chest-level origin between the arms), not in world coordinates.
Downstream IK interprets targets in the same frame. The hand position
is relative to the headset but keeps the world axes, so looking around
does not drag the target with the head.

The headset pose that the hands are made relative to is published as
is on ``pose_reference``, in the WebXR reference space, for consumers
that drive something from head motion such as a neck.

The node serves the front-end itself over HTTPS by default, which
WebXR requires of any page it runs in. Passing ``--offer`` runs it as a
pure WebRTC peer instead, with no HTTP server at all: another service
hosts the page and brokers signaling, and this node needs no
certificate of its own -- WebRTC brings its own.

The Web server and the dora-rs event loop run concurrently in a single
asyncio event loop; the node shuts down when the dora-rs node receives
a ``STOP`` event, or, in WebRTC-only mode, when the one browser leaves.
"""

import argparse
import asyncio
import collections
import dora
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import numpy as np
import os
import pathlib
import pyarrow as pa
from scipy.spatial.transform import Rotation
import time
import uvicorn
import yaml

from .smoothing import OneEuroPoseSmoother
from . import calibration
from . import video
from . import webrtc

args = None
node = None
server = None
webrtc_server = None
_state = None
_running = True


# Relative pose to robot workspace mapping.
# We may need to adjust this.
_ROBOT_ROTATION_MATRIX: np.ndarray = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
_ROBOT_ROTATION = Rotation.from_matrix(_ROBOT_ROTATION_MATRIX)

# Relative pose is computed from the viewer position.
# We need to move it to OpenArm position.
#
# Neutral hand position relative to the arm_origin site (chest level).
# Overridden by ``pose: frame_offset`` in the view configuration file.
_FRAME_OFFSET_CELL: np.ndarray = np.array([-0.085, 0, -0.14], dtype=np.float32)

# The client reads the controller from the WebXR target ray space, which is the
# OpenXR aim pose: its -Z points where the controller points, the same
# convention the Unity sender published. So the frame above and the turn below,
# both carried over from that sender, get the pose they were tuned for with no
# fixed conversion in between. The grip pose the client used to send runs its -Z
# along the handle toward the thumb instead, and turning that into a pointing
# pose here meant hardcoding one headset's handle geometry; the aim pose leaves
# every runtime to state its own.
#
# This turn maps the aim frame onto the end effector one, which points the
# gripper along its own -z and opens it across y.
_CONTROLLER_TO_EE: Rotation = Rotation.from_euler("z", 90, degrees=True)

# Eyes to the neck's rotation axis, in the headset's own frame: -Z is forward
# and +Y is up here, so this reads as below and behind the face.
#
# The head does not turn about the headset. The rotation axis is in the neck,
# and the headset rides ahead of it, so a head turn swings the headset along an
# arc. Subtracting the headset position alone therefore still reads that arc as
# the operator translating, and drags the target with it -- 4 to 6 cm at 30
# degrees and 11 cm at 90. Subtracting the pivot instead leaves the reference
# motionless under head rotation while it keeps translating when the operator
# leans or walks, so the workspace follows the body without inheriting the neck.
#
# This is the neck model that 3DOF VR used in reverse: there it synthesised the
# eye translation a headset could not measure, here it removes the one we can.
#
# An estimate, not a measurement, and anatomy varies: override it per operator
# with ``pose: neck_pivot_offset`` in the view configuration file, or measure it
# in a session started with --calibration, by holding the Y button down while
# turning the head. Setting it to [0, 0, 0] restores the plain headset
# subtraction, which is also how the two can be compared.
_NECK_PIVOT_OFFSET: np.ndarray = np.array([0.0, -0.075, 0.080], dtype=np.float32)

# Whether the Y button measures the neck pivot at all, from --calibration.
# Off unless asked for: the button is published as an output in its own right
# and a dataflow may have wired it to something the operator holds down, so
# outside a calibration session it stays an ordinary button and the hands never
# stop. Measuring is a thing the operator sets out to do, not something a press
# can start by accident.
_CALIBRATION_ENABLED: bool = False

# Where an accepted run is written and where a saved one is read back from, at
# startup, from --neck-pivot-file. A fit lives as long as the session
# otherwise, and the whole point of measuring an operator is keeping the number
# they measured.
_NECK_PIVOT_FILE = None


# How many headset poses a run may hold. A button left held down stops growing
# here instead of the process, and at the headset's display rate this is some
# twenty seconds, well past any deliberate shake.
_CALIBRATION_CAPACITY = 2000

# Fewer poses than this and the run is too thin to fit from. At the headset's
# display rate this is a second or so of holding, so it catches the operator
# who lets the button go before turning their head at all.
_CALIBRATION_MIN_SAMPLES = 100

# A run has to turn the head far enough about every axis for the fit to see
# all three offset components. 0.02 is about a 20 degree sweep, which a
# deliberate shake passes twice over.
_CALIBRATION_MIN_OBSERVABILITY = 0.02

# The headset's own axes, in the order the offset components come in.
_OFFSET_AXIS_NAMES = ("lateral", "vertical", "fore-aft")

# The head motion that pins each of those components. A rotation cannot see
# the offset along the axis it turns about, so what is missing is always a
# turn about a different one: only yawing hides the vertical offset, and
# only nodding hides the lateral one.
_PINNING_MOTION = ("side to side", "up and down", "side to side")

# How far the fitted pivot may still wander over the run, in meters. Body
# motion lands here, and so does the model error: a neck yaws and nods about
# joints a few centimeters apart rather than the one point fitted here, so
# some residual is the anatomy rather than the operator. 20 mm leaves the
# offset good to about a centimeter, against the 11 cm error it removes.
_CALIBRATION_MAX_RESIDUAL = 0.020

# Where a neck can be, relative to the eyes, in meters: on the midline, and
# below and behind them. A run that satisfies everything above can still land
# somewhere a body does not go, and this is the last thing between that and
# the arm following it.
_CALIBRATION_MAX_LATERAL = 0.05
_CALIBRATION_MAX_VERTICAL = 0.20
_CALIBRATION_MAX_FORE_AFT = 0.20


class _PivotCalibration:
    """Collects headset poses while the operator holds the Y button down.

    The button state arrives once per frame rather than as a press and a
    release event, so the edges are found here. Reading it that way is
    also the failsafe: a controller that falls asleep mid-run stops
    reporting the button at all, the caller reads that as not pressed,
    and the run ends instead of leaving the hands stopped for good.

    A node started without ``--calibration`` holds a disabled one, which
    keeps no poses and never stops the hands: the Y button reaches its
    output and nothing else happens.
    """

    def __init__(self, enabled=False, capacity=_CALIBRATION_CAPACITY):
        self._enabled = enabled
        self._running = False
        self._samples = collections.deque(maxlen=capacity)

    @property
    def collecting(self):
        """Whether a run is under way, so poses belong to it."""
        return self._running

    def update(self, pressed):
        """Take the button state for a frame, returning a finished run.

        The press is the run: it starts the frame the button goes down
        and is fitted the frame it comes up. Nothing waits for a hold,
        because only a session started with ``--calibration`` reaches
        here, and in one the operator pressing Y wants exactly this. The
        headset says so on its own display, so the hands stopping is no
        longer the only signal the operator gets.

        Args:
          pressed: whether the Y button is down this frame.

        Returns:
          The run's poses as ``(N, 7)`` rows of ``[x, y, z, qx, qy, qz,
          qw]``, which is the quaternion order ``Rotation.from_quat``
          reads, or None if this frame ended no run worth fitting.

        """
        if not self._enabled:
            return None

        if pressed:
            if not self._running:
                self._running = True
                self._samples.clear()
            return None

        running, self._running = self._running, False
        if not running or not self._samples:
            return None
        return np.array(self._samples, dtype=np.float64)

    def add(self, reference):
        """Keep a headset pose if a run is under way, otherwise drop it."""
        if not self.collecting:
            return
        self._samples.append(
            [
                reference["x"],
                reference["y"],
                reference["z"],
                reference["qx"],
                reference["qy"],
                reference["qz"],
                reference["qw"],
            ]
        )


def _apply_pivot_calibration(samples):
    """Fit the neck pivot from a run, say what came of it, and use it.

    The fitted offset only replaces the one in use if it passes every
    check, so a run that went wrong costs the operator the shake and
    nothing else.

    Returns what to tell the headset about the run, since the operator is
    wearing one and cannot read the node's output from inside it.
    """
    offset, diagnostics = calibration.fit_pivot_offset(
        samples[:, :3], Rotation.from_quat(samples[:, 3:])
    )
    reason = _check_pivot_offset(offset, diagnostics)
    if reason is not None:
        print(f"neck pivot calibration rejected: {reason}", flush=True)
        return {"type": "calibration-result", "accepted": False, "reason": reason}

    global _NECK_PIVOT_OFFSET
    _NECK_PIVOT_OFFSET = offset.astype(np.float32)
    saved_to = _save_neck_pivot_offset(offset)

    formatted = ", ".join(f"{component:.3f}" for component in offset)
    kept = (
        f"  Written to {saved_to}, which this node reads at startup."
        if saved_to is not None
        else "  Kept for this session only; --neck-pivot-file writes it down."
    )
    print(
        f"neck pivot calibration applied from {diagnostics['samples']} poses: "
        f"the pivot held to {diagnostics['residual_rms'] * 1000:.1f} mm while "
        f"the headset moved {diagnostics['headset_rms'] * 1000:.1f} mm.\n"
        f"  neck_pivot_offset: [{formatted}]\n" + kept,
        flush=True,
    )
    return {
        "type": "calibration-result",
        "accepted": True,
        "offset": [float(component) for component in offset],
        "samples": int(diagnostics["samples"]),
        "residual_mm": float(diagnostics["residual_rms"]) * 1000.0,
        "headset_mm": float(diagnostics["headset_rms"]) * 1000.0,
        "saved_to": saved_to,
    }


def _save_neck_pivot_offset(offset):
    """Write an accepted offset where the next run reads it.

    The document nests the offset the way the view configuration file
    nests it, so the measurement can be pasted there as it stands.

    Returns the path written, or None if there was nowhere to write it or
    the write failed. A measurement that cannot be saved is still worth
    having for the session, so a failure is reported and nothing else.
    """
    if _NECK_PIVOT_FILE is None:
        return None
    document = {
        "pose": {
            "neck_pivot_offset": [round(float(component), 4) for component in offset]
        }
    }
    try:
        with open(_NECK_PIVOT_FILE, "w", encoding="utf-8") as output:
            output.write("# Measured by dora-openarm-webxr --calibration.\n")
            yaml.safe_dump(document, output)
    except OSError as error:
        print(f"cannot write {_NECK_PIVOT_FILE}: {error}", flush=True)
        return None
    return str(_NECK_PIVOT_FILE)


def _read_neck_pivot_offset(path):
    """Read a measured offset back, or None if there is not one to read."""
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as input:
            document = yaml.safe_load(input)
    except FileNotFoundError:
        # Nothing measured yet, which is every first run.
        return None
    except (OSError, yaml.YAMLError) as error:
        print(f"cannot read {path}: {error}", flush=True)
        return None
    pose = document.get("pose") if isinstance(document, dict) else None
    offset = pose.get("neck_pivot_offset") if isinstance(pose, dict) else None
    if offset is None:
        return None
    return np.array(offset, dtype=np.float32).reshape(3)


def _configure_neck_pivot(pose_configuration, path):
    """Settle which neck pivot offset the session starts with.

    The built-in estimate is the floor, the view configuration file
    overrides it, and a measured one overrides that: of the three it is
    the only one that came from the operator wearing the headset. Said
    out loud rather than applied quietly, because a file left behind by
    someone else's head would otherwise be invisible.
    """
    global _NECK_PIVOT_OFFSET
    configured = pose_configuration.get("neck_pivot_offset")
    if configured is not None:
        _NECK_PIVOT_OFFSET = np.array(configured, dtype=np.float32).reshape(3)

    measured = _read_neck_pivot_offset(path)
    if measured is not None:
        _NECK_PIVOT_OFFSET = measured
        formatted = ", ".join(f"{component:.3f}" for component in measured)
        print(f"neck pivot offset read from {path}: [{formatted}]", flush=True)


def _check_pivot_offset(offset, diagnostics):
    """Why a fitted neck pivot offset is not worth applying, or None.

    The fit itself reports what the run could and could not see; the
    thresholds live here, next to the operator who has to be told what to
    do differently.
    """
    samples = diagnostics["samples"]
    if samples < _CALIBRATION_MIN_SAMPLES:
        return (
            f"only {samples} headset poses came in; "
            "hold Y down for the whole head turn, not just a tap"
        )

    observability = diagnostics["observability"]
    if observability[0] < _CALIBRATION_MIN_OBSERVABILITY:
        axis = int(np.argmax(np.abs(diagnostics["observability_axes"][0])))
        return (
            f"the head did not turn enough to see the {_OFFSET_AXIS_NAMES[axis]} "
            f"offset; turn it {_PINNING_MOTION[axis]} as well"
        )

    residual = diagnostics["residual_rms"]
    if residual > _CALIBRATION_MAX_RESIDUAL:
        return (
            f"the pivot still moved {residual * 1000:.0f} mm over the run; "
            "hold the body still and turn only the head"
        )

    lateral, vertical, fore_aft = offset
    if (
        abs(lateral) > _CALIBRATION_MAX_LATERAL
        or not -_CALIBRATION_MAX_VERTICAL <= vertical <= 0.0
        or not 0.0 <= fore_aft <= _CALIBRATION_MAX_FORE_AFT
    ):
        return (
            f"the fitted offset [{lateral:.3f}, {vertical:.3f}, {fore_aft:.3f}] "
            "is not where a neck is: it belongs on the midline, below the eyes "
            "and behind them"
        )
    return None


app = FastAPI()


def _map_trigger_to_gripper(trigger: float, side: str) -> float:
    """Trigger 0.0~1.0 -> gripper angle."""
    if side == "right":
        return (-1.57 / 2.0) * (1.0 - trigger)  # 0->-1.57, 1->0
    else:
        return (1.57 / 2.0) * (1.0 - trigger)  # 0-> 1.57, 1->0


def _adjust_pose(pose, reference, smoother, smoother_time):
    """Convert WebXR style pose to our style.

    ``pose`` and ``reference`` (the viewer pose) are in the same
    world-fixed reference space. Only the position is made relative to
    the viewer, by subtracting it in the world axes. The viewer
    rotation is never applied: turning the head must not move the
    target. The controller orientation is passed through as its world
    orientation for the same reason.

    What is subtracted is the neck pivot rather than the headset itself,
    since the headset orbits that pivot as the head turns and would
    otherwise carry the arc into the target. The viewer rotation is used
    to place the pivot, which is not the same as applying it to the
    hand: it only says which way the operator is facing, so the point
    behind their face can be found.

    WebXR style:
      * right-handed
      * {
          x: X, (meter)
          y: Y, (meter)
          z: Z, (meter)
          qx: QX, (quaternion)
          qy: QY, (quaternion)
          qz: QZ, (quaternion)
          qw: QW, (quaternion)
        }

    Our style:
      * right-handed
      * [x, y, z, qw, qx, qy, qz]
    """
    reference_rotation = Rotation.from_quat(
        [reference["qx"], reference["qy"], reference["qz"], reference["qw"]]
    )
    # Turned into world axes, so "behind the face" follows where the head faces.
    pivot = np.array(
        [reference["x"], reference["y"], reference["z"]], dtype=np.float32
    ) + reference_rotation.apply(_NECK_PIVOT_OFFSET)
    position = (
        np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32) - pivot
    ).astype(np.float32)
    rotation = Rotation.from_quat([pose["qx"], pose["qy"], pose["qz"], pose["qw"]])

    position = _ROBOT_ROTATION.apply(position) + _FRAME_OFFSET_CELL
    rotation = _ROBOT_ROTATION * rotation * _CONTROLLER_TO_EE
    quaternion = rotation.as_quat()

    adjusted_pose = np.array(
        [
            position[0],  # x
            position[1],  # y
            position[2],  # z
            quaternion[3],  # qw
            quaternion[0],  # qx
            quaternion[1],  # qy
            quaternion[2],  # qz
        ],
        dtype=np.float32,
    )
    return pa.array(smoother.smooth(smoother_time, adjusted_pose))


_POSE_STRUCT_TYPE = pa.struct({"pose": pa.list_(pa.float32())})


def _build_pose_output(pose: np.ndarray) -> pa.Array:
    """Wrap a pose array as a length-1 StructArray: [{"pose": [...]}]."""
    return pa.array([{"pose": pose}], type=_POSE_STRUCT_TYPE)


def _build_head_pose_output(pose: dict) -> pa.Array:
    """Wrap the headset pose, in our style: [x, y, z, qw, qx, qy, qz].

    Unrotated, unlike the hands. `_adjust_pose` applies `_ROBOT_ROTATION` and a
    z+90 fix to put a controller where the arm expects its end effector; both
    are arm conventions and neither means anything on a head. Consumers map the
    WebXR frame (x right, y up, -z forward) into their own body frame, so the
    rig's wiring stays in the consumer's config rather than baked in here.

    Not made relative either: the hand positions are relative to this pose, so
    subtracting it from itself would leave nothing to read the head from.

    Not smoothed either: the One Euro smoothers are per-hand and stateful, and
    a neck has its own rate limiting downstream.
    """
    head_pose = np.array(
        [
            pose["x"],
            pose["y"],
            pose["z"],
            pose["qw"],
            pose["qx"],
            pose["qy"],
            pose["qz"],
        ],
        dtype=np.float32,
    )
    return _build_pose_output(head_pose)


class _ConnectionState:
    """Per-session state carried across the frame messages.

    The smoothers and the pivot calibration are stateful, and a new
    session must not inherit the last one's history, so ``session-start``
    replaces this wholesale rather than clearing it piece by piece.
    """

    def __init__(self):
        """Start with fresh smoothers and no frame seen yet."""
        self.smoothers = {
            "right": OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5),
            "left": OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5),
        }
        self.pivot_calibration = _PivotCalibration(enabled=_CALIBRATION_ENABLED)
        self.last_sequence = -1


def _process_frame(response, state):
    """Publish dora outputs for one frame message.

    Frames ride an unordered channel that never retransmits, so one can
    arrive after a newer one has already been published. Publishing it
    would drag the arms back to a pose the operator has left, so a frame
    whose ``sequence`` is not newer than the newest processed one is
    dropped. A frame without ``sequence`` is always processed.
    """
    sequence = response.get("sequence")
    if sequence is not None:
        if sequence <= state.last_sequence:
            return
        state.last_sequence = sequence
    metadata = {"timestamp": time.time_ns()}
    smoother_time = time.perf_counter()
    # An absent button is a released one, so a controller that falls
    # asleep mid-run cannot leave the hands stopped. The client only
    # sends the buttons on the profiles it knows, and on those it sends
    # them every frame.
    samples = state.pivot_calibration.update(bool(response.get("button_y")))
    if samples is not None:
        # Back to the headset as well as the node's output: the operator
        # cannot see the output while wearing one.
        webrtc_server.send_control(_apply_pivot_calibration(samples))
    node.send_output(
        "vr_receive_times",
        pa.array([metadata["timestamp"]], type=pa.int64()),
        metadata,
    )
    reference = response.get("pose_reference")
    if reference:
        node.send_output(
            "pose_reference",
            _build_head_pose_output(reference),
            metadata,
        )
        state.pivot_calibration.add(reference)
    for button in ["a", "b", "x", "y"]:
        name = f"button_{button}"
        if name in response:
            node.send_output(
                name,
                pa.array([bool(response[name])], type=pa.bool_()),
                metadata,
            )
    for side in ["right", "left"]:
        pose = f"pose_{side}"
        trigger = f"trigger_{side}"
        # The hands stop while a run is under way. Turning the head moves
        # the target by the very arc being measured, and the operator is
        # shaking their head, not reaching. Without --calibration no run
        # is ever under way.
        if (
            pose in response
            and trigger in response
            and reference
            and not state.pivot_calibration.collecting
        ):
            smoother = state.smoothers[side]
            adjusted_pose = _adjust_pose(
                response[pose], reference, smoother, smoother_time
            )
            gripper_angle = _map_trigger_to_gripper(response[trigger], side)
            gripper_array = np.array([gripper_angle], dtype=np.float32)
            pose_with_gripper = np.concatenate([adjusted_pose, gripper_array])
            node.send_output(
                pose,
                _build_pose_output(pose_with_gripper),
                metadata,
            )
        if trigger in response:
            node.send_output(
                trigger,
                pa.array([response[trigger]], type=pa.float32()),
                metadata,
            )
        grip = f"grip_{side}"
        if grip in response:
            node.send_output(
                grip,
                pa.array([response[grip]], type=pa.float32()),
                metadata,
            )
        joystick = f"joystick_{side}"
        if joystick in response:
            axes = response[joystick]
            # The xr-standard mapping reserves the first axis pair for
            # the touchpad and the second for the thumbstick, so the
            # stick is axes[2:4] wherever the controller has one; a
            # device with only a touchpad reports that pair alone and it
            # takes its place. The y sign is flipped to keep the
            # convention the Unity sender published, which is the one the
            # downstream nodes were written against.
            x, y = (axes[2], axes[3]) if len(axes) >= 4 else axes[:2]
            y = -y
            node.send_output(
                f"joystick_x_{side}",
                pa.array([x], type=pa.float32()),
                metadata,
            )
            node.send_output(
                f"joystick_y_{side}",
                pa.array([y], type=pa.float32()),
                metadata,
            )


def _on_frame(payload):
    """Hand one frame message from the transport to the pose pipeline."""
    _process_frame(payload, _state)


def _on_session_start():
    """Begin a session: fresh smoothers, fresh frame numbering."""
    global _state
    _state = _ConnectionState()
    node.send_output("status", pa.array(["ready"]), {"timestamp": time.time_ns()})


@app.post("/offer")
async def _offer_endpoint(offer: dict):
    """Answer a browser's SDP offer.

    Only reachable when this node serves the page itself. In WebRTC-only
    mode the offer arrives at startup instead and no HTTP server runs.
    """
    return {"sdp": await webrtc_server.answer(offer["sdp"]), "type": "answer"}


base_dir = os.path.dirname(__file__)
app.mount("/", StaticFiles(directory=f"{base_dir}/static", html=True), name="static")


def _serving():
    """Whether the node should keep running, in either signaling mode.

    In WebRTC-only mode the one browser leaving ends the session: with
    no HTTP server, no other browser can ever take its place.
    """
    if not _running:
        return False
    return webrtc_server is None or webrtc_server.running


def _stop():
    """Stop both halves, whichever noticed first."""
    global _running
    _running = False
    if server is not None:
        server.should_exit = True


async def _main_uvicorn():
    await server.serve()
    _stop()


async def _main_dora():
    while _serving():
        if node.is_empty():
            await asyncio.sleep(0.001)
            continue
        event = node.next()
        if event["type"] == "STOP":
            break
        video.handle_event(event)
    _stop()


async def _main_hosted():
    """Serve the page over HTTPS and answer offers posted to it."""
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.tls_key_file,
        ssl_certfile=args.tls_certificate_file,
        log_level="info",
    )
    global server
    server = uvicorn.Server(config)

    task_uvicorn = asyncio.create_task(_main_uvicorn())
    task_dora = asyncio.create_task(_main_dora())

    await task_uvicorn
    await task_dora


async def _main_webrtc_only():
    """Run the single peer whose offer was handed in at startup.

    This is a one-shot connection: the offer is fixed at startup, so the
    node runs that one peer for its whole life and reconnecting means
    restarting the node.
    """
    task_dora = asyncio.create_task(_main_dora())
    try:
        await webrtc_server.negotiate_oneshot(
            args.offer,
            args.answer_host,
            args.answer_port,
            args.connect_timeout,
        )
    except (OSError, RuntimeError) as error:
        # Nobody applied the answer, or the media path never came up.
        # Exiting lets a supervisor restart us for a fresh offer.
        print(f"WebRTC-only mode failed: {error}", flush=True)
        _stop()
    await task_dora
    await webrtc_server.close()


async def _main_async():
    global webrtc_server, _state
    _state = _ConnectionState()
    webrtc_server = webrtc.WebRTCServer(
        on_frame=_on_frame,
        on_session_start=_on_session_start,
        calibration_enabled=_CALIBRATION_ENABLED,
    )
    if args.offer:
        await _main_webrtc_only()
    else:
        await _main_hosted()


def _answer_port_default():
    """Return the answer port from ANSWER_PORT, or None if it is unset."""
    raw = os.getenv("ANSWER_PORT")
    if raw is None:
        return None
    return int(raw)


def _environment_flag(name):
    """Read an on/off option's environment variable.

    Anything but a plain negative turns it on, so that a dataflow YAML
    can carry ``CALIBRATION: false`` and have it mean off.
    """
    return os.getenv(name, "").strip().lower() not in ("", "0", "false", "no")


def main():
    """Run WebXR server."""
    parser = argparse.ArgumentParser(description="WebXR server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8443")),
        help="Server port (default: 8443)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--tls-certificate-file",
        type=pathlib.Path,
        default=os.getenv("TLS_CERTIFICATE_FILE"),
        help="TLS certificate file. Required unless --offer is given",
    )
    parser.add_argument(
        "--tls-key-file",
        type=pathlib.Path,
        default=os.getenv("TLS_KEY_FILE"),
        help=(
            "TLS key file for the certificate file. Required unless --offer is given"
        ),
    )
    parser.add_argument(
        "--offer",
        type=str,
        default=os.getenv("OFFER"),
        help=(
            "Browser SDP offer, which runs the node as a pure WebRTC peer "
            "with no HTTP server: another service hosts the page and brokers "
            "signaling, so no TLS certificate is needed. --host, --port and "
            "the TLS options are ignored"
        ),
    )
    parser.add_argument(
        "--answer-host",
        type=str,
        default=os.getenv("ANSWER_HOST", "127.0.0.1"),
        help="Host to write the SDP answer to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--answer-port",
        type=int,
        default=_answer_port_default(),
        help="Port to write the SDP answer to",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=float(os.getenv("CONNECT_TIMEOUT", "60")),
        help="Seconds to wait for the browser to connect (default: 60)",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        default=_environment_flag("CALIBRATION"),
        help="Measure the neck pivot with the Y button (default: off)",
    )
    parser.add_argument(
        "--neck-pivot-file",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("NECK_PIVOT_FILE", "neck_pivot.yaml")),
        help=(
            "YAML file a measured neck pivot offset is written to and read "
            "back from (default: neck_pivot.yaml)"
        ),
    )
    video.add_arguments(parser)

    global args
    args = parser.parse_args()

    # argparse cannot express "required unless another option is given",
    # and WebXR only runs on an HTTPS page, so the certificate is required
    # of every mode that serves the page itself.
    if args.offer:
        if args.answer_port is None:
            parser.error("--answer-port is required with --offer")
    elif args.tls_certificate_file is None or args.tls_key_file is None:
        parser.error(
            "--tls-certificate-file and --tls-key-file are required unless "
            "--offer is given"
        )

    video.configure(args)

    global _CALIBRATION_ENABLED, _NECK_PIVOT_FILE
    _CALIBRATION_ENABLED = args.calibration
    _NECK_PIVOT_FILE = args.neck_pivot_file

    # Read once at startup; restart the dataflow to apply a change.
    pose_configuration = video.view_configuration().get("pose") or {}

    frame_offset = pose_configuration.get("frame_offset")
    if frame_offset is not None:
        global _FRAME_OFFSET_CELL
        _FRAME_OFFSET_CELL = np.array(frame_offset, dtype=np.float32).reshape(3)

    _configure_neck_pivot(pose_configuration, args.neck_pivot_file)

    global node
    node = dora.Node()

    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
