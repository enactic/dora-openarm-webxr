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

This dora-rs node serves the WebXR front-end over HTTPS and accepts a
WebSocket connection from a VR device such as Meta Quest 3 or PICO 4.
For each frame received from the device, it converts the controller
pose from WebXR coordinates into the OpenArm workspace, smooths it with
a One Euro filter, and publishes the pose, trigger, joystick and button
state as dora-rs outputs.

The published poses are expressed in the scene's ``arm_origin`` site
frame (chest-level origin between the arms), not in world coordinates.
Downstream IK interprets targets in the same frame.

The Web server and the dora-rs event loop run concurrently in a single
asyncio event loop; the server shuts down when the dora-rs node
receives a ``STOP`` event.
"""

import argparse
import asyncio
import dora
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import json
import numpy as np
import os
import pathlib
import pyarrow as pa
from scipy.spatial.transform import Rotation
import time
import uvicorn

from .smoothing import OneEuroPoseSmoother
from . import video

args = None
node = None
server = None


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

# Relative pose is computed from viewer.
# We need to move it to OpenArm position.
#
# Neutral hand position relative to the arm_origin site (chest level).
# Overridden by ``pose: frame_offset`` in the view configuration file.
_FRAME_OFFSET_CELL: np.ndarray = np.array([-0.085, 0, -0.14], dtype=np.float32)


# When the operator tilts the head, the viewer reference space tilts
# with it and the controller poses rotate even though the hands did not
# move. We cancel the head pitch and roll every frame and keep only the
# yaw, so the hand to robot mapping stays gravity-aligned.
def _head_tilt_compensation(head_rotation: Rotation) -> Rotation | None:
    """Pitch/roll residual ``yaw(H)^-1 * H`` of the headset rotation.

    ``None`` when the head looks nearly straight up or down, where the
    yaw is undefined; the caller keeps the previous compensation.
    """
    forward = head_rotation.apply([0.0, 0.0, -1.0])
    # Within about 0.06 degrees of straight up or down the yaw estimate
    # is dominated by tracking noise.
    if np.hypot(forward[0], forward[2]) < 1e-3:
        return None
    yaw = np.arctan2(-forward[0], -forward[2])
    return Rotation.from_euler("y", yaw).inv() * head_rotation


# Bound how fast the compensation may change so that the yaw branch
# flip at the gaze pole, a pose_head dropout snapping back, or yaw
# noise near vertical can never step the arm target. The residual
# carries no yaw, so real head pitch and roll stay far below this.
_MAX_COMPENSATION_STEP = np.deg2rad(4.0)


def _limit_compensation_step(previous: Rotation, target: Rotation) -> Rotation:
    """Rotate ``previous`` toward ``target`` by at most the step limit."""
    delta = (target * previous.inv()).as_rotvec()
    angle = np.linalg.norm(delta)
    if angle <= _MAX_COMPENSATION_STEP:
        return target
    return Rotation.from_rotvec(delta * (_MAX_COMPENSATION_STEP / angle)) * previous


app = FastAPI()


def _map_trigger_to_gripper(trigger: float, side: str) -> float:
    """Trigger 0.0~1.0 -> gripper angle."""
    if side == "right":
        return (-1.57 / 2.0) * (1.0 - trigger)  # 0->-1.57, 1->0
    else:
        return (1.57 / 2.0) * (1.0 - trigger)  # 0-> 1.57, 1->0


def _adjust_pose(pose, compensation, smoother, smoother_time):
    """Convert WebXR style pose to our style.

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

    ``compensation`` is the head tilt residual from
    ``_head_tilt_compensation``, applied in viewer space before the
    robot-frame transform.
    """
    position = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32)
    position = compensation.apply(position)
    position = _ROBOT_ROTATION.apply(position) + _FRAME_OFFSET_CELL
    rotation = Rotation.from_quat([pose["qx"], pose["qy"], pose["qz"], pose["qw"]])
    # TODO: Add a comment why we need this
    rotation_fix = Rotation.from_euler("z", 90, degrees=True)
    rotation = _ROBOT_ROTATION * compensation * rotation * rotation_fix
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


@app.websocket("/websocket")
async def _websocket_endpoint(websocket: WebSocket):
    smoothers = {
        "right": OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5),
        "left": OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5),
    }
    compensation = Rotation.identity()

    await websocket.accept()
    try:
        while not server.should_exit:
            data = await websocket.receive_text()
            response = json.loads(data)
            type = response["type"]
            metadata = {"timestamp": time.time_ns()}
            if type == "session-start":
                node.send_output("status", pa.array(["ready"]), metadata)
            elif type == "frame":
                smoother_time = time.perf_counter()
                node.send_output(
                    "vr_receive_times",
                    pa.array([metadata["timestamp"]], type=pa.int64()),
                    metadata,
                )
                for button in ["a", "b", "x", "y"]:
                    name = f"button_{button}"
                    if name in response:
                        node.send_output(
                            name,
                            pa.array([bool(response[name])], type=pa.bool_()),
                            metadata,
                        )
                head = response.get("pose_head")
                if head is not None:
                    head_rotation = Rotation.from_quat(
                        [head["qx"], head["qy"], head["qz"], head["qw"]]
                    )
                    new_compensation = _head_tilt_compensation(head_rotation)
                    if new_compensation is not None:
                        compensation = _limit_compensation_step(
                            compensation, new_compensation
                        )
                for side in ["right", "left"]:
                    pose = f"pose_{side}"
                    trigger = f"trigger_{side}"
                    if pose in response and trigger in response:
                        smoother = smoothers[side]
                        adjusted_pose = _adjust_pose(
                            response[pose], compensation, smoother, smoother_time
                        )
                        gripper_angle = _map_trigger_to_gripper(response[trigger], side)
                        gripper_array = np.array([gripper_angle], dtype=np.float32)
                        pose_with_gripper = np.concatenate(
                            [adjusted_pose, gripper_array]
                        )
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
                    joystick = f"joystick_{side}"
                    if joystick in response:
                        axes = response[joystick]
                        x = axes[1] - axes[3]
                        y = axes[2] - axes[0]
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
        await websocket.close()
    except WebSocketDisconnect:
        pass


# The head camera routes are registered before the static files are
# mounted on "/" because the mount matches every remaining path.
video.register_routes(app, lambda: server.should_exit)


base_dir = os.path.dirname(__file__)
app.mount("/", StaticFiles(directory=f"{base_dir}/static", html=True), name="static")


async def _main_uvicorn():
    await server.serve()


async def _main_dora():
    while not server.should_exit:
        if node.is_empty():
            await asyncio.sleep(0.001)
            continue
        event = node.next()
        if event["type"] == "STOP":
            break
        video.handle_event(event)
    server.should_exit = True


async def _main_async():
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
    tls_certificate_file_default = os.getenv("TLS_CERTIFICATE_FILE")
    parser.add_argument(
        "--tls-certificate-file",
        type=pathlib.Path,
        default=tls_certificate_file_default,
        required=tls_certificate_file_default is None,
        help="TLS certificate file",
    )
    tls_key_file_default = os.getenv("TLS_KEY_FILE")
    parser.add_argument(
        "--tls-key-file",
        type=pathlib.Path,
        default=tls_key_file_default,
        required=tls_key_file_default is None,
        help="TLS key file for the certificate file",
    )
    video.add_arguments(parser)

    global args
    args = parser.parse_args()

    video.configure(args)

    # Read once at startup; restart the dataflow to apply a change.
    frame_offset = (video.view_configuration().get("pose") or {}).get("frame_offset")
    if frame_offset is not None:
        global _FRAME_OFFSET_CELL
        _FRAME_OFFSET_CELL = np.array(frame_offset, dtype=np.float32).reshape(3)

    global node
    node = dora.Node()

    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
