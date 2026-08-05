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

"""Head camera video downlink for the WebXR front-end.

Takes the per-eye JPEG images a splitter node has cut out of a
side-by-side stereo camera and forwards them to the VR device, so the
operator can see through the robot's head while teleoperating.

Frames go on their own WebSocket so they never delay the pose messages
that feed IK. How they are drawn is tuned in
``example/head_cam_view.yaml``.
"""

import argparse
import asyncio
import os
import pathlib
import yaml

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


# dora-rs input IDs mapped to the eye that the frame is rendered on.
CAMERA_INPUTS = {
    "camera_head_left": "left",
    "camera_head_right": "right",
}

# Each frame is sent as a binary WebSocket message prefixed with one
# byte identifying the eye, followed by the JPEG data.
EYE_PREFIX = {"left": b"\x00", "right": b"\x01"}

# Used when no --view-configuration-file is given.
DEFAULT_VIEW_CONFIGURATION: dict = {
    "session": {"mode": "immersive-ar"},
    "camera": {"horizontal_fov": 79.4, "vertical_fov": 50.1},
    "panel": {"distance": 1.0},
    "stereo": {"vertical_align": 0.0, "convergence": 0.0},
}

_frames: dict = {"left": None, "right": None}
# Incremented on every frame so that the video endpoint can tell a new
# frame from a repeated one and always send the most recent pair.
_sequences: dict = {"left": 0, "right": 0}
_frame_event = asyncio.Event()

_view_configuration_file: pathlib.Path | None = None
# Last good one, so saving a half-written file cannot break a session.
_view_configuration: dict = DEFAULT_VIEW_CONFIGURATION


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the head camera options to the node's argument parser."""
    parser.add_argument(
        "--view-configuration-file",
        type=pathlib.Path,
        default=os.getenv("VIEW_CONFIGURATION_FILE"),
        help="YAML file with the head camera stereo panel parameters",
    )


def configure(args: argparse.Namespace) -> None:
    """Remember the view configuration file if one was given."""
    global _view_configuration_file
    _view_configuration_file = getattr(args, "view_configuration_file", None)


def _read_view_configuration() -> dict:
    """Read the view configuration file.

    Read per request, not once at startup, so the panel can be tuned by
    editing the file and reloading the page in the VR device.
    """
    global _view_configuration
    if _view_configuration_file is None:
        return _view_configuration
    try:
        with open(_view_configuration_file, encoding="utf-8") as input:
            _view_configuration = yaml.safe_load(input)
    except (OSError, yaml.YAMLError) as error:
        print(f"cannot read {_view_configuration_file}: {error}", flush=True)
    return _view_configuration


def view_configuration() -> dict:
    """Return the view configuration, reading the file when one is set."""
    return _read_view_configuration()


def handle_event(event) -> bool:
    """Store a head camera frame. Return whether the event was ours."""
    if event["type"] != "INPUT" or event["id"] not in CAMERA_INPUTS:
        return False
    eye = CAMERA_INPUTS[event["id"]]
    # The splitter sends JPEG data as a uint8 array.
    _frames[eye] = event["value"].to_numpy(zero_copy_only=False).tobytes()
    _sequences[eye] += 1
    _frame_event.set()
    return True


def register_routes(app: FastAPI, should_exit) -> None:
    """Register the head camera routes on the node's Web application.

    ``should_exit`` is a callable so this module need not know how the
    node shuts its server down.
    """

    @app.get("/view_configuration")
    async def _view_configuration_endpoint():
        """Serve the stereo panel parameters to the WebXR front-end."""
        return _read_view_configuration()

    @app.websocket("/video")
    async def _video_endpoint(websocket: WebSocket):
        await websocket.accept()
        sent = {"left": -1, "right": -1}
        try:
            while not should_exit():
                try:
                    await asyncio.wait_for(_frame_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Loop so shutdown is noticed.
                    continue
                _frame_event.clear()
                if any(_frames[eye] is None for eye in EYE_PREFIX):
                    continue
                if all(_sequences[eye] == sent[eye] for eye in EYE_PREFIX):
                    continue
                # Together, so the eyes never show different frames.
                for eye, prefix in EYE_PREFIX.items():
                    sent[eye] = _sequences[eye]
                    await websocket.send_bytes(prefix + _frames[eye])
            await websocket.close()
        except WebSocketDisconnect:
            pass
