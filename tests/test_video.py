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

import argparse
import asyncio

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_webxr import video


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Register the module globals for restore before any test changes
    # them, then start every test from a clean slate.
    monkeypatch.setattr(video, "_view_configuration", video.DEFAULT_VIEW_CONFIGURATION)
    video.reset()
    yield
    video.reset()


def _configure_view(tmp_path, view: str) -> None:
    path = tmp_path / "view.yaml"
    path.write_text(f"view: {view}\n", encoding="utf-8")
    video.configure(argparse.Namespace(view_configuration_file=path))


def _push(eye: str, payload: bytes) -> None:
    video.handle_event(
        {
            "type": "INPUT",
            "id": f"camera_head_{eye}",
            "value": pa.array(np.frombuffer(payload, dtype=np.uint8)),
        }
    )


def test_eyes_default_mono():
    assert video.eyes() == ["right"]


def test_eyes_stereo(tmp_path):
    _configure_view(tmp_path, "stereo")
    assert video.eyes() == ["left", "right"]


def test_eyes_none(tmp_path):
    _configure_view(tmp_path, "none")
    assert video.eyes() == []


def test_handle_event_ignores_other_inputs():
    assert not video.handle_event({"type": "INPUT", "id": "pose_right"})
    assert not video.handle_event({"type": "STOP"})


def test_wait_next_skips_to_newest():
    # A slow encoder must skip ahead to the newest frame rather than
    # build a queue: only the newest one is worth showing.
    asyncio.run(_run_wait_next_skips())


async def _run_wait_next_skips():
    _push("right", b"one")
    _push("right", b"two")
    payload, sequence = await video.wait_next("right", 0)
    assert payload == b"two"
    assert sequence == 2


def test_wait_next_blocks_until_new_frame():
    asyncio.run(_run_wait_next_blocks())


async def _run_wait_next_blocks():
    _push("right", b"one")
    payload, sequence = await video.wait_next("right", 0)
    assert payload == b"one"

    waiter = asyncio.ensure_future(video.wait_next("right", sequence))
    await asyncio.sleep(0.05)
    assert not waiter.done()

    _push("right", b"two")
    payload, sequence = await asyncio.wait_for(waiter, 5.0)
    assert payload == b"two"
    assert sequence == 2


def test_wait_next_eyes_are_independent():
    asyncio.run(_run_wait_next_eyes())


async def _run_wait_next_eyes():
    _push("left", b"left-frame")
    waiter = asyncio.ensure_future(video.wait_next("right", 0))
    await asyncio.sleep(0.05)
    # A left frame must not wake the right eye's track.
    assert not waiter.done()

    _push("right", b"right-frame")
    payload, _sequence = await asyncio.wait_for(waiter, 5.0)
    assert payload == b"right-frame"
