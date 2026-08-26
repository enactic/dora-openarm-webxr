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
import fractions
import json
import time

import av
import numpy as np
import pyarrow as pa
import pytest
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)

from dora_openarm_webxr import video, webrtc


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # No STUN: the tests connect over loopback, and an unreachable STUN
    # server would only slow gathering down.
    monkeypatch.setattr(webrtc, "ICE_SERVERS", [])
    video.reset()
    yield
    video.reset()


def _encode_jpeg() -> bytes:
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = 64
    codec.height = 48
    codec.pix_fmt = "yuvj420p"
    codec.time_base = fractions.Fraction(1, 30)
    image = np.full((48, 64, 3), 128, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    packets = codec.encode(frame.reformat(format="yuvj420p"))
    packets += codec.encode(None)
    return b"".join(bytes(packet) for packet in packets)


def _push_jpeg(jpeg: bytes) -> None:
    # The public path a camera frame takes into this node: a dora INPUT
    # event whose value is the JPEG bytes as a uint8 array.
    video.handle_event(
        {
            "type": "INPUT",
            "id": "camera_head_right",
            "value": pa.array(np.frombuffer(jpeg, dtype=np.uint8)),
        }
    )


async def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.05)


def _browser_peer(received: dict) -> tuple:
    # An in-process stand-in for connection.js: the "xr" channel, the
    # VIDEO_TRANSCEIVERS recvonly video slots, and handlers that collect
    # the "control" channel, its messages by type, and the first decoded
    # video frame.
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    xr = pc.createDataChannel("xr", ordered=False, maxRetransmits=0)

    @pc.on("datachannel")
    def on_datachannel(channel):
        if channel.label != "control":
            return
        received["control"] = channel

        @channel.on("message")
        def on_message(message):
            payload = json.loads(message)
            received[payload["type"]] = payload

    @pc.on("track")
    def on_track(track):
        async def read_one():
            received["frame"] = await track.recv()

        asyncio.ensure_future(read_one())

    for _ in range(webrtc.VIDEO_TRANSCEIVERS):
        pc.addTransceiver("video", direction="recvonly")
    return pc, xr


async def _connect(server, pc, received):
    # The offer/answer exchange and the configuration push: the preamble
    # every hosted-mode test shares.
    await pc.setLocalDescription(await pc.createOffer())
    answer = await server.answer(pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
    await _wait_for(lambda: "configuration" in received)


def test_answer_session():
    # Hosted mode's transport, minus the HTTP wrapper around answer():
    # the configuration is pushed on "control", session-start and frame
    # messages reach the callbacks, and a pushed JPEG comes back as
    # decoded video.
    asyncio.run(_run_answer_session())


async def _run_answer_session():
    frames = []
    sessions = []
    server = webrtc.WebRTCServer(
        on_frame=frames.append,
        on_session_start=lambda: sessions.append(True),
        calibration_enabled=True,
    )
    received: dict = {}
    pc, xr = _browser_peer(received)
    try:
        await pc.setLocalDescription(await pc.createOffer())
        answer = await server.answer(pc.localDescription.sdp)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))

        # The node speaks first: the configuration rides "control" as
        # soon as it opens, so a page the node never served still knows
        # how to draw itself.
        await _wait_for(lambda: "configuration" in received)
        configuration = received["configuration"]
        assert configuration["eyes"] == ["right"]  # The default mono view.
        assert configuration["calibration"] is True
        assert configuration["view_configuration"]["view"] == "mono"

        received["control"].send(json.dumps({"type": "session-start"}))
        await _wait_for(lambda: sessions)

        await _wait_for(lambda: xr.readyState == "open")
        xr.send(json.dumps({"type": "frame", "sequence": 1, "button_a": True}))
        xr.send("not json")
        xr.send(json.dumps(["not", "an", "object"]))
        await _wait_for(lambda: frames)
        assert frames[0]["button_a"] is True

        jpeg = _encode_jpeg()
        deadline = time.monotonic() + 10.0
        while "frame" not in received:
            if time.monotonic() > deadline:
                raise TimeoutError("no video frame arrived")
            _push_jpeg(jpeg)
            await asyncio.sleep(0.05)
        assert received["frame"].width == 64
        assert received["frame"].height == 48
    finally:
        await pc.close()
        await server.close()


def test_calibration_result_broadcast():
    # What came of a calibration run goes back over "control": the
    # operator cannot see the node's output while wearing a headset.
    asyncio.run(_run_calibration_result())


async def _run_calibration_result():
    server = webrtc.WebRTCServer(
        on_frame=lambda payload: None,
        on_session_start=lambda: None,
        calibration_enabled=True,
    )
    received: dict = {}
    pc, _xr = _browser_peer(received)
    try:
        await _connect(server, pc, received)

        server.send_control({"type": "calibration-result", "accepted": False})
        await _wait_for(lambda: "calibration-result" in received)
        assert received["calibration-result"]["accepted"] is False
    finally:
        await pc.close()
        await server.close()


def test_answer_peer_cleanup():
    # Hosted mode outlives any one browser: a peer that fails or closes
    # is discarded and closed, so reloads do not stack dead video
    # encoders, while the server keeps serving for the next browser.
    server = webrtc.WebRTCServer(
        on_frame=lambda payload: None,
        on_session_start=lambda: None,
    )
    asyncio.run(_run_answer_peer_cleanup(server))


async def _run_answer_peer_cleanup(server):
    received: dict = {}
    pc, _xr = _browser_peer(received)
    try:
        await _connect(server, pc, received)
        assert server._pcs

        await pc.close()
        await _wait_for(lambda: not server._pcs)
        # Unlike WebRTC-only mode, the browser leaving does not end the
        # node: another browser can still post a fresh offer.
        assert server.running
    finally:
        await pc.close()
        await server.close()


def test_close_reaches_browser():
    # The node shutting down must reach the browser promptly: the page
    # ends its session on the control channel's close, not on an ICE
    # timeout tens of seconds after the node went quiet.
    server = webrtc.WebRTCServer(
        on_frame=lambda payload: None,
        on_session_start=lambda: None,
    )
    asyncio.run(_run_close_reaches_browser(server))


async def _run_close_reaches_browser(server):
    received: dict = {}
    pc, _xr = _browser_peer(received)
    try:
        await _connect(server, pc, received)

        control_closed = []
        received["control"].on("close", lambda: control_closed.append(True))
        await server.close()
        await _wait_for(lambda: control_closed, timeout=5.0)
    finally:
        await pc.close()
        await server.close()


def test_oneshot_signaling():
    # WebRTC-only mode: no HTTP server. An offer is handed to
    # negotiate_oneshot, the answer comes back over a TCP socket the
    # caller listens on, and once the browser applies it the "xr"
    # channel carries frames to on_frame as usual.
    frames = []
    server = webrtc.WebRTCServer(
        on_frame=frames.append,
        on_session_start=lambda: None,
    )
    asyncio.run(_run_oneshot(server, frames))
    assert frames[0]["sequence"] == 1


async def _relay(pc, server, connect_timeout: float):
    # negotiate_oneshot blocks until the peer connects, so relaying the
    # answer to the browser has to happen concurrently, the way a real
    # signaling broker would.
    answer: dict = {}

    async def handle_answer(reader, writer):
        answer["sdp"] = (await reader.read()).decode("utf-8")
        writer.close()

    tcp = await asyncio.start_server(handle_answer, "127.0.0.1", 0)
    host, port = tcp.sockets[0].getsockname()[:2]
    async with tcp:
        negotiate = asyncio.ensure_future(
            server.negotiate_oneshot(
                pc.localDescription.sdp, host, port, connect_timeout
            )
        )
        await _wait_for(lambda: "sdp" in answer)
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type="answer")
        )
        await negotiate


async def _run_oneshot(server, frames):
    received: dict = {}
    pc, xr = _browser_peer(received)
    try:
        await pc.setLocalDescription(await pc.createOffer())
        await _relay(pc, server, 10.0)

        await _wait_for(lambda: "configuration" in received)
        await _wait_for(lambda: xr.readyState == "open")
        xr.send(json.dumps({"type": "frame", "sequence": 1}))
        await _wait_for(lambda: frames)
    finally:
        await pc.close()
        await server.close()


def test_oneshot_disconnect():
    # Once the one browser of WebRTC-only mode goes away, the server
    # stops running so the node can exit: with no HTTP server, no other
    # browser can ever take its place.
    server = webrtc.WebRTCServer(
        on_frame=lambda payload: None,
        on_session_start=lambda: None,
    )
    asyncio.run(_run_oneshot_disconnect(server))


async def _run_oneshot_disconnect(server):
    received: dict = {}
    pc, _xr = _browser_peer(received)
    try:
        await pc.setLocalDescription(await pc.createOffer())
        await _relay(pc, server, 10.0)
        assert server.running

        await pc.close()
        await _wait_for(lambda: not server.running)
    finally:
        await pc.close()
        await server.close()


def test_oneshot_connect_timeout():
    # The answer is sent but the browser never applies it, so the peer
    # never connects: negotiate_oneshot gives up after the timeout and
    # raises, letting a supervisor restart the node for a fresh offer.
    server = webrtc.WebRTCServer(
        on_frame=lambda payload: None,
        on_session_start=lambda: None,
    )
    asyncio.run(_run_oneshot_timeout(server))


async def _run_oneshot_timeout(server):
    async def handle_answer(reader, writer):
        await reader.read()
        writer.close()

    tcp = await asyncio.start_server(handle_answer, "127.0.0.1", 0)
    host, port = tcp.sockets[0].getsockname()[:2]

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    try:
        async with tcp:
            pc.createDataChannel("xr")
            pc.addTransceiver("video", direction="recvonly")
            await pc.setLocalDescription(await pc.createOffer())
            with pytest.raises(RuntimeError):
                await server.negotiate_oneshot(pc.localDescription.sdp, host, port, 0.5)
    finally:
        await pc.close()
        await server.close()
