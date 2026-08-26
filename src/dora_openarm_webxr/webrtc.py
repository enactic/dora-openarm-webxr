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

"""WebXR teleoperation over WebRTC.

Everything the VR device and the node say to each other rides one peer
connection, so the page can be served by anyone -- this node over HTTPS
during development, or a hosted service in production. That is the whole
point: WebRTC authenticates itself with a self-signed certificate and an
SDP fingerprint, so a node reached this way needs no certificate of its
own and no HTTPS server.

Two data channels, split by what they can afford to lose:

* ``xr`` -- opened by the browser, **unordered and never retransmitted**.
  Carries the ``frame`` messages at the WebXR animation frame rate
  (72-120 Hz on a Quest 3). Only the newest pose is worth anything, so a
  lost frame is dropped rather than retransmitted; the next one
  supersedes it. Frames carry a ``sequence`` so the receiver can drop the
  stale ones an unordered channel occasionally delivers late.
* ``control`` -- opened by this node, reliable and ordered. Carries the
  things that must not be lost: the node pushes the view configuration
  and the calibration flag once on open, the browser sends
  ``session-start``, and the node sends back what came of a calibration
  run. Pushing the configuration is what lets a page this node never
  served still know how to draw itself.

Camera frames leave on WebRTC video tracks, one per eye, in the order
:func:`.video.eyes` gives. The eye order rides the ``control`` channel's
configuration message rather than being implied by track order, because
the mono view sends one track and the browser must not have to guess
which eye it is. Both tracks are stamped from one clock so the receiver
can line the eyes up; two clocks would let the eyes drift apart, which
in a headset is worse than either eye being late.
"""

import asyncio
import fractions
import json
import time

import av
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import VideoStreamTrack

from . import video

# RTP video clock; pts for outgoing frames are expressed in this rate.
_CLOCK_RATE = 90_000

# Public so the front-end and the tests agree on one number: the browser
# always negotiates this many recvonly video transceivers, whatever the
# view draws, because the one-shot signaling here cannot renegotiate
# later and the browser does not know the view when it makes the offer.
VIDEO_TRANSCEIVERS = 2

_STUN_URL = "stun:stun.cloudflare.com:3478"

# The ICE servers every peer is built with. A public STUN server is
# enough for the hosted case, where the page and the node sit on
# different networks; on one LAN the host candidates alone connect.
ICE_SERVERS = [RTCIceServer(urls=[_STUN_URL])]


class _SharedClock:
    """One timeline for every eye on a connection.

    Each eye is encoded on its own track, and tracks are paced
    independently, so nothing else keeps them together. Stamping both
    from one clock gives the receiver what it needs to present them as
    one moment.
    """

    def __init__(self) -> None:
        """Start unset; the first frame of either eye fixes the origin."""
        self._t0: float | None = None

    def timestamp(self) -> int:
        """Return the current time in RTP clock ticks since the origin."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        return int((now - self._t0) * _CLOCK_RATE)


class _EyeVideoTrack(VideoStreamTrack):
    """Video track that decodes one eye's JPEG frames on demand."""

    def __init__(self, eye: str, clock: _SharedClock) -> None:
        """Attach to an eye before its first frame, sharing ``clock``."""
        super().__init__()
        self._eye = eye
        self._clock = clock
        self._seen_sequence = 0
        # The MJPEG decoder is stateful, so each track owns one.
        self._decoder = av.CodecContext.create("mjpeg", "r")
        self._warned_decode = False

    async def recv(self) -> av.VideoFrame:
        """Return the eye's next camera frame, stamped from the shared clock."""
        while True:
            jpeg, self._seen_sequence = await video.wait_next(
                self._eye, self._seen_sequence
            )
            try:
                frames = self._decoder.decode(av.Packet(jpeg))
            except av.error.FFmpegError:
                if not self._warned_decode:
                    self._warned_decode = True
                    print(
                        f"WARNING: the {self._eye} camera input is not decodable JPEG",
                        flush=True,
                    )
                continue
            if frames:
                break
        frame = frames[-1]
        # Wall-clock pts instead of VideoStreamTrack's fixed 30 fps pacing:
        # frames should leave when the camera produces them, at its rate.
        frame.pts = self._clock.timestamp()
        frame.time_base = fractions.Fraction(1, _CLOCK_RATE)
        return frame


class WebRTCServer:
    """The robot half of the WebXR peer connection.

    Owns the peers, the data channels and the video tracks. It knows
    nothing about poses: frame payloads go straight to the ``on_frame``
    callback, which is where ``main`` turns them into dora outputs.
    """

    def __init__(
        self,
        on_frame,
        on_session_start,
        calibration_enabled: bool = False,
    ) -> None:
        """Prepare a server; no peer exists until an offer is answered."""
        self._on_frame = on_frame
        self._on_session_start = on_session_start
        self._calibration_enabled = calibration_enabled
        self._pcs: set = set()
        self._controls: set = set()
        self._running = True

    @property
    def running(self) -> bool:
        """Whether the node should keep serving.

        True until :meth:`stop`. In WebRTC-only mode it also turns False
        when the one peer goes away, since with no HTTP server no other
        browser can ever take its place.
        """
        return self._running

    def stop(self) -> None:
        """Stop serving and close every peer."""
        self._running = False

    async def close(self) -> None:
        """Close every peer connection."""
        pcs = list(self._pcs)
        self._pcs.clear()
        self._controls.clear()
        for pc in pcs:
            await pc.close()

    def send_control(self, payload: dict) -> None:
        """Send a message to every connected browser on ``control``.

        Used for what came of a calibration run: the operator cannot see
        the node's output while wearing a headset.
        """
        message = json.dumps(payload)
        for channel in list(self._controls):
            if channel.readyState == "open":
                channel.send(message)

    async def answer(self, offer_sdp: str) -> str:
        """Answer one offer and return the bare answer SDP."""
        pc = self._create_peer()
        return await self._answer(
            pc, RTCSessionDescription(sdp=offer_sdp, type="offer")
        )

    async def negotiate_oneshot(
        self,
        offer_sdp: str,
        answer_host: str,
        answer_port: int,
        connect_timeout: float = 60.0,
    ) -> None:
        """Answer a single offer handed in at startup, with no HTTP server.

        The offer arrives out of band (a command-line argument or an
        environment variable) as the bare SDP -- its type is always
        ``offer`` -- and the answer goes back as the bare answer SDP
        written to the TCP socket the caller is listening on. This is the
        WebRTC-only mode: another service hosts the page and brokers
        signaling, and this node just runs the peer for its lifetime.

        After sending the answer, waits up to ``connect_timeout`` seconds
        for the peer to connect. If it never does, the peer is closed and
        this raises :class:`RuntimeError`, so a stranded node exits
        instead of holding a dead connection forever.
        """
        pc = self._create_peer()
        established = asyncio.get_running_loop().create_future()

        @pc.on("connectionstatechange")
        def on_connectionstatechange() -> None:
            if pc.connectionState == "connected":
                if not established.done():
                    established.set_result(True)
            elif pc.connectionState in ("failed", "closed"):
                if not established.done():
                    established.set_result(False)
                # No HTTP server means no replacement browser, ever.
                self._running = False

        answer_sdp = await self._answer(
            pc, RTCSessionDescription(sdp=offer_sdp, type="offer")
        )

        _reader, writer = await asyncio.open_connection(answer_host, answer_port)
        writer.write(answer_sdp.encode("utf-8"))
        await writer.drain()
        writer.write_eof()
        writer.close()
        await writer.wait_closed()

        try:
            connected = await asyncio.wait_for(established, connect_timeout)
        except TimeoutError:
            connected = False
        if not connected:
            await pc.close()
            self._pcs.discard(pc)
            raise RuntimeError(
                f"no WebRTC connection within {connect_timeout:g}s of the answer"
            )

    async def _answer(self, pc: RTCPeerConnection, offer: RTCSessionDescription) -> str:
        """Consume an offer and return the answer SDP.

        The video tracks are attached between setting the remote and the
        local description, so they bind to the transceivers the browser
        offered rather than adding new ones the browser never asked for.

        aiortc's ``setLocalDescription`` waits for ICE gathering to
        finish, so the returned SDP already carries every candidate:
        signaling is a single exchange with no trickle.
        """
        await pc.setRemoteDescription(offer)
        clock = _SharedClock()
        for eye in video.eyes():
            pc.addTrack(_EyeVideoTrack(eye, clock))
        await pc.setLocalDescription(await pc.createAnswer())
        return pc.localDescription.sdp

    def _create_peer(self) -> RTCPeerConnection:
        """Build a peer with its data channels wired up."""
        configuration = RTCConfiguration(iceServers=list(ICE_SERVERS))
        pc = RTCPeerConnection(configuration)
        self._pcs.add(pc)

        # Opened by this node, like the page's own configuration: a page
        # this node never served has no other way to learn it.
        control = pc.createDataChannel("control")

        @control.on("open")
        def on_control_open() -> None:
            self._controls.add(control)
            control.send(
                json.dumps(
                    {
                        "type": "configuration",
                        "view_configuration": video.view_configuration(),
                        "calibration": self._calibration_enabled,
                        # Which eye each video track carries, in track
                        # order. The mono view sends one track, so the
                        # browser cannot infer the eye from the count.
                        "eyes": video.eyes(),
                    }
                )
            )

        @control.on("close")
        def on_control_close() -> None:
            self._controls.discard(control)

        @control.on("message")
        def on_control_message(message: object) -> None:
            self._handle_control_message(message)

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            if channel.label != "xr":
                return

            @channel.on("message")
            def on_message(message: object) -> None:
                self._handle_frame_message(message)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            # A browser that leaves without closing -- a reload, a
            # sleeping headset -- is only ever noticed here. Left alone,
            # the peer's video encoders would keep running for nobody
            # until the node exits, and every reload would stack another
            # one on. Only the terminal states count: "disconnected" is
            # also what a brief network blip looks like, and that can
            # still recover.
            if pc.connectionState in ("failed", "closed"):
                self._pcs.discard(pc)
                self._controls.discard(control)
                await pc.close()

        return pc

    def _handle_control_message(self, message: object) -> None:
        payload = _decode(message)
        if payload is None:
            return
        if payload.get("type") == "session-start":
            self._on_session_start()

    def _handle_frame_message(self, message: object) -> None:
        payload = _decode(message)
        if payload is None:
            return
        if payload.get("type") == "frame":
            self._on_frame(payload)


def _decode(message: object) -> dict | None:
    """Parse one channel message, or None if it is not a JSON object.

    A malformed payload is ignored rather than fatal, so a single bad
    message cannot take the transport down mid-session.
    """
    if not isinstance(message, str):
        return None
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
