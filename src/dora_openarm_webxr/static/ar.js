// Copyright 2026 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

if (navigator.xr) {
  let websocket = new WebSocket("wss://" + location.host + "/websocket");
  let runningSession = null;

  websocket.addEventListener("close", (event) => {
    websocket = null;
    if (runningSession) {
      runningSession.end();
      runningSession = null;
    }
  });
  websocket.addEventListener("error", (event) => {
    websocket = null;
    if (runningSession) {
      runningSession.end();
      runningSession = null;
    }
  });

  function log(message) {
    // document.getElementById("log").innerText += `${message}\n`;
    // websocket.send(JSON.stringify({type: "log", message: `${message}`}));
  }
  function onSessionEnd(event) {
    log("ended");
    runningSession = null;
    if (websocket) {
      websocket.close();
      websocket = null;
    }
  }
  function onSelectStart(event) {
    const response = {
      type: "select-start",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    websocket.send(JSON.stringify(response));
  }
  function onSelect(event) {
    const response = {
      type: "select",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    websocket.send(JSON.stringify(response));
  }
  function onSelectEnd(event) {
    const response = {
      type: "select-end",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    websocket.send(JSON.stringify(response));
  }
  function onSqueezeStart(event) {
    const response = {
      type: "squeeze-start",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    websocket.send(JSON.stringify(response));
  }
  function onSqueeze(event) {
    const response = {
      type: "squeeze",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    websocket.send(JSON.stringify(response));
  }
  function onSqueezeEnd(event) {
    const response = {
      type: "squeeze-end",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    websocket.send(JSON.stringify(response));
  }
  function sendFrame(session, space, time, frame) {
    if (session.inputSources.length < 2) {
      return;
    }
    const response = {
      type: "frame",
      time: time,
    };
    for (const source of session.inputSources) {
      if (source.handedness === "none") {
        continue;
      }
      const suffix = `_${source.handedness}`;
      const pose = frame.getPose(source.gripSpace, space);
      response[`pose${suffix}`] = {
        x: pose.transform.position.x,
        y: pose.transform.position.y,
        z: pose.transform.position.z,
        qx: pose.transform.orientation.x,
        qy: pose.transform.orientation.y,
        qz: pose.transform.orientation.z,
        qw: pose.transform.orientation.w,
      };
      const gamepad = source.gamepad;
      if (gamepad) {
        if (
          source.profiles.includes("pico-4u") ||
          source.profiles.includes("meta-quest-touch-plus")
        ) {
          const trigger = gamepad.buttons[0];
          response[`trigger${suffix}`] = trigger.value;
          if (source.handedness === "right") {
            const a = gamepad.buttons[4];
            const b = gamepad.buttons[5];
            response.button_a = a.pressed;
            response.button_b = b.pressed;
          } else {
            const x = gamepad.buttons[4];
            const y = gamepad.buttons[5];
            response.button_x = x.pressed;
            response.button_y = y.pressed;
          }
        }
        if (
          gamepad.axes[0] &&
          gamepad.axes[1] &&
          gamepad.axes[2] &&
          gamepad.axes[3]
        ) {
          response[`joystick${suffix}`] = gamepad.axes;
        }
      }
    }
    websocket.send(JSON.stringify(response));
  }
  function onSessionStart(session) {
    runningSession = session;
    // TODO: Add device information
    const response = {
      type: "session-start",
    };
    websocket.send(JSON.stringify(response));

    session.addEventListener("end", onSessionEnd);

    session.addEventListener("selectstart", onSelectStart);
    session.addEventListener("select", onSelect);
    session.addEventListener("selectend", onSelectEnd);

    session.addEventListener("squeezestart", onSqueezeStart);
    session.addEventListener("squeeze", onSqueeze);
    session.addEventListener("squeezeend", onSqueezeEnd);

    // We don't render anything but we need to setup render state to use
    // immersive AR.
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl", { xrCompatible: true });
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });

    session
      // We send relative position from viewer to the dora-rs node.
      .requestReferenceSpace("viewer")
      .then((space) => {
        function onFrame(time, frame) {
          log("sources: " + session.inputSources.length);
          sendFrame(session, space, time, frame);
          session.requestAnimationFrame(onFrame);
        }
        session.requestAnimationFrame(onFrame);
      })
      .catch((error) => {
        alert(error);
      });
  }
  function onStart() {
    navigator.xr.requestSession("immersive-ar").then(onSessionStart);
  }

  websocket.addEventListener("open", () => {
    navigator.xr.isSessionSupported("immersive-ar").then((isSupported) => {
      if (isSupported) {
        // WebXR requires explicit user interaction on start. We use
        // button click here.
        document.getElementById("start").addEventListener("click", onStart);
      }
    });
  });
}
