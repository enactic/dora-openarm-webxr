# dora-openarm-webxr

A [dora-rs](https://dora-rs.ai) node that reads the pose and
controller state of a VR device such as Meta Quest 3 or PICO 4 through
[WebXR](https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API)
and publishes them to a dora-rs dataflow. You can use it for OpenArm
teleoperation with a VR device.

## Install

```bash
pip install dora-openarm-webxr
```

## Setup

This dora-rs node starts a Web server because WebXR runs as JavaScript
in the Web browser on a VR device. The VR device connects to this
server to stream its pose and controller state.

WebXR requires HTTPS, so this dora-rs node needs a certificate. A
self-signed certificate is enough because the dora-rs node and the VR
device communicate only within your local network. You can generate
one with
[`example/prepare_tls.sh`](example/prepare_tls.sh):

```bash
git clone https://github.com/enactic/dora-openarm-webxr.git
cd dora-openarm-webxr
example/prepare_tls.sh ${YOUR_HOST_NAME}
```

Replace `${YOUR_HOST_NAME}` with a host name that your VR device can
resolve. A `.local` host name configured automatically by Avahi is a
convenient choice. You can check whether your `.local` host name is
available with the following command:

```bash
avahi-resolve --name $(hostname).local
```

If it resolves to your host's IP address, you can generate the
self-signed certificate with the following command line:

```bash
example/prepare_tls.sh $(hostname).local
```

This writes `server.crt` and `server.key` (and the `root.*` files used
to sign them) into the `example/` directory.

You can run
[`example/dataflow_mujoco.yaml`](example/dataflow_mujoco.yaml) with the
generated self-signed certificate by the following command lines:

```bash
pip install dora-rs-cli
dora build example/dataflow_mujoco.yaml
TLS_CERTIFICATE_FILE=server.crt TLS_KEY_FILE=server.key dora run example/dataflow_mujoco.yaml
```

Open http://localhost:8000/ on the local machine for
[dora-openarm-data-collection-ui](https://github.com/enactic/dora-openarm-data-collection-ui).

Open `https://$(hostname).local:8443/` in the Web browser on your VR
device, not in the browser on your local machine. Because the
certificate is self-signed, the Web browser shows a security
warning. You can continue to the page from its "Advanced" options.

Press the "Start" button on the page to start teleoperation with your
VR device.

## Head camera

This dora-rs node can also show the robot's head camera in the VR
device, so that the operator sees the robot's workspace while
teleoperating.

The node accepts JPEG images on the `camera_head_right` input and
forwards them to the VR device, where they are drawn on a panel locked
to the room: straight ahead of where the headset was when the session
started, at eye height. Both eyes see the same image, and the panel
stays put when the operator moves their head. See
[`example/dataflow_mujoco_camera.yaml`](example/dataflow_mujoco_camera.yaml)
and
[`example/dataflow_mujoco_camera_stereo.yaml`](example/dataflow_mujoco_camera_stereo.yaml).

How the panel is drawn is described by a view configuration file,
passed with `--view-configuration-file`. The node reads it once when
it starts, so restart the dataflow to apply a change. Two keys describe
the view, and they are independent: `view` says how many images are
drawn, `panel.lock` says what they hang off.

`view` is `mono` (one image, from `camera_head_right`), `stereo` (one
per eye, which also needs `camera_head_left`) or `none` (no camera at
all: the operator sees the passthrough and only the controller poses are
used).

`panel.lock` is `room` (the panel stays where the headset was looking
when the session started, so a head turn looks away from it) or `head`
(the panel travels with the operator's gaze).

The example files pair them the two useful ways:

- [`example/view_camera.yaml`](example/view_camera.yaml) — the default
  `mono` view above, locked to the `room`. Parameters: the session mode,
  the panel distance and the panel width (the height follows the image
  aspect ratio).
- [`example/view_camera_stereo.yaml`](example/view_camera_stereo.yaml)
  — the `stereo` view, locked to the `head`, for a side-by-side stereo
  camera such as a ZED Mini.
  [`example/zed_view_parameters.py`](example/zed_view_parameters.py)
  works its camera and alignment parameters out from a ZED camera's
  factory calibration.

Both files also take `pose.frame_offset`: the neutral hand position
relative to the `arm_origin` site in meters, overriding the built-in
default of `[-0.085, 0, -0.14]`.

They also take `pose.neck_pivot_offset`: the operator's eyes to the
neck's rotation axis, in the headset's own frame, overriding the
built-in default of `[0.0, -0.075, 0.080]`. Hand positions are made
relative to that pivot rather than to the headset itself, so turning the
head does not swing the target along the arc the headset travels.
Anatomy varies, so tune it per operator, or measure it as described in
[Calibrating the neck pivot](#calibrating-the-neck-pivot) below;
`[0, 0, 0]` goes back to subtracting the headset position. A measured
offset in `--neck-pivot-file` wins over this one.

## Calibrating the neck pivot

Rather than guessing the offset, measure it: the operator turns their
head, and the node fits the one point that stayed put through the turn.

Calibration is off unless you ask for it. Start the dataflow with
`--calibration`, or `CALIBRATION` in its YAML, for a session that
measures. Without it the Y button is an ordinary button: nothing is
collected and the hands never stop.

```bash
TLS_CERTIFICATE_FILE=server.crt TLS_KEY_FILE=server.key \
  CALIBRATION=true dora run example/dataflow_mujoco.yaml
```

1. Start a session as above. The headset shows what to do, so the
   operator needs nothing else in front of them.
2. Have the operator stand facing the workspace with their feet
   planted.
3. Press and hold the **Y button** on the left controller. The run
   starts at once: the panel turns green and the hands stop following,
   though the gripper still follows the trigger, so leave the triggers
   alone until the run is over.
4. Keeping the body still and the button down, turn the head slowly
   some 40 degrees **side to side**, twice over, and then some 40
   degrees **up and down**, twice over. Four to six seconds in all.
5. Release Y. The fit runs at that moment and the result appears on the
   panel.

Both turn directions are needed: a rotation says nothing about the
offset along the axis it turns about, so shaking only sideways leaves
the vertical offset unmeasured. The lower bounds are a hundred poses
and some 20 degrees about every axis, so the run above has a wide
margin over both.

An accepted run is applied immediately, so the operator can turn their
head and see for themselves whether the target now stays put. It is
also written to `--neck-pivot-file`, which defaults to `neck_pivot.yaml`
in the directory the node runs in, and read back from there at startup:

```yaml
# Measured by dora-openarm-webxr --calibration.
pose:
  neck_pivot_offset: [0.004, -0.081, 0.076]
```

So calibrate once with `--calibration`, then run without it and the
measurement is still in use. The node says so when it reads the file,
and the same document can be pasted into the view configuration file
instead. The node's output carries the run either way:

```
neck pivot calibration applied from 412 poses: the pivot held to 6.1 mm while the headset moved 47.2 mm.
  neck_pivot_offset: [0.004, -0.081, 0.076]
  Written to neck_pivot.yaml, which this node reads at startup.
```

The smaller the distance the pivot held to, and the larger the one the
headset moved, the better the run.

A refused run changes nothing and says on the panel, and on the node's
output, what to do differently: hold Y down longer, add the turn that is
missing, or keep the body still and turn only the head. Holding Y again
starts a fresh run, so it can be repeated until it takes.

## Debug

You can use [Immersive Web
Emulator](https://chromewebstore.google.com/detail/immersive-web-emulator/cgffilbpcibhmcfbgggfhfolhkfbhmik)
and Chrome to debug this node without a VR device.

## Inputs

This dora-rs node accepts the following data. Both are optional and
only needed to show a head camera in the VR device.

| Input                | Type      | Description                                                              |
|----------------------|-----------|--------------------------------------------------------------------------|
| `camera_head_right`  | `uint8[]` | A JPEG image of the robot's head camera.                                 |
| `camera_head_left`   | `uint8[]` | A JPEG image for the left eye. Only used by the stereo view.             |

## Outputs

This dora-rs node outputs the following data. Pose, trigger, grip and
joystick outputs are sent on each `frame` message received from the VR
device. Button outputs are sent only when the corresponding button is
included in a `frame` message. `pose_reference` is sent whenever the
headset is tracked, even while the controllers are off, and the
controller poses are sent only when it is.

| Output             | Type              | Description                                                                                                                                    |
|--------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `status`           | `string`          | `"ready"` when a WebXR session is started.                                                                                                      |
| `vr_receive_times` | `int64`           | The timestamp in nanoseconds when a frame is received from the VR device.                                                                       |
| `pose_right`       | `float32[7]`      | The pose of the right controller as `[x, y, z, qw, qx, qy, qz]`, expressed in the scene's `arm_origin` site frame. Position is in meters and orientation is a quaternion. |
| `pose_left`        | `float32[7]`      | The pose of the left controller. The format is the same as `pose_right`.                                                                        |
| `pose_reference`   | `float32[7]`      | The pose of the headset, in the WebXR reference space (x right, y up, -z forward). The hand poses are made relative to this pose, so it is unrotated and unsmoothed and left in the WebXR frame for consumers that drive something from head motion such as a neck. |
| `trigger_right`    | `float32`         | The value of the right trigger from `0.0` (released) to `1.0` (fully pressed).                                                                  |
| `trigger_left`     | `float32`         | The value of the left trigger from `0.0` (released) to `1.0` (fully pressed).                                                                   |
| `grip_right`       | `float32`         | The value of the right grip (squeeze) button from `0.0` (released) to `1.0` (fully pressed).                                                    |
| `grip_left`        | `float32`         | The value of the left grip. The format is the same as `grip_right`.                                                                             |
| `joystick_x_right` | `float32`         | The X axis value of the right joystick.                                                                                                         |
| `joystick_y_right` | `float32`         | The Y axis value of the right joystick.                                                                                                         |
| `joystick_x_left`  | `float32`         | The X axis value of the left joystick.                                                                                                          |
| `joystick_y_left`  | `float32`         | The Y axis value of the left joystick.                                                                                                          |
| `button_a`         | `bool`            | Whether the A button is pressed or not.                                                                                                         |
| `button_b`         | `bool`            | Whether the B button is pressed or not.                                                                                                         |
| `button_x`         | `bool`            | Whether the X button is pressed or not.                                                                                                         |
| `button_y`         | `bool`            | Whether the Y button is pressed or not.                                                                                                         |

`button_y` is published whatever the press does here. In a session
started with `--calibration` it also runs a [neck pivot
calibration](#calibrating-the-neck-pivot) for as long as it is held,
and the hand poses stop until it is released. Without that option the
press does nothing but reach the output, so a dataflow that wires the
button to something the operator holds down is unaffected.

## Command line options

You can configure this dora-rs node by the following command line
options. Each option also has a corresponding environment variable
that is used as the default value. Setting the environment variable is
useful in a dora-rs dataflow YAML.

| Option                   | Environment variable   | Default     | Description                                                                       |
|--------------------------|------------------------|-------------|-----------------------------------------------------------------------------------|
| `--host`                 | `HOST`                 | `0.0.0.0`   | The host that the Web server listens on.                                          |
| `--port`                 | `PORT`                 | `8443`      | The port that the Web server listens on.                                          |
| `--tls-certificate-file` | `TLS_CERTIFICATE_FILE` | (required)  | The TLS certificate file for HTTPS. Required because WebXR requires HTTPS.        |
| `--tls-key-file`         | `TLS_KEY_FILE`         | (required)  | The TLS key file for the certificate file. Required because WebXR requires HTTPS. |
| `--view-configuration-file` | `VIEW_CONFIGURATION_FILE` | (none)  | The YAML file that describes how the head camera is drawn in the VR device. Read once when the node starts. |
| `--calibration`          | `CALIBRATION`          | off         | Measure the neck pivot with the Y button, and show the instructions for it in the headset. Off unless asked for. |
| `--neck-pivot-file`      | `NECK_PIVOT_FILE`      | `neck_pivot.yaml` | The YAML file a measured neck pivot offset is written to, and read back from at startup. |

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
