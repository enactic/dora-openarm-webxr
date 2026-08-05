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

This dora-rs node can also show a stereo head camera in the VR device,
so that the operator sees through the robot's head while teleoperating.

The camera is a side-by-side stereo camera such as a ZED Mini. One node
captures it and a splitter cuts each frame into one JPEG image per eye,
which this node accepts as the `camera_head_left` and
`camera_head_right` inputs and forwards to the VR device. A dataset
recorder can read the same two outputs, so what the operator sees and
what is recorded are the same images. See
[`example/head_cam_webxr.yaml`](example/head_cam_webxr.yaml):

```bash
dora build example/head_cam_webxr.yaml
dora run example/head_cam_webxr.yaml
```

How the images are drawn is described by a view configuration file,
passed with `--view-configuration-file`. The node reads it on every
request, so the view can be tuned by editing the file and reloading the
page in the VR device. See
[`example/head_cam_view.yaml`](example/head_cam_view.yaml) for the
parameters.

The images are captured over plain video capture rather than through
the camera's own SDK, so they are not rectified and the two lenses'
misalignment has to be corrected when they are drawn.
[`example/zed_view_parameters.py`](example/zed_view_parameters.py)
works the parameters out from a ZED camera's factory calibration:

```bash
example/zed_view_parameters.py
```

Pass `--measure`, with the camera not in use, to measure the alignment
from the camera itself instead of trusting the calibration.

## Debug

You can use [Immersive Web
Emulator](https://chromewebstore.google.com/detail/immersive-web-emulator/cgffilbpcibhmcfbgggfhfolhkfbhmik)
and Chrome to debug this node without a VR device.

## Inputs

This dora-rs node accepts the following data. Both are optional and are
only needed to show a head camera in the VR device.

| Input                | Type      | Description                                                                                       |
|----------------------|-----------|---------------------------------------------------------------------------------------------------|
| `camera_head_left`   | `uint8[]` | A JPEG image for the left eye, as produced by an image splitter from a side-by-side stereo camera. |
| `camera_head_right`  | `uint8[]` | A JPEG image for the right eye. The format is the same as `camera_head_left`.                      |

## Outputs

This dora-rs node outputs the following data. Pose, trigger and
joystick outputs are sent on each `frame` message received from the VR
device. Button outputs are sent only when the corresponding button is
included in a `frame` message.

Controller poses are compensated for the headset tilt: the head pitch
and roll are cancelled every frame so the mapping stays gravity-aligned,
while the headset yaw and position are still followed. Frames that do
not include a headset pose are published uncompensated.

| Output             | Type              | Description                                                                                                                                    |
|--------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `status`           | `string`          | `"ready"` when a WebXR session is started.                                                                                                      |
| `vr_receive_times` | `int64`           | The timestamp in nanoseconds when a frame is received from the VR device.                                                                       |
| `pose_right`       | `float32[7]`      | The pose of the right controller as `[x, y, z, qw, qx, qy, qz]`, expressed in the scene's `arm_origin` site frame. Position is in meters and orientation is a quaternion. |
| `pose_left`        | `float32[7]`      | The pose of the left controller. The format is the same as `pose_right`.                                                                        |
| `trigger_right`    | `float32`         | The value of the right trigger from `0.0` (released) to `1.0` (fully pressed).                                                                  |
| `trigger_left`     | `float32`         | The value of the left trigger from `0.0` (released) to `1.0` (fully pressed).                                                                   |
| `joystick_x_right` | `float32`         | The X axis value of the right joystick.                                                                                                         |
| `joystick_y_right` | `float32`         | The Y axis value of the right joystick.                                                                                                         |
| `joystick_x_left`  | `float32`         | The X axis value of the left joystick.                                                                                                          |
| `joystick_y_left`  | `float32`         | The Y axis value of the left joystick.                                                                                                          |
| `button_a`         | `bool`            | Whether the A button is pressed or not.                                                                                                         |
| `button_b`         | `bool`            | Whether the B button is pressed or not.                                                                                                         |
| `button_x`         | `bool`            | Whether the X button is pressed or not.                                                                                                         |
| `button_y`         | `bool`            | Whether the Y button is pressed or not.                                                                                                         |

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
| `--view-configuration-file` | `VIEW_CONFIGURATION_FILE` | (none)  | The YAML file that describes how the head camera is drawn in the VR device. Read on every request, so it can be edited while the node is running. |

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
