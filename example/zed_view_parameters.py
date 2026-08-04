#!/usr/bin/env python3
#
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

"""Suggest head camera view parameters for a ZED camera.

Prints a view configuration such as ``head_cam_view.yaml``, worked out
from the camera's calibration instead of guessed at.

The node reads the camera over plain video capture, not the ZED SDK, so
the images are unrectified and the lenses' misalignment has to be
corrected when they are drawn.

Pass --measure to check the eye alignment against the camera itself.
That needs the camera free, so stop any dataflow using it first.
"""

import argparse
import configparser
import glob
import math
import sys

# Per-eye image size of each ZED capture mode.
RESOLUTIONS = {
    "2K": (2208, 1242),
    "FHD": (1920, 1080),
    "HD": (1280, 720),
    "VGA": (672, 376),
}

DEFAULT_CALIBRATION_GLOB = "/usr/local/zed/settings/SN*.conf"

# Images further apart than this are uncomfortable to fuse.
COMFORTABLE_DISPARITY_DEGREES = 1.0

# Roughly the gap between an adult's eyes.
HUMAN_INTERPUPILLARY_DISTANCE = 0.063

# Where a headset's lenses focus, whatever is drawn. Putting the panel
# here lets the eyes converge and focus at the same depth. A Quest 3.
HEADSET_FOCAL_DISTANCE = 1.3


def find_calibration_file() -> str:
    """Find the ZED SDK's calibration file for the attached camera."""
    matches = sorted(glob.glob(DEFAULT_CALIBRATION_GLOB))
    if not matches:
        raise SystemExit(
            f"no calibration file matching {DEFAULT_CALIBRATION_GLOB}.\n"
            "Install the ZED SDK, or pass --calibration-file."
        )
    if len(matches) > 1:
        print(f"note: several calibration files, using {matches[0]}", file=sys.stderr)
    return matches[0]


def read_calibration(path: str, resolution: str) -> dict:
    """Read the intrinsics and stereo alignment for one capture mode."""
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise SystemExit(f"cannot read {path}")
    try:
        left = parser[f"LEFT_CAM_{resolution}"]
        right = parser[f"RIGHT_CAM_{resolution}"]
        stereo = parser["STEREO"]
    except KeyError as error:
        raise SystemExit(f"{path} has no section {error}") from error
    return {
        "left": {key: float(left[key]) for key in ["fx", "fy", "cx", "cy"]},
        "right": {key: float(right[key]) for key in ["fx", "fy", "cx", "cy"]},
        "baseline": float(stereo["Baseline"]) / 1000.0,
        "rx": float(stereo[f"RX_{resolution}"]),
        "rz": float(stereo[f"RZ_{resolution}"]),
        "cv": float(stereo[f"CV_{resolution}"]),
    }


def compute(
    calibration: dict,
    resolution: str,
    headset_fov: tuple,
    magnification: float,
) -> dict:
    """Work out the view parameters the calibration determines."""
    width, height = RESOLUTIONS[resolution]
    left, right = calibration["left"], calibration["right"]

    # Drawn across the angle it was taken over, the image is life sized.
    reach = (width / (2.0 * left["fx"]), height / (2.0 * left["fy"]))
    degrees_per_pixel = 2.0 * math.degrees(math.atan(reach[1])) / height

    # How far the right image sits from the left: the lens centres
    # differ, and one lens is pitched and turned against the other.
    # Correcting only that leaves the stereo as the camera saw it, so
    # infinity fuses and nearer things converge as they would in person.
    vertical_centres = right["cy"] - left["cy"]
    vertical_pitch = left["fy"] * math.tan(calibration["rx"])
    horizontal_centres = right["cx"] - left["cx"]
    horizontal_yaw = left["fx"] * math.tan(calibration["cv"])

    # Rolled slightly too. Nothing here corrects that; just report.
    roll = abs(math.tan(calibration["rz"]) * width / 2.0)

    head_reach = (
        math.tan(math.radians(headset_fov[0]) / 2.0),
        math.tan(math.radians(headset_fov[1]) / 2.0),
    )
    return {
        "horizontal_fov": 2.0 * math.degrees(math.atan(magnification * reach[0])),
        "vertical_fov": 2.0 * math.degrees(math.atan(magnification * reach[1])),
        "degrees_per_pixel": degrees_per_pixel,
        "vertical_align": vertical_centres + vertical_pitch,
        "vertical_centres": vertical_centres,
        "vertical_pitch": vertical_pitch,
        "convergence": horizontal_centres + horizontal_yaw,
        "horizontal_centres": horizontal_centres,
        "horizontal_yaw": horizontal_yaw,
        "roll": roll,
        "baseline": calibration["baseline"],
        "magnification": magnification,
        "fill": max(head_reach[0] / reach[0], head_reach[1] / reach[1]),
        "filled": min(1.0, magnification * reach[1] / head_reach[1]),
    }


def measure_vertical_align(device: str, resolution: str) -> float:
    """Measure the vertical offset between the two images.

    Matches features between the two halves of a frame and takes the
    median vertical gap. Settles the sign, easy to get wrong on paper.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise SystemExit(f"--measure needs OpenCV and NumPy: {error}") from error

    width, height = RESOLUTIONS[resolution]
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise SystemExit(
            f"cannot open {device}. It can only be opened once, so stop any "
            "dataflow that is using it first."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # A driver can quietly give a different mode, which would put the
    # measurement on a different scale to the calibration.
    actual = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if actual != (width * 2, height):
        print(
            f"warning: asked for {width * 2}x{height} but got"
            f" {actual[0]}x{actual[1]}. Pass --resolution to match.",
            file=sys.stderr,
        )
    # The first frames are often still being exposed.
    for _ in range(10):
        captured, frame = capture.read()
    capture.release()
    if not captured:
        raise SystemExit(f"no frame from {device}")

    half = frame.shape[1] // 2
    left = cv2.cvtColor(frame[:, :half], cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(frame[:, half:], cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(4000)
    left_points, left_description = orb.detectAndCompute(left, None)
    right_points, right_description = orb.detectAndCompute(right, None)
    if left_description is None or right_description is None:
        raise SystemExit("no features found. Point the camera at some detail.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(left_description, right_description)
    if len(matches) < 20:
        raise SystemExit(
            f"only {len(matches)} features matched. Point the camera at "
            "something with more detail."
        )

    offsets = np.array(
        [
            right_points[match.trainIdx].pt[1] - left_points[match.queryIdx].pt[1]
            for match in matches
        ]
    )
    # Mismatches scatter, real ones agree: keep the middle.
    middle = np.median(offsets)
    spread = np.median(np.abs(offsets - middle)) or 1.0
    kept = offsets[np.abs(offsets - middle) < 3.0 * spread]
    print(f"measured from {len(kept)} of {len(matches)} features", file=sys.stderr)
    return float(np.median(kept))


def report(values: dict, resolution: str, path: str) -> None:
    """Explain where the numbers came from."""
    width, height = RESOLUTIONS[resolution]
    degrees = values["degrees_per_pixel"]
    print(f"# From {path}, {resolution} ({width}x{height} per eye).")
    print("#")
    print("# Lens alignment, which the raw images still carry:")
    print(
        f"#   vertically {values['vertical_align']:+7.1f} px"
        f" ({abs(values['vertical_align']) * degrees:.2f} deg)"
        f" = {values['vertical_centres']:+.1f} centres"
        f" {values['vertical_pitch']:+.1f} pitch"
    )
    print(
        f"#   sideways   {values['convergence']:+7.1f} px"
        f" = {values['horizontal_centres']:+.1f} centres"
        f" {values['horizontal_yaw']:+.1f} yaw"
    )
    print(f"#   roll       {values['roll']:7.1f} px at the corner")
    for name, pixels in [
        ("vertical", abs(values["vertical_align"])),
        ("roll", values["roll"]),
    ]:
        if pixels * degrees > COMFORTABLE_DISPARITY_DEGREES:
            print(
                f"#   -> the {name} offset is past what fuses comfortably"
                + (
                    ", and nothing here corrects it."
                    if name == "roll"
                    else ", so worth correcting."
                )
            )
    print("#")
    print(
        f"# Baseline {values['baseline'] * 1000:.1f} mm against about"
        f" {HUMAN_INTERPUPILLARY_DISTANCE * 1000:.0f} mm between human eyes,"
    )
    if (
        abs(values["baseline"] - HUMAN_INTERPUPILLARY_DISTANCE)
        < 0.1 * HUMAN_INTERPUPILLARY_DISTANCE
    ):
        print("# so the lenses sit about where eyes would. The view is")
        print("# already life sized and nothing needs reprojecting.")
    else:
        scale = values["baseline"] / HUMAN_INTERPUPILLARY_DISTANCE
        print(f"# so depth looks exaggerated by about {scale:.1f} times.")


def main() -> None:
    """Print suggested view parameters."""
    parser = argparse.ArgumentParser(
        description="Suggest head camera view parameters for a ZED camera"
    )
    parser.add_argument(
        "--calibration-file",
        help="ZED calibration file (default: found under /usr/local/zed)",
    )
    parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTIONS),
        default="HD",
        help="Capture mode, per eye (default: HD, 1280x720)",
    )
    parser.add_argument(
        "--magnification",
        type=float,
        default=1.0,
        help="Enlarge the image to fill more of the display (default: 1.0)",
    )
    parser.add_argument(
        "--headset-fov",
        default="110x96",
        help="Field of view of one eye of the headset (default: 110x96)",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Measure the eye alignment from the camera itself",
    )
    parser.add_argument(
        "--device",
        default="/dev/camera_head_stereo",
        help="Camera to measure (default: /dev/camera_head_stereo)",
    )
    args = parser.parse_args()

    try:
        headset_fov = tuple(float(part) for part in args.headset_fov.split("x"))
        if len(headset_fov) != 2:
            raise ValueError
    except ValueError:
        raise SystemExit("--headset-fov must look like 110x96") from None

    path = args.calibration_file or find_calibration_file()
    calibration = read_calibration(path, args.resolution)
    values = compute(calibration, args.resolution, headset_fov, args.magnification)

    report(values, args.resolution, path)

    vertical_align = values["vertical_align"]
    if args.measure:
        measured = measure_vertical_align(args.device, args.resolution)
        print("#")
        print(f"# Measured from the camera: {measured:+.1f} px.")
        if abs(measured - vertical_align) > 3.0:
            print("# Disagrees with the calibration; trusting the measurement.")
        vertical_align = measured

    print()
    print("session:")
    print("  # A preference, not from the calibration.")
    print("  mode: immersive-vr")
    print()
    print("camera:")
    print(
        f"  # {values['magnification']:.2f}x, covering"
        f" {values['filled'] * 100:.0f}% of the display."
        f" {values['fill']:.2f}x would fill it."
    )
    print(f"  horizontal_fov: {values['horizontal_fov']:.1f}")
    print(f"  vertical_fov: {values['vertical_fov']:.1f}")
    print()
    print("panel:")
    print("  # Where the headset's lenses focus, so the eyes converge and")
    print("  # focus at the same depth.")
    print(f"  distance: {HEADSET_FOCAL_DISTANCE:.1f}")
    print()
    print("stereo:")
    print(f"  vertical_align: {vertical_align:.1f}")
    print(f"  convergence: {values['convergence']:.1f}")


if __name__ == "__main__":
    main()
