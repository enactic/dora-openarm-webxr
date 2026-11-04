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

"""Per-hand teleoperation clutch.

The operator squeezes the grip to engage a hand and lets go to release it.
While engaged, the target moves by exactly as much as the controller does.
While released, it does not move at all, and the controller can go anywhere:
engaging again continues from the held target instead of jumping to the hand.

This module owns the policy and nothing else: it knows no transport, clock or
filter, so that the same inputs always give the same targets.

Frames: X is the XR reference space, C the controller, F the operator frame,
which is X with its axes frozen at engagement, A the robot's arm_origin and E
the end effector target. With M = A_R_F * X_R_F^-1 taking directions from X to
A, and the subscript 0 marking what was captured at engagement::

    p_E = p_E0 + M (p_C - p_C0)
    R_E = (M R_C R_C0^-1 M^-1) R_E0

Both increments are the identity at engagement, so the target starts exactly at
the reference, and M is a rotation, so the target never moves further than the
controller did. Freezing F is what keeps a head movement out of the target.

This keeps a target continuous. It does not make a robot safe: nothing here
knows where the arm actually is.
"""

import dataclasses
import enum

import numpy as np
from scipy.spatial.transform import Rotation

# A hysteresis band, so that a grip held near one value cannot chatter.
GRIP_ENGAGE = 0.6
GRIP_RELEASE = 0.4

# How far a quaternion may be from unit length before it is not a rotation.
_UNIT_TOLERANCE = 0.01


class ClutchState(enum.Enum):
    """What a hand is doing."""

    # Waiting to see the grip open. Nothing can engage from here, which is what
    # makes a squeeze held through a fault or a new session not count.
    AWAIT_RELEASE = "await_release"
    RELEASED = "released"
    ENGAGED = "engaged"


@dataclasses.dataclass(frozen=True)
class Pose:
    """A position in meters and a rotation."""

    position: np.ndarray
    rotation: Rotation


@dataclasses.dataclass(frozen=True)
class Sample:
    """One validated observation of a hand."""

    # Squeeze, 0..1.
    grip: float
    # The controller in X.
    controller: Pose
    # X_R_F: what the axes of F would be if this sample were to engage.
    axes: Rotation
    # The gripper angle the trigger asks for, in radians.
    gripper: float


def parse_pose(fields) -> Pose | None:
    """Validate a {x, y, z, qx, qy, qz, qw} mapping, or None if it is not a pose."""
    try:
        values = np.array(
            [float(fields[key]) for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(values)):
        return None
    if abs(np.linalg.norm(values[3:]) - 1.0) > _UNIT_TOLERANCE:
        return None
    return Pose(values[:3], Rotation.from_quat(values[3:]))


def parse_grip(value) -> float | None:
    """Validate a squeeze value, or None if it is not one."""
    try:
        grip = float(value)
    except (TypeError, ValueError):
        return None
    return grip if 0.0 <= grip <= 1.0 else None


class HandClutch:
    """The clutch of one hand."""

    def __init__(
        self, initial_target: Pose, initial_gripper: float, robot_axes: Rotation
    ):
        """Hold the initial target until a first engagement moves it.

        Args:
          initial_target: where the target starts, in A. It has to be where IK
            starts for a first engagement not to jump.
          initial_gripper: the gripper angle to start with, in radians.
          robot_axes: A_R_F.

        """
        self._robot_axes = robot_axes
        self.state = ClutchState.AWAIT_RELEASE
        self.target = initial_target
        self.gripper = initial_gripper
        self._gripper_side = 0.0
        self._controller0 = None
        self._target0 = None
        self._to_robot = None

    def revoke(self) -> None:
        """Withdraw authorization: the grip must be seen open before it counts."""
        self.state = ClutchState.AWAIT_RELEASE

    def update(self, sample: Sample | None) -> ClutchState:
        """Take one observation, or None if the hand cannot be observed."""
        if sample is None:
            self.revoke()
        elif self.state is ClutchState.AWAIT_RELEASE:
            if sample.grip <= GRIP_RELEASE:
                self.state = ClutchState.RELEASED
        elif self.state is ClutchState.RELEASED:
            if sample.grip >= GRIP_ENGAGE:
                self._engage(sample)
        elif sample.grip <= GRIP_RELEASE:
            self.state = ClutchState.RELEASED
        else:
            self._follow(sample)
        return self.state

    def _engage(self, sample: Sample) -> None:
        self.state = ClutchState.ENGAGED
        self._controller0 = sample.controller
        self._target0 = self.target
        self._to_robot = self._robot_axes * sample.axes.inv()
        self._gripper_side = np.sign(sample.gripper - self.gripper)

    def _follow(self, sample: Sample) -> None:
        to_robot = self._to_robot
        increment = sample.controller.rotation * self._controller0.rotation.inv()
        self.target = Pose(
            self._target0.position
            + to_robot.apply(sample.controller.position - self._controller0.position),
            to_robot * increment * to_robot.inv() * self._target0.rotation,
        )
        self._follow_gripper(sample.gripper)

    def _follow_gripper(self, requested: float) -> None:
        # The trigger may be anywhere on a re-grip, so the gripper holds until
        # the trigger reaches it: re-gripping with the trigger open must not
        # open a closed gripper. Once they meet, the side becomes 0 for good.
        if np.sign(requested - self.gripper) != self._gripper_side:
            self._gripper_side = 0.0
        if self._gripper_side == 0.0:
            self.gripper = requested
