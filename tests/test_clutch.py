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

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from dora_openarm_webxr.clutch import (
    ClutchState,
    HandClutch,
    Pose,
    Sample,
    parse_grip,
    parse_pose,
)

ROBOT_AXES = Rotation.from_matrix([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
IDENTITY = Rotation.identity()
INITIAL = Pose(
    np.array([0.2, -0.15, -0.2]), Rotation.from_euler("y", -90, degrees=True)
)
OPEN_GRIPPER = -0.8


def _clutch():
    return HandClutch(INITIAL, OPEN_GRIPPER, ROBOT_AXES)


def _sample(
    grip=1.0, position=(0.0, 0.0, 0.0), rotation=IDENTITY, axes=IDENTITY, gripper=-0.8
):
    return Sample(grip, Pose(np.array(position, dtype=float), rotation), axes, gripper)


def _engaged(**fields):
    hand = _clutch()
    hand.update(_sample(grip=0.0))
    hand.update(_sample(grip=1.0, **fields))
    assert hand.state is ClutchState.ENGAGED
    return hand


def _same_pose(a, b):
    np.testing.assert_allclose(a.position, b.position, atol=1e-9)
    assert (a.rotation * b.rotation.inv()).magnitude() == pytest.approx(0.0, abs=1e-7)


def test_a_held_grip_does_not_engage_a_new_clutch():
    hand = _clutch()
    for _ in range(3):
        assert hand.update(_sample(grip=1.0)) is ClutchState.AWAIT_RELEASE


def test_open_then_squeeze_engages_exactly_at_the_reference():
    hand = _clutch()
    assert (
        hand.update(_sample(grip=0.0, position=(9.0, 9.0, 9.0))) is ClutchState.RELEASED
    )
    state = hand.update(
        _sample(
            grip=1.0, position=(9.0, 9.0, 9.0), rotation=Rotation.from_euler("x", 2.0)
        )
    )
    assert state is ClutchState.ENGAGED
    _same_pose(hand.target, INITIAL)


def test_regrip_from_a_different_pose_does_not_jump():
    hand = _engaged()
    hand.update(_sample(position=(0.1, 0.0, 0.0)))
    held = hand.target
    hand.update(_sample(grip=0.0, position=(0.1, 0.0, 0.0)))
    assert hand.state is ClutchState.RELEASED

    far = Rotation.from_euler("zyx", [1.0, -2.0, 0.5])
    hand.update(_sample(grip=0.0, position=(5.0, -3.0, 2.0), rotation=far))
    _same_pose(hand.target, held)
    hand.update(_sample(grip=1.0, position=(5.0, -3.0, 2.0), rotation=far))
    assert hand.state is ClutchState.ENGAGED
    _same_pose(hand.target, held)


def test_translation_maps_through_the_robot_axes():
    hand = _engaged()
    hand.update(_sample(position=(0.1, 0.2, -0.3)))
    # x right, y up, -z forward in XR is -y, +z and +x at the robot.
    np.testing.assert_allclose(
        hand.target.position - INITIAL.position, [0.3, -0.1, 0.2], atol=1e-12
    )
    assert (hand.target.rotation * INITIAL.rotation.inv()).magnitude() < 1e-9


def test_rotation_is_about_the_mapped_axis_and_keeps_the_position():
    hand = _engaged()
    hand.update(_sample(rotation=Rotation.from_euler("y", 0.3)))
    # Turning about XR's up is turning about the robot's up, wherever the end
    # effector happens to be pointing.
    increment = hand.target.rotation * INITIAL.rotation.inv()
    np.testing.assert_allclose(increment.as_rotvec(), [0.0, 0.0, 0.3], atol=1e-9)
    np.testing.assert_allclose(hand.target.position, INITIAL.position, atol=1e-12)


def test_axes_are_those_of_the_engagement():
    # Engaged facing left: forward for the operator is -x in XR.
    facing_left = Rotation.from_euler("y", 90, degrees=True)
    hand = _engaged(axes=facing_left)
    hand.update(_sample(position=(-0.1, 0.0, 0.0), axes=facing_left))
    np.testing.assert_allclose(
        hand.target.position - INITIAL.position, [0.1, 0.0, 0.0], atol=1e-12
    )


def test_head_movement_while_engaged_does_not_move_the_target():
    hand = _engaged()
    hand.update(_sample(position=(0.1, 0.0, 0.0)))
    held = hand.target
    hand.update(_sample(position=(0.1, 0.0, 0.0), axes=Rotation.from_euler("y", 1.0)))
    _same_pose(hand.target, held)


@pytest.mark.parametrize("grip", [0.45, 0.5, 0.55])
def test_the_hysteresis_band_changes_nothing(grip):
    released = _clutch()
    released.update(_sample(grip=0.0))
    assert released.update(_sample(grip=grip)) is ClutchState.RELEASED
    engaged = _engaged()
    assert engaged.update(_sample(grip=grip)) is ClutchState.ENGAGED


@pytest.mark.parametrize("from_engaged", [True, False])
def test_a_fault_needs_an_open_grip_before_a_squeeze_counts(from_engaged):
    hand = _engaged() if from_engaged else _clutch()
    held = hand.target
    assert hand.update(None) is ClutchState.AWAIT_RELEASE
    assert (
        hand.update(_sample(grip=1.0, position=(1.0, 0.0, 0.0)))
        is ClutchState.AWAIT_RELEASE
    )
    _same_pose(hand.target, held)
    assert hand.update(_sample(grip=0.0)) is ClutchState.RELEASED
    assert (
        hand.update(_sample(grip=1.0, position=(2.0, 0.0, 0.0))) is ClutchState.ENGAGED
    )
    _same_pose(hand.target, held)


def test_revoke_keeps_the_target_and_withdraws_authorization():
    hand = _engaged()
    hand.update(_sample(position=(0.1, 0.0, 0.0)))
    held = hand.target
    hand.revoke()
    assert (
        hand.update(_sample(grip=1.0, position=(0.2, 0.0, 0.0)))
        is ClutchState.AWAIT_RELEASE
    )
    _same_pose(hand.target, held)


def test_release_holds_the_gripper():
    hand = _engaged(gripper=-0.8)
    hand.update(_sample(gripper=0.1))
    assert hand.gripper == 0.1
    hand.update(_sample(grip=0.0, gripper=-0.8))
    assert hand.gripper == 0.1


@pytest.mark.parametrize(
    ("held", "start", "meets"), [(0.1, -0.8, 0.2), (-0.5, 0.1, -0.6)]
)
def test_regrip_keeps_the_gripper_until_the_trigger_reaches_it(held, start, meets):
    hand = _engaged(gripper=-0.8)
    hand.update(_sample(gripper=held))
    hand.update(_sample(grip=0.0, gripper=held))

    hand.update(_sample(grip=1.0, gripper=start))
    assert hand.gripper == held
    hand.update(_sample(gripper=start))
    assert hand.gripper == held
    hand.update(_sample(gripper=meets))
    assert hand.gripper == meets
    # Once they have met it follows the trigger in both directions.
    hand.update(_sample(gripper=start))
    assert hand.gripper == start


GOOD_POSE = {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}


@pytest.mark.parametrize(
    "fields",
    [
        None,
        {},
        "pose",
        {**GOOD_POSE, "x": None},
        {**GOOD_POSE, "y": "up"},
        {**GOOD_POSE, "z": math.nan},
        {**GOOD_POSE, "x": math.inf},
        {**GOOD_POSE, "qw": math.nan},
        {**GOOD_POSE, "qw": 0.0},
        {**GOOD_POSE, "qw": 2.0},
        {key: value for key, value in GOOD_POSE.items() if key != "qz"},
    ],
)
def test_parse_pose_rejects(fields):
    assert parse_pose(fields) is None


def test_parse_pose_accepts():
    pose = parse_pose({**GOOD_POSE, "x": 0.5})
    np.testing.assert_array_equal(pose.position, [0.5, 0.0, 0.0])
    assert pose.rotation.magnitude() == 0.0


@pytest.mark.parametrize("value", [None, "x", [], -0.01, 1.01, math.nan, math.inf])
def test_parse_grip_rejects(value):
    assert parse_grip(value) is None


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_parse_grip_accepts(value):
    assert parse_grip(value) == value


# What an operator can do between two ticks. Anything goes, and each property
# below has to hold whatever the sequence.
_steps = st.lists(
    st.one_of(
        st.none(),
        st.tuples(
            st.sampled_from([0.0, 0.39, 0.4, 0.5, 0.6, 0.61, 1.0]),
            st.tuples(*[st.floats(-3.0, 3.0)] * 3),
            st.tuples(*[st.floats(-3.0, 3.0)] * 3),
            st.floats(-3.0, 3.0),
            st.floats(-1.0, 1.0),
        ),
    ),
    max_size=40,
)


def _to_sample(step):
    if step is None:
        return None
    grip, position, rotvec, yaw, gripper = step
    return _sample(
        grip=grip,
        position=position,
        rotation=Rotation.from_rotvec(rotvec),
        axes=Rotation.from_euler("y", yaw),
        gripper=gripper,
    )


@settings(deadline=None)
@given(_steps)
def test_property_authorization(steps):
    # An independent statement of the rule: engaged only by a squeeze after an
    # open grip with no fault in between, and only ever left by a release or a
    # fault. The clutch must agree with it at every step.
    hand = _clutch()
    expected = "waiting"
    for step in steps:
        grip = None if step is None else step[0]
        if grip is None:
            expected = "waiting"
        elif expected == "waiting":
            expected = "open" if grip <= 0.4 else "waiting"
        elif expected == "open":
            expected = "engaged" if grip >= 0.6 else "open"
        elif grip <= 0.4:
            expected = "open"
        state = hand.update(_to_sample(step))
        assert (
            state
            is {
                "waiting": ClutchState.AWAIT_RELEASE,
                "open": ClutchState.RELEASED,
                "engaged": ClutchState.ENGAGED,
            }[expected]
        )


@settings(deadline=None)
@given(_steps)
def test_property_continuity(steps):
    # The target only ever moves on a step that follows an engaged one, and then
    # by exactly as much as the controller moved.
    hand = _clutch()
    previous_sample = None
    for step in steps:
        sample = _to_sample(step)
        was_engaged = hand.state is ClutchState.ENGAGED
        before = hand.target
        state = hand.update(sample)
        moved = np.linalg.norm(hand.target.position - before.position)
        turned = (hand.target.rotation * before.rotation.inv()).magnitude()
        if was_engaged and state is ClutchState.ENGAGED:
            controller_moved = np.linalg.norm(
                sample.controller.position - previous_sample.controller.position
            )
            controller_turned = (
                sample.controller.rotation * previous_sample.controller.rotation.inv()
            ).magnitude()
            assert moved == pytest.approx(controller_moved, abs=1e-9)
            assert turned == pytest.approx(controller_turned, abs=1e-6)
        else:
            assert moved == 0.0
            assert turned == 0.0
        assert np.all(np.isfinite(hand.target.position))
        previous_sample = sample


@settings(deadline=None)
@given(_steps)
def test_property_deterministic(steps):
    first, second = _clutch(), _clutch()
    for step in steps:
        assert first.update(_to_sample(step)) is second.update(_to_sample(step))
        np.testing.assert_array_equal(first.target.position, second.target.position)
        np.testing.assert_array_equal(
            first.target.rotation.as_quat(), second.target.rotation.as_quat()
        )
        assert first.gripper == second.gripper
