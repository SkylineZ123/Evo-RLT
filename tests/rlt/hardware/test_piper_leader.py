# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""The Piper leader's public key space, which has to match the follower it drives.

Neither the leader nor the follower touches CAN until `connect()`, so these construct the
real objects and read nothing but configuration.
"""

import time
from types import SimpleNamespace

import pytest

from evo_rlt.adapters.lerobot.hardware.piper import Piper, PiperConfig
from evo_rlt.adapters.lerobot.hardware.piper.piper_leader import PiperLeader
from evo_rlt.adapters.lerobot.hardware.piper.config_piper_leader import PiperLeaderConfig


@pytest.fixture
def leader_factory(tmp_path):
    def _leader(**kwargs) -> PiperLeader:
        return PiperLeader(PiperLeaderConfig(port="can_left_l", calibration_dir=tmp_path, **kwargs))

    return _leader


def test_leader_and_follower_agree_on_the_action_key_space(leader_factory, tmp_path):
    # Regression: the leader used to emit `gripper.pos` while the follower's 7th motor is
    # `joint_7`, so the first recorded frame raised KeyError twice over - once in
    # `robot.send_action`, once in `build_dataset_frame` against the follower-derived
    # `action` feature.
    follower = Piper(PiperConfig(can_port="can_left_f", cameras={}, calibration_dir=tmp_path))
    leader = leader_factory()

    assert set(leader.action_features) == set(follower.action_features)
    assert "joint_7.pos" in leader.action_features
    assert "gripper.pos" not in leader.action_features


def test_constructing_a_leader_does_not_open_can(leader_factory):
    leader = leader_factory()
    assert leader.arm is None
    assert leader.gripper is None
    assert not leader.is_connected


def test_feedback_features_match_action_features(leader_factory):
    leader = leader_factory()
    assert leader.feedback_features == leader.action_features


def test_gripper_joint_is_configurable_for_composed_leaders(leader_factory):
    # DualPiperLeader drives two of these and does its own joint_7 / joint_14 remapping,
    # so it pins the sub-leaders back to the internal name.
    leader = leader_factory(gripper_joint="gripper")
    assert "gripper.pos" in leader.action_features
    assert "joint_7.pos" not in leader.action_features


def test_to_public_action_renames_only_the_gripper(leader_factory):
    leader = leader_factory()
    raw = {f"joint_{i}.pos": float(i) for i in range(1, 7)}
    raw["gripper.pos"] = 0.5

    public = leader._to_public_action(raw)

    assert public == {**{f"joint_{i}.pos": float(i) for i in range(1, 7)}, "joint_7.pos": 0.5}
    # The internal dict is left alone; only a renamed copy goes out.
    assert "gripper.pos" in raw


def test_to_public_action_is_a_no_op_when_the_names_already_agree(leader_factory):
    leader = leader_factory(gripper_joint="gripper")
    raw = {"joint_1.pos": 1.0, "gripper.pos": 0.5}
    assert leader._to_public_action(raw) == raw


@pytest.mark.parametrize("bad", ["", "   ", "joint_3"])
def test_gripper_joint_is_validated(bad):
    with pytest.raises(ValueError):
        PiperLeaderConfig(port="can_left_l", gripper_joint=bad)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm_model", "piper_z"),
        ("firmware_version", "S-V1.8-8"),  # a firmware string, not a driver profile
        ("gripper_force_n", 5.0),
    ],
)
def test_sdk_options_are_validated(field, value):
    with pytest.raises(ValueError):
        PiperLeaderConfig(port="can_left_l", **{field: value})


class _FakeMessage:
    def __init__(self, values, timestamp):
        self.msg = values
        self.timestamp = timestamp


class _FakeGripperMessage:
    def __init__(self, value, timestamp):
        self.msg = SimpleNamespace(value=value)
        self.timestamp = timestamp


class _FakeLeaderArm:
    """An arm whose feedback can be frozen, the way the CAN leader role freezes a Piper.

    pyAgxArm keeps handing back the last frame it parsed once the arm stops broadcasting,
    so "frozen" here means a constant timestamp, not `None`.
    """

    def __init__(self):
        self.pose = [0.1] * 6
        self.gripper_value = 0.05
        self.frozen = False
        self._stamp = time.time()
        self.follower_mode_calls = 0

    def _timestamp(self):
        if not self.frozen:
            self._stamp = time.time()
        return self._stamp

    def get_joint_angles(self):
        return _FakeMessage(list(self.pose), self._timestamp())

    def get_gripper_status(self):
        return _FakeGripperMessage(self.gripper_value, self._timestamp())

    def get_gripper_ctrl_states(self):
        return self.get_gripper_status()

    def disable_gripper(self):
        pass

    def move_gripper_m(self, position, force):
        pass

    def set_follower_mode(self):
        self.follower_mode_calls += 1
        self.frozen = False

    def set_speed_percent(self, percent):
        pass

    def set_motion_mode(self, mode):
        pass

    def enable(self):
        return True


@pytest.fixture
def connected_leader(leader_factory, monkeypatch):
    def _connected(**kwargs) -> PiperLeader:
        leader = leader_factory(**kwargs)
        arm = _FakeLeaderArm()
        leader.arm = arm
        leader.gripper = arm
        leader._is_connected = True
        # Manual control is the state the recorder hands the arm over in.
        leader._manual_control_enabled = True
        monkeypatch.setattr(leader, "is_teach_mode_active", lambda: False)
        return leader

    return _connected


def test_a_frozen_leader_does_not_command_the_follower_to_zero(connected_leader):
    """Regression: a stale pose used to fall through to an all-zero action.

    `Piper.send_action` forwards whatever it is given, so zeros meant every joint was
    commanded to home the moment the leader went quiet.
    """
    leader = connected_leader()
    first = leader.get_action()
    leader.arm.frozen = True
    time.sleep(leader.config.max_feedback_age_s + 0.05)

    held = leader.get_action()

    assert held == first
    assert not any(value == 0.0 for value in held.values())


def test_a_leader_that_never_reported_refuses_to_invent_an_action(connected_leader):
    leader = connected_leader()
    leader.arm.frozen = True
    leader.arm._stamp = time.time() - 60.0

    with pytest.raises(RuntimeError, match="never reported"):
        leader.get_action()


def test_fresh_feedback_still_flows_through(connected_leader):
    leader = connected_leader()
    leader.arm.pose = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    action = leader.get_action()

    assert action["joint_1.pos"] == pytest.approx(0.2)
    assert action["joint_6.pos"] == pytest.approx(0.7)


def test_max_feedback_age_zero_disables_the_staleness_check(connected_leader):
    leader = connected_leader(max_feedback_age_s=0.0)
    leader.get_action()
    leader.arm.frozen = True
    leader.arm._stamp = time.time() - 60.0
    leader.arm.pose = [0.9] * 6

    # The check is off, so even an ancient frame is taken at face value.
    assert leader.get_action()["joint_1.pos"] == pytest.approx(0.9)


def test_the_handover_never_touches_the_can_leader_role(connected_leader):
    """The arm the host has to keep reading must stay in the follower role.

    The firmware's leader role silences a Piper completely, which used to freeze the
    follower on the pose held at the handover. `_FakeLeaderArm` has no `set_leader_mode`
    at all, so reaching for it would raise here.
    """
    leader = connected_leader()

    leader.begin_alignment()
    leader.release_to_operator()

    assert leader.arm.follower_mode_calls == 1
    # Still readable, which is the whole point.
    assert leader.get_action()["joint_1.pos"] == pytest.approx(0.1)


def test_release_to_operator_stops_commanding_the_arm(connected_leader):
    leader = connected_leader()
    leader.begin_alignment()
    assert leader._manual_control_enabled is False

    leader.release_to_operator()

    assert leader._manual_control_enabled is True


@pytest.mark.parametrize(("field", "value"), [("max_feedback_age_s", -1.0)])
def test_new_options_are_validated(field, value):
    with pytest.raises(ValueError):
        PiperLeaderConfig(port="can_left_l", **{field: value})


# --------------------------------------------------------------------------- teach button


class _TeachButtonArm(_FakeLeaderArm):
    """Reports ctrl_mode / teach_status the way a Piper leader does around the teach button.

    Measured on the hardware: pressing the button off ends the drag (teach_status 2) but the
    arm stays in ctrl_mode TEACH (0x02) until the host sends a CAN-control mode frame.
    """

    def __init__(self, ctrl_mode=0x01, teach_status=0):
        super().__init__()
        self.ctrl_mode = ctrl_mode
        self.teach_status = teach_status
        self.calls = []

    def get_arm_status(self):
        msg = SimpleNamespace(ctrl_mode=self.ctrl_mode, teach_status=self.teach_status)
        return SimpleNamespace(msg=msg, timestamp=time.time())

    def move_j(self, joints):
        self.calls.append(("move_j", list(joints)))

    def set_motion_mode(self, mode):
        self.calls.append(("set_motion_mode", mode))
        if self.ctrl_mode == 0x02 and self.teach_status == 0x02:
            # The mode frame is what takes a released arm out of TEACH.
            self.ctrl_mode, self.teach_status = 0x01, 0x00


@pytest.fixture
def teach_leader(leader_factory):
    def _leader(ctrl_mode, teach_status) -> PiperLeader:
        leader = leader_factory(teach_mode_poll_interval_s=0.0, teach_mode_debounce_s=0.0)
        leader.arm = leader.gripper = _TeachButtonArm(ctrl_mode, teach_status)
        leader._is_connected = True
        leader._manual_control_enabled = True
        return leader

    return _leader


@pytest.mark.parametrize(
    ("ctrl_mode", "teach_status", "engaged"),
    [
        (0x01, 0x00, False),  # CAN control
        (0x00, 0x00, False),  # standby
        (0x02, 0x01, True),  # dragging
        (0x02, 0x02, False),  # drag ended, still in ctrl_mode TEACH
        (0x02, 0x00, True),  # unknown teach state: keep the operator in charge
        (0x06, 0x00, True),  # linkage teach input (leader role)
    ],
)
def test_teach_engaged_follows_the_drag_not_the_ctrl_mode(ctrl_mode, teach_status, engaged):
    from evo_rlt.adapters.lerobot.hardware.piper.agx_arm import piper_teach_engaged

    assert piper_teach_engaged(ctrl_mode, teach_status) is engaged


def test_releasing_the_teach_button_is_reported_although_ctrl_mode_stays_teach(teach_leader):
    leader = teach_leader(0x02, 0x01)
    # A first reading only arms the debounce; the second one reports it.
    leader.is_teach_mode_active()
    assert leader.is_teach_mode_active() is True

    leader.arm.teach_status = 0x02
    leader.is_teach_mode_active()
    assert leader.is_teach_mode_active() is False


def test_handing_back_after_the_drag_seeds_the_pose_then_requests_can_control(teach_leader):
    leader = teach_leader(0x02, 0x02)

    leader.set_manual_control(False)

    assert leader._manual_control_enabled is False
    assert leader.arm.calls[0] == ("move_j", leader.arm.pose)
    assert leader.arm.calls[1] == ("set_motion_mode", "js")
    assert leader.arm.ctrl_mode == 0x01


def test_no_can_request_while_the_operator_is_still_dragging(teach_leader):
    leader = teach_leader(0x02, 0x01)
    leader.is_teach_mode_active()  # debounce: report the drag

    leader.set_manual_control(False)

    assert leader.arm.calls == []
    assert leader._manual_control_enabled is True


def test_no_seed_when_the_arm_is_already_under_can_control(teach_leader):
    leader = teach_leader(0x01, 0x00)

    leader.set_manual_control(False)

    assert leader.arm.calls == [("set_motion_mode", "js")]
