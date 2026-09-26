#!/usr/bin/env python

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

from dataclasses import dataclass

from evo_rlt.adapters.lerobot.hardware.piper.agx_arm import (
    PIPER_JOINT_NAMES,
    validate_piper_arm_model,
    validate_piper_firmware_version,
    validate_piper_gripper_force,
)

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class PiperLeaderConfigBase:
    """Configuration for a Piper leader arm used as teleoperator.

    Backdriving the leader relies on the arm's built-in teaching mode: the operator
    presses the teach button on the arm (the button lights up green) and the firmware
    releases the joints. Pressing it again returns the arm to its normal mode. No
    software gravity compensation is involved, so the host never has to model the
    arm's dynamics or stream MIT torque commands.
    """

    # CAN interface name (e.g. "can1")
    port: str

    # pyAgxArm connection options. `arm_model` must match the hardware; `firmware_version`
    # defaults to "auto", which reads the version off the arm and picks the matching driver
    # profile. Pin it (default/v183/v188/v189) to skip the extra probe connection.
    arm_model: str = "piper"
    firmware_version: str = "auto"
    can_interface: str = "socketcan"
    can_bitrate: int = 1_000_000
    log_level: str = "WARNING"
    startup_sleep_s: float = 0.1
    # SDK-side clamping to the `arm_model` joint presets; off matches the SDK default.
    enforce_joint_limits: bool = False

    # Leave the arm passive on connect so the operator can drive it with the
    # teach button. When false, the arm is put in CAN command mode instead.
    manual_control: bool = True

    # Time given to the firmware to apply the follower-role frame before the arm is read
    # back or commanded. `begin_alignment` waits this long after sending it.
    follower_mode_settle_s: float = 0.5

    # Reject joint/gripper samples older than this before building an action, so a leader
    # that has gone quiet cannot drive the follower from a stale pose. The last good action
    # is held instead. 0 disables the check.
    max_feedback_age_s: float = 0.5

    # Read control messages from leader first, fallback to feedback state if missing
    prefer_ctrl_messages: bool = True
    fallback_to_feedback: bool = True
    # In manual-control mode, optionally wait a short time for a newer feedback sample.
    # Set to 0 to always use the latest available sample without blocking the control loop.
    manual_control_fresh_feedback_wait_timeout_s: float = 0.0
    manual_control_fresh_feedback_poll_s: float = 0.001

    # Teach-button detection. `is_teach_mode_active()` reads the arm's ctrl_mode
    # from the status frame; the result is cached for this interval so callers can poll
    # it every control tick without hammering the SDK.
    teach_mode_poll_interval_s: float = 0.02
    # A teach-mode reading must persist for this long before it is reported, which
    # rejects the transient ctrl_mode flicker seen while the firmware switches modes.
    teach_mode_debounce_s: float = 0.05

    # Gripper handling
    sync_gripper: bool = True
    # Clamping force in newtons; the firmware saturates above 3 N.
    gripper_force_n: float = 1.0
    # Name the gripper carries in the action/feedback dicts this leader exchanges with the
    # rest of LeRobot, written as ``<gripper_joint>.pos``. The Piper follower exposes its
    # gripper as the 7th motor (``joint_7.pos``), and the recorder feeds the teleoperator's
    # action straight into ``robot.send_action`` and into the dataset's ``action`` column,
    # so the two key spaces have to agree — hence the default. ``DualPiperLeader`` sets this
    # to "gripper" because it does the left/right joint_7/joint_14 remapping itself.
    gripper_joint: str = "joint_7"

    # Command mode for send_feedback
    command_speed_ratio: int = 100
    command_high_follow: bool = True
    mode_refresh_interval_s: float = 1.0
    enable_timeout_s: float = 3.0

    # Calibration precision:
    # homing_offset/range_min/range_max are stored as "radian * calibration_scale"
    # (meters for the gripper entry).
    calibration_scale: int = 1000
    # Whether `connect(calibrate=True)` offers to run the calibration flow when no
    # calibration file exists. It does NOT decide how a missing calibration is read back:
    # an uncalibrated leader always forwards absolute values (radians / meters), which is
    # the right key space for a same-model Piper leader/follower pair.
    require_calibration: bool = True

    # Safety behavior on disconnect
    disable_on_disconnect: bool = False


def _validate_piper_leader_config(config: PiperLeaderConfigBase) -> None:
    if not (0 <= config.command_speed_ratio <= 100):
        raise ValueError("`command_speed_ratio` must be between 0 and 100.")
    if config.mode_refresh_interval_s < 0:
        raise ValueError("`mode_refresh_interval_s` must be >= 0.")
    if config.enable_timeout_s < 0:
        raise ValueError("`enable_timeout_s` must be >= 0.")
    if config.teach_mode_poll_interval_s < 0:
        raise ValueError("`teach_mode_poll_interval_s` must be >= 0.")
    if config.teach_mode_debounce_s < 0:
        raise ValueError("`teach_mode_debounce_s` must be >= 0.")
    if config.follower_mode_settle_s < 0:
        raise ValueError("`follower_mode_settle_s` must be >= 0.")
    if config.max_feedback_age_s < 0:
        raise ValueError("`max_feedback_age_s` must be >= 0.")
    if config.calibration_scale <= 0:
        raise ValueError("`calibration_scale` must be > 0.")
    if not isinstance(config.require_calibration, bool):
        raise ValueError("require_calibration must be true or false.")
    if config.startup_sleep_s < 0:
        raise ValueError("`startup_sleep_s` must be >= 0.")
    if config.manual_control_fresh_feedback_wait_timeout_s < 0:
        raise ValueError("`manual_control_fresh_feedback_wait_timeout_s` must be >= 0.")
    if config.manual_control_fresh_feedback_poll_s <= 0:
        raise ValueError("`manual_control_fresh_feedback_poll_s` must be > 0.")
    validate_piper_arm_model(config.arm_model)
    validate_piper_firmware_version(config.firmware_version)
    validate_piper_gripper_force(config.gripper_force_n)
    if not config.gripper_joint or not config.gripper_joint.strip():
        raise ValueError("`gripper_joint` must be a non-empty joint name.")
    if config.gripper_joint in PIPER_JOINT_NAMES:
        raise ValueError(
            f"`gripper_joint` must not collide with an arm joint; got '{config.gripper_joint}', "
            f"which is one of {list(PIPER_JOINT_NAMES)}."
        )


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig, PiperLeaderConfigBase):
    def __post_init__(self):
        _validate_piper_leader_config(self)


@dataclass
class PiperXLeaderConfigBase(PiperLeaderConfigBase):
    pass


@TeleoperatorConfig.register_subclass("piperx_leader")
@dataclass
class PiperXLeaderConfig(TeleoperatorConfig, PiperXLeaderConfigBase):
    def __post_init__(self):
        _validate_piper_leader_config(self)
