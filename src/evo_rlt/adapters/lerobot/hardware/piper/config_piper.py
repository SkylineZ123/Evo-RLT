#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401  (registers "intelrealsense")
from lerobot.robots.config import RobotConfig
from evo_rlt.adapters.lerobot.hardware.piper.agx_arm import (
    validate_piper_arm_model,
    validate_piper_firmware_version,
    validate_piper_gripper_force,
)


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    can_port: str = "can_left"
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i + 1}" for i in range(7)])

    arm_model: str = "piper"
    firmware_version: str = "auto"
    can_interface: str = "socketcan"
    can_bitrate: int = 1_000_000
    log_level: str = "WARNING"
    speed_percent: int = 60  # Speed percentage applied to every joint command
    enforce_joint_limits: bool = False  # SDK-side clamping to the `arm_model` joint presets

    # Reset and safety: smooth interpolation prevents abrupt motion from an unknown MIT-mode pose.
    home_position: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )  # Initial pose for six joints and the gripper
    reset_hz: float = 100.0  # Smooth-reset command frequency
    reset_duration_s: float = 4.0  # Desired smooth-reset duration
    max_joint_step_rad: float = 0.01  # Maximum joint increment per step
    open_gripper_on_init: bool = True  # Open the gripper during the initial reset
    gripper_open_range: float = 0.07  # Gripper opening in meters (0 to 0.08)
    gripper_force_n: float = 1.0  # Gripper clamping force in newtons (0 to 3)
    # Cameras come from the setup manifest (`--robot.cameras=...`); no hard-coded serials.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        validate_piper_arm_model(self.arm_model)
        validate_piper_firmware_version(self.firmware_version)
        validate_piper_gripper_force(self.gripper_force_n)
        if not (0 <= self.speed_percent <= 100):
            raise ValueError("`speed_percent` must be between 0 and 100.")
