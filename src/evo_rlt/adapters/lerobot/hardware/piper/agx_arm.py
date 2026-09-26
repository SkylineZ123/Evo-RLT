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

"""Shared helpers for AgileX arms driven through the official `pyAgxArm` SDK.

`pyAgxArm` replaces the older bare-protocol `piper_sdk`. The important consequences for
everything built on top of this module:

* joint angles are **radians** and gripper width is **meters** on both the read and the
  write side, so no unit conversion belongs in LeRobot anymore;
* gripper force is **newtons**;
* the end effector is a separate driver object obtained from ``arm.init_effector(...)``,
  which may only be called once per arm instance;
* every read API returns ``MessageAbstract | None`` — ``None`` meaning "no such frame has
  been received yet" — instead of the zero-filled struct the old SDK handed back.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

PIPER_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
PIPER_JOINT_ACTION_KEYS = tuple(f"{joint}.pos" for joint in PIPER_JOINT_NAMES)
PIPER_ACTION_KEYS = PIPER_JOINT_ACTION_KEYS + ("gripper.pos",)

# `get_arm_status().msg.ctrl_mode` is an IntEnum whose values match the CAN protocol.
PIPER_CTRL_MODE_STANDBY = 0x00
PIPER_CTRL_MODE_CAN = 0x01
PIPER_CTRL_MODE_TEACH = 0x02
PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT = 0x06
# In these modes the arm acts as a leader/teaching device and rejects CAN commands.
PIPER_TEACH_CTRL_MODES = frozenset({PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT})

# Arm models `pyAgxArm` ships a Piper-series driver for.
PIPER_ARM_MODELS = ("piper", "piper_h", "piper_l", "piper_x")
# Driver profiles per firmware range, plus "auto" (LeRobot-side sentinel meaning
# "read the firmware version off the arm and pick the profile from it").
PIPER_FW_PROFILES = ("auto", "default", "v183", "v188", "v189")

# Gripper force is expressed in newtons and the firmware saturates above 3 N.
PIPER_GRIPPER_MAX_FORCE_N = 3.0


@lru_cache(maxsize=1)
def get_pyagxarm() -> SimpleNamespace:
    """Import `pyAgxArm` lazily so the module stays importable without the extra."""
    try:
        from pyAgxArm import (
            AgxArmFactory,
            ArmModel,
            PiperFW,
            create_agx_arm_config,
            resolve_firmware_profile,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `pyAgxArm`. Install the Piper extra first "
            "(`uv sync --extra piper`, or "
            "`pip install 'pyAgxArm @ git+https://github.com/agilexrobotics/pyAgxArm.git'`)."
        ) from exc

    return SimpleNamespace(
        AgxArmFactory=AgxArmFactory,
        ArmModel=ArmModel,
        PiperFW=PiperFW,
        create_agx_arm_config=create_agx_arm_config,
        resolve_firmware_profile=resolve_firmware_profile,
    )


def validate_piper_arm_model(arm_model: str) -> None:
    if arm_model not in PIPER_ARM_MODELS:
        raise ValueError(f"`arm_model` must be one of {list(PIPER_ARM_MODELS)}; got {arm_model!r}.")


def validate_piper_firmware_version(firmware_version: str) -> None:
    if firmware_version not in PIPER_FW_PROFILES:
        raise ValueError(
            f"`firmware_version` must be one of {list(PIPER_FW_PROFILES)}; got {firmware_version!r}."
        )


def validate_piper_gripper_force(force_n: float) -> None:
    if not (0.0 <= float(force_n) <= PIPER_GRIPPER_MAX_FORCE_N):
        raise ValueError(f"`gripper_force_n` must be between 0 and {PIPER_GRIPPER_MAX_FORCE_N} N.")


def _build_piper_arm(
    *,
    channel: str,
    arm_model: str,
    firmware_profile: str,
    interface: str,
    bitrate: int,
    log_level: str,
) -> Any:
    sdk = get_pyagxarm()
    config = sdk.create_agx_arm_config(
        robot=arm_model,
        firmeware_version=firmware_profile,  # vendor spelling
        channel=channel,
        interface=interface,
        bitrate=bitrate,
        log_level=log_level.upper(),
    )
    return sdk.AgxArmFactory.create_arm(config)


def _probe_firmware_profile(
    *,
    channel: str,
    arm_model: str,
    interface: str,
    bitrate: int,
    log_level: str,
    timeout_s: float,
) -> str:
    """Connect with the baseline driver, read the firmware version, and map it to a profile.

    Returns the ``default`` profile when the arm does not answer in time: a CAN hiccup at
    connect time should not abort a recording session.
    """
    sdk = get_pyagxarm()
    arm = _build_piper_arm(
        channel=channel,
        arm_model=arm_model,
        firmware_profile=sdk.PiperFW.DEFAULT,
        interface=interface,
        bitrate=bitrate,
        log_level=log_level,
    )
    try:
        arm.connect()
        deadline = time.monotonic() + max(0.0, timeout_s)
        firmware = None
        while True:
            firmware = arm.get_firmware()
            if firmware is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.1)

        if firmware is None:
            logger.warning(
                "[%s] No firmware answer within %.1fs; falling back to the '%s' driver profile.",
                channel,
                timeout_s,
                sdk.PiperFW.DEFAULT,
            )
            return sdk.PiperFW.DEFAULT

        software_version = firmware["software_version"]
        try:
            profile = sdk.resolve_firmware_profile(arm_model, software_version)
        except (TypeError, ValueError):
            logger.warning(
                "[%s] Could not map firmware %r to a driver profile; using '%s'.",
                channel,
                software_version,
                sdk.PiperFW.DEFAULT,
            )
            return sdk.PiperFW.DEFAULT

        logger.info(
            "[%s] Detected %s firmware %s -> driver profile '%s'.",
            channel,
            arm_model,
            software_version,
            profile,
        )
        return profile
    finally:
        arm.disconnect()


def make_piper_arm(
    *,
    channel: str,
    arm_model: str = "piper",
    firmware_version: str = "auto",
    interface: str = "socketcan",
    bitrate: int = 1_000_000,
    log_level: str = "WARNING",
    with_gripper: bool = True,
    enforce_joint_limits: bool = False,
    firmware_timeout_s: float = 3.0,
) -> tuple[Any, Any | None, str]:
    """Build, configure and connect one Piper-series arm.

    Returns ``(arm, gripper, firmware_profile)``; ``gripper`` is None when
    ``with_gripper=False``.

    With ``firmware_version="auto"`` the arm is first opened with the baseline driver
    just to read ``get_firmware()``, then closed and rebuilt on the profile that firmware
    maps to — the flow the SDK's own `detect_piper_series` demo uses. Pin
    ``firmware_version`` to the measured profile to skip that extra connect cycle.

    ``enforce_joint_limits`` mirrors the SDK's own default (off). Turning it on makes
    ``move_j``/``move_js`` clamp to the model's preset joint limits, which silently
    truncates motion if ``arm_model`` does not match the hardware.
    """
    validate_piper_arm_model(arm_model)
    validate_piper_firmware_version(firmware_version)

    if firmware_version == "auto":
        firmware_profile = _probe_firmware_profile(
            channel=channel,
            arm_model=arm_model,
            interface=interface,
            bitrate=bitrate,
            log_level=log_level,
            timeout_s=firmware_timeout_s,
        )
    else:
        firmware_profile = firmware_version

    arm = _build_piper_arm(
        channel=channel,
        arm_model=arm_model,
        firmware_profile=firmware_profile,
        interface=interface,
        bitrate=bitrate,
        log_level=log_level,
    )
    try:
        # `init_effector` registers the gripper parser on the driver context, so do it
        # before the read thread starts to avoid dropping the first frames.
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER) if with_gripper else None
        arm.set_joint_limits_enabled(bool(enforce_joint_limits))
        arm.connect()
    except Exception:
        arm.disconnect()
        raise

    logger.info(
        "[%s] Connected %s arm on %s (driver profile '%s', joint limits %s).",
        channel,
        arm_model,
        interface,
        firmware_profile,
        "on" if enforce_joint_limits else "off",
    )
    return arm, gripper, firmware_profile


def wait_enable_piper(arm: Any, timeout_s: float, retry_interval_s: float = 0.2) -> bool:
    """Keep requesting the enabled state until the arm reports every joint enabled."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    interval_s = max(0.01, retry_interval_s)
    while True:
        if bool(arm.enable()):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return False
        time.sleep(min(interval_s, remaining_s))


def _piper_mode_to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "value") and isinstance(value.value, int):
        return int(value.value)
    if hasattr(value, "__int__"):
        return int(value)
    return None


def read_piper_ctrl_mode(
    arm: Any,
    timeout_s: float = 1.0,
    poll_s: float = 0.02,
    min_timestamp: float = 0.0,
) -> int | None:
    """Read the arm's current ctrl_mode, returning None when CAN data is unavailable.

    Only accept status frames newer than `min_timestamp`. The SDK caches the latest
    frame, which may predate a mode switch; pair this with
    :func:`read_piper_status_timestamp` to require fresh feedback.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        status = arm.get_arm_status()
        if status is not None:
            stamp = float(getattr(status, "timestamp", 0.0) or 0.0)
            if stamp > 0.0 and stamp > min_timestamp:
                mode = _piper_mode_to_int(getattr(status.msg, "ctrl_mode", None))
                if mode is not None:
                    return mode
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.005, poll_s))


def read_piper_status_timestamp(arm: Any) -> float:
    """Return the cached status timestamp used as a freshness baseline."""
    status = arm.get_arm_status()
    if status is None:
        return 0.0
    return float(getattr(status, "timestamp", 0.0) or 0.0)


def seed_piper_position_target(arm: Any) -> bool:
    """Seed the CAN position target with the measured current pose.

    When switching from teaching mode, the position loop otherwise tracks the last
    target stored in firmware, which may be stale and cause a sudden jump. Sending the
    current measured joint pose first keeps the arm stationary during the switch.
    """
    joint_angles = arm.get_joint_angles()
    if joint_angles is None:
        return False
    arm.move_j([float(value) for value in joint_angles.msg])
    return True


def recover_piper_from_teach_mode(
    arm: Any,
    *,
    interface_name: str,
    attempts: int = 3,
    settle_s: float = 0.5,
    timeout_s: float = 1.0,
    poll_s: float = 0.02,
    speed_ratio: int = 30,
) -> int | None:
    """Recover an arm from leader/teaching role and return its resulting ctrl_mode.

    Recovery has two stages, checking after each one:

    1. `set_follower_mode()` restores the follower role.
    2. If teaching control remains active, seed the position target and explicitly
       request CAN control by switching back to joint motion mode.

    None means no status was received. A teaching-mode result means recovery failed and
    the arm firmware probably requires a power cycle.
    """
    mode: int | None = None
    for attempt in range(1, max(1, attempts) + 1):
        stamp = read_piper_status_timestamp(arm)
        arm.set_follower_mode()
        if settle_s > 0:
            time.sleep(settle_s)
        mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s, min_timestamp=stamp)
        if mode is not None and mode not in PIPER_TEACH_CTRL_MODES:
            return mode

        stamp = read_piper_status_timestamp(arm)
        # Only kick the arm into joint motion mode once the position target holds the measured
        # pose. Without a seed the firmware would resume tracking whatever target it still has
        # stored, which is the very jump `seed_piper_position_target` exists to prevent — and
        # seeding fails exactly when the arm sends no feedback, i.e. when it is still a leader.
        if not seed_piper_position_target(arm):
            logger.warning(
                "[%s] No joint feedback to seed the position target; skipping the motion-mode "
                "request this round rather than risking a jump.",
                interface_name,
            )
            continue
        arm.set_speed_percent(int(speed_ratio))
        arm.set_motion_mode("j")
        if settle_s > 0:
            time.sleep(settle_s)
        mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s, min_timestamp=stamp)
        if mode is not None and mode not in PIPER_TEACH_CTRL_MODES:
            return mode

        logger.warning(
            "[%s] Follower-mode recovery attempt %d/%d did not take effect (ctrl_mode=%s); retrying.",
            interface_name,
            attempt,
            attempts,
            "unavailable" if mode is None else f"0x{mode:02X}",
        )
    return mode


def guard_piper_ctrl_mode_on_connect(
    arm: Any,
    *,
    interface_name: str,
    timeout_s: float = 0.5,
    poll_s: float = 0.02,
    settle_s: float = 0.05,
    auto_recover: bool = True,
    recover_attempts: int = 3,
    recover_settle_s: float = 0.5,
) -> None:
    """Verify after connection that the arm is in a mode that accepts CAN commands.

    Teaching mode is automatically recovered to follower mode by default. Set
    `auto_recover=False` to request follower mode once and require a power cycle instead.

    A *silent* arm is treated as another symptom of the same fault rather than as a wiring
    problem. An arm left in the leader role — `set_leader_mode()`, i.e. linkage config 0xFA,
    which survives a power cycle — stops broadcasting its status and joint feedback
    altogether; it only answers explicit queries such as the firmware read, so the bus looks
    dead to `read_piper_ctrl_mode` even though the arm is powered and wired. Recovery is
    therefore attempted before the wiring is blamed, and the wiring error is only raised once
    a follower-mode request has failed to bring the status frames back.
    """
    mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s)
    if mode is not None and mode not in PIPER_TEACH_CTRL_MODES:
        return

    # No status frame at all: most likely the leader role described above, but a genuinely
    # unpowered or unplugged arm looks identical from here, hence the two-sided wording.
    silent = mode is None

    if not auto_recover:
        arm.set_follower_mode()
        if settle_s > 0:
            time.sleep(settle_s)
        if silent:
            raise RuntimeError(
                f"[{interface_name}] no arm status frame within {timeout_s:.2f}s. Either the arm is "
                "unpowered/unwired, or it is still in the leader role, in which case it broadcasts "
                "nothing. Follower mode has been requested. Power-cycle this arm, then retry."
            )
        raise RuntimeError(
            f"[{interface_name}] arm is in master/teaching role (ctrl_mode=0x{mode:02X}). "
            "Follower mode has been requested. Power-cycle this arm, then retry."
        )

    if silent:
        logger.warning(
            "[%s] Arm sends no status frames; it is probably still in the leader role from an "
            "earlier session (that role persists across power cycles and suppresses feedback). "
            "Requesting follower mode. The position loop will engage during the switch; keep "
            "clear of the motion envelope.",
            interface_name,
        )
    else:
        logger.warning(
            "[%s] Arm is in leader/teaching mode (ctrl_mode=0x%02X); recovering follower mode. "
            "The position loop will engage during the switch; keep clear of the motion envelope.",
            interface_name,
            mode,
        )
    recovered = recover_piper_from_teach_mode(
        arm,
        interface_name=interface_name,
        attempts=recover_attempts,
        settle_s=recover_settle_s,
        timeout_s=max(timeout_s, 1.0),
        poll_s=poll_s,
    )
    if recovered is None:
        raise RuntimeError(
            f"[{interface_name}] no arm status frame after {recover_attempts} follower-mode requests. "
            "The arm is unpowered, not wired to this CAN interface, or its firmware needs a power "
            "cycle. Check power and CAN wiring, then rerun."
        )
    if recovered in PIPER_TEACH_CTRL_MODES:
        raise RuntimeError(
            f"[{interface_name}] automatic follower-mode recovery failed after {recover_attempts} attempts "
            f"(ctrl_mode=0x{recovered:02X}). Follower mode was requested; power-cycle this arm and retry."
        )
    logger.info(
        "[%s] Recovered follower mode (ctrl_mode=0x%02X); continuing connection.", interface_name, recovered
    )
