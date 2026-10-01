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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.motors import MotorCalibration
from evo_rlt.adapters.lerobot.hardware.piper.agx_arm import (
    PIPER_ACTION_KEYS,
    PIPER_JOINT_ACTION_KEYS,
    PIPER_JOINT_NAMES,
    PIPER_TEACH_CTRL_MODES,
    guard_piper_ctrl_mode_on_connect,
    make_piper_arm,
    piper_teach_engaged,
    read_piper_mode_status,
    seed_piper_position_target,
    wait_enable_piper,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.utils import enter_pressed, move_cursor_up

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig, PiperXLeaderConfig

logger = logging.getLogger(__name__)
PIPER_CALIB_KEYS = list(PIPER_ACTION_KEYS)
PIPER_CALIB_IDS = {key: idx for idx, key in enumerate(PIPER_CALIB_KEYS)}
# Seconds between "the leader has gone quiet" warnings, which are otherwise emitted on
# every control tick.
_STALE_WARNING_INTERVAL_S = 5.0


class PiperLeader(Teleoperator):
    """Piper leader arm used as a teleoperator, driven through the pyAgxArm CAN driver.

    The operator backdrives the arm with its built-in teaching mode: pressing the
    teach button on the arm (it lights up green) releases the joints in firmware,
    and pressing it again hands control back. :meth:`is_teach_mode_active` exposes
    that button state so a DAgger session can start and stop human interventions
    from the arm itself, with no software gravity compensation in the loop.
    """

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig | PiperXLeaderConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._manual_control_enabled: bool | None = None
        self._last_mode_refresh_t = 0.0
        self._last_feedback_joint_timestamp = 0.0
        self._last_feedback_gripper_timestamp = 0.0
        # Last pose the arm actually reported, held while it is quiet so the follower is
        # never commanded a synthesised zero.
        self._last_raw_action: RobotAction | None = None
        self._last_stale_warning_t = 0.0

        self._teach_mode_active = False
        self._teach_mode_last_poll_t = 0.0
        self._teach_mode_pending: bool | None = None
        self._teach_mode_pending_since = 0.0
        # ctrl_mode from the latest status poll. It stays TEACH after the drag has ended,
        # until the host requests CAN control.
        self._last_ctrl_mode: int | None = None

        self.arm = None
        self.gripper = None
        self.firmware_profile: str | None = None

    @property
    def _gripper_key(self) -> str:
        """The gripper's key in the action/feedback dicts exchanged with the rest of LeRobot.

        Internally — raw SDK reads, calibration, :data:`PIPER_CALIB_KEYS` — the gripper is
        always ``gripper.pos``. Outward it is named after ``config.gripper_joint`` so the
        leader speaks the follower's key space (``joint_7.pos`` by default): the recorder
        feeds this action straight into ``robot.send_action`` and into the dataset's
        ``action`` column, and a mismatch there is a ``KeyError`` on the first frame.
        """
        return f"{self.config.gripper_joint}.pos"

    def _to_public_action(self, action: RobotAction) -> RobotAction:
        """Rename the internal ``gripper.pos`` entry to :attr:`_gripper_key`."""
        if self._gripper_key == "gripper.pos":
            return action
        renamed = {key: value for key, value in action.items() if key != "gripper.pos"}
        renamed[self._gripper_key] = action["gripper.pos"]
        return renamed

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_JOINT_ACTION_KEYS + (self._gripper_key,), float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict(self.action_features)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.arm, self.gripper, self.firmware_profile = make_piper_arm(
            channel=self.config.port,
            arm_model=self.config.arm_model,
            firmware_version=self.config.firmware_version,
            interface=self.config.can_interface,
            bitrate=self.config.can_bitrate,
            log_level=self.config.log_level,
            enforce_joint_limits=self.config.enforce_joint_limits,
        )
        if self.config.startup_sleep_s > 0:
            time.sleep(self.config.startup_sleep_s)
        try:
            guard_piper_ctrl_mode_on_connect(arm=self.arm, interface_name=self.config.port)
        except Exception:
            self._disconnect_arm()
            raise
        # This class throttles mode frames itself through `mode_refresh_interval_s`, so the
        # SDK does not need to re-send one ahead of every move command.
        self.arm.set_auto_set_motion_mode_enabled(False)

        self._is_connected = True
        # Recompute control mode on every fresh connection.
        self._manual_control_enabled = None
        self._last_feedback_joint_timestamp = 0.0
        self._last_feedback_gripper_timestamp = 0.0
        # A pose cached before this connection says nothing about where the arm is now.
        self._last_raw_action = None
        self._reset_teach_mode_state()
        try:
            self.configure()
            if not self.is_calibrated and calibrate and self.config.require_calibration:
                logger.info(
                    "No piper-leader calibration file found for '%s'. Running lerobot-calibrate flow.",
                    self.id,
                )
                self.calibrate()
        except Exception:
            self._disconnect_arm()
            self._is_connected = False
            raise

        if not self.is_calibrated:
            logger.warning(
                "%s has no usable calibration; running absolute passthrough — joint angles in "
                "radians and gripper opening in meters are forwarded unchanged. That is the right "
                "mode for a same-model Piper leader/follower pair, where both sides share one key "
                "space. Run `lerobot-calibrate --teleop.type=%s --teleop.id=%s` only if you need "
                "actions expressed as offsets from a neutral pose.",
                self,
                self.config.type,
                self.id,
            )

        logger.info("%s connected.", self)

    def _disconnect_arm(self) -> None:
        if self.arm is not None:
            self.arm.disconnect()
        self.arm = None
        self.gripper = None

    def _use_uncalibrated_passthrough(self) -> bool:
        """Whether to forward absolute values instead of offsets from the neutral pose.

        Keyed on :attr:`is_calibrated` alone, which already checks that *every*
        :data:`PIPER_CALIB_KEYS` entry is present with a valid range. Anything weaker
        would let :meth:`get_action` index ``self.calibration`` for a key that is not
        there. ``require_calibration`` decides whether :meth:`connect` offers to run the
        calibration flow; it must not decide how a *missing* calibration is read back.
        """
        return not self.is_calibrated

    @property
    def is_calibrated(self) -> bool:
        if not all(key in self.calibration for key in PIPER_CALIB_KEYS):
            return False
        for key in PIPER_CALIB_KEYS:
            cal = self.calibration[key]
            if cal.range_max <= cal.range_min:
                return False
        return True

    def calibrate(self) -> None:
        if self.calibration and self.is_calibrated:
            user_input = input(
                f"Press ENTER to use existing calibration file for id '{self.id}', "
                "or type 'c' and press ENTER to run a new calibration: "
            )
            if user_input.strip().lower() != "c":
                return

        logger.info("Running calibration for %s", self)
        input("Move piper-leader to your desired neutral/center pose, then press ENTER...")
        neutral = self._read_raw_action()
        print("Move all piper-leader joints through full range. Press ENTER to stop recording...")
        range_mins, range_maxes = self._record_ranges_of_motion()

        self.calibration = {}
        for key in PIPER_CALIB_KEYS:
            min_deg = range_mins[key]
            max_deg = range_maxes[key]
            if max_deg <= min_deg:
                raise ValueError(f"Invalid range for {key}: min={min_deg:.3f}, max={max_deg:.3f}")

            neutral_deg = min(max_deg, max(min_deg, neutral[key]))
            self.calibration[key] = MotorCalibration(
                id=PIPER_CALIB_IDS[key],
                drive_mode=0,
                homing_offset=self._to_calibration_units(neutral_deg),
                range_min=self._to_calibration_units(min_deg),
                range_max=self._to_calibration_units(max_deg),
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def _send_command_mode(self) -> None:
        self.arm.set_speed_percent(int(self.config.command_speed_ratio))
        # 'js' is the high-following (MIT) joint mode, the old `MotionCtrl_2(..., 0xAD)`.
        self.arm.set_motion_mode("js" if self.config.command_high_follow else "j")
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_command_mode_if_needed(self) -> None:
        interval_s = self.config.mode_refresh_interval_s
        if interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_mode_refresh_t >= interval_s:
            self._send_command_mode()

    def _wait_enable(self, timeout_s: float) -> bool:
        return wait_enable_piper(self.arm, timeout_s)

    def _send_gripper_ctrl(self, gripper_pos_m: float, enabled: bool) -> None:
        if enabled:
            self.gripper.move_gripper_m(gripper_pos_m, self.config.gripper_force_n)
        else:
            self.gripper.disable_gripper()

    def _set_gripper_enabled(self, enabled: bool) -> None:
        gripper_pos_m = 0.0
        try:
            gripper_status = self.gripper.get_gripper_status()
            if gripper_status is not None:
                gripper_pos_m = abs(float(gripper_status.msg.value))
        except Exception:
            logger.debug("Could not read current gripper opening before setting enable=%s.", enabled)
        self._send_gripper_ctrl(gripper_pos_m, enabled)

    def set_manual_control(self, enabled: bool) -> None:
        """Choose whether the host commands the arm or leaves it to the operator.

        ``enabled=True`` only stops the host from commanding the arm; it does not make
        the joints backdrivable. Releasing the joints is the teach button's job, and
        the arm holds its pose until the operator presses it.
        """
        if not self._is_connected:
            return
        if enabled and self._manual_control_enabled is not True:
            if self.config.sync_gripper:
                self._set_gripper_enabled(False)
            self._manual_control_enabled = True
            return
        if not enabled and self._manual_control_enabled is not False:
            if self.is_teach_mode_active():
                # Teaching mode rejects CAN motion commands, and forcing the arm out of it
                # would yank it away from the operator's hand. Leave the button in charge.
                logger.warning(
                    "%s is in teaching mode; skipping the switch to CAN command mode. "
                    "Press the teach button on the arm to hand control back.",
                    self,
                )
                return
            if self._last_ctrl_mode in PIPER_TEACH_CTRL_MODES:
                # The drag has ended, but the arm stays in ctrl_mode TEACH until the mode
                # frame below requests CAN control. Hold the measured pose as the position
                # target first, so the switch does not chase a stale target stored in firmware.
                if not seed_piper_position_target(self.arm):
                    logger.warning(
                        "%s: no joint feedback to seed the position target; staying out of CAN "
                        "command mode rather than risking a jump. Will retry on the next command.",
                        self,
                    )
                    return
                logger.info("%s: drag ended; requesting CAN control from teach mode.", self)
            self._send_command_mode()
            if not self._wait_enable(self.config.enable_timeout_s):
                logger.warning("Piper leader did not report enabled state before timeout.")
            if self.config.sync_gripper:
                self._set_gripper_enabled(True)
            self._manual_control_enabled = False

    # ------------------------------------------------------------------
    # Teach button (示教)
    # ------------------------------------------------------------------

    def _reset_teach_mode_state(self) -> None:
        self._teach_mode_active = False
        self._teach_mode_last_poll_t = 0.0
        self._teach_mode_pending = None
        self._teach_mode_pending_since = 0.0
        self._last_ctrl_mode = None

    def is_teach_mode_active(self) -> bool:
        """Report whether the operator has the arm in teaching mode (teach button engaged).

        Pressing the teach button switches the arm's ``ctrl_mode`` to TEACH and starts a drag
        (``teach_status`` 1). Pressing it again ends the drag (``teach_status`` 2) but leaves
        ``ctrl_mode`` at TEACH until the host requests CAN control, which
        :meth:`set_manual_control` does; so the drag state, not ``ctrl_mode``, is reported
        (see :func:`piper_teach_engaged`). The status frame is read at most every ``teach_mode_poll_interval_s`` and a new
        reading must hold for ``teach_mode_debounce_s`` before it is reported, so this
        is cheap enough to call once per control tick and does not chatter while the
        firmware switches modes.

        A missing status frame keeps the last reported value: dropping to "not
        teaching" mid-intervention would end the recording under the operator's hand.
        """
        if not self._is_connected:
            return False

        now = time.monotonic()
        if now - self._teach_mode_last_poll_t < self.config.teach_mode_poll_interval_s:
            return self._teach_mode_active
        self._teach_mode_last_poll_t = now

        reading = read_piper_mode_status(self.arm)
        if reading is None:
            logger.debug("%s: no arm status frame available; keeping last teach-mode state.", self)
            return self._teach_mode_active

        mode, teach_status = reading
        self._last_ctrl_mode = mode
        observed = piper_teach_engaged(mode, teach_status)
        if observed == self._teach_mode_active:
            self._teach_mode_pending = None
            return self._teach_mode_active

        if self._teach_mode_pending != observed:
            self._teach_mode_pending = observed
            self._teach_mode_pending_since = now
        elif now - self._teach_mode_pending_since >= self.config.teach_mode_debounce_s:
            self._teach_mode_active = observed
            self._teach_mode_pending = None
            logger.info(
                "%s teach button %s (ctrl_mode=0x%02X, teach_status=%s).",
                self,
                "pressed - operator has the arm" if observed else "released - control can return to the host",
                mode,
                teach_status,
            )
        return self._teach_mode_active

    # ------------------------------------------------------------------
    # Handover (从动臂角色 + 示教按钮)
    # ------------------------------------------------------------------
    #
    # The arm stays in the firmware's follower role for the whole session. Backdriving
    # comes from the teach button on the arm, which releases the joints and is the only
    # mechanism that leaves the arm broadcasting — the host has to keep reading it to
    # relay onto a follower robot on another CAN interface.
    #
    # The firmware's master-slave leader role (0x470 linkage_config=0xFA) is deliberately
    # never used: an arm in it stops broadcasting every feedback frame, the SDK then hands
    # back its last cached sample forever, and the follower silently freezes on the pose
    # the leader held at the switch. The role also persists across a power cycle, so it
    # strands the *next* session on a CAN bus that looks dead
    # (see `guard_piper_ctrl_mode_on_connect`, which recovers exactly that).
    #
    # A recording session walks two steps:
    #
    #   begin_alignment()      arm enabled and commanded -> it holds its pose rigidly
    #                          while the follower robot is ramped onto it
    #   release_to_operator()  host stops commanding     -> the operator presses the teach
    #                          button and backdrives; the host relays the pose it reads

    def _settle(self) -> None:
        """Give the firmware time to apply the follower-role frame before it is read back."""
        settle_s = self.config.follower_mode_settle_s
        if settle_s > 0:
            time.sleep(settle_s)

    @check_if_not_connected
    def begin_alignment(self) -> None:
        """Put the arm in the enabled follower role so it holds its current pose.

        This is the state the follower robot is aligned *to*: commanding the leader keeps
        it rigid, so the alignment target cannot drift away under the ramp.
        """
        logger.info("%s entering follower role for alignment.", self)
        self.arm.set_follower_mode()
        self._settle()
        # `is_teach_mode_active()` caches its reading for `teach_mode_poll_interval_s` and
        # debounces changes. Without this reset `set_manual_control(False)` below can see a
        # stale "teaching" state and skip the switch to CAN command mode entirely.
        self._reset_teach_mode_state()
        self.set_manual_control(False)

    @check_if_not_connected
    def release_to_operator(self) -> None:
        """Stop commanding the arm so the operator can take it.

        This does not make the joints backdrivable — that is the teach button's job, and
        the arm holds its pose until it is pressed. The arm stays in the follower role
        throughout, which is what keeps :meth:`get_action` readable.
        """
        # Releasing the gripper before the operator takes over, so the firmware is not
        # holding a stale grip target while the arm is pushed around.
        self.set_manual_control(True)
        self._reset_teach_mode_state()
        logger.info(
            "%s is released to the operator. Press the teach button on the arm (it lights "
            "up green) to backdrive it; press it again to hand control back.",
            self,
        )

    def configure(self) -> None:
        self.set_manual_control(self.config.manual_control)

    def _is_fresh(self, message: Any) -> bool:
        """Whether *message* is a live sample rather than the SDK's last cached one.

        Every pyAgxArm read hands back the most recent frame it ever parsed, with no
        indication that the arm has since gone quiet. Without an age check a leader that
        stops broadcasting — entering the CAN leader role does exactly that — reads as a
        perfectly valid, perfectly constant pose, and the follower simply stops moving.
        """
        max_age_s = self.config.max_feedback_age_s
        stamp = float(getattr(message, "timestamp", 0.0) or 0.0)
        if stamp <= 0.0:
            return False
        if max_age_s <= 0:
            return True
        # SDK timestamps are wall-clock seconds, on the same epoch as `time.time()`.
        return (time.time() - stamp) <= max_age_s

    def _read_joint_from_ctrl(self) -> dict[str, float] | None:
        """Read the joint targets the leader broadcasts to its follower, in radians."""
        leader_angles = self.arm.get_joint_angles()
        if leader_angles is None or not self._is_fresh(leader_angles):
            return None
        return {
            f"{joint_name}.pos": float(leader_angles.msg[index])
            for index, joint_name in enumerate(PIPER_JOINT_NAMES)
        }

    def _read_joint_from_feedback(self) -> dict[str, float] | None:
        """Read the leader's measured joint pose, in radians."""
        joint_angles = self.arm.get_joint_angles()
        if joint_angles is None or not self._is_fresh(joint_angles):
            return None
        return {
            f"{joint_name}.pos": float(joint_angles.msg[index])
            for index, joint_name in enumerate(PIPER_JOINT_NAMES)
        }

    def _read_gripper_from_ctrl(self) -> float | None:
        gripper_ctrl = self.gripper.get_gripper_ctrl_states()
        if gripper_ctrl is None or not self._is_fresh(gripper_ctrl):
            return None
        return abs(float(gripper_ctrl.msg.value))

    def _read_gripper_from_feedback(self) -> float | None:
        gripper_status = self.gripper.get_gripper_status()
        if gripper_status is None or not self._is_fresh(gripper_status):
            return None
        return abs(float(gripper_status.msg.value))

    def _joint_feedback_timestamp(self) -> float:
        joint_angles = self.arm.get_joint_angles()
        if joint_angles is None:
            return 0.0
        return float(getattr(joint_angles, "timestamp", 0.0) or 0.0)

    def _gripper_feedback_timestamp(self) -> float:
        gripper_status = self.gripper.get_gripper_status()
        if gripper_status is None:
            return 0.0
        return float(getattr(gripper_status, "timestamp", 0.0) or 0.0)

    def _wait_for_fresh_feedback_if_needed(self, prefer_feedback: bool) -> None:
        if not prefer_feedback:
            return

        timeout_s = self.config.manual_control_fresh_feedback_wait_timeout_s
        if timeout_s <= 0:
            return

        poll_s = self.config.manual_control_fresh_feedback_poll_s
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            joint_ts = self._joint_feedback_timestamp()
            has_fresh_joint = joint_ts > self._last_feedback_joint_timestamp
            if not has_fresh_joint:
                time.sleep(poll_s)
                continue
            # In manual-control teleop, blocking the whole control loop on fresh gripper feedback
            # adds visible latency to the arm, especially during intervention. Joints still wait
            # for a fresh sample, but gripper uses the latest available reading without stalling.
            return

    def _read_raw_action(self) -> RobotAction:
        used_feedback_for_joints = False
        action: dict[str, float] | None = None

        # Whenever the operator physically drives the arm — the host is not commanding it,
        # or the teach button is held down — `GetArmJointCtrl` reflects the last transmitted
        # control target rather than the live pose, so it lags or jumps behind the hand.
        # Read the joint feedback directly so teleop follows the real arm.
        prefer_feedback = self._manual_control_enabled is True or self.is_teach_mode_active()
        if prefer_feedback:
            self._wait_for_fresh_feedback_if_needed(prefer_feedback)
            action = self._read_joint_from_feedback()
            used_feedback_for_joints = action is not None
            self._last_feedback_joint_timestamp = self._joint_feedback_timestamp()

        if action is None and self.config.prefer_ctrl_messages:
            action = self._read_joint_from_ctrl()

        if action is None and self.config.fallback_to_feedback and not used_feedback_for_joints:
            action = self._read_joint_from_feedback()
            used_feedback_for_joints = action is not None

        if action is None:
            # A synthesised zero pose must never reach the follower: `send_action` forwards
            # it verbatim, so every joint would be commanded to home at once. Hold the last
            # pose the leader actually reported instead, and say why the arm went quiet.
            self._warn_stale_feedback()
            if self._last_raw_action is None:
                raise RuntimeError(
                    f"{self} has never reported a joint pose, so there is no action to send. The "
                    "arm is silent on CAN. Check power and wiring, and that it is not stuck in "
                    "the firmware's master-slave leader role, which suppresses all feedback."
                )
            action = {key: self._last_raw_action[key] for key in PIPER_JOINT_ACTION_KEYS}

        use_ctrl_for_gripper = (
            self.config.prefer_ctrl_messages and not prefer_feedback and not used_feedback_for_joints
        )
        gripper_pos = self._read_gripper_from_ctrl() if use_ctrl_for_gripper else None
        if gripper_pos is None and self.config.fallback_to_feedback:
            gripper_pos = self._read_gripper_from_feedback()
            self._last_feedback_gripper_timestamp = self._gripper_feedback_timestamp()
        if gripper_pos is None:
            # Same reasoning as the joints: 0.0 is "fully closed", not "unknown".
            last = self._last_raw_action
            gripper_pos = 0.0 if last is None else last["gripper.pos"]
        action["gripper.pos"] = gripper_pos

        self._last_raw_action = dict(action)
        return action

    def _warn_stale_feedback(self) -> None:
        """Warn that the leader has gone quiet, at most once every few seconds.

        Throttled because this is read every control tick: an unthrottled warning would
        bury the log it is meant to make visible.
        """
        now = time.monotonic()
        if now - self._last_stale_warning_t < _STALE_WARNING_INTERVAL_S:
            return
        self._last_stale_warning_t = now
        logger.warning(
            "%s has sent no joint feedback for over %.2fs; holding the last pose it reported. "
            "The follower will not move until the arm broadcasts again.",
            self,
            self.config.max_feedback_age_s,
        )

    def _to_calibration_units(self, angle_deg: float) -> int:
        return int(round(angle_deg * self.config.calibration_scale))

    def _from_calibration_units(self, value: int) -> float:
        return float(value) / float(self.config.calibration_scale)

    def _calibrated_to_offset(self, key: str, raw_deg: float) -> float:
        cal = self.calibration[key]
        min_deg = self._from_calibration_units(cal.range_min)
        max_deg = self._from_calibration_units(cal.range_max)
        home_deg = self._from_calibration_units(cal.homing_offset)
        bounded = min(max_deg, max(min_deg, raw_deg))
        centered = bounded - home_deg
        return -centered if cal.drive_mode else centered

    def _offset_to_calibrated(self, key: str, offset_deg: float) -> float:
        cal = self.calibration[key]
        min_deg = self._from_calibration_units(cal.range_min)
        max_deg = self._from_calibration_units(cal.range_max)
        home_deg = self._from_calibration_units(cal.homing_offset)
        centered = -offset_deg if cal.drive_mode else offset_deg
        target = home_deg + centered
        return min(max_deg, max(min_deg, target))

    def _record_ranges_of_motion(self) -> tuple[dict[str, float], dict[str, float]]:
        current = self._read_raw_action()
        mins = current.copy()
        maxes = current.copy()

        while True:
            current = self._read_raw_action()
            mins = {key: min(mins[key], current[key]) for key in PIPER_CALIB_KEYS}
            maxes = {key: max(maxes[key], current[key]) for key in PIPER_CALIB_KEYS}

            print("\n-----------------------------")
            print("JOINT       |    MIN |    POS |    MAX")
            for key in PIPER_CALIB_KEYS:
                print(f"{key:<11} | {mins[key]:>6.2f} | {current[key]:>6.2f} | {maxes[key]:>6.2f}")

            if enter_pressed():
                break
            move_cursor_up(len(PIPER_CALIB_KEYS) + 3)

        return mins, maxes

    @check_if_not_connected
    def get_raw_action(self) -> RobotAction:
        """Return absolute raw joints in radians and gripper position in meters.

        After calibration, `get_action()` returns offsets relative to the neutral pose,
        which suits leader-to-follower zero alignment. Online-RL interventions instead
        need the same absolute action space as the stage-one dataset to avoid systematic
        bias in the BC term. Callers such as PiperLeaderIntervention use this method.
        """
        action: RobotAction = dict(self._read_raw_action())
        if not self.config.sync_gripper:
            action["gripper.pos"] = 0.0
        return self._to_public_action(action)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        raw_action = self._read_raw_action()
        if self._use_uncalibrated_passthrough():
            action: RobotAction = dict(raw_action)
            if not self.config.sync_gripper:
                action["gripper.pos"] = 0.0
            return self._to_public_action(action)

        action: RobotAction = {
            key: self._calibrated_to_offset(key, raw_action[key]) for key in PIPER_CALIB_KEYS
        }
        if not self.config.sync_gripper:
            action["gripper.pos"] = 0.0
        return self._to_public_action(action)

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if self.is_teach_mode_active():
            # The arm is being taught by hand and ignores CAN motion commands anyway.
            # Driving it here would only fight the operator once they release the button.
            logger.debug("%s is in teaching mode; ignoring send_feedback.", self)
            return

        self.set_manual_control(False)
        self._refresh_command_mode_if_needed()

        joint_keys = PIPER_JOINT_ACTION_KEYS
        has_all_joints = all(key in feedback for key in joint_keys)
        if has_all_joints:
            if self._use_uncalibrated_passthrough():
                joint_targets = [feedback[key] for key in joint_keys]
            else:
                joint_targets = [self._offset_to_calibrated(key, feedback[key]) for key in joint_keys]
            joint_commands = [float(value) for value in joint_targets]
            if self.config.command_high_follow:
                self.arm.move_js(joint_commands)
            else:
                self.arm.move_j(joint_commands)

        if self.config.sync_gripper and self._gripper_key in feedback:
            # Calibration is keyed on the internal name; the caller uses the public one.
            if self._use_uncalibrated_passthrough():
                gripper_target = feedback[self._gripper_key]
            else:
                gripper_target = self._offset_to_calibrated("gripper.pos", feedback[self._gripper_key])
            # The gripper target is already in meters, the unit `move_gripper_m` expects.
            # The old code pushed it through the joint radian->protocol conversion, which
            # made the commanded opening two orders of magnitude off from the follower's.
            self._send_gripper_ctrl(abs(float(gripper_target)), enabled=True)

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self.config.disable_on_disconnect:
                self.arm.disable()
        finally:
            self._disconnect_arm()
            self._is_connected = False
            self._manual_control_enabled = None
            self._reset_teach_mode_state()
            self._last_raw_action = None
            logger.info("%s disconnected.", self)


class PiperXLeader(PiperLeader):
    config_class = PiperXLeaderConfig
    name = "piperx_leader"
