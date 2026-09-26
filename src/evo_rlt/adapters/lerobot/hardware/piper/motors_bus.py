import logging
import math
import time
from dataclasses import dataclass, field

from evo_rlt.adapters.lerobot.hardware.piper.agx_arm import (
    make_piper_arm,
    validate_piper_arm_model,
    validate_piper_firmware_version,
    validate_piper_gripper_force,
    wait_enable_piper,
)

logger = logging.getLogger(__name__)


def _to_float(x) -> float:
    # Extract scalar tensors with item(); otherwise cast directly.
    if "torch" in str(type(x)) and hasattr(x, "item"):
        return float(x.item())
    return float(x)


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]
    # pyAgxArm connection options
    arm_model: str = "piper"  # piper / piper_h / piper_l / piper_x
    firmware_version: str = "auto"  # auto / default / v183 / v188 / v189
    can_interface: str = "socketcan"
    can_bitrate: int = 1_000_000
    log_level: str = "WARNING"
    # Speed percentage applied to every joint command (was hardcoded to 60 with the old SDK).
    speed_percent: int = 60
    # Software joint-limit clamping inside the SDK. Off by default, matching both the SDK's
    # own default and the previous behaviour; turning it on clamps to the `arm_model` preset.
    enforce_joint_limits: bool = False
    # Reset and safety parameters
    reset_hz: float = 100.0  # Command frequency during a smooth reset
    reset_duration_s: float = 4.0  # Desired duration of a smooth reset
    max_joint_step_rad: float = 0.01  # Maximum joint increment per step in MIT mode
    open_gripper_on_init: bool = True  # Open the gripper during the initial reset
    gripper_open_range: float = 0.07  # Gripper opening in meters (0 to 0.08)
    gripper_force_n: float = 1.0  # Gripper clamping force in newtons (0 to 3)
    home_position: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __post_init__(self):
        validate_piper_arm_model(self.arm_model)
        validate_piper_firmware_version(self.firmware_version)
        validate_piper_gripper_force(self.gripper_force_n)
        if not (0 <= self.speed_percent <= 100):
            raise ValueError("`speed_percent` must be between 0 and 100.")


class PiperMotorsBus:
    """LeRobot wrapper around the pyAgxArm Piper driver."""

    def __init__(self, config: PiperMotorsBusConfig):
        # Construction only builds configuration. Opening CAN belongs to connect(), which
        # keeps config introspection and failed dataset setup side-effect free.
        self.config = config
        self.arm = None
        self.gripper = None
        self.firmware_profile: str | None = None
        self.motors = config.motors
        # Use zero as the dataset-recording home pose by default.
        self.init_joint_position = list(config.home_position)  # [6 joints + 1 gripper]
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        # Reset and safety parameters
        self.reset_hz = config.reset_hz
        self.reset_duration_s = config.reset_duration_s
        self.max_joint_step_rad = config.max_joint_step_rad
        self.open_gripper_on_init = config.open_gripper_on_init
        self.gripper_open_range = config.gripper_open_range
        self.gripper_force_n = config.gripper_force_n
        # The SDK does not expose reliable enabled/reset state. Track it here for
        # Robot.is_connected and Robot.is_calibrated (DualPiper reads these directly).
        self._is_connected = False
        self._is_calibrated = False
        # pyAgxArm read APIs return None until the matching CAN frame has been seen. Keep
        # the last good sample so a fresh connection never reports an all-zero pose, which
        # would otherwise yank the arm toward zero on the first command.
        self._last_state: list[float] | None = None

    @property
    def is_connected(self) -> bool:
        """Whether the arm has been enabled successfully.

        The SDK has no reliable queryable connection state, so this flag is maintained locally.
        """
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Whether the smooth reset (`apply_calibration`) has completed."""
        return self._is_calibrated

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def _open(self) -> None:
        self.arm, self.gripper, self.firmware_profile = make_piper_arm(
            channel=self.config.can_name,
            arm_model=self.config.arm_model,
            firmware_version=self.config.firmware_version,
            interface=self.config.can_interface,
            bitrate=self.config.can_bitrate,
            log_level=self.config.log_level,
            enforce_joint_limits=self.config.enforce_joint_limits,
        )
        self.arm.set_speed_percent(int(self.config.speed_percent))

    def _close(self) -> None:
        if self.arm is not None:
            self.arm.disconnect()
        self.arm = None
        self.gripper = None
        self._last_state = None

    def connect(self, enable: bool) -> bool:
        """Enable or disable the arm and wait up to five seconds for the requested state."""
        if not enable and self.arm is None:
            self._is_connected = False
            self._is_calibrated = False
            return True

        if self.arm is None:
            self._open()

        timeout_s = 5.0
        if enable:
            ok = wait_enable_piper(self.arm, timeout_s=timeout_s, retry_interval_s=0.5)
            if ok:
                # Close the gripper to a known state, mirroring the old enable path.
                self.gripper.move_gripper_m(0.0, self.gripper_force_n)
        else:
            ok = self._wait_disable(timeout_s=timeout_s)

        if not ok:
            logger.warning(
                "[%s] Timed out waiting for the arm to report %s.",
                self.config.can_name,
                "enabled" if enable else "disabled",
            )

        # Only a successful enable request counts as connected; enable=False is the shutdown path.
        self._is_connected = bool(ok) and enable
        if not self._is_connected:
            self._is_calibrated = False
        if not enable:
            self._close()
        logger.info("[%s] connect(enable=%s) -> %s", self.config.can_name, enable, ok)
        return ok

    def _wait_disable(self, timeout_s: float, retry_interval_s: float = 0.5) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            disabled = not any(self.arm.get_joints_enable_status_list())
            self.arm.disable()
            self.gripper.disable_gripper()
            if disabled:
                return True
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return False
            time.sleep(min(retry_interval_s, remaining_s))

    def set_calibration(self):
        return

    def revert_calibration(self):
        return

    def apply_calibration(self):
        """Move smoothly to the configured home position.

        Sending a distant target directly in MIT mode can make the arm rush from an
        unknown pose toward home. Read the current pose and use small smoothstep
        increments to reduce shock. Optionally open the gripper during the reset.
        """
        target = list(self.init_joint_position)
        if self.open_gripper_on_init:
            target[6] = self.gripper_open_range
        self.move_to_joint_smoothly(target)
        self._is_calibrated = True

    def _read_joint_list(self) -> list:
        """Read six joints and the gripper in write order, in radians and meters."""
        joint_angles = self.arm.get_joint_angles()
        gripper_status = self.gripper.get_gripper_status()

        if joint_angles is None or gripper_status is None:
            if self._last_state is not None:
                logger.debug(
                    "[%s] Incomplete CAN feedback; reusing the last known joint state.",
                    self.config.can_name,
                )
                return list(self._last_state)
            # Nothing has ever been received: zeros are the only answer available, and the
            # callers (smooth reset, observation) all tolerate a single stale frame.
            logger.warning(
                "[%s] No joint/gripper feedback received yet; reporting zeros.", self.config.can_name
            )
            return [0.0] * 7

        state = [float(value) for value in joint_angles.msg] + [float(gripper_status.msg.value)]
        self._last_state = state
        return list(state)

    def move_to_joint_smoothly(self, target_joint: list, duration_s=None, hz=None, max_joint_step_rad=None):
        """Interpolate smoothly from the current joint pose to a target pose.

        The step count satisfies both the requested duration and the maximum joint
        increment. Smoothstep easing further reduces start/stop vibration.
        """
        hz = self.reset_hz if hz is None else hz
        duration_s = self.reset_duration_s if duration_s is None else duration_s
        max_step = self.max_joint_step_rad if max_joint_step_rad is None else max_joint_step_rad

        # Joint feedback may be stale immediately after enabling; wait briefly before reading.
        time.sleep(0.1)
        start = self._read_joint_list()
        target = [_to_float(x) for x in target_joint]

        # Estimate velocity-limited steps from the six arm joints; exclude the gripper.
        max_delta = max((abs(t - s) for t, s in zip(target[:6], start[:6], strict=True)), default=0.0)
        steps_by_time = max(int(round(duration_s * hz)), 1)
        steps_by_vel = max(int(math.ceil(max_delta / max_step)), 1) if max_step > 0 else 1
        steps = max(steps_by_time, steps_by_vel)

        dt = 1.0 / hz
        for i in range(1, steps + 1):
            alpha = i / steps
            s = alpha * alpha * (3 - 2 * alpha)  # Smoothstep easing
            interp = [st + (tg - st) * s for st, tg in zip(start, target, strict=True)]
            self.write(interp)
            time.sleep(dt)

    def open_gripper(self, gripper_range=None):
        """Open the gripper; gripper_range is in meters from 0 to 0.08."""
        r = self.gripper_open_range if gripper_range is None else gripper_range
        self.gripper.move_gripper_m(abs(_to_float(r)), self.gripper_force_n)

    def close_gripper(self):
        """Close the gripper."""
        self.gripper.move_gripper_m(0.0, self.gripper_force_n)

    def write(self, target_joint: list):
        """
        Joint control
        - target_joint[0:6]: joint angles in radians, ordered joint_1 to joint_6
        - target_joint[6]: gripper opening in meters, 0 to 0.08

        `move_js` is the high-following (MIT) variant, matching the old
        `MotionCtrl_2(0x01, 0x01, speed, 0xAD)` + `JointCtrl(...)` pair.
        """
        joints = [_to_float(value) for value in target_joint[:6]]
        gripper_m = abs(_to_float(target_joint[6]))

        self.arm.move_js(joints)
        self.gripper.move_gripper_m(gripper_m, self.gripper_force_n)

    def read(self) -> dict:
        """Read the arm state, in radians for joints and meters for the gripper.

        Returns a dictionary with keys from self.motors configuration.
        Maps physical joint indices (1-7) to configured motor names.
        """
        physical_values = self._read_joint_list()

        # Build result dict based on motors configuration
        result = {}
        for motor_name, (physical_idx, _model) in self.motors.items():
            # physical_idx is 1-based, convert to 0-based for list access
            result[motor_name] = physical_values[physical_idx - 1]

        return result

    def safe_disconnect(self):
        """Move smoothly to the safe shutdown pose before disabling the arm."""
        if self.arm is None:
            # Nothing was ever opened (e.g. cleanup after a failed connect); there is no
            # pose to move to and no driver to command.
            logger.debug("[%s] safe_disconnect on a closed bus; nothing to do.", self.config.can_name)
            return
        self.move_to_joint_smoothly(self.safe_disable_position)
