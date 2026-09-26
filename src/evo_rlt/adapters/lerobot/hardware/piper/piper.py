import contextlib
import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from evo_rlt.adapters.lerobot.hardware.piper.motors_bus import PiperMotorsBus, PiperMotorsBusConfig
from evo_rlt.adapters.lerobot.hardware.piper.config_piper import PiperConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


def get_motor_names(arm: dict[str, Any]) -> list[str]:
    return [motor for arm_key, bus in arm.items() for motor in bus.motors]


class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)

        self.config = config
        self.bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=self.config.can_port,
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "joint_7": (7, "agilex_piper"),
                },
                arm_model=self.config.arm_model,
                firmware_version=self.config.firmware_version,
                can_interface=self.config.can_interface,
                can_bitrate=self.config.can_bitrate,
                log_level=self.config.log_level,
                speed_percent=self.config.speed_percent,
                enforce_joint_limits=self.config.enforce_joint_limits,
                reset_hz=self.config.reset_hz,
                reset_duration_s=self.config.reset_duration_s,
                max_joint_step_rad=self.config.max_joint_step_rad,
                open_gripper_on_init=self.config.open_gripper_on_init,
                gripper_open_range=self.config.gripper_open_range,
                gripper_force_n=self.config.gripper_force_n,
                home_position=list(self.config.home_position),
            )
        )
        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, 3),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        arm_dict = {"piper": self.bus}
        action_names = get_motor_names(arm_dict)
        state_names = get_motor_names(arm_dict)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor action description used for recording and replay."""
        arm_dict = {"piper": self.bus}
        motor_names = get_motor_names(arm_dict)
        return {f"{name}.pos": float for name in motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera image description used for recording and replay."""
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:

        return {**self._motors_ft, **self._cameras_ft}

    def configure(self, **kwargs):
        # No additional configuration is required; this implements the abstract hook.
        pass

    @property
    def is_connected(self) -> bool:
        """Whether the robot and all configured cameras are connected.

        This intentionally uses the locally maintained `_is_connected` flag. Connection,
        disconnection, calibration, and their precondition checks all update this state.
        """
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """Whether robot calibration has completed."""
        return self._is_calibrated

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self, calibrate=True) -> None:
        """Connect piper and cameras"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError("Piper is already connected. Do not run robot.connect() twice.")

        try:
            if not self.bus.connect(enable=True):
                raise ConnectionError(f"Could not enable Piper on {self.config.can_port}")
            print("piper follower connected")
            for name, camera in self.cameras.items():
                camera.connect()
                if not camera.is_connected:
                    raise ConnectionError(f"Camera {name} did not connect")
                print(f"camera {name} connected")
            self._is_connected = True
            print("All connected")
            if calibrate:
                self.calibrate()
        except Exception:
            for camera in self.cameras.values():
                if camera.is_connected:
                    with contextlib.suppress(Exception):
                        camera.disconnect()
            with contextlib.suppress(Exception):
                self.bus.connect(enable=False)
            self._is_connected = False
            raise

    def disconnect(self) -> None:
        """move to home position, disenable piper and cameras"""
        self.bus.safe_disconnect()
        print("piper disable after 5 seconds")
        time.sleep(5)
        self.bus.connect(enable=False)

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False

    def emergency_disconnect(self) -> None:
        """Disable immediately, without executing the normal safe-position move."""
        self.disconnect_without_homing()

    def disconnect_without_homing(self) -> None:
        """Disable at the current pose after an external controller handled homing."""
        self.bus.connect(enable=False)
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        self._is_connected = False

    def calibrate(self):
        """move piper to the home position"""
        if not self._is_connected:
            raise ConnectionError()

        self.bus.apply_calibration()
        self._is_calibrated = True

    def get_observation(self) -> dict:
        """Capture current joint positions and camera images"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")
        # Read joint state.
        state = self.bus.read()
        obs_dict = {f"{joint}.pos": float(val) for joint, val in state.items()}
        # Read camera images.
        for name, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[f"{name}"] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {name}: {dt_ms:.1f}ms")
        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Receive action dict from record() and send to motor"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")
        # t0 = time.perf_counter()
        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
        # if action["joint_7.pos"]<0.03:
        #     action["joint_7.pos"]=0.1
        target_joints = [action[f"{motor}.pos"] for motor in motor_order]
        self.bus.write(target_joints)
        # t1 = time.perf_counter()
        # print(f"Left arm: {(t1 - t0) * 1000:.2f}ms")
        return action
