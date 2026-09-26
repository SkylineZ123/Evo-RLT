from __future__ import annotations

from types import SimpleNamespace

import pytest

from evo_rlt.adapters.lerobot.hardware.piper.motors_bus import PiperMotorsBus, PiperMotorsBusConfig


class _FakeArm:
    """Minimal stand-in for a pyAgxArm Piper driver."""

    def __init__(self):
        self.enabled = False
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.speed_percent = None
        self.joint_commands: list[list[float]] = []

    def connect(self):
        self.connected = True
        self.connect_calls += 1

    def disconnect(self, join_timeout: float = 1.0):
        self.connected = False
        self.disconnect_calls += 1

    def enable(self):
        was_enabled = self.enabled
        self.enabled = True
        return was_enabled

    def disable(self):
        was_disabled = not self.enabled
        self.enabled = False
        return was_disabled

    def get_joints_enable_status_list(self):
        return [self.enabled] * 6

    def get_joint_angles(self):
        return SimpleNamespace(msg=[0.0] * 6, timestamp=1.0, hz=100.0)

    def set_speed_percent(self, percent: int):
        self.speed_percent = percent

    def move_js(self, joints):
        self.joint_commands.append(list(joints))


class _FakeGripper:
    def __init__(self):
        self.value = 0.0
        self.disable_calls = 0

    def move_gripper_m(self, value: float = 0.0, force: float = 1.0):
        self.value = value

    def disable_gripper(self):
        self.disable_calls += 1
        return True

    def get_gripper_status(self):
        return SimpleNamespace(msg=SimpleNamespace(value=self.value, force=1.0), timestamp=1.0, hz=100.0)


@pytest.fixture
def fake_arm(monkeypatch):
    arm = _FakeArm()
    gripper = _FakeGripper()

    def _make_piper_arm(**_kwargs):
        # `make_piper_arm` is the seam: it builds AND connects the driver.
        arm.connect()
        return arm, gripper, "default"

    monkeypatch.setattr("evo_rlt.adapters.lerobot.hardware.piper.motors_bus.make_piper_arm", _make_piper_arm)
    monkeypatch.setattr("evo_rlt.adapters.lerobot.hardware.piper.motors_bus.time.sleep", lambda _seconds: None)
    return arm, gripper


def _bus() -> PiperMotorsBus:
    return PiperMotorsBus(
        PiperMotorsBusConfig(
            can_name="can-test",
            motors={f"joint_{i}": (i, "agilex_piper") for i in range(1, 8)},
        )
    )


def test_piper_bus_opens_can_only_during_connect_and_closes_on_disable(fake_arm):
    arm, _gripper = fake_arm
    bus = _bus()

    assert arm.connect_calls == 0
    assert bus.arm is None

    assert bus.connect(enable=True) is True
    assert bus.is_connected
    assert arm.connect_calls == 1

    assert bus.connect(enable=False) is True
    assert not bus.is_connected
    assert arm.disconnect_calls == 1
    assert not arm.connected
    assert bus.arm is None


def test_disconnecting_a_bus_that_never_connected_is_a_no_op(fake_arm):
    arm, _gripper = fake_arm
    bus = _bus()

    assert bus.connect(enable=False) is True
    assert arm.connect_calls == 0
    assert arm.disconnect_calls == 0


def test_write_sends_radians_and_meters_straight_through(fake_arm):
    arm, gripper = fake_arm
    bus = _bus()
    bus.connect(enable=True)

    bus.write([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05])

    assert arm.joint_commands[-1] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert gripper.value == pytest.approx(0.05)


def test_read_maps_physical_indices_onto_motor_names(fake_arm):
    _arm, gripper = fake_arm
    bus = _bus()
    bus.connect(enable=True)
    gripper.value = 0.042

    state = bus.read()

    assert set(state) == {f"joint_{i}" for i in range(1, 8)}
    assert state["joint_7"] == pytest.approx(0.042)


def test_read_reuses_the_last_state_when_a_frame_is_missing(fake_arm, monkeypatch):
    arm, gripper = fake_arm
    bus = _bus()
    bus.connect(enable=True)
    gripper.value = 0.03
    bus.read()

    # A dropped CAN frame must not report an all-zero pose: the follower would be
    # commanded toward zero on the next tick.
    monkeypatch.setattr(arm, "get_joint_angles", lambda: None)
    assert bus.read()["joint_7"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm_model", "piper_z"),
        ("firmware_version", "v999"),
        ("gripper_force_n", 5.0),
        ("speed_percent", 120),
    ],
)
def test_config_rejects_out_of_range_values(field, value):
    with pytest.raises(ValueError):
        PiperMotorsBusConfig(can_name="can-test", motors={}, **{field: value})
