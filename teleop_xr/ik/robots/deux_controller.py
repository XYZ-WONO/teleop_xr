"""
DEUX IK Controller

IKController를 상속하며 아래 동작만 변경:
  - deadman switch 제거 (grip 버튼 유지 불필요)
  - X버튼 long-press (>= 3초) → IK 토글 (활성↔비활성, 비활성 시 즉시 현재 위치 기준 활성화)
  - 오른쪽 thumbstick Y → lift 높이 제어 (controller 모드만)

홈이동은 bridge(teleop_xr_deux_bridge.py)에서 담당:
  - X버튼 short-press → bridge가 arm controller로 홈 trajectory 직접 퍼블리시
"""

import time
import numpy as np
import jax.numpy as jnp
import jaxlie
from loguru import logger

from teleop_xr.messages import XRState, XRDeviceRole, XRHandedness
from teleop_xr.ik.controller import IKController
from teleop_xr.ik.control_mode import ControlMode


LONG_PRESS_SEC = 3.0

LIFT_SPEED    = 0.1    # m/s
LIFT_MIN      = -0.45
LIFT_MAX      = 0.04
LIFT_DEADZONE = 0.3


class DEUXIKController(IKController):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_x_pressed: bool = False
        self._lift_pos: float = 0.0
        self._lift_last_time: float = time.monotonic()
        self._lift_joint_idx: int | None = None
        self._press_start: dict[str, float] = {}
        self._long_press_fired: bool = False

    def _read_axis(self, state: XRState, handedness: XRHandedness, index: int) -> float:
        for device in state.devices:
            if (
                device.role == XRDeviceRole.CONTROLLER
                and device.handedness == handedness
                and device.gamepad is not None
                and len(device.gamepad.axes) > index
            ):
                return device.gamepad.axes[index]
        return 0.0

    def _apply_lift(self, config: np.ndarray) -> np.ndarray:
        """lift_joint 위치를 _lift_pos로 오버라이드."""
        if self._lift_joint_idx is None:
            return config
        config = config.copy()
        config[self._lift_joint_idx] = self._lift_pos
        return config

    def _check_button(self, state: XRState, handedness: XRHandedness, index: int) -> bool:
        for device in state.devices:
            if (
                device.role == XRDeviceRole.CONTROLLER
                and device.handedness == handedness
                and device.gamepad is not None
                and len(device.gamepad.buttons) > index
            ):
                return device.gamepad.buttons[index].pressed
        return False

    # ── 버튼 래퍼 (왼쪽) ──────────────────────────────────────────────────
    def _check_trigger_left(self, state: XRState) -> bool:
        """왼쪽 Trigger (buttons[0])."""
        return self._check_button(state, XRHandedness.LEFT, 0)

    def _check_grip_left(self, state: XRState) -> bool:
        """왼쪽 Grip/Squeeze (buttons[1])."""
        return self._check_button(state, XRHandedness.LEFT, 1)

    def _check_thumbstick_click_left(self, state: XRState) -> bool:
        """왼쪽 Thumbstick 클릭 (buttons[3])."""
        return self._check_button(state, XRHandedness.LEFT, 3)

    def _check_x_button(self, state: XRState) -> bool:
        """왼쪽 X버튼 (buttons[4])."""
        return self._check_button(state, XRHandedness.LEFT, 4)

    def _check_y_button(self, state: XRState) -> bool:
        """왼쪽 Y버튼 (buttons[5])."""
        return self._check_button(state, XRHandedness.LEFT, 5)

    # ── 버튼 래퍼 (오른쪽) ─────────────────────────────────────────────────
    def _check_trigger_right(self, state: XRState) -> bool:
        """오른쪽 Trigger (buttons[0])."""
        return self._check_button(state, XRHandedness.RIGHT, 0)

    def _check_grip_right(self, state: XRState) -> bool:
        """오른쪽 Grip/Squeeze (buttons[1])."""
        return self._check_button(state, XRHandedness.RIGHT, 1)

    def _check_thumbstick_click_right(self, state: XRState) -> bool:
        """오른쪽 Thumbstick 클릭 (buttons[3])."""
        return self._check_button(state, XRHandedness.RIGHT, 3)

    def _check_a_button(self, state: XRState) -> bool:
        """오른쪽 A버튼 (buttons[4])."""
        return self._check_button(state, XRHandedness.RIGHT, 4)

    def _check_b_button(self, state: XRState) -> bool:
        """오른쪽 B버튼 (buttons[5])."""
        return self._check_button(state, XRHandedness.RIGHT, 5)

    # ── 버튼 press 지속시간 헬퍼 ──────────────────────────────────────────────
    def _hold_duration(self, key: str, pressed: bool, now: float) -> float:
        """버튼 누름 지속 시간(초) 반환. 떼는 순간 타이머 초기화 후 0.0 반환."""
        if pressed:
            if key not in self._press_start:
                self._press_start[key] = now
            return now - self._press_start[key]
        else:
            self._press_start.pop(key, None)
            return 0.0

    # ── thumbstick axes 래퍼 ───────────────────────────────────────────────
    def _read_thumbstick_left_x(self, state: XRState) -> float:
        """왼쪽 thumbstick X (axes[2])."""
        return self._read_axis(state, XRHandedness.LEFT, 2)

    def _read_thumbstick_left_y(self, state: XRState) -> float:
        """왼쪽 thumbstick Y (axes[3])."""
        return self._read_axis(state, XRHandedness.LEFT, 3)

    def _read_thumbstick_right_x(self, state: XRState) -> float:
        """오른쪽 thumbstick X (axes[2])."""
        return self._read_axis(state, XRHandedness.RIGHT, 2)

    def _read_thumbstick_right_y(self, state: XRState) -> float:
        """오른쪽 thumbstick Y (axes[3])."""
        return self._read_axis(state, XRHandedness.RIGHT, 3)

    def _check_deadman(self, state: XRState) -> bool:
        """deadman switch 없이 항상 활성."""
        return True

    def reset(self) -> None:
        super().reset()
        self._prev_x_pressed = False
        self._press_start.clear()
        self._long_press_fired = False

    def step(self, state: XRState, q_current: np.ndarray) -> np.ndarray:
        if self._mode != ControlMode.TELEOP:
            return q_current

        # lift_joint 인덱스 초기화 (lazy)
        if self._lift_joint_idx is None:
            names = self.robot.actuated_joint_names
            if "lift_joint" in names:
                self._lift_joint_idx = names.index("lift_joint")

        is_controller_mode = getattr(self.robot, "mode", None) == "controller"

        # controller 모드: 오른쪽 thumbstick Y (axes[3]) → lift 위치 업데이트
        # tracking 모드: thumbstick 무시, lift는 IK가 제어
        now = time.monotonic()
        dt = now - self._lift_last_time
        self._lift_last_time = now
        if is_controller_mode:
            thumbstick_y = self._read_thumbstick_right_y(state)
            if abs(thumbstick_y) > LIFT_DEADZONE:
                self._lift_pos = max(LIFT_MIN, min(LIFT_MAX, self._lift_pos + (-thumbstick_y * LIFT_SPEED * dt)))

        curr_xr_poses = self._get_device_poses(state)
        required_keys = self.robot.supported_frames
        has_all_poses = all(k in curr_xr_poses for k in required_keys)

        # X버튼 long-press 감지
        x_pressed = self._check_x_button(state)
        x_falling_edge = not x_pressed and self._prev_x_pressed
        self._prev_x_pressed = x_pressed
        x_hold = self._hold_duration("x", x_pressed, now)

        # falling edge 시 long-press 플래그 초기화
        if x_falling_edge:
            self._long_press_fired = False

        # long-press (3초 이상, 누르는 중): IK 토글 — 한 번만 발동
        if x_hold >= LONG_PRESS_SEC and not self._long_press_fired:
            self._long_press_fired = True
            if self.active:
                self.active = False
                if self.filter is not None:
                    self.filter.reset()
                logger.info("[DEUXIKController] X버튼 long-press → IK 비활성화")
            elif has_all_poses:
                fk_poses = self.robot.forward_kinematics(jnp.asarray(q_current))
                self.snapshot_xr = curr_xr_poses
                self.snapshot_robot = {k: fk_poses[k] for k in required_keys}
                self.active = True
                if self.filter is not None:
                    self.filter.reset()
                logger.info("[DEUXIKController] X버튼 long-press → IK 즉시 활성화")
            return self._apply_lift(q_current) if is_controller_mode else q_current

        if not self.active:
            return self._apply_lift(q_current) if is_controller_mode else q_current

        # IK 활성: 포즈 추적
        target_L: jaxlie.SE3 | None = None
        target_R: jaxlie.SE3 | None = None
        target_Head: jaxlie.SE3 | None = None

        if "left" in required_keys:
            target_L = self.compute_teleop_transform(
                curr_xr_poses["left"],
                self.snapshot_xr["left"],
                self.snapshot_robot["left"],
            )
        if "right" in required_keys:
            target_R = self.compute_teleop_transform(
                curr_xr_poses["right"],
                self.snapshot_xr["right"],
                self.snapshot_robot["right"],
            )
        if "head" in required_keys:
            target_Head = self.compute_teleop_transform(
                curr_xr_poses["head"],
                self.snapshot_xr["head"],
                self.snapshot_robot["head"],
            )

        if self.solver is not None:
            # controller 모드: lift=0 고정 후 IK (thumbstick lift와 분리)
            # tracking 모드: q_current 그대로 전달 (lift도 IK가 제어)
            q_for_ik = q_current.copy()
            if is_controller_mode and self._lift_joint_idx is not None:
                q_for_ik[self._lift_joint_idx] = 0.0
            new_config_jax = self.solver.solve(
                target_L, target_R, target_Head, jnp.asarray(q_for_ik)
            )
            new_config = np.array(new_config_jax)

            if self.filter is not None:
                self.filter.add_data(new_config)
                if self.filter.data_ready():
                    result = self.filter.filtered_data
                    return self._apply_lift(result) if is_controller_mode else result

            return self._apply_lift(new_config) if is_controller_mode else new_config

        return self._apply_lift(q_current) if is_controller_mode else q_current
