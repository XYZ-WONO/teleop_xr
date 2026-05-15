"""
DEUX IK Controller

IKController를 상속하며 아래 동작만 변경:
  - deadman switch 제거 (grip 버튼 유지 불필요)
  - X버튼 long-press (>= 3초) → IK 토글 (활성↔비활성, 비활성 시 즉시 현재 위치 기준 활성화)
  - X버튼 short-press → IK 비활성화 (홈이동은 bridge 담당)

버튼/축 제어 분리:
  - grip, trigger → deux_hand_node가 Joy 토픽으로 직접 처리
  - right thumbstick Y (lift) → bridge가 Joy 토픽으로 USB relay LiftCommand 처리
  - 나머지 버튼 → bridge에서 Joy 토픽으로 처리
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


class DEUXIKController(IKController):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_x_pressed: bool = False
        self._press_start: dict[str, float] = {}
        self._long_press_fired: bool = False

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

    def _check_x_button(self, state: XRState) -> bool:
        """왼쪽 X버튼 (buttons[4])."""
        return self._check_button(state, XRHandedness.LEFT, 4)

    def _hold_duration(self, key: str, pressed: bool, now: float) -> float:
        """버튼 누름 지속 시간(초) 반환. 떼는 순간 타이머 초기화 후 0.0 반환."""
        if pressed:
            if key not in self._press_start:
                self._press_start[key] = now
            return now - self._press_start[key]
        else:
            self._press_start.pop(key, None)
            return 0.0

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

        now = time.monotonic()
        curr_xr_poses = self._get_device_poses(state)
        required_keys = self.robot.supported_frames
        has_all_poses = all(k in curr_xr_poses for k in required_keys)

        # X버튼 long-press 감지
        x_pressed = self._check_x_button(state)
        x_falling_edge = not x_pressed and self._prev_x_pressed
        self._prev_x_pressed = x_pressed
        x_hold = self._hold_duration("x", x_pressed, now)

        # falling edge 시 처리
        if x_falling_edge:
            if not self._long_press_fired and self.active:
                # short-press: IK 비활성화 (bridge가 홈 trajectory 담당)
                self.active = False
                if self.filter is not None:
                    self.filter.reset()
                logger.info("[DEUXIKController] X버튼 short-press → IK 비활성화 (bridge 홈이동 대기)")
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
            return q_current

        if not self.active:
            return q_current

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
            new_config_jax = self.solver.solve(
                target_L, target_R, target_Head, jnp.asarray(q_current)
            )
            new_config = np.array(new_config_jax)

            if self.filter is not None:
                self.filter.add_data(new_config)
                if self.filter.data_ready():
                    return self.filter.filtered_data

            return new_config

        return q_current
