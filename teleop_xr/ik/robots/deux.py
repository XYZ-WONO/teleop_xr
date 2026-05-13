# pyright: reportCallIssue=false
import sys
from pathlib import Path
from typing import Any, Literal

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr import ram
from teleop_xr.ik.robot import BaseRobot, Cost


# DEUX description package root (relative to this file's repo location)
_DEUX_DESCRIPTION_ROOT = (
    Path(__file__).resolve().parents[4] / "deux_description"
)
_DEUX_URDF_PATH = "urdf/deux/deux.urdf"

_SWERVE_JOINTS = frozenset({
    "front_left_steer_joint",
    "rear_left_steer_joint",
    "rear_right_steer_joint",
    "front_right_steer_joint",
    "front_left_drive_joint",
    "rear_left_drive_joint",
    "rear_right_drive_joint",
    "front_right_drive_joint",
})
_HAND_JOINTS = frozenset({
    "left_hand_thumb_joint_1", "left_hand_thumb_joint_2", "left_hand_thumb_joint_3",
    "left_hand_index_joint_1", "left_hand_index_joint_2",
    "left_hand_third_joint_1", "left_hand_third_joint_2",
    "right_hand_thumb_joint_1", "right_hand_thumb_joint_2", "right_hand_thumb_joint_3",
    "right_hand_index_joint_1", "right_hand_index_joint_2",
    "right_hand_third_joint_1", "right_hand_third_joint_2",
})


class DEUX(BaseRobot):
    """DEUX bimanual mobile robot IK base class.

    mode="controller": lift excluded from IK — Meta Quest3 + controller.
    mode="tracking":   lift included in IK  — hand tracking.
    """

    def __init__(
        self,
        mode: Literal["controller", "tracking"] = "controller",
        urdf_string: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.mode = mode
        super().__init__()
        urdf = self._load_urdf(urdf_string)
        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        self.L_ee = "left_arm_hand"
        self.R_ee = "right_arm_hand"
        self.lift_link = "upper_body"

        names = self.robot.links.names

        if self.L_ee not in names:
            raise ValueError(f"Link {self.L_ee} not found in URDF")
        if self.R_ee not in names:
            raise ValueError(f"Link {self.R_ee} not found in URDF")
        if self.lift_link not in names:
            raise ValueError(f"Link {self.lift_link} not found in URDF")

        self.L_ee_link_idx: int = names.index(self.L_ee)
        self.R_ee_link_idx: int = names.index(self.R_ee)
        self.lift_link_idx: int = names.index(self.lift_link)

    @override
    def _load_default_urdf(self) -> yourdfpy.URDF:
        if not _DEUX_DESCRIPTION_ROOT.exists():
            raise FileNotFoundError(
                f"deux_description not found at {_DEUX_DESCRIPTION_ROOT}. "
                "Ensure deux_description is cloned next to teleop_xr."
            )
        urdf_path = ram.get_resource(
            repo_root=_DEUX_DESCRIPTION_ROOT,
            path_inside_repo=_DEUX_URDF_PATH,
            resolve_packages=True,
        )
        self.urdf_path = str(urdf_path)
        self.mesh_path = str(_DEUX_DESCRIPTION_ROOT)
        return yourdfpy.URDF.load(
            self.urdf_path,
            load_meshes=False,
            load_collision_meshes=False,
        )

    @property
    @override
    def supported_frames(self) -> set[str]:
        if self.mode == "tracking":
            return {"left", "right", "head"}
        return {"left", "right"}

    @property
    @override
    def joint_var_cls(self) -> Any:
        return self.robot.joint_var_cls

    @property
    @override
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @override
    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        result: dict[str, jaxlie.SE3] = {
            "left": jaxlie.SE3(fk[self.L_ee_link_idx]),
            "right": jaxlie.SE3(fk[self.R_ee_link_idx]),
        }
        if self.mode == "tracking":
            result["head"] = jaxlie.SE3(fk[self.lift_link_idx])
        return result

    @override
    def get_default_config(self) -> jax.Array:
        defaults = {
            "arm_l_joint4": 1.57,
            "arm_r_joint4": 1.57,
        }
        return jnp.array(
            [defaults.get(n, 0.0) for n in self.actuated_joint_names]
        )

    @override
    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        costs = []
        JointVar = self.robot.joint_var_cls
        n_joints = len(self.actuated_joint_names)

        # controller: lift frozen (high weight), tracking: lift participates in IK
        lift_energy = 500.0 if self.mode == "controller" else 8.0

        arm_energy = {
            "lift_joint": lift_energy,
            "arm_l_joint1": 5.0, "arm_l_joint2": 5.0, "arm_l_joint3": 4.0,
            "arm_l_joint4": 4.0, "arm_l_joint5": 3.0, "arm_l_joint6": 3.0,
            "arm_l_joint7": 2.0,
            "arm_r_joint1": 5.0, "arm_r_joint2": 5.0, "arm_r_joint3": 4.0,
            "arm_r_joint4": 4.0, "arm_r_joint5": 3.0, "arm_r_joint6": 3.0,
            "arm_r_joint7": 2.0,
        }
        energy_weights = jnp.array([
            arm_energy.get(
                name,
                20.0 if name in _SWERVE_JOINTS else
                10.0 if name in _HAND_JOINTS else
                1.0,
            )
            for name in self.actuated_joint_names
        ])

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=energy_weights,
                )
            )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.005,
            )
        )

        if target_L is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_L,
                    jnp.array(self.L_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.R_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        if self.mode == "tracking" and target_Head is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_Head,
                    jnp.array(self.lift_link_idx, dtype=jnp.int32),
                    pos_weight=30.0,
                    ori_weight=0.0,
                )
            )

        lift_centering = 500.0 if self.mode == "controller" else 0.0
        centering_weights = jnp.array([
            20.0 if name in _SWERVE_JOINTS else
            lift_centering if name == "lift_joint" else
            10.0 if name in _HAND_JOINTS else
            0.0
            for name in self.actuated_joint_names
        ])
        costs.append(
            pk.costs.rest_cost(
                JointVar(0),
                rest_pose=jnp.zeros(n_joints),
                weight=centering_weights,
            )
        )

        costs.append(pk.costs.limit_cost(self.robot, JointVar(0), weight=50.0))

        costs.append(
            pk.costs.self_collision_cost(
                self.robot,
                self.robot_coll,
                JointVar(0),
                margin=0.05,
                weight=10.0,
            )
        )

        return costs


class DEUX_controller(DEUX):
    """Meta Quest3 + controller mode. Lift excluded from IK."""

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        super().__init__(mode="controller", urdf_string=urdf_string, **kwargs)


class DEUX_tracking(DEUX):
    """Hand tracking mode. Lift included in IK."""

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        super().__init__(mode="tracking", urdf_string=urdf_string, **kwargs)
